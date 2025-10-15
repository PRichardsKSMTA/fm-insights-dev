#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize.py
Reads a freight CSV and produces a JSON summary with per-entity aggregates, impacts,
and driver details.

Implements:
- Period selection anchored to week-ending Saturdays (Mon=0 ... Sat=5 Sun=6).
- Normalization rule:
    * If the two periods have different week counts, normalize ONLY the longer period to weekly averages.
    * If the two periods have the same week count (e.g., 4wk vs 4wk), keep them as-is (multi-week totals).
- 0.75% low-volume filter, using:
    * Equal lengths: compare totals vs totals for each period.
    * Different lengths: compare weekly averages vs weekly averages for each period.
  Exclude an entity only if it is below threshold in BOTH periods.
- Substitution: when an entity has 0 loads in a period, use comparison-period network averages
  for per-load metrics (Core OR and Profit/Load) to compute impacts/composite (prevents distortions).
- Driver details with no minimums:
    * For customer rows: lanes-by-customer, inbound-areas-by-customer, outbound-areas-by-customer
    * For lane rows: customers-by-lane
    * For inbound/outbound rows: customers-by-area
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

REQUIRED_COLS = [
    "DELIVERY_DT",
    "BILLTO_NAME",
    "ORIG_AREA",
    "DEST_AREA",
    "LOADED_OP_MILES",
    "TOTAL_REVENUE",
    "TOTAL_VARIABLE_COST",
    "TOTAL_OVERHEAD_COST",
]

L0 = 15            # stability constant
LOW_VOL_PCT = 0.0075  # 0.75%

# ---------------------- Helpers ----------------------

def safe_float(x, default=0.0) -> float:
    try:
        f = float(x)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default

def safe_int(x, default=0) -> int:
    try:
        f = float(x)
        if np.isnan(f):
            return default
        return int(round(f))
    except Exception:
        return default

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        b = float(b)
        if b == 0 or np.isnan(b):
            return default
        a = float(a)
        if np.isnan(a):
            return default
        return a / b
    except Exception:
        return default

def saturday_week_end(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    dow = dt.dt.dayofweek  # Mon=0 ... Sat=5 Sun=6
    return dt + pd.to_timedelta((5 - dow) % 7, unit="D")

def ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def pick_periods_by_weeks(weeks_sorted: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    n = len(weeks_sorted)
    if n < 2:
        return weeks_sorted, []
    if n >= 8:
        A = weeks_sorted[-4:]
        B = weeks_sorted[-8:-4]
    elif 5 <= n <= 7:
        A = weeks_sorted[-1:]
        k = min(4, n - 1)
        B = weeks_sorted[-(1 + k):-1]
    elif 3 <= n <= 4:
        A = weeks_sorted[-1:]
        B = weeks_sorted[:-1]
    else:  # n == 2
        A = weeks_sorted[-1:]
        B = weeks_sorted[-2:-1]
    return A, B

def format_label(sat_list: List[pd.Timestamp]) -> str:
    if not sat_list:
        return "N/A (0wk)"
    end = sat_list[-1]
    weeks = len(sat_list)
    return f"{end.strftime('%m/%d/%y')} ({weeks}wk)"

def compute_network_metrics(df: pd.DataFrame) -> Dict[str, float]:
    raw = len(df)
    tot_rev = safe_float(df["TOTAL_REVENUE"].sum())
    tot_var = safe_float(df["TOTAL_VARIABLE_COST"].sum())
    tot_oh  = safe_float(df["TOTAL_OVERHEAD_COST"].sum())
    miles   = safe_float(df["LOADED_OP_MILES"].sum())
    tot_cost = tot_var + tot_oh

    rpl = safe_div(tot_rev, raw)
    vpl = safe_div(tot_var, raw)
    ohl = safe_div(tot_oh,  raw)
    loh = safe_div(miles,   raw)
    core_or = safe_div(tot_cost, tot_rev, default=0.0)
    ppl = rpl - (vpl + ohl)

    return {
        "raw_loads": float(raw),
        "RPL": rpl,
        "VPL": vpl,
        "OHL": ohl,
        "LOH": loh,
        "Core_OR": core_or,
        "ProfitPerLoad": ppl,
    }

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize entity text fields
    for col in ["BILLTO_NAME", "ORIG_AREA", "DEST_AREA"]:
        df[col] = df[col].astype(str).fillna("").str.strip()
        df[col] = df[col].replace({"nan": "", "None": ""})
    # Parse dates and week-ending Saturday
    df["DELIVERY_DT"] = pd.to_datetime(df["DELIVERY_DT"], errors="coerce")
    df["WEEK_ENDING"] = saturday_week_end(df["DELIVERY_DT"])
    # Clean numerics
    for col in ["LOADED_OP_MILES", "TOTAL_REVENUE", "TOTAL_VARIABLE_COST", "TOTAL_OVERHEAD_COST"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Drop rows without valid week ending
    df = df.dropna(subset=["WEEK_ENDING"])
    return df

def effective_loads(raw_A: float, raw_B: float, weeks_A: int, weeks_B: int) -> Tuple[float, float]:
    """
    Normalization for impact math:
    - If weeks_A == weeks_B: use raw totals (e.g., 4wk vs 4wk → totals).
    - If weeks differ: normalize ONLY the longer period to weekly averages.
    """
    if weeks_A == weeks_B:
        return float(raw_A), float(raw_B)
    if weeks_A > weeks_B:
        return float(raw_A) / float(weeks_A), float(raw_B)
    else:  # weeks_B > weeks_A
        return float(raw_A), float(raw_B) / float(weeks_B)

def weekly_loads(raw: float, weeks: int) -> float:
    return float(raw) / float(max(weeks, 1))

def build_agg(df: pd.DataFrame, keys: List[str], suffix: str) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = pd.DataFrame({
        f"raw_loads_{suffix}": g.size(),
        f"total_rev_{suffix}": g["TOTAL_REVENUE"].sum(),
        f"total_var_{suffix}": g["TOTAL_VARIABLE_COST"].sum(),
        f"total_oh_{suffix}": g["TOTAL_OVERHEAD_COST"].sum(),
        f"miles_{suffix}": g["LOADED_OP_MILES"].sum(),
    })
    return out

def per_load_metrics(df_row, suffix: str, net: Dict[str, float]) -> Dict[str, float]:
    raw = safe_float(df_row.get(f"raw_loads_{suffix}", 0.0))
    total_rev = safe_float(df_row.get(f"total_rev_{suffix}", 0.0))
    total_var = safe_float(df_row.get(f"total_var_{suffix}", 0.0))
    total_oh  = safe_float(df_row.get(f"total_oh_{suffix}", 0.0))
    miles     = safe_float(df_row.get(f"miles_{suffix}", 0.0))

    if raw > 0:
        rpl = safe_div(total_rev, raw)
        vpl = safe_div(total_var, raw)
        ohl = safe_div(total_oh,  raw)
        loh = safe_div(miles,     raw)
        core_or = safe_div(total_var + total_oh, total_rev, default=net["Core_OR"]) if total_rev > 0 else net["Core_OR"]
        ppl = rpl - (vpl + ohl)
        substituted = False
    else:
        # Substitute network averages from comparison period
        rpl = net["RPL"]
        vpl = net["VPL"]
        ohl = net["OHL"]
        loh = net["LOH"]
        core_or = net["Core_OR"]
        ppl = net["ProfitPerLoad"]
        substituted = True

    return {
        f"Revenue_per_Load_{suffix.upper()}": rpl,
        f"Variable_per_Load_{suffix.upper()}": vpl,
        f"Overhead_per_Load_{suffix.upper()}": ohl,
        f"LOH_{suffix.upper()}": loh,
        f"Core_OR_{suffix.upper()}": core_or,
        f"ProfitPerLoad_{suffix.upper()}": ppl,
        f"Core_OR_{suffix.upper()}_is_substituted": substituted,
    }

def compute_impacts(row: Dict[str, Any]) -> None:
    # Effective loads used in impact math
    lA = safe_float(row["loads_A"])
    lB = safe_float(row["loads_B"])
    dL = lA - lB

    pplA = safe_float(row["ProfitPerLoad_A"])
    pplB = safe_float(row["ProfitPerLoad_B"])
    net_pplB = safe_float(row.get("Net_ProfitPerLoad_B", pplB))

    # Volume-shift valued at prior-period quality
    row["Dollarized_S"] = dL * pplB
    # Shift relative to network baseline quality
    row["Impact_S"] = dL * (pplB - net_pplB)
    # Performance delta at average effective volume
    avg_eff_loads = 0.5 * (lA + lB)
    row["Impact_D"] = (pplA - pplB) * avg_eff_loads

    # Composite with stability weighting and conflict guard
    stability = min(1.0, avg_eff_loads / float(L0))
    base_comp = stability * row["Impact_D"] + (1.0 - stability) * row["Dollarized_S"]
    conflict = (row["Impact_D"] * row["Dollarized_S"]) < 0
    row["Composite"] = 0.7 * base_comp if conflict else base_comp

def attach_driver_details(
    section: str,
    rec: Dict[str, Any],
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    weeks_A: int,
    weeks_B: int,
) -> None:
    """
    Populate driver detail arrays with no thresholds:
    - For 'customers': lanes, inbound_areas, outbound_areas for that customer
    - For 'lanes': customers for that lane
    - For 'inbound': customers for that dest area
    - For 'outbound': customers for that orig area
    Each driver item includes raw_loads_A/B and effective loads_A/B per normalization rule.
    """
    def pack(rawA, rawB, extra: Dict[str, Any]):
        lA, lB = effective_loads(rawA, rawB, weeks_A, weeks_B)
        item = {
            "raw_loads_A": int(rawA),
            "raw_loads_B": int(rawB),
            "loads_A": float(lA),
            "loads_B": float(lB),
        }
        item.update(extra)
        return item

    if section == "customers":
        cust = rec.get("BILLTO_NAME", rec.get("name", ""))
        A_c = df_A[df_A["BILLTO_NAME"] == cust]
        B_c = df_B[df_B["BILLTO_NAME"] == cust]
        # lanes
        gA = A_c.groupby(["ORIG_AREA", "DEST_AREA"]).size()
        gB = B_c.groupby(["ORIG_AREA", "DEST_AREA"]).size()
        lanes = []
        for key in set(gA.index).union(set(gB.index)):
            rawA = int(gA.get(key, 0))
            rawB = int(gB.get(key, 0))
            lanes.append(pack(rawA, rawB, {"ORIG_AREA": key[0], "DEST_AREA": key[1]}))
        # inbound areas
        gA = A_c.groupby(["DEST_AREA"]).size()
        gB = B_c.groupby(["DEST_AREA"]).size()
        inbound = []
        for dest in set(gA.index).union(set(gB.index)):
            rawA = int(gA.get(dest, 0))
            rawB = int(gB.get(dest, 0))
            inbound.append(pack(rawA, rawB, {"DEST_AREA": dest}))
        # outbound areas
        gA = A_c.groupby(["ORIG_AREA"]).size()
        gB = B_c.groupby(["ORIG_AREA"]).size()
        outbound = []
        for orig in set(gA.index).union(set(gB.index)):
            rawA = int(gA.get(orig, 0))
            rawB = int(gB.get(orig, 0))
            outbound.append(pack(rawA, rawB, {"ORIG_AREA": orig}))
        rec["drivers"] = {
            "customer_lanes": lanes,
            "customer_inbound_areas": inbound,
            "customer_outbound_areas": outbound,
        }
    elif section == "lanes":
        orig = rec.get("ORIG_AREA", "")
        dest = rec.get("DEST_AREA", "")
        A_l = df_A[(df_A["ORIG_AREA"] == orig) & (df_A["DEST_AREA"] == dest)]
        B_l = df_B[(df_B["ORIG_AREA"] == orig) & (df_B["DEST_AREA"] == dest)]
        gA = A_l.groupby(["BILLTO_NAME"]).size()
        gB = B_l.groupby(["BILLTO_NAME"]).size()
        customers = []
        for cust in set(gA.index).union(set(gB.index)):
            rawA = int(gA.get(cust, 0))
            rawB = int(gB.get(cust, 0))
            customers.append(pack(rawA, rawB, {"BILLTO_NAME": cust}))
        rec["drivers"] = {"lane_customers": customers}
    elif section == "inbound":
        dest = rec.get("DEST_AREA", rec.get("name", ""))
        A_i = df_A[df_A["DEST_AREA"] == dest]
        B_i = df_B[df_B["DEST_AREA"] == dest]
        gA = A_i.groupby(["BILLTO_NAME"]).size()
        gB = B_i.groupby(["BILLTO_NAME"]).size()
        customers = []
        for cust in set(gA.index).union(set(gB.index)):
            rawA = int(gA.get(cust, 0))
            rawB = int(gB.get(cust, 0))
            customers.append(pack(rawA, rawB, {"BILLTO_NAME": cust}))
        rec["drivers"] = {"inbound_customers": customers}
    elif section == "outbound":
        orig = rec.get("ORIG_AREA", rec.get("name", ""))
        A_o = df_A[df_A["ORIG_AREA"] == orig]
        B_o = df_B[df_B["ORIG_AREA"] == orig]
        gA = A_o.groupby(["BILLTO_NAME"]).size()
        gB = B_o.groupby(["BILLTO_NAME"]).size()
        customers = []
        for cust in set(gA.index).union(set(gB.index)):
            rawA = int(gA.get(cust, 0))
            rawB = int(gB.get(cust, 0))
            customers.append(pack(rawA, rawB, {"BILLTO_NAME": cust}))
        rec["drivers"] = {"outbound_customers": customers}

def build_entity_table(
    section: str,
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    key_cols: List[str],
    weeks_A: int,
    weeks_B: int,
    net_for_sub: Dict[str, float],
) -> List[Dict[str, Any]]:
    # Aggregations
    A = build_agg(df_A, key_cols, "A")
    B = build_agg(df_B, key_cols, "B")
    T = A.join(B, how="outer").fillna(0.0).reset_index()

    rows: List[Dict[str, Any]] = []
    for _, row in T.iterrows():
        rec = {k: row[k] for k in key_cols}
        # Name label
        if section == "customers":
            rec["name"] = str(rec.get("BILLTO_NAME", "")).strip()
        elif section == "outbound":
            rec["name"] = str(rec.get("ORIG_AREA", "")).strip()
        elif section == "inbound":
            rec["name"] = str(rec.get("DEST_AREA", "")).strip()
        elif section == "lanes":
            o = str(rec.get("ORIG_AREA", "")).strip()
            d = str(rec.get("DEST_AREA", "")).strip()
            rec["name"] = f"{o} → {d}".strip()
        else:
            rec["name"] = "Unknown"

        rawA = safe_int(row.get("raw_loads_A", 0))
        rawB = safe_int(row.get("raw_loads_B", 0))
        rec["raw_loads_A"] = rawA
        rec["raw_loads_B"] = rawB

        # Effective loads for impact math
        effA, effB = effective_loads(rawA, rawB, weeks_A, weeks_B)
        rec["loads_A"] = float(effA)
        rec["loads_B"] = float(effB)
        rec["d_loads"] = float(effA - effB)

        # Weekly loads (for transparency; also used if periods differ in filter path)
        rec["loads_A_weekly"] = weekly_loads(rawA, weeks_A)
        rec["loads_B_weekly"] = weekly_loads(rawB, weeks_B)

        # Per-load metrics with substitution
        plA = per_load_metrics(row, "A", net_for_sub)
        plB = per_load_metrics(row, "B", net_for_sub)
        rec.update(plA)
        rec.update(plB)
        rec["Net_ProfitPerLoad_B"] = net_for_sub["ProfitPerLoad"]

        # Impacts & composite
        compute_impacts(rec)

        # Driver details (no thresholds)
        attach_driver_details(section, rec, df_A, df_B, weeks_A, weeks_B)

        rows.append(rec)

    return rows

def apply_low_volume_filter(
    rows: List[Dict[str, Any]],
    weeks_A: int,
    weeks_B: int,
    network_total_A: int,
    network_total_B: int
) -> List[Dict[str, Any]]:
    """
    Apply 0.75% rule per your spec:
      - If periods are equal length: compare totals vs totals (entity_raw vs network_total) each period.
      - If periods differ: compare weekly averages vs weekly averages (entity_weekly vs network_weekly) each period.
    Drop only if entity is <= threshold in BOTH periods.
    """
    equal_lengths = (weeks_A == weeks_B)

    if equal_lengths:
        # thresholds on totals
        thr_A = LOW_VOL_PCT * float(network_total_A)
        thr_B = LOW_VOL_PCT * float(network_total_B)
        def _entity_basis_A(r): return safe_float(r.get("raw_loads_A", 0.0))
        def _entity_basis_B(r): return safe_float(r.get("raw_loads_B", 0.0))
    else:
        # thresholds on weekly averages
        net_weekly_A = safe_div(network_total_A, weeks_A)
        net_weekly_B = safe_div(network_total_B, weeks_B)
        thr_A = LOW_VOL_PCT * net_weekly_A
        thr_B = LOW_VOL_PCT * net_weekly_B
        def _entity_basis_A(r): return safe_float(r.get("loads_A_weekly", 0.0))
        def _entity_basis_B(r): return safe_float(r.get("loads_B_weekly", 0.0))

    out = []
    for r in rows:
        a = _entity_basis_A(r)
        b = _entity_basis_B(r)
        if (a <= thr_A) and (b <= thr_B):
            continue
        out.append(r)
    return out

def dedupe_top10(items: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in sorted(items, key=lambda x: abs(safe_float(x.get("Composite", 0.0))), reverse=True):
        key = (it.get("section"), it.get("name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out

# ---------------------- Main ----------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize freight CSV into JSON for narrate/render.")
    ap.add_argument("--csv", required=True, help="Path to input CSV, e.g. in/SCAC_YYYYMMDD_OpenAIAPI.csv")
    ap.add_argument("--outdir", default="work", help="Output directory (default: work)")
    return ap.parse_args()

def compute_summary(
    df_current: pd.DataFrame,
    df_baseline: pd.DataFrame,
    op_code: str,
    label_current: str,
    label_baseline: str,
) -> Dict[str, Any]:
    """Compute the MMQB summary payload for the given periods."""

    df_A = df_current.copy()
    df_B = df_baseline.copy()

    weeks_A = len(sorted(pd.to_datetime(df_A["WEEK_ENDING"].dropna()).unique().tolist()))
    weeks_B = len(sorted(pd.to_datetime(df_B["WEEK_ENDING"].dropna()).unique().tolist()))
    if weeks_A == 0 or weeks_B == 0:
        raise ValueError(f"Invalid periods: weeks_A={weeks_A}, weeks_B={weeks_B}")

    # Network metrics for substitution baseline (prefer baseline; fallback to current)
    net_B = compute_network_metrics(df_B)
    if net_B["raw_loads"] == 0:
        net_B = compute_network_metrics(df_A)

    # Totals needed for filter thresholds (per spec)
    network_total_A = int(len(df_A))
    network_total_B = int(len(df_B))

    # Weekly totals (for transparency in meta)
    weekly_total_A = weekly_loads(network_total_A, weeks_A)
    weekly_total_B = weekly_loads(network_total_B, weeks_B)

    customers = build_entity_table("customers", df_A, df_B, ["BILLTO_NAME"], weeks_A, weeks_B, net_B)
    outbound  = build_entity_table("outbound",  df_A, df_B, ["ORIG_AREA"],  weeks_A, weeks_B, net_B)
    inbound   = build_entity_table("inbound",   df_A, df_B, ["DEST_AREA"],  weeks_A, weeks_B, net_B)
    lanes     = build_entity_table("lanes",     df_A, df_B, ["ORIG_AREA","DEST_AREA"], weeks_A, weeks_B, net_B)

    # Apply low-volume filter as specified (totals for equal periods, weekly avgs if different)
    cust_f  = apply_low_volume_filter(customers, weeks_A, weeks_B, network_total_A, network_total_B)
    outb_f  = apply_low_volume_filter(outbound,  weeks_A, weeks_B, network_total_A, network_total_B)
    inb_f   = apply_low_volume_filter(inbound,   weeks_A, weeks_B, network_total_A, network_total_B)
    lanes_f = apply_low_volume_filter(lanes,     weeks_A, weeks_B, network_total_A, network_total_B)

    # Top 10 by |Composite|
    top_candidates = []
    for section_name, rows in [("customers", cust_f), ("outbound", outb_f), ("inbound", inb_f), ("lanes", lanes_f)]:
        for r in rows:
            top_candidates.append({
                "section": section_name,
                "name": r["name"],
                "Composite": r.get("Composite", 0.0),
                "Impact_D": r.get("Impact_D", 0.0),
                "Impact_S": r.get("Impact_S", 0.0),
                "Dollarized_S": r.get("Dollarized_S", 0.0),
                "loads_A": r.get("loads_A", 0.0),
                "loads_B": r.get("loads_B", 0.0),
            })
    top10 = dedupe_top10(top_candidates, limit=10)

    top_lookup = {(t["section"], t["name"]) for t in top10}

    def add_stars(section, rows):
        out = []
        for r in rows:
            rr = dict(r)
            rr["star"] = (section, r["name"]) in top_lookup
            out.append(rr)
        return out

    cust_f  = add_stars("customers", cust_f)
    outb_f  = add_stars("outbound",  outb_f)
    inb_f   = add_stars("inbound",   inb_f)
    lanes_f = add_stars("lanes",     lanes_f)

    normalization_mode = "equal_period_totals" if weeks_A == weeks_B else "longer_period_weekly"

    meta = {
        "SCAC": op_code,
        "A_label": label_current,
        "B_label": label_baseline,
        "weeks_A": weeks_A,
        "weeks_B": weeks_B,
        "rows_A": int(len(df_A)),
        "rows_B": int(len(df_B)),
        "weekly_total_A": weekly_total_A,
        "weekly_total_B": weekly_total_B,
        "network_total_A": network_total_A,  # for transparency
        "network_total_B": network_total_B,  # for transparency
        "normalization_mode": normalization_mode,
        "display_loads_ceil_A": bool(weeks_A > 1 and weeks_A != weeks_B),
        "display_loads_ceil_B": bool(weeks_B > 1 and weeks_A != weeks_B),
        "network_B": {
            "raw_loads": net_B["raw_loads"],
            "RPL": net_B["RPL"],
            "VPL": net_B["VPL"],
            "OHL": net_B["OHL"],
            "LOH": net_B["LOH"],
            "Core_OR": net_B["Core_OR"],
            "ProfitPerLoad": net_B["ProfitPerLoad"],
        },
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return {
        "meta": meta,
        "tables": {
            "customers": cust_f,
            "outbound":  outb_f,
            "inbound":   inb_f,
            "lanes":     lanes_f,
        },
        "top10": top10,
    }


def main():
    args = parse_args()
    csv_path = args.csv
    outdir = args.outdir

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    base = os.path.basename(csv_path)
    scac = base.split("_")[0] if "_" in base else "SCAC"

    df = pd.read_csv(csv_path)
    ensure_columns(df)
    df = prepare_df(df)

    uniq_weeks = sorted(df["WEEK_ENDING"].dropna().unique().tolist())
    uniq_weeks = [pd.Timestamp(w) for w in uniq_weeks]
    if len(uniq_weeks) < 2:
        raise ValueError("Not enough distinct Saturdays to form two periods.")

    A_weeks, B_weeks = pick_periods_by_weeks(uniq_weeks)
    weeks_A = len(A_weeks)
    weeks_B = len(B_weeks)
    if weeks_A == 0 or weeks_B == 0:
        raise ValueError(f"Invalid periods: weeks_A={weeks_A}, weeks_B={weeks_B}")

    df_A = df[df["WEEK_ENDING"].isin(A_weeks)].copy()
    df_B = df[df["WEEK_ENDING"].isin(B_weeks)].copy()

    A_label = format_label(A_weeks)
    B_label = format_label(B_weeks)

    summary = compute_summary(df_A, df_B, scac, A_label, B_label)

    os.makedirs(outdir, exist_ok=True)
    a_end = A_label.split(" ")[0].replace("/", "")
    b_end = B_label.split(" ")[0].replace("/", "")
    out_name = f"{scac}_{a_end}_vs_{b_end}_summary.json"
    out_path = os.path.join(outdir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(out_path)

if __name__ == "__main__":
    main()
