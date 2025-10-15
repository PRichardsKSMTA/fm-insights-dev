#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
narrate.py
Produces:
  - out/<SCAC>_<Aend>_vs_<Bend>_client.json  (client-facing stories via OpenAI; substitution-aware)
  - out/<SCAC>_<Aend>_vs_<Bend>_internal.json (numeric tables preserved)

Key behaviors:
- Headlines DO NOT include the arrow (renderer adds it).
- When a side has zero loads (substitution used), the headline says freight **appeared** (0→X) or **disappeared** (X→0).
- Core OR uses **baseline phrasing** when a substitution occurs; otherwise shows A vs B (×100 with 1 decimal).
- RPM mentioned sparingly and never when substitution is used.
- Fallback writer varies phrasing to avoid robotic repetition.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

OUT_DIR = "out"
PROMPT_PATH = os.path.join("prompts", "report_system.md")

# RPM significance thresholds (sparingly)
RPM_ABS_CUTOFF = 0.15   # $/mi
RPM_PCT_CUTOFF = 0.12   # 12%

# --- optional env (.env) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- optional OpenAI v1 client ---
_OPENAI = None
try:
    from openai import (
        APIError,
        BadRequestError,
        OpenAI,
        RateLimitError,
    )
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        if api_key.startswith("sk-proj"):
            _OPENAI = OpenAI(default_headers={"OpenAI-Beta": "use-project-api"})
        else:
            _OPENAI = OpenAI()
except Exception:  # pragma: no cover - import/runtime guard
    APIError = BadRequestError = RateLimitError = Exception  # type: ignore
    OpenAI = None  # type: ignore
    _OPENAI = None

OPENAI_ERRORS = (BadRequestError, RateLimitError, APIError)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default

def _rpm_from_metrics(rpl: Any, loh: Any) -> float:
    rpl = _safe_float(rpl)
    loh = _safe_float(loh)
    return rpl / loh if loh > 0 else 0.0

def _rpm_flag(a: float, b: float) -> Tuple[bool, float, float]:
    delta = a - b
    pct = (delta / b) if abs(b) > 1e-9 else (0.0 if abs(a) < 1e-9 else 1.0)
    flag = (abs(delta) >= RPM_ABS_CUTOFF) or (abs(pct) >= RPM_PCT_CUTOFF)
    return flag, delta, pct

def log(s: str) -> None:
    print(s, flush=True)


def _get_openai_client() -> Optional["OpenAI"]:
    global _OPENAI  # pylint: disable=global-statement
    if _OPENAI is not None:
        return _OPENAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    try:
        if api_key.startswith("sk-proj"):
            _OPENAI = OpenAI(default_headers={"OpenAI-Beta": "use-project-api"})
        else:
            _OPENAI = OpenAI()
    except Exception as exc:  # pragma: no cover - network/env specific
        try:
            log(json.dumps({"event": "openai_client_error", "error": str(exc)}))
        except Exception:
            log(f"[openai] client init failed: {exc}")
        _OPENAI = None
    return _OPENAI

def resolve_summary_path(p: str) -> str:
    if os.path.exists(p):
        return os.path.abspath(p)
    candidate = os.path.join("work", os.path.basename(p))
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    raise FileNotFoundError(f"Summary not found: {p} (also tried {candidate})")


def _call_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    *,
    purpose: str,
) -> Optional[Dict[str, Any]]:
    client = _get_openai_client()
    if client is None:
        return None

    try:
        log(json.dumps({"event": "openai_call_start", "purpose": purpose}))
    except Exception:
        log(f"[openai] start {purpose}")

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
    except OPENAI_ERRORS as exc:  # pragma: no cover - requires live API
        try:
            log(
                json.dumps(
                    {
                        "event": "openai_call_error",
                        "purpose": purpose,
                        "error": str(exc),
                        "type": exc.__class__.__name__,
                    }
                )
            )
        except Exception:
            log(f"[openai] error {purpose}: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - network/runtime specific
        try:
            log(
                json.dumps(
                    {
                        "event": "openai_call_error",
                        "purpose": purpose,
                        "error": str(exc),
                        "type": exc.__class__.__name__,
                    }
                )
            )
        except Exception:
            log(f"[openai] error {purpose}: {exc}")
        return None

    content = ""
    if resp.choices:
        content = (resp.choices[0].message.content or "").strip()

    try:
        log(
            json.dumps(
                {
                    "event": "openai_call_end",
                    "purpose": purpose,
                    "finish_reason": resp.choices[0].finish_reason if resp.choices else None,
                }
            )
        )
    except Exception:
        log(f"[openai] end {purpose}")

    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        try:
            log(
                json.dumps(
                    {
                        "event": "openai_parse_error",
                        "purpose": purpose,
                        "error": str(exc),
                    }
                )
            )
        except Exception:
            log(f"[openai] parse error {purpose}: {exc}")
        return None

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_prompt_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return (
            "You are a freight analyst. Write concise, business-ready client narratives using the provided data. "
            "Return strict JSON per the requested schema."
        )

def derive_id_bits(meta: Dict[str, Any], fallback_name: str) -> Tuple[str, str, str]:
    client_code = str(meta.get("client_code") or meta.get("SCAC") or "CLIENT").strip() or "CLIENT"
    def _end_mmddyy(label: Optional[str]) -> str:
        if not label:
            return "000000"
        d = str(label).split(" ")[0].replace("/", "")
        return d if d.isdigit() else "000000"
    a_label = meta.get("A_label") or meta.get("period_A_label") or ""
    b_label = meta.get("B_label") or meta.get("period_B_label") or ""
    a_end = _end_mmddyy(a_label); b_end = _end_mmddyy(b_label)
    if client_code == "CLIENT":
        base = os.path.basename(fallback_name)
        if "_" in base:
            client_code = base.split("_")[0]
    return client_code, a_end, b_end


def _period_label(meta: Dict[str, Any], which: str) -> str:
    key = which.upper()
    for candidate in (
        f"{key}_label",
        f"label_{key}",
        f"{key}Label",
        f"label{key}",
    ):
        value = meta.get(candidate)
        if value:
            return str(value)

    period = meta.get(f"period{key}") or meta.get(f"period_{key}")
    saturday = None
    weeks = None
    if isinstance(period, dict):
        saturday = period.get("saturday") or period.get("end") or period.get("label")
        weeks = period.get("weeks") or period.get("week_count")

    if isinstance(saturday, str):
        try:
            dt = datetime.fromisoformat(saturday)
            saturday = dt.strftime("%m/%d/%y")
        except Exception:
            pass

    label_core = str(saturday or key)
    if weeks:
        return f"{label_core} ({weeks}wk)"
    return label_core

def tiny_exec_summary(meta: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    weeks_A = int(meta.get("weeks_A", 0) or 0)
    weeks_B = int(meta.get("weeks_B", 0) or 0)
    a_label = meta.get("A_label", "A")
    b_label = meta.get("B_label", "B")
    net_tot_A = meta.get("network_total_A", meta.get("rows_A", 0))
    net_tot_B = meta.get("network_total_B", meta.get("rows_B", 0))
    weekly_A = meta.get("weekly_total_A", None)
    weekly_B = meta.get("weekly_total_B", None)
    top10 = summary.get("top10", [])
    top_names = [t.get("name") for t in top10[:5] if isinstance(t, dict)]
    return {
        "Period A": f"{a_label} ({weeks_A} wk)",
        "Period B": f"{b_label} ({weeks_B} wk)",
        "Low-volume filter basis": "Totals per period" if weeks_A == weeks_B else "Weekly averages per period",
        "Network loads A": f"{int(net_tot_A):,}" if isinstance(net_tot_A, (int, float)) else str(net_tot_A),
        "Network loads B": f"{int(net_tot_B):,}" if isinstance(net_tot_B, (int, float)) else str(net_tot_B),
        "Weekly total A": f"{weekly_A:,.2f}" if isinstance(weekly_A, (int, float)) else "",
        "Weekly total B": f"{weekly_B:,.2f}" if isinstance(weekly_B, (int, float)) else "",
        "Top movers (sample)": ", ".join([n for n in top_names if n]),
        "Generated at": datetime.utcnow().strftime("%Y-%m-%d %H:%MZ"),
    }

# ---------- driver selection ----------

def select_primary_driver(section: str, entity_name: str, tables: Dict[str, Any]) -> str:
    def _best(items: List[Dict[str, Any]], label_fn):
        if not items:
            return None
        best = max(items, key=lambda it: abs(float(it.get("loads_A", 0.0)) - float(it.get("loads_B", 0.0))))
        try:
            return label_fn(best)
        except Exception:
            return None

    section_rows = (tables or {}).get(section, [])
    row = next((r for r in section_rows if str(r.get("name","")).strip() == str(entity_name).strip()), None)
    if not isinstance(row, dict):
        return ""

    drv = row.get("drivers", {}) or {}
    if section == "customers":
        label = _best(drv.get("customer_lanes", []), lambda it: f"Lane {it.get('ORIG_AREA','')} → {it.get('DEST_AREA','')}")
        if label: return label
        label = _best(drv.get("customer_outbound_areas", []), lambda it: f"Outbound {it.get('ORIG_AREA','')}")
        if label: return label
        label = _best(drv.get("customer_inbound_areas", []), lambda it: f"Inbound {it.get('DEST_AREA','')}")
        if label: return label
    elif section == "lanes":
        label = _best(drv.get("lane_customers", []), lambda it: f"Customer {it.get('BILLTO_NAME','')}")
        if label: return label
    elif section in ("inbound", "outbound"):
        key = "inbound_customers" if section == "inbound" else "outbound_customers"
        label = _best(drv.get(key, []), lambda it: f"Customer {it.get('BILLTO_NAME','')}")
        if label: return label
    return ""

# ---------- client building ----------

def build_client_payload(
    summary: Dict[str, Any],
    model: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    meta = summary.get("meta", {})
    top10 = summary.get("top10", [])
    tables = summary.get("tables", {})

    items: List[Dict[str, Any]] = []
    for t in top10:
        section = t.get("section")
        name = t.get("name")
        src_rows = (tables or {}).get(section, [])
        src = next((r for r in src_rows if str(r.get("name","")).strip()==str(name).strip()), {})

        loads_A = _safe_float(src.get("loads_A", t.get("loads_A")))
        loads_B = _safe_float(src.get("loads_B", t.get("loads_B")))
        zero_A = loads_A <= 1e-6
        zero_B = loads_B <= 1e-6

        core_a  = _safe_float(src.get("Core_OR_A"))
        core_b  = _safe_float(src.get("Core_OR_B"))
        rpl_a   = src.get("Revenue_per_Load_A"); rpl_b = src.get("Revenue_per_Load_B")
        loh_a   = src.get("LOH_A"); loh_b = src.get("LOH_B")

        rpm_a = _rpm_from_metrics(rpl_a, loh_a)
        rpm_b = _rpm_from_metrics(rpl_b, loh_b)
        rpm_big, rpm_delta, rpm_pct = _rpm_flag(rpm_a, rpm_b)

        subA = bool(src.get("Core_OR_A_is_substituted", False)) or zero_A
        subB = bool(src.get("Core_OR_B_is_substituted", False)) or zero_B
        # Never mention RPM if substitution occurred
        rpm_big = rpm_big and not (subA or subB)

        driver_line = select_primary_driver(section, name, tables)
        comp = _safe_float(t.get("Composite", 0.0))
        arrow = "▲" if comp > 0 else ("▼" if comp < 0 else "→")

        items.append({
            "section": section, "name": name, "arrow": arrow,
            "loads_A": loads_A, "loads_B": loads_B,
            "zero_A": zero_A, "zero_B": zero_B,
            "Core_OR_A": core_a, "Core_OR_B": core_b,
            "Core_OR_A_pct": core_a * 100.0, "Core_OR_B_pct": core_b * 100.0,
            "Composite": comp, "Impact_D": t.get("Impact_D"), "Impact_S": t.get("Impact_S"),
            "driver_hint": driver_line,
            "Core_OR_A_is_substituted": subA, "Core_OR_B_is_substituted": subB,
            "rpm_A": rpm_a, "rpm_B": rpm_b,
            "rpm_delta": rpm_delta, "rpm_pct": rpm_pct,
            "rpm_is_big_factor": bool(rpm_big),
        })

    sys_prompt = system_prompt or read_prompt_text(PROMPT_PATH)
    labels = {"A": _period_label(meta, "A"), "B": _period_label(meta, "B")}

    user_payload = {
        "schema": {
            "type": "object",
            "properties": {
                "highlights": {"type": "array", "items": {"type": "string"}},
                "stories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "arrow": {"type": "string"},
                            "headline": {"type": "string"},
                            "driver_detail": {"type": "string"},
                        },
                        "required": ["arrow", "headline", "driver_detail"],
                    },
                },
                "final_word": {"type": "string"},
            },
            "required": ["stories", "final_word"],
        },
        "data": {
            "period_labels": labels,
            "items": items,
        },
    }

    data = _call_chat_completion(
        model,
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
        purpose="client_report",
    )
    if isinstance(data, dict) and "stories" in data:
        return {
            "highlights": data.get("highlights", []),
            "stories": data["stories"],
            "final_word": data.get("final_word", ""),
        }

    # Fallback: varied, deterministic phrasing with substitution-aware logic
    A = labels["A"]
    B = labels["B"]
    stories: List[Dict[str, Any]] = []
    i = 0
    for it in items:
        arrow = it["arrow"]
        sec = it["section"]; nm = it["name"] or ""
        la = int(round(_safe_float(it.get("loads_A", 0))))
        lb = int(round(_safe_float(it.get("loads_B", 0))))
        dL = la - lb
        coreA_pct = _safe_float(it.get("Core_OR_A_pct"))
        coreB_pct = _safe_float(it.get("Core_OR_B_pct"))
        subA = bool(it.get("Core_OR_A_is_substituted"))
        subB = bool(it.get("Core_OR_B_is_substituted"))
        zeroA = bool(it.get("zero_A")); zeroB = bool(it.get("zero_B"))
        rpm_a = _safe_float(it.get("rpm_A")); rpm_b = _safe_float(it.get("rpm_B"))
        rpm_big = bool(it.get("rpm_is_big_factor"))

        # Entity label
        if sec == "lanes":
            ent = f"Lane: {nm}"
        elif sec == "inbound":
            ent = f"Inbound Area: {nm}"
        elif sec == "outbound":
            ent = f"Outbound Area: {nm}"
        else:
            ent = f"Customer: {nm}"

        # Core OR text with baseline phrasing when substituted
        if subA and subB:
            core_txt = "Core OR at network baseline in both periods"
        elif subB and not subA:
            core_txt = f"Core OR {coreA_pct:.1f} vs baseline in {B}"
        elif subA and not subB:
            core_txt = f"Core OR baseline in {A}; {coreB_pct:.1f} in {B}"
        else:
            core_improved = [
                "Core OR improved to {A:.1f} (from {B:.1f})",
                "Core OR moved down to {A:.1f} (was {B:.1f})",
                "Core OR better at {A:.1f} (vs {B:.1f})",
            ]
            core_worsened = [
                "Core OR worsened to {A:.1f} (from {B:.1f})",
                "Core OR rose to {A:.1f} (was {B:.1f})",
                "Core OR higher at {A:.1f} (vs {B:.1f})",
            ]
            core_steady = [
                "Core OR held near {A:.1f} (from {B:.1f})",
                "Core OR steady at {A:.1f} (vs {B:.1f})",
                "Core OR about {A:.1f} (from {B:.1f})",
            ]
            if coreA_pct < coreB_pct - 0.1:
                core_txt = core_improved[i % len(core_improved)].format(A=coreA_pct, B=coreB_pct)
            elif coreA_pct > coreB_pct + 0.1:
                core_txt = core_worsened[i % len(core_worsened)].format(A=coreA_pct, B=coreB_pct)
            else:
                core_txt = core_steady[i % len(core_steady)].format(A=coreA_pct, B=coreB_pct)

        # Loads phrase with appear/disappear
        if zeroB and not zeroA:
            loads_txt = f"and freight appeared (0→{la})"
        elif zeroA and not zeroB:
            loads_txt = f"and freight disappeared ({lb}→0)"
        else:
            if dL >= 1:
                loads_opts = [f"and loads increased by {dL}", f"with loads up {dL}", f"and volume up {dL} loads"]
                loads_txt = loads_opts[i % len(loads_opts)]
            elif dL <= -1:
                loads_opts = [f"and loads decreased by {abs(dL)}", f"with loads down {abs(dL)}", f"and volume down {abs(dL)} loads"]
                loads_txt = loads_opts[i % len(loads_opts)]
            else:
                loads_opts = ["and loads held steady", "with volume flat", "and little change in loads"]
                loads_txt = loads_opts[i % len(loads_opts)]

        # Optional RPM clause (never if substitution happened)
        rpm_txt = ""
        if rpm_big and not (subA or subB):
            if rpm_a > rpm_b:
                rpm_txt = f"; RPM improved by ${abs(rpm_a - rpm_b):.2f}/mi"
            elif rpm_a < rpm_b:
                rpm_txt = f"; RPM worsened by ${abs(rpm_a - rpm_b):.2f}/mi"

        headline = f"{ent} — {core_txt} {loads_txt}{rpm_txt}."

        # Driver line (variety)
        helped = (arrow == "▲") or (arrow == "→" and _safe_float(it.get("Impact_D",0))+_safe_float(it.get("Impact_S",0)) >= 0)
        driver_hint = it.get("driver_hint") or "Mix/volume shift."
        verb = "helped" if helped else "hurt"
        if sec == "lanes":
            driver_opts = [
                f"Biggest driver on this lane: {driver_hint} {verb} the network most.",
                f"Top driver on this lane: {driver_hint} {verb} the network most.",
                f"Primary driver: {driver_hint} {verb} the network most.",
            ]
        elif sec in ("inbound", "outbound"):
            driver_opts = [
                f"Biggest driver in this area: {driver_hint} {verb} the network most.",
                f"Top driver in this area: {driver_hint} {verb} the network most.",
                f"Primary driver: {driver_hint} {verb} the network most.",
            ]
        else:
            driver_opts = [
                f"Biggest driver within this customer: {driver_hint} {verb} the network most.",
                f"Top driver within this customer: {driver_hint} {verb} the network most.",
                f"Primary driver: {driver_hint} {verb} the network most.",
            ]
        driver_line = driver_opts[i % len(driver_opts)]

        stories.append({"arrow": arrow, "headline": headline, "driver_detail": driver_line})
        i += 1

    final_word = f"From {B} to {A}, these items drove the biggest changes. Focus on ▲ items for upside, mitigate ▼."
    return {"highlights": [], "stories": stories, "final_word": final_word}


def _fmt_internal_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if abs(value) >= 1:
                return f"{value:,.1f}" if not float(value).is_integer() else f"{int(value):,}"
            return f"{value:,.2f}"
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return _fmt_internal_number(number)


def _maybe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if value in (None, ""):
        return None
    try:
        cleaned = str(value).replace(",", "").strip()
        if not cleaned:
            return None
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _format_metric_change(a: Any, b: Any, *, better_when_down: bool = False) -> str:
    a_num = _maybe_float(a)
    b_num = _maybe_float(b)
    if a_num is None or b_num is None:
        if a == b:
            return "flat"
        return "vs"
    delta = a_num - b_num
    if abs(delta) < 1e-9:
        return "flat"
    direction = "▲" if delta > 0 else "▼"
    better = delta < 0 if better_when_down else delta > 0
    status = "better" if better else "worse"
    return f"{direction} {_fmt_internal_number(abs(delta))} ({status})"


def build_internal_exec_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    meta = summary.get("meta", {})
    network = summary.get("network", {})

    label_a = _period_label(meta, "A")
    label_b = _period_label(meta, "B")

    network_a = network.get("A", {}) or {}
    network_b = network.get("B", {}) or {}

    metrics_config = [
        {
            "name": "Load Count",
            "A": network_a.get("loads"),
            "B": network_b.get("loads"),
        },
        {
            "name": "Core OR",
            "A": network_a.get("or_pct"),
            "B": network_b.get("or_pct"),
            "better_when_down": True,
        },
        {
            "name": "Revenue/Load",
            "A": network_a.get("rpl"),
            "B": network_b.get("rpl"),
        },
        {
            "name": "Overhead/Load",
            "A": network_a.get("overhead"),
            "B": network_b.get("overhead"),
        },
        {
            "name": "Variable/Load",
            "A": network_a.get("variable"),
            "B": network_b.get("variable"),
        },
        {
            "name": "LOH",
            "A": network_a.get("loh"),
            "B": network_b.get("loh"),
        },
    ]

    metrics: List[Dict[str, str]] = []
    for metric in metrics_config:
        a_val = metric.get("A")
        b_val = metric.get("B")
        if a_val is None and b_val is None:
            continue
        metrics.append(
            {
                "name": metric["name"],
                "A_txt": _fmt_internal_number(a_val),
                "B_txt": _fmt_internal_number(b_val),
                "change": _format_metric_change(
                    a_val,
                    b_val,
                    better_when_down=bool(metric.get("better_when_down")),
                ),
            }
        )

    return {
        "label_A": label_a,
        "label_B": label_b,
        "metrics": metrics,
    }


def build_internal_payload(
    summary: Dict[str, Any],
    model: str,
    system_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    del model, system_prompt  # internal path is deterministic

    return {"executive_summary": build_internal_exec_summary(summary)}

def build_docs(
    summary: Dict[str, Any],
    model: str,
    system_prompt: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = summary.get("meta", {})
    meta_client = dict(meta); meta_client["report_type"] = "client"
    meta_internal = dict(meta); meta_internal["report_type"] = "internal"

    client_payload = build_client_payload(summary, model, system_prompt=system_prompt)
    internal_payload = build_internal_payload(summary, model, system_prompt=system_prompt)

    client_doc = {
        "title": f"{meta.get('SCAC','CLIENT')} Performance Report",
        "meta": meta_client,
        "highlights": client_payload.get("highlights", []),
        "stories": client_payload.get("stories", []),
        "final_word": client_payload.get("final_word", ""),
        "top10": summary.get("top10", []),
    }
    tiny_summary = tiny_exec_summary(meta_internal, summary)
    exec_summary = build_internal_exec_summary(summary)
    if internal_payload and isinstance(internal_payload.get("executive_summary"), dict):
        exec_summary = internal_payload["executive_summary"]
    if not exec_summary.get("metrics"):
        exec_summary = tiny_summary

    internal_doc = {
        "title": f"{meta.get('SCAC','CLIENT')} Performance Report — Internal View",
        "meta": meta_internal,
        "tables": summary.get("tables", {}),
        "top10": summary.get("top10", []),
        "exec_summary": exec_summary,
        "tiny_exec_summary": tiny_summary,
        "network": summary.get("network", {}),
    }
    return client_doc, internal_doc

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Turn summary JSON into client & internal JSONs.")
    ap.add_argument("--summary", required=True, help="Path to summary JSON (or name in ./work).")
    default_model = os.environ.get("MMQB_DEFAULT_MODEL", "gpt-4o-mini")
    ap.add_argument(
        "--model",
        default=default_model,
        help=f"OpenAI model to use for narratives (default: {default_model}).",
    )
    return ap.parse_args()

def main():
    args = parse_args()
    summary_path = resolve_summary_path(args.summary)
    summary = load_json(summary_path)

    meta = summary.get("meta", {})
    client_code, a_end, b_end = derive_id_bits(meta, os.path.basename(summary_path))
    meta = dict(meta); meta.setdefault("client_code", client_code); summary["meta"] = meta

    client_doc, internal_doc = build_docs(summary, args.model)

    os.makedirs(OUT_DIR, exist_ok=True)
    client_out = os.path.join(OUT_DIR, f"{client_code}_{a_end}_vs_{b_end}_client.json")
    internal_out = os.path.join(OUT_DIR, f"{client_code}_{a_end}_vs_{b_end}_internal.json")

    with open(client_out, "w", encoding="utf-8") as f:
        json.dump(client_doc, f, indent=2)
    with open(internal_out, "w", encoding="utf-8") as f:
        json.dump(internal_doc, f, indent=2)

    print(client_out)
    print(internal_out)

if __name__ == "__main__":
    main()
