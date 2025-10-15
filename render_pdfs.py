#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
render_pdfs.py
Client:
- Renders Highlights, Top-10 stories, Final Word.
- Prevents duplicate arrows (adds arrow only if headline doesn’t already start with one).

Internal:
- Renders the 4 tables (Load Count display + raw, Core OR, Rev/Load, Var/Load, OH/Load, LOH).
- Shows Core OR as percentage-style (×100) with 1 decimal.
- Keeps rounding/filter notes.
"""

import argparse
import json
import os
import re
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional, Tuple

# ---------- Utilities ----------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _infer_weeks_from_label(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    m = re.search(r"\((\d+)\s*wk\)", label, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def _infer_weeks(meta: Dict[str, Any]) -> Tuple[int, int]:
    for a_key, b_key in [("weeks_A","weeks_B"),("period_A_weeks","period_B_weeks"),("weeksA","weeksB")]:
        if a_key in meta and b_key in meta:
            try:
                a = int(meta[a_key]); b = int(meta[b_key])
                if a >= 1 and b >= 1: return a, b
            except Exception:
                pass
    for a_key, b_key in [("A_label","B_label"),("period_A_label","period_B_label"),("label_A","label_B")]:
        a = meta.get(a_key); b = meta.get(b_key)
        if a or b:
            aw = _infer_weeks_from_label(a); bw = _infer_weeks_from_label(b)
            if aw and bw: return aw, bw
    return 1, 1

def _pick(row: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default

def _coerce_float(x: Any) -> Optional[float]:
    try: return float(x)
    except Exception: return None

def _coerce_int(x: Any) -> Optional[int]:
    try: return int(round(float(x)))
    except Exception: return None

def _fmt(v: Any, decimals: int = 2) -> str:
    try:
        f = float(v); return f"{f:,.{decimals}f}"
    except Exception:
        return escape(str(v)) if v is not None else ""

def _fmt_int(v: Any) -> str:
    try:
        return f"{int(round(float(v))):,d}"
    except Exception:
        return "0"

def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"

# ---------- Notes (filter + rounding) ----------

def _should_ceil_display(meta: Dict[str, Any], side: str, weeks_side: int) -> bool:
    key = f"display_loads_ceil_{side}"
    if key in meta:
        try: return bool(meta[key])
        except Exception: pass
    return weeks_side > 1 and (meta.get("normalization_mode") != "equal_period_totals")

def _filter_basis_note(meta: Dict[str, Any]) -> str:
    weeks_A, weeks_B = _infer_weeks(meta)
    equal = weeks_A == weeks_B
    basis = "totals" if equal or meta.get("normalization_mode") == "equal_period_totals" else "weekly averages"
    network_total_A = _coerce_float(meta.get("network_total_A"))
    network_total_B = _coerce_float(meta.get("network_total_B"))
    weekly_total_A  = _coerce_float(meta.get("weekly_total_A"))
    weekly_total_B  = _coerce_float(meta.get("weekly_total_B"))
    thr_A = thr_B = None
    if basis == "totals":
        if network_total_A is not None and network_total_B is not None:
            thr_A = 0.0075 * network_total_A
            thr_B = 0.0075 * network_total_B
    else:
        if weekly_total_A is not None and weekly_total_B is not None:
            thr_A = 0.0075 * weekly_total_A
            thr_B = 0.0075 * weekly_total_B
    parts = []
    parts.append("Low-volume filter 0.75% used total loads for each period." if basis=="totals"
                 else "Low-volume filter 0.75% used weekly average loads for each period.")
    if thr_A is not None and thr_B is not None:
        parts.append(f"(A threshold {_fmt(thr_A, 2)}, B threshold {_fmt(thr_B, 2)}.)")
    return " ".join(parts)

def _rounding_note(meta: Dict[str, Any], weeks_A: int, weeks_B: int) -> str:
    notes = []
    if _should_ceil_display(meta, "A", weeks_A):
        notes.append("A-period loads shown are rounded up when averaged across multiple weeks.")
    if _should_ceil_display(meta, "B", weeks_B):
        notes.append("B-period loads shown are rounded up when averaged across multiple weeks.")
    return " ".join(notes)

# ---------- HTML Shell ----------

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; margin: 24px; color: #111; }
  h1 { font-size: 24px; margin: 0 0 8px; }
  h2 { font-size: 18px; margin: 24px 0 8px; }
  .meta { color: #444; margin-bottom: 16px; }
  .pill { display:inline-block; border:1px solid #e5e7eb; border-radius:999px; padding:2px 8px; font-size:12px; color:#444; margin-left:6px; }
  .note { font-size: 12px; color: #555; margin: 4px 0 18px; }
  .list { margin: 8px 0 16px; }
  .story { margin: 8px 0 12px; }
  .driver { color: #444; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; }
  th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  thead tr { background: #f8fafc; }
  .star { color: #d97706; font-weight: 700; }
</style>
"""

def _html_header(title: str) -> str:
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{escape(title)}</title>{CSS}</head><body>"

def _html_footer() -> str:
    return "</body></html>"

# ---------- Client rendering ----------

def _render_client(doc: Dict[str, Any]) -> str:
    meta = doc.get("meta", {})
    title = doc.get("title", "Client Report")
    weeks_A, weeks_B = _infer_weeks(meta)
    a_label = meta.get("A_label",""); b_label = meta.get("B_label","")

    parts = [_html_header(title)]
    parts.append(f"<h1>{escape(title)}</h1>")
    sub = []
    if a_label: sub.append(f"A: {escape(a_label)}")
    if b_label: sub.append(f"B: {escape(b_label)}")
    if weeks_A and weeks_B: sub.append(f"({_plural(weeks_A,'week')} vs {_plural(weeks_B,'week')})")
    if sub: parts.append(f"<div class='meta'>{' | '.join(sub)}</div>")

    # Highlights
    hl = [h for h in doc.get("highlights", []) if isinstance(h, str)]
    if hl:
        parts.append("<h2>Highlights</h2><div class='list'><ul>")
        for h in hl:
            parts.append(f"<li>{escape(h)}</li>")
        parts.append("</ul></div>")

    # Stories
    stories = [s for s in doc.get("stories", []) if isinstance(s, dict)]
    if stories:
        parts.append("<h2>Top 10 Stories</h2>")
        for i, s in enumerate(stories, 1):
            arrow = str(s.get("arrow","")).strip()
            head  = str(s.get("headline","")).strip()
            driver= str(s.get("driver_detail","")).strip()

            # Prevent double arrows: if headline already starts with one, don't prepend.
            if re.match(r"^[\u25B2\u25BC\u2192▲▼→]\s", head):
                prefix = ""
            else:
                prefix = f"{escape(arrow)} " if arrow else ""

            parts.append(f"<div class='story'><b>{i}. {prefix}{escape(head)}</b><div class='driver'>{escape(driver)}</div></div>")
    else:
        parts.append("<div class='note'>No story payload found. Check narrate step.</div>")

    # Filter note
    parts.append(f"<div class='note'>{escape(_filter_basis_note(meta))}</div>")

    # Final word
    fw = str(doc.get("final_word","")).strip()
    if fw:
        parts.append("<h2>Final Word</h2>")
        parts.append(f"<div>{escape(fw)}</div>")

    parts.append(_html_footer())
    return "".join(parts)

# ---------- Internal rendering ----------

INTERNAL_COLS = [
    ("loads_A_display", "Loads A (display)"),
    ("loads_B_display", "Loads B (display)"),
    ("raw_loads_A", "Raw Loads A"),
    ("raw_loads_B", "Raw Loads B"),
    ("Core_OR_A", "Core OR A"),
    ("Core_OR_B", "Core OR B"),
    ("Revenue_per_Load_A", "Rev/Load A"),
    ("Revenue_per_Load_B", "Rev/Load B"),
    ("Overhead_per_Load_A", "OH/Load A"),
    ("Overhead_per_Load_B", "OH/Load B"),
    ("Variable_per_Load_A", "Var/Load A"),
    ("Variable_per_Load_B", "Var/Load B"),
    ("LOH_A", "LOH A"),
    ("LOH_B", "LOH B"),
]

def _add_display_loads(rows: List[Dict[str, Any]], meta: Dict[str, Any], weeks_A: int, weeks_B: int) -> None:
    ceil_A = _should_ceil_display(meta, "A", weeks_A)
    ceil_B = _should_ceil_display(meta, "B", weeks_B)
    for r in rows:
        loads_A = _coerce_float(_pick(r, ["loads_A","weekly_loads_A"]))
        loads_B = _coerce_float(_pick(r, ["loads_B","weekly_loads_B"]))
        raw_A = _coerce_int(r.get("raw_loads_A"))
        raw_B = _coerce_int(r.get("raw_loads_B"))
        if ceil_A:
            r["loads_A_display"] = int(-(-loads_A // 1)) if (loads_A is not None and loads_A > 0) else (raw_A or 0)
        else:
            r["loads_A_display"] = raw_A if raw_A is not None else (_coerce_int(loads_A) or 0)
        if ceil_B:
            r["loads_B_display"] = int(-(-loads_B // 1)) if (loads_B is not None and loads_B > 0) else (raw_B or 0)
        else:
            r["loads_B_display"] = raw_B if raw_B is not None else (_coerce_int(loads_B) or 0)

def _entity_name(section: str, row: Dict[str, Any]) -> str:
    if section == "customers":
        return str(_pick(row, ["BILLTO_NAME","customer","name"], ""))
    if section == "outbound":
        return str(_pick(row, ["ORIG_AREA","origin","name"], ""))
    if section == "inbound":
        return str(_pick(row, ["DEST_AREA","destination","name"], ""))
    if section == "lanes":
        lane = _pick(row, ["lane","name"], None)
        if lane: return str(lane)
        o = str(_pick(row, ["ORIG_AREA","origin"], "")).strip()
        d = str(_pick(row, ["DEST_AREA","destination"], "")).strip()
        if o or d: return f"{o} → {d}"
    return str(_pick(row, ["name","id"], ""))

def _render_internal_table(section: str, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    if not rows:
        return ""
    weeks_A, weeks_B = _infer_weeks(meta)
    _add_display_loads(rows, meta, weeks_A, weeks_B)

    present = [(k, lbl) for (k, lbl) in INTERNAL_COLS if any(k in r for r in rows)]

    html = [f"<h2>{section.capitalize()}",
            f"<span class='pill'>A: {_plural(weeks_A,'week')}</span>",
            f"<span class='pill'>B: {_plural(weeks_B,'week')}</span>",
            "</h2>"]
    html.append("<table><thead><tr>")
    html.append("<th>Entity</th>")
    for _, lbl in present:
        html.append(f"<th>{escape(lbl)}</th>")
    html.append("</tr></thead><tbody>")

    for r in rows:
        name = _entity_name(section, r)
        star = "★ " if r.get("star") else ""
        html.append("<tr>")
        html.append(f"<td>{star and f'<span class=\"star\">{star}</span>' or ''}{escape(name)}</td>")
        for key, _ in present:
            val = r.get(key, "")
            if key.endswith("_display") or key.startswith("raw_loads"):
                html.append(f"<td>{_fmt_int(val)}</td>")
            elif key in ("Core_OR_A","Core_OR_B"):
                # show ×100 with 1 decimal
                html.append(f"<td>{_fmt(_coerce_float(val)*100.0 if _coerce_float(val) is not None else val, 1)}</td>")
            else:
                html.append(f"<td>{_fmt(val, 2)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    notes = " ".join([_rounding_note(meta, weeks_A, weeks_B), _filter_basis_note(meta)])
    if notes.strip():
        html.append(f"<div class='note'>{escape(notes)}</div>")

    return "".join(html)

def _render_internal(doc: Dict[str, Any]) -> str:
    meta = doc.get("meta", {})
    title = doc.get("title", "Internal Report")
    weeks_A, weeks_B = _infer_weeks(meta)
    a_label = meta.get("A_label",""); b_label = meta.get("B_label","")
    parts = [_html_header(title)]
    parts.append(f"<h1>{escape(title)}</h1>")
    sub = []
    if a_label: sub.append(f"A: {escape(a_label)}")
    if b_label: sub.append(f"B: {escape(b_label)}")
    if weeks_A and weeks_B: sub.append(f"({_plural(weeks_A,'week')} vs {_plural(weeks_B,'week')})")
    if sub: parts.append(f"<div class='meta'>{' | '.join(sub)}</div>")

    tables = doc.get("tables", {})
    for key in ("customers","outbound","inbound","lanes"):
        rows = tables.get(key, [])
        if rows:
            parts.append(_render_internal_table(key, rows, meta))

    parts.append(_html_footer())
    return "".join(parts)

# ---------- Dispatcher ----------

def render_doc(doc: Dict[str, Any]) -> str:
    rpt = (doc.get("meta", {}) or {}).get("report_type", "").lower()
    if rpt == "client":
        return _render_client(doc)
    return _render_internal(doc)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Render client/internal JSON to HTML.")
    ap.add_argument("--client_json", required=True, help="Path to client JSON file.")
    ap.add_argument("--internal_json", required=True, help="Path to internal JSON file.")
    ap.add_argument("--outdir", default="out", help="Output directory (default: out).")
    args = ap.parse_args()

    client = _load_json(args.client_json)
    internal = _load_json(args.internal_json)

    client_html = render_doc(client)
    internal_html = render_doc(internal)

    client_base = os.path.splitext(os.path.basename(args.client_json))[0]
    internal_base = os.path.splitext(os.path.basename(args.internal_json))[0]

    client_out = os.path.join(args.outdir, f"{client_base}.html")
    internal_out = os.path.join(args.outdir, f"{internal_base}.html")

    _write_text(client_out, client_html)
    _write_text(internal_out, internal_html)

    print("Rendered HTML:")
    print(client_out)
    print(internal_out)

if __name__ == "__main__":
    main()
