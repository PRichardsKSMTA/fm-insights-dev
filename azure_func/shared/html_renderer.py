"""HTML rendering helpers for client/internal MMQB documents."""

from __future__ import annotations

import copy
import re
from html import escape
from typing import Any, Dict, List, Tuple


def _infer_weeks_from_label(label: str | None) -> int | None:
    if not label:
        return None
    match = re.search(r"\((\d+)\s*wk\)", label, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:  # pragma: no cover - defensive programming
        return None


def _infer_weeks(meta: Dict[str, Any]) -> Tuple[int, int]:
    for a_key, b_key in [
        ("weeks_A", "weeks_B"),
        ("period_A_weeks", "period_B_weeks"),
        ("weeksA", "weeksB"),
    ]:
        if a_key in meta and b_key in meta:
            try:
                a = int(meta[a_key])
                b = int(meta[b_key])
            except Exception:  # pragma: no cover - defensive programming
                continue
            if a >= 1 and b >= 1:
                return a, b
    for a_key, b_key in [
        ("A_label", "B_label"),
        ("period_A_label", "period_B_label"),
        ("label_A", "label_B"),
    ]:
        a = meta.get(a_key)
        b = meta.get(b_key)
        if not (a or b):
            continue
        weeks_a = _infer_weeks_from_label(a if isinstance(a, str) else None)
        weeks_b = _infer_weeks_from_label(b if isinstance(b, str) else None)
        if weeks_a and weeks_b:
            return weeks_a, weeks_b
    return 1, 1


def _pick(row: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:  # pragma: no cover - defensive programming
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except Exception:  # pragma: no cover - defensive programming
        return None


def _fmt(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except Exception:
        return escape(str(value)) if value is not None else ""
    return f"{number:,.{decimals}f}"


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value))):,d}"
    except Exception:  # pragma: no cover - defensive programming
        return "0"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _should_ceil_display(meta: Dict[str, Any], side: str, weeks_side: int) -> bool:
    key = f"display_loads_ceil_{side}"
    if key in meta:
        try:
            return bool(meta[key])
        except Exception:  # pragma: no cover - defensive programming
            return False
    return weeks_side > 1 and meta.get("normalization_mode") != "equal_period_totals"


def _filter_basis_note(meta: Dict[str, Any]) -> str:
    weeks_a, weeks_b = _infer_weeks(meta)
    equal = weeks_a == weeks_b
    basis = (
        "totals"
        if equal or meta.get("normalization_mode") == "equal_period_totals"
        else "weekly averages"
    )
    network_total_a = _coerce_float(meta.get("network_total_A"))
    network_total_b = _coerce_float(meta.get("network_total_B"))
    weekly_total_a = _coerce_float(meta.get("weekly_total_A"))
    weekly_total_b = _coerce_float(meta.get("weekly_total_B"))
    threshold_a = threshold_b = None
    if basis == "totals":
        if network_total_a is not None and network_total_b is not None:
            threshold_a = 0.0075 * network_total_a
            threshold_b = 0.0075 * network_total_b
    elif weekly_total_a is not None and weekly_total_b is not None:
        threshold_a = 0.0075 * weekly_total_a
        threshold_b = 0.0075 * weekly_total_b

    parts = []
    if basis == "totals":
        parts.append("Low-volume filter 0.75% used total loads for each period.")
    else:
        parts.append(
            "Low-volume filter 0.75% used weekly average loads for each period."
        )
    if threshold_a is not None and threshold_b is not None:
        parts.append(
            f"(A threshold {_fmt(threshold_a, 2)}, B threshold {_fmt(threshold_b, 2)}.)"
        )
    return " ".join(parts)


def _rounding_note(meta: Dict[str, Any], weeks_a: int, weeks_b: int) -> str:
    notes: List[str] = []
    if _should_ceil_display(meta, "A", weeks_a):
        notes.append(
            "A-period loads shown are rounded up when averaged across multiple weeks."
        )
    if _should_ceil_display(meta, "B", weeks_b):
        notes.append(
            "B-period loads shown are rounded up when averaged across multiple weeks."
        )
    return " ".join(notes)


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
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{escape(title)}</title>{CSS}</head><body>"
    )


def _html_footer() -> str:
    return "</body></html>"

def _render_client_with_template(doc: Dict[str, Any], template: str) -> str:
    """
    Render a client-facing document using an HTML skeleton pulled from the DB.

    The template supports simple Mustache-style placeholders:

      Global (single value):
        {{CLIENT_CODE}}
        {{TITLE}}
        {{LABEL_A}}
        {{LABEL_B}}
        {{COMPARISON_LABEL}}
        {{LOW_VOLUME_NOTE}}
        {{FINAL_WORD}}

      Repeating sections:
        {{#HIGHLIGHTS}}...{{/HIGHLIGHTS}}
          Inside the block: {{HIGHLIGHT}}

        {{#STORIES}}...{{/STORIES}}
          Inside the block:
            {{INDEX}}          1..10
            {{ARROW}}          e.g. ▲, ▼, →
            {{ARROW_WORD}}     "up" | "down" | "flat"
            {{ENTITY_LABEL}}   "Inbound Area: IN-IND"
            {{HEADLINE}}       headline text without entity label
            {{HEADLINE_FULL}}  full headline text
            {{DRIVER_DETAIL}}  driver detail line

    Everything substituted is HTML-escaped.
    """

    meta = doc.get("meta", {}) or {}
    title = str(doc.get("title", "Client Report"))
    client_code = str(meta.get("client_code") or meta.get("SCAC") or "CLIENT")

    # Period labels and comparison
    weeks_a, weeks_b = _infer_weeks(meta)
    label_a = str(meta.get("A_label") or meta.get("period_A_label") or "")
    label_b = str(meta.get("B_label") or meta.get("period_B_label") or "")
    comparison_label = ""
    if weeks_a and weeks_b:
        comparison_label = f"{_plural(weeks_a, 'week')} vs {_plural(weeks_b, 'week')}"

    low_volume_note = _filter_basis_note(meta)

    highlights = [h for h in doc.get("highlights", []) if isinstance(h, str)]
    stories = [s for s in doc.get("stories", []) if isinstance(s, dict)]

    def _split_headline(raw: str) -> Tuple[str, str]:
        """Split 'Entity — rest' into (entity, rest)."""
        text = raw.strip()
        parts = text.split(" — ", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return text, text

    def _arrow_word(ch: str) -> str:
        ch = (ch or "").strip()
        if ch in ("▲", "△", "↑"):
            return "up"
        if ch in ("▼", "▽", "↓"):
            return "down"
        return "flat"

    def _render_section(section: str, items: List[Dict[str, str]], text: str) -> str:
        """Very small Mustache-like repeater for {{#SECTION}}...{{/SECTION}}."""
        start_tag = "{{#" + section + "}}"
        end_tag = "{{/" + section + "}}"
        start = text.find(start_tag)
        end = text.find(end_tag)
        if start == -1 or end == -1 or end <= start:
            return text

        before = text[:start]
        block = text[start + len(start_tag) : end]
        after = text[end + len(end_tag) :]

        rendered_blocks: List[str] = []
        for item in items:
            block_text = block
            for key, value in item.items():
                block_text = block_text.replace("{{" + key + "}}", value)
            rendered_blocks.append(block_text)

        return before + "".join(rendered_blocks) + after

    # Build items for HIGHLIGHTS
    highlight_items: List[Dict[str, str]] = [
        {"HIGHLIGHT": escape(h)} for h in highlights
    ]

    # Build items for STORIES
    story_items: List[Dict[str, str]] = []
    for index, story in enumerate(stories, 1):
        raw_headline = str(story.get("headline", "")).strip()
        entity_label, headline_body = _split_headline(raw_headline)
        arrow = str(story.get("arrow", "")).strip()
        driver_detail = str(story.get("driver_detail", "")).strip()
        story_items.append(
            {
                "INDEX": str(index),
                "ARROW": escape(arrow),
                "ARROW_WORD": escape(_arrow_word(arrow)),
                "ENTITY_LABEL": escape(entity_label),
                "HEADLINE": escape(headline_body),
                "HEADLINE_FULL": escape(raw_headline),
                "DRIVER_DETAIL": escape(driver_detail),
            }
        )

    # Apply section expansions
    rendered = template
    rendered = _render_section("STORIES", story_items, rendered)
    rendered = _render_section("HIGHLIGHTS", highlight_items, rendered)

    # Single-value placeholders
    final_word = escape(str(doc.get("final_word", "")).strip())

    replacements = {
        "CLIENT_CODE": escape(client_code),
        "TITLE": escape(title),
        "LABEL_A": escape(label_a),
        "LABEL_B": escape(label_b),
        "COMPARISON_LABEL": escape(comparison_label),
        "LOW_VOLUME_NOTE": escape(low_volume_note),
        "FINAL_WORD": final_word,
    }

    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)

    return rendered


def _render_client(doc: Dict[str, Any]) -> str:
    meta = doc.get("meta", {}) or {}
    title = doc.get("title", "Client Report")
    weeks_a, weeks_b = _infer_weeks(meta)
    a_label = meta.get("A_label", "")
    b_label = meta.get("B_label", "")

    parts = [_html_header(title)]
    parts.append(f"<h1>{escape(title)}</h1>")
    subparts = []
    if a_label:
        subparts.append(f"A: {escape(str(a_label))}")
    if b_label:
        subparts.append(f"B: {escape(str(b_label))}")
    if weeks_a and weeks_b:
        subparts.append(f"({_plural(weeks_a, 'week')} vs {_plural(weeks_b, 'week')})")
    if subparts:
        parts.append(f"<div class='meta'>{' | '.join(subparts)}</div>")

    highlights = [h for h in doc.get("highlights", []) if isinstance(h, str)]
    if highlights:
        parts.append("<h2>Highlights</h2><div class='list'><ul>")
        for highlight in highlights:
            parts.append(f"<li>{escape(highlight)}</li>")
        parts.append("</ul></div>")

    stories = [s for s in doc.get("stories", []) if isinstance(s, dict)]
    if stories:
        parts.append("<h2>Top 10 Stories</h2>")
        for index, story in enumerate(stories, 1):
            arrow = str(story.get("arrow", "")).strip()
            headline = str(story.get("headline", "")).strip()
            driver = str(story.get("driver_detail", "")).strip()
            if re.match(r"^[\u25B2\u25BC\u2192▲▼→]\s", headline):
                prefix = ""
            else:
                prefix = f"{escape(arrow)} " if arrow else ""
            parts.append(
                "<div class='story'><b>"
                f"{index}. {prefix}{escape(headline)}</b>"
                f"<div class='driver'>{escape(driver)}</div></div>"
            )
    else:
        parts.append(
            "<div class='note'>No story payload found. Check narrate step.</div>"
        )

    parts.append(f"<div class='note'>{escape(_filter_basis_note(meta))}</div>")

    final_word = str(doc.get("final_word", "")).strip()
    if final_word:
        parts.append("<h2>Final Word</h2>")
        parts.append(f"<div>{escape(final_word)}</div>")

    parts.append(_html_footer())
    return "".join(parts)


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


def _add_display_loads(
    rows: List[Dict[str, Any]], meta: Dict[str, Any], weeks_a: int, weeks_b: int
) -> None:
    ceil_a = _should_ceil_display(meta, "A", weeks_a)
    ceil_b = _should_ceil_display(meta, "B", weeks_b)
    for row in rows:
        loads_a = _coerce_float(_pick(row, ["loads_A", "weekly_loads_A"]))
        loads_b = _coerce_float(_pick(row, ["loads_B", "weekly_loads_B"]))
        raw_a = _coerce_int(row.get("raw_loads_A"))
        raw_b = _coerce_int(row.get("raw_loads_B"))
        if ceil_a:
            row["loads_A_display"] = (
                int(-(-loads_a // 1))
                if loads_a is not None and loads_a > 0
                else (raw_a or 0)
            )
        else:
            row["loads_A_display"] = (
                raw_a if raw_a is not None else (_coerce_int(loads_a) or 0)
            )
        if ceil_b:
            row["loads_B_display"] = (
                int(-(-loads_b // 1))
                if loads_b is not None and loads_b > 0
                else (raw_b or 0)
            )
        else:
            row["loads_B_display"] = (
                raw_b if raw_b is not None else (_coerce_int(loads_b) or 0)
            )


def _entity_name(section: str, row: Dict[str, Any]) -> str:
    if section == "customers":
        return str(_pick(row, ["BILLTO_NAME", "customer", "name"], ""))
    if section == "outbound":
        return str(_pick(row, ["ORIG_AREA", "origin", "name"], ""))
    if section == "inbound":
        return str(_pick(row, ["DEST_AREA", "destination", "name"], ""))
    if section == "lanes":
        lane = _pick(row, ["lane", "name"], None)
        if lane:
            return str(lane)
        origin = str(_pick(row, ["ORIG_AREA", "origin"], "")).strip()
        destination = str(_pick(row, ["DEST_AREA", "destination"], "")).strip()
        if origin or destination:
            return f"{origin} → {destination}"
    return str(_pick(row, ["name", "id"], ""))


def _render_internal_table(
    section: str, rows: List[Dict[str, Any]], meta: Dict[str, Any]
) -> str:
    if not rows:
        return ""
    weeks_a, weeks_b = _infer_weeks(meta)
    _add_display_loads(rows, meta, weeks_a, weeks_b)

    present = [(key, label) for key, label in INTERNAL_COLS if any(key in r for r in rows)]

    html = [
        f"<h2>{section.capitalize()}",
        f"<span class='pill'>A: {_plural(weeks_a, 'week')}</span>",
        f"<span class='pill'>B: {_plural(weeks_b, 'week')}</span>",
        "</h2>",
    ]
    html.append("<table><thead><tr>")
    html.append("<th>Entity</th>")
    for _, label in present:
        html.append(f"<th>{escape(label)}</th>")
    html.append("</tr></thead><tbody>")

    for row in rows:
        name = _entity_name(section, row)
        star = "★ " if row.get("star") else ""
        html.append("<tr>")
        star_html = f"<span class=\"star\">{star}</span>" if star else ""
        html.append(f"<td>{star_html}{escape(name)}</td>")
        for key, _ in present:
            value = row.get(key, "")
            if key.endswith("_display") or key.startswith("raw_loads"):
                html.append(f"<td>{_fmt_int(value)}</td>")
            elif key in ("Core_OR_A", "Core_OR_B"):
                coerced = _coerce_float(value)
                html.append(
                    f"<td>{_fmt(coerced * 100.0 if coerced is not None else value, 1)}</td>"
                )
            else:
                html.append(f"<td>{_fmt(value, 2)}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")

    notes = " ".join([
        _rounding_note(meta, weeks_a, weeks_b),
        _filter_basis_note(meta),
    ]).strip()
    if notes:
        html.append(f"<div class='note'>{escape(notes)}</div>")

    return "".join(html)


def _render_internal(doc: Dict[str, Any]) -> str:
    meta = doc.get("meta", {}) or {}
    title = doc.get("title", "Internal Report")
    weeks_a, weeks_b = _infer_weeks(meta)
    a_label = meta.get("A_label", "")
    b_label = meta.get("B_label", "")

    parts = [_html_header(title)]
    parts.append(f"<h1>{escape(title)}</h1>")
    subparts = []
    if a_label:
        subparts.append(f"A: {escape(str(a_label))}")
    if b_label:
        subparts.append(f"B: {escape(str(b_label))}")
    if weeks_a and weeks_b:
        subparts.append(f"({_plural(weeks_a, 'week')} vs {_plural(weeks_b, 'week')})")
    if subparts:
        parts.append(f"<div class='meta'>{' | '.join(subparts)}</div>")

    tables = doc.get("tables", {}) or {}
    for key in ("customers", "outbound", "inbound", "lanes"):
        rows = tables.get(key, [])
        if rows:
            parts.append(_render_internal_table(key, rows, meta))

    parts.append(_html_footer())
    return "".join(parts)


def _copy_doc(doc: Dict[str, Any] | None) -> Dict[str, Any]:
    return copy.deepcopy(doc or {})


def render_client(doc: Dict[str, Any], template: str | None = None) -> str:
    """Render a client-facing document to HTML without mutating the source.

    If *template* is provided, it should be a full HTML skeleton string with
    Mustache-style placeholders supported by _render_client_with_template.
    """
    copied = _copy_doc(doc)
    if template:
        return _render_client_with_template(copied, template)
    return _render_client(copied)


def render_internal(doc: Dict[str, Any]) -> str:
    """Render an internal document to HTML without mutating the source."""

    return _render_internal(_copy_doc(doc))


def render_html(doc: Dict[str, Any]) -> str:
    """Render a document by inspecting its report type."""

    copied = _copy_doc(doc)
    report_type = str((copied.get("meta", {}) or {}).get("report_type", "")).lower()
    if report_type == "client":
        return _render_client(copied)
    return _render_internal(copied)


__all__ = ["render_client", "render_internal", "render_html"]

