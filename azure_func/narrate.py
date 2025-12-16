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
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI, RateLimitError, APIError

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception:  # pragma: no cover - optional dependency
    BlobServiceClient = None  # type: ignore
    ContentSettings = None # type: ignore

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
LOG_BLOB_CONN_ENV = "MMQB_LOG_BLOB_STORAGE"
LOG_CONTAINER_NAME = "fm-insights-logs"

_BLOB_SERVICE_CLIENT: Optional["BlobServiceClient"] = None # type: ignore
_BLOB_INIT_FAILED = False

BANNED_SUBJECTIVE_WORDS = {
    "significant", "significantly",
    "notable", "notably",
    "dramatic", "dramatically",
    "surprising", "surprisingly",
    "material", "materially",
    "strong", "strongly",
    "weak", "weakly",
}

def _driver_effect(arrow: Any, impact_d: Any, impact_s: Any) -> str:
    """Map arrow + impact into 'helped' or 'hurt' per Scenario Rules v3."""
    ch = str(arrow or "").strip()
    try:
        impact_total = float(impact_d or 0) + float(impact_s or 0)
    except Exception:
        impact_total = 0.0

    if ch in ("▲", "△", "↑"):
        return "helped"
    if ch in ("▼", "▽", "↓"):
        return "hurt"
    # flat: use sign of Impact_D + Impact_S
    return "helped" if impact_total >= 0 else "hurt"


def _driver_detail_from_item(item: Dict[str, Any]) -> str:
    """Construct driver_detail exactly following Scenario Rules v3 templates."""
    section = str(item.get("section") or "").strip().lower()
    driver_hint = str(item.get("driver_hint") or "").strip()
    arrow = item.get("arrow")
    effect = _driver_effect(arrow, item.get("Impact_D"), item.get("Impact_S"))

    if section == "customers":
        template = "Biggest driver within this customer: {hint} {effect} the network most."
    elif section == "lanes":
        template = "Biggest driver on this lane: {hint} {effect} the network most."
    else:  # inbound / outbound / anything else treated as area
        template = "Biggest driver in this area: {hint} {effect} the network most."

    return template.format(hint=driver_hint, effect=effect)


def _normalize_driver_details(data: Dict[str, Any], items: list[Dict[str, Any]]) -> None:
    """
    Override driver_detail from the LLM with the canonical template output,
    using the original items list (1:1 with stories).
    """
    stories = data.get("stories") or []
    if not isinstance(stories, list):
        return

    for idx, story in enumerate(stories):
        if not isinstance(story, dict):
            continue
        if idx >= len(items):
            break
        story["driver_detail"] = _driver_detail_from_item(items[idx])
        

def _sanitize_subjective_language(data: Dict[str, Any]) -> None:
    """
    Remove banned subjective words from highlights and final_word
    per Narrative Formatting Rules v3.
    """
    def _clean_text(text: str) -> str:
        words = text.split()
        cleaned_words = []
        for w in words:
            bare = re.sub(r"[^\w]", "", w).lower()
            if bare in BANNED_SUBJECTIVE_WORDS:
                # drop the word entirely
                continue
            cleaned_words.append(w)
        return " ".join(cleaned_words)

    # Clean highlights
    hl = data.get("highlights") or []
    if isinstance(hl, list):
        data["highlights"] = [
            _clean_text(str(h)) for h in hl if isinstance(h, (str,))
        ]

    # Clean final_word
    fw = data.get("final_word")
    if isinstance(fw, str):
        data["final_word"] = _clean_text(fw)
        
        
def _maybe_int(value: Any) -> Optional[int]:
    """Best-effort cast to int, returning None on failure."""
    if value in (None, ""):
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _entity_label_from_item(item: Dict[str, Any]) -> str:
    """
    Build the ENTITY_LABEL per Scenario Rules v3:

    customers -> "Customer: {name}"
    lanes     -> "Lane: {name}"
    inbound   -> "Inbound Area: {name}"
    outbound  -> "Outbound Area: {name}"
    """
    section = str(item.get("section") or "").strip().lower()
    name = str(item.get("name") or "").strip()

    if section == "customers":
        return f"Customer: {name}"
    if section == "lanes":
        return f"Lane: {name}"
    if section == "inbound":
        return f"Inbound Area: {name}"
    if section == "outbound":
        return f"Outbound Area: {name}"
    return name or "Entity"


def _core_or_clause_from_item(item: Dict[str, Any]) -> Optional[str]:
    """
    Construct the Core OR clause using the strict Scenario Rules v3:

    - OR_A_display = round(Core_OR_A * 100, 1)
    - OR_B_display = round(Core_OR_B * 100, 1)
    - abs_diff_pp = abs(Core_OR_A - Core_OR_B) * 100

    Verbs (closed list):
      tightened, improved, surged, slipped, eroded, collapsed, held near

    Thresholds:
      If abs_diff_pp < 1.0:
          "Core OR held near {OR_A_display}"
      Else if Core_OR_A < Core_OR_B:
          improvement_pp = (Core_OR_B - Core_OR_A) * 100
          1.0–5.0  -> tightened
          5.1–10.0 -> improved
          ≥10.1    -> surged
      Else if Core_OR_A > Core_OR_B:
          worsening_pp = (Core_OR_A - Core_OR_B) * 100
          1.0–5.0  -> slipped
          5.1–10.0 -> eroded
          ≥10.1    -> collapsed
    """
    a = _maybe_float(item.get("Core_OR_A"))
    b = _maybe_float(item.get("Core_OR_B"))
    if a is None or b is None:
        # We can't safely compute the clause; leave headline as-is.
        return None

    or_a_display = round(a * 100.0, 1)
    or_b_display = round(b * 100.0, 1)

    abs_diff_pp = abs(a - b) * 100.0

    # Near-zero movement rule
    if abs_diff_pp < 1.0:
        return f"Core OR held near {or_a_display:.1f}"

    # Profitability improved (Core_OR_A < Core_OR_B)
    if a < b:
        improvement_pp = (b - a) * 100.0
        if improvement_pp <= 5.0:
            verb = "tightened"
        elif improvement_pp <= 10.0:
            verb = "improved"
        else:
            verb = "surged"
    # Profitability worsened (Core_OR_A > Core_OR_B)
    elif a > b:
        worsening_pp = (a - b) * 100.0
        if worsening_pp <= 5.0:
            verb = "slipped"
        elif worsening_pp <= 10.0:
            verb = "eroded"
        else:
            verb = "collapsed"
    else:
        # Exact equality (should have been caught by abs_diff_pp < 1.0, but keep safe)
        return f"Core OR held near {or_a_display:.1f}"

    return f"Core OR {verb} to {or_a_display:.1f} (from {or_b_display:.1f})"


def _loads_clause_from_item(item: Dict[str, Any]) -> Optional[str]:
    """
    Construct the loads clause using the exact Scenario Rules v3 load-change rules:

      diff = Loads_A - Loads_B

      If diff > 0:
          'loads increased by [abs_diff]'
      If diff < 0:
          'loads decreased by [abs_diff]'
      If diff = 0:
          'loads were unchanged at [Loads_A]'

    No 'held steady' or any synonyms are allowed.
    """
    loads_a = _maybe_int(item.get("loads_A"))
    loads_b = _maybe_int(item.get("loads_B"))
    if loads_a is None or loads_b is None:
        return None

    diff = loads_a - loads_b
    if diff > 0:
        return f"loads increased by {diff}"
    if diff < 0:
        return f"loads decreased by {abs(diff)}"
    return f"loads were unchanged at {loads_a}"


def _rpm_clause_from_item(item: Dict[str, Any]) -> str:
    """
    Construct the RPM clause using the strict gating rules:

    RPM is allowed only when ALL are true:
      1) rpm_is_big_factor is true
      2) Core_OR_A_is_substituted is false
      3) Core_OR_B_is_substituted is false

    Formatting (only when allowed):
      delta = rpm_A - rpm_B

      If delta > 0:
          '; RPM increased by $X.XX/mi'
      If delta < 0:
          '; RPM fell by $X.XX/mi'
      If delta == 0:
          no RPM clause
    """
    rpm_big = bool(item.get("rpm_is_big_factor"))
    sub_a = bool(item.get("Core_OR_A_is_substituted"))
    sub_b = bool(item.get("Core_OR_B_is_substituted"))

    # Strict gating
    if not rpm_big or sub_a or sub_b:
        return ""

    rpm_a = _maybe_float(item.get("rpm_A"))
    rpm_b = _maybe_float(item.get("rpm_B"))
    if rpm_a is None or rpm_b is None:
        return ""

    delta = rpm_a - rpm_b
    if abs(delta) < 1e-9:
        return ""

    change = abs(delta)
    if delta > 0:
        return f"; RPM increased by ${change:.2f}/mi"
    return f"; RPM fell by ${change:.2f}/mi"


def _rebuild_headlines(data: Dict[str, Any], items: list[Dict[str, Any]]) -> None:
    """
    Override the LLM-provided 'headline' for each story and rebuild it
    deterministically from the numeric 'items', using Scenario Rules v3.

    Pattern (must match Scenario Rules v3 / Narrative Formatting v3):

      [ENTITY_LABEL] - [Core OR clause] and [loads clause][; RPM clause]

    We also enforce that the 'arrow' in the story matches the original item.
    """
    stories = data.get("stories") or []
    if not isinstance(stories, list):
        return

    for idx, story in enumerate(stories):
        if not isinstance(story, dict):
            continue
        if idx >= len(items):
            break

        item = items[idx]

        core_clause = _core_or_clause_from_item(item)
        loads_clause = _loads_clause_from_item(item)
        if not core_clause or not loads_clause:
            # If we can't safely compute both, leave the original headline untouched.
            continue

        rpm_clause = _rpm_clause_from_item(item)
        entity_label = _entity_label_from_item(item)

        headline = f"{entity_label} - {core_clause} and {loads_clause}{rpm_clause}"

        story["headline"] = headline
        # Ensure arrow matches the numeric item, not the LLM.
        story["arrow"] = str(item.get("arrow") or "")


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


def _get_openai_client() -> Optional["OpenAI"]: # type: ignore
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


def _label_to_yyyymmdd(label: Optional[str]) -> str:
    """
    Convert a label like '11/15/25 (1wk)' into '20251115'.
    Falls back to stripping non-digits if it can't parse.
    """
    if not label:
        return "00000000"

    token = str(label).strip().split()[0]  # e.g. '11/15/25'

    # Try common formats explicitly.
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            dt = datetime.strptime(token, fmt)
            return dt.strftime("%Y%m%d")
        except Exception:
            continue

    # Fallback: keep only digits and interpret mmddyy if length==6
    digits = re.sub(r"\D", "", token)
    if len(digits) == 8:
        return digits
    if len(digits) == 6:
        try:
            mm = int(digits[0:2])
            dd = int(digits[2:4])
            yy = int(digits[4:6])
            year = 2000 + yy if yy < 50 else 1900 + yy
            dt = datetime(year, mm, dd)
            return dt.strftime("%Y%m%d")
        except Exception:
            return digits
    return (digits + "00000000")[:8] if digits else "00000000"


def _derive_log_identity(meta: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """
    Return (operation_cd, label_A, label_B, A_YYYYMMDD, B_YYYYMMDD) for logging.
    """
    op = (
        str(
            meta.get("operation_cd")
            or meta.get("client_code")
            or meta.get("SCAC")
            or "CLIENT"
        )
        .strip()
        or "CLIENT"
    )

    label_a = _period_label(meta, "A")  # uses your existing helper
    label_b = _period_label(meta, "B")

    a_ymd = _label_to_yyyymmdd(label_a)
    b_ymd = _label_to_yyyymmdd(label_b)

    return op, (label_a or ""), (label_b or ""), a_ymd, b_ymd


def _get_log_blob_client() -> Optional["BlobServiceClient"]: # type: ignore
    """
    Lazily construct a BlobServiceClient from MMQB_LOG_BLOB_STORAGE.
    Never raises; on failure, it disables blob logging for the process lifetime.
    """
    global _BLOB_SERVICE_CLIENT  # type: ignore[global-variable-not-assigned]
    global _BLOB_INIT_FAILED

    if _BLOB_INIT_FAILED:
        return None
    if _BLOB_SERVICE_CLIENT is not None:
        return _BLOB_SERVICE_CLIENT

    if BlobServiceClient is None:
        _BLOB_INIT_FAILED = True
        try:
            log(json.dumps({"event": "log_blob_unavailable", "reason": "azure.storage.blob not installed"}))
        except Exception:
            log("[log_blob] azure.storage.blob not installed")
        return None

    conn_str = os.environ.get(LOG_BLOB_CONN_ENV)
    if not conn_str:
        _BLOB_INIT_FAILED = True
        try:
            log(json.dumps({"event": "log_blob_unavailable", "reason": f"{LOG_BLOB_CONN_ENV} not set"}))
        except Exception:
            log(f"[log_blob] {LOG_BLOB_CONN_ENV} not set")
        return None

    try:
        _BLOB_SERVICE_CLIENT = BlobServiceClient.from_connection_string(conn_str)
        return _BLOB_SERVICE_CLIENT
    except Exception as exc:
        _BLOB_INIT_FAILED = True
        try:
            log(json.dumps({"event": "log_blob_init_error", "error": str(exc)}))
        except Exception:
            log(f"[log_blob] init error: {exc}")
        return None


def _write_log_blob(
    kind: str,
    operation_cd: str,
    a_ymd: str,
    b_ymd: str,
    timestamp: str,
    payload: Dict[str, Any],
) -> None:
    """
    Persist a JSON payload to Azure Blob Storage.

    kind: 'requests' or 'responses'
    """
    client = _get_log_blob_client()
    if client is None:
        return

    try:
        container = client.get_container_client(LOG_CONTAINER_NAME)
        try:
            container.create_container()
        except Exception:
            # Container already exists or cannot be created; best-effort only.
            pass

        blob_name = f"{kind}/{operation_cd}/{a_ymd}_vs_{b_ymd}_{operation_cd}_{kind}_{timestamp}.json"
        data = json.dumps(payload, indent=2, default=str).encode("utf-8")

        if ContentSettings is not None:
            content_settings = ContentSettings(content_type="application/json")
            container.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True,
                content_settings=content_settings,
            )
        else:
            container.upload_blob(
                name=blob_name,
                data=data,
                overwrite=True,
            )
    except Exception as exc:
        # Blob logging must never break the main pipeline.
        try:
            log(json.dumps({"event": "log_blob_write_error", "kind": kind, "error": str(exc)}))
        except Exception:
            log(f"[log_blob] write error ({kind}): {exc}")


def _call_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    *,
    purpose: str,
    log_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    ChatCompletion call that retrieves rule text from vector store
    using client.vector_stores.search (Assistants File Search backend),
    injects retrieved rules into the system prompt, and returns JSON.
    Compatible with openai==2.9.0.
    """

    client = _get_openai_client()
    if client is None:
        return None

    meta = log_meta or {}
    operation_cd, label_a, label_b, a_ymd, b_ymd = _derive_log_identity(meta)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # ---------------------------
    # VECTOR STORE RETRIEVAL
    # ---------------------------
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    retrieved_rules_text = ""

    if vector_store_id:
        try:
            results = client.vector_stores.search(
                vector_store_id=vector_store_id,
                query="MMQB numeric rules, core OR rules, freight narrative rules",
                max_num_results=10,
            )

            chunks = []
            for result in results.data:
                for item in result.content:
                    if hasattr(item, "text"):
                        chunks.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        chunks.append(item["text"])

            retrieved_rules_text = "\n\n".join(chunks)

        except Exception as exc:
            log(f"[vector_store] retrieval error: {exc}")

    # ---------------------------
    # FINAL SYSTEM MESSAGE
    # ---------------------------
    base_system_prompt = messages[0]["content"]

    if retrieved_rules_text:
        system_message = (
            "THE FOLLOWING RULES MUST BE OBEYED:\n\n"
            + retrieved_rules_text
            + "\n\n"
            + "END OF RULES.\n\n"
            + base_system_prompt
        )
    else:
        system_message = base_system_prompt

    final_messages = [
        {"role": "system", "content": system_message},
        messages[1],
    ]

    # ---------------------------
    # BUILD REQUEST BODY
    # ---------------------------
    request_body = {
        "model": model,
        "messages": final_messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    # Log request
    _write_log_blob(
        "requests",
        operation_cd,
        a_ymd,
        b_ymd,
        timestamp,
        {"event": "openai_request", "request": request_body},
    )

    # ---------------------------
    # CALL OPENAI
    # ---------------------------
    try:
        resp = client.chat.completions.create(**request_body)
    except Exception as exc:
        _write_log_blob(
            "responses",
            operation_cd,
            a_ymd,
            b_ymd,
            timestamp,
            {"event": "openai_call_error", "error": str(exc)},
        )
        return None

    # ---------------------------
    # EXTRACT RAW CONTENT
    # ---------------------------
    content = ""
    if resp.choices and resp.choices[0].message:
        content = (resp.choices[0].message.content or "").strip()

    _write_log_blob(
        "responses",
        operation_cd,
        a_ymd,
        b_ymd,
        timestamp,
        {"event": "openai_response", "raw_content": content},
    )

    if not content:
        return None

    # ---------------------------
    # PARSE JSON
    # ---------------------------
    try:
        return json.loads(content)
    except Exception as exc:
        log(f"[openai] parse error: {exc}")
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

    items = []
    for t in top10:
        section = t.get("section")
        name = t.get("name")

        src_rows = tables.get(section, [])
        src = next((r for r in src_rows if str(r.get("name","")).strip()==str(name).strip()), {})

        loads_A = _safe_float(src.get("loads_A", t.get("loads_A")))
        loads_B = _safe_float(src.get("loads_B", t.get("loads_B")))
        zero_A = loads_A <= 1e-6
        zero_B = loads_B <= 1e-6

        core_a = _safe_float(src.get("Core_OR_A"))
        core_b = _safe_float(src.get("Core_OR_B"))

        rpl_a = src.get("Revenue_per_Load_A")
        rpl_b = src.get("Revenue_per_Load_B")
        loh_a = src.get("LOH_A")
        loh_b = src.get("LOH_B")

        rpm_a = _rpm_from_metrics(rpl_a, loh_a)
        rpm_b = _rpm_from_metrics(rpl_b, loh_b)
        rpm_big, rpm_delta, rpm_pct = _rpm_flag(rpm_a, rpm_b)

        subA = bool(src.get("Core_OR_A_is_substituted")) or zero_A
        subB = bool(src.get("Core_OR_B_is_substituted")) or zero_B

        rpm_big = rpm_big and not (subA or subB)

        driver_hint = select_primary_driver(section, name, tables)
        comp = _safe_float(t.get("Composite", 0.0))
        arrow = "▲" if comp > 0 else ("▼" if comp < 0 else "→")

        items.append({
            "section": section,
            "name": name,
            "arrow": arrow,
            "loads_A": loads_A,
            "loads_B": loads_B,
            "zero_A": zero_A,
            "zero_B": zero_B,
            "Core_OR_A": core_a,
            "Core_OR_B": core_b,
            "Core_OR_A_pct": core_a * 100 if core_a is not None else None,
            "Core_OR_B_pct": core_b * 100 if core_b is not None else None,
            "Composite": comp,
            "Impact_D": t.get("Impact_D"),
            "Impact_S": t.get("Impact_S"),
            "driver_hint": driver_hint,
            "Core_OR_A_is_substituted": subA,
            "Core_OR_B_is_substituted": subB,
            "rpm_A": rpm_a,
            "rpm_B": rpm_b,
            "rpm_delta": rpm_delta,
            "rpm_pct": rpm_pct,
            "rpm_is_big_factor": bool(rpm_big),
        })

    labels = {
        "A": _period_label(meta, "A"),
        "B": _period_label(meta, "B"),
    }

    sys_prompt = system_prompt or read_prompt_text(PROMPT_PATH)

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
                            "arrow": { "type": "string" },
                            "headline": { "type": "string" },
                            "driver_detail": { "type": "string" }
                        },
                        "required": ["arrow", "headline", "driver_detail"]
                    },
                    "minItems": 10,
                    "maxItems": 10
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

    # -------------------------------
    # LLM CALL
    # -------------------------------
    data = _call_chat_completion(
        model,
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": json.dumps(user_payload, indent=2)},
        ],
        purpose="client_report",
        log_meta=meta,
    )

    if not isinstance(data, dict):
        raise RuntimeError("LLM returned invalid structure.")

    if "stories" not in data or not isinstance(data.get("stories"), list):
        raise RuntimeError("LLM did not return a valid 'stories' list.")

    # Enforce Scenario Rules v3 for driver_detail and Narrative Formatting v3 tone
    _normalize_driver_details(data, items)
    _rebuild_headlines(data, items)
    _sanitize_subjective_language(data)

    return {
        "highlights": data.get("highlights", []),
        "stories": data["stories"],
        "final_word": data.get("final_word", ""),
    }


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
    """Construct the internal executive summary directly from the summary payload."""

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
    """Return a deterministic internal summary derived from the raw data."""

    del model, system_prompt  # internal path no longer calls OpenAI
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

    client_doc = {
        "title": f"{meta.get('SCAC','CLIENT')} Performance Report",
        "meta": meta_client,
        "highlights": client_payload.get("highlights", []),
        "stories": client_payload.get("stories", []),
        "final_word": client_payload.get("final_word", ""),
        "top10": summary.get("top10", []),
    }
    exec_summary = build_internal_exec_summary(summary)
    tiny_summary = tiny_exec_summary(meta_internal, summary)
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
    default_model = os.environ.get("MMQB_DEFAULT_MODEL", "gpt-4.1")
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
