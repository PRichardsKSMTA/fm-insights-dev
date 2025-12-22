"""HTTP-triggered Azure Function that runs the MMQB pipeline."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from requests import RequestException

import azure.functions as func

from ..shared import (
    DatabaseClient,
    DatabaseError,
    ValidationError,
    generate_documents,
    generate_summary_artifacts,
    ensure_stored_procedure_env,
    get_json_logger,
    log_exception,
    render_client,
    render_internal,
)

LOGGER = get_json_logger("mmqb.HttpMmqbRun")
DEFAULT_MODEL_ENV = "MMQB_DEFAULT_MODEL"
UPDATE_STATUS_PROC_ENV = "MMQB_UPDATE_STATUS_PROC"
NO_DATA_FLOW_URL_ENV = "MMQB_NO_DATA_FLOW_URL"

PROC_DATA_CURRENT = "data_current"
PROC_PROMPT_TEXT = "prompt_text"
PROC_SAVE = "save"

CSV_CANDIDATES = ["csv_payload", "CsvPayload", "csv", "Csv", "csv_text", "CsvText", "payload"]
CLIENT_VIEW_SUFFIX_RE = re.compile(r"\s*[—-]\s*Client View\s*$")


class StoredProcedureUnavailableError(RuntimeError):
    """Raised when a required stored procedure cannot be executed."""

    def __init__(self, proc_name: str, message: str | None = None) -> None:
        super().__init__(message or f"Stored procedure unavailable: {proc_name}")
        self.proc_name = proc_name


def _ensure_json_response(payload: Dict[str, Any], status: int) -> func.HttpResponse:
    return func.HttpResponse(body=json.dumps(payload), status_code=status, mimetype="application/json")


def _sanitize_client_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *doc* with any client-view suffix removed from its title."""

    sanitized = dict(doc)
    title = sanitized.get("title")
    if isinstance(title, str):
        sanitized_title = CLIENT_VIEW_SUFFIX_RE.sub("", title).strip()
        sanitized["title"] = sanitized_title or title
    return sanitized


def _extract_csv_payload(row: Dict[str, Any]) -> Optional[bytes]:
    for key in CSV_CANDIDATES:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if isinstance(value, bytes):
            return value
        text = str(value).strip()
        if text:
            return text.encode("utf-8")
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("No rows available to create CSV payload")
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({col: _stringify(row.get(col)) for col in columns})
    return buffer.getvalue().encode("utf-8")


def _decode_csv_rows(csv_bytes: bytes) -> Tuple[List[Dict[str, Any]], List[str]]:
    text = csv_bytes.decode("utf-8")
    buffer = io.StringIO(text)
    reader = csv.DictReader(buffer)
    rows = [dict(row) for row in reader]
    columns = list(reader.fieldnames or [])
    return rows, columns


def _load_dataset(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    LOGGER.info("Normalizing dataset rows", extra={"event": "load_dataset_start", "rows_in": len(rows)})
    if not rows:
        return [], []
    if len(rows) == 1:
        csv_bytes = _extract_csv_payload(rows[0])
        if csv_bytes:
            LOGGER.info("CSV payload detected; decoding", extra={"event": "csv_decode"})
            return _decode_csv_rows(csv_bytes)
    columns: List[str] = []
    seen = set()
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        normalized_row = {key: row.get(key) for key in row.keys()}
        normalized.append(normalized_row)
        for key in normalized_row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)
    LOGGER.info(
        "Dataset normalized",
        extra={"event": "load_dataset_ok", "rows_out": len(normalized), "column_count": len(columns), "columns_sample": columns[:50]},
    )
    return normalized, columns


def _resolve_model() -> str:
    model = os.environ.get(DEFAULT_MODEL_ENV, "gpt-4.1")
    LOGGER.info("Model resolved", extra={"event": "model_resolved", "model": model})
    return model


def _update_summary_meta(
    summary: Dict[str, Any],
    meta_source: Optional[Dict[str, Any]],
    *,
    operation_cd: str,
    upload_id: str,
    prompt_text: Optional[str],
) -> Dict[str, Any]:
    """Return a summary with metadata populated for downstream consumers."""
    summary = dict(summary)
    meta = dict(summary.get("meta", {}))
    if meta_source:
        scac = (
            meta_source.get("client_code")
            or meta_source.get("ClientCode")
            or meta_source.get("SCAC")
            or meta_source.get("Scac")
            or meta.get("SCAC")
            or "CLIENT"
        )
        meta.setdefault("client_code", meta_source.get("ClientCode") or meta_source.get("client_code"))
        meta.setdefault("SCAC", scac)
    meta["operation_cd"] = operation_cd
    meta["upload_id"] = upload_id
    if prompt_text:
        meta["operation_prompt"] = prompt_text
    summary["meta"] = meta
    LOGGER.info("Summary meta updated", extra={"event": "summary_meta", "operation_cd": operation_cd, "upload_id": upload_id})
    return summary


def _call_procedure(db: DatabaseClient, proc_name: str, params: Sequence[Any] | None = None) -> List[Dict[str, Any]]:
    try:
        return db.call_procedure(proc_name, params)
    except DatabaseError as exc:  # pragma: no cover - requires live DB
        LOGGER.error("Stored procedure unavailable", extra={"event": "missing_proc", "proc": proc_name})
        raise StoredProcedureUnavailableError(proc_name, str(exc)) from exc


def _call_procedure_no_results(db: DatabaseClient, proc_name: str, params: Sequence[Any] | None = None) -> None:
    try:
        db.call_procedure_no_results(proc_name, params)
    except DatabaseError as exc:  # pragma: no cover - requires live DB
        LOGGER.error("Stored procedure unavailable", extra={"event": "missing_proc", "proc": proc_name})
        raise StoredProcedureUnavailableError(proc_name, str(exc)) from exc


def _normalize_operation_cd(data: Dict[str, Any]) -> str:
    operation = str(data.get("OPERATION_CD") or data.get("operation_cd") or data.get("operation") or "").strip()
    if not operation:
        raise ValidationError("'OPERATION_CD' is required.")
    return operation


def _normalize_upload_id(data: Dict[str, Any]) -> str:
    upload = str(data.get("UPLOAD_ID") or data.get("upload_id") or data.get("upload") or "").strip()
    if not upload:
        raise ValidationError("'UPLOAD_ID' is required.")
    match = re.fullmatch(r"(?P<date>\d{8})([A-Za-z0-9_-]*)", upload)
    if not match:
        raise ValidationError("'UPLOAD_ID' must start with YYYYMMDD and may include only letters, numbers, underscores, or hyphens.")
    date_str = match.group("date")
    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError as exc:
        raise ValidationError("'UPLOAD_ID' must start with a valid YYYYMMDD date.") from exc
    return upload


def _normalize_week_ct(data: Dict[str, Any]) -> Optional[int]:
    """
    Normalize optional WEEK_CT:

    - Accepts WEEK_CT / week_ct / Weeks / weeks
    - Must be a positive integer if provided
    - Returns None when omitted so we can keep current behavior
      (proc called with only OPERATION_CD).
    """
    raw = (
        data.get("WEEK_CT")
        or data.get("week_ct")
        or data.get("Weeks")
        or data.get("weeks")
    )

    if raw is None or raw == "":
        return None

    try:
        week_ct = int(raw)
    except (TypeError, ValueError):
        raise ValidationError("'WEEK_CT' must be an integer if provided.")

    if week_ct <= 0:
        raise ValidationError("'WEEK_CT' must be a positive integer if provided.")

    return week_ct


def _fetch_dataset(
    db: DatabaseClient,
    proc_name: str,
    params: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    LOGGER.info(
        f"Fetching dataset via {proc_name}",
        extra={"event": "data_fetch_start", "proc": proc_name, "param_preview": [str(p) for p in params]},
    )
    rows = _call_procedure(db, proc_name, params)
    if rows:
        sample = {k: (str(v)[:120] + "…") if v is not None and len(str(v)) > 120 else v for k, v in list(rows[0].items())[:25]}
        LOGGER.info(
            "Dataset sample row",
            extra={"event": "data_sample", "proc": proc_name, "sample": sample, "row_count": len(rows)},
        )
    LOGGER.info(
        f"Dataset fetched rows={len(rows)}",
        extra={"event": "data_fetch_ok", "proc": proc_name, "row_count": len(rows)},
    )
    return _load_dataset(rows)


def _complete_operation_without_data(
    db: DatabaseClient,
    *,
    operation_cd: str,
    upload_id: str,
    data_proc_name: str,
) -> None:
    message = f"Stored procedure {data_proc_name} completed. No dataset rows returned from proc."

    # update_proc = os.environ.get(UPDATE_STATUS_PROC_ENV, "dbo.UPDATE_CLIENT_UPLOAD_OPERATION_STATUS")
    # try:
    #     _call_procedure_no_results(db, update_proc, [operation_cd, "MMQB-COMPLETE"])
    #     LOGGER.info(
    #         "Operation marked complete with no data",
    #         extra={
    #             "event": "operation_marked_complete",
    #             "operation_cd": operation_cd,
    #             "upload_id": upload_id,
    #             "update_proc": update_proc,
    #         },
    #     )
    # except DatabaseError as exc:
    #     log_exception(
    #         LOGGER,
    #         "Failed to mark operation complete after no-data result",
    #         extra={
    #             "event": "operation_mark_complete_failed",
    #             "operation_cd": operation_cd,
    #             "upload_id": upload_id,
    #             "update_proc": update_proc,
    #         },
    #     )
    #     LOGGER.error("Operation completion update failed", extra={"error": str(exc)})

    flow_url = os.environ.get(
        NO_DATA_FLOW_URL_ENV,
        "https://defaulta5b5103d48344aa580dba1fed2b830.80.environment.api.powerplatform.com:443/powerautomate/automations/direct/workflows/6726239554044079b1ea25754a298b80/triggers/manual/paths/invoke?api-version=1&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=lEjm45iJsfijqzsfoM6ww9H9ErL4jNx4jK1KqlxbLsg",
    )

    if not flow_url:
        LOGGER.error(
            "No Power Automate flow URL configured",
            extra={
                "event": "no_flow_url",
                "operation_cd": operation_cd,
                "upload_id": upload_id,
            },
        )
        return

    payload = {
        "SUBJECT": "MMQB - No Data Found",
        "OPERATION_CD": operation_cd,
        "UPLOAD_ID": upload_id,
        "ERROR_MESSAGE": message,
        "INFO_MESSAGE": "MMQB will NOT be calculated for this operation unless manual actions are taken."
    }

    try:
        response = requests.post(flow_url, json=payload, timeout=15)
        response.raise_for_status()
        LOGGER.info(
            "Power Automate notification sent",
            extra={
                "event": "no_data_notification_sent",
                "operation_cd": operation_cd,
                "upload_id": upload_id,
                "status_code": response.status_code,
            },
        )
    except RequestException as exc:
        log_exception(
            LOGGER,
            "Failed to send Power Automate notification",
            extra={
                "event": "no_data_notification_failed",
                "operation_cd": operation_cd,
                "upload_id": upload_id,
                "flow_url": flow_url,
            },
        )
        LOGGER.error("Notification request failed", extra={"error": str(exc)})


def _fetch_prompt_text(db: DatabaseClient, proc_name: str, operation_cd: str) -> Optional[str]:
    LOGGER.info("Fetching prompt text", extra={"event": "prompt_fetch_start", "proc": proc_name, "operation_cd": operation_cd})
    rows = _call_procedure(db, proc_name, [operation_cd])
    if not rows:
        LOGGER.warning("Prompt proc returned no rows", extra={"event": "prompt_fetch_empty", "proc": proc_name})
        return None
    row = rows[0]
    for key in ("PROMPT_TEXT", "prompt_text", "PROMPT", "PromptText"):
        value = row.get(key)
        if value:
            LOGGER.info("Prompt text found", extra={"event": "prompt_fetch_ok", "length": len(str(value))})
            return str(value)
    value = next(iter(row.values()), None)
    LOGGER.info("Prompt text fallback column used", extra={"event": "prompt_fetch_fallback", "has_value": bool(value)})
    return str(value) if value else None


def _fetch_html_skeleton(
    db: DatabaseClient,
    proc_name: str,
    operation_cd: str,
) -> Optional[str]:
    """
    Call the HTML skeleton stored procedure and return the template text
    from the MMQB_HTML_SKELETON_TEXT column (nvarchar(max)).

    Returns None if no row or no template is present.
    """
    LOGGER.info(
        "Fetching HTML skeleton",
        extra={
            "event": "html_skeleton_fetch_start",
            "proc": proc_name,
            "operation_cd": operation_cd,
        },
    )

    rows = _call_procedure(db, proc_name, [operation_cd])
    if not rows:
        LOGGER.warning(
            "HTML skeleton proc returned no rows",
            extra={
                "event": "html_skeleton_fetch_empty",
                "proc": proc_name,
                "operation_cd": operation_cd,
            },
        )
        return None

    row = rows[0]
    value = row.get("MMQB_HTML_SKELETON_TEXT")
    if not value:
        LOGGER.warning(
            "HTML skeleton column missing or empty",
            extra={
                "event": "html_skeleton_missing",
                "proc": proc_name,
                "operation_cd": operation_cd,
                "keys": list(row.keys()),
            },
        )
        return None

    template = str(value)
    LOGGER.info(
        "HTML skeleton fetched",
        extra={
            "event": "html_skeleton_fetch_ok",
            "proc": proc_name,
            "operation_cd": operation_cd,
            "length": len(template),
        },
    )
    return template


def main(req: func.HttpRequest) -> func.HttpResponse:
    LOGGER.info("HttpMmqbRun triggered", extra={"event": "start"})
    try:
        raw = req.get_body()
        raw_len = 0
        if isinstance(raw, (bytes, bytearray)):
            raw_len = len(raw)
        else:
            try:
                raw_len = len(raw)
            except TypeError:
                raw_len = 0
        LOGGER.info("Request body received", extra={"event": "request_body", "bytes": raw_len})
        data = req.get_json()
    except ValueError:  # pragma: no cover - azure runtime handles validation
        return _ensure_json_response({"error": "Request body must be valid JSON."}, 400)

    if not isinstance(data, dict):
        return _ensure_json_response({"error": "Request JSON must be an object."}, 400)

    try:
        operation_cd = _normalize_operation_cd(data)
        upload_id = _normalize_upload_id(data)
        week_ct = _normalize_week_ct(data)
    except ValidationError as exc:
        LOGGER.warning("Invalid request payload", extra={"event": "request_invalid", "error": str(exc)})
        return _ensure_json_response({"error": str(exc)}, 400)

    log_context = {"operation_cd": operation_cd, "upload_id": upload_id, "week_ct": week_ct}
    LOGGER.info("Request received", extra={"event": "request_received", **log_context})

    try:
        procs = ensure_stored_procedure_env()
        LOGGER.info(
            f"Stored procedures resolved data_current={procs.get(PROC_DATA_CURRENT)} "
            f"prompt_text={procs.get(PROC_PROMPT_TEXT)} save={procs.get(PROC_SAVE)}",
            extra={
                "event": "proc_env_ok",
                "resolved": {
                    PROC_DATA_CURRENT: procs.get(PROC_DATA_CURRENT),
                    PROC_PROMPT_TEXT: procs.get(PROC_PROMPT_TEXT),
                    PROC_SAVE: procs.get(PROC_SAVE),
                },
            },
        )
    except RuntimeError as exc:
        LOGGER.error("Stored procedure configuration missing", extra={"event": "proc_env_missing", "error": str(exc), **log_context})
        return _ensure_json_response({"error": str(exc)}, 500)

    try:
        db = DatabaseClient()
    except DatabaseError as exc:
        log_exception(LOGGER, "Failed to initialize database client", extra=log_context)
        return _ensure_json_response({"error": str(exc)}, 500)

    try:
        db.introspect_procedure(procs[PROC_DATA_CURRENT])
    except Exception:
        pass

    try:
        # 1) Load single dataset (proc expects OPERATION_CD, and optionally WEEK_CT)
        data_proc = procs[PROC_DATA_CURRENT]

        if week_ct is None:
            data_params: List[Any] = [operation_cd]
        else:
            data_params = [operation_cd, week_ct]

        current_rows, columns = _fetch_dataset(db, data_proc, data_params)
        if not current_rows:
            LOGGER.warning("No dataset rows returned from proc", extra={"event": "no_data_from_proc", **log_context})
            # _complete_operation_without_data(
            #     db,
            #     operation_cd=operation_cd,
            #     upload_id=upload_id,
            #     data_proc_name=data_proc,
            # )
            return _ensure_json_response({"error": "No data available for supplied upload."}, 404)
        LOGGER.info(
            "Dataset loaded",
            extra={
                "event": "dataset_loaded",
                "current_rows": len(current_rows),
                "column_count": len(columns),
                "columns_sample": columns[:50],
                **log_context,
            },
        )

        # 2) Get prompt text and html template (optional but logged)
        prompt_text = _fetch_prompt_text(db, procs["prompt_text"], operation_cd)
        html_template = _fetch_html_skeleton(db, procs["html_skeleton"], operation_cd)

        # 3) Summarize (A/B derived from the single dataset)
        LOGGER.info("Summarization start", extra={"event": "summarize_start", **log_context})
        artifacts = generate_summary_artifacts(current_rows, operation_cd=operation_cd)
        meta_source = current_rows[0]
        summary = _update_summary_meta(artifacts.summary, meta_source, operation_cd=operation_cd, upload_id=upload_id, prompt_text=prompt_text)
        LOGGER.info("Summarization done", extra={"event": "summarize_done", **log_context})

        # 4) Narrate
        model = _resolve_model()
        LOGGER.info("Narration start", extra={"event": "narrate_start", "model": model, **log_context})
        client_doc, internal_doc = generate_documents(summary, model, prompt_text)
        client_doc = _sanitize_client_doc(client_doc)
        LOGGER.info(
            "Narration done",
            extra={
                "event": "narration_done",
                "client_len": len(json.dumps(client_doc)),
                "internal_len": len(json.dumps(internal_doc)),
                **log_context,
            },
        )

        # 5) Save
        internal_json = json.dumps(internal_doc)
        client_json = json.dumps(client_doc)

        client_html = render_client(copy.deepcopy(client_doc), template=html_template)
        internal_html = render_internal(copy.deepcopy(internal_doc))
        LOGGER.info(
            "HTML rendered",
            extra={
                "event": "html_rendered",
                "client_html_len": len(client_html),
                "internal_html_len": len(internal_html),
                **log_context,
            },
        )
        LOGGER.info("Saving results", extra={"event": "save_start", **log_context})
        _call_procedure_no_results(
            db,
            procs[PROC_SAVE],
            [
                operation_cd,
                upload_id,
                internal_json,
                client_json,
                internal_html,
                client_html,
            ],
        )
        LOGGER.info("Results saved", extra={"event": "save_done", **log_context})

        response = {"operation_cd": operation_cd, "upload_id": upload_id}
        LOGGER.info("Request complete", extra={"event": "done", **log_context})
        return _ensure_json_response(response, 200)

    except StoredProcedureUnavailableError as exc:
        LOGGER.error("Required stored procedure missing", extra={"event": "missing_proc", "proc": exc.proc_name, **log_context})
        return _ensure_json_response({"error": f"Stored procedure unavailable: {exc.proc_name}"}, 500)
    except DatabaseError as exc:  # pragma: no cover - requires live DB
        log_exception(LOGGER, "Database failure", extra=log_context)
        return _ensure_json_response({"error": str(exc)}, 500)
    except Exception as exc:  # pragma: no cover - runtime error path
        log_exception(LOGGER, "Pipeline execution failed", extra=log_context)
        return _ensure_json_response({"error": "Pipeline execution failed.", "details": str(exc)}, 500)
