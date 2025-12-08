"""Validation helpers for the Azure Function request pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import azure.functions as func


class ValidationError(ValueError):
    """Raised when the HTTP request payload is invalid."""


def parse_request_payload(req: func.HttpRequest) -> Dict[str, Any]:
    """Validate and return the normalized request payload."""

    try:
        data = req.get_json()
    except ValueError as exc:  # pragma: no cover - azure runtime handles validation
        raise ValidationError("Request body must be valid JSON.") from exc

    if not isinstance(data, dict):
        raise ValidationError("Request JSON must be an object.")

    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        raise ValidationError("'run_id' is required.")

    model = str(data.get("model") or "").strip() or None

    payload: Dict[str, Any] = {"run_id": run_id}
    if model:
        payload["model"] = model

    return payload


def ensure_stored_procedure_env() -> Dict[str, str]:
    """Ensure required stored procedure names are provided via environment variables."""

    required_envs = {
        "data_current": "MMQB_PROC_DATA_CURRENT",
        "prompt_text": "MMQB_PROC_PROMPT_TEXT",
        "html_skeleton": "MMQB_PROC_HTML_SKELETON",  # NEW
        "save": "MMQB_SAVE_PROC",
    }

    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for key, env_name in required_envs.items():
        value = os.environ.get(env_name)
        if not value:
            missing.append(env_name)
            continue
        override = os.environ.get(f"{env_name}_OVERRIDE")
        resolved[key] = override or value

    if missing:
        raise RuntimeError(
            "Missing stored procedure environment variables: " + ", ".join(sorted(missing))
        )

    return resolved
