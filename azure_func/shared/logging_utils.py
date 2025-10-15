# azure_func/shared/logging_utils.py

"""Logging helpers used by the Azure Function implementation."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

# Optional imports for nicer handling when present
try:  # NumPy scalars, arrays
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

try:  # Pandas Timestamp/NA/etc.
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None  # type: ignore


def _json_safe(obj: Any) -> Any:
    """
    Convert common non-JSON-serializable types to JSON-safe representations.
    This function is used by JsonFormatter to guarantee logging never crashes.
    """
    # Datetime-like
    if isinstance(obj, (datetime, )):
        # Always use UTC-style ISO if tz-aware, otherwise plain ISO
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (date, )):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (time, )):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, timedelta):
        return obj.total_seconds()

    # Numeric-like
    if isinstance(obj, Decimal):
        # Prefer float; if not finite, fallback to str
        f = float(obj)
        if f == float("inf") or f == float("-inf") or f != f:  # NaN check
            return str(obj)
        return f

    # UUID
    if isinstance(obj, UUID):
        return str(obj)

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # Bytes
    if isinstance(obj, (bytes, bytearray, memoryview)):
        # Avoid base64 bloat in logs; short preview instead
        try:
            # Try utf-8 first, fall back to hex preview
            s = bytes(obj).decode("utf-8")
            return s
        except Exception:
            return {"__bytes__": True, "len": len(obj)}

    # Sets
    if isinstance(obj, (set, frozenset)):
        return list(obj)

    # NumPy scalars / arrays
    if _np is not None:
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()

    # Pandas Timestamp / NA / Series / DataFrame (last two: summarize)
    if _pd is not None:
        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        # Avoid dumping whole DataFrames to logs; provide a concise summary
        if hasattr(_pd, "Series") and isinstance(obj, _pd.Series):  # type: ignore[attr-defined]
            return {"__pandas_series__": True, "shape": [int(obj.shape[0])]}
        if hasattr(_pd, "DataFrame") and isinstance(obj, _pd.DataFrame):  # type: ignore[attr-defined]
            # Only basic shape + first few columns
            cols = list(obj.columns[:10])
            return {
                "__pandas_dataframe__": True,
                "shape": [int(obj.shape[0]), int(obj.shape[1])],
                "columns_sample": cols,
            }

    # Mapping-like fallback: ensure keys are strings
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # Iterable fallback (tuples/lists/generators)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    # Last resort
    return str(obj)


def _json_dumps(payload: Dict[str, Any]) -> str:
    """
    Safe json.dumps that uses _json_safe for unknown types
    and never raises out of the logger.
    """
    try:
        return json.dumps(payload, ensure_ascii=False, default=_json_safe)
    except Exception as exc:  # pragma: no cover
        # Last-ditch: stringified payload
        try:
            return json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                    "level": payload.get("level", "INFO"),
                    "message": f"[logging-fallback] {payload.get('message', '')}",
                    "logger": payload.get("logger", "mmqb"),
                    "fallback_error": str(exc),
                },
                ensure_ascii=False,
            )
        except Exception:
            # Absolute last resort
            return f'{{"level":"ERROR","message":"logging serialization failure","logger":"mmqb"}}'


class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Attach exception info if present
        if record.exc_info:
            try:
                payload["exception"] = self.formatException(record.exc_info)
            except Exception:
                payload["exception"] = "unable to format exception"

        # Merge structured extras, if present
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            # Sanitize extras to avoid serialization failures
            try:
                payload.update(_json_safe(extra))
            except Exception:
                # If anything weird happens, fall back to str(extra)
                payload["extra"] = str(extra)

        return _json_dumps(payload)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    return handler


def get_json_logger(name: str = "mmqb") -> logging.Logger:
    """Return a module-level logger configured for JSON output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _build_handler()
        logger.addHandler(handler)
        level_name = os.environ.get("MMQB_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        logger.propagate = False
    return logger


def log_exception(logger: logging.Logger, message: str, **context: Any) -> None:
    """Helper to log an exception with structured context."""
    context.setdefault("extra", {})
    context["extra"].update({"event": "exception"})
    logger.exception(message, **context)
