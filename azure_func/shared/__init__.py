"""Shared utilities for the Azure Function app."""

from .logging_utils import get_json_logger, log_exception
from .db import DatabaseClient, DatabaseError
from .summarize_bridge import SummaryArtifacts, generate_summary_artifacts
from .narrate_bridge import generate_documents
from .html_renderer import render_client, render_internal, render_html
from .validators import (
    ValidationError,
    parse_request_payload,
    ensure_stored_procedure_env,
)

__all__ = [
    "get_json_logger",
    "log_exception",
    "DatabaseClient",
    "DatabaseError",
    "SummaryArtifacts",
    "generate_summary_artifacts",
    "generate_documents",
    "render_client",
    "render_internal",
    "render_html",
    "ValidationError",
    "parse_request_payload",
    "ensure_stored_procedure_env",
]
