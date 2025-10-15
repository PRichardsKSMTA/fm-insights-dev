"""Bridge helpers for invoking :mod:`narrate` from the Azure Function."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

FUNC_ROOT = Path(__file__).resolve().parent.parent
if str(FUNC_ROOT) not in sys.path:
    sys.path.insert(0, str(FUNC_ROOT))

import narrate  # type: ignore  # pylint: disable=wrong-import-position


def generate_documents(
    summary: Dict[str, Any],
    model: str,
    system_prompt: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate client and internal documents from a summary payload."""

    meta = summary.get("meta", {})
    client_code = meta.get("client_code")
    if not client_code:
        summary = dict(summary)
        meta = dict(meta)
        meta.setdefault("client_code", meta.get("SCAC", "CLIENT"))
        summary["meta"] = meta

    if system_prompt:
        client_doc, internal_doc = narrate.build_docs(
            summary, model, system_prompt=system_prompt
        )
    else:
        client_doc, internal_doc = narrate.build_docs(summary, model)
    return client_doc, internal_doc
