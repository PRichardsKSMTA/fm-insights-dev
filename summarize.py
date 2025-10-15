#!/usr/bin/env python3
"""Thin wrapper that exposes the Azure Function summarize module at the repo root."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_AZURE_FUNC_DIR = Path(__file__).resolve().parent / "azure_func"
_AZURE_SUMMARIZE = _AZURE_FUNC_DIR / "summarize.py"

if not _AZURE_SUMMARIZE.exists():
    raise ImportError(
        "Expected summarize.py to exist at azure_func/summarize.py; the Azure Function "
        "module is the single source of truth for summarization logic."
    )

_spec = importlib.util.spec_from_file_location("azure_func_summarize", _AZURE_SUMMARIZE)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load azure_func/summarize.py module spec")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export the Azure module's public interface so `import summarize` continues to work.
for name, value in vars(_module).items():
    if name.startswith("__") and name not in {"__doc__", "__all__"}:
        continue
    globals()[name] = value

# Ensure subsequent imports receive the wrapped module instance.
sys.modules.setdefault(__name__, _module)


def main(*args: Any, **kwargs: Any) -> Any:
    """Delegate CLI invocations to the Azure Function implementation."""

    if not hasattr(_module, "main"):
        raise AttributeError("azure_func/summarize.py does not define a main() function")
    return _module.main(*args, **kwargs)


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    sys.exit(main())
