"""Compatibility layer re-exporting the refactored uclip package."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add the `src/` directory for editable installs if needed."""

    try:
        import_module("uclip")
        return
    except ModuleNotFoundError:
        pass

    src_dir = Path(__file__).resolve().parent.parent / "src"
    if src_dir.exists():
        sys.path.append(str(src_dir))


_ensure_src_on_path()

from uclip import analysis, cli, core, viz  # noqa: E402

__all__ = ["analysis", "cli", "core", "viz"]
