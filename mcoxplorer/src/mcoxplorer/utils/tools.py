from __future__ import annotations

import shutil
from pathlib import Path


def resolve_tool_path(name_or_path: str) -> str | None:
    """Resolve executable path from configured value or PATH."""

    expanded = str(Path(name_or_path).expanduser())
    if Path(expanded).exists():
        return expanded
    return shutil.which(name_or_path)


def tool_available(name_or_path: str) -> bool:
    return resolve_tool_path(name_or_path) is not None
