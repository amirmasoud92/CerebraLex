"""Code provenance recorded with every output."""

from __future__ import annotations

import subprocess
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def git_commit(root: Path | None = None) -> str:
    """Short commit of HEAD, suffixed ``-dirty`` when tracked files differ from it.

    Returns ``unknown`` outside a git checkout, for example from an installed wheel.
    """
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7", "--exclude", "*"],
                             cwd=root or _PACKAGE_ROOT, capture_output=True, text=True, check=False,
                             timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
