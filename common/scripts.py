"""Where the analysis scripts are: each in the folder of its method."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def script_path(name: str) -> Path:
    """The analysis script ``<name>.py``, in whichever method folder holds it."""
    matches = sorted(ROOT.glob(f"*/{name}.py"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{name}.py: {len(matches)} matches under {ROOT}")
    return matches[0]


def script_module(name: str) -> str:
    """The same script as a module name, for ``python -m`` from the repository root."""
    path = script_path(name)
    return f"{path.parent.name}.{path.stem}"
