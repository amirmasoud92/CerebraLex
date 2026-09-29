"""Discovery of the project root, its standard output directories and the data locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ROOT_ENV = "CEREBRALEX_ROOT"

# Derivative roots of the samples, relative to the repository root unless absolute; each can be moved with its
# environment variable.
DATA_LOCATIONS = {
    "HBN": ("CEREBRALEX_DERIVATIVES_ROOT", "derivatives/hbn"),
    "MIPDB": ("CEREBRALEX_MIPDB_ROOT", "derivatives/mipdb"),
    "R10R11": ("CEREBRALEX_R10R11_ROOT", "derivatives/r10_r11"),
    "SYS": ("CEREBRALEX_SYS_ROOT", "derivatives/symbol_search"),
    "SYS_R10R11": ("CEREBRALEX_SYS_R10R11_ROOT", "derivatives/symbol_search_r10_r11"),
}


def data_location(key: str) -> str:
    """Derivative root of a sample: its environment variable, else the default above."""
    variable, default = DATA_LOCATIONS[key]
    return os.environ.get(variable, default)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls) -> ProjectPaths:
        """The repository root: ``CEREBRALEX_ROOT`` if set, else the folder that holds ``common/``."""
        env_root = os.environ.get(_ROOT_ENV)
        return cls(Path(env_root).resolve() if env_root else Path(__file__).resolve().parents[1])

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    @property
    def derivatives(self) -> Path:
        return self.root / "derivatives"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def ensure(self) -> None:
        for directory in (
            self.audit, self.derivatives, self.results, self.figures, self.tables,
            self.logs, self.reports, self.models,
        ):
            directory.mkdir(parents=True, exist_ok=True)
