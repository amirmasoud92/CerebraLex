"""Map the machine-specific local folder layout onto per-release BIDS datasets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from common.schema import RawDataConfig

_LOG = logging.getLogger(__name__)
_RELEASE_DIR_RE = re.compile(r"^(R\d+|NC)$")


@dataclass(frozen=True)
class BIDSDatasetLocation:
    release: str
    root: Path
    source_folder: str

    @property
    def has_dataset_description(self) -> bool:
        return (self.root / "dataset_description.json").is_file()

    @property
    def participants_tsv(self) -> Path:
        return self.root / "participants.tsv"

    @property
    def subject_dirs(self) -> list[Path]:
        return sorted(p for p in self.root.glob("sub-*") if p.is_dir())


class LocalHBNLayout:
    """Enumerate BIDS datasets present in the configured raw store."""

    def __init__(self, config: RawDataConfig) -> None:
        self.config = config

    def locate(self) -> list[BIDSDatasetLocation]:
        locations: list[BIDSDatasetLocation] = []
        for spec in self.config.bids_roots:
            root = self.config.resolve(spec)
            if not root.is_dir():
                _LOG.warning("configured raw folder missing: %s", root)
                continue
            if spec.layout == "single_release":
                assert spec.release is not None
                locations.append(BIDSDatasetLocation(spec.release, root, spec.path.as_posix()))
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir() and _RELEASE_DIR_RE.match(child.name):
                    if child.name not in self.config.releases:
                        _LOG.warning("release folder %s not in configured releases; skipped", child)
                        continue
                    locations.append(BIDSDatasetLocation(child.name, child, spec.path.as_posix()))
        return locations
