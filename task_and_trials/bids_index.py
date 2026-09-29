"""Index every EEG run in the local store as a flat table of file locations."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.io import OnExists, read_dataframe, write_dataframe
from task_and_trials.layout import BIDSDatasetLocation
from task_and_trials.releases import get_release

_LOG = logging.getLogger(__name__)
RUN_FILENAME_RE = re.compile(
    r"^sub-(?P<subject>[A-Za-z0-9]+)_task-(?P<task>[A-Za-z0-9]+)(?:_run-(?P<run>\d+))?_eeg\.set$"
)
_EEG_SUFFIX = "_eeg.set"


@dataclass(frozen=True)
class RunRecord:
    subject: str
    release: str
    task: str
    run: int | None
    source_folder: str
    bids_root: Path
    eeg_path: Path

    @property
    def run_key(self) -> str:
        run = "na" if self.run is None else str(self.run)
        return f"{self.subject}_{self.release}_{self.task}_run-{run}"

    @property
    def license(self) -> str:
        return get_release(self.release).license.value

    def sidecar(self, suffix: str) -> Path:
        base = self.eeg_path.name[: -len(_EEG_SUFFIX)]
        return self.eeg_path.with_name(base + suffix)

    @property
    def eeg_json(self) -> Path:
        return self.sidecar("_eeg.json")

    @property
    def events_tsv(self) -> Path:
        return self.sidecar("_events.tsv")

    @property
    def channels_tsv(self) -> Path:
        return self.sidecar("_channels.tsv")

    @property
    def electrodes_tsv(self) -> Path:
        return self.sidecar("_electrodes.tsv")

    @property
    def coordsystem_json(self) -> Path:
        return self.sidecar("_coordsystem.json")


class BIDSRunIndex:
    """Immutable collection of :class:`RunRecord` with filtering and table export."""

    def __init__(self, records: Sequence[RunRecord]) -> None:
        self._records = tuple(records)

    @classmethod
    def scan(cls, locations: Iterable[BIDSDatasetLocation]) -> BIDSRunIndex:
        records: list[RunRecord] = []
        for location in locations:
            n_before = len(records)
            for subject_dir in location.subject_dirs:
                eeg_dir = subject_dir / "eeg"
                if not eeg_dir.is_dir():
                    _LOG.warning("no eeg/ directory in %s", subject_dir)
                    continue
                for eeg_file in sorted(eeg_dir.glob(f"*{_EEG_SUFFIX}")):
                    match = RUN_FILENAME_RE.match(eeg_file.name)
                    if match is None:
                        _LOG.warning("unparseable EEG filename skipped: %s", eeg_file)
                        continue
                    subject = match["subject"]
                    if subject_dir.name != f"sub-{subject}":
                        _LOG.warning("subject folder/file mismatch: %s", eeg_file)
                    run = int(match["run"]) if match["run"] else None
                    records.append(RunRecord(
                        subject=subject,
                        release=location.release,
                        task=match["task"],
                        run=run,
                        source_folder=location.source_folder,
                        bids_root=location.root,
                        eeg_path=eeg_file,
                    ))
            _LOG.info(
                "%s/%s: %d runs", location.source_folder, location.release, len(records) - n_before
            )
        return cls(records)

    @property
    def records(self) -> tuple[RunRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._records)

    @property
    def subjects(self) -> list[str]:
        return sorted({r.subject for r in self._records})

    @property
    def tasks(self) -> list[str]:
        return sorted({r.task for r in self._records})

    @property
    def releases(self) -> list[str]:
        return sorted({r.release for r in self._records})

    def filter(
        self,
        task: str | None = None,
        release: str | None = None,
        subjects: Iterable[str] | None = None,
    ) -> BIDSRunIndex:
        subject_set = set(subjects) if subjects is not None else None
        kept = [
            r for r in self._records
            if (task is None or r.task == task)
            and (release is None or r.release == release)
            and (subject_set is None or r.subject in subject_set)
        ]
        return BIDSRunIndex(kept)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{
            "subject": r.subject, "release": r.release, "task": r.task, "run": r.run,
            "source_folder": r.source_folder, "bids_root": str(r.bids_root),
            "eeg_path": str(r.eeg_path), "license": r.license,
        } for r in self._records]
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> BIDSRunIndex:
        records = [RunRecord(
            subject=row.subject, release=row.release, task=row.task,
            run=None if pd.isna(row.run) else int(row.run),
            source_folder=row.source_folder, bids_root=Path(row.bids_root),
            eeg_path=Path(row.eeg_path),
        ) for row in frame.itertuples(index=False)]
        return cls(records)

    def save(self, path: Path, on_exists: OnExists = "error") -> Path:
        return write_dataframe(self.to_dataframe(), path, on_exists=on_exists)

    @classmethod
    def load(cls, path: Path) -> BIDSRunIndex:
        return cls.from_dataframe(read_dataframe(path))
