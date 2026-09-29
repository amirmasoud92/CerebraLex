"""Lazy access to one BIDS EEG run: sidecars are cheap, the raw signal loads on demand."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd

from task_and_trials.bids_index import RunRecord

_NA_VALUES = ["n/a", "N/A", ""]


@dataclass(frozen=True)
class RecordingHeader:
    sampling_rate_hz: float
    n_channels: int
    n_samples: int
    duration_s: float
    channel_names: tuple[str, ...]
    file_size_mb: float
    n_annotations: int


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=_NA_VALUES, keep_default_na=False)


class EEGRecording:
    def __init__(self, record: RunRecord) -> None:
        self.record = record

    @cached_property
    def sidecar(self) -> dict:
        with open(self.record.eeg_json, encoding="utf-8") as handle:
            return json.load(handle)

    @cached_property
    def channels(self) -> pd.DataFrame:
        return _read_tsv(self.record.channels_tsv)

    @cached_property
    def electrodes(self) -> pd.DataFrame | None:
        path = self.record.electrodes_tsv
        return _read_tsv(path) if path.is_file() else None

    @cached_property
    def coordsystem(self) -> dict | None:
        path = self.record.coordsystem_json
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    @cached_property
    def events(self) -> pd.DataFrame:
        return _read_tsv(self.record.events_tsv)

    def read_header(self) -> RecordingHeader:
        """Open the EEGLAB file without keeping data in memory and report its header."""
        import mne

        raw = mne.io.read_raw_eeglab(self.record.eeg_path, preload=False, verbose="error")
        info = raw.info
        return RecordingHeader(
            sampling_rate_hz=float(info["sfreq"]),
            n_channels=len(raw.ch_names),
            n_samples=raw.n_times,
            duration_s=raw.n_times / float(info["sfreq"]),
            channel_names=tuple(raw.ch_names),
            file_size_mb=self.record.eeg_path.stat().st_size / 1e6,
            n_annotations=len(raw.annotations),
        )

    def load_raw(self, preload: bool = True):
        import mne

        return mne.io.read_raw_eeglab(self.record.eeg_path, preload=preload, verbose="error")
