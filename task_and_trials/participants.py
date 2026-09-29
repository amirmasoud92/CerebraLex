"""Participant metadata: demographics, bifactor psychopathology scores and availability flags."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

PHENOTYPE_COLUMNS = ("p_factor", "attention", "internalizing", "externalizing")
DEMOGRAPHIC_COLUMNS = ("sex", "age", "ehq_total")
AVAILABILITY_LEVELS = ("available", "caution", "unavailable")
_NA_VALUES = ["n/a", "N/A", "NaN", ""]


def availability_column(task: str, run: int | None) -> str:
    """Name of the participants.tsv column holding the availability flag of a run."""
    return task if run is None else f"{task}_{run}"


def strip_sub_prefix(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("^sub-", "", regex=True)


class ParticipantsTable:
    """Wrapper over one or several concatenated participants.tsv files."""

    def __init__(self, frame: pd.DataFrame) -> None:
        if "participant_id" not in frame.columns:
            raise ValueError("participants table lacks 'participant_id'")
        self._frame = frame.reset_index(drop=True)

    @classmethod
    def from_tsv(cls, path: Path, source_release: str) -> ParticipantsTable:
        frame = pd.read_csv(path, sep="\t", na_values=_NA_VALUES, keep_default_na=False)
        frame.insert(1, "source_release", source_release)
        frame.insert(2, "source_file", str(path))
        return cls(frame)

    @classmethod
    def concat(cls, tables: Iterable[ParticipantsTable]) -> ParticipantsTable:
        frames = [t.frame for t in tables]
        if not frames:
            return cls(pd.DataFrame({"participant_id": pd.Series(dtype=str)}))
        return cls(pd.concat(frames, ignore_index=True, sort=False))

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def __len__(self) -> int:
        return len(self._frame)

    @property
    def subject_ids(self) -> list[str]:
        return sorted(strip_sub_prefix(self._frame["participant_id"]).unique())

    @property
    def availability_columns(self) -> list[str]:
        columns = []
        for column in self._frame.columns:
            values = self._frame[column].dropna().astype(str).str.lower()
            if len(values) and values.isin(AVAILABILITY_LEVELS).all():
                columns.append(column)
        return columns

    def availability(self, task: str, run: int | None) -> pd.Series:
        column = availability_column(task, run)
        if column not in self._frame.columns:
            raise KeyError(f"no availability column '{column}'")
        series = self._frame[column].astype("string").str.lower()
        series.index = pd.Index(strip_sub_prefix(self._frame["participant_id"]))
        return series

    def duplicated_participants(self) -> pd.DataFrame:
        counts = self._frame.groupby("participant_id")["source_file"].nunique()
        dupes = counts[counts > 1].index
        return self._frame[self._frame["participant_id"].isin(dupes)].sort_values("participant_id")

    def phenotype_summary(self) -> pd.DataFrame:
        present = [
            c for c in (*DEMOGRAPHIC_COLUMNS, *PHENOTYPE_COLUMNS) if c in self._frame.columns
        ]
        numeric = self._frame[present].apply(pd.to_numeric, errors="coerce")
        summary = numeric.describe().T
        summary["n_missing"] = numeric.isna().sum()
        return summary
