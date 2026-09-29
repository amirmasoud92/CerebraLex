"""Frozen, hashed, subject-grouped cross-validation folds.

Each subject is assigned exactly one fold from a salted hash of its identifier, balanced within
strata of release and age band. Held-out releases (R5 by challenge convention) receive fold -1
and never enter cross-validation. The assignment is serialised to JSON with a content hash so
that every downstream result can cite the exact split.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common.hashing import sha256_text, stable_subject_hash
from common.io import write_json

HOLDOUT_FOLD = -1
DEFAULT_AGE_EDGES = (0.0, 8.0, 11.0, 14.0, 100.0)


@dataclass(frozen=True)
class SplitSpec:
    n_folds: int = 10
    salt: str = "cerebralex-split-v1"
    holdout_releases: tuple[str, ...] = ("R5",)
    age_edges: tuple[float, ...] = DEFAULT_AGE_EDGES

    def to_dict(self) -> dict:
        return {"n_folds": self.n_folds, "salt": self.salt,
                "holdout_releases": list(self.holdout_releases), "age_edges": list(self.age_edges)}


@dataclass
class SplitAssignment:
    spec: SplitSpec
    table: pd.DataFrame            # subject, release, age_band, subject_hash, fold
    fingerprint: str = field(default="")

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        payload = self.table.sort_values("subject")[["subject", "fold"]].to_csv(index=False)
        return sha256_text(json.dumps(self.spec.to_dict(), sort_keys=True) + payload)

    @property
    def cv_subjects(self) -> pd.DataFrame:
        return self.table[self.table["fold"] != HOLDOUT_FOLD]

    @property
    def holdout_subjects(self) -> list[str]:
        return sorted(self.table.loc[self.table["fold"] == HOLDOUT_FOLD, "subject"])

    def fold_of(self, subjects: pd.Series) -> pd.Series:
        mapping = self.table.set_index("subject")["fold"]
        return subjects.map(mapping)

    def iter_folds(self) -> Iterator[tuple[int, list[str], list[str]]]:
        cv = self.cv_subjects
        for fold in sorted(cv["fold"].unique()):
            test = sorted(cv.loc[cv["fold"] == fold, "subject"])
            train = sorted(cv.loc[cv["fold"] != fold, "subject"])
            yield int(fold), train, test

    def save(self, path: Path) -> Path:
        payload = {
            "fingerprint": self.fingerprint,
            "spec": self.spec.to_dict(),
            "n_subjects": int(len(self.table)),
            "assignment": self.table.sort_values("subject").to_dict(orient="records"),
        }
        return write_json(payload, path, on_exists="error")

    @classmethod
    def load(cls, path: Path) -> SplitAssignment:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        spec_dict = payload["spec"]
        spec = SplitSpec(n_folds=spec_dict["n_folds"], salt=spec_dict["salt"],
                         holdout_releases=tuple(spec_dict["holdout_releases"]),
                         age_edges=tuple(spec_dict["age_edges"]))
        table = pd.DataFrame(payload["assignment"])
        assignment = cls(spec, table, fingerprint=payload["fingerprint"])
        if assignment._compute_fingerprint() != payload["fingerprint"]:
            raise ValueError(f"split file {path} does not match its fingerprint")
        return assignment


class SubjectGroupedSplitter:
    def __init__(self, spec: SplitSpec | None = None) -> None:
        self.spec = spec or SplitSpec()

    def assign(self, subjects: pd.DataFrame) -> SplitAssignment:
        """``subjects`` needs columns subject, release, age (age may be missing)."""
        required = {"subject", "release", "age"}
        missing = required - set(subjects.columns)
        if missing:
            raise ValueError(f"subjects table lacks {sorted(missing)}")
        frame = subjects.drop_duplicates("subject").copy()
        if frame["subject"].duplicated().any():
            raise ValueError("duplicate subject identifiers")
        edges = np.asarray(self.spec.age_edges)
        ages = pd.to_numeric(frame["age"], errors="coerce")
        frame["age_band"] = pd.cut(ages, bins=edges, labels=False, right=False).astype("Int64")
        frame["age_band"] = frame["age_band"].astype(object).where(ages.notna(), "unknown").astype(str)
        frame["subject_hash"] = frame["subject"].map(
            lambda s: stable_subject_hash(str(s), self.spec.salt))
        frame["fold"] = HOLDOUT_FOLD
        cv_mask = ~frame["release"].isin(self.spec.holdout_releases)
        for key, group in frame[cv_mask].groupby(["release", "age_band"], sort=True):
            ordered = group.sort_values("subject_hash").index
            folds = np.arange(len(ordered)) % self.spec.n_folds
            offset = int(sha256_text(f"{self.spec.salt}:{key}")[:8], 16) % self.spec.n_folds
            frame.loc[ordered, "fold"] = (folds + offset) % self.spec.n_folds
        frame["fold"] = frame["fold"].astype(int)
        columns = ["subject", "release", "age", "age_band", "subject_hash", "fold"]
        return SplitAssignment(self.spec, frame[columns].reset_index(drop=True))
