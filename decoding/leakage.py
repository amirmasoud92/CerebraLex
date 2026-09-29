"""Programmatic leakage checks executed at every fit and a subject-fingerprinting probe."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class LeakageError(RuntimeError):
    """Raised when a subject, or a fitted transform, crosses the train/test boundary."""


class LeakageAuditor:
    @staticmethod
    def assert_disjoint(train_subjects: Iterable[str], test_subjects: Iterable[str]) -> None:
        overlap = set(train_subjects) & set(test_subjects)
        if overlap:
            raise LeakageError(f"{len(overlap)} subjects appear in both train and test: "
                               f"{sorted(overlap)[:5]}")

    @staticmethod
    def assert_rows_grouped(rows: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray,
                            subject_column: str = "subject") -> None:
        train = set(rows.loc[train_mask, subject_column])
        test = set(rows.loc[test_mask, subject_column])
        LeakageAuditor.assert_disjoint(train, test)
        if (train_mask & test_mask).any():
            raise LeakageError("rows assigned to both train and test")

    @staticmethod
    def assert_fit_subjects(fitted_subjects: Iterable[str], test_subjects: Iterable[str],
                            what: str) -> None:
        overlap = set(fitted_subjects) & set(test_subjects)
        if overlap:
            raise LeakageError(f"{what} was fitted on {len(overlap)} test subjects")

    @staticmethod
    def cross_block_fingerprint_accuracy(features: np.ndarray, subjects: np.ndarray,
                                         blocks: np.ndarray, n_components: int | None = 50,
                                         seed: int = 0) -> dict[str, float]:
        """Nearest-neighbour subject identification ignoring neighbours from the same block.

        Trials of one block share interpolated channels, ICA decisions and electrode state, so
        within-block identification partly measures the recording rather than the child. Only
        trials whose subject has at least one other block are scored.
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        x = StandardScaler().fit_transform(np.asarray(features, dtype=float))
        if n_components is not None and x.shape[1] > n_components:
            x = PCA(n_components=n_components, random_state=seed).fit_transform(x)
        subjects = np.asarray(subjects)
        blocks = np.asarray(blocks)
        frame = pd.DataFrame({"subject": subjects, "block": blocks})
        n_blocks = frame.groupby("subject")["block"].transform("nunique").to_numpy()
        block_size = int(frame.groupby("block").size().max())
        k = min(len(x), block_size + 1)
        _, idx = NearestNeighbors(n_neighbors=k).fit(x).kneighbors(x)
        hits, chances = [], []
        n = len(x)
        subject_counts = frame.groupby("subject").size()
        block_counts = frame.groupby("block").size()
        for i in np.flatnonzero(n_blocks > 1):
            candidates = [j for j in idx[i, 1:] if blocks[j] != blocks[i]]
            if not candidates:
                continue
            hits.append(subjects[candidates[0]] == subjects[i])
            other_same_subject = subject_counts[subjects[i]] - block_counts[blocks[i]]
            chances.append(other_same_subject / (n - block_counts[blocks[i]]))
        return {"accuracy": float(np.mean(hits)) if hits else float("nan"),
                "chance": float(np.mean(chances)) if chances else float("nan"),
                "n_scored": int(len(hits)), "n_subjects": int(frame["subject"].nunique())}

    @staticmethod
    def fingerprint_accuracy(features: np.ndarray, subjects: np.ndarray,
                             n_components: int | None = 50, seed: int = 0) -> dict[str, float]:
        """Leave-one-out nearest-neighbour subject identification on standardised features.

        Returns the identification accuracy and the chance level (mean over trials of
        1 / n_subjects). Values far above chance mean the representation carries identity
        information that a decoder could exploit if subjects were split at the trial level.
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        x = StandardScaler().fit_transform(np.asarray(features, dtype=float))
        if n_components is not None and x.shape[1] > n_components:
            x = PCA(n_components=n_components, random_state=seed).fit_transform(x)
        subjects = np.asarray(subjects)
        n = len(subjects)
        if n < 2:
            return {"accuracy": float("nan"), "chance": float("nan"), "n_samples": n}
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(x)
        _, idx = nn.kneighbors(x)
        neighbour = idx[:, 1]
        accuracy = float(np.mean(subjects[neighbour] == subjects))
        counts = pd.Series(subjects).value_counts()
        chance = float(np.mean([(counts[s] - 1) / (n - 1) for s in subjects]))
        return {"accuracy": accuracy, "chance": chance, "n_samples": int(n),
                "n_subjects": int(counts.size)}
