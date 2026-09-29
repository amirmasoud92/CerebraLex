"""Benchmark metrics: normalised RMSE (challenge definition), R2, AUROC, subject bootstraps.

The subject bootstraps resample children with replacement. For nRMSE and AUROC they work on
per-child sufficient statistics weighted by each child's multiplicity in the draw, which gives the
same values as concatenating the resampled children's trials (same draws, same seed) without
materialising them; any other metric uses the direct loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE divided by the standard deviation of the true values (Aristimunha et al. 2025)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sd = y_true.std()
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / sd)


def r2_from_nrmse(value: float) -> float:
    return float(1.0 - value ** 2)


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(score, dtype=float)))


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_bootstrap: int

    def to_dict(self) -> dict[str, float | int]:
        return {"estimate": self.estimate, "ci_low": self.ci_low, "ci_high": self.ci_high,
                "n_bootstrap": self.n_bootstrap}


def _draw_counts(subjects: np.ndarray, n_bootstrap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Child index of every trial and the multiplicity of each child in each draw.

    The draws equal those of ``rng.choice(np.unique(subjects), size=n_children, replace=True)``.
    """
    _, inverse = np.unique(np.asarray(subjects), return_inverse=True)
    k = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    counts = np.empty((n_bootstrap, k))
    for b in range(n_bootstrap):
        counts[b] = np.bincount(rng.choice(k, size=k, replace=True), minlength=k)
    return inverse, counts


def _nrmse_draws(y_true: np.ndarray, preds: list[np.ndarray], inverse: np.ndarray,
                 counts: np.ndarray) -> list[np.ndarray]:
    k = counts.shape[1]
    y = np.asarray(y_true, dtype=float)
    n = np.bincount(inverse, minlength=k).astype(float)
    total = counts @ n
    mean = (counts @ np.bincount(inverse, weights=y, minlength=k)) / total
    var = (counts @ np.bincount(inverse, weights=y ** 2, minlength=k)) / total - mean ** 2
    sd = np.sqrt(np.clip(var, 0.0, None))
    out = []
    for pred in preds:
        sse = counts @ np.bincount(inverse, weights=(y - np.asarray(pred, dtype=float)) ** 2, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(sd > 0, np.sqrt(sse / total) / sd, np.nan))
    return out


def _auroc_draws(y_true: np.ndarray, score: np.ndarray, inverse: np.ndarray, counts: np.ndarray,
                 chunk: int = 50) -> np.ndarray:
    """Weighted Mann-Whitney AUROC per draw (ties count one half, as in roc_auc_score)."""
    from scipy import sparse

    k = counts.shape[1]
    positive = np.asarray(y_true, dtype=int) == 1
    _, group = np.unique(np.asarray(score, dtype=float), return_inverse=True)
    g = int(group.max()) + 1

    def child_by_group(mask: np.ndarray):
        return sparse.csr_matrix((np.ones(int(mask.sum())), (group[mask], inverse[mask])), shape=(g, k))

    pos_t, neg_t = child_by_group(positive), child_by_group(~positive)
    out = np.empty(len(counts))
    for start in range(0, len(counts), chunk):
        c = counts[start:start + chunk].T                      # k x b
        wp = np.asarray(pos_t @ c)                             # g x b
        wn = np.asarray(neg_t @ c)
        below = np.cumsum(wn, axis=0) - wn
        numerator = (wp * (below + 0.5 * wn)).sum(axis=0)
        denominator = wp.sum(axis=0) * wn.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:start + chunk] = np.where(denominator > 0, numerator / denominator, np.nan)
    return out


def _metric_draws(metric, y_true, preds, subjects, n_bootstrap, seed) -> list[np.ndarray] | None:
    if metric is not nrmse and metric is not auroc:
        return None
    inverse, counts = _draw_counts(subjects, n_bootstrap, seed)
    if metric is nrmse:
        return _nrmse_draws(y_true, preds, inverse, counts)
    return [_auroc_draws(y_true, pred, inverse, counts) for pred in preds]


def subject_bootstrap(metric, y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray,
                      n_bootstrap: int = 2000, seed: int = 12345,
                      alpha: float = 0.05) -> BootstrapResult:
    """Percentile CI of ``metric`` over resampled subjects (clusters), not trials."""
    fast = _metric_draws(metric, y_true, [y_pred], subjects, n_bootstrap, seed)
    if fast is not None:
        low, high = np.nanpercentile(fast[0], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return BootstrapResult(float(metric(y_true, y_pred)), float(low), float(high), n_bootstrap)
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    index_by_subject = {s: np.flatnonzero(subjects == s) for s in unique}
    estimate = float(metric(y_true, y_pred))
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_subject[s] for s in chosen])
        draws[b] = metric(y_true[idx], y_pred[idx])
    low, high = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(estimate, float(low), float(high), n_bootstrap)


def paired_subject_bootstrap(metric, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                             subjects: np.ndarray, n_bootstrap: int = 2000,
                             seed: int = 12345) -> BootstrapResult:
    """CI of metric(a) - metric(b) over resampled subjects, for baseline comparisons."""
    fast = _metric_draws(metric, y_true, [pred_a, pred_b], subjects, n_bootstrap, seed)
    if fast is not None:
        low, high = np.nanpercentile(fast[0] - fast[1], [2.5, 97.5])
        return BootstrapResult(float(metric(y_true, pred_a) - metric(y_true, pred_b)), float(low), float(high),
                               n_bootstrap)
    rng = np.random.default_rng(seed)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    index_by_subject = {s: np.flatnonzero(subjects == s) for s in unique}
    estimate = float(metric(y_true, pred_a) - metric(y_true, pred_b))
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_subject[s] for s in chosen])
        draws[b] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return BootstrapResult(estimate, float(low), float(high), n_bootstrap)


def per_fold_table(predictions: pd.DataFrame, target: str = "y_true",
                   pred: str = "y_pred") -> pd.DataFrame:
    rows = []
    for fold, group in predictions.groupby("fold"):
        value = nrmse(group[target].to_numpy(), group[pred].to_numpy())
        rows.append({"fold": fold, "n_trials": len(group), "n_subjects": group["subject"].nunique(),
                     "nrmse": value, "r2": r2_from_nrmse(value)})
    return pd.DataFrame(rows)
