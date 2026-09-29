"""Within-child and between-child decomposition of trial-level predictions.

The RT nRMSE of a benchmark pools two sources of variance: differences between children (mean RT
changes with age and maturation) and trial-to-trial variation within a child, which is the
cognitive-control state that H1 concerns. For predictions of held-out children this module
reports, with percentile intervals from resampling children:

* ``nrmse_within``: error of the child-centred predictions against the child-centred targets,
  divided by the pooled within-child SD. A model that is constant within each child (mean,
  demographics) scores exactly 1; values below 1 mean trial-level information.
* ``r_within``: pooled within-child correlation of centred predictions and targets.
* ``r_between``: correlation of child-mean predictions with child-mean targets.

Resampled children are weighted by their multiplicity, which reproduces concatenating duplicated
children without materialising the arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChildStats:
    sse: np.ndarray        # sum of squared differences of centred target and prediction
    sst: np.ndarray        # sum of squared centred targets
    spp: np.ndarray        # sum of squared centred predictions
    stp: np.ndarray        # sum of centred target x centred prediction
    mean_true: np.ndarray
    mean_pred: np.ndarray


def child_statistics(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray) -> ChildStats:
    frame = pd.DataFrame({"s": np.asarray(subjects), "t": np.asarray(y_true, dtype=float),
                          "p": np.asarray(y_pred, dtype=float)})
    grouped = frame.groupby("s", sort=True)
    frame["tc"] = frame["t"] - grouped["t"].transform("mean")
    frame["pc"] = frame["p"] - grouped["p"].transform("mean")
    frame["sse"] = (frame["tc"] - frame["pc"]) ** 2
    frame["sst"] = frame["tc"] ** 2
    frame["spp"] = frame["pc"] ** 2
    frame["stp"] = frame["tc"] * frame["pc"]
    sums = frame.groupby("s", sort=True)[["sse", "sst", "spp", "stp"]].sum()
    means = grouped[["t", "p"]].mean()
    return ChildStats(sums["sse"].to_numpy(), sums["sst"].to_numpy(), sums["spp"].to_numpy(),
                      sums["stp"].to_numpy(), means["t"].to_numpy(), means["p"].to_numpy())


def _metrics(stats: ChildStats, weights: np.ndarray) -> tuple[float, float, float]:
    sst = weights @ stats.sst
    nrmse_within = float(np.sqrt((weights @ stats.sse) / sst)) if sst > 0 else float("nan")
    denom = np.sqrt(sst * (weights @ stats.spp))
    r_within = float((weights @ stats.stp) / denom) if denom > 0 else float("nan")
    w = weights / weights.sum()
    mt = stats.mean_true - w @ stats.mean_true
    mp = stats.mean_pred - w @ stats.mean_pred
    denom_b = np.sqrt((w @ mt ** 2) * (w @ mp ** 2))
    r_between = float((w @ (mt * mp)) / denom_b) if denom_b > 0 else float("nan")
    return nrmse_within, r_within, r_between


def within_between(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray,
                   n_bootstrap: int = 2000, seed: int = 12345, alpha: float = 0.05) -> dict:
    """Estimates and child-resampling percentile intervals of the three components."""
    stats = child_statistics(y_true, y_pred, subjects)
    n = len(stats.sse)
    estimate = _metrics(stats, np.ones(n))
    rng = np.random.default_rng(seed)
    draws = np.empty((n_bootstrap, 3))
    for b in range(n_bootstrap):
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        draws[b] = _metrics(stats, weights)
    low = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    high = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    out = {"n_children": int(n)}
    for i, name in enumerate(("nrmse_within", "r_within", "r_between")):
        out[name] = estimate[i]
        out[f"{name}_ci_low"] = float(low[i])
        out[f"{name}_ci_high"] = float(high[i])
    return out


def decompose_predictions(predictions: pd.DataFrame, n_bootstrap: int = 2000,
                          seed: int = 12345) -> pd.DataFrame:
    """Decomposition for every RT model, window and scope in a benchmark predictions table."""
    rows = []
    rt = predictions[predictions["target"] == "rt_s"]
    for (window, model), part in rt.groupby(["window", "model"], sort=True):
        for scope, sub in (("cv", part[part["fold"] != -1]), ("holdout_R5", part[part["fold"] == -1])):
            if sub.empty:
                continue
            result = within_between(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                                    sub["subject"].to_numpy(), n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"window": window, "model": model, "scope": scope, **result})
    return pd.DataFrame(rows)
