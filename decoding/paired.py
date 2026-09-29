"""Paired comparison of two benchmark prediction tables on their shared trials.

Used for the implementation check of EEGNet (two runs) and for the sensitivity variants
against the primary run. Both tables come from the frozen split, so a trial has the same fold in
both; the comparison is restricted to trials present in both, and the true values must agree.
The difference is metric(a) - metric(b): for nRMSE a negative value means a is better, for AUROC a
positive value means a is better. Intervals resample children (clusters), as everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decoding.metrics import auroc, nrmse, paired_subject_bootstrap

KEYS = ["window", "target", "model", "subject", "block_key", "trial_index"]
REGRESSION_TARGETS = ("rt_s",)


def compare_predictions(a: pd.DataFrame, b: pd.DataFrame, n_bootstrap: int = 2000,
                        seed: int = 12345) -> pd.DataFrame:
    columns = [*KEYS, "fold", "y_true", "y_pred"]
    merged = a[columns].merge(b[columns], on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    if not np.allclose(merged["y_true_a"], merged["y_true_b"]):
        raise ValueError("the two runs disagree on true values of shared trials")
    if (merged["fold_a"] != merged["fold_b"]).any():
        raise ValueError("the two runs place shared trials in different folds")
    counts_a = a.groupby(["window", "target", "model"]).size()
    counts_b = b.groupby(["window", "target", "model"]).size()
    rows = []
    for (window, target, model), part in merged.groupby(["window", "target", "model"], sort=True):
        metric = nrmse if target in REGRESSION_TARGETS else auroc
        for scope, sub in (("cv", part[part["fold_a"] != -1]), ("holdout_R5", part[part["fold_a"] == -1])):
            if sub.empty:
                continue
            y = sub["y_true_a"].to_numpy(float)
            pa, pb = sub["y_pred_a"].to_numpy(float), sub["y_pred_b"].to_numpy(float)
            result = paired_subject_bootstrap(metric, y, pa, pb, sub["subject"].to_numpy(),
                                              n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"window": window, "target": target, "model": model, "scope": scope,
                         "metric": metric.__name__, "n_shared_trials": len(sub),
                         "n_children": sub["subject"].nunique(),
                         "n_trials_a": int(counts_a.get((window, target, model), 0)),
                         "n_trials_b": int(counts_b.get((window, target, model), 0)),
                         "metric_a": metric(y, pa), "metric_b": metric(y, pb),
                         "delta": result.estimate, "delta_ci_low": result.ci_low,
                         "delta_ci_high": result.ci_high,
                         "r_predictions": float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 0 and pb.std() > 0
                         else float("nan")})
    return pd.DataFrame(rows)
