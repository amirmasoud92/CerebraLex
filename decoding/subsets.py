"""Scores of every model restricted to test children from chosen releases.

The frozen REVE probe is compared with the classical models only on children whose EEG was
not in REVE's pretraining corpus (releases R4 and NC). The models
are trained exactly as in the benchmark; only the evaluation is restricted, so every model is
scored on the same trials.
"""

from __future__ import annotations

import pandas as pd

from decoding.metrics import auroc, nrmse, paired_subject_bootstrap, subject_bootstrap
from decoding.splits import HOLDOUT_FOLD

_METRICS = {"rt_s": ("nrmse", nrmse), "log_rt_centred": ("nrmse", nrmse), "lapse": ("auroc", auroc)}


def release_subset_summary(predictions: pd.DataFrame, releases: list[str],
                           reference: str = "demographics", n_bootstrap: int = 2000,
                           seed: int = 12345) -> pd.DataFrame:
    """Cross-validation predictions of test children from ``releases``, per window, target and model,
    with subject-bootstrap intervals and the paired difference against ``reference``."""
    keys = ["subject", "block_key", "trial_index"]
    cv = predictions[(predictions["fold"] != HOLDOUT_FOLD) & predictions["release"].isin(releases)]
    rows = []
    for (window, target), part in cv.groupby(["window", "target"], sort=True):
        metric_name, metric = _METRICS[target]
        ref = part[part["model"] == reference]
        for model, sub in part.groupby("model", sort=True):
            boot = subject_bootstrap(metric, sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                                     sub["subject"].to_numpy(), n_bootstrap=n_bootstrap, seed=seed)
            row = {"window": window, "target": target, "model": model, "releases": "+".join(releases),
                   "n_children": int(sub["subject"].nunique()), "n_trials": int(len(sub)),
                   "metric": metric_name, "estimate": boot.estimate, "ci_low": boot.ci_low,
                   "ci_high": boot.ci_high}
            if model != reference and not ref.empty:
                merged = sub.merge(ref[[*keys, "y_pred"]], on=keys, suffixes=("", "_ref"))
                paired = paired_subject_bootstrap(metric, merged["y_true"].to_numpy(),
                                                  merged["y_pred"].to_numpy(),
                                                  merged["y_pred_ref"].to_numpy(),
                                                  merged["subject"].to_numpy(),
                                                  n_bootstrap=n_bootstrap, seed=seed)
                row.update({f"delta_vs_{reference}": paired.estimate,
                            f"delta_vs_{reference}_ci_low": paired.ci_low,
                            f"delta_vs_{reference}_ci_high": paired.ci_high})
            rows.append(row)
    return pd.DataFrame(rows)
