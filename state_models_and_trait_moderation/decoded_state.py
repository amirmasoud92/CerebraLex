"""Out-of-fold decoded state for every trial of the benchmark cohort (H2 state measure).

For each frozen fold a benchmark RT decoder (by default the spectral ridge with the unpenalised
demographic block; the model, its inputs and its settings come from ``BenchmarkRunner``, so they
are identical to the benchmark) is fitted on the training children's valid responses and applied to
every trial of the test children, misses included. Each child's value therefore comes from a
model that never saw that child. The decoded RT is then centred within child: the demographic
part is constant within a child and drops out, leaving the trial's deviation from the child's own
typical EEG state.

With ``child_standardise`` every feature is first standardised within child over the child's own
cohort trials (``features.normalisation``), so the decoder can use within-child
variation only and nothing that identifies the child (a robustness analysis of H2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decoding.base import DecoderInput
from decoding.benchmark import BenchmarkRunner, FeatureCache
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD
from features.normalisation import standardise_within_groups

STATE_MODELS = ("spectral", "timecourse")


def standardised_within_child(values: pd.Series, subjects: pd.Series) -> pd.Series:
    """Deviation from the child's mean, divided by the pooled SD of the deviations."""
    centred = values - values.groupby(subjects).transform("mean")
    return centred / centred.std()


def out_of_fold_state(bench: BenchmarkRunner, window_name: str, model: str = "spectral",
                      child_standardise: bool = False) -> pd.DataFrame:
    if model not in STATE_MODELS:
        raise ValueError(f"state model must be one of {STATE_MODELS}, got {model!r}")
    cfg = bench.cfg
    window = bench.windows[window_name]
    device = "cuda" if bench.compute.device != "cpu" else None
    cache = FeatureCache(bench.feature_root, window, cfg.aperiodic_regions, bench.positions, device,
                         n_jobs=cfg.feature_n_jobs)
    rows = bench.cohort(window).reset_index(drop=True)
    valid = rows["valid_response"].to_numpy(dtype=bool)
    rt = rows["rt_s"].to_numpy(dtype=float)
    full = None
    if child_standardise:
        full = bench._inputs(model, window, cache, rows)
        full = DecoderInput(standardise_within_groups(full.x, full.subjects), full.subjects, full.covariates)

    def inputs(mask: np.ndarray) -> DecoderInput:
        if full is None:
            return bench._inputs(model, window, cache, rows[mask].reset_index(drop=True))
        covariates = None if full.covariates is None else full.covariates[mask]
        return DecoderInput(full.x[mask], full.subjects[mask], covariates)

    decoded = np.full(len(rows), np.nan)
    fold_of = np.full(len(rows), -99)
    folds = list(bench.split.iter_folds())
    folds.append((HOLDOUT_FOLD, sorted(bench.split.cv_subjects["subject"]), bench.split.holdout_subjects))
    for fold, train_subjects, test_subjects in folds:
        train = rows["subject"].isin(train_subjects).to_numpy()
        test = rows["subject"].isin(test_subjects).to_numpy()
        if not test.any():
            continue
        LeakageAuditor.assert_rows_grouped(rows, train, test)
        fit_rows = train & valid
        decoder = bench._make_decoder(model, window)
        decoder.fit(inputs(fit_rows), rt[fit_rows])
        decoded[test] = decoder.predict(inputs(test))
        fold_of[test] = fold
    if np.isnan(decoded).any():
        raise ValueError(f"{int(np.isnan(decoded).sum())} cohort trials received no out-of-fold state")
    out = rows[["subject", "release", "block_key", "trial_index", "trial_in_block", "valid_response", "rt_s",
                "log_rt_centred", "lapse", "miss", "age", "sex", "ehq_total"]].copy()
    out["fold"] = fold_of
    out["decoded_rt"] = decoded
    out["state"] = standardised_within_child(out["decoded_rt"], out["subject"])
    return out
