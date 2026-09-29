"""How much of a child's age does each representation carry?

A ridge without demographic covariates (benchmark penalty grid and inner folds) predicts the child's age from single
pre-stimulus trials, trained on the frozen folds (10-fold cross-validation, and the R5 hold-out fitted on all
cross-validation children); trial predictions are averaged per child. Feature sets: band power, band power plus
aperiodic, 100 ms time course, and the two frozen REVE representations (read from the benchmark's feature cache;
this script never runs the encoder). Scopes: all cross-validation children, the R5 hold-out, and cross-validation
test children from R4 and NC, the releases absent from REVE's pretraining corpus. Children are resampled for the
intervals. Report only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import (
    BenchmarkConfig,
    ComputeConfig,
    DerivativesConfig,
    PreprocessingConfig,
    REVEConfig,
    TrialConfig,
)
from decoding.base import DecoderInput
from decoding.benchmark import (
    BenchmarkRunner,
    FeatureCache,
    demographics_table,
    load_window_data,
)
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD, SplitAssignment

FEATURE_SETS = ("spectral", "spectral_aperiodic", "timecourse", "reve_attnpool", "reve_meanfinal")
UNSEEN_BY_REVE = ("R4", "NC")


def _no_embedder():
    raise RuntimeError("REVE embeddings must already be cached by the REVE benchmark run; the age probe never runs the encoder")


def child_metrics(child: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    """Correlation, mean absolute error and R^2 of child-mean predictions, with child-bootstrap CIs."""
    age, pred = child["age"].to_numpy(float), child["pred"].to_numpy(float)

    def stats(a: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
        return (float(np.corrcoef(a, p)[0, 1]), float(np.mean(np.abs(a - p))),
                float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)))

    r, mae, r2 = stats(age, pred)
    rng = np.random.default_rng(seed)
    draws = np.array([stats(age[i], pred[i]) for i in
                      (rng.integers(0, len(age), len(age)) for _ in range(n_bootstrap))])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"n_children": int(len(age)), "age_sd": float(age.std(ddof=1)), "r": r, "r_ci_low": low[0],
            "r_ci_high": high[0], "mae_years": mae, "mae_ci_low": low[1], "mae_ci_high": high[1],
            "r2": r2, "r2_ci_low": low[2], "r2_ci_high": high[2]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--store-name", default="windows-v1")
    parser.add_argument("--window", default="prestim")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("age-probe", paths.logs)
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    trial_cfg = load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    reve_cfg = load_config(paths.root / bcfg.reve_config, REVEConfig)
    reve_tag = f"{reve_cfg.model.revision[:7]}-{reve_cfg.amplitude}"
    spec = next(w for w in trial_cfg.windows if w.name == args.window)
    window = load_window_data(deriv.root / args.store_name, spec, preproc.resample_hz)
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    import mne

    positions = np.array(list(mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"].values()))
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    out_dir = paths.results / "age_probe" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv, "reve_tag": reve_tag}, out_dir / "provenance.json")
    bench = BenchmarkRunner(bcfg, compute, trials, {spec.name: window}, split, demographics_table(participants),
                            positions, out_dir, deriv.root / "features" / args.store_name)
    cache = FeatureCache(bench.feature_root, window, bcfg.aperiodic_regions, positions, None,
                         n_jobs=bcfg.feature_n_jobs, embedder_factory=_no_embedder, reve_tag=reve_tag)
    rows = bench.cohort(window)
    rows = rows[pd.to_numeric(rows["age"], errors="coerce").notna()].reset_index(drop=True)
    age = pd.to_numeric(rows["age"]).to_numpy(float)
    subjects = rows["subject"].to_numpy()
    folds = list(split.iter_folds())
    folds.append((HOLDOUT_FOLD, sorted(split.cv_subjects["subject"]), split.holdout_subjects))
    summaries, per_child = [], []
    for name in args.feature_sets:
        start = time.perf_counter()
        full = bench._inputs(name, window, cache, rows)
        x = np.asarray(full.x)
        pred = np.full(len(rows), np.nan)
        fold_of = np.full(len(rows), -99)
        for fold, train_subjects, test_subjects in folds:
            train = rows["subject"].isin(train_subjects).to_numpy()
            test = rows["subject"].isin(test_subjects).to_numpy()
            if not test.any():
                continue
            LeakageAuditor.assert_rows_grouped(rows, train, test)
            decoder = bench._make_decoder(name, window)
            decoder.fit(DecoderInput(x[train], subjects[train], None), age[train])
            pred[test] = decoder.predict(DecoderInput(x[test], subjects[test], None))
            fold_of[test] = fold
        child = (pd.DataFrame({"subject": subjects, "release": rows["release"].to_numpy(), "fold": fold_of,
                               "age": age, "pred": pred})
                 .groupby(["subject", "release", "fold"], as_index=False).mean().assign(feature_set=name))
        per_child.append(child)
        scopes = {"cv": child[child["fold"] >= 0], "holdout_R5": child[child["fold"] == HOLDOUT_FOLD],
                  "cv_R4_NC": child[(child["fold"] >= 0) & child["release"].isin(UNSEEN_BY_REVE)]}
        for scope, part in scopes.items():
            summaries.append({"feature_set": name, "scope": scope,
                              **child_metrics(part, args.n_bootstrap, bcfg.seed)})
        log.info("%s done (%d trials, %.0f s)", name, len(rows), time.perf_counter() - start)
    summary = pd.DataFrame(summaries)
    write_dataframe(summary, out_dir / "age_probe.csv")
    write_dataframe(pd.concat(per_child, ignore_index=True), out_dir / "per_child.parquet")
    log.info("age probe written to %s", out_dir)
    print(json.dumps({"out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
