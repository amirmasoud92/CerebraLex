"""Does the EEG predict the RT before the response?

The benchmark's time-course ridge (100 ms bins of all 129 channels, standardised, PCA to 500 components, the
unpenalised demographic block, benchmark penalty grid and inner folds) is fitted in the frozen folds on windows that
end before the response on every trial used:

* E0: 0 to 500 ms after the target, trials with RT above 0.6 s;
* E1: 500 to 1,000 ms after the target, trials with RT above 1.1 s.

The cohort follows the benchmark rules on the full-channel store ``windows-v1-early`` (target to +1.0 s). Support
for pre-response information: a within-child r interval above 0 in cross-validation, confirmed in the R5 hold-out.
Writes results/preresponse_decoding/<run>/ (predictions, decomposition, summary).
"""

from __future__ import annotations

import argparse
import json
import sys

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
    TrialConfig,
    WindowSpec,
)
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.benchmark import (
    BenchmarkRunner,
    FeatureCache,
    demographics_table,
    load_window_data,
)
from decoding.decomposition import within_between
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD, SplitAssignment

WINDOWS = {"E0": ((0, 5), 0.6), "E1": ((5, 10), 1.1)}   # bins [start, stop) of 100 ms; minimum RT (s)
N_BINS = 10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("preresponse-decoding", paths.logs)
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    spec: WindowSpec = load_config(paths.root / "features" / "windows_early.yaml", TrialConfig).windows[0]
    window = load_window_data(deriv.root / "windows-v1-early", spec, preproc.resample_hz)
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    import mne

    positions = np.array(list(mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"].values()))
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    out_dir = paths.results / "preresponse_decoding" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    bench = BenchmarkRunner(bcfg, compute, trials, {"early": window}, split, demographics_table(participants),
                            positions, out_dir, deriv.root / "features" / "windows-v1-early")
    cache = FeatureCache(bench.feature_root, window, bcfg.aperiodic_regions, positions, None, n_jobs=bcfg.feature_n_jobs)
    features = cache.timecourse(bcfg.timecourse_bin_ms)
    cohort = bench.cohort(window)
    predictions, rows = [], []
    for name, ((b0, b1), min_rt) in WINDOWS.items():
        columns = np.array([ch * N_BINS + b for ch in range(window.array.shape[1]) for b in range(b0, b1)])
        part = cohort[cohort["valid_response"].astype(bool) & (cohort["rt_s"] > min_rt)].reset_index(drop=True)
        x = np.asarray(features[part["row"].to_numpy()][:, columns], dtype=np.float64)
        covariates = bench._covariates(part)
        subjects = part["subject"].to_numpy()
        pred = np.full(len(part), np.nan)
        folds = list(split.iter_folds()) + [(HOLDOUT_FOLD, sorted(split.cv_subjects["subject"]), split.holdout_subjects)]
        for fold, train_subjects, test_subjects in folds:
            train = part["subject"].isin(train_subjects).to_numpy()
            test = part["subject"].isin(test_subjects).to_numpy()
            LeakageAuditor.assert_rows_grouped(part, train, test)
            model = GroupedRidge(name=f"timecourse_{name}", pca_components=bcfg.timecourse_pca_components,
                                 alphas=tuple(bcfg.ridge_alphas), n_inner_folds=bcfg.inner_folds, seed=bcfg.seed)
            model.fit(DecoderInput(x[train], subjects[train], covariates[train]), part["rt_s"].to_numpy()[train])
            pred[test] = model.predict(DecoderInput(x[test], subjects[test], covariates[test]))
            log.info("%s fold %s done", name, fold)
        out = part[["subject", "block_key", "trial_index", "rt_s"]].assign(window=name, pred=pred,
                                                                             fold=split.fold_of(part["subject"]).to_numpy())
        predictions.append(out)
        for scope, sub in (("cv", out[out["fold"] != HOLDOUT_FOLD]), ("holdout_R5", out[out["fold"] == HOLDOUT_FOLD])):
            d = within_between(sub["rt_s"].to_numpy(float), sub["pred"].to_numpy(float), sub["subject"].to_numpy(),
                               n_bootstrap=args.n_bootstrap, seed=bcfg.seed)
            rows.append({"window": name, "min_rt_s": min_rt, "scope": scope, "n_trials": len(sub), **d})
    write_dataframe(pd.concat(predictions, ignore_index=True), out_dir / "predictions.parquet")
    table = pd.DataFrame(rows)
    write_dataframe(table, out_dir / "decomposition.csv")
    summary = {r.window + "_" + r.scope: {"r_within": r.r_within, "ci_low": r.r_within_ci_low, "ci_high": r.r_within_ci_high,
                                          "n_children": int(r.n_children), "n_trials": int(r.n_trials)}
               for r in table.itertuples()}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
