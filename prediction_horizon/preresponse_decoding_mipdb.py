"""Pre-response RT decoding in an external sample (MIPDB, or R10/R11).

The time-course ridges of preresponse_decoding.py (E0: 0 to 500 ms on trials with RT above 0.6 s; E1: 500 to
1,000 ms on trials with RT above 1.1 s) are refitted once on all HBN decoding-cohort children (every frozen fold, R5
included), as in external_validation.py, and applied to the external cohort built with the same rules on the
full-channel store ``windows-v1-early``. Reports the within-child decomposition. Writes results/preresponse_decoding_mipdb/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import (
    BenchmarkConfig,
    ComputeConfig,
    DerivativesConfig,
    PreprocessingConfig,
    TrialConfig,
    WindowSpec,
)
from common.scripts import script_path
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.benchmark import (
    BenchmarkRunner,
    FeatureCache,
    demographics_table,
    load_window_data,
)
from decoding.decomposition import within_between
from decoding.splits import SplitAssignment


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--external-root", default=data_location("MIPDB"))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    preresponse, external = _script("preresponse_decoding"), _script("external_validation")
    paths = ProjectPaths.discover()
    log = configure_logging("preresponse-decoding-mipdb", paths.logs)
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    spec: WindowSpec = load_config(paths.root / "features" / "windows_early.yaml", TrialConfig).windows[0]
    ext_root = Path(args.external_root)
    hbn = load_window_data(deriv.root / "windows-v1-early", spec, preproc.resample_hz)
    ext = load_window_data(ext_root / "windows-v1-early", spec, preproc.resample_hz)
    people = pd.read_csv(ext_root / "manifest" / "participants.csv")
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    import mne

    positions = np.array(list(mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"].values()))
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    out_dir = paths.results / "preresponse_decoding_mipdb" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv, "external_root": str(ext_root)},
               out_dir / "provenance.json")
    bench = BenchmarkRunner(bcfg, compute, trials, {"early": hbn}, split, demographics_table(participants),
                            positions, out_dir, deriv.root / "features" / "windows-v1-early")
    features = FeatureCache(bench.feature_root, hbn, bcfg.aperiodic_regions, positions, None,
                            n_jobs=bcfg.feature_n_jobs).timecourse(bcfg.timecourse_bin_ms)
    ext_features = FeatureCache(ext_root / "features" / "windows-v1-early", ext, bcfg.aperiodic_regions, positions,
                                None, n_jobs=bcfg.feature_n_jobs).timecourse(bcfg.timecourse_bin_ms)
    train_all, test_all = bench.cohort(hbn), external.external_cohort(ext, people, bcfg)
    predictions, rows = [], []
    for name, ((b0, b1), min_rt) in preresponse.WINDOWS.items():
        columns = np.array([ch * preresponse.N_BINS + b for ch in range(hbn.array.shape[1]) for b in range(b0, b1)])
        train = train_all[train_all["valid_response"].astype(bool) & (train_all["rt_s"] > min_rt)].reset_index(drop=True)
        test = test_all[test_all["valid_response"].astype(bool) & (test_all["rt_s"] > min_rt)].reset_index(drop=True)
        model = GroupedRidge(name=f"timecourse_{name}", pca_components=bcfg.timecourse_pca_components,
                             alphas=tuple(bcfg.ridge_alphas), n_inner_folds=bcfg.inner_folds, seed=bcfg.seed)
        model.fit(DecoderInput(np.asarray(features[train["row"].to_numpy()][:, columns], dtype=np.float64),
                               train["subject"].to_numpy(), bench._covariates(train)), train["rt_s"].to_numpy(float))
        pred = model.predict(DecoderInput(np.asarray(ext_features[test["row"].to_numpy()][:, columns], dtype=np.float64),
                                          test["subject"].to_numpy(), bench._covariates(test)))
        log.info("%s: fitted on %d HBN trials (%d children), applied to %d MIPDB trials (%d children)", name,
                 len(train), train["subject"].nunique(), len(test), test["subject"].nunique())
        predictions.append(test[["subject", "block_key", "trial_index", "rt_s"]].assign(window=name, pred=pred))
        d = within_between(test["rt_s"].to_numpy(float), pred, test["subject"].to_numpy(),
                           n_bootstrap=args.n_bootstrap, seed=bcfg.seed)
        rows.append({"window": name, "min_rt_s": min_rt, "scope": "mipdb", "n_trials": len(test), **d})
    write_dataframe(pd.concat(predictions, ignore_index=True), out_dir / "predictions.parquet")
    table = pd.DataFrame(rows)
    write_dataframe(table, out_dir / "decomposition.csv")
    summary = {r.window + "_mipdb": {"r_within": r.r_within, "ci_low": r.r_within_ci_low, "ci_high": r.r_within_ci_high,
                                     "n_children": int(r.n_children), "n_trials": int(r.n_trials)}
               for r in table.itertuples()}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
