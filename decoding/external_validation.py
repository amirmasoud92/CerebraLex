"""External validation of the HBN decoders on another sample (MIPDB, or R10/R11).

Every requested benchmark model is refitted once on all HBN decoding-cohort children (all frozen folds, R5
included; valid responses for RT, all trials for lapse) with the benchmark's own decoders, inputs and settings, then
applied to the external cohort built with the same rules (valid window, no artifact window, at least 10 valid
responses). Reported per window, target and model: nRMSE or AUROC with child-bootstrap intervals, the paired
difference from demographics and, for RT, the within-child decomposition. Checks: E1, the
post-target time course has a within-child r interval above 0.10; E2, the pre-stimulus band-power model has a
within-child r interval above 0. Writes results/external_validation/<run>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import config_fingerprint, load_config
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
)
from decoding.benchmark import (
    BenchmarkRunner,
    FeatureCache,
    demographics_table,
    load_window_data,
)
from decoding.decomposition import within_between
from decoding.metrics import auroc, nrmse, paired_subject_bootstrap, subject_bootstrap
from decoding.splits import SplitAssignment

MODELS = ("mean", "demographics", "spectral", "spectral_aperiodic", "riemann", "timecourse", "eegnet")
E_CHECKS = {"E1": ("challenge", "timecourse", 0.10), "E2": ("prestim", "spectral", 0.0)}


def external_cohort(window, people: pd.DataFrame, cfg: BenchmarkConfig) -> pd.DataFrame:
    """MIPDB rows with the benchmark cohort rules (BenchmarkRunner.cohort without the HBN split)."""
    index = window.index.copy()
    valid = index["window_valid"].astype(bool)
    if cfg.exclude_artifact_windows:
        valid &= ~index[cfg.artifact_flag].fillna(False).astype(bool)
    per_child = index[valid & index["valid_response"]].groupby("subject").size()
    enough = per_child[per_child >= cfg.min_valid_responses_per_subject].index
    cohort = index[valid & index["subject"].isin(enough)].copy()
    cohort = cohort.merge(people[["subject", "age", "sex", "ehq_total"]], on="subject", how="inner")
    return cohort.reset_index(drop=True)


def evaluate(pred: pd.DataFrame, n_bootstrap: int, seed: int) -> tuple[list[dict], list[dict]]:
    summary, decomposition = [], []
    for (window, target), part in pred.groupby(["window", "target"], sort=True):
        metric = nrmse if target == "rt_s" else auroc
        wide = part.pivot_table(index=["subject", "block_key", "trial_index", "y_true"], columns="model",
                                values="y_pred").reset_index()
        y, subjects = wide["y_true"].to_numpy(float), wide["subject"].to_numpy()
        for model in [m for m in MODELS if m in wide]:
            boot = subject_bootstrap(metric, y, wide[model].to_numpy(float), subjects, n_bootstrap=n_bootstrap, seed=seed)
            row = {"window": window, "target": target, "model": model, "metric": metric.__name__,
                   "n_children": int(pd.unique(subjects).size), "n_trials": int(len(y)), **boot.to_dict()}
            if "demographics" in wide and model != "demographics":
                paired = paired_subject_bootstrap(metric, y, wide[model].to_numpy(float),
                                                  wide["demographics"].to_numpy(float), subjects,
                                                  n_bootstrap=n_bootstrap, seed=seed)
                row.update({"delta_vs_demographics": paired.estimate, "delta_vs_demographics_ci_low": paired.ci_low,
                            "delta_vs_demographics_ci_high": paired.ci_high})
            summary.append(row)
            if target == "rt_s":
                decomposition.append({"window": window, "model": model,
                                      **within_between(y, wide[model].to_numpy(float), subjects,
                                                       n_bootstrap=n_bootstrap, seed=seed)})
    return summary, decomposition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--external-root", default=data_location("MIPDB"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--windows", nargs="+", default=["prestim", "challenge"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("external-validation", paths.logs)
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    trial_cfg = load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    specs = {w.name: w for w in trial_cfg.windows if w.name in args.windows}
    hbn_windows = {n: load_window_data(deriv.root / "windows-v1", s, preproc.resample_hz) for n, s in specs.items()}
    ext_root = Path(args.external_root)
    ext_windows = {n: load_window_data(ext_root / "windows-v1", s, preproc.resample_hz) for n, s in specs.items()}
    people = pd.read_csv(ext_root / "manifest" / "participants.csv")
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    import mne

    positions = np.array(list(mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"].values()))
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    out_dir = paths.results / "external_validation" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv, "benchmark_config": config_fingerprint(bcfg),
                "external_root": str(ext_root)}, out_dir / "provenance.json")
    bench = BenchmarkRunner(bcfg, compute, trials, hbn_windows, split, demographics_table(participants), positions,
                            out_dir, deriv.root / "features" / "windows-v1")
    device = "cuda" if compute.device != "cpu" else None
    predictions = []
    for name, window in hbn_windows.items():
        ext_window = ext_windows[name]
        cache = FeatureCache(bench.feature_root, window, bcfg.aperiodic_regions, positions, device,
                             n_jobs=bcfg.feature_n_jobs)
        ext_cache = FeatureCache(ext_root / "features" / "windows-v1", ext_window, bcfg.aperiodic_regions, positions,
                                 device, n_jobs=bcfg.feature_n_jobs)
        train_all = bench.cohort(window)
        test_all = external_cohort(ext_window, people, bcfg)
        log.info("%s: HBN %d trials of %d children; MIPDB %d trials of %d children", name, len(train_all),
                 train_all["subject"].nunique(), len(test_all), test_all["subject"].nunique())
        for target in bcfg.targets:
            train = (train_all if target == "lapse" else train_all[train_all["valid_response"]]).reset_index(drop=True)
            test = (test_all if target == "lapse" else test_all[test_all["valid_response"]]).reset_index(drop=True)
            for model in args.models:
                start = time.perf_counter()
                decoder = bench._make_decoder(model, window)
                decoder.fit(bench._inputs(model, window, cache, train), train[target].to_numpy(float))
                pred = decoder.predict(bench._inputs(model, ext_window, ext_cache, test))
                out = test[["subject", "block_key", "trial_index"]].copy()
                out["window"], out["target"], out["model"] = name, target, model
                out["y_true"], out["y_pred"] = test[target].to_numpy(float), pred
                predictions.append(out)
                log.info("%s | %s | %s fitted on HBN and applied to MIPDB (%.0f s)", name, target, model,
                         time.perf_counter() - start)
    pred = pd.concat(predictions, ignore_index=True)
    write_dataframe(pred, out_dir / "predictions.parquet")
    summary, decomposition = evaluate(pred, args.n_bootstrap, bcfg.seed)
    summary, decomposition = pd.DataFrame(summary), pd.DataFrame(decomposition)
    write_dataframe(summary, out_dir / "summary.csv")
    write_dataframe(decomposition, out_dir / "decomposition.csv")
    checks = {}
    for check, (window, model, threshold) in E_CHECKS.items():
        row = decomposition[(decomposition["window"] == window) & (decomposition["model"] == model)]
        if row.empty:
            continue
        r = row.iloc[0]
        checks[check] = {"window": window, "model": model, "threshold": threshold, "r_within": r["r_within"],
                         "ci_low": r["r_within_ci_low"], "ci_high": r["r_within_ci_high"],
                         "holds": bool(r["r_within_ci_low"] > threshold), "n_children": int(r["n_children"])}
    write_json(checks, out_dir / "checks.json")
    log.info("external validation written to %s: %s", out_dir, checks)
    print(json.dumps({"out_dir": str(out_dir), "checks": checks}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
