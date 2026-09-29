"""Latent state models over trial sequences, evaluated on the benchmark's frozen folds.

Writes results/<name>/<run>/: predictions.parquet, selection.csv (held-out log-likelihood per HMM state count and LGSSM
dimension, restart spread, fitted dynamics), summary.csv (metric per window, target, feature set and scope with
child-bootstrap intervals and paired differences against the raw current-trial components and the smoothing
baseline), decomposition.csv (within-child and between-child parts of the RT predictions) and provenance.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from common.config import load_config, snapshot_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import (
    BenchmarkConfig,
    ComputeConfig,
    DerivativesConfig,
    PreprocessingConfig,
    StateModelConfig,
    TrialConfig,
)
from decoding.benchmark import BenchmarkRunner, demographics_table, load_window_data
from decoding.decomposition import decompose_predictions
from decoding.metrics import auroc, nrmse, paired_subject_bootstrap, subject_bootstrap
from decoding.splits import HOLDOUT_FOLD, SplitAssignment
from state_models_and_trait_moderation.runner import FEATURE_SETS, StateModelRunner

_KEYS = ["subject", "block_key", "trial_index"]


def summarise(predictions: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rows = []
    for (window, target), part in predictions.groupby(["window", "target"]):
        metric = nrmse if target == "rt_s" else auroc
        for scope, sub in (("cv", part[part["fold"] != HOLDOUT_FOLD]), ("holdout_R5", part[part["fold"] == HOLDOUT_FOLD])):
            if sub.empty:
                continue
            refs = {r: sub[sub["model"] == r][_KEYS + ["y_pred"]] for r in ("raw", "ewma")}
            for model in FEATURE_SETS:
                m = sub[sub["model"] == model]
                boot = subject_bootstrap(metric, m["y_true"].to_numpy(), m["y_pred"].to_numpy(),
                                         m["subject"].to_numpy(), n_bootstrap=n_bootstrap, seed=seed)
                row = {"window": window, "target": target, "scope": scope, "model": model,
                       "metric": "nrmse" if target == "rt_s" else "auroc", "n_subjects": int(m["subject"].nunique()),
                       "n_trials": int(len(m)), "estimate": boot.estimate, "ci_low": boot.ci_low, "ci_high": boot.ci_high}
                for ref_name, ref in refs.items():
                    if model == ref_name:
                        continue
                    merged = m.merge(ref, on=_KEYS, suffixes=("", "_ref"))
                    d = paired_subject_bootstrap(metric, merged["y_true"].to_numpy(), merged["y_pred"].to_numpy(),
                                                 merged["y_pred_ref"].to_numpy(), merged["subject"].to_numpy(),
                                                 n_bootstrap=n_bootstrap, seed=seed)
                    row.update({f"delta_vs_{ref_name}": d.estimate, f"delta_vs_{ref_name}_ci_low": d.ci_low,
                                f"delta_vs_{ref_name}_ci_high": d.ci_high})
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="state_models_and_trait_moderation/state_models.yaml")
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--store-name", default="windows-v1")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("state-space-models", paths.logs)
    cfg = load_config(paths.root / args.config, StateModelConfig, args.set)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    trial_cfg = load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    commit = git_commit(paths.root)
    windows = {w.name: load_window_data(deriv.root / args.store_name, w, preproc.resample_hz)
               for w in trial_cfg.windows if w.name in cfg.windows}
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    import mne

    montage = mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"]
    positions = np.array(list(montage.values()))
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    out_dir = paths.results / cfg.name / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(cfg, out_dir, "state_models")
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    bench = BenchmarkRunner(bcfg, compute, trials, windows, split, demographics_table(participants), positions,
                            out_dir, deriv.root / "features" / args.store_name)
    predictions, selection = StateModelRunner(cfg, bench).run()
    write_dataframe(predictions, out_dir / "predictions.parquet")
    selection["restart_logliks"] = selection["restart_logliks"].map(lambda v: json.dumps(v) if v is not None else "")
    write_dataframe(selection, out_dir / "selection.csv")
    summary = summarise(predictions, cfg.n_bootstrap, cfg.seed)
    write_dataframe(summary, out_dir / "summary.csv")
    write_dataframe(decompose_predictions(predictions, n_bootstrap=cfg.n_bootstrap, seed=cfg.seed),
                    out_dir / "decomposition.csv")
    log.info("state models written to %s", out_dir)
    print(json.dumps({"out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
