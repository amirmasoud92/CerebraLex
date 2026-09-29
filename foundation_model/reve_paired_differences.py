"""Paired model-versus-model differences on test children unseen by REVE (R4 and NC).

Each REVE representation against every classical model and EEGNet, and EEGNet against the time course, on the same
trials (cross-validation folds, R4 and NC children), with the paired child bootstrap. nRMSE differences below zero
and AUROC differences above zero favour the first model.

Usage: python -m foundation_model.reve_paired_differences --runs <run> ... [--drop RUN=MODEL]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import REVEConfig
from common.scripts import script_path
from decoding.metrics import auroc, nrmse, paired_subject_bootstrap

REVE = ("reve_attnpool", "reve_meanfinal")
OTHERS = ("spectral", "spectral_aperiodic", "riemann", "timecourse", "eegnet")
TRIAL = ["subject", "block_key", "trial_index"]


def _combine():
    path = script_path("reve_unseen_children")
    spec = importlib.util.spec_from_file_location("reve_unseen_children", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.combine


def pairs_table(predictions: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    pairs = [(a, b) for a in REVE for b in OTHERS] + [("eegnet", "timecourse")]
    rows = []
    for (window, target), part in predictions.groupby(["window", "target"], sort=True):
        metric = nrmse if target == "rt_s" else auroc
        wide = part.pivot_table(index=[*TRIAL, "y_true"], columns="model", values="y_pred").reset_index()
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            sub = wide.dropna(subset=[a, b])
            result = paired_subject_bootstrap(metric, sub["y_true"].to_numpy(float), sub[a].to_numpy(float),
                                              sub[b].to_numpy(float), sub["subject"].to_numpy(),
                                              n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"window": window, "target": target, "model_a": a, "model_b": b,
                         "metric": metric.__name__, "n_children": sub["subject"].nunique(), "n_trials": len(sub),
                         "metric_a": metric(sub["y_true"], sub[a]), "metric_b": metric(sub["y_true"], sub[b]),
                         "delta": result.estimate, "delta_ci_low": result.ci_low, "delta_ci_high": result.ci_high})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--drop", action="append", default=[], metavar="RUN=MODEL[,MODEL]")
    parser.add_argument("--reve-config", default="foundation_model/reve.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    unseen = load_config(paths.root / args.reve_config, REVEConfig).unseen_releases
    drop: dict[Path, set[str]] = {}
    for item in args.drop:
        run, _, models = item.partition("=")
        drop.setdefault(Path(run).resolve(), set()).update(m for m in models.split(",") if m)
    predictions = _combine()(args.runs, drop)
    subset = predictions[predictions["release"].isin(unseen) & (predictions["fold"] != -1)]
    table = pairs_table(subset, args.n_bootstrap, args.seed)
    out_dir = paths.results / "reve_paired_differences" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "pairs.csv")
    write_json({"runs": args.runs, "dropped": args.drop, "unseen_releases": unseen,
                "git_commit": git_commit(paths.root)}, out_dir / "report.json")
    print(json.dumps({"out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
