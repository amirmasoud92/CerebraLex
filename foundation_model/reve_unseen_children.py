"""Every model scored on test children unseen by REVE's pretraining (releases R4 and NC).

Combines the predictions of several benchmark runs on the same frozen folds (for example the classical, time-course
and REVE runs), checks that models present in more than one run agree, and writes the release-subset summary with
paired differences against demographics and the within-child and between-child decomposition on that subset.

Usage: python -m foundation_model.reve_unseen_children --runs <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import REVEConfig
from decoding.decomposition import decompose_predictions
from decoding.subsets import release_subset_summary

_KEYS = ["window", "target", "model", "fold", "subject", "block_key", "trial_index"]


def combine(run_dirs: list[str], drop: dict[Path, set[str]] | None = None) -> pd.DataFrame:
    drop = drop or {}
    frames = []
    for d in run_dirs:
        frame = pd.read_parquet(Path(d) / "predictions.parquet")
        frames.append(frame[~frame["model"].isin(drop.get(Path(d).resolve(), set()))])
    merged = pd.concat(frames, ignore_index=True)
    repeated = merged[merged.duplicated(_KEYS, keep=False)]
    if not repeated.empty:
        spread = repeated.groupby(_KEYS)["y_pred"].agg(lambda v: float(np.ptp(v)))
        if spread.max() > 1e-9:
            raise ValueError(f"runs disagree on {int((spread > 1e-9).sum())} shared predictions")
    return merged.drop_duplicates(_KEYS, keep="first").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--drop", action="append", default=[], metavar="RUN=MODEL[,MODEL]",
                        help="leave these models of one run out")
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
    predictions = combine(args.runs, drop)
    subset = release_subset_summary(predictions, unseen, n_bootstrap=args.n_bootstrap, seed=args.seed)
    unseen_cv = predictions[predictions["release"].isin(unseen) & (predictions["fold"] != -1)]
    decomposition = decompose_predictions(unseen_cv, n_bootstrap=args.n_bootstrap, seed=args.seed)
    out_dir = paths.results / "reve_unseen_children" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(subset, out_dir / "unseen_subset_summary.csv")
    write_dataframe(decomposition, out_dir / "unseen_subset_decomposition.csv")
    write_json({"runs": args.runs, "dropped": args.drop, "unseen_releases": unseen, "git_commit": git_commit(paths.root),
                "n_predictions": int(len(predictions))}, out_dir / "report.json")
    rt = subset[subset["target"] == "rt_s"]
    print(rt[["window", "model", "n_children", "estimate", "ci_low", "ci_high",
              *[c for c in rt.columns if c.startswith("delta_vs_demographics")]]].round(4).to_string(index=False))
    print(json.dumps({"out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
