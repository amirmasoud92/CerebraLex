"""Fit the accumulation model to every child.

Cohort: children with at least 10 valid responses in their analysis-usable blocks, with every target of those blocks
(responses with side and correctness, and misses). Each child is fitted with all four models (full, rate-only,
onset-only, none). Writes results/accumulation_model_fit/<tag>/<run>/ (per-child parameters, negative log-likelihood and BIC per
model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from accumulation_model import accumulation as acc
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit


def cohort_trials(trials_path: Path, qc_path: Path, min_valid: int = 10) -> pd.DataFrame:
    trials = pd.read_parquet(trials_path, columns=["subject", "block_key", "rt_s", "correct", "valid_response"])
    qc = pd.read_parquet(qc_path, columns=["block_key", "analysis_usable"])
    usable = set(qc.loc[qc["analysis_usable"].astype(bool), "block_key"])
    trials = trials[trials["block_key"].isin(usable)]
    n_valid = trials[trials["valid_response"].astype(bool)].groupby("subject").size()
    return trials[trials["subject"].isin(n_valid[n_valid >= min_valid].index)].reset_index(drop=True)


def fit_child(subject: str, part: pd.DataFrame) -> dict:
    rt = part["rt_s"].where(part["valid_response"].astype(bool)).to_numpy(float)
    data = acc.ChildData.from_trials(rt, part["correct"].fillna(False).to_numpy(bool))
    row = {"subject": subject, "n_targets": len(part), "n_responses": len(data.rt), "n_miss": data.n_miss}
    for model, res in acc.fit_all(data).items():
        row.update({f"{model}_{k}": res[k] for k in (*acc.PARAMS, "nll", "bic", "converged")})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=data_location("HBN"))
    parser.add_argument("--tag", default="hbn")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    root = Path(args.root)
    trials = cohort_trials(root / "trials-v1" / "trials_ccd.parquet", root / "qc-preproc-v1" / "block_qc.parquet")
    groups = list(trials.groupby("subject", sort=True))
    if args.limit:
        groups = groups[:args.limit]
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(fit_child)(s, part) for s, part in groups)
    table = pd.DataFrame(rows)
    out_dir = paths.results / "accumulation_model_fit" / args.tag / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "fits.csv")
    summary = {"git_commit": git_commit(paths.root), "argv": sys.argv, "root": str(root), "n_children": len(table),
               "converged_share": {m: float(table[f"{m}_converged"].mean()) for m in acc.MODELS}}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
