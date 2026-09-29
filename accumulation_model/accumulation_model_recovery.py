"""Parameter and model recovery of the accumulation model from 65 trials per child.

150 simulated children per generating model (rate-only, onset-only, none), each fitted with all four models. Writes
results/accumulation_model_recovery/<run>/ (per-child fits, summary).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from accumulation_model import accumulation as acc
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit

TRUTHS = ("rate", "onset", "none")


def draw(truth: str, rng: np.random.Generator) -> dict:
    p = {"k": rng.uniform(1.0, 5.0), "a": rng.uniform(0.8, 1.6), "te": rng.uniform(0.3, 0.8),
         "lapse": rng.uniform(0.02, 0.2), "guess": rng.uniform(0.0, 0.05), "sv": 0.0, "st": 0.0}
    if truth == "rate":
        p["sv"] = rng.uniform(0.5, 2.0)
    if truth == "onset":
        p["st"] = rng.uniform(0.1, 0.5)
    return p


def one_child(truth: str, index: int, seed: int, n_trials: int) -> dict:
    rng = np.random.default_rng([seed, TRUTHS.index(truth), index])
    params = draw(truth, rng)
    sim = acc.simulate(params, n_trials, rng)
    data = acc.ChildData.from_trials(sim["rt"], sim["correct"])
    row = {"truth": truth, "child": index, **{f"true_{k}": v for k, v in params.items()},
           "n_responses": len(data.rt), "n_miss": data.n_miss}
    for model, res in acc.fit_all(data).items():
        row.update({f"{model}_{k}": res[k] for k in (*acc.PARAMS, "nll", "bic", "converged")})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-children", type=int, default=150)
    parser.add_argument("--n-trials", type=int, default=65)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    jobs = [(t, i) for t in TRUTHS for i in range(args.n_children)]
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(one_child)(t, i, args.seed, args.n_trials) for t, i in jobs)
    table = pd.DataFrame(rows)
    summary = {"git_commit": git_commit(paths.root), "argv": sys.argv, "model_recovery": {}, "parameter_recovery": {}}
    rng = np.random.default_rng(args.seed)
    for truth, part in table.groupby("truth"):
        delta = (part["onset_nll"] - part["rate_nll"]).to_numpy()
        draws = [delta[rng.integers(0, len(delta), len(delta))].sum() for _ in range(args.n_bootstrap)]
        bic_pick = part[[f"{m}_bic" for m in acc.MODELS]].idxmin(axis=1).str.replace("_bic", "")
        summary["model_recovery"][truth] = {
            "sum_nll_onset_minus_rate": float(delta.sum()), "ci": np.percentile(draws, [2.5, 97.5]).tolist(),
            "share_children_rate_better": float((delta > 0).mean()),
            "bic_choice_shares": bic_pick.value_counts(normalize=True).round(3).to_dict()}
        summary["model_recovery"][truth]["nesting_violations"] = {
            "rate_above_none": float((part["rate_nll"] - part["none_nll"] > 1e-6).mean()),
            "onset_above_none": float((part["onset_nll"] - part["none_nll"] > 1e-6).mean()),
            "full_above_best_nested": float((part["full_nll"] - part[["rate_nll", "onset_nll"]].min(axis=1) > 1e-6).mean())}
        gen = truth if truth != "none" else "none"
        summary["parameter_recovery"][truth] = {
            f"{p}_{m}": float(np.corrcoef(part[f"true_{p}"], part[f"{m}_{p}"])[0, 1])
            for m in (gen, "full") for p in acc.PARAMS
            if part[f"true_{p}"].std() > 0 and part[f"{m}_{p}"].std() > 0}
    out_dir = paths.results / "accumulation_model_recovery" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "recovery_fits.csv")
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
