"""Can the variance shares and the predicted CPP build-up ratio of accumulation_model_analysis.py be trusted?

Uses the simulated children of accumulation_model_recovery.py.

Variance shares: for every simulated child, the rate, onset and within-trial shares of
accumulation_model_analysis.variance_shares from its true parameters and from its fitted full-model parameters;
medians per generating model.

Build-up ratio: for every simulated child, 1,000 trials are simulated from its true parameters and from its fitted
full, rate-only and onset-only parameters, and put through the rate estimator of cpp_onset_rate.py (decision
variable over a, RT tertiles of diffusion responses with RT 0.65 to 2.4 s, slope from -0.35 to -0.05 s). Per
generating model: the slow/fast ratio of child means implied by the true parameters and by each fit, with
child-bootstrap intervals. Writes results/accumulation_model_checks/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from accumulation_model import accumulation as acc
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.scripts import script_path

SOURCES = ("true", "full", "rate", "onset")
SHARES = ("rate_share", "onset_share", "within_trial_share")


def _analysis(path: str):
    spec = importlib.util.spec_from_file_location("accumulation_model_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child_rates(build_rate, variance_shares, consts: tuple[float, float, float], row: pd.Series, index: int,
                n_sim: int, seed: int) -> dict:
    """``build_rate``, ``variance_shares`` and ``consts`` (PATH_DT, MIN_RT_S, MAX_RT_S) come from accumulation_model_analysis.py."""
    path_dt, min_rt, max_rt = consts
    out = {"truth": row["truth"], "child": int(row["child"])}
    for source in ("true", "full"):
        params = {p: float(row[f"{source}_{p}"]) for p in acc.PARAMS}
        shares = variance_shares("", params)
        out.update({f"{source}_{k}": shares[k] for k in SHARES})
    for s_index, source in enumerate(SOURCES):
        prefix = "true" if source == "true" else source
        params = {p: float(row[f"{prefix}_{p}"]) for p in acc.PARAMS}
        sim = acc.simulate(params, n_sim, np.random.default_rng([seed, index, s_index]), dt=path_dt,
                           keep_paths=True, path_window_s=0.6)
        rt = sim["rt"]
        ok = sim["diffusion"] & np.isfinite(rt) & (rt >= min_rt) & (rt <= max_rt)
        if ok.sum() >= 30:
            out[f"{source}_fast"], out[f"{source}_slow"] = build_rate(sim["paths"][ok] / params["a"], rt[ok])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recovery", required=True, help="results/model/<ts>_recovery directory")
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    m = _analysis(str(script_path("accumulation_model_analysis")))
    consts = (m.PATH_DT, m.MIN_RT_S, m.MAX_RT_S)
    fits = pd.read_csv(Path(args.recovery) / "recovery_fits.csv")
    rows = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(
        delayed(child_rates)(m.build_rate, m.variance_shares, consts, r, i, args.n_sim, args.seed)
        for i, r in fits.iterrows()))
    summary = {"git_commit": git_commit(paths.root), "argv": sys.argv, "recovery": args.recovery, "ratios": {},
               "variance_share_medians": {truth: {source: {k: float(part[f"{source}_{k}"].median()) for k in SHARES}
                                                  for source in ("true", "full")}
                                          for truth, part in rows.groupby("truth")}}
    for truth, part in rows.groupby("truth"):
        summary["ratios"][truth] = {
            source: m.ratio_of_means(part[f"{source}_fast"].to_numpy(float), part[f"{source}_slow"].to_numpy(float),
                                     args.n_bootstrap, args.seed)
            for source in SOURCES if f"{source}_fast" in part}
    out_dir = paths.results / "accumulation_model_checks" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(rows, out_dir / "child_rates.csv")
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
