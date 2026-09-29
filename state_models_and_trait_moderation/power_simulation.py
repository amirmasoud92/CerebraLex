"""Simulation-based power for the trait-by-state interaction (H2).

Fits the trial-level mixed model to simulated data with planted standardised interactions and reports the rejection
rate at alpha = 0.05 after FDR over four traits. The sample size defaults to the audited HBN cohort (about 1,800
children with about 60 valid responses each).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from joblib import Parallel, delayed

from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from state_models_and_trait_moderation.mixed_models import TRAITS, TraitStateModel
from state_models_and_trait_moderation.permutation import fdr_correct
from state_models_and_trait_moderation.simulate import simulate_trait_state_data


def one_replicate(gamma: float, n_subjects: int, trials: int, seed: int, state_slope: float,
                  slope_sd: float, residual_sd: float) -> dict:
    data = simulate_trait_state_data(n_subjects=n_subjects, trials_per_subject=trials,
                                     interaction={"externalizing": gamma}, seed=seed,
                                     state_slope=state_slope, slope_sd=slope_sd,
                                     residual_sd=residual_sd)
    result = TraitStateModel().fit(data)
    p = result.interactions.set_index("trait")["p"].reindex(list(TRAITS)).to_numpy()
    _, reject = fdr_correct(p, q=0.05)
    est = result.interaction_estimates()
    return {"gamma": gamma, "seed": seed, "converged": result.converged,
            "estimate_externalizing": est["externalizing"],
            "reject_externalizing": bool(reject[list(TRAITS).index("externalizing")]),
            "false_rejections": int(reject.sum() - reject[list(TRAITS).index("externalizing")])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=1800)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--gammas", type=float, nargs="*", default=[0.0, 0.03, 0.05, 0.08])
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--state-slope", type=float, default=0.15,
                        help="mean within-child state slope; 0.0175 matches the observed r of 0.05")
    parser.add_argument("--slope-sd", type=float, default=0.08, help="between-child SD of the slope")
    parser.add_argument("--residual-sd", type=float, default=0.35)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("power-simulation", paths.logs)
    jobs = [(g, args.n_subjects, args.trials, 1000 * i + int(g * 1000), args.state_slope,
             args.slope_sd, args.residual_sd)
            for g in args.gammas for i in range(args.replicates)]
    rows = Parallel(n_jobs=args.n_jobs)(delayed(one_replicate)(*job) for job in jobs)
    frame = pd.DataFrame(rows)
    power = frame.groupby("gamma").agg(power=("reject_externalizing", "mean"),
                                       mean_estimate=("estimate_externalizing", "mean"),
                                       sd_estimate=("estimate_externalizing", "std"),
                                       false_rejections_per_run=("false_rejections", "mean"),
                                       convergence=("converged", "mean")).reset_index()
    out_dir = paths.results / "power_simulation" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(frame, out_dir / "replicates.csv")
    write_dataframe(power, out_dir / "power.csv")
    write_json(vars(args), out_dir / "args.json")
    log.info("power table:\n%s", power.to_string(index=False))
    print(json.dumps(power.to_dict(orient="records"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
