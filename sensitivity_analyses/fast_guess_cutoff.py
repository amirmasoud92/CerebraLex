"""Fast-guess cutoff for the S-antic sensitivity analysis (behaviour only, no EEG).

Bins valid responses of the decoding cohort by reaction time and finds the latency from which choice accuracy stays
above chance (Ratcliff & Tuerlinckx 2002, p. 442). Writes the bin table, the cutoff and the per-child share of
responses below it to results/fast_guess_cutoff/<run>/.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common.config import load_config, snapshot_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import DerivativesConfig, SensitivityConfig
from task_and_trials.behaviour import fast_guess_cutoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sensitivity_analyses/sensitivity.yaml")
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("fast-guess-cutoff", paths.logs)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    config = load_config(paths.root / args.config, SensitivityConfig)
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    qc = pd.read_parquet(deriv.root / "qc-preproc-v1" / "block_qc.parquet",
                         columns=["block_key", "analysis_usable"])
    trials = trials.merge(qc, on="block_key", how="left", validate="many_to_one")
    usable = trials["analysis_usable"].fillna(False).astype(bool)
    valid = trials[usable & trials["valid_response"].astype(bool)]
    spec = config.fast_guess
    result = fast_guess_cutoff(valid["rt_s"].to_numpy(), valid["correct"].to_numpy(), step_s=spec.step_s,
                               max_rt_s=spec.max_rt_s, ci_level=spec.ci_level, chance=spec.chance)
    summary = result.summary(valid["rt_s"].to_numpy())
    if result.cutoff_s is not None:
        share = (valid.assign(fast=valid["rt_s"] < result.cutoff_s)
                 .groupby("subject")["fast"].mean())
        summary.update({"n_subjects": int(share.size),
                        "n_subjects_over_5pct_below": int((share > 0.05).sum()),
                        "n_subjects_over_10pct_below": int((share > 0.10).sum())})
    summary["git_commit"] = git_commit(paths.root)
    out_dir = paths.results / "fast_guess_cutoff" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(result.bins, out_dir / "accuracy_by_rt_bin.csv")
    write_json(summary, out_dir / "cutoff.json")
    snapshot_config(config, out_dir, "sensitivity")
    log.info("fast-guess cutoff %s s; %s of %s valid responses below it", summary["cutoff_s"],
             summary["n_below_cutoff"], summary["n_responses"])
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
