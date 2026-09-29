"""Average within-child coupling, and trait moderation relative to it.

Reads a finished trait_state_moderation.py run directory (``child_slopes.parquet``, ``two_stage_results.csv``,
``state_table.parquet``) and writes ``coupling_summary.csv``: for every analysis family and outcome, the
random-effects mean of the child slopes (intercept-only meta-regression over the analysed children, method-of-moments
tau^2) and every trait coefficient as a percentage of that mean per trait SD, with a delta-method interval. The
power analysis states moderation in these units (25% and 50% of the slope per trait SD), so the
intervals show which moderation sizes the data exclude. Descriptive; no new test.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import write_dataframe
from common.paths import ProjectPaths
from common.scripts import script_path
from state_models_and_trait_moderation.trait_state import load_traits, meta_regression


def _driver():
    path = script_path("trait_state_moderation")
    spec = importlib.util.spec_from_file_location("trait_state_moderation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coupling_summary(run_dir: Path, traits: pd.DataFrame) -> pd.DataFrame:
    slopes = pd.read_parquet(run_dir / "child_slopes.parquet")
    results = pd.read_csv(run_dir / "two_stage_results.csv")
    table = pd.read_parquet(run_dir / "state_table.parquet")
    child_covariates = _driver().child_covariates
    rows = []
    for (analysis, outcome), part in slopes.groupby(["analysis", "outcome"], sort=False):
        scope = table[table["release"] == "R5"] if analysis == "replication_R5" else table
        analysed = set(traits["subject"]) & set(child_covariates(scope)["subject"])
        part = part[part["subject"].isin(analysed)]
        fit = meta_regression(part["slope"].to_numpy(), part["se"].to_numpy(), np.ones((len(part), 1)),
                              ("mean",))
        mean, mean_se = float(fit.coef[0]), float(fit.se[0])
        for r in results[(results["analysis"] == analysis) & (results["outcome"] == outcome)].itertuples():
            ratio = r.estimate / mean
            ratio_se = np.sqrt(r.se ** 2 / mean ** 2 + r.estimate ** 2 * mean_se ** 2 / mean ** 4)
            rows.append({"analysis": analysis, "outcome": outcome, "n_children": len(part),
                         "mean_slope": mean, "mean_slope_ci_low": mean - 1.96 * mean_se,
                         "mean_slope_ci_high": mean + 1.96 * mean_se, "tau": float(np.sqrt(fit.tau2)),
                         "trait": r.trait, "estimate": r.estimate, "se": r.se,
                         "percent_of_mean_slope": 100 * ratio,
                         "percent_ci_low": 100 * (ratio - 1.96 * ratio_se),
                         "percent_ci_high": 100 * (ratio + 1.96 * ratio_se),
                         "p_permutation": r.p_permutation, "p_bh": r.p_bh})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    traits = load_traits(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    run_dir = Path(args.run_dir)
    out = coupling_summary(run_dir, traits)
    write_dataframe(out, run_dir / "coupling_summary.csv")
    with pd.option_context("display.width", 200):
        print(out.to_string(float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
