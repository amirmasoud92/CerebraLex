"""The shape of the age effect on the within-child CPP-RT coupling (descriptive).

Recomputes the child slopes of single_trial_associations.py (log RT on the standardised response-locked CPP slope,
trial position as covariate) and the same random-effects meta-regression on the age spline plus sex, checks that the
Wald statistic matches the saved test, and writes the fitted coupling by age with 95% intervals (sex at the sample
mean). No new test. Writes results/cpp_coupling_by_age/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import BenchmarkConfig, DerivativesConfig
from common.scripts import script_path
from state_models_and_trait_moderation.trait_state import child_slopes, meta_regression

REPORTED_WALD = 12.117169238268435


def _associations():
    spec = importlib.util.spec_from_file_location("single_trial_associations", script_path("single_trial_associations"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    args = parser.parse_args()
    assoc = _associations()
    paths = ProjectPaths.discover()
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    measures = pd.read_parquet(deriv.root / "features" / "epochs-v1" / "trial_measures.parquet")
    frame = measures[measures["window_valid"].astype(bool) & ~measures["artifact_window"].astype(bool)].copy()
    per_child = frame[frame["valid_response"].astype(bool)].groupby("subject").size()
    frame = frame[frame["subject"].isin(per_child[per_child >= bcfg.min_valid_responses_per_subject].index)]
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    demo = participants.drop_duplicates("participant_id").assign(
        subject=lambda f: f["participant_id"].astype(str).str.replace("^sub-", "", regex=True))
    demo = demo[["subject", "age", "sex"]].assign(age=lambda f: pd.to_numeric(f["age"], errors="coerce"),
                                                   sex_male=lambda f: f["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}))
    frame["log_rt"] = np.log(frame["rt_s"].where(frame["valid_response"].astype(bool)))
    valid = frame[frame["valid_response"].astype(bool)].dropna(subset=["cpp_slope_resp", "log_rt"])
    z = valid.assign(m=(valid["cpp_slope_resp"] - valid["cpp_slope_resp"].mean()) / valid["cpp_slope_resp"].std())
    slopes = child_slopes(z, "log_rt", "m", within=("trial_index",)).merge(demo, on="subject").dropna(subset=["age", "sex_male"])
    knots = np.quantile(slopes["age"], np.linspace(0, 1, 5))
    design = np.column_stack([np.ones(len(slopes)), assoc.natural_basis(slopes["age"].to_numpy(), knots), slopes["sex_male"]])
    fit = meta_regression(slopes["slope"].to_numpy(), slopes["se"].to_numpy(), design,
                          ("intercept", *[f"age{i}" for i in range(design.shape[1] - 2)], "sex_male"))
    cov = np.linalg.inv((design.T / (slopes["se"].to_numpy() ** 2 + fit.tau2)) @ design)
    idx = slice(1, design.shape[1] - 1)
    b = fit.coef[idx]
    wald = float(b @ np.linalg.solve(cov[idx, idx], b))
    if not np.isclose(wald, REPORTED_WALD, rtol=1e-9):
        raise RuntimeError(f"Wald {wald} does not reproduce the reported {REPORTED_WALD}")
    grid = np.arange(5.5, 21.5, 0.5)
    g = np.column_stack([np.ones(len(grid)), assoc.natural_basis(grid, knots), np.full(len(grid), slopes["sex_male"].mean())])
    pred = g @ fit.coef
    se = np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))
    curve = pd.DataFrame({"age": grid, "coupling": pred, "ci_low": pred - 1.96 * se, "ci_high": pred + 1.96 * se})
    ages = slopes["age"].to_numpy()
    curve["n_children_within_1y"] = [int((np.abs(ages - a) <= 1).sum()) for a in grid]
    out_dir = paths.results / "cpp_coupling_by_age" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(curve, out_dir / "coupling_by_age.csv")
    write_dataframe(slopes, out_dir / "child_slopes.csv")
    summary = {"git_commit": git_commit(paths.root), "wald_chi2": wald, "reproduces_reported": True,
               "n_children": len(slopes), "units": "change in log RT per SD of the response-locked CPP slope",
               "mean_coupling": float(np.average(slopes["slope"], weights=1 / (slopes["se"] ** 2 + fit.tau2)))}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1))
    print(curve.iloc[::4].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
