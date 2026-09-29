"""Diagnostics of the failed positive control (attention problems and RT variability).

The primary control (attention and the SD of log RT, four bifactor CBCL dimensions entered together) failed.
Two diagnostics look for the reason. They do not replace the failed control:

* D1, trait: the attention-specific factor excludes the general factor, whereas the meta-analytic effect compares
  ADHD diagnoses with controls. Refit with attention alone and with the general factor alone.
* D2, measure: misses (no response within the 2.5 s window) are the slow tail that the SD of valid log RTs cannot
  see. Recompute the SD and the interquartile range of log RT with every miss placed at the 2.5 s deadline.

Cohort, covariates (age spline, sex, release), outcome standardisation, permutation scheme and seed are those of
single_trial_associations.py. Writes results/attention_control_diagnostics/<run>/.
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
from state_models_and_trait_moderation.trait_state import TRAIT_COLUMNS, load_traits

DEADLINE_S = 2.5
TRAIT_SETS = {"joint (primary)": list(TRAIT_COLUMNS), "attention alone": ["attention"],
              "p_factor alone": ["p_factor"]}


def _associations():
    spec = importlib.util.spec_from_file_location("single_trial_associations", script_path("single_trial_associations"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variability(frame: pd.DataFrame) -> pd.DataFrame:
    """Per child: SD of valid log RTs, and SD and IQR of log RT with misses placed at the deadline."""
    rows = []
    for subject, part in frame.groupby("subject"):
        valid = part["valid_response"].astype(bool).to_numpy()
        rt = part["rt_s"].to_numpy(float)
        log_valid = np.log(np.clip(rt[valid], 1e-3, None))
        log_all = np.log(np.where(valid, np.clip(rt, 1e-3, None), DEADLINE_S))
        q25, q75 = np.percentile(log_all, [25, 75])
        rows.append({"subject": subject, "sd_log_rt": float(log_valid.std(ddof=1)),
                     "sd_log_rt_deadline": float(log_all.std(ddof=1)), "iqr_log_rt_deadline": float(q75 - q25),
                     "miss_rate": float(1 - valid.mean()), "n_targets": len(part)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
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
    data = (variability(frame).merge(demo, on="subject", how="left").merge(load_traits(participants), on="subject")
            .merge(measures.drop_duplicates("subject")[["subject", "release"]], on="subject", how="left"))
    rows = []
    for measure in ("sd_log_rt", "sd_log_rt_deadline", "iqr_log_rt_deadline"):
        part = data.dropna(subset=[measure, "age", "sex_male", *TRAIT_COLUMNS])
        knots = np.quantile(part["age"], np.linspace(0, 1, 5))
        releases = pd.get_dummies(part["release"], drop_first=True, dtype=float).to_numpy()
        cov = np.column_stack([assoc.natural_basis(part["age"].to_numpy(), knots), part["sex_male"], releases])
        y = part[measure].to_numpy(float)
        y = (y - y.mean()) / y.std()
        for label, columns in TRAIT_SETS.items():
            beta, se, t, p = assoc.trait_permutation(y, part[columns].to_numpy(float), cov, args.n_permutations, args.seed)
            for i, trait in enumerate(columns):
                rows.append({"measure": measure, "model": label, "trait": trait, "n_children": len(part),
                             "beta_sd_per_sd": beta[i], "se": se[i], "t": t[i], "p_permutation": p[i]})
    table = pd.DataFrame(rows)
    out_dir = paths.results / "attention_control_diagnostics" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "control_diagnostics.csv")
    write_dataframe(data.drop(columns=list(TRAIT_COLUMNS)), out_dir / "child_variability.csv")
    summary = {"git_commit": git_commit(paths.root), "n_children": int(data["subject"].nunique()),
               "deadline_s": DEADLINE_S, "mean_miss_rate": float(data["miss_rate"].mean()),
               "corr_sd_valid_vs_deadline": float(data[["sd_log_rt", "sd_log_rt_deadline"]].corr().iloc[0, 1])}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1))
    focus = table[table["trait"].isin(["attention", "p_factor"])]
    print(focus.to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
