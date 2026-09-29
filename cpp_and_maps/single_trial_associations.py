"""Single-trial associations, development and psychopathology in HBN (exploratory).

Within child: the association of each single-trial measure with log RT (valid responses) and with lapse (pre-target
and target-locked measures): pooled within-child r with child-bootstrap intervals and the random-effects mean of
per-child slopes; Benjamini-Hochberg over measures per outcome. The RT-free measures (target-locked CPP slope,
pre-target alpha and SSVEP) also enter a ridge decoder in the frozen folds, for comparison with the benchmark
decoders.

Development: child-level behaviour (median RT, SD of log RT, lapse rate, fast-guess rate, accuracy, EZ-diffusion
drift, boundary and non-decision time) and child means of the measures as functions of age (natural spline, 4 df,
plus sex): the variance explained by age, its F test and fitted trajectories; and whether the within-child CPP-RT
coupling changes with age.

Psychopathology: the four CBCL bifactor dimensions (entered together) against every child-level measure, with age
spline, sex and release; studentised permutation p-values (5,000 joint permutations of the traits);
Benjamini-Hochberg over all trait-by-measure tests. Positive control: attention and the SD of log RT.

Writes results/single_trial_associations/<run>/ (tables and summary.json).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import BenchmarkConfig, DerivativesConfig
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.decomposition import within_between
from decoding.splits import HOLDOUT_FOLD, SplitAssignment
from state_models_and_trait_moderation.permutation import fdr_correct
from state_models_and_trait_moderation.trait_state import (
    TRAIT_COLUMNS,
    child_slopes,
    load_traits,
    meta_regression,
)

MEASURES = ("cpp_slope_resp", "cpp_amp_resp", "cpp_slope_target", "alpha_logpow", "ssvep_logpow")
RT_FREE = ("cpp_slope_target", "alpha_logpow", "ssvep_logpow")
LAPSE_MEASURES = RT_FREE
FAST_GUESS_S = 0.65
BEHAVIOUR = ("median_rt", "sd_log_rt", "lapse_rate", "fast_guess_rate", "accuracy", "ez_drift", "ez_boundary",
             "ez_nondecision")


def natural_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis with the given knots (the formula of mixed_models.natural_spline_basis)."""
    x = np.asarray(x, dtype=float)
    kn = knots[-1]

    def d(k):
        return (np.clip(x - k, 0, None) ** 3 - np.clip(x - kn, 0, None) ** 3) / (kn - k)

    return np.column_stack([x] + [d(k) - d(knots[-2]) for k in knots[:-2]])


def ez_diffusion(pc: float, vrt: float, mrt: float, n: int, s: float = 0.1) -> tuple[float, float, float]:
    """EZ-diffusion drift, boundary and non-decision time (Wagenmakers et al. 2007), edge-corrected."""
    if not np.isfinite([pc, vrt, mrt]).all() or vrt <= 0 or n < 2:
        return np.nan, np.nan, np.nan
    if pc >= 1.0:
        pc = 1.0 - 1.0 / (2 * n)
    if pc == 0.5:
        return np.nan, np.nan, np.nan
    logit = np.log(pc / (1 - pc))
    x = logit * (logit * pc ** 2 - logit * pc + pc - 0.5) / vrt
    v = np.sign(pc - 0.5) * s * x ** 0.25
    a = s ** 2 * logit / v
    y = -v * a / s ** 2
    mdt = (a / (2 * v)) * (1 - np.exp(y)) / (1 + np.exp(y))
    return float(v), float(a), float(mrt - mdt)


def within_association(frame: pd.DataFrame, measure: str, outcome: str, n_bootstrap: int, seed: int) -> dict:
    part = frame.dropna(subset=[measure, outcome])
    dec = within_between(part[outcome].to_numpy(float), part[measure].to_numpy(float), part["subject"].to_numpy(),
                         n_bootstrap=n_bootstrap, seed=seed)
    z = part.assign(m=(part[measure] - part[measure].mean()) / part[measure].std())
    slopes = child_slopes(z, outcome, "m", within=("trial_index",))
    slopes = slopes[slopes["se"] > 0]
    fit = meta_regression(slopes["slope"].to_numpy(), slopes["se"].to_numpy(), np.ones((len(slopes), 1)), ("mean",))
    return {"measure": measure, "outcome": outcome, "n_trials": len(part), "n_children": part["subject"].nunique(),
            "r_within": dec["r_within"], "r_within_ci_low": dec["r_within_ci_low"],
            "r_within_ci_high": dec["r_within_ci_high"], "mean_slope_per_sd": float(fit.coef[0]),
            "mean_slope_ci_low": float(fit.coef[0] - 1.96 * fit.se[0]),
            "mean_slope_ci_high": float(fit.coef[0] + 1.96 * fit.se[0]), "tau": float(np.sqrt(fit.tau2)),
            "z_for_p": float(fit.coef[0] / fit.se[0])}


def child_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, part in frame.groupby("subject"):
        valid = part[part["valid_response"].astype(bool)]
        correct = valid[valid["correct"].fillna(False).astype(bool)]
        v, a, ter = ez_diffusion(float(correct.shape[0] / max(len(valid), 1)), float(correct["rt_s"].var(ddof=1)),
                                 float(correct["rt_s"].mean()), len(valid))
        row = {"subject": subject, "median_rt": float(valid["rt_s"].median()),
               "sd_log_rt": float(np.log(valid["rt_s"].clip(lower=1e-3)).std(ddof=1)),
               "lapse_rate": float(part["lapse"].astype(float).mean()),
               "fast_guess_rate": float((valid["rt_s"] < FAST_GUESS_S).mean()),
               "accuracy": float(correct.shape[0] / max(len(valid), 1)),
               "ez_drift": v, "ez_boundary": a, "ez_nondecision": ter, "n_trials": len(part)}
        for m in MEASURES:
            row[m] = float(part[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def age_trajectory(children: pd.DataFrame, measure: str, grid: np.ndarray, n_bootstrap: int, seed: int) -> tuple[dict, pd.DataFrame]:
    part = children.dropna(subset=[measure, "age", "sex_male"])
    knots = np.quantile(part["age"], np.linspace(0, 1, 5))
    x_age = natural_basis(part["age"].to_numpy(), knots)
    full = np.column_stack([np.ones(len(part)), x_age, part["sex_male"]])
    reduced = np.column_stack([np.ones(len(part)), part["sex_male"]])
    y = part[measure].to_numpy(float)
    rss = lambda x: float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))  # noqa: E731
    rss_full, rss_red = rss(full), rss(reduced)
    df1, df2 = full.shape[1] - reduced.shape[1], len(y) - full.shape[1]
    f = ((rss_red - rss_full) / df1) / (rss_full / df2)
    grid_x = np.column_stack([np.ones(len(grid)), natural_basis(grid, knots), np.full(len(grid), part["sex_male"].mean())])
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y), len(y))
        beta = np.linalg.lstsq(full[i], y[i], rcond=None)[0]
        curves.append(grid_x @ beta)
    beta = np.linalg.lstsq(full, y, rcond=None)[0]
    low, high = np.percentile(np.array(curves), [2.5, 97.5], axis=0)
    total = float(np.sum((y - y.mean()) ** 2))
    summary = {"measure": measure, "n_children": len(y), "r2_age_given_sex": (rss_red - rss_full) / total,
               "f": f, "df1": df1, "df2": df2, "p": float(stats.f.sf(f, df1, df2))}
    curve = pd.DataFrame({"measure": measure, "age": grid, "fit": grid_x @ beta, "ci_low": low, "ci_high": high})
    return summary, curve


def trait_permutation(y: np.ndarray, traits: np.ndarray, covariates: np.ndarray, n_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients, SEs, t and studentised joint-permutation p-values of the trait columns."""
    def fit(t):
        x = np.column_stack([np.ones(len(y)), covariates, t])
        xtx_inv = np.linalg.pinv(x.T @ x)
        beta = xtx_inv @ x.T @ y
        resid = y - x @ beta
        se = np.sqrt(np.diag(xtx_inv) * (resid @ resid) / (len(y) - x.shape[1]))
        k = t.shape[1]
        return beta[-k:], se[-k:]

    beta, se = fit(traits)
    t_obs = beta / se
    rng = np.random.default_rng(seed)
    exceed = np.zeros(traits.shape[1])
    for _ in range(n_perm):
        b, s = fit(traits[rng.permutation(len(traits))])
        exceed += np.abs(b / s) >= np.abs(t_obs)
    return beta, se, t_obs, (1 + exceed) / (1 + n_perm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("single-trial-associations", paths.logs)
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    measures = pd.read_parquet(deriv.root / "features" / "epochs-v1" / "trial_measures.parquet")
    usable = measures["window_valid"].astype(bool) & ~measures["artifact_window"].astype(bool)
    frame = measures[usable].copy()
    per_child = frame[frame["valid_response"].astype(bool)].groupby("subject").size()
    frame = frame[frame["subject"].isin(per_child[per_child >= bcfg.min_valid_responses_per_subject].index)]
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    demo = participants.drop_duplicates("participant_id").assign(
        subject=lambda f: f["participant_id"].astype(str).str.replace("^sub-", "", regex=True))
    demo = demo[["subject", "age", "sex"]].assign(age=lambda f: pd.to_numeric(f["age"], errors="coerce"),
                                                   sex_male=lambda f: f["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}))
    frame["log_rt"] = np.log(frame["rt_s"].where(frame["valid_response"].astype(bool)))
    frame["lapse_f"] = frame["lapse"].astype(float)
    out_dir = paths.results / "single_trial_associations" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    summary: dict = {"n_trials": len(frame), "n_children": int(frame["subject"].nunique())}

    # within-child associations
    within_child_table = [within_association(frame[frame["valid_response"].astype(bool)], m, "log_rt", args.n_bootstrap, args.seed)
                          for m in MEASURES]
    within_child_table += [within_association(frame, m, "lapse_f", args.n_bootstrap, args.seed) for m in LAPSE_MEASURES]
    within_child_table = pd.DataFrame(within_child_table)
    within_child_table["p_normal"] = 2 * stats.norm.sf(np.abs(within_child_table["z_for_p"]))
    within_child_table["p_bh"] = np.nan
    for _outcome, idx in within_child_table.groupby("outcome").groups.items():
        within_child_table.loc[idx, "p_bh"] = fdr_correct(within_child_table.loc[idx, "p_normal"].to_numpy())[0]
    write_dataframe(within_child_table, out_dir / "within_child.csv")
    log.info("within-child associations done")

    # RT-free measures as a decoder in the frozen folds (RT > 0.6 s, the target-locked CPP window)
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    dec = frame[frame["valid_response"].astype(bool) & frame["rt_s"].gt(0.6)].dropna(subset=list(RT_FREE)).copy()
    dec = dec.merge(demo[["subject", "age", "sex_male"]], on="subject", how="left")
    dec["fold"] = split.fold_of(dec["subject"]).astype("Int64")
    dec = dec[dec["fold"].notna()].reset_index(drop=True)
    covariates = np.column_stack([dec["age"], dec["sex_male"]])
    pred = np.full(len(dec), np.nan)
    folds = list(split.iter_folds()) + [(HOLDOUT_FOLD, sorted(split.cv_subjects["subject"]), split.holdout_subjects)]
    for _, train_subjects, test_subjects in folds:
        train = dec["subject"].isin(train_subjects).to_numpy()
        test = dec["subject"].isin(test_subjects).to_numpy()
        model = GroupedRidge(name="rt_free", alphas=tuple(bcfg.ridge_alphas), n_inner_folds=bcfg.inner_folds, seed=bcfg.seed)
        x = dec[list(RT_FREE)].to_numpy(float)
        model.fit(DecoderInput(x[train], dec["subject"].to_numpy()[train], covariates[train]), dec["rt_s"].to_numpy()[train])
        pred[test] = model.predict(DecoderInput(x[test], dec["subject"].to_numpy()[test], covariates[test]))
    dec["pred"] = pred
    rows = []
    for scope, part in (("cv", dec[dec["fold"] != HOLDOUT_FOLD]), ("holdout_R5", dec[dec["fold"] == HOLDOUT_FOLD])):
        d = within_between(part["rt_s"].to_numpy(float), part["pred"].to_numpy(float), part["subject"].to_numpy(),
                           n_bootstrap=args.n_bootstrap, seed=args.seed)
        rows.append({"scope": scope, "n_trials": len(part), **d})
    write_dataframe(pd.DataFrame(rows), out_dir / "rt_free_decoder.csv")
    write_dataframe(dec[["subject", "block_key", "trial_index", "fold", "rt_s", "pred"]], out_dir / "rt_free_predictions.parquet")
    log.info("RT-free decoder done")

    # development
    children = child_table(frame).merge(demo, on="subject", how="left")
    grid = np.arange(5.5, 21.5, 0.5)
    development, curves = [], []
    for measure in (*BEHAVIOUR, *MEASURES):
        s, c = age_trajectory(children, measure, grid, args.n_bootstrap, args.seed)
        development.append(s)
        curves.append(c)
    write_dataframe(pd.DataFrame(development), out_dir / "age_effects.csv")
    write_dataframe(pd.concat(curves, ignore_index=True), out_dir / "age_trajectories.csv")
    valid = frame[frame["valid_response"].astype(bool)].dropna(subset=["cpp_slope_resp", "log_rt"])
    z = valid.assign(m=(valid["cpp_slope_resp"] - valid["cpp_slope_resp"].mean()) / valid["cpp_slope_resp"].std())
    slopes = child_slopes(z, "log_rt", "m", within=("trial_index",)).merge(demo, on="subject").dropna(subset=["age", "sex_male"])
    knots = np.quantile(slopes["age"], np.linspace(0, 1, 5))
    design = np.column_stack([np.ones(len(slopes)), natural_basis(slopes["age"].to_numpy(), knots), slopes["sex_male"]])
    fit = meta_regression(slopes["slope"].to_numpy(), slopes["se"].to_numpy(), design,
                          ("intercept", *[f"age{i}" for i in range(design.shape[1] - 2)], "sex_male"))
    age_idx = slice(1, design.shape[1] - 1)
    cov = np.linalg.inv((design.T / (slopes["se"].to_numpy() ** 2 + fit.tau2)) @ design)
    b = fit.coef[age_idx]
    wald = float(b @ np.linalg.solve(cov[age_idx, age_idx], b))
    summary["cpp_coupling_by_age"] = {"n_children": len(slopes), "wald_chi2": wald, "df": len(b),
                                         "p": float(stats.chi2.sf(wald, len(b)))}
    write_dataframe(children, out_dir / "child_level_measures.csv")
    log.info("development done")

    # psychopathology at the child level
    traits = load_traits(participants)
    data = children.merge(traits, on="subject").merge(
        measures.drop_duplicates("subject")[["subject", "release"]], on="subject", how="left")
    psychopathology = []
    for measure in (*BEHAVIOUR, *MEASURES):
        part = data.dropna(subset=[measure, "age", "sex_male", *TRAIT_COLUMNS])
        knots = np.quantile(part["age"], np.linspace(0, 1, 5))
        releases = pd.get_dummies(part["release"], drop_first=True, dtype=float).to_numpy()
        cov = np.column_stack([natural_basis(part["age"].to_numpy(), knots), part["sex_male"], releases])
        y = part[measure].to_numpy(float)
        y = (y - y.mean()) / y.std()
        beta, se, t, p = trait_permutation(y, part[list(TRAIT_COLUMNS)].to_numpy(float), cov, args.n_permutations,
                                           args.seed)
        for i, trait in enumerate(TRAIT_COLUMNS):
            psychopathology.append({"measure": measure, "trait": trait, "n_children": len(part), "beta_sd_per_sd": beta[i],
                                    "se": se[i], "t": t[i], "p_permutation": p[i]})
    psychopathology = pd.DataFrame(psychopathology)
    psychopathology["p_bh"], psychopathology["fdr_reject"] = fdr_correct(psychopathology["p_permutation"].to_numpy())
    write_dataframe(psychopathology, out_dir / "traits.csv")
    control = psychopathology[(psychopathology["measure"] == "sd_log_rt") & (psychopathology["trait"] == "attention")].iloc[0]
    summary["positive_control"] = {"beta": float(control["beta_sd_per_sd"]), "p": float(control["p_permutation"]),
                                      "p_bh": float(control["p_bh"])}
    write_json(summary, out_dir / "summary.json")
    log.info("written to %s", out_dir)
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
