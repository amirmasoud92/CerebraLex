"""H2: trait moderation of the within-child state-behaviour coupling.

The state measure is regenerated out of fold for every trial, misses included (``state_models_and_trait_moderation.decoded_state``).
Stage 1 estimates each child's slope of the outcome (log RT of valid responses; lapse of all trials) on the state;
stage 2 is a random-effects meta-regression of the slopes on the four standardised traits with age spline, sex,
handedness and release, with p-values from 5,000 joint permutations of the traits and Benjamini-Hochberg over the
8 tests.

Variants:
  primary             pre-stimulus band-power decoded state. Also runs the R5 replication, the secondary state
                      (raw plus LGSSM decoder; log RT only), the two scale robustness estimates, the mixed model
                      once, and the comparison of excluded and constant-outcome children with the analysed children.
  s_antic             primary analysis without fast guesses: the cohort drops valid responses below the fast-guess
                      cutoff and the decoder is refitted on that cohort.
  challenge_contrast  descriptive contrast: decoded state of the post-target time-course model, which is dominated
                      by the response itself; not a state measure.
  fingerprint_orthogonal  the state is rederived from band-power features standardised within child, so the decoder
                      cannot identify the child.

Writes results/trait_state_moderation/<variant>/<run>/. ``--mixed-only RUN_DIR`` refits the secondary mixed model from a finished
run's state table and writes it beside the original file (which is kept).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from common.config import config_fingerprint, load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import (
    BenchmarkConfig,
    ComputeConfig,
    DerivativesConfig,
    PreprocessingConfig,
    TrialConfig,
)
from decoding.benchmark import BenchmarkRunner, demographics_table, load_window_data
from decoding.splits import SplitAssignment
from state_models_and_trait_moderation.decoded_state import out_of_fold_state, standardised_within_child
from state_models_and_trait_moderation.mixed_models import TraitStateModel, natural_spline_basis
from state_models_and_trait_moderation.permutation import fdr_correct
from state_models_and_trait_moderation.trait_state import (
    TRAIT_COLUMNS,
    child_slopes,
    constant_outcome_children,
    load_traits,
    logit_scale_slopes,
    partial_correlation_slopes,
    permutation_pvalues,
)

OUTCOMES = {"log_rt": ("log_rt_centred", True), "lapse": ("lapse", False)}   # (column, valid responses only)
VARIANTS = {   # window, state decoder, description (fingerprint_orthogonal: child-standardised features)
    "primary": ("prestim", "spectral", "H2 trait moderation of the state coupling"),
    "s_antic": ("prestim", "spectral", "H2 robustness: fast guesses removed (S-antic)"),
    "challenge_contrast": ("challenge", "timecourse",
                           "H2 descriptive contrast: challenge-window decoded state (not a state measure)"),
    "fingerprint_orthogonal": ("prestim", "spectral",
                               "H2 robustness: state from fingerprint-orthogonalised (child-standardised) features"),
}
KEY = ["subject", "block_key", "trial_index"]


def child_covariates(table: pd.DataFrame) -> pd.DataFrame:
    """Child-level covariates: natural spline of age (4 df), sex, handedness, release dummies."""
    children = table.drop_duplicates("subject").reset_index(drop=True)
    age = pd.to_numeric(children["age"], errors="coerce")
    keep = age.notna()
    children = children[keep].reset_index(drop=True)
    spline = natural_spline_basis(age[keep].to_numpy(float), df=4)
    out = pd.DataFrame(spline, columns=[f"age_ns{i}" for i in range(spline.shape[1])])
    out.insert(0, "subject", children["subject"].to_numpy())
    out["sex_male"] = children["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).to_numpy()
    out["ehq_total"] = pd.to_numeric(children["ehq_total"], errors="coerce").to_numpy()
    releases = pd.get_dummies(children["release"], prefix="release", drop_first=True, dtype=float)
    return pd.concat([out, releases.reset_index(drop=True)], axis=1).dropna().reset_index(drop=True)


def outcome_rows(table: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, str]:
    column, valid_only = OUTCOMES[outcome]
    return (table[table["valid_response"]] if valid_only else table), column


def two_stage(table: pd.DataFrame, traits: pd.DataFrame, covariates: pd.DataFrame, outcome: str,
              state: str, n_permutations: int, seed: int,
              rescale: Callable[[pd.DataFrame], pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    part, column = outcome_rows(table, outcome)
    slopes = child_slopes(part.dropna(subset=[state]), column, state)
    if rescale is not None:
        slopes = rescale(slopes)
    result = permutation_pvalues(slopes, traits, covariates, n_permutations=n_permutations, seed=seed)
    analysed = set(slopes["subject"]) & set(traits["subject"]) & set(covariates["subject"])
    n_trials = int(slopes.loc[slopes["subject"].isin(analysed), "n_trials"].sum())
    return result.assign(outcome=outcome, state=state, n_trials=n_trials), slopes


def with_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    adjusted, reject = fdr_correct(frame["p_permutation"].to_numpy(), q=0.05)
    return frame.assign(p_bh=adjusted, fdr_reject=reject)


def run_family(name: str, table: pd.DataFrame, traits: pd.DataFrame, outcomes: tuple[str, ...], state: str,
               args: argparse.Namespace, rescale: dict[str, Callable] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One family of tests (all traits for the given outcomes), BH-corrected within the family."""
    covariates = child_covariates(table)
    results, slopes = [], []
    for outcome in outcomes:
        res, sl = two_stage(table, traits, covariates, outcome, state, args.n_permutations, args.seed,
                            (rescale or {}).get(outcome))
        results.append(res)
        slopes.append(sl.assign(analysis=name, outcome=outcome))
    return with_fdr(pd.concat(results, ignore_index=True)).assign(analysis=name), pd.concat(slopes, ignore_index=True)


def within_child_r(table: pd.DataFrame, state: str) -> float:
    """Pooled within-child correlation of the state with log RT (valid responses)."""
    part = table[table["valid_response"]].dropna(subset=[state, "log_rt_centred"])
    centre = lambda c: part[c] - part.groupby("subject")[c].transform("mean")  # noqa: E731
    return float(np.corrcoef(centre(state), centre("log_rt_centred"))[0, 1])


def lgssm_state(path: str, window: str) -> pd.DataFrame:
    """Raw plus LGSSM (state_space_models.py) out-of-fold decoded RT, standardised within child (valid responses)."""
    preds = pd.read_parquet(path, columns=[*KEY, "window", "target", "model", "y_pred"])
    preds = preds[(preds["window"] == window) & (preds["target"] == "rt_s") & (preds["model"] == "raw_lgssm")]
    if preds.empty or preds.duplicated(KEY).any():
        raise ValueError(f"{path}: no unique raw_lgssm RT predictions for window {window}")
    out = preds[KEY].copy()
    out["state_lgssm"] = standardised_within_child(preds["y_pred"], preds["subject"]).to_numpy()
    return out


def compare_traits(traits: pd.DataFrame, reference: set, other: set, group: str) -> list[dict]:
    values = traits.set_index("subject")
    rows = []
    for trait in TRAIT_COLUMNS:
        a = values.loc[values.index.isin(reference), trait]
        b = values.loc[values.index.isin(other), trait]
        if len(b) > 1:
            rows.append({"group": group, "trait": trait, "analysed_n": len(a), "group_n": len(b),
                         "analysed_mean_z": float(a.mean()), "group_mean_z": float(b.mean()),
                         "mann_whitney_p": float(mannwhitneyu(a, b).pvalue)})
    return rows


def mixed_models(table: pd.DataFrame, traits: pd.DataFrame, log) -> pd.DataFrame:
    out = []
    for outcome in OUTCOMES:
        part, column = outcome_rows(table, outcome)
        data = part.merge(traits, on="subject").copy()
        data["sex_male"] = data["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
        data["ehq_total"] = pd.to_numeric(data["ehq_total"], errors="coerce")
        data = data.rename(columns={column: "outcome_value"}).dropna(
            subset=["outcome_value", "state", "age", "sex_male", "ehq_total", "artifact_fraction"])
        start = time.perf_counter()
        result = TraitStateModel(outcome="outcome_value", state="state").fit(data)
        log.info("mixed model %s: %d trials, %d children, converged %s, %.0f s", outcome, result.n_trials,
                 result.n_subjects, result.converged, time.perf_counter() - start)
        out.append(result.interactions.assign(outcome=outcome, converged=result.converged,
                                              optimizer=result.optimizer, log_likelihood=result.log_likelihood,
                                              n_trials=result.n_trials, n_children=result.n_subjects,
                                              notes="; ".join(result.notes)))
    return pd.concat(out, ignore_index=True)


def refit_mixed(run_dir: Path, args: argparse.Namespace, paths: ProjectPaths, log) -> int:
    """Refit the secondary mixed model of a finished primary run; the original file is kept."""
    commit = git_commit(paths.root)
    stamp = utc_timestamp()
    table = pd.read_parquet(run_dir / "state_table.parquet")
    traits = load_traits(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    mixed = mixed_models(table, traits, log)
    write_dataframe(mixed, run_dir / f"secondary_mixed_model_refit_{stamp}.csv")
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, run_dir / f"refit_provenance_{stamp}.json")
    log.info("mixed model refit written to %s", run_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="primary")
    parser.add_argument("--benchmark-config", default="decoding/benchmark.yaml")
    parser.add_argument("--store-name", default="windows-v1")
    parser.add_argument("--state-model-run", default="results/states-v1/predictions.parquet")
    parser.add_argument("--fast-guess", default="results/fast_guess_cutoff/cutoff.json")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--mixed-only", metavar="RUN_DIR", help="refit the mixed model of a finished primary run")
    args = parser.parse_args()
    window_name, state_model, _description = VARIANTS[args.variant]
    paths = ProjectPaths.discover()
    log = configure_logging("trait-state-moderation", paths.logs)
    if args.mixed_only:
        return refit_mixed(Path(args.mixed_only), args, paths, log)
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / args.benchmark_config, BenchmarkConfig)
    if args.variant == "s_antic":
        cutoff = json.loads((paths.root / args.fast_guess).read_text(encoding="utf-8"))["cutoff_s"]
        bcfg = bcfg.model_copy(update={"min_rt_s": float(cutoff)})
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    trial_cfg = load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig)
    compute = load_config(paths.root / "common" / "compute.yaml", ComputeConfig)
    windows = {w.name: load_window_data(deriv.root / args.store_name, w, preproc.resample_hz)
               for w in trial_cfg.windows if w.name == window_name}
    split = SplitAssignment.load(deriv.root / "splits" / bcfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    import mne

    positions = np.array(list(mne.channels.make_standard_montage(preproc.montage).get_positions()["ch_pos"].values()))
    trials = pd.read_parquet(deriv.root / "trials-v1" / "trials_ccd.parquet")
    settings = {"variant": args.variant, "window": window_name, "state_model": state_model,
                "benchmark_config": config_fingerprint(bcfg), "min_rt_s": bcfg.min_rt_s,
                "child_standardise": args.variant == "fingerprint_orthogonal",
                "n_permutations": args.n_permutations, "seed": args.seed,
                "state_model_run": args.state_model_run if args.variant == "primary" else None}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    out_dir = paths.results / "trait_state_moderation" / args.variant / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    write_json(settings, out_dir / f"settings_{fingerprint}.json")
    bench = BenchmarkRunner(bcfg, compute, trials, windows, split, demographics_table(participants), positions,
                            out_dir, deriv.root / "features" / args.store_name)
    start = time.perf_counter()
    table = out_of_fold_state(bench, window_name, state_model,
                              child_standardise=args.variant == "fingerprint_orthogonal")
    log.info("decoded state: %d trials, %d children, %.0f s", len(table), table["subject"].nunique(),
             time.perf_counter() - start)
    block_qc = pd.read_parquet(deriv.root / f"qc-{deriv.preprocessing_version}" / "block_qc.parquet",
                               columns=["block_key", "artifact_variance_fraction"])
    table = table.merge(block_qc.rename(columns={"artifact_variance_fraction": "artifact_fraction"}),
                        on="block_key", how="left", validate="many_to_one")
    traits = load_traits(participants)
    summary = {"variant": args.variant, "window": window_name, "state_model": state_model,
               "min_rt_s": bcfg.min_rt_s, "n_trials": len(table), "n_children": int(table["subject"].nunique()),
               "n_valid_responses": int(table["valid_response"].sum()),
               "within_child_r_state_log_rt": within_child_r(table, "state")}
    families = [("primary", table, OUTCOMES.keys(), "state", None)]
    if args.variant == "primary":
        lgssm = lgssm_state(str(paths.root / args.state_model_run), window_name)
        table = table.merge(lgssm, on=KEY, how="left", validate="one_to_one")
        missing = int(table.loc[table["valid_response"], "state_lgssm"].isna().sum())
        extra = int(table.loc[~table["valid_response"], "state_lgssm"].notna().sum())
        if missing or extra or len(lgssm) != int(table["valid_response"].sum()):
            raise ValueError(f"state-model predictions do not match the cohort: {missing} valid responses "
                             f"without a state, {extra} misses with one, {len(lgssm)} predictions")
        summary["within_child_r_state_lgssm_log_rt"] = within_child_r(table, "state_lgssm")
        families += [("replication_R5", table[table["release"] == "R5"], OUTCOMES.keys(), "state", None),
                     ("secondary_lgssm", table, ("log_rt",), "state_lgssm", None),
                     ("lapse_logit_scale", table, ("lapse",), "state", {"lapse": logit_scale_slopes}),
                     ("log_rt_partial_correlation", table, ("log_rt",), "state",
                      {"log_rt": partial_correlation_slopes})]
    write_dataframe(table, out_dir / "state_table.parquet")
    results, slopes = [], []
    for name, data, outcomes, state, rescale in families:
        res, sl = run_family(name, data, traits, tuple(outcomes), state, args, rescale)
        results.append(res)
        slopes.append(sl)
        log.info("family %s done (%d tests)", name, len(res))
    results = pd.concat(results, ignore_index=True)
    write_dataframe(results, out_dir / "two_stage_results.csv")
    write_dataframe(pd.concat(slopes, ignore_index=True), out_dir / "child_slopes.parquet")
    included = set(table["subject"])
    comparison = compare_traits(traits, included, set(trials["subject"]) - included, "excluded_from_cohort")
    constant = {}
    for outcome in OUTCOMES:
        part, column = outcome_rows(table, outcome)
        children = set(constant_outcome_children(part, column))
        constant[outcome] = len(children)
        comparison += compare_traits(traits, included - children, children, f"constant_{outcome}")
    summary["constant_outcome_children"] = constant
    write_dataframe(pd.DataFrame(comparison), out_dir / "excluded_children.csv")
    mixed = None
    if args.variant == "primary":
        mixed = mixed_models(table, traits, log)
        write_dataframe(mixed, out_dir / "secondary_mixed_model.csv")
    summary["elapsed_s"] = round(time.perf_counter() - start)
    write_json(summary, out_dir / "summary.json")
    log.info("H2 (%s) written to %s", args.variant, out_dir)
    print(json.dumps({"out_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
