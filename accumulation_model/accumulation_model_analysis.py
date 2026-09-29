"""What the per-child accumulation-model fits of accumulation_model_fit.py say.

(i)   model comparison: group sums of NLL differences with child-bootstrap intervals, BIC choices, and the
      model-implied RT variance decomposition (rate, onset and within-trial shares) from the full fits;
(ii)  the CPP build-up ratio predicted from behaviour alone: simulated decision variables of every child of the
      onset/rate analysis, put through its response-locked rate estimator (full, rate-only and onset-only fits),
      against the observed slow/fast ratio;
(iii) development of the none-model parameters with age;
(iv)  rank partial correlations (sex and an age spline removed): drift gain k with the normalised CPP slope and the
      EZ drift, lapse with CBCL attention (HBN only).
Writes results/accumulation_model_analysis/<tag>/<run>/.
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
from scipy.stats import rankdata, spearmanr

from accumulation_model import accumulation as acc
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.scripts import script_path
from state_models_and_trait_moderation.mixed_models import natural_spline_basis

SFREQ = 250.0
PATH_DT = 0.001
MIN_RT_S, MAX_RT_S = 0.65, 2.4                     # eligibility of cpp_onset_rate.py (cpp_slope_resp)
AGE_BINS = [5, 8, 11, 14, 17, 22]
COMPARISONS = {"onset_minus_rate": ("onset", "rate"), "none_minus_full": ("none", "full"),
               "rate_minus_full": ("rate", "full"), "onset_minus_full": ("onset", "full")}


def params_of(row: pd.Series, model: str) -> dict:
    return {p: float(row[f"{model}_{p}"]) for p in acc.PARAMS}


def bootstrap_sum(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[float]:
    draws = [values[rng.integers(0, len(values), len(values))].sum() for _ in range(n_boot)]
    return np.percentile(draws, [2.5, 97.5]).tolist()


def model_comparison(fits: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, (a, b) in COMPARISONS.items():
        delta = (fits[f"{a}_nll"] - fits[f"{b}_nll"]).to_numpy(float)
        ci = bootstrap_sum(delta, n_boot, rng)
        out[name] = {"sum": float(delta.sum()), "ci": ci, "per_child": float(delta.mean()),
                     "per_child_ci": [c / len(delta) for c in ci],
                     "share_children_positive": float((delta > 0).mean())}
    # per-child onset - rate under each generating model of the recovery study
    out["onset_minus_rate"]["simulated_per_child_reference"] = {"rate": 0.27, "onset": -0.21, "none": 0.11}
    choice = fits[[f"{m}_bic" for m in acc.MODELS]].idxmin(axis=1).str.replace("_bic", "", regex=False)
    out["bic_choice_shares"] = choice.value_counts(normalize=True).round(4).to_dict()
    out["sum_bic"] = {m: float(fits[f"{m}_bic"].sum()) for m in acc.MODELS}
    return out


def rt_variance(params: dict, step: float = 0.005) -> float:
    """Variance of the RT of diffusion responses before the deadline, from the model's densities."""
    grid = np.arange(step / 2, acc.DEADLINE_S, step)
    f_up, f_low, _ = acc.response_densities(params, grid)
    dens = f_up + f_low
    if dens.sum() <= 0:
        return float("nan")
    w = dens / dens.sum()
    mean = float((w * grid).sum())
    return float((w * (grid - mean) ** 2).sum())


def variance_shares(subject: str, params: dict) -> dict:
    full = rt_variance(params)
    no_rate = rt_variance(params | {"sv": 0.0})
    no_onset = rt_variance(params | {"st": 0.0})
    neither = rt_variance(params | {"sv": 0.0, "st": 0.0})
    return {"subject": subject, "var_full": full, "rate_share": 1 - no_rate / full,
            "onset_share": 1 - no_onset / full, "within_trial_share": neither / full}


def build_rate(paths: np.ndarray, rt: np.ndarray) -> tuple[float, float]:
    """The rate estimator of cpp_onset_rate.py on simulated decision variables: RT tertiles, response-locked tertile mean, least-squares
    slope from -0.35 to -0.05 s at 250 Hz (the sample grid of cpp_onset_rate.child_tertile_measures)."""
    n_hist = paths.shape[1]
    offsets = np.arange(-int(round(0.35 * SFREQ)), int(round(0.1 * SFREQ)) + 1)
    t = offsets / SFREQ
    build = (t >= -0.35) & (t <= -0.05)
    idx = (n_hist - 1) + np.round(t[build] / PATH_DT).astype(int)
    tertile = pd.qcut(pd.Series(rt).rank(method="first"), 3, labels=False).to_numpy()
    rates = []
    for value in (0, 2):
        mean_path = np.nanmean(paths[tertile == value][:, idx], axis=0)
        rates.append(float(np.polyfit(t[build], mean_path, 1)[0]))
    return rates[0], rates[1]


def cpp_child(subject: str, index: int, row: pd.Series, n_sim: int, seed: int) -> dict:
    out = {"subject": subject}
    for m_index, model in enumerate(("full", "rate", "onset")):
        params = params_of(row, model)
        rng = np.random.default_rng([seed, index, m_index])
        sim = acc.simulate(params, n_sim, rng, dt=PATH_DT, keep_paths=True, path_window_s=0.6)
        rt = sim["rt"]
        ok = sim["diffusion"] & np.isfinite(rt) & (rt >= MIN_RT_S) & (rt <= MAX_RT_S)
        out[f"{model}_n_eligible"] = int(ok.sum())
        if ok.sum() < 30:
            continue
        for label, scale in (("norm", params["a"]), ("raw", 1.0)):
            fast, slow = build_rate(sim["paths"][ok] / scale, rt[ok])
            out[f"{model}_{label}_fast_rate"], out[f"{model}_{label}_slow_rate"] = fast, slow
    return out


def ratio_of_means(fast: np.ndarray, slow: np.ndarray, n_boot: int, seed: int) -> dict:
    keep = np.isfinite(fast) & np.isfinite(slow)
    fast, slow = fast[keep], slow[keep]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(fast), len(fast))
        draws.append(slow[i].mean() / fast[i].mean())
    return {"n_children": int(keep.sum()), "ratio": float(slow.mean() / fast.mean()),
            "ci": np.percentile(draws, [2.5, 97.5]).tolist()}


def observed_ratio(onset_rate_dir: Path, paths: ProjectPaths, n_boot: int, seed: int) -> dict:
    """The observed slow/fast rate ratio, with the calibration script's own estimator and interval."""
    spec = importlib.util.spec_from_file_location("cpp_onset_rate_calibration", script_path("cpp_onset_rate_calibration"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.share_and_ratio(pd.read_csv(onset_rate_dir / "child_tertiles.csv"), n_boot, seed)
    return {"n_children": result["n_children"], "ratio": result["rate_ratio"], "ci": result["rate_ratio_ci"]}


def rank_partial(x: np.ndarray, y: np.ndarray, covariates: np.ndarray | None) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if covariates is not None:
        design = np.column_stack([np.ones(len(rx)), covariates])
        rx = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
        ry = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_with_ci(frame: pd.DataFrame, x: str, y: str, spline: bool, n_boot: int, seed: int) -> dict:
    f = frame.dropna(subset=[x, y, "age", "sex_male"]).reset_index(drop=True)

    def covariates(part: pd.DataFrame) -> np.ndarray:
        cols = [part["sex_male"].to_numpy(float)]
        if spline:
            cols.append(natural_spline_basis(part["age"].to_numpy(float), df=4))
        return np.column_stack(cols)

    estimate = rank_partial(f[x].to_numpy(float), f[y].to_numpy(float), covariates(f))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        part = f.iloc[rng.integers(0, len(f), len(f))].reset_index(drop=True)
        draws.append(rank_partial(part[x].to_numpy(float), part[y].to_numpy(float), covariates(part)))
    return {"n": len(f), "r": estimate, "ci": np.percentile(draws, [2.5, 97.5]).tolist()}


def demographics(paths: ProjectPaths, people_csv: str | None) -> pd.DataFrame:
    if people_csv:
        frame = pd.read_csv(people_csv)
        frame["subject"] = frame["subject"].astype(str)
    else:
        participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
        frame = participants.drop_duplicates("participant_id").assign(
            subject=lambda f: f["participant_id"].astype(str).str.replace("^sub-", "", regex=True))
    frame = frame.assign(age=pd.to_numeric(frame["age"], errors="coerce"),
                         sex_male=frame["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}))
    keep = ["subject", "age", "sex_male"] + (["attention"] if "attention" in frame.columns else [])
    if "attention" in keep:
        frame["attention"] = pd.to_numeric(frame["attention"], errors="coerce")
    return frame[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", required=True, help="results/model/<ts>_fits_<tag> directory")
    parser.add_argument("--tag", default="hbn")
    parser.add_argument("--onset-rate", default="results/cpp_onset_rate/hbn")
    parser.add_argument("--normalised", default="results/cpp_normalised_amplitude", help="HBN only; 'none' to skip")
    parser.add_argument("--people", default=None, help="participants CSV with subject, age, sex (MIPDB)")
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--out-dir", default=None, help="default results/model/<utc>_analysis_<tag>")
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    fits = pd.read_csv(Path(args.fits) / "fits.csv", dtype={"subject": str})
    summary = {"git_commit": git_commit(paths.root), "argv": sys.argv, "fits": args.fits, "n_children": len(fits)}
    # Rules fixed before the analysis: (i) and (ii) use every child (with nested starts the
    # success flag of rate, onset and full is uninformative); (iii) and (iv) use the none-model parameters of
    # children whose none fit converged.
    summary["converged_share"] = {m: float(fits[f"{m}_converged"].mean()) for m in acc.MODELS}
    none_ok = fits["none_converged"].astype(bool)
    summary["n_none_converged"] = int(none_ok.sum())
    summary["at_bound"] = {m: {p: int(((fits[f"{m}_{p}"] <= acc.BOUNDS[p][0] + 1e-6)
                                       | (fits[f"{m}_{p}"] >= acc.BOUNDS[p][1] - 1e-6)).sum())
                               for p in acc.PARAMS if p not in acc.MODELS[m]}
                           for m in ("none", "full")}
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "accumulation_model_analysis" / args.tag / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)

    # (i) model comparison and variance decomposition
    summary["comparison"] = model_comparison(fits, args.n_bootstrap, args.seed)
    shares = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(
        delayed(variance_shares)(r["subject"], params_of(r, "full")) for _, r in fits.iterrows()))
    write_dataframe(shares, out_dir / "variance_shares.csv")
    summary["variance_shares"] = {c: shares[c].quantile([0.25, 0.5, 0.75]).round(4).tolist()
                                  for c in ("rate_share", "onset_share", "within_trial_share")}

    # (ii) CPP build-up ratio predicted from behaviour
    onset_rate = pd.read_csv(Path(args.onset_rate) / "child_tertiles.csv", dtype={"subject": str})
    cohort = fits[fits["subject"].isin(set(onset_rate["subject"]))].reset_index(drop=True)
    cpp = pd.DataFrame(Parallel(n_jobs=args.n_jobs)(
        delayed(cpp_child)(r["subject"], i, r, args.n_sim, args.seed) for i, r in cohort.iterrows()))
    write_dataframe(cpp, out_dir / "cpp_prediction.csv")
    summary["cpp_ratio_observed"] = observed_ratio(Path(args.onset_rate), paths, args.n_bootstrap, args.seed)
    summary["cpp_ratio_predicted"] = {}
    low, high = summary["cpp_ratio_observed"]["ci"]
    for model in ("full", "rate", "onset"):
        for label in ("norm", "raw"):
            fast, slow = f"{model}_{label}_fast_rate", f"{model}_{label}_slow_rate"
            if fast not in cpp:
                continue
            res = ratio_of_means(cpp[fast].to_numpy(float), cpp[slow].to_numpy(float), args.n_bootstrap, args.seed)
            res["inside_observed_interval"] = bool(low <= res["ratio"] <= high)
            summary["cpp_ratio_predicted"][f"{model}_{label}"] = res

    # (iii) development and (iv) links to the EEG and to attention
    demo = demographics(paths, args.people)
    table = fits[none_ok].reset_index(drop=True).merge(demo, on="subject", how="left")
    development = {}
    bins = pd.cut(table["age"], AGE_BINS, right=False)
    for p in ("k", "a", "te", "lapse", "guess"):
        col = f"none_{p}"
        f = table.dropna(subset=[col, "age"])
        rho = spearmanr(f[col], f["age"]).statistic
        development[p] = {"spearman_age": float(rho),
                          "partial_age_given_sex": partial_with_ci(table.assign(age_copy=table["age"]), col,
                                                                   "age_copy", False, args.n_bootstrap, args.seed),
                          "median_by_age_bin": table.groupby(bins, observed=False)[col].median().round(4)
                                                    .rename(index=str).to_dict()}
    summary["development"] = development
    links = {}
    if args.normalised and args.normalised != "none":
        normalised = pd.read_csv(Path(args.normalised) / "children.csv", dtype={"subject": str})
        linked = table.merge(normalised[["subject", "cpp_slope_resp_norm", "ez_drift"]], on="subject", how="left")
        for other in ("cpp_slope_resp_norm", "ez_drift"):
            links[f"k_with_{other}"] = partial_with_ci(linked, "none_k", other, True, args.n_bootstrap, args.seed)
    if "attention" in table.columns:
        links["lapse_with_attention"] = partial_with_ci(table, "none_lapse", "attention", True,
                                                        args.n_bootstrap, args.seed)
    summary["links"] = links
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
