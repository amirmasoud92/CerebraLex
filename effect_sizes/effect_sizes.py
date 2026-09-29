"""Effect sizes in interpretable units, the predictable ceiling, lapses and reliability.

(a) units: for the out-of-fold RT predictions of record, the within-child r, r^2, ms of RT per within-child SD of the
    prediction, and the mean RT difference between each child's fifth of trials predicted slowest and fastest;
    behavioural yardsticks (previous RT, time in block, foreperiod) in the same units;
(b) ceiling: the explainable share of within-child RT variance from the accumulation model's full fits (1 - pooled
    within-trial share), which bounds predictors defined before the stimulus, with a parametric-bootstrap validation;
(c) lapses: pooled and within-child AUROC of the lapse predictions;
(d) reliability: random split halves (stratified by block) of child-level behaviour and couplings, Spearman-Brown.
No model is refitted. ``--parts`` selects the parts; the paper's run used
``--parts units ceiling ceiling_validation lapses reliability``. Writes results/effect_sizes/<run>/.
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

from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.scripts import script_path
from decoding.metrics import auroc, subject_bootstrap

ROOTS = {"HBN": data_location("HBN"), "MIPDB": data_location("MIPDB"),
         "R10R11": data_location("R10R11")}
RUNS = {  # name: (run directory, window, model)
    "HBN": {"prestim_spectral": ("results/benchmark-v1/prestim", "prestim", "spectral"),
            "prestim_eegnet": ("results/benchmark-v1/prestim_eegnet", "prestim", "eegnet"),
            "prestim_timecourse": ("results/benchmark-v1/prestim_timecourse", "prestim", "timecourse"),
            "post_timecourse": ("results/benchmark-v1/challenge_timecourse", "challenge", "timecourse"),
            "post_eegnet": ("results/benchmark-v1/challenge", "challenge", "eegnet"),
            "prestim_demographics": ("results/benchmark-v1/prestim", "prestim", "demographics"),
            "post_demographics": ("results/benchmark-v1/challenge", "challenge", "demographics")},
    "MIPDB": {"prestim_spectral": ("results/external_validation/mipdb", "prestim", "spectral"),
              "prestim_eegnet": ("results/external_validation/mipdb_eegnet", "prestim", "eegnet"),
              "prestim_timecourse": ("results/external_validation/mipdb", "prestim", "timecourse"),
              "post_timecourse": ("results/external_validation/mipdb", "challenge", "timecourse"),
              "post_eegnet": ("results/external_validation/mipdb_eegnet", "challenge", "eegnet"),
              "prestim_demographics": ("results/external_validation/mipdb", "prestim", "demographics"),
              "post_demographics": ("results/external_validation/mipdb", "challenge", "demographics")},
    "R10R11": {"prestim_spectral": ("results/external_validation/r10_r11", "prestim", "spectral"),
               "prestim_eegnet": ("results/external_validation/r10_r11_eegnet", "prestim", "eegnet"),
               "prestim_timecourse": ("results/external_validation/r10_r11", "prestim", "timecourse"),
               "post_timecourse": ("results/external_validation/r10_r11", "challenge", "timecourse"),
               "post_eegnet": ("results/external_validation/r10_r11_eegnet", "challenge", "eegnet"),
               "prestim_demographics": ("results/external_validation/r10_r11", "prestim", "demographics"),
               "post_demographics": ("results/external_validation/r10_r11", "challenge", "demographics")},
}
RT_DECODERS = ("prestim_spectral", "prestim_eegnet", "prestim_timecourse", "post_timecourse", "post_eegnet")
LAPSE_DECODERS = (*RT_DECODERS, "prestim_demographics", "post_demographics")
PRE_STIMULUS = {"prestim_spectral", "prestim_eegnet", "prestim_timecourse", "horizon_C"}
HORIZON, HORIZON_T, HORIZON_MARGIN = "results/prediction_horizon", (0.9, 1.1, 1.3), 0.1
MODEL_ANALYSIS = {"HBN": "results/accumulation_model_analysis/hbn", "MIPDB": "results/accumulation_model_analysis/mipdb",
                  "R10R11": "results/accumulation_model_analysis/confirm"}
HBN_FITS = "results/accumulation_model_fit/hbn/fits.csv"
YARDSTICKS = {"previous_rt": "prev_rt_s", "time_in_block": "time_in_block_s", "foreperiod": "foreperiod_s"}
MIN_FIFTHS, MIN_RELIABILITY = 10, 20
N_BOOT, SEED = 2000, 12345
N_SPLITS, N_BOOT_REL = 200, 1000
N_CEILING_CHILDREN = 300
CEILING_TOLERANCE = 0.05


def load_predictions(paths: ProjectPaths, sample: str, name: str, target: str) -> pd.DataFrame:
    run, window, model = RUNS[sample][name]
    p = pd.read_parquet(paths.root / run / "predictions.parquet")
    p = p[(p["window"] == window) & (p["target"] == target) & (p["model"] == model)].copy()
    if sample == "HBN":
        p["scope"] = np.where(p["fold"] == -1, "holdout_R5", "cv")
    else:
        p["scope"] = sample
    return p[["subject", "block_key", "trial_index", "y_true", "y_pred", "scope"]].reset_index(drop=True)


def draw_counts(k: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)


def interval(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(low), float(high)


# ---------------------------------------------------------------- (a) units
def centred_sums(y: np.ndarray, x: np.ndarray, codes: np.ndarray, k: int):
    n = np.bincount(codes, minlength=k).astype(float)
    sx, sy = np.bincount(codes, x, k), np.bincount(codes, y, k)
    cxx = np.bincount(codes, x * x, k) - sx ** 2 / n
    cyy = np.bincount(codes, y * y, k) - sy ** 2 / n
    cxy = np.bincount(codes, x * y, k) - sx * sy / n
    return n, cxx, cyy, cxy


def fifths_contrast(y: np.ndarray, x: np.ndarray, subjects: np.ndarray) -> pd.Series:
    """Per child with at least MIN_FIFTHS trials: mean y of the fifth with the highest x minus that of the lowest."""
    f = pd.DataFrame({"s": subjects, "y": y, "x": x})
    n = f.groupby("s")["x"].transform("size")
    f = f[n >= MIN_FIFTHS].copy()
    rank = f.groupby("s")["x"].rank(method="first")
    f["fifth"] = np.floor(5 * (rank - 1) / f.groupby("s")["x"].transform("size")).astype(int)
    means = f.groupby(["s", "fifth"])["y"].mean().unstack()
    return means[4] - means[0]


def units(y_s: np.ndarray, x: np.ndarray, subjects: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Within-child r, r^2, ms per within-child SD of x, and the fifths contrast (ms), with child bootstraps."""
    y = np.asarray(y_s, dtype=float) * 1000.0
    x = np.asarray(x, dtype=float)
    codes, uniq = pd.factorize(pd.Series(subjects), sort=True)
    k = len(uniq)
    n, cxx, cyy, cxy = centred_sums(y, x, codes, k)
    fifths = fifths_contrast(y, x, np.asarray(subjects)).reindex(uniq)
    f_ok = fifths.notna().to_numpy()
    f_val = fifths.fillna(0.0).to_numpy()

    def stat(w):
        r = (w @ cxy) / np.sqrt((w @ cxx) * (w @ cyy))
        sd_y = np.sqrt((w @ cyy) / (w @ (n - 1)))
        wf = w * f_ok
        return np.array([r, r ** 2, r * sd_y, (wf @ f_val) / wf.sum(), sd_y])

    est = stat(np.ones(k))
    draws = np.stack([stat(w) for w in draw_counts(k, n_boot, seed)])
    out = {"n_children": k, "n_trials": len(y), "n_children_fifths": int(f_ok.sum())}
    for i, name in enumerate(("r_within", "r2_within", "ms_per_sd", "fifths_ms", "sd_rt_within_ms")):
        lo, hi = interval(draws[:, i])
        out.update({name: float(est[i]), f"{name}_ci_low": lo, f"{name}_ci_high": hi})
    return out


def units_table(paths: ProjectPaths, samples: list[str]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        for name in RT_DECODERS:
            p = load_predictions(paths, sample, name, "rt_s")
            for scope, part in p.groupby("scope", sort=True):
                rows.append({"sample": sample, "scope": scope, "decoder": name,
                             **units(part["y_true"].to_numpy(), part["y_pred"].to_numpy(), part["subject"].to_numpy())})
    h = pd.read_parquet(paths.root / HORIZON / "predictions.parquet")
    h = h[(h["margin"] == HORIZON_MARGIN) & h["decoder"].isin(["A", "C"]) & h["T"].isin(HORIZON_T)]
    keep = {"HBN": ("cv", "holdout_R5"), "MIPDB": ("MIPDB",), "R10R11": ("R10R11",)}
    for (scope, t, decoder), part in h.groupby(["scope", "T", "decoder"], sort=True):
        sample = next(s for s, scopes in keep.items() if scope in scopes)
        if sample in samples:
            rows.append({"sample": sample, "scope": scope, "decoder": f"horizon_{decoder}", "T": t,
                         **units(part["rt_s"].to_numpy(), part["pred"].to_numpy(), part["subject"].to_numpy())})
    return pd.DataFrame(rows)


def yardsticks_table(paths: ProjectPaths, samples: list[str]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        trials = pd.read_parquet(Path(ROOTS[sample]) / "trials-v1" / "trials_ccd.parquet",
                                 columns=["block_key", "trial_index", *YARDSTICKS.values()])
        p = load_predictions(paths, sample, "prestim_spectral", "rt_s")
        p = p.merge(trials, on=["block_key", "trial_index"], how="left", validate="one_to_one")
        for scope, part in p.groupby("scope", sort=True):
            for name, col in YARDSTICKS.items():
                sub = part.dropna(subset=[col])
                rows.append({"sample": sample, "scope": scope, "yardstick": name,
                             **units(sub["y_true"].to_numpy(), sub[col].to_numpy(float), sub["subject"].to_numpy())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (b) ceiling
def pooled_within_share(var_full: np.ndarray, within_share: np.ndarray) -> float:
    return float((within_share * var_full).sum() / var_full.sum())


def ceiling_observed(paths: ProjectPaths, samples: list[str]) -> dict:
    out = {}
    for sample in samples:
        v = pd.read_csv(paths.root / MODEL_ANALYSIS[sample] / "variance_shares.csv")
        pooled = pooled_within_share(v["var_full"].to_numpy(float), v["within_trial_share"].to_numpy(float))
        median = float(v["within_trial_share"].median())
        out[sample] = {"n_children": len(v), "pooled_within_share": pooled, "ceiling": 1 - pooled,
                       "median_within_share": median, "ceiling_from_median": 1 - median}
    return out


def simulate_refit(subject: str, params: dict, n_targets: int, index: int, model_analysis_path: str) -> dict:
    """One child of the parametric bootstrap: simulate from its full fit, refit all four models, and compute the
    variance shares of the generating and the refitted full-model parameters (accumulation_model_analysis.variance_shares)."""
    from accumulation_model import accumulation as acc

    spec = importlib.util.spec_from_file_location("accumulation_model_analysis", model_analysis_path)
    ma = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ma)
    rng = np.random.default_rng([SEED, index])
    sim = acc.simulate(params, n_targets, rng)
    fits = acc.fit_all(acc.ChildData.from_trials(sim["rt"], sim["correct"]))
    refit = {p: float(fits["full"][p]) for p in acc.PARAMS}
    gen, ref = ma.variance_shares(subject, params), ma.variance_shares(subject, refit)
    return {"subject": subject, "n_targets": n_targets, "gen_var_full": gen["var_full"],
            "gen_within_share": gen["within_trial_share"], "ref_var_full": ref["var_full"],
            "ref_within_share": ref["within_trial_share"], "ref_converged": bool(fits["full"]["converged"]),
            **{f"gen_{p}": v for p, v in params.items()}, **{f"ref_{p}": v for p, v in refit.items()}}


def ceiling_validation(paths: ProjectPaths, n_jobs: int, n_children: int = N_CEILING_CHILDREN) -> tuple[pd.DataFrame, dict]:
    from accumulation_model import accumulation as acc

    fits = pd.read_csv(paths.root / HBN_FITS, dtype={"subject": str})
    chosen = fits.sample(n=n_children, random_state=SEED).reset_index(drop=True)
    path = str(script_path("accumulation_model_analysis"))
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(simulate_refit)(r["subject"], {p: float(r[f"full_{p}"]) for p in acc.PARAMS}, int(r["n_targets"]), i, path)
        for i, r in chosen.iterrows())
    table = pd.DataFrame(rows)
    ok = table[["gen_var_full", "gen_within_share", "ref_var_full", "ref_within_share"]].notna().all(axis=1)
    t = table[ok]
    gen = pooled_within_share(t["gen_var_full"].to_numpy(), t["gen_within_share"].to_numpy())
    ref = pooled_within_share(t["ref_var_full"].to_numpy(), t["ref_within_share"].to_numpy())
    summary = {"n_children": len(table), "n_finite": int(ok.sum()), "pooled_within_generating": gen,
               "pooled_within_refitted": ref, "bias_pooled": ref - gen,
               "median_within_generating": float(t["gen_within_share"].median()),
               "median_within_refitted": float(t["ref_within_share"].median()),
               "r_child_generating_refitted": float(np.corrcoef(t["gen_within_share"], t["ref_within_share"])[0, 1]),
               "within_tolerance": bool(abs(ref - gen) <= CEILING_TOLERANCE)}
    return table, summary


# ---------------------------------------------------------------- (c) lapses
def within_child_auroc(y: np.ndarray, score: np.ndarray, subjects: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = SEED) -> dict:
    """Share of concordant (lapse, non-lapse) pairs within children; ties count half."""
    f = pd.DataFrame({"s": subjects, "y": np.asarray(y, dtype=int), "x": np.asarray(score, dtype=float)})
    f["rank"] = f.groupby("s")["x"].rank(method="average")
    g = f.groupby("s").agg(n1=("y", "sum"), n=("y", "size"))
    g["n0"] = g["n"] - g["n1"]
    g["rank_sum"] = f[f["y"] == 1].groupby("s")["rank"].sum()
    g = g[(g["n1"] > 0) & (g["n0"] > 0)]
    u = (g["rank_sum"] - g["n1"] * (g["n1"] + 1) / 2).to_numpy(float)
    pairs = (g["n1"] * g["n0"]).to_numpy(float)
    draws = np.array([(w @ u) / (w @ pairs) for w in draw_counts(len(g), n_boot, seed)])
    lo, hi = interval(draws)
    return {"within_auroc": float(u.sum() / pairs.sum()), "within_auroc_ci_low": lo, "within_auroc_ci_high": hi,
            "n_children_within": len(g)}


def lapses_table(paths: ProjectPaths, samples: list[str]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        for name in LAPSE_DECODERS:
            p = load_predictions(paths, sample, name, "lapse")
            for scope, part in p.groupby("scope", sort=True):
                y, x, s = part["y_true"].to_numpy(int), part["y_pred"].to_numpy(float), part["subject"].to_numpy()
                pooled = subject_bootstrap(auroc, y, x, s, N_BOOT, SEED)
                rows.append({"sample": sample, "scope": scope, "decoder": name, "n_trials": len(part),
                             "n_children": int(pd.Series(s).nunique()), "lapse_rate": float(y.mean()),
                             "pooled_auroc": pooled.estimate, "pooled_auroc_ci_low": pooled.ci_low,
                             "pooled_auroc_ci_high": pooled.ci_high, **within_child_auroc(y, x, s)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (d) reliability
def split_halves(block_codes: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """(n_trials, n_splits) half labels: within each block, trials shuffled and assigned alternately, from a random
    start, to half 0 or 1."""
    rng = np.random.default_rng(seed)
    n = len(block_codes)
    n_blocks = int(block_codes.max()) + 1
    starts = np.r_[0, np.cumsum(np.bincount(block_codes, minlength=n_blocks))[:-1]]
    out = np.empty((n, n_splits), dtype=np.int8)
    for s in range(n_splits):
        order = np.lexsort((rng.random(n), block_codes))
        position = np.arange(n) - starts[block_codes[order]]
        offset = rng.integers(0, 2, n_blocks)
        half = np.empty(n, dtype=np.int8)
        half[order] = (position + offset[block_codes[order]]) % 2
        out[:, s] = half
    return out


def half_values(kind: str, y: np.ndarray, x: np.ndarray | None, codes: np.ndarray, half: np.ndarray, k: int) -> np.ndarray:
    """(k, 2) child values in each half: mean, sd (ddof 1) or Fisher z of the Pearson r of x and y."""
    idx = codes * 2 + half
    n = np.bincount(idx, minlength=2 * k).astype(float)
    sy, syy = np.bincount(idx, y, 2 * k), np.bincount(idx, y * y, 2 * k)
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "mean":
            v = sy / n
        elif kind == "sd":
            v = np.sqrt((syy - sy ** 2 / n) / (n - 1))
        else:
            sx, sxx, sxy = np.bincount(idx, x, 2 * k), np.bincount(idx, x * x, 2 * k), np.bincount(idx, x * y, 2 * k)
            r = (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
            v = np.arctanh(np.clip(r, -0.999, 0.999))
    v[n < 3] = np.nan
    return v.reshape(k, 2)


def weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Pearson correlation of a and b (children) for each row of weights w (draws x children)."""
    sw = w.sum(axis=1)
    ma, mb = (w @ a) / sw, (w @ b) / sw
    cov = (w @ (a * b)) / sw - ma * mb
    va, vb = (w @ (a * a)) / sw - ma ** 2, (w @ (b * b)) / sw - mb ** 2
    return cov / np.sqrt(va * vb)


def reliability(kind: str, y: np.ndarray, x: np.ndarray | None, subjects: np.ndarray, blocks: np.ndarray,
                n_splits: int = N_SPLITS, n_boot: int = N_BOOT_REL, seed: int = SEED) -> dict:
    counts = pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy()
    keep = counts >= MIN_RELIABILITY
    y, subjects, blocks = np.asarray(y, float)[keep], np.asarray(subjects)[keep], np.asarray(blocks)[keep]
    x = None if x is None else np.asarray(x, float)[keep]
    codes, uniq = pd.factorize(pd.Series(subjects), sort=True)
    k = len(uniq)
    block_codes = pd.factorize(pd.Series(blocks))[0]
    halves = split_halves(block_codes, n_splits, seed)
    a = np.empty((k, n_splits))
    b = np.empty((k, n_splits))
    for s in range(n_splits):
        v = half_values(kind, y, x, codes, halves[:, s].astype(int), k)
        a[:, s], b[:, s] = v[:, 0], v[:, 1]
    full = half_values(kind, y, x, codes, np.zeros(len(y), dtype=int), k)[:, 0]
    weights = np.vstack([np.ones(k), draw_counts(k, n_boot, seed + 1)])
    rel = np.empty((len(weights), n_splits))
    for s in range(n_splits):
        ok = np.isfinite(a[:, s]) & np.isfinite(b[:, s])
        rho = weighted_corr(a[ok, s], b[ok, s], weights[:, ok])
        rel[:, s] = 2 * rho / (1 + rho)
    per_draw = np.nanmean(rel, axis=1)
    est = float(per_draw[0])
    lo, hi = interval(per_draw[1:])
    var_full = float(np.nanvar(full, ddof=1))
    return {"n_children": k, "n_trials": len(y), "reliability": est, "ci_low": lo, "ci_high": hi,
            "split_sd": float(np.nanstd(rel[0])), "child_value_mean": float(np.nanmean(full)), "child_value_var": var_full,
            "true_between_child_sd": float(np.sqrt(max(est, 0.0) * var_full)),
            "attenuation": float(np.sqrt(max(est, 0.0)))}


def reliability_table(paths: ProjectPaths, samples: list[str]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        rt = load_predictions(paths, sample, "prestim_spectral", "rt_s")
        lapse = load_predictions(paths, sample, "prestim_spectral", "lapse")
        tc = load_predictions(paths, sample, "post_timecourse", "rt_s")
        en = load_predictions(paths, sample, "post_eegnet", "rt_s")
        m = pd.read_parquet(Path(ROOTS[sample]) / "features" / "epochs-v1" / "trial_measures.parquet",
                            columns=["subject", "block_key", "rt_s", "cpp_slope_resp"]).dropna(subset=["cpp_slope_resp"])
        log_rt = np.log(rt["y_true"].to_numpy(float))
        specs = {
            "mean_log_rt": ("mean", log_rt, None, rt),
            "sd_log_rt": ("sd", log_rt, None, rt),
            "lapse_rate": ("mean", lapse["y_true"].to_numpy(float), None, lapse),
            "coupling_prestim_spectral_logrt": ("corr", log_rt, rt["y_pred"].to_numpy(float), rt),
            "coupling_post_timecourse_rt": ("corr", tc["y_true"].to_numpy(float), tc["y_pred"].to_numpy(float), tc),
            "coupling_post_eegnet_rt": ("corr", en["y_true"].to_numpy(float), en["y_pred"].to_numpy(float), en),
            "coupling_cpp_slope_logrt": ("corr", np.log(m["rt_s"].to_numpy(float)), m["cpp_slope_resp"].to_numpy(float), m),
        }
        for name, (kind, y, x, frame) in specs.items():
            rows.append({"sample": sample, "measure": name, "kind": kind,
                         **reliability(kind, y, x, frame["subject"].to_numpy(), frame["block_key"].to_numpy())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", default=["units", "ceiling", "lapses", "reliability"],
                        choices=["units", "ceiling", "ceiling_validation", "lapses", "reliability"])
    parser.add_argument("--samples", nargs="+", default=list(ROOTS))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--ceiling-children", type=int, default=N_CEILING_CHILDREN, help="smoke tests only")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "effect_sizes" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    summary, tables = {"git_commit": commit, "out_dir": str(out_dir)}, {}
    if "units" in args.parts:
        tables["units"] = units_table(paths, args.samples)
        tables["yardsticks"] = yardsticks_table(paths, args.samples)
    if "ceiling" in args.parts:
        summary["ceiling"] = ceiling_observed(paths, args.samples)
    if "ceiling_validation" in args.parts:
        table, val = ceiling_validation(paths, args.n_jobs, args.ceiling_children)
        write_dataframe(table, out_dir / "ceiling_validation.csv")
        summary["ceiling_validation"] = val
    if "lapses" in args.parts:
        tables["lapses"] = lapses_table(paths, args.samples)
    if "reliability" in args.parts:
        tables["reliability"] = reliability_table(paths, args.samples)
    for name, table in tables.items():
        write_dataframe(table, out_dir / f"{name}.csv")
    if "units" in tables and "ceiling" in summary:
        u = tables["units"]
        pre = u[u["decoder"].isin(PRE_STIMULUS)].copy()
        pre["ceiling"] = pre["sample"].map({s: v["ceiling"] for s, v in summary["ceiling"].items()})
        pre["share_of_ceiling"] = pre["r2_within"] / pre["ceiling"]
        write_dataframe(pre, out_dir / "ceiling_fractions.csv")
    if "lapses" in tables:
        lt = tables["lapses"]
        spec = lt[lt["decoder"] == "prestim_spectral"].set_index("scope")
        need = [s for s in ("cv", "holdout_R5") if s in spec.index]
        summary["lapse_rule"] = {
            "within_auroc_prestim_spectral": spec["within_auroc"].round(4).to_dict(),
            "supported": bool(need == ["cv", "holdout_R5"] and all(spec.loc[s, "within_auroc_ci_low"] > 0.5 for s in need)
                              and all(spec.loc[s, "within_auroc"] > 0.5 for s in ("MIPDB", "R10R11") if s in spec.index))}
    if "reliability" in tables:
        rt = tables["reliability"]
        summary["reliability_rule"] = {f"{r.sample}:{r.measure}": bool(r.ci_high < 0.3)
                                       for r in rt.itertuples() if r.measure.startswith("coupling")}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({k: v for k, v in summary.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
