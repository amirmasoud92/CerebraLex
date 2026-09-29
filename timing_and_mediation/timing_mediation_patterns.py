"""Task timing, mediation and forward patterns (exploratory).

timing    the pre-stimulus band-power signal against task timing (foreperiod, time in block, previous RT) and the
          eyes: residualised saved predictions (1a) and retrained ridge variants E, T, TE, P, O, F (1b).
mediation does the pre-stimulus state act through the unfolding decision? Within-child paths between the horizon
          decoders C (pre-stimulus) and A (0 to T) and RT, after timing is residualised.
patterns  forward patterns (Haufe et al. 2014) of the held-out predictions of horizon decoder A (T = 1.3 s; channel x
          100 ms bin) and of the pre-stimulus band-power model (channel x band), with SVD, split halves and
          replication.
``--parts`` selects timing, mediation and patterns (all by default). Writes results/timing_mediation_patterns/<run>/.
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
from scipy.stats import spearmanr

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import BenchmarkConfig, PreprocessingConfig, TrialConfig
from common.scripts import script_path
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.benchmark import FeatureCache, demographics_table, load_window_data
from decoding.decomposition import within_between
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD

ROOTS = {"HBN": data_location("HBN"), "MIPDB": data_location("MIPDB"),
         "R10R11": data_location("R10R11")}
PEOPLE = {"MIPDB": f"{data_location('MIPDB')}/manifest/participants.csv",
          "R10R11": f"{data_location('R10R11')}/manifest/participants.csv"}
PRESTIM_RUNS = {"HBN": "results/benchmark-v1/prestim", "MIPDB": "results/external_validation/mipdb",
                "R10R11": "results/external_validation/r10_r11"}
HORIZON = "results/prediction_horizon"
SCOPE_SAMPLE = {"cv": "HBN", "holdout_R5": "HBN", "MIPDB": "MIPDB", "R10R11": "R10R11"}
PERIOCULAR = ("E8", "E14", "E21", "E25", "E125", "E126", "E127", "E128")
CPP = ("E55", "E54", "E86", "E62")
BANDS = ("delta", "theta", "alpha", "beta_low", "beta_high", "gamma", "total")
FP_EDGES = (2.5, 3.5, 5.2)            # foreperiod categories; 2.5 to 3.5 s is the reference
MEDIATION_T = (0.5, 0.9, 1.1, 1.3)
PATTERN_T, MARGIN = 1.3, 0.1
N_CH = 129
VARIANTS = ("E", "T", "TE", "P", "O", "F")
N_BOOT, SEED = 2000, 12345


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- timing variables
def foreperiod_dummies(fp: np.ndarray) -> np.ndarray:
    """Dummies for foreperiods below 2.5 s, 3.5 to 5.2 s and above 5.2 s (reference 2.5 to 3.5 s)."""
    fp = np.asarray(fp, dtype=float)
    lo, mid, hi = FP_EDGES
    return np.column_stack([fp < lo, (fp >= mid) & (fp < hi), fp >= hi]).astype(float)


def timing_matrix(frame: pd.DataFrame, which: str) -> np.ndarray:
    """'fp': foreperiod dummies; 'fp_time': plus time in block; 'fp_time_prev': plus previous RT."""
    parts = [foreperiod_dummies(frame["foreperiod_s"].to_numpy(float))]
    if which in ("fp_time", "fp_time_prev"):
        parts.append(frame["time_in_block_s"].to_numpy(float)[:, None])
    if which == "fp_time_prev":
        parts.append(frame["prev_rt_s"].to_numpy(float)[:, None])
    return np.column_stack(parts)


def timing_covariates(frame: pd.DataFrame) -> np.ndarray:
    """Covariate block of models T and TE: foreperiod dummies, time in block, previous RT (NaN, imputed in the
    model) and a missing-previous-RT indicator."""
    prev = frame["prev_rt_s"].to_numpy(float)
    return np.column_stack([foreperiod_dummies(frame["foreperiod_s"].to_numpy(float)),
                            frame["time_in_block_s"].to_numpy(float), prev, np.isnan(prev).astype(float)])


def centre_within(values: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    v = pd.DataFrame(np.asarray(values, dtype=float).reshape(len(subjects), -1))
    return (v - v.groupby(np.asarray(subjects)).transform("mean")).to_numpy()


def residualise_within(y: np.ndarray, x: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Child-centred y minus its projection on child-centred x, with pooled within-child coefficients."""
    yc = centre_within(y, subjects)
    xc = centre_within(x, subjects)
    keep = np.abs(xc).sum(axis=0) > 0                       # constant columns carry nothing within child
    beta = np.linalg.lstsq(xc[:, keep], yc, rcond=None)[0]
    out = yc - xc[:, keep] @ beta
    return out[:, 0] if out.shape[1] == 1 else out


# ---------------------------------------------------------------- task timing
def demographics_for(sample: str, paths: ProjectPaths) -> pd.DataFrame:
    if sample == "HBN":
        return demographics_table(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    return pd.read_csv(PEOPLE[sample], dtype={"subject": str})


def prestim_sample(sample: str, paths: ProjectPaths, bcfg: BenchmarkConfig, spec) -> dict:
    """Rows of the saved pre-stimulus band-power RT predictions, in window-row order (the benchmark's cohort
    order), with timing variables, demographics and the path of the cached band-power features."""
    root = Path(ROOTS[sample])
    window = load_window_data(root / "windows-v1", spec, 250.0)
    cache = FeatureCache(root / "features" / "windows-v1", window, bcfg.aperiodic_regions, np.zeros((N_CH, 3)), None)
    cache.spectral()
    pred = pd.read_parquet(paths.root / PRESTIM_RUNS[sample] / "predictions.parquet")
    pred = pred[(pred["window"] == "prestim") & (pred["target"] == "rt_s") & (pred["model"] == "spectral")]
    cols = ["subject", "block_key", "trial_index", "y_true", "y_pred"] + (["fold"] if "fold" in pred else [])
    rows = pred[cols].merge(window.index[["subject", "block_key", "trial_index", "row"]],
                            on=["subject", "block_key", "trial_index"], how="left", validate="one_to_one")
    trials = pd.read_parquet(root / "trials-v1" / "trials_ccd.parquet",
                             columns=["block_key", "trial_index", "foreperiod_s", "time_in_block_s", "prev_rt_s"])
    rows = rows.merge(trials, on=["block_key", "trial_index"], how="left", validate="one_to_one")
    rows = rows.merge(demographics_for(sample, paths)[["subject", "age", "sex", "ehq_total"]], on="subject", how="left")
    assert rows["row"].notna().all() and rows["foreperiod_s"].notna().all()
    if "fold" not in rows:
        rows["fold"] = pd.array([pd.NA] * len(rows), dtype="Int64")
        rows["scope"] = sample
    else:
        rows["scope"] = np.where(rows["fold"] == HOLDOUT_FOLD, "holdout_R5", "cv")
    rows = rows.sort_values("row").reset_index(drop=True)
    channels = list(window.meta.get("channels") or [])
    if len(channels) != N_CH:
        import mne

        channels = mne.channels.make_standard_montage("GSN-HydroCel-129").ch_names
    return {"rows": rows, "spectral": str(cache.root / "spectral.npy"), "channels": channels}


def channel_columns(channels: list[str], keep: np.ndarray) -> np.ndarray:
    """Band-power columns (band-major blocks of 129 channels) of the channels in ``keep``."""
    idx = np.flatnonzero(keep)
    return np.concatenate([b * N_CH + idx for b in range(len(BANDS))])


def demographic_covariates(rows: pd.DataFrame) -> np.ndarray:
    sex = rows["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
    return np.column_stack([pd.to_numeric(rows["age"], errors="coerce").to_numpy(float), sex.to_numpy(float),
                            pd.to_numeric(rows["ehq_total"], errors="coerce").to_numpy(float)])


def variant_inputs(variant: str, rows: pd.DataFrame, spectral_path: str, cols: dict) -> tuple[np.ndarray, np.ndarray]:
    covs = demographic_covariates(rows)
    if variant in ("T", "TE"):
        covs = np.column_stack([covs, timing_covariates(rows)])
    if variant == "T":
        return np.zeros((len(rows), 0)), covs
    x = np.load(spectral_path, mmap_mode="r")[rows["row"].to_numpy(int)]
    if variant in ("P", "O"):
        x = x[:, cols[variant]]
    return np.asarray(x, dtype=np.float64), covs


def fit_predict(variant: str, train: pd.DataFrame, tests: dict[str, tuple[pd.DataFrame, str]], spectral_path: str,
                cols: dict, bcfg: BenchmarkConfig) -> dict[str, np.ndarray]:
    target = "foreperiod_s" if variant == "F" else "y_true"
    x, c = variant_inputs(variant, train, spectral_path, cols)
    model = GroupedRidge(name=f"timing_{variant}", alphas=tuple(bcfg.ridge_alphas), n_inner_folds=bcfg.inner_folds,
                         seed=bcfg.seed)
    model.fit(DecoderInput(x, train["subject"].to_numpy(), c), train[target].to_numpy(float))
    out = {}
    for name, (rows, path) in tests.items():
        xt, ct = variant_inputs(variant, rows, path, cols)
        out[name] = model.predict(DecoderInput(xt, rows["subject"].to_numpy(), ct))
    return out


def run_variants(samples: dict, bcfg: BenchmarkConfig, n_jobs: int) -> pd.DataFrame:
    hbn = samples["HBN"]["rows"]
    cols = {"P": channel_columns(samples["HBN"]["channels"], posterior_mask(samples["HBN"]["channels"])),
            "O": channel_columns(samples["HBN"]["channels"], np.isin(samples["HBN"]["channels"], PERIOCULAR))}
    fold = hbn["fold"].astype(int).to_numpy()
    jobs, keys = [], []
    for variant in VARIANTS:
        for f in [*sorted(int(v) for v in np.unique(fold) if v != HOLDOUT_FOLD), HOLDOUT_FOLD]:
            test = fold == f
            train = (fold != f) & (fold != HOLDOUT_FOLD)
            LeakageAuditor.assert_rows_grouped(hbn, train, test)
            jobs.append(delayed(fit_predict)(variant, hbn[train], {"hbn": (hbn[test], samples["HBN"]["spectral"])},
                                             samples["HBN"]["spectral"], cols, bcfg))
            keys.append((variant, f))
        externals = {s: (v["rows"], v["spectral"]) for s, v in samples.items() if s != "HBN"}
        if externals:
            jobs.append(delayed(fit_predict)(variant, hbn, externals, samples["HBN"]["spectral"], cols, bcfg))
            keys.append((variant, "external"))
    results = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
    frames = []
    for (variant, f), res in zip(keys, results, strict=True):
        for name, pred in res.items():
            rows = hbn[fold == f] if name == "hbn" else samples[name]["rows"]
            frames.append(rows[["subject", "block_key", "trial_index", "scope", "y_true", "y_pred", "foreperiod_s"]]
                          .assign(variant=variant, pred=pred))
    return pd.concat(frames, ignore_index=True)


def posterior_mask(channels: list[str]) -> np.ndarray:
    import mne

    pos = mne.channels.make_standard_montage("GSN-HydroCel-129").get_positions()["ch_pos"]
    return np.array([pos[c][1] < 0 for c in channels])


def timing(samples: dict, bcfg: BenchmarkConfig, n_jobs: int, horizon) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    resid_rows = []
    for sample, s in samples.items():
        for scope, part in s["rows"].groupby("scope", sort=True):
            subj = part["subject"].to_numpy()
            fp_r = within_between(part["foreperiod_s"].to_numpy(float), part["y_pred"].to_numpy(float), subj,
                                  n_bootstrap=N_BOOT, seed=SEED)
            resid_rows.append({"scope": scope, "adjustment": "none (prediction with foreperiod)", "n_trials": len(part),
                               **{k: fp_r[k] for k in ("n_children", "r_within", "r_within_ci_low", "r_within_ci_high")}})
            for which in ("none", "fp", "fp_time", "fp_time_prev"):
                sub = part.dropna(subset=["prev_rt_s"]) if which == "fp_time_prev" else part
                y, p = sub["y_true"].to_numpy(float), sub["y_pred"].to_numpy(float)
                if which != "none":
                    x = timing_matrix(sub, which)
                    y, p = residualise_within(y, x, sub["subject"].to_numpy()), residualise_within(p, x, sub["subject"].to_numpy())
                r = within_between(y, p, sub["subject"].to_numpy(), n_bootstrap=N_BOOT, seed=SEED)
                resid_rows.append({"scope": scope, "adjustment": which, "n_trials": len(sub),
                                   **{k: r[k] for k in ("n_children", "r_within", "r_within_ci_low", "r_within_ci_high")}})
    resid = pd.DataFrame(resid_rows)
    preds = run_variants(samples, bcfg, n_jobs)
    e_cv = preds[(preds["variant"] == "E") & (preds["scope"] == "cv")]
    repro = float(np.abs(e_cv["pred"] - e_cv["y_pred"]).max())
    curve, paired = [], []
    for scope, part in preds.groupby("scope", sort=True):
        wide = part.pivot_table(index=["subject", "block_key", "trial_index", "y_true", "foreperiod_s"],
                                columns="variant", values="pred").reset_index()
        subj = wide["subject"].to_numpy()
        for v in VARIANTS:
            target = wide["foreperiod_s"] if v == "F" else wide["y_true"]
            r = within_between(target.to_numpy(float), wide[v].to_numpy(float), subj, n_bootstrap=N_BOOT, seed=SEED)
            curve.append({"scope": scope, "variant": v, "target": "foreperiod" if v == "F" else "rt", "n_trials": len(wide),
                          **{k: r[k] for k in ("n_children", "r_within", "r_within_ci_low", "r_within_ci_high")}})
        for a, b in (("TE", "T"), ("E", "P"), ("E", "O")):
            paired.append({"scope": scope, "comparison": f"{a}-{b}", "n_trials": len(wide),
                           **horizon.paired_bootstrap(wide["y_true"].to_numpy(float), wide[a].to_numpy(float),
                                                      wide[b].to_numpy(float), subj, "difference", N_BOOT, SEED)})
    curve, paired = pd.DataFrame(curve), pd.DataFrame(paired)

    def get(table, scope, **kw):
        t = table[table["scope"] == scope]
        for k, v in kw.items():
            t = t[t[k] == v]
        return t.iloc[0] if len(t) else None

    full = {s: get(resid, s, adjustment="fp_time_prev") for s in ("cv", "holdout_R5", "MIPDB", "R10R11")}
    te_t = {s: get(paired, s, comparison="TE-T") for s in ("cv", "holdout_R5")}
    a = all(full[s] is not None and full[s]["r_within_ci_low"] > 0 for s in ("cv", "holdout_R5")) and \
        all(full[s] is None or full[s]["r_within"] > 0 for s in ("MIPDB", "R10R11"))
    b = all(te_t[s] is not None and te_t[s]["ci_low"] > 0 for s in ("cv", "holdout_R5"))
    rule1 = "endogenous state supported" if a and b else "mixed" if a or b else "task timing"
    cv = curve[curve["scope"] == "cv"].set_index("variant")["r_within"]
    p_share, o_share = float(cv["P"] / cv["E"]), float(cv["O"] / cv["E"])
    rule2 = ("not mainly ocular" if p_share >= 0.5 and cv["O"] < cv["E"]
             else "possibly ocular" if o_share >= 0.8 and p_share < 0.5 else "mixed")
    summary = {"reproduction_max_abs_diff_cv": repro, "reproduced": repro < 1e-6,
               "rule1": {"residual_r_positive": a, "te_minus_t_positive": b, "verdict": rule1},
               "rule2": {"posterior_share_of_E_cv": p_share, "periocular_share_of_E_cv": o_share, "verdict": rule2}}
    return summary, resid, pd.concat([curve.assign(kind="r"), paired.assign(kind="paired")], ignore_index=True)


# ---------------------------------------------------------------- mediation
def path_coefficients(s: np.ndarray) -> dict:
    """Standardised paths from a 3 x 3 within-child cross-product matrix of (RT, A, C)."""
    d = np.sqrt(np.diag(s))
    r = s / np.outer(d, d)
    r_ya, r_yc, r_ac = r[0, 1], r[0, 2], r[1, 2]
    denom = 1 - r_ac ** 2
    b = (r_ya - r_yc * r_ac) / denom
    direct = (r_yc - r_ya * r_ac) / denom
    return {"total": r_yc, "a": r_ac, "b": b, "direct": direct, "indirect": r_ac * b,
            "share": (r_ac * b) / r_yc if r_yc != 0 else np.nan}


def child_crossproducts(v: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-child sums of outer products of child-centred rows of v (n x k)."""
    codes, uniq = pd.factorize(pd.Series(subjects), sort=True)
    k = v.shape[1]
    out = np.zeros((len(uniq), k, k))
    for i in range(k):
        for j in range(i, k):
            out[:, i, j] = out[:, j, i] = np.bincount(codes, v[:, i] * v[:, j], len(uniq))
    return out, uniq


def boot_paths(v: np.ndarray, subjects: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    cp, uniq = child_crossproducts(v, subjects)
    est = path_coefficients(cp.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        draws.append(path_coefficients(np.tensordot(w, cp, axes=1)))
    out = {"n_children": len(uniq), "n_trials": len(v)}
    for key, value in est.items():
        vals = np.array([d[key] for d in draws])
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        out.update({key: float(value), f"{key}_ci_low": float(lo), f"{key}_ci_high": float(hi)})
    return out


def horizon_wide(paths: ProjectPaths) -> pd.DataFrame:
    h = pd.read_parquet(paths.root / HORIZON / "predictions.parquet")
    h = h[(h["margin"] == MARGIN) & h["decoder"].isin(["A", "C"]) & h["T"].isin(MEDIATION_T)]
    wide = h.pivot_table(index=["scope", "T", "subject", "block_key", "trial_index", "rt_s"], columns="decoder",
                         values="pred").reset_index().dropna(subset=["A", "C"])
    frames = []
    for sample in ("HBN", "MIPDB", "R10R11"):
        trials = pd.read_parquet(Path(ROOTS[sample]) / "trials-v1" / "trials_ccd.parquet",
                                 columns=["block_key", "trial_index", "foreperiod_s", "time_in_block_s"])
        part = wide[wide["scope"].map(SCOPE_SAMPLE) == sample]
        frames.append(part.merge(trials, on=["block_key", "trial_index"], how="left", validate="many_to_one"))
    return pd.concat(frames, ignore_index=True)


def prepared(part: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Columns residualised on foreperiod and time in block within child, scaled by pooled within-child SD."""
    subj = part["subject"].to_numpy()
    x = timing_matrix(part, "fp_time")
    v = np.column_stack([residualise_within(part[c].to_numpy(float), x, subj) for c in columns])
    return v / v.std(axis=0)


def mediation(paths: ProjectPaths) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    wide = horizon_wide(paths)
    rows = []
    for (scope, t), part in wide.groupby(["scope", "T"], sort=True):
        rows.append({"scope": scope, "T": t, **boot_paths(prepared(part, ("rt_s", "A", "C")), part["subject"].to_numpy())})
    paths_table = pd.DataFrame(rows)
    growth = []
    for scope, part in wide.groupby("scope", sort=True):
        early = part[part["T"] == 0.5].set_index(["subject", "block_key", "trial_index"])
        late = part[part["T"] == 1.3].set_index(["subject", "block_key", "trial_index"])
        common = early.index.intersection(late.index)
        e, l = early.loc[common].reset_index(), late.loc[common].reset_index()
        v = np.column_stack([prepared(e, ("A", "C")), prepared(l, ("A", "C"))])      # A_0.5, C_0.5, A_1.3, C_1.3
        cp, uniq = child_crossproducts(v, e["subject"].to_numpy())

        def diff(s):
            d = np.sqrt(np.diag(s))
            r = s / np.outer(d, d)
            return r[2, 3] - r[0, 1]

        rng = np.random.default_rng(SEED)
        draws = [diff(np.tensordot(np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float),
                                   cp, axes=1)) for _ in range(N_BOOT)]
        s = cp.sum(axis=0)
        d = np.sqrt(np.diag(s))
        r = s / np.outer(d, d)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        growth.append({"scope": scope, "n_trials": len(e), "n_children": len(uniq), "a_T0.5": float(r[0, 1]),
                       "a_T1.3": float(r[2, 3]), "difference": float(diff(s)), "ci_low": float(lo), "ci_high": float(hi)})
    growth = pd.DataFrame(growth)

    def at(scope):
        t = paths_table[(paths_table["scope"] == scope) & (paths_table["T"] == 1.3)]
        return t.iloc[0] if len(t) else None

    main = all(at(s) is not None and at(s)["indirect_ci_low"] > 0
               and (at(s)["direct_ci_low"] <= 0 <= at(s)["direct_ci_high"] or at(s)["direct"] < 0.5 * at(s)["total"])
               for s in ("cv", "holdout_R5"))
    ext = all(at(s) is None or (at(s)["indirect"] > 0 and at(s)["direct"] < 0.5 * at(s)["total"]) for s in ("MIPDB", "R10R11"))
    g = growth.set_index("scope")
    grows = bool("cv" in g.index and g.loc["cv", "ci_low"] > 0)
    indirect_positive = all(at(s) is not None and at(s)["indirect_ci_low"] > 0 for s in ("cv", "holdout_R5"))
    verdict = ("state acts through the unfolding decision" if main and ext and grows
               else "persisting state seen in both windows" if indirect_positive and not grows else "independent accounts")
    return {"rule3": {"main_cv_r5": main, "external_pattern": ext, "a_path_grows_cv": grows, "verdict": verdict}}, paths_table, growth


# ---------------------------------------------------------------- forward patterns
def forward_pattern(x: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Within-child covariance of each feature with the prediction per unit within-child prediction variance."""
    pc = centre_within(pred, subjects)[:, 0]
    xc = centre_within(x, subjects)
    return (xc.T @ pc) / (pc @ pc)


def svd_summary(pattern: np.ndarray, channels: list[str]) -> dict:
    u, s, vt = np.linalg.svd(pattern, full_matrices=False)
    cpp = np.isin(channels, CPP)
    u1, v1 = u[:, 0], vt[0]
    if u1[cpp].mean() > 0:                       # more positive CPP, shorter predicted RT
        u1, v1 = -u1, -v1
    share = s ** 2 / (s ** 2).sum()
    return {"shares": share[:3].tolist(), "u1": u1, "v1": v1 * s[0],
            "cpp_mean_abs": float(np.abs(u1[cpp]).mean()), "p90_abs": float(np.percentile(np.abs(u1), 90)),
            "top_channels": [channels[i] for i in np.argsort(-np.abs(u1))[:8]]}


def timecourse_rows(paths: ProjectPaths, trials: pd.DataFrame, sample: str, specs: dict, bcfg) -> np.ndarray:
    """(n, 129, 13) time-course features of [0, 1.3] s for the given trials (early bins 0 to 1.0 s, challenge bins
    1.0 to 1.3 s), in microvolts."""
    root = Path(ROOTS[sample])
    early = load_window_data(root / "windows-v1-early", specs["early"], 250.0)
    chal = load_window_data(root / "windows-v1", specs["challenge"], 250.0)
    ce = FeatureCache(root / "features" / "windows-v1-early", early, bcfg.aperiodic_regions, np.zeros((N_CH, 3)), None)
    cc = FeatureCache(root / "features" / "windows-v1", chal, bcfg.aperiodic_regions, np.zeros((N_CH, 3)), None)
    te, tc = ce.timecourse(bcfg.timecourse_bin_ms), cc.timecourse(bcfg.timecourse_bin_ms)
    keys = ["subject", "block_key", "trial_index"]
    re_ = trials[keys].merge(early.index[[*keys, "row"]], on=keys, how="left", validate="one_to_one")["row"].to_numpy(int)
    rc = trials[keys].merge(chal.index[[*keys, "row"]], on=keys, how="left", validate="one_to_one")["row"].to_numpy(int)
    a = np.asarray(te[re_], dtype=np.float64).reshape(len(trials), N_CH, 10)
    b = np.asarray(tc[rc], dtype=np.float64).reshape(len(trials), N_CH, 20)[:, :, 5:8]
    return np.concatenate([a, b], axis=2)


def forward_patterns(paths: ProjectPaths, samples: dict, specs: dict, bcfg, keep: set | None = None) -> tuple[dict, dict]:
    channels = samples["HBN"]["channels"]
    h = pd.read_parquet(paths.root / HORIZON / "predictions.parquet")
    h = h[(h["margin"] == MARGIN) & (h["decoder"] == "A") & (h["T"] == PATTERN_T)].reset_index(drop=True)
    if keep is not None:                                  # smoke tests: HBN children limited
        h = h[~h["scope"].isin(["cv", "holdout_R5"]) | h["subject"].isin(keep)].reset_index(drop=True)
    h = h[h["scope"].map(SCOPE_SAMPLE).isin(samples)].reset_index(drop=True)
    patterns, out = {}, {}
    for scope, part in h.groupby("scope", sort=True):
        part = part.reset_index(drop=True)
        x = timecourse_rows(paths, part, SCOPE_SAMPLE[scope], specs, bcfg)
        pat = forward_pattern(x.reshape(len(part), -1), part["pred"].to_numpy(float), part["subject"].to_numpy())
        patterns[("A", scope)] = pat.reshape(N_CH, 13)
        if scope == "cv":
            for half, keep in (("even", part["fold"] % 2 == 0), ("odd", part["fold"] % 2 == 1)):
                k = keep.to_numpy(bool)
                patterns[("A", f"cv_{half}")] = forward_pattern(x[k].reshape(k.sum(), -1), part.loc[k, "pred"].to_numpy(float),
                                                                part.loc[k, "subject"].to_numpy()).reshape(N_CH, 13)
    for sample, s in samples.items():
        rows = s["rows"]
        spec = np.load(s["spectral"], mmap_mode="r")
        for scope, part in rows.groupby("scope", sort=True):
            x = np.asarray(spec[part["row"].to_numpy(int)], dtype=np.float64)
            pat = forward_pattern(x, part["y_pred"].to_numpy(float), part["subject"].to_numpy())
            patterns[("E", scope)] = pat.reshape(len(BANDS), N_CH).T
            if scope == "cv":
                for half, keep in (("even", part["fold"] % 2 == 0), ("odd", part["fold"] % 2 == 1)):
                    k = keep.to_numpy(bool)
                    patterns[("E", f"cv_{half}")] = forward_pattern(
                        x[k], part.loc[k, "y_pred"].to_numpy(float), part.loc[k, "subject"].to_numpy()).reshape(len(BANDS), N_CH).T

    def corr(m, a, b):
        pa, pb = patterns.get((m, a)), patterns.get((m, b))
        return float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1]) if pa is not None and pb is not None else None

    for model in ("A", "E"):
        svd = svd_summary(patterns[(model, "cv")], channels)
        out[model] = {"shares": svd["shares"], "cpp_mean_abs": svd["cpp_mean_abs"], "p90_abs": svd["p90_abs"],
                      "top_channels": svd["top_channels"], "u1": svd["u1"].tolist(), "v1": svd["v1"].tolist(),
                      "split_half_r": corr(model, "cv_even", "cv_odd"), "r_with_R5": corr(model, "cv", "holdout_R5"),
                      "r_with_R10R11": corr(model, "cv", "R10R11"), "r_with_MIPDB": corr(model, "cv", "MIPDB")}
    bins = np.r_[np.arange(10) * 0.1 + 0.05, 1.05, 1.15, 1.25]
    late = bins >= 0.5
    rho = float(spearmanr(bins[late], np.abs(np.asarray(out["A"]["v1"])[late]))[0])
    a = out["A"]
    rule4 = {"rank1_share": a["shares"][0] >= 0.5, "cpp_above_p90": a["cpp_mean_abs"] > a["p90_abs"],
             "ramp_rho": rho, "ramp": rho >= 0.8,
             "stable": (a["split_half_r"] or 0) >= 0.8 and (a["r_with_R10R11"] or 0) >= 0.5}
    rule4["verdict"] = "centro-parietal ramp" if all(rule4[k] for k in ("rank1_share", "cpp_above_p90", "ramp", "stable")) \
        else "components as found"
    e = patterns[("E", "cv")]
    peri = np.isin(channels, PERIOCULAR)
    post = posterior_mask(channels)
    out["E"]["band_mean"] = {b: float(e[:, i].mean()) for i, b in enumerate(BANDS)}
    out["E"]["periocular_vs_posterior"] = {b: {"periocular": float(e[peri, i].mean()), "posterior": float(e[post, i].mean())}
                                           for i, b in enumerate(BANDS)}
    out["rule4"] = rule4
    out["bins_s"] = bins.tolist()
    out["channels"] = channels
    return out, {f"{m}:{s}": p.tolist() for (m, s), p in patterns.items()}


# ---------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="+", default=["timing", "mediation", "patterns"],
                        choices=["timing", "mediation", "patterns"])
    parser.add_argument("--samples", nargs="+", default=list(ROOTS))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--limit-children", type=int, default=None, help="smoke tests: this many HBN children")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    specs = {"early": load_config(paths.root / "features" / "windows_early.yaml", TrialConfig).windows[0]}
    for w in load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig).windows:
        specs[w.name] = w
    horizon = sibling("prediction_horizon")
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "timing_mediation_patterns" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    summary = {"git_commit": commit, "out_dir": str(out_dir)}
    samples = {s: prestim_sample(s, paths, bcfg, specs["prestim"]) for s in args.samples} \
        if {"timing", "patterns"} & set(args.parts) else {}
    keep = None
    if args.limit_children and "HBN" in samples:
        rows = samples["HBN"]["rows"]
        keep = set(sorted(rows["subject"].unique())[:args.limit_children])
        samples["HBN"]["rows"] = rows[rows["subject"].isin(keep)].reset_index(drop=True)
    if "timing" in args.parts:
        s1, resid, models = timing(samples, bcfg, args.n_jobs, horizon)
        write_dataframe(resid, out_dir / "timing_residualised.csv")
        write_dataframe(models, out_dir / "timing_models.csv")
        summary["timing"] = s1
        write_json(summary, out_dir / "summary.json", on_exists="overwrite")
        if not s1["reproduced"] and not args.limit_children:
            raise SystemExit(f"model E did not reproduce the saved predictions (max diff {s1['reproduction_max_abs_diff_cv']:.3g})")
    if "mediation" in args.parts:
        s2, paths_table, growth = mediation(paths)
        write_dataframe(paths_table, out_dir / "mediation_paths.csv")
        write_dataframe(growth, out_dir / "mediation_a_growth.csv")
        summary["mediation"] = s2
    if "patterns" in args.parts:
        s3, pats = forward_patterns(paths, samples, specs, bcfg, keep)
        write_json(pats, out_dir / "forward_patterns.json")
        summary["patterns"] = s3
    write_json(summary, out_dir / "summary.json", on_exists="overwrite")
    print(json.dumps({k: v for k, v in summary.items() if k != "patterns"} | ({"patterns_rule4": summary["patterns"]["rule4"]}
                                                                          if "patterns" in summary else {}), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
