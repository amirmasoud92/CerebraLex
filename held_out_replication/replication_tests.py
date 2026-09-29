"""Replication tests on the held-out HBN releases R10 and R11.

1. ``<root>/manifest/participants.csv`` (subject, age, sex, ehq_total, release) from the root's audit and manifest.
2. The analysis scripts, unchanged, with the root as the external sample: single_trial_measures.py;
   external_validation.py (mean, demographics, band power, time course; EEGNet): C1, C2, C3;
   preresponse_decoding_mipdb.py: C4; rt_information_maps.py: point estimates of C6a, C6b and S5;
   cpp_onset_rate.py: C7 and S1.
3. Here, with the functions of those scripts: C5a and C5b (consecutive-difference variant of trial_specificity.py);
   child-bootstrap intervals of C6a, C6b and S5 (per-child sums of the maps, checked against the script's point
   estimates); S2 and S3 (child measures and the amplitude normalisation); S4 (the lapse-rate model of
   single_trial_associations.py); and the tests of sets A (accumulation model) and B (Symbol Search).
4. tests.csv (estimate, 95% child-bootstrap interval with 2,000 draws, criterion, verdict) and summary.json in
   results/replication_tests/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import BenchmarkConfig, PreprocessingConfig, TrialConfig
from common.scripts import script_module, script_path
from decoding.benchmark import FeatureCache, load_window_data
from features.clusters import cluster_channels
from features.spectral import DEFAULT_BANDS
from features.windows import WindowStore
from state_models_and_trait_moderation.trait_state import TRAIT_COLUMNS, load_traits

SEED, N_BOOT = 12345, 2000
EARLY_BIN = 9                                   # 0.9 to 1.0 s after the target (100 ms bins of the early window)
# Power at the realised n (the rule below 400 children). Its method (least favourable estimate, HBN
# standard error scaled to n) gives Phi((Phi^-1(P_m) + 1.96) sqrt(n / m) - 1.96) from the power P_m at m children;
# m = 350, the values logged when the tests were specified (the most precise available; entries logged as ">x" are lower bounds
# and so are their scaled values). S5 has no logged power: its standard error was not estimated in HBN.
POWER_350 = {"C1": (0.999, True), "C2": (0.999, True), "C3": (0.98, False), "C4": (0.999, True), "C5a": (0.999, True),
             "C5b": (0.998, False), "C6a": (0.99, False), "C6b": (0.99, True), "C7": (0.86, False), "S1": (0.5, False),
             "S2": (0.55, False), "S3": (0.25, False), "S4": (0.63, False), "S5": (float("nan"), False)}
# id, tier, bound tested, threshold, claim
HYPOTHESES = [
    ("C1", "primary", "ci_low", 0.10, "Challenge-window time-course ridge: within-child r"),
    ("C2", "primary", "ci_low", 0.10, "Challenge-window EEGNet: within-child r"),
    ("C3", "primary", "ci_low", 0.0, "Pre-stimulus band-power model: within-child r"),
    ("C4", "primary", "ci_low", 0.0, "Pre-response window E1 (500 to 1,000 ms, RT above 1.1 s): within-child r"),
    ("C5a", "primary", "ci_low", 0.0, "C1's decoder, consecutive-trial differences: within-child r"),
    ("C5b", "primary", "ci_low", 0.0, "C3's decoder, consecutive-trial differences: within-child r"),
    ("C6a", "primary", "ci_high", 0.0, "CPP-cluster within-child r with log RT at 0.9 to 1.0 s (RT above 1.1 s)"),
    ("C6b", "primary", "ci_low", 0.0, "Correlation of the 0 to 1.0 s map (1,290 cells) with the HBN CV map"),
    ("C7", "primary", "ci_high", 0.96, "Slow/fast CPP build-up rate ratio (onset/rate estimator)"),
    ("S1", "secondary", "ci_high", 0.71, "Slow/fast CPP build-up rate ratio (onset/rate estimator)"),
    ("S2", "secondary", "ci_low", 0.0, "Partial correlation of the normalised CPP slope with age, given sex"),
    ("S3", "secondary", "ci_low", 0.0, "Partial correlation of the normalised slope with EZ drift, given age spline and sex"),
    ("S4", "secondary", "ci_low", 0.0, "Attention beta for lapse rate (four dimensions, age spline, sex, release)"),
    ("S5", "secondary", "ci_low", 0.0, "Mean over channels of the within-child r of pre-stimulus delta power with log RT"),
]


# Tests of sets A and B: id, tier, bound, threshold, claim; the
# statistic each reads (SET_KEYS); and for power at the realised n, the set's lowest HBN estimate with the
# HBN CV standard error and its number of children (SET_POWER).
SET_HYPOTHESES: list[tuple[str, str, str, float, str]] = [
    ("A1", "primary (set A)", "ci_low", 0.0, "Accumulation model, R10/R11: per-child NLL(none) - NLL(full)"),
    ("A2", "primary (set A)", "ci_low", 0.0, "Accumulation model: rank partial r of the none-model drift gain with age, given sex"),
    ("A3", "secondary (set A)", "ci_low", 0.11, "Accumulation model: per-child NLL(onset) - NLL(rate)"),
    ("A4", "secondary (set A)", "ci_low", 0.0, "Accumulation model: rank partial r of the drift gain with the normalised CPP slope"),
    ("A5", "secondary (set A)", "above_observed", float("nan"),
     "Accumulation model: interval of the predicted slow/fast CPP ratio above the observed ratio's interval"),
    ("B1", "secondary (set B)", "ci_low", 0.0, "Symbol Search, R10: early time-course decoder, within-child r with log item time"),
    ("B2", "secondary (set B)", "ci_low", 0.0, "Symbol Search, R10: posterior-only early time-course decoder, within-child r"),
    ("B3", "secondary (set B)", "ci_low", 0.0, "Symbol Search, R10: B1's decoder on consecutive-item differences"),
]
SET_KEYS = {"A1": "A_none_minus_full", "A2": "A_k_age", "A3": "A_onset_minus_rate",
            "A4": "A_k_with_cpp_slope_resp_norm", "A5": "A_cpp_ratio_predicted",
            "B1": "B_early_timecourse", "B2": "B_early_timecourse_posterior", "B3": "B_early_timecourse_diff"}
SET_POWER = {"A1": (5.337, (5.661 - 5.159) / 3.92, 1741), "A2": (0.364, (0.465 - 0.382) / 3.92, 1557),
             "A3": (0.370 - 0.11, (0.529 - 0.200) / 3.92, 1741), "A4": (0.098, (0.150 - 0.048) / 3.92, 1546),
             "A5": (0.904 - 0.686, (0.6186 - 0.4529) / 3.92, 1625),
             "B1": (0.136, (0.157 - 0.116) / 3.92, 792), "B2": (0.075, (0.096 - 0.054) / 3.92, 792),
             "B3": (0.131, (0.157 - 0.105) / 3.92, 792)}


def power_scaled(n: int, estimate: float, se_ref: float, n_ref: int) -> float:
    from scipy.stats import norm

    if not n:
        return float("nan")
    return float(norm.cdf(abs(estimate) / (se_ref * np.sqrt(n_ref / n)) - 1.96))


def alias_set_results(results: dict) -> None:
    for hid, key in SET_KEYS.items():
        if key in results:
            results[hid] = dict(results[key])
            if hid in SET_POWER:
                results[hid]["power_at_realised_n"] = power_scaled(results[hid]["n_children"], *SET_POWER[hid])


def set_a_tests(summary: dict) -> dict:
    """Statistics of set A from accumulation_model_analysis.py's summary (the tests are listed in SET_HYPOTHESES)."""
    out = {}
    for name, comp in summary["comparison"].items():
        if isinstance(comp, dict) and "per_child" in comp:
            out[f"A_{name}"] = {"estimate": comp["per_child"], "ci_low": comp["per_child_ci"][0],
                                "ci_high": comp["per_child_ci"][1], "n_children": summary["n_children"]}
    for name, link in summary.get("links", {}).items():
        out[f"A_{name}"] = {"estimate": link["r"], "ci_low": link["ci"][0], "ci_high": link["ci"][1], "n_children": link["n"]}
    for p, d in summary["development"].items():
        part = d["partial_age_given_sex"]
        out[f"A_{p}_age"] = {"estimate": part["r"], "ci_low": part["ci"][0], "ci_high": part["ci"][1], "n_children": part["n"]}
    obs, pred = summary["cpp_ratio_observed"], summary["cpp_ratio_predicted"].get("full_norm")
    if pred:
        out["A_cpp_ratio_predicted"] = {"estimate": pred["ratio"], "ci_low": pred["ci"][0], "ci_high": pred["ci"][1],
                                        "n_children": pred["n_children"], "observed_ci_high": obs["ci"][1],
                                        "note": f"observed {obs['ratio']:.3f} [{obs['ci'][0]:.3f}, {obs['ci'][1]:.3f}]"}
    return out


def set_b_tests(b: dict) -> dict:
    """Statistics of set B from symbol_search.py --step external."""
    out = {}
    dec, tl = b["decomposition"], b["trial_level"]
    for _, r in dec[dec["model"] != "demographics"].iterrows():
        out[f"B_{r['window']}_{r['model']}"] = {"estimate": r["r_within"], "ci_low": r["r_within_ci_low"],
                                                 "ci_high": r["r_within_ci_high"], "n_children": int(r["n_children"])}
    for _, r in tl.iterrows():
        out[f"B_{r['window']}_{r['model']}_{r['variant']}"] = {"estimate": r["r"], "ci_low": r["ci_low"],
                                                                "ci_high": r["ci_high"], "n_children": int(r["n_children"])}
    return out


def script(paths: ProjectPaths, name: str):
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_component(paths: ProjectPaths, name: str, args: list[str], log_dir: Path) -> Path:
    """Run a frozen script and return the out_dir it reports; its console output is kept in log_dir."""
    proc = subprocess.run([sys.executable, "-m", script_module(name), *args],
                          capture_output=True, text=True, cwd=paths.root)
    (log_dir / f"{name}_{len(list(log_dir.glob(name + '_*')))}.log").write_text(
        proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed ({proc.returncode}); see {log_dir}")
    match = re.search(r'"out_dir": ("(?:[^"\\]|\\.)*")', proc.stdout)
    return Path(json.loads(match.group(1))) if match else Path()


def participants_file(root: Path) -> pd.DataFrame:
    path = root / "manifest" / "participants.csv"
    if path.is_file():
        return pd.read_csv(path, dtype={"subject": str})
    merged = pd.read_csv(root / "audit" / "participants_merged.csv", low_memory=False)
    people = merged.drop_duplicates("participant_id").assign(
        subject=lambda f: f["participant_id"].astype(str).str.replace("^sub-", "", regex=True))
    release = pd.read_parquet(root / "manifest" / "ccd_blocks.parquet").drop_duplicates("subject")[["subject", "release"]]
    people = people[["subject", "age", "sex", "ehq_total"]].merge(release, on="subject", how="inner")
    write_dataframe(people, path)
    return people


def participants_all(root: Path) -> Path:
    """Every participant of the root's audit (children without CCD blocks included) with the CBCL dimensions."""
    path = root / "manifest" / "participants_all.csv"
    if not path.is_file():
        merged = pd.read_csv(root / "audit" / "participants_merged.csv", low_memory=False)
        people = merged.drop_duplicates("participant_id").assign(
            subject=lambda f: f["participant_id"].astype(str).str.replace("^sub-", "", regex=True))
        cols = ["subject", "age", "sex", "ehq_total", *[c for c in TRAIT_COLUMNS if c in people.columns]]
        write_dataframe(people[cols], path)
    return path


def run_set_a(paths, root: Path, onset_rate_dir: Path, children: pd.DataFrame, out_dir: Path, log_dir: Path) -> dict:
    """Set A: the accumulation model fitted to the held-out children and analysed with their own CPP ratio,
    normalised CPP slopes, EZ drift and CBCL attention (accumulation_model_fit.py and accumulation_model_analysis.py)."""
    fits = run_component(paths, "accumulation_model_fit", ["--root", str(root), "--tag", "confirm", "--n-jobs", "24"],
                         log_dir)
    normalised_children = out_dir / "normalised_children"
    normalised_children.mkdir(exist_ok=True)
    write_dataframe(children[["subject", "cpp_slope_resp_norm", "ez_drift"]], normalised_children / "children.csv")
    analysis = run_component(
        paths, "accumulation_model_analysis", ["--fits", str(fits), "--tag", "confirm", "--onset-rate", str(onset_rate_dir),
                                               "--normalised", str(normalised_children), "--people", str(participants_all(root))],
        log_dir)
    return {"fits": str(fits), "analysis": str(analysis),
            "summary": json.loads((Path(analysis) / "summary.json").read_text(encoding="utf-8"))}


def run_set_b(paths, root: Path, sys_root: Path, sys_confirm_root: Path, log_dir: Path) -> dict:
    """Set B: the Symbol Search decoders refitted on all HBN Symbol Search children and applied to R10."""
    ext = run_component(
        paths, "symbol_search", ["--step", "external", "--root", str(sys_root), "--external-root", str(sys_confirm_root),
                                 "--people", str(participants_all(root))], log_dir)
    return {"dir": str(ext), "decomposition": pd.read_csv(Path(ext) / "decomposition.csv"),
            "trial_level": pd.read_csv(Path(ext) / "trial_level.csv")}


def child_sums(x: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-child sums of centred cross-products (children x cells), centred squares, and centred y squares."""
    order = np.argsort(subjects, kind="stable")
    s, x, y = subjects[order], np.asarray(x, dtype=np.float64)[order], np.asarray(y, dtype=np.float64)[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    counts = np.diff(np.r_[starts, len(s)])
    xc = x - np.repeat(np.add.reduceat(x, starts, axis=0) / counts[:, None], counts, axis=0)
    yc = y - np.repeat(np.add.reduceat(y, starts) / counts, counts)
    return (np.add.reduceat(xc * yc[:, None], starts, axis=0), np.add.reduceat(xc ** 2, starts, axis=0),
            np.add.reduceat(yc ** 2, starts))


def cell_r(sums, weights: np.ndarray) -> np.ndarray:
    sxy, sxx, syy = sums
    return (weights @ sxy) / np.sqrt((weights @ sxx) * (weights @ syy))


def bootstrap(sums, statistic, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    n = len(sums[2])
    rng = np.random.default_rng(seed)
    draws = [statistic(cell_r(sums, np.bincount(rng.integers(0, n, n), minlength=n).astype(float)))
             for _ in range(n_boot)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(statistic(cell_r(sums, np.ones(n)))), "ci_low": float(low), "ci_high": float(high),
            "n_children": int(n)}


def row_bootstrap(estimator, n: int, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    draws = [estimator(rng.integers(0, n, n)) for _ in range(n_boot)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(estimator(np.arange(n))), "ci_low": float(low), "ci_high": float(high), "n_children": n}


def map_tests(paths, root, people, maps_dir, external, bcfg, preproc) -> dict:
    """C6a, C6b and S5 with child-bootstrap intervals; point estimates checked against rt_information_maps.py."""
    import mne

    ch_names = list(mne.channels.make_standard_montage(preproc.montage).ch_names)
    maps = pd.read_csv(maps_dir / "maps.csv")
    ext_scope = "mipdb"                                   # rt_information_maps.py labels the external sample so
    out = {}
    for name, config, store, feature, min_rt in (("early", "windows_early.yaml", "windows-v1-early", "timecourse", 1.1),
                                                 ("prestim", "trials_and_windows.yaml", "windows-v1", "spectral", None)):
        spec = next(w for w in load_config(paths.root / "features" / config, TrialConfig).windows if w.name == name)
        window = load_window_data(root / store, spec, preproc.resample_hz)
        cache = FeatureCache(root / "features" / store, window, bcfg.aperiodic_regions, np.zeros((129, 3)), None,
                             n_jobs=bcfg.feature_n_jobs)
        x_all = cache.timecourse(bcfg.timecourse_bin_ms) if feature == "timecourse" else cache.spectral()
        rows = external.external_cohort(window, people, bcfg)
        rows = rows[rows["valid_response"].astype(bool)]
        if min_rt is not None:
            rows = rows[rows["rt_s"] > min_rt]
        sums = child_sums(x_all[rows["row"].to_numpy()], np.log(rows["rt_s"].to_numpy(float)), rows["subject"].to_numpy())
        script_map = maps[(maps["map"] == name) & (maps["scope"] == ext_scope)]["r"].to_numpy()
        check = float(np.max(np.abs(cell_r(sums, np.ones(len(sums[2]))) - script_map)))
        if name == "early":
            n_bins = x_all.shape[1] // len(ch_names)
            cells = np.array([ch_names.index(c) * n_bins + EARLY_BIN for c in cluster_channels()["cpp"]])
            out["C6a"] = bootstrap(sums, lambda r: r[cells].mean())
            hbn_cv = maps[(maps["map"] == name) & (maps["scope"] == "cv")]["r"].to_numpy()
            out["C6b"] = bootstrap(sums, lambda r: np.corrcoef(r, hbn_cv)[0, 1])
            out["C6a"]["script_check_max_abs_diff"] = out["C6b"]["script_check_max_abs_diff"] = check
        else:
            bands = [*DEFAULT_BANDS, "total"]
            delta = np.arange(len(ch_names)) + bands.index("delta") * len(ch_names)
            out["S5"] = bootstrap(sums, lambda r: r[delta].mean())
            out["S5"]["script_check_max_abs_diff"] = check
    return out


def child_tests(paths, root, people, assoc, normalised, bcfg) -> dict:
    """S2 and S3 (child measures and their amplitude normalisation) and S4 (lapse-rate model), child bootstrap."""
    measures = pd.read_parquet(root / "features" / "epochs-v1" / "trial_measures.parquet")
    usable = (measures["window_valid"].astype(bool) & ~measures["artifact_window"].astype(bool)).to_numpy()
    frame = measures[usable]
    per_child = frame[frame["valid_response"].astype(bool)].groupby("subject").size()
    frame = frame[frame["subject"].isin(per_child[per_child >= bcfg.min_valid_responses_per_subject].index)]
    demo = people.assign(age=pd.to_numeric(people["age"], errors="coerce"),
                         sex_male=people["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}))
    children = assoc.child_table(frame).merge(demo[["subject", "age", "sex_male"]], on="subject", how="left")
    spec = load_config(paths.root / "features" / "epochs.yaml", TrialConfig).windows[0]
    array, index, meta = WindowStore(root / "epochs-v1", spec, 250.0).open()
    for key in ("block_key", "trial_in_block"):
        if not (index[key].to_numpy() == measures[key].to_numpy()).all():
            raise ValueError(f"trial measures and epoch store are not aligned on {key}")
    sfreq, zero = float(meta["sfreq"]), int(round(-spec.tmin_s * float(meta["sfreq"])))
    picks = [meta["channels"].index(ch) for ch in cluster_channels()["cpp"]]
    sos = butter(4, 8.0, btype="low", fs=sfreq, output="sos")
    rms = {}
    rows = index[usable & measures["subject"].isin(children["subject"]).to_numpy()]
    for subject, part in rows.sort_values("row").groupby("subject", sort=True):
        cpp = sosfiltfilt(sos, np.asarray(array[part["row"].to_numpy()][:, picks], dtype=np.float64).mean(axis=1) * 1e6,
                          axis=-1)
        rms[subject] = normalised.pretarget_rms(cpp, zero)
    children["cpp_slope_resp_norm"] = children["cpp_slope_resp"] / children["subject"].map(rms)
    out = {}
    s2 = children.dropna(subset=["cpp_slope_resp_norm", "age", "sex_male"]).reset_index(drop=True)
    y2, x2, c2 = s2["cpp_slope_resp_norm"].to_numpy(float), s2["age"].to_numpy(float), s2[["sex_male"]].to_numpy(float)
    out["S2"] = row_bootstrap(lambda i: normalised.partial_r(y2[i], x2[i], c2[i]), len(s2))
    s3 = children.dropna(subset=["cpp_slope_resp_norm", "ez_drift", "age", "sex_male"]).reset_index(drop=True)
    knots = np.quantile(s3["age"], np.linspace(0, 1, 5))
    c3 = np.column_stack([assoc.natural_basis(s3["age"].to_numpy(), knots), s3["sex_male"]])
    y3, x3 = s3["ez_drift"].to_numpy(float), s3["cpp_slope_resp_norm"].to_numpy(float)
    out["S3"] = row_bootstrap(lambda i: normalised.partial_r(y3[i], x3[i], c3[i]), len(s3))
    merged_path = root / "audit" / "participants_merged.csv"
    traits = load_traits(pd.read_csv(merged_path, low_memory=False)) if merged_path.is_file() else pd.DataFrame()
    if traits.empty or not set(TRAIT_COLUMNS) <= set(traits.columns):
        out["S4"] = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_children": 0,
                     "note": "no CBCL dimensions in this sample"}
    else:
        data = children.merge(traits, on="subject").merge(
            measures.drop_duplicates("subject")[["subject", "release"]], on="subject", how="left")
        part = data.dropna(subset=["lapse_rate", "age", "sex_male", *TRAIT_COLUMNS]).reset_index(drop=True)
        knots = np.quantile(part["age"], np.linspace(0, 1, 5))
        releases = pd.get_dummies(part["release"], drop_first=True, dtype=float).to_numpy()
        design = np.column_stack([np.ones(len(part)), assoc.natural_basis(part["age"].to_numpy(), knots), part["sex_male"],
                                  releases, part[list(TRAIT_COLUMNS)].to_numpy(float)])
        y = part["lapse_rate"].to_numpy(float)
        y = (y - y.mean()) / y.std()
        col = design.shape[1] - len(TRAIT_COLUMNS) + list(TRAIT_COLUMNS).index("attention")
        out["S4"] = row_bootstrap(lambda i: np.linalg.lstsq(design[i], y[i], rcond=None)[0][col], len(part))
    out["_children"] = children
    return out


def power_at(n: int, p_m: float, m: int = 350) -> float:
    from scipy.stats import norm

    if not n or not np.isfinite(p_m):
        return float("nan")
    return float(norm.cdf((norm.ppf(p_m) + 1.96) * np.sqrt(n / m) - 1.96))


def verdict(result: dict, bound: str, threshold: float) -> bool:
    if bound == "above_observed":                     # A5: the predicted interval lies above the observed one
        low, high = result.get("ci_low", np.nan), result.get("observed_ci_high", np.nan)
        return bool(np.isfinite(low) and np.isfinite(high) and low > high)
    value = result.get(bound, np.nan)
    if not np.isfinite(value):
        return False
    return bool(value > threshold) if bound == "ci_low" else bool(value < threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=data_location("R10R11"))
    parser.add_argument("--sys-root", default=data_location("SYS"), help="HBN Symbol Search root (set B)")
    parser.add_argument("--sys-confirm-root", default=data_location("SYS_R10R11"),
                        help="held-out Symbol Search root (set B); 'none' skips set B")
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    root = Path(args.root)
    out_dir = paths.results / "replication_tests" / utc_timestamp()
    log_dir = out_dir / "components"
    log_dir.mkdir(parents=True, exist_ok=True)
    bcfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    people = participants_file(root)
    modules = ("external_validation", "trial_specificity", "single_trial_associations", "cpp_normalised_amplitude",
               "cpp_onset_rate_calibration")
    external, specificity, assoc, normalised, calibration = (script(paths, n) for n in modules)
    ext = ["--external-root", str(root)]
    if not (root / "features" / "epochs-v1" / "trial_measures.parquet").is_file():
        run_component(paths, "single_trial_measures", ["--root", str(root)], log_dir)
    dirs = {"external": run_component(paths, "external_validation", [*ext, "--models", "mean", "demographics",
                                                                    "spectral", "timecourse"], log_dir),
            "external_eegnet": run_component(paths, "external_validation", [*ext, "--models", "mean", "demographics",
                                                                           "eegnet"], log_dir),
            "preresponse": run_component(paths, "preresponse_decoding_mipdb", ext, log_dir),
            "maps": run_component(paths, "rt_information_maps", ext, log_dir),
            "onset_rate": run_component(paths, "cpp_onset_rate", ["--root", str(root)], log_dir)}
    results = {}
    dec = pd.concat([pd.read_csv(dirs[k] / "decomposition.csv") for k in ("external", "external_eegnet")],
                    ignore_index=True)
    for hid, window, model in (("C1", "challenge", "timecourse"), ("C2", "challenge", "eegnet"), ("C3", "prestim", "spectral")):
        r = dec[(dec["window"] == window) & (dec["model"] == model)].iloc[0]
        results[hid] = {"estimate": r["r_within"], "ci_low": r["r_within_ci_low"], "ci_high": r["r_within_ci_high"],
                        "n_children": int(r["n_children"])}
    preresponse = pd.read_csv(dirs["preresponse"] / "decomposition.csv").set_index("window").loc["E1"]
    results["C4"] = {"estimate": preresponse["r_within"], "ci_low": preresponse["r_within_ci_low"], "ci_high": preresponse["r_within_ci_high"],
                     "n_children": int(preresponse["n_children"])}
    preds = pd.read_parquet(dirs["external"] / "predictions.parquet")
    trials = pd.read_parquet(root / "trials-v1" / "trials_ccd.parquet",
                             columns=["subject", "block_key", "trial_index", "trial_in_block", "prev_rt_s"])
    for hid, window, model in (("C5a", "challenge", "timecourse"), ("C5b", "prestim", "spectral")):
        part = preds[(preds["window"] == window) & (preds["model"] == model) & (preds["target"] == "rt_s")]
        merged = part.merge(trials, on=["subject", "block_key", "trial_index"], how="left", validate="one_to_one")
        frame = merged.rename(columns={"y_pred": "pred", "y_true": "rt"})[
            ["subject", "block_key", "trial_in_block", "prev_rt_s", "pred", "rt"]].reset_index(drop=True)
        r = specificity.pooled_r(specificity.variant_frame(frame, "diff"), N_BOOT, SEED)
        results[hid] = {"estimate": r["r"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n_children": r["n_children"]}
    results.update(map_tests(paths, root, people, dirs["maps"], external, bcfg, preproc))
    ratio = calibration.share_and_ratio(pd.read_csv(dirs["onset_rate"] / "child_tertiles.csv"), N_BOOT, SEED)
    for hid in ("C7", "S1"):
        results[hid] = {"estimate": ratio["rate_ratio"], "ci_low": ratio["rate_ratio_ci"][0],
                        "ci_high": ratio["rate_ratio_ci"][1], "n_children": ratio["n_children"]}
    child = child_tests(paths, root, people, assoc, normalised, bcfg)
    children = child.pop("_children")
    write_dataframe(children, out_dir / "child_measures.csv")
    results.update(child)
    later_sets = {"A": run_set_a(paths, root, dirs["onset_rate"], children, out_dir, log_dir)}
    results.update(set_a_tests(later_sets["A"]["summary"]))
    if args.sys_confirm_root != "none":
        later_sets["B"] = run_set_b(paths, root, Path(args.sys_root), Path(args.sys_confirm_root), log_dir)
        results.update(set_b_tests(later_sets["B"]))
    alias_set_results(results)
    rows = []
    for hid, tier, bound, threshold, claim in HYPOTHESES + SET_HYPOTHESES:
        if hid not in results:
            continue
        res = results[hid]
        rows.append({"id": hid, "tier": tier, "claim": claim, "estimate": res["estimate"], "ci_low": res["ci_low"],
                     "ci_high": res["ci_high"], "n_children": res["n_children"],
                     "criterion": ("predicted interval above the observed interval" if bound == "above_observed" else
                                   f"{'lower' if bound == 'ci_low' else 'upper'} bound {'>' if bound == 'ci_low' else '<'} {threshold}"),
                     "supported": verdict(res, bound, threshold),
                     "note": res.get("note", ""),
                     "power_at_realised_n": (power_at(res["n_children"], POWER_350[hid][0]) if hid in POWER_350
                                             else res.get("power_at_realised_n", float("nan"))),
                     "power_is_lower_bound": POWER_350.get(hid, (0, False))[1],
                     "script_check_max_abs_diff": res.get("script_check_max_abs_diff", np.nan)})
    tests = pd.DataFrame(rows)
    write_dataframe(tests, out_dir / "tests.csv")
    summary = {"git_commit": git_commit(paths.root), "argv": sys.argv, "root": str(root),
               "components": {k: str(v) for k, v in dirs.items()},
               "n_children_benchmark": results["C1"]["n_children"],
               "later_sets": {k: {kk: vv for kk, vv in v.items() if kk in ("fits", "analysis", "dir")} for k, v in later_sets.items()},
               "tests": tests.to_dict(orient="records")}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir)}, indent=1))
    print(tests[["id", "estimate", "ci_low", "ci_high", "n_children", "criterion", "supported"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
