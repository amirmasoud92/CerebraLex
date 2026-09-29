"""Robustness of the slow/fast CPP build-up ratio, and the sensory evidence (post hoc).

Build-up ratio. The trials and per-child tertile rates of cpp_onset_rate.py (slope of each tertile's
response-locked average) give
  E1  the ratio of child means (cpp_onset_rate_calibration.share_and_ratio, the estimator of test C7);
  E2  the ratio of 20% trimmed means across the same children;
the single-trial slopes (``cpp_slope_resp``) of the same trials give
  E3  a mixed model of the slopes on tertile (fast, slow) with child random intercepts (REML), ratio
      (b0 + b_slow) / b0 with a delta-method interval;
  E4  the pooled within-child correlation of the slope with log RT on all these trials.
HBN (all children), MIPDB and R10/R11.

Sensory evidence. Per trial, the SSVEP cluster mean, 200 ms rectangular sliding-window DFT amplitudes at 20 and 25 Hz;
the differential SSVEP is the amplitude at the target's frequency minus the other, baseline-corrected. Marker
validity, a slow/fast tertile contrast on trials with RT above 0.9 s, and a single-trial within-child r.

``--part ratio`` runs the build-up ratio only, ``--part ssvep`` the sensory evidence only. Writes
results/buildup_ratio/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import trim_mean

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import TrialConfig
from common.scripts import script_path
from decoding.decomposition import within_between
from features.clusters import cluster_channels
from features.windows import WindowStore

SAMPLES = {"HBN": data_location("HBN"), "MIPDB": data_location("MIPDB"),
           "R10R11": data_location("R10R11")}
ONSET_RATE_RUNS = {"HBN": "results/cpp_onset_rate/hbn", "MIPDB": "results/cpp_onset_rate/mipdb",
                   "R10R11": "results/cpp_onset_rate/r10_r11"}
MIN_TRIALS_ONSET_RATE, MIN_VALID = 6, 10
SFREQ, TMIN = 250.0, -1.0
WIN = 50                                   # 200 ms at 250 Hz: 4 cycles of 20 Hz and 5 of 25 Hz, so orthogonal
CENTRES = np.round(np.arange(-0.3, 1.5 + 1e-9, 0.02), 2)
BASELINE, RISE, LATE = (-0.3, -0.1), (0.2, 0.8), (0.6, 0.8)
FREQ = {"left": 20.0, "right": 25.0}       # left-tilted grating reverses at 20 Hz, right-tilted at 25 Hz (BIDS task text)
MIN_RT_SSVEP = 0.9
N_BOOT, SEED = 2000, 12345              # the defaults of cpp_onset_rate.py and its calibration


def sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, script_path(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_epochs(root: Path, spec):
    array, index, meta = WindowStore(root / "epochs-v1", spec, SFREQ).open()
    measures = pd.read_parquet(root / "features" / "epochs-v1" / "trial_measures.parquet")
    assert (index["row"].to_numpy() == np.arange(len(index))).all()
    assert (measures["block_key"].to_numpy() == index["block_key"].to_numpy()).all()
    assert (measures["trial_in_block"].to_numpy() == index["trial_in_block"].to_numpy()).all()
    return array, index, meta, measures


# ---------------------------------------------------------------- 2a
def onset_rate_children_and_trials(root: Path, spec, onset_rate) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The loop of cpp_onset_rate.main (per-child tertile measures), keeping the single-trial slopes and tertiles."""
    array, index, meta, measures = open_epochs(root, spec)
    sfreq, zero = float(meta["sfreq"]), int(round(-spec.tmin_s * float(meta["sfreq"])))
    eligible = index[measures["cpp_slope_resp"].notna().to_numpy()].copy()
    eligible["cpp_slope_resp"] = measures["cpp_slope_resp"].to_numpy()[eligible["row"].to_numpy()]
    counts = eligible.groupby("subject").size()
    eligible = eligible[eligible["subject"].isin(counts[counts >= MIN_TRIALS_ONSET_RATE].index)].sort_values("row")
    picks = [meta["channels"].index(ch) for ch in cluster_channels()["cpp"]]
    sos = butter(4, 8.0, btype="low", fs=sfreq, output="sos")
    rows, trials = [], []
    for subject, part in eligible.groupby("subject", sort=True):
        cpp = sosfiltfilt(sos, np.asarray(array[part["row"].to_numpy()][:, picks], dtype=np.float64).mean(axis=1) * 1e6,
                          axis=-1)
        cpp -= cpp[:, zero - int(round(0.1 * sfreq)):zero].mean(axis=1, keepdims=True)
        rt = part["rt_s"].to_numpy(float)
        tertile = pd.qcut(pd.Series(rt).rank(method="first"), 3, labels=False).to_numpy()
        row = {"subject": subject}
        for label, value in (("fast", 0), ("slow", 2)):
            m = onset_rate.child_tertile_measures(cpp[tertile == value], rt[tertile == value], sfreq, zero)
            row.update({f"{label}_{k}": v for k, v in m.items()})
        rows.append(row)
        trials.append(part[["subject", "rt_s", "cpp_slope_resp"]].assign(tertile=tertile))
    return pd.DataFrame(rows), pd.concat(trials, ignore_index=True)


def ratio_set(children: pd.DataFrame) -> pd.DataFrame:
    """The children share_and_ratio uses (complete onset, RT and rate differences)."""
    c = children.assign(diff_onset=children["slow_onset_s"] - children["fast_onset_s"],
                        diff_rt=children["slow_median_rt"] - children["fast_median_rt"])
    return c.dropna(subset=["diff_onset", "diff_rt", "slow_rate_uv_per_s", "fast_rate_uv_per_s"])


def trimmed_ratio(children: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    c = ratio_set(children)
    slow, fast = c["slow_rate_uv_per_s"].to_numpy(float), c["fast_rate_uv_per_s"].to_numpy(float)

    def stat(i):
        return trim_mean(slow[i], 0.2) / trim_mean(fast[i], 0.2)

    rng = np.random.default_rng(seed)
    draws = [stat(rng.integers(0, len(c), len(c))) for _ in range(n_boot)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(stat(np.arange(len(c)))), "ci_low": float(low), "ci_high": float(high),
            "n_children": len(c)}


def delta_ratio(b0: float, b1: float, cov: np.ndarray) -> tuple[float, float]:
    """(b0 + b1) / b0 and its delta-method standard error; cov is the 2 x 2 covariance of (b0, b1)."""
    grad = np.array([-b1 / b0 ** 2, 1.0 / b0])
    return (b0 + b1) / b0, float(np.sqrt(grad @ cov @ grad))


def mixed_model_ratio(trials: pd.DataFrame) -> dict:
    import statsmodels.formula.api as smf

    d = trials[trials["tertile"].isin([0, 2])].assign(slow=lambda f: (f["tertile"] == 2).astype(float))
    fit = smf.mixedlm("cpp_slope_resp ~ slow", d, groups=d["subject"]).fit(reml=True)
    b0, b1 = float(fit.params["Intercept"]), float(fit.params["slow"])
    ratio, se = delta_ratio(b0, b1, fit.cov_params().loc[["Intercept", "slow"], ["Intercept", "slow"]].to_numpy())
    return {"estimate": float(ratio), "ci_low": float(ratio - 1.96 * se), "ci_high": float(ratio + 1.96 * se),
            "n_children": int(d["subject"].nunique()), "n_trials": len(d), "b0": b0, "b_slow": b1,
            "converged": bool(fit.converged)}


def analysis_ratio(samples: dict[str, Path], spec, root: Path, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    onset_rate, calibration = sibling("cpp_onset_rate"), sibling("cpp_onset_rate_calibration")
    rows, record = [], {}
    for name, sample_root in samples.items():
        children, trials = onset_rate_children_and_trials(sample_root, spec, onset_rate)
        write_dataframe(children, out_dir / f"onset_rate_children_{name}.csv")
        e1 = calibration.share_and_ratio(children, N_BOOT, SEED)
        rec = calibration.share_and_ratio(pd.read_csv(root / ONSET_RATE_RUNS[name] / "child_tertiles.csv"), N_BOOT, SEED)
        record[name] = {"record": ONSET_RATE_RUNS[name], "record_ratio": rec["rate_ratio"], "recomputed": e1["rate_ratio"],
                        "n_children_record": rec["n_children"], "n_children_recomputed": e1["n_children"]}
        rows.append({"sample": name, "estimator": "E1_ratio_of_child_means", "estimate": e1["rate_ratio"],
                     "ci_low": e1["rate_ratio_ci"][0], "ci_high": e1["rate_ratio_ci"][1], "n_children": e1["n_children"]})
        rows.append({"sample": name, "estimator": "E2_ratio_of_20pct_trimmed_means", **trimmed_ratio(children)})
        rows.append({"sample": name, "estimator": "E3_mixed_model_single_trial", **mixed_model_ratio(trials)})
        r = within_between(np.log(trials["rt_s"].to_numpy(float)), trials["cpp_slope_resp"].to_numpy(float),
                           trials["subject"].to_numpy(), n_bootstrap=N_BOOT, seed=SEED)
        rows.append({"sample": name, "estimator": "E4_within_child_r_slope_logRT", "estimate": r["r_within"],
                     "ci_low": r["r_within_ci_low"], "ci_high": r["r_within_ci_high"], "n_children": r["n_children"],
                     "n_trials": len(trials)})
    table = pd.DataFrame(rows)
    if not {"HBN", "R10R11"} <= set(samples):
        return table, {"record_check": record}

    def get(sample, estimator):
        return table[(table["sample"] == sample) & (table["estimator"] == estimator)].iloc[0]

    hbn_e1, conf_e1 = get("HBN", "E1_ratio_of_child_means")["estimate"], get("R10R11", "E1_ratio_of_child_means")["estimate"]
    closes = {}
    for est in ("E2_ratio_of_20pct_trimmed_means", "E3_mixed_model_single_trial"):
        h, c = get("HBN", est), get("R10R11", est)
        closes[est] = {"hbn_interval_includes_r10r11_e1": bool(h["ci_low"] <= conf_e1 <= h["ci_high"]),
                       "r10r11_interval_includes_hbn_e1": bool(c["ci_low"] <= hbn_e1 <= c["ci_high"])}
        closes[est]["met"] = any(closes[est].values())
    e4 = get("R10R11", "E4_within_child_r_slope_logRT")
    rule = {"record_check": record,
            "E1_reproduces_record_HBN": bool(abs(record["HBN"]["recomputed"] - record["HBN"]["record_ratio"]) < 0.01),
            "hbn_e1": float(hbn_e1), "r10r11_e1": float(conf_e1), "per_estimator": closes,
            "estimator_fragility_E2_and_E3": all(v["met"] for v in closes.values()),
            "estimator_fragility_E2_or_E3": any(v["met"] for v in closes.values()),
            "single_trial_slower_buildup_R10R11": bool(e4["ci_high"] < 0)}
    return table, rule


# ---------------------------------------------------------------- 2b
def dft_templates(freqs: tuple[float, ...], n: int = WIN, sfreq: float = SFREQ) -> np.ndarray:
    t = np.arange(n) / sfreq
    return np.stack([np.exp(-2j * np.pi * f * t) for f in freqs])          # (n_freq, n)


def between(lo: float, hi: float) -> np.ndarray:
    return (CENTRES >= lo - 1e-9) & (CENTRES <= hi + 1e-9)


def differential_ssvep(signal: np.ndarray, target_hz: np.ndarray) -> np.ndarray:
    """Per trial (rows of ``signal``: SSVEP cluster mean in volts, epoch from TMIN at SFREQ): amplitude (uV) at the
    target's frequency minus the other, in 200 ms windows centred on CENTRES, minus its mean over BASELINE centres."""
    zero = int(round(-TMIN * SFREQ))
    starts = np.array([zero + int(round((c - WIN / SFREQ / 2) * SFREQ)) for c in CENTRES])
    segs = np.stack([signal[:, s:s + WIN] for s in starts], axis=1)          # (n, n_centres, WIN)
    amp = np.abs(segs @ dft_templates((20.0, 25.0)).T) * 2.0 / WIN * 1e6    # (n, n_centres, 2)
    diff = np.where((target_hz == 20.0)[:, None], amp[:, :, 0] - amp[:, :, 1], amp[:, :, 1] - amp[:, :, 0])
    return diff - diff[:, between(*BASELINE)].mean(axis=1, keepdims=True)


def ssvep_trials(root: Path, spec, chunk: int = 2048) -> tuple[pd.DataFrame, np.ndarray]:
    array, index, meta, _ = open_epochs(root, spec)
    ok = (index["window_valid"].astype(bool) & ~index["artifact_window"].astype(bool)
          & index["target_side"].isin(list(FREQ))).to_numpy()
    valid_counts = index[index["valid_response"].astype(bool)].groupby("subject").size()
    ok = ok & (index["subject"].map(valid_counts).fillna(0).to_numpy() >= MIN_VALID)
    idx = index[ok].reset_index(drop=True)
    picks = [meta["channels"].index(ch) for ch in cluster_channels()["ssvep"]]
    target_hz = idx["target_side"].map(FREQ).to_numpy(float)
    rows = idx["row"].to_numpy()
    out = np.empty((len(idx), len(CENTRES)), dtype=np.float64)
    for s in range(0, len(idx), chunk):
        sig = np.asarray(array[rows[s:s + chunk]][:, picks], dtype=np.float64).mean(axis=1)
        out[s:s + chunk] = differential_ssvep(sig, target_hz[s:s + chunk])
    return idx, out


def boot_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"estimate": float(v.mean()), "ci_low": float(low), "ci_high": float(high), "n_children": len(v)}


def rise_slope(diff: np.ndarray) -> np.ndarray:
    t = CENTRES[between(*RISE)]
    t = t - t.mean()
    return diff[:, between(*RISE)] @ t / (t @ t)


def child_curve(diff: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Mean over children of each child's mean curve."""
    return pd.DataFrame(diff).groupby(subjects).mean().mean().to_numpy()


def analysis_ssvep(samples: dict[str, Path], spec) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validity, tertile_rows, single_rows, curves = {}, [], [], []
    for name, root in samples.items():
        idx, diff = ssvep_trials(root, spec)
        frame = idx[["subject", "rt_s", "valid_response"]].assign(amp=diff[:, between(*LATE)].mean(axis=1),
                                                                  slope=rise_slope(diff))
        validity[name] = {**boot_mean(frame.groupby("subject")["amp"].mean().to_numpy()), "n_trials": len(frame)}
        curves.append(pd.DataFrame({"sample": name, "group": "all", "centre_s": CENTRES,
                                    "ssvep_uv": child_curve(diff, idx["subject"].to_numpy())}))
        keep = (frame["valid_response"].astype(bool) & (frame["rt_s"] > MIN_RT_SSVEP)).to_numpy()
        sub, sub_diff = frame[keep].reset_index(drop=True), diff[keep]
        n_per = sub.groupby("subject")["rt_s"].transform("size")
        sub["tertile"] = np.nan
        big = (n_per >= 3).to_numpy()
        sub.loc[big, "tertile"] = sub[big].groupby("subject")["rt_s"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)).astype(float)
        per = sub[sub["tertile"].isin([0.0, 2.0])].groupby(["subject", "tertile"])[["amp", "slope"]].mean().unstack().dropna()
        for measure in ("amp", "slope"):
            fast, slow = per[(measure, 0.0)], per[(measure, 2.0)]
            b = boot_mean((slow - fast).to_numpy())
            tertile_rows.append({"sample": name, "measure": measure, "fast": float(fast.mean()), "slow": float(slow.mean()),
                                 "slow_minus_fast": b["estimate"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                                 "n_children": b["n_children"]})
        both = sub["subject"].isin(per.index).to_numpy()
        for label, value in (("fast", 0.0), ("slow", 2.0)):
            mask = both & (sub["tertile"] == value).to_numpy()
            curves.append(pd.DataFrame({"sample": name, "group": label, "centre_s": CENTRES,
                                        "ssvep_uv": child_curve(sub_diff[mask], sub.loc[mask, "subject"].to_numpy())}))
        r = within_between(sub["rt_s"].to_numpy(float), sub["amp"].to_numpy(float), sub["subject"].to_numpy(),
                           n_bootstrap=N_BOOT, seed=SEED)
        single_rows.append({"sample": name, "n_trials": len(sub), **r})
    return validity, pd.DataFrame(tertile_rows), pd.DataFrame(single_rows), pd.concat(curves, ignore_index=True)


def rule_ssvep(validity: dict, tertiles: pd.DataFrame) -> dict:
    amp = tertiles[tertiles["measure"] == "amp"].set_index("sample")
    grand = validity["HBN"]["estimate"]
    h = amp.loc["HBN"]
    rule = {"marker_valid": bool(validity["HBN"]["ci_low"] > 0), "grand_average_uv_HBN": grand,
            "sensory_contribution": bool(h["ci_high"] < 0 and all(amp.loc[s, "slow_minus_fast"] < 0
                                                                  for s in ("MIPDB", "R10R11") if s in amp.index)),
            "post_sensory_delay": bool(-0.25 * grand < h["ci_low"] and h["ci_high"] < 0.25 * grand)}
    rule["verdict"] = ("failed marker" if not rule["marker_valid"] else "sensory contribution" if rule["sensory_contribution"]
                       else "post-sensory delay" if rule["post_sensory_delay"] else "inconclusive")
    return rule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["ratio", "ssvep", "both"], default="both")
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    samples = {k: Path(SAMPLES[k]) for k in args.samples}
    spec = load_config(paths.root / "features" / "epochs.yaml", TrialConfig).windows[0]
    assert abs(spec.tmin_s - TMIN) < 1e-9
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "buildup_ratio" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    summary = {"git_commit": commit, "out_dir": str(out_dir)}
    if args.part in ("ratio", "both"):
        table, rule = analysis_ratio(samples, spec, paths.root, out_dir)
        write_dataframe(table, out_dir / "slope_ratio.csv")
        summary["ratio"] = rule
    if args.part in ("ssvep", "both"):
        validity, tertiles, single, curves = analysis_ssvep(samples, spec)
        write_json(validity, out_dir / "ssvep_validity.json")
        write_dataframe(tertiles, out_dir / "ssvep_tertiles.csv")
        write_dataframe(single, out_dir / "ssvep_single_trial.csv")
        write_dataframe(curves, out_dir / "ssvep_curves.csv")
        rule = rule_ssvep(validity, tertiles) if "HBN" in validity else {}
        summary["ssvep"] = {"validity": validity, "rule": rule}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
