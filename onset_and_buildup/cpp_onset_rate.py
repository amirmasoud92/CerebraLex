"""Does within-child RT variability arise from the onset of evidence accumulation or from its rate?

Per child with at least 6 eligible trials (valid responses, 0.65 s <= RT <= 2.4 s, artifact-free epoch), the CPP
cluster (8 Hz low-pass, -100 to 0 ms pre-target baseline) is averaged within the fast and the slow tertile of the
child's RTs:

* onset: target-locked 50% fractional peak latency, the first time after 0.2 s at which the average reaches half its
  maximum between 0.2 s and the tertile's median RT plus 0.1 s;
* rate: slope of the response-locked average from -350 to -50 ms;
* peak: maximum of the response-locked average from -200 to +100 ms.

Reports the mean slow-minus-fast differences across children with child-bootstrap intervals, and the share of the RT
difference that the onset difference accounts for. Writes results/cpp_onset_rate/<run>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import DerivativesConfig, TrialConfig
from features.clusters import cluster_channels
from features.windows import WindowStore


def child_tertile_measures(cpp: np.ndarray, rts: np.ndarray, sfreq: float, zero: int) -> dict:
    """Onset, rate and peak of one tertile's average CPP (``cpp``: trials x samples, baselined, uV)."""
    target_avg = cpp.mean(axis=0)
    med = float(np.median(rts))
    lo, hi = zero + int(round(0.2 * sfreq)), min(cpp.shape[1], zero + int(round((med + 0.1) * sfreq)))
    onset = np.nan
    if hi > lo + 1:
        peak = target_avg[lo:hi].max()
        if peak > 0:
            above = np.flatnonzero(target_avg[lo:hi] >= 0.5 * peak)
            onset = (lo + above[0] - zero) / sfreq if len(above) else np.nan
    offsets = np.arange(-int(round(0.35 * sfreq)), int(round(0.1 * sfreq)) + 1)
    resp = np.full((len(rts), len(offsets)), np.nan)
    for i, rt in enumerate(rts):
        at = zero + int(round(rt * sfreq)) + offsets
        ok = (at >= 0) & (at < cpp.shape[1])
        resp[i, ok] = cpp[i, at[ok]]
    resp_avg = np.nanmean(resp, axis=0)
    t = offsets / sfreq
    build = (t >= -0.35) & (t <= -0.05)
    rate = np.polyfit(t[build], resp_avg[build], 1)[0]
    peak_window = (t >= -0.2) & (t <= 0.1)
    return {"median_rt": med, "onset_s": onset, "rate_uv_per_s": float(rate),
            "peak_uv": float(np.nanmax(resp_avg[peak_window])), "n_trials": len(rts)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="derivatives root (default: HBN)")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--min-trials", type=int, default=6)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    root = Path(args.root) if args.root else deriv.root
    spec = load_config(paths.root / "features" / "epochs.yaml", TrialConfig).windows[0]
    array, index, meta = WindowStore(root / "epochs-v1", spec, 250.0).open()
    sfreq, zero = float(meta["sfreq"]), int(round(-spec.tmin_s * float(meta["sfreq"])))
    measures = pd.read_parquet(root / "features" / "epochs-v1" / "trial_measures.parquet")
    eligible = index[measures["cpp_slope_resp"].notna().to_numpy()].copy()
    counts = eligible.groupby("subject").size()
    eligible = eligible[eligible["subject"].isin(counts[counts >= args.min_trials].index)].sort_values("row")
    picks = [meta["channels"].index(ch) for ch in cluster_channels()["cpp"]]
    sos = butter(4, 8.0, btype="low", fs=sfreq, output="sos")
    rows = []
    for subject, part in eligible.groupby("subject", sort=True):
        cpp = sosfiltfilt(sos, np.asarray(array[part["row"].to_numpy()][:, picks], dtype=np.float64).mean(axis=1) * 1e6,
                          axis=-1)
        cpp -= cpp[:, zero - int(round(0.1 * sfreq)):zero].mean(axis=1, keepdims=True)
        rt = part["rt_s"].to_numpy(float)
        tertile = pd.qcut(pd.Series(rt).rank(method="first"), 3, labels=False).to_numpy()
        row = {"subject": subject}
        for label, value in (("fast", 0), ("slow", 2)):
            m = child_tertile_measures(cpp[tertile == value], rt[tertile == value], sfreq, zero)
            row.update({f"{label}_{k}": v for k, v in m.items()})
        rows.append(row)
    children = pd.DataFrame(rows)
    for measure in ("median_rt", "onset_s", "rate_uv_per_s", "peak_uv"):
        children[f"diff_{measure}"] = children[f"slow_{measure}"] - children[f"fast_{measure}"]
    rng = np.random.default_rng(args.seed)
    summary = {"root": str(root), "git_commit": git_commit(paths.root), "n_children": len(children)}
    complete = children.dropna(subset=["diff_onset_s", "diff_rate_uv_per_s", "diff_median_rt"])
    summary["n_children_complete"] = len(complete)
    draws = {m: [] for m in ("median_rt", "onset_s", "rate_uv_per_s", "peak_uv", "share")}
    for _ in range(args.n_bootstrap):
        s = complete.iloc[rng.integers(0, len(complete), len(complete))]
        for m in ("median_rt", "onset_s", "rate_uv_per_s", "peak_uv"):
            draws[m].append(s[f"diff_{m}"].mean())
        draws["share"].append(s["diff_onset_s"].mean() / s["diff_median_rt"].mean())
    for m, values in draws.items():
        estimate = (complete["diff_onset_s"].mean() / complete["diff_median_rt"].mean() if m == "share"
                    else complete[f"diff_{m}"].mean())
        low, high = np.percentile(values, [2.5, 97.5])
        summary[f"slow_minus_fast_{m}"] = {"estimate": float(estimate), "ci_low": float(low), "ci_high": float(high)}
    for label in ("fast", "slow"):
        summary[f"{label}_means"] = {m: float(complete[f"{label}_{m}"].mean())
                                     for m in ("median_rt", "onset_s", "rate_uv_per_s", "peak_uv")}
    out_dir = paths.results / "cpp_onset_rate" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(children, out_dir / "child_tertiles.csv")
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
