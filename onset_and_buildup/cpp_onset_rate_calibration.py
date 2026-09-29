"""Calibration of the onset and build-up-rate estimators on semi-synthetic data.

The 50% fractional peak latency of an average moves when only the build-up rate differs, so the onset share of
cpp_onset_rate.py cannot separate an onset account from a rate account on its own. This runs the unchanged
estimators on data with a known answer:

* noise: each eligible trial's CPP-cluster mean is replaced by a phase-randomised surrogate of itself (same
  amplitude spectrum, random phases; nothing stays locked to the target or the response);
* signal: a ramp that reaches the peak amplitude (default 6 uV) at the response and returns linearly to 0 over
  300 ms, with onset (1 - w) t0 + w (RT - 0.45 s); w = 0 is a pure rate account (common onset t0, default 0.2 s),
  w = 1 a pure onset account (fixed 0.45 s build-up). ``--onset0`` and ``--peak`` give the sensitivity grid.

For each w the onset share and the slow/fast rate ratio are compared with the observed values; a w is consistent
with the data when both simulated values lie inside the observed 95% intervals. A noise-free run of each w (signal
only) is reported as well, to separate estimator bias from noise. Writes results/cpp_onset_rate_calibration/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from common.scripts import script_path
from features.clusters import cluster_channels
from features.windows import WindowStore

W_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
PEAK_UV = 6.0
RETURN_S = 0.3
ONSET0_S = 0.2
MIN_RT_S = 0.65
BUILD_S = 0.45


def _onset_rate():
    spec = importlib.util.spec_from_file_location("cpp_onset_rate", script_path("cpp_onset_rate"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate of each row with its amplitude spectrum and uniform random phases (DC and Nyquist kept)."""
    spectrum = np.fft.rfft(x, axis=-1)
    out = np.abs(spectrum) * np.exp(1j * rng.uniform(0, 2 * np.pi, spectrum.shape))
    out[:, 0] = spectrum[:, 0]
    if x.shape[-1] % 2 == 0:
        out[:, -1] = spectrum[:, -1]
    return np.fft.irfft(out, n=x.shape[-1], axis=-1)


def ramp(rts: np.ndarray, w: float, n_samples: int, sfreq: float, zero: int,
         onset0: float = ONSET0_S, peak: float = PEAK_UV) -> np.ndarray:
    """Linear build-up from onset to ``peak`` at the response, then a linear return to 0 over RETURN_S."""
    t = (np.arange(n_samples) - zero) / sfreq
    onset = (1 - w) * onset0 + w * (rts - BUILD_S)
    up = peak * (t[None, :] - onset[:, None]) / (rts - onset)[:, None]
    down = peak * (1 - (t[None, :] - rts[:, None]) / RETURN_S)
    signal = np.where((t >= onset[:, None]) & (t <= rts[:, None]), up, 0.0)
    return np.where((t > rts[:, None]) & (t <= rts[:, None] + RETURN_S), down, signal)


def child_row(onset_rate, cpp: np.ndarray, rt: np.ndarray, sfreq: float, zero: int) -> dict:
    """Fast and slow tertile measures of one child, exactly as in cpp_onset_rate.py."""
    tertile = pd.qcut(pd.Series(rt).rank(method="first"), 3, labels=False).to_numpy()
    row = {}
    for label, value in (("fast", 0), ("slow", 2)):
        m = onset_rate.child_tertile_measures(cpp[tertile == value], rt[tertile == value], sfreq, zero)
        row.update({f"{label}_{k}": v for k, v in m.items()})
    return row


def share_and_ratio(children: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    """Onset share (mean onset difference / mean RT difference) and slow/fast rate ratio with intervals."""
    c = children.assign(diff_onset=children["slow_onset_s"] - children["fast_onset_s"],
                        diff_rt=children["slow_median_rt"] - children["fast_median_rt"])
    c = c.dropna(subset=["diff_onset", "diff_rt", "slow_rate_uv_per_s", "fast_rate_uv_per_s"])

    def stat(s):
        return (s["diff_onset"].mean() / s["diff_rt"].mean(),
                s["slow_rate_uv_per_s"].mean() / s["fast_rate_uv_per_s"].mean())

    rng = np.random.default_rng(seed)
    draws = np.array([stat(c.iloc[rng.integers(0, len(c), len(c))]) for _ in range(n_bootstrap)])
    share, ratio = stat(c)
    return {"n_children": len(c), "share": float(share), "share_ci": np.percentile(draws[:, 0], [2.5, 97.5]).tolist(),
            "rate_ratio": float(ratio), "rate_ratio_ci": np.percentile(draws[:, 1], [2.5, 97.5]).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="derivatives root (default: HBN)")
    parser.add_argument("--observed", required=True, help="cpp_onset_rate.py result directory of the same root")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--min-trials", type=int, default=6)
    parser.add_argument("--onset0", type=float, nargs="+", default=[ONSET0_S], help="common onsets t0 (s)")
    parser.add_argument("--peak", type=float, nargs="+", default=[PEAK_UV], help="ramp amplitudes (uV)")
    args = parser.parse_args()
    onset_rate = _onset_rate()
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
    base = slice(zero - int(round(0.1 * sfreq)), zero)
    rng = np.random.default_rng(args.seed)
    rows = []
    for subject, part in eligible.groupby("subject", sort=True):
        raw = np.asarray(array[part["row"].to_numpy()][:, picks], dtype=np.float64).mean(axis=1) * 1e6
        noise = phase_randomise(raw, rng)
        rt = part["rt_s"].to_numpy(float)
        for onset0, peak, w in ((o, a, w) for o in args.onset0 for a in args.peak for w in W_VALUES):
            signal = ramp(rt, w, raw.shape[1], sfreq, zero, onset0, peak)
            for condition, x in (("surrogate", noise + signal), ("noise_free", signal)):
                cpp = sosfiltfilt(sos, x, axis=-1)
                cpp -= cpp[:, base].mean(axis=1, keepdims=True)
                rows.append({"subject": subject, "onset0": onset0, "peak": peak, "w": w, "condition": condition,
                             **child_row(onset_rate, cpp, rt, sfreq, zero)})
    sims = pd.DataFrame(rows)
    observed = share_and_ratio(pd.read_csv(Path(args.observed) / "child_tertiles.csv"), args.n_bootstrap, args.seed)
    summary = {"root": str(root), "observed_dir": str(args.observed), "git_commit": git_commit(paths.root),
               "observed": observed, "simulated": {}}
    for (condition, onset0, peak, w), part in sims.groupby(["condition", "onset0", "peak", "w"]):
        s = share_and_ratio(part, 500, args.seed)
        s["consistent"] = bool(observed["share_ci"][0] <= s["share"] <= observed["share_ci"][1]
                               and observed["rate_ratio_ci"][0] <= s["rate_ratio"] <= observed["rate_ratio_ci"][1])
        key = f"{condition}_w{w:.2f}" if len(args.onset0) == len(args.peak) == 1 else f"{condition}_t{onset0:.1f}_a{peak:g}_w{w:.2f}"
        summary["simulated"][key] = s
    out_dir = paths.results / "cpp_onset_rate_calibration" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(sims, out_dir / "simulated_child_tertiles.parquet")
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
