"""The CPP cluster averaged within RT tertiles (a check of its canonical behaviour).

Averages the CPP cluster (8 Hz low-pass, -100 to 0 ms pre-target baseline) over trials split into tertiles of RT
within each child, both target-locked and response-locked. The expected pattern (O'Connell et al. 2012; Kelly and
O'Connell 2013) is a positive build-up that is steeper for fast responses when target-locked, and that reaches a
similar level at the response when response-locked. Writes results/cpp_by_rt_tertile/<run>/ (waveforms and summary).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths
from common.schema import DerivativesConfig, TrialConfig
from features.clusters import cluster_channels
from features.windows import WindowStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-trials", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    spec = load_config(paths.root / "features" / "epochs.yaml", TrialConfig).windows[0]
    array, index, meta = WindowStore(deriv.root / "epochs-v1", spec, 250.0).open()
    sfreq, zero = float(meta["sfreq"]), int(round(-spec.tmin_s * float(meta["sfreq"])))
    measures = pd.read_parquet(deriv.root / "features" / "epochs-v1" / "trial_measures.parquet")
    keep = measures["cpp_slope_resp"].notna()
    trials = index[keep.to_numpy()].copy()
    trials["tertile"] = trials.groupby("subject")["rt_s"].transform(
        lambda r: pd.qcut(r.rank(method="first"), 3, labels=False) if len(r) >= 6 else np.nan)
    trials = trials.dropna(subset=["tertile"])
    trials = trials.sample(n=min(args.max_trials, len(trials)), random_state=args.seed).sort_values("row")
    picks = [meta["channels"].index(ch) for ch in cluster_channels()["cpp"]]
    sos = butter(4, 8.0, btype="low", fs=sfreq, output="sos")
    cpp = sosfiltfilt(sos, np.asarray(array[trials["row"].to_numpy()][:, picks]).mean(axis=1) * 1e6, axis=-1)
    cpp -= cpp[:, zero - 25:zero].mean(axis=1, keepdims=True)
    t_target = (np.arange(cpp.shape[1]) - zero) / sfreq
    resp_offsets = np.arange(-250, 51)                       # -1.0 to +0.2 s around the response
    rows, summary = [], {}
    for tertile, part in trials.groupby("tertile"):
        idx = trials.index.get_indexer(part.index)
        target_avg = cpp[idx].mean(axis=0)
        resp = np.full((len(idx), len(resp_offsets)), np.nan)
        for j, (i, rt) in enumerate(zip(idx, part["rt_s"].to_numpy(), strict=True)):
            at = zero + int(round(rt * sfreq)) + resp_offsets
            ok = (at >= 0) & (at < cpp.shape[1])
            resp[j, ok] = cpp[i, at[ok]]
        resp_avg = np.nanmean(resp, axis=0)
        name = ["fast", "middle", "slow"][int(tertile)]
        rows += [{"tertile": name, "lock": "target", "t_s": t, "uv": v} for t, v in zip(t_target, target_avg, strict=True)]
        rows += [{"tertile": name, "lock": "response", "t_s": o / sfreq, "uv": v} for o, v in zip(resp_offsets, resp_avg, strict=True)]
        pre = (resp_offsets >= -88) & (resp_offsets <= -12)
        slope = np.polyfit(resp_offsets[pre] / sfreq, resp_avg[pre], 1)[0]
        summary[name] = {"n_trials": len(idx), "median_rt_s": float(part["rt_s"].median()),
                         "resp_locked_peak_uv": float(np.nanmax(resp_avg[(resp_offsets >= -50) & (resp_offsets <= 25)])),
                         "resp_locked_slope_uv_per_s": float(slope),
                         "target_locked_uv_at_500ms": float(target_avg[zero + 125])}
    out_dir = paths.results / "cpp_by_rt_tertile" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    waves = pd.DataFrame(rows)
    write_dataframe(waves, out_dir / "cpp_by_rt_tertile.csv")
    write_json(summary, out_dir / "summary.json")
    print(json.dumps({"out_dir": str(out_dir), **summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
