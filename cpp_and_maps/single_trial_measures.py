"""Single-trial signal measures for every trial of an epoch store.

Reads ``<root>/<store>/epoch`` (target-locked epochs, -1.0 to +2.5 s, cluster channels), computes the measures of
``features.trial_measures`` for valid, artifact-free windows in chunks, and writes
``<root>/features/<store>/trial_measures.parquet`` with the trial keys, behaviour, window quality flags and measures
(NaN where a measure does not apply). Works for the HBN, MIPDB and R10/R11 roots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_config
from common.io import write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths
from common.provenance import git_commit
from common.schema import DerivativesConfig, TrialConfig
from features.clusters import cluster_channels
from features.trial_measures import trial_measures
from features.windows import WindowStore

KEEP = ["block_key", "subject", "release", "block", "trial_in_block", "trial_index", "rt_s", "valid_response",
        "lapse", "miss", "correct", "window_valid", "artifact_window", "high_amplitude_window", "flat_window"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="derivatives root (default: HBN)")
    parser.add_argument("--store-name", default="epochs-v1")
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("single-trial-measures", paths.logs)
    deriv = load_config(paths.root / "common" / "derivatives.yaml", DerivativesConfig)
    root = Path(args.root) if args.root else deriv.root
    spec = load_config(paths.root / "features" / "epochs.yaml", TrialConfig).windows[0]
    store = WindowStore(root / args.store_name, spec, 250.0)
    array, index, meta = store.open()
    if not np.array_equal(index["row"].to_numpy(), np.arange(len(index))):
        raise ValueError("epoch index is not in storage order")
    clusters = cluster_channels()
    usable = (index["window_valid"].astype(bool) & ~index["artifact_window"].astype(bool)).to_numpy()
    parts = []
    for start in range(0, len(index), args.chunk):
        stop = min(start + args.chunk, len(index))
        rows = np.flatnonzero(usable[start:stop]) + start
        frame = pd.DataFrame(index=range(start, stop), columns=["cpp_slope_resp", "cpp_amp_resp", "cpp_slope_target",
                                                                  "alpha_logpow", "ssvep_logpow"], dtype=float)
        if len(rows):
            measures = trial_measures(np.asarray(array[rows], dtype=np.float64), meta["channels"], clusters,
                                      index.iloc[rows], float(meta["sfreq"]), spec.tmin_s)
            frame.loc[rows] = measures.to_numpy()
        parts.append(frame)
    measures = pd.concat(parts)
    out = pd.concat([index[[c for c in KEEP if c in index]].reset_index(drop=True), measures.reset_index(drop=True)], axis=1)
    out_dir = root / "features" / args.store_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(out, out_dir / "trial_measures.parquet")
    summary = {"root": str(root), "store": args.store_name, "git_commit": git_commit(paths.root),
               "clusters": clusters, "channels": meta["channels"], "n_trials": int(len(out)),
               "n_usable_windows": int(usable.sum()),
               "n_finite": {c: int(np.isfinite(out[c]).sum()) for c in measures.columns}}
    write_json(summary, out_dir / "trial_measures.json")
    log.info("trial measures written: %s", summary["n_finite"])
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
