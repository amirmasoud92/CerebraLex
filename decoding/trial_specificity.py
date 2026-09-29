"""Is the within-child RT information trial to trial, or slow drift and carry-over?

The saved out-of-fold predictions of the primary RT decoders (benchmark: cross-validation and R5; external
validation: MIPDB) are compared with RT in five pooled within-child correlations:

1. ``child``: centred within child (the benchmark's within-child r);
2. ``block``: centred within block;
3. ``trend``: residuals after a linear trend on trial position within each block;
4. ``diff``: consecutive-trial differences within a block (adjacent trials, both in the cohort), centred within
   child; removes slow components;
5. ``carry``: residuals within child after block, trial position and the previous trial's RT (trials whose previous
   trial had a valid response); removes carry-over.

Child-bootstrap 95% intervals on per-child sums. Writes results/trial_specificity/<run>/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit

HBN_RUNS = {   # primary benchmark runs and the models taken from each
    "results/benchmark-v1/prestim": ("prestim", ("spectral", "riemann")),  # EEGNet from its own run
    "results/benchmark-v1/prestim_eegnet": ("prestim", ("eegnet",)),
    "results/benchmark-v1/challenge": ("challenge", ("spectral", "eegnet")),
    "results/benchmark-v1/challenge_timecourse": ("challenge", ("timecourse",)),
}
MIPDB_MODELS = {"prestim": ("spectral", "riemann", "eegnet"), "challenge": ("timecourse", "spectral", "eegnet")}
VARIANTS = ("child", "block", "trend", "diff", "carry")
KEYS = ["subject", "block_key", "trial_index"]


def _centre(frame: pd.DataFrame, columns: list[str], by: list[str]) -> pd.DataFrame:
    return frame[columns] - frame.groupby(by)[columns].transform("mean")


def _residualise(frame: pd.DataFrame, columns: list[str], by: list[str], design) -> pd.DataFrame:
    """Per group, least-squares residuals of ``columns`` on ``design(group)``; groups too small are dropped."""
    parts = []
    for _, g in frame.groupby(by, sort=False):
        x = design(g)
        if len(g) < x.shape[1] + 3:
            continue
        y = g[columns].to_numpy(float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        parts.append(pd.DataFrame(y - x @ beta, index=g.index, columns=columns))
    return pd.concat(parts) if parts else pd.DataFrame(columns=columns)


def variant_frame(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Columns ``a`` (prediction) and ``b`` (RT) transformed for one variant, with ``subject``."""
    cols = ["pred", "rt"]
    if variant == "child":
        out = _centre(frame, cols, ["subject"])
    elif variant == "block":
        out = _centre(frame, cols, ["subject", "block_key"])
    elif variant == "trend":
        out = _residualise(frame, cols, ["subject", "block_key"],
                           lambda g: np.column_stack([np.ones(len(g)), g["trial_in_block"].to_numpy(float)]))
    elif variant == "diff":
        f = frame.sort_values(["subject", "block_key", "trial_in_block"])
        same = (f["subject"].eq(f["subject"].shift()) & f["block_key"].eq(f["block_key"].shift())
                & f["trial_in_block"].diff().eq(1))
        d = (f[cols] - f[cols].shift())[same]
        d["subject"] = f.loc[same, "subject"]
        out = d[cols] - d.groupby("subject")[cols].transform("mean")
    elif variant == "carry":
        f = frame[frame["prev_rt_s"].notna()]

        def design(g):
            blocks = pd.get_dummies(g["block_key"], dtype=float).to_numpy()
            return np.column_stack([blocks, g["trial_in_block"].to_numpy(float), g["prev_rt_s"].to_numpy(float)])

        out = _residualise(f, cols, ["subject"], design)
    else:
        raise ValueError(variant)
    out = out.rename(columns={"pred": "a", "rt": "b"})
    out["subject"] = frame.loc[out.index, "subject"].to_numpy()
    return out


def pooled_r(transformed: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    """Pooled correlation of a and b with a child bootstrap on per-child sums."""
    t = transformed.dropna()
    sums = t.assign(ab=t["a"] * t["b"], aa=t["a"] ** 2, bb=t["b"] ** 2).groupby("subject")[["ab", "aa", "bb"]].sum()
    s = sums.to_numpy()
    r = s[:, 0].sum() / np.sqrt(s[:, 1].sum() * s[:, 2].sum())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        b = s[rng.integers(0, len(s), len(s))].sum(axis=0)
        draws.append(b[0] / np.sqrt(b[1] * b[2]))
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"n_children": len(s), "n_rows": len(t), "r": float(r), "ci_low": float(low), "ci_high": float(high)}


def load_predictions(root: Path, external_run: Path | None, external_eegnet_run: Path | None) -> pd.DataFrame:
    frames = []
    for run, (window, models) in HBN_RUNS.items():
        p = pd.read_parquet(root / run / "predictions.parquet")
        p = p[(p["target"] == "rt_s") & (p["window"] == window) & p["model"].isin(models)]
        frames.append(p.assign(source="HBN", scope=np.where(p["fold"] == -1, "holdout_R5", "cv"), run=run))
    for path in (external_run, external_eegnet_run):
        if path is None:
            continue
        p = pd.read_parquet(path / "predictions.parquet")
        keep = p["target"].eq("rt_s") & np.array([m in MIPDB_MODELS.get(w, ()) for w, m in zip(p["window"], p["model"], strict=True)])
        frames.append(p[keep].assign(source="MIPDB", scope="mipdb", run=str(path.relative_to(root))))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hbn-trials", default=f"{data_location('HBN')}/trials-v1/trials_ccd.parquet")
    parser.add_argument("--mipdb-trials", default=f"{data_location('MIPDB')}/trials-v1/trials_ccd.parquet")
    parser.add_argument("--external-run", default="results/external_validation/mipdb")
    parser.add_argument("--external-eegnet-run", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    preds = load_predictions(paths.root, paths.root / args.external_run if args.external_run else None,
                             paths.root / args.external_eegnet_run if args.external_eegnet_run else None)
    trial_cols = [*KEYS, "trial_in_block", "prev_rt_s"]
    trials = {"HBN": pd.read_parquet(args.hbn_trials, columns=trial_cols),
              "MIPDB": pd.read_parquet(args.mipdb_trials, columns=trial_cols)}
    rows = []
    for (source, scope, window, model), part in preds.groupby(["source", "scope", "window", "model"], sort=True):
        merged = part.merge(trials[source], on=KEYS, how="left", validate="one_to_one")
        if merged["trial_in_block"].isna().any():
            raise ValueError(f"{source} {window} {model}: predictions without a trial row")
        frame = merged.rename(columns={"y_pred": "pred", "y_true": "rt"})[
            ["subject", "block_key", "trial_in_block", "prev_rt_s", "pred", "rt"]].reset_index(drop=True)
        for variant in VARIANTS:
            rows.append({"source": source, "scope": scope, "window": window, "model": model, "variant": variant,
                         "run": part["run"].iloc[0],
                         **pooled_r(variant_frame(frame, variant), args.n_bootstrap, args.seed)})
        print(f"{source} {scope} {window} {model}: done", flush=True)
    table = pd.DataFrame(rows)
    out_dir = paths.results / "trial_specificity" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "trial_level.csv")
    write_json({"git_commit": commit, "argv": sys.argv}, out_dir / "provenance.json")
    wide = table.pivot_table(index=["source", "scope", "window", "model"], columns="variant", values="r")[list(VARIANTS)]
    print(json.dumps({"out_dir": str(out_dir)}))
    print(wide.to_string(float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
