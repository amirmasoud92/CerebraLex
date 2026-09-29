"""Prediction horizon: how does within-child RT prediction grow with the time available before the response?

For T in T_VALUES, windows [0, T] after the target are assembled from the time-course features (100 ms bins) of the
early (0 to 1.0 s) and post-target (0.5 to 2.5 s) stores. The trial set S_T: valid responses with RT above
T + margin, the [0, T] segment clean under the window QC (applied to that segment through 100 ms bin extremes), a
clean pre-stimulus window, and children with at least 10 valid responses and at least 5 trials in S_T.

Decoders (benchmark settings), trained for each T on S_T only, in the frozen folds with R5 held out:
  A  time course of [0, T], all channels;       B  the same, posterior channels (template y below 0);
  C  pre-stimulus band power (-2 to 0 s);       D  A and C together.
References on S_T: the benchmark's post-target predictions (time course "E_tc", EEGNet "E_eegnet").
MIPDB and R10/R11: A to D refitted on all HBN children's S_T trials and applied.
Writes results/prediction_horizon/<run>/ (curve.csv, paired.csv, qc_check.json, predictions.parquet, summary.json).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.config import load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import BenchmarkConfig, PreprocessingConfig, TrialConfig, WindowQCConfig
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.benchmark import FeatureCache, demographics_table, load_window_data
from decoding.decomposition import within_between
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD, SplitAssignment

T_VALUES = (0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7)
MARGIN, SENS_MARGIN, SENS_T = 0.1, 0.3, (0.9, 1.3)
MIN_VALID, MIN_TRIALS = 10, 5
BIN_S, SFREQ, N_CH = 0.1, 250.0, 129
EARLY_BINS, CHAL_FIRST_S = 10, 0.5            # early store: 0 to 1.0 s; challenge store starts at 0.5 s
DECODERS = ("A", "B", "C", "D")
SAMPLES = {
    "HBN": {"root": data_location("HBN"), "people": None,
            "refs": {"E_tc": ("results/benchmark-v1/challenge_timecourse", "timecourse"),
                     "E_eegnet": ("results/benchmark-v1/challenge", "eegnet")}},
    "MIPDB": {"root": data_location("MIPDB"), "people": f"{data_location('MIPDB')}/manifest/participants.csv",
              "refs": {"E_tc": ("results/external_validation/mipdb", "timecourse"),
                       "E_eegnet": ("results/external_validation/mipdb_eegnet", "eegnet")}},
    "R10R11": {"root": data_location("R10R11"), "people": f"{data_location('R10R11')}/manifest/participants.csv",
               "refs": {"E_tc": ("results/external_validation/r10_r11", "timecourse"),
                        "E_eegnet": ("results/external_validation/r10_r11_eegnet", "eegnet")}},
}


def n_bins(t: float) -> int:
    return int(round(t / BIN_S))


def timecourse_columns(t: float, channels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns of the early (10 bins per channel) and challenge (20 bins per channel) time-course features
    that make up the [0, t] window, channel-major; the challenge part covers 1.0 s to t."""
    nb = n_bins(t)
    early = np.array([ch * EARLY_BINS + b for ch in channels for b in range(min(nb, EARLY_BINS))], dtype=int)
    first = n_bins(1.0 - CHAL_FIRST_S)                                   # challenge bin covering 1.0 to 1.1 s
    chal = np.array([ch * 20 + j for ch in channels for j in range(first, first + max(nb - EARLY_BINS, 0))], dtype=int)
    return early, chal


def bin_extremes(early, chal, rows_e: np.ndarray, rows_c: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial, per-channel maxima and minima of 100 ms bins from 0 to 2.5 s (volts, float32): bins 0 to 9 from
    the early store, bins 10 to 24 from the challenge store (its samples from 1.0 s). Rows are read in chunks."""
    n = len(rows_e)
    per_bin = int(round(BIN_S * SFREQ))
    start = int(round((1.0 - CHAL_FIRST_S) * SFREQ))
    bmax = np.empty((n, N_CH, 25), dtype=np.float32)
    bmin = np.empty((n, N_CH, 25), dtype=np.float32)
    for s in range(0, n, chunk):
        e = np.asarray(early[rows_e[s:s + chunk]], dtype=np.float32).reshape(-1, N_CH, EARLY_BINS, per_bin)
        c = np.asarray(chal[rows_c[s:s + chunk]], dtype=np.float32)[:, :, start:].reshape(-1, N_CH, 15, per_bin)
        bmax[s:s + chunk, :, :10], bmin[s:s + chunk, :, :10] = e.max(axis=3), e.min(axis=3)
        bmax[s:s + chunk, :, 10:], bmin[s:s + chunk, :, 10:] = c.max(axis=3), c.min(axis=3)
    return bmax, bmin


def segment_artifact(bmax: np.ndarray, bmin: np.ndarray, t: float, qc: WindowQCConfig) -> np.ndarray:
    """The frozen window QC applied to the [0, t] segment: high amplitude in more than the allowed fraction of
    channels, or any flat channel."""
    nb = n_bins(t)
    p2p = (bmax[:, :, :nb].max(axis=2) - bmin[:, :, :nb].min(axis=2)).astype(np.float64) * 1e6
    high = (p2p > qc.p2p_threshold_uv).mean(axis=1) > qc.max_bad_channel_fraction
    flat = (p2p < qc.flat_threshold_uv).any(axis=1)
    return high | flat


def load_sample(name: str, spec: dict, paths: ProjectPaths, bcfg: BenchmarkConfig, preproc, qc, specs: dict,
                split: SplitAssignment, demo: pd.DataFrame) -> dict:
    root = Path(spec["root"])
    early = load_window_data(root / "windows-v1-early", specs["early"], preproc.resample_hz)
    chal = load_window_data(root / "windows-v1", specs["challenge"], preproc.resample_hz)
    pre = load_window_data(root / "windows-v1", specs["prestim"], preproc.resample_hz)
    cache = {k: FeatureCache(root / "features" / store, w, bcfg.aperiodic_regions, np.zeros((N_CH, 3)), None,
                             n_jobs=bcfg.feature_n_jobs)
             for k, store, w in (("early", "windows-v1-early", early), ("challenge", "windows-v1", chal),
                                 ("prestim", "windows-v1", pre))}
    cache["early"].timecourse(bcfg.timecourse_bin_ms)
    cache["challenge"].timecourse(bcfg.timecourse_bin_ms)
    cache["prestim"].spectral()
    feature_paths = {"tc_early": str(cache["early"].root / f"timecourse_{bcfg.timecourse_bin_ms:g}ms.npy"),
                     "tc_chal": str(cache["challenge"].root / f"timecourse_{bcfg.timecourse_bin_ms:g}ms.npy"),
                     "spectral": str(cache["prestim"].root / "spectral.npy")}
    keys = ["block_key", "trial_in_block"]
    ei = early.index.rename(columns={"row": "row_e", "window_valid": "valid_e", "artifact_window": "art_e"})
    ci = chal.index[[*keys, "row", "window_valid", "artifact_window"]].rename(
        columns={"row": "row_c", "window_valid": "valid_c", "artifact_window": "art_c"})
    pi = pre.index[[*keys, "row", "window_valid", "artifact_window"]].rename(
        columns={"row": "row_p", "window_valid": "valid_p", "artifact_window": "art_p"})
    table = (ei.merge(ci, on=keys, how="inner", validate="one_to_one")
               .merge(pi, on=keys, how="inner", validate="one_to_one")
               .sort_values("row_e").reset_index(drop=True))
    bmax, bmin = bin_extremes(early.array, chal.array, table["row_e"].to_numpy(), table["row_c"].to_numpy())
    art = {t: segment_artifact(bmax, bmin, t, qc) for t in sorted(set(T_VALUES) | {1.0})}
    del bmax, bmin
    qc_check = {"n": int(len(table)),
                "t1_flags_equal_early_store": int((art[1.0] == table["art_e"].astype(bool).to_numpy()).sum())}
    table["pos"] = np.arange(len(table))
    if spec["people"] is None:
        table["fold"] = split.fold_of(table["subject"]).astype("Int64")
        table = table[table["fold"].notna()]
        people = demo
    else:
        table["fold"] = pd.array([pd.NA] * len(table), dtype="Int64")
        people = pd.read_csv(spec["people"], dtype={"subject": str})
    table = table.merge(people[["subject", "age", "sex", "ehq_total"]], on="subject", how="inner")
    valid_counts = table[table["valid_response"].astype(bool)].groupby("subject").size()
    table["enough_valid"] = table["subject"].map(valid_counts).fillna(0) >= MIN_VALID
    for ref, (run, model) in spec["refs"].items():
        pr = pd.read_parquet(paths.root / run / "predictions.parquet")
        pr = pr[(pr["window"] == "challenge") & (pr["target"] == "rt_s") & (pr["model"] == model)]
        table = table.merge(pr[["subject", "block_key", "trial_index", "y_pred"]].rename(columns={"y_pred": ref}),
                            on=["subject", "block_key", "trial_index"], how="left", validate="one_to_one")
    table = table.reset_index(drop=True)
    art = {t: a[table["pos"].to_numpy()] for t, a in art.items()}
    return {"name": name, "table": table, "art": art, "features": feature_paths, "qc_check": qc_check}


def trial_set(sample: dict, t: float, margin: float) -> pd.DataFrame:
    tab = sample["table"]
    ok = (tab["valid_response"].astype(bool) & (tab["rt_s"] > t + margin) & tab["enough_valid"]
          & ~sample["art"][t] & tab["valid_e"].astype(bool) & tab["valid_c"].astype(bool) & tab["valid_p"].astype(bool)
          & ~tab["art_p"].astype(bool))
    part = tab[ok]
    counts = part.groupby("subject").size()
    return part[part["subject"].isin(counts[counts >= MIN_TRIALS].index)].reset_index(drop=True)


def covariates(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack([pd.to_numeric(rows["age"], errors="coerce").to_numpy(float),
                            rows["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).to_numpy(float),
                            pd.to_numeric(rows["ehq_total"], errors="coerce").to_numpy(float)])


def features_for(rows: pd.DataFrame, feature_paths: dict, decoder: str, t: float, posterior: np.ndarray) -> np.ndarray:
    parts = []
    if decoder in ("A", "B", "D"):
        channels = np.flatnonzero(posterior) if decoder == "B" else np.arange(N_CH)
        cols_e, cols_c = timecourse_columns(t, channels)
        parts.append(np.asarray(np.load(feature_paths["tc_early"], mmap_mode="r")[rows["row_e"].to_numpy()][:, cols_e],
                                dtype=np.float64))
        if len(cols_c):
            parts.append(np.asarray(np.load(feature_paths["tc_chal"], mmap_mode="r")[rows["row_c"].to_numpy()][:, cols_c],
                                    dtype=np.float64))
    if decoder in ("C", "D"):
        parts.append(np.asarray(np.load(feature_paths["spectral"], mmap_mode="r")[rows["row_p"].to_numpy()],
                                dtype=np.float64))
    return np.concatenate(parts, axis=1)


def make_model(decoder: str, bcfg: BenchmarkConfig) -> GroupedRidge:
    kwargs = {"alphas": tuple(bcfg.ridge_alphas), "n_inner_folds": bcfg.inner_folds, "seed": bcfg.seed}
    if decoder != "C":
        kwargs["pca_components"] = bcfg.timecourse_pca_components
    return GroupedRidge(name=f"horizon_{decoder}", **kwargs)


def run_task(t: float, margin: float, decoder: str, hbn: pd.DataFrame, hbn_features: dict, externals: dict,
             posterior: np.ndarray, bcfg: BenchmarkConfig) -> pd.DataFrame:
    x = features_for(hbn, hbn_features, decoder, t, posterior)
    y, subjects, cov = hbn["rt_s"].to_numpy(float), hbn["subject"].to_numpy(), covariates(hbn)
    fold = hbn["fold"].astype(int).to_numpy()
    pred = np.full(len(hbn), np.nan)
    for f in [*sorted(int(v) for v in np.unique(fold) if v != HOLDOUT_FOLD), HOLDOUT_FOLD]:
        test = fold == f
        train = (fold != f) & (fold != HOLDOUT_FOLD)
        if not test.any():
            continue
        LeakageAuditor.assert_rows_grouped(hbn, train, test)
        model = make_model(decoder, bcfg)
        model.fit(DecoderInput(x[train], subjects[train], cov[train]), y[train])
        pred[test] = model.predict(DecoderInput(x[test], subjects[test], cov[test]))
    out = [hbn[["subject", "block_key", "trial_index", "rt_s", "fold"]].assign(
        sample="HBN", scope=np.where(fold == HOLDOUT_FOLD, "holdout_R5", "cv"), pred=pred)]
    if externals:
        model = make_model(decoder, bcfg)
        model.fit(DecoderInput(x, subjects, cov), y)
        for name, (rows, feats) in externals.items():
            if len(rows) == 0:
                continue
            xe = features_for(rows, feats, decoder, t, posterior)
            out.append(rows[["subject", "block_key", "trial_index", "rt_s", "fold"]].assign(
                sample=name, scope=name, pred=model.predict(DecoderInput(xe, rows["subject"].to_numpy(), covariates(rows)))))
    return pd.concat(out, ignore_index=True).assign(T=t, margin=margin, decoder=decoder)


def child_sums(y: np.ndarray, x: np.ndarray, subjects: np.ndarray):
    f = pd.DataFrame({"s": subjects, "y": y, "x": x})
    g = f.groupby("s", sort=True)
    yc, xc = f["y"] - g["y"].transform("mean"), f["x"] - g["x"].transform("mean")
    sums = pd.DataFrame({"s": f["s"], "xy": xc * yc, "xx": xc ** 2, "yy": yc ** 2}).groupby("s", sort=True).sum()
    return sums["xy"].to_numpy(), sums["xx"].to_numpy(), sums["yy"].to_numpy()


def paired_bootstrap(y, xa, xb, subjects, statistic: str, n_boot: int, seed: int) -> dict:
    """Paired child bootstrap of r_a - r_b ("difference") or r_a^2 / r_b^2 ("r2_ratio") on the same trials."""
    a, b = child_sums(y, xa, subjects), child_sums(y, xb, subjects)
    n = len(a[0])

    def stat(w):
        ra = (w @ a[0]) / np.sqrt((w @ a[1]) * (w @ a[2]))
        rb = (w @ b[0]) / np.sqrt((w @ b[1]) * (w @ b[2]))
        return ra - rb if statistic == "difference" else (ra ** 2) / (rb ** 2)

    rng = np.random.default_rng(seed)
    draws = [stat(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)) for _ in range(n_boot)]
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": float(stat(np.ones(n))), "ci_low": float(low), "ci_high": float(high), "n_children": int(n)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--limit-children", type=int, default=None, help="smoke tests: this many HBN children")
    parser.add_argument("--t-values", type=float, nargs="+", default=list(T_VALUES))
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    commit = git_commit(paths.root)
    bcfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    preproc = load_config(paths.root / "preprocessing" / "preprocessing.yaml", PreprocessingConfig)
    qc = load_config(paths.root / "features" / "window_quality.yaml", WindowQCConfig)
    specs = {"early": load_config(paths.root / "features" / "windows_early.yaml", TrialConfig).windows[0]}
    for w in load_config(paths.root / "features" / "trials_and_windows.yaml", TrialConfig).windows:
        specs[w.name] = w
    split = SplitAssignment.load(Path(f"{data_location('HBN')}/splits") / bcfg.split_file)
    demo = demographics_table(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    import mne

    montage = mne.channels.make_standard_montage(preproc.montage)
    positions = montage.get_positions()["ch_pos"]
    posterior = np.array([positions[c][1] < 0 for c in montage.ch_names])
    out_dir = Path(args.out_dir) if args.out_dir else paths.results / "prediction_horizon" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit_at_launch": commit, "argv": sys.argv}, out_dir / "provenance.json")
    samples = {name: load_sample(name, SAMPLES[name], paths, bcfg, preproc, qc, specs, split, demo)
               for name in args.samples}
    write_json({k: v["qc_check"] for k, v in samples.items()}, out_dir / "qc_check.json")
    print(json.dumps({k: v["qc_check"] for k, v in samples.items()}), flush=True)
    if args.limit_children:                        # smoke tests: the table and its aligned QC arrays together
        tab = samples["HBN"]["table"]
        keep = tab["subject"].isin(sorted(tab["subject"].unique())[:args.limit_children]).to_numpy()
        samples["HBN"]["table"] = tab[keep].reset_index(drop=True)
        samples["HBN"]["art"] = {t: a[keep] for t, a in samples["HBN"]["art"].items()}
    tasks = [(t, MARGIN, d) for t in args.t_values for d in DECODERS]
    tasks += [(t, SENS_MARGIN, d) for t in SENS_T if t in args.t_values for d in ("A", "C")]
    jobs = []
    for t, margin, d in tasks:
        hbn = trial_set(samples["HBN"], t, margin)
        externals = {n: (trial_set(s, t, margin), s["features"]) for n, s in samples.items() if n != "HBN"}
        jobs.append(delayed(run_task)(t, margin, d, hbn, samples["HBN"]["features"], externals, posterior, bcfg))
    predictions = pd.concat(Parallel(n_jobs=args.n_jobs, verbose=5)(jobs), ignore_index=True)
    # references on the same trials
    refs = []
    for (sample, t, margin), part in predictions[predictions["decoder"] == "A"].groupby(["sample", "T", "margin"]):
        tab = samples[sample]["table"][["subject", "block_key", "trial_index", "E_tc", "E_eegnet"]]
        merged = part.drop(columns=["pred", "decoder"]).merge(tab, on=["subject", "block_key", "trial_index"], how="left")
        for ref in ("E_tc", "E_eegnet"):
            refs.append(merged.assign(decoder=ref, pred=merged[ref]).drop(columns=["E_tc", "E_eegnet"]))
    predictions = pd.concat([predictions, *refs], ignore_index=True)
    write_dataframe(predictions, out_dir / "predictions.parquet")
    curve, paired = [], []
    for (scope, t, margin), part in predictions.groupby(["scope", "T", "margin"], sort=True):
        wide = part.pivot_table(index=["subject", "block_key", "trial_index", "rt_s"], columns="decoder",
                                values="pred").reset_index()
        for d in [c for c in ("A", "B", "C", "D", "E_tc", "E_eegnet") if c in wide]:
            sub = wide.dropna(subset=[d])
            if sub["subject"].nunique() < 3:
                continue
            res = within_between(sub["rt_s"].to_numpy(float), sub[d].to_numpy(float), sub["subject"].to_numpy(),
                                 n_bootstrap=args.n_bootstrap, seed=bcfg.seed)
            curve.append({"scope": scope, "T": t, "margin": margin, "decoder": d, "n_trials": len(sub), **res})
        for a, b, stat in (("A", "C", "difference"), ("D", "A", "difference"), ("B", "A", "difference"),
                           ("A", "E_tc", "r2_ratio"), ("A", "E_eegnet", "r2_ratio")):
            if a in wide and b in wide:
                sub = wide.dropna(subset=[a, b])
                if sub["subject"].nunique() >= 3:
                    paired.append({"scope": scope, "T": t, "margin": margin, "comparison": f"{a}-{b}" if stat == "difference"
                                   else f"{a}^2/{b}^2", "n_trials": len(sub),
                                   **paired_bootstrap(sub["rt_s"].to_numpy(float), sub[a].to_numpy(float), sub[b].to_numpy(float),
                                                      sub["subject"].to_numpy(), stat, args.n_bootstrap, bcfg.seed)})
    curve, paired = pd.DataFrame(curve), pd.DataFrame(paired)
    write_dataframe(curve, out_dir / "curve.csv")
    write_dataframe(paired, out_dir / "paired.csv")
    # decision rules
    ac = paired[(paired["comparison"] == "A-C") & (paired["margin"] == MARGIN)]
    rule1_t = []
    for t in sorted(ac["T"].unique()):
        if t > 1.3:
            continue
        row = {s: ac[(ac["T"] == t) & (ac["scope"] == s)] for s in ("cv", "holdout_R5", "MIPDB", "R10R11")}
        ok = all(not row[s].empty and row[s]["ci_low"].iloc[0] > 0 for s in ("cv", "holdout_R5"))
        ok &= all(row[s].empty or row[s]["estimate"].iloc[0] > 0 for s in ("MIPDB", "R10R11"))
        if ok:
            rule1_t.append(t)
    a13 = curve[(curve["scope"] == "cv") & (curve["T"] == 1.3) & (curve["margin"] == MARGIN)].set_index("decoder")
    rule3 = (float(a13.loc["B", "r_within"] / a13.loc["A", "r_within"]) if {"A", "B"} <= set(a13.index) else None)
    summary = {"git_commit": commit, "out_dir": str(out_dir), "qc_check": {k: v["qc_check"] for k, v in samples.items()},
               "rule1_title_kept": bool(rule1_t), "rule1_T_meeting": rule1_t,
               "rule3_posterior_share_T1.3_cv": rule3, "rule3_not_mainly_ocular": (rule3 is not None and rule3 >= 0.5)}
    write_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
