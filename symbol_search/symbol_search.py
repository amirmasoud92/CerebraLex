"""Does the within-child decoding generalise to Symbol Search, a self-paced decision?

Steps (each writes under ``--root``; R1 to R9 and NC):

``qc``       quality control of the preprocessed recordings: processed, and not a common-mode recording (raw
             channel-SD coefficient of variation below the threshold of preprocessing/quality_control.yaml) ->
             qc-preproc-v1/block_qc.parquet, common_mode.parquet.
``items``    one row per response: the item starts at the previous response or at the page onset, whichever came
             last; item time = response - start; valid items last 1 to 30 s; early-window items last more than
             3.0 s; accuracy from user_answer == correct_answer -> items-v1/items.parquet.
``windows``  the benchmark window stores (window QC of features/window_quality.yaml) locked to the item start:
             early (+0.5 to +2.5 s) and preceding (-2.0 to 0 s) -> windows-v1/<window>/.
``decode``   the benchmark's demographics, band-power and time-course ridge decoders with their frozen settings
             (decoding/benchmark.yaml) on the frozen folds (10 cross-validation folds; every R5 child
             held out; children outside the frozen split are excluded); target log item time; children with at
             least 8 valid items per window. Reports the within- and between-child decomposition and the trial
             specificity variants (child, trend, diff, carry) -> results/symbol_search/decode/<run>/.
``maps``     eye-movement check: within-child r maps of the early window (time course and band power,
             cross-validation children), periocular summaries, and the early time-course decoder refitted without
             the periocular channels and on posterior channels only -> results/symbol_search/maps/<run>/.
``external`` every decoder refitted once on all children of the decode cohort (every fold, R5 included) and applied
             to another Symbol Search root (``--external-root``, participants from ``--people``) with the same item,
             window and cohort rules -> results/symbol_search/external/<run>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import config_fingerprint, load_config
from common.io import utc_timestamp, write_dataframe, write_json
from common.log import configure_logging
from common.paths import ProjectPaths, data_location
from common.provenance import git_commit
from common.schema import BenchmarkConfig, QCConfig, WindowQCConfig, WindowSpec
from common.scripts import script_path
from decoding.base import DecoderInput
from decoding.baselines import GroupedRidge
from decoding.benchmark import FeatureCache, WindowData, demographics_table, store_fingerprint
from decoding.decomposition import within_between
from decoding.leakage import LeakageAuditor
from decoding.splits import HOLDOUT_FOLD, SplitAssignment
from features.windows import WindowStore

SFREQ = 250.0
MIN_ITEM_S, MAX_ITEM_S, EARLY_MIN_ITEM_S = 1.0, 30.0, 3.0
MIN_VALID_ITEMS = 8
WINDOWS = {"early": WindowSpec(name="early", tmin_s=0.5, tmax_s=2.5, description="item start +0.5 to +2.5 s"),
           "preceding": WindowSpec(name="preceding", tmin_s=-2.0, tmax_s=0.0,
                                   description="the 2 s before the item start (end of the previous item)")}
MODELS = ("demographics", "spectral", "timecourse")
VARIANTS = ("child", "trend", "diff", "carry")


class SysWindowStore(WindowStore):
    """Benchmark window store over Symbol Search recordings (task-SyS file names, one block per child)."""

    @staticmethod
    def _block_paths(trials: pd.DataFrame, preproc_dir) -> dict[str, Path]:
        paths = {}
        for block_key, group in trials.groupby("block_key"):
            first = group.iloc[0]
            path = Path(preproc_dir) / f"sub-{first['subject']}" / f"sub-{first['subject']}_{first['release']}_task-SyS_preproc.h5"
            if path.is_file():
                paths[block_key] = path
        return paths


def step_qc(root: Path, paths: ProjectPaths, n_jobs: int) -> dict:
    from preprocessing.quality_control import load_preprocessing_log
    from preprocessing.signal_audit import audit_common_mode

    qc_cfg = load_config(paths.root / "preprocessing" / "quality_control.yaml", QCConfig)
    manifest = pd.read_parquet(root / "manifest" / "sys_recordings.parquet")
    manifest = manifest[manifest["usable"]]
    log = load_preprocessing_log(root / "preproc-v1")[["block_key", "status"]]
    table = manifest.merge(log, on="block_key", how="left")
    processed = table["status"].isin(["ok", "skipped"])
    diagnostics = audit_common_mode(table[processed], n_jobs=n_jobs, seconds=60.0)
    out_dir = root / "qc-preproc-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(diagnostics, out_dir / "common_mode.parquet")
    table = table.merge(diagnostics[["block_key", "raw_sd_cv", "interchannel_r_median"]], on="block_key", how="left")
    table["common_mode"] = (table["raw_sd_cv"] < qc_cfg.common_mode_sd_cv_threshold).fillna(False)
    table["analysis_usable"] = processed & ~table["common_mode"]
    write_dataframe(table, out_dir / "block_qc.parquet")
    return {"recordings": len(table), "processed": int(processed.sum()), "common_mode": int(table["common_mode"].sum()),
            "analysis_usable": int(table["analysis_usable"].sum()),
            "status": table["status"].fillna("missing").value_counts().to_dict()}


def parse_items(events: pd.DataFrame) -> pd.DataFrame:
    """Items of one recording: each response ends an item that started at the latest page onset or response."""
    e = events.sort_values("onset", kind="stable")
    start = e.loc[e["value"] == "symbolSearch_start", "onset"]
    e = e[e["onset"] >= float(start.iloc[0])]
    rows, last, kind, page = [], None, None, 0
    for r in e.itertuples(index=False):
        if r.value == "newPage":
            last, kind, page = float(r.onset), "page", page + 1
        elif r.value == "trialResponse":
            if last is not None:
                rows.append({"item_start_onset_s": last, "start_kind": kind, "page": page,
                             "response_onset_s": float(r.onset), "user_answer": str(r.user_answer),
                             "correct_answer": str(r.correct_answer)})
            last, kind = float(r.onset), "response"
    return pd.DataFrame(rows)


def step_items(root: Path) -> dict:
    qc = pd.read_parquet(root / "qc-preproc-v1" / "block_qc.parquet")
    frames = []
    for rec in qc[qc["analysis_usable"]].itertuples(index=False):
        events = pd.read_csv(Path(rec.eeg_path.replace("_eeg.set", "_events.tsv")), sep="\t")
        items = parse_items(events)
        if items.empty:
            continue
        items.insert(0, "block_key", rec.block_key)
        items.insert(1, "subject", rec.subject)
        items.insert(2, "release", rec.release)
        items["block"] = 1
        items["trial_in_block"] = np.arange(len(items))
        items["item_time_s"] = items["response_onset_s"] - items["item_start_onset_s"]
        items["correct"] = items["user_answer"] == items["correct_answer"]
        items["valid_response"] = items["item_time_s"].between(MIN_ITEM_S, MAX_ITEM_S)
        items["early_eligible"] = items["item_time_s"] > EARLY_MIN_ITEM_S
        items["rt_s"] = items["item_time_s"]
        items["log_item_time"] = np.log(items["item_time_s"].where(items["item_time_s"] > 0))
        prev = items["item_time_s"].where(items["valid_response"]).shift(1)
        items["prev_rt_s"] = np.log(prev)                     # previous valid item's log time (carry-over)
        items["target_sample"] = np.round((items["item_start_onset_s"] - rec.crop_tmin_s) * SFREQ).astype(int)
        items["block_n_times"] = int(round((rec.crop_tmax_s - rec.crop_tmin_s) * SFREQ))
        items["spans_boundary"] = False                      # no boundary inside the task (manifest rule)
        frames.append(items)
    table = pd.concat(frames, ignore_index=True)
    table["trial_index"] = table.groupby("subject").cumcount()
    out_dir = root / "items-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(table, out_dir / "items.parquet")
    return {"recordings": int(table["block_key"].nunique()), "items": len(table),
            "valid_items": int(table["valid_response"].sum()), "early_eligible_valid": int((table["valid_response"] & table["early_eligible"]).sum())}


def step_windows(root: Path, paths: ProjectPaths) -> dict:
    qc = load_config(paths.root / "features" / "window_quality.yaml", WindowQCConfig)
    items = pd.read_parquet(root / "items-v1" / "items.parquet")
    out = {}
    for name, spec in WINDOWS.items():
        store = SysWindowStore(root / "windows-v1", spec, SFREQ, qc=qc)
        if store.exists():
            raise FileExistsError(f"{store.root} exists; windows are written once")
        index = store.build(items, root / "preproc-v1")
        out[name] = {"rows": len(index), "valid": int(index["window_valid"].sum()),
                     "artifact": int(index["artifact_window"].sum())}
    return out


def specificity_module(paths: ProjectPaths):
    spec = importlib.util.spec_from_file_location("trial_specificity", script_path("trial_specificity"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohort(index: pd.DataFrame, window: str, split: SplitAssignment) -> pd.DataFrame:
    index = index.copy()
    fold = split.fold_of(index["subject"])
    fold = fold.where(fold.notna() | (index["release"] != "R5"), HOLDOUT_FOLD)
    index["fold"] = fold.astype("Int64")
    keep = (index["window_valid"] & ~index["artifact_window"].astype(bool) & index["fold"].notna()
            & index["valid_response"])
    if window == "early":
        keep &= index["early_eligible"]
    per_child = index[keep].groupby("subject").size()
    keep &= index["subject"].isin(per_child[per_child >= MIN_VALID_ITEMS].index)
    return index[keep].reset_index(drop=True)


def external_cohort(index: pd.DataFrame, window: str) -> pd.DataFrame:
    """The decode cohort rules without the split (every child of another sample is a test child)."""
    keep = index["window_valid"] & ~index["artifact_window"].astype(bool) & index["valid_response"]
    if window == "early":
        keep &= index["early_eligible"]
    per_child = index[keep].groupby("subject").size()
    keep &= index["subject"].isin(per_child[per_child >= MIN_VALID_ITEMS].index)
    return index[keep].reset_index(drop=True)


def covariates_of(rows: pd.DataFrame) -> np.ndarray:
    return np.column_stack([pd.to_numeric(rows["age"], errors="coerce").to_numpy(float),
                            rows["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).to_numpy(float),
                            pd.to_numeric(rows["ehq_total"], errors="coerce").to_numpy(float)])


PERIOCULAR = ("E8", "E14", "E21", "E25", "E125", "E126", "E127", "E128")


def step_maps(root: Path, paths: ProjectPaths, n_bootstrap: int) -> dict:
    import mne

    cfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    split = SplitAssignment.load(Path(f"{data_location('HBN')}/splits") / cfg.split_file)
    demo = demographics_table(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    spec_maps = importlib.util.spec_from_file_location("rt_information_maps", script_path("rt_information_maps"))
    maps_module = importlib.util.module_from_spec(spec_maps)
    spec_maps.loader.exec_module(maps_module)
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ch_names = list(montage.ch_names)
    positions = montage.get_positions()["ch_pos"]
    spec = WINDOWS["early"]
    array, index, meta = SysWindowStore(root / "windows-v1", spec, SFREQ).open()
    window = WindowData(spec, array, index, float(meta["sfreq"]), store_fingerprint(index, meta), meta)
    cache = FeatureCache(root / "features" / "windows-v1", window, cfg.aperiodic_regions, np.zeros((129, 3)), None,
                         n_jobs=cfg.feature_n_jobs)
    rows = cohort(index, "early", split).merge(demo, on="subject", how="left")
    cv = rows[rows["fold"] != HOLDOUT_FOLD]
    y_cv = cv["log_item_time"].to_numpy(float)
    tc, sp = cache.timecourse(cfg.timecourse_bin_ms), cache.spectral()
    n_bins = tc.shape[1] // len(ch_names)
    peri = np.array([c in PERIOCULAR for c in ch_names])
    out_dir = paths.results / "symbol_search" / "maps" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"git_commit": git_commit(paths.root), "periocular": list(PERIOCULAR), "n_children_cv": int(cv["subject"].nunique())}
    frames = []
    for feature, x_all, layout in (("timecourse", tc, "channel_major"), ("spectral", sp, "band_major")):
        r = maps_module.within_child_r(x_all[cv["row"].to_numpy()], y_cv, cv["subject"].to_numpy())
        if layout == "channel_major":
            channel = np.repeat(np.arange(len(ch_names)), n_bins)
            label = np.tile(np.arange(n_bins), len(ch_names)).astype(str)
        else:
            n_bands = x_all.shape[1] // len(ch_names)
            channel = np.tile(np.arange(len(ch_names)), n_bands)
            label = np.repeat(maps_module.cell_labels("spectral", x_all.shape[1], len(ch_names))["band"].unique(), len(ch_names))
        cells = pd.DataFrame({"feature": feature, "channel": [ch_names[c] for c in channel], "cell": label, "r": r,
                              "periocular": peri[channel]})
        frames.append(cells)
        top = cells.assign(abs_r=cells["r"].abs()).nlargest(100, "abs_r")
        summary[feature] = {"top10": top.head(10)[["channel", "cell", "r"]].to_dict(orient="records"),
                            "mean_abs_r_periocular": float(cells.loc[cells["periocular"], "r"].abs().mean()),
                            "mean_abs_r_other": float(cells.loc[~cells["periocular"], "r"].abs().mean()),
                            "periocular_share_top100": float(top["periocular"].mean()),
                            "chance_share": float(peri.mean())}
    write_dataframe(pd.concat(frames, ignore_index=True), out_dir / "maps.csv")
    posterior = np.array([positions[c][1] < 0 for c in ch_names])
    subsets = {"all": np.ones(len(ch_names), bool), "no_periocular": ~peri, "posterior_only": posterior}
    summary["posterior_channels"] = int(posterior.sum())
    covariates = np.column_stack([pd.to_numeric(rows["age"], errors="coerce").to_numpy(float),
                                  rows["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).to_numpy(float),
                                  pd.to_numeric(rows["ehq_total"], errors="coerce").to_numpy(float)])
    y = rows["log_item_time"].to_numpy(float)
    folds = [(f, rows["fold"].ne(f) & rows["fold"].ne(HOLDOUT_FOLD), rows["fold"].eq(f))
             for f in sorted(rows.loc[rows["fold"] != HOLDOUT_FOLD, "fold"].unique())]
    folds.append((HOLDOUT_FOLD, rows["fold"].ne(HOLDOUT_FOLD), rows["fold"].eq(HOLDOUT_FOLD)))
    controls = []
    for name, keep in subsets.items():
        cols = np.flatnonzero(np.repeat(keep, n_bins))
        x = np.asarray(tc[rows["row"].to_numpy()][:, cols], dtype=np.float64)
        pred = np.full(len(rows), np.nan)
        for _, train, test in folds:
            train, test = train.to_numpy(bool), test.to_numpy(bool)
            model = GroupedRidge(name=f"timecourse_{name}", pca_components=cfg.timecourse_pca_components,
                                 alphas=tuple(cfg.ridge_alphas), n_inner_folds=cfg.inner_folds, seed=cfg.seed)
            model.fit(DecoderInput(x[train], rows["subject"].to_numpy()[train], covariates[train]), y[train])
            pred[test] = model.predict(DecoderInput(x[test], rows["subject"].to_numpy()[test], covariates[test]))
        for scope, mask in (("cv", rows["fold"].ne(HOLDOUT_FOLD).to_numpy(bool)), ("holdout_R5", rows["fold"].eq(HOLDOUT_FOLD).to_numpy(bool))):
            res = within_between(y[mask], pred[mask], rows["subject"].to_numpy()[mask], n_bootstrap=n_bootstrap, seed=cfg.seed)
            controls.append({"channels": name, "n_channels": int(keep.sum()), "scope": scope, **res})
        print(f"control {name}: done", flush=True)
    controls = pd.DataFrame(controls)
    write_dataframe(controls, out_dir / "control_decoders.csv")
    summary["controls"] = controls[["channels", "scope", "r_within", "r_within_ci_low", "r_within_ci_high"]].to_dict(orient="records")
    summary["out_dir"] = str(out_dir)
    write_json(summary, out_dir / "summary.json")
    return summary


def posterior_columns(n_columns: int) -> np.ndarray:
    """Time-course columns (channel-major) of the posterior channels (template y below 0)."""
    import mne

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    positions = montage.get_positions()["ch_pos"]
    keep = np.array([positions[c][1] < 0 for c in montage.ch_names])
    return np.flatnonzero(np.repeat(keep, n_columns // len(keep)))


EXTERNAL_MODELS = (*MODELS, "timecourse_posterior")


def step_external(root: Path, ext_root: Path, people_csv: Path, paths: ProjectPaths, n_bootstrap: int,
                  out_dir: Path | None = None) -> dict:
    cfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    split = SplitAssignment.load(Path(f"{data_location('HBN')}/splits") / cfg.split_file)
    demo = demographics_table(pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False))
    people = pd.read_csv(people_csv, dtype={"subject": str})[["subject", "age", "sex", "ehq_total"]]
    specificity = specificity_module(paths)
    commit = git_commit(paths.root)
    out_dir = Path(out_dir) if out_dir else paths.results / "symbol_search" / "external" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit": commit, "argv": sys.argv, "external_root": str(ext_root)}, out_dir / "provenance.json")
    predictions, decomposition, trial_level, cohorts = [], [], [], {}
    for name, spec in WINDOWS.items():
        sources = {}
        for label, base in (("train", root), ("test", ext_root)):
            array, index, meta = SysWindowStore(base / "windows-v1", spec, SFREQ).open()
            window = WindowData(spec, array, index, float(meta["sfreq"]), store_fingerprint(index, meta), meta)
            cache = FeatureCache(base / "features" / "windows-v1", window, cfg.aperiodic_regions, np.zeros((129, 3)),
                                 None, n_jobs=cfg.feature_n_jobs)
            rows = (cohort(index, name, split).merge(demo, on="subject", how="left") if label == "train"
                    else external_cohort(index, name).merge(people, on="subject", how="inner"))
            sources[label] = (cache, rows)
        (train_cache, train), (test_cache, test) = sources["train"], sources["test"]
        cohorts[name] = {"train_children": int(train["subject"].nunique()),
                         "test_children": int(test["subject"].nunique()), "test_items": len(test)}
        for model in EXTERNAL_MODELS:
            kwargs = {"alphas": tuple(cfg.ridge_alphas), "n_inner_folds": cfg.inner_folds, "seed": cfg.seed}
            if model.startswith("timecourse"):
                kwargs["pca_components"] = cfg.timecourse_pca_components

            def inputs(cache, rows, model=model):
                if model == "demographics":
                    x = np.zeros((len(rows), 0))
                elif model == "spectral":
                    x = np.asarray(cache.spectral()[rows["row"].to_numpy()])
                else:
                    x = np.asarray(cache.timecourse(cfg.timecourse_bin_ms)[rows["row"].to_numpy()])
                    if model == "timecourse_posterior":
                        x = x[:, posterior_columns(x.shape[1])]
                return DecoderInput(x, rows["subject"].to_numpy(), covariates_of(rows))

            decoder = GroupedRidge(name=model, **kwargs)
            decoder.fit(inputs(train_cache, train), train["log_item_time"].to_numpy(float))
            pred = test[["subject", "release", "block_key", "trial_in_block", "trial_index", "prev_rt_s"]].copy()
            pred["y_true"] = test["log_item_time"].to_numpy(float)
            pred["y_pred"] = decoder.predict(inputs(test_cache, test))
            pred["model"], pred["window"] = model, name
            predictions.append(pred)
            res = within_between(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy(), pred["subject"].to_numpy(),
                                 n_bootstrap=n_bootstrap, seed=cfg.seed)
            decomposition.append({"window": name, "model": model, "scope": "external", **res})
            if model != "demographics":
                frame = pred.rename(columns={"y_pred": "pred", "y_true": "rt"})
                for variant in VARIANTS:
                    trial_level.append({"window": name, "model": model, "scope": "external", "variant": variant,
                                        **specificity.pooled_r(specificity.variant_frame(frame, variant), n_bootstrap, cfg.seed)})
            print(f"{name} {model}: done", flush=True)
    write_dataframe(pd.concat(predictions, ignore_index=True), out_dir / "predictions.parquet")
    write_dataframe(pd.DataFrame(decomposition), out_dir / "decomposition.csv")
    write_dataframe(pd.DataFrame(trial_level), out_dir / "trial_level.csv")
    summary = {"git_commit": commit, "cohorts": cohorts, "out_dir": str(out_dir)}
    write_json(summary, out_dir / "summary.json")
    return summary


def step_decode(root: Path, paths: ProjectPaths, n_bootstrap: int) -> dict:
    cfg = load_config(paths.root / "decoding" / "benchmark.yaml", BenchmarkConfig)
    split = SplitAssignment.load(Path(f"{data_location('HBN')}/splits") / cfg.split_file)
    participants = pd.read_csv(paths.root / "audit" / "participants_merged.csv", low_memory=False)
    demo = demographics_table(participants)
    specificity = specificity_module(paths)
    commit = git_commit(paths.root)
    out_dir = paths.results / "symbol_search" / "decode" / utc_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"git_commit": commit, "argv": sys.argv, "benchmark_config": config_fingerprint(cfg)},
               out_dir / "provenance.json")
    predictions, decomposition, trial_level, cohorts = [], [], [], {}
    for name, spec in WINDOWS.items():
        array, index, meta = SysWindowStore(root / "windows-v1", spec, SFREQ).open()
        window = WindowData(spec, array, index, float(meta["sfreq"]), store_fingerprint(index, meta), meta)
        cache = FeatureCache(root / "features" / "windows-v1", window, cfg.aperiodic_regions, np.zeros((129, 3)),
                             None, n_jobs=cfg.feature_n_jobs)
        rows = cohort(index, name, split).merge(demo, on="subject", how="left")
        cohorts[name] = {"children": int(rows["subject"].nunique()), "items": len(rows),
                         "holdout_children": int(rows.loc[rows["fold"] == HOLDOUT_FOLD, "subject"].nunique())}
        y = rows["log_item_time"].to_numpy(float)
        covariates = np.column_stack([pd.to_numeric(rows["age"], errors="coerce").to_numpy(float),
                                      rows["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0}).to_numpy(float),
                                      pd.to_numeric(rows["ehq_total"], errors="coerce").to_numpy(float)])
        features = {"spectral": lambda r: np.asarray(cache.spectral()[r]),
                    "timecourse": lambda r: np.asarray(cache.timecourse(cfg.timecourse_bin_ms)[r])}
        folds = [(f, rows["fold"].ne(f) & rows["fold"].ne(HOLDOUT_FOLD), rows["fold"].eq(f))
                 for f in sorted(rows.loc[rows["fold"] != HOLDOUT_FOLD, "fold"].unique())]
        folds.append((HOLDOUT_FOLD, rows["fold"].ne(HOLDOUT_FOLD), rows["fold"].eq(HOLDOUT_FOLD)))
        for model in MODELS:
            kwargs = {"alphas": tuple(cfg.ridge_alphas), "n_inner_folds": cfg.inner_folds, "seed": cfg.seed}
            if model == "timecourse":
                kwargs["pca_components"] = cfg.timecourse_pca_components
            outs = []
            for fold, train, test in folds:
                train, test = train.to_numpy(bool), test.to_numpy(bool)
                LeakageAuditor.assert_rows_grouped(rows, train, test)

                def inputs(mask):
                    x = np.zeros((int(mask.sum()), 0)) if model == "demographics" else features[model](rows["row"].to_numpy()[mask])
                    return DecoderInput(x, rows["subject"].to_numpy()[mask], covariates[mask])

                decoder = GroupedRidge(name=model, **kwargs)
                decoder.fit(inputs(train), y[train])
                out = rows.loc[test, ["subject", "release", "block_key", "trial_in_block", "trial_index", "prev_rt_s"]].copy()
                out["fold"], out["y_true"], out["y_pred"] = fold, y[test], decoder.predict(inputs(test))
                out["model"], out["window"] = model, name
                outs.append(out)
            pred = pd.concat(outs, ignore_index=True)
            predictions.append(pred)
            for scope, part in (("cv", pred[pred["fold"] != HOLDOUT_FOLD]), ("holdout_R5", pred[pred["fold"] == HOLDOUT_FOLD])):
                res = within_between(part["y_true"].to_numpy(), part["y_pred"].to_numpy(), part["subject"].to_numpy(),
                                     n_bootstrap=n_bootstrap, seed=cfg.seed)
                decomposition.append({"window": name, "model": model, "scope": scope, **res})
                if model == "demographics":
                    continue
                frame = part.rename(columns={"y_pred": "pred", "y_true": "rt"})
                for variant in VARIANTS:
                    trial_level.append({"window": name, "model": model, "scope": scope, "variant": variant,
                                        **specificity.pooled_r(specificity.variant_frame(frame, variant), n_bootstrap, cfg.seed)})
            print(f"{name} {model}: done", flush=True)
    write_dataframe(pd.concat(predictions, ignore_index=True), out_dir / "predictions.parquet")
    decomposition, trial_level = pd.DataFrame(decomposition), pd.DataFrame(trial_level)
    write_dataframe(decomposition, out_dir / "decomposition.csv")
    write_dataframe(trial_level, out_dir / "trial_level.csv")
    summary = {"git_commit": commit, "cohorts": cohorts, "out_dir": str(out_dir)}
    write_json(summary, out_dir / "summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", choices=["qc", "items", "windows", "decode", "maps", "external"], required=True)
    parser.add_argument("--root", default=data_location("SYS"))
    parser.add_argument("--external-root", default=None, help="external step: the other Symbol Search root")
    parser.add_argument("--people", default=None, help="external step: participants CSV (subject, age, sex, ehq_total)")
    parser.add_argument("--out-dir", default=None, help="external step: output directory (code tests)")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    log = configure_logging("symbol-search", paths.logs)
    root = Path(args.root)
    if args.step == "qc":
        out = step_qc(root, paths, args.n_jobs)
    elif args.step == "items":
        out = step_items(root)
    elif args.step == "windows":
        out = step_windows(root, paths)
    elif args.step == "decode":
        out = step_decode(root, paths, args.n_bootstrap)
    elif args.step == "maps":
        out = step_maps(root, paths, args.n_bootstrap)
    else:
        out = step_external(root, Path(args.external_root), Path(args.people), paths, args.n_bootstrap, args.out_dir)
    log.info("%s: %s", args.step, out)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
