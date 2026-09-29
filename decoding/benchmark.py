"""Benchmark runner: identical folds, windows and targets for every decoder."""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import config_fingerprint
from common.hashing import sha256_text
from common.io import write_dataframe, write_json
from common.schema import BenchmarkConfig, ComputeConfig, WindowSpec
from decoding.base import Decoder, DecoderInput
from decoding.baselines import GroupedRidge, MeanPredictor, TangentSpaceRidge
from decoding.leakage import LeakageAuditor
from decoding.metrics import (
    auroc,
    nrmse,
    paired_subject_bootstrap,
    per_fold_table,
    r2_from_nrmse,
    subject_bootstrap,
)
from decoding.splits import HOLDOUT_FOLD, SplitAssignment
from features.aperiodic import RegionalAverager, aperiodic_features_parallel
from features.riemannian import covariance_upper_triangle, window_covariances
from features.spectral import SpectralFeatures
from features.windows import WindowStore
from task_and_trials.participants import DEMOGRAPHIC_COLUMNS

_LOG = logging.getLogger(__name__)
_REGRESSION_TARGETS = ("rt_s", "log_rt_centred")
REVE_MODELS = {"reve_attnpool": "attention_pool_all_layers", "reve_meanfinal": "mean_final"}


@dataclass
class WindowData:
    spec: WindowSpec
    array: np.ndarray                 # memmap [n, channels, samples]
    index: pd.DataFrame               # row-aligned trial table with `row` and `window_valid`
    sfreq: float
    fingerprint: str                  # content hash of the row order, keys the feature cache
    meta: dict = field(default_factory=dict)   # store metadata, including window-QC rules


class RowView:
    """Row-indexed view over a memmap so decoders never receive a full copy."""

    def __init__(self, array: np.ndarray, rows: np.ndarray) -> None:
        self.array, self.rows = array, np.asarray(rows)
        self.shape = (len(self.rows), *array.shape[1:])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item):
        return self.array[self.rows[item]]


class FeatureCache:
    """Per-window feature matrices computed once for all rows and cached on disk.

    The cache directory is keyed by the window store's content fingerprint, so a store rebuilt
    with different rows (for example the pilot versus the full corpus) never reuses stale
    features; row counts are re-checked on every load.
    """

    def __init__(self, root: Path, window: WindowData, n_regions: int, positions: np.ndarray,
                 device: str | None, n_jobs: int = 8,
                 embedder_factory: Callable[[], object] | None = None,
                 reve_tag: str | None = None) -> None:
        self.root = root / f"{window.spec.name}_{window.fingerprint[:12]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.n_regions = n_regions
        self.positions = positions
        self.device = device
        self.n_jobs = n_jobs
        self.embedder_factory = embedder_factory
        self.reve_tag = reve_tag

    def _store(self, name: str, values: np.ndarray, seconds: float, extra: dict | None = None) -> None:
        np.save(self.root / f"{name}.npy", values)
        (self.root / f"{name}.json").write_text(json.dumps({
            "rows": int(len(values)), "shape": list(values.shape), "seconds": round(seconds, 1),
            "window_fingerprint": self.window.fingerprint, **(extra or {})}, default=str),
            encoding="utf-8")
        _LOG.info("%s: %s in %.0f s", name, values.shape, seconds)

    def _load(self, name: str) -> np.ndarray:
        path = self.root / f"{name}.npy"
        values = np.load(path, mmap_mode="r")
        if len(values) != len(self.window.array):
            raise ValueError(f"{path} has {len(values)} rows, window store has {len(self.window.array)}")
        return values

    def _load_or_compute(self, name: str, compute) -> np.ndarray:
        if not (self.root / f"{name}.npy").is_file():
            _LOG.info("computing %s features for window %s", name, self.window.spec.name)
            start = time.perf_counter()
            self._store(name, compute(), time.perf_counter() - start)
        return self._load(name)

    def reve(self, representation: str, chunk: int = 4096) -> np.ndarray:
        """Frozen REVE embeddings; both representations are computed in one pass and
        cached under the model revision and amplitude mode, so another model never reuses them."""
        if self.embedder_factory is None or self.reve_tag is None:
            raise ValueError("REVE features requested but no embedder is configured")
        name = f"reve-{self.reve_tag}_{representation}"
        if not (self.root / f"{name}.npy").is_file():
            _LOG.info("computing REVE embeddings (%s) for window %s", self.reve_tag, self.window.spec.name)
            embedder = self.embedder_factory()
            start = time.perf_counter()
            parts: dict[str, list[np.ndarray]] = {}
            for begin in range(0, len(self.window.array), chunk):
                out = embedder.embed(np.asarray(self.window.array[begin:begin + chunk]))
                for key, value in out.items():
                    parts.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
            seconds = time.perf_counter() - start
            for key, values in parts.items():
                self._store(f"reve-{self.reve_tag}_{key}", np.concatenate(values), seconds,
                            {"embedder": embedder.describe()})
        return self._load(name)

    def spectral(self) -> np.ndarray:
        extractor = SpectralFeatures(self.window.sfreq)

        def compute() -> np.ndarray:
            out = []
            for start in range(0, len(self.window.array), 2048):
                feats, _ = extractor.transform(np.asarray(self.window.array[start:start + 2048]))
                out.append(feats)
            return np.concatenate(out)

        return self._load_or_compute("spectral", compute)

    def aperiodic(self) -> np.ndarray:
        averager = RegionalAverager(self.positions, n_regions=self.n_regions)

        def compute() -> np.ndarray:
            return aperiodic_features_parallel(self.window.array, self.window.sfreq, averager.labels,
                                               self.n_regions, n_jobs=self.n_jobs)

        return self._load_or_compute("aperiodic", compute)

    def timecourse(self, bin_ms: float, chunk: int = 2048) -> np.ndarray:
        """Window time courses averaged in ``bin_ms`` bins per channel, in microvolts.

        Keeps when activity happens (for example the press-locked centro-parietal positivity),
        which band power and covariance discard; 100 ms bins resolve components below 5 Hz.
        """
        bin_samples = int(round(bin_ms / 1000.0 * self.window.sfreq))

        def compute() -> np.ndarray:
            n, c, t = self.window.array.shape
            n_bins = t // bin_samples
            out = np.empty((n, c * n_bins), dtype=np.float32)
            for start in range(0, n, chunk):
                block = np.asarray(self.window.array[start:start + chunk, :, :n_bins * bin_samples],
                                   dtype=np.float32) * 1e6
                out[start:start + len(block)] = (block.reshape(len(block), c, n_bins, bin_samples)
                                                 .mean(axis=3).reshape(len(block), -1))
            return out

        return self._load_or_compute(f"timecourse_{bin_ms:g}ms", compute)

    def covariances(self) -> np.ndarray:
        def compute() -> np.ndarray:
            out = []
            for start in range(0, len(self.window.array), 1024):
                cov = window_covariances(np.asarray(self.window.array[start:start + 1024]),
                                         device=self.device)
                out.append(covariance_upper_triangle(cov))
            return np.concatenate(out)

        return self._load_or_compute("covariance_upper", compute)


class BenchmarkRunner:
    def __init__(self, cfg: BenchmarkConfig, compute: ComputeConfig, trials: pd.DataFrame,
                 windows: dict[str, WindowData], split: SplitAssignment,
                 demographics: pd.DataFrame, positions: np.ndarray, out_dir: Path,
                 feature_root: Path, cohort_flags: dict[str, pd.DataFrame] | None = None,
                 excluded_blocks: set[str] | None = None,
                 embedder_factory: Callable[[], object] | None = None,
                 reve_tag: str | None = None) -> None:
        """``cohort_flags`` maps a window name to an index (``block_key``, ``trial_in_block``,
        ``artifact_window``) from another store whose artifact flags define the cohort, so an
        ablation on differently processed windows uses exactly the primary analysis's trials."""
        self.cfg = cfg
        self.cohort_flags = cohort_flags or {}
        self.excluded_blocks = set(excluded_blocks or ())
        self.embedder_factory = embedder_factory
        self.reve_tag = reve_tag
        if cfg.exclude_amplitude_outlier_blocks and not self.excluded_blocks:
            raise ValueError("exclude_amplitude_outlier_blocks is set but no blocks were supplied")
        self.compute = compute
        self.trials = trials
        self.windows = windows
        self.split = split
        self.demographics = demographics
        self.positions = positions
        self.out_dir = out_dir
        self.feature_root = feature_root
        self.fingerprint = config_fingerprint(cfg)
        self.started = datetime.now(tz=UTC).isoformat(timespec="seconds")
        self.failures: list[dict] = []

    def cohort(self, window: WindowData) -> pd.DataFrame:
        index = window.index.copy()
        flags = self.cohort_flags.get(window.spec.name)
        if flags is not None:
            keys = ["block_key", "trial_in_block"]
            flag_columns = [c for c in ("artifact_window", "high_amplitude_window", "flat_window") if c in flags]
            merged = index.drop(columns=[c for c in flag_columns if c in index]).merge(
                flags[[*keys, *flag_columns]], on=keys, how="left", validate="one_to_one")
            if merged["artifact_window"].isna().any() or len(merged) != len(index):
                raise ValueError(f"window {window.spec.name}: cohort flags do not cover every row")
            index = merged
        index["fold"] = self.split.fold_of(index["subject"]).astype("Int64")
        valid = index["window_valid"] & index["fold"].notna()
        if self.cfg.exclude_artifact_windows:
            if self.cfg.artifact_flag in index:
                valid &= ~index[self.cfg.artifact_flag].fillna(False).astype(bool)
            elif self.cfg.artifact_flag != "artifact_window":
                raise ValueError(f"window {window.spec.name}: index lacks the artifact flag {self.cfg.artifact_flag!r}")
        if not self.cfg.evaluate_holdout_release:
            valid &= index["fold"] != HOLDOUT_FOLD
        if self.excluded_blocks:
            valid &= ~index["block_key"].isin(self.excluded_blocks)
        per_subject = index[valid & index["valid_response"]].groupby("subject").size()
        enough = per_subject[per_subject >= self.cfg.min_valid_responses_per_subject].index
        if self.cfg.min_rt_s is not None:            # after the 10-response rule: removes no child
            valid &= ~(index["valid_response"].astype(bool) & (index["rt_s"] < self.cfg.min_rt_s))
        cohort = index[valid & index["subject"].isin(enough)].copy()
        if self.cfg.subsample_subjects:
            rng = np.random.default_rng(self.cfg.seed)
            keep = rng.choice(np.array(sorted(cohort["subject"].unique())),
                              size=min(self.cfg.subsample_subjects, cohort["subject"].nunique()),
                              replace=False)
            cohort = cohort[cohort["subject"].isin(keep)]
        cohort = cohort.merge(self.demographics, on="subject", how="left")
        return cohort.reset_index(drop=True)

    @staticmethod
    def _covariates(rows: pd.DataFrame) -> np.ndarray:
        sex = rows["sex"].astype(str).str.upper().map({"M": 1.0, "F": 0.0})
        return np.column_stack([
            pd.to_numeric(rows["age"], errors="coerce").to_numpy(dtype=float),
            sex.to_numpy(dtype=float),
            pd.to_numeric(rows["ehq_total"], errors="coerce").to_numpy(dtype=float),
        ])

    def _make_decoder(self, name: str, window: WindowData) -> Decoder:
        ridge_kwargs = {"alphas": tuple(self.cfg.ridge_alphas), "n_inner_folds": self.cfg.inner_folds,
                        "seed": self.cfg.seed}
        if name == "mean":
            return MeanPredictor()
        if name in ("demographics", "spectral", "spectral_aperiodic", *REVE_MODELS):
            return GroupedRidge(name=name, **ridge_kwargs)
        if name == "timecourse":
            return GroupedRidge(name=name, pca_components=self.cfg.timecourse_pca_components,
                                **ridge_kwargs)
        if name == "riemann":
            device = None if self.compute.device == "cpu" else f"cuda:{self.compute.cuda_device_index}"
            return TangentSpaceRidge(name="riemann", projection_rank=self.cfg.riemann_projection_rank,
                                     mean_subsample=self.cfg.riemann_mean_subsample,
                                     pca_components=self.cfg.riemann_pca_components, device=device,
                                     **ridge_kwargs)
        if name == "eegnet":
            from decoding.eegnet import EEGNetRegressor

            e = self.cfg.eegnet
            return EEGNetRegressor(window.array.shape[1], window.array.shape[2],
                                   compute=self.compute, max_epochs=e.max_epochs,
                                   batch_size=e.batch_size, lr=e.lr, weight_decay=e.weight_decay,
                                   patience=e.patience, seeds=tuple(e.seeds), decimate=e.decimate,
                                   kernel_length=e.kernel_length, fast_spatial=e.fast_spatial,
                                   autocast=e.autocast_bf16)
        raise ValueError(f"unknown model '{name}'")

    def _inputs(self, name: str, window: WindowData, cache: FeatureCache,
                rows: pd.DataFrame) -> DecoderInput:
        """EEG feature models also receive age, sex and handedness, so their paired difference
        against the demographics-only model isolates the contribution of the EEG."""
        subjects = rows["subject"].to_numpy()
        covariates = self._covariates(rows)
        row_idx = rows["row"].to_numpy()
        if name in ("mean", "demographics"):
            return DecoderInput(np.zeros((len(rows), 0)), subjects, covariates)
        if name == "spectral":
            return DecoderInput(np.asarray(cache.spectral()[row_idx]), subjects, covariates)
        if name == "spectral_aperiodic":
            aperiodic = np.asarray(cache.aperiodic()[row_idx])
            if not self.cfg.aperiodic_include_fit_r2:          # columns: exponent, offset, fit R2
                aperiodic = aperiodic[:, :2 * self.cfg.aperiodic_regions]
            x = np.concatenate([np.asarray(cache.spectral()[row_idx]), aperiodic], axis=1)
            return DecoderInput(x, subjects, covariates)
        if name == "riemann":
            return DecoderInput(np.asarray(cache.covariances()[row_idx]), subjects, covariates)
        if name == "timecourse":
            return DecoderInput(np.asarray(cache.timecourse(self.cfg.timecourse_bin_ms)[row_idx]),
                                subjects, covariates)
        if name == "eegnet":
            return DecoderInput(RowView(window.array, row_idx), subjects, None)
        if name in REVE_MODELS:
            return DecoderInput(np.asarray(cache.reve(REVE_MODELS[name])[row_idx]), subjects, covariates)
        raise ValueError(name)

    def run(self) -> pd.DataFrame:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        device = "cuda" if self.compute.device != "cpu" else None
        all_predictions: list[pd.DataFrame] = []
        summaries: list[dict] = []
        for window_name in self.cfg.windows:
            window = self.windows[window_name]
            cache = FeatureCache(self.feature_root, window, self.cfg.aperiodic_regions,
                                 self.positions, device, n_jobs=self.cfg.feature_n_jobs,
                                 embedder_factory=self.embedder_factory, reve_tag=self.reve_tag)
            cohort = self.cohort(window)
            if cohort.empty:
                raise ValueError(f"window {window_name}: empty cohort (no subject with at least "
                                 f"{self.cfg.min_valid_responses_per_subject} valid responses)")
            _LOG.info("window %s: cohort %d trials, %d subjects", window_name, len(cohort),
                      cohort["subject"].nunique())
            for target in self.cfg.targets:
                rows = cohort if target == "lapse" else cohort[cohort["valid_response"]]
                rows = rows.reset_index(drop=True)
                y_all = rows[target].to_numpy(dtype=float)
                for model_name in self.cfg.models:
                    try:
                        preds = self._cross_validate(model_name, window, cache, rows, y_all, target)
                    except Exception as error:  # noqa: BLE001 - recorded, reported and re-raised on request
                        if not self.cfg.continue_on_model_error:
                            raise
                        _LOG.exception("%s | %s | %s failed; continuing with the next model",
                                       window_name, target, model_name)
                        self.failures.append({
                            "window": window_name, "target": target, "model": model_name,
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                            "utc": datetime.now(tz=UTC).isoformat(timespec="seconds")})
                        self._checkpoint(all_predictions, summaries)
                        continue
                    all_predictions.append(preds)
                    summaries.extend(self._summarise(preds, window_name, target, model_name,
                                                     all_predictions))
                    self._checkpoint(all_predictions, summaries)
        return pd.DataFrame(summaries)

    def _cross_validate(self, model_name: str, window: WindowData, cache: FeatureCache,
                        rows: pd.DataFrame, y_all: np.ndarray, target: str) -> pd.DataFrame:
        outputs = []
        folds = list(self.split.iter_folds())
        if self.cfg.evaluate_holdout_release:
            folds.append((HOLDOUT_FOLD, sorted(self.split.cv_subjects["subject"]),
                          self.split.holdout_subjects))
        for fold, train_subjects, test_subjects in folds:
            train_mask = rows["subject"].isin(train_subjects).to_numpy()
            test_mask = rows["subject"].isin(test_subjects).to_numpy()
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            LeakageAuditor.assert_rows_grouped(rows, train_mask, test_mask)
            start = time.perf_counter()
            decoder = self._make_decoder(model_name, window)
            train_rows = rows[train_mask].reset_index(drop=True)
            test_rows = rows[test_mask].reset_index(drop=True)
            decoder.fit(self._inputs(model_name, window, cache, train_rows), y_all[train_mask])
            pred = decoder.predict(self._inputs(model_name, window, cache, test_rows))
            elapsed = time.perf_counter() - start
            out = test_rows[["subject", "release", "block_key", "trial_index"]].copy()
            out["fold"] = fold
            out["y_true"] = y_all[test_mask]
            out["y_pred"] = pred
            out["model"] = model_name
            out["window"] = window.spec.name
            out["target"] = target
            out["fit_seconds"] = elapsed
            out["decoder_info"] = json.dumps(decoder.describe(), default=str)
            outputs.append(out)
            if self.cfg.log_fold_scores:
                score = nrmse(out["y_true"], out["y_pred"]) if target in _REGRESSION_TARGETS \
                    else auroc(out["y_true"], out["y_pred"])
                _LOG.info("%s | %s | %s | fold %s: %.4f (%d trials, %.0f s)", window.spec.name,
                          target, model_name, fold, score, len(out), elapsed)
            else:                                    # blinded progress log; scores are in the files
                _LOG.info("%s | %s | %s | fold %s done (%d trials, %.0f s)", window.spec.name,
                          target, model_name, fold, len(out), elapsed)
        if not outputs:
            raise ValueError(f"{model_name}: no fold had both training and test rows")
        return pd.concat(outputs, ignore_index=True)

    def _summarise(self, preds: pd.DataFrame, window: str, target: str, model: str,
                   all_predictions: list[pd.DataFrame]) -> list[dict]:
        metric = nrmse if target in _REGRESSION_TARGETS else auroc
        rows = []
        for scope, part in (("cv", preds[preds["fold"] != HOLDOUT_FOLD]),
                            ("holdout_R5", preds[preds["fold"] == HOLDOUT_FOLD])):
            if part.empty:
                continue
            boot = subject_bootstrap(metric, part["y_true"].to_numpy(), part["y_pred"].to_numpy(),
                                     part["subject"].to_numpy(), n_bootstrap=self.cfg.n_bootstrap,
                                     seed=self.cfg.seed)
            row = {
                "benchmark": self.cfg.name, "config_fingerprint": self.fingerprint,
                "started_utc": self.started, "window": window, "target": target, "model": model,
                "scope": scope, "n_trials": int(len(part)), "n_subjects": int(part["subject"].nunique()),
                "metric": "nrmse" if target in _REGRESSION_TARGETS else "auroc",
                "estimate": boot.estimate, "ci_low": boot.ci_low, "ci_high": boot.ci_high,
                "r2": r2_from_nrmse(boot.estimate) if target in _REGRESSION_TARGETS else None,
                "fit_seconds_total": float(part.drop_duplicates("fold")["fit_seconds"].sum()),
            }
            if target in _REGRESSION_TARGETS and scope == "cv":
                folds = per_fold_table(part)
                row["nrmse_fold_mean"] = float(folds["nrmse"].mean())
                row["nrmse_fold_sd"] = float(folds["nrmse"].std())
            for reference in ("mean", "demographics"):
                ref = [p for p in all_predictions
                       if p["model"].iloc[0] == reference and p["window"].iloc[0] == window
                       and p["target"].iloc[0] == target]
                if ref and reference != model:
                    ref_part = ref[0]
                    ref_part = (ref_part[ref_part["fold"] == HOLDOUT_FOLD] if scope == "holdout_R5"
                                else ref_part[ref_part["fold"] != HOLDOUT_FOLD])
                    merged = part.merge(ref_part[["subject", "block_key", "trial_index", "y_pred"]],
                                        on=["subject", "block_key", "trial_index"],
                                        suffixes=("", "_ref"))
                    paired = paired_subject_bootstrap(
                        metric, merged["y_true"].to_numpy(), merged["y_pred"].to_numpy(),
                        merged["y_pred_ref"].to_numpy(), merged["subject"].to_numpy(),
                        n_bootstrap=self.cfg.n_bootstrap, seed=self.cfg.seed)
                    row[f"delta_vs_{reference}"] = paired.estimate
                    row[f"delta_vs_{reference}_ci_low"] = paired.ci_low
                    row[f"delta_vs_{reference}_ci_high"] = paired.ci_high
            rows.append(row)
        return rows

    def _checkpoint(self, predictions: list[pd.DataFrame], summaries: list[dict]) -> None:
        if predictions:
            write_dataframe(pd.concat(predictions, ignore_index=True),
                            self.out_dir / "predictions.parquet", on_exists="overwrite")
        write_dataframe(pd.DataFrame(summaries), self.out_dir / "summary.csv", on_exists="overwrite")
        write_json({"config_fingerprint": self.fingerprint, "config": self.cfg.model_dump(mode="json"),
                    "split_fingerprint": self.split.fingerprint, "started_utc": self.started,
                    "window_fingerprints": {k: w.fingerprint for k, w in self.windows.items()},
                    "window_meta": {k: w.meta for k, w in self.windows.items()},
                    "failures": self.failures},
                   self.out_dir / "run.json", on_exists="overwrite")


def index_fingerprint(index: pd.DataFrame) -> str:
    return sha256_text(index[["block_key", "trial_in_block"]].astype(str).to_csv(index=False))


def store_fingerprint(index: pd.DataFrame, meta: dict) -> str:
    """Row identity plus the build identifier of the store, so a store rebuilt under the same
    name never reuses cached features. Stores built before build identifiers existed keep the
    row-identity fingerprint (and therefore their existing caches)."""
    rows = index_fingerprint(index)
    return sha256_text(rows + meta["store_id"]) if meta.get("store_id") else rows


def load_window_data(root: Path, spec: WindowSpec, sfreq: float) -> WindowData:
    store = WindowStore(root, spec, sfreq)
    array, index, meta = store.open()
    return WindowData(spec, array, index, float(meta["sfreq"]), store_fingerprint(index, meta), meta)


def demographics_table(participants: pd.DataFrame) -> pd.DataFrame:
    """Only age, sex and handedness leave the participants table; never phenotypes."""
    frame = participants.drop_duplicates("participant_id").copy()
    frame["subject"] = frame["participant_id"].astype(str).str.replace("^sub-", "", regex=True)
    return frame[["subject", *DEMOGRAPHIC_COLUMNS]].reset_index(drop=True)
