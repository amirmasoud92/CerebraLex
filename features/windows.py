"""Extract fixed-length windows around the contrast-change onset and store them as memmaps.

Windows are cut from the cleaned 250 Hz blocks using the sample indices of the trial table, so
every model and feature set downstream sees identical inputs. The store keeps one float32
memmap per window definition plus an index table linking rows to trials.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.schema import WindowQCConfig, WindowSpec
from preprocessing.storage import read_block

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowExtractor:
    window: WindowSpec
    sfreq: float

    @property
    def n_samples(self) -> int:
        return int(round(self.window.length_s * self.sfreq))

    @property
    def offset_samples(self) -> int:
        return int(round(self.window.tmin_s * self.sfreq))

    def extract(self, data: np.ndarray, target_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (windows, valid_mask); windows for invalid rows are zero."""
        starts = np.asarray(target_samples, dtype=int) + self.offset_samples
        stops = starts + self.n_samples
        valid = (starts >= 0) & (stops <= data.shape[1])
        out = np.zeros((len(starts), data.shape[0], self.n_samples), dtype=np.float32)
        for i in np.flatnonzero(valid):
            out[i] = data[:, starts[i]:stops[i]]
        return out, valid


class WindowStore:
    """Memmap-backed window array plus an index parquet, written once per window definition.

    The index carries per-window amplitude statistics and the artifact flags defined by
    ``WindowQCConfig``: a window is a ``high_amplitude_window`` when more than
    ``max_bad_channel_fraction`` of channels exceed ``p2p_threshold_uv`` peak-to-peak, and a
    ``flat_window`` when any channel's peak-to-peak lies below ``flat_threshold_uv`` (dead
    signal). ``artifact_window`` is their union.

    Signal variants for sensitivity analyses: ``highpass_hz`` applies an extra zero-phase
    FIR high-pass (MNE defaults) to each cleaned continuous block before cutting, and a store rate
    ``sfreq`` below the block rate resamples the continuous block (polyphase, anti-aliased) and
    rescales the trial samples. Blocks may come from several preprocessing directories (first match
    wins). ``open`` refuses a window definition that differs from the stored one.
    """

    QUALITY_COLUMNS = ("p2p_max_uv", "p2p_median_uv", "p2p_min_uv",
                       "frac_channels_p2p_over_threshold", "high_amplitude_window", "flat_window",
                       "artifact_window")

    def __init__(self, root: Path, window: WindowSpec, sfreq: float, n_channels: int = 129,
                 qc: WindowQCConfig | None = None, highpass_hz: float | None = None,
                 channels: list[str] | None = None) -> None:
        self.root = root / window.name
        self.window = window
        self.sfreq = sfreq
        self.source_channels = n_channels                 # every cleaned block must have these
        self.channels = list(channels) if channels else None
        self.n_channels = len(self.channels) if self.channels else n_channels
        self.extractor = WindowExtractor(window, sfreq)
        self.qc = qc or WindowQCConfig()
        self.highpass_hz = highpass_hz

    def _condition(self, data: np.ndarray, source_sfreq: float) -> np.ndarray:
        """Extra high-pass at the block rate, then resampling to the store rate (both optional)."""
        if self.highpass_hz is not None:
            import mne

            data = mne.filter.filter_data(np.asarray(data, dtype=np.float64), source_sfreq,
                                          l_freq=self.highpass_hz, h_freq=None, verbose=False)
        if not np.isclose(source_sfreq, self.sfreq):
            from fractions import Fraction

            from scipy.signal import resample_poly

            ratio = Fraction(float(self.sfreq) / float(source_sfreq)).limit_denominator(1000)
            data = resample_poly(np.asarray(data, dtype=np.float64), ratio.numerator, ratio.denominator, axis=1)
        return np.asarray(data, dtype=np.float32)

    @property
    def array_path(self) -> Path:
        return self.root / "windows.float32"

    @property
    def index_path(self) -> Path:
        return self.root / "index.parquet"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def exists(self) -> bool:
        return self.array_path.is_file() and self.index_path.is_file() and self.meta_path.is_file()

    def quality(self, windows: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
        """Amplitude statistics (uV) and artifact flags for a batch of windows stored in volts."""
        p2p = np.ptp(np.asarray(windows, dtype=np.float32), axis=2).astype(np.float64) * 1e6
        frac_high = (p2p > self.qc.p2p_threshold_uv).mean(axis=1)
        high = frac_high > self.qc.max_bad_channel_fraction
        flat = np.asarray(valid, dtype=bool) & (p2p.min(axis=1) < self.qc.flat_threshold_uv)
        return {"p2p_max_uv": p2p.max(axis=1).astype(np.float32),
                "p2p_median_uv": np.median(p2p, axis=1).astype(np.float32),
                "p2p_min_uv": p2p.min(axis=1).astype(np.float32),
                "frac_channels_p2p_over_threshold": frac_high.astype(np.float32),
                "high_amplitude_window": high, "flat_window": flat, "artifact_window": high | flat}

    def build(self, trials: pd.DataFrame, preproc_dir: Path | list[Path], uncleaned: bool = False) -> pd.DataFrame:
        """Cut windows for every trial whose block exists on disk; returns the index table."""
        self.root.mkdir(parents=True, exist_ok=True)
        block_paths = self._block_paths(trials, preproc_dir)
        trials = trials[trials["block_key"].isin(block_paths)].copy()
        trials = trials.sort_values(["block_key", "trial_in_block"]).reset_index(drop=True)
        n = len(trials)
        shape = (n, self.n_channels, self.extractor.n_samples)
        _LOG.info("window store %s: %d trials, shape %s", self.window.name, n, shape)
        array = np.lib.format.open_memmap(self.array_path, mode="w+", dtype=np.float32, shape=shape)
        valid = np.zeros(n, dtype=bool)
        parts: list[dict[str, np.ndarray]] = []
        source_rates: set[float] = set()
        row = 0
        for block_key, group in trials.groupby("block_key", sort=False):
            block = read_block(block_paths[block_key], load_removed=uncleaned)
            data = block.uncleaned() if uncleaned else block.data
            if data.shape[0] != self.source_channels:
                raise ValueError(f"{block_key}: {data.shape[0]} channels, expected {self.source_channels}")
            if self.channels:
                data = data[[block.ch_names.index(ch) for ch in self.channels]]
            source_rates.add(float(block.sfreq))
            data = self._condition(data, block.sfreq)
            targets = group["target_sample"].to_numpy()
            if not np.isclose(block.sfreq, self.sfreq):
                targets = np.round(targets * self.sfreq / block.sfreq).astype(int)
            windows, ok = self.extractor.extract(data, targets)
            array[row:row + len(group)] = windows
            ok = ok & ~group["spans_boundary"].to_numpy(dtype=bool)
            valid[row:row + len(group)] = ok
            parts.append(self.quality(windows, ok))
            row += len(group)
        array.flush()
        del array
        index = trials.copy()
        index["row"] = np.arange(n)
        index["window_valid"] = valid
        for column in self.QUALITY_COLUMNS:
            index[column] = np.concatenate([part[column] for part in parts]) if parts else []
        index.to_parquet(self.index_path, index=False)
        meta = {"window": self.window.model_dump(), "sfreq": self.sfreq, "shape": list(shape),
                "uncleaned": uncleaned, "n_valid": int(valid.sum()),
                "highpass_hz": self.highpass_hz, "source_sfreq": sorted(source_rates),
                "channels": self.channels,
                "store_id": uuid.uuid4().hex,          # new for every build; keys feature caches
                **self._qc_meta(index)}
        self.meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return index

    def _qc_meta(self, index: pd.DataFrame) -> dict:
        return {"qc": self.qc.model_dump(),
                "n_high_amplitude_windows": int(index["high_amplitude_window"].sum()),
                "n_flat_windows": int(index["flat_window"].sum()),
                "n_artifact_windows": int(index["artifact_window"].sum())}

    @staticmethod
    def channel_names(trials: pd.DataFrame, preproc_dir: Path) -> list[str]:
        """Channel order shared by every stored block of these trials; raises if any block differs."""
        from preprocessing.storage import read_block_header

        names: list[str] | None = None
        for key, path in WindowStore._block_paths(trials, preproc_dir).items():
            current = list(read_block_header(path)["ch_names"])
            if names is None:
                names = current
            elif current != names:
                raise ValueError(f"{key}: channel order differs from the other blocks")
        if names is None:
            raise ValueError("no stored block found for these trials")
        return names

    @staticmethod
    def _block_paths(trials: pd.DataFrame, preproc_dir: Path | list[Path]) -> dict[str, Path]:
        directories = [Path(preproc_dir)] if isinstance(preproc_dir, str | Path) else [Path(d) for d in preproc_dir]
        paths: dict[str, Path] = {}
        for block_key, group in trials.groupby("block_key"):
            first = group.iloc[0]
            stem = (f"sub-{first['subject']}_{first['release']}_task-CCD_block-"
                    f"{int(first['block'])}_preproc.h5")
            for directory in directories:
                path = directory / f"sub-{first['subject']}" / stem
                if path.is_file():
                    paths[block_key] = path
                    break
        return paths

    def open(self) -> tuple[np.memmap, pd.DataFrame, dict]:
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        stored = meta.get("window", {})
        for key in ("tmin_s", "tmax_s"):
            if key in stored and not np.isclose(float(stored[key]), float(getattr(self.window, key))):
                raise ValueError(f"{self.root}: stored window {stored} differs from the requested {self.window}")
        array = np.load(self.array_path, mmap_mode="r")
        index = pd.read_parquet(self.index_path)
        return array, index, meta
