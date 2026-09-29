"""Aperiodic (exponent, offset) parameterisation with specparam on regional average spectra.

Fitting every channel of every trial is computationally prohibitive at HBN scale, so spectra are
averaged inside regions defined by clustering the template electrode coordinates, following the
regional approach in the specification (T3.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

_LOG = logging.getLogger(__name__)


def average_regions(psd: np.ndarray, labels: np.ndarray, n_regions: int) -> np.ndarray:
    """psd[n, channels, f] -> [n, regions, f] mean spectrum per region label."""
    out = np.zeros((psd.shape[0], n_regions, psd.shape[2]), dtype=psd.dtype)
    for region in range(n_regions):
        out[:, region] = psd[:, labels == region].mean(axis=1)
    return out


def _aperiodic_chunk(array, start: int, stop: int, sfreq: float, labels: np.ndarray,
                     n_regions: int, fitter_kwargs: dict) -> tuple[int, np.ndarray]:
    from features.spectral import SpectralFeatures

    freqs, psd = SpectralFeatures(sfreq).psd(np.asarray(array[start:stop]))
    feats, _ = AperiodicFeatures(**fitter_kwargs, n_jobs=1).transform(
        freqs, average_regions(psd, labels, n_regions))
    return start, feats


def aperiodic_features_parallel(array, sfreq: float, labels: np.ndarray, n_regions: int,
                                n_jobs: int = 8, chunk: int = 2048, **fitter_kwargs) -> np.ndarray:
    """Regional specparam features for every row of ``array`` (memmap-friendly, one pool)."""
    from joblib import Parallel, delayed

    n = len(array)
    results = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_aperiodic_chunk)(array, start, min(start + chunk, n), sfreq, labels, n_regions,
                                  fitter_kwargs)
        for start in range(0, n, chunk))
    out = np.full((n, 3 * n_regions), np.nan, dtype=np.float32)
    for start, feats in results:
        out[start:start + len(feats)] = feats
    return out


class RegionalAverager:
    """Assign channels to k regions by k-means on 3D template positions (deterministic)."""

    def __init__(self, positions: np.ndarray, n_regions: int = 10, seed: int = 12345) -> None:
        from sklearn.cluster import KMeans

        self.positions = np.asarray(positions, dtype=float)
        self.n_regions = n_regions
        model = KMeans(n_clusters=n_regions, n_init=10, random_state=seed)
        self.labels = model.fit_predict(self.positions)
        self.centres = model.cluster_centers_
        order = np.lexsort((self.centres[:, 0], -self.centres[:, 1]))
        remap = {old: new for new, old in enumerate(order)}
        self.labels = np.array([remap[label] for label in self.labels])
        self.centres = self.centres[order]

    def average(self, psd: np.ndarray) -> np.ndarray:
        """psd[n, channels, f] -> [n, regions, f] mean spectrum per region."""
        out = np.zeros((psd.shape[0], self.n_regions, psd.shape[2]), dtype=psd.dtype)
        for region in range(self.n_regions):
            out[:, region] = psd[:, self.labels == region].mean(axis=1)
        return out


@dataclass
class AperiodicFeatures:
    """specparam (Donoghue et al. 2020) fits of regional spectra.

    The lower peak-width limit is 2 Hz because the 1 s Welch segments give 1 Hz resolution and
    specparam's documentation recommends a lower bound of at least twice the frequency
    resolution; the specification's 1 Hz lower bound assumed finer resolution.
    """

    freq_range: tuple[float, float] = (2.0, 40.0)
    max_n_peaks: int = 6
    peak_width_limits: tuple[float, float] = (2.0, 8.0)
    aperiodic_mode: str = "fixed"
    min_r_squared: float = 0.0
    n_jobs: int = 1

    def transform(self, freqs: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """psd[n, regions, f] -> exponent, offset, r2 per region, flattened."""
        from specparam import SpectralGroupModel

        n, n_regions, _ = psd.shape
        exponent = np.full((n, n_regions), np.nan, dtype=np.float32)
        offset = np.full((n, n_regions), np.nan, dtype=np.float32)
        r2 = np.full((n, n_regions), np.nan, dtype=np.float32)
        in_range = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        # Spectra that are zero or non-finite in the fit range (zero-filled invalid windows)
        # cannot be log-transformed; they keep NaN features.
        fittable = np.isfinite(psd[:, :, in_range]).all(axis=2) & (psd[:, :, in_range] > 0).all(axis=2)
        for region in range(n_regions):
            rows = np.flatnonzero(fittable[:, region])
            if len(rows) == 0:
                continue
            model = SpectralGroupModel(peak_width_limits=self.peak_width_limits,
                                       max_n_peaks=self.max_n_peaks,
                                       aperiodic_mode=self.aperiodic_mode, verbose=False)
            model.fit(freqs, psd[rows, region], freq_range=list(self.freq_range), n_jobs=self.n_jobs)
            params = model.get_params("aperiodic")
            exponent[rows, region] = params[:, -1]
            offset[rows, region] = params[:, 0]
            r2[rows, region] = np.asarray(model.get_metrics("gof_rsquared"), dtype=float)
        poor = r2 < self.min_r_squared
        exponent[poor] = np.nan
        offset[poor] = np.nan
        names = ([f"exponent_r{r}" for r in range(n_regions)]
                 + [f"offset_r{r}" for r in range(n_regions)]
                 + [f"r2_r{r}" for r in range(n_regions)])
        return np.concatenate([exponent, offset, r2], axis=1), names
