"""Welch band-power features per channel; the 2015-era baseline representation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta_low": (13.0, 20.0), "beta_high": (20.0, 30.0), "gamma": (30.0, 45.0),
}


@dataclass
class SpectralFeatures:
    sfreq: float
    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BANDS))
    nperseg_s: float = 1.0
    log_floor: float = 1e-30

    def psd(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Welch PSD with a Hann window and 50% overlap; returns (freqs, psd[n, ch, f])."""
        nperseg = int(round(self.nperseg_s * self.sfreq))
        freqs, psd = welch(windows, fs=self.sfreq, window="hann", nperseg=nperseg,
                           noverlap=nperseg // 2, axis=-1, detrend="constant")
        return freqs, psd

    def transform(self, windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Log band power per channel and band, flattened as [channel-major] features."""
        freqs, psd = self.psd(windows)
        feats = []
        names = []
        n_channels = windows.shape[1]
        for band, (lo, hi) in self.bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            power = psd[:, :, mask].mean(axis=-1)
            feats.append(np.log10(np.maximum(power, self.log_floor)))
            names.extend(f"{band}_ch{c}" for c in range(n_channels))
        total = psd[:, :, (freqs >= 1.0) & (freqs < 45.0)].mean(axis=-1)
        feats.append(np.log10(np.maximum(total, self.log_floor)))
        names.extend(f"total_ch{c}" for c in range(n_channels))
        return np.concatenate(feats, axis=1).astype(np.float32), names
