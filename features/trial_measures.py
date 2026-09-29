"""Single-trial signal measures from target-locked epochs.

Epochs run from ``tmin_s`` (-1.0 s) to +2.5 s around the target on the cluster channels
(``features.clusters``), in volts at ``sfreq``. Measures:

* ``cpp_slope_resp``: least-squares slope (uV/s) of the CPP cluster mean from 350 to 50 ms before
  the response, after an 8 Hz zero-phase low-pass (4th-order Butterworth). Valid responses with
  0.65 s <= RT <= 2.4 s. Placed by the RT: an association with RT, never a predictor of it.
* ``cpp_amp_resp``: CPP mean from 100 to 0 ms before the response minus the mean from 100 to 0 ms
  before the target; same trials.
* ``cpp_slope_target``: slope from 200 to 500 ms after the target, same filter; trials with RT above
  0.6 s (the window ends before the response) and misses. Defined without the RT.
* ``alpha_logpow``: log10 mean 8-13 Hz power of the alpha cluster from -1.0 to 0 s (Welch, 500 ms
  Hann segments, 50% overlap). Every trial.
* ``ssvep_logpow``: mean over the SSVEP cluster and over 20 and 25 Hz of log10 power of the
  -1.0 to 0 s segment (Hann-windowed FFT, 1 Hz resolution). Every trial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

CPP_RESP_WINDOW_S = (-0.35, -0.05)
CPP_AMP_WINDOW_S = (-0.10, 0.0)
CPP_TARGET_WINDOW_S = (0.20, 0.50)
BASELINE_S = (-0.10, 0.0)
MIN_RT_RESP_S, MAX_RT_RESP_S = 0.65, 2.4
MIN_RT_TARGET_S = 0.6
ALPHA_BAND = (8.0, 13.0)
SSVEP_HZ = (20.0, 25.0)


def _slope(y: np.ndarray, sfreq: float) -> float:
    t = np.arange(len(y)) / sfreq
    t = t - t.mean()
    return float(t @ (y - y.mean()) / (t @ t))


def trial_measures(epochs: np.ndarray, channels: list[str], clusters: dict[str, list[str]], trials: pd.DataFrame,
                   sfreq: float, tmin_s: float = -1.0) -> pd.DataFrame:
    """One row per epoch (same order as ``trials``, which needs rt_s and valid_response)."""
    index = {ch: i for i, ch in enumerate(channels)}
    pick = {name: [index[ch] for ch in chans] for name, chans in clusters.items()}
    zero = int(round(-tmin_s * sfreq))

    def sample(t_s: float) -> int:
        return zero + int(round(t_s * sfreq))

    sos = butter(4, 8.0, btype="low", fs=sfreq, output="sos")
    cpp = sosfiltfilt(sos, epochs[:, pick["cpp"]].mean(axis=1) * 1e6, axis=-1)         # uV
    base = cpp[:, sample(BASELINE_S[0]):sample(BASELINE_S[1])].mean(axis=1)
    rt = trials["rt_s"].to_numpy(float)
    valid = trials["valid_response"].to_numpy(bool)
    n = len(trials)
    slope_resp, amp_resp, slope_target = (np.full(n, np.nan) for _ in range(3))
    for i in range(n):
        if valid[i] and MIN_RT_RESP_S <= rt[i] <= MAX_RT_RESP_S:
            a, b = sample(rt[i] + CPP_RESP_WINDOW_S[0]), sample(rt[i] + CPP_RESP_WINDOW_S[1])
            slope_resp[i] = _slope(cpp[i, a:b], sfreq)
            c, d = sample(rt[i] + CPP_AMP_WINDOW_S[0]), sample(rt[i] + CPP_AMP_WINDOW_S[1])
            amp_resp[i] = cpp[i, c:d].mean() - base[i]
        if (not valid[i]) or rt[i] > MIN_RT_TARGET_S:
            slope_target[i] = _slope(cpp[i, sample(CPP_TARGET_WINDOW_S[0]):sample(CPP_TARGET_WINDOW_S[1])], sfreq)
    pre = slice(sample(-1.0), sample(0.0))
    freqs, psd = welch(epochs[:, pick["alpha"], pre], fs=sfreq, window="hann", nperseg=int(0.5 * sfreq),
                       noverlap=int(0.25 * sfreq), axis=-1)
    band = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha = np.log10(psd[..., band].mean(axis=-1).mean(axis=1))
    segment = epochs[:, pick["ssvep"], pre]
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.shape[-1]), axis=-1)) ** 2
    fft_freqs = np.fft.rfftfreq(segment.shape[-1], d=1.0 / sfreq)
    bins = [int(np.argmin(np.abs(fft_freqs - f))) for f in SSVEP_HZ]
    ssvep = np.log10(spectrum[..., bins]).mean(axis=(1, 2))
    return pd.DataFrame({"cpp_slope_resp": slope_resp, "cpp_amp_resp": amp_resp, "cpp_slope_target": slope_target,
                         "alpha_logpow": alpha, "ssvep_logpow": ssvep})
