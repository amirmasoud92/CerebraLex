"""Individual preprocessing steps operating on MNE Raw objects. Each step returns QC facts."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import mne
import numpy as np

from common.schema import BadChannelConfig, FilterConfig, ICAConfig

_LOG = logging.getLogger(__name__)


@dataclass
class StepReport:
    name: str
    seconds: float
    facts: dict = field(default_factory=dict)


def timed(name: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result, facts = fn(*args, **kwargs)
            return result, StepReport(name, time.perf_counter() - start, facts)
        return wrapper
    return decorator


@timed("montage")
def apply_montage(raw: mne.io.BaseRaw, montage: str):
    raw.set_montage(mne.channels.make_standard_montage(montage), on_missing="raise")
    return raw, {"montage": montage}


@timed("crop")
def crop(raw: mne.io.BaseRaw, tmin: float, tmax: float):
    tmax = min(tmax, float(raw.times[-1]))
    raw.crop(tmin=tmin, tmax=tmax)
    return raw, {"tmin_s": tmin, "tmax_s": tmax, "duration_s": float(raw.times[-1])}


FLAT_THRESHOLD_UV = 1e-3


def flat_recording(raw: mne.io.BaseRaw, max_flat_fraction: float) -> StepReport:
    """Detect railed recordings before PREP: too many channels flat by SD or by MAD, or NaN.

    Mirrors pyprep's flat criterion (standard deviation or median absolute deviation near zero).
    Seventeen production blocks were railed (constant 194,000 uV, sometimes with rare jumps, so
    the MAD is zero while the SD is large); pyprep cannot run once no usable channel remains.
    Blocks with more flat channels than the bad-channel limit are excluded here, which is the
    decision the 20% rule would reach anyway.
    """
    start = time.perf_counter()
    data = raw.get_data(picks="eeg") * 1e6
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(data, axis=1)
        mad = np.nanmedian(np.abs(data - np.nanmedian(data, axis=1, keepdims=True)), axis=1)
    all_nan = np.isnan(data).all(axis=1)
    flat = (sd < FLAT_THRESHOLD_UV) | (mad < FLAT_THRESHOLD_UV) | all_nan
    facts = {"n_flat_or_nan_channels": int(flat.sum()), "n_channels": int(len(flat)),
             "flat_recording": bool(flat.sum() > max_flat_fraction * len(flat)),
             "max_flat_fraction": max_flat_fraction, "threshold_uv": FLAT_THRESHOLD_UV,
             "criterion": "SD or MAD below threshold, or all NaN"}
    return StepReport("flat_recording", time.perf_counter() - start, facts)


@timed("bad_channels")
def detect_bad_channels(raw: mne.io.BaseRaw, cfg: BadChannelConfig, filt: FilterConfig):
    """PREP-style detection on a broadband copy: 1 Hz high-pass plus notch, no low-pass."""
    from pyprep import NoisyChannels

    detect_raw = raw.copy()
    detect_raw.filter(l_freq=filt.ica_l_freq, h_freq=None, fir_design=filt.fir_design,
                      verbose="error")
    notches = [f for f in filt.notch_freqs if f < detect_raw.info["sfreq"] / 2]
    if notches:
        detect_raw.notch_filter(notches, verbose="error")
    detector = NoisyChannels(detect_raw, do_detrend=False, random_state=cfg.random_state)
    # Only the criteria published with PREP (Bigdely-Shamlo et al. 2015): pyprep 0.9 adds a
    # spectral z-score test (bad_by_psd) that flagged up to 40 channels per block on HBN.
    detector.find_bad_by_nan_flat()
    detector.find_bad_by_deviation()
    detector.find_bad_by_hfnoise()
    detector.find_bad_by_correlation()
    detector.find_bad_by_SNR()
    if cfg.ransac:
        detector.find_bad_by_ransac()
    bads = sorted(set(detector.get_bads()))
    by_type = {k: sorted(v) for k, v in detector.get_bads(as_dict=True).items() if v}
    return bads, {"bad_channels": bads, "n_bad": len(bads), "by_criterion": by_type,
                  "ransac": cfg.ransac, "criteria": ["nan_flat", "deviation", "hf_noise",
                                                     "correlation", "SNR", "ransac" if cfg.ransac else None]}


@timed("interpolate_reference")
def interpolate_and_rereference(raw: mne.io.BaseRaw, bads: list[str]):
    raw.info["bads"] = list(bads)
    if bads:
        raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="error")
    raw.set_eeg_reference("average", projection=False, verbose="error")
    return raw, {"n_interpolated": len(bads), "reference": "average"}


@timed("filter_resample")
def filter_and_resample(raw: mne.io.BaseRaw, l_freq: float, h_freq: float | None,
                        sfreq: float, fir_design: str):
    """Band-pass with a zero-phase FIR and resample after the anti-aliasing low-pass."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design=fir_design, phase="zero",
               verbose="error")
    original = float(raw.info["sfreq"])
    if not np.isclose(original, sfreq):
        raw.resample(sfreq, verbose="error")
    return raw, {"l_freq": l_freq, "h_freq": h_freq, "fir_design": fir_design, "phase": "zero",
                 "window": "hamming", "sfreq_in": original, "sfreq_out": float(raw.info["sfreq"])}


DEGENERATE_MEDIAN_SD_UV = 4.0
DEGENERATE_SD_CV = 0.10


def signal_quality(raw: mne.io.BaseRaw, exclude: list[str] | None = None) -> StepReport:
    """Amplitude statistics of a block and a flag for recordings without EEG.

    A block whose channels all share a tiny, nearly identical standard deviation carries no
    physiological signal (one such recording, 3 uV on every channel, made ICA run to its
    iteration cap). Interpolated channels are excluded from the uniformity statistic because
    interpolation lowers their variance. The thresholds are set from that observation,
    not from the literature, and are reported as such.
    """
    start = time.perf_counter()
    picks = [ch for ch in raw.ch_names if ch not in set(exclude or [])] or raw.ch_names
    data = raw.get_data(picks=picks) * 1e6
    sd = data.std(axis=1)
    p2p = np.ptp(data, axis=1)
    median_sd = float(np.median(sd))
    cv = float(sd.std() / max(sd.mean(), 1e-12))
    facts = {
        "channel_sd_uv_median": median_sd, "channel_sd_uv_min": float(sd.min()),
        "channel_sd_uv_max": float(sd.max()), "channel_sd_cv": cv,
        "p2p_uv_median": float(np.median(p2p)), "p2p_uv_max": float(p2p.max()),
        "abs_max_uv": float(np.abs(data).max()),
        "fraction_samples_abs_gt_100uv": float((np.abs(data) > 100).mean()),
        "degenerate_signal": bool(median_sd < DEGENERATE_MEDIAN_SD_UV and cv < DEGENERATE_SD_CV),
        "thresholds": {"median_sd_uv": DEGENERATE_MEDIAN_SD_UV, "sd_cv": DEGENERATE_SD_CV},
        "n_channels_evaluated": len(picks),
    }
    return StepReport("signal_quality", time.perf_counter() - start, facts)


@timed("notch")
def notch(raw: mne.io.BaseRaw, freqs: list[float]):
    """Zero-phase FIR notch below Nyquist; used on the ICA copy so line noise does not
    dominate the decomposition and the ICLabel line-noise class."""
    usable = [f for f in freqs if f < raw.info["sfreq"] / 2]
    if usable:
        raw.notch_filter(usable, verbose="error")
    return raw, {"freqs": usable}


@timed("ica")
def ica_clean(analysis: mne.io.BaseRaw, ica_copy: mne.io.BaseRaw, cfg: ICAConfig,
              n_interpolated: int):
    """Fit ICA on the 1 to 100 Hz copy, label with ICLabel, remove artifacts from the analysis copy.

    ICLabel was trained on average-referenced data filtered between 1 and 100 Hz, so the
    decomposition and its features are computed on ``ica_copy``; the unmixing is then applied
    to the 0.5 to 45 Hz analysis signal. The removed component time series are returned so the
    uncleaned analysis signal can be reconstructed exactly.
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    rank = len(analysis.ch_names) - n_interpolated - 1
    n_components = int(min(cfg.max_components, rank))
    if cfg.method == "infomax":
        fit_params: dict | None = {"extended": cfg.extended}
    elif cfg.method == "picard":
        fit_params = {"ortho": False, "extended": cfg.extended}
    else:
        fit_params = None
    ica = ICA(n_components=n_components, method=cfg.method, fit_params=fit_params,
              random_state=cfg.random_state, max_iter="auto")
    ica.fit(ica_copy, verbose="error")
    labels = label_components(ica_copy, ica, method="iclabel")
    names = list(labels["labels"])
    probs = np.asarray(labels["y_pred_proba"], dtype=float)
    exclude = [i for i, (name, p) in enumerate(zip(names, probs, strict=True))
               if name in cfg.reject_labels and p >= cfg.reject_threshold]
    sources = ica.get_sources(analysis).get_data()[exclude] if exclude else None
    mixing = None
    if exclude:
        pca = ica.pca_components_[: ica.n_components_]
        mixing = (pca.T @ ica.mixing_matrix_[:, exclude]) * ica.pre_whitener_
    variance_ratio = (
        ica.get_explained_variance_ratio(analysis, components=exclude, ch_type="eeg")["eeg"]
        if exclude else 0.0
    )
    n_pca_components = int(min(rank, ica.pca_components_.shape[0]))
    ica.apply(analysis, exclude=exclude, n_pca_components=n_pca_components, verbose="error")
    label_counts: dict[str, int] = {}
    for name in names:
        label_counts[name] = label_counts.get(name, 0) + 1
    facts = {
        "n_components": n_components, "n_pca_components_applied": n_pca_components,
        "method": cfg.method, "fit_params": fit_params, "n_iter": int(ica.n_iter_),
        "labels": names, "probabilities": probs.round(4).tolist(),
        "excluded": exclude, "n_excluded": len(exclude), "label_counts": label_counts,
        "artifact_variance_fraction": float(variance_ratio),
    }
    return (analysis, sources, mixing), facts
