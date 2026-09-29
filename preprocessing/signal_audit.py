"""Raw-signal diagnostics for quality control: detect recordings that carry no differential EEG.

Some HBN recordings contain the same signal on every channel (raw inter-channel correlation of
1.000, identical channel standard deviations, a common offset near -85 to -99 mV), which is what
an electrode montage without a working reference produces. Average referencing removes the
common signal and leaves only a residual of 0.1 to 3 uV. PREP-style detection cannot flag these
recordings because its correlation criterion looks for channels that correlate too little with
the rest, not for all channels correlating perfectly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_LOG = logging.getLogger(__name__)


@dataclass
class CommonModeDiagnostics:
    block_key: str
    seconds_used: float
    raw_sd_uv_median: float
    raw_sd_cv: float
    raw_offset_uv_median: float
    interchannel_r_median: float
    interchannel_r_p05: float
    error: str | None = None


def common_mode_diagnostics(block_key: str, eeg_path: str, tmin: float, tmax: float,
                            seconds: float = 60.0, exclude: tuple[str, ...] = ("Cz",)) -> CommonModeDiagnostics:
    """Median pairwise correlation of raw (recording-reference) channels over ``seconds``."""
    import mne

    try:
        raw = mne.io.read_raw_eeglab(eeg_path, preload=False, verbose="error")
        stop = min(tmax, tmin + seconds, raw.times[-1])
        picks = [c for c in raw.ch_names if c not in exclude]
        x = raw.get_data(picks=picks, tmin=tmin, tmax=stop) * 1e6
        sd = x.std(axis=1)
        usable = sd > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(x[usable])
        upper = r[np.triu_indices(r.shape[0], 1)] if r.ndim == 2 else np.array([np.nan])
        return CommonModeDiagnostics(
            block_key=block_key, seconds_used=float(stop - tmin),
            raw_sd_uv_median=float(np.median(sd)),
            raw_sd_cv=float(sd.std() / sd.mean()) if sd.mean() > 0 else float("nan"),
            raw_offset_uv_median=float(np.median(np.median(x, axis=1))),
            interchannel_r_median=float(np.nanmedian(upper)),
            interchannel_r_p05=float(np.nanpercentile(upper, 5)))
    except Exception as exc:  # noqa: BLE001 - recorded per block
        return CommonModeDiagnostics(block_key, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan,
                                     error=f"{type(exc).__name__}: {exc}")


def audit_common_mode(blocks: pd.DataFrame, n_jobs: int = 8, seconds: float = 60.0) -> pd.DataFrame:
    """Diagnostics for every row of ``blocks`` (needs block_key, eeg_path, crop_tmin_s, crop_tmax_s)."""
    _LOG.info("common-mode audit of %d blocks with %d workers", len(blocks), n_jobs)
    rows = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(common_mode_diagnostics)(r.block_key, r.eeg_path, float(r.crop_tmin_s),
                                         float(r.crop_tmax_s), seconds)
        for r in blocks.itertuples(index=False))
    return pd.DataFrame([asdict(r) for r in rows])
