"""Block-level preprocessing pipeline and a resumable parallel runner."""

from __future__ import annotations

import json
import logging
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import config_fingerprint
from common.provenance import git_commit
from common.schema import PreprocessingConfig
from preprocessing import steps
from preprocessing.storage import CleanedBlock, read_block_header, write_block

_LOG = logging.getLogger(__name__)
_PACKAGES = ("mne", "pyprep", "mne-icalabel", "numpy", "scipy")


@dataclass
class BlockResult:
    block_key: str
    subject: str
    release: str
    block: int
    status: str                       # ok | excluded_bad_channels | failed | skipped
    output_path: str | None
    qc_path: str | None
    seconds_total: float
    exclusion_reason: str | None = None
    error: str | None = None
    facts: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        facts = row.pop("facts")
        row.update({
            "n_bad_channels": facts.get("bad_channels", {}).get("n_bad"),
            "n_interpolated": facts.get("interpolate_reference", {}).get("n_interpolated"),
            "n_ica_components": facts.get("ica", {}).get("n_components"),
            "n_ica_excluded": facts.get("ica", {}).get("n_excluded"),
            "artifact_variance_fraction": facts.get("ica", {}).get("artifact_variance_fraction"),
            "duration_s": facts.get("crop", {}).get("duration_s"),
            "channel_sd_uv_median": facts.get("signal_quality", {}).get("channel_sd_uv_median"),
            "degenerate_signal": facts.get("signal_quality", {}).get("degenerate_signal"),
        })
        return row


_COMMIT = git_commit()


def read_raw_block(block: pd.Series):
    """Raw recording of a manifest row: EEGLAB (HBN BIDS) by default, raw EGI for MIPDB,
    where only the EEG channels are kept and the flat vertex reference E129 becomes HBN's Cz."""
    import mne

    if str(block.get("reader", "eeglab")) == "egi":
        from task_and_trials.mipdb import REFERENCE_CHANNEL

        raw = mne.io.read_raw_egi(block["eeg_path"], preload=True, verbose="error")
        raw.pick("eeg")
        if REFERENCE_CHANNEL in raw.ch_names:
            raw.rename_channels({REFERENCE_CHANNEL: "Cz"})
        return raw
    return mne.io.read_raw_eeglab(block["eeg_path"], preload=True, verbose="error")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "cerebralex_commit": _COMMIT}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


class BlockPreprocessor:
    """Apply the configured steps to one canonical CCD block and persist the result."""

    def __init__(self, config: PreprocessingConfig, out_dir: Path) -> None:
        self.config = config
        self.out_dir = out_dir
        self.fingerprint = config_fingerprint(config)

    def output_paths(self, block: pd.Series) -> tuple[Path, Path]:
        subject_dir = self.out_dir / f"sub-{block['subject']}"
        stem = f"sub-{block['subject']}_{block['release']}_task-CCD_block-{int(block['block'])}"
        return subject_dir / f"{stem}_preproc.h5", subject_dir / f"{stem}_qc.json"

    def is_done(self, block: pd.Series) -> bool:
        """Output exists for the current configuration, or the block was deliberately excluded
        under it; failures are retried."""
        h5_path, qc_path = self.output_paths(block)
        if not qc_path.is_file():
            return False
        if h5_path.is_file():
            try:
                return read_block_header(h5_path)["config_fingerprint"] == self.fingerprint
            except (OSError, KeyError):
                return False
        try:
            with open(qc_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        status = str(payload.get("result", {}).get("status", ""))
        return status.startswith("excluded_") and payload.get("config_fingerprint") == self.fingerprint

    def process(self, block: pd.Series) -> BlockResult:
        start = time.perf_counter()
        h5_path, qc_path = self.output_paths(block)
        key = str(block["block_key"])
        base = {"block_key": key, "subject": str(block["subject"]), "release": str(block["release"]),
                "block": int(block["block"])}
        if self.is_done(block):
            return BlockResult(**base, status="skipped", output_path=str(h5_path),
                               qc_path=str(qc_path), seconds_total=0.0)
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=self.config.threads_per_job):
                result = self._run(block, h5_path, qc_path, base, start)
        except Exception as exc:  # noqa: BLE001 - failures are data, recorded per block
            _LOG.error("block %s failed: %s", key, exc)
            failure = BlockResult(**base, status="failed", output_path=None, qc_path=str(qc_path),
                                  seconds_total=time.perf_counter() - start,
                                  error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            self._write_qc(qc_path, failure, block)
            return failure
        return result

    def _run(self, block: pd.Series, h5_path: Path, qc_path: Path, base: dict,
             start: float) -> BlockResult:

        cfg = self.config
        reports: list[steps.StepReport] = []
        raw = read_raw_block(block)
        raw, rep = steps.apply_montage(raw, cfg.montage)
        reports.append(rep)
        raw, rep = steps.crop(raw, float(block["crop_tmin_s"]), float(block["crop_tmax_s"]))
        reports.append(rep)
        flat_report = steps.flat_recording(raw, cfg.bad_channels.max_bad_fraction)
        reports.append(flat_report)
        if flat_report.facts["flat_recording"]:
            n_flat = flat_report.facts["n_flat_or_nan_channels"]
            result = BlockResult(**base, status="excluded_flat_recording", output_path=None,
                                 qc_path=str(qc_path), seconds_total=time.perf_counter() - start,
                                 exclusion_reason=f"{n_flat} of {len(raw.ch_names)} channels flat or NaN",
                                 facts={r.name: r.facts for r in reports})
            self._write_qc(qc_path, result, block, reports)
            return result
        bads, rep = steps.detect_bad_channels(raw, cfg.bad_channels, cfg.filter)
        reports.append(rep)
        facts = {r.name: r.facts for r in reports}
        if len(bads) > cfg.bad_channels.max_bad_fraction * len(raw.ch_names):
            result = BlockResult(**base, status="excluded_bad_channels", output_path=None,
                                 qc_path=str(qc_path), seconds_total=time.perf_counter() - start,
                                 exclusion_reason=f"{len(bads)} bad channels exceeds "
                                                  f"{cfg.bad_channels.max_bad_fraction:.0%}",
                                 facts=facts)
            self._write_qc(qc_path, result, block, reports)
            return result
        raw, rep = steps.interpolate_and_rereference(raw, bads)
        reports.append(rep)
        ica_copy = raw.copy() if cfg.ica.enabled else None
        raw, rep = steps.filter_and_resample(raw, cfg.filter.l_freq, cfg.filter.h_freq,
                                             cfg.resample_hz, cfg.filter.fir_design)
        reports.append(rep)
        signal_report = steps.signal_quality(raw, exclude=bads)
        reports.append(signal_report)
        if signal_report.facts["degenerate_signal"]:
            facts = {r.name: r.facts for r in reports}
            result = BlockResult(**base, status="excluded_degenerate_signal", output_path=None,
                                 qc_path=str(qc_path), seconds_total=time.perf_counter() - start,
                                 exclusion_reason=(
                                     f"median channel SD {signal_report.facts['channel_sd_uv_median']:.1f} uV "
                                     f"with SD coefficient of variation {signal_report.facts['channel_sd_cv']:.3f}"),
                                 facts=facts)
            self._write_qc(qc_path, result, block, reports)
            return result
        removed_sources = mixing_removed = None
        if cfg.ica.enabled and ica_copy is not None:
            ica_copy, rep = steps.filter_and_resample(
                ica_copy, cfg.filter.ica_l_freq, cfg.filter.ica_h_freq, cfg.resample_hz,
                cfg.filter.fir_design)
            rep.name = "filter_resample_ica_copy"
            reports.append(rep)
            if cfg.filter.ica_notch:
                ica_copy, rep = steps.notch(ica_copy, cfg.filter.notch_freqs)
                rep.name = "notch_ica_copy"
                reports.append(rep)
            (raw, removed_sources, mixing_removed), rep = steps.ica_clean(
                raw, ica_copy, cfg.ica, n_interpolated=len(bads))
            reports.append(rep)
            del ica_copy
        cleaned_report = steps.signal_quality(raw, exclude=bads)
        cleaned_report.name = "signal_quality_cleaned"
        reports.append(cleaned_report)
        facts = {r.name: r.facts for r in reports}
        cleaned = CleanedBlock(
            data=raw.get_data().astype(np.float32), sfreq=float(raw.info["sfreq"]),
            ch_names=list(raw.ch_names), crop_tmin_s=float(facts["crop"]["tmin_s"]),
            block_key=base["block_key"], subject=base["subject"], release=base["release"],
            block=base["block"], config_fingerprint=self.fingerprint,
            removed_sources=removed_sources, mixing_removed=mixing_removed,
            attrs={"source_file": str(block["eeg_path"]), "source_run": block.get("source_run"),
                   "trial_signature": block.get("trial_signature")},
        )
        write_block(cleaned, h5_path, compression=cfg.hdf5_compression)
        result = BlockResult(**base, status="ok", output_path=str(h5_path), qc_path=str(qc_path),
                             seconds_total=time.perf_counter() - start, facts=facts)
        self._write_qc(qc_path, result, block, reports)
        return result

    def _write_qc(self, qc_path: Path, result: BlockResult, block: pd.Series,
                  reports: list[steps.StepReport] | None = None) -> None:
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "config_fingerprint": self.fingerprint,
            "config": self.config.model_dump(mode="json"),
            "versions": _versions(),
            "block": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in block.to_dict().items()},
            "result": {k: v for k, v in asdict(result).items() if k != "facts"},
            "steps": [{"name": r.name, "seconds": round(r.seconds, 3), **r.facts}
                      for r in (reports or [])],
        }
        with open(qc_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, default=str)


class PreprocessingRunner:
    """Process many blocks in parallel, resuming from existing outputs, and aggregate a log."""

    def __init__(self, config: PreprocessingConfig, out_dir: Path) -> None:
        self.config = config
        self.out_dir = out_dir
        self.preprocessor = BlockPreprocessor(config, out_dir)

    def run(self, blocks: pd.DataFrame, n_jobs: int | None = None,
            block_timeout_s: float = 3600.0) -> pd.DataFrame:
        """Process pending blocks in worker processes; a dead or stalled worker fails one block only."""
        from concurrent.futures import TimeoutError as FutureTimeout

        from joblib.externals.loky import get_reusable_executor
        from joblib.externals.loky.process_executor import TerminatedWorkerError

        n_jobs = n_jobs or self.config.n_jobs
        pending = [row for _, row in blocks.iterrows() if not self.preprocessor.is_done(row)]
        _LOG.info("%d blocks requested, %d pending, %d workers", len(blocks), len(pending), n_jobs)
        results: list[BlockResult] = []
        if pending:
            executor = get_reusable_executor(max_workers=n_jobs, timeout=300)
            try:
                futures = {executor.submit(self.preprocessor.process, row): row for row in pending}
                for i, future in enumerate(futures, start=1):
                    row = futures[future]
                    try:
                        result = future.result(timeout=block_timeout_s)
                    except (FutureTimeout, TerminatedWorkerError, Exception) as exc:  # noqa: BLE001
                        _LOG.error("block %s failed at the worker level: %s", row["block_key"], exc)
                        result = BlockResult(
                            block_key=str(row["block_key"]), subject=str(row["subject"]),
                            release=str(row["release"]), block=int(row["block"]), status="failed",
                            output_path=None, qc_path=None, seconds_total=float("nan"),
                            error=f"worker: {type(exc).__name__}: {exc}")
                    results.append(result)
                    if i % 25 == 0 or i == len(futures):
                        _LOG.info("progress %d/%d (%s)", i, len(futures),
                                  pd.Series([r.status for r in results]).value_counts().to_dict())
            finally:
                # loky's reusable executor is a process-wide singleton shared with joblib.Parallel;
                # shutting it down (and killing stalled workers) lets later joblib calls start clean.
                executor.shutdown(wait=True, kill_workers=True)
        done = [self._skipped(row) for _, row in blocks.iterrows()
                if self.preprocessor.is_done(row) and row["block_key"] not in
                {r.block_key for r in results}]
        frame = pd.DataFrame([r.to_row() for r in [*results, *done]])
        if len(frame):
            counts = frame["status"].value_counts().to_dict()
            _LOG.info("preprocessing statuses: %s", counts)
        return frame

    def _skipped(self, row: pd.Series) -> BlockResult:
        h5_path, qc_path = self.preprocessor.output_paths(row)
        return BlockResult(block_key=str(row["block_key"]), subject=str(row["subject"]),
                           release=str(row["release"]), block=int(row["block"]), status="skipped",
                           output_path=str(h5_path), qc_path=str(qc_path), seconds_total=0.0)

    def collect_log(self) -> pd.DataFrame:
        """Rebuild the aggregate log from every QC JSON on disk."""
        rows = []
        for qc_path in sorted(self.out_dir.glob("sub-*/*_qc.json")):
            with open(qc_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            step_facts = {s["name"]: s for s in payload.get("steps", [])}
            row = dict(payload["result"])
            row.update({
                "config_fingerprint": payload["config_fingerprint"],
                "n_bad_channels": step_facts.get("bad_channels", {}).get("n_bad"),
                "bad_channels": ";".join(step_facts.get("bad_channels", {}).get("bad_channels", [])),
                "n_ica_excluded": step_facts.get("ica", {}).get("n_excluded"),
                "artifact_variance_fraction": step_facts.get("ica", {}).get("artifact_variance_fraction"),
                "duration_s": step_facts.get("crop", {}).get("duration_s"),
                "seconds_ica": step_facts.get("ica", {}).get("seconds"),
                "seconds_bad_channels": step_facts.get("bad_channels", {}).get("seconds"),
                "ica_n_iter": step_facts.get("ica", {}).get("n_iter"),
                "ica_label_counts": json.dumps(step_facts.get("ica", {}).get("label_counts")),
                "channel_sd_uv_median": step_facts.get("signal_quality", {}).get("channel_sd_uv_median"),
                "channel_sd_cv": step_facts.get("signal_quality", {}).get("channel_sd_cv"),
                "p2p_uv_max": step_facts.get("signal_quality", {}).get("p2p_uv_max"),
                "degenerate_signal": step_facts.get("signal_quality", {}).get("degenerate_signal"),
            })
            rows.append(row)
        return pd.DataFrame(rows)
