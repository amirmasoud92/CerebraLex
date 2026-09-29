"""Corpus-level quality-control summary from the per-block preprocessing log."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import OnExists, write_dataframe, write_json


def _quantiles(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(values)), "mean": float(values.mean()), "p05": float(q[0.05]),
            "p25": float(q[0.25]), "median": float(q[0.5]), "p75": float(q[0.75]),
            "p95": float(q[0.95]), "max": float(values.max())}


class QCReport:
    """Summaries and exclusion tables derived from ``preprocessing_log`` rows."""

    def __init__(self, log: pd.DataFrame, blocks: pd.DataFrame,
                 trials: pd.DataFrame | None = None, n_channels: int = 129,
                 common_mode: pd.DataFrame | None = None, common_mode_sd_cv_threshold: float = 0.01,
                 amplitude_n_mad: float = 3.0) -> None:
        self.log = log.copy()
        self.blocks = blocks
        self.trials = trials
        self.n_channels = n_channels
        self.common_mode = common_mode
        self.common_mode_sd_cv_threshold = common_mode_sd_cv_threshold
        self.amplitude_n_mad = amplitude_n_mad

    def analysis_table(self) -> pd.DataFrame:
        """Block table with quality-control flags: common-mode recording and analysis usability.

        A block is a common-mode recording when its raw channels have (near-)identical standard
        deviations, i.e. every channel carries the same signal and no differential EEG exists.
        """
        table = self.block_table()
        processed = table["status"].isin(["ok", "skipped"])
        table["common_mode"] = False
        if self.common_mode is not None and len(self.common_mode):
            audit = self.common_mode.set_index("block_key")
            table["raw_sd_cv"] = table["block_key"].map(audit["raw_sd_cv"])
            table["interchannel_r_median"] = table["block_key"].map(audit["interchannel_r_median"])
            table["common_mode"] = (table["raw_sd_cv"] < self.common_mode_sd_cv_threshold).fillna(False)
        table["amplitude_outlier"] = table["block_key"].map(
            self.amplitude_outliers(self.amplitude_n_mad)).fillna(False).astype(bool)
        table["analysis_usable"] = processed & ~table["common_mode"].fillna(False).astype(bool)
        reasons = np.where(~processed, table["status"].fillna("pending"),
                           np.where(table["common_mode"], "excluded_common_mode_recording", ""))
        table["analysis_exclusion"] = reasons
        return table

    def cohort_flow(self) -> dict:
        """Subjects, blocks, targets and valid responses at each stage of the pipeline."""
        table = self.analysis_table()
        usable = table[table["usable"]] if "usable" in table else table
        processed = usable[usable["status"].isin(["ok", "skipped"])]
        ok = usable[usable["analysis_usable"]]
        flow = {
            "usable_blocks": int(len(usable)), "usable_subjects": int(usable["subject"].nunique()),
            "preprocessed_blocks": int(len(processed)),
            "preprocessed_subjects": int(processed["subject"].nunique()),
            "common_mode_blocks": int(usable["common_mode"].sum()),
            "common_mode_subjects": int(usable.loc[usable["common_mode"], "subject"].nunique()),
            "analysis_blocks": int(len(ok)), "analysis_subjects": int(ok["subject"].nunique()),
            "subjects_lost_entirely": int(usable["subject"].nunique() - ok["subject"].nunique()),
        }
        if self.trials is not None and len(self.trials):
            kept = self.trials[self.trials["block_key"].isin(set(ok["block_key"]))]
            per_subject = kept.groupby("subject")["valid_response"].sum()
            flow.update({
                "targets_in_preprocessed_blocks": int(len(kept)),
                "valid_responses_in_preprocessed_blocks": int(kept["valid_response"].sum()),
                "subjects_with_at_least_10_valid": int((per_subject >= 10).sum()),
                "valid_responses_per_subject": _quantiles(per_subject),
            })
        return flow

    def amplitude_outliers(self, n_mad: float = 3.0) -> pd.Series:
        """Blocks whose log10 median channel SD lies more than ``n_mad`` scaled MADs from the
        corpus median (Leys et al. 2013); used for a sensitivity analysis only."""
        table = self.block_table()
        sd = pd.to_numeric(table["channel_sd_uv_median"], errors="coerce")
        log_sd = np.log10(sd.where(sd > 0))
        centre = np.nanmedian(log_sd)
        mad = 1.4826 * np.nanmedian(np.abs(log_sd - centre))
        flag = (np.abs(log_sd - centre) > n_mad * mad) & table["status"].isin(["ok", "skipped"])
        return pd.Series(flag.to_numpy(), index=table["block_key"].to_numpy(), name="amplitude_outlier")

    def bad_channel_sensitivity(self) -> dict:
        """Blocks that would be excluded under the MADE 10% rule and a lenient 30% rule."""
        table = self.block_table()
        detected = pd.to_numeric(table["n_bad_channels"], errors="coerce").dropna()
        out = {}
        for fraction in (0.10, 0.20, 0.30):
            limit = fraction * self.n_channels
            out[f"blocks_over_{int(fraction * 100)}pct"] = int((detected > limit).sum())
        out["blocks_with_detection"] = int(len(detected))
        return out

    def block_table(self) -> pd.DataFrame:
        table = self.blocks.merge(
            self.log[["block_key", "status", "exclusion_reason", "n_bad_channels",
                      "n_ica_excluded", "artifact_variance_fraction", "channel_sd_uv_median",
                      "degenerate_signal", "ica_n_iter", "seconds_total"]],
            on="block_key", how="left")
        table["processed"] = table["status"].isin(["ok", "skipped"])
        table["ica_hit_cap"] = pd.to_numeric(table["ica_n_iter"], errors="coerce") >= 500
        return table

    def summary(self) -> dict:
        table = self.block_table()
        usable = table[table["usable"]] if "usable" in table else table
        processed = usable[usable["processed"]]
        by_status = usable["status"].fillna("pending").value_counts().to_dict()
        by_release = (usable.assign(status=usable["status"].fillna("pending"))
                      .groupby(["release", "status"]).size().unstack(fill_value=0))
        return {
            "n_usable_blocks": int(len(usable)),
            "status_counts": by_status,
            "status_by_release": by_release.to_dict(orient="index"),
            "subjects_with_all_blocks_ok": int(
                processed.groupby("subject").size().eq(usable.groupby("subject").size()).sum()),
            "n_subjects_processed": int(processed["subject"].nunique()),
            "bad_channels": _quantiles(processed["n_bad_channels"]),
            "ica_excluded": _quantiles(processed["n_ica_excluded"]),
            "artifact_variance_fraction": _quantiles(processed["artifact_variance_fraction"]),
            "channel_sd_uv_median": _quantiles(processed["channel_sd_uv_median"]),
            "seconds_total": _quantiles(processed["seconds_total"]),
            "n_ica_hit_iteration_cap": int(processed["ica_hit_cap"].fillna(False).sum()),
            "n_degenerate_excluded": int((usable["status"] == "excluded_degenerate_signal").sum()),
            "n_flat_recording_excluded": int((usable["status"] == "excluded_flat_recording").sum()),
            "n_bad_channel_excluded": int((usable["status"] == "excluded_bad_channels").sum()),
            "n_failed": int((usable["status"] == "failed").sum()),
            "n_pending": int(usable["status"].isna().sum()),
            "bad_channel_sensitivity": self.bad_channel_sensitivity(),
            "n_amplitude_outlier_blocks": int(self.amplitude_outliers(self.amplitude_n_mad).sum()),
            "amplitude_outlier_n_mad": self.amplitude_n_mad,
            "common_mode_sd_cv_threshold": self.common_mode_sd_cv_threshold,
            "common_mode_audited": self.common_mode is not None,
            "cohort_flow": self.cohort_flow(),
            "exclusion_reasons": usable.loc[usable["exclusion_reason"].notna(), "exclusion_reason"]
            .str.slice(0, 60).value_counts().head(20).to_dict(),
        }

    def write(self, out_dir: Path, on_exists: OnExists = "overwrite") -> dict[str, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        table = self.analysis_table()
        paths = {
            "block_qc": write_dataframe(table, out_dir / "block_qc.parquet", on_exists),
            "block_qc_csv": write_dataframe(table, out_dir / "block_qc.csv", on_exists),
            "summary": write_json(self.summary(), out_dir / "qc_summary.json", on_exists),
        }
        return paths


def load_preprocessing_log(preproc_dir: Path) -> pd.DataFrame:
    """Read the aggregate log if present, else rebuild it from the QC JSON files."""
    parquet = preproc_dir / "preprocessing_log.parquet"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    rows = []
    for qc_path in sorted(preproc_dir.glob("sub-*/*_qc.json")):
        with open(qc_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        steps = {s["name"]: s for s in payload.get("steps", [])}
        row = dict(payload["result"])
        row.update({
            "n_bad_channels": steps.get("bad_channels", {}).get("n_bad"),
            "n_ica_excluded": steps.get("ica", {}).get("n_excluded"),
            "artifact_variance_fraction": steps.get("ica", {}).get("artifact_variance_fraction"),
            "channel_sd_uv_median": steps.get("signal_quality", {}).get("channel_sd_uv_median"),
            "degenerate_signal": steps.get("signal_quality", {}).get("degenerate_signal"),
            "ica_n_iter": steps.get("ica", {}).get("n_iter"),
        })
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["block_key", "status"])
