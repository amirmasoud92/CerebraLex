"""Trial table for canonical CCD blocks, aligned to the cleaned-signal sample grid."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from task_and_trials.events import CCDTrialParser
from task_and_trials.recording import _read_tsv

_LOG = logging.getLogger(__name__)
LAPSE_RT_FACTOR = 1.5
RESPONSE_WINDOW_S = 2.4


class TrialTableBuilder:
    """Turn the block manifest into one row per target with sample indices and behaviour."""

    def __init__(self, sfreq: float, lapse_rt_factor: float = LAPSE_RT_FACTOR) -> None:
        self.sfreq = sfreq
        self.lapse_rt_factor = lapse_rt_factor

    def build(self, blocks: pd.DataFrame) -> pd.DataFrame:
        usable = blocks[blocks["usable"]] if "usable" in blocks else blocks[blocks["canonical"]]
        frames = [self._block_trials(row) for _, row in usable.iterrows()]
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return pd.DataFrame()
        table = pd.concat(frames, ignore_index=True)
        table = self._add_subject_level(table)
        _LOG.info("trial table: %d trials, %d subjects, %d blocks", len(table),
                  table["subject"].nunique(), table["block_key"].nunique())
        return table

    def _block_trials(self, block: pd.Series) -> pd.DataFrame | None:
        if str(block.get("reader", "eeglab")) == "egi":                   # MIPDB
            from task_and_trials.mipdb import read_annotations, translate_events

            events = translate_events(read_annotations(Path(str(block["eeg_path"]))))
        else:
            events_path = Path(str(block["eeg_path"]).replace("_eeg.set", "_events.tsv"))
            if not events_path.is_file():
                _LOG.warning("events missing for %s", block["block_key"])
                return None
            events = _read_tsv(events_path)
        trials = CCDTrialParser().parse(events)
        trials = trials[trials["block"] == int(block["block"])].copy()
        trials = trials[trials["target_onset"].notna()].reset_index(drop=True)
        if trials.empty:
            return None
        tmin = float(block["crop_tmin_s"])
        n_times = int(round((float(block["crop_tmax_s"]) - tmin) * self.sfreq))
        out = pd.DataFrame({
            "block_key": block["block_key"],
            "subject": block["subject"],
            "release": block["release"],
            "block": int(block["block"]),
            "source_run": block.get("source_run"),
            "trial_in_block": np.arange(len(trials)),
            "trial_start_onset_s": trials["trial_start_onset"].astype(float),
            "target_onset_s": trials["target_onset"].astype(float),
            "target_side": trials["target_side"],
            "foreperiod_s": trials["foreperiod_s"].astype(float),
            "response_onset_s": pd.to_numeric(trials["response_onset"], errors="coerce"),
            "response_side": trials["response_side"],
            "rt_s": pd.to_numeric(trials["rt_s"], errors="coerce"),
            "correct": trials["correct"].astype("boolean"),
            "feedback": trials["feedback"],
            "n_premature": trials["n_premature_responses"].astype(int),
            "n_late": trials["n_late_responses"].astype(int),
            "spans_boundary": trials["spans_boundary"].astype(bool),
        })
        out["valid_response"] = out["rt_s"].notna() & (out["rt_s"] > 0)
        out["miss"] = ~out["valid_response"]
        out["target_sample"] = np.round((out["target_onset_s"] - tmin) * self.sfreq).astype(int)
        out["trial_start_sample"] = np.round((out["trial_start_onset_s"] - tmin) * self.sfreq).astype(int)
        out["response_sample"] = np.round((out["response_onset_s"] - tmin) * self.sfreq)
        out["block_n_times"] = n_times
        out["boundary_onsets_s"] = str(block.get("boundary_onsets_s", "") or "")
        out["time_in_block_s"] = out["target_onset_s"] - float(block["block_start_onset_s"])
        out["prev_correct"] = out["correct"].shift(1)
        out["prev_miss"] = out["miss"].shift(1)
        out["prev_rt_s"] = out["rt_s"].shift(1)
        return out

    def _add_subject_level(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.sort_values(["subject", "block", "trial_in_block"]).reset_index(drop=True)
        valid = table[table["valid_response"]]
        medians = valid.groupby("subject")["rt_s"].median().rename("subject_median_rt_s")
        table = table.merge(medians, on="subject", how="left")
        table["log_rt"] = np.log(table["rt_s"])
        table["log_rt_centred"] = table["log_rt"] - np.log(table["subject_median_rt_s"])
        table["lapse"] = table["miss"] | (
            table["rt_s"] > self.lapse_rt_factor * table["subject_median_rt_s"]
        )
        block_order = (table.drop_duplicates(["subject", "block"])
                       .sort_values(["subject", "block"])
                       .assign(block_order=lambda f: f.groupby("subject").cumcount())
                       [["subject", "block", "block_order"]])
        table = table.merge(block_order, on=["subject", "block"], how="left")
        table["trial_index"] = table.groupby("subject").cumcount()
        table["n_valid_responses_subject"] = table.groupby("subject")["valid_response"].transform("sum")
        return table

    @staticmethod
    def window_valid(table: pd.DataFrame, tmin_s: float, tmax_s: float, sfreq: float) -> pd.Series:
        """True when [target + tmin, target + tmax] lies inside the block and contains no boundary."""
        start = table["target_sample"] + int(round(tmin_s * sfreq))
        stop = table["target_sample"] + int(round(tmax_s * sfreq))
        inside = (start >= 0) & (stop <= table["block_n_times"])
        window_lo = table["target_onset_s"] + tmin_s
        window_hi = table["target_onset_s"] + tmax_s
        onsets = table.get("boundary_onsets_s", pd.Series("", index=table.index)).fillna("")
        crosses = np.zeros(len(table), dtype=bool)
        for i, (cell, lo, hi) in enumerate(zip(onsets, window_lo, window_hi, strict=True)):
            if cell:
                crosses[i] = any(lo <= float(b) <= hi for b in str(cell).split(";") if b)
        return inside & ~pd.Series(crosses, index=table.index)
