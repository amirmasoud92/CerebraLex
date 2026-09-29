"""Trial extraction from the HED-annotated events.tsv of the Contrast Change Detection task.

Event vocabulary, verified from every local CCD events.tsv (audit/event_vocabulary.csv):

* ``contrastChangeB{n}_start`` marks a block.
* ``contrastTrial_start`` marks the onset of the two flickering gratings.
* ``left_target`` / ``right_target`` mark the contrast-change onset, the stimulus to detect.
* ``left_buttonPress`` / ``right_buttonPress`` are responses. Their ``feedback`` column is
  ``smiley_face`` (correct), ``sad_face`` (incorrect) or ``non_target`` (press outside the
  response window: before the target, or after the trial closed).
* ``break cnt``, ``9999``, ``video_start``, ``video_stop``, ``fixpoint_ON`` are recording or
  session markers outside the trial structure.
* ``boundary`` marks a discontinuity in the EEG record (EEGLAB convention).

Anything outside this vocabulary is counted in ``unknown_values`` and never silently dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRIAL_START = "contrastTrial_start"
TARGET_SIDES = {"left_target": "left", "right_target": "right"}
RESPONSE_SIDES = {"left_buttonPress": "left", "right_buttonPress": "right"}
BLOCK_START_RE = re.compile(r"^contrastChangeB(?P<block>\d+)_start$")
BOUNDARY = "boundary"
SESSION_MARKERS = frozenset({"break cnt", "9999", "video_start", "video_stop", "fixpoint_ON"})
VALID_FEEDBACK = frozenset({"smiley_face", "sad_face"})
NON_TARGET_FEEDBACK = "non_target"

TRIAL_COLUMNS = (
    "trial_index", "block", "trial_start_onset", "target_onset", "target_side",
    "foreperiod_s", "response_onset", "response_side", "rt_s", "n_responses",
    "n_premature_responses", "n_late_responses", "correct", "feedback",
    "feedback_consistent", "next_trial_start_onset", "spans_boundary", "flags",
)


@dataclass
class _OpenTrial:
    trial_index: int
    block: int | None
    start_onset: float
    target_onset: float | None = None
    target_side: str | None = None
    responses: list[tuple[float, str, str | None]] = field(default_factory=list)
    n_premature: int = 0
    n_late: int = 0
    spans_boundary: bool = False
    flags: list[str] = field(default_factory=list)

    def close(self, next_start: float | None) -> dict:
        first = self.responses[0] if self.responses else None
        rt = None
        correct: bool | None = None
        consistent: bool | None = None
        if first is not None and self.target_onset is not None:
            rt = first[0] - self.target_onset
            correct = first[1] == self.target_side
            if first[2] in VALID_FEEDBACK:
                consistent = (first[2] == "smiley_face") == correct
        if self.target_onset is None:
            self.flags.append("no_target")
        elif not self.responses:
            self.flags.append("no_response")
        if len(self.responses) > 1:
            self.flags.append("multiple_responses")
        if self.n_premature:
            self.flags.append("premature_response")
        if self.n_late:
            self.flags.append("late_response")
        if consistent is False:
            self.flags.append("feedback_inconsistent")
        if self.spans_boundary:
            self.flags.append("spans_boundary")
        return {
            "trial_index": self.trial_index,
            "block": self.block,
            "trial_start_onset": self.start_onset,
            "target_onset": self.target_onset,
            "target_side": self.target_side,
            "foreperiod_s": None if self.target_onset is None else self.target_onset - self.start_onset,
            "response_onset": None if first is None else first[0],
            "response_side": None if first is None else first[1],
            "rt_s": rt,
            "n_responses": len(self.responses),
            "n_premature_responses": self.n_premature,
            "n_late_responses": self.n_late,
            "correct": correct,
            "feedback": None if first is None else first[2],
            "feedback_consistent": consistent,
            "next_trial_start_onset": next_start,
            "spans_boundary": self.spans_boundary,
            "flags": ";".join(self.flags),
        }


class CCDTrialParser:
    """State machine turning an events table into one row per trial.

    A press after the target with ``smiley_face``/``sad_face`` feedback is a valid response;
    the first such press defines the reaction time. Presses before the target are premature;
    presses after the target flagged ``non_target`` by the experiment are late and excluded
    from reaction-time computation. A trial closes at the next ``contrastTrial_start`` or at
    any session marker (block end, task end, video segment), so discontinuities recorded
    after the task are never attributed to a trial.
    """

    def __init__(self) -> None:
        self.unknown_values: Counter[str] = Counter()
        self.orphan_events: Counter[str] = Counter()
        self.marker_counts: Counter[str] = Counter()
        self.n_boundaries = 0

    def parse(self, events: pd.DataFrame) -> pd.DataFrame:
        required = {"onset", "value"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"events table lacks columns {sorted(missing)}")
        frame = events.sort_values("onset", kind="stable").reset_index(drop=True)
        has_feedback = "feedback" in frame.columns
        rows: list[dict] = []
        current: _OpenTrial | None = None
        block: int | None = None
        for row in frame.itertuples(index=False):
            value = str(row.value)
            onset = float(row.onset)
            block_match = BLOCK_START_RE.match(value)
            if block_match:
                block = int(block_match["block"])
                self.marker_counts[value] += 1
                continue
            if value == BOUNDARY:
                self.n_boundaries += 1
                if current is not None:
                    current.spans_boundary = True
                continue
            if value in SESSION_MARKERS:
                self.marker_counts[value] += 1
                if current is not None:
                    rows.append(current.close(None))
                    current = None
                continue
            if value == TRIAL_START:
                if current is not None:
                    rows.append(current.close(onset))
                current = _OpenTrial(trial_index=len(rows), block=block, start_onset=onset)
                continue
            if value in TARGET_SIDES:
                if current is None:
                    self.orphan_events[value] += 1
                    continue
                if current.target_onset is not None:
                    current.flags.append("multiple_targets")
                    continue
                current.target_onset = onset
                current.target_side = TARGET_SIDES[value]
                continue
            if value in RESPONSE_SIDES:
                if current is None:
                    self.orphan_events[value] += 1
                    continue
                feedback = getattr(row, "feedback", None) if has_feedback else None
                feedback = None if feedback is None or pd.isna(feedback) else str(feedback)
                if current.target_onset is None:
                    current.n_premature += 1
                elif feedback == NON_TARGET_FEEDBACK:
                    current.n_late += 1
                else:
                    current.responses.append((onset, RESPONSE_SIDES[value], feedback))
                continue
            self.unknown_values[value] += 1
        if current is not None:
            rows.append(current.close(None))
        return pd.DataFrame(rows, columns=list(TRIAL_COLUMNS))


def summarize_ccd_trials(trials: pd.DataFrame) -> dict[str, float | int | None]:
    """Per-run behavioural summary used by the audit and the benchmark cohort filter."""
    n_trials = int(len(trials))
    with_target = trials["target_onset"].notna()
    responded = with_target & trials["response_onset"].notna()
    rt = pd.to_numeric(trials.loc[responded, "rt_s"], errors="coerce")
    correct = trials.loc[responded, "correct"].astype("boolean")
    consistent = trials.loc[responded, "feedback_consistent"].astype("boolean")
    n_responded = int(responded.sum())
    return {
        "n_trials": n_trials,
        "n_with_target": int(with_target.sum()),
        "n_responded": n_responded,
        "n_correct": int(correct.fillna(False).sum()),
        "n_incorrect": int((~correct.fillna(True)).sum()),
        "n_miss": int((with_target & ~responded).sum()),
        "n_premature_trials": int((trials["n_premature_responses"] > 0).sum()),
        "n_late_trials": int((trials["n_late_responses"] > 0).sum()),
        "n_multiple_responses": int((trials["n_responses"] > 1).sum()),
        "n_feedback_inconsistent": int((consistent == False).sum()),  # noqa: E712
        "n_spanning_boundary": int(trials["spans_boundary"].fillna(False).astype(bool).sum()),
        "accuracy": float(correct.fillna(False).mean()) if n_responded else None,
        "median_rt_s": float(np.nanmedian(rt)) if n_responded else None,
        "mean_rt_s": float(np.nanmean(rt)) if n_responded else None,
        "min_rt_s": float(np.nanmin(rt)) if n_responded else None,
        "max_rt_s": float(np.nanmax(rt)) if n_responded else None,
        "n_blocks": int(trials["block"].dropna().nunique()),
        "median_foreperiod_s": (
            float(np.nanmedian(pd.to_numeric(trials["foreperiod_s"], errors="coerce")))
            if with_target.any() else None
        ),
    }
