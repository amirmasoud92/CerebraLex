"""Resolve unique Contrast Change Detection blocks per subject from possibly redundant run files.

The audit found that early releases distribute block 3 twice (inside the run-2 file and again as
the run-3 file) and that some run-1 files contain an aborted block 2 followed by a restart in the
run-2 file. A block is identified by its trial signature (target sides and reaction times), so
identical signatures are duplicates and differing signatures with the same block index are
restarts. Each resolved block also carries crop boundaries that exclude appended movie segments.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from task_and_trials.bids_index import RunRecord
from task_and_trials.events import BLOCK_START_RE, BOUNDARY, CCDTrialParser
from task_and_trials.recording import EEGRecording

_LOG = logging.getLogger(__name__)
RESPONSE_WINDOW_S = 2.4
DEFAULT_PRE_MARGIN_S = 1.0
DEFAULT_POST_MARGIN_S = 3.0


@dataclass(frozen=True)
class CCDBlock:
    subject: str
    release: str
    block: int
    source_run: int | None
    run_key: str
    eeg_path: str
    trial_signature: str
    n_trials: int
    n_targets: int
    n_valid_responses: int
    block_start_onset_s: float
    first_trial_onset_s: float
    last_trial_end_s: float
    crop_tmin_s: float
    crop_tmax_s: float
    n_boundaries_inside_crop: int
    n_trials_spanning_boundary: int
    boundary_onsets_s: str = ""
    is_duplicate: bool = False
    duplicate_of: str | None = None
    is_restart: bool = False
    canonical: bool = True

    @property
    def block_key(self) -> str:
        return f"{self.subject}_{self.release}_block-{self.block}_{self.run_key}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["block_key"] = self.block_key
        return payload


def trial_signature(trials: pd.DataFrame) -> str:
    """Hash of the target-side and reaction-time sequence of a block."""
    parts = []
    for row in trials.itertuples(index=False):
        side = "" if row.target_side is None or pd.isna(row.target_side) else str(row.target_side)
        rt = "" if row.rt_s is None or pd.isna(row.rt_s) else f"{float(row.rt_s):.3f}"
        parts.append(f"{side}:{rt}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class CCDBlockResolver:
    """Split run files into blocks, compute crop windows and mark duplicates and restarts."""

    def __init__(
        self,
        pre_margin_s: float = DEFAULT_PRE_MARGIN_S,
        post_margin_s: float = DEFAULT_POST_MARGIN_S,
    ) -> None:
        self.pre_margin_s = pre_margin_s
        self.post_margin_s = post_margin_s

    def resolve_run(self, record: RunRecord) -> list[CCDBlock]:
        recording = EEGRecording(record)
        events = recording.events
        parser = CCDTrialParser()
        trials = parser.parse(events)
        if trials.empty:
            return []
        values = events["value"].astype(str)
        onsets = pd.to_numeric(events["onset"], errors="coerce")
        block_marker_onsets = {
            int(BLOCK_START_RE.match(v)["block"]): float(o)
            for v, o in zip(values, onsets, strict=True) if BLOCK_START_RE.match(v)
        }
        boundary_onsets = [float(o) for v, o in zip(values, onsets, strict=True) if v == BOUNDARY]
        blocks: list[CCDBlock] = []
        for block_id, group in trials.groupby(trials["block"].fillna(-1).astype(int), sort=True):
            group = group.sort_values("trial_start_onset")
            first_onset = float(group["trial_start_onset"].min())
            marker = block_marker_onsets.get(int(block_id), first_onset)
            block_start = min(marker, first_onset)
            last_end = self._last_trial_end(group)
            tmin = max(0.0, block_start - self.pre_margin_s)
            tmax = last_end + self.post_margin_s
            inside = [b for b in boundary_onsets if tmin <= b <= tmax]
            responded = group["response_onset"].notna() & group["target_onset"].notna()
            blocks.append(CCDBlock(
                subject=record.subject,
                release=record.release,
                block=int(block_id),
                source_run=record.run,
                run_key=record.run_key,
                eeg_path=str(record.eeg_path),
                trial_signature=trial_signature(group),
                n_trials=int(len(group)),
                n_targets=int(group["target_onset"].notna().sum()),
                n_valid_responses=int(responded.sum()),
                block_start_onset_s=block_start,
                first_trial_onset_s=first_onset,
                last_trial_end_s=last_end,
                crop_tmin_s=tmin,
                crop_tmax_s=tmax,
                n_boundaries_inside_crop=len(inside),
                n_trials_spanning_boundary=int(group["spans_boundary"].astype(bool).sum()),
                boundary_onsets_s=";".join(f"{b:.3f}" for b in boundary_onsets),
            ))
        return blocks

    @staticmethod
    def _last_trial_end(group: pd.DataFrame) -> float:
        last = group.iloc[-1]
        next_start = last["next_trial_start_onset"]
        if next_start is not None and not pd.isna(next_start):
            return float(next_start)
        target = last["target_onset"]
        if target is not None and not pd.isna(target):
            return float(target) + RESPONSE_WINDOW_S
        return float(last["trial_start_onset"]) + RESPONSE_WINDOW_S

    def resolve_subject(self, blocks: Sequence[CCDBlock]) -> list[CCDBlock]:
        """Mark duplicate and restarted blocks for one subject; canonical blocks keep the data."""
        resolved: list[CCDBlock] = []
        by_block: dict[int, list[CCDBlock]] = {}
        for block in blocks:
            by_block.setdefault(block.block, []).append(block)
        for block_id, candidates in sorted(by_block.items()):
            by_signature: dict[str, list[CCDBlock]] = {}
            for candidate in sorted(candidates, key=lambda b: (b.source_run or 0)):
                by_signature.setdefault(candidate.trial_signature, []).append(candidate)
            groups = list(by_signature.values())
            complete = max(groups, key=lambda g: g[0].n_targets)
            for group in groups:
                keeper = self._prefer(group, block_id)
                for candidate in group:
                    is_dup = candidate is not keeper
                    is_restart = group is not complete
                    resolved.append(CCDBlock(
                        **{**asdict(candidate),
                           "is_duplicate": is_dup,
                           "duplicate_of": keeper.run_key if is_dup else None,
                           "is_restart": is_restart,
                           "canonical": (not is_dup) and (not is_restart)}
                    ))
        return resolved

    @staticmethod
    def _prefer(group: Sequence[CCDBlock], block_id: int) -> CCDBlock:
        matching = [b for b in group if b.source_run == block_id]
        return matching[0] if matching else group[0]

    def resolve_records(self, records: Iterable[RunRecord]) -> pd.DataFrame:
        per_subject: dict[tuple[str, str], list[CCDBlock]] = {}
        for record in records:
            if record.task != "contrastChangeDetection":
                continue
            try:
                for block in self.resolve_run(record):
                    per_subject.setdefault((record.subject, record.release), []).append(block)
            except Exception as exc:  # noqa: BLE001 - keep going, report in the table
                _LOG.warning("block resolution failed for %s: %s", record.run_key, exc)
        rows = []
        for key in sorted(per_subject):
            rows.extend(b.to_dict() for b in self.resolve_subject(per_subject[key]))
        frame = pd.DataFrame(rows)
        if len(frame):
            _LOG.info(
                "resolved %d blocks for %d subjects: %d canonical, %d duplicates, %d restarts",
                len(frame), frame["subject"].nunique(), int(frame["canonical"].sum()),
                int(frame["is_duplicate"].sum()), int(frame["is_restart"].sum()),
            )
        return frame
