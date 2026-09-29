"""MIPDB (Langer et al. 2017, Scientific Data 4:170040) for the external validation.

MIPDB recorded the contrast change detection paradigm (``SAIIT_2AFC``, three blocks of 24 targets)
with the same EGI 128-channel nets as HBN, as numbered raw EGI recordings per participant. This
module finds the CCD recordings by their event codes, translates the codes to the HBN event
vocabulary so that the HBN trial parser (``task_and_trials.events``) applies unchanged, computes the
HBN crop window, and builds the participant table.

Codes (MIPDB README and behavioural files): 5 trial start (flicker; the interval to the target is
the task's foreperiod of 2.8, 4.4 or 6.0 s), 8 and 9 left and right target, 12 and 13 left and right
button press, 94, 95 and 96 the start of blocks 1, 2 and 3. MIPDB records no task feedback: a press
more than ``LATE_AFTER_S`` after the target is late (HBN's non-target feedback), all other presses
after the target are responses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from task_and_trials.blocks import (
    DEFAULT_POST_MARGIN_S,
    DEFAULT_PRE_MARGIN_S,
    CCDBlock,
    CCDBlockResolver,
    trial_signature,
)
from task_and_trials.events import NON_TARGET_FEEDBACK, CCDTrialParser

_LOG = logging.getLogger(__name__)
RELEASE = "MIPDB"
CODE_TO_VALUE = {"5": "contrastTrial_start", "8": "left_target", "9": "right_target",
                 "12": "left_buttonPress", "13": "right_buttonPress",
                 "94": "contrastChangeB1_start", "95": "contrastChangeB2_start", "96": "contrastChangeB3_start"}
PRESS_VALUES = ("left_buttonPress", "right_buttonPress")
TARGET_VALUES = ("left_target", "right_target")
LATE_AFTER_S = 2.5          # longest valid HBN response: 2.498 s
REFERENCE_CHANNEL = "E129"  # flat vertex reference of the EGI recordings; HBN names it Cz


def read_annotations(raw_path: Path) -> pd.DataFrame:
    """Event onsets (s) and stripped codes of one raw EGI recording."""
    import mne

    raw = mne.io.read_raw_egi(raw_path, preload=False, verbose="error")
    ann = raw.annotations
    return pd.DataFrame({"onset": np.asarray(ann.onset, dtype=float),
                         "code": [str(d).strip() for d in ann.description]})


def translate_events(codes: pd.DataFrame) -> pd.DataFrame:
    """HBN-vocabulary events table (onset, value, feedback) from MIPDB codes.

    Unknown codes keep their code as value, so the HBN parser counts them instead of dropping them.
    """
    frame = codes.sort_values("onset", kind="stable").reset_index(drop=True)
    values = frame["code"].map(CODE_TO_VALUE).fillna(frame["code"])
    feedback: list[str | None] = []
    last_start = last_target = None
    for onset, value in zip(frame["onset"], values, strict=True):
        if value == "contrastTrial_start":
            last_start, last_target = onset, None
        elif value in TARGET_VALUES and last_start is not None:
            last_target = onset
        late = value in PRESS_VALUES and last_target is not None and onset - last_target > LATE_AFTER_S
        feedback.append(NON_TARGET_FEEDBACK if late else None)
    return pd.DataFrame({"onset": frame["onset"], "value": values, "feedback": feedback})


def is_ccd(codes: pd.DataFrame) -> bool:
    present = set(codes["code"])
    return {"5", "12", "13"} <= present and bool({"8", "9"} & present) and bool({"94", "95", "96"} & present)


def ccd_blocks(subject: str, raw_path: Path, events: pd.DataFrame) -> list[CCDBlock]:
    """Blocks of one CCD recording with the HBN crop rule (CCDBlockResolver)."""
    trials = CCDTrialParser().parse(events)
    if trials.empty:
        return []
    starts = {v: o for v, o in zip(events["value"], events["onset"], strict=True) if str(v).startswith("contrastChangeB")}
    run = int(raw_path.stem[-3:])
    blocks = []
    for block_id, group in trials.groupby(trials["block"].fillna(-1).astype(int), sort=True):
        group = group.sort_values("trial_start_onset")
        first_onset = float(group["trial_start_onset"].min())
        marker = starts.get(f"contrastChangeB{block_id}_start", first_onset)
        block_start = min(marker, first_onset)
        last_end = CCDBlockResolver._last_trial_end(group)
        responded = group["response_onset"].notna() & group["target_onset"].notna()
        blocks.append(CCDBlock(
            subject=subject, release=RELEASE, block=int(block_id), source_run=run,
            run_key=f"{subject}_{RELEASE}_contrastChangeDetection_run-{run}", eeg_path=str(raw_path),
            trial_signature=trial_signature(group), n_trials=int(len(group)),
            n_targets=int(group["target_onset"].notna().sum()), n_valid_responses=int(responded.sum()),
            block_start_onset_s=block_start, first_trial_onset_s=first_onset, last_trial_end_s=last_end,
            crop_tmin_s=max(0.0, block_start - DEFAULT_PRE_MARGIN_S), crop_tmax_s=last_end + DEFAULT_POST_MARGIN_S,
            n_boundaries_inside_crop=0, n_trials_spanning_boundary=0))
    return blocks


def find_ccd_recordings(data_root: Path, subject: str) -> list[Path]:
    """Raw recordings whose events contain the CCD codes.

    The authors' event exports (``EEG/raw/csv_format/<file>_events.csv``) are read when present, to
    avoid opening every raw file; otherwise the raw file's own annotations.
    """
    raw_dir = data_root / subject / "EEG" / "raw" / "raw_format"
    csv_dir = data_root / subject / "EEG" / "raw" / "csv_format"
    found = []
    for raw_path in sorted(raw_dir.glob(f"{subject}[0-9][0-9][0-9].raw")):
        export = csv_dir / f"{raw_path.stem}_events.csv"
        if export.is_file():
            codes = pd.DataFrame({"code": pd.read_csv(export)["type"].astype(str).str.strip()})
        else:
            codes = read_annotations(raw_path)
        if is_ccd(codes):
            found.append(raw_path)
    return found


def build_manifest(data_root: Path, subjects: list[str]) -> pd.DataFrame:
    """One row per CCD block of the given participants, in the HBN manifest schema plus ``reader``.

    Crop windows come from each raw recording's own annotations (the primary source). Repeated and
    restarted blocks are resolved and ``usable`` is defined exactly as for HBN
    (``CCDBlockResolver.resolve_subject`` and ``ManifestBuilder.block_manifest``).
    """
    resolver = CCDBlockResolver()
    rows = []
    for subject in subjects:
        blocks = [block for raw_path in find_ccd_recordings(data_root, subject)
                  for block in ccd_blocks(subject, raw_path, translate_events(read_annotations(raw_path)))]
        rows.extend(block.to_dict() for block in resolver.resolve_subject(blocks))
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["reader"] = "egi"
    manifest["usable"] = (manifest["canonical"] & (manifest["n_boundaries_inside_crop"] == 0)
                          & (manifest["block"] >= 1) & (manifest["n_targets"] > 0))
    _LOG.info("MIPDB manifest: %d CCD blocks of %d participants (%d duplicates, %d restarts, %d usable)",
              len(manifest), manifest["subject"].nunique(), int(manifest["is_duplicate"].sum()),
              int(manifest["is_restart"].sum()), int(manifest["usable"].sum()))
    return manifest


def participants(public_file: Path, hbn_median_ehq: float, max_age: float = 21.0) -> pd.DataFrame:
    """Participant table in the HBN demographics schema (subject, age, sex, ehq_total, release)."""
    frame = pd.read_csv(public_file)
    out = pd.DataFrame({"subject": frame["ID"].astype(str),
                        "age": pd.to_numeric(frame["Age"], errors="coerce"),
                        "sex": frame["Sex"].map({1: "M", 2: "F"}),   # data dictionary DEM_002
                        "ehq_total": float(hbn_median_ehq),           # constant: MIPDB scales differ
                        "dx_status": frame.get("DX_Status"),
                        "release": RELEASE})
    return out[out["age"] <= max_age].reset_index(drop=True)
