"""Mechanical evaluation of the benchmark criteria.

Criterion 1 (pipeline sanity, post-target window):
the best classical model (spectral, spectral plus aperiodic, Riemannian, time-course) must recover
trial-level RT information: it beats the demographics-only model in cross-validation with the
paired bootstrap interval below zero, and its within-child correlation interval lies above 0.10.
The original criterion (nRMSE at most 0.95 with the interval below 1.0) is reported beside it; an
absolute threshold mixes pipeline quality with the share of RT variance that is trial-level noise.
Criterion 2 (first H1 result, pre-stimulus window): at least one EEG model has a lower RT nRMSE
than the demographics-only model in cross-validation with the paired interval below zero.
Criterion 3 (leakage) holds when every run completed its leakage assertions. Lapse (AUROC) and the
R5 hold-out are reported beside the criteria but decide none of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CLASSICAL_MODELS = ("spectral", "spectral_aperiodic", "riemann", "timecourse")
EEG_MODELS = ("spectral", "spectral_aperiodic", "riemann", "timecourse", "eegnet")
ORIGINAL_THRESHOLD = 0.95
DEBUG_THRESHOLD = 0.98
WITHIN_R_THRESHOLD = 0.10
_KEYS = ["window", "target", "scope", "model"]


@dataclass(frozen=True)
class GateResult:
    criterion: str
    passed: bool
    detail: str


def deduplicate(table: pd.DataFrame, keys: list[str], value: str) -> pd.DataFrame:
    """Drop rows repeated across combined runs after checking that repeats agree."""
    duplicated = table[table.duplicated(keys, keep=False)]
    for _, group in duplicated.groupby(keys):
        if group[value].max() - group[value].min() > 1e-9:
            raise ValueError(f"combined runs disagree for {dict(zip(keys, group[keys].iloc[0], strict=True))}")
    return table.drop_duplicates(keys, keep="first").reset_index(drop=True)


def _rt_cv(summary: pd.DataFrame, window: str) -> pd.DataFrame:
    part = summary[(summary["window"] == window) & (summary["target"] == "rt_s")
                   & (summary["scope"] == "cv")]
    return part.set_index("model")


def criterion_1(summary: pd.DataFrame, decomposition: pd.DataFrame) -> GateResult:
    rt = _rt_cv(summary, "challenge")
    candidates = rt.loc[rt.index.intersection(CLASSICAL_MODELS)]
    if candidates.empty:
        return GateResult("1", False, "no classical model result for the challenge window")
    best = candidates["estimate"].idxmin()
    row = candidates.loc[best]
    beats = bool(row.get("delta_vs_demographics", 0.0) < 0
                 and row.get("delta_vs_demographics_ci_high", 0.0) < 0)
    dec = decomposition[(decomposition["window"] == "challenge") & (decomposition["model"] == best)
                        & (decomposition["scope"] == "cv")]
    if dec.empty:
        return GateResult("1", False, f"no within-child decomposition for {best}")
    r, r_low, r_high = (float(dec[c].iloc[0]) for c in ("r_within", "r_within_ci_low", "r_within_ci_high"))
    within = r_low > WITHIN_R_THRESHOLD
    original = bool(row["estimate"] <= ORIGINAL_THRESHOLD and row["ci_high"] < 1.0)
    detail = (f"best classical model {best}: nRMSE {row['estimate']:.4f} [{row['ci_low']:.4f}, "
              f"{row['ci_high']:.4f}]; delta vs demographics {row['delta_vs_demographics']:+.4f} "
              f"[{row['delta_vs_demographics_ci_low']:+.4f}, {row['delta_vs_demographics_ci_high']:+.4f}]; "
              f"within-child r {r:.3f} [{r_low:.3f}, {r_high:.3f}] (threshold {WITHIN_R_THRESHOLD}); "
              f"original criterion (nRMSE <= {ORIGINAL_THRESHOLD}) {'met' if original else 'not met'}")
    return GateResult("1", beats and within, detail)


def criterion_2(summary: pd.DataFrame) -> GateResult:
    rt = _rt_cv(summary, "prestim")
    models = [m for m in EEG_MODELS if m in rt.index]
    if not models or "delta_vs_demographics_ci_high" not in rt:
        return GateResult("2", False, "no EEG model with a paired comparison in the pre-stimulus window")
    lines, passed = [], False
    for model in models:
        row = rt.loc[model]
        beats = bool(row["delta_vs_demographics"] < 0 and row["delta_vs_demographics_ci_high"] < 0)
        passed |= beats
        lines.append(f"{model}: delta {row['delta_vs_demographics']:+.4f} "
                     f"[{row['delta_vs_demographics_ci_low']:+.4f}, "
                     f"{row['delta_vs_demographics_ci_high']:+.4f}]{' (beats)' if beats else ''}")
    return GateResult("2", passed, "; ".join(lines))


def criterion_3(failures: list[dict]) -> GateResult:
    leakage = [f for f in failures if "Leakage" in str(f.get("error", ""))]
    if leakage:
        return GateResult("3", False, f"{len(leakage)} model runs raised leakage errors")
    return GateResult("3", True, "every completed fit passed the leakage assertions; fingerprint "
                                 "accuracy reported separately")


def evaluate_gate(summary: pd.DataFrame, failures: list[dict],
                  decomposition: pd.DataFrame) -> list[GateResult]:
    summary = deduplicate(summary, _KEYS, "estimate")
    decomposition = deduplicate(decomposition, ["window", "scope", "model"], "nrmse_within")
    return [criterion_1(summary, decomposition), criterion_2(summary), criterion_3(failures)]


def secondary_table(summary: pd.DataFrame) -> pd.DataFrame:
    """All models, windows, targets and scopes with estimates and paired deltas, for reporting."""
    columns = ["window", "target", "scope", "model", "n_subjects", "n_trials", "metric", "estimate",
               "ci_low", "ci_high", "delta_vs_demographics", "delta_vs_demographics_ci_low",
               "delta_vs_demographics_ci_high", "delta_vs_mean", "delta_vs_mean_ci_low",
               "delta_vs_mean_ci_high"]
    summary = deduplicate(summary, _KEYS, "estimate")
    present = [c for c in columns if c in summary]
    return summary[present].sort_values(["window", "target", "scope", "model"]).reset_index(drop=True)
