"""Behavioural descriptives that set sensitivity-analysis parameters without touching the EEG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest


@dataclass(frozen=True)
class FastGuessCutoff:
    """Result of the accuracy-by-latency procedure for fast guesses."""

    cutoff_s: float | None
    bins: pd.DataFrame
    step_s: float
    max_rt_s: float
    ci_level: float
    chance: float

    def summary(self, rt_s: np.ndarray) -> dict[str, Any]:
        rt = np.asarray(rt_s, dtype=float)
        below = int(np.sum(rt < self.cutoff_s)) if self.cutoff_s is not None else 0
        return {"cutoff_s": self.cutoff_s, "step_s": self.step_s, "max_rt_s": self.max_rt_s,
                "ci_level": self.ci_level, "chance": self.chance, "n_responses": int(rt.size),
                "n_below_cutoff": below, "fraction_below_cutoff": below / rt.size if rt.size else float("nan")}


def fast_guess_cutoff(rt_s: np.ndarray, correct: np.ndarray, step_s: float = 0.05,
                      max_rt_s: float = 1.0, ci_level: float = 0.95,
                      chance: float = 0.5) -> FastGuessCutoff:
    """Lower RT cutoff below which choice accuracy is at chance.

    Follows Ratcliff and Tuerlinckx (2002, Psychon Bull Rev 9:438, p. 442): true fast guesses
    are responses at chance accuracy, and the lower cutoff is raised in small steps until accuracy
    begins to rise above chance. Responses are binned in steps of ``step_s`` up to ``max_rt_s``
    and each bin receives a Wilson interval. The cutoff is the lower edge of the first bin from
    which every bin up to ``max_rt_s`` has its lower bound above ``chance``; an isolated bin above
    chance followed by chance-level bins does not end the search. ``None`` means accuracy never
    rises above chance below ``max_rt_s``; 0.0 means no bin is at chance.
    """
    rt = np.asarray(rt_s, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    if rt.shape != ok.shape:
        raise ValueError("rt_s and correct must have the same shape")
    finite = np.isfinite(rt)
    rt, ok = rt[finite], ok[finite]
    n_bins = int(round(max_rt_s / step_s))
    edges = np.round(np.arange(n_bins + 1) * step_s, 10)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (rt >= lo) & (rt < hi)
        n, k = int(in_bin.sum()), int(ok[in_bin].sum())
        if n:
            ci = binomtest(k, n, chance).proportion_ci(confidence_level=ci_level, method="wilson")
            low, high = float(ci.low), float(ci.high)
        else:
            low = high = float("nan")
        rows.append({"rt_bin_start_s": float(lo), "rt_bin_end_s": float(hi), "n": n, "n_correct": k,
                     "accuracy": k / n if n else float("nan"), "ci_low": low, "ci_high": high,
                     "above_chance": bool(n and low > chance)})
    bins = pd.DataFrame(rows)
    above = bins["above_chance"].to_numpy()
    cutoff: float | None = None
    for i in range(len(bins)):
        if above[i:].all():
            cutoff = float(bins["rt_bin_start_s"].iloc[i])
            break
    return FastGuessCutoff(cutoff_s=cutoff, bins=bins, step_s=step_s, max_rt_s=max_rt_s,
                           ci_level=ci_level, chance=chance)
