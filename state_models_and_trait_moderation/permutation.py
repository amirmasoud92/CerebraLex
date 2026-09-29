"""Subject-level trait permutation nulls and FDR control for the trait-by-state interactions."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_LOG = logging.getLogger(__name__)


def permute_traits_across_subjects(data: pd.DataFrame, traits: list[str], seed: int) -> pd.DataFrame:
    """Shuffle each subject's trait vector to another subject; within-subject structure intact."""
    rng = np.random.default_rng(seed)
    table = data.drop_duplicates("subject")[["subject", *traits]].reset_index(drop=True)
    permuted = table.copy()
    order = rng.permutation(len(table))
    permuted[traits] = table.loc[order, traits].to_numpy()
    mapping = permuted.set_index("subject")[traits]
    out = data.copy()
    out[traits] = mapping.loc[out["subject"]].to_numpy()
    return out


def permutation_test_traits(
    data: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], dict[str, float]],
    traits: list[str],
    n_permutations: int = 5000,
    seed: int = 12345,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Two-sided permutation p-values for each trait's interaction estimate.

    ``fit_fn`` maps a data frame to ``{trait: interaction estimate}``; it is applied to the
    observed data and to ``n_permutations`` trait-shuffled copies.
    """
    observed = fit_fn(data)

    def one(k: int) -> dict[str, float]:
        return fit_fn(permute_traits_across_subjects(data, traits, seed + k))

    null = Parallel(n_jobs=n_jobs)(delayed(one)(k) for k in range(n_permutations))
    null_frame = pd.DataFrame(null)
    rows = []
    for trait in traits:
        draws = null_frame[trait].to_numpy(dtype=float)
        obs = float(observed[trait])
        p = (np.sum(np.abs(draws) >= abs(obs)) + 1) / (len(draws) + 1)
        rows.append({"trait": trait, "estimate": obs, "null_mean": float(np.nanmean(draws)),
                     "null_sd": float(np.nanstd(draws)), "p_perm": float(p),
                     "n_permutations": int(len(draws))})
    return pd.DataFrame(rows)


def fdr_correct(pvalues: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg adjusted p-values and rejection flags."""
    from statsmodels.stats.multitest import multipletests

    reject, adjusted, _, _ = multipletests(np.asarray(pvalues, dtype=float), alpha=q, method="fdr_bh")
    return adjusted, reject
