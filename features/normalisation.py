"""Standardisation of features within groups (for example within child).

Removing each child's own mean and scale from every feature is the first-order way to make a
representation blind to the child's fingerprint: it uses only that child's unlabelled trials and
fits nothing across children, like unsupervised per-subject alignment (He and Wu 2020, IEEE TBME
67:399-410). NaN entries stay NaN and are ignored in the statistics; a feature that is constant
within a group becomes 0 there.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def standardise_within_groups(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return ``x`` with every column centred and scaled to unit SD (ddof 0) within each group."""
    x = np.asarray(x, dtype=np.float64)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    indicator = sparse.csr_matrix((np.ones(len(inverse)), (inverse, np.arange(len(inverse)))))
    observed = ~np.isnan(x)
    counts = np.asarray(indicator @ observed.astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.asarray(indicator @ np.where(observed, x, 0.0)) / counts
        centred = x - means[inverse]
        variances = np.asarray(indicator @ np.where(observed, centred ** 2, 0.0)) / counts
    scale = np.sqrt(variances)
    scale[~(scale > 0)] = 1.0
    return centred / scale[inverse]
