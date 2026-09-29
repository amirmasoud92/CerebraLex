"""Channel covariance matrices for the Riemannian tangent-space baseline.

Only the covariance estimation is a feature; the common-subspace projection and the
tangent-space reference point are fitted transforms that must be estimated inside training folds
(see ``decoding.baselines.TangentSpaceRidge``).
"""

from __future__ import annotations

import numpy as np


def window_covariances(windows: np.ndarray, shrinkage: str = "lwf", device: str | None = None) -> np.ndarray:
    """Covariance per window with Ledoit-Wolf shrinkage; optional CUDA batched computation.

    Ledoit-Wolf shrinkage follows pyriemann's ``Covariances(estimator="lwf")`` (sklearn's
    ``ledoit_wolf`` on centred data) for parity with the library implementation; the GPU path
    computes the same estimator and is used only when it is measurably faster.
    """
    if device is not None and device != "cpu":
        return _covariances_torch(windows, device)
    from pyriemann.estimation import Covariances

    return Covariances(estimator=shrinkage).transform(np.asarray(windows, dtype=np.float64))


def _covariances_torch(windows: np.ndarray, device: str) -> np.ndarray:
    import torch

    x = torch.as_tensor(np.asarray(windows, dtype=np.float32), device=device)
    n, c, t = x.shape
    x = x - x.mean(dim=2, keepdim=True)
    sample = x @ x.transpose(1, 2) / t
    mu = torch.diagonal(sample, dim1=1, dim2=2).mean(dim=1)
    x2 = x ** 2
    eye = torch.eye(c, device=device)
    delta = ((sample - mu[:, None, None] * eye) ** 2).sum(dim=(1, 2)) / c
    beta = (((x2 @ x2.transpose(1, 2)) / t - sample ** 2).sum(dim=(1, 2)) / (c * t))
    shrink = torch.clamp(beta / delta, 0.0, 1.0)
    out = (1 - shrink)[:, None, None] * sample + (shrink * mu)[:, None, None] * eye
    return out.double().cpu().numpy()


def covariance_upper_triangle(cov: np.ndarray) -> np.ndarray:
    """Vectorise symmetric matrices [n, c, c] into their upper triangle including the diagonal."""
    c = cov.shape[1]
    iu = np.triu_indices(c)
    return cov[:, iu[0], iu[1]].astype(np.float32)


def n_channels_from_upper(n_values: int) -> int:
    """Invert c * (c + 1) / 2 = n_values."""
    c = int(round((np.sqrt(8 * n_values + 1) - 1) / 2))
    if c * (c + 1) // 2 != n_values:
        raise ValueError(f"{n_values} is not a triangular number")
    return c


def upper_triangle_to_matrices(upper: np.ndarray, n_channels: int | None = None) -> np.ndarray:
    """Rebuild symmetric float64 matrices [n, c, c] from upper-triangle vectors."""
    upper = np.asarray(upper, dtype=np.float64)
    c = n_channels or n_channels_from_upper(upper.shape[1])
    iu = np.triu_indices(c)
    out = np.zeros((len(upper), c, c), dtype=np.float64)
    out[:, iu[0], iu[1]] = upper
    out = out + np.transpose(out, (0, 2, 1))
    diag = np.arange(c)
    out[:, diag, diag] /= 2.0
    return out
