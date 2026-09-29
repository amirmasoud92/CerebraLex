"""Gaussian hidden Markov model over trial sequences, vectorised across sequences.

Each sequence is one block of trials (padded to a common length). ``mask`` marks trials with an
observation; unobserved trials (artifact windows, padding) contribute no emission, but the hidden
state still evolves through them, and transitions are counted only inside each sequence's length.
Emissions are Gaussian with diagonal covariance. Inference uses the scaled forward-backward
recursions (Rabiner 1989); ``filter`` returns the causal posteriors P(z_t | y_1..t) used as state
features on held-out children, so no future trial ever informs the state of the current one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class HMMParams:
    start: np.ndarray        # (K,)
    trans: np.ndarray        # (K, K), rows sum to one
    means: np.ndarray        # (K, d)
    variances: np.ndarray    # (K, d)


def _log_emissions(y: np.ndarray, mask: np.ndarray, params: HMMParams) -> np.ndarray:
    """log p(y_nt | z = k) with zeros where the trial is unobserved; shape (N, T, K)."""
    var = params.variances[None, None]                                   # (1, 1, K, d)
    diff = y[:, :, None, :] - params.means[None, None]                   # (N, T, K, d)
    logb = -0.5 * (_LOG_2PI + np.log(var) + diff ** 2 / var).sum(axis=-1)
    return np.where(mask[:, :, None], logb, 0.0)


def _forward_backward(y, mask, lengths, params, backward: bool = True):
    n, t_max = mask.shape
    k = params.start.shape[0]
    logb = _log_emissions(y, mask, params)
    shift = logb.max(axis=2, keepdims=True)
    b = np.exp(logb - shift)                                             # (N, T, K)
    alpha = np.empty((n, t_max, k))
    scale = np.empty((n, t_max))
    a = params.start[None] * b[:, 0]
    scale[:, 0] = a.sum(axis=1)
    alpha[:, 0] = a / scale[:, 0:1]
    for t in range(1, t_max):
        a = (alpha[:, t - 1] @ params.trans) * b[:, t]
        scale[:, t] = a.sum(axis=1)
        alpha[:, t] = a / scale[:, t:t + 1]
    inside = np.arange(t_max)[None, :] < lengths[:, None]
    loglik = ((np.log(scale) + shift[:, :, 0]) * inside).sum(axis=1)
    if not backward:
        return alpha, None, None, loglik
    beta = np.ones((n, t_max, k))
    for t in range(t_max - 2, -1, -1):
        beta[:, t] = ((b[:, t + 1] * beta[:, t + 1]) @ params.trans.T) / scale[:, t + 1:t + 2]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    # expected transitions summed over time steps inside each sequence: (N, K, K)
    xi_num = (alpha[:, :-1, :, None] * params.trans[None, None]
              * (b[:, 1:] * beta[:, 1:])[:, :, None, :] / scale[:, 1:, None, None])
    real = (np.arange(t_max - 1)[None, :] < (lengths[:, None] - 1))[:, :, None, None]
    xi = (xi_num * real).sum(axis=1)
    return alpha, gamma, xi, loglik


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM with masked emissions, fitted by EM with restarts."""

    def __init__(self, n_states: int, n_restarts: int = 5, max_iter: int = 200, tol: float = 1e-6,
                 sticky: float = 0.9, variance_floor: float = 1e-3, seed: int = 0) -> None:
        self.n_states = n_states
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.sticky = sticky
        self.variance_floor = variance_floor
        self.seed = seed

    def _init(self, y, mask, rng: np.random.Generator) -> HMMParams:
        from sklearn.cluster import KMeans

        k = self.n_states
        obs = y[mask]
        sample = obs[rng.choice(len(obs), size=min(len(obs), 20000), replace=False)]
        km = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(1 << 31))).fit(sample)
        variances = np.tile(obs.var(axis=0), (k, 1))
        trans = np.full((k, k), (1.0 - self.sticky) / max(k - 1, 1))
        np.fill_diagonal(trans, self.sticky if k > 1 else 1.0)
        return HMMParams(np.full(k, 1.0 / k), trans, km.cluster_centers_.astype(float), variances)

    def _m_step(self, y, mask, gamma, xi, floor) -> HMMParams:
        start = gamma[:, 0].mean(axis=0)
        trans = xi.sum(axis=0)
        trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-300)
        weights = gamma * mask[:, :, None]                               # (N, T, K)
        total = weights.sum(axis=(0, 1))                                 # (K,)
        means = np.einsum("ntk,ntd->kd", weights, y) / np.maximum(total[:, None], 1e-300)
        sq = np.einsum("ntk,ntkd->kd", weights, (y[:, :, None, :] - means[None, None]) ** 2)
        variances = sq / np.maximum(total[:, None], 1e-300) + floor
        return HMMParams(start, trans, means, variances)

    def fit(self, y: np.ndarray, mask: np.ndarray, lengths: np.ndarray) -> GaussianHMM:
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        floor = self.variance_floor * y[mask].var(axis=0)
        rng = np.random.default_rng(self.seed)
        best, best_ll, histories = None, -np.inf, []
        for _restart in range(self.n_restarts):
            params = self._init(y, mask, rng)
            history = []
            for _ in range(self.max_iter):
                _, gamma, xi, loglik = _forward_backward(y, mask, lengths, params)
                total = float(loglik.sum())
                history.append(total)
                if len(history) > 1 and abs(history[-1] - history[-2]) <= self.tol * abs(history[-2]):
                    break
                params = self._m_step(y, mask, gamma, xi, floor)
            histories.append(history)
            if history[-1] > best_ll:
                best, best_ll = params, history[-1]
        self.params_ = best
        self.train_loglik_ = best_ll
        self.restart_logliks_ = [h[-1] for h in histories]
        self.histories_ = histories
        self.n_observed_ = int(mask.sum())
        return self

    def loglik_per_trial(self, y, mask, lengths) -> float:
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        _, _, _, loglik = _forward_backward(y, mask, lengths, self.params_, backward=False)
        return float(loglik.sum() / max(int(mask.sum()), 1))

    def filter(self, y, mask, lengths) -> np.ndarray:
        """Causal state posteriors P(z_t | y_1..t), shape (N, T, K)."""
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        alpha, _, _, _ = _forward_backward(y, mask, lengths, self.params_, backward=False)
        return alpha

    def smooth(self, y, mask, lengths) -> np.ndarray:
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        _, gamma, _, _ = _forward_backward(y, mask, lengths, self.params_)
        return gamma

    def n_parameters(self, d: int) -> int:
        k = self.n_states
        return (k - 1) + k * (k - 1) + 2 * k * d
