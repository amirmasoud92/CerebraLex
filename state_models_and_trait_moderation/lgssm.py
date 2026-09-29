"""Linear-Gaussian state-space model over trial sequences, vectorised across sequences.

    x_1 ~ N(m0, V0),  x_t = A x_{t-1} + w_t,  w_t ~ N(0, Q)
    y_t = C x_t + d + v_t,  v_t ~ N(0, diag(r))

Filtering and RTS smoothing follow Shumway and Stoffer (1982); parameters are fitted by EM
(Ghahramani and Hinton 1996). Observations may be missing (``mask`` false): the state evolves
through them without an update. Because the observation noise is diagonal, each update works in
the latent dimension through the Woodbury identity, so no observation-sized matrix is inverted.
``filter`` returns causal estimates E[x_t | y_1..t] for held-out children.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class LGSSMParams:
    m0: np.ndarray   # (q,)
    V0: np.ndarray   # (q, q)
    A: np.ndarray    # (q, q)
    Q: np.ndarray    # (q, q)
    C: np.ndarray    # (d, q)
    d: np.ndarray    # (d,)
    r: np.ndarray    # (d,) diagonal observation variances


def _sym(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + np.swapaxes(m, -1, -2))


def kalman_filter(y, mask, p: LGSSMParams):
    """Filtered and predicted moments plus per-sequence log-likelihood (inside each length)."""
    n, t_max, _ = y.shape
    q = p.A.shape[0]
    rinv = 1.0 / p.r
    info = p.C.T @ (rinv[:, None] * p.C)                                 # (q, q) = C' R^-1 C
    x_pred = np.empty((n, t_max, q))
    p_pred = np.empty((n, t_max, q, q))
    x_filt = np.empty((n, t_max, q))
    p_filt = np.empty((n, t_max, q, q))
    loglik = np.zeros(n)
    xp = np.broadcast_to(p.m0, (n, q)).copy()
    pp = np.broadcast_to(p.V0, (n, q, q)).copy()
    eye = np.eye(q)
    logdet_r = np.log(p.r).sum()
    for t in range(t_max):
        if t:
            xp = x_filt[:, t - 1] @ p.A.T
            pp = _sym(p.A @ p_filt[:, t - 1] @ p.A.T + p.Q)
        x_pred[:, t], p_pred[:, t] = xp, pp
        obs = mask[:, t]
        pp_inv = np.linalg.inv(pp)
        e = y[:, t] - xp @ p.C.T - p.d                                   # innovations (n, d)
        g = (e * rinv) @ p.C                                             # C' R^-1 e  (n, q)
        pf = _sym(np.linalg.inv(pp_inv + info))                          # (n, q, q)
        xf = xp + np.einsum("nij,nj->ni", pf, g)
        # log N(e; 0, S) with S = C P C' + R via the determinant lemma and Woodbury
        _, logdet_ip = np.linalg.slogdet(eye + pp @ info)
        quad = (e * e * rinv).sum(axis=1) - np.einsum("ni,nij,nj->n", g, pf, g)
        ll = -0.5 * (y.shape[2] * _LOG_2PI + logdet_r + logdet_ip + quad)
        loglik += np.where(obs, ll, 0.0)
        x_filt[:, t] = np.where(obs[:, None], xf, xp)
        p_filt[:, t] = np.where(obs[:, None, None], pf, pp)
    return x_pred, p_pred, x_filt, p_filt, loglik


def rts_smoother(p: LGSSMParams, x_pred, p_pred, x_filt, p_filt):
    """Smoothed moments and lag-one cross-covariances Cov(x_t, x_{t-1} | Y) for t >= 1."""
    n, t_max, q = x_filt.shape
    x_s = x_filt.copy()
    p_s = p_filt.copy()
    p_lag = np.zeros((n, t_max, q, q))
    for t in range(t_max - 2, -1, -1):
        gain = p_filt[:, t] @ p.A.T @ np.linalg.inv(p_pred[:, t + 1])   # (n, q, q)
        x_s[:, t] = x_filt[:, t] + np.einsum("nij,nj->ni", gain, x_s[:, t + 1] - x_pred[:, t + 1])
        p_s[:, t] = _sym(p_filt[:, t] + gain @ (p_s[:, t + 1] - p_pred[:, t + 1]) @ np.swapaxes(gain, 1, 2))
        p_lag[:, t + 1] = p_s[:, t + 1] @ np.swapaxes(gain, 1, 2)
    return x_s, p_s, p_lag


class LinearGaussianSSM:
    """LGSSM fitted by EM on masked sequences; latent dimension ``n_latent``."""

    def __init__(self, n_latent: int, max_iter: int = 200, tol: float = 1e-6,
                 variance_floor: float = 1e-3, seed: int = 0) -> None:
        self.n_latent = n_latent
        self.max_iter = max_iter
        self.tol = tol
        self.variance_floor = variance_floor
        self.seed = seed

    def _init(self, y, mask) -> LGSSMParams:
        obs = y[mask]
        d = obs.mean(axis=0)
        centred = obs - d
        cov = np.atleast_2d(np.cov(centred, rowvar=False))       # a 0-d array when d == 1
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1][: self.n_latent]
        # start with half of each leading component's variance in the latent state and the rest as
        # observation noise; putting it all in the state lets EM settle on white "dynamics"
        loading = eigvec[:, order] * np.sqrt(np.maximum(0.5 * eigval[order], 1e-12))
        resid = np.maximum(np.diag(cov) - (loading ** 2).sum(axis=1), self.variance_floor * np.diag(cov))
        q = self.n_latent
        return LGSSMParams(np.zeros(q), np.eye(q), 0.9 * np.eye(q), 0.19 * np.eye(q), loading, d, resid)

    def _e_step(self, y, mask, lengths, p):
        x_pred, p_pred, x_filt, p_filt, loglik = kalman_filter(y, mask, p)
        x_s, p_s, p_lag = rts_smoother(p, x_pred, p_pred, x_filt, p_filt)
        return x_s, p_s, p_lag, float(loglik.sum())

    def _m_step(self, y, mask, lengths, x_s, p_s, p_lag, floor) -> LGSSMParams:
        n, t_max, q = x_s.shape
        inside = np.arange(t_max)[None, :] < lengths[:, None]           # (n, T)
        exx = p_s + x_s[..., :, None] * x_s[..., None, :]               # E[x_t x_t']
        m0 = x_s[:, 0].mean(axis=0)
        dev = x_s[:, 0] - m0
        v0 = _sym((p_s[:, 0] + dev[:, :, None] * dev[:, None, :]).mean(axis=0)) + 1e-9 * np.eye(q)
        trans = inside[:, 1:]                                           # transitions t-1 -> t
        e_lag = p_lag[:, 1:] + x_s[:, 1:, :, None] * x_s[:, :-1, None, :]   # E[x_t x_{t-1}']
        s_lag = (e_lag * trans[..., None, None]).sum(axis=(0, 1))
        s_prev = (exx[:, :-1] * trans[..., None, None]).sum(axis=(0, 1))
        s_curr = (exx[:, 1:] * trans[..., None, None]).sum(axis=(0, 1))
        a_mat = s_lag @ np.linalg.inv(s_prev)
        q_mat = _sym((s_curr - a_mat @ s_lag.T) / max(int(trans.sum()), 1)) + 1e-9 * np.eye(q)
        w = (mask & inside).astype(float)                               # observed trials
        n_obs = w.sum()
        sx = (x_s * w[..., None]).sum(axis=(0, 1))
        sxx = (exx * w[..., None, None]).sum(axis=(0, 1))
        sy = (y * w[..., None]).sum(axis=(0, 1))
        syx = np.einsum("ntd,ntq->dq", y * w[..., None], x_s)
        big = np.block([[sxx, sx[:, None]], [sx[None, :], np.array([[n_obs]])]])
        cd = np.concatenate([syx, sy[:, None]], axis=1) @ np.linalg.inv(big)
        c_mat, dvec = cd[:, :q], cd[:, q]
        resid = y - x_s @ c_mat.T - dvec                                    # E[y - Cx - d] parts
        r = ((resid ** 2 * w[..., None]).sum(axis=(0, 1))
             + np.einsum("dq,qp,dp->d", c_mat, (p_s * w[..., None, None]).sum(axis=(0, 1)), c_mat)) / n_obs
        return LGSSMParams(m0, v0, a_mat, q_mat, c_mat, dvec, np.maximum(r, floor))

    def fit(self, y: np.ndarray, mask: np.ndarray, lengths: np.ndarray) -> LinearGaussianSSM:
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        floor = self.variance_floor * y[mask].var(axis=0)
        params = self._init(y, mask)
        history = []
        for _ in range(self.max_iter):
            x_s, p_s, p_lag, total = self._e_step(y, mask, lengths, params)
            history.append(total)
            if len(history) > 1 and abs(history[-1] - history[-2]) <= self.tol * abs(history[-2]):
                break
            params = self._m_step(y, mask, lengths, x_s, p_s, p_lag, floor)
        self.params_ = params
        self.history_ = history
        self.train_loglik_ = history[-1]
        return self

    def loglik_per_trial(self, y, mask, lengths) -> float:
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        *_, loglik = kalman_filter(y, mask, self.params_)
        return float(loglik.sum() / max(int(mask.sum()), 1))

    def filter(self, y, mask, lengths) -> np.ndarray:
        """Causal state means E[x_t | y_1..t], shape (N, T, q)."""
        y = np.where(mask[:, :, None], np.asarray(y, dtype=np.float64), 0.0)
        _, _, x_filt, _, _ = kalman_filter(y, mask, self.params_)
        return x_filt
