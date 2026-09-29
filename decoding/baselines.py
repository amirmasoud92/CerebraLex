"""Classical baselines: mean predictor, grouped ridge, Riemannian tangent-space ridge."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from decoding.base import Decoder, DecoderInput
from decoding.leakage import LeakageAuditor
from features.riemannian import n_channels_from_upper, upper_triangle_to_matrices

_LOG = logging.getLogger(__name__)
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-2, 8, 11))


@dataclass(frozen=True)
class RidgePath:
    """Coefficients of a ridge path with an unpenalised block ``A = [1, C]``.

    ``g`` regresses the penalised features on ``A`` and ``h`` regresses the target on ``A``
    (least squares on training data); ``betas`` holds one penalised coefficient vector per alpha.
    """

    g: np.ndarray
    h: np.ndarray
    betas: np.ndarray
    alphas: tuple[float, ...]


def _unpenalised_block(n: int, covariates: np.ndarray | None) -> np.ndarray:
    ones = np.ones((n, 1))
    if covariates is None or np.asarray(covariates).shape[1] == 0:
        return ones
    return np.column_stack([ones, np.asarray(covariates, dtype=np.float64)])


def fit_ridge_path(x: np.ndarray, y: np.ndarray, alphas: tuple[float, ...],
                   covariates: np.ndarray | None = None) -> RidgePath:
    """Solve ``min ||y - A g0 - X b||^2 + alpha ||b||^2`` for every alpha, ``A = [1, C]``.

    Only ``b`` is penalised. The unpenalised block is profiled out (Frisch-Waugh-Lovell):
    ``X`` and ``y`` are residualised on ``A`` and ridge is solved on the residuals from one
    decomposition (Gram eigendecomposition when n >= p, SVD otherwise). Without covariates
    ``A`` is the intercept and the result equals ``sklearn.linear_model.Ridge``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = _unpenalised_block(len(y), covariates)
    h = np.linalg.lstsq(a, y, rcond=None)[0]
    yr = y - a @ h
    n, p = x.shape
    if p == 0:
        return RidgePath(np.zeros((a.shape[1], 0)), h, np.zeros((len(alphas), 0)), tuple(alphas))
    g = np.linalg.lstsq(a, x, rcond=None)[0]
    xr = x - a @ g
    if n >= p:
        eigval, eigvec = np.linalg.eigh(xr.T @ xr)
        eigval = np.clip(eigval, 0.0, None)
        rhs = eigvec.T @ (xr.T @ yr)
        betas = np.stack([eigvec @ (rhs / (eigval + alpha)) for alpha in alphas])
    else:
        u, sv, vt = np.linalg.svd(xr, full_matrices=False)
        uty = u.T @ yr
        betas = np.stack([vt.T @ (sv / (sv ** 2 + alpha) * uty) for alpha in alphas])
    return RidgePath(g, h, betas, tuple(alphas))


def predict_ridge_path(path: RidgePath, x: np.ndarray,
                       covariates: np.ndarray | None = None) -> np.ndarray:
    """Predictions [n_alphas, n] of a fitted ridge path."""
    x = np.asarray(x, dtype=np.float64)
    a = _unpenalised_block(len(x), covariates)
    base = a @ path.h
    if x.shape[1] == 0:
        return np.tile(base, (len(path.alphas), 1))
    return base[None, :] + ((x - a @ path.g) @ path.betas.T).T


def ridge_path_predictions(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray,
                           alphas: tuple[float, ...], c_train: np.ndarray | None = None,
                           c_eval: np.ndarray | None = None) -> np.ndarray:
    """Predictions of the ridge path for every alpha; see ``fit_ridge_path``."""
    path = fit_ridge_path(x_train, y_train, alphas, c_train)
    return predict_ridge_path(path, x_eval, c_eval)


class MedianImputer(TransformerMixin, BaseEstimator):
    """Training-median imputation, numerically identical to
    ``SimpleImputer(strategy="median", keep_empty_features=True)`` but about four times faster on
    wide dense matrices: the NaN-aware median is computed only for columns that contain NaN, and a
    column that is entirely NaN is filled with 0 as scikit-learn does."""

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> MedianImputer:
        x = np.asarray(x, dtype=np.float64)
        has_nan = np.isnan(x).any(axis=0)
        medians = np.zeros(x.shape[1])
        if (~has_nan).any():
            medians[~has_nan] = np.median(x[:, ~has_nan], axis=0)
        if has_nan.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns
                gaps = np.nanmedian(x[:, has_nan], axis=0)
            medians[has_nan] = np.where(np.isnan(gaps), 0.0, gaps)
        self.statistics_ = medians
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        missing = np.isnan(x)
        if not missing.any():
            return x
        x = x.copy()
        rows, cols = np.nonzero(missing)
        x[rows, cols] = self.statistics_[cols]
        return x


class MeanPredictor(Decoder):
    """Predicts the training mean; the reference level for nRMSE."""

    name = "mean"

    def fit(self, data: DecoderInput, y: np.ndarray) -> MeanPredictor:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, data: DecoderInput) -> np.ndarray:
        return np.full(len(data), self.mean_)


class GroupedRidge(Decoder):
    """Ridge on EEG features with an unpenalised covariate block; alpha by inner grouped CV.

    The EEG block is imputed (training median), standardised and optionally reduced by PCA; the
    covariates (age, sex, handedness) are imputed and standardised separately, kept out of the
    PCA and left unpenalised, so an EEG model contains the demographics-only model as its
    alpha-to-infinity limit and its difference from that model isolates the EEG contribution.
    Every transform is fitted on training subjects only; the inner CV groups by subject.
    With no EEG columns the model is ordinary least squares on the covariates.
    """

    def __init__(self, name: str = "ridge", alphas: tuple[float, ...] = DEFAULT_ALPHAS,
                 n_inner_folds: int = 5, pca_components: int | None = None,
                 seed: int = 12345) -> None:
        self.name = name
        self.alphas = tuple(alphas)
        self.n_inner_folds = n_inner_folds
        self.pca_components = pca_components
        self.seed = seed

    def _make_preprocessor(self, n_samples: int, n_features: int):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        steps = [("impute", MedianImputer()), ("scale", StandardScaler())]
        if self.pca_components is not None:
            k = int(min(self.pca_components, n_samples - 1, n_features))
            steps.append(("pca", PCA(n_components=k, svd_solver="randomized",
                                     random_state=self.seed)))
        return Pipeline(steps)

    @staticmethod
    def _make_covariate_preprocessor():
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline([("impute", MedianImputer()), ("scale", StandardScaler())])

    def _transform_pair(self, x: np.ndarray, c: np.ndarray | None, fit_rows: np.ndarray,
                        eval_rows: np.ndarray):
        z_fit, z_eval = np.zeros((len(fit_rows), 0)), np.zeros((len(eval_rows), 0))
        if x.shape[1]:
            prep = self._make_preprocessor(len(fit_rows), x.shape[1]).fit(x[fit_rows])
            z_fit, z_eval = prep.transform(x[fit_rows]), prep.transform(x[eval_rows])
        c_fit = c_eval = None
        if c is not None:
            cprep = self._make_covariate_preprocessor().fit(c[fit_rows])
            c_fit, c_eval = cprep.transform(c[fit_rows]), cprep.transform(c[eval_rows])
        return z_fit, z_eval, c_fit, c_eval

    def fit(self, data: DecoderInput, y: np.ndarray) -> GroupedRidge:
        from sklearn.model_selection import GroupKFold

        x = self._eeg(data)
        c = self._covariates(data)
        y = np.asarray(y, dtype=np.float64)
        self.inner_mse_ = None
        self.alpha_at_grid_edge_ = False
        if x.shape[1] == 0:
            self.alpha_ = None                                  # nothing is penalised
        else:
            n_groups = len(np.unique(data.subjects))
            splitter = GroupKFold(n_splits=min(self.n_inner_folds, n_groups))
            mse = np.zeros((splitter.get_n_splits(), len(self.alphas)))
            for f, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups=data.subjects)):
                LeakageAuditor.assert_disjoint(data.subjects[train_idx], data.subjects[val_idx])
                z_tr, z_va, c_tr, c_va = self._transform_pair(x, c, train_idx, val_idx)
                preds = ridge_path_predictions(z_tr, y[train_idx], z_va, self.alphas, c_tr, c_va)
                mse[f] = ((preds - y[val_idx]) ** 2).mean(axis=1)
            mean_mse = mse.mean(axis=0)
            best = int(np.argmin(mean_mse))
            self.alpha_ = self.alphas[best]
            self.inner_mse_ = dict(zip(self.alphas, mean_mse.round(8).tolist(), strict=True))
            self.alpha_at_grid_edge_ = best in (0, len(self.alphas) - 1)
            if self.alpha_at_grid_edge_:
                _LOG.warning("%s: selected alpha %.3g lies on the grid edge", self.name, self.alpha_)
        self.preprocessor_ = (self._make_preprocessor(len(x), x.shape[1]).fit(x)
                              if x.shape[1] else None)
        self.covariate_preprocessor_ = (self._make_covariate_preprocessor().fit(c)
                                        if c is not None else None)
        z, c_fit = self._apply(x, c)
        alphas = (self.alpha_,) if self.alpha_ is not None else (0.0,)
        self.path_ = fit_ridge_path(z, y, alphas, c_fit)
        self.fitted_subjects_ = np.unique(data.subjects)
        return self

    def predict(self, data: DecoderInput) -> np.ndarray:
        LeakageAuditor.assert_fit_subjects(self.fitted_subjects_, data.subjects, self.name)
        z, c_eval = self._apply(self._eeg(data), self._covariates(data))
        return predict_ridge_path(self.path_, z, c_eval)[0]

    def _apply(self, x: np.ndarray, c: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform with the preprocessors fitted on all training rows."""
        z = self.preprocessor_.transform(x) if self.preprocessor_ is not None else np.zeros((len(x), 0))
        if self.covariate_preprocessor_ is None:
            return z, None
        if c is None:
            raise ValueError(f"{self.name} was fitted with covariates but none were given")
        return z, self.covariate_preprocessor_.transform(c)

    @staticmethod
    def _eeg(data: DecoderInput) -> np.ndarray:
        x = np.asarray(data.x, dtype=np.float64)
        if x.ndim > 2:
            x = x.reshape(len(x), -1)
        return x

    @staticmethod
    def _covariates(data: DecoderInput) -> np.ndarray | None:
        if data.covariates is None:
            return None
        c = np.asarray(data.covariates, dtype=np.float64)
        return c if c.ndim == 2 and c.shape[1] else None

    def describe(self) -> dict:
        return {"name": self.name, "alpha": getattr(self, "alpha_", None),
                "alpha_at_grid_edge": getattr(self, "alpha_at_grid_edge_", None),
                "inner_mse": getattr(self, "inner_mse_", None),
                "pca_components": self.pca_components, "n_inner_folds": self.n_inner_folds,
                "covariates": "unpenalised block"}


def assert_positive_definite(matrices, what: str, rel_tol: float = 1e-10) -> None:
    """Raise when any symmetric matrix has a smallest eigenvalue at or below ``rel_tol`` times
    its mean eigenvalue; the Riemannian log map is undefined for such matrices."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dependency
        torch = None
    if torch is not None and isinstance(matrices, torch.Tensor):
        eig = torch.linalg.eigvalsh(matrices)
        rel = (eig[:, 0] / eig.mean(dim=1)).detach().cpu().numpy()
    else:
        eig = np.linalg.eigvalsh(np.asarray(matrices))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = eig[:, 0] / eig.mean(axis=1)
    bad = np.flatnonzero(~(rel > rel_tol))
    if len(bad):
        raise ValueError(f"{what}: {len(bad)} of {len(rel)} matrices are not positive definite "
                         f"(first positions {bad[:10].tolist()}); check window quality flags")


class TangentSpaceRidge(GroupedRidge):
    """Common-subspace projection, Riemannian tangent space, then grouped ridge.

    Input rows are upper-triangle vectors of channel covariance matrices. Average referencing
    and spherical-spline interpolation make the 129 x 129 matrices rank deficient (rank at most
    128 minus the number of interpolated channels), so the log map would be dominated by
    near-zero eigenvalues. Following Sabbagh et al. (NeurIPS 2019), every matrix is first
    projected onto the leading ``projection_rank`` eigenvectors of the training-fold mean
    covariance; with at most 20% interpolated channels the minimum rank of a usable block is
    103, so a rank of 100 keeps every projected matrix full rank. The projection and the
    tangent-space reference point are fitted on training subjects only.
    """

    def __init__(self, name: str = "riemann_ts", projection_rank: int | None = 100,
                 mean_subsample: int | None = 4000, metric: str = "riemann",
                 chunk_size: int = 4096, device: str | None = None, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.projection_rank = projection_rank
        self.mean_subsample = mean_subsample
        self.metric = metric
        self.chunk_size = chunk_size
        self.device = device

    def _backend_device(self):
        """CUDA tensors when requested and available: pyriemann (>= 0.12) runs the identical
        algorithm through its array namespace (parity to 1e-11 relative, about 20x faster)."""
        if self.device in (None, "cpu"):
            return None
        import torch

        return torch.device(self.device) if torch.cuda.is_available() else None

    def _project(self, matrices: np.ndarray):
        device = self._backend_device()
        if device is None:
            w = self.projection_
            return np.einsum("ci,ncd,dj->nij", w, matrices, w, optimize=True)
        import torch

        w = torch.as_tensor(self.projection_, dtype=torch.float64, device=device)
        x = torch.as_tensor(matrices, dtype=torch.float64, device=device)
        return w.T @ x @ w

    def _tangent_vectors(self, upper) -> np.ndarray:
        n = len(upper)
        out = None
        for start in range(0, n, self.chunk_size):
            block = np.asarray(upper[start:start + self.chunk_size], dtype=np.float64)
            projected = self._project(upper_triangle_to_matrices(block, self.n_channels_))
            assert_positive_definite(projected, f"{self.name} rows {start} to {start + len(block)}")
            vectors = self.tangent_.transform(projected)
            if not isinstance(vectors, np.ndarray):
                vectors = vectors.detach().cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError(f"{self.name}: non-finite tangent vectors in rows {start} to "
                                 f"{start + len(block)}")
            if out is None:
                out = np.empty((n, vectors.shape[1]), dtype=np.float32)
            out[start:start + len(vectors)] = vectors
        return out

    def fit_mapping(self, upper) -> TangentSpaceRidge:
        """Fit the common-subspace projection and the tangent-space reference point."""
        from pyriemann.tangentspace import TangentSpace

        self.n_channels_ = n_channels_from_upper(np.asarray(upper[:1]).shape[1])
        rng = np.random.default_rng(self.seed)
        n = len(upper)
        idx = np.sort(rng.choice(n, size=min(self.mean_subsample or n, n), replace=False))
        reference = upper_triangle_to_matrices(np.asarray(upper[idx]), self.n_channels_)
        rank = self.n_channels_ if self.projection_rank is None else min(self.projection_rank,
                                                                         self.n_channels_)
        eigval, eigvec = np.linalg.eigh(reference.mean(axis=0))
        self.projection_ = eigvec[:, np.argsort(eigval)[::-1][:rank]]
        projected = self._project(reference)
        assert_positive_definite(projected, f"{self.name} reference covariances")
        self.tangent_ = TangentSpace(metric=self.metric).fit(projected)
        self.n_reference_ = int(len(idx))
        self.rank_ = int(rank)
        return self

    def tangent_vectors(self, upper) -> np.ndarray:
        """Tangent-space vectors (float32) of upper-triangle covariance rows."""
        return self._tangent_vectors(upper)

    def fit(self, data: DecoderInput, y: np.ndarray) -> TangentSpaceRidge:
        self.fit_mapping(data.x)
        vectors = self._tangent_vectors(data.x)
        return super().fit(DecoderInput(vectors, data.subjects, data.covariates), y)

    def predict(self, data: DecoderInput) -> np.ndarray:
        vectors = self._tangent_vectors(data.x)
        return super().predict(DecoderInput(vectors, data.subjects, data.covariates))

    def describe(self) -> dict:
        info = super().describe()
        info.update({"metric": self.metric, "n_reference": getattr(self, "n_reference_", None),
                     "projection_rank": getattr(self, "rank_", None),
                     "n_channels": getattr(self, "n_channels_", None),
                     "backend": "cuda" if self._backend_device() is not None else "cpu"})
        return info
