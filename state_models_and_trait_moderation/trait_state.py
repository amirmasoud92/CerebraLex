"""Two-stage estimation of trait moderation of the within-child state-behaviour coupling (H2).

Stage 1 estimates each child's slope of the outcome on the state by least squares within the
child (with trial index as a within-child covariate). Stage 2 is a random-effects meta-regression
of the slopes on the standardised traits and child-level covariates, with weights
1 / (SE_i^2 + tau^2) and tau^2 from the method-of-moments estimator for meta-regression
(DerSimonian and Laird 1986; Thompson and Sharp 1999). Inference on the trait coefficients comes
from permuting the trait vectors jointly across children, which keeps each child's behaviour and
EEG together and needs no distributional assumption for the slopes. The permutation statistic is
the studentised coefficient (estimate / SE), which stays pivotal when traits correlate with the
covariates (Winkler et al. 2014).

Two rescalings of the stage 1 slopes serve as robustness estimates: the first-order log-odds
slope of a binary outcome (removes the dependence of a probability-scale slope on the child's
base rate) and the Fisher-z partial correlation (removes the dependence of a slope on the
child's outcome variability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from task_and_trials.participants import PHENOTYPE_COLUMNS

TRAIT_COLUMNS = tuple(PHENOTYPE_COLUMNS)


def load_traits(participants: pd.DataFrame) -> pd.DataFrame:
    """Standardised trait scores of children with all four dimensions (complete cases)."""
    frame = participants.drop_duplicates("participant_id").copy()
    frame["subject"] = frame["participant_id"].astype(str).str.replace("^sub-", "", regex=True)
    traits = frame[["subject", *TRAIT_COLUMNS]].copy()
    for column in TRAIT_COLUMNS:
        traits[column] = pd.to_numeric(traits[column], errors="coerce")
    traits = traits.dropna().reset_index(drop=True)
    for column in TRAIT_COLUMNS:
        traits[column] = (traits[column] - traits[column].mean()) / traits[column].std()
    return traits


def child_slopes(table: pd.DataFrame, outcome: str, state: str,
                 within: tuple[str, ...] = ("trial_index",), min_trials: int = 10) -> pd.DataFrame:
    """Per-child least-squares slope of ``outcome`` on ``state`` with its standard error."""
    rows = []
    for subject, part in table.groupby("subject", sort=True):
        part = part.dropna(subset=[outcome, state, *within])
        n = len(part)
        if n < max(min_trials, len(within) + 3):
            continue
        x = np.column_stack([np.ones(n), part[state].to_numpy(float), *(part[c].to_numpy(float) for c in within)])
        y = part[outcome].to_numpy(float)
        if np.ptp(y) == 0:          # constant outcome: slope undefined (rounding would give SE ~1e-17)
            continue
        xtx = x.T @ x
        if np.linalg.matrix_rank(xtx) < x.shape[1]:
            continue
        beta = np.linalg.solve(xtx, x.T @ y)
        resid = y - x @ beta
        sigma2 = float(resid @ resid) / (n - x.shape[1])
        se = float(np.sqrt(sigma2 * np.linalg.inv(xtx)[1, 1]))
        if se > 0:
            rows.append({"subject": subject, "slope": float(beta[1]), "se": se, "n_trials": n,
                         "outcome_mean": float(y.mean())})
    return pd.DataFrame(rows)


def constant_outcome_children(table: pd.DataFrame, outcome: str) -> pd.Index:
    """Children whose outcome never varies (for lapse: no lapse, or only lapses); their slope and
    its standard error are undefined, so ``child_slopes`` leaves them out."""
    spread = table.dropna(subset=[outcome]).groupby("subject")[outcome].agg(["min", "max"])
    return spread.index[spread["min"] == spread["max"]]


def logit_scale_slopes(slopes: pd.DataFrame) -> pd.DataFrame:
    """First-order log-odds slopes from linear-probability slopes of a binary outcome.

    Under P(y = 1) = logistic(a + b * state), dP/dstate = b * p * (1 - p), so the least-squares
    slope divided by p(1 - p) at the child's outcome rate approximates b; the SE is scaled alike.
    """
    scale = slopes["outcome_mean"] * (1.0 - slopes["outcome_mean"])
    if (scale <= 0).any():
        raise ValueError("logit scaling needs every child to have outcome rate strictly between 0 and 1")
    return slopes.assign(slope=slopes["slope"] / scale, se=slopes["se"] / scale)


def partial_correlation_slopes(slopes: pd.DataFrame, n_within: int = 1) -> pd.DataFrame:
    """Fisher-z partial correlation of outcome and state within child, with its standard error.

    r = t / sqrt(t^2 + df), t = slope / SE and df = n - n_within - 2 (the residual degrees of
    freedom of the stage 1 regression); SE(z) = 1 / sqrt(n - n_within - 3).
    """
    t = slopes["slope"] / slopes["se"]
    df = slopes["n_trials"] - n_within - 2
    r = t / np.sqrt(t ** 2 + df)
    return slopes.assign(slope=np.arctanh(r), se=1.0 / np.sqrt(slopes["n_trials"] - n_within - 3))


@dataclass(frozen=True)
class MetaRegression:
    coef: np.ndarray
    se: np.ndarray
    tau2: float
    names: tuple[str, ...]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"term": self.names, "estimate": self.coef, "se": self.se})


def meta_regression(slopes: np.ndarray, se: np.ndarray, design: np.ndarray,
                    names: tuple[str, ...]) -> MetaRegression:
    """Random-effects meta-regression with the method-of-moments between-child variance."""
    se = np.asarray(se, dtype=float)
    if not np.isfinite(se).all() or (se <= 0).any():
        raise ValueError("meta-regression needs finite, positive standard errors for every child")
    v = se ** 2
    y = np.asarray(slopes, dtype=float)
    x = np.asarray(design, dtype=float)
    k, p = x.shape
    w = 1.0 / v
    xtw = x.T * w
    cov_fixed = np.linalg.inv(xtw @ x)
    beta_fixed = cov_fixed @ (xtw @ y)
    resid = y - x @ beta_fixed
    q = float((w * resid ** 2).sum())
    trace = float(w.sum() - np.trace(cov_fixed @ ((x.T * w ** 2) @ x)))
    tau2 = max(0.0, (q - (k - p)) / trace) if trace > 0 else 0.0
    w_star = 1.0 / (v + tau2)
    xtw_star = x.T * w_star
    cov = np.linalg.inv(xtw_star @ x)
    beta = cov @ (xtw_star @ y)
    return MetaRegression(beta, np.sqrt(np.diag(cov)), tau2, tuple(names))


def permutation_pvalues(slopes: pd.DataFrame, traits: pd.DataFrame, covariates: pd.DataFrame,
                        n_permutations: int = 5000, seed: int = 12345) -> pd.DataFrame:
    """Trait coefficients of the meta-regression and two-sided permutation p-values.

    ``slopes``: subject, slope, se; ``traits``: subject and the trait columns; ``covariates``:
    subject and child-level covariate columns (numeric, already encoded). Trait vectors are
    permuted jointly across children; covariates, slopes and their SEs stay with the child.
    """
    merged = slopes.merge(traits, on="subject").merge(covariates, on="subject").dropna()
    trait_cols = [c for c in traits.columns if c != "subject"]
    cov_cols = [c for c in covariates.columns if c != "subject"]
    base = np.column_stack([np.ones(len(merged)), merged[cov_cols].to_numpy(float)])
    t = merged[trait_cols].to_numpy(float)
    names = ("intercept", *cov_cols, *trait_cols)

    def fit(trait_matrix: np.ndarray) -> MetaRegression:
        return meta_regression(merged["slope"].to_numpy(), merged["se"].to_numpy(),
                               np.column_stack([base, trait_matrix]), names)

    k = len(trait_cols)
    observed = fit(t)
    obs_z = observed.coef[-k:] / observed.se[-k:]
    if not np.isfinite(obs_z).all():
        raise ValueError(f"non-finite trait statistics {obs_z}: degenerate meta-regression")
    rng = np.random.default_rng(seed)
    exceed = np.zeros(k)
    for _ in range(n_permutations):
        perm = fit(t[rng.permutation(len(t))])
        perm_z = perm.coef[-k:] / perm.se[-k:]
        if not np.isfinite(perm_z).all():
            raise ValueError("non-finite permuted trait statistic: degenerate meta-regression")
        exceed += np.abs(perm_z) >= np.abs(obs_z)
    return pd.DataFrame({"trait": trait_cols, "estimate": observed.coef[-k:], "se": observed.se[-k:],
                         "z": obs_z, "p_permutation": (1.0 + exceed) / (1.0 + n_permutations),
                         "tau2": observed.tau2, "n_children": len(merged),
                         "n_permutations": n_permutations})
