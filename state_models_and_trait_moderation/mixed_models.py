"""Trial-level mixed-effects regression of behaviour on latent state with trait moderation.

Model (specification T2.5): outcome ~ state + traits + trait:state + f(age) + sex + handedness +
artifact + trial index + release, with a subject random intercept and a subject random slope
for the state feature. statsmodels' MixedLM supports one grouping factor, so release enters as
fixed effects (nine levels) rather than as a second random intercept; the deviation from the
specification is documented in the results.

Optimisation: the REML likelihood is maximised with L-BFGS and with Powell's method (Nelder-Mead
as well if neither converges), and the converged fit with the highest REML log-likelihood is
reported. On the H2 data L-BFGS stopped 255 log-likelihood units short of the optimum that
Powell and Nelder-Mead both reached, with a small random-slope variance.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)
TRAITS = ("p_factor", "attention", "internalizing", "externalizing")
DEFAULT_COVARIATES = ("sex_male", "ehq_total", "artifact_fraction", "trial_index")
GRADIENT_OPTIMIZERS = ("lbfgs", "bfgs", "cg", "newton")


def natural_spline_basis(x: np.ndarray, df: int = 4) -> np.ndarray:
    """Natural cubic spline basis with ``df`` columns, knots at quantiles (patsy-free)."""
    x = np.asarray(x, dtype=float)
    knots = np.quantile(x[np.isfinite(x)], np.linspace(0, 1, df + 1))
    kn = knots[-1]

    def d(k):
        return (np.clip(x - k, 0, None) ** 3 - np.clip(x - kn, 0, None) ** 3) / (kn - k)

    columns = [x] + [d(k) - d(knots[-2]) for k in knots[:-2]]
    basis = np.column_stack(columns)
    basis = (basis - basis.mean(axis=0)) / (basis.std(axis=0) + 1e-12)
    return basis[:, :df]


@dataclass
class TraitStateResult:
    outcome: str
    state: str
    coefficients: pd.DataFrame            # term, estimate, se, z, p, ci_low, ci_high
    interactions: pd.DataFrame            # rows for trait:state terms only
    n_trials: int
    n_subjects: int
    converged: bool
    random_effects_variance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    optimizer: str = ""
    log_likelihood: float = float("nan")
    attempts: dict = field(default_factory=dict)        # optimizer -> {"converged", "llf"}

    def interaction_estimates(self) -> dict[str, float]:
        return dict(zip(self.interactions["trait"], self.interactions["estimate"], strict=True))


class TraitStateModel:
    """Fit the trait-by-state mixed model with statsmodels MixedLM."""

    def __init__(self, outcome: str = "log_rt_centred", state: str = "state",
                 traits: tuple[str, ...] = TRAITS, covariates: tuple[str, ...] = DEFAULT_COVARIATES,
                 age_df: int = 4, random_slope: bool = True, release_fixed_effects: bool = True,
                 maxiter: int = 200, optimizers: tuple[str, ...] = ("lbfgs", "powell"),
                 fallback_optimizers: tuple[str, ...] = ("nm",), maxiter_derivative_free: int = 5000) -> None:
        self.outcome = outcome
        self.state = state
        self.traits = traits
        self.covariates = covariates
        self.age_df = age_df
        self.random_slope = random_slope
        self.release_fixed_effects = release_fixed_effects
        self.maxiter = maxiter
        self.optimizers = optimizers
        self.fallback_optimizers = fallback_optimizers
        self.maxiter_derivative_free = maxiter_derivative_free

    def design(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
        frame = data.dropna(subset=[self.outcome, self.state, "subject", "age", *self.traits]).copy()
        frame = frame.reset_index(drop=True)
        z = lambda s: (s - s.mean()) / (s.std() + 1e-12)  # noqa: E731
        x = pd.DataFrame(index=frame.index)
        x["state"] = z(frame[self.state].astype(float))
        for trait in self.traits:
            x[trait] = z(frame[trait].astype(float))
            x[f"{trait}:state"] = x[trait] * x["state"]
        spline = natural_spline_basis(frame["age"].to_numpy(), df=self.age_df)
        for i in range(spline.shape[1]):
            x[f"age_s{i + 1}"] = spline[:, i]
        for cov in self.covariates:
            if cov in frame:
                x[cov] = z(frame[cov].astype(float))
        if self.release_fixed_effects and "release" in frame:
            dummies = pd.get_dummies(frame["release"], prefix="release", drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)
        x.insert(0, "intercept", 1.0)
        y = frame[self.outcome].astype(float)
        interaction_terms = [f"{t}:state" for t in self.traits]
        return x, y, frame["subject"], interaction_terms

    def fit(self, data: pd.DataFrame) -> TraitStateResult:
        import statsmodels.api as sm

        x, y, groups, interaction_terms = self.design(data)
        exog_re = x[["intercept", "state"]] if self.random_slope else x[["intercept"]]
        model = sm.MixedLM(y.to_numpy(), x, groups=groups.to_numpy(), exog_re=exog_re)
        notes: list[str] = []
        attempts: dict[str, dict] = {}
        fits = {}
        for method in (*self.optimizers, *self.fallback_optimizers):
            if method in self.fallback_optimizers and any(a["converged"] for a in attempts.values()):
                break
            maxiter = self.maxiter if method in GRADIENT_OPTIMIZERS else self.maxiter_derivative_free
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    fit = model.fit(reml=True, maxiter=maxiter, method=[method])
                except Exception as exc:  # noqa: BLE001 - recorded; the next optimizer is tried
                    notes.append(f"{method} failed: {type(exc).__name__}: {exc}")
                    continue
            fits[method] = fit
            attempts[method] = {"converged": bool(fit.converged), "llf": float(fit.llf)}
            messages = sorted({str(w.message).split(".")[0] for w in caught})
            notes.append(f"{method}: converged {bool(fit.converged)}, REML log-likelihood {fit.llf:.3f}"
                         + (f" ({'; '.join(messages)})" if messages else ""))
        if not fits:
            raise RuntimeError("every optimizer failed: " + " | ".join(notes))
        candidates = [m for m in fits if attempts[m]["converged"]] or list(fits)
        optimizer = max(candidates, key=lambda m: attempts[m]["llf"])
        fit = fits[optimizer]
        converged = attempts[optimizer]["converged"]
        notes.append(f"reported fit: {optimizer}")
        params = fit.params.reindex(x.columns)
        bse = fit.bse.reindex(x.columns)
        zval = params / bse
        pval = fit.pvalues.reindex(x.columns)
        coefficients = pd.DataFrame({
            "term": x.columns, "estimate": params.to_numpy(), "se": bse.to_numpy(),
            "z": zval.to_numpy(), "p": pval.to_numpy(),
            "ci_low": (params - 1.96 * bse).to_numpy(), "ci_high": (params + 1.96 * bse).to_numpy(),
        })
        interactions = coefficients[coefficients["term"].isin(interaction_terms)].copy()
        interactions.insert(0, "trait", interactions["term"].str.replace(":state", "", regex=False))
        re_var = {}
        try:
            cov_re = np.asarray(fit.cov_re)
            re_var = {"intercept": float(cov_re[0, 0])}
            if cov_re.shape[0] > 1:
                re_var["state_slope"] = float(cov_re[1, 1])
            re_var["residual"] = float(fit.scale)
        except Exception:  # noqa: BLE001
            pass
        if not self.random_slope:
            notes.append("random slope disabled")
        notes.append("release modelled as fixed effects (MixedLM supports one grouping factor)")
        return TraitStateResult(self.outcome, self.state, coefficients, interactions.reset_index(drop=True),
                                int(len(y)), int(groups.nunique()), converged, re_var, notes,
                                optimizer, float(fit.llf), attempts)
