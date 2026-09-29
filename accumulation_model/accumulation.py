"""Formal accumulation model of contrast change detection behaviour.

Two-boundary diffusion (bounds at +-a, unbiased start, unit noise) whose drift towards the correct
side follows the target's contrast difference: mu(s) = k * c(s), where s is time since accumulation
started (the encoding delay te after the target) and c rises linearly from 0 to 1 over 1.6 s and
returns to 0 over the next 0.8 s. The response follows the bound crossing after a fixed motor time.
Trial-to-trial variability: accumulation rate (k_i ~ N(k, sv^2)) and onset (te_i ~ U(te - st/2,
te + st/2)). Contaminants: lapses (no response) and fast guesses (uniform RT on [0, 0.65] s, random
side). Responses after the 2.5 s deadline are misses.

First-passage densities come from the Fokker-Planck equation (Crank-Nicolson with implicit start-up
steps); all rate quadrature nodes are solved as one block-tridiagonal system per time step. The
absorbed mass per step is exact (mass lost from the grid) and split between the bounds by their
one-sided boundary fluxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize

RAMP_UP_S, RAMP_DOWN_S = 1.6, 0.8
MOTOR_S = 0.1
DEADLINE_S = 2.5
GUESS_MAX_S = 0.65
TE_MIN_S = 0.02
PARAMS = ("k", "a", "te", "sv", "st", "lapse", "guess")
BOUNDS = {"k": (0.0, 25.0), "a": (0.2, 4.0), "te": (0.05, 1.5), "sv": (0.0, 12.0), "st": (0.0, 1.5),
          "lapse": (0.0, 0.95), "guess": (0.0, 0.5)}
MODELS = {"full": (), "rate": ("st",), "onset": ("sv",), "none": ("sv", "st")}   # parameters fixed at 0


def ramp(s: np.ndarray) -> np.ndarray:
    """Contrast difference of the CCD target, s seconds after its onset (0 to 1)."""
    s = np.asarray(s, dtype=float)
    up = np.clip(s / RAMP_UP_S, 0.0, 1.0)
    down = np.clip(1.0 - (s - RAMP_UP_S) / RAMP_DOWN_S, 0.0, 1.0)
    return np.where(s <= RAMP_UP_S, up, down) * (s >= 0)


@dataclass
class Grid:
    dt: float = 0.01          # s
    n_x: int = 61             # grid points including both absorbing bounds
    s_max: float = DEADLINE_S - MOTOR_S - TE_MIN_S
    startup_steps: int = 4    # implicit Euler half-steps before Crank-Nicolson (Rannacher start-up)

    @property
    def times(self) -> np.ndarray:
        n = int(round(self.s_max / self.dt))
        return (np.arange(n) + 1) * self.dt


def first_passage(k: np.ndarray, a: float, grid: Grid | None = None, drift_profile=ramp) -> tuple[np.ndarray, ...]:
    """First-passage densities (per second) at the upper (correct) and lower bound, and the survival
    probability, for drift gains ``k`` (shape (B,)). Returns (times, up, low, survival): ``times`` are the
    ends of the time steps; ``up`` and ``low`` are the mean densities over each step, which belong to the
    step midpoints (times - dt / 2); ``survival`` is the probability of no crossing at ``times``."""
    grid = grid or Grid()
    k = np.atleast_1d(np.asarray(k, dtype=float))
    b, n = len(k), grid.n_x - 2                      # interior points
    dx = 2.0 * a / (grid.n_x - 1)
    d = 0.5                                          # diffusion coefficient for unit noise
    p = np.zeros((b, n))
    p[:, n // 2] = 1.0 / dx                          # delta at the start point (grid centre)
    times = grid.times
    up, low, surv = np.zeros((b, len(times))), np.zeros((b, len(times))), np.zeros((b, len(times)))
    mass = np.ones(b)

    def operator(mu: np.ndarray):
        lower = d / dx ** 2 + mu / (2 * dx)          # coefficient of p[j-1]
        diag = np.full(b, -2 * d / dx ** 2)
        upper = d / dx ** 2 - mu / (2 * dx)          # coefficient of p[j+1]
        return lower, diag, upper

    def banded(lower, diag, upper, factor):
        """Banded form (3, B*n) of I - factor * L for every batch member (blocks decoupled)."""
        ab = np.zeros((3, b * n))
        ab[1] = np.repeat(1.0 - factor * diag, n)
        sup = np.repeat(-factor * upper, n)
        sub = np.repeat(-factor * lower, n)
        sup[n - 1::n] = 0.0                          # last row of a block: no right neighbour in the block
        sub[0::n] = 0.0                              # first row of a block: no left neighbour in the block
        ab[0, 1:] = sup[:-1]
        ab[2, :-1] = sub[1:]
        return ab

    def apply(lower, diag, upper, x, factor):
        """x + factor * L x for every batch member (x: (B, n))."""
        out = x * (1.0 + factor * diag[:, None])
        out[:, 1:] += factor * lower[:, None] * x[:, :-1]
        out[:, :-1] += factor * upper[:, None] * x[:, 1:]
        return out

    # Summing the discrete equation over the interior telescopes to d(mass)/dt = -dx (alpha p[-1] + gamma p[0]),
    # with alpha = lower (moves mass from the last interior point into the upper bound) and gamma = upper (from
    # the first interior point into the lower bound): the absorbed mass per bound is exact for the scheme.
    s_prev = 0.0
    for i, s in enumerate(times):
        mu_prev, mu_now = k * drift_profile(s_prev), k * drift_profile(s)
        sub_steps = 2 if i < grid.startup_steps // 2 else 1
        h = (s - s_prev) / sub_steps
        into_up, into_low = np.zeros(b), np.zeros(b)
        for j in range(sub_steps):
            if sub_steps == 2:                       # implicit Euler half-steps (damps the delta start)
                mu = mu_prev + (mu_now - mu_prev) * (j + 1) / 2
                alpha, _, gamma = operator(mu)
                ab = banded(*operator(mu), h)
                p = solve_banded((1, 1), ab, p.reshape(-1)).reshape(b, n)
                into_up += h * dx * alpha * p[:, -1]
                into_low += h * dx * gamma * p[:, 0]
            else:                                    # Crank-Nicolson
                alpha_p, _, gamma_p = operator(mu_prev)
                alpha_n, _, gamma_n = operator(mu_now)
                into_up += 0.5 * h * dx * (alpha_p * p[:, -1])
                into_low += 0.5 * h * dx * (gamma_p * p[:, 0])
                rhs = apply(*operator(mu_prev), p, h / 2)
                ab = banded(*operator(mu_now), h / 2)
                p = solve_banded((1, 1), ab, rhs.reshape(-1)).reshape(b, n)
                into_up += 0.5 * h * dx * (alpha_n * p[:, -1])
                into_low += 0.5 * h * dx * (gamma_n * p[:, 0])
        up[:, i] = np.maximum(into_up, 0.0) / (s - s_prev)
        low[:, i] = np.maximum(into_low, 0.0) / (s - s_prev)
        mass = mass - into_up - into_low
        surv[:, i] = np.maximum(mass, 0.0)
        s_prev = s
    return times, up, low, surv


@dataclass
class ChildData:
    rt: np.ndarray            # s, responses only
    correct: np.ndarray       # bool, responses only
    n_miss: int

    @classmethod
    def from_trials(cls, rt_s: np.ndarray, correct: np.ndarray) -> ChildData:
        rt_s = np.asarray(rt_s, dtype=float)
        responded = np.isfinite(rt_s) & (rt_s > 0) & (rt_s <= DEADLINE_S)
        return cls(rt=rt_s[responded], correct=np.asarray(correct, dtype=bool)[responded],
                   n_miss=int((~responded).sum()))


@dataclass
class Quadrature:
    n_sv: int = 5
    n_st: int = 5
    grid: Grid = field(default_factory=Grid)


def _mixture(params: dict, quad: Quadrature):
    """Rate-mixed densities and survival on the accumulation-time grid."""
    if params["sv"] > 0:
        x, w = np.polynomial.hermite_e.hermegauss(quad.n_sv)
        ks = params["k"] + params["sv"] * x
        w = w / w.sum()
    else:
        ks, w = np.array([params["k"]]), np.array([1.0])
    times, up, low, surv = first_passage(ks, params["a"], quad.grid)
    return times, w @ up, w @ low, w @ surv


def _onset_nodes(params: dict, quad: Quadrature):
    if params["st"] > 0:
        x, w = np.polynomial.legendre.leggauss(quad.n_st)
        return params["te"] + 0.5 * params["st"] * x, w / w.sum()
    return np.array([params["te"]]), np.array([1.0])


def response_densities(params: dict, rt: np.ndarray, quad: Quadrature | None = None):
    """Density of each (rt, bound) from the diffusion part, and the diffusion miss probability."""
    quad = quad or Quadrature()
    times, up, low, surv = _mixture(params, quad)
    te_nodes, te_w = _onset_nodes(params, quad)
    t_mid = np.r_[0.0, times - quad.grid.dt / 2]       # densities belong to the step midpoints
    t_end = np.r_[0.0, times]
    up0, low0, surv0 = np.r_[0.0, up], np.r_[0.0, low], np.r_[1.0, surv]
    f_up = np.zeros(len(rt))
    f_low = np.zeros(len(rt))
    p_nocross = 0.0
    for te, wt in zip(te_nodes, te_w, strict=True):
        s = rt - MOTOR_S - te
        f_up += wt * np.interp(s, t_mid, up0, left=0.0, right=0.0)
        f_low += wt * np.interp(s, t_mid, low0, left=0.0, right=0.0)
        p_nocross += wt * np.interp(DEADLINE_S - MOTOR_S - te, t_end, surv0, left=1.0, right=surv0[-1])
    return f_up, f_low, p_nocross


def neg_loglik(params: dict, data: ChildData, quad: Quadrature | None = None) -> float:
    if params["te"] - params["st"] / 2 < TE_MIN_S:
        return 1e10
    f_up, f_low, p_nocross = response_densities(params, data.rt, quad)
    lapse, guess = params["lapse"], params["guess"]
    if lapse + guess >= 0.99:
        return 1e10
    diffusion = np.where(data.correct, f_up, f_low)
    guess_density = np.where(data.rt <= GUESS_MAX_S, 0.5 / GUESS_MAX_S, 0.0)
    like = (1 - lapse - guess) * diffusion + guess * guess_density
    miss_like = lapse + (1 - lapse - guess) * p_nocross
    ll = np.log(np.maximum(like, 1e-12)).sum() + data.n_miss * np.log(max(miss_like, 1e-12))
    return float(-ll)


def fit(data: ChildData, model: str = "full", starts: list[dict] | None = None,
        quad: Quadrature | None = None, maxiter: int = 200) -> dict:
    """Maximum likelihood fit of one child; ``model`` fixes sv and/or st at 0."""
    quad = quad or Quadrature()
    free = [p for p in PARAMS if p not in MODELS[model]]
    bounds = [BOUNDS[p] for p in free]
    starts = starts or [default_start(data)]
    best = None
    for start in starts:
        x0 = np.array([np.clip(start[p], *BOUNDS[p]) for p in free])

        def objective(x):
            params = dict.fromkeys(PARAMS, 0.0) | dict(zip(free, x, strict=True))
            return neg_loglik(params, data, quad)

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "eps": 1e-4})
        if best is None or res.fun < best.fun:
            best = res
    params = dict.fromkeys(PARAMS, 0.0) | dict(zip(free, best.x, strict=True))
    n_obs = len(data.rt) + data.n_miss
    return {**params, "model": model, "nll": float(best.fun), "n_params": len(free), "n_obs": n_obs,
            "bic": float(2 * best.fun + len(free) * np.log(n_obs)), "converged": bool(best.success)}


def fit_all(data: ChildData, quad: Quadrature | None = None, maxiter: int = 200) -> dict[str, dict]:
    """The four models with nested starts, so that no model ends with a worse likelihood than a model it nests: none from two starts; rate and onset
    from the default start and from the none solution with the extra parameter at 0; full from the default start
    and from the rate and onset solutions with the other parameter at 0."""
    base = default_start(data)
    alternative = base | {"k": base["k"] * 0.5, "a": 1.3}
    out = {"none": fit(data, "none", [base, alternative], quad, maxiter)}
    nested = {p: out["none"][p] for p in PARAMS}
    out["rate"] = fit(data, "rate", [base, nested | {"sv": 0.0}], quad, maxiter)
    out["onset"] = fit(data, "onset", [base, nested | {"st": 0.0}], quad, maxiter)
    rate_p = {p: out["rate"][p] for p in PARAMS}
    onset_p = {p: out["onset"][p] for p in PARAMS}
    out["full"] = fit(data, "full", [base, rate_p | {"st": 0.0}, onset_p | {"sv": 0.0}], quad, maxiter)
    return out


def default_start(data: ChildData) -> dict:
    """A start from simple summaries: bound and gain matched loosely to the median RT."""
    med = float(np.median(data.rt)) if len(data.rt) else 1.6
    miss = data.n_miss / max(len(data.rt) + data.n_miss, 1)
    return {"k": float(np.clip(3.0 * 1.6 / max(med - 0.35, 0.3), 0.5, 15.0)), "a": 1.0, "te": 0.3,
            "sv": 1.0, "st": 0.2, "lapse": float(np.clip(miss * 0.5, 0.01, 0.5)), "guess": 0.02}


def simulate(params: dict, n: int, rng: np.random.Generator, dt: float = 0.001,
             keep_paths: bool = False, path_window_s: float = 0.6) -> dict:
    """Simulate n trials (Euler-Maruyama). Returns rt (nan for misses) and correct; with keep_paths,
    the decision variable towards the chosen bound over the ``path_window_s`` before each response
    (held at the bound between the crossing and the response), sampled every dt."""
    k_i = params["k"] + params["sv"] * rng.standard_normal(n)
    te_i = params["te"] + params["st"] * (rng.random(n) - 0.5)
    lapse = rng.random(n) < params["lapse"]
    guess = (~lapse) & (rng.random(n) < params["guess"] / max(1 - params["lapse"], 1e-9))
    steps = int(round((DEADLINE_S - MOTOR_S) / dt))
    x = np.zeros(n)
    crossed = np.full(n, -1)
    choice = np.zeros(n, dtype=bool)
    n_hist = int(round(path_window_s / dt)) + 1
    history = np.zeros((n, steps + 1)) if keep_paths else None
    for step in range(1, steps + 1):
        t = step * dt
        active = crossed < 0
        s = t - te_i
        mu = np.where(s > 0, k_i * ramp(s), 0.0)
        noise = np.sqrt(dt) * rng.standard_normal(n)
        x = np.where(active & (s > 0), x + mu * dt + noise, x)
        hit_up, hit_low = active & (x >= params["a"]), active & (x <= -params["a"])
        crossed[hit_up | hit_low] = step
        choice[hit_up] = True
        x = np.clip(x, -params["a"], params["a"])
        if keep_paths:
            history[:, step] = x
    decision = np.where(crossed > 0, crossed * dt, np.nan)
    rt = decision + MOTOR_S
    rt[lapse] = np.nan
    rt[rt > DEADLINE_S] = np.nan
    guess_rt = rng.random(n) * GUESS_MAX_S
    rt[guess] = guess_rt[guess]
    correct = np.where(guess, rng.random(n) < 0.5, choice)
    out = {"rt": rt, "correct": correct.astype(bool), "diffusion": ~(lapse | guess)}
    if keep_paths:
        motor = int(round(MOTOR_S / dt))
        paths = np.full((n, n_hist), np.nan)
        for i in np.flatnonzero(np.isfinite(rt) & out["diffusion"]):
            end = crossed[i] + motor                   # response sample
            hold = np.r_[history[i, :crossed[i] + 1], np.full(motor, history[i, crossed[i]])]
            seg = hold[max(0, end - n_hist + 1):end + 1]
            paths[i, n_hist - len(seg):] = seg * (1 if choice[i] else -1)
        out["paths"] = paths
    return out
