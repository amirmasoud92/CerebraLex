"""Simulate trial-level data with a planted cross-level trait-by-state interaction (T2.5 power)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from state_models_and_trait_moderation.mixed_models import TRAITS


def simulate_trait_state_data(
    n_subjects: int = 300,
    trials_per_subject: int = 60,
    interaction: dict[str, float] | None = None,
    state_slope: float = 0.15,
    trait_main: float = 0.03,
    subject_sd: float = 0.25,
    slope_sd: float = 0.08,
    residual_sd: float = 0.35,
    seed: int = 0,
) -> pd.DataFrame:
    """Standardised log-RT outcome with subject random intercept and state slope."""
    rng = np.random.default_rng(seed)
    interaction = interaction or {}
    subjects = [f"S{i:05d}" for i in range(n_subjects)]
    traits = rng.standard_normal((n_subjects, len(TRAITS)))
    ages = rng.uniform(5, 21, size=n_subjects)
    sex = rng.integers(0, 2, size=n_subjects)
    ehq = rng.normal(60, 40, size=n_subjects)
    releases = rng.choice([f"R{i}" for i in range(1, 10)], size=n_subjects)
    u = rng.normal(0, subject_sd, size=n_subjects)
    v = rng.normal(0, slope_sd, size=n_subjects)
    rows = []
    for i, subject in enumerate(subjects):
        state = rng.standard_normal(trials_per_subject)
        artifact = rng.uniform(0, 0.3)
        slope = state_slope + v[i] + sum(interaction.get(t, 0.0) * traits[i, k]
                                         for k, t in enumerate(TRAITS))
        y = (u[i] + slope * state + trait_main * traits[i].sum() + 0.01 * (ages[i] - 12)
             + rng.normal(0, residual_sd, size=trials_per_subject))
        for t in range(trials_per_subject):
            rows.append({"subject": subject, "release": releases[i], "age": ages[i],
                         "sex_male": sex[i], "ehq_total": ehq[i], "artifact_fraction": artifact,
                         "trial_index": t, "state": state[t], "log_rt_centred": y[t],
                         **{name: traits[i, k] for k, name in enumerate(TRAITS)}})
    return pd.DataFrame(rows)
