"""Deterministic seeding across Python, NumPy and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, object]:
    """Seed all random number generators in use and return a record of what was set.

    With ``deterministic=True`` PyTorch is configured for reproducible kernels; this
    can slow some GPU operations and is the default for scientific runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    record: dict[str, object] = {"seed": seed, "deterministic": deterministic, "torch": False}
    try:
        import torch
    except ImportError:
        return record
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    record["torch"] = True
    return record
