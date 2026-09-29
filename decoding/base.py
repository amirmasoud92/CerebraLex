"""Decoder interface. Inputs are explicit so no model can reach for data it was not given."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DecoderInput:
    """One fold's view of the data; ``x`` is whatever representation the decoder consumes."""

    x: np.ndarray
    subjects: np.ndarray
    covariates: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.subjects)


class Decoder(ABC):
    name: str = "decoder"

    @abstractmethod
    def fit(self, data: DecoderInput, y: np.ndarray) -> Decoder: ...

    @abstractmethod
    def predict(self, data: DecoderInput) -> np.ndarray: ...

    def describe(self) -> dict:
        return {"name": self.name}
