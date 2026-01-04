"""Abstract base classes for dimensionality reduction strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class DimensionalityReducer(ABC):
    """Shared interface for dimensionality reduction strategies.

    Reducers operate on fully-prepared numeric feature matrices and return the
    embedding along with auxiliary metadata (e.g. training statistics).
    """

    #: Canonical string identifier for the reducer. Subclasses must override.
    method: str

    def __init__(self, *, n_components: int, random_state: int | None, **kwargs) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.extra_params = kwargs

    @abstractmethod
    def fit_transform(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        """Return the low-dimensional embedding and optional summary metrics."""


__all__ = ["DimensionalityReducer"]
