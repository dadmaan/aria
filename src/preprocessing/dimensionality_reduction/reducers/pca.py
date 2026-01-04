"""PCA dimensionality reduction strategy."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA

from .base import DimensionalityReducer
from ..registry import register_reducer


@register_reducer("pca")
class PCAReducer(DimensionalityReducer):
    """Apply PCA to a numeric feature matrix."""

    method = "pca"

    def __init__(
        self,
        *,
        n_components: int,
        random_state: Optional[int],
        **kwargs,
    ) -> None:
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)

    def fit_transform(self, features: np.ndarray) -> tuple[np.ndarray, Dict[str, object]]:
        pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            **self.extra_params,
        )
        embedding = pca.fit_transform(features)
        summary = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "singular_values": pca.singular_values_.tolist(),
        }
        return embedding, summary


__all__ = ["PCAReducer"]
