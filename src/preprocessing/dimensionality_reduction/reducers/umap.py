"""UMAP dimensionality reduction strategy."""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import umap

from .base import DimensionalityReducer
from ..registry import register_reducer


@register_reducer("umap")
class UMAPReducer(DimensionalityReducer):
    """Apply UMAP to a numeric feature matrix.

    UMAP (Uniform Manifold Approximation and Projection) is a dimensionality
    reduction technique that preserves both local and global structure better
    than t-SNE while being significantly faster. It is particularly well-suited
    for visualizing high-dimensional data and can scale to larger datasets.

    The reducer wraps the umap-learn package and supports all standard UMAP
    parameters via the extra_params mechanism.
    """

    method = "umap"

    def __init__(
        self,
        *,
        n_components: int,
        random_state: Optional[int],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        **kwargs,
    ) -> None:
        """Initialize the UMAP reducer.

        Args:
                n_components: Number of dimensions for the projection.
                random_state: Random seed for reproducibility.
                n_neighbors: Number of neighboring points used in manifold approximation.
                        Larger values result in more global structure preservation, smaller
                        values preserve more local structure. Default: 15.
                min_dist: Minimum distance between points in the low-dimensional
                        representation. Smaller values result in more clustered embeddings.
                        Default: 0.1.
                metric: Distance metric to use. Supports any metric from scipy.spatial.distance
                        or custom callables. Common options: "euclidean", "manhattan", "cosine",
                        "correlation". Default: "euclidean".
                **kwargs: Additional UMAP parameters (e.g., n_epochs, learning_rate,
                        spread, set_op_mix_ratio). Passed directly to umap.UMAP().
        """
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self.n_neighbors = int(n_neighbors)
        self.min_dist = float(min_dist)
        self.metric = str(metric)

    def fit_transform(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, Dict[str, object]]:
        """Apply UMAP projection to the feature matrix.

        Args:
                features: Input feature matrix, shape (n_samples, n_features).

        Returns:
                A tuple containing:
                        - embedding: Projected feature matrix, shape (n_samples, n_components).
                        - summary: Dictionary with projection metadata including n_neighbors used,
                          embedding coordinate ranges, and optimization info.
        """
        n_samples = features.shape[0]

        # Auto-adjust n_neighbors if dataset is too small
        # UMAP requires n_neighbors < n_samples
        n_neighbors = self.n_neighbors
        if n_samples <= n_neighbors:
            n_neighbors = max(2, n_samples - 1)

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            **self.extra_params,
        )

        # Suppress UMAP warning about n_jobs being overridden when random_state is set
        # This is expected behavior and not an issue
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value .* overridden to 1 by setting random_state",
                category=UserWarning,
            )
            embedding = reducer.fit_transform(features)

        # Collect summary statistics
        summary = {
            "n_neighbors": n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "embedding_min": float(np.min(embedding)),
            "embedding_max": float(np.max(embedding)),
            "embedding_mean": float(np.mean(embedding)),
            "embedding_std": float(np.std(embedding)),
        }

        # Add n_epochs if available (UMAP tracks this internally)
        if hasattr(reducer, "n_epochs") and reducer.n_epochs is not None:
            summary["n_epochs"] = int(reducer.n_epochs)

        return embedding, summary


__all__ = ["UMAPReducer"]
