"""Reducer implementations for dimensionality reduction."""

from .base import DimensionalityReducer
from .experimental import ExperimentalReducer, MSPTSNEReducer  # noqa: F401
from .pca import PCAReducer  # noqa: F401
from .tsne import TSNEReducer  # noqa: F401

# UMAP is optional
try:
    from .umap import UMAPReducer  # noqa: F401

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    UMAPReducer = None  # type: ignore[misc,assignment]

__all__ = [
    "DimensionalityReducer",
    "ExperimentalReducer",
    "MSPTSNEReducer",
    "PCAReducer",
    "TSNEReducer",
    "UMAPReducer",
    "HAS_UMAP",
]
