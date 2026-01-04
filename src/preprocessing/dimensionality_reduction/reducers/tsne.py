"""t-SNE dimensionality reduction strategy."""

from __future__ import annotations

from typing import Dict, Literal, Optional, cast

import numpy as np
from sklearn.manifold import TSNE

from .base import DimensionalityReducer
from ..registry import register_reducer


InitMode = Literal["random", "pca"]


@register_reducer("tsne")
class TSNEReducer(DimensionalityReducer):
	"""Apply t-SNE to a numeric feature matrix."""

	method = "tsne"

	def __init__(
		self,
		*,
		n_components: int,
		random_state: Optional[int],
		perplexity: float = 30.0,
		learning_rate: float = 200.0,
		init: Optional[str] = None,
		**kwargs,
	) -> None:
		super().__init__(n_components=n_components, random_state=random_state, **kwargs)
		self.perplexity = float(perplexity)
		self.learning_rate = float(learning_rate)
		self.init = init

	def fit_transform(self, features: np.ndarray) -> tuple[np.ndarray, Dict[str, object]]:
		n_samples = features.shape[0]
		perplexity = min(self.perplexity, (n_samples - 1) / 3) if n_samples > 1 else self.perplexity
		if perplexity < 1:
			perplexity = max(1.0, min(self.perplexity, float(n_samples - 1)))

		if self.init is not None:
			init_value = self.init.lower()
			if init_value not in {"random", "pca"}:
				raise ValueError("t-SNE init must be 'random' or 'pca'.")
			init = cast(InitMode, init_value)
		else:
			default_init = "pca" if self.n_components > 1 else "random"
			init = cast(InitMode, default_init)

		tsne = TSNE(
			n_components=self.n_components,
			perplexity=perplexity,
			learning_rate=self.learning_rate,
			random_state=self.random_state,
			init=init,
			**self.extra_params,
		)
		embedding = tsne.fit_transform(features)
		summary = {
			"computed_perplexity": perplexity,
			"kl_divergence_": getattr(tsne, "kl_divergence_", None),
			"n_iter_": getattr(tsne, "n_iter_", None),
		}
		return embedding, summary


__all__ = ["TSNEReducer"]
