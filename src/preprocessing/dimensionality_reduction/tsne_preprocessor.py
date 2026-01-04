"""Compatibility exports for the legacy t-SNE preprocessor API."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Optional

from .config import DimensionalityReductionConfig
from .preprocessor import DimensionalityReductionPreprocessor


@dataclass
class TSNEPreprocessorConfig(DimensionalityReductionConfig):
    """Legacy-compatible configuration wrapper for the generic reducer config."""

    method: str = "tsne"
    perplexity: InitVar[Optional[float]] = None
    learning_rate: InitVar[Optional[float]] = None

    def __post_init__(
        self,
        perplexity: Optional[float],
        learning_rate: Optional[float],
    ) -> None:
        self.method = (self.method or "tsne").lower()

        defaults = {
            "perplexity": 30.0,
            "learning_rate": 200.0,
        }
        provided = {
            "perplexity": perplexity,
            "learning_rate": learning_rate,
        }

        for key, default in defaults.items():
            if key in self.method_params:
                continue  # Already set
            value = provided[key]
            if value is None:
                value = default  # Use default
            self.method_params[key] = value


class TSNEPreprocessor(DimensionalityReductionPreprocessor):
    """Legacy name for the generic dimensionality reduction preprocessor."""

    config: TSNEPreprocessorConfig

    def __init__(self, config: TSNEPreprocessorConfig) -> None:
        if config.method.lower() != "tsne":
            raise ValueError(
                "TSNEPreprocessor requires method='tsne'. Use DimensionalityReductionPreprocessor for other methods."
            )
        super().__init__(config)


__all__ = ["TSNEPreprocessor", "TSNEPreprocessorConfig"]
