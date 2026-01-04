"""Configuration objects for dimensionality reduction strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import PreprocessorConfig


@dataclass
class DimensionalityReductionConfig(PreprocessorConfig):
    """Configures a dimensionality reduction preprocessing run.

    The configuration keeps method-agnostic options at the top level and
    delegates algorithm-specific tuning knobs to ``method_params`` so that the
    orchestration layer can remain generic.
    """

    input_features: Path = Path("artifacts/features/raw/latest/features_numeric.csv")
    output_root: Path = Path("artifacts/features/tsne")
    method: str = "tsne"
    n_components: int = 2
    random_state: Optional[int] = 42
    standardise: bool = True
    metadata_columns: List[str] = field(
        default_factory=lambda: ["track_id", "metadata_index"]
    )
    method_params: Dict[str, Any] = field(default_factory=dict)

    def to_serialisable_dict(self) -> Dict[str, Any]:
        data = super().to_serialisable_dict()
        data.update(
            {
                "input_features": str(self.input_features),
                "method": self.method,
                "n_components": self.n_components,
                "random_state": self.random_state,
                "standardise": self.standardise,
                "metadata_columns": list(self.metadata_columns),
                "method_params": self.method_params,
            }
        )
        return data


__all__ = ["DimensionalityReductionConfig"]
