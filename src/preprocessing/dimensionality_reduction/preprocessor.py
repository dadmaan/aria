"""Generic preprocessing orchestration for dimensionality reduction."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from ..base import Preprocessor, PreprocessorResult, _serialise_value
from .config import DimensionalityReductionConfig
from .registry import global_reducer_registry

# Ensure default reducers are registered when the module is imported.
from .reducers import experimental as _experimental_reducer  # noqa: F401
from .reducers import pca as _pca_reducer  # noqa: F401
from .reducers import tsne as _tsne_reducer  # noqa: F401

# UMAP is optional
try:
    from .reducers import umap as _umap_reducer  # noqa: F401
except ImportError:
    pass  # UMAP not available

matplotlib.use("Agg")


def _normalise_metadata_columns(columns: Optional[Iterable[str]]) -> List[str]:
    if columns is None:
        return []
    return [str(col) for col in columns]


class DimensionalityReductionPreprocessor(Preprocessor):
    """Run dimensionality reduction using a registered strategy."""

    config: DimensionalityReductionConfig

    def __init__(self, config: DimensionalityReductionConfig) -> None:
        super().__init__(config)
        self.config = config

    def load_inputs(self) -> pd.DataFrame:
        self.logger.info("Loading feature matrix from %s", self.config.input_features)
        if not self.config.input_features.exists():
            raise FileNotFoundError(
                f"Feature matrix not found at {self.config.input_features}"
            )
        df = pd.read_csv(self.config.input_features)
        if df.empty:
            raise ValueError("Feature matrix is empty; cannot compute projection.")
        return df

    def process(self, inputs: pd.DataFrame) -> Dict[str, Any]:
        metadata_cols = _normalise_metadata_columns(self.config.metadata_columns)
        for column in metadata_cols:
            if column not in inputs.columns:
                self.logger.warning(
                    "Metadata column '%s' not found in feature matrix.", column
                )

        numeric_df = inputs.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError(
                "No numeric columns available for dimensionality reduction."
            )

        # Replace infinities with NaN
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

        # Drop columns that are entirely NaN
        numeric_df = numeric_df.dropna(axis=1, how="all")

        # Fill remaining NaNs with column mean (for partially missing columns)
        numeric_df = numeric_df.fillna(numeric_df.mean())

        # If any NaNs remain (shouldn't happen but be safe), fill with 0
        if numeric_df.isna().any().any():
            self.logger.warning(
                "Some NaN values remain after mean imputation; filling with 0"
            )
            numeric_df = numeric_df.fillna(0)

        features = numeric_df.values.astype(np.float64)
        feature_columns = numeric_df.columns.tolist()

        if self.config.standardise:
            # Suppress sklearn warnings about NaN in standardization (we've already handled them)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="invalid value encountered in divide",
                )
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
            self.logger.debug("Applied standardisation to feature matrix.")

        if features.shape[0] <= 1:
            raise ValueError(
                "At least two samples are required for dimensionality reduction."
            )

        method = self.config.method.lower()
        self.logger.info(
            "Running %s reduction with %d components", method, self.config.n_components
        )

        reducer = self._create_reducer(method)
        embedding, model_summary = reducer.fit_transform(features)

        projection_df = self._build_projection_dataframe(
            inputs, metadata_cols, embedding
        )

        return {
            "embedding": embedding,
            "projection_df": projection_df,
            "feature_columns": feature_columns,
            "model_summary": model_summary,
        }

    def save(self, processed: Dict[str, Any]) -> PreprocessorResult:
        embedding: np.ndarray = processed["embedding"]
        projection_df: pd.DataFrame = processed["projection_df"]
        model_summary: Dict[str, Any] = processed["model_summary"]

        artifacts: Dict[str, Path] = {}

        embedding_npy = self.run_dir / "embedding.npy"
        np.save(embedding_npy, embedding)
        artifacts["embedding_npy"] = embedding_npy

        embedding_csv = self.run_dir / "embedding.csv"
        projection_df.to_csv(embedding_csv, index=False)
        artifacts["embedding_csv"] = embedding_csv

        preview_path = self._save_preview_plot(embedding, projection_df)
        artifacts["preview_plot"] = preview_path

        config_path, legacy_config_path = self._write_projection_config(
            processed, model_summary
        )
        artifacts["projection_config"] = config_path
        artifacts["tsne_config"] = legacy_config_path

        metadata = {
            "method": self.config.method.lower(),
            "n_components": self.config.n_components,
            "n_samples": int(embedding.shape[0]),
            "n_features": len(processed["feature_columns"]),
        }
        metadata.update(model_summary)

        return PreprocessorResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            artifacts=artifacts,
            metadata=metadata,
        )

    # Internal helpers -----------------------------------------------------

    def _create_reducer(self, method: str):
        try:
            reducer = global_reducer_registry.create(
                method,
                n_components=self.config.n_components,
                random_state=self.config.random_state,
                **self.config.method_params,
            )
        except KeyError as exc:  # pragma: no cover - exercised in integration tests
            available = (
                ", ".join(sorted(global_reducer_registry.available_methods().keys()))
                or "<none>"
            )
            raise ValueError(
                f"Unsupported reduction method '{method}'. Available methods: {available}."
            ) from exc
        return reducer

    def _build_projection_dataframe(
        self,
        original_df: pd.DataFrame,
        metadata_cols: List[str],
        embedding: np.ndarray,
    ) -> pd.DataFrame:
        dim_columns = [f"dim{idx + 1}" for idx in range(embedding.shape[1])]
        projection_df = pd.DataFrame(embedding, columns=dim_columns)
        for column in metadata_cols:
            if column in original_df.columns:
                projection_df[column] = original_df[column].values
        ordered_columns = metadata_cols + [
            col for col in projection_df.columns if col not in metadata_cols
        ]
        return projection_df[ordered_columns]

    def _save_preview_plot(
        self, embedding: np.ndarray, projection_df: pd.DataFrame
    ) -> Path:
        preview_path = self.run_dir / "embedding_preview.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            if embedding.shape[1] >= 2:
                ax.scatter(embedding[:, 0], embedding[:, 1], s=16, alpha=0.7)
                ax.set_xlabel("dim1")
                ax.set_ylabel("dim2")
            else:
                ax.scatter(
                    np.arange(embedding.shape[0]), embedding[:, 0], s=16, alpha=0.7
                )
                ax.set_xlabel("sample_index")
                ax.set_ylabel("dim1")
            ax.set_title(
                f"{self.config.method.upper()} projection ({len(projection_df)} samples)"
            )
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(preview_path), dpi=200)
        finally:
            plt.close(fig)
        return preview_path

    def _write_projection_config(
        self,
        processed: Dict[str, Any],
        model_summary: Dict[str, Any],
    ) -> Tuple[Path, Path]:
        payload = {
            "run_id": self.run_id,
            "config": self.config.to_serialisable_dict(),
            "feature_columns": processed["feature_columns"],
            "model_summary": _serialise_value(model_summary),
        }
        config_path = self.run_dir / "projection_config.json"
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        legacy_path = self.run_dir / "tsne_config.json"
        with legacy_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
        return config_path, legacy_path


__all__ = ["DimensionalityReductionPreprocessor"]
