"""Inter-stage validation utilities for the preprocessing pipeline.

This module provides validation functions that check intermediate outputs between
pipeline stages to catch issues early and provide actionable error messages.

Functions:
    validate_feature_extraction_output: Validate feature extraction stage output.
    validate_dimensionality_reduction_output: Validate dimensionality reduction output.
    validate_ghsom_training_output: Validate GHSOM training output.

Classes:
    ValidationReport: Structured validation result container.
    ValidationWarning: Non-fatal validation issue.
    ValidationError: Fatal validation issue.

Example:
    >>> report = validate_feature_extraction_output(
    ...     features_csv=Path("features.csv"),
    ...     min_samples=100,
    ...     max_nan_ratio=0.1
    ... )
    >>> if not report.is_valid:
    ...     raise ValueError(f"Validation failed: {report.errors}")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging.logging_manager import get_logger

logger = get_logger("preprocessing.validation")


# ========================================================================
# VALIDATION REPORT CLASSES
# ========================================================================


@dataclass
class ValidationWarning:
    """Non-fatal validation issue.

    Attributes:
        message: Description of the warning.
        context: Additional context (e.g., affected columns, counts).
    """

    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationError:
    """Fatal validation issue that should stop pipeline execution.

    Attributes:
        message: Description of the error.
        context: Additional context (e.g., affected columns, counts).
    """

    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Structured validation result container.

    Attributes:
        stage: Pipeline stage name (e.g., "feature_extraction").
        is_valid: Whether validation passed (no errors).
        warnings: List of non-fatal warnings.
        errors: List of fatal errors.
        metrics: Validation metrics (e.g., sample_count, nan_ratio).
    """

    stage: str
    is_valid: bool
    warnings: List[ValidationWarning] = field(default_factory=list)
    errors: List[ValidationError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "stage": self.stage,
            "is_valid": self.is_valid,
            "warnings": [
                {"message": w.message, "context": w.context} for w in self.warnings
            ],
            "errors": [
                {"message": e.message, "context": e.context} for e in self.errors
            ],
            "metrics": self.metrics,
        }

    def save(self, path: Path) -> None:
        """Save validation report to JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved validation report to {path}")


# ========================================================================
# FEATURE EXTRACTION VALIDATION
# ========================================================================


def validate_feature_extraction_output(
    features_csv: Path,
    min_samples: int = 100,
    max_nan_ratio: float = 0.1,
    min_feature_variance: float = 1e-6,
) -> ValidationReport:
    """Validate feature extraction stage output.

    Checks:
        - Features CSV exists and is readable
        - Sufficient number of samples
        - NaN ratio per column below threshold
        - Feature variance above threshold (non-constant features)
        - At least some numeric columns present

    Args:
        features_csv: Path to features_numeric.csv or features_with_metadata.csv.
        min_samples: Minimum number of samples required.
        max_nan_ratio: Maximum allowed NaN ratio per column (0.0-1.0).
        min_feature_variance: Minimum variance required to keep a feature.

    Returns:
        ValidationReport with validation results and metrics.

    Example:
        >>> report = validate_feature_extraction_output(
        ...     features_csv=Path("artifacts/preprocessing/run1/features/features_numeric.csv"),
        ...     min_samples=100,
        ...     max_nan_ratio=0.1
        ... )
        >>> assert report.is_valid
    """
    logger.info(f"Validating feature extraction output: {features_csv}")

    report = ValidationReport(stage="feature_extraction", is_valid=True)

    # Check file existence
    if not features_csv.exists():
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Features CSV does not exist",
                context={"path": str(features_csv)},
            )
        )
        return report

    # Load features
    try:
        df = pd.read_csv(features_csv)
    except Exception as exc:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Failed to read features CSV: {exc}",
                context={"path": str(features_csv)},
            )
        )
        return report

    n_samples = len(df)
    report.metrics["sample_count"] = n_samples
    report.metrics["column_count"] = len(df.columns)

    # Check sample count
    if n_samples < min_samples:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Insufficient samples: {n_samples} < {min_samples}",
                context={"sample_count": n_samples, "min_samples": min_samples},
            )
        )

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report.metrics["numeric_column_count"] = len(numeric_cols)

    if len(numeric_cols) == 0:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message="No numeric feature columns found",
                context={"columns": df.columns.tolist()},
            )
        )
        return report

    # Check NaN ratios
    nan_ratios = df[numeric_cols].isna().mean()
    high_nan_cols = [col for col, ratio in nan_ratios.items() if ratio > max_nan_ratio]

    if high_nan_cols:
        report.warnings.append(
            ValidationWarning(
                message=f"{len(high_nan_cols)} columns exceed max NaN ratio",
                context={
                    "high_nan_columns": high_nan_cols,
                    "max_nan_ratio": max_nan_ratio,
                    "nan_ratios": {
                        col: float(nan_ratios[col]) for col in high_nan_cols
                    },
                },
            )
        )

    report.metrics["mean_nan_ratio"] = float(nan_ratios.mean())
    report.metrics["max_nan_ratio"] = float(nan_ratios.max())

    # Check feature variance (on non-NaN values)
    variances = df[numeric_cols].var(skipna=True)
    low_var_cols = [
        col
        for col, var in variances.items()
        if not np.isnan(var) and var < min_feature_variance
    ]

    if low_var_cols:
        report.warnings.append(
            ValidationWarning(
                message=f"{len(low_var_cols)} columns have near-zero variance",
                context={
                    "low_variance_columns": low_var_cols,
                    "min_feature_variance": min_feature_variance,
                },
            )
        )

    report.metrics["low_variance_column_count"] = len(low_var_cols)

    logger.info(
        f"Feature extraction validation complete: "
        f"valid={report.is_valid}, samples={n_samples}, "
        f"numeric_cols={len(numeric_cols)}, warnings={len(report.warnings)}"
    )

    return report


# ========================================================================
# DIMENSIONALITY REDUCTION VALIDATION
# ========================================================================


def validate_dimensionality_reduction_output(
    embedding_csv: Path,
    embedding_npy: Optional[Path] = None,
    expected_n_components: int = 2,
    min_samples: int = 100,
) -> ValidationReport:
    """Validate dimensionality reduction stage output.

    Checks:
        - Embedding CSV exists and is readable
        - Expected number of dimensions (e.g., 2 for t-SNE)
        - Sufficient number of samples
        - No NaN/Inf values in embedding
        - Embedding has non-zero variance (not collapsed)
        - NPY array matches CSV if both provided

    Args:
        embedding_csv: Path to embedding.csv.
        embedding_npy: Optional path to embedding.npy for cross-validation.
        expected_n_components: Expected number of dimensions.
        min_samples: Minimum number of samples required.

    Returns:
        ValidationReport with validation results and metrics.

    Example:
        >>> report = validate_dimensionality_reduction_output(
        ...     embedding_csv=Path("artifacts/preprocessing/run1/reduced/embedding.csv"),
        ...     expected_n_components=2
        ... )
        >>> assert report.is_valid
    """
    logger.info(f"Validating dimensionality reduction output: {embedding_csv}")

    report = ValidationReport(stage="dimensionality_reduction", is_valid=True)

    # Check file existence
    if not embedding_csv.exists():
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message="Embedding CSV does not exist",
                context={"path": str(embedding_csv)},
            )
        )
        return report

    # Load embedding CSV
    try:
        df = pd.read_csv(embedding_csv)
    except Exception as exc:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Failed to read embedding CSV: {exc}",
                context={"path": str(embedding_csv)},
            )
        )
        return report

    n_samples = len(df)
    report.metrics["sample_count"] = n_samples

    # Check sample count
    if n_samples < min_samples:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Insufficient samples: {n_samples} < {min_samples}",
                context={"sample_count": n_samples, "min_samples": min_samples},
            )
        )

    # Identify dimension columns (dim0, dim1, ...) or (dim1, dim2, ...)
    dim_cols = [col for col in df.columns if col.startswith("dim")]
    report.metrics["n_components"] = len(dim_cols)

    if len(dim_cols) != expected_n_components:
        report.warnings.append(
            ValidationWarning(
                message=f"Unexpected number of dimensions: {len(dim_cols)} != {expected_n_components}",
                context={
                    "found_dimensions": len(dim_cols),
                    "expected_dimensions": expected_n_components,
                    "dimension_columns": dim_cols,
                },
            )
        )

    if len(dim_cols) == 0:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message="No dimension columns found in embedding CSV",
                context={"columns": df.columns.tolist()},
            )
        )
        return report

    # Check for NaN/Inf values
    embedding_values = df[dim_cols].values
    nan_count = np.isnan(embedding_values).sum()
    inf_count = np.isinf(embedding_values).sum()

    if nan_count > 0:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Embedding contains {nan_count} NaN values",
                context={"nan_count": int(nan_count)},
            )
        )

    if inf_count > 0:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Embedding contains {inf_count} Inf values",
                context={"inf_count": int(inf_count)},
            )
        )

    # Check variance (should not collapse to a point)
    if nan_count == 0 and inf_count == 0:
        variances = df[dim_cols].var()
        report.metrics["dimension_variances"] = {
            col: float(var) for col, var in variances.items()
        }

        zero_var_dims = [col for col, var in variances.items() if var < 1e-12]
        if zero_var_dims:
            report.is_valid = False
            report.errors.append(
                ValidationError(
                    message=f"Embedding has collapsed dimensions (zero variance): {zero_var_dims}",
                    context={"zero_variance_dimensions": zero_var_dims},
                )
            )

    # Cross-validate with NPY if provided
    if embedding_npy is not None and embedding_npy.exists():
        try:
            npy_array = np.load(embedding_npy)
            if npy_array.shape != embedding_values.shape:
                report.warnings.append(
                    ValidationWarning(
                        message="CSV and NPY embeddings have different shapes",
                        context={
                            "csv_shape": embedding_values.shape,
                            "npy_shape": npy_array.shape,
                        },
                    )
                )
            elif not np.allclose(npy_array, embedding_values, equal_nan=True):
                report.warnings.append(
                    ValidationWarning(
                        message="CSV and NPY embeddings have different values",
                        context={
                            "max_diff": float(
                                np.max(np.abs(npy_array - embedding_values))
                            )
                        },
                    )
                )
        except Exception as exc:
            report.warnings.append(
                ValidationWarning(
                    message=f"Failed to validate NPY array: {exc}",
                    context={"npy_path": str(embedding_npy)},
                )
            )

    logger.info(
        f"Dimensionality reduction validation complete: "
        f"valid={report.is_valid}, samples={n_samples}, "
        f"dims={len(dim_cols)}, warnings={len(report.warnings)}"
    )

    return report


# ========================================================================
# GHSOM TRAINING VALIDATION
# ========================================================================


def validate_ghsom_training_output(
    ghsom_model_path: Path,
    neuron_table_path: Path,
    sample_to_cluster_path: Path,
    min_clusters: int = 2,
) -> ValidationReport:
    """Validate GHSOM training stage output.

    Checks:
        - GHSOM model pickle exists
        - Neuron table pickle exists
        - Sample-to-cluster mapping CSV exists and is readable
        - Sufficient number of clusters (action space size)
        - All cluster IDs are non-negative integers
        - No samples assigned to invalid clusters

    Args:
        ghsom_model_path: Path to ghsom_model.pkl.
        neuron_table_path: Path to neuron_table.pkl.
        sample_to_cluster_path: Path to sample_to_cluster.csv.
        min_clusters: Minimum number of clusters required.

    Returns:
        ValidationReport with validation results and metrics.

    Example:
        >>> report = validate_ghsom_training_output(
        ...     ghsom_model_path=Path("artifacts/preprocessing/run1/ghsom/ghsom_model.pkl"),
        ...     neuron_table_path=Path("artifacts/preprocessing/run1/ghsom/neuron_table.pkl"),
        ...     sample_to_cluster_path=Path("artifacts/preprocessing/run1/ghsom/sample_to_cluster.csv")
        ... )
        >>> assert report.is_valid
    """
    logger.info(f"Validating GHSOM training output: {ghsom_model_path.parent}")

    report = ValidationReport(stage="ghsom_training", is_valid=True)

    # Check file existence
    missing_files = []
    if not ghsom_model_path.exists():
        missing_files.append(str(ghsom_model_path))
    if not neuron_table_path.exists():
        missing_files.append(str(neuron_table_path))
    if not sample_to_cluster_path.exists():
        missing_files.append(str(sample_to_cluster_path))

    if missing_files:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Missing GHSOM output files: {missing_files}",
                context={"missing_files": missing_files},
            )
        )
        return report

    # Load sample-to-cluster mapping
    try:
        cluster_df = pd.read_csv(sample_to_cluster_path)
    except Exception as exc:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Failed to read sample_to_cluster.csv: {exc}",
                context={"path": str(sample_to_cluster_path)},
            )
        )
        return report

    n_samples = len(cluster_df)
    report.metrics["sample_count"] = n_samples

    # Check for cluster column
    cluster_col = "GHSOM_cluster"
    if cluster_col not in cluster_df.columns:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Missing '{cluster_col}' column in sample_to_cluster.csv",
                context={"columns": cluster_df.columns.tolist()},
            )
        )
        return report

    # Analyze cluster assignments
    cluster_ids = cluster_df[cluster_col].dropna()
    unique_clusters = cluster_ids.unique()
    n_clusters = len(unique_clusters)

    report.metrics["num_clusters"] = n_clusters
    report.metrics["cluster_ids"] = sorted(
        [int(c) for c in unique_clusters if not np.isnan(c)]
    )

    # Check minimum cluster count
    if n_clusters < min_clusters:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Insufficient clusters: {n_clusters} < {min_clusters}",
                context={"num_clusters": n_clusters, "min_clusters": min_clusters},
            )
        )

    # Check for invalid cluster IDs (negative or NaN)
    invalid_ids = cluster_ids[cluster_ids < 0]
    if len(invalid_ids) > 0:
        report.is_valid = False
        report.errors.append(
            ValidationError(
                message=f"Found {len(invalid_ids)} negative cluster IDs",
                context={"invalid_count": len(invalid_ids)},
            )
        )

    nan_clusters = cluster_df[cluster_col].isna().sum()
    if nan_clusters > 0:
        report.warnings.append(
            ValidationWarning(
                message=f"{nan_clusters} samples have NaN cluster assignments",
                context={"nan_count": int(nan_clusters)},
            )
        )

    # Cluster size distribution
    cluster_sizes = cluster_df[cluster_col].value_counts()
    report.metrics["min_cluster_size"] = int(cluster_sizes.min())
    report.metrics["max_cluster_size"] = int(cluster_sizes.max())
    report.metrics["mean_cluster_size"] = float(cluster_sizes.mean())

    # Warn about very small clusters
    tiny_clusters = cluster_sizes[cluster_sizes < 5]
    if len(tiny_clusters) > 0:
        report.warnings.append(
            ValidationWarning(
                message=f"{len(tiny_clusters)} clusters have < 5 samples",
                context={
                    "tiny_cluster_count": len(tiny_clusters),
                    "tiny_cluster_ids": [int(c) for c in tiny_clusters.index],
                },
            )
        )

    logger.info(
        f"GHSOM training validation complete: "
        f"valid={report.is_valid}, samples={n_samples}, "
        f"clusters={n_clusters}, warnings={len(report.warnings)}"
    )

    return report
