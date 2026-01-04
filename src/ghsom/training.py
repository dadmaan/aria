from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

# Import from the installed ghsom-py package (not src.ghsom)
import sys
_ghsom_pkg = sys.modules.get('ghsom')
if _ghsom_pkg is None or not hasattr(_ghsom_pkg, 'GHSOM'):
    # Force import from site-packages
    import importlib.util
    spec = importlib.util.find_spec('ghsom')
    if spec and spec.origin:
        _ghsom_pkg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ghsom_pkg)

from ghsom.core.ghsom import GHSOM
from ghsom.evaluation import metrics as evl

from src.ghsom_manager import GHSOMManager
from src.utils.features.feature_loader import FeatureType, load_feature_dataset


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_paths_to_strings(item) for item in obj)
    else:
        return obj


@dataclass(frozen=True)
class GHSOMTrainingConfig:
    """Configuration parameters for training a GHSOM model."""

    t1: float = 0.5
    t2: float = 0.05
    learning_rate: float = 0.1
    decay: float = 0.99
    gaussian_sigma: float = 1.0
    epochs: int = 20
    grow_maxiter: int = 20
    seed: int = 0

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | Any | None = None,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "GHSOMTrainingConfig":
        """Create a config instance from a generic mapping or attribute object."""

        payload: Dict[str, Any] = {}
        field_names = {field.name for field in fields(cls)}
        if mapping is not None:
            for field_name in field_names:
                value = None
                if isinstance(mapping, Mapping) and field_name in mapping:
                    value = mapping[field_name]
                elif hasattr(mapping, field_name):
                    value = getattr(mapping, field_name)
                if value is not None:
                    payload[field_name] = value
        if overrides:
            for key, value in overrides.items():
                if value is not None and key in field_names:
                    payload[key] = value
        return cls(**payload)


@dataclass
class GHSOMTrainingResult:
    """Dataclass capturing the outcome of a single GHSOM training run."""

    model: Any
    zero_unit: Any
    metrics: Dict[str, float]
    config: GHSOMTrainingConfig


def load_training_dataset(
    feature_path: Path,
    feature_type: FeatureType,
    metadata_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Load feature matrix and optional metadata for GHSOM training."""

    features_df, metadata_df, artifact_metadata = load_feature_dataset(
        feature_path,
        feature_type,
        metadata_columns=metadata_columns,
    )
    return features_df, metadata_df, artifact_metadata


def _compute_metrics(zero_unit: Any, data: np.ndarray) -> Dict[str, float]:
    mean_activation, std_activation = evl.mean_data_centroid_activation(zero_unit, data)

    # Helper function to safely convert to float
    def to_float(value):
        if isinstance(value, np.ndarray):
            return float(np.mean(value)) if value.size > 1 else float(value.item())
        return float(value)

    metrics: Dict[str, float] = {
        "mean_activation": to_float(mean_activation),
        "std_activation": to_float(std_activation),
        "dispersion_rate": to_float(evl.dispersion_rate(zero_unit, data)),
        "num_neurons": to_float(evl.get_total_number_of_neurons(zero_unit)),
        "num_clusters": to_float(evl.get_number_of_clusters(zero_unit.child_map)),
        "num_maps": to_float(evl.get_number_of_maps(zero_unit)),
        "ghsom_depth": to_float(evl.get_ghsom_depth(zero_unit.child_map, 1)),
        "max_neurons_child_map": to_float(
            evl.get_max_neurons_in_child_map(zero_unit.child_map)
        ),
        "avg_child_map_weights": to_float(evl.get_mean_child_map_size(zero_unit.child_map)),
    }
    return metrics


def train_ghsom_model(
    data: np.ndarray,
    config: GHSOMTrainingConfig,
    *,
    n_workers: int = -1,
) -> GHSOMTrainingResult:
    """Train a GHSOM model given a feature matrix and configuration."""

    # Validate input data for NaN values
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        total_count = data.size
        nan_percentage = 100 * nan_count / total_count
        raise ValueError(
            f"Input data contains {nan_count} NaN values ({nan_percentage:.2f}%). "
            f"Please handle missing values before training using imputation or removal."
        )

    ghsom = GHSOM(
        input_dataset=data,
        t1=config.t1,
        t2=config.t2,
        learning_rate=config.learning_rate,
        decay=config.decay,
        gaussian_sigma=config.gaussian_sigma,
    )

    zero_unit = ghsom.train(
        epochs_number=config.epochs,
        dataset_percentage=1.0,
        seed=config.seed,
        grow_maxiter=config.grow_maxiter,
        n_workers=n_workers,
    )

    metrics = _compute_metrics(zero_unit, data)
    return GHSOMTrainingResult(
        model=ghsom,
        zero_unit=zero_unit,
        metrics=metrics,
        config=replace(config),
    )


def _prepare_assignment_table(
    manager: GHSOMManager,
    metadata_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    assignments = manager.assign_ghsom_clusters()
    assignment_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(assignments)),
            "GHSOM_cluster": assignments["GHSOM_cluster"].to_numpy(),
        }
    )
    if metadata_df is not None:
        metadata_reset = metadata_df.reset_index(drop=True)
        assignment_df = pd.concat([assignment_df, metadata_reset], axis=1)
    return assignment_df


def export_artifacts(
    result: GHSOMTrainingResult,
    features_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    *,
    feature_type: FeatureType,
    artifact_metadata: Dict[str, Any],
    output_dir: Path,
    run_id: str,
    overwrite: bool = False,
    manager_cls: Type[GHSOMManager] = GHSOMManager,
) -> Dict[str, Path]:
    """Persist trained GHSOM artifacts and derived tables to disk."""

    run_dir = output_dir / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory {run_dir} already exists; pass overwrite=True to replace it."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "ghsom_model.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(result.zero_unit, model_file)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(result.metrics, metrics_file, indent=2, sort_keys=True)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(asdict(result.config), config_file, indent=2, sort_keys=True)

    if artifact_metadata:
        metadata_path = run_dir / "feature_artifact_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            # Convert Path objects to strings before JSON serialization
            serializable_metadata = _convert_paths_to_strings(artifact_metadata)
            json.dump(serializable_metadata, metadata_file, indent=2, sort_keys=True)
    else:
        metadata_path = None

    manager = manager_cls(
        model_path,
        features_df,
        metadata=metadata_df,
        feature_type=feature_type,
        artifact_metadata=artifact_metadata,
    )

    neuron_table_path = run_dir / "neuron_table.pkl"
    with neuron_table_path.open("wb") as neuron_file:
        pickle.dump(manager.neuron_table, neuron_file)

    assignment_path = run_dir / "sample_to_cluster.csv"
    assignment_df = _prepare_assignment_table(manager, metadata_df)
    assignment_df.to_csv(assignment_path, index=False)

    unique_ids_path = run_dir / "unique_cluster_ids.json"
    unique_ids = {"cluster_ids": manager.get_unique_cluster_ids_list()}
    with unique_ids_path.open("w", encoding="utf-8") as ids_file:
        json.dump(unique_ids, ids_file, indent=2, sort_keys=True)

    manifest = {
        "ghsom_model": str(model_path),
        "metrics": str(metrics_path),
        "config": str(config_path),
        "neuron_table": str(neuron_table_path),
        "sample_to_cluster": str(assignment_path),
        "unique_cluster_ids": str(unique_ids_path),
    }
    if metadata_path is not None:
        manifest["feature_artifact_metadata"] = str(metadata_path)

    manifest_path = run_dir / "artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)

    path_lookup: Dict[str, Path] = {
        "run_dir": run_dir,
        "ghsom_model": model_path,
        "metrics": metrics_path,
        "config": config_path,
        "neuron_table": neuron_table_path,
        "sample_to_cluster": assignment_path,
        "unique_cluster_ids": unique_ids_path,
        "manifest": manifest_path,
    }
    if metadata_path is not None:
        path_lookup["feature_artifact_metadata"] = metadata_path
    return path_lookup


def train_and_export(
    feature_path: Path,
    feature_type: FeatureType = "tsne",
    *,
    metadata_columns: Optional[Iterable[str]] = None,
    config: Optional[GHSOMTrainingConfig] = None,
    output_dir: Path,
    run_id: str,
    n_workers: int = -1,
    overwrite: bool = False,
    manager_cls: Type[GHSOMManager] = GHSOMManager,
) -> Dict[str, Any]:
    """High-level helper to train a GHSOM model and export its artifacts."""

    resolved_config = config or GHSOMTrainingConfig()
    features_df, metadata_df, artifact_metadata = load_training_dataset(
        feature_path,
        feature_type,
        metadata_columns=metadata_columns,
    )
    data = features_df.to_numpy(dtype=np.float64)

    result = train_ghsom_model(data, resolved_config, n_workers=n_workers)
    artifact_paths = export_artifacts(
        result,
        features_df,
        metadata_df,
        feature_type=feature_type,
        artifact_metadata=artifact_metadata,
        output_dir=output_dir,
        run_id=run_id,
        overwrite=overwrite,
        manager_cls=manager_cls,
    )

    return {
        "config": asdict(resolved_config),
        "metrics": result.metrics,
        "artifacts": artifact_paths,
    }


__all__ = [
    "GHSOMTrainingConfig",
    "GHSOMTrainingResult",
    "load_training_dataset",
    "train_ghsom_model",
    "export_artifacts",
    "train_and_export",
]
