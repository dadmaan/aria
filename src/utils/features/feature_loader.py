"""Utilities for loading feature matrices used by the GHSOM pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd

FeatureType = Literal["raw", "tsne"]


def _resolve_metadata_columns(
	df: pd.DataFrame, metadata_columns: Optional[Iterable[str]]
) -> Optional[pd.DataFrame]:
	if not metadata_columns:
		return None
	available_columns = [col for col in metadata_columns if col in df.columns]
	if not available_columns:
		return None
	return df.loc[:, available_columns].reset_index(drop=True)


def _load_json(path: Path) -> Dict[str, Any]:
	with path.open(encoding="utf-8") as fp:
		return json.load(fp)


def _coerce_artifact_path(raw_path: Optional[str], base_dir: Path) -> Optional[Path]:
	if not raw_path:
		return None
	candidate = Path(raw_path)
	if not candidate.is_absolute():
		candidate = base_dir / candidate
	return candidate


def _resolve_feature_source(
	path: Path,
	feature_type: str,
) -> Tuple[Path, Dict[str, Any]]:
	artifact_metadata: Dict[str, Any] = {}
	base_dir = path if path.is_dir() else path.parent

	def _load_run_metadata(run_metadata_path: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
		run_payload = _load_json(run_metadata_path)
		artifact_metadata.setdefault("run_metadata", run_payload)
		artifact_metadata.setdefault("metadata", run_payload.get("metadata", {}))
		artifact_paths = {
			name: _coerce_artifact_path(
				str(location) if location is not None else None,
				base_dir,
			)
			for name, location in run_payload.get("artifacts", {}).items()
		}
		artifact_metadata.setdefault("artifacts", artifact_paths)
		return (
			artifact_paths.get("embedding_csv")
			or artifact_paths.get("embedding_npy")
			or None,
			artifact_metadata,
		)

	def _maybe_attach_projection_config(config_path: Path) -> None:
		if config_path.exists():
			artifact_metadata.setdefault("projection_config", _load_json(config_path))

	candidate_path: Optional[Path] = None

	if path.is_dir():
		run_meta_path = path / "run_metadata.json"
		if run_meta_path.exists():
			candidate_path, _ = _load_run_metadata(run_meta_path)
		_maybe_attach_projection_config(path / "projection_config.json")
		_maybe_attach_projection_config(path / "tsne_config.json")
	elif path.suffix.lower() == ".json":
		json_payload = _load_json(path)
		if "artifacts" in json_payload:
			artifact_metadata.setdefault("run_metadata", json_payload)
			artifact_metadata.setdefault("metadata", json_payload.get("metadata", {}))
			artifact_paths = {
				name: _coerce_artifact_path(
					str(location) if location is not None else None,
					base_dir,
				)
				for name, location in json_payload.get("artifacts", {}).items()
			}
			artifact_metadata.setdefault("artifacts", artifact_paths)
			candidate_path = artifact_paths.get("embedding_csv") or artifact_paths.get("embedding_npy")
		else:
			artifact_metadata.setdefault("projection_config", json_payload)
		candidate_path = candidate_path or (base_dir / "embedding.csv")
		if not candidate_path.exists():
			candidate_path = base_dir / "embedding.npy"
		_maybe_attach_projection_config(base_dir / "projection_config.json")
		_maybe_attach_projection_config(base_dir / "tsne_config.json")
	else:
		candidate_path = path

	if candidate_path is None or not candidate_path.exists():
		if feature_type == "raw":
			fallback = base_dir / "features_numeric.csv"
			if fallback.exists():
				candidate_path = fallback
		else:
			csv_path = base_dir / "embedding.csv"
			npy_path = base_dir / "embedding.npy"
			if csv_path.exists():
				candidate_path = csv_path
			elif npy_path.exists():
				candidate_path = npy_path

	if candidate_path is None or not candidate_path.exists():
		raise FileNotFoundError(
			f"Could not locate feature artifact for feature_type='{feature_type}' under {path}"
		)

	return candidate_path, artifact_metadata


def load_feature_dataset(
	path: Path,
	feature_type: FeatureType = "tsne",
	*,
	metadata_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
	"""Load feature matrices and optional metadata for GHSOM training.

	Parameters
	----------
	path:
		Path to the artifact containing features. This may point directly to a
		CSV/NumPy file or to a run directory containing projection metadata.
	feature_type:
		Selects whether raw numeric features or reduced (t-SNE/PCA) features are
		returned.
	metadata_columns:
		Optional iterable of column names to retain alongside the returned
		features when present in the source artifact.

	Returns
	-------
	Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]
		A tuple containing the feature matrix (always provided) and the optional
		metadata DataFrame if requested columns are available, along with the
		resolved artifact metadata dictionary (run metadata, projection config,
		etc.)
	"""

	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"Feature artifact not found at {path}")

	ftype = cast(str, feature_type).lower()
	if ftype not in ("raw", "tsne"):
		raise ValueError(f"Unsupported feature type '{feature_type}'. Use 'raw' or 'tsne'.")

	resolved_path, artifact_metadata = _resolve_feature_source(path, ftype)

	if ftype == "raw":
		df = pd.read_csv(resolved_path)
		numeric_df = df.select_dtypes(include=[np.number]).reset_index(drop=True)
		if numeric_df.empty:
			raise ValueError("Raw feature artifact does not contain numeric columns.")
		metadata_df = _resolve_metadata_columns(df, metadata_columns)
		return numeric_df, metadata_df, artifact_metadata

	# Reduced features branch
	if resolved_path.suffix.lower() == ".npy":
		embedding = np.load(resolved_path)
		columns = [f"dim{idx+1}" for idx in range(embedding.shape[1])]
		features_df = pd.DataFrame(embedding, columns=columns)
		return features_df, None, artifact_metadata

	df = pd.read_csv(resolved_path)
	dim_columns = [col for col in df.columns if col.lower().startswith("dim")]
	if not dim_columns:
		dim_columns = df.select_dtypes(include=[np.number]).columns.tolist()
	if not dim_columns:
		raise ValueError("t-SNE artifact does not contain numeric projection columns.")

	features_df = df.loc[:, dim_columns].reset_index(drop=True)
	metadata_df = _resolve_metadata_columns(df, metadata_columns)
	return features_df, metadata_df, artifact_metadata


__all__ = ["FeatureType", "load_feature_dataset"]
