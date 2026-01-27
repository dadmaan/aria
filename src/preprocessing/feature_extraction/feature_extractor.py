"""Feature extraction pipeline for MIDI datasets.

This module centralises the feature computation logic so it can be reused
by scripts and tests. It produces deterministic, reproducible feature
artifacts that combine PrettyMIDI, MusPy, and lightweight music-theory
metrics.

Features:
    - Feature selection: Extract all features ("full") or a specific subset
    - Parallel processing: Use multiple workers for faster extraction
    - Robust error handling: Continue processing on individual file failures
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pretty_midi import PrettyMIDI

from src.utils.logging.logging_manager import get_logger
from . import features_music_theory as fmt
from . import features_muspy as fmus
from . import features_pretty_midi as fpm

_DEFAULT_OUTPUT_ROOT = Path("artifacts/features/raw")


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy/pandas objects to built-in Python types.

    The resulting object is JSON serialisable which keeps downstream
    persistence simple.
    """

    if isinstance(value, (np.generic,)):  # type: ignore[arg-type]
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if pd.api.types.is_scalar(value):
        return value
    return value


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for a feature extraction job.

    Attributes:
        dataset_root: Root directory containing MIDI files.
        metadata_csv: Optional CSV with file metadata.
        output_root: Root directory for output artifacts.
        metadata_index_column: Column name for track IDs in metadata CSV.
        metadata_path_column: Column name for file paths in metadata CSV.
        metadata_split_column: Column name for train/valid/test split.
        include_splits: Splits to include (None = all).
        extensions: MIDI file extensions to process.
        run_id: Unique run identifier (auto-generated if None).
        seed: Random seed for reproducibility.
        overwrite: Allow overwriting existing output directory.
        max_files: Maximum number of files to process (None = all).
        num_workers: Number of worker processes for parallel extraction.
            1 = sequential, -1 = all available CPUs, N = use N workers.
        save_per_file: Save individual JSON per MIDI file.
        features: Feature selection - "full" for all features, or list of
            specific feature names (e.g., ["pm_note_count", "muspy_pitch_entropy"]).
    """

    dataset_root: Optional[Path]
    metadata_csv: Optional[Path] = None
    output_root: Path = field(default_factory=lambda: _DEFAULT_OUTPUT_ROOT)
    metadata_index_column: str = "id"
    metadata_path_column: str = "file_path"
    metadata_split_column: Optional[str] = "split"
    include_splits: Optional[Sequence[str]] = None
    extensions: Sequence[str] = (".mid", ".midi")
    run_id: Optional[str] = None
    seed: int = 7
    overwrite: bool = False
    max_files: Optional[int] = None
    num_workers: int = 4
    save_per_file: bool = True
    features: Union[str, Sequence[str]] = "full"

    def to_serialisable_dict(self) -> Dict[str, Any]:
        data = {
            "dataset_root": str(self.dataset_root) if self.dataset_root else None,
            "metadata_csv": str(self.metadata_csv) if self.metadata_csv else None,
            "output_root": str(self.output_root),
            "metadata_index_column": self.metadata_index_column,
            "metadata_path_column": self.metadata_path_column,
            "metadata_split_column": self.metadata_split_column,
            "include_splits": (
                list(self.include_splits) if self.include_splits else None
            ),  # Convert to list for JSON
            "extensions": list(self.extensions),
            "run_id": self.run_id,
            "seed": self.seed,
            "overwrite": self.overwrite,
            "max_files": self.max_files,
            "num_workers": self.num_workers,
            "save_per_file": self.save_per_file,
            "features": (
                list(self.features)
                if isinstance(self.features, (list, tuple))
                else self.features
            ),
        }
        return data


@dataclass
class FeatureExtractionReport:
    """Summary of a completed feature extraction job."""

    run_id: str
    run_dir: Path
    feature_matrix_path: Path
    index_path: Path
    metadata_join_path: Optional[Path]
    config_path: Path
    log_path: Path
    processed_files: int
    success_count: int
    failure_count: int
    error_records: List[Dict[str, Any]]


@dataclass
class _ProcessingItem:
    midi_path: Path
    track_id: str
    metadata: Optional[Dict[str, Any]]
    metadata_index: int


@dataclass
class _FeatureExtractionResult:
    item: _ProcessingItem
    features: Optional[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


def _normalise_extensions(extensions: Sequence[str]) -> List[str]:
    normalised = []
    for ext in extensions:
        if not ext:
            continue
        if not ext.startswith("."):
            normalised.append(f".{ext}")
        else:
            normalised.append(ext)
    return normalised


def _generate_run_id(config: FeatureExtractionConfig) -> str:
    if config.run_id:
        return config.run_id
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"features_{timestamp}"


def _initialise_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _select_metadata_rows(
    metadata_df: pd.DataFrame,
    split_column: Optional[str],
    include_splits: Optional[Sequence[str]],
) -> pd.DataFrame:
    if split_column and include_splits:
        mask = metadata_df[split_column].astype(str).isin(include_splits)
        return metadata_df.loc[mask].reset_index(drop=True)
    return metadata_df.reset_index(drop=True)


def _resolve_midi_path(path_value: Any, dataset_root: Optional[Path]) -> Path:
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate  # Use absolute path as-is
    if not dataset_root:
        raise ValueError(
            "Relative MIDI paths require a dataset_root to be specified in the configuration."
        )
    return dataset_root.joinpath(candidate)  # Resolve relative to dataset_root


def _ensure_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)  # Remove existing dir
        else:
            raise FileExistsError(
                f"Output directory {path} already exists. Use overwrite=True to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)  # Create directory


def _collect_processing_items(
    config: FeatureExtractionConfig, logger: logging.Logger
) -> List[_ProcessingItem]:
    items: List[_ProcessingItem] = []

    if config.metadata_csv:
        metadata_df = pd.read_csv(config.metadata_csv, low_memory=False)
        if config.metadata_index_column not in metadata_df.columns:
            raise KeyError(
                f"Metadata column '{config.metadata_index_column}' was not found in {config.metadata_csv}."
            )
        if config.metadata_path_column not in metadata_df.columns:
            raise KeyError(
                f"Metadata column '{config.metadata_path_column}' was not found in {config.metadata_csv}."
            )
        filtered_df = _select_metadata_rows(
            metadata_df,
            config.metadata_split_column,
            config.include_splits,
        )
        if filtered_df.empty:
            logger.warning("No metadata rows matched the provided filters.")

        filtered_reset = filtered_df.reset_index(drop=True)
        records = filtered_reset.to_dict(orient="records")
        for metadata_index, row in enumerate(records):
            path_value = row[config.metadata_path_column]
            midi_path = _resolve_midi_path(path_value, config.dataset_root)
            metadata_dict = {
                column: _to_builtin(row[column]) for column in filtered_reset.columns
            }
            track_id = str(row[config.metadata_index_column])
            items.append(
                _ProcessingItem(
                    midi_path=midi_path,
                    track_id=track_id,
                    metadata=metadata_dict,
                    metadata_index=metadata_index,
                )
            )
    else:
        dataset_root = config.dataset_root
        if not dataset_root:
            raise ValueError(
                "Either metadata_csv or dataset_root must be provided to discover MIDI files."
            )
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root {dataset_root} does not exist.")
        extensions = _normalise_extensions(config.extensions)
        midi_paths: List[Path] = []
        for extension in extensions:
            midi_paths.extend(sorted(dataset_root.rglob(f"*{extension}")))
        midi_paths = sorted(set(midi_paths))
        if config.max_files is not None:
            midi_paths = midi_paths[: config.max_files]
        for idx, midi_path in enumerate(midi_paths):
            items.append(
                _ProcessingItem(
                    midi_path=midi_path,
                    track_id=midi_path.stem,
                    metadata=None,
                    metadata_index=idx,
                )
            )
    if config.max_files is not None:
        items = items[: config.max_files]
    return items


def _extract_features_for_file(midi_path: Path) -> tuple[Dict[str, Any], List[str]]:
    # Suppress numpy warnings about degrees of freedom and division in statistical calculations
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="Degrees of freedom <= 0 for slice",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in divide",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in scalar divide",
        )
        return _extract_features_for_file_impl(midi_path)


def _extract_features_for_file_impl(
    midi_path: Path,
) -> tuple[Dict[str, Any], List[str]]:
    midi_data = PrettyMIDI(str(midi_path))
    notes = [note for instrument in midi_data.instruments for note in instrument.notes]
    if not notes:
        raise ValueError("No notes found in MIDI file.")

    note_velocities = [note.velocity for note in notes]

    (min_pitch_freq, min_pitch_value), (max_pitch_freq, max_pitch_value) = (
        fpm.get_pitch_range(notes)
    )
    pitch_contours = fpm.get_pitch_contours(notes)
    overall_pitch_contour = fpm.get_overall_pitch_contours(
        pitch_contours, threshold=0.7
    )

    interval_analysis = fpm.get_interval_analysis(midi_data)
    interval_range = interval_analysis.get("interval_range", (None, None))

    energy, groove = fpm.get_energy_and_groove(midi_data)
    texture_info = fpm.extract_texture_and_polyphony(midi_data, time_step=0.5)
    number_of_bars = fpm.get_number_of_bars(midi_data)
    tempo = fpm.get_tempo(midi_data)
    instrumentation = fpm.extract_instrumentation(midi_data)
    pitch_classes = sorted(fpm.get_pitch_classes(midi_data))
    time_signatures = fpm.extract_time_signatures(midi_data)
    primary_time_signature: Optional[str] = None
    if time_signatures:
        signature = time_signatures[0]
        primary_time_signature = f"{signature['numerator']}/{signature['denominator']}"

    average_velocity = float(np.mean(note_velocities)) if note_velocities else 0.0

    pretty_midi_features = {
        "pm_note_count": len(notes),
        "pm_length_seconds": float(midi_data.get_end_time()),
        "pm_pitch_range_min_freq": float(min_pitch_freq),
        "pm_pitch_range_min_note": int(min_pitch_value),
        "pm_pitch_range_max_freq": float(max_pitch_freq),
        "pm_pitch_range_max_note": int(max_pitch_value),
        "pm_average_pitch_hz": float(fpm.get_average_pitch(notes)),
        "pm_average_velocity": average_velocity,
        "pm_overall_pitch_contour": overall_pitch_contour[1],
        "pm_interval_range_min": (
            int(interval_range[0]) if interval_range[0] is not None else None
        ),
        "pm_interval_range_max": (
            int(interval_range[1]) if interval_range[1] is not None else None
        ),
        "pm_energy": float(energy),
        "pm_groove": float(groove),
        "pm_max_polyphony": int(texture_info.get("max_polyphony", 0)),
        "pm_average_polyphony": float(texture_info.get("average_polyphony", 0.0)),
        "pm_instrument_count": int(texture_info.get("instrument_count", 0)),
        "pm_note_density": float(texture_info.get("note_density", 0.0)),
        "pm_bar_count": float(number_of_bars),
        "pm_tempo_bpm": float(tempo),
        "pm_pitch_classes": pitch_classes,
        "pm_instrumentation": instrumentation,
        "pm_time_signature": primary_time_signature,
    }

    muspy_features = fmus.get_midi_metrics_from_muspy(str(midi_path))
    warnings: List[str] = []
    if "error" in muspy_features:
        warnings.append(f"muspy: {muspy_features['error']}")
        muspy_features = {}
    muspy_prefixed = {
        f"muspy_{key}": _to_builtin(value) for key, value in muspy_features.items()
    }

    theory_features_raw: Dict[str, Any] = {}
    note_pitches = [note.pitch for note in notes]
    if len(note_pitches) >= 12:
        try:
            theory_features_raw = fmt.get_music_theory_features(note_pitches)
        except Exception as exc:  # noqa: BLE001 - capture all theory computation issues
            warnings.append(f"music_theory: {exc}")
            theory_features_raw = {}
    else:
        warnings.append(
            "music_theory: skipped (requires at least 12 notes for stable window analysis)"
        )
    theory_features = {
        f"theory_{key}": _to_builtin(value)
        for key, value in theory_features_raw.items()
    }

    features = {
        **pretty_midi_features,
        **muspy_prefixed,
        **theory_features,
    }
    serialisable_features = {key: _to_builtin(value) for key, value in features.items()}
    return serialisable_features, warnings


def _create_logger(log_path: Path, log_level: int) -> logging.Logger:
    # Use centralized logging manager - it returns LoggingManager, but we need the underlying logger
    # for compatibility with existing code that adds handlers
    logger_manager = get_logger("feature_extraction")
    logger = logger_manager.logger  # Get the underlying logging.Logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def _numeric_feature_subset(features: Dict[str, Any]) -> Dict[str, Any]:
    numerical: Dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, (int, float)):
            numerical[key] = float(value)
        elif isinstance(value, (np.integer, np.floating)):
            numerical[key] = float(value)
    return numerical


def _filter_features(
    features: Dict[str, Any],
    requested_features: Union[str, Sequence[str]],
) -> Tuple[Dict[str, Any], List[str]]:
    """Filter extracted features to keep only requested ones.

    Args:
        features: Full dictionary of extracted features.
        requested_features: "full" to keep all features, or list of specific
            feature names to keep.

    Returns:
        Tuple of (filtered_features, missing_features_list).
        missing_features_list contains features that were requested but not
        present in this specific file's extraction (may be conditionally
        available for other files).
    """
    if requested_features == "full":
        return features, []

    requested_set = set(requested_features)
    available_set = set(features.keys())
    missing_features = list(requested_set - available_set)

    filtered = {key: features[key] for key in requested_features if key in features}
    return filtered, missing_features


def _get_effective_num_workers(num_workers: int) -> int:
    """Resolve num_workers to an actual worker count.

    Args:
        num_workers: Configured worker count. 1 = sequential, -1 = all CPUs.

    Returns:
        Effective number of worker processes.
    """
    if num_workers == -1:
        return max(1, multiprocessing.cpu_count())
    return max(1, num_workers)


def _extract_single_file_worker(
    midi_path_str: str,
) -> Tuple[str, Optional[Dict[str, Any]], List[str], Optional[str]]:
    """Worker function for parallel feature extraction.

    This function runs in a separate process and extracts features for a single
    MIDI file. It must be a module-level function to be picklable.

    Args:
        midi_path_str: String path to the MIDI file.

    Returns:
        Tuple of (midi_path_str, features, warnings, error).
    """
    midi_path = Path(midi_path_str)
    try:
        features, extraction_warnings = _extract_features_for_file(midi_path)
        return midi_path_str, features, extraction_warnings, None
    except Exception as exc:  # noqa: BLE001 - capture all exceptions in worker
        return midi_path_str, None, [], str(exc)


def run_feature_extraction(
    config: FeatureExtractionConfig,
    *,
    log_level: int = logging.INFO,
) -> FeatureExtractionReport:
    """Execute the feature extraction pipeline.

    This function extracts features from MIDI files using three backends:
    - PrettyMIDI (pm_* features)
    - MusPy (muspy_* features)
    - Music theory (theory_* features)

    Supports parallel processing with configurable number of workers and
    feature filtering to extract only specific features.

    Args:
        config: Feature extraction configuration.
        log_level: Logging level (default: INFO).

    Returns:
        FeatureExtractionReport with paths to output artifacts and statistics.
    """
    _initialise_seeds(config.seed)

    run_id = _generate_run_id(config)
    output_root = config.output_root
    run_dir = output_root / run_id
    _ensure_directory(run_dir, overwrite=config.overwrite)

    log_path = run_dir / "feature_extraction.log"
    logger = _create_logger(log_path, log_level)
    logger.info("Starting feature extraction run %s", run_id)

    # Log configuration
    effective_workers = _get_effective_num_workers(config.num_workers)
    logger.info(f"Using {effective_workers} worker(s) for feature extraction")
    if config.features != "full":
        logger.info(
            f"Feature filtering enabled: {len(config.features)} features requested"
        )

    items = _collect_processing_items(config, logger)
    if not items:
        logger.warning("No MIDI files discovered. Nothing to process.")

    # Build a mapping from midi_path_str to item for result matching
    path_to_item: Dict[str, _ProcessingItem] = {
        str(item.midi_path): item for item in items
    }

    results: List[_FeatureExtractionResult] = []
    # Track features that were missing in at least one file (conditionally available)
    missing_features_per_file: Set[str] = set()
    # Track features that were found in at least one file (to distinguish truly unknown)
    features_seen_in_any_file: Set[str] = set()

    # Choose sequential or parallel processing
    if effective_workers == 1:
        # Sequential processing
        logger.info(f"Processing {len(items)} files sequentially...")
        for idx, item in enumerate(items):
            if (idx + 1) % 100 == 0 or idx == 0:
                logger.info(f"Processing file {idx + 1}/{len(items)}")
            try:
                features, extraction_warnings = _extract_features_for_file(
                    item.midi_path
                )
                features_seen_in_any_file.update(features.keys())
                # Apply feature filtering
                filtered_features, missing = _filter_features(
                    features,
                    config.features,
                )
                missing_features_per_file.update(missing)
                results.append(
                    _FeatureExtractionResult(
                        item=item,
                        features=filtered_features,
                        warnings=extraction_warnings,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - log full exception details
                logger.exception("Failed to process %s", item.midi_path)
                results.append(
                    _FeatureExtractionResult(
                        item=item,
                        features=None,
                        warnings=[],
                        error=str(exc),
                    )
                )
    else:
        # Parallel processing with ProcessPoolExecutor
        logger.info(
            f"Processing {len(items)} files with {effective_workers} workers..."
        )
        midi_paths = [str(item.midi_path) for item in items]

        completed = 0
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_path = {
                executor.submit(_extract_single_file_worker, path): path
                for path in midi_paths
            }

            for future in as_completed(future_to_path):
                midi_path_str, features, extraction_warnings, error = future.result()
                item = path_to_item[midi_path_str]

                if features is not None:
                    features_seen_in_any_file.update(features.keys())
                    # Apply feature filtering
                    filtered_features, missing = _filter_features(
                        features,
                        config.features,
                    )
                    missing_features_per_file.update(missing)
                    results.append(
                        _FeatureExtractionResult(
                            item=item,
                            features=filtered_features,
                            warnings=extraction_warnings,
                        )
                    )
                else:
                    logger.error(f"Failed to process {midi_path_str}: {error}")
                    results.append(
                        _FeatureExtractionResult(
                            item=item,
                            features=None,
                            warnings=extraction_warnings,
                            error=error,
                        )
                    )

                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Completed {completed}/{len(items)} files")

        logger.info(f"Parallel processing complete: {completed} files processed")

    # Log warnings about features that were never found in any file (truly unknown)
    if config.features != "full":
        requested_set = set(config.features)
        truly_unknown = missing_features_per_file - features_seen_in_any_file
        if truly_unknown:
            logger.warning(
                f"Requested features never found in any file: {sorted(truly_unknown)}"
            )
        # Conditionally available features (found in some files but not others) are normal
        # and don't need a warning - they'll result in NaN values which is expected

    per_file_dir: Optional[Path] = None
    if config.save_per_file:
        per_file_dir = run_dir / "per_file"
        per_file_dir.mkdir(exist_ok=True)

    index_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, Any]] = []

    for result in results:
        item = result.item
        record = {
            "track_id": item.track_id,
            "midi_path": str(item.midi_path),
            "metadata_index": item.metadata_index,
            "status": "ok" if result.features else "error",
            "error": result.error,
        }
        if item.metadata is not None:
            record["metadata"] = item.metadata
        if result.warnings:
            record["warnings"] = result.warnings
        index_records.append(record)

        if result.error:
            error_records.append(record)
            continue

        if per_file_dir is not None:
            per_file_payload = {
                "track_id": item.track_id,
                "midi_path": str(item.midi_path),
                "metadata": item.metadata,
                "features": result.features,
                "warnings": result.warnings,
            }
            output_name = f"{item.track_id}.json"
            output_path = per_file_dir / output_name
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(per_file_payload, handle, indent=2)

    index_path = run_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_records, handle, indent=2)

    numeric_rows: List[Dict[str, Any]] = []
    for result in results:
        if not result.features:
            continue
        numeric_features = _numeric_feature_subset(result.features)
        numeric_features["track_id"] = result.item.track_id
        numeric_features["metadata_index"] = result.item.metadata_index
        numeric_rows.append(numeric_features)

    if numeric_rows:
        feature_matrix = pd.DataFrame(numeric_rows)
        feature_matrix.sort_values("metadata_index", inplace=True)
    else:
        feature_matrix = pd.DataFrame(columns=["track_id", "metadata_index"])

    feature_matrix_path = run_dir / "features_numeric.csv"
    feature_matrix.to_csv(feature_matrix_path, index=False)

    metadata_join_path: Optional[Path] = None
    if config.metadata_csv:
        metadata_df = pd.read_csv(config.metadata_csv, low_memory=False)
        filtered_metadata = _select_metadata_rows(
            metadata_df,
            config.metadata_split_column,
            config.include_splits,
        )
        filtered_metadata = filtered_metadata.reset_index(drop=True)
        merged = filtered_metadata.copy()
        merged["metadata_index"] = merged.index
        merged = merged.merge(
            feature_matrix,
            on="metadata_index",
            how="left",
            suffixes=("", "_features"),
        )
        metadata_join_path = run_dir / "features_with_metadata.csv"
        merged.to_csv(metadata_join_path, index=False)

    config_path = run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        config_payload = config.to_serialisable_dict()
        config_payload["run_id"] = run_id
        json.dump(config_payload, handle, indent=2)

    summary = {
        "run_id": run_id,
        "processed_files": len(results),
        "success_count": sum(1 for result in results if result.features),
        "failure_count": sum(1 for result in results if result.error),
        "index_path": str(index_path),
        "feature_matrix_path": str(feature_matrix_path),
        "metadata_join_path": str(metadata_join_path) if metadata_join_path else None,
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info(
        "Completed feature extraction: %s succeeded, %s failed.",
        summary["success_count"],
        summary["failure_count"],
    )

    return FeatureExtractionReport(
        run_id=run_id,
        run_dir=run_dir,
        feature_matrix_path=feature_matrix_path,
        index_path=index_path,
        metadata_join_path=metadata_join_path,
        config_path=config_path,
        log_path=log_path,
        processed_files=summary["processed_files"],
        success_count=summary["success_count"],
        failure_count=summary["failure_count"],
        error_records=error_records,
    )
