"""Metadata adaptation for feature extraction pipeline.

This module provides intelligent metadata transformation capabilities, allowing
the feature extraction pipeline to work with raw metadata files that lack
explicit file path columns. It infers file paths from ID columns and folder
structure, enabling out-of-the-box processing of diverse dataset formats.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

__version__ = "1.0.0"


@dataclass(frozen=True)
class MetadataAdapterConfig:
    """Configuration for metadata adaptation."""

    dataset_root: Path
    metadata_csv: Path
    output_csv: Optional[Path] = None  # None = use temp file
    metadata_index_column: str = "id"
    metadata_path_column: str = "file_path"
    metadata_split_column: Optional[str] = "split"
    extensions: Sequence[str] = field(default_factory=lambda: (".mid", ".midi"))
    path_inference_strategy: str = "auto"  # "auto", "id_match", "split_based", "glob"
    split_to_folder_map: Optional[Dict[str, str]] = None
    use_relative_paths: bool = False
    validate_paths: bool = True
    add_provenance: bool = True


@dataclass
class AdaptationReport:
    """Report of metadata adaptation process."""

    strategy_used: str
    input_metadata: Dict[str, Any]
    output_metadata: Dict[str, Any]
    path_inference: Dict[str, Any]
    validation: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


def _normalize_extensions(extensions: Sequence[str]) -> List[str]:
    """Normalize file extensions to start with dot."""
    normalized = []
    for ext in extensions:
        if not ext:
            continue
        if not ext.startswith("."):
            normalized.append(f".{ext}")
        else:
            normalized.append(ext)
    return normalized


def _discover_midi_files(
    dataset_root: Path,
    extensions: Sequence[str],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Discover all MIDI files and build stem-to-path mapping.

    Returns:
        Dictionary mapping file stem (without extension) to absolute path.
    """
    logger.info(f"Scanning {dataset_root} for MIDI files...")
    extensions = _normalize_extensions(extensions)

    midi_files: Dict[str, Path] = {}
    duplicates: List[str] = []

    for ext in extensions:
        for midi_path in dataset_root.rglob(f"*{ext}"):
            if midi_path.is_file():
                stem = midi_path.stem
                if stem in midi_files:
                    duplicates.append(stem)
                    logger.warning(
                        f"Duplicate file stem '{stem}': {midi_files[stem]} and {midi_path}"
                    )
                else:
                    midi_files[stem] = midi_path

    logger.info(f"Found {len(midi_files)} MIDI files")
    if duplicates:
        logger.warning(f"Found {len(set(duplicates))} duplicate stems")

    return midi_files


def _infer_paths_id_match(
    metadata_df: pd.DataFrame,
    id_column: str,
    midi_files: Dict[str, Path],
    dataset_root: Path,
    use_relative_paths: bool,
    logger: logging.Logger,
) -> tuple[List[Optional[str]], List[str], int, int]:
    """Infer file paths by matching IDs to file stems.

    Returns:
        (inferred_paths, unmatched_ids, matched_count, unmatched_count)
    """
    logger.info("Using ID-based path inference strategy...")

    inferred_paths: List[Optional[str]] = []
    unmatched_ids: List[str] = []
    matched = 0
    unmatched = 0

    for idx, row in metadata_df.iterrows():
        track_id = str(row[id_column])

        if track_id in midi_files:
            file_path = midi_files[track_id]
            if use_relative_paths:
                try:
                    file_path = file_path.relative_to(dataset_root)
                except ValueError:
                    pass  # Keep absolute if relative fails
            inferred_paths.append(str(file_path))
            matched += 1
        else:
            inferred_paths.append(None)
            unmatched_ids.append(track_id)
            unmatched += 1

    logger.info(f"Matched {matched}/{len(metadata_df)} IDs to files")
    if unmatched > 0:
        logger.warning(f"{unmatched} IDs could not be matched to files")

    return inferred_paths, unmatched_ids, matched, unmatched


def _infer_paths_split_based(
    metadata_df: pd.DataFrame,
    id_column: str,
    split_column: str,
    dataset_root: Path,
    extensions: Sequence[str],
    split_to_folder_map: Optional[Dict[str, str]],
    use_relative_paths: bool,
    logger: logging.Logger,
) -> tuple[List[Optional[str]], List[str], int, int]:
    """Infer file paths using split column and folder structure.

    Returns:
        (inferred_paths, unmatched_ids, matched_count, unmatched_count)
    """
    logger.info("Using split-based path inference strategy...")

    extensions = _normalize_extensions(extensions)
    inferred_paths: List[Optional[str]] = []
    unmatched_ids: List[str] = []
    matched = 0
    unmatched = 0

    # If no custom mapping, try to auto-detect split folders
    if split_to_folder_map is None:
        split_to_folder_map = _detect_split_folders(
            dataset_root, metadata_df[split_column].unique(), logger
        )

    for idx, row in metadata_df.iterrows():
        track_id = str(row[id_column])
        split_value = str(row[split_column]) if split_column in row else None

        if split_value and split_value in split_to_folder_map:
            folder_path = split_to_folder_map[split_value]
            # Try each extension
            found = False
            for ext in extensions:
                candidate = dataset_root / folder_path / f"{track_id}{ext}"
                if candidate.exists():
                    if use_relative_paths:
                        try:
                            candidate = candidate.relative_to(dataset_root)
                        except ValueError:
                            pass
                    inferred_paths.append(str(candidate))
                    matched += 1
                    found = True
                    break

            if not found:
                inferred_paths.append(None)
                unmatched_ids.append(track_id)
                unmatched += 1
        else:
            inferred_paths.append(None)
            unmatched_ids.append(track_id)
            unmatched += 1

    logger.info(f"Matched {matched}/{len(metadata_df)} IDs to files")
    if unmatched > 0:
        logger.warning(f"{unmatched} IDs could not be matched to files")

    return inferred_paths, unmatched_ids, matched, unmatched


def _detect_split_folders(
    dataset_root: Path,
    split_values: Sequence[str],
    logger: logging.Logger,
) -> Dict[str, str]:
    """Auto-detect folder structure for split values."""
    split_map = {}

    for split_val in split_values:
        split_val_str = str(split_val)
        # Try common patterns - prioritize deepest structure first
        candidates = [
            f"{split_val_str}/raw",  # train/raw/
            split_val_str,  # train/
            f"full/{split_val_str}/raw",  # full/train/raw/
            f"full/{split_val_str}",  # full/train/
        ]

        for candidate in candidates:
            candidate_path = dataset_root / candidate
            if candidate_path.exists() and candidate_path.is_dir():
                # Check if it actually contains MIDI files
                has_midi = any(candidate_path.glob("*.mid")) or any(
                    candidate_path.glob("*.midi")
                )
                if has_midi:
                    split_map[split_val_str] = candidate
                    logger.debug(f"Detected split '{split_val_str}' → '{candidate}'")
                    break

    return split_map


def _determine_strategy(
    config: MetadataAdapterConfig,
    metadata_df: pd.DataFrame,
    logger: logging.Logger,
) -> str:
    """Determine the best path inference strategy."""
    if config.path_inference_strategy != "auto":
        return config.path_inference_strategy

    has_split = (
        config.metadata_split_column
        and config.metadata_split_column in metadata_df.columns
    )

    # Check if split-based folders exist
    if has_split:
        split_values = metadata_df[config.metadata_split_column].unique()
        split_folders_exist = any(
            (config.dataset_root / str(sv)).exists() for sv in split_values
        )
        if split_folders_exist:
            logger.info("Auto-detected: split-based organization")
            return "split_based"

    # Default to ID matching via glob
    logger.info("Auto-detected: ID-based matching")
    return "id_match"


def adapt_metadata(
    config: MetadataAdapterConfig,
    logger: Optional[logging.Logger] = None,
) -> AdaptationReport:
    """Adapt metadata by inferring missing file paths.

    Args:
        config: Adaptation configuration
        logger: Optional logger instance

    Returns:
        AdaptationReport with details of the adaptation process
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    warnings: List[str] = []
    errors: List[str] = []

    # Resolve dataset_root to absolute path for consistent path handling
    dataset_root = config.dataset_root.resolve()

    # Load metadata
    logger.info(f"Loading metadata from {config.metadata_csv}")
    try:
        metadata_df = pd.read_csv(config.metadata_csv, low_memory=False)
    except Exception as exc:
        errors.append(f"Failed to load metadata CSV: {exc}")
        return AdaptationReport(
            strategy_used="none",
            input_metadata={"path": str(config.metadata_csv), "rows": 0, "columns": []},
            output_metadata={"path": "", "rows": 0, "new_columns": []},
            path_inference={
                "matched": 0,
                "unmatched": 0,
                "match_rate": 0.0,
                "unmatched_ids": [],
            },
            validation={
                "validated": False,
                "existing_files": 0,
                "missing_files": 0,
                "missing_paths": [],
            },
            warnings=warnings,
            errors=errors,
        )

    input_rows = len(metadata_df)
    input_columns = list(metadata_df.columns)
    logger.info(f"Loaded {input_rows} rows with {len(input_columns)} columns")

    # Check if file_path already exists
    if config.metadata_path_column in metadata_df.columns:
        logger.info(
            f"Column '{config.metadata_path_column}' already exists, no adaptation needed"
        )
        return AdaptationReport(
            strategy_used="none",
            input_metadata={
                "path": str(config.metadata_csv),
                "rows": input_rows,
                "columns": input_columns,
            },
            output_metadata={
                "path": str(config.metadata_csv),
                "rows": input_rows,
                "new_columns": [],
            },
            path_inference={
                "matched": input_rows,
                "unmatched": 0,
                "match_rate": 1.0,
                "unmatched_ids": [],
            },
            validation={
                "validated": False,
                "existing_files": 0,
                "missing_files": 0,
                "missing_paths": [],
            },
            warnings=[],
            errors=[],
        )

    # Verify ID column exists
    if config.metadata_index_column not in metadata_df.columns:
        errors.append(
            f"ID column '{config.metadata_index_column}' not found in metadata"
        )
        return AdaptationReport(
            strategy_used="none",
            input_metadata={
                "path": str(config.metadata_csv),
                "rows": input_rows,
                "columns": input_columns,
            },
            output_metadata={"path": "", "rows": 0, "new_columns": []},
            path_inference={
                "matched": 0,
                "unmatched": 0,
                "match_rate": 0.0,
                "unmatched_ids": [],
            },
            validation={
                "validated": False,
                "existing_files": 0,
                "missing_files": 0,
                "missing_paths": [],
            },
            warnings=warnings,
            errors=errors,
        )

    # Determine strategy
    strategy = _determine_strategy(config, metadata_df, logger)

    # Infer paths based on strategy
    if strategy == "split_based":
        inferred_paths, unmatched_ids, matched, unmatched = _infer_paths_split_based(
            metadata_df,
            config.metadata_index_column,
            config.metadata_split_column or "split",
            dataset_root,
            config.extensions,
            config.split_to_folder_map,
            config.use_relative_paths,
            logger,
        )
    else:  # id_match or glob
        midi_files = _discover_midi_files(dataset_root, config.extensions, logger)
        inferred_paths, unmatched_ids, matched, unmatched = _infer_paths_id_match(
            metadata_df,
            config.metadata_index_column,
            midi_files,
            dataset_root,
            config.use_relative_paths,
            logger,
        )

    # Add inferred paths to dataframe
    metadata_df[config.metadata_path_column] = inferred_paths
    new_columns = [config.metadata_path_column]

    # Validate paths if requested
    existing_files = 0
    missing_files = 0
    missing_paths: List[str] = []

    if config.validate_paths:
        logger.info("Validating inferred paths...")
        file_exists_list = []
        for path_str in inferred_paths:
            if path_str is None:
                file_exists_list.append(False)
                missing_files += 1
            else:
                path = Path(path_str)
                if not path.is_absolute():
                    path = dataset_root / path
                exists = path.exists()
                file_exists_list.append(exists)
                if exists:
                    existing_files += 1
                else:
                    missing_files += 1
                    missing_paths.append(str(path_str))

        metadata_df["file_exists"] = file_exists_list
        new_columns.append("file_exists")
        logger.info(f"Validation: {existing_files} exist, {missing_files} missing")

    # Add provenance if requested
    if config.add_provenance:
        metadata_df["_adapted"] = datetime.utcnow().isoformat()
        metadata_df["_adapter_strategy"] = strategy
        metadata_df["_adapter_version"] = __version__
        new_columns.extend(["_adapted", "_adapter_strategy", "_adapter_version"])

    # Determine output path
    if config.output_csv:
        output_path = config.output_csv
    else:
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = temp_dir / f"adapted_metadata_{timestamp}.csv"

    # Write adapted metadata
    logger.info(f"Writing adapted metadata to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)

    match_rate = matched / input_rows if input_rows > 0 else 0.0

    # Generate warnings
    if match_rate < 0.95:
        warnings.append(f"Low match rate: {match_rate:.1%} ({matched}/{input_rows})")
    if missing_files > 0:
        warnings.append(f"{missing_files} inferred paths point to non-existent files")

    return AdaptationReport(
        strategy_used=strategy,
        input_metadata={
            "path": str(config.metadata_csv),
            "rows": input_rows,
            "columns": input_columns,
        },
        output_metadata={
            "path": str(output_path),
            "rows": len(metadata_df),
            "new_columns": new_columns,
        },
        path_inference={
            "matched": matched,
            "unmatched": unmatched,
            "match_rate": match_rate,
            "unmatched_ids": unmatched_ids[:100],  # Limit to first 100
        },
        validation={
            "validated": config.validate_paths,
            "existing_files": existing_files,
            "missing_files": missing_files,
            "missing_paths": missing_paths[:100],  # Limit to first 100
        },
        warnings=warnings,
        errors=errors,
    )
