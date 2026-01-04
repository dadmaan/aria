"""Dataset validation for feature extraction pipeline.

This module provides comprehensive validation of dataset structure and metadata
schema before feature extraction, helping identify issues early and suggesting
corrective actions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


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


def _scan_directory_structure(
    dataset_root: Path,
    extensions: Sequence[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Scan dataset directory for MIDI files and analyze structure."""
    logger.info(f"Scanning directory: {dataset_root}")

    extensions = _normalize_extensions(extensions)
    midi_files: List[Path] = []

    for ext in extensions:
        midi_files.extend(dataset_root.rglob(f"*{ext}"))

    # Analyze organization pattern
    folder_structure: Dict[str, int] = {}
    for midi_file in midi_files:
        relative = midi_file.relative_to(dataset_root)
        # Get first-level subdirectory
        if len(relative.parts) > 1:
            first_dir = relative.parts[0]
            folder_structure[first_dir] = folder_structure.get(first_dir, 0) + 1

    # Determine organization pattern
    if not folder_structure:
        organization_pattern = "flat"
    elif any(
        key in ["train", "val", "test", "validation"] for key in folder_structure.keys()
    ):
        organization_pattern = "split_based"
    else:
        organization_pattern = "hierarchical"

    return {
        "midi_files_found": len(midi_files),
        "folder_structure": folder_structure,
        "organization_pattern": organization_pattern,
        "extensions": list(extensions),
        "issues": [],
    }


def _validate_metadata_schema(
    metadata_df: pd.DataFrame,
    metadata_path_column: str,
    metadata_index_column: str,
    metadata_split_column: Optional[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Validate metadata CSV schema."""
    logger.info("Validating metadata schema...")

    columns = list(metadata_df.columns)
    missing_required = []
    issues = []

    has_file_path = metadata_path_column in columns
    has_id = metadata_index_column in columns
    has_split = metadata_split_column and metadata_split_column in columns

    if not has_id:
        missing_required.append(metadata_index_column)
        issues.append(f"Missing required ID column: '{metadata_index_column}'")

    # Extract ID pattern if ID column exists
    id_pattern = None
    if has_id:
        sample_ids = metadata_df[metadata_index_column].head(10).astype(str)
        # Try to detect pattern (e.g., "commu00001", "track_001")
        first_id = str(sample_ids.iloc[0])
        if first_id:
            # Simple pattern detection
            import re

            if re.match(r"^[a-z]+\d+$", first_id):
                id_pattern = "alphanumeric"
            elif re.match(r"^\d+$", first_id):
                id_pattern = "numeric"
            else:
                id_pattern = "custom"

    return {
        "rows": len(metadata_df),
        "columns": columns,
        "missing_required": missing_required,
        "has_file_path": has_file_path,
        "has_id": has_id,
        "has_split": has_split,
        "id_pattern": id_pattern,
        "issues": issues,
    }


def _validate_compatibility(
    metadata_df: pd.DataFrame,
    metadata_index_column: str,
    metadata_path_column: str,
    dataset_root: Path,
    extensions: Sequence[str],
    midi_files_found: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Validate compatibility between metadata and files."""
    logger.info("Validating metadata-file compatibility...")

    issues = []
    can_match_files = False
    match_strategy = None
    matched_count = 0
    unmatched_ids: List[str] = []

    # If file_path column exists, validate paths
    if metadata_path_column in metadata_df.columns:
        can_match_files = True
        match_strategy = "explicit_paths"

        for idx, row in metadata_df.head(100).iterrows():  # Sample first 100
            path_value = row[metadata_path_column]
            if pd.isna(path_value):
                continue

            path = Path(str(path_value))
            if not path.is_absolute():
                path = dataset_root / path

            if path.exists():
                matched_count += 1
            else:
                if len(unmatched_ids) < 10:
                    unmatched_ids.append(str(row[metadata_index_column]))

        # Extrapolate to full dataset
        sample_rate = matched_count / min(100, len(metadata_df))
        matched_count = int(sample_rate * len(metadata_df))

    elif metadata_index_column in metadata_df.columns and midi_files_found > 0:
        # Try ID-based matching
        can_match_files = True
        match_strategy = "id_inference"

        # Build quick lookup of file stems
        extensions_norm = _normalize_extensions(extensions)
        file_stems = set()
        for ext in extensions_norm:
            for midi_file in dataset_root.rglob(f"*{ext}"):
                file_stems.add(midi_file.stem)

        # Sample matching
        for idx, row in metadata_df.head(100).iterrows():
            track_id = str(row[metadata_index_column])
            if track_id in file_stems:
                matched_count += 1
            else:
                if len(unmatched_ids) < 10:
                    unmatched_ids.append(track_id)

        # Extrapolate
        sample_rate = matched_count / min(100, len(metadata_df))
        matched_count = int(sample_rate * len(metadata_df))
    else:
        issues.append(
            "Cannot match metadata to files: missing both file_path and id columns"
        )

    return {
        "can_match_files": can_match_files,
        "match_strategy": match_strategy,
        "matched_count": matched_count,
        "unmatched_ids": unmatched_ids,
        "issues": issues,
    }


def validate_dataset(
    dataset_root: Path,
    metadata_csv: Optional[Path],
    metadata_index_column: str = "id",
    metadata_path_column: str = "file_path",
    metadata_split_column: Optional[str] = "split",
    extensions: Sequence[str] = (".mid", ".midi"),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Validate dataset structure and metadata for feature extraction.

    Args:
        dataset_root: Root directory containing MIDI files
        metadata_csv: Path to metadata CSV (optional)
        metadata_index_column: Column name for track IDs
        metadata_path_column: Column name for file paths
        metadata_split_column: Column name for split information
        extensions: MIDI file extensions to search for
        logger: Optional logger instance

    Returns:
        Validation report dictionary
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting dataset validation...")

    # Initialize report
    report = {
        "status": "pass",
        "metadata_validation": {},
        "dataset_validation": {},
        "compatibility_validation": {},
        "recommendations": [],
    }

    # Validate dataset structure
    if not dataset_root.exists():
        report["status"] = "fail"
        report["dataset_validation"]["issues"] = [
            f"Dataset root does not exist: {dataset_root}"
        ]
        report["recommendations"].append(
            f"Create directory or fix path: {dataset_root}"
        )
        return report

    dataset_validation = _scan_directory_structure(dataset_root, extensions, logger)
    report["dataset_validation"] = dataset_validation

    if dataset_validation["midi_files_found"] == 0:
        report["status"] = "fail"
        dataset_validation["issues"].append("No MIDI files found in dataset root")
        report["recommendations"].append(
            f"Ensure MIDI files with extensions {extensions} exist under {dataset_root}"
        )

    # Validate metadata if provided
    if metadata_csv:
        if not metadata_csv.exists():
            report["status"] = "fail"
            report["metadata_validation"]["issues"] = [
                f"Metadata CSV not found: {metadata_csv}"
            ]
            report["recommendations"].append(
                f"Create or fix metadata CSV path: {metadata_csv}"
            )
            return report

        try:
            metadata_df = pd.read_csv(metadata_csv, low_memory=False)
        except Exception as exc:
            report["status"] = "fail"
            report["metadata_validation"]["issues"] = [
                f"Failed to load metadata CSV: {exc}"
            ]
            report["recommendations"].append("Fix CSV format or encoding issues")
            return report

        metadata_validation = _validate_metadata_schema(
            metadata_df,
            metadata_path_column,
            metadata_index_column,
            metadata_split_column,
            logger,
        )
        report["metadata_validation"] = metadata_validation

        if metadata_validation["issues"]:
            report["status"] = "warning"

        # Validate compatibility
        compatibility_validation = _validate_compatibility(
            metadata_df,
            metadata_index_column,
            metadata_path_column,
            dataset_root,
            extensions,
            dataset_validation["midi_files_found"],
            logger,
        )
        report["compatibility_validation"] = compatibility_validation

        if compatibility_validation["issues"]:
            report["status"] = "fail"
            report["recommendations"].extend(compatibility_validation["issues"])

        # Generate recommendations
        if not metadata_validation["has_file_path"]:
            if metadata_validation["has_id"]:
                report["recommendations"].append(
                    "Missing 'file_path' column. Use --auto-adapt flag to infer paths from IDs."
                )
            else:
                report["status"] = "fail"
                report["recommendations"].append(
                    f"Metadata must have either '{metadata_path_column}' or '{metadata_index_column}' column."
                )

        if compatibility_validation["can_match_files"]:
            match_rate = (
                compatibility_validation["matched_count"] / metadata_validation["rows"]
            )
            if match_rate < 0.95:
                report["status"] = "warning"
                report["recommendations"].append(
                    f"Low file match rate: {match_rate:.1%}. Some files may be missing."
                )
    else:
        # No metadata provided - direct directory scan mode
        report["metadata_validation"] = {
            "rows": 0,
            "columns": [],
            "missing_required": [],
            "has_file_path": False,
            "has_id": False,
            "has_split": False,
            "id_pattern": None,
            "issues": [],
        }
        report["compatibility_validation"] = {
            "can_match_files": True,
            "match_strategy": "directory_scan",
            "matched_count": dataset_validation["midi_files_found"],
            "unmatched_ids": [],
            "issues": [],
        }
        report["recommendations"].append(
            "No metadata provided. Files will be discovered via directory scan."
        )

    logger.info(f"Validation complete: {report['status']}")
    return report


def print_validation_report(report: Dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    status = report["status"]
    status_symbol = {
        "pass": "✓",
        "warning": "⚠",
        "fail": "✗",
    }.get(status, "?")

    print(f"\n{status_symbol} Validation Status: {status.upper()}")
    print("=" * 60)

    # Metadata validation
    if report["metadata_validation"]:
        mv = report["metadata_validation"]
        print("\nMetadata Validation:")
        if mv.get("rows", 0) > 0:
            print(f"  ✓ Metadata CSV found: {mv['rows']:,} rows")
        if mv.get("has_file_path"):
            print(f"  ✓ Has '{mv.get('path_column', 'file_path')}' column")
        else:
            print(f"  ✗ Missing 'file_path' column")
        if mv.get("has_id"):
            print(f"  ✓ Has ID column with pattern: {mv.get('id_pattern', 'unknown')}")
        if mv.get("has_split"):
            print(f"  ✓ Has split column")
        if mv.get("issues"):
            for issue in mv["issues"]:
                print(f"  ✗ {issue}")

    # Dataset validation
    if report["dataset_validation"]:
        dv = report["dataset_validation"]
        print("\nDataset Validation:")
        print(f"  ✓ Found {dv['midi_files_found']:,} MIDI files")
        print(f"  ✓ Organization: {dv['organization_pattern']}")
        if dv.get("folder_structure"):
            print(f"  ✓ Folder structure:")
            for folder, count in sorted(dv["folder_structure"].items()):
                print(f"      {folder}/: {count} files")
        if dv.get("issues"):
            for issue in dv["issues"]:
                print(f"  ✗ {issue}")

    # Compatibility validation
    if report["compatibility_validation"]:
        cv = report["compatibility_validation"]
        print("\nCompatibility Check:")
        if cv.get("can_match_files"):
            print(f"  ✓ Can match files using: {cv.get('match_strategy', 'unknown')}")
            if cv.get("matched_count"):
                print(f"  ✓ Matched: {cv['matched_count']:,} files")
        else:
            print(f"  ✗ Cannot match metadata to files")
        if cv.get("unmatched_ids"):
            print(f"  ⚠ Sample unmatched IDs: {', '.join(cv['unmatched_ids'][:5])}")
        if cv.get("issues"):
            for issue in cv["issues"]:
                print(f"  ✗ {issue}")

    # Recommendations
    if report["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

    print("=" * 60)
