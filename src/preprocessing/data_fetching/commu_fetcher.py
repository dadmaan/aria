"""Utilities for curating the ComMu Bass subset.

This module provides a small, testable surface that powers the
`scripts/fetch_commu_bass.py` CLI.  The fetcher performs three steps:

1. Load a raw metadata table that contains file level information.
2. Filter the rows down to the bass-only subset using a configurable
   pattern match on an instrumentation column.
3. Copy the matching files into a dedicated directory and emit a clean
   metadata CSV with a deterministic train/validation/test split.

The public entry-point is :func:`curate_bass_subset`.  The function is
pure with respect to randomness (the split is driven by an explicit
seed) which keeps it easy to test.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.utils.logging.logging_manager import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """Ratios used to partition the cleaned dataset."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):  # Check if ratios sum to 1
            raise ValueError(
                "Split ratios must add up to 1.0 "
                f"(got {total:.4f} for train/val/test)."
            )
        for name, value in (
            ("train_ratio", self.train_ratio),
            ("val_ratio", self.val_ratio),
            ("test_ratio", self.test_ratio),
        ):
            if value < 0:  # Ensure non-negative ratios
                raise ValueError(f"{name} must be non-negative (got {value}).")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load raw metadata from CSV.

    The function prefers UTF-8 encoded CSV files.  Any duplicate rows are
    dropped to provide a stable base for downstream processing.
    """

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    LOGGER.debug("Loading metadata from %s", metadata_path)
    df = pd.read_csv(metadata_path)
    df = df.drop_duplicates().reset_index(drop=True)  # Remove duplicates
    return df


def filter_bass_rows(
    df: pd.DataFrame,
    instrument_columns: Union[str, Iterable[str]] = ("inst", "track_role"),
    pattern: str = "bass",
) -> pd.DataFrame:
    """Return rows where any of the ``instrument_columns`` contains ``pattern``.

    Both the column values and the pattern are compared case-insensitively. When
    a string is provided the function interprets it as a comma-separated list of
    column names.  Missing columns are ignored as long as at least one
    candidate exists in the dataframe.
    """

    if isinstance(instrument_columns, str):
        requested = [
            col.strip() for col in instrument_columns.split(",") if col.strip()
        ]  # Parse string to list
    else:
        requested = [str(col).strip() for col in instrument_columns]

    fallback = ["inst", "track_role", "instrument", "role"]
    search_columns = []
    for column in requested + fallback:
        if column in df.columns and column not in search_columns:
            search_columns.append(column)  # Collect valid columns

    if not search_columns:
        raise KeyError(
            "None of the instrument columns were found in the metadata. Tried: "
            f"{requested or instrument_columns}. Available columns: {df.columns.tolist()}"
        )

    mask = pd.Series(False, index=df.index, dtype=bool)  # Initialize mask
    for column in search_columns:
        column_mask = (
            df[column]
            .fillna("")
            .astype(str)
            .str.contains(pattern, case=False, regex=True)
        )
        mask |= column_mask  # OR with existing mask

    filtered = df.loc[mask].copy()
    LOGGER.debug(
        "Filtered %s rows -> %s bass rows using columns %s",
        len(df),
        len(filtered),
        ", ".join(search_columns),
    )
    return filtered


def _resolve_file_paths(df: pd.DataFrame) -> pd.Series:
    """Return a series with relative MIDI paths for each row."""

    if "file_path" in df.columns:
        return df["file_path"].astype(str)  # Use existing file_path

    for candidate in ("filepath", "midi_path", "path"):
        if candidate in df.columns:
            return df[candidate].astype(str)  # Use alternative column

    if {"split_data", "id"}.issubset(df.columns):
        split_series = (
            df["split_data"].fillna("train").astype(str).str.strip().str.lower()
        )
        midi_ids = df["id"].astype(str).str.strip()
        return split_series.map(
            lambda split: f"full/{split}/raw/"
        ) + midi_ids.map(  # Construct path from split and id
            lambda midi_id: f"{midi_id}.mid"
        )

    raise KeyError(
        "Metadata must provide a column with relative file paths (one of "
        "'file_path', 'filepath', 'midi_path', 'path') or the pair ('split_data', 'id')."
    )


def _augment_bass_metadata(
    df: pd.DataFrame,
    output_raw_root: Path,
    common_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Preserve original columns while ensuring a ``file_path`` field exists.

    Parameters
    ----------
    df:
        Input dataframe with bass metadata.
    output_raw_root:
        Directory where files will be copied.
    common_root:
        Optional common base directory. When provided, file_path will be stored
        relative to this directory instead of output_raw_root. This enables
        multiple datasets to share a common reference point.
    """

    augmented = df.copy()
    raw_paths = _resolve_file_paths(df)  # Get original relative paths

    if common_root is not None:
        # Compute dataset prefix relative to common_root
        try:
            dataset_prefix = output_raw_root.relative_to(common_root)
        except ValueError:
            # output_raw_root is not under common_root
            LOGGER.warning(
                "output_raw_root (%s) is not under common_root (%s). "
                "Falling back to paths relative to output_raw_root.",
                output_raw_root,
                common_root,
            )
            augmented["file_path"] = raw_paths
            return augmented

        # Prepend dataset prefix to each path
        augmented["file_path"] = raw_paths.apply(lambda p: str(dataset_prefix / p))
    else:
        augmented["file_path"] = raw_paths

    return augmented


def _assign_splits(
    df: pd.DataFrame,
    split_config: SplitConfig,
    seed: int,
) -> pd.DataFrame:
    split_config.validate()
    rng = np.random.default_rng(seed)  # Create seeded random generator
    shuffled_indices = rng.permutation(len(df))  # Shuffle indices

    train_cutoff = int(len(df) * split_config.train_ratio)
    val_cutoff = train_cutoff + int(len(df) * split_config.val_ratio)

    splits = np.empty(len(df), dtype=object)
    splits[shuffled_indices[:train_cutoff]] = "train"
    splits[shuffled_indices[train_cutoff:val_cutoff]] = "val"
    splits[shuffled_indices[val_cutoff:]] = "test"

    df = df.copy()
    df["split"] = splits  # Add split column
    return df


def copy_files(
    df: pd.DataFrame,
    source_root: Path,
    destination_root: Path,
    overwrite: bool = False,
    num_workers: int = 1,
) -> Iterable[Path]:
    """Copy the files referenced in ``df`` to ``destination_root``.

    Parameters
    ----------
    df:
        Dataframe with a ``file_path`` column containing paths relative to
        ``source_root``.
    source_root:
        Base directory for existing files.
    destination_root:
        Where the bass subset will be replicated (same relative layout).
    overwrite:
        When ``True`` existing files will be replaced.  Otherwise they are
        left untouched.
    num_workers:
        Number of worker threads to use for copying files. More workers
        may speed up the operation but will consume more system resources.
    """

    destination_root.mkdir(parents=True, exist_ok=True)

    def _copy_single(row):
        relative_path = Path(row["file_path"]).as_posix()
        source_path = (source_root / relative_path).resolve()
        target_path = (destination_root / relative_path).resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing source file referenced in metadata: {source_path}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not overwrite:
            LOGGER.debug("Skipping existing file %s", target_path)
            return target_path
        else:
            shutil.copy2(source_path, target_path)  # Copy with metadata
            return target_path

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            copied_paths = list(executor.map(_copy_single, df.to_dict("records")))
    else:
        copied_paths = [_copy_single(row) for row in df.to_dict("records")]

    return copied_paths


def _copy_file(
    row: pd.Series, source_root: Path, destination_root: Path, overwrite: bool
) -> Path:
    """Helper function to copy a single file, used by copy_files in a thread pool."""

    relative_path = Path(row["file_path"]).as_posix()
    source_path = (source_root / relative_path).resolve()
    target_path = (destination_root / relative_path).resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing source file referenced in metadata: {source_path}"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        LOGGER.debug("Skipping existing file %s", target_path)
    else:
        shutil.copy2(source_path, target_path)  # Copy with metadata

    return target_path


def curate_bass_subset(
    metadata_path: Path,
    source_root: Path,
    output_raw_root: Path,
    output_processed_root: Path,
    *,
    instrument_columns: Union[str, Iterable[str]] = ("inst", "track_role"),
    pattern: str = "bass",
    split_config: Optional[SplitConfig] = None,
    seed: int = 7,
    overwrite: bool = False,
    num_workers: int = 1,
    common_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main orchestration helper used by the CLI.

    Parameters
    ----------
    metadata_path:
        Path to source metadata CSV file.
    source_root:
        Root directory containing source MIDI files.
    output_raw_root:
        Directory where subset files will be copied.
    output_processed_root:
        Directory for cleaned metadata output.
    instrument_columns:
        Columns to search for bass instrument pattern.
    pattern:
        Regex pattern to identify bass rows.
    split_config:
        Train/val/test split ratios.
    seed:
        Random seed for deterministic splits.
    overwrite:
        Whether to overwrite existing files.
    num_workers:
        Number of parallel workers for file copying.
    common_root:
        Optional common base directory for relative paths. When provided,
        file_path entries will be stored relative to this directory,
        enabling proper path resolution in combined datasets.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Raw bass metadata and cleaned metadata with split information.
    """

    if split_config is None:
        split_config = SplitConfig()  # Use default split config

    raw_df = load_metadata(metadata_path)  # Load full metadata
    bass_df = filter_bass_rows(
        raw_df, instrument_columns=instrument_columns, pattern=pattern
    )  # Filter to bass
    if bass_df.empty:
        raise ValueError("No bass rows matched the provided instrument filters.")

    # First, augment metadata with original paths for copying
    augmented = _augment_bass_metadata(
        bass_df, output_raw_root=output_raw_root, common_root=None
    )  # Get original paths first

    copy_files(
        augmented,
        source_root=source_root,
        destination_root=output_raw_root,
        overwrite=overwrite,
        num_workers=num_workers,
    )  # Copy files using original paths

    # Now apply common_root transformation if provided
    if common_root is not None:
        augmented = _augment_bass_metadata(
            augmented, output_raw_root=output_raw_root, common_root=common_root
        )  # Transform paths to common_root

    cleaned = _assign_splits(
        augmented, split_config=split_config, seed=seed
    )  # Assign train/val/test

    output_raw_root.mkdir(parents=True, exist_ok=True)
    output_processed_root.mkdir(parents=True, exist_ok=True)

    raw_metadata_path = output_raw_root / "metadata.csv"
    clean_metadata_path = output_processed_root / "metadata_clean.csv"

    augmented.to_csv(raw_metadata_path, index=False)  # Save raw metadata
    cleaned.to_csv(clean_metadata_path, index=False)  # Save cleaned metadata

    LOGGER.info("Wrote raw metadata to %s", raw_metadata_path)
    LOGGER.info("Wrote cleaned metadata to %s", clean_metadata_path)

    return augmented, cleaned


__all__ = [
    "SplitConfig",
    "curate_bass_subset",
    "filter_bass_rows",
    "load_metadata",
    "copy_files",
]
