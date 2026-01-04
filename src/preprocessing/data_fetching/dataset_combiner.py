"""Utilities for combining processed datasets into a unified metadata file."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.utils.logging.logging_manager import get_logger

LOGGER = get_logger("dataset_combiner")


def combine_datasets(
    processed_dir: Path = Path("data/subset"),
    output_path: Path = Path("data/subset/combined_metadata_clean.csv"),
    metadata_filename: str = "metadata_clean.csv",
    filter_by_source: Optional[Sequence[str]] = None,
    sample_size: Optional[int] = None,
    stratified_sample: bool = False,
    compress: bool = False,
) -> pd.DataFrame:
    """Combine all metadata_clean.csv files under processed_dir into a single dataset.

    Parameters
    ----------
    processed_dir : Path
        Root directory containing processed dataset subdirectories.
    output_path : Path
        Path to save the combined metadata CSV.
    metadata_filename : str
        Name of the metadata file to look for in each subdirectory.
    filter_by_source : Optional[Sequence[str]]
        If provided, only include sources in this list.
    sample_size : Optional[int]
        If provided, randomly sample this many rows from the combined dataset.
    stratified_sample : bool
        If True and sample_size is provided, maintain split proportions.
    compress : bool
        If True, save the output as compressed CSV (.csv.gz).

    Returns
    -------
    pd.DataFrame
        The combined (and optionally filtered/sampled) DataFrame with an added 'source' column indicating the origin.
    """
    pattern = str(processed_dir / "**" / metadata_filename)
    metadata_paths = glob.glob(pattern, recursive=True)

    if not metadata_paths:
        raise FileNotFoundError(
            f"No {metadata_filename} files found under {processed_dir}"
        )

    dfs = []
    for path_str in metadata_paths:
        path = Path(path_str)
        LOGGER.info(f"Loading metadata from {path}")
        df = pd.read_csv(path)

        # Filter by source if specified
        if filter_by_source is not None:
            df = df[df["source"].isin(filter_by_source)]

        # Sample rows if sample_size is specified
        if sample_size is not None:
            if stratified_sample:
                df = (
                    df.groupby("source")
                    .apply(lambda x: x.sample(min(len(x), sample_size)))
                    .reset_index(drop=True)
                )
            else:
                df = df.sample(min(len(df), sample_size))

        source = path.parent.name  # e.g., 'commu_bass', 'bass_loops'
        df["source"] = source
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    LOGGER.info(f"Combined {len(dfs)} datasets with {len(combined)} total rows")

    # Check for ID conflicts across sources
    if "id" in combined.columns:
        duplicates = combined[combined.duplicated(subset=["id"], keep=False)]
        if not duplicates.empty:
            conflict_sources = duplicates.groupby("id")["source"].apply(list).to_dict()
            LOGGER.warning(f"Found ID conflicts across sources: {conflict_sources}")

    # Filter by source if specified
    if filter_by_source:
        combined = combined[combined["source"].isin(filter_by_source)]
        LOGGER.info(
            f"Filtered to sources {filter_by_source}, {len(combined)} rows remaining"
        )

    # Sample if specified
    if sample_size and len(combined) > sample_size:
        if stratified_sample and "split" in combined.columns:
            combined = (
                combined.groupby("split", group_keys=False)
                .apply(
                    lambda x: x.sample(
                        min(len(x), int(sample_size * len(x) / len(combined))),
                        random_state=42,
                    )
                )
                .reset_index(drop=True)
            )
            LOGGER.info(f"Stratified sampled to ~{sample_size} rows")
        else:
            combined = combined.sample(n=sample_size, random_state=42).reset_index(
                drop=True
            )
            LOGGER.info(f"Randomly sampled to {sample_size} rows")

    # Clean up index column if present
    if "Unnamed: 0" in combined.columns:
        combined = combined.drop(columns=["Unnamed: 0"])

    combined.reset_index(drop=True, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compress and not str(output_path).endswith(".gz"):
        output_path = output_path.with_suffix(output_path.suffix + ".gz")
    combined.to_csv(output_path, index=False, compression="gzip" if compress else None)
    LOGGER.info(f"Saved combined metadata to {output_path}")

    return combined


__all__ = ["combine_datasets"]
