#!/usr/bin/env python3
"""
Add relative file paths to ComMU metadata.csv

This script adds a 'file_path' column to the metadata.csv file containing
relative paths to the corresponding MIDI files.

Usage:
    python add_filepaths_to_commu_metadata.py [--metadata PATH] [--base-dir PATH]
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.logging.logging_manager import LoggingManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add relative file paths to ComMU metadata.csv"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/workspace/data/raw/commu/metadata.csv",
        help="Path to metadata CSV file (default: /workspace/data/raw/commu/metadata.csv)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/workspace/data/raw/commu",
        help="Base directory for relative paths (default: /workspace/data/raw/commu)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup of original metadata file",
    )
    return parser.parse_args()


def create_backup(metadata_path, logger):
    """Create a timestamped backup of the metadata file.

    Args:
        metadata_path: Path to the metadata file
        logger: Logger instance

    Returns:
        Path to the backup file
    """
    metadata_file = Path(metadata_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = (
        metadata_file.parent
        / f"{metadata_file.stem}_backup_{timestamp}{metadata_file.suffix}"
    )

    try:
        shutil.copy2(metadata_path, backup_path)
        logger.info("Created backup: %s", backup_path)
        return backup_path
    except Exception as e:
        logger.error("Failed to create backup: %s", str(e))
        raise


def load_metadata(metadata_path, logger):
    """Load metadata CSV file.

    Args:
        metadata_path: Path to the metadata file
        logger: Logger instance

    Returns:
        DataFrame containing metadata
    """
    logger.info("=" * 80)
    logger.info("ADD FILE PATHS TO COMMU METADATA")
    logger.info("=" * 80)
    logger.info("Loading metadata from: %s", metadata_path)

    try:
        df = pd.read_csv(metadata_path)
        logger.info("Successfully loaded metadata: %d entries", len(df))
        logger.info("Columns: %s", ", ".join(df.columns.tolist()))
        return df
    except Exception as e:
        logger.error("Failed to load metadata: %s", str(e))
        raise


def add_file_paths(df, logger):
    """Add file_path column to metadata DataFrame.

    Args:
        df: Metadata DataFrame
        logger: Logger instance

    Returns:
        DataFrame with added file_path column
    """
    logger.info("-" * 80)
    logger.info("Adding file_path column...")

    # Check required columns
    required_columns = ["split_data", "id"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        logger.error("Missing required columns: %s", ", ".join(missing_columns))
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Create file paths
    df["file_path"] = df.apply(
        lambda row: f"full/{row['split_data']}/raw/{row['id']}.mid", axis=1
    )

    logger.info("Successfully added file_path column")
    logger.debug("Sample paths:")
    for i, path in enumerate(df["file_path"].head(3)):
        logger.debug("  [%d] %s", i + 1, path)

    return df


def verify_files(df, base_dir, logger):
    """Verify that files referenced in file_path column exist.

    Args:
        df: Metadata DataFrame with file_path column
        base_dir: Base directory for resolving relative paths
        logger: Logger instance

    Returns:
        List of missing file IDs
    """
    logger.info("-" * 80)
    logger.info("Verifying file existence...")

    base_path = Path(base_dir)
    missing_files = []

    for idx, row in df.iterrows():
        full_path = base_path / row["file_path"]
        if not full_path.exists():
            missing_files.append(row["id"])

    # Report results
    total_files = len(df)
    found_files = total_files - len(missing_files)

    logger.info("Total entries: %d", total_files)
    logger.info(
        "Files found: %d (%.2f%%)", found_files, (found_files / total_files) * 100
    )

    if missing_files:
        logger.warning(
            "Missing files: %d (%.2f%%)",
            len(missing_files),
            (len(missing_files) / total_files) * 100,
        )

        # Show first 10 missing files
        if len(missing_files) <= 10:
            logger.warning("Missing file IDs: %s", ", ".join(missing_files))
        else:
            logger.warning(
                "First 10 missing file IDs: %s", ", ".join(missing_files[:10])
            )
            logger.warning("... and %d more", len(missing_files) - 10)
    else:
        logger.info("All files verified successfully!")

    return missing_files


def save_metadata(df, metadata_path, logger):
    """Save updated metadata to CSV file.

    Args:
        df: DataFrame to save
        metadata_path: Path to save the metadata
        logger: Logger instance
    """
    logger.info("-" * 80)
    logger.info("Saving updated metadata...")

    try:
        df.to_csv(metadata_path, index=False)
        logger.info("Successfully saved metadata to: %s", metadata_path)
        logger.info("New column 'file_path' has been added")
    except Exception as e:
        logger.error("Failed to save metadata: %s", str(e))
        raise


def main():
    """Main execution function."""
    args = parse_arguments()

    # Initialize logging
    log_file = args.log_file or "/workspace/logs/add_filepaths_to_metadata.log"
    logger = LoggingManager(
        name="add_filepaths",
        log_file=log_file,
    )

    try:
        # Create backup if requested
        if not args.no_backup:
            backup_path = create_backup(args.metadata, logger)
        else:
            logger.info("Skipping backup creation (--no-backup flag set)")

        # Load metadata
        df = load_metadata(args.metadata, logger)

        # Add file paths
        df = add_file_paths(df, logger)

        # Verify files exist
        missing_files = verify_files(df, args.base_dir, logger)

        # Save updated metadata
        save_metadata(df, args.metadata, logger)

        # Final summary
        logger.info("=" * 80)
        logger.info("COMPLETION SUMMARY")
        logger.info("=" * 80)
        logger.info("Metadata file: %s", args.metadata)
        logger.info("Total entries: %d", len(df))
        logger.info("Missing files: %d", len(missing_files))
        if not args.no_backup:
            logger.info("Backup saved to: %s", backup_path)
        logger.info("=" * 80)
        logger.info("Process completed successfully!")

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("PROCESS FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
