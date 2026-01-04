"""CLI to combine processed datasets into a unified metadata file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from src.utils.logging.logging_manager import get_logger, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_fetching import combine_datasets

LOGGER = get_logger("combine_datasets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/subset"),
        help="Root directory containing processed dataset subdirectories.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/subset/combined_metadata_clean.csv"),
        help="Path to save the combined metadata CSV.",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata_clean.csv",
        help="Name of the metadata file to look for in each subdirectory.",
    )
    parser.add_argument(
        "--filter-by-source",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of sources to include (e.g. commu_bass bass_loops).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to randomly sample from the combined dataset.",
    )
    parser.add_argument(
        "--stratified-sample",
        action="store_true",
        help="When sampling, maintain split proportions.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Save the output as compressed CSV (.csv.gz).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity of log output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(name="combine_datasets", level=getattr(logging, args.log_level))

    LOGGER.info(
        f"Starting dataset combination (processed_dir={args.processed_dir}, output={args.output_path})"
    )

    combine_datasets(
        processed_dir=args.processed_dir,
        output_path=args.output_path,
        metadata_filename=args.metadata_filename,
        filter_by_source=args.filter_by_source,
        sample_size=args.sample_size,
        stratified_sample=args.stratified_sample,
        compress=args.compress,
    )

    LOGGER.info(
        f"Dataset combination complete. Combined metadata saved to {args.output_path}."
    )


if __name__ == "__main__":
    main()
