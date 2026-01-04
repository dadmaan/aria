"""CLI for curating the Bass Loops dataset into ComMu-compatible artefacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_fetching import (
    DEFAULT_RELATIVE_ROOT,
    curate_bass_loops,
    SplitConfig,
)

LOGGER = logging.getLogger("fetch_bass_loops")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/LM_bass_loops"),
        help="Root directory containing the bass_loops dataset (expects processed/<split>/metadata.json).",
    )
    parser.add_argument(
        "--output-raw-dir",
        type=Path,
        default=Path("data/subset/LM_bass_loops_matched"),
        help="Target directory for copied MIDI files and raw metadata.csv.",
    )
    parser.add_argument(
        "--output-processed-dir",
        type=Path,
        default=Path("data/subset/LM_bass_loops_matched"),
        help="Target directory for the cleaned metadata_clean.csv.",
    )
    parser.add_argument(
        "--relative-root",
        type=str,
        default=DEFAULT_RELATIVE_ROOT,
        help="Relative prefix used inside file_path entries (defaults to 'bass_loops_midi').",
    )
    parser.add_argument(
        "--common-root",
        type=Path,
        default=None,
        help=(
            "Optional common base directory for relative paths. When provided, "
            "file_path entries will be stored relative to this directory (e.g., data/subset), "
            "enabling proper path resolution when combining multiple datasets."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of samples assigned to the train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of samples assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of samples assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling the deterministic split.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads for parallel file copying.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory if present.",
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

    logging.basicConfig(
        level=args.log_level, format="%(levelname)s %(name)s: %(message)s"
    )

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    LOGGER.info(
        "Starting Bass Loops curation (dataset_root=%s, output_raw=%s, output_processed=%s)",
        args.dataset_root,
        args.output_raw_dir,
        args.output_processed_dir,
    )

    curate_bass_loops(
        dataset_root=args.dataset_root,
        output_raw_root=args.output_raw_dir,
        output_processed_root=args.output_processed_dir,
        split_config=split_config,
        seed=args.seed,
        relative_root=args.relative_root,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        common_root=args.common_root,
    )

    LOGGER.info(
        "Bass Loops artefacts ready. Raw subset lives in %s, cleaned metadata in %s.",
        args.output_raw_dir,
        args.output_processed_dir,
    )


if __name__ == "__main__":
    main()
