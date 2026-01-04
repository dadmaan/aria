"""CLI to curate the ComMu bass subset.

The script is intentionally lightweight and delegates the heavy lifting to
:func:`src.preprocessing.commu_fetcher.curate_bass_subset`.  It accepts a source
folder that mirrors the original ComMu dataset layout and emits two
artifacts:

- ``data/raw/commu/bass``: contains copied MIDI files and ``metadata.csv``
  describing the raw subset.
- ``data/processed/commu_bass/metadata_clean.csv``: a normalised table with a
  deterministic train/validation/test split.

Example
-------
```
python scripts/fetch_commu_bass.py \
    --source-dir data/raw/commu/full \
    --metadata-file data/raw/commu/full/metadata.csv
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_fetching import SplitConfig, curate_bass_subset

LOGGER = logging.getLogger("fetch_commu_bass")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/raw/commu"),
        help="Root directory of the raw ComMu dataset.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit metadata CSV. When omitted the script searches for "
            "'metadata.csv' under --source-dir."
        ),
    )
    parser.add_argument(
        "--output-raw-dir",
        type=Path,
        default=Path("data/subset/commu/bass"),
        help="Directory that will receive the bass subset files and metadata.csv.",
    )
    parser.add_argument(
        "--output-processed-dir",
        type=Path,
        default=Path("data/subset/commu/bass"),
        help="Directory for the cleaned metadata (metadata_clean.csv).",
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
        "--instrument-column",
        type=str,
        default="inst,track_role",
        help=(
            "Comma separated list of metadata columns used to detect bass rows. "
            "Defaults to 'inst,track_role' for the ComMu dataset."
        ),
    )
    parser.add_argument(
        "--instrument-pattern",
        type=str,
        default="bass",
        help="Case-insensitive regex pattern used to match bass rows.",
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
        default=7,
        help="Random seed used for the train/val/test split.",
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

    source_dir: Path = args.source_dir
    metadata_file: Path
    if args.metadata_file is None:
        metadata_file = source_dir / "metadata.csv"
    else:
        metadata_file = args.metadata_file

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    LOGGER.info(
        "Starting ComMu bass curation (source=%s, metadata=%s)",
        source_dir,
        metadata_file,
    )

    instrument_columns = [
        col.strip() for col in args.instrument_column.split(",") if col.strip()
    ]
    if not instrument_columns:
        instrument_columns = ["inst", "track_role"]

    curate_bass_subset(
        metadata_path=metadata_file,
        source_root=source_dir,
        output_raw_root=args.output_raw_dir,
        output_processed_root=args.output_processed_dir,
        instrument_columns=instrument_columns,
        pattern=args.instrument_pattern,
        split_config=split_config,
        seed=args.seed,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        common_root=args.common_root,
    )

    LOGGER.info(
        "Bass subset ready. Raw artifacts live in %s and cleaned metadata in %s.",
        args.output_raw_dir,
        args.output_processed_dir,
    )


if __name__ == "__main__":
    main()
