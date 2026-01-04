from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.feature_extraction import (
    FeatureExtractionConfig,
    run_feature_extraction,
)
from src.preprocessing.data_validation import validate_dataset, print_validation_report
from src.preprocessing.metadata_adapter import adapt_metadata, MetadataAdapterConfig


LOG_LEVELS: Sequence[str] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deterministic feature matrices from MIDI datasets."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root directory containing MIDI files when metadata is not provided.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/combined_metadata_clean.csv"),
        help="Optional metadata CSV with columns describing the dataset. Defaults to the combined dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/features/raw"),
        help="Directory where the run artifact folder will be created.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional identifier for the run (defaults to timestamp).",
    )
    parser.add_argument(
        "--metadata-index-column",
        type=str,
        default="id",
        help="Column name in the metadata CSV that provides a stable track identifier.",
    )
    parser.add_argument(
        "--metadata-path-column",
        type=str,
        default="file_path",
        help="Column name in the metadata CSV that points to the MIDI file.",
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default="split",
        help="Column used to filter metadata rows by split (ignored when metadata not provided).",
    )
    parser.add_argument(
        "--include-splits",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of split names to include (e.g. train val).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mid", ".midi"],
        help="MIDI file extensions to consider when scanning directories.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on the number of files to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic ordering.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Reserved for future parallelism support; currently processed sequentially.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    parser.add_argument(
        "--save-per-file",
        action="store_true",
        default=True,
        help="Save individual feature JSON files in per_file/ directory (default: True).",
    )
    parser.add_argument(
        "--no-save-per-file",
        dest="save_per_file",
        action="store_false",
        help="Skip saving individual per-file JSON outputs to reduce disk usage.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity for the run.",
    )
    parser.add_argument(
        "--auto-adapt",
        action="store_true",
        help="Automatically adapt metadata by inferring missing file_path column from IDs and folder structure.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate dataset and metadata compatibility without running extraction.",
    )
    return parser.parse_args()


def _resolve_optional_path(value: Path | None) -> Path | None:
    if value is None:
        return None
    return value.resolve()


def main() -> None:
    args = _parse_args()
    dataset_root = _resolve_optional_path(args.dataset_root)
    metadata_csv = _resolve_optional_path(args.metadata_csv)
    output_root = args.output_root.resolve()

    include_splits = args.include_splits or None
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # Setup logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Validation mode
    if args.validate_only:
        logger.info("Running validation only (no extraction)...")
        validation_report = validate_dataset(
            dataset_root=dataset_root,
            metadata_csv=metadata_csv,
            metadata_index_column=args.metadata_index_column,
            metadata_path_column=args.metadata_path_column,
            metadata_split_column=args.split_column,
            extensions=tuple(args.extensions),
            logger=logger,
        )
        print_validation_report(validation_report)

        if validation_report["status"] == "fail":
            sys.exit(1)
        elif validation_report["status"] == "warning":
            sys.exit(0)
        else:
            sys.exit(0)

    # Auto-adapt metadata if requested
    adapted_metadata_csv = metadata_csv
    if args.auto_adapt and metadata_csv:
        logger.info("🔍 Validating metadata...")
        validation_report = validate_dataset(
            dataset_root=dataset_root,
            metadata_csv=metadata_csv,
            metadata_index_column=args.metadata_index_column,
            metadata_path_column=args.metadata_path_column,
            metadata_split_column=args.split_column,
            extensions=tuple(args.extensions),
            logger=logger,
        )

        if validation_report["status"] == "fail":
            print_validation_report(validation_report)
            logger.error("Validation failed. Cannot proceed with auto-adaptation.")
            sys.exit(1)

        logger.info("✓ Validation passed")

        # Check if adaptation is needed
        if not validation_report["metadata_validation"].get("has_file_path"):
            logger.info("🔧 Adapting metadata (inferring file paths)...")

            adapter_config = MetadataAdapterConfig(
                dataset_root=dataset_root,
                metadata_csv=metadata_csv,
                metadata_index_column=args.metadata_index_column,
                metadata_path_column=args.metadata_path_column,
                metadata_split_column=args.split_column,
                extensions=tuple(args.extensions),
                path_inference_strategy="auto",
                use_relative_paths=False,
                validate_paths=True,
                add_provenance=True,
            )

            adaptation_report = adapt_metadata(adapter_config, logger=logger)

            if adaptation_report.errors:
                logger.error("Adaptation failed:")
                for error in adaptation_report.errors:
                    logger.error(f"  - {error}")
                sys.exit(1)

            match_rate = adaptation_report.path_inference["match_rate"]
            logger.info(
                f"✓ Inferred {adaptation_report.path_inference['matched']:,} file paths "
                f"({match_rate:.1%} match rate)"
            )

            if adaptation_report.warnings:
                for warning in adaptation_report.warnings:
                    logger.warning(f"  ⚠ {warning}")

            if match_rate < 0.95:
                logger.warning(
                    f"Low match rate ({match_rate:.1%}). Proceeding with matched files only."
                )

            adapted_metadata_csv = Path(adaptation_report.output_metadata["path"])
            logger.info(f"✓ Created adapted metadata: {adapted_metadata_csv}")
        else:
            logger.info("✓ Metadata already has file_path column, no adaptation needed")

    # Run feature extraction
    config = FeatureExtractionConfig(
        dataset_root=dataset_root,
        metadata_csv=adapted_metadata_csv,
        output_root=output_root,
        metadata_index_column=args.metadata_index_column,
        metadata_path_column=args.metadata_path_column,
        metadata_split_column=args.split_column,
        include_splits=include_splits,
        extensions=tuple(args.extensions),
        run_id=args.run_id,
        seed=args.seed,
        overwrite=args.overwrite,
        max_files=args.max_files,
        num_workers=args.num_workers,
        save_per_file=args.save_per_file,
    )

    logger.info("🎵 Starting feature extraction...")
    report = run_feature_extraction(config, log_level=log_level)

    print(f"\n✓ Run directory: {report.run_dir}")
    print(f"✓ Feature matrix: {report.feature_matrix_path}")
    if report.metadata_join_path:
        print(f"✓ Metadata join: {report.metadata_join_path}")
    if report.failure_count:
        print(
            f"⚠ Completed with {report.failure_count} failures. See {report.log_path} for details.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
