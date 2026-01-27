#!/usr/bin/env python3
"""Unified preprocessing pipeline CLI for ARIA.

This script orchestrates the complete data preparation workflow from raw MIDI
files to a trained GHSOM model, producing all artifacts needed for RL training.

Usage:
    # Full pipeline from YAML config
    python scripts/preprocessing/run_preprocessing_pipeline.py --config configs/preprocessing.yaml

    # With custom run ID and seed
    python scripts/preprocessing/run_preprocessing_pipeline.py \\
        --config configs/preprocessing.yaml \\
        --run-id my_experiment \\
        --seed 123

    # Resume from existing features
    python scripts/preprocessing/run_preprocessing_pipeline.py \\
        --config configs/preprocessing.yaml \\
        --skip-feature-extraction \\
        --features-artifact artifacts/features/raw/existing_run

    # Resume from existing reduced features
    python scripts/preprocessing/run_preprocessing_pipeline.py \\
        --config configs/preprocessing.yaml \\
        --skip-feature-extraction \\
        --skip-dimensionality-reduction \\
        --reduced-artifact artifacts/features/tsne/existing_tsne

Example Output:
    The pipeline produces artifacts under artifacts/preprocessing/<run_id>/:
        - features/extracted/          Feature extraction outputs
        - reduced/reduced/             Dimensionality reduction outputs
        - ghsom/trained/               GHSOM model and cluster assignments
        - pipeline_manifest.json       Complete artifact manifest
        - pipeline_config.yaml         Full configuration snapshot

    After completion, the script prints paths to add to configs/training.yaml:
        ghsom:
          default_model_path: "artifacts/preprocessing/<run_id>/ghsom/trained/ghsom_model.pkl"
        features:
          artifact_path: "artifacts/preprocessing/<run_id>/reduced/reduced"
"""

import argparse
import logging
import sys
from pathlib import Path

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.pipeline_config import PreprocessingPipelineConfig

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the unified preprocessing pipeline for ARIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to preprocessing.yaml configuration file",
    )

    # Pipeline control overrides
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run ID (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run directory",
    )
    parser.add_argument(
        "--no-stop-on-error",
        action="store_true",
        help="Continue pipeline execution after stage failures",
    )

    # Stage skip flags
    parser.add_argument(
        "--skip-feature-extraction",
        action="store_true",
        help="Skip feature extraction stage (use with --features-artifact)",
    )
    parser.add_argument(
        "--skip-dimensionality-reduction",
        action="store_true",
        help="Skip dimensionality reduction stage (use with --reduced-artifact)",
    )
    parser.add_argument(
        "--skip-ghsom-training",
        action="store_true",
        help="Skip GHSOM training stage",
    )

    # Resume artifact paths
    parser.add_argument(
        "--features-artifact",
        type=Path,
        default=None,
        help="Path to existing feature extraction output (skips stage 1)",
    )
    parser.add_argument(
        "--reduced-artifact",
        type=Path,
        default=None,
        help="Path to existing dimensionality reduction output (skips stages 1 and 2)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=LOG_LEVELS,
        help="Override logging level",
    )

    return parser.parse_args()


def _apply_cli_overrides(
    config: PreprocessingPipelineConfig, args: argparse.Namespace
) -> None:
    """Apply command-line argument overrides to configuration.

    Args:
        config: Configuration object (modified in-place).
        args: Parsed CLI arguments.
    """
    # Pipeline control overrides
    if args.run_id is not None:
        config.run_id = args.run_id
    if args.seed is not None:
        config.seed = args.seed
    if args.output_root is not None:
        config.output_root = args.output_root
    if args.overwrite:
        config.overwrite = True
    if args.no_stop_on_error:
        config.stop_on_error = False

    # Stage skip flags
    if args.skip_feature_extraction:
        config.stages.feature_extraction = False
    if args.skip_dimensionality_reduction:
        config.stages.dimensionality_reduction = False
    if args.skip_ghsom_training:
        config.stages.ghsom_training = False

    # Resume artifact paths
    if args.features_artifact is not None:
        config.resume.features_artifact = args.features_artifact
        config.stages.feature_extraction = False  # Auto-disable stage
    if args.reduced_artifact is not None:
        config.resume.reduced_artifact = args.reduced_artifact
        config.stages.feature_extraction = False  # Auto-disable stages
        config.stages.dimensionality_reduction = False

    # Logging
    if args.log_level is not None:
        config.logging.log_level = args.log_level


def _print_completion_summary(result) -> None:
    """Print pipeline completion summary with training.yaml integration paths.

    Args:
        result: PipelineResult object.
    """
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETION SUMMARY")
    print("=" * 80)
    print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Run ID: {result.run_id}")
    print(f"Run Directory: {result.run_dir}")

    if result.features_dir:
        print(f"Features Directory: {result.features_dir}")
    if result.reduced_dir:
        print(f"Reduced Features Directory: {result.reduced_dir}")
    if result.ghsom_dir:
        print(f"GHSOM Directory: {result.ghsom_dir}")

    # Print validation summary
    print("\nValidation Summary:")
    for stage, report in result.validation_reports.items():
        status = "PASS" if report.is_valid else "FAIL"
        warning_count = len(report.warnings)
        error_count = len(report.errors)
        print(f"  {stage}: {status} ({warning_count} warnings, {error_count} errors)")

    # Print stage metrics
    print("\nStage Metrics:")
    for stage, metrics in result.stage_metrics.items():
        print(f"  {stage}:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")

    if result.success and result.ghsom_model_path:
        print("\n" + "=" * 80)
        print("ADD THESE PATHS TO configs/training.yaml:")
        print("=" * 80)
        print("ghsom:")
        print(f'  default_model_path: "{result.ghsom_model_path}"')
        print("features:")
        print(f'  artifact_path: "{result.reduced_artifact_path}"')
        print("=" * 80)

    if not result.success:
        print(f"\nError: {result.error_message}")


def main() -> int:
    """Main entry point for the preprocessing pipeline CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = _parse_args()

    # Configure basic logging (detailed logging happens in pipeline)
    logging.basicConfig(
        level=args.log_level or "INFO",
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load configuration from YAML
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        config = PreprocessingPipelineConfig.from_yaml(args.config)
    except Exception as exc:
        print(f"ERROR: Failed to load config: {exc}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    _apply_cli_overrides(config, args)

    # Create and execute pipeline
    pipeline = PreprocessingPipeline(config)

    try:
        result = pipeline.execute()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nPipeline failed with error: {exc}", file=sys.stderr)
        logging.exception("Pipeline execution failed")
        return 1

    # Print completion summary
    _print_completion_summary(result)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
