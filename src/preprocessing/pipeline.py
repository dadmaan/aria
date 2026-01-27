"""Unified preprocessing pipeline orchestrator.

This module provides the main PreprocessingPipeline class that orchestrates
the complete data preparation workflow: feature extraction, dimensionality
reduction, and GHSOM training.

Classes:
    PipelineResult: Complete pipeline execution result with all artifact paths.
    PreprocessingPipeline: Main orchestrator class.

Example:
    >>> config = PreprocessingPipelineConfig.from_yaml("configs/preprocessing.yaml")
    >>> pipeline = PreprocessingPipeline(config)
    >>> result = pipeline.execute()
    >>> print(f"GHSOM model: {result.ghsom_model_path}")
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from src.preprocessing.dimensionality_reduction.config import (
    DimensionalityReductionConfig,
)
from src.preprocessing.dimensionality_reduction.preprocessor import (
    DimensionalityReductionPreprocessor,
)
from src.preprocessing.feature_extraction.feature_extractor import (
    FeatureExtractionConfig,
    FeatureExtractionReport,
    run_feature_extraction,
)
from src.preprocessing.pipeline_config import PreprocessingPipelineConfig
from src.preprocessing.validation import (
    ValidationReport,
    validate_dimensionality_reduction_output,
    validate_feature_extraction_output,
    validate_ghsom_training_output,
)
from src.ghsom.training import GHSOMTrainingConfig, train_and_export
from src.utils.features.feature_loader import load_feature_dataset
from src.utils.logging.logging_manager import get_logger


@dataclass
class PipelineResult:
    """Complete pipeline execution result with all artifact paths.

    Attributes:
        success: Whether the pipeline completed successfully.
        run_id: Unique identifier for this pipeline run.
        run_dir: Root directory for pipeline outputs.
        features_dir: Feature extraction output directory.
        reduced_dir: Dimensionality reduction output directory.
        ghsom_dir: GHSOM training output directory.
        ghsom_model_path: Path to trained GHSOM model (ghsom_model.pkl).
        reduced_artifact_path: Path to reduced features directory (for training.yaml).
        manifest_path: Path to pipeline manifest JSON.
        validation_reports: Per-stage validation reports.
        stage_metrics: Collected metrics from each stage.
        error_message: Error description if pipeline failed.
    """

    success: bool
    run_id: str
    run_dir: Path
    features_dir: Optional[Path] = None
    reduced_dir: Optional[Path] = None
    ghsom_dir: Optional[Path] = None
    ghsom_model_path: Optional[Path] = None
    reduced_artifact_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    validation_reports: Dict[str, ValidationReport] = field(default_factory=dict)
    stage_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "success": self.success,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "features_dir": str(self.features_dir) if self.features_dir else None,
            "reduced_dir": str(self.reduced_dir) if self.reduced_dir else None,
            "ghsom_dir": str(self.ghsom_dir) if self.ghsom_dir else None,
            "ghsom_model_path": str(self.ghsom_model_path)
            if self.ghsom_model_path
            else None,
            "reduced_artifact_path": (
                str(self.reduced_artifact_path) if self.reduced_artifact_path else None
            ),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "validation_reports": {
                stage: report.to_dict()
                for stage, report in self.validation_reports.items()
            },
            "stage_metrics": self.stage_metrics,
            "error_message": self.error_message,
        }


class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline.

    This class coordinates three sequential stages:
    1. Feature extraction from raw MIDI files
    2. Dimensionality reduction (t-SNE/PCA)
    3. GHSOM clustering training

    Each stage can be skipped/resumed by providing existing artifact paths.
    Inter-stage validation checks ensure data quality before proceeding.

    Attributes:
        config: Complete pipeline configuration.
        logger: Logging manager instance.
        run_dir: Root directory for this pipeline run.
        artifacts: Dictionary of artifact paths (populated during execution).

    Example:
        >>> config = PreprocessingPipelineConfig.from_yaml("configs/preprocessing.yaml")
        >>> pipeline = PreprocessingPipeline(config)
        >>> result = pipeline.execute()
        >>> if result.success:
        ...     print(f"Pipeline complete: {result.manifest_path}")
    """

    def __init__(self, config: PreprocessingPipelineConfig):
        """Initialize the pipeline with configuration.

        Args:
            config: Complete pipeline configuration.
        """
        self.config = config
        self.logger = get_logger("preprocessing.pipeline")
        self.run_dir: Optional[Path] = None
        self.artifacts: Dict[str, Path] = {}

    def execute(self) -> PipelineResult:
        """Run the complete pipeline end-to-end.

        Returns:
            PipelineResult with execution status and artifact paths.

        Raises:
            Exception: If stop_on_error=True and any stage fails.
        """
        self.logger.info("=" * 80)
        self.logger.info("ARIA PREPROCESSING PIPELINE")
        self.logger.info("=" * 80)

        # Initialize result container
        result = PipelineResult(
            success=False,
            run_id=self.config.run_id or self._generate_run_id(),
            run_dir=Path(""),  # Will be set in _prepare_output_directory
        )

        try:
            # Prepare output directory and save config
            self._prepare_output_directory(result.run_id)
            result.run_dir = self.run_dir
            self._save_config()

            # Initialize random seeds for reproducibility
            self._initialize_seeds()

            # Stage 1: Feature Extraction
            if self.config.resume.reduced_artifact:
                # Skip both feature extraction and dimensionality reduction
                self.logger.info("Skipping feature extraction (using reduced_artifact)")
                reduced_artifact_path = self.config.resume.reduced_artifact
                features_result = None
            elif self.config.resume.features_artifact:
                # Skip feature extraction only
                self.logger.info(
                    "Skipping feature extraction (using features_artifact)"
                )
                features_artifact_path = self.config.resume.features_artifact
                features_result = None
            elif self.config.stages.feature_extraction:
                self.logger.info("STAGE 1: Feature Extraction")
                features_result = self._run_feature_extraction()
                self._validate_features(features_result, result)
                features_artifact_path = features_result.run_dir
                result.features_dir = features_result.run_dir
            else:
                raise ValueError(
                    "Feature extraction stage is disabled but no resume.features_artifact provided"
                )

            # Stage 2: Dimensionality Reduction
            if self.config.resume.reduced_artifact:
                # Use provided reduced artifact
                self.logger.info(
                    "Skipping dimensionality reduction (using reduced_artifact)"
                )
                reduced_artifact_path = self.config.resume.reduced_artifact
            elif self.config.stages.dimensionality_reduction:
                self.logger.info("STAGE 2: Dimensionality Reduction")
                reduced_result = self._run_dimensionality_reduction(
                    features_artifact_path
                )
                self._validate_reduced_features(reduced_result, result)
                reduced_artifact_path = reduced_result.run_dir
                result.reduced_dir = reduced_result.run_dir
            else:
                raise ValueError(
                    "Dimensionality reduction stage is disabled but no resume.reduced_artifact provided"
                )

            # Stage 3: GHSOM Training
            if self.config.stages.ghsom_training:
                self.logger.info("STAGE 3: GHSOM Training")
                ghsom_result = self._run_ghsom_training(reduced_artifact_path)
                self._validate_ghsom_output(ghsom_result, result)
                result.ghsom_dir = ghsom_result["run_dir"]
                result.ghsom_model_path = ghsom_result["ghsom_model"]
            else:
                self.logger.info("Skipping GHSOM training (stage disabled)")
                result.ghsom_dir = None
                result.ghsom_model_path = None

            # Set final artifact paths
            result.reduced_artifact_path = reduced_artifact_path
            result.success = True

            # Generate final manifest
            result.manifest_path = self._save_manifest(result)

            self.logger.info("=" * 80)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("=" * 80)

        except Exception as exc:
            result.success = False
            result.error_message = str(exc)
            self.logger.exception("Pipeline failed with error: %s", exc)
            if self.config.stop_on_error:
                raise

        return result

    def _generate_run_id(self) -> str:
        """Generate a timestamp-based run ID.

        Returns:
            Run ID string in format: YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _prepare_output_directory(self, run_id: str) -> None:
        """Create and configure the output directory structure.

        Args:
            run_id: Unique identifier for this run.

        Raises:
            FileExistsError: If run directory exists and overwrite=False.
        """
        self.run_dir = self.config.output_root / run_id

        if self.run_dir.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"Run directory {self.run_dir} already exists. "
                    "Pass overwrite=True to replace it."
                )
            self.logger.warning(f"Overwriting existing run directory: {self.run_dir}")
            shutil.rmtree(self.run_dir)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created run directory: {self.run_dir}")

        # Create subdirectories for each stage
        (self.run_dir / "features").mkdir(exist_ok=True)
        (self.run_dir / "reduced").mkdir(exist_ok=True)
        (self.run_dir / "ghsom").mkdir(exist_ok=True)

    def _save_config(self) -> None:
        """Save the full pipeline configuration to YAML."""
        config_path = self.run_dir / "pipeline_config.yaml"
        config_dict = self.config.to_serialisable_dict()

        with open(config_path, "w") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, sort_keys=False, indent=2
            )

        self.logger.info(f"Saved pipeline config to {config_path}")

    def _initialize_seeds(self) -> None:
        """Initialize random seeds for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.logger.info(f"Initialized random seeds: {self.config.seed}")

    # ========================================================================
    # STAGE 1: FEATURE EXTRACTION
    # ========================================================================

    def _run_feature_extraction(self) -> FeatureExtractionReport:
        """Run the feature extraction stage.

        Returns:
            FeatureExtractionReport with paths to extracted features.
        """
        self.logger.info("-" * 80)
        self.logger.info("Running feature extraction...")
        self.logger.info("-" * 80)

        # Build feature extraction config
        fe_config = FeatureExtractionConfig(
            dataset_root=self.config.feature_extraction.dataset_root,
            metadata_csv=self.config.feature_extraction.metadata_csv,
            output_root=self.run_dir / "features",
            metadata_index_column=self.config.feature_extraction.metadata_index_column,
            metadata_path_column=self.config.feature_extraction.metadata_path_column,
            metadata_split_column=self.config.feature_extraction.metadata_split_column,
            include_splits=self.config.feature_extraction.include_splits,
            extensions=self.config.feature_extraction.extensions,
            run_id="extracted",  # Fixed subdirectory name
            seed=self.config.seed,
            overwrite=self.config.overwrite,
            max_files=self.config.feature_extraction.max_files,
            save_per_file=self.config.feature_extraction.save_per_file,
            features=self.config.feature_extraction.features,
            num_workers=self.config.feature_extraction.num_workers,
        )

        # Run extraction
        report = run_feature_extraction(
            fe_config, log_level=self.config.logging.log_level
        )

        self.logger.info(
            f"Feature extraction complete: {report.success_count} successful, "
            f"{report.failure_count} failed"
        )

        if report.failure_count > 0:
            self.logger.warning(f"{report.failure_count} files failed to process")

        return report

    def _validate_features(
        self, report: FeatureExtractionReport, result: PipelineResult
    ) -> None:
        """Validate feature extraction output.

        Args:
            report: Feature extraction report.
            result: Pipeline result container (updated in-place).

        Raises:
            ValueError: If validation fails and stop_on_error=True.
        """
        self.logger.info("Validating feature extraction output...")

        validation = validate_feature_extraction_output(
            features_csv=report.feature_matrix_path,
            min_samples=self.config.validation.min_samples,
            max_nan_ratio=self.config.validation.max_nan_ratio,
            min_feature_variance=self.config.validation.min_feature_variance,
        )

        # Save validation report
        validation_path = report.run_dir / "validation_report.json"
        validation.save(validation_path)

        result.validation_reports["feature_extraction"] = validation
        result.stage_metrics["feature_extraction"] = validation.metrics

        # Log warnings
        for warning in validation.warnings:
            self.logger.warning(f"Feature validation warning: {warning.message}")

        # Handle errors
        if not validation.is_valid:
            for error in validation.errors:
                self.logger.error(f"Feature validation error: {error.message}")

            if self.config.stop_on_error:
                raise ValueError(
                    f"Feature extraction validation failed with {len(validation.errors)} errors"
                )

    # ========================================================================
    # STAGE 2: DIMENSIONALITY REDUCTION
    # ========================================================================

    def _run_dimensionality_reduction(self, features_path: Path) -> Any:
        """Run the dimensionality reduction stage.

        Args:
            features_path: Path to feature extraction output directory.

        Returns:
            PreprocessorResult with paths to reduced features.
        """
        self.logger.info("-" * 80)
        self.logger.info("Running dimensionality reduction...")
        self.logger.info("-" * 80)

        # Locate the features CSV
        features_csv = features_path / "features_numeric.csv"
        if not features_csv.exists():
            # Try alternative locations
            features_csv = features_path / "features_with_metadata.csv"
            if not features_csv.exists():
                raise FileNotFoundError(
                    f"Could not find features CSV in {features_path}"
                )

        # Build dimensionality reduction config
        dr_config = DimensionalityReductionConfig(
            input_features=features_csv,
            output_root=self.run_dir / "reduced",
            run_id="reduced",  # Fixed subdirectory name
            method=self.config.dimensionality_reduction.method,
            n_components=self.config.dimensionality_reduction.n_components,
            random_state=self.config.seed,
            standardise=self.config.dimensionality_reduction.standardise,
            metadata_columns=self.config.dimensionality_reduction.metadata_columns,
            method_params=self.config.dimensionality_reduction.method_params,
            seed=self.config.seed,
            overwrite=self.config.overwrite,
        )

        # Run dimensionality reduction
        preprocessor = DimensionalityReductionPreprocessor(dr_config)
        result = preprocessor.execute()

        self.logger.info(f"Dimensionality reduction complete: {result.run_dir}")

        return result

    def _validate_reduced_features(
        self, dr_result: Any, result: PipelineResult
    ) -> None:
        """Validate dimensionality reduction output.

        Args:
            dr_result: Dimensionality reduction PreprocessorResult.
            result: Pipeline result container (updated in-place).

        Raises:
            ValueError: If validation fails and stop_on_error=True.
        """
        self.logger.info("Validating dimensionality reduction output...")

        embedding_csv = dr_result.artifacts.get("embedding_csv")
        embedding_npy = dr_result.artifacts.get("embedding_npy")

        if not embedding_csv:
            raise ValueError(
                "Dimensionality reduction did not produce embedding_csv artifact"
            )

        validation = validate_dimensionality_reduction_output(
            embedding_csv=Path(embedding_csv),
            embedding_npy=Path(embedding_npy) if embedding_npy else None,
            expected_n_components=self.config.dimensionality_reduction.n_components,
            min_samples=self.config.validation.min_samples,
        )

        # Save validation report
        validation_path = dr_result.run_dir / "validation_report.json"
        validation.save(validation_path)

        result.validation_reports["dimensionality_reduction"] = validation
        result.stage_metrics["dimensionality_reduction"] = validation.metrics

        # Log warnings
        for warning in validation.warnings:
            self.logger.warning(
                f"Dimensionality reduction validation warning: {warning.message}"
            )

        # Handle errors
        if not validation.is_valid:
            for error in validation.errors:
                self.logger.error(
                    f"Dimensionality reduction validation error: {error.message}"
                )

            if self.config.stop_on_error:
                raise ValueError(
                    f"Dimensionality reduction validation failed with {len(validation.errors)} errors"
                )

    # ========================================================================
    # STAGE 3: GHSOM TRAINING
    # ========================================================================

    def _run_ghsom_training(self, reduced_features_path: Path) -> Dict[str, Path]:
        """Run the GHSOM training stage.

        Args:
            reduced_features_path: Path to dimensionality reduction output directory.

        Returns:
            Dictionary with paths to GHSOM artifacts.
        """
        self.logger.info("-" * 80)
        self.logger.info("Running GHSOM training...")
        self.logger.info("-" * 80)

        # Build GHSOM training config
        ghsom_config = GHSOMTrainingConfig(
            t1=self.config.ghsom.t1,
            t2=self.config.ghsom.t2,
            learning_rate=self.config.ghsom.learning_rate,
            decay=self.config.ghsom.decay,
            gaussian_sigma=self.config.ghsom.gaussian_sigma,
            epochs=self.config.ghsom.epochs,
            grow_maxiter=self.config.ghsom.grow_maxiter,
            seed=self.config.seed,
        )

        # Run GHSOM training and export
        result = train_and_export(
            feature_path=reduced_features_path,
            feature_type=self.config.ghsom.feature_type,
            metadata_columns=self.config.dimensionality_reduction.metadata_columns,
            config=ghsom_config,
            output_dir=self.run_dir / "ghsom",
            run_id="trained",  # Fixed subdirectory name
            n_workers=-1,  # Use all available cores
            overwrite=self.config.overwrite,
        )

        metrics = result["metrics"]
        artifact_paths = result["artifacts"]

        self.logger.info(
            f"GHSOM training complete: {metrics.get('num_clusters', 0)} clusters, "
            f"{metrics.get('ghsom_depth', 0)} depth"
        )

        return artifact_paths

    def _validate_ghsom_output(
        self, ghsom_result: Dict[str, Path], result: PipelineResult
    ) -> None:
        """Validate GHSOM training output.

        Args:
            ghsom_result: Dictionary with GHSOM artifact paths.
            result: Pipeline result container (updated in-place).

        Raises:
            ValueError: If validation fails and stop_on_error=True.
        """
        self.logger.info("Validating GHSOM training output...")

        validation = validate_ghsom_training_output(
            ghsom_model_path=ghsom_result["ghsom_model"],
            neuron_table_path=ghsom_result["neuron_table"],
            sample_to_cluster_path=ghsom_result["sample_to_cluster"],
            min_clusters=2,
        )

        # Save validation report
        validation_path = ghsom_result["run_dir"] / "validation_report.json"
        validation.save(validation_path)

        result.validation_reports["ghsom_training"] = validation
        result.stage_metrics["ghsom_training"] = validation.metrics

        # Log warnings
        for warning in validation.warnings:
            self.logger.warning(f"GHSOM validation warning: {warning.message}")

        # Handle errors
        if not validation.is_valid:
            for error in validation.errors:
                self.logger.error(f"GHSOM validation error: {error.message}")

            if self.config.stop_on_error:
                raise ValueError(
                    f"GHSOM training validation failed with {len(validation.errors)} errors"
                )

    # ========================================================================
    # FINALIZATION
    # ========================================================================

    def _save_manifest(self, result: PipelineResult) -> Path:
        """Generate and save the final pipeline manifest.

        Args:
            result: Complete pipeline result.

        Returns:
            Path to saved manifest file.
        """
        manifest_path = self.run_dir / "pipeline_manifest.json"

        manifest = result.to_dict()

        # Add timestamp
        manifest["completed_at"] = datetime.now().isoformat()

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Saved pipeline manifest to {manifest_path}")

        return manifest_path
