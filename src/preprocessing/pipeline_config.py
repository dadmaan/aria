"""Configuration dataclasses for the unified preprocessing pipeline.

This module defines the complete configuration hierarchy for the preprocessing
pipeline, which orchestrates feature extraction, dimensionality reduction, and
GHSOM training.

Classes:
    StageConfig: Controls which pipeline stages are enabled.
    ResumeConfig: Specifies existing artifacts to resume from.
    FeatureExtractionStageConfig: Feature extraction parameters.
    DimensionalityReductionStageConfig: Dimensionality reduction parameters.
    GHSOMStageConfig: GHSOM training parameters.
    ValidationConfig: Inter-stage validation thresholds.
    LoggingConfig: Pipeline logging configuration.
    PreprocessingPipelineConfig: Root configuration for the entire pipeline.

Example:
    >>> config = PreprocessingPipelineConfig.from_yaml("configs/preprocessing.yaml")
    >>> pipeline = PreprocessingPipeline(config)
    >>> result = pipeline.execute()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


@dataclass
class StageConfig:
    """Controls which pipeline stages are enabled.

    Attributes:
        feature_extraction: Enable MIDI feature extraction stage.
        dimensionality_reduction: Enable dimensionality reduction stage.
        ghsom_training: Enable GHSOM clustering stage.
    """

    feature_extraction: bool = True
    dimensionality_reduction: bool = True
    ghsom_training: bool = True


@dataclass
class ResumeConfig:
    """Specifies existing artifacts to resume from (skip earlier stages).

    Attributes:
        features_artifact: Path to existing feature extraction output. If set,
            skips feature extraction stage.
        reduced_artifact: Path to existing dimensionality reduction output. If set,
            skips both feature extraction and dimensionality reduction.
    """

    features_artifact: Optional[Path] = None
    reduced_artifact: Optional[Path] = None

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.features_artifact, str):
            self.features_artifact = Path(self.features_artifact)
        if isinstance(self.reduced_artifact, str):
            self.reduced_artifact = Path(self.reduced_artifact)


@dataclass
class FeatureExtractionStageConfig:
    """Feature extraction stage configuration.

    Mirrors FeatureExtractionConfig from feature_extractor.py but adapted for
    pipeline context.

    Attributes:
        dataset_root: Root directory containing MIDI files.
        metadata_csv: Optional CSV with file metadata (id, file_path, split, etc.).
        metadata_index_column: Column name for track IDs.
        metadata_path_column: Column name for MIDI file paths.
        metadata_split_column: Column name for train/valid/test split.
        include_splits: Splits to include (None = all splits).
        extensions: MIDI file extensions to process.
        max_files: Maximum number of files to process (None = all).
        save_per_file: Save individual JSON per MIDI file.
        features: Feature selection - "full" for all features, or list of
            specific feature names (e.g., ["pm_note_count", "muspy_pitch_entropy"]).
        num_workers: Number of worker processes for parallel extraction.
            1 = sequential, -1 = all available CPUs, N = use N workers.
    """

    dataset_root: Optional[Path] = None
    metadata_csv: Optional[Path] = None
    metadata_index_column: str = "id"
    metadata_path_column: str = "file_path"
    metadata_split_column: Optional[str] = "split"
    include_splits: Optional[List[str]] = None
    extensions: List[str] = field(default_factory=lambda: [".mid", ".midi"])
    max_files: Optional[int] = None
    save_per_file: bool = False
    features: Union[str, List[str]] = "full"
    num_workers: int = 4

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.dataset_root, str):
            self.dataset_root = Path(self.dataset_root)
        if isinstance(self.metadata_csv, str):
            self.metadata_csv = Path(self.metadata_csv)


@dataclass
class DimensionalityReductionStageConfig:
    """Dimensionality reduction stage configuration.

    Mirrors DimensionalityReductionConfig but adapted for pipeline context.

    Attributes:
        method: Reduction method name ("tsne", "pca", etc.).
        n_components: Target number of dimensions.
        standardise: Apply z-score standardization before reduction.
        metadata_columns: Column names to preserve in output.
        method_params: Method-specific parameters (e.g., perplexity for t-SNE).
    """

    method: str = "tsne"
    n_components: int = 2
    standardise: bool = True
    metadata_columns: List[str] = field(default_factory=lambda: ["id"])
    method_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GHSOMStageConfig:
    """GHSOM training stage configuration.

    Mirrors GHSOMTrainingConfig from ghsom/training.py.

    Attributes:
        t1: Quantization threshold for root map (controls initial map size).
        t2: Quantization threshold for child maps (controls hierarchy depth).
        learning_rate: SOM learning rate.
        decay: Learning rate decay factor per epoch.
        gaussian_sigma: Neighborhood function width.
        epochs: Number of training epochs.
        grow_maxiter: Maximum grow iterations per level.
        feature_type: Which features to use for GHSOM ("tsne" or "raw").
    """

    t1: float = 0.35
    t2: float = 0.05
    learning_rate: float = 0.1
    decay: float = 0.99
    gaussian_sigma: float = 1.0
    epochs: int = 30
    grow_maxiter: int = 25
    feature_type: str = "tsne"


@dataclass
class ValidationConfig:
    """Inter-stage validation thresholds.

    Attributes:
        min_samples: Minimum number of samples required after feature extraction.
        max_nan_ratio: Maximum allowed NaN ratio per column (0.0-1.0).
        min_feature_variance: Minimum variance required to keep a feature.
    """

    min_samples: int = 100
    max_nan_ratio: float = 0.1
    min_feature_variance: float = 1e-6

    def __post_init__(self) -> None:
        """Convert string values to proper types (YAML may parse scientific notation as strings)."""
        if isinstance(self.min_feature_variance, str):
            self.min_feature_variance = float(self.min_feature_variance)
        if isinstance(self.max_nan_ratio, str):
            self.max_nan_ratio = float(self.max_nan_ratio)


@dataclass
class LoggingConfig:
    """Pipeline logging configuration.

    Attributes:
        log_level: Logging level (CRITICAL, ERROR, WARNING, INFO, DEBUG).
        log_to_file: Write logs to pipeline.log in run directory.
    """

    log_level: str = "INFO"
    log_to_file: bool = True


@dataclass
class PreprocessingPipelineConfig:
    """Root configuration for the preprocessing pipeline.

    This dataclass defines the complete configuration hierarchy for the pipeline,
    spanning all three stages (feature extraction, dimensionality reduction, GHSOM)
    plus pipeline control settings.

    Attributes:
        run_id: Unique identifier for this run (auto-generated if None).
        seed: Random seed for reproducibility.
        output_root: Root directory for pipeline outputs.
        overwrite: Allow overwriting existing run_id directory.
        stop_on_error: Stop pipeline execution on stage failure.
        stages: Stage enable/disable controls.
        resume: Existing artifact paths for resuming.
        feature_extraction: Feature extraction stage config.
        dimensionality_reduction: Dimensionality reduction stage config.
        ghsom: GHSOM training stage config.
        validation: Inter-stage validation config.
        logging: Logging configuration.

    Example:
        >>> config = PreprocessingPipelineConfig.from_yaml("configs/preprocessing.yaml")
        >>> config.run_id = "my_experiment"
        >>> config.seed = 42
    """

    run_id: Optional[str] = None
    seed: int = 42
    output_root: Path = Path("artifacts/preprocessing")
    overwrite: bool = False
    stop_on_error: bool = True

    stages: StageConfig = field(default_factory=StageConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    feature_extraction: FeatureExtractionStageConfig = field(
        default_factory=FeatureExtractionStageConfig
    )
    dimensionality_reduction: DimensionalityReductionStageConfig = field(
        default_factory=DimensionalityReductionStageConfig
    )
    ghsom: GHSOMStageConfig = field(default_factory=GHSOMStageConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.output_root, str):
            self.output_root = Path(self.output_root)

    @classmethod
    def from_yaml(cls, path: Path) -> PreprocessingPipelineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded configuration object.

        Raises:
            FileNotFoundError: If config file does not exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Build nested configs from YAML sections
        pipeline_section = raw_config.get("pipeline", {})
        stages_section = pipeline_section.get("stages", {})
        resume_section = raw_config.get("resume", {})
        feature_extraction_section = raw_config.get("feature_extraction", {})
        dim_reduction_section = raw_config.get("dimensionality_reduction", {})
        ghsom_section = raw_config.get("ghsom", {})
        validation_section = raw_config.get("validation", {})
        logging_section = raw_config.get("logging", {})

        return cls(
            run_id=pipeline_section.get("run_id"),
            seed=pipeline_section.get("seed", 42),
            output_root=Path(
                pipeline_section.get("output_root", "artifacts/preprocessing")
            ),
            overwrite=pipeline_section.get("overwrite", False),
            stop_on_error=pipeline_section.get("stop_on_error", True),
            stages=StageConfig(**stages_section),
            resume=ResumeConfig(**resume_section),
            feature_extraction=FeatureExtractionStageConfig(
                **feature_extraction_section
            ),
            dimensionality_reduction=DimensionalityReductionStageConfig(
                **dim_reduction_section
            ),
            ghsom=GHSOMStageConfig(**ghsom_section),
            validation=ValidationConfig(**validation_section),
            logging=LoggingConfig(**logging_section),
        )

    def to_serialisable_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation with Paths converted to strings.
        """

        def _convert_value(value: Any) -> Any:
            """Recursively convert non-serializable types."""
            if isinstance(value, Path):
                return str(value)
            elif isinstance(value, dict):
                return {k: _convert_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [_convert_value(v) for v in value]
            elif hasattr(value, "__dict__"):
                # Dataclass instance
                return {k: _convert_value(v) for k, v in value.__dict__.items()}
            else:
                return value

        return _convert_value(self.__dict__)
