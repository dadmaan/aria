"""Preprocessing utilities for preparing musical data."""

from .base import Preprocessor, PreprocessorConfig, PreprocessorResult  # noqa: F401
from .data_validation import (  # noqa: F401
    print_validation_report,
    validate_dataset,
)
from .dimensionality_reduction import (  # noqa: F401
    TSNEPreprocessor,
    TSNEPreprocessorConfig,
)
from .feature_extraction import (  # noqa: F401
    FeatureExtractionConfig,
    FeatureExtractionReport,
    run_feature_extraction,
)
from .metadata_adapter import (  # noqa: F401
    AdaptationReport,
    MetadataAdapterConfig,
    adapt_metadata,
)
from .pipeline import (  # noqa: F401
    PipelineResult,
    PreprocessingPipeline,
)
from .pipeline_config import (  # noqa: F401
    DimensionalityReductionStageConfig,
    FeatureExtractionStageConfig,
    GHSOMStageConfig,
    LoggingConfig,
    PreprocessingPipelineConfig,
    ResumeConfig,
    StageConfig,
    ValidationConfig,
)
from .validation import (  # noqa: F401
    ValidationError,
    ValidationReport,
    ValidationWarning,
    validate_dimensionality_reduction_output,
    validate_feature_extraction_output,
    validate_ghsom_training_output,
)
