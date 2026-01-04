"""Preprocessing utilities for preparing musical data."""

from .base import Preprocessor, PreprocessorConfig, PreprocessorResult  # noqa: F401
from .feature_extraction import (  # noqa: F401
    FeatureExtractionConfig,
    FeatureExtractionReport,
    run_feature_extraction,
)
from .dimensionality_reduction import (  # noqa: F401
    TSNEPreprocessor,
    TSNEPreprocessorConfig,
)
from .data_validation import (  # noqa: F401
    validate_dataset,
    print_validation_report,
)
from .metadata_adapter import (  # noqa: F401
    adapt_metadata,
    MetadataAdapterConfig,
    AdaptationReport,
)
