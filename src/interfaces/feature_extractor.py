"""Abstract interface for loading pretrained feature extractors.

This module defines the contract for loading and instantiating feature extractors
from saved checkpoints. It provides a backend-agnostic way to restore trained
models that can extract features from musical sequences. The interface supports
different machine learning frameworks while maintaining a consistent API for
the rest of the system.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Generic, TypeVar

ExtractorT = TypeVar("ExtractorT")
LoaderT = TypeVar("LoaderT", bound="FeatureExtractorLoader[Any]")


class FeatureExtractorLoader(Generic[ExtractorT], metaclass=abc.ABCMeta):
    """Contract for constructing feature extractors from persisted checkpoints."""

    @classmethod
    @abc.abstractmethod
    def from_checkpoint(
        cls, checkpoint_path: Path
    ) -> "FeatureExtractorLoader[ExtractorT]":
        """Load a feature extractor using the contents of a checkpoint directory."""

    @abc.abstractmethod
    def to_extractor(self, **kwargs: Any) -> ExtractorT:
        """Materialize the concrete feature extractor instance.

        Adapters may require additional contextual keyword arguments (e.g.
        observation spaces) at instantiation time. The loader returned from
        :meth:`from_checkpoint` encapsulates the state needed to build the
        extractor while keeping the abstract signature backend-agnostic.
        """
