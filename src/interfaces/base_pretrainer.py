"""Abstract base class definitions for LSTM pretraining workflows.

This module provides the interface for pretraining components that prepare
feature extractors using LSTM-based models. It defines a generic contract
that can be implemented by different backend frameworks (PyTorch, TensorFlow, etc.)
to ensure interoperability. The pretraining process involves loading data,
training models, and exporting artifacts for use in the main reinforcement
learning pipeline.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

DatasetT = TypeVar("DatasetT")
VocabT = TypeVar("VocabT")
TrainingResultT = TypeVar("TrainingResultT")
ArtifactsT = TypeVar("ArtifactsT")

DatasetMetadata = Mapping[str, Sequence[Any]]
VocabMetadata = Mapping[Any, Any]


class BasePretrainer(
    Generic[DatasetT, VocabT, TrainingResultT, ArtifactsT], metaclass=abc.ABCMeta
):
    """Contract for LSTM-style pretraining flows.

    Implementations wrap concrete training backends (e.g. PyTorch, TensorFlow)
    and expose a unified API so higher-level orchestration code can remain
    backend-agnostic. All methods must be implemented by thin adapters around
    the existing training logic to avoid behavior changes.
    """

    @abc.abstractmethod
    def load_and_prepare(
        self,
        csv_path: Path,
        *,
        cluster_column: str,
        group_column: Optional[str],
    ) -> Tuple[DatasetT, VocabT, DatasetMetadata]:
        """Load a CSV file and prepare the training dataset.

        Args:
            csv_path: Path to a CSV file containing clustered token sequences.
            cluster_column: Name of the CSV column containing cluster labels.
            group_column: Optional column name that groups sequences by an
                identifier (e.g. track ID).

        Returns:
            A tuple containing the backend-specific dataset object, the
            vocabulary mapping, and auxiliary dataset metadata.
        """

    @abc.abstractmethod
    def train(
        self,
        dataset: DatasetT,
        vocab: VocabT,
    ) -> TrainingResultT:
        """Train the underlying model on the prepared dataset."""

    @abc.abstractmethod
    def export(
        self,
        result: TrainingResultT,
        output_dir: Path,
        run_id: str,
        *,
        overwrite: bool = False,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ) -> ArtifactsT:
        """Export training artifacts to disk using the unified schema."""

    @abc.abstractmethod
    def train_and_export(
        self,
        csv_path: Path,
        *,
        output_dir: Path,
        run_id: str,
        cluster_column: str,
        group_column: Optional[str],
        overwrite: bool = False,
    ) -> ArtifactsT:
        """End-to-end convenience wrapper that trains and exports artifacts."""

    @abc.abstractmethod
    def seed(self, random_state: Optional[int]) -> None:
        """Seed all relevant random number generators for reproducibility."""
