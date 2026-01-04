"""Shared dataclasses and type aliases for LSTM pretraining."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

ModelT = TypeVar("ModelT")
MetricsT = TypeVar("MetricsT", bound=Mapping[str, Any])


DatasetMetadata = Mapping[str, Union[int, float, Sequence[int]]]
VocabMapping = MutableMapping[int, int]


@dataclass
class TrainingResult(Generic[ModelT, MetricsT]):
    """Container for model training outputs."""

    model: ModelT
    vocab: VocabMapping
    config: Any
    metrics: MetricsT


@dataclass
class ArtifactFile:
    """Metadata for a single artifact emitted by pretraining."""

    name: str
    path: Path
    sha256: str


@dataclass
class Manifest:
    """Inventory of training artifacts with checksums."""

    backend: str
    run_id: str
    artifacts: Dict[str, ArtifactFile] = field(default_factory=dict)


@dataclass
class ArtifactBundle:
    """Resolved artifact paths for a training run."""

    run_dir: Path
    config_path: Path
    metrics_path: Path
    vocab_path: Optional[Path]
    model_path: Path
    manifest_path: Path
    dataset_metadata_path: Optional[Path] = None
