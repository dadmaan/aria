"""Abstract preprocessing interfaces used across feature pipelines.

This module defines shared configuration and orchestration utilities that keep
preprocessing jobs (feature extraction, dimensionality reduction, etc.)
consistent. Subclasses implement the domain-specific logic while inheriting
deterministic seeding, logging, and artifact bookkeeping.
"""

from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.utils.logging.logging_manager import get_logger


def _serialise_value(value: Any) -> Any:
    """Convert dataclass values into JSON-friendly structures."""

    if isinstance(value, Path):  # Convert Path to string
        return str(value)
    if isinstance(value, (list, tuple)):  # Recursively serialize sequences
        return [_serialise_value(item) for item in value]
    if isinstance(value, dict):  # Recursively serialize dicts
        return {str(key): _serialise_value(val) for key, val in value.items()}
    return value  # Return primitive types as-is


@dataclass
class PreprocessorConfig:
    """Base configuration shared by preprocessing jobs."""

    input_path: Optional[Path] = None
    output_root: Path = Path("artifacts")
    run_id: Optional[str] = None
    seed: int = 7
    overwrite: bool = False
    log_level: int = logging.INFO
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_serialisable_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""

        data = asdict(self)
        return {key: _serialise_value(value) for key, value in data.items()}


@dataclass
class PreprocessorResult:
    """Container describing the outcomes of a preprocessing run."""

    run_id: str
    run_dir: Path
    artifacts: Dict[str, Path]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Preprocessor(ABC):
    """Abstract template that coordinates preprocessing pipelines."""

    config: PreprocessorConfig
    run_id: str
    run_dir: Path
    logger: logging.Logger

    def __init__(
        self, config: PreprocessorConfig, *, logger: Optional[logging.Logger] = None
    ) -> None:
        self.config = config
        self.run_id = config.run_id or self._generate_run_id()
        self.run_dir = config.output_root / self.run_id
        self.logger = logger or self._create_logger(config.log_level)

    # Public API -----------------------------------------------------------------

    def execute(self) -> PreprocessorResult:
        """Run the preprocessing pipeline end-to-end."""

        self.logger.info("Starting run %s", self.run_id)
        self._prepare_run_directory()  # Create output directory
        self._initialise_seeds()  # Set random seeds for reproducibility
        inputs = self.load_inputs()  # Load input data
        processed = self.process(inputs)  # Process the data
        result = self.save(processed)  # Save outputs
        self._persist_run_metadata(result)  # Save metadata
        self.logger.info("Completed run %s", self.run_id)
        return result

    # Template methods -----------------------------------------------------------

    @abstractmethod
    def load_inputs(self) -> Any:
        """Load all resources required for processing."""

    @abstractmethod
    def process(self, inputs: Any) -> Any:  # Transform inputs into processed form
        """Execute the core transformation logic."""

    @abstractmethod
    def save(self, processed: Any) -> PreprocessorResult:
        """Persist the processed outputs and return a result summary."""

    # Helpers --------------------------------------------------------------------

    def _initialise_seeds(self) -> None:
        random.seed(self.config.seed)  # Set Python random seed
        np.random.seed(self.config.seed)  # Set NumPy random seed

    def _prepare_run_directory(self) -> None:
        if self.run_dir.exists():
            if not self.config.overwrite:
                raise FileExistsError(
                    f"Output directory {self.run_dir} already exists. Set overwrite=True to replace it."
                )
            for child in self.run_dir.iterdir():  # Clean existing directory
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    self._remove_tree(child)
        self.run_dir.mkdir(parents=True, exist_ok=True)  # Create directory
        self._write_config()  # Save config

    def _write_config(self) -> None:
        config_path = self.run_dir / "config.json"
        serialisable = self.config.to_serialisable_dict()
        serialisable.update({"run_id": self.run_id})
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(serialisable, fp, indent=2, sort_keys=True)

    def _persist_run_metadata(self, result: PreprocessorResult) -> None:
        metadata_path = self.run_dir / "run_metadata.json"
        payload = {
            "run_id": result.run_id,
            "artifacts": {name: str(path) for name, path in result.artifacts.items()},
            "metadata": _serialise_value(result.metadata),
            "timestamp": datetime.utcnow().isoformat(),
        }
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    def _create_logger(self, log_level: int) -> logging.Logger:
        # Use centralized logging manager
        logger = get_logger(f"preprocessor.{self.__class__.__name__}")
        return logger

    def _generate_run_id(self) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"preprocess_{timestamp}"

    def _remove_tree(self, path: Path) -> None:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()  # Remove files/links
            else:
                self._remove_tree(child)  # Recurse on dirs
        path.rmdir()  # Remove empty dir
