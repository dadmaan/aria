"""
Benchmark Discovery System for Training Analysis

This module provides a flexible architecture for discovering and analyzing training runs
across multiple directory structure patterns. It handles various organizational schemes
used in different phases of the project.

Supported Patterns:
    - FLAT_BENCHMARK: Direct run directories under phase folders (phase1-5)
    - NESTED_BENCHMARK: Run directories nested under benchmark folders (phase6-7)
    - METHOD_SEED_HIERARCHY: Organized by method/seed/run structure
    - METHOD_GROUPED: Run directories grouped by method/condition
    - NESTED_TIMESTAMP: Timestamp/method/seed/run structure for curriculum learning
    - ORIGINAL_TRAINING: Simple training artifacts structure
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


class StructurePattern(Enum):
    """Enumeration of supported directory structure patterns."""
    FLAT_BENCHMARK = "flat_benchmark"
    NESTED_BENCHMARK = "nested_benchmark"
    METHOD_SEED_HIERARCHY = "method_seed_hierarchy"
    METHOD_GROUPED = "method_grouped"
    NESTED_TIMESTAMP = "nested_timestamp"
    ORIGINAL_TRAINING = "original_training"


@dataclass
class RunInfo:
    """
    Stores comprehensive information about a discovered training run.

    Attributes:
        path: Absolute path to the run directory
        name: Name of the run directory
        phase: Training phase identifier (e.g., "phase1", "phase6")
        method: Method/algorithm name (e.g., "drqn", "lstm")
        seed: Random seed used for the run
        variant: Specific variant or configuration name
        config_path: Path to config.yaml if it exists
        metrics_path: Path to training_metrics.json if it exists
        tensorboard_path: Path to tensorboard directory if it exists
        metadata: Flexible dictionary for additional extracted metadata
        pattern: The structure pattern this run was discovered under
    """
    path: Path
    name: str
    phase: Optional[str] = None
    method: Optional[str] = None
    seed: Optional[int] = None
    variant: Optional[str] = None
    config_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    tensorboard_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    pattern: Optional[StructurePattern] = None

    def __post_init__(self):
        """Ensure path is a Path object."""
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

    def is_valid(self) -> bool:
        """
        Check if this run has valid data files.

        Returns:
            True if at least metrics or tensorboard data exists
        """
        return (
            (self.metrics_path and self.metrics_path.exists()) or
            (self.tensorboard_path and self.tensorboard_path.exists())
        )

    def has_config(self) -> bool:
        """Check if config file exists."""
        return self.config_path is not None and self.config_path.exists()

    def has_metrics(self) -> bool:
        """Check if metrics file exists."""
        return self.metrics_path is not None and self.metrics_path.exists()

    def has_tensorboard(self) -> bool:
        """Check if tensorboard directory exists and has events."""
        if not self.tensorboard_path or not self.tensorboard_path.exists():
            return False
        # Check for tensorboard event files
        return any(self.tensorboard_path.glob("events.out.tfevents.*"))

    def __repr__(self) -> str:
        """Readable representation of the run."""
        parts = [f"RunInfo(name={self.name}"]
        if self.phase:
            parts.append(f"phase={self.phase}")
        if self.method:
            parts.append(f"method={self.method}")
        if self.seed:
            parts.append(f"seed={self.seed}")
        parts.append(f"pattern={self.pattern.value if self.pattern else 'unknown'})")
        return ", ".join(parts)


class MetadataExtractor:
    """
    Extracts structured metadata from directory and file names.

    This class handles parsing of various naming conventions used across
    different benchmark phases and organizational structures.
    """

    # Pattern for benchmark directory names: bench_p1_drqn_h128_s42_251205_1547
    BENCHMARK_PATTERN = re.compile(
        r"bench_p(?P<phase>\d+)_(?P<network>\w+)_(?P<params>.*?)_s(?P<seed>\d+)_(?P<timestamp>\d+_\d+)"
    )

    # Pattern for observation benchmark: bench_p6_obs_centraw_s42_251206_1333
    OBS_PATTERN = re.compile(
        r"bench_p(?P<phase>\d+)_obs_(?P<obs_type>\w+)_s(?P<seed>\d+)_(?P<timestamp>\d+_\d+)"
    )

    # Pattern for run directory names: run_drqn_20251207_195352
    RUN_PATTERN = re.compile(
        r"run_(?P<method>\w+)_(?P<timestamp>\d{8}_\d{6})"
    )

    # Pattern for seed directories: seed_123
    SEED_PATTERN = re.compile(r"seed_(?P<seed>\d+)")

    # Pattern for extracting key-value pairs from params: h128_lr0.001
    PARAM_PATTERN = re.compile(r"(?P<key>[a-z]+)(?P<value>[\d.]+)")

    @staticmethod
    def extract_benchmark_metadata(dirname: str) -> Dict[str, Any]:
        """
        Extract metadata from benchmark directory names.

        Examples:
            bench_p1_drqn_h128_s42_251205_1547 -> {phase: 1, network: drqn, ...}
            bench_p6_obs_centraw_s42_251206_1333 -> {phase: 6, obs_type: centraw, ...}

        Args:
            dirname: Directory name to parse

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}

        # Try observation pattern first (more specific)
        match = MetadataExtractor.OBS_PATTERN.match(dirname)
        if match:
            metadata["phase"] = int(match.group("phase"))
            metadata["obs_type"] = match.group("obs_type")
            metadata["seed"] = int(match.group("seed"))
            metadata["timestamp"] = match.group("timestamp")
            metadata["variant"] = f"obs_{match.group('obs_type')}"
            return metadata

        # Try general benchmark pattern
        match = MetadataExtractor.BENCHMARK_PATTERN.match(dirname)
        if match:
            metadata["phase"] = int(match.group("phase"))
            metadata["network"] = match.group("network")
            metadata["seed"] = int(match.group("seed"))
            metadata["timestamp"] = match.group("timestamp")

            # Parse parameter string (e.g., "h128" -> hidden: 128)
            params_str = match.group("params")
            params = MetadataExtractor._parse_params(params_str)
            metadata.update(params)

            # Create variant name from network and params
            variant_parts = [match.group("network")]
            if params:
                variant_parts.append("_".join(f"{k}{v}" for k, v in params.items()))
            metadata["variant"] = "_".join(variant_parts)

        return metadata

    @staticmethod
    def extract_run_metadata(dirname: str) -> Dict[str, Any]:
        """
        Extract metadata from run directory names.

        Example:
            run_drqn_20251207_195352 -> {method: drqn, timestamp: ...}

        Args:
            dirname: Directory name to parse

        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        match = MetadataExtractor.RUN_PATTERN.match(dirname)
        if match:
            metadata["method"] = match.group("method")
            metadata["timestamp"] = match.group("timestamp")
        return metadata

    @staticmethod
    def extract_seed_from_dirname(dirname: str) -> Optional[int]:
        """
        Extract seed from directory name.

        Args:
            dirname: Directory name to parse

        Returns:
            Seed value if found, None otherwise
        """
        match = MetadataExtractor.SEED_PATTERN.match(dirname)
        if match:
            return int(match.group("seed"))
        return None

    @staticmethod
    def _parse_params(params_str: str) -> Dict[str, Any]:
        """
        Parse parameter string into key-value pairs.

        Example:
            "h128_lr0.001" -> {hidden: 128, lr: 0.001}

        Args:
            params_str: Parameter string to parse

        Returns:
            Dictionary of parameters
        """
        params = {}
        parts = params_str.split("_")

        for part in parts:
            match = MetadataExtractor.PARAM_PATTERN.match(part)
            if match:
                key = match.group("key")
                value = match.group("value")

                # Expand common abbreviations
                key_mapping = {
                    "h": "hidden",
                    "lr": "learning_rate",
                    "bs": "batch_size",
                    "ep": "epochs",
                }
                key = key_mapping.get(key, key)

                # Convert to appropriate type
                try:
                    if "." in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value

        return params

    @staticmethod
    def extract_phase_from_path(path: Path) -> Optional[str]:
        """
        Extract phase identifier from path components.

        Args:
            path: Path to analyze

        Returns:
            Phase identifier if found (e.g., "phase1", "phase6")
        """
        for part in path.parts:
            if part.startswith("phase") and part[5:].isdigit():
                return part
            # Also check for phase in benchmark names
            if "bench_p" in part:
                match = re.search(r"bench_p(\d+)", part)
                if match:
                    return f"phase{match.group(1)}"
        return None


class BenchmarkDiscovery:
    """
    Discovers and catalogs training runs across different directory structures.

    This class provides automatic detection of directory patterns and extraction
    of run information, making it easy to analyze training data regardless of
    how it's organized.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the discovery system.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.extractor = MetadataExtractor()

        if verbose:
            logger.setLevel(logging.DEBUG)

    def detect_structure(self, input_dir: Path) -> StructurePattern:
        """
        Automatically detect the directory structure pattern.

        Args:
            input_dir: Directory to analyze

        Returns:
            Detected structure pattern

        Raises:
            ValueError: If structure cannot be determined
        """
        input_dir = Path(input_dir)

        if not input_dir.exists():
            raise ValueError(f"Directory does not exist: {input_dir}")

        # Check for METHOD_SEED_HIERARCHY pattern (baseline/seed_X/run_*)
        baseline_dirs = list(input_dir.glob("baseline"))
        if baseline_dirs:
            baseline = baseline_dirs[0]
            seed_dirs = list(baseline.glob("seed_*"))
            if seed_dirs:
                run_dirs = list(seed_dirs[0].glob("run_*"))
                if run_dirs:
                    logger.info(f"Detected METHOD_SEED_HIERARCHY pattern in {input_dir}")
                    return StructurePattern.METHOD_SEED_HIERARCHY

        # Check for NESTED_BENCHMARK pattern (bench_*/run_*)
        bench_dirs = list(input_dir.glob("bench_p*"))
        if bench_dirs:
            for bench_dir in bench_dirs:
                run_dirs = list(bench_dir.glob("run_*"))
                if run_dirs:
                    logger.info(f"Detected NESTED_BENCHMARK pattern in {input_dir}")
                    return StructurePattern.NESTED_BENCHMARK

        # Check for FLAT_BENCHMARK pattern (phase*/bench_*)
        phase_dirs = list(input_dir.glob("phase*"))
        if phase_dirs:
            for phase_dir in phase_dirs:
                bench_dirs = list(phase_dir.glob("bench_p*"))
                if bench_dirs:
                    logger.info(f"Detected FLAT_BENCHMARK pattern in {input_dir}")
                    return StructurePattern.FLAT_BENCHMARK

        # Check for NESTED_TIMESTAMP pattern (YYYYMMDD_HHMMSS/method/seed_*/run_*)
        # Look for timestamp-like directories containing method/seed/run structure
        timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")
        timestamp_dirs = [d for d in input_dir.iterdir()
                         if d.is_dir() and timestamp_pattern.match(d.name)]

        if timestamp_dirs:
            # Check if any timestamp dir contains method/seed/run structure
            for ts_dir in timestamp_dirs:
                method_dirs = [d for d in ts_dir.iterdir() if d.is_dir()]
                for method_dir in method_dirs:
                    seed_dirs = list(method_dir.glob("seed_*"))
                    if seed_dirs:
                        run_dirs = list(seed_dirs[0].glob("run_*"))
                        if run_dirs:
                            logger.info(f"Detected NESTED_TIMESTAMP pattern in {input_dir}")
                            return StructurePattern.NESTED_TIMESTAMP

        # Check for METHOD_GROUPED pattern (method_dir/run_*)
        # Get immediate subdirectories that are not phase*, bench*, or seed_*
        subdirs = [d for d in input_dir.iterdir() if d.is_dir() and
                   not d.name.startswith(("phase", "bench", "seed_"))]

        if subdirs:
            # Check if any of these subdirectories contain run_* directories
            for subdir in subdirs:
                run_dirs = list(subdir.glob("run_*"))
                if run_dirs:
                    logger.info(f"Detected METHOD_GROUPED pattern in {input_dir}")
                    return StructurePattern.METHOD_GROUPED

        # Check for ORIGINAL_TRAINING pattern (run_*)
        run_dirs = list(input_dir.glob("run_*"))
        if run_dirs:
            # Make sure it's not nested (which would be NESTED_BENCHMARK)
            has_nested = any((input_dir / "bench_p*").exists() for _ in range(1))
            if not has_nested:
                logger.info(f"Detected ORIGINAL_TRAINING pattern in {input_dir}")
                return StructurePattern.ORIGINAL_TRAINING

        raise ValueError(f"Could not detect structure pattern for {input_dir}")

    def discover_runs(
        self,
        input_dir: Path,
        recursive: bool = True,
        pattern: Optional[StructurePattern] = None
    ) -> List[RunInfo]:
        """
        Discover all training runs in the given directory.

        Args:
            input_dir: Root directory to search
            recursive: Whether to search recursively
            pattern: Specific pattern to use (auto-detect if None)

        Returns:
            List of discovered RunInfo objects
        """
        input_dir = Path(input_dir)

        if pattern is None:
            pattern = self.detect_structure(input_dir)

        logger.info(f"Discovering runs in {input_dir} using {pattern.value} pattern")

        # Dispatch to pattern-specific method
        if pattern == StructurePattern.FLAT_BENCHMARK:
            runs = self._discover_flat_benchmark(input_dir, recursive)
        elif pattern == StructurePattern.NESTED_BENCHMARK:
            runs = self._discover_nested_benchmark(input_dir, recursive)
        elif pattern == StructurePattern.METHOD_SEED_HIERARCHY:
            runs = self._discover_method_seed_hierarchy(input_dir, recursive)
        elif pattern == StructurePattern.METHOD_GROUPED:
            runs = self._discover_method_grouped(input_dir, recursive)
        elif pattern == StructurePattern.NESTED_TIMESTAMP:
            runs = self._discover_nested_timestamp(input_dir, recursive)
        elif pattern == StructurePattern.ORIGINAL_TRAINING:
            runs = self._discover_original_training(input_dir, recursive)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        logger.info(f"Discovered {len(runs)} runs")
        return runs

    def _discover_flat_benchmark(self, input_dir: Path, recursive: bool) -> List[RunInfo]:
        """
        Discover runs in FLAT_BENCHMARK structure.

        Pattern: artifacts/benchmark/DATE/phaseX/bench_pX_*

        Args:
            input_dir: Root directory
            recursive: Search recursively

        Returns:
            List of discovered runs
        """
        runs = []

        # Find all bench_p* directories
        if recursive:
            bench_dirs = input_dir.rglob("bench_p*")
        else:
            bench_dirs = input_dir.glob("*/bench_p*")

        for bench_dir in bench_dirs:
            if not bench_dir.is_dir():
                continue

            run_info = self._create_run_info(
                bench_dir,
                pattern=StructurePattern.FLAT_BENCHMARK
            )

            if run_info and run_info.is_valid():
                runs.append(run_info)

        return runs

    def _discover_nested_benchmark(self, input_dir: Path, recursive: bool) -> List[RunInfo]:
        """
        Discover runs in NESTED_BENCHMARK structure.

        Pattern: artifacts/benchmark/DATE/phaseX/bench_pX_*/run_*

        Args:
            input_dir: Root directory
            recursive: Search recursively

        Returns:
            List of discovered runs
        """
        runs = []

        # Find all run_* directories under bench_p* directories
        if recursive:
            bench_dirs = input_dir.rglob("bench_p*")
        else:
            bench_dirs = input_dir.glob("*/bench_p*")

        for bench_dir in bench_dirs:
            if not bench_dir.is_dir():
                continue

            # Look for run_* subdirectories
            for run_dir in bench_dir.glob("run_*"):
                if not run_dir.is_dir():
                    continue

                run_info = self._create_run_info(
                    run_dir,
                    parent_dir=bench_dir,
                    pattern=StructurePattern.NESTED_BENCHMARK
                )

                if run_info and run_info.is_valid():
                    runs.append(run_info)

        return runs

    def _discover_method_seed_hierarchy(self, input_dir: Path, recursive: bool) -> List[RunInfo]:
        """
        Discover runs in METHOD_SEED_HIERARCHY structure.

        Pattern: artifacts/benchmark/DATE/METHOD/seed_X/run_*

        Args:
            input_dir: Root directory
            recursive: Search recursively

        Returns:
            List of discovered runs
        """
        runs = []

        # Find all seed_* directories
        if recursive:
            seed_dirs = input_dir.rglob("seed_*")
        else:
            seed_dirs = input_dir.glob("*/seed_*")

        for seed_dir in seed_dirs:
            if not seed_dir.is_dir():
                continue

            # Extract seed from directory name
            seed = self.extractor.extract_seed_from_dirname(seed_dir.name)

            # Find run_* subdirectories
            for run_dir in seed_dir.glob("run_*"):
                if not run_dir.is_dir():
                    continue

                run_info = self._create_run_info(
                    run_dir,
                    parent_dir=seed_dir,
                    pattern=StructurePattern.METHOD_SEED_HIERARCHY
                )

                if run_info:
                    run_info.seed = seed
                    # Extract method from parent directory
                    method_dir = seed_dir.parent
                    if method_dir.name not in ["benchmark", "training"]:
                        run_info.metadata["method_dir"] = method_dir.name

                    if run_info.is_valid():
                        runs.append(run_info)

        return runs

    def _discover_method_grouped(self, input_dir: Path, recursive: bool) -> List[RunInfo]:
        """
        Discover runs in METHOD_GROUPED structure.

        Pattern: artifacts/benchmark/DATE/METHOD/run_*

        Args:
            input_dir: Root directory
            recursive: Search recursively

        Returns:
            List of discovered runs
        """
        runs = []

        # Define directories to exclude (common non-method directories)
        excluded_dirs = {"phase", "bench", "seed_", "logs", "tensorboard", "metrics", "configs"}

        # Find all immediate subdirectories that contain run_* directories
        if recursive:
            # In recursive mode, find all directories that might be method directories
            method_dirs = []
            for item in input_dir.rglob("*"):
                if item.is_dir() and not any(item.name.startswith(prefix) for prefix in ["phase", "bench", "seed_"]):
                    # Check if this directory contains run_* subdirectories
                    if any(item.glob("run_*")):
                        method_dirs.append(item)
        else:
            # In non-recursive mode, only check immediate subdirectories
            method_dirs = [
                d for d in input_dir.iterdir()
                if d.is_dir() and not any(d.name.startswith(prefix) for prefix in ["phase", "bench", "seed_"])
            ]

        for method_dir in method_dirs:
            method_name = method_dir.name

            # Find run_* subdirectories
            for run_dir in method_dir.glob("run_*"):
                if not run_dir.is_dir():
                    continue

                run_info = self._create_run_info(
                    run_dir,
                    parent_dir=method_dir,
                    pattern=StructurePattern.METHOD_GROUPED
                )

                if run_info:
                    # Override method with the parent directory name
                    run_info.method = method_name
                    run_info.metadata["method_dir"] = method_name

                    if run_info.is_valid():
                        runs.append(run_info)

        logger.info(f"Found {len(runs)} runs in METHOD_GROUPED structure")
        return runs

    def _discover_nested_timestamp(self, input_dir: Path, recursive: bool) -> List[RunInfo]:
        """
        Discover runs in NESTED_TIMESTAMP structure.

        Pattern: artifacts/benchmark/DIR/TIMESTAMP/METHOD/SEED/run_*

        This pattern supports curriculum learning benchmarks where runs are organized
        under timestamp directories, then by method, then by seed.

        Args:
            input_dir: Root directory
            recursive: Search recursively

        Returns:
            List of discovered runs
        """
        runs = []

        # Pattern to match timestamp directories (YYYYMMDD_HHMMSS)
        timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")

        # Find all timestamp directories
        timestamp_dirs = [d for d in input_dir.iterdir()
                         if d.is_dir() and timestamp_pattern.match(d.name)]

        for ts_dir in timestamp_dirs:
            timestamp = ts_dir.name

            # Find method directories within each timestamp
            method_dirs = [d for d in ts_dir.iterdir()
                          if d.is_dir() and not d.name.startswith((".", "_"))]

            for method_dir in method_dirs:
                method_name = method_dir.name

                # Find seed directories within each method
                seed_dirs = list(method_dir.glob("seed_*"))

                for seed_dir in seed_dirs:
                    # Extract seed number
                    seed = self.extractor.extract_seed_from_dirname(seed_dir.name)

                    # Find run directories within each seed
                    for run_dir in seed_dir.glob("run_*"):
                        if not run_dir.is_dir():
                            continue

                        run_info = self._create_run_info(
                            run_dir,
                            parent_dir=seed_dir,
                            pattern=StructurePattern.NESTED_TIMESTAMP
                        )

                        if run_info:
                            run_info.method = method_name
                            run_info.seed = seed
                            run_info.metadata["timestamp"] = timestamp
                            run_info.metadata["method_dir"] = method_name
                            run_info.metadata["seed_dir"] = seed_dir.name

                            if run_info.is_valid():
                                runs.append(run_info)

        # Also check for any non-timestamp directories with METHOD/run_* structure
        # (like the root-level 'baseline' directory)
        non_timestamp_dirs = [d for d in input_dir.iterdir()
                              if d.is_dir() and not timestamp_pattern.match(d.name)
                              and not d.name.startswith((".", "_"))]

        for method_dir in non_timestamp_dirs:
            # Check for direct run_* children (METHOD_GROUPED style)
            for run_dir in method_dir.glob("run_*"):
                if not run_dir.is_dir():
                    continue

                run_info = self._create_run_info(
                    run_dir,
                    parent_dir=method_dir,
                    pattern=StructurePattern.NESTED_TIMESTAMP
                )

                if run_info:
                    run_info.method = method_dir.name
                    run_info.metadata["method_dir"] = method_dir.name

                    if run_info.is_valid():
                        runs.append(run_info)

            # Also check for seed_*/run_* structure
            for seed_dir in method_dir.glob("seed_*"):
                seed = self.extractor.extract_seed_from_dirname(seed_dir.name)

                for run_dir in seed_dir.glob("run_*"):
                    if not run_dir.is_dir():
                        continue

                    run_info = self._create_run_info(
                        run_dir,
                        parent_dir=seed_dir,
                        pattern=StructurePattern.NESTED_TIMESTAMP
                    )

                    if run_info:
                        run_info.method = method_dir.name
                        run_info.seed = seed
                        run_info.metadata["method_dir"] = method_dir.name
                        run_info.metadata["seed_dir"] = seed_dir.name

                        if run_info.is_valid():
                            runs.append(run_info)

        logger.info(f"Found {len(runs)} runs in NESTED_TIMESTAMP structure")
        return runs

    def _discover_original_training(self, input_dir: Path, recursive: bool) -> List[RunInfo]:
        """
        Discover runs in ORIGINAL_TRAINING structure.

        Pattern: artifacts/training/run_*

        Args:
            input_dir: Root directory
            recursive: Search recursively

        Returns:
            List of discovered runs
        """
        runs = []

        # Find all run_* directories
        if recursive:
            run_dirs = input_dir.rglob("run_*")
        else:
            run_dirs = input_dir.glob("run_*")

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue

            run_info = self._create_run_info(
                run_dir,
                pattern=StructurePattern.ORIGINAL_TRAINING
            )

            if run_info and run_info.is_valid():
                runs.append(run_info)

        return runs

    def _create_run_info(
        self,
        run_dir: Path,
        parent_dir: Optional[Path] = None,
        pattern: Optional[StructurePattern] = None
    ) -> Optional[RunInfo]:
        """
        Create a RunInfo object from a directory.

        Args:
            run_dir: Directory containing run data
            parent_dir: Parent directory (for extracting additional metadata)
            pattern: Structure pattern being used

        Returns:
            RunInfo object or None if invalid
        """
        metadata = {}

        # Extract metadata from directory names
        if parent_dir:
            parent_metadata = self.extractor.extract_benchmark_metadata(parent_dir.name)
            metadata.update(parent_metadata)

        run_metadata = self.extractor.extract_run_metadata(run_dir.name)
        metadata.update(run_metadata)

        # Also try extracting from current directory if it's a benchmark dir
        bench_metadata = self.extractor.extract_benchmark_metadata(run_dir.name)
        metadata.update(bench_metadata)

        # Extract phase from path
        phase = self.extractor.extract_phase_from_path(run_dir)

        # Find data files
        config_path = self._find_config(run_dir)
        metrics_path = self._find_metrics(run_dir)
        tensorboard_path = self._find_tensorboard(run_dir)

        # Create RunInfo
        run_info = RunInfo(
            path=run_dir.resolve(),
            name=run_dir.name,
            phase=phase or metadata.get("phase"),
            method=metadata.get("method") or metadata.get("network"),
            seed=metadata.get("seed"),
            variant=metadata.get("variant"),
            config_path=config_path,
            metrics_path=metrics_path,
            tensorboard_path=tensorboard_path,
            metadata=metadata,
            pattern=pattern
        )

        return run_info

    def _find_config(self, run_dir: Path) -> Optional[Path]:
        """Find config.yaml in run directory."""
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            return config_path.resolve()
        return None

    def _find_metrics(self, run_dir: Path) -> Optional[Path]:
        """Find training_metrics.json in run directory."""
        # Check metrics subdirectory first
        metrics_path = run_dir / "metrics" / "training_metrics.json"
        if metrics_path.exists():
            return metrics_path.resolve()

        # Check root directory
        metrics_path = run_dir / "training_metrics.json"
        if metrics_path.exists():
            return metrics_path.resolve()

        return None

    def _find_tensorboard(self, run_dir: Path) -> Optional[Path]:
        """Find tensorboard directory in run directory."""
        # Check tensorboard subdirectory
        tb_path = run_dir / "tensorboard"
        if tb_path.exists() and tb_path.is_dir():
            return tb_path.resolve()

        # Check logs subdirectory (original training pattern)
        logs_path = run_dir / "logs"
        if logs_path.exists() and logs_path.is_dir():
            # Check if it has tensorboard events
            if any(logs_path.glob("events.out.tfevents.*")):
                return logs_path.resolve()

        return None

    def filter_runs(
        self,
        runs: List[RunInfo],
        phase: Optional[str] = None,
        method: Optional[str] = None,
        seed: Optional[int] = None,
        has_config: bool = False,
        has_metrics: bool = False,
        has_tensorboard: bool = False
    ) -> List[RunInfo]:
        """
        Filter runs based on various criteria.

        Args:
            runs: List of runs to filter
            phase: Filter by phase
            method: Filter by method
            seed: Filter by seed
            has_config: Require config file
            has_metrics: Require metrics file
            has_tensorboard: Require tensorboard data

        Returns:
            Filtered list of runs
        """
        filtered = runs

        if phase is not None:
            filtered = [r for r in filtered if r.phase == phase]

        if method is not None:
            filtered = [r for r in filtered if r.method == method]

        if seed is not None:
            filtered = [r for r in filtered if r.seed == seed]

        if has_config:
            filtered = [r for r in filtered if r.has_config()]

        if has_metrics:
            filtered = [r for r in filtered if r.has_metrics()]

        if has_tensorboard:
            filtered = [r for r in filtered if r.has_tensorboard()]

        return filtered

    def group_runs_by(
        self,
        runs: List[RunInfo],
        group_by: str = "phase"
    ) -> Dict[Any, List[RunInfo]]:
        """
        Group runs by a specific attribute.

        Args:
            runs: List of runs to group
            group_by: Attribute to group by (phase, method, seed, pattern)

        Returns:
            Dictionary mapping group values to lists of runs
        """
        groups = {}

        for run in runs:
            key = getattr(run, group_by, None)
            if key is not None:
                if key not in groups:
                    groups[key] = []
                groups[key].append(run)

        return groups


def main():
    """Example usage of the benchmark discovery system."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python benchmark_discovery.py <directory>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])

    # Create discovery instance
    discovery = BenchmarkDiscovery(verbose=True)

    try:
        # Detect structure
        pattern = discovery.detect_structure(input_dir)
        print(f"\nDetected pattern: {pattern.value}")

        # Discover runs
        runs = discovery.discover_runs(input_dir)

        print(f"\nFound {len(runs)} runs:")
        for run in runs:
            print(f"\n{run}")
            print(f"  Path: {run.path}")
            print(f"  Config: {run.has_config()}")
            print(f"  Metrics: {run.has_metrics()}")
            print(f"  Tensorboard: {run.has_tensorboard()}")
            if run.metadata:
                print(f"  Metadata: {run.metadata}")

        # Group by phase
        if runs:
            groups = discovery.group_runs_by(runs, "phase")
            print(f"\n\nRuns by phase:")
            for phase, phase_runs in sorted(groups.items()):
                print(f"  {phase}: {len(phase_runs)} runs")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
