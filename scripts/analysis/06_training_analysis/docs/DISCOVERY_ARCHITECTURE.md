# Benchmark Discovery Architecture

## Overview

The Benchmark Discovery System provides a flexible, pattern-aware architecture for discovering and analyzing training runs across multiple directory structures. It automatically detects organizational patterns and extracts structured metadata from directory names.

## Core Components

### 1. StructurePattern Enum

Defines the supported directory structure patterns:

```python
class StructurePattern(Enum):
    FLAT_BENCHMARK = "flat_benchmark"              # phase1-5
    NESTED_BENCHMARK = "nested_benchmark"          # phase6-7
    METHOD_SEED_HIERARCHY = "method_seed_hierarchy"  # baseline/seed_X/run_*
    ORIGINAL_TRAINING = "original_training"        # simple training/run_*
```

### 2. RunInfo Dataclass

Stores comprehensive information about a discovered training run:

```python
@dataclass
class RunInfo:
    path: Path                          # Absolute path to run directory
    name: str                           # Directory name
    phase: Optional[str]                # Training phase (e.g., "phase1")
    method: Optional[str]               # Algorithm name (e.g., "drqn")
    seed: Optional[int]                 # Random seed
    variant: Optional[str]              # Configuration variant
    config_path: Optional[Path]         # Path to config.yaml
    metrics_path: Optional[Path]        # Path to training_metrics.json
    tensorboard_path: Optional[Path]    # Path to tensorboard directory
    metadata: Dict[str, Any]            # Additional extracted metadata
    pattern: Optional[StructurePattern] # Structure pattern
```

**Key Methods:**
- `is_valid()` - Check if run has valid data files
- `has_config()` - Check if config file exists
- `has_metrics()` - Check if metrics file exists
- `has_tensorboard()` - Check if tensorboard data exists

### 3. MetadataExtractor Class

Extracts structured metadata from directory and file names:

```python
class MetadataExtractor:
    # Patterns for different naming conventions
    BENCHMARK_PATTERN = r"bench_p(?P<phase>\d+)_(?P<network>\w+)_(?P<params>.*?)_s(?P<seed>\d+)_(?P<timestamp>\d+_\d+)"
    OBS_PATTERN = r"bench_p(?P<phase>\d+)_obs_(?P<obs_type>\w+)_s(?P<seed>\d+)_(?P<timestamp>\d+_\d+)"
    RUN_PATTERN = r"run_(?P<method>\w+)_(?P<timestamp>\d{8}_\d{6})"
    SEED_PATTERN = r"seed_(?P<seed>\d+)"
    PARAM_PATTERN = r"(?P<key>[a-z]+)(?P<value>[\d.]+)"
```

**Key Methods:**
- `extract_benchmark_metadata(dirname)` - Parse benchmark directory names
- `extract_run_metadata(dirname)` - Parse run directory names
- `extract_seed_from_dirname(dirname)` - Extract seed from directory name
- `extract_phase_from_path(path)` - Extract phase from path components

**Extraction Examples:**

```python
# bench_p1_drqn_h128_s42_251205_1547 →
{
    'phase': 1,
    'network': 'drqn',
    'hidden': 128,
    'seed': 42,
    'timestamp': '251205_1547',
    'variant': 'drqn_hidden128'
}

# bench_p6_obs_centraw_s42_251206_1333 →
{
    'phase': 6,
    'obs_type': 'centraw',
    'seed': 42,
    'timestamp': '251206_1333',
    'variant': 'obs_centraw'
}

# run_drqn_20251207_195352 →
{
    'method': 'drqn',
    'timestamp': '20251207_195352'
}
```

### 4. BenchmarkDiscovery Class

Main interface for discovering and managing training runs:

```python
class BenchmarkDiscovery:
    def __init__(self, verbose: bool = False)

    # Core discovery methods
    def detect_structure(self, input_dir: Path) -> StructurePattern
    def discover_runs(self, input_dir: Path, recursive: bool = True,
                      pattern: Optional[StructurePattern] = None) -> List[RunInfo]

    # Pattern-specific discovery
    def _discover_flat_benchmark(self, input_dir: Path, recursive: bool) -> List[RunInfo]
    def _discover_nested_benchmark(self, input_dir: Path, recursive: bool) -> List[RunInfo]
    def _discover_method_seed_hierarchy(self, input_dir: Path, recursive: bool) -> List[RunInfo]
    def _discover_original_training(self, input_dir: Path, recursive: bool) -> List[RunInfo]

    # Filtering and grouping
    def filter_runs(self, runs: List[RunInfo], **criteria) -> List[RunInfo]
    def group_runs_by(self, runs: List[RunInfo], group_by: str) -> Dict[Any, List[RunInfo]]
```

## Supported Directory Patterns

### Pattern 1: FLAT_BENCHMARK (Phase 1-5)

```
artifacts/benchmark/20251206_initial/
├── phase1_network/
│   ├── bench_p1_drqn_h128_s42_251205_1547/
│   │   ├── config.yaml
│   │   ├── metrics/training_metrics.json
│   │   └── tensorboard/events.out.tfevents.*
│   └── bench_p1_drqn_h256_s42_251205_1548/
├── phase2_memory/
└── phase3_exploration/
```

**Characteristics:**
- Direct run directories under phase folders
- Metadata in benchmark directory name
- Used in initial benchmarks (phase1-5)

### Pattern 2: NESTED_BENCHMARK (Phase 6-7)

```
artifacts/benchmark/20251206_initial/
├── phase6_observation/
│   └── bench_p6_obs_centraw_s42_251206_1333/
│       └── run_drqn_20251206_133329/
│           ├── metrics/training_metrics.json
│           └── tensorboard/
```

**Characteristics:**
- Run directories nested under benchmark folders
- Metadata split between benchmark and run directories
- Used in later benchmarks (phase6-7)

### Pattern 3: METHOD_SEED_HIERARCHY

```
artifacts/benchmark/20251207_stateless_lstm/
├── baseline/
│   ├── seed_42/
│   │   └── run_drqn_20251207_203043/
│   │       ├── config.yaml
│   │       ├── metrics/training_metrics.json
│   │       └── tensorboard/
│   └── seed_123/
│       └── run_drqn_20251208_153057/
```

**Characteristics:**
- Organized by method/seed/run hierarchy
- Clear separation of seeds and runs
- Used for controlled experiments

### Pattern 4: ORIGINAL_TRAINING

```
artifacts/training/
├── run_drqn_20251207_195352/
│   ├── metrics/training_metrics.json
│   └── logs/events.out.tfevents.*
└── run_drqn_20251208_101234/
```

**Characteristics:**
- Simple flat structure
- Direct run directories
- Used for ad-hoc training runs

## Usage Examples

### Basic Discovery

```python
from pathlib import Path
from benchmark_discovery import BenchmarkDiscovery

discovery = BenchmarkDiscovery(verbose=True)

# Auto-detect structure and discover runs
benchmark_dir = Path("/workspace/artifacts/benchmark/20251206_initial")
runs = discovery.discover_runs(benchmark_dir)

print(f"Found {len(runs)} runs")
for run in runs:
    print(f"{run.name}: phase={run.phase}, method={run.method}, seed={run.seed}")
```

### Filter Runs

```python
# Filter by multiple criteria
filtered = discovery.filter_runs(
    runs,
    phase="phase1",
    method="drqn",
    has_metrics=True,
    has_tensorboard=True
)

print(f"Found {len(filtered)} phase1 DRQN runs with complete data")
```

### Group and Analyze

```python
# Group runs by phase
by_phase = discovery.group_runs_by(runs, "phase")

for phase, phase_runs in sorted(by_phase.items()):
    print(f"{phase}: {len(phase_runs)} runs")

    # Count by method
    by_method = discovery.group_runs_by(phase_runs, "method")
    for method, method_runs in by_method.items():
        print(f"  {method}: {len(method_runs)} runs")
```

### Access Metadata

```python
for run in runs:
    print(f"\n{run.name}:")
    print(f"  Path: {run.path}")
    print(f"  Phase: {run.phase}")
    print(f"  Method: {run.method}")
    print(f"  Seed: {run.seed}")

    # Access extracted metadata
    if "hidden" in run.metadata:
        print(f"  Hidden size: {run.metadata['hidden']}")

    # Check data availability
    print(f"  Has metrics: {run.has_metrics()}")
    print(f"  Has tensorboard: {run.has_tensorboard()}")
```

### Cross-Structure Discovery

```python
# Discover runs across multiple structure patterns
test_dirs = [
    "/workspace/artifacts/benchmark/20251206_initial",      # FLAT_BENCHMARK
    "/workspace/artifacts/benchmark/20251207_stateless_lstm",  # METHOD_SEED_HIERARCHY
    "/workspace/artifacts/training",                        # ORIGINAL_TRAINING
]

all_runs = []
for test_dir in test_dirs:
    path = Path(test_dir)
    if path.exists():
        runs = discovery.discover_runs(path)
        all_runs.extend(runs)

print(f"Total runs: {len(all_runs)}")

# Group by pattern
by_pattern = discovery.group_runs_by(all_runs, "pattern")
for pattern, pattern_runs in by_pattern.items():
    print(f"{pattern.value}: {len(pattern_runs)} runs")
```

### Manual Structure Detection

```python
# Explicitly detect and use a specific pattern
pattern = discovery.detect_structure(benchmark_dir)
print(f"Detected pattern: {pattern.value}")

# Discover with specific pattern
runs = discovery.discover_runs(benchmark_dir, pattern=pattern)
```

## Integration with Analysis Tools

### Loading Training Metrics

```python
import json

for run in runs:
    if run.has_metrics():
        with open(run.metrics_path) as f:
            metrics = json.load(f)

        print(f"{run.name}:")
        print(f"  Episodes: {len(metrics)}")
        print(f"  Final reward: {metrics[-1]['reward']}")
```

### Loading TensorBoard Data

```python
from tensorboard.backend.event_processing import event_accumulator

for run in runs:
    if run.has_tensorboard():
        ea = event_accumulator.EventAccumulator(str(run.tensorboard_path))
        ea.Reload()

        # Access scalar data
        tags = ea.Tags()['scalars']
        print(f"{run.name}: {len(tags)} scalar metrics")
```

### Loading Config Files

```python
import yaml

for run in runs:
    if run.has_config():
        with open(run.config_path) as f:
            config = yaml.safe_load(f)

        print(f"{run.name}:")
        print(f"  Learning rate: {config.get('learning_rate')}")
        print(f"  Batch size: {config.get('batch_size')}")
```

## Design Principles

1. **Pattern Agnostic**: Automatically detects and handles multiple directory structures
2. **Metadata Extraction**: Intelligent parsing of directory names for structured metadata
3. **Flexible Filtering**: Rich filtering and grouping capabilities
4. **Data Validation**: Built-in checks for data availability and completeness
5. **Extensible**: Easy to add new patterns and metadata extractors
6. **Type Safe**: Full type hints and dataclasses for better IDE support
7. **Well Documented**: Comprehensive docstrings and examples

## Future Extensions

Potential enhancements to the architecture:

1. **Additional Patterns**: Support for new directory structures as they emerge
2. **Config Parsing**: Direct integration with config files for metadata
3. **Lazy Loading**: Deferred loading of metrics and tensorboard data
4. **Caching**: Cache discovered runs for faster repeated access
5. **Comparison Tools**: Built-in methods for comparing runs
6. **Export Functions**: Export run information to CSV, JSON, etc.
7. **Validation Rules**: Configurable validation for run completeness
8. **Query Language**: SQL-like query interface for complex filtering

## Testing

Run the test suite:

```bash
python analysis/training_analysis/test_discovery.py
```

Run example usage:

```bash
python analysis/training_analysis/example_usage.py
```

Run as standalone CLI:

```bash
python analysis/training_analysis/benchmark_discovery.py /path/to/benchmark/dir
```

## File Locations

- **Core Module**: `/workspace/analysis/training_analysis/benchmark_discovery.py`
- **Test Suite**: `/workspace/analysis/training_analysis/test_discovery.py`
- **Examples**: `/workspace/analysis/training_analysis/example_usage.py`
- **Documentation**: `/workspace/analysis/training_analysis/DISCOVERY_ARCHITECTURE.md`
