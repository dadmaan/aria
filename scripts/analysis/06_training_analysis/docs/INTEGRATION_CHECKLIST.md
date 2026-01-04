# Integration Checklist

This document tracks the integration of new modules into `analyze_training_runs.py`.

## Module Dependencies

The refactored script requires these three modules in the same directory:

### 1. benchmark_discovery.py
**Required Classes:**
- [x] `BenchmarkDiscovery` - Main discovery class
- [x] `RunInfo` - Dataclass for run metadata

**Integration Points:**
- Line 38: Import statement
- Lines 991-993: Discovery instantiation and usage
```python
discovery = BenchmarkDiscovery(config.input_dir)
run_infos = discovery.discover_runs(recursive=config.recursive)
```

**Expected API:**
```python
class BenchmarkDiscovery:
    def __init__(self, base_dir: Path): ...
    def discover_runs(self, recursive: bool = False) -> List[RunInfo]: ...

@dataclass
class RunInfo:
    name: str
    path: Path
    # ... other metadata
```

### 2. data_loaders.py
**Required Classes:**
- [x] `ConfigLoader` - Load agent configurations
- [x] `MetricsLoader` - Load training metrics
- [x] `TensorBoardLoader` - Load TensorBoard data
- [x] `TrainingConfig` - Configuration dataclass

**Integration Points:**
- Line 39: Import statement
- Lines 153-155: Loader instantiation in `load_run_data()`
- Lines 158-167: Config loading
- Lines 164-167: Metrics loading
- Lines 182-185: TensorBoard loading

**Expected API:**
```python
class ConfigLoader:
    def load_config(self, run_dir: Path) -> Optional[Dict[str, Any]]: ...

class MetricsLoader:
    def load_metrics(self, run_dir: Path) -> Optional[Dict[str, Any]]: ...

class TensorBoardLoader:
    def load_tensorboard_data(self, run_dir: Path) -> Dict[str, List[Tuple[int, float]]]: ...
```

### 3. latex_tables.py
**Required Classes:**
- [x] `LatexTableGenerator` - Generate LaTeX tables
- [x] `AblationAnalyzer` - Perform ablation analysis

**Integration Points:**
- Line 40: Import statement
- Line 1071: Generator instantiation
- Lines 1074-1086: Performance table generation
- Lines 1089-1102: Ablation analysis and table generation

**Expected API:**
```python
class LatexTableGenerator:
    def generate_performance_table(self, data: List[Dict]) -> str: ...
    def generate_ablation_table(self, results: List[Dict]) -> str: ...

class AblationAnalyzer:
    def compare_to_baseline(
        self, baseline: RunMetrics, runs: List[RunMetrics]
    ) -> List[Dict]: ...
```

## Feature Implementation Status

### Core Refactoring
- [x] Import new modules
- [x] Update AnalysisConfig dataclass
- [x] Refactor load_run_data() to use new loaders
- [x] Update run_analysis() to use BenchmarkDiscovery
- [x] Preserve all existing visualization functions
- [x] Preserve all existing statistics functions

### New Features
- [x] Recursive run discovery
- [x] Seed aggregation (`aggregate_across_seeds()`)
- [x] Run grouping (`generate_grouped_report()`)
- [x] Multiple output formats
- [x] LaTeX table generation
- [x] Baseline comparison
- [x] Ablation analysis

### CLI Arguments
- [x] --recursive
- [x] --group-by {phase,method,seed,variant}
- [x] --output-format {markdown,latex,json,all}
- [x] --aggregate-seeds
- [x] --baseline

### Documentation
- [x] REFACTORING_SUMMARY.md
- [x] NEW_FEATURES_GUIDE.md
- [x] INTEGRATION_CHECKLIST.md (this file)

## Testing Strategy

### Unit Tests Needed
1. **Test aggregate_across_seeds()**
   - Single seed per run (should pass through)
   - Multiple seeds per run (should aggregate)
   - Mixed scenario (some single, some multiple)

2. **Test generate_grouped_report()**
   - Group by phase
   - Group by method
   - Group by seed
   - Group by variant
   - Handle missing group identifiers

3. **Test load_run_data() with new loaders**
   - Standard directory structure
   - Alternative file locations
   - Missing files (should handle gracefully)

### Integration Tests Needed
1. **End-to-end with simple structure**
   ```bash
   python analyze_training_runs.py --input-dir test_data/simple
   ```

2. **End-to-end with recursive discovery**
   ```bash
   python analyze_training_runs.py --input-dir test_data/nested --recursive
   ```

3. **End-to-end with seed aggregation**
   ```bash
   python analyze_training_runs.py --input-dir test_data/seeds --aggregate-seeds
   ```

4. **End-to-end with grouping**
   ```bash
   python analyze_training_runs.py --input-dir test_data/phases --group-by phase
   ```

5. **End-to-end with LaTeX output**
   ```bash
   python analyze_training_runs.py --input-dir test_data/simple --output-format latex --baseline run_baseline
   ```

### Manual Testing Checklist
- [ ] Run with default arguments (backward compatibility)
- [ ] Run with --recursive on nested structure
- [ ] Run with --aggregate-seeds on seeded runs
- [ ] Run with each --group-by option
- [ ] Run with each --output-format option
- [ ] Run with --baseline and verify ablation table
- [ ] Verify all visualizations generate correctly
- [ ] Verify markdown reports generate correctly
- [ ] Verify JSON output is valid
- [ ] Verify LaTeX tables compile correctly

## API Contracts

### Input Expectations
The script expects run directories with this structure:
```
run_name/
├── configs/
│   └── agent_config.json      # Agent configuration
├── metrics/
│   ├── detailed_rewards.json  # Primary metrics file
│   └── training_metrics.json  # Alternative metrics file
└── logs/
    └── events.out.tfevents.*  # TensorBoard logs
```

### Output Guarantees
The script produces:
```
outputs/training_analysis/analysis_TIMESTAMP/
├── summary_report.md           # Always (if markdown format)
├── config_comparison.md        # Always (if markdown format)
├── performance_table.tex       # If latex format
├── ablation_table.tex         # If latex format + baseline
├── plots/
│   ├── reward_curves_overlay.png
│   ├── performance_ranking.png
│   ├── loss_curves.png
│   ├── lr_schedules.png
│   ├── reward_distributions.png
│   ├── stability_comparison.png
│   └── radar_comparison.png
├── data/
│   └── analysis_results.json  # If json format
└── grouped_reports/           # If --group-by specified
    ├── group1/
    │   ├── summary.md
    │   └── plots/
    └── group2/
        ├── summary.md
        └── plots/
```

## Known Dependencies

### Python Packages (existing)
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- tqdm
- tensorboard (optional, for TB data)

### Python Packages (new requirements - if any)
- None additional beyond what the new modules require

## Backward Compatibility

### Preserved Behavior
- Default behavior without flags remains unchanged
- All existing CLI arguments work as before
- RunMetrics dataclass unchanged (no breaking changes)
- All visualization functions unchanged
- All statistics functions unchanged
- Output directory structure compatible (adds new files, doesn't remove)

### Potential Breaking Changes
- None identified
- New imports might fail if modules not present (expected)
- Directory structure differences handled by BenchmarkDiscovery

## Migration Path

### For Existing Users
1. Add three new module files to directory
2. Existing scripts continue to work with default arguments
3. Opt-in to new features via CLI flags

### For New Features
1. Use `--recursive` for complex directory structures
2. Use `--aggregate-seeds` for multi-seed experiments
3. Use `--group-by` for dimensional analysis
4. Use `--output-format latex` for papers

## Validation Checklist

Before deploying:
- [x] Script compiles without syntax errors
- [x] Help text displays correctly
- [x] All imports resolve (when modules present)
- [ ] Test with sample data
- [ ] Generate all output formats
- [ ] Verify visualizations render
- [ ] Check LaTeX tables compile
- [ ] Validate JSON structure

## Notes

### Design Decisions
1. **Loader instantiation in load_run_data()**: Could be moved to class level for efficiency, but kept local for clarity and modularity
2. **Grouping logic in generate_grouped_report()**: Extraction logic could be moved to separate functions for extensibility
3. **LaTeX generation only with baseline**: Could make baseline optional in ablation table, but typically ablation requires baseline

### Future Enhancements
1. Add support for custom grouping logic via plugins
2. Add support for custom metrics in LaTeX tables
3. Add caching for large datasets
4. Add parallel processing for group reports
5. Add interactive HTML reports
6. Add comparison across multiple analysis runs

### Performance Considerations
1. Seed aggregation requires loading all seeds into memory
2. Grouping creates multiple plot files (disk I/O)
3. LaTeX table generation is fast (no heavy computation)
4. BenchmarkDiscovery recursive scan could be slow on large filesystems
