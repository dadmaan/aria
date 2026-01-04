# Training Analysis Module

This module provides tools for analyzing and visualizing training runs.

## Usage

```bash
python analyze_training_runs.py --input-dir artifacts/training --output-dir outputs/training_analysis
```

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `artifacts/training` | Directory containing training runs |
| `--output-dir` | `outputs/training_analysis` | Base output directory |
| `--n-jobs` | `1` | Number of parallel workers |
| `--window` | `20` | Rolling window size for smoothing |
| `--runs` | `all` | Comma-separated list of specific runs |
