# LaTeX Table Generation for NeurIPS Paper

This module provides tools for generating publication-ready LaTeX tables for the NeurIPS 2024 paper on reinforcement learning for music generation.

## Overview

The module consists of two main components:

1. **`LatexTableGenerator`**: Generates various types of LaTeX tables
2. **`AblationAnalyzer`**: Analyzes ablation studies and groups runs by dimensions

## Files

- `latex_tables.py` - Main module with generator and analyzer classes
- `example_latex_tables.py` - Comprehensive examples demonstrating usage
- `README_LATEX_TABLES.md` - This documentation

## Installation

No additional dependencies beyond the standard training analysis pipeline:

```bash
# Already installed with training analysis
pip install numpy scipy pandas matplotlib seaborn
```

## Quick Start

### Basic Usage

```python
from latex_tables import LatexTableGenerator, AblationAnalyzer

# Initialize generator
generator = LatexTableGenerator(decimals=2, bold_best=True)

# Generate training comparison table
table = generator.generate_training_comparison_table(
    runs=runs,
    baseline_name="end-to-end-dqn",
)
print(table)
```

### Running Examples

```bash
# Run with mock data (no training runs needed)
python example_latex_tables.py --mode mock

# Run with real data (requires training runs in artifacts/training/)
python example_latex_tables.py --mode real

# Run both
python example_latex_tables.py --mode both
```

## Table Types

### 1. Training Comparison Table

Compares different methods on training performance metrics.

**Example:**

```python
table = generator.generate_training_comparison_table(
    runs=runs,
    baseline_name="end-to-end-dqn",
    caption="Training performance comparison. Mean $\\pm$ std over 5 seeds.",
    label="tab:training_results",
    include_metrics=['reward', 'time', 'efficiency'],
)
```

**Output:**

```latex
\begin{table}[t]
\centering
\caption{Training performance comparison. Mean $\pm$ std over 5 seeds.}
\label{tab:training_results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Episode Reward} & \textbf{Training Time} & \textbf{Sample Efficiency} \\
\midrule
\textbf{Rainbow Drqn} & $\mathbf{0.52 \pm 0.02}$ & 1.8h & 1.2$\times$ \\
Vanilla Drqn & $0.48 \pm 0.04$ & 1.9h & 1.1$\times$ \\
End To End Dqn & $0.45 \pm 0.03$ & 2.1h & 1.0$\times$ \\
\bottomrule
\end{tabular}
\end{table}
```

### 2. Ablation Study Table

Shows the impact of removing individual components.

**Example:**

```python
table = generator.generate_ablation_table(
    runs=runs,
    full_method_name="full-rainbow-drqn",
    components=["c51", "prioritized-replay", "dueling"],
    caption="Ablation: Rainbow DRQN components.",
    label="tab:rainbow_ablation",
)
```

**Output:**

```latex
\begin{table}[t]
\centering
\caption{Ablation: Rainbow DRQN components.}
\label{tab:rainbow_ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{Reward} & \textbf{$\Delta$} \\
\midrule
Full method & $0.52$ & --- \\
$-$ Prioritized-replay & $0.49$ & $-5.8\%$ \\
$-$ C51 & $0.48$ & $-7.7\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

### 3. Learning Rate Schedule Table

Compares different learning rate schedules.

**Example:**

```python
table = generator.generate_lr_schedule_table(
    runs=runs,
    caption="Ablation: learning rate schedule.",
    label="tab:lr_ablation",
)
```

**Output:**

```latex
\begin{table}[t]
\centering
\caption{Ablation: learning rate schedule.}
\label{tab:lr_ablation}
\begin{tabular}{lccc}
\toprule
\textbf{LR Schedule} & \textbf{Initial LR} & \textbf{Reward} & \textbf{$\Delta$} \\
\midrule
\textbf{Exponential decay} & \textbf{$1e-03 \to 1e-04$} & $\mathbf{0.52}$ & $\mathbf{+18.2\%}$ \\
Linear decay & $1e-03 \to 1e-04$ & $0.50$ & $+13.6\%$ \\
Constant ($1e-03$) & $1e-03$ & $0.47$ & $+6.8\%$ \\
\bottomrule
\end{tabular}
\end{table}
```

### 4. Hyperparameter Grid Search Table

Shows results from grid search over hyperparameters.

**Example:**

```python
table = generator.generate_hyperparameter_grid_table(
    runs=runs,
    param_name="diversity_range",
    param_display_name="Diversity Range",
    caption="Grid search: diversity reward range optimization.",
    label="tab:diversity_grid",
    additional_columns={"Unique Clusters": "metadata.unique_clusters"},
)
```

## Helper Functions

### format_mean_std

Format mean ± standard deviation for LaTeX.

```python
LatexTableGenerator.format_mean_std(0.52, 0.03, decimals=2)
# Returns: "$0.52 \pm 0.03$"
```

### format_delta

Format relative change as percentage.

```python
LatexTableGenerator.format_delta(0.45, 0.52, decimals=1)
# Returns: "$+15.6\%$"

LatexTableGenerator.format_delta(0.52, 0.48, decimals=1)
# Returns: "$-7.7\%$"
```

### bold_best_values

Format values with the best value in bold.

```python
LatexTableGenerator.bold_best_values([0.45, 0.52, 0.48])
# Returns: ['$0.45$', '$\mathbf{0.52}$', '$0.48$']
```

### escape_latex

Escape special LaTeX characters.

```python
LatexTableGenerator.escape_latex("Train_accuracy: 95%")
# Returns: "Train\\_accuracy: 95\\%"
```

## AblationAnalyzer

The `AblationAnalyzer` class provides tools for analyzing ablation studies.

### Group Runs by Dimension

```python
analyzer = AblationAnalyzer(significance_threshold=0.05)

group = analyzer.group_by_dimension(
    runs=runs,
    dimension="training.learning_rate_scheduler.type",
    baseline_value="exponential",
)

print(f"Baseline: {group.baseline_run.name}")
print(f"Variants: {len(group.variant_runs)}")
```

### Compute Relative Changes

```python
changes = analyzer.compute_relative_changes(
    group,
    metric="final_mean_reward",
)

for run_name, change in changes.items():
    print(f"{run_name}: {change:+.2f}%")
```

### Test Statistical Significance

```python
p_value, is_significant = analyzer.test_significance(
    run1=runs[0],
    run2=runs[1],
    use_final_phase=True,  # Use last 20% of episodes
    test='mannwhitneyu',    # or 't-test'
)

print(f"p-value: {p_value:.4f}, significant: {is_significant}")
```

### Aggregate Across Seeds

```python
aggregated = analyzer.aggregate_across_seeds(
    runs=runs,
    group_by='config_hash',  # or any config path
)

for group_name, (mean, std) in aggregated.items():
    print(f"{group_name}: {mean:.3f} ± {std:.3f}")
```

### Identify Best Configuration

```python
best_run = analyzer.identify_best_configuration(
    runs=runs,
    metric='final_mean_reward',
    prefer_stable=True,  # Penalize unstable runs
)

print(f"Best: {best_run.name}")
print(f"Reward: {best_run.final_mean_reward:.3f}")
```

## Integration with Training Analysis

The module is designed to work seamlessly with the existing training analysis pipeline:

```python
from analyze_training_runs import (
    load_run_data,
    compute_statistics,
    AnalysisConfig,
)
from latex_tables import LatexTableGenerator

# Load runs
input_dir = Path("artifacts/training")
runs = []

for run_dir in input_dir.glob("run_*"):
    run_data = load_run_data(run_dir)
    if run_data:
        config = AnalysisConfig(input_dir=input_dir, output_dir=Path("outputs"))
        run_data = compute_statistics(run_data, config)
        runs.append(run_data)

# Generate tables
generator = LatexTableGenerator()
table = generator.generate_training_comparison_table(runs, baseline_name="baseline")

# Save to file
output_path = Path("outputs/latex_tables/training_comparison.tex")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(table)
```

## Configuration

The `LatexTableGenerator` can be customized:

```python
generator = LatexTableGenerator(
    decimals=2,           # Number of decimal places
    use_booktabs=True,    # Use booktabs package (\toprule, \midrule, \bottomrule)
    bold_best=True,       # Automatically bold best values
    show_delta=True,      # Show relative change columns in ablation tables
)
```

## Expected Data Structure

The module expects `RunMetrics` objects with the following attributes:

```python
@dataclass
class RunMetrics:
    name: str                          # Run identifier
    final_mean_reward: float           # Mean reward in final phase
    final_std_reward: float            # Std reward in final phase
    episode_rewards: np.ndarray        # All episode rewards
    config: Dict[str, Any]             # Configuration dict
    metadata: Dict[str, Any]           # Additional metadata

    # Optional attributes
    stability_score: float = 0.8
    convergence_episode: int = None
    trend_slope: float = 0.0
```

## Paper Integration

To use these tables in your NeurIPS paper:

1. Generate tables using the module
2. Copy the LaTeX code into your paper
3. Ensure your paper includes the `booktabs` package:

```latex
\usepackage{booktabs}
```

4. Reference tables using their labels:

```latex
Table~\ref{tab:training_results} shows that our method...
```

## Tips for NeurIPS Tables

1. **Keep tables concise**: Focus on key metrics only
2. **Use consistent formatting**: All tables should use the same style
3. **Bold best values**: Makes it easy to spot winners
4. **Include error bars**: Mean ± std demonstrates robustness
5. **Show relative changes**: Percentage deltas are more interpretable
6. **Statistical significance**: Report p-values when comparing methods
7. **Limit decimal places**: 2-3 decimals is usually sufficient

## Troubleshooting

### Table too wide

Reduce the number of columns or use smaller font:

```latex
\begin{table}[t]
\centering
\small  % Add this line
\caption{...}
...
```

### Missing data

The module handles missing data gracefully by inserting `[TBD]` placeholders.

### Custom formatting

Extend the `LatexTableGenerator` class for custom table types:

```python
class MyTableGenerator(LatexTableGenerator):
    def generate_custom_table(self, runs, **kwargs):
        # Your custom logic here
        pass
```

## Contributing

To add new table types:

1. Add a new method to `LatexTableGenerator`
2. Follow the naming convention: `generate_<type>_table()`
3. Accept `caption` and `label` parameters
4. Return complete LaTeX table as string
5. Add example to `example_latex_tables.py`
6. Update this README

## License

This module is part of the ARIA (Adaptive Reinforcement-learning for Interactive Arts) project.

## Contact

For questions or issues, please refer to the main project documentation.
