# LaTeX Tables Quick Start Guide

## TL;DR

```bash
# Generate all paper tables in one command
python generate_paper_tables.py --input-dir artifacts/training --output-dir outputs/paper_tables

# View examples with mock data
python example_latex_tables.py --mode mock

# Test with a single table type
python -c "from latex_tables import LatexTableGenerator; g = LatexTableGenerator(); print(g.format_mean_std(0.52, 0.03))"
```

## Three Files, Three Purposes

| File | Purpose | When to Use |
|------|---------|-------------|
| `latex_tables.py` | Core module with generator classes | Import into your own scripts |
| `example_latex_tables.py` | Comprehensive examples | Learning and testing |
| `generate_paper_tables.py` | Batch generate all paper tables | Final table generation |

## Common Use Cases

### 1. Generate All Paper Tables

```bash
python generate_paper_tables.py \
    --input-dir artifacts/training \
    --output-dir outputs/paper_tables \
    --decimals 2
```

**Output:** All 8 tables as `.tex` files in `outputs/paper_tables/`

### 2. Generate Single Table Type

```python
from latex_tables import LatexTableGenerator
from analyze_training_runs import load_run_data, compute_statistics, AnalysisConfig

# Load runs
runs = [...]  # Your RunMetrics objects

# Generate table
generator = LatexTableGenerator()
table = generator.generate_training_comparison_table(
    runs=runs,
    baseline_name="baseline",
)

# Save
Path("table.tex").write_text(table)
```

### 3. Custom Ablation Analysis

```python
from latex_tables import AblationAnalyzer

analyzer = AblationAnalyzer()

# Group by dimension
group = analyzer.group_by_dimension(
    runs=runs,
    dimension="training.learning_rate",
)

# Get changes
changes = analyzer.compute_relative_changes(group)
for name, change in changes.items():
    print(f"{name}: {change:+.1f}%")
```

## Table Types Reference

| Table | Method | Description |
|-------|--------|-------------|
| Training Comparison | `generate_training_comparison_table()` | Compare methods on reward/time/efficiency |
| Ablation Study | `generate_ablation_table()` | Component removal impact |
| LR Schedule | `generate_lr_schedule_table()` | Learning rate comparison |
| Grid Search | `generate_hyperparameter_grid_table()` | Hyperparameter sweep results |

## Helper Functions Cheat Sheet

```python
from latex_tables import LatexTableGenerator as LTG

# Format mean ± std
LTG.format_mean_std(0.52, 0.03)
# → "$0.52 \pm 0.03$"

# Format percentage change
LTG.format_delta(0.45, 0.52)
# → "$+15.6\%$"

# Bold best value
LTG.bold_best_values([0.45, 0.52, 0.48])
# → ['$0.45$', '$\mathbf{0.52}$', '$0.48$']

# Escape LaTeX characters
LTG.escape_latex("Train_accuracy: 95%")
# → "Train\\_accuracy: 95\\%"
```

## Expected Run Naming Conventions

For automatic filtering in `generate_paper_tables.py`, use these patterns:

| Study | Patterns |
|-------|----------|
| Baseline comparison | `end-to-end`, `single-agent`, `vanilla-drqn`, `rainbow-drqn` |
| Coordination | `prototype`, `centroid`, `random-action`, `one-hot` |
| Curriculum | `no-curriculum`, `2-stage`, `3-stage` |
| Rainbow | `full-rainbow`, `no-c51`, `no-prioritized`, `no-dueling` |
| Features | `full-features`, `no-prettymidi`, `no-muspy` |
| Rewards | `full-reward`, `no-structure`, `no-transition`, `no-diversity` |
| LR Schedule | `constant`, `exponential`, `linear` |
| Diversity | `diversity` (with metadata) |

## Integration with Paper

1. **Generate tables:**
   ```bash
   python generate_paper_tables.py
   ```

2. **In your LaTeX preamble:**
   ```latex
   \usepackage{booktabs}
   ```

3. **Include tables:**
   ```latex
   \input{outputs/paper_tables/tab1_training_comparison.tex}
   ```

4. **Or copy-paste** the LaTeX code directly

## Troubleshooting

### "No runs found"
- Check `--input-dir` points to correct location
- Ensure directories start with `run_`
- Verify `metrics/detailed_rewards.json` exists

### "No tables generated"
- Check run names match expected patterns
- Use `--verbose` flag (if added) to see filtering
- Manually specify runs if needed

### Tables too wide
Add `\small` or `\footnotesize` before `\begin{tabular}`:
```latex
\begin{table}[t]
\centering
\small  % <-- Add this
\caption{...}
```

### Missing values ([TBD])
- Ensure runs have required metadata
- Check `training_time_hours` in metadata
- Verify config has required fields

## Performance Tips

- Load runs once, generate multiple tables
- Filter runs before generating tables
- Use `decimals=1` for compact tables
- Set `bold_best=False` for neutral formatting

## Example Workflow

```python
# 1. Load all runs
from pathlib import Path
from analyze_training_runs import load_run_data, compute_statistics, AnalysisConfig
from latex_tables import LatexTableGenerator

input_dir = Path("artifacts/training")
config = AnalysisConfig(input_dir=input_dir, output_dir=Path("outputs"))

runs = []
for run_dir in input_dir.glob("run_*"):
    run = load_run_data(run_dir)
    if run:
        run = compute_statistics(run, config)
        runs.append(run)

# 2. Filter for specific study
rainbow_runs = [r for r in runs if 'rainbow' in r.name.lower()]

# 3. Generate table
generator = LatexTableGenerator()
table = generator.generate_ablation_table(
    runs=rainbow_runs,
    full_method_name="full-rainbow",
    components=["c51", "prioritized", "dueling"],
)

# 4. Save
output_file = Path("outputs/rainbow_ablation.tex")
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(table)

print(f"Saved to {output_file}")
```

## Next Steps

- Read full documentation: `README_LATEX_TABLES.md`
- Study examples: `python example_latex_tables.py --mode mock`
- Generate paper tables: `python generate_paper_tables.py`
- Customize for your needs by extending classes

## Support

For issues or questions:
1. Check the full README
2. Review example scripts
3. Inspect the LaTeX output
4. Verify run data structure

---

**Quick tip:** Start with `example_latex_tables.py --mode mock` to see all capabilities without needing real training data!
