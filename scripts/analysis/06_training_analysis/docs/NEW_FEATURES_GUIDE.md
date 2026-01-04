# New Features Quick Reference Guide

## 1. Recursive Run Discovery

**Use Case:** When your training runs are organized in subdirectories

**Example Structure:**
```
artifacts/training/
├── phase1/
│   ├── run_baseline/
│   └── run_variant1/
├── phase2/
│   ├── run_baseline/
│   └── run_variant2/
└── seeds/
    ├── run_exp_seed0/
    ├── run_exp_seed1/
    └── run_exp_seed2/
```

**Usage:**
```bash
python analyze_training_runs.py --recursive --input-dir artifacts/training
```

**What it does:**
- Discovers all `run_*` directories recursively
- Handles multiple benchmark layouts automatically
- No need to flatten directory structure

---

## 2. Run Grouping

**Use Case:** Analyze and compare runs organized by different dimensions

**Available Grouping Options:**
- `phase` - Group by training phase (e.g., phase1, phase2)
- `method` - Group by algorithm/method variant
- `seed` - Group by random seed
- `variant` - Group by experiment variant

**Usage:**
```bash
# Group by phase
python analyze_training_runs.py --group-by phase

# Group by method
python analyze_training_runs.py --group-by method

# Group by seed
python analyze_training_runs.py --group-by seed
```

**Output Structure:**
```
outputs/training_analysis/analysis_TIMESTAMP/
├── summary_report.md              # Overall summary
├── grouped_reports/
│   ├── phase1/
│   │   ├── summary.md            # Phase 1 specific summary
│   │   └── plots/                # Phase 1 visualizations
│   ├── phase2/
│   │   ├── summary.md
│   │   └── plots/
│   └── ...
└── plots/                         # Overall visualizations
```

**What it does:**
- Creates separate analysis for each group
- Generates group-specific visualizations
- Enables within-group and between-group comparisons

---

## 3. Seed Aggregation

**Use Case:** When you have multiple seeds for the same configuration and want aggregate statistics

**Naming Convention:**
Your runs should follow this pattern:
- `run_experiment_seed0`
- `run_experiment_seed1`
- `run_experiment_seed2`

**Usage:**
```bash
python analyze_training_runs.py --aggregate-seeds
```

**What it does:**
- Groups runs by base name (removes `_seedX` suffix)
- Computes mean ± std across all seeds
- Creates aggregated RunMetrics with:
  - Mean episode rewards across seeds
  - Aggregated statistics (mean of means, etc.)
  - Representative configuration from first seed

**Example Output:**
```
Input:  run_dqn_seed0, run_dqn_seed1, run_dqn_seed2
Output: run_dqn_aggregated (with mean ± std statistics)
```

---

## 4. Multiple Output Formats

**Use Case:** Generate reports in different formats for various audiences

**Available Formats:**
- `markdown` (default) - Human-readable reports
- `latex` - Tables for academic papers
- `json` - Programmatic access to results
- `all` - Generate all formats

**Usage:**
```bash
# LaTeX tables for paper
python analyze_training_runs.py --output-format latex

# JSON for further processing
python analyze_training_runs.py --output-format json

# Everything
python analyze_training_runs.py --output-format all
```

**Output Files:**

**Markdown:**
- `summary_report.md` - Comprehensive analysis report
- `config_comparison.md` - Configuration differences table

**JSON:**
- `data/analysis_results.json` - All metrics and statistics

**LaTeX:**
- `performance_table.tex` - Performance comparison table
- `ablation_table.tex` - Ablation study results (with baseline)

**What it does:**
- Generates publication-ready LaTeX tables
- Provides structured JSON for automation
- Maintains readable markdown reports

---

## 5. Baseline Comparison & Ablation Analysis

**Use Case:** Compare experimental runs against a baseline configuration

**Requirements:**
- Must specify `--output-format latex` or `--output-format all`
- Baseline run must exist in the analyzed runs

**Usage:**
```bash
python analyze_training_runs.py \
    --output-format latex \
    --baseline run_baseline
```

**What it does:**
- Identifies the baseline run by name pattern
- Compares all other runs to baseline
- Generates ablation_table.tex with:
  - Relative performance improvements
  - Statistical significance tests
  - Effect sizes

**Example Ablation Table:**
```latex
\begin{tabular}{lrrr}
\toprule
Variant & Δ Reward & p-value & Effect \\
\midrule
baseline & 0.00 & --- & --- \\
+feature_A & +5.2 & 0.001 & Large \\
+feature_B & +2.1 & 0.043 & Medium \\
full_model & +8.3 & <0.001 & Large \\
\bottomrule
\end{tabular}
```

---

## 6. Combined Usage Examples

### Example 1: Full Paper Analysis
```bash
python analyze_training_runs.py \
    --input-dir experiments/final_runs \
    --recursive \
    --aggregate-seeds \
    --group-by phase \
    --output-format all \
    --baseline run_baseline
```

**Produces:**
- Aggregated results across seeds
- Phase-specific analyses
- LaTeX tables for paper
- JSON data for plots
- Markdown reports for sharing

### Example 2: Quick Comparison
```bash
python analyze_training_runs.py \
    --runs run_A,run_B,run_C \
    --output-format markdown
```

**Produces:**
- Fast analysis of specific runs only
- Markdown report with visualizations

### Example 3: Method Comparison
```bash
python analyze_training_runs.py \
    --recursive \
    --group-by method \
    --output-format latex
```

**Produces:**
- Method-grouped analyses
- LaTeX tables comparing methods
- Method-specific visualizations

---

## Tips & Best Practices

### 1. Naming Conventions
Use consistent naming for automatic grouping:
- **Seeds:** `run_name_seed0`, `run_name_seed1`, ...
- **Phases:** `run_phase1_name`, `run_phase2_name`, ...
- **Methods:** Include method in config or name
- **Variants:** Use clear variant identifiers

### 2. Directory Organization
Organize runs logically for easier analysis:
```
artifacts/training/
├── ablation_study/
│   ├── baseline/
│   ├── +feature_A/
│   └── +feature_B/
├── hyperparameter_search/
│   ├── lr_0.001/
│   └── lr_0.0001/
└── final_runs/
    └── best_config/
```

### 3. Output Format Selection
- **During development:** Use `markdown` for quick reviews
- **For papers:** Use `latex` with baseline comparison
- **For automation:** Use `json` for downstream processing
- **For comprehensive reports:** Use `all`

### 4. Performance Considerations
- Use `--runs` to analyze specific runs during development
- Use `--recursive` only when needed (faster without)
- Aggregate seeds after confirming individual seeds look correct

### 5. Grouping Strategy
Choose grouping dimension based on analysis goal:
- **Phase-wise comparison:** `--group-by phase`
- **Algorithm comparison:** `--group-by method`
- **Seed variability:** `--group-by seed`
- **Ablation components:** `--group-by variant`

---

## Troubleshooting

### Issue: "No runs found"
**Solution:**
- Check directory structure matches expected format
- Try `--recursive` if runs are in subdirectories
- Verify runs start with `run_` prefix

### Issue: "Baseline not found"
**Solution:**
- Ensure baseline run name matches exactly
- Baseline should be part of analyzed runs
- Check for typos in `--baseline` argument

### Issue: "Seed aggregation produces empty results"
**Solution:**
- Verify runs follow `_seedX` naming convention
- Check that runs with same base name actually exist
- Ensure at least one run per configuration

### Issue: "LaTeX table not generated"
**Solution:**
- Must use `--output-format latex` or `--output-format all`
- Check that runs have required metrics
- Verify latex_tables.py module is present
