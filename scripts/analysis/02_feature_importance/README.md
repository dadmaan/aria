# Feature Importance Analysis - COMMU Bass Dataset

**Experiment ID:** feature_analysis_20251113_214037
**Date:** November 13, 2025
**Dataset:** COMMU Bass (532 samples, 35 features)

## Overview

This experiment performs comprehensive feature importance analysis using multiple complementary methods:
- **SHAP (SHapley Additive exPlanations):** Global and local interpretability
- **RFE (Recursive Feature Elimination):** Optimal feature subset selection
- **Correlation Analysis:** Linear dependencies and redundancy
- **VIF Analysis:** Multicollinearity assessment
- **Mutual Information:** Non-linear relationships

## Quick Start

### Run the reproducible script:
```bash
cd experiments_feature_analysis/feature_analysis_20251113_214037
python scripts/run_feature_analysis.py \
    --data_path ../../artifacts/features/raw/commu_bass/features_numeric.csv \
    --output_dir .
```

### Requirements:
```bash
pip install pandas numpy scipy scikit-learn shap statsmodels matplotlib seaborn
```

## Results Summary

### Top 10 Features (Consensus Ranking)

1. **pm_note_count** (76.9) - Total number of notes
2. **muspy_polyphony** (63.9) - Average simultaneous notes ⚠️ VIF issue
3. **muspy_n_pitches_used** (63.7) - Number of unique pitches
4. **pm_length_seconds** (63.0) - Track duration
5. **pm_average_polyphony** (58.9) - Average polyphony
6. **pm_interval_range_min** (56.9) - Minimum melodic interval
7. **pm_average_velocity** (55.9) - Mean note velocity
8. **muspy_n_pitch_classes_used** (55.7) - Unique pitch classes
9. **pm_tempo_bpm** (49.2) - Tempo in BPM
10. **muspy_pitch_class_entropy** (48.8) - Pitch class entropy

### Key Findings

- **Feature Reduction:** From 35 to 10 features (71% reduction) with maintained performance
- **Dominant Feature:** `pm_note_count` is 2.5× more important than the second-ranked feature
- **Data Quality Issues:** 10 features with VIF > 100 (severe multicollinearity)
- **Method Agreement:** Moderate RFE-SHAP correlation (ρ = 0.483) due to different targets

## Directory Structure

```
feature_analysis_20251113_214037/
├── README.md                          # This file
├── ANALYSIS.md                        # Comprehensive analysis report
├── data/                              # Analysis results (CSV/NPY)
│   ├── correlation_matrix.csv
│   ├── vif_scores.csv
│   ├── mutual_information.csv
│   ├── rfe_rankings.csv
│   ├── shap_importance.csv
│   ├── consensus_ranking.csv
│   └── ...
├── plots/                             # Visualizations (PNG)
│   ├── correlation_heatmap.png
│   ├── rfe_cv_scores.png
│   ├── shap_summary_beeswarm.png
│   ├── consensus_ranking_comparison.png
│   └── ...
├── reports/                           # Detailed reports (MD/JSON)
│   ├── eda_summary.json
│   ├── EDA_FINAL_REPORT.md
│   ├── rfe_summary.json
│   ├── RFE_ANALYSIS_REPORT.md
│   ├── shap_summary.json
│   ├── SHAP_ANALYSIS_SUMMARY.md
│   ├── consensus_summary.json
│   └── COMPARISON_REPORT.md
└── scripts/                           # Reproducible code
    └── run_feature_analysis.py
```

## Files Generated

### Data Files (22 files)
- Correlation matrices, VIF scores, mutual information
- RFE rankings and elimination order
- SHAP values and importance scores
- Consensus rankings and feature groups
- Statistical significance tests

### Visualizations (11 files)
- Distribution plots, correlation heatmaps
- RFE cross-validation curves
- SHAP summary plots (beeswarm, bar, dependence, waterfall, force)
- Consensus comparison plots
- Feature dependency networks

### Reports (8 files)
- Machine-readable JSON summaries
- Human-readable Markdown reports
- Complete methodology and findings documentation

## Recommendations

### For Predictive Modeling:
Use these **10 features** (Tier 1 + Tier 2):
- pm_note_count, muspy_n_pitches_used, pm_length_seconds
- pm_average_polyphony, pm_average_velocity
- pm_tempo_bpm, pm_groove, pm_interval_range_max
- muspy_scale_consistency, muspy_polyphony

### For Maximum Parsimony:
Use these **5 features** (RFE-selected):
- pm_note_count, muspy_n_pitches_used
- muspy_n_pitch_classes_used, muspy_pitch_class_entropy
- muspy_pitch_range

### Features to Exclude:
- 10 features with VIF > 100 (severe multicollinearity)
- 3 drum features (100% missing values)
- Highly redundant features (r > 0.9 with selected features)

## Reproducibility

All analyses use `random_state=42` for reproducibility. The provided script can regenerate all results from the raw data.

## Contact

For questions or issues, see the main repository documentation.
