# Feature Importance Consensus Analysis Report

**Generated:** 2025-11-13 22:02:03

## Executive Summary

This report synthesizes feature importance analysis from multiple methods:
- **RFE (Recursive Feature Elimination)** with GradientBoosting (target: `muspy_pitch_entropy`)
- **SHAP (SHapley Additive exPlanations)** with Ridge Regression (target: `pm_energy`)
- **VIF (Variance Inflation Factor)** for multicollinearity detection
- **Correlation Analysis** for feature redundancy
- **Mutual Information** for feature interactions

**Key Finding:** Different methods target different aspects of the data, leading to some disagreements.
Features that rank highly across multiple methods are the most robust predictors.

---

## 1. Consensus Ranking

### Weighting Scheme
- **RFE Ranking:** 35.0%
- **SHAP Importance:** 35.0%
- **VIF Penalty:** 15.0%
- **Correlation Penalty:** 10.0%
- **Mutual Information:** 5.0%

### Top 15 Features by Consensus

| Rank | Feature | Consensus Score | RFE Rank | SHAP Rank | VIF Score | Corr Score | MI Score |
|------|---------|-----------------|----------|-----------|-----------|------------|----------|
| 1 | `pm_note_count` | 76.9 | 1 | 1 | 0.0 | 28.1 | 82.0 |
| 2 | `muspy_polyphony` | 63.9 | 2 | 5 | 85.5 | 37.2 | 38.3 |
| 3 | `muspy_n_pitches_used` | 63.7 | 1 | 3 | 46.0 | 52.7 | 78.9 |
| 4 | `pm_length_seconds` | 63.0 | 4 | 18 | 100.0 | 100.0 | 99.9 |
| 5 | `pm_average_polyphony` | 58.9 | 13 | 2 | 98.5 | 100.0 | 65.9 |
| 6 | `pm_interval_range_min` | 56.9 | 3 | 19 | 100.0 | 49.5 | 53.7 |
| 7 | `pm_average_velocity` | 55.9 | 10 | 8 | 100.0 | 100.0 | 60.3 |
| 8 | `muspy_n_pitch_classes_used` | 55.7 | 1 | 4 | 12.6 | 34.4 | 65.5 |
| 9 | `pm_tempo_bpm` | 49.2 | 11 | 29 | 100.0 | 100.0 | 81.0 |
| 10 | `muspy_pitch_class_entropy` | 48.8 | 1 | 9 | 0.0 | 34.4 | 86.5 |
| 11 | `pm_note_density` | 48.5 | 5 | 14 | 47.6 | 28.1 | 96.1 |
| 12 | `pm_groove` | 48.3 | 15 | 10 | 100.0 | 100.0 | 74.2 |
| 13 | `muspy_groove_consistency` | 47.7 | 12 | 27 | 100.0 | 100.0 | 81.5 |
| 14 | `pm_interval_range_max` | 43.4 | 14 | 21 | 100.0 | 89.1 | 50.1 |
| 15 | `muspy_scale_consistency` | 39.0 | 16 | 24 | 100.0 | 100.0 | 16.0 |


---

## 2. Method Agreement Analysis

### Rank Correlations (Spearman's ρ)

- **RFE_vs_SHAP:** ρ = 0.483 (p = 0.0125)
- **RFE_vs_Consensus:** ρ = 0.682 (p = 0.0001)
- **SHAP_vs_Consensus:** ρ = 0.594 (p = 0.0014)


### Top-10 Overlap (Jaccard Similarity)

- **RFE ∩ SHAP:** 0.333 (5/10 shared features)
- **RFE ∩ Consensus:** 0.538 (7/10 shared)
- **SHAP ∩ Consensus:** 0.538 (7/10 shared)
- **All Three Methods:** 5 features in top-10 of all methods

### Features with High Agreement

These features rank consistently across methods (low rank variance):

- `pm_note_count` (rank variance: 0.0)
- `muspy_polyphony` (rank variance: 2.0)
- `muspy_n_pitches_used` (rank variance: 0.9)
- `pm_average_velocity` (rank variance: 1.6)
- `muspy_n_pitch_classes_used` (rank variance: 8.2)
- `pm_groove` (rank variance: 4.2)
- `pm_bar_count` (rank variance: 2.0)
- `theory_limited_macroharmony` (rank variance: 0.9)


### Features with High Disagreement

These features have large ranking differences between methods:

- `pm_interval_range_min` (variance: 48.2, RFE: 3.0, SHAP: 19.0)
- `pm_tempo_bpm` (variance: 80.9, RFE: 11.0, SHAP: 29.0)
- `muspy_groove_consistency` (variance: 46.9, RFE: 12.0, SHAP: 27.0)
- `muspy_pitch_range` (variance: 86.0, RFE: 1.0, SHAP: 22.0)
- `pm_average_pitch_hz` (variance: 96.9, RFE: 6.0, SHAP: 30.0)
- `muspy_pitch_entropy` (variance: 88.9, RFE: nan, SHAP: 6.0)
- `theory_conjuction_melody_motion` (variance: 107.6, RFE: nan, SHAP: 7.0)
- `theory_general_tonality` (variance: 64.2, RFE: nan, SHAP: 15.0)


**Note:** Disagreements often arise because RFE optimizes for `muspy_pitch_entropy` while SHAP analyzes `pm_energy`.
Features important for one target may not be important for the other.

---

## 3. Data Quality Assessment

### Features to Exclude (VIF > 100)

**⚠ WARNING:** The following features have severe multicollinearity issues:

- `muspy_pitch_range` (VIF: inf)
- `pm_pitch_range_max_note` (VIF: inf)
- `muspy_empty_measure_rate` (VIF: inf)
- `muspy_pitch_entropy` (VIF: 108.0)
- `theory_limited_macroharmony` (VIF: 138.6)
- `muspy_empty_beat_rate` (VIF: inf)
- `theory_conjuction_melody_motion` (VIF: 11992.7)
- `pm_instrument_count` (VIF: 2935688.9)
- `pm_pitch_range_min_note` (VIF: inf)
- `theory_general_tonality` (VIF: 10964.2)

**Recommendation:** Exclude these features from final model.


### High Correlation in Top 15

**⚠ WARNING:** High correlation (>0.9) detected between top-ranked features:

- `pm_note_count` ↔ `pm_note_density`: r = 0.944
- `muspy_n_pitch_classes_used` ↔ `muspy_pitch_class_entropy`: r = 0.931

**Recommendation:** Consider keeping only one feature from each highly correlated pair.


---

## 4. Feature Groups and Clusters

Hierarchical clustering identified **20 feature clusters** based on correlation and mutual information.

### Cluster Representatives (Recommended Features)

For each cluster, the feature with the highest consensus score is recommended:

- **Cluster 4** (size: 2): Keep `pm_pitch_range_min_freq`, drop pm_pitch_range_min_note
- **Cluster 5** (size: 3): Keep `pm_average_pitch_hz`, drop pm_pitch_range_max_note, pm_pitch_range_max_freq
- **Cluster 6** (size: 6): Keep `muspy_polyphony`, drop pm_interval_range_min, pm_interval_range_max, muspy_pitch_range, theory_conjuction_melody_motion, theory_general_tonality
- **Cluster 10** (size: 5): Keep `muspy_n_pitches_used`, drop muspy_n_pitch_classes_used, muspy_pitch_class_entropy, theory_centricity, muspy_pitch_entropy
- **Cluster 12** (size: 2): Keep `muspy_empty_measure_rate`, drop muspy_empty_beat_rate
- **Cluster 14** (size: 3): Keep `pm_note_count`, drop pm_note_density, pm_energy


**Rationale:** When features are highly correlated or have high mutual information, keeping the top-ranked
feature from each cluster reduces redundancy while preserving predictive power.

---

## 5. Statistical Significance

### RFE Feature Selection
- **Selected by RFE:** 5 features
- RFE uses cross-validated recursive elimination, so selected features are significant for predicting `muspy_pitch_entropy`

### SHAP Confidence Intervals
- **Significant SHAP values (95% CI > 0):** 30 features
- Bootstrap resampling (n=1000) used to compute confidence intervals

### Overall Significance
- **Both RFE and SHAP agree:** 5 features
- These are the most statistically robust features

---

## 6. Final Recommendations

### Recommended Feature Set (Top 15 by Consensus)

Based on consensus ranking, data quality, and statistical significance:

1. `pm_note_count` (score: 76.9, RFE: ✓, SHAP: ✓) ⚠ (HighCorr(pm_note_density))
2. `muspy_polyphony` (score: 63.9, RFE: ✗, SHAP: ✓)
3. `muspy_n_pitches_used` (score: 63.7, RFE: ✓, SHAP: ✓)
4. `pm_length_seconds` (score: 63.0, RFE: ✗, SHAP: ✓)
5. `pm_average_polyphony` (score: 58.9, RFE: ✗, SHAP: ✓)
6. `pm_interval_range_min` (score: 56.9, RFE: ✗, SHAP: ✓)
7. `pm_average_velocity` (score: 55.9, RFE: ✗, SHAP: ✓)
8. `muspy_n_pitch_classes_used` (score: 55.7, RFE: ✓, SHAP: ✓) ⚠ (HighCorr(muspy_pitch_class_entropy))
9. `pm_tempo_bpm` (score: 49.2, RFE: ✗, SHAP: ✓)
10. `muspy_pitch_class_entropy` (score: 48.8, RFE: ✓, SHAP: ✓) ⚠ (HighCorr(muspy_n_pitch_classes_used))
11. `pm_note_density` (score: 48.5, RFE: ✗, SHAP: ✗) ⚠ (HighCorr(pm_note_count))
12. `pm_groove` (score: 48.3, RFE: ✗, SHAP: ✓)
13. `muspy_groove_consistency` (score: 47.7, RFE: ✗, SHAP: ✓)
14. `pm_interval_range_max` (score: 43.4, RFE: ✗, SHAP: ✓)
15. `muspy_scale_consistency` (score: 39.0, RFE: ✗, SHAP: ✓)


**Legend:**
- ✓ = Selected/significant by method
- ✗ = Not selected/significant
- ⚠ = Data quality warning

### Features to Consider Excluding

1. **Features with VIF > 100:** 10 features (perfect multicollinearity)
2. **Features in highly correlated pairs (r > 0.9):** Consider keeping only one from each pair
3. **Features with >50% missing values:** Review necessity vs. data availability

### Usage Guidelines

1. **For predictive modeling targeting `muspy_pitch_entropy`:** Prioritize RFE-selected features
2. **For predictive modeling targeting `pm_energy`:** Prioritize SHAP-significant features
3. **For general feature selection:** Use consensus ranking
4. **For dimensionality reduction:** Use cluster representatives (one per cluster)
5. **For interpretability:** Focus on features significant in both RFE and SHAP

---

## 7. Limitations and Caveats

1. **Different Targets:** RFE was optimized for `muspy_pitch_entropy` while SHAP analyzed `pm_energy`.
   This explains some ranking disagreements.

2. **Missing Values:** Theory features have ~50% missing values, which may affect their rankings.

3. **Multicollinearity:** Several features show perfect or near-perfect multicollinearity (infinite VIF).
   These should be excluded to avoid model instability.

4. **Linear Assumptions:** VIF and correlation assume linear relationships. Non-linear dependencies
   may not be fully captured.

5. **Model-Specific Importance:** SHAP values are specific to Ridge Regression. Importance may differ
   with other model types.

---

## 8. Files Generated

### Data Files
- `consensus_ranking.csv` - Unified ranking with all scores
- `method_comparison.csv` - Comparison metrics between methods
- `feature_groups.csv` - Hierarchical clustering results
- `cluster_recommendations.csv` - Which features to keep/drop per cluster
- `statistical_significance.csv` - Significance tests for all features

### Visualizations
- `consensus_ranking_comparison.png` - Visual comparison of rankings
- `method_agreement_heatmap.png` - Rank correlation heatmap
- `feature_dependency_network.png` - Network graph of feature dependencies

### Reports
- `consensus_summary.json` - Complete results in JSON format
- `COMPARISON_REPORT.md` - This human-readable report

---

## Conclusion

The consensus analysis identifies **pm_note_count** as the most important
feature overall, followed by **muspy_polyphony** and
**muspy_n_pitches_used**.

Features that appear in the top 10 of both RFE and SHAP (5 features) are
the most robust and should be prioritized for downstream analysis.

Consider the specific modeling task, data quality constraints, and interpretability requirements when
making final feature selection decisions.
