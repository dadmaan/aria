# SHAP Feature Importance Analysis - Comprehensive Summary

**Analysis Date:** 2025-11-13
**Dataset:** `artifacts/features/raw/commu_bass/features_numeric.csv`
**Output Directory:** `experiments_feature_analysis/feature_analysis_20251113_214037/`

---

## Executive Summary

This report presents a comprehensive SHAP (SHapley Additive exPlanations) analysis for feature importance and interpretability on the musical feature dataset. SHAP values provide a unified measure of feature importance based on game theory, offering both global feature importance rankings and local explanations for individual predictions.

**Key Findings:**
- **Best Model:** Ridge Regression (R² = 0.7414, RMSE = 9.04)
- **SHAP Explainer:** LinearExplainer (optimized for linear models)
- **Top Feature:** `pm_note_count` (mean |SHAP| = 22.06)
- **Feature Distribution:** Muspy features dominate (7/15 in top 15)
- **Target Variable:** `pm_energy` (musical energy level)

---

## 1. Model Selection and SHAP Explainer Type

### Models Evaluated

| Model | CV RMSE | Test R² | Test RMSE | Test MAE | Selected |
|-------|---------|---------|-----------|----------|----------|
| **Ridge** | 16.92 ± 7.39 | **0.7414** | **9.04** | 7.16 | ✓ |
| Lasso | 18.66 ± 10.43 | 0.7253 | 9.32 | 6.20 | |
| RandomForest | 30.08 ± 29.94 | 0.6777 | 10.09 | 3.22 | |
| GradientBoosting | 27.13 ± 26.91 | -0.3372 | 20.56 | 3.34 | |

**Selected Model:** Ridge Regression
- Best test R² score (0.7414)
- Good generalization (low CV standard deviation)
- Excellent balance between bias and variance
- Interpretable linear model

### SHAP Explainer Type

**LinearExplainer** was used for Ridge regression:
- **Advantage:** Exact SHAP values for linear models (no approximation)
- **Speed:** Fast computation on all test samples
- **Accuracy:** No sampling required, mathematically exact
- **Expected Value:** 18.38 (base prediction without any features)

**Note:** Feature interactions were not computed because they are only available for tree-based models (RandomForest, GradientBoosting, XGBoost).

---

## 2. Data Preprocessing Approach

### Data Loading
- **Initial samples:** 532
- **Initial features:** 37
- **Training samples:** 425 (80%)
- **Test samples:** 107 (20%)

### Feature Removal
**Excluded:** `track_id`, `metadata_index` (metadata, not features)

**Removed (100% missing):**
- `muspy_drum_in_pattern_rate_triple`
- `muspy_drum_in_pattern_rate_duple`
- `muspy_drum_pattern_consistency`

**Final feature count:** 31 features

### Missing Value Imputation

| Feature Type | Missing % | Imputation Method | Median Value |
|--------------|-----------|-------------------|--------------|
| theory_centricity | 50.38% | Median | 0.2667 |
| theory_limited_macroharmony | 50.38% | Median | 0.8736 |
| theory_conjuction_melody_motion | 50.38% | Median | 3.9667 |
| theory_general_tonality | 50.38% | Median | 5.2186 |
| pm_interval_range_min | 3.20% | Median | -7.0000 |
| pm_interval_range_max | 3.20% | Median | 7.0000 |
| pm_groove | 3.20% | Median | 0.4437 |

### Feature Scaling
- **Method:** StandardScaler (zero mean, unit variance)
- **Applied to:** All 31 features
- **Fitted on:** Training set only (no data leakage)

### Target Variable
- **Name:** `pm_energy`
- **Type:** Continuous (regression task)
- **Range:** [0.03, 975.36]
- **Mean:** 14.83 ± 52.20

---

## 3. Top 15 Features by SHAP Importance

SHAP importance is measured by the mean absolute SHAP value across all test samples.

| Rank | Feature | Mean \|SHAP\| | Mean SHAP | Direction | Category |
|------|---------|---------------|-----------|-----------|----------|
| 1 | **pm_note_count** | 22.0639 | -10.13 | Mixed (primarily negative) | PM |
| 2 | **pm_average_polyphony** | 8.9191 | +3.23 | Positive | PM |
| 3 | **muspy_n_pitches_used** | 7.9204 | -1.18 | Negative | Muspy |
| 4 | **muspy_n_pitch_classes_used** | 7.5931 | -0.68 | Negative | Muspy |
| 5 | **muspy_polyphony** | 7.5182 | -0.79 | Negative | Muspy |
| 6 | **muspy_pitch_entropy** | 5.3133 | +0.32 | Positive | Muspy |
| 7 | **theory_conjuction_melody_motion** | 5.1868 | -0.93 | Negative | Theory |
| 8 | **pm_average_velocity** | 4.1448 | +0.72 | Positive | PM |
| 9 | **muspy_pitch_class_entropy** | 3.7860 | ~0.00 | Neutral | Muspy |
| 10 | **pm_groove** | 3.7202 | +0.75 | Positive | PM |
| 11 | **theory_centricity** | 3.6365 | +2.42 | Positive | Theory |
| 12 | **muspy_empty_measure_rate** | 3.5411 | -0.60 | Negative | Muspy |
| 13 | **muspy_empty_beat_rate** | 3.5411 | -0.60 | Negative | Muspy |
| 14 | **pm_note_density** | 3.0742 | -1.37 | Negative | PM |
| 15 | **theory_general_tonality** | 2.3499 | +0.44 | Positive | Theory |

### Feature Category Distribution (Top 15)
- **PM features:** 5/15 (33%)
- **Muspy features:** 7/15 (47%)
- **Theory features:** 3/15 (20%)

---

## 4. Key Feature Interactions

**Status:** Not Available

**Reason:** SHAP interaction values are only computable for tree-based models (RandomForest, GradientBoosting, XGBoost). The best performing model was Ridge Regression (linear model), which does not support TreeExplainer-based interaction analysis.

**Alternative:** To obtain feature interactions, one could:
1. Use RandomForest model (R² = 0.6777) with TreeExplainer
2. Manually compute correlation-based interactions from the data
3. Use the multicollinearity analysis from EDA (see VIF scores)

**Known Correlated Pairs (from EDA):**
- `pm_note_count` ↔ `pm_energy` (r = 0.94)
- `pm_note_count` ↔ `pm_note_density` (r = 0.94)
- `pm_pitch_range_max_freq` ↔ `pm_average_pitch_hz` (r = 0.96)
- `muspy_empty_beat_rate` ↔ `muspy_empty_measure_rate` (r = perfect correlation)

---

## 5. Interpretation of Global Importance Patterns

### 5.1 Dominant Feature: `pm_note_count`

**Mean |SHAP| = 22.06** (2.5× more important than #2)

- **Interpretation:** Note count is by far the most important feature for predicting energy
- **Behavior:** Variable effect (mean SHAP = -10.13, but high variance)
- **Insight:** More notes typically correlate with energy, but relationship is complex
- **Max Impact:** Can shift prediction by ±127 units (massive influence)

### 5.2 Polyphony Features

**Two polyphony measures in top 5:**
- `pm_average_polyphony` (#2, mean |SHAP| = 8.92): **Positive effect**
- `muspy_polyphony` (#5, mean |SHAP| = 7.52): **Negative effect**

**Interpretation:** These measure different aspects:
- PM average polyphony: sustained concurrent notes → increases energy
- Muspy polyphony: instantaneous polyphony → decreases energy (possibly due to texture differences)

### 5.3 Pitch Complexity Cluster

**Features #3-6 all relate to pitch usage:**
- `muspy_n_pitches_used` (#3): More unique pitches → lower energy
- `muspy_n_pitch_classes_used` (#4): More pitch classes → lower energy
- `muspy_pitch_entropy` (#6): Higher entropy → higher energy

**Interpretation:**
- Raw pitch diversity reduces energy (simpler melodies = more energetic)
- BUT higher pitch entropy (unpredictability) increases energy
- Suggests energy comes from rhythmic/dynamic variation, not melodic complexity

### 5.4 Theory Features

**3 of 4 theory features in top 15:**
- `theory_conjuction_melody_motion` (#7): Negative effect
- `theory_centricity` (#11): Positive effect (mean SHAP = +2.42)
- `theory_general_tonality` (#15): Positive effect

**Interpretation:**
- Tonal centricity (clear tonal center) → higher energy
- Complex melodic motion patterns → lower energy
- Strong tonality → higher energy
- Theory features are important despite 50% missing values!

### 5.5 Rhythmic/Dynamic Features

- `pm_average_velocity` (#8): Positive (louder = more energy)
- `pm_groove` (#10): Positive (groovier = more energy)
- `muspy_empty_measure_rate` (#12): Negative (more silence = less energy)
- `muspy_empty_beat_rate` (#13): Negative (more silence = less energy)

**Interpretation:** Clear and expected patterns:
- Dynamic intensity increases energy
- Groove increases energy
- Silence decreases energy

### 5.6 Feature Effect Direction Summary

**Positive Effects (7 features):** Increase target value
- pm_average_polyphony, pm_average_velocity, pm_groove
- muspy_pitch_entropy, theory_centricity, theory_general_tonality
- (and 1 more)

**Negative Effects (8 features):** Decrease target value
- pm_note_count (complex relationship), pm_note_density
- muspy_n_pitches_used, muspy_n_pitch_classes_used, muspy_polyphony
- theory_conjuction_melody_motion, muspy_empty_measure_rate, muspy_empty_beat_rate

---

## 6. Notable Local Explanations from Representative Samples

### Sample 1: Low Target (Actual = 0.81, Predicted = -2.95, Error = -3.76)

**Top Contributing Features:**
1. `pm_note_count` (SHAP = -35.69): Very low note count pushes prediction down
2. `muspy_pitch_entropy` (SHAP = -13.66): Low entropy reduces energy
3. `muspy_n_pitch_classes_used` (SHAP = +13.37): Fewer pitch classes surprisingly push up
4. `muspy_n_pitches_used` (SHAP = +11.68): Fewer pitches push up
5. `muspy_pitch_class_entropy` (SHAP = -9.34): Low entropy reduces energy

**Interpretation:** Simple piece with few notes, low pitch complexity → correctly predicted as very low energy

---

### Sample 2: Medium Target (Actual = 4.67, Predicted = 0.59, Error = -4.08)

**Top Contributing Features:**
1. `pm_note_count` (SHAP = -21.23): Below average note count
2. `muspy_polyphony` (SHAP = +4.45): Polyphony increases prediction
3. `pm_note_density` (SHAP = -4.28): Lower density decreases prediction
4. `pm_average_velocity` (SHAP = +4.03): Higher velocity increases prediction
5. `muspy_n_pitches_used` (SHAP = +3.98): Pitch usage increases prediction

**Interpretation:** Medium energy piece underpredicted due to low note count dominating other positive factors

---

### Sample 3: High Target (Actual = 164.54, Predicted = 157.75, Error = -6.79)

**Top Contributing Features:**
1. `pm_note_count` (SHAP = +127.46): Very high note count → massive positive impact
2. `pm_note_density` (SHAP = +41.19): High density amplifies effect
3. `theory_conjuction_melody_motion` (SHAP = -18.57): Complex motion reduces
4. `muspy_polyphony` (SHAP = -18.35): High polyphony reduces
5. `theory_general_tonality` (SHAP = +8.54): Strong tonality increases

**Interpretation:** High energy piece correctly predicted. Note count and density drive high prediction, moderated by complexity features.

---

### Sample 4: Largest Error (Actual = 1.59, Predicted = 25.51, Error = +23.92)

**Top Contributing Features:**
1. `pm_note_count` (SHAP = -31.55): Low note count pushes down
2. `pm_average_polyphony` (SHAP = -18.95): Very low polyphony pushes down
3. `pm_groove` (SHAP = +17.52): **Anomalously high groove pushes up**
4. `muspy_n_pitch_classes_used` (SHAP = +13.37): Few pitch classes push up
5. `muspy_empty_measure_rate` (SHAP = +13.35): **High empty rate pushes up**

**Interpretation:** **Model failure case!** High groove and empty measure rate contradict low note count → model confused. This suggests:
- Groove feature may be noisy or unreliable
- Model struggles with sparse, groovy pieces
- Potential outlier or data quality issue

---

## 7. Paths to All Saved Files

### Data Files
- **SHAP values (NumPy array):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/data/shap_values.npy`
- **SHAP importance rankings (CSV):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/data/shap_importance.csv`
- **Feature interactions (Note):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/data/shap_interactions.csv` *(Not available for linear model)*

### Visualization Files
- **SHAP summary plot (beeswarm):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/plots/shap_summary_beeswarm.png`
- **SHAP importance bar chart:** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/plots/shap_importance_bar.png`
- **Dependence plots (top 5):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/plots/shap_dependence_top5.png`
- **Waterfall plots (samples):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/plots/shap_waterfall_samples.png`
- **Force plots (samples):** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/plots/shap_force_plots.png`

### Report Files
- **JSON summary:** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/reports/shap_summary.json`
- **This summary:** `/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/reports/SHAP_ANALYSIS_SUMMARY.md`

---

## 8. Key Insights and Recommendations

### 8.1 Feature Importance Hierarchy

**Tier 1 (Critical):**
- `pm_note_count`: Overwhelmingly most important

**Tier 2 (Very Important):**
- `pm_average_polyphony`, `muspy_n_pitches_used`, `muspy_n_pitch_classes_used`, `muspy_polyphony`

**Tier 3 (Moderately Important):**
- `muspy_pitch_entropy`, `theory_conjuction_melody_motion`, `pm_average_velocity`, `muspy_pitch_class_entropy`, `pm_groove`

**Tier 4 (Less Important):**
- All other features (mean |SHAP| < 3.0)

### 8.2 Interpretability Insights

1. **Note count dominates prediction** - Any model predicting energy will heavily rely on note count
2. **Polyphony has dual effects** - Different polyphony measures have opposite effects
3. **Pitch complexity is nuanced** - Raw diversity ≠ entropy; they affect energy differently
4. **Theory features matter** - Despite 50% missing values, theory features rank highly
5. **Empty space matters** - Silence/emptiness features are important negative predictors

### 8.3 Model Limitations

1. **Overfitting to note count** - Model may overfit to the single dominant feature
2. **Error patterns** - Large errors occur when groove/sparsity conflict with note count
3. **Linear assumptions** - Ridge assumes linear relationships; reality may be non-linear
4. **Multicollinearity** - High VIF features may cause instability in SHAP values

### 8.4 Recommendations

**For Feature Engineering:**
1. Consider interaction terms: `pm_note_count × pm_note_density`
2. Create ratio features: `polyphony_ratio = pm_average_polyphony / muspy_polyphony`
3. Investigate `pm_groove` feature quality (causes large errors)

**For Model Improvement:**
1. Try regularization on note count to reduce dominance
2. Consider polynomial features for non-linear relationships
3. Ensemble linear + tree models to capture both linear and interaction effects

**For Feature Selection:**
- If reducing features, keep all Tier 1-2 features (top 5)
- Tier 3 can be selectively removed
- Tier 4 features have minimal impact and can be dropped

**For Data Quality:**
- Investigate sample with largest error (test_index=101)
- Validate `pm_groove` feature computation
- Consider handling theory features differently (imputation may not be ideal)

---

## Conclusion

This SHAP analysis reveals that **note count is the overwhelmingly dominant predictor of musical energy**, with polyphony and pitch complexity features playing important but secondary roles. The Ridge regression model achieves strong performance (R² = 0.74) with interpretable linear effects.

Key takeaways:
- Feature importance has a clear hierarchy with note count at the top
- Muspy features (7/15) dominate the top features
- Theory features remain important despite 50% missing values
- Model is interpretable and performs well, but may overfit to note count
- Opportunities exist for feature engineering and model ensembling

---

**Analysis completed:** 2025-11-13
**Total runtime:** ~8 minutes
**SHAP explainer:** LinearExplainer
**Test samples analyzed:** 107
**Features analyzed:** 31
