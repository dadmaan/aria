# Comprehensive Exploratory Data Analysis Report
## Music Features Dataset Analysis

**Analysis Date:** November 13, 2025
**Dataset:** `artifacts/features/raw/commu_bass/features_numeric.csv`
**Output Directory:** `experiments_feature_analysis/feature_analysis_20251113_214037/`

---

## 1. DATASET OVERVIEW

### Basic Information
- **Total Samples:** 532
- **Total Features:** 35 (excluding track_id and metadata_index)
- **Feature Categories:**
  - `pm_*` features (Pretty MIDI): 18 features
  - `muspy_*` features (Muspy Library): 13 features
  - `theory_*` features (Music Theory): 4 features

---

## 2. DATA QUALITY ASSESSMENT

### 2.1 Missing Values

**Total Features with Missing Values:** 10 out of 35

| Feature | Missing Count | Missing % | Impact |
|---------|--------------|-----------|--------|
| `muspy_drum_in_pattern_rate_triple` | 532 | 100.0% | **CRITICAL - Completely empty** |
| `muspy_drum_in_pattern_rate_duple` | 532 | 100.0% | **CRITICAL - Completely empty** |
| `muspy_drum_pattern_consistency` | 532 | 100.0% | **CRITICAL - Completely empty** |
| `theory_limited_macroharmony` | 268 | 50.4% | **HIGH - Half missing** |
| `theory_conjuction_melody_motion` | 268 | 50.4% | **HIGH - Half missing** |
| `theory_centricity` | 268 | 50.4% | **HIGH - Half missing** |
| `theory_general_tonality` | 268 | 50.4% | **HIGH - Half missing** |
| `pm_interval_range_min` | 17 | 3.2% | Low |
| `pm_interval_range_max` | 17 | 3.2% | Low |
| `pm_groove` | 17 | 3.2% | Low |

**Key Findings:**
- 3 drum-related features are completely empty (100% missing) - these features should be excluded from analysis
- 4 theory features have 50% missing values - requires careful handling in modeling
- 3 pm_* features have minimal missing values (3.2%) - acceptable for most analyses

### 2.2 Outliers

**Top Features with Most Outliers (IQR Method):**

| Feature | IQR Outliers | % of Data | Z-Score Outliers |
|---------|-------------|-----------|------------------|
| `muspy_polyphony` | 119 | 22.4% | 7 |
| `pm_average_polyphony` | 99 | 18.6% | 8 |
| `muspy_scale_consistency` | 90 | 16.9% | 10 |
| `muspy_pitch_class_entropy` | 74 | 13.9% | 5 |
| `pm_bar_count` | 65 | 12.2% | 7 |
| `pm_energy` | 56 | 10.5% | 17 |
| `pm_note_count` | 50 | 9.4% | 20 |
| `pm_pitch_range_min_freq` | 50 | 9.4% | 15 |

**Key Findings:**
- Several features show high outlier rates (>10% of data)
- Outliers may represent genuine musical variations or data quality issues
- Z-score method shows fewer outliers, suggesting most are not extreme

---

## 3. DISTRIBUTION CHARACTERISTICS

### 3.1 Highly Skewed Features (|skewness| > 1)

**16 features exhibit significant skewness:**

| Feature | Skewness | Kurtosis | Interpretation |
|---------|----------|----------|----------------|
| `pm_energy` | 13.27 | 222.20 | **Extreme right skew** |
| `pm_note_count` | 8.68 | 112.11 | **Extreme right skew** |
| `pm_note_density` | 8.14 | 100.09 | **Extreme right skew** |
| `pm_pitch_range_max_freq` | 5.80 | 59.66 | **High right skew** |
| `pm_average_pitch_hz` | 5.57 | 54.36 | **High right skew** |
| `pm_pitch_range_min_freq` | 2.78 | 11.22 | Moderate right skew |
| `pm_groove` | 1.73 | 5.14 | Moderate right skew |
| `pm_max_polyphony` | 1.69 | 4.26 | Moderate right skew |
| `muspy_empty_measure_rate` | 1.56 | 1.10 | Moderate right skew |
| `muspy_empty_beat_rate` | 1.56 | 1.10 | Moderate right skew |
| `muspy_n_pitches_used` | 1.52 | 3.55 | Moderate right skew |
| `muspy_polyphony` | 1.39 | -0.04 | Moderate right skew |
| `pm_bar_count` | 1.01 | 0.73 | Slight right skew |
| `muspy_pitch_class_entropy` | -1.08 | 0.27 | Slight left skew |
| `muspy_groove_consistency` | -2.07 | 6.76 | Moderate left skew |
| `muspy_scale_consistency` | -2.62 | 5.93 | Moderate left skew |

**Skewness Statistics:**
- Mean absolute skewness: 2.43
- Maximum skewness: 13.27 (pm_energy)
- Minimum skewness: -2.62 (muspy_scale_consistency)

**Recommendations:**
- Consider log transformation for highly skewed features (skewness > 5)
- Power transformation (Box-Cox, Yeo-Johnson) may help normalize distributions
- Robust scaling methods for outlier-prone features

---

## 4. CORRELATION ANALYSIS

### 4.1 Highly Correlated Feature Pairs (|r| > 0.8)

**Found 25 highly correlated pairs:**

#### Perfect/Near-Perfect Correlations (r ≥ 0.95):
1. **muspy_empty_beat_rate ↔ muspy_empty_measure_rate**: r = 1.000
   - *Perfect linear relationship - redundant features*

2. **theory_conjuction_melody_motion ↔ theory_general_tonality**: r = 0.997
   - *Near-perfect correlation - extremely redundant*

3. **pm_pitch_range_max_freq ↔ pm_average_pitch_hz**: r = 0.958
   - *Max frequency highly predicts average pitch*

#### Very High Correlations (0.90 ≤ r < 0.95):
4. **pm_note_count ↔ pm_note_density**: r = 0.944
5. **pm_note_count ↔ pm_energy**: r = 0.937
6. **pm_pitch_range_min_freq ↔ pm_pitch_range_min_note**: r = 0.933
7. **muspy_n_pitch_classes_used ↔ muspy_pitch_class_entropy**: r = 0.931
8. **muspy_polyphony ↔ theory_conjuction_melody_motion**: r = 0.926
9. **pm_energy ↔ pm_note_density**: r = 0.918
10. **muspy_polyphony ↔ theory_general_tonality**: r = 0.909
11. **pm_interval_range_min ↔ theory_conjuction_melody_motion**: r = -0.901
12. **muspy_n_pitches_used ↔ muspy_pitch_entropy**: r = 0.895
13. **pm_interval_range_min ↔ theory_general_tonality**: r = -0.894

#### High Correlations (0.80 ≤ r < 0.90):
14-25. Additional 12 pairs (see correlation_matrix.csv)

**Key Correlation Groups:**
1. **Note Activity Group**: pm_note_count, pm_energy, pm_note_density (r > 0.91)
2. **Pitch Frequency Group**: pm_pitch_range_max_freq, pm_average_pitch_hz (r = 0.96)
3. **Pitch Entropy Group**: muspy_pitch_entropy, muspy_pitch_class_entropy, muspy_n_pitches_used (r > 0.85)
4. **Theory Tonality Group**: theory_conjuction_melody_motion, theory_general_tonality (r = 0.997)
5. **Empty Measure Group**: muspy_empty_beat_rate, muspy_empty_measure_rate (r = 1.0)

---

## 5. MULTICOLLINEARITY ANALYSIS (VIF)

### 5.1 VIF Scores Summary

**Analysis Method:** Median imputation for missing values
**Features Analyzed:** 32 (excluding 3 features with 100% missing values)

**VIF Statistics:**
- Mean VIF: 109,604.09 (heavily skewed by extreme values)
- Median VIF: 20.49
- Max finite VIF: 2,935,688.91 (pm_instrument_count)
- Min VIF: 1.51 (muspy_scale_consistency)

### 5.2 Features with Perfect Multicollinearity (VIF = ∞)

**5 features with infinite VIF** - indicating perfect linear dependence:
1. `pm_pitch_range_min_note`
2. `pm_pitch_range_max_note`
3. `muspy_empty_beat_rate`
4. `muspy_pitch_range`
5. `muspy_empty_measure_rate`

**Implication:** These features can be perfectly predicted from other features in the dataset. They provide no additional independent information and should be removed before modeling.

### 5.3 Features with Extreme VIF (VIF > 1000)

| Feature | VIF Score | Severity |
|---------|-----------|----------|
| `pm_instrument_count` | 2,935,688.91 | **CRITICAL** |
| `theory_conjuction_melody_motion` | 11,992.72 | **CRITICAL** |
| `theory_general_tonality` | 10,964.18 | **CRITICAL** |

### 5.4 Features with High VIF (10 < VIF ≤ 1000)

| Feature | VIF Score | Action Needed |
|---------|-----------|---------------|
| `theory_limited_macroharmony` | 138.64 | High priority removal |
| `muspy_pitch_entropy` | 108.01 | High priority removal |
| `muspy_pitch_class_entropy` | 74.27 | High priority removal |
| `pm_average_pitch_hz` | 55.82 | Consider removal |
| `pm_pitch_range_max_freq` | 53.17 | Consider removal |
| `pm_note_count` | 40.73 | Consider removal |
| `muspy_n_pitch_classes_used` | 27.48 | Consider removal |
| `theory_centricity` | 26.66 | Consider removal |
| `pm_energy` | 24.35 | Consider removal |
| `muspy_n_pitches_used` | 20.79 | Consider removal |
| `pm_note_density` | 20.49 | Consider removal |
| `pm_pitch_range_min_freq` | 13.43 | Monitor |
| `muspy_polyphony` | 12.90 | Monitor |
| `pm_average_polyphony` | 10.31 | Monitor |

**Total Features with VIF > 10:** 22 out of 32 (68.75%)

### 5.5 Features with Acceptable VIF (VIF ≤ 10)

Only 10 features have acceptable VIF scores:
- `muspy_scale_consistency`: 1.51
- `pm_average_velocity`: 1.61
- `muspy_groove_consistency`: 1.98
- `pm_groove`: 2.03
- `pm_max_polyphony`: 2.79
- `pm_interval_range_max`: 3.81
- `pm_interval_range_min`: 3.86
- `pm_length_seconds`: 6.16
- `pm_bar_count`: 6.26
- `pm_tempo_bpm`: 7.66

---

## 6. MUTUAL INFORMATION ANALYSIS

### 6.1 Top 10 Feature Pairs by Mutual Information

Mutual information captures both linear and non-linear relationships:

| Rank | Feature 1 | Feature 2 | MI Score | Type |
|------|-----------|-----------|----------|------|
| 1 | `pm_pitch_range_min_freq` | `pm_pitch_range_min_note` | 2.779 | Linear |
| 2 | `pm_pitch_range_max_freq` | `pm_pitch_range_max_note` | 2.706 | Linear |
| 3 | `muspy_pitch_entropy` | `muspy_pitch_class_entropy` | 2.346 | Linear |
| 4 | `muspy_n_pitches_used` | `muspy_pitch_entropy` | 2.175 | Linear |
| 5 | `pm_length_seconds` | `pm_tempo_bpm` | 1.996 | Mixed |
| 6 | `theory_conjuction_melody_motion` | `theory_general_tonality` | 1.877 | Linear |
| 7 | `muspy_n_pitch_classes_used` | `muspy_pitch_class_entropy` | 1.853 | Linear |
| 8 | `muspy_empty_beat_rate` | `muspy_empty_measure_rate` | 1.685 | Linear |
| 9 | `pm_pitch_range_max_freq` | `pm_average_pitch_hz` | 1.503 | Linear |
| 10 | `pm_note_count` | `pm_note_density` | 1.481 | Linear |

**Total Pairs Analyzed:** 496

**Key Findings:**
- Most high-MI pairs also show high Pearson correlation
- MI confirms the redundancy identified by correlation analysis
- pm_length_seconds ↔ pm_tempo_bpm shows interesting non-linear relationship

---

## 7. KEY FINDINGS AND RECOMMENDATIONS

### 7.1 Critical Data Quality Issues

1. **3 Features with 100% Missing Values** - MUST BE REMOVED:
   - muspy_drum_in_pattern_rate_triple
   - muspy_drum_in_pattern_rate_duple
   - muspy_drum_pattern_consistency

2. **4 Theory Features with 50% Missing Values** - REQUIRES CAREFUL HANDLING:
   - theory_limited_macroharmony
   - theory_conjuction_melody_motion
   - theory_centricity
   - theory_general_tonality
   - *Recommendation:* Either impute carefully or use only for analyses with sufficient data

### 7.2 Severe Multicollinearity Issues

**68.75% of features (22 out of 32) have VIF > 10**, indicating severe multicollinearity.

**Priority Removal Candidates:**

**Tier 1 - MUST REMOVE (VIF = ∞ or > 10,000):**
- pm_pitch_range_min_note (VIF = ∞)
- pm_pitch_range_max_note (VIF = ∞)
- muspy_empty_beat_rate (VIF = ∞)
- muspy_pitch_range (VIF = ∞)
- muspy_empty_measure_rate (VIF = ∞)
- pm_instrument_count (VIF = 2,935,689)
- theory_conjuction_melody_motion (VIF = 11,993)
- theory_general_tonality (VIF = 10,964)

**Tier 2 - SHOULD REMOVE (VIF > 50):**
- muspy_pitch_entropy (VIF = 108)
- muspy_pitch_class_entropy (VIF = 74)
- pm_average_pitch_hz (VIF = 56)
- pm_pitch_range_max_freq (VIF = 53)

**Tier 3 - CONSIDER REMOVING (10 < VIF ≤ 50):**
- theory_limited_macroharmony (VIF = 139)
- pm_note_count (VIF = 41)
- muspy_n_pitch_classes_used (VIF = 27)
- theory_centricity (VIF = 27)
- pm_energy (VIF = 24)
- muspy_n_pitches_used (VIF = 21)
- pm_note_density (VIF = 20)
- pm_pitch_range_min_freq (VIF = 13)
- muspy_polyphony (VIF = 13)
- pm_average_polyphony (VIF = 10)

### 7.3 Feature Selection Strategy

**Recommended Approach:**

1. **Remove completely empty features (3 features)**
2. **Remove features with infinite VIF (5 features)**
3. **Remove extreme VIF features (3 features with VIF > 10,000)**
4. **Choose one representative from each highly correlated group:**
   - From note activity group: Keep `pm_note_count`, remove `pm_energy` & `pm_note_density`
   - From pitch frequency group: Keep `pm_pitch_range_max_freq`, remove `pm_average_pitch_hz`
   - From pitch entropy group: Keep `muspy_pitch_entropy`, remove `muspy_pitch_class_entropy`
   - From theory group: Keep `theory_centricity`, remove others (if using theory features)

**Expected Result:** ~15-18 independent features suitable for modeling

### 7.4 Distribution Transformations Needed

**Features requiring transformation:**
- `pm_energy` (skew = 13.27) → Log transformation
- `pm_note_count` (skew = 8.68) → Log transformation
- `pm_note_density` (skew = 8.14) → Log transformation
- `pm_pitch_range_max_freq` (skew = 5.80) → Log or Box-Cox
- `pm_average_pitch_hz` (skew = 5.57) → Log or Box-Cox

---

## 8. OUTPUT FILES

### 8.1 Data Files
All files located in: `experiments_feature_analysis/feature_analysis_20251113_214037/data/`

- `correlation_matrix.csv` (21 KB) - Full 35x35 Pearson correlation matrix
- `vif_scores.csv` (1.7 KB) - VIF scores for 32 features (median imputation)
- `vif_scores_no_missing.csv` (1.2 KB) - VIF scores for 25 features (no missing values)
- `mutual_information.csv` (30 KB) - MI scores for 496 feature pairs
- `distribution_stats.csv` (3.3 KB) - Skewness, kurtosis, and distribution metrics
- `outlier_detection.csv` (1.8 KB) - IQR and Z-score outlier counts
- `missing_values_report.csv` (476 B) - Detailed missing value analysis
- `highly_correlated_pairs.csv` (1.6 KB) - 25 pairs with |r| > 0.8

### 8.2 Visualizations
All files located in: `experiments_feature_analysis/feature_analysis_20251113_214037/plots/`

- `distributions.png` (1.4 MB) - Histograms with KDE for all 35 features
- `boxplots.png` (762 KB) - Box plots showing outliers for all features
- `correlation_heatmap.png` (674 KB) - Triangular heatmap of correlation matrix
- `vif_scores.png` (367 KB) - Horizontal bar chart of VIF scores with color coding

### 8.3 Reports
All files located in: `experiments_feature_analysis/feature_analysis_20251113_214037/reports/`

- `eda_summary.json` (7.5 KB) - Machine-readable comprehensive summary
- `EDA_FINAL_REPORT.md` (this file) - Human-readable comprehensive report

### 8.4 Scripts
All files located in: `experiments_feature_analysis/feature_analysis_20251113_214037/scripts/`

- `comprehensive_eda.py` - Main EDA analysis script
- `vif_analysis.py` - Detailed VIF analysis with multiple strategies
- `update_summary.py` - Summary report update script

---

## 9. CONCLUSIONS

### Dataset Characteristics:
- **Size:** Adequate (532 samples) for most analyses
- **Completeness:** 71% features complete, 29% have missing values
- **Quality:** Moderate - significant multicollinearity issues

### Main Issues:
1. **Severe multicollinearity:** 68.75% of features have VIF > 10
2. **Feature redundancy:** 25 highly correlated pairs (|r| > 0.8)
3. **Missing data:** 3 features completely empty, 4 features 50% missing
4. **Distribution skewness:** 16 features highly skewed (|skew| > 1)

### Actionable Recommendations:
1. Remove 11 features immediately (3 empty + 5 infinite VIF + 3 extreme VIF)
2. Perform careful feature selection to reduce from 35 to ~15-18 features
3. Apply transformations to highly skewed features
4. Handle missing values in theory features appropriately
5. Consider PCA or other dimensionality reduction after feature selection

### Next Steps:
1. Implement feature selection based on VIF and correlation analysis
2. Apply appropriate transformations to skewed features
3. Evaluate feature importance using tree-based methods or SHAP
4. Perform RFE (Recursive Feature Elimination) if needed
5. Validate selected features with cross-validation

---

**Analysis Completed:** November 13, 2025
**Analyst:** Claude Code AI Assistant
**Analysis Duration:** ~5 minutes
**Total Output Size:** ~5.8 MB
