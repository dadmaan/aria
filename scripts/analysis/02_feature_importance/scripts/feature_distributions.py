#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) for Music Features Dataset
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
import json
import warnings
import argparse
from pathlib import Path

# Add src to path for logging imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Parse arguments
parser = argparse.ArgumentParser(description="Comprehensive EDA for Music Features")
parser.add_argument(
    "--data_path",
    type=str,
    default="artifacts/features/raw/commu_full/features_numeric.csv",
    help="Path to input CSV file",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/feature_importance_analysis/comprehensive_commu_full",
    help="Output directory path",
)

args = parser.parse_args()

DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir

# Create output directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/data").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/plots").mkdir(parents=True, exist_ok=True)
Path(f"{OUTPUT_DIR}/reports").mkdir(parents=True, exist_ok=True)

# Initialize logging
log_file = Path(OUTPUT_DIR) / "comprehensive_eda.log"
logger = LoggingManager(
    name="comprehensive_eda",
    log_file=str(log_file),
    enable_wandb=False,
)

logger.info("=" * 80)
logger.info("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
logger.info("=" * 80)
logger.info("Data: %s", DATA_PATH)
logger.info("Output: %s", OUTPUT_DIR)
logger.info("=" * 80)

# ============================================================================
# 1. LOAD DATA AND INITIAL EXPLORATION
# ============================================================================

logger.info("")
logger.info("1. LOADING DATA AND INITIAL EXPLORATION")
logger.info("-" * 80)

# Load dataset
df = pd.read_csv(DATA_PATH)
logger.info("✓ Dataset loaded successfully")
logger.info("  Shape: %s", df.shape)
logger.info("  Rows: %d, Columns: %d", df.shape[0], df.shape[1])

# Separate features and metadata
metadata_cols = ["track_id", "metadata_index"]
feature_cols = [col for col in df.columns if col not in metadata_cols]
logger.info("")
logger.info("  Feature columns: %d", len(feature_cols))
logger.info("  Metadata columns: %d", len(metadata_cols))

# Extract features for analysis
X = df[feature_cols].copy()

# Check data types
logger.info("")
logger.info("  Data types:")
logger.info("%s", X.dtypes.value_counts())

# Basic statistics
logger.info("")
logger.info("  Basic Statistics:")
logger.info("%s", X.describe())

# ============================================================================
# 2. MISSING VALUES ANALYSIS
# ============================================================================

logger.info("")
logger.info("2. MISSING VALUES ANALYSIS")
logger.info("-" * 80)

missing_values = X.isnull().sum()
missing_percentage = (missing_values / len(X)) * 100
missing_df = pd.DataFrame(
    {
        "feature": missing_values.index,
        "missing_count": missing_values.values,
        "missing_percentage": missing_percentage.values,
    }
)
missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
    "missing_count", ascending=False
)

if len(missing_df) > 0:
    logger.info("")
    logger.info("  Features with missing values: %d", len(missing_df))
    logger.info("%s", missing_df.to_string(index=False))

    # Save missing values report
    missing_df.to_csv(f"{OUTPUT_DIR}/data/missing_values_report.csv", index=False)
    logger.info("")
    logger.info("  ✓ Missing values report saved")
else:
    logger.info("  ✓ No missing values found in dataset")

# ============================================================================
# 3. DISTRIBUTION ANALYSIS (SKEWNESS & KURTOSIS)
# ============================================================================

logger.info("")
logger.info("3. DISTRIBUTION ANALYSIS")
logger.info("-" * 80)

distribution_stats = []
for col in feature_cols:
    # Skip columns with too many NaN values
    if X[col].isnull().sum() / len(X) > 0.5:
        continue

    data = X[col].dropna()
    if len(data) > 0:
        distribution_stats.append(
            {
                "feature": col,
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "skewness": skew(data),
                "kurtosis": kurtosis(data),
                "n_zeros": (data == 0).sum(),
                "n_unique": data.nunique(),
            }
        )

dist_df = pd.DataFrame(distribution_stats)
logger.info("")
logger.info("  Features analyzed: %d", len(dist_df))
logger.info("")
logger.info("  Highly skewed features (|skewness| > 1):")
highly_skewed = dist_df[abs(dist_df["skewness"]) > 1.0].sort_values(
    "skewness", ascending=False
)
if len(highly_skewed) > 0:
    logger.info(
        "%s", highly_skewed[["feature", "skewness", "kurtosis"]].to_string(index=False)
    )
else:
    logger.info("  None found")

# Save distribution stats
dist_df.to_csv(f"{OUTPUT_DIR}/data/distribution_stats.csv", index=False)
logger.info("")
logger.info("  ✓ Distribution statistics saved")

# ============================================================================
# 4. OUTLIER DETECTION
# ============================================================================

logger.info("")
logger.info("4. OUTLIER DETECTION")
logger.info("-" * 80)

outlier_results = []

for col in feature_cols:
    data = X[col].dropna()
    if len(data) < 10:  # Skip if too few samples
        continue

    # IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()

    # Z-score method (|z| > 3)
    z_scores = np.abs(stats.zscore(data))
    z_outliers = (z_scores > 3).sum()

    outlier_results.append(
        {
            "feature": col,
            "iqr_outliers": iqr_outliers,
            "iqr_outlier_pct": (iqr_outliers / len(data)) * 100,
            "zscore_outliers": z_outliers,
            "zscore_outlier_pct": (z_outliers / len(data)) * 100,
        }
    )

outlier_df = pd.DataFrame(outlier_results)
outlier_df = outlier_df.sort_values("iqr_outliers", ascending=False)

logger.info("")
logger.info("  Top 10 features with most outliers (IQR method):")
logger.info(
    "%s",
    outlier_df.head(10)[["feature", "iqr_outliers", "iqr_outlier_pct"]].to_string(
        index=False
    ),
)

# Save outlier results
outlier_df.to_csv(f"{OUTPUT_DIR}/data/outlier_detection.csv", index=False)
logger.info("")
logger.info("  ✓ Outlier detection results saved")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================

logger.info("")
logger.info("5. CORRELATION ANALYSIS")
logger.info("-" * 80)

# Calculate correlation matrix
# Handle missing values by using pairwise complete observations
corr_matrix = X.corr(method="pearson")

# Save correlation matrix
corr_matrix.to_csv(f"{OUTPUT_DIR}/data/correlation_matrix.csv")
logger.info(
    "  ✓ Correlation matrix saved (%d x %d)", corr_matrix.shape[0], corr_matrix.shape[1]
)

# Find highly correlated pairs
threshold = 0.8
highly_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > threshold and not pd.isna(corr_val):
            highly_corr_pairs.append(
                {
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": corr_val,
                }
            )

if highly_corr_pairs:
    highly_corr_df = pd.DataFrame(highly_corr_pairs).sort_values(
        "correlation", key=abs, ascending=False
    )
    logger.info("")
    logger.info(
        "  Highly correlated pairs (|r| > %f): %d", threshold, len(highly_corr_df)
    )
    logger.info("%s", highly_corr_df.to_string(index=False))

    # Save highly correlated pairs
    highly_corr_df.to_csv(f"{OUTPUT_DIR}/data/highly_correlated_pairs.csv", index=False)
else:
    logger.info("")
    logger.info("  No feature pairs with |correlation| > %f", threshold)

# ============================================================================
# 6. VIF (VARIANCE INFLATION FACTOR) ANALYSIS
# ============================================================================

logger.info("")
logger.info("6. VIF ANALYSIS (MULTICOLLINEARITY)")
logger.info("-" * 80)

# Prepare data for VIF (remove NaN values)
X_clean = X.dropna()

if len(X_clean) < 10:
    logger.warning(
        "  ⚠ Insufficient data after removing NaN values for VIF calculation"
    )
    vif_scores = []
else:
    logger.info("  Computing VIF for %d features...", len(feature_cols))
    logger.info("  (Using %d complete samples)", len(X_clean))

    vif_scores = []
    for i, col in enumerate(feature_cols):
        try:
            if col in X_clean.columns and X_clean[col].std() > 0:
                vif = variance_inflation_factor(X_clean[feature_cols].values, i)
                vif_scores.append({"feature": col, "VIF": vif})
        except Exception as e:
            logger.warning("  ⚠ Error calculating VIF for %s: %s", col, str(e))
            vif_scores.append({"feature": col, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_scores).sort_values("VIF", ascending=False)

    # Save VIF scores
    vif_df.to_csv(f"{OUTPUT_DIR}/data/vif_scores.csv", index=False)
    logger.info("")
    logger.info("  ✓ VIF scores calculated and saved")

    # Show high VIF features
    high_vif = vif_df[vif_df["VIF"] > 10]
    if len(high_vif) > 0:
        logger.info("")
        logger.info(
            "  Features with high multicollinearity (VIF > 10): %d", len(high_vif)
        )
        logger.info("%s", high_vif.to_string(index=False))
    else:
        logger.info("")
        logger.info("  No features with VIF > 10")

# ============================================================================
# 7. MUTUAL INFORMATION ANALYSIS
# ============================================================================

logger.info("")
logger.info("7. MUTUAL INFORMATION ANALYSIS")
logger.info("-" * 80)

# Calculate mutual information between all feature pairs
# This is computationally expensive, so we'll do a sample if needed
logger.info("  Computing mutual information between feature pairs...")

mi_results = []
X_clean_mi = X.fillna(X.median())  # Simple imputation for MI

for i, col1 in enumerate(feature_cols):
    logger.debug("  Progress: %d/%d", i + 1, len(feature_cols))
    for col2 in feature_cols[i + 1 :]:
        if col1 != col2:
            try:
                # Calculate MI
                mi_score = mutual_info_regression(
                    X_clean_mi[[col1]].values, X_clean_mi[col2].values, random_state=42
                )[0]

                mi_results.append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "mutual_information": mi_score,
                    }
                )
            except:
                continue

mi_df = pd.DataFrame(mi_results).sort_values("mutual_information", ascending=False)

# Save MI results
mi_df.to_csv(f"{OUTPUT_DIR}/data/mutual_information.csv", index=False)
logger.info("")
logger.info("  ✓ Mutual information scores saved (%d pairs)", len(mi_df))

logger.info("")
logger.info("  Top 10 feature pairs by mutual information:")
logger.info("%s", mi_df.head(10).to_string(index=False))

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

logger.info("")
logger.info("8. GENERATING VISUALIZATIONS")
logger.info("-" * 80)

# 8.1 Distribution plots
logger.info("")
logger.info("  8.1 Creating distribution plots...")
n_features = len(feature_cols)
n_cols = 5
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    data = X[col].dropna()

    if len(data) > 0:
        # Histogram with KDE
        ax.hist(
            data, bins=30, density=True, alpha=0.7, color="skyblue", edgecolor="black"
        )

        # Add KDE if enough data points
        if len(data) > 5:
            try:
                data_sorted = np.sort(data)
                kde = stats.gaussian_kde(data)
                ax.plot(data_sorted, kde(data_sorted), "r-", linewidth=2, label="KDE")
            except:
                pass

        ax.set_title(f"{col}\n(μ={data.mean():.2f}, σ={data.std():.2f})", fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(labelsize=6)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(col, fontsize=8)

# Hide extra subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/distributions.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info("  ✓ Distribution plots saved")

# 8.2 Box plots for outlier visualization
logger.info("")
logger.info("  8.2 Creating box plots...")
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
axes = axes.flatten()

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    data = X[col].dropna()

    if len(data) > 0:
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][0].set_alpha(0.7)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xticklabels([])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(col, fontsize=8)

# Hide extra subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/boxplots.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info("  ✓ Box plots saved")

# 8.3 Correlation heatmap
logger.info("")
logger.info("  8.3 Creating correlation heatmap...")
plt.figure(figsize=(16, 14))

# Create mask for better visualization
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)

plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
plt.xticks(rotation=90, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info("  ✓ Correlation heatmap saved")

# 8.4 VIF bar plot (if available)
if len(vif_scores) > 0:
    logger.info("")
    logger.info("  8.4 Creating VIF bar plot...")
    vif_df_plot = vif_df[vif_df["VIF"].notna()].copy()
    vif_df_plot = vif_df_plot.sort_values("VIF", ascending=True)

    plt.figure(figsize=(12, max(8, len(vif_df_plot) * 0.3)))
    colors = [
        "red" if x > 10 else "orange" if x > 5 else "green" for x in vif_df_plot["VIF"]
    ]

    plt.barh(range(len(vif_df_plot)), vif_df_plot["VIF"], color=colors, alpha=0.7)
    plt.yticks(range(len(vif_df_plot)), vif_df_plot["feature"], fontsize=8)
    plt.xlabel("VIF Score", fontsize=12)
    plt.title(
        "Variance Inflation Factor (VIF) by Feature\nGreen: Low (<5), Orange: Medium (5-10), Red: High (>10)",
        fontsize=14,
        fontweight="bold",
    )
    plt.axvline(
        x=5, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="VIF=5"
    )
    plt.axvline(
        x=10, color="red", linestyle="--", linewidth=1, alpha=0.5, label="VIF=10"
    )
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/vif_scores.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  ✓ VIF bar plot saved")

# ============================================================================
# 9. COMPREHENSIVE SUMMARY REPORT
# ============================================================================

logger.info("")
logger.info("9. GENERATING COMPREHENSIVE SUMMARY REPORT")
logger.info("-" * 80)

# Prepare summary report
summary_report = {
    "dataset_info": {
        "total_samples": int(df.shape[0]),
        "total_features": len(feature_cols),
        "feature_categories": {
            "pm_features": len([c for c in feature_cols if c.startswith("pm_")]),
            "muspy_features": len([c for c in feature_cols if c.startswith("muspy_")]),
            "theory_features": len(
                [c for c in feature_cols if c.startswith("theory_")]
            ),
        },
    },
    "data_quality": {
        "missing_values": {
            "total_features_with_missing": int(len(missing_df)),
            "features_with_missing": (
                missing_df["feature"].tolist() if len(missing_df) > 0 else []
            ),
        },
        "outliers": {
            "features_with_many_outliers_iqr": outlier_df[
                outlier_df["iqr_outlier_pct"] > 5
            ]["feature"].tolist()[:10]
        },
    },
    "distribution_characteristics": {
        "highly_skewed_features": (
            highly_skewed["feature"].tolist() if len(highly_skewed) > 0 else []
        ),
        "skewness_stats": {
            "mean_abs_skewness": float(abs(dist_df["skewness"]).mean()),
            "max_skewness": float(dist_df["skewness"].max()),
            "min_skewness": float(dist_df["skewness"].min()),
        },
    },
    "correlation_analysis": {
        "highly_correlated_pairs_count": len(highly_corr_pairs),
        "top_correlated_pairs": (
            [
                {
                    "feature_1": pair["feature_1"],
                    "feature_2": pair["feature_2"],
                    "correlation": float(pair["correlation"]),
                }
                for pair in highly_corr_pairs[:10]
            ]
            if highly_corr_pairs
            else []
        ),
    },
    "multicollinearity_analysis": {
        "vif_computed": len(vif_scores) > 0,
        "features_with_high_vif": (
            vif_df[vif_df["VIF"] > 10]["feature"].tolist()
            if len(vif_scores) > 0
            else []
        ),
        "vif_stats": {
            "mean_vif": (
                float(vif_df["VIF"].mean())
                if len(vif_scores) > 0 and not vif_df["VIF"].isna().all()
                else None
            ),
            "median_vif": (
                float(vif_df["VIF"].median())
                if len(vif_scores) > 0 and not vif_df["VIF"].isna().all()
                else None
            ),
            "max_vif": (
                float(vif_df["VIF"].max())
                if len(vif_scores) > 0 and not vif_df["VIF"].isna().all()
                else None
            ),
        },
    },
    "mutual_information_analysis": {
        "total_pairs_analyzed": len(mi_df),
        "top_mi_pairs": [
            {
                "feature_1": row["feature_1"],
                "feature_2": row["feature_2"],
                "mutual_information": float(row["mutual_information"]),
            }
            for _, row in mi_df.head(10).iterrows()
        ],
    },
}

# Save summary report
with open(f"{OUTPUT_DIR}/reports/eda_summary.json", "w") as f:
    json.dump(summary_report, f, indent=2)

logger.info("  ✓ Comprehensive summary report saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

logger.info("")
logger.info("=" * 80)
logger.info("EDA COMPLETE - SUMMARY")
logger.info("=" * 80)

logger.info("")
logger.info("📊 Dataset Overview:")
logger.info("  • Total samples: %d", df.shape[0])
logger.info("  • Total features: %d", len(feature_cols))
logger.info(
    "    - pm_* features: %d", len([c for c in feature_cols if c.startswith("pm_")])
)
logger.info(
    "    - muspy_* features: %d",
    len([c for c in feature_cols if c.startswith("muspy_")]),
)
logger.info(
    "    - theory_* features: %d",
    len([c for c in feature_cols if c.startswith("theory_")]),
)

logger.info("")
logger.info("🔍 Data Quality:")
logger.info("  • Features with missing values: %d", len(missing_df))
if len(missing_df) > 0:
    for _, row in missing_df.iterrows():
        logger.info(
            "    - %s: %d (%.2f%%)",
            row["feature"],
            row["missing_count"],
            row["missing_percentage"],
        )

logger.info("")
logger.info("📈 Distribution Characteristics:")
logger.info("  • Highly skewed features (|skew| > 1): %d", len(highly_skewed))
if len(highly_skewed) > 0:
    for _, row in highly_skewed.head(5).iterrows():
        logger.info("    - %s: skewness=%.2f", row["feature"], row["skewness"])

logger.info("")
logger.info("🔗 Correlation Findings:")
logger.info("  • Highly correlated pairs (|r| > 0.8): %d", len(highly_corr_pairs))
if highly_corr_pairs:
    for pair in highly_corr_pairs[:5]:
        logger.info(
            "    - %s ↔ %s: r=%.3f",
            pair["feature_1"],
            pair["feature_2"],
            pair["correlation"],
        )

logger.info("")
logger.info("🎯 Multicollinearity (VIF):")
if len(vif_scores) > 0:
    high_vif_features = vif_df[vif_df["VIF"] > 10]
    logger.info("  • Features with VIF > 10: %d", len(high_vif_features))
    for _, row in high_vif_features.head(5).iterrows():
        logger.info("    - %s: VIF=%.2f", row["feature"], row["VIF"])
else:
    logger.info("  • VIF analysis not performed (insufficient data)")

logger.info("")
logger.info("📁 Output Files Generated:")
logger.info("  Data files:")
logger.info("    ✓ %s/data/correlation_matrix.csv", OUTPUT_DIR)
logger.info("    ✓ %s/data/vif_scores.csv", OUTPUT_DIR)
logger.info("    ✓ %s/data/mutual_information.csv", OUTPUT_DIR)
logger.info("    ✓ %s/data/distribution_stats.csv", OUTPUT_DIR)
logger.info("    ✓ %s/data/outlier_detection.csv", OUTPUT_DIR)
if len(missing_df) > 0:
    logger.info("    ✓ %s/data/missing_values_report.csv", OUTPUT_DIR)
if highly_corr_pairs:
    logger.info("    ✓ %s/data/highly_correlated_pairs.csv", OUTPUT_DIR)

logger.info("")
logger.info("  Visualizations:")
logger.info("    ✓ %s/plots/distributions.png", OUTPUT_DIR)
logger.info("    ✓ %s/plots/boxplots.png", OUTPUT_DIR)
logger.info("    ✓ %s/plots/correlation_heatmap.png", OUTPUT_DIR)
if len(vif_scores) > 0:
    logger.info("    ✓ %s/plots/vif_scores.png", OUTPUT_DIR)

logger.info("")
logger.info("  Reports:")
logger.info("    ✓ %s/reports/eda_summary.json", OUTPUT_DIR)

logger.info("")
logger.info("=" * 80)
logger.info("✅ ALL EDA TASKS COMPLETED SUCCESSFULLY")
logger.info("=" * 80)
