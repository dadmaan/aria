#!/usr/bin/env python3
"""
VIF Analysis with proper handling of missing values
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")

# Paths
DATA_PATH = (
    "/home/user/pilot_study/artifacts/features/raw/commu_bass/features_numeric.csv"
)
OUTPUT_DIR = "/home/user/pilot_study/experiments_feature_analysis/feature_analysis_20251113_214037/"

# Initialize logging
log_manager = LoggingManager(
    name="ghsom_feature_analysis.vif_analysis",
    log_file=f"{OUTPUT_DIR}logs/vif_analysis.log",
)
logger = log_manager.logger

logger.info("=" * 80)
logger.info("VIF ANALYSIS WITH PROPER MISSING VALUE HANDLING")
logger.info("=" * 80)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Separate features and metadata
metadata_cols = ["track_id", "metadata_index"]
feature_cols = [col for col in df.columns if col not in metadata_cols]
X = df[feature_cols].copy()

logger.info("\nOriginal dataset: %d samples, %d features", X.shape[0], X.shape[1])

# Strategy 1: Use only features with no missing values
logger.info("\n" + "-" * 80)
logger.info("STRATEGY 1: VIF for features with NO missing values")
logger.info("-" * 80)

# Identify features with no missing values
no_missing_features = [col for col in feature_cols if X[col].isnull().sum() == 0]
logger.info("\nFeatures with no missing values: %d", len(no_missing_features))

if len(no_missing_features) > 1:
    X_no_missing = X[no_missing_features].copy()
    logger.info(
        "Using %d samples with %d features", len(X_no_missing), len(no_missing_features)
    )

    vif_scores_no_missing = []
    for i, col in enumerate(no_missing_features):
        try:
            vif = variance_inflation_factor(X_no_missing.values, i)
            vif_scores_no_missing.append(
                {"feature": col, "VIF": vif, "strategy": "no_missing"}
            )
            logger.debug(
                "  %d/%d: %s VIF=%.2f",
                i + 1,
                len(no_missing_features),
                col.ljust(50),
                vif,
            )
        except Exception as e:
            logger.error(
                "  %d/%d: %s ERROR: %s",
                i + 1,
                len(no_missing_features),
                col.ljust(50),
                str(e),
            )
            vif_scores_no_missing.append(
                {"feature": col, "VIF": np.nan, "strategy": "no_missing"}
            )

    vif_df_no_missing = pd.DataFrame(vif_scores_no_missing).sort_values(
        "VIF", ascending=False
    )
    logger.info("\n✓ VIF calculated for %d features", len(vif_scores_no_missing))

    # Show high VIF features
    high_vif = vif_df_no_missing[vif_df_no_missing["VIF"] > 10]
    if len(high_vif) > 0:
        logger.info(
            "\nFeatures with high multicollinearity (VIF > 10): %d", len(high_vif)
        )
        logger.info("\n%s", high_vif[["feature", "VIF"]].to_string(index=False))
    else:
        logger.info("\nNo features with VIF > 10")

    # Save results
    vif_df_no_missing.to_csv(f"{OUTPUT_DIR}data/vif_scores_no_missing.csv", index=False)
    logger.info("\n✓ Saved to: vif_scores_no_missing.csv")

# Strategy 2: Use median imputation for all features
logger.info("\n" + "-" * 80)
logger.info("STRATEGY 2: VIF with median imputation for ALL features")
logger.info("-" * 80)

# Exclude features with 100% missing values
valid_features = [col for col in feature_cols if X[col].isnull().sum() < len(X)]
logger.info("\nValid features (not 100%% missing): %d", len(valid_features))

if len(valid_features) > 1:
    X_imputed = X[valid_features].copy()

    # Impute missing values with median
    for col in valid_features:
        if X_imputed[col].isnull().sum() > 0:
            X_imputed[col].fillna(X_imputed[col].median(), inplace=True)

    logger.info(
        "Using %d samples with %d features (median imputed)",
        len(X_imputed),
        len(valid_features),
    )

    vif_scores_imputed = []
    for i, col in enumerate(valid_features):
        try:
            vif = variance_inflation_factor(X_imputed.values, i)
            vif_scores_imputed.append(
                {"feature": col, "VIF": vif, "strategy": "median_imputed"}
            )
            logger.debug(
                "  %d/%d: %s VIF=%.2f", i + 1, len(valid_features), col.ljust(50), vif
            )
        except Exception as e:
            logger.error(
                "  %d/%d: %s ERROR: %s",
                i + 1,
                len(valid_features),
                col.ljust(50),
                str(e),
            )
            vif_scores_imputed.append(
                {"feature": col, "VIF": np.nan, "strategy": "median_imputed"}
            )

    vif_df_imputed = pd.DataFrame(vif_scores_imputed).sort_values(
        "VIF", ascending=False
    )
    logger.info("\n✓ VIF calculated for %d features", len(vif_scores_imputed))

    # Show high VIF features
    high_vif = vif_df_imputed[vif_df_imputed["VIF"] > 10]
    if len(high_vif) > 0:
        logger.info(
            "\nFeatures with high multicollinearity (VIF > 10): %d", len(high_vif)
        )
        logger.info("\n%s", high_vif[["feature", "VIF"]].to_string(index=False))
    else:
        logger.info("\nNo features with VIF > 10")

    # Save results
    vif_df_imputed.to_csv(f"{OUTPUT_DIR}data/vif_scores.csv", index=False)
    logger.info("\n✓ Saved to: vif_scores.csv (main file)")

# Create visualization
logger.info("\n" + "-" * 80)
logger.info("CREATING VIF VISUALIZATION")
logger.info("-" * 80)

import matplotlib.pyplot as plt

if len(valid_features) > 1:
    vif_df_plot = vif_df_imputed[vif_df_imputed["VIF"].notna()].copy()
    vif_df_plot = vif_df_plot.sort_values("VIF", ascending=True)

    plt.figure(figsize=(12, max(8, len(vif_df_plot) * 0.3)))
    colors = [
        "red" if x > 10 else "orange" if x > 5 else "green" for x in vif_df_plot["VIF"]
    ]

    plt.barh(range(len(vif_df_plot)), vif_df_plot["VIF"], color=colors, alpha=0.7)
    plt.yticks(range(len(vif_df_plot)), vif_df_plot["feature"], fontsize=8)
    plt.xlabel("VIF Score", fontsize=12)
    plt.title(
        "Variance Inflation Factor (VIF) by Feature\n(Median Imputation)\nGreen: Low (<5), Orange: Medium (5-10), Red: High (>10)",
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
    plt.savefig(f"{OUTPUT_DIR}plots/vif_scores.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("✓ VIF visualization saved")

logger.info("\n" + "=" * 80)
logger.info("✅ VIF ANALYSIS COMPLETE")
logger.info("=" * 80)
