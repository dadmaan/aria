#!/usr/bin/env python3
"""
Recursive Feature Elimination (RFE) Analysis for Music Features
===============================================================
This script performs comprehensive RFE analysis on music features dataset.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import argparse
from pathlib import Path
import sys
import logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Parse arguments
parser = argparse.ArgumentParser(description="RFE Analysis for Music Features")
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
DATA_DIR = OUTPUT_DIR + "/data/"
PLOTS_DIR = OUTPUT_DIR + "/plots/"
REPORTS_DIR = OUTPUT_DIR + "/reports/"

# Create output directories
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

# Initialize logging
log_file = Path(OUTPUT_DIR) / "logs" / "rfe_analysis.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = LoggingManager(
    name="ghsom_feature_analysis.rfe_analysis",
    level=logging.INFO,
    log_file=str(log_file),
    enable_wandb=False,
).logger

# Configuration
FEATURES_TO_REMOVE_100_MISSING = [
    "muspy_drum_in_pattern_rate_triple",
    "muspy_drum_in_pattern_rate_duple",
    "muspy_drum_pattern_consistency",
]
EXCLUDE_COLUMNS = ["track_id", "metadata_index"]
CV_FOLDS = 5
STEP_SIZES = [1, 2, 5]
MIN_FEATURES_TO_SELECT = [5, 10, 15]

logger.info("=" * 80)
logger.info("RECURSIVE FEATURE ELIMINATION (RFE) ANALYSIS")
logger.info("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING AND INITIAL PROCESSING
# ============================================================================
logger.info("STEP 1: Loading and Initial Data Processing")
logger.info("-" * 80)

df = pd.read_csv(DATA_PATH)
logger.info("Initial dataset shape: %s", df.shape)
logger.info("Total samples: %d", df.shape[0])
logger.info("Total features: %d", df.shape[1])

# Remove metadata columns
df_work = df.drop(columns=EXCLUDE_COLUMNS, errors="ignore")
logger.info("After removing metadata columns: %s", df_work.shape)

# Remove features with 100% missing values
df_work = df_work.drop(columns=FEATURES_TO_REMOVE_100_MISSING, errors="ignore")
logger.info("After removing 100%% missing features: %s", df_work.shape)
logger.info("Removed features: %s", FEATURES_TO_REMOVE_100_MISSING)

# ============================================================================
# STEP 2: MISSING VALUE ANALYSIS AND HANDLING
# ============================================================================
logger.info("STEP 2: Missing Value Analysis and Handling")
logger.info("-" * 80)

missing_info = []
for col in df_work.columns:
    missing_count = df_work[col].isna().sum()
    missing_pct = (missing_count / len(df_work)) * 100
    if missing_count > 0:
        missing_info.append(
            {
                "feature": col,
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
            }
        )

missing_df = pd.DataFrame(missing_info).sort_values(
    "missing_percentage", ascending=False
)
logger.info("Features with missing values:")
logger.info("\n%s", missing_df.to_string(index=False))

# Strategy:
# - Drop features with >40% missing (theory features with 50% missing)
# - Median imputation for features with <10% missing
# - Median imputation for remaining features (pm_interval_range_min, pm_interval_range_max, pm_groove)

high_missing_features = missing_df[missing_df["missing_percentage"] > 40][
    "feature"
].tolist()
logger.info(
    "Dropping %d features with >40%% missing values:", len(high_missing_features)
)
logger.info("%s", high_missing_features)

df_work = df_work.drop(columns=high_missing_features, errors="ignore")
logger.info("Shape after dropping high-missing features: %s", df_work.shape)

# Median imputation for remaining missing values
for col in df_work.columns:
    if df_work[col].isna().any():
        median_val = df_work[col].median()
        n_missing = df_work[col].isna().sum()
        df_work[col] = df_work[col].fillna(median_val)
        logger.info(
            "Imputed %d missing values in '%s' with median: %.4f",
            n_missing,
            col,
            median_val,
        )

logger.info("Final dataset shape after missing value handling: %s", df_work.shape)

# ============================================================================
# STEP 3: TARGET VARIABLE SELECTION
# ============================================================================
logger.info("STEP 3: Target Variable Selection")
logger.info("-" * 80)

# Use muspy_pitch_entropy as target - it represents musical complexity/diversity
TARGET = "muspy_pitch_entropy"

logger.info("Selected target variable: %s", TARGET)
logger.info("Justification:")
logger.info("- Represents musical complexity and pitch diversity")
logger.info("- Well-distributed continuous variable")
logger.info("- Not highly redundant with other single features")
logger.info(
    "- Meaningful for understanding what features contribute to musical complexity"
)

if TARGET not in df_work.columns:
    raise ValueError(f"Target variable '{TARGET}' not found in dataset!")

y = df_work[TARGET].copy()
X = df_work.drop(columns=[TARGET])

logger.info("Target variable statistics:")
logger.info("  Mean: %.4f", y.mean())
logger.info("  Std: %.4f", y.std())
logger.info("  Min: %.4f", y.min())
logger.info("  Max: %.4f", y.max())
logger.info("  Missing: %d", y.isna().sum())
logger.info("Feature matrix shape: %s", X.shape)
logger.info("Number of features: %d", X.shape[1])

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================
logger.info("STEP 4: Feature Scaling")
logger.info("-" * 80)

# Use RobustScaler due to outliers identified in EDA
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

logger.info("Applied RobustScaler (due to many outliers in EDA)")
logger.info("Scaled feature matrix shape: %s", X_scaled_df.shape)

# ============================================================================
# STEP 5: BASELINE MODEL TRAINING AND COMPARISON
# ============================================================================
logger.info("STEP 5: Baseline Model Training and Comparison")
logger.info("-" * 80)

models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    "LinearRegression": LinearRegression(),
}

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

model_results = []
trained_models = {}

logger.info(
    "Training %d baseline models with %d-fold cross-validation...",
    len(models),
    CV_FOLDS,
)

for name, model in models.items():
    logger.info("Training %s...", name)

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled_df, y, cv=cv, scoring="r2", n_jobs=-1)
    cv_scores_neg_mse = cross_val_score(
        model, X_scaled_df, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    )
    cv_scores_neg_mae = cross_val_score(
        model, X_scaled_df, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )

    # Train on full dataset for feature importance
    model.fit(X_scaled_df, y)
    trained_models[name] = model

    # Predictions
    y_pred = model.predict(X_scaled_df)

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    model_results.append(
        {
            "Model": name,
            "CV_R2_Mean": cv_scores.mean(),
            "CV_R2_Std": cv_scores.std(),
            "CV_RMSE_Mean": np.sqrt(-cv_scores_neg_mse.mean()),
            "CV_RMSE_Std": np.sqrt(cv_scores_neg_mse.std()),
            "CV_MAE_Mean": -cv_scores_neg_mae.mean(),
            "CV_MAE_Std": cv_scores_neg_mae.std(),
            "Train_R2": r2,
            "Train_RMSE": rmse,
            "Train_MAE": mae,
        }
    )

    logger.info("  CV R² = %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())
    logger.info("  CV RMSE = %.4f", np.sqrt(-cv_scores_neg_mse.mean()))
    logger.info("  CV MAE = %.4f", -cv_scores_neg_mae.mean())

results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values("CV_R2_Mean", ascending=False)

logger.info("Model Comparison Summary:")
logger.info("\n%s", results_df.to_string(index=False))

# Save model comparison
results_df.to_csv(DATA_DIR + "model_comparison.csv", index=False)
logger.info("Saved: %smodel_comparison.csv", DATA_DIR)

# Select best model for RFE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_cv_r2 = results_df.iloc[0]["CV_R2_Mean"]

logger.info("Best model for RFE: %s (CV R² = %.4f)", best_model_name, best_cv_r2)

# ============================================================================
# STEP 6: RFE WITH CROSS-VALIDATION (RFECV)
# ============================================================================
logger.info("STEP 6: Recursive Feature Elimination with Cross-Validation (RFECV)")
logger.info("-" * 80)

# Test different configurations
rfecv_results = []
best_rfecv = None
best_rfecv_score = -np.inf
best_rfecv_config = None

logger.info("Testing RFECV with different configurations...")
logger.info("Step sizes: %s", STEP_SIZES)
logger.info("Min features to select: %s", MIN_FEATURES_TO_SELECT)

for step in STEP_SIZES:
    for min_features in MIN_FEATURES_TO_SELECT:
        logger.info("Testing: step=%d, min_features_to_select=%d", step, min_features)

        # Create fresh model instance for RFECV
        if best_model_name == "RandomForest":
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif best_model_name == "GradientBoosting":
            estimator = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            )
        else:
            estimator = LinearRegression()

        rfecv = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring="r2",
            min_features_to_select=min_features,
            n_jobs=-1,
        )

        rfecv.fit(X_scaled_df, y)

        optimal_features = rfecv.n_features_
        max_score = rfecv.cv_results_["mean_test_score"].max()

        rfecv_results.append(
            {
                "step": step,
                "min_features_to_select": min_features,
                "optimal_n_features": optimal_features,
                "max_cv_score": max_score,
            }
        )

        logger.info(
            "  Optimal features: %d, Max CV score: %.4f", optimal_features, max_score
        )

        if max_score > best_rfecv_score:
            best_rfecv_score = max_score
            best_rfecv = rfecv
            best_rfecv_config = {"step": step, "min_features": min_features}

logger.info(
    "Best RFECV configuration: step=%d, min_features=%d",
    best_rfecv_config["step"],
    best_rfecv_config["min_features"],
)
logger.info("Optimal number of features: %d", best_rfecv.n_features_)
logger.info("Best CV score: %.4f", best_rfecv_score)

# Save RFECV configuration results
rfecv_config_df = pd.DataFrame(rfecv_results)
rfecv_config_df.to_csv(DATA_DIR + "rfecv_configurations.csv", index=False)
logger.info("Saved: %srfecv_configurations.csv", DATA_DIR)

# ============================================================================
# STEP 7: PLOT RFECV CROSS-VALIDATION SCORES
# ============================================================================
logger.info("STEP 7: Creating RFECV Visualization")
logger.info("-" * 80)

fig, ax = plt.subplots(figsize=(12, 6))

n_features_range = range(1, len(best_rfecv.cv_results_["mean_test_score"]) + 1)
mean_scores = best_rfecv.cv_results_["mean_test_score"]
std_scores = best_rfecv.cv_results_["std_test_score"]

ax.plot(
    n_features_range,
    mean_scores,
    "o-",
    linewidth=2,
    markersize=6,
    label="Mean CV Score",
    color="#2E86AB",
)
ax.fill_between(
    n_features_range,
    mean_scores - std_scores,
    mean_scores + std_scores,
    alpha=0.2,
    color="#2E86AB",
    label="±1 Std Dev",
)

# Mark optimal point
optimal_idx = np.argmax(mean_scores)
optimal_n_features = n_features_range[optimal_idx]
ax.axvline(
    x=optimal_n_features,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Optimal: {optimal_n_features} features",
)
ax.plot(
    optimal_n_features,
    mean_scores[optimal_idx],
    "r*",
    markersize=20,
    label=f"Best Score: {mean_scores[optimal_idx]:.4f}",
)

ax.set_xlabel("Number of Features Selected", fontsize=12, fontweight="bold")
ax.set_ylabel("Cross-Validation R² Score", fontsize=12, fontweight="bold")
ax.set_title(
    f"RFECV: {best_model_name} - Cross-Validation Scores vs Number of Features\n"
    f"Target: {TARGET}",
    fontsize=14,
    fontweight="bold",
)
ax.legend(loc="best", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig(PLOTS_DIR + "rfe_cv_scores.png", dpi=300, bbox_inches="tight")
plt.close()
logger.info("Saved: %srfe_cv_scores.png", PLOTS_DIR)

# ============================================================================
# STEP 8: FEATURE RANKINGS FROM RFE
# ============================================================================
logger.info("STEP 8: Extracting Feature Rankings from RFE")
logger.info("-" * 80)

feature_rankings = pd.DataFrame(
    {
        "feature": X.columns,
        "rfe_rank": best_rfecv.ranking_,
        "rfe_selected": best_rfecv.support_,
    }
)

feature_rankings = feature_rankings.sort_values("rfe_rank")

logger.info("RFE Feature Rankings (1 = most important):")
logger.info("\n%s", feature_rankings.to_string(index=False))

# Save rankings
feature_rankings.to_csv(DATA_DIR + "rfe_rankings.csv", index=False)
logger.info("Saved: %srfe_rankings.csv", DATA_DIR)

# Top features
top_features = feature_rankings[feature_rankings["rfe_selected"]]["feature"].tolist()
logger.info("Top %d features selected by RFE:", len(top_features))
for i, feat in enumerate(top_features, 1):
    logger.info("  %d. %s", i, feat)

# ============================================================================
# STEP 9: ELIMINATION ORDER ANALYSIS
# ============================================================================
logger.info("STEP 9: Feature Elimination Order Analysis")
logger.info("-" * 80)

# Create elimination order dataframe
elimination_order = []
for rank in range(1, feature_rankings["rfe_rank"].max() + 1):
    features_at_rank = feature_rankings[feature_rankings["rfe_rank"] == rank][
        "feature"
    ].tolist()

    # Get the score at this step if available
    n_features_remaining = len(feature_rankings[feature_rankings["rfe_rank"] >= rank])
    if n_features_remaining <= len(best_rfecv.cv_results_["mean_test_score"]):
        score = best_rfecv.cv_results_["mean_test_score"][n_features_remaining - 1]
    else:
        score = np.nan

    elimination_order.append(
        {
            "elimination_step": rank,
            "features_eliminated": (
                ", ".join(features_at_rank) if rank > 1 else "None (selected)"
            ),
            "n_features_remaining": n_features_remaining,
            "cv_score": score,
        }
    )

elimination_df = pd.DataFrame(elimination_order)
elimination_df.to_csv(DATA_DIR + "rfe_elimination_order.csv", index=False)
logger.info("Saved: %srfe_elimination_order.csv", DATA_DIR)

logger.info("Elimination order (first 10 steps):")
logger.info("\n%s", elimination_df.head(10).to_string(index=False))

# ============================================================================
# STEP 10: FEATURE IMPORTANCE FROM MODELS
# ============================================================================
logger.info("STEP 10: Extracting Feature Importances from Models")
logger.info("-" * 80)

importance_dict = {"feature": X.columns.tolist()}

# Tree-based model importances
for name, model in trained_models.items():
    if hasattr(model, "feature_importances_"):
        importance_dict[f"{name}_importance"] = model.feature_importances_

# Permutation importance for all models
logger.info("Calculating permutation importance (this may take a moment)...")
for name, model in trained_models.items():
    logger.info("  %s...", name)
    perm_importance = permutation_importance(
        model, X_scaled_df, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_dict[f"{name}_permutation_importance"] = perm_importance.importances_mean
    importance_dict[f"{name}_permutation_std"] = perm_importance.importances_std

# Combine with RFE rankings
importance_df = pd.DataFrame(importance_dict)
importance_df = importance_df.merge(
    feature_rankings[["feature", "rfe_rank", "rfe_selected"]], on="feature"
)

# Sort by RFE rank
importance_df = importance_df.sort_values("rfe_rank")

logger.info("Combined Feature Importance (top 15 features):")
logger.info("\n%s", importance_df.head(15).to_string(index=False))

# Save combined importance
importance_df.to_csv(DATA_DIR + "feature_importance_combined.csv", index=False)
logger.info("Saved: %sfeature_importance_combined.csv", DATA_DIR)

# ============================================================================
# STEP 11: FEATURE IMPORTANCE COMPARISON PLOT
# ============================================================================
logger.info("STEP 11: Creating Feature Importance Comparison Plot")
logger.info("-" * 80)

# Select top 15 features by RFE
top_15_features = importance_df.head(15)

# Prepare data for plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: RFE Rank (lower is better)
ax = axes[0, 0]
top_15_sorted = top_15_features.sort_values("rfe_rank")
colors = ["green" if x else "red" for x in top_15_sorted["rfe_selected"]]
ax.barh(range(len(top_15_sorted)), top_15_sorted["rfe_rank"], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_15_sorted)))
ax.set_yticklabels(top_15_sorted["feature"], fontsize=9)
ax.set_xlabel("RFE Rank (1 = best)", fontsize=10, fontweight="bold")
ax.set_title("RFE Feature Rankings", fontsize=12, fontweight="bold")
ax.invert_xaxis()  # Lower rank is better
ax.grid(True, alpha=0.3, axis="x")

# Plot 2: RandomForest Importance (if available)
ax = axes[0, 1]
if "RandomForest_importance" in importance_df.columns:
    top_15_sorted = top_15_features.sort_values("RandomForest_importance")
    ax.barh(
        range(len(top_15_sorted)),
        top_15_sorted["RandomForest_importance"],
        color="#2E86AB",
        alpha=0.7,
    )
    ax.set_yticks(range(len(top_15_sorted)))
    ax.set_yticklabels(top_15_sorted["feature"], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=10, fontweight="bold")
    ax.set_title("Random Forest Feature Importance", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

# Plot 3: GradientBoosting Importance (if available)
ax = axes[1, 0]
if "GradientBoosting_importance" in importance_df.columns:
    top_15_sorted = top_15_features.sort_values("GradientBoosting_importance")
    ax.barh(
        range(len(top_15_sorted)),
        top_15_sorted["GradientBoosting_importance"],
        color="#A23B72",
        alpha=0.7,
    )
    ax.set_yticks(range(len(top_15_sorted)))
    ax.set_yticklabels(top_15_sorted["feature"], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=10, fontweight="bold")
    ax.set_title("Gradient Boosting Feature Importance", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

# Plot 4: Permutation Importance (from best model)
ax = axes[1, 1]
perm_col = f"{best_model_name}_permutation_importance"
if perm_col in importance_df.columns:
    top_15_sorted = top_15_features.sort_values(perm_col)
    ax.barh(
        range(len(top_15_sorted)), top_15_sorted[perm_col], color="#F18F01", alpha=0.7
    )
    ax.set_yticks(range(len(top_15_sorted)))
    ax.set_yticklabels(top_15_sorted["feature"], fontsize=9)
    ax.set_xlabel("Permutation Importance", fontsize=10, fontweight="bold")
    ax.set_title(
        f"{best_model_name} Permutation Importance", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="x")

plt.suptitle(
    f"Feature Importance Comparison - Top 15 Features\nTarget: {TARGET}",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig(
    PLOTS_DIR + "feature_importance_comparison.png", dpi=300, bbox_inches="tight"
)
plt.close()
logger.info("Saved: %sfeature_importance_comparison.png", PLOTS_DIR)

# ============================================================================
# STEP 12: COMPREHENSIVE SUMMARY REPORT
# ============================================================================
logger.info("STEP 12: Creating Comprehensive Summary Report")
logger.info("-" * 80)

summary = {
    "analysis_info": {
        "date": "2025-11-13",
        "target_variable": TARGET,
        "target_justification": "Represents musical complexity and pitch diversity. Well-distributed continuous variable that captures an important aspect of musical structure.",
        "total_samples": len(y),
        "initial_features": len(X.columns)
        + len(FEATURES_TO_REMOVE_100_MISSING)
        + len(high_missing_features),
        "features_after_preprocessing": X.shape[1],
        "random_state": RANDOM_STATE,
    },
    "data_preprocessing": {
        "features_removed_100_missing": FEATURES_TO_REMOVE_100_MISSING,
        "features_removed_high_missing": high_missing_features,
        "total_features_removed": len(FEATURES_TO_REMOVE_100_MISSING)
        + len(high_missing_features),
        "missing_value_strategy": "Median imputation for features with <40% missing",
        "scaling_method": "RobustScaler (due to outliers)",
        "final_feature_count": X.shape[1],
    },
    "target_statistics": {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "median": float(y.median()),
    },
    "baseline_models": {
        "models_tested": list(models.keys()),
        "cv_folds": CV_FOLDS,
        "best_model": best_model_name,
        "best_cv_r2": float(best_cv_r2),
        "model_comparison": results_df.to_dict("records"),
    },
    "rfecv_analysis": {
        "best_configuration": {
            "step": int(best_rfecv_config["step"]),
            "min_features": int(best_rfecv_config["min_features"]),
        },
        "optimal_n_features": int(best_rfecv.n_features_),
        "best_cv_score": float(best_rfecv_score),
        "configurations_tested": [
            {
                k: (
                    int(v)
                    if isinstance(v, (np.integer, np.int64))
                    else float(v) if isinstance(v, (np.floating, np.float64)) else v
                )
                for k, v in config.items()
            }
            for config in rfecv_results
        ],
        "improvement_over_baseline": float(best_rfecv_score - best_cv_r2),
    },
    "feature_selection_results": {
        "selected_features": top_features,
        "n_selected_features": len(top_features),
        "selection_rate": len(top_features) / X.shape[1],
        "top_15_features": feature_rankings.head(15)["feature"].tolist(),
    },
    "feature_importance_methods": {
        "methods_used": [
            "RFE Ranking",
            "Tree-based Feature Importance (RF, GB)",
            "Permutation Importance (all models)",
        ],
        "consensus_top_10": feature_rankings.head(10)["feature"].tolist(),
    },
    "output_files": {
        "data_files": [
            "rfe_rankings.csv",
            "rfe_elimination_order.csv",
            "model_comparison.csv",
            "feature_importance_combined.csv",
            "rfecv_configurations.csv",
        ],
        "plot_files": ["rfe_cv_scores.png", "feature_importance_comparison.png"],
        "report_files": ["rfe_summary.json"],
    },
}

# Save summary
with open(REPORTS_DIR + "rfe_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

logger.info("Saved: %srfe_summary.json", REPORTS_DIR)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
logger.info("=" * 80)
logger.info("ANALYSIS COMPLETE - SUMMARY")
logger.info("=" * 80)

logger.info("1. TARGET VARIABLE:")
logger.info("   - Variable: %s", TARGET)
logger.info("   - Justification: Musical complexity and pitch diversity measure")

logger.info("2. DATA PREPROCESSING:")
logger.info(
    "   - Initial features: %d",
    len(X.columns) + len(FEATURES_TO_REMOVE_100_MISSING) + len(high_missing_features),
)
logger.info("   - Removed (100%% missing): %d", len(FEATURES_TO_REMOVE_100_MISSING))
logger.info("   - Removed (>40%% missing): %d", len(high_missing_features))
logger.info("   - Final features: %d", X.shape[1])
logger.info("   - Scaling: RobustScaler")

logger.info("3. BASELINE MODEL COMPARISON:")
for idx, row in results_df.iterrows():
    logger.info(
        "   - %s: CV R² = %.4f (±%.4f)",
        row["Model"],
        row["CV_R2_Mean"],
        row["CV_R2_Std"],
    )
logger.info("   - Best: %s", best_model_name)

logger.info("4. RFECV RESULTS:")
logger.info("   - Optimal number of features: %d", best_rfecv.n_features_)
logger.info("   - Best CV R² score: %.4f", best_rfecv_score)
logger.info(
    "   - Configuration: step=%d, min_features=%d",
    best_rfecv_config["step"],
    best_rfecv_config["min_features"],
)

logger.info("5. TOP 15 FEATURES (by RFE rank):")
for i, feat in enumerate(feature_rankings.head(15)["feature"], 1):
    logger.info("   %2d. %s", i, feat)

logger.info("6. OUTPUT FILES:")
logger.info("   Data files:")
for f in summary["output_files"]["data_files"]:
    logger.info("     - %s%s", DATA_DIR, f)
logger.info("   Plot files:")
for f in summary["output_files"]["plot_files"]:
    logger.info("     - %s%s", PLOTS_DIR, f)
logger.info("   Report files:")
for f in summary["output_files"]["report_files"]:
    logger.info("     - %s%s", REPORTS_DIR, f)

logger.info("=" * 80)
logger.info("All analysis completed successfully!")
logger.info("=" * 80)
