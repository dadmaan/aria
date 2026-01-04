#!/usr/bin/env python3
"""
Reproducible Feature Importance Analysis Script

This script performs comprehensive feature analysis on the COMMU bass dataset using:
- Exploratory Data Analysis (correlation, VIF, mutual information)
- Recursive Feature Elimination (RFE) with cross-validation
- SHAP (SHapley Additive exPlanations) values
- Consensus ranking from multiple methods

Requirements:
    pip install pandas numpy scipy scikit-learn shap statsmodels matplotlib seaborn

Usage:
    python run_feature_analysis.py --data_path <path> --output_dir <dir>
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def setup_output_dirs(output_dir):
    """Create output directory structure."""
    output_path = Path(output_dir)
    dirs = {
        "data": output_path / "data",
        "plots": output_path / "plots",
        "reports": output_path / "reports",
        "scripts": output_path / "scripts",
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_and_preprocess_data(data_path, logger):
    """Load and preprocess the feature data."""
    logger.info("Loading data...")
    df = pd.read_csv(data_path)

    # Exclude metadata columns
    exclude_cols = ["track_id", "metadata_index"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove features with 100% missing values
    missing_pct = df[feature_cols].isnull().sum() / len(df) * 100
    features_to_remove = missing_pct[missing_pct == 100].index.tolist()

    logger.info(
        "Removing %d features with 100%% missing values", len(features_to_remove)
    )
    feature_cols = [col for col in feature_cols if col not in features_to_remove]

    # Get feature data
    X = df[feature_cols].copy()

    logger.info("Loaded %d samples with %d features", len(df), len(feature_cols))
    logger.info("Missing values: %d total", X.isnull().sum().sum())

    return X, feature_cols


def perform_eda(X, output_dirs, logger):
    """Perform exploratory data analysis."""
    logger.info("=" * 60)
    logger.info("Exploratory Data Analysis")
    logger.info("=" * 60)

    results = {}

    # 1. Basic statistics
    logger.info("Computing basic statistics...")
    stats_df = X.describe().T
    stats_df["missing_pct"] = X.isnull().sum() / len(X) * 100
    stats_df["skewness"] = X.skew()
    stats_df["kurtosis"] = X.kurtosis()
    stats_df.to_csv(output_dirs["data"] / "distribution_stats.csv")
    results["statistics"] = stats_df.to_dict()

    # 2. Correlation matrix
    logger.info("Computing correlation matrix...")
    corr_matrix = X.corr()
    corr_matrix.to_csv(output_dirs["data"] / "correlation_matrix.csv")

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append(
                    {
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    pd.DataFrame(high_corr_pairs).to_csv(
        output_dirs["data"] / "highly_correlated_pairs.csv", index=False
    )
    results["high_corr_pairs_count"] = len(high_corr_pairs)

    # 3. VIF scores
    logger.info("Computing VIF scores...")
    X_filled = X.fillna(X.median())
    vif_data = []

    for i, col in enumerate(X_filled.columns):
        try:
            vif = variance_inflation_factor(X_filled.values, i)
            vif_data.append({"feature": col, "vif": vif})
        except Exception as e:
            logger.warning("Could not compute VIF for %s: %s", col, e)
            vif_data.append({"feature": col, "vif": np.nan})

    vif_df = pd.DataFrame(vif_data)
    vif_df.to_csv(output_dirs["data"] / "vif_scores.csv", index=False)
    results["high_vif_count"] = (vif_df["vif"] > 10).sum()

    # 4. Visualization - Correlation heatmap
    logger.info("Creating correlation heatmap...")
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )
    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(
        output_dirs["plots"] / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(
        "EDA complete. Found %d highly correlated pairs, %d features with VIF > 10",
        len(high_corr_pairs),
        results["high_vif_count"],
    )

    return results


def perform_rfe_analysis(X, output_dirs, logger):
    """Perform Recursive Feature Elimination analysis."""
    logger.info("=" * 60)
    logger.info("RFE Analysis")
    logger.info("=" * 60)

    # Prepare data
    X_filled = X.fillna(X.median())

    # Select target (use a feature as target for demonstration)
    # In real scenarios, you might have an external target
    target_feature = (
        "muspy_pitch_entropy"
        if "muspy_pitch_entropy" in X_filled.columns
        else X_filled.columns[0]
    )

    feature_cols = [col for col in X_filled.columns if col != target_feature]
    X_features = X_filled[feature_cols]
    y = X_filled[target_feature]

    logger.info("Target variable: %s", target_feature)
    logger.info("Features: %d", len(feature_cols))

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_features)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # Train baseline model
    logger.info("Training baseline GradientBoosting model...")
    model = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2", n_jobs=-1)

    logger.info("Baseline CV R²: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # Perform RFECV
    logger.info("Performing RFECV...")
    rfecv = RFECV(
        estimator=model, step=1, cv=5, scoring="r2", min_features_to_select=5, n_jobs=-1
    )
    rfecv.fit(X_scaled, y)

    logger.info("Optimal number of features: %d", rfecv.n_features_)

    # Get rankings
    rfe_results = pd.DataFrame(
        {
            "feature": feature_cols,
            "rfe_rank": rfecv.ranking_,
            "selected": rfecv.support_,
        }
    ).sort_values("rfe_rank")

    rfe_results.to_csv(output_dirs["data"] / "rfe_rankings.csv", index=False)

    # Plot CV scores
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
        rfecv.cv_results_["mean_test_score"],
        marker="o",
    )
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Cross-validation R² Score", fontsize=12)
    plt.title("RFE Cross-validation Scores", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(
        x=rfecv.n_features_,
        color="r",
        linestyle="--",
        label=f"Optimal: {rfecv.n_features_} features",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dirs["plots"] / "rfe_cv_scores.png", dpi=300)
    plt.close()

    results = {
        "target": target_feature,
        "n_features": len(feature_cols),
        "optimal_features": int(rfecv.n_features_),
        "baseline_cv_r2": float(cv_scores.mean()),
        "optimal_cv_r2": float(rfecv.cv_results_["mean_test_score"].max()),
        "top_10_features": rfe_results.head(10)["feature"].tolist(),
    }

    logger.info(
        "Top 5 features: %s", ", ".join(rfe_results.head(5)["feature"].tolist())
    )

    return results, rfe_results


def perform_shap_analysis(X, output_dirs, logger):
    """Perform SHAP analysis."""
    logger.info("=" * 60)
    logger.info("SHAP Analysis")
    logger.info("=" * 60)

    # Prepare data
    X_filled = X.fillna(X.median())

    # Select target
    target_feature = (
        "pm_energy" if "pm_energy" in X_filled.columns else X_filled.columns[0]
    )
    feature_cols = [col for col in X_filled.columns if col != target_feature]
    X_features = X_filled[feature_cols]
    y = X_filled[target_feature]

    logger.info("Target variable: %s", target_feature)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model (Ridge for interpretability)
    logger.info("Training Ridge model...")
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    logger.info("Test R²: %.4f", r2)

    # Calculate SHAP values
    logger.info("Computing SHAP values...")
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    # Save SHAP values
    np.save(output_dirs["data"] / "shap_values.npy", shap_values)

    # Calculate global importance
    shap_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(output_dirs["data"] / "shap_importance.csv", index=False)

    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_cols,
        show=False,
        plot_type="bar",
    )
    plt.title("SHAP Feature Importance", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(
        output_dirs["plots"] / "shap_importance_bar.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    results = {
        "target": target_feature,
        "test_r2": float(r2),
        "n_samples_test": len(X_test),
        "top_10_features": shap_importance.head(10)["feature"].tolist(),
    }

    logger.info(
        "Top 5 features: %s", ", ".join(shap_importance.head(5)["feature"].tolist())
    )

    return results, shap_importance


def create_consensus_ranking(rfe_results, shap_results, X, output_dirs, logger):
    """Create consensus feature ranking."""
    logger.info("=" * 60)
    logger.info("Creating Consensus Ranking")
    logger.info("=" * 60)

    # Normalize RFE ranks (1 = best, higher = worse) to 0-100 scale
    max_rank = rfe_results["rfe_rank"].max()
    rfe_results["rfe_score"] = 100 * (1 - (rfe_results["rfe_rank"] - 1) / max_rank)

    # Normalize SHAP importance to 0-100 scale
    max_shap = shap_results["mean_abs_shap"].max()
    shap_results["shap_score"] = 100 * (shap_results["mean_abs_shap"] / max_shap)

    # Merge
    consensus = pd.merge(
        rfe_results[["feature", "rfe_rank", "rfe_score"]],
        shap_results[["feature", "mean_abs_shap", "shap_score"]],
        on="feature",
        how="outer",
    )

    # Fill NaN with 0 (features not in one of the analyses)
    consensus = consensus.fillna(0)

    # Calculate weighted consensus score
    # RFE: 50%, SHAP: 50%
    consensus["consensus_score"] = (
        0.5 * consensus["rfe_score"] + 0.5 * consensus["shap_score"]
    )

    # Rank by consensus
    consensus = consensus.sort_values("consensus_score", ascending=False)
    consensus["consensus_rank"] = range(1, len(consensus) + 1)

    consensus.to_csv(output_dirs["data"] / "consensus_ranking.csv", index=False)

    # Calculate method agreement
    from scipy.stats import spearmanr

    # Filter to features present in both
    both = consensus[(consensus["rfe_rank"] > 0) & (consensus["mean_abs_shap"] > 0)]

    if len(both) > 3:
        corr, p_value = spearmanr(both["rfe_rank"], both["shap_score"])
        logger.info("RFE vs SHAP rank correlation: ρ = %.3f (p = %.4f)", corr, p_value)

    # Create comparison plot
    top_15 = consensus.head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(top_15))
    width = 0.35

    ax.barh(x - width / 2, top_15["rfe_score"], width, label="RFE Score", alpha=0.8)
    ax.barh(x + width / 2, top_15["shap_score"], width, label="SHAP Score", alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(top_15["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Score (0-100)", fontsize=12)
    ax.set_title("Top 15 Features: RFE vs SHAP Comparison", fontsize=14)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dirs["plots"] / "consensus_ranking_comparison.png", dpi=300)
    plt.close()

    results = {
        "top_10_consensus": top_15.head(10)["feature"].tolist(),
        "consensus_scores": top_15.head(10)["consensus_score"].tolist(),
    }

    logger.info("Top 10 consensus features:")
    for i, row in top_15.head(10).iterrows():
        logger.info(
            "  %.0f. %s (score: %.1f)",
            row["consensus_rank"],
            row["feature"],
            row["consensus_score"],
        )

    return results, consensus


def main():
    parser = argparse.ArgumentParser(description="Run feature importance analysis")
    parser.add_argument(
        "--data_path",
        type=str,
        default="artifacts/features/raw/commu_full/features_numeric.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory path (default: outputs/feature_importance_analysis/{dataset_name}_{timestamp})",
    )

    args = parser.parse_args()

    # Extract dataset name from data path
    data_path = Path(args.data_path)
    dataset_name = data_path.parent.name  # Gets 'commu_full' from path

    # Set default output directory with dataset name if not provided
    if args.output_dir is None:
        args.output_dir = f'outputs/feature_importance_analysis/{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # Initialize logger
    log_file = Path(args.output_dir) / "logs" / "feature_analysis.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = LoggingManager(
        name="ghsom_feature_analysis.quick_feature_analysis",
        level=logging.INFO,
        log_file=str(log_file),
    )

    logger.info("=" * 60)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 60)
    logger.info("Data: %s", args.data_path)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 60)

    # Setup output directories
    output_dirs = setup_output_dirs(args.output_dir)

    # Load data
    X, feature_cols = load_and_preprocess_data(args.data_path, logger)

    # Perform analyses
    eda_results = perform_eda(X, output_dirs, logger)
    rfe_results_dict, rfe_df = perform_rfe_analysis(X, output_dirs, logger)
    shap_results_dict, shap_df = perform_shap_analysis(X, output_dirs, logger)
    consensus_results, consensus_df = create_consensus_ranking(
        rfe_df, shap_df, X, output_dirs, logger
    )

    # Save final summary
    final_summary = {
        "analysis_date": datetime.now().isoformat(),
        "data_path": args.data_path,
        "n_samples": len(X),
        "n_features_original": len(feature_cols),
        "eda": eda_results,
        "rfe": rfe_results_dict,
        "shap": shap_results_dict,
        "consensus": consensus_results,
    }

    with open(output_dirs["reports"] / "analysis_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2, cls=NumpyEncoder)

    logger.info("=" * 60)
    logger.info("Analysis Complete!")
    logger.info("=" * 60)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Data files: %d files", len(list(output_dirs["data"].glob("*"))))
    logger.info("Plots: %d files", len(list(output_dirs["plots"].glob("*"))))
    logger.info("Reports: %d files", len(list(output_dirs["reports"].glob("*"))))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
