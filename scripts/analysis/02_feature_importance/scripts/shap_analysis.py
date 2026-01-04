#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis
Comprehensive interpretability analysis using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import json
from pathlib import Path
import warnings
import argparse
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Parse arguments
parser = argparse.ArgumentParser(description="SHAP Feature Importance Analysis")
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

# Initialize logging
log_file = Path(args.output_dir) / "logs" / "shap_analysis.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger = LoggingManager(
    name="ghsom_feature_analysis.shap_analysis", log_file=str(log_file)
)

logger.info("=" * 80)
logger.info("SHAP FEATURE IMPORTANCE ANALYSIS")
logger.info("=" * 80)
logger.info("Data: %s", args.data_path)
logger.info("Output: %s", args.output_dir)
logger.info("=" * 80)


class SHAPAnalyzer:
    """SHAP-based feature importance and interpretability analyzer"""

    def __init__(self, data_path, output_dir, logger):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.data_dir = self.output_dir / "data"
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.shap_values = None
        self.shap_explainer = None
        self.results = {}

    def load_and_prepare_data(self):
        """Load data and handle preprocessing"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: DATA LOADING AND PREPARATION")
        self.logger.info("=" * 80)

        # Load data
        self.df = pd.read_csv(self.data_path)
        self.logger.info(
            "Loaded dataset: %d samples, %d features",
            self.df.shape[0],
            self.df.shape[1],
        )

        # Features to exclude
        exclude_features = ["track_id", "metadata_index"]

        # Features to remove (100% missing - drum features)
        remove_features = [
            "muspy_drum_in_pattern_rate_triple",
            "muspy_drum_in_pattern_rate_duple",
            "muspy_drum_pattern_consistency",
        ]

        self.logger.info("Excluding features: %s", exclude_features)
        self.logger.info("Removing features (100%% missing): %s", remove_features)

        # Drop excluded and removed features
        features_to_drop = exclude_features + remove_features
        self.df = self.df.drop(
            columns=[col for col in features_to_drop if col in self.df.columns]
        )

        self.logger.info("After removal: %d features remaining", self.df.shape[1])

        # Check missing values
        missing_info = self.df.isnull().sum()
        missing_pct = (missing_info / len(self.df)) * 100
        features_with_missing = missing_pct[missing_pct > 0].sort_values(
            ascending=False
        )

        if len(features_with_missing) > 0:
            self.logger.info("Features with missing values:")
            for feat, pct in features_with_missing.items():
                self.logger.info(
                    "  %s: %.2f%% (%d samples)", feat, pct, missing_info[feat]
                )

        # Handle missing values
        self.logger.info("Handling missing values...")

        # Theory features with ~50% missing - impute with median
        theory_features = [col for col in self.df.columns if col.startswith("theory_")]
        for col in theory_features:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                self.logger.debug("  Imputed %s with median: %.4f", col, median_val)

        # Other features with <10% missing - impute with median
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                missing_pct_col = (self.df[col].isnull().sum() / len(self.df)) * 100
                if missing_pct_col < 10:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    self.logger.debug("  Imputed %s with median: %.4f", col, median_val)

        # Verify no missing values remain
        remaining_missing = self.df.isnull().sum().sum()
        self.logger.info("Remaining missing values: %d", remaining_missing)

        # Use pm_energy as target (continuous variable for regression)
        target_col = "pm_energy"
        self.logger.info("Using target variable: %s", target_col)

        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Separate features and target
        y = self.df[target_col].values
        X = self.df.drop(columns=[target_col])

        self.feature_names = X.columns.tolist()
        self.logger.info("Final feature set: %d features", len(self.feature_names))
        self.logger.info("Target variable range: [%.2f, %.2f]", y.min(), y.max())
        self.logger.info("Target variable mean: %.2f, std: %.2f", y.mean(), y.std())

        # Train-test split (80/20)
        self.logger.info("Splitting data: 80%% train, 20%% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

        self.logger.info("Train set: %d samples", X_train.shape[0])
        self.logger.info("Test set: %d samples", X_test.shape[0])

        # Scale features
        self.logger.info("Applying StandardScaler...")
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_names,
            index=X_train.index,
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_names,
            index=X_test.index,
        )
        self.y_train = y_train
        self.y_test = y_test

        self.results["data_info"] = {
            "total_samples": len(self.df),
            "train_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "num_features": len(self.feature_names),
            "target_variable": target_col,
            "features_removed": remove_features,
            "features_excluded": exclude_features,
            "final_features": self.feature_names,
        }

        self.logger.info("Data preparation completed successfully!")

    def train_and_select_model(self):
        """Train multiple models and select the best one"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: MODEL TRAINING AND SELECTION")
        self.logger.info("=" * 80)

        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "Lasso": Lasso(alpha=1.0, random_state=RANDOM_STATE, max_iter=10000),
        }

        # Try XGBoost if available
        try:
            import xgboost as xgb

            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            self.logger.info("XGBoost is available and will be included in training")
        except ImportError:
            self.logger.info("XGBoost not available, skipping...")

        model_results = {}

        self.logger.info(
            "Training %d models with 5-fold cross-validation...", len(models)
        )

        for name, model in models.items():
            self.logger.info("%s:", name)

            # Cross-validation
            cv_scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse = np.sqrt(-cv_scores)

            # Train on full training set
            model.fit(self.X_train, self.y_train)

            # Evaluate on test set
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)

            model_results[name] = {
                "model": model,
                "cv_rmse_mean": cv_rmse.mean(),
                "cv_rmse_std": cv_rmse.std(),
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "train_mae": train_mae,
                "test_mae": test_mae,
            }

            self.logger.info(
                "  CV RMSE: %.4f (+/- %.4f)", cv_rmse.mean(), cv_rmse.std()
            )
            self.logger.info("  Train RMSE: %.4f, R²: %.4f", train_rmse, train_r2)
            self.logger.info("  Test RMSE: %.4f, R²: %.4f", test_rmse, test_r2)
            self.logger.info("  Test MAE: %.4f", test_mae)

        # Select best model based on test R²
        best_name = max(model_results.keys(), key=lambda k: model_results[k]["test_r2"])
        self.best_model = model_results[best_name]["model"]
        self.best_model_name = best_name

        self.logger.info("=" * 80)
        self.logger.info("BEST MODEL: %s", best_name)
        self.logger.info("Test R²: %.4f", model_results[best_name]["test_r2"])
        self.logger.info("Test RMSE: %.4f", model_results[best_name]["test_rmse"])
        self.logger.info("=" * 80)

        # Store results
        self.results["model_comparison"] = {
            name: {k: v for k, v in results.items() if k != "model"}
            for name, results in model_results.items()
        }
        self.results["best_model"] = {
            "name": best_name,
            "test_r2": model_results[best_name]["test_r2"],
            "test_rmse": model_results[best_name]["test_rmse"],
            "test_mae": model_results[best_name]["test_mae"],
        }

    def calculate_shap_values(self):
        """Calculate SHAP values using appropriate explainer"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: SHAP VALUE CALCULATION")
        self.logger.info("=" * 80)

        # Select appropriate explainer based on model type
        if self.best_model_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
            self.logger.info("Using TreeExplainer for %s", self.best_model_name)
            explainer_type = "TreeExplainer"
            self.shap_explainer = shap.TreeExplainer(self.best_model)
            self.shap_values = self.shap_explainer.shap_values(self.X_test)

        elif self.best_model_name in ["Ridge", "Lasso"]:
            self.logger.info("Using LinearExplainer for %s", self.best_model_name)
            explainer_type = "LinearExplainer"
            self.shap_explainer = shap.LinearExplainer(self.best_model, self.X_train)
            self.shap_values = self.shap_explainer.shap_values(self.X_test)

        else:
            self.logger.info("Using KernelExplainer for %s", self.best_model_name)
            explainer_type = "KernelExplainer"
            # Sample background data for efficiency
            background = shap.sample(self.X_train, 100, random_state=RANDOM_STATE)
            self.shap_explainer = shap.KernelExplainer(
                self.best_model.predict, background
            )
            self.shap_values = self.shap_explainer.shap_values(self.X_test)

        self.logger.info("SHAP values shape: %s", self.shap_values.shape)
        self.logger.info("Expected value: %s", self.shap_explainer.expected_value)

        # Save SHAP values
        shap_values_path = self.data_dir / "shap_values.npy"
        np.save(shap_values_path, self.shap_values)
        self.logger.info("Saved SHAP values to: %s", shap_values_path)

        self.results["shap_info"] = {
            "explainer_type": explainer_type,
            "expected_value": float(self.shap_explainer.expected_value),
            "shap_values_shape": list(self.shap_values.shape),
            "test_samples": self.shap_values.shape[0],
            "num_features": self.shap_values.shape[1],
        }

    def analyze_global_importance(self):
        """Calculate global feature importance from SHAP values"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: GLOBAL FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("=" * 80)

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": mean_abs_shap,
                "mean_shap": self.shap_values.mean(axis=0),
                "std_shap": self.shap_values.std(axis=0),
                "max_abs_shap": np.abs(self.shap_values).max(axis=0),
            }
        )

        # Sort by importance
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)

        # Save to CSV
        importance_path = self.data_dir / "shap_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        self.logger.info("Saved SHAP importance to: %s", importance_path)

        # Display top 15 features
        self.logger.info("=" * 80)
        self.logger.info("TOP 15 FEATURES BY SHAP IMPORTANCE")
        self.logger.info("=" * 80)
        self.logger.info(
            "%-6s %-40s %-15s %-15s", "Rank", "Feature", "Mean |SHAP|", "Mean SHAP"
        )
        self.logger.info("-" * 80)

        for idx, row in importance_df.head(15).iterrows():
            self.logger.info(
                "%-6d %-40s %-15.4f %-15.4f",
                row["rank"],
                row["feature"],
                row["mean_abs_shap"],
                row["mean_shap"],
            )

        self.results["global_importance"] = {
            "top_15_features": importance_df.head(15).to_dict("records"),
            "all_features": importance_df.to_dict("records"),
        }

        return importance_df

    def analyze_feature_interactions(self, top_n=5):
        """Analyze SHAP interaction values for top features"""
        self.logger.info("=" * 80)
        self.logger.info(
            "STEP 5: FEATURE INTERACTION ANALYSIS (Top %d features)", top_n
        )
        self.logger.info("=" * 80)

        # Only compute interactions for tree models (computationally expensive)
        if self.best_model_name not in ["RandomForest", "GradientBoosting", "XGBoost"]:
            self.logger.info(
                "Skipping interaction analysis for %s", self.best_model_name
            )
            self.logger.info(
                "(Interaction values only available for tree-based models)"
            )
            self.results["interactions"] = {
                "available": False,
                "reason": "Not a tree-based model",
            }
            return None

        # Get top N features
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": np.abs(self.shap_values).mean(axis=0),
            }
        ).sort_values("importance", ascending=False)

        top_features = importance_df.head(top_n)["feature"].tolist()
        top_indices = [self.feature_names.index(f) for f in top_features]

        self.logger.info("Analyzing interactions for top %d features:", top_n)
        for i, feat in enumerate(top_features, 1):
            self.logger.info("  %d. %s", i, feat)

        # Sample data for efficiency (interactions are computationally expensive)
        sample_size = min(100, len(self.X_test))
        X_sample = self.X_test.iloc[:sample_size]

        self.logger.info(
            "Computing SHAP interaction values on %d samples...", sample_size
        )
        self.logger.info("(This may take a few minutes...)")

        try:
            shap_interaction_values = shap.TreeExplainer(
                self.best_model
            ).shap_interaction_values(X_sample)

            # Extract interactions for top features
            interactions = []
            for i, feat1 in enumerate(top_features):
                idx1 = self.feature_names.index(feat1)
                for j, feat2 in enumerate(top_features):
                    if i < j:  # Only upper triangle
                        idx2 = self.feature_names.index(feat2)
                        interaction_strength = np.abs(
                            shap_interaction_values[:, idx1, idx2]
                        ).mean()
                        interactions.append(
                            {
                                "feature_1": feat1,
                                "feature_2": feat2,
                                "interaction_strength": interaction_strength,
                            }
                        )

            # Sort by strength
            interactions_df = pd.DataFrame(interactions)
            interactions_df = interactions_df.sort_values(
                "interaction_strength", ascending=False
            )

            # Save to CSV
            interactions_path = self.data_dir / "shap_interactions.csv"
            interactions_df.to_csv(interactions_path, index=False)
            self.logger.info("Saved interactions to: %s", interactions_path)

            self.logger.info("Top Feature Interactions:")
            self.logger.info("-" * 80)
            for idx, row in interactions_df.head(10).iterrows():
                self.logger.info(
                    "%s <-> %s : %.4f",
                    row["feature_1"],
                    row["feature_2"],
                    row["interaction_strength"],
                )

            self.results["interactions"] = {
                "available": True,
                "top_interactions": interactions_df.head(10).to_dict("records"),
                "all_interactions": interactions_df.to_dict("records"),
            }

            return interactions_df

        except Exception as e:
            self.logger.error("Error computing interactions: %s", e)
            self.logger.warning("Skipping interaction analysis...")
            self.results["interactions"] = {"available": False, "reason": str(e)}
            return None

    def analyze_local_explanations(self):
        """Analyze local explanations for representative samples"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 6: LOCAL INTERPRETABILITY ANALYSIS")
        self.logger.info("=" * 80)

        # Get predictions
        y_pred = self.best_model.predict(self.X_test)

        # Select representative samples: low, medium, high target values
        sorted_indices = np.argsort(self.y_test)

        # Get indices for low, medium, high
        low_idx = sorted_indices[0]
        medium_idx = sorted_indices[len(sorted_indices) // 2]
        high_idx = sorted_indices[-1]

        # Also get samples with largest prediction errors
        errors = np.abs(self.y_test - y_pred)
        largest_error_idx = np.argmax(errors)

        representative_samples = {
            "low_target": low_idx,
            "medium_target": medium_idx,
            "high_target": high_idx,
            "largest_error": largest_error_idx,
        }

        self.logger.info("Representative Samples:")
        self.logger.info("-" * 80)

        local_explanations = []
        for name, idx in representative_samples.items():
            actual = self.y_test[idx]
            predicted = y_pred[idx]
            error = predicted - actual

            # Get top 5 contributing features for this sample
            sample_shap = self.shap_values[idx]
            top_contrib_indices = np.argsort(np.abs(sample_shap))[-5:][::-1]

            contributions = []
            for feat_idx in top_contrib_indices:
                contributions.append(
                    {
                        "feature": self.feature_names[feat_idx],
                        "shap_value": float(sample_shap[feat_idx]),
                        "feature_value": float(self.X_test.iloc[idx, feat_idx]),
                    }
                )

            explanation = {
                "sample_type": name,
                "test_index": int(idx),
                "actual_value": float(actual),
                "predicted_value": float(predicted),
                "prediction_error": float(error),
                "top_5_contributions": contributions,
            }

            local_explanations.append(explanation)

            self.logger.info("%s:", name.upper().replace("_", " "))
            self.logger.info(
                "  Actual: %.2f, Predicted: %.2f, Error: %.2f", actual, predicted, error
            )
            self.logger.info("  Top 5 contributing features:")
            for contrib in contributions:
                self.logger.info(
                    "    %-35s SHAP: %8.4f  Value: %8.4f",
                    contrib["feature"],
                    contrib["shap_value"],
                    contrib["feature_value"],
                )

        self.results["local_explanations"] = local_explanations

        return representative_samples

    def create_visualizations(self, importance_df, representative_samples):
        """Create all SHAP visualizations"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 7: CREATING VISUALIZATIONS")
        self.logger.info("=" * 80)

        # 1. Summary plot (beeswarm)
        self.logger.info("1. Creating SHAP summary plot (beeswarm)...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        summary_path = self.plots_dir / "shap_summary_beeswarm.png"
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info("   Saved: %s", summary_path)

        # 2. Bar plot of mean absolute SHAP values
        self.logger.info("2. Creating SHAP importance bar plot...")
        plt.figure(figsize=(12, 10))
        top_15 = importance_df.head(15).iloc[::-1]  # Reverse for plotting
        plt.barh(range(len(top_15)), top_15["mean_abs_shap"])
        plt.yticks(range(len(top_15)), top_15["feature"])
        plt.xlabel("Mean |SHAP value|", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title("Top 15 Features by SHAP Importance", fontsize=14, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        bar_path = self.plots_dir / "shap_importance_bar.png"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info("   Saved: %s", bar_path)

        # 3. Dependence plots for top 5 features
        self.logger.info("3. Creating dependence plots for top 5 features...")
        top_5_features = importance_df.head(5)["feature"].tolist()

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.flatten()

        for i, feature in enumerate(top_5_features):
            plt.sca(axes[i])
            shap.dependence_plot(
                feature, self.shap_values, self.X_test, show=False, ax=axes[i]
            )
            axes[i].set_title(
                f"SHAP Dependence: {feature}", fontsize=11, fontweight="bold"
            )

        # Hide the last subplot
        axes[5].axis("off")

        plt.tight_layout()
        dependence_path = self.plots_dir / "shap_dependence_top5.png"
        plt.savefig(dependence_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info("   Saved: %s", dependence_path)

        # 4. Waterfall plots for representative samples
        self.logger.info("4. Creating waterfall plots for representative samples...")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for idx, (name, sample_idx) in enumerate(representative_samples.items()):
            plt.sca(axes[idx])
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.shap_explainer.expected_value,
                    data=self.X_test.iloc[sample_idx],
                    feature_names=self.feature_names,
                ),
                show=False,
                max_display=15,
            )
            axes[idx].set_title(
                f'{name.replace("_", " ").title()}\n'
                f"Actual: {self.y_test[sample_idx]:.2f}, "
                f"Predicted: {self.best_model.predict(self.X_test.iloc[sample_idx:sample_idx+1])[0]:.2f}",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        waterfall_path = self.plots_dir / "shap_waterfall_samples.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info("   Saved: %s", waterfall_path)

        # 5. Force plots for representative samples
        self.logger.info("5. Creating force plots for representative samples...")

        # Force plots are better as HTML, but we'll create a matplotlib version
        fig, axes = plt.subplots(len(representative_samples), 1, figsize=(18, 12))
        if len(representative_samples) == 1:
            axes = [axes]

        for idx, (name, sample_idx) in enumerate(representative_samples.items()):
            # Use matplotlib force plot
            plt.sca(axes[idx])
            shap.plots.force(
                self.shap_explainer.expected_value,
                self.shap_values[sample_idx],
                self.X_test.iloc[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False,
            )
            axes[idx].set_title(
                f'{name.replace("_", " ").title()} - '
                f"Actual: {self.y_test[sample_idx]:.2f}",
                fontsize=10,
                fontweight="bold",
                pad=10,
            )

        plt.tight_layout()
        force_path = self.plots_dir / "shap_force_plots.png"
        plt.savefig(force_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.info("   Saved: %s", force_path)

        self.logger.info("All visualizations created successfully!")

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        self.logger.info("=" * 80)
        self.logger.info("STEP 8: GENERATING SUMMARY REPORT")
        self.logger.info("=" * 80)

        # Add interpretation insights
        top_15 = self.results["global_importance"]["top_15_features"]

        # Categorize top features
        feature_categories = {
            "pm_features": [f for f in self.feature_names if f.startswith("pm_")],
            "muspy_features": [f for f in self.feature_names if f.startswith("muspy_")],
            "theory_features": [
                f for f in self.feature_names if f.startswith("theory_")
            ],
        }

        category_counts = {cat: 0 for cat in feature_categories.keys()}
        for feat_info in top_15:
            for cat, features in feature_categories.items():
                if feat_info["feature"] in features:
                    category_counts[cat] += 1

        self.results["interpretation"] = {
            "top_15_category_distribution": category_counts,
            "model_performance": self.results["best_model"],
            "explainer_info": self.results["shap_info"],
            "key_insights": self._generate_insights(),
        }

        # Save to JSON
        report_path = self.reports_dir / "shap_summary.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        self.logger.info("Saved summary report to: %s", report_path)

        # Print summary
        self.logger.info("=" * 80)
        self.logger.info("ANALYSIS SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(
            "Dataset: %d samples, %d features",
            self.results["data_info"]["total_samples"],
            self.results["data_info"]["num_features"],
        )
        self.logger.info("Best Model: %s", self.results["best_model"]["name"])
        self.logger.info("Test R²: %.4f", self.results["best_model"]["test_r2"])
        self.logger.info("Test RMSE: %.4f", self.results["best_model"]["test_rmse"])
        self.logger.info(
            "SHAP Explainer: %s", self.results["shap_info"]["explainer_type"]
        )
        self.logger.info("Top 15 Features by Category:")
        for cat, count in category_counts.items():
            self.logger.info("  %s: %d features", cat, count)

        return report_path

    def _generate_insights(self):
        """Generate key insights from SHAP analysis"""
        insights = []

        top_15 = self.results["global_importance"]["top_15_features"]

        # Top feature
        top_feat = top_15[0]
        insights.append(
            f"The most important feature is '{top_feat['feature']}' with mean |SHAP| = {top_feat['mean_abs_shap']:.4f}"
        )

        # Positive vs negative effects
        positive_effects = [f for f in top_15 if f["mean_shap"] > 0]
        negative_effects = [f for f in top_15 if f["mean_shap"] < 0]

        if positive_effects:
            insights.append(
                f"{len(positive_effects)} of the top 15 features have predominantly positive effects on the target"
            )
        if negative_effects:
            insights.append(
                f"{len(negative_effects)} of the top 15 features have predominantly negative effects on the target"
            )

        # Feature categories
        pm_count = sum(1 for f in top_15 if f["feature"].startswith("pm_"))
        muspy_count = sum(1 for f in top_15 if f["feature"].startswith("muspy_"))
        theory_count = sum(1 for f in top_15 if f["feature"].startswith("theory_"))

        if pm_count > muspy_count and pm_count > theory_count:
            insights.append(
                f"PM features dominate the top 15 most important features ({pm_count}/15)"
            )
        elif muspy_count > pm_count and muspy_count > theory_count:
            insights.append(
                f"Muspy features dominate the top 15 most important features ({muspy_count}/15)"
            )
        elif theory_count > pm_count and theory_count > muspy_count:
            insights.append(
                f"Theory features dominate the top 15 most important features ({theory_count}/15)"
            )

        # Interactions
        if self.results.get("interactions", {}).get("available"):
            top_interaction = self.results["interactions"]["top_interactions"][0]
            insights.append(
                f"Strongest feature interaction: {top_interaction['feature_1']} <-> {top_interaction['feature_2']}"
            )

        return insights

    def run_full_analysis(self):
        """Run complete SHAP analysis pipeline"""
        self.logger.info("*" * 80)
        self.logger.info("SHAP FEATURE IMPORTANCE AND INTERPRETABILITY ANALYSIS")
        self.logger.info("*" * 80)

        # Step 1: Load and prepare data
        self.load_and_prepare_data()

        # Step 2: Train and select best model
        self.train_and_select_model()

        # Step 3: Calculate SHAP values
        self.calculate_shap_values()

        # Step 4: Analyze global importance
        importance_df = self.analyze_global_importance()

        # Step 5: Analyze feature interactions
        self.analyze_feature_interactions(top_n=5)

        # Step 6: Analyze local explanations
        representative_samples = self.analyze_local_explanations()

        # Step 7: Create visualizations
        self.create_visualizations(importance_df, representative_samples)

        # Step 8: Generate summary report
        report_path = self.generate_summary_report()

        self.logger.info("=" * 80)
        self.logger.info("ANALYSIS COMPLETE!")
        self.logger.info("=" * 80)
        self.logger.info("All outputs saved to: %s", self.output_dir)
        self.logger.info("Key files:")
        self.logger.info("  - SHAP values: %s", self.data_dir / "shap_values.npy")
        self.logger.info(
            "  - Importance rankings: %s", self.data_dir / "shap_importance.csv"
        )
        self.logger.info(
            "  - Interactions: %s", self.data_dir / "shap_interactions.csv"
        )
        self.logger.info("  - Summary report: %s", report_path)
        self.logger.info("  - Visualizations: %s", self.plots_dir)

        return self.results


# Initialize analyzer and run full analysis
analyzer = SHAPAnalyzer(args.data_path, args.output_dir, logger)
results = analyzer.run_full_analysis()

logger.info("=" * 80)
logger.info("KEY INSIGHTS")
logger.info("=" * 80)
for insight in results["interpretation"]["key_insights"]:
    logger.info("  • %s", insight)
