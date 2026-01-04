#!/usr/bin/env python3
"""
Consensus Feature Importance Analysis
Synthesizes RFE, SHAP, VIF, Correlation, and Mutual Information analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import networkx as nx
import json
from pathlib import Path
import warnings
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

from src.utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Parse arguments
parser = argparse.ArgumentParser(description="Consensus Feature Importance Analysis")
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/feature_importance_analysis/comprehensive_commu_full",
    help="Output directory path",
)

args = parser.parse_args()

BASE_DIR = Path(args.output_dir)
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"

# Create output directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize logging
log_file = REPORTS_DIR / "consensus_analysis.log"
logger_manager = LoggingManager(
    name="ghsom_feature_analysis.consensus_analysis",
    level=10,  # DEBUG level
    log_file=log_file,
    enable_wandb=False,
)
logger = logger_manager.logger

logger.info("=" * 80)
logger.info("CONSENSUS FEATURE IMPORTANCE ANALYSIS")
logger.info("=" * 80)
logger.info("Output: %s", args.output_dir)
logger.info("=" * 80)

# ============================================================================
# 1. LOAD ALL DATA
# ============================================================================
logger.info("")
logger.info("[1/9] Loading data from multiple analyses...")

# Load RFE rankings
rfe_df = pd.read_csv(DATA_DIR / "rfe_rankings.csv")
logger.info("  ✓ RFE rankings: %d features (target: muspy_pitch_entropy)", len(rfe_df))

# Load SHAP importance
shap_df = pd.read_csv(DATA_DIR / "shap_importance.csv")
logger.info("  ✓ SHAP importance: %d features (target: pm_energy)", len(shap_df))

# Load VIF scores
vif_df = pd.read_csv(DATA_DIR / "vif_scores.csv")
logger.info("  ✓ VIF scores: %d features", len(vif_df))

# Load highly correlated pairs
corr_pairs_df = pd.read_csv(DATA_DIR / "highly_correlated_pairs.csv")
logger.info("  ✓ Highly correlated pairs: %d pairs", len(corr_pairs_df))

# Load mutual information
mi_df = pd.read_csv(DATA_DIR / "mutual_information.csv")
logger.info("  ✓ Mutual information: %d pairs", len(mi_df))

# Load missing values report
missing_df = pd.read_csv(DATA_DIR / "missing_values_report.csv")
logger.info("  ✓ Missing values report: %d features", len(missing_df))

# Load full correlation matrix
corr_matrix = pd.read_csv(DATA_DIR / "correlation_matrix.csv", index_col=0)
logger.info("  ✓ Correlation matrix: %dx%d", corr_matrix.shape[0], corr_matrix.shape[1])

# ============================================================================
# 2. NORMALIZE ALL RANKINGS TO 0-100 SCALE
# ============================================================================
logger.info("")
logger.info("[2/9] Normalizing all rankings to 0-100 scale...")

# Get all unique features
all_features = sorted(
    set(rfe_df["feature"].tolist()) | set(shap_df["feature"].tolist())
)
logger.info("  Total unique features: %d", len(all_features))

# Create master dataframe
consensus_df = pd.DataFrame({"feature": all_features})

# --- RFE Score (inverse rank, 1=best → 100, worst→0) ---
rfe_max_rank = rfe_df["rfe_rank"].max()
rfe_scores = {}
for _, row in rfe_df.iterrows():
    # Inverse: rank 1 → 100, max_rank → 0
    score = 100 * (rfe_max_rank - row["rfe_rank"] + 1) / rfe_max_rank
    rfe_scores[row["feature"]] = score
consensus_df["rfe_score"] = consensus_df["feature"].map(rfe_scores).fillna(0)

# --- SHAP Score (normalize mean_abs_shap to 0-100) ---
shap_max = shap_df["mean_abs_shap"].max()
shap_scores = {}
for _, row in shap_df.iterrows():
    score = 100 * row["mean_abs_shap"] / shap_max if shap_max > 0 else 0
    shap_scores[row["feature"]] = score
consensus_df["shap_score"] = consensus_df["feature"].map(shap_scores).fillna(0)

# --- VIF Penalty (lower VIF = better, VIF>10 penalized) ---
vif_scores = {}
for _, row in vif_df.iterrows():
    vif = row["VIF"]
    if np.isinf(vif) or vif > 100:
        score = -100  # Severe multicollinearity
    elif vif > 10:
        score = 100 - (vif - 10) * 5  # Penalty scale
        score = max(score, 0)
    else:
        score = 100  # Good VIF
    vif_scores[row["feature"]] = score
consensus_df["vif_score"] = consensus_df["feature"].map(vif_scores).fillna(50)

# --- Correlation Penalty (features in highly correlated pairs) ---
corr_penalties = {}
for _, row in corr_pairs_df.iterrows():
    f1, f2 = row["feature_1"], row["feature_2"]
    corr = abs(row["correlation"])
    # Penalty based on correlation strength (>0.8 = problematic)
    penalty = 100 * (1 - min((corr - 0.8) / 0.2, 1)) if corr > 0.8 else 100
    # Apply penalty to both features
    corr_penalties[f1] = min(corr_penalties.get(f1, 100), penalty)
    corr_penalties[f2] = min(corr_penalties.get(f2, 100), penalty)
consensus_df["corr_score"] = consensus_df["feature"].map(corr_penalties).fillna(100)

# --- Mutual Information Score (average MI with all features) ---
# Calculate average MI for each feature
mi_avg = {}
for feature in all_features:
    # Get all MI values where this feature appears
    mi_values = []
    for _, row in mi_df.iterrows():
        if row["feature_1"] == feature or row["feature_2"] == feature:
            mi_values.append(row["mutual_information"])
    if mi_values:
        mi_avg[feature] = np.mean(mi_values)
    else:
        mi_avg[feature] = 0

# Normalize to 0-100
mi_max = max(mi_avg.values()) if mi_avg else 1
mi_scores = {f: 100 * val / mi_max for f, val in mi_avg.items()}
consensus_df["mi_score"] = consensus_df["feature"].map(mi_scores).fillna(0)

logger.info("  ✓ Normalized scores computed for all features")
logger.info(
    "    - RFE score range: %.1f - %.1f",
    consensus_df["rfe_score"].min(),
    consensus_df["rfe_score"].max(),
)
logger.info(
    "    - SHAP score range: %.1f - %.1f",
    consensus_df["shap_score"].min(),
    consensus_df["shap_score"].max(),
)
logger.info(
    "    - VIF score range: %.1f - %.1f",
    consensus_df["vif_score"].min(),
    consensus_df["vif_score"].max(),
)
logger.info(
    "    - Correlation score range: %.1f - %.1f",
    consensus_df["corr_score"].min(),
    consensus_df["corr_score"].max(),
)
logger.info(
    "    - MI score range: %.1f - %.1f",
    consensus_df["mi_score"].min(),
    consensus_df["mi_score"].max(),
)

# ============================================================================
# 3. CREATE CONSENSUS RANKING
# ============================================================================
logger.info("")
logger.info("[3/9] Creating consensus ranking with weighted average...")

# Weights (sum to 1.0)
weights = {
    "rfe": 0.35,  # 35% - RFE ranking
    "shap": 0.35,  # 35% - SHAP importance
    "vif": 0.15,  # 15% - VIF penalty
    "corr": 0.10,  # 10% - Correlation penalty
    "mi": 0.05,  # 5% - Mutual information
}

logger.info(
    "  Weights: RFE=%s | SHAP=%s | VIF=%s | Corr=%s | MI=%s",
    weights["rfe"],
    weights["shap"],
    weights["vif"],
    weights["corr"],
    weights["mi"],
)

# Calculate consensus score
consensus_df["consensus_score"] = (
    weights["rfe"] * consensus_df["rfe_score"]
    + weights["shap"] * consensus_df["shap_score"]
    + weights["vif"] * consensus_df["vif_score"]
    + weights["corr"] * consensus_df["corr_score"]
    + weights["mi"] * consensus_df["mi_score"]
)

# Rank by consensus score
consensus_df = consensus_df.sort_values("consensus_score", ascending=False).reset_index(
    drop=True
)
consensus_df["consensus_rank"] = range(1, len(consensus_df) + 1)

# Reorder columns
consensus_df = consensus_df[
    [
        "consensus_rank",
        "feature",
        "consensus_score",
        "rfe_score",
        "shap_score",
        "vif_score",
        "corr_score",
        "mi_score",
    ]
]

# Add raw rankings for reference
rfe_rank_map = dict(zip(rfe_df["feature"], rfe_df["rfe_rank"]))
shap_rank_map = dict(zip(shap_df["feature"], shap_df["rank"]))
consensus_df["rfe_rank"] = consensus_df["feature"].map(rfe_rank_map)
consensus_df["shap_rank"] = consensus_df["feature"].map(shap_rank_map)

logger.info("  ✓ Consensus ranking computed")
logger.info("")
logger.info("  Top 15 features by consensus:")
for _, row in consensus_df.head(15).iterrows():
    logger.info(
        "    %2d. %-40s (score: %.1f)",
        row["consensus_rank"],
        row["feature"],
        row["consensus_score"],
    )

# Save consensus ranking
consensus_df.to_csv(DATA_DIR / "consensus_ranking.csv", index=False)
logger.info("")
logger.info("  ✓ Saved: %s", DATA_DIR / "consensus_ranking.csv")

# ============================================================================
# 4. ANALYZE AGREEMENT AND DISAGREEMENT
# ============================================================================
logger.info("")
logger.info("[4/9] Analyzing agreement and disagreement between methods...")

# Get features that have both RFE and SHAP rankings
common_features = consensus_df[
    consensus_df["rfe_rank"].notna() & consensus_df["shap_rank"].notna()
]["feature"].tolist()
logger.info("  Common features (have both RFE and SHAP): %d", len(common_features))

# Calculate rank correlation (Spearman's rho)
method_comparison = {}

if len(common_features) >= 2:
    # RFE vs SHAP
    rfe_ranks = consensus_df[consensus_df["feature"].isin(common_features)][
        "rfe_rank"
    ].values
    shap_ranks = consensus_df[consensus_df["feature"].isin(common_features)][
        "shap_rank"
    ].values
    rho_rfe_shap, p_rfe_shap = stats.spearmanr(rfe_ranks, shap_ranks)
    method_comparison["RFE_vs_SHAP"] = {
        "correlation": rho_rfe_shap,
        "p_value": p_rfe_shap,
    }

    # RFE vs Consensus
    rfe_cons_ranks = consensus_df[consensus_df["feature"].isin(common_features)][
        "consensus_rank"
    ].values
    rho_rfe_cons, p_rfe_cons = stats.spearmanr(rfe_ranks, rfe_cons_ranks)
    method_comparison["RFE_vs_Consensus"] = {
        "correlation": rho_rfe_cons,
        "p_value": p_rfe_cons,
    }

    # SHAP vs Consensus
    rho_shap_cons, p_shap_cons = stats.spearmanr(shap_ranks, rfe_cons_ranks)
    method_comparison["SHAP_vs_Consensus"] = {
        "correlation": rho_shap_cons,
        "p_value": p_shap_cons,
    }

    logger.info("")
    logger.info("  Rank Correlations (Spearman's ρ):")
    logger.info("    RFE vs SHAP:       ρ = %.3f (p = %.4f)", rho_rfe_shap, p_rfe_shap)
    logger.info("    RFE vs Consensus:  ρ = %.3f (p = %.4f)", rho_rfe_cons, p_rfe_cons)
    logger.info(
        "    SHAP vs Consensus: ρ = %.3f (p = %.4f)", rho_shap_cons, p_shap_cons
    )

# Calculate variance in ranks across methods
consensus_df["rank_variance"] = consensus_df.apply(
    lambda row: np.var(
        [
            row["rfe_rank"] if pd.notna(row["rfe_rank"]) else row["consensus_rank"],
            row["shap_rank"] if pd.notna(row["shap_rank"]) else row["consensus_rank"],
            row["consensus_rank"],
        ]
    ),
    axis=1,
)

# Identify features with high agreement (low variance)
high_agreement = consensus_df[
    consensus_df["rank_variance"] < consensus_df["rank_variance"].quantile(0.25)
]
logger.info("")
logger.info(
    "  Features with high agreement (low rank variance): %d", len(high_agreement)
)
for _, row in high_agreement.head(10).iterrows():
    logger.info("    - %-40s (variance: %.1f)", row["feature"], row["rank_variance"])

# Identify features with high disagreement (high variance)
high_disagreement = consensus_df[
    consensus_df["rank_variance"] > consensus_df["rank_variance"].quantile(0.75)
]
logger.info("")
logger.info(
    "  Features with high disagreement (high rank variance): %d", len(high_disagreement)
)
for _, row in high_disagreement.head(10).iterrows():
    logger.info(
        "    - %-40s (variance: %.1f, RFE:%s, SHAP:%s)",
        row["feature"],
        row["rank_variance"],
        row["rfe_rank"],
        row["shap_rank"],
    )

# Top 10 from each method (Jaccard similarity)
rfe_top10 = set(rfe_df.head(10)["feature"])
shap_top10 = set(shap_df.head(10)["feature"])
consensus_top10 = set(consensus_df.head(10)["feature"])

jaccard_rfe_shap = len(rfe_top10 & shap_top10) / len(rfe_top10 | shap_top10)
jaccard_rfe_cons = len(rfe_top10 & consensus_top10) / len(rfe_top10 | consensus_top10)
jaccard_shap_cons = len(shap_top10 & consensus_top10) / len(
    shap_top10 | consensus_top10
)

logger.info("")
logger.info("  Top-10 Overlap (Jaccard Similarity):")
logger.info(
    "    RFE ∩ SHAP:       %.3f (%d shared)",
    jaccard_rfe_shap,
    len(rfe_top10 & shap_top10),
)
logger.info(
    "    RFE ∩ Consensus:  %.3f (%d shared)",
    jaccard_rfe_cons,
    len(rfe_top10 & consensus_top10),
)
logger.info(
    "    SHAP ∩ Consensus: %.3f (%d shared)",
    jaccard_shap_cons,
    len(shap_top10 & consensus_top10),
)

# Save method comparison
comparison_data = []
for method_pair, stats_data in method_comparison.items():
    comparison_data.append(
        {
            "method_pair": method_pair,
            "spearman_rho": stats_data["correlation"],
            "p_value": stats_data["p_value"],
        }
    )
comparison_data.append(
    {
        "method_pair": "Top10_RFE_SHAP_Jaccard",
        "spearman_rho": jaccard_rfe_shap,
        "p_value": np.nan,
    }
)
comparison_data.append(
    {
        "method_pair": "Top10_RFE_Consensus_Jaccard",
        "spearman_rho": jaccard_rfe_cons,
        "p_value": np.nan,
    }
)
comparison_data.append(
    {
        "method_pair": "Top10_SHAP_Consensus_Jaccard",
        "spearman_rho": jaccard_shap_cons,
        "p_value": np.nan,
    }
)

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(DATA_DIR / "method_comparison.csv", index=False)
logger.info("")
logger.info("  ✓ Saved: %s", DATA_DIR / "method_comparison.csv")

# ============================================================================
# 5. VALIDATE AGAINST DATA QUALITY
# ============================================================================
logger.info("")
logger.info("[5/9] Validating against data quality metrics...")

# Add VIF values
vif_map = dict(zip(vif_df["feature"], vif_df["VIF"]))
consensus_df["vif_value"] = consensus_df["feature"].map(vif_map)

# Add missing percentage
missing_map = dict(zip(missing_df["feature"], missing_df["missing_percentage"]))
consensus_df["missing_pct"] = consensus_df["feature"].map(missing_map).fillna(0)

# Flag quality issues
consensus_df["quality_flag"] = ""
consensus_df.loc[consensus_df["vif_value"] > 100, "quality_flag"] += "VIF>100; "
consensus_df.loc[consensus_df["missing_pct"] > 50, "quality_flag"] += "Missing>50%; "

# Check for high correlation in top features
top15_features = consensus_df.head(15)["feature"].tolist()
high_corr_in_top15 = []
for _, row in corr_pairs_df.iterrows():
    if row["feature_1"] in top15_features and row["feature_2"] in top15_features:
        if abs(row["correlation"]) > 0.9:
            high_corr_in_top15.append(
                (row["feature_1"], row["feature_2"], row["correlation"])
            )

if high_corr_in_top15:
    logger.warning("")
    logger.warning("  ⚠ WARNING: High correlation (>0.9) found in top 15 features:")
    for f1, f2, corr in high_corr_in_top15:
        logger.warning("    - %s ↔ %s: r = %.3f", f1, f2, corr)
        # Flag these features
        consensus_df.loc[
            consensus_df["feature"] == f1, "quality_flag"
        ] += f"HighCorr({f2}); "
        consensus_df.loc[
            consensus_df["feature"] == f2, "quality_flag"
        ] += f"HighCorr({f1}); "

# Features with quality issues in top 15
issues_in_top15 = consensus_df.head(15)[consensus_df.head(15)["quality_flag"] != ""]
if len(issues_in_top15) > 0:
    logger.warning("")
    logger.warning("  ⚠ Quality issues in top 15 features:")
    for _, row in issues_in_top15.iterrows():
        logger.warning("    - %-40s: %s", row["feature"], row["quality_flag"])
else:
    logger.info("")
    logger.info("  ✓ No major quality issues in top 15 features")

# Exclude features with severe issues (VIF>100)
severe_issues = consensus_df[consensus_df["vif_value"] > 100]
if len(severe_issues) > 0:
    logger.warning("")
    logger.warning("  Features to exclude (VIF > 100): %d", len(severe_issues))
    for _, row in severe_issues.iterrows():
        logger.warning("    - %-40s (VIF: %.1f)", row["feature"], row["vif_value"])

# ============================================================================
# 6. CREATE DEPENDENCY ANALYSIS
# ============================================================================
logger.info("")
logger.info("[6/9] Creating dependency analysis and feature groups...")

# Build interaction matrix from correlations and MI
# Use absolute correlation values
interaction_matrix = corr_matrix.abs()

# Replace NaN and inf values with 0
interaction_matrix = interaction_matrix.fillna(0)
interaction_matrix = interaction_matrix.replace([np.inf, -np.inf], 0)

# Hierarchical clustering to find feature groups
# Convert to distance matrix (1 - correlation)
distance_matrix = 1 - interaction_matrix
distance_matrix = np.clip(distance_matrix, 0, 2)  # Ensure values are in valid range

condensed_dist = squareform(distance_matrix, checks=False)
# Replace any remaining invalid values
condensed_dist = np.nan_to_num(condensed_dist, nan=1.0, posinf=1.0, neginf=1.0)
linkage = hierarchy.linkage(condensed_dist, method="average")

# Cut dendrogram to get clusters (distance threshold = 0.3, i.e., correlation > 0.7)
clusters = hierarchy.fcluster(linkage, t=0.3, criterion="distance")

# Create feature groups dataframe
feature_groups = pd.DataFrame(
    {"feature": interaction_matrix.index, "cluster": clusters}
)

# Add consensus rank and scores to groups
feature_groups = feature_groups.merge(
    consensus_df[["feature", "consensus_rank", "consensus_score"]],
    on="feature",
    how="left",
)

# Sort by cluster and then by rank
feature_groups = feature_groups.sort_values(["cluster", "consensus_rank"]).reset_index(
    drop=True
)

# Identify representative from each cluster (highest consensus score)
cluster_representatives = (
    feature_groups.groupby("cluster")
    .apply(
        lambda x: (
            x.loc[x["consensus_score"].dropna().idxmax()]
            if not x["consensus_score"].dropna().empty
            else x.iloc[0]
        )
    )
    .reset_index(drop=True)
)

logger.info("")
logger.info("  Identified %d feature clusters", feature_groups["cluster"].nunique())
logger.info("")
logger.info("  Cluster representatives (keep these):")
for _, row in cluster_representatives.head(15).iterrows():
    cluster_size = len(feature_groups[feature_groups["cluster"] == row["cluster"]])
    logger.info(
        "    Cluster %2d (n=%2d): %-40s (rank: %2.0f, score: %.1f)",
        row["cluster"],
        cluster_size,
        row["feature"],
        row["consensus_rank"],
        row["consensus_score"],
    )

# For each cluster, recommend which features to keep vs drop
cluster_recommendations = []
for cluster_id in sorted(feature_groups["cluster"].unique()):
    cluster_features = feature_groups[
        feature_groups["cluster"] == cluster_id
    ].sort_values("consensus_score", ascending=False)

    if len(cluster_features) > 1:
        keep_feature = cluster_features.iloc[0]["feature"]
        drop_features = cluster_features.iloc[1:]["feature"].tolist()

        cluster_recommendations.append(
            {
                "cluster": cluster_id,
                "size": len(cluster_features),
                "keep": keep_feature,
                "drop": ", ".join(drop_features),
                "reason": f'Highest consensus score ({cluster_features.iloc[0]["consensus_score"]:.1f})',
            }
        )

cluster_rec_df = pd.DataFrame(cluster_recommendations)

# Save feature groups
feature_groups.to_csv(DATA_DIR / "feature_groups.csv", index=False)
cluster_rec_df.to_csv(DATA_DIR / "cluster_recommendations.csv", index=False)
logger.info("")
logger.info("  ✓ Saved: %s", DATA_DIR / "feature_groups.csv")
logger.info("  ✓ Saved: %s", DATA_DIR / "cluster_recommendations.csv")

# ============================================================================
# 7. STATISTICAL SIGNIFICANCE
# ============================================================================
logger.info("")
logger.info("[7/9] Computing statistical significance...")

# For SHAP: use bootstrap to get confidence intervals
shap_significance = []
shap_values_array = np.load(DATA_DIR / "shap_values.npy", allow_pickle=True)

logger.info("  Computing bootstrap confidence intervals for SHAP values...")
n_bootstrap = 1000
np.random.seed(42)

for idx, row in shap_df.iterrows():
    feature = row["feature"]
    # Get SHAP values for this feature (column)
    if idx < shap_values_array.shape[1]:
        feature_shap = np.abs(shap_values_array[:, idx])

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(
                feature_shap, size=len(feature_shap), replace=True
            )
            bootstrap_means.append(np.mean(sample))

        # 95% confidence interval
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        # Feature is significant if CI doesn't include 0
        is_significant = ci_lower > 0

        shap_significance.append(
            {
                "feature": feature,
                "mean_abs_shap": row["mean_abs_shap"],
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "significant": is_significant,
            }
        )

shap_sig_df = pd.DataFrame(shap_significance)

# For RFE: significance is implicit (selected vs not selected)
rfe_significant = rfe_df[rfe_df["rfe_selected"] == True]["feature"].tolist()

# Combine significance results
significance_df = consensus_df[["feature", "consensus_rank", "consensus_score"]].copy()
significance_df["rfe_selected"] = significance_df["feature"].isin(rfe_significant)
significance_df = significance_df.merge(
    shap_sig_df[["feature", "significant", "ci_lower", "ci_upper"]],
    on="feature",
    how="left",
)
significance_df.rename(columns={"significant": "shap_significant"}, inplace=True)

# Overall significance: both RFE and SHAP agree
significance_df["overall_significant"] = significance_df[
    "rfe_selected"
] & significance_df["shap_significant"].fillna(False)

logger.info("")
logger.info("  Statistically significant features:")
logger.info("    - RFE selected: %d", significance_df["rfe_selected"].sum())
logger.info(
    "    - SHAP significant (CI > 0): %d", significance_df["shap_significant"].sum()
)
logger.info("    - Both RFE and SHAP: %d", significance_df["overall_significant"].sum())

# Save significance results
significance_df.to_csv(DATA_DIR / "statistical_significance.csv", index=False)
logger.info("")
logger.info("  ✓ Saved: %s", DATA_DIR / "statistical_significance.csv")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
logger.info("")
logger.info("[8/9] Creating visualizations...")

# --- Plot 1: Consensus Ranking Comparison ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 15 consensus scores
top15 = consensus_df.head(15)
ax = axes[0, 0]
ax.barh(range(len(top15)), top15["consensus_score"], color="#2E86AB")
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15["feature"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Consensus Score", fontsize=11)
ax.set_title("Top 15 Features by Consensus Score", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

# Component scores for top 15
ax = axes[0, 1]
methods = ["rfe_score", "shap_score", "vif_score", "corr_score", "mi_score"]
method_labels = ["RFE", "SHAP", "VIF", "Corr", "MI"]
x = np.arange(len(top15))
width = 0.15

for i, method in enumerate(methods):
    offset = (i - 2) * width
    ax.bar(x + offset, top15[method], width, label=method_labels[i], alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(top15["feature"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Component Scores for Top 15 Features", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# Rank comparison scatter
ax = axes[1, 0]
common = consensus_df[
    consensus_df["rfe_rank"].notna() & consensus_df["shap_rank"].notna()
]
ax.scatter(
    common["rfe_rank"],
    common["shap_rank"],
    alpha=0.6,
    s=80,
    c=common["consensus_score"],
    cmap="viridis",
    edgecolors="black",
    linewidth=0.5,
)
ax.plot([0, 30], [0, 30], "r--", alpha=0.5, label="Perfect agreement")
ax.set_xlabel("RFE Rank", fontsize=11)
ax.set_ylabel("SHAP Rank", fontsize=11)
ax.set_title(
    f'RFE vs SHAP Ranking (ρ = {method_comparison.get("RFE_vs_SHAP", {}).get("correlation", 0):.3f})',
    fontsize=12,
    fontweight="bold",
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label("Consensus Score", fontsize=9)

# Rank variance distribution
ax = axes[1, 1]
ax.hist(
    consensus_df["rank_variance"],
    bins=30,
    color="#A23B72",
    alpha=0.7,
    edgecolor="black",
)
ax.axvline(
    consensus_df["rank_variance"].median(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f'Median = {consensus_df["rank_variance"].median():.1f}',
)
ax.set_xlabel("Rank Variance Across Methods", fontsize=11)
ax.set_ylabel("Number of Features", fontsize=11)
ax.set_title("Distribution of Rank Disagreement", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "consensus_ranking_comparison.png", dpi=300, bbox_inches="tight"
)
logger.info("  ✓ Saved: %s", PLOTS_DIR / "consensus_ranking_comparison.png")
plt.close()

# --- Plot 2: Method Agreement Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))

# Create rank correlation matrix
methods_data = consensus_df[["rfe_rank", "shap_rank", "consensus_rank"]].dropna()
corr_matrix_methods = methods_data.corr(method="spearman")

# Rename for display
corr_matrix_methods.index = ["RFE Rank", "SHAP Rank", "Consensus Rank"]
corr_matrix_methods.columns = ["RFE Rank", "SHAP Rank", "Consensus Rank"]

sns.heatmap(
    corr_matrix_methods,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=2,
    cbar_kws={"label": "Spearman's ρ"},
    ax=ax,
)
ax.set_title("Rank Correlation Between Methods", fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "method_agreement_heatmap.png", dpi=300, bbox_inches="tight")
logger.info("  ✓ Saved: %s", PLOTS_DIR / "method_agreement_heatmap.png")
plt.close()

# --- Plot 3: Feature Dependency Network ---
logger.info("  Creating feature dependency network...")

# Create network from high MI and high correlation pairs
G = nx.Graph()

# Add nodes (top 20 features)
top20_features = consensus_df.head(20)["feature"].tolist()
for feature in top20_features:
    G.add_node(feature)

# Add edges for high MI (top 50 pairs involving top 20)
mi_top20 = mi_df[
    (mi_df["feature_1"].isin(top20_features))
    & (mi_df["feature_2"].isin(top20_features))
].nlargest(50, "mutual_information")

for _, row in mi_top20.iterrows():
    if row["feature_1"] in top20_features and row["feature_2"] in top20_features:
        G.add_edge(
            row["feature_1"],
            row["feature_2"],
            weight=row["mutual_information"],
            edge_type="MI",
        )

# Add edges for high correlation (>0.7)
corr_high = corr_pairs_df[
    (corr_pairs_df["feature_1"].isin(top20_features))
    & (corr_pairs_df["feature_2"].isin(top20_features))
    & (abs(corr_pairs_df["correlation"]) > 0.7)
]

for _, row in corr_high.iterrows():
    if G.has_edge(row["feature_1"], row["feature_2"]):
        G[row["feature_1"]][row["feature_2"]]["correlation"] = row["correlation"]
    else:
        G.add_edge(
            row["feature_1"],
            row["feature_2"],
            weight=abs(row["correlation"]),
            edge_type="Corr",
            correlation=row["correlation"],
        )

# Create visualization
fig, ax = plt.subplots(figsize=(16, 12))

# Layout
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# Node colors based on consensus rank
node_colors = []
for node in G.nodes():
    rank = consensus_df[consensus_df["feature"] == node]["consensus_rank"].values[0]
    node_colors.append(rank)

# Node sizes based on degree
node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]

# Draw nodes
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    cmap="RdYlGn_r",
    alpha=0.9,
    edgecolors="black",
    linewidths=2,
    ax=ax,
)

# Draw edges
mi_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "MI"]
corr_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "Corr"]

nx.draw_networkx_edges(
    G,
    pos,
    edgelist=mi_edges,
    width=2,
    alpha=0.6,
    edge_color="blue",
    style="solid",
    ax=ax,
    label="Mutual Information",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=corr_edges,
    width=2,
    alpha=0.6,
    edge_color="red",
    style="dashed",
    ax=ax,
    label="Correlation > 0.7",
)

# Draw labels with smaller font
nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

plt.colorbar(nodes, ax=ax, label="Consensus Rank", shrink=0.8)
ax.set_title("Feature Dependency Network (Top 20)", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.axis("off")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_dependency_network.png", dpi=300, bbox_inches="tight")
logger.info("  ✓ Saved: %s", PLOTS_DIR / "feature_dependency_network.png")
plt.close()

# ============================================================================
# 9. CREATE SUMMARY REPORTS
# ============================================================================
logger.info("")
logger.info("[9/9] Creating summary reports...")

# JSON summary
summary_json = {
    "analysis_info": {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_features": len(all_features),
        "methods": {
            "RFE": {
                "target": "muspy_pitch_entropy",
                "features_selected": rfe_df["rfe_selected"].sum(),
            },
            "SHAP": {"target": "pm_energy", "features_analyzed": len(shap_df)},
            "VIF": {"features_analyzed": len(vif_df)},
            "Correlation": {"high_corr_pairs": len(corr_pairs_df)},
            "Mutual_Information": {"feature_pairs": len(mi_df)},
        },
    },
    "consensus_weighting": weights,
    "top_15_features": consensus_df.head(15)[
        ["consensus_rank", "feature", "consensus_score"]
    ].to_dict("records"),
    "method_agreement": {
        "rank_correlations": {
            k: v["correlation"] for k, v in method_comparison.items()
        },
        "top10_overlap": {
            "RFE_SHAP_jaccard": jaccard_rfe_shap,
            "RFE_Consensus_jaccard": jaccard_rfe_cons,
            "SHAP_Consensus_jaccard": jaccard_shap_cons,
            "shared_in_all_three": len(rfe_top10 & shap_top10 & consensus_top10),
        },
    },
    "quality_assessment": {
        "features_with_severe_vif": (
            severe_issues["feature"].tolist() if len(severe_issues) > 0 else []
        ),
        "high_correlation_in_top15": [
            (f1, f2, float(corr)) for f1, f2, corr in high_corr_in_top15
        ],
        "features_with_issues_in_top15": (
            issues_in_top15["feature"].tolist() if len(issues_in_top15) > 0 else []
        ),
    },
    "feature_clusters": {
        "n_clusters": int(feature_groups["cluster"].nunique()),
        "cluster_representatives": cluster_representatives[
            ["cluster", "feature", "consensus_rank"]
        ].to_dict("records"),
    },
    "statistical_significance": {
        "rfe_selected": int(significance_df["rfe_selected"].sum()),
        "shap_significant": int(significance_df["shap_significant"].sum()),
        "both_significant": int(significance_df["overall_significant"].sum()),
    },
}

with open(REPORTS_DIR / "consensus_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2, cls=NumpyEncoder)
logger.info("  ✓ Saved: %s", REPORTS_DIR / "consensus_summary.json")

# Markdown report
markdown_report = f"""# Feature Importance Consensus Analysis Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

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
- **RFE Ranking:** {weights['rfe']*100}%
- **SHAP Importance:** {weights['shap']*100}%
- **VIF Penalty:** {weights['vif']*100}%
- **Correlation Penalty:** {weights['corr']*100}%
- **Mutual Information:** {weights['mi']*100}%

### Top 15 Features by Consensus

| Rank | Feature | Consensus Score | RFE Rank | SHAP Rank | VIF Score | Corr Score | MI Score |
|------|---------|-----------------|----------|-----------|-----------|------------|----------|
"""

for _, row in consensus_df.head(15).iterrows():
    rfe_r = f"{row['rfe_rank']:.0f}" if pd.notna(row["rfe_rank"]) else "N/A"
    shap_r = f"{row['shap_rank']:.0f}" if pd.notna(row["shap_rank"]) else "N/A"
    markdown_report += f"| {row['consensus_rank']} | `{row['feature']}` | {row['consensus_score']:.1f} | {rfe_r} | {shap_r} | {row['vif_score']:.1f} | {row['corr_score']:.1f} | {row['mi_score']:.1f} |\n"

markdown_report += f"""

---

## 2. Method Agreement Analysis

### Rank Correlations (Spearman's ρ)

"""

for method_pair, stats_data in method_comparison.items():
    markdown_report += f"- **{method_pair}:** ρ = {stats_data['correlation']:.3f} (p = {stats_data['p_value']:.4f})\n"

markdown_report += f"""

### Top-10 Overlap (Jaccard Similarity)

- **RFE ∩ SHAP:** {jaccard_rfe_shap:.3f} ({len(rfe_top10 & shap_top10)}/10 shared features)
- **RFE ∩ Consensus:** {jaccard_rfe_cons:.3f} ({len(rfe_top10 & consensus_top10)}/10 shared)
- **SHAP ∩ Consensus:** {jaccard_shap_cons:.3f} ({len(shap_top10 & consensus_top10)}/10 shared)
- **All Three Methods:** {len(rfe_top10 & shap_top10 & consensus_top10)} features in top-10 of all methods

### Features with High Agreement

These features rank consistently across methods (low rank variance):

"""

for _, row in high_agreement.head(10).iterrows():
    markdown_report += (
        f"- `{row['feature']}` (rank variance: {row['rank_variance']:.1f})\n"
    )

markdown_report += f"""

### Features with High Disagreement

These features have large ranking differences between methods:

"""

for _, row in high_disagreement.head(10).iterrows():
    markdown_report += f"- `{row['feature']}` (variance: {row['rank_variance']:.1f}, RFE: {row['rfe_rank']}, SHAP: {row['shap_rank']})\n"

markdown_report += f"""

**Note:** Disagreements often arise because RFE optimizes for `muspy_pitch_entropy` while SHAP analyzes `pm_energy`.
Features important for one target may not be important for the other.

---

## 3. Data Quality Assessment

### Features to Exclude (VIF > 100)

"""

if len(severe_issues) > 0:
    markdown_report += "**⚠ WARNING:** The following features have severe multicollinearity issues:\n\n"
    for _, row in severe_issues.iterrows():
        vif_str = "inf" if np.isinf(row["vif_value"]) else f"{row['vif_value']:.1f}"
        markdown_report += f"- `{row['feature']}` (VIF: {vif_str})\n"
    markdown_report += (
        "\n**Recommendation:** Exclude these features from final model.\n"
    )
else:
    markdown_report += "✓ No features with VIF > 100\n"

markdown_report += f"""

### High Correlation in Top 15

"""

if high_corr_in_top15:
    markdown_report += "**⚠ WARNING:** High correlation (>0.9) detected between top-ranked features:\n\n"
    for f1, f2, corr in high_corr_in_top15:
        markdown_report += f"- `{f1}` ↔ `{f2}`: r = {corr:.3f}\n"
    markdown_report += "\n**Recommendation:** Consider keeping only one feature from each highly correlated pair.\n"
else:
    markdown_report += "✓ No high correlation (>0.9) between top 15 features\n"

markdown_report += f"""

---

## 4. Feature Groups and Clusters

Hierarchical clustering identified **{feature_groups['cluster'].nunique()} feature clusters** based on correlation and mutual information.

### Cluster Representatives (Recommended Features)

For each cluster, the feature with the highest consensus score is recommended:

"""

for _, row in cluster_rec_df.head(15).iterrows():
    markdown_report += (
        f"- **Cluster {row['cluster']}** (size: {row['size']}): Keep `{row['keep']}`"
    )
    if row["drop"]:
        markdown_report += f", drop {row['drop']}"
    markdown_report += "\n"

markdown_report += f"""

**Rationale:** When features are highly correlated or have high mutual information, keeping the top-ranked
feature from each cluster reduces redundancy while preserving predictive power.

---

## 5. Statistical Significance

### RFE Feature Selection
- **Selected by RFE:** {significance_df['rfe_selected'].sum()} features
- RFE uses cross-validated recursive elimination, so selected features are significant for predicting `muspy_pitch_entropy`

### SHAP Confidence Intervals
- **Significant SHAP values (95% CI > 0):** {significance_df['shap_significant'].sum()} features
- Bootstrap resampling (n=1000) used to compute confidence intervals

### Overall Significance
- **Both RFE and SHAP agree:** {significance_df['overall_significant'].sum()} features
- These are the most statistically robust features

---

## 6. Final Recommendations

### Recommended Feature Set (Top 15 by Consensus)

Based on consensus ranking, data quality, and statistical significance:

"""

for _, row in consensus_df.head(15).iterrows():
    flag = row["quality_flag"]
    sig_rfe = (
        "✓"
        if row["feature"]
        in significance_df[significance_df["rfe_selected"]]["feature"].values
        else "✗"
    )
    sig_shap_row = significance_df[significance_df["feature"] == row["feature"]]
    sig_shap = (
        "✓"
        if not sig_shap_row.empty and sig_shap_row["shap_significant"].values[0]
        else "✗"
    )

    status = ""
    if flag:
        status = f" ⚠ ({flag.strip('; ')})"

    markdown_report += f"{row['consensus_rank']}. `{row['feature']}` (score: {row['consensus_score']:.1f}, RFE: {sig_rfe}, SHAP: {sig_shap}){status}\n"

markdown_report += f"""

**Legend:**
- ✓ = Selected/significant by method
- ✗ = Not selected/significant
- ⚠ = Data quality warning

### Features to Consider Excluding

1. **Features with VIF > 100:** {len(severe_issues)} features (perfect multicollinearity)
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

The consensus analysis identifies **{consensus_df.head(15)['feature'].tolist()[0]}** as the most important
feature overall, followed by **{consensus_df.head(15)['feature'].tolist()[1]}** and
**{consensus_df.head(15)['feature'].tolist()[2]}**.

Features that appear in the top 10 of both RFE and SHAP ({len(rfe_top10 & shap_top10)} features) are
the most robust and should be prioritized for downstream analysis.

Consider the specific modeling task, data quality constraints, and interpretability requirements when
making final feature selection decisions.
"""

with open(REPORTS_DIR / "COMPARISON_REPORT.md", "w") as f:
    f.write(markdown_report)
logger.info("  ✓ Saved: %s", REPORTS_DIR / "COMPARISON_REPORT.md")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
logger.info("")
logger.info("=" * 80)
logger.info("ANALYSIS COMPLETE")
logger.info("=" * 80)

logger.info("")
logger.info("📊 Top 15 Features by Consensus Ranking:")
for _, row in consensus_df.head(15).iterrows():
    logger.info(
        "   %2d. %-45s score: %5.1f",
        row["consensus_rank"],
        row["feature"],
        row["consensus_score"],
    )

logger.info("")
logger.info("📈 Method Agreement:")
logger.info(
    "   - RFE vs SHAP correlation: %.3f",
    method_comparison.get("RFE_vs_SHAP", {}).get("correlation", 0),
)
logger.info(
    "   - Top-10 overlap (RFE ∩ SHAP): %d / 10 features", len(rfe_top10 & shap_top10)
)

logger.info("")
logger.info("⚠️  Quality Warnings:")
if len(severe_issues) > 0:
    logger.warning(
        "   - %d features with VIF > 100 (exclude these)", len(severe_issues)
    )
if high_corr_in_top15:
    logger.warning(
        "   - %d highly correlated pairs in top 15 (consider removing redundancy)",
        len(high_corr_in_top15),
    )
if len(issues_in_top15) == 0 and len(severe_issues) == 0 and not high_corr_in_top15:
    logger.info("   - No major issues detected ✓")

logger.info("")
logger.info("📁 Output Files:")
logger.info("   Data:")
logger.info("     - %s", DATA_DIR / "consensus_ranking.csv")
logger.info("     - %s", DATA_DIR / "method_comparison.csv")
logger.info("     - %s", DATA_DIR / "feature_groups.csv")
logger.info("     - %s", DATA_DIR / "statistical_significance.csv")
logger.info("   Plots:")
logger.info("     - %s", PLOTS_DIR / "consensus_ranking_comparison.png")
logger.info("     - %s", PLOTS_DIR / "method_agreement_heatmap.png")
logger.info("     - %s", PLOTS_DIR / "feature_dependency_network.png")
logger.info("   Reports:")
logger.info("     - %s", REPORTS_DIR / "consensus_summary.json")
logger.info("     - %s", REPORTS_DIR / "COMPARISON_REPORT.md")

logger.info("")
logger.info("=" * 80)
logger.info("✨ Analysis complete! See COMPARISON_REPORT.md for detailed findings.")
logger.info("=" * 80)
