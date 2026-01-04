#!/usr/bin/env python3
"""
Intra-Cluster Similarity Analyzer - GHSOM Clustering Quality Evaluation

This script analyzes feature-based similarity within GHSOM clusters to evaluate
clustering quality and effectiveness. Computes multiple similarity metrics and
generates comprehensive reports.

Author: GHSOM Analysis Pipeline
Date: 2025-11-19
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import variation
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class ClusterSimilarityAnalyzer:
    """Analyze intra-cluster similarity using metadata features."""

    def __init__(
        self,
        cluster_csv: Path,
        metadata_csv: Path,
        output_dir: Path,
        feature_columns: Optional[List[str]] = None,
        distance_metric: str = "euclidean",
    ):
        """
        Initialize the similarity analyzer.

        Args:
            cluster_csv: Path to sample_to_cluster.csv
            metadata_csv: Path to features_with_metadata.csv
            output_dir: Output directory for analysis results
            feature_columns: List of feature columns to use (None = all numeric)
            distance_metric: Distance metric for similarity calculation
        """
        self.cluster_csv = Path(cluster_csv)
        self.metadata_csv = Path(metadata_csv)
        self.output_dir = Path(output_dir)
        self.feature_columns = feature_columns
        self.distance_metric = distance_metric

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.cluster_data = None
        self.metadata = None
        self.merged_data = None
        self.numeric_features = None

    def load_data(self) -> None:
        """Load and merge cluster assignments with metadata."""
        print(f"Loading cluster assignments from {self.cluster_csv}")
        self.cluster_data = pd.read_csv(self.cluster_csv)
        print(f"  Loaded {len(self.cluster_data)} assignments")

        print(f"\nLoading metadata from {self.metadata_csv}")
        self.metadata = pd.read_csv(self.metadata_csv)
        print(
            f"  Loaded {len(self.metadata)} entries with {len(self.metadata.columns)} columns"
        )

        # Merge data
        self.merged_data = self.cluster_data.merge(
            self.metadata, left_on="sample_index", right_on="metadata_index", how="left"
        )
        print(f"  Merged data shape: {self.merged_data.shape}")

        # Identify numeric feature columns
        if self.feature_columns:
            self.numeric_features = self.feature_columns
        else:
            # Auto-detect numeric columns (exclude identifiers and metadata)
            exclude_prefixes = [
                "Unnamed",
                "id",
                "track_id",
                "file_",
                "split_",
                "_adapted",
                "_adapter",
                "metadata_index",
                "sample_index",
                "GHSOM_cluster",
                "audio_key",
                "chord_progressions",
                "genre",
                "track_role",
                "inst",
                "sample_rhythm",
                "time_signature",
            ]

            numeric_cols = self.metadata.select_dtypes(include=[np.number]).columns
            self.numeric_features = [
                col
                for col in numeric_cols
                if not any(col.startswith(prefix) for prefix in exclude_prefixes)
            ]

        print(f"  Using {len(self.numeric_features)} numeric features")
        print(f"  Feature preview: {self.numeric_features[:5]}...")

    def compute_cluster_statistics(self) -> pd.DataFrame:
        """
        Compute statistical measures for each cluster.

        Returns:
            DataFrame with cluster statistics
        """
        print(f"\n{'='*60}")
        print("Computing cluster statistics...")
        print(f"{'='*60}\n")

        cluster_stats = []
        unique_clusters = sorted(self.merged_data["GHSOM_cluster"].unique())

        for cluster_id in unique_clusters:
            cluster_samples = self.merged_data[
                self.merged_data["GHSOM_cluster"] == cluster_id
            ]

            if len(cluster_samples) < 2:
                print(
                    f"  Cluster {cluster_id}: Only {len(cluster_samples)} sample(s) - skipping"
                )
                continue

            # Extract feature values
            features = cluster_samples[self.numeric_features].values

            # Handle NaN values
            features = np.nan_to_num(features, nan=0.0)

            # Compute statistics
            stats = {
                "cluster_id": int(cluster_id),
                "n_samples": len(cluster_samples),
                "mean_pairwise_distance": None,
                "std_pairwise_distance": None,
                "min_pairwise_distance": None,
                "max_pairwise_distance": None,
                "coefficient_of_variation": None,
                "feature_variance_mean": None,
                "feature_variance_std": None,
                "cohesion": None,
            }

            # Pairwise distances
            if len(cluster_samples) >= 2:
                distances = pdist(features, metric=self.distance_metric)

                stats["mean_pairwise_distance"] = float(np.mean(distances))
                stats["std_pairwise_distance"] = float(np.std(distances))
                stats["min_pairwise_distance"] = float(np.min(distances))
                stats["max_pairwise_distance"] = float(np.max(distances))
                stats["coefficient_of_variation"] = float(variation(distances))

            # Feature-wise variance
            feature_vars = np.var(features, axis=0)
            stats["feature_variance_mean"] = float(np.mean(feature_vars))
            stats["feature_variance_std"] = float(np.std(feature_vars))

            # Cohesion (inverse of mean distance)
            if (
                stats["mean_pairwise_distance"] is not None
                and stats["mean_pairwise_distance"] > 0
            ):
                stats["cohesion"] = 1.0 / stats["mean_pairwise_distance"]

            cluster_stats.append(stats)
            print(
                f"  Cluster {cluster_id}: {len(cluster_samples)} samples, "
                f"mean dist = {stats['mean_pairwise_distance']:.4f}"
            )

        stats_df = pd.DataFrame(cluster_stats)

        print(f"\n{'='*60}")
        print("CLUSTER STATISTICS SUMMARY")
        print(f"{'='*60}")
        print(f"Total clusters analyzed: {len(stats_df)}")
        print(f"\nMean pairwise distance:")
        print(f"  Overall mean: {stats_df['mean_pairwise_distance'].mean():.4f}")
        print(f"  Overall std: {stats_df['mean_pairwise_distance'].std():.4f}")
        print(
            f"  Range: [{stats_df['mean_pairwise_distance'].min():.4f}, "
            f"{stats_df['mean_pairwise_distance'].max():.4f}]"
        )
        print(f"{'='*60}\n")

        return stats_df

    def compute_inter_cluster_distances(self) -> pd.DataFrame:
        """
        Compute distances between cluster centroids.

        Returns:
            DataFrame with inter-cluster distance matrix
        """
        print("Computing inter-cluster distances...")

        unique_clusters = sorted(self.merged_data["GHSOM_cluster"].unique())
        centroids = []
        cluster_ids = []

        for cluster_id in unique_clusters:
            cluster_samples = self.merged_data[
                self.merged_data["GHSOM_cluster"] == cluster_id
            ]
            features = cluster_samples[self.numeric_features].values
            features = np.nan_to_num(features, nan=0.0)

            centroid = np.mean(features, axis=0)
            centroids.append(centroid)
            cluster_ids.append(cluster_id)

        # Compute pairwise distances between centroids
        centroids = np.array(centroids)
        dist_matrix = squareform(pdist(centroids, metric=self.distance_metric))

        # Create DataFrame
        dist_df = pd.DataFrame(dist_matrix, index=cluster_ids, columns=cluster_ids)

        print(f"  Computed distances for {len(cluster_ids)} clusters")
        print(
            f"  Mean inter-cluster distance: {dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean():.4f}"
        )

        return dist_df

    def compute_separation_score(
        self, intra_stats: pd.DataFrame, inter_distances: pd.DataFrame
    ) -> Dict:
        """
        Compute cluster separation quality scores.

        Args:
            intra_stats: Intra-cluster statistics
            inter_distances: Inter-cluster distance matrix

        Returns:
            Dictionary with separation metrics
        """
        print("\nComputing separation scores...")

        # Average intra-cluster distance
        mean_intra = intra_stats["mean_pairwise_distance"].mean()

        # Average inter-cluster distance (upper triangle, excluding diagonal)
        inter_vals = inter_distances.values[
            np.triu_indices_from(inter_distances.values, k=1)
        ]
        mean_inter = inter_vals.mean()

        # Separation ratio (higher is better)
        separation_ratio = mean_inter / mean_intra if mean_intra > 0 else 0

        # Dunn index (min inter / max intra)
        min_inter = inter_vals.min()
        max_intra = intra_stats["max_pairwise_distance"].max()
        dunn_index = min_inter / max_intra if max_intra > 0 else 0

        scores = {
            "mean_intra_cluster_distance": float(mean_intra),
            "mean_inter_cluster_distance": float(mean_inter),
            "separation_ratio": float(separation_ratio),
            "dunn_index": float(dunn_index),
            "interpretation": {
                "separation_ratio": (
                    "Excellent"
                    if separation_ratio > 2.0
                    else "Good" if separation_ratio > 1.5 else "Fair"
                ),
                "dunn_index": (
                    "Excellent"
                    if dunn_index > 1.0
                    else "Good" if dunn_index > 0.5 else "Poor"
                ),
            },
        }

        print(f"\n{'='*60}")
        print("SEPARATION SCORES")
        print(f"{'='*60}")
        print(
            f"Mean intra-cluster distance: {scores['mean_intra_cluster_distance']:.4f}"
        )
        print(
            f"Mean inter-cluster distance: {scores['mean_inter_cluster_distance']:.4f}"
        )
        print(
            f"Separation ratio: {scores['separation_ratio']:.4f} ({scores['interpretation']['separation_ratio']})"
        )
        print(
            f"Dunn index: {scores['dunn_index']:.4f} ({scores['interpretation']['dunn_index']})"
        )
        print(f"{'='*60}\n")

        return scores

    def visualize_results(
        self, intra_stats: pd.DataFrame, inter_distances: pd.DataFrame
    ) -> None:
        """Generate visualization plots."""
        print("Generating visualizations...")

        # Set style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))

        # 1. Intra-cluster distance distribution
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(
            intra_stats["mean_pairwise_distance"],
            bins=20,
            edgecolor="black",
            alpha=0.7,
            color="steelblue",
        )
        ax1.set_xlabel("Mean Pairwise Distance")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Intra-Cluster Distance Distribution")
        ax1.axvline(
            intra_stats["mean_pairwise_distance"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cluster size vs cohesion
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(
            intra_stats["n_samples"],
            intra_stats["mean_pairwise_distance"],
            c=intra_stats["cluster_id"],
            cmap="viridis",
            s=50,
            alpha=0.6,
            edgecolors="black",
        )
        ax2.set_xlabel("Number of Samples")
        ax2.set_ylabel("Mean Pairwise Distance")
        ax2.set_title("Cluster Size vs Intra-Cluster Distance")
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label="Cluster ID")

        # 3. Coefficient of variation
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(
            range(len(intra_stats)),
            intra_stats["coefficient_of_variation"],
            color="coral",
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_xlabel("Cluster Index")
        ax3.set_ylabel("Coefficient of Variation")
        ax3.set_title("Intra-Cluster Variability")
        ax3.axhline(
            y=intra_stats["coefficient_of_variation"].mean(),
            color="red",
            linestyle="--",
            label="Mean",
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Inter-cluster distance heatmap
        ax4 = plt.subplot(2, 3, 4)
        sns.heatmap(
            inter_distances,
            cmap="YlOrRd",
            ax=ax4,
            cbar_kws={"label": "Distance"},
            square=True,
        )
        ax4.set_title("Inter-Cluster Distance Matrix")
        ax4.set_xlabel("Cluster ID")
        ax4.set_ylabel("Cluster ID")

        # 5. Distance comparison boxplot
        ax5 = plt.subplot(2, 3, 5)
        intra_vals = intra_stats["mean_pairwise_distance"].values
        inter_vals = inter_distances.values[
            np.triu_indices_from(inter_distances.values, k=1)
        ]

        bp = ax5.boxplot(
            [intra_vals, inter_vals],
            labels=["Intra-Cluster", "Inter-Cluster"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightcoral")
        ax5.set_ylabel("Distance")
        ax5.set_title("Intra vs Inter-Cluster Distances")
        ax5.grid(True, alpha=0.3, axis="y")

        # 6. Feature variance per cluster
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(
            intra_stats["feature_variance_mean"],
            intra_stats["feature_variance_std"],
            c=intra_stats["cluster_id"],
            cmap="plasma",
            s=intra_stats["n_samples"] * 2,
            alpha=0.6,
            edgecolors="black",
        )
        ax6.set_xlabel("Mean Feature Variance")
        ax6.set_ylabel("Std Feature Variance")
        ax6.set_title("Feature Variance Distribution (size = n_samples)")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = self.output_dir / "cluster_similarity_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  Saved visualization to {output_file}")
        plt.close()

    def generate_report(
        self,
        intra_stats: pd.DataFrame,
        inter_distances: pd.DataFrame,
        separation_scores: Dict,
    ) -> None:
        """Generate comprehensive analysis report."""
        print("\nGenerating analysis report...")

        # Save statistics to CSV
        stats_file = self.output_dir / "intra_cluster_statistics.csv"
        intra_stats.to_csv(stats_file, index=False)
        print(f"  Saved intra-cluster statistics to {stats_file}")

        # Save inter-cluster distances
        inter_file = self.output_dir / "inter_cluster_distances.csv"
        inter_distances.to_csv(inter_file)
        print(f"  Saved inter-cluster distances to {inter_file}")

        # Save JSON summary
        summary = {
            "analysis_config": {
                "cluster_csv": str(self.cluster_csv),
                "metadata_csv": str(self.metadata_csv),
                "n_features": len(self.numeric_features),
                "distance_metric": self.distance_metric,
            },
            "cluster_summary": {
                "n_clusters": len(intra_stats),
                "total_samples": int(intra_stats["n_samples"].sum()),
                "mean_cluster_size": float(intra_stats["n_samples"].mean()),
                "std_cluster_size": float(intra_stats["n_samples"].std()),
            },
            "intra_cluster_analysis": {
                "mean_distance": float(intra_stats["mean_pairwise_distance"].mean()),
                "std_distance": float(intra_stats["mean_pairwise_distance"].std()),
                "min_distance": float(intra_stats["mean_pairwise_distance"].min()),
                "max_distance": float(intra_stats["mean_pairwise_distance"].max()),
                "most_cohesive_cluster": int(
                    intra_stats.loc[
                        intra_stats["mean_pairwise_distance"].idxmin(), "cluster_id"
                    ]
                ),
                "least_cohesive_cluster": int(
                    intra_stats.loc[
                        intra_stats["mean_pairwise_distance"].idxmax(), "cluster_id"
                    ]
                ),
            },
            "separation_analysis": separation_scores,
            "feature_columns_used": self.numeric_features,
        }

        summary_file = self.output_dir / "analysis_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved analysis summary to {summary_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze intra-cluster similarity for GHSOM clustering evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cluster-csv", type=Path, required=True, help="Path to sample_to_cluster.csv"
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Path to features_with_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--distance-metric",
        default="euclidean",
        choices=["euclidean", "manhattan", "cosine", "correlation"],
        help="Distance metric for similarity calculation (default: euclidean)",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("GHSOM INTRA-CLUSTER SIMILARITY ANALYSIS")
    print(f"{'='*70}\n")

    # Initialize analyzer
    analyzer = ClusterSimilarityAnalyzer(
        cluster_csv=args.cluster_csv,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        distance_metric=args.distance_metric,
    )

    # Execute analysis
    analyzer.load_data()
    intra_stats = analyzer.compute_cluster_statistics()
    inter_distances = analyzer.compute_inter_cluster_distances()
    separation_scores = analyzer.compute_separation_score(intra_stats, inter_distances)
    analyzer.visualize_results(intra_stats, inter_distances)
    analyzer.generate_report(intra_stats, inter_distances, separation_scores)

    print(f"\n{'='*70}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*70}\n")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Analyzed {len(intra_stats)} clusters with {intra_stats['n_samples'].sum():.0f} total samples\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
