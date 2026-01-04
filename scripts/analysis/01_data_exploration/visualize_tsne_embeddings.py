#!/usr/bin/env python
"""
Dynamic t-SNE Embeddings Visualization Script
Generates comprehensive visualizations comparing t-SNE embeddings from multiple datasets.
Dataset names can be specified via command-line arguments.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import warnings
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logging.logging_manager import LoggingManager

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# Define paths
BASE_PATH = Path("/workspace")
TSNE_PATH = BASE_PATH / "artifacts/features/tsne"
FEATURES_PATH = BASE_PATH / "artifacts/features/raw"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualizations for specified datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default datasets (commu_full and commu_bass)
  python run_tsne_analysis.py
  
  # Specify custom datasets
  python run_tsne_analysis.py --datasets commu_full commu_bass lm_bass
  
  # Specify output directory
  python run_tsne_analysis.py --output-dir /custom/path
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["commu_full", "commu_bass"],
        help="Names of datasets to visualize (default: commu_full commu_bass)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: outputs/eda/tsne_visualizations_<timestamp>)",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters for K-means (default: auto-detect optimal k)",
    )

    parser.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        default=[5, 15],
        help="Range of k values to test for auto-detection (default: 5 15)",
    )

    return parser.parse_args()


def setup_output_directory(output_dir=None, dataset_names=None):
    """Create and return the output directory path."""
    if output_dir:
        output_path = Path(output_dir)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_str = "_".join(dataset_names[:2]) if dataset_names else "default"
        output_path = (
            BASE_PATH / f"outputs/eda/tsne_visualizations_{dataset_str}_{timestamp}"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_dataset(dataset_name, logger):
    """Load t-SNE embeddings and merge with metadata."""
    # Try with _tsne suffix first, then without
    tsne_path = TSNE_PATH / f"{dataset_name}_tsne" / "embedding.csv"
    if not tsne_path.exists():
        tsne_path = TSNE_PATH / f"{dataset_name}" / "embedding.csv"

    if not tsne_path.exists():
        logger.error("t-SNE file not found: %s", tsne_path)
        raise FileNotFoundError(
            f"t-SNE embeddings not found for dataset: {dataset_name}"
        )

    embeddings = pd.read_csv(tsne_path)

    # Determine which metadata file to use
    metadata_path = FEATURES_PATH / dataset_name / "features_with_metadata.csv"

    # If filtered dataset, try to use the base commu_full metadata
    if not metadata_path.exists() and "filtered" in dataset_name:
        base_dataset = dataset_name.replace("_filtered", "")
        metadata_path = FEATURES_PATH / base_dataset / "features_with_metadata.csv"
        if metadata_path.exists():
            logger.info("Using metadata from base dataset: %s", base_dataset)

    if not metadata_path.exists():
        logger.warning(
            "Metadata file not found: %s. Using embeddings only.", metadata_path
        )
        logger.info(
            "✓ %s: %d samples loaded (embeddings only)", dataset_name, len(embeddings)
        )
        return embeddings

    metadata = pd.read_csv(metadata_path, low_memory=False)

    # Determine merge columns based on what's available in embeddings
    merge_cols = []
    if "track_id" in embeddings.columns:
        merge_cols.append("track_id")
    if "metadata_index" in embeddings.columns:
        merge_cols.append("metadata_index")

    if not merge_cols:
        logger.warning("No common merge columns found. Using embeddings only.")
        logger.info(
            "✓ %s: %d samples loaded (embeddings only)", dataset_name, len(embeddings)
        )
        return embeddings

    merged = embeddings.merge(metadata, on=merge_cols, how="left")
    logger.info(
        "✓ %s: %d samples loaded (merged on %s)",
        dataset_name,
        len(merged),
        ", ".join(merge_cols),
    )
    return merged


def generate_statistics(datasets_data, dataset_names, logger):
    """Generate and log dataset statistics."""
    logger.info("")
    logger.info("2. Dataset Statistics:")
    logger.info("-" * 80)

    for name, df in zip(dataset_names, datasets_data):
        logger.info("")
        logger.info("%s:", name.upper())
        logger.info("  Samples: %d", len(df))
        logger.info(
            "  Embedding range: dim1=[%.2f, %.2f], dim2=[%.2f, %.2f]",
            df["dim1"].min(),
            df["dim1"].max(),
            df["dim2"].min(),
            df["dim2"].max(),
        )
        if "genre" in df.columns:
            logger.info("  Unique genres: %d", df["genre"].nunique())
        if "bpm" in df.columns:
            logger.info("  BPM range: [%.1f, %.1f]", df["bpm"].min(), df["bpm"].max())
        if "source" in df.columns:
            logger.info("  Data sources: %s", ", ".join(df["source"].unique()))


def visualize_individual_datasets(datasets_data, dataset_names, output_path, logger):
    """Generate individual visualizations for each dataset."""
    logger.info("")
    logger.info("3. Generating Individual Dataset Visualizations...")

    for name, df in zip(dataset_names, datasets_data):
        logger.info("  → Processing %s...", name)

        # Create figure with multiple subplots
        n_plots = 2
        if "genre" in df.columns and df["genre"].nunique() > 1:
            n_plots += 1
        if "bpm" in df.columns:
            n_plots += 1

        rows = (n_plots + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(20, 8 * rows))
        axes = axes.flatten() if n_plots > 1 else [axes]

        plot_idx = 0

        # Plot 1: Basic scatter
        axes[plot_idx].scatter(
            df["dim1"], df["dim2"], alpha=0.6, s=20, color="steelblue"
        )
        axes[plot_idx].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[plot_idx].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[plot_idx].set_title(f"{name}: All Samples", fontsize=14, fontweight="bold")
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 2: Density heatmap
        axes[plot_idx].hexbin(
            df["dim1"], df["dim2"], gridsize=50, cmap="YlOrRd", mincnt=1
        )
        axes[plot_idx].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[plot_idx].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[plot_idx].set_title(
            f"{name}: Density Heatmap", fontsize=14, fontweight="bold"
        )
        plt.colorbar(
            axes[plot_idx].collections[0], ax=axes[plot_idx], label="Point Density"
        )
        plot_idx += 1

        # Plot 3: By genre (if available)
        if "genre" in df.columns and df["genre"].nunique() > 1:
            top_n = min(10, df["genre"].nunique())
            top_genres = df["genre"].value_counts().head(top_n).index
            df_plot = df[df["genre"].isin(top_genres)].copy()
            colors = sns.color_palette("tab10", n_colors=len(top_genres))
            genre_colors = dict(zip(top_genres, colors))

            for genre in top_genres:
                mask = df_plot["genre"] == genre
                axes[plot_idx].scatter(
                    df_plot.loc[mask, "dim1"],
                    df_plot.loc[mask, "dim2"],
                    alpha=0.6,
                    s=20,
                    color=genre_colors[genre],
                    label=str(genre)[:25],
                )
            axes[plot_idx].set_xlabel("t-SNE Dimension 1", fontsize=12)
            axes[plot_idx].set_ylabel("t-SNE Dimension 2", fontsize=12)
            axes[plot_idx].set_title(
                f"{name}: By Genre (Top {top_n})", fontsize=14, fontweight="bold"
            )
            axes[plot_idx].legend(fontsize=9, loc="best", ncol=2)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Plot 4: By BPM (if available)
        if "bpm" in df.columns:
            df_clean = df.dropna(subset=["bpm"])
            if len(df_clean) > 0:
                scatter = axes[plot_idx].scatter(
                    df_clean["dim1"],
                    df_clean["dim2"],
                    c=df_clean["bpm"],
                    cmap="viridis",
                    alpha=0.6,
                    s=20,
                    vmin=df_clean["bpm"].quantile(0.05),
                    vmax=df_clean["bpm"].quantile(0.95),
                )
                axes[plot_idx].set_xlabel("t-SNE Dimension 1", fontsize=12)
                axes[plot_idx].set_ylabel("t-SNE Dimension 2", fontsize=12)
                axes[plot_idx].set_title(
                    f"{name}: By BPM (Tempo)", fontsize=14, fontweight="bold"
                )
                plt.colorbar(scatter, ax=axes[plot_idx], label="BPM")
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            output_path / f"tsne_{safe_name}_detailed.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def visualize_comparison(datasets_data, dataset_names, output_path, logger):
    """Generate side-by-side comparison visualization."""
    logger.info("")
    logger.info("4. Generating Comparison Visualization...")

    n_datasets = len(datasets_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 7))

    if n_datasets == 1:
        axes = [axes]

    for idx, (name, df) in enumerate(zip(dataset_names, datasets_data)):
        ax = axes[idx]

        if (
            "genre" in df.columns
            and df["genre"].nunique() <= 10
            and df["genre"].nunique() > 1
        ):
            genres = df["genre"].unique()
            colors = sns.color_palette("husl", n_colors=len(genres))
            genre_colors = dict(zip(genres, colors))

            for genre in genres:
                mask = df["genre"] == genre
                ax.scatter(
                    df.loc[mask, "dim1"],
                    df.loc[mask, "dim2"],
                    alpha=0.6,
                    s=20,
                    color=genre_colors[genre],
                    label=str(genre)[:20],
                )
            ax.legend(title="Genre", fontsize=8, loc="best")
        else:
            ax.scatter(df["dim1"], df["dim2"], alpha=0.6, s=20, color="steelblue")

        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        ax.set_title(f"{name}\n({len(df):,} samples)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "tsne_comparison_all.png", dpi=150, bbox_inches="tight")
    plt.close()


def find_optimal_clusters(X, k_range=(5, 15), logger=None):
    """Automatically determine optimal number of clusters using multiple metrics.

    Args:
        X: Data array (n_samples, 2)
        k_range: Tuple of (min_k, max_k) to test
        logger: Logger instance

    Returns:
        dict with optimal_k, votes, metrics, and k_values
    """
    k_values = list(range(k_range[0], k_range[1] + 1))

    metrics = {
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
        "inertia": [],
    }

    # Calculate metrics for each k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        metrics["silhouette"].append(silhouette_score(X, labels))
        metrics["davies_bouldin"].append(davies_bouldin_score(X, labels))
        metrics["calinski_harabasz"].append(calinski_harabasz_score(X, labels))
        metrics["inertia"].append(kmeans.inertia_)

    # Voting system: Each metric suggests optimal k
    votes = []
    vote_methods = []

    # Silhouette: maximize
    silhouette_best_idx = np.argmax(metrics["silhouette"])
    votes.append(k_values[silhouette_best_idx])
    vote_methods.append("silhouette")

    # Davies-Bouldin: minimize
    db_best_idx = np.argmin(metrics["davies_bouldin"])
    votes.append(k_values[db_best_idx])
    vote_methods.append("davies_bouldin")

    # Calinski-Harabasz: maximize with penalty for high k
    weighted_ch = [
        score / (1 + 0.1 * (k - k_range[0]))
        for k, score in zip(k_values, metrics["calinski_harabasz"])
    ]
    ch_best_idx = np.argmax(weighted_ch)
    votes.append(k_values[ch_best_idx])
    vote_methods.append("calinski_harabasz")

    # Elbow detection for inertia (simple implementation)
    inertia_diffs = np.diff(metrics["inertia"])
    inertia_diffs2 = np.diff(inertia_diffs)
    if len(inertia_diffs2) > 0:
        elbow_idx = np.argmax(inertia_diffs2) + 2  # +2 because of two diffs
        if elbow_idx < len(k_values):
            votes.append(k_values[elbow_idx])
            vote_methods.append("elbow")

    # Select most voted k (mode)
    from collections import Counter

    vote_counts = Counter(votes)
    optimal_k = vote_counts.most_common(1)[0][0]

    return {
        "optimal_k": optimal_k,
        "votes": votes,
        "vote_methods": vote_methods,
        "vote_counts": dict(vote_counts),
        "metrics": metrics,
        "k_values": k_values,
    }


def generate_cluster_selection_plots(
    datasets_data, dataset_names, cluster_results, output_path, logger
):
    """Generate diagnostic plots showing cluster selection process."""
    logger.info("")
    logger.info("  → Generating cluster selection diagnostic plots...")

    for name, df, (labels, optimal_k, result) in zip(
        dataset_names, datasets_data, cluster_results
    ):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        k_values = result["k_values"]
        metrics = result["metrics"]

        # Plot 1: Silhouette Score
        axes[0, 0].plot(
            k_values, metrics["silhouette"], "o-", linewidth=2, markersize=8
        )
        silhouette_best = k_values[np.argmax(metrics["silhouette"])]
        axes[0, 0].axvline(
            x=silhouette_best,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Best k={silhouette_best}",
        )
        axes[0, 0].axvline(
            x=optimal_k,
            color="g",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"Selected k={optimal_k}",
        )
        axes[0, 0].set_xlabel("Number of Clusters (k)", fontsize=11)
        axes[0, 0].set_ylabel("Silhouette Score", fontsize=11)
        axes[0, 0].set_title(
            f"{name}: Silhouette Score vs k", fontsize=12, fontweight="bold"
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Davies-Bouldin Index (lower is better)
        axes[0, 1].plot(
            k_values,
            metrics["davies_bouldin"],
            "o-",
            linewidth=2,
            markersize=8,
            color="orange",
        )
        db_best = k_values[np.argmin(metrics["davies_bouldin"])]
        axes[0, 1].axvline(
            x=db_best, color="r", linestyle="--", alpha=0.7, label=f"Best k={db_best}"
        )
        axes[0, 1].axvline(
            x=optimal_k,
            color="g",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"Selected k={optimal_k}",
        )
        axes[0, 1].set_xlabel("Number of Clusters (k)", fontsize=11)
        axes[0, 1].set_ylabel("Davies-Bouldin Index", fontsize=11)
        axes[0, 1].set_title(
            f"{name}: Davies-Bouldin Index vs k (lower=better)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Calinski-Harabasz Score
        axes[1, 0].plot(
            k_values,
            metrics["calinski_harabasz"],
            "o-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ch_best = k_values[np.argmax(metrics["calinski_harabasz"])]
        axes[1, 0].axvline(
            x=ch_best, color="r", linestyle="--", alpha=0.7, label=f"Best k={ch_best}"
        )
        axes[1, 0].axvline(
            x=optimal_k,
            color="g",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"Selected k={optimal_k}",
        )
        axes[1, 0].set_xlabel("Number of Clusters (k)", fontsize=11)
        axes[1, 0].set_ylabel("Calinski-Harabasz Score", fontsize=11)
        axes[1, 0].set_title(
            f"{name}: Calinski-Harabasz Score vs k", fontsize=12, fontweight="bold"
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Inertia/Elbow
        axes[1, 1].plot(
            k_values,
            metrics["inertia"],
            "o-",
            linewidth=2,
            markersize=8,
            color="purple",
        )
        axes[1, 1].axvline(
            x=optimal_k,
            color="g",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"Selected k={optimal_k}",
        )
        axes[1, 1].set_xlabel("Number of Clusters (k)", fontsize=11)
        axes[1, 1].set_ylabel("Inertia (Within-cluster sum of squares)", fontsize=11)
        axes[1, 1].set_title(
            f"{name}: Elbow Method (Inertia vs k)", fontsize=12, fontweight="bold"
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add voting summary as text
        vote_text = f"Votes: {result['votes']}\nMethods: {result['vote_methods']}\nCounts: {result['vote_counts']}"
        fig.text(
            0.5,
            0.02,
            vote_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            output_path / f"tsne_cluster_optimization_{safe_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def perform_cluster_analysis(
    datasets_data, dataset_names, n_clusters, k_range, output_path, logger
):
    """Perform K-means clustering analysis with automatic or manual k selection."""

    if n_clusters is None:
        logger.info("")
        logger.info(
            "5. Cluster Analysis (Auto K-means with k=%d to %d)...",
            k_range[0],
            k_range[1],
        )
        logger.info("-" * 80)
        auto_mode = True
    else:
        logger.info("")
        logger.info(
            "5. Cluster Analysis (K-means with k=%d - manual mode)...", n_clusters
        )
        logger.info("-" * 80)
        auto_mode = False

    cluster_results = []
    optimal_ks = {}
    selection_summary = []

    for name, df in zip(dataset_names, datasets_data):
        X = df[["dim1", "dim2"]].values

        if auto_mode:
            # Find optimal k
            result = find_optimal_clusters(X, k_range, logger)
            optimal_k = result["optimal_k"]
            optimal_ks[name] = optimal_k

            # Perform final clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette = silhouette_score(X, labels)

            logger.info("")
            logger.info("%s:", name)
            logger.info("  Optimal k: %d (auto-detected)", optimal_k)
            logger.info("  Method votes: %s", result["votes"])
            logger.info("  Vote methods: %s", result["vote_methods"])
            logger.info("  Vote counts: %s", result["vote_counts"])
            logger.info("  Silhouette score: %.3f", silhouette)
            logger.info(
                "  Davies-Bouldin: %.3f",
                result["metrics"]["davies_bouldin"][optimal_k - k_range[0]],
            )
            logger.info(
                "  Calinski-Harabasz: %.1f",
                result["metrics"]["calinski_harabasz"][optimal_k - k_range[0]],
            )
            logger.info("  Cluster sizes: %s", np.bincount(labels))

            cluster_results.append((labels, optimal_k, result))

            # Store for CSV export
            selection_summary.append(
                {
                    "dataset": name,
                    "optimal_k": optimal_k,
                    "silhouette": silhouette,
                    "davies_bouldin": result["metrics"]["davies_bouldin"][
                        optimal_k - k_range[0]
                    ],
                    "calinski_harabasz": result["metrics"]["calinski_harabasz"][
                        optimal_k - k_range[0]
                    ],
                    "votes": str(result["votes"]),
                    "vote_methods": str(result["vote_methods"]),
                }
            )
        else:
            # Use fixed k
            optimal_k = n_clusters
            optimal_ks[name] = optimal_k

            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            silhouette = silhouette_score(X, labels)

            logger.info("")
            logger.info("%s:", name)
            logger.info("  k: %d (manual)", optimal_k)
            logger.info("  Silhouette score: %.3f", silhouette)
            logger.info("  Cluster sizes: %s", np.bincount(labels))

            cluster_results.append((labels, optimal_k, None))

    # Visualization
    logger.info("")
    logger.info("  → Cluster visualization...")

    n_datasets = len(datasets_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 7))

    if n_datasets == 1:
        axes = [axes]

    for idx, (name, df, (labels, optimal_k, _)) in enumerate(
        zip(dataset_names, datasets_data, cluster_results)
    ):
        ax = axes[idx]

        # Create legend by plotting each cluster separately
        colors = sns.color_palette("tab10", n_colors=optimal_k)
        for cluster_id in range(optimal_k):
            mask = labels == cluster_id
            ax.scatter(
                df.loc[mask, "dim1"],
                df.loc[mask, "dim2"],
                alpha=0.6,
                s=20,
                color=colors[cluster_id],
                label=f"Cluster {cluster_id}",
            )

        ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
        mode_text = "auto" if auto_mode else "manual"
        ax.set_title(
            f"{name}: K-means Clusters (k={optimal_k}, {mode_text})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=8, loc="best", title="Clusters")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "tsne_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Generate diagnostic plots if auto mode
    if auto_mode:
        generate_cluster_selection_plots(
            datasets_data, dataset_names, cluster_results, output_path, logger
        )

        # Export cluster selection summary
        if selection_summary:
            summary_df = pd.DataFrame(selection_summary)
            summary_df.to_csv(
                output_path / "tsne_cluster_selection_summary.csv", index=False
            )
            logger.info("  → Cluster selection summary exported")

    return cluster_results, optimal_ks


def export_summary(datasets_data, dataset_names, output_path, logger):
    """Export summary statistics to CSV."""
    logger.info("")
    logger.info("6. Exporting Summary Statistics...")

    summary_data = []

    for name, df in zip(dataset_names, datasets_data):
        summary_data.append(
            {
                "dataset": name,
                "n_samples": len(df),
                "dim1_mean": df["dim1"].mean(),
                "dim1_std": df["dim1"].std(),
                "dim2_mean": df["dim2"].mean(),
                "dim2_std": df["dim2"].std(),
                "n_genres": df["genre"].nunique() if "genre" in df.columns else None,
                "bpm_mean": df["bpm"].mean() if "bpm" in df.columns else None,
                "bpm_std": df["bpm"].std() if "bpm" in df.columns else None,
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "tsne_summary_statistics.csv", index=False)


def visualize_by_audio_key(datasets_data, dataset_names, output_path, logger):
    """Generate visualizations colored by audio key for each dataset."""
    logger.info("")
    logger.info("7. Generating Audio Key Visualizations...")

    for name, df in zip(dataset_names, datasets_data):
        if "audio_key" not in df.columns:
            logger.warning("  ⊘ Skipping %s: 'audio_key' column not found", name)
            continue

        # Filter out missing values
        df_clean = df.dropna(subset=["audio_key"])
        if len(df_clean) == 0:
            logger.warning("  ⊘ Skipping %s: no valid audio_key data", name)
            continue

        logger.info(
            "  → Processing %s (%d samples with audio_key)...", name, len(df_clean)
        )

        # Get top N keys
        top_n = min(12, df_clean["audio_key"].nunique())
        top_keys = df_clean["audio_key"].value_counts().head(top_n).index
        df_plot = df_clean[df_clean["audio_key"].isin(top_keys)].copy()

        # Create figure with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: All samples by key
        colors = sns.color_palette("husl", n_colors=len(top_keys))
        key_colors = dict(zip(top_keys, colors))

        for key in top_keys:
            mask = df_plot["audio_key"] == key
            axes[0].scatter(
                df_plot.loc[mask, "dim1"],
                df_plot.loc[mask, "dim2"],
                alpha=0.6,
                s=30,
                color=key_colors[key],
                label=str(key),
            )
        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title(
            f"{name}: By Audio Key (Top {top_n})\n{len(df_plot):,} samples",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].legend(fontsize=9, loc="best", ncol=2, title="Key")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Distribution summary
        key_counts = df_clean["audio_key"].value_counts().head(top_n)
        axes[1].barh(range(len(key_counts)), key_counts.values)
        axes[1].set_yticks(range(len(key_counts)))
        axes[1].set_yticklabels(key_counts.index)
        axes[1].set_xlabel("Number of Samples", fontsize=12)
        axes[1].set_title(
            f"{name}: Audio Key Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            output_path / f"tsne_{safe_name}_by_audio_key.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def visualize_by_num_measures(datasets_data, dataset_names, output_path, logger):
    """Generate visualizations colored by number of measures for each dataset."""
    logger.info("")
    logger.info("8. Generating Num Measures Visualizations...")

    for name, df in zip(dataset_names, datasets_data):
        if "num_measures" not in df.columns:
            logger.warning("  ⊘ Skipping %s: 'num_measures' column not found", name)
            continue

        # Filter out missing values
        df_clean = df.dropna(subset=["num_measures"])
        if len(df_clean) == 0:
            logger.warning("  ⊘ Skipping %s: no valid num_measures data", name)
            continue

        logger.info(
            "  → Processing %s (%d samples with num_measures)...", name, len(df_clean)
        )

        # Create figure with multiple views
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Continuous color gradient
        scatter = axes[0].scatter(
            df_clean["dim1"],
            df_clean["dim2"],
            c=df_clean["num_measures"],
            cmap="viridis",
            alpha=0.6,
            s=30,
            vmin=df_clean["num_measures"].quantile(0.05),
            vmax=df_clean["num_measures"].quantile(0.95),
        )
        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title(
            f"{name}: By Number of Measures\n{len(df_clean):,} samples",
            fontsize=14,
            fontweight="bold",
        )
        plt.colorbar(scatter, ax=axes[0], label="Number of Measures")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Binned categories
        bins = [0, 4, 8, 16, 32, np.inf]
        labels = ["1-4", "5-8", "9-16", "17-32", "33+"]
        df_clean["measure_bin"] = pd.cut(
            df_clean["num_measures"], bins=bins, labels=labels
        )

        colors = sns.color_palette("RdYlGn_r", n_colors=len(labels))
        bin_colors = dict(zip(labels, colors))

        for bin_label in labels:
            mask = df_clean["measure_bin"] == bin_label
            if mask.sum() > 0:
                axes[1].scatter(
                    df_clean.loc[mask, "dim1"],
                    df_clean.loc[mask, "dim2"],
                    alpha=0.6,
                    s=30,
                    color=bin_colors[bin_label],
                    label=f"{bin_label} measures",
                )
        axes[1].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[1].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[1].set_title(
            f"{name}: By Measure Range (Binned)", fontsize=14, fontweight="bold"
        )
        axes[1].legend(fontsize=10, loc="best", title="Measures")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            output_path / f"tsne_{safe_name}_by_num_measures.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def visualize_by_track_role(datasets_data, dataset_names, output_path, logger):
    """Generate visualizations colored by track role for each dataset."""
    logger.info("")
    logger.info("9. Generating Track Role Visualizations...")

    for name, df in zip(dataset_names, datasets_data):
        if "track_role" not in df.columns:
            logger.warning("  ⊘ Skipping %s: 'track_role' column not found", name)
            continue

        # Filter out missing values
        df_clean = df.dropna(subset=["track_role"])
        if len(df_clean) == 0:
            logger.warning("  ⊘ Skipping %s: no valid track_role data", name)
            continue

        logger.info(
            "  → Processing %s (%d samples with track_role)...", name, len(df_clean)
        )

        # Get all unique roles
        unique_roles = df_clean["track_role"].unique()
        n_roles = len(unique_roles)

        # Create figure with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: All roles with colors
        colors = sns.color_palette("Set2", n_colors=n_roles)
        role_colors = dict(zip(unique_roles, colors))

        for role in unique_roles:
            mask = df_clean["track_role"] == role
            axes[0].scatter(
                df_clean.loc[mask, "dim1"],
                df_clean.loc[mask, "dim2"],
                alpha=0.6,
                s=30,
                color=role_colors[role],
                label=str(role),
            )
        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title(
            f"{name}: By Track Role\n{len(df_clean):,} samples",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].legend(fontsize=10, loc="best", title="Role")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Distribution summary
        role_counts = df_clean["track_role"].value_counts()
        axes[1].barh(range(len(role_counts)), role_counts.values)
        axes[1].set_yticks(range(len(role_counts)))
        axes[1].set_yticklabels(role_counts.index)
        axes[1].set_xlabel("Number of Samples", fontsize=12)
        axes[1].set_title(
            f"{name}: Track Role Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            output_path / f"tsne_{safe_name}_by_track_role.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def visualize_by_instrument(datasets_data, dataset_names, output_path, logger):
    """Generate visualizations colored by instrument for each dataset."""
    logger.info("")
    logger.info("10. Generating Instrument Visualizations...")

    for name, df in zip(dataset_names, datasets_data):
        if "inst" not in df.columns:
            logger.warning("  ⊘ Skipping %s: 'inst' column not found", name)
            continue

        # Filter out missing values
        df_clean = df.dropna(subset=["inst"])
        if len(df_clean) == 0:
            logger.warning("  ⊘ Skipping %s: no valid inst data", name)
            continue

        logger.info("  → Processing %s (%d samples with inst)...", name, len(df_clean))

        # Get top N instruments
        top_n = min(15, df_clean["inst"].nunique())
        top_instruments = df_clean["inst"].value_counts().head(top_n).index
        df_plot = df_clean[df_clean["inst"].isin(top_instruments)].copy()

        # Create figure with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: All instruments with colors
        colors = sns.color_palette("tab20", n_colors=len(top_instruments))
        inst_colors = dict(zip(top_instruments, colors))

        for inst in top_instruments:
            mask = df_plot["inst"] == inst
            axes[0].scatter(
                df_plot.loc[mask, "dim1"],
                df_plot.loc[mask, "dim2"],
                alpha=0.6,
                s=30,
                color=inst_colors[inst],
                label=str(inst).replace("_", " "),
            )
        axes[0].set_xlabel("t-SNE Dimension 1", fontsize=12)
        axes[0].set_ylabel("t-SNE Dimension 2", fontsize=12)
        axes[0].set_title(
            f"{name}: By Instrument (Top {top_n})\n{len(df_plot):,} samples",
            fontsize=14,
            fontweight="bold",
        )
        axes[0].legend(fontsize=9, loc="best", ncol=2, title="Instrument")
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Distribution summary
        inst_counts = df_clean["inst"].value_counts().head(top_n)
        axes[1].barh(range(len(inst_counts)), inst_counts.values)
        axes[1].set_yticks(range(len(inst_counts)))
        axes[1].set_yticklabels([str(x).replace("_", " ") for x in inst_counts.index])
        axes[1].set_xlabel("Number of Samples", fontsize=12)
        axes[1].set_title(
            f"{name}: Instrument Distribution", fontsize=14, fontweight="bold"
        )
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        safe_name = name.replace("/", "_").replace(" ", "_")
        plt.savefig(
            output_path / f"tsne_{safe_name}_by_instrument.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Setup output directory
    output_path = setup_output_directory(args.output_dir, args.datasets)

    # Initialize logging
    log_path = BASE_PATH / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    logger_manager = LoggingManager(
        name="run_tsne_analysis",
        log_file=log_path / "run_tsne_analysis.log",
        enable_wandb=False,
    )
    logger = logger_manager.logger

    logger.info("=" * 80)
    logger.info("t-SNE EMBEDDINGS VISUALIZATION AND ANALYSIS (DYNAMIC)")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Datasets to process: %s", ", ".join(args.datasets))
    logger.info("Output directory: %s", output_path)
    if args.n_clusters is None:
        logger.info(
            "Cluster detection: Auto (k range: %d to %d)",
            args.k_range[0],
            args.k_range[1],
        )
    else:
        logger.info("Number of clusters: %d (manual)", args.n_clusters)

    # Load all datasets
    logger.info("")
    logger.info("1. Loading Data...")
    datasets_data = []
    successful_datasets = []

    for dataset_name in args.datasets:
        try:
            df = load_dataset(dataset_name, logger)
            datasets_data.append(df)
            successful_datasets.append(dataset_name)
        except Exception as e:
            logger.error("Failed to load dataset '%s': %s", dataset_name, str(e))
            logger.warning("Skipping dataset '%s'", dataset_name)

    if not datasets_data:
        logger.error("No datasets were successfully loaded. Exiting.")
        return 1

    # Update dataset names to only successful ones
    dataset_names = successful_datasets

    # Generate statistics
    generate_statistics(datasets_data, dataset_names, logger)

    # Generate individual visualizations
    visualize_individual_datasets(datasets_data, dataset_names, output_path, logger)

    # Generate comparison visualization
    visualize_comparison(datasets_data, dataset_names, output_path, logger)

    # Perform cluster analysis
    cluster_results, optimal_ks = perform_cluster_analysis(
        datasets_data, dataset_names, args.n_clusters, args.k_range, output_path, logger
    )

    # Export summary
    export_summary(datasets_data, dataset_names, output_path, logger)

    # Generate metadata-based visualizations
    visualize_by_audio_key(datasets_data, dataset_names, output_path, logger)
    visualize_by_num_measures(datasets_data, dataset_names, output_path, logger)
    visualize_by_track_role(datasets_data, dataset_names, output_path, logger)
    visualize_by_instrument(datasets_data, dataset_names, output_path, logger)

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("All visualizations saved to: %s", output_path)
    logger.info("")
    logger.info("Generated files:")

    for name in dataset_names:
        safe_name = name.replace("/", "_").replace(" ", "_")
        logger.info("  • tsne_%s_detailed.png", safe_name)
        logger.info("  • tsne_%s_by_audio_key.png", safe_name)
        logger.info("  • tsne_%s_by_num_measures.png", safe_name)
        logger.info("  • tsne_%s_by_track_role.png", safe_name)
        logger.info("  • tsne_%s_by_instrument.png", safe_name)

    logger.info("  • tsne_comparison_all.png")
    logger.info("  • tsne_clusters.png")
    logger.info("  • tsne_summary_statistics.csv")

    if args.n_clusters is None:
        logger.info("")
        logger.info("Cluster optimization files:")
        for name in dataset_names:
            safe_name = name.replace("/", "_").replace(" ", "_")
            logger.info("  • tsne_cluster_optimization_%s.png", safe_name)
        logger.info("  • tsne_cluster_selection_summary.csv")
    logger.info("")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
