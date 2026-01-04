#!/usr/bin/env python3
"""
Visualize t-SNE embeddings colored by GHSOM cluster assignments.

Analyzes cluster separation, quality, and semantic coherence by combining:
- t-SNE 2D embeddings
- GHSOM hierarchical cluster assignments
- Cluster profiles (communicative function, genre, arousal, etc.)
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist
from collections import Counter


def load_data(
    tsne_path: str,
    cluster_path: str,
    profiles_path: str,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load t-SNE embeddings, cluster assignments, and cluster profiles."""
    # Load t-SNE embeddings
    embeddings = np.load(tsne_path)
    print(f"Loaded t-SNE embeddings: {embeddings.shape}")

    # Load cluster assignments
    clusters = pd.read_csv(cluster_path)
    print(f"Loaded cluster assignments: {len(clusters)} samples, {clusters['GHSOM_cluster'].nunique()} clusters")

    # Load cluster profiles
    profiles = pd.read_csv(profiles_path)
    print(f"Loaded cluster profiles: {len(profiles)} clusters")

    return embeddings, clusters, profiles


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute cluster quality metrics."""
    metrics = {}

    # Filter out noise labels if any (-1)
    valid_mask = labels >= 0
    if not all(valid_mask):
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]

    # Silhouette Score (-1 to 1, higher is better)
    if len(np.unique(labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(embeddings, labels)

    # Davies-Bouldin Index (lower is better)
    if len(np.unique(labels)) > 1:
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)

    # Calinski-Harabasz Index (higher is better)
    if len(np.unique(labels)) > 1:
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)

    # Compute per-cluster metrics
    unique_labels = np.unique(labels)
    cluster_sizes = []
    cluster_compactness = []

    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        cluster_sizes.append(len(cluster_points))

        # Compactness: average distance to centroid
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        cluster_compactness.append(distances.mean())

    metrics['avg_cluster_size'] = np.mean(cluster_sizes)
    metrics['std_cluster_size'] = np.std(cluster_sizes)
    metrics['avg_compactness'] = np.mean(cluster_compactness)

    return metrics


def compute_cluster_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute centroid for each cluster."""
    centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        centroids[label] = embeddings[mask].mean(axis=0)
    return centroids


def create_colormap(n_colors: int) -> ListedColormap:
    """Create a distinctive colormap for clusters."""
    if n_colors <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_colors]
    elif n_colors <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_colors]
    else:
        # Use a combination of colormaps for many clusters
        colors = plt.cm.nipy_spectral(np.linspace(0.1, 0.9, n_colors))
    return ListedColormap(colors)


def plot_tsne_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    output_path: str,
    title: str = "t-SNE Visualization with GHSOM Clusters",
    figsize: Tuple[int, int] = (16, 12),
    alpha: float = 0.6,
    point_size: int = 15,
    show_centroids: bool = True,
    show_legend: bool = True,
):
    """Create main t-SNE scatter plot colored by cluster."""
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)
    cmap = create_colormap(n_clusters)

    # Create label to index mapping for colormap
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = [label_to_idx[l] for l in labels]

    # Scatter plot
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=color_indices,
        cmap=cmap,
        alpha=alpha,
        s=point_size,
        edgecolors='none',
    )

    # Add centroids
    if show_centroids:
        centroids = compute_cluster_centroids(embeddings, labels)
        for label, centroid in centroids.items():
            idx = label_to_idx[label]
            ax.scatter(
                centroid[0], centroid[1],
                c=[cmap(idx / n_clusters)],
                s=200,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                zorder=10,
            )
            ax.annotate(
                str(label),
                (centroid[0], centroid[1]),
                fontsize=8,
                fontweight='bold',
                ha='center',
                va='bottom',
                xytext=(0, 5),
                textcoords='offset points',
            )

    # Legend with cluster info
    if show_legend and len(unique_labels) <= 30:
        # Create legend with communicative function info
        legend_elements = []
        for label in unique_labels:
            idx = label_to_idx[label]
            color = cmap(idx / n_clusters)

            # Get profile info
            profile_row = profiles[profiles['cluster_id'] == label]
            if len(profile_row) > 0:
                comm_func = profile_row['communicative_function'].values[0]
                sample_count = profile_row['sample_count'].values[0]
                label_text = f"C{label}: {comm_func} (n={sample_count})"
            else:
                count = (labels == label).sum()
                label_text = f"Cluster {label} (n={count})"

            legend_elements.append(
                mpatches.Patch(color=color, label=label_text)
            )

        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            title="Clusters",
        )

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_communicative_function(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = (16, 12),
):
    """Plot t-SNE colored by communicative function category."""
    # Map cluster labels to communicative functions
    cluster_to_func = dict(zip(profiles['cluster_id'], profiles['communicative_function']))
    functions = [cluster_to_func.get(l, 'Unknown') for l in labels]

    unique_funcs = sorted(set(functions))
    func_to_idx = {f: i for i, f in enumerate(unique_funcs)}
    func_indices = [func_to_idx[f] for f in functions]

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.Set2
    colors = [cmap(i / len(unique_funcs)) for i in range(len(unique_funcs))]

    for i, func in enumerate(unique_funcs):
        mask = np.array(functions) == func
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[colors[i]],
            label=f"{func} (n={mask.sum()})",
            alpha=0.6,
            s=15,
        )

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE by Communicative Function", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_by_arousal(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    output_path: str,
    figsize: Tuple[int, int] = (14, 10),
):
    """Plot t-SNE with arousal score as color gradient."""
    # Map cluster labels to arousal scores
    cluster_to_arousal = dict(zip(profiles['cluster_id'], profiles['arousal_score']))
    arousal_scores = [cluster_to_arousal.get(l, 0) for l in labels]

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=arousal_scores,
        cmap='RdYlBu_r',  # Red=high arousal, Blue=low
        alpha=0.6,
        s=15,
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Arousal Score', fontsize=11)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE Colored by Arousal Score", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cluster_grid(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    output_path: str,
    n_cols: int = 5,
):
    """Create grid of individual cluster plots."""
    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Global bounds for consistent axes
    x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
    y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()
    margin = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min

    for idx, (ax, label) in enumerate(zip(axes, unique_labels)):
        mask = labels == label

        # Plot all points in gray
        ax.scatter(
            embeddings[~mask, 0],
            embeddings[~mask, 1],
            c='lightgray',
            alpha=0.2,
            s=5,
        )

        # Highlight cluster points
        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c='crimson',
            alpha=0.7,
            s=15,
        )

        # Get profile info
        profile_row = profiles[profiles['cluster_id'] == label]
        if len(profile_row) > 0:
            comm_func = profile_row['communicative_function'].values[0]
            sample_count = profile_row['sample_count'].values[0]
            title = f"C{label}: {comm_func}\n(n={sample_count})"
        else:
            title = f"Cluster {label}\n(n={mask.sum()})"

        ax.set_title(title, fontsize=9)
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[n_clusters:]:
        ax.axis('off')

    plt.suptitle("Individual Cluster Distributions in t-SNE Space", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    output_path: str,
):
    """Create cluster quality analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    unique_labels = sorted(np.unique(labels))

    # 1. Cluster size distribution
    ax = axes[0, 0]
    sizes = [profiles[profiles['cluster_id'] == l]['sample_count'].values[0]
             if len(profiles[profiles['cluster_id'] == l]) > 0
             else (labels == l).sum()
             for l in unique_labels]

    bars = ax.bar(range(len(unique_labels)), sizes, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f"C{l}" for l in unique_labels], rotation=45, ha='right')
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Sample Count")
    ax.set_title("Cluster Size Distribution")
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Silhouette scores per cluster
    ax = axes[0, 1]
    from sklearn.metrics import silhouette_samples

    silhouette_vals = silhouette_samples(embeddings, labels)
    cluster_silhouettes = []
    for label in unique_labels:
        mask = labels == label
        cluster_silhouettes.append(silhouette_vals[mask].mean())

    colors = ['crimson' if s < 0 else 'steelblue' for s in cluster_silhouettes]
    ax.bar(range(len(unique_labels)), cluster_silhouettes, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=np.mean(cluster_silhouettes), color='red', linestyle='--',
               label=f'Mean: {np.mean(cluster_silhouettes):.3f}')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f"C{l}" for l in unique_labels], rotation=45, ha='right')
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Per-Cluster Silhouette Scores")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Cluster compactness (avg distance to centroid)
    ax = axes[1, 0]
    compactness = []
    for label in unique_labels:
        mask = labels == label
        cluster_points = embeddings[mask]
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        compactness.append(distances.mean())

    ax.bar(range(len(unique_labels)), compactness, color='teal', alpha=0.7)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f"C{l}" for l in unique_labels], rotation=45, ha='right')
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Avg Distance to Centroid")
    ax.set_title("Cluster Compactness (lower = more compact)")
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Arousal score distribution
    ax = axes[1, 1]
    arousal_scores = []
    for label in unique_labels:
        profile_row = profiles[profiles['cluster_id'] == label]
        if len(profile_row) > 0:
            arousal_scores.append(profile_row['arousal_score'].values[0])
        else:
            arousal_scores.append(0)

    colors = plt.cm.RdYlBu_r(np.array(arousal_scores) / max(arousal_scores) if max(arousal_scores) > 0 else arousal_scores)
    ax.bar(range(len(unique_labels)), arousal_scores, color=colors, alpha=0.8)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f"C{l}" for l in unique_labels], rotation=45, ha='right')
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Arousal Score")
    ax.set_title("Arousal Score by Cluster")
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Cluster Quality Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_density_heatmap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    figsize: Tuple[int, int] = (14, 10),
):
    """Create density heatmap of t-SNE embeddings."""
    fig, ax = plt.subplots(figsize=figsize)

    # Hexbin plot for density
    hb = ax.hexbin(
        embeddings[:, 0],
        embeddings[:, 1],
        gridsize=50,
        cmap='YlOrRd',
        mincnt=1,
    )

    cbar = plt.colorbar(hb, ax=ax, shrink=0.8)
    cbar.set_label('Sample Count', fontsize=11)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title("t-SNE Sample Density", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_metrics_report(
    embeddings: np.ndarray,
    labels: np.ndarray,
    profiles: pd.DataFrame,
    output_path: str,
):
    """Generate a text report with cluster quality metrics."""
    metrics = compute_cluster_metrics(embeddings, labels)

    report_lines = [
        "=" * 60,
        "GHSOM CLUSTER QUALITY METRICS REPORT",
        "=" * 60,
        "",
        "GLOBAL METRICS:",
        "-" * 40,
        f"  Silhouette Score:       {metrics.get('silhouette_score', 'N/A'):.4f}",
        f"    (Range: -1 to 1, higher is better)",
        f"  Davies-Bouldin Score:   {metrics.get('davies_bouldin_score', 'N/A'):.4f}",
        f"    (Lower is better)",
        f"  Calinski-Harabasz:      {metrics.get('calinski_harabasz_score', 'N/A'):.2f}",
        f"    (Higher is better)",
        "",
        "CLUSTER SIZE STATISTICS:",
        "-" * 40,
        f"  Number of clusters:     {len(np.unique(labels))}",
        f"  Total samples:          {len(labels)}",
        f"  Average cluster size:   {metrics['avg_cluster_size']:.1f}",
        f"  Std cluster size:       {metrics['std_cluster_size']:.1f}",
        f"  Average compactness:    {metrics['avg_compactness']:.4f}",
        "",
        "PER-CLUSTER BREAKDOWN:",
        "-" * 40,
    ]

    # Per-cluster details
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(embeddings, labels)

    unique_labels = sorted(np.unique(labels))
    for label in unique_labels:
        mask = labels == label
        cluster_size = mask.sum()
        cluster_silhouette = silhouette_vals[mask].mean()

        # Get profile info
        profile_row = profiles[profiles['cluster_id'] == label]
        if len(profile_row) > 0:
            comm_func = profile_row['communicative_function'].values[0]
            arousal = profile_row['arousal_score'].values[0]
        else:
            comm_func = "Unknown"
            arousal = 0

        report_lines.append(
            f"  Cluster {label:2d}: n={cluster_size:4d}, "
            f"silhouette={cluster_silhouette:+.3f}, "
            f"arousal={arousal:5.1f}, "
            f"{comm_func}"
        )

    report_lines.extend([
        "",
        "=" * 60,
        "Interpretation Guide:",
        "  - Silhouette > 0.5: Strong cluster structure",
        "  - Silhouette 0.25-0.5: Reasonable structure",
        "  - Silhouette < 0.25: Weak or overlapping clusters",
        "  - Negative silhouette: Sample may be misassigned",
        "=" * 60,
    ])

    report = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved: {output_path}")
    print("\n" + report)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Visualize t-SNE embeddings with GHSOM cluster assignments"
    )
    parser.add_argument(
        "--tsne-path",
        type=str,
        default="/workspace/artifacts/features/tsne/commu_full_filtered_tsne/embedding.npy",
        help="Path to t-SNE embeddings (.npy file)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="/workspace/experiments/ghsom_commu_full_tsne_optimized_20251125",
        help="Path to GHSOM experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/outputs/tsne_ghsom_visualization",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    embeddings, clusters, profiles = load_data(
        args.tsne_path,
        experiment_dir / "sample_to_cluster.csv",
        experiment_dir / "cluster_profiles.csv",
    )

    labels = clusters['GHSOM_cluster'].values

    # Verify alignment
    assert len(embeddings) == len(labels), f"Mismatch: {len(embeddings)} embeddings vs {len(labels)} labels"

    print(f"\nGenerating visualizations in: {output_dir}\n")

    # Generate all visualizations
    plot_tsne_clusters(
        embeddings, labels, profiles,
        output_dir / "tsne_clusters.png",
        title="t-SNE Visualization with GHSOM Clusters",
    )

    plot_by_communicative_function(
        embeddings, labels, profiles,
        output_dir / "tsne_communicative_function.png",
    )

    plot_by_arousal(
        embeddings, labels, profiles,
        output_dir / "tsne_arousal.png",
    )

    plot_cluster_grid(
        embeddings, labels, profiles,
        output_dir / "cluster_grid.png",
    )

    plot_cluster_quality(
        embeddings, labels, profiles,
        output_dir / "cluster_quality.png",
    )

    plot_density_heatmap(
        embeddings, labels,
        output_dir / "tsne_density.png",
    )

    # Generate metrics report
    generate_metrics_report(
        embeddings, labels, profiles,
        output_dir / "cluster_metrics_report.txt",
    )

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
