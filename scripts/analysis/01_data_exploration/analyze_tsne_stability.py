#!/usr/bin/env python3
"""
t-SNE Stability and Robustness Analysis

This script performs comprehensive stability analysis for t-SNE embeddings, including:
    - Random seed stability testing using Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)
    - Perplexity sensitivity analysis across multiple values
    - Embedding quality assessment using trustworthiness and continuity metrics

Outputs:
    - Seed stability heatmaps (ARI/NMI matrices)
    - Perplexity sensitivity curves
    - Embedding quality metrics and visualizations
    - Combined summary report

Author: Analysis Team
Date: 2026-01-27
"""

# Standard library imports
import argparse
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

# Suppress warnings
warnings.filterwarnings("ignore")


# ============================================================================
# Configuration & Constants
# ============================================================================


@dataclass
class StabilityConfig:
    """Central configuration for t-SNE stability analysis."""

    # Input data paths
    workspace_root: Path = Path("/workspace")
    filtered_17d_features: Path = Path(
        "/workspace/artifacts/features/filtered/commu_full_mapped/features_numeric.csv"
    )

    # Stability parameters
    n_random_seeds: int = 10
    perplexities: List[int] = field(default_factory=lambda: [5, 30, 50, 100])
    n_iterations: int = 1000
    learning_rate: float = 200.0

    # Clustering for stability measurement
    n_clusters: int = 10  # Fixed k for fair comparison across runs

    # Embedding quality parameters
    k_neighbors_range: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 30])

    # Output configuration
    output_dir: Path = Path(
        "/workspace/outputs/analysis_results/tsne_stability_analysis"
    )
    default_format: str = "png"
    dpi: int = 300

    # Figure dimensions
    summary_figure_size: Tuple[float, float] = (16, 12)
    individual_figure_size: Tuple[float, float] = (10, 8)

    # Color scheme
    color_primary: str = "#3498DB"  # Blue
    color_secondary: str = "#E74C3C"  # Red
    colormap: str = "RdYlGn"


@dataclass
class SeedStabilityResults:
    """Container for seed stability analysis results."""

    ari_matrix: np.ndarray
    nmi_matrix: np.ndarray
    ari_mean: float
    ari_std: float
    nmi_mean: float
    nmi_std: float
    cluster_labels: List[np.ndarray]
    embeddings: List[np.ndarray]
    seeds: List[int]


@dataclass
class PerplexitySensitivityResults:
    """Container for perplexity sensitivity analysis results."""

    perplexities: List[int]
    silhouette_scores: List[float]
    silhouette_stds: List[float]
    davies_bouldin_scores: List[float]
    davies_bouldin_stds: List[float]
    calinski_harabasz_scores: List[float]
    calinski_harabasz_stds: List[float]
    kl_divergences: List[float]
    kl_divergences_stds: List[float]
    embeddings: Dict[int, List[np.ndarray]]  # perplexity -> list of embeddings


@dataclass
class EmbeddingQualityResults:
    """Container for embedding quality metrics."""

    trustworthiness_scores: Dict[int, float]  # k_neighbors -> score
    continuity_scores: Dict[int, float]  # k_neighbors -> score
    k_neighbors_tested: List[int]
    default_k: int = 15


# ============================================================================
# Data Loading & Validation
# ============================================================================


def load_and_validate_features(
    config: StabilityConfig,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load and validate 17D filtered features.

    Args:
        config: Configuration object with input paths

    Returns:
        Tuple of (feature_array, dataframe)

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data validation fails
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Check file existence
    if not config.filtered_17d_features.exists():
        raise FileNotFoundError(
            f"Filtered 17D features not found: {config.filtered_17d_features}"
        )

    # Load data
    print(f"\nLoading features from: {config.filtered_17d_features}")
    df = pd.read_csv(config.filtered_17d_features)

    print(f"  ✓ Loaded: {len(df)} samples × {len(df.columns)} features")

    # Validate dimensionality
    expected_dims = 17
    if len(df.columns) != expected_dims:
        raise ValueError(
            f"Expected {expected_dims} features, got {len(df.columns)}. "
            f"Columns: {df.columns.tolist()}"
        )

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(
            f"  ⚠ Warning: {nan_count} NaN values detected. Filling with column means..."
        )
        df = df.fillna(df.mean())

    # Convert to numpy array
    X = df.values

    print(f"  ✓ Feature array shape: {X.shape}")
    print(f"  ✓ Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  ✓ Data mean: {X.mean():.3f}, std: {X.std():.3f}")

    return X, df


# ============================================================================
# Seed Stability Analysis
# ============================================================================


def run_seed_stability_analysis(
    X: np.ndarray, config: StabilityConfig
) -> SeedStabilityResults:
    """
    Run t-SNE with multiple random seeds and measure cluster assignment stability.

    Args:
        X: Feature array (n_samples, 17)
        config: Configuration object

    Returns:
        SeedStabilityResults with ARI/NMI metrics
    """
    print("\n" + "=" * 80)
    print("SEED STABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nRunning t-SNE with {config.n_random_seeds} different random seeds...")
    print(f"  Perplexity: {config.perplexities[1]}")  # Use default perplexity=30
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Clustering with k={config.n_clusters}")

    # Generate random seeds
    base_seed = 42
    np.random.seed(base_seed)
    seeds = np.random.randint(0, 10000, size=config.n_random_seeds).tolist()

    embeddings = []
    cluster_labels = []

    # Run t-SNE for each seed
    for i, seed in enumerate(seeds, 1):
        print(f"\n  [{i}/{config.n_random_seeds}] Seed {seed}...", end=" ", flush=True)
        start_time = time.time()

        # Run t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=config.perplexities[1],  # Use perplexity=30
            max_iter=config.n_iterations,
            learning_rate=config.learning_rate,
            random_state=seed,
            verbose=0,
        )
        embedding = tsne.fit_transform(X)
        embeddings.append(embedding)

        # Cluster the embedding
        kmeans = KMeans(n_clusters=config.n_clusters, random_state=base_seed, n_init=10)
        labels = kmeans.fit_predict(embedding)
        cluster_labels.append(labels)

        elapsed = time.time() - start_time
        print(f"✓ ({elapsed:.1f}s, KL={tsne.kl_divergence_:.2f})")

    # Compute pairwise ARI and NMI
    print("\n\nComputing pairwise stability metrics...")
    n_seeds = len(seeds)
    ari_matrix = np.zeros((n_seeds, n_seeds))
    nmi_matrix = np.zeros((n_seeds, n_seeds))

    for i in range(n_seeds):
        for j in range(n_seeds):
            if i == j:
                ari_matrix[i, j] = 1.0
                nmi_matrix[i, j] = 1.0
            else:
                ari_matrix[i, j] = adjusted_rand_score(
                    cluster_labels[i], cluster_labels[j]
                )
                nmi_matrix[i, j] = normalized_mutual_info_score(
                    cluster_labels[i], cluster_labels[j]
                )

    # Compute summary statistics (upper triangular only, excluding diagonal)
    mask = np.triu(np.ones_like(ari_matrix, dtype=bool), k=1)
    ari_mean = ari_matrix[mask].mean()
    ari_std = ari_matrix[mask].std()
    nmi_mean = nmi_matrix[mask].mean()
    nmi_std = nmi_matrix[mask].std()

    print(f"\n  ✓ ARI: {ari_mean:.4f} ± {ari_std:.4f}")
    print(f"  ✓ NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")

    return SeedStabilityResults(
        ari_matrix=ari_matrix,
        nmi_matrix=nmi_matrix,
        ari_mean=ari_mean,
        ari_std=ari_std,
        nmi_mean=nmi_mean,
        nmi_std=nmi_std,
        cluster_labels=cluster_labels,
        embeddings=embeddings,
        seeds=seeds,
    )


# ============================================================================
# Perplexity Sensitivity Analysis
# ============================================================================


def run_perplexity_sensitivity_analysis(
    X: np.ndarray, config: StabilityConfig
) -> PerplexitySensitivityResults:
    """
    Test t-SNE across multiple perplexity values and measure impact on clustering.

    Args:
        X: Feature array (n_samples, 17)
        config: Configuration object

    Returns:
        PerplexitySensitivityResults with metrics per perplexity
    """
    print("\n" + "=" * 80)
    print("PERPLEXITY SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nTesting perplexities: {config.perplexities}")
    print(f"  Running {3} seeds per perplexity for robustness")

    # Storage for results
    results_per_perp = {
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
        "kl_divergence": [],
    }
    embeddings_per_perp = {}

    # Test each perplexity
    for perp in config.perplexities:
        print(f"\n  Perplexity = {perp}")
        print("  " + "-" * 40)

        perp_metrics = {
            "silhouette": [],
            "davies_bouldin": [],
            "calinski_harabasz": [],
            "kl_divergence": [],
        }
        perp_embeddings = []

        # Run with 3 different seeds for each perplexity
        test_seeds = [42, 123, 456]
        for seed in test_seeds:
            print(f"    Seed {seed}...", end=" ", flush=True)
            start_time = time.time()

            # Run t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                max_iter=config.n_iterations,
                learning_rate=config.learning_rate,
                random_state=seed,
                verbose=0,
            )
            embedding = tsne.fit_transform(X)
            perp_embeddings.append(embedding)

            # Cluster and evaluate
            kmeans = KMeans(n_clusters=config.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embedding)

            # Compute metrics
            sil = silhouette_score(embedding, labels)
            db = davies_bouldin_score(embedding, labels)
            ch = calinski_harabasz_score(embedding, labels)
            kl = tsne.kl_divergence_

            perp_metrics["silhouette"].append(sil)
            perp_metrics["davies_bouldin"].append(db)
            perp_metrics["calinski_harabasz"].append(ch)
            perp_metrics["kl_divergence"].append(kl)

            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.1f}s)")

        # Store aggregated results
        embeddings_per_perp[perp] = perp_embeddings
        for metric_name in perp_metrics:
            results_per_perp[metric_name].append(perp_metrics[metric_name])

        # Print summary for this perplexity
        print(
            f"    Silhouette:        {np.mean(perp_metrics['silhouette']):.4f} ± {np.std(perp_metrics['silhouette']):.4f}"
        )
        print(
            f"    Davies-Bouldin:    {np.mean(perp_metrics['davies_bouldin']):.4f} ± {np.std(perp_metrics['davies_bouldin']):.4f}"
        )
        print(
            f"    Calinski-Harabasz: {np.mean(perp_metrics['calinski_harabasz']):.1f} ± {np.std(perp_metrics['calinski_harabasz']):.1f}"
        )
        print(
            f"    KL Divergence:     {np.mean(perp_metrics['kl_divergence']):.4f} ± {np.std(perp_metrics['kl_divergence']):.4f}"
        )

    # Compute means and stds
    silhouette_means = [np.mean(x) for x in results_per_perp["silhouette"]]
    silhouette_stds = [np.std(x) for x in results_per_perp["silhouette"]]
    db_means = [np.mean(x) for x in results_per_perp["davies_bouldin"]]
    db_stds = [np.std(x) for x in results_per_perp["davies_bouldin"]]
    ch_means = [np.mean(x) for x in results_per_perp["calinski_harabasz"]]
    ch_stds = [np.std(x) for x in results_per_perp["calinski_harabasz"]]
    kl_means = [np.mean(x) for x in results_per_perp["kl_divergence"]]
    kl_stds = [np.std(x) for x in results_per_perp["kl_divergence"]]

    return PerplexitySensitivityResults(
        perplexities=config.perplexities,
        silhouette_scores=silhouette_means,
        silhouette_stds=silhouette_stds,
        davies_bouldin_scores=db_means,
        davies_bouldin_stds=db_stds,
        calinski_harabasz_scores=ch_means,
        calinski_harabasz_stds=ch_stds,
        kl_divergences=kl_means,
        kl_divergences_stds=kl_stds,
        embeddings=embeddings_per_perp,
    )


# ============================================================================
# Embedding Quality Metrics
# ============================================================================


def compute_continuity(X_high: np.ndarray, X_low: np.ndarray, k: int = 15) -> float:
    """
    Compute continuity metric for dimensionality reduction.

    Continuity measures whether points that were neighbors in high-D remain
    neighbors in low-D. Higher is better.

    Args:
        X_high: High-dimensional data (n_samples, n_features_high)
        X_low: Low-dimensional embedding (n_samples, n_features_low)
        k: Number of nearest neighbors to consider

    Returns:
        Continuity score (0-1, higher is better)
    """
    n_samples = X_high.shape[0]

    # Find k-nearest neighbors in high-dimensional space
    nbrs_high = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs_high.fit(X_high)
    _, indices_high = nbrs_high.kneighbors(X_high)
    indices_high = indices_high[:, 1:]  # Remove self

    # Find k-nearest neighbors in low-dimensional space
    nbrs_low = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs_low.fit(X_low)
    _, indices_low = nbrs_low.kneighbors(X_low)
    indices_low = indices_low[:, 1:]  # Remove self

    # For each point, compute rank of high-D neighbors in low-D space
    continuity_sum = 0.0
    for i in range(n_samples):
        # Neighbors in high-D
        high_neighbors = set(indices_high[i])

        # Get all distances in low-D for point i
        distances = np.sum((X_low - X_low[i]) ** 2, axis=1)
        ranks = np.argsort(distances)

        # For each high-D neighbor, find its rank in low-D
        for neighbor in high_neighbors:
            rank = np.where(ranks == neighbor)[0][0]
            if rank > k:
                # Penalize based on how far away the neighbor moved
                continuity_sum += rank - k

    # Normalize
    n = n_samples
    continuity = 1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * continuity_sum

    return continuity


def run_embedding_quality_analysis(
    X: np.ndarray, config: StabilityConfig
) -> EmbeddingQualityResults:
    """
    Compute trustworthiness and continuity for the 17D→2D embedding.

    Args:
        X: Feature array (n_samples, 17)
        config: Configuration object

    Returns:
        EmbeddingQualityResults with quality metrics
    """
    print("\n" + "=" * 80)
    print("EMBEDDING QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nComputing trustworthiness and continuity metrics...")
    print(f"  Using perplexity={config.perplexities[1]}, seed=42")
    print(f"  Testing k_neighbors: {config.k_neighbors_range}")

    # Generate single t-SNE embedding
    print("\n  Running t-SNE...", end=" ", flush=True)
    start_time = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=config.perplexities[1],
        max_iter=config.n_iterations,
        learning_rate=config.learning_rate,
        random_state=42,
        verbose=0,
    )
    embedding = tsne.fit_transform(X)
    elapsed = time.time() - start_time
    print(f"✓ ({elapsed:.1f}s, KL={tsne.kl_divergence_:.2f})")

    # Compute metrics for different k values
    print("\n  Computing quality metrics:")
    trustworthiness_scores = {}
    continuity_scores = {}

    for k in config.k_neighbors_range:
        print(f"    k={k:2d}...", end=" ", flush=True)

        # Trustworthiness (sklearn implementation)
        trust = trustworthiness(X, embedding, n_neighbors=k, metric="euclidean")
        trustworthiness_scores[k] = trust

        # Continuity (custom implementation)
        cont = compute_continuity(X, embedding, k=k)
        continuity_scores[k] = cont

        print(f"Trust={trust:.4f}, Cont={cont:.4f}")

    # Summary
    default_k = 15
    print(f"\n  Summary (k={default_k}):")
    print(f"    Trustworthiness: {trustworthiness_scores[default_k]:.4f}")
    print(f"    Continuity:      {continuity_scores[default_k]:.4f}")

    return EmbeddingQualityResults(
        trustworthiness_scores=trustworthiness_scores,
        continuity_scores=continuity_scores,
        k_neighbors_tested=config.k_neighbors_range,
        default_k=default_k,
    )


# ============================================================================
# Visualization Functions
# ============================================================================


def setup_matplotlib_style(config: StabilityConfig) -> None:
    """Configure matplotlib with publication-quality settings."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": config.dpi,
            "savefig.dpi": config.dpi,
            "savefig.bbox": "tight",
        }
    )


def visualize_seed_stability(
    results: SeedStabilityResults, config: StabilityConfig, output_path: Path
) -> None:
    """Generate heatmaps for seed stability matrices."""
    print("\n  → Generating seed stability heatmaps...")

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ARI heatmap
    sns.heatmap(
        results.ari_matrix,
        annot=False,
        cmap=config.colormap,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "ARI Score"},
        ax=axes[0],
    )
    axes[0].set_title(
        f"Adjusted Rand Index (ARI)\nMean: {results.ari_mean:.4f} ± {results.ari_std:.4f}",
        fontweight="bold",
    )
    axes[0].set_xlabel("Seed Index")
    axes[0].set_ylabel("Seed Index")

    # NMI heatmap
    sns.heatmap(
        results.nmi_matrix,
        annot=False,
        cmap=config.colormap,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "NMI Score"},
        ax=axes[1],
    )
    axes[1].set_title(
        f"Normalized Mutual Information (NMI)\nMean: {results.nmi_mean:.4f} ± {results.nmi_std:.4f}",
        fontweight="bold",
    )
    axes[1].set_xlabel("Seed Index")
    axes[1].set_ylabel("Seed Index")

    plt.tight_layout()
    save_path = output_path / "seed_stability_heatmaps.png"
    plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {save_path.name}")


def visualize_perplexity_sensitivity(
    results: PerplexitySensitivityResults, config: StabilityConfig, output_path: Path
) -> None:
    """Generate curves showing metric variation across perplexities."""
    print("\n  → Generating perplexity sensitivity curves...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    perplexities = results.perplexities

    # Plot 1: Silhouette Score
    axes[0, 0].errorbar(
        perplexities,
        results.silhouette_scores,
        yerr=results.silhouette_stds,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        color=config.color_primary,
    )
    axes[0, 0].set_xlabel("Perplexity")
    axes[0, 0].set_ylabel("Silhouette Score")
    axes[0, 0].set_title(
        "Silhouette Score vs Perplexity (higher is better)", fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")

    # Plot 2: Davies-Bouldin Index
    axes[0, 1].errorbar(
        perplexities,
        results.davies_bouldin_scores,
        yerr=results.davies_bouldin_stds,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        color=config.color_secondary,
    )
    axes[0, 1].set_xlabel("Perplexity")
    axes[0, 1].set_ylabel("Davies-Bouldin Index")
    axes[0, 1].set_title(
        "Davies-Bouldin Index vs Perplexity (lower is better)", fontweight="bold"
    )
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Plot 3: Calinski-Harabasz Score
    axes[1, 0].errorbar(
        perplexities,
        results.calinski_harabasz_scores,
        yerr=results.calinski_harabasz_stds,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#2ECC71",
    )
    axes[1, 0].set_xlabel("Perplexity")
    axes[1, 0].set_ylabel("Calinski-Harabasz Score")
    axes[1, 0].set_title(
        "Calinski-Harabasz Score vs Perplexity (higher is better)", fontweight="bold"
    )
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")

    # Plot 4: KL Divergence
    axes[1, 1].errorbar(
        perplexities,
        results.kl_divergences,
        yerr=results.kl_divergences_stds,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#9B59B6",
    )
    axes[1, 1].set_xlabel("Perplexity")
    axes[1, 1].set_ylabel("KL Divergence")
    axes[1, 1].set_title(
        "Final KL Divergence vs Perplexity (lower is better)", fontweight="bold"
    )
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale("log")

    plt.tight_layout()
    save_path = output_path / "perplexity_sensitivity_curves.png"
    plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {save_path.name}")


def visualize_embedding_quality(
    results: EmbeddingQualityResults, config: StabilityConfig, output_path: Path
) -> None:
    """Generate plots showing trustworthiness and continuity metrics."""
    print("\n  → Generating embedding quality plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = results.k_neighbors_tested
    trust_values = [results.trustworthiness_scores[k] for k in k_values]
    cont_values = [results.continuity_scores[k] for k in k_values]

    # Plot 1: Trustworthiness
    axes[0].plot(
        k_values,
        trust_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color=config.color_primary,
    )
    axes[0].axhline(
        y=0.95, color="green", linestyle="--", alpha=0.5, label="Excellent (>0.95)"
    )
    axes[0].axhline(
        y=0.90, color="orange", linestyle="--", alpha=0.5, label="Good (>0.90)"
    )
    axes[0].set_xlabel("Number of Neighbors (k)")
    axes[0].set_ylabel("Trustworthiness Score")
    axes[0].set_title("Trustworthiness vs k (17D → 2D)", fontweight="bold")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: Continuity
    axes[1].plot(
        k_values,
        cont_values,
        marker="o",
        linewidth=2,
        markersize=8,
        color=config.color_secondary,
    )
    axes[1].axhline(
        y=0.90, color="green", linestyle="--", alpha=0.5, label="Excellent (>0.90)"
    )
    axes[1].axhline(
        y=0.85, color="orange", linestyle="--", alpha=0.5, label="Good (>0.85)"
    )
    axes[1].set_xlabel("Number of Neighbors (k)")
    axes[1].set_ylabel("Continuity Score")
    axes[1].set_title("Continuity vs k (17D → 2D)", fontweight="bold")
    axes[1].set_ylim(0.80, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_path = output_path / "embedding_quality_metrics.png"
    plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {save_path.name}")


def visualize_best_perplexity_embedding(
    X: np.ndarray,
    perp_results: PerplexitySensitivityResults,
    config: StabilityConfig,
    output_path: Path,
) -> None:
    """
    Visualize the t-SNE embedding at the optimal perplexity value.

    Creates a 2D scatter plot of the t-SNE embedding colored by K-means clusters,
    using the perplexity value that achieved the highest silhouette score.

    Args:
        X: Original high-dimensional feature matrix (n_samples, n_features).
        perp_results: Results from perplexity sensitivity analysis containing
                     embeddings and evaluation metrics.
        config: Configuration object with visualization settings.
        output_path: Directory where the visualization will be saved.

    Note:
        Uses the first random seed's embedding (seed=42) for the optimal perplexity.
        Cluster assignments are computed via K-means on the 2D embedding.
    """
    print("\n  → Generating optimal perplexity embedding scatter plot...")

    # Determine best perplexity (highest silhouette score)
    best_perp_idx = np.argmax(perp_results.silhouette_scores)
    best_perp = perp_results.perplexities[best_perp_idx]
    best_silhouette = perp_results.silhouette_scores[best_perp_idx]

    # Get embedding for best perplexity (use first seed: 42)
    embedding = perp_results.embeddings[best_perp][0]  # shape: (n_samples, 2)

    # Cluster the embedding
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=config.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Create color map for clusters
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / config.n_clusters) for i in range(config.n_clusters)]

    # Plot each cluster
    for cluster_id in range(config.n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[cluster_id]],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=20,
            edgecolors="none",
        )

    # Styling
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.set_title(
        f"t-SNE Embedding at Optimal Perplexity\n"
        f"Perplexity = {best_perp} | Silhouette Score = {best_silhouette:.4f} | "
        f"Seed = 42",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
    )
    ax.grid(True, alpha=0.2)

    # Add cluster count annotation
    ax.text(
        0.02,
        0.98,
        f"K-means clusters: {config.n_clusters}\nSamples: {len(embedding):,}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    save_path = output_path / "best_perplexity_embedding.png"
    plt.savefig(save_path, dpi=config.dpi, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {save_path.name}")
    print(f"      └─ Using perplexity={best_perp}, silhouette={best_silhouette:.4f}")


# ============================================================================
# Report Generation & Export
# ============================================================================


def export_results(
    seed_results: SeedStabilityResults,
    perp_results: PerplexitySensitivityResults,
    quality_results: EmbeddingQualityResults,
    output_path: Path,
) -> None:
    """Export all results to CSV files."""
    print("\n  → Exporting results to CSV...")

    # Seed stability metrics
    seed_df = pd.DataFrame(
        {
            "metric": ["ARI", "NMI"],
            "mean": [seed_results.ari_mean, seed_results.nmi_mean],
            "std": [seed_results.ari_std, seed_results.nmi_std],
        }
    )
    seed_df.to_csv(output_path / "seed_stability_summary.csv", index=False)
    print(f"    ✓ seed_stability_summary.csv")

    # Perplexity sensitivity
    perp_df = pd.DataFrame(
        {
            "perplexity": perp_results.perplexities,
            "silhouette_mean": perp_results.silhouette_scores,
            "silhouette_std": perp_results.silhouette_stds,
            "davies_bouldin_mean": perp_results.davies_bouldin_scores,
            "davies_bouldin_std": perp_results.davies_bouldin_stds,
            "calinski_harabasz_mean": perp_results.calinski_harabasz_scores,
            "calinski_harabasz_std": perp_results.calinski_harabasz_stds,
            "kl_divergence_mean": perp_results.kl_divergences,
            "kl_divergence_std": perp_results.kl_divergences_stds,
        }
    )
    perp_df.to_csv(output_path / "perplexity_sensitivity_metrics.csv", index=False)
    print(f"    ✓ perplexity_sensitivity_metrics.csv")

    # Embedding quality
    quality_df = pd.DataFrame(
        {
            "k_neighbors": quality_results.k_neighbors_tested,
            "trustworthiness": [
                quality_results.trustworthiness_scores[k]
                for k in quality_results.k_neighbors_tested
            ],
            "continuity": [
                quality_results.continuity_scores[k]
                for k in quality_results.k_neighbors_tested
            ],
        }
    )
    quality_df.to_csv(output_path / "embedding_quality_metrics.csv", index=False)
    print(f"    ✓ embedding_quality_metrics.csv")


def print_summary_report(
    seed_results: SeedStabilityResults,
    perp_results: PerplexitySensitivityResults,
    quality_results: EmbeddingQualityResults,
) -> None:
    """Print comprehensive summary report to console."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Seed stability
    print("\n1. Seed Stability (Cluster Assignment Consistency)")
    print("   " + "-" * 76)
    print(
        f"   Adjusted Rand Index:           {seed_results.ari_mean:.4f} ± {seed_results.ari_std:.4f}"
    )
    print(
        f"   Normalized Mutual Information: {seed_results.nmi_mean:.4f} ± {seed_results.nmi_std:.4f}"
    )

    if seed_results.ari_mean > 0.8:
        print("   ✓ EXCELLENT: Clustering is highly stable across random seeds")
    elif seed_results.ari_mean > 0.6:
        print("   ⚠ MODERATE: Clustering shows some variation across seeds")
    else:
        print("   ✗ POOR: Clustering is unstable, results may not be reliable")

    # Perplexity sensitivity
    print("\n2. Perplexity Sensitivity")
    print("   " + "-" * 76)
    best_perp_idx = np.argmax(perp_results.silhouette_scores)
    best_perp = perp_results.perplexities[best_perp_idx]
    print(f"   Best perplexity (by Silhouette): {best_perp}")
    print(f"   Perplexity range tested:         {perp_results.perplexities}")

    # Show metrics for best perplexity
    print(f"\n   Metrics at perplexity={best_perp}:")
    print(
        f"     Silhouette:        {perp_results.silhouette_scores[best_perp_idx]:.4f} ± {perp_results.silhouette_stds[best_perp_idx]:.4f}"
    )
    print(
        f"     Davies-Bouldin:    {perp_results.davies_bouldin_scores[best_perp_idx]:.4f} ± {perp_results.davies_bouldin_stds[best_perp_idx]:.4f}"
    )
    print(
        f"     Calinski-Harabasz: {perp_results.calinski_harabasz_scores[best_perp_idx]:.1f} ± {perp_results.calinski_harabasz_stds[best_perp_idx]:.1f}"
    )

    # Embedding quality
    print("\n3. Embedding Quality (17D → 2D)")
    print("   " + "-" * 76)
    default_k = quality_results.default_k
    trust = quality_results.trustworthiness_scores[default_k]
    cont = quality_results.continuity_scores[default_k]

    print(f"   Trustworthiness (k={default_k}): {trust:.4f}")
    if trust > 0.95:
        print("     ✓ EXCELLENT: Local structure is very well preserved")
    elif trust > 0.90:
        print("     ✓ GOOD: Local structure is well preserved")
    else:
        print("     ⚠ MODERATE: Some local structure distortion present")

    print(f"   Continuity (k={default_k}):      {cont:.4f}")
    if cont > 0.90:
        print("     ✓ EXCELLENT: Neighbors remain close in low-D")
    elif cont > 0.85:
        print("     ✓ GOOD: Most neighbors remain close in low-D")
    else:
        print("     ⚠ MODERATE: Some neighborhood distortion present")

    # Overall recommendation
    print("\n4. Recommendations")
    print("   " + "-" * 76)
    if seed_results.ari_mean > 0.8 and trust > 0.95:
        print(
            "   ✓ The 17D filtered feature set produces STABLE and HIGH-QUALITY t-SNE embeddings"
        )
        print(f"   ✓ Recommended perplexity: {best_perp}")
        print("   ✓ Results are reliable for downstream analysis and visualization")
    elif seed_results.ari_mean > 0.6 and trust > 0.90:
        print("   ⚠ The embeddings are ACCEPTABLE but show some variation")
        print(f"   → Consider using perplexity={best_perp} for best results")
        print("   → Run multiple seeds and ensemble results for critical analyses")
    else:
        print("   ✗ Embedding quality needs improvement")
        print("   → Consider different dimensionality reduction techniques")
        print("   → Investigate feature scaling and preprocessing")


# ============================================================================
# Main Orchestration
# ============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="t-SNE Stability and Robustness Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (10 seeds, perplexities [5,30,50,100])
  python analyze_tsne_stability.py
  
  # Custom parameters
  python analyze_tsne_stability.py --n-seeds 20 --perplexities 5 15 30 50 100
  
  # Custom output directory
  python analyze_tsne_stability.py --output-dir /custom/path
        """,
    )

    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of random seeds to test (default: 10)",
    )

    parser.add_argument(
        "--perplexities",
        type=int,
        nargs="+",
        default=[5, 30, 50, 100],
        help="Perplexity values to test (default: 5 30 50 100)",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters for K-means (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/analysis_results/tsne_stability_analysis)",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Initialize configuration
    config = StabilityConfig(
        n_random_seeds=args.n_seeds,
        perplexities=args.perplexities,
        n_clusters=args.n_clusters,
    )

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup matplotlib
    setup_matplotlib_style(config)

    print("=" * 80)
    print("t-SNE STABILITY AND ROBUSTNESS ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input features:    {config.filtered_17d_features}")
    print(f"  Random seeds:      {config.n_random_seeds}")
    print(f"  Perplexities:      {config.perplexities}")
    print(f"  Clustering (k):    {config.n_clusters}")
    print(f"  Output directory:  {config.output_dir}")

    try:
        # Load data
        X, df = load_and_validate_features(config)

        # Run analyses
        seed_results = run_seed_stability_analysis(X, config)
        perp_results = run_perplexity_sensitivity_analysis(X, config)
        quality_results = run_embedding_quality_analysis(X, config)

        # Generate visualizations
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        visualize_seed_stability(seed_results, config, config.output_dir)
        visualize_perplexity_sensitivity(perp_results, config, config.output_dir)
        visualize_embedding_quality(quality_results, config, config.output_dir)
        visualize_best_perplexity_embedding(X, perp_results, config, config.output_dir)

        # Export results
        print("\n" + "=" * 80)
        print("EXPORTING RESULTS")
        print("=" * 80)

        export_results(seed_results, perp_results, quality_results, config.output_dir)

        # Print summary
        print_summary_report(seed_results, perp_results, quality_results)

        # Final message
        print("\n" + "=" * 80)
        print("✓ ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll outputs saved to: {config.output_dir}")
        print("\nGenerated files:")
        print("  • seed_stability_heatmaps.png")
        print("  • perplexity_sensitivity_curves.png")
        print("  • embedding_quality_metrics.png")
        print("  • best_perplexity_embedding.png")
        print("  • seed_stability_summary.csv")
        print("  • perplexity_sensitivity_metrics.csv")
        print("  • embedding_quality_metrics.csv")
        print("\n" + "=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
