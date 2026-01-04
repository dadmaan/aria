"""
Diagnostic script to inspect cluster features and validate mapping.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ghsom_manager import GHSOMManager
from src.agents.cluster_feature_mapper import ClusterFeatureMapper


def diagnose_cluster_features(
    ghsom_checkpoint: Path,
    feature_artifact: Path,
    feature_type: str = "tsne",
    mode: str = "prototype",
    output_dir: Path = None,
):
    """
    Diagnose cluster features and visualize them.

    Args:
        ghsom_checkpoint: Path to GHSOM model checkpoint
        feature_artifact: Path to feature artifact directory
        feature_type: 'tsne' or 'raw'
        mode: 'prototype' or 'centroid'
        output_dir: Directory to save visualizations (default: current directory)
    """
    if output_dir is None:
        output_dir = Path.cwd()

    print("=" * 80)
    print("CLUSTER FEATURE DIAGNOSTIC")
    print("=" * 80)

    # Load GHSOM
    print(f"\n1. Loading GHSOM from {ghsom_checkpoint}...")
    try:
        ghsom_manager = GHSOMManager.from_artifact(
            ghsom_model_path=ghsom_checkpoint / "ghsom_model.pkl",
            feature_artifact=feature_artifact,
            feature_type=feature_type,
        )
        print(f"   ✓ Loaded GHSOM with {len(ghsom_manager.neuron_table)} leaf clusters")
    except Exception as e:
        print(f"   ✗ Failed to load GHSOM: {e}")
        return

    # Create mapper
    print(f"\n2. Creating ClusterFeatureMapper (mode={mode}, source={feature_type})...")
    try:
        mapper = ClusterFeatureMapper(
            ghsom_manager=ghsom_manager,
            mode=mode,
            feature_source=feature_type,
        )
        print(f"   ✓ Mapper created: {mapper.n_clusters} clusters, {mapper.feature_dim}D features")
    except Exception as e:
        print(f"   ✗ Failed to create mapper: {e}")
        return

    # Inspect features
    print(f"\n3. Cluster feature statistics:")
    all_features = np.array([mapper.get_features(i) for i in range(mapper.n_clusters)])
    print(f"   Shape: {all_features.shape}")
    print(f"   Min: {all_features.min():.4f}")
    print(f"   Max: {all_features.max():.4f}")
    print(f"   Mean: {all_features.mean():.4f}")
    print(f"   Std: {all_features.std():.4f}")

    # Show sample features
    print(f"\n4. Sample cluster features:")
    for i in range(min(5, mapper.n_clusters)):
        feat = mapper.get_features(i)
        feat_str = np.array2string(feat, precision=4, suppress_small=True)
        print(f"   Cluster {i:2d}: {feat_str}")

    # Test sequence mapping
    print(f"\n5. Test sequence mapping:")
    test_seq = np.array([0, 1, 2, 1, 0])
    mapped = mapper.map_sequence(test_seq)
    print(f"   Input: {test_seq}")
    print(f"   Output shape: {mapped.shape}")
    print(f"   First 3 vectors:")
    for i in range(min(3, len(mapped))):
        vec_str = np.array2string(mapped[i], precision=4, suppress_small=True)
        print(f"      {vec_str}")

    # Visualize if 2D
    if mapper.feature_dim == 2:
        print(f"\n6. Visualizing 2D cluster features...")
        plt.figure(figsize=(12, 10))

        # Plot clusters
        plt.scatter(all_features[:, 0], all_features[:, 1], s=150, alpha=0.6, c='blue', edgecolors='black')

        # Annotate cluster IDs
        for i in range(mapper.n_clusters):
            plt.annotate(
                str(i),
                (all_features[i, 0], all_features[i, 1]),
                fontsize=9,
                ha='center',
                va='center',
                color='white',
                weight='bold'
            )

        plt.xlabel('Feature Dimension 1', fontsize=12)
        plt.ylabel('Feature Dimension 2', fontsize=12)
        plt.title(f'Cluster Features in 2D Space (mode={mode}, source={feature_type})', fontsize=14)
        plt.grid(True, alpha=0.3)

        output_path = output_dir / f"cluster_features_{mode}_{feature_type}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved to {output_path}")
        plt.close()

    # Create distance matrix visualization
    print(f"\n7. Computing pairwise distance matrix...")
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(all_features, metric='euclidean'))

    plt.figure(figsize=(10, 8))
    im = plt.imshow(distances, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Euclidean Distance')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Cluster ID', fontsize=12)
    plt.title(f'Pairwise Cluster Distance Matrix (mode={mode})', fontsize=14)

    output_path = output_dir / f"cluster_distances_{mode}_{feature_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Distance matrix saved to {output_path}")
    plt.close()

    # Compute and display cluster statistics
    print(f"\n8. Cluster relationship analysis:")
    min_dist_idx = np.unravel_index(np.argmin(distances + np.eye(len(distances)) * 1e10), distances.shape)
    max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)

    print(f"   Most similar clusters: {min_dist_idx[0]} and {min_dist_idx[1]} (distance: {distances[min_dist_idx]:.4f})")
    print(f"   Most dissimilar clusters: {max_dist_idx[0]} and {max_dist_idx[1]} (distance: {distances[max_dist_idx]:.4f})")

    # Compute average distance to nearest neighbor
    np.fill_diagonal(distances, np.inf)
    nearest_neighbor_dists = np.min(distances, axis=1)
    print(f"   Average nearest neighbor distance: {nearest_neighbor_dists.mean():.4f} ± {nearest_neighbor_dists.std():.4f}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose cluster features")
    parser.add_argument(
        "--ghsom-checkpoint",
        type=Path,
        default=Path("experiments/ghsom_commu_full_tsne_optimized_20251125"),
        help="Path to GHSOM model checkpoint directory"
    )
    parser.add_argument(
        "--feature-artifact",
        type=Path,
        default=Path("artifacts/features/tsne/commu_full_filtered_tsne"),
        help="Path to feature artifact directory"
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="tsne",
        choices=["tsne", "raw"],
        help="Feature type"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prototype",
        choices=["prototype", "centroid"],
        help="Mapping mode"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for visualizations (default: current directory)"
    )

    args = parser.parse_args()

    diagnose_cluster_features(
        ghsom_checkpoint=args.ghsom_checkpoint,
        feature_artifact=args.feature_artifact,
        feature_type=args.feature_type,
        mode=args.mode,
        output_dir=args.output_dir,
    )
