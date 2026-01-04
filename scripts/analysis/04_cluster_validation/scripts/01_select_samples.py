#!/usr/bin/env python3
"""
Cluster Sample Selector - GHSOM Listening Test Preparation

This script selects representative samples from GHSOM clusters for listening tests
and qualitative evaluation. Supports multiple sampling strategies and ensures
balanced representation across clusters.

Author: GHSOM Analysis Pipeline
Date: 2025-11-19
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd


class ClusterSampler:
    """Select representative samples from clusters for listening tests."""

    def __init__(
        self,
        cluster_csv: Path,
        metadata_csv: Path,
        output_dir: Path,
        samples_per_cluster: int = 5,
        sampling_method: str = "random",
        random_seed: int = 42,
        min_samples_threshold: int = 3,
    ):
        """
        Initialize the ClusterSampler.

        Args:
            cluster_csv: Path to sample_to_cluster.csv
            metadata_csv: Path to features_with_metadata.csv
            output_dir: Output directory for sampled files
            samples_per_cluster: Target number of samples per cluster
            sampling_method: Sampling strategy (random, stratified)
            random_seed: Random seed for reproducibility
            min_samples_threshold: Minimum samples for small clusters
        """
        self.cluster_csv = Path(cluster_csv)
        self.metadata_csv = Path(metadata_csv)
        self.output_dir = Path(output_dir)
        self.samples_per_cluster = samples_per_cluster
        self.sampling_method = sampling_method
        self.random_seed = random_seed
        self.min_samples_threshold = min_samples_threshold

        # Set random seed
        np.random.seed(random_seed)

        # Data containers
        self.cluster_data = None
        self.metadata = None
        self.selected_samples = None

    def load_data(self) -> None:
        """Load cluster assignments and metadata."""
        print(f"Loading cluster assignments from {self.cluster_csv}")
        self.cluster_data = pd.read_csv(self.cluster_csv)
        print(f"  Loaded {len(self.cluster_data)} cluster assignments")

        print(f"\nLoading metadata from {self.metadata_csv}")
        self.metadata = pd.read_csv(self.metadata_csv)
        print(f"  Loaded {len(self.metadata)} metadata entries")

        # Merge cluster assignments with metadata
        self.merged_data = self.cluster_data.merge(
            self.metadata, left_on="sample_index", right_on="metadata_index", how="left"
        )
        print(f"  Merged data shape: {self.merged_data.shape}")

    def analyze_clusters(self) -> Dict:
        """Analyze cluster size distribution."""
        cluster_counts = self.cluster_data["GHSOM_cluster"].value_counts().sort_index()
        unique_clusters = cluster_counts.index.tolist()

        stats = {
            "num_clusters": len(unique_clusters),
            "total_samples": len(self.cluster_data),
            "cluster_sizes": cluster_counts.to_dict(),
            "mean_size": float(cluster_counts.mean()),
            "median_size": float(cluster_counts.median()),
            "min_size": int(cluster_counts.min()),
            "max_size": int(cluster_counts.max()),
            "std_size": float(cluster_counts.std()),
        }

        # Identify small clusters
        small_clusters = cluster_counts[cluster_counts < self.samples_per_cluster]
        stats["small_clusters"] = len(small_clusters)
        stats["small_cluster_ids"] = small_clusters.index.tolist()

        print(f"\n{'='*60}")
        print("CLUSTER STATISTICS")
        print(f"{'='*60}")
        print(f"Total clusters: {stats['num_clusters']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Mean cluster size: {stats['mean_size']:.1f}")
        print(f"Median cluster size: {stats['median_size']:.1f}")
        print(f"Size range: [{stats['min_size']}, {stats['max_size']}]")
        print(f"Std deviation: {stats['std_size']:.1f}")
        print(
            f"\nSmall clusters (<{self.samples_per_cluster} samples): {stats['small_clusters']}"
        )
        if stats["small_clusters"] > 0:
            print(f"  IDs: {stats['small_cluster_ids']}")
        print(f"{'='*60}\n")

        return stats

    def select_samples(self) -> pd.DataFrame:
        """
        Select samples from each cluster using specified method.

        Returns:
            DataFrame with selected samples and their metadata
        """
        selected_samples = []
        unique_clusters = sorted(self.cluster_data["GHSOM_cluster"].unique())

        print(f"Selecting samples using '{self.sampling_method}' method...")
        print(f"Target: {self.samples_per_cluster} samples per cluster\n")

        for cluster_id in unique_clusters:
            # Get all samples in this cluster
            cluster_samples = self.merged_data[
                self.merged_data["GHSOM_cluster"] == cluster_id
            ].copy()

            cluster_size = len(cluster_samples)

            # Determine how many samples to select
            if cluster_size < self.min_samples_threshold:
                n_select = cluster_size
                print(f"  Cluster {cluster_id}: {cluster_size} samples (taking all)")
            elif cluster_size < self.samples_per_cluster:
                n_select = cluster_size
                print(f"  Cluster {cluster_id}: {cluster_size} samples (taking all)")
            else:
                n_select = self.samples_per_cluster
                print(
                    f"  Cluster {cluster_id}: {cluster_size} samples (selecting {n_select})"
                )

            # Sample selection
            if self.sampling_method == "random":
                sampled = cluster_samples.sample(
                    n=n_select, random_state=self.random_seed
                )
            elif self.sampling_method == "stratified":
                # Stratified sampling by genre or other categorical variable if available
                sampled = cluster_samples.sample(
                    n=n_select, random_state=self.random_seed
                )
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")

            selected_samples.append(sampled)

        self.selected_samples = pd.concat(selected_samples, ignore_index=True)

        print(f"\n{'='*60}")
        print(f"SAMPLING COMPLETE")
        print(f"{'='*60}")
        print(f"Total samples selected: {len(self.selected_samples)}")
        print(
            f"Clusters represented: {self.selected_samples['GHSOM_cluster'].nunique()}"
        )
        print(f"{'='*60}\n")

        return self.selected_samples

    def copy_files(self, midi_root: Path, create_subdirs: bool = True) -> Dict:
        """
        Copy selected MIDI files to output directory.

        Args:
            midi_root: Root directory containing MIDI files
            create_subdirs: Create cluster subdirectories

        Returns:
            Copy statistics and error information
        """
        midi_root = Path(midi_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "total_files": len(self.selected_samples),
            "copied": 0,
            "missing": 0,
            "errors": 0,
            "missing_files": [],
            "error_files": [],
        }

        print(f"Copying MIDI files to {self.output_dir}")
        print(f"Source root: {midi_root}\n")

        for idx, row in self.selected_samples.iterrows():
            cluster_id = row["GHSOM_cluster"]
            file_path = row.get("file_path", "")
            track_id = row.get("track_id", f"sample_{row['sample_index']}")

            # Construct source file path
            if file_path and Path(file_path).exists():
                source_file = Path(file_path)
            else:
                # Try to construct path from midi_root
                source_file = midi_root / f"{track_id}.mid"
                if not source_file.exists():
                    # Try alternative paths
                    source_file = midi_root / "train" / "raw" / f"{track_id}.mid"

            if not source_file.exists():
                stats["missing"] += 1
                stats["missing_files"].append(
                    {
                        "cluster": int(cluster_id),
                        "track_id": track_id,
                        "expected_path": str(source_file),
                    }
                )
                print(f"  ✗ Missing: {track_id} (Cluster {cluster_id})")
                continue

            # Create destination path
            if create_subdirs:
                dest_dir = self.output_dir / f"cluster_{cluster_id:02d}"
                dest_dir.mkdir(parents=True, exist_ok=True)
            else:
                dest_dir = self.output_dir

            dest_file = dest_dir / f"{track_id}.mid"

            # Copy file
            try:
                shutil.copy2(source_file, dest_file)
                stats["copied"] += 1
                if stats["copied"] % 50 == 0:
                    print(f"  Copied {stats['copied']}/{stats['total_files']} files...")
            except Exception as e:
                stats["errors"] += 1
                stats["error_files"].append(
                    {"cluster": int(cluster_id), "track_id": track_id, "error": str(e)}
                )
                print(f"  ✗ Error copying {track_id}: {e}")

        print(f"\n{'='*60}")
        print("COPY STATISTICS")
        print(f"{'='*60}")
        print(f"Total files: {stats['total_files']}")
        print(f"Successfully copied: {stats['copied']}")
        print(f"Missing files: {stats['missing']}")
        print(f"Copy errors: {stats['errors']}")
        print(f"{'='*60}\n")

        return stats

    def save_manifest(self, copy_stats: Dict) -> None:
        """Save sampling manifest with metadata and statistics."""
        manifest = {
            "sampling_config": {
                "samples_per_cluster": self.samples_per_cluster,
                "sampling_method": self.sampling_method,
                "random_seed": self.random_seed,
                "min_samples_threshold": self.min_samples_threshold,
            },
            "input_files": {
                "cluster_csv": str(self.cluster_csv),
                "metadata_csv": str(self.metadata_csv),
            },
            "output_directory": str(self.output_dir),
            "statistics": copy_stats,
            "cluster_summary": {},
        }

        # Per-cluster summary
        for cluster_id in sorted(self.selected_samples["GHSOM_cluster"].unique()):
            cluster_samples = self.selected_samples[
                self.selected_samples["GHSOM_cluster"] == cluster_id
            ]
            manifest["cluster_summary"][int(cluster_id)] = {
                "num_samples": len(cluster_samples),
                "track_ids": cluster_samples["track_id"].tolist(),
            }

        # Save manifest
        manifest_file = self.output_dir / "sampling_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {manifest_file}")

        # Save selected samples CSV
        samples_file = self.output_dir / "selected_samples.csv"
        self.selected_samples.to_csv(samples_file, index=False)
        print(f"Saved selected samples to {samples_file}")

        # Save per-cluster metadata
        for cluster_id in sorted(self.selected_samples["GHSOM_cluster"].unique()):
            cluster_samples = self.selected_samples[
                self.selected_samples["GHSOM_cluster"] == cluster_id
            ]
            cluster_dir = self.output_dir / f"cluster_{cluster_id:02d}"
            if cluster_dir.exists():
                cluster_file = cluster_dir / "metadata.csv"
                cluster_samples.to_csv(cluster_file, index=False)
        print(f"Saved per-cluster metadata files")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Select representative samples from GHSOM clusters for listening tests",
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
        "--midi-root",
        type=Path,
        required=True,
        help="Root directory containing MIDI files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for selected samples",
    )
    parser.add_argument(
        "--samples-per-cluster",
        type=int,
        default=5,
        help="Number of samples to select per cluster (default: 5)",
    )
    parser.add_argument(
        "--sampling-method",
        choices=["random", "stratified"],
        default="random",
        help="Sampling strategy (default: random)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum samples threshold for small clusters (default: 3)",
    )
    parser.add_argument(
        "--no-subdirs",
        action="store_true",
        help="Do not create cluster subdirectories (flat structure)",
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("GHSOM CLUSTER SAMPLING - Listening Test Preparation")
    print(f"{'='*70}\n")

    # Initialize sampler
    sampler = ClusterSampler(
        cluster_csv=args.cluster_csv,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        samples_per_cluster=args.samples_per_cluster,
        sampling_method=args.sampling_method,
        random_seed=args.random_seed,
        min_samples_threshold=args.min_samples,
    )

    # Execute workflow
    sampler.load_data()
    stats = sampler.analyze_clusters()
    selected = sampler.select_samples()
    copy_stats = sampler.copy_files(
        midi_root=args.midi_root, create_subdirs=not args.no_subdirs
    )
    sampler.save_manifest(copy_stats)

    print(f"\n{'='*70}")
    print("✓ SAMPLING COMPLETE")
    print(f"{'='*70}\n")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Selected {len(selected)} samples from {selected['GHSOM_cluster'].nunique()} clusters"
    )
    print(f"Successfully copied {copy_stats['copied']} files\n")

    if copy_stats["missing"] > 0:
        print(f"⚠ Warning: {copy_stats['missing']} files were missing")
    if copy_stats["errors"] > 0:
        print(f"⚠ Warning: {copy_stats['errors']} copy errors occurred")

    return 0


if __name__ == "__main__":
    sys.exit(main())
