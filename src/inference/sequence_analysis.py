#!/usr/bin/env python3
"""Sequence Analysis and Visualization Tools.

This module provides comprehensive analysis tools for generated sequences
including cluster distribution, transition patterns, diversity metrics,
and visualization functions for paper-ready figures.

Features:
    - Cluster distribution analysis
    - Transition matrix computation
    - Diversity metrics (entropy, uniqueness, repetition)
    - Comparison with training data distribution
    - Publication-quality visualizations
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.ghsom_manager import GHSOMManager
from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class DiversityMetrics:
    """Diversity metrics for a collection of sequences."""

    unique_clusters_total: int  # Total unique clusters used
    unique_per_sequence_mean: float  # Average unique clusters per sequence
    unique_per_sequence_std: float  # Std dev of unique per sequence
    repetition_ratio_mean: float  # Average ratio of consecutive repeats
    repetition_ratio_std: float  # Std dev of repetition ratio
    entropy: float  # Shannon entropy of cluster distribution
    coverage_ratio: float  # Fraction of total clusters used

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TransitionMetrics:
    """Transition pattern metrics."""

    num_unique_transitions: int  # Number of unique A→B transitions
    avg_transition_entropy: float  # Average entropy of outgoing transitions
    self_transition_ratio: float  # Ratio of A→A transitions
    max_transition_prob: float  # Maximum single transition probability

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


class SequenceAnalyzer:
    """Analyzer for generated musical sequences.

    Provides comprehensive analysis of cluster sequences including
    distribution analysis, transition patterns, and diversity metrics.

    Example:
        >>> analyzer = SequenceAnalyzer(ghsom_manager)
        >>> analyzer.load_sequences("outputs/inference/sequences.json")
        >>> metrics = analyzer.compute_all_metrics()
        >>> analyzer.generate_report("outputs/analysis/")
    """

    def __init__(
        self,
        ghsom_manager: Optional[GHSOMManager] = None,
        total_clusters: Optional[int] = None,
    ):
        """Initialize sequence analyzer.

        Args:
            ghsom_manager: GHSOM manager for cluster info.
            total_clusters: Total number of possible clusters (if no ghsom_manager).
        """
        self.ghsom_manager = ghsom_manager
        self.sequences: List[List[int]] = []
        self.metadata: Dict[str, Any] = {}

        # Determine total clusters
        if ghsom_manager is not None:
            self.total_clusters = len(ghsom_manager.cluster_ids)
        elif total_clusters is not None:
            self.total_clusters = total_clusters
        else:
            self.total_clusters = None  # Will infer from sequences

    def load_sequences(
        self,
        source: Union[str, Path, List[Dict[str, Any]], List[List[int]]],
    ) -> int:
        """Load sequences from file or data structure.

        Args:
            source: Path to JSON file, list of result dicts, or list of sequences.

        Returns:
            Number of sequences loaded.
        """
        if isinstance(source, (str, Path)):
            # Load from file
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    # List of result dicts
                    self.sequences = [d.get("sequence", []) for d in data]
                    self.metadata["source"] = str(path)
                    self.metadata["reward_data"] = [
                        d.get("episode_reward") for d in data
                    ]
                else:
                    # List of sequences directly
                    self.sequences = data
                    self.metadata["source"] = str(path)
            elif isinstance(data, dict) and "results" in data:
                # Results with wrapper
                self.sequences = [r.get("sequence", []) for r in data["results"]]
                self.metadata.update(data.get("aggregate_metrics", {}))

        elif isinstance(source, list):
            if len(source) > 0 and isinstance(source[0], dict):
                self.sequences = [d.get("sequence", d) for d in source]
            else:
                self.sequences = source

        # Infer total clusters if not set
        if self.total_clusters is None:
            all_clusters = set(c for seq in self.sequences for c in seq)
            self.total_clusters = max(all_clusters) + 1 if all_clusters else 22

        logger.info(f"Loaded {len(self.sequences)} sequences")
        return len(self.sequences)

    def analyze_cluster_distribution(self) -> pd.DataFrame:
        """Analyze cluster usage patterns in sequences.

        Returns:
            DataFrame with cluster distribution analysis.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        # Count all cluster occurrences
        all_clusters = []
        for seq in self.sequences:
            all_clusters.extend(seq)

        cluster_counts = Counter(all_clusters)
        total = sum(cluster_counts.values())

        # Build analysis data
        analysis = []
        for cluster_id in sorted(cluster_counts.keys()):
            count = cluster_counts[cluster_id]

            row = {
                "cluster_id": cluster_id,
                "count": count,
                "frequency": count / total,
                "percentage": 100 * count / total,
            }

            # Add hierarchy info if ghsom_manager available
            if self.ghsom_manager is not None:
                try:
                    path = self.ghsom_manager.get_node_relative_path_by_id(
                        cluster_id, print_output=False
                    )
                    row["hierarchy_level"] = len(path) if path else 0
                    row["path"] = " → ".join(path) if path else "root"
                except Exception:
                    row["hierarchy_level"] = 0
                    row["path"] = "unknown"

            analysis.append(row)

        df = pd.DataFrame(analysis)
        df = df.sort_values("count", ascending=False).reset_index(drop=True)

        return df

    def compute_transition_matrix(
        self,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, List[int]]:
        """Compute transition probability matrix.

        Args:
            normalize: Whether to normalize to probabilities.

        Returns:
            Tuple of (transition matrix, cluster list).
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        # Build transition counts
        transition_counts = defaultdict(lambda: defaultdict(int))

        for seq in self.sequences:
            for i in range(len(seq) - 1):
                from_cluster = seq[i]
                to_cluster = seq[i + 1]
                transition_counts[from_cluster][to_cluster] += 1

        # Get all unique clusters
        all_clusters = sorted(set(c for seq in self.sequences for c in seq))
        n = len(all_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(all_clusters)}

        # Build matrix
        matrix = np.zeros((n, n))
        for from_c, to_dict in transition_counts.items():
            for to_c, count in to_dict.items():
                i = cluster_to_idx[from_c]
                j = cluster_to_idx[to_c]
                matrix[i, j] = count

        # Normalize if requested
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / row_sums

        return matrix, all_clusters

    def calculate_diversity_metrics(self) -> DiversityMetrics:
        """Calculate diversity metrics for sequences.

        Returns:
            DiversityMetrics dataclass with computed values.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        # Unique clusters
        all_unique = set(c for seq in self.sequences for c in seq)
        unique_clusters_total = len(all_unique)

        # Per-sequence unique counts
        per_seq_unique = [len(set(seq)) for seq in self.sequences]

        # Repetition ratios (consecutive same cluster)
        repetition_ratios = []
        for seq in self.sequences:
            if len(seq) > 1:
                reps = sum(1 for i in range(len(seq) - 1) if seq[i] == seq[i + 1])
                repetition_ratios.append(reps / (len(seq) - 1))
            else:
                repetition_ratios.append(0.0)

        # Entropy
        all_clusters = [c for seq in self.sequences for c in seq]
        counts = np.bincount(all_clusters)
        probs = counts[counts > 0] / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Coverage ratio
        coverage = (
            unique_clusters_total / self.total_clusters if self.total_clusters else 0
        )

        return DiversityMetrics(
            unique_clusters_total=unique_clusters_total,
            unique_per_sequence_mean=float(np.mean(per_seq_unique)),
            unique_per_sequence_std=float(np.std(per_seq_unique)),
            repetition_ratio_mean=float(np.mean(repetition_ratios)),
            repetition_ratio_std=float(np.std(repetition_ratios)),
            entropy=float(entropy),
            coverage_ratio=float(coverage),
        )

    def calculate_transition_metrics(self) -> TransitionMetrics:
        """Calculate transition pattern metrics.

        Returns:
            TransitionMetrics dataclass.
        """
        matrix, clusters = self.compute_transition_matrix(normalize=True)

        # Unique transitions (non-zero entries)
        num_transitions = np.sum(matrix > 0)

        # Per-row entropy
        row_entropies = []
        for row in matrix:
            row_probs = row[row > 0]
            if len(row_probs) > 0:
                entropy = -np.sum(row_probs * np.log(row_probs + 1e-10))
                row_entropies.append(entropy)
        avg_entropy = float(np.mean(row_entropies)) if row_entropies else 0.0

        # Self-transitions (diagonal)
        self_trans = np.diag(matrix).mean()

        # Max transition probability
        max_prob = float(matrix.max())

        return TransitionMetrics(
            num_unique_transitions=int(num_transitions),
            avg_transition_entropy=avg_entropy,
            self_transition_ratio=float(self_trans),
            max_transition_prob=max_prob,
        )

    def compare_with_training_distribution(
        self,
        training_clusters: Optional[List[int]] = None,
    ) -> Tuple[pd.DataFrame, float]:
        """Compare generated distribution with training data.

        Args:
            training_clusters: Pre-computed training cluster assignments.
                If None, will extract from ghsom_manager if available.

        Returns:
            Tuple of (comparison DataFrame, KL divergence).
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        # Get training distribution
        if training_clusters is None and self.ghsom_manager is not None:
            # Extract from GHSOM training data
            training_clusters = []
            # This would require accessing the GHSOM's BMU assignments
            # For now, use uniform distribution as fallback
            logger.warning("No training clusters provided, using uniform comparison")
            training_clusters = list(range(self.total_clusters))

        if training_clusters is None:
            raise ValueError("Training clusters required for comparison")

        training_dist = Counter(training_clusters)
        generated_clusters = [c for seq in self.sequences for c in seq]
        generated_dist = Counter(generated_clusters)

        # Normalize
        training_total = sum(training_dist.values())
        generated_total = sum(generated_dist.values())

        all_clusters = sorted(set(training_dist.keys()) | set(generated_dist.keys()))

        comparison = []
        for c in all_clusters:
            comparison.append(
                {
                    "cluster_id": c,
                    "training_freq": training_dist.get(c, 0) / training_total,
                    "generated_freq": generated_dist.get(c, 0) / generated_total,
                    "training_count": training_dist.get(c, 0),
                    "generated_count": generated_dist.get(c, 0),
                }
            )

        df = pd.DataFrame(comparison)
        df["diff"] = df["generated_freq"] - df["training_freq"]
        df["abs_diff"] = np.abs(df["diff"])

        # KL divergence (generated || training)
        p = np.array([training_dist.get(c, 1e-10) for c in all_clusters])
        q = np.array([generated_dist.get(c, 1e-10) for c in all_clusters])
        p = p / p.sum()
        q = q / q.sum()

        # KL(Q || P) - divergence of generated from training
        kl_divergence = float(np.sum(q * np.log((q + 1e-10) / (p + 1e-10))))

        return df, kl_divergence

    def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute all available metrics.

        Returns:
            Dictionary containing all metrics.
        """
        diversity = self.calculate_diversity_metrics()
        transitions = self.calculate_transition_metrics()

        return {
            "num_sequences": len(self.sequences),
            "avg_sequence_length": np.mean([len(seq) for seq in self.sequences]),
            "diversity": diversity.to_dict(),
            "transitions": transitions.to_dict(),
            "cluster_distribution": self.analyze_cluster_distribution().to_dict(
                "records"
            ),
        }

    def generate_report(
        self,
        output_dir: Union[str, Path],
        include_visualizations: bool = True,
    ) -> Path:
        """Generate comprehensive analysis report.

        Args:
            output_dir: Output directory for report files.
            include_visualizations: Whether to generate visualizations.

        Returns:
            Path to report directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute all metrics
        metrics = self.compute_all_metrics()

        # Save metrics JSON
        metrics_path = output_dir / "analysis_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Save cluster distribution CSV
        dist_df = self.analyze_cluster_distribution()
        dist_df.to_csv(output_dir / "cluster_distribution.csv", index=False)

        # Save transition matrix
        trans_matrix, clusters = self.compute_transition_matrix(normalize=True)
        trans_df = pd.DataFrame(trans_matrix, index=clusters, columns=clusters)
        trans_df.to_csv(output_dir / "transition_matrix.csv")

        # Generate visualizations if requested
        if include_visualizations:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            self._generate_visualizations(viz_dir)

        # Generate markdown report
        report_path = output_dir / "analysis_report.md"
        self._generate_markdown_report(report_path, metrics)

        logger.info(f"Report generated at: {output_dir}")
        return output_dir

    def _generate_visualizations(self, output_dir: Path) -> None:
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
        except ImportError:
            logger.warning("matplotlib not available, skipping visualizations")
            return

        plt.style.use("seaborn-v0_8-paper")

        # 1. Cluster distribution bar plot
        self._plot_cluster_distribution(output_dir / "cluster_distribution.png")

        # 2. Transition heatmap
        self._plot_transition_heatmap(output_dir / "transition_heatmap.png")

        # 3. Diversity metrics summary
        self._plot_diversity_summary(output_dir / "diversity_summary.png")

        # 4. Sequence examples
        if len(self.sequences) > 0:
            self._plot_sequence_examples(output_dir / "sequence_examples.png")

    def _plot_cluster_distribution(self, output_path: Path) -> None:
        """Plot cluster distribution bar chart."""
        import matplotlib.pyplot as plt

        dist_df = self.analyze_cluster_distribution()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Top 20 clusters
        top_n = min(20, len(dist_df))
        top_df = dist_df.head(top_n)

        ax.bar(range(top_n), top_df["percentage"], color="steelblue")
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(top_df["cluster_id"], rotation=45)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Usage (%)")
        ax.set_title(f"Top {top_n} Most Used Clusters")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _plot_transition_heatmap(self, output_path: Path) -> None:
        """Plot transition probability heatmap."""
        import matplotlib.pyplot as plt

        matrix, clusters = self.compute_transition_matrix(normalize=True)

        # Limit to top clusters for readability
        n = min(15, len(clusters))

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(matrix[:n, :n], cmap="Blues", aspect="auto")

        ax.set_xticks(range(n))
        ax.set_xticklabels(clusters[:n], rotation=45)
        ax.set_yticks(range(n))
        ax.set_yticklabels(clusters[:n])

        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        ax.set_title("Cluster Transition Probabilities")

        plt.colorbar(im, ax=ax, label="Probability")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _plot_diversity_summary(self, output_path: Path) -> None:
        """Plot diversity metrics summary."""
        import matplotlib.pyplot as plt

        metrics = self.calculate_diversity_metrics()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # 1. Unique clusters per sequence distribution
        ax1 = axes[0]
        unique_counts = [len(set(seq)) for seq in self.sequences]
        ax1.hist(
            unique_counts,
            bins=range(1, max(unique_counts) + 2),
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax1.axvline(
            metrics.unique_per_sequence_mean,
            color="red",
            linestyle="--",
            label=f"Mean: {metrics.unique_per_sequence_mean:.1f}",
        )
        ax1.set_xlabel("Unique Clusters")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Unique Clusters per Sequence")
        ax1.legend()

        # 2. Repetition ratio distribution
        ax2 = axes[1]
        rep_ratios = []
        for seq in self.sequences:
            if len(seq) > 1:
                reps = sum(1 for i in range(len(seq) - 1) if seq[i] == seq[i + 1])
                rep_ratios.append(reps / (len(seq) - 1))
        ax2.hist(rep_ratios, bins=20, color="coral", edgecolor="black", alpha=0.7)
        ax2.axvline(
            metrics.repetition_ratio_mean,
            color="red",
            linestyle="--",
            label=f"Mean: {metrics.repetition_ratio_mean:.2f}",
        )
        ax2.set_xlabel("Repetition Ratio")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Consecutive Repetition Ratio")
        ax2.legend()

        # 3. Summary metrics bar
        ax3 = axes[2]
        metric_names = ["Coverage", "Entropy\n(norm)", "Diversity"]
        metric_values = [
            metrics.coverage_ratio,
            (
                metrics.entropy / np.log(self.total_clusters)
                if self.total_clusters > 1
                else 0
            ),  # Normalized
            1 - metrics.repetition_ratio_mean,  # Diversity = 1 - repetition
        ]
        colors = ["steelblue", "coral", "seagreen"]
        ax3.bar(metric_names, metric_values, color=colors, edgecolor="black")
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Value (0-1)")
        ax3.set_title("Summary Metrics")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _plot_sequence_examples(self, output_path: Path, n_examples: int = 5) -> None:
        """Plot example sequences."""
        import matplotlib.pyplot as plt

        n = min(n_examples, len(self.sequences))

        fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            seq = self.sequences[i]
            ax.plot(seq, "o-", markersize=8, color="steelblue")
            ax.set_ylabel("Cluster ID")
            ax.set_title(f"Sequence {i+1} (len={len(seq)}, unique={len(set(seq))})")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _generate_markdown_report(
        self,
        output_path: Path,
        metrics: Dict[str, Any],
    ) -> None:
        """Generate markdown analysis report."""
        div_metrics = metrics.get("diversity", {})
        trans_metrics = metrics.get("transitions", {})

        report = f"""# Sequence Analysis Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Number of Sequences | {metrics.get('num_sequences', 'N/A')} |
| Avg Sequence Length | {metrics.get('avg_sequence_length', 'N/A'):.1f} |

## Diversity Metrics

| Metric | Value |
|--------|-------|
| Unique Clusters (Total) | {div_metrics.get('unique_clusters_total', 'N/A')} |
| Unique per Sequence | {div_metrics.get('unique_per_sequence_mean', 0):.2f} ± {div_metrics.get('unique_per_sequence_std', 0):.2f} |
| Repetition Ratio | {div_metrics.get('repetition_ratio_mean', 0):.3f} ± {div_metrics.get('repetition_ratio_std', 0):.3f} |
| Entropy | {div_metrics.get('entropy', 0):.3f} |
| Coverage Ratio | {div_metrics.get('coverage_ratio', 0):.2%} |

## Transition Metrics

| Metric | Value |
|--------|-------|
| Unique Transitions | {trans_metrics.get('num_unique_transitions', 'N/A')} |
| Avg Transition Entropy | {trans_metrics.get('avg_transition_entropy', 0):.3f} |
| Self-Transition Ratio | {trans_metrics.get('self_transition_ratio', 0):.3f} |
| Max Transition Prob | {trans_metrics.get('max_transition_prob', 0):.3f} |

## Files Generated

- `analysis_metrics.json` - Full metrics in JSON format
- `cluster_distribution.csv` - Cluster usage statistics
- `transition_matrix.csv` - Transition probability matrix
- `visualizations/` - Generated plots

"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)


def analyze_sequences(
    sequences_path: Union[str, Path],
    output_dir: Union[str, Path],
    ghsom_manager: Optional[GHSOMManager] = None,
) -> Dict[str, Any]:
    """Convenience function to analyze sequences from file.

    Args:
        sequences_path: Path to sequences JSON file.
        output_dir: Directory for analysis output.
        ghsom_manager: Optional GHSOM manager.

    Returns:
        Dictionary of computed metrics.
    """
    analyzer = SequenceAnalyzer(ghsom_manager)
    analyzer.load_sequences(sequences_path)
    analyzer.generate_report(output_dir)
    return analyzer.compute_all_metrics()
