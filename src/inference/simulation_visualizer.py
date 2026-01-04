"""Simulation visualizer for publication-quality figures.

This module provides the SimulationVisualizer class for generating
publication-ready visualizations of HIL simulation results.

Enhanced visualizations include:
- Cluster distribution analysis with heatmaps
- Transition matrix visualization
- Entropy and diversity evolution
- Comprehensive multi-panel academic figures
- Statistical significance annotations
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# Lazy import for matplotlib to avoid import errors in headless environments
def _get_matplotlib():
    """Lazy import matplotlib with proper backend configuration."""
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    return plt


def _get_seaborn():
    """Lazy import seaborn."""
    import seaborn as sns

    return sns


# Default style settings for publication figures (NeurIPS/ICML compatible)
PAPER_STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Extended color palette for academic figures
ACADEMIC_COLORS = {
    "desirable": "#2E86AB",      # Professional blue
    "undesirable": "#D64045",    # Professional red
    "neutral": "#9B9B9B",        # Gray
    "feedback": "#28A745",       # Green
    "entropy": "#6C5CE7",        # Purple
    "reward": "#F39C12",         # Orange
    "adaptation": "#E84393",     # Pink
    "highlight": "#00B894",      # Teal
    "secondary": "#636E72",      # Dark gray
}


class SimulationVisualizer:
    """Generate publication-quality figures for simulation results.

    This class provides methods to visualize HIL simulation results including:
    - Distribution shift plots (before/after comparison)
    - Learning curves (adaptation dynamics over time)
    - Scenario comparison bar charts
    - Ablation study results
    - Cluster distribution analysis with heatmaps
    - Transition matrix visualization
    - Entropy and diversity evolution
    - Comprehensive multi-panel academic figures

    Attributes:
        output_dir: Directory for saving figures.
        style: Plot style settings.
        figure_format: Output format (pdf/png).
    """

    def __init__(
        self,
        output_dir: Path,
        style: str = "paper",
        figure_format: str = "pdf",
    ) -> None:
        """Initialize visualizer.

        Args:
            output_dir: Directory for saving figures.
            style: Plot style ("paper" or "notebook").
            figure_format: Output format ("pdf" or "png").
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_format = figure_format

        # Set style
        self.plt = _get_matplotlib()
        if style == "paper":
            self.plt.rcParams.update(PAPER_STYLE)

        # Use academic color palette
        self.colors = ACADEMIC_COLORS.copy()
        # Add legacy color mappings for backward compatibility
        self.colors.update({
            "q_penalty": "#6C5CE7",
            "reward_shaping": "#F39C12",
        })

        logger.info(
            "Initialized SimulationVisualizer: output=%s, format=%s",
            output_dir,
            figure_format,
        )

    # =========================================================================
    # Helper Methods for Data Extraction
    # =========================================================================

    def _extract_sequences(
        self, result: Dict[str, Any]
    ) -> Tuple[List[List[int]], List[int]]:
        """Extract sequences from metrics history.

        Returns:
            Tuple of (list of sequences, flattened list of all clusters).
        """
        metrics = result.get("metrics_history", [])
        sequences = [m.get("sequence", []) for m in metrics if m.get("sequence")]
        all_clusters = [c for seq in sequences for c in seq]
        return sequences, all_clusters

    def _compute_transition_matrix(
        self, sequences: List[List[int]], normalize: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """Compute transition probability matrix from sequences.

        Returns:
            Tuple of (transition matrix, sorted cluster list).
        """
        transition_counts = defaultdict(lambda: defaultdict(int))

        for seq in sequences:
            for i in range(len(seq) - 1):
                transition_counts[seq[i]][seq[i + 1]] += 1

        all_clusters = sorted(set(c for seq in sequences for c in seq))
        n = len(all_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(all_clusters)}

        matrix = np.zeros((n, n))
        for from_c, to_dict in transition_counts.items():
            for to_c, count in to_dict.items():
                i = cluster_to_idx[from_c]
                j = cluster_to_idx[to_c]
                matrix[i, j] = count

        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            matrix = matrix / row_sums

        return matrix, all_clusters

    def _compute_diversity_metrics(
        self, sequences: List[List[int]]
    ) -> Dict[str, List[float]]:
        """Compute diversity metrics per sequence.

        Returns:
            Dictionary with lists of metrics per sequence.
        """
        unique_per_seq = []
        repetition_ratios = []
        entropies = []

        for seq in sequences:
            if not seq:
                continue

            # Unique clusters
            unique_per_seq.append(len(set(seq)))

            # Repetition ratio
            if len(seq) > 1:
                reps = sum(1 for i in range(len(seq) - 1) if seq[i] == seq[i + 1])
                repetition_ratios.append(reps / (len(seq) - 1))
            else:
                repetition_ratios.append(0.0)

            # Entropy
            counts = Counter(seq)
            total = len(seq)
            probs = np.array([c / total for c in counts.values()])
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        return {
            "unique_per_seq": unique_per_seq,
            "repetition_ratios": repetition_ratios,
            "entropies": entropies,
        }

    def plot_distribution_shift(
        self,
        result: Dict[str, Any],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Plot before/after cluster distribution as stacked bar chart.

        Args:
            result: SimulationResult dictionary.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        fig, ax = plt.subplots(figsize=(8, 5))

        # Extract data
        initial_des = result.get("initial_desirable_ratio", 0)
        final_des = result.get("final_desirable_ratio", 0)
        initial_und = result.get("initial_undesirable_ratio", 0)
        final_und = result.get("final_undesirable_ratio", 0)
        initial_neu = 1 - initial_des - initial_und
        final_neu = 1 - final_des - final_und

        # Data for plotting
        x = np.arange(2)
        width = 0.5

        # Stacked bars
        ax.bar(
            x,
            [initial_des, final_des],
            width,
            label="Desirable",
            color=self.colors["desirable"],
        )
        ax.bar(
            x,
            [initial_neu, final_neu],
            width,
            bottom=[initial_des, final_des],
            label="Neutral",
            color=self.colors["neutral"],
        )
        ax.bar(
            x,
            [initial_und, final_und],
            width,
            bottom=[initial_des + initial_neu, final_des + final_neu],
            label="Undesirable",
            color=self.colors["undesirable"],
        )

        # Formatting
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        ax.set_xticklabels(["Initial", "Final"])
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.05)

        scenario_name = result.get("scenario_name", "Unknown")
        ax.set_title(title or f"Distribution Shift: {scenario_name}")

        # Add improvement annotation
        des_change = final_des - initial_des
        und_change = initial_und - final_und
        ax.annotate(
            f"Desirable: {des_change:+.1%}\nUndesirable: {und_change:+.1%}",
            xy=(1.5, 0.95),
            fontsize=10,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved distribution shift plot to %s", save_path)

        return fig

    def plot_learning_curve(
        self,
        result: Dict[str, Any],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Plot preference alignment over iterations.

        Shows desirable ratio, undesirable ratio, and feedback rating curves.

        Args:
            result: SimulationResult dictionary.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        metrics = result.get("metrics_history", [])
        if not metrics:
            logger.warning("No metrics history available for learning curve")
            return fig

        iterations = [m.get("iteration", i) for i, m in enumerate(metrics)]
        desirable = [m.get("desirable_ratio", 0) for m in metrics]
        undesirable = [m.get("undesirable_ratio", 0) for m in metrics]
        feedback = [m.get("feedback_rating", 3) for m in metrics]

        # Smooth curves using rolling average
        window = min(5, len(iterations) // 5) if len(iterations) > 10 else 1
        if window > 1:
            desirable_smooth = np.convolve(
                desirable, np.ones(window) / window, mode="valid"
            )
            undesirable_smooth = np.convolve(
                undesirable, np.ones(window) / window, mode="valid"
            )
            feedback_smooth = np.convolve(
                feedback, np.ones(window) / window, mode="valid"
            )
            iterations_smooth = iterations[window - 1 :]
        else:
            desirable_smooth = desirable
            undesirable_smooth = undesirable
            feedback_smooth = feedback
            iterations_smooth = iterations

        # Plot ratios
        ax1.plot(
            iterations_smooth,
            desirable_smooth,
            label="Desirable",
            color=self.colors["desirable"],
            linewidth=2,
        )
        ax1.plot(
            iterations_smooth,
            undesirable_smooth,
            label="Undesirable",
            color=self.colors["undesirable"],
            linewidth=2,
        )
        ax1.fill_between(
            iterations, desirable, alpha=0.2, color=self.colors["desirable"]
        )
        ax1.fill_between(
            iterations, undesirable, alpha=0.2, color=self.colors["undesirable"]
        )

        ax1.set_ylabel("Cluster Ratio")
        ax1.legend(loc="upper right")
        ax1.set_ylim(0, 1)

        # Add target line if available
        target_value = result.get("target_value")
        if target_value:
            ax1.axhline(
                y=target_value,
                color="gray",
                linestyle="--",
                label=f"Target ({target_value:.0%})",
                alpha=0.7,
            )

        # Plot feedback
        ax2.plot(
            iterations_smooth,
            feedback_smooth,
            label="Feedback Rating",
            color=self.colors["feedback"],
            linewidth=2,
        )
        ax2.fill_between(iterations, feedback, alpha=0.2, color=self.colors["feedback"])

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Feedback Rating (1-5)")
        ax2.set_ylim(1, 5)
        ax2.axhline(y=3.0, color="gray", linestyle=":", alpha=0.5)

        scenario_name = result.get("scenario_name", "Unknown")
        fig.suptitle(title or f"Learning Dynamics: {scenario_name}", fontsize=14)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved learning curve to %s", save_path)

        return fig

    def plot_scenario_comparison(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        metric: str = "distribution_shift",
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Bar chart comparing scenarios on a key metric.

        Args:
            results: Dictionary mapping scenario names to lists of result dicts.
            metric: Metric to compare ("distribution_shift", "feedback_improvement", etc.).
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        fig, ax = plt.subplots(figsize=(10, 6))

        scenarios = list(results.keys())
        means = []
        stds = []

        metric_key_map = {
            "distribution_shift": lambda r: r.get("final_desirable_ratio", 0)
            - r.get("initial_desirable_ratio", 0),
            "feedback_improvement": lambda r: r.get("feedback_improvement", 0),
            "undesirable_reduction": lambda r: r.get("initial_undesirable_ratio", 0)
            - r.get("final_undesirable_ratio", 0),
            "final_desirable": lambda r: r.get("final_desirable_ratio", 0),
        }

        metric_fn = metric_key_map.get(metric, metric_key_map["distribution_shift"])

        for scenario in scenarios:
            scenario_results = results[scenario]
            values = [metric_fn(r) for r in scenario_results]
            means.append(np.mean(values))
            stds.append(np.std(values))

        x = np.arange(len(scenarios))
        width = 0.6

        bars = ax.bar(
            x, means, width, yerr=stds, capsize=5, color=self.colors["desirable"]
        )

        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.02,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Scenario")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], rotation=0)

        metric_titles = {
            "distribution_shift": "Desirable Ratio Improvement by Scenario",
            "feedback_improvement": "Feedback Improvement by Scenario",
            "undesirable_reduction": "Undesirable Cluster Reduction by Scenario",
            "final_desirable": "Final Desirable Ratio by Scenario",
        }
        ax.set_title(title or metric_titles.get(metric, f"{metric} by Scenario"))

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved scenario comparison to %s", save_path)

        return fig

    def plot_ablation_study(
        self,
        results: Dict[float, List[Dict[str, Any]]],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Plot ablation results for adaptation strength.

        Args:
            results: Dictionary mapping strength values to result lists.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        fig, ax = plt.subplots(figsize=(8, 5))

        strengths = sorted(results.keys())
        means = []
        stds = []

        for strength in strengths:
            strength_results = results[strength]
            improvements = [
                r.get("final_desirable_ratio", 0) - r.get("initial_desirable_ratio", 0)
                for r in strength_results
            ]
            means.append(np.mean(improvements))
            stds.append(np.std(improvements))

        ax.errorbar(
            strengths,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=5,
            color=self.colors["q_penalty"],
            markersize=8,
        )

        ax.set_xlabel("Adaptation Strength")
        ax.set_ylabel("Desirable Ratio Improvement")
        ax.set_title(title or "Ablation Study: Effect of Adaptation Strength")

        # Add trend indication
        if len(means) > 1:
            best_idx = np.argmax(means)
            ax.axvline(
                x=strengths[best_idx],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"Optimal: {strengths[best_idx]}",
            )
            ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved ablation study plot to %s", save_path)

        return fig

    def plot_mode_comparison(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Plot comparison between adaptation modes.

        Args:
            results: Dictionary mapping mode names to result lists.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        fig, ax = plt.subplots(figsize=(8, 5))

        modes = list(results.keys())
        x = np.arange(len(modes))
        width = 0.35

        # Calculate metrics for each mode
        des_improvements = []
        des_stds = []
        fb_improvements = []
        fb_stds = []

        for mode in modes:
            mode_results = results[mode]
            des = [
                r.get("final_desirable_ratio", 0) - r.get("initial_desirable_ratio", 0)
                for r in mode_results
            ]
            fb = [r.get("feedback_improvement", 0) for r in mode_results]

            des_improvements.append(np.mean(des))
            des_stds.append(np.std(des))
            fb_improvements.append(np.mean(fb))
            fb_stds.append(np.std(fb))

        # Plot grouped bars
        bars1 = ax.bar(
            x - width / 2,
            des_improvements,
            width,
            yerr=des_stds,
            label="Desirable Improvement",
            color=self.colors["desirable"],
            capsize=3,
        )
        bars2 = ax.bar(
            x + width / 2,
            fb_improvements,
            width,
            yerr=fb_stds,
            label="Feedback Improvement",
            color=self.colors["feedback"],
            capsize=3,
        )

        ax.set_xlabel("Adaptation Mode")
        ax.set_ylabel("Improvement")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in modes])
        ax.legend()
        ax.set_title(title or "Adaptation Mode Comparison")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved mode comparison to %s", save_path)

        return fig

    # =========================================================================
    # Enhanced Academic Visualization Methods
    # =========================================================================

    def plot_cluster_distribution(
        self,
        result: Dict[str, Any],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        top_n: int = 15,
    ) -> Any:
        """Plot cluster usage distribution as horizontal bar chart.

        Args:
            result: SimulationResult dictionary with metrics_history.
            save_path: Optional path to save figure.
            title: Optional custom title.
            top_n: Number of top clusters to show.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt
        sequences, all_clusters = self._extract_sequences(result)

        if not all_clusters:
            logger.warning("No sequence data for cluster distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No sequence data available", ha="center", va="center")
            return fig

        # Count cluster frequencies
        cluster_counts = Counter(all_clusters)
        total = sum(cluster_counts.values())

        # Get top N clusters
        top_clusters = cluster_counts.most_common(top_n)
        clusters, counts = zip(*top_clusters)
        percentages = [c / total * 100 for c in counts]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create horizontal bars
        y_pos = np.arange(len(clusters))
        bars = ax.barh(y_pos, percentages, color=self.colors["desirable"], alpha=0.8)

        # Add value labels
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=9,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"Cluster {c}" for c in clusters])
        ax.invert_yaxis()
        ax.set_xlabel("Usage Frequency (%)")
        ax.set_title(title or "Cluster Usage Distribution")

        # Add summary statistics
        unique_total = len(set(all_clusters))
        entropy = -sum((c / total) * np.log(c / total + 1e-10) for c in cluster_counts.values())
        ax.text(
            0.98, 0.02,
            f"Unique clusters: {unique_total}\nEntropy: {entropy:.2f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved cluster distribution to %s", save_path)

        return fig

    def plot_transition_heatmap(
        self,
        result: Dict[str, Any],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
        max_clusters: int = 12,
    ) -> Any:
        """Plot transition probability heatmap.

        Args:
            result: SimulationResult dictionary with metrics_history.
            save_path: Optional path to save figure.
            title: Optional custom title.
            max_clusters: Maximum clusters to show.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt
        sns = _get_seaborn()

        sequences, _ = self._extract_sequences(result)

        if not sequences:
            logger.warning("No sequence data for transition heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No sequence data available", ha="center", va="center")
            return fig

        matrix, clusters = self._compute_transition_matrix(sequences)

        # Limit to top clusters by frequency
        if len(clusters) > max_clusters:
            all_clusters_flat = [c for seq in sequences for c in seq]
            top_clusters = [c for c, _ in Counter(all_clusters_flat).most_common(max_clusters)]
            idx_mask = [i for i, c in enumerate(clusters) if c in top_clusters]
            matrix = matrix[np.ix_(idx_mask, idx_mask)]
            clusters = [clusters[i] for i in idx_mask]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = sns.heatmap(
            matrix,
            ax=ax,
            cmap="Blues",
            annot=True if len(clusters) <= 10 else False,
            fmt=".2f",
            xticklabels=[f"C{c}" for c in clusters],
            yticklabels=[f"C{c}" for c in clusters],
            cbar_kws={"label": "Transition Probability"},
            square=True,
            linewidths=0.5,
        )

        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        ax.set_title(title or "Cluster Transition Probabilities")

        # Add self-transition statistic
        self_trans = np.diag(matrix).mean()
        ax.text(
            0.98, 0.02,
            f"Self-transition: {self_trans:.2%}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved transition heatmap to %s", save_path)

        return fig

    def plot_entropy_evolution(
        self,
        result: Dict[str, Any],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Plot entropy and diversity metrics evolution over iterations.

        Args:
            result: SimulationResult dictionary with metrics_history.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        metrics = result.get("metrics_history", [])
        if not metrics:
            logger.warning("No metrics history for entropy evolution")
            fig, ax = plt.subplots()
            return fig

        iterations = [m.get("iteration", i) for i, m in enumerate(metrics)]
        entropies = [m.get("entropy", 0) for m in metrics]
        rewards = [m.get("episode_reward", 0) for m in metrics]
        feedback = [m.get("feedback_rating", 3) for m in metrics]
        adaptation_applied = [m.get("adaptation_applied", False) for m in metrics]

        # Create multi-panel figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Panel 1: Entropy
        ax1 = axes[0]
        ax1.plot(iterations, entropies, color=self.colors["entropy"], linewidth=2, label="Entropy")
        ax1.fill_between(iterations, entropies, alpha=0.2, color=self.colors["entropy"])

        # Mark adaptation points
        adapt_iters = [it for it, applied in zip(iterations, adaptation_applied) if applied]
        adapt_entropy = [e for e, applied in zip(entropies, adaptation_applied) if applied]
        ax1.scatter(adapt_iters, adapt_entropy, color=self.colors["adaptation"],
                   marker="^", s=60, zorder=5, label="Adaptation Applied")

        ax1.set_ylabel("Entropy")
        ax1.legend(loc="upper right")
        ax1.set_title("(a) Sequence Diversity (Entropy)")

        # Panel 2: Episode Reward
        ax2 = axes[1]
        ax2.plot(iterations, rewards, color=self.colors["reward"], linewidth=2, label="Episode Reward")
        ax2.fill_between(iterations, rewards, alpha=0.2, color=self.colors["reward"])
        ax2.scatter(adapt_iters, [rewards[iterations.index(i)] for i in adapt_iters],
                   color=self.colors["adaptation"], marker="^", s=60, zorder=5)
        ax2.set_ylabel("Episode Reward")
        ax2.legend(loc="upper right")
        ax2.set_title("(b) Episode Reward Dynamics")

        # Panel 3: Feedback with moving average
        ax3 = axes[2]
        ax3.plot(iterations, feedback, color=self.colors["feedback"], linewidth=1.5,
                alpha=0.5, label="Feedback")

        # Add moving average
        window = min(5, len(feedback) // 5) if len(feedback) > 10 else 1
        if window > 1:
            feedback_smooth = np.convolve(feedback, np.ones(window) / window, mode="valid")
            iter_smooth = iterations[window - 1:]
            ax3.plot(iter_smooth, feedback_smooth, color=self.colors["feedback"],
                    linewidth=2.5, label=f"Moving Avg (w={window})")

        ax3.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="Neutral (3.0)")
        ax3.scatter(adapt_iters, [feedback[iterations.index(i)] for i in adapt_iters],
                   color=self.colors["adaptation"], marker="^", s=60, zorder=5,
                   label="Adaptation Applied")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Feedback Rating")
        ax3.set_ylim(1, 5)
        ax3.legend(loc="upper right", ncol=2)
        ax3.set_title("(c) Human Feedback Alignment")

        scenario_name = result.get("scenario_name", "")
        fig.suptitle(
            title or f"HIL Adaptation Dynamics: {scenario_name.replace('_', ' ').title()}",
            fontsize=14, fontweight="bold"
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved entropy evolution to %s", save_path)

        return fig

    def plot_comprehensive_analysis(
        self,
        result: Dict[str, Any],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Create comprehensive multi-panel figure for academic papers.

        This figure combines:
        - Distribution shift (before/after)
        - Cluster usage distribution
        - Transition heatmap
        - Learning dynamics

        Args:
            result: SimulationResult dictionary.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt
        sns = _get_seaborn()

        # Create figure with GridSpec for flexible layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # =====================================================================
        # Panel A (top-left): Distribution Shift
        # =====================================================================
        ax_dist = fig.add_subplot(gs[0, 0])

        initial_des = result.get("initial_desirable_ratio", 0)
        final_des = result.get("final_desirable_ratio", 0)
        initial_und = result.get("initial_undesirable_ratio", 0)
        final_und = result.get("final_undesirable_ratio", 0)

        categories = ["Initial", "Final"]
        x = np.arange(len(categories))
        width = 0.35

        ax_dist.bar(x - width/2, [initial_des, final_des], width,
                   label="Desirable", color=self.colors["desirable"])
        ax_dist.bar(x + width/2, [initial_und, final_und], width,
                   label="Undesirable", color=self.colors["undesirable"])

        ax_dist.set_ylabel("Ratio")
        ax_dist.set_xticks(x)
        ax_dist.set_xticklabels(categories)
        ax_dist.legend(loc="upper right")
        ax_dist.set_title("(A) Preference Distribution Shift")
        ax_dist.set_ylim(0, 1)

        # Add change annotation
        des_change = final_des - initial_des
        ax_dist.annotate(
            f"Δ = {des_change:+.1%}",
            xy=(1, final_des), xytext=(1.3, final_des + 0.1),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

        # =====================================================================
        # Panel B (top-center): Cluster Usage Histogram
        # =====================================================================
        ax_cluster = fig.add_subplot(gs[0, 1])

        sequences, all_clusters = self._extract_sequences(result)
        if all_clusters:
            cluster_counts = Counter(all_clusters)
            top_clusters = cluster_counts.most_common(10)
            clusters, counts = zip(*top_clusters)
            total = sum(cluster_counts.values())
            percentages = [c / total * 100 for c in counts]

            bars = ax_cluster.bar(range(len(clusters)), percentages,
                                 color=self.colors["desirable"], alpha=0.8)
            ax_cluster.set_xticks(range(len(clusters)))
            ax_cluster.set_xticklabels([f"C{c}" for c in clusters], rotation=45, ha="right")
            ax_cluster.set_ylabel("Usage (%)")
            ax_cluster.set_title("(B) Top 10 Cluster Usage")
        else:
            ax_cluster.text(0.5, 0.5, "No data", ha="center", va="center")

        # =====================================================================
        # Panel C (top-right): Key Metrics Summary
        # =====================================================================
        ax_metrics = fig.add_subplot(gs[0, 2])

        metrics_data = {
            "Feedback Δ": result.get("feedback_improvement", 0),
            "Desirable Δ": final_des - initial_des,
            "Undesirable Δ": initial_und - final_und,
        }

        colors_list = [self.colors["feedback"], self.colors["desirable"], self.colors["highlight"]]
        bars = ax_metrics.barh(list(metrics_data.keys()), list(metrics_data.values()),
                              color=colors_list, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, metrics_data.values()):
            ax_metrics.text(bar.get_width() + 0.01 if val >= 0 else bar.get_width() - 0.01,
                           bar.get_y() + bar.get_height() / 2,
                           f"{val:+.3f}", va="center",
                           ha="left" if val >= 0 else "right", fontsize=9)

        ax_metrics.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
        ax_metrics.set_xlabel("Change")
        ax_metrics.set_title("(C) Performance Metrics")

        # =====================================================================
        # Panel D (middle row): Learning Curves
        # =====================================================================
        ax_learning = fig.add_subplot(gs[1, :])

        metrics_history = result.get("metrics_history", [])
        if metrics_history:
            iterations = [m.get("iteration", i) for i, m in enumerate(metrics_history)]
            desirable = [m.get("desirable_ratio", 0) for m in metrics_history]
            undesirable = [m.get("undesirable_ratio", 0) for m in metrics_history]
            feedback = [m.get("feedback_rating", 3) for m in metrics_history]

            ax_learning.plot(iterations, desirable, color=self.colors["desirable"],
                           linewidth=2, label="Desirable Ratio")
            ax_learning.plot(iterations, undesirable, color=self.colors["undesirable"],
                           linewidth=2, label="Undesirable Ratio")

            # Secondary y-axis for feedback
            ax_feedback = ax_learning.twinx()
            ax_feedback.plot(iterations, feedback, color=self.colors["feedback"],
                           linewidth=2, linestyle="--", label="Feedback")
            ax_feedback.set_ylabel("Feedback Rating", color=self.colors["feedback"])
            ax_feedback.set_ylim(1, 5)
            ax_feedback.tick_params(axis="y", labelcolor=self.colors["feedback"])

            ax_learning.set_xlabel("Iteration")
            ax_learning.set_ylabel("Cluster Ratio")
            ax_learning.set_ylim(0, 1)
            ax_learning.legend(loc="upper left")
            ax_feedback.legend(loc="upper right")

        ax_learning.set_title("(D) Learning Dynamics Over Iterations")

        # =====================================================================
        # Panel E (bottom-left): Transition Heatmap
        # =====================================================================
        ax_trans = fig.add_subplot(gs[2, 0:2])

        if sequences:
            matrix, cluster_list = self._compute_transition_matrix(sequences)

            # Limit to top 8 clusters
            if len(cluster_list) > 8:
                all_clusters_flat = [c for seq in sequences for c in seq]
                top_clusters = [c for c, _ in Counter(all_clusters_flat).most_common(8)]
                idx_mask = [i for i, c in enumerate(cluster_list) if c in top_clusters]
                matrix = matrix[np.ix_(idx_mask, idx_mask)]
                cluster_list = [cluster_list[i] for i in idx_mask]

            sns.heatmap(
                matrix, ax=ax_trans, cmap="Blues",
                annot=True, fmt=".2f",
                xticklabels=[f"C{c}" for c in cluster_list],
                yticklabels=[f"C{c}" for c in cluster_list],
                cbar_kws={"label": "P(transition)"},
                square=True,
            )
            ax_trans.set_xlabel("To Cluster")
            ax_trans.set_ylabel("From Cluster")
        ax_trans.set_title("(E) Cluster Transition Matrix")

        # =====================================================================
        # Panel F (bottom-right): Diversity Metrics
        # =====================================================================
        ax_div = fig.add_subplot(gs[2, 2])

        if sequences:
            div_metrics = self._compute_diversity_metrics(sequences)

            # Create box plots
            data_to_plot = [
                div_metrics["unique_per_seq"],
                [r * 10 for r in div_metrics["repetition_ratios"]],  # Scale for visibility
                div_metrics["entropies"],
            ]
            labels = ["Unique\nClusters", "Repetition\n(×10)", "Entropy"]

            bp = ax_div.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)

            colors_box = [self.colors["desirable"], self.colors["undesirable"], self.colors["entropy"]]
            for patch, color in zip(bp["boxes"], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax_div.set_ylabel("Value")
        ax_div.set_title("(F) Sequence Diversity Metrics")

        # Main title
        scenario_name = result.get("scenario_name", "HIL Simulation")
        fig.suptitle(
            title or f"Comprehensive HIL Analysis: {scenario_name.replace('_', ' ').title()}",
            fontsize=16, fontweight="bold", y=1.02
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format, bbox_inches="tight")
            logger.info("Saved comprehensive analysis to %s", save_path)

        return fig

    def plot_multi_scenario_heatmap(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Create heatmap comparing multiple scenarios across metrics.

        Args:
            results: Dictionary mapping scenario names to result lists.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt
        sns = _get_seaborn()

        scenarios = list(results.keys())
        metrics = [
            "Desirable Δ",
            "Undesirable Δ",
            "Feedback Δ",
            "Final Desirable",
            "Final Undesirable",
        ]

        # Build data matrix
        data = np.zeros((len(scenarios), len(metrics)))

        for i, scenario in enumerate(scenarios):
            scenario_results = results[scenario]

            # Aggregate across seeds
            des_imp = np.mean([r.get("final_desirable_ratio", 0) - r.get("initial_desirable_ratio", 0)
                             for r in scenario_results])
            und_red = np.mean([r.get("initial_undesirable_ratio", 0) - r.get("final_undesirable_ratio", 0)
                             for r in scenario_results])
            fb_imp = np.mean([r.get("feedback_improvement", 0) for r in scenario_results])
            final_des = np.mean([r.get("final_desirable_ratio", 0) for r in scenario_results])
            final_und = np.mean([r.get("final_undesirable_ratio", 0) for r in scenario_results])

            data[i] = [des_imp, und_red, fb_imp, final_des, final_und]

        # Normalize each column for better visualization
        data_normalized = data.copy()
        for j in range(data.shape[1]):
            col = data[:, j]
            if col.max() != col.min():
                data_normalized[:, j] = (col - col.min()) / (col.max() - col.min())
            else:
                data_normalized[:, j] = 0.5

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap with original values as annotations
        im = sns.heatmap(
            data_normalized,
            ax=ax,
            cmap="RdYlGn",
            annot=data,
            fmt=".3f",
            xticklabels=metrics,
            yticklabels=[s.replace("_", "\n") for s in scenarios],
            cbar_kws={"label": "Normalized Score"},
            linewidths=0.5,
        )

        ax.set_xlabel("Metric")
        ax.set_ylabel("Scenario")
        ax.set_title(title or "Multi-Scenario Performance Comparison")

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved multi-scenario heatmap to %s", save_path)

        return fig

    def plot_statistical_comparison(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        save_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> Any:
        """Create statistical comparison with significance annotations.

        Args:
            results: Dictionary mapping scenario names to result lists.
            save_path: Optional path to save figure.
            title: Optional custom title.

        Returns:
            Matplotlib figure object.
        """
        plt = self.plt

        scenarios = list(results.keys())
        n_scenarios = len(scenarios)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_config = [
            ("Desirable Improvement", lambda r: r.get("final_desirable_ratio", 0) - r.get("initial_desirable_ratio", 0)),
            ("Feedback Improvement", lambda r: r.get("feedback_improvement", 0)),
            ("Undesirable Reduction", lambda r: r.get("initial_undesirable_ratio", 0) - r.get("final_undesirable_ratio", 0)),
        ]

        for ax, (metric_name, metric_fn) in zip(axes, metrics_config):
            # Collect data
            all_values = []
            for scenario in scenarios:
                values = [metric_fn(r) for r in results[scenario]]
                all_values.append(values)

            # Create violin plot
            parts = ax.violinplot(all_values, positions=range(n_scenarios), showmeans=True)

            # Color violins
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(self.colors["desirable"])
                pc.set_alpha(0.7)

            # Add individual points
            for i, values in enumerate(all_values):
                x = np.random.normal(i, 0.04, len(values))
                ax.scatter(x, values, alpha=0.6, s=20, color=self.colors["secondary"])

            # Add mean markers
            means = [np.mean(v) for v in all_values]
            ax.scatter(range(n_scenarios), means, color=self.colors["highlight"],
                      marker="D", s=50, zorder=5, label="Mean")

            ax.set_xticks(range(n_scenarios))
            ax.set_xticklabels([s.replace("_", "\n")[:15] for s in scenarios], rotation=45, ha="right")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

            # Perform ANOVA if multiple scenarios
            if n_scenarios > 1 and all(len(v) > 1 for v in all_values):
                try:
                    f_stat, p_value = stats.f_oneway(*all_values)
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    ax.text(0.98, 0.98, f"ANOVA: p={p_value:.3f} ({sig_marker})",
                           transform=ax.transAxes, ha="right", va="top",
                           fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
                except Exception:
                    pass

        fig.suptitle(title or "Statistical Comparison Across Scenarios", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, format=self.figure_format)
            logger.info("Saved statistical comparison to %s", save_path)

        return fig

    def generate_scenario_figures(
        self,
        scenario_name: str,
        scenario_results: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Generate detailed figures for a single scenario.

        Creates scenario-specific visualizations including comprehensive analysis,
        entropy evolution, cluster distribution, transition heatmaps, and more.

        Args:
            scenario_name: Name of the scenario.
            scenario_results: List of result dictionaries for this scenario.
            output_dir: Output directory. If None, uses self.output_dir/scenario_name.

        Returns:
            List of paths to generated figures.
        """
        if not scenario_results:
            logger.warning("No results provided for scenario: %s", scenario_name)
            return []

        # Use scenario-specific subdirectory
        if output_dir is None:
            output_dir = self.output_dir / scenario_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: List[Path] = []
        result = scenario_results[0]  # Use first seed for detailed analysis
        scenario_title = scenario_name.replace("_", " ").title()

        # Figure 1: Comprehensive Multi-Panel Analysis
        path = output_dir / f"comprehensive_analysis.{self.figure_format}"
        self.plot_comprehensive_analysis(
            result,
            save_path=path,
            title=f"Comprehensive Analysis: {scenario_title}",
        )
        generated_files.append(path)

        # Figure 2: Entropy Evolution and Dynamics
        path = output_dir / f"entropy_evolution.{self.figure_format}"
        self.plot_entropy_evolution(
            result,
            save_path=path,
            title=f"Adaptation Dynamics: {scenario_title}",
        )
        generated_files.append(path)

        # Figure 3: Cluster Usage Distribution
        path = output_dir / f"cluster_distribution.{self.figure_format}"
        self.plot_cluster_distribution(
            result,
            save_path=path,
            title=f"Cluster Usage: {scenario_title}",
        )
        generated_files.append(path)

        # Figure 4: Transition Heatmap
        path = output_dir / f"transition_heatmap.{self.figure_format}"
        self.plot_transition_heatmap(
            result,
            save_path=path,
            title=f"Cluster Transitions: {scenario_title}",
        )
        generated_files.append(path)

        # Figure 5: Distribution Shift
        path = output_dir / f"distribution_shift.{self.figure_format}"
        self.plot_distribution_shift(
            result,
            save_path=path,
            title=f"Distribution Shift: {scenario_title}",
        )
        generated_files.append(path)

        # Figure 6: Learning Curve
        path = output_dir / f"learning_curves.{self.figure_format}"
        self.plot_learning_curve(
            result,
            save_path=path,
            title=f"Learning Dynamics: {scenario_title}",
        )
        generated_files.append(path)

        logger.info(
            "Generated %d scenario figures for '%s' in %s",
            len(generated_files),
            scenario_name,
            output_dir,
        )
        return generated_files

    def generate_paper_figures(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        output_dir: Optional[Path] = None,
        ablation_results: Optional[Dict[float, List[Dict[str, Any]]]] = None,
        mode_results: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        per_scenario_figures: bool = True,
    ) -> List[Path]:
        """Generate comprehensive academic-quality figures for paper.

        Generates a complete set of publication-ready visualizations including:
        - Multi-scenario comparison heatmap
        - Statistical comparison with significance tests
        - Per-scenario detailed analysis figures (in scenario subdirectories)
        - Cluster distribution and transition analyses
        - Entropy evolution and diversity metrics

        Args:
            all_results: Results for all scenarios.
            output_dir: Output directory. If None, uses self.output_dir.
            ablation_results: Optional ablation study results.
            mode_results: Optional mode comparison results.
            per_scenario_figures: If True, generate detailed figures for each
                scenario in scenario-specific subdirectories. Default True.

        Returns:
            List of paths to generated figures.
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files: List[Path] = []

        # =====================================================================
        # Multi-scenario comparison figures (in comparison/ subdirectory)
        # =====================================================================
        if len(all_results) > 1:
            comparison_dir = output_dir / "comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)

            # Figure 1: Multi-Scenario Heatmap (Main Result)
            path = comparison_dir / f"fig1_scenario_heatmap.{self.figure_format}"
            self.plot_multi_scenario_heatmap(
                all_results,
                save_path=path,
                title="HIL Preference Adaptation: Multi-Scenario Performance",
            )
            generated_files.append(path)

            # Figure 2: Statistical Comparison with Significance
            path = comparison_dir / f"fig2_statistical_comparison.{self.figure_format}"
            self.plot_statistical_comparison(
                all_results,
                save_path=path,
                title="Statistical Analysis Across Scenarios",
            )
            generated_files.append(path)

            # Figure 3: Scenario Comparison Bar Chart
            path = comparison_dir / f"fig3_scenario_comparison.{self.figure_format}"
            self.plot_scenario_comparison(
                all_results,
                metric="distribution_shift",
                save_path=path,
                title="Desirable Cluster Improvement by Scenario",
            )
            generated_files.append(path)
        elif len(all_results) == 1:
            # Single scenario - save comparison figure to main output dir
            path = output_dir / f"fig3_scenario_comparison.{self.figure_format}"
            self.plot_scenario_comparison(
                all_results,
                metric="distribution_shift",
                save_path=path,
                title="Desirable Cluster Improvement by Scenario",
            )
            generated_files.append(path)

        # =====================================================================
        # Per-scenario detailed figures (in scenario-specific subdirectories)
        # =====================================================================
        if per_scenario_figures and all_results:
            for scenario_name, scenario_results in all_results.items():
                scenario_dir = output_dir / scenario_name
                scenario_figures = self.generate_scenario_figures(
                    scenario_name=scenario_name,
                    scenario_results=scenario_results,
                    output_dir=scenario_dir,
                )
                generated_files.extend(scenario_figures)

        # =====================================================================
        # Ablation Study figures (in ablation/ subdirectory)
        # =====================================================================
        if ablation_results:
            ablation_dir = output_dir / "ablation"
            ablation_dir.mkdir(parents=True, exist_ok=True)
            path = ablation_dir / f"ablation_study.{self.figure_format}"
            self.plot_ablation_study(
                ablation_results,
                save_path=path,
                title="Ablation: Effect of Adaptation Strength",
            )
            generated_files.append(path)

        # =====================================================================
        # Mode Comparison figures (in mode_comparison/ subdirectory)
        # =====================================================================
        if mode_results:
            mode_dir = output_dir / "mode_comparison"
            mode_dir.mkdir(parents=True, exist_ok=True)
            path = mode_dir / f"mode_comparison.{self.figure_format}"
            self.plot_mode_comparison(
                mode_results,
                save_path=path,
                title="Comparison of Adaptation Modes",
            )
            generated_files.append(path)

        logger.info(
            "Generated %d academic-quality figures in %s",
            len(generated_files),
            output_dir,
        )
        return generated_files

    def close_all(self) -> None:
        """Close all matplotlib figures to free memory."""
        self.plt.close("all")
