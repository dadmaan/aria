#!/usr/bin/env python3
"""GHSOM sweep results analysis.

Analyzes completed GHSOM sweep runs and generates:
- Pareto front visualizations
- Hyperparameter importance plots
- Best configurations report
- CSV/JSON exports

Usage:
    # Analyze from local results directory
    python scripts/ghsom/analyze_ghsom_sweep.py --input outputs/ghsom_sweep/

    # Export top N configurations
    python scripts/ghsom/analyze_ghsom_sweep.py --input outputs/ghsom_sweep/ --top-k 10

    # Generate all visualizations
    python scripts/ghsom/analyze_ghsom_sweep.py --input outputs/ghsom_sweep/ --plot-all

    # Quiet mode (minimal output)
    python scripts/ghsom/analyze_ghsom_sweep.py --input outputs/ghsom_sweep/ --quiet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ghsom_sweep"

# Hyperparameter names for analysis
HYPERPARAMS = [
    "t1",
    "t2",
    "learning_rate",
    "decay",
    "gaussian_sigma",
    "epochs",
    "grow_maxiter",
]

# Score component names
SCORE_COMPONENTS = [
    "composite_score",
    "cluster_score",
    "activation_score",
    "dispersion_score",
]

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ghsom_sweep_analyzer")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SweepResult:
    """Single sweep trial result.

    Attributes:
        trial_id: Unique identifier for the trial.
        hyperparams: Dictionary of hyperparameter values.
        metrics: Dictionary of GHSOM metrics (num_clusters, etc.).
        scores: Dictionary of score components.
        success: Whether training succeeded.
        error: Error message if training failed.
    """

    trial_id: str
    hyperparams: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], trial_id: str = "") -> "SweepResult":
        """Create SweepResult from a trial result dictionary.

        Args:
            data: Trial result dictionary (from run_ghsom_trial output).
            trial_id: Unique identifier for this trial.

        Returns:
            SweepResult instance.
        """
        return cls(
            trial_id=trial_id or data.get("trial_id", "unknown"),
            hyperparams=data.get("config", {}),
            metrics=data.get("metrics", {}),
            scores=data.get("scores", {}),
            success=data.get("success", False),
            error=data.get("error"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @property
    def composite_score(self) -> float:
        """Get composite score value."""
        return float(self.scores.get("composite_score", 0.0))

    @property
    def num_clusters(self) -> int:
        """Get number of clusters."""
        return int(self.metrics.get("num_clusters", 0))


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


class SweepAnalyzer:
    """Analyzer for GHSOM sweep results.

    Provides methods for loading, analyzing, and visualizing sweep results.
    """

    def __init__(self, results: Optional[List[SweepResult]] = None):
        """Initialize analyzer with optional results.

        Args:
            results: List of sweep results to analyze.
        """
        self.results: List[SweepResult] = results or []

    def load_from_directory(self, path: Path) -> List[SweepResult]:
        """Load sweep results from a directory containing JSON files.

        Expected structure:
            path/
                sweep_results.json  (contains list of trial results)
                OR
                trial_*.json        (individual trial files)

        Args:
            path: Directory containing sweep results.

        Returns:
            List of SweepResult objects.
        """
        results: List[SweepResult] = []
        path = Path(path)

        if not path.exists():
            logger.warning(f"Directory not found: {path}")
            return results

        # Try to load combined results file first
        combined_file = path / "sweep_results.json"
        if combined_file.exists():
            logger.info(f"Loading combined results from: {combined_file}")
            with combined_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                trials = data.get("trials", data) if isinstance(data, dict) else data
                for i, trial in enumerate(trials):
                    trial_id = trial.get("trial_id", f"trial_{i:04d}")
                    results.append(SweepResult.from_dict(trial, trial_id))
        else:
            # Load individual trial files
            trial_files = sorted(path.glob("trial_*.json"))
            logger.info(f"Loading {len(trial_files)} individual trial files")
            for trial_file in trial_files:
                with trial_file.open("r", encoding="utf-8") as f:
                    trial = json.load(f)
                    trial_id = trial_file.stem
                    results.append(SweepResult.from_dict(trial, trial_id))

        self.results = results
        logger.info(f"Loaded {len(results)} sweep results")
        return results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis.

        Returns:
            DataFrame with columns for hyperparameters, metrics, and scores.
        """
        records = []
        for result in self.results:
            record = {
                "trial_id": result.trial_id,
                "success": result.success,
                **{f"hp_{k}": v for k, v in result.hyperparams.items()},
                **{f"metric_{k}": v for k, v in result.metrics.items()},
                **result.scores,
            }
            records.append(record)
        return pd.DataFrame(records)

    def get_successful_results(self) -> List[SweepResult]:
        """Get only successful trial results.

        Returns:
            List of successful SweepResult objects.
        """
        return [r for r in self.results if r.success]

    def get_top_configs(
        self, n: int = 10, metric: str = "composite_score"
    ) -> List[SweepResult]:
        """Get top N configurations ranked by a metric.

        Args:
            n: Number of top configurations to return.
            metric: Metric to rank by (default: composite_score).

        Returns:
            List of top N SweepResult objects.
        """
        successful = self.get_successful_results()
        sorted_results = sorted(
            successful, key=lambda r: r.scores.get(metric, 0.0), reverse=True
        )
        return sorted_results[:n]

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the sweep.

        Returns:
            Dictionary with statistics for scores and metrics.
        """
        successful = self.get_successful_results()
        if not successful:
            return {"error": "No successful results"}

        df = self.to_dataframe()
        df_success = df[df["success"]]

        stats: Dict[str, Any] = {
            "total_trials": len(self.results),
            "successful_trials": len(successful),
            "failed_trials": len(self.results) - len(successful),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
        }

        # Score statistics
        for score in SCORE_COMPONENTS:
            if score in df_success.columns:
                stats[f"{score}_mean"] = float(df_success[score].mean())
                stats[f"{score}_std"] = float(df_success[score].std())
                stats[f"{score}_min"] = float(df_success[score].min())
                stats[f"{score}_max"] = float(df_success[score].max())

        # Cluster count statistics
        if "metric_num_clusters" in df_success.columns:
            stats["num_clusters_mean"] = float(df_success["metric_num_clusters"].mean())
            stats["num_clusters_std"] = float(df_success["metric_num_clusters"].std())
            stats["num_clusters_min"] = int(df_success["metric_num_clusters"].min())
            stats["num_clusters_max"] = int(df_success["metric_num_clusters"].max())

        return stats

    def compute_hyperparameter_importance(self) -> Dict[str, float]:
        """Compute hyperparameter importance using Spearman correlation.

        Returns:
            Dictionary mapping hyperparameter names to correlation coefficients
            with composite_score.
        """
        df = self.to_dataframe()
        df_success = df[df["success"]]

        if len(df_success) < 3:
            logger.warning("Not enough data for correlation analysis")
            return {}

        importance: Dict[str, float] = {}
        target = df_success["composite_score"]

        for hp in HYPERPARAMS:
            col = f"hp_{hp}"
            if col in df_success.columns:
                # Spearman correlation for non-linear relationships
                correlation = df_success[col].corr(target, method="spearman")
                importance[hp] = float(correlation) if pd.notna(correlation) else 0.0

        # Sort by absolute importance
        importance = dict(
            sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        return importance

    def compute_pareto_front(
        self,
        objectives: Optional[List[str]] = None,
    ) -> List[SweepResult]:
        """Compute Pareto-optimal configurations.

        A configuration is Pareto-optimal if no other configuration is better
        in all objectives.

        Args:
            objectives: List of score components to consider (default: all).

        Returns:
            List of Pareto-optimal SweepResult objects.
        """
        objectives = objectives or [
            "cluster_score",
            "activation_score",
            "dispersion_score",
        ]
        successful = self.get_successful_results()

        if not successful:
            return []

        # Extract objective values
        points = []
        for r in successful:
            point = [r.scores.get(obj, 0.0) for obj in objectives]
            points.append((point, r))

        # Find Pareto front
        pareto_front: List[SweepResult] = []
        for i, (point_i, result_i) in enumerate(points):
            is_dominated = False
            for j, (point_j, _) in enumerate(points):
                if i == j:
                    continue
                # Check if point_j dominates point_i
                # (all objectives >= and at least one >)
                if all(pj >= pi for pj, pi in zip(point_j, point_i)) and any(
                    pj > pi for pj, pi in zip(point_j, point_i)
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(result_i)

        return pareto_front

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def plot_composite_score_distribution(
        self, output_path: Optional[Path] = None, show: bool = False
    ) -> Optional[Path]:
        """Plot histogram of composite scores.

        Args:
            output_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if only shown.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return None

        successful = self.get_successful_results()
        if not successful:
            logger.warning("No successful results to plot")
            return None

        scores = [r.composite_score for r in successful]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(scores, bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Composite Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Composite Scores")
        ax.axvline(
            np.mean(scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(scores):.3f}",
        )
        ax.axvline(
            np.max(scores),
            color="green",
            linestyle="--",
            label=f"Max: {np.max(scores):.3f}",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def plot_hyperparameter_importance(
        self, output_path: Optional[Path] = None, show: bool = False
    ) -> Optional[Path]:
        """Plot hyperparameter importance bar chart.

        Args:
            output_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if only shown.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return None

        importance = self.compute_hyperparameter_importance()
        if not importance:
            logger.warning("No importance data to plot")
            return None

        names = list(importance.keys())
        values = list(importance.values())
        colors = ["green" if v > 0 else "red" for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Spearman Correlation with Composite Score")
        ax.set_ylabel("Hyperparameter")
        ax.set_title("Hyperparameter Importance")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, values):
            x_pos = val + 0.02 if val > 0 else val - 0.02
            ha = "left" if val > 0 else "right"
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha=ha,
            )

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def plot_hyperparameter_vs_score(
        self, output_path: Optional[Path] = None, show: bool = False
    ) -> Optional[Path]:
        """Plot scatter plots of each hyperparameter vs composite score.

        Args:
            output_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if only shown.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return None

        df = self.to_dataframe()
        df_success = df[df["success"]]

        if len(df_success) < 3:
            logger.warning("Not enough data for scatter plots")
            return None

        # Determine grid size
        n_params = len(HYPERPARAMS)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = (
            axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()
        )

        for i, hp in enumerate(HYPERPARAMS):
            ax = axes[i]
            col = f"hp_{hp}"
            if col in df_success.columns:
                ax.scatter(
                    df_success[col],
                    df_success["composite_score"],
                    alpha=0.6,
                    edgecolors="black",
                    linewidths=0.5,
                )
                ax.set_xlabel(hp)
                ax.set_ylabel("Composite Score")
                ax.set_title(f"{hp} vs Composite Score")
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    def plot_cluster_count_distribution(
        self, output_path: Optional[Path] = None, show: bool = False
    ) -> Optional[Path]:
        """Plot distribution of cluster counts.

        Args:
            output_path: Path to save the figure.
            show: Whether to display the figure.

        Returns:
            Path to saved figure, or None if only shown.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return None

        successful = self.get_successful_results()
        if not successful:
            logger.warning("No successful results to plot")
            return None

        clusters = [r.num_clusters for r in successful]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(clusters, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Cluster Counts")

        # Mark target range
        ax.axvline(50, color="green", linestyle="--", label="Target Min (50)")
        ax.axvline(200, color="green", linestyle="--", label="Target Max (200)")
        ax.axvspan(50, 200, alpha=0.2, color="green", label="Target Range")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {output_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return output_path

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_to_csv(self, output_path: Path) -> Path:
        """Export results to CSV file.

        Args:
            output_path: Path for the CSV file.

        Returns:
            Path to the saved CSV file.
        """
        df = self.to_dataframe()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported results to CSV: {output_path}")
        return output_path

    def export_to_json(self, output_path: Path) -> Path:
        """Export results to JSON file.

        Args:
            output_path: Path for the JSON file.

        Returns:
            Path to the saved JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "statistics": self.compute_statistics(),
            "trials": [r.to_dict() for r in self.results],
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported results to JSON: {output_path}")
        return output_path

    def export_top_configs(self, output_path: Path, n: int = 10) -> Path:
        """Export top N configurations to JSON.

        Args:
            output_path: Path for the JSON file.
            n: Number of top configurations to export.

        Returns:
            Path to the saved JSON file.
        """
        top_configs = self.get_top_configs(n)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "description": f"Top {len(top_configs)} GHSOM configurations by composite score",
            "configs": [
                {
                    "rank": i + 1,
                    "trial_id": r.trial_id,
                    "composite_score": r.composite_score,
                    "num_clusters": r.num_clusters,
                    "hyperparameters": r.hyperparams,
                    "all_scores": r.scores,
                    "all_metrics": r.metrics,
                }
                for i, r in enumerate(top_configs)
            ],
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported top {len(top_configs)} configs to: {output_path}")
        return output_path

    def generate_report(self, output_path: Path) -> Path:
        """Generate a comprehensive markdown report.

        Args:
            output_path: Path for the markdown file.

        Returns:
            Path to the saved report file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.compute_statistics()
        importance = self.compute_hyperparameter_importance()
        top_5 = self.get_top_configs(5)
        pareto = self.compute_pareto_front()

        lines = [
            "# GHSOM Sweep Analysis Report",
            "",
            "## Summary Statistics",
            "",
            f"- **Total Trials:** {stats.get('total_trials', 0)}",
            f"- **Successful Trials:** {stats.get('successful_trials', 0)}",
            f"- **Failed Trials:** {stats.get('failed_trials', 0)}",
            f"- **Success Rate:** {stats.get('success_rate', 0):.1%}",
            "",
            "### Composite Score",
            "",
            f"- Mean: {stats.get('composite_score_mean', 0):.4f}",
            f"- Std: {stats.get('composite_score_std', 0):.4f}",
            f"- Min: {stats.get('composite_score_min', 0):.4f}",
            f"- Max: {stats.get('composite_score_max', 0):.4f}",
            "",
            "### Cluster Count",
            "",
            f"- Mean: {stats.get('num_clusters_mean', 0):.1f}",
            f"- Std: {stats.get('num_clusters_std', 0):.1f}",
            f"- Min: {stats.get('num_clusters_min', 0)}",
            f"- Max: {stats.get('num_clusters_max', 0)}",
            "",
            "## Hyperparameter Importance",
            "",
            "Spearman correlation with composite score:",
            "",
            "| Hyperparameter | Correlation |",
            "|----------------|-------------|",
        ]

        for hp, corr in importance.items():
            lines.append(f"| {hp} | {corr:+.4f} |")

        lines.extend(
            [
                "",
                "## Top 5 Configurations",
                "",
            ]
        )

        for i, r in enumerate(top_5):
            lines.extend(
                [
                    f"### #{i + 1}: {r.trial_id}",
                    "",
                    f"- **Composite Score:** {r.composite_score:.4f}",
                    f"- **Cluster Count:** {r.num_clusters}",
                    f"- **Cluster Score:** {r.scores.get('cluster_score', 0):.4f}",
                    f"- **Activation Score:** {r.scores.get('activation_score', 0):.4f}",
                    f"- **Dispersion Score:** {r.scores.get('dispersion_score', 0):.4f}",
                    "",
                    "**Hyperparameters:**",
                    "",
                    "```yaml",
                ]
            )
            for hp, val in r.hyperparams.items():
                lines.append(f"{hp}: {val}")
            lines.extend(["```", ""])

        lines.extend(
            [
                "## Pareto Front",
                "",
                f"Number of Pareto-optimal configurations: {len(pareto)}",
                "",
            ]
        )

        if pareto:
            lines.extend(
                [
                    "| Trial | Composite | Clusters | Cluster Score | Activation Score | Dispersion Score |",
                    "|-------|-----------|----------|---------------|------------------|------------------|",
                ]
            )
            for r in sorted(pareto, key=lambda x: x.composite_score, reverse=True)[:10]:
                lines.append(
                    f"| {r.trial_id} | {r.composite_score:.4f} | {r.num_clusters} | "
                    f"{r.scores.get('cluster_score', 0):.4f} | "
                    f"{r.scores.get('activation_score', 0):.4f} | "
                    f"{r.scores.get('dispersion_score', 0):.4f} |"
                )

        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated report: {output_path}")
        return output_path


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze GHSOM sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing sweep results",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for analysis results (default: <input>/analysis)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        dest="top",
        help="Number of top configurations to export (default: 10)",
    )

    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Generate all visualization plots",
    )

    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only errors)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the analyzer."""
    args = _parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Determine output directory
    input_path = Path(args.input)
    output_path = args.output or (input_path / "analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and analyze results
    analyzer = SweepAnalyzer()
    results = analyzer.load_from_directory(input_path)

    if not results:
        logger.error("No results found to analyze")
        return 1

    # Compute and print statistics
    stats = analyzer.compute_statistics()
    print("\n" + "=" * 60)
    print("GHSOM SWEEP ANALYSIS")
    print("=" * 60)
    print(f"Total trials: {stats.get('total_trials', 0)}")
    print(f"Successful: {stats.get('successful_trials', 0)}")
    print(f"Success rate: {stats.get('success_rate', 0):.1%}")
    print(
        f"Composite score: {stats.get('composite_score_mean', 0):.4f} "
        f"(± {stats.get('composite_score_std', 0):.4f})"
    )
    print(f"Best score: {stats.get('composite_score_max', 0):.4f}")
    print(
        f"Cluster count: {stats.get('num_clusters_mean', 0):.1f} "
        f"[{stats.get('num_clusters_min', 0)}-{stats.get('num_clusters_max', 0)}]"
    )
    print("=" * 60 + "\n")

    # Print hyperparameter importance
    importance = analyzer.compute_hyperparameter_importance()
    if importance:
        print("Hyperparameter Importance (Spearman correlation):")
        for hp, corr in importance.items():
            sign = "+" if corr > 0 else ""
            print(f"  {hp:20s}: {sign}{corr:.4f}")
        print()

    # Print top configurations
    top_configs = analyzer.get_top_configs(min(args.top, 5))
    if top_configs:
        print(f"Top {len(top_configs)} Configurations:")
        for i, r in enumerate(top_configs):
            print(
                f"  #{i + 1}: score={r.composite_score:.4f}, "
                f"clusters={r.num_clusters}, "
                f"t1={r.hyperparams.get('t1', 0):.3f}, "
                f"t2={r.hyperparams.get('t2', 0):.3f}"
            )
        print()

    # Generate outputs
    analyzer.generate_report(output_path / "report.md")
    analyzer.export_top_configs(output_path / "top_configs.json", n=args.top)
    analyzer.export_to_json(output_path / "full_results.json")

    if args.export_csv:
        analyzer.export_to_csv(output_path / "results.csv")

    if args.plot_all:
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        analyzer.plot_composite_score_distribution(
            plots_dir / "composite_score_dist.png"
        )
        analyzer.plot_hyperparameter_importance(
            plots_dir / "hyperparameter_importance.png"
        )
        analyzer.plot_hyperparameter_vs_score(plots_dir / "hyperparameter_scatter.png")
        analyzer.plot_cluster_count_distribution(plots_dir / "cluster_count_dist.png")
        print(f"Plots saved to: {plots_dir}")

    print(f"Analysis complete. Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
