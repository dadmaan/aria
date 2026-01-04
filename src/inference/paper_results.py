#!/usr/bin/env python3
"""Paper Results Generation for NIPS/ICLR Submission.

This module provides tools for generating publication-quality results
including LaTeX tables, figures, and formatted reports.

Features:
    - LaTeX table generation with automatic formatting
    - Publication-ready figure generation
    - Statistical significance annotations
    - User study protocol support
    - Report compilation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.inference.benchmark import BenchmarkResult, load_benchmark_results
from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class TableConfig:
    """Configuration for LaTeX table generation."""

    caption: str = "Generation quality comparison across models."
    label: str = "tab:main_results"
    float_format: str = ".2f"
    bold_best: bool = True
    show_std: bool = True
    significance_level: float = 0.05


class LaTeXTableGenerator:
    """Generator for publication-ready LaTeX tables."""

    def __init__(
        self,
        results: List[BenchmarkResult],
        config: Optional[TableConfig] = None,
    ):
        """Initialize LaTeX table generator.

        Args:
            results: List of benchmark results.
            config: Table configuration.
        """
        self.results = results
        self.config = config or TableConfig()
        self.df = pd.DataFrame([r.to_dict() for r in results])

    def generate_main_results_table(
        self,
        columns: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """Generate main results comparison table.

        Args:
            columns: List of (column_name, header) tuples to include.

        Returns:
            LaTeX table string.
        """
        if columns is None:
            columns = [
                ("avg_episode_reward", "Reward ↑"),
                ("avg_structure_score", "Structure ↑"),
                ("avg_transition_score", "Transition ↑"),
                ("avg_diversity_score", "Diversity ↑"),
                ("entropy", "Entropy ↑"),
            ]

        # Find best values for each column
        best_values = {}
        for col, _ in columns:
            if col in self.df.columns:
                best_values[col] = self.df[col].max()

        # Build LaTeX table
        latex = (
            r"""
\begin{table}[t]
\centering
\caption{"""
            + self.config.caption
            + r"""}
\label{"""
            + self.config.label
            + r"""}
\begin{tabular}{l"""
            + "c" * len(columns)
            + r"""}
\toprule
Model"""
        )

        # Header row
        for _, header in columns:
            latex += f" & {header}"
        latex += r" \\" + "\n" + r"\midrule" + "\n"

        # Data rows
        for result in self.results:
            row = result.checkpoint_name

            for col, _ in columns:
                value = getattr(result, col, None)
                if value is None:
                    row += " & -"
                    continue

                # Format value
                if self.config.show_std and col == "avg_episode_reward":
                    std_col = "std_episode_reward"
                    std = getattr(result, std_col, 0)
                    formatted = f"{value:{self.config.float_format}} ± {std:{self.config.float_format}}"
                else:
                    formatted = f"{value:{self.config.float_format}}"

                # Bold best values
                if self.config.bold_best and col in best_values:
                    if abs(value - best_values[col]) < 1e-6:
                        formatted = r"\textbf{" + formatted + "}"

                row += f" & {formatted}"

            latex += row + r" \\" + "\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_ablation_table(
        self,
        ablation_results: Dict[str, BenchmarkResult],
    ) -> str:
        """Generate ablation study table.

        Args:
            ablation_results: Dict mapping condition name to result.

        Returns:
            LaTeX table string.
        """
        latex = r"""
\begin{table}[t]
\centering
\caption{Ablation study: Impact of Human-in-the-Loop feedback.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Condition & Reward & Human Rating & Improvement \\
\midrule
"""
        baseline_reward = None
        for name, result in ablation_results.items():
            reward = result.avg_episode_reward
            human = result.human_rating_mean if result.human_rating_mean else "-"

            if baseline_reward is None:
                baseline_reward = reward
                improvement = "-"
            else:
                improvement = f"+{((reward/baseline_reward - 1) * 100):.1f}\\%"

            latex += f"{name} & {reward:{self.config.float_format}} & {human} & {improvement} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_statistical_significance_table(
        self,
        comparisons: pd.DataFrame,
    ) -> str:
        """Generate statistical significance table.

        Args:
            comparisons: DataFrame from BenchmarkRunner.compare_all_pairs()

        Returns:
            LaTeX table string.
        """
        latex = r"""
\begin{table}[t]
\centering
\caption{Statistical significance tests between model pairs.}
\label{tab:significance}
\footnotesize
\begin{tabular}{llcccc}
\toprule
Metric & Comparison & $p$-value & Effect Size & Significant \\
\midrule
"""
        for _, row in comparisons.iterrows():
            metric = row["metric_name"]
            comparison = f"{row['baseline_name']} vs {row['comparison_name']}"
            p_val = f"{row['p_value']:.4f}"
            effect = f"{row['cohens_d']:.2f} ({row['effect_size_interpretation']})"
            sig = r"\checkmark" if row["significant"] else "-"

            latex += f"{metric} & {comparison} & {p_val} & {effect} & {sig} \\\\\n"

        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex


class FigureGenerator:
    """Generator for publication-quality figures."""

    def __init__(
        self,
        results: List[BenchmarkResult],
        output_dir: Path,
    ):
        """Initialize figure generator.

        Args:
            results: Benchmark results.
            output_dir: Output directory for figures.
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style for publication
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
            plt.style.use("seaborn-v0_8-paper")
            plt.rcParams.update(
                {
                    "font.size": 10,
                    "axes.labelsize": 10,
                    "axes.titlesize": 11,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "legend.fontsize": 9,
                    "figure.titlesize": 12,
                    "font.family": "serif",
                }
            )
        except ImportError:
            logger.warning("matplotlib not available")

    def generate_all_figures(self) -> List[Path]:
        """Generate all paper figures.

        Returns:
            List of generated figure paths.
        """
        figures = []

        # Main comparison figure
        fig_path = self.generate_main_comparison_figure()
        if fig_path:
            figures.append(fig_path)

        # Training curves (placeholder - would need actual training logs)
        # figures.append(self.generate_training_curves_figure())

        # Diversity analysis
        fig_path = self.generate_diversity_figure()
        if fig_path:
            figures.append(fig_path)

        return figures

    def generate_main_comparison_figure(self) -> Optional[Path]:
        """Generate main results comparison figure (Figure 2).

        Returns:
            Path to generated figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        names = [r.checkpoint_name for r in self.results]
        x = np.arange(len(names))

        # Left: Reward comparison with error bars
        ax1 = axes[0]
        rewards = [r.avg_episode_reward for r in self.results]
        stds = [r.std_episode_reward for r in self.results]

        bars = ax1.bar(
            x,
            rewards,
            yerr=stds,
            capsize=5,
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Average Episode Reward")
        ax1.set_title("(a) Generation Quality")
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Right: Radar chart of metrics (simplified as grouped bar)
        ax2 = axes[1]
        metrics = ["Structure", "Transition", "Diversity"]
        width = 0.8 / len(self.results)

        for i, result in enumerate(self.results):
            values = [
                result.avg_structure_score,
                result.avg_transition_score,
                result.avg_diversity_score,
            ]
            offset = (i - len(self.results) / 2 + 0.5) * width
            ax2.bar(
                np.arange(len(metrics)) + offset,
                values,
                width,
                label=result.checkpoint_name,
                alpha=0.8,
            )

        ax2.set_xticks(np.arange(len(metrics)))
        ax2.set_xticklabels(metrics)
        ax2.set_ylabel("Score")
        ax2.set_title("(b) Reward Components")
        ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        output_path = self.output_dir / "fig2_main_comparison.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".png"), dpi=150)
        plt.close()

        return output_path

    def generate_diversity_figure(self) -> Optional[Path]:
        """Generate diversity analysis figure (Figure 4).

        Returns:
            Path to generated figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        names = [r.checkpoint_name for r in self.results]
        x = np.arange(len(names))

        # Unique clusters
        ax1 = axes[0]
        unique = [r.unique_per_sequence for r in self.results]
        ax1.bar(x, unique, color="seagreen", edgecolor="black", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Unique Clusters per Sequence")
        ax1.set_title("(a) Cluster Diversity")

        # Entropy
        ax2 = axes[1]
        entropy = [r.entropy for r in self.results]
        ax2.bar(x, entropy, color="coral", edgecolor="black", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("Shannon Entropy")
        ax2.set_title("(b) Distribution Entropy")

        # Coverage
        ax3 = axes[2]
        coverage = [r.coverage_ratio * 100 for r in self.results]
        ax3.bar(x, coverage, color="mediumpurple", edgecolor="black", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha="right")
        ax3.set_ylabel("Coverage (%)")
        ax3.set_title("(c) Cluster Coverage")

        plt.tight_layout()

        output_path = self.output_dir / "fig4_diversity_analysis.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.with_suffix(".png"), dpi=150)
        plt.close()

        return output_path

    def generate_hil_figure(
        self,
        hil_results: Dict[str, Any],
    ) -> Optional[Path]:
        """Generate Human-in-the-Loop feedback figure.

        Args:
            hil_results: HIL session results.

        Returns:
            Path to generated figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: Feedback distribution
        ax1 = axes[0]
        feedbacks = hil_results.get("feedbacks", [3.0] * 10)
        ax1.hist(
            feedbacks,
            bins=5,
            range=(1, 5),
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        ax1.set_xlabel("Human Rating (1-5)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("(a) Human Feedback Distribution")

        # Right: Correlation with reward
        ax2 = axes[1]
        rewards = hil_results.get("rewards", np.random.randn(10))
        ax2.scatter(feedbacks, rewards, c="steelblue", alpha=0.6, s=50)
        ax2.set_xlabel("Human Rating")
        ax2.set_ylabel("Episode Reward")
        ax2.set_title("(b) Human vs Automated Evaluation")

        # Add correlation coefficient
        if len(feedbacks) > 2:
            corr = np.corrcoef(feedbacks, rewards)[0, 1]
            ax2.annotate(
                f"r = {corr:.2f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=10,
            )

        plt.tight_layout()

        output_path = self.output_dir / "fig5_hil_analysis.pdf"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path


class PaperResultsCompiler:
    """Compiler for all paper results and materials."""

    def __init__(
        self,
        results_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        """Initialize paper results compiler.

        Args:
            results_dir: Directory containing benchmark results.
            output_dir: Output directory for paper materials.
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BenchmarkResult] = []
        self.comparisons: Optional[pd.DataFrame] = None

    def load_results(self) -> int:
        """Load benchmark results from directory.

        Returns:
            Number of results loaded.
        """
        json_path = self.results_dir / "benchmark_results.json"
        csv_path = self.results_dir / "benchmark_results.csv"

        if json_path.exists():
            self.results = load_benchmark_results(json_path)
        elif csv_path.exists():
            self.results = load_benchmark_results(csv_path)
        else:
            raise FileNotFoundError(f"No results found in {self.results_dir}")

        # Load comparisons if available
        comp_path = self.results_dir / "statistical_comparisons.csv"
        if comp_path.exists():
            self.comparisons = pd.read_csv(comp_path)

        logger.info(f"Loaded {len(self.results)} benchmark results")
        return len(self.results)

    def compile_all(self) -> Dict[str, Path]:
        """Compile all paper materials.

        Returns:
            Dictionary mapping material type to path.
        """
        outputs = {}

        # Generate LaTeX tables
        table_gen = LaTeXTableGenerator(self.results)

        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # Main results table
        main_table = table_gen.generate_main_results_table()
        main_table_path = tables_dir / "main_results.tex"
        main_table_path.write_text(main_table)
        outputs["main_table"] = main_table_path

        # Statistical significance table
        if self.comparisons is not None:
            sig_table = table_gen.generate_statistical_significance_table(
                self.comparisons
            )
            sig_table_path = tables_dir / "significance.tex"
            sig_table_path.write_text(sig_table)
            outputs["significance_table"] = sig_table_path

        # Generate figures
        figures_dir = self.output_dir / "figures"
        fig_gen = FigureGenerator(self.results, figures_dir)
        figure_paths = fig_gen.generate_all_figures()
        outputs["figures"] = figure_paths

        # Generate summary report
        report_path = self._generate_summary_report()
        outputs["report"] = report_path

        logger.info(f"Paper materials compiled to: {self.output_dir}")
        return outputs

    def _generate_summary_report(self) -> Path:
        """Generate summary report of all results."""
        report_path = self.output_dir / "paper_results_summary.md"

        # Find best model
        best_model = max(self.results, key=lambda x: x.avg_episode_reward)

        report = f"""# Paper Results Summary

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Key Findings

1. **Best Model**: {best_model.checkpoint_name}
   - Reward: {best_model.avg_episode_reward:.3f} ± {best_model.std_episode_reward:.3f}
   - Diversity (Entropy): {best_model.entropy:.3f}
   - Coverage: {best_model.coverage_ratio:.1%}

2. **Comparison Summary**:
   - Models evaluated: {len(self.results)}
   - Sequences per model: {self.results[0].num_sequences if self.results else 'N/A'}

## Results Summary

| Model | Reward | Structure | Transition | Diversity | Entropy |
|-------|--------|-----------|------------|-----------|---------|
"""
        for r in self.results:
            report += f"| {r.checkpoint_name} | {r.avg_episode_reward:.3f} | "
            report += f"{r.avg_structure_score:.3f} | {r.avg_transition_score:.3f} | "
            report += f"{r.avg_diversity_score:.3f} | {r.entropy:.3f} |\n"

        report += """

## Generated Materials

### Tables (LaTeX)
- `tables/main_results.tex` - Main comparison table
- `tables/significance.tex` - Statistical significance tests

### Figures
- `figures/fig2_main_comparison.pdf` - Main comparison figure
- `figures/fig4_diversity_analysis.pdf` - Diversity analysis

## Usage in Paper

Include tables:
```latex
\\input{tables/main_results.tex}
\\input{tables/significance.tex}
```

Include figures:
```latex
\\includegraphics[width=\\linewidth]{figures/fig2_main_comparison.pdf}
```
"""
        report_path.write_text(report)
        return report_path


def generate_paper_results(
    benchmark_results_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Convenience function to generate all paper results.

    Args:
        benchmark_results_dir: Directory with benchmark results.
        output_dir: Output directory for paper materials.

    Returns:
        Dictionary of generated material paths.
    """
    compiler = PaperResultsCompiler(benchmark_results_dir, output_dir)
    compiler.load_results()
    return compiler.compile_all()
