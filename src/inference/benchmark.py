#!/usr/bin/env python3
"""Benchmark tools for comparative evaluation of trained agents.

This module provides tools for running benchmarks across multiple
checkpoints, computing statistical significance tests, and generating
comparison results for paper submission.

Features:
    - BenchmarkResult dataclass for standardized metrics
    - Cross-checkpoint comparison
    - Statistical significance testing (t-test, Mann-Whitney)
    - Effect size computation (Cohen's d)
    - Publication-ready visualizations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from src.inference.sequence_analysis import SequenceAnalyzer, DiversityMetrics
from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    # Identification
    checkpoint_name: str
    checkpoint_path: str
    algorithm: str
    training_steps: int

    # Generation metrics
    num_sequences: int
    avg_episode_reward: float
    std_episode_reward: float
    min_reward: float
    max_reward: float

    # Diversity metrics
    unique_clusters_used: int
    unique_per_sequence: float
    unique_per_sequence_std: float
    repetition_ratio: float
    entropy: float
    coverage_ratio: float

    # Structural reward components
    avg_structure_score: float
    avg_transition_score: float
    avg_diversity_score: float

    # Optional: human feedback
    human_rating_mean: Optional[float] = None
    human_rating_std: Optional[float] = None
    human_rating_count: int = 0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_time_total: float = 0.0
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StatisticalComparison:
    """Statistical comparison between two benchmark results."""

    metric_name: str
    baseline_name: str
    comparison_name: str
    baseline_mean: float
    baseline_std: float
    comparison_mean: float
    comparison_std: float
    test_type: str  # "t-test" or "Mann-Whitney U"
    test_statistic: float
    p_value: float
    cohens_d: float
    significant: bool  # p < 0.05
    effect_size_interpretation: str  # "small", "medium", "large"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkRunner:
    """Runner for benchmark experiments across multiple checkpoints.

    Example:
        >>> runner = BenchmarkRunner(config)
        >>> runner.add_checkpoint("DRQN", "artifacts/training/drqn/final.pth")
        >>> runner.add_checkpoint("Rainbow", "artifacts/training/rainbow/final.pth")
        >>> results = runner.run_all(num_sequences=100)
        >>> runner.generate_comparison_report("outputs/benchmark/")
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize benchmark runner.

        Args:
            base_config: Base configuration dictionary.
            output_dir: Directory for benchmark outputs.
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/benchmark")
        self.checkpoints: List[Tuple[str, str, Optional[str]]] = (
            []
        )  # (name, path, config_path)
        self.results: List[BenchmarkResult] = []
        self.raw_data: Dict[str, List[Dict]] = {}  # Per-sequence data for stats

    def add_checkpoint(
        self,
        name: str,
        checkpoint_path: str,
        config_path: Optional[str] = None,
    ) -> None:
        """Add checkpoint to benchmark.

        Args:
            name: Display name for checkpoint.
            checkpoint_path: Path to checkpoint file.
            config_path: Optional config override.
        """
        self.checkpoints.append((name, checkpoint_path, config_path))
        logger.info(f"Added checkpoint: {name} ({checkpoint_path})")

    def load_checkpoints_from_config(
        self,
        benchmark_config: Union[str, Path, Dict],
    ) -> int:
        """Load checkpoint definitions from benchmark config.

        Args:
            benchmark_config: Path to benchmark YAML or config dict.

        Returns:
            Number of checkpoints loaded.
        """
        if isinstance(benchmark_config, (str, Path)):
            path = Path(benchmark_config)
            if not path.exists():
                raise FileNotFoundError(f"Config not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = benchmark_config

        checkpoints = config.get("checkpoints", [])
        for cp in checkpoints:
            self.add_checkpoint(
                name=cp.get("name", "Unknown"),
                checkpoint_path=cp.get("path", ""),
                config_path=cp.get("config"),
            )

        return len(checkpoints)

    def run_single_benchmark(
        self,
        name: str,
        checkpoint_path: str,
        config_path: Optional[str],
        num_sequences: int = 100,
        deterministic: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark on a single checkpoint.

        Args:
            name: Checkpoint display name.
            checkpoint_path: Path to checkpoint.
            config_path: Optional config override.
            num_sequences: Number of sequences to generate.
            deterministic: Use greedy policy.

        Returns:
            BenchmarkResult with metrics.
        """
        import time

        # Load checkpoint-specific config if provided
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        else:
            config = self.base_config.copy()

        # Import here to avoid circular imports
        from scripts.inference.run_inference import (
            setup_environment,
            load_agent,
            generate_sequence,
            load_checkpoint_config,
            detect_action_space_from_checkpoint,
        )

        start_time = time.time()

        # Load checkpoint config for architecture matching
        checkpoint_config = load_checkpoint_config(checkpoint_path)

        # Merge configs
        if checkpoint_config:
            if "network" in checkpoint_config:
                config["network"] = checkpoint_config["network"]
            if "algorithm" in checkpoint_config:
                config["algorithm"] = checkpoint_config["algorithm"]
            if "use_feature_observations" in checkpoint_config:
                config["use_feature_observations"] = checkpoint_config[
                    "use_feature_observations"
                ]

        # Detect action space
        action_space_size = detect_action_space_from_checkpoint(checkpoint_path)

        # Setup environment and load agent
        env, ghsom_manager = setup_environment(config, action_space_size)
        trainer = load_agent(checkpoint_path, env, config)

        # Generate sequences
        sequences = []
        rewards = []
        reward_components_list = []

        for _ in range(num_sequences):
            result = generate_sequence(trainer, env, deterministic)
            sequences.append(result.sequence)
            rewards.append(result.episode_reward)
            reward_components_list.append(result.reward_components)

        generation_time = time.time() - start_time

        # Analyze sequences
        analyzer = SequenceAnalyzer(ghsom_manager)
        analyzer.sequences = sequences
        diversity = analyzer.calculate_diversity_metrics()

        # Aggregate reward components
        avg_structure = np.mean([r.get("structure", 0) for r in reward_components_list])
        avg_transition = np.mean(
            [r.get("transition", 0) for r in reward_components_list]
        )
        avg_diversity_reward = np.mean(
            [r.get("diversity", 0) for r in reward_components_list]
        )

        # Get training steps from checkpoint
        try:
            training_steps = trainer.num_timesteps
        except AttributeError:
            training_steps = 0

        # Store raw data for statistical tests
        self.raw_data[name] = [
            {
                "sequence": seq,
                "reward": rew,
                "components": comp,
            }
            for seq, rew, comp in zip(sequences, rewards, reward_components_list)
        ]

        return BenchmarkResult(
            checkpoint_name=name,
            checkpoint_path=checkpoint_path,
            algorithm=config.get("algorithm", {}).get("type", "unknown"),
            training_steps=training_steps,
            num_sequences=num_sequences,
            avg_episode_reward=float(np.mean(rewards)),
            std_episode_reward=float(np.std(rewards)),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
            unique_clusters_used=diversity.unique_clusters_total,
            unique_per_sequence=diversity.unique_per_sequence_mean,
            unique_per_sequence_std=diversity.unique_per_sequence_std,
            repetition_ratio=diversity.repetition_ratio_mean,
            entropy=diversity.entropy,
            coverage_ratio=diversity.coverage_ratio,
            avg_structure_score=float(avg_structure),
            avg_transition_score=float(avg_transition),
            avg_diversity_score=float(avg_diversity_reward),
            generation_time_total=generation_time,
        )

    def run_all(
        self,
        num_sequences: int = 100,
        deterministic: bool = True,
    ) -> List[BenchmarkResult]:
        """Run benchmark on all registered checkpoints.

        Args:
            num_sequences: Sequences per checkpoint.
            deterministic: Use greedy policy.

        Returns:
            List of BenchmarkResult objects.
        """
        self.results = []

        for name, path, config_path in self.checkpoints:
            logger.info(f"Benchmarking: {name}")
            try:
                result = self.run_single_benchmark(
                    name,
                    path,
                    config_path,
                    num_sequences=num_sequences,
                    deterministic=deterministic,
                )
                self.results.append(result)
                logger.info(f"  ✓ {name}: reward={result.avg_episode_reward:.3f}")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")

        return self.results

    def compare_statistical(
        self,
        baseline_name: str,
        comparison_name: str,
        metric: str = "reward",
    ) -> StatisticalComparison:
        """Perform statistical comparison between two checkpoints.

        Args:
            baseline_name: Name of baseline checkpoint.
            comparison_name: Name of comparison checkpoint.
            metric: Metric to compare ("reward", "diversity", etc.).

        Returns:
            StatisticalComparison with test results.
        """
        from scipy import stats

        # Get raw data
        if baseline_name not in self.raw_data:
            raise ValueError(f"No data for: {baseline_name}")
        if comparison_name not in self.raw_data:
            raise ValueError(f"No data for: {comparison_name}")

        # Extract metric values
        if metric == "reward":
            baseline_values = [d["reward"] for d in self.raw_data[baseline_name]]
            comparison_values = [d["reward"] for d in self.raw_data[comparison_name]]
        elif metric == "diversity":
            baseline_values = [
                len(set(d["sequence"])) for d in self.raw_data[baseline_name]
            ]
            comparison_values = [
                len(set(d["sequence"])) for d in self.raw_data[comparison_name]
            ]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        baseline_values = np.array(baseline_values)
        comparison_values = np.array(comparison_values)

        # Normality test
        _, p_norm_a = stats.shapiro(baseline_values[:50])  # Limit for shapiro
        _, p_norm_b = stats.shapiro(comparison_values[:50])

        if p_norm_a > 0.05 and p_norm_b > 0.05:
            # Use t-test for normal distributions
            stat, p_value = stats.ttest_ind(baseline_values, comparison_values)
            test_type = "t-test"
        else:
            # Use Mann-Whitney U for non-normal
            stat, p_value = stats.mannwhitneyu(baseline_values, comparison_values)
            test_type = "Mann-Whitney U"

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(baseline_values) - 1) * np.var(baseline_values)
                + (len(comparison_values) - 1) * np.var(comparison_values)
            )
            / (len(baseline_values) + len(comparison_values) - 2)
        )
        cohens_d = (
            (np.mean(comparison_values) - np.mean(baseline_values)) / pooled_std
            if pooled_std > 0
            else 0
        )

        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_interpretation = "small"
        elif abs_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        return StatisticalComparison(
            metric_name=metric,
            baseline_name=baseline_name,
            comparison_name=comparison_name,
            baseline_mean=float(np.mean(baseline_values)),
            baseline_std=float(np.std(baseline_values)),
            comparison_mean=float(np.mean(comparison_values)),
            comparison_std=float(np.std(comparison_values)),
            test_type=test_type,
            test_statistic=float(stat),
            p_value=float(p_value),
            cohens_d=float(cohens_d),
            significant=p_value < 0.05,
            effect_size_interpretation=effect_interpretation,
        )

    def compare_all_pairs(
        self,
        metrics: List[str] = ["reward", "diversity"],
    ) -> pd.DataFrame:
        """Compare all checkpoint pairs statistically.

        Args:
            metrics: List of metrics to compare.

        Returns:
            DataFrame with comparison results.
        """
        comparisons = []
        names = list(self.raw_data.keys())

        for metric in metrics:
            for i, baseline in enumerate(names):
                for comparison in names[i + 1 :]:
                    try:
                        result = self.compare_statistical(baseline, comparison, metric)
                        comparisons.append(result.to_dict())
                    except Exception as e:
                        logger.warning(
                            f"Comparison failed ({baseline} vs {comparison}): {e}"
                        )

        return pd.DataFrame(comparisons)

    def generate_comparison_report(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate comprehensive benchmark comparison report.

        Args:
            output_dir: Output directory (uses self.output_dir if None).

        Returns:
            Path to report directory.
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        results_df = pd.DataFrame([r.to_dict() for r in self.results])
        results_df.to_csv(output_dir / "benchmark_results.csv", index=False)

        # Save JSON
        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

        # Statistical comparisons
        if len(self.results) > 1:
            comparisons_df = self.compare_all_pairs()
            comparisons_df.to_csv(
                output_dir / "statistical_comparisons.csv", index=False
            )

        # Generate visualizations
        self._generate_benchmark_visualizations(output_dir)

        # Generate markdown report
        self._generate_markdown_report(output_dir / "benchmark_report.md")

        logger.info(f"Benchmark report saved to: {output_dir}")
        return output_dir

    def _generate_benchmark_visualizations(self, output_dir: Path) -> None:
        """Generate benchmark comparison visualizations."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            logger.warning("matplotlib not available")
            return

        if not self.results:
            return

        plt.style.use("seaborn-v0_8-paper")

        # 1. Reward comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r.checkpoint_name for r in self.results]
        rewards = [r.avg_episode_reward for r in self.results]
        stds = [r.std_episode_reward for r in self.results]

        x = range(len(names))
        bars = ax.bar(
            x,
            rewards,
            yerr=stds,
            capsize=5,
            color="steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Average Episode Reward")
        ax.set_title("Benchmark: Episode Reward Comparison")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "reward_comparison.png", dpi=150)
        plt.savefig(output_dir / "reward_comparison.pdf", dpi=300)
        plt.close()

        # 2. Diversity metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # Unique clusters
        ax1 = axes[0]
        unique = [r.unique_per_sequence for r in self.results]
        ax1.bar(x, unique, color="seagreen", edgecolor="black", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Unique Clusters per Sequence")
        ax1.set_title("Diversity: Unique Clusters")

        # Entropy
        ax2 = axes[1]
        entropy = [r.entropy for r in self.results]
        ax2.bar(x, entropy, color="coral", edgecolor="black", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Diversity: Entropy")

        # Coverage
        ax3 = axes[2]
        coverage = [r.coverage_ratio * 100 for r in self.results]
        ax3.bar(x, coverage, color="mediumpurple", edgecolor="black", alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha="right")
        ax3.set_ylabel("Coverage (%)")
        ax3.set_title("Diversity: Cluster Coverage")

        plt.tight_layout()
        plt.savefig(output_dir / "diversity_comparison.png", dpi=150)
        plt.close()

        # 3. Reward component breakdown
        fig, ax = plt.subplots(figsize=(10, 6))

        width = 0.25
        x_arr = np.arange(len(names))

        structure = [r.avg_structure_score for r in self.results]
        transition = [r.avg_transition_score for r in self.results]
        diversity_s = [r.avg_diversity_score for r in self.results]

        ax.bar(x_arr - width, structure, width, label="Structure", color="steelblue")
        ax.bar(x_arr, transition, width, label="Transition", color="coral")
        ax.bar(x_arr + width, diversity_s, width, label="Diversity", color="seagreen")

        ax.set_xticks(x_arr)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Reward Component Breakdown")
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "reward_components.png", dpi=150)
        plt.close()

    def _generate_markdown_report(self, output_path: Path) -> None:
        """Generate markdown benchmark report."""
        if not self.results:
            return

        # Build results table
        table_rows = []
        for r in self.results:
            table_rows.append(
                f"| {r.checkpoint_name} | {r.avg_episode_reward:.3f} ± {r.std_episode_reward:.3f} | "
                f"{r.avg_structure_score:.3f} | {r.avg_transition_score:.3f} | "
                f"{r.avg_diversity_score:.3f} | {r.entropy:.3f} | "
                f"{'-' if r.human_rating_mean is None else f'{r.human_rating_mean:.2f}'} |"
            )

        results_table = "\n".join(table_rows)

        report = f"""# Benchmark Comparison Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration

- Sequences per checkpoint: {self.results[0].num_sequences if self.results else 'N/A'}
- Checkpoints evaluated: {len(self.results)}

## Main Results

| Model | Reward ↑ | Structure ↑ | Transition ↑ | Diversity ↑ | Entropy ↑ | Human ↑ |
|-------|----------|-------------|--------------|-------------|-----------|---------|
{results_table}

## Best Performers

- **Highest Reward**: {max(self.results, key=lambda x: x.avg_episode_reward).checkpoint_name}
- **Highest Diversity**: {max(self.results, key=lambda x: x.unique_per_sequence).checkpoint_name}
- **Highest Entropy**: {max(self.results, key=lambda x: x.entropy).checkpoint_name}

## Visualizations

- [Reward Comparison](reward_comparison.png)
- [Diversity Comparison](diversity_comparison.png)
- [Reward Components](reward_components.png)

## Data Files

- `benchmark_results.csv` - Full results table
- `benchmark_results.json` - JSON format results
- `statistical_comparisons.csv` - Pairwise statistical tests

"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)


def load_benchmark_results(
    results_path: Union[str, Path],
) -> List[BenchmarkResult]:
    """Load benchmark results from file.

    Args:
        results_path: Path to results JSON or CSV.

    Returns:
        List of BenchmarkResult objects.
    """
    path = Path(results_path)

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return [BenchmarkResult.from_dict(d) for d in data]

    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        return [BenchmarkResult.from_dict(row.to_dict()) for _, row in df.iterrows()]

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
