#!/usr/bin/env python3
"""Learning Trajectory Visualization Script.

This script performs comprehensive post-training analysis on all benchmark runs,
replicating the analysis from notebooks/post_training_analysis.ipynb.

Analysis includes:
1. State Visitation Heatmap - Temporal snapshots of cluster exploration
2. Hierarchy Navigation - GHSOM tree structure with visitation patterns
3. Reward per Cluster - Which states yield highest rewards
4. Reward Component Evolution - How components change during training
5. Episode Reward Analysis - Overall training performance

Usage:
    python scripts/benchmark/visualize_trajectories.py \
        --runs-dir artifacts/benchmark/main/20251216_140256/runs \
        --output-dir outputs/benchmark/trajectories

    # With specific options:
    python scripts/benchmark/visualize_trajectories.py \
        --runs-dir artifacts/benchmark/main/20251216_140256/runs \
        --output-dir outputs/benchmark/trajectories \
        --skip-hierarchy \
        --parallel 4 \
        --formats png pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils.post_training import (
    TrainingDataLoader,
    TrainingRunData,
    plot_state_visitation_heatmap,
    plot_hierarchy_navigation,
    plot_reward_per_cluster,
    plot_reward_component_evolution,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)

DEFAULT_FIGURE_FORMATS = ['png']
DEFAULT_DPI = 300
FIGURE_SIZES = {
    'visitation_heatmap': (15, 4),
    'hierarchy_navigation': (12, 8),
    'reward_per_cluster': (10, 6),
    'component_evolution': (10, 6),
    'episode_rewards': (14, 5),
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RunAnalysisResult:
    """Result of analyzing a single run."""
    run_name: str
    variant: str
    seed: int
    status: str
    total_episodes: int
    total_timesteps: int
    mean_reward: float
    std_reward: float
    best_reward: float
    final_reward: float
    cumulative_reward: float  # Sum of all episode rewards
    reward_components: Dict[str, float]
    figures_generated: List[str]
    errors: List[str]
    output_dir: str


@dataclass
class BenchmarkAnalysisResult:
    """Result of analyzing all benchmark runs."""
    benchmark_dir: str
    output_dir: str
    timestamp: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    run_results: List[RunAnalysisResult]
    summary_statistics: Dict[str, Any]


# ============================================================================
# RUN DISCOVERY
# ============================================================================

def discover_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Discover all benchmark runs in the given directory.

    Expected structure:
        runs_dir/
            baseline_s42/
                run_drqn_TIMESTAMP/
                    config.yaml
                    metrics/
                        ...
            rainbow_s123/
                run_rainbow_drqn_TIMESTAMP/
                    ...

    Args:
        runs_dir: Path to runs directory

    Returns:
        List of run information dictionaries
    """
    runs = []

    if not runs_dir.exists():
        logger.error(f"Runs directory not found: {runs_dir}")
        return runs

    # Iterate over variant directories
    for variant_dir in sorted(runs_dir.iterdir()):
        if not variant_dir.is_dir():
            continue

        # Parse variant name and seed
        variant_name = variant_dir.name
        parts = variant_name.rsplit('_s', 1)

        if len(parts) == 2:
            variant = parts[0]
            try:
                seed = int(parts[1])
            except ValueError:
                seed = 0
        else:
            variant = variant_name
            seed = 0

        # Find the actual run directory inside
        run_dirs = [d for d in variant_dir.iterdir()
                    if d.is_dir() and d.name.startswith('run_')]

        if not run_dirs:
            logger.warning(f"No run directory found in {variant_dir}")
            continue

        # Use the first (or only) run directory
        run_dir = run_dirs[0]

        # Verify required files exist
        config_path = run_dir / 'config.yaml'
        metrics_dir = run_dir / 'metrics'

        if not config_path.exists():
            logger.warning(f"Config not found in {run_dir}")
            continue

        runs.append({
            'variant_name': variant_name,
            'variant': variant,
            'seed': seed,
            'run_dir': run_dir,
            'variant_dir': variant_dir,
            'config_path': config_path,
            'metrics_dir': metrics_dir,
        })

    logger.info(f"Discovered {len(runs)} runs in {runs_dir}")
    return runs


# ============================================================================
# VISUALIZATION 5: EPISODE REWARD ANALYSIS (from notebook)
# ============================================================================

def plot_episode_rewards(
    data: TrainingRunData,
    window_pct: float = 0.05,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot episode reward analysis with time series and histogram.

    Replicates the inline analysis from post_training_analysis.ipynb.

    Args:
        data: TrainingRunData object
        window_pct: Rolling window as percentage of total episodes (default 5%)
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    sns.set_style("whitegrid")

    rewards = data.episode_rewards
    n_episodes = len(rewards)

    # Adaptive window size
    window_size = max(1, int(n_episodes * window_pct))
    window_size = min(window_size, n_episodes // 10)  # Cap at 10% of episodes
    window_size = max(1, window_size)

    # Compute rolling mean
    rolling_mean = pd.Series(rewards).rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).mean()

    # Statistics
    stats = {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
    }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Time series
    episodes = np.arange(n_episodes)
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.5, label='Raw')
    ax1.plot(episodes, rolling_mean, color='red', linewidth=2,
             label=f'Rolling Mean (w={window_size})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Episode Rewards: {data.run_name}')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    # Add stats annotation
    stats_text = (
        f"Mean: {stats['mean']:.4f}\n"
        f"Std: {stats['std']:.4f}\n"
        f"Min: {stats['min']:.4f}\n"
        f"Max: {stats['max']:.4f}\n"
        f"Final 100: {stats['final_100_mean']:.4f}"
    )
    ax1.text(
        0.98, 0.02, stats_text,
        transform=ax1.transAxes,
        verticalalignment='bottom',
        horizontalalignment='right',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Right: Histogram
    ax2.hist(rewards, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.4f}")
    ax2.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2,
                label=f"Median: {np.median(rewards):.4f}")
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count')
    ax2.set_title('Reward Distribution')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    plt.suptitle(f'Episode Reward Analysis: {data.run_name}', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=DEFAULT_DPI)

    return fig


# ============================================================================
# SINGLE RUN ANALYSIS
# ============================================================================

def analyze_single_run(
    run_info: Dict[str, Any],
    output_dir: Path,
    formats: List[str],
    skip_hierarchy: bool = False,
    skip_ghsom: bool = False,
) -> RunAnalysisResult:
    """Analyze a single training run.

    Args:
        run_info: Run information dictionary from discover_runs()
        output_dir: Base output directory
        formats: List of output formats (png, pdf, etc.)
        skip_hierarchy: Skip hierarchy navigation visualization
        skip_ghsom: Skip GHSOM-dependent visualizations entirely

    Returns:
        RunAnalysisResult with analysis outcome
    """
    run_dir = run_info['run_dir']
    variant_name = run_info['variant_name']
    variant = run_info['variant']
    seed = run_info['seed']

    logger.info(f"Analyzing run: {variant_name}")

    # Create output subdirectory
    run_output_dir = output_dir / variant_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    figures_generated = []

    # Load data
    try:
        loader = TrainingDataLoader(run_dir)
        data = loader.load(load_ghsom=not skip_ghsom, load_comprehensive=False)
    except Exception as e:
        logger.error(f"Failed to load run data for {variant_name}: {e}")
        return RunAnalysisResult(
            run_name=variant_name,
            variant=variant,
            seed=seed,
            status='failed',
            total_episodes=0,
            total_timesteps=0,
            mean_reward=0.0,
            std_reward=0.0,
            best_reward=0.0,
            final_reward=0.0,
            cumulative_reward=0.0,
            reward_components={},
            figures_generated=[],
            errors=[str(e)],
            output_dir=str(run_output_dir),
        )

    # Calculate statistics
    rewards = data.episode_rewards
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    best_reward = float(np.max(rewards))
    final_reward = float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
    cumulative_reward = float(np.sum(rewards))

    # Component means
    component_means = {}
    for comp_name, comp_values in data.reward_components.items():
        component_means[comp_name] = float(np.mean(comp_values))

    # Generate visualizations
    def save_figure(fig: plt.Figure, name: str):
        """Save figure in multiple formats."""
        for fmt in formats:
            path = run_output_dir / f"{name}.{fmt}"
            fig.savefig(path, bbox_inches='tight', dpi=DEFAULT_DPI, format=fmt)
        plt.close(fig)
        figures_generated.append(name)

    # Viz 1: State Visitation Heatmap
    if data.episode_sequences and not skip_ghsom:
        try:
            fig = plot_state_visitation_heatmap(data)
            save_figure(fig, 'visitation_heatmap')
        except Exception as e:
            errors.append(f"visitation_heatmap: {e}")
            logger.warning(f"Failed to generate visitation heatmap for {variant_name}: {e}")
    else:
        logger.info(f"Skipping visitation heatmap for {variant_name} (no sequences)")

    # Viz 2: Hierarchy Navigation
    if not skip_hierarchy and data.episode_sequences and data.ghsom_manager is not None:
        try:
            fig = plot_hierarchy_navigation(data, min_visits=3)
            save_figure(fig, 'hierarchy_navigation')
        except Exception as e:
            errors.append(f"hierarchy_navigation: {e}")
            logger.warning(f"Failed to generate hierarchy navigation for {variant_name}: {e}")
    else:
        logger.info(f"Skipping hierarchy navigation for {variant_name}")

    # Viz 3: Reward per Cluster
    if data.episode_sequences:
        try:
            fig = plot_reward_per_cluster(data, top_k=15)
            save_figure(fig, 'reward_per_cluster')
        except Exception as e:
            errors.append(f"reward_per_cluster: {e}")
            logger.warning(f"Failed to generate reward per cluster for {variant_name}: {e}")
    else:
        logger.info(f"Skipping reward per cluster for {variant_name} (no sequences)")

    # Viz 4: Reward Component Evolution
    if data.reward_components:
        try:
            fig = plot_reward_component_evolution(data, window_size=50)
            save_figure(fig, 'component_evolution')
        except Exception as e:
            errors.append(f"component_evolution: {e}")
            logger.warning(f"Failed to generate component evolution for {variant_name}: {e}")

    # Viz 5: Episode Reward Analysis
    try:
        fig = plot_episode_rewards(data)
        save_figure(fig, 'episode_rewards')
    except Exception as e:
        errors.append(f"episode_rewards: {e}")
        logger.warning(f"Failed to generate episode rewards for {variant_name}: {e}")

    # Save run statistics as JSON
    run_stats = {
        'run_name': variant_name,
        'variant': variant,
        'seed': seed,
        'total_episodes': data.total_episodes,
        'total_timesteps': data.total_timesteps,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'best_reward': best_reward,
        'final_reward': final_reward,
        'cumulative_reward': cumulative_reward,
        'reward_components': component_means,
        'figures_generated': figures_generated,
        'errors': errors,
    }

    with open(run_output_dir / 'run_statistics.json', 'w') as f:
        json.dump(run_stats, f, indent=2)

    return RunAnalysisResult(
        run_name=variant_name,
        variant=variant,
        seed=seed,
        status='success' if not errors else 'partial',
        total_episodes=data.total_episodes,
        total_timesteps=data.total_timesteps,
        mean_reward=mean_reward,
        std_reward=std_reward,
        best_reward=best_reward,
        final_reward=final_reward,
        cumulative_reward=cumulative_reward,
        reward_components=component_means,
        figures_generated=figures_generated,
        errors=errors,
        output_dir=str(run_output_dir),
    )


def analyze_single_run_wrapper(args: Tuple) -> RunAnalysisResult:
    """Wrapper for parallel execution."""
    run_info, output_dir, formats, skip_hierarchy, skip_ghsom = args
    return analyze_single_run(run_info, output_dir, formats, skip_hierarchy, skip_ghsom)


# ============================================================================
# SUMMARY STATISTICS & REPORTING
# ============================================================================

def compute_summary_statistics(results: List[RunAnalysisResult]) -> Dict[str, Any]:
    """Compute summary statistics across all runs.

    Args:
        results: List of RunAnalysisResult objects

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    # Group by variant
    variant_stats = {}
    for result in results:
        variant = result.variant
        if variant not in variant_stats:
            variant_stats[variant] = {
                'runs': [],
                'mean_rewards': [],
                'best_rewards': [],
                'final_rewards': [],
                'cumulative_rewards': [],
            }
        variant_stats[variant]['runs'].append(result.run_name)
        variant_stats[variant]['mean_rewards'].append(result.mean_reward)
        variant_stats[variant]['best_rewards'].append(result.best_reward)
        variant_stats[variant]['final_rewards'].append(result.final_reward)
        variant_stats[variant]['cumulative_rewards'].append(result.cumulative_reward)

    # Compute per-variant summaries
    variant_summaries = {}
    for variant, stats in variant_stats.items():
        variant_summaries[variant] = {
            'n_runs': len(stats['runs']),
            'mean_reward': {
                'mean': float(np.mean(stats['mean_rewards'])),
                'std': float(np.std(stats['mean_rewards'])),
            },
            'best_reward': {
                'mean': float(np.mean(stats['best_rewards'])),
                'std': float(np.std(stats['best_rewards'])),
            },
            'final_reward': {
                'mean': float(np.mean(stats['final_rewards'])),
                'std': float(np.std(stats['final_rewards'])),
            },
            'cumulative_reward': {
                'mean': float(np.mean(stats['cumulative_rewards'])),
                'std': float(np.std(stats['cumulative_rewards'])),
            },
        }

    # Overall statistics
    all_mean_rewards = [r.mean_reward for r in results]
    all_best_rewards = [r.best_reward for r in results]
    all_cumulative_rewards = [r.cumulative_reward for r in results]

    return {
        'n_total_runs': len(results),
        'n_successful': sum(1 for r in results if r.status == 'success'),
        'n_partial': sum(1 for r in results if r.status == 'partial'),
        'n_failed': sum(1 for r in results if r.status == 'failed'),
        'overall': {
            'mean_reward': {
                'mean': float(np.mean(all_mean_rewards)),
                'std': float(np.std(all_mean_rewards)),
            },
            'best_reward': {
                'mean': float(np.mean(all_best_rewards)),
                'max': float(np.max(all_best_rewards)),
            },
            'cumulative_reward': {
                'mean': float(np.mean(all_cumulative_rewards)),
                'std': float(np.std(all_cumulative_rewards)),
            },
        },
        'by_variant': variant_summaries,
    }


def generate_summary_report(
    benchmark_result: BenchmarkAnalysisResult,
    output_dir: Path,
) -> Path:
    """Generate markdown summary report.

    Args:
        benchmark_result: Complete benchmark analysis result
        output_dir: Output directory

    Returns:
        Path to generated report
    """
    report_path = output_dir / 'LEARNING_TRAJECTORY_REPORT.md'

    lines = [
        "# Learning Trajectory Analysis Report",
        "",
        f"**Generated:** {benchmark_result.timestamp}",
        f"**Benchmark Directory:** `{benchmark_result.benchmark_dir}`",
        f"**Output Directory:** `{benchmark_result.output_dir}`",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Total Runs Analyzed:** {benchmark_result.total_runs}",
        f"- **Successful:** {benchmark_result.successful_runs}",
        f"- **Failed:** {benchmark_result.failed_runs}",
        "",
    ]

    # Per-variant summary table
    stats = benchmark_result.summary_statistics
    if 'by_variant' in stats:
        lines.extend([
            "## Performance by Variant",
            "",
            "| Variant | N | Mean Reward | Std | Best Reward | Final Reward | Cumulative Reward |",
            "|---------|---|-------------|-----|-------------|--------------|-------------------|",
        ])

        for variant, v_stats in stats['by_variant'].items():
            mean_r = v_stats['mean_reward']
            best_r = v_stats['best_reward']
            final_r = v_stats['final_reward']
            cum_r = v_stats.get('cumulative_reward', {'mean': 0.0})
            lines.append(
                f"| {variant} | {v_stats['n_runs']} | "
                f"{mean_r['mean']:.4f} | {mean_r['std']:.4f} | "
                f"{best_r['mean']:.4f} | {final_r['mean']:.4f} | {cum_r['mean']:.2f} |"
            )
        lines.append("")

    # Individual run details
    lines.extend([
        "## Individual Run Results",
        "",
    ])

    for result in benchmark_result.run_results:
        status_emoji = "✅" if result.status == 'success' else ("⚠️" if result.status == 'partial' else "❌")
        lines.extend([
            f"### {result.run_name} {status_emoji}",
            "",
            f"- **Variant:** {result.variant}",
            f"- **Seed:** {result.seed}",
            f"- **Episodes:** {result.total_episodes}",
            f"- **Timesteps:** {result.total_timesteps}",
            f"- **Mean Reward:** {result.mean_reward:.4f} ± {result.std_reward:.4f}",
            f"- **Best Reward:** {result.best_reward:.4f}",
            f"- **Final Reward (last 100):** {result.final_reward:.4f}",
            f"- **Cumulative Reward:** {result.cumulative_reward:.2f}",
            "",
        ])

        if result.reward_components:
            lines.append("**Reward Components (mean):**")
            for comp, val in result.reward_components.items():
                lines.append(f"  - {comp}: {val:.4f}")
            lines.append("")

        if result.figures_generated:
            lines.append(f"**Figures Generated:** {', '.join(result.figures_generated)}")
            lines.append("")

        if result.errors:
            lines.append("**Errors:**")
            for err in result.errors:
                lines.append(f"  - {err}")
            lines.append("")

    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return report_path


def generate_comparison_figures(
    results: List[RunAnalysisResult],
    output_dir: Path,
    formats: List[str],
) -> List[str]:
    """Generate comparison figures across all runs.

    Args:
        results: List of RunAnalysisResult objects
        output_dir: Output directory
        formats: Output formats

    Returns:
        List of generated figure names
    """
    generated = []

    if len(results) < 2:
        return generated

    sns.set_style("whitegrid")

    # Prepare dataframe
    df = pd.DataFrame([
        {
            'variant': r.variant,
            'seed': r.seed,
            'mean_reward': r.mean_reward,
            'std_reward': r.std_reward,
            'best_reward': r.best_reward,
            'final_reward': r.final_reward,
            'cumulative_reward': r.cumulative_reward,
        }
        for r in results if r.status != 'failed'
    ])

    if df.empty:
        return generated

    # Figure 1: Mean reward comparison by variant
    fig, ax = plt.subplots(figsize=(10, 6))

    variant_order = sorted(df['variant'].unique())
    colors = sns.color_palette('husl', len(variant_order))

    x = np.arange(len(variant_order))
    width = 0.25

    for i, (variant, color) in enumerate(zip(variant_order, colors)):
        variant_df = df[df['variant'] == variant]
        mean_val = variant_df['mean_reward'].mean()
        std_val = variant_df['mean_reward'].std()

        ax.bar(i, mean_val, width, yerr=std_val, color=color,
               label=variant, capsize=5, alpha=0.8)

        # Plot individual seeds
        for j, (_, row) in enumerate(variant_df.iterrows()):
            ax.scatter(i + (j - len(variant_df)/2) * 0.1, row['mean_reward'],
                      color='black', s=30, zorder=5)

    ax.set_xlabel('Variant')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Mean Reward Comparison Across Variants')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_order, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    for fmt in formats:
        fig.savefig(output_dir / f'comparison_mean_reward.{fmt}',
                   bbox_inches='tight', dpi=DEFAULT_DPI)
    plt.close(fig)
    generated.append('comparison_mean_reward')

    # Figure 2: Box plot of rewards by variant
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for box plot
    box_data = [df[df['variant'] == v]['mean_reward'].values for v in variant_order]

    bp = ax.boxplot(box_data, labels=variant_order, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Variant')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('Reward Distribution by Variant')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for fmt in formats:
        fig.savefig(output_dir / f'comparison_boxplot.{fmt}',
                   bbox_inches='tight', dpi=DEFAULT_DPI)
    plt.close(fig)
    generated.append('comparison_boxplot')

    # Figure 3: Final reward heatmap (variant x seed)
    if len(df['seed'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))

        pivot_df = df.pivot(index='variant', columns='seed', values='final_reward')

        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=ax, cbar_kws={'label': 'Final Reward'})

        ax.set_title('Final Reward Heatmap (Variant × Seed)')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Variant')

        plt.tight_layout()

        for fmt in formats:
            fig.savefig(output_dir / f'comparison_heatmap.{fmt}',
                       bbox_inches='tight', dpi=DEFAULT_DPI)
        plt.close(fig)
        generated.append('comparison_heatmap')

    # Figure 4: Cumulative reward bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (variant, color) in enumerate(zip(variant_order, colors)):
        variant_df = df[df['variant'] == variant]
        mean_val = variant_df['cumulative_reward'].mean()
        std_val = variant_df['cumulative_reward'].std()

        ax.bar(i, mean_val, width, yerr=std_val, color=color,
               label=variant, capsize=5, alpha=0.8)

        # Plot individual seeds
        for j, (_, row) in enumerate(variant_df.iterrows()):
            ax.scatter(i + (j - len(variant_df)/2) * 0.1, row['cumulative_reward'],
                      color='black', s=30, zorder=5)

    ax.set_xlabel('Variant')
    ax.set_ylabel('Cumulative Reward (sum over all episodes)')
    ax.set_title('Cumulative Reward Comparison Across Variants')
    ax.set_xticks(x)
    ax.set_xticklabels(variant_order, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    for fmt in formats:
        fig.savefig(output_dir / f'comparison_cumulative_reward.{fmt}',
                   bbox_inches='tight', dpi=DEFAULT_DPI)
    plt.close(fig)
    generated.append('comparison_cumulative_reward')

    return generated


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for learning trajectory analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze learning trajectories for all NIPS benchmark runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/nips_benchmark/learning_trajectory_analysis.py \\
        --runs-dir artifacts/benchmark/nips_benchmark/20251216_140256/runs \\
        --output-dir outputs/benchmark_reports/nips_benchmark

    # With parallel processing and multiple formats
    python scripts/nips_benchmark/learning_trajectory_analysis.py \\
        --runs-dir artifacts/benchmark/nips_benchmark/20251216_140256/runs \\
        --output-dir outputs/benchmark_reports/nips_benchmark \\
        --parallel 4 \\
        --formats png pdf

    # Skip hierarchy visualization (faster, no networkx needed)
    python scripts/nips_benchmark/learning_trajectory_analysis.py \\
        --runs-dir artifacts/benchmark/nips_benchmark/20251216_140256/runs \\
        --output-dir outputs/benchmark_reports/nips_benchmark \\
        --skip-hierarchy
        """
    )

    parser.add_argument(
        '--runs-dir', '-r',
        type=Path,
        required=True,
        help='Path to benchmark runs directory'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Path to output directory'
    )

    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        default=DEFAULT_FIGURE_FORMATS,
        help=f'Output figure formats (default: {DEFAULT_FIGURE_FORMATS})'
    )

    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1, sequential)'
    )

    parser.add_argument(
        '--skip-hierarchy',
        action='store_true',
        help='Skip hierarchy navigation visualization (faster, no networkx)'
    )

    parser.add_argument(
        '--skip-ghsom',
        action='store_true',
        help='Skip all GHSOM-dependent visualizations'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Suppress matplotlib debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # Validate paths
    if not args.runs_dir.exists():
        logger.error(f"Runs directory not found: {args.runs_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Learning Trajectory Analysis")
    logger.info(f"  Runs directory: {args.runs_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Output formats: {args.formats}")
    logger.info(f"  Parallel workers: {args.parallel}")

    # Discover runs
    runs = discover_runs(args.runs_dir)

    if not runs:
        logger.error("No runs found to analyze")
        sys.exit(1)

    # Analyze runs
    results: List[RunAnalysisResult] = []

    if args.parallel > 1:
        # Parallel execution
        logger.info(f"Analyzing {len(runs)} runs in parallel with {args.parallel} workers...")

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    analyze_single_run_wrapper,
                    (run, args.output_dir, args.formats, args.skip_hierarchy, args.skip_ghsom)
                ): run['variant_name']
                for run in runs
            }

            for future in as_completed(futures):
                variant_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"  Completed: {variant_name} ({result.status})")
                except Exception as e:
                    logger.error(f"  Failed: {variant_name} - {e}")
    else:
        # Sequential execution
        logger.info(f"Analyzing {len(runs)} runs sequentially...")

        for run in runs:
            try:
                result = analyze_single_run(
                    run, args.output_dir, args.formats,
                    args.skip_hierarchy, args.skip_ghsom
                )
                results.append(result)
                logger.info(f"  Completed: {run['variant_name']} ({result.status})")
            except Exception as e:
                logger.error(f"  Failed: {run['variant_name']} - {e}")

    # Compute summary statistics
    logger.info("Computing summary statistics...")
    summary_stats = compute_summary_statistics(results)

    # Save summary statistics as JSON
    stats_path = args.output_dir / 'analysis_summary.json'
    with open(stats_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"  Saved: {stats_path}")

    # Generate comparison figures
    logger.info("Generating comparison figures...")
    comparison_figs = generate_comparison_figures(results, args.output_dir, args.formats)
    for fig_name in comparison_figs:
        logger.info(f"  Generated: {fig_name}")

    # Create benchmark result
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    benchmark_result = BenchmarkAnalysisResult(
        benchmark_dir=str(args.runs_dir),
        output_dir=str(args.output_dir),
        timestamp=timestamp,
        total_runs=len(runs),
        successful_runs=sum(1 for r in results if r.status in ('success', 'partial')),
        failed_runs=sum(1 for r in results if r.status == 'failed'),
        run_results=results,
        summary_statistics=summary_stats,
    )

    # Generate markdown report
    logger.info("Generating summary report...")
    report_path = generate_summary_report(benchmark_result, args.output_dir)
    logger.info(f"  Saved: {report_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("LEARNING TRAJECTORY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total runs: {len(runs)}")
    print(f"  Successful: {benchmark_result.successful_runs}")
    print(f"  Failed: {benchmark_result.failed_runs}")
    print(f"  Output: {args.output_dir}")
    print(f"  Report: {report_path}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
