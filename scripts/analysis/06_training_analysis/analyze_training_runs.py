#!/usr/bin/env python3
"""
Training Runs Analysis Pipeline

Analyzes training runs from artifacts/training/ directory, generating
comprehensive reports and visualizations for comparison and debugging.

Usage:
    python analyze_training_runs.py --input-dir artifacts/training --output-dir outputs/training_analysis
"""

import argparse
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress TensorFlow/TensorBoard logging before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorboard").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Import new modules
from run_discovery import BenchmarkDiscovery, RunInfo
from data_loaders import ConfigLoader, MetricsLoader, TensorBoardLoader, TrainingConfig, get_nested
from latex_tables import LatexTableGenerator, AblationAnalyzer, generate_all_tables_from_runs

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


@dataclass
class RunMetrics:
    """Container for a single run's metrics."""

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    episode_rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    episode_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    total_episodes: int = 0
    total_steps: int = 0

    # Metadata from discovery
    method: Optional[str] = None
    phase: Optional[str] = None
    seed: Optional[int] = None
    variant: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # TensorBoard metrics
    tb_loss: List[Tuple[int, float]] = field(default_factory=list)
    tb_lr: List[Tuple[int, float]] = field(default_factory=list)
    tb_exploration_rate: List[Tuple[int, float]] = field(default_factory=list)

    # LR history from file (fallback if no TensorBoard data)
    lr_history: List[Tuple[int, float]] = field(default_factory=list)

    # Computed statistics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    final_mean_reward: float = 0.0
    final_std_reward: float = 0.0
    trend_slope: float = 0.0
    stability_score: float = 0.0
    convergence_episode: Optional[int] = None
    anomalies: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""

    input_dir: Path
    output_dir: Path
    n_jobs: int = 1
    window: int = 20
    runs: Optional[List[str]] = None
    late_phase_ratio: float = 0.2
    convergence_threshold: float = 0.05
    outlier_sigma: float = 3.0
    recursive: bool = False
    group_by: Optional[str] = None
    output_format: str = "markdown"
    aggregate_seeds: bool = False
    baseline: Optional[str] = None


def load_tensorboard_data(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load scalar data from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        logger.warning("TensorBoard not available, skipping TB metrics")
        return {}

    result = {}

    # Find the subdirectory with events
    event_dirs = list(log_dir.glob("**/events.out.tfevents.*"))
    if not event_dirs:
        return result

    event_dir = event_dirs[0].parent

    try:
        ea = EventAccumulator(str(event_dir))
        ea.Reload()

        scalar_tags = ea.Tags().get("scalars", [])

        tag_mapping = {
            "train/loss": "loss",
            "train/learning_rate": "lr",
            "rollout/exploration_rate": "exploration_rate",
        }

        for tb_tag, key in tag_mapping.items():
            if tb_tag in scalar_tags:
                events = ea.Scalars(tb_tag)
                result[key] = [(e.step, e.value) for e in events]
    except Exception as e:
        logger.debug(f"Could not load TensorBoard data: {e}")

    return result


def load_run_data(run_info: RunInfo) -> Optional[RunMetrics]:
    """Load all data for a single run using new data loaders."""
    run_dir = run_info.path
    run_name = run_dir.name

    # Load config - check multiple possible locations
    config = {}
    config_paths = [
        run_dir / "config.yaml",
        run_dir / "configs" / "agent_config.json",
        run_dir / "config.json",
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                if config_path.suffix == ".yaml":
                    config_loader = ConfigLoader(config_path)
                    training_config = config_loader.load()
                    config = config_loader.get_raw_config()
                else:
                    with open(config_path) as f:
                        config = json.load(f)
                break
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")

    if not config:
        logger.warning(f"No config found for {run_name}")

    # Load metrics - check multiple possible locations
    metrics_data = None
    metrics_paths = [
        run_dir / "metrics" / "training_metrics.json",
        run_dir / "training_metrics.json",
        run_dir / "metrics" / "detailed_rewards.json",
    ]

    for metrics_path in metrics_paths:
        if metrics_path.exists():
            try:
                metrics_loader = MetricsLoader(metrics_path)
                training_metrics = metrics_loader.load()
                # Convert TrainingMetrics to dict format expected by RunMetrics
                metrics_data = {
                    "episode_rewards": training_metrics.episode_rewards.tolist(),
                    "episode_lengths": training_metrics.episode_lengths.tolist(),
                    "total_episodes": training_metrics.total_episodes,
                    "total_steps": training_metrics.total_steps,
                }
                break
            except Exception as e:
                logger.warning(f"Error loading metrics from {metrics_path}: {e}")

    if metrics_data is None:
        logger.warning(f"No metrics found for {run_name}")
        return None

    # Create RunMetrics object with metadata from RunInfo
    run_metrics = RunMetrics(
        name=run_name,
        config=config,
        episode_rewards=np.array(metrics_data.get("episode_rewards", [])),
        episode_lengths=np.array(metrics_data.get("episode_lengths", [])),
        total_episodes=metrics_data.get(
            "total_episodes", len(metrics_data.get("episode_rewards", []))
        ),
        total_steps=metrics_data.get("total_steps", 0),
        method=run_info.method,
        phase=run_info.phase,
        seed=run_info.seed,
        variant=run_info.variant,
        metadata=run_info.metadata,
    )

    # Load TensorBoard data - check multiple possible locations
    tb_data = {}
    tb_paths = [run_dir / "tensorboard", run_dir / "logs"]

    for tb_path in tb_paths:
        if tb_path.exists():
            try:
                tb_loader = TensorBoardLoader(tb_path)
                tb_result = tb_loader.load()
                # Convert TensorBoardData to dict format
                tb_data = {
                    "loss": list(tb_result.loss.items()) if tb_result.loss else [],
                    "lr": list(tb_result.learning_rate.items()) if tb_result.learning_rate else [],
                    "exploration_rate": list(tb_result.exploration_rate.items()) if tb_result.exploration_rate else [],
                }
                break
            except Exception as e:
                logger.debug(f"Could not load TensorBoard from {tb_path}: {e}")

    run_metrics.tb_loss = tb_data.get("loss", [])
    run_metrics.tb_lr = tb_data.get("lr", [])
    run_metrics.tb_exploration_rate = tb_data.get("exploration_rate", [])

    # Load LR history from file (as fallback/alternative to TensorBoard)
    lr_history_path = run_dir / "lr_history.json"
    if lr_history_path.exists():
        try:
            with open(lr_history_path) as f:
                lr_data = json.load(f)
                # Convert to list of tuples (step, lr)
                run_metrics.lr_history = [(int(item[0]), float(item[1])) for item in lr_data]
        except Exception as e:
            logger.debug(f"Could not load lr_history from {lr_history_path}: {e}")

    return run_metrics


def compute_statistics(run: RunMetrics, config: AnalysisConfig) -> RunMetrics:
    """Compute all statistics for a run."""
    rewards = run.episode_rewards

    if len(rewards) == 0:
        return run

    # Basic statistics
    run.mean_reward = float(np.mean(rewards))
    run.std_reward = float(np.std(rewards))
    run.min_reward = float(np.min(rewards))
    run.max_reward = float(np.max(rewards))

    # Final phase statistics
    late_start = int(len(rewards) * (1 - config.late_phase_ratio))
    late_rewards = rewards[late_start:]
    run.final_mean_reward = float(np.mean(late_rewards))
    run.final_std_reward = float(np.std(late_rewards))

    # Trend (linear regression slope)
    x = np.arange(len(rewards))
    slope, _, _, _, _ = stats.linregress(x, rewards)
    run.trend_slope = float(slope)

    # Stability score (inverse of coefficient of variation in late phase)
    cv = (
        run.final_std_reward / run.final_mean_reward
        if run.final_mean_reward != 0
        else float("inf")
    )
    run.stability_score = float(1 / (1 + cv))  # Higher is more stable

    # Convergence detection
    window = config.window
    rolling_std = pd.Series(rewards).rolling(window=window).std()
    rolling_mean = pd.Series(rewards).rolling(window=window).mean()
    rolling_cv = rolling_std / rolling_mean

    # Find first episode where CV stays below threshold
    threshold = config.convergence_threshold
    for i in range(window, len(rolling_cv)):
        if all(rolling_cv[i : min(i + window, len(rolling_cv))] < threshold):
            run.convergence_episode = i
            break

    # Anomaly detection
    anomalies = []

    # Check for reward collapse (>50% drop from rolling mean)
    rolling_mean_arr = rolling_mean.values
    for i in range(window, len(rewards)):
        if not np.isnan(rolling_mean_arr[i - 1]) and rolling_mean_arr[i - 1] > 0:
            if rewards[i] < rolling_mean_arr[i - 1] * 0.5:
                anomalies.append(f"Reward collapse at episode {i}")
                break

    # Check for stuck rewards (constant for >30 episodes)
    for i in range(30, len(rewards)):
        if np.std(rewards[i - 30 : i]) < 0.01:
            anomalies.append(f"Stuck rewards around episode {i-15}")
            break

    # Check for high variance in late phase
    if cv > 0.3:
        anomalies.append("High variance in late training phase")

    # Check for outliers
    z_scores = np.abs(stats.zscore(rewards))
    outlier_episodes = np.where(z_scores > config.outlier_sigma)[0]
    if len(outlier_episodes) > 0:
        anomalies.append(f"Outliers at episodes: {outlier_episodes[:5].tolist()}")

    run.anomalies = anomalies

    return run


def compare_configs(runs: List[RunMetrics]) -> pd.DataFrame:
    """Create a comparison table of config differences.

    Extracts key configuration parameters including:
    - Learning rate and scheduler settings
    - Reward component weights (diversity, structure, transition)
    - Curriculum learning settings
    - Network architecture options (noisy net, n_step)
    - Feature observation mode
    """
    if not runs:
        return pd.DataFrame()

    # Extract key config parameters
    config_data = []

    for run in runs:
        cfg = run.config

        # Basic training parameters
        lr = get_nested(cfg, "training", "learning_rate", default="N/A")
        batch_size = get_nested(cfg, "training", "batch_size", default="N/A")
        gamma = get_nested(cfg, "training", "gamma", default="N/A")
        total_timesteps = get_nested(cfg, "training", "total_timesteps", default="N/A")
        n_step = get_nested(cfg, "training", "n_step", default="N/A")

        # Learning rate scheduler
        lr_sched_enabled = get_nested(cfg, "training", "learning_rate_scheduler", "enabled", default=False)
        lr_sched_type = get_nested(cfg, "training", "learning_rate_scheduler", "type", default="none")
        initial_lr = get_nested(cfg, "training", "learning_rate_scheduler", "initial_lr", default="N/A")
        final_lr = get_nested(cfg, "training", "learning_rate_scheduler", "final_lr", default="N/A")
        decay_rate = get_nested(cfg, "training", "learning_rate_scheduler", "decay_rate", default="N/A")

        # Reward component weights
        diversity_weight = get_nested(cfg, "reward_components", "diversity", "weight", default="N/A")
        structure_weight = get_nested(cfg, "reward_components", "structure", "weight", default="N/A")
        transition_weight = get_nested(cfg, "reward_components", "transition", "weight", default="N/A")

        # Feature observation mode (root level)
        feature_obs_mode = get_nested(cfg, "feature_observation_mode", default="N/A")

        # Curriculum learning settings
        curriculum_enabled = get_nested(cfg, "curriculum", "enabled", default=False)
        timesteps_per_action = get_nested(cfg, "curriculum", "timesteps_per_action", default="N/A")

        # Network options
        noisy_net_enabled = get_nested(cfg, "network", "noisy_net", "enabled", default=False)

        config_data.append(
            {
                "Run": run.name.replace("run_", ""),
                # Training basics
                "LR": lr,
                "Batch": batch_size,
                "γ": gamma,
                "Timesteps": total_timesteps,
                "N-Step": n_step,
                # LR Scheduler
                "LR Sched": lr_sched_type if lr_sched_enabled else "none",
                "Init LR": initial_lr,
                "Final LR": final_lr,
                "Decay": decay_rate,
                # Reward weights
                "Div W": diversity_weight,
                "Struct W": structure_weight,
                "Trans W": transition_weight,
                # Feature & Curriculum
                "Feat Mode": feature_obs_mode,
                "CL": "Yes" if curriculum_enabled else "No",
                "CL Steps": timesteps_per_action if curriculum_enabled else "N/A",
                # Network
                "Noisy": "Yes" if noisy_net_enabled else "No",
            }
        )

    return pd.DataFrame(config_data)


def statistical_comparison(runs: List[RunMetrics]) -> pd.DataFrame:
    """Perform pairwise statistical tests between runs."""
    n = len(runs)
    results = []

    for i in range(n):
        for j in range(i + 1, n):
            run1, run2 = runs[i], runs[j]

            # Use late phase rewards for comparison
            late_start1 = int(len(run1.episode_rewards) * 0.8)
            late_start2 = int(len(run2.episode_rewards) * 0.8)

            rewards1 = run1.episode_rewards[late_start1:]
            rewards2 = run2.episode_rewards[late_start2:]

            # Mann-Whitney U test (non-parametric)
            stat, p_value = stats.mannwhitneyu(
                rewards1, rewards2, alternative="two-sided"
            )

            results.append(
                {
                    "Run 1": run1.name.replace("run_", ""),
                    "Run 2": run2.name.replace("run_", ""),
                    "p-value": p_value,
                    "Significant": "Yes" if p_value < 0.05 else "No",
                }
            )

    return pd.DataFrame(results)


# ============== Visualization Functions ==============


def plot_reward_curves_overlay(
    runs: List[RunMetrics], output_path: Path, window: int = 20
):
    """Plot all reward curves on one figure."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for run, color in zip(runs, colors):
        episodes = np.arange(len(run.episode_rewards))

        # Raw rewards (faint)
        ax.plot(episodes, run.episode_rewards, alpha=0.2, color=color, linewidth=0.5)

        # Smoothed rewards (bold)
        smoothed = (
            pd.Series(run.episode_rewards).rolling(window=window, min_periods=1).mean()
        )
        ax.plot(
            episodes,
            smoothed,
            color=color,
            linewidth=2,
            label=run.name.replace("run_", ""),
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Reward Curves Comparison (with Rolling Mean)", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_performance_ranking(runs: List[RunMetrics], output_path: Path):
    """Bar chart of final mean rewards with error bars."""
    # Sort by final mean reward
    sorted_runs = sorted(runs, key=lambda r: r.final_mean_reward, reverse=True)

    names = [r.name.replace("run_", "") for r in sorted_runs]
    means = [r.final_mean_reward for r in sorted_runs]
    stds = [r.final_std_reward for r in sorted_runs]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(runs)))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    # Add value labels on bars (positioned just above error bars)
    max_val = max(m + s for m, s in zip(means, stds)) if stds else max(means)
    offset = max_val * 0.02  # Small offset relative to data range
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + offset,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Final Mean Reward", fontsize=12)
    ax.set_title("Performance Ranking (Last 20% of Training)", fontsize=14)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_curves(runs: List[RunMetrics], output_path: Path):
    """Plot loss curves from TensorBoard data."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    has_data = False

    for run, color in zip(runs, colors):
        if run.tb_loss:
            steps, losses = zip(*run.tb_loss)
            ax.plot(
                steps,
                losses,
                color=color,
                linewidth=1.5,
                label=run.name.replace("run_", ""),
                alpha=0.8,
            )
            has_data = True

    if has_data:
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss Curves", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No TensorBoard loss data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title("Training Loss Curves", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lr_schedules(runs: List[RunMetrics], output_path: Path):
    """Plot learning rate schedules from lr_history.json or TensorBoard."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    has_data = False

    for run, color in zip(runs, colors):
        # Prefer lr_history (from file), fall back to tb_lr (from TensorBoard)
        lr_data = run.lr_history if run.lr_history else run.tb_lr
        if lr_data:
            steps, lrs = zip(*lr_data)
            # Subsample if too many points (for performance)
            if len(steps) > 1000:
                step_size = len(steps) // 500
                steps = steps[::step_size]
                lrs = lrs[::step_size]
            ax.plot(
                steps, lrs, color=color, linewidth=1.5, label=run.name.replace("run_", ""),
                alpha=0.8
            )
            has_data = True

    if has_data:
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title("Learning Rate Schedules", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No LR data available",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title("Learning Rate Schedules", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_reward_distributions(runs: List[RunMetrics], output_path: Path):
    """Side-by-side boxplots of reward distributions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = []
    labels = []

    for run in runs:
        data.append(run.episode_rewards)
        labels.append(run.name.replace("run_", ""))

    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticklabels(labels)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(runs)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Reward Distributions", fontsize=14)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stability_comparison(
    runs: List[RunMetrics], output_path: Path, window: int = 20
):
    """Plot rolling standard deviation over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for run, color in zip(runs, colors):
        rolling_std = (
            pd.Series(run.episode_rewards).rolling(window=window, min_periods=1).std()
        )
        episodes = np.arange(len(rolling_std))

        ax.plot(
            episodes,
            rolling_std,
            color=color,
            linewidth=1.5,
            label=run.name.replace("run_", ""),
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Rolling Std (window={})".format(window), fontsize=12)
    ax.set_title("Training Stability Comparison", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_comparison(runs: List[RunMetrics], output_path: Path):
    """Spider/Radar chart for multi-dimensional comparison."""
    # Define metrics for radar chart
    metrics = ["Final Reward", "Stability", "Trend", "Consistency", "Peak Performance"]

    # Normalize each metric to 0-1 range
    data = []
    for run in runs:
        # Final Reward (normalized)
        final_rewards = [r.final_mean_reward for r in runs]
        norm_final = (run.final_mean_reward - min(final_rewards)) / (
            max(final_rewards) - min(final_rewards) + 1e-8
        )

        # Stability (already 0-1)
        norm_stability = run.stability_score

        # Trend (normalize, positive is better)
        trends = [r.trend_slope for r in runs]
        min_trend, max_trend = min(trends), max(trends)
        norm_trend = (run.trend_slope - min_trend) / (max_trend - min_trend + 1e-8)

        # Consistency (inverse of CV, normalized)
        cvs = [r.final_std_reward / (r.final_mean_reward + 1e-8) for r in runs]
        max_cv = max(cvs)
        cv = run.final_std_reward / (run.final_mean_reward + 1e-8)
        norm_consistency = 1 - (cv / (max_cv + 1e-8))

        # Peak performance (max reward normalized)
        max_rewards = [r.max_reward for r in runs]
        norm_peak = (run.max_reward - min(max_rewards)) / (
            max(max_rewards) - min(max_rewards) + 1e-8
        )

        data.append(
            [norm_final, norm_stability, norm_trend, norm_consistency, norm_peak]
        )

    # Number of variables
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for i, (run, color) in enumerate(zip(runs, colors)):
        values = data[i]
        values += values[:1]  # Complete the loop

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            color=color,
            label=run.name.replace("run_", ""),
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    ax.set_title("Multi-Dimensional Performance Comparison", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============== New Helper Functions ==============


def aggregate_across_seeds(runs: List[RunMetrics]) -> List[RunMetrics]:
    """
    Find runs with same config but different seeds and aggregate them.

    Returns aggregated RunMetrics with mean ± std for all metrics.
    """
    # Group runs by their base name (without seed suffix)
    groups = defaultdict(list)

    for run in runs:
        # Extract base name (remove seed suffix like _seed0, _seed1, etc.)
        base_name = run.name
        if "_seed" in base_name:
            base_name = base_name.rsplit("_seed", 1)[0]
        groups[base_name].append(run)

    aggregated_runs = []

    for base_name, group_runs in groups.items():
        if len(group_runs) == 1:
            # Only one seed, keep as is
            aggregated_runs.append(group_runs[0])
            continue

        # Aggregate across seeds
        all_rewards = [run.episode_rewards for run in group_runs]
        min_length = min(len(r) for r in all_rewards)

        # Truncate all to same length
        all_rewards_truncated = [r[:min_length] for r in all_rewards]

        # Compute mean and std across seeds
        mean_rewards = np.mean(all_rewards_truncated, axis=0)
        std_rewards = np.std(all_rewards_truncated, axis=0)

        # Aggregate statistics
        final_means = [run.final_mean_reward for run in group_runs]
        final_stds = [run.final_std_reward for run in group_runs]

        aggregated = RunMetrics(
            name=f"{base_name}_aggregated",
            config=group_runs[0].config,  # Use first config
            episode_rewards=mean_rewards,
            episode_lengths=np.mean([run.episode_lengths[:min_length] for run in group_runs], axis=0),
            total_episodes=min_length,
            total_steps=int(np.mean([run.total_steps for run in group_runs])),
        )

        # Aggregate computed statistics
        aggregated.mean_reward = float(np.mean([r.mean_reward for r in group_runs]))
        aggregated.std_reward = float(np.mean([r.std_reward for r in group_runs]))
        aggregated.min_reward = float(np.mean([r.min_reward for r in group_runs]))
        aggregated.max_reward = float(np.mean([r.max_reward for r in group_runs]))
        aggregated.final_mean_reward = float(np.mean(final_means))
        aggregated.final_std_reward = float(np.mean(final_stds))
        aggregated.trend_slope = float(np.mean([r.trend_slope for r in group_runs]))
        aggregated.stability_score = float(np.mean([r.stability_score for r in group_runs]))

        aggregated_runs.append(aggregated)

        logger.info(f"Aggregated {len(group_runs)} seeds for {base_name}")

    return aggregated_runs


def generate_grouped_report(
    runs: List[RunMetrics],
    group_by: str,
    output_dir: Path,
    config: AnalysisConfig
) -> Dict[str, List[RunMetrics]]:
    """
    Group runs by specified dimension and generate separate analyses.

    Args:
        runs: List of all runs
        group_by: Dimension to group by (phase, method, seed, variant)
        output_dir: Directory for grouped outputs
        config: Analysis configuration

    Returns:
        Dictionary mapping group names to run lists
    """
    groups = defaultdict(list)

    # Group runs based on specified dimension
    for run in runs:
        if group_by == "phase":
            # Extract phase from run name (e.g., run_phase1_...)
            if "_phase" in run.name:
                phase = run.name.split("_phase")[1].split("_")[0]
                group_key = f"phase{phase}"
            else:
                group_key = "default"
        elif group_by == "method":
            # Extract method from config or name
            method = run.config.get("method", "unknown")
            if method == "unknown":
                # Try to extract from name
                parts = run.name.split("_")
                method = parts[1] if len(parts) > 1 else "unknown"
            group_key = method
        elif group_by == "seed":
            # Group by seed number
            if "_seed" in run.name:
                seed = run.name.split("_seed")[1].split("_")[0]
                group_key = f"seed{seed}"
            else:
                group_key = "seed0"
        elif group_by == "variant":
            # Group by variant name
            if "variant" in run.config:
                group_key = run.config["variant"]
            else:
                # Try to extract from name
                parts = run.name.replace("run_", "").split("_")
                group_key = parts[0] if parts else "default"
        else:
            group_key = "all"

        groups[group_key].append(run)

    logger.info(f"Created {len(groups)} groups by {group_by}")

    # Generate report for each group
    group_reports_dir = output_dir / "grouped_reports"
    group_reports_dir.mkdir(exist_ok=True)

    for group_name, group_runs in groups.items():
        logger.info(f"Generating report for group: {group_name} ({len(group_runs)} runs)")

        group_dir = group_reports_dir / group_name
        group_dir.mkdir(exist_ok=True)

        # Create group-specific plots
        plots_dir = group_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Generate visualizations for this group
        if len(group_runs) > 1:
            plot_reward_curves_overlay(group_runs, plots_dir / "reward_curves.png", config.window)
            plot_performance_ranking(group_runs, plots_dir / "performance_ranking.png")
            plot_reward_distributions(group_runs, plots_dir / "reward_distributions.png")

        # Generate group summary
        summary_path = group_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# Group: {group_name}\n\n")
            f.write(f"**Number of runs:** {len(group_runs)}\n\n")
            f.write("## Runs in this group\n\n")
            for run in sorted(group_runs, key=lambda r: r.final_mean_reward, reverse=True):
                f.write(f"- {run.name}: {run.final_mean_reward:.2f} ± {run.final_std_reward:.2f}\n")

    return dict(groups)


# ============== Report Generation ==============


def generate_summary_report(
    runs: List[RunMetrics],
    config_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_path: Path,
    analysis_config: AnalysisConfig,
):
    """Generate the main markdown summary report."""

    # Sort runs by performance
    sorted_runs = sorted(runs, key=lambda r: r.final_mean_reward, reverse=True)

    report = []
    report.append("# Training Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Input Directory:** `{analysis_config.input_dir}`")
    report.append(f"\n**Runs Analyzed:** {len(runs)}")
    report.append("")

    # Performance Ranking
    report.append("## Performance Ranking")
    report.append("")
    report.append("| Rank | Run | Final Mean | Std | Trend | Stability |")
    report.append("|------|-----|------------|-----|-------|-----------|")

    for i, run in enumerate(sorted_runs, 1):
        trend_symbol = (
            "↑" if run.trend_slope > 0.01 else ("↓" if run.trend_slope < -0.01 else "→")
        )
        report.append(
            f"| {i} | {run.name.replace('run_', '')} | {run.final_mean_reward:.2f} | "
            f"{run.final_std_reward:.2f} | {trend_symbol} {run.trend_slope:.4f} | {run.stability_score:.3f} |"
        )

    report.append("")

    # Key Findings
    report.append("## Key Findings")
    report.append("")

    best_run = sorted_runs[0]
    most_stable = max(runs, key=lambda r: r.stability_score)
    fastest_learner = max(runs, key=lambda r: r.trend_slope)

    report.append(
        f"- **Best Performer:** `{best_run.name.replace('run_', '')}` "
        f"(final mean = {best_run.final_mean_reward:.2f})"
    )
    report.append(
        f"- **Most Stable:** `{most_stable.name.replace('run_', '')}` "
        f"(stability score = {most_stable.stability_score:.3f})"
    )
    report.append(
        f"- **Best Learning Trend:** `{fastest_learner.name.replace('run_', '')}` "
        f"(slope = {fastest_learner.trend_slope:.4f})"
    )
    report.append("")

    # Statistical Significance
    if not stats_df.empty:
        report.append("## Statistical Significance (Mann-Whitney U Test)")
        report.append("")
        sig_pairs = stats_df[stats_df["Significant"] == "Yes"]
        if len(sig_pairs) > 0:
            report.append("Significant differences (p < 0.05):")
            report.append("")
            for _, row in sig_pairs.iterrows():
                report.append(
                    f"- `{row['Run 1']}` vs `{row['Run 2']}` (p = {row['p-value']:.4f})"
                )
        else:
            report.append(
                "No statistically significant differences found between runs."
            )
        report.append("")

    # Anomalies
    report.append("## Anomalies Detected")
    report.append("")

    any_anomalies = False
    for run in runs:
        if run.anomalies:
            any_anomalies = True
            report.append(f"### {run.name.replace('run_', '')}")
            for anomaly in run.anomalies:
                report.append(f"- {anomaly}")
            report.append("")

    if not any_anomalies:
        report.append("No significant anomalies detected in any run.")
        report.append("")

    # Detailed Statistics
    report.append("## Detailed Statistics")
    report.append("")
    report.append("| Run | Episodes | Mean | Std | Min | Max | Convergence |")
    report.append("|-----|----------|------|-----|-----|-----|-------------|")

    for run in sorted_runs:
        conv = (
            f"Ep {run.convergence_episode}"
            if run.convergence_episode
            else "Not converged"
        )
        report.append(
            f"| {run.name.replace('run_', '')} | {run.total_episodes} | "
            f"{run.mean_reward:.2f} | {run.std_reward:.2f} | "
            f"{run.min_reward:.2f} | {run.max_reward:.2f} | {conv} |"
        )

    report.append("")

    # Visualizations
    report.append("## Visualizations")
    report.append("")
    report.append("See the `plots/` directory for:")
    report.append("- `reward_curves_overlay.png` - All reward curves comparison")
    report.append("- `performance_ranking.png` - Final performance bar chart")
    report.append("- `loss_curves.png` - Training loss over time")
    report.append("- `lr_schedules.png` - Learning rate schedules")
    report.append("- `reward_distributions.png` - Reward distribution boxplots")
    report.append("- `stability_comparison.png` - Rolling std over time")
    report.append("- `radar_comparison.png` - Multi-dimensional comparison")
    report.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report))


def generate_config_comparison(config_df: pd.DataFrame, output_path: Path):
    """Generate config comparison markdown file."""
    report = []
    report.append("# Configuration Comparison")
    report.append("")
    report.append(config_df.to_markdown(index=False))

    with open(output_path, "w") as f:
        f.write("\n".join(report))


def save_analysis_data(
    runs: List[RunMetrics], stats_df: pd.DataFrame, output_path: Path
):
    """Save analysis results as JSON for programmatic access."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "runs": [],
        "statistical_tests": stats_df.to_dict("records") if not stats_df.empty else [],
    }

    for run in runs:
        run_data = {
            "name": run.name,
            "method": run.method,
            "phase": run.phase,
            "seed": run.seed,
            "variant": run.variant,
            "metadata": run.metadata,
            "total_episodes": run.total_episodes,
            "total_steps": run.total_steps,
            "mean_reward": run.mean_reward,
            "std_reward": run.std_reward,
            "min_reward": run.min_reward,
            "max_reward": run.max_reward,
            "final_mean_reward": run.final_mean_reward,
            "final_std_reward": run.final_std_reward,
            "trend_slope": run.trend_slope,
            "stability_score": run.stability_score,
            "convergence_episode": run.convergence_episode,
            "anomalies": run.anomalies,
        }
        data["runs"].append(run_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ============== Main Pipeline ==============


def run_analysis(config: AnalysisConfig) -> None:
    """Main analysis pipeline with new discovery and grouping capabilities."""

    logger.info(f"Starting training analysis pipeline")
    logger.info(f"Input directory: {config.input_dir}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / f"analysis_{timestamp}"
    plots_dir = output_dir / "plots"
    data_dir = output_dir / "data"

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Use BenchmarkDiscovery to find runs
    logger.info("Discovering training runs...")
    discovery = BenchmarkDiscovery()
    run_infos = discovery.discover_runs(config.input_dir, recursive=config.recursive)

    # Filter by specific runs if requested
    if config.runs:
        run_infos = [r for r in run_infos if r.name in config.runs]

    logger.info(f"Found {len(run_infos)} runs to analyze")

    if not run_infos:
        logger.error("No valid runs found!")
        return

    # Load data for each discovered run
    logger.info("Loading run data...")
    runs: List[RunMetrics] = []

    for run_info in tqdm(run_infos, desc="Loading runs"):
        run_data = load_run_data(run_info)
        if run_data:
            runs.append(run_data)

    if not runs:
        logger.error("No valid run data could be loaded!")
        return

    # Aggregate across seeds if requested
    if config.aggregate_seeds:
        logger.info("Aggregating across seeds...")
        runs = aggregate_across_seeds(runs)
        logger.info(f"After aggregation: {len(runs)} runs")

    # Compute statistics
    logger.info("Computing statistics...")
    for i, run in enumerate(tqdm(runs, desc="Computing stats")):
        runs[i] = compute_statistics(run, config)

    # Generate grouped reports if requested
    if config.group_by:
        logger.info(f"Generating grouped reports by {config.group_by}...")
        groups = generate_grouped_report(runs, config.group_by, output_dir, config)
        logger.info(f"Created {len(groups)} groups")

    # Generate comparisons
    logger.info("Generating comparisons...")
    config_df = compare_configs(runs)
    stats_df = statistical_comparison(runs)

    # Generate visualizations
    logger.info("Generating visualizations...")

    plot_reward_curves_overlay(
        runs, plots_dir / "reward_curves_overlay.png", config.window
    )
    plot_performance_ranking(runs, plots_dir / "performance_ranking.png")
    plot_loss_curves(runs, plots_dir / "loss_curves.png")
    plot_lr_schedules(runs, plots_dir / "lr_schedules.png")
    plot_reward_distributions(runs, plots_dir / "reward_distributions.png")
    plot_stability_comparison(
        runs, plots_dir / "stability_comparison.png", config.window
    )
    plot_radar_comparison(runs, plots_dir / "radar_comparison.png")

    logger.info("Visualizations saved to plots/")

    # Generate reports in requested formats
    logger.info("Generating reports...")

    if config.output_format in ["markdown", "all"]:
        generate_summary_report(
            runs, config_df, stats_df, output_dir / "summary_report.md", config
        )
        generate_config_comparison(config_df, output_dir / "config_comparison.md")

    if config.output_format in ["json", "all"]:
        save_analysis_data(runs, stats_df, data_dir / "analysis_results.json")

    if config.output_format in ["latex", "all"]:
        logger.info("Generating LaTeX tables...")

        # Use convenience function to generate tables
        baseline_name = config.baseline if config.baseline else None
        tables = generate_all_tables_from_runs(
            runs=runs,
            output_dir=output_dir,
            baseline_name=baseline_name,
        )

        logger.info(f"Generated {len(tables)} LaTeX tables")

    logger.info(f"Analysis complete! Results saved to: {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles generated:")
    if config.output_format in ["markdown", "all"]:
        print(f"  - summary_report.md")
        print(f"  - config_comparison.md")
    if config.output_format in ["json", "all"]:
        print(f"  - data/analysis_results.json")
    if config.output_format in ["latex", "all"]:
        print(f"  - training_comparison.tex")
    print(f"  - plots/ (7 visualizations)")
    if config.group_by:
        print(f"  - grouped_reports/ (by {config.group_by})")

    # Quick summary
    best_run = max(runs, key=lambda r: r.final_mean_reward)
    print(f"\nBest performer: {best_run.name.replace('run_', '')}")
    print(f"  Final mean reward: {best_run.final_mean_reward:.2f}")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze training runs and generate comparison reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_training_runs.py
  python analyze_training_runs.py --input-dir artifacts/training --n-jobs 4
  python analyze_training_runs.py --runs run_linear-lr-decay,run_exponential-lr-decay
  python analyze_training_runs.py --recursive --group-by phase --output-format latex
  python analyze_training_runs.py --aggregate-seeds --baseline run_baseline
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/training"),
        help="Directory containing training runs (default: artifacts/training)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_analysis"),
        help="Base output directory (default: outputs/training_analysis)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size for smoothing (default: 20)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Comma-separated list of specific runs to analyze (default: all)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover runs in subdirectories",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["phase", "method", "seed", "variant"],
        default=None,
        help="Group runs by: phase, method, seed, or variant",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["markdown", "latex", "json", "all"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="Aggregate results across seeds (mean ± std)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Name of baseline run for relative comparisons",
    )

    args = parser.parse_args()

    # Parse runs if specified
    runs_list = None
    if args.runs:
        runs_list = [r.strip() for r in args.runs.split(",")]

    # Create config
    config = AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        window=args.window,
        runs=runs_list,
        recursive=args.recursive,
        group_by=args.group_by,
        output_format=args.output_format,
        aggregate_seeds=args.aggregate_seeds,
        baseline=args.baseline,
    )

    # Run analysis
    run_analysis(config)


if __name__ == "__main__":
    main()
