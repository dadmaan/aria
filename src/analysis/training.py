"""
Training analysis functions for DQN learning dynamics.

Provides functions to load training metrics, analyze convergence,
compute episode statistics, and extract reward components.
"""

from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


@dataclass
class TrainingMetrics:
    """Container for training metrics loaded from outputs."""

    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    timesteps: np.ndarray
    reward_components: Optional[Dict[str, np.ndarray]] = None
    action_distribution: Optional[np.ndarray] = None
    q_values: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    # Training dynamics metrics
    loss_history: Optional[np.ndarray] = None
    q_value_stats: Optional[Dict[str, np.ndarray]] = None
    gradient_norms: Optional[np.ndarray] = None
    td_errors: Optional[np.ndarray] = None


@dataclass
class ConvergenceAnalysis:
    """Results of convergence analysis."""

    converged: bool
    convergence_timestep: Optional[int]
    convergence_episode: Optional[int]
    final_mean_reward: float
    final_std_reward: float
    stability_window: int
    stability_threshold: float
    trend_slope: float
    trend_pvalue: float


def load_training_metrics(
    output_dir: Path,
    include_components: bool = True,
) -> TrainingMetrics:
    """
    Load training metrics from output directory.

    Args:
        output_dir: Path to output directory (e.g., outputs/run_20231001_120000/)
        include_components: Whether to load reward components if available

    Returns:
        TrainingMetrics object with loaded data
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    # Load main metrics
    metrics_file = output_dir / "metrics" / "training_metrics.json"

    if not metrics_file.exists():
        raise ValueError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Extract episode data
    episode_rewards = np.array(data.get("episode_rewards", []))
    episode_lengths = np.array(data.get("episode_lengths", []))

    # Generate timesteps if not available
    if "timesteps" in data:
        timesteps = np.array(data["timesteps"])
    else:
        timesteps = np.cumsum(episode_lengths)

    # Load reward components if available
    reward_components = None
    if include_components and "reward_components" in data:
        reward_components = {
            component: np.array(values)
            for component, values in data["reward_components"].items()
        }

    # Load action distribution if available
    action_distribution = None
    if "action_distribution" in data:
        action_distribution = np.array(data["action_distribution"])

    # Load Q-values if available
    q_values = None
    if "q_values" in data:
        q_values = np.array(data["q_values"])

    # Load metadata
    metadata = data.get("metadata", {})

    return TrainingMetrics(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        timesteps=timesteps,
        reward_components=reward_components,
        action_distribution=action_distribution,
        q_values=q_values,
        metadata=metadata,
    )


def analyze_convergence(
    rewards: np.ndarray,
    window_size: int = 100,
    stability_threshold: float = 0.05,
    min_episodes: int = 200,
) -> ConvergenceAnalysis:
    """
    Analyze whether training has converged.

    Convergence criteria:
    1. Coefficient of variation (std/mean) < stability_threshold
    2. No significant upward/downward trend in moving average
    3. Sustained for at least window_size episodes

    Args:
        rewards: Array of episode rewards
        window_size: Size of moving average window
        stability_threshold: Maximum coefficient of variation for stability
        min_episodes: Minimum episodes before checking convergence

    Returns:
        ConvergenceAnalysis with convergence status and statistics
    """
    rewards = np.asarray(rewards).flatten()
    n_episodes = len(rewards)

    if n_episodes < min_episodes:
        return ConvergenceAnalysis(
            converged=False,
            convergence_timestep=None,
            convergence_episode=None,
            final_mean_reward=np.mean(rewards[-window_size:]) if n_episodes >= window_size else np.mean(rewards),
            final_std_reward=np.std(rewards[-window_size:]) if n_episodes >= window_size else np.std(rewards),
            stability_window=window_size,
            stability_threshold=stability_threshold,
            trend_slope=np.nan,
            trend_pvalue=np.nan,
        )

    # Compute moving average and std
    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    moving_std = pd.Series(rewards).rolling(window=window_size).std().values[window_size - 1:]

    # Coefficient of variation
    cv = moving_std / np.abs(moving_avg)
    cv[np.abs(moving_avg) < 1e-8] = np.inf  # Avoid division by zero

    # Find first point where CV is stable
    stable_indices = np.where(cv < stability_threshold)[0]

    if len(stable_indices) == 0:
        converged = False
        convergence_episode = None
        convergence_timestep = None
    else:
        # Check if stability is sustained
        first_stable = stable_indices[0]

        # Verify stability for remaining episodes
        if all(cv[first_stable:] < stability_threshold * 1.5):  # Allow 50% margin
            converged = True
            convergence_episode = first_stable + window_size - 1
            convergence_timestep = convergence_episode  # Approximate
        else:
            converged = False
            convergence_episode = None
            convergence_timestep = None

    # Compute trend in final window
    final_rewards = rewards[-window_size:]
    x = np.arange(len(final_rewards))

    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(x, final_rewards)

    # Final statistics
    final_mean = np.mean(final_rewards)
    final_std = np.std(final_rewards)

    return ConvergenceAnalysis(
        converged=converged,
        convergence_timestep=convergence_timestep,
        convergence_episode=convergence_episode,
        final_mean_reward=final_mean,
        final_std_reward=final_std,
        stability_window=window_size,
        stability_threshold=stability_threshold,
        trend_slope=slope,
        trend_pvalue=p_value,
    )


def compute_episode_statistics(
    metrics: TrainingMetrics,
    window_size: int = 100,
) -> pd.DataFrame:
    """
    Compute rolling statistics over episodes.

    Args:
        metrics: TrainingMetrics object
        window_size: Window size for rolling statistics

    Returns:
        DataFrame with columns: episode, timestep, reward, reward_ma,
                                reward_std, reward_min, reward_max
    """
    n_episodes = len(metrics.episode_rewards)

    df = pd.DataFrame({
        'episode': np.arange(n_episodes),
        'timestep': metrics.timesteps,
        'reward': metrics.episode_rewards,
        'length': metrics.episode_lengths,
    })

    # Rolling statistics
    df['reward_ma'] = df['reward'].rolling(window=window_size, min_periods=1).mean()
    df['reward_std'] = df['reward'].rolling(window=window_size, min_periods=1).std()
    df['reward_min'] = df['reward'].rolling(window=window_size, min_periods=1).min()
    df['reward_max'] = df['reward'].rolling(window=window_size, min_periods=1).max()

    # Add confidence intervals
    df['reward_ci_lower'] = df['reward_ma'] - 1.96 * df['reward_std'] / np.sqrt(window_size)
    df['reward_ci_upper'] = df['reward_ma'] + 1.96 * df['reward_std'] / np.sqrt(window_size)

    return df


def smooth_curve(
    data: np.ndarray,
    method: str = "savgol",
    window_size: int = 51,
    **kwargs
) -> np.ndarray:
    """
    Smooth a curve using various methods.

    Args:
        data: Array of values to smooth
        method: Smoothing method ("savgol", "gaussian", "moving_avg")
        window_size: Window size for smoothing
        **kwargs: Additional arguments for smoothing method

    Returns:
        Smoothed array
    """
    data = np.asarray(data).flatten()

    if len(data) < window_size:
        window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_size < 3:
            return data

    if method == "savgol":
        polyorder = kwargs.get("polyorder", 3)
        polyorder = min(polyorder, window_size - 1)
        return savgol_filter(data, window_size, polyorder)

    elif method == "gaussian":
        sigma = kwargs.get("sigma", window_size / 6)
        return gaussian_filter1d(data, sigma)

    elif method == "moving_avg":
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def extract_reward_components(
    metrics: TrainingMetrics,
) -> Optional[pd.DataFrame]:
    """
    Extract reward components into a DataFrame.

    Args:
        metrics: TrainingMetrics object

    Returns:
        DataFrame with columns: episode, total_reward, similarity, structure, human
        Returns None if reward components are not available
    """
    if metrics.reward_components is None:
        return None

    df = pd.DataFrame({
        'episode': np.arange(len(metrics.episode_rewards)),
        'total_reward': metrics.episode_rewards,
    })

    for component, values in metrics.reward_components.items():
        df[component] = values

    return df


def compute_learning_rate(
    rewards: np.ndarray,
    window_size: int = 100,
) -> np.ndarray:
    """
    Compute instantaneous learning rate (reward improvement per episode).

    Args:
        rewards: Array of episode rewards
        window_size: Window for smoothing

    Returns:
        Array of learning rates (gradient of smoothed rewards)
    """
    rewards = np.asarray(rewards).flatten()

    # Smooth rewards first
    smoothed = smooth_curve(rewards, method="gaussian", window_size=window_size)

    # Compute gradient
    learning_rate = np.gradient(smoothed)

    return learning_rate


def identify_learning_phases(
    rewards: np.ndarray,
    phases: List[str] = None,
    thresholds: List[float] = None,
) -> Dict[str, Tuple[int, int]]:
    """
    Identify different learning phases based on reward levels.

    Args:
        rewards: Array of episode rewards
        phases: List of phase names (default: ["exploration", "learning", "convergence"])
        thresholds: Reward thresholds for phase transitions (percentiles)

    Returns:
        Dictionary mapping phase names to (start_episode, end_episode) tuples
    """
    if phases is None:
        phases = ["exploration", "learning", "convergence"]

    if thresholds is None:
        # Use quartiles as default thresholds
        thresholds = [25, 75]

    rewards = np.asarray(rewards).flatten()
    n_episodes = len(rewards)

    # Compute smoothed rewards
    smoothed = smooth_curve(rewards, method="gaussian", window_size=min(101, n_episodes))

    # Compute threshold values
    threshold_values = [np.percentile(smoothed, t) for t in thresholds]

    # Find phase boundaries
    phase_dict = {}

    # First phase: start to first threshold
    if len(phases) >= 1:
        first_cross = np.where(smoothed >= threshold_values[0])[0]
        end_idx = first_cross[0] if len(first_cross) > 0 else n_episodes
        phase_dict[phases[0]] = (0, end_idx)

    # Middle phases
    for i in range(1, len(phases) - 1):
        if i < len(threshold_values):
            start_idx = phase_dict[phases[i - 1]][1]
            cross_indices = np.where(smoothed >= threshold_values[i])[0]
            end_idx = cross_indices[0] if len(cross_indices) > 0 else n_episodes
            phase_dict[phases[i]] = (start_idx, end_idx)

    # Last phase: last threshold to end
    if len(phases) >= 2:
        start_idx = phase_dict[phases[-2]][1]
        phase_dict[phases[-1]] = (start_idx, n_episodes)

    return phase_dict


def compute_q_value_stability(
    q_values: np.ndarray,
    window_size: int = 100,
) -> np.ndarray:
    """
    Compute Q-value stability over time (coefficient of variation).

    Args:
        q_values: Array of Q-values over episodes
        window_size: Window for computing stability

    Returns:
        Array of stability metrics (CV) over episodes
    """
    q_values = np.asarray(q_values)

    if q_values.ndim == 1:
        # Single Q-value per episode
        q_values = q_values.reshape(-1, 1)

    # Compute rolling mean and std across actions
    mean_q = np.mean(q_values, axis=1)
    std_q = np.std(q_values, axis=1)

    # Compute CV with rolling window
    cv = pd.Series(std_q / (np.abs(mean_q) + 1e-8)).rolling(window=window_size).mean().values

    return cv


def compare_training_runs(
    metrics_list: List[TrainingMetrics],
    run_names: List[str],
) -> pd.DataFrame:
    """
    Compare multiple training runs.

    Args:
        metrics_list: List of TrainingMetrics objects
        run_names: List of run names

    Returns:
        DataFrame with comparison statistics
    """
    results = []

    for metrics, name in zip(metrics_list, run_names):
        convergence = analyze_convergence(metrics.episode_rewards)

        results.append({
            'run_name': name,
            'n_episodes': len(metrics.episode_rewards),
            'total_timesteps': metrics.timesteps[-1] if len(metrics.timesteps) > 0 else 0,
            'converged': convergence.converged,
            'convergence_episode': convergence.convergence_episode,
            'final_mean_reward': convergence.final_mean_reward,
            'final_std_reward': convergence.final_std_reward,
            'max_reward': np.max(metrics.episode_rewards),
            'min_reward': np.min(metrics.episode_rewards),
            'mean_reward': np.mean(metrics.episode_rewards),
            'median_reward': np.median(metrics.episode_rewards),
            'mean_episode_length': np.mean(metrics.episode_lengths),
        })

    return pd.DataFrame(results)
