"""
Visualization functions for analysis results and paper figures.

Provides publication-quality plotting functions with consistent styling.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from pathlib import Path

# Publication-quality styling
FIGURE_DPI = 300
FIGURE_FORMAT_RASTER = 'png'
FIGURE_FORMAT_VECTOR = 'pdf'

# Color schemes
COLORS_QUALITATIVE = sns.color_palette("Set2")
COLORS_SEQUENTIAL = sns.color_palette("viridis", as_cmap=False)
COLORS_DIVERGING = sns.color_palette("RdYlGn", as_cmap=False)

# Font settings
FONT_SIZE_SMALL = 10
FONT_SIZE_MEDIUM = 12
FONT_SIZE_LARGE = 14


def setup_publication_style():
    """Set up matplotlib styling for publication-quality figures."""
    plt.style.use('seaborn-v0_8-paper')

    mpl.rcParams.update({
        'figure.dpi': FIGURE_DPI,
        'savefig.dpi': FIGURE_DPI,
        'font.size': FONT_SIZE_MEDIUM,
        'axes.labelsize': FONT_SIZE_MEDIUM,
        'axes.titlesize': FONT_SIZE_LARGE,
        'xtick.labelsize': FONT_SIZE_SMALL,
        'ytick.labelsize': FONT_SIZE_SMALL,
        'legend.fontsize': FONT_SIZE_SMALL,
        'figure.titlesize': FONT_SIZE_LARGE,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'figure.figsize': (8, 6),
    })


def plot_learning_curves(
    metrics_dict: Dict[str, Any],
    window_size: int = 100,
    smooth: bool = True,
    show_ci: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot learning curves for multiple training runs.

    Args:
        metrics_dict: Dictionary mapping run names to TrainingMetrics objects
        window_size: Window size for smoothing
        smooth: Whether to apply smoothing
        show_ci: Whether to show confidence intervals
        ax: Matplotlib axis (creates new figure if None)

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        rewards = metrics.episode_rewards
        episodes = np.arange(len(rewards))

        color = COLORS_QUALITATIVE[idx % len(COLORS_QUALITATIVE)]

        if smooth:
            from .training import smooth_curve
            smoothed = smooth_curve(rewards, method="gaussian", window_size=window_size)
            ax.plot(episodes, smoothed, label=name, color=color, linewidth=2)

            # Show confidence interval
            if show_ci:
                std = pd.Series(rewards).rolling(window=window_size, min_periods=1).std()
                ci_lower = smoothed - 1.96 * std / np.sqrt(window_size)
                ci_upper = smoothed + 1.96 * std / np.sqrt(window_size)
                ax.fill_between(episodes, ci_lower, ci_upper, alpha=0.2, color=color)
        else:
            ax.plot(episodes, rewards, label=name, color=color, alpha=0.7)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Learning Curves Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_reward_components(
    reward_df: pd.DataFrame,
    window_size: int = 100,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot individual reward components over training.

    Args:
        reward_df: DataFrame with columns: episode, total_reward, similarity, structure, human
        window_size: Window size for smoothing
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    components = [col for col in reward_df.columns if col not in ['episode', 'total_reward']]

    for idx, component in enumerate(components):
        values = reward_df[component].values
        episodes = reward_df['episode'].values

        # Smooth
        from .training import smooth_curve
        smoothed = smooth_curve(values, method="gaussian", window_size=window_size)

        color = COLORS_QUALITATIVE[idx % len(COLORS_QUALITATIVE)]
        ax.plot(episodes, smoothed, label=component.capitalize(), color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Component Value')
    ax.set_title('Reward Components Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_action_distribution(
    action_counts: Dict[int, int],
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot action distribution (cluster utilization).

    Args:
        action_counts: Dictionary mapping action IDs to counts
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    total = sum(counts)
    probabilities = [c / total * 100 for c in counts]

    bars = ax.bar(actions, probabilities, color=COLORS_QUALITATIVE[0], alpha=0.7, edgecolor='black')

    # Highlight top actions
    top_k = 5
    top_indices = np.argsort(probabilities)[-top_k:]
    for idx in top_indices:
        bars[idx].set_color(COLORS_QUALITATIVE[1])
        bars[idx].set_alpha(0.9)

    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Utilization (%)')
    ax.set_title('Action (Cluster) Utilization Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line for uniform distribution
    uniform_prob = 100 / len(actions)
    ax.axhline(uniform_prob, color='red', linestyle='--', linewidth=2, label=f'Uniform ({uniform_prob:.1f}%)')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_baseline_comparison(
    evaluations: Dict[str, Any],
    metric: str = "mean_reward",
    show_error_bars: bool = True,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot bar chart comparing baseline and trained policies.

    Args:
        evaluations: Dictionary mapping policy names to PolicyEvaluation objects
        metric: Metric to compare ("mean_reward", "median_reward", "max_reward")
        show_error_bars: Whether to show standard error bars
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    policy_names = list(evaluations.keys())
    values = []
    errors = []

    for policy in policy_names:
        eval_obj = evaluations[policy]

        if metric == "mean_reward":
            values.append(eval_obj.statistics.mean)
            errors.append(eval_obj.statistics.std / np.sqrt(eval_obj.statistics.n_samples))
        elif metric == "median_reward":
            values.append(eval_obj.statistics.median)
            errors.append(eval_obj.statistics.iqr / 2)  # Semi-IQR as error
        elif metric == "max_reward":
            values.append(eval_obj.statistics.max)
            errors.append(0)

    x_pos = np.arange(len(policy_names))

    bars = ax.bar(x_pos, values, color=COLORS_QUALITATIVE[:len(policy_names)],
                   alpha=0.7, edgecolor='black')

    if show_error_bars and any(e > 0 for e in errors):
        ax.errorbar(x_pos, values, yerr=errors, fmt='none', color='black',
                    capsize=5, capthick=2, linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_names, rotation=15, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Policy Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_ablation_results(
    ablation_analysis: Dict[str, Any],
    sort_by: str = "performance_drop_pct",
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot ablation study results showing component importance.

    Args:
        ablation_analysis: Results from analyze_ablation_results
        sort_by: Metric to sort by ("performance_drop_pct", "effect_size")
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    baseline_reward = ablation_analysis["baseline"]["mean_reward"]

    # Extract ablation data
    ablations = ablation_analysis["ablations"]
    ablations_sorted = sorted(ablations, key=lambda x: x[sort_by], reverse=True)

    names = [a["name"] for a in ablations_sorted]
    drops = [a["performance_drop_pct"] for a in ablations_sorted]
    significant = [a["statistically_significant"] for a in ablations_sorted]

    y_pos = np.arange(len(names))

    # Color by significance
    colors = [COLORS_QUALITATIVE[1] if sig else COLORS_QUALITATIVE[0] for sig in significant]

    bars = ax.barh(y_pos, drops, color=colors, alpha=0.7, edgecolor='black')

    # Add markers for statistical significance
    for i, sig in enumerate(significant):
        if sig:
            ax.text(drops[i] + 0.5, i, '*', fontsize=16, fontweight='bold', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Performance Drop (%)')
    ax.set_title('Ablation Study: Component Importance')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS_QUALITATIVE[1], alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor=COLORS_QUALITATIVE[0], alpha=0.7, label='Not Significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return fig


def plot_statistical_comparison(
    comparison_results: Dict[Tuple[str, str], Any],
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot pairwise statistical comparison matrix.

    Args:
        comparison_results: Dictionary from compare_multiple_policies
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    # Extract policy names
    policy_names = set()
    for (p1, p2) in comparison_results.keys():
        policy_names.add(p1)
        policy_names.add(p2)

    policy_names = sorted(policy_names)
    n_policies = len(policy_names)

    # Create matrices for p-values and effect sizes
    p_matrix = np.ones((n_policies, n_policies))
    effect_matrix = np.zeros((n_policies, n_policies))

    for (p1, p2), comparison in comparison_results.items():
        i = policy_names.index(p1)
        j = policy_names.index(p2)

        p_matrix[i, j] = comparison.statistical_test.p_value
        p_matrix[j, i] = comparison.statistical_test.p_value

        effect_matrix[i, j] = comparison.statistical_test.effect_size
        effect_matrix[j, i] = -comparison.statistical_test.effect_size

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Plot effect size matrix with significance markers
    im = ax.imshow(effect_matrix, cmap='RdYlGn', vmin=-2, vmax=2, aspect='auto')

    # Add significance markers
    for i in range(n_policies):
        for j in range(n_policies):
            if i != j:
                text = f"{effect_matrix[i, j]:.2f}"
                if p_matrix[i, j] < 0.05:
                    text += "*"
                if p_matrix[i, j] < 0.01:
                    text += "*"

                ax.text(j, i, text, ha='center', va='center', fontsize=FONT_SIZE_SMALL,
                        color='black' if abs(effect_matrix[i, j]) < 1 else 'white')

    ax.set_xticks(np.arange(n_policies))
    ax.set_yticks(np.arange(n_policies))
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_yticklabels(policy_names)
    ax.set_title('Pairwise Comparison: Effect Sizes\n(* p<0.05, ** p<0.01)')

    plt.colorbar(im, ax=ax, label='Cohen\'s d (row - column)')
    plt.tight_layout()

    return fig


def plot_convergence_analysis(
    metrics: Any,
    window_size: int = 100,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot convergence analysis with CV and trend.

    Args:
        metrics: TrainingMetrics object
        window_size: Window size for analysis
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    else:
        fig = ax.figure
        ax1 = ax

    rewards = metrics.episode_rewards
    episodes = np.arange(len(rewards))

    # Plot rewards with moving average
    from .training import smooth_curve
    smoothed = smooth_curve(rewards, method="gaussian", window_size=window_size)

    ax1.plot(episodes, rewards, alpha=0.3, color='gray', label='Raw Rewards')
    ax1.plot(episodes, smoothed, color=COLORS_QUALITATIVE[0], linewidth=2, label='Smoothed')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Convergence Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Coefficient of variation
    if ax is None:
        cv = pd.Series(rewards).rolling(window=window_size).std() / pd.Series(rewards).rolling(window=window_size).mean().abs()

        ax2.plot(episodes[window_size-1:], cv[window_size-1:], color=COLORS_QUALITATIVE[1], linewidth=2)
        ax2.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Stability Threshold (CV=0.05)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_publication_figure(
    fig: Figure,
    output_path: Path,
    formats: List[str] = None,
    dpi: int = FIGURE_DPI,
) -> List[Path]:
    """
    Save figure in publication-quality formats.

    Args:
        fig: Matplotlib Figure object
        output_path: Output path (without extension)
        formats: List of formats to save ('png', 'pdf', 'svg')
        dpi: DPI for raster formats

    Returns:
        List of paths to saved files
    """
    if formats is None:
        formats = ['png', 'pdf']

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        file_path = output_path.with_suffix(f'.{fmt}')

        fig.savefig(
            file_path,
            dpi=dpi,
            bbox_inches='tight',
            format=fmt,
            transparent=False,
        )

        saved_paths.append(file_path)

    return saved_paths


def create_multi_panel_figure(
    plot_functions: List[callable],
    nrows: int,
    ncols: int,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Create multi-panel figure for comprehensive visualization.

    Args:
        plot_functions: List of plotting functions that take ax as argument
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size (width, height)

    Returns:
        Figure object
    """
    setup_publication_style()

    if figsize is None:
        figsize = (8 * ncols, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, plot_fn in enumerate(plot_functions):
        if idx < len(axes):
            plot_fn(axes[idx])

    # Hide unused axes
    for idx in range(len(plot_functions), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def plot_q_value_evolution(
    q_values: np.ndarray,
    window_size: int = 100,
    ax: Optional[Axes] = None,
) -> Figure:
    """
    Plot Q-value evolution and stability over training.

    Args:
        q_values: Array of Q-values (episodes x actions)
        window_size: Window for smoothing
        ax: Matplotlib axis

    Returns:
        Figure object
    """
    setup_publication_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if q_values.ndim == 1:
        q_values = q_values.reshape(-1, 1)

    episodes = np.arange(len(q_values))

    # Plot mean Q-value
    mean_q = np.mean(q_values, axis=1)
    std_q = np.std(q_values, axis=1)

    from .training import smooth_curve
    smoothed_mean = smooth_curve(mean_q, method="gaussian", window_size=window_size)
    smoothed_std = smooth_curve(std_q, method="gaussian", window_size=window_size)

    ax.plot(episodes, smoothed_mean, color=COLORS_QUALITATIVE[0], linewidth=2, label='Mean Q-value')
    ax.fill_between(episodes, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                     alpha=0.3, color=COLORS_QUALITATIVE[0], label='± 1 STD')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value')
    ax.set_title('Q-value Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
