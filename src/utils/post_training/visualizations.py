"""Post-training visualization functions for agent learning trajectories.

Provides 4 core visualizations:
1. State visitation heatmap (temporal snapshots)
2. Hierarchy navigation tree (layer-wise exploration)
3. Reward-per-cluster analysis (which states are valuable)
4. Reward component evolution (component breakdown over time)

Design principles:
- Simple, not complex
- Conference-ready aesthetics
- Flexible for different GHSOM sizes
- Handles missing data gracefully
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loader import TrainingRunData


# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

COLORS_SEQUENTIAL = sns.color_palette("viridis", as_cmap=False)
COLORS_DIVERGING = sns.color_palette("RdYlGn", as_cmap=False)
COLORS_COMPONENTS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

FONT_SIZE_SMALL = 9
FONT_SIZE_MEDIUM = 11
FONT_SIZE_LARGE = 13


def setup_plot_style():
    """Apply consistent styling to all plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.dpi': 100,
        'font.size': FONT_SIZE_MEDIUM,
        'axes.labelsize': FONT_SIZE_MEDIUM,
        'axes.titlesize': FONT_SIZE_LARGE,
        'xtick.labelsize': FONT_SIZE_SMALL,
        'ytick.labelsize': FONT_SIZE_SMALL,
        'legend.fontsize': FONT_SIZE_SMALL,
    })


# ============================================================================
# VIZ 1: STATE VISITATION HEATMAP
# ============================================================================

def plot_state_visitation_heatmap(
    data: TrainingRunData,
    phases: Optional[List[Tuple[int, int]]] = None,
    phase_names: Optional[List[str]] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    figsize: Tuple[int, int] = (15, 4),
    cmap: str = 'viridis',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot state visitation heatmap across training phases.

    Shows which GHSOM clusters the agent visited during different phases
    of training (e.g., early exploration, mid-training, late convergence).

    Args:
        data: TrainingRunData object
        phases: List of (start_episode, end_episode) tuples for temporal snapshots
                Default: [(0, N/4), (N/2-N/8, N/2+N/8), (3N/4, N)]
        phase_names: Names for each phase (default: "Early", "Mid", "Late")
        grid_shape: (rows, cols) for cluster grid layout
                    If None, auto-compute near-square grid
        figsize: Figure size
        cmap: Colormap name
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure

    Raises:
        ValueError: If episode_sequences is empty (missing data)
    """
    setup_plot_style()

    if not data.episode_sequences:
        raise ValueError(
            "Episode sequences not available. "
            "Re-run training with updated comprehensive_metrics_callback."
        )

    n_episodes = data.total_episodes
    n_clusters = data.total_clusters

    # Default phases: early, mid, late
    if phases is None:
        phases = [
            (0, n_episodes // 4),
            (n_episodes // 2 - n_episodes // 8, n_episodes // 2 + n_episodes // 8),
            (3 * n_episodes // 4, n_episodes),
        ]

    if phase_names is None:
        phase_names = ["Early Training", "Mid Training", "Late Training"]

    # Auto-compute grid shape if not provided
    if grid_shape is None:
        rows = int(np.ceil(np.sqrt(n_clusters)))
        cols = int(np.ceil(n_clusters / rows))
        grid_shape = (rows, cols)

    # Create subplots
    fig, axes = plt.subplots(1, len(phases), figsize=figsize)
    if len(phases) == 1:
        axes = [axes]

    for ax, (start, end), phase_name in zip(axes, phases, phase_names):
        # Count cluster visits in this phase
        visit_counts = np.zeros(n_clusters)

        for ep_idx in range(max(0, start), min(end, len(data.episode_sequences))):
            sequence = data.episode_sequences[ep_idx]
            for cluster_id in sequence:
                if 0 <= cluster_id < n_clusters:
                    visit_counts[cluster_id] += 1

        # Pad to fit grid
        grid_size = grid_shape[0] * grid_shape[1]
        if n_clusters < grid_size:
            visit_counts = np.pad(
                visit_counts,
                (0, grid_size - n_clusters),
                constant_values=np.nan
            )

        # Reshape to grid
        grid = visit_counts[:grid_size].reshape(grid_shape)

        # Plot heatmap
        im = ax.imshow(grid, cmap=cmap, aspect='auto')
        ax.set_title(f"{phase_name}\n(Episodes {start}-{end})")
        ax.set_xlabel("Cluster Column")
        ax.set_ylabel("Cluster Row")

        # Colorbar
        plt.colorbar(im, ax=ax, label="Visit Count")

        # Optional: annotate with cluster IDs
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                cluster_id = i * grid_shape[1] + j
                if cluster_id < n_clusters:
                    # Use white text on dark background, black on light
                    text_color = "white" if grid[i, j] > np.nanmean(grid) else "black"
                    ax.text(
                        j, i, str(cluster_id),
                        ha="center", va="center",
                        color=text_color,
                        fontsize=7
                    )

    plt.suptitle(f"State Visitation Heatmap: {data.run_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


# ============================================================================
# VIZ 2: HIERARCHY NAVIGATION TREE
# ============================================================================

def plot_hierarchy_navigation(
    data: TrainingRunData,
    max_depth: int = 3,
    min_visits: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    layout: str = 'spring',
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot GHSOM hierarchy as tree showing layer-wise navigation patterns.

    Visualizes the hierarchical structure with node sizes proportional to
    visitation frequency. Helps understand if agent explores all layers
    or gets stuck in specific branches.

    Args:
        data: TrainingRunData object
        max_depth: Maximum hierarchy depth to display
        min_visits: Minimum visits to display a node (reduces clutter)
        figsize: Figure size
        layout: Layout algorithm ('spring', 'hierarchical')
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    setup_plot_style()

    if not data.episode_sequences:
        raise ValueError("Episode sequences not available.")

    # Count visits per cluster
    all_visits = Counter()
    for sequence in data.episode_sequences:
        all_visits.update(sequence)

    # Get hierarchy info from GHSOM manager
    ghsom_mgr = data.ghsom_manager
    if ghsom_mgr is None:
        raise ValueError("GHSOM manager not loaded. Set load_ghsom=True.")

    # Use networkx for tree layout
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx required for hierarchy visualization: pip install networkx")

    # Build graph with hierarchy information
    G = nx.DiGraph()
    node_levels = {}

    # Add nodes from lookup table
    for cluster_id, node in ghsom_mgr.lookup_table.items():
        if cluster_id == 'root':
            continue

        visits = all_visits.get(cluster_id, 0)
        level = node.level

        # Filter by depth and visits
        if level <= max_depth and visits >= min_visits:
            G.add_node(
                cluster_id,
                level=level,
                visits=visits,
            )
            node_levels[cluster_id] = level

    # Note: For simplicity, we're showing nodes without explicit edges
    # Full parent-child relationships would require additional GHSOM metadata

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if len(G.nodes()) == 0:
        ax.text(
            0.5, 0.5,
            "No nodes meet criteria\n(try lowering min_visits or increasing max_depth)",
            ha='center', va='center', fontsize=14
        )
        ax.set_title(f"GHSOM Hierarchy Navigation: {data.run_name}")
        ax.axis('off')
        return fig

    # Layout
    if layout == 'hierarchical' and node_levels:
        # Group by level for hierarchical layout
        pos = {}
        level_counts = Counter(node_levels.values())
        level_positions = {level: 0 for level in level_counts.keys()}

        for node_id in sorted(G.nodes(), key=lambda n: (node_levels.get(n, 0), n)):
            level = node_levels.get(node_id, 0)
            x = level_positions[level]
            y = -level  # Negative so higher levels are at top
            pos[node_id] = (x, y)
            level_positions[level] += 1
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes with size proportional to visits
    node_sizes = [G.nodes[n]['visits'] * 10 for n in G.nodes()]
    node_colors = [G.nodes[n]['level'] for n in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap='viridis',
        alpha=0.8,
        vmin=0,
        vmax=max_depth,
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: str(n) for n in G.nodes()},
        font_size=8,
    )

    ax.set_title(f"GHSOM Hierarchy Navigation: {data.run_name}")
    ax.axis('off')

    # Legend for levels
    if node_levels:
        unique_levels = sorted(set(node_levels.values()))
        legend_elements = [
            mpatches.Patch(
                color=plt.cm.viridis(level / max(unique_levels)),
                label=f"Level {level}"
            )
            for level in unique_levels
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


# ============================================================================
# VIZ 3: REWARD PER CLUSTER
# ============================================================================

def plot_reward_per_cluster(
    data: TrainingRunData,
    top_k: int = 15,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot average reward when visiting each cluster.

    Shows which states (clusters) are most valuable. Helps identify
    if agent learns to favor high-reward states.

    Args:
        data: TrainingRunData object
        top_k: Number of top clusters to display
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    setup_plot_style()

    if not data.episode_sequences:
        raise ValueError("Episode sequences not available.")

    # Compute average reward per cluster
    cluster_rewards: Dict[int, List[float]] = {}

    for ep_idx, sequence in enumerate(data.episode_sequences):
        if ep_idx >= len(data.episode_rewards):
            break

        ep_reward = data.episode_rewards[ep_idx]

        for cluster_id in set(sequence):  # unique clusters in episode
            cluster_rewards.setdefault(cluster_id, []).append(ep_reward)

    # Compute mean and std
    cluster_stats = []
    for cluster_id, rewards in cluster_rewards.items():
        cluster_stats.append({
            'cluster_id': cluster_id,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'visit_count': len(rewards),
        })

    df = pd.DataFrame(cluster_stats)
    df = df.sort_values('mean_reward', ascending=False).head(top_k)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    bars = ax.bar(
        x, df['mean_reward'],
        yerr=df['std_reward'],
        capsize=5,
        color=COLORS_SEQUENTIAL[3],
        alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df['cluster_id'], rotation=45)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title(f'Top {top_k} Clusters by Average Reward: {data.run_name}')
    ax.grid(axis='y', alpha=0.3)

    # Annotate visit counts
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(
            i, row['mean_reward'] + row['std_reward'],
            f"n={row['visit_count']}",
            ha='center', va='bottom', fontsize=7
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


# ============================================================================
# VIZ 4: REWARD COMPONENT EVOLUTION
# ============================================================================

def plot_reward_component_evolution(
    data: TrainingRunData,
    window_size: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot evolution of reward components over training.

    Shows how different reward components (diversity, structure, transition)
    change as agent learns. Helps understand what the agent optimizes for.

    Args:
        data: TrainingRunData object
        window_size: Smoothing window size
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    setup_plot_style()

    if not data.reward_components:
        raise ValueError("Reward components not available.")

    fig, ax = plt.subplots(figsize=figsize)

    episodes = np.arange(data.total_episodes)

    for idx, (component_name, values) in enumerate(data.reward_components.items()):
        # Smooth with rolling mean
        if len(values) >= window_size:
            smoothed = pd.Series(values).rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).mean()
        else:
            smoothed = values

        color = COLORS_COMPONENTS[idx % len(COLORS_COMPONENTS)]
        ax.plot(
            episodes, smoothed,
            label=component_name.capitalize(),
            color=color,
            linewidth=2
        )

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Component Value')
    ax.set_title(f'Reward Component Evolution: {data.run_name}')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig
