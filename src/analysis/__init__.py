"""
Analysis module for Multi-Agent RL Music Generation PoC.

Provides reusable functions for training analysis, baseline comparisons,
ablation studies, computational benchmarks, and result visualization.
"""

from .metrics import (
    calculate_statistical_significance,
    compute_effect_size,
    calculate_confidence_interval,
    compute_reward_statistics,
)
from .training import (
    load_training_metrics,
    analyze_convergence,
    compute_episode_statistics,
)
from .baseline import (
    evaluate_random_policy,
    evaluate_uniform_policy,
    compare_policies,
)
from .ablation import (
    create_ablation_config,
    run_ablation_experiment,
)
from .visualization import (
    plot_learning_curves,
    plot_reward_components,
    plot_action_distribution,
    save_publication_figure,
)

__all__ = [
    # Metrics
    "calculate_statistical_significance",
    "compute_effect_size",
    "calculate_confidence_interval",
    "compute_reward_statistics",
    # Training
    "load_training_metrics",
    "analyze_convergence",
    "compute_episode_statistics",
    # Baseline
    "evaluate_random_policy",
    "evaluate_uniform_policy",
    "compare_policies",
    # Ablation
    "create_ablation_config",
    "run_ablation_experiment",
    # Visualization
    "plot_learning_curves",
    "plot_reward_components",
    "plot_action_distribution",
    "save_publication_figure",
]
