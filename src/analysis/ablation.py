"""
Ablation study utilities for systematic component removal.

Provides functions to create ablation configs, run experiments,
and analyze the impact of removing individual components.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

from .training import TrainingMetrics
from .metrics import compute_reward_statistics, calculate_statistical_significance


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""

    name: str
    description: str
    base_config: Dict[str, Any]
    modifications: Dict[str, Any]
    seed: int


@dataclass
class AblationResult:
    """Results from an ablation experiment."""

    config_name: str
    description: str
    metrics: TrainingMetrics
    final_mean_reward: float
    final_std_reward: float
    convergence_episode: Optional[int]
    modifications: Dict[str, Any]


def create_reward_component_ablations(
    base_config_path: Path,
    output_dir: Path,
    seeds: List[int] = None,
) -> List[AblationConfig]:
    """
    Create ablation configs for reward component removal.

    Creates configs with each reward component (w1, w2, w3) set to 0.

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save ablation configs
        seeds: List of random seeds for multiple runs

    Returns:
        List of AblationConfig objects
    """
    if seeds is None:
        seeds = [0]

    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    ablation_configs = []

    # Get current weights
    w1 = base_config.get("reward_weights", {}).get("w1", 0.4)
    w2 = base_config.get("reward_weights", {}).get("w2", 0.3)
    w3 = base_config.get("reward_weights", {}).get("w3", 0.3)

    # Ablation 1: Remove similarity (w1 = 0)
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["reward_weights"]["w1"] = 0.0
        # Renormalize remaining weights
        total = config["reward_weights"]["w2"] + config["reward_weights"]["w3"]
        if total > 0:
            config["reward_weights"]["w2"] = w2 / total
            config["reward_weights"]["w3"] = w3 / total

        ablation_configs.append(AblationConfig(
            name=f"no_similarity_seed{seed}",
            description="Ablation: Remove similarity reward component (w1=0)",
            base_config=base_config,
            modifications={"reward_weights": config["reward_weights"]},
            seed=seed,
        ))

        # Save config
        output_path = output_dir / f"ablation_no_similarity_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Ablation 2: Remove structure (w2 = 0)
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["reward_weights"]["w2"] = 0.0
        # Renormalize
        total = config["reward_weights"]["w1"] + config["reward_weights"]["w3"]
        if total > 0:
            config["reward_weights"]["w1"] = w1 / total
            config["reward_weights"]["w3"] = w3 / total

        ablation_configs.append(AblationConfig(
            name=f"no_structure_seed{seed}",
            description="Ablation: Remove structure reward component (w2=0)",
            base_config=base_config,
            modifications={"reward_weights": config["reward_weights"]},
            seed=seed,
        ))

        output_path = output_dir / f"ablation_no_structure_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Ablation 3: Remove human feedback (w3 = 0) - equivalent to non-interactive
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["reward_weights"]["w3"] = 0.0
        config["non_interactive_mode"] = True
        # Renormalize
        total = config["reward_weights"]["w1"] + config["reward_weights"]["w2"]
        if total > 0:
            config["reward_weights"]["w1"] = w1 / total
            config["reward_weights"]["w2"] = w2 / total

        ablation_configs.append(AblationConfig(
            name=f"no_human_seed{seed}",
            description="Ablation: Remove human feedback component (w3=0)",
            base_config=base_config,
            modifications={
                "reward_weights": config["reward_weights"],
                "non_interactive_mode": True,
            },
            seed=seed,
        ))

        output_path = output_dir / f"ablation_no_human_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    return ablation_configs


def create_architecture_ablations(
    base_config_path: Path,
    output_dir: Path,
    seeds: List[int] = None,
) -> List[AblationConfig]:
    """
    Create ablation configs for architecture changes.

    Creates configs with:
    - LSTM vs MLP feature extractor
    - Different network sizes

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save ablation configs
        seeds: List of random seeds

    Returns:
        List of AblationConfig objects
    """
    if seeds is None:
        seeds = [0]

    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    ablation_configs = []

    # Ablation 1: MLP instead of LSTM
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["lstm_q_net"] = False

        ablation_configs.append(AblationConfig(
            name=f"mlp_policy_seed{seed}",
            description="Ablation: Use MLP policy instead of LSTM",
            base_config=base_config,
            modifications={"lstm_q_net": False},
            seed=seed,
        ))

        output_path = output_dir / f"ablation_mlp_policy_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Ablation 2: Smaller network
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["lstm_layer_params"] = [32]  # Half size
        config["input_fc_layer_params"] = [32]
        config["output_fc_layer_params"] = [32]

        ablation_configs.append(AblationConfig(
            name=f"small_network_seed{seed}",
            description="Ablation: Use smaller network (32 units)",
            base_config=base_config,
            modifications={
                "lstm_layer_params": [32],
                "input_fc_layer_params": [32],
                "output_fc_layer_params": [32],
            },
            seed=seed,
        ))

        output_path = output_dir / f"ablation_small_network_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    # Ablation 3: Larger network
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["lstm_layer_params"] = [128]  # Double size
        config["input_fc_layer_params"] = [128]
        config["output_fc_layer_params"] = [128]

        ablation_configs.append(AblationConfig(
            name=f"large_network_seed{seed}",
            description="Ablation: Use larger network (128 units)",
            base_config=base_config,
            modifications={
                "lstm_layer_params": [128],
                "input_fc_layer_params": [128],
                "output_fc_layer_params": [128],
            },
            seed=seed,
        ))

        output_path = output_dir / f"ablation_large_network_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    return ablation_configs


def create_ghsom_ablation(
    base_config_path: Path,
    output_dir: Path,
    n_random_clusters: int = 10,
    seeds: List[int] = None,
) -> List[AblationConfig]:
    """
    Create ablation config for GHSOM vs random clustering.

    Note: This requires implementation changes, not just config changes.
    This function documents the intended ablation for manual implementation.

    Args:
        base_config_path: Path to base configuration file
        output_dir: Directory to save ablation configs
        n_random_clusters: Number of random clusters to use
        seeds: List of random seeds

    Returns:
        List of AblationConfig objects with metadata
    """
    if seeds is None:
        seeds = [0]

    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    ablation_configs = []

    for seed in seeds:
        config = copy.deepcopy(base_config)

        # Add metadata for implementation
        config["clustering_method"] = "random"  # Instead of GHSOM
        config["n_clusters"] = n_random_clusters

        ablation_configs.append(AblationConfig(
            name=f"random_clustering_seed{seed}",
            description=f"Ablation: Use random clustering ({n_random_clusters} clusters) instead of GHSOM",
            base_config=base_config,
            modifications={
                "clustering_method": "random",
                "n_clusters": n_random_clusters,
            },
            seed=seed,
        ))

        output_path = output_dir / f"ablation_random_clustering_seed{seed}.json"
        config["seed"] = seed
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

    return ablation_configs


def run_ablation_experiment(
    config: AblationConfig,
    train_fn: callable,
    eval_fn: callable,
) -> AblationResult:
    """
    Run a single ablation experiment.

    Args:
        config: AblationConfig for the experiment
        train_fn: Training function that takes config dict and returns model
        eval_fn: Evaluation function that takes model and returns TrainingMetrics

    Returns:
        AblationResult with experiment results
    """
    # Apply modifications to base config
    experiment_config = copy.deepcopy(config.base_config)
    experiment_config.update(config.modifications)
    experiment_config["seed"] = config.seed

    # Train model
    model = train_fn(experiment_config)

    # Evaluate
    metrics = eval_fn(model)

    # Analyze convergence
    from .training import analyze_convergence
    convergence = analyze_convergence(metrics.episode_rewards)

    return AblationResult(
        config_name=config.name,
        description=config.description,
        metrics=metrics,
        final_mean_reward=convergence.final_mean_reward,
        final_std_reward=convergence.final_std_reward,
        convergence_episode=convergence.convergence_episode,
        modifications=config.modifications,
    )


def analyze_ablation_results(
    baseline_result: AblationResult,
    ablation_results: List[AblationResult],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Analyze ablation results compared to baseline.

    Args:
        baseline_result: Baseline (full model) result
        ablation_results: List of ablation results
        alpha: Significance level for tests

    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "baseline": {
            "name": baseline_result.config_name,
            "mean_reward": baseline_result.final_mean_reward,
            "std_reward": baseline_result.final_std_reward,
        },
        "ablations": [],
    }

    baseline_rewards = baseline_result.metrics.episode_rewards

    for ablation in ablation_results:
        ablation_rewards = ablation.metrics.episode_rewards

        # Statistical test
        test_result = calculate_statistical_significance(
            baseline_rewards,
            ablation_rewards,
            alpha=alpha,
        )

        # Compute performance drop
        performance_drop = (
            (baseline_result.final_mean_reward - ablation.final_mean_reward) /
            abs(baseline_result.final_mean_reward) * 100
            if baseline_result.final_mean_reward != 0
            else 0.0
        )

        analysis["ablations"].append({
            "name": ablation.config_name,
            "description": ablation.description,
            "modifications": ablation.modifications,
            "mean_reward": ablation.final_mean_reward,
            "std_reward": ablation.final_std_reward,
            "performance_drop_pct": performance_drop,
            "statistically_significant": test_result.significant,
            "p_value": test_result.p_value,
            "effect_size": test_result.effect_size,
            "effect_size_ci": test_result.effect_size_ci,
        })

    return analysis


def rank_component_importance(
    ablation_analysis: Dict[str, Any],
) -> List[Tuple[str, float]]:
    """
    Rank components by importance based on ablation results.

    Args:
        ablation_analysis: Results from analyze_ablation_results

    Returns:
        List of (component_name, importance_score) tuples, sorted by importance
    """
    rankings = []

    for ablation in ablation_analysis["ablations"]:
        # Importance score = performance drop * effect size
        importance = (
            ablation["performance_drop_pct"] *
            abs(ablation["effect_size"])
        )

        rankings.append((ablation["name"], importance))

    # Sort by importance (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)

    return rankings


def save_ablation_analysis(
    analysis: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save ablation analysis results to JSON.

    Args:
        analysis: Analysis results from analyze_ablation_results
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tuples to lists for JSON serialization
    serializable_analysis = copy.deepcopy(analysis)

    for ablation in serializable_analysis["ablations"]:
        if "effect_size_ci" in ablation:
            ablation["effect_size_ci"] = list(ablation["effect_size_ci"])

    with open(output_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)


def create_ablation_config(
    base_config: Dict[str, Any],
    modifications: Dict[str, Any],
    name: str,
    description: str = "",
) -> Dict[str, Any]:
    """
    Create an ablation config by modifying a base config.

    Args:
        base_config: Base configuration dictionary
        modifications: Dictionary of modifications to apply
        name: Name for this ablation
        description: Description of what's being ablated

    Returns:
        Modified configuration dictionary
    """
    config = copy.deepcopy(base_config)
    config.update(modifications)

    # Add metadata
    config["ablation_metadata"] = {
        "name": name,
        "description": description,
        "modifications": modifications,
    }

    return config
