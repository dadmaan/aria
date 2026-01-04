"""
Baseline policy evaluation and comparison functions.

Provides functions to evaluate random and uniform policies,
and compare them against trained DQN policies.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from pathlib import Path
from dataclasses import dataclass
import json

from .metrics import (
    compute_reward_statistics,
    calculate_statistical_significance,
    StatisticalTestResult,
    RewardStatistics,
)


@dataclass
class PolicyEvaluation:
    """Results from policy evaluation."""

    policy_name: str
    n_episodes: int
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    action_counts: Dict[int, int]
    statistics: RewardStatistics
    metadata: Dict[str, Any]


@dataclass
class PolicyComparison:
    """Comparison between two policies."""

    policy1_name: str
    policy2_name: str
    policy1_stats: RewardStatistics
    policy2_stats: RewardStatistics
    statistical_test: StatisticalTestResult
    improvement_pct: float


def evaluate_random_policy(
    env: gym.Env,
    n_episodes: int = 100,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> PolicyEvaluation:
    """
    Evaluate a random policy (uniform random action selection).

    Args:
        env: Gymnasium environment
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode (None = env default)
        seed: Random seed for reproducibility

    Returns:
        PolicyEvaluation with results
    """
    if seed is not None:
        np.random.seed(seed)

    episode_rewards = []
    episode_lengths = []
    action_counts = {}

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()

            # Track action
            action_counts[action] = action_counts.get(action, 0) + 1

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if max_steps is not None and episode_length >= max_steps:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    episode_rewards_arr = np.array(episode_rewards)
    episode_lengths_arr = np.array(episode_lengths)

    statistics = compute_reward_statistics(episode_rewards_arr)

    return PolicyEvaluation(
        policy_name="random",
        n_episodes=n_episodes,
        episode_rewards=episode_rewards_arr,
        episode_lengths=episode_lengths_arr,
        action_counts=action_counts,
        statistics=statistics,
        metadata={
            "seed": seed,
            "max_steps": max_steps,
        }
    )


def evaluate_uniform_policy(
    env: gym.Env,
    n_episodes: int = 100,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> PolicyEvaluation:
    """
    Evaluate a uniform policy (cyclic action selection).

    Args:
        env: Gymnasium environment
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode (None = env default)
        seed: Random seed for reproducibility

    Returns:
        PolicyEvaluation with results
    """
    if seed is not None:
        np.random.seed(seed)

    n_actions = env.action_space.n

    episode_rewards = []
    episode_lengths = []
    action_counts = {i: 0 for i in range(n_actions)}

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0
        episode_length = 0
        done = False
        action_idx = 0

        while not done:
            # Uniform cyclic action
            action = action_idx % n_actions
            action_idx += 1

            # Track action
            action_counts[action] += 1

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if max_steps is not None and episode_length >= max_steps:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    episode_rewards_arr = np.array(episode_rewards)
    episode_lengths_arr = np.array(episode_lengths)

    statistics = compute_reward_statistics(episode_rewards_arr)

    return PolicyEvaluation(
        policy_name="uniform",
        n_episodes=n_episodes,
        episode_rewards=episode_rewards_arr,
        episode_lengths=episode_lengths_arr,
        action_counts=action_counts,
        statistics=statistics,
        metadata={
            "seed": seed,
            "max_steps": max_steps,
            "n_actions": n_actions,
        }
    )


def evaluate_trained_policy(
    env: gym.Env,
    model,  # Stable-Baselines3 model
    n_episodes: int = 100,
    max_steps: Optional[int] = None,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> PolicyEvaluation:
    """
    Evaluate a trained SB3 policy.

    Args:
        env: Gymnasium environment
        model: Trained Stable-Baselines3 model
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy
        seed: Random seed for reproducibility

    Returns:
        PolicyEvaluation with results
    """
    if seed is not None:
        np.random.seed(seed)

    episode_rewards = []
    episode_lengths = []
    action_counts = {}

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode if seed is not None else None)
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Predict action
            action, _states = model.predict(obs, deterministic=deterministic)

            # Convert to int if needed
            if isinstance(action, np.ndarray):
                action = int(action.item())

            # Track action
            action_counts[action] = action_counts.get(action, 0) + 1

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if max_steps is not None and episode_length >= max_steps:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    episode_rewards_arr = np.array(episode_rewards)
    episode_lengths_arr = np.array(episode_lengths)

    statistics = compute_reward_statistics(episode_rewards_arr)

    return PolicyEvaluation(
        policy_name="trained_dqn",
        n_episodes=n_episodes,
        episode_rewards=episode_rewards_arr,
        episode_lengths=episode_lengths_arr,
        action_counts=action_counts,
        statistics=statistics,
        metadata={
            "seed": seed,
            "max_steps": max_steps,
            "deterministic": deterministic,
        }
    )


def compare_policies(
    policy1: PolicyEvaluation,
    policy2: PolicyEvaluation,
    alpha: float = 0.05,
) -> PolicyComparison:
    """
    Compare two policies using statistical tests.

    Args:
        policy1: First policy evaluation
        policy2: Second policy evaluation
        alpha: Significance level

    Returns:
        PolicyComparison with test results
    """
    # Perform statistical test
    test_result = calculate_statistical_significance(
        policy1.episode_rewards,
        policy2.episode_rewards,
        alpha=alpha,
    )

    # Calculate improvement percentage
    if policy2.statistics.mean != 0:
        improvement_pct = (
            (policy1.statistics.mean - policy2.statistics.mean) /
            abs(policy2.statistics.mean) * 100
        )
    else:
        improvement_pct = np.inf if policy1.statistics.mean > 0 else -np.inf

    return PolicyComparison(
        policy1_name=policy1.policy_name,
        policy2_name=policy2.policy_name,
        policy1_stats=policy1.statistics,
        policy2_stats=policy2.statistics,
        statistical_test=test_result,
        improvement_pct=improvement_pct,
    )


def compare_multiple_policies(
    evaluations: List[PolicyEvaluation],
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], PolicyComparison]:
    """
    Perform pairwise comparisons between multiple policies.

    Args:
        evaluations: List of PolicyEvaluation objects
        alpha: Significance level

    Returns:
        Dictionary mapping (policy1_name, policy2_name) to PolicyComparison
    """
    comparisons = {}

    for i in range(len(evaluations)):
        for j in range(i + 1, len(evaluations)):
            policy1 = evaluations[i]
            policy2 = evaluations[j]

            comparison = compare_policies(policy1, policy2, alpha=alpha)
            comparisons[(policy1.policy_name, policy2.policy_name)] = comparison

    return comparisons


def compute_action_distribution_similarity(
    eval1: PolicyEvaluation,
    eval2: PolicyEvaluation,
    metric: str = "kl_divergence",
) -> float:
    """
    Compute similarity between action distributions of two policies.

    Args:
        eval1: First policy evaluation
        eval2: Second policy evaluation
        metric: Similarity metric ("kl_divergence", "js_divergence", "cosine")

    Returns:
        Similarity score
    """
    # Get all possible actions
    all_actions = set(eval1.action_counts.keys()) | set(eval2.action_counts.keys())

    # Convert counts to probabilities
    total1 = sum(eval1.action_counts.values())
    total2 = sum(eval2.action_counts.values())

    p = np.array([eval1.action_counts.get(a, 0) / total1 for a in sorted(all_actions)])
    q = np.array([eval2.action_counts.get(a, 0) / total2 for a in sorted(all_actions)])

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # Renormalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    if metric == "kl_divergence":
        # KL(P || Q)
        return np.sum(p * np.log(p / q))

    elif metric == "js_divergence":
        # Jensen-Shannon divergence (symmetric)
        m = (p + q) / 2
        return (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2

    elif metric == "cosine":
        # Cosine similarity
        return np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))

    else:
        raise ValueError(f"Unknown metric: {metric}")


def save_evaluation_results(
    evaluation: PolicyEvaluation,
    output_path: Path,
) -> None:
    """
    Save evaluation results to JSON.

    Args:
        evaluation: PolicyEvaluation object
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "policy_name": evaluation.policy_name,
        "n_episodes": evaluation.n_episodes,
        "episode_rewards": evaluation.episode_rewards.tolist(),
        "episode_lengths": evaluation.episode_lengths.tolist(),
        "action_counts": evaluation.action_counts,
        "statistics": {
            "mean": evaluation.statistics.mean,
            "std": evaluation.statistics.std,
            "median": evaluation.statistics.median,
            "min": evaluation.statistics.min,
            "max": evaluation.statistics.max,
            "q25": evaluation.statistics.q25,
            "q75": evaluation.statistics.q75,
            "iqr": evaluation.statistics.iqr,
            "n_samples": evaluation.statistics.n_samples,
            "ci_95": evaluation.statistics.ci_95,
        },
        "metadata": evaluation.metadata,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_evaluation_results(input_path: Path) -> PolicyEvaluation:
    """
    Load evaluation results from JSON.

    Args:
        input_path: Input file path

    Returns:
        PolicyEvaluation object
    """
    input_path = Path(input_path)

    with open(input_path, 'r') as f:
        data = json.load(f)

    from .metrics import RewardStatistics

    statistics = RewardStatistics(
        mean=data["statistics"]["mean"],
        std=data["statistics"]["std"],
        median=data["statistics"]["median"],
        min=data["statistics"]["min"],
        max=data["statistics"]["max"],
        q25=data["statistics"]["q25"],
        q75=data["statistics"]["q75"],
        iqr=data["statistics"]["iqr"],
        n_samples=data["statistics"]["n_samples"],
        ci_95=tuple(data["statistics"]["ci_95"]),
    )

    return PolicyEvaluation(
        policy_name=data["policy_name"],
        n_episodes=data["n_episodes"],
        episode_rewards=np.array(data["episode_rewards"]),
        episode_lengths=np.array(data["episode_lengths"]),
        action_counts=data["action_counts"],
        statistics=statistics,
        metadata=data["metadata"],
    )
