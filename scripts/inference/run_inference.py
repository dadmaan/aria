#!/usr/bin/env python3
"""Tianshou-native inference script for trained agents.

This script generates musical sequences from trained Tianshou agents,
supporting both batch generation and interactive human-in-the-loop modes.

Features:
    - Tianshou checkpoint loading (DQN, C51, Rainbow variants)
    - DRQN hidden state management for recurrent networks
    - Interactive human feedback collection
    - Batch sequence generation with analysis
    - MIDI export and visualization

Usage:
    # Basic inference
    python scripts/training/run_inference.py \\
        --checkpoint artifacts/training/run_drqn_*/checkpoints/final.pth \\
        --config configs/agent_config.yaml

    # Interactive HIL mode
    python scripts/training/run_inference.py \\
        --checkpoint PATH \\
        --mode interactive \\
        --num-iterations 10

    # Batch generation with analysis
    python scripts/training/run_inference.py \\
        --checkpoint PATH \\
        --mode batch \\
        --num-sequences 100 \\
        --output outputs/generated_sequences

Example:
    python scripts/training/run_inference.py \\
        --checkpoint artifacts/training/run_drqn_20251207_195352/checkpoints/final.pth \\
        --config configs/agent_config.yaml \\
        --num-sequences 5 \\
        --output outputs/inference_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import gymnasium as gym

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table


# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.agents.ghsom_perceiving_agent import GHSOMPerceivingAgent
from src.agents.cluster_feature_mapper import ClusterFeatureMapper
from src.environments.music_env_gym import MusicGenerationGymEnv
from src.ghsom_manager import GHSOMManager
from src.training.tianshou_trainer import TianshouTrainer
from src.utils.human.human_feedback import HumanFeedbackCollector
from src.utils.logging.logging_manager import get_logger

logger = get_logger("run_inference")
console = Console()


class FeatureObservationWrapper(gym.Wrapper):
    """Wrapper that converts cluster ID observations to feature vectors.

    This wrapper takes the base environment's observation (sequence of cluster IDs)
    and converts each cluster ID to its corresponding feature vector using
    the ClusterFeatureMapper.
    """

    def __init__(self, env: gym.Env, mapper: ClusterFeatureMapper):
        """Initialize the wrapper.

        Args:
            env: Base gymnasium environment
            mapper: ClusterFeatureMapper instance
        """
        super().__init__(env)
        self.mapper = mapper

        # Update observation space to feature vectors
        seq_length = env.observation_space.shape[0]
        feature_dim = mapper.feature_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_length, feature_dim),
            dtype=np.float32,
        )

    def _convert_observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert cluster ID observation to feature observation."""
        return self.mapper.map_sequence(obs.tolist())

    def reset(self, **kwargs):
        """Reset the environment and convert observation."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_observation(obs), info

    def step(self, action):
        """Take a step and convert observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_observation(obs), reward, terminated, truncated, info


@dataclass
class GenerationResult:
    """Result of a single sequence generation."""

    sequence: List[int]
    episode_reward: float
    reward_components: Dict[str, float]
    steps: int
    generation_time: float
    human_feedback: Optional[float] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate sequences from trained Tianshou agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file).",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/agent_config.yaml",
        help="Path to configuration file (YAML).",
    )

    # Generation mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "interactive", "simulated"],
        default="batch",
        help="Generation mode: batch, interactive (HIL), or simulated feedback.",
    )

    # Generation parameters
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=10,
        help="Number of sequences to generate (batch mode).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of iterations (interactive mode).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic (greedy) policy.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (epsilon-greedy).",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--save-midi",
        action="store_true",
        help="Save sequences as MIDI files.",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level.",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def load_checkpoint_config(checkpoint_path: str) -> Dict[str, Any]:
    """Load configuration from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Configuration dictionary from checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint_data = torch.load(path, map_location="cpu")
    return checkpoint_data.get("config", {})


def detect_action_space_from_checkpoint(checkpoint_path: str) -> int:
    """Detect the action space size from checkpoint network weights.

    For curriculum learning checkpoints, the final layer size indicates
    the action space at the time of saving.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Action space size (number of actions).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint_data = torch.load(path, map_location="cpu")
    network_state = checkpoint_data.get("network", {})
    config = checkpoint_data.get("config", {})

    # Check network type
    network_type = config.get("network", {}).get("type", "drqn").lower()
    is_distributional = "c51" in network_type or "rainbow" in network_type
    # Rainbow and Dueling both use advantage_stream architecture
    is_dueling = "dueling" in network_type or "rainbow" in network_type

    # Get num_atoms for distributional networks
    num_atoms = 1
    if is_distributional:
        num_atoms = config.get("network", {}).get("num_atoms", 51)
        # Also check algorithm config
        if num_atoms == 1:
            num_atoms = (
                config.get("algorithm", {})
                .get("distributional", {})
                .get("num_atoms", 51)
            )

    action_size = None

    # For Dueling networks (including Rainbow), check advantage_stream output layer
    if is_dueling:
        # Find the highest numbered advantage_stream layer
        advantage_layers = {}
        for key, value in network_state.items():
            if "advantage_stream" in key and "bias" in key:
                # Extract layer number from key like "advantage_stream.3.bias"
                parts = key.split(".")
                for part in parts:
                    if part.isdigit():
                        layer_num = int(part)
                        advantage_layers[layer_num] = value.shape[0]
                        break

        if advantage_layers:
            # Get the highest layer number (output layer)
            max_layer = max(advantage_layers.keys())
            output_size = advantage_layers[max_layer]
            # For Rainbow (dueling + distributional), divide by num_atoms
            action_size = output_size // num_atoms
            logger.info(
                f"Detected action space from checkpoint (dueling): {action_size}"
                + (
                    f" (distributional with {num_atoms} atoms)"
                    if is_distributional
                    else ""
                )
            )
            return action_size

    # For standard networks, look for fc.6 output layer
    for key, value in network_state.items():
        # Look for output layer patterns
        if "fc.6.bias" in key or "output.bias" in key:
            output_size = value.shape[0]
            # For distributional networks, divide by num_atoms
            action_size = output_size // num_atoms
            break
        # For weight layer as fallback
        if "fc.6.weight" in key or "output.weight" in key:
            output_size = value.shape[0]
            action_size = output_size // num_atoms
            break

    if action_size is None:
        # Fallback: try to find any layer that looks like output
        for key, value in sorted(network_state.items(), reverse=True):
            if "weight" in key and len(value.shape) == 2:
                output_size = value.shape[0]
                # Divide by num_atoms for distributional
                potential_action_size = output_size // num_atoms
                if potential_action_size <= 100:  # Reasonable action space size
                    action_size = potential_action_size
                    break

    if action_size is None:
        logger.warning("Could not detect action space from checkpoint, using default")
        return 22  # Default full action space

    logger.info(
        f"Detected action space from checkpoint: {action_size}"
        + (f" (distributional with {num_atoms} atoms)" if is_distributional else "")
    )
    return action_size


def setup_environment(
    config: Dict[str, Any],
    action_space_size: Optional[int] = None,
) -> Tuple[MusicGenerationGymEnv, GHSOMManager]:
    """Set up the music generation environment.

    Args:
        config: Configuration dictionary.
        action_space_size: Optional override for action space size.
            Used for curriculum learning checkpoints that have smaller action spaces.

    Returns:
        Tuple of (environment, ghsom_manager).
    """
    # Get paths from config
    ghsom_path = Path(
        config.get("ghsom", {}).get(
            "default_model_path",
            "experiments/ghsom_commu_full_tsne_optimized_20251125/ghsom_model.pkl",
        )
    )
    feature_path = Path(
        config.get("features", {}).get(
            "artifact_path", "artifacts/features/tsne/commu_full_filtered_tsne"
        )
    )
    feature_type = config.get("features", {}).get("type", "tsne")

    logger.info(f"Loading GHSOM from: {ghsom_path}")
    logger.info(f"Loading features from: {feature_path}")

    # Create GHSOM manager
    ghsom_manager = GHSOMManager.from_artifact(
        ghsom_model_path=ghsom_path,
        feature_artifact=feature_path,
        feature_type=feature_type,
    )

    # Create perceiving agent
    perceiving_agent = GHSOMPerceivingAgent(
        config=config,
        ghsom_manager=ghsom_manager,
        features_dataset=ghsom_manager.train_data,
    )

    # Create environment
    sequence_length = config.get("music", {}).get("sequence_length", 16)
    env = MusicGenerationGymEnv(
        perceiving_agent=perceiving_agent,
        sequence_length=sequence_length,
        config=config,
    )

    # Override action space if specified (for curriculum learning checkpoints)
    if action_space_size is not None and action_space_size != env.action_space.n:
        logger.info(
            f"Overriding action space: {env.action_space.n} -> {action_space_size} "
            "(curriculum learning checkpoint)"
        )
        # Create a new discrete action space with the checkpoint's size
        env.action_space = gym.spaces.Discrete(action_space_size)
        env.n_clusters = action_space_size
        # Also limit cluster_ids to match
        if action_space_size <= len(env.cluster_ids):
            env.cluster_ids = env.cluster_ids[:action_space_size]

    # Wrap with feature observations if configured
    use_features = config.get("use_feature_observations", False)
    if use_features:
        feature_mode = config.get("feature_observation_mode", "centroid")

        logger.info(f"Using feature observations (mode={feature_mode})")

        # Create feature mapper using the ghsom_manager
        mapper = ClusterFeatureMapper(
            ghsom_manager=ghsom_manager,
            mode=feature_mode,
            feature_source=feature_type,
        )

        # Wrap environment with feature observations
        env = FeatureObservationWrapper(env, mapper)
        logger.info(f"  Observation shape: {env.observation_space.shape}")

    logger.info("Environment created:")
    logger.info(f"  Action space: {env.action_space.n} clusters")
    logger.info(f"  Sequence length: {sequence_length}")

    return env, ghsom_manager


def load_agent(
    checkpoint_path: str,
    env: MusicGenerationGymEnv,
    config: Dict[str, Any],
) -> TianshouTrainer:
    """Load trained agent from checkpoint.

    Uses the config saved in the checkpoint to ensure network architecture
    matches. Falls back to provided config for missing keys.

    Args:
        checkpoint_path: Path to checkpoint file.
        env: Environment instance.
        config: Configuration dictionary (used as fallback).

    Returns:
        Loaded TianshouTrainer instance.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint: {path}")

    # First, load checkpoint to extract its saved config
    checkpoint_data = torch.load(path, map_location="cpu")
    checkpoint_config = checkpoint_data.get("config", {})

    # Merge configs: checkpoint config takes precedence for network settings
    merged_config = config.copy()

    # Use checkpoint's network config to ensure architecture matches
    if "network" in checkpoint_config:
        merged_config["network"] = checkpoint_config["network"]
        logger.info(
            f"Using checkpoint's network config: {checkpoint_config['network']}"
        )

    # Use checkpoint's algorithm config
    if "algorithm" in checkpoint_config:
        merged_config["algorithm"] = checkpoint_config["algorithm"]

    # Load trainer from checkpoint with merged config
    # Note: TianshouTrainer.load() will use config=None to get checkpoint's config
    trainer = TianshouTrainer.load(path, env=env, config=None)

    logger.info("Agent loaded:")
    logger.info(f"  Network type: {trainer.network_config.get('type', 'drqn')}")
    logger.info(f"  Training timesteps: {trainer.num_timesteps}")
    logger.info(f"  Is recurrent: {trainer.network.is_recurrent}")

    return trainer


def generate_sequence(
    trainer: TianshouTrainer,
    env: MusicGenerationGymEnv,
    deterministic: bool = True,
) -> GenerationResult:
    """Generate a single sequence using the trained policy.

    Args:
        trainer: Trained TianshouTrainer.
        env: Environment instance.
        deterministic: Whether to use greedy action selection.

    Returns:
        GenerationResult with sequence and metrics.
    """
    start_time = time.time()

    obs, info = env.reset()
    episode_reward = 0.0
    steps = 0
    hidden_state = None  # For recurrent networks

    # Set evaluation mode
    trainer.network.eval()

    # Check if this is a distributional network (C51, Rainbow)
    is_distributional = hasattr(trainer.network, "num_atoms")
    if is_distributional:
        # For C51/Rainbow, we need to compute expected Q-values from distributions
        v_min = getattr(trainer.network, "v_min", -10.0)
        v_max = getattr(trainer.network, "v_max", 10.0)
        num_atoms = trainer.network.num_atoms
        support = torch.linspace(v_min, v_max, num_atoms).to(trainer.device)

    with torch.no_grad():
        while True:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)

            # Handle recurrent networks
            if trainer.network.is_recurrent:
                if hidden_state is None:
                    # Initialize hidden state using DRQN's get_initial_state method
                    batch_size = 1
                    hidden_state = trainer.network.get_initial_state(batch_size)

                # Forward pass with hidden state (returns dict with h, c)
                output, hidden_state = trainer.network(obs_tensor, hidden_state)
            else:
                output = trainer.network(obs_tensor)

            # Convert distributional output to Q-values if needed
            if is_distributional:
                # output shape: (batch, n_actions, num_atoms) - probability distributions
                # Compute expected Q-values: sum(probs * support)
                q_values = (output * support.view(1, 1, -1)).sum(
                    dim=2
                )  # (batch, n_actions)
            else:
                q_values = output

            # Action selection
            if deterministic:
                action = q_values.argmax(dim=1).item()
            else:
                # Epsilon-greedy with evaluation epsilon
                eps = trainer.policy.eps_inference
                if np.random.random() < eps:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

    generation_time = time.time() - start_time

    # Extract sequence from info
    sequence = info.get("sequence", [])
    if hasattr(sequence, "tolist"):
        # Filter out padding tokens (-1)
        sequence = [x for x in sequence.tolist() if x >= 0]

    return GenerationResult(
        sequence=sequence,
        episode_reward=episode_reward,
        reward_components=info.get("reward_components", {}),
        steps=steps,
        generation_time=generation_time,
    )


def run_batch_generation(
    trainer: TianshouTrainer,
    env: MusicGenerationGymEnv,
    num_sequences: int,
    deterministic: bool = True,
    verbose: int = 1,
) -> List[GenerationResult]:
    """Generate multiple sequences in batch mode.

    Args:
        trainer: Trained TianshouTrainer.
        env: Environment instance.
        num_sequences: Number of sequences to generate.
        deterministic: Whether to use greedy policy.
        verbose: Verbosity level.

    Returns:
        List of GenerationResult objects.
    """
    results = []

    if verbose >= 1:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[cyan]{task.fields[reward]:.2f}"),
        )
        task = progress.add_task(
            "Generating sequences...",
            total=num_sequences,
            reward=0.0,
        )

        with Live(progress, console=console, refresh_per_second=4):
            for _ in range(num_sequences):
                result = generate_sequence(trainer, env, deterministic)
                results.append(result)

                progress.update(
                    task,
                    advance=1,
                    reward=result.episode_reward,
                )
    else:
        for _ in range(num_sequences):
            result = generate_sequence(trainer, env, deterministic)
            results.append(result)

    return results


def run_interactive_session(
    trainer: TianshouTrainer,
    env: MusicGenerationGymEnv,
    ghsom_manager: GHSOMManager,
    num_iterations: int,
    config: Dict[str, Any],
) -> List[GenerationResult]:
    """Run interactive human-in-the-loop session.

    Args:
        trainer: Trained TianshouTrainer.
        env: Environment instance.
        ghsom_manager: GHSOM manager for analysis.
        num_iterations: Number of iterations.
        config: Configuration dictionary.

    Returns:
        List of GenerationResult objects with human feedback.
    """
    results = []

    # Create feedback collector
    feedback_collector = HumanFeedbackCollector(
        timeout=config.get("human_feedback_timeout", 30),
        non_interactive_mode=False,  # Force interactive
    )

    console.print(
        Panel(
            "[bold cyan]Interactive Inference Session[/bold cyan]\n\n"
            "You will be shown generated sequences and asked to rate them.\n"
            "Enter a rating from 1 (poor) to 5 (excellent), or 0 to skip.\n"
            "Press Ctrl+C to end the session early.",
            title="Human-in-the-Loop Mode",
        )
    )

    try:
        for i in range(num_iterations):
            console.print(f"\n[bold]Iteration {i+1}/{num_iterations}[/bold]")

            # Generate sequence
            result = generate_sequence(trainer, env, deterministic=True)

            # Display sequence
            display_sequence(result.sequence, ghsom_manager, result)

            # Collect feedback
            try:
                feedback = feedback_collector.collect_feedback(
                    prompt="\nRate this sequence (1-5, or 0 to skip): "
                )
                result.human_feedback = feedback

                if feedback > 0:
                    console.print(f"[green]Rating recorded: {feedback}[/green]")
                else:
                    console.print("[yellow]Skipped[/yellow]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Feedback collection interrupted[/yellow]")
                result.human_feedback = None

            results.append(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Session ended by user[/yellow]")

    return results


def run_simulated_session(
    trainer: TianshouTrainer,
    env: MusicGenerationGymEnv,
    num_sequences: int,
    seed: Optional[int] = None,
) -> List[GenerationResult]:
    """Run session with simulated human feedback for testing.

    Args:
        trainer: Trained TianshouTrainer.
        env: Environment instance.
        num_sequences: Number of sequences to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of GenerationResult objects with simulated feedback.
    """
    results = []

    if seed is not None:
        np.random.seed(seed)

    for _ in range(num_sequences):
        result = generate_sequence(trainer, env, deterministic=True)

        # Simulate feedback based on reward
        # Higher reward = more likely positive feedback
        feedback_base = (result.episode_reward + 5) / 10  # Normalize to ~[0, 1]
        noise = np.random.normal(0, 0.2)
        simulated_feedback = np.clip(feedback_base + noise, 1, 5)
        result.human_feedback = round(simulated_feedback, 1)

        results.append(result)

    return results


def display_sequence(
    sequence: List[int],
    ghsom_manager: GHSOMManager,  # noqa: ARG001 - Reserved for hierarchy visualization
    result: GenerationResult,
) -> None:
    """Display sequence visualization in terminal.

    Args:
        sequence: List of cluster IDs.
        ghsom_manager: GHSOM manager for hierarchy info (reserved for future use).
        result: Generation result with metrics.
    """
    # Create sequence visualization
    seq_visual = " → ".join([str(c) for c in sequence])

    # Get unique clusters
    unique = len(set(sequence))

    # Create table
    table = Table(title="Generated Sequence", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Sequence", seq_visual)
    table.add_row("Length", str(len(sequence)))
    table.add_row("Unique Clusters", f"{unique} ({100*unique/len(sequence):.1f}%)")
    table.add_row("Episode Reward", f"{result.episode_reward:.3f}")

    if result.reward_components:
        for comp, value in result.reward_components.items():
            table.add_row(f"  {comp}", f"{value:.3f}")

    console.print(table)


def calculate_aggregate_metrics(results: List[GenerationResult]) -> Dict[str, float]:
    """Calculate aggregate metrics from results.

    Args:
        results: List of generation results.

    Returns:
        Dictionary of aggregate metrics.
    """
    rewards = [r.episode_reward for r in results]
    steps = [r.steps for r in results]

    all_clusters = []
    for r in results:
        all_clusters.extend(r.sequence)

    # Per-sequence diversity
    unique_per_seq = [len(set(r.sequence)) for r in results]

    return {
        "avg_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "avg_steps": np.mean(steps),
        "unique_clusters_total": len(set(all_clusters)),
        "unique_per_sequence_mean": np.mean(unique_per_seq),
        "unique_per_sequence_std": np.std(unique_per_seq),
    }


def save_results(
    results: List[GenerationResult],
    metrics: Dict[str, float],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Save inference results to disk.

    Args:
        results: List of generation results.
        metrics: Aggregate metrics.
        output_dir: Output directory.
        args: Command-line arguments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual sequences
    sequences_file = output_dir / "sequences.json"
    sequences_data = [
        {
            "index": i,
            "sequence": r.sequence,
            "episode_reward": r.episode_reward,
            "reward_components": r.reward_components,
            "steps": r.steps,
            "generation_time": r.generation_time,
            "human_feedback": r.human_feedback,
        }
        for i, r in enumerate(results)
    ]
    with open(sequences_file, "w", encoding="utf-8") as f:
        json.dump(sequences_data, f, indent=2)

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    metrics_data = {
        "aggregate": metrics,
        "config": {
            "checkpoint": args.checkpoint,
            "config_file": args.config,
            "mode": args.mode,
            "num_sequences": len(results),
            "deterministic": args.deterministic,
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"Results saved to: {output_dir}")


def print_summary(
    results: List[GenerationResult],
    metrics: Dict[str, float],
) -> None:
    """Print summary of generation results.

    Args:
        results: List of generation results.
        metrics: Aggregate metrics.
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Inference Summary[/bold cyan]")
    console.print("=" * 60)

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Sequences Generated", str(len(results)))
    table.add_row(
        "Average Reward", f"{metrics['avg_reward']:.3f} ± {metrics['std_reward']:.3f}"
    )
    table.add_row(
        "Reward Range", f"[{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]"
    )
    table.add_row("Unique Clusters Used", str(metrics["unique_clusters_total"]))
    table.add_row(
        "Unique per Sequence",
        f"{metrics['unique_per_sequence_mean']:.1f} ± {metrics['unique_per_sequence_std']:.1f}",
    )

    # Human feedback stats if available
    feedbacks = [r.human_feedback for r in results if r.human_feedback is not None]
    if feedbacks:
        table.add_row("Human Feedback Mean", f"{np.mean(feedbacks):.2f}")
        table.add_row("Human Feedback Std", f"{np.std(feedbacks):.2f}")

    console.print(table)


def main():
    """Main entry point for inference."""
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Determine deterministic mode
    deterministic = not args.stochastic

    try:
        # Load base configuration from file
        base_config = load_config(args.config)

        # Load checkpoint's saved config to ensure compatibility
        checkpoint_config = load_checkpoint_config(args.checkpoint)

        # Merge configs: use checkpoint's settings for network/observation compatibility
        # but allow base_config to provide missing values (paths, etc.)
        config = base_config.copy()

        # Critical settings that must match checkpoint's training config
        if checkpoint_config:
            logger.info("Merging checkpoint config with base config...")

            # Network config must match checkpoint
            if "network" in checkpoint_config:
                config["network"] = checkpoint_config["network"]

            # Algorithm config must match checkpoint
            if "algorithm" in checkpoint_config:
                config["algorithm"] = checkpoint_config["algorithm"]

            # Feature observation settings must match checkpoint
            if "use_feature_observations" in checkpoint_config:
                config["use_feature_observations"] = checkpoint_config[
                    "use_feature_observations"
                ]
            if "feature_observation_mode" in checkpoint_config:
                config["feature_observation_mode"] = checkpoint_config[
                    "feature_observation_mode"
                ]
            if "feature_observation_source" in checkpoint_config:
                config["feature_observation_source"] = checkpoint_config[
                    "feature_observation_source"
                ]

        # Detect action space from checkpoint (for curriculum learning support)
        action_space_size = detect_action_space_from_checkpoint(args.checkpoint)

        # Setup environment with merged config and detected action space
        env, ghsom_manager = setup_environment(config, action_space_size)

        # Load trained agent
        trainer = load_agent(args.checkpoint, env, config)

        # Run generation based on mode
        if args.mode == "batch":
            console.print(
                f"\n[cyan]Generating {args.num_sequences} sequences (batch mode)...[/cyan]"
            )
            results = run_batch_generation(
                trainer, env, args.num_sequences, deterministic, args.verbose
            )

        elif args.mode == "interactive":
            results = run_interactive_session(
                trainer, env, ghsom_manager, args.num_iterations, config
            )

        elif args.mode == "simulated":
            console.print(
                f"\n[cyan]Generating {args.num_sequences} sequences with simulated feedback...[/cyan]"
            )
            results = run_simulated_session(trainer, env, args.num_sequences, args.seed)

        else:
            # Should not reach here due to argparse choices
            raise ValueError(f"Unknown mode: {args.mode}")

        # Calculate metrics
        metrics = calculate_aggregate_metrics(results)

        # Print summary
        print_summary(results, metrics)

        # Save results
        if args.output:
            output_dir = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"outputs/inference_{timestamp}")

        save_results(results, metrics, output_dir, args)

        console.print(
            f"\n[green]✓ Inference complete! Results saved to: {output_dir}[/green]"
        )

        return 0

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Error during inference: {e}[/red]")
        if args.verbose >= 2:
            import traceback

            console.print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
