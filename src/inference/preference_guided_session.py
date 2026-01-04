"""Preference-guided simulation session for Human-in-the-Loop music generation.

This module provides the PreferenceGuidedSession class that extends InteractiveInferenceSession
to support preference-based simulation with both Q-value penalty and reward shaping adaptation modes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import gymnasium as gym

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .interactive_session import InteractiveInferenceSession, InteractiveResult
from .preference_simulation import (
    PreferenceScenario,
    PreferenceFeedbackSimulator,
    SimulatedFeedback,
)
from .cluster_profiles import ClusterProfileLoader
from .config_loader import InferenceConfig
from .conservative_policy_updater import (
    ConservativePolicyUpdater,
    PreferenceShapedReward,
    LearningVerifier,
)
from .cluster_action_mapper import ClusterActionMapping
from .seed_manager import SeedManager
from .exploration import ExplorationStrategy, create_exploration_strategy
from .adaptive_threshold import AdaptiveThresholdManager
from .learning_verifier import SimulationLearningVerifier, LearningVerificationResult

from src.training.tianshou_trainer import TianshouTrainer
from src.ghsom_manager import GHSOMManager

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class SimulationMetrics:
    """Metrics for a single simulation step.

    Attributes:
        iteration: Step number in the simulation.
        sequence: Generated cluster sequence.
        feedback_rating: Simulated user rating (1-5).
        desirable_ratio: Proportion of desirable clusters.
        undesirable_ratio: Proportion of undesirable clusters.
        neutral_ratio: Proportion of neutral clusters.
        entropy: Shannon entropy of cluster distribution.
        target_metric: Value of scenario-specific target metric.
        episode_reward: RL episode reward.
        accumulated_penalty_count: Number of clusters in penalty list.
        adaptation_applied: Whether adaptation was applied this step.
    """

    iteration: int
    sequence: List[int]
    feedback_rating: float
    desirable_ratio: float
    undesirable_ratio: float
    neutral_ratio: float
    entropy: float
    target_metric: float
    episode_reward: float
    accumulated_penalty_count: int
    adaptation_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class SimulationResult:
    """Complete simulation result with all metrics and history.

    Attributes:
        scenario_name: Name of the preference scenario.
        num_iterations: Total iterations in simulation.
        adaptation_mode: Mode used (q_penalty or reward_shaping).
        adaptation_strength: Strength parameter for adaptation.
        seed: Random seed used.
        initial_desirable_ratio: Desirable ratio at start.
        final_desirable_ratio: Desirable ratio at end.
        initial_undesirable_ratio: Undesirable ratio at start.
        final_undesirable_ratio: Undesirable ratio at end.
        distribution_shift: KL-divergence between initial and final distributions.
        mean_feedback_first_10: Average feedback in first 10 iterations.
        mean_feedback_last_10: Average feedback in last 10 iterations.
        feedback_improvement: Change in feedback rating.
        convergence_iteration: Iteration when target was achieved (if any).
        total_time_seconds: Total simulation time.
        metrics_history: List of per-step metrics.
    """

    scenario_name: str
    num_iterations: int
    adaptation_mode: str
    adaptation_strength: float
    seed: int
    initial_desirable_ratio: float = 0.0
    final_desirable_ratio: float = 0.0
    initial_undesirable_ratio: float = 0.0
    final_undesirable_ratio: float = 0.0
    distribution_shift: float = 0.0
    mean_feedback_first_10: float = 0.0
    mean_feedback_last_10: float = 0.0
    feedback_improvement: float = 0.0
    convergence_iteration: Optional[int] = None
    total_time_seconds: float = 0.0
    metrics_history: List[SimulationMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "num_iterations": self.num_iterations,
            "adaptation_mode": self.adaptation_mode,
            "adaptation_strength": self.adaptation_strength,
            "seed": self.seed,
            "initial_desirable_ratio": self.initial_desirable_ratio,
            "final_desirable_ratio": self.final_desirable_ratio,
            "initial_undesirable_ratio": self.initial_undesirable_ratio,
            "final_undesirable_ratio": self.final_undesirable_ratio,
            "distribution_shift": self.distribution_shift,
            "mean_feedback_first_10": self.mean_feedback_first_10,
            "mean_feedback_last_10": self.mean_feedback_last_10,
            "feedback_improvement": self.feedback_improvement,
            "convergence_iteration": self.convergence_iteration,
            "total_time_seconds": self.total_time_seconds,
            "metrics_history": [m.to_dict() for m in self.metrics_history],
        }

    def save(self, path: Path) -> None:
        """Save result to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info("Saved simulation result to %s", path)

    @classmethod
    def load(cls, path: Path) -> "SimulationResult":
        """Load result from JSON file.

        Args:
            path: Input file path.

        Returns:
            SimulationResult instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct metrics history
        metrics_history = [
            SimulationMetrics(**m) for m in data.pop("metrics_history", [])
        ]

        return cls(**data, metrics_history=metrics_history)


class PreferenceGuidedSession(InteractiveInferenceSession):
    """Interactive session with preference-guided feedback simulation.

    This class extends InteractiveInferenceSession to support automated
    preference simulation for demonstrating de-learning of undesirable patterns.

    Supports two adaptation modes:
    1. q_penalty: Apply Q-value penalties to undesirable clusters during inference
    2. reward_shaping: Modify reward weights based on accumulated feedback

    Attributes:
        scenario: The preference scenario defining desirable/undesirable clusters.
        cluster_profiles: Loader for cluster profile metadata.
        feedback_simulator: Simulator for generating preference-aligned feedback.
        adaptation_mode: Mode for adapting agent behavior.
        adaptation_strength: Strength of Q-value penalty or reward modification.
        feedback_threshold: Rating threshold for triggering adaptation.
        distribution_history: History of cluster distributions per iteration.
        metrics_history: History of simulation metrics.
        inference_config: Optional InferenceConfig for parameter management.
    """

    def __init__(
        self,
        trainer: TianshouTrainer,
        env: gym.Env,
        ghsom_manager: GHSOMManager,
        config: Dict[str, Any],
        scenario: PreferenceScenario,
        cluster_profiles: ClusterProfileLoader,
        adaptation_mode: Literal["q_penalty", "reward_shaping"] = "q_penalty",
        adaptation_strength: Optional[float] = None,
        feedback_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        inference_config: Optional[InferenceConfig] = None,
        cluster_mapping: Optional[ClusterActionMapping] = None,
        enable_policy_learning: bool = False,
    ) -> None:
        """Initialize preference-guided session.

        Args:
            trainer: Trained TianshouTrainer instance.
            env: Environment for sequence generation.
            ghsom_manager: GHSOM manager for cluster operations.
            config: Configuration dictionary.
            scenario: Preference scenario for simulation.
            cluster_profiles: Cluster profile loader.
            adaptation_mode: Adaptation mode ("q_penalty" or "reward_shaping").
            adaptation_strength: Strength of adaptation. If None and inference_config
                is provided, uses config value for the adaptation_mode.
            feedback_threshold: Rating threshold for triggering adaptation. If None
                and inference_config is provided, uses config value.
            seed: Random seed for reproducibility.
            inference_config: Optional InferenceConfig instance for parameter management.
        """
        # Initialize parent with non-interactive mode
        config_copy = dict(config)
        config_copy.setdefault("human_feedback", {})
        config_copy["human_feedback"]["non_interactive"] = True

        super().__init__(trainer, env, ghsom_manager, config_copy)

        self.scenario = scenario
        self.cluster_profiles = cluster_profiles
        self.adaptation_mode = adaptation_mode
        self.inference_config = inference_config
        self.seed = seed

        # Use config values if provided, otherwise fall back to explicit parameters or defaults
        if adaptation_strength is None and inference_config is not None:
            self.adaptation_strength = inference_config.get_adaptation_strength(
                adaptation_mode
            )
        elif adaptation_strength is None:
            self.adaptation_strength = 5.0
        else:
            self.adaptation_strength = adaptation_strength

        if feedback_threshold is None and inference_config is not None:
            self.feedback_threshold = inference_config.get_feedback_threshold()
        elif feedback_threshold is None:
            self.feedback_threshold = 3.0
        else:
            self.feedback_threshold = feedback_threshold

        # --- SeedManager integration ---
        self.seed_manager = SeedManager(seed or 42)
        self.seed_manager.seed_all()

        # Pass component-specific seeds
        feedback_seed = self.seed_manager.get_feedback_seed()
        # If feedback_simulator exists, reset its seed
        if hasattr(self, "feedback_simulator") and self.feedback_simulator is not None:
            self.feedback_simulator.reset_seed(feedback_seed)

        # Initialize feedback simulator with config if available
        if inference_config is not None:
            self.feedback_simulator = PreferenceFeedbackSimulator(
                scenario=scenario,
                strictness=inference_config.get_strictness(),
                noise_std=inference_config.get_noise_std(),
                seed=seed,
                config=inference_config,
            )
        else:
            # Backward compatibility: use hardcoded defaults
            self.feedback_simulator = PreferenceFeedbackSimulator(
                scenario=scenario,
                strictness=1.0,
                noise_std=0.3,
                seed=seed,
            )

        # Tracking for simulation analysis
        self.distribution_history: List[Dict[str, float]] = []
        self.metrics_history: List[SimulationMetrics] = []

        # Cluster penalty tracking (extends parent's problematic_clusters)
        self.cluster_penalties: Dict[int, float] = {}

        # Reward shaping state
        self.reward_modifier: float = 0.0
        self.accumulated_negative_feedback: float = 0.0

        # Pre-compute cluster sets for fast lookup
        self._desirable_set = set(scenario.desirable_clusters)
        self._undesirable_set = set(scenario.undesirable_clusters)
        self._all_clusters = (
            set(self.env.cluster_ids) if hasattr(self.env, "cluster_ids") else set()
        )

        # Store cluster mapping for action index conversion
        self.cluster_mapping = cluster_mapping

        # Initialize policy updater for true learning (optional, enabled by flag)
        self.enable_policy_learning = enable_policy_learning or config_copy.get(
            "enable_policy_learning", False
        )
        self.policy_updater: Optional[ConservativePolicyUpdater] = None
        self.learning_verifier: Optional[LearningVerifier] = None

        if self.enable_policy_learning:
            self._initialize_policy_updater()

        # Initialize exploration strategy (Fix 7.1-2 & 7.3-A)
        self.exploration_strategy: Optional[ExplorationStrategy] = None
        self._current_iteration = 0
        self._total_iterations = 500  # Will be updated in run_simulation
        self._initialize_exploration_strategy()

        # Initialize adaptive threshold manager (Fix 7.3-B)
        self.adaptive_threshold_manager: Optional[AdaptiveThresholdManager] = None
        self._initialize_adaptive_threshold()

        # Initialize simulation learning verifier (Fix 7.3-C)
        self.simulation_verifier: Optional[SimulationLearningVerifier] = None
        self._initialize_learning_verifier()
        self._initial_params: Optional[Dict[str, torch.Tensor]] = None

        logger.info(
            "Initialized PreferenceGuidedSession: scenario='%s', mode='%s', strength=%.1f, policy_learning=%s",
            scenario.name,
            adaptation_mode,
            adaptation_strength,
            self.enable_policy_learning,
        )

    def _initialize_policy_updater(self) -> None:
        """Initialize the conservative policy updater for true policy learning.

        This creates a ConservativePolicyUpdater that performs gradient-based
        updates on the policy network, enabling actual learning from preferences.
        """
        # Get the policy network from the trainer
        policy_network = self.trainer.network

        # Get policy learning config from inference config or use defaults
        policy_config = {}
        if self.inference_config is not None:
            policy_config = self.inference_config.get("policy_learning", {})

        # Create reward shaper with config values
        reward_shaper = PreferenceShapedReward(
            alpha=policy_config.get("reward_alpha", 0.1),
            cluster_bonus=policy_config.get("cluster_bonus", 0.05),
            normalize_feedback=True,
        )

        # Create the policy updater with config values
        self.policy_updater = ConservativePolicyUpdater(
            policy=policy_network,
            learning_rate=policy_config.get("learning_rate", 1e-4),
            update_frequency=policy_config.get("update_frequency", 10),
            min_buffer_size=policy_config.get("min_buffer_size", 32),
            batch_size=policy_config.get("batch_size", 16),
            gamma=policy_config.get("gamma", 0.95),
            gradient_clip=policy_config.get("gradient_clip", 1.0),
            reward_shaper=reward_shaper,
            buffer_size=policy_config.get("buffer_size", 1000),
        )

        # Set cluster preferences
        self.policy_updater.set_cluster_preferences(
            list(self._desirable_set),
            list(self._undesirable_set),
        )

        # Create learning verifier and take initial snapshot
        self.learning_verifier = LearningVerifier(track_history=True)
        self.learning_verifier.snapshot_parameters(policy_network, step=0)

        logger.info(
            "Policy updater initialized: learning_rate=%s, update_freq=%d, "
            "desirable=%d clusters, undesirable=%d clusters",
            policy_config.get("learning_rate", 1e-4),
            policy_config.get("update_frequency", 10),
            len(self._desirable_set),
            len(self._undesirable_set),
        )

    def _initialize_exploration_strategy(self) -> None:
        """Initialize the exploration strategy for decaying exploration rate.

        Creates an ExplorationStrategy based on configuration settings.
        Supports epsilon-greedy, Boltzmann, and UCB exploration modes
        with configurable decay schedules.
        """
        if self.inference_config is None:
            # Default to epsilon-greedy with standard decay
            self.exploration_strategy = create_exploration_strategy({
                "mode": "epsilon_greedy",
                "epsilon": {
                    "initial": 0.5,
                    "final": 0.05,
                    "decay_schedule": "linear",
                    "warmup_iterations": 50,
                }
            })
        else:
            exploration_config = self.inference_config.get_exploration_config()
            self.exploration_strategy = create_exploration_strategy(exploration_config)

        logger.info(
            "Exploration strategy initialized: mode=%s",
            type(self.exploration_strategy).__name__,
        )

    def _initialize_adaptive_threshold(self) -> None:
        """Initialize the adaptive threshold manager for dynamic feedback thresholds.

        Creates an AdaptiveThresholdManager if enabled in configuration.
        The manager adjusts feedback threshold based on learning progress.
        """
        if self.inference_config is None:
            return

        if not self.inference_config.get_adaptive_threshold_enabled():
            return

        threshold_config = self.inference_config.get_adaptive_threshold_config()
        self.adaptive_threshold_manager = AdaptiveThresholdManager.from_config(
            threshold_config
        )

        logger.info(
            "Adaptive threshold manager initialized: initial=%.2f, range=[%.2f, %.2f]",
            threshold_config.get("initial_threshold", 3.0),
            threshold_config.get("min_threshold", 2.5),
            threshold_config.get("max_threshold", 3.5),
        )

    def _initialize_learning_verifier(self) -> None:
        """Initialize the simulation learning verifier for post-simulation analysis.

        Creates a SimulationLearningVerifier to verify whether meaningful
        learning occurred during the HIL simulation.
        """
        if self.inference_config is None:
            # Use defaults if no config
            self.simulation_verifier = SimulationLearningVerifier()
        else:
            if not self.inference_config.get_learning_verification_enabled():
                return

            verifier_config = self.inference_config.get_learning_verification_config()
            self.simulation_verifier = SimulationLearningVerifier.from_config(
                verifier_config
            )

        logger.info("Simulation learning verifier initialized")

    def _capture_initial_params(self) -> None:
        """Capture initial policy parameters for verification.

        Stores a snapshot of policy parameters at the start of simulation
        for comparison during learning verification.
        """
        self._initial_params = {
            name: param.detach().clone()
            for name, param in self.trainer.network.named_parameters()
            if param.requires_grad
        }
        logger.debug("Captured initial parameters for %d tensors", len(self._initial_params))

    def _get_current_params(self) -> Dict[str, torch.Tensor]:
        """Get current policy parameters for verification.

        Returns:
            Dictionary mapping parameter names to current values.
        """
        return {
            name: param.detach().clone()
            for name, param in self.trainer.network.named_parameters()
            if param.requires_grad
        }

    def _get_exploration_rate(self, iteration: int, total_iterations: int) -> float:
        """Calculate exploration rate with decay schedule.

        Convenience method to get current exploration rate from strategy.

        Args:
            iteration: Current iteration number.
            total_iterations: Total number of iterations in simulation.

        Returns:
            Epsilon or temperature value for current iteration.
        """
        if self.exploration_strategy is not None:
            return self.exploration_strategy.get_exploration_rate(
                iteration, total_iterations
            )

        # Fallback to config-based epsilon if no strategy
        if self.inference_config is not None:
            eps_config = self.inference_config.get("adaptation.exploration.epsilon", {})
            initial_eps = eps_config.get("initial", 0.5)
            final_eps = eps_config.get("final", 0.05)
            warmup = eps_config.get("warmup_iterations", 50)
            schedule = eps_config.get("decay_schedule", "linear")

            if iteration < warmup:
                return initial_eps

            progress = (iteration - warmup) / max(total_iterations - warmup, 1)
            progress = min(progress, 1.0)

            if schedule == "exponential":
                return initial_eps * (final_eps / max(initial_eps, 1e-10)) ** progress
            else:
                return initial_eps - (initial_eps - final_eps) * progress

        return 0.1  # Default

    def save_checkpoint(self, path: Union[str, Path]) -> Path:
        """Save the current policy checkpoint after HIL adaptation.

        This saves the potentially modified policy network along with
        the adaptation state and metadata.

        Args:
            path: Path to save the checkpoint (without extension).

        Returns:
            Path to the saved checkpoint file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_path = path.with_suffix(".pth")

        # Prepare checkpoint data
        checkpoint = {
            # Policy state
            "network": self.trainer.network.state_dict(),
            "network_type": getattr(self.trainer, "network_type", "unknown"),
            # Optimizer state (if available)
            "optimizer": (
                self.trainer.algorithm.optim.state_dict()
                if hasattr(self.trainer.algorithm, "optim")
                else None
            ),
            # Training progress
            "num_timesteps": getattr(self.trainer, "num_timesteps", 0),
            "num_episodes": getattr(self.trainer, "num_episodes", 0),
            "exploration_rate": getattr(self.trainer, "exploration_rate", 0.0),
            # Config
            "config": self.config,
            # HIL adaptation metadata
            "hil_metadata": {
                "scenario_name": self.scenario.name,
                "adaptation_mode": self.adaptation_mode,
                "adaptation_strength": self.adaptation_strength,
                "num_hil_iterations": len(self.metrics_history),
                "desirable_clusters": list(self._desirable_set),
                "undesirable_clusters": list(self._undesirable_set),
                "cluster_penalties": dict(self.cluster_penalties),
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Add policy updater state if available
        if self.policy_updater is not None:
            checkpoint["policy_updater_stats"] = self.policy_updater.get_update_stats()

        # Add learning verification if available
        if self.learning_verifier is not None:
            verification = self.learning_verifier.verify_learning_occurred(
                self.trainer.network
            )
            checkpoint["learning_verification"] = verification

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        logger.info("Saved HIL-adapted checkpoint to %s", checkpoint_path)

        return checkpoint_path

    def _compute_distribution_metrics(self, sequence: List[int]) -> Dict[str, float]:
        """Compute distribution metrics for a sequence.

        Args:
            sequence: List of cluster IDs.

        Returns:
            Dictionary with distribution metrics.
        """
        if not sequence:
            return {
                "desirable_ratio": 0.0,
                "undesirable_ratio": 0.0,
                "neutral_ratio": 1.0,
                "entropy": 0.0,
                "target_metric": 0.0,
            }

        seq_len = len(sequence)
        desirable_count = sum(1 for c in sequence if c in self._desirable_set)
        undesirable_count = sum(1 for c in sequence if c in self._undesirable_set)
        neutral_count = seq_len - desirable_count - undesirable_count

        desirable_ratio = desirable_count / seq_len
        undesirable_ratio = undesirable_count / seq_len
        neutral_ratio = neutral_count / seq_len

        # Calculate Shannon entropy of cluster distribution
        cluster_counts: Dict[int, int] = {}
        for c in sequence:
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

        probs = np.array(list(cluster_counts.values())) / seq_len
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        # Target metric is just desirable ratio for most scenarios
        target_metric = desirable_ratio

        return {
            "desirable_ratio": desirable_ratio,
            "undesirable_ratio": undesirable_ratio,
            "neutral_ratio": neutral_ratio,
            "entropy": entropy,
            "target_metric": target_metric,
        }

    def _apply_q_penalty_adaptation(
        self,
        feedback: SimulatedFeedback,
        sequence: List[int],
    ) -> bool:
        """Apply Q-value penalty adaptation based on feedback.

        Args:
            feedback: Simulated feedback for the sequence.
            sequence: The generated sequence.

        Returns:
            True if adaptation was applied.
        """
        if feedback.overall >= self.feedback_threshold:
            return False

        # Penalize undesirable clusters that appeared in this sequence
        undesirable_in_seq = [c for c in sequence if c in self._undesirable_set]

        for cluster_id in undesirable_in_seq:
            current_penalty = self.cluster_penalties.get(cluster_id, 0.0)
            # Accumulate penalty, stronger for lower ratings
            penalty_increment = self.adaptation_strength * (
                self.feedback_threshold - feedback.overall
            )
            self.cluster_penalties[cluster_id] = current_penalty + penalty_increment

        # Also update parent's problematic_clusters for compatibility
        for cluster_id in undesirable_in_seq:
            self.problematic_clusters[cluster_id] = (
                self.problematic_clusters.get(cluster_id, 0) + 1
            )

        logger.debug(
            "Applied Q-penalty adaptation: %d clusters penalized, feedback=%.2f",
            len(undesirable_in_seq),
            feedback.overall,
        )

        return len(undesirable_in_seq) > 0

    def _apply_reward_shaping_adaptation(
        self,
        feedback: SimulatedFeedback,
        sequence: List[int],
    ) -> bool:
        """Apply reward shaping adaptation based on feedback.

        This modifies the reward weighting to discourage undesirable patterns.

        Args:
            feedback: Simulated feedback for the sequence.
            sequence: The generated sequence.

        Returns:
            True if adaptation was applied.
        """
        if feedback.overall >= self.feedback_threshold:
            # Positive feedback - reduce negative modifier
            self.accumulated_negative_feedback = max(
                0, self.accumulated_negative_feedback - 0.1
            )
            return False

        # Accumulate negative feedback
        deficit = self.feedback_threshold - feedback.overall
        self.accumulated_negative_feedback += deficit * 0.5

        # Also apply Q-penalty for compound effect
        self._apply_q_penalty_adaptation(feedback, sequence)

        logger.debug(
            "Applied reward-shaping adaptation: accumulated_negative=%.2f",
            self.accumulated_negative_feedback,
        )

        return True

    def _apply_adaptation(
        self,
        feedback: SimulatedFeedback,
        sequence: List[int],
    ) -> bool:
        """Apply adaptation based on the configured mode.

        Args:
            feedback: Simulated feedback.
            sequence: Generated sequence.

        Returns:
            True if adaptation was applied.
        """
        applied = False

        if self.adaptation_mode == "q_penalty":
            applied = self._apply_q_penalty_adaptation(feedback, sequence)
        elif self.adaptation_mode == "reward_shaping":
            applied = self._apply_reward_shaping_adaptation(feedback, sequence)
        else:
            logger.warning("Unknown adaptation mode: %s", self.adaptation_mode)

        # If policy learning is enabled, also update the policy
        if self.enable_policy_learning and self.policy_updater is not None:
            policy_updated = self._apply_policy_learning(feedback, sequence)
            applied = applied or policy_updated

        return applied

    def _apply_policy_learning(
        self,
        feedback: SimulatedFeedback,
        sequence: List[int],
    ) -> bool:
        """Apply true policy learning using gradient updates.

        This uses the ConservativePolicyUpdater to perform actual gradient
        updates on the policy network, enabling persistent learning.

        Args:
            feedback: Simulated feedback for the sequence.
            sequence: The generated sequence.

        Returns:
            True if policy was updated.
        """
        if self.policy_updater is None:
            return False

        # Convert feedback to preference signal in [-1, 1] range
        # Map rating from [1, 5] to [-1, 1]
        preference_signal = (feedback.overall - 3.0) / 2.0

        # Add experiences for each step in the sequence
        # Since we don't have per-step states, we use a simplified approach:
        # Create pseudo-experiences from the sequence
        for i, cluster_id in enumerate(sequence):
            # Create simple state representation (sequence so far)
            if i == 0:
                state = np.zeros((16, 2), dtype=np.float32)
            else:
                # Use previous clusters as state context
                state = self._create_state_from_sequence(sequence[:i])

            # Next state
            if i < len(sequence) - 1:
                next_state = self._create_state_from_sequence(sequence[: i + 1])
                done = False
            else:
                next_state = self._create_state_from_sequence(sequence)
                done = True

            # Determine reward based on cluster type
            if cluster_id in self._desirable_set:
                base_reward = 1.0
            elif cluster_id in self._undesirable_set:
                base_reward = -1.0
            else:
                base_reward = 0.0

            # Convert cluster_id to action index using mapping
            # If mapping is not available, skip this experience
            if self.cluster_mapping is not None:
                try:
                    action_idx = self.cluster_mapping.map_to_action(cluster_id)
                except ValueError:
                    # Cluster ID not in valid set, skip
                    logger.debug(
                        "Skipping cluster_id=%d: not in action space", cluster_id
                    )
                    continue
            else:
                # No mapping - assume cluster_id IS the action index
                # This is fallback behavior and may cause issues
                action_idx = cluster_id
                if action_idx >= self.trainer.network.n_actions:
                    logger.warning(
                        "cluster_id=%d exceeds action space (%d), skipping",
                        cluster_id,
                        self.trainer.network.n_actions,
                    )
                    continue

            # Add experience to buffer
            self.policy_updater.add_experience(
                state=state,
                action=action_idx,  # Use action index, not cluster ID
                reward=base_reward,
                next_state=next_state,
                done=done,
                feedback=preference_signal,
                cluster_id=cluster_id,  # Keep original cluster_id for reference
            )

        # Check if we should update the policy
        if self.policy_updater.should_update():
            update_stats = self.policy_updater.update_policy()
            if update_stats:
                logger.debug(
                    "Policy updated: loss=%.4f, grad_norm=%.4f",
                    update_stats.get("loss", 0),
                    update_stats.get("gradient_norm", 0),
                )
                return True

        return False

    def _create_state_from_sequence(self, sequence: List[int]) -> np.ndarray:
        """Create a state representation from a sequence.

        This creates a simple state array that can be used for the policy updater.

        Args:
            sequence: List of cluster IDs.

        Returns:
            State array of shape (16, 2).
        """
        # Create a (16, 2) array
        state = np.zeros((16, 2), dtype=np.float32)

        # Fill with normalized cluster IDs and position
        for i, cluster_id in enumerate(sequence[-16:]):  # Last 16 clusters
            pos = i if len(sequence) <= 16 else i + (16 - min(16, len(sequence)))
            if pos < 16:
                state[pos, 0] = cluster_id / 50.0  # Normalize cluster ID
                state[pos, 1] = i / 16.0  # Position in sequence

        return state

    def generate_sequence_with_preference(
        self,
        deterministic: bool = True,
    ) -> Tuple[List[int], float, Dict[str, float], int, float]:
        """Generate sequence with preference-based Q-value penalties.

        This extends the parent's generate_sequence method by applying
        accumulated cluster penalties to Q-values during action selection.

        Args:
            deterministic: Use greedy action selection.

        Returns:
            Tuple of (sequence, reward, reward_components, steps, time).
        """
        start_time = time.time()

        obs, info = self.env.reset()
        episode_reward = 0.0
        steps = 0
        hidden_state = None

        self.trainer.network.eval()

        # Check for distributional network
        is_distributional = hasattr(self.trainer.network, "num_atoms")
        if is_distributional:
            v_min = getattr(self.trainer.network, "v_min", -10.0)
            v_max = getattr(self.trainer.network, "v_max", 10.0)
            num_atoms = self.trainer.network.num_atoms
            support = torch.linspace(v_min, v_max, num_atoms).to(self.trainer.device)

        with torch.no_grad():
            while True:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.trainer.device)

                # Handle recurrent networks
                if self.trainer.network.is_recurrent:
                    if hidden_state is None:
                        hidden_state = self.trainer.network.get_initial_state(1)
                    output, hidden_state = self.trainer.network(
                        obs_tensor, hidden_state
                    )
                else:
                    output = self.trainer.network(obs_tensor)

                # Convert to Q-values if distributional
                if is_distributional:
                    q_values = (output * support.view(1, 1, -1)).sum(dim=2)
                else:
                    q_values = output

                # Apply accumulated cluster penalties (map cluster_id to action index)
                if self.cluster_mapping is not None:
                    for cluster_id, penalty in self.cluster_penalties.items():
                        try:
                            action_idx = self.cluster_mapping.map_to_action(cluster_id)
                        except ValueError:
                            continue
                        if action_idx < q_values.shape[1]:
                            q_values[0, action_idx] -= penalty
                else:
                    for cluster_id, penalty in self.cluster_penalties.items():
                        if cluster_id < q_values.shape[1]:
                            q_values[0, cluster_id] -= penalty

                # Apply reward shaping modifier if applicable (map cluster_id to action index)
                if (
                    self.adaptation_mode == "reward_shaping"
                    and self.accumulated_negative_feedback > 0
                ):
                    if self.cluster_mapping is not None:
                        for cluster_id in self._undesirable_set:
                            try:
                                action_idx = self.cluster_mapping.map_to_action(
                                    cluster_id
                                )
                            except ValueError:
                                continue
                            if action_idx < q_values.shape[1]:
                                modifier = self.accumulated_negative_feedback * 0.5
                                q_values[0, action_idx] -= modifier
                    else:
                        for cluster_id in self._undesirable_set:
                            if cluster_id < q_values.shape[1]:
                                modifier = self.accumulated_negative_feedback * 0.5
                                q_values[0, cluster_id] -= modifier

                # Action selection with exploration strategy
                if deterministic:
                    action = q_values.argmax(dim=1).item()
                elif self.exploration_strategy is not None:
                    # Use configured exploration strategy with decay
                    action = self.exploration_strategy.select_action(
                        q_values,
                        self._current_iteration,
                        self._total_iterations,
                    )
                else:
                    # Fallback to simple epsilon-greedy
                    eps = (
                        self.inference_config.get_epsilon_greedy()
                        if self.inference_config is not None
                        else 0.1
                    )
                    if np.random.random() < eps:
                        action = self.env.action_space.sample()
                    else:
                        action = q_values.argmax(dim=1).item()

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

        generation_time = time.time() - start_time

        sequence = info.get("sequence", [])
        if hasattr(sequence, "tolist"):
            sequence = [x for x in sequence.tolist() if x >= 0]

        reward_components = info.get("reward_components", {})

        return sequence, episode_reward, reward_components, steps, generation_time

    def run_simulation(
        self,
        num_iterations: Optional[int] = None,
        log_interval: Optional[int] = None,
        verbose: bool = True,
    ) -> SimulationResult:
        """Run preference-guided simulation.

        Args:
            num_iterations: Number of sequences to generate. If None and inference_config
                is available, uses config value.
            log_interval: Interval for logging progress. If None and inference_config
                is available, uses config value.
            verbose: Whether to print progress to console.

        Returns:
            SimulationResult with all metrics and history.
        """
        # Use config values if available and parameters not explicitly provided
        if num_iterations is None:
            num_iterations = (
                self.inference_config.get_num_iterations()
                if self.inference_config is not None
                else 50
            )

        if log_interval is None:
            log_interval = (
                self.inference_config.get_log_interval()
                if self.inference_config is not None
                else 10
            )

        start_time = time.time()

        # Reset state
        self.cluster_penalties.clear()
        self.problematic_clusters.clear()
        self.accumulated_negative_feedback = 0.0
        self.metrics_history.clear()
        self.distribution_history.clear()

        # Set iteration counters for exploration strategy
        self._current_iteration = 0
        self._total_iterations = num_iterations

        # Reset feedback simulator seed
        if self.seed is not None:
            self.feedback_simulator.reset_seed(self.seed)

        # Reset adaptive threshold manager if enabled
        if self.adaptive_threshold_manager is not None:
            self.adaptive_threshold_manager.reset()

        # Capture initial parameters for learning verification
        if self.simulation_verifier is not None:
            self._capture_initial_params()

        if verbose:
            console.print(
                Panel(
                    f"[bold cyan]Preference-Guided Simulation[/bold cyan]\n\n"
                    f"Scenario: {self.scenario.name}\n"
                    f"Adaptation Mode: {self.adaptation_mode}\n"
                    f"Adaptation Strength: {self.adaptation_strength}\n"
                    f"Iterations: {num_iterations}\n"
                    f"Feedback Threshold: {self.feedback_threshold}",
                    title="HIL Simulation",
                )
            )

        # Run simulation iterations
        # OPTIMIZED: Use exploration during simulation for better learning
        use_exploration = (
            self.inference_config.get("adaptation.exploration.enable_during_simulation", True)
            if self.inference_config is not None
            else True
        )
        for i in range(num_iterations):
            # Update iteration counter for exploration strategy
            self._current_iteration = i

            # Generate sequence with preference penalties
            # CRITICAL FIX: Use deterministic=False to enable exploration
            sequence, episode_reward, reward_components, steps, gen_time = (
                self.generate_sequence_with_preference(deterministic=not use_exploration)
            )

            # Compute simulated feedback
            feedback = self.feedback_simulator.compute_feedback(sequence)

            # Compute distribution metrics
            dist_metrics = self._compute_distribution_metrics(sequence)
            self.distribution_history.append(dist_metrics)

            # Update adaptive threshold if enabled
            if self.adaptive_threshold_manager is not None:
                threshold_adjusted = self.adaptive_threshold_manager.update(
                    feedback.overall, i
                )
                if threshold_adjusted:
                    self.feedback_threshold = self.adaptive_threshold_manager.get_threshold()

            # Apply adaptation based on feedback
            adaptation_applied = self._apply_adaptation(feedback, sequence)

            # Record metrics
            metrics = SimulationMetrics(
                iteration=i,
                sequence=sequence,
                feedback_rating=feedback.overall,
                desirable_ratio=dist_metrics["desirable_ratio"],
                undesirable_ratio=dist_metrics["undesirable_ratio"],
                neutral_ratio=dist_metrics["neutral_ratio"],
                entropy=dist_metrics["entropy"],
                target_metric=dist_metrics["target_metric"],
                episode_reward=episode_reward,
                accumulated_penalty_count=len(self.cluster_penalties),
                adaptation_applied=adaptation_applied,
            )
            self.metrics_history.append(metrics)

            # Log progress with exploration rate info
            if verbose and (i + 1) % log_interval == 0:
                eps = self._get_exploration_rate(i, num_iterations)
                console.print(
                    f"[dim]Iteration {i+1}/{num_iterations}: "
                    f"feedback={feedback.overall:.2f}, "
                    f"desirable={dist_metrics['desirable_ratio']:.2f}, "
                    f"undesirable={dist_metrics['undesirable_ratio']:.2f}, "
                    f"eps={eps:.3f}[/dim]"
                )

        # Compute summary statistics
        result = self._compute_result_summary(num_iterations, start_time)

        # Run learning verification (Fix 7.3-C)
        verification_result = None
        if self.simulation_verifier is not None:
            verification_result = self.simulation_verifier.verify(
                [m.to_dict() for m in self.metrics_history],
                initial_params=self._initial_params,
                final_params=self._get_current_params(),
            )
            logger.info(
                "Learning verification: detected=%s, confidence=%.2f",
                verification_result.learning_detected,
                verification_result.confidence,
            )

        if verbose:
            self._print_simulation_summary(result, verification_result)

        return result

    def _compute_result_summary(
        self,
        num_iterations: int,
        start_time: float,
    ) -> SimulationResult:
        """Compute summary statistics for simulation result.

        Args:
            num_iterations: Total iterations.
            start_time: Start time for duration calculation.

        Returns:
            SimulationResult with all computed statistics.
        """
        # Get initial and final metrics
        initial_metrics = (
            self.metrics_history[:10]
            if len(self.metrics_history) >= 10
            else self.metrics_history[:1]
        )
        final_metrics = (
            self.metrics_history[-10:]
            if len(self.metrics_history) >= 10
            else self.metrics_history[-1:]
        )

        initial_desirable = np.mean([m.desirable_ratio for m in initial_metrics])
        final_desirable = np.mean([m.desirable_ratio for m in final_metrics])
        initial_undesirable = np.mean([m.undesirable_ratio for m in initial_metrics])
        final_undesirable = np.mean([m.undesirable_ratio for m in final_metrics])

        # Compute distribution shift (simplified KL-divergence proxy)
        distribution_shift = abs(final_desirable - initial_desirable) + abs(
            initial_undesirable - final_undesirable
        )

        # Compute feedback improvement
        mean_feedback_first_10 = np.mean([m.feedback_rating for m in initial_metrics])
        mean_feedback_last_10 = np.mean([m.feedback_rating for m in final_metrics])
        feedback_improvement = mean_feedback_last_10 - mean_feedback_first_10

        # Find convergence iteration (when target achieved)
        target_value = self.scenario.target_value
        convergence_iteration = None
        for m in self.metrics_history:
            if m.desirable_ratio >= target_value:
                convergence_iteration = m.iteration
                break

        return SimulationResult(
            scenario_name=self.scenario.name,
            num_iterations=num_iterations,
            adaptation_mode=self.adaptation_mode,
            adaptation_strength=self.adaptation_strength,
            seed=self.seed or 0,
            initial_desirable_ratio=float(initial_desirable),
            final_desirable_ratio=float(final_desirable),
            initial_undesirable_ratio=float(initial_undesirable),
            final_undesirable_ratio=float(final_undesirable),
            distribution_shift=float(distribution_shift),
            mean_feedback_first_10=float(mean_feedback_first_10),
            mean_feedback_last_10=float(mean_feedback_last_10),
            feedback_improvement=float(feedback_improvement),
            convergence_iteration=convergence_iteration,
            total_time_seconds=time.time() - start_time,
            metrics_history=list(self.metrics_history),
        )

    def _print_simulation_summary(
        self,
        result: SimulationResult,
        verification: Optional[LearningVerificationResult] = None,
    ) -> None:
        """Print simulation summary to console.

        Args:
            result: SimulationResult to summarize.
            verification: Optional LearningVerificationResult to include.
        """
        table = Table(title="Simulation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Initial", style="yellow")
        table.add_column("Final", style="green")
        table.add_column("Change", style="magenta")

        # Desirable ratio
        change = result.final_desirable_ratio - result.initial_desirable_ratio
        table.add_row(
            "Desirable Ratio",
            f"{result.initial_desirable_ratio:.3f}",
            f"{result.final_desirable_ratio:.3f}",
            f"{change:+.3f}",
        )

        # Undesirable ratio
        change = result.final_undesirable_ratio - result.initial_undesirable_ratio
        table.add_row(
            "Undesirable Ratio",
            f"{result.initial_undesirable_ratio:.3f}",
            f"{result.final_undesirable_ratio:.3f}",
            f"{change:+.3f}",
        )

        # Feedback
        table.add_row(
            "Avg Feedback",
            f"{result.mean_feedback_first_10:.2f}",
            f"{result.mean_feedback_last_10:.2f}",
            f"{result.feedback_improvement:+.2f}",
        )

        console.print(table)

        # Additional info
        console.print(
            f"\n[dim]Distribution Shift: {result.distribution_shift:.3f}[/dim]"
        )
        if result.convergence_iteration is not None:
            console.print(
                f"[green]Target achieved at iteration {result.convergence_iteration}[/green]"
            )
        console.print(f"[dim]Total time: {result.total_time_seconds:.1f}s[/dim]")

        # Learning verification results
        if verification is not None:
            status_color = "green" if verification.learning_detected else "yellow"
            status_text = "DETECTED" if verification.learning_detected else "NOT DETECTED"
            console.print(
                f"\n[bold]Learning Verification:[/bold] "
                f"[{status_color}]{status_text}[/{status_color}]"
            )
            console.print(
                f"[dim]  Confidence: {verification.confidence:.2%}, "
                f"Diversity: {verification.unique_sequences}/{verification.total_sequences} "
                f"({verification.diversity_ratio:.1%})[/dim]"
            )
            if verification.parameter_l2_change is not None:
                console.print(
                    f"[dim]  Parameter change: {verification.parameter_l2_change:.6f}[/dim]"
                )

        # Adaptive threshold info
        if self.adaptive_threshold_manager is not None:
            adjustments = self.adaptive_threshold_manager.get_adjustment_history()
            if adjustments:
                console.print(
                    f"\n[dim]Adaptive threshold adjustments: {len(adjustments)} "
                    f"(final: {self.adaptive_threshold_manager.get_threshold():.3f})[/dim]"
                )

    def reset_adaptation_state(self) -> None:
        """Reset all adaptation state for a new simulation run."""
        self.cluster_penalties.clear()
        self.problematic_clusters.clear()
        self.accumulated_negative_feedback = 0.0
        self.metrics_history.clear()
        self.distribution_history.clear()

        # Reset new components
        self._current_iteration = 0
        if self.adaptive_threshold_manager is not None:
            self.adaptive_threshold_manager.reset()
        self._initial_params = None

        logger.info("Reset adaptation state for new simulation")

    def get_epsilon_greedy(self) -> float:
        """Get epsilon-greedy exploration rate.

        Returns:
            Epsilon value from config if available, otherwise 0.1.
        """
        if self.inference_config is not None:
            return self.inference_config.get_epsilon_greedy()
        return 0.1
