"""Curriculum Learning Callback for Tianshou Training.

This module provides a Tianshou callback that handles curriculum
phase transitions during training. It monitors training progress,
performs weight mitosis, and rebuilds the algorithm when phases change.

Supports multiple algorithm types:
- DQN/DuelingDQN: Standard Q-learning
- C51: Distributional RL with categorical distribution
- Rainbow: Dueling + Distributional RL

Classes:
    CurriculumCallback: Main callback for curriculum learning.

Example:
    >>> # Standard DQN callback
    >>> callback = CurriculumCallback(
    ...     hierarchy=hierarchy,
    ...     env_wrapper=curriculum_wrapper,
    ...     epsilon_boost=0.3,
    ... )
    >>>
    >>> # C51/Rainbow callback
    >>> callback = CurriculumCallback(
    ...     hierarchy=hierarchy,
    ...     algo_type="c51",
    ...     num_atoms=51,
    ...     v_min=-3.0,
    ...     v_max=7.0,
    ... )
    >>>
    >>> trainer = TianshouTrainer(env=env, callbacks=[callback])
    >>> trainer.train()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import gymnasium as gym

from tianshou.algorithm import DQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.curriculum.hierarchy_extractor import DynamicHierarchy
from src.curriculum.phase_manager import DynamicPhaseManager
from src.curriculum.weight_mitosis import WeightMitosis
from src.training.callbacks.base import BaseTianshouCallback
from src.utils.logging.logging_manager import get_logger

if TYPE_CHECKING:
    from src.training.tianshou_trainer import TianshouTrainer


class CurriculumCallback(BaseTianshouCallback):
    """Tianshou callback for curriculum learning phase transitions.

    This callback integrates with the Tianshou training loop to:
    1. Monitor training progress for phase transitions
    2. Perform weight mitosis when transitioning
    3. Update the environment wrapper's action space
    4. Rebuild the Tianshou algorithm after network modification
    5. Boost exploration (epsilon) on phase entry

    The callback is designed to work with TianshouTrainer and
    CurriculumEnvironmentWrapper to provide seamless curriculum
    learning integration.

    Supports multiple algorithm types:
    - DQN/DuelingDQN: Standard Q-learning with DiscreteQLearningPolicy
    - C51: Distributional RL with C51Policy
    - Rainbow: Dueling + Distributional with RainbowDQN algorithm

    Attributes:
        hierarchy: Curriculum hierarchy from GHSOM.
        phase_manager: Phase transition manager.
        env_wrapper: Environment wrapper for action mapping (optional).
        weight_mitosis: Weight expansion handler.
        epsilon_boost: Epsilon value to set on phase transition.
        flush_buffer: Whether to flush replay buffer on transition.
        algo_type: Algorithm type ('auto', 'dqn', 'dueling_dqn', 'c51', 'rainbow').
        num_atoms: Number of atoms for distributional networks.
        v_min: Minimum value for value distribution.
        v_max: Maximum value for value distribution.
        logger: Logger instance.

    Example:
        >>> # Standard DQN
        >>> callback = CurriculumCallback(
        ...     hierarchy=hierarchy,
        ...     timesteps_per_action=2500,
        ...     epsilon_boost=0.3,
        ... )
        >>>
        >>> # C51 Distributional
        >>> callback = CurriculumCallback(
        ...     hierarchy=hierarchy,
        ...     algo_type="c51",
        ...     num_atoms=51,
        ...     v_min=-3.0,
        ...     v_max=7.0,
        ... )
        >>>
        >>> trainer = TianshouTrainer(env=env, callbacks=[callback])
        >>> trainer.train()
    """

    def __init__(
        self,
        hierarchy: DynamicHierarchy,
        env_wrapper: Optional[Any] = None,
        timesteps_per_action: int = 2500,
        patience_per_action: int = 150,
        plateau_threshold: float = 0.01,
        epsilon_boost: Optional[float] = 0.3,
        flush_buffer: bool = True,
        add_mitosis_noise: bool = True,
        mitosis_noise_scale: float = 0.01,
        verbose: int = 1,
        # Algorithm-specific parameters
        algo_type: str = "auto",
        num_atoms: int = 51,
        v_min: float = -3.0,
        v_max: float = 7.0,
    ):
        """Initialize curriculum callback.

        Args:
            hierarchy: Curriculum hierarchy from HierarchyExtractor.
            env_wrapper: CurriculumEnvironmentWrapper for action mapping.
                If None, callback operates in network-only mode.
            timesteps_per_action: Timesteps per action for phase duration.
            patience_per_action: Patience per action for plateau detection.
            plateau_threshold: Threshold for plateau detection.
            epsilon_boost: Epsilon to set on phase transition.
                Set to None to skip epsilon modification.
            flush_buffer: Whether to flush replay buffer on transition.
            add_mitosis_noise: Add noise to differentiate siblings.
            mitosis_noise_scale: Scale of mitosis noise.
            verbose: Verbosity level.
            algo_type: Algorithm type for rebuild. One of:
                'auto' (detect from network), 'dqn', 'dueling_dqn', 'c51', 'rainbow'.
            num_atoms: Number of atoms for distributional RL (C51/Rainbow).
            v_min: Minimum value for value distribution.
            v_max: Maximum value for value distribution.
        """
        super().__init__(verbose=verbose)

        self.hierarchy = hierarchy
        self.env_wrapper = env_wrapper
        self.epsilon_boost = epsilon_boost
        self.flush_buffer = flush_buffer

        # Algorithm-specific config
        self.algo_type = algo_type
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Create phase manager
        self.phase_manager = DynamicPhaseManager(
            hierarchy=hierarchy,
            timesteps_per_action=timesteps_per_action,
            patience_per_action=patience_per_action,
            plateau_threshold=plateau_threshold,
        )

        # Create weight mitosis handler
        self.weight_mitosis = WeightMitosis(
            add_noise=add_mitosis_noise,
            noise_scale=mitosis_noise_scale,
        )

        self.logger = get_logger("CurriculumCallback")

        # Track metrics
        self._total_transitions = 0
        self._last_transition_step = 0
        self._trainer: Optional["TianshouTrainer"] = None

        # Phase transition log file (set during training start)
        self._transition_log_path: Optional[Path] = None
        self._transition_log: List[Dict[str, Any]] = []

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Called at the start of training.

        Logs initial curriculum state and sets up phase transition log file.

        Args:
            trainer: TianshouTrainer instance.
        """
        self.logger.info(
            f"Curriculum learning initialized:"
            f"\n  Total phases: {self.hierarchy.total_phases}"
            f"\n  Starting phase: {self.phase_manager.current_phase}"
            f"\n  Initial action space: {self.phase_manager.current_action_space_size}"
            f"\n  Final action space: {self.hierarchy.total_leaf_clusters}"
        )

        # Store reference to trainer for later use
        self._trainer = trainer

        # Set up phase transition log file (if output_dir is configured)
        output_dir = None
        if hasattr(trainer, "config") and trainer.config:
            output_dir = trainer.config.get("paths", {}).get("output_dir")
        if output_dir:
            self._transition_log_path = (
                Path(output_dir) / "logs" / "phase_transitions.json"
            )
            self._transition_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                f"Phase transitions will be logged to: {self._transition_log_path}"
            )

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Called after each data collection step.

        Checks for phase transitions and performs necessary updates.

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        # Get episode rewards from collection
        if collect_result.n_collected_episodes > 0:
            for reward in collect_result.returns:
                # Update phase manager with each episode reward
                timesteps = collect_result.n_collected_steps // max(
                    1, collect_result.n_collected_episodes
                )

                transitioned = self.phase_manager.update_episode(
                    episode_reward=reward,
                    episode_timesteps=timesteps,
                )

                if transitioned:
                    self._handle_phase_transition(trainer)
        else:
            # Step-based update (no complete episodes)
            # Use average recent reward if available
            avg_reward = 0.0
            if hasattr(trainer, "_recent_losses") and trainer._recent_losses:
                # Use negative loss as proxy for "reward" in absence of episodes
                avg_reward = -trainer._recent_losses[-1]

            transitioned = self.phase_manager.update(
                reward=avg_reward,
                timesteps=collect_result.n_collected_steps,
            )

            if transitioned:
                self._handle_phase_transition(trainer)

    def _handle_phase_transition(self, trainer: "TianshouTrainer") -> None:
        """Handle a phase transition.

        Performs:
        1. Weight mitosis on Q-network
        2. Update environment wrapper action space
        3. Rebuild Tianshou algorithm
        4. Boost epsilon for exploration
        5. Optionally flush replay buffer

        Args:
            trainer: TianshouTrainer instance.
        """
        old_phase = self.phase_manager.current_phase - 1
        new_phase = self.phase_manager.current_phase

        self.logger.info(f"Phase transition: {old_phase} -> {new_phase}")

        # Get mitosis mapping
        mapping = self.phase_manager.get_last_mitosis_mapping()

        if mapping is None:
            self.logger.error("No mitosis mapping available!")
            return

        old_action_size = self.hierarchy.get_phase(old_phase).action_space_size
        new_action_size = self.hierarchy.get_phase(new_phase).action_space_size

        self.logger.info(f"Action space: {old_action_size} -> {new_action_size}")

        # Log phase transition to file
        transition_record = {
            "timestamp": datetime.now().isoformat(),
            "from_phase": old_phase,
            "to_phase": new_phase,
            "old_action_size": old_action_size,
            "new_action_size": new_action_size,
            "timestep": trainer.num_timesteps,
            "episode": getattr(trainer, "num_episodes", 0),
        }
        self._transition_log.append(transition_record)
        self._save_transition_log()

        # 1. Perform weight mitosis
        self._perform_weight_mitosis(
            trainer=trainer,
            mapping=mapping,
            old_action_size=old_action_size,
            new_action_size=new_action_size,
        )

        # 2. Update environment wrapper
        if self.env_wrapper is not None:
            self._update_env_wrapper(new_phase)

        # 3. Rebuild algorithm with new action space
        self._rebuild_algorithm(trainer, new_action_size)

        # 4. Boost epsilon
        if self.epsilon_boost is not None:
            trainer.policy.set_eps_training(self.epsilon_boost)
            self.logger.info(f"Epsilon boosted to {self.epsilon_boost}")

        # 5. Optionally flush buffer
        if self.flush_buffer:
            trainer.buffer.reset()
            self.logger.info("Replay buffer flushed")

        self._total_transitions += 1
        self._last_transition_step = trainer.num_timesteps

        self.logger.info(
            f"Phase transition complete. " f"New action space: {new_action_size}"
        )

    def _save_transition_log(self) -> None:
        """Save phase transition log to file."""
        if self._transition_log_path is not None:
            try:
                with open(self._transition_log_path, "w", encoding="utf-8") as f:
                    json.dump(self._transition_log, f, indent=2)
            except (IOError, OSError) as e:
                self.logger.warning(f"Failed to save transition log: {e}")

    def _perform_weight_mitosis(
        self,
        trainer: "TianshouTrainer",
        mapping: Dict[int, List[int]],
        old_action_size: int,
        new_action_size: int,
    ) -> None:
        """Perform weight mitosis on the network.

        Args:
            trainer: TianshouTrainer instance.
            mapping: Parent to children action mapping.
            old_action_size: Old action space size.
            new_action_size: New action space size.
        """
        # Detect num_atoms for distributional networks
        algo_type = self._detect_algo_type(trainer)
        if algo_type in ("c51", "rainbow"):
            num_atoms = self.num_atoms
        else:
            num_atoms = 1

        self.weight_mitosis.expand_q_head(
            q_network=trainer.network,
            parent_to_children=mapping,
            old_action_size=old_action_size,
            new_action_size=new_action_size,
            num_atoms=num_atoms,
        )

    def _update_env_wrapper(self, new_phase: int) -> None:
        """Update environment wrapper for new phase.

        Args:
            new_phase: New phase number.
        """
        if hasattr(self.env_wrapper, "update_phase"):
            self.env_wrapper.update_phase(new_phase)
            self.logger.info(f"Environment wrapper updated to phase {new_phase}")

    def _rebuild_algorithm(
        self,
        trainer: "TianshouTrainer",
        new_action_size: int,
    ) -> None:
        """Rebuild Tianshou algorithm after network modification.

        Creates the correct policy and algorithm based on algo_type:
        - dqn/dueling_dqn: DiscreteQLearningPolicy + DQN
        - c51: C51Policy + C51
        - rainbow: C51Policy + RainbowDQN

        Args:
            trainer: TianshouTrainer instance.
            new_action_size: New action space size.
        """
        # Get current configuration from trainer
        training_config = trainer.training_config
        lr = training_config.get("learning_rate", 1e-3)
        gamma = training_config.get("gamma", 0.99)
        target_update = training_config.get("target_update_freq", 1000)

        # Get n-step from trainer (set during initial algorithm build)
        n_step = getattr(trainer, "n_step", 1)

        # Get current epsilon
        current_eps = (
            self.epsilon_boost
            if self.epsilon_boost is not None
            else getattr(trainer.policy, "eps_training", 1.0)
        )
        final_eps = training_config.get("exploration", {}).get("final_eps", 0.05)

        # Update action space reference
        new_action_space = gym.spaces.Discrete(new_action_size)

        # Detect algorithm type
        algo_type = self._detect_algo_type(trainer)

        if algo_type in ("c51", "rainbow"):
            # Import distributional RL classes
            from tianshou.algorithm import C51, RainbowDQN
            from tianshou.algorithm.modelfree.c51 import C51Policy

            # Distributional RL: Use C51Policy
            trainer.policy = C51Policy(
                model=trainer.network,
                action_space=new_action_space,
                observation_space=trainer.observation_space,
                num_atoms=self.num_atoms,
                v_min=self.v_min,
                v_max=self.v_max,
                eps_training=current_eps,
                eps_inference=final_eps,
            )

            # Move policy to correct device
            if hasattr(trainer, "device"):
                trainer.policy = trainer.policy.to(trainer.device)

            if algo_type == "c51":
                trainer.algorithm = C51(
                    policy=trainer.policy,
                    optim=AdamOptimizerFactory(lr=lr),
                    gamma=gamma,
                    n_step_return_horizon=n_step,
                    target_update_freq=target_update,
                )
                self.logger.info("C51 algorithm rebuilt with new action space")
            else:  # rainbow
                trainer.algorithm = RainbowDQN(
                    policy=trainer.policy,
                    optim=AdamOptimizerFactory(lr=lr),
                    gamma=gamma,
                    n_step_return_horizon=n_step,
                    target_update_freq=target_update,
                )
                self.logger.info("RainbowDQN algorithm rebuilt with new action space")
        else:
            # Standard DQN / Dueling DQN
            trainer.policy = DiscreteQLearningPolicy(
                model=trainer.network,
                action_space=new_action_space,
                observation_space=trainer.observation_space,
                eps_training=current_eps,
                eps_inference=final_eps,
            )

            trainer.algorithm = DQN(
                policy=trainer.policy,
                optim=AdamOptimizerFactory(lr=lr),
                gamma=gamma,
                n_step_return_horizon=n_step,
                target_update_freq=target_update,
            )
            self.logger.info("DQN algorithm rebuilt with new action space")

        # Update collector to use new policy
        # Note: Tianshou collector stores algorithm.policy, not algorithm itself
        trainer.train_collector.policy = trainer.policy

        # Reset collector state for the new policy
        # This is critical because the collector caches observations and
        # the policy reference - we need a clean state after policy change
        trainer.train_collector.reset()

        # Update action space reference in trainer
        trainer.action_space = new_action_space

    def _detect_algo_type(self, trainer: "TianshouTrainer") -> str:
        """Detect algorithm type from trainer or stored config.

        Detection priority:
        1. Explicitly set algo_type in callback (not 'auto')
        2. Trainer's algo_type attribute
        3. Network type inference
        4. Default to 'dqn'

        Args:
            trainer: TianshouTrainer instance.

        Returns:
            Algorithm type string: 'dqn', 'dueling_dqn', 'c51', or 'rainbow'.
        """
        # Use explicitly set type
        if self.algo_type != "auto":
            return self.algo_type

        # Try trainer attribute
        if hasattr(trainer, "algo_type"):
            return trainer.algo_type

        # Infer from network type
        network_class = trainer.network.__class__.__name__
        if network_class == "RainbowDRQN":
            return "rainbow"
        elif network_class == "C51DRQN":
            return "c51"
        elif network_class == "DuelingDRQN":
            return "dueling_dqn"
        else:
            return "dqn"

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Called at the end of training.

        Logs curriculum statistics and saves final phase transition log.

        Args:
            trainer: TianshouTrainer instance.
        """
        self.logger.info(
            f"Curriculum training complete:"
            f"\n  Final phase: {self.phase_manager.current_phase}"
            f"\n  Total transitions: {self._total_transitions}"
            f"\n  Phase history: {self.phase_manager.phase_history}"
        )

        # Save final transition log with summary
        if self._transition_log_path is not None:
            summary = {
                "transitions": self._transition_log,
                "summary": {
                    "total_transitions": self._total_transitions,
                    "final_phase": self.phase_manager.current_phase,
                    "total_phases": self.hierarchy.total_phases,
                    "phase_history": self.phase_manager.phase_history,
                    "training_completed": datetime.now().isoformat(),
                },
            }
            try:
                with open(self._transition_log_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                self.logger.info(
                    f"Phase transition log saved to: {self._transition_log_path}"
                )
            except (IOError, OSError) as e:
                self.logger.warning(f"Failed to save final transition log: {e}")

    @property
    def current_phase(self) -> int:
        """Get current curriculum phase."""
        return self.phase_manager.current_phase

    @property
    def is_final_phase(self) -> bool:
        """Check if in final phase."""
        return self.phase_manager.is_final_phase

    @property
    def transition_progress(self) -> float:
        """Get progress towards next transition."""
        return self.phase_manager.transition_progress

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpointing.

        Returns:
            Dict containing callback state including algorithm config.
        """
        return {
            "phase_manager_state": self.phase_manager.get_state(),
            "total_transitions": self._total_transitions,
            "last_transition_step": self._last_transition_step,
            # Algorithm-specific config for restore
            "algo_type": self.algo_type,
            "num_atoms": self.num_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: Dict from get_state().
        """
        if "phase_manager_state" in state:
            self.phase_manager.load_state(state["phase_manager_state"])

        self._total_transitions = state.get("total_transitions", 0)
        self._last_transition_step = state.get("last_transition_step", 0)

        # Restore algorithm config
        if "algo_type" in state:
            self.algo_type = state["algo_type"]
        if "num_atoms" in state:
            self.num_atoms = state["num_atoms"]
        if "v_min" in state:
            self.v_min = state["v_min"]
        if "v_max" in state:
            self.v_max = state["v_max"]

        self.logger.info(
            f"Callback state loaded: phase {self.current_phase}, "
            f"{self._total_transitions} transitions, algo_type={self.algo_type}"
        )
