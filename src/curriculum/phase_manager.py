"""Phase Manager for Dynamic Curriculum Learning.

This module manages phase transitions during curriculum learning training.
It tracks the current phase, detects plateau conditions, and triggers
phase transitions when appropriate.

Classes:
    PhaseRuntimeConfig: Runtime configuration per phase.
    PhaseTransitionTrigger: Plateau detection for transitions.
    DynamicPhaseManager: Main phase management logic.

Example:
    >>> hierarchy = extractor.extract()
    >>> manager = DynamicPhaseManager(
    ...     hierarchy=hierarchy,
    ...     timesteps_per_action=2500,
    ...     patience_per_action=150,
    ... )
    >>>
    >>> # During training loop
    >>> transitioned = manager.update(reward=0.5)
    >>> if transitioned:
    ...     mapping = manager.get_last_mitosis_mapping()
    ...     # Perform weight mitosis with mapping
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from src.curriculum.hierarchy_extractor import DynamicHierarchy, PhaseDefinition
from src.utils.logging.logging_manager import get_logger


@dataclass
class PhaseRuntimeConfig:
    """Runtime configuration for a curriculum phase.

    Auto-calculated based on action space size to ensure sufficient
    exploration time for each phase.

    Attributes:
        phase_number: Phase identifier.
        action_space_size: Number of actions in this phase.
        min_timesteps: Minimum timesteps before transition allowed.
        patience: Plateau detection patience (reward window size).
        plateau_threshold: Threshold for plateau detection.
    """

    phase_number: int
    action_space_size: int
    min_timesteps: int
    patience: int
    plateau_threshold: float = 0.01

    @classmethod
    def from_phase_def(
        cls,
        phase_def: PhaseDefinition,
        timesteps_per_action: int = 2500,
        patience_per_action: int = 150,
        plateau_threshold: float = 0.01,
    ) -> "PhaseRuntimeConfig":
        """Create runtime config from phase definition.

        Args:
            phase_def: Phase definition from hierarchy.
            timesteps_per_action: Timesteps per action for min calculation.
            patience_per_action: Patience per action for window size.
            plateau_threshold: Threshold for plateau detection.

        Returns:
            PhaseRuntimeConfig instance.
        """
        action_count = phase_def.action_space_size

        return cls(
            phase_number=phase_def.phase_number,
            action_space_size=action_count,
            min_timesteps=action_count * timesteps_per_action,
            patience=action_count * patience_per_action,
            plateau_threshold=plateau_threshold,
        )


class PhaseTransitionTrigger:
    """Plateau detection for phase transitions.

    Uses a sliding window of rewards to detect when learning has
    plateaued and the agent is ready for the next phase.

    Attributes:
        patience: Window size for plateau detection.
        threshold: Threshold for considering rewards plateaued.
        reward_history: Sliding window of rewards.
    """

    def __init__(
        self,
        patience: int = 100,
        threshold: float = 0.01,
    ):
        """Initialize transition trigger.

        Args:
            patience: Number of rewards to track for plateau detection.
            threshold: Minimum improvement to reset plateau counter.
        """
        self.patience = patience
        self.threshold = threshold
        self.reward_history: Deque[float] = deque(maxlen=patience)
        self._best_mean_reward: Optional[float] = None
        self._plateau_counter: int = 0

    def update(self, reward: float) -> bool:
        """Update with new reward and check for plateau.

        Args:
            reward: Latest episode reward.

        Returns:
            True if plateau detected, False otherwise.
        """
        self.reward_history.append(reward)

        # Need enough history for comparison
        if len(self.reward_history) < self.patience // 2:
            return False

        # Calculate current mean
        current_mean = sum(self.reward_history) / len(self.reward_history)

        # Initialize best if first time
        if self._best_mean_reward is None:
            self._best_mean_reward = current_mean
            return False

        # Check for improvement
        improvement = current_mean - self._best_mean_reward

        if improvement > self.threshold:
            # Significant improvement - reset counter
            self._best_mean_reward = current_mean
            self._plateau_counter = 0
        else:
            # No improvement - increment counter
            self._plateau_counter += 1

        # Plateau detected when counter exceeds patience
        return self._plateau_counter >= self.patience

    def reset(self) -> None:
        """Reset trigger state for new phase."""
        self.reward_history.clear()
        self._best_mean_reward = None
        self._plateau_counter = 0

    @property
    def plateau_progress(self) -> float:
        """Get progress towards plateau (0.0 to 1.0)."""
        return min(1.0, self._plateau_counter / self.patience)


class DynamicPhaseManager:
    """Manager for curriculum phase transitions.

    Tracks the current phase, monitors training progress, and triggers
    phase transitions when appropriate conditions are met.

    Attributes:
        hierarchy: Complete curriculum hierarchy.
        current_phase: Current phase number (1-indexed).
        phase_configs: Runtime configs for each phase.
        transition_trigger: Plateau detector for current phase.
        timesteps_in_phase: Timesteps spent in current phase.
        logger: Logger instance.

    Example:
        >>> manager = DynamicPhaseManager(hierarchy)
        >>>
        >>> # Training loop
        >>> for step in range(total_steps):
        ...     reward = env.step(action)
        ...     if manager.update(reward):
        ...         # Phase transition occurred
        ...         mapping = manager.get_last_mitosis_mapping()
        ...         perform_weight_mitosis(network, mapping)
    """

    def __init__(
        self,
        hierarchy: DynamicHierarchy,
        timesteps_per_action: int = 2500,
        patience_per_action: int = 150,
        plateau_threshold: float = 0.01,
    ):
        """Initialize phase manager.

        Args:
            hierarchy: Extracted curriculum hierarchy.
            timesteps_per_action: Timesteps per action for min phase duration.
            patience_per_action: Patience per action for plateau detection.
            plateau_threshold: Threshold for plateau detection.
        """
        self.hierarchy = hierarchy
        self.logger = get_logger("DynamicPhaseManager")

        # Configuration
        self.timesteps_per_action = timesteps_per_action
        self.patience_per_action = patience_per_action
        self.plateau_threshold = plateau_threshold

        # Build runtime configs for all phases
        self.phase_configs: Dict[int, PhaseRuntimeConfig] = {}
        for phase_num, phase_def in hierarchy.phases.items():
            self.phase_configs[phase_num] = PhaseRuntimeConfig.from_phase_def(
                phase_def,
                timesteps_per_action=timesteps_per_action,
                patience_per_action=patience_per_action,
                plateau_threshold=plateau_threshold,
            )

        # Initialize to phase 1
        self.current_phase = 1
        self.timesteps_in_phase = 0
        self.episodes_in_phase = 0

        # Create transition trigger for current phase
        self._create_trigger_for_current_phase()

        # Track last transition for mitosis
        self._last_transition_from: Optional[int] = None
        self._last_transition_to: Optional[int] = None
        self._last_mitosis_mapping: Optional[Dict[int, List[int]]] = None

        # Phase history
        self.phase_history: List[Dict] = []

        self.logger.info(
            f"PhaseManager initialized: {hierarchy.total_phases} phases, "
            f"starting at phase 1"
        )

    def _create_trigger_for_current_phase(self) -> None:
        """Create plateau trigger for current phase."""
        config = self.phase_configs[self.current_phase]
        self.transition_trigger = PhaseTransitionTrigger(
            patience=config.patience,
            threshold=config.plateau_threshold,
        )

    def update(self, reward: float, timesteps: int = 1) -> bool:
        """Update manager with training progress.

        Args:
            reward: Episode reward (or step reward if using step updates).
            timesteps: Number of timesteps to add (default 1).

        Returns:
            True if phase transition occurred, False otherwise.
        """
        self.timesteps_in_phase += timesteps

        # Check if we're at the final phase (no more transitions)
        if self.current_phase >= self.hierarchy.total_phases:
            return False

        # Check minimum timesteps requirement
        config = self.phase_configs[self.current_phase]
        if self.timesteps_in_phase < config.min_timesteps:
            # Update trigger but don't allow transition yet
            self.transition_trigger.update(reward)
            return False

        # Check for plateau
        plateau_detected = self.transition_trigger.update(reward)

        if plateau_detected:
            return self._perform_transition()

        return False

    def update_episode(self, episode_reward: float, episode_timesteps: int) -> bool:
        """Update manager at episode end.

        Convenience method for episode-based updates.

        Args:
            episode_reward: Total episode reward.
            episode_timesteps: Timesteps in this episode.

        Returns:
            True if phase transition occurred, False otherwise.
        """
        self.episodes_in_phase += 1
        return self.update(episode_reward, timesteps=episode_timesteps)

    def _perform_transition(self) -> bool:
        """Perform phase transition.

        Returns:
            True if transition successful, False if at final phase.
        """
        if self.current_phase >= self.hierarchy.total_phases:
            return False

        old_phase = self.current_phase
        new_phase = self.current_phase + 1

        # Get mitosis mapping before transition
        mitosis_mapping = self.hierarchy.get_mitosis_mapping(old_phase, new_phase)

        # Record history
        self.phase_history.append({
            "from_phase": old_phase,
            "to_phase": new_phase,
            "timesteps_in_phase": self.timesteps_in_phase,
            "episodes_in_phase": self.episodes_in_phase,
        })

        # Store last transition info
        self._last_transition_from = old_phase
        self._last_transition_to = new_phase
        self._last_mitosis_mapping = mitosis_mapping

        # Update phase
        self.current_phase = new_phase
        self.timesteps_in_phase = 0
        self.episodes_in_phase = 0

        # Create new trigger for new phase
        self._create_trigger_for_current_phase()

        self.logger.info(
            f"Phase transition: {old_phase} -> {new_phase} "
            f"(action space: {self.current_phase_def.action_space_size})"
        )

        return True

    def force_transition(self) -> bool:
        """Force a phase transition regardless of conditions.

        Useful for debugging or manual control.

        Returns:
            True if transition successful, False if at final phase.
        """
        return self._perform_transition()

    def get_last_mitosis_mapping(self) -> Optional[Dict[int, List[int]]]:
        """Get the mitosis mapping from the last transition.

        Returns:
            Dict mapping old actions to new actions, or None if no transition yet.
        """
        return self._last_mitosis_mapping

    def get_phase_mitosis_mapping(
        self, from_phase: int, to_phase: int
    ) -> Dict[int, List[int]]:
        """Get mitosis mapping between specific phases.

        Args:
            from_phase: Source phase number.
            to_phase: Target phase number.

        Returns:
            Dict mapping source actions to target actions.
        """
        return self.hierarchy.get_mitosis_mapping(from_phase, to_phase)

    @property
    def current_phase_def(self) -> PhaseDefinition:
        """Get current phase definition."""
        return self.hierarchy.get_phase(self.current_phase)

    @property
    def current_action_space_size(self) -> int:
        """Get current phase action space size."""
        return self.current_phase_def.action_space_size

    @property
    def is_final_phase(self) -> bool:
        """Check if currently in final phase."""
        return self.current_phase >= self.hierarchy.total_phases

    @property
    def transition_progress(self) -> float:
        """Get progress towards next transition (0.0 to 1.0).

        Combines timestep progress and plateau progress.
        """
        if self.is_final_phase:
            return 1.0

        config = self.phase_configs[self.current_phase]

        # Timestep progress (must meet minimum first)
        timestep_progress = min(
            1.0, self.timesteps_in_phase / config.min_timesteps
        )

        if timestep_progress < 1.0:
            # Still in minimum timesteps period
            return timestep_progress * 0.5

        # Past minimum - use plateau progress
        plateau_progress = self.transition_trigger.plateau_progress
        return 0.5 + (plateau_progress * 0.5)

    def get_state(self) -> Dict:
        """Get serializable state for checkpointing.

        Returns:
            Dict containing manager state.
        """
        return {
            "current_phase": self.current_phase,
            "timesteps_in_phase": self.timesteps_in_phase,
            "episodes_in_phase": self.episodes_in_phase,
            "phase_history": self.phase_history,
            "reward_history": list(self.transition_trigger.reward_history),
            "best_mean_reward": self.transition_trigger._best_mean_reward,
            "plateau_counter": self.transition_trigger._plateau_counter,
        }

    def load_state(self, state: Dict) -> None:
        """Load state from checkpoint.

        Args:
            state: Dict from get_state().
        """
        self.current_phase = state["current_phase"]
        self.timesteps_in_phase = state["timesteps_in_phase"]
        self.episodes_in_phase = state["episodes_in_phase"]
        self.phase_history = state["phase_history"]

        # Recreate trigger for current phase
        self._create_trigger_for_current_phase()

        # Restore trigger state
        for reward in state.get("reward_history", []):
            self.transition_trigger.reward_history.append(reward)
        self.transition_trigger._best_mean_reward = state.get("best_mean_reward")
        self.transition_trigger._plateau_counter = state.get("plateau_counter", 0)

        self.logger.info(f"State loaded: phase {self.current_phase}")
