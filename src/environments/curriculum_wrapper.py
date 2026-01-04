"""Curriculum Environment Wrapper for Action Space Mapping.

This module provides a Gymnasium wrapper that maps coarse curriculum
actions to fine-grained leaf cluster actions. During early phases,
the agent selects from a smaller action space, and the wrapper maps
these actions to the full leaf cluster space.

Classes:
    CurriculumEnvironmentWrapper: Main wrapper for action mapping.

Example:
    >>> wrapper = CurriculumEnvironmentWrapper(
    ...     env=base_env,
    ...     hierarchy=hierarchy,
    ...     initial_phase=1,
    ... )
    >>>
    >>> # Agent selects action 0 (coarse parent)
    >>> # Wrapper maps to random leaf descendant
    >>> obs, reward, done, truncated, info = wrapper.step(action=0)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.curriculum.hierarchy_extractor import DynamicHierarchy, PhaseDefinition
from src.utils.logging.logging_manager import get_logger


class CurriculumEnvironmentWrapper(gym.Wrapper):
    """Environment wrapper for curriculum action space mapping.

    This wrapper modifies the action space based on the current
    curriculum phase. In early phases, the agent selects from a
    coarse action space, and the wrapper maps these actions to
    fine-grained leaf cluster actions.

    Action mapping strategy:
    - Agent selects action in current phase's action space
    - Wrapper looks up the leaf descendants of this action
    - Wrapper randomly samples one leaf descendant
    - Underlying environment receives the leaf action

    Attributes:
        hierarchy: Curriculum hierarchy.
        current_phase: Current phase number.
        phase_def: Current phase definition.
        logger: Logger instance.

    Example:
        >>> wrapper = CurriculumEnvironmentWrapper(env, hierarchy)
        >>>
        >>> # Phase 1: 2 actions available
        >>> assert wrapper.action_space.n == 2
        >>>
        >>> # Agent selects action 0, wrapper maps to leaf
        >>> obs, reward, done, truncated, info = wrapper.step(0)
        >>>
        >>> # Update to phase 2: 4 actions available
        >>> wrapper.update_phase(2)
        >>> assert wrapper.action_space.n == 4
    """

    def __init__(
        self,
        env: gym.Env,
        hierarchy: DynamicHierarchy,
        initial_phase: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize curriculum wrapper.

        Args:
            env: Base Gymnasium environment.
            hierarchy: Curriculum hierarchy from HierarchyExtractor.
            initial_phase: Starting phase number (1-indexed).
            seed: Random seed for action sampling.
        """
        super().__init__(env)

        self.hierarchy = hierarchy
        self.logger = get_logger("CurriculumEnvironmentWrapper")

        # Set up random generator
        self._rng = np.random.default_rng(seed)

        # Initialize to starting phase
        self._current_phase = initial_phase
        self._phase_def = hierarchy.get_phase(initial_phase)

        if self._phase_def is None:
            raise ValueError(f"Invalid initial phase: {initial_phase}")

        # Update action space to match current phase
        self._update_action_space()

        # Store original action space for reference
        self._original_action_space = env.action_space

        self.logger.info(
            f"CurriculumWrapper initialized: phase={initial_phase}, "
            f"action_space={self.action_space.n}"
        )

    def _update_action_space(self) -> None:
        """Update action space based on current phase.

        Also updates the unwrapped env's action_space to work around
        Tianshou's DummyEnvWorker.get_env_attr which uses env.unwrapped.
        """
        new_action_size = self._phase_def.action_space_size
        self.action_space = spaces.Discrete(new_action_size)

        # IMPORTANT: Also update the unwrapped env's action_space
        # This is needed because Tianshou's DummyEnvWorker.get_env_attr()
        # uses env.unwrapped which bypasses all wrappers
        self.env.unwrapped.action_space = spaces.Discrete(new_action_size)

    def update_phase(self, new_phase: int) -> None:
        """Update wrapper to new curriculum phase.

        Args:
            new_phase: New phase number.

        Raises:
            ValueError: If phase is invalid.
        """
        phase_def = self.hierarchy.get_phase(new_phase)

        if phase_def is None:
            raise ValueError(f"Invalid phase: {new_phase}")

        old_phase = self._current_phase
        old_actions = self._phase_def.action_space_size

        self._current_phase = new_phase
        self._phase_def = phase_def
        self._update_action_space()

        self.logger.info(
            f"Phase updated: {old_phase} -> {new_phase}, "
            f"actions: {old_actions} -> {self.action_space.n}"
        )

    def _map_action(self, action: int) -> int:
        """Map curriculum action to leaf cluster action.

        Args:
            action: Action in current phase's action space.

        Returns:
            Leaf cluster action for underlying environment.
        """
        # Get leaf descendants for this action
        leaf_descendants = self._phase_def.get_leaf_actions_for_action(action)

        if not leaf_descendants:
            # Fallback: use action directly (shouldn't happen)
            self.logger.warning(
                f"No leaf descendants for action {action}, using directly"
            )
            return action

        if len(leaf_descendants) == 1:
            # Only one leaf - use it directly
            return leaf_descendants[0]

        # Multiple leaves - sample randomly
        selected = self._rng.choice(leaf_descendants)
        return int(selected)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Maps the curriculum action to a leaf action and executes it.

        Args:
            action: Action in current phase's action space.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            Info dict includes curriculum-related information.
        """
        # Validate action
        if not 0 <= action < self.action_space.n:
            raise ValueError(
                f"Invalid action {action} for action space of size "
                f"{self.action_space.n}"
            )

        # Map to leaf action
        leaf_action = self._map_action(action)

        # Execute in underlying environment
        obs, reward, terminated, truncated, info = self.env.step(leaf_action)

        # Add curriculum info to info dict
        info["curriculum"] = {
            "phase": self._current_phase,
            "selected_action": action,
            "mapped_leaf_action": leaf_action,
            "action_space_size": self.action_space.n,
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options.

        Returns:
            Tuple of (observation, info).
        """
        # Update RNG if seed provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, info = self.env.reset(seed=seed, options=options)

        # Add curriculum info
        info["curriculum"] = {
            "phase": self._current_phase,
            "action_space_size": self.action_space.n,
        }

        return obs, info

    @property
    def current_phase(self) -> int:
        """Get current curriculum phase."""
        return self._current_phase

    @property
    def phase_definition(self) -> PhaseDefinition:
        """Get current phase definition."""
        return self._phase_def

    @property
    def is_final_phase(self) -> bool:
        """Check if currently in final phase."""
        return self._current_phase >= self.hierarchy.total_phases

    def get_action_mapping(self) -> Dict[int, List[int]]:
        """Get current action-to-leaf mapping.

        Returns:
            Dict mapping curriculum actions to leaf cluster IDs.
        """
        mapping = {}
        for action in range(self.action_space.n):
            mapping[action] = self._phase_def.get_leaf_actions_for_action(action)
        return mapping


def create_curriculum_env(
    base_env: gym.Env,
    hierarchy: DynamicHierarchy,
    initial_phase: int = 1,
    seed: Optional[int] = None,
) -> CurriculumEnvironmentWrapper:
    """Convenience function to create a curriculum-wrapped environment.

    Args:
        base_env: Base Gymnasium environment.
        hierarchy: Curriculum hierarchy from HierarchyExtractor.
        initial_phase: Starting phase number.
        seed: Random seed.

    Returns:
        CurriculumEnvironmentWrapper instance.
    """
    return CurriculumEnvironmentWrapper(
        env=base_env,
        hierarchy=hierarchy,
        initial_phase=initial_phase,
        seed=seed,
    )
