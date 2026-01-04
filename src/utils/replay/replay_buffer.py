"""Backend-agnostic replay buffer and experience management."""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


# Standard experience tuple format
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayExperience:
    """Wrapper for experience tuples with reward shaping capabilities."""

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        """Initialize experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.info = info or {}

        # Track original reward for shaping
        self.original_reward = reward
        self.shaped_rewards: Dict[str, float] = {}

    def add_shaped_reward(self, source: str, reward: float) -> None:
        """Add a shaped reward component.

        Args:
            source: Source identifier (e.g., 'human', 'perceiver')
            reward: Reward value to add
        """
        self.shaped_rewards[source] = reward
        self.reward = self.original_reward + sum(self.shaped_rewards.values())

    def to_tuple(self) -> Experience:
        """Convert to standard experience tuple.

        Returns:
            Experience namedtuple
        """
        return Experience(
            self.state, self.action, self.reward, self.next_state, self.done
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Experience as dictionary
        """
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "info": self.info,
            "original_reward": self.original_reward,
            "shaped_rewards": self.shaped_rewards.copy(),
        }


class ReplayAdapter:
    """Backend-agnostic replay buffer adapter."""

    def __init__(self, max_size: int = 1000000):
        """Initialize replay adapter.

        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.experiences: List[ReplayExperience] = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
        """
        experience = ReplayExperience(state, action, reward, next_state, done, info)

        if len(self.experiences) < self.max_size:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
            self.position = (self.position + 1) % self.max_size

    def add_experience(self, experience: ReplayExperience) -> None:
        """Add pre-constructed experience to buffer.

        Args:
            experience: ReplayExperience instance
        """
        if len(self.experiences) < self.max_size:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size: int) -> List[ReplayExperience]:
        """Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if len(self.experiences) < batch_size:
            return self.experiences.copy()

        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def apply_reward_shaping(
        self,
        filter_fn: Optional[Callable[[ReplayExperience], bool]] = None,
        shaping_fn: Optional[Callable[[ReplayExperience], float]] = None,
        source: str = "shaped",
    ) -> int:
        """Apply reward shaping to experiences in buffer.

        Args:
            filter_fn: Function to filter experiences (optional)
            shaping_fn: Function to compute shaped reward (optional)
            source: Source identifier for shaped rewards

        Returns:
            Number of experiences modified
        """
        modified_count = 0

        for experience in self.experiences:
            if filter_fn and not filter_fn(experience):
                continue

            if shaping_fn:
                shaped_reward = shaping_fn(experience)
                experience.add_shaped_reward(source, shaped_reward)
                modified_count += 1

        return modified_count

    def get_recent_experiences(self, n: int) -> List[ReplayExperience]:
        """Get the n most recent experiences.

        Args:
            n: Number of recent experiences to get

        Returns:
            List of recent experiences
        """
        if not self.experiences:
            return []

        if len(self.experiences) < self.max_size:
            # Buffer not full, return last n
            return (
                self.experiences[-n:]
                if n < len(self.experiences)
                else self.experiences.copy()
            )
        else:
            # Buffer is full, need to handle circular buffer
            if n >= len(self.experiences):
                return self.experiences.copy()

            # Get n experiences before current position
            start_idx = (self.position - n) % self.max_size
            if start_idx < self.position:
                return self.experiences[start_idx : self.position]
            else:
                return self.experiences[start_idx:] + self.experiences[: self.position]

    def size(self) -> int:
        """Get current buffer size.

        Returns:
            Number of experiences in buffer
        """
        return len(self.experiences)

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.experiences.clear()
        self.position = 0

    def to_sb3_format(self, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Convert experiences to SB3 replay buffer format.

        Args:
            batch_size: Optional batch size for sampling

        Returns:
            Dictionary with SB3-compatible arrays
        """
        experiences = self.sample(batch_size) if batch_size else self.experiences

        if not experiences:
            return {
                "observations": np.array([]),
                "actions": np.array([]),
                "rewards": np.array([]),
                "next_observations": np.array([]),
                "dones": np.array([]),
            }

        return {
            "observations": np.array([exp.state for exp in experiences]),
            "actions": np.array([exp.action for exp in experiences]),
            "rewards": np.array([exp.reward for exp in experiences]),
            "next_observations": np.array([exp.next_state for exp in experiences]),
            "dones": np.array([exp.done for exp in experiences]),
        }

    def to_tfa_format(self, batch_size: Optional[int] = None) -> List[Experience]:
        """Convert experiences to TF-Agents format.

        Args:
            batch_size: Optional batch size for sampling

        Returns:
            List of Experience tuples
        """
        experiences = self.sample(batch_size) if batch_size else self.experiences
        return [exp.to_tuple() for exp in experiences]
