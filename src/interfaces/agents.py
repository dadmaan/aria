"""Abstract base classes for agents in the multi-agent reinforcement learning framework.

This module defines the core interfaces for agents that participate in the music
generation process. It includes perceiving agents that analyze musical structure,
generative agents that create sequences, and the music environment interface.
These abstract base classes ensure that all implementations follow the same
contract, enabling interoperability between different agent types and environments.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class PerceivingAgent(metaclass=abc.ABCMeta):
    """Abstract base class for agents that analyze musical structure and provide context.

    The PerceivingAgent is responsible for:
    - Analyzing musical sequences to discover structure (motifs, patterns)
    - Providing cluster IDs that represent structural elements
    - Defining the action space for generative agents
    - Computing structural/similarity rewards
    """

    @abc.abstractmethod
    def evaluate_sequence(self, sequence: np.ndarray) -> Dict[str, float]:
        """Evaluate a musical sequence and return reward components.

        Args:
            sequence: Array of cluster IDs representing a musical sequence

        Returns:
            Dictionary with reward components, e.g.:
            {'similarity': 0.8, 'structure': 0.6, 'coherence': 0.9}
        """

    @abc.abstractmethod
    def get_unique_cluster_ids_list(self) -> List[int]:
        """Get the list of all available cluster IDs.

        Returns:
            List of cluster IDs that can be used as actions
        """

    @abc.abstractmethod
    def get_action_space(self) -> int:
        """Get the size of the discrete action space.

        Returns:
            Number of available actions (cluster IDs)
        """

    @abc.abstractmethod
    def seed(self, random_state: Optional[int]) -> None:
        """Seed random number generators for reproducibility.

        Args:
            random_state: Random seed value
        """


class GenerativeAgent(metaclass=abc.ABCMeta):
    """Abstract base class for RL agents that generate musical sequences.

    The GenerativeAgent is responsible for:
    - Learning from experiences in the environment
    - Predicting next actions given current state
    - Saving and loading trained models
    """

    @abc.abstractmethod
    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Predict the next action given an observation.

        Args:
            observation: Current state/observation
            deterministic: Whether to use deterministic (greedy) policy

        Returns:
            Tuple of (action, optional_info_dict)
        """

    @abc.abstractmethod
    def learn(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Update the agent's policy from experience.

        Args:
            *args: Framework-specific learning arguments
            **kwargs: Framework-specific learning keyword arguments

        Returns:
            Optional dictionary with learning metrics
        """

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's model to disk.

        Args:
            path: File path to save the model
        """

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's model from disk.

        Args:
            path: File path to load the model from
        """

    @abc.abstractmethod
    def seed(self, random_state: Optional[int]) -> None:
        """Seed random number generators for reproducibility.

        Args:
            random_state: Random seed value
        """


class MusicEnvironment(metaclass=abc.ABCMeta):
    """Abstract base class for music generation environments.

    Defines the interface for environments that support both
    Gymnasium and TF-Agents APIs.
    """

    @abc.abstractmethod
    def reset(self) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Reset environment to initial state.

        Returns:
            Initial observation (Gym) or (observation, info) tuple (Gym v0.26+)
        """

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, done, info)
        """

    @abc.abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generators.

        Args:
            seed: Random seed

        Returns:
            List of seeds used by RNG(s)
        """

    @abc.abstractmethod
    def get_observation_space(self) -> Any:
        """Get the observation space definition.

        Returns:
            Space object (gym.Space or tf_agents.specs.ArraySpec)
        """

    @abc.abstractmethod
    def get_action_space(self) -> Any:
        """Get the action space definition.

        Returns:
            Space object (gym.Space or tf_agents.specs.BoundedArraySpec)
        """
