"""Exploration strategies for HIL preference-guided simulation.

This module provides configurable exploration strategies that support
decaying exploration rates, including epsilon-greedy and Boltzmann
(softmax) exploration with warmup phases and multiple decay schedules.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ExplorationStrategy(ABC):
    """Base class for exploration strategies.

    Defines the interface for exploration strategies used during
    HIL preference-guided simulation. All strategies support:
    - Warmup phase with constant exploration rate
    - Decay schedule (linear or exponential)
    - Action selection based on Q-values
    """

    @abstractmethod
    def select_action(
        self,
        q_values: torch.Tensor,
        iteration: int,
        total_iterations: int,
    ) -> int:
        """Select action given Q-values and current iteration.

        Args:
            q_values: Q-values tensor of shape (1, num_actions).
            iteration: Current iteration number.
            total_iterations: Total number of iterations in simulation.

        Returns:
            Selected action index.
        """
        pass

    @abstractmethod
    def get_exploration_rate(self, iteration: int, total_iterations: int) -> float:
        """Get current exploration rate for logging.

        Args:
            iteration: Current iteration number.
            total_iterations: Total number of iterations in simulation.

        Returns:
            Current exploration rate (epsilon or temperature).
        """
        pass


class EpsilonGreedyExploration(ExplorationStrategy):
    """Epsilon-greedy exploration with decay schedule.

    Implements epsilon-greedy action selection where with probability
    epsilon a random action is selected, otherwise the greedy action
    (highest Q-value) is selected.

    Supports warmup phase and linear/exponential decay schedules.

    Attributes:
        initial_eps: Starting epsilon value (default 0.5).
        final_eps: Final epsilon value after decay (default 0.05).
        decay_schedule: Type of decay ("linear" or "exponential").
        warmup: Number of iterations to maintain initial epsilon.
    """

    def __init__(
        self,
        initial_eps: float = 0.5,
        final_eps: float = 0.05,
        decay_schedule: str = "linear",
        warmup: int = 50,
    ):
        """Initialize epsilon-greedy exploration.

        Args:
            initial_eps: Starting epsilon value.
            final_eps: Final epsilon value after full decay.
            decay_schedule: "linear" or "exponential" decay.
            warmup: Warmup iterations before decay starts.
        """
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.decay_schedule = decay_schedule
        self.warmup = warmup

        logger.info(
            "EpsilonGreedyExploration: initial=%.3f, final=%.3f, "
            "schedule=%s, warmup=%d",
            initial_eps, final_eps, decay_schedule, warmup
        )

    def get_exploration_rate(self, iteration: int, total_iterations: int) -> float:
        """Calculate epsilon value for current iteration.

        Args:
            iteration: Current iteration number.
            total_iterations: Total number of iterations.

        Returns:
            Epsilon value for current iteration.
        """
        # Warmup phase: maintain initial epsilon
        if iteration < self.warmup:
            return self.initial_eps

        # Calculate progress after warmup
        progress = (iteration - self.warmup) / max(total_iterations - self.warmup, 1)
        progress = min(progress, 1.0)

        if self.decay_schedule == "exponential":
            # Exponential decay: eps = initial * (final/initial)^progress
            ratio = self.final_eps / max(self.initial_eps, 1e-10)
            return self.initial_eps * (ratio ** progress)
        else:
            # Linear decay (default)
            return self.initial_eps - (self.initial_eps - self.final_eps) * progress

    def select_action(
        self,
        q_values: torch.Tensor,
        iteration: int,
        total_iterations: int,
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            q_values: Q-values tensor of shape (1, num_actions).
            iteration: Current iteration number.
            total_iterations: Total number of iterations.

        Returns:
            Selected action index.
        """
        eps = self.get_exploration_rate(iteration, total_iterations)

        if np.random.random() < eps:
            # Random exploration
            num_actions = q_values.shape[1]
            return np.random.randint(num_actions)
        else:
            # Greedy exploitation
            return q_values.argmax(dim=1).item()


class BoltzmannExploration(ExplorationStrategy):
    """Boltzmann (softmax) exploration with temperature decay.

    Implements softmax action selection where actions are sampled
    according to a probability distribution derived from Q-values:

        P(a) = exp(Q(a)/T) / sum(exp(Q/T))

    Higher temperature leads to more uniform (exploratory) distribution,
    lower temperature approaches greedy selection.

    Attributes:
        initial_temp: Starting temperature (default 2.0).
        final_temp: Final temperature after decay (default 0.1).
        decay_schedule: Type of decay ("linear" or "exponential").
        warmup: Number of iterations to maintain initial temperature.
    """

    def __init__(
        self,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        decay_schedule: str = "linear",
        warmup: int = 50,
    ):
        """Initialize Boltzmann exploration.

        Args:
            initial_temp: Starting temperature.
            final_temp: Final temperature after full decay.
            decay_schedule: "linear" or "exponential" decay.
            warmup: Warmup iterations before decay starts.
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_schedule = decay_schedule
        self.warmup = warmup

        logger.info(
            "BoltzmannExploration: initial_temp=%.3f, final_temp=%.3f, "
            "schedule=%s, warmup=%d",
            initial_temp, final_temp, decay_schedule, warmup
        )

    def get_exploration_rate(self, iteration: int, total_iterations: int) -> float:
        """Get current temperature as exploration rate.

        Args:
            iteration: Current iteration number.
            total_iterations: Total number of iterations.

        Returns:
            Temperature value for current iteration.
        """
        # Warmup phase: maintain initial temperature
        if iteration < self.warmup:
            return self.initial_temp

        # Calculate progress after warmup
        progress = (iteration - self.warmup) / max(total_iterations - self.warmup, 1)
        progress = min(progress, 1.0)

        if self.decay_schedule == "exponential":
            # Exponential decay: temp = initial * (final/initial)^progress
            ratio = self.final_temp / max(self.initial_temp, 1e-10)
            return self.initial_temp * (ratio ** progress)
        else:
            # Linear decay (default)
            return self.initial_temp - (self.initial_temp - self.final_temp) * progress

    def select_action(
        self,
        q_values: torch.Tensor,
        iteration: int,
        total_iterations: int,
    ) -> int:
        """Select action using Boltzmann (softmax) policy.

        Args:
            q_values: Q-values tensor of shape (1, num_actions).
            iteration: Current iteration number.
            total_iterations: Total number of iterations.

        Returns:
            Selected action index.
        """
        temp = self.get_exploration_rate(iteration, total_iterations)

        # Compute softmax probabilities with temperature
        scaled_q = q_values / max(temp, 1e-10)
        probs = torch.softmax(scaled_q, dim=1)

        # Sample action from distribution
        return torch.multinomial(probs, 1).item()


class UCBExploration(ExplorationStrategy):
    """Upper Confidence Bound exploration strategy.

    Implements UCB action selection that balances exploitation (Q-values)
    with exploration bonus based on visit counts:

        UCB(a) = Q(a) + c * sqrt(log(N) / n(a))

    where c is the exploration coefficient, N is total visits,
    and n(a) is visits to action a.

    Attributes:
        exploration_coef: Exploration coefficient c (default 2.0).
        initial_coef: Starting exploration coefficient.
        final_coef: Final exploration coefficient after decay.
        warmup: Warmup iterations before decay starts.
    """

    def __init__(
        self,
        initial_coef: float = 2.0,
        final_coef: float = 0.5,
        decay_schedule: str = "linear",
        warmup: int = 50,
    ):
        """Initialize UCB exploration.

        Args:
            initial_coef: Starting exploration coefficient.
            final_coef: Final coefficient after full decay.
            decay_schedule: "linear" or "exponential" decay.
            warmup: Warmup iterations before decay starts.
        """
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.decay_schedule = decay_schedule
        self.warmup = warmup

        # Track action visit counts
        self._action_counts: Dict[int, int] = {}
        self._total_visits = 0

        logger.info(
            "UCBExploration: initial_coef=%.3f, final_coef=%.3f, "
            "schedule=%s, warmup=%d",
            initial_coef, final_coef, decay_schedule, warmup
        )

    def get_exploration_rate(self, iteration: int, total_iterations: int) -> float:
        """Get current exploration coefficient.

        Args:
            iteration: Current iteration number.
            total_iterations: Total number of iterations.

        Returns:
            Exploration coefficient for current iteration.
        """
        if iteration < self.warmup:
            return self.initial_coef

        progress = (iteration - self.warmup) / max(total_iterations - self.warmup, 1)
        progress = min(progress, 1.0)

        if self.decay_schedule == "exponential":
            ratio = self.final_coef / max(self.initial_coef, 1e-10)
            return self.initial_coef * (ratio ** progress)
        else:
            return self.initial_coef - (self.initial_coef - self.final_coef) * progress

    def select_action(
        self,
        q_values: torch.Tensor,
        iteration: int,
        total_iterations: int,
    ) -> int:
        """Select action using UCB policy.

        Args:
            q_values: Q-values tensor of shape (1, num_actions).
            iteration: Current iteration number.
            total_iterations: Total number of iterations.

        Returns:
            Selected action index.
        """
        coef = self.get_exploration_rate(iteration, total_iterations)
        num_actions = q_values.shape[1]

        # Compute UCB values
        ucb_values = q_values.clone().squeeze(0)

        for action in range(num_actions):
            count = self._action_counts.get(action, 0)
            if count == 0:
                # Unvisited actions get high bonus
                ucb_values[action] = float('inf')
            else:
                # UCB exploration bonus
                bonus = coef * np.sqrt(np.log(self._total_visits + 1) / count)
                ucb_values[action] += bonus

        # Select action with highest UCB value
        selected_action = ucb_values.argmax().item()

        # Update counts
        self._action_counts[selected_action] = self._action_counts.get(selected_action, 0) + 1
        self._total_visits += 1

        return selected_action

    def reset_counts(self) -> None:
        """Reset action visit counts."""
        self._action_counts.clear()
        self._total_visits = 0


def create_exploration_strategy(config: Dict[str, Any]) -> ExplorationStrategy:
    """Factory function to create exploration strategy from config.

    Args:
        config: Configuration dictionary with exploration settings.
            Expected keys:
            - mode: "epsilon_greedy", "boltzmann", or "ucb"
            - epsilon: dict with initial, final, decay_schedule, warmup_iterations
            - temperature: dict with initial, final, decay_schedule, warmup_iterations
            - ucb: dict with initial_coef, final_coef, decay_schedule, warmup_iterations

    Returns:
        Configured ExplorationStrategy instance.

    Example config:
        {
            "mode": "epsilon_greedy",
            "epsilon": {
                "initial": 0.5,
                "final": 0.05,
                "decay_schedule": "linear",
                "warmup_iterations": 50
            }
        }
    """
    mode = config.get("mode", "epsilon_greedy")

    if mode == "epsilon_greedy":
        eps_config = config.get("epsilon", {})
        return EpsilonGreedyExploration(
            initial_eps=eps_config.get("initial", 0.5),
            final_eps=eps_config.get("final", 0.05),
            decay_schedule=eps_config.get("decay_schedule", "linear"),
            warmup=eps_config.get("warmup_iterations", 50),
        )
    elif mode == "boltzmann":
        temp_config = config.get("temperature", {})
        return BoltzmannExploration(
            initial_temp=temp_config.get("initial", 2.0),
            final_temp=temp_config.get("final", 0.1),
            decay_schedule=temp_config.get("decay_schedule", "linear"),
            warmup=temp_config.get("warmup_iterations", 50),
        )
    elif mode == "ucb":
        ucb_config = config.get("ucb", {})
        return UCBExploration(
            initial_coef=ucb_config.get("initial_coef", 2.0),
            final_coef=ucb_config.get("final_coef", 0.5),
            decay_schedule=ucb_config.get("decay_schedule", "linear"),
            warmup=ucb_config.get("warmup_iterations", 50),
        )
    else:
        logger.warning("Unknown exploration mode '%s', defaulting to epsilon_greedy", mode)
        return EpsilonGreedyExploration()
