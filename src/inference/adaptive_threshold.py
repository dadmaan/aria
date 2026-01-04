"""Adaptive threshold manager for HIL feedback system.

This module provides dynamic feedback threshold adjustment that responds
to learning progress during HIL simulation. The threshold is adjusted
based on recent performance:

- Lowers threshold if no improvement detected (encourage more adaptation)
- Raises threshold if consistent improvement (tighten standards)

This creates a feedback loop that helps maintain appropriate learning
pressure throughout the simulation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholdManager:
    """Manages adaptive feedback threshold during HIL simulation.

    Adjusts threshold based on learning progress:
    - Lowers threshold if no improvement detected (encourage more adaptation)
    - Raises threshold if consistent improvement (tighten standards)

    The threshold affects when adaptation (Q-value penalties, reward shaping)
    is triggered based on feedback ratings.

    Attributes:
        initial_threshold: Starting feedback threshold.
        min_threshold: Minimum allowed threshold.
        max_threshold: Maximum allowed threshold.
        adjustment_rate: Amount to adjust threshold per step.
        no_improvement_patience: Iterations to wait before lowering threshold.
        current_threshold: Current active threshold value.
    """

    initial_threshold: float = 3.0
    min_threshold: float = 2.5
    max_threshold: float = 3.5
    adjustment_rate: float = 0.03
    no_improvement_patience: int = 100

    # Internal state (initialized in __post_init__)
    current_threshold: float = field(init=False)
    _feedback_history: List[float] = field(default_factory=list)
    _best_avg_feedback: float = field(default=0.0)
    _iterations_without_improvement: int = field(default=0)
    _adjustment_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Initialize internal state after dataclass initialization."""
        self.current_threshold = self.initial_threshold
        self._best_avg_feedback = 0.0
        self._iterations_without_improvement = 0

        logger.info(
            "AdaptiveThresholdManager initialized: initial=%.3f, "
            "min=%.3f, max=%.3f, rate=%.3f, patience=%d",
            self.initial_threshold,
            self.min_threshold,
            self.max_threshold,
            self.adjustment_rate,
            self.no_improvement_patience,
        )

    def update(self, feedback_rating: float, iteration: int) -> bool:
        """Update threshold based on feedback.

        Analyzes recent feedback history and adjusts threshold:
        - If recent average improved: raise threshold (tighten standards)
        - If no improvement for patience iterations: lower threshold

        Args:
            feedback_rating: Current feedback rating (1-5 scale).
            iteration: Current iteration number.

        Returns:
            True if threshold was adjusted.
        """
        self._feedback_history.append(feedback_rating)

        # Need enough history for meaningful comparison
        if len(self._feedback_history) < 50:
            return False

        # Calculate recent average (last 50 iterations)
        recent_avg = sum(self._feedback_history[-50:]) / 50

        # Check for meaningful improvement (at least 0.05 improvement)
        if recent_avg > self._best_avg_feedback + 0.05:
            self._best_avg_feedback = recent_avg
            self._iterations_without_improvement = 0

            # Raise threshold if consistent improvement (tighten standards)
            if self.current_threshold < self.max_threshold:
                old_threshold = self.current_threshold
                self.current_threshold = min(
                    self.current_threshold + self.adjustment_rate,
                    self.max_threshold,
                )
                self._log_adjustment(
                    iteration, old_threshold, "improvement", recent_avg
                )
                return True
        else:
            self._iterations_without_improvement += 1

            # Lower threshold if no improvement for patience iterations
            if self._iterations_without_improvement >= self.no_improvement_patience:
                if self.current_threshold > self.min_threshold:
                    old_threshold = self.current_threshold
                    self.current_threshold = max(
                        self.current_threshold - self.adjustment_rate,
                        self.min_threshold,
                    )
                    self._iterations_without_improvement = 0
                    self._log_adjustment(
                        iteration, old_threshold, "no_improvement", recent_avg
                    )
                    return True

        return False

    def _log_adjustment(
        self,
        iteration: int,
        old_threshold: float,
        reason: str,
        avg_feedback: float,
    ) -> None:
        """Log threshold adjustment.

        Args:
            iteration: Current iteration number.
            old_threshold: Previous threshold value.
            reason: Reason for adjustment ("improvement" or "no_improvement").
            avg_feedback: Current average feedback.
        """
        adjustment = {
            "iteration": iteration,
            "old_threshold": old_threshold,
            "new_threshold": self.current_threshold,
            "reason": reason,
            "avg_feedback": avg_feedback,
        }
        self._adjustment_history.append(adjustment)

        logger.info(
            "Adaptive threshold: %.3f -> %.3f (reason: %s, avg_feedback: %.3f)",
            old_threshold,
            self.current_threshold,
            reason,
            avg_feedback,
        )

    def get_threshold(self) -> float:
        """Get current threshold value.

        Returns:
            Current active feedback threshold.
        """
        return self.current_threshold

    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """Get history of threshold adjustments.

        Returns:
            List of adjustment records with iteration, old/new threshold,
            reason, and average feedback at time of adjustment.
        """
        return self._adjustment_history.copy()

    def get_state(self) -> Dict[str, Any]:
        """Get complete manager state for serialization.

        Returns:
            Dictionary containing all state information.
        """
        return {
            "initial_threshold": self.initial_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "adjustment_rate": self.adjustment_rate,
            "no_improvement_patience": self.no_improvement_patience,
            "current_threshold": self.current_threshold,
            "best_avg_feedback": self._best_avg_feedback,
            "iterations_without_improvement": self._iterations_without_improvement,
            "num_adjustments": len(self._adjustment_history),
            "feedback_history_length": len(self._feedback_history),
        }

    def reset(self) -> None:
        """Reset manager state to initial values."""
        self.current_threshold = self.initial_threshold
        self._feedback_history.clear()
        self._best_avg_feedback = 0.0
        self._iterations_without_improvement = 0
        self._adjustment_history.clear()

        logger.info("AdaptiveThresholdManager reset to initial state")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AdaptiveThresholdManager":
        """Create manager from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - initial_threshold: Starting threshold (default 3.0)
                - min_threshold: Minimum threshold (default 2.5)
                - max_threshold: Maximum threshold (default 3.5)
                - adjustment_rate: Adjustment step size (default 0.03)
                - no_improvement_patience: Patience iterations (default 100)

        Returns:
            Configured AdaptiveThresholdManager instance.
        """
        return cls(
            initial_threshold=config.get("initial_threshold", 3.0),
            min_threshold=config.get("min_threshold", 2.5),
            max_threshold=config.get("max_threshold", 3.5),
            adjustment_rate=config.get("adjustment_rate", 0.03),
            no_improvement_patience=config.get("no_improvement_patience", 100),
        )
