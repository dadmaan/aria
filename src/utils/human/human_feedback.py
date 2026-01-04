"""
Human feedback interface for reward shaping.

This module provides configurable human feedback collection with timeout support,
CLI integration for testing, and fallback to default values in non-interactive mode.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from typing import Optional, Union

import numpy as np

from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)


class HumanFeedbackCollector:
    """
    Configurable human feedback collector with timeout and non-interactive mode support.
    
    Supports:
    - Interactive input with configurable timeout
    - Non-interactive mode (returns default values)
    - CLI-provided feedback for testing
    - Robust error handling and fallbacks
    """
    
    def __init__(
        self,
        timeout: int = 30,
        non_interactive_mode: bool = False,
        default_feedback: float = 0.0,
        cli_feedback: Optional[float] = None
    ):
        """
        Initialize human feedback collector.
        
        Args:
            timeout: Timeout in seconds for human input
            non_interactive_mode: If True, always return default feedback
            default_feedback: Default feedback value when no input received
            cli_feedback: Pre-provided feedback value for testing
        """
        self.timeout = timeout
        self.non_interactive_mode = non_interactive_mode
        self.default_feedback = default_feedback
        self.cli_feedback = cli_feedback
        
        # Validate feedback values are in reasonable range
        self._validate_feedback_value(default_feedback)
        if cli_feedback is not None:
            self._validate_feedback_value(cli_feedback)
    
    def collect_feedback(
        self,
        prompt: str = "Please provide your feedback as a reward (e.g., -1 for bad, 0 for neutral, 1 for good): ",
        alpha: float = 1.0
    ) -> float:
        """
        Collect human feedback with timeout and fallback handling.
        
        Args:
            prompt: Prompt message to display to user
            alpha: Scaling factor for feedback
            
        Returns:
            Human feedback value scaled by alpha
        """
        try:
            # Non-interactive mode: return default immediately
            if self.non_interactive_mode:
                logger.debug("Non-interactive mode: returning default feedback")
                return self.default_feedback * alpha
            
            # CLI-provided feedback: return pre-configured value
            if self.cli_feedback is not None:
                logger.debug(f"Using CLI-provided feedback: {self.cli_feedback}")
                return self.cli_feedback * alpha
            
            # Interactive mode: collect user input with timeout
            feedback = self._collect_interactive_feedback(prompt)
            return feedback * alpha
            
        except Exception as e:
            logger.error(f"Failed to collect human feedback: {e}. Using default.")
            return self.default_feedback * alpha
    
    def _collect_interactive_feedback(self, prompt: str) -> float:
        """
        Collect interactive feedback from user with timeout.
        
        Args:
            prompt: Prompt message to display
            
        Returns:
            User feedback value or default if timeout/error
        """
        def input_thread(prompt: str, result_queue: queue.Queue) -> None:
            """Thread function to collect user input."""
            try:
                # Clear any existing input
                while True:
                    try:
                        # Check if there's any input waiting
                        if sys.stdin in []:  # Would need select on Unix
                            break
                    except (OSError, IOError):
                        break
                
                # Prompt user for input
                print(prompt, end='', flush=True)
                user_input = None
                start_time = time.time()
                
                while True:
                    time_elapsed = time.time() - start_time
                    if time_elapsed > self.timeout:
                        break
                    
                    try:
                        # Use readline with timeout simulation
                        user_input = sys.stdin.readline().strip()
                        if user_input:
                            break
                    except (OSError, IOError):
                        pass
                    
                    time.sleep(0.1)  # Brief sleep to avoid busy waiting
                
                result_queue.put(user_input)
                
            except Exception as e:
                logger.error(f"Input thread error: {e}")
                result_queue.put(None)
        
        # Create queue for thread communication
        input_queue: queue.Queue = queue.Queue()
        
        # Start input collection thread
        thread = threading.Thread(
            target=input_thread,
            args=(prompt, input_queue),
            daemon=True
        )
        thread.start()
        
        try:
            # Wait for input with timeout
            user_input = input_queue.get(block=True, timeout=self.timeout)
            
            if user_input is None:
                raise ValueError("No valid input received")
            
            # Convert input to float
            feedback = float(user_input)
            self._validate_feedback_value(feedback)
            
            return feedback
            
        except queue.Empty:
            logger.warning(f"\nNo input received within {self.timeout} seconds. Using default feedback of {self.default_feedback}")
            return self.default_feedback

        except (ValueError, TypeError):
            logger.warning(f"Invalid input. Using default feedback of {self.default_feedback}")
            return self.default_feedback
    
    def _validate_feedback_value(self, value: float) -> None:
        """
        Validate that feedback value is reasonable.
        
        Args:
            value: Feedback value to validate
            
        Raises:
            ValueError: If value is not reasonable
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Feedback must be numeric, got {type(value)}")
        
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Feedback must be finite, got {value}")
        
        # Allow reasonable range
        if not -10.0 <= value <= 10.0:
            logger.warning(f"Feedback value {value} is outside typical range [-10, 10]")
    
    def simulate_random_feedback(self, seed: Optional[int] = None) -> float:
        """
        Generate simulated human feedback for testing.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Simulated feedback value
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate feedback from a distribution that favors neutral/positive
        # This simulates a human who gives mostly reasonable feedback
        feedback_options = [-1.0, -0.5, 0.0, 0.5, 1.0]
        weights = [0.1, 0.15, 0.3, 0.25, 0.2]  # Slight positive bias
        
        feedback = np.random.choice(feedback_options, p=weights)
        logger.debug(f"Simulated human feedback: {feedback}")
        
        return float(feedback)


def create_human_feedback_collector(config: dict) -> HumanFeedbackCollector:
    """
    Factory function to create human feedback collector from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured HumanFeedbackCollector instance
    """
    return HumanFeedbackCollector(
        timeout=config.get("interaction_timeout", 30),
        non_interactive_mode=config.get("non_interactive_mode", False),
        default_feedback=0.0,  # Always default to neutral
        cli_feedback=config.get("cli_human_feedback", None)
    )