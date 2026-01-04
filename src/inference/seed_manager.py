"""Centralized seed management for reproducible simulations.

This module provides the SeedManager class for coordinating random number
generation across all simulation components to ensure full reproducibility.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SeedManager:
    """Manages random seeds across all simulation components.

    This class ensures that all random number generators (Python, NumPy, PyTorch,
    environment, etc.) are properly seeded for reproducible simulations.

    Attributes:
        master_seed: The master seed for this simulation.
        component_seeds: Seeds derived for each component.
    """

    def __init__(self, master_seed: int):
        """Initialize seed manager with master seed.

        Args:
            master_seed: Master seed for reproducibility. All component
                seeds will be deterministically derived from this.

        Raises:
            ValueError: If master_seed is negative.
        """
        if master_seed < 0:
            raise ValueError(f"master_seed must be non-negative, got {master_seed}")

        self.master_seed = master_seed
        self.component_seeds = self._derive_component_seeds()
        logger.info(f"SeedManager initialized with master_seed={master_seed}")

    def _derive_component_seeds(self) -> Dict[str, int]:
        """Derive deterministic seeds for each component.

        Uses the master seed to create a deterministic sequence of seeds
        for different components.

        Returns:
            Dictionary mapping component names to their seeds.
        """
        # Use master seed to create deterministic component seeds
        rng = np.random.RandomState(self.master_seed)

        seeds = {
            "python": int(rng.randint(0, 2**31 - 1)),
            "numpy": int(rng.randint(0, 2**31 - 1)),
            "torch": int(rng.randint(0, 2**31 - 1)),
            "environment": int(rng.randint(0, 2**31 - 1)),
            "feedback": int(rng.randint(0, 2**31 - 1)),
            "trainer": int(rng.randint(0, 2**31 - 1)),
            "network": int(rng.randint(0, 2**31 - 1)),
        }

        logger.debug(f"Derived component seeds from master={self.master_seed}")
        return seeds

    def seed_all(self) -> None:
        """Seed all random number generators.

        This method should be called at the start of each simulation run
        to ensure reproducibility.
        """
        # Seed Python's built-in random
        random.seed(self.component_seeds["python"])
        logger.debug(f"Seeded Python random: {self.component_seeds['python']}")

        # Seed NumPy
        np.random.seed(self.component_seeds["numpy"])
        logger.debug(f"Seeded NumPy: {self.component_seeds['numpy']}")

        # Seed PyTorch
        torch.manual_seed(self.component_seeds["torch"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.component_seeds["torch"])
        logger.debug(f"Seeded PyTorch: {self.component_seeds['torch']}")

        # Set deterministic mode for PyTorch (for better reproducibility)
        # Note: This may impact performance
        # Use warn_only=True to allow operations without deterministic implementations
        # to proceed with a warning instead of throwing an error
        try:
            # Try with warn_only=True first (PyTorch >= 1.11)
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.debug("Enabled PyTorch deterministic mode (warn_only=True)")
        except TypeError:
            # Fallback for older PyTorch versions without warn_only parameter
            try:
                torch.use_deterministic_algorithms(True)
                logger.debug("Enabled PyTorch deterministic mode")
            except Exception as e:
                logger.warning(
                    f"Could not enable PyTorch deterministic mode: {e}. "
                    "Results may not be fully reproducible."
                )
        except Exception as e:
            logger.warning(
                f"Could not enable PyTorch deterministic mode: {e}. "
                "Results may not be fully reproducible."
            )

        # Also set CUDNN to deterministic mode for additional reproducibility
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("Set CUDNN to deterministic mode")

    def get_environment_seed(self) -> int:
        """Get seed for environment initialization.

        Returns:
            Environment seed value.
        """
        return self.component_seeds["environment"]

    def get_feedback_seed(self) -> int:
        """Get seed for feedback simulator.

        Returns:
            Feedback simulator seed value.
        """
        return self.component_seeds["feedback"]

    def get_trainer_seed(self) -> int:
        """Get seed for trainer/network operations.

        Returns:
            Trainer seed value.
        """
        return self.component_seeds["trainer"]

    def get_network_seed(self) -> int:
        """Get seed for network-specific operations.

        Returns:
            Network seed value.
        """
        return self.component_seeds["network"]

    def get_component_seed(self, component: str) -> Optional[int]:
        """Get seed for a specific component.

        Args:
            component: Component name (e.g., 'python', 'numpy', 'environment').

        Returns:
            Seed value for the component, or None if component not found.
        """
        return self.component_seeds.get(component)

    def to_dict(self) -> Dict[str, Any]:
        """Export seed configuration to dictionary.

        Returns:
            Dictionary with master seed and all component seeds.
        """
        return {
            "master_seed": self.master_seed,
            "component_seeds": self.component_seeds.copy(),
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> SeedManager:
        """Create SeedManager from configuration dictionary.

        Args:
            config: Dictionary with 'master_seed' key.

        Returns:
            SeedManager instance.

        Raises:
            ValueError: If master_seed not in config.
        """
        if "master_seed" not in config:
            raise ValueError("config must contain 'master_seed' key")

        return cls(master_seed=config["master_seed"])

    def reset_with_new_seed(self, new_seed: int) -> None:
        """Reset seed manager with a new master seed.

        Args:
            new_seed: New master seed value.
        """
        self.master_seed = new_seed
        self.component_seeds = self._derive_component_seeds()
        self.seed_all()
        logger.info(f"SeedManager reset with new master_seed={new_seed}")

    def __repr__(self) -> str:
        """String representation of SeedManager."""
        return f"SeedManager(master_seed={self.master_seed})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another SeedManager."""
        if not isinstance(other, SeedManager):
            return NotImplemented
        return self.master_seed == other.master_seed
