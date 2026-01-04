"""GHSOM-based perceiving agent implementation that complies with the ABC."""

from typing import Dict, List, Optional

import numpy as np

from ..interfaces.agents import PerceivingAgent as BasePerceivingAgent
from .ghsom_agent import PerceivingAgent as LegacyPerceivingAgent
from .incremental_reward_calculator import IncrementalRewardCalculator


class GHSOMPerceivingAgent(BasePerceivingAgent):
    """Wrapper around the existing PerceivingAgent to comply with ABC interface."""

    def __init__(
        self, config: Dict, ghsom_manager=None, features_dataset=None, **kwargs
    ):
        """Initialize GHSOM perceiving agent.

        Args:
            config: Configuration dictionary
            ghsom_manager: GHSOM manager instance
            features_dataset: Features dataset
            **kwargs: Additional arguments
        """
        self.config = config

        # Use provided instances or get from kwargs
        self.ghsom_manager = (
            ghsom_manager if ghsom_manager is not None else kwargs.get("ghsom_manager")
        )
        self.features_dataset = (
            features_dataset
            if features_dataset is not None
            else kwargs.get("features_dataset")
        )

        if self.ghsom_manager is None or self.features_dataset is None:
            # Create instances if not provided - this might need adjustment
            # based on how these are typically constructed
            if self.ghsom_manager is None:
                # For now, raise an error - proper initialization needs to be handled
                # by the calling code that has the required parameters
                raise ValueError("ghsom_manager is required for GHSOMPerceivingAgent")
            if self.features_dataset is None:
                # This would need actual dataset loading logic
                self.features_dataset = None

        # Create the legacy agent (Phase 2: pass config for feature weights)
        self.legacy_agent = LegacyPerceivingAgent(
            self.ghsom_manager, self.features_dataset, config=self.config
        )

        # Create incremental reward calculator (Phase 2)
        self.incremental_calculator = IncrementalRewardCalculator(
            self.ghsom_manager, config
        )
        self.use_incremental = config.get("use_incremental_rewards", True)

    def evaluate_sequence(self, sequence: np.ndarray) -> Dict[str, float]:
        """Evaluate a musical sequence and return reward components.

        Returns three orthogonal, normalized reward components:
        - structure: GHSOM-based quality (Euclidean distance)
        - transition: Smoothness between consecutive clusters
        - diversity: Controlled variety with diminishing returns

        Args:
            sequence: Array of cluster IDs representing a musical sequence

        Returns:
            Dictionary with reward components, each normalized to [0, 1]
        """
        # Convert numpy array to list for compatibility
        sequence_list = (
            sequence.tolist() if isinstance(sequence, np.ndarray) else sequence
        )

        if not sequence_list:
            return {"structure": 0.0, "transition": 0.0, "diversity": 0.0}

        if self.use_incremental:
            # Incremental calculation: accumulate step rewards per component
            structure_rewards = []
            transition_rewards = []
            diversity_rewards = []

            for step in range(len(sequence_list)):
                if step == 0:
                    # First step: baseline values
                    structure_rewards.append(0.5)
                    transition_rewards.append(0.5)
                    diversity_rewards.append(0.5)
                else:
                    # Get component rewards from incremental calculator
                    s_reward = self.incremental_calculator._calculate_structure_reward(
                        sequence_list, step
                    )
                    t_reward = self.incremental_calculator._calculate_transition_reward(
                        sequence_list[step - 1], sequence_list[step]
                    )
                    d_reward = (
                        self.incremental_calculator._calculate_controlled_diversity(
                            sequence_list, step
                        )
                    )

                    structure_rewards.append(s_reward)
                    transition_rewards.append(t_reward)
                    diversity_rewards.append(d_reward)

            # Return mean of each component (all in [0, 1])
            return {
                "structure": float(np.mean(structure_rewards)),
                "transition": float(np.mean(transition_rewards)),
                "diversity": float(np.mean(diversity_rewards)),
            }
        else:
            # Legacy fallback (deprecated)
            total_structure, _ = self.legacy_agent.evaluate_sequence(sequence_list)
            if np.isnan(total_structure) or np.isinf(total_structure):
                total_structure = 0.0

            return {
                "structure": total_structure,
                "transition": 0.0,
                "diversity": 0.0,
            }

    def get_unique_cluster_ids_list(self) -> List[int]:
        """Get the list of all available cluster IDs.

        Returns:
            List of cluster IDs that can be used as actions
        """
        cluster_array = self.legacy_agent.get_ghsom_clusters_list()
        return (
            cluster_array.tolist()
            if hasattr(cluster_array, "tolist")
            else list(cluster_array)
        )

    def get_action_space(self) -> int:
        """Get the size of the discrete action space.

        Returns:
            Number of available actions (cluster IDs)
        """
        cluster_ids = self.get_unique_cluster_ids_list()
        return len(cluster_ids)

    def seed(self, random_state: Optional[int]) -> None:
        """Seed random number generators for reproducibility.

        Args:
            random_state: Random seed value
        """
        if random_state is not None:
            np.random.seed(random_state)
            # If the legacy agent has any random components, seed them too
            import random

            random.seed(random_state)

    def calculate_structure_reward(self, sequence, features_weight):
        """Delegate to legacy agent for backward compatibility."""
        return self.legacy_agent.calculate_structure_reward(sequence, features_weight)

    def get_best_matching_unit(self, input_data):
        """Delegate to legacy agent for backward compatibility."""
        return self.legacy_agent.get_best_matching_unit(input_data)

    def get_sample_midi_from_node(self, node_id, num_samples=1):
        """Delegate to legacy agent for backward compatibility."""
        return self.legacy_agent.get_sample_midi_from_node(node_id, num_samples)
