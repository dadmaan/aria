"""Incremental reward calculator for RL-compliant causal reward computation.

This module implements an incremental reward calculation system that respects
RL causality principles: rewards at step t depend only on past context, not
future actions.

Reward Components (v2 - Euclidean-only):
- Structure: GHSOM-based quality using Euclidean distance
- Transition: Smoothness between consecutive clusters
- Diversity: Controlled variety with diminishing returns
"""

from typing import Dict, List, Tuple
import numpy as np


class IncrementalRewardCalculator:
    """Calculate rewards incrementally using only past context.

    This calculator ensures that reward at step t only depends on:
    - Current action at step t
    - Past context (steps 0 to t-1)
    - NOT on future actions (steps t+1 onwards)

    This respects the fundamental RL causality principle and enables proper
    credit assignment for learning.

    Reward Components (default weights from config):
    - Structure (weight: 0.35): GHSOM clustering quality via Euclidean distance
    - Transition (weight: 0.30): Smooth progressions between consecutive clusters
    - Diversity (weight: 0.35): Controlled variety with diminishing returns
    """

    def __init__(self, ghsom_manager, config: Dict):
        """Initialize incremental reward calculator.

        Args:
            ghsom_manager: GHSOM manager instance with neuron table and similarity methods
            config: Configuration dictionary with reward_components settings
        """
        self.ghsom_manager = ghsom_manager
        self.context_window = config.get("reward_context_window", 5)

        # Get component-specific configurations
        reward_components = config.get("reward_components", {})

        # Structure component
        structure_cfg = reward_components.get("structure", {})
        self.structure_weight = structure_cfg.get("weight", 0.40)
        self.structure_enabled = structure_cfg.get("enabled", True)

        # Transition component
        transition_cfg = reward_components.get("transition", {})
        self.transition_weight = transition_cfg.get("weight", 0.35)
        self.transition_enabled = transition_cfg.get("enabled", True)
        self.transition_structural_weight = transition_cfg.get("structural_weight", 0.3)
        self.transition_max_distance = transition_cfg.get("max_distance", 3.5)

        # Diversity component
        diversity_cfg = reward_components.get("diversity", {})
        self.diversity_weight = diversity_cfg.get("weight", 0.25)
        self.diversity_enabled = diversity_cfg.get("enabled", True)
        self.optimal_diversity_low = diversity_cfg.get("optimal_ratio_low", 0.62)
        self.optimal_diversity_high = diversity_cfg.get("optimal_ratio_high", 0.75)
        self.repetition_penalty = diversity_cfg.get("repetition_penalty", -0.3)

        # Feature weights for structure sub-component (Euclidean-only)
        self.feature_weights = config.get(
            "feature_weights",
            {
                "distance_weight": 1.0,
                "neighbor_weight": 0.5,
                "grid_weight": 0.25,
            },
        )

    def calculate_step_reward(self, sequence: List[int], current_step: int) -> float:
        """Calculate composite reward for action at current_step.

        Combines three orthogonal components:
        1. Structure: GHSOM-based quality using Euclidean distance
        2. Transition: Smoothness between consecutive clusters
        3. Diversity: Controlled variety with diminishing returns

        All components are normalized to [0, 1] before weighting.

        Args:
            sequence: Full sequence up to current_step+1 (list of cluster IDs)
            current_step: Index of current action (0-based)

        Returns:
            float: Composite reward for current step
        """
        # Input validation
        if not sequence:
            return 0.0

        if current_step < 0:
            return 0.0

        if current_step >= len(sequence):
            return 0.0

        if current_step == 0:
            # First step: baseline reward
            return 0.1

        current_cluster_id = sequence[current_step]
        prev_cluster_id = sequence[current_step - 1]

        # Validate cluster ID
        if current_cluster_id not in self.ghsom_manager.neuron_table:
            return -1.0  # Penalty for invalid cluster

        # --- Component 1: Structure Reward ---
        structure_reward = 0.0
        if self.structure_enabled:
            structure_reward = self._calculate_structure_reward(sequence, current_step)

        # --- Component 2: Transition Reward ---
        transition_reward = 0.0
        if self.transition_enabled:
            transition_reward = self._calculate_transition_reward(
                prev_cluster_id, current_cluster_id
            )

        # --- Component 3: Diversity Reward ---
        diversity_reward = 0.0
        if self.diversity_enabled:
            diversity_reward = self._calculate_controlled_diversity(
                sequence, current_step
            )

        # --- Weighted Combination ---
        # All components already normalized to [0, 1]
        total = (
            self.structure_weight * structure_reward
            + self.transition_weight * transition_reward
            + self.diversity_weight * diversity_reward
        )

        # Sanitize output
        if np.isnan(total) or np.isinf(total):
            return 0.0

        return total

    def _calculate_structure_reward(
        self, sequence: List[int], current_step: int
    ) -> float:
        """Calculate structure reward based on GHSOM similarity to recent context.

        Args:
            sequence: Full sequence up to current_step+1
            current_step: Current step index

        Returns:
            float: Structure reward normalized to [0, 1]
        """
        current_cluster_id = sequence[current_step]

        # Context window (respects causality)
        context_start = max(0, current_step - self.context_window)
        context = sequence[context_start:current_step]

        if not context:
            return 0.5  # Neutral for no context

        try:
            similarity_dict = self.ghsom_manager.get_neurons_similarity_matrix(
                current_cluster_id, sort=True
            )
        except (KeyError, IndexError, AttributeError):
            return 0.0

        # Calculate average reward across context window
        total_reward = 0.0
        comparison_count = 0

        for past_cluster_id in context:
            # Skip self-comparisons to avoid rewarding repetition
            # (self-similarity distance = 0, which would give max reward)
            if past_cluster_id == current_cluster_id:
                continue

            if past_cluster_id in similarity_dict:
                try:
                    neuron_features = similarity_dict[past_cluster_id]
                    reward = self._calculate_neuron_reward(
                        neuron_features, similarity_dict
                    )
                    if not np.isnan(reward) and not np.isinf(reward):
                        total_reward += reward
                        comparison_count += 1
                except (TypeError, ValueError, IndexError):
                    continue

        if comparison_count > 0:
            return total_reward / comparison_count
        return 0.0

    def _calculate_neuron_reward(
        self, neuron_features: Tuple, similarity_dict: Dict
    ) -> float:
        """Calculate reward for a single cluster comparison using Euclidean distance only.

        Args:
            neuron_features: Tuple of (l2_distance, is_neighbor, is_same_col, is_same_row)
            similarity_dict: Dictionary of all similarity features for normalization

        Returns:
            float: Reward for this specific cluster pair comparison (normalized to [0, 1])
        """
        (
            l2_distance,
            is_neighbor,
            is_same_col,
            is_same_row,
        ) = neuron_features

        # Sanitize L2 distance
        if np.isnan(l2_distance) or np.isinf(l2_distance):
            l2_distance = 0.0

        # Find max distance for normalization
        max_l2_distance = max(
            (
                sim[0]
                for sim in similarity_dict.values()
                if not np.isnan(sim[0]) and not np.isinf(sim[0])
            ),
            default=1.0,
        )
        if max_l2_distance == 0:
            max_l2_distance = 1.0

        # Calculate normalized distance reward (closer = higher reward)
        distance_reward = max(0.0, 1.0 - l2_distance / max_l2_distance)

        # Structural bonuses
        neighbor_bonus = 1.0 if is_neighbor else 0.0
        grid_bonus = 0.5 if (is_same_row or is_same_col) else 0.0

        # Weighted combination
        total = (
            distance_reward * self.feature_weights.get("distance_weight", 1.0)
            + neighbor_bonus * self.feature_weights.get("neighbor_weight", 0.5)
            + grid_bonus * self.feature_weights.get("grid_weight", 0.25)
        )

        # Normalize to [0, 1]
        max_possible = (
            self.feature_weights.get("distance_weight", 1.0)
            + self.feature_weights.get("neighbor_weight", 0.5)
            + self.feature_weights.get("grid_weight", 0.25)
        )

        normalized = total / max_possible if max_possible > 0 else 0.0

        if np.isnan(normalized) or np.isinf(normalized):
            return 0.0

        return normalized

    def _calculate_transition_reward(
        self, prev_cluster_id: int, curr_cluster_id: int
    ) -> float:
        """Calculate transition smoothness reward between consecutive clusters.

        Uses GHSOM topology (neighbor relationships) weighted by Euclidean distance.
        Rewards smooth transitions where consecutive clusters are nearby in the
        learned GHSOM space.

        Args:
            prev_cluster_id: Cluster ID at step t-1
            curr_cluster_id: Cluster ID at step t

        Returns:
            float: Transition reward normalized to [0, 1]
        """
        if prev_cluster_id == curr_cluster_id:
            # Same cluster: LOW reward to discourage repetition
            # Changed from 0.7 (bug) to 0.1 - exploration should be rewarded
            return 0.1

        if curr_cluster_id not in self.ghsom_manager.neuron_table:
            return 0.0

        try:
            similarity_dict = self.ghsom_manager.get_neurons_similarity_matrix(
                curr_cluster_id, sort=True
            )
        except (KeyError, IndexError, AttributeError):
            return 0.0

        if prev_cluster_id not in similarity_dict:
            return 0.0

        # 4-element tuple: (l2_distance, is_neighbor, same_col, same_row)
        features = similarity_dict[prev_cluster_id]
        l2_distance = features[0]
        is_neighbor = features[1]
        same_col = features[2]
        same_row = features[3]

        if np.isnan(l2_distance) or np.isinf(l2_distance):
            l2_distance = self.transition_max_distance

        # Structural score (GHSOM topology)
        structural_score = 0.0
        if is_neighbor:
            structural_score = 1.0  # Direct neighbors in GHSOM grid
        elif same_row or same_col:
            structural_score = 0.5  # Same row/column but not adjacent

        # Distance score (Euclidean, normalized)
        distance_score = max(0.0, 1.0 - l2_distance / self.transition_max_distance)

        # Hybrid combination
        hybrid_score = (
            self.transition_structural_weight * structural_score
            + (1.0 - self.transition_structural_weight) * distance_score
        )

        return np.clip(hybrid_score, 0.0, 1.0)

    def _calculate_controlled_diversity(
        self, sequence: List[int], current_step: int
    ) -> float:
        """Calculate diversity reward with diminishing returns.

        Targets optimal diversity range of 62-75% unique clusters for 16-step sequences.
        Penalizes both under-diversity (repetitive) and over-diversity (chaotic).

        Musical rationale:
        - 50-62%: Allows motif repetition, thematic development
        - 62-75%: Optimal balance of variety and coherence
        - 75-87%: Approaching chaos, diminishing returns
        - >87%: Over-diversified, no thematic unity

        Args:
            sequence: Current sequence up to current_step+1
            current_step: Current step index (0-based)

        Returns:
            float: Diversity reward normalized to [0, 1]
        """
        if current_step == 0:
            return 0.5  # Neutral baseline for first step

        current_cluster = sequence[current_step]
        past_clusters = sequence[:current_step]

        # --- Immediate Repetition Penalty ---
        repetition_count = past_clusters.count(current_cluster)
        if repetition_count > 0:
            # Penalize repeated clusters (scales with count, capped at 3)
            repetition_penalty_value = self.repetition_penalty * min(
                repetition_count, 3
            )
        else:
            repetition_penalty_value = 0.0

        # --- Controlled Diversity with Diminishing Returns ---
        unique_count = len(set(sequence[: current_step + 1]))
        total_count = current_step + 1
        diversity_ratio = unique_count / total_count

        # Piecewise reward function
        if diversity_ratio < self.optimal_diversity_low:
            # Below optimal: linear reward encouraging more diversity
            # Maps [0, optimal_low] -> [0.2, 0.6]
            base_reward = 0.2 + (diversity_ratio / self.optimal_diversity_low) * 0.4
        elif diversity_ratio <= self.optimal_diversity_high:
            # In optimal range: maximum reward plateau
            # Maps [optimal_low, optimal_high] -> [0.6, 0.8]
            progress = (diversity_ratio - self.optimal_diversity_low) / (
                self.optimal_diversity_high - self.optimal_diversity_low
            )
            base_reward = 0.6 + progress * 0.2
        else:
            # Above optimal: diminishing returns (logarithmic decay)
            # Maps [optimal_high, 1.0] -> [0.8, 0.3]
            excess = diversity_ratio - self.optimal_diversity_high
            max_excess = 1.0 - self.optimal_diversity_high
            if max_excess > 0:
                decay = np.log1p(excess / max_excess * 5) / np.log1p(5)
            else:
                decay = 0.0
            base_reward = 0.8 - decay * 0.5

        # Combine base reward with repetition penalty
        total_reward = base_reward + repetition_penalty_value

        return np.clip(total_reward, 0.0, 1.0)
