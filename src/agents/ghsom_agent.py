"""GHSOM (Growing Hierarchical Self-Organizing Map) agent implementation for music generation.

This module contains the PerceivingAgent class that uses GHSOM for clustering and
reward calculation based on musical sequence structures. It analyzes sequences of
cluster IDs and computes rewards based on feature similarities and structural
properties of the generated music.
"""

import random

import numpy as np
import pandas as pd

from src.utils.logging.logging_manager import get_logger

logger = get_logger("ghsom_agent")


class PerceivingAgent:
    def __init__(self, ghsom_manager, features_dataset, config=None):
        """
        Initializes the Perceiving Agent with a specified number of clusters.

        Args:
            ghsom_manager: GHSOM manager instance
            features_dataset: Features dataset
            config: Optional configuration dictionary (Phase 2 addition)
        """

        self.ghsom_manager = ghsom_manager
        # self.replay_buffer_analyzer = replay_buffer_analyzer
        self.features_dataset = features_dataset
        self.config = config or {}

        self.neuron_rewards = {}
        self.suggestion_matrix = {}

    def calculate_structure_reward(self, sequence, features_weight):
        """
        Calculate the reward for a given sequence of cluster IDs based on the similarity features and weight factors.

        Parameters:
            sequence (list): The sequence of cluster IDs chosen by the agent.
            features_weight (dict): A dictionary of weight factors for each feature.

        Returns:
            Tuple: The computed reward for the given sequence.
                    - A total reward for the sequence.
                    - A list of rewards for each action (cluster) in the sequence.
        """

        if not sequence:
            return -20.0, []  # No reward if the sequence is empty

        total_reward = 0.0
        each_action_rewards = []

        for i, id in enumerate(sequence):
            # print(f"Calculating reward for cluster {i} with ID {id}")
            if id not in self.ghsom_manager.neuron_table:
                logger.warning(f"Neuron ID {id} not found in neuron table.")
                each_action_rewards.append(-10.0)
                continue

            neuron = self.ghsom_manager.neuron_table[id][1]
            similarity_dict = self.ghsom_manager.get_neurons_similarity_matrix(
                neuron, sort=True
            )
            max_l2_distance = max(
                (sim[0] for sim in similarity_dict.values()), default=1.0
            )
            # Ensure max_l2_distance is never zero to avoid division by zero
            if max_l2_distance == 0:
                max_l2_distance = 1.0

            for neuron_id in sequence[i + 1 :]:
                if neuron_id in similarity_dict:
                    neuron_reward = self._calculate_neuron_reward(
                        similarity_dict[neuron_id], max_l2_distance, features_weight
                    )
                    total_reward += neuron_reward
                    each_action_rewards.append(neuron_reward)

                    # Update cumulative reward for the neuron
                    if neuron_id not in self.neuron_rewards:
                        self.neuron_rewards[neuron_id] = 0
                    self.neuron_rewards[neuron_id] += neuron_reward

        return total_reward, each_action_rewards

    def _calculate_neuron_reward(
        self, neuron_features, max_l2_distance, features_weight
    ):
        """
        Calculate the reward for a single neuron based on its features and the given weights.

        Parameters:
            neuron_features (tuple): Features of the neuron (l2_distance, is_neighbor, is_same_row, is_same_column).
                Note: cosine_similarity has been removed (v2 reward system uses Euclidean only).
            max_l2_distance (float): The maximum L2 distance used for normalization.
            features_weight (dict): Weights for each feature.

        Returns:
            float: The calculated reward for the neuron.
        """
        # Unpack 4-element tuple (v2: Euclidean-only, no cosine similarity)
        (
            l2_distance,
            is_neighbor,
            is_same_row,
            is_same_column,
        ) = neuron_features

        # Sanitize inputs to handle NaN/inf values
        if np.isnan(l2_distance) or np.isinf(l2_distance):
            l2_distance = 0.0

        distance_reward = (
            max(0, 1 - l2_distance / max_l2_distance)
            * features_weight["distance_weight"]
        )
        # Note: similarity_weight still supported in config for backward compat,
        # but cosine similarity removed - weight is effectively ignored
        neighbor_reward = (1.0 if is_neighbor else 0.0) * features_weight[
            "neighbor_weight"
        ]
        row_reward = (1.0 if is_same_row else 0.0) * features_weight["row_weight"]
        column_reward = (1.0 if is_same_column else 0.0) * features_weight[
            "column_weight"
        ]

        total_reward = distance_reward + neighbor_reward + row_reward + column_reward

        # Final sanitization to ensure no NaN in output
        if np.isnan(total_reward) or np.isinf(total_reward):
            return 0.0

        return total_reward

    def get_neuron_rewards(self):
        """
        Retrieve the cumulative rewards for each neuron.

        Returns:
            dict: A dictionary with neuron IDs as keys and their cumulative rewards as values.
        """
        return self.neuron_rewards

    def get_ghsom_clusters_list(self):
        return np.array(self.ghsom_manager.get_unique_cluster_ids_list())

    def get_best_matching_unit(self, input):
        _, neuron, location = self.ghsom_manager.find_best_matching_unit(input)
        node_id = self.ghsom_manager.encode_ghsom_node(location)
        return neuron, node_id

    def get_cluster_id_with_feature_pairs(self, feature):
        features = self.features_dataset[feature]
        data = self.ghsom_manager.get_cluster_ids()
        data = pd.concat([data, features], axis=1)
        return data

    def evaluate_sequence(self, sequence):
        # Read feature weights from config (Phase 2: Issue #7 fix)
        # Merge with defaults so missing keys get default values
        default_weights = {
            "distance_weight": 1.0,
            "similarity_weight": 1.0,
            "neighbor_weight": 0.5,
            "row_weight": 0.5,
            "column_weight": 0.5,
        }
        config_weights = self.config.get("feature_weights", {})
        rewards_weights = {**default_weights, **config_weights}
        total_reward, actions_rewards = self.calculate_structure_reward(
            sequence, rewards_weights
        )
        return (total_reward, actions_rewards)

    def get_sample_midi_from_node(self, node_id, num_samples=1):
        """
        Retrieves MIDI file paths associated with samples from a specific neuron in a GHSOM model.

        This function finds the MIDI file paths for samples that are associated with a given neuron,
        identified by `node_id`. The neuron's samples are matched against a training dataset to extract
        the corresponding MIDI paths.

        Parameters:
        - node_id (int): The identifier for the neuron in the GHSOM model.
        - num_samples (int, optional): The number of random samples to return. If not specified, all matched samples are returned.

        Returns:
        - list of tuples: Each tuple contains (index, midi_path) where 'index' is the DataFrame index of the sample,
                        and 'midi_path' is the path to the MIDI file.

        Raises:
        - ValueError: If the specified node_id is not found in the neuron_table.
        """
        if node_id not in self.ghsom_manager.neuron_table:
            raise ValueError(f"Node ID {node_id} not found in the neuron table.")

        # Extract the list of sample indices from the neuron's input dataset
        neuron_samples = self.ghsom_manager.neuron_table[node_id][1].input_dataset

        matched_rows = []
        for ns in neuron_samples:
            # Query the DataFrame for matching MIDI paths
            sample_idx = self.ghsom_manager.train_data[
                (self.ghsom_manager.train_data["dim1"] == ns[0])
                & (self.ghsom_manager.train_data["dim2"] == ns[1])
            ].index
            midi_path = self.features_dataset.iloc[sample_idx]["midi_path"]
            if not midi_path.empty:
                matched_rows.append((sample_idx[0], midi_path.iloc[0]))

        # Handle the case where a specific number of samples is requested
        if num_samples is not None:
            if num_samples > len(matched_rows):
                logger.warning(
                    f"Only {len(matched_rows)} samples can be collected, less than the requested {num_samples}."
                )
            matched_rows = random.sample(
                matched_rows, min(num_samples, len(matched_rows))
            )

        return matched_rows

    def recommend_suitable_action(self, action, step):
        action = None  # top actions, Use ReplayBufferAnalyzer
        # self.update_suggestion_matrix(action, step)
        return action

    def update_suggestion_matrix(self, action, step):
        # keep track of suggestions and their outcomes to rank them
        # so actions with better outcome would be prioritize
        return None
