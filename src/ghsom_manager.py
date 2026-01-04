"""
GHSOM Manager Module

This module provides the GHSOMManager class for managing Growing Hierarchical
Self-Organizing Map (GHSOM) models. It handles model loading, hierarchy parsing,
neuron table creation, and various analysis operations on GHSOM structures.

Key Features:
- Hierarchical GHSOM model parsing and representation
- Neuron lookup tables and cluster ID mapping
- Similarity calculations between neurons
- Statistical analysis of GHSOM hierarchies
- Integration with feature datasets and metadata

Classes:
    GHSOMManager: Main class for GHSOM model management and operations

Note:
    GHSOMNode and parsing utilities are now imported from ghsom_toolkits.core
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ghsom_toolkits.core import (
    GHSOMNode,
    create_lookup_table,
    decode_ghsom_node,
    encode_ghsom_node,
    get_ghsom_statistics,
    parse_ghsom_hierarchy,
)

from src.utils.features.feature_loader import FeatureType, load_feature_dataset
from src.utils.logging.logging_manager import get_logger


class GHSOMManager:
    def __init__(
        self,
        ghsom_model_path,
        ghsom_train_data,
        *,
        metadata: Optional[pd.DataFrame] = None,
        feature_type: FeatureType = "tsne",
        artifact_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger("ghsom_manager")
        self.ghsom_model = self.load_model(ghsom_model_path)
        self.train_data = ghsom_train_data
        self.metadata = metadata
        self.feature_type: FeatureType = feature_type
        self.feature_columns = list(getattr(ghsom_train_data, "columns", []))
        self.artifact_metadata: Dict[str, Any] = artifact_metadata or {}

        self.root_node = parse_ghsom_hierarchy(str(self.ghsom_model))
        self.lookup_table = create_lookup_table(self.root_node, short_id=True)
        self.neuron_table = self.create_neuron_table()
        self.stats = get_ghsom_statistics(self.root_node)

    @classmethod
    def from_artifact(
        cls,
        ghsom_model_path: Path,
        feature_artifact: Path,
        *,
        feature_type: FeatureType = "tsne",
        metadata_columns: Optional[Iterable[str]] = None,
    ) -> "GHSOMManager":
        """Convenience constructor that loads feature artifacts via the loader pipeline."""

        features_df, metadata_df, artifact_metadata = load_feature_dataset(
            feature_artifact,
            feature_type,
            metadata_columns=metadata_columns,
        )
        return cls(
            ghsom_model_path,
            features_df,
            metadata=metadata_df,
            feature_type=feature_type,
            artifact_metadata=artifact_metadata,
        )

    def __repr__(self):
        return f"Total GHSOM Nodes: {self.stats['total_nodes']} -- Levels -> Nodes {self.stats['levels']} -- Maximum Chilren {self.stats['max_children']} -- Dataset Size {self.stats['max_input_dataset_size']}"

    def create_neuron_table(self):
        """
        Creates a table mapping GHSOM cluster IDs to their corresponding neuron representations and neurons.
        This function traverses the GHSOM hierarchy starting from the root node to find the best matching unit
        for each sample in the train_data. It then creates a neuron table with these mappings.

        Returns:
        - dict: A dictionary where each key is a GHSOM cluster ID (node ID) and each value is a tuple containing
        the neuron representation and the neuron object. This table provides a direct lookup from a cluster ID
        to its corresponding neuron details.

        """
        data = self.train_data.copy()
        neuron_table = {}

        for value in data.values:
            _, neuron, neuron_representation = self.find_best_matching_unit(value)
            node_id = encode_ghsom_node(self.lookup_table, neuron_representation)
            neuron_table[node_id] = (neuron_representation, neuron)

        return neuron_table

    def get_node_relative_path_by_id(self, node, print_output=True):
        """
        Retrieves the relative path in the GHSOM hierarchy from root node to the given node.
        The given node can be either a neuron within GHSOM hierarchy or a node ID in lookup table.

        Parameters:
        - node (GHSOM neuron or node_id (int)) :The root node of the GHSOM tree or a node ID in lookup table.
        - print_output (bool): Prints the path.

        Returns:
        - str: A string representation of the path from the root node to the target node, or None if not found.

        Note:
            This method wraps the ghsom_toolkits.core.get_node_relative_path_by_id function
            but uses the instance logger instead of the module logger.
        """
        from ghsom_toolkits.core.parsing import (
            get_node_relative_path_by_id as core_get_path,
        )

        # Use the core function but handle logging locally if needed
        path = core_get_path(self.lookup_table, node, print_output=False)

        if path and print_output:
            path_str = "\n".join(path)
            self.logger.info(path_str)

        return path

    def find_best_matching_unit(self, data_point):
        """
        Finds the best matching unit (BMU) for a given data point in a GHSOM and updates the BMU's level information.

        This function traverses down the GHSOM hierarchy to find the winning neuron (BMU) for the data point.
        It also updates the BMU's string representation to reflect the correct level in the hierarchy where the BMU was found.

        Parameters:
        - ghsom (MapNode): The root neuron of the GHSOM.
        - data_point (list): The data point to find the BMU for.

        Returns:
        - tuple: A tuple containing the parent BMU, the BMU neuron, and the updated string representation of the BMU.
        """
        _neuron = self.ghsom_model
        # Get the parent node
        parent = _neuron.child_map.winner_neuron(data_point)[0][0]

        # Traverse down the GHSOM hierarchy to find the winning neuron for the data_point
        # Count number of times traversing the GHSOM hierarchy starting at root (zero unit)
        level = 0
        while _neuron.child_map is not None:
            _gsom = _neuron.child_map
            _neuron = _gsom.winner_neuron(data_point)[0][0]
            level += 1

        # Define the regular expression pattern to match 'level' followed by a space and a number
        pattern = r"(level\s)(\d+)"

        # Replacement string, which includes the new level value
        replacement = r"\g<1>{}".format(level)

        # Substitute the new level value
        exact_location = re.sub(pattern, replacement, str(_neuron))

        # Remove newline characters from the updated string
        exact_location = exact_location.rstrip(" \n")

        return parent, _neuron, exact_location

    def assign_ghsom_clusters(self):
        """
        Assigns GHSOM cluster IDs to each sample in the train_data DataFrame.

        Returns:
        - pandas.DataFrame: A copy of the input DataFrame with an additional column 'GHSOM_cluster' containing the cluster IDs.
        """
        # Create a copy of the input DataFrame to avoid modifying the original data
        data = self.train_data.copy()

        # Ensure the column 'GHSOM_cluster' is of integer type
        if "GHSOM_cluster" in data.columns:
            data["GHSOM_cluster"] = pd.to_numeric(
                data["GHSOM_cluster"], downcast="integer", errors="coerce"
            )
        else:
            # If the column does not exist, initialize it as integer with NaNs which will be filled later
            data["GHSOM_cluster"] = pd.Series(dtype="Int64")

        # Iterate over the DataFrame and find the best matching unit for each sample
        for idx, v in enumerate(self.train_data.values):
            _, _, nr = self.find_best_matching_unit(v)
            # Use .at for faster access to update a single value
            data.at[idx, "GHSOM_cluster"] = encode_ghsom_node(self.lookup_table, nr)

        return data

    def get_neurons_similarity_matrix(self, neuron, sort=True):
        """
        Calculates distances and neighborhood relationships between a given neuron and all other neurons
        in a GHSOM model using Euclidean distance only.

        Parameters:
        - neuron (int or GHSOMNode): The neuron for which relationships are to be calculated. This can be
                                    either an integer index or a GHSOMNode object.

        Returns:
        - dict: A dictionary where keys are neuron indices and values are tuples containing:
                - The L2/Euclidean weight distance from the given neuron to the other neuron.
                - A boolean indicating if the two neurons are neighbors.
                - A boolean indicating if the two neurons are in the same column.
                - A boolean indicating if the two neurons are in the same row.
        """
        # Access the child map of the GHSOM model
        _gsom = self.ghsom_model.child_map
        distances = {}

        # If the neuron parameter is an integer, retrieve the corresponding GHSOMNode from the neuron_table
        if isinstance(neuron, int):
            neuron = self.neuron_table[neuron][1]

        # Iterate over all neurons in the neuron_table to calculate distances and relationships
        for idx, n in self.neuron_table.items():
            distances[idx] = (
                neuron.weight_distance_from_other_unit(n[1]),  # L2/Euclidean distance
                _gsom.are_neurons_neighbours(neuron, n[1]),  # Topology: adjacent
                _gsom.are_in_same_column(neuron, n[1]),  # Topology: column
                _gsom.are_in_same_row(neuron, n[1]),  # Topology: row
            )

        if sort:
            distances = sorted(distances.items(), key=lambda item: item[1][0])
            distances = {node[0]: node[1] for node in distances}
        return distances

    def get_unique_cluster_ids_list(self):
        """Returns a list of unique cluster IDs."""
        clusters = self.assign_ghsom_clusters()["GHSOM_cluster"]
        return clusters.dropna().unique().tolist()

    def get_cluster_ids(self):
        return self.assign_ghsom_clusters()["GHSOM_cluster"]

    def get_ghsom_node_statistics(self, node_id):
        """
        Reports statistics for a specific GHSOM node using its unique ID and the lookup table.

        Parameters:
        - node_id (str): The unique ID of the target node.

        Returns:
        - dict: A dictionary containing statistics for the target node, or None if not found.

        Note:
            This method wraps the ghsom_toolkits.core.get_ghsom_node_statistics function.
        """
        from ghsom_toolkits.core.parsing import (
            get_ghsom_node_statistics as core_get_stats,
        )

        return core_get_stats(self.lookup_table, node_id)

    def load_model(self, filename):
        """
        Load a Python model from a file using pickle.

        Parameters:
        - filename (str): The path to the file where the model is saved.

        Returns:
        - The loaded model object.
        """
        with open(filename, "rb") as file:
            return pickle.load(file)
