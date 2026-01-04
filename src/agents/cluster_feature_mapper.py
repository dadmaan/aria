"""Cluster Feature Mapper for Fixed Feature Vector Observations.

Maps cluster IDs to their representative feature vectors (GHSOM prototypes
or computed centroids in t-SNE space).

Handles special padding cluster ID (PADDING_CLUSTER_ID = -1) with a distinct
embedding vector to prevent confusion with valid GHSOM clusters.
"""

import numpy as np
from typing import Dict, Optional, Literal
from pathlib import Path
import logging

from ..environments.constants import PADDING_CLUSTER_ID, PADDING_EMBEDDING

logger = logging.getLogger(__name__)


class ClusterFeatureMapper:
    """
    Maps cluster IDs to fixed feature vectors for RL observations.

    Supports two modes:
    - 'prototype': Use GHSOM node weight vectors (learned cluster centers)
    - 'centroid': Compute mean of samples assigned to each cluster

    Example:
        mapper = ClusterFeatureMapper(ghsom_manager, mode='prototype')

        # Map single cluster ID
        features = mapper.get_features(5)  # Shape: (2,)

        # Map sequence of cluster IDs
        seq_features = mapper.map_sequence([5, 12, 5, 8])  # Shape: (4, 2)
    """

    def __init__(
        self,
        ghsom_manager,
        mode: Literal["prototype", "centroid"] = "prototype",
        feature_source: Literal["tsne", "raw"] = "tsne",
    ):
        """
        Initialize cluster feature mapper.

        Args:
            ghsom_manager: Trained GHSOM manager instance
            mode: How to derive cluster features
                - 'prototype': Use GHSOM learned weight vectors (recommended)
                - 'centroid': Compute mean of samples in cluster
            feature_source: Which feature space to use
                - 'tsne': 2D t-SNE embeddings (recommended, preserves similarity)
                - 'raw': Original features (richer but higher-dimensional)
        """
        self.ghsom_manager = ghsom_manager
        self.mode = mode
        self.feature_source = feature_source

        # Build cluster ID -> feature vector mapping
        self.cluster_to_features = self._build_mapping()

        # Validate we have features for all clusters
        if not self.cluster_to_features:
            raise ValueError("No cluster features found. Check GHSOM manager setup.")

        # Get feature dimensionality
        first_feature = next(iter(self.cluster_to_features.values()))
        self.feature_dim = first_feature.shape[0]
        self.n_clusters = len(self.cluster_to_features)

        logger.info(
            f"ClusterFeatureMapper initialized: "
            f"mode={mode}, source={feature_source}, "
            f"n_clusters={self.n_clusters}, feature_dim={self.feature_dim}"
        )

    def _build_mapping(self) -> Dict[int, np.ndarray]:
        """
        Build mapping from cluster IDs to feature vectors.

        Returns:
            Dictionary: {cluster_id: feature_vector}
            where feature_vector has shape (feature_dim,)
        """
        if self.mode == "prototype":
            return self._build_from_prototypes()
        elif self.mode == "centroid":
            return self._build_from_centroids()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _build_from_prototypes(self) -> Dict[int, np.ndarray]:
        """
        Extract GHSOM node weight vectors (learned prototypes).

        GHSOM nodes store learned weight vectors that represent
        the cluster center in the input space (2D t-SNE or original features).

        Returns:
            Dictionary mapping cluster_id -> prototype vector
        """
        mapping = {}

        # Get all leaf nodes (clusters) from GHSOMManager.neuron_table
        # neuron_table: {node_id: (representation, neuron)}
        for node_id, (
            representation,
            neuron,
        ) in self.ghsom_manager.neuron_table.items():
            # Extract weight vector from neuron object
            prototype = neuron.weight_vector()

            # Ensure correct dtype and shape
            mapping[node_id] = np.array(prototype, dtype=np.float32)

        logger.info(
            f"Built prototype mapping: {len(mapping)} clusters, "
            f"feature_dim={mapping[next(iter(mapping))].shape[0] if mapping else 'unknown'}"
        )

        return mapping

    def _build_from_centroids(self) -> Dict[int, np.ndarray]:
        """
        Compute centroids from samples assigned to each cluster.

        Returns:
            Dictionary mapping cluster_id -> centroid vector
        """
        mapping = {}

        # Get cluster assignments: DataFrame with 'GHSOM_cluster' column
        cluster_assignments_df = self.ghsom_manager.assign_ghsom_clusters()
        cluster_assignments = cluster_assignments_df["GHSOM_cluster"].values

        # Get feature embeddings: train_data already contains features (tsne or raw)
        # depending on feature_type used during GHSOMManager initialization
        features = self.ghsom_manager.train_data.values

        if len(cluster_assignments) != len(features):
            raise ValueError(
                f"Mismatch: {len(cluster_assignments)} assignments but "
                f"{len(features)} features"
            )

        # Compute centroid for each unique cluster
        unique_clusters = np.unique(cluster_assignments)

        for cluster_id in unique_clusters:
            # Find all samples belonging to this cluster
            mask = cluster_assignments == cluster_id
            cluster_samples = features[mask]

            # Compute mean (centroid)
            centroid = np.mean(cluster_samples, axis=0)

            mapping[int(cluster_id)] = centroid.astype(np.float32)

        logger.info(
            f"Built centroid mapping: {len(mapping)} clusters, "
            f"feature_dim={mapping[next(iter(mapping))].shape[0] if mapping else 'unknown'}"
        )

        return mapping

    def get_features(self, cluster_id: int) -> np.ndarray:
        """
        Get feature vector for a single cluster ID.

        Args:
            cluster_id: Integer cluster ID (or PADDING_CLUSTER_ID for padding)

        Returns:
            Feature vector of shape (feature_dim,)
            - Returns PADDING_EMBEDDING for PADDING_CLUSTER_ID (-1)
            - Returns zero vector for other invalid cluster IDs

        Example:
            >>> mapper = ClusterFeatureMapper(ghsom_manager)
            >>> features = mapper.get_features(5)
            >>> features.shape
            (2,)
            >>> features
            array([0.34, -1.21], dtype=float32)

            >>> # Padding returns distinct embedding
            >>> features = mapper.get_features(PADDING_CLUSTER_ID)
            >>> features
            array([-100., -100.], dtype=float32)
        """
        # Handle padding cluster ID with distinct embedding
        if cluster_id == PADDING_CLUSTER_ID:
            return self._get_padding_embedding()

        if cluster_id not in self.cluster_to_features:
            # Return zero vector for truly invalid cluster IDs (not padding)
            # Log warning first time this cluster ID is encountered
            if not hasattr(self, "_warned_clusters"):
                self._warned_clusters = set()

            if cluster_id not in self._warned_clusters:
                logger.warning(
                    f"Cluster ID {cluster_id} not found in GHSOM mapping. "
                    f"Returning zero vector. Valid IDs: {sorted(self.cluster_to_features.keys())}"
                )
                self._warned_clusters.add(cluster_id)

            return np.zeros(self.feature_dim, dtype=np.float32)

        return self.cluster_to_features[cluster_id]

    def _get_padding_embedding(self) -> np.ndarray:
        """
        Get the padding embedding vector.

        Returns:
            Padding embedding with shape (feature_dim,), values from PADDING_EMBEDDING.
            If feature_dim != 2, extends/truncates PADDING_EMBEDDING appropriately.
        """
        if self.feature_dim == 2:
            return np.array(PADDING_EMBEDDING, dtype=np.float32)
        elif self.feature_dim < 2:
            return np.array(PADDING_EMBEDDING[: self.feature_dim], dtype=np.float32)
        else:
            # Extend with the same value for higher dimensions
            return np.full(self.feature_dim, PADDING_EMBEDDING[0], dtype=np.float32)

    def is_padding(self, cluster_id: int) -> bool:
        """
        Check if a cluster ID is the padding token.

        Args:
            cluster_id: Cluster ID to check

        Returns:
            True if cluster_id is PADDING_CLUSTER_ID
        """
        return cluster_id == PADDING_CLUSTER_ID

    def map_sequence(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Convert sequence of cluster IDs to sequence of feature vectors.

        Args:
            cluster_ids: Array of cluster IDs, shape (seq_len,)

        Returns:
            Feature sequence, shape (seq_len, feature_dim)

        Example:
            >>> mapper = ClusterFeatureMapper(ghsom_manager)
            >>> seq = np.array([5, 12, 5, 8])
            >>> features = mapper.map_sequence(seq)
            >>> features.shape
            (4, 2)
            >>> features
            array([[0.34, -1.21],   # cluster 5 prototype
                   [0.81,  0.43],   # cluster 12 prototype
                   [0.34, -1.21],   # cluster 5 (repeated)
                   [-0.52, 0.91]],  # cluster 8 prototype
                  dtype=float32)
        """
        feature_sequence = np.array(
            [self.get_features(int(cid)) for cid in cluster_ids], dtype=np.float32
        )

        return feature_sequence

    def save(self, path: Path) -> None:
        """
        Save cluster feature mapping to disk.

        Args:
            path: Path to save .npz file
        """
        np.savez(
            path,
            cluster_to_features=self.cluster_to_features,
            mode=self.mode,
            feature_source=self.feature_source,
            feature_dim=self.feature_dim,
            n_clusters=self.n_clusters,
        )
        logger.info(f"Saved ClusterFeatureMapper to {path}")

    @classmethod
    def load(cls, path: Path):
        """
        Load cluster feature mapping from disk.

        Args:
            path: Path to .npz file

        Returns:
            ClusterFeatureMapper instance
        """
        data = np.load(path, allow_pickle=True)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.cluster_to_features = data["cluster_to_features"].item()
        instance.mode = str(data["mode"])
        instance.feature_source = str(data["feature_source"])
        instance.feature_dim = int(data["feature_dim"])
        instance.n_clusters = int(data["n_clusters"])
        instance.ghsom_manager = None  # Not serialized

        logger.info(f"Loaded ClusterFeatureMapper from {path}")
        return instance
