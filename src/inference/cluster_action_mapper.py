"""Cluster-to-action mapping and validation for preference-guided simulation.

This module provides mapping between GHSOM cluster IDs and RL action space indices,
with validation to ensure alignment and prevent silent failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ClusterActionMapping:
    """Maps cluster IDs to action space indices.

    This class ensures that cluster IDs used in preference scenarios
    are valid and map correctly to the agent's action space.

    Attributes:
        cluster_to_action: Dict mapping cluster ID to action index.
        action_to_cluster: Dict mapping action index to cluster ID.
        valid_cluster_ids: Set of all valid cluster IDs.
        action_space_size: Size of the action space.
    """

    cluster_to_action: Dict[int, int]
    action_to_cluster: Dict[int, int]
    valid_cluster_ids: Set[int]
    action_space_size: int

    def validate_cluster_ids(self, cluster_ids: List[int]) -> None:
        """Validate that cluster IDs are mappable to actions.

        Args:
            cluster_ids: List of cluster IDs to validate.

        Raises:
            ValueError: If any cluster ID is invalid or not in mapping.
        """
        if not cluster_ids:
            logger.warning("Empty cluster ID list provided for validation")
            return

        invalid = [cid for cid in cluster_ids if cid not in self.valid_cluster_ids]
        if invalid:
            raise ValueError(
                f"Invalid cluster IDs {invalid}. "
                f"Valid cluster IDs: {sorted(self.valid_cluster_ids)}"
            )

    def map_to_action(self, cluster_id: int) -> int:
        """Map cluster ID to action index.

        Args:
            cluster_id: Cluster ID to map.

        Returns:
            Action index corresponding to the cluster.

        Raises:
            ValueError: If cluster ID is not in mapping.
        """
        if cluster_id not in self.cluster_to_action:
            raise ValueError(
                f"Cluster ID {cluster_id} not in mapping. "
                f"Valid IDs: {sorted(self.valid_cluster_ids)}"
            )
        return self.cluster_to_action[cluster_id]

    def map_to_cluster(self, action_idx: int) -> int:
        """Map action index to cluster ID.

        Args:
            action_idx: Action index to map.

        Returns:
            Cluster ID corresponding to the action.

        Raises:
            ValueError: If action index is out of range.
        """
        if action_idx not in self.action_to_cluster:
            raise ValueError(
                f"Action index {action_idx} not in mapping. "
                f"Valid range: [0, {self.action_space_size})"
            )
        return self.action_to_cluster[action_idx]

    def get_cluster_for_action(self, action_idx: int) -> int:
        """Alias for map_to_cluster for backward compatibility."""
        return self.map_to_cluster(action_idx)

    def get_action_for_cluster(self, cluster_id: int) -> int:
        """Alias for map_to_action for backward compatibility."""
        return self.map_to_action(cluster_id)

    def to_dict(self) -> Dict:
        """Export mapping to dictionary."""
        return {
            "cluster_to_action": self.cluster_to_action,
            "action_to_cluster": self.action_to_cluster,
            "valid_cluster_ids": sorted(self.valid_cluster_ids),
            "action_space_size": self.action_space_size,
        }


def create_identity_mapping(action_space_size: int) -> ClusterActionMapping:
    """Create identity mapping where cluster_id == action_idx.

    This is the default mapping when cluster IDs directly correspond
    to action indices (e.g., when using cluster indices as actions).

    Args:
        action_space_size: Size of the action space.

    Returns:
        ClusterActionMapping with identity mapping.

    Raises:
        ValueError: If action_space_size is invalid.
    """
    if action_space_size <= 0:
        raise ValueError(f"Invalid action_space_size: {action_space_size}")

    cluster_to_action = {i: i for i in range(action_space_size)}
    action_to_cluster = {i: i for i in range(action_space_size)}
    valid_cluster_ids = set(range(action_space_size))

    logger.info(f"Created identity mapping: {action_space_size} clusters/actions")

    return ClusterActionMapping(
        cluster_to_action=cluster_to_action,
        action_to_cluster=action_to_cluster,
        valid_cluster_ids=valid_cluster_ids,
        action_space_size=action_space_size,
    )


def create_mapping_from_ghsom(
    ghsom_manager,
    action_space_size: int,
) -> ClusterActionMapping:
    """Create mapping from GHSOM cluster IDs to action indices.

    This function extracts cluster IDs from the GHSOM manager and creates
    a mapping to the action space. It validates that the number of clusters
    matches the action space size.

    Args:
        ghsom_manager: GHSOMManager instance with loaded model.
        action_space_size: Size of the agent's action space.

    Returns:
        ClusterActionMapping based on GHSOM structure.

    Raises:
        ValueError: If GHSOM clusters don't align with action space.
        AttributeError: If ghsom_manager doesn't have required methods.
    """
    # Validate ghsom_manager has required interface
    if not hasattr(ghsom_manager, "get_unique_cluster_ids_list"):
        raise AttributeError(
            "ghsom_manager must have get_unique_cluster_ids_list() method"
        )

    # Get cluster IDs from GHSOM
    try:
        cluster_ids = ghsom_manager.get_unique_cluster_ids_list()
    except Exception as e:
        raise ValueError(f"Failed to get cluster IDs from GHSOM: {e}") from e

    if not cluster_ids:
        raise ValueError("GHSOM returned empty cluster ID list")

    # Validate cluster count matches action space
    num_clusters = len(cluster_ids)
    if num_clusters != action_space_size:
        raise ValueError(
            f"GHSOM has {num_clusters} clusters but action space is "
            f"{action_space_size}. These must match for proper alignment. "
            f"GHSOM cluster IDs: {sorted(cluster_ids)}"
        )

    # Create mapping (sorted cluster IDs map to sequential action indices)
    sorted_cluster_ids = sorted(cluster_ids)
    cluster_to_action = {cid: idx for idx, cid in enumerate(sorted_cluster_ids)}
    action_to_cluster = {idx: cid for idx, cid in enumerate(sorted_cluster_ids)}
    valid_cluster_ids = set(cluster_ids)

    logger.info(
        f"Created GHSOM mapping: {num_clusters} clusters -> "
        f"{action_space_size} actions. "
        f"Cluster ID range: [{min(cluster_ids)}, {max(cluster_ids)}]"
    )

    return ClusterActionMapping(
        cluster_to_action=cluster_to_action,
        action_to_cluster=action_to_cluster,
        valid_cluster_ids=valid_cluster_ids,
        action_space_size=action_space_size,
    )


def create_mapping_from_perceiving_agent(perceiving_agent) -> ClusterActionMapping:
    """Create mapping from perceiving agent's cluster list.

    This creates a mapping that matches exactly what the environment uses,
    preserving the order from get_unique_cluster_ids_list() which is how
    actions map to clusters during training.

    Args:
        perceiving_agent: GHSOMPerceivingAgent instance.

    Returns:
        ClusterActionMapping matching environment's action-to-cluster mapping.

    Raises:
        AttributeError: If perceiving_agent doesn't have required method.
        ValueError: If no cluster IDs are available.
    """
    if not hasattr(perceiving_agent, "get_unique_cluster_ids_list"):
        raise AttributeError(
            "perceiving_agent must have get_unique_cluster_ids_list() method"
        )

    cluster_ids = perceiving_agent.get_unique_cluster_ids_list()

    if not cluster_ids:
        raise ValueError("Perceiving agent returned empty cluster ID list")

    # Create mapping preserving the EXACT order from perceiving agent
    # This matches how MusicGenerationGymEnv.step() maps actions to clusters
    cluster_to_action = {cid: idx for idx, cid in enumerate(cluster_ids)}
    action_to_cluster = {idx: cid for idx, cid in enumerate(cluster_ids)}
    valid_cluster_ids = set(cluster_ids)
    action_space_size = len(cluster_ids)

    logger.info(
        f"Created perceiving agent mapping: {action_space_size} actions. "
        f"Action 0 -> Cluster {cluster_ids[0]}, "
        f"Action {action_space_size-1} -> Cluster {cluster_ids[-1]}"
    )

    return ClusterActionMapping(
        cluster_to_action=cluster_to_action,
        action_to_cluster=action_to_cluster,
        valid_cluster_ids=valid_cluster_ids,
        action_space_size=action_space_size,
    )


def create_mapping_from_environment(env) -> ClusterActionMapping:
    """Create mapping from environment's cluster list.

    This extracts cluster IDs directly from the environment, handling
    wrapped environments by walking through wrappers.

    Args:
        env: Gymnasium environment (possibly wrapped).

    Returns:
        ClusterActionMapping matching environment's action-to-cluster mapping.

    Raises:
        ValueError: If environment doesn't have cluster_ids attribute.
    """
    # Walk through wrappers to find base environment with cluster_ids
    current = env
    for _ in range(10):  # prevent infinite loops
        if hasattr(current, "cluster_ids"):
            cluster_ids = current.cluster_ids
            break
        if hasattr(current, "env"):
            current = current.env
        else:
            break
    else:
        raise ValueError(
            "Could not find cluster_ids in environment. "
            "Ensure the environment has cluster_ids attribute."
        )

    if not cluster_ids:
        raise ValueError("Environment has empty cluster_ids list")

    # Create mapping preserving order
    cluster_to_action = {cid: idx for idx, cid in enumerate(cluster_ids)}
    action_to_cluster = {idx: cid for idx, cid in enumerate(cluster_ids)}
    valid_cluster_ids = set(cluster_ids)
    action_space_size = len(cluster_ids)

    logger.info(
        f"Created environment mapping: {action_space_size} actions. "
        f"Cluster IDs: {sorted(valid_cluster_ids)}"
    )

    return ClusterActionMapping(
        cluster_to_action=cluster_to_action,
        action_to_cluster=action_to_cluster,
        valid_cluster_ids=valid_cluster_ids,
        action_space_size=action_space_size,
    )


def create_custom_mapping(
    cluster_ids: List[int],
    action_space_size: int,
) -> ClusterActionMapping:
    """Create custom mapping from explicit cluster ID list.

    Useful when cluster IDs don't follow a simple pattern or when
    you need to specify a custom mapping.

    Args:
        cluster_ids: List of cluster IDs in order of action indices.
        action_space_size: Size of the action space.

    Returns:
        ClusterActionMapping with custom mapping.

    Raises:
        ValueError: If cluster_ids length doesn't match action_space_size
            or contains duplicates.
    """
    if len(cluster_ids) != action_space_size:
        raise ValueError(
            f"cluster_ids length ({len(cluster_ids)}) must match "
            f"action_space_size ({action_space_size})"
        )

    if len(set(cluster_ids)) != len(cluster_ids):
        duplicates = [cid for cid in cluster_ids if cluster_ids.count(cid) > 1]
        raise ValueError(f"cluster_ids contains duplicates: {set(duplicates)}")

    cluster_to_action = {cid: idx for idx, cid in enumerate(cluster_ids)}
    action_to_cluster = {idx: cid for idx, cid in enumerate(cluster_ids)}
    valid_cluster_ids = set(cluster_ids)

    logger.info(f"Created custom mapping: {len(cluster_ids)} clusters/actions")

    return ClusterActionMapping(
        cluster_to_action=cluster_to_action,
        action_to_cluster=action_to_cluster,
        valid_cluster_ids=valid_cluster_ids,
        action_space_size=action_space_size,
    )
