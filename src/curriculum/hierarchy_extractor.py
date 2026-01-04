"""Hierarchy Extractor for Dynamic Curriculum Learning.

This module extracts curriculum phases from a GHSOM hierarchy structure.
It automatically identifies levels in the GHSOM tree and creates phase
definitions for progressive action space expansion.

Classes:
    PhaseDefinition: Configuration for a single curriculum phase.
    DynamicHierarchy: Complete hierarchy data structure with all phases.
    HierarchyExtractor: Main extraction logic from GHSOM models.

Example:
    >>> extractor = HierarchyExtractor.from_experiment_dir(
    ...     Path("experiments/ghsom_model/")
    ... )
    >>> hierarchy = extractor.extract()
    >>> print(f"Total phases: {hierarchy.total_phases}")
    >>> for phase_num, phase_def in hierarchy.phases.items():
    ...     print(f"Phase {phase_num}: {phase_def.action_space_size} actions")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from ghsom_toolkits.core import (
    GHSOMNode,
    create_lookup_table,
    get_clusters_by_level,
    get_ghsom_statistics,
    parse_ghsom_hierarchy,
)

from src.utils.logging.logging_manager import get_logger


@dataclass
class PhaseDefinition:
    """Configuration for a single curriculum phase.

    Each phase defines a coarse-grained action space where actions
    correspond to nodes at a specific level in the GHSOM hierarchy.

    Attributes:
        phase_number: 1-indexed phase number.
        level: GHSOM hierarchy level for this phase.
        action_space_size: Number of actions in this phase.
        action_to_node_id: Mapping from action index to node ID.
        node_id_to_action: Reverse mapping from node ID to action index.
        parent_to_children: Mapping from parent action to child actions
            (used for weight mitosis in next phase transition).
        leaf_descendants: For each action, list of leaf cluster IDs
            that are descendants (for action mapping).
    """

    phase_number: int
    level: int
    action_space_size: int
    action_to_node_id: Dict[int, Union[str, int]]
    node_id_to_action: Dict[Union[str, int], int]
    parent_to_children: Dict[int, List[int]] = field(default_factory=dict)
    leaf_descendants: Dict[int, List[int]] = field(default_factory=dict)

    def get_leaf_actions_for_action(self, action: int) -> List[int]:
        """Get leaf cluster actions that descend from this action.

        Args:
            action: Action index in current phase.

        Returns:
            List of leaf cluster IDs (final phase actions).
        """
        return self.leaf_descendants.get(action, [action])


@dataclass
class DynamicHierarchy:
    """Complete hierarchy data structure for curriculum learning.

    Contains all phase definitions extracted from the GHSOM model,
    along with mappings between phases for weight mitosis.

    Attributes:
        phases: Dict mapping phase number to PhaseDefinition.
        total_phases: Total number of phases in the curriculum.
        total_leaf_clusters: Number of leaf clusters (final actions).
        leaf_lookup: Mapping from leaf cluster ID to node.
        root_node: Root node of the GHSOM hierarchy.
    """

    phases: Dict[int, PhaseDefinition]
    total_phases: int
    total_leaf_clusters: int
    leaf_lookup: Dict[int, GHSOMNode]
    root_node: GHSOMNode

    def get_phase(self, phase_number: int) -> Optional[PhaseDefinition]:
        """Get phase definition by number.

        Args:
            phase_number: 1-indexed phase number.

        Returns:
            PhaseDefinition or None if not found.
        """
        return self.phases.get(phase_number)

    def get_final_phase(self) -> PhaseDefinition:
        """Get the final phase (leaf clusters).

        Returns:
            PhaseDefinition for the final phase.
        """
        return self.phases[self.total_phases]

    def get_mitosis_mapping(
        self, from_phase: int, to_phase: int
    ) -> Dict[int, List[int]]:
        """Get action mapping for weight mitosis between phases.

        Args:
            from_phase: Source phase number.
            to_phase: Target phase number.

        Returns:
            Dict mapping source actions to target actions.

        Raises:
            ValueError: If phases are not adjacent.
        """
        if to_phase != from_phase + 1:
            raise ValueError(
                f"Mitosis mapping only supports adjacent phases. "
                f"Got from_phase={from_phase}, to_phase={to_phase}"
            )

        return self.phases[to_phase].parent_to_children


class HierarchyExtractor:
    """Extract curriculum phases from GHSOM model.

    This class analyzes a GHSOM hierarchy and creates phase definitions
    for coarse-to-fine curriculum learning. Each level in the GHSOM
    becomes a phase, with the final phase containing all leaf clusters.

    Attributes:
        root_node: Root of the GHSOM hierarchy.
        lookup_table: Node ID lookup table.
        stats: GHSOM statistics.
        logger: Logger instance.

    Example:
        >>> # From experiment directory
        >>> extractor = HierarchyExtractor.from_experiment_dir(
        ...     Path("experiments/ghsom_model/")
        ... )
        >>> hierarchy = extractor.extract()
        >>>
        >>> # Or from parsed root node
        >>> extractor = HierarchyExtractor(root_node, lookup_table)
        >>> hierarchy = extractor.extract()
    """

    def __init__(
        self,
        root_node: GHSOMNode,
        lookup_table: Dict[Union[str, int], GHSOMNode],
    ):
        """Initialize hierarchy extractor.

        Args:
            root_node: Root node of the GHSOM hierarchy.
            lookup_table: Node ID lookup table from create_lookup_table().
        """
        self.root_node = root_node
        self.lookup_table = lookup_table
        self.stats = get_ghsom_statistics(root_node)
        self.logger = get_logger("HierarchyExtractor")

    @classmethod
    def from_experiment_dir(cls, experiment_dir: Path) -> "HierarchyExtractor":
        """Create extractor from experiment directory.

        Loads the GHSOM model from the standard experiment directory
        structure (ghsom_model.pkl).

        Args:
            experiment_dir: Path to experiment directory.

        Returns:
            HierarchyExtractor instance.

        Raises:
            FileNotFoundError: If model file not found.
        """
        import pickle

        model_path = experiment_dir / "ghsom_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"GHSOM model not found: {model_path}")

        with open(model_path, "rb") as f:
            ghsom_model = pickle.load(f)

        # Parse hierarchy from model string representation
        root_node = parse_ghsom_hierarchy(str(ghsom_model))
        lookup_table = create_lookup_table(root_node, short_id=True)

        return cls(root_node, lookup_table)

    @classmethod
    def from_ghsom_manager(cls, ghsom_manager) -> "HierarchyExtractor":
        """Create extractor from GHSOMManager instance.

        Args:
            ghsom_manager: GHSOMManager instance with loaded model.

        Returns:
            HierarchyExtractor instance.
        """
        return cls(ghsom_manager.root_node, ghsom_manager.lookup_table)

    def extract(self) -> DynamicHierarchy:
        """Extract curriculum hierarchy from GHSOM.

        Implements a PROGRESSIVE EXPANSION curriculum where:
        - Phase 1: Coarse actions (level-1 nodes representing groups of leaves)
        - Phase N: Previous phase actions + newly expanded children
        - Final Phase: All leaf clusters (full action space)

        At each phase, non-leaf nodes from the previous phase are "expanded"
        (replaced by their children), while leaf nodes are retained. This
        ensures the action space GROWS progressively until all leaves are
        available.

        Returns:
            DynamicHierarchy with all phase definitions.
        """
        self.logger.info("Extracting curriculum hierarchy from GHSOM...")
        self.logger.info(f"GHSOM stats: {self.stats}")

        # Get all levels in the hierarchy
        levels_dict = self.stats["levels"]
        levels = sorted(levels_dict.keys())

        self.logger.info(f"Found {len(levels)} levels: {levels}")

        # Get all leaf clusters (nodes with no children)
        leaf_nodes = self.root_node.get_leaves()
        leaf_lookup = self._create_leaf_lookup(leaf_nodes)
        total_leaf_clusters = len(leaf_nodes)

        self.logger.info(f"Total leaf clusters: {total_leaf_clusters}")

        # Create a mapping from leaf node IDs to 0-indexed action indices
        # in the FINAL phase (all leaves)
        leaf_node_id_to_leaf_action = {}
        for leaf_action_idx, (node_id, _) in enumerate(leaf_lookup.items()):
            leaf_node_id_to_leaf_action[node_id] = leaf_action_idx

        # Build phases using PROGRESSIVE EXPANSION
        # Each phase contains: retained leaves from previous + expanded children
        phases: Dict[int, PhaseDefinition] = {}

        # Phase 1: All level-1 nodes (children of root)
        level_1_nodes = get_clusters_by_level(self.root_node, 1)
        if not level_1_nodes:
            raise ValueError("GHSOM has no level-1 nodes (root has no children)")

        # Track current phase's active nodes (starts with level-1)
        current_phase_nodes: List[GHSOMNode] = list(level_1_nodes)
        phase_number = 1
        max_level = max(levels)

        while True:
            # Create action mappings for current phase nodes
            action_to_node_id, node_id_to_action = self._create_action_mappings(
                current_phase_nodes
            )

            # Get leaf descendants for each action
            leaf_descendants = self._get_leaf_descendants(
                current_phase_nodes, leaf_lookup, leaf_node_id_to_leaf_action
            )

            # Determine the "level" for this phase (max level of nodes in this phase)
            phase_level = max(n.level for n in current_phase_nodes)

            phase_def = PhaseDefinition(
                phase_number=phase_number,
                level=phase_level,
                action_space_size=len(current_phase_nodes),
                action_to_node_id=action_to_node_id,
                node_id_to_action=node_id_to_action,
                leaf_descendants=leaf_descendants,
            )

            phases[phase_number] = phase_def
            self.logger.info(
                f"Phase {phase_number}: level={phase_level}, "
                f"actions={len(current_phase_nodes)}"
            )

            # Check if we've reached the final phase (all nodes are leaves)
            all_leaves = all(len(n.children) == 0 for n in current_phase_nodes)
            if all_leaves:
                self.logger.info(
                    f"Phase {phase_number} is final (all {len(current_phase_nodes)} nodes are leaves)"
                )
                break

            # Build next phase: expand non-leaf nodes, keep leaf nodes
            next_phase_nodes: List[GHSOMNode] = []
            for node in current_phase_nodes:
                if len(node.children) == 0:
                    # Leaf node: keep it in next phase
                    next_phase_nodes.append(node)
                else:
                    # Non-leaf node: replace with its children
                    next_phase_nodes.extend(node.children)

            current_phase_nodes = next_phase_nodes
            phase_number += 1

            # Safety check to prevent infinite loops
            if phase_number > max_level + 5:
                self.logger.error(
                    f"Too many phases ({phase_number}), stopping extraction"
                )
                break

        # Build parent-to-children mappings for mitosis
        self._build_parent_child_mappings_progressive(phases)

        total_phases = len(phases)

        hierarchy = DynamicHierarchy(
            phases=phases,
            total_phases=total_phases,
            total_leaf_clusters=total_leaf_clusters,
            leaf_lookup=leaf_lookup,
            root_node=self.root_node,
        )

        self.logger.info(
            f"Extraction complete: {total_phases} phases, "
            f"{total_leaf_clusters} leaf clusters"
        )

        return hierarchy

    def _create_leaf_lookup(self, leaf_nodes: List[GHSOMNode]) -> Dict[int, GHSOMNode]:
        """Create lookup from leaf cluster ID to node.

        Args:
            leaf_nodes: List of leaf nodes.

        Returns:
            Dict mapping cluster ID to node.
        """
        leaf_lookup = {}
        for node in leaf_nodes:
            # Find the node ID in lookup table
            for node_id, lookup_node in self.lookup_table.items():
                if node_id != "root" and lookup_node is node:
                    leaf_lookup[node_id] = node
                    break
        return leaf_lookup

    def _create_action_mappings(self, nodes: List[GHSOMNode]) -> tuple:
        """Create bidirectional action-node mappings.

        Args:
            nodes: List of nodes at a level.

        Returns:
            Tuple of (action_to_node_id, node_id_to_action).
        """
        action_to_node_id = {}
        node_id_to_action = {}

        action_idx = 0
        for node in nodes:
            # Find node ID in lookup table
            node_id = self._find_node_id(node)
            if node_id is not None:
                action_to_node_id[action_idx] = node_id
                node_id_to_action[node_id] = action_idx
                action_idx += 1

        return action_to_node_id, node_id_to_action

    def _find_node_id(self, node: GHSOMNode) -> Optional[Union[str, int]]:
        """Find node ID in lookup table.

        Args:
            node: GHSOMNode to find.

        Returns:
            Node ID or None if not found.
        """
        for node_id, lookup_node in self.lookup_table.items():
            if lookup_node is node:
                return node_id
        return None

    def _get_leaf_descendants(
        self,
        nodes: List[GHSOMNode],
        leaf_lookup: Dict[int, GHSOMNode],
        leaf_node_id_to_action: Dict[int, int],
    ) -> Dict[int, List[int]]:
        """Get leaf cluster action indices descended from each node.

        Args:
            nodes: List of nodes at current level.
            leaf_lookup: Mapping from leaf ID to node.
            leaf_node_id_to_action: Mapping from leaf node ID to final
                phase action index.

        Returns:
            Dict mapping action index to list of leaf action indices.
        """
        leaf_descendants = {}

        # Reverse lookup: node -> leaf_node_id
        node_to_leaf_id = {node: lid for lid, node in leaf_lookup.items()}

        for action_idx, node in enumerate(nodes):
            node_id = self._find_node_id(node)
            if node_id is None:
                continue

            # Get all leaves under this node
            leaves = node.get_leaves()

            # Map to leaf action indices (not node IDs!)
            leaf_action_indices = []
            for leaf in leaves:
                if leaf in node_to_leaf_id:
                    leaf_node_id = node_to_leaf_id[leaf]
                    # Convert node ID to action index in final phase
                    if leaf_node_id in leaf_node_id_to_action:
                        leaf_action_indices.append(leaf_node_id_to_action[leaf_node_id])

            leaf_descendants[action_idx] = leaf_action_indices

        return leaf_descendants

    def _build_parent_child_mappings_progressive(
        self, phases: Dict[int, PhaseDefinition]
    ) -> None:
        """Build parent-to-children mappings for progressive expansion curriculum.

        In progressive expansion:
        - Leaf nodes in phase N remain as the SAME action in phase N+1
        - Non-leaf nodes in phase N are EXPANDED into their children in phase N+1

        The mapping tracks:
        - Which parent actions are expanded (non-leaves)
        - Which actions their children become in the next phase
        - Leaf actions that remain unchanged (identity mapping)

        Modifies phases in place to add parent_to_children mappings.

        Args:
            phases: Dict of phase definitions.
        """
        phase_nums = sorted(phases.keys())

        for phase_num in phase_nums[:-1]:
            parent_phase = phases[phase_num]
            child_phase = phases[phase_num + 1]

            parent_to_children: Dict[int, List[int]] = {}

            # For each parent action, determine its mapping to child phase
            for parent_action, parent_node_id in parent_phase.action_to_node_id.items():
                parent_node = self.lookup_table.get(parent_node_id)
                if parent_node is None:
                    continue

                if len(parent_node.children) == 0:
                    # Leaf node: maps to itself in child phase (find its new action index)
                    # The same node should exist in child phase
                    if parent_node_id in child_phase.node_id_to_action:
                        child_action = child_phase.node_id_to_action[parent_node_id]
                        parent_to_children[parent_action] = [child_action]
                else:
                    # Non-leaf node: expands into its children
                    child_actions = []
                    for child_node in parent_node.children:
                        child_node_id = self._find_node_id(child_node)
                        if (
                            child_node_id
                            and child_node_id in child_phase.node_id_to_action
                        ):
                            child_action = child_phase.node_id_to_action[child_node_id]
                            child_actions.append(child_action)

                    if child_actions:
                        parent_to_children[parent_action] = sorted(child_actions)

            # Store in the child phase (used during transition to child phase)
            child_phase.parent_to_children = parent_to_children

    def _build_parent_child_mappings(self, phases: Dict[int, PhaseDefinition]) -> None:
        """Build parent-to-children mappings between adjacent phases.

        Modifies phases in place to add parent_to_children mappings.

        Args:
            phases: Dict of phase definitions.
        """
        phase_nums = sorted(phases.keys())

        for _, phase_num in enumerate(phase_nums[:-1]):
            parent_phase = phases[phase_num]
            child_phase = phases[phase_num + 1]

            parent_to_children = {}

            # For each parent action, find which child actions descend from it
            for parent_action, parent_node_id in parent_phase.action_to_node_id.items():
                parent_node = self.lookup_table.get(parent_node_id)
                if parent_node is None:
                    continue

                # Find children at the next level
                child_actions = []
                for (
                    child_action,
                    child_node_id,
                ) in child_phase.action_to_node_id.items():
                    child_node = self.lookup_table.get(child_node_id)
                    if child_node is None:
                        continue

                    # Check if child is a descendant of parent
                    if self._is_descendant(parent_node, child_node):
                        child_actions.append(child_action)

                if child_actions:
                    parent_to_children[parent_action] = child_actions

            # Store in the child phase (used during transition to child phase)
            child_phase.parent_to_children = parent_to_children

    def _is_descendant(self, ancestor: GHSOMNode, descendant: GHSOMNode) -> bool:
        """Check if one node is a descendant of another.

        Args:
            ancestor: Potential ancestor node.
            descendant: Potential descendant node.

        Returns:
            True if descendant is a child/grandchild/etc of ancestor.
        """
        # Direct children check
        if descendant in ancestor.children:
            return True

        # Recursive check through all descendants
        for child in ancestor.children:
            if self._is_descendant(child, descendant):
                return True

        return False
