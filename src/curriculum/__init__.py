"""Curriculum Learning Module for Dynamic Coarse-to-Fine Training.

This module provides curriculum learning capabilities for hierarchical
action spaces based on GHSOM structure. It enables progressive expansion
of action spaces during training.

Components:
    - HierarchyExtractor: Extract curriculum phases from GHSOM
    - PhaseManager: Manage phase transitions during training
    - WeightMitosis: Expand Q-network output layers
    - CurriculumCallback: Tianshou callback for curriculum
    - CurriculumEnvironmentWrapper: Action mapping wrapper
"""

from src.curriculum.hierarchy_extractor import (
    DynamicHierarchy,
    HierarchyExtractor,
    PhaseDefinition,
)
from src.curriculum.phase_manager import (
    DynamicPhaseManager,
    PhaseRuntimeConfig,
    PhaseTransitionTrigger,
)
from src.curriculum.weight_mitosis import WeightMitosis
from src.curriculum.curriculum_callback import CurriculumCallback

__all__ = [
    # Hierarchy extraction
    "HierarchyExtractor",
    "DynamicHierarchy",
    "PhaseDefinition",
    # Phase management
    "DynamicPhaseManager",
    "PhaseRuntimeConfig",
    "PhaseTransitionTrigger",
    # Weight mitosis
    "WeightMitosis",
    # Callback
    "CurriculumCallback",
]
