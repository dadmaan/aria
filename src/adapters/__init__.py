"""RL framework adapters.

This module provides the adapter layer between the music environment
and the Tianshou training framework.

Note:
    SB3 support has been removed in Phase 2D migration. For legacy SB3 code, see
    archive/sb3_legacy/adapters/

    Legacy pretraining (TorchPretrainer, factory, feature_loader) has been archived.
    Use the new training infrastructure from src/training/ and src/networks/ instead.

Components:
    - BaseAgentAdapter: Abstract base class for RL adapters
    - BackendRegistry: Registry for adapter instantiation
    - TianshouDRQNAdapter: Tianshou DRQN adapter (primary training interface)
"""

from .base import BaseAgentAdapter, BackendRegistry
from .tianshou_adapter import TianshouDRQNAdapter

__all__ = [
    # Base classes
    "BaseAgentAdapter",
    "BackendRegistry",
    # Tianshou adapter (primary)
    "TianshouDRQNAdapter",
]
