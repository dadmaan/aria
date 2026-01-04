"""
Visualization utilities for terminal UI display.

This module provides Rich-based terminal UI components for real-time
training visualization in multi-agent RL systems.
"""

from src.utils.visualization.training_ui import TrainingUI

# Note: TerminalUICallback was moved to archive/sb3_legacy/ as part of
# migration from SB3 to Tianshou. Import only if available.
try:
    from src.utils.visualization.callbacks import TerminalUICallback

    __all__ = ["TrainingUI", "TerminalUICallback"]
except ImportError:
    __all__ = ["TrainingUI"]
