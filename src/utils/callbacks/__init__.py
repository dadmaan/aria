"""Tianshou callback utilities.

This module exports Tianshou-specific callbacks for training.

Note:
    Legacy SB3 callbacks have been archived to archive/sb3_legacy/callbacks/.
    Use the Tianshou callbacks from this module or src/training/callbacks/.

For the new standard callbacks, import from:
    from src.training.callbacks import TianshouWandBCallback, TianshouTensorBoardCallback, ...
"""

from .tianshou_callbacks import (
    BaseTianshouCallback,
    TianshouMetricsCallback,
    TianshouCheckpointCallback,
    TianshouLearningRateSchedulerCallback,
    TianshouDetailedRewardCallback,
    TianshouSimpleDashboardCallback,
    CallbackList,
)

__all__ = [
    "BaseTianshouCallback",
    "TianshouMetricsCallback",
    "TianshouCheckpointCallback",
    "TianshouLearningRateSchedulerCallback",
    "TianshouDetailedRewardCallback",
    "TianshouSimpleDashboardCallback",
    "CallbackList",
]
