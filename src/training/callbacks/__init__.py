"""Training callbacks for Tianshou.

This module provides callbacks for logging, checkpointing, and
monitoring during Tianshou training.

Available Callbacks:
    - BaseTianshouCallback: Abstract base class for callbacks
    - CallbackList: Container for multiple callbacks
    - TianshouWandBCallback: Weights & Biases logging
    - TianshouTensorBoardCallback: TensorBoard logging
    - TianshouCheckpointCallback: Automatic checkpoint saving
    - TianshouComprehensiveMetricsCallback: Comprehensive metrics tracking
    - TianshouRewardComponentsCallback: Individual reward component logging
    - TianshouLRSchedulerCallback: Learning rate scheduling with pulse mechanism

Example:
    >>> from src.training.callbacks import (
    ...     TianshouWandBCallback,
    ...     TianshouTensorBoardCallback,
    ...     TianshouCheckpointCallback,
    ...     TianshouComprehensiveMetricsCallback,
    ...     TianshouRewardComponentsCallback,
    ...     TianshouLRSchedulerCallback,
    ...     CallbackList,
    ... )
    >>>
    >>> callbacks = CallbackList([
    ...     TianshouWandBCallback(project="music-rl"),
    ...     TianshouTensorBoardCallback(log_dir="runs/exp1"),
    ...     TianshouCheckpointCallback(checkpoint_dir="checkpoints"),
    ...     TianshouComprehensiveMetricsCallback(config=config),
    ...     TianshouRewardComponentsCallback(output_dir="artifacts/training"),
    ...     TianshouLRSchedulerCallback(initial_lr=0.001, final_lr=0.0001),
    ... ])
"""

from .base import BaseTianshouCallback, CallbackList
from .checkpoint_callback import TianshouCheckpointCallback
from .comprehensive_metrics_callback import TianshouComprehensiveMetricsCallback
from .lr_scheduler_callback import TianshouLRSchedulerCallback
from .reward_components_callback import TianshouRewardComponentsCallback
from .tensorboard_callback import TianshouTensorBoardCallback
from .wandb_callback import TianshouWandBCallback

__all__ = [
    "BaseTianshouCallback",
    "CallbackList",
    "TianshouWandBCallback",
    "TianshouTensorBoardCallback",
    "TianshouCheckpointCallback",
    "TianshouComprehensiveMetricsCallback",
    "TianshouRewardComponentsCallback",
    "TianshouLRSchedulerCallback",
]
