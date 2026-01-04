"""Training module for Tianshou-based RL training.

This module provides the core training infrastructure:
    - TianshouTrainer: Main trainer class
    - Callbacks for logging and checkpointing
    - SequenceReplayBuffer: Sequence-based replay for recurrent networks (Phase 3)
    - PrioritizedSequenceReplayBuffer: PER variant with sequence-level priorities

Example:
    >>> from src.training import TianshouTrainer
    >>> from src.training.callbacks import TianshouWandBCallback
    >>>
    >>> callback = TianshouWandBCallback(project="test")
    >>> trainer = TianshouTrainer(env=env, config=config, callbacks=[callback])
    >>> trainer.train()
"""

from .tianshou_trainer import TianshouTrainer
from .callbacks import (
    BaseTianshouCallback,
    CallbackList,
    TianshouWandBCallback,
    TianshouTensorBoardCallback,
    TianshouCheckpointCallback,
)
from .sequence_replay_buffer import (
    SequenceReplayBuffer,
    PrioritizedSequenceReplayBuffer,
    SequenceConfig,
)

__all__ = [
    "TianshouTrainer",
    "BaseTianshouCallback",
    "CallbackList",
    "TianshouWandBCallback",
    "TianshouTensorBoardCallback",
    "TianshouCheckpointCallback",
    "SequenceReplayBuffer",
    "PrioritizedSequenceReplayBuffer",
    "SequenceConfig",
]
