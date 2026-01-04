"""Base callback class for Tianshou training.

This module defines the callback interface that all training callbacks
must implement. Callbacks are called at specific points during training
to enable logging, checkpointing, and other side effects.

Classes:
    BaseTianshouCallback: Abstract base class for callbacks.
    CallbackList: Container for multiple callbacks.

Example:
    >>> class MyCallback(BaseTianshouCallback):
    ...     def on_epoch_end(self, epoch, metrics, trainer):
    ...         print(f"Epoch {epoch}: reward={metrics['reward_mean']:.2f}")
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...utils.logging.logging_manager import get_logger

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class BaseTianshouCallback(ABC):
    """Abstract base class for Tianshou training callbacks.

    Callbacks are called at specific points during training:
        - on_training_start: Once before training begins
        - on_epoch_start: At the start of each epoch
        - on_collect_end: After each data collection step
        - on_update_end: After each policy update
        - on_epoch_end: At the end of each epoch
        - on_training_end: Once after training completes

    Subclasses should override the methods they need. All methods
    receive the trainer instance for access to training state.

    Attributes:
        verbose (int): Verbosity level (0=silent, 1=info, 2=debug).
        logger: Logger instance for this callback.

    Example:
        >>> class PrintCallback(BaseTianshouCallback):
        ...     def on_epoch_end(self, epoch, metrics, trainer):
        ...         self.logger.info(f"Epoch {epoch} complete")
    """

    def __init__(self, verbose: int = 0):
        """Initialize callback.

        Args:
            verbose: Verbosity level.
        """
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Called once before training begins.

        Args:
            trainer: TianshouTrainer instance.
        """
        pass

    def on_epoch_start(self, epoch: int, trainer: "TianshouTrainer") -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            trainer: TianshouTrainer instance.
        """
        pass

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Called after each data collection step.

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        pass

    def on_update_end(
        self,
        epoch: int,
        update_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Called after each policy update.

        Args:
            epoch: Current epoch number.
            update_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        pass

    def on_train_step_end(
        self,
        train_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Called after each training step (alias for on_update_end).

        This provides compatibility with existing callback implementations
        that use on_train_step_end instead of on_update_end.

        Args:
            train_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        pass

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Called at the end of each epoch.

        This is the primary callback for logging metrics, as Tianshou's
        OffpolicyTrainer computes metrics at epoch boundaries.

        Args:
            epoch: Current epoch number.
            metrics: Dict with training/eval metrics:
                - train/reward: Training episode reward
                - train/length: Training episode length
                - test/reward: Evaluation reward (if test_collector)
                - loss: Policy loss
                - etc.
            trainer: TianshouTrainer instance.
        """
        pass

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Called once after training completes.

        Args:
            trainer: TianshouTrainer instance.
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks.

    Dispatches callback events to all registered callbacks
    in order of registration.

    Example:
        >>> callbacks = CallbackList([
        ...     WandBCallback(project="test"),
        ...     TensorBoardCallback(log_dir="logs"),
        ... ])
        >>> callbacks.on_training_start(trainer)  # Calls both
    """

    def __init__(self, callbacks: Optional[List[BaseTianshouCallback]] = None):
        """Initialize callback list.

        Args:
            callbacks: List of callback instances.
        """
        self.callbacks = callbacks or []

    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)

    def append(self, callback: BaseTianshouCallback) -> None:
        """Add a callback.

        Args:
            callback: Callback to add.
        """
        self.callbacks.append(callback)

    def extend(self, callbacks: List[BaseTianshouCallback]) -> None:
        """Add multiple callbacks.

        Args:
            callbacks: List of callbacks to add.
        """
        self.callbacks.extend(callbacks)

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_training_start(trainer)

    def on_epoch_start(self, epoch: int, trainer: "TianshouTrainer") -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_epoch_start(epoch, trainer)

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_collect_end(collect_result, trainer)

    def on_update_end(
        self,
        epoch: int,
        update_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_update_end(epoch, update_result, trainer)

    def on_train_step_end(
        self,
        train_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_train_step_end(train_result, trainer)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, trainer)

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Dispatch to all callbacks."""
        for cb in self.callbacks:
            cb.on_training_end(trainer)
