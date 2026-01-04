"""Checkpoint callback for Tianshou training.

This module provides automatic checkpoint saving during training,
with support for best model tracking and checkpoint rotation.

Classes:
    TianshouCheckpointCallback: Automatic checkpoint saving callback.

Example:
    >>> callback = TianshouCheckpointCallback(
    ...     checkpoint_dir="checkpoints",
    ...     save_freq=1000,
    ...     save_best=True,
    ... )
    >>> trainer.train(callbacks=[callback])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import BaseTianshouCallback

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class TianshouCheckpointCallback(BaseTianshouCallback):
    """Checkpoint callback for saving training progress.

    Saves model checkpoints at regular intervals and optionally
    tracks the best model based on evaluation reward.

    Features:
        - Periodic checkpoint saving
        - Best model tracking
        - Checkpoint rotation (keep N most recent)
        - Configurable save frequency

    Attributes:
        checkpoint_dir (Path): Directory for checkpoints.
        save_freq (int): Save every N steps.
        save_best (bool): Whether to save best model.
        keep_last_n (int): Number of recent checkpoints to keep.

    Example:
        >>> callback = TianshouCheckpointCallback(
        ...     checkpoint_dir="artifacts/checkpoints",
        ...     save_freq=5000,
        ...     save_best=True,
        ...     keep_last_n=3,
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 10000,
        save_best: bool = True,
        keep_last_n: int = 5,
        verbose: int = 0,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory for saving checkpoints.
            save_freq: Save checkpoint every N steps.
            save_best: Whether to save best model based on reward.
            keep_last_n: Number of recent checkpoints to keep.
            verbose: Verbosity level.
        """
        super().__init__(verbose)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_freq = save_freq
        self.save_best = save_best
        self.keep_last_n = keep_last_n

        # State tracking
        self.step_count = 0
        self.best_reward = float("-inf")
        self.saved_checkpoints: List[Path] = []

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Create checkpoint directory.

        Args:
            trainer: TianshouTrainer instance.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

    def on_update_end(
        self,
        epoch: int,
        update_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Check if checkpoint should be saved.

        Args:
            epoch: Current epoch.
            update_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        self.step_count += 1

        # Save at intervals
        if self.step_count % self.save_freq == 0:
            checkpoint_path = self.checkpoint_dir / f"step_{self.step_count}"
            trainer.save(checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path.with_suffix(".pth"))

            if self.verbose >= 1:
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Rotate old checkpoints
            self._rotate_checkpoints()

    def on_train_step_end(
        self,
        train_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Compatibility method - calls on_update_end.

        Args:
            train_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        self.on_update_end(0, train_result, trainer)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Check if best model should be saved.

        Args:
            epoch: Current epoch number.
            metrics: Tianshou trainer metrics.
            trainer: TianshouTrainer instance.
        """
        if not self.save_best:
            return

        # Get current reward
        current_reward = metrics.get(
            "test_reward", metrics.get("train_reward", float("-inf"))
        )

        # Save if better
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            best_path = self.checkpoint_dir / "best"
            trainer.save(best_path)

            if self.verbose >= 1:
                self.logger.info(
                    f"New best model! Reward: {current_reward:.3f} -> {best_path}"
                )

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Save final checkpoint.

        Args:
            trainer: TianshouTrainer instance.
        """
        final_path = self.checkpoint_dir / "final"
        trainer.save(final_path)
        self.logger.info(f"Final checkpoint saved: {final_path}")

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to maintain keep_last_n."""
        while len(self.saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                if self.verbose >= 2:
                    self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")
