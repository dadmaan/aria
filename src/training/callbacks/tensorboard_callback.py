"""TensorBoard callback for Tianshou training.

This module provides TensorBoard integration for local visualization
of training metrics, including scalars, histograms, and graphs.

Classes:
    TianshouTensorBoardCallback: TensorBoard logging callback.

Example:
    >>> callback = TianshouTensorBoardCallback(
    ...     log_dir="runs/experiment-1",
    ...     log_histograms=True,
    ... )
    >>> trainer.train(callbacks=[callback])
    >>> # View with: tensorboard --logdir runs/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .base import BaseTianshouCallback

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class TianshouTensorBoardCallback(BaseTianshouCallback):
    """TensorBoard callback for local experiment visualization.

    Logs training metrics to TensorBoard for visualization with
    the TensorBoard UI. Supports scalars, histograms, and graphs.

    Features:
        - Scalar metrics (rewards, loss, learning rate)
        - Optional weight/gradient histograms
        - Hyperparameter logging
        - Graph visualization

    Attributes:
        log_dir (Path): TensorBoard log directory.
        writer (SummaryWriter): TensorBoard writer.
        step_count (int): Global step counter.

    Example:
        >>> callback = TianshouTensorBoardCallback(
        ...     log_dir="runs/drqn-baseline",
        ...     log_histograms=True,
        ... )
        >>> trainer.train(callbacks=[callback])
        >>> # Launch TensorBoard: tensorboard --logdir runs/
    """

    def __init__(
        self,
        log_dir: str,
        log_interval: int = 100,
        log_histograms: bool = False,
        histogram_interval: int = 1000,
        verbose: int = 0,
    ):
        """Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard event files.
            log_interval: Log scalar metrics every N steps.
            log_histograms: Whether to log weight/gradient histograms.
            histogram_interval: Log histograms every N steps.
            verbose: Verbosity level.

        Raises:
            ImportError: If tensorboard is not installed.
        """
        super().__init__(verbose)

        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "tensorboard is required. " "Install with: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.log_histograms = log_histograms
        self.histogram_interval = histogram_interval

        # State tracking
        self.writer: Optional[Any] = None
        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Initialize TensorBoard writer.

        Args:
            trainer: TianshouTrainer instance.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Log hyperparameters as text
        config_text = self._config_to_text(trainer.config)
        self.writer.add_text("hyperparameters", config_text, 0)

        # Log network architecture
        network_type = trainer.config.get("network", {}).get("type", "unknown")
        self.writer.add_text(
            "network",
            f"Type: {network_type}\n"
            f"Recurrent: {trainer.network.is_recurrent}\n"
            f"Parameters: {sum(p.numel() for p in trainer.network.parameters()):,}",
            0,
        )

        self.logger.info(f"TensorBoard logging to: {self.log_dir}")

    def _config_to_text(self, config: Dict[str, Any], indent: int = 0) -> str:
        """Convert config dict to formatted text.

        Args:
            config: Configuration dictionary.
            indent: Indentation level.

        Returns:
            Formatted text representation.
        """
        lines = []
        prefix = "  " * indent
        for key, value in config.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._config_to_text(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Track collection metrics.

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        if collect_result.n_collected_episodes > 0:
            self.episode_rewards.extend(collect_result.returns)
            self.episode_lengths.extend(collect_result.lens)
            self.episode_count += collect_result.n_collected_episodes

    def on_update_end(
        self,
        epoch: int,
        update_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Log training step metrics.

        Args:
            epoch: Current epoch.
            update_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        self.step_count += 1

        # Track loss
        if hasattr(update_result, "loss"):
            self.losses.append(float(update_result.loss))
        elif isinstance(update_result, dict) and "loss" in update_result:
            self.losses.append(float(update_result["loss"]))

        # Log scalars at intervals
        if self.step_count % self.log_interval == 0:
            self._log_scalars(trainer)

        # Log histograms at intervals
        if self.log_histograms and self.step_count % self.histogram_interval == 0:
            self._log_histograms(trainer)

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
        """Log epoch-level metrics.

        Args:
            epoch: Current epoch number.
            metrics: Tianshou trainer metrics.
            trainer: TianshouTrainer instance.
        """
        if self.writer is None:
            return

        step = self.step_count

        # Log Tianshou's native metrics
        if "train_reward" in metrics:
            self.writer.add_scalar("epoch/train_reward", metrics["train_reward"], epoch)
        if "test_reward" in metrics:
            self.writer.add_scalar("epoch/test_reward", metrics["test_reward"], epoch)
        if "loss" in metrics:
            self.writer.add_scalar("epoch/loss", metrics["loss"], epoch)

    def _log_scalars(self, trainer: "TianshouTrainer") -> None:
        """Log scalar metrics.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.writer is None:
            return

        step = self.step_count

        # Episode metrics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            self.writer.add_scalar(
                "train/episode_reward_mean",
                np.mean(recent_rewards),
                step,
            )
            self.writer.add_scalar(
                "train/episode_reward_max",
                np.max(recent_rewards),
                step,
            )

        if self.episode_lengths:
            self.writer.add_scalar(
                "train/episode_length_mean",
                np.mean(self.episode_lengths[-100:]),
                step,
            )

        # Loss
        if self.losses:
            self.writer.add_scalar(
                "train/loss",
                np.mean(self.losses[-100:]),
                step,
            )

        # Exploration rate
        self.writer.add_scalar("train/epsilon", trainer.exploration_rate, step)

        # Learning rate
        if hasattr(trainer, "optimizer"):
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/learning_rate", current_lr, step)

        # Buffer size
        if hasattr(trainer, "buffer"):
            self.writer.add_scalar("train/buffer_size", len(trainer.buffer), step)

        # Episode count
        self.writer.add_scalar("train/episodes", self.episode_count, step)

    def _log_histograms(self, trainer: "TianshouTrainer") -> None:
        """Log weight and gradient histograms.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.writer is None or not hasattr(trainer, "network"):
            return

        for name, param in trainer.network.named_parameters():
            # Weights
            self.writer.add_histogram(
                f"weights/{name}",
                param.data.cpu().numpy(),
                self.step_count,
            )

            # Gradients
            if param.grad is not None:
                self.writer.add_histogram(
                    f"gradients/{name}",
                    param.grad.cpu().numpy(),
                    self.step_count,
                )

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Close TensorBoard writer.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.writer is None:
            return

        # Log final summary
        if self.episode_rewards:
            self.writer.add_scalar(
                "summary/final_reward_mean",
                np.mean(self.episode_rewards[-100:]),
                self.step_count,
            )
            self.writer.add_scalar(
                "summary/best_reward",
                np.max(self.episode_rewards),
                self.step_count,
            )

        self.writer.add_scalar(
            "summary/total_episodes",
            self.episode_count,
            self.step_count,
        )
        self.writer.add_scalar(
            "summary/total_steps",
            self.step_count,
            self.step_count,
        )

        self.writer.close()
        self.logger.info("TensorBoard writer closed")
