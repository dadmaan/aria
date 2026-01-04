"""Weights & Biases callback for Tianshou training.

This module provides WandB integration for experiment tracking,
including metrics logging, model checkpoints, and config tracking.

Classes:
    TianshouWandBCallback: WandB logging callback.

Example:
    >>> callback = TianshouWandBCallback(
    ...     project="music-rl",
    ...     name="drqn-experiment",
    ...     config=training_config,
    ... )
    >>> trainer.train(callbacks=[callback])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .base import BaseTianshouCallback

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class TianshouWandBCallback(BaseTianshouCallback):
    """Weights & Biases callback for experiment tracking.

    Logs training metrics to WandB for visualization and comparison.
    Supports automatic network type tagging for DRQN vs MLP experiments.

    Features:
        - Real-time metric logging (rewards, loss, epsilon)
        - Config tracking and comparison
        - Model artifact logging (optional)
        - Automatic tagging by network type
        - Episode-level and step-level metrics

    Attributes:
        project (str): WandB project name.
        run: WandB run instance.
        step_count (int): Global step counter.
        episode_rewards (list): Episode reward history.

    Example:
        >>> callback = TianshouWandBCallback(
        ...     project="ARIA_rl_V03",
        ...     name="drqn-baseline",
        ...     tags=["baseline", "drqn"],
        ... )
        >>> trainer = TianshouTrainer(env, config, callbacks=[callback])
        >>> trainer.train()
        >>> # View at https://wandb.ai/<entity>/<project>/runs/<run_id>
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        log_interval: int = 100,
        log_model: bool = False,
        verbose: int = 0,
    ):
        """Initialize WandB callback.

        Args:
            project: WandB project name.
            name: Run name. Auto-generated if None.
            entity: WandB entity (team or username).
            config: Configuration dict to log.
            tags: List of tags for the run.
            notes: Notes/description for the run.
            log_interval: Log metrics every N steps.
            log_model: Whether to log model checkpoints as artifacts.
            verbose: Verbosity level.

        Raises:
            ImportError: If wandb is not installed.
        """
        super().__init__(verbose)

        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required for WandB logging. "
                "Install with: pip install wandb"
            )

        self.project = project
        self.name = name
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.log_interval = log_interval
        self.log_model = log_model

        # State tracking
        self.run = None
        self.step_count = 0
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self._initialized = False

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Initialize WandB run.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self._initialized:
            return

        # Add network type to tags
        network_type = trainer.config.get("network", {}).get("type", "unknown")
        all_tags = self.tags + [network_type]

        # Merge trainer config with provided config
        full_config = {**self.config, **trainer.config}

        # Initialize WandB run
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            config=full_config,
            tags=all_tags,
            notes=self.notes,
            reinit=True,
        )

        # Log network architecture info
        wandb.config.update(
            {
                "network_type": network_type,
                "is_recurrent": trainer.network.is_recurrent,
                "total_params": sum(p.numel() for p in trainer.network.parameters()),
            }
        )

        self._initialized = True
        self.logger.info(f"WandB run initialized: {self.run.url}")

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

        # Log at intervals
        if self.step_count % self.log_interval == 0:
            self._log_step_metrics(trainer)

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
        if not self._initialized or self.run is None:
            return

        # Log Tianshou's native metrics
        log_data = {
            "epoch": epoch,
            "train/step": self.step_count,
            "train/episode": self.episode_count,
        }

        # Map Tianshou metrics to WandB format
        metric_mapping = {
            "train_reward": "train/episode_reward",
            "train_reward_std": "train/episode_reward_std",
            "train_length": "train/episode_length",
            "test_reward": "eval/episode_reward",
            "test_reward_std": "eval/episode_reward_std",
            "test_length": "eval/episode_length",
            "loss": "train/loss",
        }

        for src_key, dst_key in metric_mapping.items():
            if src_key in metrics:
                log_data[dst_key] = metrics[src_key]

        # Add exploration rate
        log_data["train/epsilon"] = trainer.exploration_rate

        # Add learning rate
        if hasattr(trainer, "optimizer"):
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            log_data["train/learning_rate"] = current_lr

        # Add buffer size
        if hasattr(trainer, "buffer"):
            log_data["train/buffer_size"] = len(trainer.buffer)

        wandb.log(log_data, step=self.step_count)

        if self.verbose >= 1:
            self.logger.info(
                f"Epoch {epoch}: "
                f"reward={metrics.get('train_reward', 0):.2f}, "
                f"loss={metrics.get('loss', 0):.4f}"
            )

    def _log_step_metrics(self, trainer: "TianshouTrainer") -> None:
        """Log step-level metrics.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not self._initialized or self.run is None:
            return

        metrics = {"train/step": self.step_count}

        # Episode metrics (rolling average)
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-100:]
            metrics.update(
                {
                    "train/episode_reward_mean": np.mean(recent_rewards),
                    "train/episode_reward_max": np.max(recent_rewards),
                    "train/episode_reward_min": np.min(recent_rewards),
                }
            )

        if self.episode_lengths:
            metrics["train/episode_length_mean"] = np.mean(self.episode_lengths[-100:])

        # Loss (rolling average)
        if self.losses:
            metrics["train/loss"] = np.mean(self.losses[-100:])

        # Exploration rate
        metrics["train/epsilon"] = trainer.exploration_rate

        wandb.log(metrics, step=self.step_count)

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Finalize WandB run.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.run is None:
            return

        # Log final summary
        wandb.summary["total_episodes"] = self.episode_count
        wandb.summary["total_steps"] = self.step_count

        if self.episode_rewards:
            wandb.summary["final_reward_mean"] = np.mean(self.episode_rewards[-100:])
            wandb.summary["best_reward"] = np.max(self.episode_rewards)

        # Log model artifact if requested
        if self.log_model and hasattr(trainer, "_get_checkpoint_path"):
            checkpoint_path = trainer._get_checkpoint_path("final")
            trainer.save(checkpoint_path)

            artifact = wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path.with_suffix(".pth")))
            self.run.log_artifact(artifact)

        wandb.finish()
        self.logger.info("WandB run finished")
