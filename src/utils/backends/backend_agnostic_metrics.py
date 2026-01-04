"""Backend-agnostic metrics interface for standardized logging.

This module provides a common metrics interface that works across both
SB3 and TF-Agents backends, reducing code duplication and ensuring
consistent metric collection regardless of the backend used.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np

from src.utils.logging.logging_manager import get_logger

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    try:
        from torch.utils.tensorboard.writer import SummaryWriter

        HAS_TENSORBOARD = True
    except ImportError:
        from tensorboardX import SummaryWriter  # type: ignore

        HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None


@dataclass
class MetricSnapshot:
    """Immutable snapshot of metrics at a point in time."""

    timestamp: float
    step: int
    episode: int

    # Core reward metrics
    total_reward: Optional[float] = None
    similarity_reward: Optional[float] = None
    structure_reward: Optional[float] = None
    human_reward: Optional[float] = None

    # Episode metrics
    episode_length: Optional[int] = None
    episode_done: bool = False

    # Training metrics
    loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Custom metrics
    custom_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        if self.custom_metrics:
            # Flatten custom metrics into main dict
            data.update(self.custom_metrics)
            del data["custom_metrics"]
        return data


class MetricsBuffer:
    """Thread-safe buffer for collecting metrics."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.snapshots = deque(maxlen=max_size)
        self.episode_snapshots = deque(
            maxlen=max_size // 10
        )  # Keep fewer episode-level metrics

    def add_snapshot(self, snapshot: MetricSnapshot):
        """Add a metric snapshot."""
        self.snapshots.append(snapshot)
        if snapshot.episode_done:
            self.episode_snapshots.append(snapshot)

    def get_recent_snapshots(self, n: int = 100) -> List[MetricSnapshot]:
        """Get the most recent N snapshots."""
        return list(self.snapshots)[-n:]

    def get_recent_episodes(self, n: int = 10) -> List[MetricSnapshot]:
        """Get the most recent N episode snapshots."""
        return list(self.episode_snapshots)[-n:]

    def get_stats(self, metric_name: str, n_recent: int = 100) -> Dict[str, float]:
        """Get statistics for a specific metric."""
        recent = self.get_recent_snapshots(n_recent)
        values = []

        for snapshot in recent:
            value = getattr(snapshot, metric_name, None)
            if value is not None:
                values.append(value)

        if not values:
            return {"count": 0}

        return {
            "count": float(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "latest": float(values[-1]) if values else 0.0,
        }


class BackendAgnosticLogger:
    """Backend-agnostic logging interface."""

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        enable_wandb: bool = False,
        enable_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize logger with optional external integrations."""
        self.logger = get_logger("backend_agnostic_metrics")
        self.log_dir = Path(log_dir) if log_dir else None
        self.enable_wandb = enable_wandb and HAS_WANDB
        self.enable_tensorboard = enable_tensorboard and HAS_TENSORBOARD

        # Initialize external loggers
        self.wandb_run = None
        self.tensorboard_writer = None

        if self.enable_wandb:
            self._init_wandb(wandb_project, wandb_config)

        if self.enable_tensorboard and self.log_dir:
            self._init_tensorboard()

    def _init_wandb(self, project: Optional[str], config: Optional[Dict[str, Any]]):
        """Initialize Weights & Biases."""
        try:
            # Check if a WandB run is already active - reuse it to prevent empty runs
            if wandb.run is not None:
                self.wandb_run = wandb.run
                self.logger.info(f"Reusing existing WandB run: {wandb.run.name}")
                return

            self.wandb_run = wandb.init(
                project=project or "music_generation", config=config
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize wandb: {e}")
            self.enable_wandb = False

    def _init_tensorboard(self):
        """Initialize TensorBoard."""
        if not self.log_dir or not SummaryWriter:
            self.enable_tensorboard = False
            return

        try:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tb_dir))
        except Exception as e:
            self.logger.error(f"Failed to initialize tensorboard: {e}")
            self.enable_tensorboard = False

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value."""
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.log({name: value}, step=step)

        if self.enable_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(name, value, step)

    def log_dict(self, metrics: Dict[str, float], step: int):
        """Log a dictionary of metrics."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not (
                np.isnan(value) or np.isinf(value)
            ):
                self.log_scalar(name, float(value), step)

    def log_snapshot(self, snapshot: MetricSnapshot):
        """Log a complete metric snapshot."""
        metrics_dict = snapshot.to_dict()
        # Remove non-numeric fields
        numeric_metrics: Dict[str, float] = {
            k: float(v)
            for k, v in metrics_dict.items()
            if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v))
        }
        self.log_dict(numeric_metrics, snapshot.step)

    def close(self):
        """Close all loggers."""
        if self.wandb_run:
            self.wandb_run.finish()

        if self.tensorboard_writer:
            self.tensorboard_writer.close()


class UnifiedMetricsCollector:
    """Main metrics collection interface that works with both backends."""

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        enable_wandb: bool = False,
        enable_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        buffer_size: int = 10000,
    ):
        """Initialize unified metrics collector."""
        self.buffer = MetricsBuffer(buffer_size)
        self.logger = BackendAgnosticLogger(
            log_dir=log_dir,
            enable_wandb=enable_wandb,
            enable_tensorboard=enable_tensorboard,
            wandb_project=wandb_project,
        )

        self.step_count = 0
        self.episode_count = 0
        self.start_time = time.time()

        # JSON log file for local storage
        self.json_log_path = None
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.json_log_path = log_path / "metrics.jsonl"

    def record_step(
        self,
        total_reward: Optional[float] = None,
        reward_components: Optional[Dict[str, float]] = None,
        episode_done: bool = False,
        episode_length: Optional[int] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> MetricSnapshot:
        """Record metrics for a single step."""
        self.step_count += 1
        if episode_done:
            self.episode_count += 1

        # Extract reward components
        similarity_reward = None
        structure_reward = None
        human_reward = None

        if reward_components:
            similarity_reward = reward_components.get(
                "similarity", reward_components.get("w1")
            )
            structure_reward = reward_components.get(
                "structure", reward_components.get("w2")
            )
            human_reward = reward_components.get("human", reward_components.get("w3"))

        # Create snapshot
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            step=self.step_count,
            episode=self.episode_count,
            total_reward=total_reward,
            similarity_reward=similarity_reward,
            structure_reward=structure_reward,
            human_reward=human_reward,
            episode_length=episode_length,
            episode_done=episode_done,
            custom_metrics=custom_metrics,
        )

        # Store and log
        self.buffer.add_snapshot(snapshot)
        self.logger.log_snapshot(snapshot)

        # Append to JSON log
        if self.json_log_path:
            self._append_json_log(snapshot)

        return snapshot

    def record_training_metrics(
        self,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None,
    ):
        """Record training-specific metrics."""
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            step=self.step_count,
            episode=self.episode_count,
            loss=loss,
            learning_rate=learning_rate,
            custom_metrics=custom_metrics,
        )

        self.buffer.add_snapshot(snapshot)
        self.logger.log_snapshot(snapshot)

        if self.json_log_path:
            self._append_json_log(snapshot)

    def get_summary_stats(self, n_recent: int = 100) -> Dict[str, Any]:
        """Get summary statistics for recent metrics."""
        stats = {}

        # Get stats for core metrics
        for metric in [
            "total_reward",
            "similarity_reward",
            "structure_reward",
            "human_reward",
        ]:
            metric_stats = self.buffer.get_stats(metric, n_recent)
            if metric_stats["count"] > 0:
                stats[metric] = metric_stats

        # Episode-level stats
        recent_episodes = self.buffer.get_recent_episodes(min(n_recent // 10, 20))
        if recent_episodes:
            episode_rewards = [
                ep.total_reward for ep in recent_episodes if ep.total_reward is not None
            ]
            if episode_rewards:
                stats["episode_rewards"] = {
                    "count": len(episode_rewards),
                    "mean": np.mean(episode_rewards),
                    "std": np.std(episode_rewards),
                    "min": np.min(episode_rewards),
                    "max": np.max(episode_rewards),
                }

        # Overall stats
        stats["total_steps"] = self.step_count
        stats["total_episodes"] = self.episode_count
        stats["runtime_seconds"] = time.time() - self.start_time

        return stats

    def _append_json_log(self, snapshot: MetricSnapshot):
        """Append snapshot to JSON lines log file."""
        if not self.json_log_path:
            return

        try:
            with open(self.json_log_path, "a") as f:
                json.dump(snapshot.to_dict(), f)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")

    def export_summary(self, filepath: Union[str, Path]):
        """Export summary statistics to file."""
        summary = self.get_summary_stats()
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self):
        """Close all loggers and save final summary."""
        self.logger.close()

        if self.json_log_path:
            summary_path = self.json_log_path.parent / "final_summary.json"
            try:
                self.export_summary(summary_path)
            except Exception as e:
                self.logger.error(f"Failed to export final summary: {e}")


def create_metrics_collector(
    config: Dict[str, Any], log_dir: Optional[str] = None
) -> UnifiedMetricsCollector:
    """Factory function to create metrics collector from configuration."""
    return UnifiedMetricsCollector(
        log_dir=log_dir or config.get("log_dir"),
        enable_wandb=config.get("enable_wandb", False),
        enable_tensorboard=config.get("enable_tensorboard", True),
        wandb_project=config.get("wandb_project_name"),
        buffer_size=config.get("metrics_buffer_size", 10000),
    )


class MetricsCallbackMixin:
    """Mixin class to add metrics collection to existing callbacks."""

    def __init__(
        self,
        *args,
        metrics_collector: Optional[UnifiedMetricsCollector] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector

    def _record_metrics(self, **kwargs):
        """Record metrics if collector is available."""
        if self.metrics_collector:
            self.metrics_collector.record_step(**kwargs)


# Integration helpers for existing callback systems


def enhance_sb3_callback_with_metrics(callback_class):
    """Decorator to enhance SB3 callbacks with metrics collection."""

    class MetricsEnhancedCallback(callback_class, MetricsCallbackMixin):
        def __init__(
            self,
            *args,
            metrics_collector: Optional[UnifiedMetricsCollector] = None,
            **kwargs,
        ):
            MetricsCallbackMixin.__init__(self, metrics_collector=metrics_collector)
            callback_class.__init__(self, *args, **kwargs)

        def _on_step(self) -> bool:
            # Call original callback
            result = super()._on_step()

            # Extract metrics from SB3 locals
            if hasattr(self, "locals"):
                rewards = self.locals.get("rewards", [])
                infos = self.locals.get("infos", [{}])
                dones = self.locals.get("dones", [False])

                for reward, info, done in zip(rewards, infos, dones):
                    reward_components = info.get("reward_components", {})
                    episode_length = info.get(
                        "sequence_length", info.get("episode_length")
                    )

                    self._record_metrics(
                        total_reward=float(reward),
                        reward_components=reward_components,
                        episode_done=bool(done),
                        episode_length=episode_length,
                    )

            return result

    return MetricsEnhancedCallback


def enhance_tfa_observer_with_metrics(observer_class):
    """Decorator to enhance TF-Agents observers with metrics collection."""

    class MetricsEnhancedObserver(observer_class, MetricsCallbackMixin):
        def __init__(
            self,
            *args,
            metrics_collector: Optional[UnifiedMetricsCollector] = None,
            **kwargs,
        ):
            MetricsCallbackMixin.__init__(self, metrics_collector=metrics_collector)
            observer_class.__init__(self, *args, **kwargs)

        def __call__(self, trajectory):
            # Call original observer
            result = super().__call__(trajectory)

            # Extract metrics from TF-Agents trajectory
            reward = float(trajectory.reward.numpy())
            done = trajectory.is_last()

            # Try to extract reward components (would need to be added to trajectory)
            reward_components = {}
            if hasattr(trajectory, "observation") and hasattr(
                trajectory.observation, "reward_components"
            ):
                reward_components = trajectory.observation.reward_components

            self._record_metrics(
                total_reward=reward,
                reward_components=reward_components,
                episode_done=done,
            )

            return result

    return MetricsEnhancedObserver
