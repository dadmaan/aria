"""Tianshou training callbacks.

This module provides callback classes for Tianshou training loops,
enabling logging, checkpointing, and real-time monitoring.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

import numpy as np
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from ..logging.logging_manager import get_logger


class BaseTianshouCallback:
    """Base class for Tianshou callbacks.

    Provides hook methods that are called during training:
    - on_training_start(): Called once before training begins
    - on_collect_end(collect_result): Called after each data collection
    - on_train_step_end(train_result): Called after each policy update
    - on_training_end(): Called once after training completes
    """

    def __init__(self, verbose: int = 0):
        """Initialize callback.

        Args:
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__)

    def on_training_start(self, adapter: Any) -> None:
        """Called once before training begins.

        Args:
            adapter: TianshouDRQNAdapter instance
        """
        pass

    def on_collect_end(self, collect_result: Any, adapter: Any) -> None:
        """Called after each data collection step.

        Args:
            collect_result: Result from collector.collect()
            adapter: TianshouDRQNAdapter instance
        """
        pass

    def on_train_step_end(self, train_result: Dict[str, Any], adapter: Any) -> None:
        """Called after each policy update.

        Args:
            train_result: Result from policy.update()
            adapter: TianshouDRQNAdapter instance
        """
        pass

    def on_training_end(self, adapter: Any) -> None:
        """Called once after training completes.

        Args:
            adapter: TianshouDRQNAdapter instance
        """
        pass


class TianshouMetricsCallback(BaseTianshouCallback):
    """Tianshou-compatible version of MetricsCallback.

    Logs episode rewards and lengths to JSON file periodically.
    """

    def __init__(self, output_dir: Path, verbose: int = 0):
        """Initialize metrics callback.

        Args:
            output_dir: Directory to save metrics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.episode_rewards = []
        self.episode_lengths = []

        # Create metrics directory
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

    def on_collect_end(self, collect_result: Any, adapter: Any) -> None:
        """Track episode rewards and lengths."""
        if collect_result.n_collected_episodes > 0:
            # Tianshou provides returns and lens directly
            self.episode_rewards.extend(collect_result.returns)
            self.episode_lengths.extend(collect_result.lens)

            # Save after every batch of collected episodes for immediate visibility
            self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics = {
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
            "total_episodes": len(self.episode_rewards),
        }

        metrics_path = self.output_dir / "metrics" / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        if self.verbose >= 1:
            self.logger.info(
                f"Saved metrics: {len(self.episode_rewards)} episodes, "
                f"avg reward: {np.mean(self.episode_rewards[-10:]):.2f}"
            )

    def on_training_end(self, adapter: Any) -> None:
        """Save final metrics at end of training."""
        self._save_metrics()
        if self.verbose >= 1:
            self.logger.info(
                f"Training complete. Final metrics: {len(self.episode_rewards)} episodes, "
                f"avg reward: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}"
            )


class TianshouCheckpointCallback(BaseTianshouCallback):
    """Tianshou-compatible version of CheckpointCallback.

    Saves model checkpoints periodically during training.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "model",
        verbose: int = 0,
    ):
        """Initialize checkpoint callback.

        Args:
            save_freq: Save frequency (in steps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint filenames
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.step_count = 0

        # Create checkpoint directory
        self.save_path.mkdir(parents=True, exist_ok=True)

    def on_train_step_end(self, train_result: Dict[str, Any], adapter: Any) -> None:
        """Save checkpoint periodically."""
        self.step_count += 1

        if self.step_count % self.save_freq == 0:
            checkpoint_path = (
                self.save_path / f"{self.name_prefix}_{self.step_count}_steps"
            )
            adapter.save(str(checkpoint_path))

            if self.verbose >= 1:
                self.logger.info(f"Saved checkpoint: {checkpoint_path}.pth")


class TianshouLearningRateSchedulerCallback(BaseTianshouCallback):
    """Tianshou-compatible version of LearningRateSchedulerCallback.

    Supports:
    - Linear and exponential learning rate decay
    - Adaptive threshold for pausing/resuming
    - Pulse mechanism for performance recovery
    """

    def __init__(
        self,
        initiallr: float,
        finallr: float,
        decaysteps: int,
        scheduletype: str = "linear",
        decayrate: Optional[float] = 0.995,
        pausethreshold: Optional[float] = None,
        resumethreshold: Optional[float] = None,
        minepisodesbeforepausing: int = 10,
        maxtotalpausesteps: Optional[int] = None,
        adaptivethresholdenabled: bool = False,
        baselinealpha: float = 0.1,
        relativedropthreshold: float = 0.15,
        verbose: int = 0,
    ):
        """Initialize LR scheduler callback.

        Args:
            initiallr: Initial learning rate
            finallr: Final learning rate
            decaysteps: Total steps for LR decay
            scheduletype: "linear" or "exponential"
            decayrate: Decay rate for exponential schedule
            pausethreshold: Threshold for pausing LR decay
            resumethreshold: Threshold for resuming LR decay
            minepisodesbeforepausing: Minimum episodes before pausing
            maxtotalpausesteps: Maximum total pause steps
            adaptivethresholdenabled: Enable adaptive threshold
            baselinealpha: EMA smoothing factor for baseline
            relativedropthreshold: Relative drop threshold for pausing
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.initiallr = initiallr
        self.finallr = finallr
        self.decaysteps = decaysteps
        self.scheduletype = scheduletype
        self.decayrate = decayrate if decayrate is not None else 0.995
        self.pausethreshold = pausethreshold
        self.resumethreshold = resumethreshold
        self.minepisodesbeforepausing = minepisodesbeforepausing
        self.maxtotalpausesteps = maxtotalpausesteps

        # Adaptive threshold settings
        self.adaptivethresholdenabled = adaptivethresholdenabled
        self.baselinealpha = baselinealpha
        self.relativedropthreshold = relativedropthreshold

        # State tracking
        self.currentstep = 0
        self.paused = False
        self.totalpausesteps = 0
        self.pauseevents = []
        self.lrhistory = []
        self.episoderewards = []
        self.episodecount = 0
        self.rewardtrendwindow = 10

        # Adaptive threshold state
        self.emabaseline: Optional[float] = None
        self.adaptivepausethreshold: Optional[float] = None
        self.adaptiveresumethreshold: Optional[float] = None

    def on_train_step_end(self, train_result: Dict[str, Any], adapter: Any) -> None:
        """Update learning rate based on schedule and rewards."""
        self.currentstep += 1

        # Track episode rewards (Tianshou provides in adapter)
        if hasattr(adapter, "_last_episode_rewards"):
            for reward in adapter._last_episode_rewards:
                self.episoderewards.append(reward)
                self.episodecount += 1

                # Maintain fixed-size reward buffer
                if len(self.episoderewards) > self.rewardtrendwindow:
                    self.episoderewards.pop(0)

        # Calculate new learning rate
        new_lr = self._compute_lr()

        # Update optimizer
        for param_group in adapter.optimizer.param_groups:
            param_group["lr"] = new_lr

        # Track LR history
        self.lrhistory.append(new_lr)

        if self.verbose >= 2 and self.currentstep % 100 == 0:
            self.logger.info(
                f"Step {self.currentstep}: LR = {new_lr:.6f}, " f"Paused: {self.paused}"
            )

    def _compute_lr(self) -> float:
        """Compute learning rate based on schedule and pausing logic."""
        # Check if we should pause/resume
        if (
            self.pausethreshold is not None
            and len(self.episoderewards) >= self.rewardtrendwindow
        ):
            avgreward = sum(self.episoderewards) / len(self.episoderewards)

            # Update adaptive threshold if enabled
            if self.adaptivethresholdenabled:
                if self.emabaseline is None:
                    self.emabaseline = avgreward
                else:
                    self.emabaseline = (
                        self.baselinealpha * avgreward
                        + (1 - self.baselinealpha) * self.emabaseline
                    )

                # Calculate adaptive thresholds
                if self.emabaseline >= 0:
                    self.adaptivepausethreshold = self.emabaseline * (
                        1 - self.relativedropthreshold
                    )
                    self.adaptiveresumethreshold = self.emabaseline * (
                        1 - self.relativedropthreshold / 2
                    )
                else:
                    self.adaptivepausethreshold = self.emabaseline * (
                        1 + self.relativedropthreshold
                    )
                    self.adaptiveresumethreshold = self.emabaseline * (
                        1 + self.relativedropthreshold / 2
                    )

                effective_pause_threshold = self.adaptivepausethreshold
                effective_resume_threshold = self.adaptiveresumethreshold
            else:
                effective_pause_threshold = self.pausethreshold
                effective_resume_threshold = self.resumethreshold

            # Pause/resume logic
            if not self.paused and avgreward < effective_pause_threshold:
                self.paused = True
                self.pauseevents.append(
                    {
                        "step": self.currentstep,
                        "action": "pause",
                        "avg_reward": avgreward,
                        "threshold": effective_pause_threshold,
                    }
                )
                if self.verbose >= 1:
                    self.logger.info(f"Paused LR decay at step {self.currentstep}")

            elif (
                self.paused
                and effective_resume_threshold is not None
                and avgreward > effective_resume_threshold
            ):
                self.paused = False
                self.pauseevents.append(
                    {
                        "step": self.currentstep,
                        "action": "resume",
                        "avg_reward": avgreward,
                        "threshold": effective_resume_threshold,
                    }
                )
                if self.verbose >= 1:
                    self.logger.info(f"Resumed LR decay at step {self.currentstep}")

        # Compute LR based on schedule
        if self.paused:
            self.totalpausesteps += 1
            # Return last LR (no decay while paused)
            return self.lrhistory[-1] if self.lrhistory else self.initiallr

        # Effective step (excluding paused steps)
        effective_step = self.currentstep - self.totalpausesteps

        if self.scheduletype == "linear":
            progress = min(effective_step / self.decaysteps, 1.0)
            new_lr = self.initiallr - (self.initiallr - self.finallr) * progress
        elif self.scheduletype == "exponential":
            new_lr = self.initiallr * (self.decayrate**effective_step)
            new_lr = max(new_lr, self.finallr)  # Clamp to final LR
        else:
            new_lr = self.initiallr

        return new_lr


class TianshouDetailedRewardCallback(BaseTianshouCallback):
    """Tianshou-compatible version of DetailedRewardCallback.

    Tracks and exports detailed reward component statistics.
    """

    def __init__(
        self,
        verbose: int = 0,
        log_frequency: int = 100,
        export_frequency: int = 500,
        export_path: Optional[str] = None,
    ):
        """Initialize detailed reward callback.

        Args:
            verbose: Verbosity level
            log_frequency: Logging frequency (steps)
            export_frequency: Export frequency (steps)
            export_path: Path to export JSON file
        """
        super().__init__(verbose)
        self.log_frequency = log_frequency
        self.export_frequency = export_frequency
        self.export_path = Path(export_path) if export_path else None

        # Tracking
        self.step_count = 0
        self.reward_history = []

        if self.export_path:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)

    def on_collect_end(self, collect_result: Any, adapter: Any) -> None:
        """Track reward components from episode info."""
        self.step_count += 1

        # Extract reward components if available
        # Note: This requires the environment to provide reward_components in info
        if hasattr(collect_result, "info") and collect_result.info:
            for info_dict in collect_result.info:
                if "reward_components" in info_dict:
                    self.reward_history.append(info_dict["reward_components"])

        # Log periodically
        if self.step_count % self.log_frequency == 0 and self.reward_history:
            self._log_reward_stats()

        # Export periodically
        if self.export_path and self.step_count % self.export_frequency == 0:
            self._export_rewards()

    def _log_reward_stats(self) -> None:
        """Log reward component statistics."""
        if not self.reward_history:
            return

        # Calculate averages for last 10 episodes
        recent = self.reward_history[-10:]
        avg_components = {}

        # Get all keys from first reward dict
        if recent and isinstance(recent[0], dict):
            for key in recent[0].keys():
                values = [r[key] for r in recent if key in r]
                avg_components[key] = np.mean(values) if values else 0.0

        if self.verbose >= 1:
            self.logger.info(f"Reward components (avg last 10): {avg_components}")

    def _export_rewards(self) -> None:
        """Export reward history to JSON."""
        if not self.export_path:
            return

        export_data = {
            "step_count": self.step_count,
            "reward_history": self.reward_history,
        }

        with open(self.export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        if self.verbose >= 1:
            self.logger.info(f"Exported reward history to {self.export_path}")


class CallbackList:
    """Container for multiple callbacks.

    Calls all callbacks' methods in sequence.
    """

    def __init__(self, callbacks: List[BaseTianshouCallback]):
        """Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks

    def on_training_start(self, adapter: Any) -> None:
        """Call on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(adapter)

    def on_collect_end(self, collect_result: Any, adapter: Any) -> None:
        """Call on_collect_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_collect_end(collect_result, adapter)

    def on_train_step_end(self, train_result: Dict[str, Any], adapter: Any) -> None:
        """Call on_train_step_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_step_end(train_result, adapter)

    def on_training_end(self, adapter: Any) -> None:
        """Call on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(adapter)


class TianshouSimpleDashboardCallback(BaseTianshouCallback):
    """Simple visual dashboard callback for Tianshou training visualization.

    Tianshou-compatible version of SimpleDashboardCallback that provides:
    - Training progress bar
    - Live metrics (rewards, epsilon, training stats)
    - Reward components breakdown
    - GHSOM hierarchy stats
    - Episode history table
    - Previous and current sequence visualization

    Uses Rich library for terminal-based real-time visualization.
    """

    def __init__(
        self,
        total_timesteps: int,
        run_id: str,
        ghsom_manager=None,
        learning_rate: float = 1e-3,
        lr_scheduler_callback=None,
        verbose: int = 0,
    ):
        """Initialize dashboard callback.

        Args:
            total_timesteps: Total training timesteps
            run_id: Run identifier
            ghsom_manager: GHSOM manager instance for hierarchy stats
            learning_rate: Initial learning rate
            lr_scheduler_callback: Learning rate scheduler callback reference
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.console = Console()
        self.layout = self._make_layout()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.completed}/{task.total} steps"),
        )
        self.task = self.progress.add_task("[green]Training...", total=total_timesteps)
        self.live = Live(
            self.layout,
            console=self.console,
            screen=True,  # Let Rich manage the entire screen
            auto_refresh=False,
            transient=True,  # Clear display when done
            refresh_per_second=4,  # Throttle redraws
            vertical_overflow="visible",
        )
        self.episode_rewards = []
        self.episode_lengths = []
        self.run_id = run_id
        self.total_timesteps = total_timesteps
        self.ghsom_manager = ghsom_manager
        self.initial_learning_rate = learning_rate
        self.lr_scheduler_callback = lr_scheduler_callback
        self.start_time = None
        self.last_reward_components = {}
        self.previous_sequence = []
        self.current_sequence = []
        self.step_count = 0

    def _make_layout(self) -> Layout:
        """Create the dashboard layout.

        Layout structure:
        ┌────────────────────────────────────────────────┐
        │ Header: Run ID       | Training Speed          │
        ├──────────────────────┬─────────────────────────┤
        │ Training Stats       │     GHSOM Hierarchy     │
        ├──────────────────────┼─────────────────────────┤
        │  Reward Breakdown    │     Recent Episodes     │
        ├──────────────────────┼─────────────────────────┤
        │    Previous Seq      │      Current Seq        │
        ├──────────────────────┴─────────────────────────┤
        │    Footer: Progress Bar with Time Remaining    │
        └────────────────────────────────────────────────┘
        """
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="top_row", size=12),
            Layout(name="middle_row", size=10),
            Layout(name="sequence_row", size=5),
            Layout(name="footer", size=5),
        )
        # Top row: Training Stats | GHSOM Hierarchy
        layout["top_row"].split_row(
            Layout(name="metrics"),
            Layout(name="ghsom_stats"),
        )
        # Middle row: Reward Breakdown | Recent Episodes
        layout["middle_row"].split_row(
            Layout(name="reward_components"),
            Layout(name="history"),
        )
        # Sequence row: Previous | Current
        layout["sequence_row"].split_row(
            Layout(name="prev_sequence"),
            Layout(name="curr_sequence"),
        )
        return layout

    def on_training_start(self, adapter: Any) -> None:
        """Called at the start of training."""
        self.start_time = time.time()
        self.live.start()

    def on_collect_end(self, collect_result: Any, adapter: Any) -> None:
        """Called after each data collection.

        Args:
            collect_result: CollectStats from collector
            adapter: TianshouDRQNAdapter instance
        """
        # Update step count
        self.step_count += collect_result.n_collected_steps
        self.progress.update(self.task, advance=collect_result.n_collected_steps)

        # Track episode data
        if collect_result.n_collected_episodes > 0:
            self.episode_rewards.extend(collect_result.returns)
            self.episode_lengths.extend(collect_result.lens)

        # Try to get sequence data from environment
        try:
            if hasattr(adapter, "train_collector") and hasattr(
                adapter.train_collector, "env"
            ):
                # Get environment from collector
                env = adapter.train_collector.env
                # Try to get current_sequence attribute
                if hasattr(env, "get_attr"):
                    # Vectorized environment
                    new_sequence = list(env.get_attr("current_sequence")[0])
                elif hasattr(env, "current_sequence"):
                    # Single environment
                    new_sequence = list(env.current_sequence)
                else:
                    new_sequence = []

                if new_sequence and new_sequence != self.current_sequence:
                    self.previous_sequence = self.current_sequence.copy()
                    self.current_sequence = new_sequence
        except Exception:
            pass

        # Try to extract reward components from info
        try:
            if hasattr(collect_result, "info") and collect_result.info:
                # Tianshou stores info dicts differently
                for info_dict in collect_result.info:
                    if isinstance(info_dict, dict) and "reward_components" in info_dict:
                        self.last_reward_components = info_dict["reward_components"]
                        break
        except Exception:
            pass

        self._update_dashboard(adapter)

    def on_train_step_end(self, train_result: Dict[str, Any], adapter: Any) -> None:
        """Called after each training step.

        Args:
            train_result: Training statistics from policy update
            adapter: TianshouDRQNAdapter instance
        """
        # Dashboard updates happen on_collect_end for performance
        pass

    def on_training_end(self, adapter: Any) -> None:
        """Called at the end of training."""
        self.live.stop()

    def _get_training_speed(self) -> str:
        """Calculate training speed (steps per second)."""
        if self.start_time is None:
            return "N/A"
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            speed = self.step_count / elapsed
            return f"{speed:.1f} steps/s"
        return "N/A"

    def _get_buffer_info(self, adapter: Any) -> str:
        """Get replay buffer size."""
        try:
            if hasattr(adapter, "buffer") and adapter.buffer is not None:
                buffer_size = len(adapter.buffer)
                buffer_max = getattr(adapter.buffer, "maxsize", "unknown")
                return f"{buffer_size}/{buffer_max}"
        except Exception:
            pass
        return "N/A"

    def _get_current_learning_rate(self, adapter: Any) -> float:
        """Get current learning rate from optimizer."""
        try:
            if hasattr(adapter, "optimizer") and adapter.optimizer is not None:
                return adapter.optimizer.param_groups[0]["lr"]
        except Exception:
            pass
        return self.initial_learning_rate

    def _get_current_epsilon(self, adapter: Any) -> float:
        """Get current exploration rate (epsilon)."""
        try:
            if hasattr(adapter, "exploration_rate"):
                return adapter.exploration_rate
            # Fallback: try to get from policy
            if hasattr(adapter, "policy") and hasattr(adapter.policy, "eps"):
                return adapter.policy.eps
        except Exception:
            pass
        return 1.0

    def _get_lr_scheduler_status(self) -> tuple:
        """Get LR scheduler pause status.

        Returns:
            Tuple of (is_paused: bool, status_text: str, style: str)
        """
        if self.lr_scheduler_callback is None:
            return (False, "", "dim")

        try:
            is_paused = getattr(self.lr_scheduler_callback, "paused", False)
            total_pause_steps = getattr(
                self.lr_scheduler_callback, "total_pause_steps", 0
            )
            pause_events = getattr(self.lr_scheduler_callback, "pause_events", [])
            pause_count = len([e for e in pause_events if e.get("action") == "pause"])

            if is_paused:
                return (
                    True,
                    f"PAUSED ({pause_count}x, {total_pause_steps} steps)",
                    "bold red",
                )
            elif pause_count > 0:
                return (False, f"Active (resumed {pause_count}x)", "green")
            else:
                return (False, "Active", "green")
        except Exception:
            return (False, "Unknown", "dim")

    def _format_sequence(self, sequence: list, title: str) -> Panel:
        """Format a sequence for display with alternating colors."""
        if sequence:
            seq_parts = []
            for i, cluster_id in enumerate(sequence[-16:]):
                color = "red" if i % 2 else "blue"
                seq_parts.append(f"[{color}]{cluster_id}[/{color}]")
            sequence_vis = " → ".join(seq_parts)
        else:
            sequence_vis = "[dim]---[/dim]"
        return Panel(sequence_vis, title=title, border_style="blue")

    def _update_dashboard(self, adapter: Any):
        """Update the dashboard with the latest metrics."""
        # Check if LR scheduler is paused
        is_paused, _, _ = self._get_lr_scheduler_status()
        pause_indicator = " | [bold red]⏸ LR PAUSED[/bold red]" if is_paused else ""

        header = Panel(
            f"[bold]Multi-Agent Music RL (Tianshou)[/bold] | Run: {self.run_id} | {self._get_training_speed()}{pause_indicator}",
            border_style="red" if is_paused else "green",
        )

        # Metrics table - core training stats
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column("Metric", style="cyan", width=14)
        metrics_table.add_column("Value", style="magenta", justify="right")

        if self.episode_rewards:
            metrics_table.add_row("Last Reward", f"{self.episode_rewards[-1]:.2f}")
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            metrics_table.add_row("Avg Reward", f"{avg_reward:.2f}")
            metrics_table.add_row("Best Reward", f"{max(self.episode_rewards):.2f}")
            metrics_table.add_row("Episodes", str(len(self.episode_rewards)))

        # Get dynamic epsilon and learning rate
        current_epsilon = self._get_current_epsilon(adapter)
        current_lr = self._get_current_learning_rate(adapter)

        metrics_table.add_row("Epsilon", f"{current_epsilon:.4f}")
        metrics_table.add_row("Learning Rate", f"{current_lr:.2e}")
        metrics_table.add_row("Buffer", self._get_buffer_info(adapter))

        # Add LR scheduler status if available
        _, lr_status, lr_style = self._get_lr_scheduler_status()
        if lr_status:
            metrics_table.add_row(
                "LR Scheduler", f"[{lr_style}]{lr_status}[/{lr_style}]"
            )

        # Reward components breakdown
        reward_table = Table(show_header=False, box=None, padding=(0, 1))
        reward_table.add_column("Component", style="green", width=14)
        reward_table.add_column("Value", style="yellow", justify="right")
        if self.last_reward_components:
            for name, value in self.last_reward_components.items():
                display_name = name.replace("_", " ").title()
                reward_table.add_row(display_name, f"{value:.3f}")
        else:
            reward_table.add_row("[dim]Waiting...", "[dim]---")

        # GHSOM hierarchy stats
        ghsom_table = Table(show_header=False, box=None, padding=(0, 1))
        ghsom_table.add_column("Stat", style="blue", width=14)
        ghsom_table.add_column("Value", style="white", justify="right")
        if self.ghsom_manager and hasattr(self.ghsom_manager, "stats"):
            stats = self.ghsom_manager.stats
            ghsom_table.add_row("Total Nodes", str(stats.get("total_nodes", "N/A")))
            levels = stats.get("levels", {})
            if isinstance(levels, dict):
                ghsom_table.add_row("Depth", str(len(levels)))
            else:
                ghsom_table.add_row("Levels", str(levels))
            ghsom_table.add_row("Max Children", str(stats.get("max_children", "N/A")))
            ghsom_table.add_row(
                "Dataset", str(stats.get("max_input_dataset_size", "N/A"))
            )
        else:
            ghsom_table.add_row("[dim]N/A", "[dim]---")

        # History table (recent episodes)
        history_table = Table(
            show_header=True, header_style="bold", box=None, padding=(0, 1)
        )
        history_table.add_column("Ep", style="cyan", justify="right", width=4)
        history_table.add_column("Reward", style="green", justify="right", width=8)
        history_table.add_column("Len", style="yellow", justify="right", width=4)

        # Show last 6 episodes
        start_idx = max(0, len(self.episode_rewards) - 6)
        for i, (reward, length) in enumerate(
            zip(self.episode_rewards[start_idx:], self.episode_lengths[start_idx:]),
            start=start_idx + 1,
        ):
            history_table.add_row(str(i), f"{reward:.2f}", str(length))

        # Sequence panels
        prev_seq_panel = self._format_sequence(
            self.previous_sequence, "Previous Sequence"
        )
        curr_seq_panel = self._format_sequence(
            self.current_sequence, "Current Sequence"
        )

        # Update layout
        self.layout["header"].update(header)
        self.layout["footer"].update(Panel(self.progress))
        self.layout["metrics"].update(
            Panel(metrics_table, title="Training Stats", border_style="cyan")
        )
        self.layout["reward_components"].update(
            Panel(reward_table, title="Reward Breakdown", border_style="green")
        )
        self.layout["ghsom_stats"].update(
            Panel(ghsom_table, title="GHSOM Hierarchy", border_style="blue")
        )
        self.layout["history"].update(
            Panel(history_table, title="Recent Episodes", border_style="yellow")
        )
        self.layout["prev_sequence"].update(prev_seq_panel)
        self.layout["curr_sequence"].update(curr_seq_panel)

        self.live.update(self.layout)
        self.live.refresh()
