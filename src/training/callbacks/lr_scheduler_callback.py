"""Learning Rate Scheduler Callback for Tianshou training.

This module provides a learning rate scheduling callback adapted from the
SB3 legacy implementation. It supports linear and exponential decay with
optional pulse mechanism for performance stagnation recovery.

Classes:
    TianshouLRSchedulerCallback: Main callback for dynamic LR scheduling.

Example:
    >>> from src.training.callbacks import TianshouLRSchedulerCallback
    >>>
    >>> callback = TianshouLRSchedulerCallback(
    ...     initial_lr=0.001,
    ...     final_lr=0.0001,
    ...     decay_steps=5000,
    ...     schedule_type="exponential",
    ... )
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import BaseTianshouCallback

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class TianshouLRSchedulerCallback(BaseTianshouCallback):
    """Callback for dynamic learning rate scheduling during Tianshou training.

    Supports linear and exponential decay from initial to final LR over decay_steps.
    Includes optional pulse mechanism for time-limited intervention when performance
    stagnates, with adaptive threshold based on EMA baseline.

    Attributes:
        initial_lr: Starting learning rate.
        final_lr: Target final learning rate.
        decay_steps: Number of steps over which to decay.
        schedule_type: "linear" or "exponential" decay.
        current_step: Current training step counter.

    Example:
        >>> callback = TianshouLRSchedulerCallback(
        ...     initial_lr=0.001,
        ...     final_lr=0.0001,
        ...     decay_steps=5000,
        ...     schedule_type="exponential",
        ...     pulse_enabled=True,
        ... )
    """

    def __init__(
        self,
        initial_lr: float = 0.001,
        final_lr: float = 0.0001,
        decay_steps: int = 5000,
        schedule_type: str = "exponential",
        decay_rate: float = 0.995,
        # Pulse mechanism parameters
        pulse_enabled: bool = False,
        pulse_duration_episodes: int = 50,
        pulse_boost_epsilon: Optional[float] = None,
        trigger_mode: str = "adaptive",
        # Static trigger
        trigger_threshold: Optional[float] = None,
        # Adaptive trigger
        adaptive_enabled: bool = True,
        baseline_alpha: float = 0.1,
        relative_drop_threshold: float = 0.15,
        # Logging
        log_to_wandb: bool = True,
        log_to_tensorboard: bool = True,
        log_to_local: bool = True,
        output_dir: str = "./artifacts/training",
        log_frequency: int = 100,
        # Minimum episodes before pulse can trigger
        min_episodes_before_pulse: int = 10,
        reward_trend_window: int = 10,
        verbose: int = 0,
    ):
        """Initialize learning rate scheduler callback.

        Args:
            initial_lr: Starting learning rate.
            final_lr: Target final learning rate.
            decay_steps: Number of steps over which to decay.
            schedule_type: "linear" or "exponential".
            decay_rate: Decay rate for exponential (legacy, auto-calculated).
            pulse_enabled: Enable pulse mechanism for stagnation recovery.
            pulse_duration_episodes: Duration of pulse in episodes.
            pulse_boost_epsilon: Optional epsilon boost during pulse.
            trigger_mode: "static" or "adaptive" threshold mode.
            trigger_threshold: Static threshold for pulse trigger.
            adaptive_enabled: Enable adaptive EMA-based threshold.
            baseline_alpha: EMA smoothing factor (lower = slower).
            relative_drop_threshold: Fraction below baseline to trigger.
            log_to_wandb: Log to Weights & Biases.
            log_to_tensorboard: Log to TensorBoard.
            log_to_local: Log to local JSON files.
            output_dir: Directory for local file output.
            log_frequency: Steps between logging updates.
            min_episodes_before_pulse: Minimum episodes before pulse can trigger.
            reward_trend_window: Window size for reward averaging.
            verbose: Verbosity level.
        """
        super().__init__(verbose=verbose)

        # Core LR parameters
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_steps = decay_steps
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate

        # Pulse mechanism
        self.pulse_enabled = pulse_enabled
        self.pulse_duration_episodes = pulse_duration_episodes
        self.pulse_boost_epsilon = pulse_boost_epsilon
        self.trigger_mode = trigger_mode
        self.trigger_threshold = trigger_threshold

        # Adaptive threshold
        self.adaptive_enabled = adaptive_enabled
        self.baseline_alpha = baseline_alpha
        self.relative_drop_threshold = relative_drop_threshold

        # Logging config
        self.log_to_wandb = log_to_wandb
        self.log_to_tensorboard = log_to_tensorboard
        self.log_to_local = log_to_local
        self.output_dir = output_dir
        self.log_frequency = log_frequency

        # Pulse trigger requirements
        self.min_episodes_before_pulse = min_episodes_before_pulse
        self.reward_trend_window = reward_trend_window
        self.trend_threshold = -0.05

        # State tracking
        self.current_step = 0
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.lr_history: List[tuple] = []
        self.pulse_events: List[Dict[str, Any]] = []

        # Adaptive state
        self.ema_baseline: Optional[float] = None
        self.adaptive_pause_threshold: Optional[float] = None
        self.adaptive_resume_threshold: Optional[float] = None

        # Pulse state
        self.is_pulsing = False
        self.pulse_episodes_remaining = 0
        self.pulse_start_step: Optional[int] = None
        self.frozen_lr: Optional[float] = None
        self.original_epsilon: Optional[float] = None

        # Logging backends (set in on_training_start)
        self._tb_writer = None
        self._trainer: Optional["TianshouTrainer"] = None

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Initialize scheduler at training start.

        Args:
            trainer: TianshouTrainer instance.
        """
        self._trainer = trainer

        # Setup output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup TensorBoard writer if enabled
        if self.log_to_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = os.path.join(self.output_dir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                # Check if there's an existing writer we can reuse
                if hasattr(trainer, "tb_writer") and trainer.tb_writer is not None:
                    self._tb_writer = trainer.tb_writer
                else:
                    self._tb_writer = SummaryWriter(log_dir=tb_dir)
            except ImportError:
                self.logger.warning(
                    "TensorBoard not available, disabling TensorBoard logging"
                )
                self.log_to_tensorboard = False

        self.logger.info(
            f"LR Scheduler initialized: {self.schedule_type} decay "
            f"{self.initial_lr} -> {self.final_lr} over {self.decay_steps} steps"
        )
        if self.pulse_enabled:
            self.logger.info(
                f"  Pulse mechanism: enabled (mode={self.trigger_mode}, "
                f"duration={self.pulse_duration_episodes} episodes)"
            )

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Track episode completions for pulse mechanism.

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        if not self.pulse_enabled:
            return

        # Extract episode info from collect_result
        n_episode = getattr(collect_result, "n_collected_episodes", 0)
        episode_returns = getattr(collect_result, "returns", [])

        if n_episode > 0 and len(episode_returns) > 0:
            for reward in episode_returns:
                self.episode_rewards.append(float(reward))
                self.episode_count += 1

                # Maintain fixed-size window
                if len(self.episode_rewards) > self.reward_trend_window:
                    self.episode_rewards.pop(0)

            # Handle pulse countdown on episode completion
            if self.is_pulsing:
                self.pulse_episodes_remaining -= n_episode
                if self.pulse_episodes_remaining <= 0:
                    self._end_pulse(trainer)

    def on_train_step_end(
        self,
        train_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Update learning rate after each training step.

        Args:
            train_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        self.current_step += 1

        # Check pulse trigger conditions
        if self.pulse_enabled and not self.is_pulsing:
            self._check_pulse_trigger(trainer)

        # Update learning rate if not pulsing
        if not self.is_pulsing:
            self._update_lr(trainer)
        else:
            # Log frozen LR during pulse
            if self.frozen_lr is not None:
                self.lr_history.append((self.current_step, self.frozen_lr))

        # Periodic logging
        if self.current_step % self.log_frequency == 0:
            self._log_metrics(trainer)

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Save scheduler history and cleanup.

        Args:
            trainer: TianshouTrainer instance.
        """
        # End any active pulse
        if self.is_pulsing:
            self._end_pulse(trainer)

        # Export history
        self._export_history()

        # Print summary
        pulse_count = len(
            [e for e in self.pulse_events if e.get("action") == "pulse_start"]
        )
        if self.pulse_enabled:
            self.logger.info(f"LR Scheduler Summary: {pulse_count} pulses triggered")
        else:
            self.logger.info(
                f"LR Scheduler Summary: {self.schedule_type} decay completed, "
                f"final LR: {self._get_current_lr(trainer):.6f}"
            )

        # Close TensorBoard writer if we created it
        if self._tb_writer is not None and not hasattr(trainer, "tb_writer"):
            self._tb_writer.close()

    def _update_lr(self, trainer: "TianshouTrainer") -> None:
        """Apply learning rate decay.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.schedule_type == "exponential":
            # Exponential decay that reaches final_lr at decay_steps
            progress = min(self.current_step / self.decay_steps, 1.0)
            # Calculate per-step rate
            per_step_rate = (self.final_lr / self.initial_lr) ** (
                1.0 / self.decay_steps
            )
            new_lr = self.initial_lr * (per_step_rate**self.current_step)
        else:  # linear
            progress = min(self.current_step / max(self.decay_steps, 1), 1.0)
            new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * progress

        # Clamp to final_lr
        new_lr = max(new_lr, self.final_lr)

        # Update optimizer
        if hasattr(trainer, "optimizer"):
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = new_lr

        self.lr_history.append((self.current_step, new_lr))

    def _get_current_lr(self, trainer: "TianshouTrainer") -> float:
        """Get current learning rate from optimizer.

        Args:
            trainer: TianshouTrainer instance.

        Returns:
            Current learning rate.
        """
        if hasattr(trainer, "optimizer"):
            return trainer.optimizer.param_groups[0]["lr"]
        return self.initial_lr

    def _check_pulse_trigger(self, trainer: "TianshouTrainer") -> None:
        """Check if pulse should be triggered based on performance.

        Args:
            trainer: TianshouTrainer instance.
        """
        # Requirements for trigger
        if self.episode_count < self.min_episodes_before_pulse:
            return
        if len(self.episode_rewards) < self.reward_trend_window:
            return

        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        trend = self._calculate_trend(self.episode_rewards)

        # Update EMA baseline for adaptive threshold
        if self.adaptive_enabled:
            self._update_adaptive_baseline(avg_reward)

        # Determine effective threshold
        if (
            self.trigger_mode == "adaptive"
            and self.adaptive_pause_threshold is not None
        ):
            effective_threshold = self.adaptive_pause_threshold
            threshold_mode = "adaptive"
        else:
            effective_threshold = self.trigger_threshold
            threshold_mode = "static"

        # Check trigger condition: avg below threshold AND negative trend
        if effective_threshold is not None:
            if avg_reward < effective_threshold and trend < self.trend_threshold:
                self._start_pulse(trainer, avg_reward, trend, threshold_mode)

    def _update_adaptive_baseline(self, avg_reward: float) -> None:
        """Update EMA baseline for adaptive threshold.

        Args:
            avg_reward: Current average reward.
        """
        if self.ema_baseline is None:
            self.ema_baseline = avg_reward
        else:
            self.ema_baseline = (
                self.baseline_alpha * avg_reward
                + (1 - self.baseline_alpha) * self.ema_baseline
            )

        # Calculate adaptive thresholds
        if self.ema_baseline >= 0:
            self.adaptive_pause_threshold = self.ema_baseline * (
                1 - self.relative_drop_threshold
            )
        else:
            # For negative rewards, "worse" means more negative
            self.adaptive_pause_threshold = self.ema_baseline * (
                1 + self.relative_drop_threshold
            )

    def _start_pulse(
        self,
        trainer: "TianshouTrainer",
        avg_reward: float,
        trend: float,
        threshold_mode: str,
    ) -> None:
        """Start a pulse intervention.

        Args:
            trainer: TianshouTrainer instance.
            avg_reward: Current average reward.
            trend: Current reward trend.
            threshold_mode: "static" or "adaptive".
        """
        self.is_pulsing = True
        self.pulse_episodes_remaining = self.pulse_duration_episodes
        self.pulse_start_step = self.current_step
        self.frozen_lr = self._get_current_lr(trainer)

        # Optional: Boost exploration
        if self.pulse_boost_epsilon is not None:
            self._activate_exploration_boost(trainer)

        # Log pulse event
        effective_threshold = (
            self.adaptive_pause_threshold
            if threshold_mode == "adaptive"
            else self.trigger_threshold
        )
        baseline_str = f"{self.ema_baseline:.2f}" if self.ema_baseline else "N/A"

        pulse_reason = (
            f"avg_reward {avg_reward:.2f} < threshold {effective_threshold:.2f} "
            f"(mode={threshold_mode}, baseline={baseline_str}), trend {trend:.3f}"
        )

        self.pulse_events.append(
            {
                "step": self.current_step,
                "action": "pulse_start",
                "reason": pulse_reason,
                "frozen_lr": self.frozen_lr,
                "boost_epsilon": self.pulse_boost_epsilon,
                "duration_episodes": self.pulse_duration_episodes,
                "avg_reward": avg_reward,
                "trend": trend,
                "threshold_mode": threshold_mode,
                "ema_baseline": self.ema_baseline,
            }
        )

        self.logger.info(
            f"PULSE STARTED at step {self.current_step}: {pulse_reason}, "
            f"LR frozen at {self.frozen_lr:.6f} for {self.pulse_duration_episodes} episodes"
        )

    def _end_pulse(self, trainer: "TianshouTrainer") -> None:
        """End the current pulse intervention.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not self.is_pulsing:
            return

        self.is_pulsing = False
        pulse_duration = (
            self.current_step - self.pulse_start_step if self.pulse_start_step else 0
        )

        # Restore exploration if boosted
        if self.pulse_boost_epsilon is not None:
            self._deactivate_exploration_boost(trainer)

        self.pulse_events.append(
            {
                "step": self.current_step,
                "action": "pulse_end",
                "pulse_duration_actual": pulse_duration,
            }
        )

        self.logger.info(
            f"PULSE ENDED at step {self.current_step}, "
            f"duration: {pulse_duration} steps, resuming LR decay"
        )

        self.frozen_lr = None
        self.pulse_start_step = None

    def _activate_exploration_boost(self, trainer: "TianshouTrainer") -> None:
        """Temporarily boost exploration during pulse.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not hasattr(trainer, "policy"):
            return

        # Store original epsilon
        if hasattr(trainer.policy, "eps_training"):
            self.original_epsilon = trainer.policy.eps_training
            trainer.policy.eps_training = self.pulse_boost_epsilon
            if self.verbose:
                self.logger.info(
                    f"Exploration boost activated: epsilon set to {self.pulse_boost_epsilon}"
                )

    def _deactivate_exploration_boost(self, trainer: "TianshouTrainer") -> None:
        """Restore original exploration after pulse.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.original_epsilon is None:
            return

        if hasattr(trainer, "policy") and hasattr(trainer.policy, "eps_training"):
            trainer.policy.eps_training = self.original_epsilon
            if self.verbose:
                self.logger.info(
                    f"Exploration boost deactivated: epsilon restored to {self.original_epsilon:.4f}"
                )

        self.original_epsilon = None

    def _calculate_trend(self, rewards: List[float]) -> float:
        """Calculate linear trend (slope) of rewards using least squares.

        Args:
            rewards: List of recent rewards.

        Returns:
            Slope of linear trend.
        """
        n = len(rewards)
        if n < 2:
            return 0.0

        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(rewards) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, rewards))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _log_metrics(self, trainer: "TianshouTrainer") -> None:
        """Log current scheduler state to backends.

        Args:
            trainer: TianshouTrainer instance.
        """
        current_lr = self._get_current_lr(trainer)

        metrics = {
            "lr_scheduler/learning_rate": current_lr,
            "lr_scheduler/is_pulsing": float(self.is_pulsing),
            "lr_scheduler/episode_count": self.episode_count,
        }

        if self.ema_baseline is not None:
            metrics["lr_scheduler/ema_baseline"] = self.ema_baseline
        if self.adaptive_pause_threshold is not None:
            metrics["lr_scheduler/adaptive_threshold"] = self.adaptive_pause_threshold
        if self.episode_rewards:
            metrics["lr_scheduler/avg_reward"] = sum(self.episode_rewards) / len(
                self.episode_rewards
            )

        # Log to WandB
        if self.log_to_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=self.current_step)
            except ImportError:
                pass

        # Log to TensorBoard
        if self.log_to_tensorboard and self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, self.current_step)

    def _export_history(self) -> None:
        """Export LR history and pulse events to JSON files."""
        if not self.log_to_local:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Export LR history
        lr_history_path = os.path.join(self.output_dir, "lr_history.json")
        with open(lr_history_path, "w", encoding="utf-8") as f:
            json.dump(
                self.lr_history,
                f,
                indent=2,
                default=lambda x: float(x) if hasattr(x, "item") else x,
            )
        self.logger.info(f"LR history exported to {lr_history_path}")

        # Export pulse events if any
        if self.pulse_events:
            pulse_events_path = os.path.join(self.output_dir, "pulse_events.json")
            with open(pulse_events_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.pulse_events,
                    f,
                    indent=2,
                    default=lambda x: float(x) if hasattr(x, "item") else x,
                )
            self.logger.info(f"Pulse events exported to {pulse_events_path}")

    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get current scheduler state for external access.

        Returns:
            Dict containing scheduler state.
        """
        return {
            "current_step": self.current_step,
            "current_lr": (
                self.lr_history[-1][1] if self.lr_history else self.initial_lr
            ),
            "is_pulsing": self.is_pulsing,
            "pulse_episodes_remaining": self.pulse_episodes_remaining,
            "episode_count": self.episode_count,
            "ema_baseline": self.ema_baseline,
            "adaptive_threshold": self.adaptive_pause_threshold,
            "pulse_count": len(
                [e for e in self.pulse_events if e.get("action") == "pulse_start"]
            ),
        }
