"""Detailed reward component logging callback for Tianshou training.

This module provides a callback for logging individual reward components
(structure, transition, diversity) to provide detailed insights into
agent behavior during training.

"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseTianshouCallback

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class RewardComponentTracker:
    """Tracking logic for reward components with statistics aggregation.

    This tracker maintains history of step-level and episode-level rewards
    along with their component breakdowns, and computes rolling statistics.

    Attributes:
        max_history: Maximum number of episodes to retain in history.
        episode_rewards: Deque of total episode rewards.
        episode_component_rewards: Deque of component reward dicts per episode.
        step_rewards: Deque of step rewards.
        step_component_rewards: Deque of component reward dicts per step.
        total_episodes: Total number of completed episodes.
        total_steps: Total number of steps recorded.
    """

    def __init__(self, max_history: int = 1000):
        """Initialize tracker with configurable history size.

        Args:
            max_history: Maximum episodes to keep in history.
        """
        self.max_history = max_history
        self.episode_rewards: deque = deque(maxlen=max_history)
        self.episode_component_rewards: deque = deque(maxlen=max_history)
        self.step_rewards: deque = deque(maxlen=max_history * 100)
        self.step_component_rewards: deque = deque(maxlen=max_history * 100)

        # Aggregated statistics
        self.total_episodes = 0
        self.total_steps = 0

        # Component name tracking for consistent ordering
        self._known_components: set = set()

    def record_step(
        self, reward: float, reward_components: Optional[Dict[str, float]] = None
    ) -> None:
        """Record a step-level reward and its components.

        Args:
            reward: Total reward for this step.
            reward_components: Dict mapping component names to values.
        """
        self.step_rewards.append(reward)
        components = reward_components or {}
        self.step_component_rewards.append(components)
        self._known_components.update(components.keys())
        self.total_steps += 1

    def record_episode(
        self,
        total_reward: float,
        episode_components: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record an episode-level reward and its aggregated components.

        Args:
            total_reward: Total reward for the episode.
            episode_components: Dict of component totals for the episode.
        """
        self.episode_rewards.append(total_reward)
        components = episode_components or {}
        self.episode_component_rewards.append(components)
        self._known_components.update(components.keys())
        self.total_episodes += 1

    def get_recent_stats(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Get statistics for the most recent n episodes.

        Args:
            n_episodes: Number of recent episodes to include.

        Returns:
            Dictionary with statistics including means, stds, min, max,
            and per-component statistics.
        """
        if not self.episode_rewards:
            return {"message": "No episodes recorded yet"}

        recent_rewards = list(self.episode_rewards)[-n_episodes:]
        recent_components = list(self.episode_component_rewards)[-n_episodes:]

        stats = {
            "episodes_count": len(recent_rewards),
            "mean_reward": float(np.mean(recent_rewards)),
            "std_reward": float(np.std(recent_rewards)),
            "min_reward": float(np.min(recent_rewards)),
            "max_reward": float(np.max(recent_rewards)),
        }

        # Component statistics if available
        if recent_components and any(comp for comp in recent_components):
            component_stats: Dict[str, List[float]] = defaultdict(list)
            for comp_dict in recent_components:
                for key, value in comp_dict.items():
                    if value is not None:
                        component_stats[key].append(value)

            for comp_name, values in component_stats.items():
                if values:
                    stats[f"{comp_name}_mean"] = float(np.mean(values))
                    stats[f"{comp_name}_std"] = float(np.std(values))

        return stats

    def get_step_stats(self, n_steps: int = 100) -> Dict[str, Any]:
        """Get statistics for the most recent n steps.

        Args:
            n_steps: Number of recent steps to include.

        Returns:
            Dictionary with step-level statistics.
        """
        if not self.step_rewards:
            return {"message": "No steps recorded yet"}

        recent_rewards = list(self.step_rewards)[-n_steps:]
        recent_components = list(self.step_component_rewards)[-n_steps:]

        stats = {
            "steps_count": len(recent_rewards),
            "mean_step_reward": float(np.mean(recent_rewards)),
            "std_step_reward": float(np.std(recent_rewards)),
        }

        # Component statistics
        if recent_components and any(comp for comp in recent_components):
            component_stats: Dict[str, List[float]] = defaultdict(list)
            for comp_dict in recent_components:
                for key, value in comp_dict.items():
                    if value is not None:
                        component_stats[key].append(value)

            for comp_name, values in component_stats.items():
                if values:
                    stats[f"step_{comp_name}_mean"] = float(np.mean(values))

        return stats

    def _detect_frozen_values(self, n_recent: int = 100) -> Dict[str, Any]:
        """Detect if recent values are frozen (constant/unchanging).

        This can indicate a bug in reward calculation when the agent's
        policy becomes too deterministic.

        Args:
            n_recent: Number of recent episodes to check.

        Returns:
            Dictionary with frozen value warnings if detected.
        """
        warnings = {}

        if len(self.episode_rewards) < n_recent:
            return warnings

        recent_rewards = list(self.episode_rewards)[-n_recent:]
        unique_rewards = len(set(recent_rewards))

        # Check if episode rewards are frozen (< 5 unique values in last 100)
        if unique_rewards < 5:
            warnings["episode_rewards_frozen"] = {
                "unique_values": unique_rewards,
                "last_value": recent_rewards[-1] if recent_rewards else None,
                "message": f"WARNING: Only {unique_rewards} unique reward values in last {n_recent} episodes"
            }

        # Check each component for frozen values
        recent_components = list(self.episode_component_rewards)[-n_recent:]
        if recent_components:
            for comp_name in self._known_components:
                comp_values = [c.get(comp_name, 0) for c in recent_components if comp_name in c]
                if len(comp_values) >= n_recent // 2:
                    unique_comp = len(set(comp_values))
                    if unique_comp < 3:  # Very few unique values
                        warnings[f"{comp_name}_frozen"] = {
                            "unique_values": unique_comp,
                            "last_value": comp_values[-1] if comp_values else None,
                            "message": f"WARNING: Component '{comp_name}' appears frozen ({unique_comp} unique values)"
                        }

        return warnings

    def export_to_json(self, filepath: Union[str, Path]) -> None:
        """Export tracking data to JSON file.

        Args:
            filepath: Path to output JSON file.
        """
        # Check for frozen values and log warnings
        frozen_warnings = self._detect_frozen_values()
        if frozen_warnings:
            for key, warning_info in frozen_warnings.items():
                # Use print to ensure visibility even if logger is silent
                print(f"[RewardComponentTracker] {warning_info.get('message', key)}")

        data = {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "episode_rewards": list(self.episode_rewards),
            "episode_component_rewards": [
                dict(c) for c in self.episode_component_rewards
            ],
            "step_rewards": list(self.step_rewards),
            "step_component_rewards": [dict(c) for c in self.step_component_rewards],
            "recent_stats": self.get_recent_stats(),
            "step_stats": self.get_step_stats(),
            "known_components": list(self._known_components),
            "frozen_value_warnings": frozen_warnings if frozen_warnings else None,
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, np.floating) else x,
            )


class TianshouRewardComponentsCallback(BaseTianshouCallback):
    """Callback for logging individual reward components during Tianshou training.

    This callback tracks and logs the breakdown of rewards into their
    component parts (e.g., structure, transition, diversity) to provide
    detailed insights into what the agent is learning.

    Features:
        - Step-level reward component tracking
        - Episode-level aggregated statistics
        - WandB logging (optional)
        - TensorBoard logging (optional)
        - Local JSON export

    Example:
        >>> callback = TianshouRewardComponentsCallback(
        ...     log_to_wandb=True,
        ...     log_to_tensorboard=True,
        ...     output_dir="artifacts/training/run_001",
        ... )
        >>> trainer = TianshouTrainer(env, config, callbacks=[callback])
        >>> trainer.train()
    """

    def __init__(
        self,
        log_frequency: int = 100,
        export_frequency: int = 1000,
        log_to_wandb: bool = True,
        log_to_tensorboard: bool = True,
        log_to_local: bool = True,
        output_dir: Optional[str] = None,
        max_history: int = 1000,
        verbose: int = 0,
    ):
        """Initialize reward components callback.

        Args:
            log_frequency: How often to log statistics (in steps).
            export_frequency: How often to export data to file (in steps).
            log_to_wandb: Whether to log to WandB.
            log_to_tensorboard: Whether to log to TensorBoard.
            log_to_local: Whether to log to local JSON file.
            output_dir: Output directory for logs. Required if log_to_local=True.
            max_history: Maximum episodes to keep in history.
            verbose: Verbosity level (0=silent, 1=info, 2=debug).
        """
        super().__init__(verbose)

        self.log_frequency = log_frequency
        self.export_frequency = export_frequency
        self.log_to_wandb = log_to_wandb and WANDB_AVAILABLE
        self.log_to_tensorboard = log_to_tensorboard and TENSORBOARD_AVAILABLE
        self.log_to_local = log_to_local

        # Validate output_dir if local logging is enabled
        if log_to_local and output_dir is None:
            self.logger.warning(
                "log_to_local=True but no output_dir provided. Local logging disabled."
            )
            self.log_to_local = False

        self._output_dir = Path(output_dir) if output_dir else None

        # Initialize tracker
        self.tracker = RewardComponentTracker(max_history=max_history)

        # Step counter
        self.step_count = 0

        # TensorBoard writer (initialized on training start)
        self._tb_writer: Optional[Any] = None

        # Episode tracking for component aggregation
        self._current_episode_components: Dict[str, float] = defaultdict(float)
        self._current_episode_steps = 0

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Initialize TensorBoard writer on training start.

        Args:
            trainer: TianshouTrainer instance.
        """
        if self.log_to_tensorboard and self._output_dir is not None:
            tb_log_dir = self._output_dir / "tensorboard" / "reward_components"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.logger.info(f"TensorBoard logging to: {tb_log_dir}")

        if self.log_to_local and self._output_dir is not None:
            local_dir = self._output_dir / "metrics" / "reward_components"
            local_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Local metrics logging to: {local_dir}")

        self.logger.info("RewardComponentsCallback initialized")

    def _extract_reward_components_from_batch_info(self, info: Any) -> Dict[str, float]:
        """Extract reward components from Tianshou Batch info object.

        Tianshou stores info as a nested Batch object, not a dict.
        This method handles both dict and Batch formats.

        Args:
            info: Info object from buffer (Batch or dict).

        Returns:
            Dictionary of reward component names to values.
        """
        reward_components: Dict[str, float] = {}

        if info is None:
            return reward_components

        # Handle dict format
        if isinstance(info, dict):
            rc = info.get("reward_components", {})
            if isinstance(rc, dict):
                return rc

        # Handle Tianshou Batch format
        # Batch stores nested data as attributes
        if hasattr(info, "reward_components"):
            rc = info.reward_components
            if rc is not None:
                # rc is itself a Batch with component attributes
                if hasattr(rc, "__dict__"):
                    # Iterate over Batch attributes
                    for key in dir(rc):
                        if not key.startswith("_"):
                            try:
                                value = getattr(rc, key)
                                if isinstance(value, (int, float, np.floating)):
                                    reward_components[key] = float(value)
                            except (AttributeError, TypeError):
                                pass
                elif isinstance(rc, dict):
                    reward_components = dict(rc)

        return reward_components

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Process collected data for reward components.

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        # Get buffer to access recent transitions
        # Trainer uses 'buffer' not 'replay_buffer'
        if not hasattr(trainer, "buffer") or trainer.buffer is None:
            return

        buffer = trainer.buffer

        # Get the number of newly collected transitions
        n_collected = getattr(collect_result, "n_collected_steps", 1)

        # Access recent transitions from buffer
        # The buffer stores info dicts which contain reward_components
        try:
            # Get the last n_collected transitions
            buffer_len = len(buffer)
            if buffer_len == 0:
                return

            start_idx = max(0, buffer_len - n_collected)

            if self.verbose > 1:
                self.logger.debug(
                    f"Processing {n_collected} steps from buffer indices {start_idx} to {buffer_len-1}"
                )

            for idx in range(start_idx, buffer_len):
                # Access buffer data
                batch = buffer[idx]

                # Extract reward
                reward = float(batch.rew) if hasattr(batch, "rew") else 0.0

                # Extract reward components from info using helper method
                reward_components = {}
                if hasattr(batch, "info") and batch.info is not None:
                    reward_components = self._extract_reward_components_from_batch_info(
                        batch.info
                    )
                    if self.verbose > 1 and not reward_components:
                        self.logger.debug(
                            f"No reward components found in batch.info at index {idx}"
                        )
                else:
                    if self.verbose > 1:
                        self.logger.debug(f"No info field in batch at index {idx}")

                # Record step
                self.tracker.record_step(reward, reward_components)
                self.step_count += 1

                # Aggregate for current episode
                for comp_name, comp_value in reward_components.items():
                    self._current_episode_components[comp_name] += comp_value
                self._current_episode_steps += 1

                # Check if episode ended
                done = bool(batch.done) if hasattr(batch, "done") else False
                terminated = (
                    bool(batch.terminated) if hasattr(batch, "terminated") else done
                )
                truncated = (
                    bool(batch.truncated) if hasattr(batch, "truncated") else False
                )

                if terminated or truncated or done:
                    # Calculate episode totals
                    episode_reward = sum(
                        list(self.tracker.step_rewards)[-self._current_episode_steps :]
                    )

                    # Record episode
                    self.tracker.record_episode(
                        episode_reward, dict(self._current_episode_components)
                    )

                    # Log episode summary
                    if self.verbose > 0:
                        self._log_episode_summary(
                            episode_reward, dict(self._current_episode_components)
                        )

                    # Reset episode tracking
                    self._current_episode_components = defaultdict(float)
                    self._current_episode_steps = 0

        except Exception as e:
            if self.verbose > 1:
                self.logger.debug(f"Could not extract reward components: {e}")
            else:
                self.logger.warning(f"Error processing reward components: {e}")

        # Periodic logging
        if self.step_count > 0 and self.step_count % self.log_frequency == 0:
            self._log_statistics()

        # Periodic export
        if (
            self.log_to_local
            and self._output_dir is not None
            and self.step_count > 0
            and self.step_count % self.export_frequency == 0
        ):
            export_path = (
                self._output_dir / "metrics" / "reward_components" / "history.json"
            )
            self.tracker.export_to_json(export_path)

    def _log_episode_summary(
        self, total_reward: float, components: Dict[str, float]
    ) -> None:
        """Log episode summary to console.

        Args:
            total_reward: Total episode reward.
            components: Reward component breakdown.
        """
        summary = f"R={total_reward:.3f}"
        if components:
            comp_str = ", ".join(
                [f"{k}={v:.3f}" for k, v in components.items() if v is not None]
            )
            if comp_str:
                summary += f" ({comp_str})"

        self.logger.info(f"Episode {self.tracker.total_episodes}: {summary}")

    def _log_statistics(self) -> None:
        """Log current statistics to all enabled backends."""
        step_stats = self.tracker.get_step_stats()
        episode_stats = self.tracker.get_recent_stats()

        # Prepare metrics dict
        metrics = {
            "reward_components/total_steps": self.tracker.total_steps,
            "reward_components/total_episodes": self.tracker.total_episodes,
        }

        # Add step statistics
        if "mean_step_reward" in step_stats:
            metrics["reward_components/step_reward_mean"] = step_stats[
                "mean_step_reward"
            ]

        # Add component means for steps
        for key, value in step_stats.items():
            if key.startswith("step_") and key.endswith("_mean"):
                comp_name = key[5:-5]  # Remove 'step_' prefix and '_mean' suffix
                metrics[f"reward_components/step_{comp_name}"] = value

        # Add episode statistics
        if "mean_reward" in episode_stats:
            metrics["reward_components/episode_reward_mean"] = episode_stats[
                "mean_reward"
            ]
            metrics["reward_components/episode_reward_std"] = episode_stats.get(
                "std_reward", 0.0
            )

        # Add component means for episodes
        for key, value in episode_stats.items():
            if key.endswith("_mean") and not key.startswith("mean_"):
                comp_name = key[:-5]  # Remove '_mean' suffix
                metrics[f"reward_components/episode_{comp_name}"] = value

        # Log to WandB
        if self.log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=self.step_count)

        # Log to TensorBoard
        if self._tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, self.step_count)

        # Console logging at higher verbosity
        if self.verbose > 1:
            formatted = self._format_stats(episode_stats)
            self.logger.info(f"Step {self.step_count}: {formatted}")

    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """Format statistics for console output.

        Args:
            stats: Statistics dictionary.

        Returns:
            Formatted string.
        """
        if "message" in stats:
            return stats["message"]

        formatted = f"Episodes={stats.get('episodes_count', 0)}, "
        formatted += f"Mean R={stats.get('mean_reward', 0):.3f}"

        std = stats.get("std_reward", 0)
        if std > 0:
            formatted += f"±{std:.3f}"

        # Add component means
        component_keys = [
            k for k in stats.keys() if k.endswith("_mean") and not k.startswith("mean_")
        ]
        if component_keys:
            comp_parts = [
                f"{k.replace('_mean', '')}={stats[k]:.3f}" for k in component_keys[:3]
            ]
            formatted += f" | Components: {', '.join(comp_parts)}"

        return formatted

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Export final data and clean up.

        Args:
            trainer: TianshouTrainer instance.
        """
        # Final export
        if self.log_to_local and self._output_dir is not None:
            export_path = (
                self._output_dir
                / "metrics"
                / "reward_components"
                / "final_history.json"
            )
            self.tracker.export_to_json(export_path)
            self.logger.info(f"Final reward components exported to: {export_path}")

        # Close TensorBoard writer
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
            self.logger.info("TensorBoard writer closed")

        # Log final summary
        summary = self.get_training_summary()
        self.logger.info(
            f"Reward components tracking complete: "
            f"{summary['total_episodes']} episodes, {summary['total_steps']} steps"
        )

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a comprehensive training summary.

        Returns:
            Dictionary with training statistics.
        """
        return {
            "total_episodes": self.tracker.total_episodes,
            "total_steps": self.tracker.total_steps,
            "final_stats": self.tracker.get_recent_stats(50),
            "has_component_data": any(
                comp for comp in self.tracker.episode_component_rewards
            ),
            "known_components": list(self.tracker._known_components),
        }
