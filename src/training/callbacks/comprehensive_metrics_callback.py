"""Comprehensive metrics callback for Tianshou training.

This module provides comprehensive experiment tracking for Tianshou training,
including hyperparameters, system metrics, training loss, Q-values, gradients,
and GHSOM-specific metrics.

Ported from: archive/sb3_legacy/callbacks/comprehensive_metrics.py
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import numpy as np
import psutil

from .base import BaseTianshouCallback

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

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

import json

if TYPE_CHECKING:
    from ..tianshou_trainer import TianshouTrainer


class TianshouComprehensiveMetricsCallback(BaseTianshouCallback):
    """Comprehensive metrics callback for Tianshou training.

    Logs to WandB and TensorBoard:
    - Hyperparameters (model config, architecture)
    - System metrics (GPU/CPU/memory usage)
    - Learning rate trajectory
    - Training loss and TD errors
    - Q-value statistics
    - Gradient norms
    - GHSOM cluster/neuron visit patterns

    Example:
        >>> callback = TianshouComprehensiveMetricsCallback(
        ...     config=training_config,
        ...     ghsom_manager=ghsom_manager,
        ...     log_training_metrics=True,
        ... )
        >>> trainer = TianshouTrainer(env, config, callbacks=[callback])
        >>> trainer.train()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        ghsom_manager: Optional[Any] = None,
        log_system_metrics: bool = True,
        log_model_info: bool = True,
        log_ghsom_metrics: bool = True,
        log_training_metrics: bool = True,
        system_metrics_freq: int = 100,
        q_value_log_freq: int = 100,
        gradient_log_freq: int = 100,
        log_to_tensorboard: bool = True,
        log_to_local: bool = True,
        output_dir: Optional[str] = None,
        verbose: int = 0,
    ):
        """Initialize comprehensive metrics callback.

        Args:
            config: Full training configuration dictionary.
            ghsom_manager: GHSOMManager instance for cluster hierarchy info.
            log_system_metrics: Whether to log GPU/CPU metrics.
            log_model_info: Whether to log model architecture info.
            log_ghsom_metrics: Whether to log GHSOM cluster/neuron visit patterns.
            log_training_metrics: Whether to log training loss, Q-values, gradients.
            system_metrics_freq: Frequency of system metrics logging (steps).
            q_value_log_freq: Frequency of Q-value statistics logging (steps).
            gradient_log_freq: Frequency of gradient norm logging (steps).
            log_to_tensorboard: Whether to also log metrics to TensorBoard.
            log_to_local: Whether to also log metrics to local JSON files.
            output_dir: Output directory for local logs (uses config paths if None).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.config = config
        self.ghsom_manager = ghsom_manager
        self.log_system_metrics = log_system_metrics
        self.log_model_info = log_model_info
        self.log_ghsom_metrics = log_ghsom_metrics
        self.log_training_metrics = log_training_metrics
        self.system_metrics_freq = system_metrics_freq
        self.q_value_log_freq = q_value_log_freq
        self.gradient_log_freq = gradient_log_freq

        # Tracking variables
        self.start_time: Optional[float] = None
        self.step_count = 0
        self.last_system_check = 0
        self.last_q_value_check = 0
        self.last_gradient_check = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        # Training metrics tracking
        self.loss_history: List[float] = []
        self.q_value_history: List[Dict[str, float]] = []
        self.gradient_norm_history: List[float] = []
        self.n_updates = 0

        # GHSOM tracking variables
        self.all_visited_clusters: Set[int] = set()
        self.episode_sequences: List[List[int]] = []
        self.cluster_visit_counts: Counter = Counter()
        self.transition_counts: Counter = Counter()
        self.current_episode_sequence: List[int] = []
        self.last_cluster_id: Optional[int] = None

        # GPU monitoring
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.cuda.current_device()

        # Process monitoring
        self.process = psutil.Process()

        # WandB initialization flag
        self._wandb_initialized = False

        # TensorBoard and local logging
        self.log_to_tensorboard = log_to_tensorboard and TENSORBOARD_AVAILABLE
        self.log_to_local = log_to_local
        self._tb_writer: Optional[Any] = None
        self._local_metrics_dir: Optional[Path] = None
        self._local_metrics: Dict[str, List[Any]] = {
            "system": [],
            "q_values": [],
            "gradients": [],
            "episodes": [],
            "ghsom": [],
        }

        # Determine output directory
        if output_dir is not None:
            self._output_dir = Path(output_dir)
        else:
            self._output_dir = Path(
                config.get("paths", {}).get("output_dir", "artifacts/training")
            )

    def on_training_start(self, trainer: "TianshouTrainer") -> None:
        """Log initial configuration and model architecture.

        Args:
            trainer: TianshouTrainer instance.
        """
        self.start_time = time.time()

        # Initialize TensorBoard writer if enabled
        if self.log_to_tensorboard and TENSORBOARD_AVAILABLE:
            tb_log_dir = self._output_dir / "tensorboard" / "comprehensive"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.logger.info(f"TensorBoard logging to: {tb_log_dir}")

        # Initialize local logging directory
        if self.log_to_local:
            metrics_dir = self._output_dir / "metrics" / "comprehensive"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            self._local_metrics_dir = metrics_dir
            self.logger.info(f"Local metrics logging to: {metrics_dir}")

        # Initialize WandB if available and not already done by another callback
        if WANDB_AVAILABLE and wandb.run is not None:
            self._wandb_initialized = True
            self._log_hyperparameters_to_wandb(trainer)

        if self.log_model_info:
            self._log_model_architecture(trainer)

        self.logger.info("ComprehensiveMetricsCallback initialized")

    def _log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        category: str = "train",
    ) -> None:
        """Log metrics to all enabled backends (WandB, TensorBoard, local JSON).

        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step.
            category: Category for local storage (system, q_values, gradients, etc.).
        """
        # Log to WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=step)

        # Log to TensorBoard
        if self._tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, step)

        # Store for local JSON dump
        if self.log_to_local and category in self._local_metrics:
            self._local_metrics[category].append({"step": step, **metrics})

    def _save_local_metrics(self) -> None:
        """Save accumulated metrics to local JSON files."""
        if not self.log_to_local or self._local_metrics_dir is None:
            return

        for category, data in self._local_metrics.items():
            if data:
                filepath = self._local_metrics_dir / f"{category}_metrics.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=float)

    def _log_hyperparameters_to_wandb(self, trainer: "TianshouTrainer") -> None:
        """Log all hyperparameters to WandB config.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not WANDB_AVAILABLE or wandb.run is None:
            return

        # Extract training config
        training_config = self.config.get("training", {})
        exploration_config = training_config.get("exploration", {})
        lr_scheduler_config = training_config.get("learning_rate_scheduler", {})
        network_config = self.config.get("network", {})
        reward_config = self.config.get("reward_components", {})

        config_update = {
            # Training hyperparameters
            "training/total_timesteps": training_config.get("total_timesteps", 10000),
            "training/batch_size": training_config.get("batch_size", 32),
            "training/learning_rate": training_config.get("learning_rate", 0.001),
            "training/gamma": training_config.get("gamma", 0.95),
            "training/target_update_freq": training_config.get(
                "target_update_freq", 1000
            ),
            "training/buffer_size": training_config.get("buffer_size", 10000),
            "training/step_per_collect": training_config.get("step_per_collect", 1),
            # Exploration parameters
            "exploration/initial_eps": exploration_config.get("initial_eps", 1.0),
            "exploration/final_eps": exploration_config.get("final_eps", 0.05),
            "exploration/fraction": exploration_config.get("fraction", 0.5),
            # Learning rate scheduling
            "lr_scheduler/enabled": lr_scheduler_config.get("enabled", False),
            "lr_scheduler/type": lr_scheduler_config.get("type", "linear"),
            "lr_scheduler/initial_lr": lr_scheduler_config.get("initial_lr", 0.001),
            "lr_scheduler/final_lr": lr_scheduler_config.get("final_lr", 0.0001),
            # Network architecture
            "network/type": network_config.get("type", "drqn"),
            "network/embedding_dim": network_config.get("embedding_dim", 64),
            "network/fc_hidden_sizes": str(
                network_config.get("fc_hidden_sizes", [128, 64])
            ),
            "network/activation_fn": network_config.get("activation_fn", "elu"),
            "network/dropout": network_config.get("dropout", 0.2),
            # LSTM config (if DRQN)
            "network/lstm_hidden_size": network_config.get("lstm", {}).get(
                "hidden_size", 128
            ),
            "network/lstm_num_layers": network_config.get("lstm", {}).get(
                "num_layers", 1
            ),
            # Environment settings
            "music/sequence_length": self.config.get("music", {}).get(
                "sequence_length", 16
            ),
            # System settings
            "system/device": self.config.get("system", {}).get("device", "auto"),
            "system/seed": self.config.get("system", {}).get("seed", 42),
        }

        # Add reward component weights
        for key, value in reward_config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    config_update[f"reward/{key}/{sub_key}"] = sub_value
            else:
                config_update[f"reward/{key}"] = value

        try:
            wandb.config.update(config_update, allow_val_change=True)
            self.logger.info(f"Logged {len(config_update)} hyperparameters to WandB")
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters to WandB: {e}")

    def _log_model_architecture(self, trainer: "TianshouTrainer") -> None:
        """Log model architecture information.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not hasattr(trainer, "network") or trainer.network is None:
            return

        try:
            network = trainer.network
            total_params = sum(p.numel() for p in network.parameters())
            trainable_params = sum(
                p.numel() for p in network.parameters() if p.requires_grad
            )

            model_info = {
                "model_info/total_parameters": total_params,
                "model_info/trainable_parameters": trainable_params,
                "model_info/parameter_ratio": trainable_params / max(total_params, 1),
                "model_info/is_recurrent": getattr(network, "is_recurrent", False),
            }

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.config.update(model_info, allow_val_change=True)

            self.logger.info(
                f"Model: {total_params:,} params ({trainable_params:,} trainable)"
            )
        except Exception as e:
            self.logger.warning(f"Could not log model architecture: {e}")

    def on_collect_end(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Track collection metrics and GHSOM patterns.

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        # Track episode completions
        if collect_result.n_collected_episodes > 0:
            self.episode_rewards.extend(collect_result.returns)
            self.episode_lengths.extend(collect_result.lens)

            # Log episode metrics
            recent_rewards = list(collect_result.returns)
            episode_metrics = {
                "episode/reward": np.mean(recent_rewards),
                "episode/length": np.mean(collect_result.lens),
                "episode/reward_std": (
                    np.std(recent_rewards) if len(recent_rewards) > 1 else 0
                ),
            }

            # Rolling statistics
            if len(self.episode_rewards) >= 10:
                recent_10 = self.episode_rewards[-10:]
                episode_metrics.update(
                    {
                        "episode/reward_mean_10": np.mean(recent_10),
                        "episode/reward_std_10": np.std(recent_10),
                        "episode/reward_min_10": np.min(recent_10),
                        "episode/reward_max_10": np.max(recent_10),
                    }
                )

            self._log_metrics(episode_metrics, trainer.num_timesteps, "episodes")

        # Track GHSOM metrics
        if self.log_ghsom_metrics:
            self._track_ghsom_metrics(collect_result, trainer)

    def _track_ghsom_metrics(
        self,
        collect_result: Any,
        trainer: "TianshouTrainer",
    ) -> None:
        """Track GHSOM cluster visit patterns.

        Tracks cluster sequences from environment regardless of ghsom_manager.
        The ghsom_manager is only needed for coverage metrics (total_nodes).

        Note: When an episode ends, Tianshou resets the environment before calling
        on_collect_end, so env.current_sequence will be empty/reset. We handle this
        by saving the sequence when it's complete (at episode end).

        Args:
            collect_result: Result from collector.collect().
            trainer: TianshouTrainer instance.
        """
        try:
            env = trainer.train_envs
            sequence = None

            # Try Tianshou's get_env_attr first (for DummyVectorEnv/SubprocVectorEnv)
            if hasattr(env, "get_env_attr"):
                try:
                    sequences = env.get_env_attr("current_sequence")
                    if sequences and len(sequences) > 0:
                        sequence = list(sequences[0])
                except Exception:
                    pass

            # Fallback: try get_attr (older API)
            if sequence is None and hasattr(env, "get_attr"):
                try:
                    sequences = env.get_attr("current_sequence")
                    if sequences and len(sequences) > 0:
                        sequence = list(sequences[0])
                except Exception:
                    pass

            # Direct attribute access
            if sequence is None and hasattr(env, "current_sequence"):
                sequence = list(env.current_sequence)

            # Walk through wrapper layers to find the underlying env
            if sequence is None:
                current = env
                for _ in range(5):
                    if hasattr(current, "current_sequence"):
                        sequence = list(current.current_sequence)
                        break
                    if hasattr(current, "env"):
                        current = current.env
                    elif hasattr(current, "envs") and len(current.envs) > 0:
                        current = current.envs[0]
                    else:
                        break

            # Handle episode completion: env was reset, so sequence is empty/new
            # If we have a previous sequence and episodes completed, save it
            if collect_result.n_collected_episodes > 0:
                # Episode completed - save the current tracking (not env's reset sequence)
                if self.current_episode_sequence:
                    self._log_ghsom_episode_metrics(trainer)

            # Now update tracking with current sequence from env
            if sequence:
                valid_sequence = [c for c in sequence if c >= 0]  # Skip padding (-1)

                # Track new clusters in this sequence
                for cluster_id in valid_sequence:
                    self.all_visited_clusters.add(cluster_id)
                    self.cluster_visit_counts[cluster_id] += 1

                    # Track transitions (only for new transitions)
                    if (
                        self.last_cluster_id is not None
                        and self.last_cluster_id != cluster_id
                    ):
                        transition = (self.last_cluster_id, cluster_id)
                        self.transition_counts[transition] += 1
                    self.last_cluster_id = cluster_id

                # Update current episode sequence
                self.current_episode_sequence = list(valid_sequence)

                # Log step-level GHSOM metrics
                ghsom_metrics = {
                    "ghsom/sequence_length": len(valid_sequence),
                    "ghsom/unique_clusters_episode": len(set(valid_sequence)),
                    "ghsom/cluster_diversity": len(set(valid_sequence))
                    / max(len(valid_sequence), 1),
                    "ghsom/total_unique_clusters": len(self.all_visited_clusters),
                }

                # Add coverage if we have ghsom_manager with total node count
                if self.ghsom_manager is not None and hasattr(
                    self.ghsom_manager, "stats"
                ):
                    total_nodes = self.ghsom_manager.stats.get("total_nodes", 0)
                    if total_nodes > 0:
                        ghsom_metrics["ghsom/cluster_coverage"] = (
                            len(self.all_visited_clusters) / total_nodes
                        )

                self._log_metrics(ghsom_metrics, trainer.num_timesteps, "ghsom")

        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Could not track GHSOM metrics: {e}")

    def _log_ghsom_episode_metrics(self, trainer: "TianshouTrainer") -> None:
        """Log comprehensive GHSOM metrics at episode end.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not self.current_episode_sequence:
            return

        sequence = self.current_episode_sequence

        # Store episode sequence
        self.episode_sequences.append(list(sequence))

        # Calculate metrics
        unique_clusters = len(set(sequence))
        cluster_diversity = unique_clusters / max(len(sequence), 1)
        transition_entropy = self._get_transition_entropy()
        pattern_analysis = self._detect_repetitive_patterns(sequence)

        episode_metrics = {
            "ghsom_episode/total_clusters_visited": len(sequence),
            "ghsom_episode/unique_clusters": unique_clusters,
            "ghsom_episode/cluster_diversity": cluster_diversity,
            "ghsom_episode/cluster_repetition_ratio": 1.0 - cluster_diversity,
            "ghsom_episode/has_repetitive_patterns": int(
                pattern_analysis["has_repetition"]
            ),
            "ghsom_episode/num_repeated_patterns": pattern_analysis[
                "num_repeated_patterns"
            ],
            "ghsom_episode/repetition_ratio": pattern_analysis["repetition_ratio"],
            "ghsom_episode/transition_entropy": transition_entropy,
        }

        self._log_metrics(episode_metrics, trainer.num_timesteps, "ghsom")

        # Reset episode-specific tracking
        self.current_episode_sequence = []
        self.last_cluster_id = None

    def _get_transition_entropy(self) -> float:
        """Calculate entropy of cluster transitions.

        Returns:
            Shannon entropy of transition distribution.
        """
        if not self.transition_counts:
            return 0.0

        total = sum(self.transition_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in self.transition_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def _detect_repetitive_patterns(
        self,
        sequence: List[int],
        min_pattern_length: int = 2,
        max_pattern_length: int = 5,
    ) -> Dict[str, Any]:
        """Detect repetitive patterns in cluster sequence.

        Args:
            sequence: List of cluster IDs.
            min_pattern_length: Minimum pattern length to detect.
            max_pattern_length: Maximum pattern length to detect.

        Returns:
            Dictionary with pattern analysis results.
        """
        if len(sequence) < min_pattern_length * 2:
            return {
                "has_repetition": False,
                "num_repeated_patterns": 0,
                "repetition_ratio": 0.0,
            }

        patterns: Counter = Counter()

        for pattern_len in range(
            min_pattern_length, min(max_pattern_length + 1, len(sequence) // 2 + 1)
        ):
            for i in range(len(sequence) - pattern_len + 1):
                pattern = tuple(sequence[i : i + pattern_len])
                patterns[pattern] += 1

        repeated_patterns = {k: v for k, v in patterns.items() if v > 1}

        return {
            "has_repetition": len(repeated_patterns) > 0,
            "num_repeated_patterns": len(repeated_patterns),
            "repetition_ratio": sum(v for v in patterns.values() if v > 1)
            / max(len(sequence), 1),
        }

    def on_train_step_end(
        self,
        train_result: Dict[str, Any],
        trainer: "TianshouTrainer",
    ) -> None:
        """Log training step metrics.

        Args:
            train_result: Result from policy.update().
            trainer: TianshouTrainer instance.
        """
        self.step_count += 1
        self.n_updates += 1

        # Track loss
        loss = (
            train_result.get("loss")
            if isinstance(train_result, dict)
            else getattr(train_result, "loss", None)
        )
        if loss is not None:
            self.loss_history.append(float(loss))

        # Log system metrics at intervals
        if (
            self.log_system_metrics
            and self.step_count - self.last_system_check >= self.system_metrics_freq
        ):
            self._log_system_metrics(trainer)
            self.last_system_check = self.step_count

        # Log Q-value statistics at intervals
        if (
            self.log_training_metrics
            and self.step_count - self.last_q_value_check >= self.q_value_log_freq
        ):
            self._log_q_value_statistics(trainer)
            self.last_q_value_check = self.step_count

        # Log gradient norms at intervals
        if (
            self.log_training_metrics
            and self.step_count - self.last_gradient_check >= self.gradient_log_freq
        ):
            self._log_gradient_norms(trainer)
            self.last_gradient_check = self.step_count

    def _log_system_metrics(self, trainer: "TianshouTrainer") -> None:
        """Log system performance metrics.

        Args:
            trainer: TianshouTrainer instance.
        """
        metrics: Dict[str, float] = {}

        try:
            # CPU metrics
            metrics["system/cpu_percent"] = self.process.cpu_percent()
            metrics["system/memory_mb"] = self.process.memory_info().rss / 1024 / 1024

            # GPU metrics
            if self.gpu_available and TORCH_AVAILABLE:
                metrics["system/gpu_memory_allocated_mb"] = (
                    torch.cuda.memory_allocated(self.device) / 1024 / 1024
                )
                metrics["system/gpu_memory_reserved_mb"] = (
                    torch.cuda.memory_reserved(self.device) / 1024 / 1024
                )
                metrics["system/gpu_available"] = 1.0
            else:
                metrics["system/gpu_available"] = 0.0

            # Training speed
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    metrics["performance/training_speed_steps_per_sec"] = (
                        trainer.num_timesteps / elapsed
                    )

            # Current training state
            metrics["training/learning_rate"] = trainer.optimizer.param_groups[0]["lr"]
            metrics["training/epsilon"] = trainer.exploration_rate
            metrics["training/buffer_size"] = len(trainer.buffer)

            self._log_metrics(metrics, trainer.num_timesteps, "system")

        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Could not log system metrics: {e}")

    def _log_q_value_statistics(self, trainer: "TianshouTrainer") -> None:
        """Compute and log Q-value statistics.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not TORCH_AVAILABLE or not hasattr(trainer, "buffer"):
            return

        try:
            buffer = trainer.buffer
            buffer_len = len(buffer)
            batch_size = min(64, buffer_len)

            if batch_size < 1:
                return

            # Sample from buffer
            batch, _ = buffer.sample(batch_size)

            with torch.no_grad():
                # Get observations
                obs = torch.as_tensor(
                    batch.obs, device=trainer.device, dtype=torch.float32
                )

                # Compute Q-values - handle recurrent networks that return (output, hidden_state)
                result = trainer.network(obs)

                # Extract Q-values from the result
                if isinstance(result, tuple):
                    q_values = result[0]  # First element is the Q-values
                else:
                    q_values = result

                # Handle named tuple outputs (e.g., ModelOutputs with .logits)
                if hasattr(q_values, "logits"):
                    q_values = q_values.logits

                # Ensure we have a tensor, then convert to numpy
                if hasattr(q_values, "detach"):
                    q_values_np = q_values.detach().cpu().numpy()
                elif hasattr(q_values, "cpu"):
                    q_values_np = q_values.cpu().numpy()
                else:
                    q_values_np = np.array(q_values)

                q_metrics = {
                    "q_values/mean": float(np.mean(q_values_np)),
                    "q_values/std": float(np.std(q_values_np)),
                    "q_values/max": float(np.max(q_values_np)),
                    "q_values/min": float(np.min(q_values_np)),
                    "q_values/range": float(np.max(q_values_np) - np.min(q_values_np)),
                }

                self.q_value_history.append(q_metrics)
                self._log_metrics(q_metrics, trainer.num_timesteps, "q_values")

        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Could not compute Q-value statistics: {e}")

    def _log_gradient_norms(self, trainer: "TianshouTrainer") -> None:
        """Compute and log gradient norms.

        Args:
            trainer: TianshouTrainer instance.
        """
        if not TORCH_AVAILABLE or not hasattr(trainer, "network"):
            return

        try:
            total_norm = 0.0
            param_count = 0
            max_grad_norm = 0.0
            min_grad_norm = float("inf")

            for param in trainer.network.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm**2
                    param_count += 1
                    max_grad_norm = max(max_grad_norm, param_norm)
                    min_grad_norm = min(min_grad_norm, param_norm)

            if param_count > 0:
                total_norm = total_norm**0.5
                self.gradient_norm_history.append(total_norm)

                grad_metrics = {
                    "gradients/total_norm": total_norm,
                    "gradients/mean_norm": total_norm / param_count,
                    "gradients/max_norm": max_grad_norm,
                    "gradients/min_norm": (
                        min_grad_norm if min_grad_norm != float("inf") else 0.0
                    ),
                    "gradients/exploding_warning": 1.0 if total_norm > 100 else 0.0,
                    "gradients/vanishing_warning": 1.0 if total_norm < 1e-7 else 0.0,
                }

                self._log_metrics(grad_metrics, trainer.num_timesteps, "gradients")

        except Exception as e:
            if self.verbose > 0:
                self.logger.warning(f"Could not compute gradient norms: {e}")

    def on_training_end(self, trainer: "TianshouTrainer") -> None:
        """Log final training summary and cleanup resources.

        Args:
            trainer: TianshouTrainer instance.
        """
        # Save local metrics before any early returns
        self._save_local_metrics()

        # Export episode sequences for post-training analysis
        if self.log_to_local and self._local_metrics_dir is not None:
            if self.episode_sequences:
                sequences_path = self._local_metrics_dir / "episode_sequences.json"
                sequences_data = {
                    "total_episodes": len(self.episode_sequences),
                    "sequences": self.episode_sequences,
                    "metadata": {
                        "description": "Cluster IDs visited per episode",
                        "format": "List of lists, inner list = cluster_ids per episode",
                    },
                }
                with open(sequences_path, "w", encoding="utf-8") as f:
                    json.dump(sequences_data, f, indent=2)
                self.logger.info(f"Episode sequences saved to {sequences_path}")

        # Close TensorBoard writer
        if self._tb_writer is not None:
            self._tb_writer.close()
            self.logger.info("TensorBoard writer closed")

        if self.start_time is None:
            return

        total_time = time.time() - self.start_time

        summary_metrics: Dict[str, Any] = {
            "summary/total_training_time_sec": total_time,
            "summary/total_timesteps": trainer.num_timesteps,
            "summary/average_speed_steps_per_sec": trainer.num_timesteps
            / max(total_time, 1),
            "summary/total_episodes": len(self.episode_rewards),
            "summary/total_gradient_updates": self.n_updates,
        }

        # Episode reward summary
        if self.episode_rewards:
            summary_metrics.update(
                {
                    "summary/final_mean_reward": np.mean(self.episode_rewards[-100:]),
                    "summary/best_episode_reward": np.max(self.episode_rewards),
                    "summary/total_reward": np.sum(self.episode_rewards),
                    "summary/reward_std": np.std(self.episode_rewards),
                }
            )

        # Loss summary
        if self.loss_history:
            summary_metrics.update(
                {
                    "summary/final_loss": self.loss_history[-1],
                    "summary/mean_loss": np.mean(self.loss_history),
                    "summary/min_loss": np.min(self.loss_history),
                    "summary/loss_std": np.std(self.loss_history),
                }
            )

        # Gradient summary
        if self.gradient_norm_history:
            summary_metrics.update(
                {
                    "summary/mean_gradient_norm": np.mean(self.gradient_norm_history),
                    "summary/max_gradient_norm": np.max(self.gradient_norm_history),
                }
            )

        # Q-value summary
        if self.q_value_history:
            final_q = self.q_value_history[-1]
            summary_metrics.update(
                {
                    "summary/final_q_mean": final_q.get("q_values/mean", 0),
                    "summary/final_q_std": final_q.get("q_values/std", 0),
                }
            )

        # GHSOM summary
        if self.log_ghsom_metrics and self.all_visited_clusters:
            summary_metrics.update(
                {
                    "summary/ghsom_total_unique_clusters": len(
                        self.all_visited_clusters
                    ),
                    "summary/ghsom_total_cluster_visits": sum(
                        self.cluster_visit_counts.values()
                    ),
                    "summary/ghsom_total_transitions": sum(
                        self.transition_counts.values()
                    ),
                    "summary/ghsom_final_transition_entropy": self._get_transition_entropy(),
                }
            )

            if self.ghsom_manager and hasattr(self.ghsom_manager, "stats"):
                total_nodes = self.ghsom_manager.stats.get("total_nodes", 0)
                if total_nodes > 0:
                    summary_metrics["summary/ghsom_cluster_coverage"] = (
                        len(self.all_visited_clusters) / total_nodes
                    )

            # Top visited clusters
            for i, (cluster_id, count) in enumerate(
                self.cluster_visit_counts.most_common(5)
            ):
                summary_metrics[f"summary/ghsom_top{i+1}_cluster_id"] = cluster_id
                summary_metrics[f"summary/ghsom_top{i+1}_total_visits"] = count

        # Log summary to all backends
        self._log_metrics(summary_metrics, trainer.num_timesteps, "system")

        # Also save summary as standalone JSON
        if self.log_to_local and hasattr(self, "_local_metrics_dir"):
            summary_path = self._local_metrics_dir / "training_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_metrics, f, indent=2, default=float)
            self.logger.info(f"Training summary saved to {summary_path}")

        self.logger.info(f"Logged {len(summary_metrics)} summary metrics")
