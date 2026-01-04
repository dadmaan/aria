"""Tianshou-native trainer for Q-learning agents.

This module provides a clean, modular trainer that uses Tianshou's
native training patterns.

The trainer:
- Uses NetworkFactory for network instantiation
- Supports DRQN (recurrent), MLP (feedforward), and Rainbow variants
- Provides native Tianshou Collector integration
- Handles checkpoint saving/loading
- Supports callback-based logging
- Supports algorithm selection: DQN, Dueling DQN, C51, Rainbow

Classes:
    TianshouTrainer: Main trainer class for Tianshou Q-learning.

Example:
    >>> from src.training import TianshouTrainer
    >>> trainer = TianshouTrainer(env=env, config=config)
    >>> trainer.train(total_timesteps=10000)
    >>> trainer.save("checkpoints/model")
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.algorithm import DQN, C51, RainbowDQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.torch_utils import policy_within_training_step

from ..networks import NetworkFactory
from ..utils.logging.logging_manager import get_logger


class TianshouTrainer:
    """Tianshou-native trainer for Q-learning agents.

    This trainer provides a clean interface for training DQN-based agents
    using Tianshou's native patterns. It supports both recurrent (DRQN)
    and feedforward (MLP) networks through the NetworkFactory.

    Key features:
        - Modular network architecture via NetworkFactory
        - Native Tianshou Collector and Buffer integration
        - Support for both DRQN and MLP networks
        - Callback-based logging (WandB, TensorBoard)
        - Checkpoint save/load functionality
        - Proper hidden state handling for recurrent networks

    Attributes:
        env: Gymnasium environment or vectorized environment.
        config: Training configuration dictionary.
        network: Q-network instance (DRQN or MLP).
        policy: Tianshou DiscreteQLearningPolicy (action selection).
        algorithm: Tianshou DQN algorithm (learning, target network).
        buffer: Replay buffer for experience storage.
        train_collector: Collector for training data.
        test_collector: Collector for evaluation (optional).

    Example:
        >>> config = load_yaml("configs/tianshou_config.yaml")
        >>> env = make_music_env(config)
        >>> trainer = TianshouTrainer(env=env, config=config)
        >>>
        >>> # Train for 10000 steps
        >>> result = trainer.train(total_timesteps=10000)
        >>>
        >>> # Save checkpoint
        >>> trainer.save("checkpoints/final")
    """

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        test_env: Optional[gym.Env] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        """Initialize Tianshou trainer.

        Args:
            env: Training environment (Gymnasium compatible).
            config: Full configuration dictionary with keys:
                - network: Network architecture config
                - training: Training hyperparameters
                - logging: Logging configuration
                - system: System settings (device, seed)
            test_env: Optional separate environment for evaluation.
                If None, uses training env for evaluation.
            callbacks: List of callback instances for logging/checkpointing.

        Raises:
            ValueError: If config is missing required keys.
        """
        self.logger = get_logger("TianshouTrainer")
        self.env = env
        self.config = config
        self.callbacks = callbacks or []

        # Extract config sections
        self.network_config = config.get("network", {})
        self.training_config = config.get("training", {})
        self.system_config = config.get("system", {})

        # Device setup
        device_cfg = self.system_config.get("device", "auto")
        if device_cfg == "auto":
            enable_gpu = self.system_config.get("enable_gpu", True)
            self.device = (
                "cuda" if (torch.cuda.is_available() and enable_gpu) else "cpu"
            )
        else:
            self.device = device_cfg
        self.logger.info(f"Using device: {self.device}")

        # Set random seed
        seed = self.system_config.get("seed", 42)
        self._set_seed(seed)

        # Build components
        self._build_network()
        self._build_algorithm()
        self._build_collectors(test_env)

        # Training state
        self.num_timesteps = 0
        self.num_episodes = 0
        self.exploration_rate = self.training_config.get("exploration", {}).get(
            "initial_eps", 1.0
        )

        # Track losses for logging
        self._recent_losses: List[float] = []
        self._last_episode_rewards: List[float] = []

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        self.logger.info(f"Random seed set to: {seed}")

    def _build_network(self) -> None:
        """Build Q-network from configuration."""
        # Get observation and action shapes
        if hasattr(self.env, "observation_space"):
            self.observation_space = self.env.observation_space
            state_shape = self.observation_space.shape
        else:
            self.observation_space = self.env.envs[0].observation_space
            state_shape = self.observation_space.shape

        if hasattr(self.env, "action_space"):
            self.action_space = self.env.action_space
            action_shape = self.action_space.n
        else:
            self.action_space = self.env.envs[0].action_space
            action_shape = self.action_space.n

        # Create network via factory
        network_type = self.network_config.get("type", "drqn")

        self.network = NetworkFactory.create(
            network_type=network_type,
            state_shape=state_shape,
            action_shape=action_shape,
            config=self.network_config,
            device=self.device,
        )
        self.network = self.network.to(self.device)

        # Log network info
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )

        self.logger.info(f"Network created: {network_type.upper()}")
        self.logger.info(f"  State shape: {state_shape}")
        self.logger.info(f"  Action shape: {action_shape}")
        self.logger.info(f"  Is recurrent: {self.network.is_recurrent}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Log NoisyNet info if enabled
        use_noisy = getattr(self.network, "use_noisy_layers", False)
        if use_noisy:
            noisy_sigma = getattr(self.network, "noisy_sigma", 0.5)
            self.logger.info(f"  NoisyNet enabled: sigma_init={noisy_sigma}")

    def _build_algorithm(self) -> None:
        """Build Tianshou algorithm/policy based on configuration.

        Tianshou 2.0 separates:
        - Policy: Action selection, epsilon-greedy exploration
        - Algorithm: Learning, target network updates, optimization

        Supports algorithm types:
        - dqn: Standard DQN with Double DQN
        - dueling_dqn: Dueling architecture (uses DQN algorithm)
        - c51: Categorical distributional RL
        - rainbow: C51 + Dueling + PER
        """
        lr = self.training_config.get("learning_rate", 1e-3)
        gamma = self.training_config.get("gamma", 0.99)
        target_update = self.training_config.get("target_update_freq", 1000)

        # Get n-step return horizon from config (default 1 for standard TD)
        n_step = self.training_config.get("n_step", 1)
        self.n_step = n_step  # Store for curriculum callback access

        # Get initial exploration parameters
        exploration_cfg = self.training_config.get("exploration", {})
        initial_eps = exploration_cfg.get("initial_eps", 1.0)
        final_eps = exploration_cfg.get("final_eps", 0.05)

        # Get algorithm type from config
        algo_config = self.config.get("algorithm", {})
        algo_type = algo_config.get("type", "dqn").lower()

        # Store algorithm type for later use
        self.algo_type = algo_type

        # Store optimizer for checkpoint saving
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Step 1: Create Policy (action selection, epsilon-greedy)
        self.policy = DiscreteQLearningPolicy(
            model=self.network,
            action_space=self.action_space,
            observation_space=self.observation_space,
            eps_training=initial_eps,
            eps_inference=final_eps,
        )

        # Step 2: Create Algorithm based on type
        if algo_type in ("dqn", "dueling_dqn"):
            # Standard DQN or Dueling DQN (architecture handled by network)
            self.algorithm = DQN(
                policy=self.policy,
                optim=AdamOptimizerFactory(lr=lr),
                gamma=gamma,
                n_step_return_horizon=n_step,
                target_update_freq=target_update,
            )
            self.logger.info(f"{algo_type.upper()} Algorithm created (Tianshou 2.0):")
            if n_step > 1:
                self.logger.info(f"  n-step returns: {n_step}")

        elif algo_type == "c51":
            # C51 Categorical Distributional RL
            # C51 requires C51Policy instead of DiscreteQLearningPolicy
            dist_config = algo_config.get("distributional", {})
            num_atoms = dist_config.get("num_atoms", 51)
            v_min = dist_config.get("v_min", -3.0)
            v_max = dist_config.get("v_max", 7.0)

            # Create C51Policy with distributional parameters
            self.policy = C51Policy(
                model=self.network,
                action_space=self.action_space,
                observation_space=self.observation_space,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                eps_training=initial_eps,
                eps_inference=final_eps,
            )
            # Move policy to device (important: C51Policy has support tensor that needs to be on GPU)
            self.policy = self.policy.to(self.device)

            self.algorithm = C51(
                policy=self.policy,
                optim=AdamOptimizerFactory(lr=lr),
                gamma=gamma,
                n_step_return_horizon=n_step,
                target_update_freq=target_update,
            )
            self.logger.info("C51 Algorithm created (Tianshou 2.0):")
            self.logger.info(f"  num_atoms: {num_atoms}")
            self.logger.info(f"  v_min: {v_min}, v_max: {v_max}")
            if n_step > 1:
                self.logger.info(f"  n-step returns: {n_step}")

        elif algo_type == "rainbow":
            # Rainbow DQN: C51 + Dueling + PER
            # Rainbow also uses C51Policy
            dist_config = algo_config.get("distributional", {})
            num_atoms = dist_config.get("num_atoms", 51)
            v_min = dist_config.get("v_min", -3.0)
            v_max = dist_config.get("v_max", 7.0)

            # Create C51Policy with distributional parameters
            self.policy = C51Policy(
                model=self.network,
                action_space=self.action_space,
                observation_space=self.observation_space,
                num_atoms=num_atoms,
                v_min=v_min,
                v_max=v_max,
                eps_training=initial_eps,
                eps_inference=final_eps,
            )
            # Move policy to device (important: C51Policy has support tensor that needs to be on GPU)
            self.policy = self.policy.to(self.device)

            self.algorithm = RainbowDQN(
                policy=self.policy,
                optim=AdamOptimizerFactory(lr=lr),
                gamma=gamma,
                n_step_return_horizon=n_step,
                target_update_freq=target_update,
            )
            self.logger.info("Rainbow DQN Algorithm created (Tianshou 2.0):")
            self.logger.info(f"  num_atoms: {num_atoms}")
            self.logger.info(f"  v_min: {v_min}, v_max: {v_max}")
            if n_step > 1:
                self.logger.info(f"  n-step returns: {n_step}")

        else:
            raise ValueError(
                f"Unknown algorithm type: {algo_type}. "
                f"Supported: dqn, dueling_dqn, c51, rainbow"
            )

        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Gamma: {gamma}")
        self.logger.info(f"  Target update freq: {target_update}")
        self.logger.info(f"  Initial epsilon: {initial_eps}")
        self.logger.info(f"  Final epsilon: {final_eps}")

    def _build_collectors(self, test_env: Optional[gym.Env] = None) -> None:
        """Build Tianshou collectors and buffer.

        Supports both standard VectorReplayBuffer and PrioritizedVectorReplayBuffer
        based on algorithm configuration.

        Args:
            test_env: Optional separate test environment.
        """
        buffer_size = self.training_config.get("buffer_size", 10000)

        # Wrap environment if not vectorized
        if not hasattr(self.env, "envs"):
            self.train_envs = DummyVectorEnv([lambda: self.env])
            num_envs = 1
        else:
            self.train_envs = self.env
            num_envs = len(self.env.envs)

        # Check if PER is enabled
        algo_config = self.config.get("algorithm", {})
        per_config = algo_config.get("prioritized_replay", {})
        use_per = per_config.get("enabled", False)

        # Rainbow typically requires PER
        if self.algo_type == "rainbow" and not use_per:
            self.logger.warning("Rainbow DQN typically uses PER. Consider enabling it.")

        # Create replay buffer
        if use_per:
            alpha = per_config.get("alpha", 0.6)
            beta = per_config.get("beta", 0.4)

            self.buffer = PrioritizedVectorReplayBuffer(
                total_size=buffer_size,
                buffer_num=num_envs,
                alpha=alpha,
                beta=beta,
                stack_num=1,
            )

            # Store PER config for beta annealing
            self.per_config = per_config
            self.use_per = True

            self.logger.info("PrioritizedVectorReplayBuffer created:")
            self.logger.info(f"  alpha: {alpha}")
            self.logger.info(f"  beta: {beta}")
        else:
            self.buffer = VectorReplayBuffer(
                total_size=buffer_size,
                buffer_num=num_envs,
                stack_num=1,
            )
            self.use_per = False

        # Create training collector (Tianshou 2.0: pass algorithm, not policy)
        self.train_collector = Collector(
            policy=self.algorithm,
            env=self.train_envs,
            buffer=self.buffer,
            exploration_noise=True,
        )

        # Create test collector (optional)
        if test_env is not None:
            if not hasattr(test_env, "envs"):
                test_envs = DummyVectorEnv([lambda: test_env])
            else:
                test_envs = test_env

            self.test_collector = Collector(
                policy=self.algorithm,
                env=test_envs,
            )
        else:
            self.test_collector = None

        self.logger.info("Collectors created:")
        self.logger.info(f"  Buffer size: {buffer_size}")
        self.logger.info(f"  Buffer type: {'PER' if use_per else 'Uniform'}")
        self.logger.info(f"  Number of train envs: {num_envs}")
        self.logger.info(f"  Test collector: {'enabled' if test_env else 'disabled'}")

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train the agent.

        This method implements a custom training loop that provides
        fine-grained control over logging and callbacks. It follows
        Tianshou patterns while enabling detailed metric tracking.

        Args:
            total_timesteps: Total environment steps. If None, uses
                config value.
            callback: Additional callback to add during training.

        Returns:
            Dict with training statistics:
                - best_reward: Best evaluation reward
                - train_time: Total training time
                - num_episodes: Total episodes completed
                - final_loss: Final training loss
        """
        import time

        start_time = time.time()

        total_timesteps = total_timesteps or self.training_config.get(
            "total_timesteps", 10000
        )

        # Training parameters
        step_per_collect = self.training_config.get("step_per_collect", 1)
        batch_size = self.training_config.get("batch_size", 32)
        start_timesteps = self.training_config.get("start_timesteps", 1000)
        log_interval = self.training_config.get("log_interval", 100)

        # Exploration parameters
        exploration_cfg = self.training_config.get("exploration", {})
        initial_eps = exploration_cfg.get("initial_eps", 1.0)
        final_eps = exploration_cfg.get("final_eps", 0.05)
        exploration_fraction = exploration_cfg.get("fraction", 0.5)

        # Combine callbacks
        all_callbacks = self.callbacks.copy()
        if callback is not None:
            all_callbacks.append(callback)

        # Notify training start
        for cb in all_callbacks:
            if hasattr(cb, "on_training_start"):
                cb.on_training_start(self)

        self.logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.logger.info(f"  Step per collect: {step_per_collect}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Start timesteps: {start_timesteps}")

        # Reset collector
        self.train_collector.reset()

        # Pre-collect random data
        if start_timesteps > 0:
            self.train_collector.collect(n_step=start_timesteps, random=True)
            self.logger.info(f"Pre-collected {start_timesteps} random transitions")

        # Training loop
        episode_rewards: List[float] = []
        best_reward = float("-inf")
        step_count = 0

        # Check if network has NoisyNet enabled
        use_noisy = getattr(self.network, "use_noisy_layers", False)
        if use_noisy:
            self.logger.info("NoisyNet enabled - resetting noise before each step")

        while step_count < total_timesteps:
            # Reset noise for NoisyNet exploration (if enabled)
            if use_noisy and hasattr(self.network, "reset_noise"):
                self.network.reset_noise()

            # Collect data
            collect_result = self.train_collector.collect(n_step=step_per_collect)
            step_count += collect_result.n_collected_steps
            self.num_timesteps = step_count

            # Update exploration rate (epsilon decay)
            progress = min(step_count / (total_timesteps * exploration_fraction), 1.0)
            eps = initial_eps - (initial_eps - final_eps) * progress
            self.policy.set_eps_training(eps)  # Tianshou 2.0: set_eps_training
            self.exploration_rate = eps

            # Update PER beta (importance sampling weight annealing)
            if self.use_per:
                beta_start = self.per_config.get("beta", 0.4)
                beta_final = self.per_config.get("beta_final", 1.0)
                beta_anneal_steps = (
                    self.per_config.get("beta_anneal_step") or total_timesteps
                )
                beta_progress = min(step_count / beta_anneal_steps, 1.0)
                new_beta = beta_start + (beta_final - beta_start) * beta_progress
                self.buffer.set_beta(new_beta)

            # Track episode rewards
            if collect_result.n_collected_episodes > 0:
                episode_rewards.extend(collect_result.returns)
                self._last_episode_rewards = list(collect_result.returns)
                self.num_episodes += collect_result.n_collected_episodes

                # Track best reward
                current_best = max(collect_result.returns)
                if current_best > best_reward:
                    best_reward = current_best
            else:
                self._last_episode_rewards = []

            # Notify callbacks of collect end
            for cb in all_callbacks:
                if hasattr(cb, "on_collect_end"):
                    cb.on_collect_end(collect_result, self)

            # Train algorithm (Tianshou 2.0: context uses POLICY, update uses ALGORITHM)
            if len(self.buffer) >= batch_size:
                # Reset noise for NoisyNet before training (if enabled)
                if use_noisy and hasattr(self.network, "reset_noise"):
                    self.network.reset_noise()

                with policy_within_training_step(
                    self.policy
                ):  # Pass policy, not algorithm
                    train_result = self.algorithm.update(  # Call algorithm.update
                        self.buffer,  # buffer first
                        sample_size=batch_size,  # sample_size second
                    )

                loss = train_result.loss
                self._recent_losses.append(loss)

                # Keep only recent losses
                if len(self._recent_losses) > 100:
                    self._recent_losses = self._recent_losses[-100:]

                # Notify callbacks of train step end
                train_result_dict = {
                    "loss": loss,
                    "train_time": train_result.train_time,
                }
                for cb in all_callbacks:
                    if hasattr(cb, "on_train_step_end"):
                        cb.on_train_step_end(train_result_dict, self)

            # Log progress
            if step_count % log_interval == 0 or step_count >= total_timesteps:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                avg_loss = (
                    np.mean(self._recent_losses[-10:]) if self._recent_losses else 0.0
                )

                verbose = self.system_config.get("verbose", 1)
                if verbose >= 1:
                    self.logger.info(
                        f"Step {step_count}/{total_timesteps} | "
                        f"Episodes: {self.num_episodes} | "
                        f"Avg Reward (last 10): {avg_reward:.2f} | "
                        f"Avg Loss (last 10): {avg_loss:.4f} | "
                        f"Epsilon: {eps:.3f}"
                    )

        # Notify training end
        for cb in all_callbacks:
            if hasattr(cb, "on_training_end"):
                cb.on_training_end(self)

        train_time = time.time() - start_time
        self.logger.info(f"Training complete!")
        self.logger.info(f"  Total episodes: {self.num_episodes}")
        self.logger.info(f"  Best reward: {best_reward:.3f}")
        self.logger.info(f"  Training time: {train_time:.1f}s")

        return {
            "best_reward": best_reward,
            "train_time": train_time,
            "num_episodes": self.num_episodes,
            "final_loss": self._recent_losses[-1] if self._recent_losses else 0.0,
            "episode_rewards": episode_rewards,
        }

    def _get_checkpoint_path(self, name: str) -> Path:
        """Get checkpoint save path.

        Args:
            name: Checkpoint name (e.g., 'best', 'final', 'step_1000').

        Returns:
            Path to checkpoint file.
        """
        output_dir = Path(
            self.config.get("paths", {}).get("output_dir", "artifacts/training")
        )
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / name

    def save(self, path: Union[str, Path]) -> None:
        """Save trainer state to checkpoint.

        Saves:
            - Network weights
            - Optimizer state
            - Training state (timesteps, exploration rate)
            - Configuration

        Args:
            path: Path to save checkpoint (without extension).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "exploration_rate": self.exploration_rate,
            "config": self.config,
            "network_type": self.network_config.get("type", "drqn"),
        }

        save_path = path.with_suffix(".pth")
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "TianshouTrainer":
        """Load trainer from checkpoint.

        Args:
            path: Path to checkpoint file (without extension).
            env: Environment for training.
            config: Optional config override. If None, uses saved config.
            **kwargs: Additional arguments for TianshouTrainer.

        Returns:
            Loaded TianshouTrainer instance.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        load_path = Path(path).with_suffix(".pth")

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location="cpu")

        # Use saved config if not provided
        if config is None:
            config = checkpoint.get("config", {})

        # Create trainer
        trainer = cls(env=env, config=config, **kwargs)

        # Restore state
        trainer.network.load_state_dict(checkpoint["network"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.num_timesteps = checkpoint.get("num_timesteps", 0)
        trainer.num_episodes = checkpoint.get("num_episodes", 0)
        trainer.exploration_rate = checkpoint.get("exploration_rate", 0.05)

        trainer.logger.info(f"Checkpoint loaded from {load_path}")
        trainer.logger.info(f"  Timesteps: {trainer.num_timesteps}")
        trainer.logger.info(f"  Episodes: {trainer.num_episodes}")
        trainer.logger.info(f"  Exploration rate: {trainer.exploration_rate:.3f}")

        return trainer

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the current policy.

        Args:
            num_episodes: Number of evaluation episodes.
            deterministic: Whether to use greedy policy.

        Returns:
            Dict with evaluation metrics:
                - reward_mean: Mean episode reward
                - reward_std: Std of episode rewards
                - length_mean: Mean episode length
        """
        if self.test_collector is None:
            # Create temporary test collector using train envs (Tianshou 2.0: use algorithm)
            test_collector = Collector(
                policy=self.algorithm,
                env=self.train_envs,
            )
        else:
            test_collector = self.test_collector

        # Reset collector before evaluation
        test_collector.reset()

        # Set evaluation mode (Tianshou 2.0: use eps_training/eps_inference)
        original_eps = self.policy.eps_training
        if deterministic:
            self.policy.set_eps_inference(0.0)

        # Collect episodes
        result = test_collector.collect(n_episode=num_episodes)

        # Restore exploration rate
        self.policy.set_eps_training(original_eps)

        return {
            "reward_mean": float(np.mean(result.returns)),
            "reward_std": float(np.std(result.returns)),
            "length_mean": float(np.mean(result.lens)),
        }

    @property
    def last_episode_rewards(self) -> List[float]:
        """Get rewards from the last collection."""
        return self._last_episode_rewards

    @property
    def recent_losses(self) -> List[float]:
        """Get recent training losses."""
        return self._recent_losses
