"""Tianshou DRQN Adapter.

This module provides a standalone DRQN implementation using Tianshou 2.0,
with a simple learn/save/load interface for easy integration.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.algorithm import DQN
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.torch_utils import policy_within_training_step

from ..utils.logging.logging_manager import get_logger


class DRQN(nn.Module):
    """Deep Recurrent Q-Network with LSTM for sequential decision making.

    Architecture:
        Input (observation) → Embedding → LSTM → FC layers → Q-values

    This implementation matches the legacy config schema:
        - embedding_params: [embedding_dim]
        - lstm_layer_params: [lstm_hidden_size]
        - output_fc_layer_params: [fc1_size, fc2_size, ...]
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_shape: int,
        embedding_dim: int = 64,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 1,
        fc_hidden_sizes: Tuple[int, ...] = (128, 64),
        activation_fn: nn.Module = nn.ReLU,
        device: str = "cpu",
    ):
        """Initialize DRQN network.

        Args:
            state_shape: Dimension of observation space
            action_shape: Number of discrete actions
            embedding_dim: Embedding layer output dimension
            lstm_hidden_size: LSTM hidden state size
            num_lstm_layers: Number of stacked LSTM layers
            fc_hidden_sizes: Sizes of fully-connected layers after LSTM
            activation_fn: Activation function class (e.g., nn.ReLU, nn.ELU)
            device: Device to use (cpu/cuda)
        """
        super().__init__()
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        # Handle tuple state_shape
        if isinstance(state_shape, tuple):
            state_dim = state_shape[0]
        else:
            state_dim = state_shape

        # Embedding layer (project observation to embedding space)
        self.embedding = nn.Linear(state_dim, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Fully-connected layers after LSTM
        fc_layers = []
        input_dim = lstm_hidden_size
        for hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(input_dim, hidden_size))
            fc_layers.append(activation_fn())
            input_dim = hidden_size

        # Output layer (Q-values)
        fc_layers.append(nn.Linear(input_dim, action_shape))

        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through DRQN.

        Args:
            obs: Observations [batch_size, seq_len, state_dim] or [batch_size, state_dim]
            state: LSTM hidden state dict with 'h' and 'c' tensors
            info: Additional info (unused)

        Returns:
            Tuple of (Q-values, new_hidden_state)
        """
        # Convert observation to tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Handle batched input
        if obs.dim() == 2:  # [batch_size, state_dim]
            obs = obs.unsqueeze(1)  # [batch_size, 1, state_dim] - add sequence dim

        batch_size = obs.shape[0]

        # Embedding
        embedded = self.embedding(obs)  # [batch_size, seq_len, embedding_dim]

        # LSTM forward pass
        if state is None:
            # Initialize hidden state
            h = torch.zeros(
                self.num_lstm_layers,
                batch_size,
                self.lstm_hidden_size,
                device=self.device,
            )
            c = torch.zeros(
                self.num_lstm_layers,
                batch_size,
                self.lstm_hidden_size,
                device=self.device,
            )
        else:
            h = state["h"]
            c = state["c"]

        lstm_out, (h_new, c_new) = self.lstm(embedded, (h, c))

        # Take last timestep output
        last_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]

        # Fully-connected layers
        q_values = self.fc(last_out)  # [batch_size, action_shape]

        # Return Q-values and new hidden state
        new_state = {"h": h_new, "c": c_new}

        return q_values, new_state


class TianshouDRQNAdapter:
    """Tianshou DRQN adapter with a simple training interface.

    This class provides a straightforward interface (learn, save, load) for
    training Deep Recurrent Q-Networks using Tianshou's DQN algorithm.
    It handles network creation, training loop, and checkpoint management.

    Example:
        >>> config = load_config("agent_config.yaml")
        >>> env = make_music_env(config)
        >>> model = TianshouDRQNAdapter(env=env, config=config)
        >>> model.learn(total_timesteps=10000)
        >>> model.save("checkpoints/model.pth")
    """

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update_interval: int = 1000,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_fraction: float = 0.1,
        verbose: int = 0,
        device: str = "auto",
    ):
        """Initialize Tianshou DRQN Adapter.

        Args:
            env: Gymnasium environment (can be single or vectorized)
            config: Full configuration dictionary
            policy_kwargs: Network architecture parameters
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            batch_size: Training batch size
            buffer_size: Replay buffer capacity
            target_update_interval: Target network update frequency
            exploration_initial_eps: Initial epsilon for epsilon-greedy exploration
            exploration_final_eps: Final epsilon for epsilon-greedy exploration
            exploration_fraction: Fraction of training for epsilon decay
            verbose: Verbosity level
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.logger = get_logger("tianshou_adapter")
        self.env = env
        self.config = config
        self.verbose = verbose

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update_interval = target_update_interval
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        # Device setup
        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available() and config.get("enable_gpu", False)
                else "cpu"
            )
        else:
            self.device = device

        self.logger.info(f"Tianshou DRQN Adapter initialized on device: {self.device}")

        # Build components
        self._build_network(policy_kwargs or {})
        self._build_algorithm()
        self._build_collector()

        # Training state
        self.num_timesteps = 0
        self.exploration_rate = exploration_initial_eps

    def _build_network(self, policy_kwargs: Dict[str, Any]) -> None:
        """Build DRQN network from config."""
        model_config = self.config.get("model", {})

        # Extract state and action shapes
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

        # Network architecture from config
        # Priority: model.recurrent (Phase 5 new format) > legacy format (Phase 1-2)
        recurrent_config = model_config.get("recurrent", {})

        if recurrent_config and recurrent_config.get("enabled", True):
            # Phase 5+ config format: model.recurrent
            embedding_dim = recurrent_config.get("embedding_dim", 64)
            lstm_hidden_size = recurrent_config.get("lstm_hidden_size", 128)
            num_lstm_layers = recurrent_config.get("lstm_num_layers", 1)
            fc_hidden_sizes = tuple(recurrent_config.get("fc_hidden_sizes", [128, 64]))
            dropout = recurrent_config.get("dropout", 0.0)
            self.logger.info("Using Phase 5+ config format: model.recurrent")
        else:
            # Legacy config format (Phase 1-2): embedding_params, lstm_layer_params
            embedding_params = model_config.get("embedding_params", [64])
            embedding_dim = embedding_params[0] if embedding_params else 64

            lstm_params = model_config.get("lstm_layer_params", [128])
            lstm_hidden_size = lstm_params[0] if lstm_params else 128

            output_fc_params = model_config.get("output_fc_layer_params", [128, 64])
            fc_hidden_sizes = tuple(output_fc_params) if output_fc_params else (128, 64)

            num_lstm_layers = 1  # Default for legacy format
            dropout = 0.0  # No dropout in legacy format
            self.logger.info(
                "Using legacy config format: embedding_params, lstm_layer_params"
            )

        # Activation function
        activation_fn_name = model_config.get("activation_fn", "relu").lower()
        activation_fn_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leakyrelu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
        }
        activation_fn = activation_fn_map.get(activation_fn_name, nn.ReLU)

        # Override with policy_kwargs if provided
        if "net_arch" in policy_kwargs:
            fc_hidden_sizes = tuple(policy_kwargs["net_arch"])
        if "activation_fn" in policy_kwargs:
            activation_fn = policy_kwargs["activation_fn"]

        # Create Q-network
        self.q_network = DRQN(
            state_shape=state_shape,
            action_shape=action_shape,
            embedding_dim=embedding_dim,
            lstm_hidden_size=lstm_hidden_size,
            num_lstm_layers=num_lstm_layers,
            fc_hidden_sizes=fc_hidden_sizes,
            activation_fn=activation_fn,
            device=self.device,
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(
            p.numel() for p in self.q_network.parameters() if p.requires_grad
        )

        self.logger.info(f"DRQN Network Architecture:")
        self.logger.info(f"  Embedding dim: {embedding_dim}")
        self.logger.info(f"  LSTM hidden size: {lstm_hidden_size}")
        self.logger.info(f"  LSTM num layers: {num_lstm_layers}")
        self.logger.info(f"  FC layers: {fc_hidden_sizes}")
        self.logger.info(f"  Activation: {activation_fn.__name__}")
        self.logger.info(f"  Dropout: {dropout}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")

    def _build_algorithm(self) -> None:
        """Build Tianshou DQN algorithm with separate policy and algorithm.

        Tianshou 2.0 separates:
        - Policy: Action selection, epsilon-greedy exploration
        - Algorithm: Learning, target network updates, optimization
        """
        # Create optimizer (kept for checkpoint saving compatibility)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate
        )

        # Step 1: Create Policy (action selection, epsilon-greedy)
        self.policy = DiscreteQLearningPolicy(
            model=self.q_network,
            action_space=self.action_space,
            observation_space=self.observation_space,
            eps_training=self.exploration_initial_eps,
            eps_inference=self.exploration_final_eps,
        )

        # Step 2: Create Algorithm (learning, target network)
        self.algorithm = DQN(
            policy=self.policy,
            optim=AdamOptimizerFactory(lr=self.learning_rate),
            gamma=self.gamma,
            n_step_return_horizon=1,
            target_update_freq=self.target_update_interval,
        )

        self.logger.info("DQN Algorithm created (Tianshou 2.0):")
        self.logger.info(f"  Learning rate: {self.learning_rate}")
        self.logger.info(f"  Gamma: {self.gamma}")
        self.logger.info(f"  Target update freq: {self.target_update_interval}")

    def _build_collector(self) -> None:
        """Build Tianshou collector and replay buffer."""
        # Wrap environment if not already vectorized
        if not hasattr(self.env, "envs"):
            self.train_envs = DummyVectorEnv([lambda: self.env])
            num_envs = 1
        else:
            self.train_envs = self.env
            num_envs = len(self.env.envs)

        # Create replay buffer
        self.buffer = VectorReplayBuffer(
            total_size=self.buffer_size,
            buffer_num=num_envs,
        )

        # Create collector (Tianshou 2.0: pass algorithm, not policy)
        self.train_collector = Collector(
            policy=self.algorithm,
            env=self.train_envs,
            buffer=self.buffer,
            exploration_noise=True,
        )

        self.logger.info("Collector created:")
        self.logger.info(f"  Buffer size: {self.buffer_size}")
        self.logger.info(f"  Number of envs: {num_envs}")

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[Callable, List[Callable]]] = None,
        log_interval: int = 100,
        **kwargs,
    ) -> "TianshouDRQNAdapter":
        """Train the policy using Tianshou's training loop.

        Runs the DQN training loop with epsilon-greedy exploration,
        experience replay, and periodic target network updates.

        Args:
            total_timesteps: Total number of environment steps to train
            callback: Optional callback(s) to call during training
            log_interval: Logging frequency (steps)
            **kwargs: Additional arguments (unused, for interface compatibility)

        Returns:
            self (for method chaining)
        """
        self.logger.info(f"Starting training for {total_timesteps} timesteps...")

        # Prepare callbacks
        from ..utils.callbacks.tianshou_callbacks import (
            CallbackList,
            BaseTianshouCallback,
        )

        if callback is None:
            callbacks = None
        elif isinstance(callback, list):
            # Filter for Tianshou-compatible callbacks
            tianshou_callbacks = []
            for cb in callback:
                if isinstance(cb, BaseTianshouCallback):
                    tianshou_callbacks.append(cb)
                else:
                    # Non-Tianshou callback - skip with warning
                    self.logger.warning(
                        f"Skipping incompatible callback {cb.__class__.__name__} "
                        "(must extend BaseTianshouCallback)"
                    )
            callbacks = CallbackList(tianshou_callbacks) if tianshou_callbacks else None
        elif isinstance(callback, BaseTianshouCallback):
            from ..utils.callbacks.tianshou_callbacks import CallbackList

            callbacks = CallbackList([callback])
        else:
            # Non-Tianshou callback - skip with warning
            self.logger.warning(
                f"Skipping incompatible callback {callback.__class__.__name__} "
                "(must extend BaseTianshouCallback)"
            )
            callbacks = None

        # Call on_training_start
        if callbacks:
            callbacks.on_training_start(self)

        # Reset collector before first collect (required in Tianshou 1.x)
        self.train_collector.reset()

        # Pre-collect random data
        self.train_collector.collect(n_step=self.batch_size * 4, random=True)
        self.logger.info(f"Pre-collected {self.batch_size * 4} random transitions")

        # Training loop
        episode_rewards = []
        losses = []
        step_count = 0

        # Track last episode rewards for LR scheduler
        self._last_episode_rewards = []

        while step_count < total_timesteps:
            # Collect data
            collect_result = self.train_collector.collect(n_step=1)
            step_count += collect_result.n_collected_steps
            self.num_timesteps = step_count

            # Update exploration rate (epsilon decay)
            progress = min(
                step_count / (total_timesteps * self.exploration_fraction), 1.0
            )
            self.exploration_rate = (
                self.exploration_initial_eps
                - (self.exploration_initial_eps - self.exploration_final_eps) * progress
            )
            # Tianshou 2.0: use set_eps_training
            self.policy.set_eps_training(self.exploration_rate)

            # Track episode rewards
            if collect_result.n_collected_episodes > 0:
                episode_rewards.extend(collect_result.returns)
                # Store for LR scheduler
                self._last_episode_rewards = list(collect_result.returns)
            else:
                self._last_episode_rewards = []

            # Call on_collect_end
            if callbacks:
                callbacks.on_collect_end(collect_result, self)

            # Train algorithm (Tianshou 2.0: context uses POLICY, update uses ALGORITHM)
            if len(self.buffer) >= self.batch_size:
                with policy_within_training_step(
                    self.policy
                ):  # Pass policy, not algorithm
                    train_result = self.algorithm.update(  # Call algorithm.update
                        self.buffer,  # buffer first
                        sample_size=self.batch_size,  # sample_size second
                    )
                # Tianshou 2.0 returns DQNTrainingStats dataclass with .loss attribute
                losses.append(train_result.loss)

                # Call on_train_step_end (convert to dict for callback compatibility)
                if callbacks:
                    train_result_dict = {
                        "loss": train_result.loss,
                        "train_time": train_result.train_time,
                    }
                    callbacks.on_train_step_end(train_result_dict, self)

            # Log progress
            if step_count % log_interval == 0 or step_count >= total_timesteps:
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                avg_loss = np.mean(losses[-10:]) if losses else 0.0

                if self.verbose >= 1:
                    self.logger.info(
                        f"Step {step_count}/{total_timesteps} | "
                        f"Episodes: {len(episode_rewards)} | "
                        f"Avg Reward (last 10): {avg_reward:.2f} | "
                        f"Avg Loss (last 10): {avg_loss:.4f} | "
                        f"Epsilon: {self.exploration_rate:.3f}"
                    )

        # Call on_training_end
        if callbacks:
            callbacks.on_training_end(self)

        self.logger.info(f"Training complete! Total episodes: {len(episode_rewards)}")
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Save policy weights to disk.

        Args:
            path: Path to save checkpoint (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save policy state dict
        save_path = path.with_suffix(".pth")
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
                "exploration_rate": self.exploration_rate,
            },
            save_path,
        )

        self.logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: gym.Env,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "TianshouDRQNAdapter":
        """Load policy weights from disk.

        Args:
            path: Path to checkpoint file (without extension)
            env: Gymnasium environment
            config: Configuration dictionary (required for network architecture)
            **kwargs: Additional arguments for adapter initialization

        Returns:
            Loaded TianshouDRQNAdapter instance
        """
        if config is None:
            raise ValueError("config is required for loading Tianshou models")

        # Create adapter instance
        adapter = cls(env=env, config=config, **kwargs)

        # Load checkpoint
        load_path = Path(path).with_suffix(".pth")
        checkpoint = torch.load(load_path, map_location=adapter.device)

        # Restore state
        adapter.q_network.load_state_dict(checkpoint["q_network"])
        adapter.optimizer.load_state_dict(checkpoint["optimizer"])
        adapter.num_timesteps = checkpoint.get("num_timesteps", 0)
        adapter.exploration_rate = checkpoint.get("exploration_rate", 0.05)

        adapter.logger.info(f"Model loaded from {load_path}")
        adapter.logger.info(f"  Timesteps: {adapter.num_timesteps}")
        adapter.logger.info(f"  Exploration rate: {adapter.exploration_rate:.3f}")

        return adapter

    @property
    def logger_attr(self):
        """Placeholder logger attribute for interface compatibility."""
        return None
