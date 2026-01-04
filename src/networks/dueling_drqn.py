"""Dueling DRQN network for value/advantage decomposition.

This module provides a Dueling architecture variant of the DRQN,
which separates state value estimation from action advantage estimation.
This helps learning when many actions have similar values.

Architecture:
    Observation → Embedding → LSTM → Split
                                       ├─> Value Stream → V(s) [1 value]
                                       └─> Advantage Stream → A(s,a) [n_actions values]

    Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

Classes:
    DuelingDRQN: Dueling LSTM-based Q-network for Tianshou DQNPolicy.

Example:
    >>> network = DuelingDRQN(
    ...     state_shape=(32,),
    ...     action_shape=10,
    ...     embedding_dim=64,
    ...     lstm_hidden_size=256,
    ... )
    >>> obs = torch.randn(1, 32)
    >>> q_values, state = network(obs)
    >>> q_values.shape
    torch.Size([1, 10])
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import BaseQNetwork
from .noisy_linear import NoisyLinear
from src.utils.lstm_diagnostics import (
    log_state_metrics,
    warn_state_mismatch,
)
from src.utils.hidden_state_utils import HiddenStateHandler


class DuelingDRQN(BaseQNetwork):
    """Dueling Deep Recurrent Q-Network with value/advantage decomposition.

    This network separates the Q-value computation into two streams:
    - Value stream: estimates V(s), the value of being in a state
    - Advantage stream: estimates A(s,a), the advantage of each action

    The final Q-values are computed as:
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

    The mean subtraction ensures identifiability and helps with learning
    by centering the advantages around zero.

    Attributes:
        embedding (nn.Linear): Embedding layer.
        lstm (nn.LSTM): LSTM layer for temporal processing.
        value_stream (nn.Sequential): Value estimation head.
        advantage_stream (nn.Sequential): Advantage estimation head.
        lstm_hidden_size (int): LSTM hidden dimension.
        num_lstm_layers (int): Number of stacked LSTM layers.

    Example:
        >>> dueling = DuelingDRQN(
        ...     state_shape=(32,),
        ...     action_shape=10,
        ...     embedding_dim=64,
        ...     lstm_hidden_size=256,
        ... )
        >>> obs = torch.randn(4, 32)
        >>> q_values, state = dueling(obs)
        >>> q_values.shape
        torch.Size([4, 10])
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_shape: int,
        embedding_dim: int = 64,
        lstm_hidden_size: int = 256,
        num_lstm_layers: int = 1,
        fc_hidden_sizes: Tuple[int, ...] = (128, 64),
        activation_fn: type = nn.ELU,
        dropout: float = 0.2,
        device: str = "cpu",
        enable_diagnostics: bool = True,
        diagnostic_log_interval: int = 1000,
        use_noisy_layers: bool = False,
        noisy_sigma: float = 0.5,
    ):
        """Initialize Dueling DRQN network.

        Args:
            state_shape: Observation space dimension(s).
            action_shape: Number of discrete actions.
            embedding_dim: Embedding layer output dimension.
            lstm_hidden_size: LSTM hidden state size.
            num_lstm_layers: Number of stacked LSTM layers.
            fc_hidden_sizes: Sizes of fully-connected layers in each stream.
            activation_fn: Activation function class (e.g., nn.ELU).
            dropout: Dropout rate for regularization.
            device: Device for tensor operations.
            enable_diagnostics: Whether to enable LSTM diagnostic logging.
            diagnostic_log_interval: Log metrics every N forward calls.
            use_noisy_layers: Whether to use NoisyLinear for exploration.
            noisy_sigma: Initial noise scale for NoisyLinear (default: 0.5).
        """
        super().__init__(state_shape, action_shape, embedding_dim, device)

        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.use_noisy_layers = use_noisy_layers
        self.noisy_sigma = noisy_sigma

        # Diagnostic configuration (Phase 1: LSTM Hidden State Fix)
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_log_interval = diagnostic_log_interval

        # Phase 2: Hidden state handler for enhanced state management
        self._state_handler = HiddenStateHandler(
            num_layers=num_lstm_layers,
            hidden_size=lstm_hidden_size,
            device=device,
        )

        # Calculate input dimension from state_shape
        if isinstance(state_shape, tuple):
            if len(state_shape) == 2:
                state_dim = state_shape[-1]
            else:
                state_dim = state_shape[0]
        else:
            state_dim = state_shape

        # Embedding layer
        self.embedding = nn.Linear(state_dim, embedding_dim)

        # LSTM layer for temporal processing
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        # Helper to create Linear or NoisyLinear based on config
        def make_linear(in_features: int, out_features: int) -> nn.Module:
            if self.use_noisy_layers:
                return NoisyLinear(
                    in_features, out_features, sigma_init=self.noisy_sigma
                )
            return nn.Linear(in_features, out_features)

        # Value stream: outputs single value V(s)
        value_layers = [
            make_linear(lstm_hidden_size, fc_hidden_sizes[0]),
            activation_fn(),
        ]
        if dropout > 0.0:
            value_layers.append(nn.Dropout(dropout))
        value_layers.append(make_linear(fc_hidden_sizes[0], 1))
        self.value_stream = nn.Sequential(*value_layers)

        # Advantage stream: outputs advantage for each action A(s,a)
        advantage_layers = [
            make_linear(lstm_hidden_size, fc_hidden_sizes[0]),
            activation_fn(),
        ]
        if dropout > 0.0:
            advantage_layers.append(nn.Dropout(dropout))
        advantage_layers.append(make_linear(fc_hidden_sizes[0], action_shape))
        self.advantage_stream = nn.Sequential(*advantage_layers)

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers.

        Should be called at the start of each forward pass during training
        when using NoisyNet exploration. No-op if use_noisy_layers=False.
        """
        if not self.use_noisy_layers:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def _apply(self, fn):
        """Override to update device attribute when network is moved."""
        super()._apply(fn)
        # Update device tracking for hidden state handler
        if hasattr(self, "_state_handler"):
            try:
                param = next(self.parameters())
                new_device = str(param.device)
                self.device = new_device
                self._state_handler.device = new_device
            except StopIteration:
                pass
        return self

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[
            Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
        ] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with dueling aggregation.

        Args:
            obs: Observations of shape:
                - (batch, feature_dim): Single timestep per sample
                - (batch, seq_len, feature_dim): Sequence of observations
            state: LSTM hidden state, can be:
                - None: Zero-initialized hidden state
                - Dict with 'h', 'c': Single state (may be broadcast)
                - List[Dict]: List of states for batch stacking
            info: Additional information (unused).

        Returns:
            Tuple of:
                - q_values: Q-values of shape (batch, n_actions)
                - new_state: Dict with updated 'h' and 'c' tensors
        """
        # Convert numpy to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Ensure tensor is on correct device
        if obs.device != torch.device(self.device):
            obs = obs.to(self.device)

        # Handle batched input: add sequence dimension if needed
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        batch_size = obs.shape[0]

        # Embedding
        embedded = self.embedding(obs)

        # PHASE 2: Enhanced hidden state preparation
        # Supports: None, dict, list of dicts (batch sampling)
        h, c, handling_mode = self._state_handler.prepare_state(state, batch_size)

        # PHASE 1 DIAGNOSTIC: Log mismatch and track metrics
        if self.enable_diagnostics:
            if handling_mode == "fallback_zero" and state is not None:
                warn_state_mismatch(
                    network_name=self.__class__.__name__,
                    expected_batch=(
                        state["h"].shape[1] if isinstance(state, dict) else "list"
                    ),
                    actual_batch=batch_size,
                    log_level="debug",
                )
            log_state_metrics(
                h,
                c,
                network_name=self.__class__.__name__,
                log_every_n=self.diagnostic_log_interval,
            )

        # LSTM forward pass
        lstm_out, (h_new, c_new) = self.lstm(embedded, (h, c))

        # Take last timestep output
        last_out = lstm_out[:, -1, :]

        # Dueling streams
        value = self.value_stream(last_out)  # (batch, 1)
        advantages = self.advantage_stream(last_out)  # (batch, n_actions)

        # Dueling aggregation: Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        new_state = {"h": h_new, "c": c_new}

        return q_values, new_state

    def get_initial_state(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Get initial hidden state for LSTM.

        Args:
            batch_size: Number of parallel environments.

        Returns:
            Dict with 'h' and 'c' zero-initialized tensors.
        """
        return {
            "h": torch.zeros(
                self.num_lstm_layers,
                batch_size,
                self.lstm_hidden_size,
                device=self.device,
            ),
            "c": torch.zeros(
                self.num_lstm_layers,
                batch_size,
                self.lstm_hidden_size,
                device=self.device,
            ),
        }

    @property
    def is_recurrent(self) -> bool:
        """DuelingDRQN is recurrent (maintains hidden state)."""
        return True

    def forward_with_burnin(
        self,
        obs: torch.Tensor,
        burn_in_length: int,
        initial_hidden: Optional[Dict[str, torch.Tensor]] = None,
        return_all_q: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with burn-in for sequence-based training.

        Implements R2D2-style burn-in: burn-in portion runs without gradients,
        training portion runs with gradients for Q-value computation.

        Args:
            obs: Observation sequence (batch, seq_len, obs_dim)
            burn_in_length: Number of initial steps for burn-in (no gradients)
            initial_hidden: Initial hidden state from replay buffer (optional)
            return_all_q: If True, return Q-values for all training steps

        Returns:
            Tuple of (q_values, final_hidden)

        Raises:
            ValueError: If sequence length <= burn_in_length
        """
        seq_len = obs.shape[1]

        if seq_len <= burn_in_length:
            raise ValueError(
                f"Sequence length {seq_len} must be greater than "
                f"burn_in_length {burn_in_length}"
            )

        burn_in_obs = obs[:, :burn_in_length]
        train_obs = obs[:, burn_in_length:]
        train_len = train_obs.shape[1]

        with torch.no_grad():
            _, burn_in_hidden = self.forward(burn_in_obs, state=initial_hidden)

        warmed_hidden = {
            "h": burn_in_hidden["h"].detach(),
            "c": burn_in_hidden["c"].detach(),
        }

        if return_all_q:
            q_values_list = []
            hidden = warmed_hidden
            for t in range(train_len):
                step_obs = train_obs[:, t : t + 1]
                q, hidden = self.forward(step_obs, state=hidden)
                q_values_list.append(q)
            q_values = torch.stack(q_values_list, dim=1)
            final_hidden = hidden
        else:
            q_values, final_hidden = self.forward(train_obs, state=warmed_hidden)

        return q_values, final_hidden
