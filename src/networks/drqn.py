"""Deep Recurrent Q-Network (DRQN) implementation.

This module provides an LSTM-based Q-network for sequential decision making.
The DRQN maintains a hidden state across timesteps, enabling temporal
reasoning about observation sequences.

Architecture:
    Observation → Embedding → LSTM → FC Layers → Q-values

The hidden state (h, c) is passed between steps, allowing the network
to learn patterns in sequential data.

Classes:
    DRQN: LSTM-based Q-network for Tianshou DQNPolicy.

Example:
    >>> network = DRQN(
    ...     state_shape=(32,),
    ...     action_shape=10,
    ...     embedding_dim=64,
    ...     lstm_hidden_size=128,
    ... )
    >>> obs = torch.randn(1, 32)  # Single observation
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


class DRQN(BaseQNetwork):
    """Deep Recurrent Q-Network with LSTM for sequential decision making.

    This network uses an LSTM layer to maintain temporal context across
    observations. The hidden state enables the agent to reason about
    patterns in the sequence of observations.

    Architecture:
        1. Embedding: Linear projection to embedding space
        2. LSTM: Recurrent layer with hidden state (h, c)
        3. FC Layers: Fully-connected layers with activation
        4. Output: Q-values for each action

    The hidden state must be passed back to the buffer and policy
    for proper temporal credit assignment.

    Attributes:
        embedding (nn.Linear): Embedding layer.
        lstm (nn.LSTM): LSTM layer.
        fc (nn.Sequential): Fully-connected output layers.
        lstm_hidden_size (int): LSTM hidden dimension.
        num_lstm_layers (int): Number of stacked LSTM layers.

    Example:
        >>> drqn = DRQN(
        ...     state_shape=(32,),
        ...     action_shape=10,
        ...     embedding_dim=64,
        ...     lstm_hidden_size=128,
        ...     fc_hidden_sizes=(128, 64),
        ... )
        >>> obs = torch.randn(4, 32)  # Batch of 4
        >>> q_values, state = drqn(obs)
        >>> q_values.shape
        torch.Size([4, 10])
        >>> state['h'].shape
        torch.Size([1, 4, 128])  # (num_layers, batch, hidden)
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_shape: int,
        embedding_dim: int = 64,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 1,
        fc_hidden_sizes: Tuple[int, ...] = (128, 64),
        activation_fn: type = nn.ReLU,
        dropout: float = 0.0,
        device: str = "cpu",
        enable_diagnostics: bool = True,
        diagnostic_log_interval: int = 1000,
        use_noisy_layers: bool = False,
        noisy_sigma: float = 0.5,
    ):
        """Initialize DRQN network.

        Args:
            state_shape: Observation space dimension(s).
            action_shape: Number of discrete actions.
            embedding_dim: Embedding layer output dimension.
            lstm_hidden_size: LSTM hidden state size.
            num_lstm_layers: Number of stacked LSTM layers.
            fc_hidden_sizes: Sizes of fully-connected layers after LSTM.
            activation_fn: Activation function class (e.g., nn.ReLU).
            dropout: Dropout rate for regularization (0.0 = no dropout).
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
        # state_shape is (seq_len, feature_dim) for sequences or (feature_dim,) for single
        if isinstance(state_shape, tuple):
            if len(state_shape) == 2:
                # (seq_len, feature_dim) - use feature_dim for embedding
                state_dim = state_shape[-1]
            else:
                # (feature_dim,) - single observation
                state_dim = state_shape[0]
        else:
            state_dim = state_shape

        # Embedding layer: project observation to embedding space
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

        # Fully-connected layers after LSTM
        fc_layers = []
        input_dim = lstm_hidden_size

        for hidden_size in fc_hidden_sizes:
            fc_layers.append(make_linear(input_dim, hidden_size))
            fc_layers.append(activation_fn())
            if dropout > 0.0:
                fc_layers.append(nn.Dropout(dropout))
            input_dim = hidden_size

        # Output layer: Q-values for each action
        fc_layers.append(make_linear(input_dim, action_shape))

        self.fc = nn.Sequential(*fc_layers)

    def reset_noise(self) -> None:
        """Reset noise in all NoisyLinear layers.

        Should be called at the start of each forward pass during training
        to get fresh exploration noise.
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
        """Forward pass through DRQN.

        Processes observations through embedding, LSTM, and FC layers.
        Returns Q-values and updated hidden state for next step.

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

        Note:
            Phase 2 enhancement: Now supports list of states for proper
            batch training with individual hidden states per sample.
        """
        # Convert numpy to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Ensure tensor is on correct device
        if obs.device != torch.device(self.device):
            obs = obs.to(self.device)

        # Handle batched input: add sequence dimension if needed
        if obs.dim() == 2:  # (batch, feature_dim)
            obs = obs.unsqueeze(1)  # (batch, 1, feature_dim)

        batch_size = obs.shape[0]

        # Embedding
        embedded = self.embedding(obs)  # (batch, seq_len, embedding_dim)

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

        # Take last timestep output for Q-value computation
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)

        # Fully-connected layers to get Q-values
        q_values = self.fc(last_out)  # (batch, n_actions)

        # Return Q-values and new hidden state
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
        """DRQN is recurrent (maintains hidden state)."""
        return True

    def forward_with_burnin(
        self,
        obs: torch.Tensor,
        burn_in_length: int,
        initial_hidden: Optional[Dict[str, torch.Tensor]] = None,
        return_all_q: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with burn-in for sequence-based training.

        This method implements the R2D2-style burn-in mechanism:
        1. Run LSTM through burn-in portion WITHOUT gradients
        2. Use resulting hidden state for training portion WITH gradients
        3. Compute Q-values only for training portion

        Args:
            obs: Observation sequence (batch, seq_len, obs_dim)
            burn_in_length: Number of initial steps for burn-in (no gradients)
            initial_hidden: Initial hidden state from replay buffer (optional)
            return_all_q: If True, return Q-values for all training steps
                         If False, return Q-value for last step only

        Returns:
            Tuple of:
                - q_values: Q-values for training portion
                  Shape: (batch, n_actions) if return_all_q=False
                  Shape: (batch, train_len, n_actions) if return_all_q=True
                - final_hidden: Hidden state after full sequence

        Raises:
            ValueError: If sequence length <= burn_in_length

        Example:
            >>> net = DRQN(obs_dim=64, n_actions=10)
            >>> obs = torch.randn(32, 16, 64)  # batch=32, seq=16
            >>> q_values, hidden = net.forward_with_burnin(
            ...     obs, burn_in_length=8, return_all_q=True
            ... )
            >>> print(q_values.shape)
            torch.Size([32, 8, 10])
        """
        seq_len = obs.shape[1]

        if seq_len <= burn_in_length:
            raise ValueError(
                f"Sequence length {seq_len} must be greater than "
                f"burn_in_length {burn_in_length}"
            )

        # Split sequence
        burn_in_obs = obs[:, :burn_in_length]
        train_obs = obs[:, burn_in_length:]
        train_len = train_obs.shape[1]

        # Burn-in: run without gradients
        with torch.no_grad():
            _, burn_in_hidden = self.forward(burn_in_obs, state=initial_hidden)

        # Detach hidden state
        warmed_hidden = {
            "h": burn_in_hidden["h"].detach(),
            "c": burn_in_hidden["c"].detach(),
        }

        # Training: with gradients
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
