"""Multi-Layer Perceptron (MLP) Q-Network implementation.

This module provides a feedforward Q-network for baseline comparison.
The MLP has no temporal memory and processes each observation independently.

Architecture:
    Observation (flattened) → Embedding → FC Layers → Q-values

This serves as a baseline to compare against DRQN to measure the
impact of temporal reasoning on learning performance.

Classes:
    MLPQNetwork: Feedforward Q-network for Tianshou DQNPolicy.

Example:
    >>> network = MLPQNetwork(
    ...     state_shape=(32,),  # Flattened observation
    ...     action_shape=10,
    ...     embedding_dim=64,
    ... )
    >>> obs = torch.randn(1, 32)
    >>> q_values, state = network(obs)
    >>> q_values.shape
    torch.Size([1, 10])
    >>> state is None
    True
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .base import BaseQNetwork


class MLPQNetwork(BaseQNetwork):
    """Multi-Layer Perceptron Q-Network for baseline comparison.

    This network is a simple feedforward architecture with no temporal
    memory. It serves as a baseline to measure the impact of LSTM
    temporal reasoning in the DRQN.

    Architecture:
        1. Embedding: Linear projection from flattened observation
        2. FC Layers: Fully-connected layers with activation
        3. Output: Q-values for each action

    Important: For fair comparison with DRQN, this network uses the
    same embedding dimension. The key difference is:
        - DRQN: obs → embedding → LSTM(temporal) → FC → Q
        - MLP:  obs → embedding → FC → Q (no temporal)

    Attributes:
        embedding (nn.Linear): Embedding layer.
        embedding_activation: Activation after embedding.
        fc (nn.Sequential): Fully-connected layers.
        input_dim (int): Flattened input dimension.

    Example:
        >>> mlp = MLPQNetwork(
        ...     state_shape=(32,),
        ...     action_shape=10,
        ...     embedding_dim=64,
        ...     fc_hidden_sizes=(128, 64),
        ... )
        >>> obs = torch.randn(4, 32)  # Batch of 4
        >>> q_values, state = mlp(obs)
        >>> q_values.shape
        torch.Size([4, 10])
        >>> state is None
        True
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_shape: int,
        embedding_dim: int = 64,
        fc_hidden_sizes: Tuple[int, ...] = (128, 64),
        activation_fn: type = nn.ReLU,
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        """Initialize MLP Q-network.

        Args:
            state_shape: Observation space dimension(s). Will be flattened.
            action_shape: Number of discrete actions.
            embedding_dim: Embedding layer output dimension.
                Note: Same as DRQN for fair comparison.
            fc_hidden_sizes: Sizes of fully-connected layers.
            activation_fn: Activation function class.
            dropout: Dropout rate for regularization.
            device: Device for tensor operations.
        """
        super().__init__(state_shape, action_shape, embedding_dim, device)

        # Calculate flattened input dimension
        if isinstance(state_shape, tuple):
            self.input_dim = int(np.prod(state_shape))
        else:
            self.input_dim = state_shape

        # Embedding layer: project flattened observation to embedding space
        # Note: Same embedding_dim as DRQN for fair comparison
        self.embedding = nn.Linear(self.input_dim, embedding_dim)
        self.embedding_activation = activation_fn()

        # Fully-connected layers
        fc_layers = []
        input_dim = embedding_dim

        for hidden_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(input_dim, hidden_size))
            fc_layers.append(activation_fn())
            if dropout > 0.0:
                fc_layers.append(nn.Dropout(dropout))
            input_dim = hidden_size

        # Output layer: Q-values for each action
        fc_layers.append(nn.Linear(input_dim, action_shape))

        self.fc = nn.Sequential(*fc_layers)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass through MLP.

        Processes observations through embedding and FC layers.
        Returns Q-values and None (no hidden state for MLP).

        Args:
            obs: Observations of shape (batch, *state_shape).
                Will be flattened to (batch, input_dim).
            state: Ignored for MLP (no hidden state).
            info: Additional information (unused).

        Returns:
            Tuple of:
                - q_values: Q-values of shape (batch, n_actions)
                - None: MLP has no hidden state

        Note:
            The state parameter is accepted for API compatibility
            with Tianshou but is ignored.
        """
        # Convert numpy to tensor if needed
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        # Ensure tensor is on correct device
        if obs.device != torch.device(self.device):
            obs = obs.to(self.device)

        # Flatten observation if needed
        if obs.dim() > 2:
            obs = obs.view(obs.shape[0], -1)  # (batch, flattened_dim)

        # Embedding with activation
        embedded = self.embedding_activation(self.embedding(obs))

        # Fully-connected layers to get Q-values
        q_values = self.fc(embedded)  # (batch, n_actions)

        # MLP has no hidden state
        return q_values, None

    @property
    def is_recurrent(self) -> bool:
        """MLP is not recurrent (no hidden state)."""
        return False
