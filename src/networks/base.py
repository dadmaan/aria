"""Base Q-Network abstract class for Tianshou-compatible networks.

This module defines the interface that all Q-networks must implement
to be compatible with Tianshou's DQNPolicy. Networks must return
Q-values and optionally a new hidden state for recurrent architectures.

Classes:
    BaseQNetwork: Abstract base class for Q-network implementations.

Example:
    class MyQNetwork(BaseQNetwork):
        def forward(self, obs, state=None, info={}):
            # Custom implementation
            return q_values, new_state
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class BaseQNetwork(nn.Module, ABC):
    """Abstract base class for Q-network implementations.

    All Q-networks must implement the forward method with a signature
    compatible with Tianshou's policy interface:

        forward(obs, state=None, info={}) -> (q_values, new_state)

    Where:
        - obs: Observation tensor of shape (batch, *obs_shape)
        - state: Optional hidden state for recurrent networks
        - info: Additional information dict (usually unused)
        - q_values: Q-values tensor of shape (batch, n_actions)
        - new_state: Updated hidden state (None for non-recurrent)

    Attributes:
        device (str): Device for tensor operations ('cpu' or 'cuda').
        state_shape (tuple): Shape of the observation space.
        action_shape (int): Number of discrete actions.
        embedding_dim (int): Dimension of the embedding layer.

    Note:
        Subclasses should call super().__init__() and set self.device
        before creating any nn.Module layers.
    """

    def __init__(
        self,
        state_shape: Union[int, Tuple[int, ...]],
        action_shape: int,
        embedding_dim: int = 64,
        device: str = "cpu",
    ):
        """Initialize base Q-network.

        Args:
            state_shape: Shape of observation space. Can be int or tuple.
            action_shape: Number of discrete actions.
            embedding_dim: Dimension of embedding layer output.
            device: Device for tensor operations ('cpu' or 'cuda').
        """
        super().__init__()
        self.device = device
        self.state_shape = (
            state_shape if isinstance(state_shape, tuple) else (state_shape,)
        )
        self.action_shape = action_shape
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through the Q-network.

        Args:
            obs: Observations. Shape depends on network type:
                - DRQN: (batch, seq_len, feature_dim) or (batch, feature_dim)
                - MLP: (batch, flattened_dim)
            state: Hidden state dict for recurrent networks. Contains:
                - 'h': Hidden state tensor (LSTM)
                - 'c': Cell state tensor (LSTM)
                For non-recurrent networks, this is None.
            info: Additional information (usually unused).

        Returns:
            Tuple of:
                - q_values: Q-values tensor of shape (batch, n_actions)
                - new_state: Updated hidden state dict (None for non-recurrent)

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def get_initial_state(
        self, batch_size: int = 1
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get initial hidden state for recurrent networks.

        Args:
            batch_size: Number of parallel environments.

        Returns:
            Initial hidden state dict, or None for non-recurrent networks.

        Note:
            Override in recurrent subclasses (e.g., DRQN).
        """
        return None

    @property
    def is_recurrent(self) -> bool:
        """Whether this network is recurrent (has hidden state).

        Returns:
            True if network maintains hidden state across steps.
        """
        return False
