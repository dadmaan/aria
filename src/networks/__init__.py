"""Network module for Tianshou Q-networks.

This module provides modular Q-network implementations for the
Tianshou RL framework. Networks are registered with a factory
for easy instantiation based on configuration.

Available Networks:
    - drqn: LSTM-based Deep Recurrent Q-Network
    - mlp: Feedforward Multi-Layer Perceptron Q-Network

Example:
    >>> from src.networks import NetworkFactory
    >>>
    >>> # Create DRQN network
    >>> drqn = NetworkFactory.create(
    ...     network_type="drqn",
    ...     state_shape=(32,),
    ...     action_shape=10,
    ...     config={"embedding_dim": 64, "lstm_hidden_size": 128},
    ... )
    >>>
    >>> # Create MLP network
    >>> mlp = NetworkFactory.create(
    ...     network_type="mlp",
    ...     state_shape=(32,),
    ...     action_shape=10,
    ...     config={"embedding_dim": 64},
    ... )
"""

from typing import Any, Dict, Optional, Tuple, Type, Union

import torch.nn as nn

from .base import BaseQNetwork
from .drqn import DRQN
from .mlp import MLPQNetwork
from .dueling_drqn import DuelingDRQN
from .c51_drqn import C51DRQN
from .rainbow_drqn import RainbowDRQN
from .noisy_linear import NoisyLinear, reset_noise_recursive


# Registry of available networks
_NETWORK_REGISTRY: Dict[str, Type[BaseQNetwork]] = {
    "drqn": DRQN,
    "mlp": MLPQNetwork,
    "dueling_drqn": DuelingDRQN,
    "c51_drqn": C51DRQN,
    "rainbow_drqn": RainbowDRQN,
}


class NetworkFactory:
    """Factory for creating Q-network instances.

    This factory provides a unified interface for instantiating
    different Q-network architectures based on configuration.
    New network types can be registered for easy extensibility.

    Class Attributes:
        _registry: Dictionary mapping network type names to classes.

    Example:
        >>> network = NetworkFactory.create(
        ...     network_type="drqn",
        ...     state_shape=(32,),
        ...     action_shape=10,
        ...     config={"embedding_dim": 64},
        ... )
        >>> type(network).__name__
        'DRQN'
    """

    _registry: Dict[str, Type[BaseQNetwork]] = _NETWORK_REGISTRY

    @classmethod
    def create(
        cls,
        network_type: str,
        state_shape: Union[int, Tuple[int, ...]],
        action_shape: int,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> BaseQNetwork:
        """Create a Q-network instance.

        Args:
            network_type: Type of network ('drqn' or 'mlp').
            state_shape: Shape of the observation space.
            action_shape: Number of discrete actions.
            config: Network configuration dict. Supported keys depend
                on network type:

                Shared keys (both networks):
                    - embedding_dim (int): Embedding dimension, default 64
                    - fc_hidden_sizes (tuple): FC layer sizes, default (128, 64)
                    - activation_fn (str): Activation name, default 'relu'
                    - dropout (float): Dropout rate, default 0.0

                DRQN-specific keys:
                    - lstm_hidden_size (int): LSTM hidden size, default 256
                    - num_lstm_layers (int): LSTM layers, default 1
                    - lstm.hidden_size (int): Alternative nested format
                    - lstm.num_layers (int): Alternative nested format

            device: Device for tensor operations ('cpu' or 'cuda').

        Returns:
            Instantiated Q-network.

        Raises:
            ValueError: If network_type is not registered.

        Example:
            >>> config = {
            ...     "embedding_dim": 64,
            ...     "lstm_hidden_size": 128,
            ...     "fc_hidden_sizes": [128, 64],
            ... }
            >>> network = NetworkFactory.create("drqn", (32,), 10, config)
        """
        if network_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown network type: '{network_type}'. "
                f"Available types: {available}"
            )

        network_cls = cls._registry[network_type]
        config = config or {}

        # Parse activation function
        activation_name = config.get("activation_fn", "relu")
        if isinstance(activation_name, str):
            activation_name = activation_name.lower()
        else:
            # Already a class
            activation_name = "relu"

        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leakyrelu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
        }
        activation_fn = activation_map.get(activation_name, nn.ReLU)

        # Common kwargs for all networks
        kwargs = {
            "state_shape": state_shape,
            "action_shape": action_shape,
            "embedding_dim": config.get("embedding_dim", 64),
            "fc_hidden_sizes": tuple(config.get("fc_hidden_sizes", [128, 64])),
            "activation_fn": activation_fn,
            "dropout": config.get("dropout", 0.0),
            "device": device,
        }

        # Add DRQN-specific kwargs (for all recurrent networks)
        if network_type in ("drqn", "dueling_drqn", "c51_drqn", "rainbow_drqn"):
            lstm_config = config.get("lstm", {})
            kwargs["lstm_hidden_size"] = lstm_config.get(
                "hidden_size", config.get("lstm_hidden_size", 256)
            )
            kwargs["num_lstm_layers"] = lstm_config.get(
                "num_layers", config.get("num_lstm_layers", 1)
            )

        # Add distributional kwargs for C51 and Rainbow
        if network_type in ("c51_drqn", "rainbow_drqn"):
            kwargs["num_atoms"] = config.get("num_atoms", 51)

        # Add NoisyNet kwargs for all recurrent networks (DRQN, Dueling, C51, Rainbow)
        if network_type in ("drqn", "dueling_drqn", "c51_drqn", "rainbow_drqn"):
            noisy_config = config.get("noisy_net", {})
            kwargs["use_noisy_layers"] = noisy_config.get(
                "enabled", config.get("use_noisy_layers", False)
            )
            kwargs["noisy_sigma"] = noisy_config.get(
                "sigma_init", config.get("noisy_sigma", 0.5)
            )

        return network_cls(**kwargs)

    @classmethod
    def register(cls, name: str, network_cls: Type[BaseQNetwork]) -> None:
        """Register a new network type.

        Args:
            name: Name to register the network under.
            network_cls: Network class (must inherit from BaseQNetwork).

        Raises:
            TypeError: If network_cls doesn't inherit from BaseQNetwork.

        Example:
            >>> class TransformerQNetwork(BaseQNetwork):
            ...     pass
            >>> NetworkFactory.register("transformer", TransformerQNetwork)
        """
        if not issubclass(network_cls, BaseQNetwork):
            raise TypeError(f"{network_cls.__name__} must inherit from BaseQNetwork")
        cls._registry[name] = network_cls

    @classmethod
    def available_networks(cls) -> list:
        """Get list of available network types.

        Returns:
            List of registered network type names.
        """
        return list(cls._registry.keys())


# Convenience exports
__all__ = [
    "BaseQNetwork",
    "DRQN",
    "DuelingDRQN",
    "C51DRQN",
    "RainbowDRQN",
    "MLPQNetwork",
    "NetworkFactory",
    "NoisyLinear",
    "reset_noise_recursive",
]
