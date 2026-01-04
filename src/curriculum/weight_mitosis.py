"""Weight Mitosis for Q-Network Expansion.

This module provides functionality to expand Q-network output layers
during curriculum learning phase transitions. Children actions inherit
the learned Q-values from their parent actions.

Supports:
    - DRQN: Standard Q-network with fc output
    - DuelingDRQN: Dueling architecture with advantage_stream output
    - C51DRQN: Distributional network with fc output (num_atoms > 1)
    - RainbowDRQN: Dueling + Distributional with advantage_stream output

Classes:
    WeightMitosis: Handles Q-network output layer expansion.

Example:
    >>> mitosis = WeightMitosis()
    >>> # Standard DRQN expansion
    >>> new_network = mitosis.expand_q_head(
    ...     q_network=network,
    ...     parent_to_children={0: [0, 1], 1: [2, 3]},
    ...     old_action_size=2,
    ...     new_action_size=4,
    ... )
    >>> # C51/Rainbow expansion with num_atoms
    >>> new_network = mitosis.expand_q_head(
    ...     q_network=c51_network,
    ...     parent_to_children={0: [0, 1], 1: [2, 3]},
    ...     old_action_size=2,
    ...     new_action_size=4,
    ...     num_atoms=51,
    ... )
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.networks.base import BaseQNetwork
from src.networks.noisy_linear import NoisyLinear
from src.utils.logging.logging_manager import get_logger


class WeightMitosis:
    """Expand Q-network output layers during phase transitions.

    When transitioning from a coarse action space to a finer one,
    this class expands the output layer of the Q-network. Child
    actions inherit the weights from their parent actions.

    Supports multiple network architectures:
    - DRQN: Uses `fc` attribute (num_atoms=1)
    - DuelingDRQN: Uses `advantage_stream` attribute (num_atoms=1)
    - C51DRQN: Uses `fc` attribute (num_atoms > 1, typically 51)
    - RainbowDRQN: Uses `advantage_stream` attribute (num_atoms > 1)

    The mitosis process:
    1. Identify the final Linear layer in the network's fc or advantage_stream
    2. Create a new Linear layer with the expanded output size
    3. Copy parent weights to children according to the mapping
       (atom-aware for distributional networks)
    4. Replace the original layer with the expanded one

    Attributes:
        logger: Logger instance.
        add_noise: Whether to add small noise to child weights.
        noise_scale: Scale of noise to add (if enabled).

    Example:
        >>> mitosis = WeightMitosis(add_noise=True, noise_scale=0.01)
        >>>
        >>> # Expand DRQN from 2 actions to 4 actions
        >>> mapping = {0: [0, 1], 1: [2, 3]}
        >>> network = mitosis.expand_q_head(network, mapping, 2, 4)
        >>>
        >>> # Expand C51DRQN with atom-aware weight copy
        >>> c51_network = mitosis.expand_q_head(
        ...     c51_network, mapping, 2, 4, num_atoms=51
        ... )
    """

    def __init__(
        self,
        add_noise: bool = True,
        noise_scale: float = 0.01,
    ):
        """Initialize WeightMitosis.

        Args:
            add_noise: Whether to add small noise to differentiate
                children with the same parent.
            noise_scale: Standard deviation of Gaussian noise.
        """
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.logger = get_logger("WeightMitosis")

    def expand_q_head(
        self,
        q_network: BaseQNetwork,
        parent_to_children: Dict[int, List[int]],
        old_action_size: int,
        new_action_size: int,
        num_atoms: int = 1,
    ) -> BaseQNetwork:
        """Expand Q-network output layer for new action space.

        Supports both standard Q-networks (DRQN, DuelingDRQN) and
        distributional networks (C51DRQN, RainbowDRQN).

        Args:
            q_network: Q-network with fc or advantage_stream attribute.
            parent_to_children: Mapping from parent action indices to
                child action indices.
            old_action_size: Number of actions in old (parent) phase.
            new_action_size: Number of actions in new (child) phase.
            num_atoms: Number of atoms per action for distributional networks.
                Use 1 for standard DQN/DRQN/DuelingDRQN.
                Use 51 (or other value) for C51/Rainbow.

        Returns:
            The same network with expanded output layer.

        Raises:
            ValueError: If network structure is incompatible.
        """
        self.logger.info(
            f"Expanding Q-head: {old_action_size} -> {new_action_size} actions"
            f" (num_atoms={num_atoms})"
        )

        # Find the output layer (last Linear/NoisyLinear in fc or advantage_stream)
        output_layer, layer_idx, attr_name = self._find_output_layer(q_network)

        if output_layer is None:
            raise ValueError(
                "Could not find output Linear/NoisyLinear layer in network.fc or "
                "network.advantage_stream"
            )

        # Check if this is a NoisyLinear layer
        is_noisy = isinstance(output_layer, NoisyLinear)

        # Validate layer dimensions (account for num_atoms)
        expected_features = old_action_size * num_atoms
        if output_layer.out_features != expected_features:
            raise ValueError(
                f"Output layer has {output_layer.out_features} features, "
                f"expected {expected_features} (actions={old_action_size}, "
                f"atoms={num_atoms})"
            )

        # Create new expanded layer on the same device as the network
        device = next(q_network.parameters()).device
        input_features = output_layer.in_features
        new_out_features = new_action_size * num_atoms

        if is_noisy:
            # Create NoisyLinear layer with same sigma_init
            sigma_init = output_layer.sigma_init
            new_layer = NoisyLinear(
                input_features, new_out_features, sigma_init=sigma_init
            )
            new_layer = new_layer.to(device)
            self.logger.info(f"Creating NoisyLinear layer (sigma_init={sigma_init})")
        else:
            new_layer = nn.Linear(input_features, new_out_features, device=device)

        # Copy weights from parents to children (atom-aware)
        self._copy_weights_with_mitosis(
            old_layer=output_layer,
            new_layer=new_layer,
            parent_to_children=parent_to_children,
            num_atoms=num_atoms,
        )

        # Replace the layer in the network
        self._replace_output_layer(q_network, new_layer, layer_idx, attr_name)

        # Reset noise in NoisyLinear layer after weight copy
        if is_noisy:
            new_layer.reset_noise()

        # Update network's action_shape attribute if present
        if hasattr(q_network, "action_shape"):
            q_network.action_shape = new_action_size
        if hasattr(q_network, "_action_shape"):
            q_network._action_shape = new_action_size

        self.logger.info(
            f"Q-head expansion complete. New output size: {new_out_features}"
        )

        return q_network

    def _find_output_layer(self, q_network: BaseQNetwork) -> tuple:
        """Find the output Linear layer in the network.

        Supports:
        - fc attribute (DRQN, C51DRQN)
        - advantage_stream attribute (DuelingDRQN, RainbowDRQN)

        Priority: advantage_stream (Dueling/Rainbow), then fc (DRQN/C51)

        Args:
            q_network: Q-network with fc or advantage_stream attribute.

        Returns:
            Tuple of (output_layer, layer_index, attribute_name) or
            (None, -1, None) if not found.
        """
        # Check both attributes, prioritize advantage_stream
        for attr_name in ["advantage_stream", "fc"]:
            if hasattr(q_network, attr_name):
                attr = getattr(q_network, attr_name)
                layer, idx = self._find_last_linear_in_module(attr)
                if layer is not None:
                    return layer, idx, attr_name

        self.logger.error("Network has no 'fc' or 'advantage_stream' attribute")
        return None, -1, None

    def _find_last_linear_in_module(self, module) -> tuple:
        """Find the last Linear or NoisyLinear layer in a module.

        Args:
            module: nn.Module, nn.Sequential, nn.Linear, or NoisyLinear.

        Returns:
            Tuple of (layer, index) or (None, -1) if not found.
        """
        if isinstance(module, nn.Sequential):
            for idx in range(len(module) - 1, -1, -1):
                if isinstance(module[idx], (nn.Linear, NoisyLinear)):
                    return module[idx], idx
        elif isinstance(module, (nn.Linear, NoisyLinear)):
            return module, 0
        return None, -1

    def _replace_output_layer(
        self,
        q_network: BaseQNetwork,
        new_layer: nn.Linear,
        layer_idx: int,
        attr_name: str = "fc",
    ) -> None:
        """Replace the output layer in the network.

        Args:
            q_network: Q-network to modify.
            new_layer: New Linear layer.
            layer_idx: Index of layer to replace.
            attr_name: Attribute name ('fc' or 'advantage_stream').
        """
        attr = getattr(q_network, attr_name)

        if isinstance(attr, nn.Sequential):
            # Replace layer at index
            attr[layer_idx] = new_layer
        else:
            # attr is a single Linear layer
            setattr(q_network, attr_name, new_layer)

    def _copy_weights_with_mitosis(
        self,
        old_layer: Union[nn.Linear, NoisyLinear],
        new_layer: Union[nn.Linear, NoisyLinear],
        parent_to_children: Dict[int, List[int]],
        num_atoms: int = 1,
    ) -> None:
        """Copy weights from parent actions to child actions.

        For distributional networks (num_atoms > 1), copies entire
        atom blocks from parent to children.

        Supports both nn.Linear and NoisyLinear layers.

        Args:
            old_layer: Original output layer.
            new_layer: New expanded output layer.
            parent_to_children: Mapping from parent to child indices.
            num_atoms: Number of atoms per action (1 for non-distributional).
        """
        # Determine weight/bias attribute names based on layer type
        is_noisy = isinstance(old_layer, NoisyLinear)
        weight_attr = "weight_mu" if is_noisy else "weight"
        bias_attr = "bias_mu" if is_noisy else "bias"

        old_weight = getattr(old_layer, weight_attr)
        old_bias = getattr(old_layer, bias_attr)
        new_weight = getattr(new_layer, weight_attr)
        new_bias = getattr(new_layer, bias_attr)

        device = new_weight.device
        with torch.no_grad():
            # Initialize new layer with zeros (will be filled)
            new_weight.zero_()
            new_bias.zero_()

            # For NoisyLinear, also handle sigma parameters
            if is_noisy:
                new_layer.weight_sigma.zero_()
                new_layer.bias_sigma.zero_()
                old_weight_sigma = old_layer.weight_sigma
                old_bias_sigma = old_layer.bias_sigma

            # Copy parent weights to children (atom-aware)
            for parent_idx, child_indices in parent_to_children.items():
                # Calculate atom-aware index ranges for parent
                old_start = parent_idx * num_atoms
                old_end = old_start + num_atoms

                parent_weight = old_weight[old_start:old_end]
                parent_bias = old_bias[old_start:old_end]

                if is_noisy:
                    parent_weight_sigma = old_weight_sigma[old_start:old_end]
                    parent_bias_sigma = old_bias_sigma[old_start:old_end]

                for child_idx in child_indices:
                    # Calculate atom-aware index ranges for child
                    new_start = child_idx * num_atoms
                    new_end = new_start + num_atoms

                    # Copy entire atom block (mean weights)
                    new_weight[new_start:new_end] = parent_weight.clone()
                    new_bias[new_start:new_end] = parent_bias.clone()

                    # For NoisyLinear, also copy sigma parameters
                    if is_noisy:
                        new_layer.weight_sigma.data[new_start:new_end] = (
                            parent_weight_sigma.clone()
                        )
                        new_layer.bias_sigma.data[new_start:new_end] = (
                            parent_bias_sigma.clone()
                        )

                    # Add small noise to differentiate siblings
                    if self.add_noise and len(child_indices) > 1:
                        noise_w = (
                            torch.randn_like(parent_weight, device=device)
                            * self.noise_scale
                        )
                        noise_b = (
                            torch.randn_like(parent_bias, device=device)
                            * self.noise_scale
                        )
                        new_weight[new_start:new_end] += noise_w
                        new_bias[new_start:new_end] += noise_b

            # Handle any orphan actions (not in mapping)
            # Initialize them with small random values
            all_children = set()
            for children in parent_to_children.values():
                all_children.update(children)

            new_action_size = new_layer.out_features // num_atoms
            orphan_actions = set(range(new_action_size)) - all_children

            if orphan_actions:
                self.logger.warning(
                    f"Found {len(orphan_actions)} orphan actions: {orphan_actions}"
                )
                for orphan_idx in orphan_actions:
                    start = orphan_idx * num_atoms
                    end = start + num_atoms
                    # Initialize with Xavier uniform
                    nn.init.xavier_uniform_(new_weight[start:end])
                    if is_noisy:
                        # Also initialize sigma for orphans
                        sigma_value = old_layer.sigma_init / (
                            new_weight.shape[1] ** 0.5
                        )
                        new_layer.weight_sigma.data[start:end].fill_(sigma_value)
                        new_layer.bias_sigma.data[start:end].fill_(sigma_value)

    def get_inherited_value_estimate(
        self,
        q_values: torch.Tensor,
        parent_to_children: Dict[int, List[int]],
    ) -> torch.Tensor:
        """Get expected Q-values for children based on parent values.

        Useful for analysis or target network initialization.

        Args:
            q_values: Q-values for parent actions [batch, old_action_size].
            parent_to_children: Parent to children mapping.

        Returns:
            Expanded Q-values [batch, new_action_size].
        """
        batch_size = q_values.shape[0]
        new_size = sum(len(c) for c in parent_to_children.values())

        expanded = torch.zeros(batch_size, new_size, device=q_values.device)

        for parent_idx, child_indices in parent_to_children.items():
            parent_value = q_values[:, parent_idx]
            for child_idx in child_indices:
                expanded[:, child_idx] = parent_value

        return expanded

    def _get_network_num_atoms(self, q_network: BaseQNetwork) -> int:
        """Get num_atoms from network if it's distributional.

        Args:
            q_network: Q-network to check.

        Returns:
            num_atoms value or 1 if non-distributional.
        """
        return getattr(q_network, "num_atoms", 1)


def create_expanded_network(
    original_network: BaseQNetwork,
    parent_to_children: Dict[int, List[int]],
    old_action_size: int,
    new_action_size: int,
    add_noise: bool = True,
    noise_scale: float = 0.01,
    num_atoms: int = 1,
) -> BaseQNetwork:
    """Convenience function to expand a Q-network.

    Args:
        original_network: Network to expand.
        parent_to_children: Action mapping for mitosis.
        old_action_size: Current action space size.
        new_action_size: Target action space size.
        add_noise: Whether to add differentiation noise.
        noise_scale: Scale of noise.
        num_atoms: Number of atoms for distributional networks.

    Returns:
        The same network with expanded output layer.
    """
    mitosis = WeightMitosis(add_noise=add_noise, noise_scale=noise_scale)
    return mitosis.expand_q_head(
        q_network=original_network,
        parent_to_children=parent_to_children,
        old_action_size=old_action_size,
        new_action_size=new_action_size,
        num_atoms=num_atoms,
    )
