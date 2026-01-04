"""LSTM burn-in utilities for recurrent training.

This module provides utilities for implementing the burn-in mechanism
as described in the R2D2 paper. Burn-in warms up LSTM hidden states
at the start of each sampled sequence without gradients.

Reference:
    Kapturowski et al. (2019) - "Recurrent Experience Replay in Distributed RL"
"""

from typing import Dict, Optional, Tuple, Protocol, runtime_checkable
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class RecurrentNetwork(Protocol):
    """Protocol for recurrent networks that support burn-in."""

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning output and hidden state."""
        ...


def split_sequence_for_burnin(
    obs: torch.Tensor,
    burn_in_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split observation sequence into burn-in and training portions.

    The burn-in portion is used to warm up LSTM hidden states without
    gradients. The training portion is used for actual loss computation.

    Args:
        obs: Observation tensor of shape (batch, seq_len, ...)
        burn_in_length: Number of steps for burn-in (no gradients)

    Returns:
        Tuple of (burn_in_obs, train_obs)
            burn_in_obs: Shape (batch, burn_in_length, ...)
            train_obs: Shape (batch, seq_len - burn_in_length, ...)

    Raises:
        ValueError: If sequence length <= burn_in_length

    Example:
        >>> obs = torch.randn(32, 16, 64)  # batch=32, seq=16, feat=64
        >>> burn_in, train = split_sequence_for_burnin(obs, burn_in_length=8)
        >>> print(burn_in.shape, train.shape)
        torch.Size([32, 8, 64]) torch.Size([32, 8, 64])
    """
    seq_len = obs.shape[1]

    if seq_len <= burn_in_length:
        raise ValueError(
            f"Sequence length {seq_len} must be greater than "
            f"burn-in length {burn_in_length}. Cannot perform burn-in."
        )

    burn_in_obs = obs[:, :burn_in_length]
    train_obs = obs[:, burn_in_length:]

    return burn_in_obs, train_obs


def compute_burnin_hidden_state(
    network: nn.Module,
    burn_in_obs: torch.Tensor,
    initial_hidden: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute LSTM hidden state after burn-in without gradients.

    Runs the network through the burn-in portion of a sequence to
    produce a warmed-up hidden state for training.

    Args:
        network: The recurrent network module (must have forward method)
        burn_in_obs: Observations for burn-in (batch, burn_in_len, ...)
        initial_hidden: Initial hidden state from replay buffer (optional)

    Returns:
        Hidden state dict after burn-in with keys "h" and "c"
        Both tensors are detached to prevent gradient flow.

    Example:
        >>> net = DRQN(obs_dim=64, n_actions=10, lstm_hidden_size=128)
        >>> burn_in_obs = torch.randn(32, 8, 64)
        >>> hidden = compute_burnin_hidden_state(net, burn_in_obs)
        >>> print(hidden["h"].shape)
        torch.Size([1, 32, 128])
    """
    with torch.no_grad():
        # Run network through burn-in sequence
        _, hidden = network.forward(burn_in_obs, state=initial_hidden)

    # Detach to ensure no gradient flow
    return {
        "h": hidden["h"].detach(),
        "c": hidden["c"].detach(),
    }


def verify_no_gradient_in_burnin(
    network: nn.Module,
    obs: torch.Tensor,
    burn_in_length: int,
    initial_hidden: Optional[Dict[str, torch.Tensor]] = None,
) -> bool:
    """Verify that burn-in portion has no gradient accumulation.

    This is a debugging utility to ensure the burn-in mechanism
    is correctly blocking gradients.

    Args:
        network: The recurrent network module
        obs: Full observation sequence (batch, seq_len, ...)
        burn_in_length: Number of burn-in steps
        initial_hidden: Initial hidden state

    Returns:
        True if no gradients are accumulated during burn-in

    Note:
        This function is primarily for testing and debugging.
    """
    burn_in_obs, train_obs = split_sequence_for_burnin(obs, burn_in_length)

    # Get burn-in hidden state
    warmed_hidden = compute_burnin_hidden_state(network, burn_in_obs, initial_hidden)

    # Check that hidden states have no grad_fn
    h_no_grad = warmed_hidden["h"].grad_fn is None
    c_no_grad = warmed_hidden["c"].grad_fn is None

    if not (h_no_grad and c_no_grad):
        logger.warning(
            "Burn-in hidden state has gradient function attached! "
            "This may cause gradient leakage."
        )
        return False

    return True


class BurnInConfig:
    """Configuration for burn-in mechanism.

    Attributes:
        enabled: Whether burn-in is enabled
        burn_in_length: Number of steps for burn-in
        use_stored_hidden: Whether to use hidden state from buffer

    Example:
        >>> config = BurnInConfig(burn_in_length=8)
        >>> print(config)
        BurnInConfig(enabled=True, burn_in_length=8, use_stored_hidden=True)
    """

    def __init__(
        self,
        enabled: bool = True,
        burn_in_length: int = 8,
        use_stored_hidden: bool = True,
    ):
        """Initialize burn-in configuration.

        Args:
            enabled: Whether burn-in is enabled
            burn_in_length: Number of burn-in steps (should be < seq_len)
            use_stored_hidden: Whether to use stored hidden from buffer
                              as starting point for burn-in
        """
        self.enabled = enabled
        self.burn_in_length = burn_in_length
        self.use_stored_hidden = use_stored_hidden

        if burn_in_length < 0:
            raise ValueError("burn_in_length must be non-negative")

    def __repr__(self) -> str:
        return (
            f"BurnInConfig(enabled={self.enabled}, "
            f"burn_in_length={self.burn_in_length}, "
            f"use_stored_hidden={self.use_stored_hidden})"
        )

    @classmethod
    def from_dict(cls, config: Dict) -> "BurnInConfig":
        """Create BurnInConfig from dictionary.

        Args:
            config: Dictionary with burn-in configuration

        Returns:
            BurnInConfig instance
        """
        return cls(
            enabled=config.get("enabled", True),
            burn_in_length=config.get("burn_in_length", 8),
            use_stored_hidden=config.get("use_stored_hidden", True),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "burn_in_length": self.burn_in_length,
            "use_stored_hidden": self.use_stored_hidden,
        }


def forward_with_burnin(
    network: nn.Module,
    obs: torch.Tensor,
    burn_in_length: int,
    initial_hidden: Optional[Dict[str, torch.Tensor]] = None,
    return_all_q: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass with burn-in for sequence-based recurrent training.

    This function implements the R2D2-style burn-in mechanism:
    1. Run LSTM through burn-in portion WITHOUT gradients
    2. Use resulting hidden state for training portion WITH gradients
    3. Return Q-values only for training portion

    Args:
        network: The recurrent network module (must have forward method)
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
            - final_hidden: Hidden state dict after full sequence

    Raises:
        ValueError: If sequence length <= burn_in_length

    Example:
        >>> net = DRQN(obs_dim=64, n_actions=10)
        >>> obs = torch.randn(32, 16, 64)  # batch=32, seq=16
        >>> q_values, hidden = forward_with_burnin(
        ...     net, obs, burn_in_length=8, return_all_q=True
        ... )
        >>> print(q_values.shape)
        torch.Size([32, 8, 10])  # Q for training portion
    """
    batch_size = obs.shape[0]
    seq_len = obs.shape[1]

    if seq_len <= burn_in_length:
        raise ValueError(
            f"Sequence length {seq_len} must be greater than "
            f"burn_in_length {burn_in_length}. Got seq={seq_len}, burn_in={burn_in_length}"
        )

    # Split sequence into burn-in and training portions
    burn_in_obs = obs[:, :burn_in_length]  # (batch, burn_in_len, obs_dim)
    train_obs = obs[:, burn_in_length:]  # (batch, train_len, obs_dim)
    train_len = train_obs.shape[1]

    # ========== BURN-IN PHASE (no gradients) ==========
    with torch.no_grad():
        _, burn_in_hidden = network.forward(burn_in_obs, state=initial_hidden)

    # Detach to ensure no gradient flow from burn-in
    warmed_hidden = {
        "h": burn_in_hidden["h"].detach(),
        "c": burn_in_hidden["c"].detach(),
    }

    # ========== TRAINING PHASE (with gradients) ==========
    if return_all_q:
        # Compute Q-values for each step in training portion
        # This is needed for proper n-step returns with LSTM
        q_values_list = []
        hidden = warmed_hidden

        for t in range(train_len):
            step_obs = train_obs[:, t : t + 1]  # (batch, 1, obs_dim)
            q, hidden = network.forward(step_obs, state=hidden)
            q_values_list.append(q)

        # Stack: (batch, train_len, n_actions)
        q_values = torch.stack(q_values_list, dim=1)
        final_hidden = hidden
    else:
        # Standard case: Q-value for last step only
        # More efficient as we can process full sequence at once
        q_values, final_hidden = network.forward(train_obs, state=warmed_hidden)

    return q_values, final_hidden


def forward_sequence(
    network: nn.Module,
    obs: torch.Tensor,
    initial_hidden: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward pass through full sequence, returning Q-values for each step.

    Unlike forward_with_burnin, this processes the full sequence and
    returns Q-values for every timestep.

    Args:
        network: The recurrent network module
        obs: Observation sequence (batch, seq_len, ...)
        initial_hidden: Initial hidden state

    Returns:
        Tuple of (q_values_sequence, final_hidden)
        q_values_sequence has shape (batch, seq_len, n_actions)

    Example:
        >>> net = DRQN(obs_dim=64, n_actions=10)
        >>> obs = torch.randn(32, 16, 64)
        >>> q_values, hidden = forward_sequence(net, obs)
        >>> print(q_values.shape)
        torch.Size([32, 16, 10])
    """
    seq_len = obs.shape[1]
    q_values_list = []
    hidden = initial_hidden

    for t in range(seq_len):
        step_obs = obs[:, t : t + 1]  # (batch, 1, ...)
        q, hidden = network.forward(step_obs, state=hidden)
        q_values_list.append(q)

    q_values = torch.stack(q_values_list, dim=1)  # (batch, seq_len, n_actions)
    return q_values, hidden
