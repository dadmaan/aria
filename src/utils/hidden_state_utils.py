"""LSTM hidden state handling utilities.

This module provides robust utilities for handling LSTM hidden states
across different scenarios: inference, multi-env rollouts, and batch training.

The key scenarios handled:
1. None state: Initialize to zeros (first step or fresh episode)
2. Single dict state with matching batch: Pass through unchanged
3. Single dict state with batch=1: Broadcast/expand to batch size
4. List of dict states: Stack individual states from batch sampling
5. Tianshou Batch state: Extract h, c from Batch object
6. Batch size mismatch: Fall back to zeros with warning

"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

# Try to import Tianshou Batch for type checking
try:
    from tianshou.data import Batch

    TIANSHOU_AVAILABLE = True
except ImportError:
    TIANSHOU_AVAILABLE = False
    Batch = None

logger = logging.getLogger(__name__)


class HiddenStateHandler:
    """Handles LSTM hidden state preparation for various batch scenarios.

    This class centralizes the logic for handling LSTM hidden states across
    different training and inference scenarios. It properly handles:

    1. List of states: Stack individual states from batch sampling
    2. Single state with smaller batch: Broadcast/expand to match
    3. Matching state: Pass through unchanged

    Attributes:
        num_layers: Number of LSTM layers.
        hidden_size: LSTM hidden dimension.
        device: Target device for tensors.

    Example:
        >>> handler = HiddenStateHandler(num_layers=1, hidden_size=128, device="cpu")
        >>> state = {"h": torch.zeros(1, 1, 128), "c": torch.zeros(1, 1, 128)}
        >>> h, c, mode = handler.prepare_state(state, batch_size=4)
        >>> h.shape
        torch.Size([1, 4, 128])
        >>> mode
        'broadcast'
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        device: Union[str, torch.device] = "cpu",
    ):
        """Initialize the hidden state handler.

        Args:
            num_layers: Number of LSTM layers.
            hidden_size: LSTM hidden dimension.
            device: Target device for tensors.
        """
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def prepare_state(
        self,
        state: Optional[Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Prepare hidden state for the given batch size.

        Args:
            state: Input state, can be:
                - None: Initialize to zeros
                - Dict with "h", "c": Single state
                - List[Dict]: List of individual states
                - Tianshou Batch with "h", "c": Batch state object
            batch_size: Target batch size.

        Returns:
            Tuple of (h, c, handling_mode) where:
                - h: Hidden state tensor (num_layers, batch_size, hidden_size)
                - c: Cell state tensor (num_layers, batch_size, hidden_size)
                - handling_mode: String describing how state was handled:
                    - "zero_init": Initialized to zeros
                    - "stacked": Stacked from list
                    - "passthrough": Passed through unchanged
                    - "broadcast": Single state broadcast to batch
                    - "fallback_zero": Mismatch, fell back to zeros
        """
        if state is None:
            h, c = self._create_zero_state(batch_size)
            return h, c, "zero_init"

        # Handle Tianshou Batch type - convert to dict
        if TIANSHOU_AVAILABLE and isinstance(state, Batch):
            state = self._convert_batch_to_dict(state, batch_size)
            if state is None:
                h, c = self._create_zero_state(batch_size)
                return h, c, "zero_init"

        if isinstance(state, list):
            h, c = self._stack_states(state, batch_size)
            return h, c, "stacked"

        if isinstance(state, dict):
            return self._handle_dict_state(state, batch_size)

        raise ValueError(f"Unsupported state type: {type(state)}")

    def _convert_batch_to_dict(
        self,
        batch_state,
        batch_size: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Convert Tianshou Batch state to dict format.

        Tianshou may pass hidden states as a Batch object. This method
        extracts the "h" and "c" tensors from the Batch.

        Args:
            batch_state: Tianshou Batch object containing hidden states.
            batch_size: Target batch size.

        Returns:
            Dict with "h" and "c" keys, or None if empty/invalid.
        """
        # Check if empty - Batch has __len__ but not is_empty()
        try:
            if len(batch_state) == 0:
                return None
        except TypeError:
            # If len() fails, try other methods
            pass

        # Try to extract h and c from Batch
        try:
            h = batch_state.get("h", None)
            c = batch_state.get("c", None)

            if h is None or c is None:
                # Check for hidden attribute
                if hasattr(batch_state, "hidden") and batch_state.hidden is not None:
                    hidden = batch_state.hidden
                    if isinstance(hidden, tuple) and len(hidden) == 2:
                        h, c = hidden
                    elif isinstance(hidden, dict):
                        h = hidden.get("h")
                        c = hidden.get("c")

            if h is not None and c is not None:
                return {"h": h, "c": c}

            return None

        except Exception:
            return None

    def _create_zero_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized hidden state.

        Args:
            batch_size: Batch size for the state.

        Returns:
            Tuple of (h, c) zero tensors.
        """
        h = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=self.device,
        )
        c = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=self.device,
        )
        return h, c

    def _stack_states(
        self,
        states: List[Dict[str, torch.Tensor]],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack a list of individual hidden states.

        This handles the case where Tianshou passes a list of states
        from different samples in the batch.

        Args:
            states: List of state dicts, each with "h" and "c".
            batch_size: Expected batch size (should match len(states)).

        Returns:
            Stacked (h, c) tensors.
        """
        if len(states) == 0:
            logger.warning("Empty state list received, creating zero state.")
            return self._create_zero_state(batch_size)

        if len(states) != batch_size:
            logger.warning(
                f"State list length {len(states)} doesn't match batch size {batch_size}. "
                f"This may indicate a bug in the training pipeline."
            )

        # Stack all hidden states along batch dimension
        # Each state["h"] is (num_layers, 1, hidden_size) or (num_layers, n, hidden_size)
        try:
            h_list = [s["h"].to(self.device) for s in states]
            c_list = [s["c"].to(self.device) for s in states]

            # Concatenate along batch dimension (dim=1)
            h = torch.cat(h_list, dim=1)  # (num_layers, batch_size, hidden_size)
            c = torch.cat(c_list, dim=1)

            return h, c
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Error stacking states: {e}. Falling back to zeros.")
            return self._create_zero_state(batch_size)

    def _handle_dict_state(
        self,
        state: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Handle a single dict state.

        Scenarios:
        1. State batch matches input batch: Pass through
        2. State has batch=1, input has batch>1: Expand/broadcast
        3. Mismatch: Log warning, fall back to zeros

        Args:
            state: Dict with "h" and "c" tensors.
            batch_size: Target batch size.

        Returns:
            Tuple of (h, c, handling_mode).
        """
        h = state["h"].to(self.device)
        c = state["c"].to(self.device)

        state_batch = h.shape[1]

        if state_batch == batch_size:
            # Perfect match - pass through
            return h, c, "passthrough"

        if state_batch == 1 and batch_size > 1:
            # Single state, broadcast to batch
            h = h.expand(-1, batch_size, -1).contiguous()
            c = c.expand(-1, batch_size, -1).contiguous()
            return h, c, "broadcast"

        # Mismatch case - fall back to zeros with debug log
        logger.debug(
            f"Hidden state batch size mismatch: state has {state_batch}, "
            f"input has {batch_size}. Falling back to zero state."
        )
        h, c = self._create_zero_state(batch_size)
        return h, c, "fallback_zero"


def prepare_hidden_state(
    state: Optional[Union[Dict, List[Dict]]],
    batch_size: int,
    num_layers: int,
    hidden_size: int,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Convenience function for preparing hidden state.

    This is a functional wrapper around HiddenStateHandler for simple use cases.

    Args:
        state: Input hidden state (None, dict, or list of dicts).
        batch_size: Target batch size.
        num_layers: Number of LSTM layers.
        hidden_size: LSTM hidden dimension.
        device: Target device.

    Returns:
        Tuple of (h, c, handling_mode).

    Example:
        >>> h, c, mode = prepare_hidden_state(
        ...     state=None,
        ...     batch_size=4,
        ...     num_layers=1,
        ...     hidden_size=128,
        ...     device="cpu",
        ... )
        >>> mode
        'zero_init'
    """
    handler = HiddenStateHandler(num_layers, hidden_size, device)
    return handler.prepare_state(state, batch_size)


def validate_hidden_state(
    h: torch.Tensor,
    c: torch.Tensor,
    expected_layers: int,
    expected_batch: int,
    expected_hidden: int,
) -> bool:
    """Validate hidden state tensor shapes.

    Args:
        h: Hidden state tensor.
        c: Cell state tensor.
        expected_layers: Expected number of LSTM layers.
        expected_batch: Expected batch size.
        expected_hidden: Expected hidden dimension.

    Returns:
        True if shapes are valid, False otherwise.

    Example:
        >>> h = torch.zeros(1, 4, 128)
        >>> c = torch.zeros(1, 4, 128)
        >>> validate_hidden_state(h, c, 1, 4, 128)
        True
    """
    expected_shape = (expected_layers, expected_batch, expected_hidden)
    return tuple(h.shape) == expected_shape and tuple(c.shape) == expected_shape


def get_handling_mode_description(mode: str) -> str:
    """Get human-readable description of handling mode.

    Args:
        mode: Handling mode string.

    Returns:
        Description of what the mode means.
    """
    descriptions = {
        "zero_init": "Hidden state initialized to zeros (no prior state)",
        "stacked": "Multiple states stacked from batch sampling",
        "passthrough": "State passed through unchanged (batch size matched)",
        "broadcast": "Single state broadcast to batch size",
        "fallback_zero": "Batch mismatch, fell back to zero state (warning)",
    }
    return descriptions.get(mode, f"Unknown mode: {mode}")
