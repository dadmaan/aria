"""Sequence-based replay buffer for recurrent networks.

This module implements a replay buffer that stores LSTM hidden states
and samples contiguous sequences for proper recurrent training.

Based on R2D2: "Recurrent Experience Replay in Distributed RL"
(Kapturowski et al., 2019)

"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import torch
import logging
import warnings

from tianshou.data import ReplayBuffer, Batch

logger = logging.getLogger(__name__)


@dataclass
class SequenceConfig:
    """Configuration for sequence-based sampling.

    Attributes:
        sequence_length: Length of sampled sequences (default: 16)
        store_hidden_states: Whether to store LSTM hidden states
        respect_episode_boundaries: Don't sample across episode boundaries
        min_sequence_length: Minimum valid sequence length for edge cases
        hidden_state_keys: Keys for LSTM hidden state tuple (h, c)
    """

    sequence_length: int = 16
    store_hidden_states: bool = True
    respect_episode_boundaries: bool = True
    min_sequence_length: int = 4
    hidden_state_keys: Tuple[str, str] = field(default=("h", "c"))


class SequenceReplayBuffer:
    """Replay buffer with sequence-based sampling for recurrent networks.

    This buffer extends standard experience replay to support:
    1. Storage of LSTM hidden states per transition
    2. Sampling of contiguous sequences
    3. Proper handling of episode boundaries

    Attributes:
        size: Maximum buffer size
        config: Sequence sampling configuration (SequenceConfig)

    Example:
        >>> buffer = SequenceReplayBuffer(size=10000, sequence_length=16)
        >>> buffer.add(obs=obs, act=act, rew=rew, hidden=hidden_state, ...)
        >>> batch = buffer.sample_sequences(batch_size=32)
        >>> # batch contains 32 sequences of length 16
    """

    def __init__(
        self,
        size: int,
        sequence_length: int = 16,
        store_hidden_states: bool = True,
        respect_episode_boundaries: bool = True,
        min_sequence_length: int = 4,
        **kwargs,
    ):
        """Initialize sequence replay buffer.

        Args:
            size: Maximum number of transitions to store
            sequence_length: Length of sequences to sample
            store_hidden_states: Whether to store LSTM hidden states
            respect_episode_boundaries: Whether to avoid cross-episode sequences
            min_sequence_length: Minimum valid sequence length
            **kwargs: Additional arguments passed to base buffer
        """
        self.size = size
        self.config = SequenceConfig(
            sequence_length=sequence_length,
            store_hidden_states=store_hidden_states,
            respect_episode_boundaries=respect_episode_boundaries,
            min_sequence_length=min_sequence_length,
        )

        # Use Tianshou's ReplayBuffer as base
        self._buffer = ReplayBuffer(size=size, **kwargs)

        # Track episode boundaries
        self._episode_starts: List[int] = []
        self._episode_ends: List[int] = []
        self._current_episode_start: int = 0

        # Hidden state storage (separate for efficiency)
        # Use int keys for indices, store CPU tensors
        self._hidden_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self._hidden_states_next: Dict[int, Dict[str, torch.Tensor]] = {}

        # Track current write position for circular buffer cleanup
        self._write_pos = 0

        logger.info(
            f"SequenceReplayBuffer initialized: "
            f"size={size}, seq_len={sequence_length}, "
            f"store_hidden={store_hidden_states}"
        )

    def add(
        self,
        obs: np.ndarray,
        act: Union[int, np.ndarray],
        rew: float,
        terminated: bool = False,
        truncated: bool = False,
        obs_next: Optional[np.ndarray] = None,
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        hidden_next: Optional[Dict[str, torch.Tensor]] = None,
        info: Optional[Dict[str, Any]] = None,
        done: Optional[bool] = None,  # Backward compatibility
        **kwargs,
    ) -> int:
        """Add a transition to the buffer.

        Args:
            obs: Current observation
            act: Action taken
            rew: Reward received
            terminated: Episode terminated (goal reached or failure)
            truncated: Episode truncated (time limit reached)
            obs_next: Next observation
            hidden: LSTM hidden state at current step (optional)
            hidden_next: LSTM hidden state at next step (optional)
            info: Additional info dict
            done: Deprecated - use terminated/truncated. Treated as terminated.
            **kwargs: Additional data to store

        Returns:
            Index where transition was stored

        Note:
            Use terminated=True for natural episode ends (goal/failure).
            Use truncated=True for artificial ends (time limit).
            The episode is done when terminated OR truncated.
        """
        # Backward compatibility: if done is passed, treat as terminated
        if done is not None:
            terminated = done

        # Clean up old hidden states if we're overwriting
        if len(self._buffer) >= self.size:
            old_idx = self._write_pos % self.size
            self._cleanup_hidden_states(old_idx)

        # Add to base buffer (Tianshou 2.0 uses terminated/truncated)
        batch_data = Batch(
            obs=obs,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            obs_next=obs_next if obs_next is not None else obs,
            info=info or {},
            **kwargs,
        )
        self._buffer.add(batch_data)  # Get the actual index where data was stored
        idx = (len(self._buffer) - 1) % self.size

        # Store hidden states if provided
        if self.config.store_hidden_states:
            if hidden is not None:
                self._hidden_states[idx] = {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in hidden.items()
                }
            if hidden_next is not None:
                self._hidden_states_next[idx] = {
                    k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in hidden_next.items()
                }

        # Track episode boundaries (done = terminated or truncated)
        done = terminated or truncated
        if done:
            self._episode_ends.append(idx)
            self._episode_starts.append(self._current_episode_start)
            self._current_episode_start = (idx + 1) % self.size

        self._write_pos += 1

        return idx

    def _cleanup_hidden_states(self, idx: int):
        """Clean up hidden states at index being overwritten."""
        if idx in self._hidden_states:
            del self._hidden_states[idx]
        if idx in self._hidden_states_next:
            del self._hidden_states_next[idx]

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: Optional[int] = None,
    ) -> Batch:
        """Sample batch of contiguous sequences.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence (default: config value)

        Returns:
            Batch containing sequences with shape (batch, seq_len, ...)
        """
        seq_len = sequence_length or self.config.sequence_length

        # Find valid sequence start indices
        valid_starts = self._find_valid_sequence_starts(seq_len)

        if len(valid_starts) == 0:
            warnings.warn(
                f"No valid sequences found for length {seq_len}. "
                f"Buffer has {len(self._buffer)} transitions."
            )
            # Return empty batch
            return Batch()

        if len(valid_starts) < batch_size:
            logger.warning(
                f"Not enough valid sequences: need {batch_size}, "
                f"have {len(valid_starts)}. Using replacement sampling."
            )
            starts = np.random.choice(valid_starts, size=batch_size, replace=True)
        else:
            starts = np.random.choice(valid_starts, size=batch_size, replace=False)

        return self._gather_sequences(starts, seq_len)

    def _gather_sequences(self, starts: np.ndarray, seq_len: int) -> Batch:
        """Gather sequences from given start indices.

        Args:
            starts: Array of start indices
            seq_len: Sequence length

        Returns:
            Batch with stacked sequences
        """
        sequences = []
        hidden_states = []
        hidden_states_next = []

        for start in starts:
            # Get contiguous indices for this sequence
            indices = [(start + i) % self.size for i in range(seq_len)]

            # Get data from buffer - collect individual transitions
            seq_data = []
            for idx in indices:
                seq_data.append(self._buffer[idx])

            # Stack into sequence batch
            seq_batch = Batch.stack(seq_data)
            sequences.append(seq_batch)

            # Get hidden states (only first step for burn-in)
            if self.config.store_hidden_states:
                h_first = self._get_hidden_state(indices[0], self._hidden_states)
                h_next_first = self._get_hidden_state(
                    indices[0], self._hidden_states_next
                )
                hidden_states.append(h_first)
                hidden_states_next.append(h_next_first)

        # Stack sequences into batch (batch_size, seq_len, ...)
        batch = Batch.stack(sequences)

        # Add hidden states if stored
        if self.config.store_hidden_states:
            if any(h is not None for h in hidden_states):
                batch.hidden = self._stack_hidden_states(hidden_states)
            if any(h is not None for h in hidden_states_next):
                batch.hidden_next = self._stack_hidden_states(hidden_states_next)

        # Store sequence starts for priority updates
        batch.seq_starts = starts

        return batch

    def _find_valid_sequence_starts(self, seq_len: int) -> List[int]:
        """Find all valid sequence start indices.

        A valid start index is one where:
        1. There are seq_len consecutive transitions
        2. The sequence doesn't cross episode boundaries (if configured)

        Args:
            seq_len: Required sequence length

        Returns:
            List of valid start indices
        """
        valid_starts = []
        buffer_len = len(self._buffer)

        if buffer_len < seq_len:
            return valid_starts

        if not self.config.respect_episode_boundaries:
            # Simple case: any index where we have enough data
            # For circular buffer, need to handle wraparound
            max_start = buffer_len - seq_len
            for i in range(max_start + 1):
                valid_starts.append(i)
            return valid_starts

        # Complex case: respect episode boundaries
        # Build set of episode end indices for fast lookup
        episode_ends_set = set(self._episode_ends)

        # For each potential start, check if sequence crosses episode boundary
        max_start = buffer_len - seq_len
        for i in range(max_start + 1):
            # Check if any index in [i, i+seq_len-1) is an episode end
            # Note: The last step CAN be a terminal (that's fine)
            is_valid = True
            for j in range(seq_len - 1):  # Check all but last step
                idx = (i + j) % self.size
                if idx in episode_ends_set:
                    is_valid = False
                    break

            if is_valid:
                valid_starts.append(i)

        return valid_starts

    def _get_hidden_state(
        self,
        idx: int,
        storage: Dict[int, Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get hidden state for a specific index.

        Args:
            idx: Buffer index
            storage: Hidden state storage dict

        Returns:
            Hidden state dict or None if not available
        """
        return storage.get(idx)

    def _stack_hidden_states(
        self,
        states: List[Optional[Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """Stack list of hidden states into batched tensors.

        Args:
            states: List of hidden state dicts (some may be None)

        Returns:
            Dict with batched hidden state tensors
        """
        # Filter out None values
        valid_states = [s for s in states if s is not None]

        if not valid_states:
            return {}

        # Stack each key
        result = {}
        for key in valid_states[0].keys():
            tensors = []
            for s in states:
                if s is not None and key in s:
                    tensors.append(s[key])
                else:
                    # Use zeros for missing states
                    ref_tensor = valid_states[0][key]
                    tensors.append(torch.zeros_like(ref_tensor))

            # Each tensor is (layers, 1, hidden) - concatenate on batch dim (dim=1)
            result[key] = torch.cat(tensors, dim=1)

        return result

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Standard sample (for compatibility with Tianshou).

        For recurrent training, prefer sample_sequences().

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (batch, indices)
        """
        return self._buffer.sample(batch_size)

    def __len__(self) -> int:
        """Return number of stored transitions."""
        return len(self._buffer)

    def __getitem__(self, index):
        """Get item by index."""
        return self._buffer[index]

    @property
    def buffer(self) -> ReplayBuffer:
        """Get underlying Tianshou buffer."""
        return self._buffer

    def reset(self):
        """Reset buffer to empty state."""
        self._buffer.reset()
        self._episode_starts.clear()
        self._episode_ends.clear()
        self._current_episode_start = 0
        self._hidden_states.clear()
        self._hidden_states_next.clear()
        self._write_pos = 0
        logger.info("SequenceReplayBuffer reset")


class PrioritizedSequenceReplayBuffer(SequenceReplayBuffer):
    """Sequence buffer with prioritized sampling.

    Priority is computed at the sequence level (max TD-error in sequence)
    as recommended by R2D2.

    Attributes:
        alpha: Priority exponent (0 = uniform, 1 = full priority)
        beta: Importance sampling exponent (starts low, anneals to 1)
    """

    def __init__(
        self,
        size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        sequence_length: int = 16,
        **kwargs,
    ):
        """Initialize prioritized sequence buffer.

        Args:
            size: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling weight exponent
            sequence_length: Length of sampled sequences
            **kwargs: Additional arguments
        """
        super().__init__(size=size, sequence_length=sequence_length, **kwargs)

        self.alpha = alpha
        self.beta = beta
        self._beta_start = beta

        # Sequence-level priorities (keyed by start index)
        self._seq_priorities: Dict[int, float] = {}
        self._max_priority: float = 1.0

        logger.info(
            f"PrioritizedSequenceReplayBuffer initialized: "
            f"alpha={alpha}, beta={beta}"
        )

    def update_priorities(
        self,
        sequence_starts: np.ndarray,
        td_errors: np.ndarray,
    ):
        """Update priorities for sampled sequences.

        Args:
            sequence_starts: Start indices of sampled sequences
            td_errors: TD errors for each sequence (typically max over sequence)
        """
        for start, error in zip(sequence_starts, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self._seq_priorities[int(start)] = priority
            self._max_priority = max(self._max_priority, priority)

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: Optional[int] = None,
    ) -> Tuple[Batch, np.ndarray, np.ndarray]:
        """Sample sequences with prioritized sampling.

        Args:
            batch_size: Number of sequences to sample
            sequence_length: Sequence length (default: config value)

        Returns:
            Tuple of (batch, sequence_starts, importance_weights)
        """
        seq_len = sequence_length or self.config.sequence_length
        valid_starts = self._find_valid_sequence_starts(seq_len)

        if len(valid_starts) == 0:
            warnings.warn(
                f"No valid sequences found for length {seq_len}. "
                f"Buffer has {len(self._buffer)} transitions."
            )
            return Batch(), np.array([]), np.array([])

        # Get priorities for valid starts
        priorities = np.array(
            [self._seq_priorities.get(s, self._max_priority) for s in valid_starts]
        )

        # Compute sampling probabilities
        probs = priorities / priorities.sum()

        # Sample based on priority
        use_replacement = len(valid_starts) < batch_size
        sample_indices = np.random.choice(
            len(valid_starts),
            size=batch_size,
            replace=use_replacement,
            p=probs,
        )

        starts = np.array([valid_starts[i] for i in sample_indices])

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta) / max_j(w_j)
        sampled_probs = probs[sample_indices]
        weights = (len(valid_starts) * sampled_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize to [0, 1]

        # Gather sequences
        batch = self._gather_sequences(starts, seq_len)

        return batch, starts, weights

    def set_beta(self, beta: float):
        """Set importance sampling exponent (for annealing).

        Args:
            beta: New beta value (should increase from initial to 1.0)
        """
        self.beta = beta

    def anneal_beta(self, fraction: float):
        """Anneal beta from start value to 1.0.

        Args:
            fraction: Training progress fraction (0.0 to 1.0)
        """
        self.beta = self._beta_start + fraction * (1.0 - self._beta_start)

    def reset(self):
        """Reset buffer including priorities."""
        super().reset()
        self._seq_priorities.clear()
        self._max_priority = 1.0
        self.beta = self._beta_start
        logger.info("PrioritizedSequenceReplayBuffer reset")
