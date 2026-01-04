"""Conservative Policy Updater for True HIL Learning.

This module implements actual policy parameter updates based on
human preference feedback, addressing the investigation finding
that previous "learning" was ephemeral Q-value modification.

The key insight is that Q-value penalties alone don't constitute
learning - we need actual gradient updates to policy parameters
for preferences to persist across sessions.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience tuple with optional preference feedback.

    Attributes:
        state: The observed state (numpy array or torch tensor).
        action: The action taken (int for discrete, array for continuous).
        reward: The base reward received from the environment.
        next_state: The resulting state after taking the action.
        done: Whether this transition ended the episode.
        preference_feedback: Optional human preference signal in [-1, 1].
            Positive values indicate desirable behavior.
            Negative values indicate undesirable behavior.
            None indicates no feedback was provided.
        cluster_id: Optional cluster ID for GHSOM-based feedback.
        timestamp: Optional timestamp for temporal ordering.
        priority: Priority weight for prioritized replay sampling.
    """

    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    preference_feedback: Optional[float] = None
    cluster_id: Optional[int] = None
    timestamp: Optional[float] = None
    priority: float = 1.0


class ExperienceBuffer:
    """Experience replay buffer with prioritized sampling support.

    Stores experience tuples for policy learning with support for
    prioritized experience replay based on preference feedback magnitude.
    Uses FIFO eviction when capacity is exceeded.

    Attributes:
        max_size: Maximum number of experiences to store.
        prioritized: Whether to use prioritized sampling.
        priority_alpha: Exponent for priority-based sampling (0=uniform, 1=full priority).
    """

    def __init__(
        self,
        max_size: int = 10000,
        prioritized: bool = True,
        priority_alpha: float = 0.6,
    ):
        """Initialize the experience buffer.

        Args:
            max_size: Maximum buffer capacity before FIFO eviction.
            prioritized: Enable prioritized experience replay.
            priority_alpha: Controls priority vs uniform sampling balance.
        """
        self.max_size = max_size
        self.prioritized = prioritized
        self.priority_alpha = priority_alpha
        self._buffer: deque = deque(maxlen=max_size)
        self._priorities: deque = deque(maxlen=max_size)

        logger.debug(
            f"Initialized ExperienceBuffer with max_size={max_size}, "
            f"prioritized={prioritized}, alpha={priority_alpha}"
        )

    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        preference_feedback: Optional[float] = None,
        cluster_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a new experience to the buffer.

        Args:
            state: The observed state.
            action: The action taken.
            reward: The base reward received.
            next_state: The resulting state.
            done: Whether the episode ended.
            preference_feedback: Optional preference signal in [-1, 1].
            cluster_id: Optional cluster ID for the state.
            timestamp: Optional timestamp for ordering.
        """
        # Compute priority based on feedback magnitude
        if preference_feedback is not None:
            # Higher magnitude feedback gets higher priority
            priority = 1.0 + abs(preference_feedback)
        else:
            priority = 1.0

        experience = Experience(
            state=np.asarray(state),
            action=action,
            reward=float(reward),
            next_state=np.asarray(next_state),
            done=bool(done),
            preference_feedback=preference_feedback,
            cluster_id=cluster_id,
            timestamp=timestamp,
            priority=priority,
        )

        self._buffer.append(experience)
        self._priorities.append(priority**self.priority_alpha)

        logger.debug(
            f"Added experience: action={action}, reward={reward:.3f}, "
            f"feedback={preference_feedback}, priority={priority:.3f}"
        )

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of Experience objects.

        Raises:
            ValueError: If batch_size exceeds buffer size.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Requested batch_size={batch_size} exceeds "
                f"buffer size={len(self._buffer)}"
            )

        if self.prioritized and len(self._priorities) > 0:
            # Prioritized sampling
            priorities = np.array(self._priorities)
            probabilities = priorities / priorities.sum()

            # Handle potential NaN from zero sum
            if np.isnan(probabilities).any():
                logger.warning("NaN probabilities detected, falling back to uniform")
                indices = np.random.choice(len(self._buffer), batch_size, replace=False)
            else:
                indices = np.random.choice(
                    len(self._buffer), batch_size, replace=False, p=probabilities
                )
        else:
            # Uniform sampling
            indices = np.random.choice(len(self._buffer), batch_size, replace=False)

        batch = [self._buffer[i] for i in indices]
        logger.debug(f"Sampled batch of {batch_size} experiences")
        return batch

    def sample_with_feedback(self, batch_size: int) -> List[Experience]:
        """Sample experiences that have preference feedback.

        Preferentially samples experiences with non-None feedback,
        useful for focused policy updates on labeled transitions.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of Experience objects with feedback (may be smaller
            than batch_size if insufficient feedback samples exist).
        """
        feedback_experiences = [
            exp for exp in self._buffer if exp.preference_feedback is not None
        ]

        if len(feedback_experiences) == 0:
            logger.warning("No experiences with feedback available")
            return []

        actual_batch_size = min(batch_size, len(feedback_experiences))
        indices = np.random.choice(
            len(feedback_experiences), actual_batch_size, replace=False
        )

        return [feedback_experiences[i] for i in indices]

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._buffer.clear()
        self._priorities.clear()
        logger.info("Experience buffer cleared")

    def __len__(self) -> int:
        """Return the current number of experiences in the buffer."""
        return len(self._buffer)

    def get_feedback_statistics(self) -> Dict[str, float]:
        """Compute statistics about feedback in the buffer.

        Returns:
            Dictionary with feedback statistics.
        """
        feedbacks = [
            exp.preference_feedback
            for exp in self._buffer
            if exp.preference_feedback is not None
        ]

        if len(feedbacks) == 0:
            return {
                "count": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "mean": 0.0,
                "std": 0.0,
            }

        feedbacks = np.array(feedbacks)
        return {
            "count": len(feedbacks),
            "positive_ratio": float((feedbacks > 0).mean()),
            "negative_ratio": float((feedbacks < 0).mean()),
            "mean": float(feedbacks.mean()),
            "std": float(feedbacks.std()),
        }


class PreferenceShapedReward:
    """Transform base rewards using human preference feedback.

    Implements reward shaping based on preference signals, allowing
    the policy to learn from human feedback. The shaped reward
    incorporates both the environment reward and preference signal.

    Formula: shaped_reward = base_reward + alpha * feedback_signal

    Where feedback_signal is normalized to [-1, 1] and can be further
    modified based on cluster membership (desirable vs undesirable).

    Attributes:
        alpha: Scaling factor for preference contribution.
        cluster_bonus: Additional bonus/penalty for cluster membership.
        normalize_feedback: Whether to clip feedback to [-1, 1].
    """

    def __init__(
        self,
        alpha: float = 0.1,
        cluster_bonus: float = 0.05,
        normalize_feedback: bool = True,
    ):
        """Initialize the preference reward shaper.

        Args:
            alpha: Scaling factor for preference signal (default 0.1).
            cluster_bonus: Bonus for desirable / penalty for undesirable clusters.
            normalize_feedback: Whether to clip feedback to [-1, 1].
        """
        self.alpha = alpha
        self.cluster_bonus = cluster_bonus
        self.normalize_feedback = normalize_feedback

        logger.debug(
            f"Initialized PreferenceShapedReward with alpha={alpha}, "
            f"cluster_bonus={cluster_bonus}"
        )

    def _normalize_feedback(self, feedback: float) -> float:
        """Normalize feedback signal to [-1, 1] range.

        Args:
            feedback: Raw feedback value.

        Returns:
            Normalized feedback in [-1, 1].
        """
        if self.normalize_feedback:
            return float(np.clip(feedback, -1.0, 1.0))
        return float(feedback)

    def shape_reward(
        self,
        base_reward: float,
        feedback: Optional[float],
        cluster_id: Optional[int] = None,
        desirable_clusters: Optional[List[int]] = None,
        undesirable_clusters: Optional[List[int]] = None,
    ) -> float:
        """Shape the reward using preference feedback and cluster info.

        Args:
            base_reward: The original environment reward.
            feedback: Human preference signal (can be None).
            cluster_id: Optional cluster ID for the current state.
            desirable_clusters: List of cluster IDs marked as desirable.
            undesirable_clusters: List of cluster IDs marked as undesirable.

        Returns:
            The shaped reward incorporating preference information.
        """
        shaped_reward = float(base_reward)

        # Apply feedback-based shaping
        if feedback is not None:
            normalized_feedback = self._normalize_feedback(feedback)
            shaped_reward += self.alpha * normalized_feedback
            logger.debug(
                f"Applied feedback shaping: {base_reward:.4f} + "
                f"{self.alpha} * {normalized_feedback:.4f} = {shaped_reward:.4f}"
            )

        # Apply cluster-based shaping
        if cluster_id is not None:
            if desirable_clusters and cluster_id in desirable_clusters:
                shaped_reward += self.cluster_bonus
                logger.debug(
                    f"Applied desirable cluster bonus: cluster={cluster_id}, "
                    f"bonus={self.cluster_bonus}"
                )
            elif undesirable_clusters and cluster_id in undesirable_clusters:
                shaped_reward -= self.cluster_bonus
                logger.debug(
                    f"Applied undesirable cluster penalty: cluster={cluster_id}, "
                    f"penalty={self.cluster_bonus}"
                )

        return shaped_reward

    def batch_shape_rewards(
        self,
        base_rewards: np.ndarray,
        feedbacks: np.ndarray,
        cluster_ids: Optional[np.ndarray] = None,
        desirable_clusters: Optional[List[int]] = None,
        undesirable_clusters: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Shape a batch of rewards efficiently.

        Args:
            base_rewards: Array of base rewards.
            feedbacks: Array of feedback values (can contain NaN for no feedback).
            cluster_ids: Optional array of cluster IDs.
            desirable_clusters: List of desirable cluster IDs.
            undesirable_clusters: List of undesirable cluster IDs.

        Returns:
            Array of shaped rewards.
        """
        shaped_rewards = base_rewards.copy().astype(np.float32)

        # Apply feedback shaping where available
        valid_feedback_mask = ~np.isnan(feedbacks)
        if valid_feedback_mask.any():
            normalized = np.clip(feedbacks[valid_feedback_mask], -1.0, 1.0)
            shaped_rewards[valid_feedback_mask] += self.alpha * normalized

        # Apply cluster shaping if provided
        if cluster_ids is not None:
            if desirable_clusters:
                desirable_mask = np.isin(cluster_ids, desirable_clusters)
                shaped_rewards[desirable_mask] += self.cluster_bonus
            if undesirable_clusters:
                undesirable_mask = np.isin(cluster_ids, undesirable_clusters)
                shaped_rewards[undesirable_mask] -= self.cluster_bonus

        return shaped_rewards


class ConservativePolicyUpdater:
    """Performs actual gradient updates on policy parameters.

    This class addresses the critical issue where previous implementations
    only modified Q-values temporarily without updating policy parameters.
    It implements conservative updates with small learning rates and
    gradient clipping to ensure stable learning from human feedback.

    The updater supports both DQN and DRQN style policies and tracks
    update statistics for monitoring learning progress.

    Attributes:
        policy: The neural network policy to update.
        optimizer: PyTorch optimizer for gradient updates.
        learning_rate: Learning rate for updates.
        update_frequency: How often to perform updates (in steps).
        min_buffer_size: Minimum experiences before updating.
        batch_size: Batch size for updates.
        gamma: Discount factor for TD learning.
        gradient_clip: Maximum gradient norm.
    """

    def __init__(
        self,
        policy: nn.Module,
        learning_rate: float = 1e-4,
        update_frequency: int = 10,
        min_buffer_size: int = 32,
        batch_size: int = 32,
        gamma: float = 0.99,
        gradient_clip: float = 1.0,
        reward_shaper: Optional[PreferenceShapedReward] = None,
        buffer_size: int = 10000,
        device: Optional[torch.device] = None,
    ):
        """Initialize the conservative policy updater.

        Args:
            policy: The policy network to update.
            learning_rate: Learning rate for Adam optimizer.
            update_frequency: Steps between policy updates.
            min_buffer_size: Minimum buffer size before updates begin.
            batch_size: Batch size for gradient updates.
            gamma: Discount factor for TD targets.
            gradient_clip: Maximum gradient norm for clipping.
            reward_shaper: Optional PreferenceShapedReward instance.
            buffer_size: Maximum size of experience buffer.
            device: Torch device (auto-detected if None).
        """
        self.policy = policy
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.gradient_clip = gradient_clip

        # Auto-detect device
        if device is None:
            self.device = next(policy.parameters()).device
        else:
            self.device = device

        # Initialize optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-8)

        # Initialize experience buffer
        self.buffer = ExperienceBuffer(max_size=buffer_size, prioritized=True)

        # Initialize reward shaper
        self.reward_shaper = reward_shaper or PreferenceShapedReward()

        # Tracking variables
        self._step_count = 0
        self._update_count = 0
        self._total_loss = 0.0
        self._total_gradient_norm = 0.0

        # Cluster preferences (can be set externally)
        self.desirable_clusters: List[int] = []
        self.undesirable_clusters: List[int] = []

        logger.info(
            f"Initialized ConservativePolicyUpdater: lr={learning_rate}, "
            f"update_freq={update_frequency}, min_buffer={min_buffer_size}, "
            f"batch_size={batch_size}, gamma={gamma}, clip={gradient_clip}"
        )

    def set_cluster_preferences(
        self, desirable: List[int], undesirable: List[int]
    ) -> None:
        """Set the desirable and undesirable cluster IDs.

        Args:
            desirable: List of cluster IDs marked as desirable.
            undesirable: List of cluster IDs marked as undesirable.
        """
        self.desirable_clusters = list(desirable)
        self.undesirable_clusters = list(undesirable)
        logger.info(
            f"Updated cluster preferences: {len(desirable)} desirable, "
            f"{len(undesirable)} undesirable"
        )

    def add_experience(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
        feedback: Optional[float] = None,
        cluster_id: Optional[int] = None,
    ) -> None:
        """Add an experience to the buffer.

        Args:
            state: The observed state.
            action: The action taken.
            reward: The base reward from the environment.
            next_state: The next state.
            done: Whether the episode ended.
            feedback: Optional preference feedback in [-1, 1].
            cluster_id: Optional cluster ID for the state.
        """
        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            preference_feedback=feedback,
            cluster_id=cluster_id,
        )
        self._step_count += 1

    def should_update(self) -> bool:
        """Check if an update should be performed.

        Returns:
            True if buffer is large enough and update is due.
        """
        buffer_ready = len(self.buffer) >= self.min_buffer_size
        update_due = self._step_count % self.update_frequency == 0
        return buffer_ready and update_due and self._step_count > 0

    def _prepare_batch(
        self, experiences: List[Experience]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert experiences to tensors for training.

        Args:
            experiences: List of Experience objects.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors.
        """
        states = np.stack([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        next_states = np.stack([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences], dtype=np.float32)

        # Shape rewards using preference feedback
        base_rewards = np.array([exp.reward for exp in experiences])
        feedbacks = np.array(
            [
                (
                    exp.preference_feedback
                    if exp.preference_feedback is not None
                    else np.nan
                )
                for exp in experiences
            ]
        )
        cluster_ids = np.array(
            [
                exp.cluster_id if exp.cluster_id is not None else -1
                for exp in experiences
            ]
        )

        # Apply reward shaping
        shaped_rewards = self.reward_shaper.batch_shape_rewards(
            base_rewards=base_rewards,
            feedbacks=feedbacks,
            cluster_ids=cluster_ids if (cluster_ids >= 0).any() else None,
            desirable_clusters=self.desirable_clusters,
            undesirable_clusters=self.undesirable_clusters,
        )

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(shaped_rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def update_policy(self) -> Dict[str, float]:
        """Perform a gradient update on the policy parameters.

        This is the core method that implements TRUE policy learning,
        as opposed to ephemeral Q-value modifications.

        Returns:
            Dictionary containing update statistics.

        Raises:
            RuntimeError: If buffer doesn't have enough samples.
        """
        if len(self.buffer) < self.min_buffer_size:
            raise RuntimeError(
                f"Buffer size {len(self.buffer)} < min_buffer_size {self.min_buffer_size}"
            )

        # Sample experiences
        experiences = self.buffer.sample(self.batch_size)

        # Prepare batch
        states, actions, rewards, next_states, dones = self._prepare_batch(experiences)

        # Forward pass - compute current Q-values
        self.policy.train()
        q_values = self.policy(states)

        # Handle tuple output (q_values, state) from recurrent networks
        if isinstance(q_values, tuple):
            q_values = q_values[0]

        # For distributional RL (Rainbow), compute expected Q-values from distribution
        # Shape: (batch, n_actions, num_atoms) -> (batch, n_actions)
        if q_values.dim() == 3:
            # Rainbow returns probability distributions over atoms
            # We need to compute expected Q-value = sum(atoms * probs)
            # Get the atom support from the network if available
            if hasattr(self.policy, "v_min") and hasattr(self.policy, "v_max"):
                v_min = self.policy.v_min
                v_max = self.policy.v_max
                num_atoms = q_values.shape[-1]
                support = torch.linspace(
                    v_min, v_max, num_atoms, device=q_values.device
                )
                q_values = (q_values * support).sum(dim=-1)  # (batch, n_actions)
            else:
                # Fallback: just take mean over atoms
                q_values = q_values.mean(dim=-1)

        # Handle different action formats
        if actions.dim() == 1:
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            current_q = q_values.gather(1, actions).squeeze(1)

        # Compute target Q-values (using same network - could add target network)
        with torch.no_grad():
            next_q_values = self.policy(next_states)

            # Handle tuple output for next_q_values too
            if isinstance(next_q_values, tuple):
                next_q_values = next_q_values[0]

            # Handle distributional output
            if next_q_values.dim() == 3:
                if hasattr(self.policy, "v_min") and hasattr(self.policy, "v_max"):
                    v_min = self.policy.v_min
                    v_max = self.policy.v_max
                    num_atoms = next_q_values.shape[-1]
                    support = torch.linspace(
                        v_min, v_max, num_atoms, device=next_q_values.device
                    )
                    next_q_values = (next_q_values * support).sum(dim=-1)
                else:
                    next_q_values = next_q_values.mean(dim=-1)

            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss
        loss = nn.functional.mse_loss(current_q, target_q)

        # Check for NaN loss
        if torch.isnan(loss):
            logger.error("NaN loss detected, skipping update")
            return {
                "loss": float("nan"),
                "gradient_norm": 0.0,
                "update_performed": False,
            }

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        total_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        gradient_norm = total_norm**0.5

        # Gradient clipping for conservative updates
        clip_grad_norm_(self.policy.parameters(), self.gradient_clip)

        # Optimizer step - THIS IS THE ACTUAL POLICY UPDATE
        self.optimizer.step()

        # Update tracking
        self._update_count += 1
        self._total_loss += loss.item()
        self._total_gradient_norm += gradient_norm

        stats = {
            "loss": loss.item(),
            "gradient_norm": gradient_norm,
            "update_performed": True,
            "buffer_size": len(self.buffer),
            "mean_reward": rewards.mean().item(),
            "mean_q_value": current_q.mean().item(),
        }

        logger.debug(
            f"Policy update #{self._update_count}: loss={loss.item():.6f}, "
            f"grad_norm={gradient_norm:.6f}"
        )

        return stats

    def update_if_ready(self) -> Optional[Dict[str, float]]:
        """Perform update if conditions are met.

        Convenience method that checks should_update() and calls
        update_policy() if appropriate.

        Returns:
            Update stats if update was performed, None otherwise.
        """
        if self.should_update():
            return self.update_policy()
        return None

    def get_update_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about policy updates.

        Returns:
            Dictionary with gradient_norm, loss, num_updates, and more.
        """
        avg_loss = (
            self._total_loss / self._update_count if self._update_count > 0 else 0.0
        )
        avg_grad_norm = (
            self._total_gradient_norm / self._update_count
            if self._update_count > 0
            else 0.0
        )

        buffer_stats = self.buffer.get_feedback_statistics()

        return {
            "num_updates": self._update_count,
            "total_steps": self._step_count,
            "average_loss": avg_loss,
            "average_gradient_norm": avg_grad_norm,
            "buffer_size": len(self.buffer),
            "learning_rate": self.learning_rate,
            "feedback_stats": buffer_stats,
        }

    def reset_stats(self) -> None:
        """Reset update statistics (not the buffer)."""
        self._update_count = 0
        self._total_loss = 0.0
        self._total_gradient_norm = 0.0
        logger.info("Update statistics reset")

    def save_state(self, path: str) -> None:
        """Save the updater state including optimizer.

        Args:
            path: File path to save the state.
        """
        state = {
            "optimizer_state": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "update_count": self._update_count,
            "total_loss": self._total_loss,
            "total_gradient_norm": self._total_gradient_norm,
            "desirable_clusters": self.desirable_clusters,
            "undesirable_clusters": self.undesirable_clusters,
        }
        torch.save(state, path)
        logger.info(f"Saved updater state to {path}")

    def load_state(self, path: str) -> None:
        """Load the updater state.

        Args:
            path: File path to load the state from.
        """
        state = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._step_count = state["step_count"]
        self._update_count = state["update_count"]
        self._total_loss = state["total_loss"]
        self._total_gradient_norm = state["total_gradient_norm"]
        self.desirable_clusters = state.get("desirable_clusters", [])
        self.undesirable_clusters = state.get("undesirable_clusters", [])
        logger.info(f"Loaded updater state from {path}")


class LearningVerifier:
    """Verify that actual policy learning is occurring.

    This class tracks policy parameter changes across updates to
    confirm that gradients are being applied and parameters are
    changing. This addresses the investigation finding that previous
    "learning" was ephemeral.

    Attributes:
        parameter_snapshots: Stored parameter values for comparison.
        change_history: History of parameter changes over time.
    """

    def __init__(self, track_history: bool = True, max_history: int = 100):
        """Initialize the learning verifier.

        Args:
            track_history: Whether to maintain a history of changes.
            max_history: Maximum number of historical changes to track.
        """
        self.track_history = track_history
        self.max_history = max_history

        self._parameter_snapshot: Optional[Dict[str, torch.Tensor]] = None
        self._change_history: deque = deque(maxlen=max_history)
        self._snapshot_step: int = 0

        logger.debug(f"Initialized LearningVerifier with track_history={track_history}")

    def snapshot_parameters(
        self, policy: nn.Module, step: Optional[int] = None
    ) -> None:
        """Save a snapshot of current policy parameters.

        Args:
            policy: The policy network to snapshot.
            step: Optional step number for reference.
        """
        self._parameter_snapshot = {
            name: param.detach().clone()
            for name, param in policy.named_parameters()
            if param.requires_grad
        }
        self._snapshot_step = step if step is not None else 0

        logger.debug(
            f"Snapshotted {len(self._parameter_snapshot)} parameter tensors "
            f"at step {self._snapshot_step}"
        )

    def compute_parameter_change(self, policy: nn.Module) -> Dict[str, float]:
        """Compute the change in parameters since last snapshot.

        Args:
            policy: The policy network to compare.

        Returns:
            Dictionary with total_change (L2 norm), per-layer changes,
            and other statistics.

        Raises:
            RuntimeError: If no snapshot has been taken.
        """
        if self._parameter_snapshot is None:
            raise RuntimeError(
                "No parameter snapshot available. Call snapshot_parameters first."
            )

        changes = {}
        total_change = 0.0
        total_params = 0
        max_change = 0.0
        changed_params = 0

        for name, param in policy.named_parameters():
            if name in self._parameter_snapshot and param.requires_grad:
                delta = param.detach() - self._parameter_snapshot[name]
                layer_change = delta.norm(2).item()
                changes[name] = layer_change
                total_change += layer_change**2
                total_params += param.numel()
                max_change = max(max_change, delta.abs().max().item())

                if layer_change > 1e-10:
                    changed_params += param.numel()

        total_l2_change = total_change**0.5

        result = {
            "total_l2_change": total_l2_change,
            "max_element_change": max_change,
            "total_parameters": total_params,
            "changed_parameter_ratio": (
                changed_params / total_params if total_params > 0 else 0.0
            ),
            "per_layer_changes": changes,
        }

        # Track history if enabled
        if self.track_history:
            self._change_history.append(
                {
                    "step": self._snapshot_step,
                    "total_l2_change": total_l2_change,
                    "max_element_change": max_change,
                }
            )

        logger.debug(
            f"Parameter change: L2={total_l2_change:.8f}, "
            f"max={max_change:.8f}, changed_ratio={result['changed_parameter_ratio']:.4f}"
        )

        return result

    def verify_learning_occurred(
        self, policy: nn.Module, threshold: float = 1e-6
    ) -> bool:
        """Verify that learning has occurred based on parameter changes.

        Args:
            policy: The policy network to check.
            threshold: Minimum L2 change to consider as learning.

        Returns:
            True if parameters have changed more than threshold.
        """
        if self._parameter_snapshot is None:
            logger.warning("No snapshot available for learning verification")
            return False

        changes = self.compute_parameter_change(policy)
        learning_occurred = changes["total_l2_change"] > threshold

        if learning_occurred:
            logger.info(
                f"Learning verified: parameter change {changes['total_l2_change']:.8f} "
                f"> threshold {threshold}"
            )
        else:
            logger.warning(
                f"Learning NOT verified: parameter change {changes['total_l2_change']:.8f} "
                f"<= threshold {threshold}"
            )

        return learning_occurred

    def get_change_history(self) -> List[Dict[str, float]]:
        """Get the history of parameter changes.

        Returns:
            List of dictionaries with step and change information.
        """
        return list(self._change_history)

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of learning progress.

        Returns:
            Dictionary with learning statistics.
        """
        if len(self._change_history) == 0:
            return {
                "num_snapshots": 0,
                "learning_detected": False,
                "average_change": 0.0,
                "max_change": 0.0,
            }

        changes = [h["total_l2_change"] for h in self._change_history]

        return {
            "num_snapshots": len(self._change_history),
            "learning_detected": any(c > 1e-6 for c in changes),
            "average_change": float(np.mean(changes)),
            "max_change": float(np.max(changes)),
            "min_change": float(np.min(changes)),
            "std_change": float(np.std(changes)),
        }

    def reset(self) -> None:
        """Reset the verifier state."""
        self._parameter_snapshot = None
        self._change_history.clear()
        self._snapshot_step = 0
        logger.info("LearningVerifier reset")


# Convenience function for integration
def create_conservative_updater(
    policy: nn.Module, config: Optional[Dict[str, Any]] = None
) -> Tuple[ConservativePolicyUpdater, LearningVerifier]:
    """Create a configured policy updater with learning verification.

    This is a convenience function for setting up the conservative
    policy learning infrastructure.

    Args:
        policy: The policy network to update.
        config: Optional configuration dictionary with keys:
            - learning_rate: float (default 1e-4)
            - update_frequency: int (default 10)
            - min_buffer_size: int (default 32)
            - batch_size: int (default 32)
            - gamma: float (default 0.99)
            - gradient_clip: float (default 1.0)
            - alpha: float for reward shaping (default 0.1)
            - buffer_size: int (default 10000)

    Returns:
        Tuple of (ConservativePolicyUpdater, LearningVerifier).
    """
    config = config or {}

    # Create reward shaper
    reward_shaper = PreferenceShapedReward(
        alpha=config.get("alpha", 0.1), cluster_bonus=config.get("cluster_bonus", 0.05)
    )

    # Create updater
    updater = ConservativePolicyUpdater(
        policy=policy,
        learning_rate=config.get("learning_rate", 1e-4),
        update_frequency=config.get("update_frequency", 10),
        min_buffer_size=config.get("min_buffer_size", 32),
        batch_size=config.get("batch_size", 32),
        gamma=config.get("gamma", 0.99),
        gradient_clip=config.get("gradient_clip", 1.0),
        reward_shaper=reward_shaper,
        buffer_size=config.get("buffer_size", 10000),
    )

    # Create verifier
    verifier = LearningVerifier(track_history=True)

    logger.info("Created conservative updater with learning verifier")

    return updater, verifier
