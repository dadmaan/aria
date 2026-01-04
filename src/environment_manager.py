"""
Environment Manager Module

This module implements the EnvironmentManager class for advanced reward shaping
and human feedback integration in reinforcement learning environments. It combines
multiple reward components including similarity-based rewards, structural coherence
rewards, and human feedback to create composite reward signals for agent training.

Key Features:
- Composite reward calculation following configurable weight formulas
- Integration with perceiving agents for structural analysis
- Human feedback collection with timeout and non-interactive modes
- Similarity-based reward computation using various metrics
- Episode context management for reference-based learning

Classes:
    EnvironmentManager: Main class for environment management and reward shaping
"""

import queue
import sys
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from src.utils.agents.agent_utils import print_sequence_matrix, render_midi
from src.utils.logging.logging_manager import get_logger
from src.utils.rewards.reward_manager import RewardManager
from src.utils.similarity.similarity_metrics import (
    SimilarityCalculator,
    normalize_reward,
)
from src.utils.human.human_feedback import HumanFeedbackCollector


class EnvironmentManager:
    def __init__(
        self,
        perceiving_agent,
        dqn_agent,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Environment manager for reward shaping and human feedback integration.

        Args:
            perceiving_agent: Agent for structural analysis and coherence rewards
            dqn_agent: DQN agent instance
            config: Configuration dictionary with reward and feedback settings
        """
        self.logger = get_logger("environment_manager")
        self.perceiving_agent = perceiving_agent
        self.dqn_agent = dqn_agent
        self.config = config or {}

        # Initialize reward management with CONTEXT.md formula compliance
        reward_weights = self._extract_reward_weights_from_config()
        self.reward_manager = RewardManager(reward_weights)

        # Initialize similarity calculator
        self.similarity_calculator = SimilarityCalculator(
            mode=self.config.get("similarity_mode", "cosine")
        )

        # Initialize human feedback collector
        self.human_feedback_collector = HumanFeedbackCollector(
            timeout=self.config.get("interaction_timeout", 30),
            non_interactive_mode=self.config.get("non_interactive_mode", False),
            cli_feedback=self.config.get("cli_human_feedback", None),
        )

        # Episode context management
        self.current_episode_reference = None
        self.reference_features = None
        self.target_sequences = None
        self.last_shaped_reward_np = None

        # Legacy compatibility properties
        self.enable_human_feedback = self.config.get("human_feedback_enabled", False)
        self.human_feedback_timeout = self.config.get("interaction_timeout", 30)

    def _extract_reward_weights_from_config(self) -> Dict[str, float]:
        """Extract reward weights from config following CONTEXT.md schema."""
        reward_weights_config = self.config.get("reward_weights", {})

        # Map CONTEXT.md schema (w1, w2, w3) to RewardManager schema
        return {
            "similarity": reward_weights_config.get("w1", 0.4),  # R_similarity
            "structure": reward_weights_config.get("w2", 0.3),  # R_structure
            "human": reward_weights_config.get("w3", 0.3),  # R_human
        }

    def process_episode_end(self, time_step, render_midi_flag, print_episode_output):
        """
        Process episode end with composite reward shaping per CONTEXT.md formula.

        Implements: R_total = w1 * R_similarity + w2 * R_structure + w3 * R_human

        Args:
            time_step: Final timestep after episode ends
            render_midi_flag: If True, renders MIDI from episode output
            print_episode_output: If True, prints episode output sequence

        Returns:
            Trajectory with modified reward based on composite reward formula
        """
        # Extract episode sequence
        episode_out_sequence = np.array(time_step.observation).flatten().tolist()

        # Get most recent trajectory from replay buffer
        trajectory, buffer_info = self.dqn_agent.replay_buffer.get_next(
            sample_batch_size=1, num_steps=2
        )

        if render_midi_flag:
            render_midi(episode_out_sequence, filename=None)

        if print_episode_output:
            print_sequence_matrix(episode_out_sequence)

        # Update target sequence periodically (legacy behavior)
        global_step_value = self._python_int_from_global_step(
            self.dqn_agent.global_step
        )
        if global_step_value % 500 == 0:
            self.update_target_sequence([85, 101, 100, 100])

        # Compute reward components in NumPy for backend independence
        r_similarity = self._compute_similarity_reward(episode_out_sequence)
        r_structure = self._compute_structure_reward(trajectory)
        r_human = self._compute_human_reward()

        # Apply CONTEXT.md formula: R_total = w1 * R_similarity + w2 * R_structure + w3 * R_human
        try:
            shaped_reward_np = self.reward_manager.compute_composite_reward(
                base_reward=r_similarity,
                structure_reward=r_structure,
                human_reward=r_human,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error(
                f"Reward shaping failed with error '{exc}'. Using similarity reward only."
            )
            shaped_reward_np = np.array([r_similarity], dtype=np.float32)

        self.last_shaped_reward_np = shaped_reward_np

        # Replace last reward in trajectory with shaped reward
        modified_reward = self._replace_last_reward_with_numpy(
            trajectory.reward, shaped_reward_np
        )
        trajectory_with_feedback = trajectory.replace(reward=modified_reward)

        return trajectory_with_feedback

    def _compute_similarity_reward(self, episode_sequence: list) -> float:
        """
        Compute R_similarity component using configurable similarity metrics.

        Args:
            episode_sequence: Generated sequence of cluster IDs

        Returns:
            Normalized similarity reward in [0, 1] range
        """
        try:
            if self.current_episode_reference is None:
                # No reference available, return neutral similarity
                return 0.5

            # Calculate similarity using configured mode
            similarity = self.similarity_calculator.calculate_similarity(
                generated_sequence=episode_sequence,
                reference_data=self.current_episode_reference,
                generated_features=None,  # TODO: Extract from perceiving agent if available
                reference_features=self.reference_features,
            )

            # Normalize to [0, 1] range
            normalized_similarity = normalize_reward(similarity, 0.0, 1.0)

            return float(normalized_similarity)

        except Exception as e:
            self.logger.warning(f"Similarity reward calculation failed: {e}. Using neutral reward.")
            return 0.5

    def _compute_structure_reward(self, trajectory) -> float:
        """
        Compute R_structure component using PerceivingAgent.calculate_structure_reward.

        Args:
            trajectory: Episode trajectory

        Returns:
            Normalized structural coherence reward in [0, 1] range
        """
        try:
            # Extract sequence from trajectory (same logic as legacy method)
            observation = trajectory.observation.numpy()
            obs_array = np.asarray(observation).squeeze()

            # Handle different observation shapes
            if obs_array.ndim == 1:
                seq = list(obs_array)
            elif obs_array.ndim == 2:
                seq = list(obs_array[-1])  # Last timestep
            else:
                seq = list(obs_array.flatten())

            # Use PerceivingAgent.calculate_structure_reward
            if hasattr(self.perceiving_agent, "calculate_structure_reward"):
                # Use default features_weight for structure calculation
                features_weight = {
                    "distance_weight": 1.0,
                    "similarity_weight": 1.0,
                    "neighbor_weight": 0.5,
                    "row_weight": 0.5,
                    "column_weight": 0.5,
                }

                total_reward, _ = self.perceiving_agent.calculate_structure_reward(
                    seq, features_weight
                )

                # Normalize structure reward to [0, 1] range
                normalized_reward = normalize_reward(total_reward, 0.0, 1.0)
                return float(normalized_reward)
            else:
                # Fallback to legacy evaluate_sequence method
                total_reward, _ = self.perceiving_agent.evaluate_sequence(seq)
                normalized_reward = normalize_reward(total_reward, 0.0, 1.0)
                return float(normalized_reward)

        except Exception as e:
            self.logger.warning(f"Structure reward calculation failed: {e}. Using zero reward.")
            return 0.0

    def _compute_human_reward(self) -> float:
        """
        Compute R_human component using configurable human feedback collector.

        Returns:
            Human feedback reward, normalized and defaulting to 0.0 in non-interactive mode
        """
        try:
            # Collect human feedback with timeout and non-interactive handling
            feedback = self.human_feedback_collector.collect_feedback()

            # Normalize human feedback to [0, 1] range
            normalized_feedback = normalize_reward(feedback, 0.0, 1.0)

            return float(normalized_feedback)

        except Exception as e:
            self.logger.warning(f"Human feedback collection failed: {e}. Using default reward.")
            return 0.0

    def set_episode_reference(
        self, reference_data: Any, reference_features: Optional[np.ndarray] = None
    ):
        """
        Set reference data for similarity calculation in current episode.

        Args:
            reference_data: Reference sequence or metadata dict
            reference_features: Optional pre-extracted features for reference
        """
        self.current_episode_reference = reference_data
        self.reference_features = reference_features

    @staticmethod
    def _python_int_from_global_step(global_step) -> int:
        if hasattr(global_step, "numpy"):
            return int(global_step.numpy())
        if isinstance(global_step, (np.integer, np.number)):
            return int(global_step)
        return int(global_step)

    def _extract_base_reward_np(self, trajectory):
        """Extract base reward from trajectory as NumPy array.

        Args:
            trajectory: Training trajectory (framework-agnostic)

        Returns:
            NumPy array of base rewards
        """
        reward_data = trajectory.reward

        # Convert to numpy if needed
        if hasattr(reward_data, 'numpy'):
            # TensorFlow tensor or similar
            reward_np = reward_data.numpy()
        elif isinstance(reward_data, np.ndarray):
            reward_np = reward_data
        else:
            # Try to convert to numpy array
            try:
                reward_np = np.array(reward_data)
            except Exception as exc:
                self.logger.error(
                    f"Failed to extract base reward: {exc}. Using zeros as fallback."
                )
                reward_np = np.zeros(1, dtype=np.float32)

        # Extract last reward
        base_reward_np = reward_np[..., -1:] if reward_np.ndim > 0 else reward_np
        return self.reward_manager.sanitize(base_reward_np)

    @staticmethod
    def _replace_last_reward_with_numpy(reward_tensor, shaped_reward_np):
        """Replace last reward in trajectory with shaped reward.

        Args:
            reward_tensor: Original reward data (tensor or numpy array)
            shaped_reward_np: Shaped reward as NumPy array

        Returns:
            Updated reward with same type as input
        """
        # Convert reward_tensor to numpy if needed
        if hasattr(reward_tensor, 'numpy'):
            reward_np = reward_tensor.numpy()
            was_tensor = True
            original_dtype = reward_tensor.dtype
        else:
            reward_np = np.array(reward_tensor)
            was_tensor = False
            original_dtype = reward_np.dtype

        # Reshape shaped_reward to match target shape
        target_shape = reward_np[..., -1:].shape
        shaped_reward_reshaped = np.reshape(shaped_reward_np, target_shape)

        # Concatenate: all but last + shaped reward
        updated_reward_np = np.concatenate(
            [reward_np[..., :-1], shaped_reward_reshaped], axis=-1
        )

        # Convert back to tensor if input was a tensor
        if was_tensor:
            # Try to import TensorFlow for backward compatibility
            try:
                import tensorflow as tf
                return tf.convert_to_tensor(updated_reward_np, dtype=original_dtype)
            except ImportError:
                # If TensorFlow not available, return numpy array
                # This is fine for SB3 which works with numpy
                return updated_reward_np

        return updated_reward_np

    def get_human_feedback_ipython(self, alpha=1.0, timeout=10):
        """
        Legacy method for human feedback collection.

        Now delegates to the new HumanFeedbackCollector for consistency.

        Args:
            alpha: Scaling factor for feedback
            timeout: Timeout in seconds (ignored, uses config timeout)

        Returns:
            Human feedback value scaled by alpha
        """
        if not self.enable_human_feedback:
            return 0.0

        # Delegate to new human feedback collector
        try:
            feedback = self.human_feedback_collector.collect_feedback(alpha=alpha)
            return feedback
        except Exception as e:
            self.logger.warning(f"Human feedback collection failed: {e}. Using default.")
            return 0.0

    def random_human_feedback(self, trajectory, alpha=1.0):
        """
        Provides binary human feedback for a given trajectory.
        This is a simplified example where human feedback is simulated. In a real-world scenario, you would
        need to implement a mechanism to obtain actual feedback from a human observer.

        Args:
            trajectory (tf_agents.trajectories.Trajectory): A trajectory representing a sequence of actions,
                observations, and rewards from the environment.
            alpha (float): Factor to control the influence of human feedback.

        Returns:
            tf.Tensor: A tensor containing the human feedback as a reward adjustment. The tensor has the same
                shape as the reward tensor in the input trajectory.

        """
        # Simulate human feedback for demonstration purposes
        # In a real-world scenario, you would collect this from a human
        # For example, you might have a user interface where a human can watch the agent's behavior
        # and provide feedback in real-time, or you might collect feedback offline by reviewing recorded behavior.

        # Here we just provide a random binary feedback for demonstration
        human_feedback = np.random.choice([-1, 1], size=trajectory.reward.shape)

        # Convert the numpy array to a TensorFlow tensor
        human_feedback_tensor = tf.convert_to_tensor(
            human_feedback * alpha, dtype=trajectory.reward.dtype
        )

        return human_feedback_tensor

    def get_perceiving_agent_feedback(self, trajectory) -> float:
        try:
            observation = trajectory.observation.numpy()
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error(
                f"Unable to retrieve observation from trajectory: {exc}. Using zero structure reward."
            )
            return 0.0

        try:
            seq = list(np.asarray(observation).squeeze()[-1])
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error(
                f"Failed to prepare sequence for perceiving agent: {exc}. Using zero structure reward."
            )
            return 0.0

        try:
            total_reward, reward_list = self.perceiving_agent.evaluate_sequence(seq)
        except (ValueError, ZeroDivisionError, IndexError) as exc:
            self.logger.warning(
                f"Perceiving agent evaluation failed: {exc}. Using zero structure reward."
            )
            return 0.0
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.error(
                f"Unexpected error during perceiving agent evaluation: {exc}. Using zero structure reward."
            )
            return 0.0

        sanitized_total = float(
            np.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
        )
        if not np.isfinite(sanitized_total):
            self.logger.warning(
                "Perceiving agent returned non-finite reward. Using zero structure reward."
            )
            return 0.0

        if reward_list is not None:
            reward_array = np.asarray(reward_list)
            if reward_array.size and not np.all(np.isfinite(reward_array)):
                self.logger.warning(
                    "Perceiving agent returned non-finite reward components. Using sanitized totals."
                )

        return sanitized_total

    def update_target_sequence(self, sequence):
        self.dqn_agent.train_py_env._rules.update_target_sequence(sequence)
        self.logger.info(f"Target sequence updated with --> {sequence}")
