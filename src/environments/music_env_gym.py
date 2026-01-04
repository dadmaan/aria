"""Gymnasium-compliant environment wrapper for music generation.

This module provides a reinforcement learning environment for music generation tasks,
implementing the Gymnasium interface for compatibility with popular RL libraries like
Stable Baselines3. The environment facilitates sequence generation based on cluster
patterns from a perceiving agent, with configurable reward structures.

Key features:
- Discrete action space based on available cluster IDs
- Sequence-based observation space with fixed maximum length
- Composite reward calculation combining similarity, structure, and coherence metrics
- Support for rendering in human-readable and array formats
- Configurable reward weights and sequence parameters

The environment is designed to work with perceiving agents that analyze musical
structures and provide evaluation metrics for generated sequences.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..interfaces.agents import MusicEnvironment, PerceivingAgent
from ..utils.config.config_loader import get_config
from ..utils.logging.logging_manager import get_logger
from .constants import PADDING_CLUSTER_ID


class MusicGenerationGymEnv(gym.Env, MusicEnvironment):
    """Gymnasium-compliant environment for music generation.

    This environment wraps the core music generation logic to provide
    a standard Gym interface that works with SB3 and other RL libraries.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        perceiving_agent: PerceivingAgent,
        sequence_length: int = 16,
        config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the music generation environment.

        Args:
            perceiving_agent: Perceiving agent for structural analysis
            sequence_length: Maximum length of generated sequences
            config: Configuration dictionary (uses global config if None)
            render_mode: Rendering mode for visualization
        """
        super().__init__()

        self.logger = get_logger("music_env_gym")
        self.perceiving_agent = perceiving_agent
        self.sequence_length = sequence_length
        self.config = config or get_config()
        self.render_mode = render_mode

        # Get available cluster IDs from perceiving agent
        self.cluster_ids = self.perceiving_agent.get_unique_cluster_ids_list()
        self.n_clusters = len(self.cluster_ids)

        if self.n_clusters == 0:
            raise ValueError("Perceiving agent returned no cluster IDs")

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_clusters)

        # Observation space: sequence of cluster IDs
        # low=PADDING_CLUSTER_ID (-1) to accommodate padding token
        # This distinguishes padding from valid cluster ID 0 (GHSOM root)
        self.observation_space = spaces.Box(
            low=PADDING_CLUSTER_ID,
            high=max(self.cluster_ids) if self.cluster_ids else 0,
            shape=(self.sequence_length,),
            dtype=np.int32,
        )

        # Environment state
        self.current_sequence: List[int] = []
        self.step_count = 0
        self.episode_reward = 0.0
        self.done = False

        # Reward components tracking for dashboard display
        self._last_reward_components: Dict[str, float] = {}

        # Track previous cumulative reward for step-wise reward calculation
        # This enables dense rewards by returning the delta at each step
        self._previous_cumulative_reward: float = 0.0

        # Random number generator
        self.np_random = np.random.default_rng()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        # Reset state
        self.current_sequence = []
        self.step_count = 0
        self.episode_reward = 0.0
        self.done = False
        self._previous_cumulative_reward = 0.0

        # Create initial observation (zeros)
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Action index (cluster ID index)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError(
                "Environment is done. Call reset() to start a new episode."
            )

        # Map action index to cluster ID
        if action < 0 or action >= self.n_clusters:
            raise ValueError(
                f"Invalid action {action}. Must be in [0, {self.n_clusters-1}]"
            )

        cluster_id = self.cluster_ids[action]
        self.current_sequence.append(cluster_id)
        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward

        # Check if episode is done
        terminated = self.step_count >= self.sequence_length
        truncated = False  # Could add early stopping conditions here
        self.done = terminated or truncated

        # Get new observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation array (padded sequence of cluster IDs)
            Padding positions use PADDING_CLUSTER_ID (-1) to distinguish
            from valid GHSOM cluster IDs (which may include 0).
        """
        # Pad sequence with PADDING_CLUSTER_ID (-1) instead of zeros
        # This prevents confusion with GHSOM root node (cluster ID 0)
        observation = np.full(self.sequence_length, PADDING_CLUSTER_ID, dtype=np.int32)

        if self.current_sequence:
            seq_len = min(len(self.current_sequence), self.sequence_length)
            observation[:seq_len] = self.current_sequence[:seq_len]

        return observation

    def _calculate_reward(self) -> float:
        """Calculate step-wise reward for current action.

        This method computes the marginal reward contribution of the current step
        by taking the difference between the cumulative reward for the current
        sequence and the previous cumulative reward. This enables proper credit
        assignment in RL by providing dense, step-wise feedback.

        Returns:
            Step-wise reward value (delta from previous cumulative)
        """
        if not self.current_sequence:
            self._last_reward_components = {}
            return 0.0

        # Get reward components from perceiving agent
        sequence_array = np.array(self.current_sequence, dtype=np.int32)
        reward_components = self.perceiving_agent.evaluate_sequence(sequence_array)

        # Store for info dict
        self._last_reward_components = reward_components

        # Get component weights from config (consolidated under reward_components)
        reward_cfg = self.config.get("reward_components", {})
        structure_weight = reward_cfg.get("structure", {}).get("weight", 0.40)
        transition_weight = reward_cfg.get("transition", {}).get("weight", 0.35)
        diversity_weight = reward_cfg.get("diversity", {}).get("weight", 0.25)

        # Apply weights to get cumulative composite reward for full sequence
        # All components are pre-normalized to [0, 1]
        current_cumulative_reward = (
            structure_weight * reward_components.get("structure", 0.0)
            + transition_weight * reward_components.get("transition", 0.0)
            + diversity_weight * reward_components.get("diversity", 0.0)
        )

        # Calculate step-wise reward as delta from previous cumulative
        # This gives the marginal contribution of the current action
        step_reward = current_cumulative_reward - self._previous_cumulative_reward
        self._previous_cumulative_reward = current_cumulative_reward

        return step_reward

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary.

        Returns:
            Info dictionary with environment state

        Note:
            All fields must have fixed shapes to be compatible with Tianshou's
            VectorReplayBuffer. Variable-length sequences cause shape mismatches
            during buffer storage.
        """
        # Get reward components (cached from last _calculate_reward call)
        reward_components = getattr(self, "_last_reward_components", {})

        # Pad sequence with PADDING_CLUSTER_ID (-1) to avoid shape mismatch
        # and prevent confusion with valid GHSOM cluster ID 0
        padded_sequence = np.full(
            self.sequence_length, PADDING_CLUSTER_ID, dtype=np.int32
        )
        if self.current_sequence:
            seq_len = min(len(self.current_sequence), self.sequence_length)
            padded_sequence[:seq_len] = self.current_sequence[:seq_len]

        # Pad cluster_ids to fixed length (use n_clusters as the fixed size)
        padded_cluster_ids = np.array(self.cluster_ids, dtype=np.int32)

        return {
            "sequence": padded_sequence,  # Fixed-length padded array
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "sequence_length": len(self.current_sequence),
            "cluster_ids": padded_cluster_ids,  # Fixed-length array
            "reward_components": reward_components,
            "n_clusters": self.n_clusters,
        }

    def render(self) -> Optional[np.ndarray]:
        """Render the environment.

        Returns:
            Optional rendered image array
        """
        if self.render_mode == "human":
            self.logger.info(
                "Step {}: Sequence = {}".format(self.step_count, self.current_sequence)
            )
            self.logger.info("Episode reward: {:.3f}".format(self.episode_reward))
        elif self.render_mode == "rgb_array":
            # Could return a visualization array here
            return None

        return None

    def close(self) -> None:
        """Close the environment."""
        # Nothing to clean up for this environment

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Seed the environment's random number generators.

        Args:
            seed: Random seed

        Returns:
            List of seeds used by RNG(s)
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.perceiving_agent.seed(seed)

        return [seed] if seed is not None else []

    def get_observation_space(self) -> Any:
        """Get the observation space definition.

        Returns:
            Gym observation space
        """
        return self.observation_space

    def get_action_space(self) -> Any:
        """Get the action space definition.

        Returns:
            Gym action space
        """
        return self.action_space


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """Normalize cluster IDs to [0, 1] range for stable neural network training.

    This wrapper addresses the critical issue of raw cluster IDs (0-255) being passed
    directly to neural networks, which causes gradient instability and prevents stable
    Q-function approximation. By normalizing observations to [0, 1], we ensure:

    - Stable gradient flow during backpropagation
    - Bounded input magnitudes suitable for neural networks
    - Consistent numerical scale across all observations
    - Reduced risk of gradient explosion/vanishing

    Example:
        >>> env = MusicGenerationGymEnv(perceiving_agent, sequence_length=32)
        >>> env = NormalizedObservationWrapper(env)  # Wrap with normalization
        >>> obs, info = env.reset()
        >>> assert obs.min() >= 0.0 and obs.max() <= 1.0  # Normalized range

    Note:
        This wrapper converts observations from int32 to float32 and updates the
        observation space accordingly. The normalization is reversible by multiplying
        by max_cluster_id if original IDs are needed.
    """

    def __init__(self, env: gym.Env):
        """Initialize the normalization wrapper.

        Args:
            env: The environment to wrap (must be MusicGenerationGymEnv or compatible)

        Raises:
            AttributeError: If env doesn't have cluster_ids attribute
            ValueError: If max_cluster_id is 0 (would cause division by zero)
        """
        super().__init__(env)

        # Get max cluster ID for normalization
        if not hasattr(env, "cluster_ids"):
            raise AttributeError(
                "Environment must have 'cluster_ids' attribute for normalization"
            )

        self.max_cluster_id = max(env.cluster_ids) if env.cluster_ids else 1

        if self.max_cluster_id == 0:
            raise ValueError(
                "max_cluster_id cannot be 0 (would cause division by zero)"
            )

        # Update observation space to float32 with [0, 1] range
        original_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=original_shape, dtype=np.float32
        )

        # Log the normalization setup
        logger = get_logger("normalized_observation_wrapper")
        logger.info(
            f"Initialized NormalizedObservationWrapper: "
            f"max_cluster_id={self.max_cluster_id}, "
            f"observation_shape={original_shape}, "
            f"dtype={self.observation_space.dtype}"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1] range.

        Args:
            obs: Raw observation array with cluster IDs (dtype: int32)

        Returns:
            Normalized observation array in [0, 1] range (dtype: float32)

        Example:
            >>> obs = np.array([10, 245, 67, 128], dtype=np.int32)
            >>> normalized = self.observation(obs)
            >>> # With max_cluster_id=255:
            >>> # normalized = [0.039, 0.961, 0.263, 0.502]
        """
        # Convert to float32 and normalize by max cluster ID
        normalized_obs = obs.astype(np.float32) / self.max_cluster_id

        # Ensure values are strictly within [0, 1] (handle potential floating point errors)
        normalized_obs = np.clip(normalized_obs, 0.0, 1.0)

        return normalized_obs


class FeatureVectorObservationWrapper(gym.ObservationWrapper):
    """
    Converts cluster ID observations to fixed feature vector observations.

    This wrapper:
    - Maps each cluster ID to its feature vector (GHSOM prototype or centroid)
    - Preserves musical semantics from GHSOM/t-SNE preprocessing
    - Provides richer input than normalized scalar IDs
    - Uses fixed features (no learnable parameters)
    - Ensures stable gradient flow (bounded float values)

    Example:
        env = MusicGenerationGymEnv(...)
        mapper = ClusterFeatureMapper(ghsom_manager)
        env = FeatureVectorObservationWrapper(env, mapper)

        obs, info = env.reset()
        # obs shape: (seq_len, feature_dim) instead of (seq_len,)
    """

    def __init__(
        self,
        env: gym.Env,
        cluster_feature_mapper,
    ):
        """
        Initialize wrapper.

        Args:
            env: Base environment (must output cluster IDs)
            cluster_feature_mapper: Mapper from cluster IDs to features
        """
        super().__init__(env)
        self.mapper = cluster_feature_mapper

        # Update observation space from 1D cluster IDs to 2D feature vectors
        # Old: Box(low=0, high=max_cluster_id, shape=(seq_len,), dtype=int32)
        # New: Box(low=-inf, high=inf, shape=(seq_len, feature_dim), dtype=float32)

        original_seq_len = env.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf,  # Features can be negative (e.g., t-SNE coords)
            high=np.inf,
            shape=(original_seq_len, self.mapper.feature_dim),
            dtype=np.float32,
        )

        logger = get_logger("feature_vector_wrapper")
        logger.info(
            f"FeatureVectorObservationWrapper initialized: "
            f"seq_len={original_seq_len}, "
            f"feature_dim={self.mapper.feature_dim}, "
            f"obs_shape={self.observation_space.shape}, "
            f"total_obs_size={np.prod(self.observation_space.shape)}"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert cluster ID observation to feature vectors.

        Args:
            obs: Cluster IDs, shape (seq_len,), dtype int32
                 Example: [5, 12, -1, -1, ...]
                 Where -1 is PADDING_CLUSTER_ID

        Returns:
            Feature vectors, shape (seq_len, feature_dim), dtype float32
            Example: [[0.34, -1.21],     # cluster 5 features
                      [0.81,  0.43],     # cluster 12 features
                      [-100., -100.],    # padding position (distinct embedding)
                      [-100., -100.],    # padding position (distinct embedding)
                      ...]

        Note:
            Padding positions (PADDING_CLUSTER_ID = -1) receive a distinct
            embedding [-100, -100] that is far from the t-SNE coordinate
            space, allowing the LSTM to learn to ignore these positions.
        """
        return self.mapper.map_sequence(obs)


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens multi-dimensional observations for MLP input.

    SB3's MlpPolicy expects 1D observations. This wrapper flattens
    2D feature observations: (seq_len, feature_dim) -> (seq_len * feature_dim,)

    Example:
        env = FeatureVectorObservationWrapper(env, mapper)  # obs: (16, 2)
        env = FlattenObservationWrapper(env)                 # obs: (32,)
    """

    def __init__(self, env: gym.Env):
        """
        Initialize flattening wrapper.

        Args:
            env: Environment with multi-dimensional observations
        """
        super().__init__(env)

        # Update observation space shape
        original_shape = env.observation_space.shape
        flattened_dim = int(np.prod(original_shape))

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flattened_dim,),
            dtype=env.observation_space.dtype,
        )

        logger = get_logger("flatten_wrapper")
        logger.info(
            f"FlattenObservationWrapper initialized: "
            f"original_shape={original_shape}, "
            f"flattened_dim={flattened_dim}"
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Flatten observation.

        Args:
            obs: Multi-dimensional observation
                 Example: [[0.34, -1.21], [0.81, 0.43], ...]  shape (16, 2)

        Returns:
            Flattened observation
            Example: [0.34, -1.21, 0.81, 0.43, ...]  shape (32,)
        """
        return obs.flatten()
