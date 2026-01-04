"""Environment constants for the music generation system.

This module defines shared constants used across the environment and
observation processing pipeline.
"""

# =============================================================================
# SEQUENCE PADDING CONSTANTS
# =============================================================================

# Special cluster ID used for padding in fixed-length sequences
# Using -1 to distinguish from valid GHSOM cluster IDs (which start at 0)
# This prevents semantic confusion where padding (0) could be mistaken for
# the GHSOM root node (cluster ID 0)
PADDING_CLUSTER_ID: int = -1

# Feature embedding for padding positions
# Uses values far outside the typical t-SNE coordinate range (-3 to +3)
# This creates a distinct cluster in embedding space that the LSTM can
# learn to recognize as "no data" rather than conflating with real clusters
PADDING_EMBEDDING: tuple = (-100.0, -100.0)

# =============================================================================
# DOCUMENTATION
# =============================================================================
"""
Why these values?

PADDING_CLUSTER_ID = -1:
- GHSOM cluster IDs are non-negative integers (0, 1, 2, ...)
- Using -1 as padding creates an unambiguous distinction
- The LSTM embedding layer will learn a separate representation
- Prevents the "cluster 0 problem" where padding looks like root node

PADDING_EMBEDDING = (-100.0, -100.0):
- t-SNE coordinates typically fall in range [-3, +3]
- Using (-100, -100) places padding FAR from any real cluster
- Creates distinct separation in the feature space
- The LSTM learns to ignore these positions during sequence processing

Usage:
    from src.environments.constants import PADDING_CLUSTER_ID, PADDING_EMBEDDING

    # In MusicGenerationGymEnv._get_observation()
    observation = np.full(seq_len, PADDING_CLUSTER_ID, dtype=np.int32)

    # In ClusterFeatureMapper.get_features()
    if cluster_id == PADDING_CLUSTER_ID:
        return np.array(PADDING_EMBEDDING, dtype=np.float32)
"""
