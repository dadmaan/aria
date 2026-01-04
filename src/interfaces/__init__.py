"""Interfaces package for the multi-agent reinforcement learning framework.

This package contains abstract base classes and interfaces that define the contracts
for various components in the music generation system, including agents, pretrainers,
and feature extractors. These interfaces ensure consistency across different
implementations and backends.
"""

from .agents import GenerativeAgent, MusicEnvironment, PerceivingAgent
from .base_pretrainer import BasePretrainer
from .feature_extractor import FeatureExtractorLoader

__all__ = [
    "BasePretrainer",
    "FeatureExtractorLoader",
    "GenerativeAgent",
    "PerceivingAgent",
    "MusicEnvironment",
]
