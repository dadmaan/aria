"""Reward shaping utilities."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class RewardManager:
    """Encapsulates reward weighting, sanitization, and composition logic."""

    _DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
        "similarity": 1.0,
        "structure": 1.0,
        "human": 1.0,
    }

    def __init__(self, user_weights: Optional[Dict[str, float]] = None):
        self._reward_weights = self._build_reward_weights(user_weights)

    @property
    def reward_weights(self) -> Dict[str, float]:
        return dict(self._reward_weights)

    def update_weights(self, user_weights: Optional[Dict[str, float]]) -> None:
        self._reward_weights = self._build_reward_weights(user_weights)

    def compute_composite_reward(
        self,
        base_reward,
        structure_reward,
        human_reward,
    ) -> np.ndarray:
        target_shape = self._target_shape(base_reward)

        base = self._sanitize_array(self._ensure_numpy_array(base_reward, target_shape))
        structure = self._sanitize_array(
            self._ensure_numpy_array(structure_reward, target_shape)
        )
        human = self._sanitize_array(
            self._ensure_numpy_array(human_reward, target_shape)
        )

        weights = self._reward_weights
        composite = (
            weights["similarity"] * base
            + weights["structure"] * structure
            + weights["human"] * human
        )

        return self._sanitize_array(composite)

    @staticmethod
    def sanitize(array, fallback: float = 0.0) -> np.ndarray:
        return RewardManager._sanitize_array(array, fallback)

    @staticmethod
    def _build_reward_weights(
        user_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        weights = dict(RewardManager._DEFAULT_REWARD_WEIGHTS)
        if user_weights:
            weights.update(
                {k: float(v) for k, v in user_weights.items() if k in weights}
            )
        total = sum(weights.values())
        if total == 0:
            return dict(RewardManager._DEFAULT_REWARD_WEIGHTS)
        return weights

    @staticmethod
    def _target_shape(array) -> tuple:
        shape = np.shape(array)
        if len(shape) == 0:
            return (1,)
        return shape

    @staticmethod
    def _ensure_numpy_array(value, target_shape, dtype=np.float32):
        array = np.asarray(value, dtype=dtype)
        return np.broadcast_to(array, target_shape)

    @staticmethod
    def _sanitize_array(array, fallback: float = 0.0) -> np.ndarray:
        sanitized = np.asarray(array, dtype=np.float32)
        if sanitized.size == 0:
            return sanitized
        return np.nan_to_num(sanitized, nan=fallback, posinf=fallback, neginf=fallback)
