"""Configuration validation utilities for RL training system.

This module provides validation for configuration parameters to catch errors
early and provide helpful feedback for parameter tuning.

Supports both legacy flat configuration format and new nested YAML format.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import logging

# Use standard logging to avoid circular import with logging_manager
logger = logging.getLogger("config_validator")


def _get_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a value from config supporting both flat and nested keys.

    Args:
        config: Configuration dictionary
        key: Key in dot notation (e.g., 'training.learning_rate')
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    # First try direct access (flat format)
    if key in config:
        return config[key]

    # Then try nested access
    keys = key.split(".")
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


class ConfigValidator:
    """Validate RL training configuration parameters.

    Supports both legacy flat format and new nested YAML format.
    """

    @staticmethod
    def validate(config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of warnings/errors.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of error/warning messages (empty if valid)
        """
        errors = []
        warnings = []

        # Validate reward weights
        reward_weights = _get_value(config, "reward_weights")
        if reward_weights:
            w = reward_weights

            # Check required keys
            for key in ["w1", "w2", "w3"]:
                if key not in w:
                    errors.append(f"Missing reward weight: {key}")

            # Check values are positive
            for key, val in w.items():
                if val < 0:
                    errors.append(f"Reward weight {key}={val} is negative")
                if val > 10:
                    warnings.append(f"Reward weight {key}={val} is unusually high")

            # Check sum (should be ~1.0 but not critical)
            if all(k in w for k in ["w1", "w2", "w3"]):
                total = w["w1"] + w["w2"] + w["w3"]
                if not np.isclose(total, 1.0, rtol=0.1):
                    warnings.append(
                        f"Reward weights sum to {total:.3f}, not 1.0. "
                        "This is allowed but unusual."
                    )

        # Validate gamma (discount factor) - support both formats
        gamma = _get_value(config, "gamma") or _get_value(config, "training.gamma")
        if gamma is not None:
            if not 0 <= gamma <= 1:
                errors.append(f"Gamma={gamma} must be in [0, 1]")
            if gamma < 0.9:
                warnings.append(
                    f"Gamma={gamma} is very low. Recommended: 0.95-0.99 "
                    "for long-horizon tasks like music generation."
                )
            if gamma > 0.99:
                warnings.append(
                    f"Gamma={gamma} is very high. May lead to unstable learning."
                )

        # Validate learning rate - support both formats
        lr = _get_value(config, "learning_rate") or _get_value(
            config, "training.learning_rate"
        )
        if lr is not None:
            if lr <= 0:
                errors.append(f"Learning rate={lr} must be positive")
            if lr > 0.01:
                warnings.append(f"Learning rate={lr} is high. May cause instability.")
            if lr < 1e-6:
                warnings.append(
                    f"Learning rate={lr} is very low. Learning may be too slow."
                )

        # Validate feature weights (v2: only distance, neighbor, grid weights required)
        feature_weights = _get_value(config, "feature_weights")
        if feature_weights:
            fw = feature_weights
            # v2 required weights (Euclidean-only, no cosine similarity)
            required = [
                "distance_weight",
                "neighbor_weight",
                "grid_weight",
            ]
            for key in required:
                if key not in fw:
                    warnings.append(f"Missing feature weight: {key}, will use default")
                elif fw[key] < 0:
                    errors.append(f"Feature weight {key}={fw[key]} cannot be negative")

        # Validate reward_components structure (v2)
        reward_components = _get_value(config, "reward_components")
        if reward_components:
            for comp_name in ["structure", "transition", "diversity"]:
                comp = reward_components.get(comp_name, {})
                weight = comp.get("weight")
                if weight is not None and (weight < 0 or weight > 1):
                    warnings.append(
                        f"reward_components.{comp_name}.weight={weight} should be in [0, 1]"
                    )

        # Validate context window
        window = _get_value(config, "reward_context_window")
        if window is not None:
            if window < 1:
                errors.append(f"Context window={window} must be >= 1")

            # Get sequence length from either format
            seq_len = (
                _get_value(config, "sequence_length")
                or _get_value(config, "music.sequence_length")
                or 32
            )
            if window > seq_len:
                warnings.append(
                    f"Context window={window} exceeds sequence length ({seq_len}). "
                    "Will be capped at sequence length."
                )

        # Validate diversity parameters
        drw = _get_value(config, "diversity_reward_weight")
        if drw is not None and drw < 0:
            warnings.append(
                f"Diversity reward weight={drw} is negative. "
                "This will penalize novel actions."
            )

        rp = _get_value(config, "repetition_penalty")
        if rp is not None and rp > 0:
            warnings.append(
                f"Repetition penalty={rp} is positive. "
                "This will reward repetition instead of penalizing it."
            )

        # Validate incremental rewards flag
        uir = _get_value(config, "use_incremental_rewards")
        if uir is not None and not isinstance(uir, bool):
            errors.append(f"use_incremental_rewards must be boolean, got {type(uir)}")

        # Validate batch size - support both formats
        bs = _get_value(config, "batch_size") or _get_value(
            config, "training.batch_size"
        )
        if bs is not None:
            if bs < 1:
                errors.append(f"Batch size={bs} must be >= 1")
            if bs > 128:
                warnings.append(
                    f"Batch size={bs} is very large. May cause memory issues."
                )

        # Validate exploration parameters (new nested format)
        exploration = _get_value(config, "training.exploration")
        if exploration:
            initial_eps = exploration.get("initial_eps")
            final_eps = exploration.get("final_eps")
            fraction = exploration.get("fraction")

            if initial_eps is not None:
                if not 0 <= initial_eps <= 1:
                    errors.append(
                        f"exploration.initial_eps={initial_eps} must be in [0, 1]"
                    )

            if final_eps is not None:
                if not 0 <= final_eps <= 1:
                    errors.append(
                        f"exploration.final_eps={final_eps} must be in [0, 1]"
                    )

            if initial_eps is not None and final_eps is not None:
                if final_eps > initial_eps:
                    warnings.append(
                        f"exploration.final_eps={final_eps} > initial_eps={initial_eps}. "
                        "Exploration will increase over training (unusual)."
                    )

            if fraction is not None:
                if not 0 <= fraction <= 1:
                    errors.append(f"exploration.fraction={fraction} must be in [0, 1]")
                if fraction > 0.5:
                    warnings.append(
                        f"exploration.fraction={fraction} is high. "
                        "Agent will explore for most of training."
                    )

        # Validate GHSOM configuration (new nested format)
        ghsom = _get_value(config, "ghsom")
        if ghsom:
            checkpoint = ghsom.get("checkpoint")
            if checkpoint is not None and checkpoint != "":
                checkpoint_path = Path(checkpoint)
                if not checkpoint_path.exists():
                    warnings.append(
                        f"GHSOM checkpoint path does not exist: {checkpoint}. "
                        "Will fail at runtime if not provided via CLI."
                    )

        # Validate features configuration (new nested format)
        features = _get_value(config, "features")
        if features:
            feature_type = features.get("type")
            if feature_type and feature_type not in ["raw", "tsne", "filtered"]:
                errors.append(
                    f"features.type={feature_type} must be one of: raw, tsne, filtered"
                )

            artifact_path = features.get("artifact_path")
            if artifact_path:
                artifact_dir = Path(artifact_path)
                if not artifact_dir.exists():
                    warnings.append(
                        f"Feature artifact path does not exist: {artifact_path}. "
                        "Will fail at runtime if not provided via CLI."
                    )

        # Validate sequence length
        seq_len = _get_value(config, "sequence_length") or _get_value(
            config, "music.sequence_length"
        )
        if seq_len is not None:
            if seq_len < 1:
                errors.append(f"Sequence length={seq_len} must be >= 1")
            if seq_len > 256:
                warnings.append(
                    f"Sequence length={seq_len} is very long. "
                    "May cause memory/performance issues."
                )

        # Validate target update interval
        tui = _get_value(config, "target_update_interval") or _get_value(
            config, "training.target_update_interval"
        )
        if tui is not None:
            if tui < 1:
                errors.append(f"Target update interval={tui} must be >= 1")
            if tui > 10000:
                warnings.append(
                    f"Target update interval={tui} is very high. "
                    "Target network may be too stale."
                )

        # Validate Tianshou-specific configuration (Phase 5)
        use_tianshou = _get_value(config, "use_tianshou")
        if use_tianshou:
            # Validate model.recurrent configuration
            recurrent = _get_value(config, "model.recurrent")
            if recurrent:
                embedding_dim = recurrent.get("embedding_dim")
                if embedding_dim is not None:
                    if embedding_dim < 1:
                        errors.append(
                            f"model.recurrent.embedding_dim={embedding_dim} must be >= 1"
                        )
                    if embedding_dim > 512:
                        warnings.append(
                            f"model.recurrent.embedding_dim={embedding_dim} is very large. "
                            "May cause memory issues."
                        )

                lstm_hidden_size = recurrent.get("lstm_hidden_size")
                if lstm_hidden_size is not None:
                    if lstm_hidden_size < 1:
                        errors.append(
                            f"model.recurrent.lstm_hidden_size={lstm_hidden_size} must be >= 1"
                        )
                    if lstm_hidden_size > 1024:
                        warnings.append(
                            f"model.recurrent.lstm_hidden_size={lstm_hidden_size} is very large. "
                            "May cause memory issues."
                        )

                lstm_num_layers = recurrent.get("lstm_num_layers")
                if lstm_num_layers is not None:
                    if lstm_num_layers < 1:
                        errors.append(
                            f"model.recurrent.lstm_num_layers={lstm_num_layers} must be >= 1"
                        )
                    if lstm_num_layers > 4:
                        warnings.append(
                            f"model.recurrent.lstm_num_layers={lstm_num_layers} is high. "
                            "Deep LSTMs may be difficult to train."
                        )

                dropout = recurrent.get("dropout")
                if dropout is not None:
                    if not 0 <= dropout < 1:
                        errors.append(
                            f"model.recurrent.dropout={dropout} must be in [0, 1)"
                        )
                    if dropout > 0.5:
                        warnings.append(
                            f"model.recurrent.dropout={dropout} is high. "
                            "May cause underfitting."
                        )

                fc_hidden_sizes = recurrent.get("fc_hidden_sizes")
                if fc_hidden_sizes is not None:
                    if not isinstance(fc_hidden_sizes, list):
                        errors.append(
                            f"model.recurrent.fc_hidden_sizes must be a list, got {type(fc_hidden_sizes)}"
                        )
                    elif len(fc_hidden_sizes) == 0:
                        warnings.append(
                            "model.recurrent.fc_hidden_sizes is empty. "
                            "No fully-connected layers after LSTM."
                        )
                    else:
                        for i, size in enumerate(fc_hidden_sizes):
                            if not isinstance(size, int) or size < 1:
                                errors.append(
                                    f"model.recurrent.fc_hidden_sizes[{i}]={size} must be a positive integer"
                                )
            else:
                warnings.append(
                    "use_tianshou=true but model.recurrent not configured. "
                    "Will use default DRQN architecture."
                )

        # Validate backend selection
        rl_backend = _get_value(config, "rl_backend")
        if rl_backend is not None:
            if rl_backend not in ["sb3", "tianshou"]:
                errors.append(
                    f"rl_backend={rl_backend} must be one of: sb3, tianshou"
                )

            # Check consistency between rl_backend and use_tianshou
            if rl_backend == "tianshou" and use_tianshou is False:
                warnings.append(
                    "rl_backend='tianshou' but use_tianshou=false. "
                    "These flags should be consistent."
                )
            elif rl_backend == "sb3" and use_tianshou is True:
                warnings.append(
                    "rl_backend='sb3' but use_tianshou=true. "
                    "These flags should be consistent."
                )

        # Combine and return
        all_messages = []
        for error in errors:
            all_messages.append(f"ERROR: {error}")
        for warning in warnings:
            all_messages.append(f"WARNING: {warning}")

        return all_messages

    @staticmethod
    def validate_or_raise(config: Dict[str, Any]):
        """Validate and raise exception if errors found.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If validation errors are found
        """
        messages = ConfigValidator.validate(config)

        errors = [m for m in messages if m.startswith("ERROR")]
        warnings = [m for m in messages if m.startswith("WARNING")]

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

        if warnings:
            for warning in warnings:
                logger.warning(warning)

    @staticmethod
    def validate_and_log(config: Dict[str, Any]) -> bool:
        """Validate configuration and log all messages.

        Args:
            config: Configuration dictionary to validate

        Returns:
            bool: True if no errors, False if errors found
        """
        messages = ConfigValidator.validate(config)

        has_errors = False
        for msg in messages:
            if msg.startswith("ERROR"):
                logger.error(msg)
                has_errors = True
            elif msg.startswith("WARNING"):
                logger.warning(msg)

        if not has_errors and not messages:
            logger.info("Configuration validation passed with no warnings")

        return not has_errors

        return not has_errors
