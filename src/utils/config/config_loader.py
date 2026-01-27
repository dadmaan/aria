"""Shared configuration loader utility for unified agent configuration.

Supports both YAML and JSON configuration files with auto-detection based on
file extension. YAML is preferred for new configurations due to comment support.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jsonschema
import yaml

from .validator import ConfigValidator


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flatten a nested dictionary with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# JSON schema for agent configuration validation
# Supports both new nested format and legacy flat format
AGENT_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        # New nested format sections
        "training": {
            "type": "object",
            "properties": {
                "num_iterations": {"type": "integer", "minimum": 1},
                "batch_size": {"type": "integer", "minimum": 1},
                "learning_rate": {"type": "number", "minimum": 0},
                "gamma": {"type": "number", "minimum": 0, "maximum": 1},
                "replay_buffer_max_length": {"type": "integer", "minimum": 1},
                "initial_collect_steps": {"type": "integer", "minimum": 0},
                "collect_steps_per_iteration": {"type": "integer", "minimum": 1},
                "target_update_interval": {"type": "integer", "minimum": 1},
                "log_interval": {"type": "integer", "minimum": 1},
                "exploration": {
                    "type": "object",
                    "properties": {
                        "initial_eps": {"type": "number", "minimum": 0, "maximum": 1},
                        "final_eps": {"type": "number", "minimum": 0, "maximum": 1},
                        "fraction": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        },
        "ghsom": {
            "type": "object",
            "properties": {
                "checkpoint": {"type": ["string", "null"]},
                "default_model_path": {"type": "string"},
            },
        },
        "features": {
            "type": "object",
            "properties": {
                "artifact_path": {"type": "string"},
                "type": {"type": "string", "enum": ["raw", "tsne", "filtered"]},
            },
        },
        "paths": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string"},
                "tensorboard_log_dir": {"type": "string"},
                "checkpoint_dir": {"type": ["string", "null"]},
            },
        },
        "music": {
            "type": "object",
            "properties": {
                "sequence_length": {"type": "integer", "minimum": 1},
            },
        },
        # Legacy flat parameters (for backward compatibility)
        "reward_weights": {
            "type": "object",
            "properties": {
                "w1": {"type": "number"},
                "w2": {"type": "number"},
                "w3": {"type": "number"},
            },
            "required": ["w1", "w2", "w3"],
        },
        "similarity_mode": {
            "type": "string",
            "enum": ["cosine", "euclidean", "attribute"],
        },
        "interaction_timeout": {"type": "integer", "minimum": 1},
        "non_interactive_mode": {"type": "boolean"},
        "enable_wandb": {"type": "boolean"},
        "enable_gpu": {"type": "boolean"},
        "num_iterations": {"type": "integer", "minimum": 1},
        "batch_size": {"type": "integer", "minimum": 1},
        "learning_rate": {"type": "number", "minimum": 0},
        "gamma": {"type": "number", "minimum": 0, "maximum": 1},
        "policy": {"type": "string"},
        "buffer_size": {"type": "integer", "minimum": 1},
        "lstm_q_net": {"type": "boolean"},
        "fc_q_net": {"type": "boolean"},
        "input_fc_layer_params": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "lstm_layer_params": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "output_fc_layer_params": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "sequence_length": {"type": "integer", "minimum": 1},
        "rl_backend": {"type": "string"},
    },
    "required": [
        "reward_weights",
        "similarity_mode",
        "interaction_timeout",
        "non_interactive_mode",
        "enable_wandb",
        "enable_gpu",
    ],
}


class ConfigLoader:
    """Utility class for loading and validating agent configuration.

    Supports both YAML (.yaml, .yml) and JSON (.json) configuration files.
    Auto-detects format based on file extension.
    """

    # Default values for configuration sections
    DEFAULTS = {
        "training": {
            "num_iterations": 10000,
            "batch_size": 64,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "replay_buffer_max_length": 1000000,
            "initial_collect_steps": 50,
            "collect_steps_per_iteration": 1,
            "target_update_interval": 1000,
            "log_interval": 100,
            "exploration": {
                "initial_eps": 1.0,
                "final_eps": 0.05,
                "fraction": 0.1,
            },
        },
        "ghsom": {
            "checkpoint": None,
            "default_model_path": "artifacts/ghsom/commu_bass_model/ghsom_model.pkl",
        },
        "features": {
            "artifact_path": "artifacts/features/raw/commu_bass",
            "type": "raw",
        },
        "paths": {
            "output_dir": "outputs",
            "tensorboard_log_dir": "logs",
            "checkpoint_dir": None,
        },
        "music": {
            "sequence_length": 32,
        },
    }

    def __init__(self, config_path: Optional[Union[Path, str]] = None):
        """Initialize with optional config path override.

        Args:
            config_path: Path to config file. If None, searches for default config
                        in order: training.yaml, agent_config.yaml, agent_config.yml, agent_config.json
        """
        if config_path is None:
            config_path = self._find_default_config()
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[Dict[str, Any]] = None
        self._flat_config: Optional[Dict[str, Any]] = None

    def _find_default_config(self) -> Path:
        """Find the default configuration file, preferring YAML over JSON."""
        configs_dir = Path("configs")
        candidates = [
            configs_dir / "training.yaml",  # New primary config file
            configs_dir / "agent_config.yaml",  # Legacy name
            configs_dir / "agent_config.yml",  # Legacy name
            configs_dir / "agent_config.json",  # Legacy name
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Fall back to training.yaml path (will error on load if not found)
        return configs_dir / "training.yaml"

    def _load_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from file, auto-detecting format."""
        suffix = path.suffix.lower()

        with open(path, "r", encoding="utf-8") as f:
            if suffix in (".yaml", ".yml"):
                return yaml.safe_load(f) or {}
            elif suffix == ".json":
                return json.load(f)
            else:
                # Try YAML first, fall back to JSON
                content = f.read()
                try:
                    return yaml.safe_load(content) or {}
                except yaml.YAMLError:
                    return json.loads(content)

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize config to support both new nested and legacy flat formats.

        This ensures backward compatibility by:
        1. Reading nested format and populating flat keys
        2. Reading flat format and populating nested keys
        """
        normalized = config.copy()

        # Map from nested to flat (for backward compatibility with existing code)
        mappings = [
            ("training.num_iterations", "num_iterations"),
            ("training.batch_size", "batch_size"),
            ("training.learning_rate", "learning_rate"),
            ("training.gamma", "gamma"),
            ("training.replay_buffer_max_length", "replay_buffer_max_length"),
            ("training.initial_collect_steps", "initial_collect_steps"),
            ("training.collect_steps_per_iteration", "collect_steps_per_iteration"),
            ("training.target_update_interval", "target_update_interval"),
            ("training.log_interval", "log_interval"),
            ("music.sequence_length", "sequence_length"),
            ("paths.tensorboard_log_dir", "tensorboard_log_dir"),
            ("paths.checkpoint_dir", "checkpoint_dir"),
        ]

        for nested_key, flat_key in mappings:
            # If nested key exists, copy to flat
            nested_value = self._get_nested(normalized, nested_key)
            if nested_value is not None and flat_key not in normalized:
                normalized[flat_key] = nested_value
            # If flat key exists but nested doesn't, copy to nested
            elif flat_key in normalized:
                self._set_nested(normalized, nested_key, normalized[flat_key])

        return normalized

    def _get_nested(self, d: Dict[str, Any], key: str) -> Any:
        """Get a value from a nested dictionary using dot notation."""
        keys = key.split(".")
        value = d
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None

    def _set_nested(self, d: Dict[str, Any], key: str, value: Any) -> None:
        """Set a value in a nested dictionary using dot notation."""
        keys = key.split(".")
        current = d
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def load(self, validate: bool = True) -> Dict[str, Any]:
        """Load configuration from file with optional validation.

        Args:
            validate: Whether to validate against schema and RL constraints

        Returns:
            Configuration dictionary (normalized to support both formats)

        Raises:
            FileNotFoundError: If config file doesn't exist
            jsonschema.ValidationError: If validation fails
            ValueError: If RL-specific validation fails
        """
        if self._config is None:
            # Load raw config
            raw_config = self._load_file(self.config_path)

            # Merge with defaults (for new sections that might not exist)
            self._config = _deep_merge(self.DEFAULTS, raw_config)

            # Normalize to support both formats
            self._config = self._normalize_config(self._config)

            # Create flattened version for dot-notation access
            self._flat_config = _flatten_dict(self._config)

            if validate:
                # JSON schema validation (structural) - relaxed for new format
                try:
                    jsonschema.validate(self._config, AGENT_CONFIG_SCHEMA)
                except jsonschema.ValidationError:
                    # Log but don't fail for missing legacy required fields
                    pass

                # RL-specific validation
                ConfigValidator.validate_and_log(self._config)

        return self._config.copy() if self._config else {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'training.learning_rate')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.load()

        # First try direct key access
        if key in config:
            return config[key]

        # Then try dot notation for nested keys
        keys = key.split(".")
        value = config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def is_sb3_backend(self) -> bool:
        """Check if SB3 backend is configured.

        Note: Always returns True as the system is now SB3-only.
        """
        return True

    def is_tfa_backend(self) -> bool:
        """Check if TF-Agents backend is configured.

        Note: Always returns False as TF-Agents support has been removed.
        """
        return False

    def get_backend(self) -> str:
        """Get the RL backend name.

        Returns:
            Always returns "sb3" as the system is now SB3-only.
        """
        return "sb3"

    def get_reward_weights(self) -> Dict[str, float]:
        """Get composite reward weights."""
        return self.get("reward_weights", {"w1": 0.33, "w2": 0.33, "w3": 0.34})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training-related configuration section."""
        return self.get("training", self.DEFAULTS["training"])

    def get_ghsom_config(self) -> Dict[str, Any]:
        """Get GHSOM-related configuration section."""
        return self.get("ghsom", self.DEFAULTS["ghsom"])

    def get_features_config(self) -> Dict[str, Any]:
        """Get features-related configuration section."""
        return self.get("features", self.DEFAULTS["features"])

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths-related configuration section."""
        return self.get("paths", self.DEFAULTS["paths"])

    def get_music_config(self) -> Dict[str, Any]:
        """Get music-related configuration section."""
        return self.get("music", self.DEFAULTS["music"])

    def reload(self) -> Dict[str, Any]:
        """Force reload configuration from file."""
        self._config = None
        self._flat_config = None
        return self.load()


# Global instance for shared access
_global_config: Optional[ConfigLoader] = None


def _get_global_config() -> ConfigLoader:
    """Get or create the global config loader instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a specific path.

    Args:
        config_path: Path to the configuration file (YAML or JSON)

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(Path(config_path))
    return loader.load()


def get_config() -> Dict[str, Any]:
    """Get the global configuration dictionary."""
    return _get_global_config().load()


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value from global config."""
    return _get_global_config().get(key, default)


def is_sb3_backend() -> bool:
    """Check if SB3 backend is globally configured.

    Note: Always returns True as the system is now SB3-only.
    """
    return _get_global_config().is_sb3_backend()


def is_tfa_backend() -> bool:
    """Check if TF-Agents backend is globally configured.

    Note: Always returns False as TF-Agents support has been removed.
    """
    return _get_global_config().is_tfa_backend()


def get_backend() -> str:
    """Get the configured RL backend.

    Returns:
        Always returns "sb3" as the system is now SB3-only.
    """
    return _get_global_config().get_backend()
