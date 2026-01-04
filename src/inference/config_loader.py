"""Configuration loader for inference and HIL simulation.

This module provides utilities for loading and accessing inference configuration
from YAML files, with support for default values and validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class InferenceConfig:
    """Configuration manager for inference and HIL simulation.

    This class loads configuration from YAML files and provides convenient
    access to all inference-related parameters with proper defaults.

    Attributes:
        config: The loaded configuration dictionary.
        config_path: Path to the loaded configuration file.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize inference configuration.

        Args:
            config_path: Path to inference config YAML. If None, uses default.
        """
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent.parent
                / "configs"
                / "inference_config.yaml"
            )

        self.config_path = Path(config_path)
        self.config = self._load_config()

        logger.info(f"Loaded inference config from: {self.config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports nested keys using dot notation (e.g., "feedback.base_rating").

        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    # ========================================================================
    # PATH ACCESSORS
    # ========================================================================

    def get_cluster_profiles_path(self) -> Path:
        """Get path to cluster profiles CSV."""
        return Path(self.get("paths.cluster_profiles"))

    def get_ghsom_dir(self) -> Path:
        """Get GHSOM directory path."""
        return Path(self.get("paths.ghsom_dir"))

    def get_features_dir(self) -> Path:
        """Get features directory path."""
        return Path(self.get("paths.features_dir"))

    def get_output_dir(self, output_type: str = "inference") -> Path:
        """Get output directory path.

        Args:
            output_type: Type of output ("inference", "simulation", "analysis").

        Returns:
            Output directory path.
        """
        key = f"paths.{output_type}_output"
        return Path(self.get(key, f"outputs/{output_type}"))

    # ========================================================================
    # FEEDBACK ACCESSORS
    # ========================================================================

    def get_base_rating(self) -> float:
        """Get base rating for neutral sequences."""
        return self.get("feedback.base_rating", 3.0)

    def get_noise_std(self) -> float:
        """Get noise standard deviation for feedback."""
        return self.get("feedback.noise_std", 0.3)

    def get_strictness(self) -> float:
        """Get feedback strictness parameter."""
        return self.get("feedback.strictness", 1.0)

    def get_feedback_threshold(self, level: str = "default") -> float:
        """Get feedback threshold by level.

        Args:
            level: Threshold level ("low", "medium", "high", "default").

        Returns:
            Threshold value.
        """
        return self.get(f"feedback.thresholds.{level}", 3.0)

    def get_feedback_weights(self) -> Dict[str, float]:
        """Get feedback dimension weights."""
        return self.get(
            "feedback.weights",
            {
                "quality": 0.3,
                "coherence": 0.2,
                "creativity": 0.2,
                "musicality": 0.3,
            },
        )

    def get_desirable_multiplier(self) -> float:
        """Get desirable cluster rating multiplier."""
        return self.get("feedback.desirable_multiplier", 2.0)

    def get_undesirable_multiplier(self) -> float:
        """Get undesirable cluster rating multiplier."""
        return self.get("feedback.undesirable_multiplier", 3.0)

    def get_dimension_noise(self, dimension: str) -> float:
        """Get noise std for specific feedback dimension."""
        return self.get(f"feedback.dimension_noise.{dimension}", 0.3)

    # ========================================================================
    # SCENARIO ACCESSORS
    # ========================================================================

    def get_scenario_target(self, scenario_name: str) -> float:
        """Get target value for a predefined scenario.

        Args:
            scenario_name: Name of the scenario.

        Returns:
            Target value (0.0-1.0).
        """
        return self.get(
            f"scenarios.target_values.{scenario_name}",
            self.get("scenarios.default_target_value", 0.5),
        )

    def get_default_target_value(self) -> float:
        """Get default target value for custom scenarios."""
        return self.get("scenarios.default_target_value", 0.5)

    # ========================================================================
    # ADAPTATION ACCESSORS
    # ========================================================================

    def get_adaptation_mode(self) -> str:
        """Get default adaptation mode."""
        return self.get("adaptation.default_mode", "q_penalty")

    def get_adaptation_strength(self, mode: Optional[str] = None) -> float:
        """Get adaptation strength for specified mode.

        Args:
            mode: Adaptation mode ("q_penalty" or "reward_shaping").
                If None, uses default mode.

        Returns:
            Adaptation strength value.
        """
        if mode is None:
            mode = self.get_adaptation_mode()

        return self.get(f"adaptation.{mode}.strength", 5.0)

    def get_epsilon_greedy(self) -> float:
        """Get epsilon-greedy exploration rate."""
        return self.get("adaptation.epsilon_greedy", 0.1)

    def get_ablation_strengths(self) -> List[float]:
        """Get list of strengths for ablation studies."""
        return self.get("adaptation.ablation_strengths", [1.0, 2.5, 5.0, 7.5, 10.0])

    def get_q_penalty_params(self) -> Dict[str, float]:
        """Get all Q-penalty mode parameters."""
        return {
            "strength": self.get("adaptation.q_penalty.strength", 5.0),
            "decay_rate": self.get("adaptation.q_penalty.decay_rate", 0.1),
            "min_penalty": self.get("adaptation.q_penalty.min_penalty", -10.0),
            "max_penalty": self.get("adaptation.q_penalty.max_penalty", 0.0),
        }

    def get_reward_shaping_params(self) -> Dict[str, float]:
        """Get all reward shaping mode parameters."""
        return {
            "strength": self.get("adaptation.reward_shaping.strength", 5.0),
            "accumulation_rate": self.get(
                "adaptation.reward_shaping.accumulation_rate", 0.5
            ),
            "decay_factor": self.get("adaptation.reward_shaping.decay_factor", 0.95),
            "min_modifier": self.get("adaptation.reward_shaping.min_modifier", -5.0),
            "max_modifier": self.get("adaptation.reward_shaping.max_modifier", 5.0),
        }

    # ========================================================================
    # SIMULATION ACCESSORS
    # ========================================================================

    def get_num_iterations(self) -> int:
        """Get default number of simulation iterations."""
        return self.get("simulation.num_iterations", 50)

    def get_num_seeds(self) -> int:
        """Get default number of random seeds."""
        return self.get("simulation.num_seeds", 3)

    def get_log_interval(self) -> int:
        """Get log interval for progress updates."""
        return self.get("simulation.log_interval", 10)

    def get_seed_start(self) -> int:
        """Get starting seed value."""
        return self.get("simulation.seed_start", 42)

    def get_convergence_params(self) -> Dict[str, Any]:
        """Get convergence criteria parameters."""
        return {
            "enable": self.get("simulation.convergence.enable", True),
            "patience": self.get("simulation.convergence.patience", 10),
            "min_improvement": self.get("simulation.convergence.min_improvement", 0.05),
        }

    # ========================================================================
    # GENERATION ACCESSORS
    # ========================================================================

    def get_max_length(self) -> int:
        """Get maximum sequence length."""
        return self.get("generation.max_length", 100)

    def get_min_length(self) -> int:
        """Get minimum sequence length."""
        return self.get("generation.min_length", 10)

    def get_generation_mode(self) -> str:
        """Get default generation mode."""
        return self.get("generation.default_mode", "sample")

    def get_temperature(self) -> float:
        """Get sampling temperature."""
        return self.get("generation.temperature", 1.0)

    def get_top_k(self) -> int:
        """Get top-k sampling parameter."""
        return self.get("generation.top_k", 0)

    def get_top_p(self) -> float:
        """Get nucleus sampling threshold."""
        return self.get("generation.top_p", 0.9)

    # ========================================================================
    # VISUALIZATION ACCESSORS
    # ========================================================================

    def get_figure_format(self) -> str:
        """Get default figure format."""
        return self.get("visualization.default_format", "pdf")

    def get_figure_size(self, size_name: str = "default") -> List[float]:
        """Get figure size by name.

        Args:
            size_name: Size name ("small", "medium", "large", "wide", "square", "default").

        Returns:
            [width, height] in inches.
        """
        return self.get(f"visualization.figure_sizes.{size_name}", [8, 6])

    def get_font_size(self, element: str = "default") -> int:
        """Get font size for plot element."""
        return self.get(f"visualization.fonts.{element}", 12)

    def get_dpi(self, context: str = "save") -> int:
        """Get DPI setting.

        Args:
            context: Context ("display" or "save").

        Returns:
            DPI value.
        """
        return self.get(
            f"visualization.dpi.{context}", 300 if context == "save" else 150
        )

    def get_color(self, element: str) -> str:
        """Get color for visualization element.

        Args:
            element: Element name (e.g., "desirable", "undesirable", "feedback").

        Returns:
            Hex color code.
        """
        return self.get(f"visualization.colors.{element}", "#000000")

    def get_colors(self) -> Dict[str, str]:
        """Get all visualization colors."""
        return self.get("visualization.colors", {})

    def get_style_params(self) -> Dict[str, float]:
        """Get plot style parameters."""
        return self.get(
            "visualization.style",
            {
                "line_width": 2.0,
                "marker_size": 6,
                "alpha": 0.7,
                "bar_width": 0.8,
            },
        )

    def get_smoothing_window(self) -> int:
        """Get smoothing window size."""
        return self.get("visualization.smoothing.window_size", 5)

    def is_smoothing_enabled(self) -> bool:
        """Check if smoothing is enabled by default."""
        return self.get("visualization.smoothing.enable", True)

    # ========================================================================
    # ANALYSIS ACCESSORS
    # ========================================================================

    def get_top_n_clusters(self) -> int:
        """Get number of top clusters to display."""
        return self.get("analysis.sequence.top_n_clusters", 20)

    def get_transition_matrix_size(self) -> int:
        """Get maximum size for transition matrix display."""
        return self.get("analysis.sequence.transition_matrix_size", 15)

    def get_num_example_sequences(self) -> int:
        """Get number of example sequences to visualize."""
        return self.get("analysis.sequence.example_sequences", 5)

    def get_total_clusters(self) -> int:
        """Get total number of clusters in GHSOM model."""
        return self.get("analysis.sequence.total_clusters", 22)

    def get_confidence_level(self) -> float:
        """Get confidence level for statistical analysis."""
        return self.get("analysis.statistical.confidence_level", 0.95)

    def get_significance_level(self) -> float:
        """Get significance level for hypothesis tests."""
        return self.get("analysis.statistical.significance_level", 0.05)

    def get_min_cluster_count(self) -> int:
        """Get minimum cluster count for profile analysis."""
        return self.get("analysis.profile.min_cluster_count", 3)

    # ========================================================================
    # CLI ACCESSORS
    # ========================================================================

    def get_cli_default(self, command: str, param: str) -> Any:
        """Get CLI default value for command parameter.

        Args:
            command: Command name (e.g., "generate", "simulate").
            param: Parameter name.

        Returns:
            Default value.
        """
        return self.get(f"cli.{command}.{param}")

    # ========================================================================
    # LOGGING ACCESSORS
    # ========================================================================

    def get_log_level(self) -> str:
        """Get logging level."""
        return self.get("logging.level", "INFO")

    def get_log_dir(self) -> Path:
        """Get log directory path."""
        return Path(self.get("logging.log_dir", "logs/inference"))

    def is_verbose(self) -> bool:
        """Check if verbose output is enabled."""
        return self.get("logging.verbose", False)

    def show_progress(self) -> bool:
        """Check if progress bars should be shown."""
        return self.get("logging.show_progress", True)

    # ========================================================================
    # POLICY LEARNING ACCESSORS
    # ========================================================================

    def get_policy_learning_enabled(self) -> bool:
        """Get whether policy learning is enabled by default."""
        return self.get("policy_learning.enable", False)

    def get_policy_learning_rate(self) -> float:
        """Get policy learning rate."""
        return self.get("policy_learning.learning_rate", 1e-4)

    def get_policy_update_frequency(self) -> int:
        """Get policy update frequency."""
        return self.get("policy_learning.update_frequency", 10)

    def get_policy_min_buffer_size(self) -> int:
        """Get minimum buffer size before training."""
        return self.get("policy_learning.min_buffer_size", 32)

    def get_policy_batch_size(self) -> int:
        """Get policy learning batch size."""
        return self.get("policy_learning.batch_size", 16)

    def get_policy_gamma(self) -> float:
        """Get policy learning discount factor."""
        return self.get("policy_learning.gamma", 0.95)

    def get_policy_gradient_clip(self) -> float:
        """Get policy gradient clipping value."""
        return self.get("policy_learning.gradient_clip", 1.0)

    def get_policy_buffer_size(self) -> int:
        """Get policy experience buffer size."""
        return self.get("policy_learning.buffer_size", 1000)

    def get_policy_reward_alpha(self) -> float:
        """Get reward shaping alpha for policy learning."""
        return self.get("policy_learning.reward_alpha", 0.1)

    def get_policy_cluster_bonus(self) -> float:
        """Get cluster bonus/penalty for policy learning."""
        return self.get("policy_learning.cluster_bonus", 0.05)

    def get_policy_learning_params(self) -> Dict[str, Any]:
        """Get all policy learning parameters as dictionary."""
        return {
            "enable": self.get_policy_learning_enabled(),
            "learning_rate": self.get_policy_learning_rate(),
            "update_frequency": self.get_policy_update_frequency(),
            "min_buffer_size": self.get_policy_min_buffer_size(),
            "batch_size": self.get_policy_batch_size(),
            "gamma": self.get_policy_gamma(),
            "gradient_clip": self.get_policy_gradient_clip(),
            "buffer_size": self.get_policy_buffer_size(),
            "reward_alpha": self.get_policy_reward_alpha(),
            "cluster_bonus": self.get_policy_cluster_bonus(),
        }

    # ========================================================================
    # CHECKPOINT ACCESSORS
    # ========================================================================

    def get_checkpoint_save_enabled(self) -> bool:
        """Get whether checkpoint saving is enabled by default."""
        return self.get("checkpoint.save_after_simulation", False)

    def get_checkpoint_output_subdir(self) -> str:
        """Get checkpoint output subdirectory."""
        return self.get("checkpoint.output_subdir", "adapted_checkpoints")

    def get_checkpoint_filename_pattern(self) -> str:
        """Get checkpoint filename pattern."""
        return self.get(
            "checkpoint.filename_pattern", "adapted_{scenario}_{timestamp}.pth"
        )

    def get_checkpoint_params(self) -> Dict[str, Any]:
        """Get all checkpoint parameters as dictionary."""
        return {
            "save_after_simulation": self.get_checkpoint_save_enabled(),
            "output_subdir": self.get_checkpoint_output_subdir(),
            "filename_pattern": self.get_checkpoint_filename_pattern(),
        }

    # ========================================================================
    # EXPLORATION ACCESSORS
    # ========================================================================

    def get_exploration_config(self) -> Dict[str, Any]:
        """Get exploration configuration for HIL simulation.

        Returns:
            Dictionary with exploration configuration including:
            - mode: exploration mode ("epsilon_greedy", "boltzmann", "ucb")
            - enable_during_simulation: whether exploration is enabled
            - epsilon: dict with initial, final, decay_schedule, warmup_iterations
            - temperature: dict for Boltzmann mode
        """
        return {
            "mode": self.get("adaptation.exploration.mode", "epsilon_greedy"),
            "enable_during_simulation": self.get(
                "adaptation.exploration.enable_during_simulation", True
            ),
            "epsilon": {
                "initial": self.get("adaptation.exploration.epsilon.initial", 0.5),
                "final": self.get("adaptation.exploration.epsilon.final", 0.05),
                "decay_schedule": self.get(
                    "adaptation.exploration.epsilon.decay_schedule", "linear"
                ),
                "warmup_iterations": self.get(
                    "adaptation.exploration.epsilon.warmup_iterations", 50
                ),
            },
            "temperature": {
                "initial": self.get("adaptation.exploration.temperature.initial", 2.0),
                "final": self.get("adaptation.exploration.temperature.final", 0.1),
                "decay_schedule": self.get(
                    "adaptation.exploration.temperature.decay_schedule", "linear"
                ),
                "warmup_iterations": self.get(
                    "adaptation.exploration.temperature.warmup_iterations", 50
                ),
            },
        }

    def get_exploration_enabled(self) -> bool:
        """Check if exploration is enabled during simulation.

        Returns:
            True if exploration is enabled during HIL simulation.
        """
        return self.get("adaptation.exploration.enable_during_simulation", True)

    def get_exploration_mode(self) -> str:
        """Get exploration mode.

        Returns:
            Exploration mode string ("epsilon_greedy", "boltzmann", "ucb").
        """
        return self.get("adaptation.exploration.mode", "epsilon_greedy")

    # ========================================================================
    # ADAPTIVE THRESHOLD ACCESSORS
    # ========================================================================

    def get_adaptive_threshold_config(self) -> Dict[str, Any]:
        """Get adaptive threshold configuration.

        Returns:
            Dictionary with adaptive threshold settings including:
            - enable: whether adaptive threshold is enabled
            - initial_threshold: starting threshold value
            - min_threshold: minimum allowed threshold
            - max_threshold: maximum allowed threshold
            - adjustment_rate: rate of threshold adjustment
            - no_improvement_patience: iterations before lowering threshold
        """
        return {
            "enable": self.get("experimental.adaptive_threshold.enable", False),
            "initial_threshold": self.get("feedback.thresholds.default", 3.0),
            "min_threshold": self.get(
                "experimental.adaptive_threshold.min_threshold", 2.5
            ),
            "max_threshold": self.get(
                "experimental.adaptive_threshold.max_threshold", 3.5
            ),
            "adjustment_rate": self.get(
                "experimental.adaptive_threshold.adjustment_rate", 0.03
            ),
            "no_improvement_patience": self.get(
                "experimental.adaptive_threshold.no_improvement_patience", 100
            ),
        }

    def get_adaptive_threshold_enabled(self) -> bool:
        """Check if adaptive threshold is enabled.

        Returns:
            True if adaptive threshold adjustment is enabled.
        """
        return self.get("experimental.adaptive_threshold.enable", False)

    # ========================================================================
    # LEARNING VERIFICATION ACCESSORS
    # ========================================================================

    def get_learning_verification_config(self) -> Dict[str, Any]:
        """Get learning verification configuration.

        Returns:
            Dictionary with learning verification settings including:
            - enable: whether verification is enabled
            - diversity_threshold: minimum diversity for detection
            - improvement_threshold: minimum improvement for detection
            - trend_significance: p-value threshold for trends
        """
        return {
            "enable": self.get("experimental.learning_verification.enable", True),
            "diversity_threshold": self.get(
                "experimental.learning_verification.diversity_threshold", 0.02
            ),
            "improvement_threshold": self.get(
                "experimental.learning_verification.improvement_threshold", 0.05
            ),
            "trend_significance": self.get(
                "experimental.learning_verification.trend_significance", 0.05
            ),
        }

    def get_learning_verification_enabled(self) -> bool:
        """Check if learning verification is enabled.

        Returns:
            True if post-simulation learning verification is enabled.
        """
        return self.get("experimental.learning_verification.enable", True)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return dict(self.config)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates (supports nested keys).
        """

        def update_nested(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = update_nested(d[k], v)
                else:
                    d[k] = v
            return d

        self.config = update_nested(self.config, updates)
        logger.debug(f"Updated configuration with {len(updates)} changes")

    def validate(self) -> bool:
        """Validate configuration integrity.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If configuration has invalid values.
        """
        # Validate feedback thresholds
        for level in ["low", "medium", "high", "default"]:
            threshold = self.get_feedback_threshold(level)
            if not 1.0 <= threshold <= 5.0:
                raise ValueError(f"Invalid feedback threshold '{level}': {threshold}")

        # Validate adaptation strengths
        for mode in ["q_penalty", "reward_shaping"]:
            strength = self.get_adaptation_strength(mode)
            if strength < 0:
                raise ValueError(f"Invalid {mode} strength: {strength}")

        # Validate simulation parameters
        if self.get_num_iterations() < 1:
            raise ValueError("num_iterations must be >= 1")

        if self.get_num_seeds() < 1:
            raise ValueError("num_seeds must be >= 1")

        logger.info("Configuration validation passed")
        return True

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"InferenceConfig(path='{self.config_path}')"


# Global config instance
_global_config: Optional[InferenceConfig] = None


def get_inference_config(
    config_path: Optional[Union[str, Path]] = None,
) -> InferenceConfig:
    """Get global inference configuration instance.

    Args:
        config_path: Path to config file. If None, uses default.
            If global config already exists and no path given, returns existing instance.

    Returns:
        InferenceConfig instance.
    """
    global _global_config

    if config_path is not None:
        # Create new instance with specified path
        _global_config = InferenceConfig(config_path)
    elif _global_config is None:
        # Create default instance
        _global_config = InferenceConfig()

    return _global_config


def load_inference_config(config_path: Union[str, Path]) -> InferenceConfig:
    """Load inference configuration from file.

    This is an alias for get_inference_config that always loads from path.

    Args:
        config_path: Path to config YAML file.

    Returns:
        InferenceConfig instance.
    """
    return get_inference_config(config_path)
