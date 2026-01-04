"""Configuration validation for HIL preference simulation.

This module provides validation utilities to ensure consistency between
checkpoints, configurations, GHSOM models, and cluster profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        valid: Whether configuration is valid.
        errors: List of error messages (blocking issues).
        warnings: List of warning messages (non-blocking issues).
    """

    valid: bool
    errors: List[str]
    warnings: List[str]

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed.

        Raises:
            ValueError: If validation failed with details from errors list.
        """
        if not self.valid:
            error_msg = "Configuration validation failed:\n"
            error_msg += "\n".join(f"  ERROR: {err}" for err in self.errors)

            if self.warnings:
                error_msg += "\n\nWarnings:\n"
                error_msg += "\n".join(f"  WARNING: {warn}" for warn in self.warnings)

            raise ValueError(error_msg)

    def log_results(self) -> None:
        """Log validation results at appropriate levels."""
        if self.valid:
            logger.info("Configuration validation passed")
            if self.warnings:
                for warning in self.warnings:
                    logger.warning(warning)
        else:
            logger.error("Configuration validation failed")
            for error in self.errors:
                logger.error(f"Validation error: {error}")
            for warning in self.warnings:
                logger.warning(f"Validation warning: {warning}")


class ConfigValidator:
    """Validates HIL simulation configuration consistency."""

    @staticmethod
    def validate_checkpoint_config_consistency(
        checkpoint_path: Path,
        config: Dict[str, Any],
        ghsom_dir: Path,
        cluster_profiles_path: Path,
    ) -> ValidationResult:
        """Validate that checkpoint, config, GHSOM, and profiles are consistent.

        Args:
            checkpoint_path: Path to model checkpoint (.pth file).
            config: Configuration dictionary.
            ghsom_dir: GHSOM directory path.
            cluster_profiles_path: Cluster profiles CSV path.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors = []
        warnings = []

        # Check file existence
        if not checkpoint_path.exists():
            errors.append(f"Checkpoint not found: {checkpoint_path}")

        if not ghsom_dir.exists():
            errors.append(f"GHSOM directory not found: {ghsom_dir}")
        elif not ghsom_dir.is_dir():
            errors.append(f"GHSOM path is not a directory: {ghsom_dir}")

        if not cluster_profiles_path.exists():
            errors.append(f"Cluster profiles not found: {cluster_profiles_path}")
        elif not cluster_profiles_path.is_file():
            errors.append(
                f"Cluster profiles path is not a file: {cluster_profiles_path}"
            )

        # Check GHSOM files
        if ghsom_dir.exists() and ghsom_dir.is_dir():
            # Support both old (codebook.pkl) and new (ghsom_model.pkl) naming conventions
            required_ghsom_files = [("ghsom_model.pkl", "codebook.pkl")]
            for file_options in required_ghsom_files:
                # Accept any of the file options (primary or fallback names)
                if isinstance(file_options, tuple):
                    found = any((ghsom_dir / fname).exists() for fname in file_options)
                    if not found:
                        errors.append(
                            f"Required GHSOM model file missing. Expected one of: {file_options}"
                        )
                else:
                    if not (ghsom_dir / file_options).exists():
                        errors.append(f"Required GHSOM file missing: {file_options}")

            # Support both config.json and config.yaml
            recommended_ghsom_files = [
                ("config.json", "config.yaml"),  # Either config format
                "unique_cluster_ids.json",  # Cluster ID mapping
            ]
            for file_options in recommended_ghsom_files:
                if isinstance(file_options, tuple):
                    found = any((ghsom_dir / fname).exists() for fname in file_options)
                    if not found:
                        warnings.append(
                            f"Recommended GHSOM file missing. Expected one of: {file_options}"
                        )
                else:
                    if not (ghsom_dir / file_options).exists():
                        warnings.append(f"Recommended GHSOM file missing: {file_options}")

        # Validate config structure - support both explicit sections and alternative structures
        # Required: training section (always needed)
        if "training" not in config:
            errors.append("Missing required config section: 'training'")

        # Environment config: accept either 'environment' section or top-level observation settings
        has_environment_config = (
            "environment" in config or
            "use_feature_observations" in config or
            "use_normalized_observations" in config
        )
        if not has_environment_config:
            errors.append(
                "Missing environment configuration: need either 'environment' section "
                "or top-level observation settings (use_feature_observations)"
            )

        # Agent config: accept either 'agent' section or 'network' section
        has_agent_config = "agent" in config or "network" in config
        if not has_agent_config:
            errors.append(
                "Missing agent configuration: need either 'agent' or 'network' section"
            )

        # Check observation settings - look in 'environment' section or top-level
        env_config = config.get("environment", {})
        has_feature_obs = (
            "use_feature_observations" in env_config or
            "use_feature_observations" in config
        )
        if not has_feature_obs:
            warnings.append(
                "use_feature_observations not set in config, assuming False. "
                "This should match checkpoint training configuration."
            )

        # Check sequence length in environment, music, or replay_buffer sections
        has_seq_length = (
            "sequence_length" in env_config or
            "sequence_length" in config.get("music", {}) or
            "sequence_length" in config.get("replay_buffer", {}).get("sequence", {})
        )
        if not has_seq_length:
            warnings.append("sequence_length not found in config")

        # Check agent/network configuration
        agent_config = config.get("agent", config.get("network", {}))
        if agent_config:
            # Check for network type - look in 'type' or 'network_type'
            has_network_type = "type" in agent_config or "network_type" in agent_config
            if not has_network_type:
                warnings.append("network_type not set in agent/network config")

            # Check for hidden size - look in various locations
            has_hidden_size = (
                "hidden_size" in agent_config or
                "hidden_size" in agent_config.get("lstm", {}) or
                "embedding_dim" in agent_config  # Alternative naming
            )
            if not has_hidden_size:
                warnings.append("hidden_size not set in agent/network config")

        valid = len(errors) == 0

        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    @staticmethod
    def validate_scenario_cluster_alignment(
        scenario_cluster_ids: List[int],
        valid_cluster_ids: set,
        scenario_name: str,
    ) -> ValidationResult:
        """Validate that scenario clusters exist in valid set.

        Args:
            scenario_cluster_ids: Cluster IDs used in scenario
                (desirable + undesirable).
            valid_cluster_ids: Set of valid cluster IDs from mapping.
            scenario_name: Name of scenario for error messages.

        Returns:
            ValidationResult indicating whether scenario is valid.
        """
        errors = []
        warnings = []

        if not scenario_cluster_ids:
            warnings.append(f"Scenario '{scenario_name}' has no cluster IDs specified")
        else:
            invalid_clusters = [
                cid for cid in scenario_cluster_ids if cid not in valid_cluster_ids
            ]

            if invalid_clusters:
                errors.append(
                    f"Scenario '{scenario_name}' contains invalid cluster IDs: "
                    f"{invalid_clusters}. Valid cluster IDs: {sorted(valid_cluster_ids)}"
                )

        valid = len(errors) == 0

        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    @staticmethod
    def validate_adaptation_parameters(
        adaptation_mode: str,
        adaptation_strength: float,
        feedback_threshold: float,
    ) -> ValidationResult:
        """Validate adaptation parameters.

        Args:
            adaptation_mode: Adaptation mode ('q_penalty', 'reward_shaping', etc.).
            adaptation_strength: Strength parameter.
            feedback_threshold: Feedback threshold for triggering adaptation.

        Returns:
            ValidationResult for adaptation parameters.
        """
        errors = []
        warnings = []

        valid_modes = ["q_penalty", "reward_shaping", "none"]
        if adaptation_mode not in valid_modes:
            errors.append(
                f"Invalid adaptation_mode: '{adaptation_mode}'. "
                f"Valid modes: {valid_modes}"
            )

        if adaptation_strength < 0:
            errors.append(
                f"adaptation_strength must be non-negative, got {adaptation_strength}"
            )
        elif adaptation_strength == 0 and adaptation_mode != "none":
            warnings.append(
                f"adaptation_strength is 0 with mode '{adaptation_mode}'. "
                "No adaptation will occur."
            )
        elif adaptation_strength > 50:
            warnings.append(
                f"adaptation_strength is very high ({adaptation_strength}). "
                "This may cause instability."
            )

        if not 1.0 <= feedback_threshold <= 5.0:
            errors.append(
                f"feedback_threshold must be in [1.0, 5.0], got {feedback_threshold}"
            )

        valid = len(errors) == 0

        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    @staticmethod
    def validate_simulation_parameters(
        num_iterations: int,
        num_seeds: int,
    ) -> ValidationResult:
        """Validate simulation execution parameters.

        Args:
            num_iterations: Number of iterations per simulation.
            num_seeds: Number of random seeds to use.

        Returns:
            ValidationResult for simulation parameters.
        """
        errors = []
        warnings = []

        if num_iterations <= 0:
            errors.append(f"num_iterations must be positive, got {num_iterations}")
        elif num_iterations < 10:
            warnings.append(
                f"num_iterations is very low ({num_iterations}). "
                "Results may not be meaningful."
            )

        if num_seeds <= 0:
            errors.append(f"num_seeds must be positive, got {num_seeds}")
        elif num_seeds == 1:
            warnings.append(
                "num_seeds is 1. Cannot compute confidence intervals or "
                "statistical significance."
            )
        elif num_seeds == 2:
            warnings.append("num_seeds is 2. Statistical tests may not be reliable.")

        valid = len(errors) == 0

        return ValidationResult(valid=valid, errors=errors, warnings=warnings)
