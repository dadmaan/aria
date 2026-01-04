"""Preference simulation for Human-in-the-Loop music generation.

This module provides the PreferenceScenario and PreferenceFeedbackSimulator classes
for simulating user preferences and generating feedback based on cluster profile alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .cluster_profiles import ClusterProfileLoader
from .config_loader import InferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class PreferenceScenario:
    """Defines a user preference scenario for simulation.

    A preference scenario specifies which clusters are desirable and undesirable
    based on user preferences, allowing simulation of feedback that guides
    the agent toward preferred musical characteristics.

    Attributes:
        name: Short identifier for the scenario.
        description: Human-readable description of the preference.
        desirable_clusters: List of cluster IDs that align with user preference.
        undesirable_clusters: List of cluster IDs that conflict with user preference.
        target_metric: Name of the metric to optimize (e.g., "low_arousal_ratio").
        target_value: Target value for the metric (e.g., 0.6 for 60%).
        metadata: Additional metadata for the scenario.
    """

    name: str
    description: str
    desirable_clusters: List[int]
    undesirable_clusters: List[int]
    target_metric: str
    target_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scenario configuration."""
        # Ensure no overlap between desirable and undesirable clusters
        overlap = set(self.desirable_clusters) & set(self.undesirable_clusters)
        if overlap:
            raise ValueError(
                f"Overlap found between desirable and undesirable clusters: {overlap}"
            )

        # Ensure target_value is in valid range
        if not 0.0 <= self.target_value <= 1.0:
            raise ValueError(
                f"target_value must be between 0 and 1, got {self.target_value}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "desirable_clusters": self.desirable_clusters,
            "undesirable_clusters": self.undesirable_clusters,
            "target_metric": self.target_metric,
            "target_value": self.target_value,
            "metadata": self.metadata,
        }


def create_scenario_from_profiles(
    name: str,
    description: str,
    loader: ClusterProfileLoader,
    desirable_filter: Dict[str, str],
    undesirable_filter: Dict[str, str],
    target_metric: str,
    target_value: Optional[float] = None,
    config: Optional[InferenceConfig] = None,
) -> PreferenceScenario:
    """Create a preference scenario using cluster profile filters.

    Args:
        name: Scenario name.
        description: Scenario description.
        loader: ClusterProfileLoader instance.
        desirable_filter: Dict with attribute-value pairs for desirable clusters.
        undesirable_filter: Dict with attribute-value pairs for undesirable clusters.
        target_metric: Metric name to optimize.
        target_value: Target value for the metric. If None and config provided,
            uses config.get_default_target_value().
        config: Optional InferenceConfig for default values.

    Returns:
        PreferenceScenario with clusters populated from filters.

    Example:
        >>> scenario = create_scenario_from_profiles(
        ...     name="calm_relaxation",
        ...     description="Prefer calm music",
        ...     loader=loader,
        ...     desirable_filter={"arousal_level": "low"},
        ...     undesirable_filter={"arousal_level": "high"},
        ...     target_metric="low_arousal_ratio",
        ...     target_value=0.6,
        ... )
    """
    desirable_clusters: List[int] = []
    undesirable_clusters: List[int] = []

    # Get desirable clusters
    for attr, value in desirable_filter.items():
        clusters = loader.get_clusters_by_attribute(attr, value)
        desirable_clusters.extend(clusters)

    # Get undesirable clusters
    for attr, value in undesirable_filter.items():
        clusters = loader.get_clusters_by_attribute(attr, value)
        undesirable_clusters.extend(clusters)

    # Remove duplicates while preserving order
    desirable_clusters = list(dict.fromkeys(desirable_clusters))
    undesirable_clusters = list(dict.fromkeys(undesirable_clusters))

    # Remove any overlap (desirable takes precedence)
    undesirable_clusters = [
        c for c in undesirable_clusters if c not in desirable_clusters
    ]

    # Use config default if target_value not provided
    if target_value is None:
        if config is not None:
            target_value = config.get_default_target_value()
        else:
            target_value = 0.5

    return PreferenceScenario(
        name=name,
        description=description,
        desirable_clusters=desirable_clusters,
        undesirable_clusters=undesirable_clusters,
        target_metric=target_metric,
        target_value=target_value,
        metadata={
            "desirable_filter": desirable_filter,
            "undesirable_filter": undesirable_filter,
        },
    )


def get_predefined_scenarios(
    loader: ClusterProfileLoader,
    config: Optional[InferenceConfig] = None,
) -> Dict[str, PreferenceScenario]:
    """Get all predefined preference scenarios.

    Args:
        loader: ClusterProfileLoader instance for cluster lookup.
        config: Optional InferenceConfig for target values.

    Returns:
        Dictionary mapping scenario names to PreferenceScenario objects.
    """
    scenarios = {}

    # 1. Calm Relaxation - Prefer low arousal, avoid high intensity
    scenarios["calm_relaxation"] = create_scenario_from_profiles(
        name="calm_relaxation",
        description="Prefer calm, low-arousal music; avoid intense sequences",
        loader=loader,
        desirable_filter={"arousal_level": "low"},
        undesirable_filter={"arousal_level": "high"},
        target_metric="low_arousal_ratio",
        target_value=config.get_scenario_target("calm_relaxation") if config else 0.6,
        config=config,
    )

    # 2. Energetic Drive - Prefer high arousal, avoid calm
    scenarios["energetic_drive"] = create_scenario_from_profiles(
        name="energetic_drive",
        description="Prefer energetic, high-arousal music; avoid calm sequences",
        loader=loader,
        desirable_filter={"arousal_level": "high"},
        undesirable_filter={"arousal_level": "low"},
        target_metric="high_arousal_ratio",
        target_value=config.get_scenario_target("energetic_drive") if config else 0.5,
        config=config,
    )

    # 3. Piano Focus - Prefer piano, avoid strings
    scenarios["piano_focus"] = create_scenario_from_profiles(
        name="piano_focus",
        description="Prefer piano-dominated sequences; avoid string instruments",
        loader=loader,
        desirable_filter={"dominant_instrument": "acoustic_piano"},
        undesirable_filter={"dominant_instrument": "string_ensemble"},
        target_metric="piano_ratio",
        target_value=config.get_scenario_target("piano_focus") if config else 0.7,
        config=config,
    )

    # 4. Strings Ensemble - Prefer strings, avoid piano
    scenarios["strings_ensemble"] = create_scenario_from_profiles(
        name="strings_ensemble",
        description="Prefer string ensemble sequences; avoid piano",
        loader=loader,
        desirable_filter={"dominant_instrument": "string_ensemble"},
        undesirable_filter={"dominant_instrument": "acoustic_piano"},
        target_metric="strings_ratio",
        target_value=config.get_scenario_target("strings_ensemble") if config else 0.6,
        config=config,
    )

    # 5. Melodic Focus - Prefer main/sub melody roles
    scenarios["melodic_focus"] = create_scenario_from_profiles(
        name="melodic_focus",
        description="Prefer melodic content (main/sub melody); avoid background",
        loader=loader,
        desirable_filter={"dominant_role": "main_melody"},
        undesirable_filter={"dominant_role": "pad"},
        target_metric="melodic_ratio",
        target_value=config.get_scenario_target("melodic_focus") if config else 0.6,
        config=config,
    )

    # Add sub_melody to melodic focus desirable clusters
    sub_melody_clusters = loader.get_role_clusters("sub_melody")
    scenarios["melodic_focus"].desirable_clusters.extend(sub_melody_clusters)
    scenarios["melodic_focus"].desirable_clusters = list(
        dict.fromkeys(scenarios["melodic_focus"].desirable_clusters)
    )

    # 6. Ambient Background - Prefer pad/accompaniment, avoid melody
    scenarios["ambient_background"] = create_scenario_from_profiles(
        name="ambient_background",
        description="Prefer atmospheric background; avoid prominent melody",
        loader=loader,
        desirable_filter={"dominant_role": "pad"},
        undesirable_filter={"dominant_role": "main_melody"},
        target_metric="ambient_ratio",
        target_value=(
            config.get_scenario_target("ambient_background") if config else 0.5
        ),
        config=config,
    )

    # Add accompaniment to ambient desirable clusters
    acc_clusters = loader.get_role_clusters("accompaniment")
    scenarios["ambient_background"].desirable_clusters.extend(acc_clusters)
    scenarios["ambient_background"].desirable_clusters = list(
        dict.fromkeys(scenarios["ambient_background"].desirable_clusters)
    )

    # 7. Intrinsic Feel - Prefer newage/intrinsic, avoid cinematic/extrinsic
    scenarios["intrinsic_feel"] = create_scenario_from_profiles(
        name="intrinsic_feel",
        description="Prefer intrinsic, personal music; avoid cinematic/extrinsic",
        loader=loader,
        desirable_filter={"preference_type": "intrinsic"},
        undesirable_filter={"preference_type": "extrinsic"},
        target_metric="intrinsic_ratio",
        target_value=config.get_scenario_target("intrinsic_feel") if config else 0.4,
        config=config,
    )

    logger.info("Created %d predefined preference scenarios", len(scenarios))
    return scenarios


def adapt_scenario_to_available_clusters(
    scenario: PreferenceScenario,
    available_cluster_ids: List[int],
    warn_on_mismatch: bool = True,
) -> PreferenceScenario:
    """Adapt a scenario to use only clusters available in the model.

    This function filters scenario's desirable and undesirable clusters
    to include only those present in the agent's action space, preventing
    the cluster/action mismatch issue.

    Args:
        scenario: Original PreferenceScenario.
        available_cluster_ids: List of cluster IDs available in the model.
        warn_on_mismatch: If True, log warnings about filtered clusters.

    Returns:
        New PreferenceScenario with filtered cluster lists.

    Raises:
        ValueError: If no desirable clusters remain after filtering.
    """
    available_set = set(available_cluster_ids)

    # Filter desirable clusters
    original_desirable = set(scenario.desirable_clusters)
    valid_desirable = [c for c in scenario.desirable_clusters if c in available_set]
    removed_desirable = original_desirable - set(valid_desirable)

    # Filter undesirable clusters
    original_undesirable = set(scenario.undesirable_clusters)
    valid_undesirable = [c for c in scenario.undesirable_clusters if c in available_set]
    removed_undesirable = original_undesirable - set(valid_undesirable)

    # Enhanced structured logging with severity levels
    warn_threshold = 0.3  # Warn if >30% clusters removed
    des_ratio = (
        len(removed_desirable) / len(original_desirable) if original_desirable else 0
    )
    undes_ratio = (
        len(removed_undesirable) / len(original_undesirable)
        if original_undesirable
        else 0
    )
    if warn_on_mismatch:
        if des_ratio > warn_threshold or undes_ratio > warn_threshold:
            logger.warning(
                "Scenario '%s' adaptation: removed %.0f%% desirable, %.0f%% undesirable clusters",
                scenario.name,
                des_ratio * 100,
                undes_ratio * 100,
            )
        else:
            logger.info(
                "Scenario '%s' adapted: desirable %d->%d, undesirable %d->%d",
                scenario.name,
                len(original_desirable),
                len(valid_desirable),
                len(original_undesirable),
                len(valid_undesirable),
            )

    # Check if scenario is still viable
    if not valid_desirable:
        raise ValueError(
            f"Scenario '{scenario.name}' has no desirable clusters after filtering. "
            f"Original desirable: {sorted(original_desirable)}, "
            f"Available: {sorted(available_set)}"
        )

    logger.info(
        "Adapted scenario '%s': desirable %d->%d, undesirable %d->%d",
        scenario.name,
        len(original_desirable),
        len(valid_desirable),
        len(original_undesirable),
        len(valid_undesirable),
    )

    # Create new scenario with filtered clusters
    return PreferenceScenario(
        name=scenario.name,
        description=scenario.description,
        desirable_clusters=valid_desirable,
        undesirable_clusters=valid_undesirable,
        target_metric=scenario.target_metric,
        target_value=scenario.target_value,
        metadata={
            **scenario.metadata,
            "original_desirable_count": len(original_desirable),
            "original_undesirable_count": len(original_undesirable),
            "removed_desirable": list(removed_desirable),
            "removed_undesirable": list(removed_undesirable),
        },
    )


def get_adapted_scenarios(
    loader: ClusterProfileLoader,
    available_cluster_ids: List[int],
    config: Optional[InferenceConfig] = None,
    skip_invalid: bool = True,
) -> Dict[str, PreferenceScenario]:
    """Get predefined scenarios adapted to available clusters.

    This is the recommended way to get scenarios for simulation - it ensures
    all cluster IDs in scenarios are valid for the model's action space.

    Args:
        loader: ClusterProfileLoader instance.
        available_cluster_ids: List of cluster IDs available in the model.
        config: Optional InferenceConfig for target values.
        skip_invalid: If True, skip scenarios that have no valid clusters.
            If False, raise ValueError for invalid scenarios.

    Returns:
        Dictionary mapping scenario names to adapted PreferenceScenario objects.
    """
    # Get predefined scenarios
    scenarios = get_predefined_scenarios(loader, config)

    # Adapt each scenario
    adapted_scenarios = {}
    for name, scenario in scenarios.items():
        try:
            adapted = adapt_scenario_to_available_clusters(
                scenario,
                available_cluster_ids,
                warn_on_mismatch=True,
            )
            adapted_scenarios[name] = adapted
        except ValueError as e:
            if skip_invalid:
                logger.warning("Skipping scenario '%s': %s", name, e)
            else:
                raise

    logger.info(
        "Adapted %d/%d scenarios to available clusters",
        len(adapted_scenarios),
        len(scenarios),
    )

    return adapted_scenarios


@dataclass
class SimulatedFeedback:
    """Simulated human feedback on a generated sequence.

    Attributes:
        quality: Overall quality rating (1-5).
        coherence: Transition coherence rating (1-5).
        creativity: Novelty/creativity rating (1-5).
        musicality: Musical quality rating (1-5).
        overall: Overall composite rating (1-5).
        preference_alignment: How well sequence aligns with preference (0-1).
        desirable_ratio: Proportion of desirable clusters in sequence.
        undesirable_ratio: Proportion of undesirable clusters in sequence.
        skipped: Whether feedback was skipped.
    """

    quality: float = 3.0
    coherence: float = 3.0
    creativity: float = 3.0
    musicality: float = 3.0
    overall: float = 3.0
    preference_alignment: float = 0.5
    desirable_ratio: float = 0.0
    undesirable_ratio: float = 0.0
    skipped: bool = False

    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted feedback score.

        Args:
            weights: Weight for each dimension. Defaults to equal weights.

        Returns:
            Weighted average score (0-5 scale).
        """
        if weights is None:
            weights = {
                "quality": 0.3,
                "coherence": 0.2,
                "creativity": 0.2,
                "musicality": 0.3,
            }

        total_weight = 0.0
        total_score = 0.0

        for dim, weight in weights.items():
            value = getattr(self, dim, None)
            if value is not None:
                total_score += value * weight
                total_weight += weight

        if total_weight == 0:
            return self.overall

        return total_score / total_weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert feedback to dictionary."""
        return {
            "quality": self.quality,
            "coherence": self.coherence,
            "creativity": self.creativity,
            "musicality": self.musicality,
            "overall": self.overall,
            "preference_alignment": self.preference_alignment,
            "desirable_ratio": self.desirable_ratio,
            "undesirable_ratio": self.undesirable_ratio,
            "skipped": self.skipped,
        }


class PreferenceFeedbackSimulator:
    """Simulate user feedback based on preference scenario alignment.

    This class generates realistic simulated feedback for sequences based on
    how well they align with a given preference scenario. The feedback includes
    human-like noise and considers multiple dimensions of quality.

    Attributes:
        scenario: The preference scenario to use for evaluation.
        strictness: How harsh negative feedback is (1.0 = normal, 2.0 = very strict).
        noise_std: Standard deviation for human-like variability.
        base_rating: Base rating when sequence is neutral.
        rng: Random number generator for reproducibility.
        config: Optional InferenceConfig for configuration values.
    """

    def __init__(
        self,
        scenario: PreferenceScenario,
        strictness: Optional[float] = None,
        noise_std: Optional[float] = None,
        base_rating: Optional[float] = None,
        seed: Optional[int] = None,
        config: Optional[InferenceConfig] = None,
    ) -> None:
        """Initialize the feedback simulator.

        Args:
            scenario: Preference scenario defining desirable/undesirable clusters.
            strictness: Multiplier for penalty on undesirable clusters.
                If None and config provided, uses config.get_strictness().
            noise_std: Standard deviation for human-like noise.
                If None and config provided, uses config.get_noise_std().
            base_rating: Base rating for neutral sequences.
                If None and config provided, uses config.get_base_rating().
            seed: Random seed for reproducibility.
            config: Optional InferenceConfig for default values.
        """
        self.scenario = scenario
        self.config = config

        # Use config values if provided, otherwise use parameter defaults
        if strictness is None:
            self.strictness = config.get_strictness() if config else 1.0
        else:
            self.strictness = strictness

        if noise_std is None:
            self.noise_std = config.get_noise_std() if config else 0.3
        else:
            self.noise_std = noise_std

        if base_rating is None:
            self.base_rating = config.get_base_rating() if config else 3.0
        else:
            self.base_rating = base_rating

        # Store multipliers from config or defaults
        self._desirable_multiplier = (
            config.get_desirable_multiplier() if config else 2.0
        )
        self._undesirable_multiplier = (
            config.get_undesirable_multiplier() if config else 3.0
        )

        # Store dimension noise from config or defaults
        if config:
            self._dimension_noise = {
                "quality": config.get_dimension_noise("quality"),
                "coherence": config.get_dimension_noise("coherence"),
                "creativity": config.get_dimension_noise("creativity"),
                "musicality": config.get_dimension_noise("musicality"),
            }
        else:
            self._dimension_noise = {
                "quality": 0.2,
                "coherence": 0.3,
                "creativity": 0.4,
                "musicality": 0.2,
            }

        self.rng = np.random.default_rng(seed)

        # Pre-compute cluster sets for fast lookup
        self._desirable_set = set(scenario.desirable_clusters)
        self._undesirable_set = set(scenario.undesirable_clusters)

        logger.info(
            "Initialized PreferenceFeedbackSimulator for scenario '%s' "
            "(desirable: %d clusters, undesirable: %d clusters)",
            scenario.name,
            len(self._desirable_set),
            len(self._undesirable_set),
        )

    def compute_feedback(self, sequence: List[int]) -> SimulatedFeedback:
        """Generate simulated feedback for a sequence.

        The rating formula:
        - Base: 3.0 (neutral)
        - +0.5 per 10% desirable clusters (max +2.0)
        - -1.0 * strictness per 10% undesirable clusters (max -3.0)
        - Gaussian noise (std=noise_std)
        - Clamp to [1.0, 5.0]

        Args:
            sequence: List of cluster IDs in the generated sequence.

        Returns:
            SimulatedFeedback with all rating dimensions populated.
        """
        if not sequence:
            return SimulatedFeedback(skipped=True)

        # Calculate cluster distribution
        seq_len = len(sequence)
        desirable_count = sum(1 for c in sequence if c in self._desirable_set)
        undesirable_count = sum(1 for c in sequence if c in self._undesirable_set)

        desirable_ratio = desirable_count / seq_len
        undesirable_ratio = undesirable_count / seq_len
        preference_alignment = desirable_ratio - undesirable_ratio

        # Calculate base rating from alignment
        rating = self.base_rating
        rating += self._desirable_multiplier * desirable_ratio
        rating -= self._undesirable_multiplier * self.strictness * undesirable_ratio

        # Add human-like noise
        noise = self.rng.normal(0, self.noise_std)
        rating_with_noise = rating + noise

        # Clamp to valid range
        overall_rating = float(np.clip(rating_with_noise, 1.0, 5.0))

        # Generate dimension-specific ratings with correlated noise
        quality = float(
            np.clip(
                overall_rating + self.rng.normal(0, self._dimension_noise["quality"]),
                1.0,
                5.0,
            )
        )
        coherence = float(
            np.clip(
                overall_rating + self.rng.normal(0, self._dimension_noise["coherence"]),
                1.0,
                5.0,
            )
        )
        creativity = float(
            np.clip(
                overall_rating
                + self.rng.normal(0, self._dimension_noise["creativity"]),
                1.0,
                5.0,
            )
        )
        musicality = float(
            np.clip(
                overall_rating
                + self.rng.normal(0, self._dimension_noise["musicality"]),
                1.0,
                5.0,
            )
        )

        return SimulatedFeedback(
            quality=round(quality, 2),
            coherence=round(coherence, 2),
            creativity=round(creativity, 2),
            musicality=round(musicality, 2),
            overall=round(overall_rating, 2),
            preference_alignment=round(preference_alignment, 3),
            desirable_ratio=round(desirable_ratio, 3),
            undesirable_ratio=round(undesirable_ratio, 3),
            skipped=False,
        )

    def compute_batch_feedback(
        self, sequences: List[List[int]]
    ) -> List[SimulatedFeedback]:
        """Compute feedback for multiple sequences.

        Args:
            sequences: List of sequences (each a list of cluster IDs).

        Returns:
            List of SimulatedFeedback objects.
        """
        return [self.compute_feedback(seq) for seq in sequences]

    def get_feedback_threshold(self, level: str = "low") -> float:
        """Get feedback threshold for triggering adaptation.

        Args:
            level: Threshold level ("low", "medium", "high").

        Returns:
            Rating threshold value.
        """
        thresholds = {
            "low": 2.5,
            "medium": 3.0,
            "high": 3.5,
        }
        return thresholds.get(level, 3.0)

    def should_regenerate(
        self, feedback: SimulatedFeedback, threshold: float = 2.5
    ) -> bool:
        """Determine if sequence should be regenerated based on feedback.

        Args:
            feedback: The simulated feedback.
            threshold: Rating threshold below which regeneration is triggered.

        Returns:
            True if regeneration is recommended.
        """
        return not feedback.skipped and feedback.overall < threshold

    def reset_seed(self, seed: int) -> None:
        """Reset the random number generator with a new seed.

        Args:
            seed: New random seed.
        """
        self.rng = np.random.default_rng(seed)
        logger.debug("Reset RNG with seed %d", seed)
