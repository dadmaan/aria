"""Feedback Model Calibration for HIL Preference Simulation.

This module provides calibrated feedback generation that properly
separates threshold logic from base ratings, addressing the
investigation finding of conflated parameters.

Key improvements over the original feedback simulation:
- Positive/negative thresholds are independent of base rating
- Persona-based noise modeling with configurable consistency
- Seeded RNG for reproducible experiments
- Comprehensive calibration analysis tools
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeedbackThresholds:
    """Configuration for feedback classification thresholds.

    This dataclass properly separates the concept of classification thresholds
    from the base/neutral rating, addressing the conflation issue identified
    in the investigation report.

    Attributes:
        positive_threshold: Score above which feedback is classified as positive.
        negative_threshold: Score below which feedback is classified as negative.
        base_rating: The neutral/default rating (independent of thresholds).
        noise_std: Standard deviation of rating noise.
    """

    positive_threshold: float = 3.5
    negative_threshold: float = 2.5
    base_rating: float = 3.0
    noise_std: float = 0.3

    def validate(self) -> bool:
        """Validate that thresholds are logically consistent.

        Returns:
            True if thresholds are valid, raises ValueError otherwise.

        Raises:
            ValueError: If thresholds are inconsistent or out of valid range.
        """
        # Check basic ordering
        if self.negative_threshold >= self.positive_threshold:
            raise ValueError(
                f"negative_threshold ({self.negative_threshold}) must be less than "
                f"positive_threshold ({self.positive_threshold})"
            )

        # Check base rating is in a sensible range
        if not (1.0 <= self.base_rating <= 5.0):
            raise ValueError(
                f"base_rating ({self.base_rating}) should be in range [1.0, 5.0]"
            )

        # Check thresholds are in valid range
        if not (1.0 <= self.negative_threshold <= 5.0):
            raise ValueError(
                f"negative_threshold ({self.negative_threshold}) should be in range [1.0, 5.0]"
            )

        if not (1.0 <= self.positive_threshold <= 5.0):
            raise ValueError(
                f"positive_threshold ({self.positive_threshold}) should be in range [1.0, 5.0]"
            )

        # Check noise is non-negative
        if self.noise_std < 0:
            raise ValueError(f"noise_std ({self.noise_std}) must be non-negative")

        # Warn if base_rating is outside the neutral zone
        if self.base_rating <= self.negative_threshold:
            logger.warning(
                f"base_rating ({self.base_rating}) is at or below negative_threshold "
                f"({self.negative_threshold}), which may cause unexpected behavior"
            )
        elif self.base_rating >= self.positive_threshold:
            logger.warning(
                f"base_rating ({self.base_rating}) is at or above positive_threshold "
                f"({self.positive_threshold}), which may cause unexpected behavior"
            )

        logger.debug(
            f"Thresholds validated: positive={self.positive_threshold}, "
            f"negative={self.negative_threshold}, base={self.base_rating}"
        )
        return True

    @property
    def neutral_zone_width(self) -> float:
        """Width of the neutral zone between thresholds."""
        return self.positive_threshold - self.negative_threshold

    def __post_init__(self):
        """Validate thresholds on creation."""
        self.validate()


@dataclass
class PersonaConfig:
    """Configuration for simulated human persona characteristics.

    Models individual differences in how humans provide feedback,
    including strictness, consistency, and preference strength.

    Attributes:
        strictness: How strict the persona is (0.0=lenient, 1.0=strict).
            Higher values result in more extreme ratings.
        consistency: How consistent ratings are (0.0=inconsistent, 1.0=very consistent).
            Higher values result in lower noise.
        preference_strength: Magnitude of preferences (0.0=weak, 1.0=strong).
            Higher values result in larger deviations from base rating.
    """

    strictness: float = 0.5
    consistency: float = 0.8
    preference_strength: float = 0.6

    def __post_init__(self):
        """Validate persona parameters."""
        for param_name, value in [
            ("strictness", self.strictness),
            ("consistency", self.consistency),
            ("preference_strength", self.preference_strength),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{param_name} ({value}) must be in range [0.0, 1.0]")

    def to_noise_params(self) -> Dict[str, float]:
        """Convert persona parameters to noise model parameters.

        Returns:
            Dictionary containing:
                - noise_std: Standard deviation for rating noise
                - bias: Systematic bias in ratings
                - rating_scale: Scale factor for preference deviations
        """
        # Lower consistency = higher noise
        noise_std = 0.1 + 0.5 * (1.0 - self.consistency)

        # Higher strictness = more negative bias
        bias = -0.3 * self.strictness + 0.15  # Range: [0.15, -0.15]

        # Preference strength affects how far ratings deviate from base
        rating_scale = 0.5 + 1.0 * self.preference_strength  # Range: [0.5, 1.5]

        return {
            "noise_std": noise_std,
            "bias": bias,
            "rating_scale": rating_scale,
        }

    def describe(self) -> str:
        """Get a human-readable description of the persona."""
        strictness_desc = (
            "lenient"
            if self.strictness < 0.3
            else "moderate" if self.strictness < 0.7 else "strict"
        )
        consistency_desc = (
            "inconsistent"
            if self.consistency < 0.3
            else "somewhat consistent" if self.consistency < 0.7 else "very consistent"
        )
        preference_desc = (
            "weak"
            if self.preference_strength < 0.3
            else "moderate" if self.preference_strength < 0.7 else "strong"
        )
        return (
            f"A {strictness_desc} rater with {consistency_desc} ratings "
            f"and {preference_desc} preferences"
        )


class CalibratedFeedbackGenerator:
    """Generates calibrated feedback ratings based on cluster membership.

    This class produces realistic human-like feedback that properly
    separates threshold logic from base ratings.

    Attributes:
        thresholds: FeedbackThresholds configuration.
        persona: PersonaConfig for the simulated human.
        rng: Seeded random number generator for reproducibility.
    """

    def __init__(
        self,
        thresholds: Optional[FeedbackThresholds] = None,
        persona: Optional[PersonaConfig] = None,
        seed: Optional[int] = None,
    ):
        """Initialize the feedback generator.

        Args:
            thresholds: Feedback threshold configuration. Uses defaults if None.
            persona: Persona configuration. Uses defaults if None.
            seed: Random seed for reproducibility. Uses random seed if None.
        """
        self.thresholds = thresholds or FeedbackThresholds()
        self.persona = persona or PersonaConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Precompute noise parameters from persona
        self._noise_params = self.persona.to_noise_params()

        logger.info(
            f"Initialized CalibratedFeedbackGenerator with seed={seed}, "
            f"persona={self.persona.describe()}"
        )

    def generate_rating(
        self,
        cluster_id: int,
        desirable_clusters: Set[int],
        undesirable_clusters: Set[int],
    ) -> float:
        """Generate a rating for a given cluster.

        Args:
            cluster_id: The cluster ID to rate.
            desirable_clusters: Set of cluster IDs considered desirable.
            undesirable_clusters: Set of cluster IDs considered undesirable.

        Returns:
            A rating value, typically in range [1.0, 5.0].
        """
        # Determine base rating based on cluster membership
        if cluster_id in desirable_clusters:
            target_rating = self._compute_desirable_rating()
        elif cluster_id in undesirable_clusters:
            target_rating = self._compute_undesirable_rating()
        else:
            target_rating = self._compute_neutral_rating()

        # Add persona-based noise
        noise = self.rng.normal(0, self._noise_params["noise_std"])
        bias = self._noise_params["bias"]

        rating = target_rating + noise + bias

        # Clamp to valid range
        rating = np.clip(rating, 1.0, 5.0)

        logger.debug(
            f"Generated rating {rating:.2f} for cluster {cluster_id} "
            f"(desirable={cluster_id in desirable_clusters}, "
            f"undesirable={cluster_id in undesirable_clusters})"
        )

        return float(rating)

    def _compute_desirable_rating(self) -> float:
        """Compute rating for a desirable cluster."""
        # Start above positive threshold
        base = self.thresholds.positive_threshold

        # Add extra based on preference strength
        scale = self._noise_params["rating_scale"]
        extra = (5.0 - base) * 0.5 * scale

        return base + extra

    def _compute_undesirable_rating(self) -> float:
        """Compute rating for an undesirable cluster."""
        # Start below negative threshold
        base = self.thresholds.negative_threshold

        # Subtract extra based on preference strength
        scale = self._noise_params["rating_scale"]
        extra = (base - 1.0) * 0.5 * scale

        return base - extra

    def _compute_neutral_rating(self) -> float:
        """Compute rating for a neutral cluster."""
        # Use base rating with small random variation
        variation = self.rng.uniform(-0.2, 0.2) * self._noise_params["rating_scale"]
        return self.thresholds.base_rating + variation

    def generate_feedback(self, rating: float) -> str:
        """Convert a numerical rating to categorical feedback.

        Args:
            rating: The numerical rating value.

        Returns:
            "positive", "negative", or "neutral" based on thresholds.
        """
        if rating >= self.thresholds.positive_threshold:
            return "positive"
        elif rating <= self.thresholds.negative_threshold:
            return "negative"
        else:
            return "neutral"

    def compute_preference_signal(self, rating: float) -> float:
        """Convert rating to a normalized preference signal.

        Maps the rating to a signal in [-1, 1] where:
        - -1 represents strong negative preference
        - 0 represents neutral
        - +1 represents strong positive preference

        Args:
            rating: The numerical rating value.

        Returns:
            Normalized preference signal in [-1, 1].
        """
        base = self.thresholds.base_rating

        if rating >= base:
            # Positive direction: map [base, 5] -> [0, 1]
            signal = (rating - base) / (5.0 - base)
        else:
            # Negative direction: map [1, base] -> [-1, 0]
            signal = (rating - base) / (base - 1.0)

        return float(np.clip(signal, -1.0, 1.0))

    def generate_batch_ratings(
        self,
        cluster_ids: List[int],
        desirable_clusters: Set[int],
        undesirable_clusters: Set[int],
    ) -> List[Tuple[int, float, str, float]]:
        """Generate ratings for a batch of clusters.

        Args:
            cluster_ids: List of cluster IDs to rate.
            desirable_clusters: Set of desirable cluster IDs.
            undesirable_clusters: Set of undesirable cluster IDs.

        Returns:
            List of tuples (cluster_id, rating, feedback, signal).
        """
        results = []
        for cid in cluster_ids:
            rating = self.generate_rating(cid, desirable_clusters, undesirable_clusters)
            feedback = self.generate_feedback(rating)
            signal = self.compute_preference_signal(rating)
            results.append((cid, rating, feedback, signal))
        return results

    def reset_rng(self, seed: Optional[int] = None):
        """Reset the random number generator.

        Args:
            seed: New seed value. Uses original seed if None.
        """
        self.rng = np.random.default_rng(seed if seed is not None else self.seed)
        logger.debug(f"Reset RNG with seed={seed if seed is not None else self.seed}")


class FeedbackCalibrationAnalyzer:
    """Analyzes feedback distributions and provides calibration recommendations.

    This class helps validate that feedback generation is properly calibrated
    and provides tools for adjusting parameters.
    """

    def analyze_feedback_distribution(self, ratings: List[float]) -> Dict[str, Any]:
        """Analyze the statistical distribution of ratings.

        Args:
            ratings: List of numerical ratings.

        Returns:
            Dictionary containing distribution statistics.
        """
        if not ratings:
            logger.warning("Empty ratings list provided for analysis")
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "quartiles": None,
            }

        ratings_arr = np.array(ratings)

        stats = {
            "count": len(ratings),
            "mean": float(np.mean(ratings_arr)),
            "std": float(np.std(ratings_arr)),
            "min": float(np.min(ratings_arr)),
            "max": float(np.max(ratings_arr)),
            "quartiles": {
                "q25": float(np.percentile(ratings_arr, 25)),
                "q50": float(np.percentile(ratings_arr, 50)),
                "q75": float(np.percentile(ratings_arr, 75)),
            },
            "skewness": float(self._compute_skewness(ratings_arr)),
            "kurtosis": float(self._compute_kurtosis(ratings_arr)),
        }

        logger.info(
            f"Analyzed {stats['count']} ratings: mean={stats['mean']:.2f}, "
            f"std={stats['std']:.2f}"
        )

        return stats

    def _compute_skewness(self, arr: np.ndarray) -> float:
        """Compute skewness of the distribution."""
        if len(arr) < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    def _compute_kurtosis(self, arr: np.ndarray) -> float:
        """Compute excess kurtosis of the distribution."""
        if len(arr) < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3)

    def compute_signal_to_noise_ratio(
        self,
        ratings: List[float],
        cluster_labels: List[str],
    ) -> float:
        """Compute the signal-to-noise ratio of feedback.

        SNR measures how well the ratings discriminate between
        desirable, neutral, and undesirable clusters.

        Args:
            ratings: List of numerical ratings.
            cluster_labels: List of labels ("desirable", "neutral", "undesirable").

        Returns:
            Signal-to-noise ratio (higher is better).
        """
        if len(ratings) != len(cluster_labels):
            raise ValueError(
                f"ratings length ({len(ratings)}) must match "
                f"cluster_labels length ({len(cluster_labels)})"
            )

        if not ratings:
            return 0.0

        ratings_arr = np.array(ratings)
        labels_arr = np.array(cluster_labels)

        # Group ratings by label
        groups = {}
        for label in ["desirable", "neutral", "undesirable"]:
            mask = labels_arr == label
            if np.any(mask):
                groups[label] = ratings_arr[mask]

        if len(groups) < 2:
            logger.warning("Need at least 2 different labels to compute SNR")
            return 0.0

        # Compute between-group variance (signal)
        group_means = [np.mean(g) for g in groups.values()]
        overall_mean = np.mean(ratings_arr)
        between_var = np.mean([(m - overall_mean) ** 2 for m in group_means])

        # Compute within-group variance (noise)
        within_vars = [np.var(g) for g in groups.values()]
        within_var = np.mean(within_vars) if within_vars else 0.0

        # Compute SNR
        if within_var == 0:
            snr = float("inf") if between_var > 0 else 0.0
        else:
            snr = between_var / within_var

        logger.info(
            f"Computed SNR: {snr:.2f} (between_var={between_var:.3f}, within_var={within_var:.3f})"
        )

        return float(snr)

    def recommend_threshold_adjustment(
        self,
        ratings: List[float],
        cluster_labels: List[str],
    ) -> Dict[str, Any]:
        """Recommend threshold adjustments based on rating distribution.

        Args:
            ratings: List of numerical ratings.
            cluster_labels: List of labels ("desirable", "neutral", "undesirable").

        Returns:
            Dictionary with adjustment recommendations.
        """
        if len(ratings) != len(cluster_labels):
            raise ValueError(
                f"ratings length ({len(ratings)}) must match "
                f"cluster_labels length ({len(cluster_labels)})"
            )

        if not ratings:
            return {"status": "error", "message": "No ratings provided"}

        ratings_arr = np.array(ratings)
        labels_arr = np.array(cluster_labels)

        recommendations = {
            "status": "ok",
            "current_snr": self.compute_signal_to_noise_ratio(ratings, cluster_labels),
            "suggestions": [],
        }

        # Analyze each group
        for label in ["desirable", "neutral", "undesirable"]:
            mask = labels_arr == label
            if np.any(mask):
                group_ratings = ratings_arr[mask]
                recommendations[f"{label}_stats"] = {
                    "mean": float(np.mean(group_ratings)),
                    "std": float(np.std(group_ratings)),
                    "count": int(np.sum(mask)),
                }

        # Check for overlap between groups
        if "desirable_stats" in recommendations and "neutral_stats" in recommendations:
            des_mean = recommendations["desirable_stats"]["mean"]
            neu_mean = recommendations["neutral_stats"]["mean"]
            des_std = recommendations["desirable_stats"]["std"]

            if des_mean - neu_mean < 2 * des_std:
                recommendations["suggestions"].append(
                    {
                        "type": "increase_positive_threshold",
                        "reason": "Desirable and neutral ratings overlap significantly",
                        "suggested_threshold": float(neu_mean + des_std),
                    }
                )

        if (
            "undesirable_stats" in recommendations
            and "neutral_stats" in recommendations
        ):
            und_mean = recommendations["undesirable_stats"]["mean"]
            neu_mean = recommendations["neutral_stats"]["mean"]
            und_std = recommendations["undesirable_stats"]["std"]

            if neu_mean - und_mean < 2 * und_std:
                recommendations["suggestions"].append(
                    {
                        "type": "decrease_negative_threshold",
                        "reason": "Undesirable and neutral ratings overlap significantly",
                        "suggested_threshold": float(neu_mean - und_std),
                    }
                )

        # Check overall noise level
        overall_std = float(np.std(ratings_arr))
        if overall_std > 0.8:
            recommendations["suggestions"].append(
                {
                    "type": "reduce_noise",
                    "reason": f"High overall variance (std={overall_std:.2f})",
                    "suggested_action": "Increase persona consistency or reduce noise_std",
                }
            )

        logger.info(
            f"Generated {len(recommendations['suggestions'])} threshold adjustment suggestions"
        )

        return recommendations

    def validate_calibration(
        self,
        generator: CalibratedFeedbackGenerator,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Validate calibration by generating sample feedback.

        Args:
            generator: The feedback generator to validate.
            n_samples: Number of samples to generate per category.

        Returns:
            Dictionary with validation metrics.
        """
        # Reset generator for reproducibility
        generator.reset_rng()

        # Generate samples for each category
        desirable_clusters = {0, 1, 2}
        undesirable_clusters = {7, 8, 9}
        neutral_cluster = 5

        desirable_ratings = []
        undesirable_ratings = []
        neutral_ratings = []

        for _ in range(n_samples):
            # Sample from desirable
            des_rating = generator.generate_rating(
                self._random_choice(list(desirable_clusters)),
                desirable_clusters,
                undesirable_clusters,
            )
            desirable_ratings.append(des_rating)

            # Sample from undesirable
            und_rating = generator.generate_rating(
                self._random_choice(list(undesirable_clusters)),
                desirable_clusters,
                undesirable_clusters,
            )
            undesirable_ratings.append(und_rating)

            # Sample from neutral
            neu_rating = generator.generate_rating(
                neutral_cluster,
                desirable_clusters,
                undesirable_clusters,
            )
            neutral_ratings.append(neu_rating)

        # Compute validation metrics
        validation = {
            "n_samples": n_samples,
            "desirable": self.analyze_feedback_distribution(desirable_ratings),
            "undesirable": self.analyze_feedback_distribution(undesirable_ratings),
            "neutral": self.analyze_feedback_distribution(neutral_ratings),
        }

        # Compute separation metrics
        des_mean = validation["desirable"]["mean"]
        und_mean = validation["undesirable"]["mean"]
        neu_mean = validation["neutral"]["mean"]

        validation["separation"] = {
            "desirable_neutral": des_mean - neu_mean if des_mean and neu_mean else None,
            "neutral_undesirable": (
                neu_mean - und_mean if neu_mean and und_mean else None
            ),
            "desirable_undesirable": (
                des_mean - und_mean if des_mean and und_mean else None
            ),
        }

        # Compute classification accuracy
        all_ratings = desirable_ratings + neutral_ratings + undesirable_ratings
        all_labels = (
            ["desirable"] * n_samples
            + ["neutral"] * n_samples
            + ["undesirable"] * n_samples
        )

        validation["snr"] = self.compute_signal_to_noise_ratio(all_ratings, all_labels)

        # Check threshold effectiveness
        des_correct = sum(
            1 for r in desirable_ratings if r >= generator.thresholds.positive_threshold
        )
        und_correct = sum(
            1
            for r in undesirable_ratings
            if r <= generator.thresholds.negative_threshold
        )
        neu_correct = sum(
            1
            for r in neutral_ratings
            if generator.thresholds.negative_threshold
            < r
            < generator.thresholds.positive_threshold
        )

        validation["classification_accuracy"] = {
            "desirable": des_correct / n_samples,
            "undesirable": und_correct / n_samples,
            "neutral": neu_correct / n_samples,
            "overall": (des_correct + und_correct + neu_correct) / (3 * n_samples),
        }

        logger.info(
            f"Calibration validation complete: SNR={validation['snr']:.2f}, "
            f"overall accuracy={validation['classification_accuracy']['overall']:.2%}"
        )

        return validation

    def _random_choice(self, items: List[Any]) -> Any:
        """Simple random choice helper."""
        import random

        return random.choice(items)


# =============================================================================
# Calibration Presets
# =============================================================================

STRICT_PERSONA = PersonaConfig(
    strictness=0.8,
    consistency=0.9,
    preference_strength=0.8,
)
"""Strict persona: high standards, consistent ratings, strong preferences."""

LENIENT_PERSONA = PersonaConfig(
    strictness=0.2,
    consistency=0.7,
    preference_strength=0.4,
)
"""Lenient persona: low standards, somewhat consistent, weak preferences."""

REALISTIC_HUMAN = PersonaConfig(
    strictness=0.5,
    consistency=0.75,
    preference_strength=0.6,
)
"""Realistic human: moderate across all dimensions."""

NOISY_HUMAN = PersonaConfig(
    strictness=0.5,
    consistency=0.4,
    preference_strength=0.5,
)
"""Noisy human: inconsistent ratings with moderate preferences."""


# =============================================================================
# Convenience Functions
# =============================================================================


def create_calibrated_generator(
    preset: str = "realistic",
    seed: Optional[int] = None,
    thresholds: Optional[FeedbackThresholds] = None,
) -> CalibratedFeedbackGenerator:
    """Create a calibrated feedback generator with a preset persona.

    Args:
        preset: One of "strict", "lenient", "realistic", "noisy".
        seed: Random seed for reproducibility.
        thresholds: Custom thresholds (uses defaults if None).

    Returns:
        Configured CalibratedFeedbackGenerator instance.
    """
    persona_map = {
        "strict": STRICT_PERSONA,
        "lenient": LENIENT_PERSONA,
        "realistic": REALISTIC_HUMAN,
        "noisy": NOISY_HUMAN,
    }

    if preset not in persona_map:
        raise ValueError(
            f"Unknown preset '{preset}'. Choose from: {list(persona_map.keys())}"
        )

    return CalibratedFeedbackGenerator(
        thresholds=thresholds,
        persona=persona_map[preset],
        seed=seed,
    )


def quick_calibration_check(
    generator: CalibratedFeedbackGenerator,
    n_samples: int = 500,
) -> bool:
    """Quick check if generator is reasonably calibrated.

    Args:
        generator: The generator to check.
        n_samples: Number of samples for validation.

    Returns:
        True if calibration passes basic checks.
    """
    analyzer = FeedbackCalibrationAnalyzer()
    validation = analyzer.validate_calibration(generator, n_samples)

    # Check basic criteria
    snr_ok = validation["snr"] > 1.0
    accuracy_ok = validation["classification_accuracy"]["overall"] > 0.6
    separation_ok = (
        validation["separation"]["desirable_undesirable"] is not None
        and validation["separation"]["desirable_undesirable"] > 1.0
    )

    passed = snr_ok and accuracy_ok and separation_ok

    if not passed:
        logger.warning(
            f"Calibration check failed: SNR={validation['snr']:.2f} (need >1.0), "
            f"accuracy={validation['classification_accuracy']['overall']:.2%} (need >60%), "
            f"separation={validation['separation']['desirable_undesirable']}"
        )
    else:
        logger.info("Calibration check passed")

    return passed
