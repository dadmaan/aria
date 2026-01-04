"""Learning verification for HIL simulation results.

This module provides post-simulation verification to confirm that meaningful
learning actually occurred during HIL preference-guided simulation. It analyzes:

1. Sequence diversity - Are different sequences being generated?
2. Metric improvements - Did desirable ratio and feedback improve?
3. Trend analysis - Is there a positive learning trend?
4. Parameter changes - Did policy parameters actually change? (optional)

This addresses the critical finding from the investigation that previous
"learning" was often ephemeral Q-value modification rather than true learning.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class LearningVerificationResult:
    """Results from learning verification analysis.

    Contains comprehensive metrics about whether learning occurred during
    the HIL simulation, including sequence diversity, metric improvements,
    and trend analysis.

    Attributes:
        learning_detected: Whether meaningful learning was detected.
        confidence: Confidence score in learning detection (0.0 to 1.0).
        unique_sequences: Number of unique sequences generated.
        total_sequences: Total number of sequences generated.
        diversity_ratio: Ratio of unique to total sequences.
        desirable_change: Change in desirable ratio (final - initial).
        undesirable_change: Change in undesirable ratio (final - initial).
        feedback_change: Change in feedback rating (final - initial).
        desirable_trend_slope: Slope of desirable ratio over time.
        feedback_trend_slope: Slope of feedback rating over time.
        trend_p_value: P-value for trend significance (if computed).
        parameter_l2_change: L2 norm of parameter change (if available).
        q_value_distribution_shift: Shift in Q-value distribution (if available).
        diversity_threshold: Threshold used for diversity detection.
        improvement_threshold: Threshold used for improvement detection.
    """

    # Overall verdict
    learning_detected: bool
    confidence: float  # 0.0 to 1.0

    # Sequence diversity metrics
    unique_sequences: int
    total_sequences: int
    diversity_ratio: float

    # Metric improvement
    desirable_change: float
    undesirable_change: float
    feedback_change: float

    # Trend analysis
    desirable_trend_slope: float
    feedback_trend_slope: float
    trend_p_value: Optional[float] = None

    # Policy change metrics (if policy learning enabled)
    parameter_l2_change: Optional[float] = None
    q_value_distribution_shift: Optional[float] = None

    # Thresholds used for detection
    diversity_threshold: float = 0.02
    improvement_threshold: float = 0.05

    # Evidence details
    evidence_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of verification results.
        """
        return {
            "learning_detected": self.learning_detected,
            "confidence": self.confidence,
            "unique_sequences": self.unique_sequences,
            "total_sequences": self.total_sequences,
            "diversity_ratio": self.diversity_ratio,
            "desirable_change": self.desirable_change,
            "undesirable_change": self.undesirable_change,
            "feedback_change": self.feedback_change,
            "desirable_trend_slope": self.desirable_trend_slope,
            "feedback_trend_slope": self.feedback_trend_slope,
            "trend_p_value": self.trend_p_value,
            "parameter_l2_change": self.parameter_l2_change,
            "q_value_distribution_shift": self.q_value_distribution_shift,
            "diversity_threshold": self.diversity_threshold,
            "improvement_threshold": self.improvement_threshold,
            "evidence_details": self.evidence_details,
        }

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Summary string describing verification results.
        """
        status = "DETECTED" if self.learning_detected else "NOT DETECTED"
        lines = [
            f"Learning Verification: {status}",
            f"  Confidence: {self.confidence:.2%}",
            f"  Diversity: {self.unique_sequences}/{self.total_sequences} "
            f"({self.diversity_ratio:.2%})",
            f"  Desirable change: {self.desirable_change:+.3f}",
            f"  Feedback change: {self.feedback_change:+.3f}",
            f"  Trend slope: {self.desirable_trend_slope:.6f}",
        ]
        if self.parameter_l2_change is not None:
            lines.append(f"  Parameter L2 change: {self.parameter_l2_change:.6f}")
        return "\n".join(lines)


class SimulationLearningVerifier:
    """Verifies whether meaningful learning occurred during HIL simulation.

    Analyzes simulation results to determine if learning actually occurred,
    as opposed to ephemeral Q-value modifications that don't persist.

    The verifier looks for multiple types of evidence:
    1. Sequence diversity - Different sequences being generated
    2. Metric improvement - Desirable ratio and feedback increasing
    3. Positive trends - Upward trajectory in key metrics
    4. Parameter changes - Actual policy parameter modifications

    Attributes:
        diversity_threshold: Minimum diversity ratio to consider as evidence.
        improvement_threshold: Minimum metric improvement to consider.
        trend_significance: P-value threshold for trend significance.
    """

    def __init__(
        self,
        diversity_threshold: float = 0.02,
        improvement_threshold: float = 0.05,
        trend_significance: float = 0.05,
    ):
        """Initialize the learning verifier.

        Args:
            diversity_threshold: Min diversity ratio for evidence (default 0.02).
            improvement_threshold: Min metric improvement for evidence (default 0.05).
            trend_significance: P-value threshold for trends (default 0.05).
        """
        self.diversity_threshold = diversity_threshold
        self.improvement_threshold = improvement_threshold
        self.trend_significance = trend_significance

        logger.info(
            "SimulationLearningVerifier: diversity_thresh=%.3f, "
            "improvement_thresh=%.3f, trend_sig=%.3f",
            diversity_threshold,
            improvement_threshold,
            trend_significance,
        )

    def verify(
        self,
        metrics_history: List[Dict[str, Any]],
        initial_params: Optional[Dict[str, torch.Tensor]] = None,
        final_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> LearningVerificationResult:
        """Verify learning from simulation metrics.

        Analyzes the metrics history from a HIL simulation to determine
        if meaningful learning occurred.

        Args:
            metrics_history: List of metrics dictionaries from simulation.
                Each dict should contain at least:
                - sequence: List of cluster IDs
                - desirable_ratio: Float ratio of desirable clusters
                - undesirable_ratio: Float ratio of undesirable clusters
                - feedback_rating: Float feedback score
            initial_params: Initial policy parameters (optional).
            final_params: Final policy parameters (optional).

        Returns:
            LearningVerificationResult with comprehensive analysis.
        """
        if not metrics_history:
            logger.warning("Empty metrics history provided")
            return LearningVerificationResult(
                learning_detected=False,
                confidence=0.0,
                unique_sequences=0,
                total_sequences=0,
                diversity_ratio=0.0,
                desirable_change=0.0,
                undesirable_change=0.0,
                feedback_change=0.0,
                desirable_trend_slope=0.0,
                feedback_trend_slope=0.0,
                diversity_threshold=self.diversity_threshold,
                improvement_threshold=self.improvement_threshold,
            )

        # Extract sequences and metrics
        sequences = [tuple(m.get("sequence", [])) for m in metrics_history]
        desirable_ratios = [m.get("desirable_ratio", 0) for m in metrics_history]
        undesirable_ratios = [m.get("undesirable_ratio", 0) for m in metrics_history]
        feedback_ratings = [m.get("feedback_rating", 0) for m in metrics_history]

        # 1. Sequence diversity analysis
        unique_sequences = len(set(sequences))
        total_sequences = len(sequences)
        diversity_ratio = (
            unique_sequences / total_sequences if total_sequences > 0 else 0
        )

        # 2. Metric changes (first 50 vs last 50 iterations)
        n_compare = min(50, len(metrics_history) // 4)
        if n_compare > 0:
            initial_des = np.mean(desirable_ratios[:n_compare])
            final_des = np.mean(desirable_ratios[-n_compare:])
            initial_und = np.mean(undesirable_ratios[:n_compare])
            final_und = np.mean(undesirable_ratios[-n_compare:])
            initial_fdbk = np.mean(feedback_ratings[:n_compare])
            final_fdbk = np.mean(feedback_ratings[-n_compare:])
        else:
            initial_des = final_des = (
                desirable_ratios[0] if desirable_ratios else 0
            )
            initial_und = final_und = (
                undesirable_ratios[0] if undesirable_ratios else 0
            )
            initial_fdbk = final_fdbk = (
                feedback_ratings[0] if feedback_ratings else 0
            )

        desirable_change = final_des - initial_des
        undesirable_change = final_und - initial_und
        feedback_change = final_fdbk - initial_fdbk

        # 3. Trend analysis (linear regression)
        iterations = np.arange(len(desirable_ratios))
        des_slope = self._compute_slope(iterations, desirable_ratios)
        fdbk_slope = self._compute_slope(iterations, feedback_ratings)

        # 4. Parameter change (if available)
        param_change = None
        if initial_params is not None and final_params is not None:
            param_change = self._compute_param_change(initial_params, final_params)

        # 5. Gather evidence for learning detection
        evidence_details = {}
        learning_evidence = []

        # Evidence 1: Sequence diversity
        if diversity_ratio >= self.diversity_threshold:
            score = diversity_ratio / 0.1  # Scale to ~1.0 at 10% diversity
            learning_evidence.append(("diversity", min(score, 1.0)))
            evidence_details["diversity"] = {
                "present": True,
                "value": diversity_ratio,
                "score": min(score, 1.0),
            }
        else:
            evidence_details["diversity"] = {
                "present": False,
                "value": diversity_ratio,
                "threshold": self.diversity_threshold,
            }

        # Evidence 2: Desirable ratio improvement
        if desirable_change > self.improvement_threshold:
            score = desirable_change / 0.2  # Scale to ~1.0 at 20% improvement
            learning_evidence.append(("desirable_improvement", min(score, 1.0)))
            evidence_details["desirable_improvement"] = {
                "present": True,
                "value": desirable_change,
                "score": min(score, 1.0),
            }
        else:
            evidence_details["desirable_improvement"] = {
                "present": False,
                "value": desirable_change,
                "threshold": self.improvement_threshold,
            }

        # Evidence 3: Feedback improvement
        if feedback_change > 0.1:
            score = feedback_change / 0.5  # Scale to ~1.0 at 0.5 improvement
            learning_evidence.append(("feedback_improvement", min(score, 1.0)))
            evidence_details["feedback_improvement"] = {
                "present": True,
                "value": feedback_change,
                "score": min(score, 1.0),
            }
        else:
            evidence_details["feedback_improvement"] = {
                "present": False,
                "value": feedback_change,
                "threshold": 0.1,
            }

        # Evidence 4: Positive trend in desirable ratio
        if des_slope > 0:
            score = min(des_slope * 1000, 1.0)  # Scale appropriately
            learning_evidence.append(("positive_trend", score))
            evidence_details["positive_trend"] = {
                "present": True,
                "value": des_slope,
                "score": score,
            }
        else:
            evidence_details["positive_trend"] = {
                "present": False,
                "value": des_slope,
            }

        # Evidence 5: Parameter change (if available)
        if param_change is not None and param_change > 0.001:
            score = min(param_change * 100, 1.0)  # Scale appropriately
            learning_evidence.append(("param_change", score))
            evidence_details["param_change"] = {
                "present": True,
                "value": param_change,
                "score": score,
            }
        elif param_change is not None:
            evidence_details["param_change"] = {
                "present": False,
                "value": param_change,
                "threshold": 0.001,
            }

        # 6. Calculate confidence and determine verdict
        if learning_evidence:
            confidence = min(
                sum(e[1] for e in learning_evidence) / len(learning_evidence), 1.0
            )
            # Learning detected if at least 2 types of evidence and confidence > 0.3
            learning_detected = len(learning_evidence) >= 2 and confidence > 0.3
        else:
            confidence = 0.0
            learning_detected = False

        result = LearningVerificationResult(
            learning_detected=learning_detected,
            confidence=confidence,
            unique_sequences=unique_sequences,
            total_sequences=total_sequences,
            diversity_ratio=diversity_ratio,
            desirable_change=desirable_change,
            undesirable_change=undesirable_change,
            feedback_change=feedback_change,
            desirable_trend_slope=des_slope,
            feedback_trend_slope=fdbk_slope,
            parameter_l2_change=param_change,
            diversity_threshold=self.diversity_threshold,
            improvement_threshold=self.improvement_threshold,
            evidence_details=evidence_details,
        )

        logger.info(
            "Learning verification: detected=%s, confidence=%.2f, "
            "diversity=%.3f, des_change=%.3f, fdbk_change=%.3f",
            result.learning_detected,
            result.confidence,
            result.diversity_ratio,
            result.desirable_change,
            result.feedback_change,
        )

        return result

    def _compute_slope(self, x: np.ndarray, y: List[float]) -> float:
        """Compute linear regression slope.

        Args:
            x: X values (iterations).
            y: Y values (metric values).

        Returns:
            Slope of the linear fit.
        """
        if len(y) < 2:
            return 0.0

        y_arr = np.array(y)

        # Handle potential NaN values
        valid_mask = ~np.isnan(y_arr)
        if valid_mask.sum() < 2:
            return 0.0

        x_valid = x[valid_mask]
        y_valid = y_arr[valid_mask]

        try:
            coeffs = np.polyfit(x_valid, y_valid, 1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def _compute_param_change(
        self,
        initial: Dict[str, torch.Tensor],
        final: Dict[str, torch.Tensor],
    ) -> float:
        """Compute L2 norm of parameter change.

        Args:
            initial: Initial parameter dictionary.
            final: Final parameter dictionary.

        Returns:
            L2 norm of total parameter change.
        """
        total_change = 0.0

        for key in initial:
            if key in final:
                try:
                    diff = (final[key] - initial[key]).float()
                    total_change += torch.norm(diff).item() ** 2
                except Exception as e:
                    logger.warning("Error computing param change for %s: %s", key, e)

        return np.sqrt(total_change)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SimulationLearningVerifier":
        """Create verifier from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - diversity_threshold: float (default 0.02)
                - improvement_threshold: float (default 0.05)
                - trend_significance: float (default 0.05)

        Returns:
            Configured SimulationLearningVerifier instance.
        """
        return cls(
            diversity_threshold=config.get("diversity_threshold", 0.02),
            improvement_threshold=config.get("improvement_threshold", 0.05),
            trend_significance=config.get("trend_significance", 0.05),
        )
