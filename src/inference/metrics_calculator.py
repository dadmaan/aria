"""Metric calculation and statistical analysis for preference-guided simulation.

This module provides extensible metric calculation and statistical analysis
utilities for evaluating simulation results.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various metrics for sequence evaluation.

    This class provides a registry of metric calculators that can be
    extended with custom metrics.
    """

    # Mapping from metric name to calculator function
    METRIC_CALCULATORS: Dict[str, Callable] = {}

    @classmethod
    def register_metric(cls, name: str):
        """Decorator to register a metric calculator.

        Args:
            name: Name of the metric.

        Returns:
            Decorator function.

        Example:
            @MetricsCalculator.register_metric("my_metric")
            def calculate_my_metric(seq, desirable, undesirable):
                return some_value
        """

        def decorator(func: Callable) -> Callable:
            cls.METRIC_CALCULATORS[name] = func
            return func

        return decorator

    @classmethod
    def calculate(
        cls,
        metric_name: str,
        sequence: List[int],
        desirable_set: set,
        undesirable_set: set,
    ) -> float:
        """Calculate named metric for sequence.

        Args:
            metric_name: Name of metric to calculate.
            sequence: Cluster sequence to evaluate.
            desirable_set: Set of desirable cluster IDs.
            undesirable_set: Set of undesirable cluster IDs.

        Returns:
            Metric value as float.

        Raises:
            ValueError: If metric name is unknown.
        """
        if metric_name not in cls.METRIC_CALCULATORS:
            raise ValueError(
                f"Unknown metric: '{metric_name}'. "
                f"Available metrics: {list(cls.METRIC_CALCULATORS.keys())}"
            )

        return cls.METRIC_CALCULATORS[metric_name](
            sequence, desirable_set, undesirable_set
        )

    @classmethod
    def get_available_metrics(cls) -> List[str]:
        """Get list of available metric names.

        Returns:
            List of registered metric names.
        """
        return list(cls.METRIC_CALCULATORS.keys())

    @classmethod
    def is_registered(cls, metric_name: str) -> bool:
        """Check if a metric is registered.

        Args:
            metric_name: Name of metric to check.

        Returns:
            True if metric is registered.
        """
        return metric_name in cls.METRIC_CALCULATORS


# Register standard metrics


@MetricsCalculator.register_metric("desirable_ratio")
def _calc_desirable_ratio(seq: List[int], desirable: set, undesirable: set) -> float:
    """Calculate ratio of desirable clusters in sequence."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in desirable) / len(seq)


@MetricsCalculator.register_metric("undesirable_ratio")
def _calc_undesirable_ratio(seq: List[int], desirable: set, undesirable: set) -> float:
    """Calculate ratio of undesirable clusters in sequence."""
    if not seq:
        return 0.0
    return sum(1 for c in seq if c in undesirable) / len(seq)


@MetricsCalculator.register_metric("neutral_ratio")
def _calc_neutral_ratio(seq: List[int], desirable: set, undesirable: set) -> float:
    """Calculate ratio of neutral clusters (neither desirable nor undesirable)."""
    if not seq:
        return 1.0
    desirable_count = sum(1 for c in seq if c in desirable)
    undesirable_count = sum(1 for c in seq if c in undesirable)
    neutral_count = len(seq) - desirable_count - undesirable_count
    return neutral_count / len(seq)


@MetricsCalculator.register_metric("desirable_minus_undesirable")
def _calc_net_desirable(seq: List[int], desirable: set, undesirable: set) -> float:
    """Calculate desirable_ratio - undesirable_ratio (net desirable score)."""
    if not seq:
        return 0.0
    des_ratio = sum(1 for c in seq if c in desirable) / len(seq)
    undes_ratio = sum(1 for c in seq if c in undesirable) / len(seq)
    return des_ratio - undes_ratio


@MetricsCalculator.register_metric("desirable_concentration")
def _calc_desirable_concentration(
    seq: List[int], desirable: set, undesirable: set
) -> float:
    """Calculate how concentrated desirable clusters are (inverse entropy).

    Higher values indicate more concentrated/focused use of desirable clusters.
    """
    if not seq:
        return 0.0

    desirable_seq = [c for c in seq if c in desirable]
    if not desirable_seq:
        return 0.0

    # Count occurrences
    counts = Counter(desirable_seq)
    probs = np.array(list(counts.values())) / len(desirable_seq)

    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Return inverse (higher = more concentrated)
    max_entropy = np.log(len(counts) + 1e-10)
    if max_entropy < 1e-9:
        return 1.0
    return 1.0 - (entropy / max_entropy)


@MetricsCalculator.register_metric("desirable_diversity")
def _calc_desirable_diversity(
    seq: List[int], desirable: set, undesirable: set
) -> float:
    """Calculate diversity of desirable clusters used (normalized unique count).

    Higher values indicate more diverse use of desirable clusters.
    """
    if not seq:
        return 0.0

    desirable_seq = [c for c in seq if c in desirable]
    if not desirable_seq:
        return 0.0

    unique_desirable = len(set(desirable_seq))
    total_possible = len(desirable)

    if total_possible == 0:
        return 0.0

    return unique_desirable / total_possible


class StatisticalAnalyzer:
    """Perform statistical analysis on simulation results.

    This class provides methods for computing confidence intervals,
    statistical tests, and effect sizes.
    """

    @staticmethod
    def compute_confidence_interval(
        values: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Compute confidence interval for mean using t-distribution.

        Args:
            values: List of values to analyze.
            confidence: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (mean, lower_bound, upper_bound).
        """
        if len(values) < 2:
            # Not enough data for CI, return mean with same bounds
            mean_val = float(np.mean(values)) if values else 0.0
            return (mean_val, mean_val, mean_val)

        mean = float(np.mean(values))
        sem = float(stats.sem(values))

        # t-distribution for small samples
        interval = stats.t.interval(
            confidence,
            len(values) - 1,
            loc=mean,
            scale=sem,
        )

        return (mean, float(interval[0]), float(interval[1]))

    @staticmethod
    def paired_comparison_test(
        treatment: List[float],
        control: List[float],
    ) -> Dict[str, Any]:
        """Perform paired statistical test using Wilcoxon signed-rank test.

        This is a non-parametric test that doesn't assume normality,
        making it robust for small sample sizes.

        Args:
            treatment: Treatment group values.
            control: Control group values (must be same length as treatment).

        Returns:
            Dictionary with test results:
                - statistic: Test statistic
                - p_value: P-value
                - significant: Whether p < 0.05
                - effect_size: Cohen's d effect size
                - warning: Optional warning message

        Raises:
            ValueError: If treatment and control have different lengths.
        """
        if len(treatment) != len(control):
            raise ValueError(
                f"Treatment ({len(treatment)}) and control ({len(control)}) "
                "must have same length for paired test"
            )

        if len(treatment) < 3:
            # Not enough samples for reliable test
            logger.warning(
                "Only %d samples - statistical test may be unreliable",
                len(treatment),
            )
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
                "warning": "Insufficient samples for reliable test",
            }

        # Wilcoxon signed-rank test (non-parametric paired test)
        # Use one-sided test to detect if treatment > control
        try:
            statistic, p_value = stats.wilcoxon(
                treatment, control, alternative="greater"
            )
        except ValueError as e:
            # Can happen if all differences are zero
            logger.warning("Wilcoxon test failed: %s", str(e))
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
                "warning": str(e),
            }

        # Cohen's d effect size for paired samples
        diff = np.array(treatment) - np.array(control)
        effect_size = float(np.mean(diff) / (np.std(diff) + 1e-10))

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "effect_size": effect_size,
        }

    @staticmethod
    def permutation_test(
        treatment: List[float],
        control: List[float],
        n_permutations: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """Perform permutation test for difference in means.

        This is a robust non-parametric test that works well for small samples.

        Args:
            treatment: Treatment group values.
            control: Control group values.
            n_permutations: Number of permutations to perform.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with test results:
                - observed_diff: Observed difference in means
                - p_value: Permutation p-value
                - significant: Whether p < 0.05
        """
        observed_diff = np.mean(treatment) - np.mean(control)

        # Combine data
        combined = np.array(treatment + control)
        n_treatment = len(treatment)

        # Permutation test
        rng = np.random.RandomState(seed)
        perm_diffs = []

        for _ in range(n_permutations):
            perm = rng.permutation(combined)
            perm_treatment = perm[:n_treatment]
            perm_control = perm[n_treatment:]
            perm_diff = np.mean(perm_treatment) - np.mean(perm_control)
            perm_diffs.append(perm_diff)

        perm_diffs = np.array(perm_diffs)

        # Two-tailed p-value
        p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

        return {
            "observed_diff": float(observed_diff),
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_permutations": n_permutations,
        }

    @staticmethod
    def compute_effect_size_cohens_d(
        treatment: List[float],
        control: List[float],
        paired: bool = False,
    ) -> float:
        """Compute Cohen's d effect size.

        Args:
            treatment: Treatment group values.
            control: Control group values.
            paired: Whether samples are paired.

        Returns:
            Cohen's d effect size. Common interpretations:
                - |d| < 0.2: negligible
                - 0.2 ≤ |d| < 0.5: small
                - 0.5 ≤ |d| < 0.8: medium
                - |d| ≥ 0.8: large
        """
        if paired:
            # For paired samples, use difference scores
            diff = np.array(treatment) - np.array(control)
            return float(np.mean(diff) / (np.std(diff) + 1e-10))
        else:
            # For independent samples, use pooled standard deviation
            mean_diff = np.mean(treatment) - np.mean(control)
            n1, n2 = len(treatment), len(control)
            var1, var2 = np.var(treatment, ddof=1), np.var(control, ddof=1)

            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

            return float(mean_diff / (pooled_std + 1e-10))

    @staticmethod
    def interpret_effect_size(effect_size: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            effect_size: Cohen's d value.

        Returns:
            Interpretation string.
        """
        abs_d = abs(effect_size)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
