"""
Statistical metrics and analysis functions for PoC validation.

Provides functions for significance testing, effect size calculation,
confidence intervals, and reward statistics computation.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """Results from a statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_ci: Tuple[float, float]
    mean_diff: float
    mean_diff_ci: Tuple[float, float]
    sample_size_1: int
    sample_size_2: int
    significant: bool  # p < 0.05
    interpretation: str


@dataclass
class RewardStatistics:
    """Comprehensive statistics for reward distributions."""

    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    n_samples: int
    ci_95: Tuple[float, float]


def calculate_statistical_significance(
    group1: np.ndarray,
    group2: np.ndarray,
    test_type: str = "auto",
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """
    Calculate statistical significance between two groups.

    Args:
        group1: First group of samples
        group2: Second group of samples
        test_type: Type of test ("auto", "t-test", "mann-whitney", "welch")
        alpha: Significance level (default: 0.05)

    Returns:
        StatisticalTestResult with test statistics and interpretation
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    # Remove NaN values
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        raise ValueError("Groups must contain at least one valid sample")

    # Auto-select test based on normality and variance
    if test_type == "auto":
        # Shapiro-Wilk test for normality (only for n > 3 and n < 5000)
        n1, n2 = len(group1), len(group2)
        normal1 = normal2 = True

        if 3 < n1 < 5000:
            _, p1 = stats.shapiro(group1)
            normal1 = p1 > 0.05

        if 3 < n2 < 5000:
            _, p2 = stats.shapiro(group2)
            normal2 = p2 > 0.05

        if normal1 and normal2:
            # Check equal variance
            _, p_var = stats.levene(group1, group2)
            test_type = "t-test" if p_var > 0.05 else "welch"
        else:
            test_type = "mann-whitney"

    # Perform the test
    if test_type == "t-test":
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        test_name = "Independent t-test"
    elif test_type == "welch":
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_name = "Welch's t-test"
    elif test_type == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Calculate effect size
    effect_size = compute_effect_size(group1, group2)
    effect_size_ci = bootstrap_effect_size_ci(group1, group2)

    # Mean difference and CI
    mean_diff = np.mean(group1) - np.mean(group2)
    mean_diff_ci = bootstrap_mean_diff_ci(group1, group2)

    # Interpret results
    significant = p_value < alpha

    if abs(effect_size) < 0.2:
        magnitude = "negligible"
    elif abs(effect_size) < 0.5:
        magnitude = "small"
    elif abs(effect_size) < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    direction = "higher" if mean_diff > 0 else "lower"
    sig_text = "significantly" if significant else "not significantly"

    interpretation = (
        f"Group 1 has {sig_text} {direction} values than Group 2 "
        f"(p={p_value:.4f}, d={effect_size:.3f}, {magnitude} effect)"
    )

    return StatisticalTestResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_ci=effect_size_ci,
        mean_diff=mean_diff,
        mean_diff_ci=mean_diff_ci,
        sample_size_1=len(group1),
        sample_size_2=len(group2),
        significant=significant,
        interpretation=interpretation,
    )


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    method: str = "cohen_d"
) -> float:
    """
    Compute effect size (Cohen's d or Glass's delta).

    Args:
        group1: First group of samples
        group2: Second group of samples
        method: Effect size method ("cohen_d" or "glass_delta")

    Returns:
        Effect size value
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    mean1, mean2 = np.mean(group1), np.mean(group2)

    if method == "cohen_d":
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    elif method == "glass_delta":
        # Use control group (group2) standard deviation
        std2 = np.std(group2, ddof=1)

        if std2 == 0:
            return 0.0

        return (mean1 - mean2) / std2

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    method: str = "bootstrap"
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean.

    Args:
        data: Array of samples
        confidence: Confidence level (default: 0.95)
        method: Method ("bootstrap", "t-distribution", "percentile")

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    data = np.asarray(data).flatten()
    data = data[~np.isnan(data)]

    if len(data) == 0:
        return (np.nan, np.nan)

    if len(data) == 1:
        return (data[0], data[0])

    if method == "bootstrap":
        return bootstrap_ci(data, confidence=confidence)

    elif method == "t-distribution":
        mean = np.mean(data)
        sem = stats.sem(data)
        df = len(data) - 1
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        margin = t_crit * sem
        return (mean - margin, mean + margin)

    elif method == "percentile":
        alpha = 1 - confidence
        lower_p = (alpha / 2) * 100
        upper_p = (1 - alpha / 2) * 100
        return (np.percentile(data, lower_p), np.percentile(data, upper_p))

    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic_fn = np.mean,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a statistic.

    Args:
        data: Array of samples
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        statistic_fn: Function to compute statistic (default: mean)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    data = np.asarray(data).flatten()
    n = len(data)

    bootstrap_statistics = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_statistics[i] = statistic_fn(sample)

    alpha = 1 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    return (np.percentile(bootstrap_statistics, lower_p),
            np.percentile(bootstrap_statistics, upper_p))


def bootstrap_effect_size_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for Cohen's d effect size."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    n1, n2 = len(group1), len(group2)

    effect_sizes = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=n1, replace=True)
        sample2 = rng.choice(group2, size=n2, replace=True)
        effect_sizes[i] = compute_effect_size(sample1, sample2)

    alpha = 1 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    return (np.percentile(effect_sizes, lower_p),
            np.percentile(effect_sizes, upper_p))


def bootstrap_mean_diff_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for mean difference."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    n1, n2 = len(group1), len(group2)

    mean_diffs = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample1 = rng.choice(group1, size=n1, replace=True)
        sample2 = rng.choice(group2, size=n2, replace=True)
        mean_diffs[i] = np.mean(sample1) - np.mean(sample2)

    alpha = 1 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100

    return (np.percentile(mean_diffs, lower_p),
            np.percentile(mean_diffs, upper_p))


def compute_reward_statistics(rewards: np.ndarray) -> RewardStatistics:
    """
    Compute comprehensive statistics for reward distribution.

    Args:
        rewards: Array of reward values

    Returns:
        RewardStatistics with all computed metrics
    """
    rewards = np.asarray(rewards).flatten()
    rewards = rewards[~np.isnan(rewards)]

    if len(rewards) == 0:
        return RewardStatistics(
            mean=np.nan, std=np.nan, median=np.nan,
            min=np.nan, max=np.nan, q25=np.nan, q75=np.nan,
            iqr=np.nan, n_samples=0, ci_95=(np.nan, np.nan)
        )

    mean = np.mean(rewards)
    std = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
    median = np.median(rewards)
    min_val = np.min(rewards)
    max_val = np.max(rewards)
    q25 = np.percentile(rewards, 25)
    q75 = np.percentile(rewards, 75)
    iqr = q75 - q25
    ci_95 = calculate_confidence_interval(rewards, confidence=0.95)

    return RewardStatistics(
        mean=mean,
        std=std,
        median=median,
        min=min_val,
        max=max_val,
        q25=q25,
        q75=q75,
        iqr=iqr,
        n_samples=len(rewards),
        ci_95=ci_95,
    )


def compare_multiple_groups(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], StatisticalTestResult]:
    """
    Perform pairwise comparisons between multiple groups.

    Args:
        groups: Dictionary mapping group names to arrays of samples
        alpha: Significance level

    Returns:
        Dictionary mapping (group1_name, group2_name) to test results
    """
    results = {}
    group_names = list(groups.keys())

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name1, name2 = group_names[i], group_names[j]
            result = calculate_statistical_significance(
                groups[name1],
                groups[name2],
                alpha=alpha,
            )
            results[(name1, name2)] = result

    return results
