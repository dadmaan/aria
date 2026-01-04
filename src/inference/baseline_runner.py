"""Baseline/control simulation for comparison with adapted policies.

This module provides baseline runners that execute simulations without
adaptation to establish control groups for statistical comparison.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .preference_guided_session import PreferenceGuidedSession, SimulationResult
from .seed_manager import SeedManager

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Run baseline simulations without adaptation for comparison.

    This class provides methods to run control simulations that can be
    compared against treatment (adapted) simulations to establish
    statistical significance of adaptation effects.
    """

    @staticmethod
    def run_no_adaptation_baseline(
        session: PreferenceGuidedSession,
        num_iterations: int,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> SimulationResult:
        """Run simulation without any adaptation.

        This temporarily disables adaptation to establish a baseline
        for the agent's behavior without preference guidance.

        Args:
            session: PreferenceGuidedSession instance.
            num_iterations: Number of iterations to run.
            seed: Optional random seed (if None, uses session's seed).
            verbose: Whether to print progress.

        Returns:
            SimulationResult with adaptation disabled.
        """
        # Store original settings
        original_mode = session.adaptation_mode
        original_strength = session.adaptation_strength
        original_penalties = session.cluster_penalties.copy()

        # Temporarily disable adaptation
        session.adaptation_mode = "none"
        session.adaptation_strength = 0.0
        session.cluster_penalties = {}

        # Reseed if requested
        if seed is not None:
            session.seed_manager = SeedManager(seed)
            session.seed_manager.seed_all()

        try:
            result = session.run_simulation(
                num_iterations=num_iterations,
                verbose=verbose,
            )
            # Mark as baseline
            result.scenario_name = f"{result.scenario_name}_baseline"
        finally:
            # Restore original settings
            session.adaptation_mode = original_mode
            session.adaptation_strength = original_strength
            session.cluster_penalties = original_penalties

        return result

    @staticmethod
    def run_random_policy_baseline(
        session: PreferenceGuidedSession,
        num_iterations: int,
        num_seeds: int = 3,
        verbose: bool = False,
    ) -> List[SimulationResult]:
        """Run random policy baseline (epsilon=1.0) with multiple seeds.

        This establishes what performance would be with completely random
        action selection, providing a lower bound for comparison.

        Args:
            session: PreferenceGuidedSession instance.
            num_iterations: Number of iterations per run.
            num_seeds: Number of random seeds to use.
            verbose: Whether to print progress.

        Returns:
            List of SimulationResults from random policy.
        """
        results = []

        # Store original epsilon
        original_epsilon = (
            getattr(session.inference_config, "epsilon_greedy", 0.1)
            if session.inference_config
            else 0.1
        )

        # Store original adaptation settings
        original_mode = session.adaptation_mode
        original_strength = session.adaptation_strength

        # Disable adaptation for random baseline
        session.adaptation_mode = "none"
        session.adaptation_strength = 0.0

        try:
            for seed in range(num_seeds):
                # Reset with new seed
                session.seed_manager = SeedManager(seed)
                session.seed_manager.seed_all()

                # Set to fully random
                if session.inference_config:
                    session.inference_config.epsilon_greedy = 1.0

                result = session.run_simulation(
                    num_iterations=num_iterations,
                    verbose=verbose,
                )

                # Mark as random baseline
                result.scenario_name = f"{result.scenario_name}_random_baseline"
                result.seed = seed
                results.append(result)

        finally:
            # Restore original settings
            session.adaptation_mode = original_mode
            session.adaptation_strength = original_strength
            if session.inference_config:
                session.inference_config.epsilon_greedy = original_epsilon

        return results

    @staticmethod
    def run_multiple_baselines(
        session: PreferenceGuidedSession,
        num_iterations: int,
        num_seeds: int = 3,
        include_random: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Run multiple baseline types for comprehensive comparison.

        Args:
            session: PreferenceGuidedSession instance.
            num_iterations: Number of iterations per run.
            num_seeds: Number of random seeds to use.
            include_random: Whether to include random policy baseline.
            verbose: Whether to print progress.

        Returns:
            Dictionary with baseline type as key and list of results as value:
                - "no_adaptation": Results without adaptation
                - "random": Results with random policy (if include_random=True)
        """
        baselines = {}

        # No-adaptation baseline
        no_adapt_results = []
        for seed in range(num_seeds):
            result = BaselineRunner.run_no_adaptation_baseline(
                session=session,
                num_iterations=num_iterations,
                seed=seed,
                verbose=verbose,
            )
            result.seed = seed
            no_adapt_results.append(result)

        baselines["no_adaptation"] = no_adapt_results

        # Random policy baseline (optional)
        if include_random:
            baselines["random"] = BaselineRunner.run_random_policy_baseline(
                session=session,
                num_iterations=num_iterations,
                num_seeds=num_seeds,
                verbose=verbose,
            )

        logger.info(
            f"Completed {len(baselines)} baseline types with {num_seeds} seeds each"
        )

        return baselines


def compare_with_baseline(
    treatment_results: List[SimulationResult],
    baseline_results: List[SimulationResult],
    metric: str = "final_desirable_ratio",
) -> dict:
    """Compare treatment and baseline results statistically.

    Args:
        treatment_results: Results from adapted (treatment) runs.
        baseline_results: Results from baseline (control) runs.
        metric: Metric attribute name to compare (default: "final_desirable_ratio").

    Returns:
        Dictionary with comparison statistics:
            - treatment_mean: Mean of treatment group
            - baseline_mean: Mean of baseline group
            - difference: Treatment - baseline
            - treatment_ci: Confidence interval for treatment
            - baseline_ci: Confidence interval for baseline
            - statistical_test: Results from statistical test (if paired)

    Raises:
        ValueError: If results lists are empty.
    """
    from .metrics_calculator import StatisticalAnalyzer

    if not treatment_results or not baseline_results:
        raise ValueError("Both treatment and baseline results must be non-empty")

    # Extract metric values
    treatment_values = [getattr(r, metric) for r in treatment_results]
    baseline_values = [getattr(r, metric) for r in baseline_results]

    # Compute means
    treatment_mean = float(np.mean(treatment_values))
    baseline_mean = float(np.mean(baseline_values))
    difference = treatment_mean - baseline_mean

    # Compute confidence intervals
    treatment_ci = StatisticalAnalyzer.compute_confidence_interval(treatment_values)
    baseline_ci = StatisticalAnalyzer.compute_confidence_interval(baseline_values)

    # Perform statistical test if paired (same number of samples)
    statistical_test = None
    if len(treatment_values) == len(baseline_values):
        statistical_test = StatisticalAnalyzer.paired_comparison_test(
            treatment=treatment_values,
            control=baseline_values,
        )

    return {
        "metric": metric,
        "treatment_mean": treatment_mean,
        "baseline_mean": baseline_mean,
        "difference": difference,
        "improvement_percent": (
            100 * difference / (baseline_mean + 1e-10) if baseline_mean != 0 else 0.0
        ),
        "treatment_ci": treatment_ci,
        "baseline_ci": baseline_ci,
        "statistical_test": statistical_test,
    }
