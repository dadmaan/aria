"""Simulation runner for batch execution of preference-guided HIL simulations.

This module provides the SimulationRunner class for running multiple preference
scenarios with multiple seeds for statistical analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import gymnasium as gym

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .cluster_profiles import ClusterProfileLoader, load_default_profiles
from .config_loader import InferenceConfig
from .preference_simulation import (
    PreferenceScenario,
    get_predefined_scenarios,
    get_adapted_scenarios,
)
from .cluster_action_mapper import (
    ClusterActionMapping,
    create_mapping_from_environment,
)
from .preference_guided_session import (
    PreferenceGuidedSession,
    SimulationResult,
)

from src.training.tianshou_trainer import TianshouTrainer
from src.ghsom_manager import GHSOMManager
from src.environments.music_env_gym import (
    MusicGenerationGymEnv,
    NormalizedObservationWrapper,
    FeatureVectorObservationWrapper,
)
from src.agents.ghsom_perceiving_agent import GHSOMPerceivingAgent
from src.utils.config.config_loader import get_config
from src.agents.cluster_feature_mapper import ClusterFeatureMapper

logger = logging.getLogger(__name__)
console = Console()


class SimulationRunner:
    """Run multiple preference scenarios for comparison.

    This class manages batch execution of preference-guided simulations,
    supporting multiple scenarios, seeds, and ablation studies.

    Attributes:
        checkpoint_path: Path to trained model checkpoint.
        ghsom_dir: Directory containing GHSOM model files.
        cluster_profiles_path: Path to cluster profiles CSV.
        output_dir: Directory for saving results.
        config: Configuration dictionary.
        trainer: Loaded TianshouTrainer instance.
        env: Environment for sequence generation.
        ghsom_manager: GHSOM manager instance.
        cluster_loader: Cluster profile loader.
        scenarios: Dictionary of available preference scenarios.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        ghsom_dir: Path,
        cluster_profiles_path: Optional[Path] = None,
        output_dir: Path = Path("outputs/hil_simulation"),
        config: Optional[Dict[str, Any]] = None,
        inference_config: Optional[InferenceConfig] = None,
    ) -> None:
        """Initialize simulation runner.

        Args:
            checkpoint_path: Path to trained model checkpoint.
            ghsom_dir: Directory containing GHSOM model files.
            cluster_profiles_path: Path to cluster profiles CSV. If None, uses default.
            output_dir: Directory for saving results.
            config: Configuration dictionary. If None, loads from default.
            inference_config: Optional InferenceConfig for parameter management.

        Raises:
            FileNotFoundError: If checkpoint or GHSOM directory doesn't exist.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.ghsom_dir = Path(ghsom_dir)
        self.output_dir = Path(output_dir)
        self.config = config or get_config()
        self.inference_config = inference_config

        # Validate paths
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not self.ghsom_dir.exists():
            raise FileNotFoundError(f"GHSOM directory not found: {ghsom_dir}")

        # Load cluster profiles
        if cluster_profiles_path:
            self.cluster_loader = ClusterProfileLoader(cluster_profiles_path)
        else:
            self.cluster_loader = load_default_profiles()

        # Scenarios will be adapted after loading environment (in _load_components)
        # Load initial predefined scenarios (may be updated later)
        self.scenarios = get_predefined_scenarios(self.cluster_loader)

        # Initialize components (deferred loading)
        self.trainer: Optional[TianshouTrainer] = None
        self.env: Optional[gym.Env] = None
        self.ghsom_manager: Optional[GHSOMManager] = None
        self.cluster_mapping: Optional[ClusterActionMapping] = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized SimulationRunner: checkpoint=%s, ghsom=%s, output=%s",
            checkpoint_path,
            ghsom_dir,
            output_dir,
        )

    def _load_components(self) -> None:
        """Load trainer, environment, and GHSOM manager if not already loaded.

        This method carefully handles observation space configuration to ensure
        compatibility with the checkpoint. The observation space must match
        what the model was trained with (feature vectors vs normalized IDs).
        """
        if self.trainer is not None:
            return  # Already loaded

        logger.info("Loading model components...")

        # Load checkpoint to extract its config for observation space compatibility
        checkpoint_data = torch.load(self.checkpoint_path, map_location="cpu")
        checkpoint_config = checkpoint_data.get("config", {})

        # Merge configs: checkpoint config takes precedence for observation settings
        merged_config = self.config.copy()

        # Use checkpoint's observation settings to ensure architecture matches
        if "use_feature_observations" in checkpoint_config:
            merged_config["use_feature_observations"] = checkpoint_config[
                "use_feature_observations"
            ]
        if "feature_observation_mode" in checkpoint_config:
            merged_config["feature_observation_mode"] = checkpoint_config[
                "feature_observation_mode"
            ]
        if "feature_observation_source" in checkpoint_config:
            merged_config["feature_observation_source"] = checkpoint_config[
                "feature_observation_source"
            ]
        if "network" in checkpoint_config:
            merged_config["network"] = checkpoint_config["network"]

        use_feature_observations = merged_config.get("use_feature_observations", False)
        feature_mode = merged_config.get("feature_observation_mode", "centroid")
        feature_source = merged_config.get("feature_observation_source", "tsne")

        logger.info(
            f"Observation config: use_features={use_feature_observations}, "
            f"mode={feature_mode}, source={feature_source}"
        )

        # Get paths from config
        ghsom_path = Path(
            merged_config.get("ghsom", {}).get(
                "default_model_path",
                "experiments/ghsom_commu_full_tsne_optimized_20251125/ghsom_model.pkl",
            )
        )
        feature_path = Path(
            merged_config.get("features", {}).get(
                "artifact_path", "artifacts/features/tsne/commu_full_filtered_tsne"
            )
        )
        feature_type = merged_config.get("features", {}).get("type", feature_source)

        # Load GHSOM manager
        self.ghsom_manager = GHSOMManager.from_artifact(
            ghsom_model_path=ghsom_path,
            feature_artifact=feature_path,
            feature_type=feature_type,
        )

        # Create perceiving agent
        perceiving_agent = GHSOMPerceivingAgent(
            config=merged_config,
            ghsom_manager=self.ghsom_manager,
            features_dataset=self.ghsom_manager.train_data,
        )

        # Create environment
        sequence_length = merged_config.get("music", {}).get("sequence_length", 16)
        env = MusicGenerationGymEnv(
            perceiving_agent=perceiving_agent,
            sequence_length=sequence_length,
            config=merged_config,
        )

        # Apply appropriate observation wrapper based on checkpoint config
        if use_feature_observations:
            # Create feature mapper and wrap environment
            mapper = ClusterFeatureMapper(
                ghsom_manager=self.ghsom_manager,
                mode=feature_mode,
                feature_source=feature_type,
            )
            env = FeatureVectorObservationWrapper(env, mapper)
            # Note: Do NOT flatten - DRQN expects (seq_len, feature_dim) shape
            # The network extracts feature_dim from state_shape for embedding input
            logger.info(
                f"Using feature observations: shape={env.observation_space.shape}"
            )
        else:
            # Use normalized cluster ID observations
            env = NormalizedObservationWrapper(env)
            logger.info(
                f"Using normalized observations: shape={env.observation_space.shape}"
            )

        self.env = env

        # Load trainer using classmethod with merged config
        self.trainer = TianshouTrainer.load(
            path=str(self.checkpoint_path),
            env=self.env,
            config=merged_config,
        )

        # Create cluster mapping from environment
        self.cluster_mapping = create_mapping_from_environment(self.env)
        available_cluster_ids = list(self.cluster_mapping.valid_cluster_ids)

        # Adapt scenarios to use only available clusters
        self.scenarios = get_adapted_scenarios(
            self.cluster_loader,
            available_cluster_ids,
            self.inference_config,
            skip_invalid=True,
        )

        logger.info(
            "Model components loaded successfully. "
            "Action space: %d, Available scenarios: %d",
            self.cluster_mapping.action_space_size,
            len(self.scenarios),
        )

    def run_single_scenario(
        self,
        scenario_name: str,
        num_iterations: Optional[int] = None,
        num_seeds: Optional[int] = None,
        adaptation_mode: str = "q_penalty",
        adaptation_strength: Optional[float] = None,
        enable_policy_learning: bool = False,
        save_checkpoint: bool = False,
        verbose: bool = True,
    ) -> List[SimulationResult]:
        """Run single scenario with multiple seeds.

        Args:
            scenario_name: Name of the scenario to run.
            num_iterations: Iterations per simulation. If None, uses config default.
            num_seeds: Number of random seeds to use. If None, uses config default.
            adaptation_mode: Adaptation mode ("q_penalty" or "reward_shaping").
            adaptation_strength: Strength of adaptation. If None, uses config default.
            enable_policy_learning: If True, enable gradient-based policy updates.
            save_checkpoint: If True, save the adapted model checkpoint after each seed.
            verbose: Whether to print progress.

        Returns:
            List of SimulationResult objects (one per seed).

        Raises:
            ValueError: If scenario name is not found.
        """
        # Use config defaults if parameters not provided
        if num_iterations is None:
            num_iterations = (
                self.inference_config.get_num_iterations()
                if self.inference_config is not None
                else 50
            )
        if num_seeds is None:
            num_seeds = (
                self.inference_config.get_num_seeds()
                if self.inference_config is not None
                else 3
            )
        if adaptation_strength is None:
            adaptation_strength = (
                self.inference_config.get_adaptation_strength(adaptation_mode)
                if self.inference_config is not None
                else 5.0
            )
        if scenario_name not in self.scenarios:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. "
                f"Available: {list(self.scenarios.keys())}"
            )

        # Ensure components are loaded
        self._load_components()

        scenario = self.scenarios[scenario_name]
        results: List[SimulationResult] = []

        if verbose:
            console.print(f"\n[bold cyan]Running scenario: {scenario_name}[/bold cyan]")
            console.print(f"[dim]{scenario.description}[/dim]")

        for seed in range(num_seeds):
            if verbose:
                console.print(f"\n[dim]Seed {seed + 1}/{num_seeds}[/dim]")

            # Create session config with policy learning flag
            session_config = self.config.copy()
            session_config["enable_policy_learning"] = enable_policy_learning

            # Create session for this seed
            session = PreferenceGuidedSession(
                trainer=self.trainer,
                env=self.env,
                ghsom_manager=self.ghsom_manager,
                config=session_config,
                scenario=scenario,
                cluster_profiles=self.cluster_loader,
                adaptation_mode=adaptation_mode,
                adaptation_strength=adaptation_strength,
                seed=seed,
                inference_config=self.inference_config,
                cluster_mapping=self.cluster_mapping,
                enable_policy_learning=enable_policy_learning,
            )

            # Run simulation
            result = session.run_simulation(
                num_iterations=num_iterations,
                log_interval=10 if verbose else num_iterations,
                verbose=verbose,
            )

            results.append(result)

            # Save individual result
            result_dir = self.output_dir / "results" / scenario_name / f"seed_{seed}"
            result_dir.mkdir(parents=True, exist_ok=True)
            result.save(result_dir / "result.json")

            # Save checkpoint if requested
            if save_checkpoint:
                checkpoint_dir = self.output_dir / "checkpoints" / scenario_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"adapted_seed_{seed}"
                saved_path = session.save_checkpoint(checkpoint_path)
                if verbose:
                    console.print(f"[green]✓ Saved checkpoint: {saved_path}[/green]")

        return results

    def run_all_scenarios(
        self,
        num_iterations: Optional[int] = None,
        num_seeds: Optional[int] = None,
        adaptation_mode: str = "q_penalty",
        adaptation_strength: Optional[float] = None,
        scenarios: Optional[List[str]] = None,
        enable_policy_learning: bool = False,
        save_checkpoint: bool = False,
        verbose: bool = True,
    ) -> Dict[str, List[SimulationResult]]:
        """Run all or selected scenarios.

        Args:
            num_iterations: Iterations per simulation. If None, uses config default.
            num_seeds: Number of random seeds. If None, uses config default.
            adaptation_mode: Adaptation mode.
            adaptation_strength: Adaptation strength. If None, uses config default.
            scenarios: List of scenario names to run. If None, runs all.
            enable_policy_learning: If True, enable gradient-based policy updates.
            save_checkpoint: If True, save adapted checkpoints after each run.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping scenario names to lists of results.
        """
        # Use config defaults if parameters not provided
        if num_iterations is None:
            num_iterations = (
                self.inference_config.get_num_iterations()
                if self.inference_config is not None
                else 50
            )
        if num_seeds is None:
            num_seeds = (
                self.inference_config.get_num_seeds()
                if self.inference_config is not None
                else 3
            )
        if adaptation_strength is None:
            adaptation_strength = (
                self.inference_config.get_adaptation_strength(adaptation_mode)
                if self.inference_config is not None
                else 5.0
            )
        scenario_names = scenarios or list(self.scenarios.keys())
        all_results: Dict[str, List[SimulationResult]] = {}

        if verbose:
            console.print(
                f"\n[bold]Running {len(scenario_names)} scenarios "
                f"with {num_seeds} seeds each[/bold]"
            )

        for scenario_name in scenario_names:
            results = self.run_single_scenario(
                scenario_name=scenario_name,
                num_iterations=num_iterations,
                num_seeds=num_seeds,
                adaptation_mode=adaptation_mode,
                adaptation_strength=adaptation_strength,
                enable_policy_learning=enable_policy_learning,
                save_checkpoint=save_checkpoint,
                verbose=verbose,
            )
            all_results[scenario_name] = results

        # Save combined results
        self._save_combined_results(all_results, adaptation_mode, adaptation_strength)

        return all_results

    def run_ablation(
        self,
        scenario_name: str,
        adaptation_strengths: Optional[List[float]] = None,
        num_iterations: Optional[int] = None,
        num_seeds: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[float, List[SimulationResult]]:
        """Run ablation study on adaptation strength.

        Args:
            scenario_name: Scenario to run ablation on.
            adaptation_strengths: List of strength values to test. If None, uses config default.
            num_iterations: Iterations per simulation. If None, uses config default.
            num_seeds: Seeds per strength value. If None, uses config default.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping strength values to result lists.
        """
        # Use config defaults if parameters not provided
        if num_iterations is None:
            num_iterations = (
                self.inference_config.get_num_iterations()
                if self.inference_config is not None
                else 50
            )
        if num_seeds is None:
            num_seeds = (
                self.inference_config.get_num_seeds()
                if self.inference_config is not None
                else 3
            )
        if adaptation_strengths is None:
            adaptation_strengths = (
                self.inference_config.get_ablation_strengths()
                if self.inference_config is not None
                else [1.0, 2.5, 5.0, 7.5, 10.0]
            )

        ablation_results: Dict[float, List[SimulationResult]] = {}

        if verbose:
            console.print(f"\n[bold]Running ablation study on '{scenario_name}'[/bold]")
            console.print(f"[dim]Testing strengths: {adaptation_strengths}[/dim]")

        for strength in adaptation_strengths:
            if verbose:
                console.print(f"\n[cyan]Strength: {strength}[/cyan]")

            results = self.run_single_scenario(
                scenario_name=scenario_name,
                num_iterations=num_iterations,
                num_seeds=num_seeds,
                adaptation_mode="q_penalty",
                adaptation_strength=strength,
                verbose=verbose,
            )
            ablation_results[strength] = results

        # Save ablation results
        self._save_ablation_results(scenario_name, ablation_results)

        return ablation_results

    def run_mode_comparison(
        self,
        scenario_name: str,
        num_iterations: Optional[int] = None,
        num_seeds: Optional[int] = None,
        adaptation_strength: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, List[SimulationResult]]:
        """Compare q_penalty vs reward_shaping modes.

        Args:
            scenario_name: Scenario to compare modes on.
            num_iterations: Iterations per simulation. If None, uses config default.
            num_seeds: Seeds per mode. If None, uses config default.
            adaptation_strength: Adaptation strength. If None, uses config default.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping mode names to result lists.
        """
        # Use config defaults if parameters not provided
        if num_iterations is None:
            num_iterations = (
                self.inference_config.get_num_iterations()
                if self.inference_config is not None
                else 50
            )
        if num_seeds is None:
            num_seeds = (
                self.inference_config.get_num_seeds()
                if self.inference_config is not None
                else 3
            )
        if adaptation_strength is None:
            # For mode comparison, use q_penalty strength as default
            adaptation_strength = (
                self.inference_config.get_adaptation_strength("q_penalty")
                if self.inference_config is not None
                else 5.0
            )
        modes = ["q_penalty", "reward_shaping"]
        mode_results: Dict[str, List[SimulationResult]] = {}

        if verbose:
            console.print(
                f"\n[bold]Comparing adaptation modes on '{scenario_name}'[/bold]"
            )

        for mode in modes:
            if verbose:
                console.print(f"\n[cyan]Mode: {mode}[/cyan]")

            results = self.run_single_scenario(
                scenario_name=scenario_name,
                num_iterations=num_iterations,
                num_seeds=num_seeds,
                adaptation_mode=mode,
                adaptation_strength=adaptation_strength,
                verbose=verbose,
            )
            mode_results[mode] = results

        # Save comparison results
        self._save_mode_comparison_results(scenario_name, mode_results)

        return mode_results

    def _save_combined_results(
        self,
        all_results: Dict[str, List[SimulationResult]],
        adaptation_mode: str,
        adaptation_strength: float,
    ) -> None:
        """Save combined results summary.

        Args:
            all_results: All scenario results.
            adaptation_mode: Mode used.
            adaptation_strength: Strength used.
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "adaptation_mode": adaptation_mode,
            "adaptation_strength": adaptation_strength,
            "scenarios": {},
        }

        for scenario_name, results in all_results.items():
            # Aggregate statistics across seeds
            desirable_improvements = [
                r.final_desirable_ratio - r.initial_desirable_ratio for r in results
            ]
            undesirable_reductions = [
                r.initial_undesirable_ratio - r.final_undesirable_ratio for r in results
            ]
            feedback_improvements = [r.feedback_improvement for r in results]

            summary["scenarios"][scenario_name] = {
                "num_seeds": len(results),
                "desirable_improvement_mean": float(np.mean(desirable_improvements)),
                "desirable_improvement_std": float(np.std(desirable_improvements)),
                "undesirable_reduction_mean": float(np.mean(undesirable_reductions)),
                "undesirable_reduction_std": float(np.std(undesirable_reductions)),
                "feedback_improvement_mean": float(np.mean(feedback_improvements)),
                "feedback_improvement_std": float(np.std(feedback_improvements)),
            }

        # Save summary
        summary_path = self.output_dir / "combined_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved combined summary to %s", summary_path)

    def _save_ablation_results(
        self,
        scenario_name: str,
        ablation_results: Dict[float, List[SimulationResult]],
    ) -> None:
        """Save ablation study results.

        Args:
            scenario_name: Scenario name.
            ablation_results: Results by strength.
        """
        ablation_dir = self.output_dir / "ablation" / scenario_name
        ablation_dir.mkdir(parents=True, exist_ok=True)

        summary = {"scenario": scenario_name, "strengths": {}}

        for strength, results in ablation_results.items():
            improvements = [
                r.final_desirable_ratio - r.initial_desirable_ratio for r in results
            ]
            summary["strengths"][str(strength)] = {
                "mean_improvement": float(np.mean(improvements)),
                "std_improvement": float(np.std(improvements)),
            }

            # Save individual results
            for i, result in enumerate(results):
                result.save(ablation_dir / f"strength_{strength}_seed_{i}.json")

        # Save ablation summary
        with open(ablation_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _save_mode_comparison_results(
        self,
        scenario_name: str,
        mode_results: Dict[str, List[SimulationResult]],
    ) -> None:
        """Save mode comparison results.

        Args:
            scenario_name: Scenario name.
            mode_results: Results by mode.
        """
        compare_dir = self.output_dir / "mode_comparison" / scenario_name
        compare_dir.mkdir(parents=True, exist_ok=True)

        summary = {"scenario": scenario_name, "modes": {}}

        for mode, results in mode_results.items():
            improvements = [
                r.final_desirable_ratio - r.initial_desirable_ratio for r in results
            ]
            summary["modes"][mode] = {
                "mean_improvement": float(np.mean(improvements)),
                "std_improvement": float(np.std(improvements)),
            }

            # Save individual results
            for i, result in enumerate(results):
                result.save(compare_dir / f"{mode}_seed_{i}.json")

        # Save comparison summary
        with open(compare_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def print_results_summary(
        self,
        results: Dict[str, List[SimulationResult]],
    ) -> None:
        """Print formatted summary of results.

        Args:
            results: Dictionary of scenario results.
        """
        table = Table(title="Simulation Results Summary", show_header=True)
        table.add_column("Scenario", style="cyan")
        table.add_column("Desirable Δ", style="green")
        table.add_column("Undesirable Δ", style="red")
        table.add_column("Feedback Δ", style="yellow")
        table.add_column("Convergence", style="magenta")

        for scenario_name, scenario_results in results.items():
            # Aggregate across seeds
            des_changes = [
                r.final_desirable_ratio - r.initial_desirable_ratio
                for r in scenario_results
            ]
            und_changes = [
                r.initial_undesirable_ratio - r.final_undesirable_ratio
                for r in scenario_results
            ]
            fb_changes = [r.feedback_improvement for r in scenario_results]
            convergences = [
                r.convergence_iteration
                for r in scenario_results
                if r.convergence_iteration
            ]

            table.add_row(
                scenario_name,
                f"{np.mean(des_changes):+.3f}±{np.std(des_changes):.3f}",
                f"{np.mean(und_changes):+.3f}±{np.std(und_changes):.3f}",
                f"{np.mean(fb_changes):+.2f}±{np.std(fb_changes):.2f}",
                f"{np.mean(convergences):.0f}" if convergences else "N/A",
            )

        console.print(table)

    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenario names.

        Returns:
            List of scenario names.
        """
        return list(self.scenarios.keys())
