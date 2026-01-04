"""Data loading utilities for post-training analysis.

Provides a unified interface to load training metrics, reward components,
episode sequences, and GHSOM hierarchy information from any training run.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import warnings

import numpy as np
import pandas as pd
import yaml

from src.ghsom_manager import GHSOMManager


@dataclass
class TrainingRunData:
    """Complete data bundle for a single training run."""

    # Identifiers
    run_path: Path
    run_name: str

    # Core metrics
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    episode_sequences: List[List[int]]  # Cluster sequences per episode

    # Reward components
    reward_components: Dict[str, np.ndarray]  # {component_name: [episode_values]}

    # GHSOM data
    ghsom_manager: Optional[GHSOMManager]
    total_clusters: int
    hierarchy_stats: Dict[str, Any]

    # Metadata
    config: Dict[str, Any]
    total_episodes: int
    total_timesteps: int

    # Optional
    comprehensive_metrics: Optional[Dict[str, Any]] = None


class TrainingDataLoader:
    """Load and validate data from training run directories."""

    def __init__(self, run_path: str | Path):
        """Initialize loader for a specific training run.

        Args:
            run_path: Path to training run directory
                     (e.g., artifacts/training/run_drqn_20251207_200443)
        """
        self.run_path = Path(run_path)
        self.run_name = self.run_path.name

        if not self.run_path.exists():
            raise ValueError(f"Training run not found: {self.run_path}")

    def load(
        self,
        load_ghsom: bool = True,
        load_comprehensive: bool = False
    ) -> TrainingRunData:
        """Load all data from the training run.

        Args:
            load_ghsom: Whether to load GHSOM manager (slower but needed for hierarchy viz)
            load_comprehensive: Whether to load comprehensive metrics (large files)

        Returns:
            TrainingRunData with all loaded information
        """
        # 1. Load config
        config = self._load_config()

        # 2. Load reward components (primary source for episode rewards)
        reward_data = self._load_reward_components()

        # 3. Load episode sequences
        episode_sequences = self._load_episode_sequences()

        # 4. Load GHSOM if requested
        ghsom_manager = None
        hierarchy_stats = {}
        total_clusters = 0
        if load_ghsom:
            ghsom_manager = self._load_ghsom_manager(config)
            hierarchy_stats = ghsom_manager.stats
            total_clusters = hierarchy_stats.get('total_nodes', 0)

        # 5. Optional: comprehensive metrics
        comprehensive_metrics = None
        if load_comprehensive:
            comprehensive_metrics = self._load_comprehensive_metrics()

        return TrainingRunData(
            run_path=self.run_path,
            run_name=self.run_name,
            episode_rewards=np.array(reward_data['episode_rewards']),
            episode_lengths=reward_data['episode_lengths'],
            episode_sequences=episode_sequences,
            reward_components=reward_data['components'],
            ghsom_manager=ghsom_manager,
            total_clusters=total_clusters,
            hierarchy_stats=hierarchy_stats,
            config=config,
            total_episodes=len(reward_data['episode_rewards']),
            total_timesteps=reward_data['total_steps'],
            comprehensive_metrics=comprehensive_metrics,
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load config.yaml from run directory."""
        config_path = self.run_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_reward_components(self) -> Dict[str, Any]:
        """Load reward components and episode rewards.

        PRIORITY ORDER for episode_rewards (to avoid phantom episode bug):
        1. training_metrics.json - most reliable source
        2. final_history.json - fallback (may have phantom episodes for non-CL runs)

        Reward components always come from final_history.json.
        """
        # Try training_metrics.json first for episode rewards (more reliable)
        training_metrics_path = self.run_path / "metrics" / "training_metrics.json"
        episode_rewards = []
        episode_lengths = []
        total_steps = 0

        if training_metrics_path.exists():
            with open(training_metrics_path, 'r') as f:
                tm_data = json.load(f)
            episode_rewards = tm_data.get('episode_rewards', [])
            episode_lengths = np.array(tm_data.get('episode_lengths', []))
            # Calculate total steps from episode lengths
            if len(episode_lengths) > 0:
                total_steps = int(np.sum(episode_lengths))

        # Load reward components from final_history.json
        rc_path = self.run_path / "metrics" / "reward_components" / "final_history.json"
        if not rc_path.exists():
            raise FileNotFoundError(f"Reward components not found: {rc_path}")

        with open(rc_path, 'r') as f:
            data = json.load(f)

        # Extract component arrays
        components = {}
        if 'episode_component_rewards' in data:
            # Convert list of dicts to dict of arrays
            comp_list = data['episode_component_rewards']
            if comp_list:
                for key in comp_list[0].keys():
                    components[key] = np.array([ep.get(key, 0) for ep in comp_list])

        # Fallback to final_history.json for episode rewards if training_metrics not available
        if not episode_rewards:
            episode_rewards = data.get('episode_rewards', [])
            total_steps = data.get('total_steps', 0)
            total_eps = data.get('total_episodes', 0)
            avg_length = total_steps / max(total_eps, 1)
            episode_lengths = np.full(total_eps, avg_length)
        else:
            # Validate: Check for phantom episodes bug
            # If final_history has more episodes than training_metrics, it's buggy
            fh_episodes = len(data.get('episode_rewards', []))
            tm_episodes = len(episode_rewards)
            if fh_episodes > tm_episodes * 1.1:  # More than 10% extra episodes
                warnings.warn(
                    f"Detected phantom episodes bug in final_history.json: "
                    f"{fh_episodes} vs {tm_episodes} in training_metrics.json. "
                    f"Using training_metrics.json data."
                )
            # Truncate component arrays to match training_metrics episode count
            for key in components:
                if len(components[key]) > tm_episodes:
                    components[key] = components[key][:tm_episodes]

        # If we still don't have episode_lengths, infer from total_steps
        if len(episode_lengths) == 0:
            total_eps = len(episode_rewards)
            if total_steps == 0:
                total_steps = data.get('total_steps', 0)
            avg_length = total_steps / max(total_eps, 1)
            episode_lengths = np.full(total_eps, avg_length)

        return {
            'episode_rewards': episode_rewards,
            'components': components,
            'total_steps': total_steps,
            'episode_lengths': episode_lengths,
        }

    def _load_episode_sequences(self) -> List[List[int]]:
        """Load episode sequences from comprehensive metrics."""
        seq_path = self.run_path / "metrics" / "comprehensive" / "episode_sequences.json"

        if not seq_path.exists():
            # Fallback: return empty list with warning
            warnings.warn(
                f"Episode sequences not found at {seq_path}. "
                "Visualizations requiring sequences will be unavailable. "
                "To fix: re-run training with updated comprehensive_metrics_callback."
            )
            return []

        with open(seq_path, 'r') as f:
            data = json.load(f)

        return data.get('sequences', [])

    def _load_ghsom_manager(self, config: Dict[str, Any]) -> GHSOMManager:
        """Load GHSOM manager from config paths."""
        ghsom_path = Path(config['ghsom']['default_model_path'])
        features_path = Path(config['features']['artifact_path'])

        if not ghsom_path.exists():
            raise FileNotFoundError(f"GHSOM model not found: {ghsom_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Features artifact not found: {features_path}")

        return GHSOMManager.from_artifact(
            ghsom_model_path=ghsom_path,
            feature_artifact=features_path,
            feature_type=config['features']['type'],
        )

    def _load_comprehensive_metrics(self) -> Dict[str, Any]:
        """Load comprehensive metrics files."""
        comp_dir = self.run_path / "metrics" / "comprehensive"

        metrics = {}
        for metric_file in ['episodes_metrics.json', 'system_metrics.json',
                           'q_values_metrics.json', 'gradients_metrics.json']:
            path = comp_dir / metric_file
            if path.exists():
                with open(path, 'r') as f:
                    metrics[metric_file.replace('.json', '')] = json.load(f)

        return metrics


def load_training_run(run_path: str | Path, **kwargs) -> TrainingRunData:
    """Convenience function to load a training run.

    Args:
        run_path: Path to training run directory
        **kwargs: Passed to TrainingDataLoader.load()

    Returns:
        TrainingRunData object

    Example:
        >>> data = load_training_run('artifacts/training/run_drqn_20251207_200443')
        >>> print(f"Episodes: {data.total_episodes}, Clusters: {data.total_clusters}")
    """
    loader = TrainingDataLoader(run_path)
    return loader.load(**kwargs)
