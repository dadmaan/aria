#!/usr/bin/env python3
"""WandB sweep agent for GHSOM hyperparameter tuning.

This script is called by the WandB sweep controller for each trial. It:
1. Receives hyperparameters from wandb.config
2. Trains a GHSOM model with those hyperparameters
3. Logs all metrics to WandB including a composite score for optimization

Usage:
    # Called automatically by WandB sweep controller
    wandb agent <entity>/<project>/<sweep_id>

    # Or run directly with default/test parameters (for debugging)
    python scripts/ghsom/run_ghsom_sweep_agent.py --dry-run

    # Run with custom reduced artifact path
    python scripts/ghsom/run_ghsom_sweep_agent.py \
        --reduced-artifact artifacts/preprocessing/umap/reduced/reduced/embedding.npy

Environment Variables:
    WANDB_PROJECT: WandB project name (default: aria-ghsom-sweep)
    WANDB_ENTITY: WandB entity/team name (optional)
    GHSOM_SWEEP_REDUCED_ARTIFACT: Path to reduced features artifact

Example Logged Metrics:
    - num_clusters: Number of leaf clusters (action space size)
    - ghsom_depth: Maximum hierarchy depth
    - mean_activation: Data fit quality (lower is better)
    - dispersion_rate: Neuron utilization (closer to 1.0 is better)
    - composite_score: Weighted combination for Bayesian optimization
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ghsom.training import (  # noqa: E402
    GHSOMTrainingConfig,
    load_training_dataset,
    train_ghsom_model,
)
from src.utils.features.feature_loader import FeatureType  # noqa: E402

# =============================================================================
# CONSTANTS
# =============================================================================

# Default paths
DEFAULT_REDUCED_ARTIFACT = "artifacts/preprocessing/umap/reduced/reduced/embedding.npy"
DEFAULT_PROJECT = "aria-ghsom-sweep"
DEFAULT_FEATURE_TYPE = "umap"

# Default hyperparameters (used in dry-run mode)
DEFAULT_HYPERPARAMS = {
    "t1": 0.35,
    "t2": 0.05,
    "learning_rate": 0.1,
    "decay": 0.99,
    "gaussian_sigma": 1.0,
    "epochs": 30,
    "grow_maxiter": 25,
}


# =============================================================================
# SCORING CONFIGURATION
# =============================================================================


@dataclass
class ScoringConfig:
    """Configuration for composite score computation.

    Attributes:
        weight_cluster: Weight for target cluster count adherence.
        weight_activation: Weight for data fit quality.
        weight_dispersion: Weight for neuron utilization.
        target_cluster_min: Minimum acceptable cluster count.
        target_cluster_max: Maximum acceptable cluster count.
        activation_min: Minimum activation bound (perfect fit).
        activation_max: Maximum activation bound (poor fit).
    """

    weight_cluster: float = 0.40
    weight_activation: float = 0.35
    weight_dispersion: float = 0.25
    target_cluster_min: int = 50
    target_cluster_max: int = 200
    activation_min: float = 0.0
    activation_max: float = 2.0

    # Computed properties
    _target_cluster_sigma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and compute derived values."""
        # Normalize weights to sum to 1.0
        total_weight = (
            self.weight_cluster + self.weight_activation + self.weight_dispersion
        )
        if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
            self.weight_cluster /= total_weight
            self.weight_activation /= total_weight
            self.weight_dispersion /= total_weight

        # Compute sigma for Gaussian decay
        self._target_cluster_sigma = (
            self.target_cluster_max - self.target_cluster_min
        ) / 2

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ScoringConfig":
        """Create ScoringConfig from a nested dictionary.

        Expected structure (from YAML):
            scoring:
              weights:
                cluster_score: 0.40
                activation: 0.35
                dispersion: 0.25
              target_cluster_range:
                min: 50
                max: 200
              activation_bounds:
                min: 0.0
                max: 2.0

        Args:
            config: Dictionary with scoring configuration.

        Returns:
            ScoringConfig instance.
        """
        weights = config.get("weights", {})
        target_range = config.get("target_cluster_range", {})
        activation_bounds = config.get("activation_bounds", {})

        return cls(
            weight_cluster=float(weights.get("cluster_score", 0.40)),
            weight_activation=float(weights.get("activation", 0.35)),
            weight_dispersion=float(weights.get("dispersion", 0.25)),
            target_cluster_min=int(target_range.get("min", 50)),
            target_cluster_max=int(target_range.get("max", 200)),
            activation_min=float(activation_bounds.get("min", 0.0)),
            activation_max=float(activation_bounds.get("max", 2.0)),
        )

    @classmethod
    def default(cls) -> "ScoringConfig":
        """Create default ScoringConfig."""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "weights": {
                "cluster_score": self.weight_cluster,
                "activation": self.weight_activation,
                "dispersion": self.weight_dispersion,
            },
            "target_cluster_range": {
                "min": self.target_cluster_min,
                "max": self.target_cluster_max,
            },
            "activation_bounds": {
                "min": self.activation_min,
                "max": self.activation_max,
            },
        }


# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ghsom_sweep_agent")


# =============================================================================
# COMPOSITE SCORE COMPUTATION
# =============================================================================


def compute_cluster_score(num_clusters: int, scoring_config: ScoringConfig) -> float:
    """Compute score based on cluster count proximity to target range.

    Uses a bell curve (Gaussian) centered on the target range. Clusters within
    the target range receive a score of 1.0. Clusters outside the range receive
    decreasing scores based on distance from the range.

    Args:
        num_clusters: Number of clusters from GHSOM training.
        scoring_config: Scoring configuration with target range.

    Returns:
        Score in [0, 1] where 1.0 means within target range.
    """
    target_min = scoring_config.target_cluster_min
    target_max = scoring_config.target_cluster_max

    if target_min <= num_clusters <= target_max:
        return 1.0

    # Distance from nearest edge of target range
    if num_clusters < target_min:
        distance = target_min - num_clusters
    else:
        distance = num_clusters - target_max

    # Gaussian decay based on distance
    # sigma controls how quickly score drops off outside the range
    sigma = scoring_config._target_cluster_sigma * 0.5  # Steeper falloff
    if sigma <= 0:
        return 0.0
    score = math.exp(-((distance / sigma) ** 2))
    return max(0.0, min(1.0, score))


def normalize_activation(
    mean_activation: float, scoring_config: ScoringConfig
) -> float:
    """Normalize mean_activation to [0, 1] where lower activation is better.

    Args:
        mean_activation: Raw mean activation from GHSOM metrics.
        scoring_config: Scoring configuration with activation bounds.

    Returns:
        Normalized score in [0, 1] where 1.0 means best (lowest activation).
    """
    act_min = scoring_config.activation_min
    act_max = scoring_config.activation_max

    # Clamp to expected range
    clamped = max(act_min, min(act_max, mean_activation))

    # Invert so lower activation = higher score
    range_size = act_max - act_min
    if range_size <= 0:
        return 1.0 if clamped <= act_min else 0.0

    normalized = 1.0 - (clamped - act_min) / range_size
    return max(0.0, min(1.0, normalized))


def compute_composite_score(
    metrics: Dict[str, float], scoring_config: ScoringConfig
) -> Dict[str, float]:
    """Compute composite score and component scores for multi-objective optimization.

    The composite score combines:
    - cluster_score: Proximity to target cluster count range
    - activation_score: Data fit quality (normalized, inverted mean_activation)
    - dispersion_score: Neuron utilization efficiency

    Args:
        metrics: GHSOM training metrics dictionary.
        scoring_config: Scoring configuration with weights and bounds.

    Returns:
        Dictionary with composite_score and individual component scores.
    """
    num_clusters = int(metrics.get("num_clusters", 0))
    mean_activation = float(metrics.get("mean_activation", 1.0))
    dispersion_rate = float(metrics.get("dispersion_rate", 0.0))

    # Compute component scores
    cluster_score = compute_cluster_score(num_clusters, scoring_config)
    activation_score = normalize_activation(mean_activation, scoring_config)
    # dispersion_rate is already in [0, 1], closer to 1 is better
    dispersion_score = max(0.0, min(1.0, dispersion_rate))

    # Weighted composite
    composite_score = (
        scoring_config.weight_cluster * cluster_score
        + scoring_config.weight_activation * activation_score
        + scoring_config.weight_dispersion * dispersion_score
    )

    return {
        "composite_score": composite_score,
        "cluster_score": cluster_score,
        "activation_score": activation_score,
        "dispersion_score": dispersion_score,
    }


# =============================================================================
# SWEEP AGENT LOGIC
# =============================================================================


def run_ghsom_trial(
    hyperparams: Dict[str, Any],
    reduced_artifact_path: Path,
    scoring_config: ScoringConfig,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    seed: int = 42,
    n_workers: int = -1,
    output_dir: Optional[Path] = None,
    trial_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single GHSOM training trial with given hyperparameters.

    Args:
        hyperparams: Dictionary of GHSOM hyperparameters.
        reduced_artifact_path: Path to reduced features artifact.
        scoring_config: Configuration for composite score computation.
        feature_type: Type of features ("umap", "tsne", "pca", "raw").
        seed: Random seed for reproducibility.
        n_workers: Number of workers for parallel training (-1 = all CPUs).
        output_dir: Optional directory to save trial results locally.
        trial_id: Optional trial identifier for local file naming.

    Returns:
        Dictionary containing:
        - metrics: All GHSOM training metrics
        - scores: Composite and component scores
        - config: Training configuration used
        - scoring_config: Scoring configuration used
        - success: Whether training succeeded
        - error: Error message if training failed
        - trial_id: Trial identifier (if provided or generated)
    """
    # Generate trial_id if not provided
    if trial_id is None:
        trial_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    result: Dict[str, Any] = {
        "trial_id": trial_id,
        "metrics": {},
        "scores": {},
        "config": {},
        "scoring_config": scoring_config.to_dict(),
        "success": False,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Build training config from hyperparams
        config = GHSOMTrainingConfig(
            t1=float(hyperparams.get("t1", DEFAULT_HYPERPARAMS["t1"])),
            t2=float(hyperparams.get("t2", DEFAULT_HYPERPARAMS["t2"])),
            learning_rate=float(
                hyperparams.get("learning_rate", DEFAULT_HYPERPARAMS["learning_rate"])
            ),
            decay=float(hyperparams.get("decay", DEFAULT_HYPERPARAMS["decay"])),
            gaussian_sigma=float(
                hyperparams.get("gaussian_sigma", DEFAULT_HYPERPARAMS["gaussian_sigma"])
            ),
            epochs=int(hyperparams.get("epochs", DEFAULT_HYPERPARAMS["epochs"])),
            grow_maxiter=int(
                hyperparams.get("grow_maxiter", DEFAULT_HYPERPARAMS["grow_maxiter"])
            ),
            seed=seed,
        )
        result["config"] = {
            "t1": config.t1,
            "t2": config.t2,
            "learning_rate": config.learning_rate,
            "decay": config.decay,
            "gaussian_sigma": config.gaussian_sigma,
            "epochs": config.epochs,
            "grow_maxiter": config.grow_maxiter,
            "seed": config.seed,
        }

        logger.info(f"Loading features from: {reduced_artifact_path}")
        features_df, metadata_df, artifact_metadata = load_training_dataset(
            reduced_artifact_path,
            cast(FeatureType, feature_type),
            metadata_columns=None,
        )

        data = features_df.to_numpy(dtype=np.float64)
        logger.info(f"Training GHSOM on {data.shape[0]} samples, {data.shape[1]} dims")
        logger.info(f"Config: t1={config.t1}, t2={config.t2}, epochs={config.epochs}")

        # Train GHSOM
        training_result = train_ghsom_model(data, config, n_workers=n_workers)

        # Extract metrics
        result["metrics"] = training_result.metrics
        result["scores"] = compute_composite_score(
            training_result.metrics, scoring_config
        )
        result["success"] = True

        logger.info(
            f"Training complete: {int(result['metrics']['num_clusters'])} clusters, "
            f"depth={int(result['metrics']['ghsom_depth'])}, "
            f"composite_score={result['scores']['composite_score']:.4f}"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        result["error"] = str(e)
        result["success"] = False
        # Set default scores for failed runs (worst case)
        result["scores"] = {
            "composite_score": 0.0,
            "cluster_score": 0.0,
            "activation_score": 0.0,
            "dispersion_score": 0.0,
        }

    # Save trial result locally if output_dir is provided
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            trial_file = output_dir / f"trial_{result['trial_id']}.json"
            with trial_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved trial result to: {trial_file}")
        except Exception as save_error:
            logger.warning(f"Failed to save trial result locally: {save_error}")

    return result


def run_sweep_agent(
    reduced_artifact_path: Path,
    scoring_config: ScoringConfig,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    project: str = DEFAULT_PROJECT,
    entity: Optional[str] = None,
    dry_run: bool = False,
    seed: int = 42,
    n_workers: int = -1,
    output_dir: Optional[Path] = None,
) -> int:
    """Run the WandB sweep agent.

    This function can be called in two ways:
    1. Via wandb.agent() - run is already initialized, use existing wandb.run
    2. Standalone CLI - needs to initialize a new WandB run

    It:
    1. Gets/initializes a WandB run (receives hyperparams via wandb.config)
    2. Runs GHSOM training with those hyperparams
    3. Logs all metrics to WandB
    4. Optionally saves results locally

    Args:
        reduced_artifact_path: Path to reduced features artifact.
        scoring_config: Configuration for composite score computation.
        feature_type: Type of features to load.
        project: WandB project name.
        entity: WandB entity/team name (optional).
        dry_run: If True, skip WandB and use default hyperparams for testing.
        seed: Random seed for reproducibility.
        n_workers: Number of workers for parallel training.
        output_dir: Optional directory to save trial results locally.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        import wandb
    except ImportError:
        logger.error("WandB is not installed. Please install with: pip install wandb")
        return 1

    # Validate artifact path
    if not reduced_artifact_path.exists():
        logger.error(f"Reduced artifact not found: {reduced_artifact_path}")
        return 1

    # Log scoring configuration
    logger.info(
        f"Scoring weights: cluster={scoring_config.weight_cluster:.2f}, "
        f"activation={scoring_config.weight_activation:.2f}, "
        f"dispersion={scoring_config.weight_dispersion:.2f}"
    )
    logger.info(
        f"Target cluster range: [{scoring_config.target_cluster_min}, "
        f"{scoring_config.target_cluster_max}]"
    )

    if dry_run:
        # Dry run mode: test without WandB
        logger.info("=== DRY RUN MODE ===")
        logger.info("Using default hyperparameters (no WandB)")
        hyperparams = DEFAULT_HYPERPARAMS.copy()

        result = run_ghsom_trial(
            hyperparams=hyperparams,
            reduced_artifact_path=reduced_artifact_path,
            scoring_config=scoring_config,
            feature_type=feature_type,
            seed=seed,
            n_workers=n_workers,
            output_dir=output_dir,
        )

        if result["success"]:
            logger.info("=== DRY RUN RESULTS ===")
            logger.info(f"Metrics: {result['metrics']}")
            logger.info(f"Scores: {result['scores']}")
            return 0
        else:
            logger.error(f"Dry run failed: {result['error']}")
            return 1

    # Check if we're already inside a wandb.agent() context
    # If wandb.run exists and is active, we're being called from wandb.agent()
    # and should NOT reinitialize - just use the existing run
    run_was_preinitialized = wandb.run is not None

    if run_was_preinitialized:
        logger.info("Using existing WandB run (called from wandb.agent)")
        run = wandb.run
    else:
        # Standalone execution - initialize a new run
        logger.info(f"Initializing new WandB run for project: {project}")
        run = wandb.init(project=project, entity=entity)

    if run is None:
        logger.error("Failed to get/initialize WandB run")
        return 1

    # Use run ID as trial ID for local saving
    trial_id = run.id if run.id else datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        # Get hyperparameters from WandB config
        hyperparams = dict(wandb.config)
        logger.info(f"Received hyperparameters: {hyperparams}")

        # Log artifact path and scoring config (only if not already set)
        wandb.config.update(
            {
                "reduced_artifact_path": str(reduced_artifact_path),
                "feature_type": feature_type,
                "seed": seed,
                "scoring_config": scoring_config.to_dict(),
            },
            allow_val_change=True,
        )

        # Run training
        result = run_ghsom_trial(
            hyperparams=hyperparams,
            reduced_artifact_path=reduced_artifact_path,
            scoring_config=scoring_config,
            feature_type=feature_type,
            seed=seed,
            n_workers=n_workers,
            output_dir=output_dir,
            trial_id=trial_id,
        )

        # Log all metrics to WandB
        log_data: Dict[str, Any] = {}

        # Log GHSOM metrics
        for key, value in result["metrics"].items():
            log_data[f"ghsom/{key}"] = value

        # Log composite scores
        for key, value in result["scores"].items():
            log_data[key] = value

        # Log training status
        log_data["training_success"] = 1.0 if result["success"] else 0.0

        if result["error"]:
            log_data["error"] = result["error"]

        wandb.log(log_data)

        # Log summary metrics
        wandb.summary.update(
            {
                "composite_score": result["scores"]["composite_score"],
                "num_clusters": result["metrics"].get("num_clusters", 0),
                "ghsom_depth": result["metrics"].get("ghsom_depth", 0),
                "mean_activation": result["metrics"].get("mean_activation", 0),
                "dispersion_rate": result["metrics"].get("dispersion_rate", 0),
                "training_success": result["success"],
            }
        )

        return 0 if result["success"] else 1

    except Exception as e:
        logger.error(f"Sweep agent failed: {e}")
        wandb.log({"error": str(e), "training_success": 0.0, "composite_score": 0.0})
        return 1

    finally:
        # Only finish the run if we initialized it ourselves
        # If called from wandb.agent(), the agent manages the run lifecycle
        if not run_was_preinitialized:
            wandb.finish()


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WandB sweep agent for GHSOM hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--reduced-artifact",
        type=Path,
        default=None,
        help=(
            f"Path to reduced features artifact "
            f"(default: {DEFAULT_REDUCED_ARTIFACT} or GHSOM_SWEEP_REDUCED_ARTIFACT env var)"
        ),
    )

    parser.add_argument(
        "--feature-type",
        type=str,
        default=DEFAULT_FEATURE_TYPE,
        choices=["umap", "tsne", "pca", "raw"],
        help=f"Type of features to load (default: {DEFAULT_FEATURE_TYPE})",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help=f"WandB project name (default: {DEFAULT_PROJECT} or WANDB_PROJECT env var)",
    )

    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity/team name (default: WANDB_ENTITY env var)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of workers for parallel GHSOM training (-1 = all CPUs)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without WandB using default hyperparameters (for testing)",
    )

    return parser.parse_args()


def _load_sweep_config() -> Dict[str, Any]:
    """Load sweep configuration from YAML file.

    Returns:
        Sweep configuration dictionary, or empty dict if loading fails.
    """
    try:
        import yaml

        config_path = PROJECT_ROOT / "configs" / "sweep" / "ghsom_sweep.yaml"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.debug(f"Config loading failed (using defaults): {e}")
    return {}


def main() -> int:
    """Main entry point for the sweep agent."""
    args = _parse_args()

    # Load sweep config for sweep_control parameters
    full_config = _load_sweep_config()
    sweep_control = full_config.get("sweep_control", {})

    # Load scoring configuration from YAML (priority: config > defaults)
    scoring_dict = sweep_control.get("scoring", {})
    scoring_config = (
        ScoringConfig.from_dict(scoring_dict)
        if scoring_dict
        else ScoringConfig.default()
    )

    # Resolve reduced artifact path (priority: CLI > config > env var > default)
    if args.reduced_artifact is not None:
        reduced_artifact_path = args.reduced_artifact.resolve()
    else:
        artifact_from_config = sweep_control.get("reduced_artifact")
        env_path = os.environ.get("GHSOM_SWEEP_REDUCED_ARTIFACT")
        if artifact_from_config:
            reduced_artifact_path = (PROJECT_ROOT / artifact_from_config).resolve()
        elif env_path:
            reduced_artifact_path = Path(env_path).resolve()
        else:
            reduced_artifact_path = (PROJECT_ROOT / DEFAULT_REDUCED_ARTIFACT).resolve()

    # Resolve project name (priority: CLI > config > env var > default)
    project = (
        args.project
        or sweep_control.get("project")
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_PROJECT
    )

    # Resolve entity (priority: CLI > config > env var)
    entity = (
        args.entity or sweep_control.get("entity") or os.environ.get("WANDB_ENTITY")
    )

    # Resolve feature type (priority: CLI if not default > config > CLI default)
    feature_type = (
        args.feature_type
        if args.feature_type != DEFAULT_FEATURE_TYPE
        else sweep_control.get("feature_type", DEFAULT_FEATURE_TYPE)
    )

    # Resolve seed (priority: CLI if not default > config > CLI default)
    seed = args.seed if args.seed != 42 else sweep_control.get("seed", 42)

    # Resolve n_workers (priority: CLI if not default > config > CLI default)
    n_workers = (
        args.n_workers if args.n_workers != -1 else sweep_control.get("n_workers", -1)
    )

    logger.info(f"Reduced artifact: {reduced_artifact_path}")
    logger.info(f"Feature type: {feature_type}")
    logger.info(f"Project: {project}")
    logger.info(f"Entity: {entity or '(default)'}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Workers: {n_workers}")

    return run_sweep_agent(
        reduced_artifact_path=reduced_artifact_path,
        scoring_config=scoring_config,
        feature_type=feature_type,
        project=project,
        entity=entity,
        dry_run=args.dry_run,
        seed=seed,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    sys.exit(main())
