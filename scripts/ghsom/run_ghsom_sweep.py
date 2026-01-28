#!/usr/bin/env python3
"""WandB sweep launcher for GHSOM hyperparameter tuning.

This script initializes a WandB sweep and optionally starts sweep agents.
It provides a convenient way to launch hyperparameter tuning for GHSOM models.

Usage:
    # Initialize sweep and run 150 trials (default config)
    python scripts/ghsom/run_ghsom_sweep.py --count 150

    # Initialize sweep with custom config
    python scripts/ghsom/run_ghsom_sweep.py --config configs/sweep/ghsom_sweep.yaml --count 150

    # Initialize sweep only (don't start agents)
    python scripts/ghsom/run_ghsom_sweep.py --init-only

    # Then run agents manually (can be distributed across machines)
    wandb agent <entity>/<project>/<sweep_id>

    # Run with custom project and artifact path
    python scripts/ghsom/run_ghsom_sweep.py \
        --project my-ghsom-sweep \
        --reduced-artifact artifacts/preprocessing/umap/reduced/reduced/embedding.npy \
        --count 100

Environment Variables:
    WANDB_PROJECT: Default project name
    WANDB_ENTITY: Default entity/team name
    GHSOM_SWEEP_REDUCED_ARTIFACT: Default reduced features artifact path

Output:
    After initialization, prints sweep URL and agent command for distributed execution.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sweep" / "ghsom_sweep.yaml"
DEFAULT_PROJECT = "aria-ghsom-sweep"
DEFAULT_REDUCED_ARTIFACT = "artifacts/preprocessing/umap/reduced/reduced/embedding.npy"
DEFAULT_COUNT = 150

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ghsom_sweep_launcher")


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================


def load_sweep_config(config_path: Path) -> Dict[str, Any]:
    """Load sweep configuration from YAML file.

    Args:
        config_path: Path to sweep config YAML file.

    Returns:
        Sweep configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded sweep config from: {config_path}")
    return config


def create_default_sweep_config() -> Dict[str, Any]:
    """Create default sweep configuration programmatically.

    This is used when no config file is specified, providing a sensible
    default configuration for GHSOM hyperparameter tuning.

    Returns:
        Default sweep configuration dictionary.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "composite_score",
            "goal": "maximize",
        },
        "parameters": {
            "t1": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.8,
            },
            "t2": {
                "distribution": "uniform",
                "min": 0.01,
                "max": 0.3,
            },
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 0.01,
                "max": 0.3,
            },
            "decay": {
                "distribution": "uniform",
                "min": 0.9,
                "max": 0.999,
            },
            "gaussian_sigma": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 3.0,
            },
            "epochs": {
                "distribution": "int_uniform",
                "min": 10,
                "max": 100,
            },
            "grow_maxiter": {
                "distribution": "int_uniform",
                "min": 10,
                "max": 50,
            },
        },
        "program": "scripts/ghsom/run_ghsom_sweep_agent.py",
        "command": ["${env}", "python", "${program}", "${args}"],
    }


# =============================================================================
# SWEEP MANAGEMENT
# =============================================================================


def initialize_sweep(
    config: Dict[str, Any],
    project: str,
    entity: Optional[str] = None,
) -> str:
    """Initialize a WandB sweep.

    Args:
        config: Sweep configuration dictionary.
        project: WandB project name.
        entity: WandB entity/team name (optional).

    Returns:
        Sweep ID string.

    Raises:
        RuntimeError: If sweep initialization fails.
    """
    try:
        import wandb
    except ImportError as err:
        raise RuntimeError(
            "WandB is not installed. Please install with: pip install wandb"
        ) from err

    logger.info(f"Initializing sweep in project: {project}")
    if entity:
        logger.info(f"Entity: {entity}")

    # Filter out non-WandB keys (sweep_control is for our scripts, not WandB)
    wandb_config = {k: v for k, v in config.items() if k != "sweep_control"}

    sweep_id = wandb.sweep(sweep=wandb_config, project=project, entity=entity)

    logger.info(f"Sweep initialized with ID: {sweep_id}")
    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    project: str,
    entity: Optional[str],
    count: int,
    reduced_artifact_path: Path,
    feature_type: str,
    scoring_config_dict: Optional[Dict[str, Any]] = None,
) -> int:
    """Run WandB sweep agent(s).

    This function uses subprocess to run the agent script, which allows
    for proper signal handling and cleanup.

    Args:
        sweep_id: WandB sweep ID.
        project: WandB project name.
        entity: WandB entity/team name (optional).
        count: Number of trials to run.
        reduced_artifact_path: Path to reduced features artifact.
        feature_type: Type of features to load.
        scoring_config_dict: Scoring configuration dictionary.

    Returns:
        Exit code from agent execution.
    """
    try:
        import wandb
    except ImportError as err:
        raise RuntimeError(
            "WandB is not installed. Please install with: pip install wandb"
        ) from err

    # Build sweep path
    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"

    logger.info(f"Starting sweep agent for: {sweep_path}")
    logger.info(f"Running {count} trials")

    # Set environment variables for the agent
    env = os.environ.copy()
    env["GHSOM_SWEEP_REDUCED_ARTIFACT"] = str(reduced_artifact_path)
    env["WANDB_PROJECT"] = project
    if entity:
        env["WANDB_ENTITY"] = entity

    # Use wandb.agent() directly
    def agent_function():
        """Agent function called by wandb.agent."""
        from scripts.ghsom.run_ghsom_sweep_agent import (
            ScoringConfig,
        )
        from scripts.ghsom.run_ghsom_sweep_agent import (
            run_sweep_agent as run_agent,
        )

        # Create scoring config from dict
        scoring_config = (
            ScoringConfig.from_dict(scoring_config_dict)
            if scoring_config_dict
            else ScoringConfig.default()
        )

        return run_agent(
            reduced_artifact_path=reduced_artifact_path,
            scoring_config=scoring_config,
            feature_type=feature_type,
            project=project,
            entity=entity,
            dry_run=False,
        )

    try:
        wandb.agent(
            sweep_id,
            function=agent_function,
            project=project,
            entity=entity,
            count=count,
        )
        return 0
    except KeyboardInterrupt:
        logger.info("Sweep interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Sweep agent failed: {e}")
        return 1


def print_sweep_info(
    sweep_id: str,
    project: str,
    entity: Optional[str],
    reduced_artifact_path: Path,
) -> None:
    """Print sweep information and usage instructions.

    Args:
        sweep_id: WandB sweep ID.
        project: WandB project name.
        entity: WandB entity/team name (optional).
        reduced_artifact_path: Path to reduced features artifact.
    """
    if entity:
        sweep_path = f"{entity}/{project}/{sweep_id}"
        sweep_url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
    else:
        sweep_path = f"{project}/{sweep_id}"
        sweep_url = f"https://wandb.ai/{project}/sweeps/{sweep_id}"

    print("\n" + "=" * 80)
    print("GHSOM HYPERPARAMETER SWEEP INITIALIZED")
    print("=" * 80)
    print(f"Sweep ID:     {sweep_id}")
    print(f"Sweep Path:   {sweep_path}")
    print(f"Sweep URL:    {sweep_url}")
    print(f"Artifact:     {reduced_artifact_path}")
    print("=" * 80)
    print("\nTo run sweep agents manually (can be distributed across machines):")
    print(f"  wandb agent {sweep_path}")
    print("\nOr with a specific count:")
    print(f"  wandb agent {sweep_path} --count 50")
    print("\nEnvironment setup for agents:")
    print(f"  export GHSOM_SWEEP_REDUCED_ARTIFACT={reduced_artifact_path}")
    print(f"  export WANDB_PROJECT={project}")
    if entity:
        print(f"  export WANDB_ENTITY={entity}")
    print("=" * 80 + "\n")


# =============================================================================
# CLI
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize and run WandB sweep for GHSOM hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to sweep config YAML (default: {DEFAULT_CONFIG_PATH})",
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
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of trials to run (default: {DEFAULT_COUNT})",
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
        default="umap",
        choices=["umap", "tsne", "pca", "raw"],
        help="Type of features to load (default: umap)",
    )

    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize sweep, don't start agents (for distributed execution)",
    )

    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Resume an existing sweep by ID (skip initialization)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the sweep launcher."""
    args = _parse_args()

    # Load or create sweep config first to get sweep_control parameters
    if args.config is not None:
        sweep_config = load_sweep_config(args.config)
    elif DEFAULT_CONFIG_PATH.exists():
        sweep_config = load_sweep_config(DEFAULT_CONFIG_PATH)
    else:
        logger.info("Using default sweep configuration")
        sweep_config = create_default_sweep_config()

    # Extract sweep_control section if it exists
    sweep_control = sweep_config.get("sweep_control", {})

    # Resolve project name (priority: CLI > sweep_control > env var > default)
    project = (
        args.project
        or sweep_control.get("project")
        or os.environ.get("WANDB_PROJECT")
        or DEFAULT_PROJECT
    )

    # Resolve entity (priority: CLI > sweep_control > env var)
    entity = (
        args.entity or sweep_control.get("entity") or os.environ.get("WANDB_ENTITY")
    )

    # Resolve count (priority: CLI if not default > sweep_control > CLI default)
    count = (
        args.count
        if args.count != DEFAULT_COUNT
        else sweep_control.get("count", DEFAULT_COUNT)
    )

    # Resolve reduced artifact path (priority: CLI > sweep_control > env var > default)
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

    # Resolve feature type (priority: CLI if not default > sweep_control > CLI default)
    feature_type = (
        args.feature_type
        if args.feature_type != "umap"
        else sweep_control.get("feature_type", "umap")
    )

    # Validate artifact path
    if not reduced_artifact_path.exists():
        logger.error(f"Reduced artifact not found: {reduced_artifact_path}")
        logger.error(
            "Please provide a valid path via:\n"
            "  1. CLI: --reduced-artifact <path>\n"
            "  2. Config: sweep_control.reduced_artifact in YAML\n"
            "  3. Environment: GHSOM_SWEEP_REDUCED_ARTIFACT variable"
        )
        return 1

    logger.info(f"Project: {project}")
    logger.info(f"Entity: {entity or '(default)'}")
    logger.info(f"Count: {count}")
    logger.info(f"Reduced artifact: {reduced_artifact_path}")
    logger.info(f"Feature type: {feature_type}")

    # Initialize or resume sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        logger.info(f"Resuming existing sweep: {sweep_id}")
    else:
        try:
            sweep_id = initialize_sweep(sweep_config, project, entity)
        except RuntimeError as e:
            logger.error(str(e))
            return 1

    # Print sweep info
    print_sweep_info(sweep_id, project, entity, reduced_artifact_path)

    # Run agents if not init-only
    if args.init_only:
        logger.info("Initialization complete. Use 'wandb agent' to start agents.")
        return 0

    logger.info(f"Starting {count} sweep trials...")

    # Extract scoring config dict from sweep_control
    scoring_config_dict = sweep_control.get("scoring", {})

    return run_sweep_agent(
        sweep_id=sweep_id,
        project=project,
        entity=entity,
        count=count,
        reduced_artifact_path=reduced_artifact_path,
        feature_type=feature_type,
        scoring_config_dict=scoring_config_dict,
    )


if __name__ == "__main__":
    sys.exit(main())
