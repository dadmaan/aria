#!/usr/bin/env python3
"""Tianshou training entry point.

This script provides a command-line interface for training RL agents
using the Tianshou backend. It supports both DRQN (LSTM) and MLP
network architectures for benchmarking.

Features:
    - Modular network selection via config (drqn/mlp)
    - WandB and TensorBoard logging integration
    - Checkpoint save/load
    - Reproducible training with seeds
    - Tianshou-native training loop

Usage:
    # Train with default config
    python scripts/training/run_training.py

    # Train with custom config
    python scripts/training/run_training.py \\
        --config configs/tianshou_config.yaml

    # Train MLP instead of DRQN
    python scripts/training/run_training.py \\
        --network-type mlp

    # Resume from checkpoint
    python scripts/training/run_training.py \\
        --checkpoint artifacts/training/run_xyz/checkpoints/best.pth

Example:
    python scripts/training/run_training.py \\
        --config configs/tianshou_config.yaml \\
        --timesteps 50000 \\
        --network-type drqn \\
        --output-dir artifacts/training \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train RL agent using Tianshou backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to configuration file (YAML).",
    )
    parser.add_argument(
        "--network-type",
        type=str,
        choices=["drqn", "mlp"],
        default=None,
        help="Override network type from config.",
    )

    # Training
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    # Network architecture parameters
    parser.add_argument(
        "--lstm-hidden",
        type=int,
        default=None,
        help="LSTM hidden size for DRQN (overrides config).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay buffer size (overrides config).",
    )
    parser.add_argument(
        "--target-update-freq",
        type=int,
        default=None,
        help="Target network update frequency (overrides config).",
    )

    # Exploration parameters
    parser.add_argument(
        "--initial-eps",
        type=float,
        default=None,
        help="Initial epsilon for exploration (overrides config).",
    )
    parser.add_argument(
        "--final-eps",
        type=float,
        default=None,
        help="Final epsilon for exploration (overrides config).",
    )
    parser.add_argument(
        "--eps-decay-fraction",
        type=float,
        default=None,
        help="Fraction of training for epsilon decay (overrides config).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor gamma (overrides config).",
    )

    # Reward component weights
    parser.add_argument(
        "--structure-weight",
        type=float,
        default=None,
        help="Structure reward component weight (overrides config).",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=None,
        help="Transition reward component weight (overrides config).",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=None,
        help="Diversity reward component weight (overrides config).",
    )

    # Reward component sub-parameters
    parser.add_argument(
        "--transition-structural-weight",
        type=float,
        default=None,
        help="Transition structural weight sub-parameter (overrides config).",
    )
    parser.add_argument(
        "--transition-max-distance",
        type=float,
        default=None,
        help="Transition max distance sub-parameter (overrides config).",
    )
    parser.add_argument(
        "--diversity-optimal-ratio-low",
        type=float,
        default=None,
        help="Diversity optimal ratio low sub-parameter (overrides config).",
    )
    parser.add_argument(
        "--diversity-optimal-ratio-high",
        type=float,
        default=None,
        help="Diversity optimal ratio high sub-parameter (overrides config).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for output directory naming.",
    )

    # System
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Device for training.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Verbosity level: 0=silent, 1=info, 2=debug.",
    )

    # Logging
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated WandB tags (e.g., 'benchmark,phase1,exp1.1').",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def apply_cli_overrides(
    config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Apply command-line argument overrides to config.

    Args:
        config: Base configuration.
        args: Parsed CLI arguments.

    Returns:
        Updated configuration.
    """
    # Network type override
    if args.network_type:
        config.setdefault("network", {})["type"] = args.network_type

    # Network architecture overrides
    if args.lstm_hidden:
        config.setdefault("network", {}).setdefault("lstm", {})[
            "hidden_size"
        ] = args.lstm_hidden

    # Training overrides
    if args.timesteps:
        config.setdefault("training", {})["total_timesteps"] = args.timesteps
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.buffer_size:
        config.setdefault("training", {})["buffer_size"] = args.buffer_size
    if args.target_update_freq:
        config.setdefault("training", {})[
            "target_update_freq"
        ] = args.target_update_freq
    if args.gamma is not None:
        config.setdefault("training", {})["gamma"] = args.gamma

    # Exploration overrides
    if args.initial_eps is not None:
        config.setdefault("training", {}).setdefault("exploration", {})[
            "initial_eps"
        ] = args.initial_eps
    if args.final_eps is not None:
        config.setdefault("training", {}).setdefault("exploration", {})[
            "final_eps"
        ] = args.final_eps
    if args.eps_decay_fraction is not None:
        config.setdefault("training", {}).setdefault("exploration", {})[
            "fraction"
        ] = args.eps_decay_fraction

    # Reward component weight overrides
    if args.structure_weight is not None:
        config.setdefault("reward_components", {}).setdefault("structure", {})[
            "weight"
        ] = args.structure_weight
    if args.transition_weight is not None:
        config.setdefault("reward_components", {}).setdefault("transition", {})[
            "weight"
        ] = args.transition_weight
    if args.diversity_weight is not None:
        config.setdefault("reward_components", {}).setdefault("diversity", {})[
            "weight"
        ] = args.diversity_weight

    # Reward component sub-parameter overrides
    if args.transition_structural_weight is not None:
        config.setdefault("reward_components", {}).setdefault("transition", {})[
            "structural_weight"
        ] = args.transition_structural_weight
    if args.transition_max_distance is not None:
        config.setdefault("reward_components", {}).setdefault("transition", {})[
            "max_distance"
        ] = args.transition_max_distance
    if args.diversity_optimal_ratio_low is not None:
        config.setdefault("reward_components", {}).setdefault("diversity", {})[
            "optimal_ratio_low"
        ] = args.diversity_optimal_ratio_low
    if args.diversity_optimal_ratio_high is not None:
        config.setdefault("reward_components", {}).setdefault("diversity", {})[
            "optimal_ratio_high"
        ] = args.diversity_optimal_ratio_high

    # System overrides
    if args.seed is not None:
        config.setdefault("system", {})["seed"] = args.seed
    if args.device:
        config.setdefault("system", {})["device"] = args.device
    if args.verbose is not None:
        config.setdefault("system", {})["verbose"] = args.verbose

    # Output overrides
    if args.output_dir:
        config.setdefault("paths", {})["output_dir"] = args.output_dir

    # Logging overrides
    if args.no_wandb:
        config.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    if args.no_tensorboard:
        config.setdefault("logging", {}).setdefault("tensorboard", {})[
            "enabled"
        ] = False
    if args.wandb_tags:
        tags = [tag.strip() for tag in args.wandb_tags.split(",")]
        config.setdefault("logging", {}).setdefault("wandb", {})["tags"] = tags

    return config


def _extract_ghsom_manager(env: Any) -> Optional[Any]:
    """Best-effort extraction of the GHSOM manager from (wrapped) envs.

    Handles plain envs with `perceiving_agent`, custom gym wrappers, and
    vectorized envs that expose `get_attr`. We walk through wrapper layers to
    find the underlying perceiving agent.
    """

    if env is None:
        return None

    current = env
    for _ in range(10):  # prevent infinite wrapper chains
        # Direct attribute on the current env
        if hasattr(current, "perceiving_agent") and hasattr(
            current.perceiving_agent, "ghsom_manager"
        ):
            return current.perceiving_agent.ghsom_manager

        # Vectorized envs may expose get_attr
        if hasattr(current, "get_attr"):
            try:
                perceiving_agents = current.get_attr("perceiving_agent")
                if perceiving_agents and hasattr(perceiving_agents[0], "ghsom_manager"):
                    return perceiving_agents[0].ghsom_manager
            except Exception:
                pass

        # Walk through wrappers
        if hasattr(current, "env"):
            current = current.env
            continue
        break

    return None


def create_output_directory(
    config: Dict[str, Any], run_id: Optional[str] = None
) -> Path:
    """Create timestamped output directory.

    Args:
        config: Configuration with paths.
        run_id: Optional run ID for naming.

    Returns:
        Path to output directory.
    """
    base_dir = Path(config.get("paths", {}).get("output_dir", "artifacts/training"))

    if run_id:
        run_dir = base_dir / run_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        network_type = config.get("network", {}).get("type", "drqn")
        run_dir = base_dir / f"run_{network_type}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)

    return run_dir


def setup_callbacks(
    config: Dict[str, Any], output_dir: Path, env: Any = None
) -> List[Any]:
    """Setup logging callbacks.

    Args:
        config: Logging configuration.
        output_dir: Output directory.
        env: Optional environment for GHSOM manager access.

    Returns:
        List of callback instances.
    """
    callbacks = []
    logging_config = config.get("logging", {})
    wandb_cfg = logging_config.get("wandb", {})
    run_name = config.get("paths", {}).get("run_id", output_dir.name)

    # TensorBoard callback
    tb_cfg = logging_config.get("tensorboard", {})
    if tb_cfg.get("enabled", True):
        try:
            from src.training.callbacks import TianshouTensorBoardCallback

            tb_log_dir = output_dir / "tensorboard"
            callbacks.append(
                TianshouTensorBoardCallback(
                    log_dir=str(tb_log_dir),
                    log_interval=tb_cfg.get("log_interval", 10),
                )
            )
            print(f"✓ TensorBoard callback enabled (log_dir: {tb_log_dir})")
        except ImportError as e:
            print(f"⚠ TensorBoard callback not available: {e}")

    # Metrics callback (local file logging)
    try:
        from src.utils.callbacks.tianshou_callbacks import TianshouMetricsCallback

        callbacks.append(
            TianshouMetricsCallback(
                output_dir=output_dir,
                verbose=config.get("system", {}).get("verbose", 1),
            )
        )
        print("✓ Metrics callback enabled")
    except ImportError as e:
        print(f"⚠ Metrics callback not available: {e}")

    # Comprehensive Metrics callback (includes system metrics, Q-values, gradients, GHSOM)
    # Logs to WandB, TensorBoard, and local JSON files
    comprehensive_cfg = logging_config.get("comprehensive", {})
    if comprehensive_cfg.get("enabled", True):
        try:
            from src.training.callbacks import TianshouComprehensiveMetricsCallback

            ghsom_manager = _extract_ghsom_manager(env)

            # Always enable GHSOM metrics for sequence tracking (works without manager)
            # The ghsom_manager is only needed for coverage metrics (total_nodes)
            log_ghsom = comprehensive_cfg.get("log_ghsom_metrics", True)

            callbacks.append(
                TianshouComprehensiveMetricsCallback(
                    config=config,
                    ghsom_manager=ghsom_manager,  # Can be None - sequence tracking still works
                    log_system_metrics=comprehensive_cfg.get(
                        "log_system_metrics", True
                    ),
                    log_model_info=comprehensive_cfg.get("log_model_info", True),
                    log_ghsom_metrics=log_ghsom,
                    log_training_metrics=comprehensive_cfg.get(
                        "log_training_metrics", True
                    ),
                    log_to_tensorboard=comprehensive_cfg.get(
                        "log_to_tensorboard", True
                    ),
                    log_to_local=comprehensive_cfg.get("log_to_local", True),
                    output_dir=str(output_dir),
                    system_metrics_freq=comprehensive_cfg.get(
                        "system_metrics_freq", 10
                    ),
                    q_value_log_freq=comprehensive_cfg.get("q_value_log_freq", 10),
                    gradient_log_freq=comprehensive_cfg.get("gradient_log_freq", 10),
                    verbose=config.get("system", {}).get("verbose", 1),
                )
            )
            print(
                f"✓ Comprehensive Metrics callback enabled (GHSOM manager: {'yes' if ghsom_manager else 'no (sequence tracking still active)'})"
            )
        except ImportError as e:
            print(f"⚠ Comprehensive Metrics callback not available: {e}")

    # Reward Components callback (individual reward component tracking)
    reward_components_cfg = logging_config.get("reward_components", {})
    if reward_components_cfg.get("enabled", True):
        try:
            from src.training.callbacks import TianshouRewardComponentsCallback

            callbacks.append(
                TianshouRewardComponentsCallback(
                    log_frequency=reward_components_cfg.get("log_frequency", 100),
                    export_frequency=reward_components_cfg.get(
                        "export_frequency", 1000
                    ),
                    log_to_wandb=reward_components_cfg.get("log_to_wandb", True),
                    log_to_tensorboard=reward_components_cfg.get(
                        "log_to_tensorboard", True
                    ),
                    log_to_local=reward_components_cfg.get("log_to_local", True),
                    output_dir=str(output_dir),
                    max_history=reward_components_cfg.get("max_history", 1000),
                    verbose=config.get("system", {}).get("verbose", 1),
                )
            )
            print("✓ Reward Components callback enabled")
        except ImportError as e:
            print(f"⚠ Reward Components callback not available: {e}")

    # Learning Rate Scheduler callback (dynamic LR scheduling with optional pulse mechanism)
    training_config = config.get("training", {})
    lr_scheduler_cfg = training_config.get("learning_rate_scheduler", {})
    if lr_scheduler_cfg.get("enabled", False):
        try:
            from src.training.callbacks import TianshouLRSchedulerCallback

            # Get pulse mechanism config
            pulse_cfg = lr_scheduler_cfg.get("pulse_mechanism", {})
            adaptive_cfg = pulse_cfg.get("adaptive_threshold", {})

            callbacks.append(
                TianshouLRSchedulerCallback(
                    initial_lr=lr_scheduler_cfg.get(
                        "initial_lr", training_config.get("learning_rate", 0.001)
                    ),
                    final_lr=lr_scheduler_cfg.get("final_lr", 0.0001),
                    decay_steps=lr_scheduler_cfg.get("decay_steps", 5000),
                    schedule_type=lr_scheduler_cfg.get("type", "exponential"),
                    decay_rate=lr_scheduler_cfg.get("decay_rate", 0.995),
                    # Pulse mechanism
                    pulse_enabled=pulse_cfg.get("enabled", False),
                    pulse_duration_episodes=pulse_cfg.get("duration_episodes", 50),
                    pulse_boost_epsilon=pulse_cfg.get("boost_epsilon"),
                    trigger_mode=pulse_cfg.get("trigger_mode", "adaptive"),
                    trigger_threshold=pulse_cfg.get("trigger_threshold"),
                    # Adaptive threshold
                    adaptive_enabled=adaptive_cfg.get("enabled", True),
                    baseline_alpha=adaptive_cfg.get("baseline_alpha", 0.1),
                    relative_drop_threshold=adaptive_cfg.get(
                        "relative_drop_threshold", 0.15
                    ),
                    # Logging
                    log_to_wandb=logging_config.get("wandb", {}).get("enabled", True),
                    log_to_tensorboard=logging_config.get("tensorboard", {}).get(
                        "enabled", True
                    ),
                    log_to_local=True,
                    output_dir=str(output_dir),
                    log_frequency=100,
                    verbose=config.get("system", {}).get("verbose", 1),
                )
            )
            schedule_type = lr_scheduler_cfg.get("type", "exponential")
            pulse_str = " (pulse enabled)" if pulse_cfg.get("enabled") else ""
            print(f"✓ LR Scheduler callback enabled ({schedule_type}{pulse_str})")
        except ImportError as e:
            print(f"⚠ LR Scheduler callback not available: {e}")

    # WandB callback - added LAST so it closes the run after other callbacks have logged
    if wandb_cfg.get("enabled", True):
        try:
            from src.training.callbacks import TianshouWandBCallback

            network_type = config.get("network", {}).get("type", "drqn")
            tags = wandb_cfg.get("tags", []) + [network_type, "tianshou"]

            callbacks.append(
                TianshouWandBCallback(
                    project=wandb_cfg.get("project", "ARIA_rl_V03"),
                    name=run_name,
                    entity=wandb_cfg.get("entity"),
                    config=config,
                    tags=tags,
                    log_interval=wandb_cfg.get("log_interval", 10),
                )
            )
            print(
                f"✓ WandB callback enabled (project: {wandb_cfg.get('project', 'ARIA_rl_V03')})"
            )
        except ImportError as e:
            print(f"⚠ WandB callback not available: {e}")

    return callbacks


def create_environment(config: Dict[str, Any]):
    """Create training environment.

    Args:
        config: Environment configuration.

    Returns:
        Gymnasium environment.
    """
    from pathlib import Path

    from src.ghsom_manager import GHSOMManager
    from src.agents.ghsom_perceiving_agent import GHSOMPerceivingAgent
    from src.environments.music_env_gym import MusicGenerationGymEnv
    from src.agents.cluster_feature_mapper import ClusterFeatureMapper

    # Get GHSOM and feature paths from config
    ghsom_config = config.get("ghsom", {})
    features_config = config.get("features", {})

    ghsom_path = ghsom_config.get("checkpoint") or ghsom_config.get(
        "default_model_path"
    )
    feature_path = features_config.get("artifact_path")
    feature_type = features_config.get("type", "tsne")

    if ghsom_path is None:
        raise ValueError(
            "GHSOM model path not specified in config (ghsom.checkpoint or ghsom.default_model_path)"
        )

    if feature_path is None:
        raise ValueError(
            "Feature artifact path not specified in config (features.artifact_path)"
        )

    # Initialize GHSOM with features via the from_artifact factory
    ghsom_manager = GHSOMManager.from_artifact(
        ghsom_model_path=Path(ghsom_path),
        feature_artifact=Path(feature_path),
        feature_type=feature_type,
    )

    # Create perceiving agent
    perceiving_agent = GHSOMPerceivingAgent(
        ghsom_manager=ghsom_manager,
        config=config,
    )

    # Get sequence length
    sequence_length = config.get("music", {}).get("sequence_length", 16)

    # Create base environment
    env = MusicGenerationGymEnv(
        perceiving_agent=perceiving_agent,
        sequence_length=sequence_length,
        config=config,
    )

    # Wrap with feature observations if configured
    use_features = config.get("use_feature_observations", False)
    if use_features:
        feature_mode = config.get("feature_observation_mode", "centroid")

        # Create feature mapper using the ghsom_manager
        mapper = ClusterFeatureMapper(
            ghsom_manager=ghsom_manager,
            mode=feature_mode,
            feature_source=feature_type,
        )

        # Wrap environment with feature observations
        env = FeatureObservationWrapper(env, mapper)

    return env


class FeatureObservationWrapper(gym.Wrapper):
    """Wrapper that converts cluster ID observations to feature vectors.

    This wrapper takes the base environment's observation (sequence of cluster IDs)
    and converts each cluster ID to its corresponding feature vector using
    the ClusterFeatureMapper.
    """

    def __init__(self, env: gym.Env, mapper):
        """Initialize the wrapper.

        Args:
            env: Base gymnasium environment
            mapper: ClusterFeatureMapper instance
        """
        super().__init__(env)
        self.mapper = mapper

        # Update observation space to feature vectors
        seq_length = env.observation_space.shape[0]
        feature_dim = mapper.feature_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_length, feature_dim),
            dtype=np.float32,
        )

    def _convert_observation(self, obs):
        """Convert cluster ID observation to feature observation."""
        return self.mapper.map_sequence(obs.tolist())

    def reset(self, **kwargs):
        """Reset the environment and convert observation."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_observation(obs), info

    def step(self, action):
        """Take a step and convert observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_observation(obs), reward, terminated, truncated, info


def setup_curriculum_learning(
    config: Dict[str, Any],
    env: gym.Env,
    callbacks: List[Any],
) -> tuple:
    """Setup curriculum learning if enabled.

    Args:
        config: Configuration dictionary.
        env: Base environment.
        callbacks: List of callbacks to append to.

    Returns:
        Tuple of (wrapped_env, curriculum_callback or None).
    """
    curriculum_config = config.get("curriculum", {})

    if not curriculum_config.get("enabled", False):
        return env, None

    print("\nSetting up curriculum learning...")

    try:
        from src.curriculum import HierarchyExtractor, CurriculumCallback
        from src.environments.curriculum_wrapper import CurriculumEnvironmentWrapper

        # Get GHSOM path
        ghsom_config = config.get("ghsom", {})
        ghsom_path = curriculum_config.get("ghsom_experiment_path")
        if ghsom_path is None:
            ghsom_path = ghsom_config.get("checkpoint") or ghsom_config.get(
                "default_model_path"
            )

        if ghsom_path is None:
            raise ValueError("GHSOM path not specified for curriculum learning")

        # Extract directory from model path if it's a file
        ghsom_path = Path(ghsom_path)
        if ghsom_path.suffix == ".pkl":
            experiment_dir = ghsom_path.parent
        else:
            experiment_dir = ghsom_path

        print(f"  Extracting hierarchy from: {experiment_dir}")

        # Extract hierarchy
        extractor = HierarchyExtractor.from_experiment_dir(experiment_dir)
        hierarchy = extractor.extract()

        print(f"  Phases: {hierarchy.total_phases}")
        print(f"  Leaf clusters: {hierarchy.total_leaf_clusters}")
        for phase_num in range(1, hierarchy.total_phases + 1):
            phase = hierarchy.get_phase(phase_num)
            print(f"    Phase {phase_num}: {phase.action_space_size} actions")

        # Wrap environment
        curriculum_wrapper = CurriculumEnvironmentWrapper(
            env=env,
            hierarchy=hierarchy,
            initial_phase=1,
            seed=config.get("system", {}).get("seed", 42),
        )

        # Get transition config
        transition_cfg = curriculum_config.get("transition", {})

        # Get algorithm config for distributional networks
        algo_config = config.get("algorithm", {})
        algo_type = algo_config.get("type", "auto")  # auto-detect from network
        dist_config = algo_config.get("distributional", {})
        num_atoms = dist_config.get("num_atoms", 51)
        v_min = dist_config.get("v_min", -3.0)
        v_max = dist_config.get("v_max", 7.0)

        # Create callback
        curriculum_callback = CurriculumCallback(
            hierarchy=hierarchy,
            env_wrapper=curriculum_wrapper,
            timesteps_per_action=curriculum_config.get("timesteps_per_action", 2500),
            patience_per_action=curriculum_config.get("patience_per_action", 150),
            plateau_threshold=curriculum_config.get("plateau_threshold", 0.01),
            epsilon_boost=transition_cfg.get("epsilon_boost", 0.3),
            flush_buffer=transition_cfg.get("flush_buffer", True),
            add_mitosis_noise=transition_cfg.get("add_mitosis_noise", True),
            mitosis_noise_scale=transition_cfg.get("mitosis_noise_scale", 0.01),
            verbose=config.get("system", {}).get("verbose", 1),
            # Algorithm-specific parameters for distributional networks
            algo_type=algo_type,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
        )

        callbacks.append(curriculum_callback)

        print(f"✓ Curriculum learning enabled")
        print(f"  Starting action space: {curriculum_wrapper.action_space.n}")

        return curriculum_wrapper, curriculum_callback

    except Exception as e:
        print(f"✗ Failed to setup curriculum learning: {e}")
        import traceback

        traceback.print_exc()
        raise


def _disable_wandb_globally() -> None:
    """Disable WandB globally by setting environment variable.

    This must be called BEFORE any imports that trigger logger creation.
    Sets ARIA_WANDB_DISABLED which is checked by LoggingManager.
    """
    import os

    # Set our custom environment variable that LoggingManager checks
    os.environ["ARIA_WANDB_DISABLED"] = "true"

    # Also set WANDB_MODE to disabled to prevent any wandb.init() from syncing
    os.environ["WANDB_MODE"] = "disabled"


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()

    # IMPORTANT: Disable WandB globally BEFORE any environment/logger creation
    # if --no-wandb was passed. This prevents LoggingManager from initializing WandB.
    if args.no_wandb:
        _disable_wandb_globally()
        print("WandB disabled globally via --no-wandb flag")

    print("=" * 60)
    print("TIANSHOU TRAINING")
    print("=" * 60)

    # Load and update config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Create output directory
    output_dir = create_output_directory(config, args.run_id)
    print(f"Output directory: {output_dir}")

    # Update config with output dir
    paths_cfg = config.setdefault("paths", {})
    paths_cfg["output_dir"] = str(output_dir)
    # Preserve run identifier for downstream logging callbacks
    paths_cfg["run_id"] = args.run_id or output_dir.name

    # Save config copy
    config_copy_path = output_dir / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_copy_path}")

    # Create environment first (needed for GHSOM manager access)
    print("\nCreating environment...")
    try:
        env = create_environment(config)
        print(f"✓ Environment created")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return 1

    # Setup callbacks (after env creation for GHSOM manager access)
    print("\nSetting up callbacks...")
    callbacks = setup_callbacks(config, output_dir, env)

    # Setup curriculum learning if enabled
    curriculum_callback = None
    if config.get("curriculum", {}).get("enabled", False):
        try:
            env, curriculum_callback = setup_curriculum_learning(config, env, callbacks)
        except Exception as e:
            print(f"✗ Failed to setup curriculum learning: {e}")
            return 1

    # Import trainer
    from src.training import TianshouTrainer

    # Create or load trainer
    network_type = config.get("network", {}).get("type", "drqn")

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        try:
            trainer = TianshouTrainer.load(
                path=args.checkpoint,
                env=env,
                config=config,
                callbacks=callbacks,
            )
            print("✓ Checkpoint loaded")
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return 1
    else:
        print(f"\nCreating {network_type.upper()} trainer...")
        trainer = TianshouTrainer(
            env=env,
            config=config,
            callbacks=callbacks,
        )
        print("✓ Trainer created")

    # Train
    print(f"\n{'=' * 60}")
    print(f"Starting training with {network_type.upper()} network...")
    print(f"{'=' * 60}\n")

    try:
        result = trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        result = {
            "interrupted": True,
            "num_episodes": trainer.num_episodes,
        }

    # Save final checkpoint
    final_checkpoint_path = output_dir / "checkpoints" / "final"
    trainer.save(final_checkpoint_path)
    print(f"\nFinal checkpoint saved to: {final_checkpoint_path}.pth")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Network type: {network_type}")
    print(f"Total timesteps: {trainer.num_timesteps}")
    print(f"Total episodes: {trainer.num_episodes}")
    print(f"Output directory: {output_dir}")
    if "best_reward" in result:
        print(f"Best reward: {result['best_reward']:.3f}")
    if "train_time" in result:
        print(f"Training time: {result['train_time']:.1f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
