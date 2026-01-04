"""
Data loading utilities for the enhanced training analysis system.

This module provides classes for loading and parsing training data from various sources:
- Configuration files (YAML)
- Training metrics (JSON)
- TensorBoard logs

Each loader handles missing data gracefully and returns structured, typed data.
"""

import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def get_nested(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely retrieve nested dictionary values.

    This is a module-level utility function for traversing nested dictionaries
    without raising KeyError exceptions.

    Args:
        data: Dictionary to traverse
        *keys: Sequence of keys to traverse
        default: Value to return if path doesn't exist

    Returns:
        Value at key path or default

    Example:
        >>> config = {"training": {"learning_rate": 0.001}}
        >>> get_nested(config, "training", "learning_rate", default=0.0)
        0.001
        >>> get_nested(config, "reward_components", "structure", "weight", default="N/A")
        'N/A'
    """
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str = "constant"
    initial_lr: Optional[float] = None
    final_lr: Optional[float] = None
    decay_rate: Optional[float] = None
    decay_steps: Optional[int] = None
    warmup_steps: Optional[int] = None


@dataclass
class RewardWeights:
    """Reward component weights."""
    structure: float = 0.0
    transition: float = 0.0
    diversity: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "structure": self.structure,
            "transition": self.transition,
            "diversity": self.diversity
        }


@dataclass
class TrainingConfig:
    """
    Standardized training configuration.

    Attributes:
        learning_rate: Initial learning rate
        batch_size: Training batch size
        gamma: Discount factor for RL
        network_type: Type of network architecture (e.g., 'drqn', 'dqn')
        hidden_size: LSTM/RNN hidden size
        embedding_dim: Embedding dimension
        lstm_layers: Number of LSTM layers
        reward_weights: Weights for reward components
        lr_scheduler: Learning rate scheduler configuration
        seed: Random seed
        total_timesteps: Total training timesteps
        target_update_freq: Frequency of target network updates
        buffer_size: Replay buffer size
        eps_start: Initial exploration rate
        eps_end: Final exploration rate
        eps_decay: Exploration decay rate
        experiment_name: Name of the experiment
        additional_params: Any other configuration parameters
    """
    learning_rate: float = 0.001
    batch_size: int = 32
    gamma: float = 0.99
    network_type: str = "dqn"
    hidden_size: int = 128
    embedding_dim: Optional[int] = None
    lstm_layers: int = 1
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    seed: int = 42
    total_timesteps: int = 1000000
    target_update_freq: int = 1000
    buffer_size: int = 10000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995
    experiment_name: str = "default"
    additional_params: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Loader for YAML configuration files.

    Parses config.yaml files and extracts training parameters into a
    standardized TrainingConfig dataclass.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the ConfigLoader.

        Args:
            config_path: Path to the config.yaml file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._raw_config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and parse the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self._raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    def _get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Safely get nested dictionary values.

        Args:
            *keys: Sequence of keys to traverse
            default: Default value if key path doesn't exist

        Returns:
            Value at the key path or default
        """
        value = self._raw_config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def _parse_reward_weights(self) -> RewardWeights:
        """Parse reward component weights from config."""
        structure_weight = self._get_nested("reward_components", "structure", "weight", default=0.0)
        transition_weight = self._get_nested("reward_components", "transition", "weight", default=0.0)
        diversity_weight = self._get_nested("reward_components", "diversity", "weight", default=0.0)

        return RewardWeights(
            structure=float(structure_weight),
            transition=float(transition_weight),
            diversity=float(diversity_weight)
        )

    def _parse_lr_scheduler(self) -> LRSchedulerConfig:
        """Parse learning rate scheduler configuration."""
        scheduler_config = self._get_nested("training", "learning_rate_scheduler", default={})

        if not scheduler_config:
            return LRSchedulerConfig()

        return LRSchedulerConfig(
            type=scheduler_config.get("type", "constant"),
            initial_lr=scheduler_config.get("initial_lr"),
            final_lr=scheduler_config.get("final_lr"),
            decay_rate=scheduler_config.get("decay_rate"),
            decay_steps=scheduler_config.get("decay_steps"),
            warmup_steps=scheduler_config.get("warmup_steps")
        )

    def load(self) -> TrainingConfig:
        """
        Load and parse configuration into TrainingConfig.

        Returns:
            TrainingConfig object with parsed parameters
        """
        # Training parameters
        learning_rate = self._get_nested("training", "learning_rate", default=0.001)
        batch_size = self._get_nested("training", "batch_size", default=32)
        gamma = self._get_nested("training", "gamma", default=0.99)
        total_timesteps = self._get_nested("training", "total_timesteps", default=1000000)
        target_update_freq = self._get_nested("training", "target_update_freq", default=1000)

        # Network parameters
        network_type = self._get_nested("network", "type", default="dqn")
        hidden_size = self._get_nested("network", "lstm", "hidden_size", default=128)
        embedding_dim = self._get_nested("network", "embedding_dim")
        lstm_layers = self._get_nested("network", "lstm", "num_layers", default=1)

        # Buffer parameters
        buffer_size = self._get_nested("training", "buffer_size", default=10000)

        # Exploration parameters
        eps_start = self._get_nested("training", "exploration", "eps_start", default=1.0)
        eps_end = self._get_nested("training", "exploration", "eps_end", default=0.05)
        eps_decay = self._get_nested("training", "exploration", "eps_decay", default=0.995)

        # System parameters
        seed = self._get_nested("system", "seed", default=42)
        experiment_name = self._get_nested("experiment", "name", default="default")

        # Parse complex structures
        reward_weights = self._parse_reward_weights()
        lr_scheduler = self._parse_lr_scheduler()

        # Collect additional parameters not explicitly handled
        additional_params = {}
        for key, value in self._raw_config.items():
            if key not in ["training", "network", "reward_components", "system", "experiment"]:
                additional_params[key] = value

        return TrainingConfig(
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            gamma=float(gamma),
            network_type=str(network_type),
            hidden_size=int(hidden_size),
            embedding_dim=int(embedding_dim) if embedding_dim is not None else None,
            lstm_layers=int(lstm_layers),
            reward_weights=reward_weights,
            lr_scheduler=lr_scheduler,
            seed=int(seed),
            total_timesteps=int(total_timesteps),
            target_update_freq=int(target_update_freq),
            buffer_size=int(buffer_size),
            eps_start=float(eps_start),
            eps_end=float(eps_end),
            eps_decay=float(eps_decay),
            experiment_name=str(experiment_name),
            additional_params=additional_params
        )

    def get_raw_config(self) -> Dict[str, Any]:
        """
        Get the raw configuration dictionary.

        Returns:
            Raw configuration dictionary
        """
        return self._raw_config.copy()


@dataclass
class TrainingMetrics:
    """
    Container for training metrics data.

    Attributes:
        episode_rewards: Array of episode rewards
        episode_lengths: Array of episode lengths
        episode_numbers: Array of episode indices
        timestamps: Array of timestamps (if available)
        total_episodes: Total number of episodes
        total_steps: Total number of steps
        reward_components: Dictionary of reward component arrays (if available)
        additional_metrics: Any other metrics found in the data
    """
    episode_rewards: np.ndarray
    episode_lengths: np.ndarray
    episode_numbers: np.ndarray
    timestamps: Optional[np.ndarray] = None
    total_episodes: int = 0
    total_steps: int = 0
    reward_components: Dict[str, np.ndarray] = field(default_factory=dict)
    additional_metrics: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.total_episodes = len(self.episode_rewards)
        self.total_steps = int(np.sum(self.episode_lengths))


class MetricsLoader:
    """
    Loader for training metrics from JSON files.

    Handles both training_metrics.json and detailed_rewards.json formats.
    Returns numpy arrays for efficient numerical operations.
    """

    def __init__(self, metrics_path: Union[str, Path]):
        """
        Initialize the MetricsLoader.

        Args:
            metrics_path: Path to metrics JSON file or directory containing metrics

        Raises:
            FileNotFoundError: If metrics file/directory doesn't exist
        """
        self.metrics_path = Path(metrics_path)

        # If it's a directory, look for standard metrics files
        if self.metrics_path.is_dir():
            metrics_dir = self.metrics_path
            if (metrics_dir / "training_metrics.json").exists():
                self.metrics_path = metrics_dir / "training_metrics.json"
            elif (metrics_dir / "detailed_rewards.json").exists():
                self.metrics_path = metrics_dir / "detailed_rewards.json"
            else:
                raise FileNotFoundError(
                    f"No metrics files found in directory: {metrics_dir}"
                )

        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")

        self._raw_data: Dict[str, Any] = {}

    def _load_json(self) -> None:
        """Load and parse the JSON metrics file."""
        try:
            with open(self.metrics_path, 'r') as f:
                self._raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON metrics: {e}")

    def _extract_from_training_metrics(self) -> TrainingMetrics:
        """
        Extract metrics from training_metrics.json format.

        Expected format:
        {
            "episodes": [
                {"episode": 0, "reward": 10.5, "length": 100, "timestamp": ...},
                ...
            ]
        }
        """
        episodes_data = self._raw_data.get("episodes", [])

        if not episodes_data:
            # Try alternative structure
            if "episode_rewards" in self._raw_data:
                episode_rewards = np.array(self._raw_data["episode_rewards"])
                episode_lengths = np.array(self._raw_data.get("episode_lengths",
                                                              [0] * len(episode_rewards)))
                episode_numbers = np.arange(len(episode_rewards))
                timestamps = None
                reward_components = {}
            else:
                raise ValueError("No episode data found in metrics file")
        else:
            # Extract from list of episode dictionaries
            episode_numbers = np.array([ep.get("episode", i) for i, ep in enumerate(episodes_data)])
            episode_rewards = np.array([ep.get("reward", ep.get("total_reward", 0.0))
                                       for ep in episodes_data])
            episode_lengths = np.array([ep.get("length", ep.get("steps", 0))
                                       for ep in episodes_data])

            # Extract timestamps if available
            if "timestamp" in episodes_data[0]:
                timestamps = np.array([ep.get("timestamp", 0.0) for ep in episodes_data])
            else:
                timestamps = None

            # Extract reward components if available
            reward_components = {}
            component_keys = ["structure_reward", "transition_reward", "diversity_reward",
                            "structure", "transition", "diversity"]

            for key in component_keys:
                if key in episodes_data[0]:
                    reward_components[key] = np.array([ep.get(key, 0.0) for ep in episodes_data])

        # Extract additional metrics
        additional_metrics = {}
        for key, value in self._raw_data.items():
            if key not in ["episodes", "episode_rewards", "episode_lengths"]:
                if isinstance(value, list) and len(value) > 0:
                    additional_metrics[key] = np.array(value)

        return TrainingMetrics(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            episode_numbers=episode_numbers,
            timestamps=timestamps,
            reward_components=reward_components,
            additional_metrics=additional_metrics
        )

    def _extract_from_detailed_rewards(self) -> TrainingMetrics:
        """
        Extract metrics from detailed_rewards.json format.

        Expected format:
        {
            "episode_0": {
                "total_reward": 10.5,
                "structure_reward": 3.5,
                "transition_reward": 3.0,
                "diversity_reward": 4.0,
                "length": 100
            },
            ...
        }
        """
        episode_rewards = []
        episode_lengths = []
        episode_numbers = []
        reward_components = {"structure": [], "transition": [], "diversity": []}

        # Sort episodes by number
        episode_keys = sorted([k for k in self._raw_data.keys() if k.startswith("episode_")],
                             key=lambda x: int(x.split("_")[1]))

        for episode_key in episode_keys:
            episode_data = self._raw_data[episode_key]
            episode_num = int(episode_key.split("_")[1])

            episode_numbers.append(episode_num)
            episode_rewards.append(episode_data.get("total_reward", 0.0))
            episode_lengths.append(episode_data.get("length", episode_data.get("steps", 0)))

            reward_components["structure"].append(episode_data.get("structure_reward", 0.0))
            reward_components["transition"].append(episode_data.get("transition_reward", 0.0))
            reward_components["diversity"].append(episode_data.get("diversity_reward", 0.0))

        # Convert to numpy arrays
        reward_components_np = {k: np.array(v) for k, v in reward_components.items()}

        return TrainingMetrics(
            episode_rewards=np.array(episode_rewards),
            episode_lengths=np.array(episode_lengths),
            episode_numbers=np.array(episode_numbers),
            timestamps=None,
            reward_components=reward_components_np
        )

    def load(self) -> TrainingMetrics:
        """
        Load and parse metrics from JSON file.

        Returns:
            TrainingMetrics object with parsed data

        Raises:
            ValueError: If metrics format is not recognized
        """
        self._load_json()

        # Detect format and extract accordingly
        if "episodes" in self._raw_data or "episode_rewards" in self._raw_data:
            return self._extract_from_training_metrics()
        elif any(k.startswith("episode_") for k in self._raw_data.keys()):
            return self._extract_from_detailed_rewards()
        else:
            raise ValueError(
                f"Unrecognized metrics format in {self.metrics_path}. "
                "Expected 'episodes', 'episode_rewards', or 'episode_X' keys."
            )

    def get_raw_data(self) -> Dict[str, Any]:
        """
        Get the raw metrics dictionary.

        Returns:
            Raw metrics dictionary
        """
        if not self._raw_data:
            self._load_json()
        return self._raw_data.copy()


@dataclass
class TensorBoardData:
    """
    Container for TensorBoard scalar data.

    Attributes:
        loss: Dictionary mapping step to loss value
        learning_rate: Dictionary mapping step to learning rate
        exploration_rate: Dictionary mapping step to exploration rate
        additional_scalars: Dictionary of other scalar metrics
        available_tags: Set of all available scalar tags
    """
    loss: Dict[int, float] = field(default_factory=dict)
    learning_rate: Dict[int, float] = field(default_factory=dict)
    exploration_rate: Dict[int, float] = field(default_factory=dict)
    additional_scalars: Dict[str, Dict[int, float]] = field(default_factory=dict)
    available_tags: set = field(default_factory=set)

    def get_as_arrays(self, tag: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get scalar data as numpy arrays.

        Args:
            tag: Scalar tag name ('loss', 'learning_rate', 'exploration_rate', or custom)

        Returns:
            Tuple of (steps, values) as numpy arrays

        Raises:
            ValueError: If tag is not found
        """
        if tag == "loss":
            data = self.loss
        elif tag == "learning_rate":
            data = self.learning_rate
        elif tag == "exploration_rate":
            data = self.exploration_rate
        elif tag in self.additional_scalars:
            data = self.additional_scalars[tag]
        else:
            raise ValueError(f"Tag '{tag}' not found in TensorBoard data")

        if not data:
            return np.array([]), np.array([])

        steps = np.array(sorted(data.keys()))
        values = np.array([data[s] for s in steps])
        return steps, values


class TensorBoardLoader:
    """
    Loader for TensorBoard event files.

    Extracts scalar metrics from TensorBoard logs, handling both
    'tensorboard/' and 'logs/' directory structures.
    """

    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize the TensorBoardLoader.

        Args:
            log_dir: Path to TensorBoard log directory or parent directory

        Raises:
            FileNotFoundError: If log directory doesn't exist
        """
        self.log_dir = Path(log_dir)

        # If it's a parent directory, look for standard log directories
        if self.log_dir.is_dir():
            if (self.log_dir / "tensorboard").exists():
                self.log_dir = self.log_dir / "tensorboard"
            elif (self.log_dir / "logs").exists():
                self.log_dir = self.log_dir / "logs"

        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {self.log_dir}")

        self._event_acc: Optional[event_accumulator.EventAccumulator] = None

    def _find_event_files(self) -> List[Path]:
        """
        Find all TensorBoard event files in the log directory.

        Returns:
            List of paths to event files
        """
        event_files = []

        # Search recursively for event files
        for path in self.log_dir.rglob("events.out.tfevents.*"):
            event_files.append(path)

        return sorted(event_files)

    def _load_event_accumulator(self) -> None:
        """Load and initialize the EventAccumulator."""
        event_files = self._find_event_files()

        if not event_files:
            raise FileNotFoundError(
                f"No TensorBoard event files found in {self.log_dir}"
            )

        # Use the most recent event file or the directory containing them
        # EventAccumulator can handle a directory path
        try:
            self._event_acc = event_accumulator.EventAccumulator(
                str(self.log_dir),
                size_guidance={
                    event_accumulator.SCALARS: 0,  # Load all scalars
                }
            )
            self._event_acc.Reload()
        except Exception as e:
            raise RuntimeError(f"Error loading TensorBoard data: {e}")

    def _extract_scalar(self, tag: str) -> Dict[int, float]:
        """
        Extract scalar values for a given tag.

        Args:
            tag: Scalar tag name

        Returns:
            Dictionary mapping step to value
        """
        if self._event_acc is None:
            return {}

        try:
            events = self._event_acc.Scalars(tag)
            return {event.step: event.value for event in events}
        except KeyError:
            # Tag not found
            return {}

    def load(self, tags: Optional[List[str]] = None) -> TensorBoardData:
        """
        Load scalar data from TensorBoard logs.

        Args:
            tags: Optional list of specific tags to extract.
                 If None, extracts common tags and all available scalars.

        Returns:
            TensorBoardData object with extracted scalars
        """
        try:
            self._load_event_accumulator()
        except (FileNotFoundError, RuntimeError) as e:
            # Return empty data if TensorBoard logs are not available
            print(f"Warning: Could not load TensorBoard data: {e}")
            return TensorBoardData()

        # Get all available tags
        available_tags = set(self._event_acc.Tags().get("scalars", []))

        # Common tag variations to search for
        loss_tags = ["loss", "train/loss", "training/loss", "Loss"]
        lr_tags = ["learning_rate", "train/lr", "training/learning_rate", "lr"]
        eps_tags = ["exploration_rate", "epsilon", "train/epsilon", "eps"]

        # Extract data for common metrics
        loss_data = {}
        for tag in loss_tags:
            if tag in available_tags:
                loss_data = self._extract_scalar(tag)
                break

        lr_data = {}
        for tag in lr_tags:
            if tag in available_tags:
                lr_data = self._extract_scalar(tag)
                break

        eps_data = {}
        for tag in eps_tags:
            if tag in available_tags:
                eps_data = self._extract_scalar(tag)
                break

        # Extract additional scalars
        additional_scalars = {}
        if tags:
            # Extract specific requested tags
            for tag in tags:
                if tag in available_tags:
                    additional_scalars[tag] = self._extract_scalar(tag)
        else:
            # Extract all remaining scalars
            extracted_tags = set(loss_tags + lr_tags + eps_tags)
            for tag in available_tags:
                if tag not in extracted_tags:
                    additional_scalars[tag] = self._extract_scalar(tag)

        return TensorBoardData(
            loss=loss_data,
            learning_rate=lr_data,
            exploration_rate=eps_data,
            additional_scalars=additional_scalars,
            available_tags=available_tags
        )

    def get_available_tags(self) -> List[str]:
        """
        Get all available scalar tags in the TensorBoard logs.

        Returns:
            List of available scalar tag names
        """
        try:
            if self._event_acc is None:
                self._load_event_accumulator()
            return self._event_acc.Tags().get("scalars", [])
        except (FileNotFoundError, RuntimeError):
            return []


def load_experiment_data(
    experiment_dir: Union[str, Path],
    load_tensorboard: bool = True
) -> Tuple[TrainingConfig, TrainingMetrics, Optional[TensorBoardData]]:
    """
    Convenience function to load all data for an experiment.

    Args:
        experiment_dir: Path to experiment directory containing:
            - config.yaml
            - metrics/training_metrics.json (or detailed_rewards.json)
            - tensorboard/ or logs/ (optional)
        load_tensorboard: Whether to load TensorBoard data

    Returns:
        Tuple of (TrainingConfig, TrainingMetrics, TensorBoardData or None)

    Raises:
        FileNotFoundError: If required files are not found
    """
    experiment_dir = Path(experiment_dir)

    # Load configuration
    config_path = experiment_dir / "config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load()

    # Load metrics
    metrics_dir = experiment_dir / "metrics"
    if not metrics_dir.exists():
        # Try loading from experiment directory directly
        metrics_dir = experiment_dir

    metrics_loader = MetricsLoader(metrics_dir)
    metrics = metrics_loader.load()

    # Load TensorBoard data if requested
    tensorboard_data = None
    if load_tensorboard:
        try:
            tb_loader = TensorBoardLoader(experiment_dir)
            tensorboard_data = tb_loader.load()
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not load TensorBoard data: {e}")

    return config, metrics, tensorboard_data
