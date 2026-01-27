"""Unified logging facade for the multi-agent RL framework."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from ..config.config_loader import get_config_value

# Environment variable to globally disable WandB (set by CLI --no-wandb flag)
WANDB_DISABLED_ENV_VAR = "ARIA_WANDB_DISABLED"


class LoggingManager:
    """Unified logging manager that handles both standard logging and WandB."""

    def __init__(
        self,
        name: str = "music_rl",
        level: int = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        enable_wandb: Optional[bool] = None,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize logging manager.

        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            enable_wandb: Whether to enable WandB (reads from config if None)
            wandb_project: WandB project name
            wandb_config: WandB configuration dictionary
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_console_handler()

            if log_file:
                self._setup_file_handler(log_file)

        # Check if WandB is globally disabled via environment variable
        # This takes precedence over config settings
        wandb_globally_disabled = os.environ.get(
            WANDB_DISABLED_ENV_VAR, ""
        ).lower() in (
            "true",
            "1",
            "yes",
        )

        # WandB setup
        if wandb_globally_disabled:
            self.enable_wandb = False
        else:
            self.enable_wandb = (
                enable_wandb
                if enable_wandb is not None
                else get_config_value("enable_wandb", False)
            )
        self.wandb_initialized = False

        if self.enable_wandb and HAS_WANDB:
            self._setup_wandb(wandb_project, wandb_config)

    def _setup_console_handler(self) -> None:
        """Set up console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, log_file: Union[str, Path]) -> None:
        """Set up file logging handler.

        Args:
            log_file: Path to log file
        """
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def _setup_wandb(
        self, project: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set up WandB logging.

        Args:
            project: WandB project name
            config: WandB configuration
        """
        try:
            # Check if a WandB run is already active - reuse it instead of creating new one
            # This prevents multiple empty runs when multiple components call init
            if wandb.run is not None:
                self.wandb_initialized = True
                self.logger.info(f"Reusing existing WandB run: {wandb.run.name}")
                return

            project_name = project or get_config_value("wandb_project_name", "music_rl")

            wandb.init(project=project_name, config=config)

            self.wandb_initialized = True
            self.logger.info(f"WandB initialized for project: {project_name}")

        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.enable_wandb = False

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting
            **kwargs: Additional key-value pairs for WandB
        """
        self.logger.info(message, *args)

        if self.enable_wandb and self.wandb_initialized and kwargs:
            try:
                wandb.log(kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting
            **kwargs: Additional key-value pairs for WandB
        """
        self.logger.debug(message, *args)

        if self.enable_wandb and self.wandb_initialized and kwargs:
            try:
                wandb.log(kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting
            **kwargs: Additional key-value pairs for WandB
        """
        self.logger.warning(message, *args)

        if self.enable_wandb and self.wandb_initialized and kwargs:
            try:
                wandb.log(kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting
            **kwargs: Additional key-value pairs for WandB
        """
        self.logger.error(message, *args)

        if self.enable_wandb and self.wandb_initialized and kwargs:
            try:
                wandb.log(kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log exception message with traceback.

        Args:
            message: Log message (may contain % formatting)
            *args: Arguments for % formatting
            **kwargs: Additional key-value pairs for WandB
        """
        self.logger.exception(message, *args)

        if self.enable_wandb and self.wandb_initialized and kwargs:
            try:
                wandb.log(kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to log to WandB: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to both standard logging and WandB.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        # Log to standard logger
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_str}: {metrics_str}")

        # Log to WandB
        if self.enable_wandb and self.wandb_initialized:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to WandB: {e}")

    def log_episode(
        self, episode: int, reward: float, length: int, **additional_metrics: Any
    ) -> None:
        """Log episode information.

        Args:
            episode: Episode number
            reward: Episode reward
            length: Episode length
            **additional_metrics: Additional metrics to log
        """
        metrics = {
            "episode": episode,
            "episode_reward": reward,
            "episode_length": length,
            **additional_metrics,
        }

        self.log_metrics(metrics, step=episode)

    def finish(self) -> None:
        """Clean up logging resources."""
        if self.enable_wandb and self.wandb_initialized:
            try:
                wandb.finish()
                self.wandb_initialized = False
            except Exception as e:
                self.logger.warning(f"Failed to finish WandB: {e}")


# Global logging manager instances cache
_logger_cache: Dict[str, LoggingManager] = {}


def get_logger(name: str = "music_rl", **kwargs: Any) -> LoggingManager:
    """Get or create logging manager for the given name.

    Args:
        name: Logger name
        **kwargs: Additional arguments for LoggingManager (only used on first call for each name)

    Returns:
        LoggingManager instance
    """
    if name not in _logger_cache:
        _logger_cache[name] = LoggingManager(name, **kwargs)

    return _logger_cache[name]


def setup_logging(
    name: str = "music_rl",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    enable_wandb: Optional[bool] = None,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> LoggingManager:
    """Set up global logging configuration.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        enable_wandb: Whether to enable WandB
        wandb_project: WandB project name
        wandb_config: WandB configuration

    Returns:
        Configured LoggingManager instance
    """
    global _global_logger

    _global_logger = LoggingManager(
        name=name,
        level=level,
        log_file=log_file,
        enable_wandb=enable_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
    )

    return _global_logger
