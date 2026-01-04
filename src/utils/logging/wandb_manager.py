"""
Centralized WandB Session Manager - Singleton Pattern

This module provides a centralized manager for WandB (Weights & Biases) sessions,
ensuring only one WandB run exists across the entire codebase. This prevents
multiple empty runs that occur when different components call wandb.init().

Usage:
    from src.utils.logging.wandb_manager import WandBManager

    # Initialize or get existing session
    manager = WandBManager()
    run = manager.init(project="my_project", config=my_config)

    # Log metrics (automatically checks if run is active)
    manager.log({"loss": 0.5, "accuracy": 0.9}, step=100)

    # Finish run when done
    manager.finish()

Best Practices:
    - Call init() once at the start of training (typically in run_training.py)
    - Use manager.log() or access wandb.run directly for logging
    - Call finish() only at the end of the entire training process
    - Other components should check wandb.run is not None before logging
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import wandb
    from wandb.sdk.wandb_run import Run

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    Run = None


class WandBManager:
    """
    Singleton manager for WandB sessions.

    Ensures only one WandB run is active at a time across the entire application.
    Provides centralized control over run initialization, logging, and cleanup.

    Attributes:
        _instance: Singleton instance
        _run: Active WandB run object
        _initialized: Whether manager has been initialized
        _project: Current project name
        _config: Current run configuration
    """

    _instance: Optional["WandBManager"] = None
    _run: Optional[Any] = None  # wandb.Run type when available
    _initialized: bool = False
    _project: Optional[str] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "WandBManager":
        """Create or return existing singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = logging.getLogger(__name__)
        return cls._instance

    @property
    def is_available(self) -> bool:
        """Check if WandB is installed and available."""
        return WANDB_AVAILABLE

    @property
    def is_active(self) -> bool:
        """Check if a WandB run is currently active."""
        if not WANDB_AVAILABLE:
            return False
        return wandb.run is not None

    @property
    def run(self) -> Optional[Any]:
        """Get the current active WandB run."""
        if not WANDB_AVAILABLE:
            return None
        return wandb.run

    @property
    def run_name(self) -> Optional[str]:
        """Get the name of the current run."""
        if self.is_active:
            return wandb.run.name
        return None

    @property
    def run_id(self) -> Optional[str]:
        """Get the ID of the current run."""
        if self.is_active:
            return wandb.run.id
        return None

    def init(
        self,
        project: str,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        mode: Optional[str] = None,
        force_new: bool = False,
    ) -> Optional[Any]:
        """
        Initialize a WandB run or return existing one.

        This method implements the singleton pattern for WandB runs:
        - If a run is already active and force_new=False, returns the existing run
        - If force_new=True, finishes any existing run and creates a new one
        - If no run exists, creates a new one

        Args:
            project: WandB project name
            config: Configuration dictionary to log with the run
            name: Human-readable run name (auto-generated if None)
            tags: List of tags for the run
            notes: Description/notes for the run
            group: Group name for organizing runs
            job_type: Type of job (e.g., "train", "eval")
            mode: WandB mode ("online", "offline", "disabled")
            force_new: If True, finish existing run and create new one

        Returns:
            Active WandB run object, or None if WandB unavailable
        """
        if not WANDB_AVAILABLE:
            self._logger.warning("WandB is not installed. Skipping initialization.")
            return None

        # Check if run already exists
        if wandb.run is not None:
            if force_new:
                self._logger.info(
                    f"Finishing existing WandB run '{wandb.run.name}' to create new one"
                )
                self.finish()
            else:
                self._logger.info(
                    f"Reusing existing WandB run: {wandb.run.name} (id: {wandb.run.id})"
                )
                # Update config if provided
                if config:
                    wandb.config.update(config, allow_val_change=True)
                return wandb.run

        # Create new run
        try:
            self._run = wandb.init(
                project=project,
                config=config,
                name=name,
                tags=tags,
                notes=notes,
                group=group,
                job_type=job_type,
                mode=mode,
            )
            self._initialized = True
            self._project = project
            self._config = config

            self._logger.info(
                f"WandB run initialized: {self._run.name} (id: {self._run.id})"
            )
            return self._run

        except Exception as e:
            self._logger.error(f"Failed to initialize WandB: {e}")
            return None

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> bool:
        """
        Log metrics to WandB.

        Safely logs data only if a run is active.

        Args:
            data: Dictionary of metrics to log
            step: Step number for the logged data
            commit: Whether to commit the data immediately

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.is_active:
            return False

        try:
            wandb.log(data, step=step, commit=commit)
            return True
        except Exception as e:
            self._logger.warning(f"Failed to log to WandB: {e}")
            return False

    def log_artifact(
        self,
        artifact_path: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Log an artifact (model, dataset, etc.) to WandB.

        Args:
            artifact_path: Path to the artifact file/directory
            name: Name for the artifact
            artifact_type: Type of artifact (e.g., "model", "dataset")
            metadata: Additional metadata for the artifact

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.is_active:
            return False

        try:
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
            self._logger.info(f"Logged artifact '{name}' to WandB")
            return True
        except Exception as e:
            self._logger.warning(f"Failed to log artifact to WandB: {e}")
            return False

    def update_config(
        self, config: Dict[str, Any], allow_val_change: bool = True
    ) -> bool:
        """
        Update the run configuration.

        Args:
            config: Configuration updates
            allow_val_change: Whether to allow changing existing values

        Returns:
            True if update succeeded, False otherwise
        """
        if not self.is_active:
            return False

        try:
            wandb.config.update(config, allow_val_change=allow_val_change)
            return True
        except Exception as e:
            self._logger.warning(f"Failed to update WandB config: {e}")
            return False

    def log_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Log summary metrics (final metrics for the run).

        Args:
            summary: Summary metrics dictionary

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.is_active:
            return False

        try:
            for key, value in summary.items():
                wandb.run.summary[key] = value
            return True
        except Exception as e:
            self._logger.warning(f"Failed to log summary to WandB: {e}")
            return False

    def finish(self, exit_code: Optional[int] = None, quiet: bool = False) -> bool:
        """
        Finish the current WandB run.

        Args:
            exit_code: Exit code for the run (0 for success)
            quiet: Whether to suppress finish messages

        Returns:
            True if finish succeeded, False otherwise
        """
        if not self.is_active:
            return False

        try:
            run_name = wandb.run.name
            wandb.finish(exit_code=exit_code, quiet=quiet)
            self._run = None
            self._initialized = False
            if not quiet:
                self._logger.info(f"WandB run '{run_name}' finished")
            return True
        except Exception as e:
            self._logger.warning(f"Failed to finish WandB run: {e}")
            return False

    def watch_model(
        self,
        model: Any,
        criterion: Optional[Any] = None,
        log: str = "gradients",
        log_freq: int = 1000,
    ) -> bool:
        """
        Watch a PyTorch model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch
            criterion: Loss function (optional)
            log: What to log ("gradients", "parameters", "all", None)
            log_freq: Frequency of logging

        Returns:
            True if watch succeeded, False otherwise
        """
        if not self.is_active:
            return False

        try:
            wandb.watch(model, criterion=criterion, log=log, log_freq=log_freq)
            self._logger.info(f"Watching model with log={log}, log_freq={log_freq}")
            return True
        except Exception as e:
            self._logger.warning(f"Failed to watch model: {e}")
            return False

    @classmethod
    def get_instance(cls) -> "WandBManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when starting a completely new training session.
        Finishes any active run before resetting.
        """
        if cls._instance is not None:
            cls._instance.finish()
            cls._instance = None


# Convenience function for quick access
def get_wandb_manager() -> WandBManager:
    """Get the WandB manager singleton instance."""
    return WandBManager.get_instance()
