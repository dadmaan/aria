"""
Training UI Module

Provides the TrainingUI class for real-time terminal visualization
of multi-agent RL training using the Rich library.
"""

import os
import sys
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.utils.logging.logging_manager import get_logger
from src.utils.visualization.formatters import (
    create_sparkline,
    format_convergence_indicator,
    format_metric_with_trend,
)


class TrainingUI:
    """
    Rich-based terminal UI for real-time RL training visualization.

    This class provides an interactive dashboard showing training metrics,
    agent statistics, reward breakdowns, and convergence indicators.

    Attributes:
        console: Rich Console instance
        live: Rich Live display context
        config: Configuration dictionary
        enabled: Whether UI is active (TTY detection)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Training UI.

        Args:
            config: Configuration dictionary with UI settings
                - refresh_rate: Update frequency in Hz (default: 4)
                - show_sparklines: Show trend sparklines (default: True)
                - show_sequence_visual: Show sequence blocks (default: True)
                - max_history_length: Max history buffer size (default: 100)
        """
        self.logger = get_logger("training_ui")
        self.config = config or {}

        # TTY detection - disable in non-interactive environments
        self.enabled = self._detect_tty()

        if not self.enabled:
            self.logger.info("Terminal UI disabled (non-TTY environment)")
            return

        # Rich console setup
        self.console = Console()
        self.live: Optional[Live] = None

        # UI configuration
        self.refresh_rate = self.config.get("refresh_rate", 4)  # Hz
        self.show_sparklines = self.config.get("show_sparklines", True)
        self.show_sequence_visual = self.config.get("show_sequence_visual", True)
        self.max_history_length = self.config.get("max_history_length", 100)

        # Training state
        self.state = {
            "total_timesteps": 0,
            "current_episode": 0,
            "current_timestep": 0,
            "start_time": None,
            "episode_rewards": deque(maxlen=self.max_history_length),
            "episode_lengths": deque(maxlen=self.max_history_length),
            "recent_episodes": deque(maxlen=10),  # Last 10 episodes for table
            "ghsom_stats": {},
            "dqn_stats": {},
            "reward_components": {},
            "last_sequence": [],
        }

        # Thread safety
        self._lock = threading.Lock()

        self.logger.info("Terminal UI initialized successfully")

    def _detect_tty(self) -> bool:
        """
        Detect if running in a TTY environment.

        Returns:
            bool: True if TTY detected and UI should be enabled
        """
        # Check if stdout is a TTY
        if not sys.stdout.isatty():
            return False

        # Check for CI environment
        if os.environ.get("CI"):
            return False

        # Check for non-interactive Docker
        if os.environ.get("DEBIAN_FRONTEND") == "noninteractive":
            return False

        return True

    def start(self, total_timesteps: int):
        """
        Start the live display.

        Args:
            total_timesteps: Total training timesteps
        """
        if not self.enabled:
            return

        with self._lock:
            self.state["total_timesteps"] = total_timesteps
            self.state["start_time"] = datetime.now()

        try:
            layout = self._create_dashboard()
            self.live = Live(
                layout,
                console=self.console,
                refresh_per_second=self.refresh_rate,
                screen=False,
            )
            self.live.start()
            self.logger.debug("Live display started")
        except Exception as e:
            self.logger.error(f"Failed to start live display: {e}", exc_info=True)
            self.enabled = False

    def stop(self):
        """Stop the live display."""
        if not self.enabled or self.live is None:
            return

        try:
            self.live.stop()
            self.logger.debug("Live display stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop live display: {e}", exc_info=True)

    def update(self, **kwargs):
        """
        Update the UI with new metrics.

        Args:
            **kwargs: Arbitrary keyword arguments for state updates
                - episode: Current episode number
                - timestep: Current timestep
                - reward: Episode reward
                - length: Episode length
                - ghsom_stats: GHSOM agent statistics
                - dqn_stats: DQN agent statistics
                - reward_components: Reward breakdown
                - sequence: Generated sequence
        """
        if not self.enabled or self.live is None:
            return

        try:
            with self._lock:
                # Update basic counters
                if "episode" in kwargs:
                    self.state["current_episode"] = kwargs["episode"]
                if "timestep" in kwargs:
                    self.state["current_timestep"] = kwargs["timestep"]

                # Update episode metrics
                if "reward" in kwargs:
                    self.state["episode_rewards"].append(kwargs["reward"])
                if "length" in kwargs:
                    self.state["episode_lengths"].append(kwargs["length"])

                # Update recent episodes table
                if "reward" in kwargs and "length" in kwargs:
                    episode_data = {
                        "episode": self.state["current_episode"],
                        "reward": kwargs["reward"],
                        "length": kwargs["length"],
                        "timestep": self.state["current_timestep"],
                    }
                    self.state["recent_episodes"].append(episode_data)

                # Update agent stats
                if "ghsom_stats" in kwargs:
                    self.state["ghsom_stats"] = kwargs["ghsom_stats"]
                if "dqn_stats" in kwargs:
                    self.state["dqn_stats"] = kwargs["dqn_stats"]

                # Update reward components
                if "reward_components" in kwargs:
                    self.state["reward_components"] = kwargs["reward_components"]

                # Update sequence
                if "sequence" in kwargs:
                    self.state["last_sequence"] = kwargs["sequence"]

                # Update the live display
                self.live.update(self._create_dashboard())

        except Exception as e:
            self.logger.error(f"Failed to update UI: {e}", exc_info=True)

    def _create_dashboard(self) -> Layout:
        """
        Create the main dashboard layout.

        Returns:
            Layout: Rich Layout object
        """
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        # Populate sections
        layout["header"].update(self._create_header())
        layout["left"].update(self._create_metrics_panel())
        layout["right"].update(self._create_agent_panels())
        layout["footer"].update(self._create_episodes_table())

        return layout

    def _create_header(self) -> Panel:
        """Create the header panel."""
        with self._lock:
            timestep = self.state["current_timestep"]
            total = self.state["total_timesteps"]
            episode = self.state["current_episode"]

            progress_pct = (timestep / total * 100) if total > 0 else 0

            # Calculate elapsed time
            elapsed = ""
            if self.state["start_time"]:
                delta = datetime.now() - self.state["start_time"]
                hours = int(delta.total_seconds() // 3600)
                minutes = int((delta.total_seconds() % 3600) // 60)
                seconds = int(delta.total_seconds() % 60)
                elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            header_text = Text()
            header_text.append("🎵 Multi-Agent RL Training ", style="bold cyan")
            header_text.append(f"| Episode: {episode} ", style="white")
            header_text.append(f"| Timestep: {timestep:,}/{total:,} ", style="white")
            header_text.append(f"({progress_pct:.1f}%) ", style="yellow")
            if elapsed:
                header_text.append(f"| Elapsed: {elapsed}", style="green")

        return Panel(header_text, style="bold blue")

    def _create_metrics_panel(self) -> Panel:
        """Create the metrics panel with trends."""
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Current", justify="right", style="white", width=15)
        table.add_column("Mean", justify="right", style="yellow", width=15)
        table.add_column("Trend", justify="left", width=20)

        with self._lock:
            rewards = list(self.state["episode_rewards"])
            lengths = list(self.state["episode_lengths"])

            # Reward metrics
            if rewards:
                current_reward = rewards[-1]
                mean_reward = sum(rewards) / len(rewards)
                trend = create_sparkline(rewards[-20:]) if self.show_sparklines else ""
                table.add_row(
                    "Episode Reward",
                    f"{current_reward:.4f}",
                    f"{mean_reward:.4f}",
                    trend,
                )

            # Length metrics
            if lengths:
                current_length = lengths[-1]
                mean_length = sum(lengths) / len(lengths)
                trend = create_sparkline(lengths[-20:]) if self.show_sparklines else ""
                table.add_row(
                    "Episode Length",
                    f"{current_length}",
                    f"{mean_length:.1f}",
                    trend,
                )

            # Convergence indicator
            if rewards:
                convergence = format_convergence_indicator(rewards[-1], rewards)
                table.add_row(
                    "Convergence",
                    convergence,
                    "",
                    "",
                )

            # Reward components breakdown
            if self.state["reward_components"]:
                table.add_row("", "", "", "")  # Separator
                for component, value in self.state["reward_components"].items():
                    table.add_row(
                        f"  {component.capitalize()}",
                        f"{value:.4f}",
                        "",
                        "",
                    )

        return Panel(table, title="📊 Training Metrics", border_style="green")

    def _create_agent_panels(self) -> Layout:
        """Create agent-specific panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="ghsom", ratio=1),
            Layout(name="dqn", ratio=1),
        )

        # GHSOM panel
        ghsom_table = Table(show_header=False, box=None)
        ghsom_table.add_column("Metric", style="cyan")
        ghsom_table.add_column("Value", justify="right", style="white")

        with self._lock:
            ghsom_stats = self.state["ghsom_stats"]
            if ghsom_stats:
                for key, value in ghsom_stats.items():
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    elif isinstance(value, int):
                        formatted_value = f"{value:,}"
                    else:
                        formatted_value = str(value)
                    ghsom_table.add_row(formatted_key, formatted_value)

        layout["ghsom"].update(
            Panel(ghsom_table, title="🧠 GHSOM Agent", border_style="blue")
        )

        # DQN panel
        dqn_table = Table(show_header=False, box=None)
        dqn_table.add_column("Metric", style="cyan")
        dqn_table.add_column("Value", justify="right", style="white")

        with self._lock:
            dqn_stats = self.state["dqn_stats"]
            if dqn_stats:
                for key, value in dqn_stats.items():
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    elif isinstance(value, int):
                        formatted_value = f"{value:,}"
                    else:
                        formatted_value = str(value)
                    dqn_table.add_row(formatted_key, formatted_value)

        layout["dqn"].update(
            Panel(dqn_table, title="🎮 DQN Agent", border_style="yellow")
        )

        return layout

    def _create_episodes_table(self) -> Panel:
        """Create recent episodes table."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Episode", justify="right", style="cyan", width=10)
        table.add_column("Reward", justify="right", style="green", width=15)
        table.add_column("Length", justify="right", style="yellow", width=10)
        table.add_column("Timestep", justify="right", style="white", width=12)

        with self._lock:
            for episode_data in self.state["recent_episodes"]:
                table.add_row(
                    str(episode_data["episode"]),
                    f"{episode_data['reward']:.4f}",
                    str(episode_data["length"]),
                    f"{episode_data['timestep']:,}",
                )

        return Panel(table, title="📝 Recent Episodes", border_style="cyan")
