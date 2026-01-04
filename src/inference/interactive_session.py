#!/usr/bin/env python3
"""Interactive Inference Session with Human-in-the-Loop feedback.

This module provides the InteractiveInferenceSession class for running
interactive music generation sessions with human feedback collection.

Features:
    - Multi-dimensional feedback (quality, coherence, creativity, musicality)
    - MIDI playback integration
    - Feedback history export (JSON/CSV)
    - Preference-based regeneration
    - Visual sequence display
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import gymnasium as gym

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.progress import Progress, BarColumn, TextColumn

from src.training.tianshou_trainer import TianshouTrainer
from src.ghsom_manager import GHSOMManager
from src.environments.music_env_gym import MusicGenerationGymEnv
from src.utils.human.human_feedback import HumanFeedbackCollector
from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class MultidimensionalFeedback:
    """Multi-dimensional human feedback on a generated sequence."""

    quality: Optional[float] = None  # Overall quality (1-5)
    coherence: Optional[float] = None  # Transition coherence (1-5)
    creativity: Optional[float] = None  # Novelty vs repetition (1-5)
    musicality: Optional[float] = None  # Pleasant sounding (1-5)
    overall: Optional[float] = None  # Quick overall rating
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    skipped: bool = False

    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted feedback score.

        Args:
            weights: Weight for each dimension. Defaults to equal weights.

        Returns:
            Weighted average score (0-5 scale).
        """
        if weights is None:
            weights = {
                "quality": 0.4,
                "coherence": 0.3,
                "creativity": 0.2,
                "musicality": 0.1,
            }

        total_weight = 0.0
        total_score = 0.0

        for dim, weight in weights.items():
            value = getattr(self, dim, None)
            if value is not None:
                total_score += value * weight
                total_weight += weight

        if total_weight == 0:
            return self.overall or 0.0

        return total_score / total_weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class InteractiveResult:
    """Result from an interactive inference iteration."""

    iteration: int
    sequence: List[int]
    episode_reward: float
    reward_components: Dict[str, float]
    steps: int
    generation_time: float
    feedback: Optional[MultidimensionalFeedback] = None
    regenerated: bool = False
    previous_sequence: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "iteration": self.iteration,
            "sequence": self.sequence,
            "episode_reward": self.episode_reward,
            "reward_components": self.reward_components,
            "steps": self.steps,
            "generation_time": self.generation_time,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "regenerated": self.regenerated,
            "previous_sequence": self.previous_sequence,
        }


class MultidimensionalFeedbackCollector(HumanFeedbackCollector):
    """Extended feedback collector supporting multi-dimensional ratings."""

    def __init__(
        self,
        timeout: int = 30,
        non_interactive_mode: bool = False,
        default_feedback: float = 3.0,
        cli_feedback: Optional[float] = None,
        feedback_dimensions: Optional[List[str]] = None,
    ):
        """Initialize multi-dimensional feedback collector.

        Args:
            timeout: Timeout in seconds for input.
            non_interactive_mode: If True, return defaults.
            default_feedback: Default value for each dimension.
            cli_feedback: Pre-provided overall feedback for testing.
            feedback_dimensions: List of dimensions to collect.
        """
        super().__init__(timeout, non_interactive_mode, default_feedback, cli_feedback)

        self.dimensions = feedback_dimensions or [
            "quality",
            "coherence",
            "creativity",
            "musicality",
        ]

    def collect_multidimensional_feedback(
        self,
        quick_mode: bool = True,
    ) -> MultidimensionalFeedback:
        """Collect multi-dimensional feedback from user.

        Args:
            quick_mode: If True, only ask for overall rating.

        Returns:
            MultidimensionalFeedback object with ratings.
        """
        feedback = MultidimensionalFeedback()

        # Non-interactive mode
        if self.non_interactive_mode:
            feedback.overall = self.default_feedback
            feedback.skipped = True
            return feedback

        # CLI-provided feedback
        if self.cli_feedback is not None:
            feedback.overall = float(self.cli_feedback)
            return feedback

        try:
            if quick_mode:
                # Quick single rating
                rating = self._collect_rating(
                    "Rate this sequence (1-5, or 0 to skip): ", min_val=0, max_val=5
                )
                if rating == 0:
                    feedback.skipped = True
                else:
                    feedback.overall = rating
            else:
                # Full multi-dimensional feedback
                console.print(
                    "\n[bold cyan]Please rate the following dimensions (1-5):[/bold cyan]"
                )
                console.print("(Enter 0 to skip a dimension)\n")

                for dim in self.dimensions:
                    prompt = f"  {dim.capitalize()}: "
                    rating = self._collect_rating(prompt, min_val=0, max_val=5)
                    if rating > 0:
                        setattr(feedback, dim, rating)

                # Also collect overall if detailed provided
                overall = self._collect_rating(
                    "\n  Overall rating: ", min_val=0, max_val=5
                )
                if overall > 0:
                    feedback.overall = overall

        except KeyboardInterrupt:
            console.print("\n[yellow]Feedback collection interrupted[/yellow]")
            feedback.skipped = True

        return feedback

    def _collect_rating(
        self,
        prompt: str,
        min_val: int = 1,
        max_val: int = 5,
    ) -> float:
        """Collect a single rating value.

        Args:
            prompt: Prompt message.
            min_val: Minimum valid value.
            max_val: Maximum valid value.

        Returns:
            Rating value or default.
        """
        try:
            value = FloatPrompt.ask(
                prompt,
                default=str(self.default_feedback),
            )
            value = float(value)

            if min_val <= value <= max_val:
                return value
            else:
                console.print(f"[yellow]Value must be {min_val}-{max_val}[/yellow]")
                return self.default_feedback

        except (ValueError, KeyboardInterrupt):
            return self.default_feedback

    def simulate_multidimensional_feedback(
        self,
        reward: float,
        seed: Optional[int] = None,
    ) -> MultidimensionalFeedback:
        """Generate simulated multi-dimensional feedback based on reward.

        Args:
            reward: Episode reward to base simulation on.
            seed: Random seed for reproducibility.

        Returns:
            Simulated MultidimensionalFeedback.
        """
        if seed is not None:
            np.random.seed(seed)

        # Base score from normalized reward
        base_score = np.clip((reward + 5) / 2, 1, 5)  # Map reward to 1-5

        feedback = MultidimensionalFeedback()

        # Add noise to each dimension
        for dim in self.dimensions:
            noise = np.random.normal(0, 0.5)
            score = np.clip(base_score + noise, 1, 5)
            setattr(feedback, dim, round(score, 1))

        # Overall is slightly different
        feedback.overall = round(
            np.clip(base_score + np.random.normal(0, 0.3), 1, 5), 1
        )

        return feedback


class MIDIPlayback:
    """MIDI playback utility for sequence audio preview."""

    def __init__(
        self,
        soundfont_path: Optional[str] = None,
        tempo: int = 120,
        output_dir: Optional[Path] = None,
    ):
        """Initialize MIDI playback.

        Args:
            soundfont_path: Path to SoundFont file for synthesis.
            tempo: Default tempo in BPM.
            output_dir: Directory to save MIDI files.
        """
        self.soundfont_path = soundfont_path
        self.tempo = tempo
        self.output_dir = (
            Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        )
        self._check_playback_available()

    def _check_playback_available(self) -> bool:
        """Check if MIDI playback tools are available."""
        self.playback_method = None

        # Check for timidity
        try:
            result = subprocess.run(
                ["which", "timidity"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                self.playback_method = "timidity"
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Check for fluidsynth
        try:
            result = subprocess.run(
                ["which", "fluidsynth"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                self.playback_method = "fluidsynth"
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.warning("No MIDI playback tool available (timidity or fluidsynth)")
        return False

    def sequence_to_midi(
        self,
        sequence: List[int],
        ghsom_manager: GHSOMManager,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Convert cluster sequence to MIDI file.

        Args:
            sequence: List of cluster IDs.
            ghsom_manager: GHSOM manager for cluster info.
            output_path: Optional output path.

        Returns:
            Path to generated MIDI file, or None if failed.
        """
        try:
            from midiutil import MIDIFile
        except ImportError:
            logger.warning("midiutil not installed, cannot create MIDI")
            return None

        # Create MIDI file
        midi = MIDIFile(1)  # One track
        track = 0
        channel = 0
        time = 0
        duration = 1  # 1 beat per cluster
        volume = 100

        midi.addTempo(track, 0, self.tempo)

        # Map clusters to MIDI notes
        for i, cluster_id in enumerate(sequence):
            # Map cluster to pitch (simple linear mapping)
            # Could be improved with feature-based mapping
            pitch = 48 + (cluster_id % 36)  # C3 to B5 range
            midi.addNote(track, channel, pitch, time + i, duration, volume)

        # Save MIDI file
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"sequence_{timestamp}.mid"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            midi.writeFile(f)

        logger.info(f"MIDI saved to: {output_path}")
        return output_path

    def play_midi(self, midi_path: Path, timeout: int = 30) -> bool:
        """Play MIDI file using system audio.

        Args:
            midi_path: Path to MIDI file.
            timeout: Playback timeout in seconds.

        Returns:
            True if playback successful, False otherwise.
        """
        if not midi_path.exists():
            logger.error(f"MIDI file not found: {midi_path}")
            return False

        if self.playback_method is None:
            console.print(f"[yellow]MIDI saved to: {midi_path}[/yellow]")
            console.print(
                "[dim]No MIDI player available. Install timidity or fluidsynth.[/dim]"
            )
            return False

        try:
            if self.playback_method == "timidity":
                cmd = ["timidity", str(midi_path)]
            elif self.playback_method == "fluidsynth":
                if self.soundfont_path:
                    cmd = ["fluidsynth", "-ni", self.soundfont_path, str(midi_path)]
                else:
                    cmd = ["fluidsynth", "-ni", str(midi_path)]
            else:
                return False

            console.print(f"[dim]Playing MIDI with {self.playback_method}...[/dim]")
            subprocess.run(cmd, timeout=timeout, check=True)
            return True

        except subprocess.TimeoutExpired:
            console.print("[yellow]Playback timed out[/yellow]")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Playback failed: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"{self.playback_method} not found")
            return False

    def play_sequence(
        self,
        sequence: List[int],
        ghsom_manager: GHSOMManager,
        timeout: int = 30,
    ) -> bool:
        """Convert sequence to MIDI and play.

        Args:
            sequence: List of cluster IDs.
            ghsom_manager: GHSOM manager.
            timeout: Playback timeout.

        Returns:
            True if playback successful.
        """
        midi_path = self.sequence_to_midi(sequence, ghsom_manager)
        if midi_path:
            return self.play_midi(midi_path, timeout)
        return False


class InteractiveInferenceSession:
    """Interactive inference session with human-in-the-loop feedback.

    This class manages interactive music generation sessions where users
    can provide feedback on generated sequences, request regeneration,
    and build up preference data.

    Example:
        >>> session = InteractiveInferenceSession(trainer, env, ghsom_manager, config)
        >>> results = session.run_session(num_iterations=10)
        >>> session.export_history("outputs/hil_session/")
    """

    def __init__(
        self,
        trainer: TianshouTrainer,
        env: gym.Env,
        ghsom_manager: GHSOMManager,
        config: Dict[str, Any],
    ):
        """Initialize interactive inference session.

        Args:
            trainer: Trained TianshouTrainer instance.
            env: Environment for sequence generation.
            ghsom_manager: GHSOM manager for cluster operations.
            config: Configuration dictionary.
        """
        self.trainer = trainer
        self.env = env
        self.ghsom_manager = ghsom_manager
        self.config = config

        # Parse config
        hil_config = config.get("human_feedback", {})

        # Initialize feedback collector
        self.feedback_collector = MultidimensionalFeedbackCollector(
            timeout=hil_config.get("timeout", config.get("human_feedback_timeout", 30)),
            non_interactive_mode=hil_config.get(
                "non_interactive", config.get("non_interactive_mode", False)
            ),
            feedback_dimensions=self._get_feedback_dimensions(hil_config),
        )

        # Initialize MIDI playback if enabled
        playback_config = hil_config.get("playback", {})
        if playback_config.get("enabled", config.get("render_midi", False)):
            self.midi_playback = MIDIPlayback(
                soundfont_path=playback_config.get("soundfont"),
                tempo=playback_config.get("tempo", 120),
            )
        else:
            self.midi_playback = None

        # Session history
        self.results: List[InteractiveResult] = []
        self.feedback_history: List[MultidimensionalFeedback] = []
        self.problematic_clusters: Dict[int, int] = {}  # cluster_id -> negative count

        # Session metadata
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

    def _get_feedback_dimensions(self, hil_config: Dict[str, Any]) -> List[str]:
        """Get feedback dimensions from config."""
        dims = hil_config.get("dimensions", [])
        if isinstance(dims, list) and len(dims) > 0:
            # Handle both string list and dict list
            if isinstance(dims[0], dict):
                return [d.get("name", "") for d in dims if d.get("name")]
            return dims
        return ["quality", "coherence", "creativity", "musicality"]

    def generate_sequence(
        self,
        deterministic: bool = True,
        avoid_clusters: Optional[List[int]] = None,
    ) -> Tuple[List[int], float, Dict[str, float], int, float]:
        """Generate a single sequence.

        Args:
            deterministic: Use greedy action selection.
            avoid_clusters: List of cluster IDs to penalize.

        Returns:
            Tuple of (sequence, reward, reward_components, steps, time).
        """
        start_time = time.time()

        obs, info = self.env.reset()
        episode_reward = 0.0
        steps = 0
        hidden_state = None

        self.trainer.network.eval()

        # Check for distributional network
        is_distributional = hasattr(self.trainer.network, "num_atoms")
        if is_distributional:
            v_min = getattr(self.trainer.network, "v_min", -10.0)
            v_max = getattr(self.trainer.network, "v_max", 10.0)
            num_atoms = self.trainer.network.num_atoms
            support = torch.linspace(v_min, v_max, num_atoms).to(self.trainer.device)

        with torch.no_grad():
            while True:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.trainer.device)

                # Handle recurrent networks
                if self.trainer.network.is_recurrent:
                    if hidden_state is None:
                        hidden_state = self.trainer.network.get_initial_state(1)
                    output, hidden_state = self.trainer.network(
                        obs_tensor, hidden_state
                    )
                else:
                    output = self.trainer.network(obs_tensor)

                # Convert to Q-values if distributional
                if is_distributional:
                    q_values = (output * support.view(1, 1, -1)).sum(dim=2)
                else:
                    q_values = output

                # Apply penalty to avoided clusters
                if avoid_clusters:
                    for cluster_id in avoid_clusters:
                        if cluster_id < q_values.shape[1]:
                            q_values[0, cluster_id] -= 10.0

                # Action selection
                if deterministic:
                    action = q_values.argmax(dim=1).item()
                else:
                    eps = 0.1  # Small exploration
                    if np.random.random() < eps:
                        action = self.env.action_space.sample()
                    else:
                        action = q_values.argmax(dim=1).item()

                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

        generation_time = time.time() - start_time

        sequence = info.get("sequence", [])
        if hasattr(sequence, "tolist"):
            sequence = [x for x in sequence.tolist() if x >= 0]

        reward_components = info.get("reward_components", {})

        return sequence, episode_reward, reward_components, steps, generation_time

    def display_sequence(
        self,
        result: InteractiveResult,
    ) -> None:
        """Display sequence information in terminal.

        Args:
            result: InteractiveResult to display.
        """
        # Create sequence visualization
        seq_str = " → ".join([str(c) for c in result.sequence])
        unique_count = len(set(result.sequence))

        # Build table
        table = Table(title=f"Iteration {result.iteration + 1}", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Sequence", seq_str)
        table.add_row("Length", str(len(result.sequence)))
        table.add_row(
            "Unique Clusters",
            f"{unique_count} ({100*unique_count/len(result.sequence):.1f}%)",
        )
        table.add_row("Episode Reward", f"{result.episode_reward:.3f}")

        if result.reward_components:
            for comp, value in result.reward_components.items():
                table.add_row(f"  {comp}", f"{value:.3f}")

        table.add_row("Generation Time", f"{result.generation_time:.3f}s")

        if result.regenerated:
            table.add_row("[yellow]Regenerated[/yellow]", "Yes")

        console.print(table)

    def run_single_iteration(
        self,
        iteration: int,
        quick_feedback: bool = True,
        enable_playback: bool = True,
        allow_regeneration: bool = True,
    ) -> InteractiveResult:
        """Run a single interactive iteration.

        Args:
            iteration: Iteration number.
            quick_feedback: Use quick single-rating feedback.
            enable_playback: Enable MIDI playback.
            allow_regeneration: Allow user to request regeneration.

        Returns:
            InteractiveResult for this iteration.
        """
        # Generate sequence (avoid problematic clusters from history)
        avoid_clusters = [
            cid for cid, count in self.problematic_clusters.items() if count >= 2
        ]

        sequence, reward, components, steps, gen_time = self.generate_sequence(
            deterministic=True,
            avoid_clusters=avoid_clusters if avoid_clusters else None,
        )

        result = InteractiveResult(
            iteration=iteration,
            sequence=sequence,
            episode_reward=reward,
            reward_components=components,
            steps=steps,
            generation_time=gen_time,
        )

        # Display sequence
        self.display_sequence(result)

        # Optional MIDI playback
        if enable_playback and self.midi_playback:
            play = Prompt.ask("\nPlay MIDI preview?", choices=["y", "n"], default="n")
            if play.lower() == "y":
                self.midi_playback.play_sequence(
                    sequence, self.ghsom_manager, timeout=30
                )

        # Collect feedback
        feedback = self.feedback_collector.collect_multidimensional_feedback(
            quick_mode=quick_feedback
        )
        result.feedback = feedback
        self.feedback_history.append(feedback)

        # Display feedback received
        if not feedback.skipped:
            score = feedback.weighted_score()
            console.print(f"\n[green]Feedback recorded: {score:.1f}/5.0[/green]")

        # Check if regeneration needed/requested
        if allow_regeneration and not feedback.skipped:
            score = feedback.weighted_score()
            if score is not None and score <= 2.0:
                regenerate = Prompt.ask(
                    "Low rating - regenerate sequence?", choices=["y", "n"], default="y"
                )
                if regenerate.lower() == "y":
                    # Track problematic clusters
                    for cluster_id in sequence:
                        self.problematic_clusters[cluster_id] = (
                            self.problematic_clusters.get(cluster_id, 0) + 1
                        )

                    # Generate new sequence
                    result.previous_sequence = result.sequence
                    sequence, reward, components, steps, gen_time = (
                        self.generate_sequence(
                            deterministic=True,
                            avoid_clusters=list(
                                set(sequence)
                            ),  # Avoid all clusters from bad sequence
                        )
                    )

                    result.sequence = sequence
                    result.episode_reward = reward
                    result.reward_components = components
                    result.steps = steps
                    result.generation_time += gen_time
                    result.regenerated = True

                    console.print("\n[cyan]Regenerated sequence:[/cyan]")
                    self.display_sequence(result)

        return result

    def run_session(
        self,
        num_iterations: int = 10,
        quick_feedback: bool = True,
        enable_playback: bool = False,
        allow_regeneration: bool = True,
    ) -> List[InteractiveResult]:
        """Run full interactive session.

        Args:
            num_iterations: Number of sequences to generate.
            quick_feedback: Use quick rating mode.
            enable_playback: Enable MIDI playback.
            allow_regeneration: Allow regeneration requests.

        Returns:
            List of InteractiveResult objects.
        """
        console.print(
            Panel(
                "[bold cyan]Interactive Inference Session[/bold cyan]\n\n"
                f"Session ID: {self.session_id}\n"
                f"Iterations: {num_iterations}\n"
                f"Mode: {'Quick' if quick_feedback else 'Detailed'} feedback\n"
                f"Playback: {'Enabled' if enable_playback and self.midi_playback else 'Disabled'}\n\n"
                "You will be shown generated sequences and asked to rate them.\n"
                "Enter a rating from 1 (poor) to 5 (excellent), or 0 to skip.\n"
                "Press Ctrl+C to end the session early.",
                title="Human-in-the-Loop Mode",
            )
        )

        try:
            for i in range(num_iterations):
                console.print(f"\n{'='*60}")
                console.print(f"[bold]Iteration {i+1}/{num_iterations}[/bold]")
                console.print(f"{'='*60}")

                result = self.run_single_iteration(
                    iteration=i,
                    quick_feedback=quick_feedback,
                    enable_playback=enable_playback,
                    allow_regeneration=allow_regeneration,
                )
                self.results.append(result)

        except KeyboardInterrupt:
            console.print("\n[yellow]Session ended by user[/yellow]")

        # Print session summary
        self._print_session_summary()

        return self.results

    def _print_session_summary(self) -> None:
        """Print summary of session results."""
        if not self.results:
            return

        console.print("\n" + "=" * 60)
        console.print("[bold cyan]Session Summary[/bold cyan]")
        console.print("=" * 60)

        # Aggregate stats
        rewards = [r.episode_reward for r in self.results]
        feedbacks = [
            r.feedback.weighted_score()
            for r in self.results
            if r.feedback and not r.feedback.skipped
        ]
        regenerated_count = sum(1 for r in self.results if r.regenerated)

        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Sequences", str(len(self.results)))
        table.add_row(
            "Avg Episode Reward", f"{np.mean(rewards):.3f} ± {np.std(rewards):.3f}"
        )

        if feedbacks:
            table.add_row(
                "Avg Human Rating",
                f"{np.mean(feedbacks):.2f} ± {np.std(feedbacks):.2f}",
            )
            table.add_row("Rated Sequences", str(len(feedbacks)))

        table.add_row("Regenerated", str(regenerated_count))
        table.add_row("Problematic Clusters", str(len(self.problematic_clusters)))

        console.print(table)

    def export_history(
        self,
        output_dir: Union[str, Path],
        export_csv: bool = True,
    ) -> None:
        """Export session history to files.

        Args:
            output_dir: Directory to save files.
            export_csv: Also export CSV format.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export JSON
        json_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "config": {
                "checkpoint": self.config.get("checkpoint_path", "unknown"),
                "num_iterations": len(self.results),
            },
            "results": [r.to_dict() for r in self.results],
            "aggregate_metrics": self._calculate_aggregate_metrics(),
            "problematic_clusters": dict(self.problematic_clusters),
        }

        json_path = output_dir / f"hil_session_{self.session_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Session history saved to: {json_path}")

        # Export CSV if requested
        if export_csv:
            csv_path = output_dir / f"hil_feedback_{self.session_id}.csv"
            self._export_feedback_csv(csv_path)

    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics for session."""
        if not self.results:
            return {}

        rewards = [r.episode_reward for r in self.results]
        feedbacks = [
            r.feedback.weighted_score()
            for r in self.results
            if r.feedback and not r.feedback.skipped and r.feedback.weighted_score() > 0
        ]

        metrics = {
            "num_sequences": len(self.results),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "num_rated": len(feedbacks),
            "num_skipped": len(self.results) - len(feedbacks),
            "num_regenerated": sum(1 for r in self.results if r.regenerated),
        }

        if feedbacks:
            metrics["avg_human_rating"] = float(np.mean(feedbacks))
            metrics["std_human_rating"] = float(np.std(feedbacks))

        return metrics

    def _export_feedback_csv(self, csv_path: Path) -> None:
        """Export feedback data to CSV."""
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "iteration",
                    "episode_reward",
                    "sequence_length",
                    "unique_clusters",
                    "feedback_quality",
                    "feedback_coherence",
                    "feedback_creativity",
                    "feedback_musicality",
                    "feedback_overall",
                    "feedback_weighted",
                    "skipped",
                    "regenerated",
                ]
            )

            # Data rows
            for result in self.results:
                fb = result.feedback
                writer.writerow(
                    [
                        result.iteration,
                        result.episode_reward,
                        len(result.sequence),
                        len(set(result.sequence)),
                        fb.quality if fb else None,
                        fb.coherence if fb else None,
                        fb.creativity if fb else None,
                        fb.musicality if fb else None,
                        fb.overall if fb else None,
                        fb.weighted_score() if fb and not fb.skipped else None,
                        fb.skipped if fb else True,
                        result.regenerated,
                    ]
                )

        logger.info(f"Feedback CSV saved to: {csv_path}")


def create_interactive_session(
    checkpoint_path: str,
    config: Dict[str, Any],
    env: gym.Env,
    ghsom_manager: GHSOMManager,
) -> InteractiveInferenceSession:
    """Factory function to create interactive session.

    Args:
        checkpoint_path: Path to trained checkpoint.
        config: Configuration dictionary.
        env: Gymnasium environment.
        ghsom_manager: GHSOM manager instance.

    Returns:
        Configured InteractiveInferenceSession.
    """
    from src.training.tianshou_trainer import TianshouTrainer

    # Load trainer
    trainer = TianshouTrainer.load(checkpoint_path, env=env, config=None)

    # Add checkpoint path to config for export
    config["checkpoint_path"] = checkpoint_path

    return InteractiveInferenceSession(
        trainer=trainer,
        env=env,
        ghsom_manager=ghsom_manager,
        config=config,
    )
