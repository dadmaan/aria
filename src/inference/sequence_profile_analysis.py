#!/usr/bin/env python3
"""Sequence Profile Analysis - Enriched Analysis Combining Sequences and Cluster Profiles.

This module provides the SequenceProfileAnalyzer class that extends sequence analysis
with cluster profile metadata, enabling rich analysis of musical characteristics,
arousal patterns, instrument usage, roles, and musical features.

Features:
    - Arousal distribution analysis over time
    - Instrument diversity and usage patterns
    - Role distribution analysis
    - Preference type analysis
    - Genre coverage analysis
    - Musical feature statistics (BPM, density, velocity, polyphony)
    - Profile-aware visualizations
    - Comparison methods for multiple sequence sets
    - Statistical testing for distribution differences
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.inference.sequence_analysis import (
    SequenceAnalyzer,
    DiversityMetrics,
    TransitionMetrics,
)
from src.inference.cluster_profiles import ClusterProfileLoader, ClusterProfile
from src.ghsom_manager import GHSOMManager
from src.utils.logging.logging_manager import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class ArousalMetrics:
    """Metrics for arousal distribution over sequences.

    Attributes:
        low_count: Number of low arousal clusters.
        medium_count: Number of medium arousal clusters.
        high_count: Number of high arousal clusters.
        low_ratio: Ratio of low arousal clusters.
        medium_ratio: Ratio of medium arousal clusters.
        high_ratio: Ratio of high arousal clusters.
        arousal_entropy: Entropy of arousal distribution.
        arousal_trajectory: List of arousal levels over time (per step).
    """

    low_count: int
    medium_count: int
    high_count: int
    low_ratio: float
    medium_ratio: float
    high_ratio: float
    arousal_entropy: float
    arousal_trajectory: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class InstrumentMetrics:
    """Metrics for instrument usage patterns.

    Attributes:
        unique_instruments: Number of unique instruments used.
        instrument_counts: Dictionary mapping instrument to usage count.
        instrument_ratios: Dictionary mapping instrument to usage ratio.
        dominant_instrument: Most used instrument.
        dominant_ratio: Ratio of dominant instrument.
        instrument_entropy: Entropy of instrument distribution.
        instrument_diversity: Simpson's diversity index for instruments.
    """

    unique_instruments: int
    instrument_counts: Dict[str, int]
    instrument_ratios: Dict[str, float]
    dominant_instrument: str
    dominant_ratio: float
    instrument_entropy: float
    instrument_diversity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RoleMetrics:
    """Metrics for musical role distribution.

    Attributes:
        unique_roles: Number of unique roles used.
        role_counts: Dictionary mapping role to usage count.
        role_ratios: Dictionary mapping role to usage ratio.
        dominant_role: Most used role.
        dominant_ratio: Ratio of dominant role.
        role_entropy: Entropy of role distribution.
        has_melody: Whether sequences include melody roles.
        has_accompaniment: Whether sequences include accompaniment roles.
        has_pad: Whether sequences include pad roles.
    """

    unique_roles: int
    role_counts: Dict[str, int]
    role_ratios: Dict[str, float]
    dominant_role: str
    dominant_ratio: float
    role_entropy: float
    has_melody: bool
    has_accompaniment: bool
    has_pad: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PreferenceMetrics:
    """Metrics for preference type distribution.

    Attributes:
        intrinsic_count: Number of intrinsic preference clusters.
        extrinsic_count: Number of extrinsic preference clusters.
        intrinsic_ratio: Ratio of intrinsic preference clusters.
        extrinsic_ratio: Ratio of extrinsic preference clusters.
        preference_balance: Balance score (0.5 = perfect balance).
    """

    intrinsic_count: int
    extrinsic_count: int
    intrinsic_ratio: float
    extrinsic_ratio: float
    preference_balance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenreMetrics:
    """Metrics for genre coverage and distribution.

    Attributes:
        unique_genres: Number of unique genres used.
        genre_counts: Dictionary mapping genre to usage count.
        genre_ratios: Dictionary mapping genre to usage ratio.
        dominant_genre: Most used genre.
        dominant_ratio: Ratio of dominant genre.
        genre_entropy: Entropy of genre distribution.
    """

    unique_genres: int
    genre_counts: Dict[str, int]
    genre_ratios: Dict[str, float]
    dominant_genre: str
    dominant_ratio: float
    genre_entropy: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MusicalFeatureStats:
    """Statistics for musical features from cluster profiles.

    Attributes:
        bpm_mean: Mean BPM across sequences.
        bpm_std: Standard deviation of BPM.
        bpm_min: Minimum BPM.
        bpm_max: Maximum BPM.
        density_mean: Mean note density.
        density_std: Standard deviation of density.
        velocity_mean: Mean velocity.
        velocity_std: Standard deviation of velocity.
        polyphony_mean: Mean polyphony level.
        polyphony_std: Standard deviation of polyphony.
    """

    bpm_mean: float
    bpm_std: float
    bpm_min: float
    bpm_max: float
    density_mean: float
    density_std: float
    velocity_mean: float
    velocity_std: float
    polyphony_mean: float
    polyphony_std: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProfileEnrichedMetrics:
    """Complete profile-enriched analysis metrics.

    Attributes:
        num_sequences: Number of sequences analyzed.
        avg_sequence_length: Average sequence length.
        total_steps: Total steps across all sequences.
        arousal: Arousal distribution metrics.
        instruments: Instrument usage metrics.
        roles: Role distribution metrics.
        preferences: Preference type metrics.
        genres: Genre coverage metrics.
        musical_features: Musical feature statistics.
        basic_diversity: Basic diversity metrics from SequenceAnalyzer.
        basic_transitions: Basic transition metrics from SequenceAnalyzer.
    """

    num_sequences: int
    avg_sequence_length: float
    total_steps: int
    arousal: ArousalMetrics
    instruments: InstrumentMetrics
    roles: RoleMetrics
    preferences: PreferenceMetrics
    genres: GenreMetrics
    musical_features: MusicalFeatureStats
    basic_diversity: DiversityMetrics
    basic_transitions: TransitionMetrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_sequences": self.num_sequences,
            "avg_sequence_length": self.avg_sequence_length,
            "total_steps": self.total_steps,
            "arousal": self.arousal.to_dict(),
            "instruments": self.instruments.to_dict(),
            "roles": self.roles.to_dict(),
            "preferences": self.preferences.to_dict(),
            "genres": self.genres.to_dict(),
            "musical_features": self.musical_features.to_dict(),
            "basic_diversity": self.basic_diversity.to_dict(),
            "basic_transitions": self.basic_transitions.to_dict(),
        }


class SequenceProfileAnalyzer:
    """Analyzer combining sequence analysis with cluster profile metadata.

    This class extends SequenceAnalyzer functionality with profile-aware
    analysis, providing insights into musical characteristics, arousal patterns,
    instrument usage, and more.

    Example:
        >>> profile_loader = ClusterProfileLoader("cluster_profiles.csv")
        >>> analyzer = SequenceProfileAnalyzer(profile_loader)
        >>> analyzer.load_sequences("outputs/inference/sequences.json")
        >>> metrics = analyzer.compute_all_metrics()
        >>> analyzer.generate_report("outputs/analysis/")
    """

    def __init__(
        self,
        profile_loader: ClusterProfileLoader,
        ghsom_manager: Optional[GHSOMManager] = None,
        sequence_analyzer: Optional[SequenceAnalyzer] = None,
    ):
        """Initialize profile analyzer.

        Args:
            profile_loader: Loader for cluster profiles.
            ghsom_manager: Optional GHSOM manager for hierarchy info.
            sequence_analyzer: Optional existing SequenceAnalyzer to wrap.
                If None, creates a new one.
        """
        self.profile_loader = profile_loader
        self.ghsom_manager = ghsom_manager

        # Create or use existing sequence analyzer
        if sequence_analyzer is None:
            self.sequence_analyzer = SequenceAnalyzer(
                ghsom_manager=ghsom_manager,
                total_clusters=len(profile_loader.profiles),
            )
        else:
            self.sequence_analyzer = sequence_analyzer

        # Cache for profile lookups
        self._profile_cache: Dict[int, Optional[ClusterProfile]] = {}

    def load_sequences(
        self,
        source: Union[str, Path, List[Dict[str, Any]], List[List[int]]],
    ) -> int:
        """Load sequences from file or data structure.

        Args:
            source: Path to JSON file, list of result dicts, or list of sequences.

        Returns:
            Number of sequences loaded.
        """
        return self.sequence_analyzer.load_sequences(source)

    @property
    def sequences(self) -> List[List[int]]:
        """Get loaded sequences."""
        return self.sequence_analyzer.sequences

    def _get_profile(self, cluster_id: int) -> Optional[ClusterProfile]:
        """Get profile with caching.

        Args:
            cluster_id: Cluster ID to look up.

        Returns:
            ClusterProfile or None if not found.
        """
        if cluster_id not in self._profile_cache:
            self._profile_cache[cluster_id] = self.profile_loader.get_profile(
                cluster_id
            )
        return self._profile_cache[cluster_id]

    def analyze_arousal_distribution(self) -> ArousalMetrics:
        """Analyze arousal level distribution across sequences.

        Returns:
            ArousalMetrics with distribution and trajectory.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        arousal_counts = Counter()
        arousal_trajectory = []

        # Collect arousal levels for each cluster
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    arousal = profile.arousal_level
                    arousal_counts[arousal] += 1
                    arousal_trajectory.append(arousal)
                else:
                    arousal_trajectory.append("unknown")

        total = sum(arousal_counts.values())

        if total == 0:
            logger.warning("No valid arousal data found in profiles")
            return ArousalMetrics(
                low_count=0,
                medium_count=0,
                high_count=0,
                low_ratio=0.0,
                medium_ratio=0.0,
                high_ratio=0.0,
                arousal_entropy=0.0,
                arousal_trajectory=[],
            )

        low_count = arousal_counts.get("low", 0)
        medium_count = arousal_counts.get("medium", 0)
        high_count = arousal_counts.get("high", 0)

        # Calculate entropy
        probs = []
        for count in [low_count, medium_count, high_count]:
            if count > 0:
                probs.append(count / total)

        entropy = -np.sum([p * np.log(p) for p in probs if p > 0]) if probs else 0.0

        return ArousalMetrics(
            low_count=low_count,
            medium_count=medium_count,
            high_count=high_count,
            low_ratio=low_count / total,
            medium_ratio=medium_count / total,
            high_ratio=high_count / total,
            arousal_entropy=float(entropy),
            arousal_trajectory=arousal_trajectory,
        )

    def analyze_instrument_diversity(self) -> InstrumentMetrics:
        """Analyze instrument usage and diversity.

        Returns:
            InstrumentMetrics with usage patterns.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        instrument_counts = Counter()

        # Collect instruments
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    instrument_counts[profile.dominant_instrument] += 1

        total = sum(instrument_counts.values())

        if total == 0:
            logger.warning("No valid instrument data found")
            return InstrumentMetrics(
                unique_instruments=0,
                instrument_counts={},
                instrument_ratios={},
                dominant_instrument="unknown",
                dominant_ratio=0.0,
                instrument_entropy=0.0,
                instrument_diversity=0.0,
            )

        # Calculate ratios
        instrument_ratios = {
            inst: count / total for inst, count in instrument_counts.items()
        }

        # Dominant instrument
        dominant = max(instrument_counts.items(), key=lambda x: x[1])

        # Entropy
        probs = np.array(list(instrument_ratios.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Simpson's diversity index: 1 - sum(p_i^2)
        diversity = 1 - np.sum(probs**2)

        return InstrumentMetrics(
            unique_instruments=len(instrument_counts),
            instrument_counts=dict(instrument_counts),
            instrument_ratios=instrument_ratios,
            dominant_instrument=dominant[0],
            dominant_ratio=dominant[1] / total,
            instrument_entropy=float(entropy),
            instrument_diversity=float(diversity),
        )

    def analyze_role_distribution(self) -> RoleMetrics:
        """Analyze musical role distribution.

        Returns:
            RoleMetrics with role patterns.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        role_counts = Counter()

        # Collect roles
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    role_counts[profile.dominant_role] += 1

        total = sum(role_counts.values())

        if total == 0:
            logger.warning("No valid role data found")
            return RoleMetrics(
                unique_roles=0,
                role_counts={},
                role_ratios={},
                dominant_role="unknown",
                dominant_ratio=0.0,
                role_entropy=0.0,
                has_melody=False,
                has_accompaniment=False,
                has_pad=False,
            )

        # Calculate ratios
        role_ratios = {role: count / total for role, count in role_counts.items()}

        # Dominant role
        dominant = max(role_counts.items(), key=lambda x: x[1])

        # Entropy
        probs = np.array(list(role_ratios.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Check for specific roles
        roles_present = set(role_counts.keys())
        has_melody = any("melody" in r.lower() for r in roles_present)
        has_accompaniment = any("accompaniment" in r.lower() for r in roles_present)
        has_pad = any("pad" in r.lower() for r in roles_present)

        return RoleMetrics(
            unique_roles=len(role_counts),
            role_counts=dict(role_counts),
            role_ratios=role_ratios,
            dominant_role=dominant[0],
            dominant_ratio=dominant[1] / total,
            role_entropy=float(entropy),
            has_melody=has_melody,
            has_accompaniment=has_accompaniment,
            has_pad=has_pad,
        )

    def analyze_preference_types(self) -> PreferenceMetrics:
        """Analyze preference type distribution.

        Returns:
            PreferenceMetrics with intrinsic/extrinsic balance.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        preference_counts = Counter()

        # Collect preference types
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    preference_counts[profile.preference_type] += 1

        total = sum(preference_counts.values())

        if total == 0:
            logger.warning("No valid preference data found")
            return PreferenceMetrics(
                intrinsic_count=0,
                extrinsic_count=0,
                intrinsic_ratio=0.0,
                extrinsic_ratio=0.0,
                preference_balance=0.0,
            )

        intrinsic = preference_counts.get("intrinsic", 0)
        extrinsic = preference_counts.get("extrinsic", 0)

        intrinsic_ratio = intrinsic / total
        extrinsic_ratio = extrinsic / total

        # Balance: 0.5 = perfect balance, 0 or 1 = completely skewed
        balance = 1 - abs(intrinsic_ratio - extrinsic_ratio)

        return PreferenceMetrics(
            intrinsic_count=intrinsic,
            extrinsic_count=extrinsic,
            intrinsic_ratio=intrinsic_ratio,
            extrinsic_ratio=extrinsic_ratio,
            preference_balance=balance,
        )

    def analyze_genre_coverage(self) -> GenreMetrics:
        """Analyze genre coverage and distribution.

        Returns:
            GenreMetrics with genre usage patterns.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        genre_counts = Counter()

        # Collect genres
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    genre_counts[profile.dominant_genre] += 1

        total = sum(genre_counts.values())

        if total == 0:
            logger.warning("No valid genre data found")
            return GenreMetrics(
                unique_genres=0,
                genre_counts={},
                genre_ratios={},
                dominant_genre="unknown",
                dominant_ratio=0.0,
                genre_entropy=0.0,
            )

        # Calculate ratios
        genre_ratios = {genre: count / total for genre, count in genre_counts.items()}

        # Dominant genre
        dominant = max(genre_counts.items(), key=lambda x: x[1])

        # Entropy
        probs = np.array(list(genre_ratios.values()))
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return GenreMetrics(
            unique_genres=len(genre_counts),
            genre_counts=dict(genre_counts),
            genre_ratios=genre_ratios,
            dominant_genre=dominant[0],
            dominant_ratio=dominant[1] / total,
            genre_entropy=float(entropy),
        )

    def analyze_musical_features(self) -> MusicalFeatureStats:
        """Analyze musical feature statistics from profiles.

        Returns:
            MusicalFeatureStats with BPM, density, velocity, polyphony stats.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        bpm_values = []
        density_values = []
        velocity_values = []
        polyphony_values = []

        # Collect feature values
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    bpm_values.append(profile.mean_bpm)
                    density_values.append(profile.mean_density)
                    velocity_values.append(profile.mean_velocity)
                    polyphony_values.append(profile.mean_polyphony)

        if not bpm_values:
            logger.warning("No valid musical feature data found")
            return MusicalFeatureStats(
                bpm_mean=0.0,
                bpm_std=0.0,
                bpm_min=0.0,
                bpm_max=0.0,
                density_mean=0.0,
                density_std=0.0,
                velocity_mean=0.0,
                velocity_std=0.0,
                polyphony_mean=0.0,
                polyphony_std=0.0,
            )

        return MusicalFeatureStats(
            bpm_mean=float(np.mean(bpm_values)),
            bpm_std=float(np.std(bpm_values)),
            bpm_min=float(np.min(bpm_values)),
            bpm_max=float(np.max(bpm_values)),
            density_mean=float(np.mean(density_values)),
            density_std=float(np.std(density_values)),
            velocity_mean=float(np.mean(velocity_values)),
            velocity_std=float(np.std(velocity_values)),
            polyphony_mean=float(np.mean(polyphony_values)),
            polyphony_std=float(np.std(polyphony_values)),
        )

    def compute_all_metrics(self) -> ProfileEnrichedMetrics:
        """Compute all profile-enriched metrics.

        Returns:
            ProfileEnrichedMetrics with complete analysis.
        """
        if not self.sequences:
            raise ValueError("No sequences loaded")

        # Basic metrics from SequenceAnalyzer
        basic_diversity = self.sequence_analyzer.calculate_diversity_metrics()
        basic_transitions = self.sequence_analyzer.calculate_transition_metrics()

        # Profile-enriched metrics
        arousal = self.analyze_arousal_distribution()
        instruments = self.analyze_instrument_diversity()
        roles = self.analyze_role_distribution()
        preferences = self.analyze_preference_types()
        genres = self.analyze_genre_coverage()
        musical_features = self.analyze_musical_features()

        # Sequence statistics
        num_sequences = len(self.sequences)
        avg_length = float(np.mean([len(seq) for seq in self.sequences]))
        total_steps = sum(len(seq) for seq in self.sequences)

        return ProfileEnrichedMetrics(
            num_sequences=num_sequences,
            avg_sequence_length=avg_length,
            total_steps=total_steps,
            arousal=arousal,
            instruments=instruments,
            roles=roles,
            preferences=preferences,
            genres=genres,
            musical_features=musical_features,
            basic_diversity=basic_diversity,
            basic_transitions=basic_transitions,
        )

    def plot_arousal_trajectory(
        self,
        output_path: Path,
        max_sequences: int = 10,
    ) -> None:
        """Plot arousal trajectory over sequences.

        Args:
            output_path: Path to save plot.
            max_sequences: Maximum number of sequences to plot.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            logger.warning("matplotlib not available, skipping arousal trajectory plot")
            return

        if not self.sequences:
            raise ValueError("No sequences loaded")

        arousal_map = {"low": 0, "medium": 1, "high": 2, "unknown": -1}

        n_plots = min(max_sequences, len(self.sequences))
        _, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            seq = self.sequences[i]
            arousal_levels = []

            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    arousal_levels.append(arousal_map.get(profile.arousal_level, -1))
                else:
                    arousal_levels.append(-1)

            # Plot with color coding
            colors = []
            for level in arousal_levels:
                if level == 0:
                    colors.append("blue")
                elif level == 1:
                    colors.append("orange")
                elif level == 2:
                    colors.append("red")
                else:
                    colors.append("gray")

            ax.scatter(
                range(len(arousal_levels)), arousal_levels, c=colors, s=50, alpha=0.7
            )
            ax.plot(arousal_levels, alpha=0.3, color="black", linestyle="--")
            ax.set_ylabel("Arousal")
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Low", "Medium", "High"])
            ax.set_title(f"Sequence {i+1}")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved arousal trajectory plot to {output_path}")

    def plot_instrument_mix(self, output_path: Path) -> None:
        """Plot instrument mix pie chart.

        Args:
            output_path: Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            logger.warning("matplotlib not available, skipping instrument mix plot")
            return

        metrics = self.analyze_instrument_diversity()

        _, ax = plt.subplots(figsize=(10, 8))

        # Sort by count for better visualization
        sorted_items = sorted(
            metrics.instrument_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        instruments = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Create pie chart
        _, texts, autotexts = ax.pie(
            counts,
            labels=instruments,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 5 else "",
            startangle=90,
        )

        # Improve text readability
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(9)

        ax.set_title(
            f"Instrument Distribution\n"
            f"(Diversity: {metrics.instrument_diversity:.3f}, "
            f"Unique: {metrics.unique_instruments})"
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved instrument mix plot to {output_path}")

    def plot_role_distribution(self, output_path: Path) -> None:
        """Plot role distribution pie chart.

        Args:
            output_path: Path to save plot.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            logger.warning("matplotlib not available, skipping role distribution plot")
            return

        metrics = self.analyze_role_distribution()

        _, ax = plt.subplots(figsize=(10, 8))

        # Sort by count
        sorted_items = sorted(
            metrics.role_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        roles = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]

        # Create pie chart
        _, texts, autotexts = ax.pie(
            counts,
            labels=roles,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
            startangle=90,
        )

        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(9)

        ax.set_title(
            f"Role Distribution\n"
            f"(Entropy: {metrics.role_entropy:.3f}, "
            f"Unique: {metrics.unique_roles})"
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved role distribution plot to {output_path}")

    def plot_musical_features_boxplots(
        self,
        output_path: Path,
        groupby: Optional[str] = None,
    ) -> None:
        """Plot musical feature boxplots.

        Args:
            output_path: Path to save plot.
            groupby: Optional attribute to group by (e.g., "arousal_level", "preference_type").
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")
        except ImportError:
            logger.warning("matplotlib not available, skipping musical features plot")
            return

        # Collect data with grouping
        data = defaultdict(lambda: defaultdict(list))

        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    group_key = (
                        "all"
                        if groupby is None
                        else str(getattr(profile, groupby, "unknown"))
                    )

                    data[group_key]["bpm"].append(profile.mean_bpm)
                    data[group_key]["density"].append(profile.mean_density)
                    data[group_key]["velocity"].append(profile.mean_velocity)
                    data[group_key]["polyphony"].append(profile.mean_polyphony)

        # Create subplots
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        features = ["bpm", "density", "velocity", "polyphony"]
        titles = [
            "BPM Distribution",
            "Density Distribution",
            "Velocity Distribution",
            "Polyphony Distribution",
        ]

        for idx, (feature, title) in enumerate(zip(features, titles)):
            ax = axes[idx]

            # Prepare data for boxplot
            groups = sorted(data.keys())
            feature_data = [data[group][feature] for group in groups]

            bp = ax.boxplot(feature_data, labels=groups, patch_artist=True)

            # Color boxes
            import matplotlib.cm as cm

            colors = cm.Set3(range(len(groups)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_title(title)
            ax.set_ylabel(feature.capitalize())
            if groupby:
                ax.set_xlabel(groupby.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved musical features boxplots to {output_path}")

    def compare_arousal_distributions(
        self,
        other_sequences: List[List[int]],
        labels: Tuple[str, str] = ("Set A", "Set B"),
    ) -> Dict[str, Any]:
        """Compare arousal distributions between two sequence sets.

        Args:
            other_sequences: Second set of sequences to compare.
            labels: Labels for the two sets.

        Returns:
            Dictionary with comparison results and statistical tests.
        """
        # Current sequences arousal
        arousal_a = []
        for seq in self.sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    arousal_a.append(profile.arousal_level)

        # Other sequences arousal
        arousal_b = []
        for seq in other_sequences:
            for cluster_id in seq:
                profile = self._get_profile(cluster_id)
                if profile:
                    arousal_b.append(profile.arousal_level)

        # Count distributions
        counts_a = Counter(arousal_a)
        counts_b = Counter(arousal_b)

        total_a = sum(counts_a.values())
        total_b = sum(counts_b.values())

        # Chi-square test for distribution difference
        arousal_levels = ["low", "medium", "high"]
        observed_a = [counts_a.get(level, 0) for level in arousal_levels]
        observed_b = [counts_b.get(level, 0) for level in arousal_levels]

        # Contingency table
        contingency = np.array([observed_a, observed_b])

        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            chi2_result = {
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "significant": p_value < 0.05,
            }
        except ValueError:
            chi2_result = {
                "error": "Chi-square test failed (insufficient data)",
            }

        return {
            "label_a": labels[0],
            "label_b": labels[1],
            "distribution_a": {
                level: counts_a.get(level, 0) / total_a if total_a > 0 else 0
                for level in arousal_levels
            },
            "distribution_b": {
                level: counts_b.get(level, 0) / total_b if total_b > 0 else 0
                for level in arousal_levels
            },
            "chi_square_test": chi2_result,
        }

    def compare_diversity_metrics(
        self,
        other_analyzer: SequenceProfileAnalyzer,
        labels: Tuple[str, str] = ("Set A", "Set B"),
    ) -> pd.DataFrame:
        """Compare diversity metrics between two analyzers.

        Args:
            other_analyzer: Another SequenceProfileAnalyzer to compare.
            labels: Labels for the two sets.

        Returns:
            DataFrame with side-by-side comparison.
        """
        metrics_a = self.compute_all_metrics()
        metrics_b = other_analyzer.compute_all_metrics()

        comparison = []

        # Arousal metrics
        comparison.append(
            {
                "Metric": "Low Arousal Ratio",
                labels[0]: f"{metrics_a.arousal.low_ratio:.3f}",
                labels[1]: f"{metrics_b.arousal.low_ratio:.3f}",
            }
        )
        comparison.append(
            {
                "Metric": "Medium Arousal Ratio",
                labels[0]: f"{metrics_a.arousal.medium_ratio:.3f}",
                labels[1]: f"{metrics_b.arousal.medium_ratio:.3f}",
            }
        )
        comparison.append(
            {
                "Metric": "High Arousal Ratio",
                labels[0]: f"{metrics_a.arousal.high_ratio:.3f}",
                labels[1]: f"{metrics_b.arousal.high_ratio:.3f}",
            }
        )

        # Instrument diversity
        comparison.append(
            {
                "Metric": "Unique Instruments",
                labels[0]: str(metrics_a.instruments.unique_instruments),
                labels[1]: str(metrics_b.instruments.unique_instruments),
            }
        )
        comparison.append(
            {
                "Metric": "Instrument Diversity",
                labels[0]: f"{metrics_a.instruments.instrument_diversity:.3f}",
                labels[1]: f"{metrics_b.instruments.instrument_diversity:.3f}",
            }
        )

        # Role diversity
        comparison.append(
            {
                "Metric": "Unique Roles",
                labels[0]: str(metrics_a.roles.unique_roles),
                labels[1]: str(metrics_b.roles.unique_roles),
            }
        )
        comparison.append(
            {
                "Metric": "Role Entropy",
                labels[0]: f"{metrics_a.roles.role_entropy:.3f}",
                labels[1]: f"{metrics_b.roles.role_entropy:.3f}",
            }
        )

        # Preference balance
        comparison.append(
            {
                "Metric": "Intrinsic Ratio",
                labels[0]: f"{metrics_a.preferences.intrinsic_ratio:.3f}",
                labels[1]: f"{metrics_b.preferences.intrinsic_ratio:.3f}",
            }
        )

        # Musical features
        comparison.append(
            {
                "Metric": "Mean BPM",
                labels[0]: f"{metrics_a.musical_features.bpm_mean:.1f}",
                labels[1]: f"{metrics_b.musical_features.bpm_mean:.1f}",
            }
        )
        comparison.append(
            {
                "Metric": "Mean Density",
                labels[0]: f"{metrics_a.musical_features.density_mean:.3f}",
                labels[1]: f"{metrics_b.musical_features.density_mean:.3f}",
            }
        )

        return pd.DataFrame(comparison)

    def generate_comprehensive_report(
        self,
        output_dir: Union[str, Path],
        include_visualizations: bool = True,
    ) -> Path:
        """Generate comprehensive profile-enriched analysis report.

        Args:
            output_dir: Output directory for report files.
            include_visualizations: Whether to generate visualizations.

        Returns:
            Path to report directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating profile-enriched report at {output_dir}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Computing metrics...", total=None)

            # Compute all metrics
            metrics = self.compute_all_metrics()

            # Save metrics JSON
            progress.update(task, description="Saving metrics...")
            metrics_path = output_dir / "profile_enriched_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics.to_dict(), f, indent=2, default=str)

            # Save individual metric CSVs
            progress.update(task, description="Saving CSV reports...")
            self._save_metric_csvs(output_dir, metrics)

            # Generate visualizations
            if include_visualizations:
                progress.update(task, description="Generating visualizations...")
                viz_dir = output_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                self._generate_profile_visualizations(viz_dir)

            # Generate markdown report
            progress.update(task, description="Generating report...")
            report_path = output_dir / "profile_enriched_report.md"
            self._generate_markdown_report(report_path, metrics)

            # Also generate basic sequence analysis report
            progress.update(task, description="Generating basic analysis...")
            self.sequence_analyzer.generate_report(
                output_dir / "basic_analysis",
                include_visualizations=include_visualizations,
            )

        console.print(f"[green]✓[/green] Report generated at: {output_dir}")
        return output_dir

    def _save_metric_csvs(
        self, output_dir: Path, metrics: ProfileEnrichedMetrics
    ) -> None:
        """Save individual metrics as CSV files."""
        # Arousal distribution
        arousal_df = pd.DataFrame(
            [
                {
                    "level": "low",
                    "count": metrics.arousal.low_count,
                    "ratio": metrics.arousal.low_ratio,
                },
                {
                    "level": "medium",
                    "count": metrics.arousal.medium_count,
                    "ratio": metrics.arousal.medium_ratio,
                },
                {
                    "level": "high",
                    "count": metrics.arousal.high_count,
                    "ratio": metrics.arousal.high_ratio,
                },
            ]
        )
        arousal_df.to_csv(output_dir / "arousal_distribution.csv", index=False)

        # Instrument distribution
        inst_df = pd.DataFrame(
            [
                {
                    "instrument": inst,
                    "count": count,
                    "ratio": metrics.instruments.instrument_ratios[inst],
                }
                for inst, count in metrics.instruments.instrument_counts.items()
            ]
        )
        inst_df = inst_df.sort_values("count", ascending=False)
        inst_df.to_csv(output_dir / "instrument_distribution.csv", index=False)

        # Role distribution
        role_df = pd.DataFrame(
            [
                {"role": role, "count": count, "ratio": metrics.roles.role_ratios[role]}
                for role, count in metrics.roles.role_counts.items()
            ]
        )
        role_df = role_df.sort_values("count", ascending=False)
        role_df.to_csv(output_dir / "role_distribution.csv", index=False)

        # Genre distribution
        genre_df = pd.DataFrame(
            [
                {
                    "genre": genre,
                    "count": count,
                    "ratio": metrics.genres.genre_ratios[genre],
                }
                for genre, count in metrics.genres.genre_counts.items()
            ]
        )
        genre_df = genre_df.sort_values("count", ascending=False)
        genre_df.to_csv(output_dir / "genre_distribution.csv", index=False)

        logger.info("Saved metric CSV files")

    def _generate_profile_visualizations(self, output_dir: Path) -> None:
        """Generate all profile-aware visualizations."""
        self.plot_arousal_trajectory(output_dir / "arousal_trajectory.png")
        self.plot_instrument_mix(output_dir / "instrument_mix.png")
        self.plot_role_distribution(output_dir / "role_distribution.png")
        self.plot_musical_features_boxplots(output_dir / "musical_features_all.png")
        self.plot_musical_features_boxplots(
            output_dir / "musical_features_by_arousal.png",
            groupby="arousal_level",
        )
        self.plot_musical_features_boxplots(
            output_dir / "musical_features_by_preference.png",
            groupby="preference_type",
        )

        logger.info("Generated profile visualizations")

    def _generate_markdown_report(
        self,
        output_path: Path,
        metrics: ProfileEnrichedMetrics,
    ) -> None:
        """Generate comprehensive markdown report."""
        report = f"""# Profile-Enriched Sequence Analysis Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview

| Metric | Value |
|--------|-------|
| Number of Sequences | {metrics.num_sequences} |
| Average Sequence Length | {metrics.avg_sequence_length:.1f} |
| Total Steps | {metrics.total_steps} |

## Arousal Distribution

| Level | Count | Ratio |
|-------|-------|-------|
| Low | {metrics.arousal.low_count} | {metrics.arousal.low_ratio:.3f} |
| Medium | {metrics.arousal.medium_count} | {metrics.arousal.medium_ratio:.3f} |
| High | {metrics.arousal.high_count} | {metrics.arousal.high_ratio:.3f} |

**Arousal Entropy:** {metrics.arousal.arousal_entropy:.3f}

## Instrument Diversity

| Metric | Value |
|--------|-------|
| Unique Instruments | {metrics.instruments.unique_instruments} |
| Dominant Instrument | {metrics.instruments.dominant_instrument} ({metrics.instruments.dominant_ratio:.1%}) |
| Instrument Diversity | {metrics.instruments.instrument_diversity:.3f} |
| Instrument Entropy | {metrics.instruments.instrument_entropy:.3f} |

### Top Instruments
"""
        # Top 5 instruments
        sorted_inst = sorted(
            metrics.instruments.instrument_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for inst, count in sorted_inst:
            ratio = metrics.instruments.instrument_ratios[inst]
            report += f"- **{inst}**: {count} ({ratio:.1%})\n"

        report += f"""
## Role Distribution

| Metric | Value |
|--------|-------|
| Unique Roles | {metrics.roles.unique_roles} |
| Dominant Role | {metrics.roles.dominant_role} ({metrics.roles.dominant_ratio:.1%}) |
| Role Entropy | {metrics.roles.role_entropy:.3f} |
| Has Melody | {'✓' if metrics.roles.has_melody else '✗'} |
| Has Accompaniment | {'✓' if metrics.roles.has_accompaniment else '✗'} |
| Has Pad | {'✓' if metrics.roles.has_pad else '✗'} |

### Role Breakdown
"""
        # Top 5 roles
        sorted_roles = sorted(
            metrics.roles.role_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for role, count in sorted_roles:
            ratio = metrics.roles.role_ratios[role]
            report += f"- **{role}**: {count} ({ratio:.1%})\n"

        report += f"""
## Preference Types

| Type | Count | Ratio |
|------|-------|-------|
| Intrinsic | {metrics.preferences.intrinsic_count} | {metrics.preferences.intrinsic_ratio:.3f} |
| Extrinsic | {metrics.preferences.extrinsic_count} | {metrics.preferences.extrinsic_ratio:.3f} |

**Preference Balance:** {metrics.preferences.preference_balance:.3f} (0.5 = perfect balance)

## Genre Coverage

| Metric | Value |
|--------|-------|
| Unique Genres | {metrics.genres.unique_genres} |
| Dominant Genre | {metrics.genres.dominant_genre} ({metrics.genres.dominant_ratio:.1%}) |
| Genre Entropy | {metrics.genres.genre_entropy:.3f} |

## Musical Features

| Feature | Mean | Std Dev | Range |
|---------|------|---------|-------|
| BPM | {metrics.musical_features.bpm_mean:.1f} | {metrics.musical_features.bpm_std:.1f} | [{metrics.musical_features.bpm_min:.1f}, {metrics.musical_features.bpm_max:.1f}] |
| Density | {metrics.musical_features.density_mean:.3f} | {metrics.musical_features.density_std:.3f} | - |
| Velocity | {metrics.musical_features.velocity_mean:.1f} | {metrics.musical_features.velocity_std:.1f} | - |
| Polyphony | {metrics.musical_features.polyphony_mean:.2f} | {metrics.musical_features.polyphony_std:.2f} | - |

## Basic Diversity Metrics

| Metric | Value |
|--------|-------|
| Unique Clusters (Total) | {metrics.basic_diversity.unique_clusters_total} |
| Unique per Sequence | {metrics.basic_diversity.unique_per_sequence_mean:.2f} ± {metrics.basic_diversity.unique_per_sequence_std:.2f} |
| Repetition Ratio | {metrics.basic_diversity.repetition_ratio_mean:.3f} ± {metrics.basic_diversity.repetition_ratio_std:.3f} |
| Entropy | {metrics.basic_diversity.entropy:.3f} |
| Coverage Ratio | {metrics.basic_diversity.coverage_ratio:.2%} |

## Transition Metrics

| Metric | Value |
|--------|-------|
| Unique Transitions | {metrics.basic_transitions.num_unique_transitions} |
| Avg Transition Entropy | {metrics.basic_transitions.avg_transition_entropy:.3f} |
| Self-Transition Ratio | {metrics.basic_transitions.self_transition_ratio:.3f} |
| Max Transition Prob | {metrics.basic_transitions.max_transition_prob:.3f} |

## Files Generated

- `profile_enriched_metrics.json` - Complete metrics in JSON format
- `arousal_distribution.csv` - Arousal level statistics
- `instrument_distribution.csv` - Instrument usage statistics
- `role_distribution.csv` - Role usage statistics
- `genre_distribution.csv` - Genre coverage statistics
- `basic_analysis/` - Basic sequence analysis reports
- `visualizations/` - Generated plots

"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Generated markdown report at {output_path}")


def analyze_sequences_with_profiles(
    sequences_path: Union[str, Path],
    profile_loader: ClusterProfileLoader,
    output_dir: Union[str, Path],
    ghsom_manager: Optional[GHSOMManager] = None,
) -> ProfileEnrichedMetrics:
    """Convenience function for profile-enriched sequence analysis.

    Args:
        sequences_path: Path to sequences JSON file.
        profile_loader: Cluster profile loader.
        output_dir: Directory for analysis output.
        ghsom_manager: Optional GHSOM manager.

    Returns:
        ProfileEnrichedMetrics with complete analysis.
    """
    analyzer = SequenceProfileAnalyzer(profile_loader, ghsom_manager)
    analyzer.load_sequences(sequences_path)
    analyzer.generate_comprehensive_report(output_dir)
    return analyzer.compute_all_metrics()
