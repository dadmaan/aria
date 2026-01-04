"""Cluster profile loading and querying for music generation simulation.

This module provides functionality to load and query cluster profiles from CSV files,
enabling the retrieval of musical characteristics for different cluster groups.
"""

import csv
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClusterProfile:
    """Data class representing a cluster's musical profile.

    Attributes:
        cluster_id: Unique identifier for the cluster.
        sample_count: Number of samples in the cluster.
        dominant_genre: Most common genre in the cluster.
        genre_percentage: Percentage of samples with the dominant genre.
        dominant_role: Most common musical role in the cluster.
        role_percentage: Percentage of samples with the dominant role.
        dominant_instrument: Most common instrument in the cluster.
        instrument_percentage: Percentage of samples with the dominant instrument.
        mean_bpm: Average beats per minute.
        mean_density: Average note density.
        mean_velocity: Average MIDI velocity.
        mean_polyphony: Average polyphony level.
        mean_intra_distance: Average intra-cluster distance.
        cohesion_score: Cluster cohesion score.
        communicative_function: The communicative function of the cluster.
        artistic_intention: The artistic intention description.
        phenomenological_quality: The phenomenological quality description.
        preference_type: Type of preference (intrinsic/extrinsic).
        arousal_level: Arousal level (low/medium/high).
    """

    cluster_id: int
    sample_count: int
    dominant_genre: str
    genre_percentage: float
    dominant_role: str
    role_percentage: float
    dominant_instrument: str
    instrument_percentage: float
    mean_bpm: float
    mean_density: float
    mean_velocity: float
    mean_polyphony: float
    mean_intra_distance: float
    cohesion_score: float
    communicative_function: str
    artistic_intention: str
    phenomenological_quality: str
    preference_type: str
    arousal_level: str


class ClusterProfileLoader:
    """Loader and query interface for cluster profiles.

    This class provides methods to load cluster profiles from a CSV file
    and query them based on various attributes.

    Attributes:
        csv_path: Path to the CSV file containing cluster profiles.
        profiles: Dictionary mapping cluster IDs to ClusterProfile objects.
    """

    def __init__(self, csv_path: Path) -> None:
        """Initialize the loader and load profiles from CSV.

        Args:
            csv_path: Path to the CSV file containing cluster profiles.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is empty or has invalid format.
        """
        self.csv_path = Path(csv_path)
        self.profiles: Dict[int, ClusterProfile] = {}

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Cluster profiles CSV not found: {csv_path}")

        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load cluster profiles from the CSV file into memory."""
        logger.info("Loading cluster profiles from %s", self.csv_path)

        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    try:
                        profile = ClusterProfile(
                            cluster_id=int(row["cluster_id"]),
                            sample_count=int(row["sample_count"]),
                            dominant_genre=row["dominant_genre"],
                            genre_percentage=float(row["genre_percentage"]),
                            dominant_role=row["dominant_role"],
                            role_percentage=float(row["role_percentage"]),
                            dominant_instrument=row["dominant_instrument"],
                            instrument_percentage=float(row["instrument_percentage"]),
                            mean_bpm=float(row["mean_bpm"]),
                            mean_density=float(row["mean_density"]),
                            mean_velocity=float(row["mean_velocity"]),
                            mean_polyphony=float(row["mean_polyphony"]),
                            mean_intra_distance=float(row["mean_intra_distance"]),
                            cohesion_score=float(row["cohesion_score"]),
                            communicative_function=row["communicative_function"],
                            artistic_intention=row["artistic_intention"],
                            phenomenological_quality=row["phenomenological_quality"],
                            preference_type=row["preference_type"],
                            arousal_level=row["arousal_level"],
                        )
                        self.profiles[profile.cluster_id] = profile
                    except (KeyError, ValueError) as e:
                        logger.warning("Skipping invalid row: %s. Error: %s", row, e)
                        continue

            logger.info("Loaded %d cluster profiles", len(self.profiles))

            if not self.profiles:
                raise ValueError("No valid cluster profiles found in CSV")

        except csv.Error as e:
            logger.error("CSV parsing error: %s", e)
            raise ValueError(f"Invalid CSV format: {e}") from e

    def get_profile(self, cluster_id: int) -> Optional[ClusterProfile]:
        """Get a single cluster profile by ID.

        Args:
            cluster_id: The cluster ID to retrieve.

        Returns:
            The ClusterProfile if found, None otherwise.
        """
        profile = self.profiles.get(cluster_id)
        if profile is None:
            logger.debug("Cluster ID %d not found", cluster_id)
        return profile

    def get_all_profiles(self) -> List[ClusterProfile]:
        """Get all loaded cluster profiles.

        Returns:
            List of all ClusterProfile objects.
        """
        return list(self.profiles.values())

    def get_clusters_by_attribute(self, attr: str, value: str) -> List[int]:
        """Get cluster IDs filtered by any string attribute.

        Args:
            attr: The attribute name to filter by.
            value: The value to match (case-insensitive).

        Returns:
            List of cluster IDs matching the filter.

        Raises:
            AttributeError: If the attribute does not exist on ClusterProfile.
        """
        # Get valid field names from the dataclass
        valid_fields = {f.name for f in fields(ClusterProfile)}

        # Validate attribute exists
        if attr not in valid_fields:
            raise AttributeError(f"ClusterProfile has no attribute '{attr}'")

        matching_clusters = []
        value_lower = value.lower()

        for cluster_id, profile in self.profiles.items():
            attr_value = getattr(profile, attr)
            # Handle both string and non-string attributes
            if isinstance(attr_value, str):
                if attr_value.lower() == value_lower:
                    matching_clusters.append(cluster_id)
            else:
                if str(attr_value) == value:
                    matching_clusters.append(cluster_id)

        logger.debug(
            "Found %d clusters with %s='%s'", len(matching_clusters), attr, value
        )
        return matching_clusters

    def get_arousal_clusters(self, level: str) -> List[int]:
        """Get cluster IDs by arousal level.

        Args:
            level: The arousal level to filter by (low/medium/high).

        Returns:
            List of cluster IDs with the specified arousal level.
        """
        return self.get_clusters_by_attribute("arousal_level", level)

    def get_instrument_clusters(self, instrument: str) -> List[int]:
        """Get cluster IDs by dominant instrument.

        Args:
            instrument: The instrument name to filter by.

        Returns:
            List of cluster IDs with the specified dominant instrument.
        """
        return self.get_clusters_by_attribute("dominant_instrument", instrument)

    def get_role_clusters(self, role: str) -> List[int]:
        """Get cluster IDs by dominant role.

        Args:
            role: The role name to filter by.

        Returns:
            List of cluster IDs with the specified dominant role.
        """
        return self.get_clusters_by_attribute("dominant_role", role)

    def get_genre_clusters(self, genre: str) -> List[int]:
        """Get cluster IDs by dominant genre.

        Args:
            genre: The genre name to filter by.

        Returns:
            List of cluster IDs with the specified dominant genre.
        """
        return self.get_clusters_by_attribute("dominant_genre", genre)

    def get_preference_type_clusters(self, ptype: str) -> List[int]:
        """Get cluster IDs by preference type.

        Args:
            ptype: The preference type to filter by (intrinsic/extrinsic).

        Returns:
            List of cluster IDs with the specified preference type.
        """
        return self.get_clusters_by_attribute("preference_type", ptype)

    def __len__(self) -> int:
        """Return the number of loaded profiles."""
        return len(self.profiles)

    def __repr__(self) -> str:
        """Return string representation of the loader."""
        return f"ClusterProfileLoader(csv_path='{self.csv_path}', profiles={len(self)})"


# Default path for cluster profiles
DEFAULT_CLUSTER_PROFILES_PATH = Path(
    "/workspace/experiments/ghsom_commu_full_tsne_optimized_20251125/cluster_profiles.csv"
)


def load_default_profiles() -> ClusterProfileLoader:
    """Load cluster profiles from the default path.

    Returns:
        ClusterProfileLoader instance with profiles loaded.

    Raises:
        FileNotFoundError: If the default CSV file does not exist.
    """
    return ClusterProfileLoader(DEFAULT_CLUSTER_PROFILES_PATH)
