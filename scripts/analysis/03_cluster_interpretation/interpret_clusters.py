#!/usr/bin/env python3
"""
=============================================================================
GHSOM Cluster Interpretation Pipeline: Artistic Intention & Musical Preference Analysis
=============================================================================

This pipeline synthesizes multiple analysis components to interpret GHSOM clustering
results from the perspective of artistic intention, communicative function, and
musical preference addressability.

**IMPORTANT LIMITATIONS:**

1. **Semantic Interpretations (Exploratory Framework):**
   - Role-function mappings (e.g., "main_melody" → "Direct Address") are interpretive
   - NOT validated through user studies or perceptual experiments
   - Based on music theory conventions, not empirical listener data
   - Should be treated as analytical framework, not scientific claims

2. **Genre-Preference Mapping (Data-Driven Bias):**
   - Dataset is 90% cinematic/extrinsic, 10% newage/intrinsic
   - Intrinsic preference scenarios may have limited viability
   - Results reflect dataset composition, not universal musical preferences

3. **Arousal Classification:**
   - Composite score based on BPM, velocity, and density
   - Weights are heuristic (0.5 BPM, 0.3 velocity, 0.2 density)
   - No validation against human arousal perception studies

**For Scientific Publication:**
- Conduct user validation studies for semantic mappings (Cohen's kappa > 0.7)
- Balance dataset to 40-60% genre split for robust preference learning
- Validate arousal scores against listener self-reports

The pipeline integrates:
1. Cluster assignment loading and metadata merging
2. Musical characteristic profiling (genre, role, instrument, dynamics)
3. Feature-based similarity analysis within and between clusters
4. Cluster quality metrics computation
5. Intentionality mapping (functional interpretation of clusters)
6. Preference categorization (intrinsic vs extrinsic appeal)
7. Comprehensive report generation

Usage:
    python analysis/run_interpret_results.py
    python analysis/run_interpret_results.py --config path/to/config.json
    python analysis/run_interpret_results.py --model-dir path/to/ghsom/output

Output:
    - cluster_profiles.csv: Detailed profiles for each cluster
    - cluster_interpretation_report.md: Human-readable interpretation
    - intentionality_mapping.json: Functional intention annotations
    - quality_metrics.json: Clustering quality statistics
    - validation_report.json: Profile-model alignment validation

Date: November 25, 2025 (Updated: December 14, 2025 - Phase 1 Fixes)
"""

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import variation
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Default paths
DEFAULT_CONFIG = {
    "model_dir": "/workspace/experiments/ghsom_commu_full_tsne_optimized_20251125",
    "metadata_csv": "/workspace/artifacts/features/raw/commu_full/features_with_metadata.csv",
    "output_dir": "/workspace/experiments/ghsom_commu_full_tsne_optimized_20251125",  # Output to model dir
    "distance_metric": "euclidean",
    "random_seed": 42,
}

# Musical role categories and their communicative functions
ROLE_FUNCTIONS = {
    "main_melody": {
        "function": "Direct Address",
        "intention": "To speak directly to the listener",
        "phenomenology": "Someone is speaking to me",
    },
    "sub_melody": {
        "function": "Narrative Support",
        "intention": "To color and extend the main message",
        "phenomenology": "Something important accompanies the voice",
    },
    "accompaniment": {
        "function": "Harmonic Foundation",
        "intention": "To provide stability and context",
        "phenomenology": "Home base, grounding",
    },
    "pad": {
        "function": "Atmospheric Creation",
        "intention": "To create space and emotional context",
        "phenomenology": "Held by sound, dissolved into atmosphere",
    },
    "riff": {
        "function": "Motoric Drive",
        "intention": "To propel forward motion and energy",
        "phenomenology": "Forward, always forward",
    },
    "bass": {
        "function": "Low-End Anchoring",
        "intention": "To provide weight and gravitas",
        "phenomenology": "Weight of the world beneath",
    },
}

# Genre-based preference mapping
GENRE_PREFERENCE_MAP = {
    "cinematic": {
        "preference_type": "extrinsic",
        "target_context": "Film/media accompaniment",
        "emotional_mode": "Variable (narrative-dependent)",
    },
    "newage": {
        "preference_type": "intrinsic",
        "target_context": "Wellness/meditation",
        "emotional_mode": "Calm, contemplative",
    },
}

# Arousal classification thresholds (legacy - kept for reference)
AROUSAL_THRESHOLDS = {
    "low": {"bpm_max": 80, "velocity_max": 60, "density_max": 2.0},
    "medium": {"bpm_max": 110, "velocity_max": 90, "density_max": 5.0},
    "high": {},  # Everything above medium
}


def compute_arousal_score(
    mean_bpm: float,
    mean_velocity: float,
    mean_density: float,
) -> Tuple[str, float]:
    """
    Compute composite arousal score and classify into low/medium/high.

    Uses weighted combination of normalized BPM, velocity, and density.
    Based on empirical distribution of the dataset.

    Args:
        mean_bpm: Average tempo in BPM
        mean_velocity: Average MIDI velocity (0-127)
        mean_density: Average note density (notes/second)

    Returns:
        (arousal_level, arousal_score)
        arousal_level: "low", "medium", or "high"
        arousal_score: Continuous value 0-100
    """
    # Normalization ranges (from empirical data distribution)
    # 5th-95th percentile of dataset
    BPM_MIN, BPM_MAX = 50, 150
    VELOCITY_MIN, VELOCITY_MAX = 40, 110
    DENSITY_MIN, DENSITY_MAX = 0.5, 8.0

    # Normalize to 0-1 range
    bpm_norm = np.clip((mean_bpm - BPM_MIN) / (BPM_MAX - BPM_MIN), 0, 1)
    vel_norm = np.clip((mean_velocity - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN), 0, 1)
    den_norm = np.clip((mean_density - DENSITY_MIN) / (DENSITY_MAX - DENSITY_MIN), 0, 1)

    # Weighted average (BPM has highest weight for arousal)
    # Based on music psychology literature (BPM strongest arousal predictor)
    weights = np.array([0.5, 0.3, 0.2])  # BPM, velocity, density
    arousal_norm = np.average([bpm_norm, vel_norm, den_norm], weights=weights)

    # Convert to 0-100 scale
    arousal_score = arousal_norm * 100

    # Classify into tertiles (aiming for ~33% in each category)
    if arousal_score < 33:
        arousal_level = "low"
    elif arousal_score < 66:
        arousal_level = "medium"
    else:
        arousal_level = "high"

    return arousal_level, arousal_score


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ClusterProfile:
    """Complete profile of a musical cluster with intentional annotations."""

    cluster_id: int
    sample_count: int

    # Dominant characteristics
    dominant_genre: str
    genre_percentage: float
    dominant_role: str
    role_percentage: float
    dominant_instrument: str
    instrument_percentage: float

    # Numeric statistics
    mean_bpm: float
    std_bpm: float
    mean_note_count: float
    mean_density: float
    mean_velocity: float
    mean_polyphony: float

    # Quality metrics
    mean_intra_distance: float
    std_intra_distance: float
    cohesion_score: float

    # Intentional annotations
    communicative_function: str = ""
    artistic_intention: str = ""
    phenomenological_quality: str = ""
    preference_type: str = ""
    arousal_level: str = ""
    arousal_score: float = 0.0  # Continuous arousal score (0-100)

    # Detailed breakdowns
    genre_distribution: Dict[str, int] = field(default_factory=dict)
    role_distribution: Dict[str, int] = field(default_factory=dict)
    instrument_top5: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class InterpretationSummary:
    """High-level summary of cluster interpretation results."""

    total_clusters: int
    total_samples: int
    analysis_timestamp: str

    # Aggregated intentionality
    function_distribution: Dict[str, int] = field(default_factory=dict)
    preference_distribution: Dict[str, int] = field(default_factory=dict)
    arousal_distribution: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    mean_silhouette: float = 0.0
    davies_bouldin_index: float = 0.0
    calinski_harabasz_index: float = 0.0


# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================


def load_training_features(
    model_dir: Path,
    metadata_csv: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the exact features used for GHSOM training.

    This ensures cohesion metrics are computed in the same
    feature space as the clustering.

    Args:
        model_dir: Path to GHSOM model directory
        metadata_csv: Path to features_with_metadata.csv

    Returns:
        (feature_df, feature_column_names)
    """
    # Check for feature artifact metadata
    feature_metadata_path = model_dir / "feature_artifact_metadata.json"

    if feature_metadata_path.exists():
        # Load from model metadata
        with open(feature_metadata_path, 'r') as f:
            feature_metadata = json.load(f)

        # Handle different metadata structures
        # New structure: artifacts.embedding_csv
        # Old structure: artifact_path + feature_type
        if 'artifacts' in feature_metadata and 'embedding_csv' in feature_metadata['artifacts']:
            # New structure with t-SNE embeddings
            embedding_csv = Path(feature_metadata['artifacts']['embedding_csv'])
            method = feature_metadata.get('metadata', {}).get('method', 'tsne')
            print(f"  → Loading {method} embeddings from: {embedding_csv}")

            if embedding_csv.exists():
                features_df = pd.read_csv(embedding_csv)
                # Columns can be: metadata_index, dim1, dim2 OR metadata_index, tsne_0, tsne_1
                # Identify dimension columns (exclude metadata_index)
                feature_cols = [col for col in features_df.columns
                               if col not in ['metadata_index', 'sample_index', 'Unnamed: 0']
                               and features_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                print(f"  → Loaded {len(feature_cols)} embedding dimensions: {feature_cols}")
            else:
                raise FileNotFoundError(f"Embedding not found: {embedding_csv}")

        elif 'artifact_path' in feature_metadata:
            # Old structure
            artifact_path = Path(feature_metadata['artifact_path'])
            feature_type = feature_metadata.get('feature_type', 'raw')
            print(f"  → Loading {feature_type} features from: {artifact_path}")

            if feature_type == 'tsne':
                # t-SNE embeddings
                tsne_csv = artifact_path / "embedding.csv"
                if tsne_csv.exists():
                    features_df = pd.read_csv(tsne_csv)
                    feature_cols = [col for col in features_df.columns
                                   if col not in ['metadata_index', 'sample_index', 'Unnamed: 0']
                                   and features_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                else:
                    raise FileNotFoundError(f"t-SNE embedding not found: {tsne_csv}")
            else:
                # Raw features
                features_df = pd.read_csv(artifact_path)
                exclude_patterns = [
                    "Unnamed", "id", "track_id", "file_", "split_",
                    "_adapted", "_adapter", "metadata_index", "sample_index",
                    "GHSOM_cluster", "audio_key", "chord_progressions",
                    "genre", "track_role", "inst", "sample_rhythm",
                    "time_signature", "pitch_range"
                ]
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [
                    col for col in numeric_cols
                    if not any(pattern in col.lower() for pattern in [p.lower() for p in exclude_patterns])
                ]
        else:
            raise ValueError(f"Unrecognized feature_artifact_metadata.json structure")

        print(f"  → Using {len(feature_cols)} features for cohesion: {feature_cols[:5]}...")
        return features_df, feature_cols

    else:
        # Fallback: use raw metadata features (old behavior)
        print(f"  ⚠️  WARNING: No feature_artifact_metadata.json found")
        print(f"  → Falling back to raw metadata features (may not match training)")

        metadata_df = pd.read_csv(metadata_csv)
        exclude_patterns = [
            "Unnamed", "id", "track_id", "file_", "split_",
            "_adapted", "_adapter", "metadata_index", "sample_index",
            "GHSOM_cluster", "audio_key", "chord_progressions",
            "genre", "track_role", "inst", "sample_rhythm",
            "time_signature", "pitch_range"
        ]
        numeric_cols = metadata_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            col for col in numeric_cols
            if not any(pattern in col.lower() for pattern in [p.lower() for p in exclude_patterns])
        ]

        return metadata_df, feature_cols


def load_cluster_data(
    model_dir: Path,
    metadata_csv: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and merge cluster assignments with training features.

    Returns:
        (merged_df, feature_columns)
    """
    cluster_csv = model_dir / "sample_to_cluster.csv"

    print(f"Loading cluster assignments from: {cluster_csv}")
    if not cluster_csv.exists():
        raise FileNotFoundError(f"Cluster assignment file not found: {cluster_csv}")
    cluster_df = pd.read_csv(cluster_csv)
    print(f"  → Loaded {len(cluster_df)} cluster assignments")

    # Load training features (not raw metadata!)
    print(f"Loading training features...")
    features_df, feature_columns = load_training_features(model_dir, metadata_csv)
    print(f"  → Loaded {len(features_df)} samples with {len(feature_columns)} features")

    # Load metadata for categorical fields
    print(f"Loading metadata from: {metadata_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
    metadata_df = pd.read_csv(metadata_csv)
    print(f"  → Loaded {len(metadata_df)} samples with {len(metadata_df.columns)} columns")

    # Create sample index
    metadata_df["sample_index"] = metadata_df.index
    features_df["sample_index"] = features_df.get("metadata_index", features_df.index)

    # Merge cluster assignments with features
    merged_df = cluster_df.merge(features_df, on="sample_index", how="left")

    # Merge categorical metadata
    categorical_cols = ["genre", "track_role", "inst", "bpm", "pm_note_count",
                       "pm_note_density", "pm_average_velocity", "pm_average_polyphony"]
    metadata_subset = metadata_df[["sample_index"] + [c for c in categorical_cols if c in metadata_df.columns]]
    merged_df = merged_df.merge(metadata_subset, on="sample_index", how="left", suffixes=('', '_meta'))

    if len(merged_df) == 0:
        raise ValueError("Merge produced no results - check column alignment")

    print(f"  → Merged dataset: {len(merged_df)} samples")
    print(f"  → Clusters found: {merged_df['GHSOM_cluster'].nunique()}")
    print(f"  → Features for cohesion: {len(feature_columns)}")

    return merged_df, feature_columns


def identify_numeric_features(df: pd.DataFrame) -> List[str]:
    """
    Identify numeric feature columns suitable for similarity analysis.

    NOTE: This is a legacy fallback function. Prefer load_training_features()
    which loads the exact features used during GHSOM training.

    Excludes metadata columns, identifiers, and adapter-related fields.

    Args:
        df: DataFrame with mixed column types

    Returns:
        List of column names containing numeric features
    """
    exclude_patterns = [
        "Unnamed",
        "id",
        "track_id",
        "file_",
        "split_",
        "_adapted",
        "_adapter",
        "metadata_index",
        "sample_index",
        "GHSOM_cluster",
        "audio_key",
        "chord_progressions",
        "genre",
        "track_role",
        "inst",
        "sample_rhythm",
        "time_signature",
        "pitch_range",
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        col
        for col in numeric_cols
        if not any(
            pattern in col.lower() for pattern in [p.lower() for p in exclude_patterns]
        )
    ]

    return feature_cols


def validate_model_profile_alignment(
    model_dir: Path,
    profiles_csv: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that cluster profiles match the GHSOM model.

    Returns:
        (is_valid, validation_report)
    """
    # Load model clusters
    cluster_csv = model_dir / "sample_to_cluster.csv"
    if not cluster_csv.exists():
        return False, {"error": f"Missing {cluster_csv}"}

    sample_to_cluster = pd.read_csv(cluster_csv)
    model_clusters = set(sample_to_cluster['GHSOM_cluster'].unique())
    model_sample_counts = sample_to_cluster['GHSOM_cluster'].value_counts().to_dict()

    # Load profile clusters
    if not profiles_csv.exists():
        return False, {"error": f"Missing {profiles_csv}"}

    profiles_df = pd.read_csv(profiles_csv)
    profile_clusters = set(profiles_df['cluster_id'].unique())
    profile_sample_counts = {
        row['cluster_id']: row['sample_count']
        for _, row in profiles_df.iterrows()
    }

    # Compare
    missing_profiles = model_clusters - profile_clusters
    extra_profiles = profile_clusters - model_clusters

    # Check sample counts for overlapping clusters
    sample_count_mismatches = []
    for cluster_id in model_clusters & profile_clusters:
        model_count = model_sample_counts.get(cluster_id, 0)
        profile_count = profile_sample_counts.get(cluster_id, 0)
        if model_count != profile_count:
            error_pct = abs(model_count - profile_count) / model_count * 100
            sample_count_mismatches.append({
                "cluster_id": cluster_id,
                "model_count": model_count,
                "profile_count": profile_count,
                "error_pct": error_pct
            })

    is_valid = (
        len(missing_profiles) == 0 and
        len(extra_profiles) == 0 and
        len(sample_count_mismatches) == 0
    )

    report = {
        "is_valid": is_valid,
        "model_clusters": [int(c) for c in sorted(model_clusters)],
        "profile_clusters": [int(c) for c in sorted(profile_clusters)],
        "missing_profiles": [int(c) for c in sorted(missing_profiles)],
        "extra_profiles": [int(c) for c in sorted(extra_profiles)],
        "sample_count_mismatches": [
            {k: int(v) if isinstance(v, (np.integer, np.int64)) else v
             for k, v in m.items()}
            for m in sample_count_mismatches
        ],
        "total_model_samples": int(len(sample_to_cluster)),
        "total_profile_samples": int(profiles_df['sample_count'].sum()),
    }

    return is_valid, report


def compute_cluster_profile(
    cluster_id: int,
    cluster_data: pd.DataFrame,
    feature_columns: List[str],
    distance_metric: str = "cosine",
) -> ClusterProfile:
    """
    Compute comprehensive profile for a single cluster.

    This function extracts:
    - Dominant categorical characteristics (genre, role, instrument)
    - Numeric feature statistics (tempo, density, dynamics)
    - Intra-cluster similarity metrics
    - Intentional annotations based on role and characteristics

    Args:
        cluster_id: Unique cluster identifier
        cluster_data: DataFrame containing only samples from this cluster
        feature_columns: List of numeric feature column names
        distance_metric: Metric for pairwise distance computation

    Returns:
        ClusterProfile with complete characterization
    """
    n_samples = len(cluster_data)

    # === CATEGORICAL ANALYSIS ===

    # Genre distribution
    genre_counts = cluster_data["genre"].value_counts()
    dominant_genre = genre_counts.index[0] if len(genre_counts) > 0 else "unknown"
    genre_pct = 100 * genre_counts.iloc[0] / n_samples if len(genre_counts) > 0 else 0

    # Track role distribution
    role_counts = cluster_data["track_role"].value_counts()
    dominant_role = role_counts.index[0] if len(role_counts) > 0 else "unknown"
    role_pct = 100 * role_counts.iloc[0] / n_samples if len(role_counts) > 0 else 0

    # Instrument distribution (top 5)
    inst_counts = cluster_data["inst"].value_counts()
    dominant_inst = inst_counts.index[0] if len(inst_counts) > 0 else "unknown"
    inst_pct = 100 * inst_counts.iloc[0] / n_samples if len(inst_counts) > 0 else 0
    inst_top5 = [(inst, count) for inst, count in inst_counts.head(5).items()]

    # === NUMERIC STATISTICS ===

    mean_bpm = cluster_data["bpm"].mean() if "bpm" in cluster_data.columns else 0
    std_bpm = cluster_data["bpm"].std() if "bpm" in cluster_data.columns else 0
    mean_notes = (
        cluster_data["pm_note_count"].mean()
        if "pm_note_count" in cluster_data.columns
        else 0
    )
    mean_density = (
        cluster_data["pm_note_density"].mean()
        if "pm_note_density" in cluster_data.columns
        else 0
    )
    mean_velocity = (
        cluster_data["pm_average_velocity"].mean()
        if "pm_average_velocity" in cluster_data.columns
        else 0
    )
    mean_polyphony = (
        cluster_data["pm_average_polyphony"].mean()
        if "pm_average_polyphony" in cluster_data.columns
        else 0
    )

    # === SIMILARITY METRICS ===

    if n_samples >= 2 and len(feature_columns) > 0:
        # Extract feature matrix
        features = cluster_data[feature_columns].values
        features = np.nan_to_num(features, nan=0.0)

        # Compute pairwise distances
        distances = pdist(features, metric=distance_metric)
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        cohesion = 1.0 / mean_dist if mean_dist > 0 else float("inf")
    else:
        mean_dist = 0.0
        std_dist = 0.0
        cohesion = float("inf")

    # === INTENTIONAL ANNOTATIONS ===

    # Get role-based function
    role_info = ROLE_FUNCTIONS.get(
        dominant_role,
        {
            "function": "General Musical Content",
            "intention": "To provide musical material",
            "phenomenology": "Music is present",
        },
    )

    # Determine preference type from genre
    genre_info = GENRE_PREFERENCE_MAP.get(
        dominant_genre,
        {
            "preference_type": "mixed",
            "target_context": "General listening",
        },
    )

    # Classify arousal level using composite score (replaces conjunctive AND logic)
    arousal_level, arousal_score = compute_arousal_score(
        mean_bpm=mean_bpm,
        mean_velocity=mean_velocity,
        mean_density=mean_density,
    )

    # === BUILD PROFILE ===

    profile = ClusterProfile(
        cluster_id=cluster_id,
        sample_count=n_samples,
        dominant_genre=dominant_genre,
        genre_percentage=genre_pct,
        dominant_role=dominant_role,
        role_percentage=role_pct,
        dominant_instrument=dominant_inst,
        instrument_percentage=inst_pct,
        mean_bpm=mean_bpm,
        std_bpm=std_bpm,
        mean_note_count=mean_notes,
        mean_density=mean_density,
        mean_velocity=mean_velocity,
        mean_polyphony=mean_polyphony,
        mean_intra_distance=mean_dist,
        std_intra_distance=std_dist,
        cohesion_score=cohesion,
        communicative_function=role_info["function"],
        artistic_intention=role_info["intention"],
        phenomenological_quality=role_info["phenomenology"],
        preference_type=genre_info["preference_type"],
        arousal_level=arousal_level,
        arousal_score=arousal_score,
        genre_distribution=genre_counts.to_dict(),
        role_distribution=role_counts.to_dict(),
        instrument_top5=inst_top5,
    )

    return profile


def compute_inter_cluster_distances(
    merged_df: pd.DataFrame,
    feature_columns: List[str],
    distance_metric: str = "euclidean",
) -> pd.DataFrame:
    """
    Compute mean pairwise distances between cluster centroids.

    This provides a measure of how well-separated clusters are
    in the feature space, informing interpretation of cluster
    distinctiveness.

    Args:
        merged_df: Full dataset with cluster assignments
        feature_columns: Numeric feature columns
        distance_metric: Distance metric to use

    Returns:
        DataFrame with cluster-to-cluster distance matrix
    """
    clusters = sorted(merged_df["GHSOM_cluster"].unique())
    n_clusters = len(clusters)

    # Compute cluster centroids
    centroids = {}
    for cluster_id in clusters:
        cluster_data = merged_df[merged_df["GHSOM_cluster"] == cluster_id]
        features = cluster_data[feature_columns].values
        features = np.nan_to_num(features, nan=0.0)
        centroids[cluster_id] = np.mean(features, axis=0)

    # Compute pairwise centroid distances
    centroid_matrix = np.array([centroids[c] for c in clusters])
    distance_matrix = squareform(pdist(centroid_matrix, metric=distance_metric))

    return pd.DataFrame(distance_matrix, index=clusters, columns=clusters)


def compute_global_quality_metrics(
    merged_df: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, float]:
    """
    Compute global clustering quality metrics.

    Computes:
    - Silhouette coefficient (cluster cohesion vs separation)
    - Davies-Bouldin index (ratio of within-cluster to between-cluster scatter)
    - Calinski-Harabasz index (ratio of between/within cluster variance)

    Args:
        merged_df: Full dataset with cluster assignments
        feature_columns: Numeric feature columns

    Returns:
        Dictionary of quality metrics
    """
    try:
        from sklearn.metrics import (
            silhouette_score,
            davies_bouldin_score,
            calinski_harabasz_score,
        )

        features = merged_df[feature_columns].values
        features = np.nan_to_num(features, nan=0.0)
        labels = merged_df["GHSOM_cluster"].values

        silhouette = silhouette_score(features, labels)
        dbi = davies_bouldin_score(features, labels)
        chi = calinski_harabasz_score(features, labels)

        return {
            "silhouette_coefficient": float(silhouette),
            "davies_bouldin_index": float(dbi),
            "calinski_harabasz_index": float(chi),
        }
    except Exception as e:
        print(f"Warning: Could not compute quality metrics: {e}")
        return {
            "silhouette_coefficient": 0.0,
            "davies_bouldin_index": 0.0,
            "calinski_harabasz_index": 0.0,
        }


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_markdown_report(
    profiles: List[ClusterProfile],
    summary: InterpretationSummary,
    output_path: Path,
) -> None:
    """
    Generate human-readable interpretation report in Markdown format.

    The report provides:
    - Executive summary with key findings
    - Intentionality distribution analysis
    - Detailed per-cluster interpretations
    - Phenomenological quality descriptions

    Args:
        profiles: List of all cluster profiles
        summary: Aggregated interpretation summary
        output_path: Where to save the report
    """
    lines = [
        "# Cluster Interpretation Report: Artistic Intention & Musical Preference",
        "",
        f"**Generated:** {summary.analysis_timestamp}",
        f"**Total Clusters:** {summary.total_clusters}",
        f"**Total Samples:** {summary.total_samples}",
        "",
        "## ⚠️ Important Limitations",
        "",
        "This report provides an **exploratory interpretive framework** for GHSOM clustering results.",
        "Semantic interpretations (role-function mappings, preference types) are based on music theory",
        "conventions and dataset characteristics, **NOT empirically validated through user studies**.",
        "",
        "**Key Limitations:**",
        "- Role-function mappings are theoretical, not perceptually validated",
        "- Dataset has 90% extrinsic (cinematic) bias",
        "- Arousal scores use heuristic weights, not validated against listener perception",
        "",
        "**Recommendation:** Treat interpretations as analytical lens, not scientific fact.",
        "Conduct user studies for publication-grade validation.",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report interprets GHSOM clustering results through the lens of artistic intention,",
        "communicative function, and listener preference addressability. Each cluster represents",
        "a distinct *mode of musical address*—a way that music speaks to or creates space for listeners.",
        "",
        "### Communicative Function Distribution",
        "",
        "| Function | Cluster Count | Description |",
        "|----------|---------------|-------------|",
    ]

    for func, count in sorted(
        summary.function_distribution.items(), key=lambda x: -x[1]
    ):
        func_desc = {
            "Direct Address": "Music that speaks directly to the listener",
            "Narrative Support": "Music that colors and extends the main message",
            "Harmonic Foundation": "Music that provides stability and context",
            "Atmospheric Creation": "Music that creates space and emotional context",
            "Motoric Drive": "Music that propels forward motion",
            "Low-End Anchoring": "Music that provides weight and gravitas",
        }.get(func, "")
        lines.append(f"| {func} | {count} | {func_desc} |")

    lines.extend(
        [
            "",
            "### Preference Type Distribution",
            "",
            "| Type | Cluster Count | Description |",
            "|------|---------------|-------------|",
        ]
    )

    for ptype, count in sorted(
        summary.preference_distribution.items(), key=lambda x: -x[1]
    ):
        ptype_desc = {
            "intrinsic": "Satisfaction from musical structure itself",
            "extrinsic": "Music serving external functions (film, ritual)",
            "mixed": "Both intrinsic and extrinsic appeal",
        }.get(ptype, "")
        lines.append(f"| {ptype.capitalize()} | {count} | {ptype_desc} |")

    lines.extend(
        [
            "",
            "### Arousal Level Distribution",
            "",
            "| Level | Cluster Count |",
            "|-------|---------------|",
        ]
    )

    for level, count in sorted(summary.arousal_distribution.items()):
        lines.append(f"| {level.capitalize()} | {count} |")

    lines.extend(
        [
            "",
            "### Quality Metrics",
            "",
            f"- **Silhouette Coefficient:** {summary.mean_silhouette:.3f}",
            f"- **Davies-Bouldin Index:** {summary.davies_bouldin_index:.3f}",
            f"- **Calinski-Harabasz Index:** {summary.calinski_harabasz_index:.1f}",
            "",
            "---",
            "",
            "## Detailed Cluster Interpretations",
            "",
        ]
    )

    # Sort profiles by sample count for importance ordering
    sorted_profiles = sorted(profiles, key=lambda p: -p.sample_count)

    for profile in sorted_profiles:
        lines.extend(
            [
                f"### Cluster {profile.cluster_id}",
                "",
                f"**Sample Count:** {profile.sample_count}",
                "",
                "#### Musical Characteristics",
                "",
                f"| Attribute | Value |",
                f"|-----------|-------|",
                f"| Genre | {profile.dominant_genre} ({profile.genre_percentage:.0f}%) |",
                f"| Role | {profile.dominant_role} ({profile.role_percentage:.0f}%) |",
                f"| Instrument | {profile.dominant_instrument} ({profile.instrument_percentage:.0f}%) |",
                f"| Tempo | {profile.mean_bpm:.0f} BPM |",
                f"| Note Density | {profile.mean_density:.2f} notes/sec |",
                f"| Velocity | {profile.mean_velocity:.0f} |",
                f"| Polyphony | {profile.mean_polyphony:.2f} |",
                "",
                "#### Intentional Interpretation",
                "",
                f"- **Communicative Function:** {profile.communicative_function}",
                f"- **Artistic Intention:** {profile.artistic_intention}",
                f'- **Phenomenological Quality:** *"{profile.phenomenological_quality}"*',
                f"- **Preference Type:** {profile.preference_type.capitalize()}",
                f"- **Arousal Level:** {profile.arousal_level.capitalize()}",
                "",
                "#### Cohesion Analysis",
                "",
                f"- **Mean Intra-Cluster Distance:** {profile.mean_intra_distance:.4f}",
                f"- **Cohesion Score:** {profile.cohesion_score:.2f}",
                "",
                "---",
                "",
            ]
        )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  → Report saved to: {output_path}")


def generate_visualization(
    profiles: List[ClusterProfile],
    output_dir: Path,
) -> None:
    """
    Generate visualization plots for cluster interpretation.

    Creates:
    - Function distribution bar chart
    - Arousal vs cluster size scatter
    - Cohesion vs function heatmap

    Args:
        profiles: List of cluster profiles
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert profiles to DataFrame for easier plotting
    df = pd.DataFrame([asdict(p) for p in profiles])

    # === Plot 1: Communicative Function Distribution ===
    fig, ax = plt.subplots(figsize=(10, 6))
    function_counts = df["communicative_function"].value_counts()
    function_counts.plot(
        kind="barh", ax=ax, color=sns.color_palette("husl", len(function_counts))
    )
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Communicative Function")
    ax.set_title("Distribution of Communicative Functions Across Clusters")
    plt.tight_layout()
    plt.savefig(output_dir / "function_distribution.png", dpi=150)
    plt.close()

    # === Plot 2: Cluster Size vs Arousal Level ===
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"low": "blue", "medium": "green", "high": "red"}
    for arousal in ["low", "medium", "high"]:
        subset = df[df["arousal_level"] == arousal]
        ax.scatter(
            subset["mean_bpm"],
            subset["sample_count"],
            c=colors[arousal],
            label=arousal.capitalize(),
            s=100,
            alpha=0.7,
        )
    ax.set_xlabel("Mean BPM")
    ax.set_ylabel("Sample Count")
    ax.set_title("Cluster Size by Tempo and Arousal Level")
    ax.legend(title="Arousal")
    plt.tight_layout()
    plt.savefig(output_dir / "arousal_by_tempo.png", dpi=150)
    plt.close()

    # === Plot 3: Cohesion by Role ===
    fig, ax = plt.subplots(figsize=(12, 6))
    role_order = list(ROLE_FUNCTIONS.keys())
    available_roles = [r for r in role_order if r in df["dominant_role"].values]
    df_subset = df[df["dominant_role"].isin(available_roles)]

    if len(df_subset) > 0:
        sns.boxplot(
            data=df_subset,
            x="dominant_role",
            y="mean_intra_distance",
            order=available_roles,
            ax=ax,
            palette="husl",
        )
        ax.set_xlabel("Dominant Track Role")
        ax.set_ylabel("Mean Intra-Cluster Distance")
        ax.set_title("Cluster Cohesion by Musical Role")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "cohesion_by_role.png", dpi=150)
    plt.close()

    print(f"  → Visualizations saved to: {output_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_interpretation_pipeline(
    model_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
    distance_metric: str = "cosine",
) -> Tuple[List[ClusterProfile], InterpretationSummary]:
    """
    Execute the complete cluster interpretation pipeline.

    Pipeline Steps:
    1. Load and merge cluster assignments with metadata
    2. Identify numeric feature columns for similarity analysis
    3. Generate profile for each cluster
    4. Compute global quality metrics
    5. Aggregate intentionality statistics
    6. Generate reports and visualizations

    Args:
        model_dir: Path to GHSOM model output
        metadata_csv: Path to feature metadata CSV
        output_dir: Directory for output artifacts
        distance_metric: Metric for distance computation

    Returns:
        Tuple of (list of ClusterProfiles, InterpretationSummary)
    """
    print("=" * 70)
    print("GHSOM CLUSTER INTERPRETATION PIPELINE")
    print("=" * 70)
    print(f"Model Directory: {model_dir}")
    print(f"Metadata CSV: {metadata_csv}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Load Data ===
    print("\n[Step 1/7] Loading cluster data with training features...")
    merged_df, feature_columns = load_cluster_data(Path(model_dir), Path(metadata_csv))
    print(f"  → Using {len(feature_columns)} training features for cohesion calculation")

    # (Step 2 merged into Step 1 - feature loading now integrated)

    # === STEP 3: Generate Cluster Profiles ===
    print("\n[Step 3/6] Generating cluster profiles...")
    profiles = []
    cluster_ids = sorted(merged_df["GHSOM_cluster"].unique())

    for cluster_id in cluster_ids:
        cluster_data = merged_df[merged_df["GHSOM_cluster"] == cluster_id]
        profile = compute_cluster_profile(
            cluster_id=cluster_id,
            cluster_data=cluster_data,
            feature_columns=feature_columns,
            distance_metric=distance_metric,
        )
        profiles.append(profile)
        print(
            f"  → Cluster {cluster_id}: {profile.sample_count} samples, "
            f"role={profile.dominant_role}, arousal={profile.arousal_level}"
        )

    # === STEP 4: Compute Quality Metrics ===
    print("\n[Step 4/6] Computing global quality metrics...")
    quality_metrics = compute_global_quality_metrics(merged_df, feature_columns)
    print(f"  → Silhouette: {quality_metrics['silhouette_coefficient']:.3f}")
    print(f"  → Davies-Bouldin: {quality_metrics['davies_bouldin_index']:.3f}")
    print(f"  → Calinski-Harabasz: {quality_metrics['calinski_harabasz_index']:.1f}")

    # === STEP 5: Aggregate Statistics ===
    print("\n[Step 5/6] Aggregating interpretation statistics...")

    function_dist = Counter(p.communicative_function for p in profiles)
    preference_dist = Counter(p.preference_type for p in profiles)
    arousal_dist = Counter(p.arousal_level for p in profiles)

    summary = InterpretationSummary(
        total_clusters=len(profiles),
        total_samples=len(merged_df),
        analysis_timestamp=datetime.now().isoformat(),
        function_distribution=dict(function_dist),
        preference_distribution=dict(preference_dist),
        arousal_distribution=dict(arousal_dist),
        mean_silhouette=quality_metrics["silhouette_coefficient"],
        davies_bouldin_index=quality_metrics["davies_bouldin_index"],
        calinski_harabasz_index=quality_metrics["calinski_harabasz_index"],
    )

    # === STEP 6: Generate Outputs ===
    print("\n[Step 6/6] Generating outputs...")

    # Save profiles as CSV
    profiles_df = pd.DataFrame(
        [
            {
                "cluster_id": p.cluster_id,
                "sample_count": p.sample_count,
                "dominant_genre": p.dominant_genre,
                "genre_percentage": p.genre_percentage,
                "dominant_role": p.dominant_role,
                "role_percentage": p.role_percentage,
                "dominant_instrument": p.dominant_instrument,
                "instrument_percentage": p.instrument_percentage,
                "mean_bpm": p.mean_bpm,
                "mean_density": p.mean_density,
                "mean_velocity": p.mean_velocity,
                "mean_polyphony": p.mean_polyphony,
                "mean_intra_distance": p.mean_intra_distance,
                "cohesion_score": p.cohesion_score,
                "communicative_function": p.communicative_function,
                "artistic_intention": p.artistic_intention,
                "phenomenological_quality": p.phenomenological_quality,
                "preference_type": p.preference_type,
                "arousal_level": p.arousal_level,
                "arousal_score": p.arousal_score,
            }
            for p in profiles
        ]
    )
    profiles_df.to_csv(output_dir / "cluster_profiles.csv", index=False)
    print(f"  → Saved cluster profiles to: {output_dir / 'cluster_profiles.csv'}")

    # === STEP 6.5: VALIDATE PROFILES ===
    print("\n[Step 6.5/7] Validating profile-model alignment...")
    profiles_csv = output_dir / "cluster_profiles.csv"

    is_valid, validation_report = validate_model_profile_alignment(
        model_dir=Path(model_dir),
        profiles_csv=profiles_csv
    )

    if not is_valid:
        print(f"\n{'='*70}")
        print("⚠️  WARNING: Profile-Model Alignment Issues Detected")
        print(f"{'='*70}")

        if validation_report.get('missing_profiles'):
            print(f"\n❌ Missing profiles for {len(validation_report['missing_profiles'])} clusters:")
            print(f"   {validation_report['missing_profiles']}")

        if validation_report.get('extra_profiles'):
            print(f"\n❌ Extra profiles for {len(validation_report['extra_profiles'])} non-existent clusters:")
            print(f"   {validation_report['extra_profiles']}")

        if validation_report.get('sample_count_mismatches'):
            print(f"\n❌ Sample count mismatches for {len(validation_report['sample_count_mismatches'])} clusters:")
            for mismatch in validation_report['sample_count_mismatches'][:5]:  # Show first 5
                print(f"   Cluster {mismatch['cluster_id']}: "
                      f"model={mismatch['model_count']}, profile={mismatch['profile_count']} "
                      f"({mismatch['error_pct']:.1f}% error)")

        # Save validation report
        with open(output_dir / "validation_report.json", "w") as f:
            json.dump(validation_report, f, indent=2)
        print(f"\n📄 Full validation report saved to: {output_dir / 'validation_report.json'}")
        print(f"{'='*70}\n")
    else:
        print("  ✅ Profile-model alignment validated successfully")
        # Save validation report even on success
        with open(output_dir / "validation_report.json", "w") as f:
            json.dump(validation_report, f, indent=2)

    # Save summary as JSON
    summary_dict = asdict(summary)
    with open(output_dir / "interpretation_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2, default=str)
    print(f"  → Saved summary to: {output_dir / 'interpretation_summary.json'}")

    # Generate Markdown report
    generate_markdown_report(profiles, summary, output_dir / "interpretation_report.md")

    # Generate visualizations
    generate_visualization(profiles, output_dir / "plots")

    # === COMPLETE ===
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Clusters analyzed: {len(profiles)}")
    print(f"Total samples: {len(merged_df)}")
    print("=" * 70)

    return profiles, summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GHSOM Cluster Interpretation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python analysis/run_interpret_results.py
    
    # Specify custom model directory
    python analysis/run_interpret_results.py --model-dir /path/to/ghsom/output
    
    # Use custom config file
    python analysis/run_interpret_results.py --config path/to/config.json
    
    # Specify all paths explicitly
    python analysis/run_interpret_results.py \\
        --model-dir /path/to/ghsom/output \\
        --metadata-csv /path/to/features.csv \\
        --output-dir /path/to/output
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (overrides defaults)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to GHSOM model output directory",
    )

    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Path to features_with_metadata.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results",
    )

    parser.add_argument(
        "--distance-metric",
        type=str,
        default="euclidean",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance metric for similarity computation (default: cosine)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the interpretation pipeline."""
    args = parse_arguments()

    # Load configuration
    config = DEFAULT_CONFIG.copy()

    # Override with config file if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config)

    # Override with command-line arguments
    if args.model_dir:
        config["model_dir"] = args.model_dir
    if args.metadata_csv:
        config["metadata_csv"] = args.metadata_csv
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.distance_metric:
        config["distance_metric"] = args.distance_metric

    # Run pipeline
    profiles, summary = run_interpretation_pipeline(
        model_dir=Path(config["model_dir"]),
        metadata_csv=Path(config["metadata_csv"]),
        output_dir=Path(config["output_dir"]),
        distance_metric=config["distance_metric"],
    )

    return profiles, summary


if __name__ == "__main__":
    main()
