#!/usr/bin/env python3
"""
Comprehensive EDA Script with Visualization Export
This script performs exploratory data analysis on music generation datasets
and saves all visualizations to a specified output directory.

Usage:
    python run_eda.py --metadata /path/to/metadata.csv --output /path/to/output
    python run_eda.py  # Uses defaults: ComMU metadata, output to /workspace/outputs/data/commu_eda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import warnings
from collections import Counter
import json
import argparse
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.logging.logging_manager import LoggingManager


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive EDA for music generation datasets"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/workspace/data/raw/commu/metadata.csv",
        help="Path to metadata CSV file (default: /workspace/data/raw/commu/metadata.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/outputs/eda/commu_eda",
        help="Output directory for visualizations (default: /workspace/outputs/data/commu_eda)",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Optional log file path"
    )
    return parser.parse_args()


def setup_visualization():
    """Configure visualization settings."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 10
    warnings.filterwarnings("ignore")
    np.random.seed(42)


def load_dataset(metadata_path, logger):
    """Load dataset from metadata file."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MUSIC DATASET EDA")
    logger.info("=" * 80)
    logger.info("Loading dataset from: %s", metadata_path)

    try:
        df = pd.read_csv(metadata_path)
        logger.info("Successfully loaded dataset: %d samples", len(df))
        logger.info("Columns: %s", ", ".join(df.columns.tolist()))
        return df
    except Exception as e:
        logger.error("Failed to load dataset: %s", str(e))
        raise


def parse_genres(df, logger):
    """Parse genre field which may contain multiple genres separated by |."""
    if "genre" not in df.columns:
        logger.warning("No genre column found in dataset")
        return pd.Series()

    all_genres = []
    for genre in df["genre"].dropna():
        if "|" in str(genre):
            all_genres.extend(str(genre).split("|"))
        else:
            all_genres.append(str(genre))

    genre_counts = pd.Series(all_genres).value_counts()
    logger.debug("Found %d unique genres", len(genre_counts))
    return genre_counts


def analyze_dataset_composition(df, output_dir, logger):
    """Analyze and visualize dataset composition."""
    logger.info("=" * 80)
    logger.info("1. DATASET COMPOSITION ANALYSIS")
    logger.info("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Source distribution
    if "source" in df.columns:
        source_counts = df["source"].value_counts()
        axes[0].pie(
            source_counts.values,
            labels=source_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0].set_title("Dataset Source Distribution", fontsize=14, fontweight="bold")
        logger.debug("Source distribution: %s", dict(source_counts))
    else:
        axes[0].text(0.5, 0.5, "No source column", ha="center", va="center")
        axes[0].set_title("Dataset Source Distribution", fontsize=14, fontweight="bold")

    # Split distribution
    if "split" in df.columns:
        split_counts = df["split"].value_counts()
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        axes[1].bar(
            split_counts.index, split_counts.values, color=colors[: len(split_counts)]
        )
        axes[1].set_title("Train/Validation/Test Split", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Number of Samples")
        axes[1].set_xlabel("Split")
        for i, v in enumerate(split_counts.values):
            axes[1].text(
                i,
                v + len(df) * 0.01,
                f"{v:,}\n({v/len(df)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        logger.info("Split distribution: %s", dict(split_counts))
    else:
        axes[1].text(0.5, 0.5, "No split column", ha="center", va="center")
        axes[1].set_title("Train/Validation/Test Split", fontsize=14, fontweight="bold")

    # Split distribution by source
    if "source" in df.columns and "split" in df.columns:
        split_source = pd.crosstab(df["split"], df["source"])
        split_source.plot(
            kind="bar", stacked=True, ax=axes[2], color=["#9b59b6", "#f39c12"]
        )
        axes[2].set_title(
            "Split Distribution by Source", fontsize=14, fontweight="bold"
        )
        axes[2].set_ylabel("Number of Samples")
        axes[2].set_xlabel("Split")
        axes[2].legend(title="Source")
        axes[2].tick_params(axis="x", rotation=0)
    else:
        axes[2].text(0.5, 0.5, "No source/split columns", ha="center", va="center")
        axes[2].set_title(
            "Split Distribution by Source", fontsize=14, fontweight="bold"
        )

    plt.tight_layout()
    output_path = output_dir / "01_dataset_composition.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def analyze_genre_and_style(df, genre_counts, output_dir, logger):
    """Analyze and visualize genre distribution."""
    logger.info("=" * 80)
    logger.info("2. GENRE AND STYLE ANALYSIS")
    logger.info("=" * 80)

    if genre_counts.empty:
        logger.warning("No genre data available, skipping genre analysis")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Genre distribution bar chart
    top_genres = genre_counts.head(15)
    axes[0].barh(
        range(len(top_genres)),
        top_genres.values,
        color=sns.color_palette("viridis", len(top_genres)),
    )
    axes[0].set_yticks(range(len(top_genres)))
    axes[0].set_yticklabels(top_genres.index)
    axes[0].set_xlabel("Number of Tracks")
    axes[0].set_title("Top 15 Genres in Dataset", fontsize=14, fontweight="bold")
    axes[0].invert_yaxis()
    for i, v in enumerate(top_genres.values):
        axes[0].text(v + max(top_genres.values) * 0.02, i, f"{v:,}", va="center")
    logger.info("Top 5 genres: %s", ", ".join(top_genres.head(5).index.tolist()))

    # Genre distribution by source
    if "source" in df.columns:
        genre_source_data = []
        for source in df["source"].unique():
            source_df = df[df["source"] == source]
            source_genres = parse_genres(source_df, logger)
            genre_source_data.append(source_genres.head(10))

        genre_comparison = pd.DataFrame(genre_source_data).T.fillna(0)
        genre_comparison.columns = df["source"].unique()
        genre_comparison.plot(kind="bar", ax=axes[1], color=["#9b59b6", "#f39c12"])
        axes[1].set_title("Top Genres by Source", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Number of Tracks")
        axes[1].set_xlabel("Genre")
        axes[1].legend(title="Source")
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No source column",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Top Genres by Source", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "02_genre_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def analyze_bpm(df, genre_counts, output_dir, logger):
    """Analyze and visualize BPM distribution."""
    logger.info("=" * 80)
    logger.info("3. BPM (TEMPO) ANALYSIS")
    logger.info("=" * 80)

    if "bpm" not in df.columns:
        logger.warning("No bpm column found, skipping BPM analysis")
        return

    # Prepare genre BPM data
    top_genre_list = (
        genre_counts.head(8).index.tolist() if not genre_counts.empty else []
    )
    bpm_genre_data = []
    genre_labels = []
    if "genre" in df.columns:
        for genre in top_genre_list:
            mask = df["genre"].str.contains(genre, na=False, case=False)
            genre_bpm = df[mask]["bpm"].dropna()
            if len(genre_bpm) > 0:
                bpm_genre_data.append(genre_bpm)
                genre_labels.append(genre)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Overall BPM distribution
    axes[0, 0].hist(
        df["bpm"].dropna(), bins=50, color="#3498db", edgecolor="black", alpha=0.7
    )
    axes[0, 0].axvline(
        df["bpm"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Median: {df["bpm"].median():.1f}',
    )
    axes[0, 0].axvline(
        df["bpm"].mean(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["bpm"].mean():.1f}',
    )
    axes[0, 0].set_xlabel("BPM (Beats Per Minute)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("BPM Distribution", fontsize=14, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # BPM by source
    if "source" in df.columns:
        sources = df["source"].unique()
        bpm_by_source = [df[df["source"] == src]["bpm"].dropna() for src in sources]
        axes[0, 1].violinplot(
            bpm_by_source,
            positions=range(len(sources)),
            showmeans=True,
            showmedians=True,
        )
        axes[0, 1].set_xticks(range(len(sources)))
        axes[0, 1].set_xticklabels(sources)
        axes[0, 1].set_ylabel("BPM")
        axes[0, 1].set_title(
            "BPM Distribution by Source", fontsize=14, fontweight="bold"
        )
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No source column",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title(
            "BPM Distribution by Source", fontsize=14, fontweight="bold"
        )

    # BPM by genre
    if len(bpm_genre_data) > 0:
        bp = axes[1, 0].boxplot(bpm_genre_data, labels=genre_labels, patch_artist=True)
        for patch, color in zip(
            bp["boxes"], sns.color_palette("Set2", len(genre_labels))
        ):
            patch.set_facecolor(color)
        axes[1, 0].set_xlabel("Genre")
        axes[1, 0].set_ylabel("BPM")
        axes[1, 0].set_title(
            "BPM Distribution by Genre", fontsize=14, fontweight="bold"
        )
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No genre data",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title(
            "BPM Distribution by Genre", fontsize=14, fontweight="bold"
        )

    # BPM statistics by split
    if "split" in df.columns:
        split_stats = df.groupby("split")["bpm"].agg(
            ["mean", "median", "std", "min", "max"]
        )
        x = np.arange(len(split_stats.index))
        width = 0.35
        axes[1, 1].bar(
            x - width / 2,
            split_stats["mean"],
            width,
            label="Mean",
            color="#3498db",
        )
        axes[1, 1].bar(
            x + width / 2,
            split_stats["median"],
            width,
            label="Median",
            color="#2ecc71",
        )
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(split_stats.index)
        axes[1, 1].set_ylabel("BPM")
        axes[1, 1].set_title("BPM Statistics by Split", fontsize=14, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No split column",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("BPM Statistics by Split", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "03_bpm_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)

    logger.info("BPM Statistics:")
    logger.info("  Mean: %.1f", df["bpm"].mean())
    logger.info("  Median: %.1f", df["bpm"].median())
    logger.info("  Range: %.1f - %.1f", df["bpm"].min(), df["bpm"].max())


def analyze_pitch_range(df, output_dir, logger):
    """Analyze and visualize pitch range distribution."""
    logger.info("=" * 80)
    logger.info("4. PITCH RANGE ANALYSIS")
    logger.info("=" * 80)

    if "pitch_range" not in df.columns:
        logger.warning("No pitch_range column found, skipping pitch range analysis")
        return

    pitch_range_mapping = {"low": 1, "mid_low": 2, "mid": 3, "mid_high": 4, "high": 5}

    df["pitch_range_numeric"] = df["pitch_range"].map(pitch_range_mapping)
    df["pitch_range_value"] = pd.to_numeric(df["pitch_range"], errors="coerce")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Pitch range distribution (categorical)
    if df["pitch_range_numeric"].notna().any():
        pitch_counts = df["pitch_range"].value_counts()
        axes[0, 0].bar(
            range(len(pitch_counts)),
            pitch_counts.values,
            color=sns.color_palette("coolwarm", len(pitch_counts)),
        )
        axes[0, 0].set_xticks(range(len(pitch_counts)))
        axes[0, 0].set_xticklabels(pitch_counts.index, rotation=45)
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title(
            "Pitch Range Distribution (Categorical)", fontsize=14, fontweight="bold"
        )
        for i, v in enumerate(pitch_counts.values):
            axes[0, 0].text(
                i,
                v + max(pitch_counts.values) * 0.02,
                f"{v:,}",
                ha="center",
                va="bottom",
            )
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No categorical pitch data",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title(
            "Pitch Range Distribution (Categorical)", fontsize=14, fontweight="bold"
        )

    # Pitch range distribution (numeric)
    if df["pitch_range_value"].notna().any():
        axes[0, 1].hist(
            df["pitch_range_value"].dropna(),
            bins=30,
            color="#9b59b6",
            edgecolor="black",
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Pitch Range (Semitones)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title(
            "Pitch Range Distribution (Numeric)", fontsize=14, fontweight="bold"
        )
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No numeric pitch data",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title(
            "Pitch Range Distribution (Numeric)", fontsize=14, fontweight="bold"
        )

    # Pitch range by source
    if "source" in df.columns and df["pitch_range_numeric"].notna().any():
        pitch_source = pd.crosstab(df["pitch_range"], df["source"])
        pitch_source.plot(kind="bar", ax=axes[1, 0], color=["#9b59b6", "#f39c12"])
        axes[1, 0].set_title("Pitch Range by Source", fontsize=14, fontweight="bold")
        axes[1, 0].set_ylabel("Number of Tracks")
        axes[1, 0].set_xlabel("Pitch Range")
        axes[1, 0].legend(title="Source")
        axes[1, 0].tick_params(axis="x", rotation=45)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No source column",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Pitch Range by Source", fontsize=14, fontweight="bold")

    # Pitch range vs BPM scatter
    if df["pitch_range_numeric"].notna().any() and "bpm" in df.columns:
        scatter_data = df[df["pitch_range_numeric"].notna()]
        axes[1, 1].scatter(
            scatter_data["bpm"],
            scatter_data["pitch_range_numeric"],
            alpha=0.5,
            c=scatter_data["pitch_range_numeric"],
            cmap="coolwarm",
            s=20,
        )
        axes[1, 1].set_xlabel("BPM")
        axes[1, 1].set_ylabel("Pitch Range (Encoded)")
        axes[1, 1].set_title("Pitch Range vs BPM", fontsize=14, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No pitch/BPM data",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Pitch Range vs BPM", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "04_pitch_range_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def analyze_time_signature(df, genre_counts, output_dir, logger):
    """Analyze and visualize time signature distribution."""
    logger.info("=" * 80)
    logger.info("5. TIME SIGNATURE ANALYSIS")
    logger.info("=" * 80)

    if "time_signature" not in df.columns:
        logger.warning(
            "No time_signature column found, skipping time signature analysis"
        )
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time signature distribution
    time_sig_counts = df["time_signature"].value_counts()
    axes[0].pie(
        time_sig_counts.values,
        labels=time_sig_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set3", len(time_sig_counts)),
    )
    axes[0].set_title("Time Signature Distribution", fontsize=14, fontweight="bold")

    # Time signature by genre
    genre_labels = genre_counts.head(5).index.tolist() if not genre_counts.empty else []
    if len(genre_labels) > 0 and "genre" in df.columns:
        time_sig_genre_data = []
        for genre in genre_labels:
            mask = df["genre"].str.contains(genre, na=False, case=False)
            genre_time_sigs = df[mask]["time_signature"].value_counts()
            time_sig_genre_data.append(genre_time_sigs)

        time_sig_df = pd.DataFrame(time_sig_genre_data, index=genre_labels).fillna(0).T
        time_sig_df.plot(kind="bar", ax=axes[1], stacked=True)
        axes[1].set_title(
            "Time Signature by Top Genres", fontsize=14, fontweight="bold"
        )
        axes[1].set_ylabel("Number of Tracks")
        axes[1].set_xlabel("Time Signature")
        axes[1].legend(title="Genre", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[1].tick_params(axis="x", rotation=0)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No genre data",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title(
            "Time Signature by Top Genres", fontsize=14, fontweight="bold"
        )

    # Time signature vs BPM
    if "bpm" in df.columns:
        time_sig_bpm = df.groupby("time_signature")["bpm"].agg(["mean", "std"])
        x = np.arange(len(time_sig_bpm.index))
        axes[2].bar(
            x,
            time_sig_bpm["mean"],
            yerr=time_sig_bpm["std"],
            capsize=5,
            color="#3498db",
            alpha=0.7,
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(time_sig_bpm.index)
        axes[2].set_ylabel("Average BPM")
        axes[2].set_xlabel("Time Signature")
        axes[2].set_title(
            "Average BPM by Time Signature", fontsize=14, fontweight="bold"
        )
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No BPM data",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title(
            "Average BPM by Time Signature", fontsize=14, fontweight="bold"
        )

    plt.tight_layout()
    output_path = output_dir / "05_time_signature_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def analyze_measures(df, output_dir, logger):
    """Analyze and visualize number of measures distribution."""
    logger.info("=" * 80)
    logger.info("6. NUMBER OF MEASURES ANALYSIS")
    logger.info("=" * 80)

    if "num_measures" not in df.columns:
        logger.warning("No num_measures column found, skipping measures analysis")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Measures distribution
    axes[0].hist(
        df["num_measures"].dropna(),
        bins=20,
        color="#2ecc71",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].axvline(
        df["num_measures"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Median: {df["num_measures"].median():.0f}',
    )
    axes[0].axvline(
        df["num_measures"].mean(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["num_measures"].mean():.1f}',
    )
    axes[0].set_xlabel("Number of Measures")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Number of Measures Distribution", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Measures by source
    if "source" in df.columns:
        sources = df["source"].unique()
        measures_by_source = [
            df[df["source"] == src]["num_measures"].dropna() for src in sources
        ]
        bp = axes[1].boxplot(measures_by_source, labels=sources, patch_artist=True)
        for patch, color in zip(bp["boxes"], sns.color_palette("Set2", len(sources))):
            patch.set_facecolor(color)
        axes[1].set_ylabel("Number of Measures")
        axes[1].set_title(
            "Measures Distribution by Source", fontsize=14, fontweight="bold"
        )
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No source column",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title(
            "Measures Distribution by Source", fontsize=14, fontweight="bold"
        )

    # Measures vs BPM
    if "bpm" in df.columns:
        axes[2].scatter(df["num_measures"], df["bpm"], alpha=0.3, s=20, c="#e74c3c")
        axes[2].set_xlabel("Number of Measures")
        axes[2].set_ylabel("BPM")
        axes[2].set_title("Number of Measures vs BPM", fontsize=14, fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        corr = df[["num_measures", "bpm"]].corr().iloc[0, 1]
        axes[2].text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=axes[2].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "No BPM data",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Number of Measures vs BPM", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "06_measures_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def analyze_instrument_and_key(df, output_dir, logger):
    """Analyze and visualize instrument and key distributions."""
    logger.info("=" * 80)
    logger.info("7. INSTRUMENT AND AUDIO KEY ANALYSIS")
    logger.info("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Instrument distribution
    if "inst" in df.columns:
        inst_counts = df["inst"].value_counts().head(15)
        axes[0, 0].barh(
            range(len(inst_counts)),
            inst_counts.values,
            color=sns.color_palette("tab20", len(inst_counts)),
        )
        axes[0, 0].set_yticks(range(len(inst_counts)))
        axes[0, 0].set_yticklabels(inst_counts.index)
        axes[0, 0].set_xlabel("Frequency")
        axes[0, 0].set_title("Top 15 Instruments", fontsize=14, fontweight="bold")
        axes[0, 0].invert_yaxis()
        for i, v in enumerate(inst_counts.values):
            axes[0, 0].text(
                v + max(inst_counts.values) * 0.02, i, f"{v:,}", va="center"
            )
        logger.debug("Top instrument: %s", inst_counts.index[0])
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No instrument column",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title("Top 15 Instruments", fontsize=14, fontweight="bold")

    # Audio key distribution
    if "audio_key" in df.columns:
        key_counts = df["audio_key"].value_counts().head(15)
        axes[0, 1].bar(
            range(len(key_counts)),
            key_counts.values,
            color=sns.color_palette("rainbow", len(key_counts)),
        )
        axes[0, 1].set_xticks(range(len(key_counts)))
        axes[0, 1].set_xticklabels(key_counts.index, rotation=45, ha="right")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Top 15 Audio Keys", fontsize=14, fontweight="bold")
        for i, v in enumerate(key_counts.values):
            axes[0, 1].text(
                i,
                v + max(key_counts.values) * 0.02,
                f"{v:,}",
                ha="center",
                va="bottom",
                rotation=0,
            )
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No audio key column",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Top 15 Audio Keys", fontsize=14, fontweight="bold")

    # Track role distribution
    if "track_role" in df.columns:
        role_counts = df["track_role"].value_counts()
        axes[1, 0].pie(
            role_counts.values,
            labels=role_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel", len(role_counts)),
        )
        axes[1, 0].set_title("Track Role Distribution", fontsize=14, fontweight="bold")
    else:
        axes[1, 0].text(0.5, 0.5, "No track role column", ha="center", va="center")
        axes[1, 0].set_title("Track Role Distribution", fontsize=14, fontweight="bold")

    # Sample rhythm distribution
    if "sample_rhythm" in df.columns:
        rhythm_counts = df["sample_rhythm"].value_counts().head(10)
        axes[1, 1].bar(
            range(len(rhythm_counts)),
            rhythm_counts.values,
            color=sns.color_palette("muted", len(rhythm_counts)),
        )
        axes[1, 1].set_xticks(range(len(rhythm_counts)))
        axes[1, 1].set_xticklabels(rhythm_counts.index, rotation=45, ha="right")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Top 10 Sample Rhythms", fontsize=14, fontweight="bold")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No sample rhythm column",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Top 10 Sample Rhythms", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "07_instrument_key_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def analyze_feature_correlation(df, output_dir, logger):
    """Analyze and visualize feature correlations."""
    logger.info("=" * 80)
    logger.info("8. FEATURE CORRELATION ANALYSIS")
    logger.info("=" * 80)

    numeric_features = []

    # Check for numeric columns
    for col in [
        "bpm",
        "num_measures",
        "pitch_range_numeric",
        "min_velocity",
        "max_velocity",
    ]:
        if col in df.columns and df[col].notna().any():
            numeric_features.append(col)

    if len(numeric_features) < 2:
        logger.warning(
            "Insufficient numeric features for correlation analysis (need at least 2)"
        )
        return

    corr_matrix = df[numeric_features].corr()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        fmt=".3f",
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "08_feature_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)
    logger.debug("Analyzed %d numeric features", len(numeric_features))


def analyze_data_quality(df, output_dir, logger):
    """Analyze and visualize data quality metrics."""
    logger.info("=" * 80)
    logger.info("9. DATA QUALITY ASSESSMENT")
    logger.info("=" * 80)

    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Count": missing_values, "Percentage": missing_pct}
    ).sort_values("Missing Count", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Missing values bar chart
    missing_features = missing_df[missing_df["Missing Count"] > 0]
    if len(missing_features) > 0:
        axes[0].barh(
            range(len(missing_features)),
            missing_features["Percentage"].values,
            color="#e74c3c",
        )
        axes[0].set_yticks(range(len(missing_features)))
        axes[0].set_yticklabels(missing_features.index)
        axes[0].set_xlabel("Percentage Missing (%)")
        axes[0].set_title("Missing Values by Feature", fontsize=14, fontweight="bold")
        axes[0].invert_yaxis()
        for i, v in enumerate(missing_features["Percentage"].values):
            axes[0].text(v + 0.5, i, f"{v:.1f}%", va="center")
        logger.info("Features with missing data: %d", len(missing_features))
    else:
        axes[0].text(
            0.5,
            0.5,
            "No Missing Values",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("Missing Values by Feature", fontsize=14, fontweight="bold")
        logger.info("No missing values in dataset")

    # Data completeness by split
    if "split" in df.columns:
        completeness_by_split = []
        split_names = []
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            completeness = (
                1
                - split_df.isnull().sum().sum()
                / (len(split_df) * len(split_df.columns))
            ) * 100
            completeness_by_split.append(completeness)
            split_names.append(split)

        axes[1].bar(
            split_names,
            completeness_by_split,
            color=["#3498db", "#2ecc71", "#e74c3c"][: len(split_names)],
        )
        axes[1].set_ylabel("Completeness (%)")
        axes[1].set_title("Data Completeness by Split", fontsize=14, fontweight="bold")
        axes[1].set_ylim([0, 105])
        for i, v in enumerate(completeness_by_split):
            axes[1].text(
                i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold"
            )
        axes[1].grid(True, alpha=0.3, axis="y")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No split column",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("Data Completeness by Split", fontsize=14, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "09_data_quality.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path.name)


def parse_chord_progressions(df, sample_size=None, logger=None):
    """Parse chord progressions from dataset."""
    if "chord_progressions" not in df.columns:
        if logger:
            logger.warning("No chord_progressions column found")
        return Counter()

    all_chords = []
    df_sample = df.sample(n=min(sample_size or len(df), len(df)), random_state=42)

    for prog in df_sample["chord_progressions"].dropna():
        try:
            if isinstance(prog, str) and prog != "[]":
                chord_list = ast.literal_eval(prog)
                if isinstance(chord_list, list) and len(chord_list) > 0:
                    for item in chord_list:
                        if isinstance(item, list):
                            all_chords.extend(item)
                        else:
                            all_chords.append(item)
        except (ValueError, SyntaxError):
            continue

    return pd.Series(all_chords).value_counts() if all_chords else Counter()


def generate_summary_report(df, genre_counts, output_dir, logger):
    """Generate and save comprehensive summary report."""
    logger.info("=" * 80)
    logger.info("10. GENERATING SUMMARY REPORT")
    logger.info("=" * 80)

    # Parse chord progressions if available
    chord_counts = parse_chord_progressions(df, sample_size=10000, logger=logger)

    summary = {
        "Dataset Size": {
            "Total Samples": int(len(df)),
        },
        "Musical Characteristics": {},
        "Complexity Metrics": {},
    }

    # Add split counts if available
    if "split" in df.columns:
        for split in df["split"].unique():
            summary["Dataset Size"][f"{split.capitalize()} Samples"] = int(
                len(df[df["split"] == split])
            )

    # Add musical characteristics
    if "bpm" in df.columns:
        summary["Musical Characteristics"][
            "BPM Range"
        ] = f"{df['bpm'].min():.1f} - {df['bpm'].max():.1f}"
        summary["Musical Characteristics"]["Average BPM"] = f"{df['bpm'].mean():.1f}"

    if "genre" in df.columns:
        summary["Musical Characteristics"]["Unique Genres"] = int(df["genre"].nunique())

    if "inst" in df.columns:
        summary["Musical Characteristics"]["Unique Instruments"] = int(
            df["inst"].nunique()
        )

    if "time_signature" in df.columns and len(df) > 0:
        summary["Musical Characteristics"]["Most Common Time Signature"] = str(
            df["time_signature"].mode()[0]
        )

    # Add complexity metrics
    if "num_measures" in df.columns:
        summary["Complexity Metrics"][
            "Average Measures"
        ] = f"{df['num_measures'].mean():.1f}"
        summary["Complexity Metrics"][
            "Measures Range"
        ] = f"{df['num_measures'].min():.0f} - {df['num_measures'].max():.0f}"

    # Add source info if available
    if "source" in df.columns:
        summary["Data Sources"] = {
            k: int(v) for k, v in df["source"].value_counts().to_dict().items()
        }

    # Add chord progression info if available
    if len(chord_counts) > 0:
        if isinstance(chord_counts, Counter):
            most_common = chord_counts.most_common(1)[0][0]
        elif hasattr(chord_counts, "index"):
            most_common = chord_counts.index[0]
        else:
            most_common = "Unknown"
        summary["Harmonic Characteristics"] = {
            "Unique Chords": int(len(chord_counts)),
            "Most Common Chord": str(most_common),
        }

    # Save summary to JSON
    output_path = output_dir / "eda_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", output_path.name)

    # Log summary
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE DATASET SUMMARY")
    logger.info("=" * 80)
    for category, metrics in summary.items():
        logger.info("%s:", category)
        for key, value in metrics.items():
            logger.info("  %s: %s", key, value)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    log_file = args.log_file if args.log_file else output_dir / "eda.log"
    logger = LoggingManager(name="eda", log_file=log_file, enable_wandb=False)

    logger.info("Starting EDA analysis")
    logger.info("Metadata file: %s", args.metadata)
    logger.info("Output directory: %s", output_dir)

    # Setup visualization
    setup_visualization()

    try:
        # Load dataset
        df = load_dataset(args.metadata, logger)

        # Parse genres
        genre_counts = parse_genres(df, logger)

        # Run all analyses
        analyze_dataset_composition(df, output_dir, logger)
        analyze_genre_and_style(df, genre_counts, output_dir, logger)
        analyze_bpm(df, genre_counts, output_dir, logger)
        analyze_pitch_range(df, output_dir, logger)
        analyze_time_signature(df, genre_counts, output_dir, logger)
        analyze_measures(df, output_dir, logger)
        analyze_instrument_and_key(df, output_dir, logger)
        analyze_feature_correlation(df, output_dir, logger)
        analyze_data_quality(df, output_dir, logger)
        generate_summary_report(df, genre_counts, output_dir, logger)

        logger.info("=" * 80)
        logger.info("✅ EDA COMPLETE!")
        logger.info("=" * 80)
        logger.info("All visualizations saved to: %s", output_dir.resolve())
        logger.info(
            "Total files generated: 10 (9 PNG images + 1 JSON summary + 1 log file)"
        )

    except Exception as e:
        logger.error("EDA failed with error: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()
