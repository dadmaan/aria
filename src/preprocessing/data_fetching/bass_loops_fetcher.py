"""Utilities for normalising the Bass Loops dataset.

This module mirrors the public surface of :mod:`commu_fetcher` while
capturing the bespoke layout of the Lakh-derived "bass loops" dataset.
It exposes a single orchestration helper, :func:`curate_bass_loops`,
which performs the following steps:

1. Load split metadata stored as JSON under ``processed/<split>/``.
2. Normalise the schema so it matches the ComMu bass metadata contract.
3. Derive deterministic train/validation/test splits (seed driven).
4. Copy MIDI assets into a dedicated subset directory.
5. Emit both a raw metadata snapshot and the cleaned schema-compatible CSV.

The public API is intentionally test-friendly; pure helpers are exposed
so unit tests can exercise individual transformations without touching
large artefacts.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from src.utils.logging.logging_manager import get_logger
from .commu_fetcher import SplitConfig

LOGGER = get_logger(__name__)

COMM_BASS_COLUMNS: Sequence[str] = (
    "Unnamed: 0",
    "audio_key",
    "chord_progressions",
    "pitch_range",
    "num_measures",
    "bpm",
    "genre",
    "track_role",
    "inst",
    "sample_rhythm",
    "time_signature",
    "min_velocity",
    "max_velocity",
    "split_data",
    "id",
    "file_path",
    "split",
)

DEFAULT_RELATIVE_ROOT = "bass_loops_midi"
DEFAULT_NUM_MEASURES = 4
DEFAULT_SAMPLE_RHYTHM = "unknown"
DEFAULT_AUDIO_KEY = "unknown"
DEFAULT_CHORD_PROGRESSIONS = "[]"
DEFAULT_TRACK_ROLE = "bass_loop"
DEFAULT_INST = "bass_loop"


@dataclass(frozen=True)
class DatasetPaths:
    """Small convenience wrapper for dataset locations."""

    dataset_root: Path
    processed_dir: Path
    splits: Sequence[str]


def _format_genre(value: object) -> str:
    if isinstance(value, (list, tuple)):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return "|".join(cleaned) if cleaned else "unknown"  # Join with | or default
    if isinstance(value, str) and value.strip():
        return value.strip()  # Clean string
    return "unknown"


def _format_time_signature(value: object) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        num, den = value
        try:
            return f"{int(num)}/{int(den)}"  # Format as num/den
        except (TypeError, ValueError):
            return "unknown"
    if isinstance(value, str) and "/" in value:
        return value  # Already formatted
    return "unknown"


def _safe_stats_lookup(
    stats: object, key: str, default: float | None = np.nan
) -> float | None:
    if isinstance(stats, dict):
        return stats.get(key, default)  # Safe dict access
    return default


def _ensure_paths(dataset_root: Path, splits: Sequence[str]) -> DatasetPaths:
    processed_dir = dataset_root / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    missing = [
        split
        for split in splits
        if not (processed_dir / split / "metadata.json").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing metadata.json for splits: " + ", ".join(missing)
        )
    return DatasetPaths(
        dataset_root=dataset_root, processed_dir=processed_dir, splits=splits
    )


def load_bass_loops_metadata(
    dataset_root: Path, *, splits: Sequence[str] = ("train", "val", "test")
) -> pd.DataFrame:
    """Aggregate split metadata JSON into a single dataframe."""

    paths = _ensure_paths(dataset_root, splits)
    records: list[dict] = []
    for split in paths.splits:
        metadata_path = paths.processed_dir / split / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            try:
                split_records = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {metadata_path}") from exc

        if not isinstance(split_records, list):
            raise ValueError(f"Expected a list of records in {metadata_path}")

        for record in split_records:
            if not isinstance(record, dict):
                raise ValueError(f"Metadata entry is not a mapping in {metadata_path}")
            enriched = record.copy()
            enriched["original_split"] = split
            records.append(enriched)

    if not records:
        return pd.DataFrame(columns=["id", "tempo", "original_split", "midi_file_path"])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["id", "segment_idx"], keep="first").reset_index(
        drop=True
    )
    return df


def normalise_metadata(
    raw_df: pd.DataFrame,
    *,
    relative_root: str = DEFAULT_RELATIVE_ROOT,
    num_measures: int = DEFAULT_NUM_MEASURES,
    default_track_role: str = DEFAULT_TRACK_ROLE,
    default_inst: str = DEFAULT_INST,
) -> pd.DataFrame:
    """Return a dataframe whose schema mirrors the ComMu metadata."""

    if raw_df.empty:
        columns = list(COMM_BASS_COLUMNS) + [
            "original_split",
            "relative_midi_path",
            "source_file",
            "segment_idx",
            "n_notes",
            "tempo",
            "stats_pitch_entropy",
            "stats_unique_pitches",
            "stats_duration_ratio",
            "stats_max_consecutive_repeats",
            "stats_interval_variety",
            "stats_mean_interval",
        ]
        return pd.DataFrame(columns=columns)

    stats_series = raw_df.get("stats", pd.Series([{}] * len(raw_df)))

    normalized = pd.DataFrame()
    normalized["id"] = raw_df["id"].astype(str)
    normalized["bpm"] = raw_df.get("tempo", pd.Series([np.nan] * len(raw_df))).astype(
        float
    )
    normalized["genre"] = raw_df.get(
        "genre_hints", pd.Series(["unknown"] * len(raw_df))
    ).apply(_format_genre)
    normalized["time_signature"] = raw_df.get(
        "time_signature", pd.Series(["unknown"] * len(raw_df))
    ).apply(_format_time_signature)
    normalized["pitch_range"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "pitch_range")
    )
    normalized["min_velocity"] = pd.Series(pd.NA, index=normalized.index)
    normalized["max_velocity"] = pd.Series(pd.NA, index=normalized.index)

    normalized["audio_key"] = DEFAULT_AUDIO_KEY
    normalized["chord_progressions"] = DEFAULT_CHORD_PROGRESSIONS
    normalized["num_measures"] = num_measures
    normalized["track_role"] = default_track_role
    normalized["inst"] = default_inst
    normalized["sample_rhythm"] = DEFAULT_SAMPLE_RHYTHM

    normalized["split"] = "unassigned"
    normalized["split_data"] = "unassigned"
    normalized["file_path"] = ""

    normalized["Unnamed: 0"] = np.arange(len(normalized))

    normalized["original_split"] = raw_df.get(
        "original_split", pd.Series(["unknown"] * len(raw_df))
    ).astype(str)
    normalized["relative_midi_path"] = (
        raw_df.get("midi_file_path", pd.Series([""] * len(raw_df)))
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )
    normalized["source_file"] = raw_df.get(
        "source_file", pd.Series([""] * len(raw_df))
    ).astype(str)
    normalized["segment_idx"] = raw_df.get(
        "segment_idx", pd.Series([np.nan] * len(raw_df))
    )
    normalized["n_notes"] = raw_df.get("n_notes", pd.Series([np.nan] * len(raw_df)))
    normalized["tempo"] = normalized["bpm"]

    normalized["stats_pitch_entropy"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "pitch_entropy")
    )
    normalized["stats_unique_pitches"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "unique_pitches")
    )
    normalized["stats_duration_ratio"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "duration_ratio")
    )
    normalized["stats_max_consecutive_repeats"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "max_consecutive_repeats")
    )
    normalized["stats_interval_variety"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "interval_variety")
    )
    normalized["stats_mean_interval"] = stats_series.apply(
        lambda stats: _safe_stats_lookup(stats, "mean_interval")
    )

    ordered_columns = list(COMM_BASS_COLUMNS) + [
        "original_split",
        "relative_midi_path",
        "source_file",
        "segment_idx",
        "n_notes",
        "tempo",
        "stats_pitch_entropy",
        "stats_unique_pitches",
        "stats_duration_ratio",
        "stats_max_consecutive_repeats",
        "stats_interval_variety",
        "stats_mean_interval",
    ]

    normalized = normalized[ordered_columns]
    return normalized


def assign_deterministic_splits(
    df: pd.DataFrame,
    *,
    split_config: Optional[SplitConfig] = None,
    seed: int = 42,
    relative_root: str = DEFAULT_RELATIVE_ROOT,
    output_raw_root: Optional[Path] = None,
    common_root: Optional[Path] = None,
) -> pd.DataFrame:
    """Assign train/val/test splits deterministically and update file paths.

    Parameters
    ----------
    df:
        Input dataframe with bass loops metadata.
    split_config:
        Train/val/test split ratios.
    seed:
        Random seed for deterministic splits.
    relative_root:
        Subdirectory structure within the dataset (e.g., 'bass_loops_midi').
    output_raw_root:
        Directory where files will be copied.
    common_root:
        Optional common base directory. When provided along with output_raw_root,
        file_path will be stored relative to common_root instead of output_raw_root.
    """

    if split_config is None:
        split_config = SplitConfig()
    split_config.validate()

    if df.empty:
        return df.copy()

    working = df.copy()

    indices = pd.Series(np.arange(len(working)), name="orig_index")
    ordered = (
        pd.DataFrame({"id": working["id"].astype(str), "orig_index": indices})
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(seed)
    ordered["shuffle_rank"] = rng.permutation(len(ordered))
    ordered = ordered.sort_values("shuffle_rank", kind="mergesort").reset_index(
        drop=True
    )

    train_cutoff = int(len(ordered) * split_config.train_ratio)
    val_cutoff = train_cutoff + int(len(ordered) * split_config.val_ratio)

    splits = np.array(["test"] * len(ordered), dtype=object)
    splits[:train_cutoff] = "train"
    splits[train_cutoff:val_cutoff] = "val"

    ordered["split"] = splits

    working.loc[ordered["orig_index"], "split"] = ordered["split"].to_numpy()
    working["split_data"] = working["split"]

    # Construct file paths, optionally relative to common_root
    if common_root is not None and output_raw_root is not None:
        try:
            dataset_prefix = output_raw_root.relative_to(common_root)
            working["file_path"] = working.apply(
                lambda row: str(
                    dataset_prefix
                    / f"{relative_root}/{row['split']}/raw/{row['id']}.mid"
                ),
                axis=1,
            )
        except ValueError:
            LOGGER.warning(
                "output_raw_root (%s) is not under common_root (%s). "
                "Falling back to paths relative to output_raw_root.",
                output_raw_root,
                common_root,
            )
            working["file_path"] = working.apply(
                lambda row: f"{relative_root}/{row['split']}/raw/{row['id']}.mid",
                axis=1,
            )
    else:
        working["file_path"] = working.apply(
            lambda row: f"{relative_root}/{row['split']}/raw/{row['id']}.mid",
            axis=1,
        )

    working["Unnamed: 0"] = np.arange(len(working))

    return working


def copy_bass_loops_files(
    df: pd.DataFrame,
    *,
    dataset_root: Path,
    output_raw_root: Path,
    overwrite: bool = False,
    num_workers: int = 1,
) -> list[Path]:
    """Copy MIDI files referenced in ``df`` to ``output_raw_root``."""

    if df.empty:
        return []

    copied: list[Path] = []
    midi_source_base = dataset_root / "processed"
    output_raw_root.mkdir(parents=True, exist_ok=True)

    def _copy_single(row):
        relative_split = str(row.original_split)
        relative_midi_path = Path(str(row.relative_midi_path))
        source_path = (midi_source_base / relative_split / relative_midi_path).resolve()
        target_path = (output_raw_root / Path(str(row.file_path))).resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing MIDI file referenced in metadata: {source_path}"
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not overwrite:
            LOGGER.debug("Skipping existing file %s", target_path)
            return target_path
        else:
            shutil.copy2(source_path, target_path)
            return target_path

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            copied = list(executor.map(_copy_single, df.itertuples(index=False)))
    else:
        copied = [_copy_single(row) for row in df.itertuples(index=False)]

    return copied


def select_commu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe restricted to the canonical ComMu schema."""

    missing = [column for column in COMM_BASS_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Dataframe missing required columns: {missing}")
    columns = list(COMM_BASS_COLUMNS)
    return df[columns]


def curate_bass_loops(
    *,
    dataset_root: Path,
    output_raw_root: Path,
    output_processed_root: Path,
    split_config: Optional[SplitConfig] = None,
    seed: int = 42,
    relative_root: str = DEFAULT_RELATIVE_ROOT,
    overwrite: bool = False,
    num_workers: int = 1,
    common_root: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High-level helper used by the CLI to prepare Bass Loops artefacts.

    Parameters
    ----------
    dataset_root:
        Root directory containing the bass_loops dataset.
    output_raw_root:
        Directory where subset files will be copied.
    output_processed_root:
        Directory for cleaned metadata output.
    split_config:
        Train/val/test split ratios.
    seed:
        Random seed for deterministic splits.
    relative_root:
        Subdirectory structure within the dataset.
    overwrite:
        Whether to overwrite existing files.
    num_workers:
        Number of parallel workers for file copying.
    common_root:
        Optional common base directory for relative paths. When provided,
        file_path entries will be stored relative to this directory.
    """

    LOGGER.info("Loading Bass Loops metadata from %s", dataset_root)
    raw_df = load_bass_loops_metadata(dataset_root)

    LOGGER.info("Normalising schema for %s rows", len(raw_df))
    normalized = normalise_metadata(raw_df, relative_root=relative_root)

    LOGGER.info("Assigning deterministic splits (seed=%s)", seed)
    assigned = assign_deterministic_splits(
        normalized,
        split_config=split_config,
        seed=seed,
        relative_root=relative_root,
        output_raw_root=output_raw_root,
        common_root=common_root,
    )

    LOGGER.info("Copying MIDI files into %s", output_raw_root)
    copy_bass_loops_files(
        assigned,
        dataset_root=dataset_root,
        output_raw_root=output_raw_root,
        overwrite=overwrite,
        num_workers=num_workers,
    )

    output_raw_root.mkdir(parents=True, exist_ok=True)
    output_processed_root.mkdir(parents=True, exist_ok=True)

    raw_metadata_path = output_raw_root / "metadata.csv"
    clean_metadata_path = output_processed_root / "metadata_clean.csv"

    assigned.to_csv(raw_metadata_path, index=False)
    select_commu_columns(assigned).to_csv(clean_metadata_path, index=False)

    LOGGER.info("Wrote raw metadata to %s", raw_metadata_path)
    LOGGER.info("Wrote clean metadata to %s", clean_metadata_path)

    return normalized, assigned


__all__ = [
    "COMM_BASS_COLUMNS",
    "DEFAULT_RELATIVE_ROOT",
    "assign_deterministic_splits",
    "copy_bass_loops_files",
    "curate_bass_loops",
    "load_bass_loops_metadata",
    "normalise_metadata",
    "select_commu_columns",
]
