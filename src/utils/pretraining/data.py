"""Shared data preparation helpers for LSTM pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .types import DatasetMetadata


def load_cluster_dataframe(csv_path: Path, cluster_column: str) -> pd.DataFrame:
    """Load the CSV file containing cluster assignments."""

    df = pd.read_csv(csv_path)
    if cluster_column not in df.columns:
        raise ValueError(f"Column '{cluster_column}' is required in {csv_path}.")
    if "sample_index" in df.columns:
        df = df.sort_values("sample_index")
    return df


def group_cluster_sequences(
    df: pd.DataFrame,
    cluster_column: str,
    group_column: Optional[str] = None,
) -> List[List[int]]:
    """Convert a dataframe into grouped cluster token sequences."""

    if group_column and group_column in df.columns:
        grouped = df.groupby(group_column)[cluster_column]
        sequences = [group.dropna().astype(int).tolist() for _, group in grouped]
    else:
        sequences = df[cluster_column].dropna().astype(int).tolist()
        sequences = [sequences]
    sequences = [seq for seq in sequences if len(seq) > 0]
    if not sequences:
        raise ValueError("No cluster sequences were found in the dataset.")
    return sequences


def build_vocab(sequences: Iterable[Sequence[int]]) -> Dict[int, int]:
    """Create a dense vocabulary mapping from raw cluster tokens."""

    unique_tokens = sorted({int(token) for seq in sequences for token in seq})
    if not unique_tokens:
        raise ValueError("Cluster sequences are empty; cannot build vocabulary.")
    return {token: idx for idx, token in enumerate(unique_tokens)}


def encode_sequences(
    sequences: Iterable[Sequence[int]],
    vocab: Mapping[int, int],
) -> List[List[int]]:
    """Encode cluster sequences using the provided vocabulary mapping."""

    return [[vocab[int(token)] for token in seq] for seq in sequences]


def build_windows(
    sequences: Iterable[Sequence[int]],
    sequence_length: int,
) -> Tuple[List[List[int]], List[int]]:
    """Construct fixed-length windows and next-token targets from sequences."""

    windows: List[List[int]] = []
    targets: List[int] = []
    for seq in sequences:
        if len(seq) <= sequence_length:
            continue
        for start in range(len(seq) - sequence_length):
            window = list(seq[start : start + sequence_length])
            target = int(seq[start + sequence_length])
            windows.append(window)
            targets.append(target)
    if not windows:
        raise ValueError(
            "Unable to construct training windows. Increase data size or reduce sequence_length."
        )
    return windows, targets


def build_dataset_metadata(sequences: Sequence[Sequence[int]]) -> DatasetMetadata:
    """Gather summary statistics about the prepared sequences."""

    lengths = [len(seq) for seq in sequences]
    original_tokens = sorted({int(token) for seq in sequences for token in seq})
    return {
        "num_sequences": len(sequences),
        "sequence_lengths": lengths,
        "original_tokens": original_tokens,
    }


def flatten_sequences(sequences: Sequence[Sequence[int]]) -> List[int]:
    """Concatenate multiple sequences into a single list."""

    flattened: List[int] = []
    for seq in sequences:
        flattened.extend(seq)
    return flattened


def ensure_seed(seed: Optional[int]) -> None:
    """Seed numpy's global RNG for reproducibility."""

    if seed is None:
        return
    np.random.seed(seed)
