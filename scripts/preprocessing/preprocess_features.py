#!/usr/bin/env python3
"""
Preprocess feature data to handle missing values for GHSOM training.

This script:
1. Removes columns with 100% missing values
2. Imputes remaining missing values using median strategy
3. Saves the cleaned dataset
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_features(
    input_path: str,
    output_path: str,
    metadata_columns: list[str] = None,
    impute_strategy: str = "median",
    verbose: bool = True,
) -> dict:
    """
    Preprocess feature data by handling missing values.

    Args:
        input_path: Path to input CSV file with features
        output_path: Path to save cleaned CSV file
        metadata_columns: List of columns to preserve (not impute)
        impute_strategy: Strategy for imputation ('median', 'mean', 'most_frequent')
        verbose: Print progress messages

    Returns:
        Dictionary with preprocessing statistics
    """
    if metadata_columns is None:
        metadata_columns = ["track_id", "metadata_index"]

    # Load data
    if verbose:
        print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    original_shape = df.shape

    # Separate metadata and feature columns
    metadata_df = df[metadata_columns] if metadata_columns else None
    feature_cols = [col for col in df.columns if col not in metadata_columns]
    features_df = df[feature_cols]

    # Analyze missing values
    nan_summary = features_df.isna().sum()
    nan_pct = 100 * nan_summary / len(features_df)

    if verbose:
        print(f"\nOriginal shape: {original_shape}")
        print(f"Feature columns: {len(feature_cols)}")
        total_nan = features_df.isna().sum().sum()
        total_values = features_df.size
        print(f"Total NaN values: {total_nan} / {total_values} ({100*total_nan/total_values:.2f}%)")

    # 1. Remove columns with 100% missing values
    cols_to_drop = nan_summary[nan_pct == 100].index.tolist()
    if cols_to_drop:
        if verbose:
            print(f"\nRemoving {len(cols_to_drop)} columns with 100% NaN:")
            for col in cols_to_drop:
                print(f"  - {col}")
        features_df = features_df.drop(columns=cols_to_drop)
        feature_cols = [col for col in feature_cols if col not in cols_to_drop]

    # 2. Impute remaining missing values
    remaining_nan = features_df.isna().sum().sum()
    if remaining_nan > 0:
        if verbose:
            print(f"\nImputing {remaining_nan} remaining NaN values using '{impute_strategy}' strategy")
            cols_with_nan = features_df.isna().sum()
            cols_with_nan = cols_with_nan[cols_with_nan > 0]
            print(f"Columns with NaN values:")
            for col, count in cols_with_nan.items():
                pct = 100 * count / len(features_df)
                print(f"  - {col}: {count} ({pct:.2f}%)")

        imputer = SimpleImputer(strategy=impute_strategy)
        features_array = imputer.fit_transform(features_df)
        features_df = pd.DataFrame(
            features_array,
            columns=features_df.columns,
            index=features_df.index,
        )

    # 3. Combine with metadata and save
    if metadata_df is not None:
        cleaned_df = pd.concat([features_df, metadata_df], axis=1)
    else:
        cleaned_df = features_df

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(output_path, index=False)

    # Save preprocessing metadata
    metadata_path = output_path.parent / f"{output_path.stem}_preprocessing_metadata.json"
    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_shape": original_shape,
        "cleaned_shape": cleaned_df.shape,
        "columns_removed": cols_to_drop,
        "imputation_strategy": impute_strategy,
        "nan_before": int(total_nan),
        "nan_after": int(cleaned_df[feature_cols].isna().sum().sum()),
        "feature_columns": feature_cols,
        "metadata_columns": metadata_columns,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\nCleaned data saved to: {output_path}")
        print(f"Final shape: {cleaned_df.shape}")
        print(f"Metadata saved to: {metadata_path}")
        print("\nPreprocessing complete!")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess feature data for GHSOM training"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file with features",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save cleaned CSV file",
    )
    parser.add_argument(
        "--metadata-columns",
        nargs="+",
        default=["track_id", "metadata_index"],
        help="Columns to preserve as metadata (not impute)",
    )
    parser.add_argument(
        "--impute-strategy",
        choices=["median", "mean", "most_frequent"],
        default="median",
        help="Strategy for imputing missing values",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    preprocess_features(
        input_path=args.input,
        output_path=args.output,
        metadata_columns=args.metadata_columns,
        impute_strategy=args.impute_strategy,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
