from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dimensionality_reduction import (
    DimensionalityReductionConfig,
    DimensionalityReductionPreprocessor,
)

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _coerce_method_param_value(raw: str):
    """Parse a method parameter value from the CLI."""

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_method_params(pairs: Iterable[str]) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid --method-param '{pair}'. Expected format KEY=VALUE."
            )
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(
                "Method parameter keys must be non-empty (format KEY=VALUE)."
            )
        params[key] = _coerce_method_param_value(raw_value.strip())
    return params


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dimensionality reduction pipeline and persist artifacts.",
    )
    parser.add_argument(
        "--input-features",
        type=Path,
        required=True,
        help="Path to the feature CSV produced by the extraction pipeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/features/tsne"),
        help="Directory where the run folder will be created.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional identifier for the run (defaults to timestamp).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (legacy convenience flag; forwarded as method param).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=200.0,
        help="t-SNE learning rate (legacy convenience flag; forwarded as method param).",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of dimensions for the projection.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for the projection algorithm.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        help="Projection method to use (e.g., tsne, pca, umap).",
    )
    parser.add_argument(
        "--method-param",
        type=str,
        action="append",
        default=None,
        help=(
            "Additional reducer-specific parameters in KEY=VALUE format (repeatable). "
            "Examples: --method-param n_neighbors=15 --method-param min_dist=0.1 (UMAP); "
            "--method-param svd_solver=full (PCA); --method-param early_exaggeration=12.0 (t-SNE)."
        ),
    )
    parser.add_argument(
        "--metadata-columns",
        type=str,
        nargs="*",
        default=["track_id", "metadata_index"],
        help="Columns from the feature matrix to preserve in the projection output.",
    )
    parser.add_argument(
        "--no-standardise",
        action="store_true",
        help="Disable z-score standardisation before projection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the run directory if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity for the projection run.",
    )
    return parser.parse_args()


def _merge_method_params(
    method: str,
    legacy_perplexity: float,
    legacy_learning_rate: float,
    manual_params: Dict[str, object],
) -> Dict[str, object]:
    params = dict(manual_params)
    method_lower = method.lower()
    if method_lower == "tsne":
        params.setdefault("perplexity", legacy_perplexity)
        params.setdefault("learning_rate", legacy_learning_rate)
    return params


def main() -> None:
    args = _parse_args()
    try:
        manual_params = _parse_method_params(args.method_param or [])
    except argparse.ArgumentTypeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    method_params = _merge_method_params(
        args.method,
        legacy_perplexity=args.perplexity,
        legacy_learning_rate=args.learning_rate,
        manual_params=manual_params,
    )

    config = DimensionalityReductionConfig(
        input_features=args.input_features.resolve(),
        output_root=args.output.resolve(),
        run_id=args.run_id,
        method=args.method.lower(),
        n_components=args.n_components,
        random_state=args.random_state,
        standardise=not args.no_standardise,
        metadata_columns=[str(col) for col in args.metadata_columns],
        seed=args.seed,
        overwrite=args.overwrite,
        log_level=log_level,
        method_params=method_params,
    )

    preprocessor = DimensionalityReductionPreprocessor(config)
    result = preprocessor.execute()

    print(f"Run directory: {result.run_dir}")
    for name, path in result.artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
