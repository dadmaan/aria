from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dimensionality_reduction import (
	TSNEPreprocessor,
	TSNEPreprocessorConfig,
)

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run t-SNE or PCA on a numeric feature matrix and persist artifacts.",
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
		help="t-SNE perplexity (ignored for PCA).",
	)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=200.0,
		help="t-SNE learning rate (ignored for PCA).",
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
		choices=["tsne", "pca"],
		help="Projection method to use.",
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


def main() -> None:
	args = _parse_args()

	log_level = getattr(logging, args.log_level.upper(), logging.INFO)

	config = TSNEPreprocessorConfig(
		input_features=args.input_features.resolve(),
		output_root=args.output.resolve(),
		run_id=args.run_id,
		perplexity=args.perplexity,
		learning_rate=args.learning_rate,
		n_components=args.n_components,
		random_state=args.random_state,
		method=args.method,
		standardise=not args.no_standardise,
		metadata_columns=args.metadata_columns,
		seed=args.seed,
		overwrite=args.overwrite,
		log_level=log_level,
	)

	preprocessor = TSNEPreprocessor(config)
	result = preprocessor.execute()

	print(f"Run directory: {result.run_dir}")
	for name, path in result.artifacts.items():
		print(f"{name}: {path}")


if __name__ == "__main__":
	main()
