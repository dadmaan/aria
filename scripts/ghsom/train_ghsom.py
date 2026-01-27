from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ghsom.training import GHSOMTrainingConfig, train_and_export
from src.utils.features.feature_loader import FeatureType


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single GHSOM model run and export artifacts.",
    )
    parser.add_argument(
        "--feature-path",
        dest="feature_path",
        type=Path,
        required=True,
        help="Path to the feature artifact (CSV/NumPy file or run directory).",
    )
	parser.add_argument(
		"--feature-type",
		dest="feature_type",
		type=str,
		default="tsne",
		choices=["raw", "tsne", "pca", "umap", "reduced"],
		help="Specify whether to load raw numeric features or reduced embeddings (tsne/pca/umap/reduced).",
	)
    parser.add_argument(
        "--metadata-columns",
        dest="metadata_columns",
        nargs="*",
        default=None,
        help="Optional metadata columns to retain when exporting assignments (e.g. track_id).",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=None,
        help="Optional JSON file containing GHSOM hyperparameters (overridden by CLI values).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=Path("artifacts/ghsom"),
        help="Directory where the artifacts should be stored (default: artifacts/ghsom).",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Custom identifier for this training run (default: timestamp-based).",
    )
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=-1,
        help="Number of workers to use when training the GHSOM model.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Allow overwriting existing run directory if it already exists.",
    )

    for field_name, field_kwargs in [
        ("t1", {"type": float, "help": "Quantization threshold for root map."}),
        ("t2", {"type": float, "help": "Quantization threshold for child maps."}),
        ("learning_rate", {"type": float, "help": "Initial learning rate."}),
        ("decay", {"type": float, "help": "Learning rate decay."}),
        ("gaussian_sigma", {"type": float, "help": "Initial neighborhood width."}),
        ("epochs", {"type": int, "help": "Number of epochs for each training phase."}),
        ("grow_maxiter", {"type": int, "help": "Maximum grow iterations per level."}),
        ("seed", {"type": int, "help": "Random seed for reproducibility."}),
    ]:
        parser.add_argument(
            f"--{field_name.replace('_', '-')}",
            dest=field_name,
            default=None,
            **field_kwargs,
        )

    return parser.parse_args()


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if not config_path:
        return {}
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for field_name in GHSOMTrainingConfig.__dataclass_fields__.keys():
        value = getattr(args, field_name, None)
        if value is not None:
            overrides[field_name] = value
    return overrides


def _default_run_id() -> str:
    return datetime.utcnow().strftime("ghsom_%Y%m%d_%H%M%S")


def main() -> None:
    args = _parse_args()

    config_payload = _load_config(args.config_path)
    overrides = _collect_overrides(args)
    training_config = GHSOMTrainingConfig.from_mapping(
        config_payload, overrides=overrides
    )

    feature_path = args.feature_path.resolve()
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature artifact not found at {feature_path}")

    output_dir = args.output_dir.resolve()
    run_id = args.run_id or _default_run_id()
    feature_type = cast(FeatureType, args.feature_type)

    result = train_and_export(
        feature_path=feature_path,
        feature_type=feature_type,
        metadata_columns=cast(Optional[Iterable[str]], args.metadata_columns),
        config=training_config,
        output_dir=output_dir,
        run_id=run_id,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
    )

    artifacts = result["artifacts"]
    metrics = result["metrics"]

    print(f"GHSOM training run '{run_id}' completed.")
    print(f"Artifacts saved to: {artifacts['run_dir']}")
    print("Metrics:")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
