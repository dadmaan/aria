"""Artifact export helpers for LSTM pretraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from jsonschema import Draft7Validator, ValidationError

from .types import ArtifactBundle, ArtifactFile, DatasetMetadata, Manifest

CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["backend", "sequence_length"],
    "properties": {
        "backend": {"type": "string"},
        "sequence_length": {"type": "integer", "minimum": 1},
        "cluster_column": {"type": ["string", "null"]},
        "group_column": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

METRICS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "anyOf": [
            {"type": "number"},
            {"type": "string"},
            {"type": "boolean"},
            {"type": "array"},
            {"type": "object"},
            {"type": "null"},
        ]
    },
}

MANIFEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["backend", "run_id", "artifacts"],
    "properties": {
        "backend": {"type": "string"},
        "run_id": {"type": "string"},
        "artifacts": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["path", "sha256"],
                "properties": {
                    "path": {"type": "string"},
                    "sha256": {"type": "string", "minLength": 64, "maxLength": 64},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

_JSON_VALIDATORS = {
    "config": Draft7Validator(CONFIG_SCHEMA),
    "metrics": Draft7Validator(METRICS_SCHEMA),
    "manifest": Draft7Validator(MANIFEST_SCHEMA),
}


def _validate_json(kind: str, payload: Mapping[str, Any]) -> None:
    try:
        _JSON_VALIDATORS[kind].validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid {kind} payload: {exc.message}") from exc


def _write_json(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    _validate_json(kind, payload)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(backend: str, run_id: str, files: Dict[str, Path]) -> Manifest:
    artifact_entries: Dict[str, ArtifactFile] = {}
    for name, file_path in files.items():
        artifact_entries[name] = ArtifactFile(
            name=name,
            path=file_path,
            sha256=_compute_sha256(file_path),
        )
    return Manifest(backend=backend, run_id=run_id, artifacts=artifact_entries)


def _serialize_manifest(manifest: Manifest) -> Dict[str, Any]:
    return {
        "backend": manifest.backend,
        "run_id": manifest.run_id,
        "artifacts": {
            name: {"path": str(entry.path), "sha256": entry.sha256}
            for name, entry in manifest.artifacts.items()
        },
    }


def export_artifact_bundle(
    *,
    backend: str,
    run_id: str,
    output_root: Path,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    model_filename: str,
    model_exporter: Callable[[Path], None],
    vocab: Optional[Mapping[int, int]] = None,
    dataset_metadata: Optional[DatasetMetadata] = None,
    overwrite: bool = False,
) -> ArtifactBundle:
    """Persist a unified set of artifacts for a pretraining run."""

    run_dir = output_root / run_id
    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory {run_dir} already exists. Pass overwrite=True to replace it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = dict(config)
    config_payload["backend"] = backend
    config_path = run_dir / "config.json"
    _write_json(config_path, config_payload, "config")

    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, dict(metrics), "metrics")

    vocab_path: Optional[Path] = None
    if vocab is not None:
        vocab_path = run_dir / "token_mappings.json"
        idx_to_token = {str(idx): token for token, idx in vocab.items()}
        vocab_payload = {
            "token_to_idx": {str(token): idx for token, idx in vocab.items()},
            "idx_to_token": idx_to_token,
        }
        with vocab_path.open("w", encoding="utf-8") as fp:
            json.dump(vocab_payload, fp, indent=2, sort_keys=True)

    dataset_metadata_path: Optional[Path] = None
    if dataset_metadata is not None:
        dataset_metadata_path = run_dir / "dataset_metadata.json"
        with dataset_metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(dataset_metadata, fp, indent=2, sort_keys=True)

    model_path = run_dir / model_filename
    model_exporter(model_path)

    manifest_files: Dict[str, Path] = {
        "config": config_path,
        "metrics": metrics_path,
        "model": model_path,
    }
    if vocab_path is not None:
        manifest_files["token_mappings"] = vocab_path
    if dataset_metadata_path is not None:
        manifest_files["dataset_metadata"] = dataset_metadata_path

    manifest = _build_manifest(backend, run_id, manifest_files)
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, _serialize_manifest(manifest), "manifest")

    return ArtifactBundle(
        run_dir=run_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        vocab_path=vocab_path,
        model_path=model_path,
        manifest_path=manifest_path,
        dataset_metadata_path=dataset_metadata_path,
    )
