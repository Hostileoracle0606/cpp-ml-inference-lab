"""Deep semantic audit for a trusted local v1.1 evidence bundle.

The lightweight manifest verifier never deserializes model files. This opt-in
audit runs only in the pinned reference ML environment: it safely loads the
weights-only checkpoint, compares its internal provenance to the JSON records,
checks the ONNX graph, and recomputes PyTorch/ONNX Runtime parity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import onnx

import evidence_manifest
from model import CHECKPOINT_FORMAT_VERSION, load_checkpoint
import verify_onnx


def compare_checkpoint_metadata(
    manifest: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    """Require the checkpoint's internal protocol records to match the bundle."""

    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("deep evidence audit requires a schema-v2 checkpoint")

    training = manifest.get("training")
    environment = manifest.get("environment")
    split = manifest.get("split")
    if not all(isinstance(value, Mapping) for value in (training, environment, split)):
        raise ValueError("deep evidence audit requires manifest protocol mappings")

    if checkpoint.get("training") != training.get("config"):
        raise ValueError("checkpoint training config differs from the evidence record")
    if checkpoint.get("metrics") != {
        "validation_accuracy": training.get("validation_accuracy"),
        "test_accuracy": training.get("test_accuracy"),
    }:
        raise ValueError("checkpoint metrics differ from the evidence record")
    if checkpoint.get("duration_seconds") != training.get("duration_seconds"):
        raise ValueError("checkpoint duration differs from the evidence record")
    if checkpoint.get("epoch_history") != training.get("epoch_history"):
        raise ValueError("checkpoint epoch history differs from the evidence record")
    if checkpoint.get("environment") != dict(environment):
        raise ValueError("checkpoint environment differs from the evidence record")

    dataset = checkpoint.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("checkpoint has no dataset provenance mapping")
    expected_dataset_fields = {
        "name": split.get("dataset", {}).get("name"),
        "implementation": split.get("dataset", {}).get("implementation"),
        "archive": split.get("dataset", {}).get("archive"),
        "split_protocol": split.get("protocol"),
        "split_digest": split.get("digest"),
        "train_class_counts": split.get("train_class_counts"),
        "validation_class_counts": split.get("validation_class_counts"),
        "training_examples": 45000,
        "validation_examples": 5000,
        "test_examples": 10000,
    }
    for field, expected in expected_dataset_fields.items():
        if dataset.get(field) != expected:
            raise ValueError(f"checkpoint dataset.{field} differs from the evidence record")


def _artifact_path(
    root: Path, manifest: Mapping[str, Any], label: str
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts must be a mapping")
    record = artifacts.get(label)
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        raise ValueError(f"manifest has no artifact path for {label}")
    return root / record["path"]


def deep_verify_evidence_bundle(bundle: str | Path) -> dict[str, Any]:
    """Run integrity, checkpoint, graph, and recomputed-parity verification."""

    root = Path(bundle)
    manifest = evidence_manifest.verify_evidence_bundle(root)
    checkpoint_path = _artifact_path(root, manifest, "checkpoint")
    onnx_path = _artifact_path(root, manifest, "onnx")

    _, checkpoint = load_checkpoint(checkpoint_path)
    compare_checkpoint_metadata(manifest, checkpoint)

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    with tempfile.TemporaryDirectory(prefix="cpp-ml-v1.1-deep-audit-") as temporary:
        parity_path = Path(temporary) / "parity.json"
        passed = verify_onnx.verify(
            str(checkpoint_path),
            str(onnx_path),
            evidence_manifest.REFERENCE_PARITY_TOLERANCE,
            seed=1337,
            batch_size=2,
            json_out=parity_path,
        )
        recomputed = json.loads(parity_path.read_text(encoding="utf-8"))
    recorded = manifest["parity"]
    if not passed or any(
        recomputed[field] != recorded[field]
        for field in ("seed", "batch_size", "tolerance", "class_match", "passed")
    ):
        raise ValueError("recomputed parity contract differs from the evidence record")
    if not math.isclose(
        recomputed["max_abs_diff"],
        recorded["max_abs_diff"],
        rel_tol=1.0e-6,
        abs_tol=1.0e-12,
    ):
        raise ValueError("recomputed parity difference differs from the evidence record")

    return {
        "checkpoint_sha256": manifest["artifacts"]["checkpoint"]["sha256"],
        "onnx_sha256": manifest["artifacts"]["onnx"]["sha256"],
        "selected_epoch": manifest["training"]["selected_epoch"],
        "test_accuracy": manifest["training"]["test_accuracy"],
        "parity": recomputed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep-audit a trusted local v1.1 evidence bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    report = deep_verify_evidence_bundle(args.bundle)
    print(
        "[deep-audit] PASS "
        f"selected_epoch={report['selected_epoch']} "
        f"test_accuracy={report['test_accuracy']:.4f} "
        f"max_abs_diff={report['parity']['max_abs_diff']:.8g}"
    )


if __name__ == "__main__":
    main()
