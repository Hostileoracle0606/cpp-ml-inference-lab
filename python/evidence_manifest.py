"""Create a deterministic, self-excluding evidence manifest.

The manifest hashes immutable evidence artifacts and copies structured parity and
benchmark fields from JSON records. It intentionally contains no timestamp and
never hashes itself, so identical inputs produce identical manifest bytes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping


REFERENCE_ACCURACY_FLOOR = 0.65
REFERENCE_SPLIT_PROTOCOL = "cifar10-stratified-4500-train-500-validation-v1"
CIFAR10_ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_ARCHIVE_SIZE = 170498071
REFERENCE_PARITY_TOLERANCE = 1.0e-4
REFERENCE_ENVIRONMENT_IDENTIFIER = (
    "darwin-arm64-python3.9.6-torch2.8.0-torchvision0.23.0-"
    "onnx1.19.1-onnxruntime1.19.2"
)
REFERENCE_PACKAGES = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "onnx": "1.19.1",
    "onnxruntime": "1.19.2",
    "numpy": "2.0.2",
    "Pillow": "11.3.0",
}
REFERENCE_TRAINING_CONFIG = {
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 1.0e-3,
    "training_seed": 1337,
    "split_seed": 1337,
    "validation_size": 5000,
    "num_workers": 0,
    "optimizer": "Adam",
    "device": "cpu",
}
PROVENANCE_CAPTURE_NOTES = {
    "executed": (
        "Verbatim command and argv captured at execution time; not inferred or "
        "reconstructed."
    ),
    "canonical": (
        "Canonical command and argv sealed before bundle finalization; execution is "
        "checked after sealing."
    ),
}
EXECUTED_PROVENANCE_COMMANDS = (
    "training",
    "export",
    "parity",
    "trained_cmake_configure",
    "trained_cmake_build",
    "trained_ctest",
    "cli_smoke",
    "benchmark",
)
CANONICAL_PROVENANCE_COMMANDS = ("deep_audit", "bundle_build", "bundle_verify")
REQUIRED_BUNDLE_ARTIFACTS = frozenset(
    {
        "checkpoint",
        "onnx",
        "training_log",
        "training_record",
        "parity_record",
        "benchmark_record",
        "environment_record",
        "provenance_record",
    }
)
JSON_RECORD_ARTIFACTS = {
    "training": "training_record",
    "parity": "parity_record",
    "benchmark": "benchmark_record",
    "environment": "environment_record",
    "provenance": "provenance_record",
}


@dataclass(frozen=True)
class EvidenceManifest:
    artifacts: dict[str, dict[str, Any]]
    split: dict[str, Any]
    training: dict[str, Any]
    parity: dict[str, Any]
    benchmark: dict[str, Any]
    environment: dict[str, Any]
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifacts": self.artifacts,
            "split": self.split,
            "training": self.training,
            "parity": self.parity,
            "benchmark": self.benchmark,
            "environment": self.environment,
            "provenance": self.provenance,
        }


def sha256_file(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"evidence artifact does not exist: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"evidence artifact does not exist: {artifact}")
    return {
        "file": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }


def _safe_relative_path(value: str) -> Path:
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or relative == Path(".")
    ):
        raise ValueError(f"unsafe evidence path: {value!r}")
    return relative


def _validate_split(split: Mapping[str, Any]) -> None:
    if split.get("dataset") != {
        "name": "CIFAR-10",
        "implementation": "torchvision.datasets.CIFAR10",
        "archive": {
            "url": CIFAR10_ARCHIVE_URL,
            "md5": CIFAR10_ARCHIVE_MD5,
            "size_bytes": CIFAR10_ARCHIVE_SIZE,
        },
    }:
        raise ValueError("evidence split dataset provenance does not match CIFAR-10")
    if split.get("protocol") != REFERENCE_SPLIT_PROTOCOL:
        raise ValueError("evidence split protocol is unsupported")
    train_indices = split.get("train_indices")
    validation_indices = split.get("validation_indices")
    if not isinstance(train_indices, list) or not isinstance(validation_indices, list):
        raise ValueError("evidence split indices must be lists")
    if len(train_indices) != 45000 or len(validation_indices) != 5000:
        raise ValueError("evidence split must contain 45,000/5,000 indices")
    if any(type(index) is not int or index < 0 for index in train_indices + validation_indices):
        raise ValueError("evidence split indices must be non-negative integers")
    if len(set(train_indices)) != len(train_indices) or len(
        set(validation_indices)
    ) != len(validation_indices):
        raise ValueError("evidence split contains duplicate indices")
    if set(train_indices).intersection(validation_indices):
        raise ValueError("evidence split indices overlap")
    if set(train_indices).union(validation_indices) != set(range(50000)):
        raise ValueError("evidence split does not partition all 50,000 examples")
    canonical = json.dumps(
        {"train": train_indices, "validation": validation_indices},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if split.get("digest") != hashlib.sha256(canonical).hexdigest():
        raise ValueError("evidence split digest does not match its indices")
    expected_train = {label: 4500 for label in _cifar10_labels()}
    expected_validation = {label: 500 for label in _cifar10_labels()}
    if split.get("train_class_counts") != expected_train or split.get(
        "validation_class_counts"
    ) != expected_validation:
        raise ValueError("evidence split class counts are not stratified")


def _cifar10_labels() -> tuple[str, ...]:
    return (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_training(training: Mapping[str, Any]) -> None:
    required_fields = {
        "selected_epoch",
        "validation_accuracy",
        "test_accuracy",
        "duration_seconds",
        "config",
        "epoch_history",
    }
    if set(training) != required_fields:
        raise ValueError("training evidence fields do not match the frozen schema")

    config = training.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("training evidence requires a config mapping")
    selected_epoch = training.get("selected_epoch")
    if type(selected_epoch) is not int or not 1 <= selected_epoch <= 20:
        raise ValueError("selected_epoch must be in the frozen 20-epoch run")
    expected_config = {**REFERENCE_TRAINING_CONFIG, "selected_epoch": selected_epoch}
    if dict(config) != expected_config:
        raise ValueError("training config does not match the frozen CPU reference run")

    duration = training.get("duration_seconds")
    if not _finite_number(duration) or duration <= 0.0:
        raise ValueError("training duration_seconds must be positive and finite")
    accuracy = training.get("test_accuracy")
    if not _finite_number(accuracy) or not (
        REFERENCE_ACCURACY_FLOOR <= accuracy <= 1.0
    ):
        raise ValueError(
            f"test_accuracy must meet the frozen {REFERENCE_ACCURACY_FLOOR:.2f} floor"
        )

    history = training.get("epoch_history")
    if not isinstance(history, list) or len(history) != 20:
        raise ValueError("training history must contain all 20 frozen epochs")
    validation_values = []
    for expected_epoch, record in enumerate(history, start=1):
        if not isinstance(record, Mapping) or set(record) != {
            "epoch",
            "training_loss",
            "validation_accuracy",
        }:
            raise ValueError("training history record does not match the frozen schema")
        if record.get("epoch") != expected_epoch:
            raise ValueError("training history epochs must be contiguous")
        loss = record.get("training_loss")
        validation_accuracy = record.get("validation_accuracy")
        if not _finite_number(loss) or loss < 0.0:
            raise ValueError("training history loss must be non-negative and finite")
        if not _finite_number(validation_accuracy) or not (
            0.0 <= validation_accuracy <= 1.0
        ):
            raise ValueError("training history validation accuracy is invalid")
        validation_values.append(validation_accuracy)

    selected_validation = training.get("validation_accuracy")
    if (
        not _finite_number(selected_validation)
        or selected_validation != validation_values[selected_epoch - 1]
    ):
        raise ValueError("selected validation accuracy does not match epoch history")
    first_best_epoch = validation_values.index(max(validation_values)) + 1
    if selected_epoch != first_best_epoch:
        raise ValueError("selected epoch is not the earliest validation-best epoch")


def _validate_parity(parity: Mapping[str, Any]) -> None:
    if set(parity) != {
        "seed",
        "batch_size",
        "max_abs_diff",
        "tolerance",
        "class_match",
        "passed",
    }:
        raise ValueError("parity evidence fields do not match the frozen schema")
    max_diff = parity.get("max_abs_diff")
    tolerance = parity.get("tolerance")
    if (
        parity.get("seed") != 1337
        or parity.get("batch_size") != 2
        or not _finite_number(max_diff)
        or not _finite_number(tolerance)
        or max_diff < 0.0
        or tolerance != REFERENCE_PARITY_TOLERANCE
        or max_diff >= REFERENCE_PARITY_TOLERANCE
        or parity.get("class_match") is not True
        or parity.get("passed") is not True
    ):
        raise ValueError("parity evidence does not satisfy the frozen gate")


def _validate_environment(environment: Mapping[str, Any]) -> None:
    required_fields = {
        "identifier",
        "python",
        "platform",
        "system",
        "machine",
        "python_executable",
        "packages",
        "deterministic_algorithms",
        "device",
    }
    if set(environment) != required_fields:
        raise ValueError("environment evidence fields do not match the frozen schema")
    if (
        environment.get("identifier") != REFERENCE_ENVIRONMENT_IDENTIFIER
        or environment.get("python") != "3.9.6"
        or environment.get("system") != "Darwin"
        or environment.get("machine") != "arm64"
        or not _nonempty_string(environment.get("platform"))
        or not environment["platform"].startswith("macOS-")
        or not _nonempty_string(environment.get("python_executable"))
        or not Path(environment["python_executable"]).is_absolute()
        or environment.get("deterministic_algorithms") is not True
        or environment.get("packages") != REFERENCE_PACKAGES
        or environment.get("device")
        != {"requested": "cpu", "resolved": "cpu", "name": "cpu"}
    ):
        raise ValueError("environment does not match the exact CPU reference profile")


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    if set(provenance) != {
        "schema_version",
        "reference_run_id",
        "working_directory",
        "capture_notes",
        "commands",
    }:
        raise ValueError("provenance fields do not match the frozen schema")
    if type(provenance.get("schema_version")) is not int or provenance.get(
        "schema_version"
    ) != 1:
        raise ValueError("provenance schema_version must be 1")
    reference_run_id = provenance.get("reference_run_id")
    if not isinstance(reference_run_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", reference_run_id
    ) is None:
        raise ValueError("provenance reference_run_id is invalid")
    working_directory = provenance.get("working_directory")
    if not _nonempty_string(working_directory):
        raise ValueError("provenance working_directory is required")
    working_path = Path(working_directory)
    if (
        not working_path.is_absolute()
        or ".." in working_path.parts
        or working_path != working_path.resolve()
    ):
        raise ValueError("provenance working_directory must be an absolute canonical path")
    if provenance.get("capture_notes") != PROVENANCE_CAPTURE_NOTES:
        raise ValueError(
            "provenance capture notes must distinguish verbatim execution from "
            "pre-registered canonical commands"
        )

    commands = provenance.get("commands")
    expected_commands = set(EXECUTED_PROVENANCE_COMMANDS) | set(
        CANONICAL_PROVENANCE_COMMANDS
    )
    if not isinstance(commands, Mapping) or set(commands) != expected_commands:
        raise ValueError("provenance commands do not match the frozen command set")
    for name in EXECUTED_PROVENANCE_COMMANDS + CANONICAL_PROVENANCE_COMMANDS:
        record = commands[name]
        if not isinstance(record, Mapping) or set(record) != {
            "status",
            "command",
            "argv",
        }:
            raise ValueError(f"provenance command record has invalid fields: {name}")
        expected_status = (
            "executed" if name in EXECUTED_PROVENANCE_COMMANDS else "canonical"
        )
        command = record.get("command")
        argv = record.get("argv")
        if record.get("status") != expected_status:
            raise ValueError(
                f"provenance command {name} must have status {expected_status}"
            )
        if not _nonempty_string(command) or "\0" in command:
            raise ValueError(f"provenance command string is invalid: {name}")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not _nonempty_string(value) or "\0" in value for value in argv)
        ):
            raise ValueError(f"provenance argv is invalid: {name}")


def _validate_benchmark(benchmark: Mapping[str, Any]) -> str:
    required_sections = {
        "artifact",
        "benchmark",
        "build",
        "machine",
        "prediction_smoke",
        "scope",
        "verification",
    }
    if not required_sections.issubset(benchmark):
        raise ValueError("benchmark evidence omits required sections")

    artifact = benchmark.get("artifact")
    if not isinstance(artifact, Mapping) or not _nonempty_string(
        artifact.get("onnx")
    ):
        raise ValueError("benchmark artifact must identify its ONNX model")
    onnx_sha256 = artifact.get("sha256")
    if not isinstance(onnx_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", onnx_sha256
    ) is None:
        raise ValueError("benchmark artifact has no valid ONNX SHA-256")

    run = benchmark.get("benchmark")
    if not isinstance(run, Mapping):
        raise ValueError("benchmark run metadata must be a mapping")
    if (
        not _nonempty_string(run.get("command"))
        or type(run.get("warmup")) is not int
        or run["warmup"] <= 0
        or type(run.get("iterations")) is not int
        or run["iterations"] <= 0
    ):
        raise ValueError("benchmark command/warmup/iterations are invalid")
    boundaries = run.get("boundaries")
    if not isinstance(boundaries, Mapping) or set(boundaries) != {
        "preprocessing",
        "runtime_only",
        "end_to_end",
    }:
        raise ValueError("benchmark must contain all three frozen boundaries")
    metric_fields = {
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "throughput_operations_per_second",
    }
    for name, metrics in boundaries.items():
        if not isinstance(metrics, Mapping) or set(metrics) != metric_fields:
            raise ValueError(f"benchmark boundary has invalid fields: {name}")
        if any(not _finite_number(metrics[field]) for field in metric_fields):
            raise ValueError(f"benchmark boundary has non-finite fields: {name}")
        if (
            metrics["mean_ms"] < 0.0
            or metrics["p50_ms"] < 0.0
            or metrics["p95_ms"] < metrics["p50_ms"]
            or metrics["throughput_operations_per_second"] <= 0.0
        ):
            raise ValueError(f"benchmark boundary has invalid measurements: {name}")

    build = benchmark.get("build")
    if not isinstance(build, Mapping) or (
        build.get("build_type") != "Release"
        or not _nonempty_string(build.get("compiler"))
        or build.get("cxx") != 201703
        or build.get("onnxruntime") != "1.19.2"
        or build.get("system") != "Darwin/arm64"
    ):
        raise ValueError("benchmark build does not match the reference release profile")
    machine = benchmark.get("machine")
    if not isinstance(machine, Mapping) or (
        not _nonempty_string(machine.get("chip"))
        or type(machine.get("memory_bytes")) is not int
        or machine["memory_bytes"] <= 0
        or not _nonempty_string(machine.get("operating_system"))
    ):
        raise ValueError("benchmark machine evidence is incomplete")
    verification = benchmark.get("verification")
    if not isinstance(verification, Mapping) or (
        not _nonempty_string(verification.get("ctest_command"))
        or type(verification.get("passed")) is not int
        or type(verification.get("total")) is not int
        or verification["total"] <= 0
        or verification["passed"] != verification["total"]
    ):
        raise ValueError("benchmark verification must record a complete passing CTest run")

    smoke = benchmark.get("prediction_smoke")
    if not isinstance(smoke, Mapping) or (
        not _nonempty_string(smoke.get("command"))
        or not _nonempty_string(smoke.get("image"))
        or not _nonempty_string(smoke.get("label"))
        or type(smoke.get("class_index")) is not int
        or not 0 <= smoke["class_index"] < 10
        or not _finite_number(smoke.get("confidence"))
        or not 0.0 <= smoke["confidence"] <= 1.0
        or any(
            not _finite_number(smoke.get(field)) or smoke[field] < 0.0
            for field in ("inference_ms", "preprocess_ms", "total_latency_ms")
        )
    ):
        raise ValueError("benchmark prediction smoke evidence is invalid")
    scope = benchmark.get("scope")
    if not isinstance(scope, Mapping) or (
        scope.get("file_decode_included") is not False
        or scope.get("session_construction_included") is not False
        or not _nonempty_string(scope.get("workload"))
    ):
        raise ValueError("benchmark scope does not match the in-memory reference boundary")
    return onnx_sha256


def _validate_results(
    training: Mapping[str, Any],
    parity: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str:
    _validate_training(training)
    _validate_parity(parity)
    _validate_environment(environment)
    return _validate_benchmark(benchmark)


def build_manifest(
    artifacts: Mapping[str, str | Path],
    *,
    parity: Mapping[str, Any] | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not artifacts:
        raise ValueError("at least one evidence artifact is required")
    if any(not name or not isinstance(name, str) for name in artifacts):
        raise ValueError("artifact labels must be non-empty strings")
    manifest = {
        "schema_version": 1,
        "artifacts": {
            name: _artifact_record(artifacts[name]) for name in sorted(artifacts)
        },
        "parity": dict(parity or {}),
        "benchmark": dict(benchmark or {}),
    }
    # Fail here rather than while writing if fields are not JSON-compatible.
    json.dumps(manifest, allow_nan=False, sort_keys=True)
    return manifest


def write_manifest(
    output: str | Path,
    artifacts: Mapping[str, str | Path],
    *,
    parity: Mapping[str, Any] | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(output)
    destination_resolved = destination.resolve()
    if any(Path(path).resolve() == destination_resolved for path in artifacts.values()):
        raise ValueError("manifest output cannot also be a hashed artifact")
    manifest = build_manifest(artifacts, parity=parity, benchmark=benchmark)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        manifest, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    destination.write_text(serialized, encoding="utf-8")
    return manifest


def build_evidence_bundle(
    destination: str | Path,
    artifacts: Mapping[str, str | Path],
    *,
    split: Mapping[str, Any],
    training: Mapping[str, Any],
    parity: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    environment: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy local evidence, finalize hashes, and atomically publish one bundle."""

    if set(artifacts) != REQUIRED_BUNDLE_ARTIFACTS:
        missing = sorted(REQUIRED_BUNDLE_ARTIFACTS - set(artifacts))
        extra = sorted(set(artifacts) - REQUIRED_BUNDLE_ARTIFACTS)
        raise ValueError(
            f"official evidence artifact labels differ; missing={missing}, extra={extra}"
        )
    _validate_split(split)
    benchmark_onnx_sha256 = _validate_results(
        training, parity, benchmark, environment
    )
    _validate_provenance(provenance)

    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"evidence bundle already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        copied = {}
        files_directory = staging / "files"
        files_directory.mkdir()
        for label in sorted(artifacts):
            if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label) is None:
                raise ValueError(f"invalid evidence artifact label: {label!r}")
            source = Path(artifacts[label])
            if not source.is_file():
                raise FileNotFoundError(f"evidence artifact does not exist: {source}")
            suffix = "".join(source.suffixes)
            relative = Path("files") / f"{label}{suffix}"
            target = staging / relative
            shutil.copyfile(source, target)
            copied[label] = {
                "path": relative.as_posix(),
                **_artifact_record(target),
            }
            copied[label]["file"] = relative.as_posix()

        manifest_records = {
            "training": dict(training),
            "parity": dict(parity),
            "benchmark": dict(benchmark),
            "environment": dict(environment),
            "provenance": dict(provenance),
        }
        for field, label in JSON_RECORD_ARTIFACTS.items():
            source_record = _load_json_object(staging / copied[label]["path"])
            if source_record != manifest_records[field]:
                raise ValueError(
                    f"{field} manifest fields do not match copied {label} JSON"
                )
        if benchmark_onnx_sha256 != copied["onnx"]["sha256"]:
            raise ValueError("benchmark ONNX SHA-256 does not match copied ONNX artifact")

        split_path = staging / "dataset_split.json"
        split_path.write_text(
            json.dumps(dict(split), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copied["dataset_split"] = {
            "path": "dataset_split.json",
            **_artifact_record(split_path),
        }
        copied["dataset_split"]["file"] = "dataset_split.json"

        manifest_value = EvidenceManifest(
            artifacts=copied,
            split={
                "dataset": split["dataset"],
                "protocol": split["protocol"],
                "digest": split["digest"],
                "train_class_counts": split["train_class_counts"],
                "validation_class_counts": split["validation_class_counts"],
            },
            training=manifest_records["training"],
            parity=manifest_records["parity"],
            benchmark=manifest_records["benchmark"],
            environment=manifest_records["environment"],
            provenance=manifest_records["provenance"],
        )
        manifest = manifest_value.as_dict()
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        hash_paths = sorted(
            path.relative_to(staging)
            for path in staging.rglob("*")
            if path.is_file()
        )
        sums = "".join(
            f"{sha256_file(staging / relative)}  {relative.as_posix()}\n"
            for relative in hash_paths
        )
        (staging / "SHA256SUMS").write_text(sums, encoding="ascii")
        if output.exists():
            raise FileExistsError(f"evidence bundle appeared during build: {output}")
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_evidence_bundle(bundle: str | Path) -> dict[str, Any]:
    """Fail closed on missing, extra, unsafe, symlinked, or modified bundle files."""

    root = Path(bundle)
    if not root.is_dir():
        raise FileNotFoundError(f"evidence bundle does not exist: {root}")
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file() or sums_path.is_symlink():
        raise ValueError("evidence bundle has no regular SHA256SUMS")
    expected_hashes = {}
    for line in sums_path.read_text(encoding="ascii").splitlines():
        if "  " not in line:
            raise ValueError("malformed SHA256SUMS line")
        digest, value = line.split("  ", 1)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("malformed SHA256SUMS digest")
        relative = _safe_relative_path(value)
        key = relative.as_posix()
        if key in expected_hashes:
            raise ValueError("duplicate SHA256SUMS path")
        expected_hashes[key] = digest

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != sums_path
    }
    if actual_files != set(expected_hashes):
        raise ValueError("evidence bundle has missing or extra files")
    root_resolved = root.resolve()
    for value, digest in expected_hashes.items():
        relative = _safe_relative_path(value)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"evidence path is not a regular file: {value}")
        if root_resolved not in path.resolve().parents:
            raise ValueError(f"evidence path escapes its bundle: {value}")
        if sha256_file(path) != digest:
            raise ValueError(f"evidence hash mismatch: {value}")

    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported evidence manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("evidence manifest artifacts must be a mapping")
    expected_artifact_labels = REQUIRED_BUNDLE_ARTIFACTS | {"dataset_split"}
    if set(artifacts) != expected_artifact_labels:
        raise ValueError("evidence manifest does not contain the official artifact labels")
    manifest_artifact_paths = set()
    for label, record in artifacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid evidence artifact record: {label}")
        value = record.get("path")
        if not isinstance(value, str):
            raise ValueError(f"evidence artifact has no path: {label}")
        relative = _safe_relative_path(value)
        if relative.as_posix() in manifest_artifact_paths:
            raise ValueError("evidence manifest aliases more than one label to one path")
        manifest_artifact_paths.add(relative.as_posix())
        path = root / relative
        if (
            value not in expected_hashes
            or record.get("file") != value
            or record.get("sha256") != sha256_file(path)
            or record.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"evidence artifact record mismatch: {label}")
    if manifest_artifact_paths | {"manifest.json"} != set(expected_hashes):
        raise ValueError("manifest does not enumerate the complete evidence bundle")

    split_record = _load_json_object(root / "dataset_split.json")
    _validate_split(split_record)
    manifest_split = manifest.get("split")
    expected_split_summary = {
        "dataset": split_record.get("dataset"),
        "protocol": split_record.get("protocol"),
        "digest": split_record.get("digest"),
        "train_class_counts": split_record.get("train_class_counts"),
        "validation_class_counts": split_record.get("validation_class_counts"),
    }
    if not isinstance(manifest_split, Mapping) or dict(
        manifest_split
    ) != expected_split_summary:
        raise ValueError("manifest split record does not match split artifact")
    manifest_records = {}
    for field, label in JSON_RECORD_ARTIFACTS.items():
        manifest_value = manifest.get(field)
        if not isinstance(manifest_value, Mapping):
            raise ValueError(f"manifest {field} evidence must be a mapping")
        record_path = root / artifacts[label]["path"]
        source_record = _load_json_object(record_path)
        if source_record != dict(manifest_value):
            raise ValueError(
                f"manifest {field} fields do not match copied {label} JSON"
            )
        manifest_records[field] = dict(manifest_value)
    benchmark_onnx_sha256 = _validate_results(
        manifest_records["training"],
        manifest_records["parity"],
        manifest_records["benchmark"],
        manifest_records["environment"],
    )
    _validate_provenance(manifest_records["provenance"])
    if benchmark_onnx_sha256 != artifacts["onnx"].get("sha256"):
        raise ValueError("benchmark ONNX SHA-256 does not match bundled ONNX artifact")
    return manifest


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"evidence JSON must contain an object: {source}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify a local v1.1 reference evidence bundle."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Atomically finalize a local bundle")
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--onnx", type=Path, required=True)
    build.add_argument("--training-log", type=Path, required=True)
    build.add_argument("--split-json", type=Path, required=True)
    build.add_argument("--training-json", type=Path, required=True)
    build.add_argument("--parity-json", type=Path, required=True)
    build.add_argument("--benchmark-json", type=Path, required=True)
    build.add_argument("--environment-json", type=Path, required=True)
    build.add_argument("--provenance-json", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify", help="Fail closed on bundle tampering")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "verify":
        manifest = verify_evidence_bundle(args.bundle)
        print(
            f"[manifest] verified {args.bundle} "
            f"with {len(manifest['artifacts'])} artifacts"
        )
        return

    artifacts = {
        "checkpoint": args.checkpoint,
        "onnx": args.onnx,
        "training_log": args.training_log,
        "training_record": args.training_json,
        "parity_record": args.parity_json,
        "benchmark_record": args.benchmark_json,
        "environment_record": args.environment_json,
        "provenance_record": args.provenance_json,
    }
    manifest = build_evidence_bundle(
        args.out,
        artifacts,
        split=_load_json_object(args.split_json),
        training=_load_json_object(args.training_json),
        parity=_load_json_object(args.parity_json),
        benchmark=_load_json_object(args.benchmark_json),
        environment=_load_json_object(args.environment_json),
        provenance=_load_json_object(args.provenance_json),
    )
    print(
        f"[manifest] finalized {args.out} "
        f"with {len(manifest['artifacts'])} artifacts"
    )


if __name__ == "__main__":
    main()
