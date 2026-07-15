"""Dependency-aware smoke tests for the Python training/export boundary.

The tests use synthetic tensors and a temporary checkpoint: they never download
CIFAR-10 and never write model artifacts into the repository.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
CPP_PREPROCESSOR_HEADER = (
    Path(__file__).resolve().parents[1] / "include" / "cpp_ml" / "preprocessor.hpp"
)
sys.path.insert(0, str(PYTHON_DIR))

import evidence_manifest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None
ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import train
    import model as model_contract
    if ONNX_AVAILABLE and ORT_AVAILABLE:
        import deep_verify_evidence


class PreprocessingContractDriftTests(unittest.TestCase):
    def test_cpp_and_python_preprocessing_contracts_match(self) -> None:
        """Guard the cross-language constants without importing ML dependencies."""

        python_source = (PYTHON_DIR / "model.py").read_text(encoding="utf-8")
        assignments = {}
        for node in ast.parse(python_source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {
                    "CIFAR10_MEAN",
                    "CIFAR10_STD",
                }:
                    assignments[target.id] = tuple(ast.literal_eval(node.value))
        self.assertEqual(set(assignments), {"CIFAR10_MEAN", "CIFAR10_STD"})

        cpp_source = CPP_PREPROCESSOR_HEADER.read_text(encoding="utf-8")

        def cpp_scalar(name: str) -> int:
            match = re.search(rf"\b{name}\s*=\s*(\d+)\s*;", cpp_source)
            self.assertIsNotNone(match, f"missing C++ preprocessing scalar {name}")
            return int(match.group(1))

        def cpp_float_array(name: str) -> tuple[float, ...]:
            match = re.search(
                rf"\b{name}\s*=\s*\{{([^}}]+)\}}\s*;", cpp_source, re.DOTALL
            )
            self.assertIsNotNone(match, f"missing C++ preprocessing array {name}")
            return tuple(
                float(token.strip().rstrip("Ff"))
                for token in match.group(1).split(",")
                if token.strip()
            )

        cpp_shape = (
            cpp_scalar("kInputChannels"),
            cpp_scalar("kInputHeight"),
            cpp_scalar("kInputWidth"),
        )
        self.assertEqual(cpp_shape, (3, 32, 32))
        self.assertIn("[N, 3, 32, 32]", python_source)
        self.assertEqual(cpp_float_array("kMean"), assignments["CIFAR10_MEAN"])
        self.assertEqual(cpp_float_array("kStd"), assignments["CIFAR10_STD"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TrainingSmokeTests(unittest.TestCase):
    def test_small_cnn_output_contract(self) -> None:
        model = train.SmallCNN().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, len(train.CIFAR10_CLASSES)))
        self.assertTrue(torch.isfinite(output).all().item())

    def test_train_one_epoch_updates_parameters_without_dataset_download(self) -> None:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
        features = torch.randn(8, 3, 4, 4)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        loss = train.train_one_epoch(model, loader, optimizer, torch.device("cpu"))

        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 0.0)
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )

    def test_evaluate_computes_exact_top1_accuracy(self) -> None:
        model = nn.Identity()
        logits = torch.tensor(
            [[4.0, 0.0], [0.0, 4.0], [3.0, 1.0], [1.0, 3.0]]
        )
        labels = torch.tensor([0, 1, 1, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        accuracy = train.evaluate(model, loader, torch.device("cpu"))
        self.assertAlmostEqual(accuracy, 0.75)
        self.assertFalse(model.training)

    @unittest.skipUnless(TORCHVISION_AVAILABLE, "torchvision is not installed")
    def test_dataloaders_use_the_shared_normalization_contract_without_download(self) -> None:
        import torchvision

        calls = []

        class SyntheticCifar10(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                calls.append(
                    {
                        "root": root,
                        "train": train,
                        "download": download,
                        "transform": transform,
                    }
                )

            def __len__(self):
                return 2

            def __getitem__(self, index):
                return torch.zeros(3, 32, 32), index % 2

        with mock.patch.object(torchvision.datasets, "CIFAR10", SyntheticCifar10):
            train_loader, test_loader = train.get_dataloaders(2, "/unused/data")

        self.assertEqual(train_loader.batch_size, 2)
        self.assertEqual(test_loader.batch_size, 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual([call["train"] for call in calls], [True, False])
        self.assertTrue(all(call["download"] for call in calls))
        for call in calls:
            transforms = call["transform"].transforms
            normalization = next(
                transform
                for transform in transforms
                if isinstance(transform, torchvision.transforms.Normalize)
            )
            self.assertEqual(tuple(normalization.mean), train.CIFAR10_MEAN)
            self.assertEqual(tuple(normalization.std), train.CIFAR10_STD)

    def test_empty_loader_is_rejected_instead_of_dividing_by_zero(self) -> None:
        model = nn.Linear(2, 2)
        empty = DataLoader(
            TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaises(ValueError):
            train.train_one_epoch(model, empty, optimizer, torch.device("cpu"))
        with self.assertRaises(ValueError):
            train.evaluate(model, empty, torch.device("cpu"))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ReferenceTrainingTests(unittest.TestCase):
    def test_frozen_split_is_deterministic_disjoint_and_stratified(self) -> None:
        targets = [class_index for class_index in range(10) for _ in range(5000)]
        first = train.stratified_reference_split(targets, split_seed=41)
        repeated = train.stratified_reference_split(targets, split_seed=41)
        different = train.stratified_reference_split(targets, split_seed=42)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        train_indices, validation_indices = first
        self.assertEqual(len(train_indices), 45000)
        self.assertEqual(len(validation_indices), 5000)
        self.assertTrue(set(train_indices).isdisjoint(validation_indices))
        self.assertEqual(
            set(train_indices) | set(validation_indices), set(range(50000))
        )
        self.assertEqual(
            set(train.class_counts(targets, train_indices).values()), {4500}
        )
        self.assertEqual(
            set(train.class_counts(targets, validation_indices).values()), {500}
        )
        digest = train.split_digest(train_indices, validation_indices)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, train.split_digest(*repeated))
        split = train.make_dataset_split(targets, split_seed=41)
        split.validate()
        self.assertEqual(split.train_indices, train_indices)
        self.assertEqual(split.validation_indices, validation_indices)
        self.assertEqual(split.digest, digest)

    @unittest.skipUnless(TORCHVISION_AVAILABLE, "torchvision is not installed")
    def test_reference_loaders_separate_training_and_validation_transforms(self) -> None:
        import torchvision

        calls = []

        class SyntheticCifar10(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.train = train
                self.transform = transform
                self.targets = (
                    [class_index for class_index in range(10) for _ in range(5000)]
                    if train
                    else [class_index for class_index in range(10) for _ in range(2)]
                )
                calls.append(self)

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, index):
                return torch.zeros(3, 32, 32), self.targets[index]

        with mock.patch.object(torchvision.datasets, "CIFAR10", SyntheticCifar10):
            data = train.get_reference_dataloaders(
                16, "/unused/data", split_seed=19, download=False
            )

        self.assertEqual(len(calls), 3)
        training_source = data.train_loader.dataset.dataset
        validation_source = data.validation_loader.dataset.dataset
        self.assertIsNot(training_source, validation_source)
        self.assertTrue(training_source.train)
        self.assertTrue(validation_source.train)
        self.assertFalse(data.test_loader.dataset.train)
        self.assertTrue(
            any(
                isinstance(transform, torchvision.transforms.RandomCrop)
                for transform in training_source.transform.transforms
            )
        )
        self.assertFalse(
            any(
                isinstance(transform, torchvision.transforms.RandomCrop)
                for transform in validation_source.transform.transforms
            )
        )
        self.assertEqual(set(data.train_class_counts.values()), {4500})
        self.assertEqual(set(data.validation_class_counts.values()), {500})

    def test_validation_best_state_is_restored_and_test_is_evaluated_once(self) -> None:
        selected_model = nn.Linear(1, 1, bias=False)
        train_loader = object()
        validation_loader = object()
        test_loader = object()
        epoch = 0
        evaluation_calls = []
        test_weight = []
        validation_scores = [0.4, 0.9, 0.9]

        def fake_train(model, loader, optimizer, device):
            nonlocal epoch
            self.assertIs(loader, train_loader)
            epoch += 1
            with torch.no_grad():
                model.weight.fill_(float(epoch))
            return float(epoch)

        def fake_evaluate(model, loader, device):
            evaluation_calls.append(loader)
            if loader is validation_loader:
                return validation_scores[len(evaluation_calls) - 1]
            self.assertIs(loader, test_loader)
            test_weight.append(float(model.weight.detach().item()))
            return 0.8

        with mock.patch.object(train, "train_one_epoch", side_effect=fake_train), mock.patch.object(
            train, "evaluate", side_effect=fake_evaluate
        ):
            selection = train.train_with_validation(
                selected_model,
                train_loader,
                validation_loader,
                optimizer=object(),
                device=torch.device("cpu"),
                epochs=3,
            )
            self.assertEqual(evaluation_calls, [validation_loader] * 3)
            result = train.evaluate_test_once(
                selected_model, test_loader, torch.device("cpu"), selection
            )

        self.assertEqual(result.selected_epoch, 2)
        self.assertEqual(result.validation_accuracy, 0.9)
        self.assertEqual(result.test_accuracy, 0.8)
        self.assertEqual(evaluation_calls, [validation_loader] * 3 + [test_loader])
        self.assertEqual(test_weight, [2.0])
        self.assertEqual(float(selected_model.weight.detach().item()), 2.0)

    def test_invalid_reference_configs_and_devices_are_rejected(self) -> None:
        invalid = [
            train.TrainingConfig(epochs=0),
            train.TrainingConfig(batch_size=0),
            train.TrainingConfig(learning_rate=float("nan")),
            train.TrainingConfig(validation_size=4999),
            train.TrainingConfig(num_workers=-1),
            train.TrainingConfig(training_seed=-1),
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        self.assertEqual(train.resolve_device("cpu"), torch.device("cpu"))
        with self.assertRaises(ValueError):
            train.resolve_device("tpu")
        with mock.patch.object(train, "mps_available", return_value=False):
            with self.assertRaises(ValueError):
                train.resolve_device("mps")

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            train.seed_everything(1337)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CheckpointContractTests(unittest.TestCase):
    def _v2_checkpoint(self):
        return {
            "format_version": 2,
            "model_state_dict": train.SmallCNN().state_dict(),
            "architecture": model_contract.MODEL_ARCHITECTURE,
            "classes": model_contract.CIFAR10_CLASSES,
            "normalization_mean": model_contract.CIFAR10_MEAN,
            "normalization_std": model_contract.CIFAR10_STD,
            "model_contract": model_contract.model_contract(),
            "dataset": {
                "name": model_contract.DATASET_NAME,
                "implementation": model_contract.DATASET_IMPLEMENTATION,
                "archive": {
                    "url": model_contract.CIFAR10_ARCHIVE_URL,
                    "md5": model_contract.CIFAR10_ARCHIVE_MD5,
                    "size_bytes": model_contract.CIFAR10_ARCHIVE_SIZE,
                },
                "training_examples": 45000,
                "validation_examples": 5000,
                "test_examples": 10000,
                "split_protocol": model_contract.REFERENCE_SPLIT_PROTOCOL,
                "split_digest": "0" * 64,
                "train_class_counts": {
                    label: 4500 for label in model_contract.CIFAR10_CLASSES
                },
                "validation_class_counts": {
                    label: 500 for label in model_contract.CIFAR10_CLASSES
                },
            },
            "training": {
                "epochs": 1,
                "batch_size": 128,
                "learning_rate": 1.0e-3,
                "training_seed": 1337,
                "split_seed": 1337,
                "validation_size": 5000,
                "num_workers": 0,
                "optimizer": "Adam",
                "device": "cpu",
                "selected_epoch": 1,
            },
            "metrics": {"validation_accuracy": 0.5, "test_accuracy": 0.4},
            "environment": {
                "identifier": "synthetic-test",
                "python": "3.9.6",
                "platform": "test",
                "system": "test",
                "machine": "test",
                "python_executable": "/python",
                "packages": {
                    name: "test"
                    for name in (
                        "torch",
                        "torchvision",
                        "onnx",
                        "onnxruntime",
                        "numpy",
                        "Pillow",
                    )
                },
                "device": {"resolved": "cpu"},
                "deterministic_algorithms": True,
            },
            "duration_seconds": 1.0,
            "epoch_history": [
                {"epoch": 1, "training_loss": 1.0, "validation_accuracy": 0.5}
            ],
        }

    def test_raw_v1_and_v2_checkpoints_remain_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            raw = directory / "raw.pt"
            v1 = directory / "v1.pt"
            v2 = directory / "v2.pt"
            state = train.SmallCNN().state_dict()
            torch.save(state, raw)
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": state,
                    "classes": model_contract.CIFAR10_CLASSES,
                    "normalization_mean": model_contract.CIFAR10_MEAN,
                    "normalization_std": model_contract.CIFAR10_STD,
                },
                v1,
            )
            torch.save(self._v2_checkpoint(), v2)

            self.assertIsNone(model_contract.load_checkpoint(raw)[1].get("format_version"))
            self.assertEqual(model_contract.load_checkpoint(v1)[1]["format_version"], 1)
            self.assertEqual(model_contract.load_checkpoint(v2)[1]["format_version"], 2)

    def test_build_checkpoint_records_selected_state_and_reference_evidence(self) -> None:
        train_indices = tuple(range(45000))
        validation_indices = tuple(range(45000, 50000))
        split = train.DatasetSplit(
            train_indices=train_indices,
            validation_indices=validation_indices,
            digest=train.split_digest(train_indices, validation_indices),
            train_class_counts={
                label: 4500 for label in model_contract.CIFAR10_CLASSES
            },
            validation_class_counts={
                label: 500 for label in model_contract.CIFAR10_CLASSES
            },
        )
        data = train.ReferenceDataLoaders(
            train_loader=object(),
            validation_loader=object(),
            test_loader=mock.Mock(dataset=range(10000)),
            split=split,
        )
        result = train.TrainingResult(
            selected_epoch=1,
            validation_accuracy=0.6,
            test_accuracy=0.55,
            duration_seconds=2.0,
            epoch_history=(train.EpochRecord(1, 1.2, 0.6),),
        )
        config = train.TrainingConfig(epochs=1, device="cpu")
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            checkpoint = train.build_checkpoint(
                train.SmallCNN(), config, data, result, "cpu", torch.device("cpu")
            )
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)

        self.assertEqual(checkpoint["format_version"], 2)
        self.assertEqual(checkpoint["training"]["selected_epoch"], 1)
        self.assertEqual(checkpoint["metrics"]["validation_accuracy"], 0.6)
        self.assertEqual(checkpoint["metrics"]["test_accuracy"], 0.55)
        self.assertEqual(
            checkpoint["dataset"]["split_protocol"],
            model_contract.REFERENCE_SPLIT_PROTOCOL,
        )
        self.assertEqual(checkpoint["dataset"]["split_digest"], data.split_digest)
        self.assertIn("packages", checkpoint["environment"])
        self.assertEqual(
            model_contract.validate_checkpoint_metadata(checkpoint), 2
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            record_dir = Path(temporary_directory) / "records"
            records = train.write_reference_records(record_dir, checkpoint, data)
            self.assertEqual(set(records), {"split", "training", "environment"})
            split_record = json.loads(records["split"].read_text(encoding="utf-8"))
            training_record = json.loads(
                records["training"].read_text(encoding="utf-8")
            )
            self.assertEqual(split_record["digest"], data.split.digest)
            self.assertEqual(
                split_record["dataset"]["archive"]["md5"],
                model_contract.CIFAR10_ARCHIVE_MD5,
            )
            self.assertEqual(training_record["test_accuracy"], 0.55)
            with self.assertRaises(FileExistsError):
                train.write_reference_records(record_dir, checkpoint, data)

    def test_metadata_contract_drift_is_rejected(self) -> None:
        mutations = []
        classes = self._v2_checkpoint()
        classes["classes"] = ("wrong",) + classes["classes"][1:]
        mutations.append(classes)
        normalization = self._v2_checkpoint()
        normalization["model_contract"]["normalization_mean"] = (0.0, 0.0, 0.0)
        mutations.append(normalization)
        architecture = self._v2_checkpoint()
        architecture["model_contract"]["architecture"] = "DifferentCNN"
        mutations.append(architecture)
        archive = self._v2_checkpoint()
        archive["dataset"]["archive"]["md5"] = "0" * 32
        mutations.append(archive)
        deterministic = self._v2_checkpoint()
        deterministic["environment"]["deterministic_algorithms"] = False
        mutations.append(deterministic)

        with tempfile.TemporaryDirectory() as temporary_directory:
            for index, payload in enumerate(mutations):
                path = Path(temporary_directory) / f"drift-{index}.pt"
                torch.save(payload, path)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    model_contract.load_checkpoint(path)

    def test_unknown_checkpoint_schema_fails_closed(self) -> None:
        payload = self._v2_checkpoint()
        payload["format_version"] = 99
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "future.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "format_version"):
                model_contract.load_checkpoint(path)

    @unittest.skipUnless(
        ONNX_AVAILABLE and ORT_AVAILABLE,
        "ONNX and ONNX Runtime are required for the deep evidence audit",
    )
    def test_deep_evidence_metadata_comparison_rejects_drift(self) -> None:
        checkpoint = self._v2_checkpoint()
        dataset = checkpoint["dataset"]
        manifest = {
            "training": {
                "config": checkpoint["training"],
                "selected_epoch": checkpoint["training"]["selected_epoch"],
                "validation_accuracy": checkpoint["metrics"]["validation_accuracy"],
                "test_accuracy": checkpoint["metrics"]["test_accuracy"],
                "duration_seconds": checkpoint["duration_seconds"],
                "epoch_history": checkpoint["epoch_history"],
            },
            "environment": checkpoint["environment"],
            "split": {
                "dataset": {
                    "name": dataset["name"],
                    "implementation": dataset["implementation"],
                    "archive": dataset["archive"],
                },
                "protocol": dataset["split_protocol"],
                "digest": dataset["split_digest"],
                "train_class_counts": dataset["train_class_counts"],
                "validation_class_counts": dataset["validation_class_counts"],
            },
        }
        deep_verify_evidence.compare_checkpoint_metadata(manifest, checkpoint)

        drifted = copy.deepcopy(checkpoint)
        drifted["metrics"]["test_accuracy"] = 0.99
        with self.assertRaisesRegex(ValueError, "metrics"):
            deep_verify_evidence.compare_checkpoint_metadata(manifest, drifted)


class EvidenceManifestTests(unittest.TestCase):
    @staticmethod
    def _reference_records(onnx_sha256: str):
        history = [
            {
                "epoch": epoch,
                "training_loss": 2.0 - epoch * 0.05,
                "validation_accuracy": 0.5 + epoch * 0.01,
            }
            for epoch in range(1, 21)
        ]
        training = {
            "selected_epoch": 20,
            "validation_accuracy": history[-1]["validation_accuracy"],
            "test_accuracy": 0.7,
            "duration_seconds": 10.0,
            "config": {
                **evidence_manifest.REFERENCE_TRAINING_CONFIG,
                "selected_epoch": 20,
            },
            "epoch_history": history,
        }
        parity = {
            "seed": 1337,
            "batch_size": 2,
            "max_abs_diff": 1.0e-8,
            "tolerance": evidence_manifest.REFERENCE_PARITY_TOLERANCE,
            "class_match": True,
            "passed": True,
        }
        environment = {
            "identifier": evidence_manifest.REFERENCE_ENVIRONMENT_IDENTIFIER,
            "python": "3.9.6",
            "platform": "macOS-test-arm64",
            "system": "Darwin",
            "machine": "arm64",
            "python_executable": "/tmp/reference/bin/python",
            "packages": dict(evidence_manifest.REFERENCE_PACKAGES),
            "deterministic_algorithms": True,
            "device": {"requested": "cpu", "resolved": "cpu", "name": "cpu"},
        }
        boundary = {
            "mean_ms": 1.0,
            "p50_ms": 0.9,
            "p95_ms": 1.2,
            "throughput_operations_per_second": 1000.0,
        }
        benchmark = {
            "artifact": {"onnx": "models/reference.onnx", "sha256": onnx_sha256},
            "benchmark": {
                "boundaries": {
                    "preprocessing": dict(boundary),
                    "runtime_only": dict(boundary),
                    "end_to_end": dict(boundary),
                },
                "command": "inference_benchmark --warmup 20 --iterations 200",
                "iterations": 200,
                "warmup": 20,
            },
            "build": {
                "build_type": "Release",
                "compiler": "AppleClang test",
                "cxx": 201703,
                "onnxruntime": "1.19.2",
                "system": "Darwin/arm64",
            },
            "machine": {
                "chip": "Apple test",
                "memory_bytes": 1024,
                "operating_system": "macOS test",
            },
            "prediction_smoke": {
                "class_index": 2,
                "command": "infer --model reference.onnx --image smoke.ppm",
                "confidence": 0.7,
                "image": "smoke.ppm",
                "inference_ms": 1.0,
                "label": "bird",
                "preprocess_ms": 0.1,
                "total_latency_ms": 1.2,
            },
            "scope": {
                "file_decode_included": False,
                "session_construction_included": False,
                "workload": "one synthetic in-memory RGB image per operation",
            },
            "verification": {
                "ctest_command": "ctest --output-on-failure",
                "passed": 9,
                "total": 9,
            },
        }
        commands = {}
        for name in evidence_manifest.EXECUTED_PROVENANCE_COMMANDS:
            commands[name] = {
                "status": "executed",
                "command": f"{name} --reference-test",
                "argv": [name, "--reference-test"],
            }
        for name in evidence_manifest.CANONICAL_PROVENANCE_COMMANDS:
            commands[name] = {
                "status": "canonical",
                "command": f"{name} --reference-test",
                "argv": [name, "--reference-test"],
            }
        commands["benchmark"]["command"] = benchmark["benchmark"]["command"]
        commands["cli_smoke"]["command"] = benchmark["prediction_smoke"][
            "command"
        ]
        commands["trained_ctest"]["command"] = benchmark["verification"][
            "ctest_command"
        ]
        provenance = {
            "schema_version": 1,
            "reference_run_id": "v1.1-run-test",
            "working_directory": str(Path.cwd().resolve()),
            "capture_notes": dict(evidence_manifest.PROVENANCE_CAPTURE_NOTES),
            "commands": commands,
        }
        return training, parity, benchmark, environment, provenance

    @staticmethod
    def _write_bundle_artifacts(
        directory: Path, training, parity, benchmark, environment, provenance
    ):
        paths = {
            "checkpoint": directory / "source.pt",
            "onnx": directory / "source.onnx",
            "training_log": directory / "training.log",
            "training_record": directory / "training.json",
            "parity_record": directory / "parity.json",
            "benchmark_record": directory / "benchmark.json",
            "environment_record": directory / "environment.json",
            "provenance_record": directory / "provenance.json",
        }
        paths["checkpoint"].write_bytes(b"checkpoint")
        paths["onnx"].write_bytes(b"onnx")
        paths["training_log"].write_text("official training log\n", encoding="utf-8")
        for label, value in (
            ("training_record", training),
            ("parity_record", parity),
            ("benchmark_record", benchmark),
            ("environment_record", environment),
            ("provenance_record", provenance),
        ):
            paths[label].write_text(
                json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return paths

    def test_manifest_hashes_are_correct_deterministic_and_self_excluding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            checkpoint = directory / "model.pt"
            onnx = directory / "model.onnx"
            first_output = directory / "manifest-a.json"
            second_output = directory / "manifest-b.json"
            checkpoint.write_bytes(b"checkpoint-bytes")
            onnx.write_bytes(b"onnx-bytes")
            artifacts = {"checkpoint": checkpoint, "onnx": onnx}
            parity = {"max_abs_diff": 1.0e-8, "tolerance": 1.0e-4}
            benchmark = {"batch_size": 1, "p95_ms": 2.5}

            first = evidence_manifest.write_manifest(
                first_output, artifacts, parity=parity, benchmark=benchmark
            )
            second = evidence_manifest.write_manifest(
                second_output, artifacts, parity=parity, benchmark=benchmark
            )

            self.assertEqual(first, second)
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(
                first["artifacts"]["checkpoint"]["sha256"],
                hashlib.sha256(b"checkpoint-bytes").hexdigest(),
            )
            self.assertEqual(first["parity"], parity)
            self.assertEqual(first["benchmark"], benchmark)
            self.assertNotIn("manifest", first["artifacts"])
            with self.assertRaises(ValueError):
                evidence_manifest.write_manifest(
                    checkpoint, {"checkpoint": checkpoint}
                )

    def test_atomic_bundle_verifies_and_rejects_integrity_failures(self) -> None:
        train_indices = list(range(45000))
        validation_indices = list(range(45000, 50000))
        canonical = json.dumps(
            {"train": train_indices, "validation": validation_indices},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        split = {
            "dataset": {
                "name": "CIFAR-10",
                "implementation": "torchvision.datasets.CIFAR10",
                "archive": {
                    "url": evidence_manifest.CIFAR10_ARCHIVE_URL,
                    "md5": evidence_manifest.CIFAR10_ARCHIVE_MD5,
                    "size_bytes": evidence_manifest.CIFAR10_ARCHIVE_SIZE,
                },
            },
            "protocol": evidence_manifest.REFERENCE_SPLIT_PROTOCOL,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "digest": hashlib.sha256(canonical).hexdigest(),
            "train_class_counts": {
                label: 4500
                for label in (
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
            },
            "validation_class_counts": {
                label: 500
                for label in (
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
            },
        }
        onnx_sha256 = hashlib.sha256(b"onnx").hexdigest()
        training, parity, benchmark, environment, provenance = (
            self._reference_records(onnx_sha256)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            artifacts = self._write_bundle_artifacts(
                directory, training, parity, benchmark, environment, provenance
            )
            bundle = directory / "bundle"
            manifest = evidence_manifest.build_evidence_bundle(
                bundle,
                artifacts,
                split=split,
                training=training,
                parity=parity,
                benchmark=benchmark,
                environment=environment,
                provenance=provenance,
            )
            self.assertEqual(
                manifest, evidence_manifest.verify_evidence_bundle(bundle)
            )
            self.assertTrue((bundle / "manifest.json").is_file())
            self.assertTrue((bundle / "SHA256SUMS").is_file())
            with self.assertRaises(FileExistsError):
                evidence_manifest.build_evidence_bundle(
                    bundle,
                    artifacts,
                    split=split,
                    training=training,
                    parity=parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=provenance,
                )

            mutations = {}
            for name in ("modified", "missing", "extra", "truncated", "escape"):
                target = directory / name
                shutil.copytree(bundle, target)
                mutations[name] = target
            with (mutations["modified"] / "files" / "checkpoint.pt").open("ab") as stream:
                stream.write(b"changed")
            (mutations["missing"] / "files" / "onnx.onnx").unlink()
            (mutations["extra"] / "unexpected.txt").write_text("extra")
            (mutations["truncated"] / "manifest.json").write_bytes(b"{")
            sums_path = mutations["escape"] / "SHA256SUMS"
            lines = sums_path.read_text(encoding="ascii").splitlines()
            digest, _ = lines[0].split("  ", 1)
            lines[0] = f"{digest}  ../escape"
            sums_path.write_text("\n".join(lines) + "\n", encoding="ascii")

            for name, target in mutations.items():
                with self.subTest(name=name), self.assertRaises(
                    (ValueError, FileNotFoundError, json.JSONDecodeError)
                ):
                    evidence_manifest.verify_evidence_bundle(target)

            with self.assertRaises(ValueError):
                evidence_manifest.build_evidence_bundle(
                    directory / "below-floor",
                    artifacts,
                    split=split,
                    training={**training, "test_accuracy": 0.64},
                    parity=parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=provenance,
                )

            invalid_cases = {}
            wrong_device = copy.deepcopy(training)
            wrong_device["config"]["device"] = "mps"
            invalid_cases["wrong-device"] = {
                "training": wrong_device,
                "parity": parity,
                "benchmark": benchmark,
                "environment": environment,
            }
            wrong_environment = copy.deepcopy(environment)
            wrong_environment["device"] = {
                "requested": "mps",
                "resolved": "mps",
                "name": "Apple Metal Performance Shaders",
            }
            invalid_cases["wrong-environment-device"] = {
                "training": training,
                "parity": parity,
                "benchmark": benchmark,
                "environment": wrong_environment,
            }
            wrong_config = copy.deepcopy(training)
            wrong_config["config"]["batch_size"] = 64
            invalid_cases["wrong-config"] = {
                "training": wrong_config,
                "parity": parity,
                "benchmark": benchmark,
                "environment": environment,
            }
            wrong_tolerance = dict(parity)
            wrong_tolerance["tolerance"] = 1.0
            invalid_cases["wrong-tolerance"] = {
                "training": training,
                "parity": wrong_tolerance,
                "benchmark": benchmark,
                "environment": environment,
            }
            unlinked_benchmark = copy.deepcopy(benchmark)
            unlinked_benchmark["artifact"]["sha256"] = "0" * 64
            invalid_cases["unlinked-onnx-hash"] = {
                "training": training,
                "parity": parity,
                "benchmark": unlinked_benchmark,
                "environment": environment,
            }
            for records in invalid_cases.values():
                records["provenance"] = provenance
            for name, records in invalid_cases.items():
                with self.subTest(name=name), self.assertRaises(ValueError):
                    evidence_manifest.build_evidence_bundle(
                        directory / name,
                        artifacts,
                        split=split,
                        **records,
                    )

            with self.assertRaisesRegex(ValueError, "artifact labels"):
                missing_provenance_artifacts = dict(artifacts)
                del missing_provenance_artifacts["provenance_record"]
                evidence_manifest.build_evidence_bundle(
                    directory / "missing-artifacts",
                    missing_provenance_artifacts,
                    split=split,
                    training=training,
                    parity=parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=provenance,
                )

            semantically_different_parity = dict(parity)
            semantically_different_parity["max_abs_diff"] = 2.0e-8
            with self.assertRaisesRegex(ValueError, "copied parity_record JSON"):
                evidence_manifest.build_evidence_bundle(
                    directory / "record-mismatch",
                    artifacts,
                    split=split,
                    training=training,
                    parity=semantically_different_parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=provenance,
                )

            non_executed_provenance = copy.deepcopy(provenance)
            non_executed_provenance["commands"]["training"]["status"] = "canonical"
            with self.assertRaisesRegex(ValueError, "must have status executed"):
                evidence_manifest.build_evidence_bundle(
                    directory / "non-executed-provenance",
                    artifacts,
                    split=split,
                    training=training,
                    parity=parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=non_executed_provenance,
                )

            inferred_provenance = copy.deepcopy(provenance)
            inferred_provenance["commands"]["training"]["inferred"] = True
            with self.assertRaisesRegex(ValueError, "invalid fields: training"):
                evidence_manifest.build_evidence_bundle(
                    directory / "inferred-provenance",
                    artifacts,
                    split=split,
                    training=training,
                    parity=parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=inferred_provenance,
                )

            altered_provenance = copy.deepcopy(provenance)
            altered_provenance["commands"]["training"]["command"] += " --altered"
            with self.assertRaisesRegex(ValueError, "copied provenance_record JSON"):
                evidence_manifest.build_evidence_bundle(
                    directory / "altered-provenance-command",
                    artifacts,
                    split=split,
                    training=training,
                    parity=parity,
                    benchmark=benchmark,
                    environment=environment,
                    provenance=altered_provenance,
                )

            command_link_drift = copy.deepcopy(provenance)
            command_link_drift["commands"]["benchmark"]["command"] += " --drift"
            with self.assertRaisesRegex(ValueError, "differs from benchmark evidence"):
                evidence_manifest._validate_command_links(
                    benchmark, command_link_drift
                )

            provenance_mismatch_bundle = directory / "provenance-manifest-mismatch"
            shutil.copytree(bundle, provenance_mismatch_bundle)
            manifest_path = provenance_mismatch_bundle / "manifest.json"
            mismatched_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mismatched_manifest["provenance"]["commands"]["training"][
                "command"
            ] += " --manifest-only"
            manifest_path.write_text(
                json.dumps(
                    mismatched_manifest, allow_nan=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            sums_path = provenance_mismatch_bundle / "SHA256SUMS"
            sums = []
            for line in sums_path.read_text(encoding="ascii").splitlines():
                _, relative = line.split("  ", 1)
                digest = (
                    evidence_manifest.sha256_file(manifest_path)
                    if relative == "manifest.json"
                    else line.split("  ", 1)[0]
                )
                sums.append(f"{digest}  {relative}")
            sums_path.write_text("\n".join(sums) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "copied provenance_record JSON"):
                evidence_manifest.verify_evidence_bundle(provenance_mismatch_bundle)


@unittest.skipUnless(
    TORCH_AVAILABLE and ONNX_AVAILABLE and ORT_AVAILABLE,
    "PyTorch, ONNX, and ONNX Runtime are required for parity smoke test",
)
class ExportParitySmokeTests(unittest.TestCase):
    def test_export_has_dynamic_batch_and_matches_pytorch(self) -> None:
        import onnx
        import export_onnx
        import verify_onnx

        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            weights_path = directory / "model.pt"
            onnx_path = directory / "model.onnx"
            parity_path = directory / "parity.json"
            torch.save(train.SmallCNN().state_dict(), weights_path)

            export_onnx.export(str(weights_path), str(onnx_path), opset=17)
            self.assertTrue(onnx_path.is_file())

            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            input_shape = model.graph.input[0].type.tensor_type.shape.dim
            output_shape = model.graph.output[0].type.tensor_type.shape.dim
            self.assertEqual(input_shape[0].dim_param, "batch")
            self.assertEqual(output_shape[0].dim_param, "batch")
            self.assertTrue(
                verify_onnx.verify(
                    str(weights_path),
                    str(onnx_path),
                    tol=1.0e-4,
                    json_out=parity_path,
                )
            )
            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            self.assertTrue(parity["passed"])
            self.assertTrue(parity["class_match"])
            self.assertLess(parity["max_abs_diff"], parity["tolerance"])

            extra_output_path = directory / "model-extra-output.onnx"
            extra_output_model = onnx.load(str(onnx_path))
            extra_output_model.graph.node.append(
                onnx.helper.make_node("Identity", ["logits"], ["extra_logits"])
            )
            extra_output_model.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    "extra_logits", onnx.TensorProto.FLOAT, ["batch", 10]
                )
            )
            onnx.save(extra_output_model, str(extra_output_path))
            with self.assertRaisesRegex(ValueError, "exactly one output"):
                verify_onnx.verify(
                    str(weights_path), str(extra_output_path), tol=1.0e-4
                )


if __name__ == "__main__":
    unittest.main()
