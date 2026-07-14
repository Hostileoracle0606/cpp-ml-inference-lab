"""Dependency-aware smoke tests for the Python training/export boundary.

The tests use synthetic tensors and a temporary checkpoint: they never download
CIFAR-10 and never write model artifacts into the repository.
"""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
CPP_PREPROCESSOR_HEADER = (
    Path(__file__).resolve().parents[1] / "include" / "cpp_ml" / "preprocessor.hpp"
)
sys.path.insert(0, str(PYTHON_DIR))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None
ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import train


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
                verify_onnx.verify(str(weights_path), str(onnx_path), tol=1.0e-4)
            )


if __name__ == "__main__":
    unittest.main()
