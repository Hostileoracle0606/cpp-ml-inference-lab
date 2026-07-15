"""Shared CIFAR-10 model and checkpoint contract.

Keeping this module independent of torchvision lets export and parity checks run
without importing the dataset/training stack.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


CIFAR10_CLASSES = (
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
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CHECKPOINT_FORMAT_VERSION = 2
DATASET_NAME = "CIFAR-10"
DATASET_IMPLEMENTATION = "torchvision.datasets.CIFAR10"
CIFAR10_ARCHIVE_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_ARCHIVE_SIZE = 170498071
MODEL_ARCHITECTURE = "SmallCNN"
MODEL_INPUT_SHAPE = (3, 32, 32)
REFERENCE_SPLIT_PROTOCOL = "cifar10-stratified-4500-train-500-validation-v1"
REFERENCE_TRAIN_PER_CLASS = 4500
REFERENCE_VALIDATION_PER_CLASS = 500
REFERENCE_VALIDATION_SIZE = REFERENCE_VALIDATION_PER_CLASS * len(CIFAR10_CLASSES)
REFERENCE_TRAINING_SIZE = REFERENCE_TRAIN_PER_CLASS * len(CIFAR10_CLASSES)


class SmallCNN(nn.Module):
    """Compact CIFAR-10 CNN with a ``[N, 3, 32, 32] -> [N, 10]`` contract."""

    def __init__(self, num_classes: int = len(CIFAR10_CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def model_contract() -> dict[str, Any]:
    """Return the serializable contract shared by training and deployment."""

    return {
        "architecture": MODEL_ARCHITECTURE,
        "input_layout": "NCHW",
        "input_shape": MODEL_INPUT_SHAPE,
        "output_size": len(CIFAR10_CLASSES),
        "classes": CIFAR10_CLASSES,
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
    }


def _tuple_field(value: Any, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"checkpoint {field} must be a sequence")
    return tuple(value)


def _validate_legacy_contract(payload: Mapping[str, Any]) -> None:
    expected_fields = {
        "classes": CIFAR10_CLASSES,
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
    }
    for field, expected in expected_fields.items():
        if field in payload and _tuple_field(payload[field], field) != tuple(expected):
            raise ValueError(f"checkpoint {field} does not match the model contract")
    if "architecture" in payload and payload["architecture"] != MODEL_ARCHITECTURE:
        raise ValueError("checkpoint architecture does not match SmallCNN")


def _validate_v2_contract(payload: Mapping[str, Any]) -> None:
    contract = payload.get("model_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("schema-v2 checkpoint requires model_contract metadata")

    expected = model_contract()
    for field in ("architecture", "input_layout", "output_size"):
        if contract.get(field) != expected[field]:
            raise ValueError(
                f"checkpoint model_contract.{field} does not match the runtime contract"
            )
    for field in (
        "input_shape",
        "classes",
        "normalization_mean",
        "normalization_std",
    ):
        if field not in contract or _tuple_field(
            contract[field], f"model_contract.{field}"
        ) != tuple(expected[field]):
            raise ValueError(
                f"checkpoint model_contract.{field} does not match the runtime contract"
            )

    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("schema-v2 checkpoint requires dataset metadata")
    if dataset.get("name") != DATASET_NAME or dataset.get(
        "implementation"
    ) != DATASET_IMPLEMENTATION:
        raise ValueError("checkpoint dataset identity does not match CIFAR-10")
    if dataset.get("archive") != {
        "url": CIFAR10_ARCHIVE_URL,
        "md5": CIFAR10_ARCHIVE_MD5,
        "size_bytes": CIFAR10_ARCHIVE_SIZE,
    }:
        raise ValueError("checkpoint dataset archive provenance does not match CIFAR-10")
    if (
        dataset.get("training_examples") != REFERENCE_TRAINING_SIZE
        or dataset.get("validation_examples") != REFERENCE_VALIDATION_SIZE
        or dataset.get("test_examples") != 10000
    ):
        raise ValueError("checkpoint dataset sizes do not match the reference protocol")
    if dataset.get("split_protocol") != REFERENCE_SPLIT_PROTOCOL:
        raise ValueError("checkpoint dataset split protocol is unsupported")
    digest = dataset.get("split_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("checkpoint dataset split_digest must be lowercase SHA-256")
    for field, expected_count in (
        ("train_class_counts", REFERENCE_TRAIN_PER_CLASS),
        ("validation_class_counts", REFERENCE_VALIDATION_PER_CLASS),
    ):
        counts = dataset.get(field)
        if not isinstance(counts, Mapping) or dict(counts) != {
            label: expected_count for label in CIFAR10_CLASSES
        }:
            raise ValueError(f"checkpoint dataset {field} violates stratification")

    training = payload.get("training")
    required_training = {
        "epochs",
        "batch_size",
        "learning_rate",
        "training_seed",
        "split_seed",
        "validation_size",
        "num_workers",
        "optimizer",
        "device",
        "selected_epoch",
    }
    if not isinstance(training, Mapping) or not required_training.issubset(training):
        raise ValueError("schema-v2 checkpoint has incomplete training metadata")
    if training.get("validation_size") != REFERENCE_VALIDATION_SIZE:
        raise ValueError("checkpoint training split is not the frozen 5,000 sample split")
    if training.get("optimizer") != "Adam" or training.get("device") not in {
        "auto",
        "cpu",
        "cuda",
        "mps",
    }:
        raise ValueError("checkpoint optimizer/device metadata violates the protocol")
    epochs = training.get("epochs")
    selected_epoch = training.get("selected_epoch")
    if (
        type(epochs) is not int
        or type(selected_epoch) is not int
        or epochs <= 0
        or selected_epoch <= 0
        or selected_epoch > epochs
    ):
        raise ValueError("checkpoint selected epoch is invalid")

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("schema-v2 checkpoint requires metrics metadata")
    for name in ("validation_accuracy", "test_accuracy"):
        value = metrics.get(name)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not (
            0.0 <= value <= 1.0
        ):
            raise ValueError(f"checkpoint metric {name} is invalid")

    duration = payload.get("duration_seconds")
    if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
        raise ValueError("checkpoint duration_seconds is invalid")
    history = payload.get("epoch_history")
    if (
        isinstance(history, (str, bytes))
        or not isinstance(history, Sequence)
        or len(history) != epochs
    ):
        raise ValueError("checkpoint epoch_history must contain one record per epoch")
    for expected_epoch, record in enumerate(history, start=1):
        if not isinstance(record, Mapping) or record.get("epoch") != expected_epoch:
            raise ValueError("checkpoint epoch_history is not contiguous")
        loss = record.get("training_loss")
        validation_accuracy = record.get("validation_accuracy")
        if (
            not isinstance(loss, (float, int))
            or not math.isfinite(loss)
            or not isinstance(validation_accuracy, (float, int))
            or not math.isfinite(validation_accuracy)
            or not 0.0 <= validation_accuracy <= 1.0
        ):
            raise ValueError("checkpoint epoch_history contains invalid metrics")
    if history[selected_epoch - 1].get("validation_accuracy") != metrics.get(
        "validation_accuracy"
    ):
        raise ValueError("checkpoint selected metric does not match epoch_history")

    environment = payload.get("environment")
    required_environment = {
        "identifier",
        "python",
        "platform",
        "system",
        "machine",
        "python_executable",
        "packages",
        "device",
        "deterministic_algorithms",
    }
    if not isinstance(environment, Mapping) or not required_environment.issubset(
        environment
    ):
        raise ValueError("schema-v2 checkpoint has incomplete environment metadata")
    if not isinstance(environment.get("packages"), Mapping) or not isinstance(
        environment.get("device"), Mapping
    ):
        raise ValueError("checkpoint package/device metadata must be mappings")
    required_packages = {"torch", "torchvision", "onnx", "onnxruntime", "numpy", "Pillow"}
    if not required_packages.issubset(environment["packages"]):
        raise ValueError("checkpoint environment omits key package versions")
    if environment.get("deterministic_algorithms") is not True:
        raise ValueError("schema-v2 checkpoint requires deterministic algorithms")


def validate_checkpoint_metadata(payload: Mapping[str, Any]) -> int | None:
    """Validate checkpoint metadata and return its schema, or ``None`` for raw weights."""

    if "model_state_dict" not in payload:
        return None
    version = payload.get("format_version")
    if type(version) is not int or version not in (1, CHECKPOINT_FORMAT_VERSION):
        raise ValueError(f"unsupported checkpoint format_version: {version!r}")
    _validate_legacy_contract(payload)
    if version == CHECKPOINT_FORMAT_VERSION:
        _validate_v2_contract(payload)
    return version


def load_checkpoint(path: str | Path) -> tuple[SmallCNN, Mapping[str, Any]]:
    """Load either a v1 metadata checkpoint or a legacy raw state dict."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must contain a state-dict mapping")
    validate_checkpoint_metadata(payload)
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")

    model = SmallCNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, payload
