"""Train, select, and record a reproducible CIFAR-10 reference model."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from model import (
    CHECKPOINT_FORMAT_VERSION,
    CIFAR10_ARCHIVE_MD5,
    CIFAR10_ARCHIVE_SIZE,
    CIFAR10_ARCHIVE_URL,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATASET_IMPLEMENTATION,
    DATASET_NAME,
    MODEL_ARCHITECTURE,
    REFERENCE_SPLIT_PROTOCOL,
    REFERENCE_TRAINING_SIZE,
    REFERENCE_TRAIN_PER_CLASS,
    REFERENCE_VALIDATION_PER_CLASS,
    REFERENCE_VALIDATION_SIZE,
    SmallCNN,
    model_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "onnx",
    "onnxruntime",
    "numpy",
    "Pillow",
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    training_seed: int = 1337
    split_seed: int = 1337
    validation_size: int = REFERENCE_VALIDATION_SIZE
    num_workers: int = 0
    optimizer: str = "Adam"
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be a positive finite number")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.validation_size != REFERENCE_VALIDATION_SIZE:
            raise ValueError(
                f"validation_size must be exactly {REFERENCE_VALIDATION_SIZE} "
                "for the frozen reference split"
            )
        if self.optimizer != "Adam":
            raise ValueError("optimizer must be Adam for the frozen reference protocol")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        for name, seed in (
            ("training_seed", self.training_seed),
            ("split_seed", self.split_seed),
        ):
            if seed < 0 or seed > (2**63 - 1):
                raise ValueError(f"{name} must be in [0, 2^63-1]")


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    digest: str
    train_class_counts: Mapping[str, int]
    validation_class_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "train_class_counts", MappingProxyType(dict(self.train_class_counts))
        )
        object.__setattr__(
            self,
            "validation_class_counts",
            MappingProxyType(dict(self.validation_class_counts)),
        )

    def validate(self) -> None:
        if len(self.train_indices) != REFERENCE_TRAINING_SIZE:
            raise ValueError("dataset split must contain 45,000 training indices")
        if len(self.validation_indices) != REFERENCE_VALIDATION_SIZE:
            raise ValueError("dataset split must contain 5,000 validation indices")
        if set(self.train_indices).intersection(self.validation_indices):
            raise ValueError("training and validation split indices overlap")
        if set(self.train_indices).union(self.validation_indices) != set(range(50000)):
            raise ValueError("dataset split does not partition all 50,000 examples")
        if self.digest != split_digest(self.train_indices, self.validation_indices):
            raise ValueError("dataset split digest does not match its indices")
        expected_train = {
            label: REFERENCE_TRAIN_PER_CLASS for label in CIFAR10_CLASSES
        }
        expected_validation = {
            label: REFERENCE_VALIDATION_PER_CLASS for label in CIFAR10_CLASSES
        }
        if self.train_class_counts != expected_train:
            raise ValueError("dataset split training class counts are not stratified")
        if self.validation_class_counts != expected_validation:
            raise ValueError("dataset split validation class counts are not stratified")


@dataclass(frozen=True)
class ReferenceDataLoaders:
    train_loader: Any
    validation_loader: Any
    test_loader: Any
    split: DatasetSplit

    @property
    def train_indices(self) -> tuple[int, ...]:
        return self.split.train_indices

    @property
    def validation_indices(self) -> tuple[int, ...]:
        return self.split.validation_indices

    @property
    def split_digest(self) -> str:
        return self.split.digest

    @property
    def train_class_counts(self) -> Mapping[str, int]:
        return self.split.train_class_counts

    @property
    def validation_class_counts(self) -> Mapping[str, int]:
        return self.split.validation_class_counts


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_accuracy: float


@dataclass(frozen=True)
class TrainingSelection:
    selected_epoch: int
    validation_accuracy: float
    duration_seconds: float
    epoch_history: tuple[EpochRecord, ...]


@dataclass(frozen=True)
class TrainingResult:
    selected_epoch: int
    validation_accuracy: float
    test_accuracy: float
    duration_seconds: float
    epoch_history: tuple[EpochRecord, ...]


def _transforms():
    from torchvision import transforms

    training = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return training, evaluation


def get_dataloaders(
    batch_size: int,
    data_dir: str,
    *,
    num_workers: int = 0,
    seed: int = 1337,
    download: bool = True,
):
    """Return the v1 two-loader API for backward compatibility.

    New reference training uses :func:`get_reference_dataloaders` so the test
    set is not consulted during model selection.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    training_transform, evaluation_transform = _transforms()
    from torchvision import datasets

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=training_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=evaluation_transform
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, generator=generator, **common
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader


def get_reference_dataloaders(
    batch_size: int,
    data_dir: str,
    *,
    validation_size: int = REFERENCE_VALIDATION_SIZE,
    num_workers: int = 0,
    training_seed: int = 1337,
    split_seed: int = 1337,
    download: bool = True,
    pin_memory: bool = False,
) -> ReferenceDataLoaders:
    """Build deterministic, disjoint train/validation/test loaders.

    Training and validation subsets use separate CIFAR-10 dataset objects, so
    random training augmentation can never leak into validation samples.
    """

    config = TrainingConfig(
        batch_size=batch_size,
        validation_size=validation_size,
        num_workers=num_workers,
        training_seed=training_seed,
        split_seed=split_seed,
    )
    config.validate()
    training_transform, evaluation_transform = _transforms()
    from torchvision import datasets

    training_source = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=training_transform
    )
    validation_source = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=evaluation_transform
    )
    test_source = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=evaluation_transform
    )
    if len(training_source) != len(validation_source):
        raise ValueError("training and validation CIFAR-10 sources differ in size")
    training_targets = getattr(training_source, "targets", None)
    validation_targets = getattr(validation_source, "targets", None)
    if training_targets is None or validation_targets is None:
        raise ValueError("CIFAR-10 sources must expose targets for stratification")
    if list(training_targets) != list(validation_targets):
        raise ValueError("training and validation CIFAR-10 targets differ")
    split = make_dataset_split(training_targets, split_seed)
    train_indices = split.train_indices
    validation_indices = split.validation_indices
    train_subset = torch.utils.data.Subset(training_source, train_indices)
    validation_subset = torch.utils.data.Subset(
        validation_source, validation_indices
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    training_generator = torch.Generator().manual_seed(training_seed)
    train_loader = torch.utils.data.DataLoader(
        train_subset, shuffle=True, generator=training_generator, **common
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_subset, shuffle=False, **common
    )
    test_loader = torch.utils.data.DataLoader(test_source, shuffle=False, **common)
    return ReferenceDataLoaders(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        split=split,
    )


def class_counts(targets: Iterable[int], indices: Iterable[int]) -> dict[str, int]:
    target_values = list(targets)
    counts = {label: 0 for label in CIFAR10_CLASSES}
    for index in indices:
        class_index = int(target_values[index])
        if class_index < 0 or class_index >= len(CIFAR10_CLASSES):
            raise ValueError(f"unexpected CIFAR-10 class index: {class_index}")
        counts[CIFAR10_CLASSES[class_index]] += 1
    return counts


def split_digest(
    train_indices: Iterable[int], validation_indices: Iterable[int]
) -> str:
    """Hash the canonical ordered split membership without platform ambiguity."""

    payload = json.dumps(
        {
            "train": list(train_indices),
            "validation": list(validation_indices),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def stratified_reference_split(
    targets: Iterable[int], split_seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Create the frozen 4,500/500-per-class CIFAR-10 split."""

    target_values = [int(value) for value in targets]
    expected_total = REFERENCE_TRAINING_SIZE + REFERENCE_VALIDATION_SIZE
    if len(target_values) != expected_total:
        raise ValueError(
            f"reference CIFAR-10 training source must contain {expected_total} examples"
        )
    by_class = [[] for _ in CIFAR10_CLASSES]
    for index, class_index in enumerate(target_values):
        if class_index < 0 or class_index >= len(CIFAR10_CLASSES):
            raise ValueError(f"unexpected CIFAR-10 class index: {class_index}")
        by_class[class_index].append(index)
    if any(len(indices) != 5000 for indices in by_class):
        raise ValueError("reference split requires exactly 5,000 examples per class")

    generator = torch.Generator().manual_seed(split_seed)
    train_indices = []
    validation_indices = []
    for indices in by_class:
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in order]
        validation_indices.extend(shuffled[:REFERENCE_VALIDATION_PER_CLASS])
        train_indices.extend(shuffled[REFERENCE_VALIDATION_PER_CLASS:])
    return tuple(sorted(train_indices)), tuple(sorted(validation_indices))


def make_dataset_split(targets: Iterable[int], split_seed: int) -> DatasetSplit:
    """Create and validate the immutable frozen reference partition."""

    target_values = [int(value) for value in targets]
    train_indices, validation_indices = stratified_reference_split(
        target_values, split_seed
    )
    split = DatasetSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        digest=split_digest(train_indices, validation_indices),
        train_class_counts=class_counts(target_values, train_indices),
        validation_class_counts=class_counts(target_values, validation_indices),
    )
    split.validate()
    return split


def train_one_epoch(model, loader: Iterable, optimizer, device: torch.device) -> float:
    """Train for one epoch and return sample-weighted mean cross-entropy."""

    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        sample_count = int(targets.shape[0])
        total_loss += float(loss.detach().item()) * sample_count
        total_samples += sample_count
    if total_samples == 0:
        raise ValueError("training loader produced no samples")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader: Iterable, device: torch.device) -> float:
    """Evaluate exact top-1 accuracy and leave the model in evaluation mode."""

    model.eval()
    correct = 0
    total = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(features).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        total += int(targets.shape[0])
    if total == 0:
        raise ValueError("evaluation loader produced no samples")
    return correct / total


def train_with_validation(
    model,
    train_loader: Iterable,
    validation_loader: Iterable,
    optimizer,
    device: torch.device,
    epochs: int,
) -> TrainingSelection:
    """Select and restore validation-best weights without access to test data."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    started = time.perf_counter()
    best_accuracy = -math.inf
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_accuracy = evaluate(model, validation_loader, device)
        history.append(EpochRecord(epoch, loss, validation_accuracy))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[train] epoch={epoch:02d}/{epochs} loss={loss:.4f} "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a selectable model state")
    model.load_state_dict(best_state, strict=True)
    duration = time.perf_counter() - started
    return TrainingSelection(
        selected_epoch=best_epoch,
        validation_accuracy=best_accuracy,
        duration_seconds=duration,
        epoch_history=tuple(history),
    )


def evaluate_test_once(
    model,
    test_loader: Iterable,
    device: torch.device,
    selection: TrainingSelection,
) -> TrainingResult:
    """Evaluate the restored selected model once and bind that metric to it."""

    started = time.perf_counter()
    test_accuracy = evaluate(model, test_loader, device)
    if not math.isfinite(test_accuracy) or not 0.0 <= test_accuracy <= 1.0:
        raise ValueError("test accuracy must be finite and in [0, 1]")
    return TrainingResult(
        selected_epoch=selection.selected_epoch,
        validation_accuracy=selection.validation_accuracy,
        test_accuracy=test_accuracy,
        duration_seconds=selection.duration_seconds + (time.perf_counter() - started),
        epoch_history=selection.epoch_history,
    )


def mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )


def resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not mps_available():
        raise ValueError("MPS was requested but is unavailable")
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif mps_available():
            requested = "mps"
        else:
            requested = "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def _package_versions() -> dict[str, str]:
    versions = {}
    for distribution in DIRECT_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def collect_environment(requested_device: str, device: torch.device) -> dict[str, Any]:
    device_name = str(device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = "Apple Metal Performance Shaders"
    packages = _package_versions()
    python_version = platform.python_version()
    system = platform.system()
    machine = platform.machine()
    identifier = (
        f"{system.lower()}-{machine.lower()}-python{python_version}-"
        f"torch{packages['torch']}-torchvision{packages['torchvision']}-"
        f"onnx{packages['onnx']}-onnxruntime{packages['onnxruntime']}"
    )
    return {
        "identifier": identifier,
        "python": python_version,
        "platform": platform.platform(),
        "system": system,
        "machine": machine,
        "python_executable": sys.executable,
        "packages": packages,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": {
            "requested": requested_device,
            "resolved": str(device),
            "name": device_name,
        },
    }


def build_checkpoint(
    model,
    config: TrainingConfig,
    data: ReferenceDataLoaders,
    result: TrainingResult,
    requested_device: str,
    device: torch.device,
) -> dict[str, Any]:
    """Create a schema-v2 checkpoint whose metrics describe its exact weights."""

    config.validate()
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("schema-v2 checkpoint requires deterministic algorithms")
    if len(result.epoch_history) != config.epochs:
        raise ValueError("training result must contain one history record per epoch")
    if not 1 <= result.selected_epoch <= config.epochs:
        raise ValueError("training result selected_epoch is outside the configured run")
    for expected_epoch, record in enumerate(result.epoch_history, start=1):
        if record.epoch != expected_epoch:
            raise ValueError("training result epoch history is not contiguous")
        if (
            not math.isfinite(record.training_loss)
            or not math.isfinite(record.validation_accuracy)
            or not 0.0 <= record.validation_accuracy <= 1.0
        ):
            raise ValueError("training result epoch history contains invalid metrics")
    selected_record = result.epoch_history[result.selected_epoch - 1]
    if selected_record.validation_accuracy != result.validation_accuracy:
        raise ValueError("selected validation metric does not match epoch history")
    if (
        not math.isfinite(result.test_accuracy)
        or not 0.0 <= result.test_accuracy <= 1.0
        or not math.isfinite(result.duration_seconds)
        or result.duration_seconds < 0.0
    ):
        raise ValueError("training result test metric or duration is invalid")
    data.split.validate()
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": state_dict,
        # Keep these v1 fields for consumers while model_contract is canonical.
        "architecture": MODEL_ARCHITECTURE,
        "classes": CIFAR10_CLASSES,
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
        "model_contract": model_contract(),
        "dataset": {
            "name": DATASET_NAME,
            "implementation": DATASET_IMPLEMENTATION,
            "archive": {
                "url": CIFAR10_ARCHIVE_URL,
                "md5": CIFAR10_ARCHIVE_MD5,
                "size_bytes": CIFAR10_ARCHIVE_SIZE,
            },
            "training_examples": len(data.train_indices),
            "validation_examples": len(data.validation_indices),
            "test_examples": len(data.test_loader.dataset),
            "split_protocol": REFERENCE_SPLIT_PROTOCOL,
            "split_digest": data.split.digest,
            "train_class_counts": dict(data.split.train_class_counts),
            "validation_class_counts": dict(data.split.validation_class_counts),
        },
        "training": {
            **asdict(config),
            "selected_epoch": result.selected_epoch,
        },
        "metrics": {
            "validation_accuracy": result.validation_accuracy,
            "test_accuracy": result.test_accuracy,
        },
        "environment": collect_environment(requested_device, device),
        "duration_seconds": result.duration_seconds,
        "epoch_history": [asdict(record) for record in result.epoch_history],
    }


def write_reference_records(
    directory: str | Path,
    checkpoint: Mapping[str, Any],
    data: ReferenceDataLoaders,
) -> dict[str, Path]:
    """Write local JSON inputs for later atomic evidence-bundle finalization."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    records = {
        "split": {
            "dataset": {
                "name": DATASET_NAME,
                "implementation": DATASET_IMPLEMENTATION,
                "archive": {
                    "url": CIFAR10_ARCHIVE_URL,
                    "md5": CIFAR10_ARCHIVE_MD5,
                    "size_bytes": CIFAR10_ARCHIVE_SIZE,
                },
            },
            "protocol": REFERENCE_SPLIT_PROTOCOL,
            "train_indices": list(data.split.train_indices),
            "validation_indices": list(data.split.validation_indices),
            "digest": data.split.digest,
            "train_class_counts": dict(data.split.train_class_counts),
            "validation_class_counts": dict(data.split.validation_class_counts),
        },
        "training": {
            "selected_epoch": checkpoint["training"]["selected_epoch"],
            "validation_accuracy": checkpoint["metrics"]["validation_accuracy"],
            "test_accuracy": checkpoint["metrics"]["test_accuracy"],
            "duration_seconds": checkpoint["duration_seconds"],
            "config": checkpoint["training"],
            "epoch_history": checkpoint["epoch_history"],
        },
        "environment": checkpoint["environment"],
    }
    paths = {}
    for name, value in records.items():
        path = destination / f"{name}.json"
        path.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and select a reproducible CIFAR-10 reference CNN."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "python" / "data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "cifar10_cnn.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337, help="Training/shuffle seed")
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="Explicit local directory for split/training/environment JSON records",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        training_seed=args.seed,
        split_seed=args.split_seed,
        validation_size=REFERENCE_VALIDATION_SIZE,
        num_workers=args.num_workers,
        optimizer="Adam",
        device=args.device,
    )
    try:
        config.validate()
        device = resolve_device(config.device)
    except ValueError as error:
        parser.error(str(error))

    seed_everything(config.training_seed)
    print(
        f"[train] device={device} requested_device={args.device} "
        f"training_seed={config.training_seed} split_seed={config.split_seed}"
    )
    data = get_reference_dataloaders(
        config.batch_size,
        str(args.data_dir),
        validation_size=config.validation_size,
        num_workers=config.num_workers,
        training_seed=config.training_seed,
        split_seed=config.split_seed,
        download=not args.no_download,
        pin_memory=device.type == "cuda",
    )
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    selection = train_with_validation(
        model,
        data.train_loader,
        data.validation_loader,
        optimizer,
        device,
        config.epochs,
    )
    result = evaluate_test_once(model, data.test_loader, device, selection)
    checkpoint = build_checkpoint(model, config, data, result, args.device, device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    if args.record_dir is not None:
        records = write_reference_records(args.record_dir, checkpoint, data)
        print(
            "[train] wrote local evidence records: "
            + ", ".join(str(path) for path in records.values())
        )
    print(
        f"[train] wrote {args.out} selected_epoch={result.selected_epoch} "
        f"validation_accuracy={result.validation_accuracy:.4f} "
        f"test_accuracy={result.test_accuracy:.4f} "
        f"duration_seconds={result.duration_seconds:.3f}"
    )


if __name__ == "__main__":
    main()
