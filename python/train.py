"""Train the v1 CIFAR-10 CNN and write a reproducible checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from model import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, SmallCNN


REPO_ROOT = Path(__file__).resolve().parents[1]


def get_dataloaders(
    batch_size: int,
    data_dir: str,
    *,
    num_workers: int = 0,
    seed: int = 1337,
    download: bool = True,
):
    """Return training and evaluation loaders with the shared normalization."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    from torchvision import datasets, transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **common
    )
    return train_loader, test_loader


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "python" / "data")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "cifar10_cnn.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("--epochs, --batch-size, and --lr must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")

    seed_everything(args.seed)
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    print(f"[train] device={device} seed={args.seed} classes={len(CIFAR10_CLASSES)}")

    train_loader, test_loader = get_dataloaders(
        args.batch_size,
        str(args.data_dir),
        num_workers=args.num_workers,
        seed=args.seed,
        download=not args.no_download,
    )
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        best_accuracy = max(best_accuracy, accuracy)
        print(
            f"[train] epoch={epoch:02d}/{args.epochs} "
            f"loss={loss:.4f} accuracy={accuracy:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.cpu().state_dict(),
        "classes": CIFAR10_CLASSES,
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
        "epochs": args.epochs,
        "best_accuracy": best_accuracy,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.out)
    print(f"[train] wrote {args.out} best_accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
