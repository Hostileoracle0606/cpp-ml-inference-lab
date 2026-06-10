"""Stage 1 — Train a small CNN on CIFAR-10.

This is the Stage 0 *skeleton*: the model architecture and program structure are
defined, but the data-loading and training/eval loops are left as Stage 1 work
(marked with `TODO (Stage 1)` and `raise NotImplementedError`).

Target usage (once Stage 1 lands):
    python train.py --epochs 20 --batch-size 128 --out ../models/cifar10_cnn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# CIFAR-10 class order is fixed by torchvision; the C++ decoder must match this.
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Per-channel normalization stats. These exact constants are the train/serve
# contract — the C++ preprocessor (Stage 3) must reuse them verbatim.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SmallCNN(nn.Module):
    """A compact conv net sized for CIFAR-10 (3x32x32 -> 10 logits).

    Deliberately small so that C++ inference latency stays in the single-digit
    millisecond range — the point of the project is the deployment layer, not
    squeezing out the last accuracy point.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_dataloaders(batch_size: int, data_dir: str):
    """Return (train_loader, test_loader) for CIFAR-10.

    TODO (Stage 1): build torchvision.datasets.CIFAR10 with a transform that
    applies ToTensor() + Normalize(CIFAR10_MEAN, CIFAR10_STD), wrap in DataLoaders.
    """
    raise NotImplementedError("Stage 1: implement CIFAR-10 data loading")


def train_one_epoch(model, loader, optimizer, device) -> float:
    """Run one training epoch; return mean loss.

    TODO (Stage 1): standard loop — zero_grad, forward, cross_entropy, backward, step.
    """
    raise NotImplementedError("Stage 1: implement the training loop")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return top-1 accuracy on the given loader.

    TODO (Stage 1): forward pass, argmax, compare to labels, accumulate accuracy.
    """
    raise NotImplementedError("Stage 1: implement evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small CNN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--out", type=str, default="../models/cifar10_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  classes={len(CIFAR10_CLASSES)}")

    model = SmallCNN().to(device)

    # TODO (Stage 1): wire up the real training run.
    #   train_loader, test_loader = get_dataloaders(args.batch_size, args.data_dir)
    #   optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #   for epoch in range(args.epochs):
    #       loss = train_one_epoch(model, train_loader, optimizer, device)
    #       acc = evaluate(model, test_loader, device)
    #       print(f"epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")
    #   Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    #   torch.save(model.state_dict(), args.out)

    raise SystemExit(
        "train.py is a Stage 0 skeleton — implement get_dataloaders / "
        "train_one_epoch / evaluate in Stage 1."
    )


if __name__ == "__main__":
    main()
