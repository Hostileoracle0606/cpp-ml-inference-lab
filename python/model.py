"""Shared CIFAR-10 model and checkpoint contract.

Keeping this module independent of torchvision lets export and parity checks run
without importing the dataset/training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping")

    model = SmallCNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, payload
