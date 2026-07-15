"""Fail-closed import gate for the fully provisioned Python release profile."""

from __future__ import annotations

import importlib


MODULES = ("torch", "torchvision", "onnx", "onnxruntime", "numpy", "PIL")


def main() -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}={version}")


if __name__ == "__main__":
    main()
