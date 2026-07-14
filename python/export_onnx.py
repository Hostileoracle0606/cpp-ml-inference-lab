"""Export a trained v1 checkpoint to the frozen ONNX endpoint contract."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from model import load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]


def export(weights_path: str, out_path: str, opset: int = 17) -> None:
    """Export dynamic-batch float32 ``input`` to ``logits``."""

    if opset <= 0:
        raise ValueError("opset must be positive")
    model, _ = load_checkpoint(weights_path)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    options = {
        "export_params": True,
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {"input": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
        "do_constant_folding": True,
    }
    # The legacy exporter has stable dynamic_axes behavior across the supported
    # PyTorch versions. Older PyTorch releases do not expose this argument.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    torch.onnx.export(model, dummy, str(output), **options)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"ONNX export did not produce a model: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch CNN to ONNX.")
    parser.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "models" / "cifar10_cnn.pt"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "models" / "cifar10_cnn.onnx"
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export(str(args.weights), str(args.out), args.opset)
    print(f"[export] wrote {args.out} (input -> logits, dynamic batch)")


if __name__ == "__main__":
    main()
