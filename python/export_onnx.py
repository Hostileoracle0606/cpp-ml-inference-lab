"""Stage 2 — Export the trained PyTorch model to ONNX.

Stage 0 skeleton: the export structure is laid out; the actual call is gated
behind a `NotImplementedError` until Stage 1 produces a real `.pt` checkpoint.

Target usage:
    python export_onnx.py --weights ../models/cifar10_cnn.pt \
                          --out     ../models/cifar10_cnn.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train import SmallCNN


def export(weights_path: str, out_path: str, opset: int = 17) -> None:
    """Load weights into SmallCNN and trace-export to ONNX.

    TODO (Stage 2):
        model = SmallCNN()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        dummy = torch.randn(1, 3, 32, 32)          # NCHW, batch of 1
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )

    The `dynamic_axes` on dim 0 are what let the C++ side run batched inference
    later (Stage 5/6) without re-exporting the model.
    """
    raise NotImplementedError("Stage 2: implement torch.onnx.export")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch CNN to ONNX.")
    parser.add_argument("--weights", default="../models/cifar10_cnn.pt")
    parser.add_argument("--out", default="../models/cifar10_cnn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export(args.weights, args.out, args.opset)
    print(f"[export] wrote {args.out}")


if __name__ == "__main__":
    main()
