"""Stage 2 — Verify PyTorch and ONNX Runtime agree.

This is the *parity gate*: only a model whose ONNX output numerically matches
PyTorch should cross into the C++ side. Stage 0 skeleton; logic lands in Stage 2.

Target usage:
    python verify_onnx.py --weights ../models/cifar10_cnn.pt \
                          --onnx    ../models/cifar10_cnn.onnx

Target output:
    PyTorch prediction: cat, confidence 0.812
    ONNX    prediction: cat, confidence 0.812
    Max output diff:    0.000013   -> PASS (< 1e-4)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from train import CIFAR10_CLASSES, SmallCNN


def verify(weights_path: str, onnx_path: str, tol: float = 1e-4) -> bool:
    """Run identical input through both runtimes and compare.

    TODO (Stage 2):
        # PyTorch side
        model = SmallCNN(); model.load_state_dict(torch.load(weights_path)); model.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            torch_logits = model(x).numpy()

        # ONNX Runtime side
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_logits = sess.run(None, {"input": x.numpy()})[0]

        max_diff = float(np.abs(torch_logits - onnx_logits).max())
        return max_diff < tol
    """
    raise NotImplementedError("Stage 2: implement the PyTorch vs ONNX parity check")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PyTorch/ONNX parity.")
    parser.add_argument("--weights", default="../models/cifar10_cnn.pt")
    parser.add_argument("--onnx", default="../models/cifar10_cnn.onnx")
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    ok = verify(args.weights, args.onnx, args.tol)
    print(f"[verify] classes={len(CIFAR10_CLASSES)}  parity={'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
