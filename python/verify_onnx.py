"""Numerical PyTorch/ONNX Runtime parity gate for the exported model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import CIFAR10_CLASSES, load_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]


def verify(
    weights_path: str,
    onnx_path: str,
    tol: float = 1.0e-4,
    *,
    seed: int = 1337,
    batch_size: int = 2,
    json_out: str | Path | None = None,
) -> bool:
    """Run deterministic identical inputs and enforce strict max-absolute error."""

    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_path = Path(onnx_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    import onnxruntime as ort

    model, _ = load_checkpoint(weights_path)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(
        batch_size, 3, 32, 32, generator=generator, dtype=torch.float32
    )
    with torch.no_grad():
        torch_logits = model(inputs).cpu().numpy()

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    if len(session.get_inputs()) != 1 or session.get_inputs()[0].name != "input":
        raise ValueError("ONNX model must expose exactly one input named 'input'")
    if len(session.get_outputs()) != 1 or session.get_outputs()[0].name != "logits":
        raise ValueError("ONNX model must expose exactly one output named 'logits'")
    onnx_logits = session.run(["logits"], {"input": inputs.numpy()})[0]
    if onnx_logits.shape != torch_logits.shape:
        raise ValueError(
            f"output shape mismatch: PyTorch {torch_logits.shape}, ONNX {onnx_logits.shape}"
        )
    if not np.isfinite(onnx_logits).all():
        raise ValueError("ONNX Runtime produced non-finite logits")

    max_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    torch_classes = np.argmax(torch_logits, axis=1)
    onnx_classes = np.argmax(onnx_logits, axis=1)
    class_match = bool(np.array_equal(torch_classes, onnx_classes))
    passed = max_diff < tol and class_match
    record = {
        "seed": seed,
        "batch_size": batch_size,
        "max_abs_diff": max_diff,
        "tolerance": tol,
        "class_match": class_match,
        "passed": passed,
    }
    if json_out is not None:
        destination = Path(json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"[verify] seed={seed} batch={batch_size} max_abs_diff={max_diff:.8g} "
        f"tolerance={tol:.8g} class_match={class_match} "
        f"result={'PASS' if passed else 'FAIL'}"
    )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PyTorch/ONNX parity.")
    parser.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "models" / "cifar10_cnn.pt"
    )
    parser.add_argument(
        "--onnx", type=Path, default=REPO_ROOT / "models" / "cifar10_cnn.onnx"
    )
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    ok = verify(
        str(args.weights),
        str(args.onnx),
        args.tol,
        seed=args.seed,
        batch_size=args.batch_size,
        json_out=args.json_out,
    )
    print(f"[verify] classes={len(CIFAR10_CLASSES)} parity={'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
