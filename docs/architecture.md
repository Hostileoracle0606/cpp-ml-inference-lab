# Architecture

This document describes how a model moves from a Python training script to a C++ runtime
serving predictions, and the contracts that keep the two sides consistent.

## Data flow

```text
 ┌─────────────────────┐
 │  python/train.py    │   Train a small CNN on CIFAR-10.
 │  (PyTorch)          │   Output: models/cifar10_cnn.pt  (weights + arch)
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │ python/export_onnx  │   torch.onnx.export(): trace the model into a
 │  .py                │   language-neutral computation graph.
 │                     │   Output: models/cifar10_cnn.onnx
 └──────────┬──────────┘
            │
            ▼
 ┌─────────────────────┐
 │ python/verify_onnx  │   Run the SAME input through PyTorch and ONNX
 │  .py                │   Runtime; assert max output diff < epsilon.
 └──────────┬──────────┘
            │   (parity gate — only a verified .onnx crosses into C++)
            ▼
 ┌─────────────────────┐
 │  C++ inference core │   ONNX Runtime loads the graph once, builds input
 │  src/ + include/    │   tensors, runs the session, decodes logits.
 └──────────┬──────────┘
            │
   ┌────────┼─────────────────────┐
   ▼        ▼                     ▼
 CLI      Benchmarks           HTTP server
 ./infer  p50/p95/throughput   /predict, /health
```

## Components and responsibilities

| Component | Lives in | Responsibility |
|-----------|----------|----------------|
| Training | `python/train.py` | Define the CNN, train on CIFAR-10, save `.pt`. |
| Export | `python/export_onnx.py` | Convert `.pt` → `.onnx` with correct input/output shapes. |
| Parity check | `python/verify_onnx.py` | Numerically compare PyTorch vs ONNX Runtime. |
| `InferenceEngine` | `include/`, `src/` | Own the ONNX Runtime session; `predict(Image) → Prediction`. |
| Preprocessing | `include/`, `src/` | Decode image → normalized CHW float tensor (must match training). |
| CLI | `src/main.cpp` | Parse args, call the engine, print label/confidence/latency. |
| Benchmarks | `benchmarks/` | Measure preprocessing vs model-execution vs end-to-end latency. |
| Server | `src/server.cpp` (Stage 7) | Expose `/predict` and `/health`. |

## Target core abstractions (Stage 4)

```cpp
struct Prediction {
    std::string label;
    float       confidence;
    double      latency_ms;
};

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path);  // load once (RAII)
    Prediction predict(const Image& image);                   // run many times
};
```

The engine loads the ONNX session **once** in its constructor and reuses it across calls —
session creation is expensive, per-request inference should not pay that cost. This is the
foundation that the Stage 6 optimization pass builds on (session reuse, buffer reuse, batching).

## The preprocessing-parity contract

The single most common source of "works in Python, wrong in C++" bugs is **train/serve skew**:
the C++ side preprocesses images differently from how the model was trained.

To run, both sides must agree on:

1. **Resize / crop** to the model's expected input size (32×32 for CIFAR-10).
2. **Channel order** — PyTorch uses CHW (channels-first); image libraries often give HWC.
3. **Dtype & scaling** — pixels `[0, 255]` uint8 → `[0, 1]` float.
4. **Normalization** — subtract per-channel mean, divide by per-channel std (the exact
   constants used in `train.py`).

`verify_onnx.py` checks model-graph parity; the C++ unit tests in Stage 4 check that C++
preprocessing reproduces the Python tensor for the same image. Both gates must pass.

## Design principles

- **Measure, don't guess** — every optimization in Stage 6 is justified by a benchmark delta.
- **Separation of concerns** — preprocessing, inference, and I/O are independent and testable.
- **RAII for resources** — the ONNX session lifetime is tied to the `InferenceEngine` object.
- **Portability is the point** — the `.onnx` artifact is the contract between Python and C++.
