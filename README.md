# C++ ML Inference Lab

A complete, deliberately small model-deployment laboratory: train a CIFAR-10 CNN in PyTorch,
export a dynamic-batch ONNX graph, prove PyTorch/ONNX numerical parity, and run the graph through a
modular C++17 inference pipeline with ONNX Runtime.

The project concentrates on the boundary after model development—artifact contracts,
train/serve preprocessing consistency, runtime resource ownership, failure behavior, and honest
latency measurement.

```text
PyTorch checkpoint -> ONNX export/parity gate -> C++ core -> CLI / benchmark
```

V1 is implemented and validated as a source release. It does not include a trained model binary or
claim CIFAR-10 accuracy; generate the artifact locally with the training command below. HTTP
serving, Docker, quantization, and GPU execution are post-v1 work.

## What v1 demonstrates

- Deterministic PyTorch training/evaluation helpers and metadata-bearing checkpoints.
- Named ONNX endpoints (`input` and `logits`) with a dynamic batch dimension.
- A strict max-absolute-logit parity gate (`< 1e-4`) plus top-class agreement.
- A reusable C++ core with explicit values, preprocessing, decoding, an injected runtime boundary,
  and RAII session ownership.
- Dependency-free P3/P6 PPM decoding for the portable CLI path.
- Stable softmax, CIFAR-10 label decoding, validation, and actionable CLI failures.
- Warmed multi-sample benchmarks reporting mean, p50, p95, and throughput.
- Offline core/CLI/benchmark tests and an optional real-ONNX-Runtime end-to-end test.

## Evidence captured for v1

| Gate | Result | Scope |
|---|---:|---|
| Clean default Release build | 4/4 CTest entries passed | Core, CLI contract, benchmark contract, Python pipeline |
| Python pipeline | 7/7 tests passed | Synthetic training/evaluation, cross-language contract, export, dynamic batch, parity |
| Export parity | `2.9802322e-08 < 1e-4` | Seed 1337, batch 2, random test checkpoint, class match |
| Real C++ runtime path | 5/5 CTest entries passed | Official ONNX Runtime 1.19.2, macOS arm64, including CLI/benchmark e2e |
| Full CIFAR-10 training/accuracy | Not measured | Intentionally no accuracy claim or committed model artifact |

The parity and runtime checks use a random-weight test checkpoint, which validates the deployment
pipeline but says nothing about classifier quality. Benchmark absolute values are likewise not
published as portable performance claims.

The Python gate was captured with Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, ONNX 1.19.1,
ONNX Runtime 1.19.2, NumPy 2.0.2, and Pillow 11.3.0.

## Architecture

```text
python/model.py + train.py
          |
          v
 metadata checkpoint (.pt) -> export_onnx.py -> model.onnx
                                      |
                                      v
                               verify_onnx.py
                                      |
                           max diff + class gate
                                      v
 CLI -> InferencePipeline -> preprocess -> InferenceEngine -> IInferenceBackend
          |                                        |
          |                                        v
          +-> PPM loader + softmax          OnnxRuntimeBackend
```

`IInferenceBackend` is the only polymorphic seam. Domain data remains value-oriented, while
`InferencePipeline` composes concrete loading, preprocessing, execution, and decoding modules.
The ONNX adapter owns one environment/session and reuses it across predictions.

See [the architecture document](docs/architecture.md), [v1 acceptance plan](docs/v1_plan.md), and
[living decision matrix](docs/decision_matrix.md) for the contracts and tradeoffs.

## Prerequisites

- CMake 3.16 or newer and a C++17 compiler.
- Python 3.9+ for training/export/tests.
- The packages in `python/requirements.txt`.
- An unpacked ONNX Runtime C/C++ release for the runtime-enabled C++ build. V1 was validated with
  the official 1.19.2 macOS arm64 release.

## Quickstart from the repository root

### 1. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

### 2. Run the portable build and tests

This path requires neither a model file nor the C++ ONNX Runtime SDK:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The benchmark can exercise preprocessing without a runtime:

```bash
./build/inference_benchmark --warmup 10 --iterations 100
```

### 3. Train, export, and verify

Training downloads CIFAR-10 unless `--no-download` is given. Start with one epoch to validate the
workflow; meaningful accuracy normally requires a longer run.

```bash
python python/train.py --epochs 1 --device cpu
python python/export_onnx.py
python python/verify_onnx.py
```

Default outputs are `models/cifar10_cnn.pt` and `models/cifar10_cnn.onnx`. Model and dataset
artifacts are ignored by Git.

### 4. Build the C++ runtime path

Download and unpack an ONNX Runtime C/C++ release, then provide its absolute root:

```bash
cmake -S . -B build-ort -DCMAKE_BUILD_TYPE=Release \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-ort --parallel
```

### 5. Prepare a PPM input and predict

The v1 decoder intentionally accepts only P3/P6 RGB PPM. Convert a PNG/JPEG with Pillow:

```bash
python -c "from PIL import Image; Image.open('input.png').convert('RGB').save('samples/input.ppm')"
./build-ort/infer \
  --model models/cifar10_cnn.onnx \
  --image samples/input.ppm
```

Output has this shape; the values depend on the trained artifact and machine:

```text
Prediction: cat
Class index: 3
Confidence: 0.812
Latency:    5.800 ms
  preprocess: 0.041 ms
  inference:  5.521 ms
```

The loader resizes inputs to 32x32 with bilinear interpolation, converts interleaved RGB bytes to
NCHW float32, and applies the frozen CIFAR-10 normalization constants.

### 6. Benchmark and opt into runtime e2e CTest

```bash
./build-ort/inference_benchmark \
  --model models/cifar10_cnn.onnx \
  --warmup 10 --iterations 100

cmake -S . -B build-ort-test -DCMAKE_BUILD_TYPE=Release \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime \
  -DCPP_ML_TEST_MODEL="$PWD/models/cifar10_cnn.onnx" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-ort-test --parallel
ctest --test-dir build-ort-test --output-on-failure
```

Benchmark boundaries are:

- `preprocessing`: in-memory RGB `Image` to normalized `Tensor`.
- `runtime_only`: validated tensor through the reused backend/session.
- `end_to_end`: in-memory image through preprocessing, runtime, and decoding; file I/O is excluded.

Nearest-rank percentiles are computed after warm-up samples are discarded. The report prints build
type, system/processor, compiler/version, C++ language level, warm-up count, and measured iteration
count.

## Repository layout

```text
benchmarks/inference_benchmark.cpp  warmed latency/throughput harness
docs/                              architecture, v1 plan, roadmap, decisions
include/cpp_ml/                    public C++ core interfaces and values
python/model.py                    shared model/checkpoint contract
python/train.py                    deterministic CIFAR-10 training entry point
python/export_onnx.py              named dynamic-batch ONNX export
python/verify_onnx.py              numerical parity gate
src/                               core implementation and CLI adapter
tests/                             C++, CLI, benchmark, Python, and optional ORT checks
models/                            generated checkpoints/graphs (ignored)
samples/                           user-provided PPM inputs
```

## Current boundary

V1 is a single-model, single-caller CPU inference lab. It deliberately makes no server,
concurrency, GPU, security-hardening, or production image-decoder claim. The next steps are listed
in [docs/roadmap.md](docs/roadmap.md); decisions must be updated in
[docs/decision_matrix.md](docs/decision_matrix.md) before the boundary changes.
