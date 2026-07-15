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

V1.1 is a validated source release with a reproducible local CPU reference-model workflow. Its
reference evidence is scoped to macOS arm64, Python 3.9.6, the exact pinned ML stack, and the
recorded local machine. No trained checkpoint, ONNX graph, evidence bundle, or portable benchmark
result is published; reproduce the workflow locally. HTTP serving, Docker, quantization, and GPU
execution remain later work.

## What v1.1 demonstrates

- Deterministic, stratified 45,000/5,000 training/validation separation with validation-only model
  selection and one test evaluation per frozen run.
- Schema-v2 checkpoints binding selected weights to data, configuration, metrics, history, and
  environment while retaining raw/v1 checkpoint compatibility.
- Named ONNX endpoints (`input` and `logits`) with a dynamic batch dimension.
- A strict max-absolute-logit parity gate (`< 1e-4`) plus top-class agreement.
- A reusable C++ core with explicit values, preprocessing, decoding, an injected runtime boundary,
  and RAII session ownership.
- Dependency-free P3/P6 PPM decoding for the portable CLI path.
- Stable softmax, CIFAR-10 label decoding, validation, and actionable CLI failures.
- Warmed multi-sample benchmarks reporting mean, p50, p95, and throughput.
- Offline core/CLI/benchmark tests, generated hostile ONNX fixtures, and a real trained-model ORT
  end-to-end path.
- An atomic, checksummed local evidence bundle with strict command provenance, a dependency-light
  integrity/schema verifier, and a trusted-local deep semantic audit.

## Evidence captured for v1.1

| Gate | Result | Scope |
|---|---:|---|
| Frozen dataset protocol | 45,000 train / 5,000 validation / 10,000 test | Seed 1337, 4,500/500 per class, split digest `a9ca2a07…1755e` |
| Selected model quality | 81.28% validation; 79.90% test | Epoch 20, CPU, 20 epochs, pre-registered test floor 65% |
| Checkpoint / ONNX | `315b201b…e0ad23` / `fec6ac78…b255c` | Local ignored artifacts; hashes are SHA-256 |
| Trained export parity | `1.6689301e-06 < 1e-4` | Seed 1337, batch 2, matching classes |
| Trained C++ runtime path | 9/9 CTest entries passed | Release, official ONNX Runtime 1.19.2, macOS arm64 |
| Python pipeline | 18/18 tests passed | Protocol, compatibility, export/parity, adversarial bundle and deep-audit logic |
| Local evidence bundle | Lightweight and deep verification passed | Manifest SHA-256 `35d391d0…c4aad`; 9 hashed artifacts |
| Runtime-only benchmark | mean 0.5322 ms; p50 0.4089 ms; p95 1.2607 ms | Apple M4, macOS 26.4, AppleClang 21, ORT 1.19.2, 20 warm-ups + 200 iterations |
| Minimum CMake | 3/3 portable tests passed | CMake 3.16.8 x86_64 under Rosetta; not native Python/ORT evidence |
| GitHub Actions | Defined, not executed | Workflow intent only; no supported-platform claim |

The benchmark is one machine-scoped capture. Its in-memory boundary excludes file decode and
session construction, and its values are not a portable performance promise. The reference Python
stack is Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, ONNX 1.19.1, ONNX Runtime 1.19.2, NumPy
2.0.2, and Pillow 11.3.0; all 20 resolved packages are pinned in the platform-labelled constraints
file.

The exact benchmark command for the table was:

```bash
/tmp/cpp-ml-v11-r2-ort/inference_benchmark \
  --model models/cifar10_cnn-v1.1-r2.onnx --warmup 20 --iterations 200
```

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

See [the architecture document](docs/architecture.md), [v1.1 acceptance plan](docs/v1_1_plan.md),
[v1.2 measured-batching plan](docs/v1_2_plan.md), and
[living decision matrix](docs/decision_matrix.md) for the contracts and tradeoffs.

## Prerequisites

- CMake 3.16 or newer and a C++17 compiler.
- Python 3.9+ for training/export/tests.
- The packages in `python/requirements.txt`.
- An unpacked ONNX Runtime C/C++ release for the runtime-enabled C++ build. V1.1 was validated with
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

For the v1.1 reference protocol, first install the exact platform lock into a fresh Python 3.9.6
environment. The recorded r2 run then used this verbatim shell invocation:

```bash
set -o pipefail
mkdir -p artifacts/v1.1-run-r2
/tmp/cpp-ml-v1.1-ref-venv-20260714/bin/python python/train.py \
  --epochs 20 --batch-size 128 --lr 0.001 --data-dir python/data \
  --out models/cifar10_cnn-v1.1-r2.pt --num-workers 0 \
  --seed 1337 --split-seed 1337 --no-download \
  --record-dir artifacts/v1.1-run-r2/records --device cpu \
  2>&1 | tee artifacts/v1.1-run-r2/training-cpu.log
```

That path names the local reference environment used for the evidence run; substitute the Python
executable from your own clean environment for a new reproduction. Do not lower the accuracy floor
or change the frozen configuration after observing test results.

After assembling a local bundle with `python/evidence_manifest.py build`, run both verification
layers:

```bash
python python/evidence_manifest.py verify \
  --bundle artifacts/v1.1.0-local-evidence-r2-final

python python/deep_verify_evidence.py \
  --bundle artifacts/v1.1.0-local-evidence-r2-final
```

The first command checks files, hashes, schemas, cross-record links, and command provenance without
loading model binaries. The second is only for a trusted local bundle in the pinned ML environment;
it safely loads weights, compares internal checkpoint metadata, checks ONNX, and recomputes parity.

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

V1.1 is a single-model, single-caller CPU inference lab with a local 79.90% CIFAR-10 reference
result in its exact recorded environment. It makes no MPS, server, concurrency, GPU,
security-hardening, production image-decoder, artifact-publication, or portable-latency claim. The
pre-registered v1.2 experiment measures runtime-only batch economics without changing the
single-image CLI/pipeline. Later boundaries are listed in [docs/roadmap.md](docs/roadmap.md), and
decisions must be updated in [docs/decision_matrix.md](docs/decision_matrix.md) before they change.
