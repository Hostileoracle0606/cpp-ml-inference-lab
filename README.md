# C++ ML Inference Lab

[![Verification CI](https://github.com/Hostileoracle0606/cpp-ml-inference-lab/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/Hostileoracle0606/cpp-ml-inference-lab/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)

A deliberately small, end-to-end lab for moving a CIFAR-10 model from PyTorch training to verified
ONNX export and modular C++17 inference with ONNX Runtime.

**Status:** research and learning project. `main` contains the validated v1.1 source state declared
by CMake as version `1.1.0`. Trained weights, ONNX graphs, and evidence bundles are generated locally
and are not distributed by this repository.

| Signal | Current evidence | Boundary |
|---|---|---|
| Default-branch verification | Seven-job [GitHub Actions workflow](https://github.com/Hostileoracle0606/cpp-ml-inference-lab/actions/workflows/ci.yml) | Scoped build/test evidence, not a platform-support guarantee |
| Portable C++ path | GCC, Clang, AppleClang, MSVC, CMake 3.16, and ASan/UBSan jobs | No model or ONNX Runtime SDK required |
| Full integration path | Python 3.11 + pinned CI dependencies + ONNX Runtime 1.19.2 on Linux | Synthetic hostile graphs and runtime integration, not model-quality evidence |
| Maintainer-local reference run | 79.90% CIFAR-10 test accuracy; max PyTorch/ORT logit error `1.6689301e-06` | One frozen macOS arm64/Python 3.9.6 environment; artifacts are unpublished |
| Production readiness | Not claimed | No server, concurrency contract, GPU path, security hardening, or production image decoder |

## Start here: verify the portable C++ core

This is the fastest useful proof. It needs only CMake 3.16+ and a C++17 compiler—no Python
packages, model file, or ONNX Runtime SDK.

```bash
cmake -S . -B build/portable \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build/portable --parallel
ctest --test-dir build/portable --output-on-failure -L portable
```

The portable label currently covers the C++ unit suite plus CLI and benchmark contracts. You can
also run the preprocessing-only benchmark without a model:

```bash
./build/portable/inference_benchmark --warmup 10 --iterations 100
```

## Why this repository exists

Many ML deployment failures happen between a correct model and its production caller. This lab
makes that boundary visible and testable:

- training and serving share an explicit preprocessing contract;
- named ONNX inputs/outputs and shape checks fail early;
- PyTorch and ONNX Runtime outputs must pass a numerical parity gate;
- runtime ownership uses RAII rather than manual lifecycle management;
- the application pipeline composes focused components around one injected runtime interface;
- benchmarks state exactly what they include and exclude;
- performance changes are accepted or rolled back against pre-registered thresholds.

```text
PyTorch training
      |
      v
metadata checkpoint -> ONNX export -> parity gate
                                        |
                                        v
PPM -> preprocess -> C++ runtime -> decode -> CLI / benchmark
                         |
                         v
                  ONNX Runtime adapter
```

## Architecture and object-oriented design

The code uses object orientation at the external-runtime boundary, not for every noun:

- `IInferenceBackend` is the single polymorphic seam. Tests inject deterministic backends; the
  production adapter owns ONNX Runtime resources.
- `InferencePipeline` uses composition to coordinate loading, preprocessing, execution, and
  decoding without inheriting from them.
- Images, tensors, outputs, and predictions are validated value types with explicit invariants.
- The ONNX adapter owns one environment/session and reuses it across predictions through RAII.
- The CLI and benchmark depend on the reusable `cpp_ml_core` library rather than duplicating the
  inference workflow.

See [architecture](docs/architecture.md) for component responsibilities and
[the decision matrix](docs/decision_matrix.md) for alternatives, tradeoffs, and revisit triggers.

## Verification model

Credibility depends on keeping different kinds of evidence separate.

### Public and repeatable in GitHub Actions

The workflow checks:

- Release builds and portable tests with Linux GCC, Linux Clang, macOS AppleClang, and Windows
  MSVC;
- compatibility with the declared CMake 3.16 minimum in a bounded Ubuntu 20.04 container;
- portable tests under Clang AddressSanitizer and UndefinedBehaviorSanitizer;
- the full Python and ONNX Runtime path on Linux, including generated valid and hostile ONNX
  fixtures.

The badge reports the latest `main` push workflow. A green badge means those scoped jobs passed for
that revision; it does not certify every compiler, OS, Python version, model, or deployment target.

### Maintainer-local reference evidence

The frozen v1.1 run recorded:

| Gate | Result | Scope |
|---|---:|---|
| Dataset protocol | 45,000 train / 5,000 validation / 10,000 test | Seed 1337; stratified 4,500/500 split per class |
| Selected model | 81.28% validation; 79.90% test | CPU, 20 epochs; pre-registered test floor 65% |
| PyTorch/ONNX parity | `1.6689301e-06 < 1e-4` | Batch 2 with matching top classes |
| Trained C++ runtime path | 9/9 CTest entries passed | AppleClang + official ONNX Runtime 1.19.2, macOS arm64 |
| Python protocol suite | 18/18 tests passed | Training, checkpoint compatibility, export, parity, and evidence logic |
| Local evidence bundle | Lightweight and deep verification passed | Nine checksummed artifacts; bundle remains local |
| Runtime-only benchmark | mean 0.5322 ms; p50 0.4089 ms; p95 1.2607 ms | Apple M4/macOS 26.4; 20 warm-ups + 200 iterations |

These numbers describe one recorded environment, not portable model quality or latency. The model,
raw logs, and bundle are ignored local artifacts, so the table is a transparent maintainer-local
report rather than independently downloadable evidence. The exact protocol, commands, hashes, and
acceptance gates are retained in the [v1.1 plan](docs/v1_1_plan.md) and
[decision record](docs/decision_matrix.md).

### A rejected optimization is part of the record

The v1.2 batch-eight candidate improved median throughput by 12.1093%, but the decision was frozen
at a required 50% improvement. The candidate commit
[`c4c4b1e`](https://github.com/Hostileoracle0606/cpp-ml-inference-lab/commit/c4c4b1ee7286c01178873c3f5efbdcb88b5abc69)
was therefore reverted by
[`ab6ca61`](https://github.com/Hostileoracle0606/cpp-ml-inference-lab/commit/ab6ca61d0823714e30d3ab2b7416f09f36b620b3).
Current runtime behavior remains v1.1. The [v1.2 plan](docs/v1_2_plan.md) documents the hypothesis,
measurement recipe, result, and rollback rationale.

## Full PyTorch to ONNX to C++ workflow

### Compatibility scopes

- **C++ core:** CMake 3.16+ and C++17. The current workflow exercises the compiler/OS matrix listed
  above.
- **Python reference evidence:** Python 3.9.6 on macOS arm64 with the exact
  [v1.1 constraints](python/constraints-v1.1-macos-arm64-python3.9.6.txt).
- **Python CI:** Python 3.11 on Linux with the CI constraints.
- **C++ runtime adapter:** an unpacked ONNX Runtime C/C++ release. V1.1 used version 1.19.2.

Other versions may work, but they are not implied by these evidence scopes.

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

### 2. Exercise training, export, and parity

The one-epoch command is a workflow smoke test, not the reference-quality protocol. Training
downloads CIFAR-10 unless `--no-download` is supplied.

```bash
python python/train.py --epochs 1 --device cpu
python python/export_onnx.py
python python/verify_onnx.py
```

Default outputs are `models/cifar10_cnn.pt` and `models/cifar10_cnn.onnx`; both are ignored by Git.
For a reference reproduction, use the frozen commands and gates in
[docs/v1_1_plan.md](docs/v1_1_plan.md) rather than changing parameters after observing results.

### 3. Build with ONNX Runtime

Download and unpack an official
[ONNX Runtime C/C++ release](https://github.com/microsoft/onnxruntime/releases), then provide its
absolute root:

```bash
cmake -S . -B build/ort -DCMAKE_BUILD_TYPE=Release \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/ort --parallel
```

### 4. Convert an image and predict

The v1 decoder intentionally accepts only P3/P6 RGB PPM.

```bash
python -c "from PIL import Image; Image.open('input.png').convert('RGB').save('samples/input.ppm')"
./build/ort/infer \
  --model models/cifar10_cnn.onnx \
  --image samples/input.ppm
```

The loader resizes to 32x32 with bilinear interpolation, converts interleaved RGB bytes to NCHW
float32, and applies the frozen CIFAR-10 normalization constants.

### 5. Benchmark and run runtime integration tests

```bash
./build/ort/inference_benchmark \
  --model models/cifar10_cnn.onnx \
  --warmup 10 --iterations 100

cmake -S . -B build/ort-test -DCMAKE_BUILD_TYPE=Release \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime \
  -DCPP_ML_TEST_MODEL="$PWD/models/cifar10_cnn.onnx" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/ort-test --parallel
ctest --test-dir build/ort-test --output-on-failure
```

Benchmark phases are `preprocessing`, `runtime_only`, and in-memory `end_to_end`; file I/O and
session construction are excluded. Warm-up observations are discarded before mean, nearest-rank
p50/p95, and throughput are reported.

## Repository map

```text
benchmarks/            warmed latency and throughput harness
docs/                  architecture, version plans, roadmap, and decisions
include/cpp_ml/        public C++ interfaces and validated values
python/                model, training, export, parity, and evidence tools
src/                   core implementation and CLI adapter
tests/                 C++, CLI, benchmark, Python, and optional ORT checks
models/                generated checkpoints and ONNX graphs (ignored)
samples/               PPM inputs
```

Key documents:

- [Architecture and component contracts](docs/architecture.md)
- [V1 acceptance plan](docs/v1_plan.md)
- [V1.1 evidence protocol](docs/v1_1_plan.md)
- [V1.2 measured experiment](docs/v1_2_plan.md)
- [Roadmap](docs/roadmap.md)
- [Living decision matrix](docs/decision_matrix.md)

## Scope and non-goals

The current project is a single-model, single-caller CPU inference lab. It does not claim:

- a production server, concurrency/backpressure contract, or deployment SLA;
- GPU, CUDA, MPS, quantization, Docker, or mobile support;
- a production-grade image decoder or arbitrary input-format handling;
- security hardening or safe processing of untrusted model bundles;
- published checkpoints, ONNX graphs, evidence archives, or portable latency;
- broad support for platforms beyond the explicitly exercised environments.

Future work is tracked in the [roadmap](docs/roadmap.md). Material scope or architecture changes
must first record their alternatives, rationale, and acceptance gate in the
[decision matrix](docs/decision_matrix.md).

## Help, contributions, and maintenance

- For reproducible bugs, questions, or focused proposals, open a
  [GitHub issue](https://github.com/Hostileoracle0606/cpp-ml-inference-lab/issues). Do not include
  secrets or sensitive vulnerability details in a public issue.
- Before a large change, describe the consumer and acceptance criteria. Pull requests should keep
  the portable path offline, update tests and decisions together, and pass the complete workflow.
- Maintainer: [@Hostileoracle0606](https://github.com/Hostileoracle0606). No response-time or support
  SLA is promised.

## License

No license file has been published. Public visibility permits viewing and forking under GitHub's
terms, but no project license grants broader permission to reproduce, modify, or distribute the
code. A future license must be selected explicitly by the repository owner.
