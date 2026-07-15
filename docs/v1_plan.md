# Version 1 plan

## Release status

This plan was created from the Stage 0 baseline described below. C2 completed on 2026-07-14: v1 is
ready as a source release, with a clean default build, seven Python pipeline tests, a real official
ONNX Runtime C++ path, and documented benchmark boundaries. The release intentionally contains no
trained checkpoint or accuracy claim; model production remains a user-run workflow.

## Product statement

Version 1 is a small but complete model-deployment laboratory: train a CIFAR-10 classifier in
PyTorch, export a portable ONNX graph, prove numerical parity, and consume that graph through a
modular C++17 command-line application. It should teach the boundaries that matter in production
inference—artifact contracts, preprocessing parity, runtime lifetime, error handling, testing, and
measurement—without pretending to be a production serving platform.

At the initial C0 audit the repository contained a sound Stage 0 outline, but every Python pipeline
operation was a `NotImplementedError`, the C++ binary was a single-file placeholder, and no tests or
benchmarks existed. A successful v1 therefore meant an executable vertical slice, not merely more
scaffolding. C1 and C2 evidence in `docs/decision_matrix.md` records how that baseline was closed.

## Scope boundary

### Required in v1

1. A deterministic, configurable Python training path that writes a loadable checkpoint.
2. ONNX export with named inputs/outputs and a dynamic batch dimension.
3. A parity gate that compares the same input in PyTorch and ONNX Runtime and fails when the
   maximum absolute logit difference is not below `1e-4`.
4. A reusable C++ inference core separated from the CLI, including image decoding,
   preprocessing, runtime execution, and logit decoding.
5. A CLI that validates arguments, loads one model session, predicts one image, and reports the
   label, confidence, and latency with actionable errors.
6. Deterministic tests for the dependency-free C++ modules, plus Python unit/smoke tests for the
   train/export/parity helpers.
7. A benchmark executable that distinguishes warm-up from measured runs and reports at least
   p50, p95, mean latency, and throughput.
8. Reproducible setup and usage documentation, including the exact model/input contract and the
   distinction between tests that do and do not require ONNX Runtime/model artifacts.

### Deferred to v1.x or v2

- HTTP serving, request concurrency, authentication, rate limiting, and deployment orchestration.
- Docker images and GPU execution providers.
- Quantization, buffer pools, multi-model routing, and broad optimization work.
- A general plug-in framework or configuration language for arbitrary model families.
- Claims about CIFAR-10 accuracy or performance without captured, reproducible evidence.

Deferring these items keeps v1 centered on one trustworthy vertical slice. The reusable core and
backend boundary leave room for a service adapter later without making server concerns part of the
first architecture.

## User-visible journeys

The final command spelling may vary slightly with packaging, but v1 must support these journeys:

```bash
# Dependency-free core build and tests
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure

# Model production and parity gate (from repository root)
python python/train.py --epochs 1 --out models/cifar10_cnn.pt
python python/export_onnx.py --weights models/cifar10_cnn.pt \
  --out models/cifar10_cnn.onnx
python python/verify_onnx.py --weights models/cifar10_cnn.pt \
  --onnx models/cifar10_cnn.onnx

# Runtime-enabled build and one prediction
cmake -S . -B build-ort -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime
cmake --build build-ort
./build-ort/infer --model models/cifar10_cnn.onnx --image samples/input.ppm
```

Long training, CIFAR-10 download, and runtime-enabled integration checks may be opt-in. The normal
build and pure unit suite must not require network access, a trained model, or ONNX Runtime.

## Architecture and dependency direction

```text
CLI adapter
   |
   v
InferencePipeline (application facade / composition root)
   |--------------------|---------------------|
   v                    v                     v
FileImageLoader   Cifar10Preprocessor   SoftmaxDecoder
                                              ^
                                              |
                                      InferenceEngine
                                              |
                                              v
                                     IInferenceBackend
                                              ^
                                              |
                                    OnnxRuntimeBackend
```

Dependencies point inward: the command-line adapter may depend on the inference core, but the
core never knows about command-line parsing or terminal formatting. Only the unstable external
runtime receives an abstraction boundary. Pure calculations remain concrete, small modules rather
than acquiring interfaces solely to look object-oriented.

Suggested source responsibilities:

| Module | Responsibility | Must not do |
|---|---|---|
| Value types (`Image`, `Tensor`, `Prediction`) | Own validated data and shapes | Load files or call ONNX Runtime |
| `FileImageLoader` | Decode P3/P6 PPM files to RGB8 HWC | Normalize or classify |
| `Cifar10Preprocessor` | Resize/validate and produce normalized NCHW floats | Read CLI flags or run a model |
| `IInferenceBackend` | Minimal seam for `Tensor -> logits` | Preprocess images or choose labels |
| `OnnxRuntimeBackend` | Own `Ort::Env`/session resources and execute the graph | Print output or recreate a session per call |
| `SoftmaxDecoder` | Stable softmax/argmax and class lookup | Depend on ONNX Runtime |
| `InferenceEngine` | Move-only owner/executor for one injected backend | Decode files or choose labels |
| `InferencePipeline` | Compose loader, transform, engine, timing, and decoding | Parse arguments or own presentation policy |
| CLI | Parse/validate flags, map errors to exit codes, format results | Contain model or tensor algorithms |
| Benchmark | Warm up, sample, summarize, and identify the measured boundary | Report unlabelled or single-sample latency |

### Core contracts

- Decoded image: RGB, unsigned 8-bit, interleaved HWC.
- Model input: contiguous `float32`, shape `[N, 3, 32, 32]`, channel-first.
- Scaling: `pixel / 255.0` before normalization.
- Normalization mean: `(0.4914, 0.4822, 0.4465)`.
- Normalization standard deviation: `(0.2470, 0.2435, 0.2616)`.
- Model input/output names: `input` and `logits`.
- Output: shape `[N, 10]`, ordered as `airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck`.
- Confidence: stable softmax of logits, not the winning raw logit.
- Latency fields and benchmark labels must say whether they measure runtime-only or end-to-end.

The CIFAR-10 constants are intentionally explicit on both sides of the language boundary. A direct
shape/mean/std declaration guard detects drift, while C++ numerical tests cover indexing and
normalization behavior. External model metadata becomes worthwhile only when a second model
contract is actually introduced.

## OOP and SOLID analysis

| Principle | v1 application | Guardrail |
|---|---|---|
| Single responsibility | Decode, transform, execute, decode logits, and present results live in distinct modules. | Do not let `InferenceEngine` absorb file parsing or output formatting. |
| Open/closed | A later backend can implement the narrow inference interface without changing pipeline policy. | Do not build a registry or plug-in loader before a second backend exists. |
| Liskov substitution | Backend implementations accept and return the same validated tensor contracts and error semantics. | Tests use a deterministic fake backend; callers do not inspect concrete runtime types. |
| Interface segregation | The backend interface exposes only inference behavior needed by the engine. | No broad `IModel` with unrelated load, benchmark, logging, and server methods. |
| Dependency inversion | High-level orchestration depends on an inference capability, not ONNX Runtime headers. | Runtime headers remain in the adapter implementation and runtime-enabled target. |
| RAII | Runtime environment, session, and image buffers have deterministic ownership. | No owning raw pointers; no session construction inside `predict`. |
| Composition over inheritance | The engine composes decoder, preprocessor, backend, and prediction logic. | Inheritance is used only where substituting a backend provides a real test or extension seam. |

This is deliberately restrained OOP. Data is represented as value types, and stateless numerical
operations may be free functions. That keeps invariants visible and prevents an inheritance tree
from obscuring the inference pipeline.

## Dependency strategy

- **ONNX Runtime:** optional at configure time and supplied through `ONNXRUNTIME_ROOT` (or a
  discovered system installation). Expose it to as few targets as possible. The official C++ API
  is a header wrapper over the C API and already applies exception/RAII semantics, so the adapter
  should preserve rather than duplicate those lifetime rules.
- **Image decoding:** v1 supports dependency-free P3/P6 PPM with a conservative decoded-image
  limit. PNG/JPEG support is deferred. If later required, prefer a pinned decoder with its license
  and provenance over a mutable configure-time download.
- **Tests:** use CTest with a small local assertion harness for v1 unless a test framework is
  already available. A framework fetched from the network is not necessary for this code volume.
- **Python:** declare minimum compatible versions in `requirements.txt`, record tested versions,
  seed pseudorandom generators, and always load checkpoints on CPU for export/parity. A fully
  locked environment is deferred to the reproducible-model evidence release.
- **Artifacts:** generated datasets, weights, and ONNX graphs remain outside source control unless
  an explicit release artifact policy is added. Tests use tiny committed fixtures, not full model
  binaries.

Official references used to check these choices:

- [ONNX Runtime C++ getting started](https://onnxruntime.ai/docs/get-started/with-cpp.html)
- [ONNX Runtime C/C++ API and RAII wrapper](https://onnxruntime.ai/docs/api/c/c_cpp_api.html)
- [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html)
- [Netpbm PPM format specification](https://netpbm.sourceforge.net/doc/ppm.html)

## Verification and acceptance gates

### Gate A — default build integrity

- A clean `cmake -S . -B build -DBUILD_TESTS=ON`, build, and CTest run succeed without ORT.
- Warnings are enabled portably for project-owned targets.
- The CLI help path succeeds; unknown flags, missing values, missing required flags, nonexistent
  files, bad images, and runtime load failures return nonzero with a concise diagnostic.

### Gate B — Python pipeline

- Data transforms use the documented normalization constants.
- Training/evaluation helpers have deterministic smoke coverage without downloading a dataset.
- Synthetic production-helper tests update weights/evaluate correctly, and a temporary checkpoint
  can be loaded and exported on CPU without a dataset download. The full CIFAR entry point remains
  user-run because v1 publishes no trained artifact or accuracy claim.
- Export creates an ONNX model with `input`/`logits` names and dynamic batch dimension.
- Parity uses identical seeded input and exits nonzero at or above the tolerance; the observed
  maximum difference is printed.

### Gate C — core correctness

- Tests cover HWC-to-CHW indexing, scaling/normalization at boundary pixel values, channel order,
  invalid dimensions/channels, stable softmax for large logits, argmax, and label bounds.
- A cross-language guard compares Python and C++ input dimensions and normalization constants;
  C++ numerical tests independently cover HWC-to-CHW and boundary normalization values.
- A fake backend proves orchestration and session reuse without ORT.
- Runtime-enabled integration validates model metadata (rank, dimensions, dtype, names) before
  inference and rejects an incompatible graph clearly.

### Gate D — measurement honesty

- Benchmark builds independently from the CLI and uses a monotonic clock.
- Warm-up samples are excluded and iteration count is reported.
- Runtime-only and end-to-end results are separate and labelled.
- p50 and p95 use a documented percentile rule; throughput is derived from the same measured
  interval. Environment/build type is printed with results.

### Gate E — documentation and release

- README commands match executable behavior from repository root.
- No roadmap checkbox or performance/accuracy number is marked complete without a command or
  captured result that reproduces it.
- The decision matrix records accepted decisions, consequences, and revisit triggers.
- No placeholder `NotImplementedError`, Stage 0 output, or commented future implementation remains
  on a required v1 path.

## Modular delivery sequence

| Slice | Vertical outcome | Exit evidence |
|---:|---|---|
| 1 | Pure C++ value/preprocessing/decoding modules | Default build and deterministic unit tests pass |
| 2 | Python checkpoint, export, and parity path | Smoke tests plus a generated graph passing parity |
| 3 | ONNX adapter and engine composition | Runtime integration predicts via a reused session |
| 4 | CLI behavior and failure paths | CLI integration matrix passes with stable exit codes |
| 5 | Benchmark and documentation reconciliation | Labelled metrics and all documented commands verified |

Each slice ends with an introspection update in `docs/decision_matrix.md`: compare the code to the
contracts, note evidence, and either affirm or supersede decisions. This event-based cadence is more
useful than introspection at arbitrary clock intervals.

## Risks and mitigations

| Risk | Impact | Mitigation / release rule |
|---|---|---|
| Training cost or dataset network access | Slow/non-reproducible checks | Synthetic smoke tests are mandatory; full training remains explicit and opt-in. |
| Python/C++ preprocessing skew | Plausible but incorrect predictions | Cross-language shape/constant drift guard plus concrete C++ indexing/normalization tests gate release. |
| ORT install differences | Runtime build friction | Optional isolated target, clear root-path configuration, tested version recorded. |
| Image decoder attack surface | Unsafe handling of arbitrary files | Limit v1 to P3/P6 PPM, cap dimensions before allocation, validate sizes, and state the non-production scope. |
| Over-engineering for future server support | More code and weaker learning signal | Add abstractions only at an exercised volatility/test seam. |
| Misleading benchmarks | Invalid portfolio claims | Release build, warm-up, multiple samples, measurement boundary and environment disclosed. |
| Concurrent edits across modules | Contract drift or build breakage | Land vertical slices, keep ownership boundaries, reconcile all docs only after tests. |

## Definition of v1-ready

V1 is ready only when Gates A–E pass, the required user journeys have been rerun from a clean build,
and the final introspection checkpoint records evidence rather than intention. C2 meets this rule
with a temporary random checkpoint and an official local ORT release: those artifacts prove export,
parity, and runtime behavior but are explicitly not classifier-quality evidence. A future accuracy
release must add a reproducible trained checkpoint rather than reinterpret the v1 evidence.
