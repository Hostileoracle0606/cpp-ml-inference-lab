# Architecture

V1 is a source-to-runtime vertical slice for one explicit model contract. It separates model
production in Python from portable inference in C++, with ONNX as the artifact boundary.

## End-to-end data flow

```text
python/model.py
   | shared CNN, labels, normalization, checkpoint loader
   v
python/train.py ------------------------------> models/cifar10_cnn.pt
                                                    |
                                                    v
python/export_onnx.py ------------------------> models/cifar10_cnn.onnx
                                                    |
                          +-------------------------+----------------------+
                          |                                                |
                          v                                                v
                python/verify_onnx.py                              C++ ONNX adapter
                PyTorch vs ORT logits                          metadata validation + run
                          |                                                |
                          +---------------- parity gate -------------------+
                                                                           v
                                                                CLI and benchmark adapters
```

The parity gate uses an identical seeded batch in PyTorch and Python ONNX Runtime. A graph passes
only when maximum absolute logit difference is below `1e-4` and every top class matches.

## V1.1 evidence architecture

```text
stratified 45k/5k split -> validation-only selection -> one test evaluation per run
          -> schema-v2 checkpoint -> ONNX export/parity -> ignored local bundle
                                                  |-> lightweight integrity/schema verifier
                                                  `-> pinned-environment deep semantic audit
```

Checkpoint schema v2 binds the selected weights to the split digest, training configuration,
validation/test metrics, exact environment, and epoch history. The loader still accepts v1
metadata checkpoints and legacy raw state dictionaries; unknown future schemas fail closed.

The lightweight bundle verifier uses the Python standard library and never deserializes a model.
It checks relative paths, regular-file ownership, exact artifact sets, sizes, SHA-256 hashes,
frozen evidence schemas, cross-record equality, model-hash references, and command provenance. The
separate deep audit is only for a trusted local bundle in the pinned ML environment. It uses
weights-only checkpoint loading, compares internal checkpoint metadata to the manifest, runs the
ONNX checker, and recomputes PyTorch/ONNX Runtime parity.

These additions are Python protocol/value modules. V1.1 introduces no new C++ polymorphic seam;
the same session-owning backend and composed single-image pipeline consume the trained graph.

## C++ dependency direction

```text
src/main.cpp (CLI)
        |
        v
InferencePipeline (application facade)
   |---------------------|--------------------|
   v                     v                    v
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

The CLI depends on `cpp_ml_core`; the core never depends on CLI parsing or terminal formatting.
ONNX Runtime headers are private to the runtime adapter implementation.

## Components and ownership

| Component | Responsibility | Ownership/lifetime |
|---|---|---|
| `Image` | Validated RGB8 HWC buffer and dimensions | Ordinary value |
| `Tensor` | Validated float buffer and shape | Ordinary value |
| `Prediction` | Class, probabilities, and labelled timings | Ordinary value |
| `FileImageLoader` | Decode P3/P6 PPM and enforce 16 MP safety cap | Stateless concrete adapter |
| `Cifar10Preprocessor` | Bilinear resize, HWC-to-CHW, scale, normalize | Stateless concrete policy |
| `SoftmaxDecoder` | Stable softmax, argmax, class lookup | Concrete value/policy |
| `IInferenceBackend` | Minimal `Tensor -> ModelOutput` runtime seam | Polymorphic test/adapter boundary |
| `OnnxRuntimeBackend` | Validate graph metadata and execute one ORT session | Owns `Ort::Env`, options, and session |
| `InferenceEngine` | Move-only owner/executor for an injected backend | Owns one backend via `unique_ptr` |
| `InferencePipeline` | Compose load/preprocess/run/decode and timings | Owns concrete policies plus engine |
| `infer` | Validate flags, map exceptions to exit codes, format prediction | Process adapter |
| `inference_benchmark` | Warm up, measure, summarize named boundaries | Process adapter |

## Frozen model and preprocessing contract

| Field | Contract |
|---|---|
| File input | One P3 or P6 PPM image, treated as RGB sample triplets |
| Decoded representation | Unsigned 8-bit interleaved HWC |
| Resize | Bilinear to 32x32 when necessary |
| Model tensor | Contiguous float32 NCHW `[N, 3, 32, 32]` |
| Pixel scaling | `value / 255.0` |
| Mean | `(0.4914, 0.4822, 0.4465)` |
| Standard deviation | `(0.2470, 0.2435, 0.2616)` |
| ONNX input | Exactly one float32 endpoint named `input`; batch is dynamic or one |
| ONNX output | Exactly one float32 endpoint named `logits`, shape `[N, 10]` |
| Class order | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| Confidence | Winning value after numerically stable softmax |

The Python checkpoint also stores the class and normalization metadata for provenance. V1 keeps
the inference contract compile-time explicit rather than adding a JSON parser/configuration system
for a single model family. C++ tests cover buffer invariants, channel order, boundary normalization,
resize behavior, stable softmax, PPM failures, backend reuse, and pipeline orchestration. Python
tests cover the corresponding normalization declarations, model shape, train/evaluate helpers,
export metadata, and parity.

## Runtime validation and RAII

`OnnxRuntimeBackend` creates its environment and session once. Construction rejects a missing
model, incorrect endpoint count/name, non-float types, incompatible rank/dimensions, fixed batch
sizes other than one, and mixed fixed/dynamic input-output batch axes. A dynamic graph may execute
validated `[N,3,32,32] -> [N,10]` runtime tensors for `1 <= N <= 256`; a fixed-batch-one graph may
execute only `N=1`. The single-image CLI and pipeline still supply `[1,3,32,32]` and decode one row.

`ModelOutput` owns both row-major logits and their runtime shape. `InferenceEngine` revalidates the
shape, buffer length, finite values, duration, and input/output batch agreement after every backend
call. This keeps a malformed custom backend from bypassing the same contract enforced at the ONNX
edge.

ONNX Runtime's `GetTensorTypeAndShapeInfo()` on a session type returns an unowned view. V1 keeps
the owning `Ort::TypeInfo` local alive while reading that view. A real-runtime checkpoint caught
this lifetime rule after a chained temporary caused a valid float model to be misread; the fixed
path was rerun against the official 1.19.2 macOS arm64 release.

The backend session is reused across calls. Owning raw pointers, global sessions, and per-request
session construction are deliberately absent.

## OOP and SOLID tradeoffs

- **Single responsibility:** file decoding, tensor transformation, runtime execution, prediction
  decoding, orchestration, and presentation change for different reasons and live separately.
- **Dependency inversion:** orchestration depends on the small runtime capability, while concrete
  ONNX details stay at the edge. A recording fake exercises the same interface in unit tests.
- **Interface segregation:** `IInferenceBackend` exposes one operation; there is no broad model,
  logging, benchmarking, or server interface.
- **Liskov substitution:** fake and ONNX backends accept the same validated tensor and return the
  same `ModelOutput` contract.
- **RAII:** engine/backend ownership and ORT wrapper lifetimes encode cleanup in object lifetime.
- **Composition over inheritance:** inheritance appears only at the runtime substitution seam;
  domain data and pure policies remain concrete values.

The design is intentionally not “class per noun.” Stateless calculations can remain concrete, and
new plug-in registries or configuration layers wait until a second model/backend proves the need.

## Measurement boundaries

The benchmark uses `std::chrono::steady_clock`, discards warm-up iterations, and reports
nearest-rank p50/p95, mean, and throughput:

- `preprocessing`: an already decoded in-memory `Image` to `Tensor`.
- `runtime_only`: engine validation plus one reused-session inference for a prepared tensor.
- `end_to_end`: in-memory `Image` through preprocessing, inference, and softmax decoding.

`end_to_end` excludes file open/decode and session construction. Output includes build type,
language level, warm-up count, and measured iterations so results are not mistaken for a single
best-case call.

## Failure policy and limitations

Core modules throw contextual standard exceptions. `main` owns terminal presentation and maps
usage errors to exit 2 and inference/runtime errors to exit 1. The portable decoder limits PPM
dimensions and decoded pixels before allocation.

V1.1 assumes one CPU model and one caller. Its 79.90% CIFAR-10 result and benchmark apply only to
the hashed local reference artifacts and recorded environment. It makes no MPS, thread-safety,
server, untrusted-media, PNG/JPEG, GPU, artifact-publication, or portable-latency claim. Those
constraints are release boundaries, not hidden production promises.
