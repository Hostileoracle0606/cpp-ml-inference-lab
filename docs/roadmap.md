# Roadmap

The original nine-stage idea is now split into a validated v1 source release and evidence-driven
follow-up work. A checkmark means the code path and its relevant automated gate exist; it does not
stand in for unmeasured model quality or portable performance.

## V1 release scope

### 1. Python model production — complete

- [x] Shared `SmallCNN`, labels, normalization, and checkpoint loader in `python/model.py`.
- [x] Deterministic CIFAR-10 dataloaders, training, evaluation, and metadata checkpoint.
- [x] Synthetic tests for optimization, exact accuracy calculation, empty loaders, and transforms.
- [x] Train and validate a local reference checkpoint/accuracy result.

V1.1 captured 79.90% test accuracy under its exact local CPU protocol. The checkpoint remains an
ignored local artifact; it is not published or committed.

### 2. ONNX export and parity — complete

- [x] Named float32 `input`/`logits` export with dynamic batch dimension.
- [x] ONNX structural checker coverage.
- [x] Deterministic batch-two PyTorch/ONNX Runtime comparison.
- [x] V1 evidence: max absolute logit difference `2.9802322e-08 < 1e-4`; classes matched.

### 3. Modular C++ inference — complete

- [x] `cpp_ml_core` target with domain, PPM loader, preprocessing, decoder, engine, and pipeline.
- [x] Narrow injected backend interface and session-owning ONNX Runtime adapter.
- [x] Graph endpoint/type/shape validation and batch-one runtime output validation.
- [x] Strict CLI argument/error contract and formatted prediction output.
- [x] Official ONNX Runtime 1.19.2 macOS arm64 end-to-end run.

### 4. Verification — complete

- [x] Dependency-free C++ value/preprocessing/softmax/PPM/orchestration suite.
- [x] CLI success/failure contract.
- [x] Python train/export/parity suite that does not download CIFAR-10.
- [x] Optional model-backed C++ ORT end-to-end CTest.
- [x] Default Release evidence: 4/4 CTest entries passed.
- [x] Runtime-enabled evidence: 5/5 CTest entries passed.

### 5. Benchmark harness — complete

- [x] Preprocessing, runtime-only, and in-memory pipeline boundaries.
- [x] Warm-up exclusion, monotonic clock, mean, nearest-rank p50/p95, and throughput.
- [x] Build type and iteration metadata in output.
- [x] Capture trained-model reference-machine results with artifact hash, machine, runtime, and
  command in the ignored local evidence bundle.

The harness and one machine-scoped capture are validated and documented with its exact command and
scope. The benchmark record/model bundle remains unpublished, and no portable claim is made;
external artifact distribution requires a separate licensing, hosting, and policy decision even
though local provenance is complete.

## Post-v1 priorities

### V1.1 — reproducible model evidence

- [x] Define verification-only compiler, minimum-CMake, sanitizer, Python, and ORT CI; do not treat
  a workflow definition as executed evidence.
- [x] Pin and clean-install a tested Python 3.9.6/macOS arm64 constraints file.
- [x] Train and reproduce a local reference checkpoint with captured command, seed, epochs,
  accuracy, environment, and hardware.
- [x] Finalize and verify a checksummed ignored local evidence bundle; make no publication claim.
- [x] Capture trained-model parity, C++ prediction, and machine-scoped benchmark evidence.
- [x] Keep the unchanged preprocessing contract covered by direct cross-language declaration and
  numerical tests; a full tensor fixture remains conditional on future contract complexity.

The GitHub Actions workflow remains unexecuted, so v1.1 makes no supported-platform claim from its
definition alone.

### V1.2 — measured optimization

- [x] Establish the v1.1 reference Release measurement and immutable model hash.
- [x] Run the pre-registered runtime-only serial-eight versus batch-eight experiment.
- [x] Apply the frozen decision: batch passed correctness, 8/10 stability, and tail latency, but its
  12.1093% median throughput improvement missed the required 50%; roll it back and retain evidence.
- [ ] Keep any ORT session/thread or graph-optimization experiment separate and require a new
  pre-registered decision before implementation or measurement.
- [ ] Consider buffer reuse only through a separate pre-registered experiment.

V1.2 was not released. The negative result is recorded at C8; project behavior and version remain
at v1.1 until a separately pre-registered change earns a release.

### V2 — serving and packaging

- [ ] Define concurrency/thread-safety requirements before adding a server.
- [ ] Add `GET /health` and `POST /predict` as adapters around `cpp_ml_core`.
- [ ] Add request limits, structured errors, and integration/load tests.
- [ ] Containerize a pinned CPU runtime and model acquisition flow.
- [ ] Turn verification CI into explicit supported-platform guarantees and define a release
  artifact policy.

### Later exploration

- [ ] PNG/JPEG decoder adapter suitable for the intended trust boundary.
- [ ] Quantization with before/after accuracy and latency evidence.
- [ ] Additional execution providers or model families after metadata/configuration design.
- [ ] Multi-model routing and concurrency only when a real consumer requires them.

## Decision discipline

Architecture changes are recorded in [decision_matrix.md](decision_matrix.md) with alternatives,
consequences, and revisit triggers. Release boundaries and gates live in
[v1_plan.md](v1_plan.md), [v1_1_plan.md](v1_1_plan.md), and
[v1_2_plan.md](v1_2_plan.md). Each vertical slice ends with an evidence checkpoint; green optional
tests or random-weight artifacts are never presented as model-quality proof.
