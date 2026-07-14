# Roadmap

The original nine-stage idea is now split into a validated v1 source release and evidence-driven
follow-up work. A checkmark means the code path and its relevant automated gate exist; it does not
stand in for unmeasured model quality or portable performance.

## V1 release scope

### 1. Python model production — complete

- [x] Shared `SmallCNN`, labels, normalization, and checkpoint loader in `python/model.py`.
- [x] Deterministic CIFAR-10 dataloaders, training, evaluation, and metadata checkpoint.
- [x] Synthetic tests for optimization, exact accuracy calculation, empty loaders, and transforms.
- [ ] Train and publish a reference checkpoint/accuracy result.

The unchecked artifact is deliberately not required for the source release. No accuracy number is
claimed until a reproducible full training run is captured.

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
- [ ] Publish portable reference-machine results for a trained model.

The harness is validated; absolute benchmark claims remain intentionally unpublished until the
model artifact, machine, runtime configuration, and command are recorded together.

## Post-v1 priorities

### V1.1 — reproducible model evidence

- [ ] Pin a tested Python environment with a lock/constraints file.
- [ ] Train a reference checkpoint with captured seed, epochs, accuracy, and hardware.
- [ ] Publish artifact provenance/checksum outside normal Git history.
- [ ] Capture trained-model parity, C++ prediction, and benchmark tables.
- [ ] Add a full Python-to-C++ preprocessing fixture if the model contract changes.

### V1.2 — measured optimization

- [ ] Establish reference release-build measurements.
- [ ] Compare buffer reuse and batched inference with the same workload.
- [ ] Evaluate ORT session/thread settings and graph optimization levels.
- [ ] Keep changes only when the benchmark delta and correctness gates justify them.

### V2 — serving and packaging

- [ ] Define concurrency/thread-safety requirements before adding a server.
- [ ] Add `GET /health` and `POST /predict` as adapters around `cpp_ml_core`.
- [ ] Add request limits, structured errors, and integration/load tests.
- [ ] Containerize a pinned CPU runtime and model acquisition flow.
- [ ] Add CI across supported platforms and an explicit release artifact policy.

### Later exploration

- [ ] PNG/JPEG decoder adapter suitable for the intended trust boundary.
- [ ] Quantization with before/after accuracy and latency evidence.
- [ ] Additional execution providers or model families after metadata/configuration design.
- [ ] Multi-model routing and concurrency only when a real consumer requires them.

## Decision discipline

Architecture changes are recorded in [decision_matrix.md](decision_matrix.md) with alternatives,
consequences, and revisit triggers. The release boundary and gates live in
[v1_plan.md](v1_plan.md). Each vertical slice ends with an evidence checkpoint; green optional tests
or random-weight artifacts are never presented as model-quality proof.
