# Roadmap

Stage-by-stage plan. Each stage produces a public artifact and a resume-ready outcome.

## Stage 0 — Project setup & positioning ✅
- [x] Directory skeleton (`python/`, `src/`, `include/`, `tests/`, `benchmarks/`, `models/`, `samples/`)
- [x] README (recruiter-readable)
- [x] Architecture doc
- [x] Roadmap doc
- [x] CMake build that compiles a CLI skeleton
- [ ] First git commit

## Stage 1 — PyTorch training baseline
- [ ] `python/train.py` — small CNN on CIFAR-10
- [ ] `models/cifar10_cnn.pt`
- [ ] Validation accuracy reported in README
- **Artifact:** accuracy table.

## Stage 2 — ONNX export & parity check
- [ ] `python/export_onnx.py` → `models/cifar10_cnn.onnx`
- [ ] `python/verify_onnx.py` — PyTorch vs ONNX Runtime max-diff < epsilon
- **Artifact:** parity table.

## Stage 3 — C++ inference CLI
- [ ] `InferenceEngine` (ONNX Runtime session)
- [ ] Image preprocessing in C++
- [ ] `./infer --model … --image …` prints label / confidence / latency
- **Artifact:** terminal demo.

## Stage 4 — Clean C++ architecture & tests
- [ ] Header/source separation (`inference_engine`, `image_preprocessor`, `tensor_utils`)
- [ ] Unit tests for preprocessing, tensor utils, decoding
- **Artifact:** architecture diagram + green tests.

## Stage 5 — Benchmarking
- [ ] `benchmarks/inference_benchmark.cpp`
- [ ] Preprocessing vs model-execution vs end-to-end latency
- [ ] p50 / p95 / throughput across batch sizes
- **Artifact:** latency table.

## Stage 6 — Optimization pass
- [ ] Session reuse, buffer reuse, batching, release build
- [ ] Before/after measurements
- **Artifact:** before/after table.

## Stage 7 — C++ inference server
- [ ] `GET /health`, `POST /predict`
- [ ] JSON response with class / confidence / latency
- **Artifact:** `curl /predict` demo.

## Stage 8 — Dockerization & reproducibility
- [ ] `Dockerfile`, `docs/setup.md`
- [ ] One-command build & run
- **Artifact:** reproducible setup.

## Stage 9 — Final polish & portfolio packaging
- [ ] Final README, demo GIF, benchmark tables
- [ ] Resume bullets, LinkedIn post
- **Artifact:** polished, pinned repo.

---

## Compressed 5-week recruiting track

| Week | Milestone | Public artifact |
|-----:|-----------|-----------------|
| 1 | PyTorch train + ONNX export | accuracy + parity tables |
| 2 | C++ inference CLI | terminal demo |
| 3 | Architecture + tests | architecture diagram |
| 4 | Benchmarks | latency table |
| 5 | Server + final polish | `curl /predict` demo |
