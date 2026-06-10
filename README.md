# C++ ML Inference Lab

> Building a C++ ML inference system to understand the **deployment layer** behind machine
> learning: model export, runtime inference, benchmarking, optimization, and serving.

A production-style project that explores what happens **after** a model is trained. It trains a
small image classifier in **PyTorch**, exports it to **ONNX**, loads it from **C++** via
**ONNX Runtime**, runs inference from a command-line tool, benchmarks latency/throughput, and
ultimately serves predictions over HTTP.

Most beginner ML projects stop at accuracy inside a notebook. This one starts where that ends.

```text
Trained model → Exported model → C++ runtime → Benchmarks → Inference service
```

---

## Why this project

Training is only the first step. Shipping a model means dealing with portability, preprocessing
consistency, runtime execution, latency, batching, and service integration. This repo is a guided
tour through that deployment layer, built to be readable by both engineers and recruiters.

It demonstrates practical ability across:

- Python ML training & PyTorch model development
- ONNX export and cross-runtime numerical parity
- Modern C++17 systems programming with CMake
- C++ inference using ONNX Runtime
- Testing, benchmarking, and latency/throughput analysis
- Backend-style API serving and reproducible packaging

---

## Architecture (target)

```text
        ┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
        │  PyTorch     │     │   ONNX file  │     │  C++ Inference Core │
        │  training    │ ──▶ │ (.onnx graph)│ ──▶ │  (ONNX Runtime)     │
        │  python/     │     │   models/    │     │  src/ + include/    │
        └──────────────┘     └──────────────┘     └──────────┬──────────┘
                                                             │
                              ┌──────────────────────────────┼───────────────┐
                              ▼                               ▼               ▼
                        ┌───────────┐                  ┌────────────┐   ┌───────────┐
                        │  CLI tool │                  │ Benchmarks │   │ HTTP API  │
                        │  ./infer  │                  │ p50 / p95  │   │ /predict  │
                        └───────────┘                  └────────────┘   └───────────┘
```

See [docs/architecture.md](docs/architecture.md) for component responsibilities and the
train/serve preprocessing-parity contract.

---

## Project status

| Stage | Milestone | Status |
|------:|-----------|:------:|
| 0 | Project setup & positioning | ✅ Done |
| 1 | PyTorch training baseline (CIFAR-10 CNN) | ⬜ Planned |
| 2 | ONNX export + PyTorch/ONNX parity check | ⬜ Planned |
| 3 | C++ inference CLI (ONNX Runtime) | ⬜ Planned |
| 4 | Clean C++ architecture + unit tests | ⬜ Planned |
| 5 | Benchmarking (latency / throughput) | ⬜ Planned |
| 6 | Optimization pass (before/after) | ⬜ Planned |
| 7 | C++ inference server (`/predict`, `/health`) | ⬜ Planned |
| 8 | Dockerization & reproducibility | ⬜ Planned |
| 9 | Final polish & portfolio packaging | ⬜ Planned |

Full plan: [docs/roadmap.md](docs/roadmap.md).

---

## Repository layout

```text
cpp-inference-lab/
├── README.md              # You are here
├── CMakeLists.txt         # C++17 build definition
├── .gitignore
├── docs/
│   ├── architecture.md    # System design & data flow
│   └── roadmap.md         # Stage-by-stage plan
├── python/
│   ├── requirements.txt   # PyTorch / ONNX toolchain
│   ├── train.py           # Stage 1: train the CNN
│   ├── export_onnx.py     # Stage 2: export .pt -> .onnx
│   └── verify_onnx.py     # Stage 2: PyTorch vs ONNX parity
├── src/
│   └── main.cpp           # Stage 3: C++ inference CLI entry point
├── include/               # C++ public headers (Stage 4)
├── tests/                 # C++ unit tests (Stage 4)
├── benchmarks/            # Latency/throughput harness (Stage 5)
├── models/                # Trained .pt and exported .onnx artifacts
└── samples/               # Example input images for inference
```

---

## Quickstart (preview)

> Stages 1–3 fill these in. The commands below are the **target** developer experience.

```bash
# 1. Train the model (Python)
cd python
pip install -r requirements.txt
python train.py            # -> ../models/cifar10_cnn.pt

# 2. Export to ONNX and verify parity
python export_onnx.py      # -> ../models/cifar10_cnn.onnx
python verify_onnx.py      # PyTorch vs ONNX Runtime diff

# 3. Build and run the C++ inference CLI
cmake -S . -B build -DWITH_ONNXRUNTIME=ON
cmake --build build
./build/infer --model models/cifar10_cnn.onnx --image samples/cat.png
```

Expected output (target):

```text
Prediction: cat
Confidence: 0.812
Latency:    5.8 ms
```

---

## Roadmap-driven sections (filled in as stages land)

- **Training** — accuracy table, training notes  _(Stage 1)_
- **ONNX Export** — parity results  _(Stage 2)_
- **C++ Inference** — CLI usage & terminal demo  _(Stage 3)_
- **Benchmarks** — latency/throughput tables  _(Stage 5)_
- **Optimization Notes** — before/after measurements  _(Stage 6)_
- **Inference Server** — `curl /predict` demo  _(Stage 7)_

---

## Positioning

> I'm not just training models; I'm learning how models are deployed, measured, and served.
