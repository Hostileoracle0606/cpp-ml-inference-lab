# Design decision matrix

This is the living architectural record for v1. A decision is **accepted** when implementation may
proceed against it, **validated** when its stated evidence exists, and **superseded** when a later
entry replaces it. Consequences are intentional costs, not omissions to hide.

## Current decisions

| ID | Decision | Alternatives considered | Why this choice | Consequences and revisit trigger | State |
|---|---|---|---|---|---|
| D-01 | V1 ends at a complete CLI inference and benchmark vertical slice. | Include HTTP server and Docker; ship only training/export. | It demonstrates the deployment boundary end to end while remaining verifiable in a small repository. | Service readiness is not claimed. Revisit before adding any service adapter. | Validated at C2 |
| D-02 | Separate adapters, application orchestration, pure core logic, and value types with inward dependency direction. | Keep one executable file; split by technical file type only. | Responsibilities and test seams match actual reasons to change. | More files and target wiring. Revisit if a boundary has no independent tests or changes. | Validated at C1 |
| D-03 | Introduce one narrow inference-backend interface; keep pure utilities concrete. | Interface for every class; no interface and direct ORT calls. | Applies DIP where volatility and test substitution are real without ceremonial OOP. | One virtual dispatch outside the measured kernel is negligible. Revisit if only one implementation and no fake-based tests remain. | Validated at C1 |
| D-04 | `InferenceEngine` is a composition facade; the ONNX adapter owns and reuses its environment/session with RAII. | Global singleton session; construct a session per prediction; expose ORT objects to the CLI. | Ownership is deterministic and initialization cost is excluded from each request. | Concrete responsibilities were clarified during implementation; D-18 replaces the original naming split. | Superseded by D-18 at C1 |
| D-05 | `Image`, `Tensor`, and `Prediction` are validated value types, not polymorphic entities. | Mutable bags; deep inheritance hierarchy. | Values make shapes/ownership explicit and are naturally testable. | Copies must be watched in benchmarks. Revisit for measured buffer-reuse needs, not speculation. | Validated at C1 |
| D-06 | Freeze the CIFAR-10 contract: RGB8 HWC input, normalized float32 NCHW `[N,3,32,32]`, fixed labels, `input`/`logits`. | Infer every policy from the graph; accept arbitrary model metadata. | ONNX describes tensor metadata but not the complete image/label preprocessing semantics. | The contract remains, but D-21 refines its v1 cross-language verification method. | Superseded by D-21 at C2 |
| D-07 | Export a dynamic batch axis and named tensor endpoints; parity threshold is max absolute logit difference `< 1e-4`. | Fixed batch one; compare only top-1 labels; compare probabilities. | Logits expose numerical drift that label agreement can hide; dynamic batch keeps benchmark evolution open. | Exporter APIs evolve. Revisit threshold only with recorded dtype/backend evidence. | Validated at C2 |
| D-08 | ONNX Runtime is optional and externally supplied; the default core build/test path has no runtime dependency. | Fetch/build ORT in CMake; require ORT for all targets. | Keeps onboarding and pure tests fast while making runtime capability explicit. | Both modes are validated on macOS arm64 with ORT 1.19.2. Revisit when a supported package manager/release image is adopted. | Validated at C1 |
| D-09 | If full image formats are required, vendor a pinned `stb_image` with license/provenance and one implementation translation unit. | OpenCV; platform codecs; PPM-only custom parser; network fetch at configure. | Small footprint and broad CLI usability suit the lab; pinning avoids mutable/network builds. | Full formats were found unnecessary for v1; D-16 chooses the smaller scope. | Superseded by D-16 at C1 |
| D-10 | Start with CTest and repository-local test executables; keep model/runtime integration tests separately labelled. | Fetch GoogleTest/Catch2; no automated tests. | Current assertion volume does not justify a configure-time dependency; CTest still provides orchestration. | Local harness is intentionally small. Revisit when fixtures/parameterization become cumbersome. | Validated at C1 |
| D-11 | Library failures carry context through exceptions; `main` owns user messages and stable exit-code mapping. | Abort/assert; print throughout core; error codes on every internal function. | Separates failure policy from presentation and preserves actionable diagnostics. | Exception behavior is documented at the process boundary. Revisit for no-exception/embedded builds. | Validated at C2 |
| D-12 | Benchmarks use a monotonic clock, exclude warm-up, disclose build/environment, and separate runtime from end-to-end samples. | One timed call; time session construction; publish only best case. | Makes results interpretable and prevents claims unsupported by the experiment. | `end_to_end` explicitly means the in-memory pipeline and excludes file/session setup. Revisit percentile method only with a versioned output schema. | Validated at C2 |
| D-13 | V1 makes no thread-safety or concurrent-inference guarantee. | Add locks/thread pool now; declare sessions universally thread-safe. | There is no server or concurrent caller in v1, and premature synchronization can distort measurements. | Single-caller scope is documented. Revisit before HTTP service or parallel benchmark work. | Validated at C2 |
| D-14 | Generated datasets, checkpoints, and ONNX graphs are build/release artifacts, not normal source files. | Commit large binaries; download artifacts implicitly in CMake. | Keeps history small and makes artifact creation explicit. | A quality release needs provenance/checksums or generation steps. Revisit if a small versioned demo model is needed for CI. | Validated at C2 |
| D-15 | Use event-based introspection after each vertical slice, not timer-based churn. | Review only at the end; rewrite design at arbitrary time intervals. | Evidence appears at integration boundaries; reviewing then catches drift while it is cheap. | C0, C1, and C2 cite evidence and changed decisions. Revisit if the delivery process changes. | Validated at C2 |
| D-16 | V1 decodes dependency-free P3/P6 PPM and caps decoded images at 16 megapixels. | Vendor `stb_image`; OpenCV; no file decoder. | PPM exercises the file-to-tensor boundary while preserving offline builds and avoiding a large decoder dependency in a teaching lab. | Common PNG/JPEG inputs require conversion, and README examples must use `.ppm`. Revisit when broad format usability becomes more valuable than dependency surface. | Validated at C1 |
| D-17 | Treat portable CTest and Python pipeline validation as separate evidence; skipped Python tests do not satisfy Gate B. | Require PyTorch for all CMake builds; count dependency skips as release success. | The C++ core should remain dependency-free, but release claims require an environment that actually executes training/export/parity tests. | Release QA uses the venv interpreter explicitly and reports skips/errors. Revisit if Python tests gain a dedicated required-dependency CMake option. | Validated at C2 |
| D-18 | `InferenceEngine` is the move-only owner/executor for one backend; `InferencePipeline` composes loading, preprocessing, execution, timing, and decoding. | Keep the original all-in-one engine sketch; merge engine and backend; expose pipeline steps to the CLI. | Separating runtime lifetime from application workflow makes both responsibilities explicit and independently testable. | Fake reuse and a real ORT CLI/benchmark run are validated. Revisit if the extra layer only forwards calls after future changes. | Validated at C1 |
| D-19 | Keep owning `Ort::TypeInfo` objects alive while using their unowned tensor-type/shape views. | Chain `GetInputTypeInfo().GetTensorTypeAndShapeInfo()` on a temporary; copy metadata immediately through lower-level C API. | The C++ wrapper explicitly returns an unowned const view; owner lifetime is part of correctness. | Locals add a few lines but prevent undefined metadata reads. Revisit only if a future ORT API returns an owning view. | Validated at C1 |
| D-20 | V1 is a source release validated with a temporary random checkpoint; a trained artifact and accuracy report are separate evidence work. | Require a long CIFAR-10 run before releasing code; commit an unproven binary; imply the random model has quality. | Training cost and classifier quality are independent of whether export/runtime contracts work, and the repository must not invent accuracy evidence. | Users must train locally; no model-quality claim is made. Revisit for a model-evidence release with seed, hardware, accuracy, and checksum. | Validated at C2 |
| D-21 | Guard cross-language preprocessing with direct shape/mean/std comparison plus C++ numerical indexing/normalization tests in v1. | Commit a 3,072-float golden tensor; duplicate only unconnected tests; parse a runtime metadata format. | The transform is a small explicit formula; the direct declaration guard detects drift and numerical tests cover behavior without a bulky fixture/config dependency. | Resizing is not compared byte-for-byte to a Python image transform. Revisit with a full tensor fixture if preprocessing becomes more complex or configurable. | Validated at C2 |

## Introspection checkpoints

### C0 — baseline and scope audit (2026-07-14, completed)

**Observed evidence**

- Repository history contains one Stage 0 skeleton commit.
- `train.py`, `export_onnx.py`, and `verify_onnx.py` all terminate through placeholder paths.
- `src/main.cpp` is a buildable argument-parser demonstration, not inference.
- `include/`, `tests/`, and `benchmarks/` contain no implementation.
- CMake already offers a useful optional `WITH_ONNXRUNTIME` direction, but it couples the eventual
  executable directly to runtime discovery and has no core library/test targets yet.

**Introspection**

The repository's stated goal is sound, but the nine-stage roadmap is larger than a credible first
release. The highest-risk correctness issue is not the CNN; it is the cross-language contract.
Accordingly, D-01 narrows v1, D-06 makes preprocessing a release contract, and D-03 creates only
the external-runtime seam needed to test orchestration. The design intentionally rejects both a
single monolithic `InferenceEngine` and an interface-per-noun architecture.

**Actions entering slice 1**

- Build pure modules and tests before runtime integration.
- Keep default builds independent of model/runtime availability.
- Do not mark roadmap stages complete until commands demonstrate them.

### C1 — first core integration/interface audit (2026-07-14, completed)

**Observed architecture so far**

- `cpp_ml_core` is the reusable library target; `infer` and the optional `inference_benchmark` are
  adapters around it.
- `Image`, `Tensor`, `ModelOutput`, and `Prediction` are non-polymorphic domain values.
- `FileImageLoader`, `Cifar10Preprocessor`, and `SoftmaxDecoder` are concrete focused modules.
- `IInferenceBackend` is the sole substitution boundary. `InferenceEngine` is a move-only backend
  owner/executor; `InferencePipeline` is the application composition facade.
- The image adapter deliberately supports P3/P6 PPM. This prompted D-16 and superseded the earlier
  conditional full-format decoder decision.
- A clean Release configuration with benchmarks temporarily disabled built and passed three CTest
  entries: `cpp_ml_core_tests`, `cpp_ml_cli_contract`, and dependency-aware
  `python_pipeline_smoke`. The first entry contained 11/11 passing core cases; the CLI contract
  covered seven help/error cases.
- The system Python selected by CMake did not have PyTorch, so all Python cases were skipped while
  CTest still reported success. Running the same suite with the project virtual environment exposed
  the true state: one pass and four errors because production train/evaluate/export paths were still
  placeholders. This prompted D-17 and keeps Gate B open.

This is a healthier result than blindly matching the initial class sketch: the words “engine” and
“pipeline” now identify executor lifetime and application composition separately. The abstraction
count stayed proportional to actual volatility. The PPM parser review also found two concrete edge
conditions—zero-valued P3 samples and allocation limits—which now have regression coverage.

**Evidence command**

```bash
rm -rf /tmp/cpp-ml-inference-lab-v1-qa
.venv/bin/cmake -S . -B /tmp/cpp-ml-inference-lab-v1-qa \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=OFF
.venv/bin/cmake --build /tmp/cpp-ml-inference-lab-v1-qa --parallel
.venv/bin/ctest --test-dir /tmp/cpp-ml-inference-lab-v1-qa --output-on-failure
.venv/bin/python -m unittest -v tests/test_python_pipeline.py
```

The first three commands produced 3/3 CTest entries passing in 0.38 seconds. The final command was
the authoritative Python check and produced one pass/four errors at this checkpoint. A default
configuration also still referenced a benchmark source that had not landed, so Gate A remained
open. These are build-in-progress observations, not release evidence.

**Decision answers**

1. A recording fake backend drives `InferenceEngine` and `InferencePipeline` without ORT headers.
2. Repeated `infer` calls reuse the same injected backend; the real runtime object owns its session,
   but runtime-enabled execution was not yet available as evidence.
3. Preprocessing, decoding, invariants, PPM parsing, and orchestration run with no model/runtime.
4. The C++ tensor side is covered; exported graph metadata/parity were still blocked by placeholder
   Python code and remain a Gate B/C requirement.
5. The only polymorphic abstraction has both recording-fake and ONNX adapter implementations; no
   unexercised interface hierarchy was introduced.
6. PPM support preserved an offline build and gained zero-sample, truncation, unsupported-format,
   and oversize rejection tests.

**Runtime follow-up evidence**

The first real ORT run rejected a valid float32 graph. The audit traced this to a chained temporary:
`GetInputTypeInfo(...).GetTensorTypeAndShapeInfo()` returns an unowned view, but its owning
`Ort::TypeInfo` had already been destroyed. Retaining input/output type-info locals fixed the
lifetime defect and produced D-19. This finding justifies keeping a real adapter check in addition
to fake-based unit coverage.

An ORT-enabled Release build was then configured against the official macOS arm64 ONNX Runtime
1.19.2 release with a generated dynamic-batch `[N,3,32,32] -> [N,10]` float32 model. Four non-Python
CTest entries passed: core, CLI contract, benchmark contract, and `cpp_ml_ort_e2e_contract`. The
last entry exercised an actual PPM prediction plus runtime-only and in-memory pipeline benchmark
paths. A separate manual prediction also printed a class, confidence, and nonnegative timing fields.

**Result:** the modular OOP direction and real runtime boundary are affirmed. The generated model
used random weights and a direct diagnostic export, so it proves C++/ORT integration—not the
production export/parity code or model quality. Python Gate B remains explicitly open, and CTest's
optional Python skip cannot be used to close it.

### C2 — release-readiness audit (2026-07-14, completed)

**Clean default evidence**

```bash
cmake -S . -B <clean-build> -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build <clean-build> --parallel
ctest --test-dir <clean-build> --output-on-failure
.venv/bin/python -m unittest -v tests/test_python_pipeline.py
```

- AppleClang 21 configured and built the library, CLI, benchmark, and test executable with project
  warnings enabled.
- CTest passed 4/4 entries: core, CLI contract, benchmark contract, and the fully provisioned Python
  pipeline. The Python suite passed 7/7 with no dataset download.
- `-DBUILD_TESTS=OFF` was separately configured and reported zero registered tests, confirming the
  compatibility option behaves as documented.
- Production paths contain no `NotImplementedError`, Stage 0 output, or required-path placeholder.

**Python/export evidence**

- The suite updates model parameters, calculates exact evaluation accuracy, rejects empty loaders,
  checks dataset transforms, and compares C++/Python NCHW dimensions and normalization constants.
- A temporary checkpoint exported named `input`/`logits` endpoints with dynamic batch axes.
- Seed 1337, batch 2 parity reported max absolute difference `2.9802322e-08` against tolerance
  `1e-4`, with matching top classes.
- No full CIFAR-10 run was represented as part of that result. D-20 explicitly separates the v1
  source release from trained-model quality evidence.

**CLI/runtime evidence**

- The portable CLI contract covers help, no arguments, unknown/duplicate options, missing values,
  individually missing required flags, and the runtime-disabled failure path with exact exit codes.
- An ORT-enabled Release build against the official ONNX Runtime 1.19.2 macOS arm64 package passed
  5/5 CTest entries. `cpp_ml_ort_e2e_contract` generated a black PPM, performed real C++ prediction,
  and exercised the model-backed benchmark.
- Runtime construction validates endpoint count/name, float types, and input/output dimensions;
  runtime output is revalidated as `[1,10]`.

**Benchmark and documentation evidence**

- The offline benchmark contract exercises preprocessing output and invalid numeric/unknown/missing
  arguments. The model-backed contract requires `runtime_only` and `end_to_end` summaries.
- Output includes build type, system, compiler/version, C++ level, warm-up, iterations, mean,
  nearest-rank p50/p95, and throughput. README and architecture explicitly state that `end_to_end`
  is an in-memory pipeline and excludes file decode/session construction.
- README commands now run from repository root, use PPM inputs, distinguish optional ORT/model
  checks, and make no HTTP, accuracy, or portable latency claim. The roadmap separates completed
  v1 code paths from trained-artifact, optimization, serving, and packaging work.

**Introspection**

C1 proved that green pure tests were insufficient: the first real ORT attempt exposed an unowned
metadata-view lifetime defect. C2 therefore keeps the real model-backed test as an explicit optional
gate rather than treating the fake backend as complete runtime proof. C1 also showed that a Python
test can appear green through dependency skips; C2 selects the venv interpreter explicitly and
reports the isolated suite. Finally, the scope is honest about evidence: random weights validate
deployment mechanics, not classifier quality, and benchmark machinery is ready without publishing
machine-specific numbers as universal results.

**Result:** Gates A–E are satisfied for the v1 source-release boundary defined by D-20. V1 is ready.

## How to update this record

1. Never silently edit an accepted decision to mean something different; add a new row and mark
   the old row superseded.
2. Attach the checkpoint identifier to each state change.
3. Prefer evidence such as a test, build target, interface, or measured result over adjectives.
4. Record tradeoffs even when the chosen implementation is simpler.
5. If reality contradicts the plan, change the decision matrix first, then reconcile the plan and
   README after the code is green.
