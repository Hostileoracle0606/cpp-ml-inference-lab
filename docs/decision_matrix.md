# Design decision matrix

This is the living architectural record for v1 and v1.1. A decision is **accepted** when
implementation may proceed against it, **validated** when its stated evidence exists, and
**superseded** when a later entry replaces it. Consequences are intentional costs, not omissions to
hide.

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

## V1.1 decisions

Their evidence gates are defined in `docs/v1_1_plan.md`; states change only at recorded
checkpoints.

| ID | Decision | Alternatives considered | Why | How | Consequences and revisit trigger | State |
|---|---|---|---|---|---|---|
| D-22 | Use deterministic stratified train/validation/test separation: 45,000/5,000 from CIFAR-10 train with seed 1337, plus the untouched 10,000 test set. | Select on the test set each epoch; random unrecorded split; no validation set. | Model selection and final quality evidence must be independent. | Record split indices, per-class counts, and digest; train/augment only the training subset, select by validation, and evaluate test once after selection. | Changes the training helper contract. Revisit only if a different dataset/protocol is versioned before an official test run. | Validated at C4/C5 |
| D-23 | Introduce checkpoint schema v2 while retaining schema-v1 metadata and legacy raw-state compatibility. | Overwrite v1 meaning; drop old checkpoints; store metrics only in logs. | The selected weights, split, configuration, and measured accuracy must remain attributable without breaking v1 export users. | Add explicit `format_version=2` fields and dispatch raw/v1/v2 loading; reject unknown future versions clearly. | More schema tests and metadata validation. Any incompatible field change after evidence exists requires schema v3. | Validated at C5 |
| D-24 | Run official v1.1 evidence only in a fully resolved, platform-labelled reference environment. | Continue with lower bounds; claim every platform; rely on an existing mutable venv. | Exact package resolution is required to interpret training and export evidence. | Pin direct and transitive packages for Python 3.9.6 macOS arm64, validate installation in a fresh environment, and capture installed versions. | Initial reproducibility scope is one platform. Add other platforms through separate validated constraints. | Validated at C4/C5 |
| D-25 | Pre-register a CIFAR-10 top-1 test-accuracy floor of at least 65%. | Report any result; choose threshold after seeing test; optimize against test. | A quality claim needs an objective, above-chance gate that cannot move after observation. | Pilot/tune with validation, freeze the protocol, evaluate official test once, and fail v1.1 if selected weights score below `0.65`. | The release may remain incomplete if the floor is missed. The floor can rise before, never fall after, the official test run. | Validated at C5 |
| D-26 | Store v1.1 model evidence only in a checksummed local bundle; make no publishing or licensing assumption. | Commit binaries; upload release assets; keep loose unverifiable local files. | Integrity/provenance are needed now, while distribution authority and licensing are outside current scope. | Build an ignored local directory atomically with relative-path manifest, file sizes, SHA-256 digests, checkpoint, ONNX, metrics, environment, benchmark, and command capture. | Other users cannot fetch the bundle. External publication requires a new explicit decision and licensing review. | Validated at C5 |
| D-27 | Artifact acquisition/generation is explicit and never occurs during CMake configure/build. | Fetch models/runtimes in CMake; silently download during tests; require implicit local state. | Default builds must stay offline and deterministic, preserving D-08 and the v1 onboarding boundary. | Require visible Python/user commands and explicit local paths such as `ONNXRUNTIME_ROOT` and `CPP_ML_TEST_MODEL`; test missing artifacts with actionable errors. | More setup remains user-driven. Revisit only with a separately designed package manager or deployment release. | Validated at C5 |
| D-28 | Treat CMake presets, generated ORT fixtures, and CI as verification adapters, not scientific or release-model evidence. | Defer all portability automation; treat synthetic/CI outputs as reference-model proof. | Compiler, minimum-CMake, sanitizer, and hostile-model coverage catch integration drift without changing D-24's exact reference environment or v1.1's product scope. | Keep raw CMake 3.16 canonical; make presets optional CMake 3.21+ conveniences; download Python/ORT only in explicit workflow steps; generate deterministic synthetic ONNX fixtures only under the build tree; never train, publish, or upload evidence in CI. | Workflow results may support scoped portability claims but never satisfy accuracy, benchmark, reference-lock, or local-bundle gates. Revisit with an explicit supported-platform and release-artifact policy. | Validated for local adapters at C5; workflow unexecuted |
| D-29 | Run the official v1.1 reference training and evaluation on CPU; treat MPS as unsupported for scientific evidence in this environment. | Accept the much faster MPS metrics; abandon local training; diagnose or upgrade the external stack before continuing. | A validation-only diagnostic showed impossible MPS metrics: 98.44% after one epoch, while the same weights scored 10% on CPU. Device output therefore cannot be trusted even though a small optimizer smoke passed. | Freeze `--device cpu`, deterministic algorithms, seed 1337, batch 128, Adam `1e-3`, and 20 epochs before the official test observation. Record the failed MPS diagnostic separately and require CPU evaluation/export evidence. | The official run is slower and no MPS quality/performance claim is made. Revisit only after a pinned-stack cross-device parity test explains and eliminates the discrepancy. | Validated at C5 |
| D-30 | Separate portable lightweight bundle verification from an opt-in deep semantic audit. | Deserialize models in the default verifier; trust hashes alone; use one dependency-heavy verifier. | Integrity/schema checking and model execution have different dependency and trust boundaries. | The standard-library verifier rejects unsafe paths, symlinks, missing/extra files, hash drift, frozen-schema drift, record disagreement, artifact-link drift, and command-provenance drift. The pinned reference environment runs `deep_verify_evidence.py`, loads the schema-v2 checkpoint with `weights_only=True`, compares checkpoint provenance to the manifest, checks ONNX, and recomputes parity. | Both checks are required for release evidence. Deep verification is only for a trusted local bundle and is not an untrusted-upload parser. | Validated at C5 |
| D-31 | Exclude r1 from release candidacy because its raw training invocation was not captured; perform one unchanged r2 command-provenance reproduction. | Infer the command from config; mutate r1; weaken Gate G; use r1 because its metrics passed. | Gate G pre-registered exact commands, and configuration fields do not capture the shell invocation, paths, redirection, or logging behavior. | Preserve r1 as diagnostic evidence. Capture the r2 command verbatim, keep the CPU/20-epoch/seed/batch/optimizer protocol and 65% floor unchanged, and accept or reject r2 on its own gates without falling back to r1. | The test set was observed in r1, so r2 is a provenance correction—not a tuning opportunity. No hyperparameter or floor may change based on r1. | Validated at C5 |

## V1.2 pre-implementation decision

| ID | Decision | Alternatives considered | Why | How | Consequences and revisit trigger | State |
|---|---|---|---|---|---|---|
| D-32 | Limit v1.2 to a measured runtime-only batch-eight experiment; keep preprocessing, decoding, pipeline, and CLI batch-one. | Add public batch APIs across every layer; build a request queue/server; tune several batch sizes and select the best; skip batching. | The dynamic ONNX graph creates a concrete runtime hypothesis, but no multi-image consumer yet defines public ordering, timing, partial-failure, cancellation, or backpressure semantics. | Add validated `ModelOutput` shape, permit capped `[N,3,32,32] -> [N,10]` runtime execution, and compare one batch-eight call with eight serial batch-one calls over identical prepared tensors. Pre-register a 50% median items/s improvement, 8/10 favorable paired runs, no group-p95 regression, and full correctness gates before implementation. | V1.2 will not expose batching to application callers even if the runtime experiment passes. Revisit public batch APIs only with a concrete CLI/server consumer; another batch size or optimization needs a new decision before measurement. | Implemented and correctness-validated at C7; performance pending |
| D-33 | Freeze the v1.2 paired sampling recipe before its first performance run. | Choose sample counts after observing variance; compare separate binaries/sessions; retain only summary statistics. | D-32 fixed the acceptance thresholds but left sample counts and tail aggregation underspecified, which could permit accidental post-result tuning. | In each of ten fresh processes, use one persistent session and eight prepared rows whose pixel at flat index `i` in zero-based row `r` is `(i*37 + r*17) % 256`. Use 20 warm-up workloads per mode and 200 measured eight-item workloads per mode. Odd runs execute serial then batch; even runs reverse the order. Retain all 2,000 group-latency samples per mode. Per-run items/s is `1600 / sum(group_ms) * 1000`; the paired ratio is batch items/s divided by serial items/s. Sort the ten ratios and average the fifth and sixth for the median. Count ratios above one for stability. Compare nearest-rank p95 across each mode's combined 2,000 group samples. | The experiment is intentionally narrow and machine-scoped. A failed/crashed/malformed run aborts the experiment and is not selectively replaced. Any different counts, input recipe, session policy, aggregation rule, or rerun after inspecting results requires a new decision and cannot replace this result silently. | Accepted pre-measurement |

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

### C3 — v1.1 pre-implementation audit (2026-07-14, completed)

**Frozen baseline**

- Local tag `v1.0.0` points to commit `c7c297a`, “Release v1.0.0 modular inference source
  baseline.”
- The v1.1 branch started from that clean baseline, so protocol changes can be compared and rolled
  back without mixing them into the v1 implementation.
- No v1.1 model training, accuracy result, evidence bundle, or benchmark result exists at C3.

**Three audits carried forward**

1. **C0 baseline/scope audit:** found the Stage 0 skeleton, narrowed v1 to a trustworthy CLI and
   benchmark vertical slice, and established that cross-language contracts were the primary risk.
2. **C1 integration/runtime audit:** validated the restrained OOP boundary, caught PPM edge cases,
   exposed an ONNX Runtime unowned-view lifetime defect through a real model run, and showed that
   dependency-skipped Python tests cannot count as evidence.
3. **C2 release-readiness audit:** validated the clean source pipeline while explicitly recording
   that random weights prove mechanics—not classifier quality—and that no trained artifact or
   accuracy claim belonged to v1.

**Post-v1 planning introspection**

- The smallest coherent next value is local reference-model evidence, not serving or optimization.
- The v1 training entry point evaluates the test loader each epoch, records a best observed metric,
  and saves final weights; that workflow cannot support a credible selected-model accuracy claim.
- Minimum dependency versions are adequate for development but not sufficient provenance for an
  official reference run.
- D-14 and D-20 anticipated an evidence policy, but current scope provides neither a trained model
  nor integrity-bound artifact bundle.
- External publishing and licensing authority are intentionally unresolved. C3 therefore chooses a
  local-only bundle and makes no distribution assumption.

**Decisions entering implementation**

- D-22 separates selection from final test evidence.
- D-23 makes selected-weight provenance explicit without breaking raw/v1 checkpoint consumers.
- D-24 defines one exact reference environment rather than claiming cross-platform identity.
- D-25 freezes the `>=65%` quality floor before the official test observation.
- D-26 binds local artifacts and evidence with SHA-256 while deferring publication.
- D-27 preserves offline CMake behavior and explicit artifact paths.

**Result:** v1.1 protocol decisions are accepted for implementation. None is validated at C3;
`docs/v1_1_plan.md` Gates A–G and future C4–C6 checkpoints control evidence and readiness.

### C3a — verification-adapter reconciliation (2026-07-14, completed)

**Observed implementation direction**

- The in-progress slice defines optional CMake presets, a compiler/minimum-CMake/sanitizer CI
  matrix, a fully provisioned Linux Python/ORT job, and deterministic valid/adversarial ONNX test
  fixtures generated under the build tree.
- C++ public interfaces remain unchanged. The code changes tighten PPM parsing, duration
  validation, and the frozen ONNX metadata/runtime contract.
- Default CMake still downloads nothing, ONNX Runtime remains optional and externally supplied,
  and explicit workflow download steps verify the pinned SDK checksum before passing its path to
  CMake. D-08 and D-27 therefore remain intact.

**Evidence boundary**

- This checkpoint records design reconciliation, not executed v1.1 gates. D-22 through D-28 are
  accepted but not yet validated, and C4 has not been reached.
- Generated zero-weight graphs test adapter mechanics and rejection paths; they are not trained
  model artifacts and say nothing about accuracy.
- The Linux/Python 3.11 workflow is portability intent, not D-24's macOS arm64/Python 3.9.6
  reference environment. A workflow file is not evidence that GitHub has executed it.

**Result:** D-28 admits verification-only automation without broadening the v1.1 scientific claim.
Later checkpoints must report local execution separately from any remotely executed workflow.

### C4 — protocol and reference-environment checkpoint (2026-07-14, completed)

**Executed protocol evidence**

- A fresh Python 3.9.6 virtual environment with system packages disabled installed all 20 exact
  macOS arm64 constraints without manual changes.
- The isolated environment passed 17/17 Python tests, the full dependency import gate, export
  parity (`2.9802322e-08 < 1e-4` with matching classes), and a 20/20 lock-version audit.
- The official 170,498,071-byte CIFAR-10 Python archive matched MD5
  `c58f30108f718f92721af3b95e74349a`; torchvision verified 50,000 training and 10,000 test
  examples.
- The frozen split contains 45,000 training and 5,000 validation indices, zero overlap, exactly
  4,500/500 examples per class, and digest
  `a9ca2a07e1b376a9333b878d2677c65dc777042c44094b69aca5560d44a1755e`.
- Local verification matrices passed 4/4 default, 3/3 sanitizer portable, 5/5 full-Python preset,
  and 9/9 full ORT tests. The GitHub workflow itself remains unexecuted intent.

**Device introspection before the official test observation**

- A deterministic MPS step with synthetic data was finite, but a full validation-only epoch was
  not credible: MPS reported loss `0.024696` and validation accuracy `0.9844`.
- The selected training/validation indices had zero overlap, and the same post-epoch weights
  evaluated on CPU scored `0.1000`. This isolates the discrepancy to the MPS execution path rather
  than the split or saved weights.
- A deterministic CPU validation-only epoch produced loss `1.703588` and accuracy `0.5102`, a
  plausible baseline. D-29 therefore freezes CPU as the official evidence device before the test
  set is evaluated.

**Result:** WP0–WP2 and Gates B/C protocol prerequisites are validated for the scoped reference
environment. No official test accuracy, trained artifact, benchmark, or completed evidence bundle
exists yet; Gates D–G remain open entering WP4.

### C5 — local evidence and provenance checkpoint (2026-07-15, completed)

**Provenance correction**

- The first CPU run selected epoch 20, reached validation accuracy `0.8128` and test accuracy
  `0.7990`, produced trained-model parity `1.6689301e-06 < 1e-4`, and passed the 9/9 trained-model
  ORT suite. Its bundle passed lightweight integrity verification and the deep semantic audit.
- That r1 bundle remains diagnostic evidence only because its raw training invocation was not
  captured. It was neither edited nor silently promoted after the omission was found.
- D-31 froze an unchanged r2 reproduction before execution. No hyperparameter, split, seed,
  device, optimizer, epoch count, parity tolerance, or 65% floor changed after observing r1.

The exact r2 invocation was:

```text
set -o pipefail
mkdir -p artifacts/v1.1-run-r2
/tmp/cpp-ml-v1.1-ref-venv-20260714/bin/python python/train.py --epochs 20 --batch-size 128 --lr 0.001 --data-dir python/data --out models/cifar10_cnn-v1.1-r2.pt --num-workers 0 --seed 1337 --split-seed 1337 --no-download --record-dir artifacts/v1.1-run-r2/records --device cpu 2>&1 | tee artifacts/v1.1-run-r2/training-cpu.log
```

**R2 measured evidence**

- R2 reproduced all 20 epoch loss/validation records exactly, selected epoch 20, reached validation
  accuracy `0.8128`, and reached test accuracy `0.7990` in its one test evaluation.
- The checkpoint SHA-256 is
  `315b201bce905c7ddb2c4789f86cf062e18a1bcb11a7e1aa3d43965570e0ad23`.
- The ONNX SHA-256 is
  `fec6ac786c75d2e328618a021763b408b993b9d6246772bf305eb3cd996b255c`; the graph is
  byte-identical to r1 because the selected weights are identical.
- Batch-two parity was `1.6689301e-06 < 1e-4` with matching classes. The r2 trained graph passed
  all 9/9 native Release ORT/Python CTest entries.
- The final local manifest SHA-256 is
  `35d391d0d2e8b41041cc01d0ea4762d5b425fa3bd3faf0647c768339226c4aad`. The lightweight verifier
  accepted nine artifacts, and the pinned-environment deep audit matched checkpoint metadata,
  checked ONNX, and recomputed the recorded parity.
- The Release Apple M4/macOS 26.4/AppleClang 21/ORT 1.19.2 benchmark recorded runtime-only mean
  `0.5322 ms`, p50 `0.4089 ms`, p95 `1.2607 ms`, and `1878.9487 operations/s`. It is one local
  capture over 200 iterations after 20 warm-ups, not a portable performance claim. Its exact
  command was `/tmp/cpp-ml-v11-r2-ort/inference_benchmark --model
  models/cifar10_cnn-v1.1-r2.onnx --warmup 20 --iterations 200`.

**Evidence boundary**

- Before the final source-version change, native profiles had passed 4/4 default, 3/3 sanitizer
  portable, 5/5 full-Python preset, and 9/9 full ORT. C6 reruns them against the final source.
- CMake 3.16.8 configured and built x86_64 outputs under Rosetta; its portable label passed 3/3.
  This is a minimum-CMake result, not native arm64 Python/ORT evidence, and C6 reruns it after the
  version change.
- GitHub Actions remains unexecuted workflow intent. No model, bundle, benchmark record, or
  download URL was published.

**Result:** Gates B–F pass for r2. Gate A has pre-final native and scoped CMake 3.16 evidence. Gate
G advances to C6 source/version/claim reconciliation.

### C6 — v1.1 source-release readiness checkpoint (2026-07-15, completed)

**Final-source verification**

- The CMake project version is `1.1.0`. A clean native Release default profile passed 4/4 CTest
  entries, and the dependency-required Python profile passed 5/5.
- The AppleClang ASan/UBSan build passed all 3/3 portable entries with
  `detect_leaks=0:halt_on_error=1`; Apple's runtime rejected the initial Linux-only
  `detect_leaks=1` option before code execution, so that option failure is not represented as a
  code test.
- The native Release trained-model profile passed 9/9 with ONNX Runtime 1.19.2 and the r2 ONNX
  hash. The isolated Python suite contains 18/18 passing tests.
- Official checksum-verified CMake 3.16.8 configured and built x86_64 Release outputs under
  Rosetta; its portable label passed 3/3 against the final source.
- The r2-final lightweight verifier passed after sealing. The canonical post-seal trusted-local
  deep-audit command then loaded the schema-v2 checkpoint with weights-only semantics, matched
  internal metadata, checked ONNX, and recomputed parity.

**Claim and release reconciliation**

- Every documented model-quality, parity, integrity, and latency number maps to the r2 manifest,
  a hashed record, or captured command output.
- Benchmark scope is limited to the recorded Apple M4/macOS 26.4/AppleClang 21/ONNX Runtime 1.19.2
  machine, hashed ONNX graph, and in-memory workload. It is not a portable performance claim.
- Checkpoint, ONNX, dataset, and evidence bundles remain ignored local artifacts. No publication,
  licensing, hosting, retention, or download guarantee is made.
- GitHub Actions remains unexecuted and supports no platform-status claim. Its workflow is a
  verification adapter whose results must be reported separately if it is later run.
- D-32 freezes the next runtime-only batch experiment before implementation or measurement; it
  does not broaden the v1.1 product boundary.

**Result:** Gates A–G pass in their stated scopes. V1.1 is ready as a source release with
reproducible local reference-model evidence, not as a binary-model distribution.

### C7 — v1.2 pre-measurement source checkpoint (2026-07-15, completed)

**Small runtime-only change**

- `ModelOutput` now carries and validates its runtime shape, logit count/finiteness, and measured
  duration. `InferenceEngine` accepts only capped CIFAR-10 `[N,3,32,32]` input and requires matching
  `[N,10]` output for `1 <= N <= 256`.
- The existing `IInferenceBackend::run(const Tensor&) -> ModelOutput` seam did not change. The ONNX
  adapter accepts coupled dynamic batch axes, preserves fixed-batch-one compatibility, rejects
  mixed axes and other fixed batches at construction, and rejects runtime shape disagreement.
- Preprocessing, decoding, pipeline, and CLI remain batch-one. No new hierarchy, public batch API,
  queue, server, threading policy, or runtime tuning was introduced.

**Correctness and experiment controls**

- Fresh native checks passed 5/5 default Release tests, 3/3 AppleClang ASan/UBSan portable tests
  with leak detection disabled, and 13/13 dependency-required trained-model ORT/Python tests. The
  Python pipeline suite remained 18/18 and the experiment orchestrator/aggregator passed 6/6 unit
  tests, including ten-run success and first-failure abort paths.
  Official CMake 3.16.8 built x86_64 outputs under Rosetta and passed 3/3 portable tests.
- Generated fixtures cover dynamic batches 1/8/256, zero/257 limits, fixed batch one, fixed batch
  two rejection, both mixed-axis directions, runtime batch/class disagreement, and deterministic
  row order. The r2 trained graph's batch-eight output matches eight single calls within `1e-5`
  with identical classes.
- Trained PyTorch-to-ORT batch-eight parity is `2.6226044e-06 < 1e-4` with matching classes. The
  v1.1 lightweight bundle verifier and deep semantic audit remain green.
- `run_batch_experiment.py` requires a clean committed source tree, verifies the full frozen ONNX
  and ORT 1.19.2 library SHA-256 values plus benchmark linkage and reference-machine build scope,
  records the source and benchmark-binary hashes, creates a new output directory, launches runs 1
  through 10 once, retains all raw samples/logs, aborts without selective replacement, and applies
  D-33's conventional median and pooled nearest-rank p95 rules. Its aggregation contract has
  independent unit coverage.

**Result:** D-32 is implemented and all correctness prerequisites pass. No official trained-model
paired performance process has run. The next event is a clean candidate commit/build followed by
the one D-33 experiment; the measured outcome, positive or negative, advances to C8.

## How to update this record

1. Never silently edit an accepted decision to mean something different; add a new row and mark
   the old row superseded.
2. Attach the checkpoint identifier to each state change.
3. Prefer evidence such as a test, build target, interface, or measured result over adjectives.
4. Record tradeoffs even when the chosen implementation is simpler.
5. If reality contradicts the plan, change the decision matrix first, then reconcile the plan and
   README after the code is green.
