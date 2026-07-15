# Version 1.1 plan — local reference-model evidence

## Status

C5 and C6 are complete for the r2 local evidence run. The frozen CPU protocol, exact reference
environment, verbatim invocation, model-quality result, export/parity result, C++ runtime result,
benchmark, checksummed bundle, final source profiles, and version reconciliation have executed
evidence. V1.1 is ready as a source release with reproducible local model evidence. Evidence
artifacts remain ignored and local; no model, bundle, benchmark artifact, or download URL is
published, and GitHub Actions remains unexecuted. Implementation started from the frozen local
`v1.0.0` tag at commit `c7c297a`.

## Goal

V1.1 turns the verified v1 source pipeline into a scientifically credible **local** reference-model
workflow. It must train with deterministic train/validation/test separation, select weights using
validation data, evaluate the official test set once, export and verify the selected model, and
assemble a checksummed local evidence bundle.

The pre-registered release floor is **at least 65% CIFAR-10 top-1 test accuracy**. The floor may be
raised before the official run, but it must not be lowered after test accuracy is observed.

## Non-goals

- Publishing models, release assets, download URLs, benchmark artifacts, or machine-independent
  performance claims externally. Source documentation may report a strictly machine-scoped local
  capture when its artifact hash, command, boundaries, and environment are explicit.
- Making any licensing or redistribution assumption about source, dataset, samples, or weights.
- Changing the C++ model contract, backend interface, pipeline composition, or single-caller scope.
- Adding CI-based training or scientific evidence, artifact publication, deployment, HTTP serving,
  Docker, GPU providers, quantization, batching, or optimization claims.
- Adding PNG/JPEG decoding or a generic runtime metadata/configuration system.
- Claiming bit-identical retraining across different hardware or software stacks.

## Local-only artifact policy

- V1.1 evidence remains under a local ignored directory such as `artifacts/v1.1/`; it is not
  committed, uploaded, or linked from public documentation.
- The local bundle contains the selected checkpoint, ONNX graph, training/evaluation metrics,
  environment record, benchmark capture, `manifest.json`, and `SHA256SUMS`.
- The manifest names only relative bundle paths and records file sizes and SHA-256 digests.
- Artifact generation or copying is an explicit user command. CMake configure/build/test never
  downloads datasets, runtimes, models, or other artifacts.
- `train.py` may download CIFAR-10 only through its existing explicit Python entry point; the
  command and network effect must remain visible to the caller.
- If future external publication is desired, licensing, ownership, hosting, retention, and
  immutability require a new decision before any upload.

## Scientific protocol

### Dataset partitions

- Source: CIFAR-10's 50,000 training examples and 10,000 official test examples.
- Split seed: `1337`.
- Training subset: 45,000 examples, stratified to 4,500 per class.
- Validation subset: 5,000 examples, stratified to 500 per class.
- Test subset: all 10,000 official test examples.
- The training transform keeps augmentation; validation and test use only tensor conversion and
  the frozen CIFAR-10 normalization.
- The split indices and their SHA-256 digest are recorded in the evidence manifest.

### Selection and test use

1. Tune or pilot only against training and validation metrics.
2. Save a candidate state whenever validation top-1 improves, with a deterministic tie rule:
   earliest epoch wins.
3. Restore the selected state after training.
4. Evaluate the official test loader exactly once in the official evidence run.
5. Save `test_accuracy` for those selected weights; never reuse v1's ambiguous `best_accuracy`
   field as quality evidence.

### Accuracy acceptance

- Test top-1 must be finite, in `[0, 1]`, and at least `0.65`.
- The selected checkpoint, reported test accuracy, ONNX export, and checksums must refer to the same
  weights.
- If the floor is missed, v1.1 is not ready. Improvements use validation evidence and require a new
  frozen protocol before another official test evaluation; the floor is not reduced retroactively.

## Interfaces and value objects

V1.1 adds Python value-oriented protocol types, not a new inheritance hierarchy:

| Value/interface | Responsibility | Design rule |
|---|---|---|
| `TrainingConfig` | Immutable seed, split, optimizer, epoch, batch, worker, and device settings | Serializable and validated before data/model work |
| `DatasetSplit` | Train/validation indices plus digest and class counts | Immutable, disjoint, deterministic for a seed |
| `TrainingResult` | Selected epoch, validation metric, test metric, and elapsed time | Contains evidence, not model ownership |
| Checkpoint schema v2 | Selected state plus model/training/provenance metadata | Mapping on disk; loader remains backward-compatible |
| `EvidenceManifest` | Relative files, hashes, sizes, contract, environment, and measured results | Local-bundle value; no C++ runtime dependency |
| `make_dataset_split(...)` | Create/validate the stratified partition | Pure apart from input dataset labels |
| `train_with_validation(...)` | Train and retain the best validation state | Test loader is not an argument |
| `evaluate_test_once(...)` | Evaluate selected weights for official evidence | Called once by the evidence orchestration path |
| `build_evidence_bundle(...)` | Write local files, then hashes and manifest | Explicit command; atomic finalization |
| `verify_evidence_bundle(...)` | Recompute sizes/hashes and validate manifest consistency | Must not trust manifest claims before verification |

The existing C++ `Image`, `Tensor`, `Prediction`, `IInferenceBackend`, `InferenceEngine`, and
`InferencePipeline` interfaces remain unchanged. A new C++ abstraction requires separate evidence
that the v1 boundary is insufficient.

## Checkpoint schema v2

Required fields:

- `format_version = 2`
- `model_state_dict`
- class labels and normalization mean/std
- training seed and split seed
- split digest and per-class train/validation counts
- epochs requested, selected epoch, batch size, learning rate, optimizer, and device
- best validation top-1 and selected-model test top-1
- exact reference environment identifier

`load_checkpoint` must continue accepting schema-v1 metadata checkpoints and legacy raw state
dicts. Export consumes all three formats, but only schema v2 can support a v1.1 model-quality claim.
Unknown future schema versions fail clearly rather than being interpreted as raw weights.

## Exact reference environment

- Reference platform begins with the already validated scope: macOS arm64, Python 3.9.6.
- The official v1.1 training/evaluation device is CPU. MPS is excluded from scientific evidence
  after a validation-only cross-device diagnostic produced incompatible results for the same
  weights; see D-29 and C4.
- Direct versions are fixed to the v1 evidence stack: PyTorch 2.8.0, torchvision 0.23.0, ONNX
  1.19.1, ONNX Runtime 1.19.2, NumPy 2.0.2, and Pillow 11.3.0.
- A fully resolved, platform-labelled constraints file pins transitive packages as well as direct
  packages. It is generated from a clean environment and reviewed as source.
- `requirements.txt` remains the broad development input; the exact constraints file is the only
  environment accepted for the official v1.1 evidence run.
- A clean environment install plus the complete Python suite is the constraints validation gate.

## Verification profiles are not scientific evidence

- Raw CMake commands remain the CMake 3.16 compatibility contract.
- `CMakePresets.json` requires CMake 3.21 or newer and is an optional convenience adapter.
- The Linux/Python 3.11 CI profile checks portability only; it is not the D-24 macOS
  arm64/Python 3.9.6 reference environment.
- `.github/constraints-ci.txt` pins the direct CI stack. It is distinct from the fully resolved,
  platform-labelled reference lock used for official v1.1 evidence.
- CI and generated zero-weight ONNX fixtures may validate build/runtime mechanics. They never
  replace trained-model accuracy, parity, benchmark, reference-lock, or bundle-integrity gates.

## Modular work packages

### WP0 — Freeze the protocol in tests

**Why:** Test-set leakage and weight/metric mismatch must fail deterministically before training is
expensive.

**How:** Add tests for stratified disjoint splits, stable split digest, training-only augmentation,
earliest-epoch tie selection, selected-state restoration, one-shot test orchestration, and invalid
configurations.

**Exit gate:** All v1 tests remain green and the new protocol tests fail against the old training
workflow but pass against the implemented protocol.

**Rollback/revisit:** Keep the v1 training entry point usable until schema-v2 tests are green. If a
stratified split requires excessive coupling to torchvision internals, isolate label extraction in
a small adapter rather than weakening split invariants.

### WP1 — Implement protocol values and checkpoint v2

**Why:** Accuracy evidence must identify the exact selected weights and the data-selection policy.

**How:** Add the value objects/functions above, refactor training into validation selection followed
by one official test evaluation, and extend `load_checkpoint` with explicit raw/v1/v2 handling.

**Exit gate:** Synthetic training proves the stored state is the best validation state, schema-v2
metrics describe that state, raw/v1 export compatibility passes, and unknown schema versions fail.

**Rollback/revisit:** Revert to the `v1.0.0` training code if compatibility fails; do not write a
partially defined schema v2. Schema changes after a local bundle exists require version 3.

### WP2 — Create and verify exact reference constraints

**Why:** Minimum versions are insufficient provenance for a reference run.

**How:** Resolve a platform-labelled constraints file from a clean Python 3.9.6 macOS arm64
environment, pin all packages, install it into a second clean environment, and run the Python suite.

**Exit gate:** The clean install resolves without manual package changes, reports the expected
direct versions, and passes every non-network Python test including export/parity.

**Rollback/revisit:** If a single lock is not portable, keep the macOS arm64 reference lock and
state its scope. Add another platform only through a separate validated lock.

### WP3 — Implement the local evidence bundle

**Why:** Raw files and console logs can drift or be confused across training runs.

**How:** Add explicit local orchestration that writes into a temporary directory, validates all
metrics/contracts, computes hashes, writes `SHA256SUMS` and `manifest.json`, then atomically renames
the completed bundle into `artifacts/v1.1/`.

**Exit gate:** Verification succeeds for an intact synthetic bundle and fails for a modified,
missing, extra, truncated, or path-escaping file. CMake configure/build performs no network
acquisition or external artifact generation. Explicit CTest may generate deterministic synthetic
fixtures solely under the build tree; those fixtures are never reference-model evidence.

**Rollback/revisit:** A failed build leaves no directory that looks complete. Delete the local
bundle and regenerate; never edit a completed manifest or replace a file without rebuilding hashes.

### WP4 — Run the official local training and model gates

**Why:** V1.1 exists to produce credible model evidence, not another random-weight integration run.

**How:** Install the exact constraints, run the frozen 20-epoch CPU seed/configuration, select on
validation, evaluate test once, export the selected checkpoint, validate ONNX structure, and run
deterministic batch-two parity.

**Exit gate:** Test top-1 is at least 65%; export has named dynamic-batch endpoints; maximum absolute
logit difference is below `1e-4`; top classes match; checkpoint and ONNX hashes enter the bundle.

**Rollback/revisit:** If accuracy misses the floor or any integrity/parity check fails, do not call
the run v1.1 evidence. Preserve failure diagnostics outside the completed-bundle path, revise only
with validation data, freeze a new protocol, and rerun.

### WP5 — Run local C++ prediction and benchmark evidence

**Why:** The selected trained graph must cross the same real C++ boundary validated by v1.

**How:** Configure a Release ORT build with explicit local `ONNXRUNTIME_ROOT` and
`CPP_ML_TEST_MODEL`, run all CTest entries, predict a locally generated PPM smoke input, and capture
the existing preprocessing/runtime/in-memory-pipeline benchmark with its environment header.

**Exit gate:** Default and ORT-enabled suites pass; output probabilities are finite; benchmark
mean/p50/p95/throughput fields are present; exact commands and output enter the local bundle.

**Rollback/revisit:** A C++ failure blocks v1.1 even when Python parity passes. Do not modify the
frozen model contract to accommodate a malformed artifact; diagnose export/runtime metadata first.

### WP6 — Reconcile local release evidence

**Why:** Code, measured claims, and the decision record must agree before changing version status.

**How:** Audit the completed local manifest, commands, test output, accuracy, parity, and benchmark;
then update version/docs and add a post-evidence introspection checkpoint. Keep artifact paths local
and omit upload/download claims.

**Exit gate:** Every number in documentation maps to a verified local manifest field or captured
output; no artifact is committed or published; the tree is clean after source/doc changes; the
project version moves to 1.1 only after Gates A–G pass.

**Rollback/revisit:** If evidence cannot be reproduced from the documented local workflow, retain
version 1.0 and mark v1.1 incomplete. A corrected completed bundle is a new run directory, never a
silent mutation.

## Exact acceptance gates

### Gate A — v1 regression safety

- Clean default Release configure/build/CTest passes with no ORT/model/network requirement.
- ORT-enabled CTest passes both the generated valid/adversarial fixture suite and the external
  `CPP_ML_TEST_MODEL` path using the selected local trained graph.
- C++ public interfaces and the frozen model/preprocessing contract do not change.
- Raw minimum-CMake commands pass; presets remain optional convenience paths.

### Gate B — evidence protocol

- Deterministic stratified 45,000/5,000 split with recorded digest and zero overlap.
- Training selection uses validation only; official test evaluation occurs once after restoration.
- Schema-v2 checkpoint metrics correspond to its stored selected weights.

### Gate C — reference environment

- Fresh Python 3.9.6 macOS arm64 environment installs from the exact constraints.
- Installed versions are captured and all Python tests pass without dependency skips.
- The Linux/Python 3.11 CI profile does not satisfy this reference-environment gate.

### Gate D — model quality

- Selected-model CIFAR-10 test top-1 is at least 65%.
- Test value is finite, reproducibly computed, and stored in checkpoint plus manifest.

### Gate E — export and parity

- ONNX passes structural validation and has dynamic `input`/`logits` batch axes.
- Batch-two max absolute logit difference is `< 1e-4` and top classes match.

### Gate F — local artifact integrity and C++ use

- All evidence-bundle file sizes and SHA-256 hashes verify.
- The trained ONNX graph passes real C++ CLI and model-backed benchmark tests.
- Bundle creation/acquisition is explicit; CMake performs no download.

### Gate G — claim discipline

- No unrun result is documented as evidence.
- No artifact is uploaded, committed, or described as externally available.
- Accuracy and benchmark claims state local platform, artifact hash, and exact command.
- Unexecuted workflow definitions and unrecorded local tests are reported as intent, not evidence.
- A post-evidence introspection checkpoint closes or defers every open gate.

## Risks and controls

| Risk | Control | Rollback/revisit trigger |
|---|---|---|
| Test leakage inflates accuracy | Separate validation selection from one-shot test evaluation | Any test-loader use inside training/selection invalidates the run |
| Saved weights differ from measured weights | Restore selected state before test/export; hash checkpoint | Recomputed accuracy mismatch invalidates bundle |
| Training nondeterminism | Exact stack, seeds, split digest, device capture | Cross-run drift is reported; no bit-identical cross-platform claim |
| Accuracy misses 65% | Tune only with validation before freezing official protocol | Do not lower floor after test observation |
| Environment resolution drifts | Fully resolved platform-labelled constraints | Resolver changes a pin or clean install needs manual repair |
| Bundle is partially written or tampered | Temporary build, atomic rename, SHA-256 verification | Any missing/extra/hash-mismatched file invalidates bundle |
| Local artifact is mistaken for published | Ignored local directory and explicit documentation | Any upload/link requires a new licensing/publication decision |
| Scope expands into optimization or serving | Preserve C++ APIs and fixed v1.1 non-goals | New consumer requirement starts a separate decision/release |

## Introspection cadence

- **C4 — protocol checkpoint:** after WP0–WP2, record split/schema/constraints test evidence and
  distinguish locally executed checks from remotely executed workflow evidence.
- **C5 — evidence checkpoint:** after WP3–WP5, record actual local accuracy, parity, hashes, C++
  runtime result, and benchmark capture without yet changing release status.
- **C6 — readiness checkpoint:** after WP6, reconcile all claims and either mark v1.1 ready or keep
  it incomplete with named open gates.

Gates A–G have executed evidence in the scopes recorded at C5/C6. Accepted design decisions and
this plan did not substitute for those commands or results.
