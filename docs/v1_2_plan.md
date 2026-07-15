# Version 1.2 plan — measured runtime batching

## Status

Pre-implementation contract. No v1.2 batching code or performance result exists yet. Work starts
only after the v1.1 source tag and its local r2 evidence bundle are complete.

## Goal

V1.2 tests one narrow hypothesis: the v1.1 dynamic-batch ONNX graph can process a batch of eight
prepared CIFAR-10 tensors with materially higher item throughput than eight batch-one calls on the
same local reference machine.

The experiment is useful even if the hypothesis fails. A negative result is recorded and the
runtime change is rolled back rather than hidden or optimized after the fact.

## Non-goals

- Multi-image CLI input, public pipeline batching, request queues, a server, or concurrency.
- Batched file loading, image decoding, resizing, or softmax decoding.
- Buffer reuse, allocator customization, thread tuning, graph-level tuning, quantization, GPU
  providers, or selecting a favorable batch size after measurement.
- A generic model/shape API or another polymorphic interface.
- A portable performance claim, published model, or remotely executed CI claim.

## Smallest design change

- Add the runtime tensor shape to `ModelOutput` so output ownership and dimensions travel together.
- Keep `IInferenceBackend::run(const Tensor&) -> ModelOutput` unchanged.
- Permit validated input `[N,3,32,32]` and require output `[N,10]`, with `N` capped at 256 to bound
  multiplication and allocation.
- Preserve batch-one `InferencePipeline`, CLI, preprocessing, and decoder behavior without adding
  overloads.
- Add a paired runtime benchmark for serial-eight (eight batch-one calls) versus one batch-eight
  call. Construct both prepared inputs from the same already-normalized tensor rows.
- Report group latency, amortized milliseconds/item, and items/second. Amortized time is never
  described as the latency experienced by an individual item.

This follows composition over inheritance: the existing runtime seam already represents tensor
execution, so batching changes validated values and adapter behavior rather than adding a new class
hierarchy.

## Correctness gates

1. Every v1.1 source, Python, sanitizer, generated-hostile-model, trained-model ORT, lightweight
   bundle, and deep evidence gate remains green.
2. Batch-one CLI and pipeline output remain unchanged.
3. Batch sizes zero, negative/dynamic-at-runtime metadata misuse, values above 256, shape/buffer
   overflow, output batch mismatch, wrong class count, and non-finite logits fail clearly.
4. A fixed-batch-one graph remains accepted for batch one and fails before execution for batch
   eight. Mixed dynamic/fixed axes and fixed batch sizes other than one fail at construction.
5. A trained-model batch of eight returns shape `[8,10]` and 80 finite logits in documented
   row-major order.
6. Each batch row matches an independent call for the same tensor with maximum absolute-logit
   difference at most `1e-5` and identical top class. PyTorch-to-ORT batch-eight parity separately
   remains below `1e-4` with matching classes.
7. Address/undefined-behavior sanitizers pass all portable batch arithmetic and fake-backend tests.

## Frozen performance experiment

- Artifact: the exact ONNX SHA-256 accepted by the v1.1 r2 bundle.
- Build/runtime: Release, AppleClang, ONNX Runtime 1.19.2 CPU, same recorded Apple reference
  machine and operating system.
- Inputs: the same prepared tensor values for both modes; session construction and file I/O remain
  outside measurement.
- Candidate: batch 8. Batch 2/4/16 are not measured and then selected.
- Sampling: exactly 10 independent paired process runs, alternating serial/batch order; each
  process uses identical warm-up and measured iteration counts and persistent identically
  configured sessions.
- Work accounting: every serial sample performs eight batch-one calls and every batch sample
  performs one batch-eight call over the same tensor rows.
- Primary metric: median items/second speedup across paired process runs.
- Stability metric: at least eight of ten runs must favor batch eight.
- Tail metric: batch-eight group p95 must not exceed serial-eight group p95. This compares the time
  to finish the same eight-item workload, not a fictional per-item latency.

## Acceptance and rollback

Keep the v1.2 runtime change only if all correctness gates pass, batch eight improves median
items/second by at least 50% over serial-eight, at least eight of ten paired runs favor batching,
and batch-eight group p95 does not exceed serial-eight group p95.

If any condition fails, revert the runtime/API change, preserve the measured negative experiment
in the decision record, and leave the source version at v1.1. A different batch size, threading
policy, or optimization technique requires a new pre-registered decision before measurement.

## Why this precedes serving

A server would add request parsing, trust limits, concurrency, queueing, cancellation, and
deployment policy at once. This slice first establishes whether the underlying model/runtime has
useful batch economics. That evidence can later inform—rather than merely decorate—a serving
design.
