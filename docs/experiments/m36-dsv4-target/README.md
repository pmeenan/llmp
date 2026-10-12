<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete DeepSeek target execution screen

2026-10-12, Spark B, main candidate based on `a441cf13`. The fixed prepared
GGUF target executes through ComponentLlmBuild, EngineLlm and ModelRunner,
with its actual token-hash rows, compressor inputs and ordinary complete
expert closure. Production serving remains unopted.

Prepared artifact SHA-256:
`8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`.
The envelope is one owner, context 3072, chunk 512, Prompt/Decode 512/1,
without a drafter or speculation. All 256 experts remain represented.
Source weight extents total 100,032,053,248 bytes and state layout is
480,247,808 bytes. Actual setup pays 106,614,572,256 bytes under the
unchanged 110 GiB cap; admission also preserves the 6 GiB physical margin.
The reported ordinary dynamic budget is 122,833,316,608 bytes, not a larger
permission to bypass the cap.

Construction is paid before artifact parsing. The graph reader reuses unused
space in the existing compiler allowance: 437,958,768 bytes, with complete
construction accounting 3,454,776,256 bytes below the 4 GiB allowance.
The validator observes actual lowering before binding and records the graph
and ordered selected nodes. State-only prefill legitimately retains 126 expert
arrays/products; headed steps retain 129. A full 43-layer host control verifies
that omitting the dead final products preserves every persistent state write.
No arithmetic policy or parity rounding is adopted.

## Observed controls

The full native build and all 307 selected host controls pass without skips in
`deepseek-target-bridge-main05`. The actual registration
`Dsv4EngineTargetGpu.FullTargetFiniteHeadsAndSnapshotContinuation` passes in
`deepseek-target-bridge-gpu04`:

- Five 512-row prompt chunks initialize prefix 2560. Two generation runs each
  take eight decode steps and expose nine complete head rows, for 18 finite
  rows of width 129,280.
- Snapshot restore checks all initialized state and nine complete heads
  (1,163,520 values); both runs reach frontier 2568.
- Nine selected programs, five with heads, pass the source/lowering checks.

The case takes 28.516 seconds including setup/storage. This is not a serving
performance measurement. Source, binary, artifact-metadata and recipe guards
pass before and after. The child process group retires, no GPU process remains,
and all memory limit/OOM event counters stay zero. The cgroup process/page-cache
peak is 2,463,272,960 bytes; it is not total GPU residency.

Initialized-state SHA-256:
`cfa4b5626de7c44bdeaae14398468aa630e04eaaac917c148376c5b9b4b8124b`.
Selected-plan receipt SHA-256:
`62ecd61a5dc74361a87a918a8914601c6c3af478b012fa35b0523051f23810f0`.
Native recipe SHA-256:
`4cd22b324c7da85afd2773d1b8067cc858c2fa95b3c7d1f95c97222d319b942d`.
Actual recipe manifest SHA-256:
`1be4c1bd9fe25d1f7b5b4b5b9bdd5340d8d933b442ac136b5544de0a15735327`.
Raw logs/XML stay outside Git in session scratch.

Earlier screens refused before execution on reader capacity, physical headroom,
and a validator that incorrectly required headed expert counts for no-head
prefill. The corrected source preserves the cap, margin and dead-work omission.

This qualifies the bounded target factory and same-implementation continuation.
DSpark, original BF16 quality, durable restart, selected-only expert paging,
production serving, throughput and matched family speed/swap gates remain open.
The original source-precision model is absent from inspected local/NAS catalog
evidence; quantized-library bit equality does not replace that reference.
