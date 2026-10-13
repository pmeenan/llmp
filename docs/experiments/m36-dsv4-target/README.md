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
Broader same-quantized-reference quality, durable restart, selected-only expert paging,
production serving, throughput and matched family speed/swap gates remain open.
Family quality qualification uses a reference implementation of the same
quantized checkpoint; bit equality is diagnostic, not its quality criterion.
The original source-precision model is absent from inspected local/NAS catalog
evidence, limiting original-agreement claims without blocking that baseline.

The separate [composed DSpark screen](dspark.md) now passes its bounded fit,
complete-proposal and combined snapshot controls. It does not widen this
target-only report's scope or close the remaining family gates.

## Same-quantized-checkpoint scoring

2026-10-13, Spark A. A closed, target-only Score domain uses the existing
ComponentLlmBuild, EngineLlm, ModelRunner and CorpusScore path. It keeps the
mapping's arithmetic policy and ordinary complete expert closure. Five added
Score shapes are independently measured alongside the three existing setup
shapes; the receipt records every shape field, source ShapeParameters and
selected implementation index. Only the three actual scored classes can enter
the plan cache; the other five are measurement-only.

Both implementations score the same 257 IDs: BOS plus 256 tokens from
`docs/async-model.md` at `e7973e5`. Loss coordinates are exactly 1–256;
the 257th complete finite head is terminal evidence, not another loss.
The complete source text has SHA-256
`772a8b2d503df4fefd1a0a974ecae3ad58f8a4d1f2940fbc6bd440b5c86fb431`;
the 1,028-byte little-endian ID file has SHA-256
`90f0132b81da76ac0102642fb8b8ca03a7e125788dda67a51d7427989df2cbb6`.
The native tokenizer independently validates those captured IDs.

The reference is llama.cpp `d81235049384534c167caea52b85a694f6103d14`
(v0.6.0), image digest
`sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`,
using the same UD-Q2_K_XL checkpoint at
`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93fb787c21338159b0af3318bb3f4d9768`.
It keeps default fusion, F16 K/V cache, flash attention, context 3072,
128-row batches, one sequence and no MTP. Native waves are also 128/128/1;
the actual compressed/read-state parameters are retained in the receipt.
The historical forced-unfused oracle is not this quality baseline.

| Implementation | Mean NLL | Perplexity |
| --- | ---: | ---: |
| Current fused llama.cpp reference | 3.728920455361628 | 41.634137984304 |
| Shared native target | 3.741492717073347 | 42.160877478476 |

Native minus reference mean NLL is +0.012572261712; relative PPL
is +1.265162%. Of 256 coordinates,
126 have lower native loss, 130
higher and 0 equal. The largest absolute coordinate
loss difference is 1.000091895. No threshold or general
quality/adoption claim was chosen from these observations.

Full native build and all 36 selected host controls pass without skips;
the raw manual native case and separate reference case both pass, with 257 finite
heads and all 256 losses. Guards, zero memory-limit/OOM events, complete child
retirement and empty final GPU process lists pass. The native process/cache
peak is 1,485,312,000 bytes. Measured retained-plan
payment is 1,505,922,656 bytes; its additional
admitted catalog extent is 939,524,096
bytes. The actual complete setup footprint is
109,492,333,136 bytes. These are bounded resource
observations, not comparative serving latency or device-residency measurements.

Native build receipt SHA-256: `dffc378c2690c17bb3ca4b66eb507f74ecea2f53ba33667eddf140d3b4b5ffb4`.
Native offline audit receipt SHA-256: `6ab4b78def95361d513e714e4880df14c5de463043797288ccc252f3ae931d10`.
Original runtime wrapper receipt SHA-256:
`e69ee4156bba2b0d60b880f289a496dc873517b73b7702a29034201d9490d541` (preserved reporting-only FAIL).
Reference actual receipt SHA-256: `dd87393d19bbd5a37394ac12ba973591271cf15663bcb6f330ff31843a44f0ed`.
The [aggregate comparison](score-comparison.json) retains paired-coordinate
summaries and receipt provenance; raw losses, logs and XML remain in scratch.
The original native wrapper marked FAIL after the native case passed: it
incorrectly required a separate physical lowering for the second 128-row wave.
The offline audit verifies that its runtime past128 reuses the past0 plan with
identical complete key dimensions, source ShapeParameters, selected sequence
and payment. Both dynamic past values and every target coordinate remain
checked. The original failure receipt is preserved; inference was not repeated
for this reporting correction. Retained-plan payment conservatively counts
three schedule records plus one replacement although two share a plan.
Final repository quality checks are reported with the landing validation.


The score path retains the existing 110 GiB node cap and 6 GiB physical
margin. The fixed compiler/source/scorer categories occupy 3,727,895,504
bytes within the existing 4 GiB construction allowance, leaving 567,071,792
bytes of credit inside the factory’s already-held complete charge. The
diagnostic checks that charge without charging the same bytes twice.
The three measured program/capture costs
plus one maximum replacement determine the retained-plan payment. Any excess
over that credit is separately admitted, including its owner overhead and
page rounding, before workspace or weight payload. Bound programs cannot
exceed their measured host/launch limits. The receipt reports the actual
retained total and extra extent; the node-fit calculation includes both.
The common scorer's host bound is 266,335,248 bytes. Source-derived receipt
storage fits the existing 32 MiB caller allowance. Native process and page-cache
confinement is 116 GiB with zero swap; the separate reference uses 110 GiB
with zero swap. Neither cgroup peak measures device residency.

### Manual replay

The built `dsv4_engine_target_gpu_test` exposes
`Dsv4CorpusScoreManual.Natural257` as a manually supervised diagnostic,
with no automatic captured-input CTest registration. Supply
`LLMP_DSV4_TARGET_STORE`, `LLMP_DSV4_CORPUS_TEXT`, `LLMP_DSV4_CORPUS_IDS`,
`LLMP_DSV4_CORPUS_SHA256` and a fresh `LLMP_TEST_SCRATCH`. Use the installed
exclusive Spark supervisor with the declared memory, file, output and
retirement guards; the 512 MiB spill-file bound is separate from the 8 MiB
log/XML bound.

[collect_reference.cc](collect_reference.cc) is a formatting-only copy of the
executed external collector, linked only against the pinned reference. Its `prepare` mode
reads vocabulary metadata and writes the fixed IDs; `score` revalidates those
IDs, uses the current default fused reference and publishes all 256 losses.
Arguments are `prepare FIRST_SHARD TEXT FRESH_OUTPUT` and
`score FIRST_SHARD TEXT FRESH_OUTPUT CAPTURED_IDS`. Compile and execute it
under the same qualified toolchain/reference closure and supervised resource
bounds. It is not linked into llmpalooza or used for native dispatch.

This is one short, fixed natural history. It does not qualify longer sparse
attention histories, DSpark quality, durable restart, production serving or
family speed/swap gates. Agreement with an unavailable original BF16 model
is not inferred; the quality baseline here is the same quantized checkpoint.
