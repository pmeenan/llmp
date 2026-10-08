<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 paired-query sparse-attention transfer

Sharing two neighboring queries' selected KV rows does not improve the
tested Qwen3.8 prefill operator. The complete shared path takes
49.7–90.7% longer warm and 56.1–82.5% longer with cache displacement.
The native path and a paired-CTA separate-list control reproduce every
captured output byte. The shared path passes the fixed operator bounds,
but its speed result rules out a model trial of this implementation.
The prototype and capture/replay code were removed; production is unchanged.
This result applies to this scheduling transfer, not every possible query tile.

## Matched operands and charged paths

The native Qwen3.8 device-selected QSA already uses F32 QK, softmax and
weighted-value accumulators, F16 KV storage, scaled F32-to-F16 queries,
and F16 probabilities. Its 12 query heads per KV head reuse each gathered
row within a token. The missing transfer tested here is sharing gathered
rows between two tokens; no cache compression or attention selection changes
were included.

Fresh ordinary native model processes captured the final 4,096-row chunks
at 32,768 and 126,976 prompt tokens, with context capacities 32,768 and
131,072. Layers 3, 23 and 47 retained their exact F32 queries, F16 K/V
caches and strides, canonical I32 selections and F32 outputs. The source
prompt contains 128,799 real token IDs; both rungs use exact prefixes.
Capture readbacks are diagnostic and are not throughput measurements.

The standalone replay compares three paths on identical operands:

1. Unchanged one-warp native QSA, one graph node.
2. Two-warps-per-CTA control with separate original lists, one graph node.
3. Two-warps-per-CTA union and shared KV staging, two graph nodes, charging
   both list construction and attention.

Both paired paths use the same shared-memory capacity. The union retains
each token's original membership and masks cells selected only by its
neighbor. It changes 16-cell softmax stage boundaries, so it can change
bits while keeping the original types and per-dot arithmetic. Odd/small
or unsupported shapes retain the original kernel; malformed lists retain
separate-list behavior. No graph builder or serving plan selected the prototype.

## Measured results

Spark `spark-c4e2` / NVIDIA GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1` (CUDA 13.4.92, LLVM 22.1.8), 2026-09-30.
Each cell is the median of nine alternating samples per arm after two
warmups. Warm samples prime the same graph outside the events. Displaced
samples write 512 MiB outside the events. Two rotating operand banks put
the selected KV working set above twice the measured 24 MiB L2 capacity;
per-bank selected bytes are 66.3–67.0 MB at 32K and 200.1–206.8 MB near 128K.
Validation, allocation, readback and displacement are excluded from the
operator events. Every raw sample retains its arm, order and bank.

| Prompt tokens | Layer | Warm native / separate / shared ms | Warm native/shared | Displaced native / separate / shared ms | Displaced native/shared |
| --- | --- | --- | --- | --- | --- |
| 32,768 | 3 | 14.673 / 15.390 / 26.071 | 0.563× | 14.749 / 16.039 / 25.730 | 0.573× |
| 32,768 | 23 | 14.221 / 15.864 / 27.116 | 0.524× | 14.846 / 15.901 / 26.970 | 0.550× |
| 32,768 | 47 | 14.801 / 16.110 / 27.186 | 0.544× | 15.016 / 16.129 / 27.410 | 0.548× |
| 126,976 | 3 | 18.506 / 19.210 / 27.709 | 0.668× | 18.024 / 19.872 / 28.135 | 0.641× |
| 126,976 | 23 | 16.074 / 17.422 / 28.715 | 0.560× | 16.174 / 17.411 / 29.400 | 0.550× |
| 126,976 | 47 | 15.362 / 17.270 / 28.695 | 0.535× | 15.968 / 17.175 / 28.493 | 0.560× |

The separate-list paired control also regresses. Median neighboring-list
overlap is 1,147–1,339 cells; the union is 2,763–2,956 cells, versus about
2,051 selected cells per token. The union saves gathered rows but makes
each query process more MMA/softmax stages, including masked cells. This
extra work and pair synchronization are source-backed explanations to test,
not a measured profiler attribution. The result does not justify a 256K
operator rung or an end-to-end candidate.

Record/count scratch peaks at 67,641,344 bytes. The replay's total device
allocation, including two input banks, 128 MiB workspace and 512 MiB
displacement buffer, is 1,275,592,704 bytes at 32K and 1,678,245,888 near
128K. These are diagnostic replay allocations, not a serving memory claim.

## Accuracy and reusable pieces

All six cases pass full native-capture byte equality, full separate-control
byte equality, shared eager repeat equality, eager-versus-graph equality,
and changed-query/changed-selection graph controls. Shared versus native
NMSE is 2.82e−11–1.64e−10 against the predeclared 1e−5 bound.
Independently fixed 96-row/head probes per case use full FP64 dot,
softmax and weighted-value references. With native rounded queries,
shared NMSE is 4.44e−10–1.48e−9 and native NMSE 4.82e−10–1.44e−9;
both pass the same 1e−5 bound. Unrounded-query controls are also retained.
These are operator checks, not full-model quality results.

The useful pieces are completion-aware native operand capture, complete
shape/stride/file-identity validation before allocation, unchanged native
byte controls, fixed FP64 probes, distinct launch/occupancy and sharing
controls, bounded union records and charged preparation, current-operand
graph checks, and selected-working-set/cache-displacement controls.
Qwen decode/verify is outside the eligible T128–4,096 prefill shapes, and
its measured QSA share of the long-context MTP step is only about 1.3%
([MTP report](../qwen38-mtp-speed/README.md)); this transfer was a prefill
experiment and does not explain the MTP speed gap.

## Provenance and verification

Base `9be3507`; target artifact/manifest
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`;
canonical I32 prompt
`c9d455d1afc75e1fd4e0f93d0312f1768196609fab8a5836f57c15a72e2e3661`.
Capture metadata hashes are
`86d92717d34426510512d789c02175123732f661cdd800f957c95f003d8745d9`
and `9e2f226b5424159e70096660a55914fc1fbf2f56d0a6a6d43a82e3f4d08b73e5`.

The expanded final Spark check passed prepare, the locked build and all
1,044 tests (205 GPU), SDK formatting/tidy, boundaries and copied-SDK
REUSE/header checks. The checked-source receipt SHA-256 is
`d43558b0017ade60b6ac347be81bd8120e7adaa9f5a4add7db926e131577b9a7`;
capture executable `2f26c7ab366ea288ace9631b9dc95272196e3a80ac109fd56f15893311ef5d50`;
replay executable `ef732654545fcdd30b334a8b8e640077db07849ae5edc3ab8c1fc8c1843ad261`.
Both native captures and all six replays qualified, with a fresh strong
preflight and successful retirement probe around each process. Capture
receipt `47c2aed4ce5c7fd497a806190a9722eee09ff1cf462e80cb02760813bf99a8f1`;
complete replay receipt
`420d79ab8b1dd5e767fa8a582dd8cc6931f04911a215bc89ce83f3b80f751129`.

Raw source snapshots, checked receipts, controllers, captures and 324
measured timing samples remain outside Git on `spark` under
`~/scratch/m3-attention-qsa/`. The complete checked source copy is
`check-capture-r3/headers-tree`. Final six-file sources, complete patches,
controllers/manifest and copied replay logs are also retained on the
workstation under `~/scratch/m3-attention-qsa-final/`. The removed
prototype is reproducible from these receipts; no rejected numerical
implementation enters production. No workstation build or test ran.
