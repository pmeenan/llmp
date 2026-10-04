<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 model foundation

`src/model/gemma4.h` describes the approved 26B-A4B and 31B text models:
fixed profiles, strict GGML artifact bindings, bounded state layouts,
initialized read/write footprints and independent request-segment inputs.
This foundation supplies no importer, execution graph, kernel selection,
runner, assistant or inference support. Both checkpoints remain unsupported
in the [support matrix](model-support.md); the family task remains open.

## Verified contracts

The profiles and tensor roles follow llama.cpp
[`b29c606e:src/models/gemma4.cpp`](https://github.com/ggml-org/llama.cpp/blob/b29c606e/src/models/gemma4.cpp)
and its `llama-kv-cache-iswa.cpp`. Immutable test fixtures preserve all text
architecture keys, tensor names/shapes/types and the 256 global RoPE factors
from these approved files. Fixture source identities include the upstream complete-file SHA-256,
captured 16 MiB prefix SHA-256 and header-through-tensor-directory SHA-256;
tests require no weights or network access. The prefix captures include the
small F32 norm/frequency-factor payloads preceding the embedding weights.

| Contract | 26B-A4B | 31B dense |
| --- | ---: | ---: |
| GGUF repository | `unsloth/gemma-4-26B-A4B-it-GGUF` | `unsloth/gemma-4-31B-it-GGUF` |
| Revision | `c099eb48e663fd284577b04978a94ffccb261841` | `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` |
| File | `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` | `gemma-4-31B-it-UD-Q4_K_XL.gguf` |
| Header through tensor directory, bytes | 15,824,659 | 15,834,157 |
| Tensor count | 658 | 833 |
| Layers / hidden width / query heads | 30 / 2,816 / 16 | 60 / 5,376 / 32 |
| Local KV heads / global KV heads | 8 / 2 | 16 / 4 |
| Local head dimension / global head dimension | 256 / 512 | 256 / 512 |
| Dense FFN width | 2,112 | 21,504 |
| Routed experts / selected / FFN width | 128 / 8 / 704 | none |

Both have five local layers then one global, a 1,024-token sliding window,
context and vocabulary 262,144, RMS epsilon 1e-6, and final softcap 30.
There are no shared KV layers or per-layer input embeddings in either file.
The 31B contract was extracted from a verified 16 MiB HTTP range response
at its pinned revision, with no full weight download. The 26B contract was
read from the existing pinned reference file on Spark. Tests compare every
profile field/layer pattern and bind every actual tensor descriptor.

The global GGUF rotary count is **512**, with shared F32
`rope_freqs.weight[256]`: the first 64 factors are one and the remaining
192 are approximately 1e30. This encodes proportional RoPE while preserving
the complete head's rotary layout. A quarter-rotation prose description
must not become `n_rot=128`. Local rotary count is 256, with base 1e4;
global base is 1e6. Binding checks shape/type and a separate factor check
refuses nonpositive or nonfinite values before a future graph uses them.

The head aliases the input embedding. Globals use K's projection as the
unnormalized V producer; the learned K norm and rotation happen separately
from V's unweighted RMSNorm. K and V consequently require distinct cache
storage. Attention scale is **1.0**, without inverse-square-root scaling.
Embeddings scale by √hidden-width. Each attention output passes its sandwich
norm before residual addition; FFNs use GeGLU, sandwich norms and the
per-layer scalar. On 26B the dense FFN runs beside routed experts, each with
its own post norm. Routing uses the attention residual's unweighted RMSNorm,
scales by 1/√hidden-width and the learned router input scale, then softmax,
top-eight selection/normalization and per-expert down scaling. These are
future graph obligations, not executed behavior of this foundation.

The actual 26B mix is F32 ×392, Q8_0 ×207, Q4_K ×30 and Q5_1 ×29. Every
expert gate/up array is fused Q4_K `[2816,1408,128]`; down arrays are
Q5_1 `[704,2816,128]` on layers 0–28 and Q8_0 on layer 29. The actual 31B
mix is F32 ×422, Q4_K ×304, Q5_K ×37 and Q6_K ×70. Binding recognizes
well-formed GGML floating/weight representations, with exact role shapes,
block geometry and sufficient readable bytes, including the GGML final-row
512-element padding over-read; this does not certify a
kernel for every recognized type. Integer matrices, foreign representations,
unread or duplicate roles and ambiguous fused/split arrays are refused.
Prepared expert arrays bind one expert slice and its count separately, with
256-byte member alignment for their group offsets.

## State and host inputs

Each request slot gets the same independent virtual layout. Each layer's
F16 K and V cache starts on a 2 MiB boundary. Global caches append through
padded context; local caches have
`pad(min(context, window + max_rows), 256)` cells. The additional chunk room
preserves every query's window when the complete chunk is written before
attention starts. Absolute position modulo local capacity selects its ring
cell. Public layouts, bounds and read alignment are checked before returning
indices or footprints. Arithmetic stays within the verified profile/context
bounds, including the full 262K virtual layout.

`Gemma4UsedState` returns nothing at zero positions, then the padded cells
actually read, capped at each tensor's capacity. A runner must materialize
and zero those ranges before dispatch; a reservation does not initialize
them. `Gemma4ChunkWrites` names only written rows and splits ring wraparound.
Global representations allow truncation; local rings advertise append only.
Speculation needs separately qualified snapshots of overwritten ring rows,
not an unsupported arbitrary rollback claim.

A chunk concatenates bounded segments with unique slot IDs and independent
past positions. Each segment retains its own global/local cell indices,
read widths, masks and output-row span; positions restart independently and
outputs refer to the flattened rows. Compatible products can later join
these rows without joining caches or nonlinear state. Total rows are bounded
by the layout's chunk limit and slots by sixteen.

Default host inputs contain O(rows + slots) descriptors; their storage does
not grow with attended context. Optional F16 reference masks follow causal
and window visibility, including ring wraparound and padded future cells.
Their byte stride is checked against the GGML I32 limit before allocation.
`Gemma4HostInputBytes` includes heap buffers and the segment descriptor array.
It is a storage envelope, not a memory grant: the future caller must fund
it before invoking the builder and register it in the engine's measured
host-input/staging budget. The builder checks actual vector capacities
against the pinned libstdc++ empty-vector reserve/resize contract. This does
not admit an otherwise unfunded multi-gigabyte reference mask.

CPU tests cover both actual contracts and refusals, tied identities,
full-context virtual bounds, initialized/write footprints, slot isolation,
solo/batch mask equality and fake cache results against unrolled causal/window
references. These validate the foundation, without executing model arithmetic.

## Required execution and optimization qualification

Every family/quant must adopt applicable selected Qwen/DeepSeek techniques
and qualify optimized batching before supported status. The
[optimization inventory](optimization-inventory.md) remains authoritative.
For these actual GGUF files the next slices owe:

| Transfer or contract | Eligibility and required qualification |
| --- | --- |
| Primitive completeness | Add/qualify GeGLU and GELU-tanh fallbacks: current ordinary GLU validation selects SwiGLU and unary validation omits GELU, even though fusion recognition mentions GeGLU. Preserve RMSNorm/scale, V norm, sandwich order, softcap/tanh and full-width proportional RoPE. |
| Q5_1 expert down | QRows decoding and MMVQ tuning entries exist, but `VecQType` excludes Q5_1. Qualify a primitive/routed product at K704, N2816, top-eight routes, scalar/prefill/batched rows, and the last-layer Q8_0 variant. Do not select a Q2_K or IQ2 kernel by analogy. |
| Shared input preparation | Reuse eligible Q8_1 preparation across ordinary Q8/K-quant products and fused gate/up reads, retaining maps/strides and Gemma's router and GeGLU arithmetic. DeepSeek's SwiGLU activation writer cannot transfer unchanged. |
| Routed prefill scheduling | Check compact expert-major tiles and full-K arithmetic for Q4_K fused gate/up and Q5_1 down at actual shapes/chunk sizes; keep only qualified speed/memory winners. Raw expert groups must remain authoritative for later paging. |
| Attention and device masks | Compare existing D256/D512 tensor-core attention with each model's GQA, local/global windows, independent caches and scale1.0. Generate causal/window masks from descriptors on device where qualified; never invent sparse global attention. |
| Join products across requests | Apply Qwen/DeepSeek joined dense/routed/head products when operand/quant contracts fit. Preserve per-segment outputs, original one-token sums, stable route pair order and separate attention/state. Qualify scalar versus joined logits/state and departed/cancelled slots. |
| Lanes, graphs and lifetimes | Reuse request cohorts, completion-aware leases, stable-address graphs, charged per-lane scratch and hazard ordering. Shape/read-alignment choices must back padded cells and preserve exact continuation across capture/replay, spill/restore and time-slicing. |
| Bounded state and staging | Use initialized read/write footprints, growing extents, bounded host inputs, shared maximum workspace and separately owned slot state; measure peak memory for solo and batched envelopes at context boundaries. |
| Assistant and draft policy | The KV-sharing Q-only assistant is a later slice, not Qwen MTP or DSpark recurrence. Qualify shared target-cache ownership, centroid/head IDs, overwritten-ring rollback and greedy/sampled acceptance before transferring adaptive depth or selected-head optimizations. |
| EXL3 / other formats | Separate representation binding, packed products, codebook/rate and expert grouping qualification are owed. GGML recognition in this slice establishes no EXL3, NVFP4 or MXFP8 support. |

For every adopted execution path, record actual registry/plan selection,
precision/layout and shape limits, isolated and whole-model timing, solo and
batched quality/exact-state controls and peak memory. No speed or optimized
batching result is claimed here.
