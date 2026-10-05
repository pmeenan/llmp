<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 model and graph foundation

`src/model/gemma4.h` describes the approved 26B-A4B and 31B text models:
fixed profiles, strict GGML artifact bindings, bounded state layouts,
initialized read/write footprints and independent request-segment inputs.
The segmented GGML graph, checked plan adapter and bounded native 26B-A4B/31B
engine runner described below extend this foundation. Both approved profiles
have bounded scalar serving routes; assistants remain unavailable. Both
checkpoints remain unsupported
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
The original 31B contract was extracted from a verified 16 MiB HTTP range response
at its pinned revision. Its complete source and prepared artifact are now
fully verified for the [bounded dense runner](experiments/gemma31-runner/README.md).
The 26B contract was
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
top-eight selection/normalization and per-expert down scaling. The segmented graph preserves these arithmetic contracts; whole-model
execution and reference qualification remain open.

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
outputs refer to the flattened rows. The graph joins compatible products across
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
references. These validate the model foundation independently of the graph tests.

## Activation primitives

The native GGML registry supplies F32 split GeGLU using GELU's tanh
approximation (`ggml.geglu`) and packed F32 unary GELU-tanh (`ggml.unary`).
GeGLU accepts uniform strided input rows, including views of the actual
704-wide expert gate/up layout, and a packed output; exact packed in-place
updates are permitted. Swapped, two-half packed, ERF and quick variants are
refused. CPU checks and Spark controls cover solo/batch equality, expert
views, near-zero and extreme finite inputs, aliases and current bindings.

The separately named `ggml.mul_mat_geglu.mmvf_fused` implementation has
primitive fallback through two products and `ggml.geglu`. It takes one
activation row with compatible floating weights and F32 accumulation;
F16 products must explicitly request F32. The planner's `geglu_fusible`
policy callback defaults off. A [bounded synthetic screen](experiments/gemma-activations/README.md)
at K2816 found a gain at N704 and a loss at N2112; neither result qualifies
a model's selection. Four-row products retain ordinary matrix products and
GeGLU. The approved Q8_0 shared weights and Q4_K expert arrays are not
eligible for this floating MMVF fusion. An explicit `VecQGlu::kGeGlu`
quantized output writer now uses the pinned GELU-tanh, with actual Q4_K
expert and Q8_0 shared-width controls. Its [paid chain screen](experiments/gemma-quant-geglu/README.md)
separates shared preparation from writer fusion; the writer remains off in
model selection. Activation/quantization writeback and model qualification
remain owed.

## Local attention primitives

D256 local attention has checked F32 Q/output and F16 K/V/mask primitives
for exactly two query heads per KV head. Query views are `[D,rows,heads,slots]`,
including a permutation of packed projection rows; caches are
`[D,cells,kv_heads,slots]`. Cells must be padded to a multiple of 256, with
16-byte bases and strides. Scale 1 is preserved; sinks, ALiBi, softcap and
sparse gathers are outside this local contract.

The planner follows the pinned GGML overall selector on GB10: one query row
and one slot use D256 vector attention, while multiple rows or slots use
group2 MMA, with query tiles 4/8/16/32. Each slot's mask/cache remains separate.
CPU planning without a device callback retains group2 as a primitive fallback.
Existing D64 vector and D256/D512 group8 identities remain available. Mask
prepasses need every row of the final query tile and every mask sequence
physically backed; the mask sequence extent must equal the query sequence
extent rather than relying on ordinary extent broadcasting. Scratch plans
include mask scanning, partial results and fixup. Plans refuse shapes beyond
the pinned signed iteration and efficiency arithmetic.

The [bounded attention controls](experiments/gemma-local-attention/README.md)
compare original selected kernels, mathematical ring-mask references and paid
captured execution. They qualify attention primitives; runner cache ownership,
whole-model quality and optimized batching remain open. The separate device-mask controls below
qualify mask production.

## Segmented graph and bound plans

`src/kernels/ggml/gemma4_graph.h` builds complete text chunks for the verified
26B-A4B and 31B bindings. Row-local dense, routed and head products join the
flattened request rows. Each segment keeps separate attention, positions,
masks and F16 K/V caches. Global attention reuses the raw K projection
for V before applying K's learned norm and full 512-dimensional factor RoPE.
Routing retains the selected expert's scale before its weighted contribution
and an ordered eight-term sum. Final attention retains every cache write
when optional frontier narrowing removes unrequested FFN/head rows.

`src/engine/gemma4_plan.h` checks every logical weight/array/slot region before
binding, places activations and selects registered primitive/fused operations.
Activation regions are checked against every supplied immutable weight/array
and retained slot, including storage unused by a diagnostic graph. Active
slots must also remain disjoint from every supplied peer slot and weight.
A prepared expert stride aligns the readable member to the joint 16-byte and
GGML block-size quantum; on-disk 256-byte member alignment alone is insufficient
for Q4_K/Q5_1 executable views. The caller still owns catalog residency,
initialized cache cells, charged activation/workspace/input envelopes and
completion fences. This adapter does not perform admission, paging or serving.

Rows, slot IDs and padded read widths define shape reuse; absolute positions
are fresh data, including at local ring wraps. `Gemma4Sources` authenticates
fresh positions and cache indices, and checks a caller-held host grant.
With `Gemma4GraphOptions::device_masks`, each segment owns one checked F16
causal and one local ring-mask producer reading fresh packed I32 positions.
These outputs are activations reused across layers, rather than host inputs.
Every padded cell/query element is initialized; queries are padded to 32,
including the attention prepass's final partial tile. Local capacity retains
at least `min(context, window + whole_chunk_rows)` cells, preventing later
chunk writes from overwriting a visible earlier query cell.

`Gemma4Sources` omits host masks only after authenticating the graph-owned
producers, their positions source and exact segment parameters. Token,
position, cache-cell and funding checks remain in both modes. Device mode
rejects supplied host mask arrays; diagnostic mode validates every supplied
causal/window bit and constructs funded padded reference masks. The device
mode's host descriptors remain O(rows + slots), with mask outputs charged to
activation storage. The graph option remains explicit and defaults off for
diagnostic callers; the checked native runner chooses device masks by default.
Whole-model qualification remains separate. [Device-mask controls](experiments/gemma-device-masks/README.md)
record exact bytes, fresh captured positions and paid complete-layer comparisons.

Explicit diagnostic options may start from hidden inputs and execute a checked
layer interval without a head, using the same arithmetic builder as a full
chunk. Production defaults require all layers from token input and the tied
softcapped head. `graph.hidden` is the pre-output-norm feature;
`output_normalized` exists on head graphs. A future assistant must consume the
pinned post-final-norm feature and preserve every required row when frontier
narrowing is enabled. No assistant behavior is implemented here.

Forward NEOX RoPE accepts a current packed F32 factor for each rotated pair,
including the verified global `[256]` vector. Ordinary and fused cache-store
implementations retain output/operand alias, stride, span and generation
checks. Frequency factors do not enable backward, non-NEOX or offset variants.
The existing unfactored extended RoPE contracts remain separate.

Shared Q8_1 preparation and row-preserving VecQ products are an explicit
experimental graph option, default off. The option shares preparation among
eligible consumers but does not select a GeGLU quantization writer. Generic
floating GeGLU fusion also remains off. A primitive plan is always available;
selected policies require complete-layer and whole-model measurement before
adoption. D256/GQA2 local and D512/GQA8 global attention use separate checked
registry implementations, with each request's cache independently bound.

The diagnostic primitive baseline leaves nine learned RMSNorm/weight
products unfused per complete local or global layer. Explicit
`DeviceChoices::fuse_norms` binds all nine fusions across one, two and four
independent segments, with synthetic complete-layer and captured-state
controls. This is a diagnostic policy, not an adopted serving default.
The default graph keeps joined K rotation and primitive stores. Explicit
`Gemma4GraphOptions::rope_store` instead rotates each segment's joined,
learned-normalized K view with its own fresh positions and supplies a complete
packed, zero-offset flattening view to the existing factor-aware fused store.
`DeviceChoices::fuse_rope_store` is an independent checked planner gate;
generic graph fusion stays disabled. An ineligible pattern falls back to
ordinary rotation/store, and a diagnostic keep of the rotation or its view
forces its ordinary producer. The option names rotations
`blk.<layer>.slot.<slot>.k_rope`; the default names joined rotation
`blk.<layer>.k_rope`. Current placement still funds rotated intermediate
storage, including when the fused store leaves its bytes unwritten, so
catalog coverage retains every descriptor and source check.
[Bounded complete-layer controls](experiments/gemma-rope-store/README.md)
show exact cache/output agreement but no decisive paid speed gain; the
option remains default off.
Qualified norm/RoPE-store selection and optimized batching remain owed.

## Native 26B-A4B and 31B runner controls

`src/engine/gemma4_runner.h` reuses `RunnerResources`, `PagedWeights`,
`LiveState`, charged plan caches, `GraphRuns` and `RequestCohort`. It binds the
actual prepared artifact, validates funded F32 RoPE factors before execution,
materializes and initializes complete padded KV read ranges, and publishes
only completed chunks. Independent slots preserve peers through clear,
checkpoint, spill/restore and clean capacity refusal. Scalar chunks delegate
to the wave path. Setup measures a shared workspace/input envelope across
slot counts, maximal ragged rows and full read depth; it does not eagerly
materialize every slot's context ceiling.

An engine-only typed variant selects one of the two approved fixed profiles;
the default remains 26B-A4B. Unknown variants refuse before opening artifacts,
and exact artifact binding precedes state allocation. The [dense 31B controls](experiments/gemma31-runner/README.md)
pass ordinary 1/2/4-request same-shape full-head/KV replay and checkpoint/spill
continuations. Source layout IDs preserve the existing 26B tag and reject
cross-variant restore/adoption before mutation. No expert slabs are created
for the dense profile. Its representative 128-row screen fails against
full-fusion llama.cpp: 14.61% higher PPL and 330 strict head argmax differences
outside the new zero native noise bound; the unfused diagnostic matches every
full head byte for byte. This does not qualify either model or identify a
shared cause with the narrower 26B routed-input diagnosis. Optional policies,
optimized batching, assistants and full serving qualification remain open.
The [dense 31B norm control](experiments/gemma31-reference-fusions/README.md)
reproduces every measured stock head with the two missing norm fusion families.
The [checked native norm chains](experiments/gemma-native-norm/README.md) now
reproduce all 1,024 stock heads at the measured dense31 128-row shape. Separate
policies remain default off: the paid 8K/ubatch-256 reference comparison has
head differences and no stable native speed gain. Full model qualification
remains open.

Checked device masks are the runner default after exact full-model agreement
with caller-funded host masks. Norm fusion, shared Q8 preparation and bounded
row-preserving products remain explicit experiments, default off. The
[first complete-model controls](experiments/gemma-runner/README.md) retain the
short reference/noise screen, ordinary joined arithmetic differences, exact
experimental solo/join state and short paid latency. Full weight-closure
acquisition is the current numerical baseline. Resident qualification still
requires selected lanes/optimization dispatch, optimized batching, long context,
complete serving behavior and representative reference quality/performance.
Route-discovered expert paging is an M7 follow-on; authoritative raw expert
groups remain available for it.

The [reference norm-fusion diagnosis](experiments/gemma-reference-fusions/README.md)
checks the two normalization fusion families individually and together.
All repeat exactly, but none resolves the representative quality gap;
no new native policy is selected.
The [stock routing/reduction controls](experiments/gemma-reference-stock/README.md)
show that each specialized family also changes the representative heads.
Checked norm chains and routed contracts have checked planner integration
as explicit default-off policies. The complete 26B quality gate remains owed.

The [short resident diagnosis](experiments/gemma-performance/README.md)
records a 4.33% scalar rate gap against the bookended fusion-enabled reference.
Shared-Q8 preparation changes full heads and initialized KV, and stays off;
norm fusion reproduces the bounded native controls exactly without a measured
whole-model win. The [representative teacher-forced screen](experiments/gemma-quality/README.md)
fails against fusion-enabled llama.cpp: PPL is 10.03% higher and 476 of
1,024 argmax choices differ outside the frozen native noise allowance.
Native matches the unfused 128-row diagnostic control exactly, which narrows
the investigation without qualifying the fastest reference or explaining
every changed operator. Numerical teacher-forcing chunk sizes do not cap
competitive reference prefill. The [paid 8K screen](experiments/gemma-prefill/README.md)
selects reference ubatch 1,024 after testing five sizes: prefill is
6.931 s native versus 2.612 / 2.605 s reference, including the native
128-row envelope and 63 extra intermediate heads. Subsequent 32 fixed-prefix
units take 0.691 s native versus 0.866 / 0.869 s reference. This does not
qualify generated-text equivalence or physical peak memory; no individual
source of the prefill gap is isolated.

## Bounded native serving route

The existing runtime driver registers both approved 26B-A4B and dense31 artifacts
through one adapter in `runtime/serving.cc`. Its factory opens the trusted artifact
and requires complete approved tensor binding before selecting the immutable
engine variant; setup repeats that binding. No configuration schema is added. Chat and literal completions share the native runner,
state skeleton, settings, continuation and restart machinery. Up to twelve
independent owners execute scalar completed units; this is continuous cohort
scheduling, with no selected joined model dispatch. Prefill is capped at 128
rows, context at 262,144, and the fallback settings are explicitly uncalibrated.
Optional norm, routing/reduction, shared-Q8, row-invariant and RoPE/store
policies remain off. Device masks
remain the native default.

Plain chat disables thinking in the actual template. Generated thought and
tool-call parsing and assistants/speculation are explicitly unavailable in
this slice. Template controls deserialize historical tool-call
argument strings once before rendering and compare the
approved `845f1ee4…` template with pinned Jinja on those normalized inputs.
The current generic HTTP parser still refuses historical tool-call messages;
these broader template controls do not establish an HTTP tool-history route.
Additional plain-chat channel stops do not affect literal completion stops or
target likelihoods. Literal scoring uses completed one-row full-vocabulary
units, so peers can progress and a resumed scorer does not repeat reported rows.

Saved boundaries require the native completed position. Restores authenticate
owned cursor/boundary metadata and initialized extent footprints; padded bytes
cannot establish a logical position. A prepared restore blocks execution until
every logical range has a proven completed copy. Contradictory kept turn
checkpoints are omitted before file adoption. Clean pre-copy capacity refusals
retain the previous completed prefix and peer leases.
[Serving controls](experiments/gemma-serving/README.md) record the bounded route;
they do not establish qualified model support or optimized batching.

The [standalone MoE contracts](experiments/gemma-moe-primitives/README.md)
now expose checked original routing and scaled ordered reduction primitives.
Their first-eight ID view retains the graph's already funded full 128-pitch
ARGSORT root; its tail remains unwritten. The default-off dispatch below
provides checked graph integration and primitive fallback for readable keeps;
whole-model quality/performance selection remains owed.

The [default-off native MoE dispatch](experiments/gemma-native-moe/README.md)
now uses checked routing and scaled ordered reduction with complete-root
funding and kept-value primitive fallback. Its graph-order-only ordinary
output remains byte exact; compound norm/MoE still has nine representative
argmax differences outside the frozen allowance. The short native-only gain
keeps production selection and all full-model/batching gates open.

The [two-profile scalar serving controls](experiments/gemma31-serving/README.md)
cover complete own frontier rows, likelihood alignment, refusal with peer
progress, cross-variant kept-record rejection and pending 26↔31 switches.
Actual HTTP solo/cohort, SSE, stop, cancellation and likelihood controls use
each profile's own solo output; they do not qualify reference quality or
optimized joining.

## Required execution and optimization qualification

Every family/quant must adopt applicable selected Qwen/DeepSeek techniques
and qualify optimized batching before supported status. The
[optimization inventory](optimization-inventory.md) remains authoritative.
For these actual GGUF files the next slices owe:

| Transfer or contract | Eligibility and required qualification |
| --- | --- |
| Primitive completeness | Split F32 GeGLU and packed F32 GELU-tanh fallbacks are checked. The graph integrates these primitives, preserving RMSNorm/scale, V norm, sandwich order, softcap/tanh and full-width proportional RoPE. Floating MMVF fusion and the explicit quantized GeGLU writer stay off in model selection; actual-width primitive controls do not establish model quality or speed. |
| Q5_1 expert down | [Legacy primitive controls](experiments/m35-legacy-quants/README.md) cover ordinary/routed products, row-preserving and joined columns at K704/N2816/top-eight routes. Synthetic overlap measurements retain ordinary MMVQ where faster; model routing, selected dispatch, prefill and the last-layer Q8_0 execution remain to be qualified. Do not select a Q2_K or IQ2 kernel by analogy. |
| Shared input preparation | Reuse eligible Q8_1 preparation across ordinary Q8/K-quant products and fused gate/up reads, retaining maps/strides and Gemma's router and GeGLU arithmetic. DeepSeek's SwiGLU activation writer cannot transfer unchanged. |
| Routed prefill scheduling | Check compact expert-major tiles and full-K arithmetic for Q4_K fused gate/up and Q5_1 down at actual shapes/chunk sizes; keep only qualified speed/memory winners. Raw expert groups must remain authoritative for later paging. |
| Attention and device masks | The graph keeps checked local D256/GQA2 vector/MMA and global D512/GQA8 attention separate. Local primitives have ring-mask, shape-selection and scratch controls. Checked per-segment device mask production preserves fresh positions and independent caches; complete model dispatch remains to be qualified. Preserve independent caches and scale1.0; never invent sparse global attention. |
| Join products across requests | Apply Qwen/DeepSeek joined dense/routed/head products when operand/quant contracts fit. Preserve per-segment outputs, original one-token sums, stable route pair order and separate attention/state. Qualify scalar versus joined logits/state and departed/cancelled slots. |
| Lanes, graphs and lifetimes | Reuse request cohorts, completion-aware leases, stable-address graphs, charged per-lane scratch and hazard ordering. Shape/read-alignment choices must back padded cells and preserve exact continuation across capture/replay, spill/restore and time-slicing. |
| Bounded state and staging | Use initialized read/write footprints, growing extents, bounded host inputs, shared maximum workspace and separately owned slot state; measure peak memory for solo and batched envelopes at context boundaries. |
| Assistant and draft policy | The KV-sharing Q-only assistant is a later slice, not Qwen MTP or DSpark recurrence. Qualify shared target-cache ownership, centroid/head IDs, overwritten-ring rollback and greedy/sampled acceptance before transferring adaptive depth or selected-head optimizations. |
| EXL3 / other formats | Separate representation binding, packed products, codebook/rate and expert grouping qualification are owed. GGML recognition in this slice establishes no EXL3, NVFP4 or MXFP8 support. |

For every adopted execution path, record actual registry/plan selection,
precision/layout and shape limits, isolated and whole-model timing, solo and
batched quality/exact-state controls and peak memory. No speed or optimized
batching result is claimed here.
