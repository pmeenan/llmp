<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 2 bounded execution

`model/gemma2.h` supplies a closed profile and checked tensor binding for the
approved Gemma 2 2B Q8_0 checkpoint. The bounded execution source adds checked
state/inputs, a native graph/plan and a bounded runner over the shared paged engine.
The approved source is authenticated and deeply imported; a representative C1
control reproduces all 33 stock full heads byte for byte. The later
[bounded serving slice](experiments/gemma2-serving/README.md) adds
chat/literal routes and compatible two-owner prefill. Broader execution support
remains unqualified.

**Being replaced (M3.6, D-107).** The per-family profile, graph and runner
described here give way to one engine of shared components over llmpalooza's own
graph IR ([engine-components.md](engine-components.md)), and per-family
optimization transfers are retired; M3.5 is parked until M3.6 exits. This file
describes the code as it is, and changes as each piece lands.

## Actual checkpoint contract

The approved source is
[`bartowski/gemma-2-2b-it-GGUF@855f67caed130e1befc571b52bd181be2e858883`](https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/tree/855f67caed130e1befc571b52bd181be2e858883),
`gemma-2-2b-it-Q8_0.gguf`, 2,784,495,456 bytes. The retained exact-revision
primary HF API LFS identity is
`2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe`;
the execution slice locally authenticated the complete file against that SHA-256
and exact length. The earlier foundation used metadata ranges only.

| Metadata or descriptor fact | Value |
| --- | ---: |
| Architecture / GGUF version | `gemma2` / 3 |
| Layers / hidden width / FFN width | 26 / 2,304 / 9,216 |
| Query / KV heads; key / value dimension | 8 / 4; 256 / 256 |
| Vocabulary / context / sliding window | 256,000 / 8,192 / 4,096 |
| RMS epsilon; attention / final logit softcaps | F32 1e-6; 50 / 30 |
| Actual tensor count | 288 |
| Tensor types | Q8_0 ×183; F32 ×105 |

The exact [d812 Gemma2 model](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/models/gemma2.cpp)
selects alternating local/global attention (even/odd layers), scales attention
queries by 1/sqrt(256) = 0.0625, and narrows the final attention/residual rows
before post-normalization and FFN. Its generic loader defaults absent RoPE
base/scaling metadata to 10,000/1. These are source-derived profile facts,
not numerical execution qualification. Unlike Gemma3, there are no Q/K norm
weights. Each layer has four F32 norm vectors and distinct Q/K/V matrices.
The output head ties to the Q8_0 token embedding; there is no `output.weight`.

## Storage and refusal contract

`BindGemma2` accepts only this exact profile, architecture and 288-resource
F32/Q8_0 role/type/shape contract. A second output role may alias the embedding
resource. Missing, repeated, empty or unused roles, independent heads, changed
profiles, other formats/shapes and insufficient readable storage are refused.
`CheckGemma2Binding` rechecks mutable public descriptors with a fixed stack
membership array, without reconstructing resources or allocating on success.
It refuses duplicate/out-of-domain indices, edited shapes/types/storage,
incomplete layers and a head that no longer aliases the embedding.

Stored bytes and readable bytes differ deliberately. Width 2,304 is a multiple
of Q8_0's 32 elements but not GGML's 512-element row padding. The existing
artifact representation contract requires 272 extra readable bytes at the
end of each affected tensor, not per row. There are 131 such tensors: the
embedding and five matrices per layer. Their source payload sizes remain
unchanged; prepared import funds and initializes the tail using the existing
artifact mechanism. Binding raw stored bytes as readable is refused.

The network-free [fixture generator](../tests/unit/data/gemma2/generate.py)
checks the retained metadata prefix and a 65,536-byte range, parses all 288
descriptors with the existing GGUF helper/native type table, checks unique names,
rank/type/block geometry, aligned nonoverlapping file ranges and the final
file boundary, and emits only aggregate metadata and tensor descriptors.
The foundation range response was exactly HTTP 206,
`bytes 6029343-6094878/2784495456`, with the checked content length.
The tensor table ends at byte 6,046,552; data starts at 6,046,560.
Raw prefixes/ranges stay outside Git. Fixture provenance includes their hashes
and explicitly distinguishes the reported whole-file identity from a local hash.

Six focused CPU controls cover actual metadata/storage, role and profile
refusals, every tensor's readable boundary, mutable descriptor identity and
alternating local/global scheduling. All six pass on spark-b alongside
the six existing Gemma3 foundation/state controls. The native ARM focused
build and tests ran under installed supervision on 2026-10-07; no model or
GPU operation ran. Whole-source checksum synchronization, an empty itemized
dry run and the checked source inventory bind the successful build. Local
fixture regeneration, format, license/header and portability checks also pass.

## Bounded execution and remaining gates

The [bounded C1 execution slice](experiments/gemma2-execution/README.md)
adds the deep-verified artifact, checked alternating local/global state and
inputs, native graph/plan and C1 runner. Its seven own controls preserve eager
and captured heads, retained Clear, refusal atomicity, initialized-state
spill/restore and device-greedy choices/state. The optimized norm/Mul, quantized
GeGLU and width-2304 norm/ADD recipe reproduces all 33 stock heads exactly at
context4096, two 128-row prompt chunks and 32 teacher-forced transitions.
All 17 focused model/graph/plan tests, 15 norm CPU checks, one width-2304 GPU
operand control and 26 shared importer controls pass on spark-b.
The [attention-softcap primitive slice](experiments/gemma2-softcap/README.md)
qualifies actual nonzero-specialization occupancy and funded vector/MMA
launches at caps 0/50/25, with FP64 operand checks, byte-exact repeats and fresh
capture replays, exact funding and refusal controls. Both new controls and
eleven focused legacy controls pass on spark-b. The opt-in
[two-owner decode slice](experiments/gemma2-owner2/README.md) now qualifies
cap50 D256/H8/GQA2 owner attention with actual true-specialization occupancy
and funding. Independent-prefill C2 controls cover small and wrapped local-ring
history, eager/capture identity, refusal atomicity, Clear and restore-next
replay. Stock comparison has zero strict choice differences, with 65/66 small
and 66/66 ring full heads byte-identical. Its short n=2 paid C2 cycle is 1.49%
slower than stock with exact natural histories/final heads; no parity claim.
Those diagnostic owner defaults remain off. The later serving recipe explicitly
selects checked common-width owner padding and compatible prefill; the later
multirow transfer below removes K/V packing for 128-row joined chunks.
The C1 control covers final logit softcap30 and width-2304 readable tails;
the later serving slice adds representative compatible prefill and departures.
Broader context/quality, sustained and switch qualification remain open.
The [checkpoint/adoption foundation](experiments/gemma2-checkpoint/README.md)
adds an internal layout-bound kept-state path: ordered whole extents must fund
all initialized logical ranges, pending writes block execution and growth, and
restore completion requires proven contiguous copies of every needed range.
Selected peers remain held while the destination footprint changes. Adoption
requires an empty idle healthy slot and an identical layout identifier. All
19 focused checks pass, including actual wrapped-ring snapshot restoration and
named-file adoption in a fresh node with an exact live-peer next head; actual
serving and compatible joint prefill were separate work, checked below.
The pin's Gemma2 HF-to-GGUF converter already adds one to norm weights;
approved GGUF import must preserve those norm payloads, with no second +1
at import or runtime.
The existing SentencePiece fixture covers tokenizer behavior. The serving
admission foundation below separately checks the actual chat template and its
refusals through the existing interpreter; the later adapter reuses those assets.

## Serving admission foundation

The settings reader recognizes the approved Q8_0 artifact with its exact source
name, length and SHA-256 and rechecks all fixed-profile tensor bindings. Its
serving envelope is at most 8,192 tokens, 128 rows per owner and two owners;
the uncalibrated default remains one owner. Speculation and drafters are
refused. That foundation installed settings only; the bounded adapter below
adds actual runtime serving without changing configuration or artifact schemas.

The kept 591-byte template has SHA-256
`ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6`.
It uses the existing native C++ Jinja interpreter, rather than Gemma3's renderer:
the actual template refuses an initial system message and requires alternating
user/assistant turns. It trims content, emits `model` for assistant turns and
optionally appends the generation prefix. Its end-of-turn token is 107;
tokenizer EOS 1 is a separate runtime stop. Focused actual-artifact tests cover
these behaviors and preserve control-token provenance inside message content.

Validated on Spark B on 2026-10-07: six focused settings controls, eight binding
controls, two checkpoint-footprint controls and three actual-artifact template
controls pass. The final diagnostic-only test rebuild repeats all three template
controls. No model inference or HTTP serving is claimed by this foundation.
The new task's latest TensorFold check still resolves to
`041d14a94e951834470fd514ed33e65b8be1059a`, with no documented Gemma2 CUDA/GGUF path.

## Bounded ordinary serving and compatible prefill

The [serving slice](experiments/gemma2-serving/README.md) installs the approved
artifact's runtime factory with context<=8192, per-owner rows<=128 and at most
two owners (default one). Norm/Mul, quantized GeGLU, norm/ADD and true-cap50
owner attention are explicitly selected; no Q/K norm/RoPE fusion is requested.
A separately funded 256-row wave joins compatible plain prefill. Real F16 K/V
concatenation supplies original C2 MMA geometry; temporary F16 zero and invisible
mask tails supply a common read width for unequal one-row owners. Actual state
roots, source bounds, per-owner 128/checkpoint identities and ring 4,352 remain
unchanged. Scoring, one-row/incompatible prefill and reuse/checkpoint units
retain scalar fallback. Both cap50 operands pass byte-exact physical-stream,
FP64 and fresh capture controls.

The first 256/768-prefix model screen has all 76 full heads byte-identical to
stock, with zero strict choices and target-NLL delta. All 45 focused checks and
five HTTP lifecycle cases pass, including the actual system-role refusal,
finite likelihoods, cached 16/4/16 replay, SSE/stops, queued peer progress after
disconnect and two kept conversations adopted/replayed after restart. Actual
joined counters are observed after drain; client reads may be buffered.
The short matched C2 RNNR is 5.7824% slower than stock (n=2), with identical
natural choices/final heads. No performance parity or broader/sustained support
is claimed; mixed wrapped-ring/joint-prefill, context, memory/swap and wider
batching gates remain separate.


## Copy-free bounded decode

The bounded serving recipe now reads unequal cap50 D256/H8/C2 cache roots
directly at their actual aligned widths, retaining the common logical width
and original attention partitioning. Entirely absent partitions emit neutral
fixup data; nonempty tiles use the unchanged upstream helper. Equal-width,
scalar and other-family paths retain their existing contracts. Actual state,
source bounds, 128 rows per owner, 256 rows per wave and local ring 4,352 are unchanged;
internal runner/graph defaults remain off.

The [causal screen](experiments/gemma2-serving/README.md#copy-free-bounded-owner-reads-2026-10-07)
passes22 focused controls, exact poisoned physical-stream operands and FP64,
eager/captured replay and refusal checks. Padded/bounded native heads and state
are byte exact at 256/768 and wrapped 4,352/4,864 prefixes. Stock heads are 76/76
and 73/76 exact respectively, with zero strict/tie differences and ring relative
conditional-loss delta 1.59279e-9. The six-arm same-binary padded/bounded control
plus original stock bookends improves native paid latency by 5.2588%, to +0.09665%
versus stock on this short n=2 C2 screen. This is not sustained or endpoint
parity; broader contexts, ring/chunk geometries, memory/swap and cohorts remain
separate gates.

The adopted runtime repeats all five HTTP cases and two clean restart epochs,
including finite likelihoods, actual system-role refusal, cached checkpoint
replay, SSE/stops and client-observed peer completion after disconnect. The
first drain reports 26 selected bounded owner plans and 180 joined groups;
these are actual selection/progress witnesses, not per-replay kernel counts.

## Near-8K model-switch state

The [scalar near-8K swap control](experiments/gemma-near8k-swaps/README.md)
saves 8063 tokens at context 8192 and checks 64 continuation tokens/full heads
against unswapped execution after switching to Qwen3.8 and back. Both saved-state
returns are SHA-256 exact over initialized bytes; the prepared return keeps 32
graphs and replays 63 times. The worst prepared handoff is 5.911946 s, with the
diagnostic snapshot charged separately from sampled MemAvailable reporting.
This repeated-text boundary is near-8K, not an exact 8192 saved state or a
real-corpus/retrieval, maximum-context or sustained-memory qualification.

## Graph-owned GPU masks

Ordinary runners now construct causal and 4,096-window ring masks on the GPU
from checked absolute positions, using the same producer as Gemma3/Gemma 4.
The [transfer control](experiments/gemma2-serving/README.md#shared-gpu-masks-2026-10-07)
keeps per-owner 128 rows, a 256-row joined wave, local capacity 4,352 and
softcap50 unchanged. The graph/plan foundation and explicit runner override
retain host-reference masks; ordinary runners default to device masks.
Shared engine helpers authenticate producers and fund optional host padding
separately from device activations.

Wrapped C2 prefill falls 6.616% in a same-binary host/device/device/host screen;
all heads, choices and initialized states remain exact. Five HTTP/restart cases
and four checkpoint GPU cases pass through the adopted path. A fresh short
llama.cpp comparison at these 4,352/4,864 prefixes showed native prefill
6.721% slower and the paid cycle 4.586% slower at that snapshot; decode is
within noise. The subsequent multirow actual-root transfer below closes
that measured warm gap for this workload.

## Prefill lookahead and joined hints

Ordinary serving now uses the shared funded lookahead lifecycle and captures
future prefill graphs. Its fixed shared group builds up to two distinct missing
next/after shapes, with independent optional funding and completed-unit
installation; [cold 256-row and warm 128-row controls](experiments/prefill-transfer/README.md#two-distinct-future-shapes-2026-10-08)
retain exact heads/state and kept restart. The same checkpoint-aware hint calculator feeds scalar
and compatible two-owner prompts; hints describe shapes without preparing
future state. Mixed future head modes suppress only that stage, and completed
owners leave later cohorts independently.

A weight-warm, plan-cold two-owner 4,352/4,864-prefix screen improves first
traversal prefill 5.252%, with identical 64 generated choices and complete final
heads. The corrected warm control is 0.604% slower at two samples per arm; no
warm speed benefit is claimed. This native optimization comparison does not
by itself replace the earlier llama.cpp comparison or close that warm gap.
[Method, focused serving/state controls and limits](experiments/prefill-transfer/README.md#gemma2-capture-and-joined-gemma-hints-2026-10-07).

## Multirow actual-root prefill

Ordinary compatible two-owner 2–128-row chunks now read their real F16 K/V
cache roots, preserving the selected packed MMA geometry and cap50 arithmetic.
Masks still concatenate. Flexible query tiles and padded tails preserve the
original Columns4/8/16/32 selection. A later 64-row native factor improves
prefill 10.474% (n=2); fresh matched stock bookends are level (+0.059%
prefill / −0.038% paid) with exact choices and full heads. Actual five-row
joined tails, initialized state and kept restart/continuation pass.
Compatible >512 rows, mixed-width roots and cohorts remain explicit ports;
ordinary chunks remain 128. The [matched timeline and transfer](experiments/gemma-prefill-copies/README.md)
found 3432 baseline D2D copies and lower native arithmetic active time.
Removing K/V packing improves same-native wrapped prefill 8.662% (n=2).
Fresh llama.cpp/native/native/llama.cpp bookends are level within observed
movement: native prefill −0.293%, paid cycle −0.202%, with all 64 choices and
both complete finite final heads byte-exact. Production adapter heads/state,
kept restart/continuation and wrapped own-state controls pass. Public 8K/two
admission is unchanged; broader shapes, corpus and sustained gates stay open.

Configured equal-C2 chunks through 256 now use the same actual-root body
with a funded 512-row wave and two-head envelope; public chunks remain 128.
The 256-row native factor improves prefill 5.096% (n=2). Twenty focused
controls include the full cap0 256×131072 primitive boundary and both
families' default-ring state/checkpoint/kept restart; Gemma2 cap50 retains
16384 cache cells. An explicit comparison control matches stock's 4608-cell
SWA ring (default configured 256 uses 4352), restoring exact full logits.
Fresh matched-ring stock bookends leave +1.242% prefill / +0.797% paid,
with decode level; these remain residuals, not parity passes. The identical
seed's cross-owner head difference is already present in packed attention,
while each owner's packed/root head is exact; its baseline cause remains
unresolved. [Bounds, controls and reference evidence](experiments/gemma-prefill-copies/README.md#configured-129256-row-equal-owner-extension).

The later configured equal-C2 extension reaches 512 rows with unchanged
CUDA arithmetic and public 128-row chunks. Native prefill improves 2.706%
(n=2). Partial/full/maximal operands and actual wrapped 512-row replay,
checkpoint, protected-peer state, spill and kept restart remain exact.
Matching stock's 5120-cell SWA ring gives exact complete heads and choices;
fresh stock bookends retain +0.786% prefill / +0.284% paid. The paid gap is
within observed movement, while prefill ranges remain disjoint; no prefill
parity claim follows. [512-row method and remaining work](experiments/gemma-prefill-copies/README.md#configured-257512-row-equal-owner-extension).

## Shared original-MMVQ preparation

The runner and serving recipe now reuse exact Q8_1 inputs across selected QKV
and eligible unfused FFN products, with original GGML consumers and C1 GeGLU
preserved. Device-specific MMVQ selection gates nine compatible quant formats
through eight columns; other routes retain ordinary products. Exact full
heads, token histories, state/refusals, partial departure and checkpoint/spill
replay pass focused controls. The short C2 decode interval improves descriptively;
whole paid movement remains inside bookend drift. Startup measures every
configured small owner-row composition, with its observed cost and host-plan
floor recorded in the [qualification report](experiments/gemma-shared-q8/README.md).
Public contexts/slots/chunks are unchanged; broader compatible routes remain open.

## Fresh state preparation beside prefill

Ordinary serving now uses the shared no-victim next-chunk preparation ticket
for eligible fresh zero-backed state. Future residency stays unpublished until
actual Use; scoped page-in retirement and complete owner retention protect
cancellation, Clear and teardown. The ordinary 128-row C2/context-8192 own
screen reduces prefill 4.126% and combined paid work 2.905% at n=2, with exact
complete heads/choices/initialized state. Decode ranges overlap; no decode or
reference-parity gain is claimed. Actual default PromptSession and separate
joined wrapped checkpoint/peer/spill/kept-restart controls pass. Kept restart,
fresh sparse-file sources and broader context/profile qualification remain
open. [Method, drift and ownership proof](experiments/gemma-state-prepare-ahead/README.md).
