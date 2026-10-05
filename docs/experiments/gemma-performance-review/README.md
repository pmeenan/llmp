<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma performance review handoff — 2026-10-05

The owner requested a pause after the in-flight source-use indexing change,
then resumed all M3.5 implementation and measurement on 2026-10-05 after the
Opus/Astra reviews. The [cold/retained-plan diagnostic](../gemma-retained-plan/README.md) now compares
identical state resets with capture disabled and separate phase timing. Similar
excess time per chunk alone does not establish a fixed host cost: reference work
per chunk is also similar. Cold planning costs 126/526 ms at Gemma26/31; state growth costs 64/164 ms.
Planning is material, but it does not explain the entire remaining gap.
The [current Gemma31 timeline](../gemma31-current-timeline/README.md) now records
592 ms in 32 internal GPU-idle gaps over 10 ms and 145 ms of extra vocabulary
projections. Product and attention duration sums are close to stock, with
different final-block row shapes and extra standalone MUL launches. These
measurements guide the resumed work; the untraced comparisons below remain
the competitive evidence.
The [state-only intermediate prefill path](../gemma-state-only-prefill/README.md)
now removes unused final-layer work while preserving exact KV continuations.
The optional 26B/all1024 and 31B/both256 recipes improve 3.04%/2.42%; the 31B
reference gap remains 8.28%, while 26B reference movement prevents a resolved
competitive gap. Ordinary serving has separate correctness controls.
M3.5 remains incomplete. The source-use optimization landed as `5ef7aac` after
1,700 Spark tests passed without failures or skips, including 311 GPU and 57
model tests. Both Sparks were checked idle at the pause.
The [plan](../../plan.md) and [model support matrix](../../model-support.md) retain
the remaining Gemma quality, batching, assistant, long-context and memory gates,
alongside the other families, EXL3/quants and media work. The review closes none
of those gates and does not extend M3's accepted speed exceptions to M3.5.

## Current matched measurements

These are untraced, same-host comparisons with the original llama.cpp engine,
not traced durations or model-load/swap times. Each screen runs reference,
unchanged native, candidate twice, then reference. The latest production change
only caches exact source-edge counts during planning; kernel arithmetic and
selected operations are unchanged. See [source-use indexing](../gemma-use-index/README.md)
and its [aggregate and pins](../gemma-use-index/results.json).

| Model / physical host | Native prefill mean | Reference prefill mean | Excess prefill time | Native / reference decode mean | Decode excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma26 / Spark-b | 2.712695 s | 2.413895 s | 0.298800 s / 12.3783% | 0.6883155 / 0.684530 s | 0.5530% |
| Gemma31 / Spark | 12.119850 s | 10.879900 s | 1.239950 s / 11.3967% | 3.187960 / 3.115565 s | 2.3237% |

Prefill pays 8,192 rows; decode pays 32 forced incoming rows with CPU argmax and
full-vocabulary publication. Both discard six warm rows, clear, and append
three untimed anchors before decode. Context is 16,384 with F16 KV, explicit
ring-cache `swa_full=false` and `kv_unified=false`. Gemma26 uses UD-Q4_K_M,
row cap/ubatch 1,024 and local/global capacities 2,048/16,384; Gemma31 uses
UD-Q4_K_XL, cap/ubatch 256 and capacities 1,280/16,384. Both are single-Spark
runs. Policies are native `all` at 26 and `both` at 31, with plain norm fusion
off; these optional math policies have not become production defaults.

Native pays 7/31 additional intermediate prefill heads; reference publishes
only the final prefill head. This difference is disclosed, not compensated
numerically. Dumping vectors/state and checking hashes occur outside timers.
One old-native arm and two candidate arms are a short screen, not proof of
sustained performance or other shapes. Native candidate state and both retained
heads match the old native exactly, as do all 32 choices. All cross-engine
choices agree. Both Gemma26 reference heads still differ; Gemma31's final head
is exact but its prefill head has maximum raw delta 0.41361475. These controls
do not establish all-32-vector, corpus quality or optimized-batching support.

Reference pin: llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` / b10964,
using the original pinned image and same-format artifacts. Native SDK is
`aarch64-c09daba6ac31edee` (Clang 22.1.8, NVCC 13.4.92, toolkit 13.4.2);
original image metadata records CUDA 13.3.0; driver is 580.178.04 on GB10.
These environment observations are inherited from the linked run records,
not a fresh environment query. TensorFold was refreshed at Task47 entry,
2026-10-05 19:25:15 UTC: `609ca419abecebdc5a059498a613680bd3aa847f`, version
0.6.5. Its checked Gemma26 recipe is MLX; no applicable Gemma31 CUDA recipe was
found. Do not assume that pin is still latest when another task begins.

## What has actually been resolved

- The older thin reference clients silently used full-length SWA caches.
  Matched ring-cache clients corrected this recipe. Earlier full-cache rates
  and memory comparisons are historical, not representative parity results.
  See [26 ring transfer](../gemma26-swa-ring-transfer/README.md) and
  [31 ring recipe](../gemma-swa-ring-h1/README.md).
- Repeated full-graph reader/private-output scans were a major Gemma26 CPU
  planning cost. The committed [root/read index](../gemma-plan-index/README.md)
  reduced prefill from a retained roughly 3.81 s baseline to roughly 2.77 s,
  about 27.3% less time. This was not a fresh old/new bookended comparison;
  fresh reference controls and exact native head/state checks accompany it.
- The [source-use extension](../gemma-use-index/README.md) gives direct fresh
  old/new reductions of 2.8011% at 26 and 2.3073% at 31. It reuses the index in
  `CanFuse` and `CanFuseSubgraph`, preserving local suffix/gather counts and
  standalone fallback. Both placement passes and SamePlan remain intact.
- [Plain RMSNorm/Mul fusion](../gemma-normmul-screen/README.md) selected 121
  additional Gemma31 fusions but gave no resolved speed gain. It remains off;
  this result does not exclude a benefit at Gemma26's different shape.

The large earlier Gemma26 host cost has a measured explanation and an adopted
fix. The current cold/retained comparison isolates material planning and state-growth
costs, alongside a remaining execution interval. It does not identify that
interval as kernel arithmetic or establish planning as the entire gap. Further
work targets unnecessary intermediate-chunk computation and a current matched
execution timeline.

## Evidence boundaries and unresolved leads

The [historical Gemma26 GPU profile](../gemma26-prefill-profile/README.md), before
the indexing changes, recorded wall spans 3.8305/2.4496 s and GPU activity unions
2.4694/2.3643 s, native/reference. The 1.3611/0.0853 s outside GPU activity was
not automatically CPU time. The subsequent [coarse diagnosis](../gemma26-prefill-coarse/README.md)
measured 1.0933 s of caller CPU in the ordered planning passes, with no GPU
overlap. Its graph-build/bind exclusive 52.3 ms is an **aggregate across eight
paid chunks**, not a per-chunk measurement. There is no equivalent current
Gemma31 CPU/GPU attribution. Do not subtract these historical categories from
the latest untraced results or transfer them to 31 as measurements.

Source-supported leads to rank by plausible contribution to the whole gap:

1. [`GraphOrder` / `Visit`](../../../src/kernels/ggml/fusion.cc) linearly search
   the visited vector for every reachable descriptor: quadratic membership
   work. [`PlanGemma4Chunk`](../../../src/engine/gemma4_plan.cc) builds twice via
   `SizedArena`; changing KV shapes cause plan misses through
   [`Gemma4Runner::Planned`](../../../src/engine/gemma4_runner.cc). Gemma31 pays
   32 prefill chunks versus 8 at 26. This is a concrete algorithmic lead, with
   **no measured contribution or implemented replacement**. A prepared outline
   uses a bounded pointer table owned/funded by `TensorArena`, preserves DFS
   source/output order and cycle/duplicate/PARAM semantics, and leaves the
   standalone path intact. [`SizedArena` and host accounting](../../../src/engine/planned.cc)
   must count that allocation; adding an uncharged hash set would be incomplete.
2. Compare actual graph construction, weight lookup, binding, validation and
   cache reuse with stock, rather than assuming the remaining time is GEMM.
   [`BuildGemma4Graph`](../../../src/kernels/ggml/gemma4_graph.cc) also performs
   linear weight-leaf lookups. Determine whether a structural difference
   repeats expensive work at every chunk.
3. Extra head publication and final-layer frontier narrowing change actual
   paid work and product shapes. The old 26 trace attributed about 26 ms of
   extra vocabulary-projection kernels, much smaller than its original host
   gap. The current 31 trace measures 145 ms of extra vocabulary projections.
   The state-only runtime path now preserves KV state and keeps scoring,
   final heads and retained-feature work full. These measurements explain
   a contributor; planning and state growth still need optimization.
4. Existing DeepSeek `dense_pair` input-quantization reuse is available but
   not enabled for Gemma. Dense31 Q4_K gate/up products are eligible at
   >=64 rows; Gemma26 Q8_0 attention projections can qualify, while its
   2,112-wide shared FFN fails the 128-output alignment. See
   [`MulMatQPairDenseFits`](../../../src/kernels/ggml/validate_ext.cc) and
   [`PlanGraph`](../../../src/kernels/ggml/graph_plan.cc). No Gemma speed result
   exists for this transfer; it is a secondary lead, not the known dominant cause.

Batch correctness/performance remains separate: [packed Gemma31 C4 attention](../gemma-packed-attention-c4/README.md)
matched all 128 stock heads but took about 12.3% more time. It jointly changes
local dispatch and local/global stream geometry; it does not establish a
precision-only cause or competitive batching.
[Gemma26 packed compound C4](../gemma26-compound-packed-c4/README.md)
still has 2/128 positive-margin disagreements, 92/128 exact heads, and a 7.15%
time deficit. [Common-input late MoE controls](../gemma26-late-moe/README.md)
match stock within each primitive/fused policy, with small policy-rounding
differences; they did not confirm a new arithmetic defect or close those misses.

## Prompt for Opus or Astra

```text
Do a read-only root-cause review of the remaining Gemma26/31 performance gap
in /home/pmeenan/src/jitLLM. Start with AGENTS.md, docs/workflow.md and
docs/experiments/gemma-performance-review/README.md, then pull only relevant
reports and source. This is a read-only review; do not edit, build, test, run inference,
profile or launch additional agents/jobs.

Latest matched 8K prefill excess: Gemma26 0.2988 s (12.38%); Gemma31
1.23995 s (11.40%). Earlier graph-reader scans were already fixed; exact
source-use counting adds only ~2–3%. Focus on a dominant whole-engine or
graph/paid-work mismatch, not a list of tiny possible kernel wins.

Rank at most three explanations by evidence and plausible magnitude. For
each, give exact source locations, what is confirmed versus hypothetical,
what existing evidence rules out, and the smallest decisive comparison.
Reconcile graph construction/reuse and CPU costs with actual model work.
Do not treat historical Gemma26 GPU-unattributed time as current Gemma31 CPU
time, or multiply the 52.3 ms aggregate graph-build cost as a per-chunk cost.
Do not reopen the corrected SWA reference recipe as an unresolved issue.
Separate solo performance from the still-open C4 correctness/performance gap.

Explain whether GraphOrder's quadratic visited search plausibly accounts
for most of Gemma31's residual, whether another structural mismatch is
stronger, and which direct before/after test would settle the leading case.
If a comparable latest TensorFold path exists, identify its current pin and
applicable model/format/backend before treating it as a performance target.
Do not claim a cause or measured speed gain from source inspection alone.
```
