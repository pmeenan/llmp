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
reference gap remained 8.28%, while 26B reference movement prevented a resolved
competitive gap. Ordinary serving has separate correctness controls.
The [bounded graph traversal](../ggml-graph-order/README.md) then reduces
state-only prefill by 1.93%/0.99%; fresh reference gaps are 8.19%/7.54%.
The [bounded next-plan lookahead](../gemma-prefill-lookahead/README.md) then
reduces same-binary prefill by 1.55%/2.36% at 26/31. The later 31B mean is
4.94% above its preceding reference bookends; this is qualified evidence, not
a fresh reference-bookended parity result. The initial noisy 31B screen is
preserved in the report.
The [plain norm retry and adoption](../gemma-state-only-norm-policy/README.md)
favors fusion on both state-only/lookahead research recipes and preserves exact
ordinary off/on heads and state. Plain RMSNorm/Mul is now a serving default;
other arithmetic policies remain opt-ins. The 31B retry has noisy effect
magnitude and no fresh reference comparison.
The [current post-lookahead phases](../gemma-current-phases/README.md) narrow
serialized work to roughly 175 ms binding, 17 ms coverage and 5 ms insertion
across 32 plans, plus 162–185 ms state growth. Those nested binding intervals
already belong to planning/publication; they are not additive. The subsequent
[per-bind wrapper cache](../gemma-binding-wrapper-cache/README.md) identifies
repeated declaration validation as a material contributor: 31B binding falls
174.483 to 45.828 / 44.989 ms, and 26B 39.755 to 12.320 / 12.097 ms. Fresh
per-step operand, arity and lane checks remain. Full heads, initialized state
and continuations remain native-exact; seven focused controls pass. The wall
screens favor the change with material spread and no fresh reference comparison.
The [fresh current solo bookends](../gemma-current-reference/README.md) now
record a stable 2.43% prefill gap at 26B. Native 31B timing is stable, but stock
varies 1.3355 s; comparison to its closing arm leaves a 2.08% gap. The separate
[current corpus screen](../gemma-current-quality/README.md) matches all 1,024
31B heads exactly at 256 rows; 26B at 1,024 rows retains nine positive-margin
choice failures. State growth, execution variance and broader qualification
remain open.
M3.5 remains incomplete. The source-use optimization landed as `5ef7aac` after
1,700 Spark tests passed without failures or skips, including 311 GPU and 57
model tests. Both Sparks were checked idle at the pause.
The [plan](../../plan.md) and [model support matrix](../../model-support.md) retain
the remaining Gemma quality, batching, assistant, long-context and memory gates,
alongside the other families, EXL3/quants and media work. The review closes none
of those gates and does not extend M3's accepted speed exceptions to M3.5.

## Latest reference-bookended measurements

The [current solo screen](../gemma-current-reference/README.md) measures the
accumulated state-only, lookahead and wrapper-cache recipe with plain norm
fusion enabled. Each host runs reference, native twice, then reference;
all four arms and the existing exact checker retire successfully.

| Model / physical host | Native prefill first / repeat | Reference prefill first / repeat | Prefill comparison | Native / reference decode mean | Decode excess |
| --- | ---: | ---: | --- | ---: | ---: |
| Gemma26 / Spark-b | 2.47975 / 2.47556 s | 2.41652 / 2.42106 s | +2.43365%, stable bookends | 0.689154 / 0.682870 s | 0.92023% |
| Gemma31 / Spark | 11.1940 / 11.2168 s | 12.3124 / 10.9769 s | +2.08164% versus closing reference; variable bookends | 3.188465 / 3.121955 s | 2.13040% |

No arm is excluded. The 31B reference spread prevents parity or gain attribution;
comparison to its slower mean is unsuitable evidence of improvement. These are
short untraced solo measurements, with load/swap times outside their scope.
Prefill pays 8,192 rows; decode pays 32 forced incoming rows with CPU argmax and
full-vocabulary publication. Both discard six warm rows, clear, and append
three untimed anchors. Context is 16,384 with F16 KV, explicit ring-cache
`swa_full=false` and `kv_unified=false`. Gemma26 uses UD-Q4_K_M, row cap/ubatch
1,024 and local/global cells 2,048/16,384; Gemma31 uses UD-Q4_K_XL, cap/ubatch
256 and cells 1,280/16,384. Native research policies are `all` at 26 and `both`
at 31, with plain norm on; other norm/MoE arithmetic policies remain opt-ins.

Both engines retain one prefill and one final complete head. Native heads,
initialized states and 32 choices stay exact to prior native controls; stock
heads stay exact to prior stock and own repeats. All cross-engine choices agree
in this fixture. Both 26B reference heads still differ; 31B's final head is
exact but its prefill head has maximum raw delta 0.41361475. These two-head
controls do not replace the current corpus quality or C4 batching gates.

Reference remains original llama.cpp `b29c606` / b10964, at the pinned CUDA 13.3
image and same-format artifacts. Native retained helper `68368ed6` uses source
`845617f` plus the state-preparation benchmark argument; selected arithmetic and
executor match later main, without claiming a latest-main binary. SDK is
`aarch64-c09daba6ac31edee`, native NVCC 13.4.92 / toolkit 13.4.2, GB10 driver
580.178.04. Full recipes, identities, phase spans and official summaries are in
[current results](../gemma-current-reference/results.json). Task-entry
TensorFold refresh at 2026-10-06 00:07:53 UTC remains `609ca419` / 0.6.5, MLX26
without a comparable Gemma26/31 CUDA recipe. Recheck it at each future task.

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
  additional Gemma31 fusions but gave no resolved speed gain in that earlier
  full-head screen. The subsequent state-only/lookahead retry favors fusion
  and qualified ordinary controls now support the plain norm default in both
  profiles.

The large earlier Gemma26 host cost has a measured explanation and an adopted
fix. The current cold/retained comparison isolates material planning and state-growth
costs, alongside a remaining execution interval. It does not identify that
interval as kernel arithmetic or establish planning as the entire gap. The
current timeline and state-only dependency cut are now recorded. Further work
has adopted bounded next-chunk CPU planning overlap; remaining binding, state
growth and execution costs are still under investigation.

## Evidence boundaries and unresolved leads

The [historical Gemma26 GPU profile](../gemma26-prefill-profile/README.md), before
the indexing changes, recorded wall spans 3.8305/2.4496 s and GPU activity unions
2.4694/2.3643 s, native/reference. The 1.3611/0.0853 s outside GPU activity was
not automatically CPU time. The subsequent [coarse diagnosis](../gemma26-prefill-coarse/README.md)
measured 1.0933 s of caller CPU in the ordered planning passes, with no GPU
overlap. Its graph-build/bind exclusive 52.3 ms is an **aggregate across eight
paid chunks**, not a per-chunk measurement. There is no equivalent current
Gemma31 caller-CPU attribution. Do not subtract these historical categories from
the latest untraced results or transfer them to 31 as measurements.

Source-supported leads to rank by plausible contribution to the whole gap:

1. [`GraphOrder`](../../../src/kernels/ggml/fusion.cc) now uses the
   [bounded arena table](../ggml-graph-order/README.md), with a small measured
   end-to-end improvement. [`PlanGemma4Chunk`](../../../src/engine/gemma4_plan.cc) still builds twice via
   `SizedArena`; changing KV shapes cause plan misses through
   [`Gemma4Runner::Planned`](../../../src/engine/gemma4_runner.cc). Gemma31 pays
   32 prefill chunks versus 8 at 26. Repeated planning remains material; a
   bounded next-plan lookahead is now adopted without future state growth
   or changes to graph arithmetic. The traversal screen did not emit separate
   planning counters, so its gain does not establish the remaining phase cost.
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
The subsequent [independent-cache C4 consumer](../gemma-owner-root-c4/README.md)
removes K/V packing alone for 8.93% lower paid latency. Plain-norm-on is now
2.10% slower than fresh original bookends, with all 128 heads byte-exact.
This context-256 diagnostic leaves wider contexts and production selection open.
The [current C4 phase split](../gemma-owner-c4-phases/README.md) records 32 plan
hits and 24.74 ms in checks. Paid time outside execution averages 78.73 ms,
including 43.57 ms outside all runner phases; these spans do not establish
active GPU cost or a fresh-reference gap.
[Gemma26 packed compound C4](../gemma26-compound-packed-c4/README.md)
still has 2/128 positive-margin disagreements, 92/128 exact heads, and a 7.15%
time deficit. [Common-input late MoE controls](../gemma26-late-moe/README.md)
match stock within each primitive/fused policy, with small policy-rounding
differences; they did not confirm a new arithmetic defect or close those misses.

The [current Gemma26 1,024-row stock dispatch](../gemma-current-dispatch/README.md)
now selects routing in 29 layers, with no layer-28 selection, while native
selects 30. All norm/reduction counts match and the observed complete stock
output is unchanged. This identifies a quality lead; the selected-chain logger
does not report refusal reasons or prove the cause of the nine disagreements.

The [single routing keep factor](../gemma-keep28-routing/README.md) now resolves
that fixed-corpus difference: retaining only layer 28's routing probabilities
matches all 1,024 stock heads byte for byte. The routing-policy difference
caused the nine disagreements for this recipe. A general selection rule remains open;
the [actual refusal observation](../gemma26-routing-gate/README.md) now confirms
32,768 bytes of weights/logits overlap under the original >8-row memory gate.
This placement-dependent result supplies no production whitelist or timing gain.

## Prompt for Opus or Astra

```text
Do a read-only root-cause review of the remaining Gemma26/31 performance gap
in /home/pmeenan/src/jitLLM. Start with AGENTS.md, docs/workflow.md and
docs/experiments/gemma-performance-review/README.md, then pull only relevant
reports and source. This is a read-only review; do not edit, build, test, run inference,
profile or launch additional agents/jobs.

Latest fresh solo bookends: Gemma26 native prefill2.47975/2.47556s versus
reference2.41652/2.42106s (+2.43%); decode+.92%. Gemma31 native11.1940/11.2168s
versus reference12.3124/10.9769s: native stable, reference variable; +2.08%
versus closing reference, decode+2.13%. Do not infer parity from the slow stock
mean. Graph-reader/source-use scans, bounded DFS membership, unused
intermediate final-layer work, next-plan CPU overlap, plain norm selection and
repeated wrapper declaration validation are already addressed. Focus on a
whole-engine or graph/paid-work mismatch supported by current evidence.

Rank at most three explanations by evidence and plausible magnitude. For
each, give exact source locations, what is confirmed versus hypothetical,
what existing evidence rules out, and the smallest decisive comparison.
Reconcile graph construction/reuse and CPU costs with actual model work.
Do not treat historical Gemma26 GPU-unattributed time as current Gemma31 CPU
time, or multiply the 52.3 ms aggregate graph-build cost as a per-chunk cost.
Do not reopen the corrected SWA reference recipe as an unresolved issue.
Separate solo performance from the still-open C4 correctness/performance gap.

Current phase spans:31B binding45ms/state161–175ms;26B binding12ms/state59–67ms.
Native completed execution fits within stock whole prefill time on both
profiles. Those intervals do not isolate GPU arithmetic or a state allocation
subcallee. Grouped upfront state preparation reduced31B state time61ms but
showed no wall win amid execution spread. The closed C4 owner-root consumer removes K/V packing for an 8.93% gain;
plain-norm-on remains 2.10% slower than fresh original bookends with exact heads.
Wider-context and Gemma26 transfer remain open.
The current teacher-forcing screen matches all1024 full31B heads at256 rows;
26B all1024's nine positive-margin failures are resolved by the bounded keep28
routing diagnostic; general policy selection remains open. Keep quality diagnosis
separate from performance and do not widen either gate. Identify the smallest
decisive test for the strongest remaining case.
If a comparable latest TensorFold path exists, identify its current pin and
applicable model/format/backend before treating it as a performance target.
Do not claim a cause or measured speed gain from source inspection alone.
```
