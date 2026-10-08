<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 C5 prefix-only retained-output diagnostic

Keeping one layer's router probabilities during independent multi-row prefixes
removes both prior paid-decode choice differences and makes all 160 paid complete
heads byte-exact to retained FIRST stock. The complete strict gate still fails:
one of five frontier choices differs at positive reference margin. The independent
160-target conditional-loss gate passes at +0.08119%. Neither failure is waived,
and this diagnostic adds no production routing policy or serving default.

| Current factor against retained FIRST stock | Result |
| --- | ---: |
| Strict positive-margin differences /165 complete finite heads | 1 (FAIL) |
| Reference-tie differences | 0 |
| Paid complete heads byte-exact /160 | 160 |
| Frontier complete heads byte-exact /5 | 0 |
| Maximum raw full-head difference | 0.558618546 |
| Mean native conditional NLL /160 targets | 10.6623191183 |
| Mean FIRST-stock conditional NLL | 10.6615075323 |
| Relative conditional loss | +0.081191539% (PASS) |

The remaining difference is at the prefill frontier, with reference margin
0.017386675. Individual rows, token IDs and vectors remain external. The phase
split follows the completed comparison's 160 exact rows plus five distinct
frontier NLL records; every paid head is therefore exact. The earlier
[Gemma26 transfer](../gemma-partial-owner-transfer/README.md) retains its two
paid misses, 64 exact rows and passing independent loss gate. Its failed result
and original FIRST reference identity are unchanged.

The new manual `llmp_gemma_joined_prefix_keep` target links the unchanged joined
benchmark and execution core with a narrow `PlanGemma4Chunk` linker wrapper.
`LLMP_GEMMA_PREFIX_KEEP28=0|1` is required and snapshotted once at first planning.
Mode 0 forwards the original keep span. Mode 1 appends only
`blk.28.router_probabilities` for the approved 26 profile, context 4096, max rows 1024,
five slots, full layers, head enabled, no feature/hidden input, a single segment
and more than one row. Measurement and both placement/planning passes receive
that same keep. Other shapes forward unchanged, including every joined five
one-query decode. No graph, kernel, runtime option or public policy is changed.

The guard is a diagnostic external reader. The prior [1024-row keep factor](../gemma-keep28-routing/README.md) and
[controller gate](../gemma26-routing-gate/README.md) preserve their separate
corpus/routing results. The allocation-dependent reference refusal at layer 28
motivated this hypothesis. The retained stock 992-prefix routing selection was
not observed.
The new keep also changes retention and placement, so this result does not prove
that stock used a particular 992 selector or justify a production layer whitelist.

Four no-launch contract processes cover off/on/invalid/missing modes, ten checks
each, without skips. They verify exact argument and original keep forwarding,
measurement and both activation-address passes, eligible prefix selection and
excluded decode/feature/context/slot/layer shapes. Only the new benchmark helper
and contract target are built; the original a921f20a helper remains immutable.

The measured recipe remains approved 26/all: context 4096, max rows 1024, head cap 5,
plain norm, norm/RoPE, norm/add and all MoE routing/reduction policies. Invariant
products, shared Q8 and RoPE-store stay off. Five checkpoint-qualified histories
receive 992 prefix rows and supplied positions 992..1023. Each arm reports five
actual prefix route 29 plans; plain norm 121/RoPE 60/ADD 90/reduce 30 are unchanged.
Decode remains route 30 with owner 60/partial 60, one five-column product wave and
real 4+1 roots on the original whole-five grid. All 32 waves/160 units complete,
with 32 replays and the unchanged 183,615,488-byte publication allowance.

Before comparison, both native arms repeat exactly across 165 finite complete
heads (173,015,040 bytes), ten initialized 230,686,720-byte state snapshots per arm,
both layouts and the full 49,152-byte carrier. All five frontier and final state
hashes differ from the failed native baseline; layouts and input remain identical.
This is expected scope for a prefix factor, not native/stock state equivalence.
The candidate helper 6ebbfa0d uses the same SDK receipt eb7a4ebc and unchanged core
bytes as a921f20a, with only the four new benchmark/registration paths.

Comparison reuses the already acquired physical-five FIRST stock pair whose
complete-head identity is 044b989f. It repeats exactly, as authenticated by the
unchanged numerical analyzer. No stock reacquisition, scalar calibration, new
margin allowance or historical rescoring occurs. Strict allowance remains zero;
full-vocabulary FP64 conditional loss uses exactly 160 within-history targets
from predictor rows 0..31, supplied positions 992..1023. Final row 32 predicts 1024
and is unscored. This is not a full 1023-transition corpus score.

The primary comparison remains officially FAILED rc 1, with one successful step
of three queued. Its complete result was written before failure. The exact skipped
final source authentication command then passed as a separate supervised job,
without scoring or model work. Build 7/native 6/own 3/postauth 1 finish DONE0; all four
owned analysis containers are absent after checked retirement. Native paid times
are 1.21322/1.28231 seconds (69.09 ms spread) against retained, noncontemporaneous
stock; no performance, parity or sustained claim follows.

[Aggregate receipts](results.json) retain actual source/helper/SDK/own/comparison,
all five official attempts and retirement identities; raw records remain outside
Git. The source base is 3165b87 and affected parent docs are seeded 358d622. Latest
TensorFold primary cb2ebf05/version 0.6.6 and its exact Gemma recipe were checked at
entry; that recipe is MLX 26/Apple, with no applicable CUDA comparator. Production
Gemma defaults, broader partial counts, unequal widths, corpus/depth and sustained
qualification remain open.
