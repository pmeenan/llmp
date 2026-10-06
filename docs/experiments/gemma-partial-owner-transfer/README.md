<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Partial-owner primitive coverage and Gemma26 C5 transfer

The expanded equal-width attention operator proof passes, but the first Gemma26
C5 comparison fails its strict zero-margin gate: two positive-margin choices
differ among 165 complete finite heads, with no reference ties. The independent
160-target conditional-loss gate passes. This step changes only the operator
test; serving defaults and all production kernel bytes remain unchanged.

| Gemma26 C5 gate | Result |
| --- | ---: |
| Positive-margin choice differences /165 heads | 2 (FAIL) |
| Reference-tie differences | 0 |
| Byte-exact complete heads | 64 |
| Maximum raw full-head difference | 0.714227438 |
| Mean native conditional NLL /160 targets | 10.6615032000 |
| Mean FIRST-stock conditional NLL | 10.6615075323 |
| Relative conditional loss | −0.000433228% (PASS) |

Both differences occur after paid decode; their reference margins range from
0.131092906 to 0.260654688. Individual rows, token IDs and vectors remain external.
The failed primary comparison is retained with supervisor rc 1, one successful
step out of three queued, followed by a separate successful authentication-only
job for the skipped final command. Nothing was rescored, reacquired or given a
new margin allowance. Conditional loss does not waive strict choice failure.

The unchanged a921f20a helper uses the approved 26B profile: max rows 1024,
context 4096, head cap 5, plain norm, norm/RoPE, norm/add and all MoE routing/reduction
policies. Row-invariant products, shared Q8 and RoPE-store remain off. Five fixed
checkpoint-qualified histories receive 992 input rows followed by 32 supplied
positions 992..1023. Native decode has one five-column product wave, real-root
groups 4+1 and the original whole-five attention grid; actual selected counts are
60 owner/partial steps, norm 121/RoPE 60/ADD 90/route 30/reduce 30. All 32 waves and 160
units complete, with 32 graph replays and the unchanged 183,615,488-byte publication
allowance.

Before the first physical-five stock acquisition, native own repeats match all
165 full-vocabulary heads (173,015,040 bytes), ten initialized 230,686,720-byte
state snapshots per arm, both layouts and the full 49,152-byte carrier. The
reference likewise repeats complete head and carrier bytes exactly. Its normal
physical-five decode uses total context 20480, batch/ubatch 1024, F16 local 2048 and
global 4096 caches, SWA-full and unified KV off. Actual 25 local and 5 global layers,
32 completed waves/160 units, 31 graph reuses and explicit public teardown are
admitted. All six owned analysis/reference containers are absent after checked
retirement. Snapshots are native own-repeat evidence, not native/stock state
identity.

Retained row 0 predicts supplied position 992; rows 0..31 supply exactly 160
within-history FP64 full-vocabulary conditional NLL targets. Row 32 predicts 1024
and has no supplied target. The independent 3% conditional bound is unchanged;
this is not a full 1,023-transition corpus score. The analyzer's only identity
change permits this explicitly declared profile 26/count 5 FIRST acquisition after
native own freeze; profile 31 recovered-reference comparisons still require the
fixed retained FIRST identity. All numerical loops and strict gates are unchanged.

Native paid times are 1.21519/1.22123 s (6.04 ms spread); stock times are
1.23460/1.23090 s (3.70 ms spread). Means are 1.21821/1.23275 s, a 14.54 ms or 1.17948%
short difference. Native arms preceded the stock pair; these are not bookended
or sustained performance evidence, and no parity claim follows.

The operator-only expansion preserves the prior N5/N6 operands and assertions,
adds N7's three-owner tail and N9's third real-root group, and tests heads 16/32 at
D256/D512 with read 1024. One test passes 16 shapes and 36 actual plan records,
without skips. Eager output and two poisoned capture replays match original
contiguous physical-stream MMA byte for byte. Actual GB10 SM48 plans keep full B48
for D256 and B96 for D512. N7 exercises general fixup; N9 heads 32/D256 exercises the
metadata-free path and its other shapes exercise general fixup. The retained N6
D512 shapes exercise uniform fixup. Every group uses a bounded shared workspace;
recorded implementation scratch is not a claim about total peak memory.

The measured helper/SDK remain a921f20a/eb7a4ebc from source 32 at b35cb0b plus the
explicitly unadopted private serving recipe. This task builds only the expanded
operator target on 89e087f; it does not rebuild or adopt those private defaults.
The existing [partial kernel's provenance](../gemma-partial-owner-attention/README.md)
and MIT/Apache-2.0 attribution are unchanged. [Aggregate receipts](results.json)
bind all six official jobs, the source and own-freeze identities, raw comparison
and retirement proofs. Latest TensorFold primary cb2ebf05/version 0.6.6 was checked;
its pinned Gemma recipe remains MLX 26-only, with no applicable CUDA comparator.

The earlier Gemma31 C5 recovery remains a separate pass. Current 26 C5 strict
qualification is open. Earlier 1024-row routing diagnostics provide a possible
prefix-state explanation, but this 992-row reference's actual routing selection
was not observed; no causal claim or production layer policy follows. Model N7/N9,
other partial counts, unequal padded reads, full corpus, depth, live defaults and
sustained performance remain separate gates.
