<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Equal-width partial Gemma owner cohorts

The opt-in adapter now retains the original whole-cohort attention grid for
5/6/7/9/10/11 owners with equal padded reads. Its Gemma31 C5 recovery passes the
unchanged strict zero-margin and 160-target conditional-loss gates against the
retained FIRST stock reference. All 160 paid full heads are byte-exact; five
prefill frontiers differ, with no choice differences. Serving owner/joined
options remain off. Private serving-default changes are excluded from this unit.

| Complete C5 screen | Original quad4 + independent tail1 | Whole-five-grid candidate |
| --- | ---: | ---: |
| Positive-margin choice differences /165 heads | 8 | 0 |
| Reference-tie differences | 0 | 0 |
| Byte-exact full heads | 0 | 160 |
| Maximum raw full-head difference | 13.95315647 | 0.78764248 |
| Native mean conditional NLL /160 targets | 11.34001901 | 11.29974368 |
| FIRST stock mean conditional NLL | 11.30183937 | 11.30183937 |
| Relative conditional loss | +3.891784% (FAIL) | −0.209349% (PASS) |

The original primary comparison remains FAILED, followed by a separate successful
source-only post-authentication job. Recovery uses the same carrier, completed
prefix state, FIRST physical-five stock heads, zero margin allowance and 3% loss
threshold. It makes no scalar calibration, new oracle acquisition or allowance
change. Individual witnesses, token IDs and full vectors stay external; tracked
results contain aggregates and their raw receipts' hashes.

For whole logical count N, every real-root group retains the original grid B.
The shader clips work to its sequence interval while preserving global block
indices, metadata slots and original descending fixup arithmetic. Only active
cache-root addresses and the output origin are local. Groups have up to four
roots, offsets 0/4/8 and a final active count of one, two or three. All active
writers and all whole-wave read widths pass preflight before transformation.
Existing C1 and 2/3/4/8/12 paths retain their prior implementation. Product grouping
is unchanged. Startup sizing and both placement passes charge whole-grid scratch
and descriptor metadata. The helper retains its unchanged 183,615,488-byte
publication allowance; zero extra scratch is not claimed.

Unequal padded reads retain the prior fallback and remain unqualified. The original
MMA helper can preload and execute a final iteration beyond a shorter view's
KV_max-clamped interval; masks alone do not make such reads safe. This slice does
not widen read limits or initialize extra backing. The selected-plan counter
reports requested partial-cohort steps, not per-replay GPU launches.

Fourteen initial focused graph/plan/operator controls pass without skips.
A subsequent operator-only test covers eight N5/N6 cases: D256/D512, heads16/32,
read 1024, separately allocated real roots against contiguous physical-stream MMA.
Complete eager output and two poisoned capture replays match byte for byte, with
an exact bounded shared workspace. Actual GB10 SM48 plans retain B48 for D256
and B96 for D512. N5 exercises general fixup; N6 D256 is metadata-free and N6
D512 exercises uniform fixup, with active4+2. The initial configure failure from
an omitted implementation-catalog filename is retained; its one-file CMake
registration correction preceded the successful build.

Each candidate model arm retains 165 finite full heads, ten initialized
922,746,880-byte snapshots, layouts and carrier bytes exactly across own repeats.
The initial frontier state/layout/input matches the original C5 proof. The five
frontier NLLs differ, so total 160/165 byte-exact rows implies every paid row is
exact. Rows 0..31 predict supplied positions 992..1023 within each history; row 32
predicts 1024 and has no supplied target. This conditional score is separate from
a full 1,023-transition corpus. Actual selected counts are 120 owner/partial steps,
32 completed five-owner waves, 160 units and 32 replays.

Candidate paid walls are 3.85516/3.99148 s (136.32 ms spread); original native walls
are 3.82749/3.82524 s and retained stock 3.76917/3.77974 s. These are not contemporary
bookends and establish no speed or sustained-parity result. The combined
whole-five partition and F16 MMA tail recover this fixed C5 numerical screen;
no isolated precision-only or general model-quality causal claim follows.

The model helper a921f20a/SDKeb7a4ebc was compiled from measured source32 based on
b35cb0b plus the explicitly unadopted private serving recipe. Final core16 is
seeded from dc8b590, with the N6 test-only followup; all production core bytes match
the measured source. The private runtime/settings/HTTP and numerical harness
changes are attribution only. No runtime-default files are adopted in this unit.
N7/9/10/11, tail3/third-group proof, Gemma26 C5 transfer, unequal widths, remaining
model cohorts, full corpus, depth, live-serving defaults and sustained performance
remain open. The earlier strict C2 result retains its separate scope.

The prior Gemma31 C3 short screen on the unchanged whole-three adapter also
passes zero-margin choices and its separate 96-target conditional-loss gate
(−0.124116%). All 96 paid heads are byte-exact; three frontiers differ without
choice differences. Its 99 finite heads and six initialized snapshots match
across own repeats before FIRST physical-three stock comparison. This earlier
source26/private-recipe measurement is not partial-cohort, Gemma26 transfer,
serving-default, full-corpus or speed admission. The aggregate receipts include
its source, own-freeze, comparison, four successful official jobs and six absent
owned containers; individual rows stay external.

The new `fattn_owner_partial_kernel.cuh` is an MIT/Apache-2.0 derivative of the
locked GGML code at b10964/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4.
Its tile-processing helper is unchanged; uniform/general fixups preserve the
original arithmetic and combination order with only sequence filtering and local
output addressing. The normalized original fixup range has SHA256
d5583c5358e46bbc137347a80c44bcdc3994f246fb78bbc3b18085de6bdc9be8.
The file retains both copyrights and SPDX terms; [GGML provenance](../../upstream/ggml.md)
records the port. [Aggregate receipts](results.json) bind source, proof, model,
comparison, failed attempts and owned analysis retirement.

The subsequent [primitive and Gemma26 transfer](../gemma-partial-owner-transfer/README.md)
adds exact N7 tail3/N9 third-group controls. Gemma26 C5 retains two strict
positive-margin differences despite a passing independent 160-target loss gate;
its strict qualification remains open. These later results do not change the
Gemma31 recovery or adopt serving defaults.
