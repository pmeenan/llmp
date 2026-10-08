<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek inverse rotation and output-A prefix

The existing ds4 output-A core is a useful private performance factor:
it combines inverse RoPE, F16 input staging, Q8 weight dequantization and
the grouped product, emitting the low-rank result in canonical order.
The whole 8K native pipeline gains 4.08% throughput even when every
invocation pays weight-plane packing and an extra copy back into the
original graph's output. Its arithmetic differs from native MMQ.
The original-checkpoint 32K forced trajectory passes the registered greedy
bound and repeats all full logits exactly; its native baseline fails one
step. The HCA/output-A combination also passes the registered 128K
held-out perplexity condition and the frozen 127K-token variable-binding
answer task, and passes the original-checkpoint 128K forced trajectory.
These are focused attribution/quality results. The guarded native adapter
also gains 4.97% in its own screen with exact candidate logits.
**The runtime now serves DeepSeek with output-A/HCA on every prefill chunk
of 64 rows or more by default.** On full 4,096-row chunks every registered
quality control passes on both GGUFs. Partial chunks fail the strict 32K
history at two rows but pass under the owner's tie-aware rule
([default-on acceptance](#default-on-acceptance),
[tie-aware re-scoring](#tie-aware-re-scoring-and-partial-chunks)).

## Captured first-layer screen

The prior ordered profile charged 392.734 ms to inverse RoPE, 804.997 ms
to output-A and 97.567 ms to the layout copy across the 8K pass: 1.295299 s
of a 13.308827-s whole wall, a 9.73% ideal ceiling rather than a predicted
gain. A copy-only change could remove at most 0.74%, so it was ruled out
without another build/model run.

The ordinary graph supplies one 4,096-row window layer's unrotated F32
attention heads, I32 positions and raw Q8_0 output-A weights. A clone
retains the native inverse-RoPE → grouped MMQ → layout-copy descriptors,
selected implementation names, dimensions, strides and operation parameters.
The candidate uses the existing `RunDs4OutA` core with no D4 emission.
Every timed call includes raw 34-byte Q8 block packing into separate
scale/code planes and the kernel's rotation-table zero/preparation.
All physical planes are checked against the original weight bytes.

| Seven-repeat median | Native before | Candidate | Native after |
| --- | ---: | ---: | ---: |
| Paid wall, ms | 14.774397 | 8.115490 | 15.079596 |
| GPU interval, ms | 14.763840 | 8.109536 | 15.032960 |

The candidate reduces prefix latency by 45.6321% (+83.9322% throughput);
the native bookends move 2.0657%. Both complete native outputs repeat
byte for byte. Across all 33,554,432 candidate low values, 33,554,345
change: maximum absolute difference 0.031732, RMS 0.0043611, NMSE
0.00010647. These are descriptive arithmetic differences, not a quality
gate. The unchanged original graph resumes after the screen and produces
both known native full frontier heads exactly.

Common clone allocation/input setup is separately recorded at 55.370 ms
and 1,377,845,248 bytes. Candidate extra allocation plus first packing/run
takes 20.528 ms and adds 170,917,888 bytes. Native scratch is 151,013,376
bytes within the existing 190,840,832-byte pool. The allocation figures
are outside the repeated operator intervals; no cold-speed claim follows.

## Whole 8K factor

Three fresh Spark A processes use the same community IQ2_XXS artifact,
literal 8,192-token input, 8K state ceiling, two 4,096-row chunks and
complete 129,280-value F32 frontier head per chunk. All retain the actual
selected Q-head fusion, native sparse attention/cache stages, compact
experts, ordered reduction, head and original output-B. HCA and Q2 D2R
remain off. The two ordinary plans are prebound outside the paid clock;
input assembly/copies, every layer, dispatch, fences and head copies are paid.

| Original before, s | Candidate, s | Original after, s | Throughput gain | Bookend movement |
| ---: | ---: | ---: | ---: | ---: |
| 12.376010162 | 11.915388466 | 12.428143403 | 4.08454% | 0.42124% |

The original placement can reuse the attention input once native RoPE
finishes. The diagnostic therefore uses one disjoint candidate low buffer
and a paid same-stream copy into the original CONT destination. It keeps
the original 3,527-step plans and activation placement in both arms.
One 170,917,888-byte scale/code/table/low allocation is initialized before
the paid clock in both arms (6.09–6.56 ms). Every one of the candidate's
86 layer/chunk prefixes pays packing, table zero/preparation, the core
and its 134,217,728-byte copy back. There are no persistent 43-layer
weight replicas, prepacked-weight credit or allocations in the hot loop.
All arms retain 1,530,920,960 activation bytes, 190,840,832 scratch bytes,
264,126,464 state bytes and 207,618,048 host staging bytes.

All four ordinary full heads match the previously qualified native hashes.
Both candidate vectors are finite and contain nonzero values; all 129,280
logit values change in each chunk. Maximum absolute differences
are 0.985439 / 0.587285, RMS 0.146911 / 0.116283 and NMSE
0.000945511 / 0.000408083. The two argmax IDs remain 1393 / 554.
These two frontier predictions do not establish task or context quality.
One bookended pass establishes a representative speed signal, not a
confidence interval, context matrix or same-quality acceptance.

## Accuracy diagnostic and transfer

On 72 fixed coordinates from the captured first-layer operands, a common
FP64 inverse rotation of the original F32 heads followed by exact original
Q8 scale/code products and `math.fsum` gives native MMQ RMS error
0.00365744 versus candidate 0.000135466, about 27× lower for the candidate.
Maximum errors are 0.0109526 / 0.000313467. This small numerical control
supports activation quantization as a cause of the native/candidate
difference on these points. It does not replace model quality tests.

The reusable pieces are inverse rotation during product input staging,
register dequantization of Q8 weights, and direct canonical output.
This measured core requires G8, K4096, rank1024, head512 and normal64-tail
rotation. Qwen3.8 and EXL3 use different projection/quantization contracts;
they cannot select this kernel by name. Compare eligible preparation and
layout pieces independently, and preserve their own precision, strides,
workspace and completion contracts when transferring the technique.

## Focused 32K quality method

The fixed-history comparison uses the original
`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93` UD-Q2_K_XL artifact
`8a355bfb…`, whose output-A projections have the same guarded Q8_0
geometry. The saved oracle is llama.cpp b11254 (`8019dc563`), flash
attention, all layers on GPU, one sequence, context 262,144, no prompt
reuse, fit off and five log-probabilities. Llmpalooza holds capacity 32,768,
chunk size 4,096, compact scheduling, frontier heads and HCA off in both
arms. Seven full chunks replace 301 output-A prefixes in the candidate;
the existing 3,032+1 remainder and 511 one-row decode steps stay native.
Both fresh processes consume the identical 31,705 prompt IDs and 512
forced oracle IDs and retain 512 complete 129,280-value F32 logit rows.

The registered oracle near-tie bound remains 0.947 nats. A differing
argmax in the oracle's top five has an exact oracle margin. An argmax
absent from those five has only a lower bound from the fifth log-probability;
that can establish a violation when above 0.947, and otherwise remains
unresolved. Baseline and candidate verdicts are separate. Direct
native/candidate vector drift and native margins are descriptive controls.
This screen does not measure perplexity, real-answer quality or a new
matched performance comparison.

The first fresh ordinary/candidate runs give:

| Against the same-checkpoint oracle, 512 rows | Equal | Within-bound near ties | Outside bound | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Current native, HCA off | 492 | 19 | 1 | 0 |
| Output-A candidate only, HCA off | 490 | 22 | 0 | 0 |

The native exception is step 249, absolute position 31,954: native chooses
10386 while the oracle chooses 82437 with an exact 2.616249211-nat
margin, exceeding the bound by 1.669249211 nats. The candidate chooses
82437 at that step. Its largest oracle near-tie margin is 0.725294083
nats, 0.221705917 below the unchanged bound. All candidate differences
have exact oracle margins; no top-five lower bounds are needed in this run.
The native baseline at this chunk size therefore fails the fixed oracle
bound, while this candidate passes it. The raw aggregate `quality_pass`
field requires both arms to pass and is false; individual verdicts are
reported separately.

Direct native/candidate argmaxes agree at 489 of 512 rows. Their 23
disagreements have native margins 0.00295639–0.81016922 nats. Across all
66,191,360 values per arm, 66,191,323 numeric values change; RMS is
0.298776545, NMSE 0.002602943 and maximum absolute difference 8.41839504.
Both full vectors are finite and their actual 7/2/511 chunk counts and
301 candidate prefixes match the protocol. Prefill/decode clocks are
48.3284/24.3297 s for native and 45.7612/24.4302 s for the candidate;
these single quality-run clocks are descriptive, without a new bookend
or matched throughput claim.

One fresh no-build candidate repeat produces all 66,191,360 logit values
byte for byte and the same 512 argmaxes, actual execution counts and
budgets. Its complete-logit SHA256 is
`dcd660fb70f8bdc58b2df616d8454a0b5f478f7900d720db1c29b1a8c1c6e9a8`.
The two candidate processes therefore pass this fixed-history
greedy/repeat control. The repeat's prefill/decode clocks are
45.6086/24.2922 s, again descriptive. No broader answer, PPL or
context-quality conclusion follows from the isolated FP64 sample or this
one fixed trajectory.

## Optional HCA interaction, one new arm

The already qualified output-A/HCA-OFF trajectory is the control for one
fresh HCA-ON process. Only the private guard and existing `--ds4-hca`
option change; the original checkpoint, history, output-A math, other
operations and tail/decode paths are retained. All 140 eligible HCA
launches and 301 output-A prefixes execute in the seven full chunks;
the two tails and 511 decode chunks keep native attention/output-A.

HCA plus output-A has 492 oracle-equal choices, 20 within-bound near ties,
zero outside-bound differences and zero unresolved margins. Its largest
oracle margin remains 0.725294083 nats and step 249 remains 82437.
It passes this fixed-history greedy comparison. A fresh no-build
combination repeat matches all 66,191,360 finite logit values byte for
byte and all 512 choices, with identical actual counts and budgets.
Its SHA256 is
`a2d70ac4928fb8b7d8d9419924112620b621e1e413de1d39d0ec17aff5461930`.
The combination passes this greedy/repeat control; broader/default
acceptance remains open.
Against output-A/HCA-OFF, 489 choices agree and 23 differ; complete-vector
RMS is 0.293928455, NMSE 0.002545173 and maximum absolute difference
4.33767986. The single ON prefill/decode clocks are 40.9253/24.2481 s;
these are descriptive quality-run clocks, without a new paid bookend.
The fresh repeat's clocks are 40.7895/24.2785 s, also descriptive.

After the ON model retires, a single streaming pass scores the same 512
oracle-chosen continuation IDs under each saved full native logit vector
using stable F64 log-sum-exp. The oracle's saved normalized log-probability
for each chosen ID supplies its reference likelihood.

| Oracle-forced continuation, 512 tokens | Mean NLL, nats | Conditional perplexity |
| --- | ---: | ---: |
| b11254 oracle | 0.285054170 | 1.329834064 |
| Current native, HCA off | 0.295788194 | 1.344185420 |
| Output-A only, HCA off | 0.304861407 | 1.356436998 |
| Output-A plus HCA on | 0.295742299 | 1.344123731 |

This measures likelihood on the oracle's own generated continuation,
not the registered held-out perplexity corpus or gate. The interaction's
conditional likelihood is almost native's, but these aggregate numbers
do not prove real-answer or broader-context quality.

## Registered 128K held-out perplexity

One fresh HCA/output-A process uses the original UD-Q2_K_XL checkpoint,
the existing 131,072-token held-out corpus and the registered last-half
scoring window. The unchanged reference is 1.9298 with a symmetric 3%
bound; no baseline is rerun. All 131,071 F64 losses are finite. Exactly
65,535 losses at indices `[65536, 131071)` give mean NLL
0.6569195756 and perplexity **1.928841523**, **−0.0496672%** relative
to the reference, passing this registered condition.

The 32 full 4,096-row chunks execute 1,376 output-A prefixes and 160 HCA
operations. Only the first eight chunks' 256 compressed cells admit the
existing HCA guard; later 512/768/1,024-cell shapes keep native attention.
There are no tail or decode chunks. This qualifies that actual mixed path,
not HCA across every 128K layer. The run takes 256.275 s for all-head PPL,
with 170,917,888 diagnostic extra bytes and 20,647,096,320 bytes available
at the sampled memory low. These single-run clocks do not establish a
matched 128K speed gain. The native graph adaptation and default
acceptance remain open.

## Frozen long-context answer control

One combined-path process uses the existing R3 chronological variable-binding
task: 126,976 frozen prompt IDs, context 131,072, output cap 128 and
authenticated EOS token 1. The tokenizer, template, facts and exact JSON
rubric are unchanged. The saved native 2,048-row result passed; it is not
rerun. The new 4,096-row process also passes, producing exactly
`["v_seed42", "v_hop42a", "v_hop42b", "v_hop42c", "v_hop42d"]`
and EOS after 34 answer tokens. All 35 raw generated IDs match that saved
native result; all 4,524,800 saved logits are finite and each row's argmax
agrees with its generated ID.

The actual path executes 31 full chunks, 1,333 output-A prefixes, 160 HCA
operations and 34 native decode chunks, with no tail. Its 197.3284-s
prefill and 1.7561-s decode clocks are descriptive; the saved native uses
a different chunk size. This is one positive synthetic answer control,
not a broader task-suite result.

The model completed rc0 and retired before its controller rejected two
identical duplicate Boolean keys in the private benchmark's stop summary.
A bounded no-model/no-build recovery admits only the top-level duplicates
`compact_experts=true` and `exact=false`, preserves the original summary
and scores a canonical copy with the same native exact-answer helper.
No answer, EOS, count or rubric changes. Raw/canonical summary SHA256:
`fdb98f3a1a961d6a0d0dc2f327eb23c1e5c567a825f7cba0f3212b5808f9ebb3` /
`41b0a85a1a5cb7d0ceacca4f5695c8640c9018cc91c1d7cbb8d797972c80831e`.

## Original-checkpoint 128K forced trajectory

One fresh combined-path process uses the existing 128,821 prompt IDs,
512 oracle-forced continuation IDs and saved same-checkpoint b11254
oracle. The unchanged 0.947-nat bound gives **500 equal choices, 12
within-bound near ties, zero outside and zero unresolved**. Every
difference has an exact top-five oracle margin. The largest is
0.475185394 nats, 0.471814606 below the bound. Step 315 selects oracle
token 295 exactly. All 66,191,360 logits are finite and each complete
row's argmax agrees with its summary.

The actual path executes 31 full 4,096-row chunks, 1,333 output-A prefixes
and 160 HCA operations. The authenticated `ChunkRows` loop then executes
1,840 and 5 native prompt-tail rows, followed by 511 one-row native decode
chunks. The raw private counters report zero tails and 513 decode chunks
because their old 31,705-position cutoff labels both tails as decode.
The model finished rc0 and retired before that reporting assertion failed.
A bounded no-build/no-model recovery preserves the raw summary and
derives the exact row sequence from the executed source and fixed input;
the oracle, logits, scoring and bound are unchanged.

The full-logit SHA256 is
`d842b48cac15b06197996619764c6acc61ce070c66848310a4961b586146160f`.
The 203.2561-s prefill and 25.7354-s decode clocks are descriptive quality
run clocks. This condition passes its fixed-history greedy bound; it has
no fresh deterministic repeat, matched performance bookend or additional
task matrix. Production/default acceptance remains separate.

## Native graph restoration

The selected adapter makes the prefix a direct native graph operation,
`llmp.dsv4.outa_prefill`, behind a default-off benchmark option. It
accepts only the qualified GB10, 4,096-row packed Q8_0/F32 shape. Exact
and unsupported shapes retain the original inverse RoPE, MMQ and layout
copy. The three direct dependencies retain unrotated heads, positions
and raw weights through the disjoint canonical output write. Weight
packing and rotation-table preparation remain paid on the provider stream;
no persistent weight replicas, D4 emission or output-B change are added.

One fresh native OFF/ON/OFF screen uses the same community artifact and
literal 8K input. Actual plan creation, binding, input copies, all layers,
dispatch, fences and full frontier-head copies are inside the paid clock.
Both arms also pay the same preallocated host retention of the two complete
heads; file writes and plan audit are outside. This scope differs from
the earlier prebound diagnostic, so its absolute clocks are separate.

| Native OFF before, s | Native ON, s | Native OFF after, s | Throughput gain | Bookend movement |
| ---: | ---: | ---: | ---: | ---: |
| 12.461957835 | 11.887680241 | 12.496068163 | 4.97433% | 0.27372% |

All four OFF full heads match the original goldens. Both ON full heads
match the earlier qualified output-A-only diagnostic byte for byte.
Each ON chunk executes 43 native output-A operations; the two actual
plans have 3,441 steps versus 3,527. Each retains 43 Q-head, weighted
reduction, compact paired and compact down consumers, with no HCA calls.
Activation extent remains 1,462,430,464 bytes (the ordinary benchmark's
1,828,716,544-byte allocation includes its usual margin). Required scratch
falls from 151,013,376 to 134,414,336 bytes; allocated scratch falls from
190,840,832 to 169,869,312. The prefix's reused 36,700,160-byte scratch
lives inside that pool. The diagnostic 170,917,888-byte extra plane/low
allocation and copy back are gone. Sampled peak memory is 0.999983× the
larger OFF reference. This selects the guarded native implementation for
source qualification; no runtime default, 2,048-row support or broader
performance claim follows.

The selected source passes the full locked Spark set (1,248 tests, including
259 GPU tests), the SDK's format and clang-tidy checks on changed units,
and the portability boundary check. Routine declaration-name and integer
style repairs are followed by the five affected descriptor/planner,
module-digest and source controls; the full suite is not repeated.
REUSE and header checks pass on the final source snapshot using the retained
pure SDK REUSE shim: 1,153 headers across 1,274 paths, no findings.
The CU lint parser uses the documented RE-042 include adaptation; the actual
NVCC13.4 compile flags and numerical core remain unchanged.

## Native attention interaction screen

One native HCA OFF/ON/OFF screen keeps output-A ON in all three arms,
using the same community artifact, literal 8K input, two 4,096-row chunks,
frontier heads and compact experts. It reuses the qualified production
libraries unchanged; only the benchmark's flag guard and actual-plan
audit change. Planning, binding, input copies, every packing/table and
attention preparation, dispatch, fences and complete head copies remain
inside the paid clock. Preallocated host head retention is identical;
file writes and audits remain outside.

| HCA OFF before, s | HCA ON, s | HCA OFF after, s | Incremental throughput gain | Bookend movement |
| ---: | ---: | ---: | ---: | ---: |
| 11.921261671 | 9.879234107 | 12.024281101 | 21.19129% | 0.86417% |

Each actual plan retains 3,441 steps, 43 output-A, Q-head, ordered
reduction, compact pair and compact down operations. The ON arm selects
20 HCA operations per chunk, 40 total; all arms execute 86 output-A
operations. Activation extent remains 1,462,430,464 bytes, required scratch
134,414,336 and allocated scratch 169,869,312. There is no diagnostic GPU
allocation. Sampled peak memory is 1.002878 times the larger OFF reference.

All four OFF full heads match the qualified output-A-only goldens exactly.
Both ON heads are complete and finite, retaining argmax 1393/554. Against
the first OFF arm their maximum absolute differences are 0.584370/0.707085
and RMS differences 0.114348/0.122808; this drift is descriptive. This
screen establishes an incremental native speed signal, separately from
the earlier original-checkpoint quality controls. It does not add a new
quality pass, 2,048-row support, serving default or matched ds4 speed ratio.

The supervised job completes rc0 in 82 seconds including the small private
compile and three model loads. All command children are reaped; the final
strong gate records 117.222 GiB with native/GPU/container probes clear.
Raw heads, actual plans, commands and receipts remain in
`/home/pmeenan/scratch/m3-ds4-qhead-short-records/outa-hca-native-r1/`,
with its supervisor records in the sibling `outa-hca-native-r1-supervisor/`.

## First 2,048-row geometry screen

A private two-unit derivative admits only 2,048 or 4,096 packed rows,
requires matching position/output rows, and passes that checked row count
to the unchanged staged core. It retains the conservative 36,700,160-byte
scratch requirement. The selected 4,096-row sources are preserved before
the warm build; neither main nor the isolated production sources change.

One native 8K OutA OFF/ON/OFF screen keeps HCA ON in all arms, with four
2,048-row chunks and the same community artifact/input/frontier/compact
conditions and paid planning/dispatch/copy scope as above.

| OutA OFF before, s | OutA ON, s | OutA OFF after, s | Incremental throughput gain | Bookend movement |
| ---: | ---: | ---: | ---: | ---: |
| 12.116122736 | 11.666596215 | 12.137733770 | 3.94573% | 0.17837% |

Every arm executes 80 HCA operations; the ON arm executes 172 output-A
operations. Four actual plans retain 43 Q-head, ordered reduction, compact
pair and down consumers each. The smaller ring has 173,948,928 state bytes.
Activation extent stays 722,860,800 bytes; required scratch falls from
75,515,904 to 67,207,168, and allocated scratch from 96,468,992 to 85,983,232.
Sampled peak memory is 0.997174 times the larger OFF reference.

All eight fresh OFF heads match their chunk's bookend bytes. The four ON
heads are complete and finite and retain argmax 4084/1393/295/554. Maximum
absolute drift is 3.109016/0.644032/1.404508/0.684175; RMS drift is
0.554431/0.123842/0.266103/0.132872. This selected the focused original-checkpoint
quality check below; the speed signal alone does not establish acceptance.
The one supervised build/three-model screen completes rc0 in 87 seconds;
all children are reaped and strong retirement records 117.231 GiB clear.
Raw results are retained in the sibling `outa2048-native-r1/` and
`outa2048-native-r1-supervisor/` evidence directories. No suite or repeat
matrix was run for this private screen.

The candidate-only 32K fixed-history check uses the original checkpoint,
the same 31,705 prompt IDs and 512 oracle-forced IDs, and the unchanged
0.947-nat bound. It executes fifteen full 2,048-row chunks, a native
985-row tail and 511 native one-row decode chunks: 645 output-A and 300
HCA operations. All 66,191,360 retained logits are finite and all 512
reported argmax IDs match the full vectors.

The result is **497 equal, 14 within the near-tie bound, one outside and
zero unresolved**. At step 249 (position 31,954), the candidate chooses
10386 instead of oracle 82437. Both IDs are present in the saved top five;
the exact oracle margin is 2.616249211 nats, exceeding the bound by
1.669249211 nats. This is the same failed discriminator previously seen
without the qualified 4,096-row combination. The 2,048-row extension is
stopped without a candidate repeat, production adoption or default change;
the separate passing 4,096-row results above remain distinct.

The quality job completes rc0 in 99 seconds including the private host
compile, model load and scoring; all children are reaped and the final
strong gate records 117.269 GiB clear. Raw evidence remains in
`outa2048-quality-r1/` and its sibling supervisor directory. The complete
logit SHA256 is
`52c0b582b45e7dcb55f00b2b9a8fca9347cad5d57eae926a112a2df348a805a6`.

## Remaining native 4K stage budget

The existing ordered-step collector is repeated once on the current
community 8K combination: native output-A and HCA, compact experts and
frontier heads, two 4,096-row chunks. The two rejected 2,048-row guard
changes are first replaced with the retained selected 4K sources and
their two libraries rebuilt. The collector then prebinds the same complete
plans in all arms and runs O/observed/O within one loaded model. It pays
input assembly/copies, operation preparation, dispatch, fences and complete
frontier-head copies; planning/event creation, audit and file writes are
outside the paid clock. It adds no intermediate synchronization.

The three paid passes take **9.810647619 / 9.762510701 / 9.754978414 seconds**.
Bookends move −0.56744%; the observed pass is −0.20753% from their mean.
All six complete heads match the saved native-combination goldens byte
for byte. Actual plans retain 43 output-A and 20 HCA operations per chunk,
plus the selected Q-head, reduction and compact expert consumers.

| Chain | Summed ordered intervals, ms |
| --- | ---: |
| Routed FFN | 3,790.039 |
| Attention output and hyper-connection expansion | 1,413.764 |
| Attention | 1,036.031 |
| Q/KV | 945.274 |
| Shared FFN and hyper-connection expansion | 701.674 |
| FFN input and routing | 634.082 |
| Hyper-connection attention input | 549.042 |
| Compression and indexer | 517.963 |

Ordered intervals sum to 9,655.759 ms within the observed 9,762.511-ms
whole pass; they include host enqueue gaps and are not kernel-busy time.
The compact pair/down consumers account for 1,870.319 / 1,549.578 ms,
3,419.897 ms together (35.03% of the observed wall). They are the largest
remaining measured consumer target. Output-A accounts for 716.212 ms;
wide attention/HCA account for 680.275 / 341.796 ms. These are current
within-pipeline budgets, not differences subtracted from older literal
pipeline profiles or predictions of removable time.

The supervised restore/build/collector/classifier job completes rc0;
all command children are reaped. The final strong gate records 117.111 GiB
with native/GPU/container probes clear. Event rows, full heads and receipts
are retained in `combined-attribution-r1/` and its sibling supervisor
directory under the same evidence root. No suite or context matrix is run.

## Native runner integration control

The normal paged DeepSeek runner now has an internal, default-off request
for the qualified 4K combination. Output-A retains its shape/type/device
fallbacks. HCA requires all 43 actual output-A insertions and the qualified
4,096-row/256-compressed shape; later full chunks retain output-A with
ordinary HCA. Exact mode, full-window diagnostics and other row sizes
retain ordinary operations. The position scalar is part of the eligible
plan's cache key and is checked against all actual host positions before
embedding, copies or dispatch. No public runtime configuration or default
changes here.

One genuine-runner control uses the frozen community 8K IDs, capacity
12,288, two 4K chunks and three common one-row continuation steps. The
existing Frontier control runs all-head/frontier/frontier/all-head, without
DSpark or decode graphs. Both frontier repeats' two complete vocabulary
heads match the previously qualified native-combination goldens byte for
byte. Each all-head and frontier arm repeats its own heads exactly; all
four prefills leave identical canonical target-state snapshots (all
initialized extents, with unused extents represented by zeros), and their
common continuation logits are exact. The state digest is
`2fe394c7f75cceae5d50b95c75c92429e20c4ca7f2a7aefee82bd270d8be6d75`.

The completed-plan audit records 43 output-A, 20 HCA, 43 Q-head,
43 reductions and 43 compact pair/down consumers for each of eight
full chunks, with the correct distinct cache positions 0 and 4,096.
All twelve one-row continuation chunks have zero output-A/HCA calls.
This is a runner/state control, not a new performance or answer-quality
comparison; diagnostic head retention and audit work are included in its
reported times.

A separate **allocation-only** 128K invocation loads no weights. It checks
all 32 full 4K shapes against the five production accounting probes,
including the new 32,768-position ordinary-HCA fallback probe. The extra
audit shapes do not enlarge funding. All raw activation/scratch/input
maxima fit: **1,682,631,424 / 134,414,336 / 103,082,752 bytes**. Funded
activation and scratch allocations are 2,103,443,456 / 169,869,312 bytes.
Actual selection is 1,376 output-A calls and 160 HCA calls: HCA runs on
the first eight chunks, with ordinary 512/768/1,024-compressed fallbacks
afterward. Registered weight-byte totals in this sizing report do not
represent reads in that invocation.

The supervised control finishes rc0 with every command reaped, and the
fresh handback gate records 117.190 GiB with native/GPU/container probes
clear. Raw controls and sizing are in `runner-control-r3/` and its sibling
supervisor directory under the evidence root. Two prior attempts stop
before model loading: a diagnostic missing header, then the harness's
sharded-only metadata filename assumption. Metadata selection now follows
the runtime's single-file fallback and first-shard preference. Selected
source now passes 1,258 locked Spark tests, including 259 GPU tests, and
its format/tidy/boundary checks. The final normal-header/archive build also
repeats all eight known complete heads, initialized state and common
continuation exactly; it contains no private runner or audit getter.
The [scoped IQ2 validation](../ds4-iq2-occ2/README.md) records the combined
source receipts and final compiled controls. Source-only
REUSE/header checking passes 1,168 headers across 1,290 source paths with
no problems. Any serving-default decision remains separate.

## Default-on acceptance

The owner's rule allows speed over bit-exactness only at the same quality.
Each configuration below runs the registered controls unchanged: the
original UD-Q2_K_XL checkpoint's 32K and 128K fixed histories against the
same-checkpoint llama.cpp oracle (0.947-nat bound, no row outside), its
128K held-out perplexity (3% around 1.9298), the frozen 127K answer task,
and on the community GGUF ds4's own 8K trajectory and the 32K perplexity
against ds4's. `llmp_dsv4_exec`, compact experts, frontier heads, Spark A,
2026-10-03.

| Configuration | 32K history (equal / near / outside) | 128K history | 128K PPL | Community 8K vs ds4 |
| --- | --- | --- | --- | --- |
| Current default (output-A/HCA off) | 492 / 19 / **1** (step 249) | — | — | 32 / 32 |
| Output-A/HCA on full 4,096-row chunks | 492 / 20 / 0 | 500 / 12 / 0 | 1.928842 (−0.05%) | 32 / 32 |
| Output-A/HCA on every chunk (partial tails too) | 491 / 19 / **2** (steps 249, 333) | 499 / 13 / 0 | (no tails) | — |
| Output-A alone on every chunk | 488 / 23 / **1** (step 306) | — | — | — |
| Output-A/HCA, 2,048-row chunks | 493 / 19 / 0 | — | — | — |

- **Full chunks pass every control.** Their logits are byte-identical to
  the earlier qualified runs: the 32K and 128K histories' full-logit
  hashes equal `a2d70ac4…` and `d842b48c…` above, and the 128K
  perplexity losses equal `b6d39c8d…`. The 127K answer task runs 31 full
  chunks and no tail, the same path, so its earlier pass stands. The
  community GGUF's 32K perplexity in 4,096-row chunks is 2.981815 against
  ds4's 2.980670 (+0.04%,
  [stage mechanisms](../ds4-prefill-stages/README.md)).
- **Partial chunks fail the 32K history.** With its 3,033-row tail also
  through output-A/HCA, two rows fall outside the bound: step 249 (oracle
  margin 2.62, the row the current default also fails) and step 333
  (1.02). Output-A alone on the tail fails step 306 (1.12). The same
  partial-chunk path passes the 128K history (its 1,840-row tail takes
  output-A; HCA is past its 256 compressed cells there) and 2,048-row
  chunks pass. Its likelihood of the oracle's continuation is level
  with the others: 1.344718 against 1.344124 (full chunks) and 1.344185
  (current default). These are near-tie flips of a strict per-row gate,
  but under the registered rule they fail, so partial chunks stayed off
  (`Dsv4Model::prefill_outa_hca_partial`; `llmp_dsv4_exec
  --outa-partial`). The owner's tie-aware rule admits them
  ([below](#tie-aware-re-scoring-and-partial-chunks)).
- **Repeats.** A fresh repeat of the every-chunk 32K run reproduces its
  logits byte for byte (`f5669a25…`).

**The default.** `llmp-runtime` turns the combination on for DeepSeek
(`Dsv4Options::prefill_outa_hca`), on full 4,096-row chunks, and since the
tie-aware rule on partial ones too (`prefill_outa_hca_partial`,
[below](#tie-aware-re-scoring-and-partial-chunks)). Its own guards keep
everything else on ordinary math:
- output-A takes only Q8_0 [4,096, 8,192] output-A weights on a GB10;
  unless every layer's are, the runner turns the combination off
  entirely, output-A included;
- HCA takes only chunks whose HCA layers' compressed cells are 256 wide
  (positions below 32,768);
- chunks under 64 rows, decode and verify chunks, and the reference mode,
  are unchanged. A DSpark model's injected prefill chunks take it like
  plain ones (acceptance [below](#dspark-with-output-ahca)).

**Plans are no longer keyed by position.** HCA reads the chunk's first
position as a host scalar. Each run now sets it on the cached plan's HCA
nodes and on the position its inputs authenticate
(`SetDsv4HcaFirstPosition`), so 4,096-row chunks of one shape at other
positions share a plan. Because the position changes each run, such a
plan is never captured as a graph, and setting a position on a captured
one is refused (a replay would keep the captured scalar). A verify never
takes HCA; its plan builder refuses a position.

**Aligned and unaligned chunks.** The registered controls prefill from
position 0, so their full chunks start at multiples of 4,096. Served
chunks often do not: a conversation's later turn starts after its reused
history. The registered controls cover the aligned case only. An A/B in
the Stages 3–4 review (original GGUF with DSpark, output-A/HCA on; short
prefixes of 16–63 rows, then two full 4,096-row chunks) found the shared
plan's frontier heads byte-identical to main's position-keyed plans at
every offset, so the mechanism is exact unaligned; quality at unaligned
positions rests on that equivalence, the kernel-level tests and the
runtime cells, not on a registered control.

**Through the runtime** (fresh service per cell, main `6052286` against
this build, same session):

| Cell | Community, main / new | Original, main / new |
| --- | --- | --- |
| 7K chat, C1: first token, s | 9.81 / 8.26 | 10.88 / 9.37 |
| 7K chat, C1: completed tok/s | 11.38 / 12.23 (+7.5%) | 11.30 / 12.12 (+7.3%) |
| 7K chat, C4: completed tok/s | 15.94 / 17.77 (+11.5%) | 14.72 / 16.43 (+11.6%) |
| Three-turn conversation, first tokens, s (7K, +7K, +7K) | 9.83 / 9.73 / 9.66 → 8.34 / 8.53 / 8.59 | 10.97 / 10.92 / 10.81 → 9.43 / 9.63 / 9.77 |

- **Multi-turn.** Later turns prefill a ~7K suffix at an unaligned
  position. Their first tokens stay 1.1–1.3 s ahead of main's, which
  plans each new shape too. A second conversation of the same first-turn
  shape reuses the plan (8.14 against 8.34 s first time).
- **Replies.** Replies change with the arithmetic (the replies are
  byte-identical across C1 and C4 within each arm). Decode rates are
  unchanged.

Records: `spark:~/scratch/dss3/` (`q32-*`, `q128-*`, `ppl128-4096`,
`comm-forced-8k`, `turns`, `records/long`).

### DSpark with output-A/HCA

With the drafter loaded, a prompt's prefill chunks inject the drafter's
ring, and an injected plan takes output-A/HCA like a plain one: on full
chunks, and since the tie-aware rule on partial ones too. Acceptance was
measured on the original artifact. (The drafter did not yet run with the
community one; that is fixed separately.)
- **Partial chunks too, through the runtime.** Fresh service per cell,
  full chunks only (`4a250b4` with the wave pair scan) against partial
  chunks too, same session, 7K chat:
  - C1: 15.09 → 16.12 tok/s completed (+6.8%); first token 9.4 → 8.3 s;
    decode after the first token 33.5 → 33.8 tok/s, so acceptance is
    level.
  - C4, corrected: 16.36 → 17.86 tok/s (+9.2%), completions 62.4–62.6 →
    55.9–57.3 s, both arms with main's round-robin schedule (a later
    session, main `2c42d6f` against the full-chunk build). The first C4
    cell here (16.86 → 17.92) was confounded. Its partial-chunk binary
    carried a stale object of the experimental oldest-prompt-first
    schedule: `rsync` kept the older source's time, so `ninja` did not
    rebuild it. That schedule, not the capacity policy, prefilled the
    prompts one after another. The cells logged no capacity waits, and the
    host read nothing from disk after the load (`/proc/vmstat` `pgpgin`).
- **Full chunks, through the runtime.** Fresh service per cell, main
  `6052286` against the full-chunk build, same session, 7K chat:
  - C1: 13.68 → 15.00 tok/s completed (+9.6%); decode after the first
    token 32.7 → 33.3 tok/s, so acceptance is level.
  - C4 (DSpark waves): 14.75 → 16.69 (+13.2%); decode 9.9–10.0 → 10.6–11.0
    tok/s a request.
- **`llmp_spec_runner --check greedy`**, the four 7K chat prompts, 32
  tokens each, output-A/HCA off against on: acceptance 86 of 111 (0.775)
  against 83 of 114 (0.728). These are short samples of different token
  sequences, so they are no evidence of a change. The short prompts, with
  no full chunk, are unchanged (0.556, 0.612, 0.778 in both arms). That
  run had output-A/HCA on full chunks only; since partial chunks take it,
  the short prompts' chunks of 64 rows or more take it too, and this
  check was not repeated.

Records: `spark:~/scratch/dss5/` (`h7`, `h11`, `h15`, `io-*.tsv`,
`r34/greedy-*`), 2026-10-03.

### Tie-aware re-scoring and partial chunks

The owner's tie-aware greedy rule (2026-10-03, D-085, refined after review
the same day) replaced the strict per-row gate. Each history is judged
against the model's pinned **reference run**. For DeepSeek that is
output-A/HCA on full 4,096-row chunks (32K +1.075%, 128K +1.127%); it stays
the reference although the shipped default is now every chunk, so the
reference does not ratchet. For Qwen3.8 it is `hq-32k-fast` and
`hq-128k-fast`. Changing a reference is the owner's decision. `judge.py`
also refuses a reference with an outside step past the per-step
tolerance, or more than two.
- **Outside steps.** At most max(2, the reference's), here 2.
- **Per-step tolerance.** Each outside step must have llmpalooza's own margin
  of its argmax over the oracle's token below the bound (0.947 nats), and
  its NLL of the oracle's token less than the bound above the oracle's
  own. This is an owner-accepted tolerance, not a calibrated test of a
  tie. At step 249, the one step it has decided, llmpalooza's paths spread
  0.65–1.13 nats of NLL excess; two runs of the phase-1 fast plan on one
  build gave 0.99 and 1.13 (RE-031). The bound sits inside that spread.
  From the oracle's side the step is no tie (probabilities 0.909 and
  0.066).
- **Continuation.** The oracle continuation's conditional perplexity ratio
  to the oracle's own may exceed the reference run's by at most 0.5
  points. That is twice the largest recorded repeat variation (0.25
  points, between those phase-1 runs; later repeats are bit-identical).
  The bound is relative because every engine sits above the oracle on the
  oracle's own greedy path, by an amount that depends on the pair:
  Qwen3.8's passing 32K control sits at +2.05%.
- **Unchanged.** Held-out perplexity (3%) and the answer task.

`judge.py greedy --reference` (long-context) reports both verdicts. Every
recorded run, judged again from its saved logits:

| Run | 32K: equal / near / outside | Outside steps: oracle margin, llmpalooza margin, NLL excess | 32K continuation | 128K: equal / near / outside, continuation | Strict / tie-aware |
| --- | --- | --- | --- | --- | --- |
| Full 4,096-row chunks (the reference) | 492 / 20 / 0 | — | +1.075% | 500 / 12 / 0, +1.127% | pass / pass |
| Output-A/HCA off (the previous default) | 492 / 19 / 1 | 249: 2.616, 0.041, 0.653 | +1.079% | — | fail / **pass** |
| Every chunk (partial tails too) | 491 / 19 / 2 | 249: 2.616, 0.458, 0.891; 333: 1.022, 0.578, 0.646 | +1.119% | 499 / 13 / 0, +0.875% | fail / **pass** |
| 2,048-row chunks | 493 / 19 / 0 | — | +0.509% | — | pass / pass |
| Output-A alone on every chunk | 488 / 23 / 1 | 306: 1.115, 0.097, 0.489 | **+1.748%** | — | fail / **fail** |
| Rejected: IQ3_XXS routed down, a warp's row | 491 / 20 / 1 | 249: 2.616, 0.808, **1.118** | +1.539% | 502 / 10 / 0, +0.430% | fail / **fail** |
| Phase 1 fast plan, run 1 / run 2 | 500 / 10 / 2; 497 / 14 / 1 | 249: 2.616, 0.625, **0.990** (333 a flip); 249: 2.616, 0.839, **1.133** | +0.825%; +1.077% | +1.386% (one run, 0 outside) | fail / **fail** |
| Phase 1 and 2 reference form (`--exact`) | 498 / 14 / 0 | — | +0.518%, +0.539% | — | pass / pass |
| Phase 2 fast plan, three runs | 493 / 19 / 0 | — | +0.610% (bit-identical) | — | pass / pass |
| Qwen3.8 32K fast (its reference), and its repeat | 474 / 38 / 0 | — | +2.046% | +1.451% (0 outside) | pass / pass |
| Qwen3.8 32K reference form (`--exact`) | 475 / 35 / 2 | 202: 1.875, 0.138, 0.710; 272: 1.875, 0.145, 0.992 | +2.466% | — | fail / pass |

Every recorded run that passed the strict rule passes the refined one, on
both models; the recorded verdicts stand. Qwen3.8's fast runs pass
against themselves by construction, as their own reference. Only its
reference form's pass is new information. The rule does not admit everything
the strict rule failed. Output-A alone fails on its continuation (+0.67
points over the reference). The rejected kernel variant
([deepseek-batching](../deepseek-batching/README.md#four-request-wave-step-and-scheduling))
and both phase-1 fast-plan runs fail on step 249's NLL excess. That step
sits inside the engine's own variation, so these verdicts follow the
owner's tolerance; they are not measured differences.

**Partial chunks by default.** Every chunk passes both histories. It uses
the whole flip cap (two) and is 0.044 points over the reference's
continuation. It matches full chunks on the held-out perplexity and the
127K answer task, which have no partial tail, and on the community 8K
trajectory, whose 8,192 tokens are two full chunks. So `llmp-runtime`
now sets `prefill_outa_hca_partial` too (`SetDsv4ServedPrefill`, pinned by
a unit test): output-A/HCA takes every prefill chunk of 64 to 4,096 rows.
The option stays as the off switch (`llmp_dsv4_exec --outa-partial`
turns it on in the harness). On this build the every-chunk 32K and 128K
histories repeat their logits byte for byte (`f5669a25…`, `55162439…`).

Through the runtime: fresh service per cell, the previous default (full
chunks; the wave pair-scan build) against partial chunks too, same
session.

| Cell | Community | Original |
| --- | --- | --- |
| 7K chat, C1: first token, s | 8.27 → 7.26 | 9.36 → 8.34 |
| 7K chat, C1: completed tok/s | 12.23 → 12.86 (+5.2%) | 12.15 → 12.76 (+5.0%) |
| 7K chat, C4: completed tok/s | 18.52 → 19.88 (+7.3%) | 16.77 → 18.19 (+8.5%) |
| Three turns, first tokens, s (7K, +7K, +7K) | 8.33 / 8.52 / 8.61 → 7.39 / 7.79 / 7.95 | — |

Replies change with the arithmetic of the 2,947-row tail. Decode rates
are unchanged.

Records: `spark:~/scratch/dss5/` (`rejudge2`, `tie`, `h8`, `t8`; the
earlier records in `~/.local/share/llmp/m3lc/raw/`), 2026-10-03.

## Provenance

Measured on `spark-c4e2`, 2026-10-02, SDK
`aarch64-e0a0c85c42806fb1` (Clang22.1.8/NVCC13.4.92), SASS `sm_121a`,
the actual native math flags and pinned cuBLAS payloads. Original source,
static/compiler/library identities, fixed inputs, complete outputs and
successful supervised retirement are retained. The final whole-factor
gate finds 117.129 GiB available; fixed-quality and fresh-repeat terminal
gates find 117.324 / 117.290 GiB; the HCA interaction terminal is
117.298 GiB, and its fresh repeat is 117.288 GiB. The registered PPL
condition retires at 117.306 GiB; the R3 model/recovery retire at
117.266 / 117.287 GiB; the 128K greedy model/recovery retire at
117.286 / 117.294 GiB. The native restoration screen retires at
117.290 GiB. The completed
whole/quality/repeat/interaction
comparisons finish rc0,
with no native/GPU/container model processes and `spark-job busy` free.

- Artifact manifest ID: `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
- Literal input SHA256: `0cbafc4bafd7a2c1e1c83f4a346815cdd500d2fb0bcb45afcfd826d591ae8e9a`.
- Captured heads SHA256: `6e1823b781b60f3999daff6341f0421d8d3ab3804e51993b95e1dec33452c0c4`.
- Captured raw Q8 SHA256: `aab88a1dd63fe965d6148f1cdefee9bbe8d01316eeeab63169f73acb91f50c51`.
- Original quality artifact ID: `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`.
- Quality prompt/force SHA256: `0961e248af647f3e9c838d43ba2cbad821fc48a7e9f066679ae7b2e8f118e6cb` /
  `e70be6122f5df8d83357b8223aab63d0f826cdd1dac85760e759cd574471480a`.
- Saved same-checkpoint oracle SHA256: `49de2d51973b4bef32cf174c861d44d74bf63b6ec8ee576641dcba8bcc042aa6`.
- Held-out PPL input/loss SHA256: `47f997b83854b52d0eb2c3ae605b505d87cd778f573fde9664d93a0e205bf2c6` /
  `b6d39c8da65f78e874092d8043e11fc3a4ff1ee9af06afbd2757b6bfacf82e2a`.
- Raw local evidence: `/home/pmeenan/scratch/m3-ds4-qhead-short-records/outa-operator-r3/`
  and `outa-whole-r1/`, `outa-quality-r1/`, `outa-quality-repeat-r1/`;
  the interaction and its repeat are in `outa-hca-r1/` and
  `outa-hca-repeat-r1/`, with the registered PPL condition in `outa-ppl-r1/`
  and the R3 model/recovery in `outa-r3-r1/` and `outa-r3-recover-r1/`;
  the 128K greedy model/scoring recovery are in `outa-128k-r1/` and
  `outa-128k-score-r1/`;
  the native restoration is in `outa-native-r2/`, with its pre-model
  compile failure retained in `outa-native-r1/`;
  sources/results on Spark are under
  `/home/pmeenan/scratch/m3-ds4-qhead-short-r1/`.

The first attempt stopped before model loading on routine host warnings.
The second captured window-layer operands but refused before timing because
the diagnostic wrapper requires positive beta metadata even with YaRN off.
An explicitly recorded 32/1 sentinel substitution applies only when
extension and both original betas are zero; the original table kernel
does not read those betas in that arm. Native descriptors remain unchanged.
Both failed supervisors retired; their records remain alongside the screen.
No new unit suite was run for these private diagnostic screens. A focused
32K fixed-history comparison uses the original UD-Q2_K_XL checkpoint and
its existing b11254 oracle; the community speed result above remains a
separate checkpoint/shape measurement.
