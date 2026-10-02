<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Qwen shared-row C2/C4 controls

Pairing selected row-local products improves aggregate paid decode throughput
by 19.1% at two requests and 19.6% at four requests, with exact independent
request outputs, acceptance, initialized state and cursors. Both results
come from private resident generation loops. HTTP continuous batching and
the concurrent Mia/TensorFold gate remain open.

The benchmark target is `jitllm_qwen38_batch`. It takes prepared target and
selected-drafter directories, exact little-endian I32 prompt files and a
fresh output directory. Both prompts have 8192 IDs in `--mode natural`; four
such prompts use `--mode natural4`. Argument order is fixed:

```text
jitllm_qwen38_batch --artifact TARGET --drafter SELECTED --input0 U0.i32 --input1 U1.i32 --out NEWDIR --mode natural
jitllm_qwen38_batch --artifact TARGET --drafter SELECTED --input0 U0.i32 --input1 U1.i32 --input2 U2.i32 --input3 U3.i32 --out NEWDIR --mode natural4
```

The prepared target is c4fb47a9 and selected drafter 8600a998, with 47172
ascending original IDs. The external fixed fixtures are members of
`m3-concurrent-r1/inputs8k-r2`, receipt SHA-256
`a947338b1d0fb09822bb37d19747711926e1e4d37cbfacc9e5cd75cc689bb791`.
They contain exact literal token IDs, without chat rendering. The native
benchmark validates 8192 IDs, vocabulary range, fresh output, and a checked
100 GiB execution envelope. Run it only on an allocated Spark under the
supervisor and strong memory/process gates; raw captures stay outside Git.
The original eager/captured/replayed complete-row controls preceded natural
ID-only timing and remain preserved with their own source/binary receipts.

## Private C2 shared-row mechanism

The first natural two-request comparison improved aggregate paid decode
throughput by **19.1%** with identical output IDs, acceptance traces,
initialized target/MTP state and pending cursors. It establishes a useful
mechanism before service integration; it does not measure HTTP concurrency or
meet the Mia/TensorFold comparison gate.

One native process owned one resident target/selected-drafter weight closure
and two independent request states. Both frozen literal prompts contained
8,192 IDs. Fixed-depth-three generation produced 256 IDs per request, including the
prefill argmax; the measured loop produced the remaining 255 each, or 510
outputs. Stop IDs were literal. Only native MXFP8 vector and routed vector
products were grouped, with paid activation/ID concatenation and split views,
at no more than eight rows. Attention, recurrence, heads, per-request
acceptance and rollback retained their original arithmetic and selected
implementations. Normal input/PLE preparation, staging, output-ID copies,
plan construction, graph capture/instantiate and final settlement were paid.
Prefill, baseline reset and complete state diagnostics were outside this
decode clock.

| ABBA arm | Complete loop, s | Aggregate paid outputs/s | Draft, s | Verify, s | Settle, s | Request0 completion, s | Request1 completion, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial0 | 13.026414 | 39.1512 | 1.373862 | 11.421414 | 0.228370 | 12.657832 | 13.026412 |
| Shared1 | 10.827008 | 47.1044 | 1.355712 | 9.237420 | 0.227293 | 10.463899 | 10.827006 |
| Shared2 | 11.039892 | 46.1961 | 1.396862 | 9.406206 | 0.229397 | 10.673171 | 11.039889 |
| Serial3 | 13.019811 | 39.1711 | 1.366135 | 11.413979 | 0.232315 | 12.651852 | 13.019809 |

This is one resident ABBA experiment with two observations per arm and two
adjacent serial/shared bookends. The ratio of the two arms' mean durations is
1.191126×. Verification accounts for approximately 85–88% of complete loop
time and nearly all the gain; draft and settlement durations remain close.
Completion is the host's observation after each request's settled wave, not a
kernel timestamp or TTFT. The corresponding per-request paid decode rates
are 20.146/19.576 and 20.155/19.586 outputs/s in serial, versus 24.370/23.552
and 23.892/23.098 in shared execution.

All four runs produced the same 512 IDs and 102 natural waves: 97 with both
requests active, then five with only request 1 active. Request 0 accepted 158
of 290 offered positions over 97 verifies; request 1 accepted 153 of 305 over 102
verifies. Their depth-position accepted counts were 70/53/35 and 73/47/33.
Finished slots performed no dummy work. Every run recorded 30 eager,
22 captured and 152 replayed phase invocations, with no refused capture or
coverage violation. Each shared run coalesced 37,248 MXFP8 launch pairs and
9,894 routed launch pairs and paid 4,662,824,880 bytes of concatenated inputs.
Splits were views; these counters do not claim an extra unpack copy or one
weight read per expert.

Three tiny untimed active/tail shapes compared serial/full, serial/IDs,
shared/full and shared/IDs outputs and settled state. Together with all four
full trajectories, the proof retained 24 tiny output checks, 32 complete state
checks, eight independent canonical state captures and 24 exact ordered-page
comparisons including cursors. The initialized paired baseline was
1,360,216,064 bytes; diagnostic copies paid 24,483,889,152 bytes and complete
state reads 21,763,457,024 bytes. State capture and comparison took 16.252801
and 3.518893 seconds separately. Total native diagnostic wall time was
91.922687 seconds. No process-memory, stop-policy, task-quality, production
default or C4 result is inferred from this control.

Provenance: Spark B, SDK 4432b0ab, immutable fe1d7c8-based 1252-file source
map bc934637, target-only build b7c6af6a, executable eac1581a and the actual
SDK cuBLAS payloads. Native 0956e9fd and supervised f6c9f51b receipts completed
rc 0; the strong terminal gate reported 116.621 GiB with no GPU/model process.
Preservation ebf83851 authenticates the exact source/executable/libraries,
DB/cache/native receipt and all 80 raw files, retained outside Git under
`~/scratch/m3-c2-natural-r1/`. Earlier eager and captured/replayed full-row
proofs remain separately preserved. Routine unit/style suites were deferred
under the owner's experimental-diagnostic override; no production change is
qualified by this report.

## Private C4 shared-row mechanism

The same native grouping mechanism improved four-request aggregate paid
decode throughput by **19.6%**, with identical 1,024 output IDs, complete
acceptance traces, initialized target/MTP state and pending cursors. This
extends the C2 mechanism result to four independent states. It measures a
private generation loop, rather than HTTP service performance or parity with
Mia/TensorFold.

One process shared resident target/selected-drafter weights and owned four
request states. The four frozen literal prompts each contained 8,192 IDs;
fixed-depth-three generation returned 256 IDs per request, including the
prefill argmax. The clock paid the remaining 255 per request, or **1,020
outputs**, with stop IDs treated literally. Matching native MXFP8 vector and
routed vector products were concatenated in ascending active-slot pairs,
each at no more than eight rows. Attention, recurrence, heads, acceptance,
rollback, kernel arithmetic and implementation choices were unchanged.
Input/PLE preparation, packing, staging, output-ID copies, plan construction,
graph capture/instantiate and final settlement were paid. Prefill, baseline
reset and complete state diagnostics were outside the decode clock.

| ABBA arm | Complete loop, s | Aggregate paid outputs/s | Draft, s | Verify, s | Settle, s | Request completion observations, s (slots 0/1/2/3) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Serial0 | 24.990011 | 40.8163 | 2.950347 | 21.583532 | 0.448239 | 24.990008 / 23.975291 / 24.990008 / 24.236730 |
| Paired1 | 20.898650 | 48.8070 | 2.967631 | 17.465072 | 0.453927 | 20.898647 / 19.945114 / 20.898647 / 20.202250 |
| Paired2 | 21.018569 | 48.5285 | 2.993341 | 17.561414 | 0.450890 | 21.018566 / 20.068081 / 21.018566 / 20.326895 |
| Serial3 | 25.132924 | 40.5842 | 3.001676 | 21.666015 | 0.451578 | 25.132921 / 24.115098 / 25.132921 / 24.376439 |

One resident ABBA experiment gives two observations per arm and two adjacent
serial/paired bookends. Mean durations are 25.061468 versus 20.958610 s:
**1.195760× throughput**, or 16.4% less time. Verification takes approximately
84–86% of the whole loop and decreases by 19.0%; draft and settlement costs
remain close. Completion values are host observations after each slot's
settled wave, not TTFT or individual kernel timestamps. C4 paired rates of
48.5–48.8 outputs/s are only modestly above the separately measured C2
paired rates of 46.2–47.1; these different cohorts do not isolate a kernel
scaling factor.

All four arms follow the same 101 natural waves: 95 with all four slots,
one with slots 0/2/3, then five with slots 0/2. Finished slots receive no
dummy work. Slots 0/1/2/3 accept 154/160/154/159 of 301/284/300/284 offered
positions over 101/95/101/96 verifies. Accepted depth-position counts are
74/46/34, 75/51/34, 73/46/35 and 75/51/33. Every arm records 84 eager,
11 captured and 107 replayed phase invocations with no refusal or coverage
violation. Each paired arm coalesces 75,264 MXFP8 pairs and 19,992 routed
launch pairs and pays 9,392,856,368 bytes of concatenated inputs. Split
outputs are views; the counters imply no extra unpack copy or guaranteed
expert-weight reuse.

Four tiny untimed shapes compare serial/full, serial/IDs, paired/full and
paired/IDs outputs and state. All 15 active masks have bounded workspace
probes, and every actual plan is checked before queueing. The conservative
shared workspace bound is 701,865,984 bytes within an 878,706,688-byte mapped
activation region; the native pool is 4,194,304 bytes. The proof retains 64
tiny output checks, 80 complete state checks, 20 independent canonical
captures and 60 exact ordered-page comparisons, including cursors. The
initialized four-slot baseline is 2,720,432,128 bytes. Diagnostic copies pay
59,849,506,816 bytes and complete state reads 54,408,642,560 bytes. State
capture and comparison take 41.135513 and 10.471710 s separately; native
diagnostic wall time is 198.595637 s. No process-memory, stop-policy,
task-quality, production default or service rate follows from this control.

Provenance: Spark B, SDK 4432b0ab, immutable fe1d7c8-based 1252-file source
map 716a93e1, target-only build 97d5528c, executable bdd6f4f2 and actual SDK
cuBLAS payloads. Native 3eafc65f and supervised 0374fa78 receipts complete rc 0;
the strong terminal gate reports 116.924 GiB with GPU/model probes clear.
Preservation 602f7411 authenticates the exact source/executable/libraries,
DB/cache/native/SDK receipts and all 210 raw files, retained outside Git
under `~/scratch/m3-c4-natural-r1/`, and reauthenticates immutable completed
C2 evidence. Its separate post-preservation gate reports 116.952 GiB.
Routine suites and style checks were deferred for the measured experiment
under the owner's diagnostic override. Landing hygiene is recorded
separately; it does not replace the measured source and binary identities.

## Serving preparation

The API driver's optional cooperative interface owns at most four stable request
frames for one model. Admission runs between completed units, with independent
deadlines and cancellation; model changes drain the active group. Every outcome
passes through retirement that proves borrowed references have ended. Fake
controls exercise admission fairness, capacity deferral, disconnects, deadlines,
stalls, shutdown and shared response-budget accounting.

The shared `Llm::GenerationSession` prepares and applies completed plain or
speculative steps using the scalar acceptance and history logic. It borrows
stable options and stateful callbacks from its caller. A model-owned
`Llm::Branch` contains conversation history, sampling scratch, turn checkpoints
and saved cursors; the legacy methods forward to this same default branch.
Controls cover processed-anchor cancellation, callback state, visible stops
after complete verification, absolute-position seeds and failed-step validity.

This prepares service integration. The production node backend still executes
serial requests; shared execution and a bounded exact prefix cache remain to
be connected and measured through the real API.

## Independent native request slots

Qwen's runner now owns four stable native slots under one shared model owner.
Each has independent target/MTP state, verify snapshot, pending cursor and
plan caches. Runtime branches forward history, sampling, checkpoints and
adaptive policy to their own slots. Only selected slots enter the execution
closure, while swaps and retention account for every initialized slot.
This adds state ownership; no shared product dispatch or HTTP concurrency
is enabled by it.

A real-model control on Spark A uses public slot operations with context
2048, chunks of 512 rows and four distinct prefixes of 255/257/511/513 IDs. Its
depth-two/three and forced-kept-row trajectories match scalar controls
exactly: 64 complete output comparisons and 82 complete initialized-page
and cursor comparisons pass. Four clean host refusals preserve state. A
separately leased destination blocks Clear, quarantines only that slot and
is rebuilt while its peers remain exact. Five extra-slot-only retention
controls pass. Host capture/clear/restore,
independently leased peers, graph replay and an away/return swap are included.
There are 21 target captures/21 replays, 19 draft captures/21 replays, and
the aggregate graph count remains capped at 16. All 37069 evicted extents
return, preserving 835354624 state bytes and all four final state identities.
Diagnostic capture and comparison pay 3341418496 and 17124769792 bytes
outside any performance claim. The native control took 43.871 seconds.

The first control attempt passed its output and page checks, then inspected
an empty-target handoff before asynchronous parked-backing release finished.
The retry changes only that away swap to completed eviction before asserting
released residency. Production source, arithmetic and return-swap behavior
remain unchanged; both attempts are retained separately.

The chosen production slice passes 1236 Spark-native tests, including 256 GPU
tests, SDK formatting and clang-tidy, the boundary check and REUSE/header
checks. Its 1260-file source map is 5553813b…, checked receipt 86b8b9db…,
SDK f38891fc… and runtime binary 124cd0ea…. The external control binds that
same source and libraries: build bba5e9ae…, binary 23e249d5…, complete outer
control 6b8d8b1c… and native 208e0dca…. All children were reaped in the
supervised group and the terminal gate reported 117.225 GiB with no active
model. Raw receipts and controls stay outside Git under
`~/scratch/m3-qwen-native-slots-r1/` on Spark A. These controls establish
slot independence and retention at the tested shapes, not task quality,
performance parity, fault recovery or deployed concurrency. x86 checks
remain deferred under the owner's optimization-run override.

## First native Slot-wave C2 screen

The public native runner now has a diagnostic shared-wave implementation
over its independent slots. One scalar-to-wave direction screen on Spark B
completed the same two 8192-ID requests at fixed depth three: 510 paid
outputs took 12.399845467 s scalar and 11.281745075 s shared, or
**41.129545→45.205772 outputs/s (+9.9107%)**. One observation per arm in
fixed order warrants integration and focused follow-up; it does not establish
repeat stability, a production default or HTTP throughput.

The selected 47172 head, context 33792, chunks of 4096 and graph policy are
identical. Fresh prefills and small full-row controls are outside the decode
clock. Active selection, PLE reads/gather, packing, planning/capture, output
reads, acceptance and final settlement are paid. Finished slots leave later
waves without dummy work. All 512 IDs, all 199 per-slot draft/verdict traces,
pending cursors and initialized-range geometry match. Six complete-row
comparisons are byte-identical and all twelve retained vectors are finite;
this screen does not compare complete state-page contents.

Each arm performs six untimed generation steps before re-prefilling for the
clocked loop: three full-row controls and three IDs-only steps also warm the
actual copy-key plans. New shapes can still plan/capture during timing, but
this is not a first-use graph measurement. The later HTTP one-token prime
warms singleton masks only and does not reproduce this two-slot warmup.

Whole Draft is 1.544364433/1.631446159 s, Verify 10.505578754/9.415343014 s,
and host commit/final settlement 0.011866362/0.009252485 s. The shared arm
executes 37,248 MXFP8 and 9,888 routed pairs, paying 4,662,501,600 packing
bytes. PLE lookups stay at 12,704; union planning reduces reads from
12,696 rows/53,055,488 B to 10,560 rows/44,126,208 B, while the measured PLE
scope grows from 0.141478699 to 0.631575164 s. That scope covers planning,
direct-read completion and offset bookkeeping, excluding hash construction
and GPU gather. It identifies no I/O or host cause. There is no persistent
row-payload cache. BF16 full-head sharing remains a separate factor.

Actual outer receipt is
`890a9956b9f8fc49496f5ba1c7ed51e01d4c22f9f86ecf11c811789b16462d68`,
native `3c13e628…`, build `e2a1dca6…` and binary `e719926f…`, with actual
B SDK `4432b0ab…` and unchanged resolved cuBLAS payloads. The 65 s supervised
job completed zero and the native child was reaped; independent retirement
recorded 117.145 GiB free with model probes clear. Compile-only failed
attempts remain excluded. Records are retained outside Git at
`/home/pmeenan/scratch/m3-qwen-native-wave-screen-control-r1-records/`.
The diagnostic overlays are not a deployed concurrency implementation.
Focused state/recovery controls and serving integration follow this useful
signal; routine production checks remain owed before adoption.

## First cooperative HTTP screen

The private service integration preserves replies and recovery, but its first
matched fresh C2 screen is **neutral (+0.34% aggregate throughput)**. It is held
from production adoption. The earlier native decode gain does not establish a
gain through the HTTP scheduler.

Both service profiles provision two wave slots at context 262144 and chunks
of 4096, use the selected 47172 head and retain adaptive depth. Only the
cooperative backend is enabled in the shared profile. The two untouched
`spec-c4-u2/u3` chat fixtures each render to 8266 prompt tokens; all four replies
contain 256 generated tokens, zero cached tokens and a length finish. Text,
reasoning, usage and finish agree exactly between profiles. One fresh pair per
profile pays prefill, any first activation, queueing and HTTP work; this is not
a pure decode or repeat-stability measurement.

| Profile | Pair wall, s | Aggregate outputs/s | u2 completion, s | u3 completion, s |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 25.887234 | 19.778088 | 25.887234 | 16.090539 |
| Cooperative | 25.798419 | 19.846177 | 25.614849 | 25.798257 |

The previously first-finishing request takes substantially longer under shared
execution. Both shared requests queue for less than 0.002 s; the scalar u2
request queues for 16.090 s. No draft-depth or paired-product counters were
retained, so these observations identify no cause. Source inspection finds
that unequal adaptive depths disable paired draft products; the short fixed-depth
successor below tests that pairing eligibility without a wider matrix.

Separate focused controls restore the rejected request to its complete
post-draft state, preserve its accepted peer and produce exact continuation
outputs. The HTTP behavior screen preserves 16 normal responses, exercises
capacity deferral/refill and records one intentional disconnect as 499. The
262144-context startup guard passes without reducing chunks or context; it
does not establish populated maximum-context memory or task quality.

Provenance: Spark B, SDK 4432b0ab, checked private target receipt d98004a1,
scalar/shared executables 26d54826/6e18e013 and unchanged resolved SDK cuBLAS.
Actual performance outer receipt is 9c06c3d0, client receipts a85ec3e7/ecd4eca7;
both service children complete zero and are reaped, with terminal probes clear
at 116.726 GiB. Behavior/startup/recovery receipts are 17e52735/02fc48d8/995d00cd.
Raw controls remain outside Git in the respective
`/home/pmeenan/scratch/m3-qwen-cooperative-screen-*-r1-records/` and
`/home/pmeenan/scratch/m3-qwen-discard-verify-control-r1-records/` directories.
Routine suites are deferred while diagnosing this neutral private candidate;
no production default or cross-engine parity follows from it.

## Short HTTP successors

Three single-pair successors preserve the same selected head, chunks, context
and fresh 8266-prompt/zero-cache/256-output fixtures. Both arms use greedy
depth three. Every scalar/shared response retains identical content,
reasoning, usage and length finish.

| Screen | Scalar pair, s | Shared pair, s | Aggregate rate change |
| --- | ---: | ---: | ---: |
| Fixed depth three | 26.211413 | 26.284096 | −0.2765% |
| Skip unchanged slot-selection closure rebuild | 26.255620 | 25.511645 | +2.9162% |
| Same targets, warm weights before fresh pair | 20.240570 | 19.728319 | +2.5965% |

Fixed depth restores pairing: 107 of 109 completed draft/verify calls are
paired. It does not recover the native ready-C2 gain. The next private
one-file shortcut retains original validation and mutation-driven closure
refreshes, skipping the repeated rebuild only for an unchanged slot mask
with an open request. Its screen has 106 paired calls and four single-slot
calls; changed bootstrap/tail grouping prevents attributing the before/after
wall difference solely to the shortcut.

The last screen uses those unchanged binaries and pays an identical unrelated
one-token request before each measured pair. Prime requests have 57 prompt
tokens, zero cached tokens and identical replies, taking 6.303806/6.301559 s;
initial page-in is outside the pair clock. Measured prompts still have zero
cached tokens. Fresh prefill, queueing, graph preparation and HTTP remain
charged, so this is not pure decode. The shared arm executes 107 paired and
two single-slot calls, including the prime. First completion worsens from
10.200529 to 19.627801 s.

These small single observations select no default adoption or wider ladder.
The next diagnostic records inclusive adapter/graph costs on the unchanged
warm-weight workload; BF16 target-head sharing remains a separate factor.
Fixed3/selection builds are 2444d4a4/4363f3aa; actual outer receipts are
73ec92f6/73656b3d/94fd760b. All services and helper commands complete zero
and are reaped, with clear terminal probes. Compact raw records remain
outside Git in the respective
`/home/pmeenan/scratch/m3-qwen-cooperative-{fixed3,selection,weight-prime}-*-r1-records/`
directories. Source remains private and production checks remain owed.

One unchanged warm-weight successor records inclusive adapter timings, with
no failures or new policy. Pair wall is 20.431959/19.718395 s (+3.6188%
aggregate rate, one observation). Ten prefill/anchor calls cost
7.497371/7.551036 s; scalar generation costs 13.002896 s over 216 calls,
shared generation 12.278710 s over 109 calls. Nested draft costs are
1.666641/1.912625 s and verification 11.333401/10.364445 s. Selection costs
only 0.008984/0.009252 s. These scopes include prime adapter work, overlap
their nested calls and are host wall time, not exclusive GPU budgets.

Existing cumulative plan costs are 0.204643/0.679720 s and PLE costs
0.321670/0.649766 s, with identical 279104 lookups but fewer shared read bytes.
Scalar/shared graph counters are eager 33/61, captured 12/23, replayed
407/154 and refused zero. These overlapping diagnostics establish no causal
phase attribution. Fresh prefill dilutes the generation benefit, and shared
verification remains the largest adapter scope; target-head sharing is the
next separate factor. All response controls and six unique 200 terminals
pass. Target 3ae5f73e and outer ce09d422 bind this diagnostic; all commands
and service children exit zero/reaped, terminal memory 117.178 GiB with probes
clear. Compact records are
`/home/pmeenan/scratch/m3-qwen-cooperative-host-cost-http-r1-records/`.

## Shared target head through HTTP

Two cooperative profiles isolate BF16 full-target-head sharing OFF/ON. Both
retain the selected drafter, private fixed depth three, unchanged-selection
shortcut, graphs and identical provisioning. Compatible four-row heads
become one ordinary eight-column MMF product, with paid input concatenation
and separate slot logits and commit paths. Earlier receipt labels
`scalar`/`wave` mean head OFF/ON here; both profiles execute shared waves.

| Profile order | Head OFF pair, s | Head ON pair, s | Aggregate rate change |
| --- | ---: | ---: | ---: |
| OFF then ON | 19.843441 | 18.826454 | +5.4019% |
| ON then OFF | 19.803395 | 18.862484 | +4.9883% |

The pooled descriptive gain is 5.1949%, with two observations per arm.
Each service first receives the same unrelated one-token weight prime,
outside the pair clock. Every measured request has 8266 prompt tokens, zero
cached tokens and 256 outputs. All eight responses preserve identical
content, reasoning, usage and length finish. Fresh prefill, queue, graph
preparation, generation, settlement and HTTP are charged; this is not a
pure-decode or cross-engine comparison.

Before HTTP, separate native OFF/ON processes at context 33792 execute the
literal 8192-token fixtures through one paired draft and full verify. All
four complete F32 target vectors, each 993280 words, are byte-identical;
draft/verdict IDs, histories, keeps and pending cursors also match. Both
four-row and combined eight-row heads select ordinary MMF over the same
BF16 weight extent. One replacement pays 81920 input bytes; placement grows
from 7987456 to 8028160 bytes and scratch stays 2377728, within unchanged
bounds. These controls qualify their fixed histories and eligible shape.

The first HTTP pair has identical work counts: 109 waves, 107 paired and
two single-slot calls. ON replaces 104 heads and pays 8519680 head-pack
bytes. The reverse confirmation has 110 ON waves, 106 paired and four
single-slot calls, replacing 103 heads; OFF retains 109/107/two. Graph and
PLE counts also differ in that confirmation. Its positive direction is
whole-workload evidence, rather than identical per-step cost attribution.

This selects the compatible BF16 head path for production integration and
checks. Production adaptive depth remains unchanged; depth two, mixed,
ragged or unsupported heads retain their original products. These fixed3
observations do not establish the same gain under adaptive traffic, broader
quality, state-page equality or a production default.

Provenance: Spark B, SDK 4432b0ab, build db1c97e2, native control 5f4c3e57,
HTTP receipts 989a8261 and 5af60b38. The confirmation reuses the same binaries
and controls. All raw responses agree with nested records and parsed bodies;
both services and twelve commands per screen exit zero and are reaped.
Terminal probes are clear at 117.138/117.123 GiB. Raw captures remain outside
Git in `/home/pmeenan/scratch/m3-qwen-cooperative-head-factor-*-r1-records/`.
Production suites are owed before adoption.
