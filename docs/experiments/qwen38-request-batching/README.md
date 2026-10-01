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
