<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Controlled Qwen request-slot cap (2026-10-04)

**Retain four slots for this measured profile.** Six slots give −1.20% /
+0.57% completed-token throughput against the two four-slot bookends, while
median request latency rises 47.63% / 48.06%. The earlier
[request-slot knee](../request-slots/README.md) remains the fallback. An
isolated settings-layer proof accepts the controlled value `max_slots: 4`
under the runtime's actual artifact/device/build/settings key. No production
state, setting, default or runtime code changes in this experiment.

This closes a controlled Qwen slot-cap measurement and its calibration
layering proof for the profile below. It does not calibrate DeepSeek, other
Qwen artifacts, long prompts, other prefill settings or other machines.
The broader D-103 controlled runs and kernel schedule tables remain open.

## Fixed work and timing scope

One unchanged native binary on Spark B serves cap **4 → 6 → 4**, each arm
with fresh owned state and one service process. Each arm first pays an
excluded one-output C1 prime on its normal cap, then two bursts of six
simultaneous requests. Every measured request pays 512 outputs. All arms use
identical complete prompt sets: the first set has six 183-token prompts;
the second has six 184-token prompts, with `spek-` tags replacing `spec-`.
The prompts retain three service records followed by a printing-press essay
question. Exact input bodies and hashes are frozen in `inputs.json`, derived
from the preserved earlier `slots/cells.py` and four `u*.request.json` files.
The old harness varied the request count with the cap; this comparison always
pays six requests, including at cap four.

The profile is target `c4fb47a9…`, selected MTP `8600a998…`, speculation,
context 33,792, explicit prefill 4,096, shared draft cap 2, `draft_wave_max=2`,
lanes enabled and wave read alignment 2,048. Requested draft vocabulary
65,536 resolves to the physical selected **47,172-row** BF16 head and its
matching I32 ID map. Both cap-four controls omit `max_slots` and resolve its
fallback of four; the candidate explicitly overrides it to six. Fresh state
starts without calibration. Passive records written during an arm take
effect only at a later service start, so they cannot change this comparison
mid-service. The excluded primes take 6.61 / 6.29 / 6.32 seconds; they are
excluded from every burst metric below.

Burst wall time runs from the earliest request submission to the last
complete streamed response, including `[DONE]` and response EOF. Throughput
is 3,072 actual completion tokens divided by that wall time. Request latency
runs from that request's submission to its complete response. The client
also retains first-visible reasoning/content text times; these are transport
observations, **not model TTFT**, and are not used as decode timings.
Connection setup precedes a barrier; measured submission spreads are
0.44–1.81 ms. Responses retain both reasoning and content channels.

The second burst runs in the same service with plans encountered in the
first burst available. It uses a different frozen prompt set. It does not
prove that all plans are warm, isolate planning cost or establish why a
second burst is faster. All 36 measured requests return HTTP 200, explicit
`[DONE]`, `finish_reason=length`, 512 completion tokens, correct total usage
and zero cached prompt tokens. Actual prompt counts match by member across
all three arms. The three one-output primes also complete correctly.

## Paid comparison

| Arm | Burst | Wall s | Completed tok/s | Median request s | Worst request s |
| --- | --- | ---: | ---: | ---: | ---: |
| Four before | First | 51.6141 | 59.5186 | 34.3820 | 51.6137 |
| Six | First | 51.7460 | 59.3670 | 49.9784 | 51.7460 |
| Four after | First | 50.6319 | 60.6732 | 33.3257 | 50.6319 |
| Four before | Second | 49.3828 | 62.2079 | 32.4530 | 49.3813 |
| Six | Second | 49.2028 | 62.4354 | 48.0665 | 49.2023 |
| Four after | Second | 49.5877 | 61.9509 | 32.4748 | 49.5868 |

Candidate rate gain is `(mean control wall / candidate wall − 1) × 100`:
**−1.2039% first, +0.5740% second**. Control rate movement from before to
after is `(before wall / after wall − 1) × 100`: **+1.9399% / −0.4131%**.
The small throughput difference does not justify changing the knee.
Candidate medians exceed the mean control medians by **47.6297% / 48.0616%**;
worst request latency is approximately unchanged. Cap six gets every request
to visible text earlier than the two queued requests at cap four, while
all six responses complete later than the first four at cap four. That transport tradeoff does not outweigh the
measured median latency cost for this workload.

Per-request wall seconds below are in frozen input-member order, which does
not imply arrival or admission order. Admission timing varies between runs.

| Arm/burst | Member 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four before / first | 30.396 | 51.614 | 50.316 | 35.171 | 33.593 | 31.707 |
| Six / first | 49.066 | 51.567 | 47.442 | 50.890 | 51.746 | 46.458 |
| Four after / first | 49.451 | 32.660 | 28.986 | 33.991 | 50.632 | 31.037 |
| Four before / second | 32.453 | 28.810 | 49.381 | 48.326 | 32.453 | 29.680 |
| Six / second | 48.592 | 44.669 | 49.202 | 48.068 | 46.988 | 48.066 |
| Four after / second | 32.818 | 27.987 | 49.587 | 49.141 | 30.958 | 32.131 |

## Plans, allocation and retirement

| Arm | Held plans | Plan MiB | Planning s | Graphs | Counted graph MiB | Measured at captures MiB | Capture s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Four before | 123 | 293.1 | 2.738 | 74 | 893.4 | 562.8 | 0.330 |
| Six | 146 | 470.5 | 5.159 | 76 | 1,268.0 | 812.5 | 0.405 |
| Four after | 119 | 321.2 | 2.976 | 73 | 720.7 | 378.1 | 0.280 |

These are final service-log totals across the prime and both bursts, not
per-burst costs, proof of a causal slowdown or peak memory. The startup
allocation logs authenticate actual capacity/branch count four or six. All
three startup budget guards pass, and no memory or conversation-capacity
wait is logged. Registered state virtual extents are 5,762,973,696 bytes at
four and 8,644,460,544 at six; these are virtual extents, not resident state.
Scratch stays 1,107,296,256 bytes in both. No peak-memory or cross-cap
logit/token/initialized-state equality claim is made. This settings screen
does not reopen quality bounds or prove broader quality parity.

The installed GPU supervisor's `qwen-slot-knee-short-screen` job completes
with rc 0 and is waited. Each service is terminated normally, returns rc 0
and is reaped without KILL; all nine client processes and all memory samplers
finish and are joined. Six strong 105-GiB preflights validate admission and
retirement, including after the final arm:

| Arm | Admission free GiB | Retirement free GiB |
| --- | ---: | ---: |
| Four before | 116.912 | 117.091 |
| Six | 117.100 | 117.123 |
| Four after | 117.109 | 116.963 |

## Controlled record, isolated from production

The model-free `qwen-slot-knee-calibration-proof` installed GPU job also
returns rc 0 and is waited. It preserves every key field from the actual
first control's runtime-generated passive record, replacing only `values`
with `{"max_slots":4}`. Passive calibration itself did not measure a slot
cap: this field is the controlled selection from the comparison above.
Existing passive floor/depth-ratio values are omitted so they cannot affect
the layering proof. The separate owned proof state uses 0700 directories
and a 0600 file, written with a private temporary file, fsync, atomic rename
and directory fsync. Nothing is written to production state.

Five actual `llmp-runtime settings --json` calls return:

| Owned proof configuration | Slot value/source | Record |
| --- | --- | --- |
| Same measured profile | 4 / calibrated | In force |
| Explicit `max_slots=4` | 4 / override | Stale |
| Explicit `max_slots=6` | 6 / override | Stale |
| `draft_vocab=16384`, no slot override | 4 / fallback | Stale |
| Restore measured profile | 4 / calibrated | In force |

All calls complete and are reaped, without starting model execution. Both
source and controlled record hashes stay unchanged across these read-only
calls. An explicit fallback-valued override invalidates the key because it
could bypass a different calibrated value (D-103); this is expected. The
key includes the explicit prefill override, so this proof does not confer
eligibility on an otherwise similar profile that omits that override. A new
build, artifact, machine or relevant policy requires its own controlled
measurement/key. DeepSeek's controlled slot calibration remains owed.

## Provenance and reproduction

The measured binary is the checked `995cced` implementation in Spark B's
read-only `~/src/llmp-wt/calkey01`; no build is performed for this
settings-only experiment. Device identity is NVIDIA GB10 sm_121, driver
580.178.04 and the runtime-reported CUDA 13.0 driver/device API; the checked
toolkit compiler is 13.4.92. The 487-file compiled source inventory, binary,
receipt, client, inputs, tokenizer/template, artifact indexes and lifetime
helper are authenticated before and after every arm and the proof. Artifact
payloads are not rehashed here. The pinned preflight also checks GPU,
container and native model processes. Inherited execution/numerical overrides
are refused. Settings, startup guards, raw SSE, response bodies, clocks,
usage, passive keys and owned cleanup receipts are retained.

| Item | SHA-256 |
| --- | --- |
| Runtime binary | `97a87140aef8a7452ec779cd2689561f4625235f587a67f9d9af846efde807f2` |
| 487-file compiled inventory | `c4332305fc048aa9b340c09751e460c2a1651e1cae188e61b2f57d534a368fe2` |
| Build receipt | `756a3bcd17b9e723b7bfa0192560d710ea982549fb71cc3a8f0b375588204377` |
| Screen controller | `0738b7fa446a883ed2171ce93e968cd57ca29795ca1c96296741f8f6378748d5` |
| Stream client | `8c0890bcca5a8aed5fdb97278ab6d01d72df1c04a7077294a14e901658ce4348` |
| Complete frozen inputs | `aebd2351ba12fb135a46f1b26e016cefa01c07ead425ce4cd441bbe6a0b7d1d6` |
| Preflight | `3d9f07a48373741a24ef4e9615e99c6bdacb7854ca5cc2356afb73914ddc3ec5` |
| Lifetime helper | `53b0cc57c2ae77f2d56c6984d0257c3c9c2ceafc86b3acf455d9f02573e342a1` |
| Target index | `5b65dcce8169374e638264d7ebdcb5cca517234b3dec26f1272a5f4f9c0e4c89` |
| Selected drafter index | `5cd45fc5354ab224d281c2416027f224c61e32e2ae0acf8e7580d063fc274d99` |
| Tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Template | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Screen report | `63872b7950555f09f344e564c212c646fe5c66375b95eb76c9aa83503e476270` |
| Calibration proof controller | `3e7fc293bddf937bb6d8d94f4c7a81ace812d58add4093b8527a8da257e82831` |
| Runtime-generated baseline record | `e979545624bc21df70022fd5ad7e525d4a3822869c5f7cd0a547b51e73730e49` |
| Owned controlled record | `86ff41d7c6aa4f24514e884dae0e45a0c4c0521de51317c1041db7935ed623cd` |
| Calibration proof report | `f77639fbdf370c275d2b97c063de36b2574640125b6e99231f5ec4e2ab9d7f7b` |

Raw evidence and pinned controllers are outside Git at Spark B
`~/scratch/qwen-slot-knee/{screen1,calibration-proof1}` and the workstation
`~/scratch/llmp-m3-qwen-slot-knee-2026-10-04/` (state payloads are not
copied to the workstation). Run the preserved `screen.py` under the installed
GPU supervisor with a fresh output directory and a 600-second bound, then
wait; it refuses reused arm directories. Run `calibration_proof.py` with a
fresh proof directory under the same supervisor, then wait. The proof pins
the original baseline record: a new screen's different passive values
require a new recorded source pin, not silent reuse. Full suites, additional
long cells, reference engines and a wider slot matrix are not run for this
unchanged-runtime decision.
