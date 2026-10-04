<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Controlled DeepSeek request-slot cap (2026-10-04)

**Retain four slots as the latency-focused knee for this measured profile.**
Six remains an explicit throughput/tail-latency option. The matched short
comparison finds a throughput/latency tradeoff:
cap six gains **16.36% / 21.79%** completed-token throughput against cap-four
bookends, with median request latency **25.74% / 22.88%** higher and worst
latency **14.06% / 17.89%** lower. One long follow-up gains 6.70% throughput
with 9.30% higher mean and 10.26% higher median latency. That tradeoff
supports retaining four for the primary latency-focused workload. No
production record or new default is written. A separate current-build
proof accepts a controlled `max_slots: 4` record in isolated owned state.
The performance measurements and their passive keys remain those of the
checked `995cced` runtime; they are not rekeyed to the later build.

## Matched six-request short work

One checked unchanged native binary on Spark B serves caps **4 → 6 → 4**,
with a fresh service, owned state and anchors per arm. The target is the
community `cd39d504…` artifact with DSpark `dd2d3f9c…`, context 262,144,
explicit prefill 4,096 and explicit production `wave_form="auto"`. The
cap-four controls omit `max_slots` and resolve fallback four; the candidate
overrides it to six. Draft rows resolve to three, with output-A/HCA and its
partial-chunk policy enabled. No external tokenizer/template is configured:
the artifact's embedded GGUF metadata is pinned before and after each arm.

Each service pays one excluded C1 prime of one output, on its normal cap.
Then it pays two bursts of **six requests at every cap**, each request with
512 outputs. The same twelve complete prompts are frozen across all arms:
three service records followed by a printing-press essay question. The first
set has `spec-` tags, the second `spek-` tags. Actual prompt counts are 124
for every member of the first set and 126 for every member of the second,
matching across all three arms. These are the frozen prompt strings from the
[Qwen controlled screen](../qwen38-slot-knee/README.md), rendered by the
DeepSeek tokenizer/template; their Qwen token counts do not apply here.

Fresh state starts without calibration. Automatic wave selection alternates
forms at newly encountered widths for its bounded exploration and updates
counted acceptance during the service. Passive wave-cost records take effect
on a later start, never mid-service. Thus the second burst includes the
current acceptance/exploration state and plans encountered in the first.
Neither burst isolates planning, proves all plans warm, runs only DSpark
waves or guarantees the same wave-form sequence between caps. This is the
actual production automatic policy, not a forced speculative-wave test.

All 36 measured requests return HTTP 200, explicit `[DONE]`, complete
response EOF, `finish_reason=length`, exactly 512 completion tokens, correct
total usage and zero cached prompt tokens. Both reasoning and content
channels, raw SSE, request bodies, response IDs and actual usage are retained.
The three excluded primes complete with nine prompt tokens and one output,
in 8.25 / 7.78 / 7.82 seconds.

The burst clock spans earliest request submission through the last complete
streamed response. Completed-token rate is 3,072 actual tokens divided by
that wall time. A request's latency spans its submission through its response
completion. First-visible reasoning/content text times are preserved as
transport observations, not model TTFT or decode timing. Connection setup
precedes a submission barrier. Input-member order is not admission order;
arrival timing and automatic forms may change replies. This experiment makes
no cross-cap token/logit/state equality or new quality claim.

| Arm | Burst | Wall s | Completed tok/s | Mean request s | Median request s | Worst request s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Four before | First | 74.2015 | 41.4008 | 56.6641 | 49.3126 | 74.2006 |
| Six | First | 62.6662 | 49.0217 | 60.9914 | 61.4223 | 62.6660 |
| Four after | First | 71.6407 | 42.8807 | 54.6622 | 48.3832 | 71.6407 |
| Four before | Second | 73.1764 | 41.9807 | 55.3097 | 47.8542 | 73.1763 |
| Six | Second | 59.4228 | 51.6973 | 58.2710 | 58.4970 | 59.4220 |
| Four after | Second | 71.5608 | 42.9285 | 54.2266 | 47.3554 | 71.5595 |

Rate gain is `(mean control wall / candidate wall − 1) × 100`:
**+16.3643% / +21.7860%**. Control rate movement, `(before wall / after
wall − 1) × 100`, is **+3.5745% / +2.2577%**. Mean request latency is the
arithmetic mean of all six request wall latencies, and rises **9.5723% / 6.3958%**. Mean, median and worst
latency changes compare the candidate with the mean of the two corresponding
control values. They are independent of the burst throughput metric.
The candidate's higher throughput is larger than the control movement, but
it also delays median completion; this is a policy tradeoff, not a neutral
screen like Qwen's.

| Arm/burst: per-request wall s in frozen input-member order | Member 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four before / first | 74.201 | 45.516 | 49.869 | 47.539 | 48.756 | 74.104 |
| Six / first | 58.318 | 62.660 | 59.460 | 61.943 | 60.901 | 62.666 |
| Four after / first | 44.506 | 70.256 | 71.641 | 48.567 | 44.804 | 48.199 |
| Four before / second | 73.176 | 48.663 | 45.923 | 47.046 | 73.061 | 43.990 |
| Six / second | 56.759 | 58.126 | 59.422 | 57.136 | 59.315 | 58.868 |
| Four after / second | 48.444 | 43.129 | 71.025 | 71.559 | 44.935 | 46.267 |

## Plans, budget and clean lifetime

| Arm | Held plans | Plan MiB | Planning s | Graphs | Counted graph MiB | Measured at captures MiB | Capture s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Four before | 48 | 169.7 | 2.060 | 26 | 1,046.9 | 941.9 | 0.393 |
| Six | 60 | 227.7 | 2.991 | 34 | 1,539.3 | 1,437.1 | 0.579 |
| Four after | 50 | 167.8 | 1.998 | 23 | 940.1 | 839.4 | 0.343 |

These are final log totals across the prime and two bursts, not per-burst
costs, isolated causes of timing differences or peak memory. Actual slot
reports and startup allocation guards are retained, all guards pass, and
no memory or conversation-state capacity wait is logged.

The installed GPU job `deepseek-slot-knee-short-screen2` returns rc 0 and
is waited, completing its three arms in 170.32 / 145.54 / 165.52 seconds.
All services return rc 0 and are reaped without forced KILL; all nine clients
are reaped and samplers joined without error. Six strong 105-GiB admission
and retirement gates pass, including after the final arm:

| Arm | Admission free GiB | Retirement free GiB |
| --- | ---: | ---: |
| Four before | 116.954 | 117.066 |
| Six | 117.054 | 117.163 |
| Four after | 117.157 | 117.161 |

An earlier `deepseek-slot-knee-short-screen` attempt returns rc 1 and is
waited: a copied Qwen-only `wave_lanes` settings-log assertion stopped it
before the prime or any measured request. That inapplicable assertion was
replaced by the actual DeepSeek `draft_rows=3` check, using a fresh output
directory. Its service returned rc 0, was reaped and passed retirement;
the failed controller and receipt are retained, not counted as a pass.
DeepSeek has no public `wave_lanes` setting; its fixed source's lane policy
is unchanged.

## One matched long-workload follow-up

The short tradeoff warranted one long cell, not a wider matrix. The same
binary/profile runs cap 4 → 6 → 4 with one six-request burst per fresh
service, 256 outputs per request and an excluded C1 prime on the normal cap.
Complete earlier `u0..u3` prompt bodies are frozen before all arms; members
four and five prepend `variant 4` / `variant 5` to the first two bodies.
Actual prompt counts are **[7,043, 7,043, 7,043, 7,043, 7,047, 7,047]**, matching
memberwise across arms. All 18 requests complete HTTP 200/DONE/EOF, length
256, correct total usage and zero cached tokens: 1,536 paid outputs per
burst. The automatic-policy and transport-clock limitations above apply.

| Arm | Wall s | Completed tok/s | Mean request s | Median request s | Worst request s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four before | 81.6038 | 18.8227 | 67.6002 | 67.6723 | 81.6038 |
| Six | 76.5603 | 20.0626 | 74.0579 | 74.7776 | 76.5603 |
| Four after | 81.7727 | 18.7838 | 67.9116 | 67.9664 | 81.7718 |

Against the mean control wall, cap six gains **6.6979%** throughput; control
rate movement is **−0.2066%**. Mean request latency rises **9.3011%**, median
**10.2600%**, while worst falls **6.2769%** against the corresponding mean
control latencies. The larger cap helps the tail and aggregate rate but
delays the early completions. This is a measured throughput/latency choice,
not evidence of an arithmetic improvement or a universal optimal cap.

| Per-request wall s in frozen input-member order | Member 0 | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four before | 57.469 | 49.806 | 81.378 | 81.604 | 68.200 | 67.144 |
| Six | 71.832 | 69.953 | 75.755 | 73.800 | 76.447 | 76.560 |
| Four after | 81.772 | 67.750 | 58.756 | 50.037 | 80.972 | 68.183 |

| Arm | Held plans | Plan MiB | Planning s | Graphs | Counted graph MiB | Measured at captures MiB | Capture s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Four before | 59 | 230.1 | 2.816 | 29 | 1,419.4 | 1,330.6 | 0.494 |
| Six | 71 | 296.0 | 4.185 | 39 | 2,076.6 | 1,916.0 | 0.721 |
| Four after | 55 | 221.1 | 2.731 | 31 | 1,603.3 | 1,487.7 | 0.578 |

These totals include the prime and long burst and do not measure peak memory
or isolate planning from other costs. The installed GPU job
`deepseek-slot-knee-long-screen` returns rc 0 and is waited, with arm times
104.03 / 100.27 / 104.27 seconds. All three services return rc 0 and are
reaped without KILL, all six clients are reaped, samplers join without error,
startup guards pass and no capacity or memory wait is logged. Six further
strong 105-GiB gates pass:

| Arm | Admission free GiB | Retirement free GiB |
| --- | ---: | ---: |
| Four before | 116.957 | 116.998 |
| Six | 117.006 | 117.164 |
| Four after | 117.154 | 117.175 |

## Current-build calibrated-source proof

The subsequent dependency-key fix (`5020bfc`) adds effective context and
DeepSeek's uncalibrated wave-cost vector plus the actual override-prefix
count to the key. It changes four runtime settings/calibration source files;
it does not change the model or kernel sources. Historical `995cced` slot
measurements and keys above remain unchanged. This separate proof tests
current record layering and eligibility, **not slot performance**.

The final checked runtime in the owned read-only `wavekey1` tree runs one
excluded C1 request in fresh state, on the same community+DSpark automatic
profile and normal fallback-four configuration. It uses the first frozen
short member: **124 uncached prompt tokens and 32 outputs**, HTTP 200,
DONE/EOF, length and correct total usage. This produces a real passive
record containing `decode_floor_tok_s: 15`. Its actual dependency key is:

```text
speculation=true draft_rows=3 prefill_chunk=4096 max_slots=4 prefill_outa_hca=true/true wave_form=auto prefill_chunk_override=true max_slots_override=false context=262144 wave_costs=[2.08,2.52,2.81,2.4,2.12,2.21,2.23] wave_costs_override=false wave_costs_override_count=0
```

The actual build identity is `0.1.0-dev+unknown (executable 195043464 bytes,
modified 1791116536895158287 ns)`. Artifact, drafter, device, build and
settings fields are preserved verbatim in a separate owned record whose
**only value is `max_slots: 4`**. A temporary file is created with mode
0600, flushed and fsynced, atomically renamed and its directory fsynced;
the proof directories are mode 0700. The original runtime-generated record
is retained and unchanged. Six model-free `settings --json` cells pass:

| Configuration | Slot value/source | Record eligibility |
| --- | --- | --- |
| Original profile | 4 / calibrated | In force |
| Explicit fallback-valued cap 4 | 4 / override | Stale |
| Explicit cap 6 | 6 / override | Stale |
| Context 131,072 | 4 / fallback | Stale |
| Explicit fallback-valued wave-cost prefix `[2.08]` | 4 / fallback | Stale |
| Original profile restored | 4 / calibrated | In force |

All six CLI processes return rc 0 and are reaped. The controlled and source
records remain byte-identical throughout the CLI proof, and all binary,
compiled-source, artifact, metadata, input and helper identities match
before and after. This does not test execution under the overridden cost
prefix or install a record in production state.

The installed GPU jobs `deepseek-slot-current-key-prime2` (300-second bound)
and `deepseek-slot-calibration-layering` (120-second bound) both return rc 0
and are waited. The prime's service and client return rc 0 and are reaped
without KILL; its sampler joins cleanly, startup guard passes and no memory
or capacity wait is logged. Two strong 105-GiB gates pass, with 116.886 GiB
before admission and 117.091 GiB after retirement. The excluded request
completes in 9.32 seconds and its whole prime arm in 21.43 seconds; these
are provenance, not a new slot timing comparison.

An earlier one-output prime completes nine uncached prompt tokens and one
output, with clean service/client/sampler retirement and both gates, but
writes no passive record. Its qualification job returns rc 1 and is waited,
with the failed receipt and original controllers retained. Passive medians
require eight samples, and its nine-token prefill is below the 1,024-row
chunk threshold. The replacement's 32 outputs require at least eight decode
steps with this profile's at-most-four kept tokens per step. The failed
attempt is not presented as a calibration pass.

| Current-proof item | SHA-256 |
| --- | --- |
| Checked runtime | `8e815e1fbe12b9c855e69b7c678f6f66dff6912f644510994ac0227418fec946` |
| 487-file compiled inventory | `cc51578b435739a78f2f6d94edc6cdac397726a32b74811e61c1760a5aa92e58` |
| Build receipt | `4e84c12e273df75fd4398b7f8148a609922963cb097b1c41f4056fddf47435d8` |
| Prime controller | `d6ce2f6693abb69daa21fd9da121ed32cf0991b3a60a4631049947b372962e32` |
| Prime client | `1de36e04c99a0033f5a7f9dc2888704fe85a2a11953f2a7443bdfe8d1d8e4c17` |
| Prime report | `ec418d7eb49c5258739d2b6fc035c140e5fd22b3739fa24998c72f320eb60ab2` |
| Prime receipt | `690db0725b32aedeecc1a5318d8d001b388b7e1e8d6cb9b6ed4d9e7932e23bc1` |
| Actual passive record | `f84ab10eae3aa4de4209a6d1e20e62e18a8369128f945d125fbae27041f9ab81` |
| Layering controller | `f3878edebaae02211c9b83aa862c434834d72491a663fc664a33fc1e41297f2d` |
| Controlled slot-only record | `1f7bf4cf427c5af9fc5daa1d4b9df0eb2cf838ba7c1e21c043b4f9401d027b1b` |
| Layering report | `db400bd1a5fe3bc30297a72f0fd26c2a0c9847b135dcff2ea187bdc4dc529f86` |

Proof raw data remains outside Git at Spark B
`~/scratch/deepseek-slot-calibration/{prime1,prime2,calibration-proof1}`
and the workstation
`~/scratch/jitllm-m3-deepseek-slot-calibration-2026-10-04/`, excluding state
payloads. Broader artifacts, contexts and slot knees remain open.

## Performance provenance and limits

The performance arms use the checked `995cced` implementation in Spark B's read-only
`~/src/jitLLM-wt/calkey01`. No runtime build or code/default change is made
for those arms. Device identity is NVIDIA GB10 sm_121, driver
580.178.04 and runtime-reported CUDA 13.0 driver/device API; the checked
toolkit compiler is 13.4.92. The 487-file compiled source inventory, binary,
build receipt, controller dependencies, client and complete inputs are
pinned. Both artifact manifests, indexes and actual embedded GGUF metadata
are authenticated before and after every arm. Weight payloads are not
rehashed. The preflight and lifetime helper are the same pinned helpers as
the Qwen screen. Inherited execution/numerical overrides are refused.

| Item | SHA-256 |
| --- | --- |
| Runtime binary | `97a87140aef8a7452ec779cd2689561f4625235f587a67f9d9af846efde807f2` |
| 487-file compiled inventory | `c4332305fc048aa9b340c09751e460c2a1651e1cae188e61b2f57d534a368fe2` |
| Build receipt | `756a3bcd17b9e723b7bfa0192560d710ea982549fb71cc3a8f0b375588204377` |
| Qualified screen controller | `feec2bbd62826d91a3a0403d9304710e532c65dd4d6f45d196fced39ed72f527` |
| Stream client | `9d434611a6e3bcc0ebf36588d32de19b7ceb0ebcfab6814c31861dc75bb09e28` |
| Complete frozen inputs | `aebd2351ba12fb135a46f1b26e016cefa01c07ead425ce4cd441bbe6a0b7d1d6` |
| Community manifest identity | `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac` |
| Community index | `9b63f352eb11cda5299d0880c121f79f5956f1b7b5e0f83b79d633d8e3d9903a` |
| Community embedded metadata | `6e18e82310f8a2157f60604a038f64968691783956a8336ba4203e7cb25f95a8` |
| DSpark manifest identity | `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5` |
| DSpark index | `437561ad6dd1e8d5fd3758be2a8eac52b8a1c81f6a1fb54b27e0b3badfeaa11c` |
| DSpark embedded metadata | `fcf6886653e8a64b70d51de8896febe2d69ae3dba8e300ed4684b518bbfcddff` |
| Long screen controller | `ac708240958214348bc34960cb3f4017dd0b8092495851360062281a480513dd` |
| Long stream client | `82fb6be2dab71402d981b06fa286a6e94333f0c4a5836ce4a0cdbcfbd4368c6f` |
| Long complete frozen inputs | `4a0136c3ba93005f34418e46a1b99da916dbf55970e0cfa9295e437f50483154` |
| Long report | `ba881611bc949265f9450e61f414cf1e37795c441aaeed6f016f2ba4f59c33c6` |
| Short report | `cca010df31af4917ba47eefa59c60f5a3df75801096b3924e8291e2801c1070f` |

The historical 9.3% warm short gain in the
[request-slots report](../request-slots/README.md) used the original
`8a355bfb…` artifact and different request counts at caps four and six.
It is not a matched control for this community-artifact factor. The two
new burst timings remain separate because their actual prompts differ and
the automatic policy evolves. No steady DSpark decode, cross-engine
quality, calibrated kernel schedule or peak-memory claim follows here.

Raw receipts, strict streamed responses, passive keys, pins and supervisor
logs are outside Git at Spark B `~/scratch/deepseek-slot-knee/{screen1,screen2,long1}`
and the workstation `~/scratch/jitllm-m3-deepseek-slot-knee-2026-10-04/`;
state payloads are not copied to the workstation. The preserved `screen.py`
requires fresh arm directories, an installed GPU-supervised job with a
600-second bound and a successful wait. Its internal 530-second work
budget reserves retirement room and refuses unbounded continuation.
No controlled calibration record is written by the performance screen. Full suites,
reference engines and a broad slot matrix are not run for this
unchanged-runtime measurement.
