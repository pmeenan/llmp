<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen verify alpha/beta fusion — 2026-10-04

Qwen's fast speculative verify now computes its two Gated DeltaNet gate
planes in one pointwise kernel. Both preceding linear products, the
recurrence and accepted-row commit remain unchanged. The representative
C4 wave screen lowers verify median latency **1.11%**, and the matched
uncached 8K C4 HTTP screen gains **1.34% completed tokens/s**. Full target
rows, tokens and initialized target/drafter state remain exact in the
fixed-history controls, including a swap with rejected rows awaiting restore.
This is a scoped native optimization; the broader Mia performance and
same-reference quality gates remain open.

Spark B (`spark-56f5`), NVIDIA driver 580.178.04, CUDA 13.4.92, pinned SDK
`aarch64-e0a0c85c42806fb1`. All model controls use target NVFP4 artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`
and MTP artifact
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
The requested draft cap is 65,536; the installed selected head executes
47,172 rows. Prefill chunks remain 4,096; shared depth and the joined-draft
limit remain two, and four request slots use wave lanes.

## Arithmetic and scope

The original graph has two unchanged Linear products and four
pointwise kernels: `sigmoid(beta)`, `alpha + dt_bias`, softplus, then
multiplication by `ssm_a`. The fusion replaces only those four pointwise
launches, producing F32 gate and beta planes with ordinary views for the
existing recurrence and saved-row consumers. It writes no recurrent state.

Eligibility is fast-plan speculative verify with 128-wide recurrent heads,
48 value heads and at most 16 rows per request. Alpha/beta are packed F32
`[48, rows]`; bias and multiplier are F32 `[48]`. Unsupported shapes,
non-verify work and the exact plan retain the original primitives. Prefill,
including a small final chunk, is unchanged. The planner binds the new
operation before activation placement, and the two views retain their
producer through recurrence and save consumers.

The kernel uses the pinned GGML expressions:
`1 / (1 + expf(-beta))` and
`x > 20 ? x : logf(1 + expf(x))` after `x = alpha + dt_bias`.
Explicit rounded F32 addition and multiplication retain the original node
boundaries and prevent contraction. It uses GGML's existing CUDA fast-math
flags, including flush-to-zero behavior; no precision option changes.

The GPU operand control runs both original GGML chains and the fused
operation at rows 1, 2, 3, 4, 8 and 16. It compares all 3,264 paired F32
values bit for bit, both ordinary output views and their reshape/set_rows
save consumers. Random values, neighboring floats around softplus's
20 threshold, cancellation, infinities, signed zero, subnormal inputs and
subnormal multiplication results are included. Host refusal coverage checks
output overlap with each input, a misaligned output, a stale input view and
17-row descriptors before CUDA launch.

## Paid model controls

The short wave screen uses four frozen fast-swap prompts, 96 generated
tokens per slot and context 16,384. It is control/candidate/control, with
fresh processes. Phase clocks include preparation, kernel work,
capture/replay and output copying; process wall also includes loading,
prefill and host checks. Phase medians are separate measurements, not a
median or measurement of their sum.

| Arm | Verify median ms | Draft median ms | Process wall s |
| --- | ---: | ---: | ---: |
| control-before | 107.085 | 21.711 | 31.440 |
| candidate | 105.957 | 21.760 | 31.019 |
| control-after | 107.198 | 21.929 | 31.304 |

Verify median falls 1.1055% against the mean control median; control median
movement is +0.1055%. Each arm executes 43 verify waves, two captures and
38 replays. All four per-slot token hashes, full target-row hashes and
actual initialized target/drafter state hashes agree across all three arms.
A private exit witness establishes eligible use: 48 heads, first rows 3,
720 host dispatches in the candidate and none in either control. These
count eager/capture host dispatches, not graph-replayed launches. The
witness prints at exit outside the clocks; its host counter updates remain
included in prototype timings. Both are removed from production.

A separate fixed four-request 8K-history control, with the same 96 outputs
and context 16,384, matches the old executable for every token, full target
row and initialized target/drafter state hash. Both processes execute
42 verify waves, one capture and 37 replays. The candidate-only exit witness
records first rows 3 and 792 host dispatches. This is a correctness control;
its two arms do not supply a bookended performance estimate.

The optional FP16 switch occurs after wave 25 while all four slots still
owe rejected-row restoration. Twelve graphs survive the swap and 15 verify
replays occur after return. Generated tokens, every full target row and
initialized target/drafter state hashes match all three unswapped arms.
The unrelated FP16 model reproduces its pinned complete-head control before
Qwen returns; this checks restore/lifetime, not swap performance.

## Matched C4 HTTP screen

Four simultaneous buffered chat requests each pay 8,256 uncached input
tokens and return 256 output tokens with HTTP 200 and `length`. Context is
33,792. A one-token weight-prime request is excluded from the measured
burst. The frozen common-v2 client checks complete public responses;
clocks cover fresh prefill, generation, queueing and HTTP through completion.
Buffered responses do not provide first-token latency.

| Arm | Completed tok/s | Burst wall s | Peak MemAvailable drop GiB |
| --- | ---: | ---: | ---: |
| native-before | 32.268872 | 31.733369 | 81.529 |
| candidate | 32.641701 | 31.370914 | 81.156 |
| native-after | 32.152750 | 31.847976 | 81.264 |

Candidate rate is 1.3380% above the inverse mean control duration; control
wall movement is +0.3612%. Each arm completes all 1,024 output tokens.
The candidate-only exit witness records first rows 4 and 1,116 host
dispatches. All three services exit zero and are reaped, samplers terminate
cleanly and all six strong 105-GiB admission/retirement gates pass. Peak
MemAvailable drop is a whole-service sample, not the fusion's allocation
size or isolated GPU memory. The known arrival-dependent cohort arithmetic
remains; HTTP reply equality and cross-engine quality are not inferred.

## Provenance and replay

The control is the checked `8ebea5f` implementation; later base `9634363`
changes documentation only. The frozen prototype has the final arithmetic
plus an exit-only dispatch witness. Final cleanup removes that witness and
adds refusal coverage without changing device arithmetic. Each measured arm
authenticates its own compiled source inventory and executable before and
after execution. Source inventories cover 487 tracked source/build files;
the unchanged benchmark source is also checked explicitly.

| Item | SHA-256 |
| --- | --- |
| Control source inventory | `f0162fbb398b67698fd164b758ae5eddda90b0d2c7abe60956b11d2e35ca0f9d` |
| Measured prototype source inventory | `02fa5899c56628b4785abacf63da6f42f46d20a4462f13204c7e122a56ed5a8c` |
| Private measured patch | `206f4f9002cf8d366a2d57787e7d2aa7bb60d85f200fbce49daeb910a198adf5` |
| Control benchmark | `a071de68bb7abd6c32c77189a8ca937015f0cc9f91d981f96ba91e0e81b7d1c8` |
| Prototype benchmark | `86bcae0acdeddee96cc1cf2833c75c106eb39fb68d1ca045370c93eef4d65a13` |
| Control runtime | `bf7747a34130ad3e16f42b773fa311f60e6079a829a3daa3a6b3743fcdbb3170` |
| Prototype runtime | `a8ac660ad33ae6b230f3a23f78727073499bb00d55237835934ad8cec9ce0013` |
| Short wave controller | `44a6736884ff3f7f32ff98fb298f7ad6299dc19cba58f7196f0da751d2878272` |
| Long-history controller | `de502a6f7dcc1a502a26cf61de0e030f49cfe55408a6a5c8b26c59de8a6c6266` |
| Owed-restore controller | `a352932903459bdfdce44d992a561c7bc98f7e5259aa43519210c94b96ad6305` |
| HTTP controller | `8ed1f72d5084376133725030f3e567b194875c53b6ad0a60990f60e103f5982f` |
| Frozen HTTP client | `4f76adb8e36bb97e04c30f22d28d8d2ffc985d67aa4621f083ac92fc5e85d8f3` |
| Short wave aggregate | `37d2a196d6fc7fab2afa39e3ea2d218849149bc92c92cf6c7a614321ceb25b4d` |
| Long-history aggregate | `f4f7efae8685c9ae80acfe20f6e19f9d3a0ee2ae415d1e859df237182409c96e` |
| Owed-restore aggregate | `b6616540ff667c8399ba86611068388fb979fc09b239cd133b13a4863c1f00c5` |
| HTTP aggregate | `047601511119ef3751de19f2a939501ae758a888bf8fc795a2d51d2283b96a3e` |
| HTTP input receipt | `d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d` |
| Short prompt JSON | `d212009dadf1ddbf945c8dc7ad0214ba444236baf57c8ed9019c3ebe6b0805b4` |

The fixed long-context fixture is supplied from the earlier wave-lane
control, SHA-256
`5613e76b48e0d31b899099ae0d127f306afe36341b5bb523c177c80cbc5c6022`.
The FP16 switch fixture is artifact
`b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073`,
with token file SHA-256
`37e46a2304b5ab5648d0f2b7fdb94224596ed690caaddc1726df27110b585e21`
and complete logits SHA-256
`bb8ae5e7e3ac6da734173edb1111160a0c80a55c4279b94e67a8f90b142e7571`.
The [wave harness](../../../benchmarks/qwen38_spec.cc) accepts these as
`--check wave --slots 4 --wave-lanes on --tokens 96 --context 16384
--prefill-chunk 4096 --draft 3 --draft-vocab 65536`, with the target,
drafter, tokenizer and frozen prompts supplied explicitly. Optional
`--fp16-artifact`, `--fp16-tokens` and `--fp16-expect` enable the switch.
These controls do not qualify different formats, wider shapes or maximum
context, and no GGUF performance claim follows.

Raw controllers, patches, inventories, logs and responses remain external
under `~/scratch/gdn-gates/` on Spark B and
`/home/pmeenan/scratch/jitllm-m3-qwen-gdn-gates-2026-10-04/` locally.
All model jobs use the installed GPU supervisor with a 600-second limit and
are waited on. An initial HTTP controller failed before model admission
because its scratch filename shadowed Python's standard library; it is
retained and excluded. The corrected three-arm run completes.
