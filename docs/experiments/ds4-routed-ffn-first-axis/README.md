<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ds4 routed FFN: paid Direct versus Materialized

Keep the fused Direct path. The existing fully paid Materialized path is
**2.35% slower** in the complete 8K pipeline, while all twelve full heads
remain byte identical to original ds4. This bounded difference does not
justify a new gather adapter. Materialized is a useful checked control for
later native consumer substitutions; this experiment changes no production
selection and does not approve cache precision or general answer quality.

The [paid-stage profile](../ds4-restoration-profile/README.md) assigned
35.55% of GPU time to routed FFN. That share was a prioritization signal,
not an expected fusion gain. This first factor compares **producer/gather +
fusion/storage**, using original aligned IQ2 gate/up and Q2 down weights,
selected-six IDs and weights, weighted/clamped activation, and downstream
stages. Direct consumes token-major D4 and fuses activation into D2S6.
Materialized pays its gathered/requantized pair-D4 producer and F32 gate,
up and activation storage before the original D2S6/down consumers. It is
not a pure fusion experiment or a current-GGML consumer comparison.

## Complete-pipeline measurement

Spark A (`spark-c4e2`), 2026-10-01 04:19:15–04:26:42 America/New_York.
One prepared model; each pass starts from freshly initialized state.
The diagnostic Direct warmup is followed by three ordinary Direct passes,
a Materialized warmup and three ordinary Materialized passes, then a
Direct return warmup and three ordinary Direct passes. Diagnostic captures
and all warmups are excluded from the performance comparison. The nine
ordinary passes create no profile marks or original-input probes.

Fixed community artifact `cd39d504…`, 8,192 supplied IDs, capacity 8,192,
two 4,096-row chunks, all 43 layers, original default FP8 KV/FP4 indexer
policy, no drafter, and one final 129,280-entry F32 head per pass. Full
request wall plus final result-copy time is the comparable scope.

| Arm | Samples, seconds | Mean seconds | Tokens/s |
| --- | --- | ---: | ---: |
| Direct before | 7.549938, 7.541421, 7.554482 | | |
| Direct after | 7.601446, 7.610772, 7.619987 | | |
| Direct combined | Six samples above | 7.579674 | 1080.785 |
| Materialized | 7.741268, 7.781157, 7.763279 | 7.761901 | 1055.412 |

Materialized/Direct throughput is **0.976523×**; wall time increases
2.404%. The six Direct samples' maximum/minimum is 1.010418 and the
Materialized samples' is 1.005153. Direct after/before mean is 1.008229.
All twelve saved heads, including warmups, match original SHA-256
`499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8`.
No greedy margin, full-head error or softmax-distribution difference exists
in this control. Wider contexts and independent quality gates remain open.

## Same-original-input controls

The diagnostic warmup captures chunk 1, layer 21, preserving the completed
Direct outputs before a disjoint Materialized probe on the same original
normalized input. All 23 operand files remain external. Expert bounds,
selected-ID correspondence, source/destination bijections and token order
are valid; maps have identical order and both arms use 533 active down-work
items. Comparisons canonicalize by destination slot rather than assuming
that helper order is interchangeable.

All **786,432** gathered D4 blocks match their corresponding token D4
blocks byte for byte. All **393,216** canonical D2S6 blocks match. Every
one of the **100,663,296** down-output F32 values matches: zero differing
values, maximum error zero and NMSE zero. Guards are zero and F32/stored
scales are finite. Twelve fixed physical Q2/D2S6 FP64 dots per arm and
24 IQ2/D4 gate/up FP64 dots retain the raw stored-scale/sign/code semantics.
Maximum absolute errors are 1.040e-7 for down and 5.329e-7 for gate/up;
the analyzer imposes no registered FP64 acceptance bound. These witnesses
are diagnostic, not quality exceptions. Gather/quantization did not introduce a numerical difference
for these actual operands, so a new data-only gather would not explain a
quality discrepancy here.

## Memory and qualification

Preparation reads 84,513,722,368 bytes and takes 290.279 seconds. All arms
symmetrically pay 603,979,776 bytes of F32 intermediates, 572,761,096 bytes
of separate device probe storage and a 134,217,728-byte host capture buffer.
No concurrent raw/aligned full expert replica is added.

The whole-process memory flag remains **false** against the previous
unprofiled bookend process: sampled RSS is 1,040,982,016 bytes, 1.153594×
its 902,381,568-byte reference; the sampled node MemAvailable drop is
104,193,019,904 bytes, 1.014289× its 102,725,160,960-byte reference. This
comparison includes symmetric diagnostic staging and cache; it does not
isolate either consumer's memory. No memory gate is waived.

The frozen source passed 1,202/1,202 locked native tests, including 256 GPU
tests, nine SDK format checks, seven actual host tidy compile arms,
boundaries, and SDK REUSE/header checks for 1,107 files. The pinned source
proof preserves 115 original numerical functions and ten prepared headers.
The native child completed and was reaped; source, binary, actual build
receipt, SDK and resolved cuBLAS payloads stayed unchanged. Final strong
retirement found 117.154 GiB available and no model/GPU job. The preceding
launch refused a wrong baseline-receipt path before model loading; its
failed record is retained and excluded from this measurement.

Raw records live at `spark:~/scratch/m3-ds4-ffn-axis-r1/{check-r1,model-r2}`
and the corresponding workstation scratch copies. Source base is
`e8e9c4fa7b982eecff5350cdfbac18a08fd1fa72`, SDK
`aarch64-e0a0c85c42806fb1`, original ds4
`76d51ef82a81b70b78e51a3a6ea11946286de976`.

| Record | SHA-256 |
| --- | --- |
| Final qualification | `90d52cf77f0f5285c17a58be7dfd9c8888b4e4711ce5552b539efe9aa13e7e40` |
| Frozen source maps | `f2047cb6533a432907e48d8051814c1d964055355970c9ea43204e189c909c16` |
| Native binary | `48aaa8aef67ba5881c61ec3c87859776bd06e04d8cac76a62527732094f783a3` |
| Model controller receipt | `b850e9231c22eb8152d12f3a35e89d166cd9e971d5ef52792756d32fddbd74cf` |
| Native receipt | `9fa7edaa3766cd8ea406a01c3f340397ed7937329b95b746a5715942e79da166` |
| Operand controls | `a902686f0e327b1cbb8e445a2618b7112f3f8705052fd95a244892fe1dbe5fd2` |
| Model controller | `cb1401c81056bcb88d7ad701442fe7e5ea6c207967f838f2f9f040c0522eb479` |
| Operand analyzer | `ee929998ce07d33a948e0b28d773d0a408a28e56b31b12e332169f38cae62d08` |

Next, establish the separate complete original 32K comparison with active
score/select. Native compact IQ2 consumer restoration is a later factor
requiring explicit weight-layout and activation-producer contracts.
