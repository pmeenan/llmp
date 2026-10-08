<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Captured native IQ2 gate/up restoration

Keep original Direct for this comparison. The current native compact IQ2
pair plus the paid original suffix delivers 13.41% less resident operator
throughput than Direct, and 7.99% less than Materialized. Its complete
finite outputs differ slightly. This is one captured layer, with no model
quality, whole-pipeline, concurrent-request or production-default result.

The subsequent [preparation capture](../ds4-native-iq2-prepared/README.md)
proves all native/original D4 payload bytes equal. Borrowing either payload
reproduces native gate/up exactly, isolating the product differences to the
consumer path. Removing preparation changes its resident time by about 4–6%.
Standalone original/native consumer timing remains open; this compound
replay does not isolate the raw loader, tile or schedule.

## Matched inputs and charged work

Spark B replays the authenticated original community-model inputs at
chunk 1, layer 21: 4,096 F32 rows of width 4,096, middle width 2,048,
output width 4,096, 256 experts and six selected experts per row. All
24,576 assignments, route weights and original expert-major map bytes are
validated. This input uses 249 experts and 533 active original down-work
items. Active gate/up weights total 1,077,018,624 bytes, versus the device's
25,165,824-byte L2.

The gate/up conversion only inverts the original aligned IQ2 layout into
66-byte raw records; every two-byte scale and 64-byte code payload is
preserved. Full round trips verify both 553,648,128-byte tensors. Validation
takes 4.432 s, and derivation plus hashing takes 3.258 / 3.237 s for gate/up.
These cold data-preparation costs are outside resident operator timers.
Raw and aligned gate/up are never resident as two device copies in a child.

| Arm | Paid resident chain |
| --- | --- |
| Suffix calibration | Uploaded original gate/up/maps, original weighted/clamped activation, D2S6/worklist, aligned Q2 down and sum; excludes gate/up production |
| Original Direct | Original maps/token D4, aligned IQ2 G1 with fused weighted activation/D2S6, original down and sum |
| Original Materialized | Original maps/gathered D4, aligned IQ2 gate/up, weighted/clamped F32 middle, original D2S6/down/sum |
| Current native pair plus original suffix | Current compact `MulMatIdQPair`, including native maps and activation quantization, raw IQ2 gate/up, a second paid original-map adapter, and the original Materialized back half |

The native arm combines activation preparation, mapping, raw weight layout,
MMQ geometry and compiler choices. It is not a pure arithmetic or layout
axis. Both native products share their existing preparation. The additional
original-map adapter is charged because native transient maps are not
exposed by the current operation.

All complete arms allocate exactly 3,659,965,448 CUDA bytes, including a
symmetric 512-MiB workspace, 603,979,776 bytes for gate/up/middle, gathered
D4 and guard, D2S6 and guard, maps/work/down/sum. Direct retains the same
unused intermediate allocations. Complete arms upload 1,879,244,800 bytes;
upload wall time is about 31–32 ms and is outside operator timing. Native
planned and observed scratch both equal 113,466,624 bytes. These are device
allocation/upload inventories, not a whole-process comparative memory gate.

## Resident timing and controls

Six fresh serial children run suffix, Direct, Materialized, native,
Materialized, Direct. Each has one warmup and nine retained positive GPU
event and wall samples. Copies, full comparisons, hashing and output files
are outside the operator timer. Every child completes and is reaped before
its strong retirement gate and the next child.

| Arm | First median ms | Final median ms |
| --- | ---: | ---: |
| Suffix calibration | 15.107328 | — |
| Original Direct | 32.017632 | 32.078079 |
| Original Materialized | 34.099648 | 34.006622 |
| Current native pair plus original suffix | 37.010559 | — |

Direct and Materialized bookend max/min ratios are 1.001888 and 1.002736.
Using the mean of each pair of original medians, native resident speed is
0.865911× Direct and 0.920092× Materialized. Suffix timing is a calibration,
not a complete-FFN comparator; subtracting it does not establish a separately
measured product cost.

All six children have exact independent repeats and finite complete outputs.
Original Direct uses its own captured D2S6/down controls; preparation also
proves full Direct/Materialized map/D2S6/down equality. Materialized and
suffix match every applicable captured gate/up/middle/D2S6/down/map byte.
The native adapter's complete maps match. All fresh original/suffix outputs,
including their sums, agree. There is no captured sum file, so this fresh
agreement is kept separate from captured-byte claims.

## Complete numerical differences

The analyzer reads every original/native F32 output, not only selected
coordinates. The denominator of NMSE is the complete original squared
energy. No registered model-quality bound is applied here.

| Native output versus Materialized | F32 entries | Maximum absolute error | NMSE |
| --- | ---: | ---: | ---: |
| Gate | 50,331,648 | 1.192093e−6 | 1.088888e−14 |
| Up | 50,331,648 | 1.430511e−6 | 1.361259e−14 |
| Weighted/clamped middle | 50,331,648 | 2.861023e−6 | 2.551202e−14 |
| Down | 100,663,296 | 0.001704775 | 9.249194e−9 |
| Summed output | 16,777,216 | 0.001704752 | 8.120071e−9 |

3,484 of 393,216 complete D2S6 blocks differ; the full guard remains zero.
The very small gate/up differences precede changed D2S6 rounding and larger
downstream differences. This observation does not isolate a native
quantization or reduction cause.

Twenty-four fixed FP64 physical IQ2 dots cover twelve coordinates in both
products. Against the original F32 activation, maximum absolute combined
quantization/reduction error is 0.009624933 for both paths. Original
consumer error against captured original D4 peaks at 5.329133e−7. Native
private D4 was not captured, so that second reference supplies no native
consumer-only bound. Both engines' authenticated IQ2 grid tables agree.
No quality exception or whole-model qualification is inferred.

## Provenance and limits

Measured on Spark B, 2026-10-01, 08:40 EDT. The six-child supervisor exits
0; every child and retirement succeeds, with 116.432 GiB available at the
terminal operator gate. Post-retirement CPU analysis finishes at 08:48 EDT.
Raw captures, derived weights, all 48 full output files, samples, logs,
controllers and build proof remain outside Git under
`spark-b:~/scratch/m3-ds4-native-iq2-r1/`.

| Identity | SHA-256 |
| --- | --- |
| Community artifact | `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac` |
| Original capture metadata | `6b26061eac3b5970cddaafad3f6fd2cb1fc3091e06be51be9df9624bb14c395e` |
| Preparation receipt | `358cc5e27f27b92e4b8291abc9269c6c4a9f4cd555f01fa4aa458e802c580721` |
| Target-only build receipt | `bc0f8574fac89726db2064ff181fe2bc33e1329f5d79168f444f47e894094e0a` |
| Unchanged execution-source map, 1,245 files | `fb239981b8b1ce7010dec08f7a870ef4bee6c9831833af98884eae4a682d964c` |
| Stateless operator binary | `fc90aca73e2aa740f557d2aaa3648e6051f654c495c4109a6d377298583ca6a8` |
| Six-child operator receipt | `a594e138000f7256fe6ad91659af7a384716e520b4d3e70cdf792467fe1e191a` |
| Complete output/FP64 analysis | `eadfb0c369c210a8be299ebf65a5724ee895355946499c45337f612becbd6691` |
| Source/build preservation receipt | `eb4c3f36d3ef9f11cabb950f5b3e32ce70a26f30b89449f262f823368c388420` |

The actual SDK is `aarch64-e0a0c85c42806fb1`, with resolved SDK cuBLAS and
cuBLASLt 13.8.0.4. The build receipt retains full source maps, static-link
closure, compiler/header identities and actual commands. The original
115-function/ten-header correspondence proof passes. Literal MoE units
retain last `-O3 --use_fast_math -lineinfo`; the current native dispatch
unit uses `-O2 -use_fast_math -extended-lambda`, and the prepared IQ2 kernel
instance uses `-O3 -use_fast_math -extended-lambda`, all with `sm_121a`
SASS. No compiler flags are changed for this comparison.

The target-only build follows the owner's experimental-cadence override:
no new routine unit/full-suite/style cycle is claimed. Initial incomplete
export, reused proof-directory and incorrect dispatch-flag expectation
failures remain in the raw record; they are excluded from measured results.
The previously matched [complete 32K pipeline](../ds4-matched-32k/README.md)
and [Direct/Materialized full-model factor](../ds4-routed-ffn-first-axis/README.md)
remain separate evidence. Production dispatch is unchanged.
