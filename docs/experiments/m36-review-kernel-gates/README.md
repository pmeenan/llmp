<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Common-head attention: D192 kernel screen

On 2026-10-17, group-one MMA at head width 192 passed the fixed correctness
and performance screen on `spark` (GB10). Its median candidate/bookend ratio
was 0.156830 uncaptured and 0.159607 captured, below the predeclared 0.90
adoption bound. Keep MMA for this new prefill path, with vector at a single
query where physical KV padding permits it. Existing served D64, D128 image,
D256 and D512 choices are unchanged. This is a focused kernel result, not a
whole-model performance or new model-admission claim.

## Fixed inputs and protocol

Q/K/V head width 192; 128 queries; 16 query heads and 2 KV heads (GQA ratio 8); 2,048
physical KV cells; one sample; a 1,024-cell ring window per query. Q and
output are F32; K, V and additive mask are F16. Seed 714, normal inputs with
standard deviation 0.25, scale 1. The baseline is corrected vector attention
at head width 192; the candidate is group-one MMA with query columns 32.
Both include paid mask and
stream-K/fixup work and use the same descriptors and workspace backing.

Each execution mode warms both arms 8 times (including 8 graph replays per
arm in captured mode), then runs 3 V/M/V rounds with 128 launches per stage.
CUDA events measure the submission stream. Candidate time is divided by the
mean of its two vector bookends; the median of 3 round ratios decides each
mode. Adoption requires both medians <= 0.90 and finite outputs with relative
RMS <= 0.003. Shape, inputs, order, repetition count and thresholds were fixed
before any timings. Pre-timing source corrections and the warmup wording
clarification are preserved in the exact [preregistration](preregistration.txt),
SHA-256 `f27b0fd2d4bc546f026abcfc6d9f82fdaf9c818ada6df616bcf989e61ca4dc5c`.

The run completed successfully under an exclusive installed `spark-job`.
All 18 stage times appear below, in microseconds per launch.

| Mode | Round | Vector before | MMA | Vector after | MMA / bookend mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncaptured | 0 | 938.352 | 147.702 | 933.548 | 0.157810 |
| Uncaptured | 1 | 932.097 | 146.921 | 941.538 | 0.156830 |
| Uncaptured | 2 | 934.745 | 145.036 | 936.083 | 0.155050 |
| Captured | 0 | 940.130 | 150.033 | 938.701 | 0.159709 |
| Captured | 1 | 938.049 | 147.309 | 946.594 | 0.156326 |
| Captured | 2 | 938.660 | 150.033 | 941.365 | 0.159607 |

Uncaptured median ratio 0.156830, range 0.155050–0.157810; captured median
0.159607, range 0.156326–0.159709. Candidate/vector relative RMS was
0.00079151713 both uncaptured and captured; vector replay against itself
was 0. Both captured graphs had 2 nodes. No timing expansion was needed.

## Resources and memory

Actual device facts: NVIDIA GB10, compute capability 12.1, 48 SMs,
driver 580.178.04, CUDA driver API 13.0, toolkit 13.4.92. CUDA reported the
candidate at 128 threads, 25,600 bytes dynamic shared memory, 0 static shared
memory, 222 registers per thread and 2 resident blocks per SM; KV batch 32.
These are queried facts, rather than the configuration table's occupancy
hint or a resource estimate.

The harness allocates 16 MiB of device workspace for both arms. Its explicit
input allocations total 5 MiB (Q 1.5 MiB, K 1.5 MiB, V 1.5 MiB, mask 0.5 MiB),
and output is 1.5 MiB: 22.5 MiB of explicit device payload/workspace allocations.
CUDA context, library/graph bookkeeping and host buffers are outside this
count; it is not a process or model peak-memory measurement.

The printed `scratch3178496` is the maximum planned scratch requirement of
the two arms. The printed `peak2408448` is the selected MMA call's actual
scratch-suballocator highwater after `ResetScratchPeak`, before capture.
Neither value changes the 16 MiB workspace allocation or measures physical
memory returned to the device. They describe required capacity and observed
usage inside the already allocated workspace, respectively.

## Correctness, source and limits

The locked GGML revision remains `d81235049384534c167caea52b85a694f6103d14`.
Patch 0008 supplies equal-width 192 MMA configuration rows, narrowly enables
the explicit group-one 192/128 and 192/192 tiles through a compile-pruning
guard, and bounds the vector final loop's tail at widths that are not
multiples of 128. Without the vector bound, a D192 specialization writes 64
extra values after its head. Existing served power-of-two heads and the
existing group 8 and group 16 MMA variants keep
their valid-element arithmetic. See [RE-060](../../rough-edges.md) and the
[GGML upstream handoff](../../upstream/ggml.md#mma-head-192-configuration-re-060).

Before the screen, the Spark targeted batch passed 253/253 tests, 0 skips,
including 126 head/GQA/KV/query combinations against an FP64 CPU formula with
output guards. Those cover equal head widths 64/80/96/112/128/192,
Q/K width 192 with V width 128, GQA ratios 1/4/8, query rows 1/9/35,
257 physical KV cells, and 256 physical KV cells with 253 logically valid
cells. Vector attention at head width 192 still needs physical KV padding;
its logical valid length may be odd and masked. No `sm_86` GPU execution was
performed.

The subsequent legacy-D64 wrapper correction only enforces the instantiated
head width before launch; it changes neither the measured device kernels nor
this benchmark's correctly dispatched common path. Its refusal regression
and the final shared/model checks are tracked in the R5 handoff, separate
from this screen evidence.

Provenance for the measured snapshot:

| Item | SHA-256 |
| --- | --- |
| Source manifest, 2,571 files | `886d04283aa3c1ce3a945e18db6b925011a1ed3c3c22f95feda18bcd15ef4708` |
| Patch 0008 | `db938628b2da71ed5deb4d3f32493eef6681f3a710cb6513c48e49464210cd21` |
| Prepared GGML tree, 321 files | `620c8c8bdcd84ba26cebca77093a0ebb3fb6ae315adea6ad32ffa6f266494d28` |
| Benchmark source | `fd35e66f7afabf97eadb43ea9390a9564c0195513cf83597fca3dc224f9e3224` |
| Measured benchmark binary | `7ae19522198f4c66ea7ce58c5d50ae0d299f83dff5863a9be3cf6ed6bfdf6877` |
| Correctness batch log | `4470fc8a0a53df500a6aabe8b15a6028e1232b494e87044c59f0c5615dd854f4` |
| Exclusive screen log | `76825b5f704ad4b130d0cb09e88371f59b3e42a9d35c3fe29d0459436988ce54` |

Raw logs remain outside the repository in the campaign evidence directory.
The prepared tree was first derived with the source patch parser and complete
base manifest, then independently verified by source preparation and a native
Spark build. No upstream revision refresh or arbitrary measured tuning pin
was introduced.
