<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current Gemma31 prefill timeline

This matched trace separates visible extra kernel work from GPU-idle intervals.
Native quantized products and attention have nearly the same summed kernel time
as the original engine. Native has 32 internal GPU-idle gaps over 10 ms, totaling
592 ms, and pays 32 vocabulary projections against one in the original.
The extra projections total 145 ms. Neither observation alone explains the
whole gap or establishes a production change.

| Paid 8K interval | Native | Original |
| --- | ---: | ---: |
| NVTX wall span, s | 12.074129 | 10.897581 |
| GPU activity union, s | 11.246289 | 10.839234 |
| Kernel union, s | 11.212976 | 10.834484 |
| Copy union, ms | 33.313 | 4.750 |
| Wall minus GPU activity, ms | 827.840 | 58.347 |
| Executed kernels | 64,704 | 60,548 |
| Internal gaps over 10 ms | 32 | 0 |
| Largest internal gap, ms | 24.444 | 3.674 |

| Observed kernel-family duration sum, ms | Native | Original |
| --- | ---: | ---: |
| Quantized products and fixups, excluding vocabulary projection | 8,229.140 | 8,211.251 |
| Quantized input preparation | 466.018 | 454.261 |
| Attention and fixups | 1,115.386 | 1,116.883 |
| Vocabulary projection | 150.176 (32 kernels) | 4.684 (1 kernel) |
| RMS-norm families, including fused chains | 524.855 | 468.283 |
| Elementwise MUL kernels | 197.719 | 29.025 |
| GeGLU kernels | 511.112 | 493.167 |

These are clipped duration sums grouped by actual symbols, not disjoint wall
costs. Native has exactly 3,872 additional standalone MUL kernels (121 × 32):
the existing plain norm/Mul flag is off, while stock's 1024-thread RMS kernel
has its multiply template enabled. This source-supported dispatch difference
is visible, but the [plain norm/Mul screen](../gemma-normmul-screen/README.md)
found no resolved gain: its two candidate prefill runs differed by 275 ms,
and their mean was 0.82% slower than one off control. The RMS-family timing
sum also includes other fused chains and different row shapes. The vocabulary
projection uses Q5_K MMVQ, grid 262144 × 1 × 1 and block 32 × 4 × 1; source
output width and actual launch geometry support that label.
Both engines execute the same observed attention templates and grids: 1,600
D256 launches at grid 96/block 32 × 4, and 320 D512 launches at grid 48/block 32 × 8,
plus corresponding fixups. This is C1 prefill evidence, not C4 decode transfer.

One native and one original application ran on Spark A under the installed
GPU supervisor with a 600-second timeout. The existing Task42 child-wait
launcher, owned Docker
retirement and PID-authenticated union extractor were reused. Only the two
observer clients' closed 31B recipe support and the extractor's optional range
label changed; the existing 26B range remains its default. No production source,
kernel, dispatch or cache-policy change was made. Native source base is
`cf71dad`, with the adopted source/read indices, both norm chains, plain norm
fusion off, default graphs and 32 paid prefill heads. Original pinned-image
math is unchanged, with physical ubatch 256, F16 KV, explicit false SWA/unified
and actual local 1,280/global 16,384 capacities. Both use the canonical 8,227 IDs,
six discarded warm rows, clear, 8K prefill, three anchors and 32 forced units.

Both traces reproduce their own complete prefill/final heads and all 32 choices
from the [existing untraced baseline](../gemma-use-index/README.md). Native also reproduces all 1,740,636,160
initialized state bytes and layout. Heads are finite. The extractor authenticates
actual model PID ownership, excludes profiler-parent waits and reports zero
unknown process events. App-child, profiler and supervisor retirement succeeded;
the original owned container was checked absent. A first thin-client build
failed on the retained headers' nested include paths; correcting only external
`-I` paths completed the build, with no model rerun or changed math.

The paid graphs also differ at the final block. The pinned original context
initializes `embeddings_nextn_masked=false`; this client does not change it.
Its Gemma graph gathers requested rows after full-row output normalization.
Native's default frontier policy gathers projected/residual rows before the
last FFN. The trace confirms native's plain RMS count as 3,808 grid 256 plus 64
grid 1 kernels, versus original's 3,872 grid 256 kernels; norm/residual kernels
show the corresponding row split. Head projection itself is one row in both.
These shapes are source-supported, not equal operator workloads. They may
contribute to the known prefill-head difference; no same-input shape control
has proved that cause. Last-block FFN output does not write earlier-layer KV.

These traced elapsed times do not replace competitive bookends. GPU-idle gaps
can include planning, state growth, waits or submission delays; they are not
CPU-busy measurements. The separate [cold/retained diagnostic](../gemma-retained-plan/README.md)
measures 526 ms cold planning and 164 ms state growth, but the two experiments
cannot assign every gap causally. Extra head projection cost does not measure
all discarded last-block work: frontier narrowing also changes FFN shape.
Two heads and 32 choices do not qualify all decode vectors or corpus quality.

[Results](results.json) retain aggregates, source and original-library identities.
The host was `spark` (`spark-c4e2`, GB10), with driver 580.178.04 from the
contemporaneous retained-plan platform query. The prepared artifact identity
was `32c92e…`; original image `837fc…` retained the exact `b29c606…` pin.
The result file records their complete identities and the observer binaries.
Formatting after measurement changed only whitespace in the C++ observers;
the result file retains both measured and committed source identities.
Raw SQLite, traces, logs, state, vectors and child/official records remain in
external gemma31-current-timeline storage. Reproduce using the existing manual
`jitllm_gemma26_prefill_profile` target with `31 both 256`, the CAPI
observer's optional `31` argument, and the Task42 Nsight 2025.3.2 CUDA/NVTX/OSRT flags; pass the explicit
`jitllm.gemma31.paid_prefill` label to `trace_v2.extract`. Native SDK math remains
CUDA 13.4 and original-image math CUDA 13.3; only observer clients were compiled.
Task-entry TensorFold HEAD remained 609ca419 (0.6.5, Gemma 26B MLX recipe, no 31B CUDA
recipe); it supplies no comparable CUDA result. No additional model ladder,
reference qualification or kernel adoption follows from this trace.
