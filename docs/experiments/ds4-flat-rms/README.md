<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek flat-HC RMS block-size screen

Keep the native 1024-thread path. One complete 8K original/candidate/original
screen on Spark A (`spark-c4e2`, 2026-10-02) gives the 256-thread candidate
**−1.2990% throughput**, with ordinary bookend movement **−0.4207%**.

| Paid prefill, two 4096-row chunks | Seconds |
| --- | ---: |
| Original 1024 threads, before | 9.433430038 |
| Candidate 256 threads | 9.537475930 |
| Original 1024 threads, after | 9.393741985 |

Candidate throughput change is `(mean(original times) / candidate time − 1)
× 100`; bookend movement is `(original-after / original-before − 1) × 100`.
This is a whole-pipeline result, not an isolated RMS latency measurement.

## Isolated change and paid work

Only plain HC input RMS at GB10 cc1210, contiguous F32 `[16384,4096]`,
canonical strides and epsilon `1e-6` selected the smaller block. Every pass
executed 172 eligible normalizations: 43 layers × attention/FFN × two
chunks. Candidate selections were exactly **0/172/0**. The existing native
kernel body and F32 output remained unchanged; the thread count changes its
sum and reduction order. Other norm geometries and fused norms stayed ordinary.

The original F16 projections, HC suffixes, consumer precision, IQ2 compact
pairs with the selected occupancy-two specialization, output-A/HCA, state,
activation placement and workspace were unchanged. Added device storage was
zero. Input assembly/copies, preparation, products, dispatch, fences and both
complete frontier-head copies were timed. Existing plan setup, state clears,
loading and file writes were outside prefill timing; allocation capacities
remained equally funded. No warm source or archive was modified.

The preceding native profile records 174 plain RMS operations totaling
375.598432 ms, including the two trailing-head normalizations. That is an
upper budget for this factor, not the exact measured cost of its 172 inputs.
The broader FFN-input/routing and HC-attention-input categories contain other
work and differ from the literal pipeline's contracts.

## Complete-output controls

All four ordinary 129280-F32 heads matched the selected native goldens byte
for byte. Both candidate heads were finite and nonzero, with unchanged
argmaxes 1393/554; every value differed in each candidate head.

| Chunk | Maximum absolute logit drift | RMS logit drift |
| --- | ---: | ---: |
| 0 | 0.680179596 | 0.116898659 |
| 1 | 0.658787727 | 0.114400967 |

These are descriptive full-head differences, without a greedy-bound or
task-quality verdict. Changed reduction order can also change downstream
routing work; this screen does not attribute the slowdown to RMS alone.

## Conditions and provenance

Numerical source is `9dd6efd`; main was at the docs-only `35f93a1` during
the screen. The
community artifact is
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`;
the fixed 8192-ID TSV SHA-256 is
`0cbafc4bafd7a2c1e1c83f4a346815cdd500d2fb0bcb45afcfd826d591ae8e9a`
(ID payload `60329b1e4ff5d19d40082666e8c08b86655d4b1173486aecbc4a752bb6aa3ac7`).
Context8192, native fast, F16 KV/F32 HC, frontier heads, compact experts,
output-A and HCA were common to all arms.

The current locked GGML tree is
`cc7b7f0b962e3f536ccdd051161b6fb2117feac5ac8e2c0d6c1a417cfb394520`.
A private norm object, compiled with the actual O3/use_fast_math/sm121a
command, preceded the current archives; the link map excluded the original
norm member. The five-parameter MMQ header and coherent selected archives
were unchanged. SDK `aarch64-e0a0c85c42806fb1` and CUDA driver 580.178.04
matched the retained current closure. Ordinary norm source SHA-256:
`8cb872a9633a7c68c5e9fa8c1f7d39afe9e66636c1299b9fc610b1b4217e7307`;
private source `b6674bccf241d4a688f6fd21c3175996eb4d7f6ed51993f781a629b589869a8e`;
binary `df8a82dc8b5a9e32f41aaaa63e95901aaa036936a0fc6755e33456b049ed2d52`.

Supervised job `m3-ds4-flat-rms-r1` completed rc0/reaped in 51.304 s:
private compile/link 8.287 s, model command 40.176 s including one 6.980 s
load. Source, compile/link and retirement receipts plus all six raw heads
remain outside Git under
`/home/pmeenan/scratch/m3-ds4-flat-rms-records/{screen-r1,supervisor-r1}/`;
the private source/controller kit is
`/home/pmeenan/scratch/m3-ds4-flat-rms-factor/`.

## Unstarted next lead

Separately isolate the HC projection's accumulation/output contract while
retaining its F16 operands and ordinary normalization. Native
[`mul_mat_cublas.cu`](../../../src/kernels/ggml/mul_mat_cublas.cu) ordinarily
uses 16F accumulation/output for these F16 products on GB10; its existing
`plan.f32_output` path uses 32F accumulation/output with the same F16
operands and avoids the final output conversion. The literal
[`dsv4_ds4_product.cu`](../../../src/kernels/ggml/dsv4_ds4_product.cu) uses
that 32F contract. A paid factor restricted to HC weights `[16384,24]`
and inputs `[16384,4096]` would separate this arithmetic/copy difference.
**No source kit or run for that factor has started**, and the current category
budget does not isolate its latency or establish a gain.
