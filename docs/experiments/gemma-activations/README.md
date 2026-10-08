<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma activation primitives and floating fusion screen

The native registry now has F32 split GeGLU with GELU-tanh and packed F32
unary GELU-tanh. The separate floating MMVF GeGLU fusion retains primitive
fallback, but its planner eligibility callback defaults off: the N2112
shared-FFN shape loses in this screen. The N704 result is a bounded
synthetic primitive result, without a model or approved-GGUF speed claim.

## Contracts and correctness

The pinned Gemma graph uses `LLM_FFN_GELU` → `ggml_geglu_split`, both for
shared FFNs and experts, in llama.cpp `b29c606e`'s `src/llama-graph.cpp`.
GGML's CUDA GELU is
`0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x*x)))`, rather than ERF or quick
GELU. CPU validators refuse other GLU/unary variants, malformed bindings,
shifted aliases and unsupported layouts before launch.
Packed unary GELU is available through the registry and primitive graph
planning (`fusion=false`); `fusion=true` retains its existing refusal of
graphs containing UNARY.

Spark controls cover F32 uniform strided input rows, views of concatenated
gate/up storage at N2112 and expert N704 ×8 selected rows ×4 tokens,
packed in-place updates, near zero, signed zero and extreme finite inputs.
Solo versus batched GeGLU results are bit-identical; the three memory
domains (cudaMalloc, device VMM and host VMM) pass. The largest point error
against the FP64 tanh definition, normalized by `1+abs(reference)`, is
8.50e-7 for the GeGLU fixture and 4.62e-6 for unary GELU's random fixture.
The unary bound is 1e-5 to allow the pinned CUDA tanh approximation.

MMVF reads its destination's first parameter as accumulation precision.
A GeGLU enum there selects F32 accumulation, while ordinary F16 products
default to half accumulation. F16 fusion consequently requires both
products to request F32 explicitly; default F16 precision stays unfused.
The fused output is exactly equal to the two matched F32-accumulation
products plus GeGLU at both measured widths, and repeats exactly through
registry dispatch. Four-row MMVF fusion is refused; those products select
the ordinary tensor-core matrix path then the GeGLU primitive.

## Paid chain screen

Measured 2026-10-04 on `spark-b` (hostname `spark-56f5`), NVIDIA GB10,
driver 580.178.04; official locked `spark-native`, aarch64-linux-gnu,
RelWithDebInfo SDK `aarch64-c09daba6ac31edee`. The prepared GGML tree is
`e86191a0c9e9655d153d1895a630780c67bb472961f6527c1c9d1d60fdcac991`.
The implementation starts from llmpalooza `fc48ca5`; the final policy remains
default off and does not change the measured arithmetic.

Weights are deterministic synthetic resident **F16**, inputs F32, both
products explicitly request F32 accumulation. K=2816; the candidate
computes both products and GELU-tanh GLU, while fallback pays both products
and the primitive. Each timing uses eight warmups and 64 complete chains,
CUDA events on llmpalooza's stream and uncaptured native launches. No product
or input preparation is borrowed from another arm. F16 needs no runtime
input quantization; upload and checkpoint loading are outside the screen.

| Screen | N / rows | Fallback A1, µs/chunk | Fused B, µs/chunk | Fallback A2, µs/chunk |
| --- | --- | ---: | ---: | ---: |
| Initial shared shape | 2112 / 1 | 54.5005 | 62.6390 | 53.9165 |
| N704 repeat 1 | 704 / 1 | 10.1560 | 7.4545 | 10.1410 |
| N704 repeat 2 | 704 / 1 | 10.2670 | 7.5505 | 10.2680 |
| Initial four-row shared shape | 2112 / 4 | 79.3680 | unsupported | 76.0755 |
| N704 four-row repeat 1 | 704 / 4 | 16.4140 | unsupported | 16.3800 |
| N704 four-row repeat 2 | 704 / 4 | 16.3815 | unsupported | 16.4145 |

N2112 fusion is 15.6% slower than its mean bookend. The N704 repeats reduce
timed chain cost by 26.5% and 26.5%; their bookends move 0.15% and 0.01%.
The first N704 screen's A1 was 20.9% above A2, prompting these two narrow
repeats instead of using that initial number to select a path.

Reproduce the correctness test and screen with `ggml_ops_test` filter
`GgmlOpsTest.Gemma*`; set `LLMP_GEMMA_ACTIVATION_TIMING=1` to time the
floating chains. Set `LLMP_GEMMA_ACTIVATION_TIMING_N704_ONLY=1` and
`--gtest_repeat=2` for the N704 repeats. Spark work uses the installed
`spark-job start --gpu` supervisor, followed by its `wait` command.

The approved GGUF shared weights are Q8_0 and routed gate/up arrays Q4_K.
They cannot use this floating MMVF fusion. Their quantized joined products
and GeGLU/quantization writers still need implementation and measurement.
This screen also supplies no capture/replay, end-to-end inference, full
model quality, optimized batching or matched reference-engine result.
Those remain mandatory before a model/quant is qualified for support.
