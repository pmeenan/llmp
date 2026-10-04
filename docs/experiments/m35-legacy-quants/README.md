<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 legacy quant primitives (2026-10-04)

This bounded slice extends native primitive coverage. It does not qualify a
model or a whole quantization format. Gemma and legacy runners still owe
matched quality and performance against their pinned same-format references,
production solo and optimized independent-request batching, paging/VMM,
state, graph and context controls before supported status.

## Implementation and limits

The pinned GGML source is llama.cpp b10964,
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. Build patch 0002 now compiles
its unchanged Q4_1/Q5_0/Q5_1 generated MMQ wrappers. Native ordinary and
paired MMQ dispatch, operand validation and get-rows use the same 21-type
compiled closure. Prepared tree SHA-256 is
`1ae467d0fced412beb16faca3d0a910441477a06f1e2bae4ccbe0cdb6230bd11`;
patch 0002 SHA-256 is
`9c64de9e3b6d7eab2486724ea78d9d48a1f6db1b730559329cdb5cb89ba6b53b`.
The source lock and MIT audit name the added units.

Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL gain the applicable row-invariant and selected
VecQ building blocks. Joined VecQ loads each supported legacy weight block
once per P-column pass and retains each column's original Q8_1 arithmetic;
the selected one-token variant reloads for subsequent live route pairs.
The existing ballot route scan, shared Q8 preparation, independent column
sums and one-token launch selector apply. No Gemma runner default changes.
The existing fused VecQ gate writer is SwiGLU; Gemma's split GeGLU is a
separate primitive/graph slice and is not qualified by this report. Compact
prefill, tensor-core choices and other transfer candidates need their actual
runner operand/shape measurements; this slice does not infer them from VecQ.

Non-512-multiple K operands must explicitly fund the canonical readable
512-value last-row tail; fixtures zero it. Expert pitches preserve complete
blocks and 256-byte alignment. The row-invariant kernel now guards partial
output-row blocks, including prefetch, reads and writes. Ordinary pinned
MMVQ reads rounded output-row blocks (RE-045), so its plan conservatively
requires N divisible by the selected rows-per-block for every admitted type.
Selection reproduces the pinned runtime host table, type traits and small-K
rules; routed multi-token kernels use two rows. Refusal submits no work.
A caller can choose the guarded row-invariant primitive where eligible;
unsupported partial ordinary shapes remain explicit refusals.
Operand validation also bounds the combined weight row/channel/sample offset
in signed 32-bit native quant-block units, including the final 512-value
padded step. Host-only exact-limit/overflow controls and planning-only
synthetic bindings prove refusal before submission for ordinary/paired
products; no out-of-range GPU operation is executed.

## Correctness controls

Controls use real Gemma down/gate widths K704 and K2816, readable tails,
strided expert arrays and 128 experts/top8. Dense products cover 1/2/4/16/48
columns, and routed products include duplicate selected experts. Paired MMQ
shares preparation and agrees exactly with separate products across the 19
non-FP4 types; the paired fixture excludes MXFP4 and NVFP4. Legacy get-rows agrees with independent dequantization (zero NMSE).
Row-invariant columns retain one-column sums bit for bit. Joined Q5_1 down
at N2816 is bit exact against solo launches for 1/2/4/16 independent slots;
clearing the final slot leaves earlier outputs unchanged. This is an operand
inactive-column control, not a whole-engine cancellation test.

The Q4_0/Q4_1/Q5_0/Q5_1 all-variant oracle independently decodes scalar nibbles/Q5 high
bits, accumulates integer products and reproduces GGML's FP16 scale products
and Q8_1 sum correction. It does not call the production packed dot helpers.
Original and candidate outputs for these four meet 1e-10 NMSE against this reference; the
separate ideal FP64 product keeps the 5e-4 upstream bound. Original reference
weights used for odd-N controls are separately padded to safe whole row
blocks. New N129/K704 tests give the guarded kernel only its canonical tail,
compare all real outputs and preserve output-stride/trailing sentinels for
legacy five plus existing Q8_0. Ordinary N129 refusal controls cover all 21
types, dense/routed and 1/2/4/8 columns, and prove unchanged outputs and no
fault after rejection whenever the selected block has multiple rows.

## Bounded paid-preparation benchmark

Harness: [gemma_quant.cc](../../../benchmarks/gemma_quant.cc), target
`jitllm_gemma_quant_bench`. Run `128 0`, `128 2`, `128 8` for the retained
128 timed launches per arm. Each arm runs original A, selected one-token
VecQ B, four-token variant P4 (four rows only), then original A. CUDA events
measure complete input preparation plus product on the same stream. All
allocation tails are zeroed; copies and timing/completion calls are checked.
Warm original/P1 outputs have zero NMSE, and warm P4/P1 outputs are bit exact.

Geometry is Q5_1 K704/N2816, 128 synthetic experts, top8 and one/four rows.
Routes change each launch by `(launch*17 + (slot<shared ? 0 : row*3) +
slot*7) % 128`. The four-row patterns share exactly 0, 2 or 8 experts with
each later row. They are declared synthetic eligibility controls; no measured
model-routing distribution is implied. Inputs and weights are seeded finite
synthetic values, with original raw quant blocks and identical route maps.
The B dispatch is `SetVecQOneToken` r2/w4/P1; P4 is variant7 r2/w4/P4.
Both use shared Q8_1 preparation. Original is pinned `MulMatVecQ` with its
own paid preparation. No persistent weight conversion or omitted adaptation.

Host: `spark`/GB10; driver 580.178.04; pinned SDK
`aarch64-c09daba6ac31edee`, NVCC 13.4.92, Clang 22.1.8. Supervised job
`gemma-quants-overlap` completed successfully on 2026-10-04. Latencies are
microseconds per complete preparation/product launch:

| Shared experts | Rows | Original A | P1 B | P4 | Original A after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 54.090 | 52.593 | — | 57.592 |
| 0 | 4 | 227.884 | 195.572 | 262.618 | 229.371 |
| 2 | 1 | 54.340 | 52.634 | — | 54.068 |
| 2 | 4 | 158.294 | 169.540 | 246.077 | 156.912 |
| 8 | 1 | 54.222 | 52.412 | — | 53.935 |
| 8 | 4 | 57.631 | 116.700 | 157.547 | 57.981 |

The zero-overlap solo bookend moves 6.5%, so it is not a strong solo adoption
result. Four-row P1 is faster for disjoint routes and slower with two/eight
shared experts. P4 loses in every measured pattern. Retain ordinary MMVQ as
the available fallback and qualify the runner's actual routing/dispatch;
there is no universal Gemma VecQ/P4 selection from this screen.

Ordinary planned peak scratch is 9,216 bytes (one row) or 36,864 bytes
(four rows). P1/P4 product scratch is zero; their separately charged Q8_1
input buffer is respectively 9,216/36,864 bytes. Weights occupy 190,316,544
logical bytes plus one 384-byte tail. The harness reserves 4 MiB scratch and
uses cudaMalloc; these primitive plan charges are not measured whole-engine
peak workspace or VMM qualification.

## Verification boundary

Focused Spark correctness includes the CPU operand tests, ordinary/refusal,
paired/get-rows, row-invariant and joined/all-variant controls described above.
Local light checks cover source-lock validation, source/closure tooling (96
controls, three existing skips), SPDX/REUSE, headers, boundaries, formatting
and diff whitespace. The three new unchanged MMQ instance units also compile
with the pinned native sm86+sm121a profile; no discrete GPU execution was run.
Raw logs remain in external session scratch and Spark job storage; this
report records aggregate results and provenance.
