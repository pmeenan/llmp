<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek HC projection: F32 accumulation screen

Keep the current product. Switching DeepSeek's HC projection from cuBLAS
16F accumulation/output (plus an output conversion) to the existing 32F
accumulation/output path is **+0.25%** in one bookended 8K screen on
Spark A (`spark-c4e2`, 2026-10-02), smaller than the **−0.41%** bookend
movement. Heads change. No 32K control was run (the gain is under 1.5%).

| Paid prefill, two 4096-row chunks | Seconds |
| --- | ---: |
| Original 16F, before | 9.438819049 |
| Candidate 32F accumulation/output | 9.396311761 |
| Original 16F, after | 9.400163342 |

Throughput change is `(mean(original times) / candidate time − 1) × 100`;
bookend movement is `(original-after / original-before − 1) × 100`.

## What changed

Only products with F16 weights `[16384,24]`, F32 input `[16384,4096]` and
F32 output `[24,4096]` on GB10 (cc 1210) took `plan.f32_output` in
`mul_mat_cublas.cu`: the same converted F16 input and F16 weights, F32
accumulation, F32 written directly, no output conversion. A private
selector chose the arm per pass; selections were 0/172/0 of 172 eligible
products per pass (43 layers × attention/FFN × two chunks). Native RMS,
the F32-to-F16 input conversion, HC suffixes, output-A/HCA, compact IQ2
pairs with occupancy two, placement and workspace were unchanged.

The result matches the budget. The output conversion this removes is only
24 × 4096 values per product. The product is bound by memory traffic, so
the accumulation type barely matters. Most of its time goes to the
F32-to-F16 conversion of the 256 MiB normalized input, which stays.

## Complete heads

Both original arms matched the native goldens `8df7719c…`/`7150fcba…` byte
for byte. Candidate heads were finite and nonzero. Their argmaxes were
unchanged (1393/554), but almost every value changed: chunk 0 changed
129,279 of 129,280 values, maximum absolute drift 0.660, RMS 0.124;
chunk 1 changed all 129,280, maximum 0.535, RMS 0.108. These are
descriptive differences, not a quality verdict.

## Conditions

Same community artifact `cd39d504…`, fixed 8192-ID input `0cbafc4b…`,
context 8192, native fast plan, F16 KV/F32 HC, frontier heads, compact
experts, output-A and HCA as the [flat-RMS screen](../ds4-flat-rms/README.md),
using its collector, with one 6.98 s load outside the paid time. Numerical
source is main `27f610b` plus the selector. A private
`mul_mat_cublas.cu` object, compiled with the actual native command, came
before the unchanged warm archives, and the link map excluded the
original member. Binary
`7d94fce57f327a36d07eb04fca70807f37121fea0d6901492a07873a709bc571`,
receipt `65e839426e2624401a5689e3111092ddf56a5b4af069a0df8da7179b0fd340b5`.
Supervised job `m3-ds4-hc-proj-r1` completed rc0 in 52 s (model command
39.9 s); final preflight showed 117.19 GiB free and clear probes. Raw heads
and logs remain outside Git under `spark:~/scratch/m3-ds4-hc-proj-r1/`.

## Where the HC input gap actually is

ds4 never materializes the F32 normalized HC row. Its HC expand kernel
also writes the next mix's F16 RMS-normalized activations. The mix GEMM
reads those directly; where no producer emitted them, ds4 fuses RMS into
the F16 conversion. Native writes and rereads a 256 MiB F32 normalized
tensor, then converts it again inside the cuBLAS launcher. A native
counterpart would extend the prepared, byte-exact HC-post+RMS fusion
(+1.94% private screen) to emit the F16-rounded normalized row, and pass
it to this product as a direct F16 operand. Product operand bytes stay
identical and 16F accumulation stays. That change has not been run.
