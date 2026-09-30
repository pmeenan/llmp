<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 adaptive-depth and MXFP8 crossover — 2026-09-30

The paired-prefix acceptance estimator does not improve the default prefix
head on the canonical 128K prompt. Its core implementation was removed;
the benchmark retains actual depth trajectories and measured phase costs.
The long-context MTP speed gate remains open.

## Independent adaptive-depth captures

The existing policy compares separate reward EWMAs for depths two and three.
The candidate instead shares the recent clipped prefix reward between them,
adding an EWMA of the extra accepted position for depth three. Both retain
the calibrated relative cost 1.16, three-percent switching margin, four-step
initial exploration and bounded periodic probes. No clock enters either
policy. The clipped reward is a prediction: different verify widths can
change arithmetic at near-ties.

`m3-qwen-paired-model` runs on `spark-b` / `spark-56f5`,
03:02:10–03:10:08, with SDK `aarch64-e0a0c85c42806fb1`. Each head runs
candidate, original, then candidate again in six fresh processes. Each has
`--repeats 1`, so the final candidate repeat makes independent policy choices
rather than replaying a saved depth sequence. The benchmark gates memory and
other GPU processes before each load; free memory is 117.15–117.22 GiB.

All runs use runtime-style prefill in 4,096-row chunks, context ceiling
262,144 and 512 generated tokens. The canonical phase-1 128K prompt has
128,799 rendered tokens and stable boundary 128,794; all six prompt-ID arrays
match. User-content SHA-256 is
`b7154f4bdfea17ea92e20ddaa95b2ece37ea306972af7039894fd0e81854da1d`.

| Head | Original tok/s | Candidate tok/s | Fresh repeat tok/s | Original depth 2 / 3 steps | Candidate depth 2 / 3 steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix 65,536 | 45.532 | 43.825 | 43.606 | 56 / 124 | 134 / 63 |
| Curated 47,172 | 43.625 | 44.393 | 43.897 | 113 / 81 | 109 / 84 |

The prefix candidate regresses 3.75%, repeated at 4.23%. The curated result
improves 1.76%, repeated at 0.62%; this small result does not justify a second
core policy. Each candidate/repeat pair has identical 512-token outputs,
actual `[anchor position, requested depth, verify rows, kept rows]` traces
and depth histograms. The prefix candidate differs from the original at
token index 354; curated outputs match the original. These are timing and
repeat controls, with no quality claim for the discarded policy.

Complete-step mean costs, excluding truncated tail verifies, are in
milliseconds. They reuse the benchmark's two existing elapsed-time reads
and never feed policy choices:

| Head / policy | Depth 2 draft / verify | Depth 3 draft / verify | Observed total-cost ratio |
| --- | ---: | ---: | ---: |
| Prefix / original | 6.774 / 50.732 | 9.225 / 55.154 | 1.120 |
| Prefix / candidate | 6.478 / 49.570 | 9.602 / 55.926 | 1.169 |
| Curated / original | 6.148 / 50.698 | 8.743 / 56.182 | 1.142 |
| Curated / candidate | 5.945 / 50.046 | 8.321 / 55.691 | 1.143 |

These costs describe the observed trajectories and graph behavior; they are
not width-independent counterfactual costs. Neither policy uses them.

The unchanged target is `c4fb47a9…`, prefix head `056a750e…`, and curated
head `8600a998…`. The curated artifact uses the externally supplied Mia
47,172-ID list, SHA-256 `ee819d25…`, not an independently generated
vocabulary. Complete pins and license provenance are in the
[draft-head study](../qwen38-draft-head/README.md#provenance-and-reproduction).

The three-file prototype overlay preserved the separate DeepSeek work in
the shared Spark mirror. Actual benchmark SHA-256 is
`21a7540e12de99e5e1c96f5c7558ae98bed80f7b283607f5cb4cf37deea020ae`.
Its header, benchmark and test source hashes, archived patch, raw JSON and
validated summary live outside Git at
`spark-b:~/scratch/m3-qwen-paired-depth/`. Seven focused policy controls
passed, including changing acceptance, copied checkpoint choices and tail
exclusions. The failed candidate is absent from the retained core; only
benchmark trajectory and depth-cost diagnostics remain.

Mia's matched 128K MTP comparator remains 48.652 tok/s. These native
diagnostics do not replace the final runtime ladder or close its gate.

## Small-column tensor-core crossover

The isolated diagnostic compares the current F32-input MXFP8 vector product
with the existing F32-to-MXFP8 quantization, weight-scale swizzle and F32
CUTLASS GEMM. Every candidate call includes all three operations. No global
eight-column threshold, recurrence dtype, HC fusion or graph policy changes.
The tensor-core input rounds differently; this is a precision comparison,
not a model-quality result.

`m3-qwen-mxfp8-crossover2`, 03:12:26–03:12:42 on the same Spark,
compiles the external diagnostic with the existing GEMM benchmark's exact
SDK host flags and linked kernel archives. It first checks nine short/tail
shapes: columns 3/4/5, outputs 4/48/52, inputs 128, and twelve F32 elements
of stride padding. Then it times eight actual products at each column count.
Each timed graph visits every weight bank, with the actual set larger than
the device's 24 MiB L2. The cap is 1,024 banks; all timed sets are at least
123.75 MiB. Three CUDA-event batches give a median, normalized by actual
graph calls, with vector controls before and after.

Mean before/after vector time and candidate time, in microseconds:

| Outputs | Inputs | 3 columns vector / TC | 4 columns vector / TC | 5 columns vector / TC |
| ---: | ---: | ---: | ---: | ---: |
| 12,288 | 2,560 | 142.692 / 171.853 | 144.896 / 172.539 | 144.514 / 170.163 |
| 10,240 | 2,560 | 119.885 / 142.870 | 119.709 / 143.672 | 118.736 / 138.719 |
| 6,144 | 2,560 | 68.264 / 83.522 | 69.434 / 83.915 | 72.090 / 84.359 |
| 2,560 | 6,144 | 70.646 / 93.360 | 75.132 / 90.791 | 74.490 / 89.856 |
| 2,560 | 640 | 8.523 / 15.038 | 9.167 / 17.177 | 11.380 / 14.765 |
| 640 | 2,560 | 8.977 / 25.433 | 10.291 / 27.336 | 10.125 / 25.615 |
| 512 | 2,560 | 7.678 / 25.095 | 8.275 / 25.156 | 9.032 / 25.546 |
| 48 | 2,560 | 3.831 / 23.812 | 4.211 / 24.132 | 4.546 / 23.827 |

Every complete candidate is slower. No model trial or precision adoption
is warranted. This result covers the existing composite path; it does not
isolate quantization, swizzle or GEMM individually and therefore cannot
attribute the slowdown to one piece. Their wrappers and tail/stride
controls remain usable for wider prefill or future diagnostics.

All nine correctness and 24 timed cases repeat bit for bit and pass finite,
guard-byte, input-quantization and weight-swizzle checks. The sampled FP64
product of the actual quantized operands passes the fixture's predeclared
NMSE bound 1e-10. Against the original unrounded F32-input vector product,
timed-fixture NMSE is approximately 5.4e-4–6.9e-4. That difference is
reported, not interpreted as an acceptable quality loss. Reported extra
GEMM workspace is zero for these shapes; quantized-input and swizzle storage
are still allocated and charged separately.

The external source SHA-256 is
`2e5a6acf2925a7e67a3a7a73709aaffaf34f14538926a3574aaf59fd4281c1f8`,
and binary SHA-256 is
`560fceb312db16aaa01783ef408b54a09b0f91ecca56a949c754850d0b1f7265`.
The archived source, exact compile/link commands, linked-archive hashes,
raw JSON, controls and validated summaries are in the same external
`m3-qwen-paired-depth` directory. The unadopted precision diagnostic stays
outside the production tree.
