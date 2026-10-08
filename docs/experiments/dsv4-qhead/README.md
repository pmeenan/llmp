<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek Q-head RMSNorm and RoPE fusion

The native fast prefill path can combine its unweighted per-head F32 RMSNorm
and normal-offset RoPE without changing either operation's arithmetic.
The selected graph node directly depends on the original Q projection and
positions, keeping its input live with its disjoint output. It removes the
normalization intermediate and uses the existing provider-owned stream,
completion and graph-capture path. Unsupported metadata and reference
graphs retain the two primitives; no precision or attention setting changes.

## Representative screen

The existing paid 8K ordered profile attributed 396.615 ms to Q RMSNorm and
397.665 ms to Q RoPE: 13.195 s of summed ordered intervals within a
13.309 s whole production pass, giving about a 6.0% ideal
latency ceiling for removing the entire pair. The output-A layout copy was
97.567 ms, under 0.74% of the pass, so copy-only tuning was ruled out
without a build or model run. These are operator-budget observations, not
an estimate of the fused kernel's measured speed.

Each arm below is a fresh Spark A process, with the same original community
IQ2_XXS ds4-comparison artifact, literal 8,192-token input, 8K state ceiling,
two 4,096-row chunks and complete final frontier head per chunk. The two
chunk plans are primed outside the paid clock; input assembly/copies,
original native products, cache writes, all 43 layers, dispatch, fences and
complete head copies are paid. Original math, Q8 preparation, F16 KV,
compact expert scheduling and ordered reduction remain selected. No event
markers or profiler run. Both diagnostic arms retain the same original
intermediates and group the pair for identical safe placement.

| Factor | Original before, s | Candidate, s | Original after, s | Throughput gain | Bookend movement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source-equivalent rotation | 12.759037849 | 12.398520076 | 12.782066159 | 3.0006% | 0.1805% |
| Native multiply/FMA order | 12.790292519 | 12.397988610 | 12.791666055 | 3.1698% | 0.0107% |

Gain is the mean original time divided by candidate time, minus one.
One before/candidate/after screen establishes a representative gain,
not a confidence interval or a context/task performance matrix.
Activation capacity is 2,067,791,872 bytes in all six arms; scratch capacity
is 190,840,832 bytes. The corrected screen's six complete 129,280-value
F32 heads are byte-identical by chunk, including the known original heads.

## Preserve actual rounding, not only the formula

The first candidate preserved the F32 normalization multiply but changed
every final logit. A model-free control isolated the first arithmetic
difference: the plain 448 normalized values per head were exact; only
the rotated tail differed at nonzero positions, by at most 2.3842e-7.
Actual native and candidate device instructions explained the difference:
native contracted `x0*sin + round(x1*cos)`, while the source-equivalent
candidate contracted `x1*cos + round(x0*sin)`.

Explicit rounded products and FMAs retain the original order for both
rotated values. The same tiny control becomes byte-exact at positions
0, 97 and 4095, and the corrected whole-model bookend becomes byte-exact
in both full heads. The source-equivalent candidate is rejected; its
final-head maximum absolute errors were 0.6983 / 0.7545, RMS 0.1476 /
0.1273 and NMSE 0.0009548 / 0.0004891, with the same two argmax IDs.
Those are diagnostic deviations, not task-quality acceptance evidence.
The correction obtains the gain while restoring exact arithmetic.

## Production confirmation

One pass of the actual selected production graph takes 12.415647269 s,
with both complete chunk heads byte-identical to the same original
hashes. Its two plans each contain 43 direct Q-head nodes and 3,527
steps, versus 3,570 primitive steps in the diagnostic controls. Actual
activation extent is 1,462,430,464 bytes and allocated capacity
1,530,920,960 bytes; the diagnostic controls allocated 2,067,791,872
bytes because they deliberately retained all original intermediates.
Scratch capacity remains 190,840,832 bytes. This single confirmation
checks production arithmetic, placement and paid work; the matched
diagnostic bookend remains the measured 3.17% causal speed result.

The production guard accepts only canonical contiguous F32
`[512,64,16..8192,1]`, one canonical I32 position per token, normal 64-value
rotation at offset 448, and finite supported parameters. Decode and small
verify chunks keep their primitives. The native operand test compares
33 and 2,048 rows, both actual window and compressed YaRN settings, with
positions 0, 97, 4095, 131071, 262143 and 1048575, including the trained
context endpoint. Each complete output is compared byte for byte.

The Spark locked build and all 1,246 tests pass. Routine CU casts,
parentheses and host-array lint repairs then pass rebuilt affected
native operand and planner tests. Clang-format, all eight changed-unit
clang-tidy checks and the source boundary check pass. REUSE and embedded
header checks also pass using the retained pure SDK REUSE package on Spark.
CUDA lint retains
the actual compile entry's definitions and include paths, replacing
NVCC-only driver flags with the SDK Clang CUDA host parser; the installed
CUDA13.0 headers supplement only its unconditional cuRAND-header include,
which the trimmed SDK lacks. Production compiles exclusively with the
actual NVCC/native flags and uses the pinned SDK cuBLAS payloads.

## Transfer and provenance

Qwen3.8 already combines weighted normalization and NEOX/IMROPE in
`QsaPrep`; this transfer fills the different DeepSeek normal-tail contract.
The current EXL3 Qwen2 path has separate RoPE but no corresponding
per-head RMSNorm to combine. Other widths, layouts, rotation types or
precisions need their own guards and operand qualification. The general
lesson is to preserve materialization rounding and inspect actual FMA
contraction when a mathematically equivalent fusion drifts.

Measured on `spark-c4e2`, 2026-10-02, the pinned AArch64 SDK
`aarch64-e0a0c85c42806fb1` (Clang22.1.8/NVCC13.4.92), SASS `sm_121a`,
the original locked native math flags and original static/library closure.
Fresh 105 GiB admission and native/GPU/container probes precede every
load, and each process is supervised and retired before the next arm.
Raw samples, sources, commands, receipts and complete head files remain
outside Git under `/home/pmeenan/scratch/m3-ds4-qhead-short-records/`;
Spark-side sources/results are under
`/home/pmeenan/scratch/m3-ds4-qhead-short-r1/`.

- Artifact manifest ID:
  `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
- Literal input file SHA256:
  `0cbafc4bafd7a2c1e1c83f4a346815cdd500d2fb0bcb45afcfd826d591ae8e9a`.
- Corrected numerical source: `qhead.cu` Git blob
  `a24d383cae83a8bc7d346c04304bcefad054b1ef`; build-r6 and model-r2
  receipts record actual selected inputs and successful supervised exits.
- Full original/corrected chunk-head SHA256 values:
  `d7481e856eaf08617cc325452a109f4abccffc6bf272372183eb17d2883829b9`
  and `fa7df546fa88bd1523d1a332da4d7028d1db95a55bdfecadb31790b5c593c786`.

Workstation checks remain deferred under the owner's optimization-run
override. No package ships from this experiment.
