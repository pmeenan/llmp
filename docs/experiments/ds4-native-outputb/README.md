<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Restore native output-B over original D4

The current native GGML Q8 MMQ consumer matches the literal ds4 consumer
in the complete 8K pipeline: 7.5891 s versus 7.5759 s, a 0.17% throughput
difference, with all twelve saved full heads byte-identical. Three controls
on identical original upstream operands also produce byte-identical complete
outputs. Keep this native seam and investigate routed FFN next; no additional
output-B tuning is justified by this result. Production dispatch is unchanged.

## Matched singleton

Spark A runs the authenticated community IQ2_XXS/Q2_K artifact, exact 8,192
IDs, all 43 layers, two 4,096-row chunks, original FP8 KV/FP4 indexer
settings and one final 129,280-entry head. One preparation takes 292.433 s.
Each pass initializes fresh state. The private `--output-b-study` performs
an original warmup, three original passes, a native warmup, three native
passes, an original return warmup and three original return passes.

Only output-B's consumer changes. All 86 calls use M=N=4,096 and K=8,192.
The original output-A still produces F32 activations and their D4 sidecar;
the consumer borrows that current producer and the same resident raw Q8_0
weights. Both arms charge the original 256-block zero guard and exact
output sanitizer in one native launch operation, using the same stream,
closure, job fences, cache, routing, FFN and head cadence. Every other
selected stage and paid pool remains identical. There are 93 native jobs
per ordinary pass and zero added profiling marks.

The generic borrowed entry is restricted to checked contiguous, single-plane
dense Q8_0 products with F32 logical input/output, current guarded D4 and
disjoint operands/workspaces. It allocates no activation or weight replica,
performs no input quantization and draws only the current native MMQ fixup.
Refused descriptors queue nothing and never retry another consumer.

Both pinned Q8 source configurations use 128-by-128 tiles, eight warps and
256-value K iterations on the qualified GB10. They share integer-dot and
F32 scaled accumulation semantics. This geometry statement follows the
configured source; it is not a kernel trace. Actual resolved workspace
records are zero for every output-B in both arms, so neither charges a
stream-K fixup buffer for this shape. Original and native stream-K policies
remain their respective implementations; their measured outputs below agree.
The current native Q8 kernel instance compiles with `-O3 -use_fast_math`
for `sm_121a`. The original units retain `-O3 --use_fast_math -lineinfo`;
no compiler flag changes are part of this singleton.

## Unmarked complete-pipeline timing

Comparable time is the full prefill wall plus final result copy, excluding
preparation and diagnostic capture. Warmups are excluded.

| Consumer and role | Individual seconds | Mean seconds |
| --- | --- | ---: |
| Original before | 7.532180142 / 7.539942246 / 7.561117666 | 7.544413351 |
| Native MMQ | 7.588052450 / 7.585954628 / 7.593160086 | 7.589055721 |
| Original after | 7.589111845 / 7.609231118 / 7.623707818 | 7.607350260 |

The six-original mean is 7.575881806 s, or 1,081.33 tokens/s. Native is
1,079.45 tokens/s, a rate ratio of 0.998264. The six original individual
maximum/minimum ratio is 1.012152; the three native samples' ratio is
1.000950. Original after/before mean ratio is 1.008342. The difference is
below the fixed 10% coarse regression threshold and smaller than bookend
drift; this is comparable performance, not a speedup claim. The earlier
[profile](../ds4-restoration-profile/README.md) measures output-A, output-B
and HC expansion together, so its 17.94% attention-output share does not
belong to this consumer alone.

## Byte and operand controls

All twelve full heads, including all three warmups, are finite and exactly
equal to the saved original head:
`499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8`.
Full-head maximum difference, NMSE and softmax total variation are zero;
both greedy IDs are 554. This fixed-input equality does not replace
real-answer, decode or long-context quality gates.

The first original diagnostic warmup captures second-chunk layers 0, 21
and 42. Each native probe consumes the same ORIGINAL F32 input, current D4
bytes and raw Q8 weights, writes a distinct paid output, and leaves the
original output intact for downstream HC expansion. Capture/readback costs
are excluded from the nine ordinary times. Complete inputs, guards,
weights, outputs and metadata remain retained outside Git.

| Captured layer | Complete output elements | Native/original bytes | Maximum difference / NMSE | Largest sampled FP64 dot error, both consumers |
| --- | ---: | --- | --- | ---: |
| 0 | 16,777,216 | Equal | 0 / 0 | 5.430e-7 |
| 21 | 16,777,216 | Equal | 0 / 0 | 3.072e-7 |
| 42 | 16,777,216 | Equal | 0 / 0 | 7.659e-7 |

All captured F32 values and Q8/D4 scales are finite; the full D4 guards
are zero. Twelve fixed coordinates per layer independently decode D4 and
raw Q8 into FP64 dot products. These check consumer arithmetic against the
same quantized operands, not against unquantized activations or model quality.

## Memory and retirement

The study symmetrically owns an extra 67,108,864-byte device control and
134,217,728-byte pinned host capture capacity in every arm. Its execution
budget is 99,123,342,848 bytes, versus 98,922,016,256 in the profile.
At 250 ms sampling, process RSS peaks at 1,045,676,032 bytes versus the
profile's 902,381,568, exceeding the 1.1 ratio. The recorded whole-process
memory gate remains **false**. This comparison includes symmetric diagnostic
staging and does not isolate consumer memory. Node MemAvailable drop is
102,825,398,272 bytes versus 102,725,160,960, about 1.001 times; it includes
paging and cache. No memory bound is waived or claimed passed automatically.

The child exits 0, is reaped and the strong retirement probe passes with
116.899 GiB available and no GPU/container/native model process. Source,
SDK, compile database, cache, controllers and binaries retain their pinned
identities after execution.

## Qualification and provenance

Measured on Spark A, 2026-10-01, 02:23:40–02:30:37 America/New_York;
supervised job `ds4-outb-model-r1`, exit 0. The final source passes prepare,
the full locked Spark-native suite (1,199/1,199 tests, including 256 GPU
tests, 80.10 s), SDK format on 19 units, all nine actual tidy compile arms,
boundaries and 1,106 SDK REUSE/header checks. The original 115 numerical
definitions and ten prepared headers retain exact source correspondence.
Independent whole-source and controller reviews pass. Workstation checks
remain deferred to the end of the optimization run.

Raw receipts, nine timing samples, twelve complete heads, actual dispatch
and three complete operand/output bundles are retained at
`spark:~/scratch/m3-ds4-outb-r4/{check-r1,model-r1}/`, mirrored locally under
`/home/pmeenan/scratch/m3-ds4-outb-{check-r1,model-r1}/`. Reproducible
prepared weight files remain on A. Aggregate provenance is below.

| Identity | SHA-256 |
| --- | --- |
| Community artifact | `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac` |
| Exact prompt ID bytes | `60329b1e4ff5d19d40082666e8c08b86655d4b1173486aecbc4a752bb6aa3ac7` |
| Qualified source map, 1,226 files | `dd337c38181433c36848ea282e13e78e21ab21798503aa2e53c9c722e9e49355` |
| Locked qualification receipt | `28c4b801e599d6ed7c18d2737e57abb08ff9ebd2d4ba0e91914bc3d1d673fbf5` |
| Native benchmark binary | `b5e72c00a536f4d0dc77abd00533fe6d57182962057932f4d36ae7649bb4a09e` |
| Model controller | `f6dcfbced4347ed4ef4dc3548cb2270488987a68d4aaac9d6118e1fa849587e6` |
| Model receipt | `1663838ec26a5df1ec9dd8fc305ac1c4702fb49fa3a6931ec7f68934d68062fc` |
| Native result receipt | `2598f5b58fcf0eee32836690db809f4be2a767821fd7761da5d71c851d30e622` |
| Same-input operator controls | `0da75e2015c1b2844a2fc1445c2d0aa4fac34b069e1c44edcdb1d090bfd0b0b7` |
| Actual native Q8 instance compile evidence | `ba26317ee17553d13fee72f40205937fe9a636e7ffb543c11e289b62e9529104` |
| SDK metadata, `aarch64-e0a0c85c42806fb1` | `f38891fcc394ddf956aaa196c6895d7309148b8b5159b0c3a0459721946992bf` |
