<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Concurrent generation measurements

## TensorFold: first 8K screen

TensorFold's explicit four-request deployment increased fresh
prompt-plus-generation throughput from 23.06 to 41.05 tokens/s in plain
mode (1.78×), and from 31.62 to 40.53 with speculation (1.28×). All fourteen
concurrent outputs matched their matching solo controls exactly. These are
one observation per cell, with serialized fresh prefills; they do not
establish pure decode scaling or parity with jitLLM's different weight format.

Spark B, 2026-10-01, one CUDA rank, image `jitllm-tensorfold:71377a53`
(`sha256:1a2afff2bd001cdea746d96bdb7614a4f937caa28e1646babd9acb951ea3aa48`),
TensorFold 0.3.6.2. Model: Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP at
`dadefa80`, affine 4-bit, group 32. This is a cross-quantization baseline
beside native/Mia NVFP4. The image and eight runtime source files were
authenticated, as were the model revision, configuration and read-only
file inventory; this run did not freshly hash every weight file.

Settings: `--parallel 4 --context 33792 --kv-dtype bf16 --mtp-drafts 6
--mtp-confidence 0.30`, Docker memory cap 110 GiB. Literal prompts contain
exactly 8,192 IDs in both the original and affine tokenizers. Every request
is greedy, has a 256-token cap, and has unique beginning/end markers.
Plain requests use the engine's per-request `draft: false` switch. All
timed requests reported `cached == 0`. The six bursts and their fourteen
matching solo controls completed at the length cap, without errors or EOS.
Solo controls can reuse their repeated prompt and are excluded from timing.
See the [protocol](protocol.md) for the common-barrier and cache conditions.

| Mode | Requests | Actual output tokens | Burst time (s) | Aggregate tokens/s | Median / worst request latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain | 1 | 256 | 11.101 | 23.060 | 11.101 / 11.101 |
| Plain | 2 | 512 | 16.016 | 31.969 | 16.014 / 16.015 |
| Plain | 4 | 1,024 | 24.948 | 41.046 | 24.944 / 24.948 |
| Speculative | 1 | 256 | 8.097 | 31.617 | 8.097 / 8.097 |
| Speculative | 2 | 512 | 13.485 | 37.968 | 13.349 / 13.485 |
| Speculative | 4 | 1,024 | 25.264 | 40.532 | 25.108 / 25.264 |

Aggregate throughput is actual completion tokens divided by first
submission to last terminal event, including fresh prefill and scheduling.
It is not the sum of per-request rates. First visible text arrived at the
following offsets from each request's own submission, in input member order:

| Mode | Requests | First visible text (s) |
| --- | ---: | --- |
| Plain | 1 | 3.595 |
| Plain | 2 | 6.744, 3.390 |
| Plain | 4 | 10.151, 6.787, 13.531, 3.421 |
| Speculative | 1 | 3.488 |
| Speculative | 2 | 3.494, 6.967 |
| Speculative | 4 | 13.964, 3.530, 10.484, 7.010 |

The roughly 3.4-second staircase shows fresh prefills running serially.
At four requests the whole-burst speculative rate was slightly below plain.
Warm conversation branches and arrivals during an existing decode are
needed to separate shared decode gains from prefill interference. SSE
chunks can contain multiple tokens, so these offsets and retained chunk
gaps are delivery measurements. This small screen supplies no stable p99.
The default single-request graph deployment, repeats and 32K remain open.

All 28 responses retained full output IDs, text, usage, finish reason and
wire events. All fourteen solo/concurrent comparisons matched IDs, text,
usage and finish exactly. The analyzer independently reconstructed all
six burst rates from per-request records. This is a same-engine isolation
check; it is not a model-quality comparison across weight formats.

Across 1,489 memory samples, node available memory ranged from
125,766,742,016 to 34,877,444,096 bytes, an observed drop of
90,889,297,920 bytes (84.65 GiB). Peak sampled server RSS was
30,775,250,944 bytes (28.66 GiB), separate from unified GPU backing.
These samples include setup and file cache; they are not interchangeable
allocation totals or an incremental concurrency-memory estimate.

Startup was 143.446 seconds; supervised total time was 373.484 seconds.
The same live server PID/start/image and source inventory were confirmed
before and after. Client and launcher retired, the container was absent,
and the strong post-run gate reported 117.146 GiB available.

## Provenance and exclusions

Raw records are external on Spark B at
`~/scratch/m3-concurrent-r1/tf8k-model-r2/` and on the workstation at
`/home/pmeenan/scratch/m3-concurrent-records/tf8k-model-r2/`. The input
receipt and analysis are beside them as `inputs8k-r2/receipt.json` and
`tf8k-analysis-r1.json`. The external frozen prepare/client/run/analyze
scripts are Spark-only; the model controller authenticates their identities
and the prepared input receipt. Repeating this screen requires those inputs,
the pinned image/model, and the same settings above.

| Input or result | SHA-256 |
| --- | --- |
| Prepared input receipt | `a947338b1d0fb09822bb37d19747711926e1e4d37cbfacc9e5cd75cc689bb791` |
| Prepare script | `47c4b02d6194a94040ebcc4d0c22c0b8ddf595e08390f44a1efd7b5ae2fa53a7` |
| Client script | `71e5497f1c24e31fd28d201ed95663d758513584bdafea337d9f2a6a6daadea0` |
| Model controller | `41033d5901fa3c605927e99c856068a44bf38e0bfc4d48606b22217d07420504` |
| Analyzer | `fc67308c4d49cb4dff6df761bca0dab77e240b54c91840721e7a897eaaed4dba` |
| Outer complete receipt | `8236f3f6ac6d00c4c93148c584a0e8ad7a4105047b856596fa11d8d351af0c02` |
| Client complete receipt | `b8aad1975031c2acb189f9e18823a3385e28c709502689f990394f038ecce5c5` |

Input preparation r1 failed before model work because its container UID
could not write the owned output directory. Preparation r2 ran as UID/GID
1000 and passed. Model job r1 was refused before model load: the strong
process check interpreted `tensorfold` in the supervisor's job name as a
model process. The neutral r2 name passed the unchanged check. Both failed
records are preserved and excluded from timings; neither ran model work.

jitLLM's queue baseline and Mia's NVFP4 speculative concurrency screen are
separate runs. No concurrent parity gate is closed by this report.
