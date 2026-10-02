<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 Flash 0731 concurrent serving baseline

One condition on Spark A (`spark-c4e2`, GB10, driver 580.178.04),
2026-10-02 13:24–14:02 EDT. Both engines run the 0731 UD-Q2_K_XL GGUF
(the M3 pinned checkpoint, `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`):
native from its prepared artifact `8a355bfb…`, llama.cpp from the GGUF.
This mirrors the [Qwen protocol](../serving-concurrent/README.md): buffered
non-streaming chat, greedy, 256 outputs, a fresh service per cell, and an
excluded one-token prime before each burst. C1 sends u0, C2 u0/u1 and C4
u0–u3 at one barrier. The clock runs from HTTP submission through the full
response, so it includes queueing, prefill and generation. These are not
pure-decode or TTFT figures. Each cell is one sample.

**Inputs.** The four Qwen common-v2 texts are reused. The one literal
`<think>` in each was changed to `[think]` so that user content cannot
forge DeepSeek's thinking token. Rendering uses the GGUF template with
thinking on, the default in both engines. The rendered text and IDs agree
between the engines. The native renderer and tokenizer, run model-free,
produce exactly the IDs that llama.cpp's own `/apply-template` and
`/tokenize` produce, checked in every llama.cpp cell, and the IDs from its
vocabulary-only `llama-tokenize`. Each prompt is 7,043 rendered tokens:
DeepSeek's tokenizer turns the Qwen texts into fewer tokens than Qwen's,
which gives 8,256. Every measured request reports 7,043 prompt tokens and
0 cached tokens, and generates all 256 outputs with finish `length`.
No cell had an error or a refused request.

## Native prefill chunk size

Native C1 runs u0 with DSpark through the production chat route, streamed,
with `context = 262144`. Each run starts a fresh process; the sizes
alternate, two runs each, and one run per size uses the canonical 32K
timing prompt. Prefill is the time from send to the first streamed piece,
which includes one decode step. The harness is m3lc `longctx.py`.

| `prefill_chunk` | 8K prefill s (tok/s), run 1 / run 2 | 32K prefill s (tok/s) | DSpark decode tok/s, 8K | Fixed / workspace / chunk inputs, GiB | Peak `MemAvailable` drop, GiB (8K ×2, 32K) |
| ---: | --- | --- | --- | --- | --- |
| 2048 (old default) | 12.633 (557.5) / 12.657 (556.4) | 55.503 (571.2) | 31.47 / 31.43 | 1.42 / 1.28 / 0.04 | 106.47, 106.44, 106.74 |
| **4096** | 11.266 (625.1) / 11.298 (623.4) | 48.323 (656.1) | 32.75 / 32.86 | 2.70 / 2.45 / 0.10 | 108.48, 107.81, 108.05 |
| 8192 | 12.026 (585.7) / 11.491 (612.9) | 48.259 (657.0) | 32.70 / 32.75 | 5.63 / 5.05 / 0.26 | 111.15, 111.10, 111.30 |

Compared with 2048, a 4096 chunk raises prefill throughput by 12.08% at 8K
(mean of the two runs) and by 14.86% at 32K. 8192 is no faster at 32K and
slower at 8K, where the whole prompt fits in one chunk. It also costs
another 2.9 GiB of fixed memory. The 32K prompt (31,705 tokens) runs at
4096 without errors, with at least 9.17 GiB available at its lowest. The
native cells below therefore use 4096, the new default to a 262,144-token
context (above it the runtime keeps 2048, whose mask, sized for the whole
context, costs 2.14 GiB less fixed memory at 1M). On the same cells, the 2048 path
gives 12.355 / 12.750 / 12.629 tok/s with DSpark and 10.491 / 10.559 /
10.573 plain at C1 / C2 / C4. The 4096 path is 8.8% and 5.7% faster at C1.

## Concurrent results

| Engine (`prefill_chunk` 4096 native) | C1 completed tok/s | C2 completed tok/s | C4 completed tok/s |
| --- | ---: | ---: | ---: |
| Native, DSpark (production) | 13.442 | 13.515 | 13.654 |
| llama.cpp b11254, DSpark n-max 3, `--parallel 4` | 8.839 | 9.503 | 10.078 |
| Native, plain | 11.091 | 11.152 | 11.192 |
| llama.cpp b11254, plain, `--parallel 4` | 7.635 | 8.949 | 9.793 |

| Engine/cell | First completion, s | All completed, s | Per-request latency, s, input order (u0…) | Peak `MemAvailable` drop / min available, GiB | Max RSS, GiB |
| --- | ---: | ---: | --- | --- | ---: |
| Native DSpark C1 | 19.04 | 19.04 | 19.04 | 107.71 / 9.37 | 1.61 |
| Native DSpark C2 | 19.02 | 37.88 | 19.02 / 37.88 | 107.88 / 9.32 | 1.72 |
| Native DSpark C4 | 18.83 | 75.00 | 74.99 / 56.30 / 18.83 / 37.50 | 107.83 / 9.35 | 1.79 |
| llama.cpp DSpark C1 | 28.96 | 28.96 | 28.96 | 106.76 / 10.50 | 2.05 |
| llama.cpp DSpark C2 | 53.48 | 53.88 | 53.87 / 53.48 | 107.10 / 10.14 | 2.39 |
| llama.cpp DSpark C4 | 100.95 | 101.61 | 101.49 / 101.61 / 100.95 / 100.95 | 107.49 / 9.79 | 2.57 |
| Native plain C1 | 23.08 | 23.08 | 23.08 | 96.77 / 20.28 | 1.55 |
| Native plain C2 | 23.09 | 45.91 | 45.91 / 23.09 | 97.03 / 20.22 | 1.63 |
| Native plain C4 | 23.03 | 91.49 | 68.67 / 91.49 / 23.03 / 45.83 | 97.04 / 20.18 | 1.65 |
| llama.cpp plain C1 | 33.53 | 33.53 | 33.53 | 95.58 / 21.63 | 1.94 |
| llama.cpp plain C2 | 57.08 | 57.21 | 57.08 / 57.21 | 95.67 / 21.55 | 2.03 |
| llama.cpp plain C4 | 103.84 | 104.56 | 104.33 / 103.84 / 104.01 / 104.56 | 95.59 / 21.64 | 1.95 |

**Gap.** On this protocol, native has no completed-token gap at C2 or C4.
With DSpark it leads llama.cpp by 52.1% / 42.2% / 35.5% at C1 / C2 / C4,
and plain by 45.3% / 24.6% / 14.3%. The lead narrows as concurrency rises.
Native runs DeepSeek requests one at a time, so its rate is flat:
+1.6% from C1 to C4 with DSpark, +0.9% plain. llama.cpp's continuous
batching scales by +14.0% with DSpark and +28.3% plain. Native's request
latency grows with queue position, about 19 s or 23 s per request ahead of
it. llama.cpp finishes all its requests together, near its total wall.
Native completes its last request sooner in every cell: 75.0 against
101.6 s at C4 with DSpark, 91.5 against 104.6 s plain. A projection beyond
C4, or for shorter prompts, where llama.cpp's batched decode weighs more,
is not measured. DeepSeek independent-state batching is still the open
native item.

Peak memory ratios against llama.cpp are 1.00× with DSpark (107.88 against
107.49 GiB at worst) and 1.01× plain (97.04 against 95.67). Both are within
the ~1.1× bound. Native's figures include the 4096 chunk's 1.3 GiB of
extra fixed memory. RSS is not the unified device allocation.

**Admission limits.** Native serves one DeepSeek request at a time; the
others wait first come, first served, in a queue of 64 places
(`max_queued`). A non-streaming request leaves the queue with a 429 after
120 s. In C4 the last request waited about 56 s (DSpark) or 68 s (plain),
so nothing was refused. At these per-request times, the eighth (DSpark) or seventh (plain)
simultaneous buffered 7K request would wait past 120 s and be refused;
this is inferred and was not run. Streams have no fixed wait. The native
context is 262,144 tokens per request. llama.cpp runs 4 slots of 65,536
tokens each (`-c 262144 -np 4`, `kv_unified = false` as logged), and
further requests queue. llama.cpp accepts DSpark with `--parallel 4`; its
per-request draft acceptance is 159–167 of 262–286 drafted tokens.

**Startup.** Native is ready in 1.0–1.5 s. The prime then pays the weight
activation: 7.6–8.6 s, outside the measured burst. llama.cpp loads in
81–105 s before it reports ready.

## Conditions and provenance

- Native: `jitllm-runtime` SHA-256
  `a8ab24049f353f5641ccb7d466e1d88c1947d1ac56875a75192ee66b5fc994c0`. It
  was built on Spark A from a tree whose `src/` equals main `7907561`.
  Target `8a355bfb…`, drafter `dd2d3f9c…`, `speculation = true|false`,
  `context = 262144`, `prefill_chunk = 4096` (2048 in the reference
  cells), F16 caches, loopback.
- llama.cpp: image `jitllm-llamacpp:b11254-cuda13`
  (`sha256:6dd025913fdf0ac258850ac1405dde3edbdb64c7c6d60533ee63e16448f66607`,
  source `8019dc56`), unpatched. Arguments: `-ngl all -fa on -c 262144
  -np 4 --fit off -cram 0`, with the request's `cache_prompt: false`, and
  `CUDA_DISABLE_PTX_JIT=1`. DSpark adds `-md
  dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf --spec-type draft-dspark
  --spec-draft-n-max 3 -ngld all`. Continuous batching is llama.cpp's
  default.
- Formats differ: native uses its prepared artifact of the GGUF with F16
  caches; llama.cpp uses its own kernels on the same GGUF. No ID-level
  equality of the generated outputs is claimed, and there is no quality
  verdict.
- Inputs receipt SHA-256 `9bbd4c15…`. The parent Qwen v2 receipt is
  `d5a35e63…`. The controller (`run.py`) is `a80fe2ec…`; the two 2048
  reference runs used `a5d60d0b…`, before the `--chunk` option was added.
  The preparation script is `e3089ba5…`. The chunk sweep used m3lc
  `longctx.py` `2c112f7f…` with `baseline.py` `41c693de…`.
- Every cell passes the shared preflight (at least 105 GiB available, no
  GPU process, container or model process) before it loads and after it
  retires. The services exit with rc 0, the containers stop and are
  removed, and all supervised jobs (`dsconc-native`, `dsconc-chunk`,
  `dsconc-main`) finish rc 0.
- Raw records stay outside Git, in `spark:~/scratch/m3-deepseek-concurrent/`
  (`inputs/`, `tools/`, `records/`).
