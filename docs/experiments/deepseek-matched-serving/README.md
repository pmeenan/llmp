<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek matched serving inputs — 2026-10-04

The representative 7,043-token prefill-plus-frontier cell takes 7.0205 /
7.0574 s in native bookends and 6.4484 s in ds4: native wall time is 9.16%
longer than ds4, with 0.525% native bookend movement. This replaces the
unmatched in-process/HTTP comparison for this input. It does not qualify
isolated prefill or decode, long context, speculation or the whole M3 gate.

Spark A (`spark-c4e2`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. Each arm starts a fresh service and state directory,
completes one excluded nine-token, one-output prime, then pays for the entire
uncached input. Both engines run plain greedy decoding, context 262,144,
with capacity for four requests. Native resolves chunk rows to the fresh
state's 4,096-row fallback, output-A/HCA and partial chunks on. No calibration
record is imported. Service startup and the excluded prime are reported
separately from request wall time.

Native receives literal token IDs through buffered `/v1/completions`;
ds4 receives the corresponding non-thinking chat through buffered
`/v1/chat/completions`. The pinned ds4 parser, renderer and tokenizer freshly
produce all prompt IDs from the archived request. Complete ID records equal
the historical input controls, including the final `</think>` token 128822.
No marker is replaced. The routes differ; model inputs and output caps match.
The one-output wall includes prefill, frontier generation and response delivery;
there is no measured streamed time to first token on either buffered route.

| One-output arm | Wall s | Prompt tokens/s over wall | Peak MemAvailable drop GiB | Ready s | Excluded prime s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native before | 7.0205 | 1,003.20 | 88.250 | 1.517 | 6.864 |
| ds4 | 6.4484 | 1,092.20 | 105.768 | 22.556 | 0.164 |
| Native after | 7.0574 | 997.97 | 88.224 | 1.506 | 6.910 |

All three cells return one output with a length finish, 7,043 prompt tokens
and explicit zero cached tokens. Their response text hashes match. Memory is
sampled every 250 ms from before service start through shutdown: peak drop
is the arm's baseline MemAvailable minus its minimum, not GPU allocator bytes.
The native mean is 0.834 times ds4's in this cell. The memory sampler reports
no errors. All children return zero and are reaped; strong admission and
retirement probes pass before and after every arm.


## Matched 256-output cells

The longer replies pay for the same uncached 7,043-token inputs plus 256
outputs per request. Requests connect before a shared barrier, then submit
together. Aggregate throughput counts usage-reported completed tokens over
the burst's first submission through its last completed buffered response.
The native C1 mean wall is 1.71% longer than ds4; its C2 mean completed-token
rate is 13.16% higher, and C4 is 18.36% higher. Native bookend wall movement is 0.578% at C1 and
0.322% at C2 and 0.359% at C4. Single bookended cells are screens, not confidence intervals.

| Cell / arm | Burst wall s | Completed tokens/s | Request latencies s, in input order | Peak MemAvailable drop GiB |
| --- | ---: | ---: | --- | ---: |
| C1 native before | 19.6343 | 13.0384 | 19.6343 | 88.126 |
| C1 ds4 | 19.3608 | 13.2226 | 19.3608 | 105.804 |
| C1 native after | 19.7477 | 12.9636 | 19.7477 | 88.222 |
| C2 native before | 29.6196 | 17.2858 | 29.6196, 29.3984 | 88.631 |
| C2 ds4 | 33.5713 | 15.2511 | 33.5713, 32.8568 | 105.279 |
| C2 native after | 29.7150 | 17.2304 | 29.7150, 29.5151 | 88.730 |
| C4 native before | 50.1649 | 20.4127 | 49.9068, 50.1639, 49.5810, 49.2468 | 89.483 |
| C4 ds4 | 59.4835 | 17.2149 | 58.7636, 56.7041, 57.8092, 59.4829 | 105.799 |
| C4 native after | 50.3449 | 20.3397 | 50.0943, 49.4535, 49.7739, 50.3444 | 89.610 |

Native bookends' response hashes match for every input; u0 also matches
its native C1 response in C2/C4, and u1 matches across C2/C4. The other two
inputs only have C4 bookends here, without separate solo controls. Full native and ds4 replies differ. This is
neither a same-format greedy quality pass nor evidence of a failed logit
bound: that requires the registered target likelihood or near-tie controls.
All 21 measured requests return 256 outputs with length finishes, 7,043
prompt tokens and explicit zero cached tokens: 5,376 completed tokens.
There are no HTTP errors or sampler failures. All nine services return zero
and are reaped; every admission and retirement probe passes. ds4 records
1/2/4 cold admissions, 7,043/14,086/28,172 computed prefill tokens and
256/512/1,024 decoded tokens, with no cached prefill, speculative drafts,
serial fallback or failed request. Native capacity is the configured four
slots, not a time/response cap; ds4's CUDA continuous batch also admits four.
The native mean memory drops are 0.833 / 0.842 / 0.846 times ds4's at C1/C2/C4.
The final Spark probes find no GPU work or busy/waiting GPU jobs.

There is no native isolated decode timing on this buffered route. ds4's
own timing record reports C1 decode at 20.0 tokens/s; it is retained as
reference instrumentation, not compared to a fabricated native phase clock.

Reproduction sources and raw records stay outside Git in
`~/scratch/deepseek-matched-prefill/` on Spark A and
`/home/pmeenan/scratch/jitllm-m3-deepseek-matched-prefill-2026-10-04/` on the
workstation. Run the recorded controller with `--workload prefill --cells 1
--engines current-before-lit,ds4,current-after-lit` under the installed
GPU supervisor, then wait for its exit. For the 256-output cells use
`--workload long` and one of `--cells 1`, `--cells 2`, `--cells 4`, a separate
fresh output directory and supervised job per concurrency. All four jobs
finish successfully and are waited on. Only these three named engine arms
are qualified; inherited unused legacy engine options are not a reusable
launcher contract. `screen2` is the qualified first
screen. The excluded `screen1` used streaming literal requests, which native
rejects with HTTP 400; it supplies no matched performance ratio. An initial
long-cell setup also failed before model execution because the additional
ID generation steps were missing; the corrected setup regenerates and checks
all four inputs.

Native artifact:
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
Both sides use the community IQ2_XXS/Q2_K checkpoint
`antirez/deepseek-v4-gguf@f71f23d5`,
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
ds4 is v0.6.5, commit `76d51ef82a81b70b78e51a3a6ea11946286de976`,
`--cuda --no-spec --no-update-check -t 8 --ctx 262144`.
Native production source is the literal-cohort implementation at `17d577e`;
later commits through `49c3707` only record experiments.

| Identity | SHA-256 |
| --- | --- |
| Native runtime | `f1ee73b98a64e20bee69e0e689156b3b43c4b9b67b861ae45d805cd96e5a21ad` |
| ds4 server | `b8ea076dc17f7a4784ae63e8f7930a7835219ffc40718e564ff96341307dd93a` |
| ds4 ID frontend | `7f741a1393d4d7094eef977cb2c46c48c5133c2ca2f599eb69f4f9ccc1e6e922` |
| Controller | `5f4d3fe3e40212bc37b5699216c04d0edbcb8edf52f04c7d272135b4c25fa917` |
| u0 IDs, compact JSON | `bdace2542d5915ff4f5a28f0529776ed2f14cef4af835aa4f06e57ad93be6d23` |
