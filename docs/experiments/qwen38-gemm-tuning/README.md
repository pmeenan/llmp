<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 small-column GEMM tuning (M3)

Fresh cuBLASLt algorithms do not materially improve Qwen3.8's MTP verify.
Hyper-connection products stay within about 2% of the existing cuBLAS
calls, and the best large vocabulary-head product saves only about
0.09 ms against GGML. No runtime kernel, depth policy or precision changes
are retained. The [128K MTP speed gap](../qwen38-mtp-speed/README.md)
remains open.

## Method

`jitllm_qwen38_gemm_tune` is a benchmark, never a serving API. It searches
the SDK's heuristic algorithms for explicit BF16 weights, BF16 or F32
input, F32 accumulation and BF16 or F32 output. It records all nine
algorithm attributes rather than copying the image or EXL3 pins. The
case list is bounded to 32 shapes, each with at most eight columns,
2 GiB of weights, 64 candidates and 32 MiB workspace. The loaded
cuBLASLt must match the SDK version, and numerical environment switches
are refused.

The benchmark uses generated finite operands. Identical weight matrices
occupy distinct addresses covering 256 MiB; tiny diagnostic shapes cap
their copy count and report the actual smaller set. Each captured graph
has at least 64 calls and covers every allocated weight copy. Three
batches are timed with CUDA events after warm-up; the median per-call
time is reported. This removes repeated host library-dispatch gaps and
avoids repeatedly crediting one L2-resident HC matrix. Each algorithm's
output is checked for exact repetition and compared with GemmEx.
These are controlled operation measurements, not model-quality evidence.

The source pin is `62fe347` plus this benchmark, on `spark-b` (GB10),
2026-09-29, driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`, cuBLASLt 13.8.0.4. Each timed job exclusively
owned the Spark; the load gate required 105 GiB MemAvailable and no other
GPU/model process. No model was loaded. Weight formats and F16 caches
are unchanged.

## Hyper-connections

The fast verify's HC down product has 10,240 inputs, 320 outputs,
BF16 operands and F32 output. HC up has 320 inputs, 10,240 outputs,
BF16 operands and BF16 output. Both use F32 accumulation. The 41 copies
of each 6,553,600-byte matrix fill 256.25 MiB and all are covered by the
64-call timing graph. Job `m3-qwen-gemm-hc2`, 23:15:38–23:15:56,
completed the sweep after the benchmark build and SDK lint.

Times in microseconds:

| Columns | HC down GemmEx | Best Lt | HC up GemmEx | Best Lt |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 32.169 | 31.987 | 30.104 | 30.074 |
| 3 | 31.425 | 31.435 | 30.042 | 29.859 |
| 4 | 31.444 | 31.395 | 30.119 | 29.594 |
| 5 | 31.448 | 31.529 | 29.672 | 29.578 |
| 6 | 31.435 | 31.416 | 29.515 | 29.607 |
| 7 | 31.778 | 31.674 | 29.596 | 29.508 |
| 8 | 31.507 | 31.569 | 29.606 | 29.586 |

At three through five columns, the best down algorithm is
`[21,5,9,4,0,0,20,0,0]`, bit-exact to GemmEx for these fixtures.
The best up algorithms are also bit-exact. Their small isolated
differences do not justify a new production dispatcher.

## Vocabulary head and router

cuBLASLt directly refuses BF16 weights with F32 input on the measured
router and head shapes. GGML's multi-row BF16 `mul_mat_f` already
converts F32 input to BF16 inside its MMA tile load, so a library product
may preserve that operand precision by making the same conversion
explicit. The one-row vector path has a different contract and was not
changed.

`--ggml-reference on` feeds the same unrounded generated F32 input to
GGML MMF and to jitLLM's existing RN `ToBf16` conversion followed by Lt.
The conversion launch and its transient buffer are included in the
candidate timing. Candidate outputs are compared with both GemmEx and
the current MMF. Reduction-order differences are reported; no arbitrary
input or model-quality conclusion follows from this fixture.

The large head has 2,560 inputs and 248,320 outputs. Its BF16 weights
occupy 1,271,398,400 bytes, already larger than L2. Job
`m3-qwen-gemm-head3`, 23:34:34–23:35:17, completed successfully.
Six heuristic candidates were returned out of the requested 16.
Times in milliseconds, including conversion on the library side:

| Verify rows | GGML MMF | GemmEx | Best Lt |
| ---: | ---: | ---: | ---: |
| 3 | 5.572 | 5.517 | 5.498 |
| 4 | 5.586 | 5.511 | 5.496 |
| 5 | 5.579 | 5.515 | 5.509 |

The best head algorithm uses `[21,5,1,0,0,0,19,0,0]` at three rows and
`[21,5,1,0,0,0,20,0,0]` at four and five. It repeats exactly and equals
GemmEx in these fixtures, but differs from MMF by up to 0.000626.
The 1.3–1.6% head gain is only about 0.1–0.2% of the measured 55–62 ms
whole verify. It cannot close the model's gap, so no model A/B or
production wrapper was warranted.

The router's 512 outputs are smaller. The final harness rerun gives MMF
14.954/15.026/14.915 microseconds at three/four/five rows, against
15.560/15.570/15.572 for conversion plus the best Lt algorithm. Its
103-call graph covers all 103 weight copies (257.5 MiB), and completes
with the provider fence recorded, waited on, queried and released before
stream closure. The router offers no gain.

CUDA errors or library execution/internal errors terminate the benchmark
without running resource destructors or queuing another candidate:
unknown completion quarantines the resources until process teardown.
Clean unsupported library candidates may be recorded and skipped.

## Repeat and raw provenance

Raw logs, JSON lines and binary hashes remain on
`spark-b:~/scratch/m3-qwen-gemm-tune/`: `hc-down.jsonl`, `hc-up.jsonl`,
`mixed-support.jsonl`, `router.jsonl`, `router-final.jsonl`, `head.jsonl`
and the build's SHA-256 records. The reusable benchmark stays in Git.

For example:

```sh
jitllm_qwen38_gemm_tune --input bf16 --output f32 \
  --shape 3,320,10240 --shape 4,320,10240 --shape 5,320,10240
jitllm_qwen38_gemm_tune --input bf16 --output bf16 --shape 4,10240,320
jitllm_qwen38_gemm_tune --input bf16 --output f32 --ggml-reference on \
  --shape 4,248320,2560 --candidates 16 --reps 64 --warmup-ms 50
```

These results reject algorithm tuning for the measured Qwen shapes.
Different model shapes remain eligible for a fresh search; these
configurations are not production pins for other shapes or devices.
