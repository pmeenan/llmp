<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek prefill chunk screen — 2026-10-04

Keep **4,096-row chunks**. On the recorded 7,043-token literal input,
8,192-row chunks take 52.43% longer through the first returned output than
the mean production bookends. Control wall movement is 0.109%. The larger
setting increases peak memory and establishes no default or calibration win.
This is a native settings screen, not another ds4 comparison or an isolated
GPU-prefill clock.

## Fresh-process bookends

The same final checked runtime serves every arm, changing only
`prefill_chunk = 4096 → 8192 → 4096`. Each service has fresh anchors/data,
completes an excluded nine-token, one-output weight prime, then pays for
all 7,043 uncached prompt IDs through buffered `/v1/completions`. The
one-output wall includes prefill, frontier generation and delivery.

| Chunk rows | One-output wall s | Peak MemAvailable drop GiB |
| ---: | ---: | ---: |
| 4,096 before | 6.991509 | 88.107 |
| 8,192 | 10.662847 | 91.681 |
| 4,096 after | 6.999155 | 88.055 |

All three measured requests return HTTP 200, one output token, length
finishes, 7,043 prompt tokens and explicit zero cached tokens. Their combined
reasoning/text digest agrees:
`8b218b1d3c571d7f8f754bcac93c85e9abf4a384dadd24abce3cd2d3755ace66`.
That single output is not a complete-logit, initialized-state or fixed-oracle
quality pass. No wider quality/context ladder follows a slower first screen.
No kernel, chunk-shape or arithmetic attribution is collected.

All three services exit zero and are reaped; samplers terminate with no
errors. Six strong 105-GiB admission/retirement gates find GPU, container
and native-model probes clear. Peak memory is the drop from each arm's
pre-start MemAvailable to its minimum through shutdown, not allocator bytes.

## Configuration and evidence

Spark B (`spark-56f5`), GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92. Community artifact
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`,
plain greedy decoding, context 262,144, four request slots, output-A/HCA and
partial chunks enabled, no imported calibration or numeric/library overrides.
The input is the [matched serving study's](../deepseek-matched-serving/README.md)
`u0`: complete compact-JSON ID digest
`bdace2542d5915ff4f5a28f0529776ed2f14cef4af835aa4f06e57ad93be6d23`.
No template/tokenizer round trip enters the native input.

The fixed binary is checked production `8ebea5f`, SHA-256
`bf7747a34130ad3e16f42b773fa311f60e6079a829a3daa3a6b3743fcdbb3170`;
its retained 487-file compiled-source inventory is
`f0162fbb398b67698fd164b758ae5eddda90b0d2c7abe60956b11d2e35ca0f9d`.
The controller validates this binary, every source file, the canonical
preflight helper and complete literal input before and after each arm.
Production passed all 1,547 Spark B tests; this rejected setting introduces
no code change and needs no repeated suite. Workstation/package checks remain owed.

Controller SHA-256:
`484c22849c61324fe57d0edec36d64b1e2738184778e5a62f6bc9b68167c9e28`;
aggregate report:
`fe13e5f1a379121969d3b45d7deb19de70d650ee06541285974b6fdb178b8184`.
Raw script, inputs, pins, configs, full responses, gates, logs and retirement
receipts remain at `spark-b:~/scratch/dsv4-prefill-chunks/` and
`~/scratch/jitllm-m3-dsv4-prefill-chunks-2026-10-04/`. Installed supervisor job
`dsv4-prefill-chunk8192` completes zero and is waited on. Reproduce the
recorded controller with `--workload prefill --cells 1 --engines
current-before-lit,current-wide-lit,current-after-lit` and fresh output,
under installed `spark-job --gpu`, then wait.
