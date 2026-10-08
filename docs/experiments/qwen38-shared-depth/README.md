<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen shared draft-depth screen — 2026-10-04

Keep **two drafts per shared wave**. On a fresh four-request 8K HTTP
triplet, configuring one draft reduces completed-token throughput by
9.75% against the mean production bookends. The control rate moves down
1.48%. All twelve measured replies complete; no production code or default
changes follow this rejected settings screen.

## Paid comparison

All arms use the same checked production binary and selected MTP head,
changing only `shared_wave_depth = 2 → 1 → 2`. A lone request retains
its adaptive depth. No acceptance counts, wave-width distribution or
individual kernel/phase times are collected. This is the full buffered
serving clock, including prefill, queueing and generation.

| Shared drafts | Completed tokens/s | Burst wall s | Peak MemAvailable drop GiB |
| ---: | ---: | ---: | ---: |
| 2, before | 32.397327 | 31.607546 | 81.217 |
| 1 | 29.021536 | 35.284142 | 81.113 |
| 2, after | 31.916612 | 32.083606 | 81.249 |

The frozen [four-request fixtures](../qwen38-four-request-waves/README.md)
each pay 8,256 uncached prompt tokens and return 256 tokens, HTTP 200 and
length finishes: 1,024 outputs per arm. A one-token weight-prime request
is excluded from the burst clock and counts. Each service exits zero and
is reaped; memory samplers terminate cleanly, and all six strong admission
and retirement gates require 105 GiB available and clear GPU, container
and native-model probes. No exact cross-arm replies, complete logits or
initialized state bytes are compared; arrival-dependent wave arithmetic
remains. This is not an expanded quality, context or acceptance gate.

## Configuration and provenance

Spark B (`spark-56f5`), GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92. Context 33,792, chunks 4,096,
slots four, normal speculation and wave lanes, unchanged draft-wave limit
two, greedy sampling, fresh anchors/data, no numeric/library overrides
and no imported calibration. Target artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
selected drafter
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Requested `draft_vocab = 65536` executes the selected 47,172-row head in
every arm, as the corrected settings log explains. It is not a head factor.

The runtime is final checked production `8ebea5f`, SHA-256
`bf7747a34130ad3e16f42b773fa311f60e6079a829a3daa3a6b3743fcdbb3170`;
its separately retained 487-file compiled-source inventory is
`f0162fbb398b67698fd164b758ae5eddda90b0d2c7abe60956b11d2e35ca0f9d`.
The controller authenticates that binary and every source file before and
after each arm, plus the frozen client and preflight helper. Source and
binary are unchanged across the triplet. Production's preceding Spark
check passes 1,547 tests; no repeated suite is required for this unchanged
binary and rejected setting. Workstation/package checks remain deferred.

The controller SHA-256 is
`64c83b585e76aae6a7812121a8771f2685acc1e1aa31afb12d078ac69e4e106b`,
aggregate report
`d5e691ed2112d1f9db0e8f465db7cdc2023f9efd9079b0361ad2264df873f790`.
Raw controller, source/binary pins, configs, full responses, gates, logs
and retirement receipts remain at
`spark-b:~/scratch/qwen-shared-depth/screen1/` and
`~/scratch/llmp-m3-qwen-shared-depth-2026-10-04/`.
Installed supervisor job `qwen-shared-depth-one` completes zero and is
waited on. Reproduce with the frozen fixtures/client and the settings
above, keeping each fresh-process arm under installed `spark-job --gpu`.
