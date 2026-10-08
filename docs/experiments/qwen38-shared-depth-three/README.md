<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen shared draft cap three — 2026-10-04

Keep the shared draft cap at **two**. Raising it to three reduces C4
completed-token throughput by **13.75%** against the bookended controls;
the control rate moves down 1.99%. All twelve measured replies complete.
No production code, default or production calibration changes follow this rejected
settings screen. The earlier [cap-one screen](../qwen38-shared-depth/README.md)
also rejects lowering the cap.

## Paid comparison

One checked runtime executes three fresh-process arms, changing only
`shared_wave_depth = 2 → 3 → 2`. Each explicitly keeps
`draft_wave_max = 2`: this does not enable the previously rejected wider
joined drafting. The cap limits each branch's adaptive depth while more
than one request is in the cohort; branches can choose two or three, and
a lone request retains its adaptive policy. Actual wave depths and
acceptance counters are not instrumented. This is a settings comparison,
not a claim that every candidate wave drafts three tokens.

| Shared draft cap | Completed tokens/s | Burst wall s |
| ---: | ---: | ---: |
| 2, before | 32.631090 | 31.381115 |
| 3 | 27.859805 | 36.755462 |
| 2, after | 31.981326 | 32.018685 |

Rate gain is `(mean control seconds / candidate seconds − 1) × 100`;
bookend movement is `(before seconds / after seconds − 1) × 100`.
Each arm pays the frozen [four-request fixtures](../qwen38-four-request-waves/README.md):
8,256 uncached prompt tokens and 256 outputs per request, HTTP 200 and
length finishes, 1,024 outputs per arm. An excluded one-token weight prime
precedes each burst. The buffered clock covers prefill, generation,
queueing and HTTP through the full response; it supplies no TTFT or
individual phase timing. Submission spreads are 0.254/0.481/0.756 ms.

All three services exit zero and are reaped without KILL; client children
are reaped and memory samplers terminate without errors. Each arm passes
a strong admission gate and a strong retirement gate: all six require
105 GiB available and clear GPU, container and native-model probes.
This screen makes no peak-memory claim.

The complete candidate choices differ from both controls for all four
members. The cap-two bookends also differ for members zero and two,
agreeing for one and three. Arrival-dependent wave arithmetic remains;
these observations are not a quality verdict or an exact-response claim.
Full logits, initialized state, acceptance and additional context controls
are not run because the representative screen rejects this setting.

## Configuration and provenance

Spark B (`spark-56f5`), GB10, driver 580.178.04, SDK
`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92. Context 33,792, chunks 4,096,
slots four, speculation enabled, normal wave lanes and 2,048-cell wave
alignment, greedy requests, fresh data/anchors and no imported calibration
or execution/numerical overrides. Target artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
selected drafter
`8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Requested `draft_vocab = 65536` executes the physical selected 47,172-row
BF16 head and its matching I32 ID map in every arm. The controller checks
those resource representations and the actual effective settings log;
this is not a vocabulary or head-implementation factor.

All arms use final checked production `8ebea5f`, runtime SHA-256
`bf7747a34130ad3e16f42b773fa311f60e6079a829a3daa3a6b3743fcdbb3170`.
The retained 487-file compiled-source inventory has SHA-256
`f0162fbb398b67698fd164b758ae5eddda90b0d2c7abe60956b11d2e35ca0f9d`.
The controller checks every listed source file, runtime, frozen client,
input receipt, preflight/lifetime helper, both artifact indexes and
tokenizer/template hashes before and after every arm; artifact payloads
are not rehashed. It also requires all twelve actual usage counts and
zero cached tokens. Source and runtime remain unchanged across the triplet.
The preceding production Spark check passes 1,547 tests; no new build or
repeated suite is run for this rejected existing setting. Workstation and
package checks remain deferred.

Controller SHA-256:
`8401e56327d4f54fe32bdec69e5d374f07ddfab8d36c9b683be2de803d438555`.
Frozen client:
`4f76adb8e36bb97e04c30f22d28d8d2ffc985d67aa4621f083ac92fc5e85d8f3`.
Aggregate report:
`6f38f0e9384daa86e913143b6e79d354f085049661e8a6c56a73351342116282`.
Raw controller, pins, configs, full responses, settings logs, six gates
and retirement receipts remain at
`spark-b:~/scratch/qwen-shared-depth-three/screen1/` and
`~/scratch/llmp-m3-qwen-shared-depth-three-2026-10-04/`;
spill payloads remain on Spark B. Installed GPU supervisor job
`qwen-shared-cap-three-screen` completes zero and is waited on, taking
165 seconds. Reproduce with the frozen fixtures/client, the checked
runtime and settings above, using fresh anchors/data and installed
`spark-job --gpu` admission and retirement for every arm.
