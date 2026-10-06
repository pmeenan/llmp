<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 C4 attention over independent cache roots

Removing K/V packing lowers the paid 32-wave interval by **7.13%** with all
128 heads, four initialized states and 128 choices unchanged. A fresh
original/current/current/original bookend leaves **0.265% higher native latency**.
Strict quality still **fails: two positive-margin choice differences out of
128**, unchanged from the [packed compound screen](../gemma26-compound-packed-c4/README.md).
This is a closed manual backward transfer of the
[31B owner-root consumer](../gemma-owner-root-c4/README.md), with no production selection.

| Paid interval | First / repeat, seconds | Mean, seconds |
| --- | --- | --- |
| Same-binary packed/plain-on | 1.075940 / 1.072380 | 1.074160 |
| Same-binary owners/plain-on | 0.997080 / 0.997994 | 0.997537 |
| Fresh original C4/u128 | 1.002550 / 0.994855 | 0.9987025 |
| Fresh owners/plain-on | 0.996020 / 1.006670 | 1.001345 |

The causal factor order was packed/owners/owners/packed; the second screen was
original/owners/owners/original. These short intervals do not establish sustained
parity. Full head publication, lower-ID argmax and remaining Q/mask packing stay
paid. Finite scans, state copies and file witnesses follow the timer. All paid
intervals replay 32 graphs without new captures.

The fixture uses four unequal 64–67-row prefixes, three forced anchors, then
32 one-query waves (128 complete head rows), context/read width 256 and max
rows 128. Ordinary products remain; existing compound policies select 60
normRoPE, 90 normADD, 30 routing, 30 reduction and 121 plain-norm fusions at
all scalar prefill shapes and the first C4 plan. Other optional policies are
zero. These are selected implementation counts, not CUDA launch counts.
The 31B CLI/defaults are preserved; the new 26B compound CLI defaults plain norms on.

Before any owner run, packed/plain-on matched the historical `983388cd…`
134,217,728-byte head file and all four 57,671,680-byte / 60-range state files.
Every subsequent native arm matches those complete files and the 128 choices,
with finite full heads and owner endpoints 99–102. Fresh stock heads repeat
`5808306e…`; 92/128 rows match native bytes exactly. The first three waves have
8/12 exact rows and no choice differences; the later 116 rows have 84 exact
rows and both differences. The maximum raw logit delta is 4.36018.

| Step / owner | Consumed query position | Native / reference winner | Reference winner-over-native margin |
| --- | --- | --- | --- |
| 3 / 1 | 71 | 107 / 108 | 0.0279255 |
| 4 / 3 | 74 | 249598 / 246977 | 0.3656230 |

Query positions are zero-based consumed input positions; the completed-position
boundary and next forced token position are query + 1. Neither reference winner
is tied. The selected 16 likelihood/TV rows in [results.json](results.json) are
not corpus PPL or a quality gate. Mixed stock prefill routing eligibility and
native final-layer frontier narrowing remain distinct recipe differences;
this result proves no unique cause and introduces no layer whitelist or alias imitation.

The unchanged checked ten-source operation preserves all eight actual cache
writer dependencies. Runtime plans for 16 query heads report original grids
**48 blocks at D256/KV8** and **64 blocks at D512/KV2**, with matching repeat
geometry and the original reduction partitions. The 31B global grid of 96 is
not inherited. Removing the balanced K/V CONCAT tree analytically avoids
0.859375 GiB of read/write traffic per wave; this is not measured bandwidth.

Build and three no-device metadata modes pass 36 profile/owner/row cases plus
16/32-head, D256/D512 descriptor/refusal controls. All six installed supervised
jobs and explicit application/container retirement checks succeed. No full
suite or wider-context/assistant qualification ran under the owner override.
Raw arrays, rows, logs and telemetry stay external.

Measured base is `017c25e`, source frame `01891de0…`, private helper `3a337352…`
and SDK receipt `fbf84c74…`. Spark A (`spark-c4e2`, GB10) uses inherited checked
driver 580.178.04 and native NVCC 13.4.92 / CUDA toolkit 13.4.2 / Clang 22.1.8.
The retained public client `01889d8c…` runs unchanged in image `837fc732…` with
physical C4, batch/ubatch 128, F16 caches, non-unified KV, total context 1024 /
per-owner 256 and allowed graphs (32 actual reuses). Its five global layers
occupy 20 MiB and 25 local layers 200 MiB. Full pins, outputs, geometry,
selected rows and official record identities are in [results.json](results.json).
Fresh TensorFold entry resolves `609ca419…` / 0.6.5, with only an MLX26 recipe.

Reproduce with the existing `jitllm_gemma_owner_c4` target and
[closed protocol](../gemma-owner-root-c4/PROTOCOL.md):
`ARTIFACT NEW_DIR 26 4 joined compound IDS_I32`, setting
`JITLLM_GEMMA_OWNER_C4=packed|owners`, `JITLLM_GEMMA_C4_NORMMUL=1` and
`JITLLM_GEMMA_C4_PHASES=0`. Supply canonical 4,096-byte `b2d7aaf6…` IDs, use
new owner-only output directories, and require complete historical packed
identity before the owner factor and fresh stock comparison. Wider reads,
window crossings and production batching remain unqualified.
