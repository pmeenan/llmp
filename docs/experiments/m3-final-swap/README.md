<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Final M3 image-inclusive swap table — 2026-10-04

The production runtime passes all **32 swaps**, all six ordered model pairs,
with no reported problems. Every endpoint is under the approximately ten-second
goal: worst LLM-to-LLM **9.853267 s**, worst prepared **9.853267 s**.
First-use rows also pass the 40-second bound and prepared rows the 20-second
bound. This corrected refresh passes the current implementation's M3 swap-time
and 8K swap-correctness criteria, alongside the completed
[standard-client gate](../m3-standard-client/README.md). M3 performance,
quality, final record and deferred workstation/package checks remain open.

Spark B (`spark-56f5`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. One unchanged production runtime process registers
original DeepSeek 0731 plus DSpark, Qwen NVFP4 plus selected MTP and
Qwen-Image-2.1. DeepSeek uses context 262,144 and forced
`wave_form = "speculative"`; Qwen uses context 33,792 and the selected
47,172-entry head (requested cap 65,536). Both have 4,096-row prefill chunks
and max slots four. No prior
calibration/state directory is adopted. The source inventory pins the
corrected production implementation, which
passes all 1,543 Spark B tests, changed-file format/tidy and source lint.

An LLM A holds 8,192 tokens from the frozen `4655685:docs/decisions.md`,
then spills and returns for a 16-token continuation. Every ordered pair
runs first-use and prepared A→B→A cycles, plus a prepared zero-context cycle
when A is an LLM. First-use clears the relevant LLM's plans/graphs within
this process; image plan clearing is a no-op. It does not claim a cold page
cache or that the model has never been registered. Prepared 8K LLM returns
retain graphs and replay them.
The endpoint includes eviction/spill, restore, page-in and setup. Fresh LLM
rows select one greedy token from the completed prefill head; 8K returns
stop after the first continued decode step, and image rows after the first
denoising-step output. State hashing is outside endpoint clocks. The fresh
token budget is one with stop handling disabled; the generation finishes
synchronously before its endpoint clock is read. Complete prefill-head
hashes remain the repeatability controls. B's short question is the frozen capital-of-France
prompt; image prompt is the frozen red teapot at 1,024², 40 steps and seed 42,
using the prior diffusers BF16 initial latents.

| A / B | Direction / preparation / saved context | Endpoint s | Evict/spill s | Restore s | Page-in s | First output s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| deepseek / image | A->B, first use (8192 context tokens) | 5.796144 | 1.878338 | 0.000000 | 2.313868 | 1.602057 |
| deepseek / image | B->A, first use (8192 context tokens) | 8.780375 | 0.537373 | 0.091978 | 8.011561 | 0.136457 |
| deepseek / image | A->B, prepared (8192 context tokens) | 5.691463 | 1.894826 | 0.000000 | 2.314399 | 1.480359 |
| deepseek / image | B->A, prepared (8192 context tokens) | 8.747877 | 0.538294 | 0.088212 | 8.008519 | 0.108495 |
| deepseek / image | A->B, prepared (0 context) | 5.676301 | 1.885429 | 0.000000 | 2.313720 | 1.475250 |
| deepseek / image | B->A, prepared (0 context) | 8.924692 | 0.550215 | 0.000000 | 8.069857 | 0.300649 |
| deepseek / qwen3.8 | A->B, first use (8192 context tokens) | 8.119611 | 1.950299 | 0.000000 | 5.785024 | 0.379850 |
| deepseek / qwen3.8 | B->A, first use (8192 context tokens) | 9.613103 | 1.374070 | 0.094414 | 8.002206 | 0.138826 |
| deepseek / qwen3.8 | A->B, prepared (8192 context tokens) | 8.050498 | 1.951611 | 0.000000 | 5.781971 | 0.312319 |
| deepseek / qwen3.8 | B->A, prepared (8192 context tokens) | 9.600624 | 1.373980 | 0.093211 | 8.020737 | 0.108285 |
| deepseek / qwen3.8 | A->B, prepared (0 context) | 8.048520 | 1.936788 | 0.000000 | 5.794922 | 0.312309 |
| deepseek / qwen3.8 | B->A, prepared (0 context) | 9.853267 | 1.489095 | 0.000000 | 8.093548 | 0.267661 |
| image / deepseek | A->B, first use (image) | 8.954589 | 0.574271 | 0.000000 | 8.074884 | 0.301615 |
| image / deepseek | B->A, first use (image) | 5.849284 | 2.058534 | 0.000000 | 2.312998 | 1.475836 |
| image / deepseek | A->B, prepared (image) | 9.015261 | 0.568914 | 0.000000 | 8.073881 | 0.368838 |
| image / deepseek | B->A, prepared (image) | 5.783008 | 1.999490 | 0.000000 | 2.320741 | 1.460862 |
| image / qwen3.8 | A->B, first use (image) | 6.623283 | 0.562692 | 0.000000 | 5.790181 | 0.266357 |
| image / qwen3.8 | B->A, first use (image) | 5.118973 | 1.329138 | 0.000000 | 2.316337 | 1.472314 |
| image / qwen3.8 | A->B, prepared (image) | 6.579878 | 0.543824 | 0.000000 | 5.792005 | 0.240070 |
| image / qwen3.8 | B->A, prepared (image) | 5.129767 | 1.331360 | 0.000000 | 2.321434 | 1.475697 |
| qwen3.8 / deepseek | A->B, first use (8192 context tokens) | 9.805604 | 1.390273 | 0.000000 | 8.080613 | 0.331051 |
| qwen3.8 / deepseek | B->A, first use (8192 context tokens) | 7.827469 | 1.863886 | 0.080925 | 5.753390 | 0.124548 |
| qwen3.8 / deepseek | A->B, prepared (8192 context tokens) | 9.758645 | 1.407784 | 0.000000 | 8.093705 | 0.253587 |
| qwen3.8 / deepseek | B->A, prepared (8192 context tokens) | 7.884479 | 1.948308 | 0.080119 | 5.749050 | 0.102522 |
| qwen3.8 / deepseek | A->B, prepared (0 context) | 9.670029 | 1.340663 | 0.000000 | 8.075400 | 0.250241 |
| qwen3.8 / deepseek | B->A, prepared (0 context) | 7.922717 | 1.893920 | 0.000000 | 5.790611 | 0.233760 |
| qwen3.8 / image | A->B, first use (8192 context tokens) | 5.229941 | 1.466799 | 0.000000 | 2.317254 | 1.444621 |
| qwen3.8 / image | B->A, first use (8192 context tokens) | 6.494272 | 0.546692 | 0.073958 | 5.750410 | 0.119529 |
| qwen3.8 / image | A->B, prepared (8192 context tokens) | 5.134153 | 1.344315 | 0.000000 | 2.314791 | 1.474099 |
| qwen3.8 / image | B->A, prepared (8192 context tokens) | 6.487812 | 0.552401 | 0.073631 | 5.757805 | 0.100506 |
| qwen3.8 / image | A->B, prepared (0 context) | 5.087226 | 1.329728 | 0.000000 | 2.316325 | 1.439990 |
| qwen3.8 / image | B->A, prepared (0 context) | 6.511126 | 0.559448 | 0.000000 | 5.779996 | 0.167774 |

Every recorded row has `exact` and `state_exact` true. For 8K LLM returns,
restored state hashes equal the actual saved state and every continued token
and full logit row equals the unswapped control. These are the relevant state
restoration checks; image rows have no LLM state, and zero-context returns
start cleared. Every 8K prepared return keeps and replays graphs. Each
zero-context LLM return's complete fresh-prefill hash also equals its return
through the other partner; this is not a saved-state or 16-token continuation
claim at zero context. B's first-output hashes repeat within each pair.

The one image-A control, shared across both pairs, equals the fixed prior fast-pipeline pixel hash
`3b7770ca720e3ae4bb0633dad36ace14e3c65beed0440dec06bde21c10e29fb7`;
all four regenerated images equal their controls. The expected hash is
fixed before this run, with `--image-expect`; it is not derived from the
new control. Earlier optional raw fast-image reports were absent on this
Spark, so the preflight's independently rechecked raw-report list is empty.
The fixed noise is 4,096 × 64 BF16 values (524,288 bytes), SHA-256
`eb7333893ca0409a100955e5305482bb9de8225ee3c21d33fd01fd6c16c042bd`.
The frozen context file SHA-256 is
`6b159ff20d198a3ff825d78b5edc0c2bd8c9cf5af7222875fb88301635e14db5`.

The report also retains bytes read/spilled, handoff/release counts, release
latencies, setup, planning and demand-read clocks for every row. These are
single-run swap measurements, not confidence intervals or new reference-engine
startup comparisons. This table does not measure full image-generation speed
or a new image-similarity bound against diffusers; it verifies the pinned
native image across swaps. Those prior image controls remain in their
[image report](../qwen-image-native/README.md).

Peak per-row tracked memory is 109.610 GiB. Process-start MemAvailable is
115.518 GiB and its lowest observed value 5.811 GiB: drop 109.707 GiB. These
whole-host and runtime-accounting measures differ; neither is presented as
a GPU allocator comparison. The strong 110-GiB gate passes before and after
the run. Runtime teardown succeeds, the installed GPU-supervised job
`final-token-table-20261004` returns zero and is waited on, with no GPU
compute process or busy/waiting job at handover. Total supervised time is
8 minutes 45 seconds, within its 600-second limit.

The four text artifact identities are unchanged from the standard-client
configuration: DeepSeek `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234`,
DSpark `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5`,
Qwen `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`,
MTP `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
Image composition is `eca21baad38229e471a44cb2479d392ffcf745fb812e8a41668f336139fa1acd`,
with text encoder `ed89ed270cf840965565d42eb61220b07399ff147e4f1de9ffcb7cbfd5ae86cf`,
transformer `d1184efda09c12175b7bae6e9acf6cf3aeafb765f4f2ab6192cbf284c37eb3a4`
and VAE `44c1a20a202ce1078588cbd4b0c594bbadf365a01f26091c715fd16d5b2eb34d`.

Raw records and the pinned reproduction script are external at
`~/scratch/m3-final-token-table-20261004/` on Spark B and
`/home/pmeenan/scratch/llmp-m3-final-token-table-2026-10-04/` locally.
Use `llmp-final-image-table.sh` with a fresh stage directory, its checked
runtime SHA and source tree under the installed GPU supervisor, then wait.
It validates complete row/pair coverage, controls, graph reuse, swap bounds
and final retirement. The local evidence copy excludes runtime spill/state.

| Identity | SHA-256 |
| --- | --- |
| Production runtime | `64fccb395bbeabd2bb7569ec7597f7d9cc6672afb8abb236d4d5deaa7b99b9ab` |
| Production source inventory | `ecdc71d00d4fea3c56c4aaa540ac49b183df852409a62e26330c3843ba54c0e9` |
| Reproduction script | `27201c88d6b939ba8a0a18374b67051826e509b5024d3e39c4dab0dd8961582b` |
| Table report | `4b4bb28256f71848c70626736a3ddcdd2d37eed6de83a05ce5e7735c54c7d71e` |
| Configuration | `0e05feaf5e4a9bf06778d7bd38c4fe9703856cb6a1f99e53466755ebbbd02828` |

An earlier 2026-10-04 refresh at production runtime `cc7706fd` stopped fresh
LLM clocks at prefill heads, before token selection. Its worst LLM endpoint
was 9.823727 s and it passed the same 8K/state/image controls, but does not
qualify the first-generated-token criterion. That raw table is retained at
`~/scratch/m3-final-table-20261004/` on Spark B, SHA-256
`2ea26fbee09e96bcec1d58060a8f5e690e1307df1a62669412bb45014d3039b0`.
The endpoint correction above preserves those head hashes while paying actual
token selection; the timings here come entirely from the corrected run.
