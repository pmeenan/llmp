<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 C4 original dispatch observation

The untouched reference selects all 30 routing fusions during four-owner decode,
but only layers 0–27 during each 64/65/66/67-row prefill. Layers 28 and 29 pass
structural and shape checks and fail the physical-memory gate. All 30 scaled
ordered reductions are selected in every evaluated call. These are observed
host decisions for the exact [packed26 recipe](../gemma26-packed-attention-c4/README.md),
whose native policies enable both norm chains and leave routing/reduction off.
They identify another arithmetic difference without attributing the remaining
68/128 positive-margin choices to one operation or treating mixed eligibility as
a bug.

All 128 observed full vocabulary heads are byte-identical to both untouched
stock bookends: 134,217,728 bytes, SHA
`5808306e0f261896500bdf9f3b44d4b1d0f18b34ba99d976da15a916aa8ff164`.
The actual 1,024 input IDs also match. No graph readers, allocation hints,
fusion predicates or floating math launch arguments were changed.

| Phase | Evaluated host calls | Routing selected | Routing memory refusals | Scaled reduction selected |
| --- | ---: | ---: | --- | ---: |
| Decode, 4 rows | 4 | 120/120 | None | 120/120 |
| Prefill, 64 rows | 2 | 56/60 | Layers 28 and 29 in both calls | 60/60 |
| Prefill, 65 rows | 2 | 56/60 | Layers 28 and 29 in both calls | 60/60 |
| Prefill, 66 rows | 2 | 56/60 | Layers 28 and 29 in both calls | 60/60 |
| Prefill, 67 rows | 2 | 56/60 | Layers 28 and 29 in both calls | 60/60 |

The controller logs 51 graph-call headers: 12 calls with evaluated decisions and
39 without operation records. The analyzer requires every one of the 12 calls
to cover all 30 routing candidates and reduction gates, and reconciles admitted
gates with selected outcomes. These counts are host build/evaluation
observations, not CUDA kernel or replay counts; traced elapsed time is not a
competitive performance result.

Actual routed gate/up products use Q4_K `[2816,1408,128,1]` weights: MMVQ at
four decode rows, MMQ at every prefill width. Routed down products use Q5_1
`[704,2816,128,1]` weights in layers 0–28 and Q8_0 with the same dimensions in
layer 29, with the same MMVQ/MMQ phase split. No fused product branch is observed.
The ordinary Q8_0 products similarly switch from MMVQ to MMQ during prefill,
except the one-row vocabulary head remains MMVQ. F32 router products use MMF
at decode and cuBLAS at prefill. [Aggregate results](results.json) retain counts
by family, weight type and phase; raw tensor addresses and graph records stay
external.

The prefill refusal is not the historical 128-row, layer-28 observation. This
recipe has two refused layers at four different widths. Since the prefill rows
exceed the topk alias allowance of eight, its memory gate can differ from the
four-row decode gate. Prefill arithmetic may also affect later KV state. A
uniform native routing/reduction policy therefore cannot be assumed to reproduce
this mixed reference history. A further compound-policy or identical-input block
probe needs its own reviewed scope; no layer whitelist, alias imitation,
production selection or quality exception follows from this observation.

The measured base is `2eb0574`, source pin
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, and the digest-pinned image is
specified in [the protocol](PROTOCOL.md). The external CUDA 13.4 controller
rebuild includes its unchanged integer pointer-preparation kernel. Floating
model kernels remain in the original image's CUDA 13.3 library. Removing marked
logger insertions reconstructs the original controller bytes exactly. Original
client `01889d8c…` is unchanged and explicitly frees its batch, context, model
and public backend after completed publication. Official process and supervisor
retirement succeeded. [Provenance](provenance.json) separates the six-source
pre-acquisition receipt from later parser-only corrections and the final
per-call coverage check. Both early parser failures and a malformed metadata
queue are retained externally; the single measured acquisition succeeded.

For reproduction, supply the approved GGUF and the external 4,096-byte input
file named and hashed in the protocol. Retain the exact pinned original client
and headers from the preceding packed26 experiment. On a fresh scratch path,
checksum-sync this checkout to Spark A, then run `reference.sh build`, followed
by `PYTHONDONTWRITEBYTECODE=1 python3 test_controls.py <original-ggml-cuda.cu>`
and `PYTHONDONTWRITEBYTECODE=1 python3 prepare.py`, through the installed GPU
supervisor. Freeze that receipt before running `reference.sh run`. Run
`PYTHONDONTWRITEBYTECODE=1 python3 analyze.py <official-run-log> <observed-dir>
<untouched-stock-A-dir> <untouched-stock-B-dir> <new-aggregate.json>` after
successful retirement. Python bytecode is disabled because the admission
inventory rejects unexpected directories. Existing scratch paths and receipts
are exclusive and must not be overwritten.

The narrow Spark build and original acquisition passed; all 11 final source/parser
controls and the metadata-only per-call coverage check passed. No native policy
changed and no model unit suite, timing gate, representative PPL or support
qualification is claimed.

The subsequent [packed compound screen](../gemma26-compound-packed-c4/README.md)
tests native routing/reduction throughout prefill and decode. It reaches 92/128
byte-exact heads but retains two positive-margin disagreements and 7.15% higher
latency. Stock's late prefill refusals remain a difference to probe with common
inputs; no layer whitelist or production policy is selected.
