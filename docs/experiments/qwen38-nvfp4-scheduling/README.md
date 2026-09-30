<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 NVFP4 expert scheduling — 2026-09-30

Smaller CTAs improve isolated routed-expert products, but the bounded
three/four-row transfer is neutral with the default prefix head and slightly
slower with the curated head at 128K. The production kernel and its tests
remain unchanged. The external schedules, exact controls and failed transfer
are retained for reuse; the long-context MTP speed gate remains open.

## Scope and arithmetic

The [MTP profile](../qwen38-mtp-speed/README.md) attributes 35.4% of
verification kernel time to NVFP4 experts. Unlike the earlier expert-sharing
and prepared-Q8 experiments, this sweep changes only rows per warp and warps
per CTA: rows 2/4/8 and warps 4/8. Every row retains its original lane's
16-value K blocks, amax/127 Q8 conversion, round/packing order, integer dots,
block-scale FMA and XOR reduction offsets 16 through 1. Gate/up pairing,
F32 SwiGLU output, routing scales and PDL locations are retained. The copied
and production CUDA units use the same SDK flags, including `-use_fast_math`.

Runs use `spark-b` / `spark-56f5`, GB10, driver 580.178.04 and SDK
`aarch64-e0a0c85c42806fb1`, with main `d753aa0` as the production base.
Artifact verification checks all 48 layers: 512 experts, ten selected slots,
hidden width 2,560, expert FFN 640, packed stride 2,764,800 bytes. Gate/up
codes and scales start at 0 / 1,638,400; down codes and scales at
1,843,200 / 2,662,400. The external fixture includes finite generated weights,
duplicate and invalid IDs, zero-amax inputs, padded strides and tails.

## Isolated schedules

`m3-qwen-nvfp4-sweep3`, 05:04:32–05:05:46 EDT, runs 40 correctness
cases across all six schedules: token rows 1–8, separate gate/up and fused
GLU/down forms, up to 64 selected slots, invalid IDs, short K, output tails
and padded input/slab strides. Thirty-three cases use the actual native
launcher as their reference; seven padded/odd-GLU cases use the copied
original eight-row/eight-warp kernel because the production validator
requires packed inputs. Every route bank repeats and matches its reference
bit for bit, including canonical invalid-ID NaNs; finite-output and guard
checks pass.

The same job times GLU/down at three/four/five rows with overlapping and
disjoint routes. Each graph visits all 64 route banks. The actually accessed
weight sets are 860.77 / 943.72 MB for GLU and 430.39 / 471.86 MB for down
(overlap / disjoint), beyond measured 24 MiB L2; unused slab bytes are not
counted. Each result is the median of three CUDA-event batches of 512 calls,
with original production controls before and after. All 72 timed candidate
comparisons also repeat and match native outputs exactly.

Mean before/after original time and eight-row/four-warp time, in microseconds:

| Form | Token rows | Route | Original | Four warps | Speed ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| Down | 3 | Overlap | 77.226 | 69.312 | 1.114 |
| Down | 3 | Disjoint | 85.009 | 79.936 | 1.063 |
| Down | 4 | Overlap | 93.232 | 83.620 | 1.115 |
| Down | 4 | Disjoint | 117.895 | 114.733 | 1.028 |
| Down | 5 | Overlap | 109.492 | 97.725 | 1.120 |
| Down | 5 | Disjoint | 200.100 | 201.301 | 0.994 |
| GLU | 3 | Overlap | 119.961 | 115.788 | 1.036 |
| GLU | 3 | Disjoint | 223.604 | 220.428 | 1.014 |
| GLU | 4 | Overlap | 137.862 | 133.263 | 1.035 |
| GLU | 4 | Disjoint | 330.070 | 325.476 | 1.014 |
| GLU | 5 | Overlap | 155.701 | 149.935 | 1.038 |
| GLU | 5 | Disjoint | 411.121 | 406.048 | 1.012 |

Reducing rows per warp lowers registers further, but two-row overlap cases
are slower. The model trial retains eight rows to isolate CTA grouping.
Down uses 98 registers originally, 96 at rows8/warps4, 64 at rows4
and 48 at rows2. GLU uses 106 originally, 105 at rows8/warps4, 70–72 at
rows4 and 51 at rows2. No schedule reports stack or local spills. These
resource counts are not a speed claim.

## Adaptive model comparison

The candidate changes only GB10 products with three/four token rows,
512 experts / ten selected slots, the exact packed layout above, and actual
GLU 640×2,560 / down 2,560×640 shapes. Eight rows per warp and every
per-slot arithmetic operation are retained; one-row, five-row and other
shapes/devices use the original eight-warps schedule. A physical 512-expert
test passes exact one-row/batched and repeat comparisons for rows 2/3/4/5/8,
duplicate expert 511 and positive/negative invalid IDs.

`m3-qwen-nvfp4-adaptive-model`, 05:22:21–05:30:19 EDT, runs six fresh
processes: candidate, original, candidate repeat for each head. Each uses
the canonical phase-1 128K prompt, 512 outputs, adaptive depth 2–3 and
runtime-style 4,096-row prefill. Rendered prompt length is 128,799, stable
generation boundary 128,794; user-content SHA-256 is
`b7154f4bdfea17ea92e20ddaa95b2ece37ea306972af7039894fd0e81854da1d`.
Every load passes the exclusive process/container and ≥105 GiB memory gate.

| Head | Variant | tok/s | Draft ms/step | Verify ms/step | Full ms/step |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefix 65,536 | Original | 45.388 | 8.526 | 53.957 | 62.547 |
| Prefix 65,536 | Candidate | 45.408 | 8.511 | 53.956 | 62.520 |
| Prefix 65,536 | Fresh candidate repeat | 45.400 | 8.494 | 53.975 | 62.530 |
| Curated 47,172 | Original | 44.035 | 7.056 | 52.720 | 59.817 |
| Curated 47,172 | Candidate | 43.623 | 7.185 | 53.143 | 60.382 |
| Curated 47,172 | Fresh candidate repeat | 43.669 | 7.149 | 53.120 | 60.318 |

Prefix gains are 0.044% / 0.026%; curated loses 0.936% / 0.831%.
Every triple has identical prompt IDs, all 512 tokens, independently chosen
depth traces and acceptance counts. Prefix uses 56 depth-two / 124 depth-three
steps, including a final partial step; curated uses 113 / 81. The complete
three/four-row verify counts are 55 / 124 and 111 / 81, respectively.
Target graphs replay 164 / 171 times with zero refusals and 13 / 17 drops,
identical within each head's triple. Peak MemAvailable drops span
89,016,877,056–89,294,921,728 bytes.

`m3-qwen-nvfp4-branch-proof`, 05:30:49–05:31:04 EDT, profiles the unchanged
candidate binary during one short 64-output adaptive generation. It confirms
the real model selects both four-warp GLU and down kernels: 924 launches each
at gridY30 (three rows) and 195 each at gridY40 (four rows), all 128-thread
CTAs. One/two-row fallback launches remain 256-thread. This proves the branch
is active; its instrumented timing is not an end-to-end performance result.

The model result rejects the isolated transfer. It does not identify a
bandwidth or occupancy bottleneck: this experiment records neither actual
DRAM transactions nor clocks. Requested weight bytes and register counts
cannot supply that missing evidence. The production kernel and test changes
are removed; their exact patch and source/binary identities stay external.
No new PPL, sampled-distribution or rollback/swap claim is made for this
unadopted code. Mia's final 128K MTP comparator remains 48.652 tok/s.

## Provenance and reusable pieces

Target artifact `c4fb47a9…`, prefix head `056a750e…` and curated head
`8600a998…` are unchanged. The curated artifact uses the externally supplied
Mia 47,172-ID list, not independently generated IDs; its complete pins,
source and license record remain in the
[draft-head study](../qwen38-draft-head/README.md#provenance-and-reproduction).

Raw records, external host/kernel/header, compile/link commands, route-bank
summaries, nsys capture and archived production/test patch are on B under
`~/scratch/m3-qwen-nvfp4-scheduling/`. SHA-256 identities:

- external kernel `68ed43bf2fd7b682deb2ba6ae4d8db8d83cdf2bf5024d617a0e32a345c5d3a3c`;
- external diagnostic binary `1d122bddb105777e466d1cb4dd21a4befb9f84a914fde5aeacb7fe524a682135`;
- original model binary `821b1842a0e2bbce034e9b6a510a4cb2f3f6a973e5b8dd981251231eebb699fa`;
- candidate kernel `2ee18d74ca401cf6f68ce219a6589952d23df2e0c8bbcd18fdf07b04d23bf67a`;
- candidate model binary `03acdfeb7673ece2369f58f57017bed1922d1318da09da4ffd43aa341376544c`.

The six-schedule fixture, exact per-slot arithmetic, physical packed-slab
controls and actual route-bank working-set accounting are reusable pieces.
A new consumer still needs its own model gain. The distinct next candidate
is fusing exact down-input Q8 preparation into the GLU producer, avoiding
the additional preparation launches that defeated the earlier prototype.
It remains unimplemented here and must retain F32 GLU outputs, original
conversion/block order, PDL readiness and completion-aware scratch ownership.
