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
A new consumer still needs its own model gain. The distinct producer-fused
down-input Q8 experiment below also remains unadopted.

## Producer-fused down-input Q8

The earlier [prepared-Q8 prototype](../qwen38-mtp-speed/README.md) adds
96 preparation launches per verify: gate/up input and down input in each
of 48 layers. This separate experiment keeps gate/up quantization inline
and exports the exact down-input Q8 from the existing GLU producer. It
adds no preparation launch. The complete GLU-plus-down pair improves only
2.4–2.9% in speed ratio for overlapping routes and is slightly slower for
disjoint routes. That small isolated result does not justify production
fusion or a model trial; production remains unchanged and the MTP gate open.

The external kernels keep eight rows per warp and eight warps per CTA.
GLU retains its original Q8 input conversion, dots, block-scale FMA,
XOR reduction, scales, F32 output and PDL positions. Each CTA produces
32 GLU outputs. After one CTA barrier, two threads quantize its two
16-value groups in the original ordered amax/127, round and code-packing
sequence. Tail warps participate in the barrier; invalid expert IDs exit
uniformly after canonical NaN F32 writes. Their unused payload is defined
zero, rather than claimed to equal quantizing NaNs.

Prepared down reads each lane's original block sequence and retains its
integer-dot, FMA and reduction order. Every Q8 read precedes its original
PDL release. The explicit payload record holds four code words, an F32
scale and zero padding in 32 bytes, aligned to 16. At three/four/five rows
and ten selected slots the transient payload is 38,400 / 51,200 / 64,000
bytes. These buffers are allocated before graph capture; allocation wall
time and capture time are reported separately from per-pair GPU time.
No production graph scratch, allocator or snapshot contract is changed.

`m3-qwen-glu-down-q8-micro2`, 06:22:46–06:23:40 EDT, uses the same
Spark, SDK, artifact offsets and unchanged production kernel as above.
Forty controls cover rows 1–8, up to 64 selected slots, duplicates, invalid
IDs, zero-amax GLU groups, padded input/ID/slab strides, short K and down
tails. Twenty-four use the native launcher; 16 padded/half-CTA controls
use the copied original kernel. All stored F32 GLU values, every Q8
record byte and final down outputs match exactly, including independent
repeats, finite/canonical-NaN checks, guard bytes and malformed-stride
refusals. Separate zero gate scales exercise the exported zero-amax path.

The six timed actual-shape pairs use physical backing for 512 experts,
ten routed slots and all 64 route banks. Their accessed gate/up-plus-down
weight sets are 1,291,161,600 / 1,415,577,600 bytes (overlap / disjoint),
above measured 24 MiB L2. Each result is the median of three CUDA-event
batches of 512 complete pairs. Original controls bracket the candidate;
each captured graph has exactly 128 kernel nodes. Post-capture controls
compare its final bank with an eager pair, including F32 outputs, every
Q8 byte and guards. Pool-offset reuse and production planning are not
tested because integration is rejected.

Mean before/after original time and fused-pair time, in microseconds:

| Token rows | Route | Original pair | Fused pair | Speed ratio |
| ---: | --- | ---: | ---: | ---: |
| 3 | Overlap | 198.257 | 192.654 | 1.029 |
| 3 | Disjoint | 380.279 | 382.714 | 0.994 |
| 4 | Overlap | 232.043 | 226.634 | 1.024 |
| 4 | Disjoint | 503.209 | 507.007 | 0.993 |
| 5 | Overlap | 265.526 | 258.112 | 1.029 |
| 5 | Disjoint | 625.474 | 631.835 | 0.990 |

Overlap latency falls 2.33–2.83%; disjoint latency rises 0.64–1.02%.
Bracketed baseline drift is −1.44% to −0.19%. Full fixture allocation
wall time is 64.3–65.5 ms, including weights, references and guards;
128-node candidate capture plus instantiation is 0.270–0.279 ms.
Neither is presented as per-step production allocation overhead.
Ptxas reports GLU 98 registers, one barrier and 128 bytes of shared memory
(original: 106 registers), and prepared down 78 registers (original: 98),
with zero stack or spills. Lower registers do not establish a model gain.

Raw records, formatted sources, build flags, identities and scripts stay on
B under `~/scratch/m3-qwen-glu-down-q8/`. The original production unit is
unchanged at `7b8c71122d032c4f05ee97c9b7b90769cec15541f60bd17018b7299dab82e291`.
The external host passes SDK format and clang-tidy; device units compile
with the original production flags, including `-use_fast_math` and PDL.
The initial job stopped at host style checks before controls; the corrected
job above completes successfully. SHA-256 identities:

- producer/prepared-down kernel `7702c7c4d94126ef32b3b3e63e718a44e469cb4602b2933a7704da1c08ccf06b`;
- host fixture `51d3a9989645108aa0aba89d6ea4769d75bc1821ef1efaa896759062c19ec42b`;
- external diagnostic binary `096c4b275df8fbdcd05ccf0d72700236aa5b2003c778c906aaa048d7cae52279`.

Exact producer export, intermediate/payload controls and complete-pair
capture accounting are reusable pieces for a different eligible consumer.
The current small pair gain leaves production planner/scratch integration,
model quality, rollback and swap evidence unclaimed. No further variant is
selected from this microbenchmark alone.
