<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma device causal and ring masks

The optional `Gemma4GraphOptions::device_masks` path produces each segment's
F16 global/local attention masks from fresh packed I32 positions. Masks are
charged activations reused across layers. The funded host-reference-mask path
remains available. This bounded slice does not qualify whole-model quality,
optimized serving batches, paging or checkpoint support.

## Primitive and source contract

The registered `jitllm.gemma4.mask` primitive is a native `kFill` operation,
with zero scratch. One thread writes one output element, either half zero or
negative infinity (`0xfc00`). Every padded query row is written and remains
negative infinity; no position is read for a padded query. Cells are padded
to 256 and queries to 32, including 1,025-to-1,056 attention prepass padding.
Negative or context-exceeding position values yield an entirely masked row.

Global cell visibility is `cell <= query`. For a local ring of capacity C,
visibility is `distance=(query%C+C-cell)%C`, `distance<window` and
`distance<=query`. Arithmetic uses widened intermediates. Checked capacity
retains `min(context, window+whole_chunk_rows)` cells. An independent retained
absolute-position map after the complete chunk validates that later writes
cannot replace a cell visible to an earlier query.

Host admission checks types, packed strides, current owner generations,
disjoint/aligned input/output, position range within its packed source,
capacity/window/context and the entire padded output before GPU submission.
The conservative output byte bound is signed-I32 maximum; the rounded launch
grid follows that bound. I32 parameter endpoints and public I64 descriptor
endpoints have metadata-only refusal controls.

Graph mode creates one global and one local producer per segment and excludes
those tensors from host inputs. `Gemma4Sources` verifies exact producer
identity, source and parameters before accepting absent host mask arrays.
Fresh tokens, positions, cache-cell indices, shapes and host funding remain
checked. Diagnostic host masks still have every logical bit validated and all
query padding filled. Both modes share the same attention/cache tensors.

## Bounded correctness controls

CPU controls independently compare 28,692,480 cell/query cases across initial, partial, wrapped and 262K-depth
rings for both approved profiles and whole chunks of 1/2/4/16/1,025 queries and
capacity including every chunk write. Graph/source controls cover 1/2/4/16
independent segments, bounded host storage, producer substitution, invalid
fresh positions/cache cells/tokens and shape reuse across a ring wrap.

GPU controls compare every F16 byte to an independent retained-cell oracle for
both profiles and 1/2/4/16 segments, with initialized output prefix/suffix
canaries. Captured replay uses freshly staged positions on the same provider
stream, including invalid position values. A bad-type runtime refusal queues
no mask kernel, preserves output and leaves the launch context usable.
The direct primitive draws zero scratch. Complete actual-width synthetic
local/global layers compare host/device masks byte exactly, with independent
scalar layer and captured state controls. Synthetic weights are not the
approved model weights.

## Paid complete-layer comparison

The short screen uses actual 26B widths/types, local layer 0 and global layer
5, one and four independent one-query slots at position 1,279. A is the funded
host-mask graph; B is the device-mask graph. Both arms pay both global and
local mask outputs even when one diagnostic layer uses only one kind. Device
producers are explicitly expanded in partial graphs; savings are not attributed
to removing unused global-mask staging.

Source validation and host mask construction happen before these warm GPU-event
blocks; their CPU time is not included. The source-allocation table below
counts the owned padded mask buffers and pointer/frontier containers, not
caller-owned hidden/token/position/cell buffers. Staged bytes count every
input actually copied, including those external buffers.

Each arm pays input staging on the provider stream, every complete-layer node,
attention scratch/fixup and completion. Four warmups precede 32 measured
iterations. Original A/B/A and captured A/B/A are separate short blocks.
Captured timing pays fresh input staging plus replay; capture/instantiation
happens outside timing. Every final measured output is compared byte exactly
to its initial output. These are synthetic warm CUDA allocations, not VMM
residency or complete-model timing.

The final bookended block (microseconds per complete synthetic layer):

| Layer | Slots × queries | Ordinary A | Device B | Ordinary A after | Captured A | Captured B | Captured A after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local 0 | 1 × 1 | 548.896 | 563.067 | 554.24 | 546.904 | 535.582 | 548.922 |
| Local 0 | 4 × 1 | 825.872 | 798.062 | 829.972 | 764.603 | 732.532 | 788.071 |
| Global 5 | 1 × 1 | 620.704 | 605.657 | 619.232 | 568.826 | 554.29 | 592.76 |
| Global 5 | 4 × 1 | 847.381 | 818.009 | 849.62 | 810.011 | 733.335 | 809.869 |
| Local 0 | 1 × 33 | 928.256 | 917.728 | 932.672 | 849.408 | 835.392 | 825.376 |
| Global 5 | 1 × 33 | 1045.09 | 1028.45 | 1021.92 | 951.232 | 932.857 | 949.47 |

A prior short screen had captured solo improvement, while the second screen's
solo/chunk results were mixed. The final four-slot captured blocks improve at
these synthetic shapes; the short solo/33-query blocks do not establish a
stable speed advantage. The option remains explicit and default false.
The measured removal of required host matrix allocation/staging is useful
independently of a universal kernel-speed result. Runner/whole-model selection
still requires its own matched reference gates.

| Layer | Slots × queries | Owned host source bytes A / B | Staged bytes A / B | Activation bytes A / B | Planned scratch A = B | Observed pool peak A = B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Local 0 | 1 × 1 | 163984 / 64 | 175872 / 12032 | 208384 / 208384 | 82688 | 82560 |
| Local 0 | 4 × 1 | 704992 / 160 | 751872 / 47360 | 811008 / 782848 | 82688 | 82560 |
| Global 5 | 1 × 1 | 163984 / 64 | 175872 / 12032 | 235520 / 235520 | 1320960 | 1320960 |
| Global 5 | 4 × 1 | 704992 / 160 | 751872 / 47360 | 861440 / 861440 | 1585152 | 1585152 |
| Local 0 | 1 × 33 | 360592 / 64 | 733440 / 372992 | 6862336 / 6862336 | 6389760 | 6389760 |
| Global 5 | 1 × 33 | 360592 / 64 | 733440 / 372992 | 7403008 / 7403008 | 6340608 | 6340608 |

Mask production itself uses no pool scratch. Total paid layer scratch follows
the unchanged products/attention; it is planned before binding/capture. Device
mask activation storage can be reused after its last consumer. The local
four-slot activation reduction reflects that lifetime placement rather than
omission of a mask producer. Both arms compute/stage both mask kinds.

## Measured-source and check provenance

Measured on `spark` GB10, driver 580.178.04, pinned SDK
`aarch64-c09daba6ac31edee` (Clang 22.1.8, NVCC 13.4.92), `spark-native`
`sm_121a`, from base `b657317`. The unchanged prepared GGML tree is
`1ae467d0fced412beb16faca3d0a910441477a06f1e2bae4ccbe0cdb6230bd11`.
Installed supervised `gemma-mask-build7` and six-step `gemma-mask-final1`
retired successfully. The final focused set passes 20 CPU and three GPU tests
with no skips. The CPU ring control is counted once; graph/source tests also
retain all diagnostic mask controls.

The SHA-256 over sorted changed `src/` and `tests/` paths, each encoded as
`path + NUL + bytes + NUL`, is
`1cc1860feb93fb4386ea25cc68e85809a33120bcb014a91a7c31ba32b87abea1`.
Final measured `gemma4_exec_test` SHA-256 is
`729300bf17e957c2f3d8041c97d6a1a1efcf1ddccd3fc3efbaec93c72069e5d4`;
`ggml_ext_ops_test` is
`e2eaba52649b10bb33030b8ff3e43de5e928b9418529ae76790862792545dcf7`.
Raw logs and samples stay outside Git.

## Reference provenance

At task entry on 2026-10-04, TensorFold default HEAD was independently refreshed
to `609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its [pinned README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
still lists Gemma 4 as MLX-only, so it supplies no CUDA GB10 comparator
for this task. The existing pinned llama.cpp arithmetic remains the applicable
CUDA reference. No TensorFold or whole-model performance claim is made.
