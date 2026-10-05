<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma per-segment RoPE/cache stores

The explicit `Gemma4GraphOptions::rope_store` option supplies eligible
per-segment K rotations to the existing factor-aware fused RoPE/cache-store
primitive. The option remains default off: the first paid complete-layer
screen did not establish a speed gain. This is conditional primitive transfer,
not selected model optimization or whole-model/batched-serving qualification.

## Graph, dispatch and funding contract

The original graph rotates joined learned-normalized K, reshapes it and stores
per-segment slices through ordinary operations. With the option, the learned K
normalization still runs on joined inputs. Each independent segment views its
own normalized K rows and fresh packed I32 positions, then creates a complete
packed RoPE output and zero-offset flattening VIEW for its own cache store.
Q rotation stays joined. Global V still reads raw K before learned K
normalization/rotation and receives its distinct unweighted normalization.

`DeviceChoices::fuse_rope_store` independently enables this exact structural
pattern while generic upstream fusion stays disabled. `PlanGemma4Chunk` derives
that policy from the model's explicit graph option. The existing
`CheckRopeSetRows` checks complete operand geometry, factors, bounds, current
views, addresses and aliases before selection. An ineligible pattern retains
ordinary RoPE and SET_ROWS. No kernel, arithmetic, source-lock or lifetime
contract changes are required.

Ordinary rotation writes F32, then the cache store narrows to F16. The fused
launcher uses the same pinned forward NEOX rotation with a direct F16 cache
writer. Local heads rotate all 256 dimensions; globals rotate all 512 with
256 packed F32 factors. Both implementations retain their original registry
identities and primitive fallbacks.

The optional graph names segment rotations
`blk.<layer>.slot.<slot>.k_rope`; the default graph names its joined rotation
`blk.<layer>.k_rope`. A keep of the segment rotation or a view of its storage
prevents fusion and selects the ordinary producer that writes those bytes.
Current activation placement still funds every rotated intermediate in the
fused plan. This saves the intermediate write and store launch, without
claiming physical allocation elision. Native catalog coverage continues
checking every graph descriptor and source; no coverage exclusion is added.
Fresh source validation, managed cache regions, input staging and completion
ownership remain unchanged.

## Bounded correctness controls

Full CPU graphs cover both verified 26B/31B contracts and 1/2/4/16 independent,
ragged segments, with dispatch counts for every actual layer. The explicit
policy selects exactly one eligible K store per layer/segment. Disabling the
policy, keeping a rotated output or its view, and valid partial flattening
exercise ordinary fallbacks. Measured/placed engine plans retain every rotated
intermediate inside their funded activation region and validate fused operands
after placement. These metadata controls do not substitute for whole-model
catalog execution.

Direct GPU controls use actual 26B/31B local/global KV head counts, joined K
views, ragged rows and 1/2/4/16 segments. Independent segments combine initial
and wrapped positions; a separate bounded cache case reaches absolute position
262,143. Every cache byte is initialized and compared to pinned ordinary
rotation/store, with a host F16 scatter of the actual ordinary F32 output
also checking untouched cells. A captured fused graph receives fresh positions
and indices on the provider stream. Capture ownership survives through proven
completion before caches are released.

Actual-width 26B synthetic complete local/global layers compare outputs and
all initialized K/V bytes exactly for one/four segments, with independent
scalar layer controls and captured replay. A kept segment rotation is read
back and its half-narrowed values match its stored cache cells. These are
structured synthetic weights, not the approved checkpoint.

## First paid complete-layer screen

The short A/B/A uses local layer 0 and global layer 5, one/four independent
one-query segments, positions starting at 1,279 and offset by seven per slot.
A is joined rotation plus ordinary stores; B is per-segment fused stores.
Both arms use device masks, ordinary product preparation, and unfused learned
normalization. Both pay all complete-layer nodes, source staging, attention
scratch/fixup and completion. Host source construction, plan creation and
capture/instantiation happen before warm timing.

Four warmups precede each 32-iteration GPU-event block. Original and captured
A/B/A are separate blocks; captured timing includes fresh input staging plus
replay. Every measured arm's final output is checked against its initial
output byte exactly. Allocations are warm CUDA allocations; VMM residency,
whole-model timing, switching and model quality are not measured here.

The first bookended block, microseconds per complete synthetic layer:

| Layer | Segments × queries | A | B | A after | Captured A | Captured B | Captured A after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local 0 | 1 × 1 | 566.715 | 565.594 | 567.354 | 515.317 | 535.860 | 534.773 |
| Local 0 | 4 × 1 | 757.114 | 786.042 | 784.954 | 742.930 | 729.069 | 715.444 |
| Global 5 | 1 × 1 | 593.529 | 597.762 | 606.746 | 568.210 | 584.023 | 572.063 |
| Global 5 | 4 × 1 | 810.454 | 810.074 | 810.969 | 761.589 | 765.371 | 762.538 |

The uncaptured blocks mostly sit inside their bookends; local four-segment B
slightly exceeds both. Captured global B loses at both shapes; local B lies
inside shifting bookends. This first screen does not justify selecting the
policy or expanding to a whole-model performance ladder.

| Layer | Segments × queries | Activation bytes A = B | Planned scratch A = B | Observed pool peak A = B |
| --- | --- | ---: | ---: | ---: |
| Local 0 | 1 × 1 | 208384 | 82688 | 82560 |
| Local 0 | 4 × 1 | 782848 | 82688 | 82560 |
| Global 5 | 1 × 1 | 235520 | 1320960 | 1320960 |
| Global 5 | 4 × 1 | 861440 | 1585152 | 1585152 |

The final source/binary verification block repeated the same short screen:

| Layer | Segments × queries | A | B | A after | Captured A | Captured B | Captured A after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local 0 | 1 × 1 | 562.300 | 565.208 | 531.995 | 531.097 | 531.481 | 538.457 |
| Local 0 | 4 × 1 | 751.899 | 791.705 | 757.250 | 720.283 | 736.788 | 708.983 |
| Global 5 | 1 × 1 | 595.454 | 620.479 | 623.265 | 589.656 | 569.171 | 590.901 |
| Global 5 | 4 × 1 | 796.729 | 812.474 | 827.175 | 740.006 | 765.014 | 737.081 |

Final captured four-segment B loses against both bookends. Global solo capture
improves in this final block but lost in the first screen; local solo lies
inside its bookends. The inconsistent solo outcome and batched loss retain
the default-off decision. No additional performance ladder was run.

The option retains the same host inputs and planned scratch. Observed pool
peak differs from the conservative planner bound in local cases; no
intermediate allocation saving is claimed.

## Reference provenance

At task entry on 2026-10-04, TensorFold default HEAD was refreshed to
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its
[pinned README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
lists Gemma through MLX; it supplies no Gemma CUDA comparison arm for this
bounded Spark task. The existing pinned llama.cpp reference remains the
whole-model CUDA quality/performance comparator, with qualification separate
under [reference policy](../../reference-comparisons.md).

Measured on `spark` GB10, driver 580.178.04, pinned SDK
`aarch64-c09daba6ac31edee` (Clang 22.1.8, NVCC 13.4.92), `spark-native`
`sm_121a`, from base `891ceea`. The unchanged prepared GGML tree is
`1ae467d0fced412beb16faca3d0a910441477a06f1e2bae4ccbe0cdb6230bd11`.
Installed supervised `gemma-store-build7` and three-step `gemma-store-final1`
retired successfully. The final focused set passes 21 CPU and three GPU tests
with no skips. The first paid screen was `gemma-store-screen3`, also rc0.
Both paid blocks use identical production bytes; the final block also covers
the final diagnostic-keep and direct-cache fixture bytes.

The SHA-256 over sorted changed `src/` and `tests/` paths, each encoded as
`path + NUL + bytes + NUL`, is
`3ab2f9e25806fb00f3706d208a645c18563c67c2f3c5230d9ac9310ac41e193c`.
Final measured `gemma4_exec_test` SHA-256 is
`d46cd07fa84ace354c5f6eb3b21f3453549a340dc2a5fc99bcc4d4298e8f4919`.
Raw logs and samples remain outside Git.
