<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ordinary two-column MMVQ preparation sharing

The private synthetic dense31 C2 operator preserves every gate/up, GeGLU and
Q6_K down output byte in eager execution and changed/restored-input capture.
It shares only original Q8_1 input preparation and retains both d812 original
**two-column** MMVQ consumers. The captured chain has six nodes versus seven.
The short complete-chain A/B/B/A mean is 1.0361% faster, with overlapping arm
ranges. This establishes the bounded operator identity; model benefit and
production adoption remain unqualified.

| Arm | 128 captured complete chains, ms |
| --- | ---: |
| Ordinary first | 146.543746948 |
| Shared first | 143.083450317 |
| Shared repeat | 144.399230957 |
| Ordinary bookend | 143.948699951 |

Ordinary/shared means are 145.246223450/143.741340637 ms, or
1.134736121/1.122979224ms per chain. Spreads are 2.595046997/1.315780640 ms.
Timing includes input preparation, both products, separate GeGLU and separate
down preparation/product. Setup, copies, validation, full-output reads and
payload authentication are outside the timed graphs. This is a short synthetic
operator screen, not a sustained or end-to-end model measurement. Stock also
prepares separate gate/up inputs twice; no residual model cause is established.

[The private launcher](../../../benchmarks/gemma31_mmvq_pair.cu) requires
GB10 and exact packed F32 input [5376,2], separate Q4_K weights [5376,21504] and
independent F32 outputs. Both original plans must choose MMVQ, with supported
source precision and authenticated readable tails. All pair operands are
mutually disjoint, including 144 readable weight-tail bytes. No registry,
runner, serving option, kernel body or default changed. Broad shared VecQ and
one-token consumers are excluded.

[The complete-chain harness](../../../benchmarks/gemma31_mmvq_pair_bench.cc)
uses current GGML quantizers to generate 64 distinct deterministic synthetic
rows per bank and repeats them at the approved layer-zero geometry. It proves
all 559,104 output bytes finite/exact, eight fresh poisoned output/workspace
capture replays and nine atomic refusals before timing. Inputs, complete
weights/readable zero tails and output trailing guards remain unchanged.
The shared pair draws 12,672 workspace bytes; separately prepared Q6_K down
keeps the whole-chain peak 48,384 bytes. Device payload 225,541,604 bytes fits
the 256 MiB caller ceiling. Known host payload allowance is 384 MiB; CUDA
context/graph overhead and total physical peak are unmeasured. Descriptors,
operands and workspace remain alive through graph destruction and observed,
released provider completion fences. This direct operator proof does not
qualify production activation placement, catalog admission or captured model
lifetime.

The approved dense31 fixture has 54 Q4_K and six Q5_K gate/up pairs; down types
vary across 26 Q6_K, six Q5_K and 28 Q4_K layers. Only the Q4_K pair/Q6_K-down
representative was tested. Gemma26 has 30 separate shared Q8_0 gate/up products
[2816,2112] and Q8_0 down [2112,2816], so this preparation-sharing technique may
transfer under independent exact and paid proof after a justified 31B decision.
Its routed merged Q4_K gate/up [2816,1408,128] already uses one product and cannot
save duplicate preparation this way. No 26B, other-format, other-column,
prefill or model execution follows from this screen.

Source is 87858b5 plus the four private benchmark paths, against locked
llama.cpp v0.6.0/d81235049384534c167caea52b85a694f6103d14 and patched GGML
d50cb7f9. The launcher derives the original dense preparation/stride mapping
and prequantized-consumer calls from upstream mmvq.cu under MIT; its own guards
and caller are Apache-2.0. The approved GGUF fixtures provide type/shape facts,
not these synthetic numerical operands. Spark A completed the six-step
single-target build and three-step operator queue successfully; no failed or
skipped step, model, reference reacquisition or full suite was run.
[Aggregate identities and statuses](results.json) retain both official jobs,
binary/build-receipt provenance and external raw-result hashes. Raw logs and
synthetic operand bytes remain outside Git.

The fresh task-entry TensorFold primary receipt fixes
cb2ebf0540f42604e2759b2ddef497861e928248/version 0.6.6 at
2026-10-06T19:17:44.666491Z. Its observed Gemma26 MLX/DFlash recipe supplies no
matching dense31 CUDA/GGUF comparator; no TensorFold workload ran.
