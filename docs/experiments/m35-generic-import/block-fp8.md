<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# HF block FP8 ingress

The generic HF reader accepts FineGrainedFP8 weights described by
`quant_method: fp8` and positive `weight_block_size: [rows, cols]`.
`M.weight` retains its E4M3FN bytes and `M.weight_scale_inv` retains its F32
multiplicative scale grid. The new `fp8_scaled` split format describes this
geometry explicitly; plain unscaled FP8 remains a dense encoding. Source
names, physical dimensions and payload bytes remain authenticated separately
from the canonical role names in the published artifact.

Scale semantics were checked against Transformers commit
`1a55addd658aaa95b52871f5b72e91a6b91449b6`:
`integrations/finegrained_fp8.py` SHA-256
`106e2f4b36ffd31d50b7b40d67a603ff1f1dbcc8b6b3167377d3d3da9b3b91cf`,
and `utils/quantization_config.py` SHA-256
`94a6b8d9385564ea6bc847b6128596c486131fc439a336b5cbc71d10654cf371`.
The dequantizer multiplies codes by the stored inverse-scale values; their
name does not imply another reciprocal. Our format also describes partial
edge blocks, verified independently below; this does not claim upstream
execution supports those edges.

The [format specification](../../artifact-format.md) defines the checked grid,
including independent stacked projections and expert matrices. CPU decoding
handles partial row ranges and matrix selection. Shared row scales cannot be
mistaken for independently addressable per-row bytes. Indexed gathering retains
whole matrix scale planes, and reshapes cannot move scale boundaries merely
because their byte counts match. No GPU implementation, numerical evidence or
model qualification is added by this slice.

## Verification

On Spark A, nine focused cases passed in the normal native build and again in
a fresh AArch64 build with CUDA disabled. They cover format geometry, edge
blocks and overflow; strict format JSON; graph identity; independent CPU decoded
values across projection/matrix boundaries; malformed HF part pairing;
normalization and complete indexed byte replay; expert gathering; unsafe
equal-byte reshapes; and private native publication with deep verification,
stored-spec reconstruction and component validation. Both runs had no skips.
The final shared non-model regression pass completed all 3,674 registrations
in 546.64 seconds with zero CTest failures or skips; the nine already-passing
focused registrations were excluded from that sweep. The shared JUnit truncates
53 unit-test outputs and its full LastTest log was not retained before later
model checks, so complete inner-case counters are not claimed. The supervisor
log and all registration results remain retained.

The first consumer build exposed oversized format descriptors; field placement
was repaired and the final native build enforced the unchanged ABI size guards.
AArch64 `TensorFormat` is 72 bytes, `OperandFormat` 120 and `TensorDesc` 160.
Placing existing fields in padding accommodates the explicit block geometry
without weakening those guards. The fresh CUDA-disabled build uses the same
layout.

The independent whole-source and adversarial reviews found an unsafe singleton
reshape and a publication fixture that confused original source names with
canonical stored roles. Both were repaired. The first native compile then
caught two test diagnostics using an import error formatter on string errors;
the first focused run retained six passes and three failures. Those failures
exposed a malformed-input fixture passing HF config in the wrong argument and
a missing inverse-scale companion suffix in spec synthesis. The generic suffix
binding was repaired and independently reviewed, and both focused runs above
used the corrected source. Every original failure remains in external logs.

The format decision is [D-127](../../decisions.md#d-127-explicit-block-geometry-for-scaled-fp8-artifacts-2026-10-23-status-accepted-extends-d-113).
The behavior delta is based on `8c692b12`; its integrated patch SHA-256 is
`386bc24277cc5341644643a42d06f3cf551e4f37f11df91df6297d83fe144b75`.
The separately reviewed ABI patch SHA-256 is
`7f94857901f1016a574d142b746578399740949c89b0ed18a3159411e697aa64`.
A final root-provided inventory checked all 1,402 code/test/tool paths on Spark A,
including files absent from its historical Git lineage. All matched after
synchronizing two already-tested Python collector files; those tools do not
change compiled C++ inputs. The remote `61f8426f.dirty` banner alone does not
identify these sources.

Raw logs, XML and source/compiler receipts remain outside Git under the Spark
supervisor store and `/tmp/llmp-m35-orchestration`. This does not close U3;
other architecture consolidation and quantization ingress remain open.
