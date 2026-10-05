<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma MoE structural matchers

`gemma_moe_fusion.h` provides allocation-free structural matching for the
[checked standalone routing and scaled-reduction primitives](experiments/gemma-moe-primitives/README.md).
The later [native dispatch](experiments/gemma-native-moe/README.md) uses these
matchers through explicit default-off policies. Whole-model quality, performance
and optimized batching qualification remain owed.

The routing matcher requires the exact ten-node native chain: unmasked,
unbiased unit-scale softmax; probability reshape; descending full 128-entry
ARGSORT; its first-eight view with 512-byte row pitch; gather and reshape;
SUM_ROWS; the exact `2^-14` denominator clamp; DIV and final weight reshape.
It returns both selected IDs and normalized weights. The entire ARGSORT root
still needs backing, although only its first eight entries are written. Keeping
or reading the full sort, probabilities or an unwritten normalization value
forces primitive fallback. Direct consumers of the exact selected-ID view and
views of the fully written DIV backing are allowed.

The reduction matcher requires a contiguous seventeen-node chain: expert
values multiplied by their selected scales, then routing weights; eight
selected-slot views; and seven ascending dependent additions. Its three input
operands remain distinct from the elided producers. A kept or additional
reader of any scale/weight product, contribution view or partial sum forces
primitive fallback. The final output may be kept.

Both matchers check shapes, strides, links, operation parameters and the
standalone operand contracts. Root/view traversal is bounded before reader
and keep scans; cyclic, stale and address-overflowed descriptors are refused.
Computed intermediates must be nonviews, and actual operands cannot depend on
any matched producer. Logical storage identities govern dependency checks;
incidental activation-address reuse does not establish a dependency. Returned
byte counts specify required logical storage, not catalog residency or leases.
Binding passes actual protected spans and preserves all ten or seventeen
descriptors in `PlanStep.nodes` for conservative placement; this
slice makes no activation-storage elision claim.

The original matcher slice observed 30 routing chains and interleaved reduction
fallback. The later graph-order seam expands the ordered raw routed sum before
its post-normalization, so the current 26B graph matches all 30 routing and
30 reduction chains for segments 1/2/4. Its ordinary complete heads remain byte
exact. Dense31 matches neither MoE operation. The dispatch report preserves
both the failed compound quality control and its default-off selection.

Controls cover row counts 1/2/4/8/128/8,192, exact clamp and sort/pitch,
independent outputs, all unwritten keeps, external consumers, wrong arithmetic
links, cyclic/stale/overflowed metadata, input dependencies on elided outputs,
computed-view substitutions and post-placement full-root funding. These are
metadata controls; original-library arithmetic fidelity remains the separate
standalone primitive evidence.

Task-entry reference refresh on 2026-10-05 at 04:05:53 UTC observed TensorFold
HEAD `609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5; its pinned
[README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and [package metadata](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/pyproject.toml)
still expose Gemma 26B through MLX, without a CUDA Gemma comparator. The
same-format pinned llama.cpp remains the numerical reference.

Locked Spark-native build and all 11 focused CPU controls passed on physical
host `spark-c4e2`; the final integrated Spark-b suite passed all 1,633 tests
(290 GPU, 37 model tests, no skips) in 191.36 s on `spark-56f5`, 2026-10-05 UTC,
from base `4f9be31` with these matcher sources, SDK `aarch64-c09daba6ac31edee`
and prepared GGML tree `026f1ac94af98011`.
