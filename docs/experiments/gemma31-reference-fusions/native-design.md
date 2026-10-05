<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Proposed checked native norm contracts

This records the design derived from the dense31 reference controls. Native
baseline `5b30341` keeps ordinary arithmetic. The later
[checked native implementation](../gemma-native-norm/README.md) implements a
narrow subset with default-off policies: NEOX D256/D512 norm/rotation and
full-width F32 norm/residual addition. Other rotation modes, residual widths
and in-place destinations below remain design possibilities. Dense31 and
routed26 require separate model and paid-work evidence against their own
immutable calibrations.

The pinned `norm.cu` and `rope.cu` units already expose
`ggml_cuda_op_rms_norm_fused_add` and
`ggml_cuda_op_rms_norm_mul_rope_fused`. They are already in jitLLM's locked
kernel inventory. No new upstream CUDA unit, source-lock patch or dispatcher
is needed for these two launchers. Routing/reduction contracts remain separate.

Add checked wrappers in the existing native norm/operation surface, with
vendor-free declarations and CPU checks, and narrow CUDA calls through
`LaunchContext::Run(Bytes(0), ...)`. Register named three-node operations for
RMSNorm/MUL/ADD and RMSNorm/MUL/RoPE through the ordinary `Kernel` entry table;
the existing two-node `RmsNormMulKernel` contract stays unchanged. Plans retain
the input, learned weight, residual, positions, optional frequency factors and
destination dependencies. No intermediate allocation elision or memory saving
is claimed in the first implementation.

Both wrappers require exact source edges, F32 source/learned weight/output,
finite nonnegative epsilon, nonempty checked extents, aligned element strides,
readable spans, integer-safe shapes and current view generations. A learned
weight may broadcast only as the pinned launcher supports; reversed MUL
operands cannot add unsupported broadcasting. Elided intermediate storage
cannot supply another live operand. Destination overlap is initially refused
except for independently checked safe complete in-place cases; uncertainty
falls back to the existing primitives.

The residual wrapper authenticates which ADD operand is the normalized
product and checks the other operand's F32 broadcast and readable extent.
The dense F32 destination must have the norm's logical shape. Both operand
orders and real 26B/31B residual widths need tests; no DeepSeek weighted
reducer or different post-norm formula is substituted.

The rotation wrapper initially handles only the three-node form, writing
the F32 rotated output; cache copies remain separate. It requires contiguous
source rows, exact norm/product/rotation shapes and connections, ordinary or
NEOX mode, even supported rotation width, zero rotation offset, checked I32
positions and optional F32 frequency factors. The actual Gemma D256/D512
factor-aware forms need operand controls. Direct F16 cache-store fusion is
a later contract requiring the existing `CheckRopeSetRows` alias, row-index,
destination-stride and bounds proof, plus state-lifetime controls.

Add separate default-false `DeviceChoices` policies for these families and
matching explicit runner diagnostics. In primitive planning, test the longer
patterns before the existing RMSNorm/MUL choice. Respect every diagnostic
`keep`, storage-root alias and external reader of either elided intermediate;
when a chain is ineligible, preserve the original primitive plan. Disabled
policies must reproduce the previous plan choices and output. Generic
upstream-fusion planning still refuses other unimplemented patterns.
In particular, global attention's raw K-as-V source and every live norm/view
consumer remain materialized; a superficially matching three-node sequence
cannot erase their required inputs.

Verify checked operand refusals before submission; source/weight/residual
order and broadcasting; D256/D512 learned norms and factors; stale views,
partial overlaps and kept intermediates; launch failure and workspace bounds.
Run complete 26B/31B layer, all-head likelihood and checkpoint/spill/replay
controls with policies off and explicitly on. Funding and completion stay
under the shared runner; no new state, lifecycle, stream or scheduler appears.
Default selection requires later quality and competitive paid performance
evidence, rather than synthetic speed or short selected-row agreement.
