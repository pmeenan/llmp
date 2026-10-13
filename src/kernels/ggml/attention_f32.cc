// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/ggml/attention_f32.h"

#include <array>
#include <cmath>

#include "kernels/ggml/validate_util.h"
namespace llmp::kernels::ggml {
std::expected<AttentionF32Plan, KernelFailure> CheckAttentionF32(const ggml_tensor* node) {
  using namespace detail;
  if (!Bound(node) || node->op != GGML_OP_FLASH_ATTN_EXT || !Bound(node->src[0]) ||
      !Bound(node->src[1]) || !Bound(node->src[2]) || !Bound(node->src[3]) || node->src[4])
    return Rejected("F32 attention requires Q/K/V and the original additive mask, without sinks");
  const auto* q = node->src[0];
  const auto* k = node->src[1];
  const auto* v = node->src[2];
  const auto* mask = node->src[3];
  for (const auto* t : {node, q, k, v, mask}) {
    if (!IsF32(t) || !AllSane({t}) || !AllCurrent({t}) || !Aligned(t, 4) || !ElementStrides(t) ||
        Span(t) > kInt32Max)
      return Rejected("F32 attention requires current bounded whole-element operands");
    // Scalar cpy uses int64 offsets; this stronger span bound also funds its
    // int dimensions, stride division, and ceil(elements / 64) grid.
    for (int i = 0; i < GGML_MAX_DIMS; ++i)
      if (t->ne[i] > static_cast<std::int64_t>(kInt32Max) || t->nb[i] > kInt32Max)
        return Rejected("F32 attention exceeds the pinned conversion integer domain");
  }
  if (!Packed(q) || !Packed(node) || !Aligned(q, 16) || !Aligned(node, 16) || q->ne[0] != 128 ||
      k->ne[0] != 128 || !ggml_are_same_shape(k, v) || q->ne[2] != k->ne[2] || q->ne[3] != 1 ||
      k->ne[3] != 1 || node->ne[0] != 128 || node->ne[1] != q->ne[2] || node->ne[2] != q->ne[1] ||
      node->ne[3] != 1 || mask->ne[0] != k->ne[1] || mask->ne[1] != q->ne[1] || mask->ne[2] != 1 ||
      mask->ne[3] != 1)
    return Rejected("F32 D128 attention requires one MHA sample and the complete real mask");
  if (!std::isfinite(ParamF32(node, 0)) || ParamF32(node, 1) != 0 || ParamF32(node, 2) != 0 ||
      node->op_params[4] != 0)
    return Rejected("F32 D128 attention has no ALiBi, softcap or sparse bound");
  for (const auto* t : {q, k, v, mask})
    if (!Disjoint(node, t, false)) return Rejected("attention output overlaps an input");
  AttentionF32Plan plan;
  plan.cells = (static_cast<std::uint64_t>(k->ne[1]) + 255) / 256 * 256;
  plan.rows = (static_cast<std::uint64_t>(q->ne[1]) + 63) / 64 * 64;
  // Bound products before constructing GGML descriptors or doing pointer
  // arithmetic. The padded K/V also meet the existing D128 head stride check.
  if (plan.cells > kInt32Max / 256 / static_cast<std::uint64_t>(k->ne[2]) ||
      plan.cells > kInt32Max / 2 / plan.rows)
    return Rejected("padded attention exceeds the kernel's byte/stride domain");
  plan.kv_bytes = plan.cells * 256 * static_cast<std::uint64_t>(k->ne[2]);
  plan.mask_bytes = plan.cells * plan.rows * 2;
  plan.packing_bytes = 2 * plan.kv_bytes + plan.mask_bytes;
  return plan;
}
}  // namespace llmp::kernels::ggml
