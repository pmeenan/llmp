// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_outa.h"

#include <cmath>
#include <cstddef>
#include <expected>

#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {

bool Dsv4OutAFits(const ggml_tensor* w, const ggml_tensor* x, const ggml_tensor* pos,
                  const Dsv4OutAParams& p) {
  if (w == nullptr || w->type != GGML_TYPE_Q8_0 || w->ne[0] != 4096 || w->ne[1] != 1024 ||
      w->ne[2] != 8 || w->ne[3] != 1 || !detail::IsF32(x) || x->ne[0] != 512 || x->ne[1] != 64 ||
      x->ne[2] < kDsv4OutAMinRows || x->ne[2] > kDsv4OutAMaxRows || x->ne[3] != 1 ||
      pos == nullptr || pos->type != GGML_TYPE_I32 || pos->ne[0] != x->ne[2] || pos->ne[1] != 1 ||
      pos->ne[2] != 1 || pos->ne[3] != 1 || !detail::Packed(w) || !detail::Packed(x) ||
      !detail::Packed(pos)) {
    return false;
  }
  if (p.original_context < 0 || !std::isfinite(p.base) || p.base <= 0 || !std::isfinite(p.scale) ||
      p.scale <= 0 || !std::isfinite(p.extension) || p.extension < 0 ||
      !std::isfinite(p.attention) || !std::isfinite(p.beta_fast) || !std::isfinite(p.beta_slow)) {
    return false;
  }
  return p.extension == 0 ||
         (p.original_context > 0 && p.beta_fast > 0 && p.beta_slow > 0 && p.base != 1);
}

Dsv4OutAParams Dsv4OutAParamsOf(const ggml_tensor* node) {
  return {.original_context = LlmpOpInt(node, 0),
          .base = LlmpOpFloat(node, 1),
          .scale = LlmpOpFloat(node, 2),
          .extension = LlmpOpFloat(node, 3),
          .attention = LlmpOpFloat(node, 4),
          .beta_fast = LlmpOpFloat(node, 5),
          .beta_slow = LlmpOpFloat(node, 6)};
}

std::expected<void, KernelFailure> CheckDsv4OutA(const ggml_tensor* node) {
  if (LlmpOpOf(node) != LlmpOp::kDsv4OutA || !detail::Bound(node) || !detail::Bound(node->src[0]) ||
      !detail::Bound(node->src[1]) || !detail::Bound(node->src[2])) {
    return detail::Rejected("not a bound DeepSeek output-A node");
  }
  for (std::size_t i = 3; i < GGML_MAX_SRC; ++i) {
    if (node->src[i] != nullptr) {
      return detail::Rejected("DeepSeek output-A takes only weights, heads and positions");
    }
  }
  const auto* w = node->src[0];
  const auto* x = node->src[1];
  const auto* pos = node->src[2];
  if (!Dsv4OutAFits(w, x, pos, Dsv4OutAParamsOf(node)) || !detail::IsF32(node) ||
      node->ne[0] != 8192 || node->ne[1] != Dsv4OutARows(x->ne[2]) || node->ne[2] != 1 ||
      node->ne[3] != 1 || !detail::Packed(node) || !detail::AllSane({node, w, x, pos}) ||
      !detail::AllCurrent({node, w, x, pos}) || !detail::Aligned(node, 16) ||
      !detail::Aligned(w, alignof(std::uint16_t)) || !detail::Aligned(x, 16) ||
      !detail::Aligned(pos, alignof(std::int32_t)) || !detail::Disjoint(node, w, false) ||
      !detail::Disjoint(node, x, false) || !detail::Disjoint(node, pos, false)) {
    return detail::Rejected("DeepSeek output-A requires bounded packed disjoint F32 output");
  }
  return {};
}

}  // namespace llmp::kernels::ggml
