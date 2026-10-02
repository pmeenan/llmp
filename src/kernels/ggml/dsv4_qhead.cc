// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_qhead.h"

#include <cmath>
#include <cstddef>
#include <expected>

#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {

bool Dsv4QHeadFits(const ggml_tensor* x, const ggml_tensor* positions, const Dsv4QHeadParams& p) {
  if (!detail::IsF32(x) || positions == nullptr || positions->type != GGML_TYPE_I32 ||
      x->ne[0] != 512 || x->ne[1] != 64 || x->ne[2] < 16 || x->ne[2] > 8192 || x->ne[3] != 1 ||
      positions->ne[0] != x->ne[2] || positions->ne[1] != 1 || positions->ne[2] != 1 ||
      positions->ne[3] != 1 || !detail::Packed(x) || !detail::Packed(positions)) {
    return false;
  }
  if (!std::isfinite(p.eps) || p.eps < 0 || p.original_context < 0 || !std::isfinite(p.base) ||
      p.base <= 0 || !std::isfinite(p.scale) || p.scale <= 0 || !std::isfinite(p.extension) ||
      p.extension < 0 || !std::isfinite(p.attention) || !std::isfinite(p.beta_fast) ||
      !std::isfinite(p.beta_slow)) {
    return false;
  }
  return p.extension == 0 ||
         (p.original_context > 0 && p.beta_fast > 0 && p.beta_slow > 0 && p.base != 1);
}

Dsv4QHeadParams Dsv4QHeadParamsOf(const ggml_tensor* node) {
  return {.eps = JitllmOpFloat(node, 0),
          .original_context = JitllmOpInt(node, 1),
          .base = JitllmOpFloat(node, 2),
          .scale = JitllmOpFloat(node, 3),
          .extension = JitllmOpFloat(node, 4),
          .attention = JitllmOpFloat(node, 5),
          .beta_fast = JitllmOpFloat(node, 6),
          .beta_slow = JitllmOpFloat(node, 7)};
}

std::expected<void, KernelFailure> CheckDsv4QHead(const ggml_tensor* node) {
  if (JitllmOpOf(node) != JitllmOp::kDsv4QHead || !detail::Bound(node) ||
      !detail::Bound(node->src[0]) || !detail::Bound(node->src[1])) {
    return detail::Rejected("not a bound DeepSeek Q-head node");
  }
  for (std::size_t i = 2; i < GGML_MAX_SRC; ++i) {
    if (node->src[i] != nullptr) {
      return detail::Rejected("DeepSeek Q-head takes only Q and positions");
    }
  }
  const auto* x = node->src[0];
  const auto* positions = node->src[1];
  if (!Dsv4QHeadFits(x, positions, Dsv4QHeadParamsOf(node)) || !detail::IsF32(node) ||
      !ggml_are_same_shape(node, x) || !detail::Packed(node) ||
      !detail::AllSane({node, x, positions}) || !detail::AllCurrent({node, x, positions}) ||
      !detail::Aligned(node, alignof(float)) || !detail::Aligned(x, alignof(float)) ||
      !detail::Aligned(positions, alignof(std::int32_t)) || !detail::Disjoint(node, x, false) ||
      !detail::Disjoint(node, positions, false)) {
    return detail::Rejected("DeepSeek Q-head requires bounded contiguous disjoint F32 outputs");
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
