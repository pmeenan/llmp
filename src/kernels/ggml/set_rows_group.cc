// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/set_rows_group.h"

#include <cstdint>
#include <limits>

#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
bool BoundedChain(const ggml_tensor* tensor) {
  for (unsigned depth = 0; tensor != nullptr && depth < 64; ++depth) {
    if (tensor->view_src == nullptr) return true;
    if (tensor->view_offs > std::numeric_limits<std::uintptr_t>::max() -
                                reinterpret_cast<std::uintptr_t>(tensor->view_src->data))
      return false;
    tensor = tensor->view_src;
  }
  return tensor == nullptr;
}
}  // namespace
std::expected<void, KernelFailure> CheckSetRowsForGrouping(const ggml_tensor* node) {
  if (node == nullptr || node->src[0] == nullptr || !BoundedChain(node))
    return detail::Rejected("invalid set_rows group node or view chain");
  for (const auto* source : node->src)
    if (!BoundedChain(source)) return detail::Rejected("invalid set_rows group operand chain");
  std::uint64_t total = 1;
  for (const auto dimension : node->src[0]->ne) {
    if (dimension <= 0 ||
        static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::uint32_t>::max() / total)
      return detail::Rejected("set_rows group exceeds checked positive U32 indexing");
    total *= static_cast<std::uint64_t>(dimension);
  }
  return CheckSetRows(node);
}
std::expected<void, KernelFailure> CheckSetRowsGroup(std::span<const ggml_tensor* const> nodes) {
  if (nodes.size() < 2 || nodes.size() > kSetRowsGroupMax)
    return detail::Rejected("set_rows group requires two through sixteen stores");
  for (const auto* node : nodes)
    if (auto checked = CheckSetRowsForGrouping(node); !checked) return checked;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (std::size_t j = i + 1; j < nodes.size(); ++j) {
      const auto* a = nodes[i];
      const auto* b = nodes[j];
      if (detail::Overlap(a, b) || detail::Overlap(a, b->src[0]) || detail::Overlap(a, b->src[1]) ||
          detail::Overlap(b, a->src[0]) || detail::Overlap(b, a->src[1]))
        return detail::Rejected("set_rows group destinations overlap a destination or operand");
    }
  }
  return {};
}
}  // namespace llmp::kernels::ggml
