// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shared causal/ring and noncausal block mask source contract for native GGML graph adapters.
#ifndef LLMP_ENGINE_GRAPH_MASK_INPUTS_H_
#define LLMP_ENGINE_GRAPH_MASK_INPUTS_H_

#include <algorithm>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/support.h"
#include "kernels/ggml/llmp_ops.h"

namespace llmp::engine {
// Device masks are graph activations, never staged inputs. Authenticate the
// complete producer before excluding its bytes from the host-input grant.
inline std::expected<std::uint64_t, std::string> GraphMaskSourceBytes(
    const ggml_tensor* mask, const ggml_tensor* positions, std::span<ggml_tensor* const> nodes,
    std::span<ggml_tensor* const> inputs, bool device, std::uint32_t first_row, std::uint32_t rows,
    std::uint32_t cells, std::uint32_t capacity, std::uint32_t window, std::uint32_t context,
    kernels::ggml::CausalMaskRows row_layout = kernels::ggml::CausalMaskRows::kPad32,
    ggml_type output_type = GGML_TYPE_F16,
    kernels::ggml::MaskPolicy policy = kernels::ggml::MaskPolicy::kCausal) {
  namespace kg = kernels::ggml;
  if (row_layout != kg::CausalMaskRows::kPad32 && row_layout != kg::CausalMaskRows::kExact)
    return support::Error("unknown causal/ring mask row layout");
  if (policy != kg::MaskPolicy::kCausal && policy != kg::MaskPolicy::kBlock)
    return support::Error("unknown causal/ring mask visibility policy");
  if (output_type != GGML_TYPE_F16 && output_type != GGML_TYPE_F32)
    return support::Error("unsupported causal/ring mask output type");
  const std::uint64_t element_bytes = output_type == GGML_TYPE_F16 ? 2 : 4;
  const auto output_rows = row_layout == kg::CausalMaskRows::kExact
                               ? std::uint64_t{rows}
                               : (std::uint64_t{rows} + 31) / 32 * 32;
  if (mask == nullptr || cells == 0 || cells > INT32_MAX / element_bytes || rows == 0 ||
      rows > static_cast<std::uint32_t>(INT32_MAX - 31) ||
      output_rows > INT32_MAX / element_bytes / cells || mask->type != output_type ||
      mask->view_src != nullptr || mask->ne[0] != cells ||
      std::cmp_not_equal(mask->ne[1], output_rows) || mask->ne[2] != 1 || mask->ne[3] != 1 ||
      mask->nb[0] != element_bytes || mask->nb[1] != std::uint64_t{cells} * element_bytes ||
      mask->nb[2] != mask->nb[1] * static_cast<std::uint64_t>(mask->ne[1]) ||
      mask->nb[3] != mask->nb[2])
    return support::Error("malformed causal/ring mask descriptor");
  if (!device) {
    if (mask->op != GGML_OP_NONE || std::ranges::count(inputs, mask) != 1 ||
        std::ranges::any_of(mask->src, [](const auto* parent) { return parent != nullptr; }))
      return support::Error("reference mask is not a unique host input");
    return ggml_nbytes(mask);
  }
  if (std::ranges::contains(inputs, mask) || !kg::Gemma4MaskFits(mask) ||
      std::ranges::count(nodes, mask) != 1 || mask->src[0] != positions ||
      std::ranges::any_of(std::span(mask->src).subspan(1),
                          [](const auto* source) { return source != nullptr; }) ||
      kg::LlmpOpInt(mask, 0) != static_cast<std::int64_t>(first_row) ||
      kg::LlmpOpInt(mask, 1) != static_cast<std::int64_t>(rows) ||
      kg::LlmpOpInt(mask, 2) != static_cast<std::int64_t>(capacity) ||
      kg::LlmpOpInt(mask, 3) != static_cast<std::int64_t>(window) ||
      kg::LlmpOpInt(mask, 4) != static_cast<std::int64_t>(context) ||
      kg::LlmpOpInt(mask, 5) != static_cast<std::int32_t>(row_layout) ||
      kg::LlmpOpInt(mask, 6) != static_cast<std::int32_t>(policy) || kg::LlmpOpInt(mask, 7) != 0)
    return support::Error("device mask differs from its graph-owned position producer");
  return 0;
}

// Call only after logical visibility and GraphMaskSourceBytes are checked.
// Reserving both vectors first keeps all staged pointers stable through launch.
inline void StageHostGraphMask(ggml_tensor* mask, std::span<const std::uint16_t> logical,
                               std::vector<std::vector<std::uint16_t>>& storage,
                               std::vector<std::pair<ggml_tensor*, const void*>>& sources) {
  auto& padded = storage.emplace_back(ggml_nbytes(mask) / sizeof(std::uint16_t), 0xFC00);
  std::ranges::copy(logical, padded.begin());
  sources.emplace_back(mask, padded.data());
}
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_GRAPH_MASK_INPUTS_H_
