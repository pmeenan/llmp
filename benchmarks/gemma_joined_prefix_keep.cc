// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Benchmark-only external reader; no production routing policy or math change.
#include "gemma_joined_prefix_keep.h"

#include <cstdlib>
#include <optional>
#include <string_view>
#include <vector>
namespace en = llmp::engine;
namespace kg = llmp::kernels::ggml;
GemmaPrefixPlanResult PrefixWrapPlan(const en::Gemma4Model& model,
                                     const kg::Gemma4ChunkShape& shape,
                                     const kg::DeviceChoices& choices, std::uint64_t activations,
                                     std::uint64_t activation_bytes,
                                     std::span<const std::string> keep) {
  static const auto enabled = []() -> std::optional<bool> {
    const char* mode = std::getenv("LLMP_GEMMA_PREFIX_KEEP28");
    if (mode == nullptr) return std::nullopt;
    if (std::string_view(mode) == "0") return false;
    if (std::string_view(mode) == "1") return true;
    return std::nullopt;
  }();
  if (!enabled) return std::unexpected("missing or invalid prefix keep process mode");
  // Apply the same keep to measurement and both normal planning passes. It
  // changes only eligible multi-row independent prefixes, never joined decode.
  if (!*enabled || model.profile == nullptr || model.state == nullptr ||
      *model.profile != llmp::model::Gemma4_26BA4B() || model.state->context != 4096 ||
      model.state->max_rows != 1024 || model.slots.size() != 5 || shape.segments.size() != 1 ||
      shape.segments.front().rows <= 1 || shape.segments.front().slot >= 5 ||
      shape.feature_outputs != 0 || model.options.first_layer != 0 ||
      model.options.layer_count != 0 || model.options.hidden_input || !model.options.head) {
    return PrefixRealPlan(model, shape, choices, activations, activation_bytes, keep);
  }
  std::vector<std::string> diagnostic_keep(keep.begin(), keep.end());
  diagnostic_keep.emplace_back("blk.28.router_probabilities");
  return PrefixRealPlan(model, shape, choices, activations, activation_bytes, diagnostic_keep);
}
