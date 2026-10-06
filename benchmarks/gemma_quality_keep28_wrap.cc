// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Benchmark-only existing keep seam. No production layer policy or math change.
#include <cstdlib>
#include <string_view>
#include <vector>

#include "engine/gemma4_plan.h"
#include "gemma_quality_keep28.h"

namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
using Result = std::expected<std::unique_ptr<en::Gemma4Planned>, std::string>;
Result
RealPlan(const en::Gemma4Model&, const kg::Gemma4ChunkShape&, const kg::DeviceChoices&, std::uint64_t, std::uint64_t, std::span<const std::string>) asm(
    "__real__ZN6jitllm6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml"
    "16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIc"
    "St11char_traitsIcESaIcEEELm18446744073709551615EE");

std::optional<bool> en::diagnostic::Keep28Routing() {
  static const auto mode = []() -> std::optional<bool> {
    const char* value = std::getenv("JITLLM_GEMMA_KEEP28_ROUTING");
    if (value == nullptr) return std::nullopt;
    if (std::string_view(value) == "0") return false;
    if (std::string_view(value) == "1") return true;
    return std::nullopt;
  }();
  return mode;
}

Result
WrapPlan(const en::Gemma4Model&, const kg::Gemma4ChunkShape&, const kg::DeviceChoices&, std::uint64_t, std::uint64_t, std::span<const std::string>) asm(
    "__wrap__ZN6jitllm6engine15PlanGemma4ChunkERKNS0_11Gemma4ModelERKNS_7kernels4ggml"
    "16Gemma4ChunkShapeERKNS5_13DeviceChoicesEmmSt4spanIKNSt7__cxx1112basic_stringIc"
    "St11char_traitsIcESaIcEEELm18446744073709551615EE");
Result WrapPlan(const en::Gemma4Model& model, const kg::Gemma4ChunkShape& shape,
                const kg::DeviceChoices& choices, std::uint64_t activations,
                std::uint64_t activation_bytes, std::span<const std::string> keep) {
  const auto enabled = en::diagnostic::Keep28Routing();
  if (!enabled) return std::unexpected("missing or invalid keep28 process mode");
  // Measurement also visits one-row, frontier and state-only shapes. Apply the
  // same keep before BOTH planning passes, so their normal size/host floors
  // account the additional live intermediate. Unsupported callers are exact
  // original fallbacks; the closed executable verifies its selected counts.
  if (!*enabled || model.profile == nullptr || model.state == nullptr ||
      *model.profile != jitllm::model::Gemma4_26BA4B() || model.state->context != 4096 ||
      model.state->max_rows != 1024 || model.slots.size() != 1 || shape.segments.size() != 1 ||
      shape.segments.front().slot != 0 || shape.feature_outputs != 0 ||
      model.options.first_layer != 0 || model.options.layer_count != 0 ||
      model.options.hidden_input || !model.options.head) {
    return RealPlan(model, shape, choices, activations, activation_bytes, keep);
  }
  std::vector<std::string> diagnostic_keep(keep.begin(), keep.end());
  diagnostic_keep.emplace_back("blk.28.router_probabilities");
  return RealPlan(model, shape, choices, activations, activation_bytes, diagnostic_keep);
}
