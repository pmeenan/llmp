// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma graph/plan foundation, not a serving runner. The caller owns and
// funds every weight/state/activation/input region through completion.
#ifndef JITLLM_ENGINE_GEMMA3_PLAN_H_
#define JITLLM_ENGINE_GEMMA3_PLAN_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/planned.h"
#include "kernels/ggml/gemma3_graph.h"
#include "model/gemma3.h"

namespace jitllm::engine {
struct Gemma3Region {
  std::uint64_t address = 0, bytes = 0;
};
struct Gemma3Model {
  const model::Gemma3Profile* profile = nullptr;
  const model::Gemma3Binding* binding = nullptr;
  const model::Gemma3StateLayout* state = nullptr;
  // Ordinary resource indices retain binding identity; each slot owns KV.
  std::vector<Gemma3Region> resources, slots;
  kernels::ggml::Gemma3GraphOptions options;
};
using Gemma3Planned = PlannedGraph<kernels::ggml::Gemma3Graph>;

// Pre-placement root binding: checked logical regions, tied resource identity,
// complete consumer/leaf correspondence and separate slot state/readability.
// Call PlaceAndPlan (which refreshes views) before use. Root relocation
// invalidates previously bound launches/captures; discard them and replan. Residency/initialization
// remains caller's obligation (Gemma3UsedState and Gemma3ChunkWrites).
std::expected<void, std::string> BindGemma3Weights(const Gemma3Model& model,
                                                   kernels::ggml::Gemma3Graph& graph);
std::expected<std::unique_ptr<Gemma3Planned>, std::string> PlanGemma3Chunk(
    const Gemma3Model& model, const kernels::ggml::Gemma3ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, std::span<const std::string> keep = {});

// Host-reference graphs stage funded padded masks; device-mask graphs keep
// those matrices in activation storage and stage only checked positions/indices.
// `chunk`/`hidden` must outlive staging; the returned object owns frontier IDs
// and any host-reference mask padding. `funded_bytes` is a caller-held grant,
// checked before allocation, not an internally obtained memory reservation.
// Move-only ownership keeps the staged pointers into owned buffers stable.
struct Gemma3HostInputs {
  Gemma3HostInputs() = default;
  Gemma3HostInputs(const Gemma3HostInputs&) = delete;
  Gemma3HostInputs& operator=(const Gemma3HostInputs&) = delete;
  Gemma3HostInputs(Gemma3HostInputs&&) noexcept = default;
  Gemma3HostInputs& operator=(Gemma3HostInputs&&) noexcept = default;
  std::vector<std::int32_t> out_ids;
  std::vector<std::vector<std::uint16_t>> masks;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};
std::expected<std::uint64_t, std::string> Gemma3SourceBytes(
    const kernels::ggml::Gemma3Graph& graph);
std::expected<Gemma3HostInputs, std::string> Gemma3Sources(const kernels::ggml::Gemma3Graph& graph,
                                                           const model::Gemma3ChunkInputs& chunk,
                                                           std::span<const std::int32_t> frontier,
                                                           std::span<const float> hidden,
                                                           std::uint64_t funded_bytes);

}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_GEMMA3_PLAN_H_
