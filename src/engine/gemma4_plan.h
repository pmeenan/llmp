// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma graph/plan foundation, not a serving runner. The caller owns and
// funds every weight/state/activation/input region through completion.
#ifndef JITLLM_ENGINE_GEMMA4_PLAN_H_
#define JITLLM_ENGINE_GEMMA4_PLAN_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "engine/planned.h"
#include "kernels/ggml/gemma4_graph.h"
#include "model/gemma4.h"

namespace jitllm::engine {
struct Gemma4Region {
  std::uint64_t address = 0, bytes = 0;
};
struct Gemma4Model {
  const model::Gemma4Profile* profile = nullptr;
  const model::Gemma4Binding* binding = nullptr;
  const model::Gemma4StateLayout* state = nullptr;
  // Ordinary/expert address domains are the binding's separate indices.
  // Array address is its member's first slice; stride includes readable tail.
  std::vector<Gemma4Region> resources, arrays, slots;
  kernels::ggml::Gemma4GraphOptions options;
};
using Gemma4Planned = PlannedGraph<kernels::ggml::Gemma4Graph>;

// Checked logical regions, tied resource identity, separate slot state and
// array readability before binding. Residency/initialization remains caller's
// obligation (Gemma4UsedState and Gemma4ChunkWrites).
std::expected<void, std::string> BindGemma4Weights(const Gemma4Model& model,
                                                   kernels::ggml::Gemma4Graph& graph);
std::expected<std::unique_ptr<Gemma4Planned>, std::string> PlanGemma4Chunk(
    const Gemma4Model& model, const kernels::ggml::Gemma4ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, std::span<const std::string> keep = {});

// Device-mask graphs stage only fresh row/segment sources. Diagnostic graphs
// also own/stage padded host reference masks; their O(rows*n_kv) host cost is
// fully funded. `chunk`/`hidden` must outlive staging, the returned object owns
// frontier IDs and every padded mask. `funded_bytes` is a caller-held grant,
// checked before allocation, not an internally obtained memory reservation.
struct Gemma4HostInputs {
  std::vector<std::int32_t> out_ids;
  std::vector<std::vector<std::uint16_t>> masks;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};
std::expected<std::uint64_t, std::string> Gemma4SourceBytes(
    const kernels::ggml::Gemma4Graph& graph);
std::expected<Gemma4HostInputs, std::string> Gemma4Sources(const kernels::ggml::Gemma4Graph& graph,
                                                           const model::Gemma4ChunkInputs& chunk,
                                                           std::span<const std::int32_t> frontier,
                                                           std::span<const float> hidden,
                                                           std::uint64_t funded_bytes);

}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_GEMMA4_PLAN_H_
