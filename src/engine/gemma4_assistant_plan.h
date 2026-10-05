// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_ENGINE_GEMMA4_ASSISTANT_PLAN_H_
#define JITLLM_ENGINE_GEMMA4_ASSISTANT_PLAN_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/gemma4_plan.h"
#include "kernels/ggml/gemma4_assistant_graph.h"

namespace jitllm::engine {
struct Gemma4AssistantModel {
  const model::Gemma4AssistantProfile* profile = nullptr;
  const model::Gemma4AssistantBinding* binding = nullptr;
  const Gemma4Model* target = nullptr;
  // Exact closed49 ordinary resource domain. Target regions retain all
  // immutable weights/peer slots, including those the assistant never reads.
  std::vector<Gemma4Region> resources;
};
using Gemma4AssistantPlanned = PlannedGraph<kernels::ggml::Gemma4AssistantGraph>;
std::expected<void, std::string> BindGemma4AssistantWeights(
    const Gemma4AssistantModel& model, kernels::ggml::Gemma4AssistantGraph& graph);
std::expected<std::unique_ptr<Gemma4AssistantPlanned>, std::string> PlanGemma4Assistant(
    const Gemma4AssistantModel& model, const kernels::ggml::Gemma4AssistantShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes);

struct Gemma4AssistantInput {
  std::uint32_t slot = 0, prefix = 0;
  std::int32_t anchor = 0;
};
struct Gemma4AssistantHostInputs {
  std::vector<std::int32_t> tokens, positions;
  std::vector<std::vector<std::uint16_t>> masks;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};
// Caller funds container/mask bytes before this allocates. Frozen cache
// initialization/residency and same-slot mutation exclusion remain engine
// obligations. With device_features, caller copies protected F32 features
// into graph.features on the same stream before the plan, through completion.
std::expected<std::uint64_t, std::string> Gemma4AssistantSourceBytes(
    const kernels::ggml::Gemma4AssistantGraph& graph, bool device_features);
std::expected<Gemma4AssistantHostInputs, std::string> Gemma4AssistantSources(
    const kernels::ggml::Gemma4AssistantGraph& graph, std::span<const Gemma4AssistantInput> inputs,
    std::span<const float> features, bool device_features, std::uint64_t funded_bytes);
}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_GEMMA4_ASSISTANT_PLAN_H_
