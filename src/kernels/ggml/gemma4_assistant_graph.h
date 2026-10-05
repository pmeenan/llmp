// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// One Q-only assistant query per independent frozen target slot. Descriptors
// only: the caller funds and protects every immutable operand through completion.
#ifndef JITLLM_KERNELS_GGML_GEMMA4_ASSISTANT_GRAPH_H_
#define JITLLM_KERNELS_GGML_GEMMA4_ASSISTANT_GRAPH_H_

#include <array>
#include <cstdint>
#include <expected>
#include <vector>

#include "kernels/ggml/gemma4_graph.h"
#include "model/gemma4_assistant.h"

namespace jitllm::kernels::ggml {
struct Gemma4AssistantSegment {
  std::uint32_t slot = 0, prefix = 0, local_n_kv = 0, global_n_kv = 0;
  // The prefix/query position is a fresh input, not a capture/cache dimension.
  bool operator==(const Gemma4AssistantSegment& other) const {
    return slot == other.slot && local_n_kv == other.local_n_kv && global_n_kv == other.global_n_kv;
  }
};
struct Gemma4AssistantShape {
  std::vector<Gemma4AssistantSegment> segments;
  bool operator==(const Gemma4AssistantShape&) const = default;
};
struct Gemma4AssistantSegmentTensors {
  Gemma4AssistantSegment shape;
  // Local then global: K/V are already normalized/rotated target state.
  std::array<std::pair<ggml_tensor*, ggml_tensor*>, 2> caches{};
  std::array<ggml_tensor*, 2> masks{};
};
struct Gemma4AssistantGraph {
  model::Gemma4AssistantProfile profile;
  model::Gemma4AssistantBinding binding;
  model::Gemma4Profile target;
  model::Gemma4Tensor target_embedding;
  std::uint32_t context = 0, max_rows = 0, local_capacity = 0, global_capacity = 0;
  Gemma4AssistantShape shape;
  ggml_tensor *tokens = nullptr, *features = nullptr, *positions = nullptr;
  ggml_tensor *logits = nullptr, *next_features = nullptr, *embedding = nullptr;
  std::vector<Gemma4WeightLeaf> weights;
  std::vector<Gemma4AssistantSegmentTensors> segments;
  std::vector<ggml_tensor*> inputs, nodes;
};
std::expected<void, KernelFailure> CheckGemma4AssistantGraph(
    const model::Gemma4AssistantProfile& profile, const model::Gemma4AssistantBinding& binding,
    const model::Gemma4Profile& target, const model::Gemma4Binding& target_binding,
    const model::Gemma4StateLayout& state, const Gemma4AssistantShape& shape);
std::size_t Gemma4AssistantGraphTensors(std::size_t segments);
std::expected<Gemma4AssistantGraph, KernelFailure> BuildGemma4AssistantGraph(
    TensorArena& arena, const model::Gemma4AssistantProfile& profile,
    const model::Gemma4AssistantBinding& binding, const model::Gemma4Profile& target,
    const model::Gemma4Binding& target_binding, const model::Gemma4StateLayout& state,
    const Gemma4AssistantShape& shape);
}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_GEMMA4_ASSISTANT_GRAPH_H_
