// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_BENCHMARKS_QWEN38_BATCH_GRAPH_H_
#define JITLLM_BENCHMARKS_QWEN38_BATCH_GRAPH_H_

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "engine/qwen38_plan.h"

namespace jitllm::benchmarks::qwen_batch {

inline constexpr std::size_t kMaxRequests = 4;
template <typename T>
using Requests = std::array<T, kMaxRequests>;

// Private C2/C4 experiment. All slots own independent tensor descriptors and
// retain every original per-request effect order. Only these two row-local
// native operations may coalesce; no attention/recurrence/head is rewritten.
struct Plan : engine::PlannedBase {
  Requests<std::unique_ptr<engine::Qwen38Planned>> target;
  Requests<std::unique_ptr<engine::Qwen38MtpPlanned>> mtp;
  std::vector<ggml_tensor*> nodes;
  std::vector<ggml_tensor*> inputs;
  std::vector<ggml_tensor*> keep;
  // Original nodes and their replacement, authenticated against the chosen
  // registry identities after placement. No wider precision tier is allowed.
  std::vector<std::array<ggml_tensor*, 3>> products;
  std::uint64_t mxfp8_pairs = 0;
  std::uint64_t routed_pairs = 0;
  std::uint64_t packed_bytes = 0;
};

std::expected<std::unique_ptr<Plan>, std::string> TargetPlan(
    const Requests<engine::Qwen38Model>& models,
    const Requests<kernels::ggml::Qwen38ChunkShape>& shapes,
    const kernels::ggml::DeviceChoices& choices, bool batch, std::uint64_t activations = 0,
    std::uint64_t activation_bytes = 0, std::uint32_t active = 3);

std::expected<std::unique_ptr<Plan>, std::string> DraftPlan(
    const Requests<engine::Qwen38Model>& models,
    const Requests<kernels::ggml::Qwen38MtpShape>& shapes,
    const kernels::ggml::DeviceChoices& choices, bool batch, std::uint64_t activations = 0,
    std::uint64_t activation_bytes = 0, std::uint32_t active = 3);

}  // namespace jitllm::benchmarks::qwen_batch

#endif  // JITLLM_BENCHMARKS_QWEN38_BATCH_GRAPH_H_
