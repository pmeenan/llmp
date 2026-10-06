// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_ENGINE_GEMMA4_GREEDY_H_
#define JITLLM_ENGINE_GEMMA4_GREEDY_H_
#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>
namespace jitllm::engine {
struct Gemma4GreedyDecision {
  std::uint32_t keep = 0;
  std::int32_t next_anchor = 0;
};
// Row j predicts draft j; the final row predicts a pending next anchor.
// All rows, including unused tails, must be finite. No scalar-width parity
// assumption or numerical allowance enters this target-authoritative judge.
std::expected<Gemma4GreedyDecision, std::string> JudgeGemma4Greedy(
    std::span<const std::int32_t> drafts, std::span<const float> heads, std::uint32_t vocab);
// Caller-funded capacities checked before dispatch; the unit never expands
// these vectors. Workspace may change on failure, published result may not.
struct Gemma4GreedyWorkspace {
  std::vector<float> draft_head, verify_heads, verify_features;
};
struct Gemma4GreedyResult {
  std::array<std::int32_t, 4> committed{};
  std::uint32_t count = 0, past = 0;
  std::int32_t next_anchor = 0;  // Predicted, not committed to target state.
  std::vector<float> head, feature;
};
}  // namespace jitllm::engine
#endif  // JITLLM_ENGINE_GEMMA4_GREEDY_H_
