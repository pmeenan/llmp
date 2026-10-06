// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include "engine/gemma4_greedy.h"

#include <algorithm>
#include <cmath>

#include "execution/sampling.h"
namespace jitllm::engine {
std::expected<Gemma4GreedyDecision, std::string> JudgeGemma4Greedy(
    std::span<const std::int32_t> drafts, std::span<const float> heads, std::uint32_t vocab) {
  if (drafts.empty() || drafts.size() > 3 || vocab == 0 || vocab > 262144 ||
      heads.size() != (drafts.size() + 1) * vocab ||
      std::ranges::any_of(
          drafts, [vocab](std::int32_t id) { return id < 0 || std::uint32_t(id) >= vocab; }) ||
      std::ranges::any_of(heads, [](float value) { return !std::isfinite(value); }))
    return std::unexpected("Gemma greedy judge needs canonical drafts and complete finite rows");
  std::uint32_t matched = 0;
  for (; matched < drafts.size(); ++matched) {
    const auto token = execution::Greedy(heads.subspan(std::size_t{matched} * vocab, vocab));
    if (!token) return std::unexpected("Gemma greedy target row refused");
    if (*token != drafts[matched]) return Gemma4GreedyDecision{matched + 1, *token};
  }
  const auto token = execution::Greedy(heads.last(vocab));
  if (!token) return std::unexpected("Gemma greedy final row refused");
  return Gemma4GreedyDecision{matched + 1, *token};
}
}  // namespace jitllm::engine
