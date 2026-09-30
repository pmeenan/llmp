// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_RUNTIME_TURN_REUSE_H_
#define JITLLM_RUNTIME_TURN_REUSE_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace jitllm::runtime {

// Branch-local rollback cache. Prefix equality identifies reusable state,
// not a conversation or its lifetime (D-031). Independent shared-prefix
// retention and forks belong to the later request scheduler.
inline constexpr std::size_t kTurnCheckpointLimit = 2;
inline constexpr auto kTurnCheckpointRetention = std::chrono::hours(24);

struct TurnBoundary {
  std::size_t position = 0;
  std::chrono::steady_clock::time_point created;
};

inline std::size_t CommonPrefix(std::span<const std::int32_t> a, std::span<const std::int32_t> b) {
  std::size_t n = 0;
  while (n < a.size() && n < b.size() && a[n] == b[n]) {
    ++n;
  }
  return n;
}

inline std::optional<std::size_t> MatchingTurnBoundary(std::span<const TurnBoundary> boundaries,
                                                       std::size_t common,
                                                       std::size_t request_tokens,
                                                       std::chrono::steady_clock::time_point now) {
  std::optional<std::size_t> best;
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    const TurnBoundary& boundary = boundaries[i];
    if (boundary.position == 0 || boundary.position > common ||
        boundary.position >= request_tokens || boundary.created > now ||
        now - boundary.created >= kTurnCheckpointRetention) {
      continue;
    }
    if (!best || boundary.position > boundaries[*best].position) {
      best = i;
    }
  }
  return best;
}

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_TURN_REUSE_H_
