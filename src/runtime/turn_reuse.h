// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_RUNTIME_TURN_REUSE_H_
#define LLMP_RUNTIME_TURN_REUSE_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace llmp::runtime {

// Branch-local rollback cache. Prefix equality identifies reusable state,
// not a conversation or its lifetime (D-031). Independent shared-prefix
// retention and forks belong to the later request scheduler. Its bounds
// are real resources' (D-102): `[memory] retention_hours`, the disk of
// `[memory] spill_budget_gib` (shared with spilled conversations, which
// come first: Server::TrimTurnCheckpoints) and a conversation's kept
// record's format bound (kept::kMostCheckpoints).
// The default of `[memory] retention_hours` (D-055's idle cap, 7 days).
inline constexpr auto kTurnCheckpointRetention = std::chrono::hours(168);

// A checkpoint is optional: admit its complete padded file before writing
// it, beside every retained checkpoint (including active/held branches)
// and spilled conversation. Subtraction keeps even oversized inputs from
// wrapping into a false grant.
inline bool TurnCheckpointFits(std::uint64_t budget, std::uint64_t spilled,
                               std::uint64_t checkpoints, std::uint64_t extra) {
  return spilled <= budget && checkpoints <= budget - spilled &&
         extra <= budget - spilled - checkpoints;
}

// Delete eligible checkpoints until the combined disk usage and `extra`
// fit. Deletion can update a kept record, whose token allocation reclaims
// other idle state synchronously: it may spill a conversation or trim more
// checkpoints. Measure both totals again after every deletion, rather than
// subtracting its bytes from a snapshot. `drop` returns false when no
// eligible checkpoint remains.
template <typename Spilled, typename Checkpoints, typename Drop>
bool TrimTurnCheckpointBudget(std::uint64_t budget, std::uint64_t extra, Spilled spilled,
                              Checkpoints checkpoints, Drop drop) {
  if (extra > budget) return false;
  for (;;) {
    const std::uint64_t cached = checkpoints();
    if (TurnCheckpointFits(budget, spilled(), cached, extra)) return true;
    if (cached == 0 || !drop()) return false;
  }
}

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

inline std::optional<std::size_t> MatchingTurnBoundary(
    std::span<const TurnBoundary> boundaries, std::size_t common, std::size_t request_tokens,
    std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::duration retention = kTurnCheckpointRetention) {
  std::optional<std::size_t> best;
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    const TurnBoundary& boundary = boundaries[i];
    if (boundary.position == 0 || boundary.position > common ||
        boundary.position >= request_tokens || boundary.created > now ||
        now - boundary.created >= retention) {
      continue;
    }
    if (!best || boundary.position > boundaries[*best].position) {
      best = i;
    }
  }
  return best;
}

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_TURN_REUSE_H_
