// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/swap_room.h"

#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <string>

namespace jitllm::runtime {

std::expected<SwapRoomSteps, std::string> MakeRoom(
    const std::function<std::uint64_t()>& shortfall,
    const std::function<std::uint64_t(std::uint64_t)>& reclaim, std::uint64_t unit, int attempts) {
  SwapRoomSteps steps;
  for (;;) {
    const std::uint64_t short_by = shortfall();
    if (short_by == 0) {
      return steps;
    }
    if (steps.reclaims >= attempts) {
      return std::unexpected(
          std::format("{} bytes short of room for the incoming model after {} reclaims", short_by,
                      steps.reclaims));
    }
    const std::uint64_t ask = unit == 0 ? short_by : (short_by + unit - 1) / unit * unit;
    const std::uint64_t freed = reclaim(ask);
    ++steps.reclaims;
    steps.asked += ask;
    steps.freed += freed;
    if (freed == 0) {
      return std::unexpected(std::format(
          "{} bytes short of room for the incoming model, and nothing left to reclaim", short_by));
    }
  }
}

}  // namespace jitllm::runtime
