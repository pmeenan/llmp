// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Room for a swap's incoming model (Server::MakeRoomForSwap): what the
// incoming model pages in beyond what the outgoing one gives back and the
// budget's free room is reclaimed first, and the room checked again after
// each reclaim, because what a reclaim reports freed need not be what the
// budget's occupancy dropped by (the plans' and graphs' charge moves in
// whole extents past its floor). Vendor-free and host-only, so the CPU
// tests drive it with a ledger of their own.

#ifndef JITLLM_RUNTIME_SWAP_ROOM_H_
#define JITLLM_RUNTIME_SWAP_ROOM_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <string>

namespace jitllm::runtime {

struct SwapRoomSteps {
  int reclaims = 0;         // reclaims asked
  std::uint64_t asked = 0;  // bytes asked in all
  std::uint64_t freed = 0;  // bytes the reclaims reported
};

// Until `shortfall()` (the bytes the swap lacks now) is 0: asks `reclaim`
// for it rounded up to whole `unit`s, then checks again, at most
// `attempts` times. Refused (the swap must not start) when a reclaim
// frees nothing or the attempts run out with room still short.
std::expected<SwapRoomSteps, std::string> MakeRoom(
    const std::function<std::uint64_t()>& shortfall,
    const std::function<std::uint64_t(std::uint64_t)>& reclaim, std::uint64_t unit,
    int attempts = 4);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_SWAP_ROOM_H_
