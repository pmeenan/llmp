// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "memory/reclaim.h"
#include "runtime/serving.h"

namespace llmp::runtime {

std::optional<double> SpillRoomCost(std::vector<SpillVictim> victims, std::uint64_t spilled,
                                    std::uint64_t checkpoints, std::uint64_t budget,
                                    std::uint64_t extra) {
  std::ranges::sort(victims, {},
                    [](const SpillVictim& v) { return std::pair(v.conversation, v.used); });
  double seconds = 0;
  for (const SpillVictim& v : victims) {
    if (TurnCheckpointFits(budget, spilled, checkpoints, extra)) {
      return seconds;
    }
    spilled -= std::min(spilled, v.counted);
    seconds += v.seconds;
  }
  if (TurnCheckpointFits(budget, spilled, checkpoints, extra)) {
    return seconds;
  }
  return std::nullopt;
}

void AddIdleStateCandidates(Llm& model, std::uint32_t owner, bool running, const Llm::Branch* spare,
                            const IdleStateRates& rates,
                            std::vector<memory::ReclaimCandidate>& out) {
  for (std::size_t slot = 0; slot < model.branches(); ++slot) {
    auto b = model.branch(slot);
    if (!b || *b == spare || !model.BranchIdle(**b) || (*b)->history().empty()) {
      continue;
    }
    const std::uint64_t bytes = model.ResidentStateBytes(**b);
    if (bytes == 0) {
      continue;
    }
    // Priced as what its take does (Server::Reclaim): spilled when the
    // budget makes room for it, the spilled conversations that room deletes
    // charged to it; otherwise dropped, which a continuation holding it
    // refuses.
    std::optional<double> room;
    if (bytes <= rates.spill_budget) {
      room = rates.spill_room ? rates.spill_room(**b, bytes) : std::optional<double>(0.0);
    }
    const bool dropped = !room.has_value();
    if (dropped && (*b)->HeldByContinuation()) {
      continue;
    }
    // A spill writes only what changed since its last one; a restore reads
    // it all.
    const auto size = static_cast<double>(bytes);
    const auto writes = static_cast<double>(model.SpillWriteBytes(**b));
    out.push_back(
        {.kind = memory::ReclaimKind::kIdleState,
         .owner = owner,
         .id = slot,
         .bytes = bytes,
         .last_use = static_cast<std::uint64_t>(model.LastUsed(**b).time_since_epoch().count()),
         .restore_seconds = dropped
                                ? static_cast<double>((*b)->history().size()) *
                                      model.settings().recompute_ms_per_token.value / 1000.0
                                : (writes / rates.spill_rate) + (size / rates.restore_rate) + *room,
         .running = running});
    memory::SetUse(out.back(), model.LastStamp(**b));
  }
}

bool RoomFor(std::uint64_t needed, std::optional<std::uint64_t> free,
             const std::function<std::uint64_t(std::uint64_t)>& reclaim) {
  if (needed == 0 || !free || *free >= needed) {
    return true;
  }
  return reclaim(needed - *free) >= needed - *free;
}

}  // namespace llmp::runtime
