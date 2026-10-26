// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Which positions a drafter stage's per-row feature tap (model::FeatureTap,
// CellRule::kPerRow) holds for a slot: host bookkeeping beside the tap's
// cells, which are the slot's state. Kept apart from model_runner.h so a
// state snapshot can carry a slot's without the runner's header.

#ifndef LLMP_ENGINE_TAP_ROWS_H_
#define LLMP_ENGINE_TAP_ROWS_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace llmp::engine {

// The tap's cells hold the features of positions [first, first + count),
// and the `preceding` positions before `first` in its carried tail.
struct TapRows {
  std::uint32_t first = 0, count = 0, preceding = 0;
  bool operator==(const TapRows&) const = default;
};

// One slot's tap bookkeeping on every stage at a committed position: what a
// state capture records beside the state, so that a restore of exactly that
// state lets the drafter catch up from the same features as at the capture
// (ModelRunner::SaveTapRows, RestoreTapRows). Fixed capacity: no allocation.
struct SlotTapRows {
  static constexpr std::size_t kStages = 16;  // spec::kMaxStages
  std::uint32_t stages = 0;                   // 0: nothing recorded
  std::uint32_t positions = 0;                // the slot's committed positions at the capture
  std::array<TapRows, kStages> held{}, previous{};
  bool operator==(const SlotTapRows&) const = default;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_TAP_ROWS_H_
