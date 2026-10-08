// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/cohort_schedule.h"

#include <cstddef>
#include <optional>
#include <span>

namespace llmp::runtime {

ScheduleChoice NextCohortUnit(std::span<const ScheduledMember> members) {
  using Stage = ScheduledMember::Stage;
  bool generating = false;
  bool due = false;
  std::optional<std::size_t> shortest;
  std::optional<std::size_t> aged;
  for (std::size_t i = 0; i < members.size(); ++i) {
    const ScheduledMember& m = members[i];
    if (m.stage == Stage::kGeneration) {
      generating = true;
      due = due || m.waited >= 1;
      continue;
    }
    const auto older = [&](std::optional<std::size_t> other) {
      return !other || m.admitted < members[*other].admitted;
    };
    if (!shortest || m.remaining < members[*shortest].remaining ||
        (m.remaining == members[*shortest].remaining && older(shortest))) {
      shortest = i;
    }
    if (m.passed >= kPromptAgeUnits && (!aged || m.passed > members[*aged].passed ||
                                        (m.passed == members[*aged].passed && older(aged)))) {
      aged = i;
    }
  }
  if (generating && (due || !shortest)) {
    return {.decode = true, .prompt = std::nullopt};
  }
  // A started prompt about to finish goes before an aged one.
  const bool finishing = shortest && members[*shortest].finishing;
  return {.decode = false, .prompt = aged && !finishing ? aged : shortest};
}

}  // namespace llmp::runtime
