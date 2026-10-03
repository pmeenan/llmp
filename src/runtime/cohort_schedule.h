// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Which unit a cooperative chat cohort runs next (docs/runtime-serving.md,
// "The chat route"): a prompt unit of one member, or a decode wave of every
// generating member. Vendor-free and host-only, so the CPU tests drive it;
// serve_api.cc keeps the counters and applies the choice.
//
// - Prompts: the one with the fewest prompt tokens left to prefill (after
//   what its conversation can reuse) goes next, the oldest of equals, so a
//   short prompt is not held behind a long one that arrived first, and
//   prompts of equal length finish one after another instead of all at the
//   end. A prompt passed over for kPromptAgeUnits other prompt units that
//   prefilled rows (a reuse or checkpoint unit prefills none: serve_api.cc)
//   goes next (the one passed over most, then the oldest), so a long prompt
//   is not starved while other members cycle short ones, unless the
//   shortest is a started prompt about to finish (kPromptFinishRows).
// - Decode: a generating member waits for at most one prompt unit: once one
//   has run since its last wave (or since its prompt ended), a wave of every
//   generating member runs before the next prompt unit. The wait is bounded
//   by a unit, not a clock: a DeepSeek chunk of 4,096 rows takes about 4.4
//   s, a Qwen3.8 one under 2 s.

#ifndef JITLLM_RUNTIME_COHORT_SCHEDULE_H_
#define JITLLM_RUNTIME_COHORT_SCHEDULE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace jitllm::runtime {

// A prompt waits for at most this many other prompt units.
inline constexpr std::uint32_t kPromptAgeUnits = 12;
// A started prompt with at most this many tokens left goes before an aged
// one: finishing it costs little and gives its first token.
inline constexpr std::uint32_t kPromptFinishRows = 256;

// One runnable cohort member, as the schedule sees it.
struct ScheduledMember {
  enum class Stage : std::uint8_t { kPrompt, kGeneration };
  Stage stage = Stage::kPrompt;
  std::uint64_t admitted = 0;   // admission order: smaller is older
  std::uint32_t remaining = 0;  // a prompt's tokens left to prefill
  std::uint32_t passed = 0;     // a prompt's: other prompt units since its own last
  std::uint32_t waited = 0;     // a generation's: prompt units since its last wave
  // A prompt's: started (its reuse settled) with at most kPromptFinishRows
  // tokens left.
  bool finishing = false;
};

struct ScheduleChoice {
  bool decode = false;                // a wave of every generating member
  std::optional<std::size_t> prompt;  // else this member's next prompt unit
};

// The next unit for `members` (runnable members only); neither a wave nor a
// prompt when there is none.
ScheduleChoice NextCohortUnit(std::span<const ScheduledMember> members);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_COHORT_SCHEDULE_H_
