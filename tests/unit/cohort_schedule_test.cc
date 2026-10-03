// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The cooperative chat cohort's schedule (runtime/cohort_schedule.h): the
// shortest remaining prompt first, the oldest of equals; a long prompt
// passed over for at most kPromptAgeUnits prompt units; a generating
// member waiting for at most one prompt unit.

#include "runtime/cohort_schedule.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace jitllm::runtime {
namespace {

using Stage = ScheduledMember::Stage;

ScheduledMember Prompt(std::uint64_t admitted, std::uint32_t remaining, std::uint32_t passed = 0) {
  return {.stage = Stage::kPrompt,
          .admitted = admitted,
          .remaining = remaining,
          .passed = passed,
          .waited = 0,
          .finishing = false};
}

ScheduledMember Generating(std::uint64_t admitted, std::uint32_t waited) {
  return {.stage = Stage::kGeneration,
          .admitted = admitted,
          .remaining = 0,
          .passed = 0,
          .waited = waited,
          .finishing = false};
}

ScheduleChoice Next(const std::vector<ScheduledMember>& members) { return NextCohortUnit(members); }

// A cohort the backend drives: each prompt unit prefills up to `chunk`
// tokens of its member, and the counters move as serve_api.cc moves them.
class Simulated {
 public:
  explicit Simulated(std::vector<ScheduledMember> members) : members_(std::move(members)) {}

  // One unit; false when nothing is runnable.
  bool Step() {
    const ScheduleChoice c = NextCohortUnit(members_);
    if (c.decode) {
      waves_.push_back(order_.size());
      for (ScheduledMember& m : members_) {
        if (m.stage == Stage::kGeneration) {
          m.waited = 0;
        }
      }
      return true;
    }
    if (!c.prompt) {
      return false;
    }
    const std::size_t p = *c.prompt;
    order_.push_back(p);
    for (std::size_t i = 0; i < members_.size(); ++i) {
      ScheduledMember& m = members_[i];
      if (m.stage == Stage::kGeneration) {
        ++m.waited;
      } else if (i != p) {
        ++m.passed;
      }
    }
    ScheduledMember& run = members_[p];
    run.passed = 0;
    run.remaining -= std::min(run.remaining, kChunk);
    if (run.remaining == 0) {
      run.stage = Stage::kGeneration;
      run.waited = 0;
    }
    return true;
  }

  std::vector<ScheduledMember>& members() { return members_; }
  const std::vector<std::size_t>& order() const { return order_; }
  const std::vector<std::size_t>& waves() const { return waves_; }

 private:
  static constexpr std::uint32_t kChunk = 4096;
  std::vector<ScheduledMember> members_;
  std::vector<std::size_t> order_;  // prompt units, by member
  std::vector<std::size_t> waves_;  // the prompt units run before each wave
};

TEST(CohortSchedule, NothingRunnableChoosesNothing) {
  const ScheduleChoice c = Next({});
  EXPECT_FALSE(c.decode);
  EXPECT_FALSE(c.prompt.has_value());
}

TEST(CohortSchedule, TheShortestPromptGoesFirstTheOldestOfEquals) {
  // A long prompt that arrived first does not hold short ones behind it.
  EXPECT_EQ(Next({Prompt(1, 32000), Prompt(2, 126), Prompt(3, 120), Prompt(4, 126)}).prompt,
            std::optional<std::size_t>(2));
  EXPECT_EQ(Next({Prompt(3, 7043), Prompt(1, 7043), Prompt(2, 7043)}).prompt,
            std::optional<std::size_t>(1));
}

TEST(CohortSchedule, EqualPromptsFinishOneAfterAnother) {
  Simulated s({Prompt(1, 7043), Prompt(2, 7043), Prompt(3, 7043), Prompt(4, 7043)});
  for (int i = 0; i < 40 && s.order().size() < 8; ++i) {
    ASSERT_TRUE(s.Step());
  }
  EXPECT_EQ(s.order(), (std::vector<std::size_t>{0, 0, 1, 1, 2, 2, 3, 3}));
}

TEST(CohortSchedule, AGeneratingMemberWaitsForAtMostOnePromptUnit) {
  // Member 0 generates; three long prompts remain.
  Simulated s({Generating(1, 0), Prompt(2, 40000), Prompt(3, 40000), Prompt(4, 40000)});
  for (int i = 0; i < 60; ++i) {
    ASSERT_TRUE(s.Step());
  }
  ASSERT_FALSE(s.waves().empty());
  // Never two prompt units between waves.
  std::size_t last = 0;
  for (const std::size_t at : s.waves()) {
    EXPECT_LE(at - last, 1U);
    last = at;
  }
  // A member that has not waited a unit lets a prompt unit run first.
  EXPECT_EQ(Next({Generating(1, 0), Prompt(2, 10)}).prompt, std::optional<std::size_t>(1));
  EXPECT_TRUE(Next({Generating(1, 1), Prompt(2, 10)}).decode);
  EXPECT_TRUE(Next({Generating(1, 0)}).decode);
}

TEST(CohortSchedule, ALongPromptIsNotStarvedByShortOnes) {
  // A long prompt beside three slots that keep receiving short prompts:
  // each short one ends after its unit and another takes its slot.
  Simulated s({Prompt(1, 1000000), Prompt(2, 100), Prompt(3, 100), Prompt(4, 100)});
  std::uint64_t admitted = 4;
  std::size_t longest_gap = 0;
  std::size_t since = 0;
  for (int i = 0; i < 200; ++i) {
    ASSERT_TRUE(s.Step());
    since = s.order().back() == 0 ? 0 : since + 1;
    longest_gap = std::max(longest_gap, since);
    for (std::size_t m = 1; m < s.members().size(); ++m) {
      if (s.members()[m].stage == Stage::kGeneration) {
        s.members()[m] = Prompt(++admitted, 100);
      }
    }
  }
  EXPECT_LE(longest_gap, kPromptAgeUnits);
  EXPECT_GE(std::ranges::count(s.order(), std::size_t{0}), (200 / (kPromptAgeUnits + 1)) - 1);
  // The most passed-over prompt first, then the oldest.
  EXPECT_EQ(
      Next({Prompt(1, 50, kPromptAgeUnits), Prompt(2, 900, kPromptAgeUnits + 2), Prompt(3, 10)})
          .prompt,
      std::optional<std::size_t>(1));
  EXPECT_EQ(Next({Prompt(2, 900, kPromptAgeUnits), Prompt(1, 950, kPromptAgeUnits), Prompt(3, 10)})
                .prompt,
            std::optional<std::size_t>(1));
}

// A started prompt about to finish (its last few rows) goes before an aged
// one: aging bounds a wait in units, and this unit costs little and gives a
// first token. One not started yet (still to settle its reuse) does not.
TEST(CohortSchedule, AStartedPromptAboutToFinishGoesBeforeAnAgedOne) {
  ScheduledMember tail = Prompt(2, 2);
  tail.finishing = true;
  EXPECT_EQ(Next({Prompt(1, 4096, kPromptAgeUnits), tail}).prompt, std::optional<std::size_t>(1));
  EXPECT_EQ(Next({Prompt(1, 4096, kPromptAgeUnits), Prompt(2, 2)}).prompt,
            std::optional<std::size_t>(0));
  EXPECT_LE(kPromptFinishRows, 512U);
}

}  // namespace
}  // namespace jitllm::runtime
