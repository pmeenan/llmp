// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// When pressure on memory from outside has the runtime trim
// (runtime/pressure_trim.h), from readings of the test's own: one trim asks
// for the target headroom, nothing more is trimmed between the mark and the
// target, pressure that persists is trimmed at a growing back-off (never
// every look), too little left to give waits the longest and logs at most
// once a log interval, and the back-off resets when pressure ends.

#include "runtime/pressure_trim.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "platform/memory_pressure.h"

namespace {

namespace rt = llmp::runtime;
namespace pf = llmp::platform;
using Clock = rt::PressureTrim::Clock;
using std::chrono::milliseconds;
using std::chrono::seconds;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr auto kLook = milliseconds(250);  // Server::Maintain's interval

pf::MemoryPressure Available(std::uint64_t bytes) { return {.available = bytes, .stall = {}}; }

TEST(PressureTrim, NothingIsTrimmedWithoutPressure) {
  rt::PressureTrim trim;
  const Clock::time_point t0{};
  EXPECT_FALSE(trim.Observe(Available(4096 * kMiB), t0).now);
  EXPECT_FALSE(trim.Observe(Available(513 * kMiB), t0 + kLook).now);
  EXPECT_FALSE(trim.Observe({}, t0 + (2 * kLook)).now);  // nothing readable
  EXPECT_FALSE(trim.Observe({.available = {},
                             .stall = pf::PressureStall{.some_avg10 = 50, .full_avg10 = 4.9}},
                            t0 + (3 * kLook))
                   .now);
  // A full stall alone, with memory available (a compile or a copy on the
  // host), or with nothing known of it, is not pressure on the runtime.
  for (const std::uint64_t mib : {std::uint64_t{1536}, std::uint64_t{4096}, std::uint64_t{35000}}) {
    EXPECT_FALSE(trim.Observe({.available = mib * kMiB,
                               .stall = pf::PressureStall{.some_avg10 = 60, .full_avg10 = 23}},
                              t0 + (4 * kLook))
                     .now);
  }
  EXPECT_FALSE(trim.Observe({.available = {},
                             .stall = pf::PressureStall{.some_avg10 = 60, .full_avg10 = 23}},
                            t0 + (5 * kLook))
                   .now);
  EXPECT_FALSE(trim.pressed());
}

TEST(PressureTrim, OneTrimAsksForTheTargetHeadroom) {
  rt::PressureTrim trim;
  const auto t = trim.Observe(Available(300 * kMiB), Clock::time_point{});
  ASSERT_TRUE(t.now);
  // To the low mark (512 MiB) plus the headroom (1 GiB), in one reclaim.
  EXPECT_EQ(t.needed, (512 + 1024 - 300) * kMiB);
  EXPECT_FALSE(t.why.empty());
  // A full stall raises the trigger to the target: with 1 GiB available
  // (above the mark) it asks for what restores the target.
  rt::PressureTrim stalled;
  const auto s = stalled.Observe(
      {.available = 1024 * kMiB, .stall = pf::PressureStall{.some_avg10 = 20, .full_avg10 = 6}},
      Clock::time_point{});
  ASSERT_TRUE(s.now);
  EXPECT_EQ(s.needed, 512 * kMiB);
}

TEST(PressureTrim, PressureEndsOnlyBackAtTheTarget) {
  rt::PressureTrim trim;
  Clock::time_point now{};
  auto t = trim.Observe(Available(300 * kMiB), now);
  ASSERT_TRUE(t.now);
  EXPECT_FALSE(trim.Trimmed(t.needed, t.needed, now));
  // Above the mark, under the target: still pressed, nothing trimmed.
  for (int i = 0; i < 40; ++i) {
    now += kLook;
    EXPECT_FALSE(trim.Observe(Available(900 * kMiB), now).now);
    EXPECT_TRUE(trim.pressed());
  }
  // Back at the target: pressure ends and its back-off resets.
  now += kLook;
  EXPECT_FALSE(trim.Observe(Available(1536 * kMiB), now).now);
  EXPECT_FALSE(trim.pressed());
  EXPECT_EQ(trim.backoff(), Clock::duration{});
  // New pressure trims at once.
  now += kLook;
  EXPECT_TRUE(trim.Observe(Available(100 * kMiB), now).now);
}

// Another process takes back everything the runtime gives (available stays
// under the mark): trims are spaced by a back-off doubling to its most, not
// made every look, so what the running model uses is not dropped and built
// again four times a second.
TEST(PressureTrim, PressureThatPersistsIsTrimmedAtAGrowingBackOff) {
  rt::PressureTrim trim;
  Clock::time_point now{};
  std::vector<Clock::time_point> at;
  for (int look = 0; look < 4 * 60 * 10; ++look) {  // ten minutes
    const auto t = trim.Observe(Available(300 * kMiB), now);
    if (t.now) {
      at.push_back(now);
      EXPECT_FALSE(trim.Trimmed(t.needed, t.needed, now));  // what it asked for, given
    }
    now += kLook;
  }
  // At 0, 1, 3, 7, 15, 31 and 63 s, then every 60 s: 15 in ten minutes.
  ASSERT_GE(at.size(), 3U);
  EXPECT_EQ(at.size(), 15U);
  for (std::size_t i = 1; i < at.size(); ++i) {
    const auto gap = at[i] - at[i - 1];
    EXPECT_GE(gap, seconds(1));
    EXPECT_LE(gap, seconds(60) + kLook);
    if (i >= 2) {
      EXPECT_GE(gap, at[i - 1] - at[i - 2]);  // never closer than the last
    }
  }
  EXPECT_EQ(at[1] - at[0], seconds(1));
  EXPECT_EQ(at[2] - at[1], seconds(2));
  EXPECT_EQ(trim.backoff(), seconds(60));
  EXPECT_EQ(trim.short_trims(), 0U);
}

// Nothing left to give (the running model's floor is all there is): the
// longest wait at once, and one log line a log interval at most.
TEST(PressureTrim, TooLittleLeftWaitsTheLongestAndLogsOnceAnInterval) {
  rt::PressureTrim trim;
  Clock::time_point now{};
  int trims = 0;
  int logs = 0;
  for (int look = 0; look < 4 * 60 * 10; ++look) {  // ten minutes
    const auto t = trim.Observe(Available(200 * kMiB), now);
    if (t.now) {
      ++trims;
      logs += trim.Trimmed(0, t.needed, now) ? 1 : 0;
      EXPECT_EQ(trim.backoff(), seconds(60));
    }
    now += kLook;
  }
  EXPECT_EQ(trims, 10);  // at 0, 60, ... 540 s
  EXPECT_EQ(logs, 10);   // once each minute, no more
  EXPECT_EQ(trim.short_trims(), 10U);
  // A shorter log interval than the back-off changes nothing; a longer one
  // logs less.
  rt::PressureTrim quiet({.log_interval = seconds(300)});
  now = Clock::time_point{};
  logs = 0;
  for (int look = 0; look < 4 * 60 * 10; ++look) {
    const auto t = quiet.Observe(Available(200 * kMiB), now);
    if (t.now) {
      logs += quiet.Trimmed(0, t.needed, now) ? 1 : 0;
    }
    now += kLook;
  }
  EXPECT_EQ(logs, 2);  // at 0 and 300 s
}

// A short trim after full ones, then relief: the back-off and the log
// start over with the next pressure.
TEST(PressureTrim, ReliefStartsTheBackOffOver) {
  rt::PressureTrim trim;
  Clock::time_point now{};
  auto t = trim.Observe(Available(300 * kMiB), now);
  ASSERT_TRUE(t.now);
  EXPECT_TRUE(trim.Trimmed(10 * kMiB, t.needed, now));
  now += seconds(5);
  EXPECT_FALSE(trim.Observe(Available(300 * kMiB), now).now);  // within the longest wait
  EXPECT_FALSE(trim.Observe(Available(4096 * kMiB), now).now);
  EXPECT_FALSE(trim.pressed());
  now += kLook;
  t = trim.Observe(Available(300 * kMiB), now);
  ASSERT_TRUE(t.now);
  EXPECT_EQ(trim.backoff(), seconds(1));
  EXPECT_TRUE(trim.Trimmed(0, t.needed, now));  // a new episode logs again
}

TEST(PressureTrim, AdjustSharedBudgetShrinksOnlyUnderPressureAndRestoresInSteps) {
  constexpr std::uint64_t kFloor = 256 * kMiB;
  constexpr std::uint64_t kCeiling = 8192 * kMiB;
  constexpr std::uint64_t kTarget = 1536 * kMiB;
  constexpr std::uint64_t kCurrentUse = 2048 * kMiB;

  // When not under pressure, a temporary dip in available memory below target
  // does not ratcheting-shrink the budget.
  EXPECT_EQ(rt::AdjustSharedBudget(1200 * kMiB, /*pressed=*/false, kCurrentUse, kFloor, kCeiling,
                                   kCeiling),
            kCeiling);

  // Under pressure with available below target, the budget is the occupancy
  // less the shortfall: what restores the target once the runtime gives it
  // back (2048 - (1536 - 1036) = 1548 MiB), never the occupancy plus some.
  const std::uint64_t shrunk = rt::AdjustSharedBudget(1036 * kMiB, /*pressed=*/true, kCurrentUse,
                                                      kFloor, kCeiling, kCeiling);
  EXPECT_EQ(shrunk, kCurrentUse - (kTarget - 1036 * kMiB));
  EXPECT_EQ(shrunk, 1548 * kMiB);
  EXPECT_LT(shrunk, kCurrentUse);
  // Never raised while pressed, even when the occupancy has since grown.
  EXPECT_EQ(
      rt::AdjustSharedBudget(1036 * kMiB, /*pressed=*/true, 4096 * kMiB, kFloor, shrunk, kCeiling),
      shrunk);

  // The floor holds against any shortfall: the running model's footprint.
  EXPECT_EQ(
      rt::AdjustSharedBudget(1 * kMiB, /*pressed=*/true, 300 * kMiB, kFloor, kCeiling, kCeiling),
      kFloor);

  // While available is above the low mark (512 MiB) but still below target
  // (1536 MiB), the budget does not restore to ceiling prematurely.
  EXPECT_EQ(
      rt::AdjustSharedBudget(900 * kMiB, /*pressed=*/false, kCurrentUse, kFloor, shrunk, kCeiling),
      shrunk);

  // Exactly at the target nothing is spare: unchanged.
  EXPECT_EQ(
      rt::AdjustSharedBudget(kTarget, /*pressed=*/false, kCurrentUse, kFloor, shrunk, kCeiling),
      shrunk);

  // Past the target, it rises by what is available past it, a step at a
  // time, and stops at the configured ceiling.
  const std::uint64_t raised = rt::AdjustSharedBudget(kTarget + 512 * kMiB, /*pressed=*/false,
                                                      kCurrentUse, kFloor, shrunk, kCeiling);
  EXPECT_EQ(raised, shrunk + 512 * kMiB);
  EXPECT_EQ(rt::AdjustSharedBudget(kTarget + 64 * 1024 * kMiB, /*pressed=*/false, kCurrentUse,
                                   kFloor, raised, kCeiling),
            kCeiling);

  // Unknown availability changes nothing.
  EXPECT_EQ(
      rt::AdjustSharedBudget(std::nullopt, /*pressed=*/true, kCurrentUse, kFloor, shrunk, kCeiling),
      shrunk);
}

}  // namespace

// The shared floor follows what the requests hold, not a context's worth:
// with nothing held it is the fixed memory and the largest footprint, it
// rises as their live state grows, and it never passes the configured
// budget (so shared mode can always give back what is not held).
TEST(PressureTrim, TheSharedFloorHoldsWhatTheRequestsHoldAndNoMore) {
  constexpr std::uint64_t kBase = 1024 * kMiB;  // fixed + largest footprint
  constexpr std::uint64_t kCeiling = 8192 * kMiB;
  EXPECT_EQ(rt::SharedBudgetFloor(kBase, 0, kCeiling), kBase);
  EXPECT_EQ(rt::SharedBudgetFloor(kBase, 300 * kMiB, kCeiling), kBase + 300 * kMiB);
  EXPECT_EQ(rt::SharedBudgetFloor(kBase, 64 * 1024 * kMiB, kCeiling), kCeiling);
  EXPECT_EQ(rt::SharedBudgetFloor(UINT64_MAX, UINT64_MAX, kCeiling), kCeiling);

  // Under heavy pressure the shrink stops at the held state above the base:
  // a running request keeps what it holds.
  const std::uint64_t held = 700 * kMiB;
  const std::uint64_t floor = rt::SharedBudgetFloor(kBase, held, kCeiling);
  EXPECT_EQ(
      rt::AdjustSharedBudget(1 * kMiB, /*pressed=*/true, 2048 * kMiB, floor, kCeiling, kCeiling),
      kBase + held);
  // Once the requests let it go, the same pressure shrinks to the base.
  EXPECT_EQ(
      rt::AdjustSharedBudget(1 * kMiB, /*pressed=*/true, 2048 * kMiB,
                             rt::SharedBudgetFloor(kBase, 0, kCeiling), kBase + held, kCeiling),
      kBase);
}

// A held request's growth that the reclaim order could not cover is lent
// the shrunk budget back, by the shortfall, never past the configured one.
TEST(PressureTrim, ARefusedGrowthLiftsTheSharedBudgetWithinTheConfiguredOne) {
  constexpr std::uint64_t kCeiling = 8192 * kMiB;
  EXPECT_EQ(rt::LiftSharedBudget(2048 * kMiB, kCeiling, 64 * kMiB), 2112 * kMiB);
  EXPECT_EQ(rt::LiftSharedBudget(8000 * kMiB, kCeiling, 1024 * kMiB), kCeiling);
  EXPECT_EQ(rt::LiftSharedBudget(kCeiling, kCeiling, 64 * kMiB), kCeiling);
  EXPECT_EQ(rt::LiftSharedBudget(2048 * kMiB, kCeiling, UINT64_MAX), kCeiling);
  EXPECT_EQ(rt::LiftSharedBudget(2048 * kMiB, kCeiling, 0), 2048 * kMiB);
  // Never lowered (a budget above the configured one is a diagnostic's).
  EXPECT_EQ(rt::LiftSharedBudget(9000 * kMiB, kCeiling, 64 * kMiB), 9000 * kMiB);
}

// A lift capped at the configured budget counts as the whole growth: the
// growth asks for all its bytes, not what it fell short of the headroom
// left, and the configured budget may hold it although the lift fell short
// of them (a request failed for the shrink alone). Short of the configured
// budget, or with nothing lent, only what was given counts; the retry's own
// refusal at the configured budget then stands.
TEST(PressureTrim, ALiftToTheConfiguredBudgetRelievesTheWholeGrowth) {
  constexpr std::uint64_t kCeiling = 8192 * kMiB;
  // 40 MiB lent to the ceiling for a 64 MiB growth with 32 MiB of headroom.
  EXPECT_EQ(rt::LiftRelief(64 * kMiB, 0, kCeiling - 40 * kMiB, kCeiling, kCeiling), 64 * kMiB);
  EXPECT_EQ(rt::LiftRelief(64 * kMiB, 8 * kMiB, kCeiling - 40 * kMiB, kCeiling, kCeiling),
            64 * kMiB);
  // Short of the ceiling: what was freed and lent.
  EXPECT_EQ(rt::LiftRelief(64 * kMiB, 8 * kMiB, 2048 * kMiB, 2104 * kMiB, kCeiling), 64 * kMiB);
  EXPECT_EQ(rt::LiftRelief(64 * kMiB, 0, 2048 * kMiB, 2080 * kMiB, kCeiling), 32 * kMiB);
  // Nothing lent (already at the ceiling): only the reclaim order's answer.
  EXPECT_EQ(rt::LiftRelief(64 * kMiB, 16 * kMiB, kCeiling, kCeiling, kCeiling), 16 * kMiB);
  EXPECT_EQ(rt::LiftRelief(64 * kMiB, UINT64_MAX, 0, kCeiling, kCeiling), UINT64_MAX);
}
