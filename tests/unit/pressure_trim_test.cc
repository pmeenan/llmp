// SPDX-FileCopyrightText: 2026 jitLLM contributors
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

namespace rt = jitllm::runtime;
namespace pf = jitllm::platform;
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

}  // namespace
