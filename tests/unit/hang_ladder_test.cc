// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// D-102's hang ladder (runtime/hang_ladder.h) on a clock of the test's own:
// slow work that moves, or that its unit's allowance still covers, is never
// a hang; a hang with the driver in a node wait, or in CPU work, is
// cancelled (rung 1), and the ladder watches again once the driver moves
// and nothing the work submitted is still in flight; a cancellation that
// does not drain within its grace, a model that hangs again before serving
// a request, or a failed recovery restarts the process (rung 3), whose
// last resort runs once; idle, nothing is watched.

#include "runtime/hang_ladder.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/work_pulse.h"

namespace {

namespace rt = llmp::runtime;
using Clock = rt::HangLadder::Clock;
using Rung = rt::HangLadder::Rung;
using std::chrono::milliseconds;
using std::chrono::seconds;

constexpr auto kHang = seconds(600);

// A ladder whose last resort and log are recorded.
struct Recorded {
  explicit Recorded(Clock::time_point t0, std::chrono::milliseconds hang = kHang)
      : ladder(hang, t0) {
    ladder.set_last_resort([this](const std::string& why) { restarts.push_back(why); });
    ladder.set_log([this](std::string_view line) { lines.emplace_back(line); });
  }
  rt::HangLadder ladder;
  std::vector<std::string> restarts;
  std::vector<std::string> lines;
};

TEST(HangLadder, NothingIsWatchedWhileIdle) {
  const Clock::time_point t0{};
  Recorded r(t0);
  r.ladder.Unit(false, milliseconds(0), "idle", t0);
  EXPECT_FALSE(r.ladder.Check(t0 + std::chrono::hours(48)).has_value());
  EXPECT_EQ(r.ladder.rung(), Rung::kWatching);
  EXPECT_TRUE(r.restarts.empty());
}

// A unit that keeps moving (a swap from a slow disk landing reads) is
// never a hang, however long it takes; nor is one long job (a wide prefill
// chunk) within its allowance.
TEST(HangLadder, SlowWorkThatMovesOrIsAllowedIsNeverAHang) {
  const Clock::time_point t0{};
  Recorded r(t0);
  r.ladder.BeginWait();
  r.ladder.Unit(true, seconds(130), "swapping", t0);
  for (int minute = 1; minute <= 120; ++minute) {
    const auto now = t0 + std::chrono::minutes(minute);
    r.ladder.Activity(static_cast<std::uint64_t>(minute), now);
    EXPECT_FALSE(r.ladder.Check(now).has_value()) << minute;
  }
  // A prefill chunk expected to take 40 minutes at the floors: allowed it
  // (the stall time plus three times that), with nothing moving inside it.
  const auto chunk = t0 + std::chrono::hours(3);
  r.ladder.Unit(true, std::chrono::minutes(122), "prefilling", chunk);
  EXPECT_FALSE(r.ladder.Check(chunk + std::chrono::minutes(30)).has_value());
  EXPECT_FALSE(r.ladder.Check(chunk + std::chrono::minutes(121)).has_value());
  // Past its allowance, and quiet for `hang`: confirmed.
  EXPECT_EQ(r.ladder.Check(chunk + std::chrono::minutes(123)), Rung::kCancel);
  EXPECT_TRUE(r.restarts.empty());
}

// Rung 1: the driver waits on the node, so its wait cancels; the
// cancellation drains (the wait fails, and the node's count moves), and
// the ladder watches again.
TEST(HangLadder, ACancellationThatDrainsReturnsToWatching) {
  const Clock::time_point t0{};
  Recorded r(t0);
  r.ladder.Unit(true, seconds(120), "decoding", t0);
  r.ladder.BeginWait();
  r.ladder.Activity(7, t0);
  EXPECT_FALSE(r.ladder.Check(t0 + seconds(599)).has_value());
  const auto confirmed = t0 + seconds(600);
  EXPECT_EQ(r.ladder.Check(confirmed), Rung::kCancel);
  EXPECT_EQ(r.ladder.rung(), Rung::kCancel);
  EXPECT_EQ(r.ladder.cancelled_at(), confirmed);
  EXPECT_EQ(r.ladder.cancels(), 1U);
  // Asked again before anything moved: still rung 1, nothing new.
  EXPECT_FALSE(r.ladder.Check(confirmed + seconds(10)).has_value());
  // The wait drains: the node's count moves.
  r.ladder.Activity(8, confirmed + seconds(11));
  r.ladder.EndWait();
  EXPECT_EQ(r.ladder.Check(confirmed + seconds(12)), Rung::kWatching);
  EXPECT_EQ(r.ladder.drained(), 1U);
  EXPECT_TRUE(r.restarts.empty());
  ASSERT_EQ(r.lines.size(), 2U);
  EXPECT_NE(r.lines[0].find("rung 1"), std::string::npos);
  EXPECT_NE(r.lines[0].find("hang_seconds"), std::string::npos);
  // A second hang later is confirmed again.
  r.ladder.BeginWait();
  r.ladder.Unit(true, seconds(120), "decoding", confirmed + seconds(20));
  EXPECT_EQ(r.ladder.Check(confirmed + seconds(620)), Rung::kCancel);
  EXPECT_EQ(r.ladder.cancels(), 2U);
}

// Rung 3: a cancellation that has not drained within its grace, with
// nothing moving (a fence that never completes), restarts; once.
TEST(HangLadder, ACancellationThatDoesNotDrainRestartsOnce) {
  const Clock::time_point t0{};
  Recorded r(t0);
  EXPECT_EQ(r.ladder.cancel_grace(), rt::kCancelGrace);
  r.ladder.Unit(true, seconds(120), "prefilling", t0);
  r.ladder.BeginWait();
  const auto confirmed = t0 + seconds(600);
  ASSERT_EQ(r.ladder.Check(confirmed), Rung::kCancel);
  EXPECT_FALSE(r.ladder.Check(confirmed + seconds(59)).has_value());
  EXPECT_EQ(r.ladder.Check(confirmed + seconds(60)), Rung::kRestart);
  ASSERT_EQ(r.restarts.size(), 1U);
  EXPECT_NE(r.restarts[0].find("did not drain"), std::string::npos);
  EXPECT_NE(r.restarts[0].find("rung 3"), std::string::npos);
  EXPECT_FALSE(r.ladder.Check(confirmed + seconds(600)).has_value());
  r.ladder.Restart("again");
  EXPECT_EQ(r.restarts.size(), 1U);
  EXPECT_EQ(r.ladder.rung(), Rung::kRestart);
}

// The driver busy outside every node wait: CPU work that beats its pulse
// (a long rendering) is progress, never a hang; one that stops beating is
// asked to stop at its next checkpoint (rung 1), and drains when it does.
TEST(HangLadder, CpuWorkThatBeatsIsNotAHangAndOneThatStopsIsCancelled) {
  const Clock::time_point t0{};
  Recorded r(t0, seconds(60));
  r.ladder.Unit(true, seconds(10), "starting a request", t0);
  for (int s = 30; s <= 600; s += 30) {
    r.ladder.pulse().Beat();
    EXPECT_FALSE(r.ladder.Check(t0 + seconds(s)).has_value()) << s;
  }
  const auto quiet = t0 + seconds(600);
  EXPECT_FALSE(r.ladder.Check(quiet + seconds(59)).has_value());
  EXPECT_EQ(r.ladder.Check(quiet + seconds(60)), Rung::kCancel);
  EXPECT_TRUE(r.ladder.pulse().cancelled());
  EXPECT_EQ(r.ladder.pulse().cancels(), 1U);
  ASSERT_FALSE(r.lines.empty());
  EXPECT_NE(r.lines.back().find("CPU work is asked to stop"), std::string::npos);
  // The work reaches a checkpoint: its thread's pulse says stop.
  llmp::base::SetThreadPulse(&r.ladder.pulse());
  EXPECT_FALSE(llmp::base::Pulse());
  llmp::base::SetThreadPulse(nullptr);
  EXPECT_EQ(r.ladder.Check(quiet + seconds(61)), Rung::kWatching);
  EXPECT_FALSE(r.ladder.pulse().cancelled());
  EXPECT_TRUE(r.restarts.empty());
}

// CPU work that never reaches a checkpoint (a deadlock): rung 3 once the
// grace passes.
TEST(HangLadder, CpuWorkThatNeverStopsRestarts) {
  const Clock::time_point t0{};
  Recorded r(t0, seconds(60));
  r.ladder.Unit(true, seconds(10), "starting a request", t0);
  ASSERT_EQ(r.ladder.Check(t0 + seconds(60)), Rung::kCancel);
  EXPECT_FALSE(r.ladder.Check(t0 + seconds(119)).has_value());
  EXPECT_EQ(r.ladder.Check(t0 + seconds(120)), Rung::kRestart);
  ASSERT_EQ(r.restarts.size(), 1U);
  EXPECT_NE(r.restarts[0].find("did not drain"), std::string::npos);
}

// A stuck read (a drive's: the cancellation does not end it): the wait that
// needed it returns and the node moves, but the read stays in flight, so
// the cancellation has not drained; past the grace, rung 3 however much
// else moves. A read submitted after the cancellation is new work.
TEST(HangLadder, AStorageOperationStillInFlightKeepsTheCancellationUndrained) {
  const Clock::time_point t0{};
  Recorded r(t0, seconds(60));
  std::optional<Clock::time_point> oldest = t0 + seconds(1);
  r.ladder.set_operations([&oldest] { return oldest; });
  r.ladder.Unit(true, seconds(10), "prefilling", t0);
  r.ladder.BeginWait();
  const auto confirmed = t0 + seconds(60);
  ASSERT_EQ(r.ladder.Check(confirmed), Rung::kCancel);
  // The wait returns (the request's interest dropped) and other work moves.
  r.ladder.EndWait();
  for (int s = 1; s < 60; ++s) {
    r.ladder.Activity(static_cast<std::uint64_t>(s), confirmed + seconds(s));
    EXPECT_FALSE(r.ladder.Check(confirmed + seconds(s)).has_value()) << s;
  }
  EXPECT_EQ(r.ladder.Check(confirmed + seconds(60)), Rung::kRestart);
  ASSERT_EQ(r.restarts.size(), 1U);
  EXPECT_NE(r.restarts[0].find("still in flight"), std::string::npos);
  EXPECT_EQ(r.ladder.drained(), 0U);
}

TEST(HangLadder, AStorageOperationThatCompletesLetsItDrain) {
  const Clock::time_point t0{};
  Recorded r(t0, seconds(60));
  std::optional<Clock::time_point> oldest = t0 + seconds(1);
  r.ladder.set_operations([&oldest] { return oldest; });
  r.ladder.Unit(true, seconds(10), "prefilling", t0);
  r.ladder.BeginWait();
  const auto confirmed = t0 + seconds(60);
  ASSERT_EQ(r.ladder.Check(confirmed), Rung::kCancel);
  r.ladder.EndWait();
  r.ladder.Activity(1, confirmed + seconds(1));
  EXPECT_FALSE(r.ladder.Check(confirmed + seconds(2)).has_value());
  // The read completes; one submitted since is new work.
  oldest = confirmed + seconds(3);
  EXPECT_EQ(r.ladder.Check(confirmed + seconds(4)), Rung::kWatching);
  EXPECT_EQ(r.ladder.drained(), 1U);
  EXPECT_TRUE(r.restarts.empty());
}

// Rung 2 resets a model once per request served: a model that hangs again
// before serving one is not reset again (rung 3 instead).
TEST(HangLadder, AModelResetTwiceWithoutServingEscalates) {
  const Clock::time_point t0{};
  Recorded r(t0);
  EXPECT_FALSE(r.ladder.Resetting("qwen3.8").has_value());
  EXPECT_FALSE(r.ladder.Resetting("deepseek").has_value());
  r.ladder.Served("qwen3.8");
  EXPECT_FALSE(r.ladder.Resetting("qwen3.8").has_value());
  const auto again = r.ladder.Resetting("qwen3.8");
  ASSERT_TRUE(again.has_value());
  EXPECT_NE(again.value_or("").find("would loop"), std::string::npos);
  EXPECT_TRUE(r.ladder.Resetting("deepseek").has_value());
}

// The grace is the shorter of `hang` and kCancelGrace.
TEST(HangLadder, TheGraceIsAtMostTheHang) {
  const Clock::time_point t0{};
  Recorded r(t0, seconds(20));
  EXPECT_EQ(r.ladder.cancel_grace(), seconds(20));
}

// A node wait outside any unit (housekeeping between units, a teardown) is
// watched too, with no allowance but `hang`.
TEST(HangLadder, AWaitOutsideAnyUnitIsWatched) {
  const Clock::time_point t0{};
  Recorded r(t0);
  r.ladder.Unit(false, milliseconds(0), "idle", t0);
  r.ladder.BeginWait();
  EXPECT_FALSE(r.ladder.Check(t0 + seconds(599)).has_value());
  EXPECT_EQ(r.ladder.Check(t0 + seconds(600)), Rung::kCancel);
  // The wait ends with nothing else moving (a cancellation of a task with
  // nothing in flight): no work is under way, so it drained.
  r.ladder.EndWait();
  EXPECT_EQ(r.ladder.Check(t0 + seconds(601)), Rung::kWatching);
}

// Rung 2 is the runtime's: one that cannot recover the model restarts.
TEST(HangLadder, AFailedRecoveryRestarts) {
  const Clock::time_point t0{};
  Recorded r(t0);
  r.ladder.Restart("the model could not be reset in place: the stream did not fence");
  ASSERT_EQ(r.restarts.size(), 1U);
  EXPECT_NE(r.restarts[0].find("could not be reset"), std::string::npos);
  EXPECT_EQ(r.ladder.rung(), Rung::kRestart);
  ASSERT_FALSE(r.lines.empty());
  EXPECT_NE(r.lines.back().find("rung 3"), std::string::npos);
}

}  // namespace
