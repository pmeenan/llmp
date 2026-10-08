// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The runtime wake (docs/experiments/runtime-wake/) on the fake device:
// the device service's two lanes on their own threads, as a program runs
// them, with the test as the board's owner and as the device, which
// completes each fence when it chooses. The completion lane sleeps
// through most of a fence's expected length and spins through its end;
// the submission lane sleeps between commands. Whatever the timing, every
// fence is seen and published (no lost wakeup), one that ends long before
// its expectation is seen within the backstop, the owner and the
// submission lane are told ahead of an expected end, a sleeping lane
// wakes for a command and for its close, and a close with a fence out
// waits for it. These run under ThreadSanitizer with the scheduler's.

#include "base/wake.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "providers/device_execution.h"
#include "providers/fake/fake_device_execution.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/services.h"

namespace {

using llmp::base::PushResult;
using llmp::base::WakeFlag;
using llmp::providers::NativeStream;
using llmp::providers::ProviderError;
using llmp::providers::StreamId;
using llmp::scheduler::Acceptance;
using llmp::scheduler::CompletionBoard;
using llmp::scheduler::DeviceCommand;
using llmp::scheduler::DeviceService;
using llmp::scheduler::DeviceSettings;
using llmp::scheduler::JobResult;
using llmp::scheduler::LaunchWork;
using llmp::scheduler::OperationId;
using llmp::scheduler::Outcome;
using llmp::scheduler::Terminal;
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

constexpr auto kPatience = std::chrono::seconds(60);  // a hang, not a timing

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

class DeviceWakeTest : public ::testing::Test {
 protected:
  void SetUp() override { stream_ = execution_.CreateStream().value(); }

  void TearDown() override {
    if (lane_ != nullptr) {
      lane_->Close();
      // Anything still queued or watched completes, so both lanes can return.
      while (!submission_done_.load() || !completion_done_.load()) {
        (void)execution_.Step(stream_);
        std::this_thread::sleep_for(milliseconds(1));
      }
      submission_.join();
      completion_.join();
    }
  }

  void Start(DeviceSettings settings = {}) {
    lane_ = std::make_unique<DeviceService>(execution_, std::span<const StreamId>(&stream_, 1),
                                            board_, settings);
    submission_ = std::jthread([this] {
      lane_->RunSubmission();
      submission_done_.store(true);
    });
    completion_ = std::jthread([this] {
      lane_->RunCompletion();
      completion_done_.store(true);
    });
  }

  // An operation whose job queues nothing but the lane's fence; `ran`, if
  // given, is set when the job runs on the submission lane.
  OperationId Launch(std::atomic<bool>* ran = nullptr) {
    const OperationId operation = board_.Open();
    EXPECT_TRUE(operation.valid());
    DeviceCommand command{.operation = operation,
                          .work = LaunchWork{.stream = 0, .job = [ran](NativeStream /*native*/) {
                                               if (ran != nullptr) {
                                                 ran->store(true);
                                               }
                                               return JobResult::kQueued;
                                             }}};
    EXPECT_EQ(lane_->Submit(std::move(command)), PushResult::kAccepted);
    return operation;
  }

  // The device: completes the next fence once the lane has recorded it.
  void CompleteNext() {
    const auto give_up = Clock::now() + kPatience;
    while (!execution_.Step(stream_)) {
      ASSERT_LT(Clock::now(), give_up) << "no fence was recorded";
      std::this_thread::yield();
    }
  }

  // The owner: harvests until `operation` has its terminal result, then
  // closes it if it is proven (an unproven one keeps its mailbox, as the
  // scheduler's quarantine does). Nothing if the patience runs out.
  std::optional<Terminal> Await(OperationId operation) {
    const auto give_up = Clock::now() + kPatience;
    while (Clock::now() < give_up) {
      (void)wake_.Consume();
      for (const auto& seen : board_.Harvest(64)) {
        if (seen.operation == operation && seen.terminal && seen.acceptance != Acceptance::kNone) {
          EXPECT_EQ(board_.Close(operation), seen.terminal->no_further_access);
          return seen.terminal;
        }
      }
      (void)wake_.WaitFor(milliseconds(10));
    }
    return std::nullopt;
  }

  // Runs one fence of `length`: launched, completed after `length`, awaited.
  void Learn(Clock::duration length) {
    const OperationId operation = Launch();
    std::this_thread::sleep_for(length);
    CompleteNext();
    ASSERT_TRUE(Await(operation).has_value());
  }

  llmp::providers::fake::FakeDeviceExecution execution_;
  WakeFlag wake_;
  CompletionBoard board_{512, wake_};
  StreamId stream_;
  std::unique_ptr<DeviceService> lane_;
  std::atomic<bool> submission_done_{false};
  std::atomic<bool> completion_done_{false};
  std::jthread submission_;
  std::jthread completion_;
};

// Fences that end on time, early, late and at once, in an order that keeps
// the expectation wrong half the time: each is seen, published with its
// proof, and closed.
TEST_F(DeviceWakeTest, EveryFenceIsSeenWhateverItsTiming) {
  Start();
  constexpr std::array<int, 16> kDelays = {5, 5, 5, 5, 0, 0, 30, 5, 1, 20, 0, 5, 2, 40, 0, 3};
  for (const int delay : kDelays) {
    const OperationId operation = Launch();
    std::this_thread::sleep_for(milliseconds(delay));
    CompleteNext();
    const std::optional<Terminal> terminal = Await(operation);
    ASSERT_TRUE(terminal.has_value()) << "a fence of " << delay << " ms was lost";
    const Terminal got = terminal.value_or(Terminal{});
    EXPECT_EQ(got.outcome, Outcome::kSucceeded);
    EXPECT_TRUE(got.no_further_access);
  }
  EXPECT_EQ(board_.open(), 0U);
}

// Expecting 100 ms fences, the lane sleeps through most of the next; one
// that ends at once is still seen at the lane's next query, within its
// backstop (1 ms by default), not at the expected end.
TEST_F(DeviceWakeTest, AFenceEndingFarAheadOfItsExpectationIsSeenWithinTheBackstop) {
  Start();
  for (int i = 0; i < 3; ++i) {
    Learn(milliseconds(100));
  }
  const OperationId operation = Launch();
  CompleteNext();
  const auto completed = Clock::now();
  ASSERT_TRUE(Await(operation).has_value());
  EXPECT_LT(Ms(Clock::now() - completed), 50.0);
}

// A long step after short ones: each fence outlasts every recent length
// (5 ms, then each longer than the last), so the lane backs off past its
// expectation, querying at least every backstop (1 ms), and sees the fence
// soon after it completes, never a tick (100 ms) or its sleep's end later.
TEST_F(DeviceWakeTest, AFenceEndingLongAfterItsExpectationIsSeenWithinTheBackstop) {
  Start();
  for (int i = 0; i < 3; ++i) {
    Learn(milliseconds(5));
  }
  std::vector<double> seen;
  for (const int length : {60, 130, 250}) {
    const OperationId operation = Launch();
    std::this_thread::sleep_for(milliseconds(length));
    CompleteNext();
    const auto completed = Clock::now();
    const std::optional<Terminal> terminal = Await(operation);
    ASSERT_TRUE(terminal.has_value()) << "a fence of " << length << " ms was lost";
    EXPECT_TRUE(terminal.value_or(Terminal{}).no_further_access);
    seen.push_back(Ms(Clock::now() - completed));
  }
  std::ranges::sort(seen);
  EXPECT_LT(seen[seen.size() / 2], 20.0);
}

// A query whose outcome is unknown (a device fault), or that the device
// keeps refusing, while the lane sleeps through a fence expected to take
// 100 ms: the lane's next queries, at least every backstop, find it, and
// the work is published unproven long before the expected end, although
// the fence never completed (D-048: an anticipation proves nothing, and
// neither does the lack of one).
TEST_F(DeviceWakeTest, AnUnknownOrPersistentlyRefusedQueryIsPublishedUnproven) {
  Start();
  for (int i = 0; i < 3; ++i) {
    Learn(milliseconds(100));
  }
  // No other fence is watched: the faults are this operation's queries.
  for (const auto [error, times] :
       {std::pair(ProviderError::kUnknown, std::size_t{1}),
        std::pair(ProviderError::kFailed, std::size_t{DeviceSettings{}.refusals})}) {
    execution_.FailNextQuery(error, times);
    const auto launched = Clock::now();
    const OperationId operation = Launch();
    const std::optional<Terminal> terminal = Await(operation);
    ASSERT_TRUE(terminal.has_value());
    const Terminal got = terminal.value_or(Terminal{});
    EXPECT_EQ(got.outcome, Outcome::kFailed);
    EXPECT_FALSE(got.no_further_access) << "published as proven without a complete query";
    EXPECT_LT(Ms(Clock::now() - launched), 50.0);
  }
  // The abandoned fences are never queried again; a later one still is.
  const OperationId operation = Launch();
  CompleteNext();
  CompleteNext();
  CompleteNext();
  const std::optional<Terminal> terminal = Await(operation);
  ASSERT_TRUE(terminal.has_value());
  EXPECT_TRUE(terminal.value_or(Terminal{}).no_further_access);
}

// As a fence nears its expected end, the completion lane has the owner and
// the submission lane poll through it, before the fence completes: the
// relay that hides their wakeups. It slept first: the relay comes about
// spin_ahead before the expected end, not at the launch.
TEST_F(DeviceWakeTest, AFenceNearingItsEndHasTheOwnerAndTheSubmissionLanePollAhead) {
  DeviceSettings settings;
  settings.spin_ahead = std::chrono::microseconds(10000);
  Start(settings);
  for (int i = 0; i < 3; ++i) {
    Learn(milliseconds(40));
  }
  std::this_thread::sleep_for(milliseconds(20));  // past every earlier relay's end
  ASSERT_FALSE(wake_.Anticipating(Clock::now()));
  const auto launched = Clock::now();
  const OperationId operation = Launch();
  const auto give_up = launched + std::chrono::seconds(5);
  while (!wake_.Anticipating(Clock::now()) && Clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  const auto relayed = Clock::now();
  ASSERT_TRUE(wake_.Anticipating(relayed)) << "no relay before the fence completed";
  EXPECT_TRUE(lane_->Anticipating(relayed));
  EXPECT_GE(Ms(relayed - launched), 15.0);  // expected ~40 ms, less spin_ahead 10 ms
  CompleteNext();
  const std::optional<Terminal> terminal = Await(operation);
  ASSERT_TRUE(terminal.has_value());
  EXPECT_TRUE(terminal.value_or(Terminal{}).no_further_access);
}

// A submission lane that never polls sleeps between commands; each command
// wakes it. The median over several is far below the lane's 100 ms tick,
// which alone would bound a lost wakeup.
TEST_F(DeviceWakeTest, ASleepingSubmissionLaneWakesForEachCommand) {
  DeviceSettings settings;
  settings.poll_window = std::chrono::microseconds(0);
  Start(settings);
  std::vector<double> waits;
  for (int i = 0; i < 9; ++i) {
    std::this_thread::sleep_for(milliseconds(20));  // both lanes asleep
    std::atomic<bool> ran{false};
    const auto submitted = Clock::now();
    const OperationId operation = Launch(&ran);
    const auto give_up = submitted + kPatience;
    while (!ran.load() && Clock::now() < give_up) {
      std::this_thread::yield();
    }
    ASSERT_TRUE(ran.load());
    waits.push_back(Ms(Clock::now() - submitted));
    CompleteNext();
    ASSERT_TRUE(Await(operation).has_value());
  }
  std::ranges::sort(waits);
  EXPECT_LT(waits[waits.size() / 2], 50.0);
}

// Many threads submit while the lanes sleep and wake between them, and the
// device completes fences as they come: every operation is published.
TEST_F(DeviceWakeTest, ManySubmittersLoseNoCommandOrFence) {
  Start(DeviceSettings{.queue = {.capacity = 64, .reserved = 8, .batch = 16},
                       .poll_window = std::chrono::microseconds(0)});
  constexpr int kSubmitters = 4;
  constexpr int kEach = 100;
  std::atomic<bool> stop{false};
  std::jthread device([&] {
    while (!stop.load()) {
      if (!execution_.Step(stream_)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    }
  });
  std::atomic<int> submitted{0};
  std::vector<std::jthread> submitters;
  submitters.reserve(kSubmitters);
  // The board is the owner's: this thread opens every operation, the
  // submitters only hand them to the lane.
  std::vector<OperationId> operations;
  for (int i = 0; i < kSubmitters * kEach; ++i) {
    operations.push_back(board_.Open());
    ASSERT_TRUE(operations.back().valid());
  }
  const std::size_t in_flight = operations.size();
  std::atomic<std::size_t> next{0};
  for (int s = 0; s < kSubmitters; ++s) {
    submitters.emplace_back([&] {
      for (std::size_t i = next.fetch_add(1); i < in_flight; i = next.fetch_add(1)) {
        DeviceCommand command{.operation = operations[i],
                              .work = LaunchWork{.stream = 0, .job = [](NativeStream /*native*/) {
                                                   return JobResult::kQueued;
                                                 }}};
        // NOLINTNEXTLINE(bugprone-use-after-move): Submit moves only what it takes
        while (lane_->Submit(std::move(command)) == PushResult::kFull) {
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        submitted.fetch_add(1);
        if (i % 7 == 0) {
          std::this_thread::sleep_for(milliseconds(1));  // let the lanes fall asleep
        }
      }
    });
  }
  submitters.clear();  // joined
  std::size_t closed = 0;
  const auto give_up = Clock::now() + kPatience;
  while (closed < in_flight && Clock::now() < give_up) {
    (void)wake_.Consume();
    for (const auto& seen : board_.Harvest(64)) {
      if (seen.terminal && seen.acceptance != Acceptance::kNone) {
        EXPECT_TRUE(seen.terminal->no_further_access);
        EXPECT_TRUE(board_.Close(seen.operation));
        ++closed;
      }
    }
    if (closed < in_flight) {
      (void)wake_.WaitFor(milliseconds(10));
    }
  }
  stop.store(true);
  EXPECT_EQ(closed, in_flight);
  EXPECT_EQ(submitted.load(), static_cast<int>(in_flight));
}

// Closing wakes both lanes from their sleep: with nothing out, each returns
// well within the lanes' tick, over several pairs.
TEST(DeviceWakeShutdown, ClosingWakesSleepingLanes) {
  std::vector<double> joins;
  for (int i = 0; i < 5; ++i) {
    llmp::providers::fake::FakeDeviceExecution execution;
    WakeFlag wake;
    CompletionBoard board(8, wake);
    const StreamId stream = execution.CreateStream().value();
    DeviceService lane(execution, std::span<const StreamId>(&stream, 1), board, DeviceSettings{});
    std::jthread submission([&] { lane.RunSubmission(); });
    std::jthread completion([&] { lane.RunCompletion(); });
    std::this_thread::sleep_for(milliseconds(20));  // both asleep
    const auto closed = Clock::now();
    lane.Close();
    submission.join();
    completion.join();
    joins.push_back(Ms(Clock::now() - closed));
  }
  std::ranges::sort(joins);
  EXPECT_LT(joins[joins.size() / 2], 50.0);
}

// A close while the completion lane sleeps on a fence expected to take
// 50 ms: the submission lane returns, but the completion lane keeps the
// fence until it is seen complete, however long that takes, and only then
// returns, with the result published.
TEST_F(DeviceWakeTest, ClosingWithAFenceOutWaitsForItsCompletion) {
  Start();
  for (int i = 0; i < 3; ++i) {
    Learn(milliseconds(50));
  }
  const OperationId operation = Launch();
  const auto give_up = Clock::now() + kPatience;
  while (execution_.fences() == 0 && Clock::now() < give_up) {
    std::this_thread::yield();  // recorded
  }
  lane_->Close();
  while (!submission_done_.load() && Clock::now() < give_up) {
    std::this_thread::sleep_for(milliseconds(1));
  }
  EXPECT_TRUE(submission_done_.load());
  std::this_thread::sleep_for(milliseconds(150));  // three times its expectation
  EXPECT_FALSE(completion_done_.load()) << "the completion lane left a fence behind";
  CompleteNext();
  const std::optional<Terminal> terminal = Await(operation);
  ASSERT_TRUE(terminal.has_value());
  EXPECT_TRUE(terminal.value_or(Terminal{}).no_further_access);
  while (!completion_done_.load() && Clock::now() < give_up) {
    std::this_thread::sleep_for(milliseconds(1));
  }
  EXPECT_TRUE(completion_done_.load());
}

}  // namespace
