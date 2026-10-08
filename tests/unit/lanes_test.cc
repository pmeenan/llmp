// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// D-048's lanes and completion ownership
// (docs/async-model.md): bounded queues with a cleanup reserve, the
// coalesced wake flag, the completion board's mailboxes, lanes that drain
// on close, task trees that unwind, and the ready queue. The concurrency
// tests use real threads; the deterministic tests do not prove memory
// ordering, the threaded ones exercise it (and run under ThreadSanitizer on
// a Spark in `check:spark`).

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/wake.h"
#include "expected_error.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"
#include "scheduler/tasks.h"

namespace {

using jitllm::base::BoundedQueue;
using jitllm::base::PushKind;
using jitllm::base::PushResult;
using jitllm::base::WakeFlag;
using jitllm::scheduler::Acceptance;
using jitllm::scheduler::CompletionBoard;
using jitllm::scheduler::Lane;
using jitllm::scheduler::LaneSettings;
using jitllm::scheduler::Observation;
using jitllm::scheduler::OperationId;
using jitllm::scheduler::Outcome;
using jitllm::scheduler::Published;
using jitllm::scheduler::ReadyQueue;
using jitllm::scheduler::TaskError;
using jitllm::scheduler::TaskId;
using jitllm::scheduler::TaskOutcome;
using jitllm::scheduler::TaskTable;
using jitllm::scheduler::Terminal;
using jitllm::test_support::Failed;
using ::testing::ElementsAre;
using ::testing::Pair;

constexpr auto kPatience = std::chrono::seconds(20);  // a hang, not a timing

int Take(std::optional<std::unique_ptr<int>> popped) {
  EXPECT_TRUE(popped.has_value() && *popped != nullptr);
  return popped.has_value() && *popped != nullptr ? **popped : -1;
}

jitllm::scheduler::TaskView View(const TaskTable& tasks, TaskId task) {
  const auto view = tasks.Describe(task);
  EXPECT_TRUE(view.has_value());
  return view.value_or(jitllm::scheduler::TaskView{});
}

TaskId Pop(ReadyQueue& ready) {
  const auto next = ready.Next();
  EXPECT_TRUE(next.has_value());
  return next.value_or(TaskId{});
}

TEST(BoundedQueue, RefusesWhenFullAndKeepsTheCleanupReserve) {
  BoundedQueue<std::unique_ptr<int>> queue(3, 1);
  auto one = std::make_unique<int>(1);
  auto two = std::make_unique<int>(2);
  auto three = std::make_unique<int>(3);
  auto four = std::make_unique<int>(4);
  EXPECT_EQ(queue.TryPush(std::move(one)), PushResult::kAccepted);
  EXPECT_EQ(queue.TryPush(std::move(two)), PushResult::kAccepted);
  EXPECT_EQ(queue.TryPush(std::move(three)), PushResult::kFull);
  ASSERT_NE(three, nullptr);  // a refused command stays with the caller
  EXPECT_EQ(queue.TryPush(std::move(three), PushKind::kCleanup), PushResult::kAccepted);
  EXPECT_EQ(queue.TryPush(std::move(four), PushKind::kCleanup), PushResult::kFull);
  EXPECT_EQ(Take(queue.TryPop()), 1);  // FIFO
  queue.Close();
  EXPECT_EQ(queue.TryPush(std::move(four), PushKind::kCleanup), PushResult::kClosed);
  // Closing drains what was accepted.
  EXPECT_EQ(Take(queue.Pop(std::stop_token{})), 2);
  EXPECT_EQ(Take(queue.Pop(std::stop_token{})), 3);
  EXPECT_FALSE(queue.Pop(std::stop_token{}).has_value());
}

TEST(BoundedQueue, ManyProducersAndConsumersLoseNothing) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 3;
  constexpr int kEach = 20000;
  BoundedQueue<int> queue(16, 0);
  std::atomic<std::int64_t> sum{0};
  std::atomic<int> received{0};
  {
    std::vector<std::jthread> consumers;
    consumers.reserve(kConsumers);
    for (int c = 0; c < kConsumers; ++c) {
      consumers.emplace_back([&] {
        while (std::optional<int> value = queue.Pop(std::stop_token{})) {
          sum.fetch_add(*value, std::memory_order_relaxed);
          received.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    {
      std::vector<std::jthread> producers;
      producers.reserve(kProducers);
      for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&queue, p] {
          for (int i = 1; i <= kEach; ++i) {
            const int value = (p * kEach) + i;
            while (queue.TryPush(int{value}) == PushResult::kFull) {
              std::this_thread::yield();  // backpressure: the producer retries
            }
          }
        });
      }
    }
    queue.Close();
  }
  constexpr std::int64_t kTotal = std::int64_t{kProducers} * kEach;
  EXPECT_EQ(received.load(), kTotal);
  EXPECT_EQ(sum.load(), kTotal * (kTotal + 1) / 2);
}

TEST(WakeFlag, SignalsCoalesceAndAreNeverLost) {
  WakeFlag wake;
  EXPECT_FALSE(wake.Consume());
  wake.Signal();
  wake.Signal();
  EXPECT_TRUE(wake.Consume());
  EXPECT_FALSE(wake.Consume());  // two signals, one wakeup
  EXPECT_FALSE(wake.WaitFor(std::chrono::milliseconds(1)));

  // Producers publish, then signal; the owner consumes, rechecks, and
  // sleeps only while nothing is pending. Every publication is seen.
  constexpr int kProducers = 4;
  constexpr int kEach = 20000;
  std::atomic<int> published{0};
  std::vector<std::jthread> producers;
  producers.reserve(kProducers);
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&] {
      for (int i = 0; i < kEach; ++i) {
        published.fetch_add(1, std::memory_order_relaxed);
        wake.Signal();
        if (i % 64 == 0) {
          std::this_thread::yield();
        }
      }
    });
  }
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  int seen = 0;
  while (seen < kProducers * kEach && std::chrono::steady_clock::now() < give_up) {
    seen = published.load(std::memory_order_acquire);
    if (seen < kProducers * kEach) {
      // A long timeout: a lost wakeup shows as a hang, not a slow pass.
      (void)wake.WaitFor(std::chrono::seconds(5));
    }
  }
  EXPECT_EQ(seen, kProducers * kEach);
}

// An anticipation (the runtime wake) raises the owner's polling deadline
// and wakes it; an earlier one changes nothing and wakes nobody. It is a
// hint: it never stands in for a publication's signal.
TEST(WakeFlag, AnAnticipationOnlyRaisesTheDeadlineAndWakesTheOwner) {
  using Clock = WakeFlag::Clock;
  WakeFlag wake;
  const auto now = Clock::now();
  EXPECT_FALSE(wake.Anticipating(now));
  wake.Anticipate(now + std::chrono::seconds(10));
  EXPECT_TRUE(wake.Anticipating(now));
  EXPECT_TRUE(wake.Anticipating(now + std::chrono::seconds(9)));
  EXPECT_FALSE(wake.Anticipating(now + std::chrono::seconds(10)));
  EXPECT_TRUE(wake.Consume());  // it woke the owner
  wake.Anticipate(now + std::chrono::seconds(5));
  EXPECT_FALSE(wake.Consume());  // earlier: no change, no wakeup
  EXPECT_FALSE(wake.Anticipating(now + std::chrono::seconds(10)));
  wake.Anticipate(now + std::chrono::seconds(20));
  EXPECT_TRUE(wake.Anticipating(now + std::chrono::seconds(10)));
  EXPECT_TRUE(wake.Consume());
}

// WaitUntil returns at once for a pending signal, early for one sent while
// it waits, and otherwise at its deadline.
TEST(WakeFlag, WaitUntilReturnsForASignalOrAtItsDeadline) {
  using Clock = WakeFlag::Clock;
  WakeFlag wake;
  wake.Signal();
  EXPECT_TRUE(wake.WaitUntil(Clock::now() + kPatience));
  auto start = Clock::now();
  EXPECT_FALSE(wake.WaitUntil(start + std::chrono::milliseconds(20)));
  EXPECT_GE(Clock::now() - start, std::chrono::milliseconds(20));
  std::jthread signaller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    wake.Signal();
  });
  start = Clock::now();
  EXPECT_TRUE(wake.WaitUntil(start + kPatience));  // a lost signal shows as a hang
  EXPECT_LT(Clock::now() - start, kPatience);
}

// The owner polls while anticipating and sleeps otherwise; producers
// publish then signal, and anticipate between publications. Every
// publication is seen, however the anticipations and sleeps interleave.
TEST(WakeFlag, AnticipationsAmongManyPublishersLoseNoWakeup) {
  using Clock = WakeFlag::Clock;
  WakeFlag wake;
  constexpr int kProducers = 4;
  constexpr int kEach = 5000;
  std::atomic<int> published{0};
  std::vector<std::jthread> producers;
  producers.reserve(kProducers);
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kEach; ++i) {
        if ((i + p) % 3 == 0) {
          wake.Anticipate(Clock::now() + std::chrono::microseconds(50));
        }
        published.fetch_add(1, std::memory_order_relaxed);
        wake.Signal();
        if (i % 32 == 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
      }
    });
  }
  const auto give_up = Clock::now() + kPatience;
  int seen = 0;
  while (seen < kProducers * kEach && Clock::now() < give_up) {
    (void)wake.Consume();
    seen = published.load(std::memory_order_acquire);
    if (seen < kProducers * kEach) {
      if (wake.Anticipating(Clock::now())) {
        std::this_thread::yield();
      } else {
        (void)wake.WaitFor(std::chrono::seconds(5));  // a lost wakeup shows as a hang
      }
    }
  }
  EXPECT_EQ(seen, kProducers * kEach);
}

// The expectation is the shortest of the last eight samples.
TEST(Expectation, IsTheShortestOfTheLastEight) {
  jitllm::base::Expectation expected;
  EXPECT_FALSE(expected.known());
  expected.Add(std::chrono::milliseconds(50));
  EXPECT_TRUE(expected.known());
  EXPECT_EQ(expected.value(), std::chrono::milliseconds(50));
  expected.Add(std::chrono::milliseconds(40));
  expected.Add(std::chrono::milliseconds(60));
  EXPECT_EQ(expected.value(), std::chrono::milliseconds(40));
  for (int i = 0; i < 7; ++i) {
    expected.Add(std::chrono::milliseconds(500));
  }
  EXPECT_EQ(expected.value(), std::chrono::milliseconds(60));  // the eighth sample back
  expected.Add(std::chrono::milliseconds(500));
  EXPECT_EQ(expected.value(), std::chrono::milliseconds(500));  // 60 is nine back
  expected.Add(std::chrono::milliseconds(5));
  EXPECT_EQ(expected.value(), std::chrono::milliseconds(5));
}

// Each recent length is a likely end: Next is the shortest not yet more
// than `past` behind the elapsed time, and none once all are.
TEST(Expectation, NextIsTheNearestLikelyEndNotYetOutlasted) {
  using std::chrono::milliseconds;
  jitllm::base::Expectation expected;
  EXPECT_FALSE(expected.Next(milliseconds(0), milliseconds(1)).has_value());
  for (const int length : {45, 5, 45, 20}) {
    expected.Add(milliseconds(length));
  }
  EXPECT_EQ(expected.Next(milliseconds(0), milliseconds(1)), milliseconds(5));
  EXPECT_EQ(expected.Next(milliseconds(5), milliseconds(1)), milliseconds(5));  // within its past
  EXPECT_EQ(expected.Next(milliseconds(6), milliseconds(1)), milliseconds(20));
  EXPECT_EQ(expected.Next(milliseconds(30), milliseconds(1)), milliseconds(45));
  EXPECT_FALSE(expected.Next(milliseconds(46), milliseconds(1)).has_value());
}

class BoardTest : public ::testing::Test {
 protected:
  WakeFlag wake_;
  CompletionBoard board_{4, wake_};
};

TEST(CompletionBoard, BoundedBatchesKeepPartialLimitsAcrossClosedNewsAndReuse) {
  WakeFlag wake;
  CompletionBoard board(130, wake);
  std::vector<OperationId> original;
  for (std::uint64_t i = 0; i < 130; ++i) {
    const auto op = board.Open();
    ASSERT_TRUE(op.valid());
    original.push_back(op);
    ASSERT_EQ(board.Accept(op, Acceptance::kAccepted), Published::kRecorded);
    ASSERT_EQ(board.Complete(op, Terminal{.bytes = i, .no_further_access = true}),
              Published::kRecorded);
  }
  for (std::size_t i = 0; i < 70; ++i) ASSERT_TRUE(board.Close(original[i]));
  (void)wake.Consume();
  const auto first = board.Harvest(5);
  ASSERT_EQ(first.size(), 5U);
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i].operation, original[i + 70]);
    ASSERT_TRUE(board.Close(first[i].operation));
  }
  EXPECT_EQ(board.harvest_stats().indices, 75U);  // closed holes do not consume the limit
  EXPECT_EQ(board.harvest_stats().pop_locks, 15U);
  EXPECT_TRUE(wake.Consume());  // the partial harvest re-signalled remaining news
  std::vector<OperationId> reused;
  for (std::size_t i = 0; i < 75; ++i) {
    const auto op = board.Open();
    ASSERT_TRUE(op.valid());
    EXPECT_EQ(op.index(), original[i].index());
    EXPECT_NE(op.generation(), original[i].generation());
    reused.push_back(op);
    ASSERT_EQ(board.Accept(op, Acceptance::kAccepted), Published::kRecorded);
    ASSERT_EQ(board.Complete(op, Terminal{.bytes = op.generation(), .no_further_access = true}),
              Published::kRecorded);
    EXPECT_EQ(board.Accept(original[i], Acceptance::kAccepted), Published::kStale);
  }
  const auto rest = board.Harvest(130);
  ASSERT_EQ(rest.size(), 130U);  // 55 original plus75 reused, across three bounded batches
  EXPECT_EQ(board.harvest_stats().indices, 205U);
  EXPECT_EQ(board.harvest_stats().pop_locks, 18U);
  for (std::size_t i = 0; i < rest.size(); ++i) {
    EXPECT_EQ(rest[i].operation, i < 55 ? original[i + 75] : reused[i - 55]);
    EXPECT_TRUE(board.Close(rest[i].operation));
  }
  const auto stats = board.harvest_stats();
  EXPECT_TRUE(board.Harvest(0).empty());
  EXPECT_TRUE(board.Harvest(130).empty());
  EXPECT_EQ(board.harvest_stats().indices, stats.indices);
  EXPECT_EQ(board.harvest_stats().pop_locks, stats.pop_locks);
}

TEST(CompletionBoard, BatchRemovalDropsRetiredNewsWithoutIssuingAnIdentityAgain) {
  WakeFlag wake;
  CompletionBoard board(3, wake, UINT32_MAX);
  for (int i = 0; i < 3; ++i) {
    const auto op = board.Open();
    ASSERT_EQ(board.Accept(op, Acceptance::kNotStarted), Published::kRecorded);
    ASSERT_TRUE(board.Close(op));
  }
  EXPECT_TRUE(board.Harvest(3).empty());
  EXPECT_EQ(board.harvest_stats().indices, 3U);
  EXPECT_EQ(board.harvest_stats().pop_locks, 1U);
  EXPECT_TRUE(board.exhausted());
  EXPECT_FALSE(board.Open().valid());
}

TEST(CompletionBoard, BatchedHarvestRacesNewObservationsProofAndGenerationReuse) {
  constexpr std::size_t kMailboxes = 64;
  constexpr int kOperations = 4000;
  WakeFlag wake;
  CompletionBoard board(kMailboxes, wake);
  BoundedQueue<OperationId> submitted(kMailboxes, 0);
  std::atomic<int> wrong{0};
  std::vector<std::jthread> providers;
  for (int p = 0; p < 3; ++p) {
    providers.emplace_back([&] {
      while (const auto op = submitted.Pop(std::stop_token{})) {
        if (board.Accept(*op, Acceptance::kAccepted) != Published::kRecorded) ++wrong;
        std::this_thread::yield();
        const Terminal unproven{.bytes = op->generation()};
        if (board.Complete(*op, unproven) != Published::kRecorded) ++wrong;
        std::this_thread::yield();
        if (board.Complete(*op, Terminal{.bytes = op->generation(), .no_further_access = true}) !=
            Published::kRecorded)
          ++wrong;
        const auto repeat =
            board.Complete(*op, Terminal{.bytes = op->generation(), .no_further_access = true});
        if (repeat != Published::kDuplicate && repeat != Published::kStale) ++wrong;
      }
    });
  }
  int opened = 0;
  int finished = 0;
  const auto until = std::chrono::steady_clock::now() + kPatience;
  while (finished < kOperations && std::chrono::steady_clock::now() < until) {
    while (opened < kOperations) {
      const auto op = board.Open();
      if (!op.valid()) break;
      if (submitted.TryPush(OperationId{op}) != PushResult::kAccepted) {
        ++wrong;
        break;
      }
      ++opened;
    }
    (void)wake.Consume();
    const auto seen = board.Harvest(64);
    for (std::size_t i = 0; i < seen.size(); ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        if (seen[i].operation == seen[j].operation) ++wrong;
      }
      const auto& item = seen[i];
      if (item.contradictory) ++wrong;
      if (item.acceptance == Acceptance::kAccepted && item.terminal &&
          item.terminal->no_further_access) {
        if (item.terminal->bytes != item.operation.generation()) ++wrong;
        if (board.Close(item.operation)) ++finished;
      }
    }
    if (finished < opened && board.open() != 0) {
      (void)wake.WaitFor(std::chrono::milliseconds(100));
    }
  }
  submitted.Close();
  providers.clear();  // publishers return before board/wake destruction
  EXPECT_EQ(finished, kOperations);
  EXPECT_EQ(wrong.load(), 0);
  EXPECT_EQ(board.publications(), 3U * kOperations);
  EXPECT_GE(board.harvest_stats().indices, static_cast<std::uint64_t>(kOperations));
  EXPECT_LE(board.harvest_stats().pop_locks, board.harvest_stats().indices);
}

TEST_F(BoardTest, CompletionBeforeAcceptanceIsKept) {
  const OperationId op = board_.Open();
  ASSERT_TRUE(op.valid());
  const Terminal done{.outcome = Outcome::kSucceeded, .bytes = 4096, .no_further_access = true};
  EXPECT_EQ(board_.Complete(op, done), Published::kRecorded);
  EXPECT_TRUE(wake_.Consume());
  auto news = board_.Harvest(8);
  ASSERT_EQ(news.size(), 1U);
  EXPECT_EQ(news[0].acceptance, Acceptance::kNone);
  EXPECT_EQ(news[0].terminal, done);
  EXPECT_EQ(board_.Accept(op, Acceptance::kAccepted), Published::kRecorded);
  news = board_.Harvest(8);
  ASSERT_EQ(news.size(), 1U);
  EXPECT_EQ(news[0].acceptance, Acceptance::kAccepted);
  EXPECT_EQ(news[0].terminal, done);
  EXPECT_TRUE(board_.Close(op));
}

TEST_F(BoardTest, DuplicatesAreIgnoredAndContradictionsReported) {
  const OperationId op = board_.Open();
  const Terminal done{.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = true};
  EXPECT_EQ(board_.Accept(op, Acceptance::kAccepted), Published::kRecorded);
  EXPECT_EQ(board_.Accept(op, Acceptance::kAccepted), Published::kDuplicate);
  EXPECT_EQ(board_.Complete(op, done), Published::kRecorded);
  EXPECT_EQ(board_.Complete(op, done), Published::kDuplicate);
  auto news = board_.Harvest(8);
  ASSERT_EQ(news.size(), 1U);  // one mailbox, one report
  EXPECT_FALSE(news[0].contradictory);
  EXPECT_EQ(board_.Accept(op, Acceptance::kNotStarted), Published::kContradiction);
  news = board_.Harvest(8);
  ASSERT_EQ(news.size(), 1U);
  EXPECT_TRUE(news[0].contradictory);
  EXPECT_EQ(news[0].acceptance, Acceptance::kAccepted);  // the first stands
  const Terminal other{.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true};
  EXPECT_EQ(board_.Complete(op, other), Published::kContradiction);
}

// Stale identities cannot update the mailbox's next operation, and a
// mailbox stays open until its terminal result is in.
TEST_F(BoardTest, StaleIdentitiesChangeNothing) {
  const OperationId first = board_.Open();
  EXPECT_FALSE(board_.Close(first));  // no terminal result yet
  EXPECT_EQ(board_.Accept(first, Acceptance::kAccepted), Published::kRecorded);
  EXPECT_EQ(board_.Complete(
                first, {.outcome = Outcome::kCancelled, .bytes = 0, .no_further_access = true}),
            Published::kRecorded);
  (void)wake_.Consume();
  (void)board_.Harvest(8);
  ASSERT_TRUE(board_.Close(first));
  EXPECT_FALSE(board_.Close(first));
  // The same mailbox comes back once every other free one has had its turn.
  for (int others = 0; others < 3; ++others) {
    ASSERT_NE(board_.Open().index(), first.index());
  }
  const OperationId second = board_.Open();
  ASSERT_EQ(second.index(), first.index());  // the same mailbox, reused
  EXPECT_NE(second.generation(), first.generation());
  EXPECT_EQ(board_.Accept(first, Acceptance::kAccepted), Published::kStale);
  EXPECT_EQ(board_.Complete(
                first, {.outcome = Outcome::kSucceeded, .bytes = 9, .no_further_access = true}),
            Published::kStale);
  EXPECT_FALSE(wake_.Consume());
  EXPECT_TRUE(board_.Harvest(8).empty());
  EXPECT_EQ(board_.Accept(OperationId{}, Acceptance::kAccepted), Published::kStale);
}

// A mailbox closes only once its operation is reconciled: not started, or
// started with a terminal result that proves no further access.
TEST_F(BoardTest, OnlyAReconciledOperationCloses) {
  const OperationId unaccepted = board_.Open();
  const Terminal done{.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true};
  ASSERT_EQ(board_.Complete(unaccepted, done), Published::kRecorded);
  EXPECT_FALSE(board_.Close(unaccepted));  // acceptance not yet reconciled
  ASSERT_EQ(board_.Accept(unaccepted, Acceptance::kUnknown), Published::kRecorded);
  EXPECT_TRUE(board_.Close(unaccepted));

  const OperationId touching = board_.Open();
  ASSERT_EQ(board_.Accept(touching, Acceptance::kAccepted), Published::kRecorded);
  ASSERT_EQ(board_.Complete(touching,
                            {.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false}),
            Published::kRecorded);
  EXPECT_FALSE(board_.Close(touching));  // it may still touch memory: stays open

  const OperationId never = board_.Open();
  ASSERT_EQ(board_.Accept(never, Acceptance::kNotStarted), Published::kRecorded);
  EXPECT_TRUE(board_.Close(never));

  const OperationId confused = board_.Open();
  ASSERT_EQ(board_.Accept(confused, Acceptance::kNotStarted), Published::kRecorded);
  ASSERT_EQ(board_.Complete(confused, done), Published::kRecorded);
  (void)wake_.Consume();
  bool contradictory = false;
  for (const Observation& seen : board_.Harvest(8)) {
    contradictory = contradictory || (seen.operation == confused && seen.contradictory);
  }
  EXPECT_TRUE(contradictory);  // not started, yet completed
  EXPECT_FALSE(board_.Close(confused));
}

// The no-further-access proof may follow the result, and a contradiction
// that loses to Close is stale, not a fault the owner never sees.
TEST_F(BoardTest, TheProofMayComeLater) {
  const OperationId op = board_.Open();
  ASSERT_EQ(board_.Accept(op, Acceptance::kAccepted), Published::kRecorded);
  ASSERT_EQ(
      board_.Complete(op, {.outcome = Outcome::kCancelled, .bytes = 0, .no_further_access = false}),
      Published::kRecorded);
  EXPECT_FALSE(board_.Close(op));  // not yet proven
  EXPECT_EQ(
      board_.Complete(op, {.outcome = Outcome::kCancelled, .bytes = 0, .no_further_access = true}),
      Published::kRecorded);
  EXPECT_EQ(
      board_.Complete(op, {.outcome = Outcome::kCancelled, .bytes = 0, .no_further_access = false}),
      Published::kDuplicate);  // never withdrawn
  (void)wake_.Consume();
  const auto news = board_.Harvest(8);
  ASSERT_EQ(news.size(), 1U);  // one report, as last read
  ASSERT_TRUE(news[0].terminal.has_value());
  EXPECT_TRUE(news[0].terminal.value_or(Terminal{}).no_further_access);
  EXPECT_FALSE(news[0].contradictory);
  EXPECT_TRUE(board_.Close(op));
  EXPECT_EQ(board_.Accept(op, Acceptance::kNotStarted), Published::kStale);
}

// News left after a partial harvest keeps the owner awake.
TEST_F(BoardTest, APartialHarvestResignals) {
  const OperationId a = board_.Open();
  const OperationId b = board_.Open();
  ASSERT_EQ(board_.Accept(a, Acceptance::kAccepted), Published::kRecorded);
  ASSERT_EQ(board_.Accept(b, Acceptance::kAccepted), Published::kRecorded);
  EXPECT_TRUE(wake_.Consume());
  EXPECT_EQ(board_.Harvest(1).size(), 1U);
  EXPECT_TRUE(wake_.Consume());
  EXPECT_EQ(board_.Harvest(1).size(), 1U);
  EXPECT_FALSE(wake_.Consume());
}

TEST_F(BoardTest, EveryMailboxInUseRefusesANewOperation) {
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(board_.Open().valid());
  }
  EXPECT_FALSE(board_.Open().valid());
  EXPECT_EQ(board_.open(), 4U);
}

// Free mailboxes are issued in rotation, the longest free first, so every
// mailbox's generation advances at the same pace: exhaustion comes after
// about capacity x 2^32 operations, not from one mailbox reused each time.
TEST(CompletionBoard, FreeMailboxesAreIssuedInRotation) {
  WakeFlag wake;
  CompletionBoard board(3, wake);
  std::vector<std::pair<std::uint32_t, std::uint32_t>> issued;
  for (int i = 0; i < 9; ++i) {
    const OperationId op = board.Open();
    ASSERT_TRUE(op.valid());
    issued.emplace_back(op.index(), op.generation());
    ASSERT_EQ(board.Accept(op, Acceptance::kNotStarted), Published::kRecorded);
    ASSERT_TRUE(board.Close(op));
  }
  EXPECT_THAT(issued, ElementsAre(Pair(0, 1), Pair(1, 1), Pair(2, 1), Pair(0, 2), Pair(1, 2),
                                  Pair(2, 2), Pair(0, 3), Pair(1, 3), Pair(2, 3)));
  // A mailbox held open keeps its place; the others rotate around it.
  const OperationId held = board.Open();
  EXPECT_EQ(held.index(), 0U);
  for (const std::uint32_t expected : {1U, 2U, 1U, 2U}) {
    const OperationId op = board.Open();
    EXPECT_EQ(op.index(), expected);
    ASSERT_EQ(board.Accept(op, Acceptance::kNotStarted), Published::kRecorded);
    ASSERT_TRUE(board.Close(op));
  }
}

// A mailbox whose generation is exhausted retires: it is skipped, never
// reissued, and a late publication for it is stale. Nothing aborts, and
// the board refuses only once every mailbox has retired.
TEST(CompletionBoard, AnExhaustedMailboxRetiresAndIsSkipped) {
  WakeFlag wake;
  CompletionBoard board(2, wake, UINT32_MAX - 1);
  const auto run = [&board](Acceptance acceptance) {
    const OperationId op = board.Open();
    EXPECT_TRUE(op.valid());
    EXPECT_EQ(board.Accept(op, acceptance), Published::kRecorded);
    if (acceptance == Acceptance::kAccepted) {
      EXPECT_EQ(board.Complete(
                    op, {.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true}),
                Published::kRecorded);
    }
    EXPECT_TRUE(board.Close(op));
    return op;
  };
  (void)run(Acceptance::kNotStarted);  // mailbox 0 at UINT32_MAX - 1
  (void)run(Acceptance::kAccepted);    // mailbox 1 at UINT32_MAX - 1
  const OperationId last0 = run(Acceptance::kAccepted);
  EXPECT_EQ(last0, OperationId(0, UINT32_MAX));
  EXPECT_EQ(board.available(), 1U);  // mailbox 0 retired
  EXPECT_FALSE(board.exhausted());
  const OperationId next = board.Open();
  EXPECT_EQ(next, OperationId(1, UINT32_MAX));  // skipped mailbox 0
  EXPECT_FALSE(board.Open().valid());
  // Late news for the retired mailbox changes nothing and reaches no one.
  EXPECT_EQ(board.Accept(last0, Acceptance::kNotStarted), Published::kStale);
  EXPECT_EQ(
      board.Complete(last0, {.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = true}),
      Published::kStale);
  EXPECT_EQ(board.Accept(next, Acceptance::kNotStarted), Published::kRecorded);
  ASSERT_TRUE(board.Close(next));
  EXPECT_EQ(board.Accept(next, Acceptance::kNotStarted), Published::kStale);
  EXPECT_EQ(
      board.Complete(next, {.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = true}),
      Published::kStale);
  // Even one that matches whatever the retired words hold.
  EXPECT_EQ(
      board.Complete(next, {.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true}),
      Published::kStale);
  EXPECT_EQ(board.Complete(last0,
                           {.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true}),
            Published::kStale);
  EXPECT_EQ(board.Accept(last0, Acceptance::kAccepted), Published::kStale);
  EXPECT_EQ(board.available(), 0U);
  EXPECT_EQ(board.open(), 0U);
  EXPECT_TRUE(board.exhausted());
  EXPECT_FALSE(board.Open().valid());
  (void)wake.Consume();
  for (const Observation& seen : board.Harvest(8)) {
    ADD_FAILURE() << "news of a retired mailbox: " << seen.operation.ToString();
  }
}

// Provider threads publish acceptance and completion in either order while
// the owner harvests, closes and reopens mailboxes. Every operation's
// result arrives exactly as published, and nothing is lost to a full queue
// or a missed wakeup.
TEST(CompletionBoard, ThreadedPublicationReachesTheOwner) {
  constexpr std::size_t kMailboxes = 8;
  constexpr int kOperations = 20000;
  WakeFlag wake;
  CompletionBoard board(kMailboxes, wake);
  BoundedQueue<OperationId> submitted(kMailboxes, 0);
  std::atomic<int> contradictions{0};
  std::vector<std::jthread> providers;
  providers.reserve(3);
  for (int p = 0; p < 3; ++p) {
    providers.emplace_back([&] {
      while (std::optional<OperationId> op = submitted.Pop(std::stop_token{})) {
        const Terminal done{
            .outcome = Outcome::kSucceeded, .bytes = op->generation(), .no_further_access = true};
        const bool complete_first = (op->generation() % 2) == 0;
        if (complete_first) {
          (void)board.Complete(*op, done);
        }
        if (board.Accept(*op, Acceptance::kAccepted) == Published::kContradiction) {
          contradictions.fetch_add(1);
        }
        if (!complete_first) {
          (void)board.Complete(*op, done);
        }
        (void)board.Complete(*op, done);  // a duplicate, which is harmless
      }
    });
  }
  int opened = 0;
  int finished = 0;
  bool wrong = false;
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (finished < kOperations && std::chrono::steady_clock::now() < give_up) {
    while (opened < kOperations) {
      const OperationId op = board.Open();
      if (!op.valid()) {
        break;
      }
      ASSERT_EQ(submitted.TryPush(OperationId{op}), PushResult::kAccepted);
      ++opened;
    }
    (void)wake.Consume();
    for (const Observation& seen : board.Harvest(kMailboxes)) {
      wrong = wrong || seen.contradictory;
      if (seen.terminal && seen.acceptance == Acceptance::kAccepted) {
        wrong = wrong || seen.terminal->bytes != seen.operation.generation();
        if (board.Close(seen.operation)) {
          ++finished;
        }
      }
    }
    if (finished < opened && board.open() > 0) {
      (void)wake.WaitFor(std::chrono::seconds(5));
    }
  }
  submitted.Close();
  providers.clear();
  EXPECT_EQ(finished, kOperations);
  EXPECT_FALSE(wrong);
  EXPECT_EQ(contradictions.load(), 0);
}

// Publishers race the owner closing a mailbox at its last generation. A
// repeat of what was recorded is a duplicate before the close and stale
// after it; a result for an operation that never started either
// contradicts it, which keeps the mailbox open for the fault, or loses to
// the close and is stale. Nothing retired is reported, nothing hangs, and
// a retired mailbox is never issued again.
TEST(CompletionBoard, PublishersRacingARetirementFindItStale) {
  constexpr int kRounds = 400;
  constexpr int kPublishers = 4;
  for (int round = 0; round < kRounds; ++round) {
    WakeFlag wake;
    CompletionBoard board(1, wake, UINT32_MAX);
    const OperationId op = board.Open();
    const bool not_started = round % 2 == 0;
    const Terminal done{.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true};
    const Acceptance acceptance = not_started ? Acceptance::kNotStarted : Acceptance::kAccepted;
    ASSERT_EQ(board.Accept(op, acceptance), Published::kRecorded);
    if (!not_started) {
      ASSERT_EQ(board.Complete(op, done), Published::kRecorded);
    }
    (void)wake.Consume();
    (void)board.Harvest(1);
    std::atomic<int> running{0};
    std::atomic<bool> go{false};
    std::atomic<bool> stop{false};
    std::atomic<int> landed{0};  // a result recorded, or a contradiction
    std::atomic<int> wrong{0};
    std::vector<std::jthread> publishers;
    publishers.reserve(kPublishers);
    for (int p = 0; p < kPublishers; ++p) {
      publishers.emplace_back([&, p] {
        running.fetch_add(1);
        // Results start when the owner is about to close.
        while (p % 2 == 1 && !go.load(std::memory_order_acquire)) {
        }
        // Publishing while the owner closes, and a little after.
        for (int after = 0; after < 20; after += stop.load(std::memory_order_acquire) ? 1 : 0) {
          const Published published =
              p % 2 == 0 ? board.Accept(op, acceptance) : board.Complete(op, done);
          if (published == Published::kRecorded || published == Published::kContradiction) {
            landed.fetch_add(1);
            // Only a result for an operation that never started can land.
            if (!not_started || p % 2 == 0) {
              wrong.fetch_add(1);
            }
          }
        }
      });
    }
    while (running.load() < kPublishers) {
      std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    // A varying head start, so a result sometimes beats the close.
    for (int spin = 0; spin < (round / 2 % 16) * 64; ++spin) {
      (void)running.load();
    }
    const bool closed = board.Close(op);
    stop.store(true, std::memory_order_release);
    publishers.clear();
    EXPECT_EQ(wrong.load(), 0) << "round " << round;
    if (closed) {
      EXPECT_TRUE(board.exhausted());
      EXPECT_FALSE(board.Open().valid());
      // Whatever landed before the close was a duplicate; after it, stale.
      EXPECT_EQ(board.Accept(op, acceptance), Published::kStale);
      EXPECT_EQ(board.Complete(op, done), Published::kStale);
      (void)wake.Consume();
      EXPECT_TRUE(board.Harvest(8).empty()) << "round " << round;
    } else {
      // Only a result that beat the close to an operation never started.
      ASSERT_TRUE(not_started) << "round " << round;
      EXPECT_GT(landed.load(), 0);
      EXPECT_FALSE(board.exhausted());
      EXPECT_EQ(board.open(), 1U);
      const std::vector<Observation> seen = board.Harvest(8);
      ASSERT_EQ(seen.size(), 1U);
      EXPECT_TRUE(seen[0].contradictory);
      EXPECT_FALSE(board.Close(op));
    }
  }
}

TEST(Lane, RunsEveryAcceptedCommandAndDrainsOnClose) {
  std::atomic<int> ran{0};
  std::atomic<bool> release{false};
  auto lane = std::make_unique<Lane<int>>(
      LaneSettings{.name = "storage", .capacity = 3, .reserved = 1, .workers = 1},
      [&](int&& value) {
        while (!release.load()) {
          std::this_thread::yield();
        }
        ran.fetch_add(value);
      });
  // One command occupies the worker; then two ordinary and one cleanup.
  EXPECT_EQ(lane->Submit(1), PushResult::kAccepted);
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (lane->queued() != 0 && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::yield();
  }
  EXPECT_EQ(lane->Submit(10), PushResult::kAccepted);
  EXPECT_EQ(lane->Submit(100), PushResult::kAccepted);
  EXPECT_EQ(lane->Submit(1000), PushResult::kFull);
  EXPECT_EQ(lane->Submit(1000, PushKind::kCleanup), PushResult::kAccepted);
  lane->Close();
  EXPECT_EQ(lane->Submit(5), PushResult::kClosed);
  release.store(true);
  lane.reset();  // joins after draining
  EXPECT_EQ(ran.load(), 1111);
}

TEST(Lane, SeveralWorkersShareTheQueue) {
  std::atomic<int> ran{0};
  {
    Lane<int> lane(LaneSettings{.name = "cpu", .capacity = 8, .reserved = 0, .workers = 4},
                   [&](int&& value) { ran.fetch_add(value); });
    for (int i = 0; i < 1000; ++i) {
      while (lane.Submit(1) == PushResult::kFull) {
        std::this_thread::yield();
      }
    }
  }
  EXPECT_EQ(ran.load(), 1000);
}

TEST(TaskTable, ATreeUnwindsOnlyWhenEverythingHasDrained) {
  TaskTable tasks(8);
  const TaskId root = tasks.Create().value();
  const TaskId a = tasks.Create(root).value();
  const TaskId b = tasks.Create(root).value();
  const TaskId a1 = tasks.Create(a).value();
  ASSERT_TRUE(tasks.PrepareOperation(a1).has_value());
  ASSERT_TRUE(tasks.PrepareOperation(b).has_value());
  EXPECT_EQ(View(tasks, root).children, 2U);

  // Cancellation closes the subtree to new work; accepted work stays.
  ASSERT_TRUE(tasks.Cancel(a).has_value());
  EXPECT_TRUE(View(tasks, a1).cancelled);
  EXPECT_FALSE(View(tasks, b).cancelled);
  EXPECT_EQ(Failed(tasks.Create(a)), TaskError::kClosed);
  EXPECT_EQ(Failed(tasks.PrepareOperation(a1)), TaskError::kClosed);
  EXPECT_EQ(View(tasks, a1).operations, 1U);

  // A late success of a cancelled task is a cancellation.
  ASSERT_TRUE(tasks.Finish(a1, TaskOutcome::kSucceeded).has_value());
  EXPECT_EQ(View(tasks, a1).outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(Failed(tasks.Retire(a1)), TaskError::kBusy);  // its read is in flight
  ASSERT_TRUE(tasks.Finish(a, TaskOutcome::kCancelled).has_value());
  EXPECT_EQ(Failed(tasks.Retire(a)), TaskError::kBusy);  // its child has not retired
  ASSERT_TRUE(tasks.RetireOperation(a1).has_value());
  ASSERT_TRUE(tasks.Retire(a1).has_value());
  ASSERT_TRUE(tasks.Retire(a).has_value());

  // A child's failure reaches its parent, which still drains the child.
  ASSERT_TRUE(tasks.Finish(b, TaskOutcome::kFailed).has_value());
  EXPECT_TRUE(View(tasks, root).child_failed);
  EXPECT_EQ(Failed(tasks.Retire(b)), TaskError::kBusy);
  ASSERT_TRUE(tasks.RetireOperation(b).has_value());
  ASSERT_TRUE(tasks.Retire(b).has_value());
  EXPECT_EQ(View(tasks, root).children, 0U);
  ASSERT_TRUE(tasks.Finish(root, TaskOutcome::kFailed).has_value());
  ASSERT_TRUE(tasks.Retire(root).has_value());
  EXPECT_EQ(tasks.size(), 0U);

  // Stale identities name nothing once their slots are reused.
  const TaskId reused = tasks.Create().value();
  EXPECT_FALSE(tasks.Describe(root).has_value());
  EXPECT_NE(reused, root);
  EXPECT_EQ(Failed(tasks.Retire(a1)), TaskError::kUnknownTask);
}

// A late wakeup for a retired task neither runs it nor blocks the task
// that reuses its slot.
TEST(TaskTable, ReadinessFollowsTheLiveTask) {
  TaskTable tasks(1);
  const TaskId old = tasks.Create().value();
  EXPECT_TRUE(tasks.MakeReady(old, 1).value());
  EXPECT_FALSE(tasks.MakeReady(old, 1).value());  // coalesced
  ASSERT_TRUE(tasks.Finish(old, TaskOutcome::kSucceeded).has_value());
  EXPECT_EQ(tasks.ready(), 0U);  // a finished task takes no more steps
  EXPECT_EQ(Failed(tasks.MakeReady(old, 1)), TaskError::kFinished);
  ASSERT_TRUE(tasks.Retire(old).has_value());
  const TaskId reused = tasks.Create().value();
  ASSERT_EQ(reused.index(), old.index());
  EXPECT_EQ(Failed(tasks.MakeReady(old, 1)), TaskError::kUnknownTask);
  EXPECT_TRUE(tasks.MakeReady(reused, 0).value());
  EXPECT_EQ(tasks.NextReady(), reused);
  EXPECT_FALSE(tasks.NextReady().has_value());
}

TEST(TaskTable, CancellationReachesADeepTreeAndCapacityIsFixed) {
  TaskTable tasks(5);
  const TaskId root = tasks.Create().value();
  TaskId at = root;
  for (int depth = 0; depth < 3; ++depth) {
    at = tasks.Create(at).value();
  }
  const TaskId sibling = tasks.Create(root).value();
  EXPECT_EQ(Failed(tasks.Create(root)), TaskError::kFull);
  ASSERT_TRUE(tasks.Cancel(root).has_value());
  EXPECT_TRUE(View(tasks, at).cancelled);
  EXPECT_TRUE(View(tasks, sibling).cancelled);
}

TEST(TaskTable, CancellationDuringSubmissionRetainsThePreparedOperation) {
  // The provider has the command, but has not resolved whether it started.
  // The task must already own its operation before cancellation can run.
  TaskTable tasks(1);
  WakeFlag wake;
  CompletionBoard board(1, wake);
  const TaskId task = tasks.Create().value();
  const OperationId op = board.Open();
  ASSERT_TRUE(tasks.PrepareOperation(task).has_value());
  ASSERT_TRUE(tasks.Cancel(task).has_value());
  ASSERT_TRUE(tasks.Finish(task, TaskOutcome::kCancelled).has_value());
  EXPECT_EQ(Failed(tasks.Retire(task)), TaskError::kBusy);
  EXPECT_EQ(Failed(tasks.Create()), TaskError::kFull);
  EXPECT_EQ(Failed(tasks.PrepareOperation(task)), TaskError::kClosed);

  // Late acceptance keeps the same hold; it requires no new admission.
  ASSERT_EQ(board.Accept(op, Acceptance::kAccepted), Published::kRecorded);
  EXPECT_FALSE(board.Close(op));
  EXPECT_EQ(Failed(tasks.Retire(task)), TaskError::kBusy);
  ASSERT_EQ(
      board.Complete(op, {.outcome = Outcome::kCancelled, .bytes = 0, .no_further_access = true}),
      Published::kRecorded);
  ASSERT_TRUE(board.Close(op));
  ASSERT_TRUE(tasks.RetireOperation(task).has_value());
  ASSERT_TRUE(tasks.Retire(task).has_value());
  EXPECT_TRUE(tasks.Create().has_value());
}

TEST(TaskTable, ANotStartedSubmissionReleasesThePreparedOperation) {
  TaskTable tasks(1);
  WakeFlag wake;
  CompletionBoard board(1, wake);
  const TaskId task = tasks.Create().value();
  const OperationId op = board.Open();
  ASSERT_TRUE(tasks.PrepareOperation(task).has_value());
  ASSERT_TRUE(tasks.Cancel(task).has_value());
  ASSERT_TRUE(tasks.Finish(task, TaskOutcome::kCancelled).has_value());
  EXPECT_EQ(Failed(tasks.Retire(task)), TaskError::kBusy);

  ASSERT_EQ(board.Accept(op, Acceptance::kNotStarted), Published::kRecorded);
  ASSERT_TRUE(board.Close(op));
  ASSERT_TRUE(tasks.RetireOperation(task).has_value());
  EXPECT_TRUE(tasks.Retire(task).has_value());
}

// Free slots are reused in rotation, so generations advance evenly across
// the table.
TEST(TaskTable, FreeSlotsAreReusedInRotation) {
  TaskTable tasks(3);
  std::vector<std::pair<std::uint32_t, std::uint32_t>> issued;
  for (int i = 0; i < 6; ++i) {
    const TaskId task = tasks.Create().value();
    issued.emplace_back(task.index(), task.generation());
    ASSERT_TRUE(tasks.Finish(task, TaskOutcome::kSucceeded).has_value());
    ASSERT_TRUE(tasks.Retire(task).has_value());
  }
  EXPECT_THAT(issued,
              ElementsAre(Pair(0, 1), Pair(1, 1), Pair(2, 1), Pair(0, 2), Pair(1, 2), Pair(2, 2)));
}

// A slot whose generation is exhausted retires and is skipped; the table
// refuses only once every slot has retired, and never reuses an identity.
TEST(TaskTable, AnExhaustedSlotRetiresAndIsSkipped) {
  TaskTable tasks(2, 2, 4, UINT32_MAX);
  const TaskId a = tasks.Create().value();
  EXPECT_EQ(a, TaskId(0, UINT32_MAX));
  EXPECT_TRUE(tasks.MakeReady(a, 1).value());
  ASSERT_TRUE(tasks.Finish(a, TaskOutcome::kSucceeded).has_value());
  ASSERT_TRUE(tasks.Retire(a).has_value());
  EXPECT_FALSE(tasks.exhausted());
  const TaskId b = tasks.Create().value();
  EXPECT_EQ(b, TaskId(1, UINT32_MAX));  // slot 0 retired: skipped
  EXPECT_EQ(Failed(tasks.Create()), TaskError::kFull);
  EXPECT_EQ(Failed(tasks.MakeReady(a, 1)), TaskError::kUnknownTask);
  EXPECT_TRUE(tasks.MakeReady(b, 1).value());
  EXPECT_EQ(tasks.NextReady(), b);
  ASSERT_TRUE(tasks.Finish(b, TaskOutcome::kSucceeded).has_value());
  ASSERT_TRUE(tasks.Retire(b).has_value());
  EXPECT_EQ(tasks.size(), 0U);
  EXPECT_TRUE(tasks.exhausted());
  EXPECT_EQ(Failed(tasks.Create()), TaskError::kFull);
  EXPECT_FALSE(tasks.Describe(b).has_value());
}

TEST(ReadyQueue, CoalescesAndServesRoundRobinByClass) {
  ReadyQueue ready(2, 8, 2);
  const TaskId a(0, 1);
  const TaskId b(1, 1);
  const TaskId c(2, 1);
  const TaskId background(3, 1);
  EXPECT_TRUE(ready.MakeReady(a, 1));
  EXPECT_FALSE(ready.MakeReady(a, 1));  // one entry per live task
  EXPECT_TRUE(ready.MakeReady(b, 1));
  EXPECT_TRUE(ready.MakeReady(background, 0));
  EXPECT_TRUE(ready.MakeReady(c, 1));
  std::vector<TaskId> order;
  // Interactive a, b; background passed over twice, so it goes next.
  for (int i = 0; i < 3; ++i) {
    order.push_back(Pop(ready));
    if (order.back() != background) {
      (void)ready.MakeReady(order.back(), 1);  // still has work
    }
  }
  EXPECT_THAT(order, ElementsAre(a, b, background));
  EXPECT_EQ(ready.Next(), c);
  ready.Remove(a);
  EXPECT_EQ(ready.Next(), b);
  EXPECT_FALSE(ready.Next().has_value());
  EXPECT_EQ(ready.size(), 0U);
}

}  // namespace
