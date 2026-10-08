// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// D-048's lane measurements (docs/m2-record.md, "Task lanes"): how quickly a
// sleeping owner wakes for a published completion, and at what CPU cost,
// against polling; lane throughput by worker count and queue capacity; and
// the completion board's round trip by mailbox count. Prints Markdown
// tables. Numbers are this host's, under whatever else it is running: the
// report beside the results says where and how they were taken
// (docs/experiments/task-lanes/).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <print>
#include <thread>
#include <vector>

#include "base/bounded_queue.h"
#include "base/wake.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"

namespace {

using Clock = std::chrono::steady_clock;
using llmp::base::PushResult;
using llmp::base::WakeFlag;
using llmp::scheduler::Acceptance;
using llmp::scheduler::CompletionBoard;
using llmp::scheduler::Lane;
using llmp::scheduler::LaneSettings;
using llmp::scheduler::Observation;
using llmp::scheduler::OperationId;
using llmp::scheduler::Outcome;
using llmp::scheduler::Terminal;

std::int64_t Nanos(Clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

// This thread's CPU time.
double ThreadCpuSeconds() {
  timespec ts{};
  (void)::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + (static_cast<double>(ts.tv_nsec) / 1e9);
}

double Percentile(std::vector<double> samples, double p) {
  if (samples.empty()) {
    return 0;
  }
  std::ranges::sort(samples);
  const auto at = static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1));
  return samples[at];
}

void Spin(std::chrono::nanoseconds work) {
  const Clock::time_point until = Clock::now() + work;
  while (Clock::now() < until) {
  }
}

enum class OwnerWait : std::uint8_t { kSleep, kPollYield, kPollSleep, kSpinThenSleep };

const char* Name(OwnerWait mode) {
  switch (mode) {
    case OwnerWait::kSleep:
      return "sleep on the wake flag";
    case OwnerWait::kPollYield:
      return "poll, yielding";
    case OwnerWait::kPollSleep:
      return "poll every 50 us";
    case OwnerWait::kSpinThenSleep:
      return "poll, yielding, for 200 us, then sleep";
  }
  return "?";
}

// One producer publishes a timestamp and signals, then waits for the owner
// to see it and idles for a while, so the owner is waiting when the next
// signal comes. Latency is signal to owner running; CPU is the owner's.
void WakeLatency(OwnerWait mode, int samples) {
  WakeFlag wake;
  std::atomic<std::int64_t> sent{0};
  std::atomic<int> seen{0};
  std::vector<double> latencies_us;
  latencies_us.reserve(static_cast<std::size_t>(samples));
  const Clock::time_point start = Clock::now();
  double owner_cpu = 0;
  std::jthread owner([&] {
    const double cpu_start = ThreadCpuSeconds();
    for (int i = 0; i < samples; ++i) {
      const Clock::time_point spin_until = Clock::now() + std::chrono::microseconds(200);
      while (true) {
        if (mode == OwnerWait::kSleep ||
            (mode == OwnerWait::kSpinThenSleep && Clock::now() >= spin_until)) {
          if (wake.WaitFor(std::chrono::seconds(1))) {
            break;
          }
        } else if (wake.Consume()) {
          break;
        } else if (mode == OwnerWait::kPollYield || mode == OwnerWait::kSpinThenSleep) {
          std::this_thread::yield();
        } else {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      }
      const std::int64_t now = Nanos(Clock::now());
      latencies_us.push_back(static_cast<double>(now - sent.load(std::memory_order_acquire)) / 1e3);
      seen.store(i + 1, std::memory_order_release);
    }
    owner_cpu = ThreadCpuSeconds() - cpu_start;
  });
  for (int i = 0; i < samples; ++i) {
    // Idle 100-400 us so the owner goes back to waiting.
    std::this_thread::sleep_for(std::chrono::microseconds(100 + ((i * 37) % 300)));
    sent.store(Nanos(Clock::now()), std::memory_order_release);
    wake.Signal();
    while (seen.load(std::memory_order_acquire) <= i) {
      std::this_thread::yield();
    }
  }
  owner.join();
  const double wall = std::chrono::duration<double>(Clock::now() - start).count();
  std::println("| {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.1f}% |", Name(mode), samples,
               Percentile(latencies_us, 0.5), Percentile(latencies_us, 0.99),
               Percentile(latencies_us, 1.0), 100.0 * owner_cpu / wall);
}

// One producer submits `commands` commands of `work` each to a lane,
// retrying when the lane is full.
void LaneThroughput(std::size_t workers, std::size_t capacity, std::chrono::nanoseconds work,
                    int commands) {
  std::atomic<int> done{0};
  std::uint64_t refusals = 0;
  const Clock::time_point start = Clock::now();
  {
    Lane<std::uint64_t> lane(
        LaneSettings{.name = "bench", .capacity = capacity, .reserved = 0, .workers = workers},
        [&](std::uint64_t&&) {
          if (work.count() > 0) {
            Spin(work);
          }
          done.fetch_add(1, std::memory_order_relaxed);
        });
    for (int i = 0; i < commands; ++i) {
      while (lane.Submit(static_cast<std::uint64_t>(i)) == PushResult::kFull) {
        ++refusals;
        std::this_thread::yield();
      }
    }
  }  // drains and joins
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  std::println("| {} | {} | {} | {:.0f} | {:.2f} |", workers, capacity, work.count(),
               static_cast<double>(done.load()) / seconds / 1e3,
               static_cast<double>(refusals) / static_cast<double>(commands));
}

// The owner keeps `mailboxes` operations in flight through a provider lane
// that accepts and completes each at once, harvesting and closing as news
// arrives: operations per second and open-to-close latency.
void BoardRoundTrip(std::size_t mailboxes, std::size_t providers, int operations) {
  WakeFlag wake;
  CompletionBoard board(mailboxes, wake);
  std::vector<std::int64_t> opened_at(mailboxes, 0);
  std::vector<double> latencies_us;
  latencies_us.reserve(static_cast<std::size_t>(operations));
  const Clock::time_point start = Clock::now();
  {
    Lane<OperationId> lane(
        LaneSettings{
            .name = "provider", .capacity = mailboxes, .reserved = 0, .workers = providers},
        [&board](OperationId&& op) {
          (void)board.Accept(op, Acceptance::kAccepted);
          (void)board.Complete(
              op, Terminal{.outcome = Outcome::kSucceeded, .bytes = 1, .no_further_access = true});
        });
    int opened = 0;
    int closed = 0;
    while (closed < operations) {
      while (opened < operations) {
        const OperationId op = board.Open();
        if (!op.valid()) {
          break;
        }
        opened_at[op.index()] = Nanos(Clock::now());
        if (lane.Submit(OperationId{op}) != PushResult::kAccepted) {
          std::println(stderr, "provider lane refused an operation");
          return;
        }
        ++opened;
      }
      (void)wake.Consume();
      for (const Observation& seen : board.Harvest(mailboxes)) {
        if (seen.terminal && seen.acceptance == Acceptance::kAccepted &&
            board.Close(seen.operation)) {
          latencies_us.push_back(
              static_cast<double>(Nanos(Clock::now()) - opened_at[seen.operation.index()]) / 1e3);
          ++closed;
        }
      }
      if (closed < opened && board.open() > 0) {
        (void)wake.WaitFor(std::chrono::seconds(1));
      }
    }
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  std::println("| {} | {} | {:.0f} | {:.1f} | {:.1f} |", mailboxes, providers,
               static_cast<double>(operations) / seconds / 1e3, Percentile(latencies_us, 0.5),
               Percentile(latencies_us, 0.99));
}

}  // namespace

int main() {
  std::println("hardware threads: {}\n", std::thread::hardware_concurrency());

  std::println("## Wakeup: signal to owner running\n");
  std::println("| Owner waits by | Samples | p50 us | p99 us | max us | Owner CPU |");
  std::println("| --- | ---: | ---: | ---: | ---: | ---: |");
  for (const OwnerWait mode : {OwnerWait::kSleep, OwnerWait::kPollYield, OwnerWait::kPollSleep,
                               OwnerWait::kSpinThenSleep}) {
    WakeLatency(mode, 5000);
  }

  std::println("\n## Lane throughput: one producer, commands of fixed work\n");
  std::println("| Workers | Capacity | Work ns | k commands/s | Refusals per command |");
  std::println("| ---: | ---: | ---: | ---: | ---: |");
  for (const std::chrono::nanoseconds work :
       {std::chrono::nanoseconds(0), std::chrono::nanoseconds(2000)}) {
    for (const std::size_t workers : {1U, 2U, 4U, 8U}) {
      for (const std::size_t capacity : {16U, 256U}) {
        LaneThroughput(workers, capacity, work, work.count() == 0 ? 400000 : 100000);
      }
    }
  }

  std::println(
      "\n## Completion board round trip: open, provider accepts and completes, harvest, close\n");
  std::println("| Mailboxes | Provider workers | k ops/s | p50 us | p99 us |");
  std::println("| ---: | ---: | ---: | ---: | ---: |");
  for (const std::size_t mailboxes : {8U, 64U, 512U}) {
    for (const std::size_t providers : {1U, 4U}) {
      BoardRoundTrip(mailboxes, providers, 200000);
    }
  }
  return 0;
}
