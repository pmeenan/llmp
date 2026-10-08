// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A service lane (D-048, docs/async-model.md#execution-and-thread-ownership):
// worker threads that take owned commands from one bounded queue and carry
// them out; storage, device submission, device completion and CPU work
// each get their own lane, with the scheduler thread outside all of them.
// A worker never touches the scheduler's records: it reports what happened
// through the completion board and signals the owner.
//
// Several workers share a lane only if the provider's ordering and context
// contract permits it; with one worker, commands run in submission order.
// A full lane refuses a command, and the command stays with the caller
// (backpressure before submission). Cleanup commands may use the capacity
// held back for them.
//
// Closing refuses new commands; the workers carry out every command already
// queued (a command handler resolves those it cannot run, such as reporting
// "not started" during shutdown) and then exit. Destruction closes and
// joins: it never detaches a worker or drops a queued command.

#ifndef LLMP_SCHEDULER_LANE_H_
#define LLMP_SCHEDULER_LANE_H_

#include <cstddef>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/check.h"

namespace llmp::scheduler {

struct LaneSettings {
  std::string name;
  std::size_t capacity = 0;  // queued commands, including the reserve
  std::size_t reserved = 0;  // held back for cleanup commands
  std::size_t workers = 1;
};

template <typename Command>
class Lane {
 public:
  // Runs on a worker thread; with several workers, concurrently.
  using Handler = std::move_only_function<void(Command&&) const>;

  // Starts the workers. Only programs build lanes (docs/architecture.md:
  // library code starts no threads of its own).
  Lane(LaneSettings settings, Handler handler)
      : settings_(std::move(settings)),
        queue_(settings_.capacity, settings_.reserved),
        handler_(std::move(handler)) {
    base::Check(settings_.workers > 0, "a lane needs a worker");
    workers_.reserve(settings_.workers);
    for (std::size_t i = 0; i < settings_.workers; ++i) {
      workers_.emplace_back([this] { Work(); });
    }
  }

  Lane(const Lane&) = delete;
  Lane& operator=(const Lane&) = delete;
  Lane(Lane&&) = delete;
  Lane& operator=(Lane&&) = delete;

  ~Lane() {
    Close();
    Join();
  }

  // Moves from `command` only if it is accepted.
  base::PushResult Submit(Command&& command, base::PushKind kind = base::PushKind::kOrdinary) {
    return queue_.TryPush(std::move(command), kind);
  }

  // No new commands; queued ones still run.
  void Close() { queue_.Close(); }

  // Waits for the workers to drain the queue and exit. Close first.
  void Join() {
    for (std::jthread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  const std::string& name() const { return settings_.name; }
  std::size_t queued() const { return queue_.size(); }

 private:
  void Work() {
    // No stop token: a worker leaves only once the queue is closed and
    // drained, so no accepted command is abandoned.
    const std::stop_token never;
    while (std::optional<Command> command = queue_.Pop(never)) {
      handler_(std::move(*command));
    }
  }

  LaneSettings settings_;
  base::BoundedQueue<Command> queue_;
  Handler handler_;
  std::vector<std::jthread> workers_;  // last: joined before the rest go
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_LANE_H_
