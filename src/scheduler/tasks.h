// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Task trees and the ready queue (D-048,
// docs/async-model.md#tasks-operations-and-three-different-endings). The
// scheduler thread owns both; nothing here blocks or starts a thread.
//
// A request owns a bounded tree of tasks. Each task has a generation-tagged
// identity, an explicit phase (its program counter), its prepared operations
// and children, a cancellation state and at most one entry in the ready
// queue. Three endings stay separate: the client's outcome lives
// elsewhere; a task's outcome is recorded when it finishes computing; and
// its slot is retired only when every prepared operation has retired and
// every child has retired, so no operation or child ever names a reused
// slot. Cancelling a task closes it, and its whole subtree, to new children
// and operations; operations already prepared stay owned until they retire,
// including submissions whose acceptance has not arrived yet.
// A child's failure is recorded on its parent, which still drains its other
// children before it can retire.
//
// The table's capacity is fixed: when it is full, or a slot's generation is
// exhausted, creation is refused (stop admitting; never reuse an identity).
// Free slots are reused in rotation, so generations advance evenly across
// the table and exhaustion comes after about capacity x 2^32 tasks.

#ifndef LLMP_SCHEDULER_TASKS_H_
#define LLMP_SCHEDULER_TASKS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "base/ids.h"

namespace llmp::scheduler {

struct TaskTag {
  static constexpr const char* kName = "task";
};
using TaskId = base::Id<TaskTag>;

enum class TaskOutcome : std::uint8_t { kSucceeded, kFailed, kCancelled };

enum class TaskError : std::uint8_t {
  kUnknownTask,  // stale or invalid identity
  kFull,         // the table is at capacity
  kClosed,       // cancelled or finished: no new children or operations
  kFinished,     // already finished
  kBusy,         // operations or children still outstanding
};

std::string ToString(TaskError error);

struct TaskView {
  std::optional<TaskId> parent;
  std::uint32_t phase = 0;
  std::uint32_t operations = 0;  // prepared and not yet retired
  std::uint32_t children = 0;    // not yet retired
  bool cancelled = false;
  bool child_failed = false;
  std::optional<TaskOutcome> outcome;  // once finished
};

// Tasks ready to take a step, at most once each, served round-robin within
// a priority class and highest class first, except that a lower class
// passed over `aging_limit` times is served next (starvation prevention).
class ReadyQueue {
 public:
  // Room for `capacity` tasks (the admitted task bound) in each of
  // `classes` priority classes, allocated now.
  ReadyQueue(std::size_t classes, std::size_t capacity, std::uint32_t aging_limit);

  // False if the task is already queued (the wakeup coalesces). An index
  // or priority out of range, or an entry for an older generation of the
  // same slot, is a fatal invariant violation: the task table removes a
  // task's entry when it retires.
  bool MakeReady(TaskId task, std::size_t priority);
  // The next task to step, removed from the queue.
  std::optional<TaskId> Next();
  // Drops a task's entry, if it has one (a task that retires while ready).
  void Remove(TaskId task);

  std::size_t size() const { return size_; }

 private:
  struct Class {
    std::vector<TaskId> ring;
    std::size_t head = 0;
    std::size_t count = 0;
    std::uint32_t passed_over = 0;
  };
  static TaskId Take(Class& c);

  std::vector<Class> classes_;
  std::uint32_t aging_limit_;
  // Membership by task index: the generation queued, or 0.
  std::vector<std::uint32_t> queued_;
  std::size_t size_ = 0;
};

// The task table owns the ready queue, so only a live, unfinished task can
// be made ready, and a task's entry goes when it retires: a late wakeup for
// a retired task can neither run it nor block the task that reuses its slot.
class TaskTable {
 public:
  // At most `capacity` tasks, served from `classes` priority classes.
  // Every slot's generation starts at `first_generation`: a test hook
  // (base::SlotTable); production code leaves it at 1.
  explicit TaskTable(std::size_t capacity, std::size_t classes = 2, std::uint32_t aging_limit = 4,
                     std::uint32_t first_generation = 1)
      : capacity_(capacity),
        tasks_(capacity, first_generation),
        ready_(classes, capacity, aging_limit) {}

  // A new task, under `parent` if given.
  std::expected<TaskId, TaskError> Create(std::optional<TaskId> parent = std::nullopt);

  std::expected<void, TaskError> SetPhase(TaskId task, std::uint32_t phase);

  // Before handing an operation to a provider, acquire the task's lifetime
  // hold. The caller must prepare before publishing its command, not wait
  // for acceptance: cancellation can arrive while submission is unresolved.
  // A refused preparation must never be submitted.
  std::expected<void, TaskError> PrepareOperation(TaskId task);
  // Release one prepared hold only after proving not-started, or reconciling
  // acceptance and a terminal result with no further access. Cancellation
  // and a missing acceptance observation are not retirement proofs.
  std::expected<void, TaskError> RetireOperation(TaskId task);

  // Cancels the task and every live descendant: none of them takes new
  // children or operations. What they already prepared stays owned.
  std::expected<void, TaskError> Cancel(TaskId task);

  // The task stops computing with an outcome. A cancelled task can only
  // finish cancelled or failed. A failure is recorded on the parent.
  std::expected<void, TaskError> Finish(TaskId task, TaskOutcome outcome);

  // Frees the slot of a finished task with nothing outstanding; its parent
  // then has one child fewer.
  std::expected<void, TaskError> Retire(TaskId task);

  // The task has a step to take. False if it is already ready (the
  // wakeup coalesces). Refused for a stale identity or a finished task.
  std::expected<bool, TaskError> MakeReady(TaskId task, std::size_t priority);
  // The next task to step, if any.
  std::optional<TaskId> NextReady() { return ready_.Next(); }

  std::optional<TaskView> Describe(TaskId task) const;
  std::size_t size() const { return tasks_.size(); }
  std::size_t ready() const { return ready_.size(); }
  // Every slot's generation is exhausted: no task can be created again.
  bool exhausted() const { return tasks_.retired() == capacity_; }

 private:
  struct Task {
    TaskView view;
    // Children, as an intrusive list: no allocation per task.
    std::optional<TaskId> first_child;
    std::optional<TaskId> next_sibling;
    std::optional<TaskId> previous_sibling;
  };

  std::size_t capacity_;
  base::SlotTable<TaskTag, Task> tasks_;
  ReadyQueue ready_;
};

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_TASKS_H_
