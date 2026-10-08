// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/tasks.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "base/check.h"

namespace llmp::scheduler {

std::string ToString(TaskError error) {
  switch (error) {
    case TaskError::kUnknownTask:
      return "stale or unknown task";
    case TaskError::kFull:
      return "the task table is full";
    case TaskError::kClosed:
      return "the task takes no new children or operations";
    case TaskError::kFinished:
      return "the task has already finished";
    case TaskError::kBusy:
      return "the task still has operations or children outstanding";
  }
  return "unknown task error";
}

std::expected<TaskId, TaskError> TaskTable::Create(std::optional<TaskId> parent) {
  Task* parent_task = nullptr;
  if (parent) {
    parent_task = tasks_.Find(*parent);
    if (parent_task == nullptr) {
      return std::unexpected(TaskError::kUnknownTask);
    }
    if (parent_task->view.cancelled || parent_task->view.outcome) {
      return std::unexpected(TaskError::kClosed);
    }
  }
  if (tasks_.size() >= capacity_) {
    return std::unexpected(TaskError::kFull);
  }
  const TaskId id = tasks_.Insert(Task{.view = {.parent = parent,
                                                .phase = 0,
                                                .operations = 0,
                                                .children = 0,
                                                .cancelled = false,
                                                .child_failed = false,
                                                .outcome = std::nullopt},
                                       .first_child = std::nullopt,
                                       .next_sibling = std::nullopt,
                                       .previous_sibling = std::nullopt});
  // The slots are fixed, so per-index tables sized to the capacity (the
  // ready queue's) cover every task. A slot whose generation is exhausted
  // is retired: with every other slot live or retired, creation is refused
  // instead of reusing an identity.
  if (!id.valid()) {
    return std::unexpected(TaskError::kFull);
  }
  base::Check(id.index() < capacity_, "a task slot beyond the table's capacity");
  if (parent) {
    parent_task = tasks_.Find(*parent);
    Task& child = *tasks_.Find(id);
    child.next_sibling = parent_task->first_child;
    if (parent_task->first_child) {
      tasks_.Find(*parent_task->first_child)->previous_sibling = id;
    }
    parent_task->first_child = id;
    ++parent_task->view.children;
  }
  return id;
}

std::expected<void, TaskError> TaskTable::SetPhase(TaskId task, std::uint32_t phase) {
  Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  if (found->view.outcome) {
    return std::unexpected(TaskError::kFinished);
  }
  found->view.phase = phase;
  return {};
}

std::expected<void, TaskError> TaskTable::PrepareOperation(TaskId task) {
  Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  if (found->view.cancelled || found->view.outcome) {
    return std::unexpected(TaskError::kClosed);
  }
  base::Check(found->view.operations < UINT32_MAX, "operation count overflow");
  ++found->view.operations;
  return {};
}

std::expected<void, TaskError> TaskTable::RetireOperation(TaskId task) {
  Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  base::Check(found->view.operations > 0, "retiring an operation the task does not have");
  --found->view.operations;
  return {};
}

std::expected<void, TaskError> TaskTable::Cancel(TaskId task) {
  Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  // Depth-first over the subtree without recursion or allocation: down
  // through first children, across siblings, back up through parents.
  TaskId at = task;
  while (true) {
    Task& current = *tasks_.Find(at);
    current.view.cancelled = true;
    if (current.first_child) {
      at = *current.first_child;
      continue;
    }
    // Climb until a node has a next sibling, stopping at the root. Every
    // node below the root has a parent.
    while (at != task && !tasks_.Find(at)->next_sibling) {
      at = tasks_.Find(at)->view.parent.value_or(task);
    }
    if (at == task) {
      return {};
    }
    at = tasks_.Find(at)->next_sibling.value_or(task);
  }
}

std::expected<void, TaskError> TaskTable::Finish(TaskId task, TaskOutcome outcome) {
  Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  if (found->view.outcome) {
    return std::unexpected(TaskError::kFinished);
  }
  if (found->view.cancelled && outcome == TaskOutcome::kSucceeded) {
    outcome = TaskOutcome::kCancelled;  // cancellation wins over a late success
  }
  found->view.outcome = outcome;
  ready_.Remove(task);  // a finished task takes no more steps
  if (outcome == TaskOutcome::kFailed && found->view.parent) {
    tasks_.Find(*found->view.parent)->view.child_failed = true;
  }
  return {};
}

std::expected<void, TaskError> TaskTable::Retire(TaskId task) {
  Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  if (!found->view.outcome) {
    return std::unexpected(TaskError::kBusy);  // still computing
  }
  if (found->view.operations > 0 || found->view.children > 0) {
    return std::unexpected(TaskError::kBusy);
  }
  if (found->view.parent) {
    Task& parent = *tasks_.Find(*found->view.parent);
    if (found->previous_sibling) {
      tasks_.Find(*found->previous_sibling)->next_sibling = found->next_sibling;
    } else {
      parent.first_child = found->next_sibling;
    }
    if (found->next_sibling) {
      tasks_.Find(*found->next_sibling)->previous_sibling = found->previous_sibling;
    }
    --parent.view.children;
  }
  ready_.Remove(task);
  (void)tasks_.Erase(task);
  return {};
}

std::expected<bool, TaskError> TaskTable::MakeReady(TaskId task, std::size_t priority) {
  const Task* found = tasks_.Find(task);
  if (found == nullptr) {
    return std::unexpected(TaskError::kUnknownTask);
  }
  if (found->view.outcome) {
    return std::unexpected(TaskError::kFinished);
  }
  return ready_.MakeReady(task, priority);
}

std::optional<TaskView> TaskTable::Describe(TaskId task) const {
  const Task* found = tasks_.Find(task);
  return found != nullptr ? std::optional<TaskView>(found->view) : std::nullopt;
}

ReadyQueue::ReadyQueue(std::size_t classes, std::size_t capacity, std::uint32_t aging_limit)
    : classes_(classes), aging_limit_(aging_limit), queued_(capacity, 0) {
  base::Check(classes > 0 && capacity > 0, "a ready queue needs classes and capacity");
  for (Class& c : classes_) {
    c.ring.resize(capacity);
  }
}

bool ReadyQueue::MakeReady(TaskId task, std::size_t priority) {
  base::Check(task.valid() && task.index() < queued_.size() && priority < classes_.size(),
              "a ready task outside the queue's bounds");
  if (queued_[task.index()] == task.generation()) {
    return false;  // coalesced: at most one entry per live task
  }
  base::Check(queued_[task.index()] == 0, "a ready entry for a retired task");
  Class& c = classes_[priority];
  base::Check(c.count < c.ring.size(), "ready queue overflow");
  c.ring[(c.head + c.count) % c.ring.size()] = task;
  ++c.count;
  queued_[task.index()] = task.generation();
  ++size_;
  return true;
}

TaskId ReadyQueue::Take(Class& c) {
  const TaskId task = c.ring[c.head];
  c.head = (c.head + 1) % c.ring.size();
  --c.count;
  if (c.count == 0) {
    c.passed_over = 0;
  }
  return task;
}

std::optional<TaskId> ReadyQueue::Next() {
  if (size_ == 0) {
    return std::nullopt;
  }
  // A class passed over often enough goes first (the most passed over;
  // ties to the higher class); otherwise the highest class with work.
  std::optional<std::size_t> chosen;
  for (std::size_t i = classes_.size(); i > 0; --i) {
    const Class& c = classes_[i - 1];
    if (c.count > 0 && c.passed_over >= aging_limit_ &&
        (!chosen || c.passed_over > classes_[*chosen].passed_over)) {
      chosen = i - 1;
    }
  }
  if (!chosen) {
    for (std::size_t i = classes_.size(); i > 0; --i) {
      if (classes_[i - 1].count > 0) {
        chosen = i - 1;
        break;
      }
    }
  }
  const std::size_t served = chosen.value_or(0);
  for (std::size_t i = 0; i < served; ++i) {
    if (classes_[i].count > 0) {
      ++classes_[i].passed_over;
    }
  }
  classes_[served].passed_over = 0;
  const TaskId task = Take(classes_[served]);
  queued_[task.index()] = 0;
  --size_;
  return task;
}

void ReadyQueue::Remove(TaskId task) {
  if (!task.valid() || task.index() >= queued_.size() ||
      queued_[task.index()] != task.generation()) {
    return;
  }
  for (Class& c : classes_) {
    for (std::size_t k = 0; k < c.count; ++k) {
      if (c.ring[(c.head + k) % c.ring.size()] != task) {
        continue;
      }
      // Close the gap, keeping the others in order.
      for (std::size_t m = k; m + 1 < c.count; ++m) {
        c.ring[(c.head + m) % c.ring.size()] = c.ring[(c.head + m + 1) % c.ring.size()];
      }
      --c.count;
      if (c.count == 0) {
        c.passed_over = 0;
      }
      queued_[task.index()] = 0;
      --size_;
      return;
    }
  }
}

}  // namespace llmp::scheduler
