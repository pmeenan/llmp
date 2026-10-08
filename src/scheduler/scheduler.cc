// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "base/bounded_queue.h"
#include "base/check.h"
#include "catalog/catalog.h"
#include "providers/direct_reader.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/tasks.h"

namespace jitllm::scheduler {
namespace {
std::atomic<std::uint64_t> next_placement_instance{1};
}  // namespace

std::uint64_t Scheduler::TakePlacementInstance(std::atomic<std::uint64_t>& next) {
  auto token = next.load(std::memory_order_relaxed);
  while (token != UINT64_MAX) {
    if (next.compare_exchange_weak(token, token + 1, std::memory_order_relaxed)) return token;
  }
  return UINT64_MAX;
}

WorkError Scheduler::ErrorOf(catalog::CatalogError error) {
  switch (error) {
    case catalog::CatalogError::kNotResident:
      return WorkError::kNotResident;
    case catalog::CatalogError::kStaleContent:
      return WorkError::kStale;
    case catalog::CatalogError::kOverBudget:
      return WorkError::kOverBudget;
    case catalog::CatalogError::kExhausted:
    case catalog::CatalogError::kHeld:
    case catalog::CatalogError::kWrongState:
      return WorkError::kBusy;
    default:
      return WorkError::kUnavailable;
  }
}

std::string ToString(WorkError error) {
  switch (error) {
    case WorkError::kClosed:
      return "the task takes no new work";
    case WorkError::kBusy:
      return "no room now: try again later";
    case WorkError::kNotResident:
      return "an extent is not resident";
    case WorkError::kStale:
      return "the contents changed since the closure was taken";
    case WorkError::kOverBudget:
      return "loading would exceed the execution budget";
    case WorkError::kUnavailable:
      return "an extent is unavailable";
    case WorkError::kInvalid:
      return "malformed work, or no lane for it";
  }
  return "unknown work error";
}

std::string ToString(Fault fault) {
  switch (fault) {
    case Fault::kContradiction:
      return "a lane's observations contradict each other";
    case Fault::kUnproven:
      return "an operation ended without proof of no further access";
    case Fault::kBacking:
      return "backing could not be unmapped or released as the catalog needs";
    case Fault::kExhausted:
      return "a page-in stage can never get an operation identity (all retired or quarantined)";
  }
  return "unknown fault";
}

Scheduler::Scheduler(catalog::Catalog& catalog, CompletionBoard& board, base::WakeFlag& wake,
                     Lanes lanes, const SchedulerSettings& settings)
    : catalog_(catalog),
      board_(board),
      wake_(wake),
      lanes_(lanes),
      settings_(settings),
      controls_(settings.controls, settings.controls_reserved),
      tasks_(settings.tasks, settings.priorities, settings.aging_limit),
      records_(settings.tasks),
      operations_(board.capacity()) {
  placement_stamp_.instance = TakePlacementInstance(next_placement_instance);
  base::Check(settings_.controls_per_turn > 0 && settings_.observations_per_turn > 0 &&
                  settings_.steps_per_turn > 0 && settings_.waiters > 0,
              "a scheduler turn needs non-zero batches and waiter lists");
  // Both are compared or waited on in the clock's nanoseconds, where an
  // unbounded value overflows (milliseconds::max() in a timed wait); a tick
  // that is not positive would spin instead of sleeping.
  base::Check(settings_.tick > std::chrono::milliseconds::zero() &&
                  settings_.tick <= SchedulerSettings::kLongest,
              "a scheduler tick must be positive and at most kLongest");
  base::Check(settings_.poll_window >= std::chrono::microseconds::zero() &&
                  settings_.poll_window <= SchedulerSettings::kLongest &&
                  settings_.follow_limit >= std::chrono::microseconds::zero() &&
                  settings_.follow_limit <= SchedulerSettings::kLongest,
              "a scheduler poll window and follow limit must be non-negative and at most kLongest");
  cancel_intents_.reserve(settings_.controls);
  base::Check(settings_.landing.slots.empty() || settings_.landing.slot_bytes > Bytes(),
              "a landing zone needs slots of a non-zero size");
  const DeviceService* copier = lanes_.copy != nullptr ? lanes_.copy : lanes_.device;
  base::Check(settings_.landing.slots.empty() ||
                  (copier != nullptr && settings_.landing.stream < copier->streams()),
              "a landing zone's stream must be one of the copy lane's streams (or, without one, "
              "the device lane's)");
  slots_.assign(settings_.landing.slots.size(), SlotState::kFree);
}

base::PushResult Scheduler::Post(Control&& control) {
  base::PushResult pushed = base::PushResult::kAccepted;
  {
    const std::scoped_lock lock(cancel_mutex_);
    if (const auto* cancel = std::get_if<CancelRequest>(&control)) {
      const std::uint64_t request = cancel->request;
      if (std::ranges::contains(cancel_intents_, request)) {
        // Queued already, with no start since: the owner applies it after
        // this call (it drops the entry before applying), so this intent
        // is covered at no cost.
        return base::PushResult::kAccepted;
      }
      pushed = controls_.TryPush(std::move(control), base::PushKind::kCleanup);
      if (pushed == base::PushResult::kAccepted) {
        cancel_intents_.push_back(request);  // one per queued control: within capacity
      }
    } else if (std::holds_alternative<SignalRequest>(control)) {
      // Orders nothing against starts or cancellations of the request.
      pushed = controls_.TryPush(std::move(control), base::PushKind::kOrdinary);
    } else {
      const std::uint64_t request = std::get<StartRequest>(control).request;
      pushed = controls_.TryPush(std::move(control), base::PushKind::kOrdinary);
      if (pushed == base::PushResult::kAccepted) {
        // A later cancellation must follow this start in the queue.
        std::erase(cancel_intents_, request);
      }
    }
  }
  if (pushed == base::PushResult::kAccepted) {
    wake_.Signal();
  }
  return pushed;
}

void Scheduler::RequestShutdown() {
  shutdown_.store(true, std::memory_order_release);
  wake_.Signal();
}

Scheduler::TaskRecord* Scheduler::Record(TaskId task) {
  if (!task.valid() || task.index() >= records_.size()) {
    return nullptr;
  }
  std::optional<TaskRecord>& record = records_[task.index()];
  return record && record->id == task ? &*record : nullptr;
}

TaskView Scheduler::ViewOf(TaskId task) const {
  const std::optional<TaskView> view = tasks_.Describe(task);
  base::Check(view.has_value(), "a view of a task that is not live");
  return view.value_or(TaskView{});
}

Scheduler::Operation* Scheduler::Find(OperationId operation) {
  if (!operation.valid() || operation.index() >= operations_.size()) {
    return nullptr;
  }
  std::optional<Operation>& record = operations_[operation.index()];
  return record && record->id == operation ? &*record : nullptr;
}

// Tasks ------------------------------------------------------------------------------

std::expected<TaskId, StartError> Scheduler::Start(std::uint64_t request,
                                                   std::unique_ptr<TaskProgram> program,
                                                   std::size_t priority) {
  return StartTask(request, program, priority, std::nullopt);
}

std::expected<TaskId, StartError> Scheduler::StartTask(std::uint64_t request,
                                                       std::unique_ptr<TaskProgram>& program,
                                                       std::size_t priority,
                                                       std::optional<TaskId> parent) {
  if (program == nullptr || priority >= settings_.priorities) {
    return std::unexpected(StartError::kInvalid);
  }
  // A faulted node admits no new request; live ones may still unwind. Nor
  // does one whose task or operation identities are all exhausted: the ID
  // space ends by refusing, never by reuse.
  if (!parent && (stopping_ || fault_ || tasks_.exhausted() || board_.exhausted())) {
    return std::unexpected(StartError::kStopped);
  }
  const auto created = tasks_.Create(parent);
  if (!created) {
    return std::unexpected(created.error() == TaskError::kFull ? StartError::kFull
                                                               : StartError::kClosed);
  }
  const TaskId id = *created;
  records_[id.index()] = TaskRecord{.id = id,
                                    .request = request,
                                    .priority = priority,
                                    .program = std::move(program),
                                    .waiting = 0,
                                    .failed = false,
                                    .finished = false};
  if (parent) {
    ++Record(*parent)->waiting;  // the parent waits for its children
  }
  MakeReady(*Record(id));
  return id;
}

bool Scheduler::Cancel(std::uint64_t request) {
  // Every root of the request, even one that finished: its children may
  // still run, and a tag used again while an earlier start drains names
  // both starts.
  bool found = false;
  for (const std::optional<TaskRecord>& record : records_) {
    // CancelTask may retire records, never move them.
    if (record && record->request == request && !ViewOf(record->id).parent) {
      found = true;
      CancelTask(record->id);
    }
  }
  return found;
}

void Scheduler::CancelTask(TaskId task) {
  // Closes the subtree to new children and operations, then finishes every
  // task in it; what they published stays owned until it drains.
  base::Check(tasks_.Cancel(task).has_value(), "cancelling a live task");
  for (const std::optional<TaskRecord>& record : records_) {
    // FinishTask may retire this or other records, never move them.
    if (record && !record->finished && ViewOf(record->id).cancelled) {
      FinishTask(record->id, TaskOutcome::kCancelled);
    }
  }
}

void Scheduler::MakeReady(TaskRecord& record) {
  if (!record.finished) {
    base::Check(tasks_.MakeReady(record.id, record.priority).has_value(),
                "an unfinished task can be made ready");
  }
}

void Scheduler::Wake(TaskId task, bool failed) {
  TaskRecord* record = Record(task);
  if (record == nullptr || record->finished) {
    return;
  }
  base::Check(record->waiting > 0, "a task woken for a dependency it did not wait on");
  --record->waiting;
  record->failed = record->failed || failed;
  if (record->waiting == 0) {
    MakeReady(*record);
  }
}

void Scheduler::StepTask(TaskId task) {
  TaskRecord* record = Record(task);
  if (record == nullptr || record->finished) {
    return;
  }
  TaskContext context(*this, task);
  const Step step = record->program->Advance(context);
  record = Record(task);  // still live: only a finished task retires
  switch (step.kind) {
    case Step::Kind::kYield:
      MakeReady(*record);
      break;
    case Step::Kind::kWait:
      if (record->waiting == 0) {
        MakeReady(*record);  // everything it waited for has already settled
      }
      break;
    case Step::Kind::kFinish:
      FinishTask(task, step.outcome);
      break;
  }
}

void Scheduler::FinishTask(TaskId task, TaskOutcome outcome) {
  TaskRecord* record = Record(task);
  if (record == nullptr || record->finished) {
    return;
  }
  base::Check(tasks_.Finish(task, outcome).has_value(), "finishing a live task");
  const TaskView view = ViewOf(task);
  const TaskOutcome finished = view.outcome.value_or(outcome);
  record->finished = true;
  // Told first: a rollback below may retire it.
  record->program->Finished(finished);
  Withdraw(task);
  if (finished == TaskOutcome::kCancelled) {
    // Cancel queued work: commands no lane has taken never reach one.
    for (std::deque<OperationId>& pending : pending_) {
      for (const OperationId id : std::vector<OperationId>(pending.begin(), pending.end())) {
        Operation* operation = Find(id);
        if (operation != nullptr && !operation->published &&
            (operation->kind == Kind::kDevice || operation->kind == Kind::kCpu) &&
            operation->task == task) {
          RollBack(*operation);
        }
      }
    }
  }
  if (view.parent) {
    Wake(*view.parent, false);  // a failure shows in its view (child_failed)
  }
  TryRetire(task);
}

void Scheduler::TryRetire(TaskId task) {
  std::optional<TaskId> next = task;
  while (next) {
    const std::optional<TaskView> view = tasks_.Describe(*next);
    TaskRecord* record = Record(*next);
    if (!view || record == nullptr || !tasks_.Retire(*next)) {
      return;  // still computing, or operations or children outstanding
    }
    record->program->Retired();
    records_[next->index()].reset();
    next = view->parent;  // the parent may have been waiting on this child
  }
}

void Scheduler::Apply(Control&& control) {
  if (auto* start = std::get_if<StartRequest>(&control)) {
    const auto started = StartTask(start->request, start->program, start->priority, std::nullopt);
    if (!started && start->program != nullptr) {
      // Never admitted: it hears so, and goes with the control.
      start->program->Finished(stopping_ ? TaskOutcome::kCancelled : TaskOutcome::kFailed);
    }
    return;
  }
  if (const auto* signal = std::get_if<SignalRequest>(&control)) {
    Signal(signal->request);
    return;
  }
  (void)Cancel(std::get<CancelRequest>(control).request);
}

void Scheduler::Signal(std::uint64_t request) {
  for (std::optional<TaskRecord>& record : records_) {
    if (!record || record->request != request || record->finished || ViewOf(record->id).parent) {
      continue;
    }
    // Kept for the task's next AwaitSignal, which takes it: woken now if
    // it waits for one.
    record->signalled = true;
    if (record->awaiting_signal) {
      record->awaiting_signal = false;
      Wake(record->id, false);
    }
  }
}

std::expected<Readiness, WorkError> Scheduler::AwaitSignal(TaskId task) {
  TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  if (record->signalled) {
    record->signalled = false;
    return Readiness::kReady;
  }
  if (!record->awaiting_signal) {
    record->awaiting_signal = true;
    ++record->waiting;
  }
  return Readiness::kWaiting;
}

void Scheduler::BeginStop() {
  if (stopping_) {
    return;
  }
  stopping_ = true;
  controls_.Close();  // what is queued is still applied, and refused
  // Every unfinished task, including children of a root that finished.
  for (const std::optional<TaskRecord>& record : records_) {
    if (record && !record->finished) {
      CancelTask(record->id);
    }
  }
}

// Work -------------------------------------------------------------------------------

std::expected<OperationId, WorkError> Scheduler::Submit(TaskId task,
                                                        const catalog::Closure* closure,
                                                        catalog::LeaseId held, Kind kind,
                                                        DeviceCommand& device, CpuJob& job) {
  bool valid = false;
  if (kind == Kind::kDevice && lanes_.device != nullptr) {
    if (const auto* copies = std::get_if<DeviceWork>(&device.work)) {
      valid = copies->count > 0 && copies->count <= kMaxDeviceCopies;
    } else if (const auto* launch = std::get_if<LaunchWork>(&device.work)) {
      valid = static_cast<bool>(launch->job);
    }
  } else if (kind == Kind::kCpu) {
    valid = lanes_.cpu != nullptr && static_cast<bool>(job);
  }
  if (!valid || (closure == nullptr) == !held.valid()) {
    return std::unexpected(WorkError::kInvalid);
  }
  TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  // Under a request's lease: the task's own, not ending.
  Held* under = nullptr;
  if (held.valid()) {
    const auto found = held_.find(held);
    if (found == held_.end() || found->second.task != task) {
      return std::unexpected(WorkError::kInvalid);
    }
    if (found->second.ending) {
      return std::unexpected(WorkError::kClosed);
    }
    under = &found->second;
  }
  // Not open() against capacity(): a mailbox whose generation is exhausted
  // is retired and never issued again.
  if (board_.available() == 0) {
    return std::unexpected(NoMailboxEver() ? WorkError::kUnavailable : WorkError::kBusy);
  }
  // Prepare under the owner: generations checked and leases taken, all or
  // none (or the request's lease counts one more operation under it), then
  // the task's lifetime hold and the record, before any lane can touch the
  // memory.
  catalog::LeaseId lease;
  if (under != nullptr) {
    ++under->operations;
    ++stats_.held_operations;
  } else {
    const auto acquired = catalog_.AcquireLease(*closure);
    if (!acquired) {
      return std::unexpected(ErrorOf(acquired.error()));
    }
    lease = *acquired;
    base::Check(catalog_.RecordUse(lease, turn_).has_value(), "recording a fresh lease's use");
  }
  base::Check(tasks_.PrepareOperation(task).has_value(), "an open task takes an operation");
  Operation& operation = Open(kind);
  operation.route = kind == Kind::kCpu ? Route::kCpu : Route::kDevice;
  operation.task = task;
  operation.lease = lease;
  operation.held = held;
  if (kind == Kind::kCpu) {
    operation.job = CpuCommand{.operation = operation.id, .job = std::move(job)};
  } else {
    operation.device = std::move(device);
    operation.device.operation = operation.id;
  }
  ++record->waiting;
  SetCritical(operation, true);
  const OperationId id = operation.id;
  if (kind == Kind::kDevice) {
    NoteFollow();  // the client's pace, from the last step's end to this one's publication
  }
  Publish(operation);
  return id;
}

// Requests' leases --------------------------------------------------------------------

std::expected<catalog::LeaseId, WorkError> Scheduler::HoldLease(
    TaskId task, const catalog::Closure& closure,
    std::shared_ptr<const std::atomic<std::uint32_t>> external) {
  const TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  if (held_.size() >= settings_.tasks) {
    return std::unexpected(WorkError::kBusy);
  }
  // All or none, at the recorded contents: exactly a per-operation lease,
  // taken once. An extent whose eviction has begun cannot be leased
  // (invariant 6), and none can begin while this is held.
  const auto lease = catalog_.AcquireLease(closure);
  if (!lease) {
    return std::unexpected(ErrorOf(lease.error()));
  }
  base::Check(catalog_.RecordUse(*lease, turn_).has_value(), "recording a fresh lease's use");
  Held held{.task = task,
            .extents = {},
            .operations = 0,
            .external = std::move(external),
            .ending = false,
            .waiters = {}};
  held.extents.reserve(closure.extents.size());
  for (const auto& [extent, generation] : closure.extents) {
    held.extents.push_back(extent);
  }
  // A closure is sorted and unique by construction; a hand-made one is
  // made so here, as the catalog's lease does.
  std::ranges::sort(held.extents);
  held.extents.erase(std::ranges::unique(held.extents).begin(), held.extents.end());
  held.waiters.reserve(settings_.waiters + 1);
  held_.emplace(*lease, std::move(held));
  ++stats_.leases_held;
  return *lease;
}

std::expected<Readiness, WorkError> Scheduler::EndLease(TaskId task, catalog::LeaseId lease) {
  TaskRecord* record = Record(task);
  const auto found = held_.find(lease);
  if (record == nullptr || found == held_.end() || found->second.task != task) {
    return std::unexpected(WorkError::kInvalid);
  }
  Held& held = found->second;
  held.ending = true;  // no new operation under it
  if (held.Drained()) {
    ReleaseHeld(found);
    return Readiness::kReady;
  }
  if (held.operations == 0 &&
      std::ranges::find(external_ending_, lease) == external_ending_.end()) {
    // The external count can reach zero after Drained above. Defer its next
    // check until ReleaseExternal, after the waiter below is registered:
    // releasing here would erase `held` before we finish using it.
    external_ending_.push_back(lease);
  }
  // Released once the last operation's fence is seen (or the external
  // work's end): the holder waits.
  if (!record->finished && std::ranges::find(held.waiters, task) == held.waiters.end()) {
    held.waiters.push_back(task);  // room for the holder beyond the waiter bound
    ++record->waiting;
  }
  return Readiness::kWaiting;
}

std::expected<Readiness, WorkError> Scheduler::AwaitRelease(
    TaskId task, std::span<const catalog::ExtentId> extents) {
  TaskRecord* record = Record(task);
  const std::optional<TaskView> view = tasks_.Describe(task);
  if (record == nullptr || record->finished || !view || view->cancelled) {
    return std::unexpected(WorkError::kClosed);
  }
  // Never hold and wait: a task that holds a request's lease, or whose
  // ancestor does (it waits for its children), would wait for a holder
  // that may be waiting, the same way, for it, a cycle no release breaks.
  // A lease that is ending is released on its operations' fences alone.
  for (std::optional<TaskId> at = task; at;) {
    if (std::ranges::any_of(held_, [&at](const auto& entry) {
          return entry.second.task == *at && !entry.second.ending;
        })) {
      return std::unexpected(WorkError::kInvalid);
    }
    const std::optional<TaskView> up = tasks_.Describe(*at);
    at = up ? up->parent : std::nullopt;
  }
  for (auto& [lease, held] : held_) {
    if (held.task == task) {
      continue;  // its own, ending: its operations' fences release it
    }
    const bool overlaps = std::ranges::any_of(extents, [&held](catalog::ExtentId extent) {
      return std::ranges::binary_search(held.extents, extent);
    });
    if (!overlaps) {
      continue;
    }
    if (std::ranges::find(held.waiters, task) == held.waiters.end()) {
      if (held.waiters.size() >= settings_.waiters) {
        return std::unexpected(WorkError::kBusy);
      }
      held.waiters.push_back(task);
      ++record->waiting;
    }
    return Readiness::kWaiting;
  }
  return Readiness::kReady;
}

void Scheduler::ConcludeHeld(catalog::LeaseId lease) {
  const auto found = held_.find(lease);
  base::Check(found != held_.end() && found->second.operations > 0,
              "an operation under a request's lease that is not counted");
  --found->second.operations;
  if (found->second.ending && found->second.operations == 0) {
    ReleaseWhenDrained(found);
  }
}

void Scheduler::ReleaseWhenDrained(std::map<catalog::LeaseId, Held>::iterator held) {
  if (held->second.Drained()) {
    ReleaseHeld(held);
  } else if (std::ranges::find(external_ending_, held->first) == external_ending_.end()) {
    external_ending_.push_back(held->first);
  }
}

bool Scheduler::ReleaseExternal() {
  bool progress = false;
  for (auto it = external_ending_.begin(); it != external_ending_.end();) {
    const auto held = held_.find(*it);
    if (held == held_.end()) {
      it = external_ending_.erase(it);
      continue;
    }
    if (!held->second.Drained()) {
      ++it;
      continue;
    }
    it = external_ending_.erase(it);
    ReleaseHeld(held);
    progress = true;
  }
  return progress;
}

void Scheduler::NoteFollow() {
  if (step_done_ == std::chrono::steady_clock::time_point{}) {
    return;  // no step ended since the last one was submitted
  }
  const std::int64_t gap = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - step_done_)
                               .count();
  step_done_ = {};
  if (gap > std::chrono::duration_cast<std::chrono::nanoseconds>(settings_.follow_limit).count()) {
    return;  // the client waited on something else (its user): not its usual pace
  }
  follow_gap_ = follow_gap_ == 0 ? gap : ((3 * follow_gap_) + gap) / 4;
}

void Scheduler::AwaitFollow() {
  const auto now = std::chrono::steady_clock::now();
  step_done_ = now;
  // At least the poll window, which may exceed the limit.
  const std::chrono::nanoseconds low = settings_.poll_window;
  const std::chrono::nanoseconds follow =
      std::clamp(std::chrono::nanoseconds((follow_gap_ * 3) / 2) + low, low,
                 std::max(low, std::chrono::nanoseconds(settings_.follow_limit)));
  follow_until_ = now + follow;
  if (lanes_.device != nullptr) {
    lanes_.device->Anticipate(follow_until_);
  }
}

void Scheduler::ReleaseHeld(std::map<catalog::LeaseId, Held>::iterator held) {
  // Its last use is its release: the extents stay resident, now eligible
  // for eviction (D-007), with the tick victim selection orders them by.
  base::Check(catalog_.RecordUse(held->first, turn_).has_value(), "recording a held lease's use");
  base::Check(catalog_.ReleaseLease(held->first).has_value(), "releasing a request's lease");
  const std::vector<TaskId> waiters = std::move(held->second.waiters);
  held_.erase(held);
  ++stats_.leases_released;
  for (const TaskId waiter : waiters) {
    Wake(waiter, false);
  }
}

bool TaskContext::TakeFailure() {
  Scheduler::TaskRecord* record = scheduler_.Record(task_);
  const bool failed = record != nullptr && record->failed;
  if (record != nullptr) {
    record->failed = false;
  }
  return failed;
}

bool TaskContext::ChildFailed() const {
  const std::optional<TaskView> view = scheduler_.tasks_.Describe(task_);
  return view && view->child_failed;
}

std::expected<TaskId, WorkError> TaskContext::Spawn(std::unique_ptr<TaskProgram> program,
                                                    std::size_t priority) {
  auto child = scheduler_.StartTask(0, program, priority, task_);
  if (!child) {
    switch (child.error()) {
      case StartError::kFull:
        return std::unexpected(WorkError::kBusy);
      case StartError::kInvalid:
        return std::unexpected(WorkError::kInvalid);
      case StartError::kClosed:
      case StartError::kStopped:
        return std::unexpected(WorkError::kClosed);
    }
  }
  return *child;
}

// Operations -------------------------------------------------------------------------

Scheduler::Operation& Scheduler::Open(Kind kind) {
  const OperationId id = board_.Open();
  base::Check(id.valid(), "opening an operation past the board's capacity");
  std::optional<Operation>& slot = operations_[id.index()];
  slot.emplace();
  slot->id = id;
  slot->kind = kind;
  return *slot;
}

void Scheduler::SetCritical(Operation& operation, bool critical) {
  if (operation.critical != critical) {
    operation.critical = critical;
    critical ? ++critical_ : --critical_;
  }
}

base::PushResult Scheduler::Push(Operation& operation) const {
  // Each moves the command only if its lane takes it. An absent lane
  // refuses it as closed: the operation is rolled back.
  switch (operation.route) {
    case Route::kStorage:
      return lanes_.storage != nullptr ? lanes_.storage->Submit(operation.read)
                                       : base::PushResult::kClosed;
    case Route::kBacking:
      return lanes_.backing != nullptr ? lanes_.backing->Submit(std::move(operation.device))
                                       : base::PushResult::kClosed;
    case Route::kDevice:
      return lanes_.device != nullptr ? lanes_.device->Submit(std::move(operation.device))
                                      : base::PushResult::kClosed;
    case Route::kCopy:
      return lanes_.copy != nullptr ? lanes_.copy->Submit(std::move(operation.device))
                                    : base::PushResult::kClosed;
    case Route::kCpu:
      return lanes_.cpu != nullptr ? lanes_.cpu->Submit(std::move(operation.job))
                                   : base::PushResult::kClosed;
  }
  return base::PushResult::kClosed;
}

void Scheduler::Publish(Operation& operation) {
  std::deque<OperationId>& pending = pending_[static_cast<std::size_t>(operation.route)];
  if (!pending.empty()) {
    pending.push_back(operation.id);  // behind what the lane refused before
    return;
  }
  switch (Push(operation)) {
    case base::PushResult::kAccepted:
      operation.published = true;
      return;
    case base::PushResult::kFull:
      pending.push_back(operation.id);  // backpressure: the next turns retry
      return;
    case base::PushResult::kClosed:
      RollBack(operation);
      return;
  }
}

bool Scheduler::Flush() {
  bool progress = false;
  for (std::deque<OperationId>& pending : pending_) {
    while (!pending.empty()) {
      Operation* operation = Find(pending.front());
      if (operation == nullptr || operation->published) {
        pending.pop_front();  // rolled back meanwhile
        continue;
      }
      const base::PushResult pushed = Push(*operation);
      if (pushed == base::PushResult::kFull) {
        break;
      }
      pending.pop_front();
      progress = true;
      if (pushed == base::PushResult::kAccepted) {
        operation->published = true;
      } else {
        RollBack(*operation);
      }
    }
  }
  progress = RetryBlocked() || progress;
  for (auto it = cancels_.begin(); it != cancels_.end();) {
    Operation* operation = Find(*it);
    if (operation != nullptr && !operation->quarantined) {
      const base::PushResult pushed =
          lanes_.storage->Submit(CancelRead{.operation = operation->id}, base::PushKind::kCleanup);
      if (pushed == base::PushResult::kFull) {
        ++it;  // the intent stays; the read drains meanwhile
        continue;
      }
    }
    it = cancels_.erase(it);
    progress = true;
  }
  return progress;
}

void Scheduler::Withdraw(TaskId task) {
  // Every load this leaves without waiters is marked cancelling before any
  // is cancelled: cancelling one can free a slot or a place in the window,
  // and the drains that hand those on skip a cancelling load, so no
  // withdrawn load starts a new stage (docs/async-model.md).
  std::vector<catalog::ExtentId> withdrawn;
  for (auto& [extent, load] : loads_) {
    const auto waiter = std::ranges::find(load.waiters, task);
    if (waiter == load.waiters.end()) {
      continue;
    }
    // Only this task's interest: other waiters and the load stay.
    load.waiters.erase(waiter);
    if (load.waiters.empty() && !load.cancelling) {
      load.cancelling = true;
      withdrawn.push_back(extent);
    }
  }
  std::vector<catalog::ExtentId> abandoned;
  for (auto& [extent, eviction] : evictions_) {
    std::erase(eviction.waiters, task);  // what a lane runs drains either way
    if (eviction.waiters.empty() && eviction.stage == EvictStage::kSlot) {
      // A write-back still waiting for a slot has touched nothing, and
      // would wait forever if every slot were quarantined: abandoned, it
      // leaves the extent resident and cannot hold up a stop.
      abandoned.push_back(extent);
    }
  }
  for (const catalog::ExtentId extent : abandoned) {
    std::erase(slot_waiters_, extent);
    EndEviction(extent, false);
  }
  // Requests' leases: it waits for none any more; its own end, released
  // now or once the operations under them drain (a quarantined one never
  // does, and keeps its lease).
  std::vector<catalog::LeaseId> ended;
  for (auto& [lease, held] : held_) {
    std::erase(held.waiters, task);
    if (held.task == task) {
      held.ending = true;
      if (held.operations == 0) {
        ended.push_back(lease);
      }
    }
  }
  for (const catalog::LeaseId lease : ended) {
    ReleaseWhenDrained(held_.find(lease));
  }
  // Backing this task kept for a handoff that no load took: released now,
  // never an idle pool (D-033).
  ReleaseParked(task);
  // After the walk: cancelling a page-in may end it, erasing its entry.
  for (const catalog::ExtentId extent : withdrawn) {
    const auto load = loads_.find(extent);
    if (load != loads_.end()) {
      CancelPageIn(extent, load->second);
    }
  }
}

void Scheduler::RollBack(Operation& operation) {
  // No lane took the command, so the scheduler itself proves it never
  // started, through the same mailbox a lane would have used.
  base::Check(board_.Accept(operation.id, Acceptance::kNotStarted) == Published::kRecorded,
              "rolling back an operation a lane may have seen");
  Conclude(operation, Outcome::kFailed, 0);
}

void Scheduler::Observe(const Observation& seen) {
  Operation* operation = Find(seen.operation);
  if (operation == nullptr || operation->quarantined) {
    return;  // a stale identity, or news of an operation already quarantined
  }
  if (seen.contradictory) {
    Quarantine(*operation, Fault::kContradiction);
    return;
  }
  if (seen.acceptance == Acceptance::kNotStarted) {
    Conclude(*operation, Outcome::kFailed, 0);  // it never touched memory
    return;
  }
  // Both halves first: a completion that came before its acceptance waits.
  if (seen.acceptance == Acceptance::kNone || !seen.terminal) {
    return;
  }
  if (!seen.terminal->no_further_access) {
    Quarantine(*operation, Fault::kUnproven);
    return;
  }
  Conclude(*operation, seen.terminal->outcome, seen.terminal->bytes);
}

void Scheduler::Conclude(Operation& operation, Outcome outcome, std::uint64_t bytes) {
  // The board must agree the operation is reconciled before anything it
  // holds is released: a contradiction that landed since the harvest keeps
  // the mailbox open, and the operation is quarantined instead.
  const OperationId id = operation.id;
  if (!board_.Close(id)) {
    Quarantine(operation, Fault::kContradiction);
    return;
  }
  SetCritical(operation, false);
  const Kind kind = operation.kind;
  const catalog::ExtentId extent = operation.extent;
  if (kind == Kind::kDevice || kind == Kind::kCpu) {
    const catalog::LeaseId held = operation.held;
    if (!held.valid()) {
      base::Check(catalog_.ReleaseLease(operation.lease).has_value(),
                  "releasing an operation's lease");
    }
    base::Check(tasks_.RetireOperation(operation.task).has_value(), "retiring a task's operation");
    const TaskId task = operation.task;
    operations_[id.index()].reset();
    if (held.valid()) {
      ConcludeHeld(held);  // the request's lease, if it is ending and this was its last
    }
    if (kind == Kind::kDevice) {
      AwaitFollow();  // a step (a request's, or one with its own lease): the next is likely soon
    }
    Wake(task, outcome != Outcome::kSucceeded);
    TryRetire(task);
    return;
  }
  operations_[id.index()].reset();
  if (kind == Kind::kLoad) {
    OnStage(extent, outcome, bytes);
  } else {
    OnEvicted(extent, outcome, bytes);
  }
}

void Scheduler::Fail(Fault fault) { fault_ = fault_.value_or(fault); }

void Scheduler::Quarantine(Operation& operation, Fault fault) {
  operation.quarantined = true;
  ++quarantined_;
  Fail(fault);
  SetCritical(operation, false);
  switch (operation.kind) {
    case Kind::kLoad:
      // Its extent stays charged as quarantined, with its slot; waiters
      // stop waiting.
      QuarantineLoad(operation.extent, fault);
      return;
    case Kind::kEvict:
      QuarantineEvicting(operation.extent, fault);
      return;
    case Kind::kDevice:
    case Kind::kCpu:
      // The lease and the task's hold stay: nothing it touched is reused.
      Wake(operation.task, true);
      return;
  }
}

// The turn loop ----------------------------------------------------------------------

bool Scheduler::Turn() {
  ++turn_;
  ++stats_.turns;
  if (shutdown_.load(std::memory_order_acquire)) {
    BeginStop();
  }
  bool progress = false;
  for (std::size_t i = 0; i < settings_.controls_per_turn; ++i) {
    std::optional<Control> control = controls_.TryPop();
    if (!control) {
      break;
    }
    if (const auto* cancel = std::get_if<CancelRequest>(&*control)) {
      // Before it is applied: a cancellation posted from now on queues
      // again instead of joining this one.
      const std::scoped_lock lock(cancel_mutex_);
      std::erase(cancel_intents_, cancel->request);
    }
    Apply(std::move(*control));
    progress = true;
  }
  for (const Observation& seen : board_.Harvest(settings_.observations_per_turn)) {
    Observe(seen);
    progress = true;
  }
  progress = Flush() || progress;
  if (!external_ending_.empty()) {
    progress = ReleaseExternal() || progress;
  }
  for (std::size_t i = 0; i < settings_.steps_per_turn; ++i) {
    const std::optional<TaskId> task = tasks_.NextReady();
    if (!task) {
      break;
    }
    StepTask(*task);
    progress = true;
  }
  return progress;
}

std::optional<std::expected<void, Fault>> Scheduler::Stopped() const {
  if (!stopping_ || board_.open() > quarantined_ || !cancels_.empty() || !controls_.drained() ||
      !loads_.empty() || !evictions_.empty()) {
    return std::nullopt;
  }
  // Every task finished at the stop; what remains is held by quarantined
  // work, whose capacity is not reclaimed (a request's lease with such an
  // operation under it included), or by external work never seen to end,
  // which is unproven alike.
  if (fault_) {
    return std::unexpected(*fault_);
  }
  if (!external_ending_.empty()) {
    return std::unexpected(Fault::kUnproven);
  }
  base::Check(tasks_.size() == 0, "a task outlived its operations after the stop");
  base::Check(held_.empty(), "a request's lease outlived its task after the stop");
  return std::expected<void, Fault>();
}

std::expected<void, Fault> Scheduler::Run() {
  auto progressed = std::chrono::steady_clock::now();
  while (true) {
    // Consumed before looking: a publication this turn misses signals again.
    (void)wake_.Consume();
    const bool progress = Turn();
    if (auto stopped = Stopped()) {
      return *stopped;
    }
    const auto now = std::chrono::steady_clock::now();
    if (progress) {
      progressed = now;
      continue;
    }
    // A critical operation in flight, or a request holding its lease, whose
    // next step is imminent: poll rather than sleep, within the window
    // (RE-017). So too while a lane anticipates a completion, and while the
    // next step is likely after one ended (the runtime wake).
    if (((critical_ > 0 || !held_.empty()) && now - progressed < settings_.poll_window) ||
        now < follow_until_ || wake_.Anticipating(now)) {
      ++stats_.polls;
      std::this_thread::yield();
      continue;
    }
    ++stats_.sleeps;
    (void)wake_.WaitFor(settings_.tick);
  }
}

}  // namespace jitllm::scheduler
