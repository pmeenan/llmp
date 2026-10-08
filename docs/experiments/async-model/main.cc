// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// M0 deterministic ownership experiment, not the M2 resource core.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>

namespace {
void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

template <typename Tag> struct Id {
  std::size_t slot{};
  std::uint64_t generation{};
  bool operator==(const Id&) const = default;
};
using TaskId = Id<struct TaskTag>;
using OpId = Id<struct OpTag>;
using BackingId = Id<struct BackingTag>;

std::uint64_t next_generation(std::uint64_t old) {
  check(old != std::numeric_limits<std::uint64_t>::max(), "generation exhausted");
  return old + 1;
}

enum class Kind { read, gpu, network };
enum class Acceptance { pending, accepted, not_started, unknown };
enum class Result { success, failed, cancelled, unknown };
enum class Phase { collecting, waiting, finished };

struct Backing {
  std::uint64_t generation{1};
  unsigned holds{};
  unsigned registrations{};
  bool resident{true};
  bool valid{true};
  int payload{7};
};

struct Task {
  std::uint64_t generation{};
  bool live{};
  bool ready{};
  bool cancel_requested{};
  bool failed{};
  bool client_finished{};
  unsigned pending{};
  unsigned resumes{};
  Phase phase{Phase::collecting};
  Result outcome{Result::unknown};
};

struct Terminal {
  Result result{};
  bool no_more_access{};
  bool content_valid{};
  bool operator==(const Terminal&) const = default;
};

struct Operation {
  std::uint64_t generation{};
  bool live{};
  TaskId task{};
  BackingId backing{};
  Kind kind{};
  bool registered{};
  bool registration_retired{};
  bool device_access_stopped{};
  bool unknown_reported{};
  Acceptance acceptance{Acceptance::pending};
  // Operation-owned mailbox: no event queue allocation or drop on saturation.
  std::optional<Acceptance> acceptance_mailbox;
  std::optional<Terminal> terminal_mailbox;
  std::optional<Terminal> observed_terminal;
  bool retirement_mailbox{};
  bool cancellation_ack{};
};

class Model {
 public:
  static constexpr std::size_t task_limit = 4;
  static constexpr std::size_t op_limit = 8;
  static constexpr std::size_t backing_limit = 4;

  std::optional<TaskId> create_task() {
    if (faulted_) return std::nullopt;
    for (std::size_t i = 0; i < tasks_.size(); ++i) {
      if (!tasks_[i].live) {
        const auto generation = next_generation(tasks_[i].generation);
        tasks_[i] = Task{};
        tasks_[i].generation = generation;
        tasks_[i].live = true;
        return TaskId{i, generation};
      }
    }
    return std::nullopt;
  }

  BackingId backing(std::size_t slot) const {
    check(slot < backing_limit, "backing index");
    return {slot, backings_[slot].generation};
  }

  const Backing& inspect(BackingId id) const {
    check(id.slot < backing_limit && backings_[id.slot].generation == id.generation,
          "inspect backing generation");
    return backings_[id.slot];
  }

  const Task& inspect(TaskId id) const {
    check(id.slot < task_limit && tasks_[id.slot].live &&
              tasks_[id.slot].generation == id.generation,
          "inspect task identity");
    return tasks_[id.slot];
  }

  bool live(OpId id) const {
    return id.slot < op_limit && ops_[id.slot].live &&
           ops_[id.slot].generation == id.generation;
  }

  // Prepare before handing a command to the fake provider. Each slot includes
  // acceptance, terminal and registration-cleanup capacity.
  std::optional<OpId> prepare(TaskId task, BackingId backing, Kind kind,
                              bool registered = false) {
    auto& t = task_ref(task);
    auto& b = backing_ref(backing);
    if (faulted_ || t.cancel_requested || t.failed ||
        t.phase != Phase::collecting || !b.resident ||
        (kind != Kind::read && !b.valid) ||
        (kind == Kind::read && b.holds != 0)) return std::nullopt;
    for (std::size_t i = 0; i < ops_.size(); ++i) {
      if (ops_[i].live) continue;
      const auto generation = next_generation(ops_[i].generation);
      auto& op = ops_[i];
      op = Operation{};
      op.generation = generation;
      op.live = true;
      op.task = task;
      op.backing = backing;
      op.kind = kind;
      op.registered = registered;
      ++b.holds;
      if (registered) ++b.registrations;
      if (kind == Kind::read) b.valid = false;
      ++t.pending;
      return OpId{i, generation};
    }
    return std::nullopt;
  }

  void seal(TaskId id) {
    auto& t = task_ref(id);
    check(t.phase == Phase::collecting, "seal once");
    t.phase = Phase::waiting;
    if (t.pending == 0) t.ready = true;
  }

  void cancel(TaskId id) {
    auto& t = task_ref(id);
    if (t.phase == Phase::finished) return;
    t.cancel_requested = true;
    t.client_finished = true;
    t.ready = true;
  }

  void acceptance(OpId id, Acceptance value) {
    if (!live(id)) return;
    check(value != Acceptance::pending, "provider must resolve acceptance");
    auto& op = ops_[id.slot];
    const auto previous = op.acceptance_mailbox.value_or(op.acceptance);
    check(previous == Acceptance::pending || previous == value,
          "contradictory acceptance faults provider");
    op.acceptance_mailbox = value;
  }

  // A cancellation CQE/ack is deliberately not a target completion.
  void cancel_ack(OpId id) {
    if (live(id)) ops_[id.slot].cancellation_ack = true;
  }

  // Fake physical provider access. The generation/hold check detects late DMA
  // after an erroneous reuse; scheduler generation filtering alone cannot fix it.
  void complete(OpId id, Result result = Result::success,
                bool no_more_access = true, bool content_valid = true) {
    check(live(id), "physical completion needs retained operation");
    auto& op = ops_[id.slot];
    auto& b = backing_ref(op.backing);
    check(b.holds > 0, "provider access protected by hold");
    check(!op.device_access_stopped, "physical provider completes once");
    if (op.kind == Kind::read) b.payload = 99;
    op.device_access_stopped = no_more_access;
    observe(id, {result, no_more_access, content_valid});
  }

  // Replayed/stale notifications contain values, never pointers to reused slots.
  void observe(OpId id, Terminal terminal) {
    if (!live(id)) return;
    auto& op = ops_[id.slot];
    const auto previous = op.terminal_mailbox ? op.terminal_mailbox : op.observed_terminal;
    check(!previous || *previous == terminal, "contradictory terminal faults provider");
    op.terminal_mailbox = terminal;
  }

  void registration_retired(OpId id) {
    check(live(id), "cleanup retains operation");
    auto& op = ops_[id.slot];
    check(op.registered, "retirement needs registration");
    const auto provider_acceptance = op.acceptance_mailbox.value_or(op.acceptance);
    check(op.device_access_stopped || provider_acceptance == Acceptance::not_started,
          "fake cleanup only after access stops");
    op.retirement_mailbox = true;
  }

  void poll() {
    for (auto& op : ops_) {
      if (!op.live) continue;
      if (op.acceptance_mailbox) {
        op.acceptance = *op.acceptance_mailbox;
        op.acceptance_mailbox.reset();
      }
      if (op.terminal_mailbox) {
        op.observed_terminal = op.terminal_mailbox;
        op.terminal_mailbox.reset();
      }
      if (op.retirement_mailbox) op.registration_retired = true;
      auto& t = task_ref(op.task);
      const bool unknown = op.acceptance == Acceptance::unknown ||
                           (op.observed_terminal && !op.observed_terminal->no_more_access);
      if (unknown) {
        faulted_ = true;
        t.failed = true;
        if (!op.unknown_reported) t.ready = true;
        op.unknown_reported = true;
        continue;  // Quarantined and charged, including registrations.
      }
      if (op.acceptance == Acceptance::pending) continue;
      const bool rejected = op.acceptance == Acceptance::not_started;
      if (!rejected && !op.observed_terminal) continue;
      check(!rejected || !op.observed_terminal, "rejected submission cannot complete");
      if (op.registered && !op.registration_retired) continue;
      auto& b = backing_ref(op.backing);
      const bool successful = !rejected && op.observed_terminal->result == Result::success &&
                              (op.kind != Kind::read || op.observed_terminal->content_valid);
      if (!successful) t.failed = true;
      if (op.kind == Kind::read) b.valid = successful && !t.cancel_requested;
      check(b.holds > 0 && t.pending > 0, "retirement accounting underflow");
      --b.holds;
      if (op.registered) {
        check(b.registrations > 0, "registration accounting underflow");
        --b.registrations;
      }
      --t.pending;
      op.live = false;
      if (t.pending == 0 && t.phase == Phase::waiting) t.ready = true;
    }
  }

  // One bounded task step, round-robin, with one readiness bit per task.
  std::optional<TaskId> step() {
    for (std::size_t count = 0; count < task_limit; ++count) {
      const auto slot = next_ready_;
      next_ready_ = (next_ready_ + 1) % task_limit;
      auto& t = tasks_[slot];
      if (!t.live || !t.ready) continue;
      t.ready = false;
      ++t.resumes;
      if (t.cancel_requested || t.failed) t.client_finished = true;
      if (t.pending == 0 && t.phase == Phase::waiting) {
        t.outcome = t.cancel_requested ? Result::cancelled :
                    t.failed ? Result::failed : Result::success;
        t.phase = Phase::finished;
        t.client_finished = true;
      }
      return TaskId{slot, t.generation};
    }
    return std::nullopt;
  }

  bool destroy_task(TaskId id) {
    auto& t = task_ref(id);
    if (t.pending != 0 || t.phase != Phase::finished) return false;
    t.live = false;
    return true;
  }

  bool reclaim(BackingId id) {
    auto& b = backing_ref(id);
    if (b.holds != 0 || b.registrations != 0) return false;
    b.resident = false;
    b.valid = false;
    return true;
  }

  std::optional<BackingId> reassign(BackingId id) {
    auto& b = backing_ref(id);
    if (b.holds != 0 || b.registrations != 0) return std::nullopt;
    const auto generation = next_generation(b.generation);
    b = Backing{};
    b.generation = generation;
    return BackingId{id.slot, generation};
  }

  bool faulted() const { return faulted_; }

 private:
  Task& task_ref(TaskId id) {
    (void)inspect(id);
    return tasks_[id.slot];
  }
  Backing& backing_ref(BackingId id) {
    (void)inspect(id);
    return backings_[id.slot];
  }
  std::array<Task, task_limit> tasks_{};
  std::array<Operation, op_limit> ops_{};
  std::array<Backing, backing_limit> backings_{};
  std::size_t next_ready_{};
  bool faulted_{};
};

template <typename T> T require(std::optional<T> value) {
  check(value.has_value(), "expected bounded allocation to succeed");
  return *value;
}

void cancelled_read_orders() {
  // Cancel, acceptance and physical completion can arrive in every order.
  std::array<int, 3> order{0, 1, 2};
  unsigned schedules = 0;
  do {
    Model m;
    auto task = require(m.create_task());
    auto backing = m.backing(0);
    auto op = require(m.prepare(task, backing, Kind::read, true));
    m.seal(task);
    for (auto event : order) {
      if (event == 0) { m.cancel(task); m.cancel_ack(op); }
      if (event == 1) m.acceptance(op, Acceptance::accepted);
      if (event == 2) m.complete(op);
      m.poll();
      (void)m.step();
      check(!m.reassign(backing), "read/registration prevents early backing reuse");
      check(!m.destroy_task(task), "cancelled task retained until operation drains");
    }
    m.registration_retired(op);
    m.poll();
    require(m.step());
    check(m.inspect(task).outcome == Result::cancelled, "cancelled task outcome");
    check(!m.inspect(backing).valid, "cancelled read does not publish content");
    check(m.inspect(backing).resident, "lease retirement is not eviction");
    check(m.destroy_task(task), "drained task can be destroyed");
    const auto new_backing = require(m.reassign(backing));
    const auto new_task = require(m.create_task());
    const auto new_op = require(m.prepare(new_task, new_backing, Kind::gpu));
    check(new_op.slot == op.slot && new_op.generation != op.generation,
          "operation slot reused with new generation");
    m.observe(op, {Result::success, true, true});
    m.acceptance(op, Acceptance::accepted);
    m.poll();
    check(m.inspect(new_backing).payload == 7 && m.inspect(new_backing).holds == 1,
          "stale notification cannot mutate replacement");
    m.seal(new_task);
    m.acceptance(new_op, Acceptance::accepted);
    m.complete(new_op);
    m.poll();
    require(m.step());
    check(m.inspect(new_task).outcome == Result::success, "replacement finishes");
    ++schedules;
  } while (std::next_permutation(order.begin(), order.end()));
  std::cout << "cancelled read: " << schedules << " event orders passed\n";
}

void joined_consumers() {
  // Three consumers, two independently retained registrations; enumerate every
  // legal terminal/cleanup ordering, with cancellation at each event boundary.
  std::array<int, 5> order{0, 1, 2, 3, 4};
  unsigned schedules = 0;
  do {
    auto position = [&](int value) { return std::find(order.begin(), order.end(), value); };
    if (position(3) < position(1) || position(4) < position(2)) continue;
    for (unsigned cancel_at = 0; cancel_at <= order.size() + 1; ++cancel_at) {
      Model m;
      const auto task = require(m.create_task());
      const auto backing = m.backing(0);
      const std::array ops{
          require(m.prepare(task, backing, Kind::gpu)),
          require(m.prepare(task, backing, Kind::network, true)),
          require(m.prepare(task, backing, Kind::network, true))};
      for (auto op : ops) m.acceptance(op, Acceptance::accepted);
      m.seal(task);
      m.poll();
      for (unsigned i = 0; i < order.size(); ++i) {
        if (i == cancel_at) m.cancel(task);
        const int event = order[i];
        if (event < 3) {
          m.complete(ops[static_cast<std::size_t>(event)]);
          m.observe(ops[static_cast<std::size_t>(event)], {Result::success, true, true});
        } else {
          m.registration_retired(ops[static_cast<std::size_t>(event - 2)]);
        }
        m.poll();
        (void)m.step();
        if (i + 1 != order.size())
          check(!m.reassign(backing), "all consumers and registrations must join");
      }
      if (cancel_at == order.size()) m.cancel(task); // Late cancel is idempotent.
      check(m.inspect(task).phase == Phase::finished, "joined phase finishes");
      check(m.inspect(task).outcome ==
                (cancel_at < order.size() ? Result::cancelled : Result::success),
            "completion/cancellation linearization");
      check(m.inspect(backing).holds == 0 && m.inspect(backing).registrations == 0,
            "every backing reference retired exactly once");
      check(m.inspect(backing).resident, "retirement keeps useful contents resident");
      check(m.reclaim(backing), "explicit reclaim succeeds after join");
      ++schedules;
    }
  } while (std::next_permutation(order.begin(), order.end()));
  std::cout << "joined consumers: " << schedules << " schedules passed\n";
}

void saturation_and_fairness() {
  Model m;
  std::array<TaskId, Model::task_limit> tasks;
  std::array<OpId, Model::op_limit> ops;
  for (auto& task : tasks) task = require(m.create_task());
  check(!m.create_task(), "task bound enforced");
  for (std::size_t i = 0; i < ops.size(); ++i)
    ops[i] = require(m.prepare(tasks[i % tasks.size()], m.backing(i % tasks.size()),
                               Kind::network, true));
  check(!m.prepare(tasks[0], m.backing(0), Kind::gpu), "operation bound enforced");
  for (auto task : tasks) m.seal(task);
  for (auto op : ops) m.acceptance(op, Acceptance::accepted);
  m.poll();
  // Full slots still have cancellation and terminal mailboxes, plus cleanup.
  for (unsigned repeat = 0; repeat < 100; ++repeat)
    for (auto task : tasks) m.cancel(task);
  for (auto op : ops) { m.cancel_ack(op); m.complete(op); m.registration_retired(op); }
  m.poll();
  for (auto task : tasks) {
    check(require(m.step()) == task, "round-robin progress while all tasks ready");
    check(m.inspect(task).resumes == 1, "duplicate wakeups coalesce");
    check(m.inspect(task).phase == Phase::finished, "saturated task drains");
    check(m.destroy_task(task), "saturated task released");
  }
  check(!m.step(), "no duplicate ready entries");
  std::cout << "saturation: task/operation bounds, cleanup and fairness passed\n";
}

void failures_and_independent_progress() {
  // Provider cleanup after a rejection need not wait for scheduler observation.
  // Exercise both a destination registration and a read-only consumer's one.
  for (bool poll_before_cleanup : {false, true}) {
    for (Kind kind : {Kind::read, Kind::network}) {
      Model rejected_model;
      auto task = require(rejected_model.create_task());
      auto backing = rejected_model.backing(0);
      auto op = require(rejected_model.prepare(task, backing, kind, true));
      rejected_model.seal(task);
      rejected_model.acceptance(op, Acceptance::not_started);
      if (poll_before_cleanup) rejected_model.poll();
      check(!rejected_model.reassign(backing), "rejection still needs registration cleanup");
      rejected_model.registration_retired(op);
      rejected_model.poll();
      require(rejected_model.step());
      check(rejected_model.inspect(task).outcome == Result::failed,
            "registered rejection fails task without provider access");
      check(rejected_model.inspect(backing).payload == 7,
            "rejected provider never touched backing");
      check(rejected_model.inspect(backing).holds == 0 &&
                rejected_model.inspect(backing).registrations == 0,
            "rejected registration and hold retire exactly once");
      check(rejected_model.destroy_task(task) && rejected_model.reclaim(backing),
            "registered rejection permits safe task and backing cleanup");
    }
  }
  // Partial submission: rejecting one command cannot release another's lease.
  Model m;
  auto t = require(m.create_task());
  auto b = m.backing(0);
  auto accepted = require(m.prepare(t, b, Kind::gpu));
  auto rejected = require(m.prepare(t, b, Kind::network));
  m.seal(t);
  m.acceptance(accepted, Acceptance::accepted);
  m.acceptance(rejected, Acceptance::not_started);
  m.poll();
  check(!m.reassign(b), "partial submission keeps accepted ownership");
  check(!m.step(), "partial submission waits for outstanding operation");

  auto other = require(m.create_task());
  auto other_op = require(m.prepare(other, m.backing(1), Kind::gpu));
  m.seal(other);
  m.complete(other_op); // Immediate completion before acceptance observation.
  m.poll();
  check(!m.step(), "early completion waits for submission reconciliation");
  m.acceptance(other_op, Acceptance::accepted);
  m.poll();
  check(require(m.step()) == other, "unrelated ready phase runs past suspended phase");
  m.complete(accepted, Result::failed);
  m.poll();
  check(require(m.step()) == t && m.inspect(t).outcome == Result::failed,
        "known terminal error permits safe failure and drain");
  check(m.reclaim(b), "known terminal failure permits reclaim");

  for (bool corrupt : {false, true}) {
    Model read_model;
    auto task = require(read_model.create_task());
    auto backing = read_model.backing(0);
    auto op = require(read_model.prepare(task, backing, Kind::read));
    check(!read_model.prepare(task, backing, Kind::gpu), "incomplete read blocks consumer");
    read_model.seal(task);
    read_model.acceptance(op, Acceptance::accepted);
    read_model.complete(op, corrupt ? Result::success : Result::failed, true, !corrupt);
    read_model.poll();
    require(read_model.step());
    check(!read_model.inspect(backing).valid, "short/error or corrupt read never publishes");
    check(read_model.inspect(task).outcome == Result::failed, "read failure propagates");
  }

  for (bool unknown_acceptance : {false, true}) {
    Model unknown;
    auto task = require(unknown.create_task());
    auto backing = unknown.backing(0);
    auto op = require(unknown.prepare(task, backing, Kind::gpu));
    unknown.seal(task);
    unknown.acceptance(op, unknown_acceptance ? Acceptance::unknown : Acceptance::accepted);
    if (!unknown_acceptance) unknown.complete(op, Result::unknown, false);
    unknown.cancel(task); // Deadline/client termination cannot supply a fence.
    unknown.poll();
    require(unknown.step());
    check(unknown.faulted() && !unknown.create_task(), "unknown completion stops admission");
    check(unknown.inspect(task).client_finished && !unknown.destroy_task(task),
          "client completion and task retirement independent");
    check(!unknown.reassign(backing) && unknown.inspect(backing).holds == 1,
          "unknown completion quarantines charged backing");
  }
  std::cout << "failures: partial/early submission, read validation, unknown quarantine passed\n";
}

void successful_read_then_compute() {
  Model m;
  auto b = m.backing(0);
  auto read_task = require(m.create_task());
  auto read = require(m.prepare(read_task, b, Kind::read));
  m.seal(read_task);
  m.acceptance(read, Acceptance::accepted);
  m.complete(read);
  m.poll();
  require(m.step());
  check(m.inspect(b).valid && m.inspect(b).payload == 99, "verified read publishes content");
  check(m.destroy_task(read_task), "read phase drained");
  auto compute_task = require(m.create_task());
  auto compute = require(m.prepare(compute_task, b, Kind::gpu));
  m.seal(compute_task);
  m.acceptance(compute, Acceptance::accepted);
  m.complete(compute);
  m.poll();
  require(m.step());
  check(m.inspect(compute_task).outcome == Result::success, "dependent phase completes");
  std::cout << "phases: verified load then consumer passed\n";
}
}  // namespace

int main() {
  cancelled_read_orders();
  joined_consumers();
  saturation_and_fairness();
  failures_and_independent_progress();
  successful_read_then_compute();
  std::cout << "PASS: deterministic CPU-only async ownership experiment\n";
}
