// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler thread (D-048, docs/async-model.md): the node's single
// writer of task states, the catalog's loads and leases, operation records
// and retirement decisions. Other threads send it controls (start or cancel
// a request, shut down) and the lanes (services.h) publish what they
// observe on the completion board; neither changes its records.
//
// A turn drains a bounded batch of controls and of board observations,
// retries publications a full lane refused, and steps a bounded batch of
// ready tasks. Nothing in a turn blocks: a full lane leaves the command
// with its operation until a later turn, and a task waits on identified
// dependencies, never on a thread. Between turns the thread polls while a
// critical-path operation is in flight, for at most a window after its
// last progress (RE-017: a sleeping thread wakes slowly on the Spark),
// while a lane anticipates a completion, and after a step (device work)
// for about as long as its client takes to hand over the next (the runtime
// wake, docs/experiments/runtime-wake/), and otherwise sleeps on the wake
// flag. It consumes the flag before it looks at anything, so a publication
// it did not see re-signals it: no wakeup is lost (base/wake.h).
//
// A task is an explicit state machine (TaskProgram): each step does bounded
// work through its TaskContext and yields, waits, or finishes. Its work:
//
//   - Materialize a closure: resident extents with the recorded contents
//     are ready; a nonresident one starts a page-in from its registered
//     source; a load in flight is joined, and so is an eviction in flight
//     (the task then materializes again). Page-ins belong to the
//     scheduler, not the task: one per extent, with a bounded waiter list.
//     A task that leaves (cancelled or finished) removes only its own
//     interest; when the last one leaves, the load starts no new stage,
//     asks the storage lane to cancel a read in flight, and still drains.
//     A page-in publishes resident contents only if its read transferred
//     the whole range and, when landed, its copy's fence has completed;
//     otherwise the load fails, and the backing is released only once
//     every stage's completion proves no further access.
//   - A page-in (docs/architecture.md#page-in-and-eviction-lifecycles-8)
//     runs in stages, each one operation with its own mailbox:
//       1. with managed backing (D-033), the VMM lane (Lanes::backing, or
//          the device lane without one) creates backing in the source's
//          allocation class, maps it at its place and sets access; or, when
//          the load was handed backing an eviction kept (below), maps that;
//       2. a landed source (D-081) waits for a landing slot, in order;
//       3. the storage lane reads the file range with direct I/O, into the
//          slot, or for a direct source into the extent's own memory;
//       4. for a landed source, the copy lane (Lanes::copy, or the device
//          lane without one) copies the slot into the extent's device
//          memory on the zone's stream, in up to kMaxDeviceCopies pieces,
//          and fences it;
//       5. the extent is published resident, and the slot freed, only once
//          that fence has completed.
//     The zone's copies have a lane of their own so that a job's launches
//     never hold them up: a launch into a stream with about 1,020
//     operations pending blocks the launching thread (RE-029), and a
//     phase can have several thousand.
//     A slot is reused only once the read into it and the copy out of it
//     have both been proven to touch it no more; one whose copy is
//     unproven is quarantined with the extent, never reused. At most twice
//     as many landed loads as the zone has slots are in flight at once
//     (the rest queue in order, unmapped), so backing is mapped at most
//     one zone ahead of the reads. A failed or withdrawn load unmaps and
//     releases the backing it mapped before the extent is nonresident
//     again. A stage that can never get a mailbox (every one retired or
//     held by quarantined work) quarantines its load instead of waiting.
//   - Evict an extent: with managed backing the VMM lane unmaps and
//     releases it (D-033) while the extent is EVICTING, and the task may
//     wait for it; otherwise the catalog alone records it. An unmap
//     refused with nothing changed leaves it resident; one of unknown
//     outcome, or refused because an earlier unknown outcome left its
//     place undetermined, quarantines it.
//   - Hand backing off (D-033): an eviction asked for with a handoff
//     unmaps the backing but keeps it, and parks: the extent stays
//     EVICTING, charged, so the catalog counts the kept backing, and the
//     evictor is told it is done. A page-in whose managed backing has the
//     same allocation class and size takes the backing of a parked
//     eviction in its own domain (B is each domain's)
//     instead of creating its own: in one step the load begins, allowed
//     the parked extent's bytes over B, and the parked eviction completes,
//     so the charge moves from one extent to the other and occupancy never
//     exceeds B. A page-in of a parked extent itself takes its own. The
//     VMM lane then maps the kept backing at the new place and sets
//     access, with no create and no release. Parked backing never waits
//     for a load that may not come: when the task that asked for the
//     eviction finishes, the VMM lane releases what is still parked and
//     the evictions complete. A load that took kept backing and ends
//     without mapping it (cancelled, or refused) releases it before the
//     extent is nonresident again.
//     With lazy_handoff (SchedulerSettings; the node's default) the park
//     keeps the backing mapped where it was: no new lease can reach the
//     EVICTING extent, and every consumer retired before its eviction
//     began (invariant 2). The VMM lane unmaps it there as the first step
//     of whatever takes it, a load's map (beside the reads of the loads
//     ahead of it) or the release of what no load took. The lane takes
//     the newest kept backing of the class and size, not the one whose
//     parked eviction the scheduler ended: an extent nonresident again may
//     still have its old backing mapped at its place (no lease reaches it
//     either), and the count of kept backing per class and size is what
//     the catalog charges. A load at a place whose own kept backing is
//     still there takes it as it is, mapped and accessible (a reuse), or
//     moves it out first, still kept (a created backing); an eviction
//     into a reservation an unknown outcome left undetermined is
//     quarantined at once, as an unmap there would be. A kept backing the
//     lane cannot unmap from its old place never fails the load taking it:
//     one refused stays kept behind the others and the next is tried; one
//     with an unknown outcome is dropped, still charged. A release that
//     finds none it can take is refused, so its eviction stays charged.
//   - Write back live mutable contents on eviction (a kPreserve extent
//     whose source is its write-back place, PageSource::write_back), the
//     page-in's reverse path through the zone (D-081): the extent is
//     EVICTING (no new lease) and unheld; it waits in order for a slot;
//     the device lane copies it into the slot on the zone's stream, and
//     fences it; the storage lane writes the slot to the place with direct
//     I/O; only once that write has moved the whole range is the slot
//     freed and the backing unmapped and released (as above). The catalog
//     then keeps the content generation and marks the contents preserved,
//     and a later load restores them from the same place. A copy or write
//     that fails with known completion, or a stage that can never get a
//     mailbox, abandons the eviction: the backing was never touched, so
//     the extent is resident again with its contents. One of unknown
//     outcome quarantines the extent and its slot. One still waiting for
//     a slot when its last waiter leaves is abandoned; once it has a slot
//     it outlives its evictor and drains. A load of a write-back
//     place whose contents are not preserved is refused: nothing claims
//     contents that were never written (invariant 4).
//   - Submit device work or a CPU job over a closure: the operation leases
//     the closure (all or none, at the recorded contents), takes a lifetime
//     hold on the task and a mailbox, and is recorded before its command is
//     published. Its lease and hold are released only once its terminal
//     result proves no further access.
//   - Hold a request's lease (D-007's residency lease, taken once for a
//     whole request rather than per step): a task that has materialized a
//     closure leases it (HoldLease) and submits each step's device work
//     under that lease, with no closure to walk and no lease to take or
//     release per step. Each such operation still takes a mailbox and the
//     task's lifetime hold, and still ends only on its fence: the lease
//     counts the operations under it, and ending it (EndLease, or the task
//     finishing, cancelled or not) releases it only once the last one's
//     terminal result proves no further access. A quarantined one keeps
//     it for good. Releasing it makes its extents eligible for eviction;
//     it evicts nothing (D-007). While it is held no eviction of its
//     extents can begin (the catalog refuses a held extent), and a task
//     that wants one (a swap) waits for its release (AwaitRelease)
//     instead of retrying; a task that holds one (or whose ancestor
//     does) is refused that wait, so no cycle of holders waiting for
//     each other can form. The wait is unbounded: nothing but its holder
//     ends a lease. The per-step path, which leases the closure
//     for each operation, stays for work whose closure changes from step
//     to step (routed experts, M7).
//   - Wait for the request's client (AwaitSignal): a task with nothing to
//     do until the client sends more (a request's next step) waits for a
//     SignalRequest control, which wakes it; a signal that comes first is
//     kept for the next wait.
//   - Spawn children, whose finishing wakes the parent.
//   - Pin places (D-090): a captured graph names device addresses, so while
//     it may replay, the places of the extents it reads must not move.
//     A pinned extent's source cannot be replaced by one that puts its
//     contents anywhere else (SamePlace): every load maps its backing at the
//     same place of the same reservation and copies its contents to the same
//     addresses, whatever backing it takes (created, or handed off).
//
// Observations reconcile acceptance and the terminal result in either
// order. Not started (from the lane, or the scheduler's own rollback of a
// command no lane ever took) releases everything at once. A terminal result
// without the proof, or contradictory observations, quarantines the
// operation: a page-in's extent becomes QUARANTINED, device or CPU work
// keeps its lease and its task's hold, everything stays charged, and the
// node faults, which stops admitting requests. Quarantine is sticky here:
// recovery needs a validated quiescence procedure (docs/async-model.md).
//
// Cancelling a request cancels its task tree: no new children or
// operations, commands no lane has taken are rolled back, page-in interests
// are withdrawn, and every task finishes cancelled. Work already published
// stays owned until it drains; a task retires only when its operations and
// children have. Shutdown stops admission, cancels every task, drains
// every accepted operation and then returns; the program then closes the
// lanes and releases backing. Work that cannot be reconciled (quarantined)
// faults the shutdown instead of reporting its capacity reclaimed.
//
// Not yet here: victim selection on a miss, admission's envelopes
// and the switching policy (admission.h) driving task starts, which
// reports a boundary only where the request's ProgramCursor::AtBoundary
// holds (execution/program.h).

#ifndef JITLLM_SCHEDULER_SCHEDULER_H_
#define JITLLM_SCHEDULER_SCHEDULER_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "providers/direct_reader.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"
#include "scheduler/services.h"
#include "scheduler/tasks.h"

namespace jitllm::scheduler {

// How a step ends.
struct Step {
  enum class Kind : std::uint8_t {
    kYield,   // ready again at once
    kWait,    // ready once every dependency it registered has settled
    kFinish,  // stop computing, with an outcome
  };
  Kind kind = Kind::kYield;
  TaskOutcome outcome = TaskOutcome::kSucceeded;

  static constexpr Step Yield() {
    return {.kind = Kind::kYield, .outcome = TaskOutcome::kSucceeded};
  }
  static constexpr Step Wait() { return {.kind = Kind::kWait, .outcome = TaskOutcome::kSucceeded}; }
  static constexpr Step Finish(TaskOutcome outcome) {
    return {.kind = Kind::kFinish, .outcome = outcome};
  }
};

enum class WorkError : std::uint8_t {
  kClosed,       // the task is cancelled or finished: no new work
  kBusy,         // no mailbox, waiter room or task slot now, or an eviction in progress
  kNotResident,  // a lease needs every extent resident
  kStale,        // the contents changed since the closure was taken
  kOverBudget,   // loading would exceed the execution budget B
  kUnavailable,  // quarantined, no registered source, unknown, or no identity will ever come
  kInvalid,      // malformed work, or no lane for it
};

std::string ToString(WorkError error);

enum class Readiness : std::uint8_t {
  kReady,    // every extent is resident with the recorded contents
  kWaiting,  // page-ins are registered: return Step::Wait()
};

enum class Fault : std::uint8_t {
  kContradiction,  // a lane's observations contradict each other
  kUnproven,       // a terminal result without proof of no further access
  kBacking,        // backing could not be unmapped or released as the catalog needs
  kExhausted,      // a page-in stage can never get an identity: all retired or quarantined
};

std::string ToString(Fault fault);

// A scheduler lifetime and its source/pin mutations, read on its thread.
// Exhausted identities never authorize a cached placement check.
struct PlacementStamp {
  std::uint64_t instance = 0;
  std::uint64_t epoch = 0;
  bool cacheable() const { return instance != 0 && instance != UINT64_MAX && epoch != UINT64_MAX; }
  bool operator==(const PlacementStamp&) const = default;
};

class TaskContext;

// A task's continuation, stepped only on the scheduler thread. It owns its
// state; the scheduler owns it, from its start until the task retires.
class TaskProgram {
 public:
  TaskProgram() = default;
  TaskProgram(const TaskProgram&) = delete;
  TaskProgram& operator=(const TaskProgram&) = delete;
  TaskProgram(TaskProgram&&) = delete;
  TaskProgram& operator=(TaskProgram&&) = delete;
  virtual ~TaskProgram() = default;

  // One bounded step.
  virtual Step Advance(TaskContext& context) = 0;
  // The task's outcome, once it has one. A refused start is cancelled
  // (shutting down) or failed.
  virtual void Finished(TaskOutcome /*outcome*/) {}
  // Every operation and child it had has retired; it is destroyed next.
  virtual void Retired() {}
};

// Managed backing (D-033): where the scheduler maps an extent's backing
// when it loads it, and unmaps and releases it when it evicts it.
struct BackingPlace {
  providers::ReservationId reservation;
  Bytes offset;
  Bytes size;  // the extent's backing, a multiple of the class's granularity
  std::size_t allocation_class = 0;
  bool operator==(const BackingPlace&) const = default;
};

// A piece of a landed read: `length` bytes at `slot_offset` in the slot,
// copied to `destination`.
struct LandedPiece {
  std::uint64_t slot_offset = 0;
  std::uint64_t destination = 0;
  Bytes length;
  bool operator==(const LandedPiece&) const = default;
};

// Where a nonresident extent's contents come from, and how they arrive.
struct PageSource {
  // The file range. A direct read lands at `read.memory`, which the CPU
  // and the storage provider reach (host backing, D-034); a landed read
  // ignores it.
  providers::ReadSpec read;
  // Landed (D-081): the read lands in a landing slot and the copy lane
  // copies it to `destination`, the extent's device address; or, with
  // pieces, each piece to its own destination (an extent whose contents
  // are not one range of the file read, such as a page of a resident
  // expert slab whose groups sit at a stride that is not the file's). The
  // caller guarantees every destination byte lies in the extent's own
  // memory. Write-back takes only the one-range form.
  bool landed = false;
  std::uint64_t destination = 0;
  std::array<LandedPiece, kMaxDeviceCopies> pieces{};
  std::size_t piece_count = 0;
  // Managed backing; without it the backing stays mapped at the
  // destination for as long as the source is registered.
  std::optional<BackingPlace> backing;
  // The file range is the extent's write-back place (state spill, D-081's
  // reverse path): only for a kPreserve extent. Evicting it writes its
  // contents here first (read.memory, or the zone for a landed source),
  // and loading it restores them, only while the catalog marks them
  // preserved. The caller owns the file: opened for direct I/O beneath the
  // spill role, never shared by two places.
  bool write_back = false;
  // Unless written back, the contents are zero bytes (a state's sparse
  // spill file before anything was saved, D-081): a load maps the backing
  // and zeroes `read.length` bytes at `destination` on the zone's copy
  // stream, with no read and no slot. Only landed, one range, managed.
  bool zero = false;
};

// Whether two sources put an extent's contents at the same addresses: the
// same managed backing place (reservation, offset, size and class), and,
// landed, the same destination or pieces' destinations and lengths, or,
// direct, the same memory. The file range may differ.
bool SamePlace(const PageSource& a, const PageSource& b);

// The landing zone (D-081): a persistent pool of `slots` host-VMM
// addresses of `slot_bytes` each, mapped read-write for the CPU and the
// device. The program maps it and registers it in the catalog before the
// scheduler starts, and keeps it until the lanes have drained. Landed
// copies run on the copy lane's stream `stream`, or without a copy lane on
// the device lane's. Slots need not be 2 MiB: a read may be longer than
// the extent it lands (PageSource's pieces), up to `slot_bytes`.
struct LandingZone {
  std::vector<std::uint64_t> slots = {};  // NOLINT(readability-redundant-member-init)
  Bytes slot_bytes = {};                  // NOLINT(readability-redundant-member-init)
  std::uint32_t stream = 0;
};

// What a page-in reached, reported to a PageInObserver.
enum class PageInEvent : std::uint8_t {
  kMapped,    // its managed backing is mapped
  kReading,   // its read was issued (published, or held while the lane is full)
  kResident,  // published resident
  kFailed,    // ended without contents (a quarantined load is not reported)
};

// Watches page-ins for measurement: called on the scheduler thread, inside
// a turn, so it must be quick and must not call the scheduler.
class PageInObserver {
 public:
  PageInObserver() = default;
  PageInObserver(const PageInObserver&) = delete;
  PageInObserver& operator=(const PageInObserver&) = delete;
  PageInObserver(PageInObserver&&) = delete;
  PageInObserver& operator=(PageInObserver&&) = delete;
  virtual ~PageInObserver() = default;
  virtual void Staged(catalog::ExtentId extent, PageInEvent event) = 0;
};

struct SchedulerSettings {
  std::size_t tasks = 64;  // the admitted task bound
  std::size_t priorities = 2;
  std::uint32_t aging_limit = 4;
  std::size_t controls = 64;           // queued controls, including the reserve ...
  std::size_t controls_reserved = 16;  // ... that only cancellations may use
  std::size_t controls_per_turn = 16;
  std::size_t observations_per_turn = 64;
  std::size_t steps_per_turn = 16;
  std::size_t waiters = 8;  // tasks waiting on one page-in
  // The execution budget B every load checks. Initialized so callers may
  // designate only the fields they change.
  base::Bytes budget = {};  // NOLINT(readability-redundant-member-init)
  // Handoff evictions park their backing still mapped (BackingWork::lazy):
  // the unmap moves into the load that takes it, beside earlier loads' reads.
  bool lazy_handoff = false;
  // The bound on both durations below, checked when the scheduler is
  // built: the clock compares and waits in nanoseconds, where an unbounded
  // value overflows.
  static constexpr std::chrono::hours kLongest{1};
  // Polling after the last progress while a critical-path operation is in
  // flight, or a request holds its lease (its next step is imminent): the
  // window the task-lanes measurement used, not a tuned value (RE-017).
  // Zero never polls. The scheduler's thread spins (yielding) for up to
  // this long after each progress, so a window longer than a decode step
  // keeps a core busy for as long as a request steps; the paged harness's
  // old 100 ms did, this default does not. The runtime wake adds to it
  // (docs/experiments/runtime-wake/): the thread also polls while a lane
  // anticipates a completion (CompletionBoard::Anticipate), and after a
  // step for about as long as its client has lately taken to hand over the
  // next (`follow_limit`).
  std::chrono::microseconds poll_window{200};
  // The longest the scheduler polls after a step (device work submitted
  // through a TaskContext, under a request's lease or its own) for the
  // next, and has the device lane poll too (DeviceService::Anticipate): 1.5
  // times the recent gap between a step's end and the next step's
  // publication, plus `poll_window`, at most this. A gap longer than this
  // (a client that waits on its user) is not counted. At most kLongest.
  std::chrono::microseconds follow_limit{10000};
  // The longest sleep: a timer tick, never needed for correctness. Positive.
  std::chrono::milliseconds tick{100};
  // Where landed page-ins land; none by default (a landed source is then
  // refused).
  LandingZone landing = {};  // NOLINT(readability-redundant-member-init)
  // Told of each page-in's progress; outlives the scheduler. Optional.
  PageInObserver* observer = nullptr;
};

// The lanes the scheduler publishes to; any may be absent (its work is
// then refused as invalid). Managed backing's VMM work goes to `backing`
// if there is one, and otherwise to the device lane; the landing zone's
// copies, in and out, go to `copy` (a DeviceService of its own, with the
// zone's stream) if there is one, and otherwise to the device lane.
struct Lanes {
  StorageService* storage = nullptr;
  DeviceService* device = nullptr;
  Lane<CpuCommand>* cpu = nullptr;
  BackingService* backing = nullptr;
  DeviceService* copy = nullptr;
};

// How an eviction goes (TaskContext::Evict).
struct EvictOptions {
  // Keep the managed backing for a load to take (the header's handoff):
  // the eviction parks once unmapped, and the backing is released only if
  // no load has taken it by the time the evicting task finishes.
  bool handoff = false;
  // Its owner's word that nothing wrote a kPreserve extent since it was
  // last restored from its write-back place (or written back there): if
  // the catalog agrees that the place saved its current content generation
  // (ExtentView::saved_generation), the write-back writes nothing, and the
  // eviction completes preserved as if it had (incremental spill).
  bool unchanged = false;
};

// Controls from other threads.
struct StartRequest {
  std::uint64_t request = 0;  // the caller's tag, for cancelling it later
  std::size_t priority = 1;
  std::unique_ptr<TaskProgram> program;
};
struct CancelRequest {
  std::uint64_t request = 0;
};
// The request's client has more for it (its next step): wakes the
// request's root task if it waits for a signal (TaskContext::AwaitSignal),
// or else is kept for its next wait. Signals do not count: one kept signal
// stands for any number posted before the task took it. What the client
// wrote for the task before posting this is visible to the task once it is
// woken (the control queue orders them).
struct SignalRequest {
  std::uint64_t request = 0;
};
using Control = std::variant<StartRequest, CancelRequest, SignalRequest>;

enum class StartError : std::uint8_t {
  kStopped,  // shutting down, faulted, or identities exhausted: admission has stopped
  kFull,     // the task table is full
  kClosed,   // the parent takes no new children
  kInvalid,  // no program, or a priority out of range
};

struct SchedulerStats {
  std::uint64_t catalog_occupancy_bytes = 0;  // the paged node's domain, at Stats
  BackingCreateStats backing_creates;         // populated by the paged node from its backing lane
  std::uint64_t turns = 0;
  std::uint64_t polls = 0;   // idle turns spent polling for a critical completion
  std::uint64_t sleeps = 0;  // waits on the wake flag
  // The handoff (D-033): evictions parked with their backing kept, loads
  // that took parked backing, and parked backing released unused.
  std::uint64_t parked = 0;
  std::uint64_t handed_off = 0;
  std::uint64_t released_unused = 0;
  // Requests' leases (HoldLease): taken, released, and the operations
  // submitted under one.
  std::uint64_t leases_held = 0;
  std::uint64_t leases_released = 0;
  std::uint64_t held_operations = 0;
  // Write-backs that wrote nothing (EvictOptions::unchanged), and their bytes.
  std::uint64_t unchanged_writebacks = 0;
  std::uint64_t unchanged_writeback_bytes = 0;
};

class Scheduler {
 public:
  // The board's mailboxes bound the operations in flight. The catalog,
  // board, wake flag and lanes outlive the scheduler; the board signals
  // `wake`, and so do controls.
  Scheduler(catalog::Catalog& catalog, CompletionBoard& board, base::WakeFlag& wake, Lanes lanes,
            const SchedulerSettings& settings);
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) = delete;
  Scheduler& operator=(Scheduler&&) = delete;
  ~Scheduler() = default;

  // Any thread. A full queue refuses the control and leaves it with the
  // caller, and so does a closed one (once shutting down). A cancellation
  // may use the reserve, and is an intent per request: while one for the
  // request is queued with no start of it posted since, another costs
  // nothing and is accepted as already queued.
  base::PushResult Post(Control&& control);
  // Any thread: a flag, however often it is set.
  void RequestShutdown();

  // The scheduler thread (or a deterministic test standing in for it).
  // Start applies at once, ahead of any control still queued: a
  // cancellation of the same request posted earlier and not yet applied
  // then reaches this start too. Post a StartRequest to order a start
  // after what was posted before it.
  std::expected<TaskId, StartError> Start(std::uint64_t request,
                                          std::unique_ptr<TaskProgram> program,
                                          std::size_t priority = 1);
  // Cancels the request's task tree, or trees if its tag was started again
  // while an earlier start drained; false if no such request is live.
  bool Cancel(std::uint64_t request);
  // The scheduler thread, or before it runs (a task's step may call these).
  // Where a nonresident extent's contents are read from, directly into its
  // backing, which stays mapped: the PageSource below with neither a zone
  // nor managed backing, whose rules it must pass (fatal otherwise).
  void SetSource(catalog::ExtentId extent, const providers::ReadSpec& source);
  // Where an extent's contents come from and how they arrive (PageSource).
  // Refused (kBusy) while a load or eviction of it is in flight, or while
  // it is resident with backing mapped somewhere else, since its unmap
  // must find what its load mapped (a resident extent with no source yet,
  // mapped by its owner, may name where that is), or, for a write-back
  // place, with its contents somewhere other than the new source says;
  // also refused (kBusy) while a nonresident extent's contents are
  // preserved, unless the new source names the same write-back range.
  // kInvalid for a landed source with no landing zone, a range that is
  // empty or exceeds a slot, no destination, a write instead of a read,
  // or a write-back place for an extent that is not kPreserve.
  // Registering a new place for a nonresident extent relocates its next
  // load (BP-P5), unless its place is pinned: then a source whose place
  // differs is refused (kBusy) in every state.
  std::expected<void, WorkError> SetSource(catalog::ExtentId extent, const PageSource& source);
  // Pins the places of `extents` (D-090), each of which must have a
  // source (else kUnavailable, pinning none). Pins count: each extent
  // pinned n times stays pinned until unpinned n times.
  std::expected<void, WorkError> PinPlaces(std::span<const catalog::ExtentId> extents);
  // Unpins one pin of each; an extent not pinned is left alone. A pin is
  // what lets a captured graph name the place (invariant 2's
  // registration): the caller unpins only once every graph naming these
  // extents is destroyed, never before; the scheduler cannot see graphs.
  void UnpinPlaces(std::span<const catalog::ExtentId> extents);
  bool PlacePinned(catalog::ExtentId extent) const { return pinned_.contains(extent); }
  PlacementStamp placement_stamp() const { return placement_stamp_; }
  // Any thread: a count that moves whenever a place may have changed (every
  // change of the stamp's epoch). Unchanged since a clean check of every
  // place, none moved, with no turn of this thread's (a driver's step).
  std::uint64_t placement_changes() const {
    return placement_changes_.load(std::memory_order_acquire);
  }
  // Requests' leases held now (HoldLease), including ones ending once
  // their operations drain, and the operations under `lease` in flight.
  std::size_t held() const { return held_.size(); }
  std::uint32_t HeldOperations(catalog::LeaseId lease) const {
    const auto found = held_.find(lease);
    return found == held_.end() ? 0 : found->second.operations;
  }
  // The source registered for an extent, if any: where its next load puts
  // its contents, and where its backing is mapped while it is resident.
  const PageSource* SourceOf(catalog::ExtentId extent) const {
    const auto found = sources_.find(extent);
    return found == sources_.end() ? nullptr : &found->second;
  }
  // One turn; true if anything happened.
  bool Turn();
  // Turns until shutdown completes. Returns its result: a fault if work
  // could not be reconciled.
  std::expected<void, Fault> Run();
  // Once shutting down: nothing while work drains, then the result.
  std::optional<std::expected<void, Fault>> Stopped() const;

  const catalog::Catalog& catalog() const { return catalog_; }
  const TaskTable& tasks() const { return tasks_; }
  std::size_t operations() const { return board_.open(); }
  std::size_t quarantined() const { return quarantined_; }
  std::optional<Fault> fault() const { return fault_; }
  // The operation of the page-in stage in flight for an extent, if any.
  std::optional<OperationId> LoadOf(catalog::ExtentId extent) const {
    const auto found = loads_.find(extent);
    return found != loads_.end() && found->second.operation.valid()
               ? std::optional(found->second.operation)
               : std::nullopt;
  }
  // Page-ins in flight, in any stage.
  std::size_t loads() const { return loads_.size(); }
  // Evictions parked with their backing kept for a handoff.
  std::size_t parked() const { return parked_count_; }
  // A policy victim's unmap completed, even if its donor backing is
  // still parked/releasing. Such backing is not free catalog capacity.
  bool EvictionUnmapped(catalog::ExtentId extent) const {
    const auto found = evictions_.find(extent);
    return found != evictions_.end() && (found->second.stage == EvictStage::kParked ||
                                         found->second.stage == EvictStage::kReleasing);
  }
  // Evictions in flight, in any stage (parked, and releasing, included).
  std::size_t evictions() const { return evictions_.size(); }
  // Landing slots in use by a read or copy, and quarantined ones.
  std::size_t slots_busy() const;
  std::size_t slots_quarantined() const;
  // Landed loads whose copy into device memory a lane has taken and whose
  // fence has not yet been seen.
  std::size_t copying() const;
  const SchedulerStats& stats() const { return stats_; }

 private:
  friend class TaskContext;

  // What an operation is: a stage of a page-in, the unmap of an eviction,
  // or a task's device or CPU work.
  enum class Kind : std::uint8_t { kLoad, kEvict, kDevice, kCpu };
  // The lane an operation's command goes to.
  enum class Route : std::uint8_t { kStorage, kDevice, kCpu, kBacking, kCopy };
  static constexpr std::size_t kRoutes = 5;
  // Where VMM work goes: the VMM lane, or the device lane without one (in
  // order with its other work, as before there was a VMM lane).
  Route BackingRoute() const {
    return lanes_.backing != nullptr ? Route::kBacking : Route::kDevice;
  }
  // Where the zone's copies go: the copy lane, or the device lane without
  // one.
  Route CopyRoute() const { return lanes_.copy != nullptr ? Route::kCopy : Route::kDevice; }
  bool CanCopy() const { return lanes_.copy != nullptr || lanes_.device != nullptr; }

  // A page-in's stages (the header's list).
  enum class Stage : std::uint8_t {
    kQueued,     // waiting for room among the landed loads in flight
    kMapping,    // 1: creating (or taking kept backing) and mapping managed backing
    kSlot,       // 2: waiting for a landing slot
    kReading,    // 3
    kCopying,    // 4
    kUnmapping,  // unwinding a failed or withdrawn load's backing
    kReleasing,  // unwinding: releasing kept backing the load took and never mapped
  };
  enum class SlotState : std::uint8_t { kFree, kBusy, kQuarantined };

  // A page-in, the scheduler's: one per extent, with its waiters.
  struct Load {
    catalog::Ticket ticket;
    PageSource source;
    Stage stage = Stage::kQueued;
    OperationId operation;  // the stage's, while one is open
    std::optional<std::size_t> slot;
    std::vector<TaskId> waiters;
    bool cancelling = false;  // every waiter left
    bool started = false;     // counted among the landed loads in flight
    bool mapped = false;      // its managed backing is mapped
    bool failed = false;      // it will not publish
    // It took backing a parked eviction kept, which the VMM lane holds
    // until this load maps it (or releases it, unwinding).
    bool reuse = false;
  };
  // An eviction's stages: a write-back's (the header's list), then the
  // unmap of managed backing, and for a handoff the park.
  enum class EvictStage : std::uint8_t {
    kSlot,       // waiting for a landing slot
    kCopyOut,    // the copy lane copies the extent into the slot
    kWriting,    // the storage lane writes it to the write-back place
    kUnmapping,  // the VMM lane unmaps and releases (or, for a handoff, keeps) the backing
    kParked,     // unmapped, its backing kept for a load to take; no operation
    kReleasing,  // parked backing no load took, being released
  };
  // An eviction whose backing the VMM lane unmaps and releases, after
  // writing its contents back if it preserves them.
  struct Eviction {
    catalog::Ticket ticket;
    OperationId operation;
    TaskId evictor;               // the task that asked for it
    std::vector<TaskId> waiters;  // it, and tasks materializing the extent
    EvictStage stage = EvictStage::kUnmapping;
    std::optional<std::size_t> slot;
    bool write_back = false;
    bool handoff = false;
    bool release_open = false;  // kReleasing and counted in releases_open_
  };
  // Releases of parked backing no load took run a few at a time, so
  // thousands of them never take every mailbox from other work.
  static constexpr std::size_t kReleaseWindow = 16;
  // Parked evictions by the backing they keep, oldest first: an entry
  // whose eviction has since ended or left kParked is skipped.
  using BackingKey = std::pair<std::size_t, std::uint64_t>;  // allocation class, bytes

  struct TaskRecord {
    TaskId id;
    std::uint64_t request = 0;
    std::size_t priority = 0;
    std::unique_ptr<TaskProgram> program;
    std::uint32_t waiting = 0;  // page-ins, operations and children it waits for
    bool failed = false;        // a dependency failed since the program last looked
    bool finished = false;
    bool signalled = false;        // a signal came while it was not waiting for one
    bool awaiting_signal = false;  // counted in `waiting` until a signal comes
  };

  // A request's lease (HoldLease), by the catalog lease that is it.
  struct Held {
    TaskId task;                             // the holder
    std::vector<catalog::ExtentId> extents;  // sorted, unique: what it holds
    std::uint32_t operations = 0;            // submitted under it, not yet concluded
    // Work its holder's client runs under it itself (a driver's direct
    // steps, HoldLease), not yet seen to end: shared with the client, so it
    // outlives either.
    std::shared_ptr<const std::atomic<std::uint32_t>> external;
    bool ending = false;          // released once drained
    std::vector<TaskId> waiters;  // tasks woken when it is released
    // No operation under it, nor external work, may still touch it.
    bool Drained() const {
      return operations == 0 &&
             (external == nullptr || external->load(std::memory_order_acquire) == 0);
    }
  };

  struct Operation {
    OperationId id;
    Kind kind = Kind::kLoad;
    Route route = Route::kStorage;
    bool published = false;  // a lane took its command
    bool quarantined = false;
    bool critical = false;
    // A page-in stage or an eviction: the extent.
    catalog::ExtentId extent;
    // Device and CPU work: the task that owns it and the lease it holds:
    // its own, or, submitted under a request's lease, that one (`held`).
    TaskId task;
    catalog::LeaseId lease;
    catalog::LeaseId held;
    // The command, until a lane takes it.
    ReadCommand read;
    DeviceCommand device;
    CpuCommand job;
  };

  static WorkError ErrorOf(catalog::CatalogError error);
  TaskRecord* Record(TaskId task);
  Operation* Find(OperationId operation);
  // A live task's view; fatal for one that is not.
  TaskView ViewOf(TaskId task) const;

  // Moves from `program` only if the task is created.
  std::expected<TaskId, StartError> StartTask(std::uint64_t request,
                                              std::unique_ptr<TaskProgram>& program,
                                              std::size_t priority, std::optional<TaskId> parent);
  void Apply(Control&& control);
  void StepTask(TaskId task);
  void CancelTask(TaskId task);
  void FinishTask(TaskId task, TaskOutcome outcome);
  void TryRetire(TaskId task);
  void MakeReady(TaskRecord& record);
  void Wake(TaskId task, bool failed);
  void BeginStop();

  // TaskContext's work.
  std::expected<Readiness, WorkError> Materialize(TaskId task, const catalog::Closure& closure);
  // Moves from `device` or `job`, whichever `kind` names, only if the
  // operation is created. Under `closure`'s own lease, or with a valid
  // `held`, under that request's lease.
  std::expected<OperationId, WorkError> Submit(TaskId task, const catalog::Closure* closure,
                                               catalog::LeaseId held, Kind kind,
                                               DeviceCommand& device, CpuJob& job);
  std::expected<Readiness, WorkError> Evict(TaskId task, catalog::ExtentId extent,
                                            EvictOptions options);
  // Requests' leases.
  std::expected<catalog::LeaseId, WorkError> HoldLease(
      TaskId task, const catalog::Closure& closure,
      std::shared_ptr<const std::atomic<std::uint32_t>> external = nullptr);
  std::expected<Readiness, WorkError> EndLease(TaskId task, catalog::LeaseId lease);
  std::expected<Readiness, WorkError> AwaitRelease(TaskId task,
                                                   std::span<const catalog::ExtentId> extents);
  std::expected<Readiness, WorkError> AwaitSignal(TaskId task);
  void Signal(std::uint64_t request);
  // An operation under a request's lease concluded: the lease is released
  // if it is ending and that was the last.
  void ConcludeHeld(catalog::LeaseId lease);
  // An ending lease with no operation left under it: released now if its
  // external work has drained too, else once a turn sees it has
  // (ReleaseExternal). Its holder, if it waits, is woken then.
  void ReleaseWhenDrained(std::map<catalog::LeaseId, Held>::iterator held);
  // Each turn: the ending leases whose external work has drained since.
  bool ReleaseExternal();
  // Releases the lease (RecordUse, then the catalog's release) and wakes
  // its waiters.
  void ReleaseHeld(std::map<catalog::LeaseId, Held>::iterator held);
  // The runtime wake's follow window (SchedulerSettings::follow_limit): a
  // step (device work) being published notes the gap since the last
  // step's end; a step's end starts the window, over this thread and the
  // device lane.
  void NoteFollow();
  void AwaitFollow();

  // Page-ins (pagein.cc). With `own`, the extent's own parked eviction
  // has just ended and its kept backing is the load's.
  std::expected<void, WorkError> BeginPageIn(TaskRecord& record, catalog::ExtentId extent,
                                             const PageSource& source, bool own = false);
  // A parked eviction in `domain` keeping backing for `place`: the budget
  // is each domain's, so a charge never moves between domains.
  std::optional<catalog::ExtentId> Donor(const BackingPlace& place, catalog::DomainId domain);
  // Ends a parked eviction whose backing a load takes: completed, the
  // charge moving to the load.
  void EndParked(catalog::ExtentId extent);
  void Join(Load& load, TaskRecord& record);
  // Runs the load's next stage, from where it stands.
  void Proceed(catalog::ExtentId extent);
  // Opens and publishes a stage's operation; false if no mailbox is free
  // (the load waits in blocked_ and is retried each turn).
  bool OpenStage(catalog::ExtentId extent, Load& load);
  void OnStage(catalog::ExtentId extent, Outcome outcome, std::uint64_t bytes);
  void Unwind(catalog::ExtentId extent, Load& load);
  void EndLoad(catalog::ExtentId extent, bool loaded);
  void QuarantineLoad(catalog::ExtentId extent, Fault fault);
  void CancelPageIn(catalog::ExtentId extent, Load& load);
  void FreeSlot(std::size_t slot);
  void StartQueued();
  bool RetryBlocked();
  // Every mailbox is retired or held by quarantined work: none will free.
  bool NoMailboxEver() const;
  // Evictions.
  // A write-back's slot, in order: the first free one, else wait for one.
  void ProceedEviction(catalog::ExtentId extent);
  // Opens and publishes the stage's operation; false if no mailbox is
  // free (it waits in blocked_). One that can never get a mailbox
  // abandons the eviction: no stage it did not open touched the backing.
  bool OpenEvictStage(catalog::ExtentId extent, Eviction& eviction);
  void OnEvicted(catalog::ExtentId extent, Outcome outcome, std::uint64_t bytes);
  // Settles the eviction: completed, or abandoned with the extent resident
  // again (only the evictor is told of that).
  void EndEviction(catalog::ExtentId extent, bool evicted);
  // A handoff's unmap succeeded: the eviction parks and its waiters are
  // told it is done.
  void Park(catalog::ExtentId extent);
  // Releases the backing of `task`'s parked evictions (every one's with no
  // valid task), which no load took: queued, and opened kReleaseWindow at
  // a time (PumpReleases).
  void ReleaseParked(std::optional<TaskId> task);
  void PumpReleases();
  void QuarantineEvicting(catalog::ExtentId extent, Fault fault);
  void Fail(Fault fault);

  // The operation lifecycle.
  Operation& Open(Kind kind);
  void Publish(Operation& operation);
  base::PushResult Push(Operation& operation) const;
  bool Flush();
  void Withdraw(TaskId task);
  void RollBack(Operation& operation);
  void Observe(const Observation& seen);
  void Conclude(Operation& operation, Outcome outcome, std::uint64_t bytes);
  void Quarantine(Operation& operation, Fault fault);
  void SetCritical(Operation& operation, bool critical);

  catalog::Catalog& catalog_;
  CompletionBoard& board_;
  base::WakeFlag& wake_;
  Lanes lanes_;
  SchedulerSettings settings_;
  base::BoundedQueue<Control> controls_;
  // Requests with a cancellation queued and no start posted since: at most
  // one entry per queued control, allocated once. Held across a push, so
  // a push and its entry change together; taken after the queue's lock,
  // never inside it.
  std::mutex cancel_mutex_;
  std::vector<std::uint64_t> cancel_intents_;
  std::atomic<bool> shutdown_{false};

  // Everything below is the scheduler thread's.
  TaskTable tasks_;
  std::vector<std::optional<TaskRecord>> records_;    // by task index
  std::vector<std::optional<Operation>> operations_;  // by mailbox index
  std::map<catalog::ExtentId, Load> loads_;           // page-ins in flight
  std::map<catalog::ExtentId, Eviction> evictions_;
  std::map<BackingKey, std::deque<catalog::ExtentId>> parked_;
  std::size_t parked_count_ = 0;                 // evictions in kParked
  std::deque<catalog::ExtentId> release_queue_;  // kReleasing, not yet opened
  std::size_t releases_open_ = 0;
  // PumpReleases is running: a release it opens that settles at once (a
  // quarantine) lets the running loop open the next, rather than recursing
  // once per queued release.
  bool pumping_releases_ = false;
  std::map<catalog::ExtentId, PageSource> sources_;
  std::map<catalog::ExtentId, std::uint32_t> pinned_;  // D-090: pins per extent
  friend struct SchedulerPlacementTestAccess;
  static std::uint64_t TakePlacementInstance(std::atomic<std::uint64_t>& next);
  void PlacementChanged() {
    if (placement_stamp_.epoch != UINT64_MAX) ++placement_stamp_.epoch;
    placement_changes_.fetch_add(1, std::memory_order_acq_rel);
  }
  PlacementStamp placement_stamp_;
  std::atomic<std::uint64_t> placement_changes_{0};
  // Requests' leases, at most `tasks` at once.
  std::map<catalog::LeaseId, Held> held_;
  // Ending leases kept only by external work still under way.
  std::vector<catalog::LeaseId> external_ending_;
  // The landing zone's slots, and the loads waiting: to start (kQueued),
  // for a slot (kSlot), or for a mailbox to open their next stage.
  std::vector<SlotState> slots_;
  std::size_t started_ = 0;  // landed loads past kQueued
  std::deque<catalog::ExtentId> queued_;
  std::deque<catalog::ExtentId> slot_waiters_;
  std::deque<catalog::ExtentId> blocked_;
  // Commands a full lane refused, per lane, in publication order.
  std::array<std::deque<OperationId>, kRoutes> pending_;
  // Page-in cancellations the storage lane refused, to ask again.
  std::vector<OperationId> cancels_;
  std::size_t critical_ = 0;
  // The runtime wake's follow window (SchedulerSettings::follow_limit): the
  // last step's end, if no step has been published since; the recent gap
  // from a step's end to the next one's publication, in nanoseconds (zero
  // until measured); and until when to poll for it.
  std::chrono::steady_clock::time_point step_done_;
  std::int64_t follow_gap_ = 0;
  std::chrono::steady_clock::time_point follow_until_;
  std::size_t quarantined_ = 0;  // quarantined operations, each keeping its mailbox
  std::optional<Fault> fault_;
  bool stopping_ = false;
  std::uint64_t turn_ = 0;
  SchedulerStats stats_;
};

// What a task's step may do, on the scheduler thread.
class TaskContext {
 public:
  TaskContext(const TaskContext&) = delete;
  TaskContext& operator=(const TaskContext&) = delete;
  TaskContext(TaskContext&&) = delete;
  TaskContext& operator=(TaskContext&&) = delete;
  ~TaskContext() = default;

  TaskId task() const { return task_; }
  // The scheduler's turn counter, the logical tick of recorded uses.
  std::uint64_t tick() const { return scheduler_.turn_; }
  const catalog::Catalog& catalog() const { return scheduler_.catalog_; }
  // Whether a dependency (a page-in, an operation) failed since the last
  // call; clears it.
  bool TakeFailure();
  bool ChildFailed() const;

  std::expected<Readiness, WorkError> Materialize(const catalog::Closure& closure) {
    return scheduler_.Materialize(task_, closure);
  }
  // The operation's result wakes the task; a failure shows in TakeFailure.
  std::expected<OperationId, WorkError> SubmitDevice(const catalog::Closure& closure,
                                                     const DeviceWork& work) {
    DeviceCommand device{.operation = {}, .work = work};
    CpuJob none;
    return scheduler_.Submit(task_, &closure, {}, Scheduler::Kind::kDevice, device, none);
  }
  // A job that queues kernel work on one of the device lane's streams
  // (commands.h): it runs on the submission lane, and the operation's
  // lease holds the closure until the fence after it completes. Refused,
  // the work is left with the caller (to try again after kBusy).
  std::expected<OperationId, WorkError> SubmitLaunch(const catalog::Closure& closure,
                                                     LaunchWork&& work) {
    return Launch(&closure, {}, std::move(work));
  }
  std::expected<OperationId, WorkError> SubmitCpu(const catalog::Closure& closure, CpuJob job) {
    DeviceCommand none;
    return scheduler_.Submit(task_, &closure, {}, Scheduler::Kind::kCpu, none, job);
  }

  // A request's lease (the header's list). Leases a materialized closure,
  // all or none, at the contents it recorded, for this task, until it ends
  // it or finishes: kNotResident, kStale or kBusy (an eviction begun) if
  // it is not all resident at those contents; kBusy if as many leases as
  // the task bound are held.
  // `external`, if given, counts work the request's client runs under the
  // lease itself (PagedNode's direct steps): the lease is never released
  // while it is not zero. The count is shared, so whichever of the lease
  // and the client goes last frees it; the client wakes the scheduler when
  // it drops to zero on an ending lease, and a count that never drops
  // keeps the lease for good (unproven work, as a quarantined operation's).
  std::expected<catalog::LeaseId, WorkError> HoldLease(
      const catalog::Closure& closure,
      std::shared_ptr<const std::atomic<std::uint32_t>> external = nullptr) {
    return scheduler_.HoldLease(task_, closure, std::move(external));
  }
  // Device work under this task's lease `held`: nothing is leased or
  // walked per operation, and the lease cannot be released before the
  // operation's fence. kInvalid for a lease this task does not hold,
  // kClosed for one it has ended. Refused, the work is left with the
  // caller.
  std::expected<OperationId, WorkError> SubmitLaunch(catalog::LeaseId held, LaunchWork&& work) {
    return Launch(nullptr, held, std::move(work));
  }
  std::expected<OperationId, WorkError> SubmitDevice(catalog::LeaseId held,
                                                     const DeviceWork& work) {
    DeviceCommand device{.operation = {}, .work = work};
    CpuJob none;
    return scheduler_.Submit(task_, nullptr, held, Scheduler::Kind::kDevice, device, none);
  }
  // Ends this task's lease: released at once (kReady) with no operation
  // under it in flight, else once the last one's fence completes
  // (kWaiting: the task is woken then). Its extents stay resident.
  // kInvalid for a lease this task does not hold.
  std::expected<Readiness, WorkError> EndLease(catalog::LeaseId held) {
    return scheduler_.EndLease(task_, held);
  }
  // Whether another task's request lease holds any of `extents`: kReady if
  // none does, else kWaiting, and the task is woken when that lease is
  // released (then asks again). kBusy if its waiter list is full. kInvalid
  // if this task or an ancestor holds a request's lease not yet ended:
  // holding one while waiting for another's could close a cycle. The wait
  // has no bound: the lease is released only when its holder ends it (or
  // finishes), whatever the waiter's priority, so whoever asks for a swap
  // ends the requests in its way first (D-093).
  std::expected<Readiness, WorkError> AwaitRelease(std::span<const catalog::ExtentId> extents) {
    return scheduler_.AwaitRelease(task_, extents);
  }
  // kReady if a signal (SignalRequest) came since the last wait, which it
  // takes; else kWaiting, and the task is woken by the next one.
  std::expected<Readiness, WorkError> AwaitSignal() { return scheduler_.AwaitSignal(task_); }
  // A child, ready at once; its finishing wakes this task.
  std::expected<TaskId, WorkError> Spawn(std::unique_ptr<TaskProgram> program,
                                         std::size_t priority = 1);
  // Evicts an unheld, evictable extent. With managed backing the VMM
  // lane unmaps and releases it first: kWaiting, and the task is woken
  // once the extent is nonresident (or, if the unmap failed, with a
  // failure). Otherwise at once, in the catalog only: its backing stays
  // mapped, as its source registered it. A kPreserve extent whose source
  // is its write-back place, and whose contents were not invalidated, is
  // written back first (kWaiting either way); an abandoned write-back
  // wakes the task with a failure and leaves the extent resident.
  //
  // With a handoff (EvictOptions), managed backing is kept rather than
  // released: the task is woken once it is unmapped, the extent stays
  // EVICTING and charged, and a load may take the backing (the header's
  // handoff). Whatever no load took is released when this task finishes.
  std::expected<Readiness, WorkError> Evict(catalog::ExtentId extent, EvictOptions options = {}) {
    return scheduler_.Evict(task_, extent, options);
  }

 private:
  friend class Scheduler;
  TaskContext(Scheduler& scheduler, TaskId task) : scheduler_(scheduler), task_(task) {}

  std::expected<OperationId, WorkError> Launch(const catalog::Closure* closure,
                                               catalog::LeaseId held, LaunchWork&& work) {
    DeviceCommand device{.operation = {}, .work = std::move(work)};
    CpuJob none;
    auto submitted =
        scheduler_.Submit(task_, closure, held, Scheduler::Kind::kDevice, device, none);
    if (!submitted) {
      work = std::move(std::get<LaunchWork>(device.work));  // Submit took nothing
    }
    return submitted;
  }

  Scheduler& scheduler_;
  TaskId task_;
};

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_SCHEDULER_H_
