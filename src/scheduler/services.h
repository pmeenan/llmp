// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The provider lanes (D-048, docs/async-model.md#execution-and-thread-ownership):
// the storage service over one Storage provider, the device service's
// submission and completion lanes over one DeviceExecution provider, and
// the handler of the CPU worker lane (a scheduler::Lane). Each takes owned
// commands from a bounded queue, carries them out on its own thread, and
// reports every operation it was handed through the completion board: its
// acceptance (not started, accepted or unknown) and later its terminal
// result, with the proof that the provider touches its memory no more.
// No lane touches the scheduler's records or runs a continuation.
//
// Each service is a set of non-blocking turns and a Run loop that repeats
// them on a thread. Deterministic tests call the turns themselves, in the
// order they choose; programs run the loops on threads (only programs start
// threads: docs/architecture.md#layers-and-dependency-rules). Close refuses
// new commands; the loops carry out what was queued, drain every operation
// they accepted, and return.

#ifndef LLMP_SCHEDULER_SERVICES_H_
#define LLMP_SCHEDULER_SERVICES_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/wake.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"

namespace llmp::scheduler {

// Internal cap-only attribution, with no per-operation history or heap storage.
// Intervals use steady_clock nanoseconds and identify one actual lane call.
struct BackingTimingEvent {
  std::uint64_t serial = 0;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;  // zero while the operation is in flight
};
struct BackingTimingTotals {
  std::uint64_t started = 0;
  std::uint64_t completed = 0;
  std::uint64_t total_ns = 0;
  BackingTimingEvent first;
  BackingTimingEvent last;
  BackingTimingEvent longest;
};
struct BackingParkStageTotals {
  std::uint64_t completed = 0;
  std::uint64_t total_ns = 0;
  BackingTimingEvent longest;
};
struct BackingTimingStats {
  bool enabled = false;
  std::uint64_t create_serial = 0;
  bool current_create_reserve = false;
  BackingTimingEvent current_create;
  BackingTimingTotals reserve;
  BackingTimingTotals ordinary;
  BackingTimingEvent current_park;
  BackingTimingTotals park;
  BackingParkStageTotals park_metadata;
  BackingParkStageTotals park_stash;
  BackingParkStageTotals park_publication;
};
static_assert(sizeof(BackingTimingStats) <= 512);

// Actual device-memory provider Create outcomes on the backing lane. These
// diagnostics do not imply that an internally retried driver call was clean.
struct BackingCreateCounts {
  std::uint64_t startup_pool_creates = 0;
  std::uint64_t reserve_attempts = 0;
  std::uint64_t reserve_failures = 0;
  std::uint64_t ordinary_attempts = 0;
  std::uint64_t ordinary_failures = 0;
  std::uint64_t last_failure_monotonic_ns = 0;
  bool last_failure_reserve = false;
  providers::ProviderError last_failure_error = providers::ProviderError::kFailed;
  BackingTimingStats timing;

  std::uint64_t post_init_creates() const noexcept { return ordinary_attempts + reserve_attempts; }
};

// Only consumers of the provider's failure text need an allocating snapshot.
struct BackingCreateStats : BackingCreateCounts {
  std::string last_failure_detail;
};

class BackingCreateCounters {
 public:
  explicit BackingCreateCounters(bool timing = false) : timing_enabled_(timing) {}
  void StartupPoolCreated() { startup_pool_creates_.fetch_add(1, std::memory_order_relaxed); }
  void StartupPoolRolledBack(std::size_t count) {
    startup_pool_creates_.fetch_sub(count, std::memory_order_relaxed);
  }
  void Attempt(bool reserve);
  void Completed(bool reserve);
  void BeginPark();
  void ParkMetadataDone();
  void ParkStashDone();
  void EndPark();
  void Failed(bool reserve, const providers::Failure& failure);
  BackingCreateStats Snapshot() const;
  BackingCreateCounts Counts() const;

 private:
  std::atomic<std::uint64_t> startup_pool_creates_ = 0;
  std::atomic<std::uint64_t> reserve_attempts_ = 0;
  std::atomic<std::uint64_t> ordinary_attempts_ = 0;
  const bool timing_enabled_;
  mutable std::mutex mutex_;  // never held across a provider call
  BackingCreateStats failures_;
  std::uint64_t park_metadata_end_ns_ = 0;
  std::uint64_t park_stash_end_ns_ = 0;
};

struct QueueSettings {
  std::size_t capacity = 64;  // queued commands, including the reserve
  std::size_t reserved = 8;   // held back for cleanup commands
  std::size_t batch = 16;     // commands a turn takes
};

// A pool of created, unmapped backing of one class and size (D-033 as
// amended 2026-10-07; D-124 as amended 2026-10-25), kept by the VMM lane: a
// map of that class and size takes one instead of creating backing, and a
// plain unmap's backing, or a kept one released, goes back to it instead of
// being released while it holds fewer than its ceiling (`count`, which
// SetCeiling may lower and RestoreCeiling raise again).
//
// What it holds is not catalog occupancy (D-124: no phantom extents), so it
// stays within the budget's headroom instead: what it holds beside the
// domain's occupancy never exceeds the budget (invariant 5, pager-
// invariants.md). Taking one for a charged load, or keeping an evicted
// extent's, moves backing between the pool and the occupancy and changes
// nothing; only the pool's own creates, a load's own create, and the
// occupancy or budget changing move the headroom. Managed (the paged
// node's), the scheduler tells it the headroom at each turn (Rebalance):
// it then sheds what lies past the headroom (the lane releases it) and may
// create (Refill) at most what lies certainly within it (the grant), which
// every create the lane makes uses up. Unmanaged (a lane on its own, as in
// the lane tests), it refills up to its ceiling.
struct ReserveSettings {
  std::size_t allocation_class = 0;
  Bytes size;
  std::size_t count = 0;  // the ceiling; 0: no reserve
};
class HandleReserve {
 public:
  explicit HandleReserve(ReserveSettings settings)
      : settings_(settings), ceiling_(settings.count) {}
  bool Holds(std::size_t allocation_class, Bytes size) const {
    std::lock_guard lock(mutex_);
    return settings_.count != 0 && allocation_class == settings_.allocation_class &&
           size == settings_.size;
  }
  // One of that class and size, if any is kept.
  std::optional<providers::BackingId> Take(std::size_t allocation_class, Bytes size);
  // Keeps `backing` (of that class and size) if the reserve is short of
  // its ceiling.
  bool Keep(std::size_t allocation_class, Bytes size, providers::BackingId backing);
  // Startup preallocation's (BackingService::Preallocate).
  void KeepUnchecked(providers::BackingId backing) {
    std::lock_guard lock(mutex_);
    kept_.push_back(backing);
    ++version_;
  }
  // A create the lane made for anything but the pool (a load's own):
  // it took headroom the grant may have counted on.
  void Created() {
    std::lock_guard lock(mutex_);
    if (managed_ && grant_ != 0) --grant_;
  }
  // A refill may start now: one create is counted in flight, and the
  // grant (managed) is used by one.
  bool BeginRefill();
  // The refill's outcome: kept (true), or, when the ceiling fell or a
  // drain began meanwhile, shed (false: the lane releases it with the
  // pool's other sheds, and a drain waits for it); or none. `lost`
  // (unknown outcome) counts the backing that may exist against the
  // ceiling for good. With `known_failure` while managed, the ceiling
  // falls to what it holds: the pool grows no further (the device is
  // short of memory), and loads create their own.
  bool EndRefill(std::optional<providers::BackingId> created, bool lost, bool known_failure);
  // The scheduler's thread: the budget's headroom beside the domain's
  // occupancy, in handles, and how many page-ins and evictions are in flight
  // (each may count one handle twice: charged and still kept, until the lane
  // has taken or kept it). Sheds what lies past the headroom even counting
  // those, and grants what lies within it counting nothing twice. True if
  // the lane has work (a shed, or a refill it may start). `seen`: the
  // version it left, read under its lock, so a change another thread makes
  // just after is never taken as seen.
  bool Rebalance(std::size_t headroom, std::size_t transients, std::uint64_t* seen = nullptr);
  // Any thread: the ceiling, now at most `ceiling` (lowered, never raised),
  // with what lies past it shed, and the pool drained until RestoreCeiling:
  // it creates nothing (no refill), so it never grows back meanwhile; an
  // unmap's backing may still be kept up to the lowered ceiling, which
  // moves bytes from the occupancy and adds none. Returns how many it shed.
  std::size_t SetCeiling(std::size_t ceiling);
  void RestoreCeiling() {
    std::lock_guard lock(mutex_);
    ceiling_ = settings_.count;
    drained_ = false;
    ++version_;
  }
  // The lane: handles shed, to release; then what became of them.
  std::vector<providers::BackingId> TakeShed();
  void Shed(std::size_t released, std::span<const providers::BackingId> refused, std::size_t lost);
  // Teardown, once the lane runs no more: everything it holds.
  std::vector<providers::BackingId> TakeAll();
  // Managed from now on, granted `grant` creates.
  void Manage(std::size_t grant) {
    std::lock_guard lock(mutex_);
    managed_ = true;
    grant_ = grant;
    ++version_;
  }
  ReserveSettings settings() const {
    std::lock_guard lock(mutex_);
    return settings_;
  }
  std::size_t size() const {
    std::lock_guard lock(mutex_);
    return kept_.size();
  }
  // What it aims to hold now: its ceiling unmanaged; managed, what it holds
  // and may still create, within the ceiling.
  std::size_t target() const;
  std::size_t shedding() const {
    std::lock_guard lock(mutex_);
    return shedding_;
  }
  // Handles a drain waits for: shed and not yet released, or being
  // created (a drained pool sheds what a create in flight makes).
  std::size_t settling() const {
    std::lock_guard lock(mutex_);
    return shedding_ + creating_;
  }
  // Moves whenever what it holds, or may create, changes (any thread).
  std::uint64_t version() const { return version_.load(std::memory_order_acquire); }

 private:
  std::size_t RemainingLocked() const { return kept_.size() + lost_ + creating_; }
  // What counts against the ceiling for a shed or a keep: drained, a
  // create in flight is shed when it ends (EndRefill), so it is not.
  std::size_t HeldLocked() const { return kept_.size() + lost_ + (drained_ ? 0 : creating_); }

  mutable std::mutex mutex_;
  ReserveSettings settings_;
  std::size_t ceiling_ = 0;
  std::vector<providers::BackingId> kept_;
  std::vector<providers::BackingId> shed_;  // taken out of kept_, for the lane to release
  std::size_t shedding_ = 0;                // shed_ and those the lane is releasing
  std::size_t lost_ = 0;
  std::size_t creating_ = 0;
  bool managed_ = false;
  std::size_t grant_ = 0;
  // A shed's release was refused: none until the lane's next command.
  bool shed_refused_ = false;
  // SetCeiling's, until RestoreCeiling: no refill.
  bool drained_ = false;
  std::atomic<std::uint64_t> version_ = 0;

  friend class BackingService;
};

// Backing kept for a handoff and not yet mapped again or released
// (D-033; BackingWork's `retain`, `lazy`, `reuse` and kRelease): the lane
// that calls the device-memory provider keeps it, on its own thread. Every
// entry is charged in the catalog to an extent the scheduler names
// (scheduler.h), never an idle pool of its own. A lazy entry is still
// mapped at the place it was evicted from (`at`) until a load or release
// takes it, which unmaps it first.
class HandoffStash {
 public:
  struct Place {
    providers::ReservationId reservation;
    Bytes offset;
    auto operator<=>(const Place&) const = default;
  };
  struct Kept {
    providers::BackingId backing;
    std::optional<Place> at;  // still mapped there
  };
  // `oldest`: one a provider call refused goes behind every other, so the
  // takes that follow try those first.
  void Put(std::size_t allocation_class, Bytes size, Kept kept, bool oldest = false);
  // The most recently kept backing of that class and size, if any.
  std::optional<Kept> Take(std::size_t allocation_class, Bytes size);
  // The kept backing still mapped at `place`, if one of that class and
  // size is, taken (one of another, filed under its own, is left there).
  std::optional<Kept> TakeAt(const Place& place, std::size_t allocation_class, Bytes size);
  std::size_t size() const { return entries_.size(); }

 private:
  struct Entry {
    std::size_t allocation_class = 0;
    Bytes size;
    Kept kept;
  };
  std::list<Entry> entries_;                            // oldest first
  std::map<Place, std::list<Entry>::iterator> mapped_;  // the lazy ones, by place
};

// The storage lane: whole reads (providers::DirectReader) into memory the
// scheduler protects, one read per operation. A read is accepted once the
// reader takes it, and ends only when every request it started has
// completed, so its terminal result always proves no further access. A
// cancelled read withdraws the operation's interest: the reader asks the
// provider to cancel what is in flight (a span shared with reads still
// wanted is left to complete) and still drains. With coalescing on
// (ReaderSettings::span_bytes, D-056; off by default), reads waiting for
// room that continue one another in a file go as vectored requests.
//
// The reader must take as many reads as the scheduler can have operations
// (ReaderSettings::reads), or an operation may be refused as not started.
// While requests are in flight and its poll window has passed, the Run loop
// waits in the provider for a completion; a queued command wakes it
// (Storage::Wake), so a new read or a
// cancellation never waits for an unrelated completion, even one that never
// comes. A cancellation the kernel cannot act on (a request stuck in an
// uninterruptible wait) still leaves the read to drain: its memory stays
// protected until then. Taking a cancellation publishes nothing, so the
// lane nudges the owner (CompletionBoard::Nudge): a command the full queue
// refused is offered again at once, not at the owner's next timer tick.
class StorageService {
 public:
  // `poll_window`: how long the Run loop keeps polling, for commands and
  // completions alike, after its last progress before it waits in the
  // provider or sleeps on its queue; zero never polls. A thread that
  // sleeps wakes slowly on the Spark (RE-017): asleep in io_uring_enter,
  // this lane took up to ~200 us (p90) to pick up a new read, so the
  // landing zone ran below depth (docs/experiments/pagein-perf/). At most
  // an hour.
  StorageService(providers::Storage& storage, providers::ReaderSettings reader,
                 CompletionBoard& board, QueueSettings queue,
                 std::chrono::microseconds poll_window = std::chrono::microseconds(200));

  // Any thread.
  base::PushResult Submit(const StorageCommand& command,
                          base::PushKind kind = base::PushKind::kOrdinary) {
    const base::PushResult pushed = queue_.TryPush(StorageCommand(command), kind);
    if (pushed == base::PushResult::kAccepted) {
      storage_.Wake();  // after the push: the lane looks again once woken
    }
    return pushed;
  }
  void Close() {
    queue_.Close();
    storage_.Wake();
  }

  // The lane's thread: one round of commands and completions. With
  // `wait`, and nothing else to do, waits in the provider for a completion
  // if a request is in flight. True if anything happened.
  bool Turn(bool wait);
  // Until closed, with every accepted read drained.
  void Run();

  std::size_t reads() const { return reader_.reads(); }

 private:
  void Handle(const StorageCommand& command);

  base::BoundedQueue<StorageCommand> queue_;
  providers::Storage& storage_;
  providers::DirectReader reader_;
  CompletionBoard& board_;
  std::size_t batch_;
  std::chrono::microseconds poll_window_;
};

struct DeviceSettings {
  QueueSettings queue;
  // Fences in transit from the submission lane to the completion lane, and
  // the most the completion lane holds at once (watched or awaiting
  // release): past that, submission waits for room (backpressure).
  std::size_t handoff = 64;
  // How the completion lane waits for a fence (the runtime wake,
  // docs/experiments/runtime-wake/). Each of the last eight lengths of
  // fences on its stream (recorded to seen, base::Expectation) is a likely
  // end: it sleeps until `spin_ahead` before the next likely end the fence
  // has not yet outlasted, querying at least every `backstop`, and spins
  // (yielding) until `spin_past` after it; from the start while its stream
  // has no history. Past the longest it backs off, sleeping from 50 us up
  // to `backstop` between queries. As it starts to spin it wakes the owner
  // (CompletionBoard::Anticipate) and its own submission lane to poll until
  // then too: a thread asleep on the Spark takes hundreds of microseconds
  // to run once woken (RE-017), which a decode step paid at every hop.
  // Each at most SchedulerSettings::kLongest; `backstop` positive.
  // Measured choices, not tuned to a model.
  std::chrono::microseconds spin_ahead{1000};
  std::chrono::microseconds spin_past{1000};
  std::chrono::microseconds backstop{1000};
  // How long the submission lane's Run loop keeps polling its queue (with
  // yield) after its last command before it sleeps; zero never polls. A
  // sleeping thread wakes slowly on the Spark (RE-017), and a page-in
  // through the zone hands this lane a copy every ~140 us at disk speed:
  // asleep between them, it took 94-199 us at the median to wake for the
  // next (docs/experiments/pagein-perf/). It also polls while an
  // anticipation runs (Anticipate). The scheduler's window, not a tuned
  // value; at most SchedulerSettings::kLongest.
  std::chrono::microseconds poll_window{200};
  // Known failures (the call changed nothing) in a row, of a query or a
  // release of one fence, before the completion lane stops asking: a
  // refusal that persists (a lost context) must not keep the lane, and
  // shutdown, waiting forever. A bound, not a tuned value.
  std::uint32_t refusals = 8;
};

// The device service's two lanes over one execution provider, and the
// device-memory provider if it manages backing. The submission lane queues
// each operation's copies, or runs its job, on one of its streams and
// records a fence after them; the completion lane queries the fences
// without blocking and publishes each completion, independent of any
// submission call that may block (D-048). Releasing a fence is a
// submission-side call, so the completion lane releases only when the
// submission lane is between calls (never waiting for it) and otherwise
// tries again next turn. VMM work (BackingWork) given to this service
// runs on the submission lane too, in order with the rest, and is
// published at once: there is nothing to fence. A program that pages
// through the zone gives it to a BackingService instead (below), so a
// load's copies never wait behind it, and then gives this service no
// device-memory provider: one lane calls a provider (device_memory.h).
// Such a program also gives the zone's copies to a second DeviceService
// over the zone's stream alone (the copy lane, scheduler.h's Lanes::copy):
// a job launching into a full stream blocks this lane's thread (RE-029),
// and the copies must not wait for it.
//
// A copy the provider refused queued nothing; if an operation's first copy
// is refused it did not start. A later refusal leaves earlier copies
// queued: the operation is accepted and fails once its fence completes. An
// unknown outcome is accepted as unknown and fenced the same way. A job
// reports the same three cases (JobResult). VMM work whose outcome is
// known succeeded, or changed nothing (not started); an unknown outcome,
// a mapping undone only in part, or an unmap the provider refuses because
// an earlier unknown outcome left its place undetermined, is published
// with no proof, and the scheduler quarantines what it names. A fence
// that cannot be recorded, or a query whose outcome is unknown, leaves the
// work unproven: the terminal result carries no proof, and the scheduler
// quarantines what the operation holds. Such a fence is never released.
// A query the provider keeps refusing (DeviceSettings::refusals) is
// treated the same way; a release it keeps refusing is abandoned, since
// the work it fenced was already proven complete, and the fence stays
// recorded (its stream cannot be destroyed).
//
// Between turns (the runtime wake, docs/experiments/runtime-wake/): the
// submission lane polls its queue for its poll window after its last
// command and while an anticipation runs (Anticipate), and otherwise
// sleeps on its own wake flag, which Submit and Close signal. The
// completion lane sleeps while it holds no fence, and otherwise through
// most of each fence, spinning only around its likely ends (its stream's
// recent lengths, DeviceSettings), and as it starts to spin it has the
// owner and the submission lane poll until then too. What the lanes publish, and when a
// fence counts as complete, do not depend on any of it: only a query that
// sees a fence complete proves it, and a sleeping lane still queries at
// least every `backstop`.
//
// The streams belong to the program, which destroys them once both lanes
// have returned.
class DeviceService {
 public:
  // Without `memory`, VMM work is refused as not started.
  DeviceService(providers::DeviceExecution& execution, std::span<const providers::StreamId> streams,
                CompletionBoard& board, DeviceSettings settings,
                providers::DeviceMemory* memory = nullptr);

  std::size_t streams() const { return streams_.size(); }
  // Backing kept for a handoff (VMM work given to this service): the
  // submission lane's, read only while it is idle (tests, teardown).
  std::size_t stashed() const { return stash_.size(); }
  // Moves from `command` only if it is accepted.
  base::PushResult Submit(DeviceCommand&& command,
                          base::PushKind kind = base::PushKind::kOrdinary) {
    const base::PushResult pushed = queue_.TryPush(std::move(command), kind);
    if (pushed == base::PushResult::kAccepted) {
      submission_wake_.Signal();  // after the push: the lane looks again once woken
    }
    return pushed;
  }
  // No new commands. The submission lane returns once it has carried out
  // what was queued; the completion lane, once every fence it was handed
  // has been seen and released (or is unproven).
  void Close() {
    queue_.Close();
    submission_wake_.Signal();
  }
  // Any thread: the submission lane should poll its queue until `until`,
  // since a command is likely by then (the scheduler, once a step is done,
  // for the next; the completion lane, as a fence's likely end nears). A
  // hint: no command is lost without it.
  void Anticipate(base::WakeFlag::Clock::time_point until) { submission_wake_.Anticipate(until); }
  // The submission lane polls now (tests).
  bool Anticipating(base::WakeFlag::Clock::time_point now) const {
    return submission_wake_.Anticipating(now);
  }

  // The submission lane's thread.
  bool SubmissionTurn();
  void RunSubmission();
  // The completion lane's thread.
  bool CompletionTurn();
  void RunCompletion();

 private:
  using Clock = base::WakeFlag::Clock;
  struct Watch {
    OperationId operation;  // invalid for a fence that only balances its stream
    providers::FenceId fence;
    Outcome outcome = Outcome::kSucceeded;
    std::uint64_t bytes = 0;
    std::uint32_t refusals = 0;  // known query failures in a row
    std::uint32_t stream = 0;    // an index into streams_
    Clock::time_point recorded;  // when the fence was recorded
    // Until when the owner and the submission lane were last told to poll
    // for it (an end it nears).
    Clock::time_point relayed;
  };
  struct Release {
    providers::FenceId fence;
    std::uint32_t refusals = 0;  // known release failures in a row
  };

  void Launch(DeviceCommand& command);
  // The parts of Launch: each resolves the operation's acceptance, and its
  // terminal result or the fence that will prove it.
  void Copy(OperationId operation, const DeviceWork& work);
  void Run(OperationId operation, LaunchWork& work);
  void Back(OperationId operation, const BackingWork& work);
  // Records a fence after what a copy or job queued on streams_[stream]
  // and hands it over.
  void Fence(OperationId operation, std::uint32_t stream, bool queued, bool refused, bool unknown,
             std::uint64_t bytes);
  // Completion lane: how long it waits (or spins) before its next turn,
  // relaying to the owner and the submission lane as fences near their end.
  void AwaitCompletion(bool releasing);
  // Submission side: hands a fence to the completion lane, or keeps it
  // until there is room; nothing more launches meanwhile.
  void Hand(const Watch& watch);
  bool HandPending();
  // The submission lane has returned: nothing more will be handed over.
  void Finish();

  providers::DeviceExecution& execution_;
  providers::DeviceMemory* memory_;
  std::vector<providers::StreamId> streams_;
  CompletionBoard& board_;
  DeviceSettings settings_;
  base::BoundedQueue<DeviceCommand> queue_;
  base::BoundedQueue<Watch> handoff_;
  // Held by the submission lane around its provider calls; the completion
  // lane only ever tries it.
  std::mutex submitting_;
  // Submission lane only: at most one fence, since nothing launches while
  // it waits.
  std::optional<Watch> unhanded_;
  bool finished_ = false;
  // Completion lane only, together at most `handoff` (allocated once).
  std::vector<Watch> watches_;
  std::vector<Release> releases_;
  // Completion lane only: each stream's fences' recent lengths, and the
  // current back-off.
  std::vector<base::Expectation> expected_;
  Clock::duration backoff_;
  // Submission lane only: backing kept for a handoff.
  HandoffStash stash_;
  // The lanes' own wake flags: a command queued, or an anticipation, wakes
  // the submission lane; a fence handed over, or the handoff's close, the
  // completion lane.
  base::WakeFlag submission_wake_;
  base::WakeFlag completion_wake_;
};

// The VMM lane (D-033, D-081): managed backing's VMM work (BackingWork),
// carried out as the device service's submission lane would, on a thread of
// its own. Creating and mapping a 2 MiB extent of device backing took
// ~110 us on the GB10 (cuMemCreate ~70-76, cuMemSetAccess ~40-43), and a
// load maps one extent per ~140 us at disk speed: on the submission lane
// the copies queued behind it (docs/experiments/pagein-perf/). Nothing
// orders VMM work with device work through a queue: the scheduler
// publishes a load's read or copy only once its mapping has completed, a
// failed or withdrawn load's unmap only once its read or copy is proven
// to touch the memory no more (else it is quarantined), and an eviction's unmap
// only once every lease on the extent is released, so this lane changes
// no ordering the protocol relies on. Each operation is published at once, with the
// same outcomes as on the submission lane; any other work is refused as
// not started. Without a device-memory provider everything is refused.
// This lane is then the provider's one caller (device_memory.h): the
// DeviceService beside it is given none, so VMM work that reached it by
// mistake is refused rather than raced.
//
// It keeps the handle pool (HandleReserve): between commands it releases
// what the pool shed and creates what the pool may refill, one at a time,
// so a command waits for one provider call at most.
class BackingService {
 public:
  BackingService(providers::DeviceMemory* memory, CompletionBoard& board, QueueSettings queue,
                 ReserveSettings reserve = {}, bool timing = false);

  // Any thread. Moves from `command` only if it is accepted.
  base::PushResult Submit(DeviceCommand&& command,
                          base::PushKind kind = base::PushKind::kOrdinary) {
    return queue_.TryPush(std::move(command), kind);
  }
  void Close() { queue_.Close(); }

  // The lane's thread: one batch of commands. True if anything happened.
  bool Turn();
  // Until closed, with everything queued carried out.
  void Run();
  // The thread that will call Run, before it starts it: from here on the
  // provider is the lane's alone, so a DrainReserve meanwhile waits for the
  // lane instead of calling the provider beside it (device_memory.h).
  void Starting() {
    std::lock_guard lock(mu_);
    worker_running_ = true;
  }
  // Backing kept for a handoff: read only while the lane is idle (tests,
  // teardown).
  std::size_t stashed() const { return stash_.size(); }
  // The pool's backing (any thread; a snapshot).
  std::size_t reserved() const { return reserve_.size(); }
  // Before the lane runs, on the caller's thread: up to `count` handles
  // created into the pool at once (D-124's dedicated pool), as many as the
  // provider gives. A refusal ends it, keeps what was made and lowers the
  // ceiling to it (`failure` says why): the node starts with a smaller
  // pool and its loads create their own past it, never a refused start.
  std::size_t Preallocate(std::size_t count, std::string* failure = nullptr);
  // The pool is the scheduler's from now on (Rebalance), granted `grant`
  // creates until its first turn.
  void ManageReserve(std::size_t grant) { reserve_.Manage(grant); }
  // The scheduler's thread, each turn: the budget less the domain's
  // occupancy, and the page-ins and evictions in flight
  // (HandleReserve::Rebalance). Wakes the lane if the pool has work for it.
  void Rebalance(Bytes headroom, std::size_t transients);
  // Any thread but the lane's: the pool's ceiling lowered to `ceiling`
  // and the pool drained until RestoreReserve (it creates nothing
  // meanwhile, so it never grows back), and what it held past the
  // ceiling, or a create in flight made, released, waited for (released
  // here while the lane is not running). Returns how many the provider
  // released. Pressure from outside and the release route give
  // the pool's empty handles back first (D-124 as amended).
  std::size_t DrainReserve(std::size_t ceiling);
  void RestoreReserve();
  std::size_t ReserveAvailable() const { return reserve_.size(); }
  std::size_t ReserveTarget() const { return reserve_.target(); }
  // Teardown, once the lane has returned: releases what the pool holds.
  // False if any release failed.
  bool ReleaseReserve();
  BackingCreateStats create_stats() const { return creates_.Snapshot(); }
  BackingCreateCounts create_counts() const { return creates_.Counts(); }

 private:
  void Handle(DeviceCommand& command);
  // Creates one pool backing if the pool may refill: true if it did.
  bool Refill();
  // Releases what the pool shed: true if there was any.
  bool ReleaseShed();
  // A drain's wait looks again (DrainReserve).
  void NotifyDrain();
  // Has the lane look at the pool (a command of no operation, at most one
  // queued at a time; a full queue means the lane is busy and looks anyway).
  void Wake();

  mutable std::mutex mu_;  // worker_running_, and the drain's waits
  std::condition_variable cv_;
  bool worker_running_ = false;
  std::atomic<bool> wake_pending_ = false;
  std::atomic<std::uint64_t> released_ = 0;  // shed handles the provider released
  // The scheduler's thread only: Rebalance's last inputs.
  std::uint64_t last_headroom_ = ~std::uint64_t{0};
  std::size_t last_transients_ = 0;
  std::uint64_t last_version_ = ~std::uint64_t{0};

  providers::DeviceMemory* memory_;
  CompletionBoard& board_;
  base::BoundedQueue<DeviceCommand> queue_;
  std::size_t batch_;
  HandoffStash stash_;  // the lane's thread only
  HandleReserve reserve_;
  BackingCreateCounters creates_;
  bool refill_ok_ = true;     // a create failed: none until the next command
  bool refill_lost_ = false;  // an unknown outcome: never again
};

// The CPU worker lane (a Lane of CpuCommand, at most four workers per
// queue, RE-017): accepts each job, runs it and publishes its result, with
// the proof that it touches its memory no more. Nothing preempts a job: it
// must return, since until it does its operation keeps its lease and the
// task its hold, and shutdown and the lane's destruction wait for it.
Lane<CpuCommand>::Handler CpuHandler(CompletionBoard& board);

}  // namespace llmp::scheduler

#endif  // LLMP_SCHEDULER_SERVICES_H_
