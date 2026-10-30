// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/services.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "base/bounded_queue.h"
#include "base/check.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"

namespace llmp::scheduler {
namespace {

// The completion lane's first back-off once a fence is later than
// expected; it doubles up to DeviceSettings::backstop.
constexpr std::chrono::microseconds kFirstBackoff{50};
// The longest a device lane sleeps with nothing to do: a timer tick, never
// needed for correctness (every command queued and fence handed over
// signals the lane that takes it).
constexpr std::chrono::milliseconds kTick{100};

Outcome OutcomeOf(providers::ReadOutcome outcome) {
  switch (outcome) {
    case providers::ReadOutcome::kComplete:
      return Outcome::kSucceeded;
    case providers::ReadOutcome::kCancelled:
      return Outcome::kCancelled;
    case providers::ReadOutcome::kEndOfFile:
    case providers::ReadOutcome::kFailed:
      return Outcome::kFailed;
  }
  return Outcome::kFailed;
}

// Run even virtual-only provider access on its owning lane: a background
// physical-handle refill can otherwise race Reserve, Free, or RangeOf.
// CarryRegion shares HandleReserve and BackingCreateCounters with CarryOut
// so carried/workspace extents recycle pre-warmed physical handles (RE-059).
std::expected<void, providers::Failure> UndoCarry(providers::DeviceMemory* memory,
                                                  const RegionWork& work, RegionResult& result,
                                                  std::uint64_t extent,
                                                  HandleReserve* reserve = nullptr) {
  while (result.mapped != 0) {
    const auto at = work.offset.value() + (result.mapped - 1) * extent;
    if (auto unmapped = memory->Unmap(work.reservation, Bytes(at), Bytes(extent)); !unmapped)
      return std::unexpected(unmapped.error());
    --result.mapped;
  }
  while (!result.backings.empty()) {
    const auto backing = result.backings.back();
    if (reserve != nullptr &&
        reserve->Keep(work.allocation_class, Bytes(extent), backing, result.pool_transfer.get())) {
      result.backings.pop_back();
      continue;
    }
    if (auto released = memory->Release(backing); !released)
      return std::unexpected(released.error());
    result.backings.pop_back();
  }
  return {};
}

void CarryRegion(providers::DeviceMemory* memory, CompletionBoard& board, OperationId operation,
                 const RegionWork& work, HandleReserve* reserve = nullptr,
                 BackingCreateCounters* creates = nullptr) {
  if (memory == nullptr || !work.result) {
    (void)board.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  auto& result = *work.result;
  result.outcome = RegionResult::Outcome::kUntouched;
  result.failure.reset();
  const auto fail = [&](providers::Failure failure) {
    const bool unknown = failure.error == providers::ProviderError::kUnknown ||
                         failure.error == providers::ProviderError::kUndetermined;
    result.failure = std::move(failure);
    (void)board.Accept(operation, unknown ? Acceptance::kUnknown : Acceptance::kNotStarted);
    if (unknown) {
      result.outcome = RegionResult::Outcome::kUnknown;
      (void)board.Complete(
          operation, Terminal{.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false});
    }
  };
  if (work.kind == RegionWork::Kind::kReserve) {
    auto reserved = memory->Reserve(work.size);
    if (!reserved) {
      fail(reserved.error());
      return;
    }
    result.reservation = *reserved;
    const auto range = memory->RangeOf(*reserved);
    if (!range) {
      // A successful reservation must remain owned even if its metadata is
      // unavailable. No retry or false known-refusal report can drop it.
      result.failure = range.error();
      result.outcome = RegionResult::Outcome::kUnknown;
      (void)board.Accept(operation, Acceptance::kAccepted);
      (void)board.Complete(
          operation, Terminal{.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false});
      return;
    }
    result.base = range->base;
  } else if (work.kind == RegionWork::Kind::kFree) {
    const auto freed = memory->Free(work.reservation);
    if (!freed) {
      fail(freed.error());
      return;
    }
  } else {
    const auto granularity = memory->Granularity().value();
    const auto extent = work.extent_size.value();
    const auto range = memory->RangeOf(work.reservation);
    if (!range || granularity == 0 || extent == 0 || extent % granularity != 0 ||
        work.size.value() % extent != 0 || work.offset.value() % extent != 0 ||
        work.offset.value() > range->size.value() ||
        work.size.value() > range->size.value() - work.offset.value()) {
      fail(range ? providers::Failure{.error = providers::ProviderError::kInvalid,
                                      .detail = "owned region geometry"}
                 : range.error());
      return;
    }
    const auto count = work.size.value() / extent;
    const bool growing = work.kind == RegionWork::Kind::kGrow;
    if (result.pool_transfer &&
        (reserve == nullptr || !reserve->Holds(work.allocation_class, Bytes(extent)) ||
         !reserve->OwnsTransfer(*result.pool_transfer))) {
      fail({.error = providers::ProviderError::kInvalid,
            .detail = "owned region has a foreign or settled pool transfer"});
      return;
    }
    if ((growing && (!result.backings.empty() || result.backings.capacity() < count)) ||
        (!growing && (result.backings.size() != count || result.mapped != count))) {
      fail({.error = providers::ProviderError::kInvalid,
            .detail = "owned region transition was not prepaid"});
      return;
    }
    // Unknown provider state poisons the shared reservation too. Retain the
    // complete transition and stop the node; neither its old prefix nor another
    // borrower's address may be advertised usable in that case.
    const auto unproven = [&](providers::Failure failure) {
      result.failure = std::move(failure);
      result.outcome = RegionResult::Outcome::kUnknown;
      (void)board.Accept(operation, Acceptance::kAccepted);
      (void)board.Complete(
          operation, Terminal{.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false});
    };
    const auto undo = [&]() -> std::expected<void, providers::Failure> {
      return UndoCarry(memory, work, result, extent, reserve);
    };
    if (growing) {
      std::optional<providers::Failure> failure;
      for (std::uint64_t i = 0; i < count; ++i) {
        std::optional<providers::BackingId> backing;
        if (reserve != nullptr && reserve->Holds(work.allocation_class, Bytes(extent))) {
          backing = reserve->Take(work.allocation_class, Bytes(extent), result.pool_transfer.get());
        }
        if (!backing) {
          if (creates != nullptr) creates->Attempt(false);
          auto created = memory->Create(work.allocation_class, Bytes(extent));
          if (creates != nullptr) creates->Completed(false);
          // A failed create still invalidates the refill grant, but did not
          // consume any of the region's admitted physical transfer.
          if (reserve != nullptr) reserve->Created(created ? result.pool_transfer.get() : nullptr);
          if (!created) {
            if (creates != nullptr) creates->Failed(false, created.error());
            failure = created.error();
            break;
          }
          backing = *created;
        }
        result.backings.push_back(*backing);
        const auto at = Bytes(work.offset.value() + i * extent);
        auto mapped = memory->Map(work.reservation, at, *backing);
        if (!mapped) {
          failure = mapped.error();
          break;
        }
        ++result.mapped;
        auto access =
            memory->SetAccess(work.reservation, at, Bytes(extent), providers::Access::kReadWrite);
        if (!access) {
          failure = access.error();
          break;
        }
      }
      if (failure) {
        if (failure->error == providers::ProviderError::kUnknown ||
            failure->error == providers::ProviderError::kUndetermined) {
          unproven(std::move(*failure));
        } else if (auto undone = undo(); !undone) {
          unproven(undone.error());
        } else {
          result.outcome = RegionResult::Outcome::kRolledBack;
          fail(std::move(*failure));  // known refusal: old prefix is intact
        }
        return;
      }
    } else if (auto retired = undo(); !retired) {
      // Partial retirement is terminal; retain handles and catalog payment.
      // Never free the reservation or publish reclaimed capacity on refusal.
      unproven(retired.error());
      return;
    }
  }
  result.outcome = RegionResult::Outcome::kComplete;
  (void)board.Accept(operation, Acceptance::kAccepted);
  (void)board.Complete(operation, Terminal{.outcome = Outcome::kSucceeded,
                                           .bytes = work.size.value(),
                                           .no_further_access = true});
}

// Carries out VMM work and publishes its acceptance and result at once
// (DeviceService and BackingService alike), keeping handed-off backing in
// the lane's `stash`.
void CarryOut(providers::DeviceMemory* memory, HandoffStash& stash, HandleReserve* reserve,
              CompletionBoard& board, OperationId operation, const BackingWork& work,
              BackingCreateCounters* creates = nullptr) {
  if (memory == nullptr) {
    (void)board.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  const auto not_started = [&] { (void)board.Accept(operation, Acceptance::kNotStarted); };
  // The provider's state is not what either outcome needs: the backing,
  // or the place, is left in a state nobody may reuse.
  const auto unproven = [&](Acceptance acceptance) {
    (void)board.Accept(operation, acceptance);
    (void)board.Complete(
        operation, Terminal{.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false});
  };
  const auto unknown = [](const providers::Failure& failure) {
    return failure.error == providers::ProviderError::kUnknown;
  };
  // A kept backing of the work's class and size, unmapped first from where
  // it was evicted if it was kept there (lazy). The unmap touches only the
  // outgoing place, never the work's: one it refuses stays kept, now the
  // oldest, and the next is tried; one with an unknown outcome is dropped,
  // never reused and still charged (the last release the scheduler asks
  // for finds none kept and is refused).
  const auto take_unmapped = [&]() -> std::optional<providers::BackingId> {
    std::optional<providers::BackingId> first_refused;
    while (true) {
      const std::optional<HandoffStash::Kept> kept = stash.Take(work.allocation_class, work.size);
      if (!kept) {
        return std::nullopt;
      }
      if (kept->backing == first_refused) {
        // Every other was tried: refused ones only remain.
        stash.Put(work.allocation_class, work.size, *kept, /*oldest=*/true);
        return std::nullopt;
      }
      if (!kept->at) {
        return kept->backing;
      }
      const auto unmapped = memory->Unmap(kept->at->reservation, kept->at->offset, work.size);
      if (unmapped) {
        return kept->backing;
      }
      if (!unknown(unmapped.error())) {
        stash.Put(work.allocation_class, work.size, *kept, /*oldest=*/true);
        first_refused = first_refused.value_or(kept->backing);
      }
    }
  };
  // Backing no extent holds any more: kept by a short reserve, or released.
  const auto release =
      [&](providers::BackingId backing) -> std::expected<void, providers::Failure> {
    if (reserve != nullptr && reserve->Keep(work.allocation_class, work.size, backing)) {
      return {};
    }
    return memory->Release(backing);
  };
  if (work.kind == BackingWork::Kind::kRelease) {
    // One handed-off backing of the class and size, released: no longer
    // charged once this succeeds.
    const std::optional<providers::BackingId> kept = take_unmapped();
    if (!kept) {
      not_started();  // none kept that could be: nothing changed
      return;
    }
    if (const auto released = release(*kept); !released) {
      if (unknown(released.error())) {
        unproven(Acceptance::kUnknown);
      } else {
        // Still there, still kept.
        stash.Put(work.allocation_class, work.size, {.backing = *kept, .at = std::nullopt},
                  /*oldest=*/true);
        not_started();
      }
      return;
    }
  } else if (work.kind == BackingWork::Kind::kMap) {
    // Handed-off backing goes back to the stash on a known failure, so a
    // refusal still changed nothing: the scheduler releases it later.
    const HandoffStash::Place place{.reservation = work.reservation, .offset = work.offset};
    // A lazily kept backing may still hold this place (its extent's
    // eviction parked it here, or ended while another's was taken).
    std::optional<HandoffStash::Kept> here = stash.TakeAt(place, work.allocation_class, work.size);
    if (here && work.reuse) {
      // Its own backing, still mapped here with its access: nothing to do,
      // unless an unknown outcome has since left the reservation
      // undetermined, where a map would be refused (kUndetermined) and the
      // extent must not be resident again: still kept, nothing changed.
      if (const auto range = memory->RangeOf(place.reservation); !range) {
        stash.Put(work.allocation_class, work.size, *here, /*oldest=*/true);
        not_started();
        return;
      }
      (void)board.Accept(operation, Acceptance::kAccepted);
      (void)board.Complete(operation, Terminal{.outcome = Outcome::kSucceeded,
                                               .bytes = work.size.value(),
                                               .no_further_access = true});
      return;
    }
    if (here) {
      // A created backing maps here: the kept one leaves the place first
      // and stays kept, unmapped. Refused (as a map here would be): still
      // kept there, nothing changed.
      if (const auto unmapped = memory->Unmap(place.reservation, place.offset, work.size);
          !unmapped) {
        if (unknown(unmapped.error())) {
          unproven(Acceptance::kUnknown);
        } else {
          stash.Put(work.allocation_class, work.size, *here, /*oldest=*/true);
          not_started();
        }
        return;
      }
      here->at.reset();
      stash.Put(work.allocation_class, work.size, *here);
    }
    std::optional<providers::BackingId> backing;
    if (work.reuse) {
      backing = take_unmapped();
      if (!backing) {
        not_started();  // none kept that could be: nothing changed
        return;
      }
    } else if (reserve != nullptr && reserve->Holds(work.allocation_class, work.size)) {
      backing = reserve->Take(work.allocation_class, work.size);
    }
    if (!backing) {
      if (creates != nullptr) creates->Attempt(false);
      const auto created = memory->Create(work.allocation_class, work.size);
      if (creates != nullptr) creates->Completed(false);
      if (reserve != nullptr) reserve->Created();
      if (!created) {
        if (creates != nullptr) creates->Failed(false, created.error());
        unknown(created.error()) ? unproven(Acceptance::kUnknown) : not_started();
        return;
      }
      backing = *created;
    }
    const auto undo = [&]() -> bool {
      if (work.reuse) {
        stash.Put(work.allocation_class, work.size, {.backing = *backing, .at = std::nullopt});
        return true;
      }
      return release(*backing).has_value();
    };
    const auto mapped = memory->Map(work.reservation, work.offset, *backing);
    if (!mapped) {
      if (unknown(mapped.error())) {
        unproven(Acceptance::kUnknown);
      } else {
        undo() ? not_started() : unproven(Acceptance::kAccepted);
      }
      return;
    }
    const auto access =
        memory->SetAccess(work.reservation, work.offset, work.size, providers::Access::kReadWrite);
    if (!access) {
      if (unknown(access.error())) {
        unproven(Acceptance::kUnknown);
      } else if (memory->Unmap(work.reservation, work.offset, work.size) && undo()) {
        not_started();
      } else {
        unproven(Acceptance::kAccepted);
      }
      return;
    }
  } else {
    const std::optional<providers::BackingId> backing =
        memory->MappedAt(work.reservation, work.offset);
    if (!backing) {
      not_started();  // nothing is mapped there: nothing changed
      return;
    }
    if (work.retain && work.lazy) {
      // Kept where it is: the load or release that takes it unmaps it. A
      // reservation an earlier unknown outcome left undetermined would
      // refuse the unmap (below): the extent is quarantined now, as then,
      // not whatever load would take its backing later.
      if (const auto range = memory->RangeOf(work.reservation); !range) {
        range.error().error == providers::ProviderError::kUndetermined
            ? unproven(Acceptance::kAccepted)
            : not_started();
        return;
      }
      if (creates != nullptr) creates->ParkMetadataDone();
      stash.Put(
          work.allocation_class, work.size,
          {.backing = *backing,
           .at = HandoffStash::Place{.reservation = work.reservation, .offset = work.offset}});
      if (creates != nullptr) creates->ParkStashDone();
      (void)board.Accept(operation, Acceptance::kAccepted);
      (void)board.Complete(operation, Terminal{.outcome = Outcome::kSucceeded,
                                               .bytes = work.size.value(),
                                               .no_further_access = true});
      return;
    }
    const auto unmapped = memory->Unmap(work.reservation, work.offset, work.size);
    if (!unmapped) {
      if (unknown(unmapped.error())) {
        unproven(Acceptance::kUnknown);
      } else if (unmapped.error().error == providers::ProviderError::kUndetermined) {
        // Refused because an earlier unknown outcome left the place
        // undetermined: nothing changed, but nothing about the place is
        // proven either, so it must never be handed back as resident.
        unproven(Acceptance::kAccepted);
      } else {
        not_started();
      }
      return;
    }
    if (work.retain) {
      // Kept for a handoff: the scheduler keeps it charged until a load
      // maps it or a kRelease releases it.
      stash.Put(work.allocation_class, work.size, {.backing = *backing, .at = std::nullopt});
    } else if (const auto released = release(*backing); !released) {
      // Unmapped, but the backing still exists until it is released: a
      // refusal here leaves it charged.
      unproven(unknown(released.error()) ? Acceptance::kUnknown : Acceptance::kAccepted);
      return;
    }
  }
  (void)board.Accept(operation, Acceptance::kAccepted);
  (void)board.Complete(operation, Terminal{.outcome = Outcome::kSucceeded,
                                           .bytes = work.size.value(),
                                           .no_further_access = true});
}

}  // namespace

void HandoffStash::Put(std::size_t allocation_class, Bytes size, Kept kept, bool oldest) {
  // The provider maps one backing at a place at a time: a second entry
  // there would leave the first's index stale.
  base::Check(!kept.at || !mapped_.contains(*kept.at), "two kept backings at one place");
  const Entry entry{.allocation_class = allocation_class, .size = size, .kept = kept};
  const auto it =
      oldest ? entries_.insert(entries_.begin(), entry) : entries_.insert(entries_.end(), entry);
  if (kept.at) {
    mapped_[*kept.at] = it;
  }
}

namespace {

std::uint64_t TimingNowNs() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

void CompleteTiming(BackingTimingTotals& totals, const BackingTimingEvent& event) {
  ++totals.completed;
  totals.total_ns += event.end_ns - event.start_ns;
  if (totals.first.serial == 0) totals.first = event;
  totals.last = event;
  if (event.end_ns - event.start_ns > totals.longest.end_ns - totals.longest.start_ns)
    totals.longest = event;
}

void CompleteParkStage(BackingParkStageTotals& totals, std::uint64_t serial, std::uint64_t started,
                       std::uint64_t ended) {
  ++totals.completed;
  totals.total_ns += ended - started;
  if (totals.longest.serial == 0 ||
      ended - started > totals.longest.end_ns - totals.longest.start_ns)
    totals.longest = {.serial = serial, .start_ns = started, .end_ns = ended};
}

}  // namespace

void BackingCreateCounters::Attempt(bool reserve) {
  if (!timing_enabled_) {
    (reserve ? reserve_attempts_ : ordinary_attempts_).fetch_add(1, std::memory_order_relaxed);
    return;
  }
  const auto started = TimingNowNs();
  std::lock_guard lock(mutex_);
  (reserve ? reserve_attempts_ : ordinary_attempts_).fetch_add(1, std::memory_order_relaxed);
  auto& timing = failures_.timing;
  base::Check(timing.current_create.serial == 0, "overlapping backing Create diagnostics");
  timing.current_create_reserve = reserve;
  timing.current_create = {.serial = ++timing.create_serial, .start_ns = started};
  ++(reserve ? timing.reserve : timing.ordinary).started;
}

void BackingCreateCounters::Completed(bool reserve) {
  if (!timing_enabled_) return;
  const auto ended = TimingNowNs();
  std::lock_guard lock(mutex_);
  auto& timing = failures_.timing;
  base::Check(timing.current_create.serial != 0 && timing.current_create_reserve == reserve,
              "completing a different backing Create diagnostic");
  timing.current_create.end_ns = ended;
  CompleteTiming(reserve ? timing.reserve : timing.ordinary, timing.current_create);
  timing.current_create = {};
}

void BackingCreateCounters::BeginPark() {
  if (!timing_enabled_) return;
  const auto started = TimingNowNs();
  std::lock_guard lock(mutex_);
  auto& timing = failures_.timing;
  base::Check(timing.current_park.serial == 0, "overlapping lazy park diagnostics");
  timing.current_park = {.serial = ++timing.park.started, .start_ns = started};
  park_metadata_end_ns_ = 0;
  park_stash_end_ns_ = 0;
}

void BackingCreateCounters::ParkMetadataDone() {
  if (!timing_enabled_) return;
  const auto ended = TimingNowNs();
  std::lock_guard lock(mutex_);
  base::Check(failures_.timing.current_park.serial != 0 && park_metadata_end_ns_ == 0,
              "completing absent or repeated lazy park metadata");
  park_metadata_end_ns_ = ended;
}

void BackingCreateCounters::ParkStashDone() {
  if (!timing_enabled_) return;
  const auto ended = TimingNowNs();
  std::lock_guard lock(mutex_);
  base::Check(park_metadata_end_ns_ != 0 && park_stash_end_ns_ == 0,
              "completing absent or repeated lazy park stash insertion");
  park_stash_end_ns_ = ended;
}

void BackingCreateCounters::EndPark() {
  if (!timing_enabled_) return;
  const auto ended = TimingNowNs();
  std::lock_guard lock(mutex_);
  auto& timing = failures_.timing;
  base::Check(timing.current_park.serial != 0, "completing an absent lazy park diagnostic");
  timing.current_park.end_ns = ended;
  if (park_metadata_end_ns_ != 0 && park_stash_end_ns_ != 0) {
    const auto& event = timing.current_park;
    CompleteParkStage(timing.park_metadata, event.serial, event.start_ns, park_metadata_end_ns_);
    CompleteParkStage(timing.park_stash, event.serial, park_metadata_end_ns_, park_stash_end_ns_);
    CompleteParkStage(timing.park_publication, event.serial, park_stash_end_ns_, event.end_ns);
  }
  CompleteTiming(timing.park, timing.current_park);
  timing.current_park = {};
  park_metadata_end_ns_ = 0;
  park_stash_end_ns_ = 0;
}

void BackingCreateCounters::Failed(bool reserve, const providers::Failure& failure) {
  std::lock_guard lock(mutex_);
  ++(reserve ? failures_.reserve_failures : failures_.ordinary_failures);
  failures_.last_failure_monotonic_ns =
      static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
  failures_.last_failure_reserve = reserve;
  failures_.last_failure_error = failure.error;
  failures_.last_failure_detail = failure.detail;
}

BackingCreateStats BackingCreateCounters::Snapshot() const {
  std::lock_guard lock(mutex_);
  BackingCreateStats result = failures_;
  result.startup_pool_creates = startup_pool_creates_.load(std::memory_order_relaxed);
  result.reserve_attempts = reserve_attempts_.load(std::memory_order_relaxed);
  result.ordinary_attempts = ordinary_attempts_.load(std::memory_order_relaxed);
  result.timing.enabled = timing_enabled_;
  return result;
}

BackingCreateCounts BackingCreateCounters::Counts() const {
  std::lock_guard lock(mutex_);
  BackingCreateCounts result = failures_;
  result.startup_pool_creates = startup_pool_creates_.load(std::memory_order_relaxed);
  result.reserve_attempts = reserve_attempts_.load(std::memory_order_relaxed);
  result.ordinary_attempts = ordinary_attempts_.load(std::memory_order_relaxed);
  result.timing.enabled = timing_enabled_;
  return result;
}

RegionPoolTransfer::RegionPoolTransfer(std::shared_ptr<State> state, std::size_t count,
                                       bool growing)
    : state_(std::move(state)), count_(count), pending_(growing ? count : 0) {
  // HandleReserve holds the shared mutex while constructing the token.
  base::Check(pending_ <= SIZE_MAX - state_->pending, "region pool overlap fits");
  state_->pending += pending_;
  ++state_->version;
}

RegionPoolTransfer::~RegionPoolTransfer() {
  std::lock_guard lock(state_->mutex);
  base::Check(settled_, "region pool transfer settles with its catalog transition");
}

void RegionPoolTransfer::ConsumeLocked() {
  base::Check(!settled_ && pending_ != 0 && state_->pending != 0,
              "region backing consumes one pending pool transfer");
  --pending_;
  --state_->pending;
  ++state_->version;
}

void RegionPoolTransfer::ReturnLocked() {
  base::Check(!settled_ && pending_ < count_ && state_->pending != SIZE_MAX,
              "region backing returns within its admitted transfer");
  ++pending_;
  ++state_->pending;
  ++state_->version;
}

void RegionPoolTransfer::Settle() {
  std::lock_guard lock(state_->mutex);
  if (settled_) return;
  base::Check(state_->pending >= pending_, "settled pool overlap remains owned");
  state_->pending -= pending_;
  pending_ = 0;
  settled_ = true;
  ++state_->version;
}

std::size_t RegionPoolTransfer::pending() const {
  std::lock_guard lock(state_->mutex);
  return pending_;
}

std::shared_ptr<RegionPoolTransfer> HandleReserve::BeginTransfer(std::size_t allocation_class,
                                                                 Bytes size, std::size_t count,
                                                                 bool growing) {
  static_assert(sizeof(RegionPoolTransfer) + sizeof(RegionPoolTransfer::State) + 4 * 64 <=
                RegionPoolTransfer::kMetadataBytes);
  std::lock_guard lock(mutex_);
  if (!managed_ || settings_.count == 0 || allocation_class != settings_.allocation_class ||
      size != settings_.size || count == 0)
    return {};
  return std::shared_ptr<RegionPoolTransfer>(new RegionPoolTransfer(transfers_, count, growing));
}

bool HandleReserve::OwnsTransfer(const RegionPoolTransfer& transfer) const {
  std::lock_guard lock(mutex_);
  return transfer.state_ == transfers_ && !transfer.settled_;
}

std::optional<providers::BackingId> HandleReserve::Take(std::size_t allocation_class, Bytes size,
                                                        RegionPoolTransfer* transfer) {
  std::lock_guard lock(mutex_);
  if (settings_.count == 0 || allocation_class != settings_.allocation_class ||
      size != settings_.size || kept_.empty()) {
    return std::nullopt;
  }
  const providers::BackingId backing = kept_.back();
  kept_.pop_back();
  if (transfer != nullptr) transfer->ConsumeLocked();
  ++version_;
  return backing;
}

bool HandleReserve::Keep(std::size_t allocation_class, Bytes size, providers::BackingId backing,
                         RegionPoolTransfer* transfer) {
  std::lock_guard lock(mutex_);
  if (settings_.count == 0 || allocation_class != settings_.allocation_class ||
      size != settings_.size || HeldLocked() >= ceiling_) {
    return false;
  }
  kept_.push_back(backing);
  if (transfer != nullptr) transfer->ReturnLocked();
  ++version_;
  return true;
}

bool HandleReserve::BeginRefill() {
  std::lock_guard lock(mutex_);
  if (settings_.count == 0 || drained_ || RemainingLocked() >= ceiling_ ||
      (managed_ && grant_ == 0)) {
    return false;
  }
  if (managed_) --grant_;
  ++creating_;
  ++version_;
  return true;
}

bool HandleReserve::EndRefill(std::optional<providers::BackingId> created, bool lost,
                              bool known_failure) {
  std::lock_guard lock(mutex_);
  --creating_;
  ++version_;
  if (lost) ++lost_;
  // Never raised: a drain's lowered ceiling stays (a shed the provider
  // refused may have left the pool above it).
  if (known_failure && managed_) ceiling_ = std::min(ceiling_, RemainingLocked());
  if (!created) return false;
  if (drained_ || RemainingLocked() >= ceiling_) {
    // The ceiling fell, or a drain began, meanwhile: shed, not kept, so
    // the lane's release accounts for it (a refusal keeps it, an unknown
    // outcome counts it lost) and a drain waits for it.
    shed_.push_back(*created);
    ++shedding_;
    return false;
  }
  kept_.push_back(*created);
  return true;
}

bool HandleReserve::Rebalance(std::size_t headroom, std::size_t transients, std::uint64_t* seen) {
  std::lock_guard lock(mutex_);
  if (!managed_) {
    if (seen != nullptr) *seen = version_;
    return false;
  }
  // Region transfers count only the backing not yet taken, or already
  // returned but still charged. They protect retention, never refill grants.
  transients =
      transients > SIZE_MAX - transfers_->pending ? SIZE_MAX : transients + transfers_->pending;
  const std::size_t allowed = headroom > SIZE_MAX - transients ? SIZE_MAX : headroom + transients;
  // Never past its ceiling either: a drain's shed the provider refused is
  // shed again here, after the lane's next command.
  const std::size_t most = std::min(allowed, ceiling_);
  std::size_t shed = 0;
  if (!shed_refused_ && HeldLocked() > most) {
    shed = std::min(kept_.size(), HeldLocked() - most);
    for (std::size_t i = 0; i < shed; ++i) {
      shed_.push_back(kept_.back());
      kept_.pop_back();
    }
    shedding_ += shed;
  }
  // Everything it holds counts here, the handles on their way out too:
  // what it may create is certainly within the headroom.
  const std::size_t pooled = RemainingLocked() + shedding_;
  const std::size_t room =
      !drained_ && ceiling_ > RemainingLocked() ? ceiling_ - RemainingLocked() : 0;
  const std::size_t grant = std::min(headroom > pooled ? headroom - pooled : 0, room);
  if (shed != 0 || grant != grant_) ++version_;
  grant_ = grant;
  if (seen != nullptr) *seen = version_;
  return shed != 0 || grant_ != 0;
}

std::size_t HandleReserve::SetCeiling(std::size_t ceiling) {
  std::lock_guard lock(mutex_);
  ceiling_ = std::min(ceiling_, ceiling);
  drained_ = true;  // first: a create in flight is shed when it ends
  std::size_t shed = 0;
  if (HeldLocked() > ceiling_) {
    shed = std::min(kept_.size(), HeldLocked() - ceiling_);
    for (std::size_t i = 0; i < shed; ++i) {
      shed_.push_back(kept_.back());
      kept_.pop_back();
    }
    shedding_ += shed;
  }
  grant_ = 0;
  ++version_;
  return shed;
}

std::vector<providers::BackingId> HandleReserve::TakeShed() {
  std::lock_guard lock(mutex_);
  return std::exchange(shed_, {});
}

void HandleReserve::Shed(std::size_t released, std::span<const providers::BackingId> refused,
                         std::size_t lost) {
  std::lock_guard lock(mutex_);
  shedding_ -= std::min(shedding_, released + refused.size() + lost);
  lost_ += lost;  // may still exist: counted, never kept again
  if (!refused.empty()) {
    // Still there, still the pool's: released at teardown, or shed again
    // after the lane's next command.
    kept_.insert(kept_.end(), refused.begin(), refused.end());
    shed_refused_ = true;
  }
  ++version_;
}

std::vector<providers::BackingId> HandleReserve::TakeAll() {
  std::lock_guard lock(mutex_);
  std::vector<providers::BackingId> all = std::exchange(kept_, {});
  all.insert(all.end(), shed_.begin(), shed_.end());
  shedding_ -= std::min(shedding_, shed_.size());
  shed_.clear();
  ++version_;
  return all;
}

std::size_t HandleReserve::target() const {
  std::lock_guard lock(mutex_);
  if (!managed_) return ceiling_;
  const std::size_t remaining = RemainingLocked();
  const std::size_t aim = grant_ > SIZE_MAX - remaining ? SIZE_MAX : remaining + grant_;
  return std::min(ceiling_, aim);
}

std::optional<HandoffStash::Kept> HandoffStash::Take(std::size_t allocation_class, Bytes size) {
  for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
    if (it->allocation_class == allocation_class && it->size == size) {
      const Kept kept = it->kept;
      if (kept.at) {
        mapped_.erase(*kept.at);
      }
      entries_.erase(std::next(it).base());
      return kept;
    }
  }
  return std::nullopt;
}

std::optional<HandoffStash::Kept> HandoffStash::TakeAt(const Place& place,
                                                       std::size_t allocation_class, Bytes size) {
  const auto found = mapped_.find(place);
  if (found == mapped_.end() || found->second->allocation_class != allocation_class ||
      found->second->size != size) {
    return std::nullopt;
  }
  const Kept kept = found->second->kept;
  entries_.erase(found->second);
  mapped_.erase(found);
  return kept;
}

StorageService::StorageService(providers::Storage& storage, providers::ReaderSettings reader,
                               CompletionBoard& board, QueueSettings queue,
                               std::chrono::microseconds poll_window)
    : queue_(queue.capacity, queue.reserved),
      storage_(storage),
      reader_(storage, reader),
      board_(board),
      batch_(queue.batch),
      poll_window_(poll_window) {
  base::Check(batch_ > 0, "a lane turn takes at least one command");
  // Compared in the clock's nanoseconds, where an unbounded value overflows.
  base::Check(
      poll_window_ >= std::chrono::microseconds::zero() && poll_window_ <= std::chrono::hours(1),
      "a storage poll window must be non-negative and at most an hour");
}

void StorageService::Handle(const StorageCommand& command) {
  if (const auto* read = std::get_if<ReadCommand>(&command)) {
    const std::uint64_t key = KeyOf(read->operation);
    // The key is the operation's own, so the reader never joins it to
    // another read; a repeated command is a duplicate observation.
    const auto started = reader_.Read(key, read->spec, key);
    (void)board_.Accept(read->operation, started ? Acceptance::kAccepted : Acceptance::kNotStarted);
    return;
  }
  const std::uint64_t key = KeyOf(std::get<CancelRead>(command).operation);
  // Best effort: a read that already ended has nothing left to cancel.
  std::ignore = reader_.Withdraw(key, key);
  // Nothing may be published for it, but the queue has room: the owner may
  // hold a command or cancellation this queue refused.
  board_.Nudge();
}

bool StorageService::Turn(bool wait) {
  bool progress = false;
  for (std::size_t i = 0; i < batch_; ++i) {
    std::optional<StorageCommand> command = queue_.TryPop();
    if (!command) {
      break;
    }
    Handle(*command);
    progress = true;
  }
  // A read ends only once every request it started has completed, so its
  // result always proves no further access.
  for (const providers::FinishedRead& read : reader_.Poll(wait && !progress && reads() > 0)) {
    (void)board_.Complete(OperationOf(read.key), Terminal{.outcome = OutcomeOf(read.outcome),
                                                          .bytes = read.bytes,
                                                          .no_further_access = true});
    progress = true;
  }
  return progress;
}

void StorageService::Run() {
  const std::stop_token never;
  auto last = std::chrono::steady_clock::now();
  while (true) {
    // Within the window after its last progress the lane polls: it neither
    // waits in the provider nor sleeps on its queue (RE-017).
    const bool polling = std::chrono::steady_clock::now() - last < poll_window_;
    if (reads() == 0 && !polling) {
      std::optional<StorageCommand> command = queue_.Pop(never);
      if (!command) {
        return;  // closed and drained, with nothing in flight
      }
      Handle(*command);
    } else if (reads() == 0 && queue_.drained()) {
      return;
    }
    if (Turn(!polling)) {
      last = std::chrono::steady_clock::now();
    } else {
      std::this_thread::yield();
    }
  }
}

DeviceService::DeviceService(providers::DeviceExecution& execution,
                             std::span<const providers::StreamId> streams, CompletionBoard& board,
                             DeviceSettings settings, providers::DeviceMemory* memory)
    : execution_(execution),
      memory_(memory),
      streams_(streams.begin(), streams.end()),
      board_(board),
      settings_(settings),
      queue_(settings.queue.capacity, settings.queue.reserved),
      handoff_(settings.handoff, 0),
      backoff_(kFirstBackoff) {
  base::Check(settings_.queue.batch > 0 && !streams_.empty() && settings_.refusals > 0,
              "a device service needs streams, a turn of at least one command and a refusal bound");
  // Compared in the clock's nanoseconds, where an unbounded value overflows.
  const auto bounded = [](std::chrono::microseconds d) {
    return d >= std::chrono::microseconds::zero() && d <= std::chrono::hours(1);
  };
  base::Check(bounded(settings_.poll_window) && bounded(settings_.spin_ahead) &&
                  bounded(settings_.spin_past) && bounded(settings_.backstop) &&
                  settings_.backstop > std::chrono::microseconds::zero(),
              "a device service's poll and spin windows must be non-negative and at most an hour, "
              "and its backstop positive");
  watches_.reserve(settings_.handoff);
  releases_.reserve(settings_.handoff);
  expected_.resize(streams_.size());
}

void DeviceService::Launch(DeviceCommand& command) {
  std::visit(
      [&]<typename Work>(Work& work) {
        if constexpr (std::is_same_v<Work, DeviceWork>) {
          Copy(command.operation, work);
        } else if constexpr (std::is_same_v<Work, LaunchWork>) {
          Run(command.operation, work);
        } else if constexpr (std::is_same_v<Work, RegionWork>) {
          CarryRegion(memory_, board_, command.operation, work);
        } else {
          Back(command.operation, work);
        }
      },
      command.work);
}

void DeviceService::Copy(OperationId operation, const DeviceWork& work) {
  if (work.stream >= streams_.size() || work.count == 0 || work.count > kMaxDeviceCopies) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  const providers::StreamId stream = streams_[work.stream];
  bool queued = false;
  bool refused = false;
  bool unknown = false;
  std::uint64_t bytes = 0;
  for (std::size_t i = 0; i < work.count; ++i) {
    const DeviceCopy& copy = work.copies.at(i);
    const auto copied = copy.zero
                            ? execution_.Zero(stream, copy.destination, copy.size)
                            : execution_.Copy(stream, copy.destination, copy.source, copy.size);
    if (copied) {
      queued = true;
      bytes += copy.size.value();
      continue;
    }
    // A known refusal queued nothing; an unknown outcome may have.
    refused = true;
    unknown = copied.error().error == providers::ProviderError::kUnknown;
    break;
  }
  Fence(operation, work.stream, queued, refused, unknown, bytes);
}

void DeviceService::Run(OperationId operation, LaunchWork& work) {
  if (work.stream >= streams_.size() || !work.job) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    return;
  }
  const providers::StreamId stream = streams_[work.stream];
  const auto native = execution_.Submission(stream);
  JobResult result = JobResult::kNotStarted;
  if (native) {
    result = work.job(*native);
  } else if (native.error().error == providers::ProviderError::kUnknown) {
    result = JobResult::kUnknown;
  }
  Fence(operation, work.stream, result == JobResult::kQueued || result == JobResult::kFailed,
        result == JobResult::kFailed || result == JobResult::kUnknown,
        result == JobResult::kUnknown, 0);
}

void DeviceService::Fence(OperationId operation, std::uint32_t stream, bool queued, bool refused,
                          bool unknown, std::uint64_t bytes) {
  // Recorded even when nothing started: the provider counts an attempted
  // copy or launch as queued work, and only a fence seen complete balances
  // it.
  const auto fence = execution_.Record(streams_[stream]);
  const Clock::time_point recorded = Clock::now();
  if (!queued && !unknown) {
    (void)board_.Accept(operation, Acceptance::kNotStarted);
    if (fence) {
      Hand(Watch{.operation = {},
                 .fence = *fence,
                 .outcome = Outcome::kFailed,
                 .bytes = 0,
                 .refusals = 0,
                 .stream = stream,
                 .recorded = recorded,
                 .relayed = {}});
    }
    return;
  }
  (void)board_.Accept(operation, unknown ? Acceptance::kUnknown : Acceptance::kAccepted);
  if (!fence) {
    // Nothing can show when the queued work ends.
    (void)board_.Complete(
        operation,
        Terminal{.outcome = Outcome::kFailed, .bytes = bytes, .no_further_access = false});
    return;
  }
  Hand(Watch{.operation = operation,
             .fence = *fence,
             .outcome = refused ? Outcome::kFailed : Outcome::kSucceeded,
             .bytes = bytes,
             .refusals = 0,
             .stream = stream,
             .recorded = recorded,
             .relayed = {}});
}

void DeviceService::Back(OperationId operation, const BackingWork& work) {
  CarryOut(memory_, stash_, nullptr, board_, operation, work);
}

void DeviceService::Hand(const Watch& watch) {
  base::Check(!unhanded_, "a launch while a fence waits for the completion lane");
  const base::PushResult pushed = handoff_.TryPush(Watch{watch});
  base::Check(pushed != base::PushResult::kClosed, "a fence handed over after the lane finished");
  if (pushed != base::PushResult::kAccepted) {
    unhanded_ = watch;  // the completion lane is full: launch nothing more until it takes it
    return;
  }
  completion_wake_.Signal();  // after the push: the lane looks again once woken
}

bool DeviceService::HandPending() {
  if (!unhanded_) {
    return false;
  }
  const base::PushResult pushed = handoff_.TryPush(Watch{*unhanded_});
  base::Check(pushed != base::PushResult::kClosed, "a fence handed over after the lane finished");
  if (pushed != base::PushResult::kAccepted) {
    return false;
  }
  unhanded_.reset();
  completion_wake_.Signal();
  return true;
}

void DeviceService::Finish() {
  if (!finished_) {
    finished_ = true;
    handoff_.Close();  // the completion lane drains what it has, then returns
    completion_wake_.Signal();
  }
}

bool DeviceService::SubmissionTurn() {
  bool progress = HandPending();
  for (std::size_t i = 0; i < settings_.queue.batch && !unhanded_; ++i) {
    std::optional<DeviceCommand> command = queue_.TryPop();
    if (!command) {
      break;
    }
    const std::scoped_lock lock(submitting_);
    Launch(*command);
    progress = true;
  }
  if (!unhanded_ && queue_.drained()) {
    Finish();
  }
  return progress;
}

void DeviceService::RunSubmission() {
  auto last = Clock::now();
  while (true) {
    if (unhanded_) {
      if (!SubmissionTurn()) {
        std::this_thread::yield();  // the completion lane is making room
      }
      continue;
    }
    // Consumed before looking: a command queued after the look signals
    // again, so the sleep below returns at once.
    (void)submission_wake_.Consume();
    std::optional<DeviceCommand> command = queue_.TryPop();
    if (!command) {
      if (queue_.drained()) {
        break;
      }
      const auto now = Clock::now();
      if (now - last < settings_.poll_window || submission_wake_.Anticipating(now)) {
        std::this_thread::yield();  // more is likely soon: stay awake (RE-017)
      } else {
        (void)submission_wake_.WaitFor(kTick);
      }
      continue;
    }
    {
      const std::scoped_lock lock(submitting_);
      Launch(*command);
    }
    last = Clock::now();
  }
  Finish();
}

bool DeviceService::CompletionTurn() {
  bool progress = false;
  // Bounded: a device that never completes fills this, then the handoff,
  // and then submission stops taking commands.
  while (watches_.size() + releases_.size() < settings_.handoff) {
    std::optional<Watch> watch = handoff_.TryPop();
    if (!watch) {
      break;
    }
    watches_.push_back(*watch);
    progress = true;
  }
  for (auto it = watches_.begin(); it != watches_.end();) {
    const auto state = execution_.Query(it->fence);
    if (state && *state == providers::FenceState::kPending) {
      it->refusals = 0;
      ++it;
      continue;
    }
    if (state) {
      if (it->operation.valid()) {
        (void)board_.Complete(
            it->operation,
            Terminal{.outcome = it->outcome, .bytes = it->bytes, .no_further_access = true});
        // Seen now, or later if the lane slept past it: then an
        // overestimate, a likely end a little past the true one.
        expected_[it->stream].Add(Clock::now() - it->recorded);
      }
      releases_.push_back(Release{.fence = it->fence, .refusals = 0});
    } else if (state.error().error == providers::ProviderError::kUnknown ||
               (state.error().error == providers::ProviderError::kFailed &&
                ++it->refusals >= settings_.refusals)) {
      // A device fault, or a refusal that persists: the work is unproven,
      // and its fence is never touched again.
      if (it->operation.valid()) {
        (void)board_.Complete(
            it->operation,
            Terminal{.outcome = Outcome::kFailed, .bytes = it->bytes, .no_further_access = false});
      }
    } else {
      base::Check(state.error().error == providers::ProviderError::kFailed,
                  "the device refused a fence its completion lane holds");
      ++it;  // a known failure changed nothing: ask again next turn
      continue;
    }
    it = watches_.erase(it);
    progress = true;
  }
  if (!releases_.empty()) {
    // Never wait for a submission call: try again next turn.
    const std::unique_lock lock(submitting_, std::try_to_lock);
    if (lock.owns_lock()) {
      for (auto it = releases_.begin(); it != releases_.end();) {
        const auto released = execution_.Release(it->fence);
        if (!released && released.error().error == providers::ProviderError::kFailed &&
            ++it->refusals < settings_.refusals) {
          ++it;  // changed nothing: try again
          continue;
        }
        base::Check(released || released.error().error == providers::ProviderError::kUnknown ||
                        released.error().error == providers::ProviderError::kFailed,
                    "the device refused to release a fence seen complete");
        // Released, undetermined, or refused too often: never touched again.
        it = releases_.erase(it);
        progress = true;
      }
    }
  }
  return progress;
}

void DeviceService::RunCompletion() {
  while (true) {
    // Consumed before looking: a fence handed over (or the handoff closed)
    // after the look signals again, so a sleep below returns at once.
    (void)completion_wake_.Consume();
    const bool progress = CompletionTurn();
    if (watches_.empty() && releases_.empty() && handoff_.drained()) {
      return;  // the submission lane finished and every fence is settled
    }
    if (progress) {
      backoff_ = kFirstBackoff;
      continue;
    }
    if (watches_.empty() && releases_.empty()) {
      (void)completion_wake_.WaitFor(kTick);  // nothing handed over yet
      continue;
    }
    AwaitCompletion(!releases_.empty());
  }
}

void DeviceService::AwaitCompletion(bool releasing) {
  const auto now = Clock::now();
  Clock::time_point until = now + settings_.backstop;
  bool spin = false;
  bool late = releasing;  // a release waits for the submission lane: back off
  for (Watch& watch : watches_) {
    const base::Expectation& expected = expected_[watch.stream];
    if (!expected.known()) {
      // No history: spin through the first `spin_past`, then back off.
      spin = spin || now < watch.recorded + settings_.spin_past;
      late = late || now >= watch.recorded + settings_.spin_past;
      continue;
    }
    // The next of the stream's recent lengths this fence has not outlasted.
    const std::optional<Clock::duration> next =
        expected.Next(now - watch.recorded, settings_.spin_past);
    if (!next) {
      late = true;  // longer than any of them
      continue;
    }
    const Clock::time_point end = watch.recorded + *next;
    if (now < end - settings_.spin_ahead) {
      until = std::min(until, end - settings_.spin_ahead);  // asleep until the spin
      continue;
    }
    if (watch.relayed < end + settings_.spin_past) {
      // The owner and the submission lane poll through the end too: a
      // step's completion is on its request's critical path, and its next
      // step follows.
      watch.relayed = end + settings_.spin_past;
      board_.Anticipate(watch.relayed);
      submission_wake_.Anticipate(watch.relayed);
    }
    spin = true;
  }
  if (spin) {
    std::this_thread::yield();
    return;
  }
  if (late) {
    until = std::min(until, now + backoff_);
    backoff_ = std::min<Clock::duration>(backoff_ * 2, settings_.backstop);
  }
  (void)completion_wake_.WaitUntil(until);
}

BackingService::BackingService(providers::DeviceMemory* memory, CompletionBoard& board,
                               QueueSettings queue, ReserveSettings reserve, bool timing)
    : memory_(memory),
      board_(board),
      queue_(queue.capacity, queue.reserved),
      batch_(queue.batch),
      reserve_(reserve),
      creates_(timing) {
  base::Check(batch_ > 0, "a lane turn takes at least one command");
}

void BackingService::Handle(DeviceCommand& command) {
  refill_ok_ = true;
  {
    // A new version if it was set: the scheduler's next Rebalance sheds
    // the refused handles again instead of skipping unchanged inputs.
    std::lock_guard lock(reserve_.mutex_);
    if (std::exchange(reserve_.shed_refused_, false)) ++reserve_.version_;
  }
  if (const auto* work = std::get_if<BackingWork>(&command.work)) {
    const bool lazy_park = work->kind == BackingWork::Kind::kUnmap && work->retain && work->lazy;
    if (lazy_park) creates_.BeginPark();
    CarryOut(memory_, stash_, &reserve_, board_, command.operation, *work, &creates_);
    if (lazy_park) creates_.EndPark();
    return;
  }
  if (const auto* work = std::get_if<RegionWork>(&command.work)) {
    CarryRegion(memory_, board_, command.operation, *work, &reserve_, &creates_);
    return;
  }
  (void)board_.Accept(command.operation, Acceptance::kNotStarted);  // not VMM work
}

bool BackingService::Refill() {
  // Closed and drained: the lane is stopping, and what it would create
  // now would only be released.
  if (memory_ == nullptr || !refill_ok_ || refill_lost_ || queue_.drained() ||
      !reserve_.BeginRefill()) {
    return false;
  }
  const ReserveSettings settings = reserve_.settings();
  creates_.Attempt(true);
  const auto created = memory_->Create(settings.allocation_class, settings.size);
  creates_.Completed(true);
  if (!created) {
    creates_.Failed(true, created.error());
    // Known: tried again after the next command (unmanaged), or never past
    // what the pool holds (managed: the device is short). Unknown: the
    // provider's state is not what the pool's count says, so it is left
    // short, the backing that may exist counted in its place.
    refill_ok_ = false;
    refill_lost_ = created.error().error == providers::ProviderError::kUnknown;
    (void)reserve_.EndRefill(std::nullopt, refill_lost_, !refill_lost_);
    NotifyDrain();
    return false;
  }
  // Kept, or shed (a drain began meanwhile) for the next ReleaseShed:
  // progress either way, so the lane releases it before it waits.
  (void)reserve_.EndRefill(*created, false, false);
  NotifyDrain();
  return true;
}

bool BackingService::ReleaseShed() {
  std::vector<providers::BackingId> handles = reserve_.TakeShed();
  if (handles.empty()) return false;
  std::size_t released = 0;
  std::size_t lost = 0;
  std::vector<providers::BackingId> refused;
  for (const providers::BackingId backing : handles) {
    if (const auto done = memory_->Release(backing); done) {
      ++released;
    } else if (done.error().error == providers::ProviderError::kUnknown) {
      ++lost;
    } else {
      refused.push_back(backing);
    }
  }
  reserve_.Shed(released, refused, lost);
  released_.fetch_add(released, std::memory_order_acq_rel);
  NotifyDrain();
  return true;
}

void BackingService::NotifyDrain() {
  {
    std::lock_guard lock(mu_);  // a drain's wait sees it, or is not yet waiting
  }
  cv_.notify_all();
}

void BackingService::Wake() {
  if (wake_pending_.exchange(true, std::memory_order_acq_rel)) return;
  if (queue_.TryPush(DeviceCommand{}) != base::PushResult::kAccepted) {
    // Full: the lane is busy and looks at the pool between commands
    // anyway. Closed: it is stopping.
    wake_pending_.store(false, std::memory_order_release);
  }
}

std::size_t BackingService::Preallocate(std::size_t count, std::string* failure) {
  if (memory_ == nullptr || count == 0) return 0;
  const ReserveSettings settings = reserve_.settings();
  std::size_t made = 0;
  for (; made < count; ++made) {
    auto created = memory_->Create(settings.allocation_class, settings.size);
    if (!created) {
      {
        // The ceiling falls to what the pool holds, the backing an unknown
        // outcome may have made counted in it, so nothing it keeps is shed.
        std::lock_guard lock(reserve_.mutex_);
        if (created.error().error == providers::ProviderError::kUnknown) ++reserve_.lost_;
        reserve_.ceiling_ = std::min(reserve_.ceiling_, reserve_.RemainingLocked());
        reserve_.grant_ = 0;
        ++reserve_.version_;
      }
      if (failure != nullptr) *failure = created.error().detail;
      break;
    }
    reserve_.KeepUnchecked(*created);
    creates_.StartupPoolCreated();
  }
  return made;
}

void BackingService::Rebalance(Bytes headroom, std::size_t transients) {
  const Bytes size = reserve_.settings().size;
  if (size.value() == 0) return;
  const std::uint64_t handles = headroom.value() / size.value();
  const std::uint64_t version = reserve_.version();
  if (handles == last_headroom_ && transients == last_transients_ && version == last_version_) {
    return;
  }
  last_headroom_ = handles;
  last_transients_ = transients;
  // The version Rebalance left under the pool's lock, not a later read: a
  // lane's change just after (a refused shed's flag cleared, a create
  // kept) is then seen at the next turn instead of skipped.
  const bool work =
      reserve_.Rebalance(handles > SIZE_MAX ? SIZE_MAX : static_cast<std::size_t>(handles),
                         transients, &last_version_);
  if (work) Wake();
}

std::size_t BackingService::DrainReserve(std::size_t ceiling) {
  if (memory_ == nullptr) return 0;
  const std::uint64_t before = released_.load(std::memory_order_acquire);
  if (reserve_.SetCeiling(ceiling) == 0 && reserve_.settling() == 0) return 0;
  std::unique_lock lock(mu_);
  if (worker_running_) {
    lock.unlock();
    Wake();
    lock.lock();
    // Its sheds released, and a create in flight ended (and shed).
    cv_.wait(lock, [&] { return reserve_.settling() == 0 || !worker_running_; });
  }
  if (!worker_running_) {
    // No lane runs (before Run, inline lanes, after the join): the
    // provider is this thread's to call, and Run starts on no other.
    lock.unlock();
    (void)ReleaseShed();
  }
  return static_cast<std::size_t>(released_.load(std::memory_order_acquire) - before);
}

void BackingService::RestoreReserve() {
  reserve_.RestoreCeiling();  // a new version: the scheduler's next turn looks again
}

bool BackingService::ReleaseReserve() {
  std::lock_guard lock(mu_);
  if (memory_ == nullptr) return true;
  bool released = true;
  for (const providers::BackingId backing : reserve_.TakeAll()) {
    released = memory_->Release(backing).has_value() && released;
  }
  return released;
}

bool BackingService::Turn() {
  bool progress = ReleaseShed();
  for (std::size_t i = 0; i < batch_; ++i) {
    std::optional<DeviceCommand> command = queue_.TryPop();
    if (!command) {
      break;
    }
    if (command->operation == OperationId{}) {
      wake_pending_.store(false, std::memory_order_release);
      continue;
    }
    Handle(*command);
    progress = true;
  }
  progress = ReleaseShed() || progress;
  return progress || Refill();
}

void BackingService::Run() {
  {
    std::lock_guard lock(mu_);
    worker_running_ = true;
  }
  const std::stop_token never;
  while (true) {
    (void)ReleaseShed();
    std::optional<DeviceCommand> command = queue_.TryPop();
    if (!command) {
      if (Refill()) {
        continue;  // one at a time, so a command waits for one create at most
      }
      command = queue_.Pop(never);
      if (!command) {
        break;
      }
    }
    if (command->operation == OperationId{}) {
      wake_pending_.store(false, std::memory_order_release);
      continue;  // a wake: the pool has work (the loop's head)
    }
    Handle(*command);
  }
  (void)ReleaseShed();
  {
    std::lock_guard lock(mu_);
    worker_running_ = false;
  }
  cv_.notify_all();
}

Lane<CpuCommand>::Handler CpuHandler(CompletionBoard& board) {
  return [&board](CpuCommand&& command) {
    (void)board.Accept(command.operation, Acceptance::kAccepted);
    const CpuResult result =
        command.job ? command.job() : CpuResult{.outcome = Outcome::kFailed, .bytes = 0};
    (void)board.Complete(
        command.operation,
        Terminal{.outcome = result.outcome, .bytes = result.bytes, .no_further_access = true});
  };
}

}  // namespace llmp::scheduler
