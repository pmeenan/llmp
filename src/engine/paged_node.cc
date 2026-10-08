// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_node.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <limits>
#include <print>
#include <span>
#include <utility>

#include "base/bounded_queue.h"
#include "engine/support.h"
#include "memory/materialize.h"
#include "memory/reclaim.h"
#include "platform/crash_policy.h"
#include "providers/device_runtime.h"
#include "providers/direct_reader.h"

namespace jitllm::engine {

namespace {

namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using Done = sc::ProgramDone;
using sc::AcquireProgram;
using sc::AcquireReport;
using sc::CallProgram;
using sc::EvictProgram;
using sc::RequestProgram;
using sc::RunProgram;
using sc::SwapProgram;
using sc::SwapReport;
using support::Address;
using support::Error;
using support::Joined;

// A request's driver spins from this long before a step's expected end
// until this long after it (the device lanes' defaults, DeviceSettings).
constexpr auto kSpinAhead = std::chrono::microseconds(1000);
constexpr auto kSpinPast = std::chrono::microseconds(1000);
// A direct step that has outlasted every recent one: how long the driver
// sleeps between looks.
constexpr auto kLateSleep = std::chrono::microseconds(200);
// The longest a direct step's driver sleeps at once: nothing wakes it at
// the step's end, so a likely end predicted too late costs at most this
// (and a wake, RE-017).
constexpr auto kDirectNap = std::chrono::microseconds(1000);
// A fence query or release refused this many times in a row (a pending
// result in between resets the count) is given up:
// its work's end unproven (the device lane's DeviceSettings::refusals).
constexpr std::uint32_t kFenceRefusals = 8;

// Each device-type lane's completion handoff; a lane holds at most twice
// that many fences unreleased (queued for its completion lane, and
// watched there) and one more waiting to be handed over.
constexpr std::size_t kLaneHandoff = 256;
// The provider's events made ahead: every fence the device and copy lanes
// can hold at once, and a few for the node's own fences, so neither lane
// ever makes one as it goes (RE-029: cuEventCreate blocks while another
// thread launches into a full stream).
constexpr std::size_t kEventsAhead = (2 * ((2 * kLaneHandoff) + 1)) + 16;

// A fence after everything queued on the stream, seen complete and
// released; false if it could not be, or its outcome is unknown, or it was
// not seen complete within `quiet` (before the scheduler exists, the fence
// is all there is to watch).
bool Fence(providers::DeviceExecution& execution, providers::StreamId stream,
           std::chrono::milliseconds quiet) {
  const auto fence = execution.Record(stream);
  if (!fence) {
    return false;
  }
  const auto give_up = std::chrono::steady_clock::now() + quiet;
  while (std::chrono::steady_clock::now() < give_up) {
    const auto state = execution.Query(*fence);
    if (!state) {
      return false;  // unknown: never released or retried
    }
    if (*state == providers::FenceState::kComplete) {
      return execution.Release(*fence).has_value();
    }
    std::this_thread::yield();
  }
  return false;
}

// Drives `round` until `done` holds, or until nothing has moved (`progress`
// unchanged) for `quiet`: an inline teardown's wait.
template <typename Done, typename Round, typename Progress>
void DriveUntil(Done done, Round round, Progress progress, std::chrono::milliseconds quiet) {
  std::uint64_t seen = progress();
  auto give_up = std::chrono::steady_clock::now() + quiet;
  while (!done()) {
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint64_t moved = progress(); moved != seen) {
      seen = moved;
      give_up = now + quiet;
    } else if (now > give_up) {
      return;
    }
    round();
  }
}

}  // namespace

WaitVerdict QuietPatience::Check(const WaitState& wait, std::uint64_t /*progress*/) {
  const auto now = std::chrono::steady_clock::now();
  if (!wait.cancelled) {
    return now - wait.progressed >= quiet_ ? WaitVerdict::kCancel : WaitVerdict::kWait;
  }
  return now - std::max(wait.progressed, *wait.cancelled) >= quiet_ ? WaitVerdict::kGiveUp
                                                                    : WaitVerdict::kWait;
}

// One wait of the driver's under its patience.
class PagedNode::Waiter {
 public:
  Waiter(PagedNode& node, std::string_view what)
      : node_(node), patience_(node.patience()), seen_(node.progress()) {
    const auto now = std::chrono::steady_clock::now();
    state_ = {.what = what, .began = now, .progressed = now, .cancelled = std::nullopt};
    next_ = now + kPatienceCheck;
    patience_.Begin();
  }
  Waiter(const Waiter&) = delete;
  Waiter& operator=(const Waiter&) = delete;
  Waiter(Waiter&&) = delete;
  Waiter& operator=(Waiter&&) = delete;
  // A wait that ends is the driver's progress.
  ~Waiter() {
    patience_.End();
    node_.waits_ended_.fetch_add(1, std::memory_order_relaxed);
  }

  // What to do now: kCancel at most once (the caller cancels its request),
  // kGiveUp, or kWait.
  WaitVerdict Poll(std::chrono::steady_clock::time_point now) {
    const std::uint64_t progress = node_.progress();
    if (progress != seen_) {
      seen_ = progress;
      state_.progressed = now;
    }
    if (now < next_) {
      return WaitVerdict::kWait;
    }
    next_ = now + kPatienceCheck;
    const WaitVerdict verdict = patience_.Check(state_, progress);
    if (verdict == WaitVerdict::kCancel) {
      if (state_.cancelled) {
        return WaitVerdict::kWait;
      }
      state_.cancelled = now;
      std::println(stderr, "{}: nothing moved for {:.0f} s: its request is cancelled (a hang)",
                   state_.what, std::chrono::duration<double>(now - state_.progressed).count());
    }
    return verdict;
  }
  bool cancelled() const { return state_.cancelled.has_value(); }

 private:
  PagedNode& node_;
  Patience& patience_;
  std::uint64_t seen_ = 0;
  WaitState state_;
  std::chrono::steady_clock::time_point next_;
};

void PagedNode::GiveUp(std::string_view what) {
  // Returning would leave the program, or a job it queued, pointing into
  // frames that are gone.
  std::println(stderr, "{} did not finish, nor drain once cancelled: aborting", what);
  std::abort();
}

std::uint64_t PagedNode::progress() const {
  return (board_ != nullptr ? board_->publications() : 0) +
         waits_ended_.load(std::memory_order_relaxed);
}

std::size_t CountingStorage::in_flight() const {
  return inner_.in_flight() + held_.size() + cancelled_.size();
}

std::optional<std::chrono::steady_clock::time_point> CountingStorage::oldest_in_flight() const {
  const std::int64_t ns = oldest_ns_.load(std::memory_order_acquire);
  if (ns == 0) {
    return std::nullopt;
  }
  return std::chrono::steady_clock::time_point(
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::nanoseconds(ns)));
}

void CountingStorage::Started(const providers::IoRequest& request, bool submitted) {
  const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  started_.insert_or_assign(request.token, StartedRead{.at = std::max<std::int64_t>(now, 1),
                                                       .kind = request.kind,
                                                       .length = request.length,
                                                       .submitted = submitted});
  if (submitted && request.kind == providers::IoKind::kRead) {
    read_submitted_bytes.fetch_add(request.length, std::memory_order_relaxed);
  }
  Publish();
}

void CountingStorage::Ended(std::span<const providers::IoCompletion> done) {
  for (const providers::IoCompletion& completion : done) {
    const auto found = started_.find(completion.token);
    if (found != started_.end() && found->second.submitted &&
        found->second.kind == providers::IoKind::kRead && completion.result > 0) {
      read_completed_bytes.fetch_add(
          std::min<std::uint64_t>(static_cast<std::uint64_t>(completion.result),
                                  found->second.length),
          std::memory_order_relaxed);
    }
    started_.erase(completion.token);
  }
  Publish();
}

void CountingStorage::Publish() {
  std::int64_t oldest = 0;
  for (const auto& entry : started_) {
    oldest = oldest == 0 ? entry.second.at : std::min(oldest, entry.second.at);
  }
  oldest_ns_.store(oldest, std::memory_order_release);
}

providers::Submission CountingStorage::Submit(const providers::IoRequest& request) {
  if (hold_ != nullptr && request.kind == providers::IoKind::kRead &&
      hold_->load(std::memory_order_acquire)) {
    if (in_flight() >= inner_.depth()) {
      return providers::Submission::kNotStarted;  // as a full ring refuses it
    }
    // The provider keeps what it needs of the segments: a copy.
    held_.push_back(request);
    held_.back().segments = {};
    held_segments_.emplace_back(request.segments.begin(), request.segments.end());
    held.fetch_add(1, std::memory_order_relaxed);
    Started(request, false);
    return providers::Submission::kAccepted;
  }
  PassHeld();
  const auto submitted = inner_.Submit(request);
  if (submitted != providers::Submission::kNotStarted) {
    requests.fetch_add(1, std::memory_order_relaxed);
    pieces.fetch_add(std::max<std::size_t>(request.segments.size(), 1), std::memory_order_relaxed);
    Started(request, true);
  }
  return submitted;
}

void CountingStorage::PassHeld() {
  if (held_.empty() || (hold_ != nullptr && hold_->load(std::memory_order_acquire))) {
    return;
  }
  while (!held_.empty()) {
    providers::IoRequest request = held_.front();
    request.segments = held_segments_.front();
    const auto submitted = inner_.Submit(request);
    if (submitted == providers::Submission::kNotStarted) {
      return;  // the ring is full: the rest go on later
    }
    // Accepted or unknown: its completion comes from the ring either way.
    requests.fetch_add(1, std::memory_order_relaxed);
    pieces.fetch_add(std::max<std::size_t>(request.segments.size(), 1), std::memory_order_relaxed);
    const auto found = started_.find(request.token);
    base::Check(found != started_.end(), "a passed held read was recorded");
    found->second.submitted = true;
    read_submitted_bytes.fetch_add(request.length, std::memory_order_relaxed);
    held_.erase(held_.begin());
    held_segments_.erase(held_segments_.begin());
  }
}

providers::Submission CountingStorage::Cancel(std::uint64_t token) {
  for (std::size_t i = 0; i < held_.size(); ++i) {
    if (held_[i].token == token) {
      if (!cancellable_) {
        // As a drive's: the cancellation is taken, the read goes on.
        return providers::Submission::kAccepted;
      }
      // Never started: it completes as cancelled at the next harvest.
      cancelled_.push_back(token);
      held_.erase(held_.begin() + static_cast<std::ptrdiff_t>(i));
      held_segments_.erase(held_segments_.begin() + static_cast<std::ptrdiff_t>(i));
      return providers::Submission::kAccepted;
    }
  }
  return inner_.Cancel(token);
}

std::size_t CountingStorage::Harvest(std::span<providers::IoCompletion> out, bool wait) {
  PassHeld();
  std::size_t produced = 0;
  while (produced < out.size() && !cancelled_.empty()) {
    out[produced++] = {.token = cancelled_.front(), .result = -ECANCELED};
    cancelled_.erase(cancelled_.begin());
  }
  if (produced == 0 && wait && !held_.empty() && inner_.in_flight() == 0) {
    // Only held reads: nothing will come from the ring, so do not wait in
    // it; the lane comes back for a cancellation or the hold's release.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return 0;
  }
  produced += inner_.Harvest(out.subspan(produced), produced == 0 && wait);
  if (produced != 0) {
    Ended(out.first(produced));
  }
  return produced;
}

PagedNode::~PagedNode() {
  if (torn_down_ || scheduler_ == nullptr) {
    return;
  }
  // TearDown never ran: stop, so no thread outlives the node. Nothing is
  // released: a model's memory may still be in use.
  scheduler_->RequestShutdown();
  if (threads_.empty()) {
    DriveUntil([this] { return scheduler_->Stopped().has_value(); }, [this] { Round(); },
               [this] { return progress(); }, settings_.quiet);
  } else {
    threads_.front().join();
  }
  storage_lane_->Close();
  device_lane_->Close();
  backing_lane_->Close();
  if (copy_lane_ != nullptr) {
    copy_lane_->Close();
  }
  threads_.clear();
}

bool ReleaseMapped(providers::VmmProvider& memory, Mapped& mapped) {
  if (!mapped.reservation.valid()) {
    return true;
  }
  bool released =
      mapped.backings.empty() ||
      memory.Unmap(mapped.reservation, Bytes(0), Bytes(mapped.backings.size() * kPagedExtent))
          .has_value();
  for (const auto backing : mapped.backings) {
    released = memory.Release(backing).has_value() && released;
  }
  if (!released || !memory.Free(mapped.reservation)) {
    return false;
  }
  mapped.backings.clear();
  mapped.reservation = {};
  return true;
}

void PlaceCheck::Check(const sc::Scheduler& scheduler, ExtentId extent,
                       const sc::PageSource& registered) {
  const sc::PageSource* now = scheduler.SourceOf(extent);
  if (now == nullptr || !sc::SamePlace(*now, registered) || !scheduler.PlacePinned(extent)) {
    Missing(std::format("extent {}", extent.index()));
  }
}

void PlaceCheck::Missing(std::string what) {
  if (moved++ == 0) {
    first = std::move(what);
  }
}

bool HavePinned(void*& pointer, std::uint64_t bytes) {
  if (pointer != nullptr) {
    return true;
  }
  auto allocated = providers::AllocatePinned(bytes);
  if (!allocated) {
    return false;
  }
  pointer = *allocated;
  return true;
}

Status PagedNode::Open() {
  auto device = providers::OpenDevice(
      0, {.fences_ahead = kEventsAhead, .fences_kept = std::max<std::size_t>(kEventsAhead, 4096)});
  if (!device) {
    return Error(device.error().detail);
  }
  memory_ = std::move(device->memory);
  execution_ = std::move(device->execution);
  for (std::size_t i = 0; i <= settings_.compute_streams; ++i) {
    auto created = execution_->CreateStream();
    if (!created) {
      return Error("CreateStream failed");
    }
    streams_.push_back(*created);
  }
  // Two timing marks per compute stream (StepTimes), made now: made while
  // another thread launches into a full stream, one would block (RE-029).
  times_.assign(settings_.compute_streams, StepTimes{});
  for (std::size_t i = 0; i < settings_.compute_streams; ++i) {
    auto begin = providers::CreateTimingMark();
    auto end = providers::CreateTimingMark();
    if (!begin || !end) {
      providers::DestroyTimingMark(begin.value_or(providers::TimingMark{}));
      providers::DestroyTimingMark(end.value_or(providers::TimingMark{}));
      return Error("the timing events");
    }
    events_.emplace_back(*begin, *end);
  }
  bool host_found = false;
  for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
    if (memory_->Classes()[i].kind == providers::BackingKind::kDevice) {
      device_class_ = i;
    } else {
      host_class_ = i;
      host_found = true;
    }
  }
  if (!host_found || memory_->Granularity() != Bytes(kPagedExtent)) {
    return Error("this device has no host-NUMA VMM at 2 MiB (D-034, D-081)");
  }
  if (settings_.slot_bytes < kPagedExtent || settings_.slot_bytes % 4096 != 0) {
    return Error("a landing slot is at least 2 MiB, in 4 KiB units");
  }
  auto storage = providers::OpenStorage(kPagedDepth);
  if (!storage) {
    return Error(std::format("the storage ring: {}", storage.error().message()));
  }
  storage_ = std::move(*storage);
  counting_ = std::make_unique<CountingStorage>(*storage_, settings_.hold_reads,
                                                settings_.hold_cancellable);
  domain_ = catalog_.AddDomain("gb10");
  // The zone first: a persistent pool, mapped before anything pages.
  return MapResident(zone_, "the landing zone", settings_.slots * settings_.slot_bytes,
                     providers::BackingKind::kHost, MemoryClass::kStaging,
                     catalog::Recovery::kPinned, kShared);
}

Status PagedNode::MapWorkspace(std::uint64_t activations, std::uint64_t pool) {
  if (auto r =
          MapResident(activations_, "the activations", activations, providers::BackingKind::kDevice,
                      MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared);
      !r) {
    return r;
  }
  return MapResident(pool_, "the GGML pool", pool, providers::BackingKind::kDevice,
                     MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared);
}

Status PagedNode::MapResident(Mapped& mapped, std::string name, std::uint64_t bytes,
                              providers::BackingKind kind, MemoryClass memory_class,
                              catalog::Recovery recovery, int owner) {
  return MapRegion(mapped, std::move(name), bytes, kind, memory_class, recovery, owner, true);
}

Status PagedNode::ReserveState(Mapped& mapped, std::string name, std::uint64_t bytes, int owner) {
  return MapRegion(mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                   MemoryClass::kLiveState, catalog::Recovery::kPreserve, owner, false);
}

Status PagedNode::MapRegion(Mapped& mapped, std::string name, std::uint64_t bytes,
                            providers::BackingKind kind, MemoryClass memory_class,
                            catalog::Recovery recovery, int owner, bool resident) {
  if (bytes > std::numeric_limits<std::uint64_t>::max() - kPagedExtent + 1) {
    return Error("a region's size overflows its extent alignment");
  }
  mapped.name = std::move(name);
  mapped.bytes =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  auto reservation = memory_->Reserve(Bytes(mapped.bytes));
  if (!reservation) {
    return Error(std::format("reserving {}: {}", mapped.name, reservation.error().detail));
  }
  mapped.reservation = *reservation;
  mapped.base = memory_->RangeOf(*reservation).value().base;
  const std::size_t allocation_class =
      kind == providers::BackingKind::kDevice ? device_class_ : host_class_;
  for (std::uint64_t at = 0; at < mapped.bytes; at += kPagedExtent) {
    if (resident) {
      auto backing = memory_->Create(allocation_class, Bytes(kPagedExtent));
      if (!backing) {
        return Error(std::format("backing {}: {}", mapped.name, backing.error().detail));
      }
      mapped.backings.push_back(*backing);
      if (auto map = memory_->Map(*reservation, Bytes(at), *backing); !map) {
        return Error(std::format("mapping {}: {}", mapped.name, map.error().detail));
      }
    }
    auto extent = catalog_.AddExtent({.domain = domain_,
                                      .memory_class = memory_class,
                                      .recovery = recovery,
                                      .size = Bytes(kPagedExtent),
                                      .content = {}},
                                     resident);
    if (!extent) {
      return Error(std::format("cataloging {}", mapped.name));
    }
    mapped.extents.push_back(*extent);
    AddSpan({.base = mapped.base + at,
             .size = kPagedExtent,
             .extent = *extent,
             .memory_class = memory_class,
             .device = kind == providers::BackingKind::kDevice,
             .owner = owner});
  }
  if (resident) {
    if (auto access = memory_->SetAccess(*reservation, Bytes(0), Bytes(mapped.bytes),
                                         providers::Access::kReadWrite);
        !access) {
      return Error(std::format("access to {}: {}", mapped.name, access.error().detail));
    }
  }
  return {};
}

std::expected<void*, std::string> PagedNode::Pinned(std::uint64_t bytes, int owner,
                                                    std::vector<ExtentId>& staging) {
  bytes = std::max<std::uint64_t>(bytes, 256);
  if (scheduler_ != nullptr) {
    // What does not fit has the reclaim order free it first, then is
    // checked again (and again when it is registered).
    for (int attempt = 0;; ++attempt) {
      std::uint64_t shortfall = 0;
      auto checked = Call(
          [&]() -> Status {
            shortfall = Shortfall(bytes);
            return {};
          },
          "checking pinned staging capacity");
      if (!checked) {
        return std::unexpected(checked.error());
      }
      if (shortfall == 0) {
        break;
      }
      // In whole extents (what the plans' charge gives back moves in them),
      // checked again after each reclaim, a few times at most.
      const std::uint64_t ask = (shortfall + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
      if (attempt >= 3 || AskReclaim(ask, ReclaimFor::kStaging) == 0) {
        return Error("pinned staging exceeds the execution budget");
      }
    }
  }
  auto allocated = providers::AllocatePinned(bytes);
  if (!allocated) {
    return Error("pinned memory");
  }
  void* const pointer = *allocated;
  auto register_memory = [&]() -> Status {
    if (scheduler_ != nullptr) {
      const auto occupancy = catalog_.OccupancyOf(domain_).Total().value();
      if (occupancy > budget_.value() || bytes > budget_.value() - occupancy) {
        return Error("pinned staging exceeds the execution budget");
      }
    }
    auto extent = catalog_.AddExtent({.domain = domain_,
                                      .memory_class = MemoryClass::kStaging,
                                      .recovery = catalog::Recovery::kPinned,
                                      .size = Bytes(bytes),
                                      .content = {}},
                                     true);
    if (!extent) {
      return Error("cataloging the staging");
    }
    staging.push_back(*extent);
    AddSpan({.base = Address(pointer),
             .size = bytes,
             .extent = *extent,
             .memory_class = MemoryClass::kStaging,
             .device = false,
             .owner = owner});
    if (scheduler_ != nullptr) {
      SortSpans();
    }
    return {};
  };
  auto registered =
      scheduler_ == nullptr ? register_memory() : Call(register_memory, "pinned staging");
  if (!registered) {
    providers::FreePinned(pointer);
    return std::unexpected(registered.error());
  }
  pinned_.push_back(pointer);
  return pointer;
}

bool PagedNode::Recharge(std::uint64_t total, bool force, std::uint64_t* shortfall) {
  const std::uint64_t past = total > host_floor_ ? total - host_floor_ : 0;
  return ChargeRuntime(host_extent_, host_charged_, past, force, shortfall, "plans and graphs");
}

bool PagedNode::SetTokenCharge(std::uint64_t bytes, std::uint64_t* shortfall) {
  return ChargeRuntime(token_extent_, token_charged_, bytes, bytes <= token_charged_, shortfall,
                       "native token histories");
}

bool PagedNode::SetRequestCharge(std::uint64_t bytes, std::uint64_t* shortfall) {
  return ChargeRuntime(request_extent_, request_charged_, bytes, bytes <= request_charged_,
                       shortfall, "request memory");
}

bool PagedNode::ChargeRuntime(catalog::ExtentId& held, std::uint64_t& charged, std::uint64_t bytes,
                              bool force, std::uint64_t* shortfall, std::string_view what) {
  const std::uint64_t charge = (bytes + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (charge == charged) {
    return true;
  }
  // Before the scheduler runs, or once it stopped, there is nothing to
  // charge against: only the count changes.
  if (scheduler_ == nullptr || torn_down_) {
    charged = charge;
    return true;
  }
  bool fits = true;
  auto changed = Call(
      [&]() -> Status {
        const std::uint64_t occupancy = catalog_.OccupancyOf(domain_).Total().value();
        const std::uint64_t others = occupancy - std::min(occupancy, charged);
        if (charge > charged && !force &&
            (others > budget_.value() || charge > budget_.value() - others)) {
          fits = false;
          if (shortfall != nullptr) {
            *shortfall = others + charge - budget_.value();
          }
          return {};
        }
        if (held.valid()) {
          if (!catalog_.ReleasePinned(held) || !catalog_.RemoveExtent(held)) {
            return Error(std::format("forgetting the {}' charge", what));
          }
          held = {};
        }
        if (charge != 0) {
          auto extent = catalog_.AddExtent({.domain = domain_,
                                            .memory_class = MemoryClass::kRuntime,
                                            .recovery = catalog::Recovery::kPinned,
                                            .size = Bytes(charge),
                                            .content = {}},
                                           true);
          if (!extent) {
            return Error(std::format("charging the {}", what));
          }
          held = *extent;
        }
        return {};
      },
      std::format("charging {}", what));
  if (!changed) {
    // Only the count is kept; the catalog's charge is what it was.
    return force;
  }
  if (fits) {
    charged = charge;
  }
  return fits;
}

bool PagedNode::ChargeHost(std::uint64_t bytes, bool required) {
  if (bytes == 0) {
    return true;
  }
  std::uint64_t shortfall = 0;
  if (Recharge(host_total_ + bytes, false, &shortfall)) {
    host_total_ += bytes;
    return true;
  }
  // What the charge lacks (all of it or nothing), from other plans and
  // graphs only (a graph's only from what costs less to restore than a
  // graph). The charge moves in whole extents past the floor, so the ask is
  // rounded up by one and the room checked again after each reclaim.
  const ReclaimFor what = required ? ReclaimFor::kPlan : ReclaimFor::kGraph;
  for (int attempt = 0; attempt < 3 && shortfall != 0; ++attempt) {
    const std::uint64_t ask =
        ((shortfall + kPagedExtent - 1) / kPagedExtent * kPagedExtent) + kPagedExtent;
    if (AskReclaim(ask, what) == 0) {
      break;
    }
    shortfall = 0;
    if (Recharge(host_total_ + bytes, false, &shortfall)) {
      host_total_ += bytes;
      return true;
    }
  }
  if (!required) {
    return false;
  }
  ++host_overcharges_;
  (void)Recharge(host_total_ + bytes, true, nullptr);
  host_total_ += bytes;
  return true;
}

std::uint64_t PagedNode::AskReclaim(std::uint64_t needed, ReclaimFor what,
                                    std::span<const ExtentId> protect) {
  if (!reclaimer_ || reclaiming_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  const std::uint64_t freed = reclaimer_(needed, what, protect);
  reclaiming_ = false;
  return freed;
}

std::uint64_t PagedNode::Shortfall(std::uint64_t bytes) const {
  const std::uint64_t occupancy = catalog_.OccupancyOf(domain_).Total().value();
  const std::uint64_t budget = budget_.value();
  return occupancy > budget || bytes > budget - occupancy ? occupancy + bytes - budget : 0;
}

Status PagedNode::ReserveStaging(std::uint64_t bytes) {
  if (staging_ != nullptr) {
    return {};
  }
  std::vector<ExtentId> extents;
  auto pinned = Pinned(bytes, kShared, extents);
  if (!pinned) {
    return std::unexpected(pinned.error());
  }
  staging_ = *pinned;
  staging_bytes_ = bytes;
  return {};
}

void* PagedNode::TakeStaging(std::uint64_t bytes) {
  if (staging_ == nullptr || staging_taken_ || bytes > staging_bytes_) {
    return nullptr;
  }
  staging_taken_ = true;
  return staging_;
}

void PagedNode::ReturnStaging(void* pointer, bool proven) {
  if (pointer == nullptr || pointer != staging_) {
    return;
  }
  if (proven) {
    staging_taken_ = false;
    return;
  }
  // A copy may still reach it: kept to the process's end, out of use.
  KeepPinned(staging_);
  staging_ = nullptr;
  staging_bytes_ = 0;
  staging_taken_ = false;
}

void PagedNode::UnchargeHost(std::uint64_t bytes) {
  host_total_ -= std::min(bytes, host_total_);
  (void)Recharge(host_total_, true, nullptr);
}

std::expected<std::uint64_t, std::string> PagedNode::FreeBytes() {
  std::uint64_t free = 0;
  auto read = Call(
      [&]() -> Status {
        const std::uint64_t occupancy = catalog_.OccupancyOf(domain_).Total().value();
        free = occupancy < budget_.value() ? budget_.value() - occupancy : 0;
        return {};
      },
      "reading the free budget");
  if (!read) {
    return std::unexpected(read.error());
  }
  return free;
}

void PagedNode::KeepPinned(void* pointer) {
  if (const auto at = std::ranges::find(pinned_, pointer); at != pinned_.end()) {
    pinned_.erase(at);
    kept_pinned_.push_back(pointer);
  }
}

Status PagedNode::FreePinned(void* pointer) {
  if (pointer == nullptr) {
    return {};
  }
  if (std::ranges::find(pinned_, pointer) == pinned_.end()) {
    return Error("the pinned allocation is not owned or its completion is unproven");
  }
  ExtentId extent;
  auto checked = Call(
      [&]() -> Status {
        const auto span = std::ranges::find(spans_, Address(pointer), &Span::base);
        if (span == spans_.end() || span->device || span->memory_class != MemoryClass::kStaging) {
          return Error("no pinned staging extent at this address");
        }
        if (!catalog_.CanRetirePinned(span->extent)) {
          return Error("pinned staging is still held or referenced");
        }
        extent = span->extent;
        return {};
      },
      "checking retired pinned staging");
  if (!checked) {
    return checked;
  }
  providers::FreePinned(pointer);
  std::erase(pinned_, pointer);
  return Call(
      [&]() -> Status {
        if (!catalog_.ReleasePinned(extent) || !catalog_.RemoveExtent(extent)) {
          return Error("forgetting retired pinned staging");
        }
        EraseSpans([&](const Span& s) { return s.extent == extent; });
        return {};
      },
      "forgetting pinned staging");
}

bool PagedNode::RetireRing(std::unique_ptr<providers::Storage> ring,
                           std::span<void* const> landings) {
  if (ring == nullptr || ring->in_flight() == 0) {
    return false;  // every read was harvested: nothing can land any more
  }
  for (void* const landing : landings) {
    if (const auto it = std::ranges::find(pinned_, landing); it != pinned_.end()) {
      pinned_.erase(it);
      kept_pinned_.push_back(landing);
    }
  }
  // Destroying a ring with reads in flight is fatal (UringStorage), and
  // closing it would not stop them landing: it is never destroyed.
  kept_rings_.push_back(ring.release());
  return true;
}

void PagedNode::SortSpans() { std::ranges::sort(spans_, {}, &Span::base); }

std::uint64_t PagedNode::StateCapacity() const {
  std::uint64_t bytes = 0;
  for (const Span& span : spans_) {
    if (span.device && span.memory_class == MemoryClass::kLiveState) {
      bytes += span.size;
    }
  }
  return bytes;
}

std::optional<MemoryClass> PagedNode::Covered(std::uint64_t address, std::uint64_t bytes, int owner,
                                              bool resident) const {
  if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) {
    return std::nullopt;
  }
  std::optional<MemoryClass> found;
  std::uint64_t at = address;
  const std::uint64_t end = address + bytes;
  auto span = std::ranges::upper_bound(spans_, at, {}, &Span::base);
  if (span == spans_.begin()) {
    return std::nullopt;
  }
  --span;
  while (at < end) {
    if (span == spans_.end() || at < span->base || at >= span->base + span->size || !span->device ||
        (span->owner != owner && span->owner != kShared) ||
        (found && *found != span->memory_class)) {
      return std::nullopt;
    }
    const auto view = catalog_.Describe(span->extent);
    if (!view || (resident && view->state != catalog::ExtentState::kResident)) {
      return std::nullopt;
    }
    found = span->memory_class;
    at = span->base + span->size;
    ++span;
  }
  return found;
}

Status PagedNode::Start(Bytes budget) {
  budget_ = budget;
  board_ = std::make_unique<sc::CompletionBoard>(1024, wake_);
  // One request per 2 MiB chunk (the reader's default), or with coalesce
  // coalesced chunk reads (BP-P1) in spans up to kSpanBytes.
  storage_lane_ = std::make_unique<sc::StorageService>(
      *counting_,
      providers::ReaderSettings{
          .alignment = 4096,
          .request_bytes = static_cast<std::uint32_t>(settings_.slot_bytes),
          .retries = 3,
          .reads = 1024,
          .waiters = 8,
          .span_bytes = settings_.coalesce ? providers::kSpanBytes : providers::kNoCoalescing,
          .span_segments = providers::kMaxSegments},
      *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32});
  sc::DeviceSettings device{.queue = {.capacity = 256, .reserved = 16, .batch = 32},
                            .handoff = kLaneHandoff};
  if (settings_.poll_window) {
    device.poll_window = *settings_.poll_window;  // a diagnostic (NodeSettings)
  }
  if (settings_.spin_ahead) {
    device.spin_ahead = *settings_.spin_ahead;  // a diagnostic (NodeSettings)
  }
  device_lane_ =
      std::make_unique<sc::DeviceService>(*execution_, streams_, *board_, device, nullptr);
  if (settings_.copy_lane) {
    // The zone's copies on their own submission and completion lanes, on
    // the copy stream alone: a model's job launching into a full stream
    // (RE-029) blocks only the device lane.
    copy_lane_ = std::make_unique<sc::DeviceService>(
        *execution_, std::span<const providers::StreamId>(&streams_.back(), 1), *board_,
        sc::DeviceSettings{.queue = {.capacity = 256, .reserved = 16, .batch = 32},
                           .handoff = kLaneHandoff},
        nullptr);
  }
  // Managed backing's VMM work on a lane of its own, so the zone's copies
  // never wait behind it (docs/experiments/pagein-perf/); that lane alone
  // calls the device-memory provider (device_memory.h).
  // Its handle reserve is charged in full before anything pages, as one
  // pinned runtime extent, whatever it holds (services.h HandleReserve).
  sc::ReserveSettings reserve{.allocation_class = device_class_,
                              .size = Bytes(kPagedExtent),
                              .count = memory_ != nullptr ? settings_.handle_reserve : 0};
  if (reserve.count != 0) {
    auto charged = catalog_.AddExtent({.domain = domain_,
                                       .memory_class = MemoryClass::kRuntime,
                                       .recovery = catalog::Recovery::kPinned,
                                       .size = Bytes(reserve.count * kPagedExtent),
                                       .content = {}},
                                      true);
    if (!charged) {
      return Error("charging the handle reserve");
    }
    reserve_extent_ = *charged;
  }
  backing_lane_ = std::make_unique<sc::BackingService>(
      memory_.get(), *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32},
      reserve, settings_.diagnostic_backing_timing);
  sc::LandingZone landing{
      .slots = {},
      .slot_bytes = Bytes(settings_.slot_bytes),
      .stream = copy_lane_ != nullptr ? 0 : static_cast<std::uint32_t>(settings_.compute_streams)};
  for (std::size_t i = 0; i < settings_.slots; ++i) {
    landing.slots.push_back(zone_.base + (i * settings_.slot_bytes));
  }
  sc::SchedulerSettings scheduling{.tasks = 16,
                                   .budget = budget,
                                   .lazy_handoff = settings_.lazy_handoff,
                                   .landing = landing,
                                   .observer = settings_.observer};
  if (settings_.poll_window) {
    scheduling.poll_window = *settings_.poll_window;  // a diagnostic (NodeSettings)
  }
  scheduler_ = std::make_unique<sc::Scheduler>(catalog_, *board_, wake_,
                                               sc::Lanes{.storage = storage_lane_.get(),
                                                         .device = device_lane_.get(),
                                                         .cpu = nullptr,
                                                         .backing = backing_lane_.get(),
                                                         .copy = copy_lane_.get()},
                                               scheduling);
  return {};
}

void PagedNode::Run() {
  SortSpans();
  if (settings_.inline_lanes) {
    return;
  }
  // Each lane's thread first gets its own alternate signal stack, so the
  // crash policy's handler runs even on a stack overflow (D-074); a harness
  // that installed no policy loses nothing by it.
  const auto thread = [](auto body) {
    return [body] {
      if (auto stack = platform::InstallThreadSignalStack(); !stack) {
        std::println(stderr, "a lane's thread runs without a signal stack: {}", stack.error());
      }
      body();
    };
  };
  threads_.emplace_back(thread([this] { stopped_ = scheduler_->Run(); }));
  threads_.emplace_back(thread([this] { storage_lane_->Run(); }));
  threads_.emplace_back(thread([this] { device_lane_->RunSubmission(); }));
  threads_.emplace_back(thread([this] { device_lane_->RunCompletion(); }));
  threads_.emplace_back(thread([this] { backing_lane_->Run(); }));
  if (copy_lane_ != nullptr) {
    threads_.emplace_back(thread([this] { copy_lane_->RunSubmission(); }));
    threads_.emplace_back(thread([this] { copy_lane_->RunCompletion(); }));
  }
}

void PagedNode::Round() {
  (void)storage_lane_->Turn(false);
  (void)backing_lane_->Turn();
  (void)device_lane_->SubmissionTurn();
  (void)device_lane_->CompletionTurn();
  if (copy_lane_ != nullptr) {
    (void)copy_lane_->SubmissionTurn();
    (void)copy_lane_->CompletionTurn();
  }
  (void)scheduler_->Turn();
}

Status PagedNode::Await(Done& done, std::string_view what, std::uint64_t request) {
  bool cancelled = false;
  {
    Waiter wait(*this, what);
    // `gone`, not `retired`: the program is destroyed after Retired(), and
    // its destructor is its last touch of `done`.
    while (!done.gone.load()) {
      switch (wait.Poll(std::chrono::steady_clock::now())) {
        case WaitVerdict::kCancel:
          Cancel(request);
          break;
        case WaitVerdict::kGiveUp:
          GiveUp(what);
        case WaitVerdict::kWait:
          break;
      }
      if (threads_.empty()) {
        Round();  // --lanes inline, or before the lane threads start
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
    }
    cancelled = wait.cancelled();
  }
  if (cancelled) {
    hang_cancelled_ = true;
    ++hang_cancels_;
    return Error(std::format(
        "{} did not finish: nothing moved, so it was cancelled (a hang), and it drained", what));
  }
  if (done.outcome.load() != static_cast<int>(sc::TaskOutcome::kSucceeded)) {
    const int error = done.error.load();
    return Error(std::format(
        "{} failed{}", what,
        error >= 0 ? ": " + sc::ToString(static_cast<sc::WorkError>(error)) : std::string()));
  }
  // The scheduler's records are its thread's: read here only when this
  // thread drives it. With threads, a fault shows in the stop's result.
  if (threads_.empty()) {
    if (const auto fault = scheduler_->fault()) {
      return Error(std::format("{}: the node faulted: {}", what, sc::ToString(*fault)));
    }
  }
  return {};
}

std::uint64_t PagedNode::Submit(std::unique_ptr<sc::TaskProgram> program) {
  sc::Control start =
      sc::StartRequest{.request = ++request_, .priority = 1, .program = std::move(program)};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(start)) == base::PushResult::kFull) {
    std::this_thread::yield();
  }
  return request_;
}

void PagedNode::Cancel(std::uint64_t request) {
  if (threads_.empty()) {
    (void)scheduler_->Cancel(request);
    return;
  }
  sc::Control cancel = sc::CancelRequest{.request = request};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(cancel)) == base::PushResult::kFull) {
    std::this_thread::yield();
  }
}

Status PagedNode::Post(std::unique_ptr<sc::TaskProgram> program, Done& done,
                       std::string_view what) {
  if (!threads_.empty()) {
    return Await(done, what, Submit(std::move(program)));
  }
  if (scheduler_ == nullptr) {
    return Error(std::format("{}: the node has not started", what));
  }
  const std::uint64_t request = ++request_;
  if (!scheduler_->Start(request, std::move(program), 1)) {
    return Error(std::format("{} was not admitted", what));
  }
  return Await(done, what, request);
}

Status PagedNode::Load(std::vector<ExtentId> extents, std::string what,
                       std::vector<LoadStats>& log) {
  LoadStats stats{.what = std::move(what), .extents = 0, .seconds = 0, .requests = 0, .pieces = 0};
  for (const ExtentId extent : extents) {
    if (catalog_.Describe(extent).value().state != catalog::ExtentState::kResident) {
      ++stats.extents;
    }
  }
  auto closure = catalog_.ClosureOfExtents(extents);
  if (!closure) {
    return Error("a load's closure");
  }
  Done done;
  const std::uint64_t requests = counting_->requests.load(std::memory_order_relaxed);
  const std::uint64_t pieces = counting_->pieces.load(std::memory_order_relaxed);
  const auto start = std::chrono::steady_clock::now();
  if (auto posted =
          Post(std::make_unique<RunProgram>(done, *closure, sc::DeviceJob{}), done, stats.what);
      !posted) {
    return posted;
  }
  stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  // The load's reads have all completed, so their counts were made before.
  stats.requests = counting_->requests.load(std::memory_order_relaxed) - requests;
  stats.pieces = counting_->pieces.load(std::memory_order_relaxed) - pieces;
  log.push_back(std::move(stats));
  return {};
}

Status PagedNode::Evict(std::vector<ExtentId> extents, sc::EvictOptions options) {
  if (auto ended = EndRequestsOver(extents); !ended) {
    return std::unexpected(ended.error());
  }
  Done done;
  return Post(std::make_unique<EvictProgram>(done, std::move(extents), options), done,
              "an eviction");
}

Status PagedNode::Swap(std::vector<ExtentId> out, const catalog::Closure& in, bool handoff,
                       SwapReport& report, std::vector<ExtentId> release,
                       catalog::Closure weights) {
  // Asked for between a request's steps: the requests holding what goes
  // out end first, since the swap would wait for their leases.
  if (!weights.extents.empty()) {
    auto checked = Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : weights.extents) {
            const auto view = catalog_.Describe(extent);
            if (!view || view->content_generation != generation ||
                !catalog::Catalog::Evictable(*view))
              return Error("a selected idle weight is stale or held");
          }
          return {};
        },
        "validating selected idle weights before request retirement");
    if (!checked) return checked;
  }
  auto ended = EndRequestsOver(out);
  if (!ended) {
    return std::unexpected(ended.error());
  }
  Done done;
  auto swapped = Post(std::make_unique<SwapProgram>(done, std::move(out), in, handoff, report,
                                                    std::move(release), std::move(weights)),
                      done, "a swap");
  report.requests_ended = *ended;
  return swapped;
}

std::expected<sc::SchedulerStats, std::string> PagedNode::Stats() {
  sc::SchedulerStats stats;
  if (auto r = Call(
          [&]() -> Status {
            stats = scheduler_->stats();
            stats.catalog_occupancy_bytes = catalog_.OccupancyOf(domain_).Total().value();
            return {};
          },
          "reading the scheduler's counters");
      !r) {
    return std::unexpected(r.error());
  }
  stats.backing_creates = backing_create_stats();
  return stats;
}

Status PagedNode::Job(const catalog::Closure& closure, sc::DeviceJob job, std::string_view what,
                      std::uint32_t stream, const std::function<void()>& meanwhile) {
  if (job_active_) return Error("a device job cannot recursively dispatch another job");
  job_active_ = true;
  struct ActiveJob {
    bool& active;
    ~ActiveJob() { active = false; }
  } active{job_active_};
  if (const auto open = requests_.find(stream); open != requests_.end()) {
    return Step(stream, *open->second, closure, std::move(job), what, meanwhile);
  }
  Done done;
  Timing timing;
  const auto called = std::chrono::steady_clock::now();
  auto program =
      std::make_unique<RunProgram>(done, closure, Timed(std::move(job), stream, timing), stream);
  Status posted;
  if (meanwhile && threaded()) {
    const auto request = Submit(std::move(program));
    meanwhile();
    posted = Await(done, what, request);
  } else {
    posted = Post(std::move(program), done, what);
  }
  if (posted) {
    Note(stream, called, timing);
  }
  return posted;
}

sc::DeviceJob PagedNode::Timed(sc::DeviceJob job, std::uint32_t stream, Timing& timing) {
  if (stream >= events_.size()) {
    return job;  // not a compute stream: untimed
  }
  const providers::TimingMark begin = events_[stream].first;
  const providers::TimingMark end = events_[stream].second;
  return [inner = std::move(job), &timing, begin,
          end](providers::NativeStream native) mutable -> sc::JobResult {
    timing.started = std::chrono::steady_clock::now();
    // For timing only: a failure leaves the span unread.
    (void)providers::RecordTimingMark(begin, native);
    const sc::JobResult result = inner(native);
    (void)providers::RecordTimingMark(end, native);
    (void)providers::TakeLastError();
    timing.queued = std::chrono::steady_clock::now();
    timing.ran = true;
    return result;
  };
}

void PagedNode::Note(std::uint32_t stream, std::chrono::steady_clock::time_point called,
                     const Timing& timing) {
  if (stream >= times_.size()) {
    return;
  }
  const auto seconds = [](std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  const auto now = std::chrono::steady_clock::now();
  StepTimes& t = times_[stream];
  ++t.steps;
  t.wall += seconds(now - called);
  if (timing.ran) {
    t.dispatch += seconds(timing.started - called);
    t.job += seconds(timing.queued - timing.started);
    t.after += seconds(now - timing.queued);
    // The job's fence has completed, so both marks have.
    if (const auto ms =
            providers::ElapsedMilliseconds(events_[stream].first, events_[stream].second);
        ms) {
      t.device += static_cast<double>(*ms) / 1e3;
    }
    (void)providers::TakeLastError();
  }
}

StepTimes PagedNode::TakeTimes(std::uint32_t stream) {
  if (stream >= times_.size()) {
    return {};
  }
  return std::exchange(times_[stream], StepTimes{});
}

void PagedNode::Signal(std::uint64_t request) {
  sc::Control signal = sc::SignalRequest{.request = request};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(signal)) == base::PushResult::kFull) {
    if (threads_.empty()) {
      Round();
    } else {
      std::this_thread::yield();
    }
  }
}

Status PagedNode::BeginRequest(std::uint32_t stream, const catalog::Closure& closure,
                               std::string_view what) {
  if (requests_.contains(stream)) {
    return Error(std::format("{}: a request is open on stream {} already", what, stream));
  }
  auto open = std::make_unique<OpenRequest>();
  open->what = what;
  open->closure = closure;
  // Sorted (a closure is by construction; a hand-made one is made so), so
  // Step and EndRequestsOver can search it.
  std::ranges::sort(open->closure.extents);
  auto program = std::make_unique<RequestProgram>(open->done, closure, open->channel);
  if (!threads_.empty()) {
    open->request = Submit(std::move(program));
  } else {
    open->request = ++request_;
    if (!scheduler_->Start(open->request, std::move(program), 1)) {
      return Error(std::format("{} was not admitted", what));
    }
  }
  // Until its lease is held, or its task has ended without one.
  bool cancelled = false;
  {
    Waiter wait(*this, what);
    while (!open->channel.held.load(std::memory_order_acquire) && !open->done.gone.load()) {
      switch (wait.Poll(std::chrono::steady_clock::now())) {
        case WaitVerdict::kCancel:
          cancelled = true;
          Cancel(open->request);  // Await below waits for the drain
          break;
        case WaitVerdict::kGiveUp:
          GiveUp(what);
        case WaitVerdict::kWait:
          break;
      }
      if (cancelled) {
        break;
      }
      if (threads_.empty()) {
        Round();
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(20));
      }
    }
  }
  if (cancelled || !open->channel.held.load(std::memory_order_acquire)) {
    const std::uint64_t counted = hang_cancels_;
    auto ended = Await(open->done, what, open->request);
    if (cancelled) {
      hang_cancelled_ = true;
      hang_cancels_ = counted + 1;
      return Error(std::format("{}: no lease: nothing moved, so it was cancelled (a hang){}", what,
                               ended ? "" : ": " + ended.error()));
    }
    return Error(std::format("{}: no lease{}", what, ended ? "" : ": " + ended.error()));
  }
  requests_.emplace(stream, std::move(open));
  return {};
}

Status PagedNode::RefreshRequest(std::uint32_t stream, const catalog::Closure& closure) {
  const auto open = requests_.find(stream);
  if (open == requests_.end() || open->second->closure.extents == closure.extents) {
    return {};
  }
  const std::string what = open->second->what;
  if (auto ended = EndRequest(stream); !ended) {
    return ended;
  }
  return BeginRequest(stream, closure, what);
}

Status PagedNode::Step(std::uint32_t stream, OpenRequest& open, const catalog::Closure& closure,
                       sc::DeviceJob job, std::string_view what,
                       const std::function<void()>& meanwhile) {
  // Within the request's lease, exactly: the same entries, or each an
  // extent it holds at the contents it recorded (a whole model's closure
  // compares equal at a fraction of the search's cost).
  if (closure.extents != open.closure.extents &&
      !std::ranges::all_of(closure.extents, [&open](const auto& extent) {
        return std::ranges::binary_search(open.closure.extents, extent);
      })) {
    return Error(std::format("{}: its closure is not within {}'s", what, open.what));
  }
  if (settings_.direct_steps) {
    return DirectStep(stream, open, std::move(job), what, meanwhile);
  }
  Timing timing;
  open.channel.job = Timed(std::move(job), stream, timing);
  open.channel.stream = stream;
  const std::uint64_t before = open.channel.steps.load(std::memory_order_acquire);
  const auto called = std::chrono::steady_clock::now();
  Signal(open.request);
  if (meanwhile && threaded()) meanwhile();
  // The step's result is the next step's input, and a sleeping thread
  // wakes slowly on the Spark (RE-017), so the driver waits as the
  // runtime's lanes do (docs/experiments/runtime-wake/): asleep through
  // most of the step, spinning around its likely ends (the last few steps'
  // walls), woken early by the task's report whenever it sleeps.
  const auto spin_ahead = settings_.spin_ahead.value_or(kSpinAhead);
  bool cancelled = false;
  {
    Waiter wait(*this, what);
    while (open.channel.steps.load(std::memory_order_acquire) == before && !open.done.gone.load()) {
      const auto now = std::chrono::steady_clock::now();
      switch (wait.Poll(now)) {
        case WaitVerdict::kCancel:
          cancelled = true;
          Cancel(open.request);
          break;
        case WaitVerdict::kGiveUp:
          GiveUp(what);
        case WaitVerdict::kWait:
          break;
      }
      const auto next = open.walls.Next(now - called, kSpinPast);
      if (threads_.empty()) {
        Round();
      } else if (open.walls.known() && !next) {
        (void)open.channel.reported.WaitFor(kSpinPast);  // longer than any: sleep, woken by it
      } else if (next && now < called + *next - spin_ahead) {
        // Woken by the report; at least every kPatienceCheck, to ask the
        // patience.
        (void)open.channel.reported.WaitUntil(
            std::min(called + *next - spin_ahead, now + kPatienceCheck));
      } else {
        std::this_thread::yield();  // around a likely end, or none known yet
      }
    }
  }
  (void)open.channel.reported.Consume();
  open.walls.Add(std::chrono::steady_clock::now() - called);
  if (open.channel.steps.load(std::memory_order_acquire) == before) {
    // The request's task ended (cancelled, or failed): nothing holds the
    // job, which never ran, any more.
    open.channel.job = nullptr;
    const std::uint64_t counted = hang_cancels_;
    auto ended = Await(open.done, open.what, open.request);
    requests_.erase(stream);  // `open` is gone from here on
    if (cancelled) {
      hang_cancelled_ = true;
      hang_cancels_ = counted + 1;
      return Error(std::format("{}: nothing moved, so its request was cancelled (a hang){}", what,
                               ended ? "" : ": " + ended.error()));
    }
    return Error(std::format("{}: the request ended{}", what, ended ? "" : ": " + ended.error()));
  }
  if (cancelled) {
    // The step completed after all (the cancellation came too late to stop
    // it); the request it belonged to is cancelled, and its next step ends.
    hang_cancelled_ = true;
    ++hang_cancels_;
  }
  Note(stream, called, timing);
  if (open.channel.step_failed.load(std::memory_order_relaxed)) {
    const int error = open.channel.step_error.load(std::memory_order_relaxed);
    return Error(std::format(
        "{} failed{}", what,
        error >= 0 ? ": " + sc::ToString(static_cast<sc::WorkError>(error)) : std::string()));
  }
  return {};
}

Status PagedNode::Ended(std::uint32_t stream, std::string_view what, std::string_view why) {
  const auto found = requests_.find(stream);
  OpenRequest& open = *found->second;
  auto ended = Await(open.done, open.what, open.request);
  requests_.erase(found);  // `open` is gone from here on
  return Error(std::format("{}: {}{}", what, why, ended ? "" : ": " + ended.error()));
}

Status PagedNode::DirectStep(std::uint32_t stream, OpenRequest& open, sc::DeviceJob job,
                             std::string_view what, const std::function<void()>& meanwhile) {
  if (open.unproven || open.cancelled || open.done.gone.load()) {
    // Its task is ending or gone (cancelled, failed), or a step's end is
    // unproven: nothing more runs under its lease.
    return Ended(
        stream, what,
        open.unproven ? "a step of its request never proved its end" : "the request ended");
  }
  Timing timing;
  const auto called = std::chrono::steady_clock::now();
  // Counted before anything is queued, so its lease outlasts the step
  // whatever ends the request meanwhile (RequestChannel::external).
  open.channel.external->fetch_add(1, std::memory_order_acq_rel);
  ++direct_steps_;
  // The wait begins before the launch: a launch that blocks on a full
  // stream (RE-029) is the driver's wait too, which the runtime's hang
  // ladder watches from its other thread (it can only reach rung 3).
  std::optional<Waiter> wait;
  wait.emplace(*this, what);
  const providers::StreamId id = streams_[stream];
  sc::JobResult result = sc::JobResult::kNotStarted;
  {
    auto timed = Timed(std::move(job), stream, timing);
    const auto native = execution_->Submission(id);
    if (native) {
      result = timed(*native);
    } else if (native.error().error == providers::ProviderError::kUnknown) {
      result = sc::JobResult::kUnknown;
    }
  }
  // Recorded even when nothing started, as the device lane does: the
  // provider counts an attempted launch as queued work, and only a fence
  // seen complete balances it.
  const auto fence = execution_->Record(id);
  const bool queued = result != sc::JobResult::kNotStarted;
  if (!fence && queued) {
    // Nothing can show when the queued work ends: its lease is kept.
    open.unproven = true;
    return Error(
        std::format("{}: its step's fence could not be recorded: {}", what, fence.error().detail));
  }
  if (meanwhile && fence && threaded()) meanwhile();
  // The step's result is the next step's input, and a sleeping thread
  // wakes slowly on the Spark (RE-017): asleep through most of the step,
  // spinning around its likely ends (the last few steps' walls), as the
  // lanes do (docs/experiments/runtime-wake/).
  const auto spin_ahead = settings_.spin_ahead.value_or(kSpinAhead);
  bool seen = !fence;  // nothing to watch: nothing was queued
  bool cancelled = false;
  if (!fence) {
    wait.reset();
  } else {
    std::uint32_t refusals = 0;
    while (true) {
      const auto state = execution_->Query(*fence);
      if (state && *state == providers::FenceState::kComplete) {
        seen = true;
        break;
      }
      if (state) {
        refusals = 0;
      } else if (state.error().error == providers::ProviderError::kUnknown ||
                 ++refusals >= kFenceRefusals) {
        break;  // a device fault, or a refusal that persists: unproven
      }
      const auto now = std::chrono::steady_clock::now();
      switch (wait->Poll(now)) {
        case WaitVerdict::kCancel:
          // A hang's rung 1: the request ends, but its lease stays until
          // this step's fence is seen (or for good).
          cancelled = true;
          Cancel(open.request);
          break;
        case WaitVerdict::kGiveUp:
          GiveUp(what);
        case WaitVerdict::kWait:
          break;
      }
      const auto next = open.walls.Next(now - called, kSpinPast);
      if (threads_.empty()) {
        Round();
      } else if (next && now < called + *next - spin_ahead) {
        std::this_thread::sleep_until(std::min(called + *next - spin_ahead, now + kDirectNap));
      } else if (open.walls.known() && !next) {
        std::this_thread::sleep_for(kLateSleep);  // longer than any known: late
      } else {
        std::this_thread::yield();  // around a likely end, or none known yet
      }
    }
  }
  wait.reset();  // seen, or given up as unproven
  if (!seen) {
    open.unproven = true;  // its lease is kept: the step may still touch it
    open.cancelled = open.cancelled || cancelled;
    if (cancelled) {
      hang_cancelled_ = true;
      ++hang_cancels_;
    }
    return Error(std::format("{}: its step's end could not be proven", what));
  }
  if (fence) {
    for (std::uint32_t tries = 0; tries < kFenceRefusals; ++tries) {
      const auto released = execution_->Release(*fence);
      if (released || released.error().error != providers::ProviderError::kFailed) {
        break;  // released, or undetermined: never touched again
      }
    }
  }
  open.channel.external->fetch_sub(1, std::memory_order_acq_rel);
  // The step's likely end, for the next: when it began on the device plus
  // its span there, rather than when it was seen (a sleep may overshoot,
  // and nothing else wakes the driver).
  std::chrono::steady_clock::duration ended = std::chrono::steady_clock::now() - called;
  if (timing.ran && stream < events_.size()) {
    if (const auto ms =
            providers::ElapsedMilliseconds(events_[stream].first, events_[stream].second);
        ms) {
      ended = std::min(ended, (timing.started - called) +
                                  std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double, std::milli>(*ms)));
    }
    (void)providers::TakeLastError();
  }
  open.walls.Add(ended);
  if (cancelled) {
    // The step completed after all; the request it belonged to is
    // cancelled (its lease released now: the scheduler is told), and its
    // next step ends.
    open.cancelled = true;
    wake_.Signal();
    hang_cancelled_ = true;
    ++hang_cancels_;
  }
  Note(stream, called, timing);
  switch (result) {
    case sc::JobResult::kQueued:
      return {};
    case sc::JobResult::kNotStarted:
      return Error(std::format("{} failed: not started", what));
    case sc::JobResult::kFailed:
      return Error(std::format("{} failed", what));
    case sc::JobResult::kUnknown:
      // Its fence was seen, so its work ended; what it did is unknown.
      return Error(std::format("{} failed: its outcome is unknown", what));
  }
  return Error(std::format("{} failed", what));
}

Status PagedNode::EndRequest(std::uint32_t stream) {
  const auto found = requests_.find(stream);
  if (found == requests_.end()) {
    return Error(std::format("no request is open on stream {}", stream));
  }
  OpenRequest& open = *found->second;
  if (open.unproven || open.cancelled) {
    // Its task ends (or has) without the driver's end: cancelled, so the
    // wait cannot be for a signal; a lease under unproven work stays held.
    Cancel(open.request);
    auto ended = Await(open.done, open.what, open.request);
    const bool unproven = open.unproven;
    requests_.erase(found);
    if (unproven) {
      return Error("the request's step never proved its end; its lease is kept");
    }
    return ended;
  }
  open.channel.end = true;
  Signal(open.request);
  auto ended = Await(open.done, open.what, open.request);
  requests_.erase(found);
  return ended;
}

Status PagedNode::WithRequest(std::uint32_t stream, const catalog::Closure& closure,
                              std::string_view what, const std::function<Status()>& body) {
  if (auto r = BeginRequest(stream, closure, what); !r) {
    return r;
  }
  Status ran = body();
  Status ended = EndRequest(stream);
  return !ran ? ran : ended;
}

std::expected<std::uint64_t, std::string> PagedNode::EndRequestsOver(
    std::span<const ExtentId> extents) {
  std::vector<std::uint32_t> holding;
  for (const auto& [stream, open] : requests_) {
    const auto& held = open->closure.extents;  // sorted by extent
    if (std::ranges::any_of(extents, [&held](ExtentId extent) {
          const auto at = std::ranges::lower_bound(held, extent, {},
                                                   &std::pair<ExtentId, std::uint64_t>::first);
          return at != held.end() && at->first == extent;
        })) {
      holding.push_back(stream);
    }
  }
  for (const std::uint32_t stream : holding) {
    if (auto ended = EndRequest(stream); !ended) {
      return std::unexpected(ended.error());
    }
  }
  return holding.size();
}

Status PagedNode::Call(std::function<Status()> call, std::string_view what) {
  Done done;
  Status status;
  if (auto r = Post(std::make_unique<CallProgram>(done,
                                                  [&call, &status]() -> Status {
                                                    status = call();
                                                    return status;
                                                  }),
                    done, what);
      !r) {
    return !status ? status : r;
  }
  return {};
}

Status PagedNode::Acquire(const catalog::Closure& closure, AcquireReport& report,
                          std::string_view what, bool* over_budget) {
  if (over_budget != nullptr) *over_budget = false;
  if (job_active_) return Error("an acquisition cannot run while a device job is in flight");
  // The shared workspace is never a victim, whether or not the closure
  // names it: it is discardable, so it would be chosen first, and nothing
  // restores it.
  std::vector<ExtentId> workspace = activations_.extents;
  workspace.insert(workspace.end(), pool_.extents.begin(), pool_.extents.end());
  if (reclaimer_) {
    std::vector<ExtentId> protect = workspace;
    for (const auto& [extent, generation] : closure.extents) {
      (void)generation;
      protect.push_back(extent);
    }
    std::uint64_t reclaimed = 1;
    for (int round = 0;; ++round) {
      memory::MaterializationPlan plan;
      if (auto checked = Call(
              [&]() -> Status {
                // A node owns one physical domain. Validate the entire
                // closure before domain-local sizing or any reclaim callback.
                for (const auto& [extent, generation] : closure.extents) {
                  const auto view = catalog_.Describe(extent);
                  if (!view || view->descriptor.domain != domain_ ||
                      view->content_generation != generation || view->discarded ||
                      view->state == catalog::ExtentState::kQuarantined) {
                    return Error("an acquisition closure is foreign, stale or quarantined");
                  }
                }
                plan = memory::PlanMaterialization(catalog_, domain_, budget_, closure, protect,
                                                   /*select_victims=*/false);
                return {};
              },
              "checking a policy-managed acquisition");
          !checked)
        return checked;
      if (!plan.stale.empty() || !plan.quarantined.empty() ||
          (!plan.feasible && plan.shortfall.value() == 0)) {
        return Error(std::format("{}: the acquisition closure is stale or quarantined", what));
      }
      if (plan.shortfall.value() == 0) break;
      if (round >= memory::kReclaimRounds || reclaimed == 0) {
        if (over_budget != nullptr) *over_budget = true;
        return Error(std::format("{}: the acquisition remains over budget", what));
      }
      reclaimed = AskReclaim(plan.shortfall.value(), ReclaimFor::kMaterialize, protect);
    }
  }
  Done done;
  auto program = std::make_unique<AcquireProgram>(done, closure, domain_, budget_, report,
                                                  std::move(workspace), !reclaimer_);
  auto acquired = Post(std::move(program), done, what);
  if (over_budget != nullptr) {
    // Read once the program is gone (Post returns only then).
    *over_budget = !acquired && done.error.load() == static_cast<int>(sc::WorkError::kOverBudget);
  }
  return acquired;
}

Status PagedNode::Copy(std::uint32_t stream, const catalog::Closure& closure, std::uint64_t device,
                       void* host, std::uint64_t bytes, bool to_host, std::string_view what) {
  if (host == nullptr || device == 0 || bytes == 0) {
    return Error(std::format("{}: nothing to copy", what));
  }
  // The device range lies in the closure's extents, which the job leases.
  return Job(
      closure,
      [device, host, bytes, to_host](providers::NativeStream native) {
        auto* on_device = reinterpret_cast<void*>(device);  // NOLINT(performance-no-int-to-ptr)
        const providers::DeviceStatus r =
            to_host ? providers::CopyAsync(native, host, on_device, bytes,
                                           providers::CopyKind::kDeviceToHost)
                    : providers::CopyAsync(native, on_device, host, bytes,
                                           providers::CopyKind::kHostToDevice);
        return r.ok() ? sc::JobResult::kQueued : sc::JobResult::kUnknown;
      },
      what, stream);
}

Status PagedNode::TearDown(std::span<PagedModel* const> models) {
  if (torn_down_) {
    return {};
  }
  torn_down_ = true;
  std::vector<std::string> problems;
  // Requests first: their leases would hold what the teardown evicts.
  std::vector<std::uint32_t> open;
  open.reserve(requests_.size());
  for (const auto& [stream, request] : requests_) {
    open.push_back(stream);
  }
  for (const std::uint32_t stream : open) {
    if (auto ended = EndRequest(stream); !ended) {
      problems.push_back(ended.error());
    }
  }
  if (scheduler_ != nullptr) {
    // A fence after anything noted on each model's stream (a measuring
    // launch context notes work even when nothing runs), so it can be
    // destroyed; then every managed backing released through its
    // eviction; then the scheduler stops and the lanes drain. The fence
    // leases nothing: it orders only the stream, and a model swapped out
    // has its state written back, perhaps with no room to page it in again
    // (and no reason to). A stream that could not be fenced may still be
    // using any model's memory, so then nothing is evicted or released.
    const bool usable = !threads_.empty() || !scheduler_->fault();
    const catalog::Closure nothing;
    std::vector<ExtentId> managed;
    bool fenced_all = usable;
    for (PagedModel* model : models) {
      if (usable) {
        if (auto fenced = Job(
                nothing, [](providers::NativeStream) { return sc::JobResult::kQueued; },
                "fencing a compute stream", model->stream());
            !fenced) {
          problems.push_back(std::format("compute stream {} could not be fenced: {}",
                                         model->stream(), fenced.error()));
          fenced_all = false;
        }
      }
      const auto extents = model->managed_extents();
      managed.insert(managed.end(), extents.begin(), extents.end());
    }
    if (fenced_all && !Evict(managed)) {
      problems.emplace_back("the weights could not be evicted at the end");
    }
    scheduler_->RequestShutdown();
    if (threads_.empty()) {
      DriveUntil([this] { return scheduler_->Stopped().has_value(); }, [this] { Round(); },
                 [this] { return progress(); }, settings_.quiet);
      stopped_ = scheduler_->Stopped();
    } else {
      threads_.front().join();
    }
    const bool driven = threads_.empty();
    storage_lane_->Close();
    device_lane_->Close();
    backing_lane_->Close();
    if (copy_lane_ != nullptr) {
      copy_lane_->Close();
    }
    if (driven) {
      for (int i = 0; i < 1000 && (storage_->in_flight() > 0 || i < 10); ++i) {
        (void)storage_lane_->Turn(false);
        (void)backing_lane_->Turn();
        (void)device_lane_->SubmissionTurn();
        (void)device_lane_->CompletionTurn();
        if (copy_lane_ != nullptr) {
          (void)copy_lane_->SubmissionTurn();
          (void)copy_lane_->CompletionTurn();
        }
      }
    }
    threads_.clear();
    // The lanes are idle for good: the reserve's backing goes first.
    if (!backing_lane_->ReleaseReserve()) {
      problems.emplace_back("the handle reserve could not be released");
    }
    if (!stopped_ || !stopped_->has_value()) {
      problems.emplace_back("the scheduler stopped with a fault: backing is left as it is");
      return Joined(problems);
    }
    if (!fenced_all) {
      problems.emplace_back("a compute stream was not fenced: backing is left as it is");
      return Joined(problems);
    }
  } else if (execution_ != nullptr) {
    // Setup ended before the scheduler existed, and a model may already
    // have queued work on its stream (the EXL3 launch context zeroes its
    // lock area): each compute stream is fenced before any memory goes.
    for (std::size_t i = 0; i < settings_.compute_streams && i < streams_.size(); ++i) {
      if (!Fence(*execution_, streams_[i], settings_.quiet)) {
        problems.emplace_back("a compute stream could not be fenced: backing is left as it is");
        return Joined(problems);
      }
    }
  }
  for (PagedModel* model : models) {
    if (auto released = model->Release(); !released) {
      problems.push_back(released.error());
    }
  }
  if (execution_ != nullptr) {
    for (const auto stream : streams_) {
      if (stream.valid() && !execution_->DestroyStream(stream)) {
        problems.emplace_back("a stream could not be destroyed");
      }
    }
  }
  for (const auto& [begin, end] : events_) {
    providers::DestroyTimingMark(begin);
    providers::DestroyTimingMark(end);
  }
  events_.clear();
  if (memory_ != nullptr) {
    for (Mapped* mapped : {&zone_, &activations_, &pool_}) {
      if (!ReleaseMapped(*memory_, *mapped)) {
        problems.push_back(std::format("{} could not be released", mapped->name));
      }
    }
    if (memory_->backings() != 0) {
      problems.push_back(std::format("{} backings were left", memory_->backings()));
    }
  }
  for (void* pointer : pinned_) {
    providers::FreePinned(pointer);
  }
  pinned_.clear();
  return Joined(problems);
}

}  // namespace jitllm::engine
