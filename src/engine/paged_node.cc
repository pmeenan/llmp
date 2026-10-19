// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_node.h"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <iterator>
#include <limits>
#include <new>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <utility>

#include "base/bounded_queue.h"
#include "base/check.h"
#include "engine/live_state.h"
#include "engine/supplemental_run.h"
#include "engine/support.h"
#include "memory/materialize.h"
#include "memory/reclaim.h"
#include "platform/crash_policy.h"
#include "providers/device_runtime.h"
#include "providers/direct_reader.h"

namespace llmp::engine {

namespace {

namespace sc = llmp::scheduler;
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
std::atomic<std::uint64_t> next_node_generation{0};

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
// A direct step on a stream with no step history yet (its first): the
// driver spins this long, then naps from kFirstNap, doubling to kDirectNap,
// as the device lanes back off without history (DeviceService).
constexpr auto kFirstSpin = std::chrono::milliseconds(20);
constexpr auto kFirstNap = std::chrono::microseconds(50);
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

// Catalog thread only. Caller-supplied occupancy is a consistency check,
// never admission authority, for both ordinary and prepaid replacements.
Status CheckReplacementClosure(const catalog::Catalog& catalog, catalog::DomainId domain,
                               const catalog::Closure& closure) {
  base::Bytes actual;
  std::optional<catalog::ExtentId> previous;
  for (const auto& [extent, generation] : closure.extents) {
    const auto view = catalog.Describe(extent);
    if (!view || view->descriptor.domain != domain || view->content_generation != generation ||
        view->discarded || view->state == catalog::ExtentState::kQuarantined ||
        (previous && extent <= *previous))
      return Error("replacement closure is foreign, stale or noncanonical");
    auto total = actual.Plus(view->descriptor.size);
    if (!total) return Error("replacement physical occupancy overflows");
    actual = *total;
    previous = extent;
  }
  if (closure.extents.empty() ? !closure.bytes_by_domain.empty()
                              : closure.bytes_by_domain.size() != 1 ||
                                    closure.bytes_by_domain.begin()->first != domain ||
                                    closure.bytes_by_domain.begin()->second != actual)
    return Error("replacement closure occupancy disagrees with its actual extents");
  return {};
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

struct RequestExtension::Data final : kernels::ResourceDependency {
  Data(HostPayment paid, PagedNode& owner, std::uint64_t issuer, std::uint32_t at,
       std::uint64_t request_id, catalog::LeaseId held)
      : payment(std::move(paid)),
        node(&owner),
        generation(issuer),
        stream(at),
        request(request_id),
        lease(held) {}
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override {
    auto paid = std::move(payment);
    delete this;  // consumer-safe metadata only; no driver or node callback
    paid.Reset();
  }
  // Forward and close use the existing serialized CallMetadata path.
  // Controls run sequentially, so one actual CallProgram/callback allocation
  // capacity suffices. Capture bounds are checked against the real closure types below.
  static constexpr auto kValidationCaptureBytes = 4 * sizeof(void*);
  static constexpr auto kRemovalCaptureBytes = 3 * sizeof(void*);
  static constexpr auto kReplacementCaptureBytes = 5 * sizeof(void*);
  static constexpr auto kControlBytes =
      sizeof(sc::CallProgram) + kReplacementCaptureBytes + 2 * std::uint64_t{64};
  HostPayment payment, installed_payment;
  std::vector<std::pair<catalog::ExtentId, std::uint64_t>> additions;
  catalog::Closure candidate;
  sc::Scheduler::LeaseReplacementStorage prepared;
  PagedNode* node;
  std::uint64_t generation;
  std::uint32_t stream;
  std::uint64_t request;
  catalog::LeaseId lease;
  enum class State : std::uint8_t { kPrepared, kActive, kClosed };
  State state = State::kPrepared;
};

PagedNode::OpenRequest::~OpenRequest() {
  // This invalidates only metadata/holder identity, never certifies native
  // work. A private batch independently retains every unproven child owner.
  if (extension) extension.data_->state = RequestExtension::Data::State::kClosed;
}

PagedNode::PagedNode(NodeSettings settings)
    : lifetime_generation_(next_node_generation.fetch_add(1, std::memory_order_relaxed) + 1),
      settings_(settings) {
  base::Check(lifetime_generation_ != 0, "node token generation overflow");
}

PagedNode::~PagedNode() {
  // Continuing ordinary member destruction would release providers/streams
  // under retained launch borrowers. Runtime uses RetireServiceOrExit on a
  // failed teardown; direct embedders must follow the same fail-stop boundary.
  base::Check(model_roots_ == nullptr,
              "a node cannot destruct with unretired adopted runner resources");
  if (staging_lifetime_) staging_lifetime_->live_ = false;
  staging_lifetime_.reset();
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
  (void)CloseWeightFiles(true);
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
  if (memory_ || execution_ || torn_down_) return Error("the node cannot be opened twice");
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

Status PagedNode::MapWorkspace(std::uint64_t activations, std::uint64_t pool,
                               std::uint64_t virtual_capacity, bool pageable) {
  if (scheduler_) return Error("workspace reservation must precede node start");
  if (activations_.reservation.valid() || pool_.reservation.valid())
    return Error("the workspace is mapped once");
  workspace_pageable_ = pageable;
  if (auto r = MapRegion(activations_, "the activations", activations,
                         providers::BackingKind::kDevice, MemoryClass::kScratch,
                         catalog::Recovery::kDiscardable, kShared, !pageable, virtual_capacity);
      !r)
    return r;
  return MapRegion(pool_, "the GGML pool", pool, providers::BackingKind::kDevice,
                   MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared, !pageable,
                   virtual_capacity);
}

std::vector<ExtentId> PagedNode::workspace_extents() const {
  std::vector<ExtentId> all = activations_.extents;
  all.insert(all.end(), pool_.extents.begin(), pool_.extents.end());
  return all;
}

sc::PageSource PagedNode::WorkspaceSource(const Mapped& mapped, std::size_t index) const {
  // Scratch has no contents to restore: a load maps backing at the
  // extent's own place and is done, with no read, no landing slot and no
  // copy (scheduler.h, PageSource::fresh). The place never moves.
  const std::uint64_t offset = index * kPagedExtent;
  return sc::PageSource{.read = {.fd = -1, .offset = 0, .memory = nullptr, .length = kPagedExtent},
                        .landed = true,
                        .destination = mapped.base + offset,
                        .backing = sc::BackingPlace{.reservation = mapped.reservation,
                                                    .offset = Bytes(offset),
                                                    .size = Bytes(kPagedExtent),
                                                    .allocation_class = device_class_},
                        .fresh = true};
}

Status PagedNode::RegisterWorkspaceSources(const Mapped& mapped) {
  for (std::size_t i = 0; i < mapped.extents.size(); ++i) {
    if (auto set = scheduler_->SetSource(mapped.extents[i], WorkspaceSource(mapped, i)); !set) {
      return Error(std::format("{}'s place: {}", mapped.name, sc::ToString(set.error())));
    }
  }
  if (auto pinned = scheduler_->PinPlaces(mapped.extents); !pinned) {
    return Error(std::format("pinning {}'s places: {}", mapped.name, sc::ToString(pinned.error())));
  }
  return {};
}

Status PagedNode::GrowPageable(Mapped& mapped, std::uint64_t bytes) {
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ ||
      bytes > UINT64_MAX - (kPagedExtent - 1))
    return Error("workspace growth has invalid size or an unretired operation");
  const auto rounded =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (rounded <= mapped.bytes) return {};
  if (!mapped.reservation.valid() || rounded > mapped.capacity_bytes)
    return Error("workspace growth exceeds its virtual reservation");
  const auto count = rounded / kPagedExtent;
  // The host metadata a growth adds (catalog, span and source entries,
  // with the vectors' transient copies), charged as GrowMapped charges it.
  if (count > (UINT64_MAX - 4096) / 1024) return Error("workspace metadata overflows");
  const std::uint64_t metadata = count * 1024 + 4096;
  if (!TryChargeHost(metadata)) return Error("workspace metadata does not fit");
  std::size_t added = 0;
  auto cataloged = RegionCatalog(
      [&]() -> Status {
        // No physical effect: each new extent is NONRESIDENT until a closure
        // that names it is materialized. An extent joins the region only
        // with its source set and its place pinned; one refused either is
        // forgotten, so no extent of the region lacks a place.
        Status failed;
        std::vector<ExtentId> placed;
        spans_.reserve(spans_.size() + static_cast<std::size_t>(count - mapped.extents.size()));
        for (std::uint64_t at = mapped.bytes; at < rounded; at += kPagedExtent) {
          const auto extent = catalog_.AddExtent({.domain = domain_,
                                                  .memory_class = MemoryClass::kScratch,
                                                  .recovery = catalog::Recovery::kDiscardable,
                                                  .size = Bytes(kPagedExtent),
                                                  .content = {}},
                                                 false);
          if (!extent) {
            failed = Error(std::format("cataloging {}", mapped.name));
            break;
          }
          const std::array one = {*extent};
          if (auto set = scheduler_->SetSource(*extent, WorkspaceSource(mapped, at / kPagedExtent));
              !set || !scheduler_->PinPlaces(one)) {
            base::Check(catalog_.RemoveExtent(*extent).has_value(), "an unplaced workspace extent");
            failed = Error(std::format("{}'s place: {}", mapped.name,
                                       set ? "not pinned" : sc::ToString(set.error())));
            break;
          }
          mapped.extents.push_back(*extent);
          AddSpan({.base = mapped.base + at,
                   .size = kPagedExtent,
                   .extent = *extent,
                   .memory_class = MemoryClass::kScratch,
                   .device = true,
                   .owner = kShared});
          mapped.bytes = at + kPagedExtent;
          ++added;
        }
        SortSpans();
        return failed;
      },
      "cataloging workspace growth");
  if (added == 0) {
    UnchargeHost(metadata);  // nothing grew: the region's charge is as it was
  } else {
    UnchargeHost(mapped.metadata_bytes);
    mapped.metadata_bytes = metadata;
  }
  return cataloged;
}

Status PagedNode::GrowWorkspace(std::uint64_t activations, std::uint64_t pool) {
  if (workspace_pageable_) {
    if (auto r = GrowPageable(activations_, activations); !r) return r;
    return GrowPageable(pool_, pool);
  }
  if (auto r = GrowMapped(activations_, activations, providers::BackingKind::kDevice,
                          MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared);
      !r)
    return r;
  // A refused second growth leaves a usable larger first prefix. It is retained
  // node-wide capacity, not an allocation owned by the failed image request.
  return GrowMapped(pool_, pool, providers::BackingKind::kDevice, MemoryClass::kScratch,
                    catalog::Recovery::kDiscardable, kShared);
}

Status PagedNode::MapResident(Mapped& mapped, std::string name, std::uint64_t bytes,
                              providers::BackingKind kind, MemoryClass memory_class,
                              catalog::Recovery recovery, int owner, bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (scheduler_) {
    if (mapped.reservation.valid()) return Error("an owned region is already mapped");
    mapped.name = std::move(name);
    return GrowMapped(mapped, bytes, kind, memory_class, recovery, owner, over_budget);
  }
  return MapRegion(mapped, std::move(name), bytes, kind, memory_class, recovery, owner, true);
}

Status PagedNode::ReserveState(Mapped& mapped, std::string name, std::uint64_t bytes, int owner) {
  if (threads_.empty()) {
    // Before Run (or lanes driven on this thread): nothing else calls the
    // provider or touches the catalog meanwhile.
    return MapRegion(mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                     MemoryClass::kLiveState, catalog::Recovery::kPreserve, owner, false);
  }
  // After Run the VMM provider is the backing lane's alone (its handle
  // reserve refills in the background) and the catalog the scheduler's: the
  // reservation goes to the lane, the extents to the scheduler's thread.
  // Never from inside a Call: both wait for the scheduler.
  if (mapped.reservation.valid()) return Error("a mapped region cannot be overwritten");
  if (job_active_ || supplemental_ || torn_down_ ||
      bytes > std::numeric_limits<std::uint64_t>::max() - kPagedExtent + 1)
    return Error("state reservation has invalid size or an unretired operation");
  const std::uint64_t rounded =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  const std::uint64_t count = rounded / kPagedExtent;
  // The host metadata it adds (catalog and span entries), as GrowPageable
  // charges it.
  if (count > (UINT64_MAX - 4096) / 1024) return Error("state reservation metadata overflows");
  const std::uint64_t metadata = count * 1024 + 4096;
  if (!TryChargeHost(metadata)) return Error("state reservation metadata does not fit");
  auto result = std::make_shared<sc::RegionResult>();
  if (auto reserved = Region({.kind = sc::RegionWork::Kind::kReserve,
                              .size = Bytes(rounded),
                              .extent_size = Bytes(kPagedExtent),
                              .offset = {},
                              .allocation_class = 0,
                              .reservation = {},
                              .result = result},
                             "reserving a state region");
      !reserved) {
    UnchargeHost(metadata);
    return reserved;
  }
  const std::uint64_t at_base = result->base;
  std::vector<ExtentId> extents;
  extents.reserve(static_cast<std::size_t>(count));
  auto cataloged = RegionCatalog(
      [&]() -> Status {
        // No physical effect: each extent is NONRESIDENT until its page
        // source materializes it, as MapRegion catalogs one before Run.
        spans_.reserve(spans_.size() + static_cast<std::size_t>(count));
        for (std::uint64_t at = 0; at < rounded; at += kPagedExtent) {
          const auto extent = catalog_.AddExtent({.domain = domain_,
                                                  .memory_class = MemoryClass::kLiveState,
                                                  .recovery = catalog::Recovery::kPreserve,
                                                  .size = Bytes(kPagedExtent),
                                                  .content = {}},
                                                 false);
          if (!extent) {
            for (const ExtentId added : extents)
              base::Check(catalog_.RemoveExtent(added).has_value(), "an unpublished state extent");
            extents.clear();
            return Error(std::format("cataloging {}", name));
          }
          extents.push_back(*extent);
        }
        for (std::size_t i = 0; i < extents.size(); ++i) {
          AddSpan({.base = at_base + (i * kPagedExtent),
                   .size = kPagedExtent,
                   .extent = extents[i],
                   .memory_class = MemoryClass::kLiveState,
                   .device = true,
                   .owner = owner});
        }
        SortSpans();
        return {};
      },
      "cataloging a state region");
  if (!cataloged) {
    // Nothing of it is cataloged: its empty reservation goes back.
    if (auto freed = Region({.kind = sc::RegionWork::Kind::kFree,
                             .size = {},
                             .extent_size = Bytes(kPagedExtent),
                             .offset = {},
                             .allocation_class = 0,
                             .reservation = result->reservation,
                             .result = result},
                            "freeing an uncataloged state reservation");
        !freed) {
      // Kept owned by the region, released with it at teardown.
      mapped.name = std::move(name);
      mapped.reservation = result->reservation;
      mapped.base = at_base;
      mapped.metadata_bytes = metadata;
      return freed;
    }
    UnchargeHost(metadata);
    return cataloged;
  }
  mapped.name = std::move(name);
  mapped.reservation = result->reservation;
  mapped.base = at_base;
  mapped.bytes = mapped.capacity_bytes = rounded;
  mapped.extents = std::move(extents);
  mapped.metadata_bytes = metadata;
  return {};
}

Status PagedNode::MapRegion(Mapped& mapped, std::string name, std::uint64_t bytes,
                            providers::BackingKind kind, MemoryClass memory_class,
                            catalog::Recovery recovery, int owner, bool resident,
                            std::uint64_t virtual_capacity) {
  if (mapped.reservation.valid()) return Error("a mapped region cannot be overwritten");
  // It calls the provider and the catalog from this thread: after Run they
  // are the backing lane's and the scheduler's (ReserveState, GrowMapped).
  if (!threads_.empty()) return Error("a region is mapped directly only before Run");
  if (bytes > std::numeric_limits<std::uint64_t>::max() - kPagedExtent + 1) {
    return Error("a region's size overflows its extent alignment");
  }
  mapped.name = std::move(name);
  mapped.bytes =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (virtual_capacity > UINT64_MAX - (kPagedExtent - 1))
    return Error("virtual reservation alignment overflows");
  mapped.capacity_bytes =
      std::max(mapped.bytes, (virtual_capacity + kPagedExtent - 1) / kPagedExtent * kPagedExtent);
  auto reservation = memory_->Reserve(Bytes(mapped.capacity_bytes));
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

Status PagedNode::Region(sc::RegionWork work, std::string_view what) {
  class RegionProgram final : public sc::ReportingProgram {
   public:
    RegionProgram(Done& done, sc::RegionWork work)
        : ReportingProgram(done), work_(std::move(work)) {}
    sc::Step Advance(sc::TaskContext& context) override {
      if (context.TakeFailure()) return sc::Step::Finish(sc::TaskOutcome::kFailed);
      if (submitted_) return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
      const auto submitted = context.SubmitRegion({}, work_);
      if (!submitted) {
        if (submitted.error() == sc::WorkError::kBusy) return sc::Step::Yield();
        return Fail(submitted.error());
      }
      submitted_ = true;
      return sc::Step::Wait();
    }

   private:
    sc::RegionWork work_;
    bool submitted_ = false;
  };
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_)
    return Error("a region transition requires a live idle node");
  Done done;
  auto result = work.result;
  // Reset before submission, including cancellation that never reaches a lane.
  result->outcome = sc::RegionResult::Outcome::kUntouched;
  result->failure.reset();
  auto ran = Post(std::make_unique<RegionProgram>(done, std::move(work)), done, what);
  // Task cancellation can follow positive provider completion. Its effects
  // still belong to this caller and must be published/retired, never refunded.
  if (result->outcome == sc::RegionResult::Outcome::kComplete) return {};
  base::Check(result->outcome != sc::RegionResult::Outcome::kUnknown,
              "unknown region completion cannot return to its owner");
  if (result->failure) return Error(std::format("{}: {}", what, result->failure->detail));
  return ran ? Error("region task retired without completed provider work") : ran;
}

Status PagedNode::RegionCatalog(std::function<Status()> mutation, std::string_view what) {
  std::optional<Status> completed;
  auto task = Call(
      [&]() -> Status {
        completed = mutation();
        return *completed;
      },
      what);
  // The completed transaction is authoritative even if its task is cancelled
  // afterwards. If it never ran, no catalog effects were made by this call.
  return completed ? std::move(*completed) : task;
}

Status PagedNode::GrowMapped(Mapped& mapped, std::uint64_t bytes, providers::BackingKind kind,
                             MemoryClass memory_class, catalog::Recovery recovery, int owner,
                             bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ ||
      bytes > UINT64_MAX - (kPagedExtent - 1))
    return Error("owned-region growth has invalid size or an unretired operation");
  const auto rounded =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (rounded <= mapped.bytes) return {};
  if (mapped.reservation.valid() && rounded > mapped.capacity_bytes)
    return Error("owned-region growth exceeds its virtual reservation");
  const auto added = rounded - mapped.bytes;
  const auto count = rounded / kPagedExtent;
  // Per extent: both old/new vector storage, provider and catalog descriptors,
  // span and transition handles. This charge precedes vector growth and is
  // retained with the mapping. The old metadata charge remains during copying.
  static_assert(sizeof(catalog::ExtentView) + sizeof(Span) + 4 * sizeof(providers::BackingId) +
                    4 * sizeof(ExtentId) <=
                512);
  if (count > (UINT64_MAX - 4096) / 1024) return Error("owned-region metadata overflows");
  std::uint64_t catalog_storage = 0;
  if (auto read = RegionCatalog(
          [&]() -> Status {
            catalog_storage = catalog_.ExtentStorageBytes();
            return {};
          },
          "measuring owned-region metadata");
      !read)
    return read;
  // Appending one extent may reallocate the whole existing catalog/span
  // vectors. Their capacity survives geometry retirement, so retain a separate
  // node-wide high-water charge, including transient old/new vector storage.
  const std::uint64_t metadata = count * 1024 + 4096;
  std::uint64_t retained = 0, charge = 0;
  if (__builtin_add_overflow(catalog_storage, std::uint64_t{spans_.capacity()} * sizeof(Span),
                             &retained) ||
      __builtin_mul_overflow(retained, std::uint64_t{2}, &retained) ||
      __builtin_add_overflow(retained, metadata, &retained))
    return Error("owned-region retained metadata copy overflows");
  const auto growth = retained > region_metadata_bytes_ ? retained - region_metadata_bytes_ : 0;
  if (__builtin_add_overflow(metadata, growth, &charge))
    return Error("owned-region metadata charge overflows");
  if (!TryChargeHost(charge)) {
    if (over_budget) *over_budget = true;
    return Error("owned-region metadata does not fit");
  }
  region_metadata_bytes_ += growth;
  auto result = std::make_shared<sc::RegionResult>();
  const auto new_count = static_cast<std::size_t>(added / kPagedExtent);
  result->backings.reserve(new_count);
  mapped.backings.reserve(static_cast<std::size_t>(count));
  mapped.extents.reserve(static_cast<std::size_t>(count));
  std::vector<catalog::Ticket> tickets;
  tickets.reserve(new_count);
  // Each new physical extent is cataloged LOADING before any provider effect.
  // Existing extents and their generations remain unchanged. The scheduler is
  // the only publisher; the backing lane receives only already-paid work.
  const auto cancel_tickets = [&] {
    for (auto ticket : tickets) {
      base::Check(catalog_.FailLoad(ticket, true).has_value(), "owned growth rollback ticket");
      base::Check(catalog_.RemoveExtent(ticket.extent).has_value(), "owned growth rollback extent");
    }
    tickets.clear();
  };
  auto admitted = RegionCatalog(
      [&]() -> Status {
        if (Shortfall(added) != 0) {
          if (over_budget) *over_budget = true;
          return Error("owned-region backing does not fit");
        }
        spans_.reserve(spans_.size() + new_count);
        for (std::size_t i = 0; i < new_count; ++i) {
          const auto extent = catalog_.AddExtent({.domain = domain_,
                                                  .memory_class = memory_class,
                                                  .recovery = recovery,
                                                  .size = Bytes(kPagedExtent),
                                                  .content = {}},
                                                 false);
          if (!extent) {
            cancel_tickets();
            return Error("owned-region catalog capacity");
          }
          const auto ticket = catalog_.BeginLoad(*extent, budget_);
          if (!ticket) {
            base::Check(catalog_.RemoveExtent(*extent).has_value(), "unadmitted owned extent");
            cancel_tickets();
            return Error("owned-region backing admission");
          }
          tickets.push_back(*ticket);
        }
        return {};
      },
      "admitting owned-region growth");
  if (!admitted) {
    UnchargeHost(metadata);
    return admitted;
  }
  const auto repay = [&] {
    auto canceled = RegionCatalog(
        [&]() -> Status {
          cancel_tickets();
          return {};
        },
        "undoing untouched owned-region growth");
    base::Check(canceled.has_value(), "paid owned growth rollback");
    UnchargeHost(metadata);
  };
  const bool fresh = !mapped.reservation.valid();
  if (fresh) {
    auto reserved = Region({.kind = sc::RegionWork::Kind::kReserve,
                            .size = Bytes(rounded),
                            .extent_size = Bytes(kPagedExtent),
                            .offset = {},
                            .allocation_class = 0,
                            .reservation = {},
                            .result = result},
                           "reserving owned region");
    if (!reserved) {
      repay();
      return reserved;
    }
    mapped.reservation = result->reservation;
    mapped.base = result->base;
    mapped.capacity_bytes = rounded;
  }
  const auto allocation_class =
      kind == providers::BackingKind::kDevice ? device_class_ : host_class_;
  auto grown = Region({.kind = sc::RegionWork::Kind::kGrow,
                       .size = Bytes(added),
                       .extent_size = Bytes(kPagedExtent),
                       .offset = Bytes(mapped.bytes),
                       .allocation_class = allocation_class,
                       .reservation = mapped.reservation,
                       .result = result},
                      "growing owned region");
  if (!grown) {
    // A known refusal has rolled back every new handle. Unknown operations
    // cannot retire their RegionProgram and are stopped by Await's failstop.
    if (fresh) {
      auto canceled = RegionCatalog(
          [&]() -> Status {
            cancel_tickets();
            return {};
          },
          "retiring rolled-back owned growth tickets");
      base::Check(canceled.has_value(), "known rolled-back growth tickets");
      if (auto freed = Region({.kind = sc::RegionWork::Kind::kFree,
                               .size = {},
                               .extent_size = Bytes(kPagedExtent),
                               .offset = {},
                               .allocation_class = 0,
                               .reservation = mapped.reservation,
                               .result = result},
                              "unwinding owned reservation");
          !freed) {
        // No physical suffix remains; retain the reservation and local metadata
        // so candidate cleanup or terminal owner teardown can release them.
        mapped.metadata_bytes = metadata;
        return freed;
      }
      mapped.reservation = {};
      mapped.base = mapped.capacity_bytes = 0;
    }
    repay();
    return grown;
  }
  mapped.backings.insert(mapped.backings.end(), result->backings.begin(), result->backings.end());
  auto published = RegionCatalog(
      [&]() -> Status {
        std::uint64_t at = mapped.bytes;
        for (auto ticket : tickets) {
          base::Check(catalog_.CompleteLoad(ticket).has_value(), "owned growth publication");
          mapped.extents.push_back(ticket.extent);
          AddSpan({.base = mapped.base + at,
                   .size = kPagedExtent,
                   .extent = ticket.extent,
                   .memory_class = memory_class,
                   .device = kind == providers::BackingKind::kDevice,
                   .owner = owner});
          at += kPagedExtent;
        }
        SortSpans();
        return {};
      },
      "publishing owned-region growth");
  base::Check(published.has_value(), "completed owned growth must publish its paid tickets");
  mapped.bytes = rounded;
  UnchargeHost(mapped.metadata_bytes);
  mapped.metadata_bytes = metadata;
  return {};
}

Status PagedNode::RetireMapped(Mapped& mapped) {
  if (!mapped.reservation.valid()) return {};
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ ||
      mapped.backings.size() != mapped.extents.size())
    return Error("owned mapping cannot retire with pending work or managed backing");
  // Metadata for this temporary copy is held in the mapping's growth charge,
  // or its original setup construction allowance. No payload copy is made.
  auto result = std::make_shared<sc::RegionResult>();
  result->backings = mapped.backings;
  result->mapped = result->backings.size();
  std::vector<catalog::Ticket> tickets;
  tickets.reserve(mapped.extents.size());
  auto checked = RegionCatalog(
      [&]() -> Status {
        for (auto extent : mapped.extents)
          if (!catalog_.CanRetirePinned(extent)) return Error("owned mapping remains borrowed");
        for (auto extent : mapped.extents) {
          const auto ticket = catalog_.BeginRetirePinned(extent);
          if (!ticket) {
            for (auto prior : tickets) (void)catalog_.CancelEvict(prior);
            tickets.clear();
            return Error("owned mapping retirement ticket");
          }
          tickets.push_back(*ticket);
        }
        return {};
      },
      "excluding new owned-region borrowers");
  if (!checked) return checked;
  auto retired = Region({.kind = sc::RegionWork::Kind::kRetire,
                         .size = Bytes(mapped.bytes),
                         .extent_size = Bytes(kPagedExtent),
                         .offset = {},
                         .allocation_class = 0,
                         .reservation = mapped.reservation,
                         .result = result},
                        "retiring owned region");
  if (!retired) {
    auto restored = RegionCatalog(
        [&]() -> Status {
          for (auto ticket : tickets) (void)catalog_.CancelEvict(ticket);
          return {};
        },
        "undoing untouched owned-region retirement");
    base::Check(restored.has_value(), "untouched retirement must restore its tickets");
    return retired;
  }
  auto forgotten = RegionCatalog(
      [&]() -> Status {
        for (auto ticket : tickets) {
          if (!catalog_.CompleteEvict(ticket) || !catalog_.RemoveExtent(ticket.extent))
            return Error("forgetting retired owned-region extent");
        }
        EraseSpans([&](const Span& span) {
          return std::ranges::find(mapped.extents, span.extent) != mapped.extents.end();
        });
        return {};
      },
      "forgetting retired owned region");
  base::Check(forgotten.has_value(), "retired backing must leave the catalog");
  mapped.backings.clear();
  mapped.extents.clear();
  mapped.bytes = 0;
  if (auto freed = Region({.kind = sc::RegionWork::Kind::kFree,
                           .size = {},
                           .extent_size = Bytes(kPagedExtent),
                           .offset = {},
                           .allocation_class = 0,
                           .reservation = mapped.reservation,
                           .result = result},
                          "freeing retired owned reservation");
      !freed)
    return freed;  // empty reservation/local charge remain owned for retry
  mapped.reservation = {};
  UnchargeHost(mapped.metadata_bytes);
  mapped = {};
  return {};
}

std::expected<void*, std::string> PagedNode::Pinned(std::uint64_t bytes, int owner,
                                                    std::vector<ExtentId>& staging,
                                                    bool* over_budget) {
  if (over_budget) *over_budget = false;
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
        if (over_budget) *over_budget = true;
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
        if (over_budget) *over_budget = true;
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
      scheduler_ == nullptr ? register_memory() : RegionCatalog(register_memory, "pinned staging");
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
  auto changed = RegionCatalog(
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
  CollectHostReleases();
  if (bytes > UINT64_MAX - host_total_ || host_total_ + bytes > UINT64_MAX - (kPagedExtent - 1))
    return false;
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

bool PagedNode::TryChargeHost(std::uint64_t bytes) {
  if (job_active_ || torn_down_) return false;
  CollectHostReleases();
  if (bytes > UINT64_MAX - host_total_ || host_total_ + bytes > UINT64_MAX - (kPagedExtent - 1))
    return false;
  if (!Recharge(host_total_ + bytes, false, nullptr)) return false;
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

void PagedNode::CollectHostReleases() {
  if (!host_releases_) return;
  const auto bytes = host_releases_->bytes.exchange(0, std::memory_order_acq_rel);
  if (bytes != 0) UnchargeHost(bytes);
}

std::expected<HostPayment, std::string> PagedNode::PrepayHost(std::uint64_t bytes) {
  if (job_active_ || torn_down_ || !memory_)
    return Error("host result payment needs a live node outside a job");
  if (bytes == 0) return HostPayment{};
  if (!host_releases_) {
    // Allocate once, on the first nonempty result. Its original metadata
    // receipt stays with this shared owner even when a result outlives the
    // node. No node methods are needed to destroy that residual paid account.
    constexpr std::uint64_t kAllocationMetadataBytes = 64;
    constexpr auto kMetadata = sizeof(HostPayment::Releases) + kAllocationMetadataBytes;
    std::uint64_t charge = 0;
    if (__builtin_add_overflow(bytes, std::uint64_t{kMetadata}, &charge) ||
        !ChargeHost(charge, false))
      return Error("owned host result bytes do not fit");
    auto* counter = new (std::nothrow) HostPayment::Releases(kMetadata);
    if (!counter) {
      UnchargeHost(charge);
      return Error("allocating paid host result release metadata failed");
    }
    host_releases_.reset(counter);
  } else if (!ChargeHost(bytes, false)) {
    return Error("owned host result bytes do not fit");
  }
  return HostPayment(host_releases_.get(), bytes);
}

void PagedNode::UnchargeHost(std::uint64_t bytes) {
  host_total_ -= std::min(bytes, host_total_);
  (void)Recharge(host_total_, true, nullptr);
}

std::expected<std::uint64_t, std::string> PagedNode::FreeBytes() {
  CollectHostReleases();
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
  catalog::Ticket ticket;
  auto checked = RegionCatalog(
      [&]() -> Status {
        const auto span = std::ranges::find(spans_, Address(pointer), &Span::base);
        if (span == spans_.end() || span->device || span->memory_class != MemoryClass::kStaging) {
          return Error("no pinned staging extent at this address");
        }
        auto retiring = catalog_.BeginRetirePinned(span->extent);
        if (!retiring) return Error("pinned staging is still held or referenced");
        ticket = *retiring;
        return {};
      },
      "checking retired pinned staging");
  if (!checked) {
    return checked;
  }
  providers::FreePinned(pointer);
  std::erase(pinned_, pointer);
  auto forgotten = RegionCatalog(
      [&]() -> Status {
        if (!catalog_.CompleteEvict(ticket) || !catalog_.RemoveExtent(ticket.extent)) {
          return Error("forgetting retired pinned staging");
        }
        EraseSpans([&](const Span& s) { return s.extent == ticket.extent; });
        return {};
      },
      "forgetting pinned staging");
  base::Check(forgotten.has_value(), "freed pinned staging must leave the catalog");
  return {};
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
  if (workspace_pageable_) {
    // Before anything pages: the workspace's places, so every closure that
    // names an extent of it can bring it in (MapWorkspace).
    if (auto r = RegisterWorkspaceSources(activations_); !r) return r;
    if (auto r = RegisterWorkspaceSources(pool_); !r) return r;
  }
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
        "{} did not finish: nothing moved, so it was cancelled (a hang); its program retired",
        what));
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
  // A page-in entry like the others (a runner's chunk submitted without
  // waiting may be the first to need an active owner's files). A refusal
  // has nothing to return here: its loads read nothing and fail.
  (void)SyncWeightFiles();
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
  if (auto files = SyncWeightFiles(); !files) {
    return files;
  }
  const std::uint64_t request = ++request_;
  if (!scheduler_->Start(request, std::move(program), 1)) {
    return Error(std::format("{} was not admitted", what));
  }
  return Await(done, what, request);
}

Status PagedNode::Load(std::vector<ExtentId> extents, std::string what,
                       std::vector<LoadStats>& log) {
  if (auto files = SyncWeightFiles(); !files) {
    return files;
  }
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
  if (auto files = SyncWeightFiles(); !files) {
    return files;
  }
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
                      std::uint32_t stream, const std::function<void()>& meanwhile,
                      JobReport* report) {
  if (report) *report = {};
  if (supplemental_) return Error("a supplemental operation has not proven retirement");
  if (job_active_) return Error("a device job cannot recursively dispatch another job");
  if (report && job) {
    job = [inner = std::move(job), report](providers::NativeStream native) mutable {
      const auto result = inner(native);
      report->callback_result = result;
      return result;
    };
  }
  job_active_ = true;
  struct ActiveJob {
    bool& active;
    ~ActiveJob() { active = false; }
  } active{job_active_};
  if (const auto open = requests_.find(stream); open != requests_.end()) {
    return Step(stream, *open->second, closure, std::move(job), what, meanwhile, report);
  }
  // A fence over nothing (TearDown's) pages nothing in.
  if (auto files = closure.extents.empty() ? Status{} : SyncWeightFiles(); !files) {
    return files;
  }
  Done done;
  Timing timing;
  const auto called = std::chrono::steady_clock::now();
  auto program =
      std::make_unique<RunProgram>(done, closure, Timed(std::move(job), stream, timing), stream);
  Status posted;
  if (report) report->retirement = JobReport::Retirement::kUnproven;
  if (meanwhile && threaded()) {
    const auto request = Submit(std::move(program));
    meanwhile();
    posted = Await(done, what, request);
  } else {
    posted = Post(std::move(program), done, what);
  }
  if (report) {
    report->retirement = done.retired.load() ? JobReport::Retirement::kRetired
                                             : JobReport::Retirement::kNotSubmitted;
    if (report->callback_result == sc::JobResult::kNotStarted)
      report->retirement = JobReport::Retirement::kNotSubmitted;
  }
  if (posted) {
    Note(stream, called, timing);
  }
  return posted;
}

Status PagedNode::CallMetadata(std::function<Status()> call, std::string_view what) {
  if (!scheduler_) return Error("metadata control needs a started node");
  Done done;
  auto program = std::make_unique<CallProgram>(done, std::move(call));
  const auto request = ++request_;
  if (threads_.empty()) {
    if (!scheduler_->Start(request, std::move(program), 1))
      return Error("metadata control was not admitted");
  } else {
    sc::Control control =
        sc::StartRequest{.request = request, .priority = 1, .program = std::move(program)};
    // Normal scheduler control admission; deliberately no SyncWeightFiles.
    // NOLINTNEXTLINE(bugprone-use-after-move): a refused post leaves control untouched
    while (scheduler_->Post(std::move(control)) == base::PushResult::kFull)
      std::this_thread::yield();
  }
  return Await(done, what, request);
}

std::expected<SupplementalCapacity, std::string> PagedNode::PrepareSupplementalCapacity(
    std::uint32_t stream, std::uint64_t max_extents,
    std::shared_ptr<ExecutionMetadataAllowance> allowance) {
  const auto found = requests_.find(stream);
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ || found == requests_.end() ||
      found->second->cancelled || found->second->unproven || found->second->done.gone.load())
    return std::unexpected("supplemental prepayment needs a completed live request boundary");
  // Pay preparation before constructing its control. Later controls borrow the
  // receipt minted inside it and make no new optional admission decision.
  if (!allowance || !allowance->Charge(SupplementalRun::kControlBytes))
    return std::unexpected("supplemental preparation control was not funded");
  struct Refund {
    std::shared_ptr<ExecutionMetadataAllowance> allowance;
    ~Refund() { allowance->Release(SupplementalRun::kControlBytes); }
  } refund{allowance};
  std::expected<SupplementalCapacity, std::string> result =
      std::unexpected("supplemental prepayment did not run");
  auto prepared = CallMetadata(
      [&]() -> Status {
        auto& open = *found->second;
        result = SupplementalRun::Prepare(catalog_, domain_, open.channel, open.done, max_extents,
                                          allowance);
        if (!result) return Error(result.error());
        return {};
      },
      "prepaying complete supplemental continuation capacity");
  if (!prepared) return std::unexpected(prepared.error());
  return result;
}

Status PagedNode::SupplementalJob(const catalog::Closure& selected, sc::DeviceJob job,
                                  std::shared_ptr<const void> payload,
                                  std::shared_ptr<ExecutionMetadataAllowance> allowance,
                                  std::string_view what, std::uint32_t stream, JobReport* report,
                                  const SupplementalCapacity& capacity) {
  if (report) *report = {};
  if (job_active_ || supplemental_ || torn_down_ || stream >= streams_.size())
    return Error("supplemental work has an invalid stream, overlapping job or unretired operation");
  const auto found = requests_.find(stream);
  if (found == requests_.end() || found->second->cancelled || found->second->unproven ||
      found->second->done.gone.load())
    return Error("supplemental work requires a live request");
  auto& open = *found->second;
  // Pay the preflight control before allocating it. The retained operation
  // charges its own Bound inside that control; this fixed overlap is separate.
  constexpr std::uint64_t control_bytes = SupplementalRun::kControlBytes;
  if (what.size() > 256 || !allowance ||
      (capacity ? !SupplementalRun::Covers(capacity, open.channel, domain_, allowance)
                : !allowance->Charge(control_bytes)))
    return Error("supplemental metadata control was not funded or its label is too long");
  struct ControlCharge {
    std::shared_ptr<ExecutionMetadataAllowance> allowance;
    bool charged;
    ~ControlCharge() {
      if (charged) allowance->Release(control_bytes);
    }
  } control_charge{allowance, !capacity};
  job_active_ = true;
  struct ActiveJob {
    bool& flag;
    ~ActiveJob() { flag = false; }
  } active{job_active_};
  Timing timing;
  const auto called = std::chrono::steady_clock::now();
  auto prepared = CallMetadata(
      [&]() -> Status {
        auto made = SupplementalRun::Create(catalog_, domain_, open.channel, open.done, selected,
                                            Timed(std::move(job), stream, timing), stream,
                                            std::move(payload), std::move(allowance), capacity);
        if (!made) return Error(made.error());
        supplemental_ = std::move(*made);
        return {};
      },
      "validating supplemental closure before file or device effects");
  if (!prepared) {
    // The control may have completed creation just before its waiter cancelled.
    // No program was offered, so destruction safely returns its counter/payment.
    supplemental_.reset();
    return prepared;
  }
  if (auto files = SyncWeightFiles(); !files) {
    supplemental_.reset();  // no selected materialization or launch was offered
    return files;
  }
  if (report) report->retirement = JobReport::Retirement::kUnproven;
  auto ran = Post(supplemental_->Program(), supplemental_->done(), what);
  if (!ran) open.cancelled = true;
  bool retired = false;
  auto proof = CallMetadata(
      [&]() -> Status {
        auto ended = supplemental_->Retire(*scheduler_);
        if (!ended) return Error(ended.error());
        retired = *ended;
        return {};
      },
      "checking supplemental operation retirement");
  if (!proof || !retired) {
    open.unproven = true;
    return Error("supplemental completion is unproven; complete payload and base lease retained");
  }
  if (report) report->retirement = JobReport::Retirement::kRetired;
  supplemental_.reset();
  wake_.Signal();
  if (ran) Note(stream, called, timing);
  return ran;
}

Status PagedNode::FinishSupplemental() {
  if (!supplemental_) return {};
  if (job_active_) return Error("supplemental retirement cannot overlap a job");
  bool retired = false;
  auto proof = CallMetadata(
      [&]() -> Status {
        auto ended = supplemental_->Retire(*scheduler_);
        if (!ended) return Error(ended.error());
        retired = *ended;
        return {};
      },
      "draining supplemental ownership");
  if (!proof || !retired)
    return Error("supplemental operations or page-ins have not proven retirement");
  supplemental_.reset();
  wake_.Signal();
  return {};
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
  if (supplemental_) return Error("an unretired supplemental operation blocks new requests");
  if (requests_.contains(stream)) {
    return Error(std::format("{}: a request is open on stream {} already", what, stream));
  }
  if (auto files = SyncWeightFiles(); !files) {
    return files;
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

std::expected<RequestExtension, std::string> PagedNode::PrepareRequestExtension(
    std::uint32_t stream, const catalog::Closure& additions, bool* over_budget) {
  if (over_budget) *over_budget = false;
  const auto found = requests_.find(stream);
  if (found == requests_.end()) return Error("request extension needs an open holder");
  OpenRequest& open = *found->second;
  if (job_active_ || supplemental_ || open.unproven || open.cancelled || open.done.gone.load() ||
      !open.channel.held.load(std::memory_order_acquire) || open.extension)
    return Error("request extension needs a healthy unextended completed boundary");
  if (additions.extents.empty()) return Error("request extension has no private additions");
  auto valid = CallMetadata(
      [&]() -> Status {
        if (auto current = CheckReplacementClosure(catalog_, domain_, open.closure); !current)
          return current;
        return CheckReplacementClosure(catalog_, domain_, additions);
      },
      "checking prepared request extension authority");
  if (!valid) return Error(valid.error());
  for (const auto& added : additions.extents) {
    const auto at = std::ranges::lower_bound(open.closure.extents, added.first, {},
                                             [](const auto& e) { return e.first; });
    if (at != open.closure.extents.end() && at->first == added.first)
      return Error("private request additions overlap the permanent holder");
  }
  constexpr std::uint64_t kAllocationMetadata = 64;
  using Entry = decltype(additions.extents)::value_type;
  std::uint64_t control = 0, installed = 0, count = 0;
  constexpr std::uint64_t kDomainNode = sizeof(decltype(additions.bytes_by_domain)::value_type) +
                                        3 * sizeof(void*) + kAllocationMetadata;
  constexpr auto kPerExtent = sizeof(Entry) + 2 * sizeof(catalog::ExtentId);
  if (__builtin_mul_overflow(std::uint64_t{additions.extents.size()}, std::uint64_t{sizeof(Entry)},
                             &control) ||
      __builtin_add_overflow(control,
                             std::uint64_t{sizeof(RequestExtension::Data)} +
                                 2 * kAllocationMetadata + RequestExtension::Data::kControlBytes,
                             &control) ||
      __builtin_add_overflow(std::uint64_t{open.closure.extents.size()},
                             std::uint64_t{additions.extents.size()}, &count) ||
      __builtin_mul_overflow(count, std::uint64_t{kPerExtent}, &installed) ||
      __builtin_add_overflow(installed, kDomainNode + 3 * kAllocationMetadata, &installed) ||
      control > PTRDIFF_MAX || installed > PTRDIFF_MAX)
    return Error("prepared request extension metadata overflows");
  const auto old_bytes = open.closure.bytes_by_domain.empty()
                             ? std::uint64_t{0}
                             : open.closure.bytes_by_domain.begin()->second.value();
  const auto added_bytes = additions.bytes_by_domain.begin()->second.value();
  std::uint64_t total = 0;
  if (__builtin_add_overflow(old_bytes, added_bytes, &total))
    return Error("prepared request extension occupancy overflows");
  auto paid = PrepayHost(control);
  if (!paid) {
    if (over_budget) *over_budget = true;
    return Error(paid.error());
  }
  auto capacity = PrepayHost(installed);
  if (!capacity) {
    if (over_budget) *over_budget = true;
    return Error(capacity.error());
  }
  auto* data = new (std::nothrow) RequestExtension::Data(
      std::move(*paid), *this, lifetime_generation_, stream, open.request, open.channel.lease);
  if (!data) return Error("allocating prepaid request extension failed");
  RequestExtension token(data->Publish(), data);
  data->installed_payment = std::move(*capacity);
  data->additions.assign(additions.extents.begin(), additions.extents.end());
  data->candidate.extents.reserve(static_cast<std::size_t>(count));
  std::ranges::merge(open.closure.extents, additions.extents,
                     std::back_inserter(data->candidate.extents));
  data->candidate.bytes_by_domain.emplace(domain_, Bytes(total));
  data->prepared.catalog_extents.reserve(static_cast<std::size_t>(count));
  data->prepared.held_extents.reserve(static_cast<std::size_t>(count));
  for (const auto& [extent, generation] : data->candidate.extents) {
    (void)generation;
    data->prepared.catalog_extents.push_back(extent);
    data->prepared.held_extents.push_back(extent);
  }
  return token;
}

Status PagedNode::ApplyRequestExtension(const RequestExtension& extension, bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (!extension || extension.data_->node != this ||
      extension.data_->generation != lifetime_generation_)
    return Error("prepared request extension belongs to another node");
  auto& data = *extension.data_;
  const auto found = requests_.find(data.stream);
  if (found == requests_.end()) return Error("prepared request extension holder ended");
  OpenRequest& open = *found->second;
  if (open.request != data.request || open.channel.lease != data.lease || job_active_ ||
      supplemental_ || open.unproven || open.cancelled || open.done.gone.load() ||
      !open.channel.held.load(std::memory_order_acquire))
    return Error("prepared request extension lost its healthy authenticated holder");
  using State = RequestExtension::Data::State;
  if (data.state == State::kActive && open.extension.data_ == &data) return {};
  if (data.state != State::kPrepared || open.extension)
    return Error("prepared request extension is closed, consumed or conflicts");
  // No snapshot prefix is restored later. This check only rejects a stale
  // PREPARE before its first publication; reverse removes private IDs in place.
  auto original = open.closure.extents.begin();
  for (const auto& entry : data.candidate.extents) {
    if (std::ranges::binary_search(data.additions, entry)) continue;
    if (original == open.closure.extents.end() || *original != entry)
      return Error("permanent request closure changed after extension preparation");
    ++original;
  }
  if (original != open.closure.extents.end())
    return Error("permanent request closure changed after extension preparation");
  auto valid =
      CallMetadata([&]() { return CheckReplacementClosure(catalog_, domain_, data.candidate); },
                   "checking prepared request extension content");
  if (!valid) return valid;
  AcquireReport acquired;
  if (auto admitted =
          Acquire(data.candidate, acquired, "admitting prepared request extension", over_budget);
      !admitted)
    return admitted;
  bool applied = false;
  auto replaced = ApplyRequestReplacement(open, data.candidate, &data.prepared,
                                          data.installed_payment, applied);
  if (applied) {
    data.state = State::kActive;
    open.extension = extension;
  }
  return replaced;
}

Status PagedNode::CloseRequestExtension(const RequestExtension& extension) {
  if (!extension || extension.data_->node != this ||
      extension.data_->generation != lifetime_generation_)
    return Error("prepared request extension belongs to another node");
  auto& data = *extension.data_;
  using State = RequestExtension::Data::State;
  if (data.state == State::kClosed) return {};
  if (data.state == State::kPrepared) {
    {
      catalog::Closure candidate;
      sc::Scheduler::LeaseReplacementStorage prepared;
      std::swap(candidate, data.candidate);
      std::swap(prepared, data.prepared);
    }
    data.installed_payment.Reset();
    data.state = State::kClosed;
    return {};
  }
  const auto found = requests_.find(data.stream);
  if (found == requests_.end()) return Error("active request extension holder ended unexpectedly");
  OpenRequest& open = *found->second;
  if (open.extension.data_ != &data || open.request != data.request ||
      open.channel.lease != data.lease || job_active_ || supplemental_ || open.unproven ||
      open.cancelled || open.done.gone.load() || !open.channel.held.load(std::memory_order_acquire))
    return Error("closing request extension needs its healthy authenticated completed holder");
  std::uint64_t removed_bytes = 0;
  const auto validate = [&]() -> Status {
    if (auto checked = CheckReplacementClosure(catalog_, domain_, open.closure); !checked)
      return checked;
    for (const auto& entry : data.additions) {
      if (!std::ranges::binary_search(open.closure.extents, entry))
        return Error("private request addition is no longer held at its prepared generation");
      const auto view = catalog_.Describe(entry.first);
      if (!view || view->state != catalog::ExtentState::kResident ||
          __builtin_add_overflow(removed_bytes, view->descriptor.size.value(), &removed_bytes))
        return Error("private request addition has invalid resident occupancy");
    }
    if (open.closure.bytes_by_domain.empty() ||
        removed_bytes > open.closure.bytes_by_domain.begin()->second.value())
      return Error("private request addition exceeds installed occupancy");
    return {};
  };
  static_assert(sizeof(validate) <= RequestExtension::Data::kValidationCaptureBytes);
  auto valid = CallMetadata(validate, "checking private request removal");
  if (!valid) return valid;
  bool applied = false;
  const auto remove = [&]() -> Status {
    auto changed = scheduler_->RemoveRequestExtents(data.request, data.lease, data.additions);
    if (!changed) return Error(sc::ToString(changed.error()));
    applied = true;
    return {};
  };
  static_assert(sizeof(remove) <= RequestExtension::Data::kRemovalCaptureBytes);
  auto removed = CallMetadata(remove, "closing a prepaid request extension");
  if (applied) {
    std::erase_if(open.closure.extents, [&](const auto& entry) {
      return std::ranges::binary_search(data.additions, entry);
    });
    if (open.closure.extents.empty())
      open.closure.bytes_by_domain.clear();
    else
      open.closure.bytes_by_domain.begin()->second =
          Bytes(open.closure.bytes_by_domain.begin()->second.value() - removed_bytes);
    data.state = State::kClosed;
    open.extension = {};
    if (!removed) open.cancelled = true;
  }
  return removed;
}

Status PagedNode::TryRefreshRequest(std::uint32_t stream, const catalog::Closure& requested,
                                    bool* over_budget) {
  if (over_budget) *over_budget = false;
  const auto found = requests_.find(stream);
  if (found == requests_.end()) return Error("request replacement needs an open owner");
  OpenRequest& open = *found->second;
  if (job_active_ || supplemental_ || open.unproven || open.cancelled || open.done.gone.load() ||
      !open.channel.held.load(std::memory_order_acquire))
    return Error("request replacement needs a healthy completed boundary");
  // Ordinary permanent growth remains canonical while a private scope is
  // active. Keep its additions protected in the SAME held closure; closing
  // later subtracts only those IDs rather than restoring an obsolete prefix.
  HostPayment merge_payment;
  catalog::Closure merged;
  const catalog::Closure* candidate = &requested;
  if (open.extension) {
    const auto& extension = *open.extension.data_;
    if (extension.state != RequestExtension::Data::State::kActive || extension.node != this ||
        extension.generation != lifetime_generation_ || extension.request != open.request ||
        extension.lease != open.channel.lease)
      return Error("request replacement lost its authenticated private extension");
    std::uint64_t missing = 0, added_bytes = 0;
    auto checked = CallMetadata(
        [&]() -> Status {
          if (auto valid = CheckReplacementClosure(catalog_, domain_, requested); !valid)
            return valid;
          for (const auto& entry : extension.additions) {
            if (!std::ranges::binary_search(open.closure.extents, entry))
              return Error("active private extension disappeared from its holder");
            const auto view = catalog_.Describe(entry.first);
            if (!view || view->descriptor.domain != domain_ ||
                view->content_generation != entry.second || view->discarded ||
                view->state != catalog::ExtentState::kResident)
              return Error("active private extension lost its resident generation");
            const auto found = std::ranges::lower_bound(requested.extents, entry.first, {},
                                                        [](const auto& e) { return e.first; });
            if (found != requested.extents.end() && found->first == entry.first) {
              if (*found != entry) return Error("permanent closure changed a private generation");
              continue;
            }
            if (__builtin_add_overflow(missing, std::uint64_t{1}, &missing) ||
                __builtin_add_overflow(added_bytes, view->descriptor.size.value(), &added_bytes))
              return Error("private extension union occupancy overflows");
          }
          return {};
        },
        "checking permanent growth beside a private extension");
    if (!checked) return checked;
    if (missing != 0) {
      constexpr std::uint64_t kAllocatorMetadata = 64;
      constexpr std::uint64_t kDomainNode =
          sizeof(decltype(requested.bytes_by_domain)::value_type) + 3 * sizeof(void*) +
          kAllocatorMetadata;
      std::uint64_t count = 0, bytes = 0, total = 0;
      const auto permanent = requested.bytes_by_domain.empty()
                                 ? std::uint64_t{0}
                                 : requested.bytes_by_domain.begin()->second.value();
      if (__builtin_add_overflow(std::uint64_t{requested.extents.size()}, missing, &count) ||
          __builtin_mul_overflow(
              count, std::uint64_t{sizeof(decltype(requested.extents)::value_type)}, &bytes) ||
          __builtin_add_overflow(bytes, kDomainNode + kAllocatorMetadata, &bytes) ||
          __builtin_add_overflow(permanent, added_bytes, &total) || bytes > PTRDIFF_MAX)
        return Error("private extension union metadata overflows");
      auto paid = PrepayHost(bytes);
      if (!paid) {
        if (over_budget) *over_budget = true;
        return Error(paid.error());
      }
      merge_payment = std::move(*paid);
      merged.extents.reserve(static_cast<std::size_t>(count));
      std::ranges::set_union(requested.extents, extension.additions,
                             std::back_inserter(merged.extents));
      merged.bytes_by_domain.emplace(domain_, Bytes(total));
      candidate = &merged;
    }
  }
  const catalog::Closure& closure = *candidate;
  // A node admits one physical domain. Validate source facts before any
  // payment or materialization, including the no-reclaimer acquisition path.
  // The supplied occupancy map is a consistency check, never authority.
  auto valid = CallMetadata([&]() { return CheckReplacementClosure(catalog_, domain_, closure); },
                            "checking replacement request authority");
  if (!valid) return valid;
  if (open.closure.extents == closure.extents) return {};
  // The three new vectors (driver closure, catalog lease and Held), the
  // closure's domain-map nodes, and named allocation headers.
  // Old capacities remain paid until the successful replacement destroys them.
  constexpr std::uint64_t kAllocationMetadata = 64;
  constexpr std::uint64_t kPerExtent =
      sizeof(decltype(closure.extents)::value_type) + 2 * sizeof(catalog::ExtentId);
  constexpr std::uint64_t kDomainNode = sizeof(decltype(closure.bytes_by_domain)::value_type) +
                                        3 * sizeof(void*) + kAllocationMetadata;
  std::uint64_t bytes = 0, domains = 0;
  if (__builtin_mul_overflow(std::uint64_t{closure.extents.size()}, kPerExtent, &bytes) ||
      __builtin_mul_overflow(std::uint64_t{closure.bytes_by_domain.size()}, kDomainNode,
                             &domains) ||
      __builtin_add_overflow(bytes, domains, &bytes) ||
      __builtin_add_overflow(bytes, 3 * kAllocationMetadata, &bytes))
    return Error("request replacement metadata extent overflows");
  auto payment = PrepayHost(bytes);
  if (!payment) {
    if (over_budget) *over_budget = true;
    return Error(payment.error());
  }
  catalog::Closure next = closure;
  std::ranges::sort(next.extents);
  AcquireReport acquired;
  if (auto admitted = Acquire(next, acquired, "admitting a replacement request", over_budget);
      !admitted)
    return admitted;
  bool applied = false;
  return ApplyRequestReplacement(open, next, nullptr, *payment, applied);
}

Status PagedNode::ApplyRequestReplacement(OpenRequest& open, catalog::Closure& next,
                                          sc::Scheduler::LeaseReplacementStorage* prepared,
                                          HostPayment& payment, bool& applied) {
  // Canonical Held::Drained checks scheduled operations and direct external
  // work. Neither a result wake nor a step count is retirement proof.
  const auto replace = [&]() -> Status {
    auto result =
        prepared
            ? scheduler_->ReplaceRequestLease(open.request, open.channel.lease, next, *prepared)
            : scheduler_->ReplaceRequestLease(open.request, open.channel.lease, next);
    if (!result) return Error(sc::ToString(result.error()));
    applied = true;
    return {};
  };
  static_assert(sizeof(replace) <= RequestExtension::Data::kReplacementCaptureBytes);
  auto replaced = CallMetadata(replace, "replacing a completed request lease");
  if (applied) {
    {
      auto previous = std::move(open.closure);
      open.closure = std::move(next);
    }
    if (prepared) {
      sc::Scheduler::LeaseReplacementStorage previous;
      std::swap(previous, *prepared);
    }  // All old driver/catalog/Held capacities precede their old refund.
    open.replacement_payment = std::move(payment);
    if (!replaced) open.cancelled = true;
  }
  return replaced;
}

Status PagedNode::Step(std::uint32_t stream, OpenRequest& open, const catalog::Closure& closure,
                       sc::DeviceJob job, std::string_view what,
                       const std::function<void()>& meanwhile, JobReport* report) {
  // Admission refusal is independent of draining a request at its end. Keep
  // the complete request/lease owner when its original work is unproven.
  if (open.unproven) return Error(std::format("{}: its request retains unproven work", what));
  if (open.cancelled || open.done.gone.load())
    return Ended(stream, what, "the request cannot continue after failed work");
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
    return DirectStep(stream, open, std::move(job), what, meanwhile, report);
  }
  Timing timing;
  open.channel.job = Timed(std::move(job), stream, timing);
  if (report) report->retirement = JobReport::Retirement::kUnproven;
  open.channel.stream = stream;
  const std::uint64_t before = open.channel.steps.load(std::memory_order_acquire);
  const auto called = std::chrono::steady_clock::now();
  Signal(open.request);
  if (meanwhile && threaded()) meanwhile();
  // The step's result is the next step's input, and a sleeping thread
  // wakes slowly on the Spark (RE-017), so the driver waits as the
  // runtime's lanes do (docs/experiments/runtime-wake/): asleep through
  // most of the step, spinning around its likely ends (the last few steps'
  // walls on this stream, whichever request ran them), woken early by the
  // task's report whenever it sleeps.
  const auto spin_ahead = settings_.spin_ahead.value_or(kSpinAhead);
  base::Expectation& walls = walls_[stream];
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
      const auto next = walls.Next(now - called, kSpinPast);
      if (threads_.empty()) {
        Round();
      } else if (walls.known() && !next) {
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
  walls.Add(std::chrono::steady_clock::now() - called);
  if (open.channel.steps.load(std::memory_order_acquire) == before) {
    // The request's task ended (cancelled, or failed): nothing holds the
    // job, which never ran, any more.
    open.channel.job = nullptr;
    const std::uint64_t counted = hang_cancels_;
    auto ended = Await(open.done, open.what, open.request);
    if (report && open.done.retired.load()) report->retirement = JobReport::Retirement::kRetired;
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
  // The acquire read of the new steps count above orders these immutable
  // facts for this dispatch. A result wake can also report quarantined work.
  const bool retired = open.channel.step_retired.load(std::memory_order_relaxed);
  if (report) {
    report->retirement =
        retired ? JobReport::Retirement::kRetired : JobReport::Retirement::kUnproven;
    if (retired && (open.channel.step_error.load(std::memory_order_relaxed) >= 0 ||
                    report->callback_result == sc::JobResult::kNotStarted))
      report->retirement = JobReport::Retirement::kNotSubmitted;
  }
  if (!retired) {
    open.unproven = true;
    return Error(std::format("{}: its step's end could not be proven", what));
  }
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
  // As EndRequest: a task idle under an unproven step waits for a signal
  // nothing sends, so it is cancelled (its lease stays held by the step).
  if (open.unproven || open.cancelled) {
    Cancel(open.request);
  }
  auto ended = Await(open.done, open.what, open.request);
  requests_.erase(found);  // `open` is gone from here on
  return Error(std::format("{}: {}{}", what, why, ended ? "" : ": " + ended.error()));
}

Status PagedNode::DirectStep(std::uint32_t stream, OpenRequest& open, sc::DeviceJob job,
                             std::string_view what, const std::function<void()>& meanwhile,
                             JobReport* report) {
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
  if (report) report->retirement = JobReport::Retirement::kUnproven;
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
  if (report && !queued) report->retirement = JobReport::Retirement::kNotSubmitted;
  if (!fence && queued) {
    // Nothing can show when the queued work ends: its lease is kept.
    open.unproven = true;
    return Error(
        std::format("{}: its step's fence could not be recorded: {}", what, fence.error().detail));
  }
  if (meanwhile && fence && threaded()) meanwhile();
  // The step's result is the next step's input, and a sleeping thread
  // wakes slowly on the Spark (RE-017): asleep through most of the step,
  // spinning around its likely ends (the last few steps' walls on this
  // stream, whichever request ran them: a refreshed lease or a new turn's
  // request keeps them), as the lanes do (docs/experiments/runtime-wake/).
  const auto spin_ahead = settings_.spin_ahead.value_or(kSpinAhead);
  base::Expectation& walls = walls_[stream];
  std::chrono::steady_clock::duration nap = kFirstNap;
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
      const auto next = walls.Next(now - called, kSpinPast);
      if (threads_.empty()) {
        Round();
      } else if (next && now < called + *next - spin_ahead) {
        std::this_thread::sleep_until(std::min(called + *next - spin_ahead, now + kDirectNap));
      } else if (walls.known() && !next) {
        std::this_thread::sleep_for(kLateSleep);  // longer than any known: late
      } else if (!walls.known() && now >= called + kFirstSpin) {
        // The stream's first step, long: rather than query its fence back
        // to back throughout, back off.
        std::this_thread::sleep_for(nap);
        nap = std::min<std::chrono::steady_clock::duration>(nap * 2, kDirectNap);
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
  if (report && queued) report->retirement = JobReport::Retirement::kRetired;
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
  walls.Add(ended);
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
    if (open.unproven) {
      // The canonical Held/catalog inventories still own these capacities.
      // Keep their payment and poisoned admission owner until true drain;
      // task retirement alone cannot refund memory retained by external work.
      return Error("the request's step never proved its end; its lease is kept");
    }
    requests_.erase(found);
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

std::expected<sc::FileId, std::string> PagedNode::AddWeightFile(int owner, WeightFileOpener open) {
  if (scheduler_ == nullptr) {
    return Error("a weight file is registered after the node's Start");
  }
  const sc::FileId id = scheduler_->files().Add();
  weight_files_.push_back({.owner = owner, .id = id, .open = std::move(open), .fd = -1});
  // Closed until something may page in (its owner activated, or the next
  // page-in entry while it is active).
  files_unsettled_ = true;
  return id;
}

Status PagedNode::OpenWeightFiles(int owner) {
  for (WeightFile& file : weight_files_) {
    if (file.owner != owner || file.fd >= 0) {
      continue;
    }
    // On the driver, outside the scheduler: the open may wait on the drive.
    auto opened = file.open();
    if (!opened) {
      // All or none: what this call opened closes again, each once no
      // read uses it (one may have begun meanwhile, for a program the
      // driver submitted without waiting).
      inactive_.insert(owner);
      files_unsettled_ = !CloseWeightFiles(false) || files_unsettled_;
      return Error(opened.error());
    }
    base::Check(scheduler_->files().Install(file.id, *opened).has_value(),
                "installing a weight file that is open already");
    file.fd = *opened;
  }
  return {};
}

bool PagedNode::CloseWeightFiles(bool every) {
  bool settled = true;
  for (WeightFile& file : weight_files_) {
    if (file.fd < 0 || (!every && active(file.owner))) {
      continue;
    }
    // Taken back only once no read that began on it is in flight; it is
    // then out of the table, so no read can take it before it closes.
    if (!scheduler_->files().Take(file.id)) {
      settled = false;
      continue;
    }
    (void)::close(std::exchange(file.fd, -1));
  }
  return settled;
}

Status PagedNode::SetActive(int owner, bool on) {
  if (!on) {
    inactive_.insert(owner);
    files_unsettled_ = !CloseWeightFiles(false) || files_unsettled_;
    return {};
  }
  inactive_.erase(owner);
  return OpenWeightFiles(owner);
}

Status PagedNode::SyncWeightFiles() {
  if (!files_unsettled_) {
    return {};
  }
  std::set<int> owners;
  for (const WeightFile& file : weight_files_) {
    if (file.fd < 0 && active(file.owner)) {
      owners.insert(file.owner);
    }
  }
  for (const int owner : owners) {
    if (auto opened = OpenWeightFiles(owner); !opened) {
      return opened;  // that owner inactive now; the rest settle at the next entry
    }
  }
  files_unsettled_ = !CloseWeightFiles(false);
  return {};
}

int PagedNode::WeightFileDescriptor(sc::FileId file) const {
  return scheduler_ != nullptr ? scheduler_->files().Descriptor(file) : -1;
}

std::size_t PagedNode::open_weight_files(std::optional<int> owner) const {
  return static_cast<std::size_t>(std::ranges::count_if(weight_files_, [&](const WeightFile& f) {
    return f.fd >= 0 && (!owner || f.owner == *owner);
  }));
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
  if (supplemental_) return Error("an unretired supplemental operation blocks acquisition");
  if (job_active_) return Error("an acquisition cannot run while a device job is in flight");
  if (auto files = SyncWeightFiles(); !files) return files;
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

void PagedNode::KeepModelRoot(RetainedModelRoot& root) {
  base::Check(root.model && root.keep && !torn_down_,
              "a paid runner root attaches to its live node");
  for (auto* saved = model_roots_; saved; saved = saved->next)
    base::Check(saved != &root && saved->model != root.model, "a runner root attaches once");
  root.next = model_roots_;
  model_roots_ = &root;
}

void PagedNode::RetireModelRoot(RetainedModelRoot& root) {
  auto** saved = &model_roots_;
  while (*saved && *saved != &root) saved = &(*saved)->next;
  base::Check(*saved == &root, "an authenticated retired runner detaches its actual root once");
  *saved = root.next;
  root.next = nullptr;
  auto keep = std::move(root.keep);
  // The visiting teardown reference outlives this callback, including when
  // the frontend and private view already dropped every other root reference.
}

Status PagedNode::TearDown(std::span<PagedModel* const> models) {
  if (torn_down_) {
    return {};
  }
  if (job_active_) return Error("node teardown cannot overlap a device job");
  // Before torn_down_ or any source/device release. False work status or gone
  // alone must never free a withdrawn materialization's payload owner.
  if (auto drained = FinishSupplemental(); !drained) return drained;
  std::vector<std::string> problems;
  // A withdrawn acquisition can be gone before its page-ins retire. Keep
  // the complete LiveState owner and node alive on an unproven scoped drain.
  while (preparing_state_ != nullptr) {
    auto* owner = preparing_state_;
    if (auto ended = owner->FinishPreparation(true); !ended) {
      if (preparing_state_ == owner) return Error(ended.error());
      problems.push_back(ended.error());
    }
  }
  // Do not mark teardown complete or invalidate staging until actual fences
  // and every model's retained-resource check permit destruction. A refusal
  // here must allow dropping an external view and retrying the same node.
  auto visit_models = [&](auto&& visit) -> Status {
    for (std::size_t i = 0; i < models.size(); ++i) {
      auto* model = models[i];
      if (!model) return Error("node teardown received an absent model");
      bool registered = false;
      for (auto* root = model_roots_; root; root = root->next)
        registered = registered || root->model == model;
      if (registered || std::find(models.begin(), models.begin() + static_cast<std::ptrdiff_t>(i),
                                  model) != models.begin() + static_cast<std::ptrdiff_t>(i))
        continue;
      if (auto visited = visit(*model); !visited) return visited;
    }
    for (auto* root = model_roots_; root;) {
      auto* next = root->next;
      auto keep = root->keep;  // Node's actual complete owner survives Release's detach.
      if (auto visited = visit(*root->model); !visited) return visited;
      root = next;
    }
    return {};
  };
  auto retire_models = [&]() -> Status {
    return visit_models([](PagedModel& model) { return model.RetireAfterFence(); });
  };
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
    auto collected = visit_models([&](PagedModel& model) -> Status {
      if (usable) {
        if (auto fenced = Job(
                nothing, [](providers::NativeStream) { return sc::JobResult::kQueued; },
                "fencing a compute stream", model.stream());
            !fenced) {
          problems.push_back(std::format("compute stream {} could not be fenced: {}",
                                         model.stream(), fenced.error()));
          fenced_all = false;
        }
      }
      const auto extents = model.managed_extents();
      managed.insert(managed.end(), extents.begin(), extents.end());
      return {};
    });
    if (!collected) return collected;
    if (fenced_all) {
      if (auto retired = retire_models(); !retired) return retired;
    }
    if (workspace_pageable_) {
      // Its backing is the VMM lane's, released through its eviction too.
      const auto workspace = workspace_extents();
      managed.insert(managed.end(), workspace.begin(), workspace.end());
    }
    if (fenced_all && !Evict(managed)) {
      problems.emplace_back("the weights and the workspace could not be evicted at the end");
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
    // Every read has concluded but those whose completion is unproven,
    // whose files stay open to the process's end.
    (void)CloseWeightFiles(true);
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
  if (scheduler_ == nullptr) {
    if (auto retired = retire_models(); !retired) return retired;
  }
  if (auto released = visit_models([](PagedModel& model) { return model.Release(); }); !released)
    return released;
  if (staging_lifetime_) staging_lifetime_->live_ = false;
  staging_lifetime_.reset();
  torn_down_ = true;
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

}  // namespace llmp::engine
