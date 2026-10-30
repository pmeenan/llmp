// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/execution_contract.h"
#include "engine/paged_node_internal.h"

namespace llmp::engine {

WaitVerdict QuietPatience::Check(const WaitState& wait, std::uint64_t /*progress*/) {
  const auto now = std::chrono::steady_clock::now();
  if (!wait.cancelled) {
    return now - wait.progressed >= quiet_ ? WaitVerdict::kCancel : WaitVerdict::kWait;
  }
  return now - std::max(wait.progressed, *wait.cancelled) >= quiet_ ? WaitVerdict::kGiveUp
                                                                    : WaitVerdict::kWait;
}

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

Status PagedNode::Open() {
  if (memory_ || execution_ || torn_down_) return Error("the node cannot be opened twice");
  auto device = providers::OpenDevice(
      settings_.device_ordinal,
      {.fences_ahead = kEventsAhead, .fences_kept = std::max<std::size_t>(kEventsAhead, 4096)});
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
  // Fixed node runtime metadata is part of the startup occupancy, before
  // choosing/filling the pool and before BackingService constructs its ledger.
  // The existing metadata owner also pays its own catalog/control overhead.
  const auto pool_bytes = sc::HandleReserve::MetadataBytes();
  auto pool_metadata = ExecutionMetadataAllowance::AdmitInBudget(
      catalog_, domain_, ExecutionMetadataAllowance::OverheadBytes() + pool_bytes, UINT64_MAX,
      nullptr);
  if (!pool_metadata) return Error(pool_metadata.error());
  pool_metadata_ = std::move(*pool_metadata);
  base::Check(pool_metadata_->Charge(pool_bytes), "fixed pool metadata was prepaid");
  // The zone first: a persistent pool, mapped before anything pages.
  return MapResident(zone_, "the landing zone", settings_.slots * settings_.slot_bytes,
                     providers::BackingKind::kHost, MemoryClass::kStaging,
                     catalog::Recovery::kPinned, kShared);
}

base::Bytes PagedNode::budget() const noexcept {
  return scheduler_ != nullptr ? scheduler_->budget() : budget_;
}

std::expected<base::Bytes, std::string> PagedNode::SetBudget(base::Bytes budget) {
  if (torn_down_) return Error("the node is torn down");
  if (scheduler_ == nullptr) {
    budget_ = budget;
    return budget;
  }
  // Without the scheduler's thread the catalog is this one's.
  if (threads_.empty()) return ApplyBudget(budget);
  base::Bytes applied;
  if (auto set = Call(
          [&]() -> Status {
            applied = ApplyBudget(budget);
            return {};
          },
          "setting the execution budget");
      !set) {
    return std::unexpected(set.error());
  }
  return applied;
}

base::Bytes PagedNode::ApplyBudget(base::Bytes budget) {
  if (scheduler_ == nullptr) {
    budget_ = budget;
    return budget;
  }
  return scheduler_->SetBudget(budget, domain_);
}

void PagedNode::KeepPinned(void* pointer, std::optional<std::uint32_t> stream) {
  if (const auto at = std::ranges::find(pinned_, pointer); at != pinned_.end()) {
    pinned_.erase(at);
    quarantine_.push_back({.pointer = pointer, .stream = stream, .ring = false});
  }
  if (stream.has_value() &&
      std::ranges::find(unproven_streams_, *stream) == unproven_streams_.end())
    unproven_streams_.push_back(*stream);
}

bool PagedNode::Quarantining(std::uint32_t stream) const noexcept {
  return std::ranges::find(unproven_streams_, stream) != unproven_streams_.end() ||
         std::ranges::any_of(quarantine_,
                             [stream](const Quarantined& q) { return q.stream == stream; });
}

bool PagedNode::unproven() const noexcept {
  return supplemental_ != nullptr || !unproven_streams_.empty() ||
         std::ranges::any_of(quarantine_, [](const Quarantined& q) { return !q.ring; }) ||
         std::ranges::any_of(requests_, [](const auto& kv) { return kv.second->unproven; });
}

Status PagedNode::ReleaseQuarantine(std::optional<std::uint32_t> stream) {
  std::vector<std::string> errors;
  std::vector<Quarantined> kept;
  for (Quarantined& q : quarantine_) {
    if (q.ring || (stream.has_value() && q.stream != stream)) {
      kept.push_back(q);
      continue;
    }
    if (auto released = ReleaseQuarantinedPinned(q.pointer); !released) {
      errors.push_back(released.error());
      kept.push_back(q);  // still the node's, still out of use
    }
  }
  quarantine_ = std::move(kept);
  if (stream.has_value()) {
    std::erase(unproven_streams_, *stream);
  } else {
    unproven_streams_.clear();
  }
  return errors.empty() ? Status{} : Joined(errors);
}

Status PagedNode::ReleaseQuarantinedPinned(void* pointer) {
  if (pointer == nullptr) {
    return {};
  }
  const auto mutate = [&](auto&& fn, std::string_view what) -> Status {
    if (scheduler_ != nullptr && !stopped_.has_value()) {
      return RegionCatalog(std::forward<decltype(fn)>(fn), what);
    }
    return fn();
  };
  catalog::Ticket ticket;
  auto checked = mutate(
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
  auto forgotten = mutate(
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
      quarantine_.push_back({.pointer = landing, .stream = std::nullopt, .ring = true});
    }
  }
  // Destroying a ring with reads in flight is fatal (UringStorage), and
  // closing it would not stop them landing: it is never destroyed, nor are
  // its landings freed.
  retired_rings_.push_back(ring.release());
  base::RetainForProcess(retired_rings_.back());
  for (void* const landing : landings) base::RetainForProcess(landing);
  return true;
}

Status PagedNode::Start(Bytes budget, std::optional<NodePool> pool) {
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
  // calls the device-memory provider (device_memory.h). Its handle pool
  // stays within the budget's headroom: the scheduler rebalances it each
  // turn (SchedulerSettings::pool_domain).
  const std::uint64_t occupied = catalog_.OccupancyOf(domain_).Total().value();
  const std::size_t headroom = static_cast<std::size_t>(
      budget.value() > occupied ? (budget.value() - occupied) / kPagedExtent : 0);
  const std::size_t ceiling = memory_ == nullptr ? 0
                              : pool.has_value()
                                  ? static_cast<std::size_t>(pool->ceiling.value() / kPagedExtent)
                                  : settings_.handle_reserve;
  sc::ReserveSettings reserve{
      .allocation_class = device_class_, .size = Bytes(kPagedExtent), .count = ceiling};
  if (!pool_metadata_ || !pool_metadata_->HeldBy(catalog_, domain_) ||
      pool_metadata_->used() < sc::HandleReserve::MetadataBytes())
    return Error("the backing lane's transfer ledger has no fixed metadata payment");
  backing_lane_ = std::make_unique<sc::BackingService>(
      memory_.get(), *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32},
      reserve, settings_.diagnostic_backing_timing);
  pool_shortfall_.clear();
  std::size_t filled = 0;
  if (ceiling > 0) {
    const std::size_t aim = std::min(ceiling, headroom);
    if (pool.has_value() && pool->fill) {
      std::string failure;
      filled = backing_lane_->Preallocate(aim, &failure);
      if (filled < aim) {
        pool_shortfall_ = std::format("{} of {} handles created ({})", filled, aim,
                                      failure.empty() ? "no reason given" : failure);
      }
    }
    backing_lane_->ManageReserve(aim > filled ? aim - filled : 0);
  }
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
                                   .observer = settings_.observer,
                                   .pool_domain = domain_};
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
  backing_lane_->Starting();  // before its thread: a drain from now on waits for it
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

Status PagedNode::FenceStream(std::uint32_t stream) {
  if (!scheduler_ || torn_down_) return Error("a stream fence needs a started node");
  if (job_active_) return Error("a stream fence cannot overlap a device job");
  // A Job refuses while a supplemental operation is unretired, and only a
  // direct step's own external count is the stream fence's to settle.
  if (auto drained = FinishSupplemental(); !drained) return drained;
  if (const auto open = requests_.find(stream);
      open != requests_.end() && open->second->supplemental_unproven) {
    // Its supplemental operation proved its end just now (the node holds
    // one at a time), which balanced its own count: only the rest remains.
    auto& request = *open->second;
    request.supplemental_unproven = false;
    if (request.direct_unproven == 0 && !request.other_unproven) {
      request.unproven = false;
      request.cancelled = true;
    }
  }
  if (const auto open = requests_.find(stream);
      open != requests_.end() && open->second->other_unproven) {
    // A scheduled step's end went unproven on the device lane, which
    // quarantined its operation and faulted the scheduler: no fence of the
    // driver's proves it, and only a restart (D-102's rung 3) recovers.
    return Error("a scheduled step's end is unproven: its scheduler faulted");
  }
  if (const auto open = requests_.find(stream);
      open != requests_.end() && open->second->unproven && open->second->direct_unproven != 0) {
    if (auto proven = ProveDirectStep(stream, *open->second); !proven) return proven;
  }
  if (InRequest(stream)) (void)EndRequest(stream);
  if (InRequest(stream)) return Error("a request is still open on the stream");
  // Withdrawn preparations would otherwise outlive the work that asked for
  // them.
  while (preparing_state_ != nullptr) {
    auto* owner = preparing_state_;
    if (auto ended = owner->FinishPreparation(true); !ended && preparing_state_ == owner)
      return Error(ended.error());
  }
  auto fenced = Job(
      catalog::Closure{}, [](providers::NativeStream) { return sc::JobResult::kQueued; },
      "fencing a model's stream after a failure", stream);
  if (!fenced) return fenced;
  // Every allocation the stream's work may have touched is tried, and what
  // cannot be freed stays quarantined (never dropped untracked).
  auto released = ReleaseQuarantine(stream);
  // Checkpoint staging an unproven copy took is reserved again once freed,
  // so a full budget still never starves a checkpoint.
  RestoreStaging();
  return released;
}

Status PagedNode::RecoverAfterFence(PagedModel& model) {
  const std::uint32_t stream = model.stream();
  if (auto fenced = FenceStream(stream); !fenced) return fenced;
  const FenceProof proof(stream);
  // Private batches first: their close clears the runner's active batch.
  std::vector<std::string> errors;
  for (auto* root = model_roots_; root;) {
    auto* next = root->next;
    auto keep = root->keep;  // Close may detach the root
    if (root->model != &model && root->model->stream() == stream) {
      if (auto acknowledged = root->model->AcknowledgeFence(proof); !acknowledged)
        errors.push_back(acknowledged.error());
    }
    root = next;
  }
  if (!errors.empty()) return Joined(errors);
  return model.AcknowledgeFence(proof);
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
    std::vector<std::string> errors;
    for (std::size_t i = 0; i < models.size(); ++i) {
      auto* model = models[i];
      if (!model) {
        errors.push_back("node teardown received an absent model");
        continue;
      }
      bool registered = false;
      for (auto* root = model_roots_; root; root = root->next)
        registered = registered || root->model == model;
      if (registered || std::find(models.begin(), models.begin() + static_cast<std::ptrdiff_t>(i),
                                  model) != models.begin() + static_cast<std::ptrdiff_t>(i))
        continue;
      if (auto visited = visit(*model); !visited) errors.push_back(visited.error());
    }
    for (auto* root = model_roots_; root;) {
      auto* next = root->next;
      auto keep = root->keep;  // Node's actual complete owner survives Release's detach.
      if (auto visited = visit(*root->model); !visited) errors.push_back(visited.error());
      root = next;
    }
    return errors.empty() ? Status{} : Joined(errors);
  };
  // Called only once each model's stream was fenced: its proof.
  auto retire_models = [&]() -> Status {
    return visit_models(
        [](PagedModel& model) { return model.RetireAfterFence(FenceProof(model.stream())); });
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
    // The lanes have joined: every stream, each model's (listed or not) and
    // the copy stream, fenced by the driver alone before the quarantine's
    // untagged allocations (whose stream nobody knew) may go.
    for (const auto id : streams_) {
      if (id.valid() && !Fence(*execution_, id, settings_.quiet)) {
        problems.emplace_back("a stream could not be fenced: the quarantine is left as it is");
        return Joined(problems);
      }
    }
  } else if (execution_ != nullptr) {
    // Setup ended before the scheduler existed, and a model may already
    // have queued work on its stream (the EXL3 launch context zeroes its
    // lock area): each compute stream is fenced before any memory goes.
    for (std::size_t i = 0; i < streams_.size(); ++i) {
      if (streams_[i].valid() && !Fence(*execution_, streams_[i], settings_.quiet)) {
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
      if (!::llmp::engine::ReleaseMapped(*memory_, *mapped)) {
        problems.push_back(std::format("{} could not be released", mapped->name));
      }
    }
    if (memory_->backings() != 0) {
      problems.push_back(std::format("{} backings were left", memory_->backings()));
    }
  }
  // Every stream was fenced above: the whole quarantine but rings' landings.
  if (auto released = ReleaseQuarantine(std::nullopt); !released) {
    problems.push_back(released.error());
  }
  for (void* pointer : pinned_) {
    providers::FreePinned(pointer);
  }
  pinned_.clear();
  return Joined(problems);
}

bool ReleaseMapped(providers::VmmProvider& memory, Mapped& mapped) {
  if (!mapped.reservation.valid()) {
    return true;
  }
  const bool unmapped =
      mapped.backings.empty() ||
      memory.Unmap(mapped.reservation, Bytes(0), Bytes(mapped.backings.size() * kPagedExtent))
          .has_value();
  if (!unmapped) {
    return false;
  }
  bool released = true;
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

}  // namespace llmp::engine
