// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_node_internal.h"

namespace llmp::engine {

PagedNode::OpenRequest::~OpenRequest() {
  // This invalidates only metadata/holder identity, never certifies native
  // work. A private batch independently retains every unproven child owner.
  if (extension) extension.data_->state = RequestExtension::Data::State::kClosed;
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
  std::vector<ExtentId> protect;
  memory::MaterializationPlan plan;
  auto prepared = CallMetadata(
      [&]() -> Status {
        auto made = SupplementalRun::Create(catalog_, domain_, open.channel, open.done, selected,
                                            Timed(std::move(job), stream, timing), stream,
                                            std::move(payload), std::move(allowance), capacity);
        if (!made) return Error(made.error());
        supplemental_ = std::move(*made);
        protect = workspace_extents();
        for (const auto& [extent, generation] : open.closure.extents) {
          (void)generation;
          protect.push_back(extent);
        }
        for (const auto& [extent, generation] : selected.extents) {
          (void)generation;
          protect.push_back(extent);
        }
        plan = memory::PlanMaterialization(catalog_, domain_, budget(), selected, protect,
                                           !reclaimer_);
        return {};
      },
      "validating supplemental closure before file or device effects");
  if (!prepared) {
    // The control may have completed creation just before its waiter cancelled.
    // No program was offered, so destruction safely returns its counter/payment.
    supplemental_.reset();
    return prepared;
  }
  if (plan.shortfall.value() != 0) {
    for (int round = 0; round < memory::kReclaimRounds; ++round) {
      if (reclaimer_) {
        job_active_ = false;
        const auto reclaimed =
            AskReclaim(plan.shortfall.value(), ReclaimFor::kMaterialize, protect);
        job_active_ = true;
        if (reclaimed == 0) break;
      } else {
        if (!plan.victims.sufficient || plan.victims.victims.empty()) break;
        std::vector<ExtentId> victim_extents;
        victim_extents.reserve(plan.victims.victims.size());
        for (const auto& v : plan.victims.victims) {
          victim_extents.push_back(v.extent);
        }
        job_active_ = false;
        auto evicted = Evict(std::move(victim_extents));
        job_active_ = true;
        if (!evicted) break;
      }
      prepared = CallMetadata(
          [&]() -> Status {
            if (!open.channel.held.load(std::memory_order_acquire) || open.done.gone.load() ||
                open.done.outcome.load() != -1 || open.channel.end) {
              return Error("supplemental work requires a live request");
            }
            plan = memory::PlanMaterialization(catalog_, domain_, budget(), selected, protect,
                                               !reclaimer_);
            if (!plan.stale.empty() || !plan.quarantined.empty()) {
              return Error("supplemental closure is unsorted, foreign, stale or quarantined");
            }
            return {};
          },
          "re-evaluating supplemental closure after reclaim");
      if (!prepared) {
        supplemental_.reset();
        return prepared;
      }
      if (plan.shortfall.value() == 0) break;
    }
    if (plan.shortfall.value() != 0) {
      supplemental_.reset();
      return Error("the supplemental closure remains over budget");
    }
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
    open.supplemental_unproven = true;
    unproven_.Mark();
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
    open.other_unproven = true;
    unproven_.Mark();
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

Status PagedNode::ProveDirectStep(std::uint32_t stream, OpenRequest& open) {
  const providers::StreamId id = streams_[stream];
  const auto fence = execution_->Record(id);
  if (!fence) {
    return Error(
        std::format("an unproven step's stream could not be fenced: {}", fence.error().detail));
  }
  bool seen = false;
  {
    Waiter wait(*this, "proving an unproven step's end");
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
        break;  // a device fault, or a refusal that persists: still unproven
      }
      // Its request is cancelled already: a hang here has nothing more to
      // cancel, and giving up is the patience's last resort.
      if (wait.Poll(std::chrono::steady_clock::now()) == WaitVerdict::kGiveUp) {
        GiveUp("proving an unproven step's end");
      }
      if (threads_.empty()) {
        Round();
      } else {
        std::this_thread::sleep_for(kDirectNap);
      }
    }
  }
  if (!seen) return Error("an unproven step's stream fence was not seen complete");
  for (std::uint32_t tries = 0; tries < kFenceRefusals; ++tries) {
    const auto released = execution_->Release(*fence);
    if (released || released.error().error != providers::ProviderError::kFailed) break;
  }
  // DirectStep counted each such step's external work and never saw it end.
  open.channel.external->fetch_sub(open.direct_unproven, std::memory_order_acq_rel);
  open.direct_unproven = 0;
  open.unproven = false;
  open.cancelled = true;
  wake_.Signal();
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
    ++open.direct_unproven;
    unproven_.Mark();
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
    ++open.direct_unproven;
    unproven_.Mark();
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
                plan = memory::PlanMaterialization(catalog_, domain_, budget(), closure, protect,
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
  auto program = std::make_unique<AcquireProgram>(
      done, closure, domain_, report, std::move(workspace), !reclaimer_, eviction_receipt_);
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

}  // namespace llmp::engine
