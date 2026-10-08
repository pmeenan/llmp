// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A request's lease (M3's lease per request; scheduler.h HoldLease,
// tests/support/paged_programs.h RequestProgram) on the deterministic
// fakes: two models' weights with managed backing and a workspace both
// closures share, one scheduler, one landing zone. A request materializes
// its model's closure and leases it once; each step is device work under
// that lease, which is neither taken nor released per step and still ends
// on its fence. While it is held no eviction of its extents can begin, and
// a swap waits for it; ending it (or cancelling the request, even with a
// step in flight) releases it only once no step can touch the memory, and
// leaves every extent resident (D-007: release changes eligibility, not
// residency). A step whose completion is unknown keeps it for good.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "expected_error.h"
#include "paged_programs.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_memory.h"
#include "providers/fake/fake_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"
#include "scheduler/tasks.h"

namespace {

namespace sc = llmp::scheduler;
namespace ts = llmp::test_support;
using llmp::base::Bytes;
using llmp::catalog::Closure;
using llmp::catalog::ExtentId;
using llmp::catalog::ExtentState;
using llmp::catalog::ExtentView;
using llmp::catalog::LeaseId;
using llmp::providers::ProviderError;
using llmp::providers::StreamId;
using llmp::test_support::Failed;

constexpr std::uint64_t kSize = 64ULL * 1024;  // one extent, one slot
constexpr std::size_t kPerModel = 3;
constexpr std::size_t kSlots = 2;

constexpr auto kPatience = std::chrono::seconds(120);  // a hang, not a timing

// Posts a control, retrying while the queue is full.
void Offer(sc::Scheduler& scheduler, sc::Control&& control) {
  // NOLINTNEXTLINE(bugprone-use-after-move): a refused control is untouched
  while (scheduler.Post(std::move(control)) == llmp::base::PushResult::kFull) {
    std::this_thread::yield();
  }
}

// A job that queues nothing and counts its runs.
sc::DeviceJob Counting(std::atomic<int>& runs) {
  return [&runs](llmp::providers::NativeStream /*stream*/) {
    runs.fetch_add(1);
    return sc::JobResult::kQueued;
  };
}

// Materializes a closure, holds it, submits one step under the lease and
// ends the lease at once, while the step is in flight; reports each result.
class EndingProgram final : public sc::TaskProgram {
 public:
  struct Report {
    std::optional<sc::TaskOutcome> outcome;
    bool retired = false;
    std::optional<LeaseId> lease;
    std::optional<std::expected<sc::Readiness, sc::WorkError>> ended;
    std::optional<std::expected<sc::Readiness, sc::WorkError>> ended_again;
    std::optional<sc::WorkError> after_end;  // a submission once it ended
  };
  EndingProgram(Report& report, Closure closure, std::atomic<int>& runs)
      : report_(report), closure_(std::move(closure)), runs_(runs) {}

  sc::Step Advance(sc::TaskContext& context) override {
    if (!report_.lease) {
      const auto ready = context.Materialize(closure_);
      if (!ready) {
        return sc::Step::Finish(sc::TaskOutcome::kFailed);
      }
      if (*ready == sc::Readiness::kWaiting) {
        return sc::Step::Wait();
      }
      const auto held = context.HoldLease(closure_);
      if (!held) {
        return sc::Step::Finish(sc::TaskOutcome::kFailed);
      }
      report_.lease = *held;
      if (!context.SubmitLaunch(*held, sc::LaunchWork{.stream = 0, .job = Counting(runs_)})) {
        return sc::Step::Finish(sc::TaskOutcome::kFailed);
      }
      report_.ended = context.EndLease(*held);
      report_.ended_again = context.EndLease(*held);  // waits once, not twice
      report_.after_end = llmp::test_support::Failed(
          context.SubmitLaunch(*held, sc::LaunchWork{.stream = 0, .job = Counting(runs_)}));
      return sc::Step::Wait();  // the step's fence, then the lease's release
    }
    return sc::Step::Finish(context.TakeFailure() ? sc::TaskOutcome::kFailed
                                                  : sc::TaskOutcome::kSucceeded);
  }
  void Finished(sc::TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override { report_.retired = true; }

 private:
  Report& report_;
  Closure closure_;
  std::atomic<int>& runs_;
};

// One call of a task's context, then finish: what it returned.
class ProbeProgram final : public sc::TaskProgram {
 public:
  using Probe = std::function<void(sc::TaskContext&)>;
  explicit ProbeProgram(Probe probe) : probe_(std::move(probe)) {}
  sc::Step Advance(sc::TaskContext& context) override {
    probe_(context);
    return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
  }

 private:
  Probe probe_;
};

// Leases its closure, then asks to wait for another request's lease on
// `others`, itself and from a child it waits for; then ends its own lease
// and asks again, and finishes once woken.
class HoldingProgram final : public sc::TaskProgram {
 public:
  using Result = std::optional<std::expected<sc::Readiness, sc::WorkError>>;
  struct Report {
    Result own;    // while it holds its lease
    Result child;  // its child's, while it holds its lease
    Result ended;  // once its lease is ended
    std::optional<sc::TaskOutcome> outcome;
  };
  HoldingProgram(Report& report, Closure closure, std::vector<ExtentId> others)
      : report_(report), closure_(std::move(closure)), others_(std::move(others)) {}

  sc::Step Advance(sc::TaskContext& context) override {
    switch (phase_) {
      case 0: {
        const auto ready = context.Materialize(closure_);
        if (!ready) {
          return sc::Step::Finish(sc::TaskOutcome::kFailed);
        }
        if (*ready == sc::Readiness::kWaiting) {
          return sc::Step::Wait();
        }
        const auto held = context.HoldLease(closure_);
        if (!held) {
          return sc::Step::Finish(sc::TaskOutcome::kFailed);
        }
        lease_ = *held;
        report_.own = context.AwaitRelease(others_);
        auto child = std::make_unique<ProbeProgram>(
            [this](sc::TaskContext& c) { report_.child = c.AwaitRelease(others_); });
        if (!context.Spawn(std::move(child))) {
          return sc::Step::Finish(sc::TaskOutcome::kFailed);
        }
        phase_ = 1;
        return sc::Step::Wait();  // for the child
      }
      case 1: {
        const auto ended = context.EndLease(lease_);
        if (!ended || *ended != sc::Readiness::kReady) {
          return sc::Step::Finish(sc::TaskOutcome::kFailed);
        }
        report_.ended = context.AwaitRelease(others_);
        phase_ = 2;
        return sc::Step::Wait();  // for the other lease's release
      }
      default:
        return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
    }
  }
  void Finished(sc::TaskOutcome outcome) override { report_.outcome = outcome; }

 private:
  Report& report_;
  Closure closure_;
  std::vector<ExtentId> others_;
  LeaseId lease_;
  int phase_ = 0;
};

class HeldLeaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    zone_ = memory_.Reserve(Bytes(kSize * kSlots)).value();
    const auto zone_backing = memory_.Create(kHostClass, Bytes(kSize * kSlots)).value();
    ASSERT_TRUE(memory_.Map(zone_, Bytes(0), zone_backing).has_value());
    ASSERT_TRUE(
        memory_
            .SetAccess(zone_, Bytes(0), Bytes(kSize * kSlots), llmp::providers::Access::kReadWrite)
            .has_value());
    domain_ = catalog_.AddDomain("node");
    workspace_ = catalog_
                     .AddExtent({.domain = domain_,
                                 .memory_class = llmp::catalog::MemoryClass::kScratch,
                                 .recovery = llmp::catalog::Recovery::kDiscardable,
                                 .size = Bytes(kSize),
                                 .content = {}},
                                true)
                     .value();
    for (std::size_t m = 0; m < 2; ++m) {
      places_.at(m) = memory_.Reserve(Bytes(kSize * kPerModel)).value();
      bases_.at(m) = memory_.RangeOf(places_.at(m)).value().base;
      std::vector<std::byte> file(kSize * kPerModel);
      for (std::size_t i = 0; i < file.size(); ++i) {
        file[i] = static_cast<std::byte>((i * 131) + (m * 71) + 1);
      }
      fds_.at(m) = storage_.AddFile(file);
      std::array<std::uint8_t, 32> artifact{};
      artifact[0] = static_cast<std::uint8_t>(m + 1);
      for (std::size_t i = 0; i < kPerModel; ++i) {
        weights_.at(m).push_back(
            catalog_
                .AddExtent({.domain = domain_,
                            .memory_class = llmp::catalog::MemoryClass::kWeights,
                            .recovery = llmp::catalog::Recovery::kFromArtifact,
                            .size = Bytes(kSize),
                            .content = {.artifact = artifact,
                                        .group = 0,
                                        .chunk = static_cast<std::uint32_t>(i)}})
                .value());
      }
    }
    stream_ = execution_.CreateStream().value();
    baseline_ = memory_.backings();
  }

  // The scheduler with budget B, in extents beyond the workspace.
  void Build(std::uint64_t budget_extents) {
    board_ = std::make_unique<sc::CompletionBoard>(32, wake_);
    storage_lane_ = std::make_unique<sc::StorageService>(
        storage_,
        llmp::providers::ReaderSettings{.alignment = 4096,
                                        .request_bytes = 16 * 1024,
                                        .retries = 2,
                                        .reads = 32,
                                        .waiters = 8,
                                        .span_bytes = llmp::providers::kNoCoalescing,
                                        .span_segments = llmp::providers::kMaxSegments},
        *board_, sc::QueueSettings{.capacity = 16, .reserved = 4, .batch = 16});
    device_lane_ = std::make_unique<sc::DeviceService>(
        execution_, std::span<const StreamId>(&stream_, 1), *board_,
        sc::DeviceSettings{.queue = {.capacity = 16, .reserved = 4, .batch = 16}, .handoff = 16},
        &memory_);
    sc::LandingZone landing{.slots = {}, .slot_bytes = Bytes(kSize), .stream = 0};
    const std::uint64_t zone_base = memory_.RangeOf(zone_).value().base;
    for (std::size_t i = 0; i < kSlots; ++i) {
      landing.slots.push_back(zone_base + (i * kSize));
    }
    scheduler_ = std::make_unique<sc::Scheduler>(
        catalog_, *board_, wake_,
        sc::Lanes{.storage = storage_lane_.get(), .device = device_lane_.get(), .cpu = nullptr},
        sc::SchedulerSettings{.tasks = 16,
                              .budget = Bytes((budget_extents + 1) * kSize),
                              .poll_window = poll_window_,
                              .landing = landing});
    for (std::size_t m = 0; m < 2; ++m) {
      for (std::size_t i = 0; i < kPerModel; ++i) {
        ASSERT_TRUE(
            scheduler_
                ->SetSource(
                    weights_.at(m)[i],
                    sc::PageSource{.read = {.fd = fds_.at(m),
                                            .offset = i * kSize,
                                            .memory = nullptr,
                                            .length = kSize},
                                   .landed = true,
                                   .destination = bases_.at(m) + (i * kSize),
                                   .backing = sc::BackingPlace{.reservation = places_.at(m),
                                                               .offset = Bytes(i * kSize),
                                                               .size = Bytes(kSize),
                                                               .allocation_class = kDeviceClass}})
                .has_value());
      }
    }
  }

  void TearDown() override {
    if (scheduler_ == nullptr) {
      return;
    }
    if (!scheduler_->fault() && !scheduler_->Stopped()) {
      std::vector<ExtentId> all = weights_[0];
      all.insert(all.end(), weights_[1].begin(), weights_[1].end());
      ts::Done done;
      EXPECT_TRUE(Run(std::make_unique<ts::EvictProgram>(done, all), done));
    }
    scheduler_->RequestShutdown();
    for (int i = 0; i < 1000 && !scheduler_->Stopped(); ++i) {
      Round();
    }
    EXPECT_TRUE(scheduler_->Stopped().has_value());
    storage_lane_->Close();
    device_lane_->Close();
    for (int i = 0; i < 100; ++i) {
      Round();
    }
  }

  // One turn of each lane and of the scheduler; with `device`, the fake
  // device runs what is queued.
  bool Round(bool device = true) {
    bool progress = storage_lane_->Turn(false);
    progress = device_lane_->SubmissionTurn() || progress;
    if (device) {
      execution_.Drain();
    }
    progress = device_lane_->CompletionTurn() || progress;
    return scheduler_->Turn() || progress;
  }
  void Settle(bool device = true) {
    for (int i = 0; i < 1000 && Round(device); ++i) {
    }
  }
  bool Run(std::unique_ptr<sc::TaskProgram> program, ts::Done& done) {
    if (!scheduler_->Start(++request_, std::move(program), 1)) {
      return false;
    }
    for (int i = 0; i < 10000 && !done.gone.load(); ++i) {
      Round();
    }
    EXPECT_TRUE(done.gone.load());
    return done.outcome.load() == static_cast<int>(sc::TaskOutcome::kSucceeded);
  }

  // A request's driver, turning the lanes itself.
  struct Request {
    ts::Done done;
    ts::RequestChannel channel;
    std::uint64_t tag = 0;
  };
  // Opens one over model m's closure; returns once it holds its lease.
  void Open(Request& request, std::size_t m) {
    request.tag = ++request_;
    ASSERT_TRUE(scheduler_
                    ->Start(request.tag,
                            std::make_unique<ts::RequestProgram>(request.done, ModelClosure(m),
                                                                 request.channel),
                            1)
                    .has_value());
    Settle();
    ASSERT_TRUE(request.channel.held.load());
  }
  void Signal(const Request& request) {
    ASSERT_EQ(scheduler_->Post(sc::SignalRequest{.request = request.tag}),
              llmp::base::PushResult::kAccepted);
  }
  // One step: hands it over and turns everything (with `device`, the fake
  // device runs it; without, its fence stays pending).
  void Step(Request& request, sc::DeviceJob job, bool device = true) {
    request.channel.job = std::move(job);
    request.channel.stream = 0;
    Signal(request);
    Settle(device);
  }
  void End(Request& request) {
    request.channel.end = true;
    Signal(request);
    Settle();
  }
  // What an eviction of `extent` returns now, from a task of its own (with
  // `device`, the fake device may run what is queued meanwhile).
  std::expected<sc::Readiness, sc::WorkError> TryEvict(ExtentId extent, bool device = true) {
    std::optional<std::expected<sc::Readiness, sc::WorkError>> result;
    EXPECT_TRUE(scheduler_
                    ->Start(++request_, std::make_unique<ProbeProgram>(
                                            [&](sc::TaskContext& c) { result = c.Evict(extent); }))
                    .has_value());
    Settle(device);
    return result.value_or(std::unexpected(sc::WorkError::kInvalid));
  }

  Closure ModelClosure(std::size_t m) const {
    std::vector<ExtentId> extents = weights_.at(m);
    extents.push_back(workspace_);
    return catalog_.ClosureOfExtents(extents).value();
  }
  ExtentView View(ExtentId extent) const { return catalog_.Describe(extent).value(); }
  std::size_t Resident(std::size_t m) const {
    std::size_t resident = 0;
    for (const ExtentId extent : weights_.at(m)) {
      resident += View(extent).state == ExtentState::kResident ? 1 : 0;
    }
    return resident;
  }
  std::uint32_t Leases(std::size_t m) const {
    std::uint32_t leases = 0;
    for (const ExtentId extent : weights_.at(m)) {
      leases += View(extent).leases;
    }
    return leases;
  }
  Bytes Held() const { return catalog_.OccupancyOf(domain_).held; }

  static constexpr std::size_t kDeviceClass = 0;
  static constexpr std::size_t kHostClass = 1;

  // Declared in dependency order: the scheduler goes first, the providers last.
  llmp::providers::fake::FakeDeviceMemory memory_{Bytes(kSize), Bytes(kSize * 64)};
  llmp::providers::fake::FakeStorage storage_{8, 4096};
  llmp::providers::fake::FakeDeviceExecution execution_;
  llmp::catalog::Catalog catalog_;
  llmp::base::WakeFlag wake_;
  std::unique_ptr<sc::CompletionBoard> board_;
  std::unique_ptr<sc::StorageService> storage_lane_;
  std::unique_ptr<sc::DeviceService> device_lane_;
  Request unproven_request_;  // its reporting program may survive until scheduler destruction
  std::unique_ptr<sc::Scheduler> scheduler_;

  llmp::providers::ReservationId zone_;
  llmp::catalog::DomainId domain_;
  ExtentId workspace_;
  std::array<llmp::providers::ReservationId, 2> places_;
  std::array<std::uint64_t, 2> bases_{};
  std::array<int, 2> fds_{};
  std::array<std::vector<ExtentId>, 2> weights_;
  StreamId stream_;
  std::size_t baseline_ = 0;
  std::uint64_t request_ = 0;
  std::chrono::microseconds poll_window_{200};  // the scheduler's, set before Build
};

// The request pages its closure in and leases it once; its steps run under
// that one lease (never a second, never none) and its end releases it with
// every extent still resident, idle, and its last use the release's.
TEST_F(HeldLeaseTest, ARequestLeasesItsClosureOnceAndEachStepRunsUnderIt) {
  Build(2 * kPerModel);
  Request request;
  Open(request, 0);
  EXPECT_EQ(Resident(0), kPerModel);
  EXPECT_EQ(Leases(0), kPerModel);
  EXPECT_EQ(View(workspace_).leases, 1U);
  EXPECT_EQ(Held(), Bytes((kPerModel + 1) * kSize));
  EXPECT_EQ(scheduler_->held(), 1U);
  const std::uint64_t used = View(weights_[0][0]).last_use;
  std::atomic<int> runs{0};
  for (int step = 1; step <= 3; ++step) {
    Step(request, Counting(runs));
    EXPECT_EQ(runs.load(), step);
    EXPECT_EQ(request.channel.steps.load(), static_cast<std::uint64_t>(step));
    EXPECT_FALSE(request.channel.step_failed.load());
    EXPECT_EQ(Leases(0), kPerModel) << "step " << step;  // the one lease, no other
    EXPECT_EQ(scheduler_->operations(), 0U);
  }
  EXPECT_EQ(execution_.fences(), 0U);  // each step's fence seen and released
  End(request);
  EXPECT_TRUE(request.done.gone.load());
  EXPECT_EQ(request.done.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded));
  EXPECT_EQ(Leases(0), 0U);
  EXPECT_EQ(View(workspace_).leases, 0U);
  EXPECT_EQ(Resident(0), kPerModel);  // release is not eviction
  EXPECT_EQ(Held(), Bytes());
  EXPECT_EQ(catalog_.OccupancyOf(domain_).idle, Bytes((kPerModel + 1) * kSize));
  EXPECT_GT(View(weights_[0][0]).last_use, used);
  EXPECT_EQ(scheduler_->held(), 0U);
  EXPECT_EQ(scheduler_->stats().leases_held, 1U);
  EXPECT_EQ(scheduler_->stats().leases_released, 1U);
  EXPECT_EQ(scheduler_->stats().held_operations, 3U);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
}

// The runtime wake's follow window: a request's step ending has the device
// lane poll for the request's next step (DeviceService::Anticipate), for
// at least the scheduler's poll window (here 10 s, so the check cannot
// race it). Paging the closure in, which runs no step, does not.
TEST_F(HeldLeaseTest, AStepsEndHasTheDeviceLanePollForTheNext) {
  poll_window_ = std::chrono::seconds(10);
  Build(2 * kPerModel);
  Request request;
  Open(request, 0);  // pages in and leases: no step yet
  EXPECT_FALSE(device_lane_->Anticipating(std::chrono::steady_clock::now()));
  std::atomic<int> runs{0};
  Step(request, Counting(runs));
  ASSERT_EQ(runs.load(), 1);
  EXPECT_TRUE(device_lane_->Anticipating(std::chrono::steady_clock::now()));
  EXPECT_TRUE(
      device_lane_->Anticipating(std::chrono::steady_clock::now() + std::chrono::seconds(5)));
  EXPECT_FALSE(
      device_lane_->Anticipating(std::chrono::steady_clock::now() + std::chrono::seconds(11)));
  End(request);
  EXPECT_TRUE(request.done.gone.load());
}

// No eviction of a held extent can begin; an evicting program waits for
// the lease's release rather than retrying, and then evicts.
TEST_F(HeldLeaseTest, EvictionIsRefusedWhileARequestHoldsItsLease) {
  Build(2 * kPerModel);
  Request request;
  Open(request, 0);
  EXPECT_EQ(Failed(TryEvict(weights_[0][1])), sc::WorkError::kBusy);
  EXPECT_EQ(View(weights_[0][1]).state, ExtentState::kResident);

  ts::Done evicting;
  ASSERT_TRUE(
      scheduler_->Start(++request_, std::make_unique<ts::EvictProgram>(evicting, weights_[0]), 1)
          .has_value());
  const std::uint64_t turns = scheduler_->stats().turns;
  Settle();
  EXPECT_FALSE(evicting.gone.load());  // waiting, not spinning:
  EXPECT_LT(scheduler_->stats().turns - turns, 10U);
  EXPECT_EQ(Resident(0), kPerModel);
  EXPECT_EQ(Leases(0), kPerModel);

  End(request);
  EXPECT_TRUE(request.done.gone.load());
  EXPECT_TRUE(evicting.gone.load());
  EXPECT_EQ(evicting.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded));
  EXPECT_EQ(Resident(0), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);  // every weight's backing released
}

// A swap asked for while a request holds the outgoing model waits for the
// request's end; then it evicts the one and pages in the other, handing
// the backing over.
TEST_F(HeldLeaseTest, ASwapWaitsForTheRequestToEnd) {
  Build(kPerModel);  // one model at a time
  Request request;
  Open(request, 0);
  std::atomic<int> runs{0};
  Step(request, Counting(runs));
  ts::SwapReport report;
  ts::Done swapping;
  ASSERT_TRUE(scheduler_
                  ->Start(++request_,
                          std::make_unique<ts::SwapProgram>(swapping, weights_[0], ModelClosure(1),
                                                            /*handoff=*/true, report),
                          1)
                  .has_value());
  Settle();
  EXPECT_FALSE(swapping.gone.load());
  EXPECT_EQ(Resident(0), kPerModel);
  EXPECT_EQ(Resident(1), 0U);
  // The request goes on meanwhile.
  Step(request, Counting(runs));
  EXPECT_EQ(runs.load(), 2);
  EXPECT_FALSE(swapping.gone.load());

  End(request);
  EXPECT_TRUE(swapping.gone.load());
  EXPECT_EQ(swapping.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded));
  EXPECT_EQ(Resident(0), 0U);
  EXPECT_EQ(Resident(1), kPerModel);
  EXPECT_EQ(report.evictions, kPerModel);
  EXPECT_EQ(scheduler_->stats().handed_off, kPerModel);
  EXPECT_LE(catalog_.OccupancyOf(domain_).Total(), Bytes((kPerModel + 1) * kSize));
}

// Cancelled with a step in flight: the request's task finishes at once,
// but the lease holds (no eviction can begin) until the step's fence is
// seen; then it is released, the extents stay resident, and the task
// retires.
TEST_F(HeldLeaseTest, CancellingWithAStepInFlightKeepsTheLeaseUntilItsFence) {
  Build(2 * kPerModel);
  Request request;
  Open(request, 0);
  std::atomic<int> runs{0};
  Step(request, Counting(runs), /*device=*/false);
  EXPECT_EQ(runs.load(), 1);  // queued; the fake device has not run it
  EXPECT_EQ(scheduler_->operations(), 1U);
  EXPECT_EQ(request.channel.steps.load(), 0U);

  ASSERT_TRUE(scheduler_->Cancel(request.tag));
  EXPECT_EQ(request.done.outcome.load(), static_cast<int>(sc::TaskOutcome::kCancelled));
  Settle(false);
  EXPECT_FALSE(request.done.gone.load());  // its step's lifetime hold
  EXPECT_EQ(Leases(0), kPerModel);
  EXPECT_EQ(scheduler_->held(), 1U);
  EXPECT_EQ(Failed(TryEvict(weights_[0][0], /*device=*/false)), sc::WorkError::kBusy);

  Settle();  // the device runs the step; the completion lane sees its fence
  EXPECT_TRUE(request.done.gone.load());
  EXPECT_EQ(Leases(0), 0U);
  EXPECT_EQ(Resident(0), kPerModel);
  EXPECT_EQ(scheduler_->held(), 0U);
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
}

// Ending the lease with a step in flight waits for the step's fence (the
// holder is woken then); a step after the end is refused.
TEST_F(HeldLeaseTest, EndingWithAStepInFlightWaitsForItsFence) {
  Build(2 * kPerModel);
  ts::Done loading;  // resident first: the step alone stays in flight below
  ASSERT_TRUE(
      Run(std::make_unique<ts::RunProgram>(loading, ModelClosure(0), sc::DeviceJob{}), loading));
  EndingProgram::Report report;
  std::atomic<int> runs{0};
  ASSERT_TRUE(
      scheduler_
          ->Start(++request_, std::make_unique<EndingProgram>(report, ModelClosure(0), runs), 1)
          .has_value());
  Settle(false);
  ASSERT_TRUE(report.lease.has_value());
  ASSERT_TRUE(report.ended.has_value() && report.ended->has_value());
  EXPECT_EQ(report.ended.value_or(std::unexpected(sc::WorkError::kInvalid)),
            sc::Readiness::kWaiting);
  // Ended twice: it waits for the one release (counted once, or the task
  // would never be woken below).
  EXPECT_EQ(report.ended_again.value_or(std::unexpected(sc::WorkError::kInvalid)),
            sc::Readiness::kWaiting);
  EXPECT_EQ(report.after_end, sc::WorkError::kClosed);
  EXPECT_EQ(runs.load(), 1);
  EXPECT_FALSE(report.outcome.has_value());
  EXPECT_EQ(Leases(0), kPerModel);  // the step may still touch it

  Settle();
  EXPECT_EQ(report.outcome, sc::TaskOutcome::kSucceeded);
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(Leases(0), 0U);
  EXPECT_EQ(Resident(0), kPerModel);
  EXPECT_EQ(scheduler_->stats().leases_released, 1U);

  // A signal for the request once it has ended (and one for a request that
  // never was) finds no task: nothing is woken, kept or released.
  ASSERT_EQ(scheduler_->Post(sc::SignalRequest{.request = request_}),
            llmp::base::PushResult::kAccepted);
  ASSERT_EQ(scheduler_->Post(sc::SignalRequest{.request = request_ + 100}),
            llmp::base::PushResult::kAccepted);
  Settle();
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
  EXPECT_EQ(scheduler_->held(), 0U);
  EXPECT_EQ(scheduler_->stats().leases_released, 1U);
}

// A task holding a request's lease is refused a wait for another's, and so
// is its child: a holder waiting for a holder could close a cycle. Once it
// has ended its own it may wait, and is woken by the other's release.
TEST_F(HeldLeaseTest, AHolderNeverWaitsForAnotherLease) {
  Build(2 * kPerModel);
  Request request;
  Open(request, 0);
  HoldingProgram::Report report;
  ASSERT_TRUE(scheduler_
                  ->Start(++request_,
                          std::make_unique<HoldingProgram>(report, ModelClosure(1), weights_[0]), 1)
                  .has_value());
  Settle();
  ASSERT_TRUE(report.own.has_value() && report.child.has_value() && report.ended.has_value());
  EXPECT_EQ(Failed(report.own.value_or(sc::Readiness::kReady)), sc::WorkError::kInvalid);
  EXPECT_EQ(Failed(report.child.value_or(sc::Readiness::kReady)), sc::WorkError::kInvalid);
  EXPECT_EQ(report.ended.value_or(std::unexpected(sc::WorkError::kInvalid)),
            sc::Readiness::kWaiting);
  EXPECT_FALSE(report.outcome.has_value());  // waiting for the request's lease
  EXPECT_EQ(Leases(1), 0U);                  // its own released

  End(request);
  EXPECT_EQ(report.outcome, sc::TaskOutcome::kSucceeded);
  EXPECT_EQ(scheduler_->held(), 0U);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
}

// A signal that comes before the task waits for one is kept: a request
// ended while it is still paging its closure in takes its lease, then
// ends.
TEST_F(HeldLeaseTest, ASignalBeforeItsWaitIsKept) {
  Build(2 * kPerModel);
  Request request;
  request.tag = ++request_;
  ASSERT_TRUE(scheduler_
                  ->Start(request.tag,
                          std::make_unique<ts::RequestProgram>(request.done, ModelClosure(0),
                                                               request.channel),
                          1)
                  .has_value());
  ASSERT_TRUE(scheduler_->Turn());  // it starts paging in
  request.channel.end = true;
  Signal(request);
  Settle();
  EXPECT_TRUE(request.channel.held.load());
  EXPECT_TRUE(request.done.gone.load());
  EXPECT_EQ(request.done.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded));
  EXPECT_EQ(scheduler_->stats().leases_held, 1U);
  EXPECT_EQ(scheduler_->stats().leases_released, 1U);
  EXPECT_EQ(Leases(0), 0U);
  EXPECT_EQ(Resident(0), kPerModel);
}

// Only the holder submits under its lease, and only a resident closure at
// its recorded contents is leased.
TEST_F(HeldLeaseTest, OnlyTheHolderSubmitsAndOnlyAResidentClosureIsHeld) {
  Build(2 * kPerModel);
  std::optional<std::expected<LeaseId, sc::WorkError>> nonresident;
  ASSERT_TRUE(scheduler_
                  ->Start(++request_, std::make_unique<ProbeProgram>([&](sc::TaskContext& c) {
                    nonresident = c.HoldLease(ModelClosure(1));
                  }))
                  .has_value());
  Settle();
  ASSERT_TRUE(nonresident.has_value());
  EXPECT_EQ(Failed(nonresident.value_or(LeaseId())), sc::WorkError::kNotResident);

  Request request;
  Open(request, 0);
  // The request's lease is the first the catalog issued (the probe above
  // took none): another task can neither submit under it nor end it.
  const LeaseId requests(0, 1);
  ASSERT_TRUE(catalog_.RecordUse(requests, 0).has_value());
  std::atomic<int> runs{0};
  std::optional<std::expected<sc::OperationId, sc::WorkError>> foreign;
  std::optional<std::expected<sc::Readiness, sc::WorkError>> foreign_end;
  ASSERT_TRUE(scheduler_
                  ->Start(++request_, std::make_unique<ProbeProgram>([&](sc::TaskContext& c) {
                    foreign = c.SubmitLaunch(requests,
                                             sc::LaunchWork{.stream = 0, .job = Counting(runs)});
                    foreign_end = c.EndLease(requests);
                  }))
                  .has_value());
  Settle();
  ASSERT_TRUE(foreign.has_value() && foreign_end.has_value());
  EXPECT_EQ(Failed(foreign.value_or(sc::OperationId())), sc::WorkError::kInvalid);
  EXPECT_EQ(Failed(foreign_end.value_or(sc::Readiness::kReady)), sc::WorkError::kInvalid);
  EXPECT_EQ(runs.load(), 0);
  EXPECT_EQ(Leases(0), kPerModel);  // still the request's
  End(request);
  EXPECT_EQ(Leases(0), 0U);
}

// A step whose fence cannot be proven quarantines it: the request's lease
// is never released (nothing may reuse what it may still touch), the node
// faults, and the stop reports it.
TEST_F(HeldLeaseTest, AnUnprovenStepKeepsTheLeaseForGood) {
  Build(2 * kPerModel);
  Request& request = unproven_request_;
  Open(request, 0);
  std::atomic<int> runs{0};
  Step(request, Counting(runs), /*device=*/false);
  execution_.FailNextQuery(ProviderError::kUnknown);
  Settle(false);
  EXPECT_EQ(scheduler_->fault(), sc::Fault::kUnproven);
  EXPECT_EQ(scheduler_->quarantined(), 1U);
  EXPECT_TRUE(request.channel.step_failed.load());
  End(request);  // ending it cannot release it
  EXPECT_EQ(Leases(0), kPerModel);
  EXPECT_EQ(scheduler_->held(), 1U);
  EXPECT_EQ(Held(), Bytes((kPerModel + 1) * kSize));  // stays charged
  scheduler_->RequestShutdown();
  Settle();
  const auto stopped = scheduler_->Stopped();
  ASSERT_TRUE(stopped.has_value());
  EXPECT_EQ(Failed(stopped.value_or(std::expected<void, sc::Fault>())), sc::Fault::kUnproven);
  EXPECT_EQ(scheduler_->held(), 1U);
}

// Steps the driver runs under the lease itself (PagedNode's direct steps,
// RequestChannel::external): ending the request with one in flight waits
// for it, as for a step of the task's own; so does cancelling it, which
// finishes the task but keeps the lease until the count drains.
TEST_F(HeldLeaseTest, ExternalStepsKeepAnEndingLeaseUntilTheyDrain) {
  Build(2 * kPerModel);
  for (const bool cancel : {false, true}) {
    Request request;
    Open(request, 0);
    request.channel.external->store(1);  // a direct step in flight
    if (cancel) {
      ASSERT_TRUE(scheduler_->Cancel(request.tag));
      Settle();
      EXPECT_EQ(request.done.outcome.load(), static_cast<int>(sc::TaskOutcome::kCancelled));
    } else {
      End(request);
      EXPECT_FALSE(request.done.outcome.load() >= 0);  // its holder waits for the release
    }
    EXPECT_EQ(Leases(0), kPerModel) << cancel;
    EXPECT_EQ(scheduler_->held(), 1U) << cancel;
    EXPECT_EQ(Failed(TryEvict(weights_[0][0])), sc::WorkError::kBusy) << cancel;
    EXPECT_FALSE(scheduler_->Stopped().has_value());

    request.channel.external->store(0);  // its fence seen
    Settle();
    EXPECT_TRUE(request.done.gone.load()) << cancel;
    EXPECT_EQ(request.done.outcome.load(),
              static_cast<int>(cancel ? sc::TaskOutcome::kCancelled : sc::TaskOutcome::kSucceeded));
    EXPECT_EQ(Leases(0), 0U) << cancel;
    EXPECT_EQ(Resident(0), kPerModel) << cancel;
    EXPECT_EQ(scheduler_->held(), 0U) << cancel;
    EXPECT_EQ(scheduler_->tasks().size(), 0U) << cancel;
  }
  EXPECT_EQ(scheduler_->stats().leases_released, 2U);
}

// The count is shared: a lease whose request's driver is gone (its channel
// freed once the task retired) is released once the count it shared drops.
TEST_F(HeldLeaseTest, AnExternalCountOutlivesItsChannel) {
  Build(2 * kPerModel);
  std::shared_ptr<std::atomic<std::uint32_t>> count;
  {
    Request request;
    Open(request, 0);
    count = request.channel.external;
    count->store(1);
    ASSERT_TRUE(scheduler_->Cancel(request.tag));
    Settle();
    ASSERT_TRUE(request.done.gone.load());
  }  // the channel is gone; the lease still counts the step
  EXPECT_EQ(Leases(0), kPerModel);
  EXPECT_EQ(scheduler_->held(), 1U);
  count->store(0);
  Settle();
  EXPECT_EQ(Leases(0), 0U);
  EXPECT_EQ(scheduler_->held(), 0U);
  EXPECT_EQ(count.use_count(), 1);  // the lease let go of it
}

// The external completion races the two parts of EndLease: deciding whether
// to wait and registering that wait. ASan checks that retirement never leaves
// EndLease accessing a freed Held; the task must also receive its wake exactly once.
TEST(HeldLeaseRaceTest, ExternalCompletionRacesLeaseEnd) {
  llmp::catalog::Catalog catalog;
  llmp::base::WakeFlag wake;
  sc::CompletionBoard board(16, wake);
  ts::Done done;  // outlives the scheduler even if the deadline assertion fails
  auto external = std::make_shared<std::atomic<std::uint32_t>>(0);
  std::atomic<std::uint64_t> offered{0};
  sc::Scheduler scheduler(catalog, board, wake, {}, {.tasks = 4});
  std::jthread completing([&](std::stop_token stop) {
    std::uint64_t seen = 0;
    while (!stop.stop_requested()) {
      const auto next = offered.load(std::memory_order_acquire);
      if (next == seen) {
        std::this_thread::yield();
        continue;
      }
      external->store(0, std::memory_order_release);
      seen = next;
    }
  });
  class Ending final : public sc::ReportingProgram {
   public:
    Ending(ts::Done& done, std::shared_ptr<std::atomic<std::uint32_t>> external,
           std::atomic<std::uint64_t>& offered)
        : ReportingProgram(done), external_(std::move(external)), offered_(offered) {}
    sc::Step Advance(sc::TaskContext& context) override {
      if (!ending_) {
        const auto held = context.HoldLease({}, external_);
        if (!held) return Fail(held.error());
        ending_ = true;
        external_->store(1, std::memory_order_relaxed);
        offered_.fetch_add(1, std::memory_order_release);
        const auto ended = context.EndLease(*held);
        if (!ended) return Fail(ended.error());
        if (*ended == sc::Readiness::kWaiting) return sc::Step::Wait();
      }
      return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
    }

   private:
    std::shared_ptr<std::atomic<std::uint32_t>> external_;
    std::atomic<std::uint64_t>& offered_;
    bool ending_ = false;
  };
  constexpr std::uint64_t kTrials = 20000;
  const auto deadline = std::chrono::steady_clock::now() + kPatience;
  for (std::uint64_t trial = 1; trial <= kTrials; ++trial) {
    done.outcome.store(-1);
    done.error.store(-1);
    done.retired.store(false);
    done.gone.store(false);
    ASSERT_TRUE(scheduler.Start(trial, std::make_unique<Ending>(done, external, offered)));
    while (!done.gone.load() && std::chrono::steady_clock::now() < deadline) {
      (void)scheduler.Turn();
      std::this_thread::yield();
    }
    ASSERT_TRUE(done.gone.load()) << trial;
    ASSERT_EQ(done.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded)) << trial;
    ASSERT_EQ(scheduler.held(), 0U) << trial;
  }
  EXPECT_EQ(scheduler.stats().leases_held, kTrials);
  EXPECT_EQ(scheduler.stats().leases_released, kTrials);
  scheduler.RequestShutdown();
  (void)scheduler.Turn();
  ASSERT_TRUE(scheduler.Stopped().has_value());
  EXPECT_TRUE(scheduler.Stopped()->has_value());
}

// External work never seen to end keeps the lease for good, and the stop
// reports it unproven.
TEST_F(HeldLeaseTest, AnUnprovenExternalStepKeepsTheLeaseForGood) {
  Build(2 * kPerModel);
  Request request;
  Open(request, 0);
  request.channel.external->store(1);
  ASSERT_TRUE(scheduler_->Cancel(request.tag));
  Settle();
  // The task retires (its driver's wait ends); its lease stays.
  EXPECT_TRUE(request.done.gone.load());
  EXPECT_EQ(Leases(0), kPerModel);
  EXPECT_EQ(scheduler_->held(), 1U);
  scheduler_->RequestShutdown();
  Settle();
  const auto stopped = scheduler_->Stopped();
  ASSERT_TRUE(stopped.has_value());
  EXPECT_EQ(Failed(stopped.value_or(std::expected<void, sc::Fault>())), sc::Fault::kUnproven);
  EXPECT_EQ(scheduler_->held(), 1U);
}

// Every lane, the scheduler and the fake device on threads of their own,
// the driver on the test's: each of many steps is handed over through the
// channel and a signal, runs once on the device lane's thread, and is seen
// done by the driver, with no signal lost; the job's plain writes are seen
// by the driver once it counts the step (the memory-ordering stress that
// runs under ThreadSanitizer). A swap started meanwhile waits for the
// request's end, then runs.
TEST_F(HeldLeaseTest, ThreadedStepsLoseNoSignalAndASwapWaitsForTheEnd) {
  Build(kPerModel);
  constexpr int kSteps = 400;
  Request request;
  ts::Done swapping;
  ts::SwapReport report;
  std::optional<std::expected<void, sc::Fault>> result;
  int ran = 0;  // written by each job on the device lane's thread
  {
    std::jthread owner([&] { result = scheduler_->Run(); });
    std::jthread storage([&] { storage_lane_->Run(); });
    std::jthread submission([&] { device_lane_->RunSubmission(); });
    std::jthread completion([&] { device_lane_->RunCompletion(); });
    std::jthread device([&](const std::stop_token& stop) {
      while (!stop.stop_requested()) {
        execution_.Drain();
        std::this_thread::yield();
      }
    });
    request.tag = 1;
    Offer(*scheduler_, sc::StartRequest{.request = request.tag,
                                        .priority = 1,
                                        .program = std::make_unique<ts::RequestProgram>(
                                            request.done, ModelClosure(0), request.channel)});
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (!request.channel.held.load(std::memory_order_acquire) && !request.done.gone.load() &&
           std::chrono::steady_clock::now() < give_up) {
      std::this_thread::yield();
    }
    // Not ASSERT: the threads must still be stopped below.
    EXPECT_TRUE(request.channel.held.load(std::memory_order_acquire));
    Offer(*scheduler_,
          sc::StartRequest{.request = 2,
                           .priority = 1,
                           .program = std::make_unique<ts::SwapProgram>(
                               swapping, weights_[0], ModelClosure(1), /*handoff=*/true, report)});
    int seen = 0;
    for (int step = 0; step < kSteps; ++step) {
      request.channel.job = [&ran, step](llmp::providers::NativeStream /*stream*/) {
        ran = step + 1;
        return sc::JobResult::kQueued;
      };
      request.channel.stream = 0;
      Offer(*scheduler_, sc::SignalRequest{.request = request.tag});
      while (request.channel.steps.load(std::memory_order_acquire) ==
                 static_cast<std::uint64_t>(step) &&
             !request.done.gone.load() && std::chrono::steady_clock::now() < give_up) {
        std::this_thread::yield();
      }
      if (request.channel.steps.load(std::memory_order_acquire) !=
          static_cast<std::uint64_t>(step) + 1) {
        break;
      }
      seen += ran == step + 1 ? 1 : 0;
    }
    EXPECT_EQ(seen, kSteps);
    EXPECT_FALSE(swapping.gone.load());  // still waiting for the request
    request.channel.end = true;
    Offer(*scheduler_, sc::SignalRequest{.request = request.tag});
    while ((!request.done.gone.load() || !swapping.gone.load()) &&
           std::chrono::steady_clock::now() < give_up) {
      std::this_thread::yield();
    }
    EXPECT_TRUE(request.done.gone.load());
    EXPECT_TRUE(swapping.gone.load());
    scheduler_->RequestShutdown();
    owner.join();
    storage_lane_->Close();
    device_lane_->Close();
    storage.join();
    submission.join();
    completion.join();
  }
  EXPECT_TRUE(result.has_value() && result->has_value());
  EXPECT_EQ(request.done.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded));
  EXPECT_EQ(swapping.outcome.load(), static_cast<int>(sc::TaskOutcome::kSucceeded));
  EXPECT_EQ(Resident(0), 0U);
  EXPECT_EQ(Resident(1), kPerModel);
  EXPECT_EQ(Leases(0) + Leases(1), 0U);
  EXPECT_EQ(scheduler_->stats().held_operations, static_cast<std::uint64_t>(kSteps));
  EXPECT_EQ(execution_.fences(), 0U);
}

}  // namespace
