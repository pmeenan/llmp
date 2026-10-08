// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler's D-081 page-in path and its evictions on the deterministic
// fakes (scheduler.h; D-033, D-048, D-081): managed backing created and
// mapped on the device lane or, in the second instantiation of every test,
// on a VMM lane of its own; direct reads into a bounded landing zone, the
// copy into device backing, publication only after the copy's fence, and
// slots reused only after it. Failed and short reads, backing failures,
// cancellation in every stage, a full zone, unproven copies and unmaps,
// evictions that really release backing, relocation and pinned places that
// a swap brings back at the same addresses (D-090), and kernel jobs whose
// lease holds until their fence, even when their first copy's outcome is
// unknown. The deterministic tests turn each lane themselves; the
// threaded one runs every lane on its own thread.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "expected_error.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_memory.h"
#include "providers/fake/fake_storage.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"
#include "scheduler/tasks.h"

namespace {

using jitllm::base::Bytes;
using jitllm::base::PushResult;
using jitllm::catalog::Closure;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;
using jitllm::catalog::ExtentView;
using jitllm::catalog::Occupancy;
using jitllm::providers::Access;
using jitllm::providers::ProviderError;
using jitllm::providers::ReaderSettings;
using jitllm::providers::ReadSpec;
using jitllm::providers::ReservationId;
using jitllm::providers::StreamId;
using jitllm::providers::Submission;
using jitllm::providers::fake::FakeDeviceExecution;
using jitllm::providers::fake::FakeDeviceMemory;
using jitllm::providers::fake::FakeStorage;
using jitllm::providers::fake::kPoison;
using jitllm::scheduler::AfterRefusal;
using jitllm::scheduler::BackingPlace;
using jitllm::scheduler::BackingService;
using jitllm::scheduler::CancelRequest;
using jitllm::scheduler::CompletionBoard;
using jitllm::scheduler::Control;
using jitllm::scheduler::DeviceJob;
using jitllm::scheduler::DeviceService;
using jitllm::scheduler::DeviceSettings;
using jitllm::scheduler::Fault;
using jitllm::scheduler::JobResult;
using jitllm::scheduler::LandingZone;
using jitllm::scheduler::Lanes;
using jitllm::scheduler::LaunchWork;
using jitllm::scheduler::PageSource;
using jitllm::scheduler::QueueSettings;
using jitllm::scheduler::Readiness;
using jitllm::scheduler::Scheduler;
using jitllm::scheduler::SchedulerSettings;
using jitllm::scheduler::StartRequest;
using jitllm::scheduler::Step;
using jitllm::scheduler::StorageService;
using jitllm::scheduler::TaskContext;
using jitllm::scheduler::TaskOutcome;
using jitllm::scheduler::TaskProgram;
using jitllm::scheduler::WorkError;
using jitllm::test_support::Failed;

constexpr std::uint64_t kSize = 64ULL * 1024;  // one extent, one slot
constexpr std::size_t kExtents = 6;
constexpr std::size_t kSlots = 2;
constexpr std::size_t kMostSlots = 4;  // the zone's backing: Build takes at most this many
// The last extent's range is shorter than its backing, as a group's last
// chunk is.
constexpr std::uint64_t kTail = kSize - 8192;
constexpr auto kPatience = std::chrono::seconds(120);

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t LengthOf(std::size_t extent) { return extent + 1 == kExtents ? kTail : kSize; }

// Materializes a closure, then optionally holds it with a kernel job.
class LoadProgram final : public TaskProgram {
 public:
  struct Report {
    std::optional<TaskOutcome> outcome;
    bool retired = false;
    std::optional<WorkError> error;
    int job_runs = 0;
  };
  LoadProgram(Report& report, Closure closure, std::optional<JobResult> job = std::nullopt)
      : report_(report), closure_(std::move(closure)), job_(job) {}
  // Holds the closure with `job` in place of one reporting a fixed result.
  LoadProgram(Report& report, Closure closure, DeviceJob job)
      : report_(report), closure_(std::move(closure)), custom_(std::move(job)) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (!loaded_) {
      const auto ready = context.Materialize(closure_);
      if (!ready) {
        report_.error = ready.error();
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*ready == Readiness::kWaiting) {
        return Step::Wait();
      }
      loaded_ = true;
      if (job_ || custom_) {
        Report& report = report_;
        DeviceJob job = std::move(custom_);
        if (job_) {
          job = [result = *job_](auto /*stream*/) { return result; };
        }
        const auto submitted = context.SubmitLaunch(
            closure_,
            LaunchWork{.stream = 0, .job = [job = std::move(job), &report](auto stream) mutable {
                         ++report.job_runs;
                         return job(stream);
                       }});
        if (!submitted) {
          report_.error = submitted.error();
          return Step::Finish(TaskOutcome::kFailed);
        }
        return Step::Wait();
      }
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override { report_.retired = true; }

 private:
  Report& report_;
  Closure closure_;
  std::optional<JobResult> job_;
  DeviceJob custom_;
  bool loaded_ = false;
};

// Evicts extents, waiting for each unmap.
class EvictProgram final : public TaskProgram {
 public:
  struct Report {
    std::optional<TaskOutcome> outcome;
    std::vector<std::expected<Readiness, WorkError>> results;
  };
  EvictProgram(Report& report, std::vector<ExtentId> extents,
               jitllm::scheduler::EvictOptions options = {})
      : report_(report), extents_(std::move(extents)), options_(options) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (next_ < extents_.size()) {
      const auto result = context.Evict(extents_[next_++], options_);
      report_.results.push_back(result);
      if (!result) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      return *result == Readiness::kWaiting ? Step::Wait() : Step::Yield();
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }

 private:
  Report& report_;
  std::vector<ExtentId> extents_;
  jitllm::scheduler::EvictOptions options_;
  std::size_t next_ = 0;
};

// The parameter: whether VMM work runs on a VMM lane (BackingService)
// rather than on the device lane.
class PageInTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    // The zone: host backing the CPU and the device reach, room for the
    // largest zone a test builds.
    zone_ = memory_.Reserve(Bytes(kSize * kMostSlots)).value();
    zone_backing_ = memory_.Create(kHostClass, Bytes(kSize * kMostSlots)).value();
    ASSERT_TRUE(memory_.Map(zone_, Bytes(0), zone_backing_).has_value());
    ASSERT_TRUE(memory_.SetAccess(zone_, Bytes(0), Bytes(kSize * kMostSlots), Access::kReadWrite)
                    .has_value());
    // Two places for the weights, the second for relocation: address
    // space only, until a load maps backing.
    weights_ = memory_.Reserve(Bytes(kSize * kExtents)).value();
    moved_ = memory_.Reserve(Bytes(kSize * kExtents)).value();
    zone_base_ = memory_.RangeOf(zone_).value().base;
    weights_base_ = memory_.RangeOf(weights_).value().base;
    moved_base_ = memory_.RangeOf(moved_).value().base;
    file_.resize(kSize * kExtents);
    for (std::size_t i = 0; i < file_.size(); ++i) {
      file_[i] = static_cast<std::byte>((i * 131) + (i >> 16) + 1);
    }
    fd_ = storage_.AddFile(file_);
    spill_ = storage_.AddFile(std::vector<std::byte>(kSize * kExtents));
    domain_ = catalog_.AddDomain("node");
    for (std::size_t i = 0; i < kExtents; ++i) {
      extents_.push_back(
          catalog_
              .AddExtent(
                  {.domain = domain_,
                   .memory_class = jitllm::catalog::MemoryClass::kWeights,
                   .recovery = jitllm::catalog::Recovery::kFromArtifact,
                   .size = Bytes(kSize),
                   .content = {.artifact = {}, .group = 0, .chunk = static_cast<std::uint32_t>(i)}})
              .value());
    }
    stream_ = execution_.CreateStream().value();
    baseline_ = memory_.backings();
  }

  // `span`: the reader's span_bytes. By default the reader's own (one
  // request per 16 KiB piece), as most tests here count them; the
  // coalescing tests raise it.
  // `copy`: the zone's copies on a copy lane of their own (Lanes::copy,
  // RE-029), with a stream of its own.
  void Build(std::size_t slots = kSlots, std::size_t board = 32, std::uint64_t budget = kSize * 64,
             std::uint32_t span = jitllm::providers::kNoCoalescing, bool copy = false) {
    board_ = std::make_unique<CompletionBoard>(board, wake_);
    storage_lane_ = std::make_unique<StorageService>(
        storage_,
        ReaderSettings{.alignment = 4096,
                       .request_bytes = 16 * 1024,
                       .retries = 2,
                       .reads = 32,
                       .waiters = 8,
                       .span_bytes = span,
                       .span_segments = jitllm::providers::kMaxSegments},
        *board_, QueueSettings{.capacity = 16, .reserved = 4, .batch = 16});
    device_lane_ = std::make_unique<DeviceService>(
        execution_, std::span<const StreamId>(&stream_, 1), *board_,
        DeviceSettings{.queue = {.capacity = 16, .reserved = 4, .batch = 16}, .handoff = 16},
        GetParam() ? nullptr : &memory_);  // one lane calls the provider
    if (GetParam()) {
      backing_lane_ = std::make_unique<BackingService>(
          &memory_, *board_, QueueSettings{.capacity = 16, .reserved = 4, .batch = 16});
    }
    if (copy) {
      copy_stream_ = execution_.CreateStream().value();
      copy_lane_ = std::make_unique<DeviceService>(
          execution_, std::span<const StreamId>(&copy_stream_, 1), *board_,
          DeviceSettings{.queue = {.capacity = 16, .reserved = 4, .batch = 16}, .handoff = 16},
          nullptr);
    }
    ASSERT_LE(slots, kMostSlots);
    LandingZone landing{.slots = {}, .slot_bytes = Bytes(kSize), .stream = 0};
    for (std::size_t i = 0; i < slots; ++i) {
      landing.slots.push_back(Slot(i));
    }
    scheduler_ =
        std::make_unique<Scheduler>(catalog_, *board_, wake_,
                                    Lanes{.storage = storage_lane_.get(),
                                          .device = device_lane_.get(),
                                          .cpu = nullptr,
                                          .backing = backing_lane_.get(),
                                          .copy = copy_lane_.get()},
                                    SchedulerSettings{.tasks = 16,
                                                      .priorities = 2,
                                                      .aging_limit = 4,
                                                      .controls = 64,
                                                      .controls_reserved = 16,
                                                      .controls_per_turn = 16,
                                                      .observations_per_turn = 64,
                                                      .steps_per_turn = 16,
                                                      .waiters = 8,
                                                      .budget = Bytes(budget),
                                                      .lazy_handoff = lazy_,
                                                      .poll_window = std::chrono::microseconds(200),
                                                      .tick = std::chrono::milliseconds(100),
                                                      .landing = landing});
    for (std::size_t i = 0; i < kExtents; ++i) {
      ASSERT_TRUE(scheduler_->SetSource(extents_[i], Source(i, weights_)).has_value());
    }
  }

  void TearDown() override {
    ReleaseReads();
    if (scheduler_ != nullptr) {
      // Evict what is resident, so every backing is released, then stop.
      EvictProgram::Report report;
      std::vector<ExtentId> resident;
      for (const ExtentId extent : extents_) {
        if (View(extent).state == ExtentState::kResident && View(extent).leases == 0) {
          resident.push_back(extent);
        }
      }
      if (!scheduler_->fault() &&
          scheduler_->Start(999, std::make_unique<EvictProgram>(report, resident)).has_value()) {
        Settle();
      }
      scheduler_->RequestShutdown();
      const auto give_up = std::chrono::steady_clock::now() + kPatience;
      while (!scheduler_->Stopped() && std::chrono::steady_clock::now() < give_up) {
        if (!Round()) {
          std::this_thread::yield();
        }
      }
      EXPECT_TRUE(scheduler_->Stopped().has_value());
      storage_lane_->Close();
      device_lane_->Close();
      if (backing_lane_ != nullptr) {
        backing_lane_->Close();
      }
      if (copy_lane_ != nullptr) {
        copy_lane_->Close();
      }
      for (int i = 0; i < 100; ++i) {
        (void)storage_lane_->Turn(false);
        if (backing_lane_ != nullptr) {
          (void)backing_lane_->Turn();
        }
        (void)device_lane_->SubmissionTurn();
        if (copy_lane_ != nullptr) {
          (void)copy_lane_->SubmissionTurn();
        }
        execution_.Drain();
        (void)device_lane_->CompletionTurn();
        if (copy_lane_ != nullptr) {
          (void)copy_lane_->CompletionTurn();
        }
      }
    }
  }

  std::uint64_t Slot(std::size_t i) const { return zone_base_ + (i * kSize); }
  // From the bases noted at setup: a lane's thread (the threaded test's
  // check) must not read the provider while the VMM lane changes it
  // (device_memory.h).
  std::uint64_t Place(ReservationId reservation, std::size_t i) const {
    return (reservation == weights_ ? weights_base_ : moved_base_) + (i * kSize);
  }
  PageSource Source(std::size_t i, ReservationId reservation) const {
    return PageSource{
        .read = ReadSpec{.fd = fd_, .offset = i * kSize, .memory = nullptr, .length = LengthOf(i)},
        .landed = true,
        .destination = Place(reservation, i),
        .backing = BackingPlace{.reservation = reservation,
                                .offset = Bytes(i * kSize),
                                .size = Bytes(kSize),
                                .allocation_class = kDeviceClass}};
  }

  // One turn of each lane and of the scheduler, in a fixed order. With
  // `device`, the fake device runs what is queued on its streams; without
  // `vmm`, the VMM lane (if there is one) does not turn; without
  // `device_lane`, the device lane does not (its submission thread is, as
  // far as the test goes, blocked in a launch: RE-029).
  bool Round(bool device = true, bool vmm = true, bool device_lane = true) {
    bool progress = storage_lane_->Turn(false);
    if (vmm && backing_lane_ != nullptr) {
      progress = backing_lane_->Turn() || progress;
    }
    if (device_lane) {
      progress = device_lane_->SubmissionTurn() || progress;
    }
    if (copy_lane_ != nullptr) {
      progress = copy_lane_->SubmissionTurn() || progress;
    }
    if (device) {
      execution_.Drain();
    }
    if (device_lane) {
      progress = device_lane_->CompletionTurn() || progress;
    }
    if (copy_lane_ != nullptr) {
      progress = copy_lane_->CompletionTurn() || progress;
    }
    return scheduler_->Turn() || progress;
  }
  void Settle(bool device = true) {
    for (int i = 0; i < 1000 && Round(device); ++i) {
    }
  }
  void HoldNext(int requests) {
    for (int i = 0; i < requests; ++i) {
      storage_.ScriptNext(
          {.submission = Submission::kAccepted, .result = std::nullopt, .hold = true});
    }
  }
  void ReleaseReads() {
    std::vector<std::uint64_t> tokens;
    for (const auto& request : storage_.submitted()) {
      tokens.push_back(request.token);
    }
    for (const std::uint64_t token : tokens) {
      (void)storage_.Release(token);
    }
  }

  // The extent's device bytes equal its file range (the fake's device
  // memory is host memory underneath).
  bool Loaded(std::size_t i, ReservationId reservation) const {
    return std::memcmp(At(Place(reservation, i)), file_.data() + (i * kSize), LengthOf(i)) == 0;
  }
  // Live state (write-back, D-081's reverse path): a kPreserve extent,
  // resident in backing mapped by hand at moved_'s place i (device, or
  // host for a direct place), holding Pattern(i), with its write-back
  // place in the spill file at i x kSize.
  static std::vector<std::byte> Pattern(std::size_t i) {
    std::vector<std::byte> bytes(kSize);
    for (std::size_t j = 0; j < bytes.size(); ++j) {
      bytes[j] = static_cast<std::byte>((i * 7) + (j * 3) + 11);
    }
    return bytes;
  }
  PageSource StatePlace(std::size_t i, bool landed = true) const {
    return PageSource{
        .read = ReadSpec{.fd = spill_,
                         .offset = i * kSize,
                         .memory = landed ? nullptr : At(Place(moved_, i)),
                         .length = kSize},
        .landed = landed,
        .destination = landed ? Place(moved_, i) : 0,
        .backing = BackingPlace{.reservation = moved_,
                                .offset = Bytes(i * kSize),
                                .size = Bytes(kSize),
                                .allocation_class = landed ? kDeviceClass : kHostClass},
        .write_back = true};
  }
  ExtentId AddState(std::size_t i, bool landed = true) {
    const ExtentId state = catalog_
                               .AddExtent({.domain = domain_,
                                           .memory_class = jitllm::catalog::MemoryClass::kLiveState,
                                           .recovery = jitllm::catalog::Recovery::kPreserve,
                                           .size = Bytes(kSize),
                                           .content = {}},
                                          true)
                               .value();
    const auto backing = memory_.Create(landed ? kDeviceClass : kHostClass, Bytes(kSize)).value();
    EXPECT_TRUE(memory_.Map(moved_, Bytes(i * kSize), backing).has_value());
    EXPECT_TRUE(
        memory_.SetAccess(moved_, Bytes(i * kSize), Bytes(kSize), Access::kReadWrite).has_value());
    const std::vector<std::byte> pattern = Pattern(i);
    std::memcpy(At(Place(moved_, i)), pattern.data(), kSize);
    EXPECT_TRUE(scheduler_->SetSource(state, StatePlace(i, landed)).has_value());
    return state;
  }
  bool StateIs(std::size_t i) const {
    return std::memcmp(At(Place(moved_, i)), Pattern(i).data(), kSize) == 0;
  }
  std::vector<jitllm::providers::IoRequest> Writes() const {
    std::vector<jitllm::providers::IoRequest> writes;
    for (const auto& request : storage_.submitted()) {
      if (request.kind == jitllm::providers::IoKind::kWrite) {
        writes.push_back(request);
      }
    }
    return writes;
  }
  static std::unique_ptr<TaskProgram> Evicting(EvictProgram::Report& report,
                                               std::vector<ExtentId> extents) {
    return std::make_unique<EvictProgram>(report, std::move(extents));
  }

  ExtentView View(ExtentId extent) const { return catalog_.Describe(extent).value(); }
  Occupancy Occupied() const { return catalog_.OccupancyOf(domain_); }
  Closure Of(std::initializer_list<std::size_t> which) const {
    std::vector<ExtentId> extents;
    for (const std::size_t i : which) {
      extents.push_back(extents_.at(i));
    }
    return catalog_.ClosureOfExtents(extents).value();
  }
  Closure All() const { return catalog_.ClosureOfExtents(extents_).value(); }
  static std::unique_ptr<TaskProgram> Load(LoadProgram::Report& report, Closure closure,
                                           std::optional<JobResult> job = std::nullopt) {
    return std::make_unique<LoadProgram>(report, std::move(closure), job);
  }

  static constexpr std::size_t kDeviceClass = 0;
  static constexpr std::size_t kHostClass = 1;

  // Declared in dependency order: the scheduler goes first, the providers last.
  FakeDeviceMemory memory_{Bytes(kSize), Bytes(kSize * 64)};
  bool lazy_ = false;  // SchedulerSettings::lazy_handoff, set before Build
  void HandoffMoves();
  FakeStorage storage_{8, 4096};
  FakeDeviceExecution execution_;
  jitllm::catalog::Catalog catalog_;
  jitllm::base::WakeFlag wake_;
  std::unique_ptr<CompletionBoard> board_;
  std::unique_ptr<StorageService> storage_lane_;
  std::unique_ptr<DeviceService> device_lane_;
  std::unique_ptr<BackingService> backing_lane_;
  std::unique_ptr<DeviceService> copy_lane_;
  std::unique_ptr<Scheduler> scheduler_;

  StreamId copy_stream_;
  ReservationId zone_;
  jitllm::providers::BackingId zone_backing_;
  ReservationId weights_;
  ReservationId moved_;
  std::uint64_t zone_base_ = 0;
  std::uint64_t weights_base_ = 0;
  std::uint64_t moved_base_ = 0;
  std::vector<std::byte> file_;
  int fd_ = -1;
  int spill_ = -1;
  jitllm::catalog::DomainId domain_;
  std::vector<ExtentId> extents_;
  StreamId stream_;
  std::size_t baseline_ = 0;
};

TEST_P(PageInTest, ALandedLoadIsPublishedOnlyAfterItsCopysFence) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle(false);  // the device runs nothing queued on its stream
  // Backing mapped first, then the read landed in a slot, not the extent.
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  ASSERT_EQ(storage_.submitted().size(), 4U);
  for (const auto& request : storage_.submitted()) {
    EXPECT_GE(reinterpret_cast<std::uint64_t>(request.memory), Slot(0));
    EXPECT_LT(reinterpret_cast<std::uint64_t>(request.memory), Slot(0) + kSize);
  }
  EXPECT_EQ(std::memcmp(At(Slot(0)), file_.data(), kSize), 0);
  // The copy is queued but its fence has not completed: still LOADING,
  // the slot still busy, the extent's bytes untouched.
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(Occupied().loading, Bytes(kSize));
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_EQ(*At(Place(weights_, 0)), kPoison);
  EXPECT_FALSE(report.outcome.has_value());

  Settle();
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(Occupied().idle, Bytes(kSize));
  EXPECT_EQ(execution_.fences(), 0U);
}

// A zero source (PageSource::zero, a state's spill file before anything
// was saved) maps its backing and zeroes it on the device: no read, no
// slot. A zeroing that fails unwinds as a failed copy does.
TEST_P(PageInTest, AZeroSourceZeroesItsBackingWithoutAReadOrASlot) {
  Build();
  for (const std::size_t i : {std::size_t{0}, std::size_t{1}}) {
    PageSource zero = Source(i, weights_);
    zero.zero = true;
    ASSERT_TRUE(scheduler_->SetSource(extents_[i], zero).has_value());
  }
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(storage_.submitted().empty());
  const std::vector<std::byte> zeros(kSize);
  EXPECT_EQ(std::memcmp(At(Place(weights_, 0)), zeros.data(), kSize), 0);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);

  execution_.FailNextCopy(ProviderError::kFailed);
  LoadProgram::Report failed;
  ASSERT_TRUE(scheduler_->Start(2, Load(failed, Of({1}))).has_value());
  Settle();
  EXPECT_EQ(failed.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  EXPECT_TRUE(storage_.submitted().empty());
  // Without a managed backing a zero source is refused.
  PageSource unmanaged = Source(2, weights_);
  unmanaged.zero = true;
  unmanaged.backing.reset();
  EXPECT_FALSE(scheduler_->SetSource(extents_[2], unmanaged).has_value());
}

// The zone bounds what is in flight: two slots, so at most two reads; the
// window lets two more loads map backing ahead and wait for a slot, and
// the rest wait unmapped. Slots are granted in the order loads asked.
TEST_P(PageInTest, AFullZoneHoldsLoadsBackInOrder) {
  Build();
  HoldNext(64);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  EXPECT_EQ(scheduler_->slots_busy(), kSlots);
  EXPECT_EQ(scheduler_->loads(), kExtents);
  EXPECT_EQ(memory_.backings(), baseline_ + (2 * kSlots));  // mapped ahead: the window
  // The reader started only the two slots' reads.
  std::vector<std::uint64_t> offsets;
  for (const auto& request : storage_.submitted()) {
    if (request.offset % kSize == 0) {
      offsets.push_back(request.offset);
    }
  }
  EXPECT_EQ(offsets, (std::vector<std::uint64_t>{0, kSize}));

  // Released one read at a time, the loads finish in order.
  for (int round = 0; round < 40 && !report.outcome; ++round) {
    ReleaseReads();
    Settle();
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  offsets.clear();
  for (const auto& request : storage_.submitted()) {
    if (request.offset % kSize == 0) {
      offsets.push_back(request.offset);
    }
  }
  std::vector<std::uint64_t> ordered;
  ordered.reserve(kExtents);
  for (std::size_t i = 0; i < kExtents; ++i) {
    ordered.push_back(i * kSize);
  }
  EXPECT_EQ(offsets, ordered);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_EQ(View(extents_[i]).state, ExtentState::kResident);
    EXPECT_TRUE(Loaded(i, weights_));
  }
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + kExtents);
}

// BP-P1: loads whose reads wait behind a full provider and continue one
// another in the file share one vectored request, a segment per piece
// into each load's own slot. The zone bounds a span: never more than its
// slots. A span left short at an aligned point publishes the loads it
// filled and continues the rest, in file order, and every load publishes
// only whole bytes, only after its copy's fence.
TEST_P(PageInTest, AdjacentLoadsCoalesceWithinTheZoneAndSurviveAShortSpan) {
  constexpr std::size_t kZone = 4;
  Build(kZone, 32, kSize * 64, 1U << 20U);
  storage_.SetDepth(1);  // later reads wait for room behind the first
  HoldNext(1);
  // The second request (extents 1-3) moves extent 1 and 8 KiB of extent 2.
  storage_.ScriptNext({.submission = Submission::kAccepted,
                       .result = static_cast<std::int64_t>(kSize + 8192),
                       .hold = false});
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  ASSERT_EQ(storage_.submitted().size(), 1U);  // extent 0's four pieces, held
  EXPECT_EQ(storage_.submitted()[0].segments.size(), 4U);
  EXPECT_EQ(scheduler_->slots_busy(), kZone);
  for (int round = 0; round < 40 && !report.outcome; ++round) {
    ReleaseReads();
    Settle();
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_EQ(View(extents_[i]).state, ExtentState::kResident) << i;
    EXPECT_TRUE(Loaded(i, weights_)) << i;
  }
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  const auto& requests = storage_.submitted();
  ASSERT_GE(requests.size(), 3U);
  // Extents 1-3 waited together: one request for their twelve pieces.
  EXPECT_EQ(requests[1].offset, kSize);
  EXPECT_EQ(requests[1].segments.size(), 12U);
  // What the short span left: extent 2 from 8 KiB on, with extent 3.
  EXPECT_EQ(requests[2].offset, (2 * kSize) + 8192);
  std::size_t pieces = 0;
  std::uint64_t reached = 0;
  for (const auto& request : requests) {
    // Reads start in file order (RE-026), each into slots of the zone,
    // never more than the zone holds.
    EXPECT_GE(request.offset, reached);
    reached = request.offset;
    EXPECT_LE(request.length, kZone * kSize);
    const std::size_t segments = std::max<std::size_t>(request.segments.size(), 1);
    pieces += segments;
    for (const auto& segment : request.segments) {
      const auto address = reinterpret_cast<std::uint64_t>(segment.memory);
      EXPECT_GE(address, Slot(0));
      EXPECT_LE(address + segment.length, Slot(kZone));
    }
  }
  EXPECT_LT(requests.size(), pieces);  // fewer requests than pieces
}

// Withdrawing one load whose read shares a span with another's never
// cancels the span: the neighbour publishes whole bytes, and the withdrawn
// load keeps its slot until the span completes, then unwinds.
TEST_P(PageInTest, CancellingOneLoadOfASpanLeavesItsNeighbourWhole) {
  Build(4, 32, kSize * 64, 1U << 20U);
  storage_.SetDepth(1);
  HoldNext(2);  // extent 0's read, then the span of extents 1 and 2
  LoadProgram::Report first;
  LoadProgram::Report cancelled;
  LoadProgram::Report neighbour;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0}))).has_value());
  Settle();
  ASSERT_TRUE(scheduler_->Start(2, Load(cancelled, Of({1}))).has_value());
  ASSERT_TRUE(scheduler_->Start(3, Load(neighbour, Of({2}))).has_value());
  Settle();
  ASSERT_EQ(storage_.submitted().size(), 1U);
  ReleaseReads();
  Settle();
  EXPECT_EQ(first.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(storage_.submitted().size(), 2U);
  EXPECT_EQ(storage_.submitted()[1].offset, kSize);
  EXPECT_EQ(storage_.submitted()[1].segments.size(), 8U);
  EXPECT_EQ(scheduler_->slots_busy(), 2U);

  ASSERT_TRUE(scheduler_->Cancel(2));
  Settle();
  // The span is still in flight and not cancelled: both slots stay busy.
  EXPECT_EQ(storage_.in_flight(), 1U);
  EXPECT_EQ(scheduler_->slots_busy(), 2U);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kLoading);
  ReleaseReads();
  Settle();
  EXPECT_EQ(cancelled.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(neighbour.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(2, weights_));
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);  // extents 0 and 2
  EXPECT_EQ(storage_.in_flight(), 0U);
}

// A slot is not read into again until the copy out of it has completed.
TEST_P(PageInTest, ASlotIsReusedOnlyAfterTheCopyOutOfItCompletes) {
  Build(1);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle(false);
  // The first read is done and its copy queued; the second load waits.
  EXPECT_EQ(storage_.submitted().size(), 4U);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kLoading);
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  execution_.Drain();  // the copy runs, then its fence
  Settle(false);
  // Its fence completed: published, and the slot handed on.
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(storage_.submitted().size(), 8U);
  EXPECT_EQ(storage_.submitted().back().offset, kSize + (3ULL * 16 * 1024));
  Settle();
  EXPECT_TRUE(Loaded(1, weights_));
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
}

TEST_P(PageInTest, AFailedReadReleasesItsSlotAndItsBacking) {
  Build();
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  LoadProgram::Report report;
  const std::uint64_t generation = View(extents_[0]).backing_generation;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  const ExtentView failed = View(extents_[0]);
  EXPECT_EQ(failed.state, ExtentState::kNonresident);
  EXPECT_EQ(failed.backing_generation, generation + 1);
  EXPECT_EQ(memory_.backings(), baseline_);  // unmapped and released
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_FALSE(scheduler_->fault().has_value());

  // A short read fails the same way, and the extent loads again later.
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = 4096, .hold = false});
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = 0, .hold = false});
  LoadProgram::Report again;
  ASSERT_TRUE(scheduler_->Start(2, Load(again, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(again.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(memory_.backings(), baseline_);
  LoadProgram::Report third;
  ASSERT_TRUE(scheduler_->Start(3, Load(third, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(third.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(0, weights_));
}

TEST_P(PageInTest, CancellingDuringTheReadDrainsItThenUnwinds) {
  Build();
  HoldNext(4);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_TRUE(report.retired);  // the load is the scheduler's, not the task's
  // Nothing is released until the read has drained.
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  Settle();  // the lane cancels the held requests; they complete
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes());
}

// A copy cannot be cancelled: the load waits for its fence, then
// publishes the whole contents, and only then frees the slot.
TEST_P(PageInTest, CancellingMidCopyWaitsForTheFence) {
  Build(1);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle(false);
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_TRUE(report.retired);
  // A second request waits for the one slot the copy still holds.
  LoadProgram::Report other;
  ASSERT_TRUE(scheduler_->Start(2, Load(other, Of({1}))).has_value());
  Settle(false);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_EQ(storage_.submitted().size(), 4U);
  EXPECT_EQ(*At(Place(weights_, 0)), kPoison);
  Settle();
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(other.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(1, weights_));
}

// A request cancelled while its loads wait for the window or a slot
// unwinds them at once: nothing was read, and backing mapped ahead is
// released.
TEST_P(PageInTest, CancellingWaitingLoadsReleasesWhatTheyMapped) {
  Build(1);
  HoldNext(64);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  EXPECT_EQ(scheduler_->loads(), kExtents);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);  // one reading, one mapped ahead
  ASSERT_TRUE(scheduler_->Cancel(1));
  // The queued ones ended at once; the read in flight and the backing
  // mapped ahead stay until the read drains and the unmap completes.
  EXPECT_EQ(scheduler_->loads(), 2U);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  Settle();
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(Occupied().Total(), Bytes());
  for (const ExtentId extent : extents_) {
    EXPECT_EQ(View(extent).state, ExtentState::kNonresident);
  }
}

// A copy whose fence cannot be proven leaves the slot it read and the
// extent it wrote undetermined: both are quarantined, never reused, and
// the node faults. Loads waiting for that slot can only be withdrawn.
TEST_P(PageInTest, AnUnprovenCopyQuarantinesItsSlotAndItsExtent) {
  Build(1);
  execution_.FailNextQuery(ProviderError::kUnknown, 1000);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->slots_quarantined(), 1U);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  execution_.FailNextQuery(ProviderError::kUnknown, 0);
  EXPECT_EQ(Failed(scheduler_->Start(2, Load(report, Of({1})))),
            jitllm::scheduler::StartError::kStopped);  // admission has stopped
}

TEST_P(PageInTest, BackingThatCannotBeMadeChangesNothing) {
  Build();
  const std::uint64_t reads = storage_.submitted().size();
  memory_.FailNext(jitllm::providers::fake::Operation::kCreate, ProviderError::kOutOfMemory);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(storage_.submitted().size(), reads);  // nothing read
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);

  // A mapping refused after the backing was made: the backing is released.
  memory_.FailNext(jitllm::providers::fake::Operation::kMap, ProviderError::kFailed);
  LoadProgram::Report mapped;
  ASSERT_TRUE(scheduler_->Start(2, Load(mapped, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(mapped.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);

  // Access refused: unmapped and released.
  memory_.FailNext(jitllm::providers::fake::Operation::kSetAccess, ProviderError::kFailed);
  LoadProgram::Report access;
  ASSERT_TRUE(scheduler_->Start(3, Load(access, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(access.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

TEST_P(PageInTest, BackingOfUnknownOutcomeIsQuarantined) {
  Build();
  memory_.FailNext(jitllm::providers::fake::Operation::kCreate, ProviderError::kUnknown, true);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
}

// Eviction unmaps and releases the backing on the device lane (D-033); a
// later load maps fresh backing and restores the same bytes, at the same
// place or, registered again, at another (relocation, BP-P5).
TEST_P(PageInTest, EvictionReleasesBackingAndReloadsAreIdentical) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(memory_.backings(), baseline_ + kExtents);
  const std::uint64_t generation = View(extents_[2]).backing_generation;
  const std::uint64_t contents = View(extents_[2]).content_generation;

  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, extents_)).has_value());
  ASSERT_TRUE(scheduler_->Turn());
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kEvicting);  // until its unmap completes
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(evicted.results.size(), kExtents);
  EXPECT_EQ(evicted.results[0], Readiness::kWaiting);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[2]).backing_generation, generation + 1);
  EXPECT_EQ(View(extents_[2]).content_generation, contents);

  LoadProgram::Report reloaded;
  ASSERT_TRUE(scheduler_->Start(3, Load(reloaded, All())).has_value());
  Settle();
  ASSERT_EQ(reloaded.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_TRUE(Loaded(i, weights_));
  }

  // Relocation: refused while resident at the old place, allowed once
  // evicted; the next load lands at the new one.
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], Source(0, moved_))), WorkError::kBusy);
  EvictProgram::Report again;
  ASSERT_TRUE(scheduler_->Start(4, std::make_unique<EvictProgram>(again, extents_)).has_value());
  Settle();
  ASSERT_EQ(again.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    ASSERT_TRUE(scheduler_->SetSource(extents_[i], Source(i, moved_)).has_value());
  }
  LoadProgram::Report relocated;
  ASSERT_TRUE(scheduler_->Start(5, Load(relocated, All())).has_value());
  Settle();
  ASSERT_EQ(relocated.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_TRUE(Loaded(i, moved_));
  }
  EXPECT_EQ(memory_.backings(), baseline_ + kExtents);
}

// A materialization that meets an eviction in flight waits for it, then
// loads again.
TEST_P(PageInTest, MaterializingDuringAnEvictionWaitsThenReloads) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  LoadProgram::Report after;
  ASSERT_TRUE(scheduler_->Start(3, Load(after, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(after.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
}

// An unmap refused with nothing changed abandons the eviction: the evictor
// hears of the failure, and a task that met the eviction while
// materializing finds the extent resident again, unharmed.
TEST_P(PageInTest, ARefusedUnmapLeavesTheExtentResidentForItsReaders) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kFailed);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  LoadProgram::Report reader;
  ASSERT_TRUE(scheduler_->Start(3, Load(reader, Of({0}))).has_value());
  ASSERT_TRUE(scheduler_->Turn());  // both step: the eviction starts, the reader joins it
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kEvicting);
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(reader.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

TEST_P(PageInTest, AnUnmapOfUnknownOutcomeQuarantinesTheEviction) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
}

// An unknown outcome leaves the whole reservation undetermined, and the
// provider refuses every later call on it. Refusing an unmap there changed
// nothing, but it proves nothing about the place either: the extent is
// quarantined, still charged, never resident again for readers.
TEST_P(PageInTest, AnUnmapRefusedAsUndeterminedQuarantinesTheEviction) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle();
  ASSERT_EQ(report.outcome, TaskOutcome::kSucceeded);
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  // Both unmaps queued in this order before the device lane runs either.
  EvictProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(2, std::make_unique<EvictProgram>(first, std::vector{extents_[1]}))
                  .has_value());
  ASSERT_TRUE(scheduler_->Turn());
  EvictProgram::Report second;
  ASSERT_TRUE(scheduler_->Start(3, std::make_unique<EvictProgram>(second, std::vector{extents_[0]}))
                  .has_value());
  ASSERT_TRUE(scheduler_->Turn());
  Settle();
  EXPECT_TRUE(memory_.Undetermined(weights_));
  EXPECT_EQ(first.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(second.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kQuarantined);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(2 * kSize));
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
}

// A task that met an eviction in flight, and waits for it, hears of a
// failure when that eviction is quarantined, and the extent stays
// unavailable.
TEST_P(PageInTest, ATaskWaitingOnAnEvictionThatIsQuarantinedFails) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  LoadProgram::Report reader;
  ASSERT_TRUE(scheduler_->Start(3, Load(reader, Of({0}))).has_value());
  ASSERT_TRUE(scheduler_->Turn());  // the eviction starts; the reader joins it
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kEvicting);
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(reader.outcome, TaskOutcome::kFailed);
  EXPECT_TRUE(reader.retired);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->loads(), 0U);
}

// A withdrawn load starts no new stage (docs/async-model.md). Here the
// device lane is full, so the first load's copy waits unpublished, while
// the storage lane has room. Withdrawing both loads rolls that copy back,
// which frees its slot: the second load, waiting for it and withdrawn in
// the same step, must unwind, not start a read into it.
TEST_P(PageInTest, AWithdrawnLoadStartsNoNewStage) {
  Build(1);
  HoldNext(4);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle();
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_EQ(memory_.backings(), baseline_ + 2);  // the second is mapped, waiting for the slot
  ASSERT_EQ(storage_.submitted().size(), 4U);
  // Fill the device lane with commands that name no operation.
  int fillers = 0;
  while (device_lane_->Submit(jitllm::scheduler::DeviceCommand{
             .operation = {}, .work = jitllm::scheduler::DeviceWork{}}) == PushResult::kAccepted) {
    ++fillers;
  }
  ASSERT_GT(fillers, 0);
  // The first read completes; its copy cannot be published.
  ReleaseReads();
  for (int i = 0; i < 20; ++i) {
    (void)storage_lane_->Turn(false);
    (void)scheduler_->Turn();
  }
  ASSERT_EQ(View(extents_[0]).state, ExtentState::kLoading);
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  Settle();
  for (const auto& request : storage_.submitted()) {
    EXPECT_LT(request.offset, kSize) << "a read for a withdrawn load";
  }
  EXPECT_EQ(storage_.submitted().size(), 4U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// Every mailbox held by quarantined work, with a withdrawn load that must
// still unmap what it mapped: no mailbox will ever come, so the load is
// quarantined (its backing stays charged) and the stop reports the fault
// instead of waiting forever.
TEST_P(PageInTest, AnUnwindThatCanNeverGetAMailboxFaultsTheStop) {
  Build(1, 2);
  LoadProgram::Report resident;
  ASSERT_TRUE(scheduler_->Start(1, Load(resident, Of({2}))).has_value());
  Settle();
  ASSERT_EQ(resident.outcome, TaskOutcome::kSucceeded);
  // The first load's copy is queued, its fence not complete; the second is
  // mapped and waits for the one slot.
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(2, Load(report, Of({0, 1}))).has_value());
  Settle(false);
  ASSERT_EQ(scheduler_->loads(), 2U);
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  ASSERT_EQ(scheduler_->operations(), 1U);
  // An unmap of unknown outcome keeps the other mailbox for good ...
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(3, std::make_unique<EvictProgram>(evicted, std::vector{extents_[2]}))
          .has_value());
  Settle(false);
  ASSERT_EQ(View(extents_[2]).state, ExtentState::kQuarantined);
  // ... and so does the copy, once its fence cannot be proven. The second
  // load still waits for the slot, which is quarantined too; the stop
  // cancels its request, and it has no mailbox to unmap with.
  execution_.FailNextQuery(ProviderError::kUnknown, 1000);
  Settle();
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(scheduler_->slots_quarantined(), 1U);
  scheduler_->RequestShutdown();
  for (int i = 0; i < 1000 && !scheduler_->Stopped(); ++i) {
    (void)Round();
  }
  const auto stopped = scheduler_->Stopped();
  ASSERT_TRUE(stopped.has_value()) << "the stop waits forever for a mailbox";
  EXPECT_TRUE(stopped.has_value() && !stopped->has_value());  // the stop reports a fault
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(3 * kSize));
  execution_.FailNextQuery(ProviderError::kUnknown, 0);
}

// The counterpart: one mailbox is held for good, but the other by a copy
// that will complete. The withdrawn load's unmap waits for that mailbox
// instead of being quarantined, and runs once the copy is concluded. (The
// unmap of unknown outcome is in another reservation, which it leaves
// undetermined; the withdrawn load's is unharmed.)
TEST_P(PageInTest, AnUnwindWaitsForAMailboxThatWillFree) {
  Build(1, 2);
  ASSERT_TRUE(scheduler_->SetSource(extents_[2], Source(2, moved_)).has_value());
  LoadProgram::Report resident;
  ASSERT_TRUE(scheduler_->Start(1, Load(resident, Of({2}))).has_value());
  Settle();
  ASSERT_EQ(resident.outcome, TaskOutcome::kSucceeded);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(2, Load(report, Of({0, 1}))).has_value());
  Settle(false);
  ASSERT_EQ(scheduler_->loads(), 2U);
  ASSERT_EQ(scheduler_->operations(), 1U);
  memory_.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(3, std::make_unique<EvictProgram>(evicted, std::vector{extents_[2]}))
          .has_value());
  Settle(false);
  ASSERT_EQ(View(extents_[2]).state, ExtentState::kQuarantined);
  ASSERT_TRUE(scheduler_->Cancel(2));
  for (int i = 0; i < 20; ++i) {
    (void)Round(false);  // the copy's fence stays pending
  }
  EXPECT_EQ(scheduler_->loads(), 2U);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kLoading);
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);  // its copy completed whole
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(memory_.backings(), baseline_ + 2);  // the quarantined one, and extent 0's
  EXPECT_EQ(scheduler_->slots_quarantined(), 0U);
}

// A withdrawal that lands in the same turn as its copy's completion: the
// copy publishes whole bytes and frees the slot, the other withdrawn load
// unwinds without a read, and the next request's queued load starts.
TEST_P(PageInTest, AWithdrawalMeetingItsCopysCompletionHandsTheSlotOn) {
  Build(1);
  LoadProgram::Report withdrawn;
  ASSERT_TRUE(scheduler_->Start(1, Load(withdrawn, Of({0, 1}))).has_value());
  LoadProgram::Report next;
  ASSERT_TRUE(scheduler_->Start(2, Load(next, Of({2}))).has_value());
  Settle(false);
  ASSERT_EQ(scheduler_->loads(), 3U);
  ASSERT_EQ(scheduler_->copying(), 1U);
  ASSERT_EQ(View(extents_[2]).state, ExtentState::kLoading);
  ASSERT_EQ(memory_.backings(), baseline_ + 2);  // the third waits, unmapped
  // The fence completes and is posted, not yet harvested; then the cancel.
  execution_.Drain();
  (void)device_lane_->CompletionTurn();
  ASSERT_TRUE(scheduler_->Cancel(1));
  Settle();
  EXPECT_EQ(withdrawn.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(next.outcome, TaskOutcome::kSucceeded);
  for (const auto& request : storage_.submitted()) {
    EXPECT_TRUE(request.offset < kSize || request.offset >= 2 * kSize)
        << "a read for a withdrawn load";
  }
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(2, weights_));
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// An unknown outcome on one map leaves the reservation undetermined, and
// the next map into it, for another extent, is refused (kUndetermined)
// before the driver is asked. That refusal changed nothing: its load fails
// cleanly with the new backing released, and its extent is not quarantined.
TEST_P(PageInTest, AMapRefusedAsUndeterminedFailsItsLoadCleanly) {
  Build();
  memory_.FailNext(jitllm::providers::fake::Operation::kMap, ProviderError::kUnknown);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0, 1}))).has_value());
  Settle();
  EXPECT_TRUE(memory_.Undetermined(weights_));
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(Occupied().Total(), Bytes(kSize));
  EXPECT_EQ(memory_.backings(), baseline_ + 1);  // the first load's, charged
  EXPECT_EQ(storage_.submitted().size(), 0U);    // neither load read
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
}

// A kernel job's lease holds its closure until the fence after it has
// completed, even once its request is cancelled; eviction is refused
// meanwhile (invariant 2). A job that queued nothing releases at once.
TEST_P(PageInTest, AJobsLeaseHoldsUntilItsFence) {
  Build();
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}), JobResult::kQueued)).has_value());
  Settle(false);
  execution_.Drain();  // the load's copy and its fence
  Settle(false);
  ASSERT_EQ(report.job_runs, 1);
  EXPECT_EQ(View(extents_[0]).leases, 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_FALSE(report.retired);  // its operation is still in flight
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle(false);
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  ASSERT_EQ(evicted.results.size(), 1U);
  EXPECT_EQ(Failed(evicted.results.front()), WorkError::kBusy);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  Settle();  // the fence completes
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(View(extents_[0]).leases, 0U);

  LoadProgram::Report refused;
  ASSERT_TRUE(scheduler_->Start(3, Load(refused, Of({0}), JobResult::kNotStarted)).has_value());
  Settle();
  EXPECT_EQ(refused.job_runs, 1);
  EXPECT_EQ(refused.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).leases, 0U);
  EXPECT_EQ(execution_.fences(), 0U);
}

// A job's first copy whose outcome the provider reports unknown may still
// run and read the job's closure, so the job reports it unknown
// (AfterRefusal), never not started: the lease holds, and eviction is
// refused, until the fence after it completes. A first copy the provider
// refused (a known failure) queued nothing, and releases at once.
TEST_P(PageInTest, AnUnknownFirstCopyHoldsItsLeaseUntilItsFence) {
  Build();
  std::vector<std::byte> out(kSize, std::byte{0});
  const std::uint64_t source = Place(weights_, 0);
  const auto destination = reinterpret_cast<std::uint64_t>(out.data());
  const auto copying = [this, source, destination](ProviderError error) -> DeviceJob {
    return [this, source, destination, error](jitllm::providers::NativeStream) {
      execution_.FailNextCopy(error);
      const auto copied = execution_.Copy(stream_, destination, source, Bytes(kSize));
      return copied ? JobResult::kQueued : AfterRefusal(copied.error().error, false);
    };
  };
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_
                  ->Start(1, std::make_unique<LoadProgram>(report, Of({0}),
                                                           copying(ProviderError::kUnknown)))
                  .has_value());
  Settle(false);
  execution_.Drain();  // the load's copy and its fence
  Settle(false);
  ASSERT_EQ(report.job_runs, 1);
  // The copy is queued but has not run: the operation is open, its lease
  // held, and the extent cannot be evicted.
  EXPECT_EQ(out.front(), std::byte{0});
  EXPECT_FALSE(report.outcome.has_value());
  EXPECT_FALSE(report.retired);
  EXPECT_EQ(View(extents_[0]).leases, 1U);
  EXPECT_FALSE(jitllm::catalog::Catalog::Evictable(View(extents_[0])));
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle(false);
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  ASSERT_EQ(evicted.results.size(), 1U);
  EXPECT_EQ(Failed(evicted.results.front()), WorkError::kBusy);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  Settle();  // the copy runs, then its fence completes
  EXPECT_EQ(std::memcmp(out.data(), file_.data(), kSize), 0);
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(View(extents_[0]).leases, 0U);
  EXPECT_FALSE(scheduler_->fault().has_value());

  std::ranges::fill(out, std::byte{0});
  LoadProgram::Report refused;
  ASSERT_TRUE(scheduler_
                  ->Start(3, std::make_unique<LoadProgram>(refused, Of({0}),
                                                           copying(ProviderError::kFailed)))
                  .has_value());
  Settle(false);
  EXPECT_EQ(refused.job_runs, 1);
  EXPECT_EQ(refused.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[0]).leases, 0U);
  Settle();
  EXPECT_EQ(out.front(), std::byte{0});  // nothing was queued
  EXPECT_EQ(execution_.fences(), 0U);
}

// A direct source reads into managed host backing the CPU maps (the token
// table the embedding lookup reads, D-081): no slot, no copy.
TEST_P(PageInTest, DirectSourcesReadIntoTheirOwnHostBacking) {
  Build();
  const ReservationId host = memory_.Reserve(Bytes(kSize)).value();
  const std::uint64_t address = memory_.RangeOf(host).value().base;
  ASSERT_TRUE(
      scheduler_
          ->SetSource(
              extents_[0],
              PageSource{
                  .read = ReadSpec{.fd = fd_, .offset = 0, .memory = At(address), .length = kSize},
                  .landed = false,
                  .destination = 0,
                  .backing = BackingPlace{.reservation = host,
                                          .offset = Bytes(0),
                                          .size = Bytes(kSize),
                                          .allocation_class = kHostClass}})
          .has_value());
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle(false);
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(std::memcmp(At(address), file_.data(), kSize), 0);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(execution_.fences(), 0U);  // nothing was copied
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[0]}))
          .has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(memory_.backings(), baseline_);
}

TEST_P(PageInTest, SourcesAreChecked) {
  Build();
  PageSource source = Source(0, weights_);
  source.read.length = kSize + 4096;  // larger than a slot
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], source)), WorkError::kInvalid);
  source = Source(0, weights_);
  source.destination = 0;
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], source)), WorkError::kInvalid);
  source = Source(0, weights_);
  source.landed = false;  // direct, with nowhere to land
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], source)), WorkError::kInvalid);
  // While a load is in flight, the source cannot move.
  HoldNext(4);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], Source(0, moved_))), WorkError::kBusy);
  ReleaseReads();
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
}

// Write-back (scheduler.h; D-081's reverse path, BP-P4): evicting live
// state copies it into a slot, fenced, writes the slot to its place, and
// only then unmaps and releases the backing. The contents keep their
// generation, marked preserved, and a load restores them into fresh
// backing, which the fake fills with poison, so every byte must come from
// the place (invariant 4).
TEST_P(PageInTest, WriteBackPreservesStateAcrossEvictionAndRestore) {
  Build();
  const ExtentId state = AddState(0);
  const Closure before = catalog_.ClosureOfExtents(std::vector{state}).value();
  const std::uint64_t contents = View(state).content_generation;
  const std::uint64_t generation = View(state).backing_generation;
  const std::size_t with_state = memory_.backings();
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(1, Evicting(evicted, {state})).has_value());
  Settle(false);  // the device runs nothing: the copy-out's fence is pending
  EXPECT_EQ(View(state).state, ExtentState::kEvicting);
  EXPECT_EQ(Occupied().evicting, Bytes(kSize));
  EXPECT_EQ(scheduler_->slots_busy(), 1U);
  EXPECT_TRUE(Writes().empty());  // nothing written before the copy's fence
  EXPECT_EQ(memory_.backings(), with_state);
  EXPECT_FALSE(evicted.outcome.has_value());

  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  const auto writes = Writes();
  ASSERT_EQ(writes.size(), 4U);  // 64 KiB in 16 KiB requests, from the slot
  for (const auto& write : writes) {
    EXPECT_EQ(write.fd, spill_);
    EXPECT_GE(reinterpret_cast<std::uint64_t>(write.memory), Slot(0));
    EXPECT_LT(reinterpret_cast<std::uint64_t>(write.memory), Slot(0) + kSize);
  }
  EXPECT_EQ(std::memcmp(storage_.Contents(spill_).data(), Pattern(0).data(), kSize), 0);
  EXPECT_EQ(View(state).state, ExtentState::kNonresident);
  EXPECT_TRUE(View(state).preserved);
  EXPECT_EQ(View(state).content_generation, contents);
  EXPECT_EQ(View(state).backing_generation, generation + 1);
  EXPECT_EQ(memory_.backings(), with_state - 1);  // released, not pooled (D-033)
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes());

  // The closure taken before the eviction is still current: restored
  // through the zone into fresh backing.
  LoadProgram::Report restored;
  ASSERT_TRUE(scheduler_->Start(2, Load(restored, before)).has_value());
  Settle();
  EXPECT_EQ(restored.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(state).state, ExtentState::kResident);
  EXPECT_TRUE(StateIs(0));
  EXPECT_FALSE(View(state).preserved);  // the resident copy is the live one
  EXPECT_EQ(View(state).content_generation, contents);
  EXPECT_EQ(memory_.backings(), with_state);
  EXPECT_FALSE(scheduler_->fault().has_value());

  // And again: a second write-back and restore keep every byte.
  EvictProgram::Report again;
  ASSERT_TRUE(scheduler_->Start(3, Evicting(again, {state})).has_value());
  Settle();
  ASSERT_EQ(again.outcome, TaskOutcome::kSucceeded);
  LoadProgram::Report restored_again;
  ASSERT_TRUE(scheduler_->Start(4, Load(restored_again, before)).has_value());
  Settle();
  EXPECT_EQ(restored_again.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(StateIs(0));
}

// Incremental spill: restored from its place and not written since (its
// owner's word, EvictOptions::unchanged, and the catalog's saved
// generation), a write-back writes nothing and still completes preserved,
// so a load restores the same bytes. Never written back, or with contents
// replaced since, the word is not enough: it is written.
TEST_P(PageInTest, AnUnchangedWriteBackWritesNothingAndStillRestores) {
  Build();
  const ExtentId state = AddState(0);
  const Closure before = catalog_.ClosureOfExtents(std::vector{state}).value();
  const jitllm::scheduler::EvictOptions unchanged{.unchanged = true};
  // Never saved: written, word or not.
  EXPECT_EQ(View(state).saved_generation, 0U);
  EvictProgram::Report first;
  ASSERT_TRUE(
      scheduler_->Start(1, std::make_unique<EvictProgram>(first, std::vector{state}, unchanged))
          .has_value());
  Settle();
  ASSERT_EQ(first.outcome, TaskOutcome::kSucceeded);
  const std::size_t written = Writes().size();
  EXPECT_GT(written, 0U);
  EXPECT_EQ(View(state).saved_generation, View(state).content_generation);
  LoadProgram::Report restored;
  ASSERT_TRUE(scheduler_->Start(2, Load(restored, before)).has_value());
  Settle();
  ASSERT_EQ(restored.outcome, TaskOutcome::kSucceeded);
  // Unchanged since: nothing written, preserved, restored whole.
  EvictProgram::Report again;
  ASSERT_TRUE(
      scheduler_->Start(3, std::make_unique<EvictProgram>(again, std::vector{state}, unchanged))
          .has_value());
  Settle();
  ASSERT_EQ(again.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(Writes().size(), written);
  EXPECT_EQ(View(state).state, ExtentState::kNonresident);
  EXPECT_TRUE(View(state).preserved);
  EXPECT_EQ(scheduler_->stats().unchanged_writebacks, 1U);
  LoadProgram::Report restored_again;
  ASSERT_TRUE(scheduler_->Start(4, Load(restored_again, before)).has_value());
  Settle();
  ASSERT_EQ(restored_again.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(StateIs(0));
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// A direct place (host backing the storage lane reaches) is written from
// the extent's own memory, with no slot.
TEST_P(PageInTest, ADirectWriteBackNeedsNoSlot) {
  Build();
  const ExtentId state = AddState(1, false);
  const Closure before = catalog_.ClosureOfExtents(std::vector{state}).value();
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(1, Evicting(evicted, {state})).has_value());
  Settle();
  ASSERT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  for (const auto& write : Writes()) {
    EXPECT_GE(reinterpret_cast<std::uint64_t>(write.memory), Place(moved_, 1));
    EXPECT_LT(reinterpret_cast<std::uint64_t>(write.memory), Place(moved_, 1) + kSize);
  }
  EXPECT_TRUE(View(state).preserved);
  LoadProgram::Report restored;
  ASSERT_TRUE(scheduler_->Start(2, Load(restored, before)).has_value());
  Settle();
  EXPECT_EQ(restored.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(StateIs(1));
}

// D-050's "spill full or failed" (M2, injected): a write that fails or
// stops short abandons the eviction. The backing was never touched, so the
// state is resident again with its contents, nothing claims they were
// preserved, and a load that needed the room fails explicitly for want of
// budget, with no quarantine and no fault. Once the spill succeeds, the
// room is there.
TEST_P(PageInTest, AFailedSpillLeavesTheStateResidentAndTheShortfallExplicit) {
  Build(kSlots, 32, 2 * kSize);
  LoadProgram::Report weights;
  ASSERT_TRUE(scheduler_->Start(1, Load(weights, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(weights.outcome, TaskOutcome::kSucceeded);
  const ExtentId state = AddState(0);
  const std::uint64_t contents = View(state).content_generation;
  const std::size_t with_state = memory_.backings();
  ASSERT_EQ(Occupied().Total(), Bytes(2 * kSize));  // the budget, exactly

  LoadProgram::Report blocked;
  ASSERT_TRUE(scheduler_->Start(2, Load(blocked, Of({1}))).has_value());
  Settle();
  EXPECT_EQ(blocked.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(blocked.error, WorkError::kOverBudget);

  for (const std::int64_t result : {std::int64_t{-ENOSPC}, std::int64_t{0}}) {
    storage_.ScriptNext({.submission = Submission::kAccepted, .result = result, .hold = false});
    EvictProgram::Report evicted;
    ASSERT_TRUE(scheduler_->Start(3, Evicting(evicted, {state})).has_value());
    Settle();
    EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed) << result;
    EXPECT_EQ(View(state).state, ExtentState::kResident);
    EXPECT_FALSE(View(state).preserved);
    EXPECT_EQ(View(state).content_generation, contents);
    EXPECT_TRUE(StateIs(0));
    EXPECT_EQ(memory_.backings(), with_state);
    EXPECT_EQ(scheduler_->slots_busy(), 0U);
    EXPECT_EQ(Occupied().quarantined, Bytes());
    LoadProgram::Report still;
    ASSERT_TRUE(scheduler_->Start(4, Load(still, Of({1}))).has_value());
    Settle();
    EXPECT_EQ(still.error, WorkError::kOverBudget);
  }
  EXPECT_FALSE(scheduler_->fault().has_value());

  EvictProgram::Report spilled;
  ASSERT_TRUE(scheduler_->Start(5, Evicting(spilled, {state})).has_value());
  Settle();
  ASSERT_EQ(spilled.outcome, TaskOutcome::kSucceeded);
  LoadProgram::Report fits;
  ASSERT_TRUE(scheduler_->Start(6, Load(fits, Of({1}))).has_value());
  Settle();
  EXPECT_EQ(fits.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(1, weights_));
}

// Invalidated state needs no write-back: its eviction writes nothing and
// advances the content generation, so the old closure is stale and a new
// one finds nothing preserved at the place (invariant 4).
TEST_P(PageInTest, InvalidatedStateIsNeverWrittenBackOrRestored) {
  Build();
  const ExtentId state = AddState(0);
  const Closure before = catalog_.ClosureOfExtents(std::vector{state}).value();
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(1, Evicting(evicted, {state})).has_value());
  Settle();
  ASSERT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Writes().empty());
  EXPECT_EQ(View(state).state, ExtentState::kNonresident);
  EXPECT_FALSE(View(state).preserved);
  LoadProgram::Report stale;
  ASSERT_TRUE(scheduler_->Start(2, Load(stale, before)).has_value());
  Settle();
  EXPECT_EQ(stale.error, WorkError::kStale);
  LoadProgram::Report nothing;
  ASSERT_TRUE(
      scheduler_->Start(3, Load(nothing, catalog_.ClosureOfExtents(std::vector{state}).value()))
          .has_value());
  Settle();
  EXPECT_EQ(nothing.error, WorkError::kUnavailable);
}

// A copy-out whose fence cannot be proven quarantines the eviction and
// its slot: the node faults rather than reuse either.
TEST_P(PageInTest, AnUnprovenCopyOutQuarantinesTheEvictionAndItsSlot) {
  Build(1);
  const ExtentId state = AddState(0);
  execution_.FailNextQuery(ProviderError::kUnknown, 1000);
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(1, Evicting(evicted, {state})).has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(state).state, ExtentState::kQuarantined);
  EXPECT_FALSE(View(state).preserved);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->slots_quarantined(), 1U);
  EXPECT_TRUE(Writes().empty());
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  execution_.FailNextQuery(ProviderError::kUnknown, 0);
}

// A write-back waits for a slot in order with loads; its evictor
// cancelled mid-write, the write still drains and the eviction completes.
TEST_P(PageInTest, AWriteBackWaitsForASlotAndOutlivesItsEvictor) {
  Build(1);
  const ExtentId state = AddState(0);
  HoldNext(4);  // the load's reads
  LoadProgram::Report loading;
  ASSERT_TRUE(scheduler_->Start(1, Load(loading, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(2, Evicting(evicted, {state})).has_value());
  Settle();
  EXPECT_EQ(View(state).state, ExtentState::kEvicting);  // waiting for the slot
  EXPECT_TRUE(Writes().empty());
  HoldNext(4);  // the write-back's writes, once it has the slot
  ReleaseReads();
  Settle();
  EXPECT_EQ(loading.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(Writes().size(), 4U);
  EXPECT_TRUE(scheduler_->Cancel(2));
  Settle();
  EXPECT_EQ(View(state).state, ExtentState::kEvicting);  // the write drains
  ReleaseReads();
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(View(state).state, ExtentState::kNonresident);
  EXPECT_TRUE(View(state).preserved);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// D-050's "cancelled phase, late DMA, registration still live, then
// replacement phase" (M2): a cancelled request's job keeps its lease until
// its fence completes; after that a registration still live (a buffer a
// device or NIC holds) keeps the extent from eviction, and nothing is
// released; only the registration's retirement lets the replacement evict
// and reload it.
TEST_P(PageInTest, ALiveRegistrationOutlivesACancelledPhase) {
  Build();
  LoadProgram::Report loaded;
  ASSERT_TRUE(scheduler_->Start(1, Load(loaded, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(loaded.outcome, TaskOutcome::kSucceeded);
  const auto registration = catalog_.AddRegistration(extents_[0]).value();
  LoadProgram::Report job;
  ASSERT_TRUE(scheduler_->Start(2, Load(job, Of({0}), JobResult::kQueued)).has_value());
  Settle(false);  // the job is queued; its fence has not completed
  ASSERT_EQ(job.job_runs, 1);
  EXPECT_TRUE(scheduler_->Cancel(2));
  Settle(false);
  EXPECT_EQ(View(extents_[0]).leases, 1U);  // held until the fence
  EXPECT_FALSE(job.retired);
  Settle();
  EXPECT_EQ(job.outcome, TaskOutcome::kCancelled);
  EXPECT_TRUE(job.retired);
  EXPECT_EQ(View(extents_[0]).leases, 0U);
  EXPECT_EQ(View(extents_[0]).registrations, 1U);
  const std::size_t backings = memory_.backings();
  EvictProgram::Report refused;
  ASSERT_TRUE(scheduler_->Start(3, Evicting(refused, {extents_[0]})).has_value());
  Settle();
  EXPECT_EQ(refused.outcome, TaskOutcome::kFailed);
  ASSERT_EQ(refused.results.size(), 1U);
  EXPECT_EQ(Failed(refused.results.front()), WorkError::kBusy);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_EQ(memory_.backings(), backings);
  ASSERT_TRUE(catalog_.RetireRegistration(registration).has_value());
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(4, Evicting(evicted, {extents_[0]})).has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  LoadProgram::Report replaced;
  ASSERT_TRUE(scheduler_->Start(5, Load(replaced, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(replaced.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(0, weights_));
}

TEST_P(PageInTest, WriteBackPlacesAreChecked) {
  Build();
  // Only live mutable contents have a write-back place.
  PageSource weights = StatePlace(0);
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], weights)), WorkError::kInvalid);
  // A source is read, never a write.
  PageSource written = Source(0, weights_);
  written.read.kind = jitllm::providers::IoKind::kWrite;
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[0], written)), WorkError::kInvalid);
}

// Invariant 4 across a new source: while live state is resident, its
// write-back place cannot claim its contents are somewhere they are not;
// once they are preserved, no source may name another range for them (a
// load would restore other bytes under their generation), though their
// backing may move.
TEST_P(PageInTest, APreservedPlaceCannotBeRenamed) {
  Build();
  const ExtentId state = AddState(0);
  PageSource elsewhere = StatePlace(0);
  elsewhere.destination = Place(moved_, 1);
  EXPECT_EQ(Failed(scheduler_->SetSource(state, elsewhere)), WorkError::kBusy);
  PageSource direct = StatePlace(0);
  direct.landed = false;
  direct.destination = 0;
  direct.read.memory = At(Place(moved_, 1));
  EXPECT_EQ(Failed(scheduler_->SetSource(state, direct)), WorkError::kBusy);
  // While resident, the copy there is the live one: the file range may
  // change before anything is written to it.
  PageSource moved_file = StatePlace(0);
  moved_file.read.offset = 2 * kSize;
  ASSERT_TRUE(scheduler_->SetSource(state, moved_file).has_value());
  const Closure before = catalog_.ClosureOfExtents(std::vector{state}).value();

  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(1, Evicting(evicted, {state})).has_value());
  Settle();
  ASSERT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  ASSERT_TRUE(View(state).preserved);
  EXPECT_EQ(std::memcmp(storage_.Contents(spill_).data() + (2 * kSize), Pattern(0).data(), kSize),
            0);
  PageSource other_range = moved_file;
  other_range.read.offset = 0;
  EXPECT_EQ(Failed(scheduler_->SetSource(state, other_range)), WorkError::kBusy);
  PageSource not_write_back = moved_file;
  not_write_back.write_back = false;
  EXPECT_EQ(Failed(scheduler_->SetSource(state, not_write_back)), WorkError::kBusy);
  PageSource other_file = moved_file;
  other_file.read.fd = fd_;
  EXPECT_EQ(Failed(scheduler_->SetSource(state, other_file)), WorkError::kBusy);
  // The same range, at the same place again: accepted, and restored.
  ASSERT_TRUE(scheduler_->SetSource(state, moved_file).has_value());
  LoadProgram::Report restored;
  ASSERT_TRUE(scheduler_->Start(2, Load(restored, before)).has_value());
  Settle();
  EXPECT_EQ(restored.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(StateIs(0));
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// A write-back still waiting for a slot has touched nothing: when its
// evictor leaves, it is abandoned and the state is resident again, with
// nothing written and nothing marked preserved (a stop never waits for a
// slot that may never free).
TEST_P(PageInTest, AWriteBackWaitingForASlotIsAbandonedWhenItsEvictorLeaves) {
  Build(1);
  const ExtentId state = AddState(0);
  HoldNext(4);  // the load's reads keep the only slot
  LoadProgram::Report loading;
  ASSERT_TRUE(scheduler_->Start(1, Load(loading, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(scheduler_->slots_busy(), 1U);
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(2, Evicting(evicted, {state})).has_value());
  Settle();
  ASSERT_EQ(View(state).state, ExtentState::kEvicting);
  EXPECT_TRUE(scheduler_->Cancel(2));
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(View(state).state, ExtentState::kResident);
  EXPECT_FALSE(View(state).preserved);
  EXPECT_EQ(Occupied().evicting, Bytes());
  ReleaseReads();
  Settle();
  EXPECT_EQ(loading.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Writes().empty());
  EXPECT_TRUE(StateIs(0));
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_FALSE(scheduler_->fault().has_value());
}

// On a VMM lane of its own, VMM work never holds up a copy: with the VMM
// lane stopped and an eviction's unmap queued on it, a landed load into
// backing mapped by hand is read, copied and published, while the evicted
// extent stays EVICTING until the VMM lane runs again.
class VmmLaneTest : public PageInTest {};

TEST_P(VmmLaneTest, CopiesRunWhileVmmWorkWaits) {
  Build();
  LoadProgram::Report resident;
  ASSERT_TRUE(scheduler_->Start(1, Load(resident, Of({2}))).has_value());
  Settle();
  ASSERT_EQ(resident.outcome, TaskOutcome::kSucceeded);
  // Extent 0's backing, mapped by hand: its load maps nothing.
  const auto backing = memory_.Create(kDeviceClass, Bytes(kSize)).value();
  ASSERT_TRUE(memory_.Map(weights_, Bytes(0), backing).has_value());
  ASSERT_TRUE(memory_.SetAccess(weights_, Bytes(0), Bytes(kSize), Access::kReadWrite).has_value());
  PageSource premapped = Source(0, weights_);
  premapped.backing.reset();
  ASSERT_TRUE(scheduler_->SetSource(extents_[0], premapped).has_value());
  EvictProgram::Report evicted;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<EvictProgram>(evicted, std::vector{extents_[2]}))
          .has_value());
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(3, Load(report, Of({0}))).has_value());
  for (int i = 0; i < 50; ++i) {
    (void)Round(true, false);  // the VMM lane does not turn
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kEvicting);
  EXPECT_FALSE(evicted.outcome.has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kNonresident);
  // Extent 0 stays resident at its hand-mapped backing: evict it in the
  // catalog, then release the backing here.
  EvictProgram::Report dropped;
  ASSERT_TRUE(
      scheduler_->Start(4, std::make_unique<EvictProgram>(dropped, std::vector{extents_[0]}))
          .has_value());
  Settle();
  EXPECT_EQ(dropped.outcome, TaskOutcome::kSucceeded);
  ASSERT_TRUE(memory_.Unmap(weights_, Bytes(0), Bytes(kSize)).has_value());
  ASSERT_TRUE(memory_.Release(backing).has_value());
  EXPECT_EQ(memory_.backings(), baseline_);
}

INSTANTIATE_TEST_SUITE_P(VmmWork, VmmLaneTest, ::testing::Values(true),
                         [](const auto& /*info*/) { return std::string("OnAVmmLane"); });

// The VMM lane refuses any other work as not started, and so does it all
// without a device-memory provider.
TEST(VmmLaneTest, OtherWorkAndMissingMemoryAreRefused) {
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  BackingService lane(nullptr, board, QueueSettings{.capacity = 4, .reserved = 1, .batch = 4});
  const auto copy = board.Open();
  const auto map = board.Open();
  ASSERT_EQ(lane.Submit(jitllm::scheduler::DeviceCommand{.operation = copy,
                                                         .work = jitllm::scheduler::DeviceWork{}}),
            PushResult::kAccepted);
  ASSERT_EQ(
      lane.Submit(jitllm::scheduler::DeviceCommand{
          .operation = map,
          .work = jitllm::scheduler::BackingWork{.kind = jitllm::scheduler::BackingWork::Kind::kMap,
                                                 .reservation = {},
                                                 .offset = Bytes(0),
                                                 .size = Bytes(kSize),
                                                 .allocation_class = 0}}),
      PushResult::kAccepted);
  EXPECT_TRUE(lane.Turn());
  const auto seen = board.Harvest(4);
  ASSERT_EQ(seen.size(), 2U);
  for (const auto& observation : seen) {
    EXPECT_EQ(observation.acceptance, jitllm::scheduler::Acceptance::kNotStarted);
    EXPECT_FALSE(observation.terminal.has_value());
  }
  lane.Close();
  lane.Run();  // closed and drained: returns at once
  EXPECT_FALSE(lane.Turn());
}

// The VMM lane's lazy handoff, step by step: a lazily kept backing stays
// mapped; a reuse at its own place takes it as it is; a reuse elsewhere
// unmaps it first; a created backing's map at a place a kept one holds
// moves that one out (kept, unmapped); a release unmaps before releasing.
TEST(VmmLaneTest, LazilyKeptBackingIsUnmappedByWhatTakesItsPlaceOrItself) {
  using jitllm::scheduler::BackingWork;
  FakeDeviceMemory memory{Bytes(kSize), Bytes(kSize * 8)};
  const auto place = memory.Reserve(Bytes(kSize * 2)).value();
  jitllm::base::WakeFlag wake;
  CompletionBoard board{16, wake};
  BackingService lane(&memory, board, QueueSettings{.capacity = 8, .reserved = 1, .batch = 8});
  const auto run = [&](BackingWork::Kind kind, std::uint64_t at, bool retain, bool lazy,
                       bool reuse) {
    const auto operation = board.Open();
    EXPECT_EQ(lane.Submit(
                  jitllm::scheduler::DeviceCommand{.operation = operation,
                                                   .work = BackingWork{.kind = kind,
                                                                       .reservation = place,
                                                                       .offset = Bytes(at * kSize),
                                                                       .size = Bytes(kSize),
                                                                       .allocation_class = 0,
                                                                       .retain = retain,
                                                                       .lazy = lazy,
                                                                       .reuse = reuse}}),
              PushResult::kAccepted);
    EXPECT_TRUE(lane.Turn());
    const auto seen = board.Harvest(4);
    EXPECT_EQ(seen.size(), 1U);
    const bool ok = seen.size() == 1 && seen[0].terminal.has_value() &&
                    seen[0].terminal->outcome == jitllm::scheduler::Outcome::kSucceeded;
    (void)board.Close(operation);
    return ok;
  };
  const auto mapped = [&](std::uint64_t at) {
    return memory.MappedAt(place, Bytes(at * kSize)).has_value();
  };
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 0, false, false, false));
  EXPECT_TRUE(mapped(0));
  // Kept lazily: still mapped, one kept.
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 0, true, true, false));
  EXPECT_TRUE(mapped(0));
  EXPECT_EQ(lane.stashed(), 1U);
  // Its own place again: taken as it is.
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 0, false, false, true));
  EXPECT_TRUE(mapped(0));
  EXPECT_EQ(lane.stashed(), 0U);
  EXPECT_EQ(memory.backings(), 1U);
  // Kept lazily, then reused elsewhere: unmapped where it was first.
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 0, true, true, false));
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 1, false, false, true));
  EXPECT_FALSE(mapped(0));
  EXPECT_TRUE(mapped(1));
  EXPECT_EQ(memory.backings(), 1U);
  // Kept lazily at 1; a created backing maps at 1: the kept one moves out.
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 1, true, true, false));
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 1, false, false, false));
  EXPECT_TRUE(mapped(1));
  EXPECT_EQ(lane.stashed(), 1U);
  EXPECT_EQ(memory.backings(), 2U);
  // Released, kept and unmapped; then lazily kept again and released.
  ASSERT_TRUE(run(BackingWork::Kind::kRelease, 0, false, false, false));
  EXPECT_EQ(lane.stashed(), 0U);
  EXPECT_EQ(memory.backings(), 1U);
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 1, true, true, false));
  EXPECT_TRUE(mapped(1));
  ASSERT_TRUE(run(BackingWork::Kind::kRelease, 1, false, false, false));
  EXPECT_FALSE(mapped(1));
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(lane.stashed(), 0U);
  lane.Close();
  lane.Run();
  ASSERT_TRUE(memory.Free(place).has_value());
}

// A kept backing whose unmap from where it was evicted is refused stays
// kept, behind the others, and the take tries the next; one with an
// unknown outcome is never reused. Either way the work's own place is
// untouched. In a reservation an unknown outcome left undetermined, a lazy
// park is refused as unproven, and a backing already parked there is
// neither taken as it is nor released.
TEST(VmmLaneTest, ALazilyKeptBackingThatCannotBeUnmappedIsSetAsideNotTaken) {
  using jitllm::scheduler::Acceptance;
  using jitllm::scheduler::BackingWork;
  using jitllm::scheduler::Outcome;
  FakeDeviceMemory memory{Bytes(kSize), Bytes(kSize * 8)};
  const auto out = memory.Reserve(Bytes(kSize * 2)).value();
  const auto in = memory.Reserve(Bytes(kSize * 2)).value();
  jitllm::base::WakeFlag wake;
  CompletionBoard board{16, wake};
  BackingService lane(&memory, board, QueueSettings{.capacity = 8, .reserved = 1, .batch = 8});
  enum class Seen : std::uint8_t { kSucceeded, kNotStarted, kUnproven };
  const auto run = [&](BackingWork::Kind kind, jitllm::providers::ReservationId place,
                       std::uint64_t at, bool lazy, bool reuse) {
    const auto operation = board.Open();
    EXPECT_EQ(lane.Submit(jitllm::scheduler::DeviceCommand{
                  .operation = operation,
                  .work = BackingWork{.kind = kind,
                                      .reservation = place,
                                      .offset = Bytes(at * kSize),
                                      .size = Bytes(kSize),
                                      .allocation_class = 0,
                                      .retain = kind == BackingWork::Kind::kUnmap,
                                      .lazy = lazy,
                                      .reuse = reuse}}),
              PushResult::kAccepted);
    EXPECT_TRUE(lane.Turn());
    const auto seen = board.Harvest(4);
    EXPECT_EQ(seen.size(), 1U);
    Seen result = Seen::kUnproven;
    if (seen.size() == 1 && seen[0].acceptance == Acceptance::kNotStarted) {
      result = Seen::kNotStarted;
    } else if (seen.size() == 1 && seen[0].terminal.has_value() &&
               seen[0].terminal->outcome == Outcome::kSucceeded) {
      result = Seen::kSucceeded;
    }
    (void)board.Close(operation);
    return result;
  };
  const auto mapped = [&](jitllm::providers::ReservationId place, std::uint64_t at) {
    return memory.MappedAt(place, Bytes(at * kSize)).has_value();
  };
  using Kind = BackingWork::Kind;
  // Two lazily kept at `out`, 1 the newer: its unmap is refused, so 0 maps
  // at `in` instead, and 1 stays kept, mapped.
  ASSERT_EQ(run(Kind::kMap, out, 0, false, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kMap, out, 1, false, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kUnmap, out, 0, true, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kUnmap, out, 1, true, false), Seen::kSucceeded);
  memory.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kFailed);
  EXPECT_EQ(run(Kind::kMap, in, 0, false, true), Seen::kSucceeded);
  EXPECT_TRUE(mapped(in, 0));
  EXPECT_FALSE(mapped(out, 0));
  EXPECT_TRUE(mapped(out, 1));
  EXPECT_EQ(lane.stashed(), 1U);
  // Only a refused one left: a release is refused, and it stays kept.
  memory.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kFailed);
  EXPECT_EQ(run(Kind::kRelease, out, 0, false, false), Seen::kNotStarted);
  EXPECT_TRUE(mapped(out, 1));
  EXPECT_EQ(lane.stashed(), 1U);
  EXPECT_EQ(run(Kind::kRelease, out, 0, false, false), Seen::kSucceeded);
  EXPECT_FALSE(mapped(out, 1));
  EXPECT_EQ(lane.stashed(), 0U);
  EXPECT_EQ(memory.backings(), 1U);
  // Unknown: that one is dropped, still existing, and its reservation is
  // undetermined (so is every other kept there); the map at `in` takes one
  // kept elsewhere.
  const auto other = memory.Reserve(Bytes(kSize * 2)).value();
  ASSERT_EQ(run(Kind::kMap, out, 0, false, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kMap, other, 0, false, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kUnmap, out, 0, true, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kUnmap, other, 0, true, false), Seen::kSucceeded);
  memory.FailNext(jitllm::providers::fake::Operation::kUnmap, ProviderError::kUnknown);
  EXPECT_EQ(run(Kind::kMap, in, 1, false, true), Seen::kSucceeded);
  EXPECT_TRUE(mapped(in, 1));
  EXPECT_FALSE(mapped(out, 0));
  EXPECT_TRUE(memory.Undetermined(other));
  EXPECT_FALSE(memory.Undetermined(out));
  EXPECT_EQ(lane.stashed(), 0U);
  EXPECT_EQ(memory.backings(), 3U);
  // Nothing kept for the charge the unknown one leaves: refused.
  EXPECT_EQ(run(Kind::kRelease, out, 0, false, false), Seen::kNotStarted);
  // One parked lazily, then an unknown outcome beside it leaves its
  // reservation undetermined: a lazy park there is unproven.
  const auto broken = memory.Reserve(Bytes(kSize * 2)).value();
  ASSERT_EQ(run(Kind::kMap, broken, 0, false, false), Seen::kSucceeded);
  ASSERT_EQ(run(Kind::kUnmap, broken, 0, true, false), Seen::kSucceeded);
  memory.FailNext(jitllm::providers::fake::Operation::kSetAccess, ProviderError::kUnknown);
  ASSERT_EQ(run(Kind::kMap, broken, 1, false, false), Seen::kUnproven);
  ASSERT_TRUE(memory.Undetermined(broken));
  EXPECT_EQ(run(Kind::kUnmap, broken, 1, true, false), Seen::kUnproven);
  // Its parked backing is neither taken as it is nor released.
  EXPECT_EQ(run(Kind::kMap, broken, 0, false, true), Seen::kNotStarted);
  EXPECT_EQ(run(Kind::kRelease, broken, 0, false, false), Seen::kNotStarted);
  EXPECT_EQ(lane.stashed(), 1U);
  lane.Close();
  lane.Run();
}

// The handle reserve (ReserveSettings): an idle lane creates backing up to
// its count; a map of its class and size takes one instead of creating; a
// plain unmap's backing, or a kept one released, refills a short reserve
// instead of being released; other classes or sizes never touch it; and
// ReleaseReserve releases what it holds.
TEST(VmmLaneTest, DiagnosticIntervalsAuthenticateBusyCompletionAndRefusalWithoutAHistory) {
  jitllm::scheduler::BackingCreateCounters counters(true);
  counters.Attempt(true);
  const auto busy = counters.Snapshot();
  ASSERT_TRUE(busy.timing.enabled);
  EXPECT_TRUE(busy.timing.current_create_reserve);
  ASSERT_NE(busy.timing.current_create.serial, 0U);
  EXPECT_GT(busy.timing.current_create.start_ns, 0U);
  EXPECT_EQ(busy.timing.current_create.end_ns, 0U);
  EXPECT_EQ(busy.timing.reserve.started, 1U);
  EXPECT_EQ(busy.timing.reserve.completed, 0U);
  counters.Completed(true);
  const auto completed = counters.Snapshot();
  EXPECT_EQ(completed.timing.current_create.serial, 0U);
  EXPECT_EQ(completed.timing.reserve.completed, 1U);
  EXPECT_EQ(completed.timing.reserve.last.serial, busy.timing.current_create.serial);
  EXPECT_EQ(completed.timing.reserve.last.start_ns, busy.timing.current_create.start_ns);
  EXPECT_GE(completed.timing.reserve.last.end_ns, completed.timing.reserve.last.start_ns);
  EXPECT_EQ(completed.timing.reserve.total_ns,
            completed.timing.reserve.last.end_ns - completed.timing.reserve.last.start_ns);
  counters.Attempt(false);
  counters.Completed(false);
  counters.Failed(false, {.error = ProviderError::kOutOfMemory, .detail = "diagnostic refusal"});
  const auto refused = counters.Snapshot();
  EXPECT_EQ(refused.ordinary_failures, 1U);
  EXPECT_EQ(refused.timing.ordinary.started, 1U);
  EXPECT_EQ(refused.timing.ordinary.completed, 1U);
  EXPECT_GT(refused.timing.ordinary.last.serial, completed.timing.reserve.last.serial);
  counters.BeginPark();
  EXPECT_NE(counters.Snapshot().timing.current_park.serial, 0U);
  counters.ParkMetadataDone();
  counters.ParkStashDone();
  counters.EndPark();
  const auto parked = counters.Snapshot();
  EXPECT_EQ(parked.timing.park.started, 1U);
  EXPECT_EQ(parked.timing.park.completed, 1U);
  EXPECT_EQ(parked.timing.current_park.serial, 0U);
  EXPECT_EQ(parked.timing.park.first.serial, parked.timing.park.last.serial);
  EXPECT_EQ(parked.timing.park_metadata.completed, 1U);
  EXPECT_EQ(parked.timing.park_stash.completed, 1U);
  EXPECT_EQ(parked.timing.park_publication.completed, 1U);
  EXPECT_EQ(parked.timing.park_metadata.total_ns + parked.timing.park_stash.total_ns +
                parked.timing.park_publication.total_ns,
            parked.timing.park.total_ns);
  EXPECT_EQ(parked.timing.park_metadata.longest.serial, parked.timing.park.last.serial);
  EXPECT_EQ(parked.timing.park_metadata.longest.start_ns, parked.timing.park.last.start_ns);
  EXPECT_EQ(parked.timing.park_metadata.longest.end_ns, parked.timing.park_stash.longest.start_ns);
  EXPECT_EQ(parked.timing.park_stash.longest.end_ns,
            parked.timing.park_publication.longest.start_ns);
  EXPECT_EQ(parked.timing.park_publication.longest.end_ns, parked.timing.park.last.end_ns);
  // A metadata refusal publishes its result without completing either
  // milestone: retain the whole handler, but do not label its stages.
  counters.BeginPark();
  counters.EndPark();
  const auto incomplete = counters.Snapshot();
  EXPECT_EQ(incomplete.timing.park.completed, 2U);
  EXPECT_EQ(incomplete.timing.park_metadata.completed, 1U);
  EXPECT_EQ(incomplete.timing.park_stash.completed, 1U);
  EXPECT_EQ(incomplete.timing.park_publication.completed, 1U);
  EXPECT_GE(incomplete.timing.park.total_ns, parked.timing.park.total_ns);

  jitllm::scheduler::BackingCreateCounters ordinary;
  ordinary.Attempt(true);
  ordinary.Completed(true);
  ordinary.BeginPark();
  ordinary.ParkMetadataDone();
  ordinary.ParkStashDone();
  ordinary.EndPark();
  const auto unset = ordinary.Snapshot();
  EXPECT_FALSE(unset.timing.enabled);
  EXPECT_EQ(unset.reserve_attempts, 1U);
  EXPECT_EQ(unset.timing.create_serial, 0U);
  EXPECT_EQ(unset.timing.park.started, 0U);
  EXPECT_EQ(unset.timing.park_metadata.completed, 0U);
  EXPECT_EQ(unset.timing.park_stash.completed, 0U);
  EXPECT_EQ(unset.timing.park_publication.completed, 0U);
}

TEST(VmmLaneTest, TheHandleReserveFillsWhileIdleAndIsTakenBeforeCreating) {
  using jitllm::scheduler::BackingWork;
  using jitllm::scheduler::ReserveSettings;
  FakeDeviceMemory memory{Bytes(kSize), Bytes(kSize * 8)};
  const auto place = memory.Reserve(Bytes(kSize * 4)).value();
  jitllm::base::WakeFlag wake;
  CompletionBoard board{16, wake};
  BackingService lane(&memory, board, QueueSettings{.capacity = 8, .reserved = 1, .batch = 8},
                      ReserveSettings{.allocation_class = 0, .size = Bytes(kSize), .count = 2},
                      true);
  const auto run = [&](BackingWork::Kind kind, std::uint64_t at, bool retain, std::uint64_t size) {
    const auto operation = board.Open();
    EXPECT_EQ(lane.Submit(
                  jitllm::scheduler::DeviceCommand{.operation = operation,
                                                   .work = BackingWork{.kind = kind,
                                                                       .reservation = place,
                                                                       .offset = Bytes(at * kSize),
                                                                       .size = Bytes(size),
                                                                       .allocation_class = 0,
                                                                       .retain = retain,
                                                                       .reuse = false}}),
              PushResult::kAccepted);
    EXPECT_TRUE(lane.Turn());
    const auto seen = board.Harvest(4);
    const bool ok = seen.size() == 1 && seen[0].terminal.has_value() &&
                    seen[0].terminal->outcome == jitllm::scheduler::Outcome::kSucceeded;
    (void)board.Close(operation);
    return ok;
  };
  // Idle turns fill it, one create each, up to its count.
  EXPECT_TRUE(lane.Turn());
  EXPECT_TRUE(lane.Turn());
  EXPECT_FALSE(lane.Turn());
  EXPECT_EQ(lane.reserved(), 2U);
  EXPECT_EQ(memory.backings(), 2U);
  EXPECT_EQ(lane.create_stats().reserve_attempts, 2U);
  EXPECT_EQ(lane.create_stats().reserve_failures, 0U);
  EXPECT_TRUE(lane.create_stats().timing.enabled);
  EXPECT_EQ(lane.create_stats().timing.reserve.started, 2U);
  EXPECT_EQ(lane.create_stats().timing.reserve.completed, 2U);
  EXPECT_EQ(lane.create_stats().timing.current_create.serial, 0U);
  // A map takes one (the same turn's idle refill is not needed: the
  // command was the turn's progress).
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 0, false, kSize));
  EXPECT_EQ(lane.reserved(), 1U);
  EXPECT_EQ(memory.backings(), 2U);
  // A plain unmap's backing refills it.
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 0, false, kSize));
  EXPECT_EQ(lane.reserved(), 2U);
  EXPECT_EQ(memory.backings(), 2U);
  // Full: a map and unmap past it create and release as before.
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 0, false, kSize));
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 1, false, kSize));
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 2, false, kSize));
  EXPECT_EQ(lane.reserved(), 0U);
  EXPECT_EQ(memory.backings(), 3U);
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 2, false, kSize));
  EXPECT_TRUE(lane.Turn());  // idle: refills the second
  EXPECT_FALSE(lane.Turn());
  EXPECT_EQ(lane.reserved(), 2U);
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 1, false, kSize));
  EXPECT_EQ(lane.reserved(), 2U);
  EXPECT_EQ(memory.backings(), 3U);  // released: the reserve was full
  // A kept backing released refills a short one.
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 1, false, kSize));
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 1, true, kSize));
  ASSERT_TRUE(run(BackingWork::Kind::kRelease, 1, false, kSize));
  EXPECT_EQ(lane.reserved(), 2U);
  // Another size never touches it.
  ASSERT_TRUE(run(BackingWork::Kind::kMap, 2, false, 2 * kSize));
  EXPECT_EQ(lane.reserved(), 2U);
  EXPECT_EQ(memory.backings(), 4U);
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 2, false, 2 * kSize));
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap, 0, false, kSize));
  EXPECT_EQ(memory.backings(), 2U);
  lane.Close();
  lane.Run();
  EXPECT_TRUE(lane.ReleaseReserve());
  EXPECT_EQ(lane.reserved(), 0U);
  EXPECT_EQ(memory.backings(), 0U);
  ASSERT_TRUE(memory.Free(place).has_value());
}

// A refill the provider refuses is tried again only after the lane's next
// command; one whose outcome is unknown is never tried again, and the
// backing that may exist counts against the reserve: an unmap's backing
// is then released, not kept.
TEST(VmmLaneTest, AFailedRefillWaitsForACommandAndAnUnknownOneShortensTheReserve) {
  using jitllm::providers::fake::Operation;
  using jitllm::scheduler::BackingWork;
  using jitllm::scheduler::ReserveSettings;
  FakeDeviceMemory memory{Bytes(kSize), Bytes(kSize * 8)};
  const auto place = memory.Reserve(Bytes(kSize * 4)).value();
  jitllm::base::WakeFlag wake;
  CompletionBoard board{16, wake};
  BackingService lane(&memory, board, QueueSettings{.capacity = 8, .reserved = 1, .batch = 8},
                      ReserveSettings{.allocation_class = 0, .size = Bytes(kSize), .count = 2},
                      true);
  const auto run = [&](BackingWork::Kind kind) {
    const auto operation = board.Open();
    EXPECT_EQ(
        lane.Submit(jitllm::scheduler::DeviceCommand{.operation = operation,
                                                     .work = BackingWork{.kind = kind,
                                                                         .reservation = place,
                                                                         .offset = Bytes(0),
                                                                         .size = Bytes(kSize),
                                                                         .allocation_class = 0,
                                                                         .retain = false,
                                                                         .reuse = false}}),
        PushResult::kAccepted);
    EXPECT_TRUE(lane.Turn());
    const auto seen = board.Harvest(4);
    const bool ok = seen.size() == 1 && seen[0].terminal.has_value() &&
                    seen[0].terminal->outcome == jitllm::scheduler::Outcome::kSucceeded;
    (void)board.Close(operation);
    return ok;
  };
  memory.FailNext(Operation::kCreate, ProviderError::kOutOfMemory);
  EXPECT_FALSE(lane.Turn());  // refused: nothing kept
  EXPECT_FALSE(lane.Turn());  // and not tried again while idle
  EXPECT_EQ(lane.reserved(), 0U);
  const auto reserve_failure = lane.create_stats();
  EXPECT_EQ(reserve_failure.reserve_attempts, 1U);
  EXPECT_EQ(reserve_failure.reserve_failures, 1U);
  EXPECT_EQ(reserve_failure.timing.reserve.started, 1U);
  EXPECT_EQ(reserve_failure.timing.reserve.completed, 1U);
  EXPECT_NE(reserve_failure.timing.reserve.last.serial, 0U);
  EXPECT_EQ(reserve_failure.ordinary_attempts, 0U);
  EXPECT_TRUE(reserve_failure.last_failure_reserve);
  EXPECT_EQ(reserve_failure.last_failure_error, ProviderError::kOutOfMemory);
  EXPECT_GT(reserve_failure.last_failure_monotonic_ns, 0U);
  memory.FailNext(Operation::kCreate, ProviderError::kOutOfMemory);
  EXPECT_FALSE(run(BackingWork::Kind::kMap));
  const auto ordinary_failure = lane.create_stats();
  EXPECT_EQ(ordinary_failure.ordinary_attempts, 1U);
  EXPECT_EQ(ordinary_failure.ordinary_failures, 1U);
  EXPECT_FALSE(ordinary_failure.last_failure_reserve);
  EXPECT_GE(ordinary_failure.last_failure_monotonic_ns, reserve_failure.last_failure_monotonic_ns);
  // A map creates its own; the next idle turn tries again.
  ASSERT_TRUE(run(BackingWork::Kind::kMap));
  EXPECT_EQ(memory.backings(), 1U);
  EXPECT_TRUE(lane.Turn());
  EXPECT_EQ(lane.reserved(), 1U);
  memory.FailNext(Operation::kCreate, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_FALSE(lane.Turn());
  EXPECT_EQ(lane.reserved(), 1U);
  EXPECT_EQ(memory.backings(), 3U);  // mapped, kept, and one undetermined
  const auto unknown_failure = lane.create_stats();
  EXPECT_EQ(unknown_failure.reserve_failures, 2U);
  EXPECT_EQ(unknown_failure.ordinary_failures, 1U);
  EXPECT_EQ(unknown_failure.ordinary_attempts, 2U);
  EXPECT_TRUE(unknown_failure.last_failure_reserve);
  EXPECT_EQ(unknown_failure.last_failure_error, ProviderError::kUnknown);
  ASSERT_TRUE(run(BackingWork::Kind::kUnmap));
  EXPECT_FALSE(lane.Turn());
  EXPECT_EQ(lane.reserved(), 1U);
  EXPECT_EQ(memory.backings(), 2U);
  lane.Close();
  lane.Run();
  EXPECT_TRUE(lane.ReleaseReserve());
  EXPECT_EQ(memory.backings(), 1U);  // the undetermined one, never released
  ASSERT_TRUE(memory.Free(place).has_value());
}

// A submission lane polling its queue (DeviceSettings::poll_window) still
// takes every command and returns once closed, however long its window.
TEST(DeviceLanePollTest, APollingSubmissionLaneTakesEveryCommandAndStopsOnClose) {
  FakeDeviceExecution execution;
  const StreamId stream = execution.CreateStream().value();
  jitllm::base::WakeFlag wake;
  CompletionBoard board{8, wake};
  DeviceService lane(execution, std::span<const StreamId>(&stream, 1), board,
                     DeviceSettings{.queue = {.capacity = 8, .reserved = 1, .batch = 4},
                                    .handoff = 8,
                                    .poll_window = SchedulerSettings::kLongest});
  std::vector<std::byte> source(kSize, std::byte{7});
  std::vector<std::byte> destination(kSize);
  std::vector<jitllm::scheduler::OperationId> operations;
  {
    std::jthread submission([&] { lane.RunSubmission(); });
    std::jthread completion([&] { lane.RunCompletion(); });
    std::jthread device([&](const std::stop_token& stop) {
      while (!stop.stop_requested()) {
        execution.Drain();
        std::this_thread::yield();
      }
    });
    for (int i = 0; i < 4; ++i) {
      jitllm::scheduler::DeviceWork work;
      work.copies.at(0) = {.destination = reinterpret_cast<std::uint64_t>(destination.data()),
                           .source = reinterpret_cast<std::uint64_t>(source.data()),
                           .size = Bytes(kSize)};
      work.count = 1;
      operations.push_back(board.Open());
      ASSERT_EQ(lane.Submit(
                    jitllm::scheduler::DeviceCommand{.operation = operations.back(), .work = work}),
                PushResult::kAccepted);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));  // polling, not asleep
    }
    std::size_t completed = 0;
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (completed < operations.size() && std::chrono::steady_clock::now() < give_up) {
      for (const auto& seen : board.Harvest(8)) {
        completed += seen.terminal &&
                             seen.terminal->outcome == jitllm::scheduler::Outcome::kSucceeded &&
                             seen.terminal->no_further_access
                         ? 1
                         : 0;
      }
      (void)wake.WaitFor(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(completed, operations.size());
    lane.Close();  // the lanes return, although the window has not passed
    submission.join();
    completion.join();
  }
  EXPECT_EQ(destination, source);
}

// A storage lane polling (its poll window) takes every read and returns
// once closed, however long its window.
TEST(StorageLanePollTest, APollingStorageLaneTakesEveryReadAndStopsOnClose) {
  FakeStorage storage{4, 4096};
  std::vector<std::byte> contents(4 * kSize);
  for (std::size_t i = 0; i < contents.size(); ++i) {
    contents[i] = static_cast<std::byte>(i * 13);
  }
  const int fd = storage.AddFile(contents);
  jitllm::base::WakeFlag wake;
  CompletionBoard board{8, wake};
  StorageService lane(storage,
                      ReaderSettings{.alignment = 4096,
                                     .request_bytes = 16 * 1024,
                                     .retries = 0,
                                     .reads = 8,
                                     .waiters = 2,
                                     .span_bytes = jitllm::providers::kNoCoalescing,
                                     .span_segments = jitllm::providers::kMaxSegments},
                      board, QueueSettings{.capacity = 8, .reserved = 1, .batch = 4},
                      SchedulerSettings::kLongest);
  auto* memory = static_cast<std::byte*>(
      std::aligned_alloc(4096, contents.size()));  // NOLINT(cppcoreguidelines-no-malloc)
  {
    std::jthread thread([&] { lane.Run(); });
    for (std::size_t i = 0; i < 4; ++i) {
      ASSERT_EQ(lane.Submit(jitllm::scheduler::ReadCommand{.operation = board.Open(),
                                                           .spec = {.fd = fd,
                                                                    .offset = i * kSize,
                                                                    .memory = memory + (i * kSize),
                                                                    .length = kSize}}),
                PushResult::kAccepted);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));  // polling, not asleep
    }
    std::size_t completed = 0;
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (completed < 4 && std::chrono::steady_clock::now() < give_up) {
      for (const auto& seen : board.Harvest(8)) {
        completed += seen.terminal &&
                             seen.terminal->outcome == jitllm::scheduler::Outcome::kSucceeded &&
                             seen.terminal->bytes == kSize
                         ? 1
                         : 0;
      }
      (void)wake.WaitFor(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(completed, 4U);
    lane.Close();  // it returns, although the window has not passed
  }
  EXPECT_EQ(std::memcmp(memory, contents.data(), contents.size()), 0);
  std::free(memory);  // NOLINT(cppcoreguidelines-no-malloc)
}

TEST(StorageLanePollDeathTest, AnUnboundedPollWindowIsRefused) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeStorage storage{4, 4096};
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  const auto build = [&](std::chrono::microseconds window) {
    const StorageService lane(storage, ReaderSettings{}, board, QueueSettings{}, window);
  };
  EXPECT_DEATH(build(std::chrono::microseconds(-1)), "storage poll window");
  EXPECT_DEATH(build(std::chrono::hours(2)), "storage poll window");
  build(std::chrono::microseconds(0));
}

TEST(DeviceLanePollDeathTest, AnUnboundedPollWindowIsRefused) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeDeviceExecution execution;
  const StreamId stream = execution.CreateStream().value();
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  const auto build = [&](std::chrono::microseconds window) {
    DeviceSettings settings;
    settings.poll_window = window;
    const DeviceService lane(execution, std::span<const StreamId>(&stream, 1), board, settings);
  };
  EXPECT_DEATH(build(std::chrono::microseconds(-1)), "poll and spin windows");
  EXPECT_DEATH(build(std::chrono::hours(2)), "poll and spin windows");
  build(std::chrono::microseconds(0));
  // The completion lane's spin around a fence's likely end, and its
  // backstop, which must be positive (a zero backstop would never sleep).
  const auto spin = [&](std::chrono::microseconds ahead, std::chrono::microseconds past,
                        std::chrono::microseconds backstop) {
    DeviceSettings settings;
    settings.spin_ahead = ahead;
    settings.spin_past = past;
    settings.backstop = backstop;
    const DeviceService lane(execution, std::span<const StreamId>(&stream, 1), board, settings);
  };
  const std::chrono::microseconds ms(1000);
  EXPECT_DEATH(spin(std::chrono::microseconds(-1), ms, ms), "poll and spin windows");
  EXPECT_DEATH(spin(ms, std::chrono::hours(2), ms), "poll and spin windows");
  EXPECT_DEATH(spin(ms, ms, std::chrono::microseconds(0)), "backstop positive");
  spin(std::chrono::microseconds(0), std::chrono::microseconds(0), std::chrono::microseconds(1));
}

TEST(PageInZoneTest, ALandedSourceNeedsAZone) {
  FakeDeviceMemory memory{Bytes(kSize), Bytes(kSize * 4)};
  FakeDeviceExecution execution;
  FakeStorage storage{4, 4096};
  jitllm::catalog::Catalog catalog;
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  const auto domain = catalog.AddDomain("node");
  const ExtentId extent = catalog
                              .AddExtent({.domain = domain,
                                          .memory_class = jitllm::catalog::MemoryClass::kWeights,
                                          .recovery = jitllm::catalog::Recovery::kFromArtifact,
                                          .size = Bytes(kSize),
                                          .content = {}})
                              .value();
  Scheduler scheduler(catalog, board, wake, Lanes{}, SchedulerSettings{});
  const PageSource landed{
      .read = ReadSpec{.fd = 3, .offset = 0, .memory = nullptr, .length = kSize},
      .landed = true,
      .destination = 4096,
      .backing = std::nullopt};
  EXPECT_EQ(Failed(scheduler.SetSource(extent, landed)), WorkError::kInvalid);
}

// The zone's copies run on one of the device lane's streams: a zone with
// no device lane, or a stream index the lane does not have, would fail
// every landed load. Refused when the scheduler is built.
TEST(PageInZoneDeathTest, TheLandingStreamMustBeADeviceLaneStream) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  FakeDeviceExecution execution;
  const StreamId stream = execution.CreateStream().value();
  jitllm::catalog::Catalog catalog;
  jitllm::base::WakeFlag wake;
  CompletionBoard board{4, wake};
  DeviceService device(execution, std::span<const StreamId>(&stream, 1), board, DeviceSettings{});
  const auto build = [&](DeviceService* lane, std::uint32_t index) {
    const Scheduler scheduler(
        catalog, board, wake, Lanes{.storage = nullptr, .device = lane, .cpu = nullptr},
        SchedulerSettings{
            .landing = LandingZone{.slots = {4096}, .slot_bytes = Bytes(kSize), .stream = index}});
  };
  EXPECT_DEATH(build(&device, 1), "landing zone's stream");
  EXPECT_DEATH(build(nullptr, 0), "landing zone's stream");
  build(&device, 0);
}

// Every lane on its own thread and the fake device on another: requests
// load every extent through a one-slot zone, evict them, and load them
// again, some cancelled, while the owner sleeps on its wake flag. A lost
// wakeup hangs; a slot reused early or a publication before the copy's
// fence shows as a mismatch, and a race under ThreadSanitizer. How many
// evictions the overlapping requests manage depends on how the threads
// are scheduled (one holding an extent makes another skip it), so the
// count is asserted only for a last, serial phase: one request at a time,
// each of which must evict every extent, and the next load them all again.
TEST_P(PageInTest, ThreadedLoadsAndEvictionsThroughTheZoneKeepEveryByte) {
  Build(1, 64);
  std::atomic<int> mismatches{0};
  std::atomic<int> done{0};
  std::atomic<int> evictions{0};
  std::atomic<int> serial_evictions{0};
  constexpr int kTries = 1000;  // yields while other requests hold an extent
  constexpr int kRounds = 40;
  constexpr int kSerial = 5;
  // Loads everything, checks the bytes while holding a job's lease, and
  // evicts what it can.
  class Cycle final : public TaskProgram {
   public:
    Cycle(Closure all, std::vector<ExtentId> extents, std::function<bool()> check,
          std::atomic<int>& mismatches, std::atomic<int>& evictions, std::atomic<int>& done)
        : all_(std::move(all)),
          extents_(std::move(extents)),
          check_(std::move(check)),
          mismatches_(mismatches),
          evictions_(evictions),
          done_(done) {}
    Step Advance(TaskContext& context) override {
      if (context.TakeFailure()) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (!submitted_) {
        const auto ready = context.Materialize(all_);
        if (!ready) {
          return ready.error() == WorkError::kBusy ? Step::Yield()
                                                   : Step::Finish(TaskOutcome::kFailed);
        }
        if (*ready == Readiness::kWaiting) {
          return Step::Wait();
        }
        std::function<bool()> check = check_;
        std::atomic<int>& mismatches = mismatches_;
        const auto job = context.SubmitLaunch(
            all_, LaunchWork{.stream = 0, .job = [check, &mismatches](auto /*stream*/) {
                               if (!check()) {
                                 mismatches.fetch_add(1);
                               }
                               return JobResult::kQueued;
                             }});
        if (!job) {
          // Evicted meanwhile, or no mailbox: materialize again.
          return job.error() == WorkError::kBusy || job.error() == WorkError::kNotResident
                     ? Step::Yield()
                     : Step::Finish(TaskOutcome::kFailed);
        }
        submitted_ = true;
        return Step::Wait();
      }
      if (next_ < extents_.size()) {
        const auto evicted = context.Evict(extents_[next_]);
        if (evicted || evicted.error() != WorkError::kBusy || ++tries_ > kTries) {
          ++next_;  // evicted, or held by other requests too long: it stays
          tries_ = 0;
        }
        if (evicted && *evicted == Readiness::kWaiting) {
          evictions_.fetch_add(1);
          return Step::Wait();
        }
        return Step::Yield();
      }
      return Step::Finish(TaskOutcome::kSucceeded);
    }
    void Retired() override { done_.fetch_add(1); }

   private:
    Closure all_;
    std::vector<ExtentId> extents_;
    std::function<bool()> check_;
    std::atomic<int>& mismatches_;
    std::atomic<int>& evictions_;
    std::atomic<int>& done_;
    bool submitted_ = false;
    std::size_t next_ = 0;
    int tries_ = 0;
  };
  const std::function<bool()> check = [this] {
    for (std::size_t i = 0; i < kExtents; ++i) {
      if (!Loaded(i, weights_)) {
        return false;
      }
    }
    return true;
  };
  std::optional<std::expected<void, Fault>> result;
  {
    std::jthread owner([&] { result = scheduler_->Run(); });
    std::jthread storage([&] { storage_lane_->Run(); });
    std::jthread submission([&] { device_lane_->RunSubmission(); });
    std::jthread completion([&] { device_lane_->RunCompletion(); });
    std::jthread vmm([&] {
      if (backing_lane_ != nullptr) {
        backing_lane_->Run();
      }
    });
    std::jthread device([&](const std::stop_token& stop) {
      while (!stop.stop_requested()) {
        execution_.Drain();
        std::this_thread::yield();
      }
    });
    for (int n = 0; n < kRounds; ++n) {
      while (n - done.load() >= 4) {
        std::this_thread::yield();
      }
      Control start = StartRequest{
          .request = static_cast<std::uint64_t>(n + 1),
          .priority = 1,
          .program = std::make_unique<Cycle>(All(), extents_, check, mismatches, evictions, done)};
      // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
      while (scheduler_->Post(std::move(start)) == PushResult::kFull) {
        std::this_thread::yield();
      }
      if (n % 5 == 4) {
        Control cancel = CancelRequest{.request = static_cast<std::uint64_t>(n)};
        (void)scheduler_->Post(std::move(cancel));
      }
    }
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (done.load() < kRounds && std::chrono::steady_clock::now() < give_up) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(done.load(), kRounds);
    // The serial phase: nothing else holds an extent, so each request
    // evicts all of them (none is ever skipped), whatever the timing.
    for (int n = 0; n < kSerial; ++n) {
      Control start = StartRequest{.request = static_cast<std::uint64_t>(kRounds + n + 1),
                                   .priority = 1,
                                   .program = std::make_unique<Cycle>(
                                       All(), extents_, check, mismatches, serial_evictions, done)};
      // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
      while (scheduler_->Post(std::move(start)) == PushResult::kFull) {
        std::this_thread::yield();
      }
      const auto serial_give_up = std::chrono::steady_clock::now() + kPatience;
      while (done.load() < kRounds + n + 1 && std::chrono::steady_clock::now() < serial_give_up) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (done.load() != kRounds + n + 1) {
        ADD_FAILURE() << "serial round " << n << " did not finish";
        break;  // not ASSERT: the threads must still be stopped below
      }
    }
    scheduler_->RequestShutdown();
    owner.join();
    storage_lane_->Close();
    device_lane_->Close();
    if (backing_lane_ != nullptr) {
      backing_lane_->Close();
    }
    storage.join();
    submission.join();
    completion.join();
    vmm.join();
  }
  EXPECT_FALSE(result.has_value() && !result->has_value());
  EXPECT_EQ(mismatches.load(), 0);
  // Extents were evicted and read again: every one in each serial round,
  // and each serial round after the first loaded all of them (4 reads
  // each), whatever the overlapping phase managed.
  EXPECT_EQ(serial_evictions.load(), static_cast<int>(kSerial * kExtents));
  EXPECT_GE(storage_.submitted().size(), (kSerial - 1) * kExtents * 4);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);
  EXPECT_EQ(scheduler_->slots_quarantined(), 0U);
  EXPECT_EQ(scheduler_->loads(), 0U);
  const Occupancy occupied = Occupied();
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.evicting, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(memory_.backings(), baseline_ + (occupied.idle.value() / kSize));
  scheduler_.reset();  // stopped: TearDown has nothing more to drain
}

// The handoff (D-033; scheduler.h): evicts `out` with its backing kept,
// all at once, then materializes `in`. With `hold`, it does not finish
// while the flag is set, so what no load took stays parked.
class HandoffProgram final : public TaskProgram {
 public:
  struct Report {
    std::optional<TaskOutcome> outcome;
    bool retired = false;
    std::optional<WorkError> error;
    bool loaded = false;
  };
  HandoffProgram(Report& report, std::vector<ExtentId> out, Closure in,
                 const std::atomic<bool>* hold = nullptr, std::optional<Closure> then = {})
      : report_(report),
        out_(std::move(out)),
        in_(std::move(in)),
        hold_(hold),
        then_(std::move(then)) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (!evicted_) {
      evicted_ = true;
      bool waiting = false;
      for (const ExtentId extent : out_) {
        const auto evicted = context.Evict(extent, {.handoff = true});
        if (!evicted) {
          report_.error = evicted.error();
          return Step::Finish(TaskOutcome::kFailed);
        }
        waiting = waiting || *evicted == Readiness::kWaiting;
      }
      if (waiting) {
        return Step::Wait();
      }
    }
    if (!report_.loaded) {
      const auto ready = context.Materialize(in_);
      if (!ready) {
        report_.error = ready.error();
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*ready == Readiness::kWaiting) {
        return Step::Wait();
      }
      report_.loaded = true;
    }
    if (then_) {
      // A second closure, materialized once the first is resident.
      const auto ready = context.Materialize(*then_);
      if (!ready) {
        report_.error = ready.error();
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*ready == Readiness::kWaiting) {
        return Step::Wait();
      }
      then_.reset();
    }
    if (hold_ != nullptr && hold_->load()) {
      return Step::Yield();
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override { report_.retired = true; }

 private:
  Report& report_;
  std::vector<ExtentId> out_;
  Closure in_;
  const std::atomic<bool>* hold_;
  std::optional<Closure> then_;
  bool evicted_ = false;
};

// Three resident extents are evicted with their backing kept, and three
// others load into it: no backing is created or released, the catalog
// counts the kept backing throughout and never exceeds B (exactly the
// three extents' bytes), and every byte loaded is the file's.
TEST_P(PageInTest, AHandoffMovesEvictedBackingToTheLoadsThatFollow) { HandoffMoves(); }
TEST_P(PageInTest, ALazyHandoffMovesEvictedBackingToTheLoadsThatFollow) {
  lazy_ = true;
  HandoffMoves();
  // Each load unmapped the place its backing was parked at.
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FALSE(memory_.MappedAt(weights_, Bytes(i * kSize)).has_value()) << i;
    EXPECT_TRUE(memory_.MappedAt(weights_, Bytes((i + 3) * kSize)).has_value()) << i + 3;
  }
}
void PageInTest::HandoffMoves() {
  Build(kSlots, 32, kSize * 3);
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0, 1, 2}))).has_value());
  Settle();
  ASSERT_EQ(first.outcome, TaskOutcome::kSucceeded);
  ASSERT_EQ(memory_.backings(), baseline_ + 3);

  HandoffProgram::Report report;
  ASSERT_TRUE(
      scheduler_
          ->Start(2, std::make_unique<HandoffProgram>(
                         report, std::vector<ExtentId>{extents_[0], extents_[1], extents_[2]},
                         Of({3, 4, 5})))
          .has_value());
  std::size_t most_backings = 0;
  Bytes most_occupied;
  for (int i = 0; i < 1000 && Round(); ++i) {
    most_backings = std::max(most_backings, memory_.backings());
    most_occupied = std::max(most_occupied, Occupied().Total());
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(most_backings, baseline_ + 3);  // never a fourth: nothing created
  EXPECT_LE(most_occupied, Bytes(kSize * 3));
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(View(extents_[i]).state, ExtentState::kNonresident) << i;
    EXPECT_EQ(View(extents_[i + 3]).state, ExtentState::kResident) << i + 3;
    EXPECT_TRUE(Loaded(i + 3, weights_)) << i + 3;
  }
  EXPECT_EQ(memory_.backings(), baseline_ + 3);
  EXPECT_EQ(scheduler_->stats().parked, 3U);
  EXPECT_EQ(scheduler_->stats().handed_off, 3U);
  EXPECT_EQ(scheduler_->stats().released_unused, 0U);
  EXPECT_EQ(scheduler_->evictions(), 0U);
  EXPECT_EQ(scheduler_->parked(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes(kSize * 3));
}

// Lazily parked backing no load took is unmapped and released when its
// evictor finishes, leaving no mapping at the outgoing places.
TEST_P(PageInTest, ALazyHandoffReleasesWhatNoLoadTookUnmapped) {
  lazy_ = true;
  Build(kSlots, 32, kSize * 3);
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0, 1, 2}))).has_value());
  Settle();
  ASSERT_EQ(first.outcome, TaskOutcome::kSucceeded);
  HandoffProgram::Report report;
  ASSERT_TRUE(
      scheduler_
          ->Start(
              2, std::make_unique<HandoffProgram>(
                     report, std::vector<ExtentId>{extents_[0], extents_[1], extents_[2]}, Of({3})))
          .has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(3, weights_));
  EXPECT_EQ(scheduler_->stats().handed_off, 1U);
  EXPECT_EQ(scheduler_->stats().released_unused, 2U);
  EXPECT_EQ(scheduler_->parked(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(View(extents_[i]).state, ExtentState::kNonresident) << i;
    EXPECT_FALSE(memory_.MappedAt(weights_, Bytes(i * kSize)).has_value()) << i;
  }
  EXPECT_EQ(Occupied().Total(), Bytes(kSize));
}

// A lazily parked extent materialized again takes its own backing back,
// still mapped at its place; and an extent whose parked eviction ended
// while another's backing went to a load reloads at its place, which its
// own kept backing still holds. Contents are always the file's.
TEST_P(PageInTest, ALazyHandoffReturnsBackingStillMappedAtItsPlace) {
  lazy_ = true;
  Build(kSlots, 32, kSize * 3);
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0, 1, 2}))).has_value());
  Settle();
  ASSERT_EQ(first.outcome, TaskOutcome::kSucceeded);
  // Out 0, 1, 2 (parked mapped). 3 takes the oldest parked eviction's
  // charge (0's), while the lane hands it the newest kept backing (2's);
  // then 0 reloads (taking 1's charge) at its place, which 0's own kept
  // backing still holds, and 1 reloads as its own parked extent.
  HandoffProgram::Report report;
  ASSERT_TRUE(
      scheduler_
          ->Start(2, std::make_unique<HandoffProgram>(
                         report, std::vector<ExtentId>{extents_[0], extents_[1], extents_[2]},
                         Of({3}), nullptr, Of({0})))
          .has_value());
  std::size_t most_backings = 0;
  for (int i = 0; i < 2000 && Round(); ++i) {
    most_backings = std::max(most_backings, memory_.backings());
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_FALSE(report.error.has_value());
  EXPECT_EQ(most_backings, baseline_ + 3);  // nothing created
  EXPECT_TRUE(Loaded(3, weights_));
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_EQ(View(extents_[3]).state, ExtentState::kResident);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[2]).state, ExtentState::kNonresident);
  EXPECT_FALSE(memory_.MappedAt(weights_, Bytes(kSize)).has_value());
  EXPECT_FALSE(memory_.MappedAt(weights_, Bytes(2 * kSize)).has_value());
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
  EXPECT_EQ(scheduler_->parked(), 0U);
  EXPECT_EQ(scheduler_->evictions(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes(kSize * 2));

  // A parked extent's own load, in the same task as its eviction.
  HandoffProgram::Report own;
  ASSERT_TRUE(scheduler_
                  ->Start(3, std::make_unique<HandoffProgram>(
                                 own, std::vector<ExtentId>{extents_[0]}, Of({0})))
                  .has_value());
  Settle();
  EXPECT_EQ(own.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(memory_.backings(), baseline_ + 2);
}

// B is each domain's: backing parked in one domain is never handed to a
// load in another, whose occupancy the move would take over its B.
TEST_P(PageInTest, AHandoffNeverMovesAChargeBetweenDomains) {
  Build(kSlots, 32, kSize);  // B: one extent per domain
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(first.outcome, TaskOutcome::kSucceeded);
  // A second domain, full: Y resident; X would need room it does not have.
  const auto other = catalog_.AddDomain("other");
  std::vector<ExtentId> there(2);
  for (std::size_t i = 0; i < there.size(); ++i) {
    there.at(i) =
        catalog_
            .AddExtent(
                {.domain = other,
                 .memory_class = jitllm::catalog::MemoryClass::kWeights,
                 .recovery = jitllm::catalog::Recovery::kFromArtifact,
                 .size = Bytes(kSize),
                 .content = {.artifact = {}, .group = 1, .chunk = static_cast<std::uint32_t>(i)}})
            .value();
    ASSERT_TRUE(scheduler_->SetSource(there.at(i), Source(i + 1, moved_)).has_value());
  }
  LoadProgram::Report full;
  ASSERT_TRUE(
      scheduler_
          ->Start(2, Load(full, catalog_.ClosureOfExtents(std::vector<ExtentId>{there[0]}).value()))
          .has_value());
  Settle();
  ASSERT_EQ(full.outcome, TaskOutcome::kSucceeded);

  HandoffProgram::Report report;
  ASSERT_TRUE(
      scheduler_
          ->Start(3, std::make_unique<HandoffProgram>(
                         report, std::vector<ExtentId>{extents_[0]},
                         catalog_.ClosureOfExtents(std::vector<ExtentId>{there[1]}).value()))
          .has_value());
  Bytes most;
  for (int i = 0; i < 1000 && Round(); ++i) {
    most = std::max(most, catalog_.OccupancyOf(other).Total());
  }
  EXPECT_EQ(report.error, WorkError::kOverBudget);
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_LE(most, Bytes(kSize));
  EXPECT_EQ(View(there[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(scheduler_->stats().handed_off, 0U);
  // The parked backing was released when its evictor finished.
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(scheduler_->evictions(), 0U);
  EXPECT_EQ(memory_.backings(), baseline_ + 1);  // Y's alone

  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(4, Evicting(evicted, {there[0]})).has_value());
  Settle();
  EXPECT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
}

// Kept backing no load takes stays charged (EVICTING) while its evictor
// runs, and is released when it finishes: no idle pool (D-033).
TEST_P(PageInTest, ParkedBackingNoLoadTookIsReleasedWhenItsEvictorFinishes) {
  Build();
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0, 1, 2}))).has_value());
  Settle();
  std::atomic<bool> hold{true};
  HandoffProgram::Report report;
  ASSERT_TRUE(
      scheduler_
          ->Start(2, std::make_unique<HandoffProgram>(
                         report, std::vector<ExtentId>{extents_[0], extents_[1], extents_[2]},
                         Of({3}), &hold))
          .has_value());
  for (int i = 0; i < 200; ++i) {
    (void)Round();
  }
  ASSERT_TRUE(report.loaded);
  EXPECT_FALSE(report.outcome.has_value());
  EXPECT_EQ(scheduler_->parked(), 2U);
  EXPECT_EQ(Occupied().evicting, Bytes(kSize * 2));  // kept, and counted
  EXPECT_EQ(memory_.backings(), baseline_ + 3);
  EXPECT_TRUE(Loaded(3, weights_));

  hold = false;
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(report.retired);
  EXPECT_EQ(scheduler_->evictions(), 0U);
  EXPECT_EQ(scheduler_->stats().released_unused, 2U);
  EXPECT_EQ(scheduler_->stats().handed_off, 1U);
  EXPECT_EQ(Occupied().evicting, Bytes());
  EXPECT_EQ(memory_.backings(), baseline_ + 1);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(View(extents_[i]).state, ExtentState::kNonresident) << i;
  }
}

// A parked extent materialized again takes its own kept backing back.
TEST_P(PageInTest, AParkedExtentTakesItsOwnBackingBack) {
  Build(kSlots, 32, kSize);
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0}))).has_value());
  Settle();
  HandoffProgram::Report report;
  ASSERT_TRUE(scheduler_
                  ->Start(2, std::make_unique<HandoffProgram>(
                                 report, std::vector<ExtentId>{extents_[0]}, Of({0})))
                  .has_value());
  std::size_t most = 0;
  for (int i = 0; i < 1000 && Round(); ++i) {
    most = std::max(most, memory_.backings());
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(most, baseline_ + 1);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kResident);
  EXPECT_TRUE(Loaded(0, weights_));
  EXPECT_EQ(scheduler_->stats().handed_off, 1U);
  EXPECT_EQ(scheduler_->evictions(), 0U);
}

// Live state written back through the zone and parked, then restored into
// its own kept backing: every byte comes back.
TEST_P(PageInTest, HandedOffStateIsWrittenBackAndRestoredWhole) {
  Build();
  const ExtentId state = AddState(0);
  const std::size_t backings = memory_.backings();
  HandoffProgram::Report report;
  ASSERT_TRUE(scheduler_
                  ->Start(1, std::make_unique<HandoffProgram>(
                                 report, std::vector<ExtentId>{state},
                                 catalog_.ClosureOfExtents(std::vector<ExtentId>{state}).value()))
                  .has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(Writes().size(), 4U);  // written back, in pieces of 16 KiB
  EXPECT_EQ(View(state).state, ExtentState::kResident);
  EXPECT_TRUE(StateIs(0));
  EXPECT_EQ(memory_.backings(), backings);
  EXPECT_EQ(scheduler_->stats().handed_off, 1U);
}

// A load that took kept backing and is cancelled before mapping it
// releases it: nothing kept, nothing charged, nothing left.
TEST_P(PageInTest, CancellingALoadThatTookKeptBackingReleasesIt) {
  Build(1);  // one slot: two landed loads in flight, the third waits unstarted
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0, 1, 2}))).has_value());
  Settle();
  HoldNext(64);
  HandoffProgram::Report report;
  ASSERT_TRUE(
      scheduler_
          ->Start(2, std::make_unique<HandoffProgram>(
                         report, std::vector<ExtentId>{extents_[0], extents_[1], extents_[2]},
                         Of({3, 4, 5})))
          .has_value());
  Settle();
  EXPECT_EQ(scheduler_->loads(), 3U);
  EXPECT_EQ(memory_.backings(), baseline_ + 3);  // taken, none created
  ASSERT_TRUE(scheduler_->Cancel(2));
  for (int round = 0; round < 40 && scheduler_->loads() > 0; ++round) {
    ReleaseReads();
    Settle();
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(scheduler_->loads(), 0U);
  EXPECT_EQ(scheduler_->evictions(), 0U);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_FALSE(scheduler_->fault().has_value());
  EXPECT_GE(scheduler_->stats().released_unused, 1U);
}

// A kept backing whose map is refused goes back, and the load that took
// it releases it as it unwinds.
TEST_P(PageInTest, AKeptBackingWhoseMapIsRefusedIsReleased) {
  Build();
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0}))).has_value());
  Settle();
  memory_.FailNext(jitllm::providers::fake::Operation::kMap, ProviderError::kFailed);
  HandoffProgram::Report report;
  ASSERT_TRUE(scheduler_
                  ->Start(2, std::make_unique<HandoffProgram>(
                                 report, std::vector<ExtentId>{extents_[0]}, Of({1})))
                  .has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kFailed);
  EXPECT_EQ(View(extents_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(extents_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(memory_.backings(), baseline_);
  EXPECT_EQ(Occupied().Total(), Bytes());
  EXPECT_FALSE(scheduler_->fault().has_value());
  EXPECT_EQ(scheduler_->stats().released_unused, 1U);
}

// D-090: a captured graph replays after a swap only if every extent it
// reads is back at the address it was captured at. A full swap A→B→A with
// the handoff maps whatever backing A's loads take at A's own places again,
// so every address a graph captured before the swap names A's bytes after
// it; a pinned place cannot be moved, resident or not; and once unpinned, a
// move is seen by comparing the extent's place with the captured one (the
// check that fails if an address changed).
TEST_P(PageInTest, AfterASwapEveryPinnedExtentIsBackAtItsAddress) {
  Build(kSlots, 32, kSize * 3);  // room for three extents: each swap evicts first
  const std::vector<ExtentId> a = {extents_[0], extents_[1], extents_[2]};
  const std::vector<ExtentId> b = {extents_[3], extents_[4], extents_[5]};
  LoadProgram::Report first;
  ASSERT_TRUE(scheduler_->Start(1, Load(first, Of({0, 1, 2}))).has_value());
  Settle();
  ASSERT_EQ(first.outcome, TaskOutcome::kSucceeded);
  ASSERT_TRUE(scheduler_->PinPlaces(a).has_value());
  // What a graph captured now holds: each extent's place.
  std::vector<PageSource> captured;
  for (const ExtentId extent : a) {
    ASSERT_NE(scheduler_->SourceOf(extent), nullptr);
    captured.push_back(*scheduler_->SourceOf(extent));
  }
  const auto mapped = [&](ReservationId reservation, std::size_t i) {
    return memory_.MappedAt(reservation, Bytes(i * kSize)).has_value();
  };
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_TRUE(mapped(weights_, i)) << i;
  }

  HandoffProgram::Report out;
  ASSERT_TRUE(
      scheduler_->Start(2, std::make_unique<HandoffProgram>(out, a, Of({3, 4, 5}))).has_value());
  Settle();
  ASSERT_EQ(out.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_FALSE(mapped(weights_, i)) << i;  // A's places are empty while B runs
  }
  HandoffProgram::Report back;
  ASSERT_TRUE(
      scheduler_->Start(3, std::make_unique<HandoffProgram>(back, b, Of({0, 1, 2}))).has_value());
  Settle();
  ASSERT_EQ(back.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(scheduler_->stats().handed_off, 6U);  // A's backing to B, and B's to A
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(View(a[i]).state, ExtentState::kResident) << i;
    EXPECT_TRUE(mapped(weights_, i)) << i;  // at its own place ...
    EXPECT_FALSE(mapped(moved_, i)) << i;   // ... and nowhere else
    EXPECT_TRUE(jitllm::scheduler::SamePlace(*scheduler_->SourceOf(a[i]), captured[i])) << i;
    EXPECT_TRUE(Loaded(i, weights_)) << i;  // its bytes at the captured address
  }

  // A pinned place cannot move, resident or not; another file range for
  // the same place is no move.
  EXPECT_EQ(Failed(scheduler_->SetSource(a[0], Source(0, moved_))), WorkError::kBusy);
  // Resident, the same backing place with its contents copied elsewhere in
  // it: only the pin refuses that (a resident extent's backing may not
  // move anyway).
  PageSource shifted = captured[1];
  shifted.destination += 256;
  EXPECT_EQ(Failed(scheduler_->SetSource(a[1], shifted)), WorkError::kBusy);
  EvictProgram::Report evicted;
  ASSERT_TRUE(scheduler_->Start(4, Evicting(evicted, a)).has_value());
  Settle();
  ASSERT_EQ(evicted.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(Failed(scheduler_->SetSource(a[0], Source(0, moved_))), WorkError::kBusy);
  PageSource reread = captured[0];
  reread.read.offset = kSize;  // another range of the file, landed at the same place
  EXPECT_TRUE(scheduler_->SetSource(a[0], reread).has_value());
  EXPECT_TRUE(scheduler_->SetSource(a[0], captured[0]).has_value());

  // Pins count, and need a source: an extent without one pins nothing.
  const ExtentId orphan = catalog_
                              .AddExtent({.domain = domain_,
                                          .memory_class = jitllm::catalog::MemoryClass::kWeights,
                                          .recovery = jitllm::catalog::Recovery::kFromArtifact,
                                          .size = Bytes(kSize),
                                          .content = {.artifact = {}, .group = 9, .chunk = 0}})
                              .value();
  EXPECT_EQ(Failed(scheduler_->PinPlaces(std::vector<ExtentId>{a[1], orphan})),
            WorkError::kUnavailable);
  ASSERT_TRUE(scheduler_->PinPlaces(std::vector<ExtentId>{a[0]}).has_value());
  scheduler_->UnpinPlaces(a);
  EXPECT_TRUE(scheduler_->PlacePinned(a[0]));  // pinned twice, unpinned once
  EXPECT_FALSE(scheduler_->PlacePinned(a[1]));
  EXPECT_EQ(Failed(scheduler_->SetSource(a[0], Source(0, moved_))), WorkError::kBusy);
  scheduler_->UnpinPlaces(std::vector<ExtentId>{a[0]});
  EXPECT_FALSE(scheduler_->PlacePinned(a[0]));

  // Unpinned, it moves: the next load maps it at the new place, and its
  // place no longer matches the captured one.
  ASSERT_TRUE(scheduler_->SetSource(a[0], Source(0, moved_)).has_value());
  LoadProgram::Report moved;
  ASSERT_TRUE(scheduler_->Start(5, Load(moved, Of({0}))).has_value());
  Settle();
  ASSERT_EQ(moved.outcome, TaskOutcome::kSucceeded);
  EXPECT_FALSE(jitllm::scheduler::SamePlace(*scheduler_->SourceOf(a[0]), captured[0]));
  EXPECT_TRUE(mapped(moved_, 0));
  EXPECT_FALSE(mapped(weights_, 0));
  EXPECT_TRUE(Loaded(0, moved_));
}

// What counts as the same place: the backing's place and where the
// contents land, not the file range they come from.
TEST(PlaceTest, SamePlaceComparesWhereContentsGoNotWhereTheyComeFrom) {
  using jitllm::scheduler::LandedPiece;
  using jitllm::scheduler::SamePlace;
  const PageSource landed{
      .read = ReadSpec{.fd = 3, .offset = 0, .memory = nullptr, .length = kSize},
      .landed = true,
      .destination = 0x10000,
      .backing = BackingPlace{
          .reservation = {}, .offset = Bytes(0), .size = Bytes(kSize), .allocation_class = 0}};
  PageSource other = landed;
  other.read.fd = 4;
  other.read.offset = kSize;
  EXPECT_TRUE(SamePlace(landed, other));
  other = landed;
  other.destination += 256;
  EXPECT_FALSE(SamePlace(landed, other));
  other = landed;
  other.backing = BackingPlace{
      .reservation = {}, .offset = Bytes(kSize), .size = Bytes(kSize), .allocation_class = 0};
  EXPECT_FALSE(SamePlace(landed, other));
  other = landed;
  other.backing.reset();
  EXPECT_FALSE(SamePlace(landed, other));
  // Pieces: each piece's destination and length.
  PageSource pieces = landed;
  pieces.destination = 0;
  pieces.piece_count = 2;
  pieces.pieces[0] = LandedPiece{.slot_offset = 0, .destination = 0x10000, .length = Bytes(64)};
  pieces.pieces[1] = LandedPiece{.slot_offset = 64, .destination = 0x10100, .length = Bytes(64)};
  other = pieces;
  other.pieces[1].slot_offset = 128;
  EXPECT_TRUE(SamePlace(pieces, other));
  other.pieces[1].destination = 0x10200;
  EXPECT_FALSE(SamePlace(pieces, other));
  other = pieces;
  other.pieces[0].length = Bytes(32);
  EXPECT_FALSE(SamePlace(pieces, other));
  EXPECT_FALSE(SamePlace(pieces, landed));
  // Direct: the memory the read lands in.
  PageSource direct = landed;
  direct.landed = false;
  direct.destination = 0;
  std::array<std::byte, 16> memory{};
  direct.read.memory = memory.data();
  other = direct;
  other.read.offset = kSize;
  EXPECT_TRUE(SamePlace(direct, other));
  other.read.memory = memory.data() + 8;
  EXPECT_FALSE(SamePlace(direct, other));
}

// RE-029: with a copy lane (and a VMM lane, as the paged node has them), a
// landed page-in never needs the device lane, whose submission thread a
// job may hold in a launch into a full stream; a job still runs on the
// device lane. Only with a VMM lane: VMM work on the device lane needs it.
class CopyLaneTest : public PageInTest {};

TEST_P(CopyLaneTest, ACopyLaneLandsPageInsWhileTheDeviceLaneIsHeld) {
  Build(kSlots, 32, kSize * 64, jitllm::providers::kNoCoalescing, /*copy=*/true);
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, All())).has_value());
  for (int i = 0; i < 1000 && Round(true, true, /*device_lane=*/false); ++i) {
  }
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  for (std::size_t i = 0; i < kExtents; ++i) {
    EXPECT_TRUE(Loaded(i, weights_)) << i;
  }
  LoadProgram::Report job;
  ASSERT_TRUE(scheduler_->Start(2, Load(job, Of({0}), JobResult::kQueued)).has_value());
  for (int i = 0; i < 100; ++i) {
    (void)Round(true, true, false);
  }
  EXPECT_EQ(job.job_runs, 0);  // the device lane is held
  Settle();
  EXPECT_EQ(job.job_runs, 1);
  EXPECT_EQ(job.outcome, TaskOutcome::kSucceeded);
}

// A landed read may land in pieces, each copied to its own destination
// (an expert slab's page); pieces are refused for write-back and beyond
// the read.
TEST_P(PageInTest, ALandedReadLandsInPieces) {
  Build();
  const std::uint64_t half = kSize / 2;
  PageSource source = Source(0, weights_);
  source.destination = 0;
  source.piece_count = 2;
  source.pieces[0] = {
      .slot_offset = 0, .destination = Place(weights_, 0) + half, .length = Bytes(half)};
  source.pieces[1] = {
      .slot_offset = half, .destination = Place(weights_, 0), .length = Bytes(half)};
  ASSERT_TRUE(scheduler_->SetSource(extents_[0], source).has_value());
  LoadProgram::Report report;
  ASSERT_TRUE(scheduler_->Start(1, Load(report, Of({0}))).has_value());
  Settle();
  EXPECT_EQ(report.outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(std::memcmp(At(Place(weights_, 0)), file_.data() + half, half), 0);
  EXPECT_EQ(std::memcmp(At(Place(weights_, 0) + half), file_.data(), half), 0);

  PageSource beyond = source;
  beyond.pieces[1].slot_offset = half + 1;
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[1], beyond)), WorkError::kInvalid);
  PageSource written = source;
  written.write_back = true;
  EXPECT_EQ(Failed(scheduler_->SetSource(extents_[1], written)), WorkError::kInvalid);
}

INSTANTIATE_TEST_SUITE_P(VmmWork, PageInTest, ::testing::Bool(), [](const auto& info) {
  return info.param ? std::string("OnAVmmLane") : std::string("OnTheDeviceLane");
});
INSTANTIATE_TEST_SUITE_P(VmmWork, CopyLaneTest, ::testing::Values(true),
                         [](const auto& /*info*/) { return std::string("OnAVmmLane"); });

}  // namespace
