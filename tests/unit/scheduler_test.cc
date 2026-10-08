// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler thread's turn loop and its lanes (D-048,
// docs/async-model.md) on the deterministic fakes: page-ins through the
// storage lane, copies through the device lanes and verification on a CPU
// worker, with cancellation through submission, completion and memory
// retirement, full queues during cleanup, late, duplicate and
// contradictory observations, unknown outcomes that quarantine, task trees
// that unwind, and shutdown. The deterministic tests turn each lane
// themselves, in a fixed order; the threaded ones run every lane on its own
// thread, with the owner sleeping on its wake flag, and are the lost-wakeup
// and memory-ordering stress that runs under ThreadSanitizer and on AArch64
// in `check:spark`.

#include "scheduler/scheduler.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <system_error>
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
#include "providers/uring_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"
#include "scheduler/services.h"
#include "scheduler/tasks.h"

#ifdef __SANITIZE_THREAD__
#define LLMP_TEST_TSAN 1
#elifdef __has_feature
#if __has_feature(thread_sanitizer)
#define LLMP_TEST_TSAN 1
#endif
#endif

namespace {

using llmp::base::Bytes;
using llmp::base::PushResult;
using llmp::catalog::Closure;
using llmp::catalog::ExtentId;
using llmp::catalog::ExtentState;
using llmp::catalog::ExtentView;
using llmp::catalog::Occupancy;
using llmp::providers::ProviderError;
using llmp::providers::ReaderSettings;
using llmp::providers::ReadSpec;
using llmp::providers::StreamId;
using llmp::providers::Submission;
using llmp::providers::UringStorage;
using llmp::providers::fake::FakeDeviceExecution;
using llmp::providers::fake::FakeDeviceMemory;
using llmp::providers::fake::FakeStorage;
using llmp::providers::fake::kPoison;
using llmp::scheduler::Acceptance;
using llmp::scheduler::CancelRead;
using llmp::scheduler::CancelRequest;
using llmp::scheduler::CompletionBoard;
using llmp::scheduler::Control;
using llmp::scheduler::CpuCommand;
using llmp::scheduler::CpuHandler;
using llmp::scheduler::CpuJob;
using llmp::scheduler::CpuResult;
using llmp::scheduler::DeviceService;
using llmp::scheduler::DeviceSettings;
using llmp::scheduler::DeviceWork;
using llmp::scheduler::Fault;
using llmp::scheduler::Lane;
using llmp::scheduler::Lanes;
using llmp::scheduler::LaneSettings;
using llmp::scheduler::OperationId;
using llmp::scheduler::Outcome;
using llmp::scheduler::Published;
using llmp::scheduler::QueueSettings;
using llmp::scheduler::ReadCommand;
using llmp::scheduler::Readiness;
using llmp::scheduler::Scheduler;
using llmp::scheduler::SchedulerSettings;
using llmp::scheduler::StartError;
using llmp::scheduler::StartRequest;
using llmp::scheduler::Step;
using llmp::scheduler::StorageService;
using llmp::scheduler::TaskContext;
using llmp::scheduler::TaskOutcome;
using llmp::scheduler::TaskProgram;
using llmp::scheduler::Terminal;
using llmp::scheduler::WorkError;
using llmp::test_support::Failed;

constexpr std::uint64_t kSize = 64ULL * 1024;          // one extent
constexpr std::size_t kSources = 4;                    // extents loaded from the file
constexpr std::size_t kScratch = 8;                    // resident destinations, one per task slot
constexpr auto kPatience = std::chrono::seconds(120);  // a hang, not a timing

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::vector<std::byte> FileContents() {
  std::vector<std::byte> bytes(kSources * kSize);
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::byte>((i * 131) + (i >> 16) + 1);
  }
  return bytes;
}

// Posts a control, retrying while the queue is full. Post moves from the
// control only when it takes it.
void Offer(Scheduler& scheduler, Control&& control) {
  // NOLINTNEXTLINE(bugprone-use-after-move): a refused control is untouched
  while (scheduler.Post(std::move(control)) == PushResult::kFull) {
    std::this_thread::yield();
  }
}

// A stopped scheduler's fault, if it had one; a scheduler still running
// reports kUnproven, so a test cannot mistake it for a clean stop.
std::optional<Fault> FaultOf(const std::optional<std::expected<void, Fault>>& stopped) {
  if (!stopped) {
    return Fault::kUnproven;
  }
  return stopped->has_value() ? std::nullopt : std::optional(stopped->error());
}

// What a test's task reports, on the scheduler thread.
struct Report {
  std::optional<TaskOutcome> outcome;
  bool retired = false;
  std::optional<WorkError> error;
  std::vector<OperationId> operations;
};

// Loads a closure, copies with one device operation, and optionally
// verifies the copy on a CPU worker.
class CopyProgram final : public TaskProgram {
 public:
  CopyProgram(Report& report, std::vector<int>& retirements, int tag, Closure load, Closure use,
              Closure check, DeviceWork work, CpuJob verify)
      : report_(report),
        retirements_(retirements),
        tag_(tag),
        load_(std::move(load)),
        use_(std::move(use)),
        check_(std::move(check)),
        work_(work),
        verify_(std::move(verify)) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    switch (phase_) {
      case Phase::kLoad: {
        const auto ready = context.Materialize(load_);
        if (!ready) {
          return Fail(ready.error());
        }
        if (*ready == Readiness::kWaiting) {
          return Step::Wait();
        }
        phase_ = Phase::kCopy;
        [[fallthrough]];
      }
      case Phase::kCopy: {
        const auto operation = context.SubmitDevice(use_, work_);
        if (!operation) {
          return Fail(operation.error());
        }
        report_.operations.push_back(*operation);
        phase_ = Phase::kVerify;
        return Step::Wait();
      }
      case Phase::kVerify: {
        if (!verify_) {
          return Step::Finish(TaskOutcome::kSucceeded);
        }
        const auto operation = context.SubmitCpu(check_, std::move(verify_));
        if (!operation) {
          return Fail(operation.error());
        }
        report_.operations.push_back(*operation);
        phase_ = Phase::kDone;
        return Step::Wait();
      }
      case Phase::kDone:
        break;
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override {
    report_.retired = true;
    retirements_.push_back(tag_);
  }

 private:
  enum class Phase : std::uint8_t { kLoad, kCopy, kVerify, kDone };
  Step Fail(WorkError error) {
    report_.error = error;
    return Step::Finish(TaskOutcome::kFailed);
  }

  Report& report_;
  std::vector<int>& retirements_;
  int tag_;
  Closure load_;
  Closure use_;
  Closure check_;
  DeviceWork work_;
  CpuJob verify_;
  Phase phase_ = Phase::kLoad;
};

// Spawns its children, waits for them, and finishes.
class ParentProgram final : public TaskProgram {
 public:
  ParentProgram(Report& report, std::vector<int>& retirements, int tag,
                std::vector<std::unique_ptr<TaskProgram>> children)
      : report_(report), retirements_(retirements), tag_(tag), children_(std::move(children)) {}

  Step Advance(TaskContext& context) override {
    if (!spawned_) {
      spawned_ = true;
      for (std::unique_ptr<TaskProgram>& child : children_) {
        if (!context.Spawn(std::move(child))) {
          return Step::Finish(TaskOutcome::kFailed);
        }
      }
      return Step::Wait();
    }
    return Step::Finish(context.ChildFailed() ? TaskOutcome::kFailed : TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override {
    report_.retired = true;
    retirements_.push_back(tag_);
  }

 private:
  Report& report_;
  std::vector<int>& retirements_;
  int tag_;
  std::vector<std::unique_ptr<TaskProgram>> children_;
  bool spawned_ = false;
};

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const Bytes total(kSize * (kSources + kScratch));
    reservation_ = memory_.Reserve(total).value();
    backing_ = memory_.Create(1, total).value();
    ASSERT_TRUE(memory_.Map(reservation_, Bytes(0), backing_).has_value());
    ASSERT_TRUE(
        memory_.SetAccess(reservation_, Bytes(0), total, llmp::providers::Access::kReadWrite)
            .has_value());
    base_ = memory_.RangeOf(reservation_).value().base;
    file_ = FileContents();
    fd_ = storage_.AddFile(file_);
    domain_ = catalog_.AddDomain("node");
    for (std::size_t i = 0; i < kSources; ++i) {
      sources_.at(i) =
          catalog_
              .AddExtent(
                  {.domain = domain_,
                   .memory_class = llmp::catalog::MemoryClass::kWeights,
                   .recovery = llmp::catalog::Recovery::kFromArtifact,
                   .size = Bytes(kSize),
                   .content = {.artifact = {}, .group = 0, .chunk = static_cast<std::uint32_t>(i)}})
              .value();
    }
    for (std::size_t i = 0; i < kScratch; ++i) {
      scratch_.at(i) = catalog_
                           .AddExtent({.domain = domain_,
                                       .memory_class = llmp::catalog::MemoryClass::kScratch,
                                       .recovery = llmp::catalog::Recovery::kDiscardable,
                                       .size = Bytes(kSize),
                                       .content = {}},
                                      true)
                           .value();
    }
    stream_ = execution_.CreateStream().value();
  }

  // The lanes and the scheduler, with a test's queue sizes and, if asked,
  // a CPU lane of `cpu` workers.
  void Build(QueueSettings storage = {.capacity = 16, .reserved = 4, .batch = 16},
             QueueSettings device = {.capacity = 16, .reserved = 4, .batch = 16},
             std::size_t cpu = 0,
             std::chrono::microseconds poll_window = std::chrono::microseconds(200),
             std::chrono::milliseconds tick = std::chrono::milliseconds(100)) {
    storage_lane_ = std::make_unique<StorageService>(
        storage_,
        // One request per 16 KiB piece, as these tests count them.
        ReaderSettings{.alignment = 4096,
                       .request_bytes = 16 * 1024,
                       .retries = 2,
                       .reads = 16,
                       .waiters = 8,
                       .span_bytes = llmp::providers::kNoCoalescing,
                       .span_segments = llmp::providers::kMaxSegments},
        board_, storage);
    device_lane_ =
        std::make_unique<DeviceService>(execution_, std::span<const StreamId>(&stream_, 1), board_,
                                        DeviceSettings{.queue = device, .handoff = 16});
    if (cpu > 0) {
      cpu_lane_ = std::make_unique<Lane<CpuCommand>>(
          LaneSettings{.name = "cpu", .capacity = 16, .reserved = 4, .workers = cpu},
          CpuHandler(board_));
    }
    scheduler_ = std::make_unique<Scheduler>(
        catalog_, board_, wake_,
        Lanes{.storage = storage_lane_.get(), .device = device_lane_.get(), .cpu = cpu_lane_.get()},
        SchedulerSettings{.tasks = kScratch,
                          .priorities = 2,
                          .aging_limit = 4,
                          .controls = 64,
                          .controls_reserved = 16,
                          .controls_per_turn = 16,
                          .observations_per_turn = 64,
                          .steps_per_turn = 16,
                          .waiters = 8,
                          .budget = Bytes(kSize * (kSources + kScratch)),
                          .poll_window = poll_window,
                          .tick = tick});
    for (std::size_t i = 0; i < kSources; ++i) {
      scheduler_->SetSource(
          sources_.at(i),
          ReadSpec{.fd = fd_, .offset = i * kSize, .memory = At(Source(i)), .length = kSize});
    }
  }

  void TearDown() override {
    // Nothing may be left in flight: held reads complete, the scheduler
    // stops, and the lanes drain.
    std::vector<std::uint64_t> tokens;
    for (const auto& request : storage_.submitted()) {
      tokens.push_back(request.token);
    }
    for (const std::uint64_t token : tokens) {
      (void)storage_.Release(token);
    }
    if (scheduler_ != nullptr) {
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
      cpu_lane_.reset();
      DrainLanes();
    }
  }

  // One turn of each lane and of the scheduler, in a fixed order. With
  // `device`, the fake device runs what is queued on its streams.
  bool Round(bool storage = true, bool device_lanes = true, bool device = true) {
    bool progress = false;
    if (storage) {
      progress = storage_lane_->Turn(false) || progress;
    }
    if (device_lanes) {
      progress = device_lane_->SubmissionTurn() || progress;
    }
    if (device) {
      execution_.Drain();
    }
    if (device_lanes) {
      progress = device_lane_->CompletionTurn() || progress;
    }
    return scheduler_->Turn() || progress;
  }
  // Rounds until nothing moves.
  void Settle(bool storage = true, bool device_lanes = true, bool device = true) {
    for (int i = 0; i < 1000 && Round(storage, device_lanes, device); ++i) {
    }
  }
  // Rounds until `done`, for work on a CPU worker's thread.
  template <typename Done>
  bool SettleUntil(Done done) {
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (!done() && std::chrono::steady_clock::now() < give_up) {
      if (!Round()) {
        std::this_thread::yield();
      }
    }
    return done();
  }
  void DrainLanes() {
    for (int i = 0; i < 100; ++i) {
      (void)storage_lane_->Turn(false);
      (void)device_lane_->SubmissionTurn();
      execution_.Drain();
      (void)device_lane_->CompletionTurn();
    }
  }

  std::uint64_t Source(std::size_t i) const { return base_ + (i * kSize); }
  std::uint64_t Scratch(std::size_t i) const { return base_ + ((kSources + i) * kSize); }
  bool Matches(std::uint64_t address, std::size_t source) const {
    return std::memcmp(At(address), file_.data() + (source * kSize), kSize) == 0;
  }
  static bool Poisoned(std::uint64_t address) {
    return *At(address) == kPoison && *At(address + kSize - 1) == kPoison;
  }
  ExtentView View(ExtentId extent) const { return catalog_.Describe(extent).value(); }
  Occupancy Occupied() const { return catalog_.OccupancyOf(domain_); }
  Closure Of(std::initializer_list<ExtentId> extents) const {
    return catalog_.ClosureOfExtents(std::span<const ExtentId>(extents.begin(), extents.size()))
        .value();
  }
  static DeviceWork CopyWork(std::uint64_t from, std::uint64_t to) {
    DeviceWork work;
    work.copies.at(0) = {.destination = to, .source = from, .size = Bytes(kSize)};
    work.count = 1;
    return work;
  }
  // A task that loads source `from` and copies it into scratch `to`,
  // reporting to reports_[report]; with `verify`, a CPU worker checks it.
  std::unique_ptr<TaskProgram> CopyTask(std::size_t report, std::size_t from, std::size_t to,
                                        bool verify = false) {
    CpuJob job;
    if (verify) {
      job = [this, from, to] {
        return CpuResult{
            .outcome = Matches(Scratch(to), from) ? Outcome::kSucceeded : Outcome::kFailed,
            .bytes = kSize};
      };
    }
    return std::make_unique<CopyProgram>(
        reports_.at(report), retirements_, static_cast<int>(report), Of({sources_.at(from)}),
        Of({sources_.at(from), scratch_.at(to)}), Of({scratch_.at(to)}),
        CopyWork(Source(from), Scratch(to)), std::move(job));
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

  // Declared in dependency order: the scheduler goes first, the providers last.
  FakeDeviceMemory memory_{Bytes(kSize), Bytes(kSize * 64)};
  FakeStorage storage_{8, 4096};
  FakeDeviceExecution execution_;
  llmp::catalog::Catalog catalog_;
  llmp::base::WakeFlag wake_;
  CompletionBoard board_{16, wake_};
  std::array<Report, 8> reports_;
  std::vector<int> retirements_;
  std::unique_ptr<StorageService> storage_lane_;
  std::unique_ptr<DeviceService> device_lane_;
  std::unique_ptr<Lane<CpuCommand>> cpu_lane_;
  std::unique_ptr<Scheduler> scheduler_;

  llmp::providers::ReservationId reservation_;
  llmp::providers::BackingId backing_;
  std::uint64_t base_ = 0;
  std::vector<std::byte> file_;
  int fd_ = -1;
  llmp::catalog::DomainId domain_;
  std::array<ExtentId, kSources> sources_;
  std::array<ExtentId, kScratch> scratch_;
  StreamId stream_;
};

TEST_F(SchedulerTest, APageInFeedsDeviceWorkAndVerificationThenEverythingRetires) {
  Build({.capacity = 16, .reserved = 4, .batch = 16}, {.capacity = 16, .reserved = 4, .batch = 16},
        1);
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0, true)).has_value());
  ASSERT_TRUE(SettleUntil([&] { return reports_[0].retired; }));
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(reports_[0].operations.size(), 2U);
  EXPECT_TRUE(Matches(Scratch(0), 0));
  EXPECT_EQ(storage_.submitted().size(), 4U);  // 64 KiB as four 16 KiB requests
  const ExtentView loaded = View(sources_[0]);
  EXPECT_EQ(loaded.state, ExtentState::kResident);
  EXPECT_EQ(loaded.leases, 0U);
  EXPECT_GT(loaded.last_use, 0U);
  EXPECT_EQ(View(scratch_[0]).leases, 0U);
  const Occupancy occupied = Occupied();
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(occupied.idle, Bytes(kSize * (1 + kScratch)));
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
  EXPECT_FALSE(scheduler_->fault().has_value());
  EXPECT_EQ(execution_.fences(), 0U);  // seen complete and released
}

// The page-in belongs to the scheduler: the task retires at once, but the
// extent stays LOADING, and charged, until the read's completion proves it
// untouched.
TEST_F(SchedulerTest, CancellingDuringAPageInReleasesBackingOnlyOnceTheReadDrains) {
  Build();
  HoldNext(4);
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle();
  ASSERT_EQ(View(sources_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(Occupied().loading, Bytes(kSize));
  EXPECT_EQ(scheduler_->operations(), 1U);
  const std::uint64_t generation = View(sources_[0]).backing_generation;

  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kCancelled);
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(Occupied().loading, Bytes(kSize));
  EXPECT_EQ(scheduler_->operations(), 1U);

  Settle();  // the storage lane asks the provider to cancel; the read drains
  const ExtentView drained = View(sources_[0]);
  EXPECT_EQ(drained.state, ExtentState::kNonresident);
  EXPECT_EQ(drained.backing_generation, generation + 1);
  EXPECT_EQ(Occupied().loading, Bytes());
  EXPECT_EQ(Occupied().Total(), Bytes(kSize * kScratch));
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_TRUE(Poisoned(Source(0)));  // the cancelled requests moved nothing
  EXPECT_TRUE(Poisoned(Scratch(0)));
}

TEST_F(SchedulerTest, CancellingOneWaiterLeavesTheSharedPageInToTheOthers) {
  Build();
  HoldNext(4);
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  ASSERT_TRUE(scheduler_->Start(2, CopyTask(1, 0, 1)).has_value());
  Settle();
  EXPECT_EQ(storage_.submitted().size(), 4U);  // one read for both
  EXPECT_EQ(scheduler_->operations(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_TRUE(reports_[0].retired);
  Settle();
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kLoading);  // not cancelled
  ReleaseReads();
  Settle();
  EXPECT_EQ(reports_[1].outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Matches(Scratch(1), 0));
  EXPECT_TRUE(Poisoned(Scratch(0)));
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kResident);
  EXPECT_EQ(scheduler_->operations(), 0U);
}

// Accepted device work cannot be recalled: the task finishes cancelled at
// once, but its lease and its slot stay until the fence is seen complete.
TEST_F(SchedulerTest, CancellingDuringDeviceWorkKeepsItsLeaseUntilTheFenceCompletes) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  EXPECT_EQ(View(sources_[0]).leases, 1U);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);
  EXPECT_EQ(Occupied().held, Bytes(2 * kSize));

  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kCancelled);
  Settle(true, true, false);
  EXPECT_FALSE(reports_[0].retired);
  EXPECT_EQ(scheduler_->tasks().size(), 1U);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);
  EXPECT_EQ(scheduler_->operations(), 1U);

  Settle();  // the device runs; the completion lane sees the fence
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_EQ(View(sources_[0]).leases, 0U);
  EXPECT_EQ(View(scratch_[0]).leases, 0U);
  EXPECT_EQ(Occupied().held, Bytes());
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
  EXPECT_EQ(execution_.fences(), 0U);
  EXPECT_TRUE(Matches(Scratch(0), 0));  // the accepted copy ran
}

// Published but not yet taken by the lane: cancellation cannot know
// whether it will start, so the operation stays owned until it resolves.
TEST_F(SchedulerTest, CancellingBeforeTheLaneTakesACommandLeavesItOwned) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, false, false);  // the command waits in the device lane's queue
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_FALSE(reports_[0].retired);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);
  Settle();
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_EQ(View(scratch_[0]).leases, 0U);
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_TRUE(Matches(Scratch(0), 0));
}

// A full lane leaves the command with its operation (backpressure, never a
// blocked scheduler); cancelling before any lane took it rolls it back.
TEST_F(SchedulerTest, AFullLaneKeepsTheCommandAndCancellingRollsItBack) {
  Build({.capacity = 16, .reserved = 4, .batch = 16}, {.capacity = 2, .reserved = 1, .batch = 16});
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  ASSERT_TRUE(scheduler_->Start(2, CopyTask(1, 1, 1)).has_value());
  Settle(true, false, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  ASSERT_EQ(reports_[1].operations.size(), 1U);
  EXPECT_EQ(scheduler_->operations(), 2U);
  EXPECT_EQ(View(scratch_[1]).leases, 1U);  // prepared: leased before publication

  ASSERT_TRUE(scheduler_->Cancel(2));
  EXPECT_TRUE(reports_[1].retired);
  EXPECT_EQ(View(scratch_[1]).leases, 0U);
  EXPECT_EQ(View(sources_[1]).leases, 0U);
  EXPECT_EQ(scheduler_->operations(), 1U);
  Settle();
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(Matches(Scratch(0), 0));
  EXPECT_TRUE(Poisoned(Scratch(1)));  // no lane ever saw it
  EXPECT_EQ(scheduler_->operations(), 0U);
}

// Cancellations use the storage lane's cleanup reserve; when even that is
// full, the intent is kept and asked again, and the reads drain meanwhile.
TEST_F(SchedulerTest, CancellationUsesTheCleanupReserveAndKeepsItsIntentWhenThatIsFull) {
  Build({.capacity = 3, .reserved = 1, .batch = 16});
  HoldNext(8);
  for (std::size_t r = 0; r < 3; ++r) {
    ASSERT_TRUE(scheduler_->Start(r + 1, CopyTask(r, r, r)).has_value());
  }
  Settle(false, true, true);                // the storage lane does not run
  EXPECT_EQ(scheduler_->operations(), 3U);  // two reads queued, one waiting in the scheduler
  ASSERT_TRUE(scheduler_->Cancel(1));       // from the reserve: the queue is now full
  ASSERT_TRUE(scheduler_->Cancel(2));       // refused: the intent stays
  ASSERT_TRUE(scheduler_->Cancel(3));       // never published: rolled back
  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_TRUE(reports_.at(r).retired);
  }
  EXPECT_EQ(View(sources_[2]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kLoading);
  EXPECT_EQ(View(sources_[1]).state, ExtentState::kLoading);
  EXPECT_EQ(scheduler_->operations(), 2U);

  Settle();
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(View(sources_[1]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().loading, Bytes());
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_TRUE(Poisoned(Source(0)));
  EXPECT_TRUE(Poisoned(Source(1)));
}

// Cancellation is an intent per request, not a queued control per call:
// repeating it costs nothing, however often. A start posted after it opens
// a new intent, so a cancellation posted after that start is not merged
// into one applied before the task existed.
TEST_F(SchedulerTest, RepeatedCancellationsOfARequestShareOneQueuedIntent) {
  Build();
  Offer(*scheduler_, StartRequest{.request = 1, .priority = 1, .program = CopyTask(0, 0, 0)});
  // More than the whole control queue holds (64).
  for (int i = 0; i < 200; ++i) {
    Control cancel = CancelRequest{.request = 1};
    ASSERT_EQ(scheduler_->Post(std::move(cancel)), PushResult::kAccepted) << i;
  }
  Control early = CancelRequest{.request = 2};
  ASSERT_EQ(scheduler_->Post(std::move(early)), PushResult::kAccepted);  // no such request yet
  Offer(*scheduler_, StartRequest{.request = 2, .priority = 1, .program = CopyTask(1, 1, 1)});
  Control late = CancelRequest{.request = 2};
  ASSERT_EQ(scheduler_->Post(std::move(late)), PushResult::kAccepted);
  // Ordinary capacity is untouched by the repeats.
  for (std::size_t r = 0; r < 3; ++r) {
    Offer(*scheduler_,
          StartRequest{.request = 10 + r, .priority = 1, .program = CopyTask(2 + r, 2, 2 + r)});
  }
  Settle();
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(reports_[1].outcome, TaskOutcome::kCancelled);
  EXPECT_TRUE(Poisoned(Scratch(0)));
  EXPECT_TRUE(Poisoned(Scratch(1)));
  for (std::size_t r = 2; r < 5; ++r) {
    EXPECT_EQ(reports_.at(r).outcome, TaskOutcome::kSucceeded);
  }
  // A tag used again: its new start opens a new intent.
  Offer(*scheduler_, StartRequest{.request = 1, .priority = 1, .program = CopyTask(5, 3, 5)});
  Control again = CancelRequest{.request = 1};
  ASSERT_EQ(scheduler_->Post(std::move(again)), PushResult::kAccepted);
  Settle();
  EXPECT_EQ(reports_[5].outcome, TaskOutcome::kCancelled);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
}

// A tag used again while its earlier start still drains names both starts:
// a cancellation reaches the live one, not only the one already finished.
TEST_F(SchedulerTest, ACancellationReachesEveryStartOfATagUsedAgain) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);  // device work accepted, held on the device
  ASSERT_TRUE(scheduler_->Cancel(1));
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kCancelled);
  EXPECT_FALSE(reports_[0].retired);  // its copy still drains

  Offer(*scheduler_, StartRequest{.request = 1, .priority = 1, .program = CopyTask(1, 1, 1)});
  Settle(true, true, false);
  ASSERT_EQ(reports_[1].operations.size(), 1U);
  Control cancel = CancelRequest{.request = 1};
  ASSERT_EQ(scheduler_->Post(std::move(cancel)), PushResult::kAccepted);
  Settle(true, true, false);
  EXPECT_EQ(reports_[1].outcome, TaskOutcome::kCancelled);
  Settle();
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_TRUE(reports_[1].retired);
  EXPECT_EQ(reports_[1].operations.size(), 1U);  // no verification after the cancel
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
  EXPECT_EQ(scheduler_->operations(), 0U);
}

// A cancellation reaches only the starts posted before it: one posted
// after it runs to completion.
TEST_F(SchedulerTest, ACancellationPostedBeforeAStartLeavesThatStartAlone) {
  Build();
  Control cancel = CancelRequest{.request = 7};
  ASSERT_EQ(scheduler_->Post(std::move(cancel)), PushResult::kAccepted);
  Offer(*scheduler_, StartRequest{.request = 7, .priority = 1, .program = CopyTask(0, 0, 0)});
  Settle();
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_TRUE(Matches(Scratch(0), 0));
}

// The storage lane publishes nothing for a cancellation it takes, since the
// read may already have ended; taking it still makes room the owner may be
// waiting for, holding a cancellation that queue refused. The lane wakes
// the owner then, or the owner sleeps a whole tick (a timer that is never
// needed for correctness) with a read it meant to cancel still in flight.
TEST_F(SchedulerTest, TheStorageLaneWakesTheOwnerWhenItMakesRoomWithoutPublishing) {
  Build({.capacity = 2, .reserved = 1, .batch = 16});
  HoldNext(100);  // refused submissions use scripts too
  for (std::size_t r = 0; r < 3; ++r) {
    ASSERT_TRUE(scheduler_->Start(r + 1, CopyTask(r, r, r)).has_value());
  }
  Settle();
  ASSERT_EQ(scheduler_->operations(), 3U);
  ASSERT_GE(storage_.submitted().size(), 8U);  // the first two reads, in flight
  for (std::size_t i = 0; i < 8; ++i) {
    ASSERT_TRUE(storage_.Release(storage_.submitted().at(i).token));
  }
  // The first two reads end and are published; the third's requests start.
  ASSERT_TRUE(storage_lane_->Turn(false));
  ASSERT_EQ(storage_lane_->reads(), 1U);
  // Before the owner harvests them, every request is cancelled: two
  // cancellations fill the storage queue, and the third waits.
  for (std::uint64_t r = 1; r <= 3; ++r) {
    ASSERT_TRUE(scheduler_->Cancel(r));
  }
  (void)wake_.Consume();
  ASSERT_TRUE(scheduler_->Turn());  // concludes the first two; the queue is still full
  (void)wake_.Consume();
  ASSERT_FALSE(scheduler_->Turn());  // nothing moved: the owner would sleep now
  ASSERT_EQ(View(sources_[2]).state, ExtentState::kLoading);

  // The lane takes both cancellations: their reads have ended, so nothing is
  // published, but the queue has room.
  (void)storage_lane_->Turn(false);
  EXPECT_TRUE(wake_.Consume()) << "the owner sleeps with a cancellation it could now send";
  ASSERT_TRUE(scheduler_->Turn());
  Settle();
  EXPECT_EQ(View(sources_[2]).state, ExtentState::kNonresident);
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_TRUE(Poisoned(Source(2)));
}

TEST_F(SchedulerTest, LateAndDuplicateObservationsChangeNothing) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  const OperationId copy = reports_[0].operations[0];
  EXPECT_EQ(board_.Accept(copy, Acceptance::kAccepted), Published::kDuplicate);
  execution_.Drain();
  (void)device_lane_->CompletionTurn();  // publishes the completion
  const Terminal done{.outcome = Outcome::kSucceeded, .bytes = kSize, .no_further_access = true};
  EXPECT_EQ(board_.Complete(copy, done), Published::kDuplicate);
  Settle();
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kSucceeded);
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_EQ(scheduler_->operations(), 0U);

  // News for the retired operation is stale, and changes nothing.
  EXPECT_EQ(
      board_.Complete(copy, {.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = true}),
      Published::kStale);
  EXPECT_EQ(board_.Accept(copy, Acceptance::kNotStarted), Published::kStale);
  const Occupancy before = Occupied();
  Settle();
  EXPECT_EQ(Occupied().Total(), before.Total());
  EXPECT_EQ(Occupied().idle, before.idle);
  EXPECT_FALSE(scheduler_->fault().has_value());
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kResident);
  EXPECT_EQ(View(sources_[0]).leases, 0U);
}

// A lane that contradicts itself is a provider fault: the operation is
// quarantined with everything it holds, and the node stops admitting.
TEST_F(SchedulerTest, ContradictoryObservationsQuarantineAndStopAdmission) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  const OperationId copy = reports_[0].operations[0];
  EXPECT_EQ(board_.Accept(copy, Acceptance::kNotStarted), Published::kContradiction);
  Settle(true, true, false);
  EXPECT_EQ(scheduler_->fault(), Fault::kContradiction);
  EXPECT_EQ(scheduler_->quarantined(), 1U);
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kFailed);
  EXPECT_FALSE(reports_[0].retired);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);
  EXPECT_EQ(Failed(scheduler_->Start(2, CopyTask(1, 1, 1))), StartError::kStopped);

  Settle();  // the device finishes; the quarantine stands
  EXPECT_EQ(View(sources_[0]).leases, 1U);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);
  EXPECT_EQ(Occupied().held, Bytes(2 * kSize));
  scheduler_->RequestShutdown();
  Settle();
  ASSERT_TRUE(scheduler_->Stopped().has_value());
  EXPECT_EQ(FaultOf(scheduler_->Stopped()), Fault::kContradiction);
}

TEST_F(SchedulerTest, AnUnknownFenceQuarantinesWhatTheWorkHolds) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  execution_.FailNextQuery(ProviderError::kUnknown);
  Settle(true, true, false);
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  EXPECT_EQ(scheduler_->quarantined(), 1U);
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kFailed);
  EXPECT_FALSE(reports_[0].retired);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);
  EXPECT_EQ(Occupied().held, Bytes(2 * kSize));  // stays charged
  scheduler_->RequestShutdown();
  Settle();
  ASSERT_TRUE(scheduler_->Stopped().has_value());
  EXPECT_EQ(FaultOf(scheduler_->Stopped()), Fault::kUnproven);
}

// A query the provider keeps refusing (a lost context: the call changes
// nothing, but no answer comes) is not retried forever: after the bound it
// leaves the work unproven, which quarantines what the work holds, so the
// completion lane and shutdown are not held hostage.
TEST_F(SchedulerTest, APersistentlyRefusedQueryQuarantinesInsteadOfAskingForever) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  execution_.FailNextQuery(ProviderError::kFailed, SIZE_MAX);
  for (int i = 0; i < 64; ++i) {
    (void)Round(true, true, false);
  }
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  EXPECT_EQ(scheduler_->quarantined(), 1U);
  EXPECT_EQ(View(scratch_[0]).leases, 1U);  // stays charged
  execution_.FailNextQuery(ProviderError::kFailed, 0);
  scheduler_->RequestShutdown();
  Settle();
  EXPECT_EQ(FaultOf(scheduler_->Stopped()), Fault::kUnproven);
}

// A release the provider keeps refusing is abandoned once the bound is
// reached (the work was already proven complete), so the completion lane
// can still return when the device service closes.
TEST_F(SchedulerTest, APersistentlyRefusedReleaseDoesNotKeepTheCompletionLane) {
  Build();
  execution_.FailNextRelease(ProviderError::kFailed, SIZE_MAX);
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  ASSERT_TRUE(SettleUntil([&] { return reports_[0].retired; }));
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kSucceeded);
  for (int i = 0; i < 64; ++i) {
    (void)Round();
  }
  EXPECT_EQ(execution_.fences(), 1U);  // never released
  scheduler_->RequestShutdown();
  Settle();
  EXPECT_EQ(FaultOf(scheduler_->Stopped()), std::nullopt);
  device_lane_->Close();
  (void)device_lane_->SubmissionTurn();  // closed and drained: hands over no more
  std::atomic<bool> returned{false};
  std::jthread completion([&] {
    device_lane_->RunCompletion();
    returned.store(true, std::memory_order_release);
  });
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!returned.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_TRUE(returned.load(std::memory_order_acquire));
  execution_.FailNextRelease(ProviderError::kFailed, 0);  // lets a stuck lane go
  completion.join();
}

// The bound counts refusals in a row: an answer (still pending) starts the
// count again, so a fence whose queries are refused now and then, one
// short of the bound each time, still completes with its proof, and a
// release refused one short of the bound still releases.
TEST_F(SchedulerTest, RefusalsThatClearNeitherQuarantineNorAbandon) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[0].operations.size(), 1U);
  constexpr std::size_t kShort = 7;  // DeviceSettings::refusals is 8
  for (int burst = 0; burst < 4; ++burst) {
    execution_.FailNextQuery(ProviderError::kFailed, kShort);
    for (std::size_t i = 0; i <= kShort; ++i) {
      (void)Round(true, true, false);  // one query a round: refused, then pending
    }
  }
  EXPECT_EQ(scheduler_->fault(), std::nullopt);
  EXPECT_EQ(scheduler_->quarantined(), 0U);
  execution_.FailNextQuery(ProviderError::kFailed, kShort);
  execution_.FailNextRelease(ProviderError::kFailed, kShort);
  ASSERT_TRUE(SettleUntil([&] { return reports_[0].retired; }));
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kSucceeded);
  EXPECT_EQ(scheduler_->fault(), std::nullopt);
  for (std::size_t i = 0; i <= kShort; ++i) {
    (void)Round();  // one release attempt a round
  }
  EXPECT_EQ(execution_.fences(), 0U);
}

// The tick bounds the owner's sleep (WakeFlag::WaitFor): one that is not
// positive spins, and a huge one overflows a timed wait's clock arithmetic
// (milliseconds::max() in nanoseconds). The poll window is compared in the
// clock's nanoseconds too. Both are refused when the scheduler is built.
using SchedulerDeathTest = SchedulerTest;
TEST_F(SchedulerDeathTest, AnUnboundedTickOrPollWindowIsRefusedAtConstruction) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  using std::chrono::hours;
  using std::chrono::microseconds;
  using std::chrono::milliseconds;
  const auto build = [this](microseconds poll_window, milliseconds tick) {
    const Scheduler scheduler(catalog_, board_, wake_, Lanes{},
                              SchedulerSettings{.poll_window = poll_window, .tick = tick});
  };
  EXPECT_DEATH(build(microseconds(200), milliseconds::max()), "scheduler tick");
  EXPECT_DEATH(build(microseconds(200), milliseconds(0)), "scheduler tick");
  EXPECT_DEATH(build(microseconds(200), milliseconds(-1)), "scheduler tick");
  EXPECT_DEATH(build(microseconds(200), hours(1) + milliseconds(1)), "scheduler tick");
  EXPECT_DEATH(build(microseconds::max(), milliseconds(100)), "poll window");
  EXPECT_DEATH(build(microseconds(-1), milliseconds(100)), "poll window");
  build(hours(1), hours(1));  // the longest of each
  build(microseconds(0), milliseconds(1));
  // The runtime wake's follow limit, bounded the same way.
  const auto follow = [this](microseconds limit) {
    const Scheduler scheduler(catalog_, board_, wake_, Lanes{},
                              SchedulerSettings{.follow_limit = limit});
  };
  EXPECT_DEATH(follow(microseconds(-1)), "follow limit");
  EXPECT_DEATH(follow(hours(1) + microseconds(1)), "follow limit");
  follow(hours(1));
  follow(microseconds(0));
}

TEST_F(SchedulerTest, AReadEndingWithoutProofQuarantinesItsExtent) {
  Build();
  HoldNext(4);
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle();
  const auto load = scheduler_->LoadOf(sources_[0]);
  ASSERT_TRUE(load.has_value());
  EXPECT_EQ(board_.Complete(load.value_or(OperationId{}),
                            {.outcome = Outcome::kFailed, .bytes = 0, .no_further_access = false}),
            Published::kRecorded);
  Settle();
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kQuarantined);
  EXPECT_EQ(Occupied().quarantined, Bytes(kSize));
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kFailed);
  EXPECT_TRUE(reports_[0].retired);  // the page-in was never the task's
  EXPECT_EQ(scheduler_->operations(), 1U);
}

TEST_F(SchedulerTest, AFailedReadPublishesNothingAndFailsItsWaiter) {
  Build();
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle();
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kFailed);
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kNonresident);
  EXPECT_EQ(Occupied().loading, Bytes());
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_TRUE(Poisoned(Scratch(0)));  // nothing copied from contents never published
  EXPECT_FALSE(scheduler_->fault().has_value());
}

TEST_F(SchedulerTest, AReadTheLaneCannotStartFailsTheLoad) {
  Build();
  // Unaligned memory: the reader refuses it before anything starts.
  scheduler_->SetSource(
      sources_[0],
      ReadSpec{.fd = fd_, .offset = 0, .memory = At(Source(0) + 512), .length = kSize});
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle();
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kFailed);
  EXPECT_TRUE(reports_[0].retired);
  EXPECT_EQ(View(sources_[0]).state, ExtentState::kNonresident);
  EXPECT_TRUE(storage_.submitted().empty());
  EXPECT_EQ(scheduler_->operations(), 0U);
}

TEST_F(SchedulerTest, ACancelledTreeRetiresOnlyAfterItsChildrenDrain) {
  Build();
  std::vector<std::unique_ptr<TaskProgram>> children;
  children.push_back(CopyTask(1, 0, 0));
  children.push_back(CopyTask(2, 1, 1));
  ASSERT_TRUE(scheduler_
                  ->Start(1, std::make_unique<ParentProgram>(reports_[0], retirements_, 0,
                                                             std::move(children)))
                  .has_value());
  Settle(true, true, false);
  ASSERT_EQ(reports_[1].operations.size(), 1U);
  ASSERT_EQ(reports_[2].operations.size(), 1U);
  ASSERT_TRUE(scheduler_->Cancel(1));
  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_EQ(reports_.at(r).outcome, TaskOutcome::kCancelled);
    EXPECT_FALSE(reports_.at(r).retired);
  }
  EXPECT_EQ(scheduler_->tasks().size(), 3U);
  EXPECT_EQ(View(scratch_[1]).leases, 1U);
  Settle();
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
  ASSERT_EQ(retirements_.size(), 3U);
  EXPECT_EQ(retirements_.back(), 0);  // the parent last
  EXPECT_EQ(View(scratch_[0]).leases, 0U);
  EXPECT_EQ(View(scratch_[1]).leases, 0U);
  EXPECT_EQ(scheduler_->operations(), 0U);
}

TEST_F(SchedulerTest, AChildsFailureReachesItsParent) {
  Build();
  storage_.ScriptNext({.submission = Submission::kAccepted, .result = -EIO, .hold = false});
  std::vector<std::unique_ptr<TaskProgram>> children;
  children.push_back(CopyTask(1, 0, 0));
  ASSERT_TRUE(scheduler_
                  ->Start(1, std::make_unique<ParentProgram>(reports_[0], retirements_, 0,
                                                             std::move(children)))
                  .has_value());
  Settle();
  EXPECT_EQ(reports_[1].outcome, TaskOutcome::kFailed);
  EXPECT_EQ(reports_[0].outcome, TaskOutcome::kFailed);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
}

// Submits CPU work over a closure, one operation per step, until a
// submission is refused; records each refusal.
class RepeatProgram final : public TaskProgram {
 public:
  RepeatProgram(Report& report, Closure closure) : report_(report), closure_(std::move(closure)) {}

  Step Advance(TaskContext& context) override {
    const auto operation =
        context.SubmitCpu(closure_, [] { return CpuResult{.outcome = Outcome::kSucceeded}; });
    if (!operation) {
      report_.error = operation.error();
      return Step::Finish(TaskOutcome::kFailed);
    }
    report_.operations.push_back(*operation);
    return Step::Wait();
  }
  void Finished(TaskOutcome outcome) override { report_.outcome = outcome; }
  void Retired() override { report_.retired = true; }

 private:
  Report& report_;
  Closure closure_;
};

// Operation identities end by refusing, never by reuse or abort: a mailbox
// at its last generation retires once its operation closes, later work is
// refused as unavailable (a permanent condition, not "busy"), and the
// node admits no new request; shutdown is still clean.
TEST_F(SchedulerTest, ExhaustedOperationIdentitiesStopAdmissionWithoutAborting) {
  llmp::base::WakeFlag wake;
  CompletionBoard board(2, wake, UINT32_MAX);
  Lane<CpuCommand> cpu(LaneSettings{.name = "cpu", .capacity = 4, .reserved = 1, .workers = 1},
                       CpuHandler(board));
  Scheduler scheduler(catalog_, board, wake,
                      Lanes{.storage = nullptr, .device = nullptr, .cpu = &cpu},
                      SchedulerSettings{.tasks = 4, .budget = Bytes(kSize * kScratch)});
  ASSERT_TRUE(scheduler.Start(1, std::make_unique<RepeatProgram>(reports_[0], Of({scratch_[0]})))
                  .has_value());
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (!reports_[0].retired && std::chrono::steady_clock::now() < give_up) {
    if (!scheduler.Turn()) {
      std::this_thread::yield();
    }
  }
  ASSERT_TRUE(reports_[0].retired);
  EXPECT_EQ(reports_[0].operations.size(), 2U);  // one per mailbox, each at its last generation
  EXPECT_EQ(reports_[0].error, WorkError::kUnavailable);
  EXPECT_TRUE(board.exhausted());
  EXPECT_EQ(board.open(), 0U);
  EXPECT_EQ(View(scratch_[0]).leases, 0U);
  EXPECT_EQ(
      Failed(scheduler.Start(2, std::make_unique<RepeatProgram>(reports_[1], Of({scratch_[1]})))),
      StartError::kStopped);
  scheduler.RequestShutdown();
  while (!scheduler.Stopped() && std::chrono::steady_clock::now() < give_up) {
    (void)scheduler.Turn();
  }
  EXPECT_EQ(FaultOf(scheduler.Stopped()), std::nullopt);
}

// Shutdown stops admission, cancels what is queued, drains what was
// accepted, and only then may the program release backing.
TEST_F(SchedulerTest, ShutdownCancelsQueuedWorkDrainsAcceptedWorkThenBackingIsReleased) {
  Build();
  ASSERT_TRUE(scheduler_->Start(1, CopyTask(0, 0, 0)).has_value());
  Settle(true, true, false);  // device work accepted, held on the device
  HoldNext(4);
  ASSERT_TRUE(scheduler_->Start(2, CopyTask(1, 1, 1)).has_value());
  Settle(true, true, false);  // a page-in in flight
  ASSERT_EQ(
      scheduler_->Post(StartRequest{.request = 3, .priority = 1, .program = CopyTask(2, 2, 2)}),
      PushResult::kAccepted);
  scheduler_->RequestShutdown();
  Settle();
  ASSERT_TRUE(scheduler_->Stopped().has_value());
  EXPECT_EQ(FaultOf(scheduler_->Stopped()), std::nullopt);
  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_EQ(reports_.at(r).outcome, TaskOutcome::kCancelled);
  }
  EXPECT_FALSE(reports_[2].retired);  // refused, never admitted
  Control late = StartRequest{.request = 4, .priority = 1, .program = CopyTask(3, 3, 3)};
  EXPECT_EQ(scheduler_->Post(std::move(late)), PushResult::kClosed);
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
  EXPECT_EQ(Occupied().held, Bytes());
  EXPECT_EQ(Occupied().loading, Bytes());
  EXPECT_EQ(View(sources_[1]).state, ExtentState::kNonresident);

  // Then the lanes close and drain, and the backing goes.
  storage_lane_->Close();
  device_lane_->Close();
  DrainLanes();
  EXPECT_EQ(storage_lane_->reads(), 0U);
  EXPECT_TRUE(execution_.DestroyStream(stream_).has_value());
  const Bytes total(kSize * (kSources + kScratch));
  EXPECT_TRUE(memory_.Unmap(reservation_, Bytes(0), total).has_value());
  EXPECT_TRUE(memory_.Release(backing_).has_value());
  EXPECT_TRUE(memory_.Free(reservation_).has_value());
}

// The storage lane over io_uring waits in the kernel while a read is in
// flight. A read of an empty pipe never completes, so only a lane that is
// woken by the queued cancellation can act on it: the cancellation reaches
// the read, which drains as cancelled with proof of no further access.
// Skipped where there is no io_uring (qemu-user, a denying policy).
TEST(StorageLaneTest, ACancellationReachesAReadThatNeverCompletes) {
  auto storage = UringStorage::Create(4);
  if (!storage && (storage.error() == std::errc::function_not_supported ||
                   storage.error() == std::errc::operation_not_permitted)) {
    GTEST_SKIP() << "io_uring is unavailable: " << storage.error().message();
  }
  ASSERT_TRUE(storage.has_value()) << storage.error().message();
  std::array<int, 2> pipe_fds{-1, -1};
  ASSERT_EQ(::pipe2(pipe_fds.data(), O_CLOEXEC), 0);
  auto* memory = static_cast<std::byte*>(
      std::aligned_alloc(4096, 4096));  // NOLINT(cppcoreguidelines-no-malloc)
  llmp::base::WakeFlag wake;
  CompletionBoard board(4, wake);
  StorageService lane(**storage,
                      ReaderSettings{.alignment = 4096,
                                     .request_bytes = 4096,
                                     .retries = 0,
                                     .reads = 4,
                                     .waiters = 2,
                                     .span_bytes = llmp::providers::kNoCoalescing,
                                     .span_segments = llmp::providers::kMaxSegments},
                      board, QueueSettings{.capacity = 4, .reserved = 1, .batch = 4});
  // The test is the board's owner here.
  const OperationId read = board.Open();
  const auto observe = [&](const auto& done) {
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < give_up) {
      for (const auto& seen : board.Harvest(4)) {
        if (done(seen)) {
          return true;
        }
      }
      (void)wake.WaitFor(std::chrono::milliseconds(10));
    }
    return false;
  };
  bool cancelled = false;
  {
    std::jthread thread([&] { lane.Run(); });
    ASSERT_EQ(lane.Submit(ReadCommand{
                  .operation = read,
                  .spec = {.fd = pipe_fds[0], .offset = 0, .memory = memory, .length = 4096}}),
              PushResult::kAccepted);
    ASSERT_TRUE(observe([](const auto& seen) { return seen.acceptance == Acceptance::kAccepted; }));
    // Most likely waiting in the kernel by now; the cancellation must reach
    // the read either way.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ASSERT_EQ(lane.Submit(CancelRead{.operation = read}, llmp::base::PushKind::kCleanup),
              PushResult::kAccepted);
    cancelled = observe([](const auto& seen) {
      return seen.terminal && seen.terminal->outcome == Outcome::kCancelled &&
             seen.terminal->no_further_access;
    });
    if (!cancelled) {
      // Release the read so the lane can drain and the test end.
      const std::vector<std::byte> fill(4096);
      EXPECT_EQ(::write(pipe_fds[1], fill.data(), fill.size()), 4096);
    }
    lane.Close();
  }
  EXPECT_TRUE(cancelled) << "the storage lane never saw the cancellation";
  EXPECT_EQ((*storage)->in_flight(), 0U);
  EXPECT_TRUE(board.Close(read));
  std::free(memory);  // NOLINT(cppcoreguidelines-no-malloc)
  (void)::close(pipe_fds[0]);
  (void)::close(pipe_fds[1]);
}

// Threads ----------------------------------------------------------------------------

// On an idle Spark the cross build runs these in about 0.1 s each; a
// loaded workstation takes seconds. ThreadSanitizer runs fewer.
#ifdef LLMP_TEST_TSAN
constexpr int kPings = 500;  // per producer
constexpr int kStressTasks = 1000;
#else
constexpr int kPings = 4000;
constexpr int kStressTasks = 10000;
#endif

struct Tally {
  std::atomic<int> succeeded{0};
  std::atomic<int> cancelled{0};
  std::atomic<int> failed{0};
  std::atomic<int> retired{0};
  std::atomic<int> mismatches{0};
  std::atomic<int> evictions{0};
  void Count(TaskOutcome outcome) {
    switch (outcome) {
      case TaskOutcome::kSucceeded:
        succeeded.fetch_add(1);
        break;
      case TaskOutcome::kCancelled:
        cancelled.fetch_add(1);
        break;
      case TaskOutcome::kFailed:
        failed.fetch_add(1);
        break;
    }
  }
};

// One CPU job, then done.
class PingProgram final : public TaskProgram {
 public:
  explicit PingProgram(Tally& tally) : tally_(tally) {}
  Step Advance(TaskContext& context) override {
    if (!submitted_) {
      if (!context.SubmitCpu(
              Closure{}, [] { return CpuResult{.outcome = Outcome::kSucceeded, .bytes = 0}; })) {
        return Step::Yield();  // no mailbox now
      }
      submitted_ = true;
      return Step::Wait();
    }
    return Step::Finish(context.TakeFailure() ? TaskOutcome::kFailed : TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { tally_.Count(outcome); }
  void Retired() override { tally_.retired.fetch_add(1); }

 private:
  Tally& tally_;
  bool submitted_ = false;
};

// Producers publish controls and CPU workers publish completions while the
// owner sleeps with a timer tick longer than the test: a lost wakeup is a
// hang, not a slow pass.
TEST_F(SchedulerTest, TheOwnerLosesNoWakeupAmongManyPublishers) {
  Build({.capacity = 16, .reserved = 4, .batch = 16}, {.capacity = 16, .reserved = 4, .batch = 16},
        4, std::chrono::microseconds(0), std::chrono::hours(1));
  constexpr int kProducers = 4;
  std::array<Tally, kProducers> tallies;
  std::optional<std::expected<void, Fault>> result;
  const auto retired = [&] {
    int sum = 0;
    for (const Tally& tally : tallies) {
      sum += tally.retired.load();
    }
    return sum;
  };
  {
    std::jthread owner([&] { result = scheduler_->Run(); });
    {
      std::vector<std::jthread> producers;
      producers.reserve(kProducers);
      for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
          Tally& tally = tallies.at(static_cast<std::size_t>(p));
          for (int i = 0; i < kPings; ++i) {
            // Each producer's share of the task bound, so no start is refused.
            while (i - tally.retired.load() >= static_cast<int>(kScratch) / kProducers) {
              std::this_thread::yield();
            }
            Offer(*scheduler_,
                  StartRequest{.request = static_cast<std::uint64_t>((p * kPings) + i + 1),
                               .priority = static_cast<std::size_t>(i % 2),
                               .program = std::make_unique<PingProgram>(tally)});
          }
        });
      }
    }
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (retired() < kProducers * kPings && std::chrono::steady_clock::now() < give_up) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(retired(), kProducers * kPings);
    scheduler_->RequestShutdown();
  }
  EXPECT_EQ(FaultOf(result), std::nullopt);
  for (const Tally& tally : tallies) {
    EXPECT_EQ(tally.succeeded.load(), kPings);
    EXPECT_EQ(tally.failed.load(), 0);
  }
  EXPECT_GT(scheduler_->stats().sleeps, 0U);
}

// Runs until it is cancelled.
class UntilCancelled final : public TaskProgram {
 public:
  explicit UntilCancelled(Tally& tally) : tally_(tally) {}
  Step Advance(TaskContext& /*context*/) override { return Step::Yield(); }
  void Finished(TaskOutcome outcome) override { tally_.Count(outcome); }
  void Retired() override { tally_.retired.fetch_add(1); }

 private:
  Tally& tally_;
};

// Each producer uses one tag for all its starts. A round posts a
// cancellation before the starts (queued as an intent that nothing is live
// to meet), two starts, then a cancellation after both, which must reach
// both: it may not join the earlier intent, nor stop at one start. The
// producers' intents share one table and queue with the owner popping them.
// No other thread cancels a producer's tag, so a lost cancellation leaves a
// start running forever: the test reports a hang instead of passing.
TEST_F(SchedulerTest, ReusedTagsLoseNoCancellationAmongManyThreads) {
  Build({.capacity = 16, .reserved = 4, .batch = 16}, {.capacity = 16, .reserved = 4, .batch = 16},
        0, std::chrono::microseconds(0), std::chrono::hours(1));
  constexpr int kProducers = 4;  // two tasks each: the task bound
#ifdef LLMP_TEST_TSAN
  constexpr int kRounds = 300;
#else
  constexpr int kRounds = 3000;
#endif
  std::array<Tally, kProducers> tallies;
  std::optional<std::expected<void, Fault>> result;
  std::atomic<bool> hung{false};
  {
    std::jthread owner([&] { result = scheduler_->Run(); });
    {
      std::vector<std::jthread> producers;
      producers.reserve(kProducers);
      for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
          Tally& tally = tallies.at(static_cast<std::size_t>(p));
          const std::uint64_t tag = static_cast<std::uint64_t>(p) + 1;
          for (int round = 0; round < kRounds && !hung.load(); ++round) {
            Offer(*scheduler_, CancelRequest{.request = tag});  // meets nothing
            for (int i = 0; i < 2; ++i) {
              Offer(*scheduler_, StartRequest{.request = tag,
                                              .priority = static_cast<std::size_t>(i),
                                              .program = std::make_unique<UntilCancelled>(tally)});
            }
            Offer(*scheduler_, CancelRequest{.request = tag});
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (tally.retired.load() < 2 * (round + 1)) {
              if (std::chrono::steady_clock::now() > give_up) {
                hung.store(true);
                return;
              }
              std::this_thread::yield();
            }
          }
        });
      }
    }
    scheduler_->RequestShutdown();  // also ends any start a lost cancellation left running
  }
  EXPECT_FALSE(hung.load()) << "a start outlived a cancellation posted after it";
  EXPECT_EQ(FaultOf(result), std::nullopt);
  for (const Tally& tally : tallies) {
    EXPECT_EQ(tally.failed.load(), 0);  // never refused: within the task bound
    EXPECT_EQ(tally.succeeded.load(), 0);
    EXPECT_EQ(tally.cancelled.load(), tally.retired.load());
  }
  EXPECT_EQ(scheduler_->tasks().size(), 0U);
}

// Loads a source (coalescing with others that want it), copies it into its
// own scratch slot, has a CPU worker check the copy against the file, and
// evicts the source so later tasks read it again.
class StressProgram final : public TaskProgram {
 public:
  StressProgram(Tally& tally, ExtentId source, std::span<const ExtentId> scratch,
                std::uint64_t from, std::uint64_t scratch_base, std::span<const std::byte> expected)
      : tally_(tally),
        source_(source),
        scratch_(scratch),
        from_(from),
        scratch_base_(scratch_base),
        expected_(expected) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    const std::size_t slot = context.task().index();
    const std::uint64_t to = scratch_base_ + (slot * kSize);
    switch (phase_) {
      case Phase::kLoad: {
        const std::array<ExtentId, 1> load{source_};
        const auto ready = context.Materialize(context.catalog().ClosureOfExtents(load).value());
        if (!ready) {
          return ready.error() == WorkError::kBusy ? Step::Yield()
                                                   : Step::Finish(TaskOutcome::kFailed);
        }
        if (*ready == Readiness::kWaiting) {
          return Step::Wait();
        }
        const std::array<ExtentId, 2> use{source_, scratch_[slot]};
        DeviceWork work;
        work.copies.at(0) = {.destination = to, .source = from_, .size = Bytes(kSize)};
        work.count = 1;
        const auto copy =
            context.SubmitDevice(context.catalog().ClosureOfExtents(use).value(), work);
        if (!copy) {
          // Evicted since it was ready, or no mailbox: try again.
          return copy.error() == WorkError::kBusy || copy.error() == WorkError::kNotResident
                     ? Step::Yield()
                     : Step::Finish(TaskOutcome::kFailed);
        }
        phase_ = Phase::kVerify;
        return Step::Wait();
      }
      case Phase::kVerify: {
        const std::array<ExtentId, 1> check{scratch_[slot]};
        Tally& tally = tally_;
        const std::span<const std::byte> expected = expected_;
        const auto verify = context.SubmitCpu(
            context.catalog().ClosureOfExtents(check).value(), [&tally, expected, to] {
              const bool same = std::memcmp(At(to), expected.data(), kSize) == 0;
              if (!same) {
                tally.mismatches.fetch_add(1);
              }
              return CpuResult{.outcome = same ? Outcome::kSucceeded : Outcome::kFailed,
                               .bytes = kSize};
            });
        if (!verify) {
          return verify.error() == WorkError::kBusy ? Step::Yield()
                                                    : Step::Finish(TaskOutcome::kFailed);
        }
        phase_ = Phase::kDone;
        return Step::Wait();
      }
      case Phase::kDone:
        break;
    }
    if (context.Evict(source_).has_value()) {
      tally_.evictions.fetch_add(1);  // otherwise held or loading elsewhere: it stays
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { tally_.Count(outcome); }
  void Retired() override { tally_.retired.fetch_add(1); }

 private:
  enum class Phase : std::uint8_t { kLoad, kVerify, kDone };
  Tally& tally_;
  ExtentId source_;
  std::span<const ExtentId> scratch_;
  std::uint64_t from_;
  std::uint64_t scratch_base_;
  std::span<const std::byte> expected_;
  Phase phase_ = Phase::kLoad;
};

// Every lane on its own thread, a fake device stepping on another, and
// requests started and cancelled from two more. The storage lane writes a
// source, a device thread copies it, and a CPU worker reads the copy, each
// ordered only through the board, the lanes and the owner's leases: a
// missing edge shows as a mismatch here, a race under ThreadSanitizer, or
// a lease released early as a source evicted and reloaded under a copy.
TEST_F(SchedulerTest, ThreadedLanesKeepEveryHandoffOrdered) {
  // A tick longer than the test here too: only signals wake the owner.
  Build({.capacity = 16, .reserved = 4, .batch = 16}, {.capacity = 16, .reserved = 4, .batch = 16},
        2, std::chrono::microseconds(200), std::chrono::hours(1));
  Tally tally;
  std::optional<std::expected<void, Fault>> result;
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
    std::atomic<int> started{0};
    std::jthread canceller([&](const std::stop_token& stop) {
      std::uint64_t victim = 1;
      while (!stop.stop_requested()) {
        const int now = started.load();
        if (now > 3 && victim + 2 < static_cast<std::uint64_t>(now)) {
          victim += 3;  // every third request or so, while it may still be live
          Control cancel = CancelRequest{.request = victim};
          (void)scheduler_->Post(std::move(cancel));
        }
        std::this_thread::yield();
      }
    });
    for (int n = 0; n < kStressTasks; ++n) {
      while (n - tally.retired.load() >= static_cast<int>(kScratch)) {
        std::this_thread::yield();
      }
      const std::size_t source = static_cast<std::size_t>(n) % kSources;
      Offer(*scheduler_,
            StartRequest{.request = static_cast<std::uint64_t>(n + 1),
                         .priority = static_cast<std::size_t>(n % 2),
                         .program = std::make_unique<StressProgram>(
                             tally, sources_.at(source), scratch_, Source(source), Scratch(0),
                             std::span<const std::byte>(file_).subspan(source * kSize, kSize))});
      started.store(n + 1);
    }
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (tally.retired.load() < kStressTasks && std::chrono::steady_clock::now() < give_up) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(tally.retired.load(), kStressTasks);
    canceller.request_stop();
    canceller.join();
    scheduler_->RequestShutdown();
    owner.join();
    storage_lane_->Close();
    device_lane_->Close();
    storage.join();
    submission.join();
    completion.join();
  }
  EXPECT_EQ(FaultOf(result), std::nullopt);
  EXPECT_EQ(tally.mismatches.load(), 0);
  EXPECT_EQ(tally.failed.load(), 0);
  EXPECT_EQ(tally.succeeded.load() + tally.cancelled.load(), kStressTasks);
  EXPECT_GT(tally.succeeded.load(), 0);
  EXPECT_GT(tally.evictions.load(), 0);  // so sources were read again
  const Occupancy occupied = Occupied();
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.evicting, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(storage_.in_flight(), 0U);
  EXPECT_EQ(execution_.fences(), 0U);
  for (const ExtentId source : sources_) {
    EXPECT_EQ(View(source).leases, 0U);
  }
}

}  // namespace
