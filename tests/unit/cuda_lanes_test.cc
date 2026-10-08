// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The scheduler thread and its lanes over the real providers on a Spark
// (label `gpu`), and on a discrete GPU the build targets (`gpu-discrete`,
// D-082) (D-034, D-048): io_uring reads a direct-I/O file into CUDA
// host VMM, the device lanes copy it to device memory and back on a CUDA
// stream, observed only by fence queries, a CPU worker checks the result,
// and everything retires. Cancellation while a read is in flight drains
// the read before its backing is released; cancellation while device work
// is in flight keeps the work's lease until its fence completes. Every
// lane runs on its own thread, as a program wires them.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/uring_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/lane.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace {

using llmp::base::Bytes;
using llmp::base::PushResult;
using llmp::catalog::Closure;
using llmp::catalog::ExtentId;
using llmp::catalog::ExtentState;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::StreamId;
using llmp::scheduler::CancelRequest;
using llmp::scheduler::CompletionBoard;
using llmp::scheduler::Control;
using llmp::scheduler::CpuCommand;
using llmp::scheduler::CpuResult;
using llmp::scheduler::DeviceService;
using llmp::scheduler::DeviceSettings;
using llmp::scheduler::DeviceWork;
using llmp::scheduler::Fault;
using llmp::scheduler::Lane;
using llmp::scheduler::Outcome;
using llmp::scheduler::QueueSettings;
using llmp::scheduler::Readiness;
using llmp::scheduler::Scheduler;
using llmp::scheduler::SchedulerSettings;
using llmp::scheduler::StartRequest;
using llmp::scheduler::Step;
using llmp::scheduler::StorageService;
using llmp::scheduler::TaskContext;
using llmp::scheduler::TaskOutcome;
using llmp::scheduler::TaskProgram;

// Large enough that a cancellation posted once the read is published finds
// requests still in flight: 32 requests of 2 MiB, four at a time.
constexpr std::uint64_t kSize = 64ULL << 20U;
constexpr auto kPatience = std::chrono::seconds(120);

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

// What a task tells the test thread.
struct Signals {
  std::atomic<bool> waiting{false};    // its page-in is published
  std::atomic<bool> submitted{false};  // its device work is published
  std::atomic<int> outcome{-1};        // TaskOutcome, once finished
  std::atomic<bool> retired{false};
  std::atomic<int> state{-1};  // an extent's state, for a probe
};

// Loads the source, copies it to the device and back, and has a CPU
// worker compare the copy with the file.
class RoundTrip final : public TaskProgram {
 public:
  RoundTrip(Signals& signals, Closure load, Closure use, Closure check, DeviceWork work,
            std::uint64_t result, std::span<const std::byte> expected)
      : signals_(signals),
        load_(std::move(load)),
        use_(std::move(use)),
        check_(std::move(check)),
        work_(work),
        result_(result),
        expected_(expected) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    switch (phase_) {
      case 0: {
        const auto ready = context.Materialize(load_);
        if (!ready) {
          return Step::Finish(TaskOutcome::kFailed);
        }
        if (*ready == Readiness::kWaiting) {
          signals_.waiting.store(true);
          return Step::Wait();
        }
        if (!context.SubmitDevice(use_, work_)) {
          return Step::Finish(TaskOutcome::kFailed);
        }
        signals_.submitted.store(true);
        phase_ = 1;
        return Step::Wait();
      }
      case 1: {
        const std::uint64_t result = result_;
        const std::span<const std::byte> expected = expected_;
        const auto verify = context.SubmitCpu(check_, [result, expected] {
          const bool same = std::memcmp(At(result), expected.data(), expected.size()) == 0;
          return CpuResult{.outcome = same ? Outcome::kSucceeded : Outcome::kFailed,
                           .bytes = expected.size()};
        });
        if (!verify) {
          return Step::Finish(TaskOutcome::kFailed);
        }
        phase_ = 2;
        return Step::Wait();
      }
      default:
        return Step::Finish(TaskOutcome::kSucceeded);
    }
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  Closure load_;
  Closure use_;
  Closure check_;
  DeviceWork work_;
  std::uint64_t result_;
  std::span<const std::byte> expected_;
  int phase_ = 0;
};

// Reports an extent's state once no load is in flight for it, and evicts
// it if it is resident, so the next read starts from the file again.
class Probe final : public TaskProgram {
 public:
  Probe(Signals& signals, ExtentId extent) : signals_(signals), extent_(extent) {}
  Step Advance(TaskContext& context) override {
    const auto view = context.catalog().Describe(extent_);
    if (!view || view->state == ExtentState::kLoading) {
      return Step::Yield();  // the cancelled read is still draining
    }
    signals_.state.store(static_cast<int>(view->state));
    if (view->state == ExtentState::kResident && !context.Evict(extent_).has_value()) {
      return Step::Finish(TaskOutcome::kFailed);  // nothing else holds it
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  ExtentId extent_;
};

bool WaitFor(const std::atomic<bool>& flag) {
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (!flag.load() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return flag.load();
}

class CudaLanes : public ::testing::Test {
 protected:
  void SetUp() override {
    auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
    ASSERT_TRUE(memory.has_value()) << (memory ? "" : memory.error().detail);
    memory_ = std::move(*memory);
    auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(execution.has_value()) << (execution ? "" : execution.error().detail);
    execution_ = std::move(*execution);
    auto storage = llmp::providers::UringStorage::Create(8);
    ASSERT_TRUE(storage.has_value()) << storage.error().message();
    storage_ = std::move(*storage);

    // The file, written through aligned memory.
    const char* scratch = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path directory = scratch != nullptr
                                                ? std::filesystem::path(scratch)
                                                : std::filesystem::path(::testing::TempDir());
    std::filesystem::create_directories(directory);
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0);
    file_.resize(kSize);
    for (std::uint64_t i = 0; i < kSize; ++i) {
      file_[i] = static_cast<std::byte>((i * 29) + (i >> 21) + 3);
    }
    auto* staging = static_cast<std::byte*>(
        std::aligned_alloc(4096, kSize));  // NOLINT(cppcoreguidelines-no-malloc)
    std::memcpy(staging, file_.data(), kSize);
    const ssize_t written = ::pwrite(fd_, staging, kSize, 0);
    std::free(staging);  // NOLINT(cppcoreguidelines-no-malloc)
    ASSERT_EQ(written, static_cast<ssize_t>(kSize));

    // Host source, device copy, host result: one reservation.
    reservation_ = memory_->Reserve(Bytes(kSize * 3)).value();
    for (std::size_t i = 0; i < 3; ++i) {
      const BackingKind kind = i == 1 ? BackingKind::kDevice : BackingKind::kHost;
      backings_.at(i) = memory_->Create(ClassOf(kind), Bytes(kSize)).value();
      ASSERT_TRUE(memory_->Map(reservation_, Bytes(kSize * i), backings_.at(i)).has_value());
    }
    ASSERT_TRUE(memory_->SetAccess(reservation_, Bytes(0), Bytes(kSize * 3), Access::kReadWrite)
                    .has_value());
    base_ = memory_->RangeOf(reservation_).value().base;
    std::memset(At(base_ + (2 * kSize)), 0, kSize);

    domain_ = catalog_.AddDomain("gb10");
    source_ = catalog_
                  .AddExtent({.domain = domain_,
                              .memory_class = llmp::catalog::MemoryClass::kWeights,
                              .recovery = llmp::catalog::Recovery::kFromArtifact,
                              .size = Bytes(kSize),
                              .content = {}})
                  .value();
    for (std::size_t i = 0; i < 2; ++i) {
      scratch_.at(i) = catalog_
                           .AddExtent({.domain = domain_,
                                       .memory_class = llmp::catalog::MemoryClass::kScratch,
                                       .recovery = llmp::catalog::Recovery::kDiscardable,
                                       .size = Bytes(kSize),
                                       .content = {}},
                                      true)
                           .value();
    }
    stream_ = execution_->CreateStream().value();

    storage_lane_ = std::make_unique<StorageService>(
        *storage_,
        llmp::providers::ReaderSettings{.alignment = 4096,
                                        .request_bytes = 2U << 20U,
                                        .retries = 3,
                                        .reads = 16,
                                        .waiters = 8,
                                        .span_bytes = llmp::providers::kNoCoalescing,
                                        .span_segments = llmp::providers::kMaxSegments},
        board_, QueueSettings{.capacity = 16, .reserved = 4, .batch = 16});
    device_lane_ = std::make_unique<DeviceService>(
        *execution_, std::span<const StreamId>(&stream_, 1), board_,
        DeviceSettings{.queue = {.capacity = 16, .reserved = 4, .batch = 16}, .handoff = 16});
    cpu_lane_ = std::make_unique<Lane<CpuCommand>>(
        llmp::scheduler::LaneSettings{.name = "cpu", .capacity = 16, .reserved = 4, .workers = 1},
        llmp::scheduler::CpuHandler(board_));
    scheduler_ = std::make_unique<Scheduler>(
        catalog_, board_, wake_,
        llmp::scheduler::Lanes{
            .storage = storage_lane_.get(), .device = device_lane_.get(), .cpu = cpu_lane_.get()},
        SchedulerSettings{.tasks = 8, .budget = Bytes(kSize * 3)});
    scheduler_->SetSource(
        source_,
        llmp::providers::ReadSpec{.fd = fd_, .offset = 0, .memory = At(base_), .length = kSize});
    threads_.emplace_back([this] { result_ = scheduler_->Run(); });
    threads_.emplace_back([this] { storage_lane_->Run(); });
    threads_.emplace_back([this] { device_lane_->RunSubmission(); });
    threads_.emplace_back([this] { device_lane_->RunCompletion(); });
  }

  void TearDown() override {
    if (scheduler_ != nullptr && !threads_.empty()) {
      Stop();
    }
    if (reservation_.valid()) {
      EXPECT_TRUE(memory_->Unmap(reservation_, Bytes(0), Bytes(kSize * 3)).has_value());
      for (const auto backing : backings_) {
        EXPECT_TRUE(memory_->Release(backing).has_value());
      }
      EXPECT_TRUE(memory_->Free(reservation_).has_value());
    }
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  // Shutdown in the program's order: the scheduler drains, then the lanes
  // close and drain, then the stream goes (backing goes in TearDown).
  void Stop() {
    scheduler_->RequestShutdown();
    threads_.front().join();
    storage_lane_->Close();
    device_lane_->Close();
    threads_.clear();  // joins the lanes
    cpu_lane_.reset();
    EXPECT_EQ(storage_->in_flight(), 0U);
    EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
  }

  std::size_t ClassOf(BackingKind kind) const {
    for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
      if (memory_->Classes()[i].kind == kind) {
        return i;
      }
    }
    ADD_FAILURE() << "no allocation class of that kind";
    return 0;
  }
  Closure Of(std::initializer_list<ExtentId> extents) const {
    return catalog_.ClosureOfExtents(std::span<const ExtentId>(extents.begin(), extents.size()))
        .value();
  }
  std::unique_ptr<TaskProgram> Trip(Signals& signals) const {
    DeviceWork work;
    work.copies.at(0) = {.destination = base_ + kSize, .source = base_, .size = Bytes(kSize)};
    work.copies.at(1) = {
        .destination = base_ + (2 * kSize), .source = base_ + kSize, .size = Bytes(kSize)};
    work.count = 2;
    return std::make_unique<RoundTrip>(signals, Of({source_}),
                                       Of({source_, scratch_[0], scratch_[1]}), Of({scratch_[1]}),
                                       work, base_ + (2 * kSize), file_);
  }
  void Post(Control&& control) {
    // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
    while (scheduler_->Post(std::move(control)) == PushResult::kFull) {
      std::this_thread::yield();
    }
  }

  std::unique_ptr<llmp::providers::VmmProvider> memory_;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  std::unique_ptr<llmp::providers::UringStorage> storage_;
  llmp::catalog::Catalog catalog_;
  llmp::base::WakeFlag wake_;
  CompletionBoard board_{16, wake_};
  std::unique_ptr<StorageService> storage_lane_;
  std::unique_ptr<DeviceService> device_lane_;
  std::unique_ptr<Lane<CpuCommand>> cpu_lane_;
  std::unique_ptr<Scheduler> scheduler_;
  std::optional<std::expected<void, Fault>> result_;
  std::vector<std::jthread> threads_;  // the scheduler first

  int fd_ = -1;
  std::vector<std::byte> file_;
  llmp::providers::ReservationId reservation_;
  std::array<llmp::providers::BackingId, 3> backings_;
  std::uint64_t base_ = 0;
  llmp::catalog::DomainId domain_;
  ExtentId source_;
  std::array<ExtentId, 2> scratch_;
  StreamId stream_;
};

TEST_F(CudaLanes, AReadFeedsDeviceWorkAndEverythingRetires) {
  Signals signals;
  Post(StartRequest{.request = 1, .priority = 1, .program = Trip(signals)});
  ASSERT_TRUE(WaitFor(signals.retired));
  Stop();
  ASSERT_TRUE(result_.has_value());
  EXPECT_TRUE(result_.value_or(std::unexpected(Fault::kUnproven)).has_value());
  EXPECT_EQ(signals.outcome.load(), static_cast<int>(TaskOutcome::kSucceeded));
  const auto source = catalog_.Describe(source_).value();
  EXPECT_EQ(source.state, ExtentState::kResident);
  EXPECT_EQ(source.leases, 0U);
  const auto occupied = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(scheduler_->operations(), 0U);
  EXPECT_EQ(std::memcmp(At(base_ + (2 * kSize)), file_.data(), kSize), 0);
}

// A cancellation posted as soon as the read is published reaches the storage
// lane while requests are in flight: the provider cancels what it can, the
// read drains, and only then is the extent's backing released. Timing
// decides whether a given attempt lands in flight, so it tries a few times;
// every attempt must end with nothing loading, held or quarantined.
TEST_F(CudaLanes, CancellingInFlightDrainsBeforeRetiring) {
  int in_flight = 0;
  for (int attempt = 0; attempt < 10 && in_flight == 0; ++attempt) {
    Signals trip;
    const std::uint64_t request = 100 + static_cast<std::uint64_t>(attempt);
    Post(StartRequest{.request = request, .priority = 1, .program = Trip(trip)});
    ASSERT_TRUE(WaitFor(trip.waiting));
    Post(CancelRequest{.request = request});
    ASSERT_TRUE(WaitFor(trip.retired));
    // A test thread descheduled for the whole trip lets it succeed first.
    const int outcome = trip.outcome.load();
    EXPECT_TRUE(outcome == static_cast<int>(TaskOutcome::kCancelled) ||
                outcome == static_cast<int>(TaskOutcome::kSucceeded))
        << "outcome " << outcome;
    Signals probe;
    Post(StartRequest{.request = 200 + request,
                      .priority = 1,
                      .program = std::make_unique<Probe>(probe, source_)});
    ASSERT_TRUE(WaitFor(probe.retired));
    const int state = probe.state.load();
    EXPECT_TRUE(state == static_cast<int>(ExtentState::kNonresident) ||
                state == static_cast<int>(ExtentState::kResident))
        << "state " << state;
    in_flight += state == static_cast<int>(ExtentState::kNonresident) ? 1 : 0;
  }
  EXPECT_GT(in_flight, 0) << "no cancellation reached a read in flight";

  // Device work cannot be recalled: cancelled once published, the task
  // finishes at once and retires only after its fence completes. The copy
  // and the check can both end before the cancellation arrives on a loaded
  // host, so it tries a few times, as above.
  int cancelled = 0;
  for (int attempt = 0; attempt < 10 && cancelled == 0; ++attempt) {
    Signals trip;
    const std::uint64_t request = 300 + static_cast<std::uint64_t>(attempt);
    Post(StartRequest{.request = request, .priority = 1, .program = Trip(trip)});
    ASSERT_TRUE(WaitFor(trip.submitted));
    Post(CancelRequest{.request = request});
    ASSERT_TRUE(WaitFor(trip.retired));
    const int outcome = trip.outcome.load();
    EXPECT_TRUE(outcome == static_cast<int>(TaskOutcome::kCancelled) ||
                outcome == static_cast<int>(TaskOutcome::kSucceeded))
        << "outcome " << outcome;
    cancelled += outcome == static_cast<int>(TaskOutcome::kCancelled) ? 1 : 0;
  }
  EXPECT_GT(cancelled, 0) << "no cancellation reached device work in flight";
  Stop();
  ASSERT_TRUE(result_.has_value());
  EXPECT_TRUE(result_.value_or(std::unexpected(Fault::kUnproven)).has_value());
  const auto occupied = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  for (const ExtentId extent : {source_, scratch_[0], scratch_[1]}) {
    EXPECT_EQ(catalog_.Describe(extent).value().leases, 0U);
  }
  EXPECT_EQ(scheduler_->operations(), 0U);
  // The copy ran to completion although its task was cancelled.
  EXPECT_EQ(std::memcmp(At(base_ + (2 * kSize)), file_.data(), kSize), 0);
}

}  // namespace
