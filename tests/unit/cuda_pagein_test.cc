// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The D-081 page-in path over the real providers on a Spark (label `gpu`)
// and on a discrete GPU the build targets (`gpu-discrete`, D-082):
// io_uring reads a direct-I/O file into a host-VMM landing zone, the device
// lane (or, in the second instantiation, a VMM lane of its own) creates and
// maps device-VMM backing for each extent, the device lane copies the
// landed bytes in on a CUDA stream, and the scheduler publishes each extent
// only once the copy's fence has completed. Device VMM is not CPU-mapped,
// so the test checks each extent by copying it back to host VMM under a
// lease. Evictions release the backing (D-033) and reloads, at the same
// place or relocated, restore the same bytes; a request cancelled with
// loads in flight drains them. Live state is written back through the
// zone to a spill file and restored exactly (CudaWriteBack, `gpu` only so
// far). Every lane runs on its own thread.

#include <cuda.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
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
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace {

using jitllm::base::Bytes;
using jitllm::base::PushResult;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;
using jitllm::providers::Access;
using jitllm::providers::BackingKind;
using jitllm::providers::ReservationId;
using jitllm::providers::StreamId;
using jitllm::scheduler::BackingPlace;
using jitllm::scheduler::CompletionBoard;
using jitllm::scheduler::Control;
using jitllm::scheduler::DeviceService;
using jitllm::scheduler::DeviceSettings;
using jitllm::scheduler::DeviceWork;
using jitllm::scheduler::Fault;
using jitllm::scheduler::LandingZone;
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

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::size_t kExtents = 32;
constexpr std::size_t kSlots = 8;                          // 2 x depth 4 (D-081)
constexpr std::uint64_t kTail = kExtent - (64ULL * 1024);  // a group's last chunk is shorter
constexpr auto kPatience = std::chrono::seconds(120);

std::byte* At(std::uint64_t address) {
  return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t LengthOf(std::size_t i) { return i + 1 == kExtents ? kTail : kExtent; }

struct Signals {
  std::atomic<int> outcome{-1};
  std::atomic<bool> retired{false};
  std::atomic<bool> waiting{false};
  std::atomic<int> mismatches{0};
};

// Materializes the extents; with `check`, copies each back to the host
// result extent under a lease and compares it with the file: extent k
// with chunk `chunks[k]`, or chunk k if none are given.
class LoadAndCheck final : public TaskProgram {
 public:
  LoadAndCheck(Signals& signals, const jitllm::catalog::Catalog& catalog,
               std::vector<ExtentId> extents, std::vector<std::uint64_t> addresses, ExtentId result,
               std::uint64_t result_address, std::span<const std::byte> file, bool check,
               std::vector<std::size_t> chunks = {})
      : signals_(signals),
        catalog_(catalog),
        extents_(std::move(extents)),
        addresses_(std::move(addresses)),
        result_(result),
        result_address_(result_address),
        file_(file),
        check_(check),
        chunks_(std::move(chunks)) {}

  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    if (!loaded_) {
      const auto ready = context.Materialize(catalog_.ClosureOfExtents(extents_).value());
      if (!ready) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*ready == Readiness::kWaiting) {
        signals_.waiting.store(true);
        return Step::Wait();
      }
      loaded_ = true;
    }
    if (!check_) {
      return Step::Finish(TaskOutcome::kSucceeded);
    }
    if (next_ > 0) {
      // The previous copy's fence completed: compare what it brought back.
      const std::size_t i = chunks_.empty() ? next_ - 1 : chunks_.at(next_ - 1);
      if (std::memcmp(At(result_address_), file_.data() + (i * kExtent), LengthOf(i)) != 0) {
        signals_.mismatches.fetch_add(1);
      }
    }
    if (next_ == extents_.size()) {
      return Step::Finish(TaskOutcome::kSucceeded);
    }
    DeviceWork work;
    work.copies.at(0) = {.destination = result_address_,
                         .source = addresses_.at(next_),
                         .size = Bytes(LengthOf(chunks_.empty() ? next_ : chunks_.at(next_)))};
    work.count = 1;
    const std::array<ExtentId, 2> both = {extents_.at(next_), result_};
    if (!context.SubmitDevice(catalog_.ClosureOfExtents(both).value(), work)) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    ++next_;
    return Step::Wait();
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  const jitllm::catalog::Catalog& catalog_;
  std::vector<ExtentId> extents_;
  std::vector<std::uint64_t> addresses_;
  ExtentId result_;
  std::uint64_t result_address_;
  std::span<const std::byte> file_;
  bool check_;
  std::vector<std::size_t> chunks_;
  bool loaded_ = false;
  std::size_t next_ = 0;
};

class EvictAll final : public TaskProgram {
 public:
  EvictAll(Signals& signals, std::vector<ExtentId> extents)
      : signals_(signals), extents_(std::move(extents)) {}
  Step Advance(TaskContext& context) override {
    if (context.TakeFailure()) {
      return Step::Finish(TaskOutcome::kFailed);
    }
    while (next_ < extents_.size()) {
      const ExtentId extent = extents_[next_++];
      if (context.catalog().Describe(extent).value().state != ExtentState::kResident) {
        continue;
      }
      const auto evicted = context.Evict(extent);
      if (!evicted) {
        return Step::Finish(TaskOutcome::kFailed);
      }
      if (*evicted == Readiness::kWaiting) {
        return Step::Wait();
      }
    }
    return Step::Finish(TaskOutcome::kSucceeded);
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  std::vector<ExtentId> extents_;
  std::size_t next_ = 0;
};

// Runs `probe` on the scheduler thread each turn until it returns true.
// The scheduler's and the catalog's records are that thread's while it
// runs: the test reads them only here, or once a task that ran after the
// change has retired.
class OnOwner final : public TaskProgram {
 public:
  OnOwner(Signals& signals, std::function<bool()> probe)
      : signals_(signals), probe_(std::move(probe)) {}
  Step Advance(TaskContext& /*context*/) override {
    return probe_() ? Step::Finish(TaskOutcome::kSucceeded) : Step::Yield();
  }
  void Finished(TaskOutcome outcome) override { signals_.outcome.store(static_cast<int>(outcome)); }
  void Retired() override { signals_.retired.store(true); }

 private:
  Signals& signals_;
  std::function<bool()> probe_;
};

bool WaitFor(const std::atomic<bool>& flag) {
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (!flag.load() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return flag.load();
}

// The parameter: whether VMM work runs on a VMM lane (BackingService).
class CudaPageIn : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    auto memory = jitllm::providers::cuda::OpenDeviceMemory(0);
    ASSERT_TRUE(memory.has_value()) << (memory ? "" : memory.error().detail);
    memory_ = std::move(*memory);
    auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(execution.has_value()) << (execution ? "" : execution.error().detail);
    execution_ = std::move(*execution);
    auto storage = jitllm::providers::UringStorage::Create(4);
    ASSERT_TRUE(storage.has_value()) << storage.error().message();
    storage_ = std::move(*storage);

    const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path directory = scratch != nullptr
                                                ? std::filesystem::path(scratch)
                                                : std::filesystem::path(::testing::TempDir());
    std::filesystem::create_directories(directory);
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0);
    file_.resize(kExtent * kExtents);
    for (std::uint64_t i = 0; i < file_.size(); ++i) {
      file_[i] = static_cast<std::byte>((i * 29) + (i >> 21) + 3);
    }
    auto* staging = static_cast<std::byte*>(
        std::aligned_alloc(4096, file_.size()));  // NOLINT(cppcoreguidelines-no-malloc)
    std::memcpy(staging, file_.data(), file_.size());
    const ssize_t written = ::pwrite(fd_, staging, file_.size(), 0);
    std::free(staging);  // NOLINT(cppcoreguidelines-no-malloc)
    ASSERT_EQ(written, static_cast<ssize_t>(file_.size()));

    // The zone and the check's result: host VMM, persistent.
    host_ = memory_->Reserve(Bytes(kExtent * (kSlots + 1))).value();
    host_backing_ =
        memory_->Create(ClassOf(BackingKind::kHost), Bytes(kExtent * (kSlots + 1))).value();
    ASSERT_TRUE(memory_->Map(host_, Bytes(0), host_backing_).has_value());
    ASSERT_TRUE(
        memory_->SetAccess(host_, Bytes(0), Bytes(kExtent * (kSlots + 1)), Access::kReadWrite)
            .has_value());
    host_base_ = memory_->RangeOf(host_).value().base;
    // Two places for the weights: address space until a load maps backing.
    for (ReservationId& place : places_) {
      place = memory_->Reserve(Bytes(kExtent * kExtents)).value();
    }
    baseline_ = memory_->backings();

    domain_ = catalog_.AddDomain("gb10");
    for (std::size_t i = 0; i < kExtents; ++i) {
      extents_.push_back(
          catalog_
              .AddExtent(
                  {.domain = domain_,
                   .memory_class = jitllm::catalog::MemoryClass::kWeights,
                   .recovery = jitllm::catalog::Recovery::kFromArtifact,
                   .size = Bytes(kExtent),
                   .content = {.artifact = {}, .group = 0, .chunk = static_cast<std::uint32_t>(i)}})
              .value());
    }
    // The zone and result, pinned: the catalog counts them (a declared pool).
    for (std::size_t i = 0; i <= kSlots; ++i) {
      const ExtentId pinned =
          catalog_
              .AddExtent({.domain = domain_,
                          .memory_class = jitllm::catalog::MemoryClass::kStaging,
                          .recovery = jitllm::catalog::Recovery::kPinned,
                          .size = Bytes(kExtent),
                          .content = {}},
                         true)
              .value();
      if (i == kSlots) {
        result_ = pinned;
      }
    }
    stream_ = execution_->CreateStream().value();
    WrapProviders();  // what the lanes call: the providers, or a test's decorators over them

    storage_lane_ = std::make_unique<StorageService>(
        lane_storage_ != nullptr ? *lane_storage_ : *storage_,
        // The reader's default (no coalescing) unless a test turns it on.
        jitllm::providers::ReaderSettings{.alignment = 4096,
                                          .request_bytes = 2U << 20U,
                                          .retries = 3,
                                          .reads = 64,
                                          .waiters = 8,
                                          .span_bytes = span_bytes_,
                                          .span_segments = jitllm::providers::kMaxSegments},
        board_, QueueSettings{.capacity = 64, .reserved = 8, .batch = 16});
    device_lane_ = std::make_unique<DeviceService>(
        lane_execution_ != nullptr ? *lane_execution_ : *execution_,
        std::span<const StreamId>(&stream_, 1), board_,
        DeviceSettings{.queue = {.capacity = 64, .reserved = 8, .batch = 16}, .handoff = 64},
        GetParam() ? nullptr : memory_.get());  // one lane calls the provider
    if (GetParam()) {
      backing_lane_ = std::make_unique<jitllm::scheduler::BackingService>(
          memory_.get(), board_, QueueSettings{.capacity = 64, .reserved = 8, .batch = 16});
    }
    LandingZone landing{.slots = {}, .slot_bytes = Bytes(kExtent), .stream = 0};
    for (std::size_t i = 0; i < kSlots; ++i) {
      landing.slots.push_back(host_base_ + (i * kExtent));
    }
    scheduler_ = std::make_unique<Scheduler>(
        catalog_, board_, wake_,
        jitllm::scheduler::Lanes{.storage = storage_lane_.get(),
                                 .device = device_lane_.get(),
                                 .cpu = nullptr,
                                 .backing = backing_lane_.get()},
        SchedulerSettings{
            .tasks = 8, .budget = Bytes(kExtent * (kExtents + kSlots + 1)), .landing = landing});
    Place(0);
    BeforeLanes();
    threads_.emplace_back([this] { result_status_ = scheduler_->Run(); });
    threads_.emplace_back([this] { storage_lane_->Run(); });
    threads_.emplace_back([this] { device_lane_->RunSubmission(); });
    threads_.emplace_back([this] { device_lane_->RunCompletion(); });
    if (backing_lane_ != nullptr) {
      threads_.emplace_back([this] { backing_lane_->Run(); });
    }
  }

  void TearDown() override {
    if (scheduler_ != nullptr && !threads_.empty()) {
      Signals evicted;
      Post(StartRequest{
          .request = 999, .priority = 1, .program = std::make_unique<EvictAll>(evicted, extents_)});
      EXPECT_TRUE(WaitFor(evicted.retired));
      Stop();
    }
    EXPECT_EQ(memory_->backings(), baseline_);  // every extent's backing released
    if (host_.valid()) {
      EXPECT_TRUE(memory_->Unmap(host_, Bytes(0), Bytes(kExtent * (kSlots + 1))).has_value());
      EXPECT_TRUE(memory_->Release(host_backing_).has_value());
      EXPECT_TRUE(memory_->Free(host_).has_value());
    }
    for (const ReservationId place : places_) {
      EXPECT_TRUE(memory_->Free(place).has_value());
    }
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
  }

  // Runs before the lanes' threads start, while the test's thread is still
  // the providers' only caller.
  virtual void BeforeLanes() {}
  // Runs before the lanes are built: may set lane_storage_ and
  // lane_execution_ to decorators over the real providers.
  virtual void WrapProviders() {}

  void Stop() {
    scheduler_->RequestShutdown();
    threads_.front().join();
    storage_lane_->Close();
    device_lane_->Close();
    if (backing_lane_ != nullptr) {
      backing_lane_->Close();
    }
    threads_.clear();
    EXPECT_EQ(storage_->in_flight(), 0U);
    EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
  }

  // Registers every extent's source at place `which`.
  void Place(std::size_t which) {
    addresses_.clear();
    for (std::size_t i = 0; i < kExtents; ++i) {
      const std::uint64_t address =
          memory_->RangeOf(places_.at(which)).value().base + (i * kExtent);
      addresses_.push_back(address);
      ASSERT_TRUE(
          scheduler_
              ->SetSource(extents_[i],
                          PageSource{.read = {.fd = fd_,
                                              .offset = i * kExtent,
                                              .memory = nullptr,
                                              .length = LengthOf(i)},
                                     .landed = true,
                                     .destination = address,
                                     .backing = BackingPlace{.reservation = places_.at(which),
                                                             .offset = Bytes(i * kExtent),
                                                             .size = Bytes(kExtent),
                                                             .allocation_class =
                                                                 ClassOf(BackingKind::kDevice)}})
              .has_value());
    }
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
  void Post(Control&& control) {
    // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
    while (scheduler_->Post(std::move(control)) == PushResult::kFull) {
      std::this_thread::yield();
    }
  }
  // Loads (and checks) every extent; returns the task's outcome.
  int Load(std::uint64_t request, bool check, Signals& signals) {
    Post(StartRequest{
        .request = request,
        .priority = 1,
        .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                  host_base_ + (kSlots * kExtent), file_, check)});
    EXPECT_TRUE(WaitFor(signals.retired));
    return signals.outcome.load();
  }
  // Runs `probe` on the scheduler thread until it returns true.
  void Owner(std::uint64_t request, std::function<bool()> probe) {
    Signals signals;
    Post(StartRequest{.request = request,
                      .priority = 1,
                      .program = std::make_unique<OnOwner>(signals, std::move(probe))});
    ASSERT_TRUE(WaitFor(signals.retired));
  }
  int Evict(std::uint64_t request) {
    Signals signals;
    Post(StartRequest{.request = request,
                      .priority = 1,
                      .program = std::make_unique<EvictAll>(signals, extents_)});
    EXPECT_TRUE(WaitFor(signals.retired));
    return signals.outcome.load();
  }

  std::unique_ptr<jitllm::providers::VmmProvider> memory_;
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  std::unique_ptr<jitllm::providers::UringStorage> storage_;
  jitllm::providers::Storage* lane_storage_ = nullptr;            // WrapProviders
  jitllm::providers::DeviceExecution* lane_execution_ = nullptr;  // WrapProviders
  // The storage lane reader's span_bytes: the default, or set by WrapProviders.
  std::uint32_t span_bytes_ = jitllm::providers::kNoCoalescing;
  jitllm::catalog::Catalog catalog_;
  jitllm::base::WakeFlag wake_;
  CompletionBoard board_{128, wake_};
  std::unique_ptr<StorageService> storage_lane_;
  std::unique_ptr<DeviceService> device_lane_;
  std::unique_ptr<jitllm::scheduler::BackingService> backing_lane_;
  std::unique_ptr<Scheduler> scheduler_;
  std::optional<std::expected<void, Fault>> result_status_;
  std::vector<std::jthread> threads_;  // the scheduler first

  int fd_ = -1;
  std::vector<std::byte> file_;
  ReservationId host_;
  jitllm::providers::BackingId host_backing_;
  std::uint64_t host_base_ = 0;
  std::array<ReservationId, 2> places_;
  std::vector<std::uint64_t> addresses_;
  std::size_t baseline_ = 0;
  jitllm::catalog::DomainId domain_;
  std::vector<ExtentId> extents_;
  ExtentId result_;
  StreamId stream_;
};

// Loads land in the zone and reach device VMM intact; eviction releases
// every backing, and reloads, in place and relocated, restore the bytes.
TEST_P(CudaPageIn, LandedLoadsReachDeviceVmmAndReloadIdentically) {
  Signals first;
  ASSERT_EQ(Load(1, true, first), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(first.mismatches.load(), 0);
  EXPECT_EQ(memory_->backings(), baseline_ + kExtents);
  EXPECT_EQ(scheduler_->slots_busy(), 0U);

  ASSERT_EQ(Evict(2), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(memory_->backings(), baseline_);  // released, not pooled (D-033)
  for (const ExtentId extent : extents_) {
    EXPECT_EQ(catalog_.Describe(extent).value().state, ExtentState::kNonresident);
  }
  Signals again;
  ASSERT_EQ(Load(3, true, again), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(again.mismatches.load(), 0);

  ASSERT_EQ(Evict(4), static_cast<int>(TaskOutcome::kSucceeded));
  Owner(6, [&] {  // relocated, on the thread that owns the sources
    Place(1);
    return true;
  });
  Signals moved;
  ASSERT_EQ(Load(5, true, moved), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(moved.mismatches.load(), 0);
  const auto occupied = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupied.loading, Bytes());
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(occupied.held, Bytes());
  EXPECT_EQ(occupied.idle, Bytes(kExtent * kExtents));
}

// A request cancelled while a read or copy of one of its loads holds a
// landing slot (the cancellation is applied on the scheduler thread the
// moment one does): every read drains, no slot stays busy, and each extent
// ends resident (its copy completed) or nonresident with its backing
// released. Whether a read the cancellation met was cancelled by io_uring
// or completed first is not observed here.
TEST_P(CudaPageIn, CancellingWithLoadsInFlightDrainsThem) {
  Signals signals;
  constexpr std::uint64_t kRequest = 10;
  Post(StartRequest{
      .request = kRequest,
      .priority = 1,
      .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                host_base_ + (kSlots * kExtent), file_, false)});
  std::size_t busy = 0;
  bool cancelled = false;
  Owner(11, [&] {
    busy = scheduler_->slots_busy();
    if (busy == 0 && scheduler_->loads() > 0) {
      return false;  // mapping still: no read has started
    }
    cancelled = scheduler_->Cancel(kRequest);
    return true;
  });
  EXPECT_GT(busy, 0U) << "every load finished before a read was seen in flight";
  EXPECT_TRUE(cancelled);
  ASSERT_TRUE(WaitFor(signals.retired));
  EXPECT_EQ(signals.outcome.load(), static_cast<int>(TaskOutcome::kCancelled));
  // Once the withdrawn loads have drained, on the scheduler thread.
  std::size_t resident = 0;
  std::size_t settled = 0;
  std::size_t busy_after = 0;
  Owner(12, [&] {
    if (scheduler_->loads() > 0) {
      return false;
    }
    for (const ExtentId extent : extents_) {
      const auto state = catalog_.Describe(extent).value().state;
      settled += state == ExtentState::kResident || state == ExtentState::kNonresident ? 1 : 0;
      resident += state == ExtentState::kResident ? 1 : 0;
    }
    busy_after = scheduler_->slots_busy();
    return true;
  });
  EXPECT_EQ(settled, kExtents);
  EXPECT_LT(resident, kExtents);
  EXPECT_EQ(busy_after, 0U);
  EXPECT_EQ(memory_->backings(), baseline_ + resident);
  ASSERT_EQ(Evict(13), static_cast<int>(TaskOutcome::kSucceeded));
  // What stays loads intact afterwards.
  Signals after;
  ASSERT_EQ(Load(50, true, after), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(after.mismatches.load(), 0);
}

// The other timing: cancelled while a landed copy is in flight (the device
// lane has taken its operation, and may or may not have queued it on the
// GPU yet; its fence has not been seen). A copy cannot be recalled, so
// each such load waits for its fence and publishes the whole extent; its
// slot is freed only then. The extents it published hold exactly the
// file's bytes.
TEST_P(CudaPageIn, CancellingDuringACopyWaitsForItsFence) {
  Signals signals;
  constexpr std::uint64_t kRequest = 20;
  Post(StartRequest{
      .request = kRequest,
      .priority = 1,
      .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                host_base_ + (kSlots * kExtent), file_, false)});
  std::size_t copying = 0;
  bool cancelled = false;
  Owner(21, [&] {
    copying = scheduler_->copying();
    if (copying == 0 && scheduler_->loads() > 0) {
      return false;  // no copy in flight yet
    }
    cancelled = scheduler_->Cancel(kRequest);
    return true;
  });
  EXPECT_GT(copying, 0U) << "every load finished before a copy was seen in flight";
  EXPECT_TRUE(cancelled);
  ASSERT_TRUE(WaitFor(signals.retired));
  EXPECT_EQ(signals.outcome.load(), static_cast<int>(TaskOutcome::kCancelled));
  std::size_t resident = 0;
  std::size_t settled = 0;
  std::size_t busy_after = 0;
  Owner(22, [&] {
    if (scheduler_->loads() > 0) {
      return false;
    }
    for (const ExtentId extent : extents_) {
      const auto state = catalog_.Describe(extent).value().state;
      settled += state == ExtentState::kResident || state == ExtentState::kNonresident ? 1 : 0;
      resident += state == ExtentState::kResident ? 1 : 0;
    }
    busy_after = scheduler_->slots_busy();
    return true;
  });
  EXPECT_EQ(settled, kExtents);
  EXPECT_GE(resident, copying);  // every copy in flight completed and published
  EXPECT_LT(resident, kExtents);
  EXPECT_EQ(busy_after, 0U);
  EXPECT_EQ(memory_->backings(), baseline_ + resident);
  // What the cancelled request published is whole: each extent copied back
  // under a lease and compared with its chunk of the file.
  std::vector<ExtentId> published;
  std::vector<std::uint64_t> places;
  std::vector<std::size_t> chunks;
  for (std::size_t i = 0; i < kExtents; ++i) {
    if (catalog_.Describe(extents_[i]).value().state == ExtentState::kResident) {
      published.push_back(extents_[i]);
      places.push_back(addresses_[i]);
      chunks.push_back(i);
    }
  }
  Signals checked;
  Post(StartRequest{.request = 30,
                    .priority = 1,
                    .program = std::make_unique<LoadAndCheck>(
                        checked, catalog_, published, places, result_,
                        host_base_ + (kSlots * kExtent), file_, true, chunks)});
  ASSERT_TRUE(WaitFor(checked.retired));
  ASSERT_EQ(checked.outcome.load(), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(checked.mismatches.load(), 0);
  ASSERT_EQ(Evict(90), static_cast<int>(TaskOutcome::kSucceeded));
  Signals after;
  ASSERT_EQ(Load(91, true, after), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(after.mismatches.load(), 0);
}

// BP-V2 on the real provider: backing the driver cannot create (more than
// the device has) and a mapping the provider refuses (outside its
// reservation) each fail their load with a known outcome and unwind it
// completely: no backing left, the extent nonresident and uncharged, no
// fault. The same extent then loads from its real place.
TEST_P(CudaPageIn, BackingThatCannotBeMadeOrMappedUnwindsCleanly) {
  const auto attempt = [&](std::uint64_t request, BackingPlace place) {
    Owner(request, [&] {
      PageSource source{.read = {.fd = fd_, .offset = 0, .memory = nullptr, .length = kExtent},
                        .landed = true,
                        .destination = addresses_[0],
                        .backing = place};
      EXPECT_TRUE(scheduler_->SetSource(extents_[0], source).has_value());
      return true;
    });
    Signals signals;
    Post(StartRequest{.request = request + 1,
                      .priority = 1,
                      .program = std::make_unique<LoadAndCheck>(
                          signals, catalog_, std::vector{extents_[0]}, std::vector{addresses_[0]},
                          result_, host_base_ + (kSlots * kExtent), file_, false)});
    EXPECT_TRUE(WaitFor(signals.retired));
    EXPECT_EQ(signals.outcome.load(), static_cast<int>(TaskOutcome::kFailed));
    bool clean = false;
    Owner(request + 2, [&] {
      const auto view = catalog_.Describe(extents_[0]).value();
      clean = view.state == ExtentState::kNonresident && !scheduler_->fault().has_value() &&
              scheduler_->slots_busy() == 0 && catalog_.OccupancyOf(domain_).loading == Bytes() &&
              catalog_.OccupancyOf(domain_).quarantined == Bytes();
      return true;
    });
    EXPECT_TRUE(clean);
    EXPECT_EQ(memory_->backings(), baseline_);
  };
  const std::size_t device = ClassOf(BackingKind::kDevice);
  // More backing than the device has: cuMemCreate refuses it.
  attempt(100, BackingPlace{.reservation = places_[0],
                            .offset = Bytes(0),
                            .size = Bytes(std::uint64_t{1} << 40U),
                            .allocation_class = device});
  // A place outside the reservation: made, refused at the map, released.
  attempt(110, BackingPlace{.reservation = places_[0],
                            .offset = Bytes(kExtent * kExtents),
                            .size = Bytes(kExtent),
                            .allocation_class = device});
  Owner(120, [&] {
    Place(0);
    return true;
  });
  Signals loaded;
  ASSERT_EQ(Load(121, true, loaded), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(loaded.mismatches.load(), 0);
}

// BP-L2 on the real providers, repeated: a request cancelled at a
// different point of its loads each round (after 1 to 8 slots were seen
// busy), then everything evicted and loaded again. Whatever io_uring did
// with each cancellation (the read completed, or was cancelled), no late
// transfer lands in a slot or extent reassigned since (invariant 3): every
// reload holds exactly the file's bytes.
TEST_P(CudaPageIn, RepeatedCancellationsNeverCorruptAReassignedSlot) {
  for (std::uint64_t round = 0; round < 8; ++round) {
    Signals signals;
    const std::uint64_t request = 200 + (round * 10);
    Post(StartRequest{
        .request = request,
        .priority = 1,
        .program = std::make_unique<LoadAndCheck>(signals, catalog_, extents_, addresses_, result_,
                                                  host_base_ + (kSlots * kExtent), file_, false)});
    Owner(request + 1, [&] {
      if (scheduler_->slots_busy() <= round && scheduler_->loads() > 0) {
        return false;  // not that many reads in flight yet
      }
      (void)scheduler_->Cancel(request);
      return true;
    });
    ASSERT_TRUE(WaitFor(signals.retired));
    Owner(request + 2, [&] { return scheduler_->loads() == 0; });
    ASSERT_EQ(Evict(request + 3), static_cast<int>(TaskOutcome::kSucceeded));
    Signals again;
    ASSERT_EQ(Load(request + 4, true, again), static_cast<int>(TaskOutcome::kSucceeded));
    EXPECT_EQ(again.mismatches.load(), 0) << "round " << round;
    ASSERT_EQ(Evict(request + 5), static_cast<int>(TaskOutcome::kSucceeded));
  }
  EXPECT_EQ(memory_->backings(), baseline_);
}

INSTANTIATE_TEST_SUITE_P(VmmWork, CudaPageIn, ::testing::Bool(), [](const auto& info) {
  return info.param ? std::string("OnAVmmLane") : std::string("OnTheDeviceLane");
});

// BP-L4: D-048's event permutations with the real providers, through
// decorators the lanes call instead of them. The storage decorator
// reports some submissions as of unknown start although io_uring took
// them, and hands every completion over twice; the execution decorator
// can report fence queries as of unknown outcome.
class PermutingStorage final : public jitllm::providers::Storage {
 public:
  explicit PermutingStorage(jitllm::providers::Storage& inner) : inner_(inner) {}
  std::atomic<int> unknown_starts{0};  // the next ones reported unknown
  std::atomic<int> duplicated{0};      // completions handed over twice

  std::size_t depth() const override { return inner_.depth(); }
  std::size_t in_flight() const override { return inner_.in_flight(); }
  jitllm::providers::Submission Submit(const jitllm::providers::IoRequest& request) override {
    const auto submitted = inner_.Submit(request);
    if (submitted == jitllm::providers::Submission::kAccepted && unknown_starts.load() > 0) {
      unknown_starts.fetch_sub(1);
      return jitllm::providers::Submission::kUnknown;
    }
    return submitted;
  }
  jitllm::providers::Submission Cancel(std::uint64_t token) override {
    return inner_.Cancel(token);
  }
  std::size_t Harvest(std::span<jitllm::providers::IoCompletion> out, bool wait) override {
    const std::size_t half = out.size() / 2;
    const std::size_t n = inner_.Harvest(out.first(half), wait);
    for (std::size_t i = 0; i < n; ++i) {
      out[n + i] = out[i];  // the same completion again, after the first
    }
    duplicated.fetch_add(static_cast<int>(n));
    return 2 * n;
  }
  void Wake() override { inner_.Wake(); }

 private:
  jitllm::providers::Storage& inner_;
};

class UnknownQueries final : public jitllm::providers::DeviceExecution {
 public:
  explicit UnknownQueries(jitllm::providers::DeviceExecution& inner) : inner_(inner) {}
  std::atomic<bool> unknown{false};  // every query from now on

  std::expected<StreamId, jitllm::providers::Failure> CreateStream() override {
    return inner_.CreateStream();
  }
  std::expected<void, jitllm::providers::Failure> DestroyStream(StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, jitllm::providers::Failure> Copy(StreamId stream, std::uint64_t destination,
                                                       std::uint64_t source, Bytes size) override {
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<void, jitllm::providers::Failure> Zero(StreamId stream, std::uint64_t destination,
                                                       Bytes size) override {
    return inner_.Zero(stream, destination, size);
  }
  std::expected<jitllm::providers::NativeStream, jitllm::providers::Failure> Submission(
      StreamId stream) override {
    return inner_.Submission(stream);
  }
  std::expected<void, jitllm::providers::Failure> Wait(StreamId stream,
                                                       jitllm::providers::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<jitllm::providers::FenceId, jitllm::providers::Failure> Record(
      StreamId stream) override {
    return inner_.Record(stream);
  }
  std::expected<jitllm::providers::FenceState, jitllm::providers::Failure> Query(
      jitllm::providers::FenceId fence) override {
    if (unknown.load()) {
      return std::unexpected(jitllm::providers::Failure{
          .error = jitllm::providers::ProviderError::kUnknown, .detail = "a scripted fault"});
    }
    return inner_.Query(fence);
  }
  std::expected<void, jitllm::providers::Failure> Release(
      jitllm::providers::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  jitllm::providers::DeviceExecution& inner_;
};

class CudaPermutations : public CudaPageIn {
 protected:
  void WrapProviders() override {
    storage_decorator_ = std::make_unique<PermutingStorage>(*storage_);
    execution_decorator_ = std::make_unique<UnknownQueries>(*execution_);
    lane_storage_ = storage_decorator_.get();
    lane_execution_ = execution_decorator_.get();
  }
  // After a quarantine the node keeps what it holds (invariant 8): the
  // lanes are stopped, then what stayed mapped is released by hand so the
  // process can go on, and the stream, whose unproven fence was never
  // released, is left as it is.
  void TearDown() override {
    if (!quarantined_) {
      CudaPageIn::TearDown();
      return;
    }
    if (!threads_.empty()) {
      StopFaulted();
    }
    for (std::size_t i = 0; i < kExtents; ++i) {
      const auto backing = memory_->MappedAt(places_[0], Bytes(i * kExtent));
      if (backing) {
        EXPECT_TRUE(memory_->Unmap(places_[0], Bytes(i * kExtent), Bytes(kExtent)).has_value());
        EXPECT_TRUE(memory_->Release(*backing).has_value());
      }
    }
    EXPECT_EQ(memory_->backings(), baseline_);
    EXPECT_TRUE(memory_->Unmap(host_, Bytes(0), Bytes(kExtent * (kSlots + 1))).has_value());
    EXPECT_TRUE(memory_->Release(host_backing_).has_value());
    EXPECT_TRUE(memory_->Free(host_).has_value());
    for (const ReservationId place : places_) {
      EXPECT_TRUE(memory_->Free(place).has_value());
    }
    (void)::close(fd_);
  }

  // A faulted node admits nothing more, so its state is read once it has
  // stopped: the scheduler's thread has returned and the lanes drained.
  void StopFaulted() {
    scheduler_->RequestShutdown();
    threads_.front().join();
    storage_lane_->Close();
    device_lane_->Close();
    if (backing_lane_ != nullptr) {
      backing_lane_->Close();
    }
    threads_.clear();
    EXPECT_EQ(storage_->in_flight(), 0U);
  }

  std::unique_ptr<PermutingStorage> storage_decorator_;
  std::unique_ptr<UnknownQueries> execution_decorator_;
  bool quarantined_ = false;
};

// Reads whose start io_uring reported as unknown are waited for like any
// other, and a completion handed over twice is taken once: every extent
// loads intact, nothing is quarantined and the node does not fault.
TEST_P(CudaPermutations, UnknownStartsAndDuplicateCompletionsChangeNothing) {
  storage_decorator_->unknown_starts.store(12);
  Signals loaded;
  ASSERT_EQ(Load(1, true, loaded), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(loaded.mismatches.load(), 0);
  EXPECT_EQ(storage_decorator_->unknown_starts.load(), 0);
  EXPECT_GE(storage_decorator_->duplicated.load(), static_cast<int>(kExtents));
  bool healthy = false;
  Owner(2, [&] {
    healthy = !scheduler_->fault().has_value() && scheduler_->quarantined() == 0 &&
              catalog_.OccupancyOf(domain_).quarantined == Bytes();
    return true;
  });
  EXPECT_TRUE(healthy);
}

// A copy whose fence can no longer be queried is unproven: the load's
// extent and its slot are quarantined and stay charged, and the node
// faults, which stops admission. Timeout is never reclaim (invariant 8).
TEST_P(CudaPermutations, AFenceOfUnknownOutcomeQuarantinesAndStopsAdmission) {
  quarantined_ = true;
  execution_decorator_->unknown.store(true);
  Signals loading;
  Post(StartRequest{.request = 1,
                    .priority = 1,
                    .program = std::make_unique<LoadAndCheck>(
                        loading, catalog_, std::vector{extents_[0]}, std::vector{addresses_[0]},
                        result_, host_base_ + (kSlots * kExtent), file_, false)});
  ASSERT_TRUE(WaitFor(loading.retired));
  EXPECT_EQ(loading.outcome.load(), static_cast<int>(TaskOutcome::kFailed));
  StopFaulted();
  ASSERT_TRUE(result_status_.has_value());
  // The stop reports the fault.
  EXPECT_TRUE(result_status_.has_value() && !result_status_->has_value());
  EXPECT_EQ(scheduler_->fault(), Fault::kUnproven);
  EXPECT_EQ(catalog_.Describe(extents_[0]).value().state, ExtentState::kQuarantined);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).quarantined, Bytes(kExtent));
  EXPECT_EQ(scheduler_->slots_quarantined(), 1U);
  EXPECT_EQ(memory_->backings(), baseline_ + 1);  // still charged, never reused
}

INSTANTIATE_TEST_SUITE_P(VmmWork, CudaPermutations, ::testing::Bool(), [](const auto& info) {
  return info.param ? std::string("OnAVmmLane") : std::string("OnTheDeviceLane");
});

// BP-P1 over the real providers: every extent evicted and restored with
// coalesced reads. A decorator records what the reader hands io_uring and
// holds a lone read back briefly, so reads wait for room whatever the
// host's timing: those that continue one another go as one READV into
// their own landing slots. Every extent comes back
// identical, the requests start in file order (RE-026), each chunk is in
// exactly one of them, and none spans more than the zone's slots.
class RecordingStorage final : public jitllm::providers::Storage {
 public:
  struct Seen {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::vector<jitllm::providers::IoSegment> segments;
  };
  // A lone read is refused as if the ring were full, for up to `grace`, so
  // later reads reliably join it however fast this host maps backing (a
  // VMM lane can publish reads no faster than the disk takes them); past
  // that, it starts alone (the load's last read, say).
  RecordingStorage(jitllm::providers::Storage& inner, std::chrono::microseconds grace)
      : inner_(inner), grace_(grace) {}

  std::size_t depth() const override { return inner_.depth(); }
  std::size_t in_flight() const override { return inner_.in_flight(); }
  jitllm::providers::Submission Submit(const jitllm::providers::IoRequest& request) override {
    const auto now = std::chrono::steady_clock::now();
    if (request.segments.empty()) {
      if (!lone_ || lone_->first != request.offset) {
        lone_.emplace(request.offset, now);
      }
      if (now - lone_->second < grace_) {
        return jitllm::providers::Submission::kNotStarted;
      }
    }
    lone_.reset();
    const auto submitted = inner_.Submit(request);
    if (submitted != jitllm::providers::Submission::kNotStarted) {
      const std::scoped_lock lock(mutex_);
      seen_.push_back(Seen{.offset = request.offset,
                           .length = request.length,
                           .segments = std::vector<jitllm::providers::IoSegment>(
                               request.segments.begin(), request.segments.end())});
    }
    return submitted;
  }
  jitllm::providers::Submission Cancel(std::uint64_t token) override {
    return inner_.Cancel(token);
  }
  std::size_t Harvest(std::span<jitllm::providers::IoCompletion> out, bool wait) override {
    return inner_.Harvest(out, wait);
  }
  void Wake() override { inner_.Wake(); }

  std::vector<Seen> Take() {
    const std::scoped_lock lock(mutex_);
    return std::exchange(seen_, {});
  }

 private:
  jitllm::providers::Storage& inner_;
  std::chrono::microseconds grace_;
  // The lone read being held back: its offset, and since when.
  std::optional<std::pair<std::uint64_t, std::chrono::steady_clock::time_point>> lone_;
  std::mutex mutex_;
  std::vector<Seen> seen_;
};

// The parameter: whether VMM work runs on a VMM lane (as CudaPageIn's).
// Coalescing is off by default: this turns it on (64 MiB spans). The
// hold-back is this decorator's, never the reader's.
class CudaCoalescing : public CudaPageIn {
 protected:
  void WrapProviders() override {
    span_bytes_ = jitllm::providers::kSpanBytes;
    recording_ = std::make_unique<RecordingStorage>(*storage_, std::chrono::milliseconds(5));
    lane_storage_ = recording_.get();
  }
  std::unique_ptr<RecordingStorage> recording_;
};

TEST_P(CudaCoalescing, EveryExtentRestoresIdenticallyFromCoalescedReads) {
  Signals first;
  ASSERT_EQ(Load(1, true, first), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(first.mismatches.load(), 0);
  (void)recording_->Take();
  for (std::uint64_t round = 0; round < 3; ++round) {
    ASSERT_EQ(Evict(10 + (2 * round)), static_cast<int>(TaskOutcome::kSucceeded));
    Signals restored;
    ASSERT_EQ(Load(11 + (2 * round), true, restored), static_cast<int>(TaskOutcome::kSucceeded));
    EXPECT_EQ(restored.mismatches.load(), 0) << round;
    const std::vector<RecordingStorage::Seen> seen = recording_->Take();
    std::size_t pieces = 0;
    std::size_t spans = 0;
    std::uint64_t bytes = 0;
    for (std::size_t r = 0; r < seen.size(); ++r) {
      const RecordingStorage::Seen& request = seen[r];
      if (r > 0) {
        EXPECT_GT(request.offset, seen[r - 1].offset) << round;  // file order (RE-026)
      }
      bytes += request.length;
      EXPECT_LE(request.length, kSlots * kExtent) << round;
      pieces += std::max<std::size_t>(request.segments.size(), 1);
      spans += request.segments.size() > 1 ? 1 : 0;
      for (const auto& segment : request.segments) {
        const auto address = reinterpret_cast<std::uint64_t>(segment.memory);
        EXPECT_GE(address, host_base_) << round;
        EXPECT_LE(address + segment.length, host_base_ + (kSlots * kExtent)) << round;
      }
    }
    EXPECT_EQ(pieces, kExtents) << round;
    EXPECT_EQ(bytes, ((kExtents - 1) * kExtent) + kTail) << round;
    EXPECT_GT(spans, 0U) << round;  // coalesced
    EXPECT_LT(seen.size(), kExtents) << round;
  }
}

INSTANTIATE_TEST_SUITE_P(VmmWork, CudaCoalescing, ::testing::Bool(), [](const auto& info) {
  return info.param ? std::string("OnAVmmLane") : std::string("OnTheDeviceLane");
});

// Write-back of live state over the real providers (BP-P4's path,
// scheduler.h): four extents of device VMM with state, their write-back
// places in a direct-I/O spill file. Evicting them copies each into a
// landing slot on the zone's stream, writes the slot with io_uring, and
// then releases the backing (the managed pair) or leaves it mapped (the
// premapped pair, whose eviction is the catalog's alone). The premapped
// pair's memory is then poisoned while nonresident, so its restore must
// bring back every byte (invariant 4); the managed pair comes back in
// fresh backing. Each extent is copied back to host VMM under a lease and
// compared with what it held before.
class CudaWriteBack : public CudaPageIn {
 protected:
  static constexpr std::size_t kStates = 4;  // 0 and 1 premapped, 2 and 3 managed
  static constexpr std::size_t kPremapped = 2;

  void BeforeLanes() override {
    const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    const std::filesystem::path directory = scratch != nullptr
                                                ? std::filesystem::path(scratch)
                                                : std::filesystem::path(::testing::TempDir());
    spill_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(spill_, 0);
    states_place_ = memory_->Reserve(Bytes(kExtent * kStates)).value();
    const std::uint64_t base = memory_->RangeOf(states_place_).value().base;
    patterns_.resize(kExtent * kStates);
    for (std::uint64_t i = 0; i < patterns_.size(); ++i) {
      patterns_[i] = static_cast<std::byte>((i * 37) + (i >> 20) + 5);
    }
    // Written in through the host result extent (host VMM, CPU-mapped),
    // one extent at a time, before any lane runs.
    const std::uint64_t result = host_base_ + (kSlots * kExtent);
    for (std::size_t i = 0; i < kStates; ++i) {
      const auto backing = memory_->Create(ClassOf(BackingKind::kDevice), Bytes(kExtent)).value();
      ASSERT_TRUE(memory_->Map(states_place_, Bytes(i * kExtent), backing).has_value());
      ASSERT_TRUE(
          memory_->SetAccess(states_place_, Bytes(i * kExtent), Bytes(kExtent), Access::kReadWrite)
              .has_value());
      if (i < kPremapped) {
        premapped_.push_back(backing);
      }
      std::memcpy(At(result), patterns_.data() + (i * kExtent), kExtent);
      const auto fence = [&]() -> bool {
        if (!execution_->Copy(stream_, base + (i * kExtent), result, Bytes(kExtent))) {
          return false;
        }
        const auto recorded = execution_->Record(stream_);
        if (!recorded) {
          return false;
        }
        const auto give_up = std::chrono::steady_clock::now() + kPatience;
        while (std::chrono::steady_clock::now() < give_up) {
          const auto state = execution_->Query(*recorded);
          if (state && *state == jitllm::providers::FenceState::kComplete) {
            return execution_->Release(*recorded).has_value();
          }
        }
        return false;
      }();
      ASSERT_TRUE(fence);
      const ExtentId state =
          catalog_
              .AddExtent({.domain = domain_,
                          .memory_class = jitllm::catalog::MemoryClass::kLiveState,
                          .recovery = jitllm::catalog::Recovery::kPreserve,
                          .size = Bytes(kExtent),
                          .content = {}},
                         true)
              .value();
      states_.push_back(state);
      addresses_of_states_.push_back(base + (i * kExtent));
      std::optional<BackingPlace> managed;
      if (i >= kPremapped) {
        managed = BackingPlace{.reservation = states_place_,
                               .offset = Bytes(i * kExtent),
                               .size = Bytes(kExtent),
                               .allocation_class = ClassOf(BackingKind::kDevice)};
      }
      ASSERT_TRUE(scheduler_
                      ->SetSource(state, PageSource{.read = {.fd = spill_,
                                                             .offset = i * kExtent,
                                                             .memory = nullptr,
                                                             .length = kExtent},
                                                    .landed = true,
                                                    .destination = base + (i * kExtent),
                                                    .backing = managed,
                                                    .write_back = true})
                      .has_value());
    }
    // The premapped pair stays mapped throughout: the base's check counts it.
    baseline_ += kPremapped;
  }

  void TearDown() override {
    if (scheduler_ != nullptr && !threads_.empty()) {
      Signals evicted;
      Post(StartRequest{
          .request = 998, .priority = 1, .program = std::make_unique<EvictAll>(evicted, states_)});
      EXPECT_TRUE(WaitFor(evicted.retired));
    }
    CudaPageIn::TearDown();  // the lanes have stopped: this thread calls the providers again
    if (states_place_.valid()) {
      EXPECT_TRUE(memory_->Unmap(states_place_, Bytes(0), Bytes(kExtent * kPremapped)).has_value());
      for (const auto backing : premapped_) {
        EXPECT_TRUE(memory_->Release(backing).has_value());
      }
      EXPECT_TRUE(memory_->Free(states_place_).has_value());
    }
    if (spill_ >= 0) {
      (void)::close(spill_);
    }
  }

  // Copies each state back under a lease and compares it with its pattern.
  int Check(std::uint64_t request, Signals& signals) {
    Post(StartRequest{.request = request,
                      .priority = 1,
                      .program = std::make_unique<LoadAndCheck>(
                          signals, catalog_, states_, addresses_of_states_, result_,
                          host_base_ + (kSlots * kExtent), patterns_, true,
                          std::vector<std::size_t>{0, 1, 2, 3})});
    EXPECT_TRUE(WaitFor(signals.retired));
    return signals.outcome.load();
  }

  int spill_ = -1;
  ReservationId states_place_;
  std::vector<jitllm::providers::BackingId> premapped_;
  std::vector<ExtentId> states_;
  std::vector<std::uint64_t> addresses_of_states_;
  std::vector<std::byte> patterns_;
};

TEST_P(CudaWriteBack, LiveStateIsWrittenBackThroughTheZoneAndRestoredExactly) {
  Signals before;
  ASSERT_EQ(Check(1, before), static_cast<int>(TaskOutcome::kSucceeded));
  ASSERT_EQ(before.mismatches.load(), 0);

  Signals evicted;
  Post(StartRequest{
      .request = 2, .priority = 1, .program = std::make_unique<EvictAll>(evicted, states_)});
  ASSERT_TRUE(WaitFor(evicted.retired));
  ASSERT_EQ(evicted.outcome.load(), static_cast<int>(TaskOutcome::kSucceeded));
  std::size_t preserved = 0;
  Owner(3, [&] {
    for (const ExtentId state : states_) {
      const auto view = catalog_.Describe(state).value();
      preserved += view.state == ExtentState::kNonresident && view.preserved ? 1 : 0;
    }
    return true;
  });
  EXPECT_EQ(preserved, kStates);
  EXPECT_EQ(memory_->backings(), baseline_);  // the managed pair's backing released (D-033)

  // Poison the premapped pair while nonresident: nothing may read it now.
  CUcontext context = nullptr;
  ASSERT_EQ(cuDevicePrimaryCtxRetain(&context, 0), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxPushCurrent(context), CUDA_SUCCESS);
  EXPECT_EQ(cuMemsetD8(addresses_of_states_[0], 0xa5, kExtent * kPremapped), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);
  EXPECT_EQ(cuCtxPopCurrent(&context), CUDA_SUCCESS);
  EXPECT_EQ(cuDevicePrimaryCtxRelease(0), CUDA_SUCCESS);

  // Restored from the spill file through the zone, then compared.
  Signals restored;
  ASSERT_EQ(Check(4, restored), static_cast<int>(TaskOutcome::kSucceeded));
  EXPECT_EQ(restored.mismatches.load(), 0);
  std::size_t live = 0;
  Owner(5, [&] {
    for (const ExtentId state : states_) {
      const auto view = catalog_.Describe(state).value();
      live += view.state == ExtentState::kResident && !view.preserved ? 1 : 0;
    }
    return true;
  });
  EXPECT_EQ(live, kStates);
  EXPECT_EQ(memory_->backings(), baseline_ + (kStates - kPremapped));
  const auto occupied = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupied.quarantined, Bytes());
  EXPECT_EQ(occupied.evicting, Bytes());
}

INSTANTIATE_TEST_SUITE_P(VmmWork, CudaWriteBack, ::testing::Bool(), [](const auto& info) {
  return info.param ? std::string("OnAVmmLane") : std::string("OnTheDeviceLane");
});

}  // namespace
