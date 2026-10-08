// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// BP-S3's node on a real device (tests/support/paged_node.h), and M3's
// swap path on it (a full swap with the handoff; the copy lane beside a
// job whose stream is full, RE-029). Two models,
// each on its own stream, share one catalog domain, one scheduler with its
// lanes, one landing zone and one workspace, under an execution budget
// that holds the larger model's weights and half the smaller's. They
// alternate: each acquisition (AcquireProgram) evicts only the other
// model's weights, releasing their backing, and pages in what is missing
// through the zone into device VMM; each model's job then copies every
// weight extent through the shared workspace to pinned staging on its own
// stream, and the bytes must be the file's. The occupancy never exceeds
// the budget, and teardown leaves no backing. The real models alternate in
// benchmarks/alternate_paged.cc.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <print>
#include <set>
#include <stop_token>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "paged_node.h"
#include "paged_programs.h"
#include "providers/fake/fake_storage.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace {

namespace sc = jitllm::scheduler;
namespace ts = jitllm::test_support;
using jitllm::base::Bytes;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;

constexpr std::uint64_t kExtent = ts::kPagedExtent;
constexpr std::array<std::size_t, 2> kExtents = {4, 3};  // model 0 is the larger

TEST(CountingStorageTest, CompletedReadBytesExcludeWritesFailuresAndUnstartedRequests) {
  namespace pr = jitllm::providers;
  pr::fake::FakeStorage inner(8, 1);
  jitllm::engine::CountingStorage storage(inner);
  std::array<std::byte, 16> memory{};
  const int fd = inner.AddFile(std::vector<std::byte>(memory.size(), std::byte{7}));
  const auto submit = [&](std::uint64_t token, pr::IoKind kind) {
    return storage.Submit({.token = token,
                           .kind = kind,
                           .fd = fd,
                           .memory = memory.data(),
                           .length = static_cast<std::uint32_t>(memory.size()),
                           .segments = {}});
  };
  inner.ScriptNext({.submission = pr::Submission::kNotStarted, .result = std::nullopt});
  EXPECT_EQ(submit(1, pr::IoKind::kRead), pr::Submission::kNotStarted);
  EXPECT_EQ(storage.read_submitted_bytes.load(), 0U);
  inner.ScriptNext({.submission = pr::Submission::kUnknown, .result = 5});
  EXPECT_EQ(submit(2, pr::IoKind::kRead), pr::Submission::kUnknown);
  inner.ScriptNext({.result = -EIO});
  EXPECT_EQ(submit(3, pr::IoKind::kRead), pr::Submission::kAccepted);
  EXPECT_EQ(submit(4, pr::IoKind::kWrite), pr::Submission::kAccepted);
  EXPECT_EQ(storage.read_submitted_bytes.load(), 32U);
  EXPECT_EQ(storage.read_completed_bytes.load(), 0U);
  std::array<pr::IoCompletion, 8> done{};
  EXPECT_EQ(storage.Harvest(done, false), 3U);
  EXPECT_EQ(storage.read_completed_bytes.load(), 5U);
  EXPECT_EQ(storage.requests.load(), 3U);
  EXPECT_FALSE(storage.oldest_in_flight());
  const std::array<pr::IoSegment, 2> pieces = {
      pr::IoSegment{.memory = memory.data(), .length = 8},
      pr::IoSegment{.memory = memory.data() + 8, .length = 8}};
  EXPECT_EQ(storage.Submit({.token = 5, .fd = fd, .length = 16, .segments = pieces}),
            pr::Submission::kAccepted);
  EXPECT_EQ(storage.Harvest(done, false), 1U);
  EXPECT_EQ(storage.read_submitted_bytes.load(), 48U);
  EXPECT_EQ(storage.read_completed_bytes.load(), 21U);
}

TEST(CountingStorageTest, HeldReadsCountOnlyOncePassedAndCancelledHeldReadsCountNothing) {
  namespace pr = jitllm::providers;
  pr::fake::FakeStorage inner(8, 1);
  std::atomic<bool> hold{true};
  jitllm::engine::CountingStorage storage(inner, &hold, true);
  std::array<std::byte, 16> memory{};
  const int fd = inner.AddFile(std::vector<std::byte>(memory.size(), std::byte{7}));
  const auto submit = [&](std::uint64_t token) {
    return storage.Submit({.token = token,
                           .fd = fd,
                           .memory = memory.data(),
                           .length = static_cast<std::uint32_t>(memory.size()),
                           .segments = {}});
  };
  EXPECT_EQ(submit(1), pr::Submission::kAccepted);
  EXPECT_EQ(storage.Cancel(1), pr::Submission::kAccepted);
  std::array<pr::IoCompletion, 8> done{};
  EXPECT_EQ(storage.Harvest(done, false), 1U);
  EXPECT_EQ(storage.read_submitted_bytes.load(), 0U);
  EXPECT_EQ(storage.read_completed_bytes.load(), 0U);
  EXPECT_EQ(submit(2), pr::Submission::kAccepted);
  hold.store(false);
  EXPECT_EQ(storage.Harvest(done, false), 1U);
  EXPECT_EQ(storage.read_submitted_bytes.load(), 16U);
  EXPECT_EQ(storage.read_completed_bytes.load(), 16U);
  EXPECT_EQ(storage.requests.load(), 1U);
  EXPECT_EQ(storage.held.load(), 2U);
  EXPECT_FALSE(storage.oldest_in_flight());
}

// A synthetic model: its weights in an unnamed direct-I/O file, read into
// managed device backing at one place, and pinned staging to read them
// back through the shared workspace.
class Model final : public ts::PagedModel {
 public:
  // `fence_weights`: its fence closure is its weights (like a runner's
  // state, nonresident once the model is swapped out), not its staging.
  Model(ts::PagedNode& node, std::uint32_t index, bool fence_weights = false,
        bool host_first = false)
      : node_(node), index_(index), fence_weights_(fence_weights), host_first_(host_first) {}

  void Setup(const std::filesystem::path& directory) {
    const std::size_t extents = kExtents.at(index_);
    file_.resize(extents * kExtent);
    for (std::uint64_t i = 0; i < file_.size(); ++i) {
      file_[i] = static_cast<std::byte>((i * 29) + (i >> 21) + (std::uint64_t{index_} * 101) + 3);
    }
    fd_ = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    ASSERT_GE(fd_, 0);
    auto* aligned = static_cast<std::byte*>(
        std::aligned_alloc(4096, file_.size()));  // NOLINT(cppcoreguidelines-no-malloc)
    std::memcpy(aligned, file_.data(), file_.size());
    const ssize_t written = ::pwrite(fd_, aligned, file_.size(), 0);
    std::free(aligned);  // NOLINT(cppcoreguidelines-no-malloc)
    ASSERT_EQ(written, static_cast<ssize_t>(file_.size()));
    place_ = node_.memory().Reserve(Bytes(file_.size())).value();
    base_ = node_.memory().RangeOf(place_).value().base;
    std::array<std::uint8_t, 32> artifact{};
    artifact[0] = static_cast<std::uint8_t>(index_ + 1);
    for (std::uint32_t i = 0; i < extents; ++i) {
      weights_.push_back(node_.catalog()
                             .AddExtent({.domain = node_.domain(),
                                         .memory_class = jitllm::catalog::MemoryClass::kWeights,
                                         .recovery = jitllm::catalog::Recovery::kFromArtifact,
                                         .size = Bytes(kExtent),
                                         .content = {.artifact = artifact, .group = 0, .chunk = i}})
                             .value());
    }
    auto staging = node_.Pinned(file_.size(), static_cast<int>(index_), staging_);
    ASSERT_TRUE(staging.has_value());
    staging_bytes_ = static_cast<std::byte*>(*staging);
  }

  void Register() {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      ASSERT_TRUE(
          node_.scheduler()
              .SetSource(
                  weights_[i],
                  sc::PageSource{
                      .read =
                          {.fd = fd_, .offset = i * kExtent, .memory = nullptr, .length = kExtent},
                      .landed = true,
                      .destination = base_ + (i * kExtent),
                      .backing = sc::BackingPlace{.reservation = place_,
                                                  .offset = Bytes(i * kExtent),
                                                  .size = Bytes(kExtent),
                                                  .allocation_class = host_first_ && i == 0
                                                                          ? node_.host_class()
                                                                          : node_.device_class()}})
              .has_value());
    }
    std::vector<ExtentId> all = weights_;
    all.insert(all.end(), node_.activations().extents.begin(), node_.activations().extents.end());
    all.insert(all.end(), node_.pool().extents.begin(), node_.pool().extents.end());
    all.insert(all.end(), staging_.begin(), staging_.end());
    closure_ = node_.catalog().ClosureOfExtents(all).value();
    fence_ = node_.catalog().ClosureOfExtents(fence_weights_ ? weights_ : staging_).value();
  }

  // Every weight extent through the workspace to the staging, on this
  // model's stream, under a lease on the whole closure.
  ts::Status ReadBack() {
    std::memset(staging_bytes_, 0, file_.size());
    const std::uint64_t workspace = node_.activations().base;
    const std::uint64_t base = base_;
    std::byte* staging = staging_bytes_;
    const std::size_t extents = weights_.size();
    return node_.Job(
        closure_,
        [=](jitllm::providers::NativeStream native) {
          auto* stream = static_cast<cudaStream_t>(native.handle);
          for (std::size_t i = 0; i < extents; ++i) {
            // NOLINTBEGIN(performance-no-int-to-ptr)
            if (cudaMemcpyAsync(reinterpret_cast<void*>(workspace),
                                reinterpret_cast<void*>(base + (i * kExtent)), kExtent,
                                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
                cudaMemcpyAsync(staging + (i * kExtent), reinterpret_cast<void*>(workspace),
                                kExtent, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
              // NOLINTEND(performance-no-int-to-ptr)
              return sc::JobResult::kUnknown;  // a runtime error, even the first (AfterRefusal)
            }
          }
          return sc::JobResult::kQueued;
        },
        "reading the weights back", index_);
  }
  bool Intact() const { return std::memcmp(staging_bytes_, file_.data(), file_.size()) == 0; }

  std::size_t Resident() const {
    std::size_t resident = 0;
    for (const ExtentId extent : weights_) {
      resident += node_.catalog().Describe(extent).value().state == ExtentState::kResident ? 1 : 0;
    }
    return resident;
  }
  const jitllm::catalog::Closure& closure() const { return closure_; }
  const std::vector<ExtentId>& weights() const { return weights_; }

  std::uint32_t stream() const override { return index_; }
  const jitllm::catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<ExtentId> managed_extents() const override { return weights_; }
  // Called first in Release, once the node's teardown has fenced the stream.
  std::function<void()> on_release;

  ts::Status Release() override {
    if (on_release) {
      on_release();
    }
    const bool freed = !place_.valid() || node_.memory().Free(place_).has_value();
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    if (!freed) {
      return std::unexpected("a model's place still has mappings");
    }
    return {};
  }

 private:
  ts::PagedNode& node_;
  std::uint32_t index_;
  bool fence_weights_ = false;
  bool host_first_ = false;
  std::vector<std::byte> file_;
  int fd_ = -1;
  jitllm::providers::ReservationId place_;
  std::uint64_t base_ = 0;
  std::vector<ExtentId> weights_;
  std::vector<ExtentId> staging_;
  std::byte* staging_bytes_ = nullptr;
  jitllm::catalog::Closure closure_;
  jitllm::catalog::Closure fence_;  // the staging, always resident, unless fence_weights_
};

TEST(CudaPagedNodeTest, TwoModelsAlternateAndEachEvictsOnlyTheOthersWeights) {
  const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path directory = scratch != nullptr
                                              ? std::filesystem::path(scratch)
                                              : std::filesystem::path(::testing::TempDir());
  std::filesystem::create_directories(directory);

  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<Model*, 2> models = {&first, &second};
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  for (Model* model : models) {
    model->Setup(directory);
  }
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  // B: what is resident now, the larger model's weights and half the
  // smaller's (rounded up to an extent).
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  const std::uint64_t budget = fixed + ((kExtents[0] + 2) * kExtent);
  ASSERT_TRUE(node.Start(Bytes(budget)).has_value());
  for (Model* model : models) {
    model->Register();
  }
  node.Run();

  for (int round = 0; round < 3 && ran; ++round) {
    for (std::uint32_t m = 0; m < 2 && ran; ++m) {
      Model& model = *models.at(m);
      const Model& other = *models.at(1 - m);
      ts::AcquireReport report;
      ran = node.Acquire(model.closure(), report, "an acquisition");
      if (!ran) {
        break;
      }
      const std::set<ExtentId> others(other.weights().begin(), other.weights().end());
      for (const ExtentId extent : report.evicted) {
        EXPECT_TRUE(others.contains(extent)) << "round " << round << " model " << m;
      }
      if (round > 0 || m > 0) {
        EXPECT_FALSE(report.evicted.empty()) << "round " << round << " model " << m;
      }
      ran = model.ReadBack();
      if (!ran) {
        break;
      }
      EXPECT_TRUE(model.Intact()) << "round " << round << " model " << m;
      std::uint64_t occupied = 0;
      std::size_t resident = 0;
      std::size_t others_resident = 0;
      ran = node.Call(
          [&]() -> ts::Status {
            occupied = node.catalog().OccupancyOf(node.domain()).Total().value();
            resident = model.Resident();
            others_resident = other.Resident();
            return {};
          },
          "reading the occupancy");
      EXPECT_LE(occupied, budget);
      EXPECT_EQ(resident, kExtents.at(m));
      EXPECT_LT(others_resident, kExtents.at(1 - m));
    }
  }
  EXPECT_TRUE(ran.has_value()) << ran.error();
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}

std::filesystem::path Scratch() {
  const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  std::filesystem::path directory = scratch != nullptr
                                        ? std::filesystem::path(scratch)
                                        : std::filesystem::path(::testing::TempDir());
  std::filesystem::create_directories(directory);
  return directory;
}

TEST(CudaPagedNodeTest, ManagedAcquireValidatesProtectsAndNeverFallsBackToCatalogLru) {
  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ASSERT_TRUE(node.Open());
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent));
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + kExtents[0] * kExtent)));
  first.Register();
  second.Register();
  node.Run();
  ASSERT_TRUE(first.ReadBack());
  ASSERT_TRUE(first.Intact());
  std::uint64_t calls = 0;
  bool take = false;
  node.SetReclaimer([&](std::uint64_t needed, jitllm::engine::PagedNode::ReclaimFor what,
                        std::span<const ExtentId> protect) -> std::uint64_t {
    ++calls;
    EXPECT_EQ(what, jitllm::engine::PagedNode::ReclaimFor::kMaterialize);
    EXPECT_EQ(needed, kExtents[1] * kExtent);
    for (const auto& [extent, generation] : second.closure().extents) {
      (void)generation;
      EXPECT_NE(std::ranges::find(protect, extent), protect.end());
    }
    if (take) {
      // Recursive acquisition refuses, without recursively entering this callback.
      bool nested_capacity = false;
      ts::AcquireReport nested;
      EXPECT_FALSE(
          node.Acquire(second.closure(), nested, "recursive acquisition", &nested_capacity));
      EXPECT_TRUE(nested_capacity);
      EXPECT_EQ(calls, 2U);
      std::vector<ExtentId> victims(first.weights().begin(), first.weights().begin() + 3);
      EXPECT_TRUE(node.Evict(std::move(victims)));
    }
    return 0;  // the caller must recheck actual occupancy, not trust this count
  });
  auto stale = second.closure();
  ++stale.extents.front().second;
  bool capacity = true;
  ts::AcquireReport report;
  EXPECT_FALSE(node.Acquire(stale, report, "stale acquisition", &capacity));
  EXPECT_FALSE(capacity);
  EXPECT_EQ(calls, 0U);
  ExtentId foreign;
  ASSERT_TRUE(node.Call(
      [&]() -> ts::Status {
        const auto domain = node.catalog().AddDomain("foreign acquisition control");
        auto added =
            node.catalog().AddExtent({.domain = domain,
                                      .memory_class = jitllm::catalog::MemoryClass::kRuntime,
                                      .recovery = jitllm::catalog::Recovery::kPinned,
                                      .size = Bytes(kExtent),
                                      .content = {}},
                                     true);
        if (!added) return std::unexpected("adding a foreign acquisition extent");
        foreign = *added;
        return {};
      },
      "constructing a foreign stale acquisition"));
  auto foreign_stale = second.closure();
  foreign_stale.extents.emplace_back(foreign, 2);
  EXPECT_FALSE(node.Acquire(foreign_stale, report, "foreign stale acquisition", &capacity));
  EXPECT_FALSE(capacity);
  EXPECT_EQ(calls, 0U);
  ASSERT_TRUE(node.Call(
      [&]() -> ts::Status {
        if (!node.catalog().ReleasePinned(foreign)) {
          return std::unexpected("releasing the foreign logical extent");
        }
        return {};
      },
      "retiring the foreign acquisition control"));
  EXPECT_FALSE(
      node.Acquire(second.closure(), report, "insufficient managed acquisition", &capacity));
  EXPECT_TRUE(capacity);
  EXPECT_EQ(calls, 1U);
  EXPECT_TRUE(report.evicted.empty());
  take = true;
  ASSERT_TRUE(node.Acquire(second.closure(), report, "managed partial acquisition", &capacity));
  EXPECT_FALSE(capacity);
  EXPECT_EQ(calls, 2U);
  ASSERT_TRUE(second.ReadBack());
  EXPECT_TRUE(second.Intact());
  bool meanwhile_ran = false;
  ASSERT_TRUE(node.Job(
      second.closure(), [](jitllm::providers::NativeStream) { return sc::JobResult::kQueued; },
      "in-flight acquisition guard", 1,
      [&]() {
        meanwhile_ran = true;
        ts::AcquireReport nested;
        bool refused_for_capacity = true;
        EXPECT_FALSE(
            node.Acquire({}, nested, "forbidden meanwhile acquisition", &refused_for_capacity));
        EXPECT_FALSE(refused_for_capacity);
      }));
  EXPECT_TRUE(meanwhile_ran);
  EXPECT_EQ(calls, 2U);
  node.SetReclaimer({});
  EXPECT_TRUE(node.TearDown(teardown));
}

// M3's full swap (SwapProgram) with the handoff (D-033) on the real VMM
// provider: each model's evicted backing is unmapped, kept, and mapped
// again under the other's weights, which are read back whole; what no load
// took is released once the swap is done, and teardown leaves no backing.
TEST(CudaPagedNodeTest, PartialSwapReleasesIncompatibleDonorBeforeHostFirstLoad) {
  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0);
  Model second(node, 1, false, true);
  const std::array<ts::PagedModel*, 2> teardown{&first, &second};
  ASSERT_TRUE(node.Open());
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent));
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  const auto budget = fixed + kExtents[0] * kExtent;
  ASSERT_TRUE(node.Start(Bytes(budget)));
  first.Register();
  second.Register();
  node.Run();
  EXPECT_FALSE(node.retired_harvest_stats());
  ASSERT_TRUE(node.BeginRequest(0, first.closure(), "the outgoing completed request"));
  ASSERT_TRUE(first.ReadBack());
  ASSERT_TRUE(first.Intact());
  std::vector<ExtentId> victims(first.weights().begin(), first.weights().begin() + 3);
  jitllm::catalog::Closure strict;
  ASSERT_TRUE(node.Call(
      [&]() -> ts::Status {
        strict = node.catalog().ClosureOfExtents(victims).value();
        return {};
      },
      "snapshotting selected policy victim generations"));
  sc::SwapReport refused;
  EXPECT_FALSE(node.Swap(victims, second.closure(), true, refused, {victims[0]}, strict));
  EXPECT_TRUE(node.InRequest(0));  // a held policy victim cannot end another request
  ASSERT_TRUE(node.EndRequest(0));
  const auto before = node.Stats();
  ASSERT_TRUE(before);
  sc::SwapReport report;
  ASSERT_TRUE(node.Swap(victims, second.closure(), true, report, {victims[0]}, strict));
  ASSERT_TRUE(second.ReadBack());
  EXPECT_TRUE(second.Intact());
  const auto after = node.Stats();
  ASSERT_TRUE(after);
  EXPECT_EQ(after->handed_off - before->handed_off, 2U);
  EXPECT_EQ(report.evictions, 3U);
  EXPECT_EQ(report.loads, 3U);
  ASSERT_TRUE(node.Call(
      [&]() -> ts::Status {
        EXPECT_EQ(first.Resident(), 1U);
        EXPECT_EQ(second.Resident(), 3U);
        EXPECT_LE(node.catalog().OccupancyOf(node.domain()).Total().value(), budget);
        return {};
      },
      "checking partial host-first occupancy and retained device weight"));
  EXPECT_TRUE(node.TearDown(teardown));
  const auto harvested = node.retired_harvest_stats();
  ASSERT_TRUE(harvested);
  EXPECT_GT(harvested->indices, 0U);
  EXPECT_GT(harvested->pop_locks, 0U);
  EXPECT_LE(harvested->pop_locks, harvested->indices);
  const auto final_creates = node.backing_create_stats();
  EXPECT_GT(final_creates.ordinary_attempts, 0U);
  EXPECT_GE(final_creates.ordinary_attempts, after->backing_creates.ordinary_attempts);
  EXPECT_EQ(final_creates.ordinary_failures, 0U);
  EXPECT_EQ(final_creates.reserve_failures, 0U);
}

TEST(CudaPagedNodeTest, SwapsHandBackingOverAndEveryByteReadsBack) {
  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  // B: room for the larger model's weights only, so no swap can create
  // backing beside what it evicts.
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + (kExtents[0] * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  ran = first.ReadBack();
  EXPECT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(first.Intact());
  const std::array<std::pair<Model*, Model*>, 2> swaps = {{{&first, &second}, {&second, &first}}};
  for (const auto& [out, in] : swaps) {
    if (!ran) {
      break;
    }
    const auto before = node.Stats();
    ASSERT_TRUE(before.has_value());
    ts::SwapReport report;
    ran = node.Swap(out->weights(), in->closure(), /*handoff=*/true, report);
    ASSERT_TRUE(ran.has_value()) << ran.error();
    ran = in->ReadBack();
    ASSERT_TRUE(ran.has_value()) << ran.error();
    EXPECT_TRUE(in->Intact());
    // What no load took is released after the swap: wait for it.
    std::size_t left = 1;
    for (int i = 0; i < 5000 && left > 0; ++i) {
      ASSERT_TRUE(node.Call(
                          [&]() -> ts::Status {
                            left = node.scheduler().evictions();
                            return {};
                          },
                          "counting evictions")
                      .has_value());
    }
    EXPECT_EQ(left, 0U);
    const auto after = node.Stats();
    ASSERT_TRUE(after.has_value());
    const std::size_t taken = std::min(out->weights().size(), in->weights().size());
    EXPECT_EQ(report.evictions, out->weights().size());
    EXPECT_EQ(report.loads, in->weights().size());
    EXPECT_EQ(after->parked - before->parked, out->weights().size());
    EXPECT_EQ(after->handed_off - before->handed_off, taken);
    EXPECT_EQ(after->released_unused - before->released_unused, out->weights().size() - taken);
    EXPECT_EQ(out->Resident(), 0U);
    EXPECT_EQ(in->Resident(), in->weights().size());
  }
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}

// Teardown fences every model's stream before it evicts or releases
// anything, a model swapped out included, without leasing that model's
// memory. Each model's fence closure here is its weights, nonresident once
// the first is swapped out (like a runner's spilled state). Without `room`
// for them beside the second model's, a fence that paged them in again
// was refused (as jitllm-runtime's stop once was with two models); with
// room, it read them back for nothing. Work queued on the first stream
// outside any job (a host function that ends late, as a launch context's
// own work may) has completed before that model's Release.
void TearDownAfterASwap(bool room) {
  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0, /*fence_weights=*/true);
  Model second(node, 1, /*fence_weights=*/true);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  const std::size_t extents = room ? kExtents[0] + kExtents[1] : kExtents[0];
  ASSERT_TRUE(node.Start(Bytes(fixed + (extents * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  ts::SwapReport report;
  ran = node.Swap(first.weights(), second.closure(), /*handoff=*/true, report);
  ASSERT_TRUE(ran.has_value()) << ran.error();
  ran = second.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(second.Intact());
  EXPECT_EQ(first.Resident(), 0U);

  // No job is in flight, so nothing else queues on the first stream now.
  std::atomic<bool> ended{false};
  const auto native = node.execution().Submission(node.stream(first.stream()));
  ASSERT_TRUE(native.has_value());
  ASSERT_EQ(cudaLaunchHostFunc(
                static_cast<cudaStream_t>(native->handle),
                [](void* flag) {
                  std::this_thread::sleep_for(std::chrono::milliseconds(500));
                  static_cast<std::atomic<bool>*>(flag)->store(true);
                },
                &ended),
            cudaSuccess);
  bool ended_before_release = false;
  const std::uint64_t reads = node.requests();
  std::uint64_t reads_at_release = 0;
  first.on_release = [&] {
    ended_before_release = ended.load();
    reads_at_release = node.requests();
  };
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
  EXPECT_TRUE(ended_before_release);
  EXPECT_EQ(reads_at_release, reads);  // nothing paged in again
  // The host function writes into this frame, fenced or not.
  for (int i = 0; i < 1000 && !ended.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(ended.load());
}

TEST(CudaPagedNodeTest, TearDownFencesASwappedOutModelsStreamWithNoRoomForIt) {
  TearDownAfterASwap(/*room=*/false);
}

TEST(CudaPagedNodeTest, TearDownFencesASwappedOutModelsStreamWithoutPagingItIn) {
  TearDownAfterASwap(/*room=*/true);
}

// M3's lease per request on the real providers: a request pages its
// model in and leases the closure once; each ReadBack is then a step under
// that lease (one lease per extent throughout, none taken per step), its
// bytes the file's; ending it leaves the model resident and unleased. A
// swap asked for between a request's steps ends the request first, then
// swaps, and the incoming model reads back whole under a request of its
// own.
// Both step paths: the driver's direct steps (the default), and steps
// through the request's task and the device lane.
class CudaPagedNodeRequest : public ::testing::TestWithParam<bool> {};
TEST_P(CudaPagedNodeRequest, ARequestLeasesOnceAndItsStepsRunUnderIt) {
  const bool direct = GetParam();
  ts::PagedNode node({.compute_streams = 2,
                      .slots = 4,
                      .inline_lanes = false,
                      .coalesce = false,
                      .direct_steps = direct});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + (kExtents[0] * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();

  const auto leases = [&](const Model& model, std::uint32_t& each, std::size_t& resident) {
    each = UINT32_MAX;
    return node.Call(
        [&]() -> ts::Status {
          resident = model.Resident();
          for (const ExtentId extent : model.weights()) {
            each = std::min(each, node.catalog().Describe(extent).value().leases);
          }
          return {};
        },
        "reading the leases");
  };
  ran = node.BeginRequest(0, first.closure(), "the first model's request");
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(node.InRequest(0));
  auto before = node.Stats();
  ASSERT_TRUE(before.has_value());
  const std::uint64_t direct_before = node.direct_steps();
  (void)node.TakeTimes(0);
  for (int step = 0; step < 4; ++step) {
    ran = first.ReadBack();
    ASSERT_TRUE(ran.has_value()) << ran.error();
    EXPECT_TRUE(first.Intact()) << "step " << step;
    std::uint32_t each = 0;
    std::size_t resident = 0;
    ASSERT_TRUE(leases(first, each, resident).has_value());
    EXPECT_EQ(each, 1U) << "step " << step;  // the request's, and only it
    EXPECT_EQ(resident, first.weights().size());
  }
  auto after = node.Stats();
  ASSERT_TRUE(after.has_value());
  // Each step one operation under the lease, or one of the driver's own.
  EXPECT_EQ(after->held_operations - before->held_operations, direct ? 0U : 4U);
  EXPECT_EQ(node.direct_steps() - direct_before, direct ? 4U : 0U);
  EXPECT_EQ(after->leases_held - before->leases_held, 0U);  // none per step
  const ts::StepTimes times = node.TakeTimes(0);
  EXPECT_EQ(times.steps, 4U);
  EXPECT_GT(times.device, 0.0);  // the events around each step
  EXPECT_LE(times.device, times.wall);
  // A step whose closure is not within the request's (the other model's)
  // is refused before it reaches the request's task, and runs nothing.
  std::atomic<bool> outside_ran{false};
  const ts::Status outside = node.Job(
      second.closure(),
      [&outside_ran](jitllm::providers::NativeStream /*native*/) {
        outside_ran.store(true);
        return sc::JobResult::kQueued;
      },
      "a step outside the request", 0);
  EXPECT_FALSE(outside.has_value());
  EXPECT_FALSE(outside_ran.load());
  EXPECT_TRUE(node.InRequest(0));
  ASSERT_TRUE(node.EndRequest(0).has_value());
  EXPECT_FALSE(node.InRequest(0));
  std::uint32_t each = 0;
  std::size_t resident = 0;
  ASSERT_TRUE(leases(first, each, resident).has_value());
  EXPECT_EQ(each, 0U);
  EXPECT_EQ(resident, first.weights().size());  // release is not eviction

  // A swap between the steps of a request over the outgoing model.
  ASSERT_TRUE(node.BeginRequest(0, first.closure(), "the first model's request").has_value());
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  ts::SwapReport report;
  ran = node.Swap(first.weights(), second.closure(), /*handoff=*/true, report);
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_EQ(report.requests_ended, 1U);
  EXPECT_FALSE(node.InRequest(0));
  EXPECT_EQ(report.evictions, first.weights().size());
  ASSERT_TRUE(node.BeginRequest(1, second.closure(), "the second model's request").has_value());
  ran = second.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(second.Intact());
  ASSERT_TRUE(leases(second, each, resident).has_value());
  EXPECT_EQ(each, 1U);
  EXPECT_EQ(resident, second.weights().size());
  // Left open: the teardown ends it before it evicts.
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}
INSTANTIATE_TEST_SUITE_P(StepPaths, CudaPagedNodeRequest, ::testing::Values(true, false));

TEST(CudaPagedNodeTest, HostLookaheadCompletesInsideHeldAndStandaloneJobsAndSkipsInline) {
  for (const bool inline_lanes : {false, true}) {
    ts::PagedNode node({.inline_lanes = inline_lanes});
    ASSERT_TRUE(node.Open());
    ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    ASSERT_TRUE(node.Start(Bytes(fixed + kExtent)));
    node.Run();
    for (const bool held : {false, true}) {
      const jitllm::catalog::Closure closure;
      if (held) ASSERT_TRUE(node.BeginRequest(0, closure, "host-only lookahead control"));
      // A direct step (a held request's, by default) runs the job's host
      // part on the driver first: the lookahead then overlaps its device
      // work only.
      const bool sequential = held && node.direct();
      for (const bool reject : {false, true}) {
        std::atomic<bool> entered{false}, release{inline_lanes || sequential}, finished{false};
        bool called = false;
        const auto ran = node.Job(
            closure,
            [&](jitllm::providers::NativeStream) {
              entered.store(true);
              const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
              while (!release.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
              finished.store(true);
              return reject ? sc::JobResult::kNotStarted : sc::JobResult::kQueued;
            },
            "host-only lookahead control", 0,
            [&] {
              called = true;
              const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
              while (!entered.load() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
              EXPECT_TRUE(entered.load());
              EXPECT_EQ(finished.load(), sequential);
              release.store(true);
            });
        EXPECT_EQ(bool(ran), !reject);
        EXPECT_EQ(called, !inline_lanes);
        EXPECT_TRUE(entered.load());
        EXPECT_TRUE(finished.load());
      }
      if (held) ASSERT_TRUE(node.EndRequest(0));
    }
    EXPECT_TRUE(node.TearDown({}));
  }
}

// RE-029 on the real device: a job whose stream fills (a wait on a host
// flag, then more operations than the stream holds pending) blocks the
// device lane's submission thread in a launch. With the zone's copies on a
// copy lane of their own (the paged node's default), and fences from the
// provider's pool of events (cuEventCreate blocks for as long as that
// launch does), a page-in completes meanwhile; the job finishes once the
// flag is set.
TEST(CudaPagedNodeTest, ACopyLaneLandsPageInsWhileAJobFillsItsStream) {
  ts::PagedNode node({.compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + ((kExtents[0] + kExtents[1]) * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  ran = first.ReadBack();  // the first model's weights, resident
  ASSERT_TRUE(ran.has_value()) << ran.error();

  void* flag = nullptr;
  ASSERT_EQ(cudaHostAlloc(&flag, sizeof(std::uint32_t), cudaHostAllocMapped), cudaSuccess);
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(0);
  void* device_flag = nullptr;
  ASSERT_EQ(cudaHostGetDevicePointer(&device_flag, flag, 0), cudaSuccess);
  constexpr int kOperations = 1500;  // well past the ~1,020 a stream holds (RE-029)
  std::atomic<int> queued{0};
  const std::uint64_t workspace = node.activations().base;
  ts::Done done;
  const std::uint64_t request = node.Submit(std::make_unique<ts::RunProgram>(
      done, first.closure(),
      [&queued, device_flag, workspace](jitllm::providers::NativeStream native) {
        if (cuStreamWaitValue32(static_cast<CUstream>(native.handle),
                                reinterpret_cast<CUdeviceptr>(device_flag), 1,
                                CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS) {
          return sc::JobResult::kUnknown;
        }
        for (int i = 0; i < kOperations; ++i) {
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          if (cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0, 4,
                              static_cast<cudaStream_t>(native.handle)) != cudaSuccess) {
            return sc::JobResult::kUnknown;
          }
          queued.store(i + 1);
        }
        return sc::JobResult::kQueued;
      },
      0));
  // The job's launches stop short: its stream is full and the device
  // lane's thread is blocked in the next one.
  for (int i = 0; i < 1000 && queued.load() < 1000; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_GE(queued.load(), 1000);
  EXPECT_LT(queued.load(), kOperations);
  // A watchdog opens the gate after 30 s, so a regression fails instead of
  // hanging.
  std::atomic<bool> fired{false};
  std::jthread watchdog([&](const std::stop_token& stop) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < until) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!stop.stop_requested()) {
      fired = true;
      std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(1);
    }
  });
  std::vector<ts::LoadStats> log;
  const auto start = std::chrono::steady_clock::now();
  ran = node.Load(second.weights(), "a page-in beside a full stream", log);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  EXPECT_TRUE(ran.has_value()) << ran.error();
  EXPECT_FALSE(fired.load()) << "the page-in waited for the job's stream";
  EXPECT_FALSE(done.gone.load());  // the job is still held
  EXPECT_LT(queued.load(), kOperations);
  std::println("page-in of {} bytes beside a full stream: {:.4f} s",
               second.weights().size() * kExtent, seconds);
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(1);
  watchdog.request_stop();
  const ts::Status job = node.Await(done, "the gated job", request);
  EXPECT_TRUE(job.has_value()) << job.error();
  EXPECT_EQ(queued.load(), kOperations);
  ran = second.ReadBack();
  EXPECT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(second.Intact());
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
  (void)cudaFreeHost(flag);
}

double ProcessCpu() {
  timespec ts{};
  (void)::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + (static_cast<double>(ts.tv_nsec) / 1e9);
}

// The runtime wake on the real device (docs/experiments/runtime-wake/): a
// request's steps, each gated on a host flag the test opens when it
// chooses, so a step ends on time, early, late or at once. Every step
// completes and returns (no completion is lost while the lanes, the
// scheduler and the driver sleep between and during steps); as a step
// nears its expected end, the scheduler and the device lane are told to
// poll ahead of it (the relay), after the completion lane slept through
// most of it; stepping costs well under a core, where polling threads
// took two, and nothing spins once the request has ended.
class CudaPagedNodeSleep : public ::testing::TestWithParam<bool> {};
TEST_P(CudaPagedNodeSleep, RequestStepsSleepBetweenAndLoseNoCompletion) {
  // Direct steps tell no lane to poll (none is between a step and its
  // driver), so the relay is looked for only through the task.
  const bool direct = GetParam();
  ts::PagedNode node({.compute_streams = 2,
                      .slots = 4,
                      .inline_lanes = false,
                      .coalesce = false,
                      .direct_steps = direct});
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + ((kExtents[0] + kExtents[1]) * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  ran = node.BeginRequest(0, first.closure(), "the gated request");
  ASSERT_TRUE(ran.has_value()) << ran.error();

  void* flag = nullptr;
  ASSERT_EQ(cudaHostAlloc(&flag, sizeof(std::uint32_t), cudaHostAllocMapped), cudaSuccess);
  std::atomic_ref<std::uint32_t> gate(*static_cast<std::uint32_t*>(flag));
  gate.store(0);
  void* device_flag = nullptr;
  ASSERT_EQ(cudaHostGetDevicePointer(&device_flag, flag, 0), cudaSuccess);
  // Milliseconds from each step's launch to its gate opening; -1 opens it
  // once the scheduler and the device lane are told to poll (the relay),
  // looked for from 5 ms on, once the last step's relay and follow window
  // have lapsed. Each of the stream's last eight fence lengths is a likely
  // end, so the relay step follows six of 20 ms (and nothing shorter).
  constexpr std::array<int, 12> kDelays = {20, 20, 20, 20, 20, 20, -1, 0, 60, 1, 20, 20};
  std::atomic<std::uint32_t> launched{0};
  std::vector<double> relays;  // ms from the launch to the relay, the gate's thread's
  std::atomic<bool> unrelayed{false};
  std::jthread opener([&](const std::stop_token& stop) {
    for (std::uint32_t s = 0; s < kDelays.size() && !stop.stop_requested(); ++s) {
      while (launched.load() < s + 1 && !stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
      const auto at = std::chrono::steady_clock::now();
      if (kDelays.at(s) < 0 && direct) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      } else if (kDelays.at(s) < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const auto give_up = at + std::chrono::seconds(5);
        while (!(node.wake().Anticipating(std::chrono::steady_clock::now()) &&
                 node.device_lane().Anticipating(std::chrono::steady_clock::now())) &&
               std::chrono::steady_clock::now() < give_up) {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        const auto relayed = std::chrono::steady_clock::now();
        if (relayed >= give_up) {
          unrelayed = true;
        }
        relays.push_back(std::chrono::duration<double, std::milli>(relayed - at).count());
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(kDelays.at(s)));
      }
      gate.store(s + 1);
    }
  });
  const std::uint64_t workspace = node.activations().base;
  std::vector<double> lags;  // ms from each direct step's end to its return
  (void)node.TakeTimes(0);
  const double cpu = ProcessCpu();
  const auto start = std::chrono::steady_clock::now();
  for (std::uint32_t s = 0; s < kDelays.size(); ++s) {
    ran = node.Job(
        first.closure(),
        [&launched, device_flag, workspace, s](jitllm::providers::NativeStream native) {
          if (cuStreamWaitValue32(static_cast<CUstream>(native.handle),
                                  reinterpret_cast<CUdeviceptr>(device_flag), s + 1,
                                  CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS ||
              // NOLINTNEXTLINE(performance-no-int-to-ptr)
              cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0, 4,
                              static_cast<cudaStream_t>(native.handle)) != cudaSuccess) {
            return sc::JobResult::kUnknown;
          }
          launched.store(s + 1);
          return sc::JobResult::kQueued;
        },
        "a gated step", 0);
    ASSERT_TRUE(ran.has_value()) << "step " << s << ": " << ran.error();
    // A direct step's driver sleeps toward the step's likely end, from the
    // device's own spans: it sees the end soon after it, whether the gate
    // opens on time, early, late or at once (a step's device span includes
    // its wait on the gate). The first has nothing to go by.
    const ts::StepTimes times = node.TakeTimes(0);
    if (direct && s > 0) {
      lags.push_back((times.wall - times.device) * 1e3);
      EXPECT_LT(lags.back(), 3.0) << "step " << s << " seen late";
    }
  }
  const double stepping =
      (ProcessCpu() - cpu) /
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  opener.join();
  EXPECT_FALSE(unrelayed.load()) << "a step neared its expected end with no relay";
  for (const double relay : relays) {
    EXPECT_GE(relay, 10.0) << "relayed at once: the completion lane never slept";  // expected ~20
  }
  EXPECT_LT(stepping, 1.0) << "cores busy while stepping";
  ASSERT_TRUE(node.EndRequest(0).has_value());
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // past every window
  const double idle_cpu = ProcessCpu();
  const auto idle_start = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const double idle =
      (ProcessCpu() - idle_cpu) /
      std::chrono::duration<double>(std::chrono::steady_clock::now() - idle_start).count();
  EXPECT_LT(idle, 0.2) << "cores busy with nothing to do";
  std::println("gated steps: {:.2f} cores stepping, {:.3f} idle; direct lags (ms): {}", stepping,
               idle, lags);
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
  (void)cudaFreeHost(flag);
}
INSTANTIATE_TEST_SUITE_P(StepPaths, CudaPagedNodeSleep, ::testing::Values(true, false));

// A model's own ring whose read never completed (Qwen3.8's n-gram rows
// after a stall), retired at the model's end: the ring and the pinned
// landing that read writes outlive the node's teardown, so the read can
// still land after it without writing freed memory. A ring with nothing
// in flight is simply destroyed.
TEST(CudaPagedNodeTest, AStalledReadKeepsItsLandingPastTeardown) {
  ts::PagedNode node({.compute_streams = 1, .slots = 4, .inline_lanes = false, .coalesce = false});
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  constexpr std::uint32_t kBytes = 4096;
  std::vector<ExtentId> staging;
  auto pinned = node.Pinned(kBytes, 0, staging);
  ASSERT_TRUE(pinned.has_value()) << pinned.error();
  auto* const landing = static_cast<std::byte*>(*pinned);
  std::memset(landing, 0, kBytes);

  auto storage = std::make_unique<jitllm::providers::fake::FakeStorage>(4, kBytes);
  jitllm::providers::fake::FakeStorage* const ring = storage.get();
  const int fd = ring->AddFile(std::vector<std::byte>(kBytes, std::byte{0x5a}));
  ring->ScriptNext({.submission = jitllm::providers::Submission::kAccepted,
                    .result = std::nullopt,
                    .hold = true});
  ASSERT_EQ(ring->Submit({.token = 7,
                          .kind = jitllm::providers::IoKind::kRead,
                          .fd = fd,
                          .offset = 0,
                          .memory = landing,
                          .length = kBytes,
                          .segments = {}}),
            jitllm::providers::Submission::kAccepted);
  const std::array<void*, 1> landings = {landing};
  EXPECT_TRUE(node.RetireRing(std::move(storage), landings));
  EXPECT_EQ(node.kept_pinned(), 1U);
  EXPECT_FALSE(
      node.RetireRing(std::make_unique<jitllm::providers::fake::FakeStorage>(4, kBytes), landings));
  const std::array<ts::PagedModel*, 0> none{};
  const ts::Status finished = node.TearDown(none);
  EXPECT_TRUE(finished.has_value()) << finished.error();
  // The read lands after the teardown, into memory still allocated.
  ASSERT_TRUE(ring->Release(7));
  std::array<jitllm::providers::IoCompletion, 1> done{};
  ASSERT_EQ(ring->Harvest(done, false), 1U);
  EXPECT_EQ(done[0].result, kBytes);
  EXPECT_EQ(landing[kBytes - 1], std::byte{0x5a});
  EXPECT_EQ(ring->in_flight(), 0U);
}

// ------------------------------------------------- D-102's hang recovery

// The node's settings for a hang test: two streams, and the node's own
// patience over `quiet`; held reads cancellable (as reads still queued) or
// not (as a drive's).
ts::NodeSettings HangSettings(std::chrono::milliseconds quiet,
                              const std::atomic<bool>* hold = nullptr, bool cancellable = true) {
  ts::NodeSettings settings{
      .compute_streams = 2, .slots = 4, .inline_lanes = false, .coalesce = false};
  settings.quiet = quiet;
  settings.hold_reads = hold;
  settings.hold_cancellable = cancellable;
  return settings;
}

// The runtime ladder's shape, with short times, recorded: a wait whose node
// made no progress for `quiet` is cancelled (rung 1); one whose
// cancellation has not drained `quiet` after it with nothing moving is
// where the ladder would restart the process (rung 3): recorded, and the
// wait goes on (the test then frees it).
class RecordingPatience final : public jitllm::engine::Patience {
 public:
  explicit RecordingPatience(std::chrono::milliseconds quiet) : quiet_(quiet) {}
  void Begin() override { ++begun; }
  void End() override { ++ended; }
  jitllm::engine::WaitVerdict Check(const jitllm::engine::WaitState& wait,
                                    std::uint64_t /*progress*/) override {
    const auto now = std::chrono::steady_clock::now();
    if (!wait.cancelled) {
      if (now - wait.progressed < quiet_) {
        return jitllm::engine::WaitVerdict::kWait;
      }
      ++cancels;
      return jitllm::engine::WaitVerdict::kCancel;
    }
    if (now - std::max(wait.progressed, *wait.cancelled) >= quiet_) {
      gave_up.store(true);
    }
    return jitllm::engine::WaitVerdict::kWait;
  }
  std::atomic<int> begun{0};
  std::atomic<int> ended{0};
  std::atomic<int> cancels{0};
  std::atomic<bool> gave_up{false};

 private:
  std::chrono::milliseconds quiet_;
};

// Rung 1 on the real device: a page-in whose reads hang (held by the
// storage's test hook, as a stuck drive would) makes no progress; the
// node's patience cancels the load's request, the held reads complete as
// cancelled, and the load fails, flagged as a hang's. Nothing else is
// lost: once reads flow again the same weights page in and read back
// whole, and the teardown releases everything. A healthy load is never
// cut short.
TEST(CudaPagedNodeTest, AHungPageInIsCancelledAndTheNodeGoesOn) {
  std::atomic<bool> hold{false};
  ts::PagedNode node(HangSettings(std::chrono::milliseconds(500), &hold));
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + ((kExtents[0] + kExtents[1]) * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  std::vector<ts::LoadStats> log;
  ran = node.Load(second.weights(), "a healthy page-in", log);
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_FALSE(node.TakeHangCancelled());

  hold.store(true);
  const auto start = std::chrono::steady_clock::now();
  ran = node.Load(first.weights(), "a hung page-in", log);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().find("a hang"), std::string::npos) << ran.error();
  EXPECT_TRUE(node.TakeHangCancelled());
  EXPECT_FALSE(node.TakeHangCancelled());  // taken
  EXPECT_EQ(node.hang_cancels(), 1U);
  EXPECT_GT(node.counting().held.load(), 0U);
  EXPECT_GE(seconds, 0.5);
  EXPECT_LT(seconds, 30.0);
  std::println("a hung page-in cancelled and drained after {:.3f} s", seconds);

  hold.store(false);
  ran = node.Load(first.weights(), "the page-in again", log);
  ASSERT_TRUE(ran.has_value()) << ran.error();
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(first.Intact());
  ran = second.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(second.Intact());
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}

// A read the drive holds is not ended by its cancellation (io_uring's is
// best effort): the hung page-in's wait still returns, failed and flagged
// as a hang's, but the read stays in flight, which the node's oldest_io
// shows (the hang ladder keeps the cancellation undrained on it, so rung 2
// never reuses what the read lands in: rung 3 instead). Once the drive
// lets it go, it completes and the node goes on.
TEST(CudaPagedNodeTest, AStuckReadOutlivesItsCancelledWait) {
  std::atomic<bool> hold{false};
  ts::PagedNode node(HangSettings(std::chrono::minutes(10), &hold, false));
  RecordingPatience patience(std::chrono::milliseconds(300));
  node.SetPatience(&patience);
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + ((kExtents[0] + kExtents[1]) * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  EXPECT_FALSE(node.oldest_io().has_value());

  hold.store(true);
  const auto start = std::chrono::steady_clock::now();
  // The drive lets the reads go after 5 s (a regression that waits for
  // them then fails instead of hanging).
  std::jthread releaser([&](const std::stop_token& stop) {
    const auto until = start + std::chrono::seconds(5);
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < until) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    hold.store(false);
  });
  std::vector<ts::LoadStats> log;
  ran = node.Load(first.weights(), "a page-in on a stuck drive", log);
  const auto returned = std::chrono::steady_clock::now();
  const auto oldest = node.oldest_io();
  const double seconds = std::chrono::duration<double>(returned - start).count();
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().find("a hang"), std::string::npos) << ran.error();
  EXPECT_TRUE(node.TakeHangCancelled());
  EXPECT_EQ(patience.cancels.load(), 1);
  EXPECT_LT(seconds, 4.5);  // the wait returned while the read was held
  ASSERT_TRUE(oldest.has_value());
  EXPECT_LE(oldest.value_or(returned), start + std::chrono::seconds(1));
  std::println(
      "a page-in on a stuck drive: its wait returned after {:.3f} s with a read still "
      "in flight",
      seconds);
  releaser.join();
  // The reads complete once let go.
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (node.oldest_io().has_value() && std::chrono::steady_clock::now() < until) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_FALSE(node.oldest_io().has_value());
  ran = node.Load(first.weights(), "the page-in again", log);
  ASSERT_TRUE(ran.has_value()) << ran.error();
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(first.Intact());
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}

// Rung 1 at a request's lease: its materialization hangs on held reads,
// the lease wait cancels, and BeginRequest fails flagged as a hang's; the
// request opens once reads flow.
TEST(CudaPagedNodeTest, AHungLeaseIsCancelledAndTheRequestOpensLater) {
  std::atomic<bool> hold{true};
  ts::PagedNode node(HangSettings(std::chrono::milliseconds(500), &hold));
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + ((kExtents[0] + kExtents[1]) * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  ran = node.BeginRequest(0, first.closure(), "a hung lease");
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().find("a hang"), std::string::npos) << ran.error();
  EXPECT_TRUE(node.TakeHangCancelled());
  EXPECT_FALSE(node.InRequest(0));
  hold.store(false);
  ran = node.BeginRequest(0, first.closure(), "the lease again");
  ASSERT_TRUE(ran.has_value()) << ran.error();
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(first.Intact());
  ran = node.EndRequest(0);
  EXPECT_TRUE(ran.has_value()) << ran.error();
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
}

// Rung 3's case on the real device: a job whose stream waits on a gate
// that never opens (a hung kernel) cannot be cancelled: the wait cancels
// its request, nothing drains, and the patience reaches the point where the
// runtime's ladder restarts the process (recorded here; the default
// patience aborts). Opened then, the stream completes, the wait drains and
// fails flagged as a hang's, and the node goes on.
TEST(CudaPagedNodeTest, AHungStreamCannotBeCancelledSoItsPatienceGivesUp) {
  ts::PagedNode node(HangSettings(std::chrono::minutes(10)));
  RecordingPatience patience(std::chrono::milliseconds(300));
  node.SetPatience(&patience);
  Model first(node, 0);
  Model second(node, 1);
  const std::array<ts::PagedModel*, 2> teardown = {&first, &second};
  ts::Status ran = node.Open();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  first.Setup(Scratch());
  second.Setup(Scratch());
  ASSERT_TRUE(node.MapWorkspace(kExtent, kExtent).has_value());
  const std::uint64_t fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(Bytes(fixed + ((kExtents[0] + kExtents[1]) * kExtent))).has_value());
  first.Register();
  second.Register();
  node.Run();
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();

  void* flag = nullptr;
  ASSERT_EQ(cudaHostAlloc(&flag, sizeof(std::uint32_t), cudaHostAllocMapped), cudaSuccess);
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(0);
  void* device_flag = nullptr;
  ASSERT_EQ(cudaHostGetDevicePointer(&device_flag, flag, 0), cudaSuccess);
  // The gate opens once the patience gave up, or after 30 s (a regression
  // then fails instead of hanging).
  std::atomic<bool> timed_out{false};
  std::jthread opener([&](const std::stop_token& stop) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!stop.stop_requested() && !patience.gave_up.load() &&
           std::chrono::steady_clock::now() < until) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    timed_out = !patience.gave_up.load();
    std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(1);
  });
  const auto gated = [device_flag](jitllm::providers::NativeStream native) {
    if (cuStreamWaitValue32(static_cast<CUstream>(native.handle),
                            reinterpret_cast<CUdeviceptr>(device_flag), 1,
                            CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS) {
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  ran = node.Job(first.closure(), gated, "a hung job", 0);
  opener.join();
  ASSERT_FALSE(ran.has_value());
  EXPECT_NE(ran.error().find("a hang"), std::string::npos) << ran.error();
  EXPECT_FALSE(timed_out.load());
  EXPECT_TRUE(patience.gave_up.load());
  EXPECT_EQ(patience.cancels.load(), 1);
  EXPECT_TRUE(node.TakeHangCancelled());

  // The same within a request: the step's wait cancels its request; the
  // step completes once the gate opens, too late to stop it, and the
  // cancellation is still flagged.
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(0);
  patience.gave_up.store(false);
  ran = node.BeginRequest(0, first.closure(), "a request");
  ASSERT_TRUE(ran.has_value()) << ran.error();
  std::jthread opener2([&](const std::stop_token& stop) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!stop.stop_requested() && !patience.gave_up.load() &&
           std::chrono::steady_clock::now() < until) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(flag)).store(1);
  });
  std::ignore = node.Job(first.closure(), gated, "a hung step", 0);
  opener2.join();
  EXPECT_TRUE(patience.gave_up.load());
  EXPECT_EQ(patience.cancels.load(), 2);
  EXPECT_TRUE(node.TakeHangCancelled());
  if (node.InRequest(0)) {
    std::ignore = node.EndRequest(0);  // its task was cancelled: it may report so
  }
  EXPECT_GT(patience.begun.load(), 0);
  EXPECT_EQ(patience.begun.load(), patience.ended.load());  // every wait told its end

  // The node goes on: the weights read back whole.
  ran = first.ReadBack();
  ASSERT_TRUE(ran.has_value()) << ran.error();
  EXPECT_TRUE(first.Intact());
  const ts::Status finished = node.TearDown(teardown);
  EXPECT_TRUE(finished.has_value()) << finished.error();
  (void)cudaFreeHost(flag);
}

}  // namespace
