// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Native preparation integration: cataloged VMM, bounded pinned staging and
// actual device-lane Jobs. In-memory shard IO is deliberately synchronous;
// these controls do not claim filesystem, model or performance qualification.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "engine/dsv4_ds4_weights.h"
#include "engine/paged_node.h"
#include "engine/runner_resources.h"
#include "expected_error.h"
#include "kernels/ggml/launch.h"
#include "providers/device_execution.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"

namespace {
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
namespace pd = jitllm::providers;
namespace sc = jitllm::scheduler;
namespace ca = jitllm::catalog;
using jitllm::base::Bytes;
using jitllm::test_support::Failed;

constexpr std::size_t kPage = 4096;
constexpr std::size_t kGuard = 256;
constexpr std::size_t kCapacity = 4096;
constexpr std::size_t kDeviceSpan = kCapacity + (2 * kGuard);
constexpr std::size_t kHostBytes = 3 * kPage;
constexpr std::byte kPoison{0xa5};

void* Pointer(std::uint64_t address) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr): a native mapped VMM address
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(address));
}

// A failed GPU operation has an unknown outcome. Keep every owner alive by
// terminating the test process rather than unwinding through its teardown.
void Require(en::Status result) {
  if (!result) {
    ADD_FAILURE() << result.error();
    std::abort();
  }
}

void RequireKnownFailure(en::Status result, std::string_view error) {
  if (result) {
    ADD_FAILURE() << "expected the synchronous refusal: " << error;
  } else if (result.error() != error) {
    ADD_FAILURE() << "unexpected preparation outcome: " << result.error();
    std::abort();
  }
}

struct Fields {
  std::size_t columns = 0;
  std::size_t blocks = 0;
  std::size_t raw_block = 0;
  std::size_t scales = 0;
  std::size_t codes = 0;
  std::size_t packed = 0;
};

// Derive physical field sections without using the production layout helper.
// Ragged section lengths exercise the deterministic zero padding as well as
// the permutation. All fixture geometries are fixed and small.
Fields ReferenceFields(const kg::Ds4AlignedShape& shape) {
  const auto columns = static_cast<std::size_t>(shape.input / 256);
  const auto blocks = columns * shape.output * shape.groups;
  const auto aligned = [](std::size_t bytes) { return ((bytes + 63) / 64) * 64; };
  if (shape.kind == kg::Ds4AlignedKind::kQ2K) {
    const auto pairs = blocks / 2;
    const auto scales = aligned(pairs * 8);
    const auto codes = scales + aligned(pairs * 32);
    return {columns, blocks, 84, scales, codes, codes + (pairs * 128)};
  }
  const auto codes = aligned(blocks * 2);
  return {columns, blocks, 66, 0, codes, codes + (blocks * 64)};
}

std::vector<std::byte> RawBytes(const kg::Ds4AlignedShape& shape, std::size_t salt) {
  const auto fields = ReferenceFields(shape);
  std::vector<std::byte> raw(fields.blocks * fields.raw_block);
  constexpr std::array<std::uint16_t, 8> kHalfBits = {0x0000, 0x8000, 0x0001, 0x7c00,
                                                      0x7d55, 0xfeab, 0x3555, 0xfbff};
  for (std::size_t block = 0; block < fields.blocks; ++block) {
    for (std::size_t byte = 0; byte < fields.raw_block; ++byte) {
      raw[(block * fields.raw_block) + byte] =
          static_cast<std::byte>(((block * 73) + (byte * 19) + (salt * 29)) & 255U);
    }
    const auto half = [&](std::size_t at, std::uint16_t bits) {
      raw[(block * fields.raw_block) + at] = static_cast<std::byte>(bits & 255U);
      raw[(block * fields.raw_block) + at + 1] = static_cast<std::byte>(bits >> 8U);
    };
    const auto index = (block + salt) % kHalfBits.size();
    if (shape.kind == kg::Ds4AlignedKind::kQ2K) {
      half(80, kHalfBits[index]);
      half(82, kHalfBits[(index + 3) % kHalfBits.size()]);
    } else {
      half(0, kHalfBits[index]);
    }
  }
  return raw;
}

// Walk expert, output-row pair and input-column fields, not the CUDA kernel's
// flattened thread mapping. Half values are copied as bytes, including NaNs.
std::vector<std::byte> ReferencePacked(const kg::Ds4AlignedShape& shape,
                                       const std::vector<std::byte>& raw) {
  const auto fields = ReferenceFields(shape);
  std::vector<std::byte> packed(fields.packed, std::byte{0});
  const auto copy = [&](std::size_t from, std::size_t to, std::size_t bytes) {
    std::memcpy(packed.data() + to, raw.data() + from, bytes);
  };
  if (shape.kind == kg::Ds4AlignedKind::kIq2Xxs) {
    for (std::size_t block = 0; block < fields.blocks; ++block) {
      copy(block * 66, block * 2, 2);
      copy((block * 66) + 2, fields.codes + (block * 64), 64);
    }
    return packed;
  }
  for (std::size_t expert = 0; expert < shape.groups; ++expert) {
    for (std::size_t pair = 0; pair < shape.output / 2; ++pair) {
      for (std::size_t column = 0; column < fields.columns; ++column) {
        const auto target = (((expert * (shape.output / 2)) + pair) * fields.columns) + column;
        for (std::size_t side = 0; side < 2; ++side) {
          const auto block =
              (((expert * shape.output) + (pair * 2) + side) * fields.columns) + column;
          copy((block * 84) + 80, (target * 8) + (side * 4), 4);
          for (std::size_t window = 0; window < 2; ++window) {
            copy((block * 84) + (window * 8),
                 fields.scales + (target * 32) + (window * 16) + (side * 8), 8);
          }
          for (std::size_t word = 0; word < 16; ++word) {
            copy((block * 84) + 16 + (word * 4),
                 fields.codes + (target * 128) + (word * 8) + (side * 4), 4);
          }
        }
      }
    }
  }
  return packed;
}

struct Shards {
  en::Ds4PreparedWeight tensor;
  std::array<std::vector<std::byte>, 2> files;
  std::vector<std::byte> expected;
};

Shards MakeShards(kg::Ds4AlignedShape shape, std::size_t salt) {
  const auto fields = ReferenceFields(shape);
  const auto raw = RawBytes(shape, salt);
  const auto layout = kg::Ds4AlignedLayoutOf(shape);
  if (!layout) std::abort();
  EXPECT_EQ(layout->packed_bytes.value(), fields.packed);
  EXPECT_EQ(layout->raw_bytes.value(), raw.size());
  Shards result{.tensor = {.name = "test experts",
                           .index = 0,
                           .expert_array = true,
                           .shape = shape,
                           .layout = *layout,
                           .sources = {}},
                .files = {},
                .expected = ReferencePacked(shape, raw)};
  constexpr std::array<std::uint32_t, 2> kShardIds = {7, 29};
  constexpr std::array<std::size_t, 2> kOffsets = {kPage + 3968, (3 * kPage) + 512};
  constexpr std::array<std::size_t, 2> kFileBytes = {3 * kPage, 5 * kPage};
  const auto bytes_per_expert = raw.size() / 2;
  for (std::size_t expert = 0; expert < result.files.size(); ++expert) {
    auto& file = result.files[expert];
    file.resize(kFileBytes[expert]);
    for (std::size_t i = 0; i < file.size(); ++i) {
      file[i] = static_cast<std::byte>(((i * 11) + (expert * 97) + salt) & 255U);
    }
    std::memcpy(file.data() + kOffsets[expert], raw.data() + (expert * bytes_per_expert),
                bytes_per_expert);
    result.tensor.sources.push_back({.shard = kShardIds[expert],
                                     .file_offset = kOffsets[expert],
                                     .bytes = bytes_per_expert,
                                     .file_end = file.size()});
  }
  return result;
}

class PreparationModel final : public en::PagedModel {
 public:
  explicit PreparationModel(en::PagedNode& node) : resources(node, 0, 0) {}

  std::uint32_t stream() const override { return 0; }
  const ca::Closure& fence_closure() const override { return closure; }
  // These fixed pinned mappings are released by RunnerResources, not evicted
  // as scheduler-owned pageable weight backing.
  std::vector<ca::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    std::vector<std::string> problems;
    resources.Release(problems);
    released = true;
    if (!problems.empty()) return std::unexpected(problems.front());
    return {};
  }

  en::Mapped raw, packed;
  en::RunnerResources resources;
  ca::Closure closure;
  bool released = false;
};

// Fence the second compute stream too: creating even an unused launch
// context notes a submission with the provider. It owns no additional memory.
class SecondStream final : public en::PagedModel {
 public:
  explicit SecondStream(const ca::Closure& closure) : closure_(closure) {}
  std::uint32_t stream() const override { return 1; }
  const ca::Closure& fence_closure() const override { return closure_; }
  std::vector<ca::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override { return {}; }

 private:
  const ca::Closure& closure_;
};

// Distinct provider object, identical underlying stream ID and CUDA stream.
// Identity refusal must happen before forwarding any preparation work.
class ForwardingExecution final : public pd::DeviceExecution {
 public:
  explicit ForwardingExecution(pd::DeviceExecution& inner) : inner_(inner) {}
  std::expected<pd::StreamId, pd::Failure> CreateStream() override { return inner_.CreateStream(); }
  std::expected<void, pd::Failure> DestroyStream(pd::StreamId stream) override {
    return inner_.DestroyStream(stream);
  }
  std::expected<void, pd::Failure> Copy(pd::StreamId stream, std::uint64_t destination,
                                        std::uint64_t source, Bytes size) override {
    return inner_.Copy(stream, destination, source, size);
  }
  std::expected<pd::NativeStream, pd::Failure> Submission(pd::StreamId stream) override {
    return inner_.Submission(stream);
  }
  std::expected<void, pd::Failure> Wait(pd::StreamId stream, pd::FenceId fence) override {
    return inner_.Wait(stream, fence);
  }
  std::expected<pd::FenceId, pd::Failure> Record(pd::StreamId stream) override {
    return inner_.Record(stream);
  }
  std::expected<pd::FenceState, pd::Failure> Query(pd::FenceId fence) override {
    return inner_.Query(fence);
  }
  std::expected<void, pd::Failure> Release(pd::FenceId fence) override {
    return inner_.Release(fence);
  }

 private:
  pd::DeviceExecution& inner_;
};

struct ReadCall {
  std::uint32_t shard = 0;
  std::uint64_t offset = 0;
  std::size_t bytes = 0;
};

struct IoLog {
  std::vector<ReadCall> reads;
  std::vector<std::pair<std::uint64_t, std::size_t>> writes;
  std::size_t completed_slots = 0;
  std::size_t written_bytes = 0;
  std::vector<std::byte> output;
};

class Ds4WeightsGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Require(node_.Open());
    Require(model_.resources.Map(model_.raw, "preparation raw", kDeviceSpan,
                                 ca::MemoryClass::kScratch));
    Require(model_.resources.Map(model_.packed, "preparation packed", kDeviceSpan,
                                 ca::MemoryClass::kWeights));
    const auto host = model_.resources.Pinned(kHostBytes);
    if (!host) {
      ADD_FAILURE() << host.error();
      std::abort();
    }
    host_ = static_cast<std::byte*>(*host);
    Require(node_.MapWorkspace(en::kPagedExtent, en::kPagedExtent));
    Require(model_.resources.BindLaunch(0));
    const auto fixed = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    Require(node_.Start(Bytes(fixed + (2 * en::kPagedExtent))));
    auto extents = model_.resources.extents();
    extents.insert(extents.end(), node_.pool().extents.begin(), node_.pool().extents.end());
    extents.insert(extents.end(), node_.activations().extents.begin(),
                   node_.activations().extents.end());
    const auto closure = node_.catalog().ClosureOfExtents(extents);
    if (!closure) std::abort();
    model_.closure = *closure;
    node_.Run();
    // Setup and poisoning run through the actual device lane, not the default
    // CUDA stream. The returned Job proves both fills and the saved handle.
    PoisonRanges();
    Require(node_.Call(
        [&]() -> en::Status {
          EXPECT_EQ(node_.Covered(model_.raw.base, kDeviceSpan, 0, true),
                    ca::MemoryClass::kScratch);
          EXPECT_EQ(node_.Covered(model_.packed.base, kDeviceSpan, 0, true),
                    ca::MemoryClass::kWeights);
          std::size_t staging = 0;
          for (const auto& entry : model_.closure.extents) {
            const auto view = node_.catalog().Describe(entry.first);
            EXPECT_TRUE(view.has_value());
            if (view) {
              EXPECT_EQ(view->state, ca::ExtentState::kResident);
              if (view->descriptor.memory_class == ca::MemoryClass::kStaging) {
                EXPECT_EQ(view->descriptor.size.value(), kHostBytes);
                EXPECT_EQ(view->descriptor.recovery, ca::Recovery::kPinned);
                ++staging;
              }
            }
          }
          EXPECT_EQ(staging, 1U);
          return {};
        },
        "checking managed preparation memory"));
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(host_) % kPage, 0U);
    cudaPointerAttributes attributes{};
    ASSERT_EQ(cudaPointerGetAttributes(&attributes, host_), cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  }

  void TearDown() override {
    const std::array<en::PagedModel*, 2> models = {&model_, &second_};
    Require(node_.TearDown(models));
    EXPECT_TRUE(model_.released);
    EXPECT_TRUE(model_.raw.backings.empty());
    EXPECT_TRUE(model_.packed.backings.empty());
    EXPECT_EQ(node_.memory().backings(), 0U);
  }

  void PoisonRanges() {
    Require(node_.Job(
        model_.closure,
        [&](pd::NativeStream native) {
          native_ = native;
          if (!pd::FillAsync(native, Pointer(model_.raw.base), 0xa5, kDeviceSpan).ok() ||
              !pd::FillAsync(native, Pointer(model_.packed.base), 0xa5, kDeviceSpan).ok()) {
            return sc::JobResult::kUnknown;
          }
          return sc::JobResult::kQueued;
        },
        "poisoning preparation guards", 0));
  }

  en::Ds4WeightPreparationBuffers Buffers(const Shards& data) const {
    return {.host = host_,
            .host_bytes = kHostBytes,
            .raw = {.address = model_.raw.base + kGuard, .bytes = kCapacity},
            .packed = {.address = model_.packed.base + kGuard, .bytes = data.expected.size()}};
  }

  void CompletedSlot(IoLog& log) const {
    const auto queried = cudaStreamQuery(static_cast<cudaStream_t>(native_.handle));
    if (queried != cudaSuccess) {
      ADD_FAILURE() << "preparation exposed an unfinished staging slot: "
                    << cudaGetErrorString(queried);
      std::abort();
    }
    ++log.completed_slots;
  }

  en::Ds4WeightPreparationIo Io(const Shards& data, IoLog& log, std::size_t fail_read = 0,
                                std::size_t fail_write = 0) {
    log.output.assign(data.expected.size(), std::byte{0xee});
    return {.read = [&, fail_read](std::uint32_t shard, std::uint64_t offset,
                                   std::span<std::byte> target) -> en::Status {
              CompletedSlot(log);
              log.reads.push_back({shard, offset, target.size()});
              if (target.data() != host_ || target.size() > kHostBytes) {
                return std::unexpected("test reader received another staging slot");
              }
              std::fill_n(host_, kHostBytes, std::byte{0xdd});
              if (log.reads.size() == fail_read)
                return std::unexpected("intentional source read failure");
              const auto source =
                  std::ranges::find(data.tensor.sources, shard, &en::Ds4WeightSource::shard);
              if (source == data.tensor.sources.end())
                return std::unexpected("test reader received another shard");
              const auto index = static_cast<std::size_t>(source - data.tensor.sources.begin());
              const auto& file = data.files[index];
              if (offset > file.size() || target.size() > file.size() - offset) {
                return std::unexpected("test reader received an out-of-bounds direct read");
              }
              EXPECT_EQ(offset % kPage, 0U);
              EXPECT_EQ(target.size() % kPage, 0U);
              std::memcpy(target.data(), file.data() + static_cast<std::size_t>(offset),
                          target.size());
              return {};
            },
            .write = [&, fail_write](std::uint64_t offset,
                                     std::span<const std::byte> source) -> en::Status {
              CompletedSlot(log);
              log.writes.emplace_back(offset, source.size());
              if (source.data() != host_ || offset != log.written_bytes ||
                  offset > log.output.size() || source.size() > log.output.size() - offset) {
                return std::unexpected(
                    "test writer received another, incomplete or reordered staging slot");
              }
              if (log.writes.size() == fail_write)
                return std::unexpected("intentional writer failure");
              std::memcpy(log.output.data() + static_cast<std::size_t>(offset), source.data(),
                          source.size());
              log.written_bytes += source.size();
              // Once the callback returns no consumer may still read this slot.
              // Poison it immediately after the completed write, before reuse.
              std::fill_n(host_, kHostBytes, std::byte{0x6d});
              return {};
            }};
  }

  en::Status Prepare(const Shards& data, std::size_t limit, const en::Ds4WeightPreparationIo& io) {
    return en::PrepareDs4Weight(node_, model_.resources.launch(), model_.closure, 0, data.tensor,
                                limit, Buffers(data), io);
  }

  std::vector<std::byte> ReadBox(std::uint64_t base) {
    Require(node_.Job(
        model_.closure,
        [&](pd::NativeStream native) {
          const auto copied =
              pd::CopyAsync(native, host_, Pointer(base), kDeviceSpan, pd::CopyKind::kDeviceToHost);
          return copied.ok() ? sc::JobResult::kQueued : sc::JobResult::kUnknown;
        },
        "reading preparation guards", 0));
    return {host_, host_ + kDeviceSpan};
  }

  void CheckOutputAndGuards(const Shards& data, std::size_t limit, const IoLog& log) {
    EXPECT_EQ(log.output, data.expected);
    EXPECT_EQ(log.written_bytes, data.expected.size());
    EXPECT_EQ(log.completed_slots, log.reads.size() + log.writes.size());
    const auto raw = ReadBox(model_.raw.base);
    const auto packed = ReadBox(model_.packed.base);
    EXPECT_TRUE(std::all_of(raw.begin(), raw.begin() + kGuard,
                            [](std::byte byte) { return byte == kPoison; }));
    EXPECT_TRUE(std::all_of(raw.begin() + static_cast<std::ptrdiff_t>(kGuard + limit), raw.end(),
                            [](std::byte byte) { return byte == kPoison; }));
    EXPECT_TRUE(std::all_of(packed.begin(), packed.begin() + kGuard,
                            [](std::byte byte) { return byte == kPoison; }));
    EXPECT_TRUE(
        std::all_of(packed.begin() + static_cast<std::ptrdiff_t>(kGuard + data.expected.size()),
                    packed.end(), [](std::byte byte) { return byte == kPoison; }));
    EXPECT_TRUE(std::equal(data.expected.begin(), data.expected.end(), packed.begin() + kGuard));
  }

  en::PagedNode node_{en::NodeSettings{.compute_streams = 2, .slots = 2}};
  PreparationModel model_{node_};
  SecondStream second_{model_.closure};
  std::byte* host_ = nullptr;
  pd::NativeStream native_{};
};

TEST_F(Ds4WeightsGpuTest, NoncontiguousExpertSlicesPreserveEveryFieldAndPaddingAcrossPartitions) {
  constexpr std::array<kg::Ds4AlignedShape, 2> kShapes = {
      kg::Ds4AlignedShape{
          .kind = kg::Ds4AlignedKind::kIq2Xxs, .input = 256, .output = 3, .groups = 2},
      kg::Ds4AlignedShape{
          .kind = kg::Ds4AlignedKind::kQ2K, .input = 512, .output = 6, .groups = 2}};
  for (const auto& shape : kShapes) {
    SCOPED_TRACE(static_cast<int>(shape.kind));
    const auto data = MakeShards(shape, 1);
    const std::size_t small_limit = shape.kind == kg::Ds4AlignedKind::kQ2K ? 400 : 100;
    std::vector<std::byte> split_output;
    for (const auto limit : {small_limit, kCapacity}) {
      PoisonRanges();
      IoLog log;
      const auto io = Io(data, log);
      Require(Prepare(data, limit, io));
      CheckOutputAndGuards(data, limit, log);
      ASSERT_EQ(log.reads.size(), limit == small_limit ? 6U : 2U);
      const auto half = log.reads.size() / 2;
      for (std::size_t i = 0; i < half; ++i) EXPECT_EQ(log.reads[i].shard, 7U);
      for (std::size_t i = half; i < log.reads.size(); ++i) EXPECT_EQ(log.reads[i].shard, 29U);
      EXPECT_EQ(log.reads.front().offset, kPage);
      EXPECT_EQ(log.reads[half].offset, 3 * kPage);
      EXPECT_TRUE(std::ranges::any_of(
          log.reads, [](const ReadCall& read) { return read.bytes == 2 * kPage; }));
      EXPECT_EQ(log.writes.size(), (data.expected.size() + limit - 1) / limit);
      if (limit == small_limit)
        split_output = log.output;
      else
        EXPECT_EQ(log.output, split_output);
    }
  }
}

TEST_F(Ds4WeightsGpuTest, RepeatedPreparationReadsCurrentShardBytesAtTheSameMappedAddresses) {
  constexpr kg::Ds4AlignedShape kShape{
      .kind = kg::Ds4AlignedKind::kQ2K, .input = 512, .output = 6, .groups = 2};
  const auto first = MakeShards(kShape, 1);
  const auto current = MakeShards(kShape, 23);
  ASSERT_NE(first.expected, current.expected);
  const auto addresses = Buffers(first);
  IoLog initial;
  const auto initial_io = Io(first, initial);
  Require(Prepare(first, 400, initial_io));
  CheckOutputAndGuards(first, 400, initial);
  PoisonRanges();
  IoLog repeated;
  const auto repeated_io = Io(current, repeated);
  Require(Prepare(current, 400, repeated_io));
  CheckOutputAndGuards(current, 400, repeated);
  EXPECT_EQ(Buffers(current).raw.address, addresses.raw.address);
  EXPECT_EQ(Buffers(current).packed.address, addresses.packed.address);
  EXPECT_EQ(Buffers(current).host, addresses.host);
  EXPECT_EQ(repeated.reads.size(), initial.reads.size());
  EXPECT_NE(repeated.output, initial.output);
}

TEST_F(Ds4WeightsGpuTest, SourceReadFailureRetiresEarlierJobsAndTheSlotCanBeReused) {
  constexpr kg::Ds4AlignedShape kShape{
      .kind = kg::Ds4AlignedKind::kIq2Xxs, .input = 256, .output = 3, .groups = 2};
  const auto data = MakeShards(kShape, 3);
  IoLog failed;
  const auto failing_io = Io(data, failed, 2);
  RequireKnownFailure(Prepare(data, 100, failing_io), "intentional source read failure");
  EXPECT_EQ(failed.reads.size(), 2U);
  EXPECT_TRUE(failed.writes.empty());
  EXPECT_EQ(failed.completed_slots, 2U);
  EXPECT_TRUE(
      std::ranges::all_of(failed.output, [](std::byte byte) { return byte == std::byte{0xee}; }));
  const auto current = MakeShards(kShape, 19);
  PoisonRanges();
  IoLog retried;
  const auto retry_io = Io(current, retried);
  Require(Prepare(current, 100, retry_io));
  CheckOutputAndGuards(current, 100, retried);
}

TEST_F(Ds4WeightsGpuTest, WriterFailureRetiresTheDownloadAndReportsOnlyCompletedOutput) {
  constexpr kg::Ds4AlignedShape kShape{
      .kind = kg::Ds4AlignedKind::kQ2K, .input = 512, .output = 6, .groups = 2};
  const auto data = MakeShards(kShape, 5);
  IoLog failed;
  const auto failing_io = Io(data, failed, 0, 2);
  RequireKnownFailure(Prepare(data, 400, failing_io), "intentional writer failure");
  EXPECT_EQ(failed.reads.size(), 6U);
  EXPECT_EQ(failed.writes.size(), 2U);
  EXPECT_EQ(failed.completed_slots, 8U);
  EXPECT_EQ(failed.written_bytes, 400U);
  EXPECT_TRUE(
      std::equal(data.expected.begin(), data.expected.begin() + 400, failed.output.begin()));
  EXPECT_TRUE(std::all_of(failed.output.begin() + 400, failed.output.end(),
                          [](std::byte byte) { return byte == std::byte{0xee}; }));
  PoisonRanges();
  IoLog retried;
  const auto retry_io = Io(data, retried);
  Require(Prepare(data, 400, retry_io));
  CheckOutputAndGuards(data, 400, retried);
}

TEST_F(Ds4WeightsGpuTest, AnotherProviderOrStreamIsRefusedBeforeAnyJobOrIo) {
  const auto data =
      MakeShards({.kind = kg::Ds4AlignedKind::kIq2Xxs, .input = 256, .output = 3, .groups = 2}, 7);
  ForwardingExecution forwarding(node_.execution());
  auto other_provider =
      kg::LaunchContext::Create(0, forwarding, node_.stream(0), {.size = Bytes(0)});
  auto other_stream =
      kg::LaunchContext::Create(0, node_.execution(), node_.stream(1), {.size = Bytes(0)});
  ASSERT_TRUE(other_provider.has_value())
      << Failed(other_provider, &kg::KernelFailure::detail).value_or("");
  ASSERT_TRUE(other_stream.has_value())
      << Failed(other_stream, &kg::KernelFailure::detail).value_or("");
  (void)node_.TakeTimes(0);
  for (auto* context : {other_provider->get(), other_stream->get()}) {
    IoLog log;
    const auto io = Io(data, log);
    RequireKnownFailure(en::PrepareDs4Weight(node_, *context, model_.closure, 0, data.tensor, 100,
                                             Buffers(data), io),
                        "invalid caller-owned bounded preparation buffers/IO");
    EXPECT_TRUE(log.reads.empty());
    EXPECT_TRUE(log.writes.empty());
    EXPECT_EQ(node_.TakeTimes(0).steps, 0U);
    EXPECT_FALSE(context->faulted());
  }
  const auto raw = ReadBox(model_.raw.base);
  const auto packed = ReadBox(model_.packed.base);
  EXPECT_TRUE(std::ranges::all_of(raw, [](std::byte byte) { return byte == kPoison; }));
  EXPECT_TRUE(std::ranges::all_of(packed, [](std::byte byte) { return byte == kPoison; }));
}

TEST_F(Ds4WeightsGpuTest, CaptureIsRefusedBeforeAnyJobOrIoAndLeavesTheContextUsable) {
  const auto data =
      MakeShards({.kind = kg::Ds4AlignedKind::kQ2K, .input = 512, .output = 6, .groups = 2}, 9);
  IoLog log;
  const auto io = Io(data, log);
  (void)node_.TakeTimes(0);
  bool entered = false;
  auto& context = model_.resources.launch();
  Require(node_.Job(
      model_.closure,
      [&](pd::NativeStream) {
        // Capture itself belongs to the device submission lane. The preparation
        // helper must refuse synchronously rather than post a nested Job.
        const auto captured = context.Capture(
            [&](kg::LaunchContext& active) -> std::expected<void, kg::KernelFailure> {
              entered = true;
              EXPECT_TRUE(active.capturing());
              RequireKnownFailure(en::PrepareDs4Weight(node_, active, model_.closure, 0,
                                                       data.tensor, 400, Buffers(data), io),
                                  "invalid caller-owned bounded preparation buffers/IO");
              return std::unexpected(kg::KernelFailure{.error = kg::KernelError::kRejected,
                                                       .detail = "expected preparation refusal"});
            });
        EXPECT_EQ(Failed(captured, &kg::KernelFailure::error), kg::KernelError::kRejected);
        return context.faulted() ? sc::JobResult::kUnknown : sc::JobResult::kQueued;
      },
      "checking preparation capture refusal", 0));
  EXPECT_TRUE(entered);
  EXPECT_EQ(node_.TakeTimes(0).steps, 1U);  // the outer capture control only
  EXPECT_TRUE(log.reads.empty());
  EXPECT_TRUE(log.writes.empty());
  EXPECT_FALSE(context.capturing());
  EXPECT_FALSE(context.faulted());
  const auto packed = ReadBox(model_.packed.base);
  EXPECT_TRUE(std::ranges::all_of(packed, [](std::byte byte) { return byte == kPoison; }));
  IoLog after;
  const auto after_io = Io(data, after);
  Require(Prepare(data, 400, after_io));
  CheckOutputAndGuards(data, 400, after);
}

}  // namespace
