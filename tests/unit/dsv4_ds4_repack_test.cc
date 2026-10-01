// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Independent field-byte permutation and inverse controls. No floating-point
// conversion is used: even signaling NaNs in half scales must retain every bit.
// These are importer component controls, not model or performance evidence.
#include "kernels/ggml/dsv4_ds4_repack.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/launch.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace pd = jitllm::providers;

constexpr std::size_t kGuardBytes = 256;

struct Fields {
  std::size_t blocks;
  std::size_t raw_block;
  std::size_t scales;
  std::size_t codes;
  std::size_t packed;
};

// Derive sections from their documented field counts, independently of the
// production layout helper. Tiny test geometry keeps all host sizes bounded.
Fields ReferenceFields(const kg::Ds4AlignedShape& shape) {
  const auto blocks = static_cast<std::size_t>(
                          shape.input / (shape.kind == kg::Ds4AlignedKind::kQ8Dense ? 32 : 256)) *
                      shape.output * shape.groups;
  const auto padded = [](std::size_t bytes) { return ((bytes + 63) / 64) * 64; };
  if (shape.kind == kg::Ds4AlignedKind::kQ2K) {
    const auto pairs = blocks / 2;
    const auto scales = padded(pairs * 8);
    const auto codes = scales + padded(pairs * 32);
    return {blocks, 84, scales, codes, codes + (pairs * 128)};
  }
  const auto payload = shape.kind == kg::Ds4AlignedKind::kIq2Xxs ? 64U : 32U;
  const auto codes = padded(blocks * 2);
  return {blocks, payload + 2, 0, codes, codes + (blocks * payload)};
}

std::vector<std::uint8_t> RawFixture(const kg::Ds4AlignedShape& shape, std::size_t salt) {
  const auto fields = ReferenceFields(shape);
  std::vector<std::uint8_t> raw(fields.blocks * fields.raw_block);
  constexpr std::array<std::uint16_t, 16> kScaleBits = {
      0x0000, 0x8000, 0x0001, 0x03ff, 0x3c00, 0xbc00, 0x7bff, 0xfbff,
      0x7c00, 0xfc00, 0x7e01, 0x7d55, 0xfeab, 0xfd23, 0x3555, 0xb555};
  for (std::size_t block = 0; block < fields.blocks; ++block) {
    for (std::size_t byte = 0; byte < fields.raw_block; ++byte)
      raw[(block * fields.raw_block) + byte] =
          static_cast<std::uint8_t>(((block * 73) + (byte * 19) + (salt * 29)) & 255);
    const auto set_half = [&](std::size_t offset, std::uint16_t bits) {
      raw[(block * fields.raw_block) + offset] = static_cast<std::uint8_t>(bits & 255);
      raw[(block * fields.raw_block) + offset + 1] = static_cast<std::uint8_t>(bits >> 8);
    };
    const auto index = ((block * 3) + salt) % kScaleBits.size();
    if (shape.kind == kg::Ds4AlignedKind::kQ2K) {
      set_half(80, kScaleBits[index]);
      set_half(82, kScaleBits[(index + 7) % kScaleBits.size()]);
    } else {
      set_half(0, kScaleBits[index]);
    }
  }
  return raw;
}

// Walk semantic expert / paired output-row / input-column fields. This does
// not reproduce the device's flattened thread indexing or lane branching.
// The inverse uses the same physical field contract and reconstructs all raw
// bytes, while padding is independently required to stay zero.
void TransferReference(const kg::Ds4AlignedShape& shape, std::vector<std::uint8_t>& raw,
                       std::vector<std::uint8_t>& packed, std::size_t first, std::size_t count,
                       bool inverse) {
  const auto fields = ReferenceFields(shape);
  const auto copy = [&](std::size_t raw_offset, std::size_t packed_offset, std::size_t bytes) {
    if (inverse)
      std::memcpy(raw.data() + raw_offset, packed.data() + packed_offset, bytes);
    else
      std::memcpy(packed.data() + packed_offset, raw.data() + raw_offset, bytes);
  };
  if (shape.kind != kg::Ds4AlignedKind::kQ2K) {
    for (std::size_t block = first; block < first + count; ++block) {
      copy(block * fields.raw_block, block * 2, 2);
      copy((block * fields.raw_block) + 2, fields.codes + (block * (fields.raw_block - 2)),
           fields.raw_block - 2);
    }
    return;
  }
  const auto columns = static_cast<std::size_t>(shape.input / 256);
  for (std::size_t expert = 0; expert < shape.groups; ++expert)
    for (std::size_t pair = 0; pair < shape.output / 2; ++pair)
      for (std::size_t column = 0; column < columns; ++column) {
        const auto target = (((expert * (shape.output / 2)) + pair) * columns) + column;
        for (std::size_t side = 0; side < 2; ++side) {
          const auto block = (((expert * shape.output) + (pair * 2) + side) * columns) + column;
          if (block < first || block >= first + count) continue;
          copy((block * 84) + 80, (target * 8) + (side * 4), 4);
          for (std::size_t window = 0; window < 2; ++window)
            copy((block * 84) + (window * 8),
                 fields.scales + (target * 32) + (window * 16) + (side * 8), 8);
          for (std::size_t word = 0; word < 16; ++word)
            copy((block * 84) + 16 + (word * 4),
                 fields.codes + (target * 128) + (word * 8) + (side * 4), 4);
        }
      }
}

class Ds4RepackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto opened = pd::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened.has_value());
    execution_ = std::move(*opened);
    auto stream = execution_->CreateStream();
    ASSERT_TRUE(stream.has_value());
    stream_ = *stream;
    auto launch =
        kg::LaunchContext::Create(0, *execution_, stream_, {.size = jitllm::base::Bytes(0)});
    ASSERT_TRUE(launch.has_value());
    launch_ = std::move(*launch);
  }

  void TearDown() override {
    if (stream_.valid()) Finish();
    if (launch_) {
      EXPECT_EQ(launch_->scratch_peak().value(), 0);
      launch_.reset();
    }
    if (stream_.valid()) EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : allocations_) EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }

  // Unknown completion keeps all owners alive until the test process dies.
  // Never destroy the stream or free operands after a timeout/error.
  void Finish() {
    const auto fence = execution_->Record(stream_);
    if (!fence) {
      ADD_FAILURE() << fence.error().detail;
      std::abort();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
      const auto queried = execution_->Query(*fence);
      if (!queried || std::chrono::steady_clock::now() >= deadline) {
        ADD_FAILURE() << "repack test could not prove GPU completion";
        std::abort();
      }
      if (*queried == pd::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution_->Release(*fence)) std::abort();
  }

  kg::Ds4CacheBuffer Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes + (2 * kGuardBytes)) != cudaSuccess) std::abort();
    allocations_.push_back(pointer);
    if (cudaMemset(pointer, 0xa5, bytes + (2 * kGuardBytes)) != cudaSuccess) std::abort();
    // Setup uses the default stream; the consumer stream is nonblocking.
    // Prove setup complete, keeping every allocation owned on failure.
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
    return {reinterpret_cast<std::uintptr_t>(pointer) + kGuardBytes, bytes};
  }

  template <typename T>
  void Replace(const kg::Ds4CacheBuffer& buffer, std::span<const T> values) {
    Finish();
    if (buffer.bytes != values.size_bytes() ||
        cudaMemcpy(std::bit_cast<void*>(buffer.address), values.data(), values.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      std::abort();
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
  }

  template <typename T>
  kg::Ds4CacheBuffer Upload(std::span<const T> values) {
    const auto buffer = Allocate(values.size_bytes());
    Replace<T>(buffer, values);
    return buffer;
  }

  template <typename T>
  std::vector<T> Download(const kg::Ds4CacheBuffer& buffer) {
    Finish();
    std::vector<T> values(static_cast<std::size_t>(buffer.bytes) / sizeof(T));
    if (cudaMemcpy(values.data(), std::bit_cast<const void*>(buffer.address), buffer.bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      std::abort();
    return values;
  }

  void Guard(const kg::Ds4CacheBuffer& buffer) {
    Finish();
    std::array<std::uint8_t, kGuardBytes> before{};
    std::array<std::uint8_t, kGuardBytes> after{};
    ASSERT_EQ(cudaMemcpy(before.data(), std::bit_cast<const void*>(buffer.address - kGuardBytes),
                         before.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(after.data(), std::bit_cast<const void*>(buffer.address + buffer.bytes),
                         after.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(before, [](auto byte) { return byte == 0xa5; }));
    EXPECT_TRUE(std::ranges::all_of(after, [](auto byte) { return byte == 0xa5; }));
  }

  static void Queued(const std::expected<void, kg::KernelFailure>& result) {
    if (!result) {
      ADD_FAILURE() << result.error().detail;
      std::abort();  // A failed launch does not prove that no work was queued.
    }
  }

  template <typename T>
  void Exact(const std::vector<T>& a, const std::vector<T>& b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(T)), 0);
  }

  kg::LaunchContext& launch() { return *launch_; }

 private:
  std::unique_ptr<pd::DeviceExecution> execution_;
  pd::StreamId stream_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<void*> allocations_;
};

struct ChunkRange {
  std::size_t first;
  std::size_t blocks;
};

class Ds4RepackBytesTest : public Ds4RepackTest {
 protected:
  void Verify(kg::Ds4AlignedShape shape, std::span<const ChunkRange> chunks) {
    const auto fields = ReferenceFields(shape);
    auto raw = RawFixture(shape, 0);
    const auto source = Upload<std::uint8_t>(raw);
    const auto single = Allocate(fields.packed);
    const auto chunked = Allocate(fields.packed);
    const auto actual_layout = kg::Ds4AlignedLayoutOf(shape);
    ASSERT_TRUE(actual_layout);
    ASSERT_EQ(actual_layout->blocks, fields.blocks);
    ASSERT_EQ(actual_layout->raw_bytes.value(), raw.size());
    ASSERT_EQ(actual_layout->packed_bytes.value(), fields.packed);
    ASSERT_EQ(actual_layout->scales_offset, fields.scales);
    ASSERT_EQ(actual_layout->codes_offset, fields.codes);
    Queued(kg::RunDs4RepackInitialize(launch(), shape, single));
    Queued(kg::RunDs4RepackChunk(launch(), {.shape = shape,
                                            .raw = source,
                                            .packed = single,
                                            .first_block = 0,
                                            .blocks = fields.blocks}));
    std::vector<std::uint8_t> expected(fields.packed, 0);
    TransferReference(shape, raw, expected, 0, fields.blocks, false);
    const auto once = Download<std::uint8_t>(single);
    Exact(once, expected);
    auto packed_for_inverse = once;
    std::vector<std::uint8_t> reconstructed(raw.size(), 0);
    TransferReference(shape, reconstructed, packed_for_inverse, 0, fields.blocks, true);
    Exact(raw, reconstructed);
    Queued(kg::RunDs4RepackInitialize(launch(), shape, chunked));
    std::vector<std::uint8_t> partial(fields.packed, 0);
    for (const auto& chunk : chunks) {
      const kg::Ds4CacheBuffer view{source.address + (chunk.first * fields.raw_block),
                                    chunk.blocks * fields.raw_block};
      Queued(kg::RunDs4RepackChunk(launch(), {.shape = shape,
                                              .raw = view,
                                              .packed = chunked,
                                              .first_block = chunk.first,
                                              .blocks = chunk.blocks}));
      TransferReference(shape, raw, partial, chunk.first, chunk.blocks, false);
      // Every untouched block and every alignment gap remains zero.
      Exact(Download<std::uint8_t>(chunked), partial);
    }
    Exact(Download<std::uint8_t>(chunked), once);
    Exact(Download<std::uint8_t>(source), raw);
    Queued(kg::RunDs4RepackInitialize(launch(), shape, single));
    Queued(kg::RunDs4RepackChunk(launch(), {.shape = shape,
                                            .raw = source,
                                            .packed = single,
                                            .first_block = 0,
                                            .blocks = fields.blocks}));
    Exact(Download<std::uint8_t>(single), once);
    Guard(source);
    Guard(single);
    Guard(chunked);
  }
};

TEST_F(Ds4RepackBytesTest, Iq2RaggedGridPreservesAllScaleBitsAndOutOfOrderChunks) {
  constexpr std::array<ChunkRange, 3> kChunks = {{{28, 46}, {0, 11}, {11, 17}}};
  Verify({kg::Ds4AlignedKind::kIq2Xxs, 256, 37, 2}, kChunks);
}

TEST_F(Ds4RepackBytesTest, Q8RaggedGridPreservesSignedCodesAndRawNanScales) {
  constexpr std::array<ChunkRange, 3> kChunks = {{{34, 53}, {0, 1}, {1, 33}}};
  Verify({kg::Ds4AlignedKind::kQ8Dense, 96, 29, 1}, kChunks);
}

TEST_F(Ds4RepackBytesTest, Q2WholeRowPairChunksRetainTwoExpertsAndAbsoluteOffsets) {
  // The first submitted chunk spans expert0's final pair and expert1's first.
  constexpr std::array<ChunkRange, 3> kChunks = {{{12, 12}, {24, 12}, {0, 12}}};
  Verify({kg::Ds4AlignedKind::kQ2K, 768, 6, 2}, kChunks);
}

TEST_F(Ds4RepackTest, CapturedRepackReadsCurrentRawBytesAndKeepsBothPaddingGaps) {
  const kg::Ds4AlignedShape shape{kg::Ds4AlignedKind::kQ2K, 768, 6, 1};
  const auto fields = ReferenceFields(shape);
  auto raw = RawFixture(shape, 0);
  const auto source = Upload<std::uint8_t>(raw);
  const auto packed = Allocate(fields.packed);
  const auto eager = Allocate(fields.packed);
  const kg::Ds4RepackChunk desc{
      .shape = shape, .raw = source, .packed = packed, .first_block = 0, .blocks = fields.blocks};
  const auto untouched = Download<std::uint8_t>(packed);
  auto graph = launch().Capture([&](auto& context) -> std::expected<void, kg::KernelFailure> {
    if (auto initialized = kg::RunDs4RepackInitialize(context, shape, packed); !initialized)
      return initialized;
    return kg::RunDs4RepackChunk(context, desc);
  });
  if (!graph) {
    ADD_FAILURE() << graph.error().detail;
    std::abort();
  }
  EXPECT_GT(graph->nodes(), 0);
  Exact(Download<std::uint8_t>(packed), untouched);  // Capture queued no operand work.
  Queued(launch().Launch(*graph));
  Finish();
  std::vector<std::uint8_t> expected(fields.packed, 0);
  TransferReference(shape, raw, expected, 0, fields.blocks, false);
  Exact(Download<std::uint8_t>(packed), expected);
  const auto previous = expected;
  raw = RawFixture(shape, 1);
  Replace<std::uint8_t>(source, raw);
  Queued(launch().Launch(*graph));
  Finish();
  std::ranges::fill(expected, 0);
  TransferReference(shape, raw, expected, 0, fields.blocks, false);
  EXPECT_NE(expected, previous);
  Exact(Download<std::uint8_t>(packed), expected);
  auto eager_desc = desc;
  eager_desc.packed = eager;
  Queued(kg::RunDs4RepackInitialize(launch(), shape, eager));
  Queued(kg::RunDs4RepackChunk(launch(), eager_desc));
  Exact(Download<std::uint8_t>(eager), expected);
  Exact(Download<std::uint8_t>(source), raw);
  Guard(source);
  Guard(packed);
  Guard(eager);
  Finish();  // Destroy the graph before its stream or borrowed operand owners.
}

TEST_F(Ds4RepackTest, RefusedChunksLeaveSourceAndCompleteOutputUntouched) {
  const kg::Ds4AlignedShape shape{kg::Ds4AlignedKind::kIq2Xxs, 256, 37, 2};
  const auto fields = ReferenceFields(shape);
  const auto raw = RawFixture(shape, 0);
  const auto source = Upload<std::uint8_t>(raw);
  const auto packed = Allocate(fields.packed);
  const auto before = Download<std::uint8_t>(packed);
  kg::Ds4RepackChunk desc{
      .shape = shape, .raw = source, .packed = packed, .first_block = 0, .blocks = fields.blocks};
  --desc.raw.bytes;
  const auto short_view = kg::RunDs4RepackChunk(launch(), desc);
  ASSERT_FALSE(short_view);
  EXPECT_EQ(short_view.error().error, kg::KernelError::kRejected);
  desc.raw = packed;
  desc.blocks = 1;  // Both views are individually large enough; alias is the refusal.
  const auto alias = kg::RunDs4RepackChunk(launch(), desc);
  ASSERT_FALSE(alias);
  EXPECT_EQ(alias.error().error, kg::KernelError::kRejected);
  Exact(Download<std::uint8_t>(source), raw);
  Exact(Download<std::uint8_t>(packed), before);
  EXPECT_FALSE(launch().faulted());
  desc.raw = source;
  desc.blocks = fields.blocks;
  Queued(kg::RunDs4RepackInitialize(launch(), shape, packed));
  Queued(kg::RunDs4RepackChunk(launch(), desc));
  auto expected_raw = raw;
  std::vector<std::uint8_t> expected(fields.packed, 0);
  TransferReference(shape, expected_raw, expected, 0, fields.blocks, false);
  Exact(Download<std::uint8_t>(packed), expected);
  Guard(source);
  Guard(packed);
}

}  // namespace
