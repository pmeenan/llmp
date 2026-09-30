// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original cache formats, not a model quality test. Fixed analytical QAT
// cases check ties/scales/signs; byte roundtrips and allocation guards check
// the producer/consumer layout. Ring writes check IEEE-F16 rounding and
// chronological wrap; capture refusal prevents stale host positions.
#include "kernels/ggml/dsv4_ds4_cache.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace pd = jitllm::providers;

constexpr std::size_t kGuardBytes = 256;

class Ds4CacheTest : public ::testing::Test {
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
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
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

  // If quiescence cannot be proved, abort this test process and leave
  // every GPU allocation owned until that boundary. Never free on timeout.
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
        ADD_FAILURE() << "cache test could not prove GPU completion";
        std::abort();
      }
      if (*queried == pd::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution_->Release(*fence)) std::abort();
  }

  kg::Ds4CacheBuffer Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes + kGuardBytes) != cudaSuccess) std::abort();
    allocations_.push_back(pointer);
    if (cudaMemset(pointer, 0xa5, bytes + kGuardBytes) != cudaSuccess) std::abort();
    // Setup uses the default stream; the consumer stream is nonblocking.
    // Prove setup complete, keeping every allocation owned on failure.
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
    return {reinterpret_cast<std::uintptr_t>(pointer), bytes};
  }

  template <typename T>
  kg::Ds4CacheBuffer Upload(std::span<const T> values) {
    const auto buffer = Allocate(values.size_bytes());
    if (cudaMemcpy(std::bit_cast<void*>(buffer.address), values.data(), values.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      std::abort();
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
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
    std::array<std::uint8_t, kGuardBytes> guard{};
    ASSERT_EQ(cudaMemcpy(guard.data(), std::bit_cast<const void*>(buffer.address + buffer.bytes),
                         guard.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(guard, [](auto byte) { return byte == 0xa5; }));
  }

  static void Exact(const std::vector<float>& a, const std::vector<float>& b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
  }

  kg::LaunchContext& launch() { return *launch_; }

 private:
  std::unique_ptr<pd::DeviceExecution> execution_;
  pd::StreamId stream_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<void*> allocations_;
};

TEST_F(Ds4CacheTest, KvChecksEvenTiesPackedRotaryTailAndGuardedRoundtrip) {
  constexpr std::uint32_t kRows = 3;
  std::vector<float> source(kRows * 512ULL, 0.0f);
  constexpr std::array<float, kRows> kScale = {1.0f, 0.5f, 2.0f};
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < 7; ++block) {
      const auto offset = (row * 512) + (block * 64);
      source[offset + 1] = -0.0f;
      source[offset + 2] = 1.0625f * kScale[row];
      source[offset + 3] = 1.1875f * kScale[row];
      source[offset + 4] = -1.1875f * kScale[row];
      source[offset + 5] = 448.0f * kScale[row];
    }
    for (std::size_t d = 448; d < 512; ++d)
      source[(row * 512) + d] = 1.001f + static_cast<float>(d);
  }
  const auto values = Upload<float>(source);
  const auto codes = Allocate(kRows * 704ULL);
  const auto scales = Allocate(kRows * 28ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(
      launch(), {.values = values, .codes = codes, .scales = scales, .rows = kRows}));
  const auto qat = Download<float>(values);
  const auto packed = Download<std::uint8_t>(codes);
  const auto factors = Download<float>(scales);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < 7; ++block) {
      const auto offset = (row * 512) + (block * 64);
      EXPECT_EQ(factors[(row * 7) + block], kScale[row]);
      EXPECT_TRUE(std::signbit(qat[offset + 1]));
      EXPECT_EQ(qat[offset + 2], kScale[row]);
      EXPECT_EQ(qat[offset + 3], 1.25f * kScale[row]);
      EXPECT_EQ(qat[offset + 4], -1.25f * kScale[row]);
      EXPECT_EQ(packed[(row * 704) + (block * 64) + 2], 56);
      EXPECT_EQ(packed[(row * 704) + (block * 64) + 1], 0x80);
      EXPECT_EQ(packed[(row * 704) + (block * 64) + 3], 58);
      EXPECT_EQ(packed[(row * 704) + (block * 64) + 4], 0x80 + 58);
    }
    // The rotary tail must preserve F32 object bytes, including signed zero.
    // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
    EXPECT_EQ(std::memcmp(packed.data() + (row * 704) + 448, source.data() + (row * 512) + 448,
                          64 * sizeof(float)),
              0);
  }
  const auto table = kg::Ds4CacheDecodeTable();
  const auto decode_table = Upload<float>(table);
  const auto expanded = Allocate(kRows * 512ULL * sizeof(float));
  ASSERT_TRUE(kg::RunDs4CacheExpand(launch(), {.values = expanded,
                                               .codes = codes,
                                               .scales = scales,
                                               .decode_table = decode_table,
                                               .rows = kRows}));
  Exact(qat, Download<float>(expanded));
  Guard(values);
  Guard(codes);
  Guard(scales);
  Guard(expanded);
  Guard(decode_table);
}

TEST_F(Ds4CacheTest, IndexerHadamardImpulsesAndScaleOnlyProducerAgree) {
  constexpr std::uint32_t kRows = 3;
  std::vector<float> source(kRows * 128ULL, 0.0f);
  source[0] = 128.0f;
  std::fill_n(source.begin() + 128, 128, 16.0f);
  source[256 + 1] = 128.0f;
  const auto values = Upload<float>(source);
  const auto codes = Allocate(kRows * 64ULL);
  const auto scales = Allocate(kRows * 16ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                            .values = values,
                                            .codes = codes,
                                            .scales = scales,
                                            .rows = kRows}));
  const auto qat = Download<float>(values);
  const auto packed = Download<std::uint8_t>(codes);
  const auto factors = Download<float>(scales);
  for (std::uint32_t d = 0; d < 128; ++d) {
    EXPECT_EQ(qat[d], 12.0f);
    EXPECT_EQ(qat[128 + d], d == 0 ? 192.0f : 0.0f);
    EXPECT_EQ(qat[256 + d], d % 2 == 0 ? 12.0f : -12.0f);
  }
  for (std::uint32_t byte = 0; byte < 64; ++byte) {
    EXPECT_EQ(packed[byte], 0x77);
    EXPECT_EQ(packed[128 + byte], 0xf7);
  }
  EXPECT_EQ(factors[0], 2.0f);
  EXPECT_EQ(factors[4], 32.0f);
  const auto expanded = Allocate(kRows * 128ULL * sizeof(float));
  ASSERT_TRUE(kg::RunDs4CacheExpand(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                               .values = expanded,
                                               .codes = codes,
                                               .scales = scales,
                                               .decode_table = {},
                                               .rows = kRows}));
  Exact(qat, Download<float>(expanded));
  const auto query = Upload<float>(source);
  const auto query_scales = Allocate(kRows * 16ULL);
  ASSERT_TRUE(kg::RunDs4CacheQat(launch(), {.kind = kg::Ds4CacheKind::kIndexer128,
                                            .values = query,
                                            .codes = {},
                                            .scales = query_scales,
                                            .rows = kRows}));
  Exact(qat, Download<float>(query));
  Exact(factors, Download<float>(query_scales));
  Guard(values);
  Guard(codes);
  Guard(scales);
  Guard(expanded);
  Guard(query);
  Guard(query_scales);
}

TEST_F(Ds4CacheTest, RingWrapRoundsF16AndRefusesStalePositionCapture) {
  std::vector<float> source(3ULL * 512);
  constexpr std::array<float, 8> kValues = {0.0f,
                                            -0.0f,
                                            1.00048828125f,
                                            1.0009765625f,
                                            65504.0f,
                                            -65504.0f,
                                            0.0000000298023223876953125f,
                                            0.000000059604644775390625f};
  for (std::size_t i = 0; i < source.size(); ++i) source[i] = kValues[i % kValues.size()];
  const auto input = Upload<float>(source);
  const std::vector<float> initial(4ULL * 512, 19.0f);
  const auto ring = Upload<float>(initial);
  const kg::Ds4CacheRawStore desc{.source = input, .ring = ring, .first = 3, .rows = 3, .cells = 4};
  auto capture =
      launch().Capture([&](auto& context) { return kg::RunDs4CacheRawStore(context, desc); });
  ASSERT_FALSE(capture.has_value());
  EXPECT_EQ(capture.error().error, kg::KernelError::kRejected);
  Exact(initial, Download<float>(ring));
  ASSERT_TRUE(kg::RunDs4CacheRawStore(launch(), desc));
  auto want = initial;
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t d = 0; d < 512; ++d) {
      const float original = source[(row * 512) + d];
      want[(((3 + row) % 4) * 512) + d] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(original));
    }
  }
  Exact(want, Download<float>(ring));
  Guard(input);
  Guard(ring);
}

}  // namespace
