// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Analytical HC/norm checks, original producer layouts and current-operand
// graph controls. These are neither model quality nor performance results.
#include "kernels/ggml/dsv4_ds4_hc.h"

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
#include <limits>
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

// Independent F64 coefficient reference: sigmoid, stable row softmax, then
// alternating row/column normalization with the original epsilon convention.
std::array<double, 24> SplitReference(std::span<const float> mix, std::span<const float> scale,
                                      std::span<const float> base, double epsilon) {
  std::array<double, 24> out{};
  for (std::size_t h = 0; h < 4; ++h) {
    out[h] = (1 / (1 + std::exp(-((double{mix[h]} * scale[0]) + base[h])))) + epsilon;
    out[4 + h] = 2 / (1 + std::exp(-((double{mix[4 + h]} * scale[1]) + base[4 + h])));
  }
  for (std::size_t r = 0; r < 4; ++r) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < 4; ++c) {
      out[8 + (r * 4) + c] = (double{mix[8 + (r * 4) + c]} * scale[2]) + base[8 + (r * 4) + c];
      maximum = std::max(maximum, out[8 + (r * 4) + c]);
    }
    double sum = 0;
    for (std::size_t c = 0; c < 4; ++c) {
      auto& value = out[8 + (r * 4) + c];
      value = std::exp(value - maximum);
      sum += value;
    }
    for (std::size_t c = 0; c < 4; ++c)
      out[8 + (r * 4) + c] = (out[8 + (r * 4) + c] / sum) + epsilon;
  }
  for (std::size_t iteration = 0; iteration < 20; ++iteration) {
    if (iteration != 0) {
      for (std::size_t r = 0; r < 4; ++r) {
        double sum = epsilon;
        for (std::size_t c = 0; c < 4; ++c) sum += out[8 + (r * 4) + c];
        for (std::size_t c = 0; c < 4; ++c) out[8 + (r * 4) + c] /= sum;
      }
    }
    for (std::size_t c = 0; c < 4; ++c) {
      double sum = epsilon;
      for (std::size_t r = 0; r < 4; ++r) sum += out[8 + (r * 4) + c];
      for (std::size_t r = 0; r < 4; ++r) out[8 + (r * 4) + c] /= sum;
    }
  }
  return out;
}

class Ds4HcTest : public ::testing::Test {
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
        ADD_FAILURE() << "HC test could not prove GPU completion";
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
    std::array<std::uint8_t, kGuardBytes> guard{};
    ASSERT_EQ(cudaMemcpy(guard.data(), std::bit_cast<const void*>(buffer.address + buffer.bytes),
                         guard.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(guard, [](auto byte) { return byte == 0xa5; }));
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

TEST_F(Ds4HcTest, RmsRaggedRowsInPlaceDualEmissionAndCurrentOperandGraph) {
  constexpr std::size_t kWidth = 257;
  constexpr std::size_t kRows = 3;
  std::vector<float> source(kRows * kWidth);
  constexpr std::array<float, kRows> kRow = {0, 2, -4};
  for (std::uint32_t r = 0; r < kRows; ++r)
    std::fill_n(source.data() + (r * kWidth), kWidth, kRow[r]);
  const auto input = Upload<float>(source);
  const auto plain = Allocate(source.size() * 4);
  const auto plain_half = Allocate(source.size() * 2);
  kg::Ds4Rms desc{.source = input, .values = plain, .width = kWidth, .rows = kRows, .epsilon = 1};
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  const auto normalized = Download<float>(plain);
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::uint32_t d = 0; d < kWidth; ++d)
      EXPECT_NEAR(normalized[(r * kWidth) + d],
                  kRow[r] / std::sqrt((static_cast<double>(kRow[r]) * kRow[r]) + 1), 2e-6);
  desc.values = {};
  desc.values_f16 = plain_half;
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  const auto halves = Download<std::uint16_t>(plain_half);
  for (std::size_t i = 0; i < normalized.size(); ++i)
    EXPECT_EQ(halves[i], ggml_fp32_to_fp16(normalized[i]));

  std::vector<float> weights(kWidth);
  for (std::uint32_t d = 0; d < kWidth; ++d)
    weights[d] = 0.5f + (static_cast<float>(d % 4) * 0.25f);
  const auto weight = Upload<float>(weights);
  const auto weighted = Allocate(source.size() * 4);
  const auto dual = Allocate(source.size() * 4);
  const auto dual_half = Allocate(source.size() * 2);
  desc = {.source = input,
          .weights = weight,
          .values = weighted,
          .width = kWidth,
          .rows = kRows,
          .epsilon = 1};
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  const auto baseline = Download<float>(weighted);
  for (std::size_t i = 0; i < baseline.size(); ++i)
    EXPECT_NEAR(baseline[i], normalized[i] * weights[i % kWidth], 2e-6);
  const auto in_place = Upload<float>(source);
  auto inplace = desc;
  inplace.source = in_place;
  inplace.values = in_place;
  ASSERT_TRUE(kg::RunDs4Rms(launch(), inplace));
  Exact(baseline, Download<float>(in_place));
  desc.values = dual;
  desc.values_f16 = dual_half;
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  Exact(baseline, Download<float>(dual));
  const auto dual_bits = Download<std::uint16_t>(dual_half);
  for (std::size_t i = 0; i < baseline.size(); ++i)
    EXPECT_EQ(dual_bits[i], ggml_fp32_to_fp16(baseline[i]));
  auto graph = launch().Capture([&](auto& context) { return kg::RunDs4Rms(context, desc); });
  ASSERT_TRUE(graph.has_value());
  std::ranges::fill(source, 3.0f);
  weights[0] = 2.0f;
  Replace<float>(input, source);
  Replace<float>(weight, weights);
  auto replayed = launch().Launch(*graph);
  Finish();
  ASSERT_TRUE(replayed);
  const auto current = Download<float>(dual);
  EXPECT_NE(std::memcmp(current.data(), baseline.data(), current.size() * 4), 0);
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  Exact(current, Download<float>(dual));
  for (const auto buffer : {input, plain, plain_half, weight, weighted, dual, dual_half, in_place})
    Guard(buffer);
  Finish();
}

TEST_F(Ds4HcTest, WideProducerD4LayoutScalesSignedCodesAndOwnGraphRepeat) {
  constexpr std::size_t kRows = 64;
  constexpr std::size_t kWidth = 256;
  std::vector<float> source(kRows * kWidth);
  for (std::uint32_t r = 0; r < kRows; ++r)
    std::fill_n(source.data() + (r * kWidth), kWidth, r % 2 == 0 ? 1.0f : -1.0f);
  const std::vector<float> weights(kWidth, 1.0f);
  const auto input = Upload<float>(source);
  const auto weight = Upload<float>(weights);
  const auto values = Allocate(source.size() * 4);
  const auto half = Allocate(source.size() * 2);
  const auto packed = Allocate(kRows * (kWidth / 128) * 144);
  const kg::Ds4Rms desc{.source = input,
                        .weights = weight,
                        .values = values,
                        .values_f16 = half,
                        .q8_d4 = packed,
                        .width = kWidth,
                        .rows = kRows,
                        .epsilon = 3};
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  const auto normalized = Download<float>(values);
  const auto half_bits = Download<std::uint16_t>(half);
  const auto codes = Download<std::uint8_t>(packed);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const float sign = r % 2 == 0 ? 1.0f : -1.0f;
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      EXPECT_NEAR(normalized[(r * kWidth) + d], sign * 0.5f, 2e-7);
      EXPECT_EQ(half_bits[(r * kWidth) + d], ggml_fp32_to_fp16(sign * 0.5f));
    }
    for (std::uint32_t block = 0; block < kWidth / 128; ++block) {
      const auto offset = ((block * kRows) + r) * 144;
      for (std::size_t group = 0; group < 4; ++group) {
        float factor = 0;
        std::memcpy(&factor, codes.data() + offset + (group * 4), 4);
        EXPECT_NEAR(factor, 0.5f / 127.0f, 2e-9);
      }
      for (std::uint32_t d = 0; d < 128; ++d)
        EXPECT_EQ(codes[offset + 16 + d], r % 2 == 0 ? 127 : 129);  // signed -127
    }
  }
  auto graph = launch().Capture([&](auto& context) { return kg::RunDs4Rms(context, desc); });
  ASSERT_TRUE(graph.has_value());
  std::fill_n(source.begin(), kWidth, -1.0f);
  Replace<float>(input, source);
  auto replayed = launch().Launch(*graph);
  Finish();
  ASSERT_TRUE(replayed);
  const auto changed = Download<std::uint8_t>(packed);
  EXPECT_NE(std::memcmp(changed.data(), codes.data(), codes.size()), 0);
  ASSERT_TRUE(kg::RunDs4Rms(launch(), desc));
  Exact(changed, Download<std::uint8_t>(packed));
  for (const auto buffer : {input, weight, values, half, packed}) Guard(buffer);
  Finish();
}

TEST_F(Ds4HcTest, SplitWeightedRaggedRowsAndFusedPreReadCurrentOperands) {
  constexpr std::size_t kRows = 257;
  constexpr std::size_t kWidth = 37;
  constexpr float kEpsilon = 1e-6f;
  std::vector<float> mix(kRows * 24);
  std::vector<float> residual(kRows * 4 * kWidth);
  const std::array<float, 3> scale = {0.75f, -0.5f, 1.25f};
  std::array<float, 24> base{};
  for (std::size_t i = 0; i < base.size(); ++i)
    base[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
  for (std::size_t i = 0; i < mix.size(); ++i)
    mix[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.0625f;
  for (std::size_t i = 0; i < residual.size(); ++i)
    residual[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
  const auto input = Upload<float>(mix);
  const auto lanes = Upload<float>(residual);
  const auto factor = Upload<float>(scale);
  const auto bias = Upload<float>(base);
  const auto split = Allocate(mix.size() * 4);
  const auto separate = Allocate(kRows * kWidth * 4);
  const auto fused = Allocate(kRows * kWidth * 4);
  const auto fused_split = Allocate(mix.size() * 4);
  kg::Ds4HcSplit coefficients{.mix = input,
                              .scale = factor,
                              .base = bias,
                              .split = split,
                              .rows = kRows,
                              .epsilon = kEpsilon};
  ASSERT_TRUE(kg::RunDs4HcSplit(launch(), coefficients));
  const kg::Ds4HcWeighted weighted{
      .residual = lanes, .weights = split, .values = separate, .width = kWidth, .rows = kRows};
  ASSERT_TRUE(kg::RunDs4HcWeighted(launch(), weighted));
  const auto actual_split = Download<float>(split);
  const auto actual_values = Download<float>(separate);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    const auto expected =
        SplitReference(std::span<const float>(mix).subspan(static_cast<std::size_t>(r) * 24, 24),
                       scale, base, kEpsilon);
    for (std::uint32_t c = 0; c < 24; ++c)
      EXPECT_NEAR(actual_split[(r * 24) + c], expected[c], 3e-6);
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      double sum = 0;
      for (std::uint32_t h = 0; h < 4; ++h)
        sum += double{residual[(((r * 4) + h) * kWidth) + d]} * expected[h];
      EXPECT_NEAR(actual_values[(r * kWidth) + d], sum, 2e-5);
    }
  }
  coefficients.split = fused_split;
  const kg::Ds4HcPre pre{
      .coefficients = coefficients, .residual = lanes, .values = fused, .width = kWidth};
  ASSERT_TRUE(kg::RunDs4HcPre(launch(), pre));
  const auto baseline = Download<float>(fused);
  const auto pre_split = Download<float>(fused_split);
  for (std::size_t i = 0; i < baseline.size(); ++i)
    EXPECT_NEAR(baseline[i], actual_values[i], 2e-6);
  for (std::size_t i = 0; i < pre_split.size(); ++i)
    EXPECT_NEAR(pre_split[i], actual_split[i], 2e-6);
  ASSERT_TRUE(kg::RunDs4HcPre(launch(), pre));
  Exact(baseline, Download<float>(fused));
  Exact(pre_split, Download<float>(fused_split));
  auto graph = launch().Capture([&](auto& context) { return kg::RunDs4HcPre(context, pre); });
  ASSERT_TRUE(graph.has_value());
  mix[0] += 2;
  residual[0] += 3;
  Replace<float>(input, mix);
  Replace<float>(lanes, residual);
  const auto replayed = launch().Launch(*graph);
  Finish();
  ASSERT_TRUE(replayed);
  const auto current = Download<float>(fused);
  const auto current_split = Download<float>(fused_split);
  EXPECT_NE(std::memcmp(current.data(), baseline.data(), current.size() * 4), 0);
  ASSERT_TRUE(kg::RunDs4HcPre(launch(), pre));
  Exact(current, Download<float>(fused));
  Exact(current_split, Download<float>(fused_split));
  for (const auto buffer : {input, lanes, factor, bias, split, separate, fused, fused_split})
    Guard(buffer);
  Finish();
}

TEST_F(Ds4HcTest, PlainExpandColumnMappingOptionalAddAndFinalHeadWeights) {
  constexpr std::size_t kRows = 3;
  constexpr std::size_t kWidth = 37;
  std::vector<float> residual(kRows * 4 * kWidth);
  std::vector<float> block(kRows * kWidth);
  std::vector<float> add(kRows * kWidth);
  std::vector<float> split(kRows * 24);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      block[(r * kWidth) + d] = static_cast<float>(1 + (d % 3));
      add[(r * kWidth) + d] = static_cast<float>(r + 2);
      for (std::uint32_t h = 0; h < 4; ++h)
        residual[(((r * 4) + h) * kWidth) + d] = static_cast<float>(h + r + (d % 5));
    }
    for (std::uint32_t h = 0; h < 4; ++h) split[(r * 24) + 4 + h] = static_cast<float>(h + 1);
    for (std::uint32_t i = 0; i < 16; ++i) split[(r * 24) + 8 + i] = static_cast<float>(i + 1);
  }
  const auto lanes = Upload<float>(residual);
  const auto contribution = Upload<float>(block);
  const auto shared = Upload<float>(add);
  const auto coefficients = Upload<float>(split);
  const auto output = Allocate(residual.size() * 4);
  kg::Ds4HcExpand expand{.block = contribution,
                         .add = shared,
                         .residual = lanes,
                         .split = coefficients,
                         .values = output,
                         .width = kWidth,
                         .rows = kRows};
  ASSERT_TRUE(kg::RunDs4HcExpand(launch(), expand));
  const auto actual = Download<float>(output);
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::uint32_t h = 0; h < 4; ++h)
      for (std::uint32_t d = 0; d < kWidth; ++d) {
        double expected = double{block[(r * kWidth) + d] + add[(r * kWidth) + d]} * (h + 1);
        for (std::uint32_t src = 0; src < 4; ++src)
          expected += double{residual[(((r * 4) + src) * kWidth) + d]} * (h + (src * 4) + 1);
        EXPECT_EQ(actual[(((r * 4) + h) * kWidth) + d], expected);
      }
  expand.add = {};
  ASSERT_TRUE(kg::RunDs4HcExpand(launch(), expand));
  const auto no_add = Download<float>(output);
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::uint32_t h = 0; h < 4; ++h)
      for (std::uint32_t d = 0; d < kWidth; ++d)
        EXPECT_EQ(actual[(((r * 4) + h) * kWidth) + d] - no_add[(((r * 4) + h) * kWidth) + d],
                  add[(r * kWidth) + d] * static_cast<float>(h + 1));
  auto refused = expand;
  refused.values = lanes;
  EXPECT_FALSE(kg::RunDs4HcExpand(launch(), refused));
  Exact(residual, Download<float>(lanes));
  Exact(no_add, Download<float>(output));

  std::array<float, kRows * 4> pre{};
  for (std::size_t i = 0; i < pre.size(); ++i)
    pre[i] = static_cast<float>(static_cast<int>(i) - 6) * 0.25f;
  const std::array<float, 1> scale = {0.5f};
  const std::array<float, 4> base = {-0.25f, 0, 0.25f, 0.5f};
  const auto head_pre = Upload<float>(pre);
  const auto head_scale = Upload<float>(scale);
  const auto head_base = Upload<float>(base);
  const auto head_weights = Allocate(pre.size() * 4);
  const auto head_values = Allocate(kRows * kWidth * 4);
  const kg::Ds4HcHeadWeights head{.pre = head_pre,
                                  .scale = head_scale,
                                  .base = head_base,
                                  .values = head_weights,
                                  .rows = kRows};
  ASSERT_TRUE(kg::RunDs4HcHeadWeights(launch(), head));
  const auto actual_weights = Download<float>(head_weights);
  std::array<double, kRows * 4> expected_weights{};
  for (std::size_t i = 0; i < pre.size(); ++i) {
    expected_weights[i] =
        (1 / (1 + std::exp(-((double{pre[i]} * scale[0]) + base[i % 4])))) + head.epsilon;
    EXPECT_NEAR(actual_weights[i], expected_weights[i], 1e-6);
  }
  const kg::Ds4HcWeighted weighted{.residual = lanes,
                                   .weights = head_weights,
                                   .values = head_values,
                                   .width = kWidth,
                                   .rows = kRows,
                                   .weight_stride = 4};
  ASSERT_TRUE(kg::RunDs4HcWeighted(launch(), weighted));
  const auto head_output = Download<float>(head_values);
  for (std::uint32_t r = 0; r < kRows; ++r)
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      double expected = 0;
      for (std::uint32_t h = 0; h < 4; ++h)
        expected += double{residual[(((r * 4) + h) * kWidth) + d]} * expected_weights[(r * 4) + h];
      EXPECT_NEAR(head_output[(r * kWidth) + d], expected, 1e-5);
    }
  for (const auto buffer : {lanes, contribution, shared, coefficients, output, head_pre, head_scale,
                            head_base, head_weights, head_values})
    Guard(buffer);
  Finish();
}

TEST_F(Ds4HcTest, FoldedGuardedSixSlotSumF16AndCurrentOperandGraph) {
  constexpr std::size_t kRows = 9;
  constexpr std::size_t kWidth = 4096;
  std::vector<float> slots(kRows * 6 * kWidth);
  std::vector<float> residual(kRows * 4 * kWidth);
  std::vector<float> split(kRows * 24);
  const std::vector<float> block(kRows * kWidth, 2);
  std::vector<float> add(kRows * kWidth, 1);
  // Ascending F32 sum gives2; reassociation or an F64 accumulator gives3.
  // NaN/+Inf expert contributions must be skipped before the shared add.
  const std::array<float, 6> contribution = {16777216,
                                             1,
                                             -16777216,
                                             2,
                                             std::numeric_limits<float>::quiet_NaN(),
                                             std::numeric_limits<float>::infinity()};
  for (std::uint32_t r = 0; r < kRows; ++r) {
    for (std::uint32_t e = 0; e < 6; ++e)
      std::fill_n(slots.data() + (((r * 6) + e) * kWidth), kWidth, contribution[e]);
    for (std::uint32_t h = 0; h < 4; ++h) {
      split[(r * 24) + 4 + h] = 1;
      split[(r * 24) + 8 + h + (h * 4)] = 1;
      for (std::uint32_t d = 0; d < kWidth; ++d)
        residual[(((r * 4) + h) * kWidth) + d] =
            static_cast<float>(h + 1) + (static_cast<float>(d % 8) / 8);
    }
  }
  const auto unsummed = Upload<float>(slots);
  const auto lanes = Upload<float>(residual);
  const auto coefficients = Upload<float>(split);
  const auto contribution_sum = Upload<float>(block);
  const auto shared = Upload<float>(add);
  const auto output = Allocate(residual.size() * 4);
  const auto half = Allocate(residual.size() * 2);
  const auto ordinary = Allocate(residual.size() * 4);
  const auto ordinary_half = Allocate(residual.size() * 2);
  const kg::Ds4HcExpand folded{.add = shared,
                               .residual = lanes,
                               .split = coefficients,
                               .values = output,
                               .values_f16 = half,
                               .moe_unsummed = unsummed,
                               .width = kWidth,
                               .rows = kRows,
                               .epsilon = 1};
  ASSERT_TRUE(kg::RunDs4HcExpand(launch(), folded));
  const auto baseline = Download<float>(output);
  const auto baseline_half = Download<std::uint16_t>(half);
  double square_sum = 0;
  for (std::size_t i = 0; i < kWidth * 4; ++i) {
    const double expected = double{residual[i]} + 3;
    square_sum += expected * expected;
  }
  const double inverse_rms = 1 / std::sqrt((square_sum / (kWidth * 4)) + 1);
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    EXPECT_EQ(baseline[i], residual[i] + 3);
    const auto expected_half =
        ggml_fp32_to_fp16(static_cast<float>((double{residual[i]} + 3) * inverse_rms));
    EXPECT_LE(std::abs(static_cast<int>(baseline_half[i]) - static_cast<int>(expected_half)), 1);
  }
  auto separate = folded;
  separate.block = contribution_sum;
  separate.moe_unsummed = {};
  separate.values = ordinary;
  separate.values_f16 = ordinary_half;
  ASSERT_TRUE(kg::RunDs4HcExpand(launch(), separate));
  Exact(baseline, Download<float>(ordinary));
  Exact(baseline_half, Download<std::uint16_t>(ordinary_half));
  ASSERT_TRUE(kg::RunDs4HcExpand(launch(), folded));
  Exact(baseline_half, Download<std::uint16_t>(half));
  auto graph = launch().Capture([&](auto& context) { return kg::RunDs4HcExpand(context, folded); });
  ASSERT_TRUE(graph.has_value());
  for (std::uint32_t r = 0; r < kRows; ++r)
    std::fill_n(slots.data() + (((r * 6) + 3) * kWidth), kWidth, 5);
  std::ranges::fill(add, 2);
  Replace<float>(unsummed, slots);
  Replace<float>(shared, add);
  const auto replayed = launch().Launch(*graph);
  Finish();
  ASSERT_TRUE(replayed);
  const auto current = Download<float>(output);
  const auto current_half = Download<std::uint16_t>(half);
  EXPECT_NE(std::memcmp(current.data(), baseline.data(), current.size() * 4), 0);
  ASSERT_TRUE(kg::RunDs4HcExpand(launch(), folded));
  Exact(current, Download<float>(output));
  Exact(current_half, Download<std::uint16_t>(half));
  for (const auto buffer : {unsummed, lanes, coefficients, contribution_sum, shared, output, half,
                            ordinary, ordinary_half})
    Guard(buffer);
  Finish();
}

}  // namespace
