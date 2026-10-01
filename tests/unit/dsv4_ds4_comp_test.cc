// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original state/frontier, pooling, mandatory QAT and current-operand
// controls. These are source/operator controls, not model quality/performance.
#include "kernels/ggml/dsv4_ds4_comp.h"

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

class Ds4CompTest : public ::testing::Test {
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
        ADD_FAILURE() << "compressor test could not prove GPU completion";
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

  kg::Ds4CompState State(kg::Ds4CacheKind kind, std::uint32_t ratio) {
    const auto head = kind == kg::Ds4CacheKind::kKv512 ? 512U : 128U;
    const auto width = head * (ratio == 4 ? 2U : 1U);
    const auto bytes = static_cast<std::size_t>(ratio == 4 ? 8U : ratio) * width * 4;
    return {.kv = Allocate(bytes), .score = Allocate(bytes), .kind = kind, .ratio = ratio};
  }

  kg::Ds4CompChunk Chunk(const kg::Ds4CompState& state, std::uint32_t first, std::uint32_t tokens,
                         kg::Ds4CacheBuffer kv, kg::Ds4CacheBuffer score, kg::Ds4CacheBuffer ape,
                         kg::Ds4CacheBuffer norm, bool packed = true) {
    const auto head = state.kind == kg::Ds4CacheKind::kKv512 ? 512U : 128U;
    const auto emitted = ((first + tokens) / state.ratio) - (first / state.ratio);
    const auto code = state.kind == kg::Ds4CacheKind::kKv512 ? 704U : 64U;
    const auto scale = state.kind == kg::Ds4CacheKind::kKv512 ? 28U : 16U;
    return {.state = state,
            .kv = kv,
            .score = score,
            .ape = ape,
            .norm = norm,
            .values = emitted != 0 ? Allocate(static_cast<std::size_t>(emitted) * head * 4)
                                   : kg::Ds4CacheBuffer{},
            .codes = packed && emitted != 0 ? Allocate(static_cast<std::size_t>(emitted) * code)
                                            : kg::Ds4CacheBuffer{},
            .scales = packed && emitted != 0 ? Allocate(static_cast<std::size_t>(emitted) * scale)
                                             : kg::Ds4CacheBuffer{},
            .rope = {.frequency_base = 10000, .frequency_scale = 1, .attention_factor = 1},
            .first = first,
            .tokens = tokens,
            .before = first / state.ratio,
            .capacity = ((first + tokens) / state.ratio) + 2,
            .rms_epsilon = 3};
  }

  kg::LaunchContext& launch() { return *launch_; }

 private:
  std::unique_ptr<pd::DeviceExecution> execution_;
  pd::StreamId stream_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<void*> allocations_;
};

TEST_F(Ds4CompTest, BootFiniteSentinelAndExactF16ApeRefreshState) {
  constexpr std::uint32_t kHead = 128;
  constexpr std::uint32_t kWidth = 2 * kHead;
  const auto state = State(kg::Ds4CacheKind::kIndexer128, 4);
  ASSERT_TRUE(kg::RunDs4CompInitialize(launch(), state));
  const auto initial = Download<float>(state.kv);
  const auto initial_score = Download<float>(state.score);
  for (const auto value : initial) EXPECT_EQ(value, 0);
  for (const auto value : initial_score) EXPECT_EQ(value, -1.0e30f);
  std::vector<float> kv(std::size_t{4} * kWidth);
  std::vector<float> score(std::size_t{4} * kWidth);
  std::vector<std::uint16_t> ape(std::size_t{4} * kWidth);
  for (std::uint32_t r = 0; r < 4; ++r)
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      kv[(r * kWidth) + d] = static_cast<float>((r * 10) + (d % 7));
      score[(r * kWidth) + d] = static_cast<float>(r + (d % 3));
      ape[(r * kWidth) + d] = ggml_fp32_to_fp16(static_cast<float>((r * 2) + (d % 5)));
    }
  const auto projection = Upload<float>(kv);
  const auto gate = Upload<float>(score);
  const auto position_bias = Upload<std::uint16_t>(ape);
  const kg::Ds4CompRefresh refresh{.state = state,
                                   .kv = projection,
                                   .score = gate,
                                   .ape = position_bias,
                                   .ape_format = kg::Ds4CompApe::kF16,
                                   .first = 5};
  ASSERT_TRUE(kg::RunDs4CompRefresh(launch(), refresh));
  const auto refreshed = Download<float>(state.kv);
  const auto refreshed_score = Download<float>(state.score);
  for (std::uint32_t r = 0; r < 8; ++r)
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      const auto i = (r * kWidth) + d;
      if (r < 4) {
        EXPECT_EQ(refreshed[i], kv[i]);
        EXPECT_EQ(refreshed_score[i], score[i] + static_cast<float>((((5 + r) % 4) * 2) + (d % 5)));
      } else {
        EXPECT_EQ(refreshed[i], 0);
        EXPECT_EQ(refreshed_score[i], -std::numeric_limits<float>::infinity());
      }
    }
  ASSERT_TRUE(kg::RunDs4CompRefresh(launch(), refresh));
  Exact(refreshed, Download<float>(state.kv));
  Exact(refreshed_score, Download<float>(state.score));
  auto refused =
      launch().Capture([&](auto& context) { return kg::RunDs4CompRefresh(context, refresh); });
  EXPECT_FALSE(refused.has_value());
  Exact(refreshed, Download<float>(state.kv));
  for (const auto buffer : {state.kv, state.score, projection, gate, position_bias}) Guard(buffer);
}

TEST_F(Ds4CompTest, BothRatiosAndIndexerPoolsMatchF64AndReadCurrentGraphOperands) {
  const std::array<std::pair<kg::Ds4CacheKind, std::uint32_t>, 3> cases = {
      {{kg::Ds4CacheKind::kKv512, 4},
       {kg::Ds4CacheKind::kIndexer128, 4},
       {kg::Ds4CacheKind::kKv512, 128}}};
  for (const auto& [kind, ratio] : cases) {
    const auto head = kind == kg::Ds4CacheKind::kKv512 ? 512U : 128U;
    const auto width = head * (ratio == 4 ? 2U : 1U);
    const auto tokens = ratio == 4 ? 12U : 256U;
    const auto rows = tokens / ratio;
    const auto state = State(kind, ratio);
    std::vector<float> state_kv(static_cast<std::size_t>(state.kv.bytes) / 4);
    std::vector<float> state_score(state_kv.size());
    std::vector<float> kv(std::size_t{tokens} * width);
    std::vector<float> score(kv.size());
    std::vector<float> ape(std::size_t{ratio} * width);
    for (std::size_t i = 0; i < state_kv.size(); ++i) {
      state_kv[i] = static_cast<float>(i % 13) * 0.125f;
      state_score[i] = static_cast<float>(i % 7) * 0.0625f;
    }
    for (std::size_t i = 0; i < kv.size(); ++i) {
      kv[i] = static_cast<float>(i % 19) * 0.125f;
      score[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0625f;
    }
    for (std::size_t i = 0; i < ape.size(); ++i) ape[i] = static_cast<float>(i % 5) * 0.125f;
    Replace<float>(state.kv, state_kv);
    Replace<float>(state.score, state_score);
    const auto input = Upload<float>(kv);
    const auto gate = Upload<float>(score);
    const auto bias = Upload<float>(ape);
    const auto values = Allocate(std::size_t{rows} * head * 4);
    const kg::Ds4CompPool pool{.state = state,
                               .kv = input,
                               .score = gate,
                               .ape = bias,
                               .values = values,
                               .first = ratio,
                               .tokens = tokens};
    ASSERT_TRUE(kg::RunDs4CompPool(launch(), pool));
    const auto baseline = Download<float>(values);
    for (std::uint32_t c = 0; c < rows; ++c)
      for (std::uint32_t d = 0; d < head; ++d) {
        std::vector<std::pair<double, double>> candidates;
        if (ratio == 4) {
          for (std::uint32_t r = 0; r < 4; ++r) {
            if (c == 0) {
              const auto i = (r * width) + d;
              candidates.emplace_back(state_kv[i], state_score[i]);
            } else {
              const auto t = ((c - 1) * 4) + r;
              const auto i = (t * width) + d;
              candidates.emplace_back(kv[i], double{score[i]} + ape[((t % 4) * width) + d]);
            }
          }
          for (std::uint32_t r = 0; r < 4; ++r) {
            const auto t = (c * 4) + r;
            const auto i = (t * width) + head + d;
            candidates.emplace_back(kv[i], double{score[i]} + ape[((t % 4) * width) + head + d]);
          }
        } else {
          for (std::uint32_t r = 0; r < ratio; ++r) {
            const auto t = (c * ratio) + r;
            const auto i = (t * width) + d;
            candidates.emplace_back(kv[i], double{score[i]} + ape[(r * width) + d]);
          }
        }
        double maximum = -std::numeric_limits<double>::infinity();
        for (const auto& candidate : candidates) maximum = std::max(maximum, candidate.second);
        double numerator = 0;
        double denominator = 0;
        for (const auto& [value, coefficient] : candidates) {
          const auto weight = std::exp(coefficient - maximum);
          numerator += value * weight;
          denominator += weight;
        }
        EXPECT_NEAR(baseline[(c * head) + d], numerator / denominator, 2e-5);
      }
    auto graph = launch().Capture([&](auto& context) { return kg::RunDs4CompPool(context, pool); });
    ASSERT_TRUE(graph.has_value());
    kv[ratio == 4 ? head : 0] += 5;
    state_kv[0] += 2;
    ape[0] += 0.5f;
    Replace<float>(input, kv);
    Replace<float>(state.kv, state_kv);
    Replace<float>(bias, ape);
    const auto replayed = launch().Launch(*graph);
    Finish();
    ASSERT_TRUE(replayed);
    const auto current = Download<float>(values);
    EXPECT_NE(std::memcmp(current.data(), baseline.data(), current.size() * 4), 0);
    ASSERT_TRUE(kg::RunDs4CompPool(launch(), pool));
    Exact(current, Download<float>(values));
    for (const auto buffer : {state.kv, state.score, input, gate, bias, values}) Guard(buffer);
    Finish();
  }
}

TEST_F(Ds4CompTest, Full4096BulkKvNormRopeQatAndRequiredSmallRefresh) {
  constexpr std::uint32_t kTokens = 4096;
  constexpr std::uint32_t kHead = 512;
  constexpr std::uint32_t kWidth = 1024;
  const auto state = State(kg::Ds4CacheKind::kKv512, 4);
  const std::vector<float> kv(std::size_t{kTokens} * kWidth, 1);
  const std::vector<float> score(kv.size(), 0);
  const std::vector<float> ape(std::size_t{4} * kWidth, 0);
  const std::vector<float> norm(kHead, 2);
  const auto input = Upload<float>(kv);
  const auto gate = Upload<float>(score);
  const auto bias = Upload<float>(ape);
  const auto weight = Upload<float>(norm);
  const auto chunk = Chunk(state, 0, kTokens, input, gate, bias, weight);
  const auto plan = kg::PlanDs4Comp(chunk);
  ASSERT_TRUE(plan.has_value());
  EXPECT_TRUE(plan->refresh_required);
  EXPECT_EQ(plan->emitted, 1024U);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), chunk));
  const auto values = Download<float>(chunk.values);
  const auto codes = Download<std::uint8_t>(chunk.codes);
  const auto scales = Download<float>(chunk.scales);
  for (std::uint32_t r = 0; r < 1024; ++r) {
    for (std::uint32_t d = 0; d < 448; ++d) EXPECT_EQ(values[(r * kHead) + d], 1);
    for (std::uint32_t pair = 0; pair < 32; ++pair) {
      const double angle =
          static_cast<double>(r * 4) * std::pow(10000.0, -static_cast<double>(pair * 2) / 64);
      EXPECT_NEAR(values[(r * kHead) + 448 + (pair * 2)], std::cos(angle) - std::sin(angle), 2e-3);
      EXPECT_NEAR(values[(r * kHead) + 449 + (pair * 2)], std::cos(angle) + std::sin(angle), 2e-3);
    }
  }
  EXPECT_TRUE(
      std::ranges::all_of(scales, [](float value) { return std::isfinite(value) && value > 0; }));
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), chunk));
  Exact(values, Download<float>(chunk.values));
  Exact(codes, Download<std::uint8_t>(chunk.codes));
  Exact(scales, Download<float>(chunk.scales));
  auto refused =
      launch().Capture([&](auto& context) { return kg::RunDs4CompChunk(context, chunk); });
  EXPECT_FALSE(refused.has_value());
  Exact(values, Download<float>(chunk.values));
  // Fixture uses distinct small-product rows to prove refresh consumes its
  // actual supplied inputs, rather than silently reusing the wide tail.
  const std::vector<float> tail_kv(std::size_t{4} * kWidth, 3);
  const std::vector<float> tail_score(std::size_t{4} * kWidth, 0.5f);
  const auto tail_input = Upload<float>(tail_kv);
  const auto tail_gate = Upload<float>(tail_score);
  const kg::Ds4CompRefresh refresh{
      .state = state, .kv = tail_input, .score = tail_gate, .ape = bias, .first = kTokens - 4};
  ASSERT_TRUE(kg::RunDs4CompRefresh(launch(), refresh));
  const auto refreshed = Download<float>(state.kv);
  const auto refreshed_score = Download<float>(state.score);
  for (std::uint32_t r = 0; r < 8; ++r)
    for (std::uint32_t d = 0; d < kWidth; ++d) {
      EXPECT_EQ(refreshed[(r * kWidth) + d], r < 4 ? 3 : 0);
      EXPECT_EQ(refreshed_score[(r * kWidth) + d],
                r < 4 ? 0.5f : -std::numeric_limits<float>::infinity());
    }
  for (const auto buffer : {state.kv, state.score, input, gate, bias, weight, chunk.values,
                            chunk.codes, chunk.scales, tail_input, tail_gate})
    Guard(buffer);
}

TEST_F(Ds4CompTest, RaggedRatio4BoundaryMatchesBulkOutputsAndPreservesOwnFrontier) {
  constexpr std::uint32_t kWidth = 1024;
  const std::vector<float> kv(std::size_t{8} * kWidth, 1);
  const std::vector<float> score(kv.size(), 0);
  const std::vector<float> ape(std::size_t{4} * kWidth, 0);
  const std::vector<float> norm(512, 2);
  const auto input = Upload<float>(kv);
  const auto gate = Upload<float>(score);
  const auto bias = Upload<float>(ape);
  const auto weight = Upload<float>(norm);
  const auto bulk_state = State(kg::Ds4CacheKind::kKv512, 4);
  const auto bulk = Chunk(bulk_state, 0, 8, input, gate, bias, weight);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), bulk));
  const auto bulk_output = Download<float>(bulk.values);
  const auto state = State(kg::Ds4CacheKind::kKv512, 4);
  const auto first = Chunk(state, 0, 1, {input.address, std::uint64_t{kWidth} * 4},
                           {gate.address, std::uint64_t{kWidth} * 4}, bias, weight);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), first));
  const auto rest = Chunk(
      state, 1, 7, {input.address + (std::uint64_t{kWidth} * 4), std::uint64_t{7} * kWidth * 4},
      {gate.address + (std::uint64_t{kWidth} * 4), std::uint64_t{7} * kWidth * 4}, bias, weight);
  const auto plan = kg::PlanDs4Comp(rest);
  ASSERT_TRUE(plan.has_value());
  EXPECT_FALSE(plan->refresh_required);
  EXPECT_EQ(plan->emitted, 2U);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), rest));
  Exact(bulk_output, Download<float>(rest.values));
  Exact(Download<std::uint8_t>(bulk.codes), Download<std::uint8_t>(rest.codes));
  const auto frontier = Download<float>(state.kv);
  const auto frontier_score = Download<float>(state.score);
  // Original per-row shift duplicates the completed group into both halves.
  for (const auto value : frontier) EXPECT_EQ(value, 1);
  for (const auto value : frontier_score) EXPECT_EQ(value, 0);
  const auto bulk_frontier = Download<float>(bulk_state.kv);
  EXPECT_NE(std::memcmp(frontier.data(), bulk_frontier.data(), frontier.size() * 4), 0);
  for (const auto buffer :
       {state.kv, state.score, bulk_state.kv, bulk_state.score, input, gate, bias, weight,
        bulk.values, bulk.codes, bulk.scales, rest.values, rest.codes, rest.scales})
    Guard(buffer);
}

TEST_F(Ds4CompTest, RaggedRatio128TailEmitsAtExactBoundaryWithOriginalStateShape) {
  constexpr std::uint32_t kWidth = 512;
  const std::vector<float> kv(std::size_t{128} * kWidth, 1);
  const std::vector<float> score(kv.size(), 0);
  const std::vector<float> ape(std::size_t{128} * kWidth, 0);
  const std::vector<float> norm(kWidth, 2);
  const auto input = Upload<float>(kv);
  const auto gate = Upload<float>(score);
  const auto bias = Upload<float>(ape);
  const auto weight = Upload<float>(norm);
  const auto state = State(kg::Ds4CacheKind::kKv512, 128);
  ASSERT_TRUE(kg::RunDs4CompInitialize(launch(), state));
  const auto prefix = Chunk(state, 256, 127, {input.address, std::uint64_t{127} * kWidth * 4},
                            {gate.address, std::uint64_t{127} * kWidth * 4}, bias, weight);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), prefix));
  const auto tail = Chunk(
      state, 383, 1, {input.address + (std::uint64_t{127} * kWidth * 4), std::uint64_t{kWidth} * 4},
      {gate.address + (std::uint64_t{127} * kWidth * 4), std::uint64_t{kWidth} * 4}, bias, weight);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), tail));
  const auto output = Download<float>(tail.values);
  for (const auto value : Download<float>(state.kv)) EXPECT_EQ(value, 1);
  for (const auto value : Download<float>(state.score)) EXPECT_EQ(value, 0);
  const auto bulk_state = State(kg::Ds4CacheKind::kKv512, 128);
  const auto aligned = Chunk(bulk_state, 256, 128, input, gate, bias, weight);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), aligned));
  Exact(output, Download<float>(aligned.values));
  for (const auto value : Download<float>(bulk_state.kv)) EXPECT_EQ(value, 0);
  for (const auto value : Download<float>(bulk_state.score))
    EXPECT_EQ(value, -std::numeric_limits<float>::infinity());
  for (const auto buffer :
       {state.kv, state.score, bulk_state.kv, bulk_state.score, input, gate, bias, weight,
        tail.values, tail.codes, tail.scales, aligned.values, aligned.codes, aligned.scales})
    Guard(buffer);
}

TEST_F(Ds4CompTest, IndexerChunkMandatoryHadamardQatMatchesItsPackedMirror) {
  constexpr std::uint32_t kHead = 128;
  constexpr std::uint32_t kWidth = 256;
  const auto state = State(kg::Ds4CacheKind::kIndexer128, 4);
  std::vector<float> kv(std::size_t{16} * kWidth);
  for (std::size_t i = 0; i < kv.size(); ++i) kv[i] = static_cast<float>(1 + (i % 7)) * 0.25f;
  const std::vector<float> score(kv.size(), 0);
  const std::vector<float> ape(std::size_t{4} * kWidth, 0);
  const std::vector<float> norm(kHead, 1);
  const auto input = Upload<float>(kv);
  const auto gate = Upload<float>(score);
  const auto bias = Upload<float>(ape);
  const auto weight = Upload<float>(norm);
  const auto chunk = Chunk(state, 0, 16, input, gate, bias, weight);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), chunk));
  const auto values = Download<float>(chunk.values);
  EXPECT_TRUE(std::ranges::all_of(values, [](float value) { return std::isfinite(value); }));
  EXPECT_TRUE(std::ranges::any_of(values, [](float value) { return value != 0; }));
  const auto decoded = Allocate(values.size() * 4);
  const kg::Ds4CacheExpand expand{.kind = kg::Ds4CacheKind::kIndexer128,
                                  .values = decoded,
                                  .codes = chunk.codes,
                                  .scales = chunk.scales,
                                  .decode_table = {},
                                  .rows = 4};
  ASSERT_TRUE(kg::RunDs4CacheExpand(launch(), expand));
  Exact(values, Download<float>(decoded));
  const auto codes = Download<std::uint8_t>(chunk.codes);
  const auto scales = Download<float>(chunk.scales);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), chunk));
  Exact(values, Download<float>(chunk.values));
  Exact(codes, Download<std::uint8_t>(chunk.codes));
  Exact(scales, Download<float>(chunk.scales));
  const auto plain_state = State(kg::Ds4CacheKind::kIndexer128, 4);
  const auto without_mirror = Chunk(plain_state, 0, 16, input, gate, bias, weight, false);
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), without_mirror));
  Exact(values, Download<float>(without_mirror.values));
  for (const auto buffer :
       {state.kv, state.score, input, gate, bias, weight, chunk.values, chunk.codes, chunk.scales,
        decoded, plain_state.kv, plain_state.score, without_mirror.values})
    Guard(buffer);
}

TEST_F(Ds4CompTest, ExtendedYarnParametersUseOriginalTailCoordinates) {
  const auto state = State(kg::Ds4CacheKind::kKv512, 4);
  ASSERT_TRUE(kg::RunDs4CompInitialize(launch(), state));
  const std::vector<float> kv(std::size_t{8} * 1024, 1);
  const std::vector<float> score(kv.size(), 0);
  const std::vector<float> ape(std::size_t{4} * 1024, 0);
  const std::vector<float> norm(512, 2);
  const auto input = Upload<float>(kv);
  const auto gate = Upload<float>(score);
  const auto bias = Upload<float>(ape);
  const auto weight = Upload<float>(norm);
  auto chunk = Chunk(state, 128, 8, input, gate, bias, weight, false);
  chunk.rope = {.original_context = 16384,
                .frequency_base = 10000,
                .frequency_scale = 0.25f,
                .extension = 1,
                .attention_factor = 1,
                .beta_fast = 32,
                .beta_slow = 1};
  ASSERT_TRUE(kg::RunDs4CompChunk(launch(), chunk));
  const auto values = Download<float>(chunk.values);
  const double pi = std::acos(-1.0);
  const double low =
      std::max(0.0, std::floor(64 * std::log(16384 / (32 * 2 * pi)) / (2 * std::log(10000))));
  const double high =
      std::min(63.0, std::ceil(64 * std::log(16384 / (2 * pi)) / (2 * std::log(10000))));
  const double magnitude = 1 + (0.1 * std::log(4.0));
  for (std::uint32_t r = 0; r < 2; ++r) {
    for (std::uint32_t d = 0; d < 448; ++d) EXPECT_EQ(values[(r * 512) + d], 1);
    for (std::uint32_t pair = 0; pair < 32; ++pair) {
      const double ramp =
          1 - std::clamp((static_cast<double>(pair) - low) / std::max(0.001, high - low), 0.0, 1.0);
      const double extrapolated = static_cast<double>(128 + (r * 4)) *
                                  std::pow(10000.0, -static_cast<double>(pair * 2) / 64);
      const double angle = extrapolated * ((0.25 * (1 - ramp)) + ramp);
      EXPECT_NEAR(values[(r * 512) + 448 + (pair * 2)],
                  magnitude * (std::cos(angle) - std::sin(angle)), 3e-5);
      EXPECT_NEAR(values[(r * 512) + 449 + (pair * 2)],
                  magnitude * (std::cos(angle) + std::sin(angle)), 3e-5);
    }
  }
  for (const auto buffer : {state.kv, state.score, input, gate, bias, weight, chunk.values})
    Guard(buffer);
}

}  // namespace
