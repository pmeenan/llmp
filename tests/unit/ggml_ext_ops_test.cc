// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML operations DeepSeek V4 Flash (GGUF) and Qwen3.8 Flash add
// (kernels/ggml/ops_ext.h) under jitLLM's launch context on a GB10 (label
// `gpu`), at small cases of the models' shapes:
// - each result against an FP64 reference on the host, judged by the
//   normalized mean squared error, NMSE = sum((got - want)^2) / sum(want^2),
//   within upstream's own bounds for the operation (test-backend-ops.cpp at
//   llama.cpp b29c606e2: 1e-7 by default, 5e-4 for quantized products and
//   flash attention, 2e-2 for MXFP4 products whose activations the GB10
//   quantizes to FP4, 1e-6 for the lightning indexer), where upstream judges
//   against GGML's CPU backend; copies, gathers, fills and index results
//   exactly;
// - quantized weights are quantized on the host by GGML's own quantizers
//   and dequantized for the reference by GGML's own to_float;
// - each implementation takes only what upstream's routing would give it on
//   the device, and draws no more scratch than its plan says (the pool
//   enforces the bound);
// - the over-read probe: with operands in device VMM flush against unmapped
//   granules and the workspace sized to the plan, the scratch-drawing and
//   the most intricate operations complete and write what they write in
//   cudaMalloc memory;
// - the registry declares and binds every new implementation (D-053).

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/shared_q8.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/gemma4.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::KernelFailure;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::QuantMulMatPath;
using jitllm::kernels::ggml::TensorArena;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;
using jitllm::test_support::FailedCode;
namespace kg = jitllm::kernels::ggml;

// Upstream's bounds (test-backend-ops.cpp).
constexpr double kDefaultNmse = 1e-7;
constexpr double kMulMatNmse = 5e-4;
constexpr double kFp4ActivationNmse = 2e-2;
constexpr double kFlashAttnNmse = 5e-4;
constexpr double kIndexerNmse = 1e-6;

constexpr std::uint64_t kWorkspace = 256ULL << 20;

std::vector<float> Normal(std::uint64_t seed, std::size_t count, float sigma = 1.0f,
                          float mean = 0.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(mean, sigma);
  std::vector<float> values(count);
  for (float& v : values) {
    v = normal(random);
  }
  return values;
}

std::vector<float> Uniform(std::uint64_t seed, std::size_t count, float low, float high) {
  std::mt19937_64 random(seed);
  std::uniform_real_distribution<float> uniform(low, high);
  std::vector<float> values(count);
  for (float& v : values) {
    v = uniform(random);
  }
  return values;
}

std::vector<ggml_fp16_t> Halves(const std::vector<float>& values) {
  std::vector<ggml_fp16_t> halves(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    halves[i] = ggml_fp32_to_fp16(values[i]);
  }
  return halves;
}

std::vector<float> Widen(const std::vector<ggml_fp16_t>& halves) {
  std::vector<float> values(halves.size());
  for (std::size_t i = 0; i < halves.size(); ++i) {
    values[i] = ggml_fp16_to_fp32(halves[i]);
  }
  return values;
}

double Nmse(const std::vector<float>& got, const std::vector<double>& want) {
  EXPECT_EQ(got.size(), want.size());
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    error += d * d;
    norm += want[i] * want[i];
  }
  return norm > 0.0 ? error / norm : error;
}

void ExpectNmse(const std::vector<float>& got, const std::vector<double>& want, double bound,
                const std::string& what) {
  const double nmse = Nmse(got, want);
  EXPECT_LE(nmse, bound) << what;
  EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); })) << what;
  std::cout << what << ": NMSE " << nmse << " (bound " << bound << ")\n";
}

void Launched(const std::expected<void, KernelFailure>& result, const std::string& what) {
  EXPECT_TRUE(result.has_value()) << what << ": " << (result ? "" : result.error().detail);
}

class GgmlExtOpsTest : public ::testing::Test {
 protected:
  void SmallOwnerControl(std::uint32_t cap, std::uint32_t small_owners = 0);
  void BoundedOwnerControl(std::uint32_t cap);
  void MultirowOwnerControl(std::span<const std::pair<std::int64_t, std::int64_t>> shapes);
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = workspace, .size = Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(512).value());
  }

  void TearDown() override {
    Finish();
    launch_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
  }

  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    FenceState state = FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    ASSERT_EQ(state, FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }

  std::uint64_t Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  ggml_context* c() const { return arena_->context(); }
  LaunchContext& launch() { return *launch_; }

  // Binds `tensor` to new device memory holding `data` if given (bytes).
  template <typename T = float>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& data = {}) {
    const std::size_t size = ggml_nbytes(tensor);
    const std::uint64_t address =
        Allocate(size + (ggml_is_quantized(tensor->type) ? ggml_row_size(tensor->type, 512) : 0));
    TensorArena::Bind(tensor, address);
    if (ggml_is_quantized(tensor->type)) {
      EXPECT_EQ(
          cudaMemset(reinterpret_cast<void*>(address + size), 0, ggml_row_size(tensor->type, 512)),
          cudaSuccess);
    }
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), size);
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), data.data(), size,  // NOLINT
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      // A copy from pageable memory may return before its DMA lands, and
      // the provider's stream does not wait for the legacy stream
      // (CU_STREAM_NON_BLOCKING): wait for it before any launch reads it.
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    return tensor;
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  static int ComputeCapability() {
    int major = 0;
    int minor = 0;
    EXPECT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    EXPECT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    return (100 * major) + (10 * minor);
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

TEST_F(GgmlExtOpsTest, Gemma3DeviceMasksMatchEveryHostByteAcrossWrapAndTrainedMaximum) {
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  const auto& profile = jitllm::model::Gemma3_4BQat();
  const auto state = *jitllm::model::Gemma3State(profile, 131072, 128);
  const std::array<std::int32_t, 128> tokens{};
  for (const auto [past, rows] :
       {std::pair{1279U, 1U}, std::pair{1152U, 128U}, std::pair{130816U, 128U}}) {
    std::array segments{
        jitllm::model::Gemma3Segment{0, past, std::span(tokens).first(rows)},
        jitllm::model::Gemma3Segment{1, past + rows, std::span(tokens).first(rows)}};
    auto host = jitllm::model::Gemma3Chunk(profile, state, segments, true, 256, 256);
    ASSERT_TRUE(host);
    auto* positions = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 2 * rows), host->positions);
    for (std::uint32_t segment = 0; segment < 2; ++segment) {
      for (const bool local : {false, true}) {
        const auto& input = host->segments[segment];
        const auto cells = local ? input.local_n_kv : input.global_n_kv;
        auto* mask = kg::Gemma4Mask(
            c(), positions, cells, static_cast<std::int32_t>(segment * rows),
            static_cast<std::int32_t>(rows),
            static_cast<std::int32_t>(local ? state.local_cells : state.global_cells),
            local ? static_cast<std::int32_t>(profile.window) : 0, 131072);
        const auto address = Allocate(ggml_nbytes(mask) + 512);
        kg::TensorArena::Bind(mask, address + 256);
        const auto expected = [&](const auto& fresh) {
          const auto& in = fresh.segments[segment];
          const auto& logical = local ? in.local_mask : in.global_mask;
          std::vector<std::uint16_t> padded(ggml_nbytes(mask) / 2, 0xFC00);
          std::ranges::copy(logical, padded.begin());
          return padded;
        };
        const auto poison = [&] {
          EXPECT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0xa5, ggml_nbytes(mask) + 512,
                                    stream),
                    cudaSuccess);
        };
        const auto guards = [&] {
          std::array<std::uint8_t, 256> before{}, after{};
          Finish();
          EXPECT_EQ(cudaMemcpy(before.data(), reinterpret_cast<void*>(address), 256,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(
              cudaMemcpy(after.data(), reinterpret_cast<void*>(address + 256 + ggml_nbytes(mask)),
                         256, cudaMemcpyDeviceToHost),
              cudaSuccess);
          EXPECT_TRUE(std::ranges::all_of(before, [](auto x) { return x == 0xa5; }));
          EXPECT_EQ(before, after);
        };
        poison();
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        const auto eager = Download<std::uint16_t>(mask);
        EXPECT_EQ(eager, expected(*host));
        auto captured =
            launch().Capture([&](auto& context) { return kg::RunGemma4Mask(context, mask); });
        ASSERT_TRUE(captured);
        poison();
        ASSERT_TRUE(launch().Launch(*captured));
        EXPECT_EQ(Download<std::uint16_t>(mask), eager);
        guards();
        // Advance only this owner, leaving the peer's offset/positions untouched.
        const std::int32_t delta = (segments[segment].n_past + rows) % 256 == 0 ? -1 : 1;
        segments[segment].n_past =
            static_cast<std::uint32_t>(static_cast<std::int64_t>(segments[segment].n_past) + delta);
        auto fresh = jitllm::model::Gemma3Chunk(profile, state, segments, true, 256, 256);
        ASSERT_TRUE(fresh);
        ASSERT_EQ(fresh->segments[segment].global_n_kv, input.global_n_kv);
        ASSERT_EQ(fresh->segments[segment].local_n_kv, input.local_n_kv);
        ASSERT_EQ(cudaMemcpyAsync(positions->data, fresh->positions.data(), ggml_nbytes(positions),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        poison();
        ASSERT_TRUE(launch().Launch(*captured));
        const auto replay = Download<std::uint16_t>(mask);
        EXPECT_EQ(replay, expected(*fresh));
        poison();
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        EXPECT_EQ(Download<std::uint16_t>(mask), replay);
        EXPECT_EQ(Download<std::int32_t>(positions), fresh->positions);
        guards();
        segments[segment].n_past =
            static_cast<std::uint32_t>(static_cast<std::int64_t>(segments[segment].n_past) - delta);
        ASSERT_EQ(cudaMemcpyAsync(positions->data, host->positions.data(), ggml_nbytes(positions),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        const auto saved = positions->type;
        positions->type = GGML_TYPE_F32;
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), mask));
        positions->type = saved;
        EXPECT_FALSE(launch().faulted());
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, Gemma2DeviceMasksMatchEveryHostByteAcrossWrapAndEightKBoundary) {
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  const auto& profile = jitllm::model::Gemma2_2B();
  const auto state = *jitllm::model::Gemma2State(profile, 8192, 128);
  const std::array<std::int32_t, 128> tokens{};
  for (const auto [past, rows] :
       {std::pair{4351U, 1U}, std::pair{4224U, 128U}, std::pair{7936U, 128U}}) {
    std::array segments{
        jitllm::model::Gemma2Segment{0, past, std::span(tokens).first(rows)},
        jitllm::model::Gemma2Segment{1, past + rows, std::span(tokens).first(rows)}};
    auto host = jitllm::model::Gemma2Chunk(profile, state, segments, true, 256, 256);
    ASSERT_TRUE(host);
    auto* positions = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 2 * rows), host->positions);
    for (std::uint32_t segment = 0; segment < 2; ++segment) {
      for (const bool local : {false, true}) {
        const auto& input = host->segments[segment];
        const auto cells = local ? input.local_n_kv : input.global_n_kv;
        auto* mask = kg::Gemma4Mask(
            c(), positions, cells, static_cast<std::int32_t>(segment * rows),
            static_cast<std::int32_t>(rows),
            static_cast<std::int32_t>(local ? state.local_cells : state.global_cells),
            local ? static_cast<std::int32_t>(profile.window) : 0, 8192);
        const auto address = Allocate(ggml_nbytes(mask) + 512);
        kg::TensorArena::Bind(mask, address + 256);
        const auto expected = [&](const auto& fresh) {
          const auto& in = fresh.segments[segment];
          const auto& logical = local ? in.local_mask : in.global_mask;
          std::vector<std::uint16_t> padded(ggml_nbytes(mask) / 2, 0xFC00);
          std::ranges::copy(logical, padded.begin());
          return padded;
        };
        const auto poison = [&] {
          EXPECT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0xa5, ggml_nbytes(mask) + 512,
                                    stream),
                    cudaSuccess);
        };
        const auto guards = [&] {
          std::array<std::uint8_t, 256> before{}, after{};
          Finish();
          EXPECT_EQ(cudaMemcpy(before.data(), reinterpret_cast<void*>(address), 256,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(
              cudaMemcpy(after.data(), reinterpret_cast<void*>(address + 256 + ggml_nbytes(mask)),
                         256, cudaMemcpyDeviceToHost),
              cudaSuccess);
          EXPECT_TRUE(std::ranges::all_of(before, [](auto x) { return x == 0xa5; }));
          EXPECT_EQ(before, after);
        };
        poison();
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        const auto eager = Download<std::uint16_t>(mask);
        EXPECT_EQ(eager, expected(*host));
        auto captured =
            launch().Capture([&](auto& context) { return kg::RunGemma4Mask(context, mask); });
        ASSERT_TRUE(captured);
        poison();
        ASSERT_TRUE(launch().Launch(*captured));
        EXPECT_EQ(Download<std::uint16_t>(mask), eager);
        guards();
        // Advance only this owner, leaving the peer's offset/positions untouched.
        const std::int32_t delta = (segments[segment].n_past + rows) % 256 == 0 ? -1 : 1;
        segments[segment].n_past =
            static_cast<std::uint32_t>(static_cast<std::int64_t>(segments[segment].n_past) + delta);
        auto fresh = jitllm::model::Gemma2Chunk(profile, state, segments, true, 256, 256);
        ASSERT_TRUE(fresh);
        ASSERT_EQ(fresh->segments[segment].global_n_kv, input.global_n_kv);
        ASSERT_EQ(fresh->segments[segment].local_n_kv, input.local_n_kv);
        ASSERT_EQ(cudaMemcpyAsync(positions->data, fresh->positions.data(), ggml_nbytes(positions),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        poison();
        ASSERT_TRUE(launch().Launch(*captured));
        const auto replay = Download<std::uint16_t>(mask);
        EXPECT_EQ(replay, expected(*fresh));
        poison();
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        EXPECT_EQ(Download<std::uint16_t>(mask), replay);
        EXPECT_EQ(Download<std::int32_t>(positions), fresh->positions);
        guards();
        segments[segment].n_past =
            static_cast<std::uint32_t>(static_cast<std::int64_t>(segments[segment].n_past) - delta);
        ASSERT_EQ(cudaMemcpyAsync(positions->data, host->positions.data(), ggml_nbytes(positions),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        const auto saved = positions->type;
        positions->type = GGML_TYPE_F32;
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), mask));
        positions->type = saved;
        EXPECT_FALSE(launch().faulted());
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, SharedCausalMasksMatchF16AndF32AndRefreshCapturedPositions) {
  const auto stream = reinterpret_cast<cudaStream_t>(execution_->Submission(stream_)->handle);
  constexpr std::int32_t cells = 512, rows = 3, first = 3, limit = 2048;
  for (const auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
    for (const auto layout : {kg::CausalMaskRows::kExact, kg::CausalMaskRows::kPad32}) {
      for (const std::int32_t window : {0, 128}) {
        SCOPED_TRACE(std::format("type={} layout={} window={}", static_cast<int>(type),
                                 static_cast<int>(layout), window));
        std::vector<std::int32_t> positions(12, 17);
        positions[first] = 0;
        positions[first + 1] = 511;
        positions[first + 2] = 1023;
        auto* input = Place(
            ggml_new_tensor_1d(c(), GGML_TYPE_I32, static_cast<std::int64_t>(positions.size())),
            positions);
        auto* mask =
            kg::CausalRingMask(c(), input, cells, first, rows, cells, window, limit, layout, type);
        const auto bytes = ggml_nbytes(mask);
        const auto address = Allocate(bytes + 512);
        TensorArena::Bind(mask, address + 256);
        auto poison = [&] {
          return cudaMemsetAsync(reinterpret_cast<void*>(address), 0xa5, bytes + 512, stream);
        };
        auto check = [&] {
          // Derive the latest retained absolute key independently of the
          // kernel's modulo-distance predicate, then compare exact output bits.
          std::vector<std::uint32_t> expected(static_cast<std::size_t>(mask->ne[0] * mask->ne[1]));
          for (std::int64_t row = 0; row < mask->ne[1]; ++row) {
            for (std::int64_t cell = 0; cell < cells; ++cell) {
              bool visible = false;
              if (row < rows) {
                const auto position = positions[static_cast<std::size_t>(first + row)];
                if (position >= 0 && position < limit && cell <= position) {
                  if (window == 0) {
                    visible = cell < limit;
                  } else {
                    const auto latest = cell + ((position - cell) / cells) * cells;
                    visible = position - latest < window;
                  }
                }
              }
              expected[static_cast<std::size_t>(row * cells + cell)] =
                  visible ? 0U : (type == GGML_TYPE_F16 ? 0xfc00U : 0xff800000U);
            }
          }
          if (type == GGML_TYPE_F16) {
            const auto got = Download<std::uint16_t>(mask);
            EXPECT_EQ(std::vector<std::uint32_t>(got.begin(), got.end()), expected);
          } else {
            EXPECT_EQ(Download<std::uint32_t>(mask), expected);
          }
          std::array<std::uint8_t, 256> prefix{}, suffix{};
          EXPECT_EQ(cudaMemcpy(prefix.data(), reinterpret_cast<void*>(address), prefix.size(),
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(cudaMemcpy(suffix.data(), reinterpret_cast<void*>(address + 256 + bytes),
                               suffix.size(), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_TRUE(std::ranges::all_of(prefix, [](auto value) { return value == 0xa5; }));
          EXPECT_TRUE(std::ranges::all_of(suffix, [](auto value) { return value == 0xa5; }));
          EXPECT_EQ(Download<std::int32_t>(input), positions);
        };
        ASSERT_EQ(poison(), cudaSuccess);
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        check();
        auto graph =
            launch().Capture([&](auto& context) { return kg::RunGemma4Mask(context, mask); });
        ASSERT_TRUE(graph);
        positions[first] = -1;
        positions[first + 1] = 512;
        positions[first + 2] = limit;
        ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(poison(), cudaSuccess);
        ASSERT_TRUE(launch().Launch(*graph));
        check();

        const auto before = Download<std::uint8_t>(mask);
        for (const std::size_t reserved : {6U, 7U}) {
          const std::int32_t invalid = reserved == 6 ? 2 : 1, zero = 0;
          auto* parameter = reinterpret_cast<std::byte*>(mask->op_params) + 32 + reserved * 4;
          std::memcpy(parameter, &invalid, 4);
          EXPECT_FALSE(kg::RunGemma4Mask(launch(), mask));
          std::memcpy(parameter, &zero, 4);
        }
        auto rejected = *mask;
        rejected.type = GGML_TYPE_I32;
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
        rejected = *mask;
        rejected.data = input->data;
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
        if (type == GGML_TYPE_F32) {
          rejected = *mask;
          rejected.data = reinterpret_cast<void*>(address + 258);
          EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
        }
        rejected = *mask;
        // Coherent cell/capacity endpoints that exceed the signed byte bound
        // must refuse before descriptor byte arithmetic or a device launch.
        rejected.ne[0] = type == GGML_TYPE_F16 ? (1LL << 30) : (1LL << 29);
        const auto oversized_capacity = static_cast<std::int32_t>(rejected.ne[0]);
        std::memcpy(reinterpret_cast<std::byte*>(rejected.op_params) + 32 + 2 * 4,
                    &oversized_capacity, 4);
        EXPECT_FALSE(kg::Gemma4MaskFits(&rejected));
        EXPECT_EQ(Download<std::uint8_t>(mask), before);
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, SharedBlockMasksMatchF16AndF32AndRefreshCapturedSegments) {
  const auto stream = reinterpret_cast<cudaStream_t>(execution_->Submission(stream_)->handle);
  constexpr std::int32_t cells = 256, first = 3, window = 128;
  for (const std::int32_t rows : {1, 3, 5}) {
    for (const auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
      for (const auto layout : {kg::CausalMaskRows::kExact, kg::CausalMaskRows::kPad32}) {
        SCOPED_TRACE(
            std::format("type={} layout={}", static_cast<int>(type), static_cast<int>(layout)));
        std::vector<std::int32_t> positions(12, 17);
        for (std::int32_t row = 0; row < rows; ++row)
          positions[static_cast<std::size_t>(first + row)] = row;
        auto* input = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 12), positions);
        auto* mask = kg::CausalRingMask(c(), input, cells, first, rows, cells, window, INT32_MAX,
                                        layout, type, kg::MaskPolicy::kBlock);
        const auto bytes = ggml_nbytes(mask);
        const auto address = Allocate(bytes + 512);
        TensorArena::Bind(mask, address + 256);
        const auto poison = [&] {
          return cudaMemsetAsync(reinterpret_cast<void*>(address), 0xa5, bytes + 512, stream);
        };
        const auto check = [&] {
          std::vector<std::uint32_t> expected(static_cast<std::size_t>(cells * mask->ne[1]),
                                              type == GGML_TYPE_F16 ? 0xfc00U : 0xff800000U);
          const std::int64_t start = positions[first];
          const std::int64_t end = start + rows;
          if (start >= 0 && end <= INT32_MAX) {
            // Enumerate actual absolute keys held after the block rather than
            // duplicating the kernel's congruence/latest-cell formula.
            for (std::int32_t row = 0; row < rows; ++row) {
              const std::int64_t query = positions[static_cast<std::size_t>(first + row)];
              if (query != start + row) continue;
              for (auto key = std::max<std::int64_t>(0, end - cells); key < end; ++key) {
                if (key > query || query - key < window)
                  expected[static_cast<std::size_t>(row * cells + key % cells)] = 0;
              }
            }
          }
          if (type == GGML_TYPE_F16) {
            const auto got = Download<std::uint16_t>(mask);
            EXPECT_EQ(std::vector<std::uint32_t>(got.begin(), got.end()), expected);
          } else {
            EXPECT_EQ(Download<std::uint32_t>(mask), expected);
          }
          std::array<std::uint8_t, 256> prefix{}, suffix{};
          EXPECT_EQ(cudaMemcpy(prefix.data(), reinterpret_cast<void*>(address), prefix.size(),
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(cudaMemcpy(suffix.data(), reinterpret_cast<void*>(address + 256 + bytes),
                               suffix.size(), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_TRUE(std::ranges::all_of(prefix, [](auto value) { return value == 0xa5; }));
          EXPECT_TRUE(std::ranges::all_of(suffix, [](auto value) { return value == 0xa5; }));
          EXPECT_EQ(Download<std::int32_t>(input), positions);
        };
        ASSERT_EQ(poison(), cudaSuccess);
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        check();
        auto graph =
            launch().Capture([&](auto& context) { return kg::RunGemma4Mask(context, mask); });
        ASSERT_TRUE(graph);
        for (const auto start : {254, 300, INT32_MAX - rows, -1, INT32_MAX}) {
          for (std::int32_t row = 0; row < rows; ++row)
            positions[static_cast<std::size_t>(first + row)] = static_cast<std::int32_t>(
                std::min<std::int64_t>(INT32_MAX, std::int64_t{start} + row));
          ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
          ASSERT_EQ(poison(), cudaSuccess);
          ASSERT_TRUE(launch().Launch(*graph));
          check();
        }
        for (std::int32_t row = 0; row < rows; ++row)
          positions[static_cast<std::size_t>(first + row)] = 254 + row;
        positions[static_cast<std::size_t>(first + (rows > 1 ? 1 : 0))] = -1;
        ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(poison(), cudaSuccess);
        ASSERT_TRUE(launch().Launch(*graph));
        check();
        auto rejected = *mask;
        const std::int32_t zero = 0, unknown = 2;
        std::memcpy(reinterpret_cast<std::byte*>(rejected.op_params) + 32 + 3 * 4, &zero, 4);
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
        rejected = *mask;
        std::memcpy(reinterpret_cast<std::byte*>(rejected.op_params) + 32 + 6 * 4, &unknown, 4);
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
        rejected = *mask;
        const std::int32_t too_wide = cells;
        std::memcpy(reinterpret_cast<std::byte*>(rejected.op_params) + 32 + 3 * 4, &too_wide, 4);
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
        rejected = *mask;
        rejected.data = input->data;
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), &rejected));
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, DeepSeekRawDeviceMasksKeepExactRowsAndRefreshCapturedRingPositions) {
  namespace md = jitllm::model;
  const auto stream = reinterpret_cast<cudaStream_t>(execution_->Submission(stream_)->handle);
  for (const auto window : {md::Dsv4Window::kRing, md::Dsv4Window::kFull}) {
    auto state = md::Dsv4State(md::Dsv4Flash(), 8704, 4096, window);
    ASSERT_TRUE(state);
    for (const std::uint32_t rows : {1U, 3U, 16U, 257U}) {
      for (const std::uint32_t past :
           {0U, std::min(state->raw_cells - 1, 8704U - rows - 2), 8704U - rows - 2}) {
        auto chunk = md::Dsv4Chunk(md::Dsv4Flash(), *state, past, rows, false);
        ASSERT_TRUE(chunk);
        // Offset three exercises an actual segment into joined position input.
        std::vector<std::int32_t> positions(rows + 3, 17);
        std::ranges::copy(chunk->positions, positions.begin() + 3);
        auto* input = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows + 3), positions);
        auto* mask = kg::CausalRingMask(
            c(), input, chunk->raw_n_kv, 3, static_cast<std::int32_t>(rows),
            static_cast<std::int32_t>(state->raw_cells), 128, 8704, kg::CausalMaskRows::kExact);
        ASSERT_EQ(mask->ne[1], rows);
        ASSERT_EQ(ggml_nbytes(mask), chunk->raw_mask.size() * 2);
        const auto bytes = ggml_nbytes(mask);
        const auto address = Allocate(bytes + 512);
        TensorArena::Bind(mask, address + 256);
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0xa5, bytes + 512, stream),
                  cudaSuccess);
        ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
        EXPECT_EQ(Download<std::uint16_t>(mask), chunk->raw_mask);
        auto graph =
            launch().Capture([&](auto& context) { return kg::RunGemma4Mask(context, mask); });
        ASSERT_TRUE(graph);
        // In full-window mode choose positions that keep the same padded cell
        // width; ring shapes retain their width at every position.
        auto fresh = md::Dsv4Chunk(md::Dsv4Flash(), *state, past + 1, rows, false);
        ASSERT_TRUE(fresh);
        if (fresh->raw_n_kv == chunk->raw_n_kv) {
          std::ranges::copy(fresh->positions, positions.begin() + 3);
          ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
          ASSERT_TRUE(launch().Launch(*graph));
          EXPECT_EQ(Download<std::uint16_t>(mask), fresh->raw_mask);
        }
        auto* unexpected = ggml_new_tensor_1d(c(), GGML_TYPE_I32, 1);
        for (std::size_t parent = 1; parent < GGML_MAX_SRC; ++parent) {
          mask->src[parent] = unexpected;
          EXPECT_FALSE(kg::Gemma4MaskFits(mask));
          mask->src[parent] = nullptr;
        }
        std::int32_t unknown_layout = 2;
        std::memcpy(reinterpret_cast<std::byte*>(mask->op_params) + 32 + 5 * 4, &unknown_layout, 4);
        const auto before = Download<std::uint16_t>(mask);
        EXPECT_FALSE(kg::RunGemma4Mask(launch(), mask));
        EXPECT_EQ(Download<std::uint16_t>(mask), before);
        std::array<std::uint8_t, 256> prefix{}, suffix{};
        Finish();
        ASSERT_EQ(cudaMemcpy(prefix.data(), reinterpret_cast<void*>(address), 256,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(suffix.data(), reinterpret_cast<void*>(address + 256 + bytes), 256,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_TRUE(std::ranges::all_of(prefix, [](auto byte) { return byte == 0xa5; }));
        EXPECT_TRUE(std::ranges::all_of(suffix, [](auto byte) { return byte == 0xa5; }));
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, GemmaDeviceMasksFillEveryPaddedByteAndRefreshCapturedPositions) {
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  for (const auto* profile : {&jitllm::model::Gemma4_26BA4B(), &jitllm::model::Gemma4_31B()}) {
    for (const auto segments : {1, 2, 4, 16}) {
      const int rows = segments == 16 ? 16 : segments;
      const int first_past = segments == 1 ? 0 : segments == 16 ? 262144 - rows - 15 * 7 : 1279;
      const int capacity = 1280;
      const int cells = segments == 16 ? 262144 : segments == 1 ? 256 : 1536;
      std::vector<std::int32_t> positions(static_cast<std::size_t>(rows * segments));
      for (int segment = 0; segment < segments; ++segment)
        for (int r = 0; r < rows; ++r)
          positions[static_cast<std::size_t>(segment * rows + r)] = first_past + segment * 7 + r;
      auto* input = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows * segments), positions);
      for (int segment = 0; segment < segments; ++segment) {
        for (const bool local : {false, true}) {
          const int read_cells = local ? (segments == 1 ? 256 : capacity) : cells;
          auto* node = kg::Gemma4Mask(c(), input, read_cells, segment * rows, rows,
                                      local ? capacity : 262144,
                                      local ? static_cast<int>(profile->window) : 0, 262144);
          const auto bytes = ggml_nbytes(node);
          const auto address = Allocate(bytes + 512);
          TensorArena::Bind(node, address + 256);
          ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(address), 0xa5, bytes + 512, stream),
                    cudaSuccess);
          const auto expected = [&] {
            std::vector<std::uint16_t> mask(bytes / 2, 0xFC00);
            const int end = positions[static_cast<std::size_t>(segment * rows)] + rows;
            for (int r = 0; r < rows; ++r) {
              const int query = positions[static_cast<std::size_t>(segment * rows + r)];
              if (query < 0 || query >= 262144) continue;
              if (!local) {
                std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(r * read_cells),
                            std::min(read_cells, query + 1), 0);
              } else {
                // Independent latest-retained absolute cell map after whole chunk.
                for (int cell = 0; cell < read_cells; ++cell) {
                  if (cell >= end) continue;
                  const int held = cell + ((end - 1 - cell) / capacity) * capacity;
                  if (held <= query && query - held < static_cast<int>(profile->window))
                    mask[static_cast<std::size_t>(r * read_cells + cell)] = 0;
                }
              }
            }
            return mask;
          };
          ASSERT_TRUE(kg::CheckGemma4Mask(node));
          launch().ResetScratchPeak();
          ASSERT_TRUE(kg::RunGemma4Mask(launch(), node));
          EXPECT_EQ(Download<std::uint16_t>(node), expected());
          EXPECT_EQ(launch().scratch_peak().value(), 0);
          auto captured =
              launch().Capture([&](auto& context) { return kg::RunGemma4Mask(context, node); });
          ASSERT_TRUE(captured);
          // Every row is a fresh position value; address/shape/capture stay fixed.
          const int delta = segments == 16 ? -32 : 32;
          for (int r = 0; r < rows; ++r)
            positions[static_cast<std::size_t>(segment * rows + r)] += delta;
          ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
          ASSERT_TRUE(launch().Launch(*captured));
          EXPECT_EQ(Download<std::uint16_t>(node), expected());
          for (int r = 0; r < rows; ++r)
            positions[static_cast<std::size_t>(segment * rows + r)] -= delta;
          ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
          // Invalid position values are safe masked rows, never unsigned indices.
          for (const auto bad_position : {-1, std::numeric_limits<std::int32_t>::max()}) {
            auto invalid = positions;
            invalid[static_cast<std::size_t>(segment * rows)] = bad_position;
            ASSERT_EQ(cudaMemcpyAsync(input->data, invalid.data(), ggml_nbytes(input),
                                      cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            ASSERT_TRUE(launch().Launch(*captured));
            const auto actual = Download<std::uint16_t>(node);
            EXPECT_TRUE(std::all_of(actual.begin(), actual.begin() + read_cells,
                                    [](auto value) { return value == 0xFC00; }));
          }
          ASSERT_EQ(cudaMemcpyAsync(input->data, positions.data(), ggml_nbytes(input),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
          const auto old_type = input->type;
          input->type = GGML_TYPE_F32;
          const auto before = Download<std::uint16_t>(node);
          EXPECT_FALSE(kg::RunGemma4Mask(launch(), node));
          EXPECT_EQ(before, Download<std::uint16_t>(node));
          EXPECT_FALSE(launch().faulted());
          input->type = old_type;
          std::array<std::uint8_t, 256> prefix{}, suffix{};
          Finish();
          ASSERT_EQ(cudaMemcpy(prefix.data(), reinterpret_cast<void*>(address), 256,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpy(suffix.data(), reinterpret_cast<void*>(address + 256 + bytes), 256,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_TRUE(
              std::all_of(prefix.begin(), prefix.end(), [](auto value) { return value == 0xa5; }));
          EXPECT_EQ(prefix, suffix);
        }
      }
    }
  }
  // Upstream's mask prepass reads 1,025 logical queries rounded to 1,056.
  constexpr int rows = 1025, cells = 1280;
  std::vector<std::int32_t> positions(rows);
  std::iota(positions.begin(), positions.end(), 0);
  auto* input = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows), positions);
  for (const int window : {0, 1024}) {
    auto* mask = Place(
        kg::Gemma4Mask(c(), input, cells, 0, rows, window == 0 ? 262144 : 2304, window, 262144));
    ASSERT_EQ(mask->ne[1], 1056);
    ASSERT_TRUE(kg::RunGemma4Mask(launch(), mask));
    const auto actual = Download<std::uint16_t>(mask);
    for (int r = 0; r < 1056; ++r)
      for (int cell = 0; cell < cells; ++cell) {
        const bool visible = r < rows && cell <= r && (window == 0 || r - cell < window);
        ASSERT_EQ(actual[static_cast<std::size_t>(r * cells + cell)], visible ? 0 : 0xFC00);
      }
  }
}

// ---- Quantized weights ----

struct Quantized {
  std::vector<std::uint8_t> bytes;  // GGML's blocks, as a GGUF stores them
  std::vector<double> values;       // their dequantized values
};

// `rows` rows of `k` values in `type`, quantized by GGML's own quantizer
// (with a flat importance matrix where the type needs one) and dequantized
// by GGML's own to_float.
Quantized Quantize(ggml_type type, std::int64_t k, std::int64_t rows, std::uint64_t seed) {
  const std::vector<float> source = Normal(seed, static_cast<std::size_t>(k * rows), 0.05f);
  const std::vector<float> importance(static_cast<std::size_t>(k), 1.0f);
  Quantized out;
  out.bytes.resize(ggml_row_size(type, k) * static_cast<std::size_t>(rows));
  const std::size_t written =
      ggml_quantize_chunk(type, source.data(), out.bytes.data(), 0, rows, k,
                          ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
  EXPECT_EQ(written, out.bytes.size());
  const auto* traits = ggml_get_type_traits(type);
  std::vector<float> row(static_cast<std::size_t>(k));
  out.values.reserve(static_cast<std::size_t>(k * rows));
  for (std::int64_t r = 0; r < rows; ++r) {
    traits->to_float(out.bytes.data() + (static_cast<std::size_t>(r) * ggml_row_size(type, k)),
                     row.data(), k);
    for (const float v : row) {
      out.values.push_back(v);
    }
  }
  return out;
}

// The quantized products' bound on this device: MXFP4's tile kernel on a
// Blackwell GPU quantizes the activations to FP4 (test-backend-ops.cpp:
// 4850-4855).
double QuantizedBound(ggml_type type, QuantMulMatPath path, int cc) {
  return type == GGML_TYPE_MXFP4 && path == QuantMulMatPath::kTile && cc >= 1200
             ? kFp4ActivationNmse
             : kMulMatNmse;
}

TEST_F(GgmlExtOpsTest, ProductPlansRefuseCombinedWeightBlockOffsets) {
  for (const ggml_type type : kg::QuantizedWeightTypes()) {
    SCOPED_TRACE(ggml_type_name(type));
    auto arena = TensorArena::Create(64).value();
    auto* context = arena.context();
    std::uint64_t next = 1ULL << 44;
    const auto bound = [&next](ggml_tensor* tensor) {
      TensorArena::Bind(tensor, next);
      next += 1ULL << 42;
      return tensor;
    };
    for (const bool routed : {false, true}) {
      for (const bool broadcast : {false, true}) {
        auto* w = bound(ggml_new_tensor_3d(context, type, 512, 128, 3));
        // Each block stride fits an int, but selecting expert/channel 2
        // wraps its combined signed block offset past INT32_MAX.
        w->nb[2] = (1ULL << 30) * ggml_type_size(type);
        w->nb[3] = w->nb[2];  // unused sample stride stays representable
        auto* input = bound(ggml_new_tensor_3d(context, GGML_TYPE_F32, 512,
                                               routed && !broadcast ? 2 : 1, routed ? 1 : 3));
        auto* ids = bound(ggml_new_tensor_2d(context, GGML_TYPE_I32, 2, 1));
        auto* node = bound(routed ? ggml_mul_mat_id(context, w, input, ids)
                                  : ggml_mul_mat(context, w, input));
        EXPECT_FALSE((routed ? kg::CheckMulMatIdQ(node) : kg::CheckMulMatQ(node)).has_value());
        EXPECT_FALSE(kg::PlanMulMatVecQ(launch(), node).has_value());
        EXPECT_FALSE(kg::PlanMulMatVecQRows(launch(), node).has_value());
        EXPECT_FALSE(kg::PlanMulMatQ(launch(), node).has_value());
        // These are planning-only calls over fictional addresses. A
        // rejected product must never try to submit either operand.
        EXPECT_FALSE(launch().faulted());
        w->nb[2] = ((1ULL << 29) * ggml_type_size(type));
        w->nb[3] = w->nb[2];
        EXPECT_TRUE(kg::PlanMulMatVecQ(launch(), node).has_value());
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, OrdinaryMmvqRefusesRoundedWeightRowsBeforeSubmission) {
  for (const ggml_type type : kg::QuantizedWeightTypes()) {
    auto arena = TensorArena::Create(64).value();
    for (const bool routed : {false, true}) {
      for (const std::int64_t columns : {1, 2, 4, 8}) {
        auto* w = Place(ggml_new_tensor_3d(arena.context(), type, 512, 129, routed ? 2 : 1));
        auto* x = Place(ggml_new_tensor_3d(arena.context(), GGML_TYPE_F32, 512,
                                           routed ? 1 : columns, routed ? columns : 1));
        auto* ids = routed
                        ? Place(ggml_new_tensor_2d(arena.context(), GGML_TYPE_I32, 2, columns),
                                std::vector<std::int32_t>(static_cast<std::size_t>(2 * columns), 0))
                        : nullptr;
        auto* node = Place(routed ? ggml_mul_mat_id(arena.context(), w, x, ids)
                                  : ggml_mul_mat(arena.context(), w, x));
        const int rpb = kg::MmvqRowsPerBlock(launch(), node);
        if (routed && columns > 1) {
          EXPECT_EQ(rpb, 2);  // pinned dedicated multi-token MoE geometry
        }
        ASSERT_GT(rpb, 0);
        if (rpb == 1) {
          continue;  // this selected one-row launch has no rounded read
        }
        ASSERT_EQ(cudaMemset(node->data, 0xAB, ggml_nbytes(node)), cudaSuccess);
        EXPECT_EQ(FailedCode(kg::PlanMulMatVecQ(launch(), node)), KernelError::kRejected);
        EXPECT_EQ(FailedCode(kg::MulMatVecQ(launch(), node)), KernelError::kRejected);
        const auto untouched = Download<std::uint8_t>(node);
        EXPECT_TRUE(std::ranges::all_of(untouched, [](std::uint8_t b) { return b == 0xAB; }));
        EXPECT_FALSE(launch().faulted());
      }
    }
  }
}

TEST_F(GgmlExtOpsTest, QuantizedProductsMatchTheReferenceAsUpstreamRoutesThem) {
  const int cc = ComputeCapability();
  // DeepSeek V4 Flash UD-Q2_K_XL's weight types, with their projections' k:
  // 4096 into attention, shared-expert gate/up and expert gate/up, 2048
  // into the down projections.
  struct Case {
    ggml_type type;
    std::int64_t k;
  };
  const std::array<Case, 14> cases = {{{GGML_TYPE_Q4_0, 704},
                                       {GGML_TYPE_Q4_1, 2816},
                                       {GGML_TYPE_Q5_0, 2816},
                                       {GGML_TYPE_Q5_1, 704},
                                       {GGML_TYPE_IQ4_NL, 2816},
                                       {GGML_TYPE_Q8_0, 4096},
                                       {GGML_TYPE_Q2_K, 2048},
                                       {GGML_TYPE_IQ2_XXS, 4096},
                                       {GGML_TYPE_Q4_K, 4096},
                                       {GGML_TYPE_Q5_K, 4096},
                                       {GGML_TYPE_Q6_K, 2048},
                                       {GGML_TYPE_IQ2_XS, 4096},
                                       {GGML_TYPE_IQ3_XXS, 2048},
                                       {GGML_TYPE_MXFP4, 2048}}};
  constexpr std::int64_t kRowsOut = 256;
  for (const Case& test : cases) {
    const std::string type = ggml_type_name(test.type);
    const Quantized weights = Quantize(test.type, test.k, kRowsOut, 11);
    for (const std::int64_t columns : {1, 2, 4, 16, 48}) {
      ggml_tensor* w = Place(ggml_new_tensor_2d(c(), test.type, test.k, kRowsOut), weights.bytes);
      kg::MarkRowPaddingReadable(w);
      const std::vector<float> x = Normal(12, static_cast<std::size_t>(test.k * columns));
      ggml_tensor* input = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, test.k, columns), x);
      ggml_tensor* product = Place(ggml_mul_mat(c(), w, input));
      const std::string what = type + " x " + std::to_string(columns);
      auto path = kg::SelectMulMatQ(launch(), product);
      ASSERT_TRUE(path.has_value()) << what << ": " << path.error().detail;
      // Upstream's routing on a GB10: the vector kernel up to 8 columns.
      EXPECT_EQ(*path, columns <= 8 ? QuantMulMatPath::kVector : QuantMulMatPath::kTile) << what;
      if (*path == QuantMulMatPath::kVector) {
        Launched(kg::MulMatVecQ(launch(), product), what);
        EXPECT_EQ(FailedCode(kg::MulMatQ(launch(), product)), KernelError::kRejected) << what;
      } else {
        Launched(kg::MulMatQ(launch(), product), what);
        EXPECT_EQ(FailedCode(kg::MulMatVecQ(launch(), product)), KernelError::kRejected) << what;
      }
      std::vector<double> want(static_cast<std::size_t>(kRowsOut * columns));
      for (std::int64_t j = 0; j < columns; ++j) {
        for (std::int64_t r = 0; r < kRowsOut; ++r) {
          double sum = 0.0;
          for (std::int64_t i = 0; i < test.k; ++i) {
            sum += weights.values[static_cast<std::size_t>((r * test.k) + i)] *
                   x[static_cast<std::size_t>((j * test.k) + i)];
          }
          want[static_cast<std::size_t>((j * kRowsOut) + r)] = sum;
        }
      }
      ExpectNmse(Download(product), want, QuantizedBound(test.type, *path, cc), what);
    }
  }
}

// Upstream's fused one-column gate/up/GLU MMVQ (stock's decode FFN) writes
// what the two products and the split GLU write, bit for bit, for Gemma's
// GeGLU and SwiGLU; it is refused where upstream does not fuse.
TEST_F(GgmlExtOpsTest, SharedPreparationKeepsOrdinaryScalarThroughEightMmvqAndChangedReplayExact) {
  // A non-512-aligned width requires authentic readable weight padding; the
  // original launch and the shared producer must agree on every padded byte.
  constexpr std::int64_t kWidth = 2304;
  constexpr std::array<std::int64_t, 3> kOutputs{128, 64, 64};
  for (const auto type :
       {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
    for (const std::int64_t columns : {1, 2, 3, 4, 5, 6, 7, 8}) {
      SCOPED_TRACE(std::format("{} C{}", ggml_type_name(type), columns));
      auto metadata = TensorArena::Create(32);
      ASSERT_TRUE(metadata);
      auto* context = metadata->context();
      auto first = Normal(194, static_cast<std::size_t>(kWidth * columns));
      auto changed = first;
      for (std::int64_t i = 0; i < kWidth; ++i)
        changed[static_cast<std::size_t>(i)] = -first[static_cast<std::size_t>(i)] + 0.25F;
      auto* x = Place(ggml_new_tensor_2d(context, GGML_TYPE_F32, kWidth, columns), first);
      kg::SharedQ8Inputs shared(context);
      const auto guarded = [&](ggml_tensor* tensor) {
        const auto bytes = ggml_nbytes(tensor);
        const auto address = Allocate(bytes + 256);
        TensorArena::Bind(tensor, address);
        EXPECT_EQ(cudaMemset(reinterpret_cast<void*>(address + bytes), 0xA5, 256), cudaSuccess);
        return tensor;
      };
      auto* q8 = guarded(shared.Get(x));
      ASSERT_EQ(shared.Get(x), q8);
      std::array<ggml_tensor*, 3> ordinary{}, candidate{}, weights{};
      std::array<std::vector<std::uint8_t>, 3> weight_bytes;
      for (std::size_t i = 0; i < kOutputs.size(); ++i) {
        const auto quant =
            Quantize(type, kWidth, kOutputs[i], 195U + static_cast<std::uint64_t>(i));
        auto* w = Place(ggml_new_tensor_2d(context, type, kWidth, kOutputs[i]), quant.bytes);
        kg::MarkRowPaddingReadable(w);
        weights[i] = w;
        weight_bytes[i] = quant.bytes;
        ordinary[i] = Place(ggml_mul_mat(context, w, x));
        candidate[i] =
            guarded(shared.Product(w, x, true, kg::DeviceChoicesOf(launch()).dense_mmvq_shape));
        ASSERT_EQ(kg::JitllmOpOf(candidate[i]), kg::JitllmOp::kMmvqPrepared);
        ASSERT_EQ(candidate[i]->src[1], q8);
        ASSERT_EQ(candidate[i]->src[2], x);
        const auto original_plan = kg::PlanMulMatVecQ(launch(), ordinary[i]);
        ASSERT_TRUE(original_plan);
        EXPECT_EQ(*original_plan, static_cast<std::uint64_t>(kg::Q8Bytes(kWidth, columns)));
        const auto plan = kg::PlanMmvqPrepared(launch(), candidate[i]);
        ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
        EXPECT_EQ(*plan, 0U);
      }
      const auto run = [&](auto& context) -> std::expected<void, KernelFailure> {
        if (auto r = kg::RunQuantizeQ8(context, q8); !r) return r;
        for (auto* product : candidate)
          if (auto r = kg::RunMmvqPrepared(context, product); !r) return r;
        return {};
      };
      std::array<std::vector<float>, 3> baseline;
      for (std::size_t i = 0; i < ordinary.size(); ++i) {
        ASSERT_TRUE(kg::MulMatVecQ(launch(), ordinary[i]));
        baseline[i] = Download(ordinary[i]);
      }
      ASSERT_TRUE(run(launch()));
      for (std::size_t i = 0; i < candidate.size(); ++i) {
        const auto got = Download(candidate[i]);
        ASSERT_EQ(got.size(), static_cast<std::size_t>(kOutputs[i] * columns));
        ASSERT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); }));
        EXPECT_EQ(got, baseline[i]);
        EXPECT_EQ(std::memcmp(got.data(), baseline[i].data(), got.size() * sizeof(float)), 0);
      }
      auto graph = launch().Capture(run);
      ASSERT_TRUE(graph);
      ASSERT_EQ(cudaMemcpy(x->data, changed.data(), ggml_nbytes(x), cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::array<std::vector<float>, 3> expected;
      for (std::size_t i = 0; i < ordinary.size(); ++i) {
        ASSERT_TRUE(kg::MulMatVecQ(launch(), ordinary[i]));
        expected[i] = Download(ordinary[i]);
        ASSERT_NE(expected[i], baseline[i]);
        if (columns > 1)
          EXPECT_TRUE(std::equal(expected[i].begin() + kOutputs[i], expected[i].end(),
                                 baseline[i].begin() + kOutputs[i]));
      }
      ASSERT_EQ(cudaMemset(q8->data, 0xFF, ggml_nbytes(q8)), cudaSuccess);
      for (auto* output : candidate)
        ASSERT_EQ(cudaMemset(output->data, 0xFF, ggml_nbytes(output)), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      ASSERT_TRUE(launch().Launch(*graph));
      for (std::size_t i = 0; i < candidate.size(); ++i) {
        const auto got = Download(candidate[i]);
        EXPECT_EQ(got, expected[i]);
        EXPECT_EQ(std::memcmp(got.data(), expected[i].data(), got.size() * sizeof(float)), 0);
      }
      EXPECT_EQ(Download(x), changed);
      for (std::size_t i = 0; i < weights.size(); ++i)
        EXPECT_EQ(Download<std::uint8_t>(weights[i]), weight_bytes[i]);
      for (const auto* tensor : {q8, candidate[0], candidate[1], candidate[2]}) {
        std::array<std::uint8_t, 256> guard{};
        ASSERT_EQ(cudaMemcpy(guard.data(),
                             static_cast<const std::byte*>(tensor->data) + ggml_nbytes(tensor),
                             guard.size(), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_TRUE(std::ranges::all_of(guard, [](auto v) { return v == 0xA5; }));
      }
      const auto output_before = Download(candidate[0]);
      const auto saved_q8 = *q8;
      q8->nb[0] = 0;
      EXPECT_FALSE(kg::RunQuantizeQ8(launch(), q8));
      EXPECT_FALSE(kg::RunMmvqPrepared(launch(), candidate[0]));
      *q8 = saved_q8;
      auto refused = *candidate[0];
      refused.src[2] = candidate[1];
      EXPECT_FALSE(kg::RunMmvqPrepared(launch(), &refused));
      refused = *candidate[0];
      refused.data = q8->data;
      EXPECT_FALSE(kg::RunMmvqPrepared(launch(), &refused));
      refused = *candidate[0];
      refused.src[7] = x;
      EXPECT_FALSE(kg::RunMmvqPrepared(launch(), &refused));
      EXPECT_EQ(Download(candidate[0]), output_before);
    }
  }
}

TEST_F(GgmlExtOpsTest, FusedQuantizedGluEqualsItsProductsAndGlu) {
  constexpr std::int64_t kK = 5376;
  constexpr std::int64_t kOut = 512;
  for (const ggml_type type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0}) {
    const Quantized gate_w = Quantize(type, kK, kOut, 21);
    const Quantized up_w = Quantize(type, kK, kOut, 22);
    for (const bool geglu : {true, false}) {
      const std::string what = std::string(ggml_type_name(type)) + (geglu ? " GeGLU" : " SwiGLU");
      ggml_tensor* wg = Place(ggml_new_tensor_2d(c(), type, kK, kOut), gate_w.bytes);
      ggml_tensor* wu = Place(ggml_new_tensor_2d(c(), type, kK, kOut), up_w.bytes);
      kg::MarkRowPaddingReadable(wg);
      kg::MarkRowPaddingReadable(wu);
      for (const std::int64_t columns : {1, 4}) {
        ggml_tensor* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kK, columns),
                               Normal(23, static_cast<std::size_t>(kK * columns)));
        ggml_tensor* gate = Place(ggml_mul_mat(c(), wg, x));
        ggml_tensor* up = Place(ggml_mul_mat(c(), wu, x));
        ggml_tensor* glu =
            Place(geglu ? ggml_geglu_split(c(), gate, up) : ggml_swiglu_split(c(), gate, up));
        if (columns != 1) {
          EXPECT_FALSE(kg::MulMatVecQGluFusible(launch(), gate, up, glu)) << what;
          EXPECT_EQ(FailedCode(kg::MulMatVecQGlu(launch(), gate, up, glu)), KernelError::kRejected)
              << what;
          continue;
        }
        ASSERT_TRUE(kg::MulMatVecQGluFusible(launch(), gate, up, glu)) << what;
        Launched(kg::MulMatVecQ(launch(), gate), what + " gate");
        Launched(kg::MulMatVecQ(launch(), up), what + " up");
        Launched(geglu ? kg::GeGlu(launch(), glu) : kg::SwiGlu(launch(), glu), what + " GLU");
        const auto separate = Download(glu);
        Launched(kg::MulMatVecQGlu(launch(), gate, up, glu), what + " fused");
        const auto fused = Download(glu);
        ASSERT_EQ(fused.size(), separate.size());
        EXPECT_EQ(std::memcmp(fused.data(), separate.data(), fused.size() * sizeof(float)), 0)
            << what;
        // A different gate input is not the up product's: no fusion.
        ggml_tensor* other = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kK, 1),
                                   Normal(24, static_cast<std::size_t>(kK)));
        ggml_tensor* gate2 = Place(ggml_mul_mat(c(), wg, other));
        ggml_tensor* glu2 = Place(ggml_geglu_split(c(), gate2, up));
        EXPECT_FALSE(kg::MulMatVecQGluFusible(launch(), gate2, up, glu2)) << what;
      }
    }
  }
  EXPECT_FALSE(launch().faulted());
}

TEST_F(GgmlExtOpsTest, ExpertProductsMatchTheReferenceAsUpstreamRoutesThem) {
  const int cc = ComputeCapability();
  // DeepSeek V4's MoE at small scale: 16 experts of 64 rows, 6 selected per
  // token. Gate and up (IQ2_XS) read each token's one activation row
  // (broadcast over the selected experts); down (IQ3_XXS and MXFP4) reads
  // one row per selected expert.
  constexpr std::int64_t kExperts = 16;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kOut = 64;
  struct Case {
    ggml_type type;
    std::int64_t k;
    bool broadcast;
  };
  const std::array<Case, 8> cases = {{{GGML_TYPE_Q4_1, 2816, true},
                                      {GGML_TYPE_Q5_0, 2816, true},
                                      {GGML_TYPE_Q5_1, 704, false},
                                      {GGML_TYPE_IQ2_XXS, 4096, true},
                                      {GGML_TYPE_Q2_K, 2048, false},
                                      {GGML_TYPE_IQ2_XS, 4096, true},
                                      {GGML_TYPE_IQ3_XXS, 2048, false},
                                      {GGML_TYPE_MXFP4, 2048, false}}};
  for (const Case& test : cases) {
    const Quantized weights = Quantize(test.type, test.k, kOut * kExperts, 21);
    for (const std::int64_t tokens : {1, 3, 40}) {
      const std::string what = std::string(ggml_type_name(test.type)) + " experts x " +
                               std::to_string(tokens) + (test.broadcast ? " (broadcast)" : "");
      ggml_tensor* w =
          Place(ggml_new_tensor_3d(c(), test.type, test.k, kOut, kExperts), weights.bytes);
      kg::MarkRowPaddingReadable(w);
      const std::int64_t rows_in = test.broadcast ? 1 : kUsed;
      const std::vector<float> x = Normal(22, static_cast<std::size_t>(test.k * rows_in * tokens));
      ggml_tensor* input =
          Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, rows_in, tokens), x);
      // Distinct experts per token, as routing selects them.
      std::mt19937 random(static_cast<unsigned>(23 + tokens));
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < tokens; ++t) {
        std::vector<std::int32_t> experts(kExperts);
        std::ranges::iota(experts, 0);
        std::shuffle(experts.begin(), experts.end(), random);
        ids.insert(ids.end(), experts.begin(), experts.begin() + kUsed);
      }
      ggml_tensor* id_tensor = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, tokens), ids);
      ggml_tensor* product = Place(ggml_mul_mat_id(c(), w, input, id_tensor));
      auto path = kg::SelectMulMatQ(launch(), product);
      ASSERT_TRUE(path.has_value()) << what << ": " << path.error().detail;
      EXPECT_EQ(*path, tokens <= 8 ? QuantMulMatPath::kVector : QuantMulMatPath::kTile) << what;
      Launched(*path == QuantMulMatPath::kVector ? kg::MulMatVecQ(launch(), product)
                                                 : kg::MulMatQ(launch(), product),
               what);
      std::vector<double> want(static_cast<std::size_t>(kOut * kUsed * tokens));
      for (std::int64_t t = 0; t < tokens; ++t) {
        for (std::int64_t s = 0; s < kUsed; ++s) {
          const std::int64_t expert = ids[static_cast<std::size_t>((t * kUsed) + s)];
          const std::int64_t row_in = test.broadcast ? 0 : s;
          for (std::int64_t r = 0; r < kOut; ++r) {
            double sum = 0.0;
            for (std::int64_t i = 0; i < test.k; ++i) {
              const auto wi = static_cast<std::size_t>((((expert * kOut) + r) * test.k) + i);
              const auto xi = static_cast<std::size_t>((((t * rows_in) + row_in) * test.k) + i);
              sum += weights.values[wi] * x[xi];
            }
            want[static_cast<std::size_t>((((t * kUsed) + s) * kOut) + r)] = sum;
          }
        }
      }
      ExpectNmse(Download(product), want, QuantizedBound(test.type, *path, cc), what);
    }
  }
}

TEST_F(GgmlExtOpsTest, ExpertProductsCoverSkewedRoutingAndEmptyExperts) {
  constexpr std::int64_t kExperts = 64;
  constexpr std::int64_t kUsed = 2;
  constexpr std::int64_t kOut = 128;
  constexpr std::int64_t kInner = 512;
  constexpr std::int64_t kTokens = 257;
  for (const ggml_type type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, GGML_TYPE_Q8_0}) {
    for (const bool broadcast : {true, false}) {
      const auto weights = Quantize(type, kInner, kOut * kExperts, 37);
      auto* w = Place(ggml_new_tensor_3d(c(), type, kInner, kOut, kExperts), weights.bytes);
      const std::int64_t rows_in = broadcast ? 1 : kUsed;
      const auto x = Normal(38, static_cast<std::size_t>(kInner * rows_in * kTokens));
      auto* input = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kInner, rows_in, kTokens), x);
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < kTokens; ++t) {
        ids.push_back(0);  // one expert has many full tiles and a partial tail
        ids.push_back(static_cast<std::int32_t>(1 + (t % 31)));  // others are empty
      }
      auto* id_tensor = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, kTokens), ids);
      auto* product = Place(ggml_mul_mat_id(c(), w, input, id_tensor));
      const std::string what =
          std::string(ggml_type_name(type)) + (broadcast ? " broadcast" : " slots");
      Launched(kg::MulMatQ(launch(), product), what);
      std::vector<double> want(static_cast<std::size_t>(kOut * kUsed * kTokens));
      for (std::int64_t t = 0; t < kTokens; ++t) {
        for (std::int64_t s = 0; s < kUsed; ++s) {
          const auto expert = ids[static_cast<std::size_t>((t * kUsed) + s)];
          for (std::int64_t r = 0; r < kOut; ++r) {
            double sum = 0;
            for (std::int64_t i = 0; i < kInner; ++i) {
              sum +=
                  weights.values[static_cast<std::size_t>((((expert * kOut) + r) * kInner) + i)] *
                  x[static_cast<std::size_t>((((t * rows_in) + (broadcast ? 0 : s)) * kInner) + i)];
            }
            want[static_cast<std::size_t>((((t * kUsed) + s) * kOut) + r)] = sum;
          }
        }
      }
      ExpectNmse(Download(product), want, kMulMatNmse, what);
    }
  }
}

TEST_F(GgmlExtOpsTest, PairedExpertPreparationMatchesSeparateProductsExactly) {
  constexpr std::int64_t kInner = 512;
  constexpr std::int64_t kOut = 128;
  constexpr std::int64_t kExperts = 64;
  constexpr std::int64_t kUsed = 2;
  constexpr std::int64_t kTokens = 257;
  for (const auto type : kg::QuantizedWeightTypes()) {
    if (type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4) {
      continue;
    }
    for (const bool broadcast : {true, false}) {
      std::array<ggml_tensor*, 2> weights{};
      for (std::size_t wi = 0; wi < weights.size(); ++wi) {
        auto quantized = Quantize(type, kInner, kOut * kExperts, static_cast<unsigned>(51 + wi));
        auto* w = ggml_new_tensor_3d(c(), type, kInner, kOut, kExperts);
        const std::size_t slice = w->nb[2];
        const std::size_t unit = std::lcm(ggml_type_size(type), std::size_t{256});
        ASSERT_NE(unit, 0U);
        const std::size_t stride = ((slice + (3000 * wi) + unit - 1) / unit) * unit;
        w->nb[2] = stride;
        w->nb[3] = stride * kExperts;
        std::vector<std::uint8_t> padded(ggml_nbytes(w), 0xAB);
        for (std::size_t e = 0; e < static_cast<std::size_t>(kExperts); ++e) {
          std::memcpy(padded.data() + (e * stride), quantized.bytes.data() + (e * slice), slice);
        }
        weights[wi] = Place(w, padded);
      }
      const auto slots = broadcast ? 1 : kUsed;
      auto* input = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kInner, slots, kTokens),
                          Normal(53, static_cast<std::size_t>(kInner * slots * kTokens)));
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < kTokens; ++t) {
        ids.push_back(0);
        ids.push_back(static_cast<std::int32_t>(1 + (t % 7)));
      }
      auto* routing = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, kTokens), ids);
      auto* first = Place(ggml_mul_mat_id(c(), weights[0], input, routing));
      auto* second = Place(ggml_mul_mat_id(c(), weights[1], input, routing));
      const std::string what =
          std::string(ggml_type_name(type)) + (broadcast ? " broadcast" : " slots");
      Launched(kg::MulMatQ(launch(), first), what);
      const auto want_first = Download(first);
      Launched(kg::MulMatQ(launch(), second), what);
      const auto want_second = Download(second);
      // Compact tiles use a full-K reduction; an ordinary low-efficiency
      // grid can instead split K and add its fixup. Bound only that
      // reduction movement tightly, before target comparisons.
      constexpr double kCompactNmse = 1e-10;
      const std::vector<double> compact_want_first(want_first.begin(), want_first.end());
      const std::vector<double> compact_want_second(want_second.begin(), want_second.end());
      const auto compact_scratch = kg::PlanMulMatIdQCompact(launch(), first);
      ASSERT_TRUE(compact_scratch.has_value()) << what << ": " << compact_scratch.error().detail;
      ASSERT_EQ(cudaMemset(first->data, 0xFF, ggml_nbytes(first)), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      launch().ResetScratchPeak();
      Launched(kg::MulMatIdQCompact(launch(), first), what);
      EXPECT_LE(launch().scratch_peak().value(), *compact_scratch);
      ExpectNmse(Download(first), compact_want_first, kCompactNmse, what);
      const auto scratch = kg::PlanMulMatIdQPair(launch(), first, second);
      ASSERT_TRUE(scratch.has_value()) << what << ": " << scratch.error().detail;
      ASSERT_EQ(cudaMemset(first->data, 0xFF, ggml_nbytes(first)), cudaSuccess);
      ASSERT_EQ(cudaMemset(second->data, 0xFF, ggml_nbytes(second)), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      launch().ResetScratchPeak();
      Launched(kg::MulMatIdQPair(launch(), first, second), what);
      EXPECT_LE(launch().scratch_peak().value(), *scratch);
      EXPECT_EQ(Download(first), want_first) << what;
      EXPECT_EQ(Download(second), want_second) << what;
      const auto compact_pair_scratch = kg::PlanMulMatIdQPair(launch(), first, second, true);
      ASSERT_TRUE(compact_pair_scratch.has_value())
          << what << ": " << compact_pair_scratch.error().detail;
      ASSERT_EQ(cudaMemset(first->data, 0xFF, ggml_nbytes(first)), cudaSuccess);
      ASSERT_EQ(cudaMemset(second->data, 0xFF, ggml_nbytes(second)), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      launch().ResetScratchPeak();
      Launched(kg::MulMatIdQPair(launch(), first, second, true), what);
      EXPECT_LE(launch().scratch_peak().value(), *compact_pair_scratch);
      ExpectNmse(Download(first), compact_want_first, kCompactNmse, what);
      ExpectNmse(Download(second), compact_want_second, kCompactNmse, what);
      const std::array<ggml_tensor*, 2> nodes = {first, second};
      auto choices = kg::DeviceChoicesOf(launch());
      const auto primitive = kg::PlanGraph(nodes, false, choices);
      ASSERT_TRUE(primitive.has_value()) << what << ": " << primitive.error().detail;
      ASSERT_EQ(primitive->steps.size(), 2U);
      EXPECT_EQ(primitive->steps[0].implementation, kg::kMulMatIdQ);
      EXPECT_EQ(primitive->steps[1].implementation, kg::kMulMatIdQ);
      choices.pair_experts = true;
      const auto plan = kg::PlanGraph(nodes, false, choices);
      ASSERT_TRUE(plan.has_value()) << what << ": " << plan.error().detail;
      ASSERT_EQ(plan->steps.size(), 1U);
      EXPECT_EQ(plan->steps[0].implementation, kg::kMulMatIdQPair);
      EXPECT_EQ(plan->steps[0].nodes.size(), 2U);
      choices.compact_experts = true;
      const auto compact_pair = kg::PlanGraph(nodes, false, choices);
      ASSERT_TRUE(compact_pair.has_value()) << what << ": " << compact_pair.error().detail;
      ASSERT_EQ(compact_pair->steps.size(), 1U);
      EXPECT_EQ(compact_pair->steps[0].implementation, kg::kMulMatIdQPairCompact);
      choices.pair_experts = false;
      const auto compact = kg::PlanGraph(nodes, false, choices);
      ASSERT_TRUE(compact.has_value()) << what << ": " << compact.error().detail;
      ASSERT_EQ(compact->steps.size(), 2U);
      EXPECT_EQ(compact->steps[0].implementation, kg::kMulMatIdQCompact);
      EXPECT_EQ(compact->steps[1].implementation, kg::kMulMatIdQCompact);
    }
  }
}

TEST_F(GgmlExtOpsTest, RawQ2D2rPreservesItsOwnCapturedProductAndPlansAllScratch) {
  constexpr std::int64_t k = 512;
  constexpr std::int64_t m = 130;
  constexpr std::int64_t experts = 16;
  constexpr std::int64_t used = 6;
  constexpr std::int64_t tokens = 129;
  auto weights = Quantize(GGML_TYPE_Q2_K, k, m * experts, 71);
  auto* w = Place(ggml_new_tensor_3d(c(), GGML_TYPE_Q2_K, k, m, experts), weights.bytes);
  auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, k, used, tokens),
                  Normal(72, static_cast<std::size_t>(k * used * tokens)));
  std::vector<std::int32_t> ids(static_cast<std::size_t>(used * tokens));
  for (std::int64_t t = 0; t < tokens; ++t) {
    for (std::int64_t slot = 0; slot < used; ++slot) {
      ids[static_cast<std::size_t>((t * used) + slot)] = static_cast<std::int32_t>(slot);
    }
  }
  auto* routes = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, used, tokens), ids);
  auto* product = Place(ggml_mul_mat_id(c(), w, x, routes));
  auto* other = Place(ggml_mul_mat_id(c(), w, x, routes));
  const auto scratch = kg::PlanMulMatIdQ2D2r(launch(), product);
  ASSERT_TRUE(scratch.has_value()) << (scratch ? "" : scratch.error().detail);
  const kg::GraphPlan plan{.steps = {{.operation = jitllm::execution::Operation::kMulMatId,
                                      .implementation = kg::kMulMatIdQ2D2r,
                                      .nodes = {product},
                                      .lane = 0}},
                           .regions = {}};
  const auto planned = kg::PlanScratch(launch(), plan);
  ASSERT_TRUE(planned.has_value()) << (planned ? "" : planned.error().detail);
  EXPECT_EQ(*planned, *scratch);
  EXPECT_GT(*scratch, 0U);
  auto insufficient = LaunchContext::Create(
      0, *execution_, stream_, {.base = Allocate(*scratch), .size = Bytes(*scratch - 1)});
  ASSERT_TRUE(insufficient.has_value()) << (insufficient ? "" : insufficient.error().detail);
  const auto refused = kg::MulMatIdQ2D2r(**insufficient, product);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, KernelError::kRejected);
  insufficient->reset();
  launch().ResetScratchPeak();
  Launched(kg::MulMatIdQ2D2r(launch(), product), "raw Q2 warmup");
  EXPECT_LE(launch().scratch_peak().value(), *scratch);
  const auto first = Download(product);
  Launched(kg::MulMatIdQ2D2r(launch(), product), "raw Q2 own repeat");
  EXPECT_EQ(Download(product), first);
  Launched(kg::MulMatIdQCompact(launch(), other), "raw Q2 original control");
  const auto control = Download(other);
  ExpectNmse(first, std::vector<double>(control.begin(), control.end()), kMulMatNmse,
             "raw Q2 versus native compact approximation");
  auto graph = launch().Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
    if (auto r = kg::MulMatIdQ2D2r(l, product); !r) return r;
    // A second operation reuses the same pool after the raw consumer.
    return kg::MulMatIdQCompact(l, other);
  });
  ASSERT_TRUE(graph.has_value()) << (graph ? "" : graph.error().detail);
  for (const bool concentrated : {false, true, false}) {
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t slot = 0; slot < used; ++slot) {
        ids[static_cast<std::size_t>((t * used) + slot)] =
            static_cast<std::int32_t>(concentrated ? slot : (slot + t) % experts);
      }
    }
    ASSERT_EQ(cudaMemcpy(routes->data, ids.data(), ids.size() * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Launched(kg::MulMatIdQ2D2r(launch(), product), "raw Q2 changing routes");
    const auto want = Download(product);
    Launched(kg::MulMatIdQCompact(launch(), other), "compact changing routes");
    const auto want_other = Download(other);
    Launched(launch().Launch(*graph), "raw Q2 graph and scratch reuse");
    EXPECT_EQ(Download(product), want);
    EXPECT_EQ(Download(other), want_other);
  }
  auto choices = kg::DeviceChoicesOf(launch());
  const std::array<ggml_tensor*, 1> nodes = {product};
  auto ordinary = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(ordinary.has_value()) << (ordinary ? "" : ordinary.error().detail);
  EXPECT_EQ(ordinary->steps[0].implementation, kg::kMulMatIdQ);
  choices.d2r_experts = true;
  auto fallback = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(fallback.has_value()) << (fallback ? "" : fallback.error().detail);
  EXPECT_EQ(fallback->steps[0].implementation, kg::kMulMatIdQ);   // unmeasured small shape
  choices.q2_d2r_fits = [](const ggml_tensor*) { return true; };  // explicit generic control
  auto selected = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(selected.has_value()) << (selected ? "" : selected.error().detail);
  EXPECT_EQ(selected->steps[0].implementation, kg::kMulMatIdQ2D2r);
  choices.row_invariant = true;
  EXPECT_FALSE(kg::PlanGraph(nodes, false, choices).has_value());  // original verify bound wins
}

// A prompt's last, partial prefill chunk takes D2R as a full chunk does: a
// token's rows do not depend on the other tokens of its chunk (the first 77
// tokens of 129 alone equal theirs among all 129 byte for byte), and the
// automatic selection admits DeepSeek V4's down experts over any chunk of
// 64 to 4,096 tokens.
TEST_F(GgmlExtOpsTest, RawQ2D2rRowsDoNotDependOnTheChunkAndPartialChunksSelectIt) {
  constexpr std::int64_t k = 512;
  constexpr std::int64_t m = 130;
  constexpr std::int64_t experts = 16;
  constexpr std::int64_t used = 6;
  constexpr std::int64_t tokens = 129;
  constexpr std::int64_t part = 77;
  auto weights = Quantize(GGML_TYPE_Q2_K, k, m * experts, 81);
  auto* w = Place(ggml_new_tensor_3d(c(), GGML_TYPE_Q2_K, k, m, experts), weights.bytes);
  auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, k, used, tokens),
                  Normal(82, static_cast<std::size_t>(k * used * tokens)));
  std::vector<std::int32_t> ids(static_cast<std::size_t>(used * tokens));
  for (std::int64_t t = 0; t < tokens; ++t) {
    for (std::int64_t slot = 0; slot < used; ++slot) {
      ids[static_cast<std::size_t>((t * used) + slot)] =
          static_cast<std::int32_t>((slot + (3 * t)) % experts);
    }
  }
  auto* routes = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, used, tokens), ids);
  auto* whole = Place(ggml_mul_mat_id(c(), w, x, routes));
  auto* x_part = ggml_view_3d(c(), x, k, used, part, x->nb[1], x->nb[2], 0);
  auto* routes_part = ggml_view_2d(c(), routes, used, part, routes->nb[1], 0);
  auto* partial = Place(ggml_mul_mat_id(c(), w, x_part, routes_part));
  Launched(kg::MulMatIdQ2D2r(launch(), whole), "raw Q2 whole chunk");
  Launched(kg::MulMatIdQ2D2r(launch(), partial), "raw Q2 partial chunk");
  const auto all = Download(whole);
  const auto some = Download(partial);
  ASSERT_EQ(some.size(), static_cast<std::size_t>(m * used * part));
  EXPECT_EQ(std::memcmp(some.data(), all.data(), some.size() * sizeof(float)), 0);

  // The selection predicate over the model's shapes (bound, never read).
  const auto model = [&](std::int64_t chunk) {
    auto* mw = ggml_new_tensor_3d(c(), GGML_TYPE_Q2_K, 2048, 4096, 256);
    auto* mx = ggml_new_tensor_3d(c(), GGML_TYPE_F32, 2048, 6, chunk);
    auto* mi = ggml_new_tensor_2d(c(), GGML_TYPE_I32, 6, chunk);
    auto* node = ggml_mul_mat_id(c(), mw, mx, mi);
    std::uint64_t at = std::uint64_t{1} << 44U;
    for (ggml_tensor* t : {mw, mx, mi, node}) {
      TensorArena::Bind(t, at);
      at += (ggml_nbytes(t) + 4095) / 4096 * 4096;
    }
    return kg::MulMatIdQ2D2rFits(launch(), node);
  };
  if (ComputeCapability() == 1210) {
    for (const std::int64_t chunk :
         {kg::kDsv4StageMinRows, std::int64_t{2947}, kg::kDsv4StageMaxRows}) {
      EXPECT_TRUE(model(chunk)) << chunk;
    }
  }
  for (const std::int64_t chunk : {kg::kDsv4StageMinRows - 1, kg::kDsv4StageMaxRows + 1}) {
    EXPECT_FALSE(model(chunk)) << chunk;
  }
}

TEST_F(GgmlExtOpsTest, CompactExpertTilesPreservePartialRowsAndFallbackShapes) {
  constexpr std::int64_t kInner = 1024;
  constexpr std::int64_t kUsed = 2;
  struct Shape {
    std::int64_t rows, experts, tokens;
  };
  for (const auto type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, GGML_TYPE_Q8_0}) {
    // T256 leaves poor wave occupancy in the original stream-K grid on
    // GB10 and exercises its split-K/fixup comparison; T257 has tails.
    const std::array<Shape, 4> shapes = {
        {{129, 64, 256}, {129, 64, 257}, {64, 64, 257}, {129, 16, 257}}};
    for (const auto& [rows, experts, tokens] : shapes) {
      auto weights = Quantize(type, kInner, rows * experts, 57);
      auto* w = Place(ggml_new_tensor_3d(c(), type, kInner, rows, experts), weights.bytes);
      auto* input = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kInner, kUsed, tokens),
                          Normal(58, static_cast<std::size_t>(kInner * kUsed * tokens)));
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < tokens; ++t) {
        ids.push_back(static_cast<std::int32_t>(experts - 1));
        ids.push_back(static_cast<std::int32_t>(t % 7));
      }
      auto* route = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, tokens), ids);
      auto* product = Place(ggml_mul_mat_id(c(), w, input, route));
      const std::string what = std::string(ggml_type_name(type)) + " M" + std::to_string(rows) +
                               " E" + std::to_string(experts) + " T" + std::to_string(tokens);
      Launched(kg::MulMatQ(launch(), product), what);
      const auto want = Download(product);
      const auto scratch = kg::PlanMulMatIdQCompact(launch(), product);
      ASSERT_TRUE(scratch.has_value()) << what << ": " << scratch.error().detail;
      ASSERT_EQ(cudaMemset(product->data, 0xFF, ggml_nbytes(product)), cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      launch().ResetScratchPeak();
      Launched(kg::MulMatIdQCompact(launch(), product), what);
      EXPECT_LE(launch().scratch_peak().value(), *scratch);
      ExpectNmse(Download(product), std::vector<double>(want.begin(), want.end()), 1e-10, what);
    }
  }
}

TEST_F(GgmlExtOpsTest, CapturedCompactPairsRebuildTilesFromEachReplaysRouting) {
  constexpr std::int64_t kInner = 1024;
  constexpr std::int64_t kOut = 129;
  constexpr std::int64_t kExperts = 64;
  constexpr std::int64_t kTokens = 257;
  const auto a = Quantize(GGML_TYPE_Q2_K, kInner, kOut * kExperts, 59);
  const auto b = Quantize(GGML_TYPE_Q2_K, kInner, kOut * kExperts, 60);
  auto* wa = Place(ggml_new_tensor_3d(c(), GGML_TYPE_Q2_K, kInner, kOut, kExperts), a.bytes);
  auto* wb = Place(ggml_new_tensor_3d(c(), GGML_TYPE_Q2_K, kInner, kOut, kExperts), b.bytes);
  auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kInner, 1, kTokens),
                  Normal(61, static_cast<std::size_t>(kInner * kTokens)));
  std::vector<std::int32_t> ids(static_cast<std::size_t>(2 * kTokens));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<std::int32_t>(i % static_cast<std::size_t>(kExperts));
  }
  auto* routes = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 2, kTokens), ids);
  auto* first = Place(ggml_mul_mat_id(c(), wa, x, routes));
  auto* second = Place(ggml_mul_mat_id(c(), wb, x, routes));
  Launched(kg::MulMatIdQPair(launch(), first, second, true), "compact graph warmup");
  Finish();
  auto graph =
      launch().Capture([&](LaunchContext& l) { return kg::MulMatIdQPair(l, first, second, true); });
  ASSERT_TRUE(graph.has_value()) << graph.error().detail;
  EXPECT_GE(graph->nodes(), 6U);  // maps, Q8, and each product's list and inner product
  for (const bool concentrated : {true, false, true}) {
    for (std::int64_t t = 0; t < kTokens; ++t) {
      ids[static_cast<std::size_t>(2 * t)] = concentrated ? 0 : 63;
      ids[static_cast<std::size_t>((2 * t) + 1)] =
          concentrated ? 1 : static_cast<std::int32_t>(17 + (t % 8));
    }
    ASSERT_EQ(cudaMemcpy(routes->data, ids.data(), ids.size() * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Launched(kg::MulMatIdQPair(launch(), first, second, true), "fresh compact control");
    const auto want_first = Download(first);
    const auto want_second = Download(second);
    ASSERT_EQ(cudaMemset(first->data, 0xFF, ggml_nbytes(first)), cudaSuccess);
    ASSERT_EQ(cudaMemset(second->data, 0xFF, ggml_nbytes(second)), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    Launched(launch().Launch(*graph), "compact graph replay");
    EXPECT_EQ(Download(first), want_first);
    EXPECT_EQ(Download(second), want_second);
  }
}

// The resident expert layout (M3; docs/artifact-format.md#executable-views):
// the repacked expert groups laid out at a uniform stride S, a whole number
// of blocks, with other bytes between the slices, give mul_mat_id's stock
// kernels exactly what the GGUF's packed [k, n, experts] tensor gives them,
// on both kernel families.
TEST_F(GgmlExtOpsTest, ExpertsAtAUniformStrideComputeWhatThePackedLayoutComputes) {
  constexpr std::int64_t kExperts = 16;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kOut = 64;
  struct Case {
    ggml_type type;
    std::int64_t k;
    bool broadcast;
  };
  const std::array<Case, 5> cases = {{{GGML_TYPE_IQ2_XXS, 4096, true},
                                      {GGML_TYPE_Q2_K, 2048, false},
                                      {GGML_TYPE_IQ2_XS, 4096, true},
                                      {GGML_TYPE_IQ3_XXS, 2048, false},
                                      {GGML_TYPE_MXFP4, 2048, false}}};
  for (const Case& test : cases) {
    const Quantized weights = Quantize(test.type, test.k, kOut * kExperts, 41);
    const std::size_t slice = ggml_row_size(test.type, test.k) * kOut;
    // The smallest multiple of the block size and 16 bytes past the slice
    // and a 3,000-byte gap (the other projections of a group).
    const std::size_t unit = std::lcm(ggml_type_size(test.type), std::size_t{16});
    ASSERT_GT(unit, 0U);
    const std::size_t stride = (slice + 3000 + unit - 1) / unit * unit;
    std::vector<std::uint8_t> slab(stride * kExperts, 0xAB);
    for (std::int64_t e = 0; e < kExperts; ++e) {
      std::memcpy(slab.data() + (static_cast<std::size_t>(e) * stride),
                  weights.bytes.data() + (static_cast<std::size_t>(e) * slice), slice);
    }
    for (const std::int64_t tokens : {1, 3, 40}) {
      const std::string what = std::string(ggml_type_name(test.type)) + " x " +
                               std::to_string(tokens) + " at stride " + std::to_string(stride);
      ggml_tensor* packed =
          Place(ggml_new_tensor_3d(c(), test.type, test.k, kOut, kExperts), weights.bytes);
      ggml_tensor* strided = ggml_new_tensor_3d(c(), test.type, test.k, kOut, kExperts);
      strided->nb[2] = stride;
      strided->nb[3] = stride * kExperts;
      ASSERT_EQ(ggml_nbytes(strided), (stride * (kExperts - 1)) + slice);
      TensorArena::Bind(strided, Allocate(slab.size()));
      ASSERT_EQ(cudaMemcpy(strided->data, slab.data(), slab.size(), cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      const std::int64_t rows_in = test.broadcast ? 1 : kUsed;
      const std::vector<float> x = Normal(42, static_cast<std::size_t>(test.k * rows_in * tokens));
      ggml_tensor* input =
          Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, rows_in, tokens), x);
      std::mt19937 random(static_cast<unsigned>(43 + tokens));
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < tokens; ++t) {
        std::vector<std::int32_t> experts(kExperts);
        std::ranges::iota(experts, 0);
        std::shuffle(experts.begin(), experts.end(), random);
        ids.insert(ids.end(), experts.begin(), experts.begin() + kUsed);
      }
      ggml_tensor* id_tensor = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, tokens), ids);
      ggml_tensor* reference = Place(ggml_mul_mat_id(c(), packed, input, id_tensor));
      ggml_tensor* product = Place(ggml_mul_mat_id(c(), strided, input, id_tensor));
      auto path = kg::SelectMulMatQ(launch(), product);
      ASSERT_TRUE(path.has_value()) << what;
      auto reference_path = kg::SelectMulMatQ(launch(), reference);
      ASSERT_TRUE(reference_path.has_value()) << what;
      EXPECT_EQ(*path, *reference_path) << what;
      const auto run = [&](ggml_tensor* node) {
        return *path == QuantMulMatPath::kVector ? kg::MulMatVecQ(launch(), node)
                                                 : kg::MulMatQ(launch(), node);
      };
      Launched(run(reference), what + " (packed)");
      Launched(run(product), what + " (strided)");
      const std::vector<std::uint32_t> want = Download<std::uint32_t>(reference);
      const std::vector<std::uint32_t> got = Download<std::uint32_t>(product);
      ASSERT_EQ(got.size(), want.size());
      EXPECT_TRUE(got == want) << what << ": the strided layout's output differs";
    }
  }
  // A stride that is not a whole number of blocks is refused, never
  // truncated to one.
  ggml_tensor* torn = ggml_new_tensor_3d(c(), GGML_TYPE_IQ2_XS, 4096, kOut, kExperts);
  torn->nb[2] = (ggml_row_size(GGML_TYPE_IQ2_XS, 4096) * kOut) + 1;
  torn->nb[3] = torn->nb[2] * kExperts;
  TensorArena::Bind(torn, Allocate(torn->nb[3]));
  ggml_tensor* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 4096, 1, 1), Normal(44, 4096));
  ggml_tensor* one = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, 1),
                           std::vector<std::int32_t>{0, 1, 2, 3, 4, 5});
  ggml_tensor* bad = Place(ggml_mul_mat_id(c(), torn, x, one));
  EXPECT_EQ(FailedCode(kg::MulMatVecQ(launch(), bad)), KernelError::kRejected);
}

TEST_F(GgmlExtOpsTest, QuantizedChecksRefuseWhatTheLaunchersWouldNotTake) {
  // A row that is not a whole 512-element step (GGML pads buffers, jitLLM
  // does not), an uncompiled type, and F16 activations.
  const Quantized weights = Quantize(GGML_TYPE_Q8_0, 4096, 32, 31);
  ggml_tensor* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, 4096, 32), weights.bytes);
  ggml_tensor* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 4096, 2));
  ggml_tensor* good = Place(ggml_mul_mat(c(), w, x));
  EXPECT_TRUE(kg::CheckMulMatQ(good).has_value());
  ggml_tensor* short_w = ggml_view_2d(c(), w, 4064, 32, w->nb[1], 0);
  ggml_tensor* short_x = ggml_view_2d(c(), x, 4064, 2, x->nb[1], 0);
  ggml_tensor* short_rows = Place(ggml_mul_mat(c(), short_w, short_x));
  EXPECT_EQ(FailedCode(kg::CheckMulMatQ(short_rows)), KernelError::kRejected);
  ggml_tensor* iq1_m = Place(ggml_new_tensor_2d(c(), GGML_TYPE_IQ1_M, 4096, 32));
  EXPECT_EQ(FailedCode(kg::CheckMulMatQ(Place(ggml_mul_mat(c(), iq1_m, x)))),
            KernelError::kRejected);
  ggml_tensor* x16 = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 4096, 2));
  EXPECT_EQ(FailedCode(kg::CheckMulMatQ(Place(ggml_mul_mat(c(), w, x16)))), KernelError::kRejected);
  // Nothing was launched for a refused node: the context still runs.
  Launched(kg::MulMatVecQ(launch(), good), "after refusals");
}

// ---- The Hadamard rotation ----

TEST_F(GgmlExtOpsTest, TheHadamardHintRunsTheNormalizedWalshHadamardTransform) {
  // DeepSeek V4's indexer rotation: 128-element heads; and 512.
  for (const std::int64_t n : {128, 512}) {
    const std::int64_t rows = std::int64_t{64} * 3;
    const std::vector<float> x = Normal(41, static_cast<std::size_t>(n * rows));
    ggml_tensor* rotation = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, n, n));
    ggml_tensor* input = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, n, rows), x);
    ggml_tensor* product = ggml_mul_mat(c(), rotation, input);
    ggml_mul_mat_set_hint(product, GGML_HINT_SRC0_IS_HADAMARD);
    Place(product);
    Launched(kg::MulMatHadamard(launch(), product), "fwht");
    // Sylvester's Hadamard matrix, H[i][j] = (-1)^popcount(i & j) / sqrt(n).
    std::vector<double> want(static_cast<std::size_t>(n * rows));
    const double scale = 1.0 / std::sqrt(static_cast<double>(n));
    for (std::int64_t r = 0; r < rows; ++r) {
      for (std::int64_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < n; ++j) {
          const double sign = std::popcount(static_cast<std::uint64_t>(i & j)) % 2 == 0 ? 1 : -1;
          sum += sign * x[static_cast<std::size_t>((r * n) + j)];
        }
        want[static_cast<std::size_t>((r * n) + i)] = sum * scale;
      }
    }
    ExpectNmse(Download(product), want, kDefaultNmse, "fwht " + std::to_string(n));
  }
  // Without the hint the transform is not this implementation's.
  ggml_tensor* rotation = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, 128));
  ggml_tensor* input = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, 4));
  EXPECT_EQ(FailedCode(kg::MulMatHadamard(launch(), Place(ggml_mul_mat(c(), rotation, input)))),
            KernelError::kRejected);
}

// ---- Elementwise ----

TEST_F(GgmlExtOpsTest, ElementwiseFunctionsMatchTheReference) {
  constexpr std::int64_t kWidth = 4096;
  constexpr std::int64_t kRows = 3;
  constexpr std::size_t kCount = kWidth * kRows;
  const std::vector<float> x = Normal(51, kCount, 3.0f);
  const std::vector<float> positive = Uniform(52, kCount, 0.001f, 50.0f);
  struct Function {
    const char* name;
    ggml_tensor* (*build)(ggml_context*, ggml_tensor*);
    double (*reference)(double);
    bool positive;
  };
  const std::array<Function, 10> functions = {{
      {"abs", ggml_abs, [](double v) { return std::abs(v); }, false},
      {"sgn", ggml_sgn,
       [](double v) {
         if (v > 0) {
           return 1.0;
         }
         return v < 0 ? -1.0 : 0.0;
       },
       false},
      {"neg", ggml_neg, [](double v) { return -v; }, false},
      {"silu", ggml_silu, [](double v) { return v / (1.0 + std::exp(-v)); }, false},
      {"tanh", ggml_tanh, [](double v) { return std::tanh(v); }, false},
      {"relu", ggml_relu, [](double v) { return v > 0 ? v : 0.0; }, false},
      {"sigmoid", ggml_sigmoid, [](double v) { return 1.0 / (1.0 + std::exp(-v)); }, false},
      {"exp", ggml_exp, [](double v) { return std::exp(v); }, false},
      {"softplus", ggml_softplus, [](double v) { return v > 20.0 ? v : std::log1p(std::exp(v)); },
       false},
      {"sqrt", ggml_sqrt, [](double v) { return std::sqrt(v); }, true},
  }};
  for (const Function& function : functions) {
    const std::vector<float>& in = function.positive ? positive : x;
    ggml_tensor* source = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, kRows), in);
    ggml_tensor* node = Place(function.build(c(), source));
    Launched(kg::Unary(launch(), node), function.name);
    std::vector<double> want(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
      want[i] = function.reference(in[i]);
    }
    ExpectNmse(Download(node), want, kDefaultNmse, function.name);
  }
  // In place, as the graphs apply sigmoid to a gate.
  ggml_tensor* gate = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, kRows), x);
  ggml_tensor* in_place = ggml_sigmoid_inplace(c(), gate);
  Launched(kg::Unary(launch(), in_place), "sigmoid in place");
  std::vector<double> want(kCount);
  for (std::size_t i = 0; i < kCount; ++i) {
    want[i] = 1.0 / (1.0 + std::exp(-static_cast<double>(x[i])));
  }
  ExpectNmse(Download(gate), want, kDefaultNmse, "sigmoid in place");
  // A function not launched here is refused.
  ggml_tensor* other = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, kRows), x);
  EXPECT_EQ(FailedCode(kg::Unary(launch(), Place(ggml_gelu_erf(c(), other)))),
            KernelError::kRejected);
}

TEST_F(GgmlExtOpsTest, ScaleClampFillRepeatSubAndDivMatchTheReference) {
  // DeepSeek V4's hyper-connection pre weights: scale_bias over [4, tokens].
  const std::vector<float> x = Normal(61, std::size_t{24} * 5);
  ggml_tensor* mixes = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 24, 5), x);
  ggml_tensor* scaled = Place(ggml_scale_bias(c(), mixes, 0.75f, 1e-6f));
  Launched(kg::Scale(launch(), scaled), "scale_bias");
  std::vector<double> want(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    want[i] = (0.75 * x[i]) + 1e-6;
  }
  ExpectNmse(Download(scaled), want, kDefaultNmse, "scale_bias");

  // The MoE's clamp of the expert logits, and Qwen3.8's.
  const std::vector<float> wide = Normal(62, std::size_t{2048} * 3, 20.0f);
  ggml_tensor* logits = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 2048, 3), wide);
  ggml_tensor* clamped = Place(ggml_clamp(c(), logits, -10.0f, 10.0f));
  Launched(kg::Clamp(launch(), clamped), "clamp");
  const auto got_clamp = Download(clamped);
  for (std::size_t i = 0; i < wide.size(); ++i) {
    ASSERT_EQ(got_clamp[i], std::clamp(wide[i], -10.0f, 10.0f)) << i;
  }

  // DeepSeek V4's top-k mask starts as -inf F16, its zeros as F16.
  ggml_tensor* mask_shape = ggml_new_tensor_2d(c(), GGML_TYPE_F16, 1000, 7);
  ggml_tensor* filled = Place(ggml_fill(c(), mask_shape, -std::numeric_limits<float>::infinity()));
  Launched(kg::Fill(launch(), filled), "fill F16");
  for (const ggml_fp16_t v : Download<ggml_fp16_t>(filled)) {
    ASSERT_EQ(std::bit_cast<std::uint16_t>(v), 0xFC00U);
  }
  ggml_tensor* ones = Place(ggml_fill(c(), ggml_new_tensor_1d(c(), GGML_TYPE_F32, 4096), 1.5f));
  Launched(kg::Fill(launch(), ones), "fill F32");
  for (const float v : Download(ones)) {
    ASSERT_EQ(v, 1.5f);
  }

  // repeat_4d: one row tiled over a chunk, as the delta net's decay mask.
  const std::vector<float> row = Normal(63, 64);
  ggml_tensor* one = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 1), row);
  ggml_tensor* tiled = Place(ggml_repeat_4d(c(), one, 64, 64, 3, 1));
  Launched(kg::Repeat(launch(), tiled), "repeat");
  const auto got_tiled = Download(tiled);
  for (std::size_t i = 0; i < got_tiled.size(); ++i) {
    ASSERT_EQ(got_tiled[i], row[i % 64]) << i;
  }

  // sub and div with broadcasting: the hyper-connections' normalization of
  // [4, 4, tokens] by a row sum [1, 4, tokens].
  const std::vector<float> a = Uniform(64, std::size_t{16} * 5, 0.1f, 2.0f);
  const std::vector<float> b = Uniform(65, std::size_t{4} * 5, 0.5f, 3.0f);
  ggml_tensor* ta = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 4, 4, 5), a);
  ggml_tensor* tb = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 4, 5), b);
  ggml_tensor* quotient = Place(ggml_div(c(), ta, tb));
  ggml_tensor* difference = Place(ggml_sub(c(), ta, tb));
  Launched(kg::Div(launch(), quotient), "div");
  Launched(kg::Sub(launch(), difference), "sub");
  std::vector<double> want_div(a.size());
  std::vector<double> want_sub(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double divisor = b[i / 4];
    want_div[i] = a[i] / divisor;
    want_sub[i] = a[i] - divisor;
  }
  ExpectNmse(Download(quotient), want_div, kDefaultNmse, "div");
  ExpectNmse(Download(difference), want_sub, kDefaultNmse, "sub");
}

TEST_F(GgmlExtOpsTest, ConcatAndSumRowsMatchTheReference) {
  // DeepSeek V4 concatenates caches and masks along each dimension; F32 and
  // F16, contiguous and a strided view.
  for (const int dim : {0, 1, 2}) {
    for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
      const std::array<std::int64_t, 3> shape_a = {64, 5, 3};
      std::array<std::int64_t, 3> shape_b = shape_a;
      shape_b[static_cast<std::size_t>(dim)] = 7;
      const auto count = [](const std::array<std::int64_t, 3>& s) {
        return static_cast<std::size_t>(s[0] * s[1] * s[2]);
      };
      const std::vector<float> va = Normal(71, count(shape_a));
      const std::vector<float> vb = Normal(72, count(shape_b));
      ggml_tensor* ta = ggml_new_tensor_3d(c(), type, shape_a[0], shape_a[1], shape_a[2]);
      ggml_tensor* tb = ggml_new_tensor_3d(c(), type, shape_b[0], shape_b[1], shape_b[2]);
      if (type == GGML_TYPE_F16) {
        Place(ta, Halves(va));
        Place(tb, Halves(vb));
      } else {
        Place(ta, va);
        Place(tb, vb);
      }
      ggml_tensor* joined = Place(ggml_concat(c(), ta, tb, dim));
      const std::string what =
          std::string("concat dim ") + std::to_string(dim) + " " + ggml_type_name(type);
      Launched(kg::Concat(launch(), joined), what);
      const std::vector<float> got =
          type == GGML_TYPE_F16 ? Widen(Download<ggml_fp16_t>(joined)) : Download(joined);
      const auto round = [type](float v) {
        return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) : v;
      };
      for (std::int64_t i2 = 0; i2 < joined->ne[2]; ++i2) {
        for (std::int64_t i1 = 0; i1 < joined->ne[1]; ++i1) {
          for (std::int64_t i0 = 0; i0 < joined->ne[0]; ++i0) {
            const std::array<std::int64_t, 3> index = {i0, i1, i2};
            const bool first =
                index[static_cast<std::size_t>(dim)] < shape_a[static_cast<std::size_t>(dim)];
            std::array<std::int64_t, 3> at = index;
            if (!first) {
              at[static_cast<std::size_t>(dim)] -= shape_a[static_cast<std::size_t>(dim)];
            }
            const auto& shape = first ? shape_a : shape_b;
            const auto from =
                static_cast<std::size_t>(at[0] + (shape[0] * (at[1] + (shape[1] * at[2]))));
            const float want = round((first ? va : vb)[from]);
            const auto to =
                static_cast<std::size_t>(i0 + (joined->ne[0] * (i1 + (joined->ne[1] * i2))));
            ASSERT_EQ(got[to], want) << what;
          }
        }
      }
    }
  }
  // A strided source: the first 48 elements of each row (the slow kernel).
  const std::vector<float> va = Normal(73, std::size_t{64} * 5);
  const std::vector<float> vb = Normal(74, std::size_t{48} * 5);
  ggml_tensor* ta = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 5), va);
  ggml_tensor* view = ggml_view_2d(c(), ta, 48, 5, ta->nb[1], 0);
  ggml_tensor* tb = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 48, 5), vb);
  ggml_tensor* joined = Place(ggml_concat(c(), view, tb, 1));
  Launched(kg::Concat(launch(), joined), "concat of a view");
  const auto got = Download(joined);
  for (std::int64_t r = 0; r < 10; ++r) {
    for (std::int64_t i = 0; i < 48; ++i) {
      const float want = r < 5 ? va[static_cast<std::size_t>((r * 64) + i)]
                               : vb[static_cast<std::size_t>(((r - 5) * 48) + i)];
      ASSERT_EQ(got[static_cast<std::size_t>((r * 48) + i)], want);
    }
  }

  // sum_rows: DeepSeek V4's expert weight normalization over 6 and a
  // hidden-width sum.
  for (const std::int64_t width : {6, 4096}) {
    const std::vector<float> x = Normal(75, static_cast<std::size_t>(width * 9));
    ggml_tensor* in = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, 9), x);
    ggml_tensor* sums = Place(ggml_sum_rows(c(), in));
    Launched(kg::SumRows(launch(), sums), "sum_rows");
    std::vector<double> want(9);
    for (std::size_t r = 0; r < 9; ++r) {
      for (std::int64_t i = 0; i < width; ++i) {
        want[r] += x[(r * static_cast<std::size_t>(width)) + static_cast<std::size_t>(i)];
      }
    }
    ExpectNmse(Download(sums), want, kDefaultNmse, "sum_rows " + std::to_string(width));
  }
}

// ---- Routing ----

TEST_F(GgmlExtOpsTest, ArgsortAndTopKFindTheLargestValues) {
  // Argsort of DeepSeek V4's 256 and Qwen3.8's 512 expert scores, as
  // ggml_argsort_top_k views it, and top-k 512 of an indexer row.
  for (const std::int64_t experts : {256, 512}) {
    std::vector<float> scores(static_cast<std::size_t>(experts * 5));
    std::mt19937 random(81);  // NOLINT(bugprone-random-generator-seed): reproducible
    for (std::size_t r = 0; r < 5; ++r) {
      std::vector<float> row(static_cast<std::size_t>(experts));
      // Distinct: no ties. std::ranges::iota needs an incrementable type.
      std::iota(row.begin(), row.end(), 0.0f);  // NOLINT(modernize-use-ranges)
      std::shuffle(row.begin(), row.end(), random);
      std::ranges::copy(row, scores.begin() + static_cast<std::ptrdiff_t>(r * row.size()));
    }
    ggml_tensor* in = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, experts, 5), scores);
    ggml_tensor* order = Place(ggml_argsort(c(), in, GGML_SORT_ORDER_DESC));
    Launched(kg::Argsort(launch(), order), "argsort");
    const auto got = Download<std::int32_t>(order);
    const auto width = static_cast<std::size_t>(experts);
    for (std::size_t r = 0; r < 5; ++r) {
      for (std::size_t i = 0; i < width; ++i) {
        const auto index =
            static_cast<std::size_t>(got[(r * static_cast<std::size_t>(experts)) + i]);
        ASSERT_EQ(scores[(r * static_cast<std::size_t>(experts)) + index],
                  static_cast<float>(experts - 1 - static_cast<std::int64_t>(i)));
      }
    }
  }
  for (const std::int64_t cells : {600, 20000}) {
    const std::int64_t rows = 3;
    const std::int64_t k = 512;
    std::vector<float> scores(static_cast<std::size_t>(cells * rows));
    std::mt19937 random(82);  // NOLINT(bugprone-random-generator-seed): reproducible
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      std::vector<float> row(static_cast<std::size_t>(cells));
      std::iota(row.begin(), row.end(), 0.0f);  // NOLINT(modernize-use-ranges)
      std::shuffle(row.begin(), row.end(), random);
      std::ranges::copy(row, scores.begin() + static_cast<std::ptrdiff_t>(r * row.size()));
    }
    ggml_tensor* in = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, cells, rows), scores);
    ggml_tensor* top = Place(ggml_top_k(c(), in, static_cast<int>(k)));
    const auto scratch = kg::PlanTopK(launch(), top);
    ASSERT_TRUE(scratch.has_value());
    launch().ResetScratchPeak();
    Launched(kg::TopK(launch(), top), "top_k");
    EXPECT_LE(launch().scratch_peak().value(), (*scratch + 255) / 256 * 256);
    const auto got = Download<std::int32_t>(top);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
      // The radix select leaves the k indices in no particular order:
      // compare as sets.
      std::set<float> values;
      for (std::size_t i = 0; i < static_cast<std::size_t>(k); ++i) {
        values.insert(scores[(r * static_cast<std::size_t>(cells)) +
                             static_cast<std::size_t>(got[(r * static_cast<std::size_t>(k)) + i])]);
      }
      ASSERT_EQ(values.size(), static_cast<std::size_t>(k));
      EXPECT_EQ(*values.begin(), static_cast<float>(cells - k));
      EXPECT_EQ(*values.rbegin(), static_cast<float>(cells - 1));
    }
  }
  // Rows beyond the bitonic kernel take CUB's sort upstream, which jitLLM's
  // build does not have: refused.
  ggml_tensor* long_rows = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 2048, 2));
  EXPECT_EQ(
      FailedCode(kg::Argsort(launch(), Place(ggml_argsort(c(), long_rows, GGML_SORT_ORDER_DESC)))),
      KernelError::kRejected);
}

TEST_F(GgmlExtOpsTest, ClampedSwiGluMatchesTheReference) {
  // DeepSeek V4's experts and shared expert: swiglu_clamp at 10, split.
  constexpr std::int64_t kFfn = 2048;
  constexpr std::int64_t kTokens = 3;
  const std::vector<float> gate = Normal(91, kFfn * kTokens, 8.0f);
  const std::vector<float> up = Normal(92, kFfn * kTokens, 8.0f);
  ggml_tensor* tg = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kFfn, kTokens), gate);
  ggml_tensor* tu = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kFfn, kTokens), up);
  ggml_tensor* glu = Place(ggml_swiglu_clamp(c(), tg, tu, 10.0f));
  Launched(kg::SwiGluClamp(launch(), glu), "swiglu_clamp");
  std::vector<double> want(gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const double g = std::min<double>(gate[i], 10.0);
    const double u = std::clamp<double>(up[i], -10.0, 10.0);
    want[i] = g / (1.0 + std::exp(-g)) * u;
  }
  ExpectNmse(Download(glu), want, kDefaultNmse, "swiglu_clamp");
}

// ---- RoPE ----

// ggml's rotation of one pair (rope.cu:15-44), in FP64.
std::pair<double, double> RopeAngle(double theta_extrap, int pair, double freq_scale,
                                    const std::array<float, 2>& corr, double ext_factor,
                                    double attn_factor, bool forward) {
  const double theta_interp = freq_scale * theta_extrap;
  double theta = theta_interp;
  double mscale = attn_factor;
  if (ext_factor != 0.0) {
    const double y = (pair - static_cast<double>(corr[0])) /
                     std::max(0.001, static_cast<double>(corr[1]) - corr[0]);
    const double ramp = (1.0 - std::min(1.0, std::max(0.0, y))) * ext_factor;
    theta = (theta_interp * (1 - ramp)) + (theta_extrap * ramp);
    mscale *= 1.0 + (0.1 * std::log(1.0 / freq_scale));
  }
  return {std::cos(theta) * mscale, std::sin(theta) * mscale * (forward ? 1.0 : -1.0)};
}

TEST_F(GgmlExtOpsTest, RopeWithOffsetsYarnAndSectionsMatchesTheReference) {
  // DeepSeek V4: normal rotation of the last 64 of 512 channels (offset
  // 448), YaRN at 16x, forward on Q and backward on the output; the
  // indexer's 128-channel heads rotate at offset 64. Qwen3.8: interleaved
  // multi-section rotation of the first 64 of 256.
  struct Case {
    const char* name;
    std::int64_t head;
    std::int64_t heads;
    int n_dims;
    int offset;
    int mode;
    bool forward;
    float freq_base;
    float freq_scale;
    float ext_factor;
  };
  const std::array<Case, 4> cases = {{
      {"deepseek q", 512, 8, 64, 448, 0, true, 10000.0f, 1.0f / 16.0f, 1.0f},
      {"deepseek output (back)", 512, 8, 64, 448, 0, false, 10000.0f, 1.0f / 16.0f, 1.0f},
      {"deepseek indexer", 128, 64, 64, 64, 0, true, 160000.0f, 1.0f / 16.0f, 1.0f},
      {"qwen3.8 imrope", 256, 24, 64, 0, GGML_ROPE_TYPE_IMROPE, true, 1e7f, 1.0f, 0.0f},
  }};
  constexpr std::int64_t kTokens = 5;
  constexpr int kContext = 65536;
  constexpr float kAttn = 1.0f;
  constexpr float kBetaFast = 32.0f;
  constexpr float kBetaSlow = 1.0f;
  std::array<int, GGML_MROPE_SECTIONS> sections = {11, 11, 10, 0};
  const std::array<std::int32_t, kTokens> positions = {0, 1, 17, 300, 999};
  for (const Case& test : cases) {
    const bool multi = test.mode == GGML_ROPE_TYPE_IMROPE;
    std::vector<std::int32_t> pos;
    for (int s = 0; s < (multi ? 4 : 1); ++s) {
      for (const std::int32_t p : positions) {
        pos.push_back(s == 0 ? p : p + (7 * s));  // distinct per section
      }
    }
    const std::vector<float> x =
        Normal(101, static_cast<std::size_t>(test.head * test.heads * kTokens));
    ggml_tensor* in =
        Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.head, test.heads, kTokens), x);
    ggml_tensor* p =
        Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, static_cast<std::int64_t>(pos.size())), pos);
    ggml_tensor* node = nullptr;
    if (multi) {
      node = ggml_rope_multi(c(), in, p, nullptr, test.n_dims, sections.data(), test.mode, kContext,
                             test.freq_base, test.freq_scale, test.ext_factor, kAttn, kBetaFast,
                             kBetaSlow);
    } else if (test.forward) {
      node = ggml_rope_ext(c(), in, p, nullptr, test.n_dims, test.mode, kContext, test.freq_base,
                           test.freq_scale, test.ext_factor, kAttn, kBetaFast, kBetaSlow);
    } else {
      node =
          ggml_rope_ext_back(c(), in, p, nullptr, test.n_dims, test.mode, kContext, test.freq_base,
                             test.freq_scale, test.ext_factor, kAttn, kBetaFast, kBetaSlow);
    }
    node = ggml_rope_set_offset(node, test.offset);
    Place(node);
    Launched(kg::RopeExt(launch(), node), test.name);

    std::array<float, 2> corr{};
    ggml_rope_yarn_corr_dims(test.n_dims, kContext, test.freq_base, kBetaFast, kBetaSlow,
                             corr.data());
    const double theta_scale = std::pow(static_cast<double>(test.freq_base), -2.0 / test.n_dims);
    std::vector<double> want(x.begin(), x.end());
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    for (std::int64_t t = 0; t < kTokens; ++t) {
      for (std::int64_t h = 0; h < test.heads; ++h) {
        const auto row = static_cast<std::size_t>(((t * test.heads) + h) * test.head);
        for (int pair = 0; pair < test.n_dims / 2; ++pair) {
          int component = 0;
          if (multi) {
            const int sector = pair % sect_dims;
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
              component = 1;
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
              component = 2;
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
              component = 0;
            } else {
              component = 3;
            }
          }
          const double position = pos[static_cast<std::size_t>((component * kTokens) + t)];
          const auto [cos_t, sin_t] =
              RopeAngle(position * std::pow(theta_scale, pair), pair, test.freq_scale, corr,
                        test.ext_factor, kAttn, test.forward);
          // Normal: adjacent pairs; NEOX-ordered for the multi-section modes.
          const std::size_t i0 =
              row + static_cast<std::size_t>(test.offset + (multi ? pair : 2 * pair));
          const std::size_t i1 = multi ? i0 + static_cast<std::size_t>(test.n_dims / 2) : i0 + 1;
          const double x0 = x[i0];
          const double x1 = x[i1];
          want[i0] = (x0 * cos_t) - (x1 * sin_t);
          want[i1] = (x0 * sin_t) + (x1 * cos_t);
        }
      }
    }
    ExpectNmse(Download(node), want, kDefaultNmse, test.name);
  }
}

// ---- Gathers and scatters ----

TEST_F(GgmlExtOpsTest, LegacyGetRowsDequantizesAllApprovedFormats) {
  constexpr std::int64_t k = 2816, rows = 8;
  const std::vector<std::int32_t> indices = {7, 0, 3, 3};
  for (const ggml_type type :
       {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
    const auto quantized = Quantize(type, k, rows, 2816);
    auto* w = Place(ggml_new_tensor_2d(c(), type, k, rows), quantized.bytes);
    auto* ids = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4), indices);
    auto* gathered = Place(ggml_get_rows(c(), w, ids));
    Launched(kg::GetRowsExt(launch(), gathered), ggml_type_name(type));
    std::vector<double> want;
    for (const auto index : indices) {
      want.insert(want.end(), quantized.values.begin() + index * k,
                  quantized.values.begin() + (index + 1) * k);
    }
    ExpectNmse(Download(gathered), want, 1e-10, ggml_type_name(type));
  }
}

TEST_F(GgmlExtOpsTest, GetRowsDequantizesAndSetRowsWritesExactly) {
  // DeepSeek V4's Q5_K token embedding, and its hash-routing table of I32
  // expert ids (ffn_gate_tid2eid [6, vocabulary]).
  const Quantized table = Quantize(GGML_TYPE_Q5_K, 4096, 40, 111);
  ggml_tensor* embd = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q5_K, 4096, 40), table.bytes);
  const std::vector<std::int32_t> tokens = {3, 39, 0, 17, 17};
  ggml_tensor* ids = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 5), tokens);
  ggml_tensor* rows = Place(ggml_get_rows(c(), embd, ids));
  Launched(kg::GetRowsExt(launch(), rows), "get_rows Q5_K");
  const auto got = Download(rows);
  for (std::size_t r = 0; r < tokens.size(); ++r) {
    for (std::size_t i = 0; i < 4096; ++i) {
      ASSERT_EQ(got[(r * 4096) + i],
                static_cast<float>(table.values[(static_cast<std::size_t>(tokens[r]) * 4096) + i]))
          << r << ", " << i;
    }
  }
  std::vector<std::int32_t> hash(std::size_t{6} * 1000);
  std::ranges::iota(hash, 0);
  ggml_tensor* tid2eid = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 6, 1000), hash);
  const std::vector<std::int32_t> vocab = {999, 0, 512};
  ggml_tensor* vocab_ids = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 3), vocab);
  ggml_tensor* routed = Place(ggml_get_rows(c(), tid2eid, vocab_ids));
  Launched(kg::GetRowsExt(launch(), routed), "get_rows I32");
  const auto got_routed = Download<std::int32_t>(routed);
  for (std::size_t r = 0; r < vocab.size(); ++r) {
    for (std::size_t i = 0; i < 6; ++i) {
      ASSERT_EQ(got_routed[(r * 6) + i], (vocab[r] * 6) + static_cast<std::int32_t>(i));
    }
  }

  // DeepSeek V4's top-k mask: F16 zeros written at I32 cells of a -inf F16
  // mask; and F32 rows into F32 at I32 indices.
  const std::vector<float> minus_inf(std::size_t{1024} * 3,
                                     -std::numeric_limits<float>::infinity());
  ggml_tensor* mask =
      Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 1, std::int64_t{1024} * 3), Halves(minus_inf));
  const std::vector<std::int32_t> cells = {5, 700, 1023, 2048, 3071};
  ggml_tensor* zeros =
      Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 1, 5), Halves(std::vector<float>(5, 0.0f)));
  ggml_tensor* cell_ids = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 5), cells);
  ggml_tensor* written = ggml_set_rows(c(), mask, zeros, cell_ids);
  Launched(kg::SetRowsExt(launch(), written), "set_rows F16 at I32");
  const auto got_mask = Widen(Download<ggml_fp16_t>(mask));
  for (std::size_t i = 0; i < got_mask.size(); ++i) {
    const bool set = std::ranges::find(cells, static_cast<std::int32_t>(i)) != cells.end();
    ASSERT_EQ(got_mask[i], set ? 0.0f : -std::numeric_limits<float>::infinity()) << i;
  }
  const std::vector<float> values = Normal(112, std::size_t{64} * 3);
  ggml_tensor* dst =
      Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 10), std::vector<float>(640, 0.0f));
  ggml_tensor* src = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 3), values);
  const std::vector<std::int32_t> slots = {9, 2, 4};
  ggml_tensor* slot_ids = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 3), slots);
  Launched(kg::SetRowsExt(launch(), ggml_set_rows(c(), dst, src, slot_ids)), "set_rows F32");
  const auto got_rows = Download(dst);
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t i = 0; i < 64; ++i) {
      ASSERT_EQ(got_rows[(static_cast<std::size_t>(slots[r]) * 64) + i], values[(r * 64) + i]);
    }
  }
}

// ---- Qwen3.8's linear attention ----

TEST_F(GgmlExtOpsTest, CausalConvolutionMatchesTheReference) {
  // Qwen3.8's Gated DeltaNet input convolution: 2 x 16 x 128 + 48 x 128
  // channels, kernel 4, for one token and a 64-token prefill (past 32
  // tokens, whole 32-token blocks: RE-032).
  constexpr std::int64_t kChannels = 10240;
  constexpr std::int64_t kConv = 4;
  for (const std::int64_t tokens : {1, 64}) {
    const std::int64_t window = kConv - 1 + tokens;
    const std::vector<float> x = Normal(121, static_cast<std::size_t>(window * kChannels));
    const std::vector<float> w = Normal(122, static_cast<std::size_t>(kConv * kChannels), 0.5f);
    ggml_tensor* tx = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, window, kChannels, 1), x);
    ggml_tensor* tw = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kConv, kChannels), w);
    ggml_tensor* conv = Place(ggml_ssm_conv(c(), tx, tw));
    Launched(kg::SsmConv(launch(), conv), "ssm_conv");
    std::vector<double> want(static_cast<std::size_t>(kChannels * tokens));
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t ch = 0; ch < kChannels; ++ch) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < kConv; ++j) {
          sum += static_cast<double>(x[static_cast<std::size_t>((ch * window) + t + j)]) *
                 w[static_cast<std::size_t>((ch * kConv) + j)];
        }
        want[static_cast<std::size_t>((t * kChannels) + ch)] = sum;
      }
    }
    ExpectNmse(Download(conv), want, kDefaultNmse, "ssm_conv x " + std::to_string(tokens));
  }
}

TEST_F(GgmlExtOpsTest, GatedDeltaRuleMatchesTheReference) {
  // Qwen3.8: 16 query/key heads and 48 value heads of 128, a scalar gate.
  constexpr std::int64_t kS = 128;
  constexpr std::int64_t kHk = 16;
  constexpr std::int64_t kHv = 48;
  for (const std::int64_t tokens : {1, 6}) {
    const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
    // Unit-norm q and k rows, as the graph's L2 normalization leaves them.
    std::vector<float> q = Normal(131, n(kS * kHk * tokens));
    std::vector<float> k = Normal(132, n(kS * kHk * tokens));
    for (std::vector<float>* rows : {&q, &k}) {
      for (std::size_t r = 0; r < rows->size() / kS; ++r) {
        double norm = 0.0;
        for (std::size_t i = 0; i < kS; ++i) {
          norm += (*rows)[(r * kS) + i] * (*rows)[(r * kS) + i];
        }
        for (std::size_t i = 0; i < kS; ++i) {
          (*rows)[(r * kS) + i] = static_cast<float>((*rows)[(r * kS) + i] / std::sqrt(norm));
        }
      }
    }
    const std::vector<float> v = Normal(133, n(kS * kHv * tokens));
    const std::vector<float> g = Uniform(134, n(kHv * tokens), -2.0f, -0.01f);  // log decay
    const std::vector<float> beta = Uniform(135, n(kHv * tokens), 0.0f, 1.0f);
    const std::vector<float> state = Normal(136, n(kS * kS * kHv), 0.1f);
    ggml_tensor* tq = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kS, kHk, tokens, 1), q);
    ggml_tensor* tk = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kS, kHk, tokens, 1), k);
    ggml_tensor* tv = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kS, kHv, tokens, 1), v);
    ggml_tensor* tg = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, 1, kHv, tokens, 1), g);
    ggml_tensor* tb = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, 1, kHv, tokens, 1), beta);
    ggml_tensor* ts = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kS, kS, kHv, 1), state);
    ggml_tensor* out = Place(ggml_gated_delta_net(c(), tq, tk, tv, tg, tb, ts, 1));
    Launched(kg::GatedDeltaNet(launch(), out), "gated_delta_net");

    // The kernel's recurrence (gated_delta_net.cu:52-110), with its layouts:
    // the state is S[i][col] at state[(h * S + col) * S + i]; value head h
    // reads query/key head h mod Hk; the output is the attention
    // [S, Hv, tokens] then the final state.
    std::vector<double> want(n((kS * kHv * tokens) + (kS * kS * kHv)));
    const double scale = 1.0 / std::sqrt(static_cast<double>(kS));
    for (std::int64_t h = 0; h < kHv; ++h) {
      std::vector<double> s(n(kS * kS));  // s[col * S + i] = S[i][col]
      for (std::size_t i = 0; i < s.size(); ++i) {
        s[i] = state[(n(h) * n(kS * kS)) + i];
      }
      const std::int64_t hk = h % kHk;
      for (std::int64_t t = 0; t < tokens; ++t) {
        const double decay = std::exp(static_cast<double>(g[n((t * kHv) + h)]));
        const double b = beta[n((t * kHv) + h)];
        const std::size_t qk_row = n(((t * kHk) + hk) * kS);
        const std::size_t v_row = n(((t * kHv) + h) * kS);
        for (std::int64_t col = 0; col < kS; ++col) {
          double kv = 0.0;
          for (std::int64_t i = 0; i < kS; ++i) {
            kv += s[n((col * kS) + i)] * k[qk_row + n(i)];
          }
          const double delta = (v[v_row + n(col)] - (decay * kv)) * b;
          double attn = 0.0;
          for (std::int64_t i = 0; i < kS; ++i) {
            double& cell = s[n((col * kS) + i)];
            cell = (decay * cell) + (k[qk_row + n(i)] * delta);
            attn += cell * q[qk_row + n(i)];
          }
          want[n(((t * kHv) + h) * kS) + n(col)] = attn * scale;
        }
      }
      for (std::size_t i = 0; i < s.size(); ++i) {
        want[n(kS * kHv * tokens) + (n(h) * n(kS * kS)) + i] = s[i];
      }
    }
    ExpectNmse(Download(out), want, kDefaultNmse, "gated_delta_net x " + std::to_string(tokens));
  }
}

// ---- DeepSeek V4's indexer and hyper-connections ----

TEST_F(GgmlExtOpsTest, LightningIndexerMatchesTheReference) {
  constexpr std::int64_t kEmbd = 128;
  constexpr std::int64_t kHeads = 64;
  for (const std::int64_t batch : {1, 5}) {
    const std::int64_t cells = 1000;  // not a whole number of 32-cell blocks
    const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
    const std::vector<float> q = Normal(141, n(kEmbd * kHeads * batch));
    const std::vector<float> k = Normal(142, n(kEmbd * cells));
    const std::vector<float> w =
        Normal(143, n(kHeads * batch), 1.0f / std::sqrt(static_cast<float>(kEmbd * kHeads)));
    std::vector<float> mask(n(cells * batch), 0.0f);
    for (std::int64_t b = 0; b < batch; ++b) {
      for (std::int64_t cell = cells - 100 + (10 * b); cell < cells; ++cell) {
        mask[n((b * cells) + cell)] = -std::numeric_limits<float>::infinity();
      }
    }
    const std::vector<ggml_fp16_t> k16 = Halves(k);
    const std::vector<ggml_fp16_t> mask16 = Halves(mask);
    ggml_tensor* tq = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kEmbd, kHeads, batch, 1), q);
    ggml_tensor* tk = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, kEmbd, 1, cells, 1), k16);
    ggml_tensor* tw = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kHeads, batch, 1, 1), w);
    ggml_tensor* tm = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, cells, batch, 1, 1), mask16);
    ggml_tensor* scores = Place(ggml_lightning_indexer(c(), tq, tk, tw, tm));
    Launched(kg::LightningIndexer(launch(), scores), "lightning_indexer");
    std::vector<double> want(n(cells * batch));
    std::vector<float> finite_got;
    std::vector<double> finite_want;
    const auto got = Download(scores);
    for (std::int64_t b = 0; b < batch; ++b) {
      for (std::int64_t cell = 0; cell < cells; ++cell) {
        double score = 0.0;
        for (std::int64_t h = 0; h < kHeads; ++h) {
          double dot = 0.0;
          for (std::int64_t i = 0; i < kEmbd; ++i) {
            dot += static_cast<double>(q[n(((b * kHeads) + h) * kEmbd) + n(i)]) *
                   ggml_fp16_to_fp32(k16[n((cell * kEmbd) + i)]);
          }
          score += std::max(dot, 0.0) * w[n((b * kHeads) + h)];
        }
        const float m = mask[n((b * cells) + cell)];
        if (std::isinf(m)) {
          EXPECT_EQ(got[n((b * cells) + cell)], m);
        } else {
          finite_got.push_back(got[n((b * cells) + cell)]);
          finite_want.push_back(score + m);
        }
      }
    }
    ExpectNmse(finite_got, finite_want, kIndexerNmse,
               "lightning_indexer x " + std::to_string(batch));
  }
}

TEST_F(GgmlExtOpsTest, HyperConnectionsMatchTheReference) {
  constexpr std::int64_t kHc = 4;
  constexpr std::int64_t kEmbd = 4096;
  constexpr float kEps = 1e-6f;
  constexpr int kIterations = 20;
  const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
  for (const std::int64_t tokens : {1, 5}) {
    // comb: softmax over destinations, then Sinkhorn normalization.
    const std::vector<float> mixes = Normal(151, n(24 * tokens));
    const std::vector<float> scale = {0.5f, 0.7f, 0.9f};
    const std::vector<float> base = Normal(152, 24, 0.2f);
    ggml_tensor* tm = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 24, tokens), mixes);
    ggml_tensor* tscale = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 3), scale);
    ggml_tensor* tbase = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 24), base);
    ggml_tensor* comb = Place(ggml_dsv4_hc_comb(c(), tm, tscale, tbase, kEps, kIterations));
    Launched(kg::HcComb(launch(), comb), "hc_comb");
    std::vector<double> want_comb(n(kHc * kHc * tokens));
    for (std::int64_t t = 0; t < tokens; ++t) {
      std::array<double, 16> m{};  // m[dst + 4 * src]
      constexpr std::size_t kStreams = 4;
      for (std::size_t src = 0; src < kStreams; ++src) {
        double max = -std::numeric_limits<double>::infinity();
        for (std::size_t dst = 0; dst < kStreams; ++dst) {
          const std::size_t idx = dst + (4 * src);
          m[idx] = (mixes[n(t * 24) + 8 + idx] * static_cast<double>(scale[2])) + base[8 + idx];
          max = std::max(max, m[idx]);
        }
        double sum = 0.0;
        for (std::size_t dst = 0; dst < kStreams; ++dst) {
          m[dst + (4 * src)] = std::exp(m[dst + (4 * src)] - max);
          sum += m[dst + (4 * src)];
        }
        for (std::size_t dst = 0; dst < kStreams; ++dst) {
          m[dst + (4 * src)] = (m[dst + (4 * src)] / sum) + kEps;
        }
      }
      const auto cols = [&m] {
        for (std::size_t dst = 0; dst < kStreams; ++dst) {
          double sum = kEps;
          for (std::size_t src = 0; src < kStreams; ++src) {
            sum += m[dst + (4 * src)];
          }
          for (std::size_t src = 0; src < kStreams; ++src) {
            m[dst + (4 * src)] /= sum;
          }
        }
      };
      const auto rows = [&m] {
        for (std::size_t src = 0; src < kStreams; ++src) {
          double sum = kEps;
          for (std::size_t dst = 0; dst < kStreams; ++dst) {
            sum += m[dst + (4 * src)];
          }
          for (std::size_t dst = 0; dst < kStreams; ++dst) {
            m[dst + (4 * src)] /= sum;
          }
        }
      };
      cols();
      for (int i = 1; i < kIterations; ++i) {
        rows();
        cols();
      }
      // dst[idst, isrc, t] at idst + 4 * isrc + 16 * t.
      for (int i = 0; i < 16; ++i) {
        want_comb[n((t * 16) + i)] = m[static_cast<std::size_t>(i)];
      }
    }
    ExpectNmse(Download(comb), want_comb, kDefaultNmse, "hc_comb x " + std::to_string(tokens));

    // pre: the streams folded by their weights; post: spread back.
    const std::vector<float> x = Normal(153, n(kEmbd * kHc * tokens));
    const std::vector<float> weights = Normal(154, n(kHc * tokens));
    ggml_tensor* tx = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kEmbd, kHc, tokens), x);
    ggml_tensor* tw = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kHc, tokens), weights);
    ggml_tensor* pre = Place(ggml_dsv4_hc_pre(c(), tx, tw));
    Launched(kg::HcPre(launch(), pre), "hc_pre");
    std::vector<double> want_pre(n(kEmbd * tokens));
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t i = 0; i < kEmbd; ++i) {
        double sum = 0.0;
        for (std::int64_t h = 0; h < kHc; ++h) {
          sum +=
              static_cast<double>(x[n((((t * kHc) + h) * kEmbd) + i)]) * weights[n((t * kHc) + h)];
        }
        want_pre[n((t * kEmbd) + i)] = sum;
      }
    }
    ExpectNmse(Download(pre), want_pre, kDefaultNmse, "hc_pre x " + std::to_string(tokens));

    const std::vector<float> y = Normal(155, n(kEmbd * tokens));
    const std::vector<float> post = Normal(156, n(kHc * tokens));
    const std::vector<float> mix = Normal(157, n(kHc * kHc * tokens));
    ggml_tensor* ty = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kEmbd, tokens), y);
    ggml_tensor* tpost = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kHc, tokens), post);
    ggml_tensor* tmix = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kHc, kHc, tokens), mix);
    ggml_tensor* spread = Place(ggml_dsv4_hc_post(c(), ty, tx, tpost, tmix));
    Launched(kg::HcPost(launch(), spread), "hc_post");
    std::vector<double> want_post(n(kEmbd * kHc * tokens));
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t dst = 0; dst < kHc; ++dst) {
        for (std::int64_t i = 0; i < kEmbd; ++i) {
          double sum = static_cast<double>(y[n((t * kEmbd) + i)]) * post[n((t * kHc) + dst)];
          for (std::int64_t src = 0; src < kHc; ++src) {
            sum += static_cast<double>(x[n((((t * kHc) + src) * kEmbd) + i)]) *
                   mix[n((t * 16) + dst + (4 * src))];
          }
          want_post[n((((t * kHc) + dst) * kEmbd) + i)] = sum;
        }
      }
    }
    ExpectNmse(Download(spread), want_post, kDefaultNmse, "hc_post x " + std::to_string(tokens));
  }
}

// ---- Flash attention ----

TEST_F(GgmlExtOpsTest, GemmaLocalAttentionRefusesOverflowingTotalIterationsBeforeSubmission) {
  // Each extent, Q stride and output index fits the primitive's contract,
  // but output tiles times KV tiles exceeds the pinned launcher's int.
  // Symbolic bindings make this a metadata-only check: no large buffers
  // are allocated and no kernel may dereference these addresses.
  constexpr std::int64_t rows = 131040, cells = 2097152, heads = 16, sequences = 4;
  auto arena = TensorArena::Create(16).value();
  auto* context = arena.context();
  std::uint64_t next = 1ULL << 44;
  const auto bound = [&next](ggml_tensor* tensor) {
    TensorArena::Bind(tensor, next);
    next += 1ULL << 42;
    return tensor;
  };
  auto* q = bound(ggml_new_tensor_4d(context, GGML_TYPE_F32, 256, rows, heads, sequences));
  q->nb[1] = 0;
  q->nb[2] = 16;
  q->nb[3] = 0;
  auto* k = bound(ggml_new_tensor_4d(context, GGML_TYPE_F16, 256, cells, heads / 2, sequences));
  auto* v = bound(ggml_new_tensor_4d(context, GGML_TYPE_F16, 256, cells, heads / 2, sequences));
  for (auto* tensor : {k, v}) tensor->nb[2] = tensor->nb[3] = 16;
  auto* mask = bound(ggml_new_tensor_4d(context, GGML_TYPE_F16, cells, rows, 1, sequences));
  // A shared visibility row is valid for every query and sequence; unused
  // dimension strides need not describe an enormous packed mask.
  auto* node = bound(ggml_flash_attn_ext(context, q, k, v, mask, 1, 0, 0));
  mask->nb[1] = 0;
  mask->nb[2] = 16;
  mask->nb[3] = 0;
  ggml_prec_set_acc(node, GGML_PREC_F32);
  const auto checked = kg::CheckFlashAttnMmaGqa2(node);
  ASSERT_TRUE(checked.has_value()) << (checked ? "" : checked.error().detail);
  EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
  const std::array<ggml_tensor*, 1> nodes = {node};
  auto graph = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch()));
  ASSERT_TRUE(graph.has_value());
  EXPECT_EQ(graph->steps.front().implementation, kg::kFlashAttnMmaGqa2Name);
  EXPECT_EQ(FailedCode(kg::PlanScratch(launch(), *graph)), KernelError::kRejected);
  launch().ResetScratchPeak();
  EXPECT_EQ(FailedCode(kg::FlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
  EXPECT_EQ(launch().scratch_peak().value(), 0);
  EXPECT_FALSE(launch().faulted());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST_F(GgmlExtOpsTest, GemmaAndLegacyMmaPlansBoundTheLastPaddedKvTile) {
  // The pinned Ampere+ configurations use KV tiles of 64 at D256/group2
  // with eight query columns, 64 at D256/group8 with one query column,
  // 64 at D512/group8 with one query column, and 128 at unmasked D128/MHA
  // with eight query columns. Each case below has
  // a final safe 256-cell block after including the kernel's transient
  // addition of one extra KV tile before modulo. The next padding block
  // is refused. Only metadata is bound; admitted plans never run.
  struct Shape {
    std::int64_t d, rows, heads, kv_heads, kv_tile;
  };
  for (const Shape shape : {Shape{256, 8, 128, 64, 64}, Shape{256, 1, 1024, 128, 64},
                            Shape{512, 1, 512, 64, 64}, Shape{128, 8, 256, 256, 128}}) {
    SCOPED_TRACE(
        std::format("D{} rows{} heads{} KV{}", shape.d, shape.rows, shape.heads, shape.kv_heads));
    auto arena = TensorArena::Create(16).value();
    auto* context = arena.context();
    std::uint64_t next = 1ULL << 44;
    const auto bound = [&next](ggml_tensor* tensor) {
      TensorArena::Bind(tensor, next);
      next += 1ULL << 42;
      return tensor;
    };
    const std::int64_t dst_tiles = shape.kv_heads;
    const std::int64_t max_iterations = (1LL << 31) / (dst_tiles + 1);
    const std::int64_t cells = (max_iterations * shape.kv_tile / 256) * 256;
    EXPECT_LE((cells / shape.kv_tile) * (dst_tiles + 1), 1LL << 31);
    EXPECT_GT(((cells + 256) / shape.kv_tile) * (dst_tiles + 1), 1LL << 31);
    auto* q =
        bound(ggml_new_tensor_4d(context, GGML_TYPE_F32, shape.d, shape.rows, shape.heads, 1));
    auto* k = bound(ggml_new_tensor_4d(context, GGML_TYPE_F16, shape.d, cells, shape.kv_heads, 1));
    auto* v = bound(ggml_new_tensor_4d(context, GGML_TYPE_F16, shape.d, cells, shape.kv_heads, 1));
    for (auto* tensor : {k, v}) {
      tensor->nb[1] = 0;
      tensor->nb[2] = tensor->nb[3] = 16;
    }
    ggml_tensor* mask = nullptr;
    if (shape.d != 128) {
      mask = bound(ggml_new_tensor_2d(context, GGML_TYPE_F16, cells, shape.rows));
    }
    auto* node = bound(ggml_flash_attn_ext(context, q, k, v, mask, 1, 0, 0));
    if (mask != nullptr) {
      mask->nb[1] = 0;
      mask->nb[2] = mask->nb[3] = 16;
    }
    ggml_prec_set_acc(node, GGML_PREC_F32);
    const bool group2 = shape.heads / shape.kv_heads == 2;
    const auto check = [&] {
      if (shape.d == 128) return kg::CheckFlashAttnMma128(node);
      return group2 ? kg::CheckFlashAttnMmaGqa2(node) : kg::CheckFlashAttnMma(node);
    };
    const auto plan = [&] {
      if (shape.d == 128) return kg::PlanFlashAttnMma128(launch(), node);
      return group2 ? kg::PlanFlashAttnMmaGqa2(launch(), node)
                    : kg::PlanFlashAttnMma(launch(), node);
    };
    ASSERT_TRUE(check().has_value());
    const auto admitted = plan();
    ASSERT_TRUE(admitted.has_value()) << (admitted ? "" : admitted.error().detail);
    EXPECT_EQ(admitted->columns, shape.rows);
    k->ne[1] += 256;
    v->ne[1] += 256;
    if (mask != nullptr) mask->ne[0] += 256;
    ASSERT_TRUE(check().has_value());
    EXPECT_EQ(FailedCode(plan()), KernelError::kRejected);
    // The total-only endpoint is also refused: a last partial
    // partition can overflow kbc+iter_k or kb0_start+kbc_stop before
    // subtraction/modulo, even though total iterations fit signed int.
    const std::int64_t total_only_iterations = ((1LL << 31) - 1) / dst_tiles;
    k->ne[1] = v->ne[1] = (total_only_iterations * shape.kv_tile / 256) * 256;
    if (mask != nullptr) mask->ne[0] = k->ne[1];
    EXPECT_LE((k->ne[1] / shape.kv_tile) * dst_tiles, (1LL << 31) - 1);
    EXPECT_GT((k->ne[1] / shape.kv_tile) * (dst_tiles + 1), 1LL << 31);
    ASSERT_TRUE(check().has_value());
    EXPECT_EQ(FailedCode(plan()), KernelError::kRejected);
    EXPECT_FALSE(launch().faulted());
  }
}

TEST_F(GgmlExtOpsTest, UnpaddedD128CellsKeepThePinnedCeilingSumRepresentable) {
  // This unmasked primitive permits unpadded cells. Its pinned eight-query
  // configuration has 128-cell KV tiles and forms (ne11+127) in int.
  // Small output/Q storage and broadcast KV rows keep other bounds valid.
  // Only metadata is planned; these fictional addresses never execute.
  constexpr std::int64_t cells = (1LL << 31) - 128;
  auto arena = TensorArena::Create(16).value();
  auto* context = arena.context();
  std::uint64_t next = 1ULL << 44;
  const auto bound = [&next](ggml_tensor* tensor) {
    TensorArena::Bind(tensor, next);
    next += 1ULL << 42;
    return tensor;
  };
  auto* q = bound(ggml_new_tensor_3d(context, GGML_TYPE_F32, 128, 1, 1));
  auto* k = bound(ggml_new_tensor_3d(context, GGML_TYPE_F16, 128, cells, 1));
  auto* v = bound(ggml_new_tensor_3d(context, GGML_TYPE_F16, 128, cells, 1));
  for (auto* kv : {k, v}) {
    kv->nb[1] = 0;
    kv->nb[2] = kv->nb[3] = 16;
  }
  auto* node = bound(ggml_flash_attn_ext(context, q, k, v, nullptr, 1, 0, 0));
  ggml_prec_set_acc(node, GGML_PREC_F32);
  ASSERT_TRUE(kg::CheckFlashAttnMma128(node).has_value());
  const auto admitted = kg::PlanFlashAttnMma128(launch(), node);
  ASSERT_TRUE(admitted.has_value()) << (admitted ? "" : admitted.error().detail);
  EXPECT_EQ(admitted->columns, 8);
  EXPECT_EQ(cells + 127, (1LL << 31) - 1);
  k->ne[1] = v->ne[1] = cells + 1;
  ASSERT_TRUE(kg::CheckFlashAttnMma128(node).has_value());
  EXPECT_EQ(k->ne[1] + 127, 1LL << 31);
  EXPECT_EQ(FailedCode(kg::PlanFlashAttnMma128(launch(), node)), KernelError::kRejected);
  EXPECT_FALSE(launch().faulted());
}

TEST_F(GgmlExtOpsTest, GemmaAndLegacyVectorPlansBoundTailEfficiencyArithmetic) {
  // The pinned launcher multiplies its trial block count by 100 in int.
  // Probe the final 256-cell padding block admitted by that bound and the
  // next block at both the existing D64 and new D256 vector identities.
  // KV row/head broadcasts keep the symbolic backing spans small.
  for (const std::int64_t d : {64, 256}) {
    SCOPED_TRACE(d);
    auto arena = TensorArena::Create(16).value();
    auto* context = arena.context();
    std::uint64_t next = 1ULL << 44;
    const auto bound = [&next](ggml_tensor* tensor) {
      TensorArena::Bind(tensor, next);
      next += 1ULL << 42;
      return tensor;
    };
    const std::int64_t cells = d == 64 ? 85899264 : 343597312;
    auto* q = bound(ggml_new_tensor_3d(context, GGML_TYPE_F32, d, 1, 16));
    auto* k = bound(ggml_new_tensor_3d(context, GGML_TYPE_F16, d, cells, 8));
    auto* v = bound(ggml_new_tensor_3d(context, GGML_TYPE_F16, d, cells, 8));
    for (auto* tensor : {k, v}) {
      tensor->nb[1] = 0;
      tensor->nb[2] = tensor->nb[3] = 16;
    }
    auto* mask = bound(ggml_new_tensor_2d(context, GGML_TYPE_F16, cells, 1));
    auto* node = bound(ggml_flash_attn_ext(context, q, k, v, mask, 1, 0, 0));
    ggml_prec_set_acc(node, GGML_PREC_F32);
    const auto check = [&] {
      return d == 64 ? kg::CheckFlashAttnVec(node) : kg::CheckFlashAttnVec256(node);
    };
    const auto plan = [&] {
      return d == 64 ? kg::PlanFlashAttnVec(launch(), node)
                     : kg::PlanFlashAttnVec256(launch(), node);
    };
    ASSERT_TRUE(check().has_value());
    const auto admitted = plan();
    ASSERT_TRUE(admitted.has_value()) << (admitted ? "" : admitted.error().detail);
    k->ne[1] += 256;
    v->ne[1] += 256;
    mask->ne[0] += 256;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) mask->nb[i] += 512;
    ASSERT_TRUE(check().has_value());
    EXPECT_EQ(FailedCode(plan()), KernelError::kRejected);
    EXPECT_FALSE(launch().faulted());
  }
}

TEST_F(GgmlExtOpsTest, GemmaLocalPrefillRefusesUnfundedQueryTilesBeforeSubmission) {
  constexpr std::int64_t rows = 1025, padded = 1056, cells = 1280, heads = 16;
  auto* q = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, 256, rows, heads, 1),
                  Normal(361, static_cast<std::size_t>(256 * rows * heads), 0.1f));
  auto* k = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, 256, cells, heads / 2, 1),
                  Halves(Normal(362, static_cast<std::size_t>(256 * cells * heads / 2), 0.1f)));
  auto* v = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, 256, cells, heads / 2, 1),
                  Halves(Normal(363, static_cast<std::size_t>(256 * cells * heads / 2))));
  std::vector<float> visibility(static_cast<std::size_t>(cells * padded),
                                -std::numeric_limits<float>::infinity());
  for (std::int64_t row = 0; row < rows; ++row) {
    visibility[static_cast<std::size_t>(row * cells + row % cells)] = 0;
  }
  auto* mask = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, cells, padded), Halves(visibility));
  auto* node = Place(ggml_flash_attn_ext(c(), q, k, v, mask, 1, 0, 0));
  ggml_prec_set_acc(node, GGML_PREC_F32);
  ASSERT_EQ(cudaMemset(node->data, 0xAB, ggml_nbytes(node)), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  mask->ne[1] = rows;
  EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
  EXPECT_EQ(FailedCode(kg::FlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
  const auto untouched = Download<std::uint8_t>(node);
  EXPECT_TRUE(std::ranges::all_of(untouched, [](std::uint8_t b) { return b == 0xAB; }));
  EXPECT_FALSE(launch().faulted());
  mask->ne[1] = padded;
  const auto plan = kg::PlanFlashAttnMmaGqa2(launch(), node);
  ASSERT_TRUE(plan.has_value());
  EXPECT_TRUE(plan->mask_prepass);
  EXPECT_EQ(plan->columns, 32);
  launch().ResetScratchPeak();
  Launched(kg::FlashAttnMmaGqa2(launch(), node), "funded partial-query prefill tile");
  EXPECT_LE(launch().scratch_peak().value(), plan->scratch);
  const auto got = Download(node);
  // Widen the actual uploaded F16 bytes. A single visible cell has the
  // independent mathematical result V; also require pinned-case agreement.
  const auto values = Download<ggml_fp16_t>(v);
  std::vector<double> want(got.size());
  for (std::int64_t row = 0; row < rows; ++row) {
    for (std::int64_t h = 0; h < heads; ++h) {
      for (std::int64_t i = 0; i < 256; ++i) {
        want[static_cast<std::size_t>((row * heads + h) * 256 + i)] = ggml_fp16_to_fp32(
            values[static_cast<std::size_t>(((h / 2) * cells + row % cells) * 256 + i)]);
      }
    }
  }
  Launched(launch().Run(jitllm::base::Bytes(plan->scratch),
                        [node](ggml_backend_cuda_context& context) {
                          kg::detail::FlashAttnMmaCaseGqa2(32)(context, node);
                        }),
           "pinned partial-query prefill tile");
  const auto reference = Download(node);
  EXPECT_EQ(std::memcmp(got.data(), reference.data(), got.size() * sizeof(float)), 0);
  ExpectNmse(reference, want, kMulMatNmse, "pinned prefill single-visible-cell V reference");
  ExpectNmse(got, want, kMulMatNmse, "registered prefill single-visible-cell V reference");
}

TEST_F(GgmlExtOpsTest, EightStreamMmaMatchesTwoFourRootCohortPartitionsExactly) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  for (const std::int64_t d : {256, 512})
    for (const std::int64_t cells : {256, 1024}) {
      SCOPED_TRACE(std::to_string(d) + "/" + std::to_string(cells));
      constexpr std::int64_t heads = 32, owners = 8;
      const auto kvh = heads / (d == 256 ? 2 : 8);
      const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
      auto arena = TensorArena::Create(128).value();
      auto* ctx = arena.context();
      auto* packed_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners),
                             Normal(8101, n(d * heads * owners), 0.25F));
      auto* q = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
      TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(packed_q->data));
      const auto kdata = Halves(Normal(8102, n(d * kvh * cells * owners), 0.25F));
      const auto vdata = Halves(Normal(8103, n(d * kvh * cells * owners), 0.25F));
      auto* packed_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
      auto* packed_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
      auto* k = ggml_permute(ctx, packed_k, 0, 2, 1, 3);
      auto* v = ggml_permute(ctx, packed_v, 0, 2, 1, 3);
      TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(packed_k->data));
      TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(packed_v->data));
      std::vector<float> masks(n(cells * 32 * owners), -std::numeric_limits<float>::infinity());
      for (std::int64_t owner = 0; owner < owners; ++owner)
        for (std::int64_t cell = 0; cell < cells - 37 - owner * 3; ++cell)
          masks[n(owner * cells * 32 + cell)] = 0;
      auto* mask =
          Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, owners), Halves(masks));
      auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, 0));
      ggml_prec_set_acc(whole, GGML_PREC_F32);
      const auto full_plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                                      : kg::PlanFlashAttnMma(launch(), whole);
      ASSERT_TRUE(full_plan) << jitllm::test_support::Failed(full_plan)->detail;
      ASSERT_EQ(full_plan->blocks % 2, 0);
      EXPECT_EQ(full_plan->columns, d == 256 ? 4 : 1);
      EXPECT_EQ(full_plan->group, d == 256 ? 2 : 8);
      EXPECT_FALSE(full_plan->sparse);
      const auto run_whole = [&](LaunchContext& l) {
        return d == 256 ? kg::FlashAttnMmaGqa2(l, whole) : kg::FlashAttnMma(l, whole);
      };
      std::array<kg::FlashAttnOwners, 2> quads;
      for (std::size_t quad = 0; quad < quads.size(); ++quad) {
        const auto first = quad * 4;
        const auto view = [&](ggml_tensor* tensor, std::array<std::int64_t, 4> ne) {
          const auto offset = first * tensor->nb[3];
          auto* slice = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3], tensor->nb[1],
                                     tensor->nb[2], tensor->nb[3], offset);
          TensorArena::Bind(slice, reinterpret_cast<std::uintptr_t>(tensor->data) + offset);
          return slice;
        };
        auto& in = quads[quad];
        in.q = view(q, {d, 1, heads, 4});
        in.mask = view(mask, {cells, 32, 1, 4});
        in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 4));
        in.logical_cohort = 8;
        for (std::size_t owner = 0; owner < 4; ++owner) {
          const auto part = n(d * kvh * cells);
          const auto begin = static_cast<std::ptrdiff_t>((first + owner) * part);
          const auto end = begin + static_cast<std::ptrdiff_t>(part);
          // Separate allocations authenticate actual roots; their bytes match
          // the corresponding planes of the contiguous eight-stream control.
          auto* raw_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                              std::vector<ggml_fp16_t>(kdata.begin() + begin, kdata.begin() + end));
          auto* raw_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                              std::vector<ggml_fp16_t>(vdata.begin() + begin, vdata.begin() + end));
          in.k[owner] = ggml_permute(ctx, raw_k, 0, 2, 1, 3);
          in.v[owner] = ggml_permute(ctx, raw_v, 0, 2, 1, 3);
          TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                            reinterpret_cast<std::uintptr_t>(raw_k->data));
          TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                            reinterpret_cast<std::uintptr_t>(raw_v->data));
        }
        const auto plan = kg::PlanFlashAttnOwners(launch(), in);
        ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
        EXPECT_EQ(plan->effective_cohort, 8U);
        EXPECT_EQ(plan->cohort_blocks, full_plan->blocks);
        EXPECT_EQ(plan->original.blocks * 2, full_plan->blocks);
        const auto quad_tiles = static_cast<int>(kvh) * 4;
        EXPECT_EQ(plan->original.blocks % quad_tiles == 0,
                  full_plan->blocks % (quad_tiles * 2) == 0);
        std::cout << "OWNER_C8_PLAN D=" << d << " cells=" << cells
                  << " full_blocks=" << full_plan->blocks
                  << " quad_blocks=" << plan->original.blocks
                  << " effective_cohort=" << plan->effective_cohort
                  << " scratch=" << plan->original.scratch << '\n';
        auto legacy = in;
        legacy.logical_cohort = 4;
        const auto legacy_plan = kg::PlanFlashAttnOwners(launch(), legacy);
        ASSERT_TRUE(legacy_plan);
        EXPECT_EQ(legacy_plan->effective_cohort, 4U);
        EXPECT_LE(plan->original.scratch, legacy_plan->original.scratch);
        launch().ResetScratchPeak();
        ASSERT_TRUE(kg::FlashAttnOwnerRoots(launch(), in));
        EXPECT_LE(launch().scratch_peak().value(), plan->original.scratch);
      }
      ASSERT_TRUE(run_whole(launch()));
      const auto expected = Download(whole);
      EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
      const auto compare = [&] {
        for (std::size_t quad = 0; quad < quads.size(); ++quad) {
          const auto actual = Download(quads[quad].output);
          const auto offset = quad * actual.size();
          EXPECT_EQ(
              std::memcmp(actual.data(), expected.data() + offset, actual.size() * sizeof(float)),
              0);
        }
      };
      compare();
      auto graph = launch().Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
        if (auto r = run_whole(l); !r) return r;
        for (const auto& quad : quads)
          if (auto r = kg::FlashAttnOwnerRoots(l, quad); !r) return r;
        return {};
      });
      ASSERT_TRUE(graph);
      ASSERT_TRUE(launch().Launch(*graph));
      EXPECT_EQ(std::memcmp(Download(whole).data(), expected.data(), expected.size() * 4), 0);
      compare();
    }
}

TEST_F(GgmlExtOpsTest, TwelveStreamMmaMatchesThreeFourRootCohortPartitionsExactly) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  ASSERT_GT(sms, 0);
  for (const std::int64_t heads : {16, 32})
    for (const std::int64_t d : {256, 512})
      for (const std::int64_t cells : {256, 1024}) {
        SCOPED_TRACE(std::to_string(heads) + "/" + std::to_string(d) + "/" + std::to_string(cells));
        constexpr std::int64_t owners = 12;
        const auto kvh = heads / (d == 256 ? 2 : 8);
        const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
        auto arena = TensorArena::Create(128).value();
        auto* ctx = arena.context();
        auto* packed_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners),
                               Normal(12101, n(d * heads * owners), 0.25F));
        auto* q = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
        TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(packed_q->data));
        const auto kdata = Halves(Normal(12102, n(d * kvh * cells * owners), 0.25F));
        const auto vdata = Halves(Normal(12103, n(d * kvh * cells * owners), 0.25F));
        auto* packed_k =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
        auto* packed_v =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
        auto* k = ggml_permute(ctx, packed_k, 0, 2, 1, 3);
        auto* v = ggml_permute(ctx, packed_v, 0, 2, 1, 3);
        TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(packed_k->data));
        TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(packed_v->data));
        std::vector<float> masks(n(cells * 32 * owners), -std::numeric_limits<float>::infinity());
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t cell = 0; cell < cells - 37 - owner * 3; ++cell)
            masks[n(owner * cells * 32 + cell)] = 0;
        auto* mask =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, owners), Halves(masks));
        auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, 0));
        ggml_prec_set_acc(whole, GGML_PREC_F32);
        const auto full_plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                                        : kg::PlanFlashAttnMma(launch(), whole);
        ASSERT_TRUE(full_plan) << jitllm::test_support::Failed(full_plan)->detail;
        ASSERT_EQ(full_plan->blocks % 3, 0);
        EXPECT_EQ(full_plan->columns, d == 256 ? 4 : 1);
        EXPECT_EQ(full_plan->group, d == 256 ? 2 : 8);
        EXPECT_FALSE(full_plan->sparse);
        EXPECT_TRUE(full_plan->mask_prepass);
        const auto run_whole = [&](LaunchContext& l) {
          return d == 256 ? kg::FlashAttnMmaGqa2(l, whole) : kg::FlashAttnMma(l, whole);
        };
        std::array<kg::FlashAttnOwners, 3> quads;
        std::array<kg::FlashAttnOwnersPlan, 3> plans;
        auto scratch = full_plan->scratch;
        for (std::size_t quad = 0; quad < quads.size(); ++quad) {
          const auto first = quad * 4;
          const auto view = [&](ggml_tensor* tensor, std::array<std::int64_t, 4> ne) {
            const auto offset = first * tensor->nb[3];
            auto* slice = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3], tensor->nb[1],
                                       tensor->nb[2], tensor->nb[3], offset);
            TensorArena::Bind(slice, reinterpret_cast<std::uintptr_t>(tensor->data) + offset);
            return slice;
          };
          auto& in = quads[quad];
          in.q = view(q, {d, 1, heads, 4});
          in.mask = view(mask, {cells, 32, 1, 4});
          in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 4));
          in.logical_cohort = 12;
          for (std::size_t owner = 0; owner < 4; ++owner) {
            const auto part = n(d * kvh * cells);
            const auto begin = static_cast<std::ptrdiff_t>((first + owner) * part);
            const auto end = begin + static_cast<std::ptrdiff_t>(part);
            // Separate allocations authenticate actual roots; their bytes match
            // the corresponding planes of the contiguous twelve-stream control.
            auto* raw_k =
                Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                      std::vector<ggml_fp16_t>(kdata.begin() + begin, kdata.begin() + end));
            auto* raw_v =
                Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                      std::vector<ggml_fp16_t>(vdata.begin() + begin, vdata.begin() + end));
            in.k[owner] = ggml_permute(ctx, raw_k, 0, 2, 1, 3);
            in.v[owner] = ggml_permute(ctx, raw_v, 0, 2, 1, 3);
            TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_k->data));
            TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_v->data));
          }
          const auto plan = kg::PlanFlashAttnOwners(launch(), in);
          ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
          EXPECT_EQ(plan->effective_cohort, 12U);
          EXPECT_EQ(plan->original.head, full_plan->head);
          EXPECT_EQ(plan->original.columns, full_plan->columns);
          EXPECT_EQ(plan->original.group, full_plan->group);
          EXPECT_EQ(plan->original.mask_prepass, full_plan->mask_prepass);
          EXPECT_FALSE(plan->original.sparse);
          EXPECT_GT(plan->original_blocks_per_sm, 0);
          EXPECT_GT(plan->owner_blocks_per_sm, 0);
          EXPECT_EQ(plan->cohort_blocks, full_plan->blocks);
          EXPECT_EQ(plan->original.blocks * 3, full_plan->blocks);
          const auto quad_tiles = static_cast<int>(kvh) * 4;
          EXPECT_EQ(plan->original.blocks % quad_tiles == 0,
                    full_plan->blocks % (quad_tiles * 3) == 0);
          std::cout << "OWNER_C12_PLAN heads=" << heads << " D=" << d << " cells=" << cells
                    << " sms=" << sms << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                    << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                    << " full_blocks=" << full_plan->blocks
                    << " quad_blocks=" << plan->original.blocks
                    << " effective_cohort=" << plan->effective_cohort
                    << " scratch=" << plan->original.scratch << '\n';
          auto legacy = in;
          legacy.logical_cohort = 4;
          const auto legacy_plan = kg::PlanFlashAttnOwners(launch(), legacy);
          ASSERT_TRUE(legacy_plan);
          EXPECT_EQ(legacy_plan->effective_cohort, 4U);
          EXPECT_LE(plan->original.scratch, legacy_plan->original.scratch);
          plans[quad] = *plan;
          scratch = std::max(scratch, plan->original.scratch);
        }
        // Sequential launches reuse the same explicitly funded
        // workspace. Do not fund three simultaneous quad workspaces or borrow
        // the fixture's larger pool for this proof.
        auto context = LaunchContext::Create(launch().device(), *execution_, stream_,
                                             {.base = Allocate(scratch), .size = Bytes(scratch)});
        ASSERT_TRUE(context) << jitllm::test_support::Failed(context)->detail;
        auto& bounded = **context;
        EXPECT_EQ(bounded.workspace().size.value(), scratch);
        EXPECT_TRUE(bounded.UsesStream(*execution_, stream_));
        const auto submission = execution_->Submission(stream_);
        ASSERT_TRUE(submission);
        const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
        ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
        for (std::size_t quad = 0; quad < quads.size(); ++quad) {
          ASSERT_EQ(cudaMemsetAsync(quads[quad].output->data, 0xFF, ggml_nbytes(quads[quad].output),
                                    stream),
                    cudaSuccess);
          bounded.ResetScratchPeak();
          ASSERT_TRUE(kg::FlashAttnOwnerRoots(bounded, quads[quad]));
          EXPECT_LE(bounded.scratch_peak().value(), plans[quad].original.scratch);
        }
        bounded.ResetScratchPeak();
        ASSERT_TRUE(run_whole(bounded));
        EXPECT_LE(bounded.scratch_peak().value(), full_plan->scratch);
        const auto expected = Download(whole);
        EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
        const auto compare = [&] {
          for (std::size_t quad = 0; quad < quads.size(); ++quad) {
            const auto actual = Download(quads[quad].output);
            const auto offset = quad * actual.size();
            EXPECT_EQ(
                std::memcmp(actual.data(), expected.data() + offset, actual.size() * sizeof(float)),
                0);
          }
        };
        compare();
        bounded.ResetScratchPeak();
        auto graph = bounded.Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
          if (auto r = run_whole(l); !r) return r;
          for (const auto& quad : quads)
            if (auto r = kg::FlashAttnOwnerRoots(l, quad); !r) return r;
          return {};
        });
        ASSERT_TRUE(graph) << jitllm::test_support::Failed(graph)->detail;
        EXPECT_LE(bounded.scratch_peak().value(), scratch);
        for (int replay = 0; replay < 2; ++replay) {
          SCOPED_TRACE("capture replay " + std::to_string(replay));
          ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
          for (const auto& quad : quads)
            ASSERT_EQ(cudaMemsetAsync(quad.output->data, 0xFF, ggml_nbytes(quad.output), stream),
                      cudaSuccess);
          ASSERT_TRUE(bounded.Launch(*graph));
          EXPECT_EQ(
              std::memcmp(Download(whole).data(), expected.data(), expected.size() * sizeof(float)),
              0);
          compare();
        }
        EXPECT_FALSE(bounded.faulted());
        Finish();
      }
}

TEST(FlashAttnOwnersPartitionTest, WideMultirowBoundDoesNotWidenDecodeOrOtherGrids) {
  EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 512, 4, 2, true));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 513, 4, 2, true));
  EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 4096, 16, 2, true, true));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 4097, 16, 2, true, true));
  for (const int tiles : {4, 8, 12, 16})
    EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 4096, tiles, 2, true, true));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 4096, 6, 2, true, true));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 4096, 16, 3, true, true));
}

TEST(FlashAttnOwnersPartitionTest, LargerQueryTilesDoNotWidenTheKvOrDecodeBounds) {
  EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 512, 32, 2, true, false, 8));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 513, 32, 2, true, false, 8));
  EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 4096, 32, 2, true, true, 8));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 4097, 32, 2, true, true, 8));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 32, 2, true));
  EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 512, 64, 2, true, false, 16));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 513, 64, 2, true, false, 16));
  EXPECT_TRUE(kg::detail::PlanOwnerPartition(96, 4096, 64, 2, true, true, 16));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 4097, 64, 2, true, true, 16));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 64, 2, true));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 68, 2, true, false, 17));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 64, 2, true, false, 15));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 64, 3, true, false, 16));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 32, 2, true, false, 7));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 32, 3, true, false, 8));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(96, 512, 4, 2, true, false, -1));
  for (const int query_tiles : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
    EXPECT_TRUE(
        kg::detail::PlanOwnerPartition(96, 512, 4 * query_tiles, 2, true, false, query_tiles));
}

TEST(FlashAttnOwnersPartitionTest, WholeTilePreferenceMatchesTheReleaseEfficiencyBoundary) {
  for (const std::uint32_t cohort : {2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U}) {
    const int tiles = 4 * static_cast<int>(cohort);
    const auto whole = kg::detail::PlanOwnerPartition(tiles, 16, 4, cohort, true);
    ASSERT_TRUE(whole) << whole.error().detail;
    EXPECT_EQ(whole->cohort_blocks, tiles);
    EXPECT_EQ(whole->effective_cohort, cohort);
    const auto under = kg::detail::PlanOwnerPartition(tiles * 2, 16, 4, cohort, true);
    const auto ordinary = kg::detail::PlanOwnerPartition(tiles * 2, 16, 4, cohort, false);
    ASSERT_TRUE(under) << under.error().detail;
    ASSERT_TRUE(ordinary) << ordinary.error().detail;
    EXPECT_EQ(under->cohort_blocks, ordinary->cohort_blocks);
    EXPECT_EQ(under->quad_blocks, ordinary->quad_blocks);
  }
  const auto boundary = kg::detail::PlanOwnerPartition(64, 16, 4, 12, true);
  const auto below = kg::detail::PlanOwnerPartition(65, 16, 4, 12, true);
  ASSERT_TRUE(boundary) << boundary.error().detail;
  ASSERT_TRUE(below) << below.error().detail;
  EXPECT_EQ(boundary->cohort_blocks, 48);  // 75 percent, no split-K metadata
  EXPECT_NE(below->cohort_blocks, 48);     // 73 percent, stream-K (four-root fallback)
}

TEST(FlashAttnOwnersPartitionTest, NonDivisibleTwelveStreamGridPreservesFourRootGeometry) {
  // A real provider with this occupancy limit cannot split the original
  // twelve-stream grid into three integral block ranges. Preserve the
  // ordinary four-root grid; do not claim equality with a full-twelve call.
  const auto four = kg::detail::PlanOwnerPartition(47, 4, 16, 4);
  const auto twelve = kg::detail::PlanOwnerPartition(47, 4, 16, 12);
  ASSERT_TRUE(four);
  ASSERT_TRUE(twelve);
  ASSERT_EQ(four->cohort_blocks, 47);
  EXPECT_EQ(twelve->effective_cohort, 4U);
  EXPECT_EQ(twelve->cohort_blocks, four->cohort_blocks);
  EXPECT_EQ(twelve->quad_blocks, four->quad_blocks);
}

TEST_F(GgmlExtOpsTest, GemmaLocalAttentionPreservesIndependentRingMasksAndPlansScratch) {
  constexpr std::int64_t d = 256, cells = 1280;
  for (const std::int64_t heads : {16, 32}) {
    for (const std::int64_t rows : {1, 2, 4, 8, 16, 33}) {
      std::vector<float> first_sequence;
      for (const std::int64_t sequences : {1, 2, 4}) {
        auto arena = TensorArena::Create(128).value();
        auto* ctx = arena.context();
        const auto count = [](std::int64_t n) { return static_cast<std::size_t>(n); };
        const auto q = Normal(371, count(d * rows * heads * sequences), 0.25f);
        const auto k = Halves(Normal(372, count(d * cells * (heads / 2) * sequences), 0.25f));
        const auto v = Halves(Normal(373, count(d * cells * (heads / 2) * sequences)));
        const std::int64_t columns = rows <= 4 ? 4 : rows <= 8 ? 8 : rows <= 16 ? 16 : 32;
        const std::int64_t mask_rows = (rows + columns - 1) / columns * columns;
        std::vector<float> mask(count(cells * mask_rows * sequences),
                                -std::numeric_limits<float>::infinity());
        for (std::int64_t seq = 0; seq < sequences; ++seq) {
          const std::int64_t first = cells - 2 + seq * 17;
          for (std::int64_t r = 0; r < rows; ++r) {
            const auto position = first + r;
            for (std::int64_t p = position - 1023; p <= position; ++p) {
              mask[count((seq * mask_rows + r) * cells + p % cells)] = 0;
            }
          }
        }
        // Projection outputs are packed [D,heads,rows,sequence]. Attention
        // consumes their permuted view [D,rows,heads,sequence] without a copy.
        std::vector<float> projection(q.size());
        for (std::int64_t seq = 0; seq < sequences; ++seq)
          for (std::int64_t h = 0; h < heads; ++h)
            for (std::int64_t r = 0; r < rows; ++r)
              for (std::int64_t i = 0; i < d; ++i)
                projection[count(((seq * rows + r) * heads + h) * d + i)] =
                    q[count(((seq * heads + h) * rows + r) * d + i)];
        auto* packed_q =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, rows, sequences), projection);
        auto* tq = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
        TensorArena::Bind(tq, reinterpret_cast<std::uintptr_t>(packed_q->data));
        auto* tk = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, cells, heads / 2, sequences), k);
        auto* tv = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, cells, heads / 2, sequences), v);
        auto* tm = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, mask_rows, 1, sequences),
                         Halves(mask));
        auto* node = Place(ggml_flash_attn_ext(ctx, tq, tk, tv, tm, 1, 0, 0));
        ggml_prec_set_acc(node, GGML_PREC_F32);
        const bool vector = rows == 1 && sequences == 1;
        EXPECT_EQ(kg::FlashAttnVec256Selected(launch(), node), vector);
        auto choices = kg::DeviceChoicesOf(launch());
        const std::array<ggml_tensor*, 1> nodes = {node};
        auto graph = kg::PlanGraph(nodes, false, choices);
        ASSERT_TRUE(graph.has_value());
        EXPECT_EQ(graph->steps.front().implementation,
                  vector ? kg::kFlashAttnVec256Name : kg::kFlashAttnMmaGqa2Name);
        const auto scratch = kg::PlanScratch(launch(), *graph);
        ASSERT_TRUE(scratch.has_value());
        auto mma = kg::PlanFlashAttnMmaGqa2(launch(), node);
        ASSERT_TRUE(mma.has_value()) << (mma ? "" : mma.error().detail);
        EXPECT_EQ(mma->group, 2);
        EXPECT_EQ(mma->columns, columns);
        EXPECT_EQ(mma->mask_prepass, sequences > 1);
        launch().ResetScratchPeak();
        Launched(
            vector ? kg::FlashAttnVec256(launch(), node) : kg::FlashAttnMmaGqa2(launch(), node),
            "selected local attention");
        EXPECT_LE(launch().scratch_peak().value(), *scratch);
        const auto got = Download(node);
        if (sequences == 1)
          first_sequence = got;
        else {
          std::vector<float> first(
              got.begin(), got.begin() + static_cast<std::ptrdiff_t>(first_sequence.size()));
          ExpectNmse(first, std::vector<double>(first_sequence.begin(), first_sequence.end()),
                     kMulMatNmse, "independent first sequence matches solo");
        }
        // Pinned original case, separately launched with the checked paid scratch.
        Launched(launch().Run(jitllm::base::Bytes(mma->scratch),
                              [node, columns](ggml_backend_cuda_context& context) {
                                kg::detail::FlashAttnMmaCaseGqa2(static_cast<int>(columns))(context,
                                                                                            node);
                              }),
                 "pinned group2 reference");
        const auto reference = Download(node);
        if (!vector)
          EXPECT_EQ(std::memcmp(got.data(), reference.data(), got.size() * sizeof(float)), 0);
        std::vector<double> want(got.size());
        std::vector<double> logits(count(cells));
        for (std::int64_t seq = 0; seq < sequences; ++seq) {
          for (std::int64_t h = 0; h < heads; ++h) {
            for (std::int64_t r = 0; r < rows; ++r) {
              double max = -std::numeric_limits<double>::infinity();
              for (std::int64_t cell = 0; cell < cells; ++cell) {
                if (std::isinf(mask[count((seq * mask_rows + r) * cells + cell)])) {
                  logits[count(cell)] = -std::numeric_limits<double>::infinity();
                  continue;
                }
                double dot = 0;
                for (std::int64_t i = 0; i < d; ++i) {
                  dot += static_cast<double>(q[count(((seq * heads + h) * rows + r) * d + i)]) *
                         ggml_fp16_to_fp32(
                             k[count(((seq * (heads / 2) + h / 2) * cells + cell) * d + i)]);
                }
                logits[count(cell)] = dot;  // scale1, never inverse sqrt(D)
                max = std::max(max, dot);
              }
              double sum = 0;
              for (double l : logits)
                if (std::isfinite(l)) sum += std::exp(l - max);
              for (std::int64_t cell = 0; cell < cells; ++cell) {
                if (!std::isfinite(logits[count(cell)])) continue;
                const double probability = std::exp(logits[count(cell)] - max) / sum;
                for (std::int64_t i = 0; i < d; ++i) {
                  want[count(((seq * rows + r) * heads + h) * d + i)] +=
                      probability *
                      ggml_fp16_to_fp32(
                          v[count(((seq * (heads / 2) + h / 2) * cells + cell) * d + i)]);
                }
              }
            }
          }
        }
        ExpectNmse(got, want, kMulMatNmse, "Gemma local independent FP64 ring attention");
        Launched(
            vector ? kg::FlashAttnVec256(launch(), node) : kg::FlashAttnMmaGqa2(launch(), node),
            "local exact repeat");
        const auto repeated = Download(node);
        EXPECT_EQ(std::memcmp(got.data(), repeated.data(), got.size() * sizeof(float)), 0);
        if (sequences > 1) {
          const auto saved = tm->ne[1];
          tm->ne[1] = rows;
          if (rows % columns != 0) {
            EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
          }
          tm->ne[1] = saved;
          tm->ne[3] = 1;  // broadcast masks cannot fund this pre-pass
          EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
          tm->ne[3] = sequences;
        }
      }
    }
  }
}

// Gemma2 operator qualification only: these cases do not admit a model or
// owner-attention adapter. Scale is the actual 2B 1/sqrt(D256) contract.
TEST_F(GgmlExtOpsTest, Gemma2SoftcapUsesFundedSpecializationsAndMatchesFp64) {
  constexpr std::int64_t d = 256, heads = 8, kv_heads = 4, cells = 512;
  constexpr float scale = 0.0625F;
  struct Case {
    bool vector;
    std::int64_t rows;
    std::int64_t sequences = 1;
  };
  int underfunded_vector = 0, underfunded_mma = 0;
  std::vector<std::vector<float>> uncapped;
  for (const float cap : {0.0F, 50.0F, 25.0F}) {
    std::size_t case_index = 0;
    for (const auto shape : {Case{true, 1}, Case{true, 3}, Case{false, 1}, Case{false, 5},
                             Case{false, 9}, Case{false, 17}, Case{false, 5, 2}}) {
      SCOPED_TRACE(std::format("softcap={} vector={} rows={}", cap, shape.vector, shape.rows));
      auto arena = TensorArena::Create(32).value();
      auto* ctx = arena.context();
      const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
      const auto rows = shape.rows;
      const auto sequences = shape.sequences;
      const auto q = Normal(471, n(d * rows * heads * sequences), 8.0F);
      const auto k = Halves(Normal(472, n(d * cells * kv_heads * sequences), 4.0F));
      const auto v = Halves(Normal(473, n(d * cells * kv_heads * sequences)));
      const auto columns = rows <= 4 ? 4 : rows <= 8 ? 8 : rows <= 16 ? 16 : 32;
      const auto mask_rows = (rows + columns - 1) / columns * columns;
      std::vector<float> mask(n(cells * mask_rows * sequences),
                              -std::numeric_limits<float>::infinity());
      for (std::int64_t seq = 0; seq < sequences; ++seq)
        for (std::int64_t r = 0; r < rows; ++r)
          for (std::int64_t cell = 224 + r; cell <= 351 + r; ++cell)
            mask[n((seq * mask_rows + r) * cells + cell)] = 0;
      auto* tq = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, rows, heads, sequences), q);
      auto* tk = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, cells, kv_heads, sequences), k);
      auto* tv = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, cells, kv_heads, sequences), v);
      auto* tm = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, mask_rows, 1, sequences),
                       Halves(mask));
      auto* node = Place(ggml_flash_attn_ext(ctx, tq, tk, tv, tm, scale, 0, cap));
      ggml_prec_set_acc(node, GGML_PREC_F32);
      if (sequences == 1) ASSERT_TRUE(kg::CheckFlashAttnVec256(node));
      ASSERT_TRUE(kg::CheckFlashAttnMmaGqa2(node));
      EXPECT_EQ(kg::FlashAttnVec256Selected(launch(), node), rows == 1 && sequences == 1);
      const std::array<ggml_tensor*, 1> nodes{node};
      auto selected_graph = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch()));
      ASSERT_TRUE(selected_graph);
      EXPECT_EQ(selected_graph->steps.front().implementation,
                rows == 1 && sequences == 1 ? kg::kFlashAttnVec256Name : kg::kFlashAttnMmaGqa2Name);
      EXPECT_TRUE(kg::PlanScratch(launch(), *selected_graph));
      const auto run = [&](LaunchContext& l) {
        return shape.vector ? kg::FlashAttnVec256(l, node) : kg::FlashAttnMmaGqa2(l, node);
      };
      std::uint64_t scratch = 0;
      if (shape.vector) {
        auto plan = kg::PlanFlashAttnVec256(launch(), node);
        ASSERT_TRUE(plan) << plan.error().detail;
        EXPECT_EQ(plan->columns_per_block, rows == 1 ? 1 : 2);
        scratch = plan->scratch;
        std::cout << "SOFTCAP_VECTOR cap=" << cap << " rows=" << rows
                  << " parallel=" << plan->parallel_blocks << " scratch=" << scratch << '\n';
      } else {
        auto plan = kg::PlanFlashAttnMmaGqa2(launch(), node);
        ASSERT_TRUE(plan) << plan.error().detail;
        auto selected = kg::detail::FlashAttnMmaShapeGqa2(columns, launch().device(), cap != 0);
        ASSERT_TRUE(selected) << selected.error();
        ASSERT_GT(selected->blocks_per_sm, 0);
        EXPECT_EQ(plan->columns, columns);
        EXPECT_EQ(plan->group, 2);
        EXPECT_EQ(plan->mask_prepass, sequences > 1);
        scratch = plan->scratch;
        // Existing owner and zero-softcap callers omit the new argument.
        auto default_zero = kg::detail::FlashAttnMmaShapeGqa2(columns, launch().device());
        auto explicit_zero = kg::detail::FlashAttnMmaShapeGqa2(columns, launch().device(), false);
        ASSERT_TRUE(default_zero);
        ASSERT_TRUE(explicit_zero);
        EXPECT_EQ(default_zero->kv_batch, explicit_zero->kv_batch);
        EXPECT_EQ(default_zero->blocks_per_sm, explicit_zero->blocks_per_sm);
        EXPECT_EQ(default_zero->async_kv_preload, explicit_zero->async_kv_preload);
        std::cout << "SOFTCAP_MMA cap=" << cap << " rows=" << rows
                  << " per_sm=" << selected->blocks_per_sm << " kv_batch=" << selected->kv_batch
                  << " blocks=" << plan->blocks << " scratch=" << scratch << '\n';
      }
      auto paid = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = Allocate(scratch), .size = Bytes(scratch)});
      ASSERT_TRUE(paid) << paid.error().detail;
      if (scratch > 0) {
        auto short_pool = LaunchContext::Create(
            0, *execution_, stream_, {.base = Allocate(scratch), .size = Bytes(scratch - 1)});
        ASSERT_TRUE(short_pool);
        EXPECT_EQ(FailedCode(run(**short_pool)), KernelError::kRejected);
        EXPECT_FALSE((*short_pool)->faulted());
        (shape.vector ? underfunded_vector : underfunded_mma)++;
      }
      ASSERT_TRUE(run(**paid));
      EXPECT_LE((*paid)->scratch_peak().value(), scratch);
      const auto got = Download(node);
      if (cap == 0)
        uncapped.push_back(got);
      else
        EXPECT_NE(got, uncapped.at(case_index));
      ++case_index;
      ASSERT_TRUE(std::ranges::all_of(got, [](float x) { return std::isfinite(x); }));
      const auto expect_bits = [&] {
        const auto actual = Download(node);
        ASSERT_EQ(got.size(), actual.size());
        EXPECT_EQ(std::memcmp(got.data(), actual.data(), got.size() * sizeof(float)), 0);
      };
      // The separately invoked pinned case consumes the same actual paid
      // workspace; equality catches occupancy-dependent reduction changes.
      if (!shape.vector) {
        ASSERT_TRUE((*paid)->Run(Bytes(scratch), [=](ggml_backend_cuda_context& context) {
          kg::detail::FlashAttnMmaCaseGqa2(static_cast<int>(columns))(context, node);
        }));
        expect_bits();
      }
      std::vector<double> want(got.size());
      std::vector<double> logits(n(cells));
      for (std::int64_t seq = 0; seq < sequences; ++seq)
        for (std::int64_t h = 0; h < heads; ++h) {
          for (std::int64_t r = 0; r < rows; ++r) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::int64_t cell = 0; cell < cells; ++cell) {
              if (!std::isfinite(mask[n((seq * mask_rows + r) * cells + cell)])) {
                logits[n(cell)] = -std::numeric_limits<double>::infinity();
                continue;
              }
              double dot = 0;
              for (std::int64_t i = 0; i < d; ++i)
                dot += static_cast<double>(q[n(((seq * heads + h) * rows + r) * d + i)]) *
                       ggml_fp16_to_fp32(k[n(((seq * kv_heads + h / 2) * cells + cell) * d + i)]);
              dot *= scale;
              logits[n(cell)] = cap == 0 ? dot : cap * std::tanh(dot / cap);
              maximum = std::max(maximum, logits[n(cell)]);
            }
            double sum = 0;
            for (const double l : logits)
              if (std::isfinite(l)) sum += std::exp(l - maximum);
            for (std::int64_t cell = 0; cell < cells; ++cell) {
              if (!std::isfinite(logits[n(cell)])) continue;
              const double probability = std::exp(logits[n(cell)] - maximum) / sum;
              for (std::int64_t i = 0; i < d; ++i)
                want[n(((seq * rows + r) * heads + h) * d + i)] +=
                    probability *
                    ggml_fp16_to_fp32(v[n(((seq * kv_heads + h / 2) * cells + cell) * d + i)]);
            }
          }
        }
      ExpectNmse(got, want, kFlashAttnNmse, "softcap independent FP64 reference");
      ASSERT_TRUE(run(**paid));
      expect_bits();
      for (int fresh = 0; fresh < 2; ++fresh) {
        auto graph = (*paid)->Capture([&](LaunchContext& l) { return run(l); });
        ASSERT_TRUE(graph) << graph.error().detail;
        for (int replay = 0; replay < 2; ++replay) {
          ASSERT_TRUE((*paid)->Launch(*graph));
          expect_bits();
          EXPECT_LE((*paid)->scratch_peak().value(), scratch);
        }
      }
      if (sequences > 1) {
        tm->ne[1] = rows;
        EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
        tm->ne[1] = mask_rows;
        tm->ne[3] = 1;
        EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
        tm->ne[3] = sequences;
        EXPECT_FALSE(launch().faulted());
      }
    }
  }
  EXPECT_GT(underfunded_vector, 0);
  EXPECT_GT(underfunded_mma, 0);
}

TEST_F(GgmlExtOpsTest, Gemma2SoftcapRejectsInvalidParametersAndOtherAttentionContracts) {
  const auto build = [&](std::int64_t d, std::int64_t h, std::int64_t kv) {
    auto* q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, 1, h));
    auto* k = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, d, 256, kv));
    auto* v = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, d, 256, kv));
    auto* m = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 256, 4));
    auto* node = Place(ggml_flash_attn_ext(c(), q, k, v, m, 0.0625F, 0, 50));
    ggml_prec_set_acc(node, GGML_PREC_F32);
    return node;
  };
  auto* node = build(256, 8, 4);
  for (const float bad :
       {-1.0F, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    std::memcpy(node->op_params + 2, &bad, sizeof(bad));
    EXPECT_EQ(FailedCode(kg::PlanFlashAttnVec256(launch(), node)), KernelError::kRejected);
    EXPECT_EQ(FailedCode(kg::FlashAttnVec256(launch(), node)), KernelError::kRejected);
    EXPECT_EQ(FailedCode(kg::PlanFlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
    EXPECT_EQ(FailedCode(kg::FlashAttnMmaGqa2(launch(), node)), KernelError::kRejected);
    EXPECT_FALSE(launch().faulted());
  }
  const float cap = 50;
  std::memcpy(node->op_params + 2, &cap, sizeof(cap));
  for (const float bad : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
    std::memcpy(node->op_params, &bad, sizeof(bad));
    EXPECT_FALSE(kg::CheckFlashAttnVec256(node));
    EXPECT_FALSE(kg::CheckFlashAttnMmaGqa2(node));
  }
  const float scale = 0.0625F;
  std::memcpy(node->op_params, &scale, sizeof(scale));
  const float bias = 1;
  std::memcpy(node->op_params + 1, &bias, sizeof(bias));
  EXPECT_FALSE(kg::CheckFlashAttnVec256(node));
  EXPECT_FALSE(kg::CheckFlashAttnMmaGqa2(node));
  EXPECT_FALSE(kg::CheckFlashAttnVec(build(64, 8, 4)));
  EXPECT_FALSE(kg::CheckFlashAttnMma(build(256, 24, 2)));
  EXPECT_FALSE(kg::CheckFlashAttnMma(build(512, 64, 1)));
  EXPECT_FALSE(kg::CheckFlashAttnMma128(build(128, 4, 4)));
  EXPECT_FALSE(kg::CheckFlashAttnVec256(build(256, 8, 2)));
  EXPECT_FALSE(kg::CheckFlashAttnMmaGqa2(build(256, 8, 2)));
}

struct Attention {
  std::int64_t head{};
  std::int64_t heads{};
  std::int64_t kv_heads{};
  std::int64_t rows{};
  std::int64_t cells{};
  bool sinks{};
  std::int32_t n_kv_max{};  // > 0: at most this many unmasked cells per row
  const char* name{};
  int sparse_pattern = 0;  // 0: random, 1: identical lists, 2: disjoint lists
};

TEST_F(GgmlExtOpsTest, TensorCoreFlashAttentionMatchesTheReference) {
  // DeepSeek V4: D = 512, 64 query heads over one KV head that is both K and
  // V, with sinks; one row, a few, a chunk, and one row over a long
  // compressed cache whose mask leaves at most 256 cells (sparse). Qwen3.8:
  // D = 256, 24 heads over 2 KV heads.
  const std::array<Attention, 15> cases = {{
      {512, 64, 1, 1, 512, true, 0, "deepseek decode"},
      {512, 64, 1, 3, 512, true, 0, "deepseek 3 rows"},
      {512, 64, 1, 9, 768, true, 0, "deepseek 9 rows"},
      {512, 64, 1, 1, 4096, true, 256, "deepseek sparse"},
      {512, 64, 1, 8, 4096, true, 256, "deepseek sparse tile"},
      {512, 64, 1, 17, 4096, true, 256, "deepseek sparse partial and empty tail"},
      {256, 24, 2, 1, 512, false, 0, "qwen3.8 decode"},
      {256, 24, 2, 2, 512, false, 0, "qwen3.8 2 rows"},
      {256, 24, 2, 20, 1024, false, 0, "qwen3.8 20 rows"},
      {256, 24, 2, 1, 4096, false, 128, "D256 sparse decode"},
      {256, 24, 2, 17, 4096, false, 128, "D256 sparse partial tile"},
      {512, 64, 1, 13, 4096, true, 128, "D512 overlapping lists and finite bias", 1},
      {512, 64, 1, 17, 4096, true, 128, "D512 disjoint lists and dummy cells", 2},
      {256, 24, 2, 13, 4096, false, 128, "D256 overlapping lists and finite bias", 1},
      {256, 24, 2, 17, 4096, false, 128, "D256 disjoint lists and dummy cells", 2},
  }};
  const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
  for (const Attention& test : cases) {
    const bool shared_kv = test.kv_heads == 1;
    const std::vector<float> q = Normal(161, n(test.head * test.rows * test.heads));
    const std::vector<ggml_fp16_t> k16 =
        Halves(Normal(162, n(test.head * test.cells * test.kv_heads)));
    const std::vector<ggml_fp16_t> v16 =
        shared_kv ? k16 : Halves(Normal(163, n(test.head * test.cells * test.kv_heads)));
    // A causal mask over the cache's last rows, or for the sparse case 256
    // chosen cells per row.
    std::vector<float> mask(n(test.cells * test.rows), -std::numeric_limits<float>::infinity());
    std::mt19937 random(164);  // NOLINT(bugprone-random-generator-seed): reproducible
    for (std::int64_t r = 0; r < test.rows; ++r) {
      if (test.n_kv_max > 0) {
        if (test.sinks && test.rows > 1 && r == test.rows - 1) {
          continue;  // a tile with no visible cells must still produce zeros
        }
        std::vector<std::int64_t> chosen(n(test.cells));
        std::ranges::iota(chosen, 0);
        if (test.sparse_pattern == 0) {
          std::shuffle(chosen.begin(), chosen.end(), random);
        }
        const auto count = test.sparse_pattern == 0 ? test.n_kv_max : test.n_kv_max - (r % 17);
        for (std::int64_t i = 0; i < count; ++i) {
          const auto cell = test.sparse_pattern == 2 ? ((r % 8) * test.n_kv_max) + i : chosen[n(i)];
          mask[n((r * test.cells) + cell)] =
              test.sparse_pattern == 0 ? 0.0f : -static_cast<float>((r + i) % 7) / 8.0f;
        }
      } else {
        const std::int64_t visible = test.cells - test.rows + r + 1 - 37;
        for (std::int64_t cell = 0; cell < visible; ++cell) {
          mask[n((r * test.cells) + cell)] = 0.0f;
        }
      }
    }
    const std::vector<ggml_fp16_t> mask16 = Halves(mask);
    const std::vector<float> sinks = Normal(165, n(test.heads));
    ggml_tensor* tq =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, test.head, test.rows, test.heads, 1), q);
    ggml_tensor* tk =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, test.head, test.cells, test.kv_heads, 1), k16);
    ggml_tensor* tv =
        shared_kv
            ? tk
            : Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, test.head, test.cells, test.kv_heads, 1),
                    v16);
    ggml_tensor* tm =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, test.cells, test.rows, 1, 1), mask16);
    const float scale = 1.0f / std::sqrt(static_cast<float>(test.head));
    ggml_tensor* node = ggml_flash_attn_ext(c(), tq, tk, tv, tm, scale, 0.0f, 0.0f);
    if (test.sinks) {
      ggml_flash_attn_ext_add_sinks(
          node, Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, test.heads), sinks));
    }
    if (test.n_kv_max > 0) {
      ggml_flash_attn_ext_set_n_kv_max(node, test.n_kv_max);
    }
    Place(node);
    const std::array<ggml_tensor*, 1> nodes = {node};
    auto choices = kg::DeviceChoicesOf(launch());
    const auto primitive = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(primitive.has_value()) << test.name;
    ASSERT_EQ(primitive->steps.size(), 1U);
    EXPECT_EQ(primitive->steps[0].implementation, kg::kFlashAttnMmaName);
    choices.wide_sparse_attention = true;
    const auto wide = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(wide.has_value()) << test.name;
    ASSERT_EQ(wide->steps.size(), 1U);
    EXPECT_EQ(wide->steps[0].implementation, kg::kFlashAttnMmaWideName);
    const auto original = kg::PlanFlashAttnMma(launch(), node);
    ASSERT_TRUE(original.has_value()) << test.name;
    EXPECT_EQ(original->sparse, test.n_kv_max > 0 && test.head == 512) << test.name;
    if (original->sparse) {
      EXPECT_EQ(original->columns, 1) << test.name;
    }
    const auto plan = kg::PlanFlashAttnMma(launch(), node, true);
    ASSERT_TRUE(plan.has_value()) << test.name << ": " << plan.error().detail;
    EXPECT_EQ(plan->sparse, test.n_kv_max > 0) << test.name;
    EXPECT_EQ(plan->columns, test.n_kv_max > 0 ? (test.rows > 4 ? 8 : 1)
                             : test.rows <= 1  ? 1
                             : test.rows <= 2  ? 2
                             : test.rows <= 4  ? 4
                                               : 8)
        << test.name;
    launch().ResetScratchPeak();
    Launched(kg::FlashAttnMma(launch(), node, true), test.name);
    EXPECT_LE(launch().scratch_peak().value(), plan->scratch) << test.name;
    const auto got = Download(node);

    // FP64 reference over the F16 K and V; the output is [D, heads, rows].
    std::vector<double> want(n(test.head * test.heads * test.rows));
    const std::int64_t gqa = test.heads / test.kv_heads;
    std::vector<double> logits(n(test.cells));
    for (std::int64_t h = 0; h < test.heads; ++h) {
      const std::int64_t kh = h / gqa;
      for (std::int64_t r = 0; r < test.rows; ++r) {
        double max = test.sinks ? static_cast<double>(sinks[n(h)])
                                : -std::numeric_limits<double>::infinity();
        for (std::int64_t cell = 0; cell < test.cells; ++cell) {
          const float m = ggml_fp16_to_fp32(mask16[n((r * test.cells) + cell)]);
          if (std::isinf(m)) {
            logits[n(cell)] = -std::numeric_limits<double>::infinity();
            continue;
          }
          double dot = 0.0;
          for (std::int64_t d = 0; d < test.head; ++d) {
            dot += static_cast<double>(q[n((((h * test.rows) + r) * test.head) + d)]) *
                   ggml_fp16_to_fp32(k16[n((((kh * test.cells) + cell) * test.head) + d)]);
          }
          logits[n(cell)] = (dot * scale) + m;
          max = std::max(max, logits[n(cell)]);
        }
        double sum = test.sinks ? std::exp(static_cast<double>(sinks[n(h)]) - max) : 0.0;
        std::vector<double> out(n(test.head));
        for (std::int64_t cell = 0; cell < test.cells; ++cell) {
          if (std::isinf(logits[n(cell)])) {
            continue;
          }
          const double p = std::exp(logits[n(cell)] - max);
          sum += p;
          for (std::int64_t d = 0; d < test.head; ++d) {
            out[n(d)] +=
                p * ggml_fp16_to_fp32(v16[n((((kh * test.cells) + cell) * test.head) + d)]);
          }
        }
        for (std::int64_t d = 0; d < test.head; ++d) {
          want[n((((r * test.heads) + h) * test.head) + d)] = out[n(d)] / sum;
        }
      }
    }
    ExpectNmse(got, want, kFlashAttnNmse, test.name);
  }
}

// The ds4 HCA core plans its whole scratch, repeats its own output exactly
// (also after the ordinary sparse attention reuses the same pool offsets),
// and refuses a capture: its chunk's first position is a launch parameter
// a replay would keep.
TEST_F(GgmlExtOpsTest, Ds4HcaPlansItsWholeScratchRefusesCaptureAndPreservesOwnRepeats) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "the measured ds4 HCA arm is GB10 only";
  const auto exact = [](const std::vector<float>& got, const std::vector<float>& want) {
    ASSERT_EQ(got.size(), want.size());
    EXPECT_EQ(std::memcmp(got.data(), want.data(), got.size() * sizeof(float)), 0);
  };
  constexpr std::int64_t raw_cells = 256;
  constexpr std::int64_t compressed = 256;
  for (const auto [first, rows] :
       std::array<std::pair<std::uint32_t, std::uint32_t>, 3>{{{0, 3}, {127, 5}, {511, 129}}}) {
    auto queries = Normal(first + 131, static_cast<std::size_t>(rows) * 64 * 512, 0.5F);
    auto* packed_q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 512, 64, rows), queries);
    auto* q = ggml_permute(c(), packed_q, 0, 2, 1, 3);
    auto* kv = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 512, raw_cells + compressed),
                     Halves(Normal(first + 137, (raw_cells + compressed) * 512, 0.5F)));
    std::vector<float> window(static_cast<std::size_t>(raw_cells) * rows,
                              -std::numeric_limits<float>::infinity());
    std::vector<std::int32_t> visible(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto position = static_cast<std::int64_t>(first) + row;
      for (auto p = std::max<std::int64_t>(0, position - 127); p <= position; ++p)
        window[(static_cast<std::size_t>(row) * raw_cells) +
               static_cast<std::size_t>(p % raw_cells)] = 0;
      visible[row] = static_cast<std::int32_t>((position + 1) / 128);
    }
    auto* mask = Place(kg::Dsv4SparseMask(
        c(), Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, raw_cells, rows), Halves(window)),
        nullptr, Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows), visible), raw_cells,
        compressed));
    Launched(kg::RunDsv4SparseMask(launch(), mask), "ds4 HCA causal mask");
    auto* sinks = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 64), Normal(139, 64));
    const auto make = [&] {
      auto* node = Place(ggml_flash_attn_ext(c(), q, kv, kv, mask, 0.04419417306780815F, 0, 0));
      ggml_flash_attn_ext_add_sinks(node, sinks);
      EXPECT_TRUE(ggml_prec_set_acc(node, GGML_PREC_F32));
      ggml_flash_attn_ext_set_n_kv_max(node, 128 + compressed);
      kg::SetFlashAttnSparseAny(node);
      kg::MarkDsv4HcaTokentile(node, first);
      return node;
    };
    auto* node = make();
    auto* control = make();
    const auto scratch = kg::PlanDsv4HcaTokentile(launch(), node);
    ASSERT_TRUE(scratch.has_value()) << (scratch ? "" : scratch.error().detail);
    const kg::GraphPlan plan{.steps = {{.operation = jitllm::execution::Operation::kFlashAttn,
                                        .implementation = kg::kDsv4HcaTokentileName,
                                        .nodes = {node},
                                        .lane = 0}},
                             .regions = {}};
    const auto planned = kg::PlanScratch(launch(), plan);
    ASSERT_TRUE(planned.has_value());
    EXPECT_EQ(*planned, *scratch);
    auto insufficient = LaunchContext::Create(
        0, *execution_, stream_, {.base = Allocate(*scratch), .size = Bytes(*scratch - 1)});
    ASSERT_TRUE(insufficient.has_value());
    EXPECT_EQ(FailedCode(kg::Dsv4HcaTokentile(**insufficient, node)), KernelError::kRejected);
    insufficient->reset();
    launch().ResetScratchPeak();
    Launched(kg::Dsv4HcaTokentile(launch(), node), "ds4 HCA warmup");
    EXPECT_LE(launch().scratch_peak().value(), *scratch);
    const auto original = Download(node);
    Launched(kg::Dsv4HcaTokentile(launch(), node), "ds4 HCA repeat");
    exact(Download(node), original);
    Launched(kg::FlashAttnMma(launch(), control), "ds4 HCA native approximation");
    const auto native = Download(control);
    ExpectNmse(original, std::vector<double>(native.begin(), native.end()), kFlashAttnNmse,
               "ds4 HCA versus native approximation");
    // A capture is refused, with the context as it was.
    const auto graph = launch().Capture(
        [&](LaunchContext& context) { return kg::Dsv4HcaTokentile(context, node); });
    EXPECT_EQ(FailedCode(graph), KernelError::kRejected);
    for (const float scale : {1.0F, 0.5F}) {
      auto changed = queries;
      for (auto& value : changed) value *= scale;
      ASSERT_EQ(cudaMemcpy(packed_q->data, changed.data(), changed.size() * sizeof(float),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      Launched(kg::Dsv4HcaTokentile(launch(), node), "ds4 HCA changed query");
      const auto want = Download(node);
      Launched(kg::FlashAttnMma(launch(), control), "native changed query");
      const auto want_control = Download(control);
      for (int repeat = 0; repeat < 2; ++repeat) {
        // Ordinary sparse preparation overwrites the same pool offsets.
        Launched(kg::Dsv4HcaTokentile(launch(), node), "ds4 HCA repeat after pool reuse");
        Launched(kg::FlashAttnMma(launch(), control), "native repeat");
        exact(Download(node), want);
        exact(Download(control), want_control);
      }
    }
    auto choices = kg::DeviceChoicesOf(launch());
    const std::array<ggml_tensor*, 1> nodes = {node};
    const auto ordinary = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(ordinary.has_value());
    EXPECT_EQ(ordinary->steps.front().implementation, kg::kFlashAttnMmaName);
    choices.ds4_hca = true;
    const auto fallback = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(fallback.has_value());
    EXPECT_EQ(fallback->steps.front().implementation, kg::kFlashAttnMmaName);
    choices.ds4_hca_fits = [](const ggml_tensor*) { return true; };  // bounded generic control
    const auto selected = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->steps.front().implementation, kg::kDsv4HcaTokentileName);
  }
}

// Qwen-Image-2.1's denoiser attention on GGML's kernels: D = 128, one query
// head per KV head, no mask, cells not a multiple of 256 (the last KV tile
// bounds-checked). The A/B its native kernel won (docs/experiments/
// qwen-image-native); kept so that the comparison reruns.
TEST_F(GgmlExtOpsTest, TensorCoreFlashAttentionAt128WithoutMaskMatchesTheReference) {
  struct Case {
    std::int64_t heads, rows, cells;
    int columns;
  };
  const std::array<Case, 5> cases = {
      {{4, 1, 77, 8}, {4, 12, 300, 16}, {3, 20, 1000, 32}, {2, 100, 257, 64}, {2, 300, 4121, 64}}};
  const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
  constexpr std::int64_t kHead = 128;
  for (const Case& test : cases) {
    const std::string name =
        std::format("{} heads, {} rows, {} cells", test.heads, test.rows, test.cells);
    const std::vector<float> q = Normal(171, n(kHead * test.rows * test.heads));
    const std::vector<ggml_fp16_t> k16 = Halves(Normal(172, n(kHead * test.cells * test.heads)));
    const std::vector<ggml_fp16_t> v16 = Halves(Normal(173, n(kHead * test.cells * test.heads)));
    ggml_tensor* tq =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kHead, test.rows, test.heads, 1), q);
    ggml_tensor* tk =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, kHead, test.cells, test.heads, 1), k16);
    ggml_tensor* tv =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, kHead, test.cells, test.heads, 1), v16);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHead));
    ggml_tensor* node = Place(ggml_flash_attn_ext(c(), tq, tk, tv, nullptr, scale, 0.0f, 0.0f));
    ASSERT_TRUE(kg::CheckFlashAttnMma128(node).has_value()) << name;
    const auto plan = kg::PlanFlashAttnMma128(launch(), node);
    ASSERT_TRUE(plan.has_value()) << name << ": " << plan.error().detail;
    EXPECT_EQ(plan->columns, test.columns) << name;
    launch().ResetScratchPeak();
    Launched(kg::FlashAttnMma128(launch(), node), name);
    EXPECT_LE(launch().scratch_peak().value(), plan->scratch) << name;
    const auto got = Download(node);
    std::vector<double> want(n(kHead * test.heads * test.rows));
    std::vector<double> logits(n(test.cells));
    for (std::int64_t h = 0; h < test.heads; ++h) {
      for (std::int64_t r = 0; r < test.rows; ++r) {
        double max = -std::numeric_limits<double>::infinity();
        for (std::int64_t cell = 0; cell < test.cells; ++cell) {
          double dot = 0.0;
          for (std::int64_t d = 0; d < kHead; ++d) {
            dot += static_cast<double>(q[n((((h * test.rows) + r) * kHead) + d)]) *
                   ggml_fp16_to_fp32(k16[n((((h * test.cells) + cell) * kHead) + d)]);
          }
          logits[n(cell)] = dot * scale;
          max = std::max(max, logits[n(cell)]);
        }
        double sum = 0.0;
        std::vector<double> out(n(kHead));
        for (std::int64_t cell = 0; cell < test.cells; ++cell) {
          const double p = std::exp(logits[n(cell)] - max);
          sum += p;
          for (std::int64_t d = 0; d < kHead; ++d) {
            out[n(d)] += p * ggml_fp16_to_fp32(v16[n((((h * test.cells) + cell) * kHead) + d)]);
          }
        }
        for (std::int64_t d = 0; d < kHead; ++d) {
          want[n((((r * test.heads) + h) * kHead) + d)] = out[n(d)] / sum;
        }
      }
    }
    ExpectNmse(got, want, kFlashAttnNmse, name);
  }
  // The checks: a mask, grouped heads, sinks or another head size are the
  // other kernels' (or none).
  const auto build = [this](std::int64_t head, std::int64_t heads, std::int64_t kv_heads,
                            bool mask) {
    ggml_tensor* q = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, head, 2, heads, 1));
    ggml_tensor* k = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, head, 300, kv_heads, 1));
    ggml_tensor* m = mask ? Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, 300, 2, 1, 1)) : nullptr;
    return Place(ggml_flash_attn_ext(c(), q, k, k, m, 0.1f, 0.0f, 0.0f));
  };
  EXPECT_TRUE(kg::CheckFlashAttnMma128(build(128, 4, 4, false)).has_value());
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma128(build(128, 4, 4, true))), KernelError::kRejected);
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma128(build(128, 8, 2, false))), KernelError::kRejected);
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma128(build(256, 4, 4, false))), KernelError::kRejected);
  ggml_tensor* sinks = build(128, 4, 4, false);
  ggml_flash_attn_ext_add_sinks(sinks, Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 4)));
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma128(sinks)), KernelError::kRejected);
}

TEST_F(GgmlExtOpsTest, FlashAttentionChecksRefuseWhatTheKernelsWouldNotTake) {
  const auto build = [this](std::int64_t head, std::int64_t heads, std::int64_t kv_heads,
                            std::int64_t cells, float max_bias) {
    ggml_tensor* q = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F32, head, 2, heads, 1));
    ggml_tensor* k = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, head, cells, kv_heads, 1));
    ggml_tensor* mask = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, cells, 2, 1, 1));
    return Place(ggml_flash_attn_ext(c(), q, k, k, mask, 0.1f, max_bias, 0.0f));
  };
  EXPECT_TRUE(kg::CheckFlashAttnMma(build(512, 64, 1, 256, 0.0f)).has_value());
  EXPECT_TRUE(kg::CheckFlashAttnMma(build(256, 24, 2, 256, 0.0f)).has_value());
  // Head sizes without these kernels, ALiBi, unpadded cells, and a query
  // group of 4 or fewer.
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma(build(128, 16, 2, 256, 0.0f))),
            KernelError::kRejected);
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma(build(512, 64, 1, 256, 1.0f))),
            KernelError::kRejected);
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma(build(512, 64, 1, 300, 0.0f))),
            KernelError::kRejected);
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma(build(256, 8, 2, 256, 0.0f))), KernelError::kRejected);
  // Sinks where a group of 8 would read past the last head's: Qwen3.8's 12
  // query heads per KV head. DeepSeek's 64 take them.
  ggml_tensor* qwen = build(256, 24, 2, 256, 0.0f);
  ggml_flash_attn_ext_add_sinks(qwen, Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 24)));
  EXPECT_EQ(FailedCode(kg::CheckFlashAttnMma(qwen)), KernelError::kRejected);
  ggml_tensor* deepseek = build(512, 64, 1, 256, 0.0f);
  ggml_flash_attn_ext_add_sinks(deepseek, Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 64)));
  EXPECT_TRUE(kg::CheckFlashAttnMma(deepseek).has_value());
}

// ---- Reads, writes and scratch stay within bounds ----

// The over-read probe, as for the EXL3 plan's operations
// (ggml_exl3_ops_test.cc): every operand, and the pool workspace sized
// exactly to the operation's plan, in device VMM, first ending flush against
// an unmapped granule, then starting flush after one. Each operation must
// complete without a fault, draw no more scratch than planned (the pool
// would end the process), and write what it writes in cudaMalloc memory.
TEST_F(GgmlExtOpsTest, OperationsReadWriteAndDrawOnlyWhatTheyShould) {
  enum class Where : std::uint8_t { kMalloc, kEnd, kStart };
  auto memory = std::move(jitllm::providers::cuda::OpenDeviceMemory(0).value());
  std::size_t device_class = 0;
  for (std::size_t i = 0; i < memory->Classes().size(); ++i) {
    if (memory->Classes()[i].kind == jitllm::providers::BackingKind::kDevice) {
      device_class = i;
    }
  }
  const std::uint64_t granule = memory->Granularity().value();
  struct Mapped {
    jitllm::providers::ReservationId reservation;
    jitllm::providers::BackingId backing;
    std::uint64_t offset;
    std::uint64_t size;
  };
  std::vector<Mapped> mapped;
  const auto allocate = [&](Where where, std::uint64_t bytes) -> std::uint64_t {
    if (where == Where::kMalloc) {
      return Allocate(bytes);
    }
    const std::uint64_t size = (bytes + granule - 1) / granule * granule;
    const auto reservation = memory->Reserve(Bytes(size + granule)).value();
    const auto backing = memory->Create(device_class, Bytes(size)).value();
    const std::uint64_t offset = where == Where::kStart ? granule : 0;
    EXPECT_TRUE(memory->Map(reservation, Bytes(offset), backing).has_value());
    EXPECT_TRUE(memory
                    ->SetAccess(reservation, Bytes(offset), Bytes(size),
                                jitllm::providers::Access::kReadWrite)
                    .has_value());
    mapped.push_back({reservation, backing, offset, size});
    const std::uint64_t base = memory->RangeOf(reservation).value().base + offset;
    return where == Where::kEnd ? base + size - bytes : base;
  };
  const auto drained = [&]() -> bool {
    const auto fence = execution_->Record(stream_);
    if (!fence) {
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < deadline) {
      const auto state = execution_->Query(*fence);
      if (!state) {
        return false;
      }
      if (*state == FenceState::kComplete) {
        return execution_->Release(*fence).has_value();
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
  };

  using PlaceFn = std::function<ggml_tensor*(ggml_tensor*)>;
  using Plan = std::function<std::expected<std::uint64_t, KernelFailure>(const LaunchContext&)>;
  using Run = std::function<std::expected<void, KernelFailure>(LaunchContext&)>;
  struct Built {
    std::vector<ggml_tensor*> outputs;
    Plan plan;
    Run run;
  };
  const auto bytes_of = [](const auto& values) {
    const auto view = std::as_bytes(std::span(values));
    return std::vector<std::byte>(view.begin(), view.end());
  };
  const auto check = [&](const std::string& what,
                         const std::vector<std::vector<std::byte>>& operands,
                         const std::function<Built(ggml_context*, const PlaceFn&)>& build) {
    SCOPED_TRACE(what);
    std::vector<std::vector<std::byte>> first;
    for (const Where where : {Where::kMalloc, Where::kEnd, Where::kStart}) {
      auto arena = TensorArena::Create(64).value();
      std::size_t placed = 0;
      const PlaceFn place = [&](ggml_tensor* tensor) {
        const std::size_t size = ggml_nbytes(tensor);
        const std::uint64_t address = allocate(where, size);
        TensorArena::Bind(tensor, address);
        void* data = reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
        if (placed < operands.size() && !operands[placed].empty()) {
          EXPECT_EQ(operands[placed].size(), size) << "operand " << placed;
          EXPECT_EQ(cudaMemcpy(data, operands[placed].data(),
                               std::min(size, operands[placed].size()), cudaMemcpyHostToDevice),
                    cudaSuccess);
        } else {
          EXPECT_EQ(cudaMemset(data, 0xA5, size), cudaSuccess);
        }
        ++placed;
        return tensor;
      };
      const Built built = build(arena.context(), place);
      ASSERT_EQ(placed, operands.size());
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      std::uint64_t scratch = 0;
      {
        auto planner =
            LaunchContext::Create(0, *execution_, stream_, {.base = 0, .size = Bytes(0)});
        ASSERT_TRUE(planner.has_value()) << planner.error().detail;
        const auto planned = built.plan(**planner);
        ASSERT_TRUE(planned.has_value()) << planned.error().detail;
        scratch = *planned;
      }
      // The workspace's base is 256-byte aligned (launch.h), so its size is
      // the plan's rounded up to the pool's blocks.
      scratch = (scratch + 255) / 256 * 256;
      const std::uint64_t pool = scratch == 0 ? 0 : allocate(where, scratch);
      auto launch =
          LaunchContext::Create(0, *execution_, stream_, {.base = pool, .size = Bytes(scratch)});
      ASSERT_TRUE(launch.has_value()) << launch.error().detail;
      const auto ran = built.run(**launch);
      ASSERT_TRUE(ran.has_value()) << ran.error().detail;
      ASSERT_TRUE(drained()) << "a fault with operands at placement " << static_cast<int>(where);
      launch->reset();
      for (std::size_t o = 0; o < built.outputs.size(); ++o) {
        std::vector<std::byte> out(ggml_nbytes(built.outputs[o]));
        ASSERT_EQ(
            cudaMemcpy(out.data(), built.outputs[o]->data, out.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
        if (where == Where::kMalloc) {
          first.push_back(std::move(out));
        } else {
          EXPECT_TRUE(out == first[o])
              << "output " << o << " at placement " << static_cast<int>(where);
        }
      }
    }
  };
  const Plan no_scratch = [](const LaunchContext&) -> std::expected<std::uint64_t, KernelFailure> {
    return 0;
  };

  // The quantized products at both families, dense and over experts: the
  // weights' last row read in whole 512-element steps and no further.
  for (const auto& [type, columns] : std::vector<std::pair<ggml_type, std::int64_t>>{
           {GGML_TYPE_Q8_0, 3}, {GGML_TYPE_Q5_K, 40}, {GGML_TYPE_MXFP4, 40}}) {
    const Quantized w = Quantize(type, 2048, 96, 201);
    check(std::string("mul_mat ") + ggml_type_name(type) + " x " + std::to_string(columns),
          {bytes_of(w.bytes), bytes_of(Normal(202, static_cast<std::size_t>(2048 * columns))), {}},
          [&](ggml_context* ctx, const PlaceFn& place) {
            ggml_tensor* tw = place(ggml_new_tensor_2d(ctx, type, 2048, 96));
            ggml_tensor* tx = place(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, columns));
            ggml_tensor* node = place(ggml_mul_mat(ctx, tw, tx));
            const bool vector = columns <= 8;
            return Built{.outputs = {node},
                         .plan =
                             [node, vector](const LaunchContext& l) {
                               return vector ? kg::PlanMulMatVecQ(l, node)
                                             : kg::PlanMulMatQ(l, node);
                             },
                         .run =
                             [node, vector](LaunchContext& l) {
                               return vector ? kg::MulMatVecQ(l, node) : kg::MulMatQ(l, node);
                             }};
          });
  }
  for (const std::int64_t tokens : {2, 24}) {
    const Quantized w = Quantize(GGML_TYPE_IQ2_XS, 2048, std::int64_t{32} * 8, 203);
    std::vector<std::int32_t> ids;
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int32_t e = 0; e < 6; ++e) {
        ids.push_back(static_cast<std::int32_t>((t + (std::int64_t{e} * 3)) % 8));  // 7 among them
      }
    }
    check("mul_mat_id IQ2_XS x " + std::to_string(tokens),
          {bytes_of(w.bytes),
           bytes_of(Normal(204, static_cast<std::size_t>(2048 * tokens))),
           bytes_of(ids),
           {}},
          [&](ggml_context* ctx, const PlaceFn& place) {
            ggml_tensor* tw = place(ggml_new_tensor_3d(ctx, GGML_TYPE_IQ2_XS, 2048, 32, 8));
            ggml_tensor* tx = place(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 1, tokens));
            ggml_tensor* tids = place(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, tokens));
            ggml_tensor* node = place(ggml_mul_mat_id(ctx, tw, tx, tids));
            const bool vector = tokens <= 8;
            return Built{.outputs = {node},
                         .plan =
                             [node, vector](const LaunchContext& l) {
                               return vector ? kg::PlanMulMatVecQ(l, node)
                                             : kg::PlanMulMatQ(l, node);
                             },
                         .run =
                             [node, vector](LaunchContext& l) {
                               return vector ? kg::MulMatVecQ(l, node) : kg::MulMatQ(l, node);
                             }};
          });
  }

  // Expert tails: empty leading/middle experts, partial token/output-row
  // tiles, and ordinary/compact single/paired preparation. T257 keeps the
  // original stream-K grid fully occupied on GB10, so no later fixup
  // allocation can hide the final quantized-column over-read. Include raw
  // FP4's ordinary preparation; compact/pair contracts remain non-FP4.
  for (const auto type :
       {GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
    constexpr std::int64_t k = 512;
    constexpr std::int64_t out = 129;
    constexpr std::int64_t experts = 64;
    constexpr std::int64_t tokens = 257;
    const auto a = Quantize(type, k, out * experts, 221);
    const auto b = Quantize(type, k, out * experts, 222);
    std::vector<std::int32_t> ids;
    for (std::int64_t t = 0; t < tokens; ++t) {
      ids.push_back(63);
      ids.push_back(static_cast<std::int32_t>(17 + (t % 8)));
    }
    const bool fp4 = type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4;
    for (const int mode : {0, 1, 2, 3}) {
      if (fp4 && mode != 0) {
        continue;
      }
      const bool paired = mode % 2 != 0;
      const bool compact = mode >= 2;
      check(std::string(compact ? "compact experts " : "ordinary experts ") + ggml_type_name(type) +
                (paired ? " pair" : " one"),
            {bytes_of(a.bytes),
             bytes_of(b.bytes),
             bytes_of(Normal(223, static_cast<std::size_t>(k * tokens))),
             bytes_of(ids),
             {},
             {}},
            [&](ggml_context* ctx, const PlaceFn& place) {
              auto* wa = place(ggml_new_tensor_3d(ctx, type, k, out, experts));
              auto* wb = place(ggml_new_tensor_3d(ctx, type, k, out, experts));
              auto* x = place(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, tokens));
              auto* routes = place(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, tokens));
              auto* first = place(ggml_mul_mat_id(ctx, wa, x, routes));
              auto* second = place(ggml_mul_mat_id(ctx, wb, x, routes));
              return Built{.outputs = paired ? std::vector{first, second} : std::vector{first},
                           .plan =
                               [first, second, paired, compact](const LaunchContext& l) {
                                 if (paired) {
                                   return kg::PlanMulMatIdQPair(l, first, second, compact);
                                 }
                                 return compact ? kg::PlanMulMatIdQCompact(l, first)
                                                : kg::PlanMulMatQ(l, first);
                               },
                           .run =
                               [first, second, paired, compact](LaunchContext& l) {
                                 if (paired) {
                                   return kg::MulMatIdQPair(l, first, second, compact);
                                 }
                                 return compact ? kg::MulMatIdQCompact(l, first)
                                                : kg::MulMatQ(l, first);
                               }};
            });
    }
  }

  // Flash attention: DeepSeek's decode, its sparse gather and a chunk;
  // Qwen3.8's rows.
  struct Fa {
    std::int64_t head, heads, kv_heads, rows, cells;
    std::int32_t n_kv_max;
  };
  for (const Fa& fa :
       {Fa{512, 64, 1, 1, 256, 0}, Fa{512, 64, 1, 1, 4096, 128}, Fa{512, 64, 1, 17, 4096, 128},
        Fa{256, 24, 2, 17, 4096, 128}, Fa{512, 64, 1, 9, 512, 0}, Fa{256, 24, 2, 5, 256, 0}}) {
    const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
    std::vector<float> mask(n(fa.cells * fa.rows), -std::numeric_limits<float>::infinity());
    for (std::int64_t r = 0; r < fa.rows; ++r) {
      const std::int64_t visible = fa.n_kv_max > 0 ? fa.n_kv_max : fa.cells - fa.rows + r + 1;
      for (std::int64_t cell = 0; cell < visible; ++cell) {
        mask[n((r * fa.cells) + (fa.n_kv_max > 0 ? cell * 31 : cell))] = 0.0f;
      }
    }
    check("flash_attn D " + std::to_string(fa.head) + " x " + std::to_string(fa.rows) +
              (fa.n_kv_max > 0 ? " sparse" : ""),
          {bytes_of(Normal(205, n(fa.head * fa.rows * fa.heads))),
           bytes_of(Halves(Normal(206, n(fa.head * fa.cells * fa.kv_heads)))),
           bytes_of(Halves(Normal(207, n(fa.head * fa.cells * fa.kv_heads)))),
           bytes_of(Halves(mask)),
           bytes_of(Normal(208, n(fa.heads))),
           {}},
          [&](ggml_context* ctx, const PlaceFn& place) {
            ggml_tensor* q =
                place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, fa.head, fa.rows, fa.heads, 1));
            ggml_tensor* k =
                place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fa.head, fa.cells, fa.kv_heads, 1));
            ggml_tensor* v =
                place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fa.head, fa.cells, fa.kv_heads, 1));
            ggml_tensor* m = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, fa.cells, fa.rows, 1, 1));
            ggml_tensor* sinks = place(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, fa.heads));
            ggml_tensor* node = ggml_flash_attn_ext(ctx, q, k, v, m, 0.05f, 0.0f, 0.0f);
            // Sinks only where the heads per KV head are whole groups of 8
            // (validate_ext.h): DeepSeek's.
            if ((fa.heads / fa.kv_heads) % 8 == 0) {
              ggml_flash_attn_ext_add_sinks(node, sinks);
            }
            if (fa.n_kv_max > 0) {
              ggml_flash_attn_ext_set_n_kv_max(node, fa.n_kv_max);
            }
            place(node);
            return Built{
                .outputs = {node},
                .plan =
                    [node](const LaunchContext& l) -> std::expected<std::uint64_t, KernelFailure> {
                  auto plan = kg::PlanFlashAttnMma(l, node, true);
                  if (!plan) {
                    return std::unexpected(plan.error());
                  }
                  return plan->scratch;
                },
                .run = [node](LaunchContext& l) { return kg::FlashAttnMma(l, node, true); }};
          });
  }

  // The indexer's scores and top-k (the radix select and the bitonic
  // argsort), and a dequantizing gather of the table's last rows.
  check("lightning_indexer",
        {bytes_of(Normal(209, std::size_t{128} * 64 * 3)),
         bytes_of(Halves(Normal(210, std::size_t{128} * 1000))),
         bytes_of(Normal(211, std::size_t{64} * 3)),
         bytes_of(Halves(std::vector<float>(std::size_t{1000} * 3, 0.0f))),
         {}},
        [&](ggml_context* ctx, const PlaceFn& place) {
          ggml_tensor* q = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 64, 3, 1));
          ggml_tensor* k = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 128, 1, 1000, 1));
          ggml_tensor* w = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 3, 1, 1));
          ggml_tensor* m = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1000, 3, 1, 1));
          ggml_tensor* node = place(ggml_lightning_indexer(ctx, q, k, w, m));
          return Built{.outputs = {node}, .plan = no_scratch, .run = [node](LaunchContext& l) {
                         return kg::LightningIndexer(l, node);
                       }};
        });
  for (const std::int64_t cells : {700, 5000}) {
    std::vector<float> scores(static_cast<std::size_t>(cells * 2));
    // Distinct: one answer. std::ranges::iota needs an incrementable type.
    std::iota(scores.begin(), scores.end(), 0.0f);  // NOLINT(modernize-use-ranges)
    check("top_k of " + std::to_string(cells), {bytes_of(scores), {}},
          [&](ggml_context* ctx, const PlaceFn& place) {
            ggml_tensor* in = place(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cells, 2));
            ggml_tensor* node = place(ggml_top_k(ctx, in, 512));
            return Built{.outputs = {},  // the radix select's order is not fixed
                         .plan = [node](const LaunchContext& l) { return kg::PlanTopK(l, node); },
                         .run = [node](LaunchContext& l) { return kg::TopK(l, node); }};
          });
  }
  {
    const Quantized table = Quantize(GGML_TYPE_Q5_K, 4096, 16, 212);
    const std::vector<std::int32_t> rows = {15, 0, 15, 7};
    check("get_rows Q5_K", {bytes_of(table.bytes), bytes_of(rows), {}},
          [&](ggml_context* ctx, const PlaceFn& place) {
            ggml_tensor* t = place(ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_K, 4096, 16));
            ggml_tensor* ids = place(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4));
            ggml_tensor* node = place(ggml_get_rows(ctx, t, ids));
            return Built{.outputs = {node}, .plan = no_scratch, .run = [node](LaunchContext& l) {
                           return kg::GetRowsExt(l, node);
                         }};
          });
  }

  // Qwen3.8's linear attention.
  check("ssm_conv",
        {bytes_of(Normal(213, std::size_t{3 + 5} * 1024)),
         bytes_of(Normal(214, std::size_t{4} * 1024)),
         {}},
        [&](ggml_context* ctx, const PlaceFn& place) {
          ggml_tensor* x = place(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 + 5, 1024, 1));
          ggml_tensor* w = place(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1024));
          ggml_tensor* node = place(ggml_ssm_conv(ctx, x, w));
          return Built{.outputs = {node}, .plan = no_scratch, .run = [node](LaunchContext& l) {
                         return kg::SsmConv(l, node);
                       }};
        });
  check("gated_delta_net",
        {bytes_of(Normal(215, std::size_t{128} * 16 * 3)),
         bytes_of(Normal(216, std::size_t{128} * 16 * 3)),
         bytes_of(Normal(217, std::size_t{128} * 48 * 3)),
         bytes_of(Uniform(218, std::size_t{48} * 3, -1.0f, -0.1f)),
         bytes_of(Uniform(219, std::size_t{48} * 3, 0.0f, 1.0f)),
         bytes_of(Normal(220, std::size_t{128} * 128 * 48, 0.1f)),
         {}},
        [&](ggml_context* ctx, const PlaceFn& place) {
          ggml_tensor* q = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, 3, 1));
          ggml_tensor* k = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, 3, 1));
          ggml_tensor* v = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, 3, 1));
          ggml_tensor* g = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, 3, 1));
          ggml_tensor* b = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, 3, 1));
          ggml_tensor* s = place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 48, 1));
          ggml_tensor* node = place(ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1));
          return Built{.outputs = {node}, .plan = no_scratch, .run = [node](LaunchContext& l) {
                         return kg::GatedDeltaNet(l, node);
                       }};
        });

  Finish();
  for (const Mapped& m : mapped) {
    EXPECT_TRUE(memory->Unmap(m.reservation, Bytes(m.offset), Bytes(m.size)).has_value());
    EXPECT_TRUE(memory->Release(m.backing).has_value());
    EXPECT_TRUE(memory->Free(m.reservation).has_value());
  }
}

// ---- The registry ----

TEST_F(GgmlExtOpsTest, TheRegistryDeclaresAndBindsEveryNewImplementation) {
  using jitllm::execution::Operation;
  const std::vector<jitllm::execution::Implementation> declared = kg::Implementations();
  const auto registry = jitllm::execution::Registry::Create(declared).value();
  const std::array<std::pair<const char*, Operation>, 33> expected = {{
      {"ggml.mul_mat.mmvq", Operation::kMatMul},
      {"ggml.mul_mat.mmq", Operation::kMatMul},
      {"ggml.mul_mat.fwht", Operation::kMatMul},
      {"ggml.mul_mat_id.mmvq", Operation::kMulMatId},
      {"ggml.mul_mat_id.mmq", Operation::kMulMatId},
      {"jitllm.mul_mat_id.mmq_pair", Operation::kMulMatId},
      {"jitllm.mul_mat_id.mmq_compact", Operation::kMulMatId},
      {"jitllm.mul_mat_id.mmq_pair_compact", Operation::kMulMatId},
      {"jitllm.mul_mat_id.q2_d2r", Operation::kMulMatId},
      {"ggml.sub", Operation::kSub},
      {"ggml.div", Operation::kDiv},
      {"ggml.scale", Operation::kScale},
      {"ggml.unary", Operation::kUnary},
      {"ggml.clamp", Operation::kClamp},
      {"ggml.fill", Operation::kFill},
      {"ggml.repeat", Operation::kRepeat},
      {"ggml.concat", Operation::kConcat},
      {"ggml.sum_rows", Operation::kSumRows},
      {"ggml.argsort.bitonic", Operation::kArgsort},
      {"ggml.top_k.radix", Operation::kTopK},
      {"ggml.swiglu_clamp", Operation::kSwiGluClamp},
      {"ggml.rope.ext", Operation::kRope},
      {"ggml.get_rows.ext", Operation::kGetRows},
      {"ggml.set_rows.ext", Operation::kSetRows},
      {"ggml.ssm_conv", Operation::kSsmConv},
      {"ggml.gated_delta_net", Operation::kGatedDeltaNet},
      {"ggml.lightning_indexer.wmma", Operation::kLightningIndexer},
      {"ggml.dsv4_hc_comb", Operation::kHcComb},
      {"ggml.dsv4_hc_pre", Operation::kHcPre},
      {"ggml.dsv4_hc_post", Operation::kHcPost},
      {"ggml.flash_attn_ext.mma", Operation::kFlashAttn},
      {"jitllm.dsv4.hca_tokentile", Operation::kFlashAttn},
      {"ggml.flash_attn_ext.mma_d128", Operation::kFlashAttn},
  }};
  for (const auto& [name, operation] : expected) {
    const std::size_t index = registry.Find(name).value_or(registry.size());
    ASSERT_LT(index, registry.size()) << name;
    EXPECT_EQ(registry.at(index).operation, operation) << name;
    const auto kernel = kg::Kernel::Bind(registry.at(index));
    ASSERT_TRUE(kernel.has_value()) << name;
    EXPECT_EQ(kernel->name(), name);
    EXPECT_EQ(kernel->arity(), std::string_view(name) == kg::kMulMatIdQPair ||
                                       std::string_view(name) == kg::kMulMatIdQPairCompact
                                   ? 2U
                                   : 1U)
        << name;
  }
  // A bound kernel checks and runs its node: scale through the registry.
  const auto scale = kg::Kernel::Bind(declared[registry.Find("ggml.scale").value_or(0)]).value();
  const std::vector<float> x = Normal(171, 64);
  ggml_tensor* in = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 64), x);
  ggml_tensor* node = Place(ggml_scale(c(), in, 2.0f));
  const std::array<ggml_tensor*, 1> nodes = {node};
  Launched(scale.Run(launch(), nodes), "ggml.scale through the registry");
  const auto got = Download(node);
  for (std::size_t i = 0; i < x.size(); ++i) {
    ASSERT_EQ(got[i], 2.0f * x[i]);
  }
  // A node of another operation is refused before any launch.
  const std::array<const ggml_tensor*, 1> wrong = {Place(ggml_relu(c(), in))};
  EXPECT_EQ(FailedCode(scale.Check(wrong)), KernelError::kRejected);
}

void GgmlExtOpsTest::MultirowOwnerControl(
    std::span<const std::pair<std::int64_t, std::int64_t>> shapes) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  constexpr std::int64_t d = 256, heads = 8, kvh = 4, owners = 2;
  const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  for (const int cap : {0, 50}) {
    for (const auto [rows, cells] : shapes) {
      if (cap == 50 && cells > 16384) continue;
      SCOPED_TRACE(cap);
      SCOPED_TRACE(cells);
      SCOPED_TRACE(rows);
      const auto padded_rows = (rows + 31) / 32 * 32;
      auto arena = TensorArena::Create(64).value();
      auto* ctx = arena.context();
      auto qdata = Normal(18301, n(d * heads * rows * owners), 8.0F);
      const auto kdata = Halves(Normal(18302, n(d * kvh * cells * owners), 4.0F));
      const auto vdata = Halves(Normal(18303, n(d * kvh * cells * owners)));
      auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, rows, owners), qdata);
      auto* raw_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
      auto* raw_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
      const auto permute = [&](ggml_tensor* tensor) {
        auto* view = ggml_permute(ctx, tensor, 0, 2, 1, 3);
        TensorArena::Bind(view, reinterpret_cast<std::uintptr_t>(tensor->data));
        return view;
      };
      auto* q = permute(raw_q);
      std::array<ggml_tensor*, 4> keys{}, values{};
      const auto part = n(d * kvh * cells);
      for (std::size_t owner = 0; owner < 2; ++owner) {
        const auto begin = static_cast<std::ptrdiff_t>(owner * part);
        const auto end = begin + static_cast<std::ptrdiff_t>(part);
        keys[owner] =
            permute(Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                          std::vector<ggml_fp16_t>(kdata.begin() + begin, kdata.begin() + end)));
        values[owner] =
            permute(Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                          std::vector<ggml_fp16_t>(vdata.begin() + begin, vdata.begin() + end)));
      }
      std::vector<ggml_fp16_t> masks(n(cells * padded_rows * owners), 0xFC00);
      const auto fresh_masks = [&](std::int64_t shift) {
        std::fill(masks.begin(), masks.end(), ggml_fp16_t{0xFC00});
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t row = 0; row < rows; ++row) {
            const auto visible = cells - rows - owner * 32 - shift + row + 1;
            ASSERT_GE(visible, 0);
            ASSERT_LE(visible, cells);
            std::fill_n(
                masks.begin() + static_cast<std::ptrdiff_t>((owner * padded_rows + row) * cells),
                visible, ggml_fp16_t{0});
          }
      };
      fresh_masks(0);
      auto* mask =
          Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, padded_rows, 1, owners), masks);
      auto* packed = Place(ggml_flash_attn_ext(ctx, q, permute(raw_k), permute(raw_v), mask, 1, 0,
                                               static_cast<float>(cap)));
      ggml_prec_set_acc(packed, GGML_PREC_F32);
      auto* node = Place(kg::FlashAttnOwnersNode(ctx, q, mask, keys, values, 2, 2, 0,
                                                 static_cast<std::uint32_t>(cap)));
      auto inputs = kg::FlashAttnOwnersFromNode(node);
      ASSERT_TRUE(inputs) << inputs.error().detail;
      auto original = kg::PlanFlashAttnMmaGqa2(launch(), packed);
      auto actual = kg::PlanFlashAttnOwners(launch(), *inputs);
      ASSERT_TRUE(original && actual);
      const int columns = rows <= 4 ? 4 : rows <= 8 ? 8 : rows <= 16 ? 16 : 32;
      EXPECT_EQ(actual->original.columns, columns);
      EXPECT_EQ(actual->original.columns, original->columns);
      EXPECT_EQ(actual->original.group, 2);
      EXPECT_EQ(actual->original.blocks, original->blocks);
      EXPECT_EQ(actual->original.scratch, original->scratch);
      EXPECT_TRUE(actual->original.mask_prepass);
      const auto scratch = original->scratch;
      auto paid = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = Allocate(scratch), .size = Bytes(scratch)});
      ASSERT_TRUE(paid);
      auto short_pool = LaunchContext::Create(
          0, *execution_, stream_, {.base = Allocate(scratch), .size = Bytes(scratch - 1)});
      ASSERT_TRUE(short_pool);
      EXPECT_EQ(FailedCode(kg::FlashAttnOwnerRoots(**short_pool, *inputs)), KernelError::kRejected);
      EXPECT_FALSE((*short_pool)->faulted());
      const auto run = [&](LaunchContext& launch) -> std::expected<void, KernelFailure> {
        if (auto result = kg::FlashAttnMmaGqa2(launch, packed); !result) return result;
        return kg::FlashAttnOwnerRoots(launch, *inputs);
      };
      const auto compare = [&]() {
        const auto reference = Download(packed), got = Download(node);
        ASSERT_EQ(got.size(), n(d * heads * rows * owners));
        ASSERT_EQ(got.size(), reference.size());
        ASSERT_TRUE(std::ranges::all_of(got, [](float value) { return std::isfinite(value); }));
        ASSERT_EQ(std::memcmp(got.data(), reference.data(), got.size() * sizeof(float)), 0);
      };
      ASSERT_TRUE(run(**paid));
      compare();
      auto graph = (*paid)->Capture(run);
      ASSERT_TRUE(graph) << graph.error().detail;
      const auto first = Download(node);
      for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
          qdata = Normal(18304, qdata.size(), 8.0F);
          fresh_masks(32);
          ASSERT_EQ(cudaMemcpyAsync(raw_q->data, qdata.data(), ggml_nbytes(raw_q),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpyAsync(mask->data, masks.data(), ggml_nbytes(mask),
                                    cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
        }
        const std::vector<float> poison(n(d * heads * rows * owners),
                                        std::numeric_limits<float>::quiet_NaN());
        ASSERT_EQ(cudaMemcpyAsync(node->data, poison.data(), ggml_nbytes(node),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(packed->data, poison.data(), ggml_nbytes(packed),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE((*paid)->Launch(*graph));
        compare();
        if (pass == 1) EXPECT_NE(first, Download(node));
      }
      auto bad = *inputs;
      bad.logit_softcap = 1;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      bad = *inputs;
      bad.bounded_roots = true;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      bad = *inputs;
      bad.logical_cohort = 3;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      bad = *inputs;
      bad.k[1] = bad.k[0];
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      ggml_tensor malformed = *mask;
      malformed.ne[0] = cap == 0 ? 131328 : 16640;
      bad = *inputs;
      bad.mask = &malformed;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      // Keep all row-dependent dimensions/strides coherent: the explicit
      // range gate, not a layout inconsistency, must refuse the next row.
      ggml_tensor too_many_q = *q, too_many_mask = *mask, too_many_output = *node;
      too_many_q.ne[1] = 513;
      too_many_q.nb[3] = n(d * heads * 513) * sizeof(float);
      too_many_mask.ne[1] = 544;
      too_many_mask.nb[2] = too_many_mask.nb[3] = n(cells * 544) * sizeof(ggml_fp16_t);
      too_many_output.ne[2] = 513;
      too_many_output.nb[3] = n(d * heads * 513) * sizeof(float);
      bad = *inputs;
      bad.q = &too_many_q;
      bad.mask = &too_many_mask;
      bad.output = &too_many_output;
      const auto range_refused = kg::CheckFlashAttnOwners(bad);
      ASSERT_FALSE(range_refused);
      EXPECT_EQ(range_refused.error().error, KernelError::kRejected);
      EXPECT_NE(range_refused.error().detail.find("2-to-512-row"), std::string::npos);
      malformed = *mask;
      malformed.ne[1] = 127;
      bad = *inputs;
      bad.mask = &malformed;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      malformed = *node;
      malformed.data = q->data;
      bad = *inputs;
      bad.output = &malformed;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      malformed = *node;
      malformed.data = mask->data;
      bad = *inputs;
      bad.output = &malformed;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      ggml_tensor mask_parent = *mask;
      malformed = *mask;
      malformed.view_src = &mask_parent;
      bad = *inputs;
      bad.mask = &malformed;
      EXPECT_TRUE(kg::CheckFlashAttnOwners(bad));
      malformed.view_offs = 16;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      malformed.view_offs = 0;
      mask_parent.nb[3] = 1ULL << 30U;
      EXPECT_EQ(FailedCode(kg::CheckFlashAttnOwners(bad)), KernelError::kRejected);
      std::cout << "GEMMA_MULTIROW_OWNER cap=" << cap << " rows=" << rows << " cells=" << cells
                << " columns=" << actual->original.columns << " blocks=" << actual->original.blocks
                << " scratch=" << scratch << " exact_eager_and_changed_replay=1\n";
    }
  }
}
TEST_F(GgmlExtOpsTest, GemmaMultirowOwnerRootsMatchPackedMmaEagerAndChangedReplay) {
  std::vector<std::pair<std::int64_t, std::int64_t>> shapes;
  for (const auto rows : {2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128})
    shapes.emplace_back(rows, 512);
  for (const auto cells : {256, 4352, 16384, 32768, 131072}) shapes.emplace_back(128, cells);
  for (const auto cells : {4352, 16384}) shapes.emplace_back(64, cells);
  shapes.emplace_back(5, 32768);
  shapes.emplace_back(33, 131072);
  MultirowOwnerControl(shapes);
}

TEST_F(GgmlExtOpsTest, GemmaLargerMultirowOwnersMatchPackedMmaEagerAndChangedReplay) {
  const std::array shapes{std::pair<std::int64_t, std::int64_t>{129, 512},
                          std::pair<std::int64_t, std::int64_t>{256, 512},
                          std::pair<std::int64_t, std::int64_t>{256, 4352},
                          std::pair<std::int64_t, std::int64_t>{256, 16384},
                          std::pair<std::int64_t, std::int64_t>{129, 131072}};
  MultirowOwnerControl(shapes);
}

TEST_F(GgmlExtOpsTest, GemmaLargestMultirowMaskMatchesPackedMmaEagerAndChangedReplay) {
  const std::array shapes{std::pair<std::int64_t, std::int64_t>{256, 131072}};
  MultirowOwnerControl(shapes);
}

TEST_F(GgmlExtOpsTest, Gemma512QueryTilesMatchPackedMmaEagerAndChangedReplay) {
  const std::array shapes{std::pair<std::int64_t, std::int64_t>{511, 4608},
                          std::pair<std::int64_t, std::int64_t>{512, 4608}};
  MultirowOwnerControl(shapes);
}

TEST_F(GgmlExtOpsTest, Gemma512LargestMaskMatchesPackedMmaEagerAndChangedReplay) {
  const std::array shapes{std::pair<std::int64_t, std::int64_t>{512, 131072}};
  MultirowOwnerControl(shapes);
}

TEST_F(GgmlExtOpsTest, Gemma3PrefillCommonOperandsExposePackedAndSplitMmaGeometry) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 prefill diagnostic";
  constexpr std::int64_t d = 256, heads = 8, kvh = 4, rows = 128, owners = 2;
  const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  for (const std::int64_t cells : {256, 512}) {
    auto arena = TensorArena::Create(64).value();
    auto* ctx = arena.context();
    auto qdata = Normal(731, n(d * heads * rows * owners), 0.0625F);
    const auto kdata = Halves(Normal(732, n(d * kvh * cells * owners)));
    const auto vdata = Halves(Normal(733, n(d * kvh * cells * owners)));
    auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, rows, owners), qdata);
    auto* raw_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
    auto* raw_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
    const auto permute = [&](ggml_tensor* raw) {
      auto* t = ggml_permute(ctx, raw, 0, 2, 1, 3);
      TensorArena::Bind(t, reinterpret_cast<std::uintptr_t>(raw->data));
      return t;
    };
    auto* q = permute(raw_q);
    auto* k = permute(raw_k);
    auto* v = permute(raw_v);
    std::vector<ggml_fp16_t> mask_data(n(cells * rows * owners), 0xFC00);
    const auto fresh_mask = [&](std::int64_t past) {
      std::fill(mask_data.begin(), mask_data.end(), 0xFC00);
      for (std::int64_t s = 0; s < owners; ++s)
        for (std::int64_t r = 0; r < rows; ++r)
          std::fill_n(mask_data.begin() + static_cast<std::ptrdiff_t>((s * rows + r) * cells),
                      past + r + 1, ggml_fp16_t{0});
    };
    fresh_mask(0);
    auto* mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, rows, 1, owners), mask_data);
    const auto node = [&](ggml_tensor* tq, ggml_tensor* tk, ggml_tensor* tv, ggml_tensor* tm) {
      auto* out = Place(ggml_flash_attn_ext(ctx, tq, tk, tv, tm, 1, 0, 0));
      ggml_prec_set_acc(out, GGML_PREC_F32);
      return out;
    };
    auto* packed = node(q, k, v, mask);
    std::array<ggml_tensor*, 2> split{};
    const auto view = [&](ggml_tensor* t, std::int64_t s) {
      const auto offset = n(s) * t->nb[3];
      auto* part = ggml_view_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], 1, t->nb[1], t->nb[2],
                                t->nb[3], offset);
      TensorArena::Bind(part, reinterpret_cast<std::uintptr_t>(t->data) + offset);
      return part;
    };
    for (std::int64_t s = 0; s < owners; ++s)
      split[n(s)] = node(view(q, s), view(k, s), view(v, s), view(mask, s));
    const auto full = kg::PlanFlashAttnMmaGqa2(launch(), packed);
    const auto single = kg::PlanFlashAttnMmaGqa2(launch(), split[0]);
    ASSERT_TRUE(full && single);
    const auto shape = kg::detail::FlashAttnMmaShapeGqa2(32, launch().device());
    ASSERT_TRUE(shape);
    EXPECT_EQ(full->columns, 32);
    EXPECT_EQ(single->columns, 32);
    EXPECT_TRUE(full->mask_prepass);
    EXPECT_FALSE(single->mask_prepass);
    const auto scratch = std::max(full->scratch, single->scratch);
    auto paid = LaunchContext::Create(0, *execution_, stream_,
                                      {.base = Allocate(scratch), .size = Bytes(scratch)});
    ASSERT_TRUE(paid);
    const auto run = [&](LaunchContext& launch) -> std::expected<void, KernelFailure> {
      if (auto result = kg::FlashAttnMmaGqa2(launch, packed); !result) return result;
      for (auto* part : split)
        if (auto result = kg::FlashAttnMmaGqa2(launch, part); !result) return result;
      return {};
    };
    ASSERT_TRUE(run(**paid));
    auto eager_packed = Download(packed);
    const auto read_split = [&] {
      std::vector<float> result;
      for (auto* part : split) {
        const auto data = Download(part);
        result.insert(result.end(), data.begin(), data.end());
      }
      return result;
    };
    auto eager_split = read_split();
    auto captured = (*paid)->Capture(run);
    ASSERT_TRUE(captured);
    std::vector<float> prior_packed, prior_split;
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) {
        qdata = Normal(734, qdata.size(), 0.0625F);
        fresh_mask(32);
        ASSERT_EQ(cudaMemcpyAsync(raw_q->data, qdata.data(), ggml_nbytes(raw_q),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(mask->data, mask_data.data(), ggml_nbytes(mask),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE(run(**paid));
        eager_packed = Download(packed);
        eager_split = read_split();
      }
      ASSERT_TRUE((*paid)->Launch(*captured));
      const auto whole = Download(packed);
      const auto parts = read_split();
      ASSERT_EQ(whole.size(), parts.size());
      EXPECT_EQ(std::memcmp(whole.data(), eager_packed.data(), whole.size() * sizeof(float)), 0);
      EXPECT_EQ(std::memcmp(parts.data(), eager_split.data(), parts.size() * sizeof(float)), 0);
      EXPECT_TRUE(std::ranges::all_of(whole, [](float x) { return std::isfinite(x); }));
      EXPECT_TRUE(std::ranges::all_of(parts, [](float x) { return std::isfinite(x); }));
      std::size_t changed = 0;
      float max_raw_difference = 0;
      for (std::size_t i = 0; i < whole.size(); ++i) {
        changed += std::bit_cast<std::uint32_t>(whole[i]) != std::bit_cast<std::uint32_t>(parts[i]);
        max_raw_difference = std::max(max_raw_difference, std::abs(whole[i] - parts[i]));
      }
      if (pass == 1) {
        EXPECT_NE(whole, prior_packed);
        EXPECT_NE(parts, prior_split);
      }
      std::vector<double> want(whole.size());
      std::vector<double> logits(n(cells));
      for (std::int64_t owner = 0; owner < owners; ++owner)
        for (std::int64_t r = 0; r < rows; ++r)
          for (std::int64_t h = 0; h < heads; ++h) {
            double maximum = -std::numeric_limits<double>::infinity();
            const auto visible = (pass == 0 ? 0 : 32) + r + 1;
            for (std::int64_t cell = 0; cell < visible; ++cell) {
              double dot = 0;
              for (std::int64_t i = 0; i < d; ++i)
                dot += static_cast<double>(qdata[n(((owner * rows + r) * heads + h) * d + i)]) *
                       ggml_fp16_to_fp32(kdata[n(((owner * cells + cell) * kvh + h / 2) * d + i)]);
              logits[n(cell)] = dot;
              maximum = std::max(maximum, dot);
            }
            double sum = 0;
            for (std::int64_t cell = 0; cell < visible; ++cell)
              sum += std::exp(logits[n(cell)] - maximum);
            for (std::int64_t cell = 0; cell < visible; ++cell) {
              const double probability = std::exp(logits[n(cell)] - maximum) / sum;
              for (std::int64_t i = 0; i < d; ++i)
                want[n(((owner * rows + r) * heads + h) * d + i)] +=
                    probability *
                    ggml_fp16_to_fp32(vdata[n(((owner * cells + cell) * kvh + h / 2) * d + i)]);
            }
          }
      ExpectNmse(whole, want, kFlashAttnNmse, "packed C2 common operands");
      ExpectNmse(parts, want, kFlashAttnNmse, "two C1 common operands");
      ASSERT_TRUE((*paid)->Launch(*captured));
      const auto repeated_packed = Download(packed);
      EXPECT_EQ(std::memcmp(whole.data(), repeated_packed.data(), whole.size() * sizeof(float)), 0);
      for (std::size_t s = 0; s < split.size(); ++s) {
        const auto repeated = Download(split[s]);
        EXPECT_EQ(std::memcmp(parts.data() + s * repeated.size(), repeated.data(),
                              repeated.size() * sizeof(float)),
                  0);
      }
      EXPECT_EQ(Download(raw_q), qdata);
      EXPECT_EQ(Download<ggml_fp16_t>(raw_k), kdata);
      EXPECT_EQ(Download<ggml_fp16_t>(raw_v), vdata);
      EXPECT_EQ(Download<ggml_fp16_t>(mask), mask_data);
      std::cout << "GEMMA3_PREFILL_COMMON cells=" << cells << " pass=" << pass
                << " columns=" << full->columns << " sms=" << sms
                << " blocks_per_sm=" << shape->blocks_per_sm << " kv_batch=" << shape->kv_batch
                << " split_tiles=16 packed_tiles=32 split_blocks=" << single->blocks
                << " packed_blocks=" << full->blocks << " split_scan=" << single->mask_prepass
                << " packed_scan=" << full->mask_prepass << " raw_differences=" << changed
                << " max_raw=" << max_raw_difference << " scratch=" << scratch << '\n';
      prior_packed = whole;
      prior_split = parts;
    }
  }
}

TEST_F(GgmlExtOpsTest, Gemma3UnequalOwnerPaddingMatchesTheFundedCommonStream) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 two-owner control";
  constexpr std::int64_t d = 256, heads = 8, kvh = 4, short_cells = 512, cells = 1024;
  const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
  auto arena = TensorArena::Create(96).value();
  auto* ctx = arena.context();
  auto qdata = Normal(741, n(d * heads * 2), 0.0625F);
  auto kdata = Halves(Normal(742, n(d * kvh * (short_cells + cells)), 0.25F));
  auto vdata = Halves(Normal(743, n(d * kvh * (short_cells + cells)), 0.25F));
  auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2), qdata);
  auto* q = ggml_permute(ctx, raw_q, 0, 2, 1, 3);
  TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
  std::array<ggml_tensor*, 2> raw_k{}, raw_v{}, masks{};
  for (std::size_t owner = 0; owner < 2; ++owner) {
    const auto count = owner == 0 ? short_cells : cells;
    const auto begin = owner == 0 ? 0 : n(d * kvh * short_cells);
    const auto slice = [&](const auto& data) {
      return std::vector<ggml_fp16_t>(
          data.begin() + static_cast<std::ptrdiff_t>(begin),
          data.begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count)));
    };
    raw_k[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count, 1), slice(kdata));
    raw_v[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count, 1), slice(vdata));
    masks[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, count, 32, 1, 1));
  }
  const auto filled = [&](std::array<std::int64_t, 4> dims, float value) {
    auto* out = ggml_fill(ctx, ggml_new_tensor(ctx, GGML_TYPE_F16, 4, dims.data()), value);
    out->src[0] = nullptr;  // shape-only template; fill reads no source
    return Place(out);
  };
  auto* zeros = filled({d, kvh, cells - short_cells, 1}, 0);
  auto* invisible = filled({cells - short_cells, 32, 1, 1}, -INFINITY);
  auto* padded_k = Place(ggml_concat(ctx, raw_k[0], zeros, 2));
  auto* padded_v = Place(ggml_concat(ctx, raw_v[0], zeros, 2));
  auto* padded_mask = Place(ggml_concat(ctx, masks[0], invisible, 0));
  auto* owner_mask = Place(ggml_concat(ctx, padded_mask, masks[1], 3));
  kg::FlashAttnOwners in;
  in.q = q;
  in.mask = owner_mask;
  in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2));
  in.owner_count = in.logical_cohort = 2;
  for (std::size_t owner = 0; owner < 2; ++owner) {
    auto* key = owner == 0 ? padded_k : raw_k[1];
    auto* value = owner == 0 ? padded_v : raw_v[1];
    in.k[owner] = ggml_permute(ctx, key, 0, 2, 1, 3);
    in.v[owner] = ggml_permute(ctx, value, 0, 2, 1, 3);
    TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                      reinterpret_cast<std::uintptr_t>(key->data));
    TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                      reinterpret_cast<std::uintptr_t>(value->data));
  }
  auto* physical_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
  auto* physical_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
  auto* physical_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
  auto* k = ggml_permute(ctx, physical_k, 0, 2, 1, 3);
  auto* v = ggml_permute(ctx, physical_v, 0, 2, 1, 3);
  TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(physical_k->data));
  TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(physical_v->data));
  auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, physical_mask, 1, 0, 0));
  ggml_prec_set_acc(whole, GGML_PREC_F32);
  auto plan = kg::PlanFlashAttnMmaGqa2(launch(), whole);
  auto owners = kg::PlanFlashAttnOwners(launch(), in);
  ASSERT_TRUE(plan);
  ASSERT_TRUE(owners);
  EXPECT_EQ(plan->columns, 4);
  EXPECT_EQ(owners->original.columns, plan->columns);
  EXPECT_EQ(owners->cohort_blocks, plan->blocks);
  EXPECT_EQ(owners->original.mask_prepass, plan->mask_prepass);
  EXPECT_TRUE(plan->mask_prepass);
  const auto scratch = std::max(plan->scratch, owners->original.scratch);
  auto paid = LaunchContext::Create(launch().device(), *execution_, stream_,
                                    {.base = Allocate(scratch), .size = Bytes(scratch)});
  ASSERT_TRUE(paid);
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  const auto run = [&](LaunchContext& context) -> std::expected<void, KernelFailure> {
    for (auto* fill : {zeros, invisible})
      if (auto x = kg::Fill(context, fill); !x) return x;
    for (auto* concat : {padded_k, padded_v, padded_mask, owner_mask})
      if (auto x = kg::Concat(context, concat); !x) return x;
    if (auto x = kg::FlashAttnMmaGqa2(context, whole); !x) return x;
    return kg::FlashAttnOwnerRoots(context, in);
  };
  std::vector<float> prior;
  std::optional<kg::CapturedGraph> graph;
  for (std::size_t pass = 0; pass < 3; ++pass) {
    if (pass != 0) {
      const auto begin = pass == 1 ? 0U : n(d * kvh * short_cells);
      const auto end = pass == 1 ? n(d * kvh * short_cells) : kdata.size();
      for (std::size_t i = begin; i < end; ++i) {
        kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * 0.75F);
        vdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(vdata[i]) + 0.125F);
      }
    }
    std::vector<ggml_fp16_t> pk(n(d * kvh * cells * 2), 0), pv(pk.size(), 0);
    std::vector<ggml_fp16_t> mask(n(cells * 32 * 2), ggml_fp32_to_fp16(-INFINITY));
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const auto count = owner == 0 ? short_cells : cells;
      const auto begin = owner == 0 ? 0U : n(d * kvh * short_cells);
      const auto length = n(d * kvh * count);
      const auto prefix = [&](const auto& data) { return std::span(data).subspan(begin, length); };
      const auto copy = [&](ggml_tensor* tensor, auto data) {
        return cudaMemcpyAsync(tensor->data, data.data(), data.size_bytes(), cudaMemcpyHostToDevice,
                               stream);
      };
      ASSERT_EQ(copy(raw_k[owner], prefix(kdata)), cudaSuccess);
      ASSERT_EQ(copy(raw_v[owner], prefix(vdata)), cudaSuccess);
      std::ranges::copy(prefix(kdata),
                        pk.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
      std::ranges::copy(prefix(vdata),
                        pv.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
      const auto visible = (owner == 0 ? 260 : 772) - static_cast<int>(pass);
      std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(owner * n(cells * 32)), visible, 0);
      std::vector<ggml_fp16_t> own_mask(n(count * 32), ggml_fp32_to_fp16(-INFINITY));
      std::fill_n(own_mask.begin(), visible, 0);
      ASSERT_EQ(copy(masks[owner], std::span(own_mask)), cudaSuccess);
    }
    ASSERT_EQ(cudaMemcpyAsync(physical_k->data, pk.data(), ggml_nbytes(physical_k),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(physical_v->data, pv.data(), ggml_nbytes(physical_v),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(physical_mask->data, mask.data(), ggml_nbytes(physical_mask),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_TRUE(run(**paid));
    const auto eager = Download(in.output);
    const auto expected = Download(whole);
    ASSERT_EQ(std::memcmp(eager.data(), expected.data(), eager.size() * sizeof(float)), 0);
    ASSERT_TRUE(std::ranges::all_of(eager, [](float x) { return std::isfinite(x); }));
    std::vector<double> want(n(d * heads * 2));
    for (std::int64_t owner = 0; owner < 2; ++owner)
      for (std::int64_t head = 0; head < heads; ++head) {
        const auto visible = (owner == 0 ? 260 : 772) - static_cast<int>(pass);
        std::vector<double> scores(n(visible));
        double largest = -std::numeric_limits<double>::infinity();
        for (std::int64_t cell = 0; cell < visible; ++cell) {
          double dot = 0;
          for (std::int64_t i = 0; i < d; ++i)
            dot += static_cast<double>(qdata[n((owner * heads + head) * d + i)]) *
                   ggml_fp16_to_fp32(pk[n(((owner * cells + cell) * kvh + head / 2) * d + i)]);
          scores[n(cell)] = dot;
          largest = std::max(largest, dot);
        }
        double sum = 0;
        for (double score : scores) sum += std::exp(score - largest);
        for (std::int64_t cell = 0; cell < visible; ++cell)
          for (std::int64_t i = 0; i < d; ++i)
            want[n((owner * heads + head) * d + i)] +=
                std::exp(scores[n(cell)] - largest) / sum *
                ggml_fp16_to_fp32(pv[n(((owner * cells + cell) * kvh + head / 2) * d + i)]);
      }
    ExpectNmse(eager, want, kFlashAttnNmse, "unequal padded owner FP64");
    if (pass != 0) EXPECT_NE(eager, prior);
    if (!graph) {
      auto captured = (*paid)->Capture(run);
      ASSERT_TRUE(captured);
      graph.emplace(std::move(*captured));
    }
    for (int replay = 0; replay < 2; ++replay) {
      ASSERT_EQ(cudaMemsetAsync(zeros->data, 0xFF, ggml_nbytes(zeros), stream), cudaSuccess);
      ASSERT_EQ(cudaMemsetAsync(invisible->data, 0, ggml_nbytes(invisible), stream), cudaSuccess);
      ASSERT_TRUE((*paid)->Launch(*graph));
      const auto actual = Download(in.output);
      EXPECT_EQ(std::memcmp(actual.data(), eager.data(), actual.size() * sizeof(float)), 0);
      EXPECT_EQ(Download<ggml_fp16_t>(zeros),
                std::vector<ggml_fp16_t>(n(d * kvh * (cells - short_cells)), 0));
      EXPECT_EQ(Download<ggml_fp16_t>(owner_mask), mask);
      EXPECT_EQ(Download<ggml_fp16_t>(padded_k),
                std::vector<ggml_fp16_t>(
                    pk.begin(), pk.begin() + static_cast<std::ptrdiff_t>(n(d * kvh * cells))));
    }
    EXPECT_EQ(raw_k[0]->ne[2], short_cells);
    EXPECT_EQ(raw_k[1]->ne[2], cells);
    EXPECT_EQ(Download(raw_q), qdata);
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const auto count = owner == 0 ? short_cells : cells;
      const auto begin = owner == 0 ? 0U : n(d * kvh * short_cells);
      for (const auto& source : {std::pair{raw_k[owner], &kdata}, std::pair{raw_v[owner], &vdata}})
        EXPECT_EQ(
            Download<ggml_fp16_t>(source.first),
            std::vector<ggml_fp16_t>(
                source.second->begin() + static_cast<std::ptrdiff_t>(begin),
                source.second->begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count))));
      std::vector<ggml_fp16_t> original_mask(n(count * 32), ggml_fp32_to_fp16(-INFINITY));
      std::fill_n(original_mask.begin(), (owner == 0 ? 260 : 772) - static_cast<int>(pass), 0);
      EXPECT_EQ(Download<ggml_fp16_t>(masks[owner]), original_mask);
    }
    std::cout << "GEMMA3_UNEQUAL_PADDING pass=" << pass
              << " source_cells=512/1024 common_cells=1024"
              << " columns=" << plan->columns << " blocks=" << plan->blocks
              << " scan=" << plan->mask_prepass << " scratch=" << scratch << '\n';
    prior = eager;
  }
}

TEST_F(GgmlExtOpsTest, Gemma4UnequalOwnerPaddingMatchesBothHeadDimensions) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 two-owner control";
  for (const std::int64_t heads : {16, 32})
    for (const std::int64_t d : {256, 512}) {
      const std::int64_t gqa = d == 256 ? 2 : 8, kvh = heads / gqa;
      constexpr std::int64_t short_cells = 512, cells = 1024;
      const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
      auto arena = TensorArena::Create(96).value();
      auto* ctx = arena.context();
      auto qdata = Normal(741, n(d * heads * 2), 0.0625F);
      auto kdata = Halves(Normal(742, n(d * kvh * (short_cells + cells)), 0.25F));
      auto vdata = Halves(Normal(743, n(d * kvh * (short_cells + cells)), 0.25F));
      auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2), qdata);
      auto* q = ggml_permute(ctx, raw_q, 0, 2, 1, 3);
      TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
      std::array<ggml_tensor*, 2> raw_k{}, raw_v{}, masks{};
      for (std::size_t owner = 0; owner < 2; ++owner) {
        const auto count = owner == 0 ? short_cells : cells;
        const auto begin = owner == 0 ? 0 : n(d * kvh * short_cells);
        const auto slice = [&](const auto& data) {
          return std::vector<ggml_fp16_t>(
              data.begin() + static_cast<std::ptrdiff_t>(begin),
              data.begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count)));
        };
        raw_k[owner] =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count, 1), slice(kdata));
        raw_v[owner] =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count, 1), slice(vdata));
        masks[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, count, 32, 1, 1));
      }
      const auto filled = [&](std::array<std::int64_t, 4> dims, float value) {
        auto* out = ggml_fill(ctx, ggml_new_tensor(ctx, GGML_TYPE_F16, 4, dims.data()), value);
        out->src[0] = nullptr;  // shape-only template; fill reads no source
        return Place(out);
      };
      auto* zeros = filled({d, kvh, cells - short_cells, 1}, 0);
      auto* invisible = filled({cells - short_cells, 32, 1, 1}, -INFINITY);
      auto* padded_k = Place(ggml_concat(ctx, raw_k[0], zeros, 2));
      auto* padded_v = Place(ggml_concat(ctx, raw_v[0], zeros, 2));
      auto* padded_mask = Place(ggml_concat(ctx, masks[0], invisible, 0));
      auto* owner_mask = Place(ggml_concat(ctx, padded_mask, masks[1], 3));
      kg::FlashAttnOwners in;
      in.q = q;
      in.mask = owner_mask;
      in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2));
      in.owner_count = in.logical_cohort = 2;
      for (std::size_t owner = 0; owner < 2; ++owner) {
        auto* key = owner == 0 ? padded_k : raw_k[1];
        auto* value = owner == 0 ? padded_v : raw_v[1];
        in.k[owner] = ggml_permute(ctx, key, 0, 2, 1, 3);
        in.v[owner] = ggml_permute(ctx, value, 0, 2, 1, 3);
        TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                          reinterpret_cast<std::uintptr_t>(key->data));
        TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                          reinterpret_cast<std::uintptr_t>(value->data));
      }
      auto* physical_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
      auto* physical_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
      auto* physical_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
      auto* k = ggml_permute(ctx, physical_k, 0, 2, 1, 3);
      auto* v = ggml_permute(ctx, physical_v, 0, 2, 1, 3);
      TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(physical_k->data));
      TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(physical_v->data));
      auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, physical_mask, 1, 0, 0));
      ggml_prec_set_acc(whole, GGML_PREC_F32);
      auto plan = (d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                            : kg::PlanFlashAttnMma(launch(), whole));
      auto owners = kg::PlanFlashAttnOwners(launch(), in);
      ASSERT_TRUE(plan);
      ASSERT_TRUE(owners);
      EXPECT_EQ(plan->columns, d == 256 ? 4 : 1);
      EXPECT_EQ(owners->original.columns, plan->columns);
      EXPECT_EQ(owners->cohort_blocks, plan->blocks);
      EXPECT_EQ(owners->original.mask_prepass, plan->mask_prepass);
      EXPECT_TRUE(plan->mask_prepass);
      const auto scratch = std::max(plan->scratch, owners->original.scratch);
      auto paid = LaunchContext::Create(launch().device(), *execution_, stream_,
                                        {.base = Allocate(scratch), .size = Bytes(scratch)});
      ASSERT_TRUE(paid);
      const auto submission = execution_->Submission(stream_);
      ASSERT_TRUE(submission);
      const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
      const auto run = [&](LaunchContext& context) -> std::expected<void, KernelFailure> {
        for (auto* fill : {zeros, invisible})
          if (auto x = kg::Fill(context, fill); !x) return x;
        for (auto* concat : {padded_k, padded_v, padded_mask, owner_mask})
          if (auto x = kg::Concat(context, concat); !x) return x;
        if (auto x = (d == 256 ? kg::FlashAttnMmaGqa2(context, whole)
                               : kg::FlashAttnMma(context, whole));
            !x)
          return x;
        return kg::FlashAttnOwnerRoots(context, in);
      };
      std::vector<float> prior;
      std::optional<kg::CapturedGraph> graph;
      for (std::size_t pass = 0; pass < 3; ++pass) {
        if (pass != 0) {
          const auto begin = pass == 1 ? 0U : n(d * kvh * short_cells);
          const auto end = pass == 1 ? n(d * kvh * short_cells) : kdata.size();
          for (std::size_t i = begin; i < end; ++i) {
            kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * 0.75F);
            vdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(vdata[i]) + 0.125F);
          }
        }
        std::vector<ggml_fp16_t> pk(n(d * kvh * cells * 2), 0), pv(pk.size(), 0);
        std::vector<ggml_fp16_t> mask(n(cells * 32 * 2), ggml_fp32_to_fp16(-INFINITY));
        for (std::size_t owner = 0; owner < 2; ++owner) {
          const auto count = owner == 0 ? short_cells : cells;
          const auto begin = owner == 0 ? 0U : n(d * kvh * short_cells);
          const auto length = n(d * kvh * count);
          const auto prefix = [&](const auto& data) {
            return std::span(data).subspan(begin, length);
          };
          const auto copy = [&](ggml_tensor* tensor, auto data) {
            return cudaMemcpyAsync(tensor->data, data.data(), data.size_bytes(),
                                   cudaMemcpyHostToDevice, stream);
          };
          ASSERT_EQ(copy(raw_k[owner], prefix(kdata)), cudaSuccess);
          ASSERT_EQ(copy(raw_v[owner], prefix(vdata)), cudaSuccess);
          std::ranges::copy(prefix(kdata),
                            pk.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
          std::ranges::copy(prefix(vdata),
                            pv.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
          const auto visible = (owner == 0 ? 260 : 772) - static_cast<int>(pass);
          std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(owner * n(cells * 32)), visible,
                      0);
          std::vector<ggml_fp16_t> own_mask(n(count * 32), ggml_fp32_to_fp16(-INFINITY));
          std::fill_n(own_mask.begin(), visible, 0);
          ASSERT_EQ(copy(masks[owner], std::span(own_mask)), cudaSuccess);
        }
        ASSERT_EQ(cudaMemcpyAsync(physical_k->data, pk.data(), ggml_nbytes(physical_k),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(physical_v->data, pv.data(), ggml_nbytes(physical_v),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(physical_mask->data, mask.data(), ggml_nbytes(physical_mask),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE(run(**paid));
        const auto eager = Download(in.output);
        const auto expected = Download(whole);
        ASSERT_EQ(std::memcmp(eager.data(), expected.data(), eager.size() * sizeof(float)), 0);
        ASSERT_TRUE(std::ranges::all_of(eager, [](float x) { return std::isfinite(x); }));
        std::vector<double> want(n(d * heads * 2));
        for (std::int64_t owner = 0; owner < 2; ++owner)
          for (std::int64_t head = 0; head < heads; ++head) {
            const auto visible = (owner == 0 ? 260 : 772) - static_cast<int>(pass);
            std::vector<double> scores(n(visible));
            double largest = -std::numeric_limits<double>::infinity();
            for (std::int64_t cell = 0; cell < visible; ++cell) {
              double dot = 0;
              for (std::int64_t i = 0; i < d; ++i)
                dot +=
                    static_cast<double>(qdata[n((owner * heads + head) * d + i)]) *
                    ggml_fp16_to_fp32(pk[n(((owner * cells + cell) * kvh + head / gqa) * d + i)]);
              scores[n(cell)] = dot;
              largest = std::max(largest, dot);
            }
            double sum = 0;
            for (double score : scores) sum += std::exp(score - largest);
            for (std::int64_t cell = 0; cell < visible; ++cell)
              for (std::int64_t i = 0; i < d; ++i)
                want[n((owner * heads + head) * d + i)] +=
                    std::exp(scores[n(cell)] - largest) / sum *
                    ggml_fp16_to_fp32(pv[n(((owner * cells + cell) * kvh + head / gqa) * d + i)]);
          }
        ExpectNmse(eager, want, kFlashAttnNmse, "unequal padded owner FP64");
        if (pass != 0) EXPECT_NE(eager, prior);
        if (!graph) {
          auto captured = (*paid)->Capture(run);
          ASSERT_TRUE(captured);
          graph.emplace(std::move(*captured));
        }
        for (int replay = 0; replay < 2; ++replay) {
          ASSERT_EQ(cudaMemsetAsync(zeros->data, 0xFF, ggml_nbytes(zeros), stream), cudaSuccess);
          ASSERT_EQ(cudaMemsetAsync(invisible->data, 0, ggml_nbytes(invisible), stream),
                    cudaSuccess);
          ASSERT_TRUE((*paid)->Launch(*graph));
          const auto actual = Download(in.output);
          EXPECT_EQ(std::memcmp(actual.data(), eager.data(), actual.size() * sizeof(float)), 0);
          EXPECT_EQ(Download<ggml_fp16_t>(zeros),
                    std::vector<ggml_fp16_t>(n(d * kvh * (cells - short_cells)), 0));
          EXPECT_EQ(Download<ggml_fp16_t>(owner_mask), mask);
          EXPECT_EQ(Download<ggml_fp16_t>(padded_k),
                    std::vector<ggml_fp16_t>(
                        pk.begin(), pk.begin() + static_cast<std::ptrdiff_t>(n(d * kvh * cells))));
        }
        EXPECT_EQ(raw_k[0]->ne[2], short_cells);
        EXPECT_EQ(raw_k[1]->ne[2], cells);
        EXPECT_EQ(Download(raw_q), qdata);
        for (std::size_t owner = 0; owner < 2; ++owner) {
          const auto count = owner == 0 ? short_cells : cells;
          const auto begin = owner == 0 ? 0U : n(d * kvh * short_cells);
          for (const auto& source :
               {std::pair{raw_k[owner], &kdata}, std::pair{raw_v[owner], &vdata}})
            EXPECT_EQ(Download<ggml_fp16_t>(source.first),
                      std::vector<ggml_fp16_t>(
                          source.second->begin() + static_cast<std::ptrdiff_t>(begin),
                          source.second->begin() +
                              static_cast<std::ptrdiff_t>(begin + n(d * kvh * count))));
          std::vector<ggml_fp16_t> original_mask(n(count * 32), ggml_fp32_to_fp16(-INFINITY));
          std::fill_n(original_mask.begin(), (owner == 0 ? 260 : 772) - static_cast<int>(pass), 0);
          EXPECT_EQ(Download<ggml_fp16_t>(masks[owner]), original_mask);
        }
        std::cout << "GEMMA4_UNEQUAL_PADDING heads=" << heads << " d=" << d << " pass=" << pass
                  << " source_cells=512/1024 common_cells=1024"
                  << " columns=" << plan->columns << " blocks=" << plan->blocks
                  << " scan=" << plan->mask_prepass << " scratch=" << scratch << '\n';
        prior = eager;
      }
    }
}

TEST_F(GgmlExtOpsTest, Gemma4BoundedOwnerRootsMatchBothHeadDimensionsAndThePaddedOracle) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 two-owner control";
  std::array<int, 2> empty_fixups_by_specialization{};
  for (const std::int64_t heads : {16, 32})
    for (const std::int64_t d : {256, 512}) {
      const std::int64_t gqa = d == 256 ? 2 : 8, kvh = heads / gqa;
      for (const auto widths : {std::array<std::int64_t, 2>{512, 1024}, {256, 1536}})
        for (const std::size_t short_owner : {0U, 1U}) {
          const auto short_cells = widths[0], cells = widths[1];
          const auto actual_cells = [&](std::size_t owner) {
            return owner == short_owner ? short_cells : cells;
          };
          const auto visible_cells = [&](std::size_t owner) {
            return owner == short_owner ? (short_cells == 512 ? 260 : 196)
                                        : static_cast<int>(cells) - 252;
          };
          const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
          auto arena = TensorArena::Create(128).value();
          auto* ctx = arena.context();
          auto qdata = Normal(741, n(d * heads * 2), 0.75F);
          auto kdata = Halves(Normal(742, n(d * kvh * (short_cells + cells)), 1.5F));
          auto vdata = Halves(Normal(743, n(d * kvh * (short_cells + cells)), 0.25F));
          auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2), qdata);
          auto* q = ggml_permute(ctx, raw_q, 0, 2, 1, 3);
          TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
          std::array<ggml_tensor*, 2> raw_k{}, raw_v{}, parents_k{}, parents_v{};
          const auto begin_offset = [&](std::size_t owner) {
            return owner == 0 ? 0U : n(d * kvh * actual_cells(0));
          };
          for (std::size_t owner = 0; owner < 2; ++owner) {
            const auto count = actual_cells(owner);
            const auto begin = begin_offset(owner);
            const auto slice = [&](const auto& data) {
              return std::vector<ggml_fp16_t>(
                  data.begin() + static_cast<std::ptrdiff_t>(begin),
                  data.begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count)));
            };
            for (const bool value : {false, true}) {
              std::vector<ggml_fp16_t> guarded(n(d * kvh * (count + 256)), ggml_fp32_to_fp16(NAN));
              const auto source = slice(value ? vdata : kdata);
              std::ranges::copy(source, guarded.begin());
              auto* parent =
                  Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count + 256, 1), guarded);
              auto* view = ggml_view_4d(ctx, parent, d, kvh, count, 1, n(d * 2), n(d * kvh * 2),
                                        n(d * kvh * count * 2), 0);
              TensorArena::Bind(view, reinterpret_cast<std::uintptr_t>(parent->data));
              (value ? parents_v[owner] : parents_k[owner]) = parent;
              (value ? raw_v[owner] : raw_k[owner]) = view;
            }
          }
          auto* owner_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
          kg::FlashAttnOwners in;
          in.q = q;
          in.mask = owner_mask;
          in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2));
          in.owner_count = in.logical_cohort = 2;
          in.logit_softcap = 0;
          in.bounded_roots = true;
          for (std::size_t owner = 0; owner < 2; ++owner) {
            auto* key = raw_k[owner];
            auto* value = raw_v[owner];
            in.k[owner] = ggml_permute(ctx, key, 0, 2, 1, 3);
            in.v[owner] = ggml_permute(ctx, value, 0, 2, 1, 3);
            TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                              reinterpret_cast<std::uintptr_t>(key->data));
            TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                              reinterpret_cast<std::uintptr_t>(value->data));
          }
          ASSERT_TRUE(kg::CheckFlashAttnOwners(in));
          auto invalid = in;
          invalid.logical_cohort = 4;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.owner_offset = 1;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.logit_softcap = 50;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.bounded_roots = false;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.k[1] = in.k[0];
          invalid.v[1] = in.v[0];
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          for (const std::int64_t bad : {255, 513, 16640}) {
            auto descriptor = *in.k[short_owner];
            descriptor.ne[1] = bad;
            invalid = in;
            invalid.k[short_owner] = &descriptor;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          auto narrow_mask = *in.mask;
          narrow_mask.ne[0] = short_cells;
          invalid = in;
          invalid.mask = &narrow_mask;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          auto* physical_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
          auto* physical_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
          auto* physical_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
          auto* k = ggml_permute(ctx, physical_k, 0, 2, 1, 3);
          auto* v = ggml_permute(ctx, physical_v, 0, 2, 1, 3);
          TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(physical_k->data));
          TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(physical_v->data));
          auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, physical_mask, 1, 0, 0));
          ggml_prec_set_acc(whole, GGML_PREC_F32);
          auto plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                               : kg::PlanFlashAttnMma(launch(), whole);
          auto owners = kg::PlanFlashAttnOwners(launch(), in);
          ASSERT_TRUE(plan);
          ASSERT_TRUE(owners);
          EXPECT_EQ(plan->columns, d == 256 ? 4 : 1);
          EXPECT_EQ(owners->original.columns, plan->columns);
          EXPECT_EQ(owners->cohort_blocks, plan->blocks);
          EXPECT_EQ(owners->original.mask_prepass, plan->mask_prepass);
          EXPECT_TRUE(plan->mask_prepass);
          const auto shape = d == 256
                                 ? kg::detail::FlashAttnMmaShapeGqa2(4, launch().device(), false)
                                 : kg::detail::FlashAttnMmaShape512(1, false, launch().device());
          ASSERT_TRUE(shape);
          EXPECT_EQ(owners->original_blocks_per_sm, shape->blocks_per_sm);
          EXPECT_GT(owners->owner_blocks_per_sm, 0);
          EXPECT_EQ(owners->original.scratch, plan->scratch);
          EXPECT_EQ(256 % shape->kv_batch, 0);
          int empty_completion = 0, empty_fixup = 0;
          const int iter = static_cast<int>(cells) / shape->kv_batch;
          const int logical_tiles = static_cast<int>(kvh * 2);
          for (int block = 0; block < plan->blocks; ++block) {
            int first = block * iter * logical_tiles / plan->blocks;
            const int stop = (block + 1) * iter * logical_tiles / plan->blocks;
            int begin = first % iter;
            int end = std::min(iter, begin + stop - first);
            while (first < stop && end == iter) {
              const auto owner = static_cast<std::size_t>(first / (iter * kvh));
              empty_completion += begin >= actual_cells(owner) / shape->kv_batch;
              first += iter;
              first -= first % iter;
              begin = 0;
              end = std::min(iter, stop - first);
            }
            if (first < stop) {
              const auto owner = static_cast<std::size_t>(first / (iter * kvh));
              empty_fixup += begin >= actual_cells(owner) / shape->kv_batch;
            }
          }
          EXPECT_GT(empty_completion, 0);
          empty_fixups_by_specialization[d == 256 ? 0U : 1U] += empty_fixup;
          const auto scratch = std::max(plan->scratch, owners->original.scratch);
          const auto workspace = Allocate(scratch);
          auto paid = LaunchContext::Create(launch().device(), *execution_, stream_,
                                            {.base = workspace, .size = Bytes(scratch)});
          ASSERT_TRUE(paid);
          auto short_paid = LaunchContext::Create(
              launch().device(), *execution_, stream_,
              {.base = workspace, .size = Bytes(owners->original.scratch - 1)});
          ASSERT_TRUE(short_paid);
          EXPECT_FALSE(kg::FlashAttnOwnerRoots(**short_paid, in));
          const auto submission = execution_->Submission(stream_);
          ASSERT_TRUE(submission);
          const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
          const auto run = [&](LaunchContext& context) -> std::expected<void, KernelFailure> {
            if (auto x = d == 256 ? kg::FlashAttnMmaGqa2(context, whole)
                                  : kg::FlashAttnMma(context, whole);
                !x)
              return x;
            if (cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0xA5, scratch, stream) !=
                    cudaSuccess ||
                cudaMemsetAsync(in.output->data, 0xFF, ggml_nbytes(in.output), stream) !=
                    cudaSuccess)
              return std::unexpected(
                  KernelFailure{.error = kg::KernelError::kUnknown, .detail = "poison refused"});
            return kg::FlashAttnOwnerRoots(context, in);
          };
          std::vector<float> prior;
          std::optional<kg::CapturedGraph> graph;
          for (std::size_t pass = 0; pass < 3; ++pass) {
            if (pass != 0) {
              const auto begin = pass == 1 ? 0U : begin_offset(1);
              const auto end = pass == 1 ? n(d * kvh * actual_cells(0)) : kdata.size();
              for (std::size_t i = begin; i < end; ++i) {
                kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * 0.75F);
                vdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(vdata[i]) + 0.125F);
              }
            }
            std::vector<ggml_fp16_t> pk(n(d * kvh * cells * 2), 0), pv(pk.size(), 0);
            std::vector<ggml_fp16_t> mask(n(cells * 32 * 2), ggml_fp32_to_fp16(-INFINITY));
            for (std::size_t owner = 0; owner < 2; ++owner) {
              const auto count = actual_cells(owner);
              const auto begin = begin_offset(owner);
              const auto length = n(d * kvh * count);
              const auto prefix = [&](const auto& data) {
                return std::span(data).subspan(begin, length);
              };
              const auto copy = [&](ggml_tensor* tensor, auto data) {
                return cudaMemcpyAsync(tensor->data, data.data(), data.size_bytes(),
                                       cudaMemcpyHostToDevice, stream);
              };
              ASSERT_EQ(copy(raw_k[owner], prefix(kdata)), cudaSuccess);
              ASSERT_EQ(copy(raw_v[owner], prefix(vdata)), cudaSuccess);
              std::ranges::copy(prefix(kdata), pk.begin() + static_cast<std::ptrdiff_t>(
                                                                owner * n(d * kvh * cells)));
              std::ranges::copy(prefix(vdata), pv.begin() + static_cast<std::ptrdiff_t>(
                                                                owner * n(d * kvh * cells)));
              const auto visible = visible_cells(owner) - static_cast<int>(pass);
              std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(owner * n(cells * 32)),
                          visible, 0);
            }
            ASSERT_EQ(cudaMemcpyAsync(physical_k->data, pk.data(), ggml_nbytes(physical_k),
                                      cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(physical_v->data, pv.data(), ggml_nbytes(physical_v),
                                      cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(physical_mask->data, mask.data(), ggml_nbytes(physical_mask),
                                      cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            auto poisoned_mask = mask;
            // Force the scan to admit the entire logical stream even though the short
            // owner has no physical cells there. All padded query rows are poisoned too.
            for (std::int64_t row = 0; row < 32; ++row)
              std::fill(poisoned_mask.begin() +
                            static_cast<std::ptrdiff_t>((short_owner * 32 + n(row)) * n(cells) +
                                                        n(short_cells)),
                        poisoned_mask.begin() +
                            static_cast<std::ptrdiff_t>((short_owner * 32 + n(row) + 1) * n(cells)),
                        ggml_fp32_to_fp16(0.0F));
            ASSERT_EQ(cudaMemcpyAsync(owner_mask->data, poisoned_mask.data(),
                                      ggml_nbytes(owner_mask), cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            ASSERT_TRUE(run(**paid));
            const auto eager = Download(in.output);
            const auto expected = Download(whole);
            ASSERT_EQ(std::memcmp(eager.data(), expected.data(), eager.size() * sizeof(float)), 0);
            ASSERT_TRUE(std::ranges::all_of(eager, [](float x) { return std::isfinite(x); }));
            std::vector<double> want(n(d * heads * 2));
            for (std::int64_t owner = 0; owner < 2; ++owner)
              for (std::int64_t head = 0; head < heads; ++head) {
                const auto visible = visible_cells(n(owner)) - static_cast<int>(pass);
                std::vector<double> scores(n(visible));
                double largest = -std::numeric_limits<double>::infinity();
                for (std::int64_t cell = 0; cell < visible; ++cell) {
                  double dot = 0;
                  for (std::int64_t i = 0; i < d; ++i)
                    dot += static_cast<double>(qdata[n((owner * heads + head) * d + i)]) *
                           ggml_fp16_to_fp32(
                               pk[n(((owner * cells + cell) * kvh + head / gqa) * d + i)]);
                  scores[n(cell)] = dot;
                  largest = std::max(largest, scores[n(cell)]);
                }
                double sum = 0;
                for (double score : scores) sum += std::exp(score - largest);
                for (std::int64_t cell = 0; cell < visible; ++cell)
                  for (std::int64_t i = 0; i < d; ++i)
                    want[n((owner * heads + head) * d + i)] +=
                        std::exp(scores[n(cell)] - largest) / sum *
                        ggml_fp16_to_fp32(
                            pv[n(((owner * cells + cell) * kvh + head / gqa) * d + i)]);
              }
            ExpectNmse(eager, want, kFlashAttnNmse, "Gemma4 bounded actual owner FP64");
            if (pass != 0) EXPECT_NE(eager, prior);
            if (!graph) {
              auto captured = (*paid)->Capture(run);
              ASSERT_TRUE(captured);
              graph.emplace(std::move(*captured));
            }
            for (int replay = 0; replay < 2; ++replay) {
              ASSERT_TRUE((*paid)->Launch(*graph));
              const auto actual = Download(in.output);
              EXPECT_EQ(std::memcmp(actual.data(), eager.data(), actual.size() * sizeof(float)), 0);
              EXPECT_EQ(Download<ggml_fp16_t>(owner_mask), poisoned_mask);
            }
            EXPECT_EQ(raw_k[0]->ne[2], actual_cells(0));
            EXPECT_EQ(raw_k[1]->ne[2], actual_cells(1));
            EXPECT_EQ(Download(raw_q), qdata);
            for (std::size_t owner = 0; owner < 2; ++owner) {
              const auto count = actual_cells(owner);
              const auto begin = begin_offset(owner);
              for (const auto& source :
                   {std::pair{raw_k[owner], &kdata}, std::pair{raw_v[owner], &vdata}})
                EXPECT_EQ(Download<ggml_fp16_t>(source.first),
                          std::vector<ggml_fp16_t>(
                              source.second->begin() + static_cast<std::ptrdiff_t>(begin),
                              source.second->begin() +
                                  static_cast<std::ptrdiff_t>(begin + n(d * kvh * count))));
              for (auto* parent : {parents_k[owner], parents_v[owner]}) {
                const auto retained = Download<ggml_fp16_t>(parent);
                EXPECT_TRUE(
                    std::ranges::all_of(std::span(retained).subspan(n(d * kvh * count)),
                                        [](ggml_fp16_t x) { return x == ggml_fp32_to_fp16(NAN); }));
              }
            }
            std::cout << "GEMMA4_BOUNDED_ROOTS heads=" << heads << " d=" << d << " pass=" << pass
                      << " actual0=" << actual_cells(0) << " actual1=" << actual_cells(1)
                      << " common_cells=" << cells << " empty_completion=" << empty_completion
                      << " empty_fixup=" << empty_fixup
                      << " bounded_occupancy=" << owners->owner_blocks_per_sm
                      << " columns=" << plan->columns << " blocks=" << plan->blocks
                      << " scan=" << plan->mask_prepass << " scratch=" << scratch << '\n';
            prior = eager;
          }
        }
    }
  // Heads are runtime inputs to the same D256/D512 specializations. Some
  // original grids have no absent final partial tile; require both banks across
  // the actual unchanged partitions rather than inventing a different grid.
  EXPECT_GT(empty_fixups_by_specialization[0], 0);
  EXPECT_GT(empty_fixups_by_specialization[1], 0);
}

void GgmlExtOpsTest::SmallOwnerControl(std::uint32_t cap, std::uint32_t small_owners) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  ASSERT_GT(sms, 0);
  for (const std::int64_t heads : {8, 16, 32})
    for (const std::int64_t owners : {2, 3, 4})
      for (const std::int64_t d : {256, 512})
        for (const std::int64_t cells : {256, 512, 1024, 4352}) {
          if (small_owners ? (heads != 8 || owners != small_owners || d != 256 ||
                              (cells != 256 && cells != 1024))
                           : owners == 4)
            continue;
          if (cap != 0 && (heads != 8 || owners != 2 || d != 256)) continue;
          if (heads == 8 ? (d != 256 || (!small_owners && owners != 2)) : cells != 1024) continue;
          SCOPED_TRACE(std::to_string(heads) + "/" + std::to_string(d) + "/" +
                       std::to_string(cells));

          const auto kvh = heads / (d == 256 ? 2 : 8);
          const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
          auto arena = TensorArena::Create(128).value();
          auto* ctx = arena.context();
          auto qdata = Normal(12101, n(d * heads * owners), cap == 0 ? 0.25F : 8.0F);
          auto* packed_q =
              Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners), qdata);
          auto* q = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
          TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(packed_q->data));
          const auto kdata =
              Halves(Normal(12102, n(d * kvh * cells * owners), cap == 0 ? 0.25F : 4.0F));
          const auto vdata = Halves(Normal(12103, n(d * kvh * cells * owners), 0.25F));
          auto* packed_k =
              Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
          auto* packed_v =
              Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
          auto* k = ggml_permute(ctx, packed_k, 0, 2, 1, 3);
          auto* v = ggml_permute(ctx, packed_v, 0, 2, 1, 3);
          TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(packed_k->data));
          TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(packed_v->data));
          std::vector<float> masks(n(cells * 32 * owners), -std::numeric_limits<float>::infinity());
          for (std::int64_t owner = 0; owner < owners; ++owner)
            for (std::int64_t cell = 0; cell < cells - 37 - owner * 3; ++cell)
              masks[n(owner * cells * 32 + cell)] = 0;
          auto* mask =
              Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, owners), Halves(masks));
          auto* whole =
              Place(ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, static_cast<float>(cap)));
          ggml_prec_set_acc(whole, GGML_PREC_F32);
          const auto full_plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                                          : kg::PlanFlashAttnMma(launch(), whole);
          ASSERT_TRUE(full_plan) << jitllm::test_support::Failed(full_plan)->detail;

          EXPECT_EQ(full_plan->columns, d == 256 ? 4 : 1);
          EXPECT_EQ(full_plan->group, d == 256 ? 2 : 8);
          EXPECT_FALSE(full_plan->sparse);
          EXPECT_TRUE(full_plan->mask_prepass);
          const auto run_whole = [&](LaunchContext& l) {
            return d == 256 ? kg::FlashAttnMmaGqa2(l, whole) : kg::FlashAttnMma(l, whole);
          };
          std::array<kg::FlashAttnOwners, 1> quads;
          std::array<kg::FlashAttnOwnersPlan, 1> plans;
          auto scratch = full_plan->scratch;
          for (std::size_t quad = 0; quad < quads.size(); ++quad) {
            const auto first = quad * 4;
            const auto view = [&](ggml_tensor* tensor, std::array<std::int64_t, 4> ne) {
              const auto offset = first * tensor->nb[3];
              auto* slice = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3], tensor->nb[1],
                                         tensor->nb[2], tensor->nb[3], offset);
              TensorArena::Bind(slice, reinterpret_cast<std::uintptr_t>(tensor->data) + offset);
              return slice;
            };
            auto& in = quads[quad];
            in.q = view(q, {d, 1, heads, owners});
            in.mask = view(mask, {cells, 32, 1, owners});
            in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners));
            in.logit_softcap = cap;
            in.logical_cohort = static_cast<std::uint32_t>(owners);
            in.owner_count = static_cast<std::uint32_t>(owners);
            for (std::size_t owner = 0; owner < n(owners); ++owner) {
              const auto part = n(d * kvh * cells);
              const auto begin = static_cast<std::ptrdiff_t>((first + owner) * part);
              const auto end = begin + static_cast<std::ptrdiff_t>(part);
              // Separate allocations authenticate actual roots; their bytes match
              // the corresponding planes of the contiguous physical-stream control.
              auto* raw_k =
                  Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                        std::vector<ggml_fp16_t>(kdata.begin() + begin, kdata.begin() + end));
              auto* raw_v =
                  Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                        std::vector<ggml_fp16_t>(vdata.begin() + begin, vdata.begin() + end));
              in.k[owner] = ggml_permute(ctx, raw_k, 0, 2, 1, 3);
              in.v[owner] = ggml_permute(ctx, raw_v, 0, 2, 1, 3);
              TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                                reinterpret_cast<std::uintptr_t>(raw_k->data));
              TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                                reinterpret_cast<std::uintptr_t>(raw_v->data));
            }
            const auto plan = kg::PlanFlashAttnOwners(launch(), in);
            ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
            EXPECT_EQ(plan->effective_cohort, static_cast<std::uint32_t>(owners));
            EXPECT_EQ(plan->original.head, full_plan->head);
            EXPECT_EQ(plan->original.columns, full_plan->columns);
            EXPECT_EQ(plan->original.group, full_plan->group);
            EXPECT_EQ(plan->original.mask_prepass, full_plan->mask_prepass);
            EXPECT_FALSE(plan->original.sparse);
            EXPECT_GT(plan->original_blocks_per_sm, 0);
            if (d == 256) {
              auto actual_shape = kg::detail::FlashAttnMmaShapeGqa2(4, launch().device(), cap != 0);
              ASSERT_TRUE(actual_shape);
              EXPECT_EQ(plan->original_blocks_per_sm, actual_shape->blocks_per_sm);
            }
            EXPECT_GT(plan->owner_blocks_per_sm, 0);
            EXPECT_EQ(plan->cohort_blocks, full_plan->blocks);
            EXPECT_EQ(plan->original.blocks, full_plan->blocks);
            if (small_owners) EXPECT_EQ(plan->original.scratch, full_plan->scratch);
            const auto quad_tiles = static_cast<int>(kvh * owners);
            EXPECT_EQ(plan->original.blocks % quad_tiles == 0, full_plan->blocks % quad_tiles == 0);
            std::cout << "OWNER_SMALL_PLAN cap=" << cap << " owners=" << owners
                      << " heads=" << heads << " D=" << d << " cells=" << cells << " sms=" << sms
                      << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                      << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                      << " full_blocks=" << full_plan->blocks
                      << " quad_blocks=" << plan->original.blocks
                      << " effective_cohort=" << plan->effective_cohort
                      << " scratch=" << plan->original.scratch << '\n';
            auto invalid = in;
            if (owners < 4) {
              invalid.k[n(owners)] = in.k[0];
              EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
            }
            invalid = in;
            invalid.v[n(owners) - 1] = nullptr;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
            invalid = in;
            invalid.logical_cohort = owners == 2 ? 3U : 2U;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
            for (const auto bad : {0U, 1U, 5U}) {
              invalid = in;
              invalid.owner_count = bad;
              EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
            }
            if (cap != 0 || small_owners != 0) {
              for (const auto bad : {1U, 25U, 51U, UINT32_MAX}) {
                invalid = in;
                invalid.logit_softcap = bad;
                EXPECT_FALSE(kg::PlanFlashAttnOwners(launch(), invalid));
              }
              std::array<ggml_tensor*, 4> roots_k{}, roots_v{};
              for (std::size_t owner = 0; owner < n(owners); ++owner) {
                roots_k[owner] = const_cast<ggml_tensor*>(in.k[owner]);
                roots_v[owner] = const_cast<ggml_tensor*>(in.v[owner]);
              }
              auto* encoded = Place(kg::FlashAttnOwnersNode(
                  ctx, const_cast<ggml_tensor*>(in.q), const_cast<ggml_tensor*>(in.mask), roots_k,
                  roots_v, static_cast<std::uint32_t>(owners), static_cast<std::uint32_t>(owners),
                  0, cap));
              auto decoded = kg::FlashAttnOwnersFromNode(encoded);
              ASSERT_TRUE(decoded);
              EXPECT_EQ(decoded->logit_softcap, cap);
              EXPECT_EQ(decoded->owner_count, static_cast<std::uint32_t>(owners));
              EXPECT_EQ(decoded->logical_cohort, static_cast<std::uint32_t>(owners));
              EXPECT_TRUE(kg::CheckFlashAttnOwners(*decoded));
            }
            plans[quad] = *plan;
            scratch = std::max(scratch, plan->original.scratch);
          }
          // The original and independent-root calls reuse one explicitly funded
          // workspace; do not borrow the fixture's larger pool for this proof.
          auto context = LaunchContext::Create(launch().device(), *execution_, stream_,
                                               {.base = Allocate(scratch), .size = Bytes(scratch)});
          ASSERT_TRUE(context) << jitllm::test_support::Failed(context)->detail;
          auto& bounded = **context;
          if ((cap != 0 || small_owners != 0) && plans[0].original.scratch > 0) {
            const auto owner_scratch = plans[0].original.scratch;
            auto short_pool = LaunchContext::Create(
                launch().device(), *execution_, stream_,
                {.base = Allocate(owner_scratch), .size = Bytes(owner_scratch - 1)});
            ASSERT_TRUE(short_pool);
            EXPECT_EQ(FailedCode(kg::FlashAttnOwnerRoots(**short_pool, quads[0])),
                      KernelError::kRejected);
            EXPECT_FALSE((*short_pool)->faulted());
          }
          EXPECT_EQ(bounded.workspace().size.value(), scratch);
          EXPECT_TRUE(bounded.UsesStream(*execution_, stream_));
          const auto submission = execution_->Submission(stream_);
          ASSERT_TRUE(submission);
          const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
          ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
          for (std::size_t quad = 0; quad < quads.size(); ++quad) {
            ASSERT_EQ(cudaMemsetAsync(quads[quad].output->data, 0xFF,
                                      ggml_nbytes(quads[quad].output), stream),
                      cudaSuccess);
            bounded.ResetScratchPeak();
            ASSERT_TRUE(kg::FlashAttnOwnerRoots(bounded, quads[quad]));
            EXPECT_LE(bounded.scratch_peak().value(), plans[quad].original.scratch);
          }
          bounded.ResetScratchPeak();
          ASSERT_TRUE(run_whole(bounded));
          EXPECT_LE(bounded.scratch_peak().value(), full_plan->scratch);
          auto expected = Download(whole);
          EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
          const auto compare = [&] {
            for (std::size_t quad = 0; quad < quads.size(); ++quad) {
              const auto actual = Download(quads[quad].output);
              const auto offset = quad * actual.size();
              EXPECT_EQ(std::memcmp(actual.data(), expected.data() + offset,
                                    actual.size() * sizeof(float)),
                        0);
            }
          };
          if (heads == 8) {
            // Scalar GQA2 reference on exactly the uploaded F16 cache values.
            // The ordinary and independent-root paths must also be byte-exact.
            const auto keys = Widen(kdata), values = Widen(vdata);
            std::vector<double> want(n(d * heads * owners));
            for (std::int64_t owner = 0; owner < owners; ++owner)
              for (std::int64_t head = 0; head < heads; ++head) {
                const auto visible = cells - 37 - owner * 3;
                std::vector<double> scores(n(visible));
                for (std::int64_t cell = 0; cell < visible; ++cell)
                  for (std::int64_t col = 0; col < d; ++col)
                    scores[n(cell)] += double(qdata[n((owner * heads + head) * d + col)]) *
                                       keys[n(((owner * cells + cell) * kvh + head / 2) * d + col)];
                if (cap != 0)
                  for (auto& score : scores) score = cap * std::tanh(score / cap);
                const auto peak = *std::max_element(scores.begin(), scores.end());
                double total = 0;
                for (auto& score : scores) {
                  score = std::exp(score - peak);
                  total += score;
                }
                for (std::int64_t col = 0; col < d; ++col)
                  for (std::int64_t cell = 0; cell < visible; ++cell)
                    want[n((owner * heads + head) * d + col)] +=
                        scores[n(cell)] / total *
                        values[n(((owner * cells + cell) * kvh + head / 2) * d + col)];
              }
            ExpectNmse(expected, want, kFlashAttnNmse, "H8 packed/independent FP64");
            ExpectNmse(Download(quads[0].output), want, kFlashAttnNmse, "H8 owners FP64");
          }
          compare();
          bounded.ResetScratchPeak();
          auto graph = bounded.Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
            if (auto r = run_whole(l); !r) return r;
            for (const auto& quad : quads)
              if (auto r = kg::FlashAttnOwnerRoots(l, quad); !r) return r;
            return {};
          });
          ASSERT_TRUE(graph) << jitllm::test_support::Failed(graph)->detail;
          EXPECT_LE(bounded.scratch_peak().value(), scratch);
          for (int replay = 0; replay < 2; ++replay) {
            SCOPED_TRACE("capture replay " + std::to_string(replay));
            ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
            for (const auto& quad : quads)
              ASSERT_EQ(cudaMemsetAsync(quad.output->data, 0xFF, ggml_nbytes(quad.output), stream),
                        cudaSuccess);
            ASSERT_TRUE(bounded.Launch(*graph));
            EXPECT_EQ(std::memcmp(Download(whole).data(), expected.data(),
                                  expected.size() * sizeof(float)),
                      0);
            compare();
          }
          if (heads == 8) {
            // Replay must read fresh Q and masks from the same stable addresses.
            qdata = Normal(12104, qdata.size(), cap == 0 ? 0.25F : 8.0F);
            for (std::int64_t owner = 0; owner < owners; ++owner)
              masks[n(owner * cells * 32 + cells - 38 - owner * 3)] =
                  -std::numeric_limits<float>::infinity();
            const auto fresh_masks = Halves(masks);
            ASSERT_EQ(cudaMemcpyAsync(packed_q->data, qdata.data(), qdata.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            ASSERT_EQ(cudaMemcpyAsync(mask->data, fresh_masks.data(), fresh_masks.size() * 2,
                                      cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
            ASSERT_TRUE(run_whole(bounded));
            const auto fresh_expected = Download(whole);
            ASSERT_TRUE(
                std::ranges::all_of(fresh_expected, [](float x) { return std::isfinite(x); }));
            EXPECT_NE(std::memcmp(fresh_expected.data(), expected.data(), expected.size() * 4), 0);
            expected = fresh_expected;
            ASSERT_TRUE(bounded.Launch(*graph));
            EXPECT_EQ(std::memcmp(Download(whole).data(), expected.data(), expected.size() * 4), 0);
            compare();
            EXPECT_EQ(std::memcmp(Download(packed_q).data(), qdata.data(), qdata.size() * 4), 0);
            const auto downloaded_mask = Download<ggml_fp16_t>(mask);
            EXPECT_EQ(downloaded_mask, fresh_masks);
            EXPECT_EQ(Download<ggml_fp16_t>(packed_k), kdata);
            EXPECT_EQ(Download<ggml_fp16_t>(packed_v), vdata);
          }
          EXPECT_FALSE(bounded.faulted());
          Finish();
        }
}

TEST_F(GgmlExtOpsTest, Gemma2PackedPrefillMatchesTheSoftcappedPhysicalStream) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 softcap50 packed prefill control";
  constexpr std::int64_t d = 256, heads = 8, kvh = 4, rows = 128, owners = 2;
  const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  for (const std::int64_t cells : {512, 1024}) {
    auto arena = TensorArena::Create(96).value();
    auto* ctx = arena.context();
    auto qdata = Normal(731, n(d * heads * rows * owners), 0.75F);
    auto kdata = Halves(Normal(732, n(d * kvh * cells * owners), 1.5F));
    auto vdata = Halves(Normal(733, n(d * kvh * cells * owners), 0.25F));
    auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, rows, owners), qdata);
    auto* raw_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
    auto* raw_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
    const auto permute = [&](ggml_tensor* raw) {
      auto* t = ggml_permute(ctx, raw, 0, 2, 1, 3);
      TensorArena::Bind(t, reinterpret_cast<std::uintptr_t>(raw->data));
      return t;
    };
    auto* q = permute(raw_q);
    auto* k = permute(raw_k);
    auto* v = permute(raw_v);
    std::vector<ggml_fp16_t> mask_data(n(cells * rows * owners), 0xFC00);
    const auto fresh_mask = [&](std::int64_t past) {
      std::fill(mask_data.begin(), mask_data.end(), 0xFC00);
      for (std::int64_t s = 0; s < owners; ++s)
        for (std::int64_t r = 0; r < rows; ++r)
          std::fill_n(mask_data.begin() + static_cast<std::ptrdiff_t>((s * rows + r) * cells),
                      past + r + 1, ggml_fp16_t{0});
    };
    fresh_mask(0);
    auto* mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, rows, 1, owners), mask_data);
    const auto node = [&](ggml_tensor* tq, ggml_tensor* tk, ggml_tensor* tv, ggml_tensor* tm) {
      auto* out = Place(ggml_flash_attn_ext(ctx, tq, tk, tv, tm, 1, 0, 50));
      ggml_prec_set_acc(out, GGML_PREC_F32);
      return out;
    };
    auto* packed = node(q, k, v, mask);
    std::array<ggml_tensor*, 2> owner_k{}, owner_v{};
    const auto part = n(d * kvh * cells);
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const auto slice = [&](const auto& data) {
        return std::vector<ggml_fp16_t>(
            data.begin() + static_cast<std::ptrdiff_t>(owner * part),
            data.begin() + static_cast<std::ptrdiff_t>((owner + 1) * part));
      };
      owner_k[owner] =
          Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1), slice(kdata));
      owner_v[owner] =
          Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1), slice(vdata));
    }
    auto* joined_k = Place(ggml_concat(ctx, owner_k[0], owner_k[1], 3));
    auto* joined_v = Place(ggml_concat(ctx, owner_v[0], owner_v[1], 3));
    auto* candidate = node(q, permute(joined_k), permute(joined_v), mask);
    const auto full = kg::PlanFlashAttnMmaGqa2(launch(), packed);
    const auto single = kg::PlanFlashAttnMmaGqa2(launch(), candidate);
    ASSERT_TRUE(full && single);
    const auto shape = kg::detail::FlashAttnMmaShapeGqa2(32, launch().device(), true);
    ASSERT_TRUE(shape);
    EXPECT_EQ(full->columns, 32);
    EXPECT_EQ(single->columns, 32);
    EXPECT_TRUE(full->mask_prepass);
    EXPECT_TRUE(single->mask_prepass);
    EXPECT_EQ(full->blocks, single->blocks);
    EXPECT_EQ(full->scratch, single->scratch);
    EXPECT_GT(shape->blocks_per_sm, 0);
    const auto scratch = std::max(full->scratch, single->scratch);
    auto paid = LaunchContext::Create(0, *execution_, stream_,
                                      {.base = Allocate(scratch), .size = Bytes(scratch)});
    ASSERT_TRUE(paid);
    const auto run = [&](LaunchContext& launch) -> std::expected<void, KernelFailure> {
      if (auto result = kg::FlashAttnMmaGqa2(launch, packed); !result) return result;
      if (auto result = kg::Concat(launch, joined_k); !result) return result;
      if (auto result = kg::Concat(launch, joined_v); !result) return result;
      if (auto result = kg::FlashAttnMmaGqa2(launch, candidate); !result) return result;
      return {};
    };
    ASSERT_TRUE(run(**paid));
    auto eager_packed = Download(packed);
    const auto read_split = [&] { return Download(candidate); };
    auto eager_split = read_split();
    auto captured = (*paid)->Capture(run);
    ASSERT_TRUE(captured);
    std::vector<float> prior_packed, prior_split;
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) {
        qdata = Normal(734, qdata.size(), 0.75F);
        fresh_mask(32);
        for (std::size_t i = part; i < kdata.size(); ++i) {
          kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * 0.75F);
          vdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(vdata[i]) + 0.125F);
        }
        for (const auto& source : {std::pair{raw_k, &kdata}, std::pair{raw_v, &vdata}})
          ASSERT_EQ(cudaMemcpyAsync(source.first->data, source.second->data(),
                                    ggml_nbytes(source.first), cudaMemcpyHostToDevice, stream),
                    cudaSuccess);
        for (std::size_t owner = 0; owner < 2; ++owner)
          for (const auto& source :
               {std::pair{owner_k[owner], &kdata}, std::pair{owner_v[owner], &vdata}})
            ASSERT_EQ(cudaMemcpyAsync(source.first->data, source.second->data() + owner * part,
                                      ggml_nbytes(source.first), cudaMemcpyHostToDevice, stream),
                      cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(raw_q->data, qdata.data(), ggml_nbytes(raw_q),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(mask->data, mask_data.data(), ggml_nbytes(mask),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE(run(**paid));
        eager_packed = Download(packed);
        eager_split = read_split();
      }
      for (auto* output : {joined_k, joined_v, packed, candidate})
        ASSERT_EQ(cudaMemsetAsync(output->data, 0xFF, ggml_nbytes(output), stream), cudaSuccess);
      ASSERT_TRUE((*paid)->Launch(*captured));
      const auto whole = Download(packed);
      const auto parts = read_split();
      ASSERT_EQ(whole.size(), parts.size());
      EXPECT_EQ(std::memcmp(whole.data(), parts.data(), whole.size() * sizeof(float)), 0);
      EXPECT_EQ(std::memcmp(whole.data(), eager_packed.data(), whole.size() * sizeof(float)), 0);
      EXPECT_EQ(std::memcmp(parts.data(), eager_split.data(), parts.size() * sizeof(float)), 0);
      EXPECT_TRUE(std::ranges::all_of(whole, [](float x) { return std::isfinite(x); }));
      EXPECT_TRUE(std::ranges::all_of(parts, [](float x) { return std::isfinite(x); }));
      std::size_t changed = 0;
      float max_raw_difference = 0;
      for (std::size_t i = 0; i < whole.size(); ++i) {
        changed += std::bit_cast<std::uint32_t>(whole[i]) != std::bit_cast<std::uint32_t>(parts[i]);
        max_raw_difference = std::max(max_raw_difference, std::abs(whole[i] - parts[i]));
      }
      if (pass == 1) {
        EXPECT_NE(whole, prior_packed);
        EXPECT_NE(parts, prior_split);
      }
      std::vector<double> want(whole.size());
      std::vector<double> logits(n(cells));
      for (std::int64_t owner = 0; owner < owners; ++owner)
        for (std::int64_t r = 0; r < rows; ++r)
          for (std::int64_t h = 0; h < heads; ++h) {
            double maximum = -std::numeric_limits<double>::infinity();
            const auto visible = (pass == 0 ? 0 : 32) + r + 1;
            for (std::int64_t cell = 0; cell < visible; ++cell) {
              double dot = 0;
              for (std::int64_t i = 0; i < d; ++i)
                dot += static_cast<double>(qdata[n(((owner * rows + r) * heads + h) * d + i)]) *
                       ggml_fp16_to_fp32(kdata[n(((owner * cells + cell) * kvh + h / 2) * d + i)]);
              logits[n(cell)] = 50.0 * std::tanh(dot / 50.0);
              maximum = std::max(maximum, logits[n(cell)]);
            }
            double sum = 0;
            for (std::int64_t cell = 0; cell < visible; ++cell)
              sum += std::exp(logits[n(cell)] - maximum);
            for (std::int64_t cell = 0; cell < visible; ++cell) {
              const double probability = std::exp(logits[n(cell)] - maximum) / sum;
              for (std::int64_t i = 0; i < d; ++i)
                want[n(((owner * rows + r) * heads + h) * d + i)] +=
                    probability *
                    ggml_fp16_to_fp32(vdata[n(((owner * cells + cell) * kvh + h / 2) * d + i)]);
            }
          }
      ExpectNmse(whole, want, kFlashAttnNmse, "softcap50 physical C2 FP64");
      ExpectNmse(parts, want, kFlashAttnNmse, "softcap50 concatenated C2 FP64");
      ASSERT_TRUE((*paid)->Launch(*captured));
      const auto repeated_packed = Download(packed);
      EXPECT_EQ(std::memcmp(whole.data(), repeated_packed.data(), whole.size() * sizeof(float)), 0);
      const auto repeated = Download(candidate);
      EXPECT_EQ(std::memcmp(parts.data(), repeated.data(), parts.size() * sizeof(float)), 0);
      EXPECT_EQ(Download<ggml_fp16_t>(joined_k), kdata);
      EXPECT_EQ(Download<ggml_fp16_t>(joined_v), vdata);
      for (std::size_t owner = 0; owner < 2; ++owner) {
        EXPECT_NE(owner_k[owner]->data, joined_k->data);
        const auto slice = [&](const auto& data) {
          return std::vector<ggml_fp16_t>(
              data.begin() + static_cast<std::ptrdiff_t>(owner * part),
              data.begin() + static_cast<std::ptrdiff_t>((owner + 1) * part));
        };
        EXPECT_EQ(Download<ggml_fp16_t>(owner_k[owner]), slice(kdata));
        EXPECT_EQ(Download<ggml_fp16_t>(owner_v[owner]), slice(vdata));
      }
      EXPECT_EQ(Download(raw_q), qdata);
      EXPECT_EQ(Download<ggml_fp16_t>(raw_k), kdata);
      EXPECT_EQ(Download<ggml_fp16_t>(raw_v), vdata);
      EXPECT_EQ(Download<ggml_fp16_t>(mask), mask_data);
      std::cout << "GEMMA2_PREFILL_COMMON cells=" << cells << " pass=" << pass
                << " columns=" << full->columns << " sms=" << sms
                << " blocks_per_sm=" << shape->blocks_per_sm << " kv_batch=" << shape->kv_batch
                << " candidate_tiles=32 packed_tiles=32 candidate_blocks=" << single->blocks
                << " packed_blocks=" << full->blocks << " split_scan=" << single->mask_prepass
                << " packed_scan=" << full->mask_prepass << " raw_differences=" << changed
                << " max_raw=" << max_raw_difference << " scratch=" << scratch << '\n';
      prior_packed = whole;
      prior_split = parts;
    }
  }
}

TEST_F(GgmlExtOpsTest, Gemma2UnequalOwnerPaddingMatchesTheSoftcappedCommonStream) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 two-owner control";
  constexpr std::int64_t d = 256, heads = 8, kvh = 4, short_cells = 512, cells = 1024;
  const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
  auto arena = TensorArena::Create(96).value();
  auto* ctx = arena.context();
  auto qdata = Normal(741, n(d * heads * 2), 0.75F);
  auto kdata = Halves(Normal(742, n(d * kvh * (short_cells + cells)), 1.5F));
  auto vdata = Halves(Normal(743, n(d * kvh * (short_cells + cells)), 0.25F));
  auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2), qdata);
  auto* q = ggml_permute(ctx, raw_q, 0, 2, 1, 3);
  TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
  std::array<ggml_tensor*, 2> raw_k{}, raw_v{}, masks{};
  for (std::size_t owner = 0; owner < 2; ++owner) {
    const auto count = owner == 0 ? short_cells : cells;
    const auto begin = owner == 0 ? 0 : n(d * kvh * short_cells);
    const auto slice = [&](const auto& data) {
      return std::vector<ggml_fp16_t>(
          data.begin() + static_cast<std::ptrdiff_t>(begin),
          data.begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count)));
    };
    raw_k[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count, 1), slice(kdata));
    raw_v[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count, 1), slice(vdata));
    masks[owner] = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, count, 32, 1, 1));
  }
  const auto filled = [&](std::array<std::int64_t, 4> dims, float value) {
    auto* out = ggml_fill(ctx, ggml_new_tensor(ctx, GGML_TYPE_F16, 4, dims.data()), value);
    out->src[0] = nullptr;  // shape-only template; fill reads no source
    return Place(out);
  };
  auto* zeros = filled({d, kvh, cells - short_cells, 1}, 0);
  auto* invisible = filled({cells - short_cells, 32, 1, 1}, -INFINITY);
  auto* padded_k = Place(ggml_concat(ctx, raw_k[0], zeros, 2));
  auto* padded_v = Place(ggml_concat(ctx, raw_v[0], zeros, 2));
  auto* padded_mask = Place(ggml_concat(ctx, masks[0], invisible, 0));
  auto* owner_mask = Place(ggml_concat(ctx, padded_mask, masks[1], 3));
  kg::FlashAttnOwners in;
  in.q = q;
  in.mask = owner_mask;
  in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2));
  in.owner_count = in.logical_cohort = 2;
  in.logit_softcap = 50;
  for (std::size_t owner = 0; owner < 2; ++owner) {
    auto* key = owner == 0 ? padded_k : raw_k[1];
    auto* value = owner == 0 ? padded_v : raw_v[1];
    in.k[owner] = ggml_permute(ctx, key, 0, 2, 1, 3);
    in.v[owner] = ggml_permute(ctx, value, 0, 2, 1, 3);
    TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                      reinterpret_cast<std::uintptr_t>(key->data));
    TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                      reinterpret_cast<std::uintptr_t>(value->data));
  }
  auto* physical_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
  auto* physical_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
  auto* physical_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
  auto* k = ggml_permute(ctx, physical_k, 0, 2, 1, 3);
  auto* v = ggml_permute(ctx, physical_v, 0, 2, 1, 3);
  TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(physical_k->data));
  TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(physical_v->data));
  auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, physical_mask, 1, 0, 50));
  ggml_prec_set_acc(whole, GGML_PREC_F32);
  auto plan = kg::PlanFlashAttnMmaGqa2(launch(), whole);
  auto owners = kg::PlanFlashAttnOwners(launch(), in);
  ASSERT_TRUE(plan);
  ASSERT_TRUE(owners);
  EXPECT_EQ(plan->columns, 4);
  EXPECT_EQ(owners->original.columns, plan->columns);
  EXPECT_EQ(owners->cohort_blocks, plan->blocks);
  EXPECT_EQ(owners->original.mask_prepass, plan->mask_prepass);
  EXPECT_TRUE(plan->mask_prepass);
  const auto shape = kg::detail::FlashAttnMmaShapeGqa2(4, launch().device(), true);
  ASSERT_TRUE(shape);
  EXPECT_EQ(owners->original_blocks_per_sm, shape->blocks_per_sm);
  EXPECT_GT(owners->owner_blocks_per_sm, 0);
  const auto scratch = std::max(plan->scratch, owners->original.scratch);
  auto paid = LaunchContext::Create(launch().device(), *execution_, stream_,
                                    {.base = Allocate(scratch), .size = Bytes(scratch)});
  ASSERT_TRUE(paid);
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
  const auto run = [&](LaunchContext& context) -> std::expected<void, KernelFailure> {
    for (auto* fill : {zeros, invisible})
      if (auto x = kg::Fill(context, fill); !x) return x;
    for (auto* concat : {padded_k, padded_v, padded_mask, owner_mask})
      if (auto x = kg::Concat(context, concat); !x) return x;
    if (auto x = kg::FlashAttnMmaGqa2(context, whole); !x) return x;
    return kg::FlashAttnOwnerRoots(context, in);
  };
  std::vector<float> prior;
  std::optional<kg::CapturedGraph> graph;
  for (std::size_t pass = 0; pass < 3; ++pass) {
    if (pass != 0) {
      const auto begin = pass == 1 ? 0U : n(d * kvh * short_cells);
      const auto end = pass == 1 ? n(d * kvh * short_cells) : kdata.size();
      for (std::size_t i = begin; i < end; ++i) {
        kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * 0.75F);
        vdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(vdata[i]) + 0.125F);
      }
    }
    std::vector<ggml_fp16_t> pk(n(d * kvh * cells * 2), 0), pv(pk.size(), 0);
    std::vector<ggml_fp16_t> mask(n(cells * 32 * 2), ggml_fp32_to_fp16(-INFINITY));
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const auto count = owner == 0 ? short_cells : cells;
      const auto begin = owner == 0 ? 0U : n(d * kvh * short_cells);
      const auto length = n(d * kvh * count);
      const auto prefix = [&](const auto& data) { return std::span(data).subspan(begin, length); };
      const auto copy = [&](ggml_tensor* tensor, auto data) {
        return cudaMemcpyAsync(tensor->data, data.data(), data.size_bytes(), cudaMemcpyHostToDevice,
                               stream);
      };
      ASSERT_EQ(copy(raw_k[owner], prefix(kdata)), cudaSuccess);
      ASSERT_EQ(copy(raw_v[owner], prefix(vdata)), cudaSuccess);
      std::ranges::copy(prefix(kdata),
                        pk.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
      std::ranges::copy(prefix(vdata),
                        pv.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
      const auto visible = (owner == 0 ? 260 : 772) - static_cast<int>(pass);
      std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(owner * n(cells * 32)), visible, 0);
      std::vector<ggml_fp16_t> own_mask(n(count * 32), ggml_fp32_to_fp16(-INFINITY));
      std::fill_n(own_mask.begin(), visible, 0);
      ASSERT_EQ(copy(masks[owner], std::span(own_mask)), cudaSuccess);
    }
    ASSERT_EQ(cudaMemcpyAsync(physical_k->data, pk.data(), ggml_nbytes(physical_k),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(physical_v->data, pv.data(), ggml_nbytes(physical_v),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(physical_mask->data, mask.data(), ggml_nbytes(physical_mask),
                              cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    ASSERT_TRUE(run(**paid));
    const auto eager = Download(in.output);
    const auto expected = Download(whole);
    ASSERT_EQ(std::memcmp(eager.data(), expected.data(), eager.size() * sizeof(float)), 0);
    ASSERT_TRUE(std::ranges::all_of(eager, [](float x) { return std::isfinite(x); }));
    std::vector<double> want(n(d * heads * 2));
    for (std::int64_t owner = 0; owner < 2; ++owner)
      for (std::int64_t head = 0; head < heads; ++head) {
        const auto visible = (owner == 0 ? 260 : 772) - static_cast<int>(pass);
        std::vector<double> scores(n(visible));
        double largest = -std::numeric_limits<double>::infinity();
        for (std::int64_t cell = 0; cell < visible; ++cell) {
          double dot = 0;
          for (std::int64_t i = 0; i < d; ++i)
            dot += static_cast<double>(qdata[n((owner * heads + head) * d + i)]) *
                   ggml_fp16_to_fp32(pk[n(((owner * cells + cell) * kvh + head / 2) * d + i)]);
          scores[n(cell)] = 50.0 * std::tanh(dot / 50.0);
          largest = std::max(largest, scores[n(cell)]);
        }
        double sum = 0;
        for (double score : scores) sum += std::exp(score - largest);
        for (std::int64_t cell = 0; cell < visible; ++cell)
          for (std::int64_t i = 0; i < d; ++i)
            want[n((owner * heads + head) * d + i)] +=
                std::exp(scores[n(cell)] - largest) / sum *
                ggml_fp16_to_fp32(pv[n(((owner * cells + cell) * kvh + head / 2) * d + i)]);
      }
    ExpectNmse(eager, want, kFlashAttnNmse, "softcap50 unequal padded owner FP64");
    if (pass != 0) EXPECT_NE(eager, prior);
    if (!graph) {
      auto captured = (*paid)->Capture(run);
      ASSERT_TRUE(captured);
      graph.emplace(std::move(*captured));
    }
    for (int replay = 0; replay < 2; ++replay) {
      ASSERT_EQ(cudaMemsetAsync(zeros->data, 0xFF, ggml_nbytes(zeros), stream), cudaSuccess);
      ASSERT_EQ(cudaMemsetAsync(invisible->data, 0, ggml_nbytes(invisible), stream), cudaSuccess);
      ASSERT_TRUE((*paid)->Launch(*graph));
      const auto actual = Download(in.output);
      EXPECT_EQ(std::memcmp(actual.data(), eager.data(), actual.size() * sizeof(float)), 0);
      EXPECT_EQ(Download<ggml_fp16_t>(zeros),
                std::vector<ggml_fp16_t>(n(d * kvh * (cells - short_cells)), 0));
      EXPECT_EQ(Download<ggml_fp16_t>(owner_mask), mask);
      EXPECT_EQ(Download<ggml_fp16_t>(padded_k),
                std::vector<ggml_fp16_t>(
                    pk.begin(), pk.begin() + static_cast<std::ptrdiff_t>(n(d * kvh * cells))));
    }
    EXPECT_EQ(raw_k[0]->ne[2], short_cells);
    EXPECT_EQ(raw_k[1]->ne[2], cells);
    EXPECT_EQ(Download(raw_q), qdata);
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const auto count = owner == 0 ? short_cells : cells;
      const auto begin = owner == 0 ? 0U : n(d * kvh * short_cells);
      for (const auto& source : {std::pair{raw_k[owner], &kdata}, std::pair{raw_v[owner], &vdata}})
        EXPECT_EQ(
            Download<ggml_fp16_t>(source.first),
            std::vector<ggml_fp16_t>(
                source.second->begin() + static_cast<std::ptrdiff_t>(begin),
                source.second->begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count))));
      std::vector<ggml_fp16_t> original_mask(n(count * 32), ggml_fp32_to_fp16(-INFINITY));
      std::fill_n(original_mask.begin(), (owner == 0 ? 260 : 772) - static_cast<int>(pass), 0);
      EXPECT_EQ(Download<ggml_fp16_t>(masks[owner]), original_mask);
    }
    std::cout << "GEMMA2_UNEQUAL_PADDING pass=" << pass
              << " source_cells=512/1024 common_cells=1024"
              << " columns=" << plan->columns << " blocks=" << plan->blocks
              << " scan=" << plan->mask_prepass << " scratch=" << scratch << '\n';
    prior = eager;
  }
}

void GgmlExtOpsTest::BoundedOwnerControl(std::uint32_t cap) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "GB10 two-owner control";
  constexpr std::int64_t d = 256, heads = 8, kvh = 4;
  for (const auto widths : {std::array<std::int64_t, 2>{512, 1024}, {256, 1536}})
    for (const std::size_t short_owner : {0U, 1U}) {
      const auto short_cells = widths[0], cells = widths[1];
      const auto actual_cells = [&](std::size_t owner) {
        return owner == short_owner ? short_cells : cells;
      };
      const auto visible_cells = [&](std::size_t owner) {
        return owner == short_owner ? (short_cells == 512 ? 260 : 196)
                                    : static_cast<int>(cells) - 252;
      };
      const auto n = [](std::int64_t x) { return static_cast<std::size_t>(x); };
      auto arena = TensorArena::Create(128).value();
      auto* ctx = arena.context();
      auto qdata = Normal(741, n(d * heads * 2), 0.75F);
      auto kdata = Halves(Normal(742, n(d * kvh * (short_cells + cells)), 1.5F));
      auto vdata = Halves(Normal(743, n(d * kvh * (short_cells + cells)), 0.25F));
      auto* raw_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2), qdata);
      auto* q = ggml_permute(ctx, raw_q, 0, 2, 1, 3);
      TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
      std::array<ggml_tensor*, 2> raw_k{}, raw_v{}, parents_k{}, parents_v{};
      const auto begin_offset = [&](std::size_t owner) {
        return owner == 0 ? 0U : n(d * kvh * actual_cells(0));
      };
      for (std::size_t owner = 0; owner < 2; ++owner) {
        const auto count = actual_cells(owner);
        const auto begin = begin_offset(owner);
        const auto slice = [&](const auto& data) {
          return std::vector<ggml_fp16_t>(
              data.begin() + static_cast<std::ptrdiff_t>(begin),
              data.begin() + static_cast<std::ptrdiff_t>(begin + n(d * kvh * count)));
        };
        for (const bool value : {false, true}) {
          std::vector<ggml_fp16_t> guarded(n(d * kvh * (count + 256)), ggml_fp32_to_fp16(NAN));
          const auto source = slice(value ? vdata : kdata);
          std::ranges::copy(source, guarded.begin());
          auto* parent =
              Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, count + 256, 1), guarded);
          auto* view = ggml_view_4d(ctx, parent, d, kvh, count, 1, d * 2, d * kvh * 2,
                                    n(d * kvh * count * 2), 0);
          TensorArena::Bind(view, reinterpret_cast<std::uintptr_t>(parent->data));
          (value ? parents_v[owner] : parents_k[owner]) = parent;
          (value ? raw_v[owner] : raw_k[owner]) = view;
        }
      }
      auto* owner_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
      kg::FlashAttnOwners in;
      in.q = q;
      in.mask = owner_mask;
      in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, 2));
      in.owner_count = in.logical_cohort = 2;
      in.logit_softcap = cap;
      in.bounded_roots = true;
      for (std::size_t owner = 0; owner < 2; ++owner) {
        auto* key = raw_k[owner];
        auto* value = raw_v[owner];
        in.k[owner] = ggml_permute(ctx, key, 0, 2, 1, 3);
        in.v[owner] = ggml_permute(ctx, value, 0, 2, 1, 3);
        TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                          reinterpret_cast<std::uintptr_t>(key->data));
        TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                          reinterpret_cast<std::uintptr_t>(value->data));
      }
      auto* physical_k = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
      auto* physical_v = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 2));
      auto* physical_mask = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, 2));
      auto* k = ggml_permute(ctx, physical_k, 0, 2, 1, 3);
      auto* v = ggml_permute(ctx, physical_v, 0, 2, 1, 3);
      TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(physical_k->data));
      TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(physical_v->data));
      auto* whole =
          Place(ggml_flash_attn_ext(ctx, q, k, v, physical_mask, 1, 0, static_cast<float>(cap)));
      ggml_prec_set_acc(whole, GGML_PREC_F32);
      auto plan = kg::PlanFlashAttnMmaGqa2(launch(), whole);
      auto owners = kg::PlanFlashAttnOwners(launch(), in);
      ASSERT_TRUE(plan);
      ASSERT_TRUE(owners);
      EXPECT_EQ(plan->columns, 4);
      EXPECT_EQ(owners->original.columns, plan->columns);
      EXPECT_EQ(owners->cohort_blocks, plan->blocks);
      EXPECT_EQ(owners->original.mask_prepass, plan->mask_prepass);
      EXPECT_TRUE(plan->mask_prepass);
      const auto shape = kg::detail::FlashAttnMmaShapeGqa2(4, launch().device(), cap != 0);
      ASSERT_TRUE(shape);
      EXPECT_EQ(owners->original_blocks_per_sm, shape->blocks_per_sm);
      EXPECT_GT(owners->owner_blocks_per_sm, 0);
      EXPECT_EQ(owners->original.scratch, plan->scratch);
      EXPECT_EQ(shape->kv_batch, 64);
      int empty_completion = 0, empty_fixup = 0;
      const int iter = static_cast<int>(cells) / shape->kv_batch;
      const int logical_tiles = static_cast<int>(kvh * 2);
      for (int block = 0; block < plan->blocks; ++block) {
        int first = block * iter * logical_tiles / plan->blocks;
        const int stop = (block + 1) * iter * logical_tiles / plan->blocks;
        int begin = first % iter;
        int end = std::min(iter, begin + stop - first);
        while (first < stop && end == iter) {
          const auto owner = static_cast<std::size_t>(first / (iter * kvh));
          empty_completion += begin >= actual_cells(owner) / shape->kv_batch;
          first += iter;
          first -= first % iter;
          begin = 0;
          end = std::min(iter, stop - first);
        }
        if (first < stop) {
          const auto owner = static_cast<std::size_t>(first / (iter * kvh));
          empty_fixup += begin >= actual_cells(owner) / shape->kv_batch;
        }
      }
      EXPECT_GT(empty_completion, 0);
      EXPECT_GT(empty_fixup, 0);
      const auto scratch = std::max(plan->scratch, owners->original.scratch);
      const auto workspace = Allocate(scratch);
      auto paid = LaunchContext::Create(launch().device(), *execution_, stream_,
                                        {.base = workspace, .size = Bytes(scratch)});
      ASSERT_TRUE(paid);
      auto short_paid =
          LaunchContext::Create(launch().device(), *execution_, stream_,
                                {.base = workspace, .size = Bytes(owners->original.scratch - 1)});
      ASSERT_TRUE(short_paid);
      EXPECT_FALSE(kg::FlashAttnOwnerRoots(**short_paid, in));
      const auto submission = execution_->Submission(stream_);
      ASSERT_TRUE(submission);
      const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
      const auto run = [&](LaunchContext& context) -> std::expected<void, KernelFailure> {
        if (auto x = kg::FlashAttnMmaGqa2(context, whole); !x) return x;
        if (cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0xA5, scratch, stream) !=
                cudaSuccess ||
            cudaMemsetAsync(in.output->data, 0xFF, ggml_nbytes(in.output), stream) != cudaSuccess)
          return std::unexpected(
              KernelFailure{.error = kg::KernelError::kUnknown, .detail = "poison refused"});
        return kg::FlashAttnOwnerRoots(context, in);
      };
      std::vector<float> prior;
      std::optional<kg::CapturedGraph> graph;
      for (std::size_t pass = 0; pass < 3; ++pass) {
        if (pass != 0) {
          const auto begin = pass == 1 ? 0U : begin_offset(1);
          const auto end = pass == 1 ? n(d * kvh * actual_cells(0)) : kdata.size();
          for (std::size_t i = begin; i < end; ++i) {
            kdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(kdata[i]) * 0.75F);
            vdata[i] = ggml_fp32_to_fp16(ggml_fp16_to_fp32(vdata[i]) + 0.125F);
          }
        }
        std::vector<ggml_fp16_t> pk(n(d * kvh * cells * 2), 0), pv(pk.size(), 0);
        std::vector<ggml_fp16_t> mask(n(cells * 32 * 2), ggml_fp32_to_fp16(-INFINITY));
        for (std::size_t owner = 0; owner < 2; ++owner) {
          const auto count = actual_cells(owner);
          const auto begin = begin_offset(owner);
          const auto length = n(d * kvh * count);
          const auto prefix = [&](const auto& data) {
            return std::span(data).subspan(begin, length);
          };
          const auto copy = [&](ggml_tensor* tensor, auto data) {
            return cudaMemcpyAsync(tensor->data, data.data(), data.size_bytes(),
                                   cudaMemcpyHostToDevice, stream);
          };
          ASSERT_EQ(copy(raw_k[owner], prefix(kdata)), cudaSuccess);
          ASSERT_EQ(copy(raw_v[owner], prefix(vdata)), cudaSuccess);
          std::ranges::copy(prefix(kdata),
                            pk.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
          std::ranges::copy(prefix(vdata),
                            pv.begin() + static_cast<std::ptrdiff_t>(owner * n(d * kvh * cells)));
          const auto visible = visible_cells(owner) - static_cast<int>(pass);
          std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(owner * n(cells * 32)), visible,
                      0);
        }
        ASSERT_EQ(cudaMemcpyAsync(physical_k->data, pk.data(), ggml_nbytes(physical_k),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(physical_v->data, pv.data(), ggml_nbytes(physical_v),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(physical_mask->data, mask.data(), ggml_nbytes(physical_mask),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        auto poisoned_mask = mask;
        // Force the scan to admit the entire logical stream even though the short
        // owner has no physical cells there. All padded query rows are poisoned too.
        for (std::int64_t row = 0; row < 32; ++row)
          std::fill(
              poisoned_mask.begin() + static_cast<std::ptrdiff_t>(
                                          (short_owner * 32 + n(row)) * n(cells) + n(short_cells)),
              poisoned_mask.begin() +
                  static_cast<std::ptrdiff_t>((short_owner * 32 + n(row) + 1) * n(cells)),
              ggml_fp32_to_fp16(0.0F));
        ASSERT_EQ(cudaMemcpyAsync(owner_mask->data, poisoned_mask.data(), ggml_nbytes(owner_mask),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE(run(**paid));
        const auto eager = Download(in.output);
        const auto expected = Download(whole);
        ASSERT_EQ(std::memcmp(eager.data(), expected.data(), eager.size() * sizeof(float)), 0);
        ASSERT_TRUE(std::ranges::all_of(eager, [](float x) { return std::isfinite(x); }));
        std::vector<double> want(n(d * heads * 2));
        for (std::int64_t owner = 0; owner < 2; ++owner)
          for (std::int64_t head = 0; head < heads; ++head) {
            const auto visible = visible_cells(n(owner)) - static_cast<int>(pass);
            std::vector<double> scores(n(visible));
            double largest = -std::numeric_limits<double>::infinity();
            for (std::int64_t cell = 0; cell < visible; ++cell) {
              double dot = 0;
              for (std::int64_t i = 0; i < d; ++i)
                dot += static_cast<double>(qdata[n((owner * heads + head) * d + i)]) *
                       ggml_fp16_to_fp32(pk[n(((owner * cells + cell) * kvh + head / 2) * d + i)]);
              scores[n(cell)] = cap == 0 ? dot : double(cap) * std::tanh(dot / double(cap));
              largest = std::max(largest, scores[n(cell)]);
            }
            double sum = 0;
            for (double score : scores) sum += std::exp(score - largest);
            for (std::int64_t cell = 0; cell < visible; ++cell)
              for (std::int64_t i = 0; i < d; ++i)
                want[n((owner * heads + head) * d + i)] +=
                    std::exp(scores[n(cell)] - largest) / sum *
                    ggml_fp16_to_fp32(pv[n(((owner * cells + cell) * kvh + head / 2) * d + i)]);
          }
        ExpectNmse(eager, want, kFlashAttnNmse, "bounded actual owner FP64");
        if (pass != 0) EXPECT_NE(eager, prior);
        if (!graph) {
          auto captured = (*paid)->Capture(run);
          ASSERT_TRUE(captured);
          graph.emplace(std::move(*captured));
        }
        for (int replay = 0; replay < 2; ++replay) {
          ASSERT_TRUE((*paid)->Launch(*graph));
          const auto actual = Download(in.output);
          EXPECT_EQ(std::memcmp(actual.data(), eager.data(), actual.size() * sizeof(float)), 0);
          EXPECT_EQ(Download<ggml_fp16_t>(owner_mask), poisoned_mask);
        }
        EXPECT_EQ(raw_k[0]->ne[2], actual_cells(0));
        EXPECT_EQ(raw_k[1]->ne[2], actual_cells(1));
        EXPECT_EQ(Download(raw_q), qdata);
        for (std::size_t owner = 0; owner < 2; ++owner) {
          const auto count = actual_cells(owner);
          const auto begin = begin_offset(owner);
          for (const auto& source :
               {std::pair{raw_k[owner], &kdata}, std::pair{raw_v[owner], &vdata}})
            EXPECT_EQ(Download<ggml_fp16_t>(source.first),
                      std::vector<ggml_fp16_t>(
                          source.second->begin() + static_cast<std::ptrdiff_t>(begin),
                          source.second->begin() +
                              static_cast<std::ptrdiff_t>(begin + n(d * kvh * count))));
          for (auto* parent : {parents_k[owner], parents_v[owner]}) {
            const auto retained = Download<ggml_fp16_t>(parent);
            EXPECT_TRUE(
                std::ranges::all_of(std::span(retained).subspan(n(d * kvh * count)),
                                    [](ggml_fp16_t x) { return x == ggml_fp32_to_fp16(NAN); }));
          }
        }
        std::cout << "GEMMA_BOUNDED_ROOTS cap=" << cap << " pass=" << pass
                  << " actual0=" << actual_cells(0) << " actual1=" << actual_cells(1)
                  << " common_cells=" << cells << " empty_completion=" << empty_completion
                  << " empty_fixup=" << empty_fixup
                  << " bounded_occupancy=" << owners->owner_blocks_per_sm
                  << " columns=" << plan->columns << " blocks=" << plan->blocks
                  << " scan=" << plan->mask_prepass << " scratch=" << scratch << '\n';
        prior = eager;
      }
    }
}

TEST_F(GgmlExtOpsTest, Gemma2BoundedOwnerRootsMatchThePaddedOracleAndFp64) {
  BoundedOwnerControl(50);
}

TEST_F(GgmlExtOpsTest, Gemma3BoundedOwnerRootsMatchThePaddedOracleAndFp64) {
  BoundedOwnerControl(0);
}

TEST_F(GgmlExtOpsTest, Gemma3ThreeRealRootsMatchWholePhysicalStreamAndFp64) {
  SmallOwnerControl(0, 3);
}

TEST_F(GgmlExtOpsTest, Gemma3FourRealRootsMatchWholePhysicalStreamAndFp64) {
  SmallOwnerControl(0, 4);
}

TEST_F(GgmlExtOpsTest, TwoAndThreeRealRootsMatchWholePhysicalStreamMmaExactly) {
  SmallOwnerControl(0);
}

TEST_F(GgmlExtOpsTest, Gemma2SoftcapOwnersMatchWholePhysicalStreamAndFp64) {
  SmallOwnerControl(50);
}

TEST_F(GgmlExtOpsTest, Gemma3GroupedWholeAndPartialRootsMatchThePhysicalCohortAndFp64) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  ASSERT_GT(sms, 0);
  for (const std::int64_t owners : {5, 6, 7, 8, 9, 10, 11, 12})
    for (const std::int64_t heads : {8})
      for (const std::int64_t d : {256}) {
        constexpr std::int64_t cells = 512;
        const bool partial = kg::detail::PartialOwnerCohort(static_cast<std::uint32_t>(owners));
        SCOPED_TRACE(std::to_string(heads) + "/" + std::to_string(d) + "/" + std::to_string(cells));

        const auto kvh = heads / (d == 256 ? 2 : 8);
        const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
        auto arena = TensorArena::Create(128).value();
        auto* ctx = arena.context();
        auto qdata = Normal(12101, n(d * heads * owners), 0.25F);
        auto* packed_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners), qdata);
        auto* q = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
        TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(packed_q->data));
        const auto kdata = Halves(Normal(12102, n(d * kvh * cells * owners), 0.25F));
        const auto vdata = Halves(Normal(12103, n(d * kvh * cells * owners), 0.25F));
        auto* packed_k =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
        auto* packed_v =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
        auto* k = ggml_permute(ctx, packed_k, 0, 2, 1, 3);
        auto* v = ggml_permute(ctx, packed_v, 0, 2, 1, 3);
        TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(packed_k->data));
        TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(packed_v->data));
        std::vector<float> masks(n(cells * 32 * owners), -std::numeric_limits<float>::infinity());
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t cell = 0; cell < cells - 37 - owner * 3; ++cell)
            masks[n(owner * cells * 32 + cell)] = 0;
        auto* mask =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, owners), Halves(masks));
        auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, 0));
        ggml_prec_set_acc(whole, GGML_PREC_F32);
        const auto full_plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                                        : kg::PlanFlashAttnMma(launch(), whole);
        ASSERT_TRUE(full_plan) << jitllm::test_support::Failed(full_plan)->detail;

        EXPECT_EQ(full_plan->columns, d == 256 ? 4 : 1);
        EXPECT_EQ(full_plan->group, d == 256 ? 2 : 8);
        EXPECT_FALSE(full_plan->sparse);
        EXPECT_TRUE(full_plan->mask_prepass);
        const auto run_whole = [&](LaunchContext& l) {
          return d == 256 ? kg::FlashAttnMmaGqa2(l, whole) : kg::FlashAttnMma(l, whole);
        };
        std::array<kg::FlashAttnOwners, 3> quads;
        const auto groups = n((owners + 3) / 4);
        std::array<kg::FlashAttnOwnersPlan, 3> plans;
        auto scratch = full_plan->scratch;
        for (std::size_t quad = 0; quad < groups; ++quad) {
          const auto first = quad * 4;
          const auto active = std::min(std::size_t{4}, n(owners) - first);
          const auto view = [&](ggml_tensor* tensor, std::array<std::int64_t, 4> ne) {
            const auto offset = first * tensor->nb[3];
            auto* slice = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3], tensor->nb[1],
                                       tensor->nb[2], tensor->nb[3], offset);
            TensorArena::Bind(slice, reinterpret_cast<std::uintptr_t>(tensor->data) + offset);
            return slice;
          };
          auto& in = quads[quad];
          in.q = view(q, {d, 1, heads, static_cast<std::int64_t>(active)});
          in.mask = view(mask, {cells, 32, 1, static_cast<std::int64_t>(active)});
          in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1,
                                               static_cast<std::int64_t>(active)));
          in.logical_cohort = static_cast<std::uint32_t>(owners);
          in.owner_count = static_cast<std::uint32_t>(active);
          in.owner_offset = partial ? static_cast<std::uint32_t>(first) : 0;
          for (std::size_t owner = 0; owner < active; ++owner) {
            const auto part = n(d * kvh * cells);
            const auto begin = static_cast<std::ptrdiff_t>((first + owner) * part);
            const auto end = begin + static_cast<std::ptrdiff_t>(part);
            // Separate allocations authenticate actual roots; their bytes match
            // the corresponding planes of the contiguous physical-stream control.
            auto* raw_k =
                Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                      std::vector<ggml_fp16_t>(kdata.begin() + begin, kdata.begin() + end));
            auto* raw_v =
                Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                      std::vector<ggml_fp16_t>(vdata.begin() + begin, vdata.begin() + end));
            in.k[owner] = ggml_permute(ctx, raw_k, 0, 2, 1, 3);
            in.v[owner] = ggml_permute(ctx, raw_v, 0, 2, 1, 3);
            TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_k->data));
            TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_v->data));
          }
          const auto plan = kg::PlanFlashAttnOwners(launch(), in);
          ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
          EXPECT_EQ(plan->effective_cohort, static_cast<std::uint32_t>(owners));
          EXPECT_EQ(plan->original.head, full_plan->head);
          EXPECT_EQ(plan->original.columns, full_plan->columns);
          EXPECT_EQ(plan->original.group, full_plan->group);
          EXPECT_EQ(plan->original.mask_prepass, full_plan->mask_prepass);
          EXPECT_FALSE(plan->original.sparse);
          EXPECT_GT(plan->original_blocks_per_sm, 0);
          EXPECT_GT(plan->owner_blocks_per_sm, 0);
          EXPECT_EQ(plan->cohort_blocks, full_plan->blocks);
          EXPECT_EQ(plan->original.blocks * (partial ? 1 : static_cast<int>(groups)),
                    full_plan->blocks);
          const auto quad_tiles = static_cast<int>(kvh * (partial ? owners : 4));
          EXPECT_EQ(plan->original.blocks % quad_tiles == 0,
                    (partial ? full_plan->blocks : full_plan->blocks / static_cast<int>(groups)) %
                            quad_tiles ==
                        0);
          // Whole cohorts divide the original grid; partial cohorts filter
          // canonical global sequence offsets while retaining that whole grid.
          // Occupancy, KV batch and whole-tile preference can change across pins.
          const auto fixup = quad_tiles % plan->original.blocks == 0   ? "none"
                             : plan->original.blocks % quad_tiles == 0 ? "uniform"
                                                                       : "general";
          std::cout << "GEMMA3_GROUPED_PLAN owners=" << owners << " active=" << active
                    << " offset=" << first << " heads=" << heads << " D=" << d << " cells=" << cells
                    << " sms=" << sms << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                    << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                    << " full_blocks=" << full_plan->blocks
                    << " quad_blocks=" << plan->original.blocks
                    << " effective_cohort=" << plan->effective_cohort
                    << " scratch=" << plan->original.scratch << " fixup=" << fixup << '\n';
          auto invalid = in;
          if (active < 4) {
            invalid.k[active] = in.k[0];
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.v[active - 1] = nullptr;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          for (const auto bad : {0U, 1U, 2U, 3U, 4U, 5U}) {
            if (bad == active) continue;
            invalid = in;
            invalid.owner_count = bad;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          for (const auto bad : {1U, 3U, 5U, UINT32_MAX}) {
            invalid = in;
            invalid.owner_offset = bad;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.logical_cohort = 1;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.logit_softcap = 50;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.bounded_roots = true;
          EXPECT_EQ(kg::CheckFlashAttnOwners(invalid).has_value(), owners == 12);
          invalid = in;
          invalid.logical_cohort = 16;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          if (active > 1) {
            invalid = in;
            invalid.k[1] = invalid.k[0];
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          plans[quad] = *plan;
          scratch = std::max(scratch, plan->original.scratch);
        }
        // The original and independent-root calls reuse one explicitly funded
        // workspace; do not borrow the fixture's larger pool for this proof.
        const auto workspace = Allocate(scratch);
        auto context = LaunchContext::Create(launch().device(), *execution_, stream_,
                                             {.base = workspace, .size = Bytes(scratch)});
        ASSERT_TRUE(context) << jitllm::test_support::Failed(context)->detail;
        auto& bounded = **context;
        EXPECT_EQ(bounded.workspace().size.value(), scratch);
        EXPECT_TRUE(bounded.UsesStream(*execution_, stream_));
        const auto submission = execution_->Submission(stream_);
        ASSERT_TRUE(submission);
        const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
        ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
        for (std::size_t quad = 0; quad < groups; ++quad) {
          ASSERT_EQ(cudaMemsetAsync(quads[quad].output->data, 0xFF, ggml_nbytes(quads[quad].output),
                                    stream),
                    cudaSuccess);
          ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0xA5, scratch, stream),
                    cudaSuccess);
          bounded.ResetScratchPeak();
          ASSERT_TRUE(kg::FlashAttnOwnerRoots(bounded, quads[quad]));
          EXPECT_LE(bounded.scratch_peak().value(), plans[quad].original.scratch);
        }
        for (std::size_t group = 0; group < groups; ++group) {
          const auto bytes = plans[group].original.scratch;
          ASSERT_GT(bytes, 0U);
          auto short_context =
              LaunchContext::Create(launch().device(), *execution_, stream_,
                                    {.base = Allocate(bytes), .size = Bytes(bytes - 1)});
          ASSERT_TRUE(short_context);
          EXPECT_EQ(FailedCode(kg::FlashAttnOwnerRoots(**short_context, quads[group])),
                    KernelError::kRejected);
          EXPECT_FALSE((*short_context)->faulted());
        }
        bounded.ResetScratchPeak();
        ASSERT_TRUE(run_whole(bounded));
        EXPECT_LE(bounded.scratch_peak().value(), full_plan->scratch);
        auto expected = Download(whole);
        EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
        const auto compare = [&] {
          for (std::size_t quad = 0; quad < groups; ++quad) {
            const auto actual = Download(quads[quad].output);
            const auto offset = quad * 4 * n(d * heads);
            EXPECT_EQ(
                std::memcmp(actual.data(), expected.data() + offset, actual.size() * sizeof(float)),
                0);
          }
        };
        compare();
        const auto keys = Widen(kdata), values = Widen(vdata);
        std::vector<double> want(n(d * heads * owners));
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t head = 0; head < heads; ++head) {
            const auto visible = cells - 37 - owner * 3;
            std::vector<double> scores(n(visible));
            for (std::int64_t cell = 0; cell < visible; ++cell)
              for (std::int64_t col = 0; col < d; ++col)
                scores[n(cell)] += double(qdata[n((owner * heads + head) * d + col)]) *
                                   keys[n(((owner * cells + cell) * kvh + head / 2) * d + col)];
            const auto peak = *std::max_element(scores.begin(), scores.end());
            double total = 0;
            for (auto& score : scores) {
              score = std::exp(score - peak);
              total += score;
            }
            for (std::int64_t col = 0; col < d; ++col)
              for (std::int64_t cell = 0; cell < visible; ++cell)
                want[n((owner * heads + head) * d + col)] +=
                    scores[n(cell)] / total *
                    values[n(((owner * cells + cell) * kvh + head / 2) * d + col)];
          }
        ExpectNmse(expected, want, kFlashAttnNmse,
                   "H8 grouped physical FP64 C" + std::to_string(owners));
        for (std::size_t group = 0; group < groups; ++group) {
          const auto actual = Download(quads[group].output);
          const auto offset = group * 4 * n(d * heads);
          std::vector<double> part(
              want.begin() + static_cast<std::ptrdiff_t>(offset),
              want.begin() + static_cast<std::ptrdiff_t>(offset + actual.size()));
          ExpectNmse(actual, part, kFlashAttnNmse, "H8 group FP64 " + std::to_string(group));
        }
        bounded.ResetScratchPeak();
        auto graph = bounded.Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
          if (auto r = run_whole(l); !r) return r;
          for (const auto& quad : std::span(quads).first(groups)) {
            if (cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0xA5, scratch, stream) !=
                cudaSuccess)
              return std::unexpected(
                  KernelFailure{.error = KernelError::kUnknown, .detail = "scratch poison failed"});
            if (auto r = kg::FlashAttnOwnerRoots(l, quad); !r) return r;
          }
          return {};
        });
        ASSERT_TRUE(graph) << jitllm::test_support::Failed(graph)->detail;
        EXPECT_LE(bounded.scratch_peak().value(), scratch);
        for (int replay = 0; replay < 2; ++replay) {
          SCOPED_TRACE("capture replay " + std::to_string(replay));
          ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
          for (const auto& quad : std::span(quads).first(groups))
            ASSERT_EQ(cudaMemsetAsync(quad.output->data, 0xFF, ggml_nbytes(quad.output), stream),
                      cudaSuccess);
          ASSERT_TRUE(bounded.Launch(*graph));
          EXPECT_EQ(
              std::memcmp(Download(whole).data(), expected.data(), expected.size() * sizeof(float)),
              0);
          compare();
        }
        qdata = Normal(12104, qdata.size(), 0.25F);
        for (std::int64_t owner = 0; owner < owners; ++owner)
          masks[n(owner * cells * 32 + cells - 38 - owner * 3)] =
              -std::numeric_limits<float>::infinity();
        const auto fresh_masks = Halves(masks);
        ASSERT_EQ(cudaMemcpyAsync(packed_q->data, qdata.data(), qdata.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(mask->data, fresh_masks.data(), fresh_masks.size() * 2,
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE(run_whole(bounded));
        const auto fresh_expected = Download(whole);
        ASSERT_TRUE(std::ranges::all_of(fresh_expected, [](float x) { return std::isfinite(x); }));
        EXPECT_NE(
            std::memcmp(fresh_expected.data(), expected.data(), expected.size() * sizeof(float)),
            0);
        expected = fresh_expected;
        ASSERT_TRUE(bounded.Launch(*graph));
        EXPECT_EQ(
            std::memcmp(Download(whole).data(), expected.data(), expected.size() * sizeof(float)),
            0);
        compare();
        EXPECT_EQ(
            std::memcmp(Download(packed_q).data(), qdata.data(), qdata.size() * sizeof(float)), 0);
        EXPECT_EQ(Download<ggml_fp16_t>(mask), fresh_masks);
        EXPECT_EQ(Download<ggml_fp16_t>(packed_k), kdata);
        EXPECT_EQ(Download<ggml_fp16_t>(packed_v), vdata);
        for (std::size_t group = 0; group < groups; ++group)
          for (std::size_t owner = 0; owner < quads[group].owner_count; ++owner) {
            const auto part = n(d * kvh * cells), first = (group * 4 + owner) * part;
            EXPECT_EQ(Download<ggml_fp16_t>(quads[group].k[owner]),
                      std::vector<ggml_fp16_t>(
                          kdata.begin() + static_cast<std::ptrdiff_t>(first),
                          kdata.begin() + static_cast<std::ptrdiff_t>(first + part)));
            EXPECT_EQ(Download<ggml_fp16_t>(quads[group].v[owner]),
                      std::vector<ggml_fp16_t>(
                          vdata.begin() + static_cast<std::ptrdiff_t>(first),
                          vdata.begin() + static_cast<std::ptrdiff_t>(first + part)));
          }
        EXPECT_FALSE(bounded.faulted());
        Finish();
      }
}

TEST_F(GgmlExtOpsTest, Gemma3WholeTwelveBoundedRootsMatchPaddedCohortAndFp64) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  ASSERT_GT(sms, 0);
  for (const std::int64_t owners : {12})
    for (const std::int64_t heads : {8})
      for (const std::int64_t d : {256}) {
        constexpr std::int64_t cells = 1024;
        const auto actual_cells = [](std::int64_t owner) {
          // Mixed, all-short, all-long groups under ONE cohort-wide maximum.
          return (owner == 2 || owner == 3 || owner >= 8) ? 1024 : 512;
        };
        const bool partial = kg::detail::PartialOwnerCohort(static_cast<std::uint32_t>(owners));
        SCOPED_TRACE(std::to_string(heads) + "/" + std::to_string(d) + "/" + std::to_string(cells));

        const auto kvh = heads / (d == 256 ? 2 : 8);
        const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
        auto arena = TensorArena::Create(128).value();
        auto* ctx = arena.context();
        auto qdata = Normal(12101, n(d * heads * owners), 0.25F);
        auto* packed_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners), qdata);
        auto* q = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
        TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(packed_q->data));
        auto kdata = Halves(Normal(12102, n(d * kvh * cells * owners), 0.25F));
        auto vdata = Halves(Normal(12103, n(d * kvh * cells * owners), 0.25F));
        for (std::int64_t owner = 0; owner < owners; ++owner) {
          const auto first = n((owner * cells + actual_cells(owner)) * d * kvh);
          const auto stop = n((owner + 1) * cells * d * kvh);
          std::fill(kdata.begin() + static_cast<std::ptrdiff_t>(first),
                    kdata.begin() + static_cast<std::ptrdiff_t>(stop), 0);
          std::fill(vdata.begin() + static_cast<std::ptrdiff_t>(first),
                    vdata.begin() + static_cast<std::ptrdiff_t>(stop), 0);
        }
        auto* packed_k =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
        auto* packed_v =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
        auto* k = ggml_permute(ctx, packed_k, 0, 2, 1, 3);
        auto* v = ggml_permute(ctx, packed_v, 0, 2, 1, 3);
        TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(packed_k->data));
        TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(packed_v->data));
        std::vector<float> masks(n(cells * 32 * owners), -std::numeric_limits<float>::infinity());
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t cell = 0; cell < actual_cells(owner) - 37 - owner * 3; ++cell)
            masks[n(owner * cells * 32 + cell)] = 0;
        auto* mask =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, owners), Halves(masks));
        auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, 0));
        ggml_prec_set_acc(whole, GGML_PREC_F32);
        const auto full_plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                                        : kg::PlanFlashAttnMma(launch(), whole);
        ASSERT_TRUE(full_plan) << jitllm::test_support::Failed(full_plan)->detail;

        EXPECT_EQ(full_plan->columns, d == 256 ? 4 : 1);
        EXPECT_EQ(full_plan->group, d == 256 ? 2 : 8);
        EXPECT_FALSE(full_plan->sparse);
        EXPECT_TRUE(full_plan->mask_prepass);
        const auto run_whole = [&](LaunchContext& l) {
          return d == 256 ? kg::FlashAttnMmaGqa2(l, whole) : kg::FlashAttnMma(l, whole);
        };
        std::array<kg::FlashAttnOwners, 3> quads;
        const auto groups = n((owners + 3) / 4);
        std::array<kg::FlashAttnOwnersPlan, 3> plans;
        auto scratch = full_plan->scratch;
        for (std::size_t quad = 0; quad < groups; ++quad) {
          const auto first = quad * 4;
          const auto active = std::min(std::size_t{4}, n(owners) - first);
          const auto view = [&](ggml_tensor* tensor, std::array<std::int64_t, 4> ne) {
            const auto offset = first * tensor->nb[3];
            auto* slice = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3], tensor->nb[1],
                                       tensor->nb[2], tensor->nb[3], offset);
            TensorArena::Bind(slice, reinterpret_cast<std::uintptr_t>(tensor->data) + offset);
            return slice;
          };
          auto& in = quads[quad];
          in.q = view(q, {d, 1, heads, static_cast<std::int64_t>(active)});
          in.mask = view(mask, {cells, 32, 1, static_cast<std::int64_t>(active)});
          in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1,
                                               static_cast<std::int64_t>(active)));
          in.logical_cohort = static_cast<std::uint32_t>(owners);
          in.owner_count = static_cast<std::uint32_t>(active);
          in.owner_offset = partial ? static_cast<std::uint32_t>(first) : 0;
          in.bounded_roots = true;
          for (std::size_t owner = 0; owner < active; ++owner) {
            const auto part = n(d * kvh * cells);
            const auto begin = static_cast<std::ptrdiff_t>((first + owner) * part);
            const auto actual = actual_cells(static_cast<std::int64_t>(first + owner));
            const auto end = begin + static_cast<std::ptrdiff_t>(n(d * kvh * actual));
            const auto guarded = [&](const auto& data) {
              std::vector<ggml_fp16_t> values(data.begin() + begin, data.begin() + end);
              values.resize(n(d * kvh * (actual + 256)), ggml_fp32_to_fp16(NAN));
              return values;
            };
            // Actual views exclude a poisoned physical guard tile. The padded
            // packed oracle instead owns real zero bytes through logical1024.
            const auto root = [&](const auto& data) {
              auto* parent = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, actual + 256, 1),
                                   guarded(data));
              auto* view =
                  ggml_view_4d(ctx, parent, d, kvh, actual, 1, static_cast<std::size_t>(d * 2),
                               static_cast<std::size_t>(d * kvh * 2), n(d * kvh * actual * 2), 0);
              TensorArena::Bind(view, reinterpret_cast<std::uintptr_t>(parent->data));
              return view;
            };
            auto* raw_k = root(kdata);
            auto* raw_v = root(vdata);
            in.k[owner] = ggml_permute(ctx, raw_k, 0, 2, 1, 3);
            in.v[owner] = ggml_permute(ctx, raw_v, 0, 2, 1, 3);
            TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_k->data));
            TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_v->data));
          }
          const auto plan = kg::PlanFlashAttnOwners(launch(), in);
          ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
          EXPECT_EQ(plan->effective_cohort, static_cast<std::uint32_t>(owners));
          EXPECT_EQ(plan->original.head, full_plan->head);
          EXPECT_EQ(plan->original.columns, full_plan->columns);
          EXPECT_EQ(plan->original.group, full_plan->group);
          EXPECT_EQ(plan->original.mask_prepass, full_plan->mask_prepass);
          EXPECT_FALSE(plan->original.sparse);
          EXPECT_GT(plan->original_blocks_per_sm, 0);
          EXPECT_GT(plan->owner_blocks_per_sm, 0);
          EXPECT_EQ(plan->cohort_blocks, full_plan->blocks);
          EXPECT_EQ(plan->original.blocks * (partial ? 1 : static_cast<int>(groups)),
                    full_plan->blocks);
          const auto quad_tiles = static_cast<int>(kvh * (partial ? owners : 4));
          EXPECT_EQ(plan->original.blocks % quad_tiles == 0,
                    (partial ? full_plan->blocks : full_plan->blocks / static_cast<int>(groups)) %
                            quad_tiles ==
                        0);
          // Whole cohorts divide the original grid; partial cohorts filter
          // canonical global sequence offsets while retaining that whole grid.
          // Occupancy, KV batch and whole-tile preference can change across pins.
          const auto fixup = quad_tiles % plan->original.blocks == 0   ? "none"
                             : plan->original.blocks % quad_tiles == 0 ? "uniform"
                                                                       : "general";
          std::cout << "GEMMA3_BOUNDED12_PLAN owners=" << owners << " active=" << active
                    << " offset=" << first << " heads=" << heads << " D=" << d << " cells=" << cells
                    << " sms=" << sms << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                    << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                    << " full_blocks=" << full_plan->blocks
                    << " quad_blocks=" << plan->original.blocks
                    << " effective_cohort=" << plan->effective_cohort
                    << " scratch=" << plan->original.scratch << " fixup=" << fixup << '\n';
          auto invalid = in;
          if (active < 4) {
            invalid.k[active] = in.k[0];
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.v[active - 1] = nullptr;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          for (const auto bad : {0U, 1U, 2U, 3U, 4U, 5U}) {
            if (bad == active) continue;
            invalid = in;
            invalid.owner_count = bad;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          for (const auto bad : {1U, 3U, 5U, UINT32_MAX}) {
            invalid = in;
            invalid.owner_offset = bad;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.logical_cohort = 1;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          invalid = in;
          invalid.logit_softcap = 50;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          for (const auto logical : {4U, 8U, 9U, 10U, 11U}) {
            invalid = in;
            invalid.logical_cohort = logical;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          if (quad < 2) {
            invalid = in;
            invalid.bounded_roots = false;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.logical_cohort = 16;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          if (active > 1) {
            invalid = in;
            invalid.k[1] = invalid.k[0];
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          plans[quad] = *plan;
          scratch = std::max(scratch, plan->original.scratch);
        }
        // The original and independent-root calls reuse one explicitly funded
        // workspace; do not borrow the fixture's larger pool for this proof.
        const auto workspace = Allocate(scratch);
        auto context = LaunchContext::Create(launch().device(), *execution_, stream_,
                                             {.base = workspace, .size = Bytes(scratch)});
        ASSERT_TRUE(context) << jitllm::test_support::Failed(context)->detail;
        auto& bounded = **context;
        EXPECT_EQ(bounded.workspace().size.value(), scratch);
        EXPECT_TRUE(bounded.UsesStream(*execution_, stream_));
        const auto submission = execution_->Submission(stream_);
        ASSERT_TRUE(submission);
        const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
        ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
        for (std::size_t quad = 0; quad < groups; ++quad) {
          ASSERT_EQ(cudaMemsetAsync(quads[quad].output->data, 0xFF, ggml_nbytes(quads[quad].output),
                                    stream),
                    cudaSuccess);
          ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0xA5, scratch, stream),
                    cudaSuccess);
          bounded.ResetScratchPeak();
          ASSERT_TRUE(kg::FlashAttnOwnerRoots(bounded, quads[quad]));
          EXPECT_LE(bounded.scratch_peak().value(), plans[quad].original.scratch);
        }
        for (std::size_t group = 0; group < groups; ++group) {
          const auto bytes = plans[group].original.scratch;
          ASSERT_GT(bytes, 0U);
          auto short_context =
              LaunchContext::Create(launch().device(), *execution_, stream_,
                                    {.base = Allocate(bytes), .size = Bytes(bytes - 1)});
          ASSERT_TRUE(short_context);
          EXPECT_EQ(FailedCode(kg::FlashAttnOwnerRoots(**short_context, quads[group])),
                    KernelError::kRejected);
          EXPECT_FALSE((*short_context)->faulted());
        }
        bounded.ResetScratchPeak();
        ASSERT_TRUE(run_whole(bounded));
        EXPECT_LE(bounded.scratch_peak().value(), full_plan->scratch);
        auto expected = Download(whole);
        EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
        const auto compare = [&] {
          for (std::size_t quad = 0; quad < groups; ++quad) {
            const auto actual = Download(quads[quad].output);
            const auto offset = quad * 4 * n(d * heads);
            EXPECT_EQ(
                std::memcmp(actual.data(), expected.data() + offset, actual.size() * sizeof(float)),
                0);
          }
        };
        compare();
        const auto keys = Widen(kdata), values = Widen(vdata);
        std::vector<double> want(n(d * heads * owners));
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t head = 0; head < heads; ++head) {
            const auto visible = actual_cells(owner) - 37 - owner * 3;
            std::vector<double> scores(n(visible));
            for (std::int64_t cell = 0; cell < visible; ++cell)
              for (std::int64_t col = 0; col < d; ++col)
                scores[n(cell)] += double(qdata[n((owner * heads + head) * d + col)]) *
                                   keys[n(((owner * cells + cell) * kvh + head / 2) * d + col)];
            const auto peak = *std::max_element(scores.begin(), scores.end());
            double total = 0;
            for (auto& score : scores) {
              score = std::exp(score - peak);
              total += score;
            }
            for (std::int64_t col = 0; col < d; ++col)
              for (std::int64_t cell = 0; cell < visible; ++cell)
                want[n((owner * heads + head) * d + col)] +=
                    scores[n(cell)] / total *
                    values[n(((owner * cells + cell) * kvh + head / 2) * d + col)];
          }
        ExpectNmse(expected, want, kFlashAttnNmse,
                   "H8 grouped physical FP64 C" + std::to_string(owners));
        for (std::size_t group = 0; group < groups; ++group) {
          const auto actual = Download(quads[group].output);
          const auto offset = group * 4 * n(d * heads);
          std::vector<double> part(
              want.begin() + static_cast<std::ptrdiff_t>(offset),
              want.begin() + static_cast<std::ptrdiff_t>(offset + actual.size()));
          ExpectNmse(actual, part, kFlashAttnNmse, "H8 group FP64 " + std::to_string(group));
        }
        bounded.ResetScratchPeak();
        auto graph = bounded.Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
          if (auto r = run_whole(l); !r) return r;
          for (const auto& quad : std::span(quads).first(groups)) {
            if (cudaMemsetAsync(reinterpret_cast<void*>(workspace), 0xA5, scratch, stream) !=
                cudaSuccess)
              return std::unexpected(
                  KernelFailure{.error = KernelError::kUnknown, .detail = "scratch poison failed"});
            if (auto r = kg::FlashAttnOwnerRoots(l, quad); !r) return r;
          }
          return {};
        });
        ASSERT_TRUE(graph) << jitllm::test_support::Failed(graph)->detail;
        EXPECT_LE(bounded.scratch_peak().value(), scratch);
        for (int replay = 0; replay < 2; ++replay) {
          SCOPED_TRACE("capture replay " + std::to_string(replay));
          ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
          for (const auto& quad : std::span(quads).first(groups))
            ASSERT_EQ(cudaMemsetAsync(quad.output->data, 0xFF, ggml_nbytes(quad.output), stream),
                      cudaSuccess);
          ASSERT_TRUE(bounded.Launch(*graph));
          EXPECT_EQ(
              std::memcmp(Download(whole).data(), expected.data(), expected.size() * sizeof(float)),
              0);
          compare();
        }
        qdata = Normal(12104, qdata.size(), 0.25F);
        for (std::int64_t owner = 0; owner < owners; ++owner)
          masks[n(owner * cells * 32 + actual_cells(owner) - 38 - owner * 3)] =
              -std::numeric_limits<float>::infinity();
        const auto fresh_masks = Halves(masks);
        ASSERT_EQ(cudaMemcpyAsync(packed_q->data, qdata.data(), qdata.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(mask->data, fresh_masks.data(), fresh_masks.size() * 2,
                                  cudaMemcpyHostToDevice, stream),
                  cudaSuccess);
        ASSERT_TRUE(run_whole(bounded));
        const auto fresh_expected = Download(whole);
        ASSERT_TRUE(std::ranges::all_of(fresh_expected, [](float x) { return std::isfinite(x); }));
        EXPECT_NE(
            std::memcmp(fresh_expected.data(), expected.data(), expected.size() * sizeof(float)),
            0);
        expected = fresh_expected;
        ASSERT_TRUE(bounded.Launch(*graph));
        EXPECT_EQ(
            std::memcmp(Download(whole).data(), expected.data(), expected.size() * sizeof(float)),
            0);
        compare();
        EXPECT_EQ(
            std::memcmp(Download(packed_q).data(), qdata.data(), qdata.size() * sizeof(float)), 0);
        EXPECT_EQ(Download<ggml_fp16_t>(mask), fresh_masks);
        EXPECT_EQ(Download<ggml_fp16_t>(packed_k), kdata);
        EXPECT_EQ(Download<ggml_fp16_t>(packed_v), vdata);
        for (std::size_t group = 0; group < groups; ++group)
          for (std::size_t owner = 0; owner < quads[group].owner_count; ++owner) {
            const auto part = n(d * kvh * cells), first = (group * 4 + owner) * part;
            const auto actual = actual_cells(static_cast<std::int64_t>(group * 4 + owner));
            for (const auto& source : {std::pair{quads[group].k[owner], &kdata},
                                       std::pair{quads[group].v[owner], &vdata}}) {
              EXPECT_EQ(Download<ggml_fp16_t>(source.first),
                        std::vector<ggml_fp16_t>(
                            source.second->begin() + static_cast<std::ptrdiff_t>(first),
                            source.second->begin() +
                                static_cast<std::ptrdiff_t>(first + n(d * kvh * actual))));
              // Download the backing parent's guard to establish no writes.
              const auto* parent = source.first->view_src;
              while (parent->view_src) parent = parent->view_src;
              const auto backing = Download<ggml_fp16_t>(parent);
              EXPECT_TRUE(std::all_of(
                  backing.begin() + static_cast<std::ptrdiff_t>(n(d * kvh * actual)), backing.end(),
                  [](auto x) { return std::isnan(ggml_fp16_to_fp32(x)); }));
            }
          }
        EXPECT_FALSE(bounded.faulted());
        Finish();
      }
}

TEST_F(GgmlExtOpsTest, PartialPhysicalStreamsMatchOffsetFilteredRealRootGroupsExactly) {
  if (ComputeCapability() != 1210) GTEST_SKIP() << "Owner implementation is GB10 only";
  int sms = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, launch().device()),
            cudaSuccess);
  ASSERT_GT(sms, 0);
  for (const std::int64_t owners : {5, 6, 7, 9})
    for (const std::int64_t heads : {16, 32})
      for (const std::int64_t d : {256, 512}) {
        constexpr std::int64_t cells = 1024;
        SCOPED_TRACE(std::to_string(heads) + "/" + std::to_string(d) + "/" + std::to_string(cells));

        const auto kvh = heads / (d == 256 ? 2 : 8);
        const auto n = [](std::int64_t value) { return static_cast<std::size_t>(value); };
        auto arena = TensorArena::Create(128).value();
        auto* ctx = arena.context();
        auto* packed_q = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1, owners),
                               Normal(12101, n(d * heads * owners), 0.25F));
        auto* q = ggml_permute(ctx, packed_q, 0, 2, 1, 3);
        TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(packed_q->data));
        const auto kdata = Halves(Normal(12102, n(d * kvh * cells * owners), 0.25F));
        const auto vdata = Halves(Normal(12103, n(d * kvh * cells * owners), 0.25F));
        auto* packed_k =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), kdata);
        auto* packed_v =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, owners), vdata);
        auto* k = ggml_permute(ctx, packed_k, 0, 2, 1, 3);
        auto* v = ggml_permute(ctx, packed_v, 0, 2, 1, 3);
        TensorArena::Bind(k, reinterpret_cast<std::uintptr_t>(packed_k->data));
        TensorArena::Bind(v, reinterpret_cast<std::uintptr_t>(packed_v->data));
        std::vector<float> masks(n(cells * 32 * owners), -std::numeric_limits<float>::infinity());
        for (std::int64_t owner = 0; owner < owners; ++owner)
          for (std::int64_t cell = 0; cell < cells - 37 - owner * 3; ++cell)
            masks[n(owner * cells * 32 + cell)] = 0;
        auto* mask =
            Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cells, 32, 1, owners), Halves(masks));
        auto* whole = Place(ggml_flash_attn_ext(ctx, q, k, v, mask, 1, 0, 0));
        ggml_prec_set_acc(whole, GGML_PREC_F32);
        const auto full_plan = d == 256 ? kg::PlanFlashAttnMmaGqa2(launch(), whole)
                                        : kg::PlanFlashAttnMma(launch(), whole);
        ASSERT_TRUE(full_plan) << jitllm::test_support::Failed(full_plan)->detail;

        EXPECT_EQ(full_plan->columns, d == 256 ? 4 : 1);
        EXPECT_EQ(full_plan->group, d == 256 ? 2 : 8);
        EXPECT_FALSE(full_plan->sparse);
        EXPECT_TRUE(full_plan->mask_prepass);
        const auto run_whole = [&](LaunchContext& l) {
          return d == 256 ? kg::FlashAttnMmaGqa2(l, whole) : kg::FlashAttnMma(l, whole);
        };
        std::array<kg::FlashAttnOwners, 3> quads;
        const auto groups = n((owners + 3) / 4);
        std::array<kg::FlashAttnOwnersPlan, 3> plans;
        auto scratch = full_plan->scratch;
        for (std::size_t quad = 0; quad < groups; ++quad) {
          const auto first = quad * 4;
          const auto active = std::min(std::size_t{4}, n(owners) - first);
          const auto view = [&](ggml_tensor* tensor, std::array<std::int64_t, 4> ne) {
            const auto offset = first * tensor->nb[3];
            auto* slice = ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3], tensor->nb[1],
                                       tensor->nb[2], tensor->nb[3], offset);
            TensorArena::Bind(slice, reinterpret_cast<std::uintptr_t>(tensor->data) + offset);
            return slice;
          };
          auto& in = quads[quad];
          in.q = view(q, {d, 1, heads, static_cast<std::int64_t>(active)});
          in.mask = view(mask, {cells, 32, 1, static_cast<std::int64_t>(active)});
          in.output = Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, 1,
                                               static_cast<std::int64_t>(active)));
          in.logical_cohort = static_cast<std::uint32_t>(owners);
          in.owner_count = static_cast<std::uint32_t>(active);
          in.owner_offset = static_cast<std::uint32_t>(first);
          for (std::size_t owner = 0; owner < active; ++owner) {
            const auto part = n(d * kvh * cells);
            const auto begin = static_cast<std::ptrdiff_t>((first + owner) * part);
            const auto end = begin + static_cast<std::ptrdiff_t>(part);
            // Separate allocations authenticate actual roots; their bytes match
            // the corresponding planes of the contiguous physical-stream control.
            auto* raw_k =
                Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                      std::vector<ggml_fp16_t>(kdata.begin() + begin, kdata.begin() + end));
            auto* raw_v =
                Place(ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, kvh, cells, 1),
                      std::vector<ggml_fp16_t>(vdata.begin() + begin, vdata.begin() + end));
            in.k[owner] = ggml_permute(ctx, raw_k, 0, 2, 1, 3);
            in.v[owner] = ggml_permute(ctx, raw_v, 0, 2, 1, 3);
            TensorArena::Bind(const_cast<ggml_tensor*>(in.k[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_k->data));
            TensorArena::Bind(const_cast<ggml_tensor*>(in.v[owner]),
                              reinterpret_cast<std::uintptr_t>(raw_v->data));
          }
          const auto plan = kg::PlanFlashAttnOwners(launch(), in);
          ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
          EXPECT_EQ(plan->effective_cohort, static_cast<std::uint32_t>(owners));
          EXPECT_EQ(plan->original.head, full_plan->head);
          EXPECT_EQ(plan->original.columns, full_plan->columns);
          EXPECT_EQ(plan->original.group, full_plan->group);
          EXPECT_EQ(plan->original.mask_prepass, full_plan->mask_prepass);
          EXPECT_FALSE(plan->original.sparse);
          EXPECT_GT(plan->original_blocks_per_sm, 0);
          EXPECT_GT(plan->owner_blocks_per_sm, 0);
          EXPECT_EQ(plan->cohort_blocks, full_plan->blocks);
          EXPECT_EQ(plan->original.blocks, full_plan->blocks);
          const auto quad_tiles = static_cast<int>(kvh * owners);
          EXPECT_EQ(plan->original.blocks % quad_tiles == 0, full_plan->blocks % quad_tiles == 0);
          // Use the actual release's planned whole grid, matching QueuePartial.
          // Occupancy, KV batch and whole-tile preference can change across pins.
          const auto fixup = quad_tiles % plan->original.blocks == 0   ? "none"
                             : plan->original.blocks % quad_tiles == 0 ? "uniform"
                                                                       : "general";
          std::cout << "OWNER_PARTIAL_PLAN owners=" << owners << " active=" << active
                    << " offset=" << first << " heads=" << heads << " D=" << d << " cells=" << cells
                    << " sms=" << sms << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                    << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                    << " full_blocks=" << full_plan->blocks
                    << " quad_blocks=" << plan->original.blocks
                    << " effective_cohort=" << plan->effective_cohort
                    << " scratch=" << plan->original.scratch << " fixup=" << fixup << '\n';
          auto invalid = in;
          if (active < 4) {
            invalid.k[active] = in.k[0];
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.v[active - 1] = nullptr;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          for (const auto bad : {0U, 1U, 2U, 3U, 4U, 5U}) {
            if (bad == active) continue;
            invalid = in;
            invalid.owner_count = bad;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          for (const auto bad : {1U, 3U, 5U, UINT32_MAX}) {
            invalid = in;
            invalid.owner_offset = bad;
            EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          }
          invalid = in;
          invalid.logical_cohort = 1;
          EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
          plans[quad] = *plan;
          scratch = std::max(scratch, plan->original.scratch);
        }
        // The original and independent-root calls reuse one explicitly funded
        // workspace; do not borrow the fixture's larger pool for this proof.
        auto context = LaunchContext::Create(launch().device(), *execution_, stream_,
                                             {.base = Allocate(scratch), .size = Bytes(scratch)});
        ASSERT_TRUE(context) << jitllm::test_support::Failed(context)->detail;
        auto& bounded = **context;
        EXPECT_EQ(bounded.workspace().size.value(), scratch);
        EXPECT_TRUE(bounded.UsesStream(*execution_, stream_));
        const auto submission = execution_->Submission(stream_);
        ASSERT_TRUE(submission);
        const auto stream = reinterpret_cast<cudaStream_t>(submission->handle);
        ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
        for (std::size_t quad = 0; quad < groups; ++quad) {
          ASSERT_EQ(cudaMemsetAsync(quads[quad].output->data, 0xFF, ggml_nbytes(quads[quad].output),
                                    stream),
                    cudaSuccess);
          bounded.ResetScratchPeak();
          ASSERT_TRUE(kg::FlashAttnOwnerRoots(bounded, quads[quad]));
          EXPECT_LE(bounded.scratch_peak().value(), plans[quad].original.scratch);
        }
        bounded.ResetScratchPeak();
        ASSERT_TRUE(run_whole(bounded));
        EXPECT_LE(bounded.scratch_peak().value(), full_plan->scratch);
        const auto expected = Download(whole);
        EXPECT_TRUE(std::ranges::all_of(expected, [](float x) { return std::isfinite(x); }));
        const auto compare = [&] {
          for (std::size_t quad = 0; quad < groups; ++quad) {
            const auto actual = Download(quads[quad].output);
            const auto offset = quad * 4 * n(d * heads);
            EXPECT_EQ(
                std::memcmp(actual.data(), expected.data() + offset, actual.size() * sizeof(float)),
                0);
          }
        };
        compare();
        bounded.ResetScratchPeak();
        auto graph = bounded.Capture([&](LaunchContext& l) -> std::expected<void, KernelFailure> {
          if (auto r = run_whole(l); !r) return r;
          for (const auto& quad : std::span(quads).first(groups))
            if (auto r = kg::FlashAttnOwnerRoots(l, quad); !r) return r;
          return {};
        });
        ASSERT_TRUE(graph) << jitllm::test_support::Failed(graph)->detail;
        EXPECT_LE(bounded.scratch_peak().value(), scratch);
        for (int replay = 0; replay < 2; ++replay) {
          SCOPED_TRACE("capture replay " + std::to_string(replay));
          ASSERT_EQ(cudaMemsetAsync(whole->data, 0xFF, ggml_nbytes(whole), stream), cudaSuccess);
          for (const auto& quad : std::span(quads).first(groups))
            ASSERT_EQ(cudaMemsetAsync(quad.output->data, 0xFF, ggml_nbytes(quad.output), stream),
                      cudaSuccess);
          ASSERT_TRUE(bounded.Launch(*graph));
          EXPECT_EQ(
              std::memcmp(Download(whole).data(), expected.data(), expected.size() * sizeof(float)),
              0);
          compare();
        }
        EXPECT_FALSE(bounded.faulted());
        Finish();
      }
}

}  // namespace
