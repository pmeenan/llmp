// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <tuple>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace kg = jitllm::kernels::ggml;
class GemmaNormGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    auto opened = jitllm::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened);
    execution = std::move(*opened);
    auto created_stream = execution->CreateStream();
    ASSERT_TRUE(created_stream);
    stream = *created_stream;
    stream_ready = true;
    auto created = kg::LaunchContext::Create(
        0, *execution, stream, {.base = Allocate(2U << 20), .size = jitllm::base::Bytes(2U << 20)});
    ASSERT_TRUE(created);
    launch = std::move(*created);
  }
  bool Finish() {
    if (retirement_failed) return false;
    if (!execution || !stream_ready) return true;
    auto fence = execution->Record(stream);
    if (!fence) {
      ADD_FAILURE() << "completion fence recording failed";
      retirement_failed = true;
      return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      auto state = execution->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= deadline) {
        ADD_FAILURE() << "unproven retirement; retain fixture owners";
        retirement_failed = true;
        return false;
      }
      if (*state == jitllm::providers::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution->Release(*fence)) {
      retirement_failed = true;
      ADD_FAILURE() << "completed fence release failed";
      return false;
    }
    return true;
  }
  void TearDown() override {
    if (!Finish()) {
      std::ignore = new std::vector<kg::CapturedGraph>(std::move(graphs));
      std::ignore = new std::vector<std::unique_ptr<kg::TensorArena>>(std::move(arenas));
      std::ignore = launch.release();
      std::ignore = execution.release();
      return;
    }
    graphs.clear();
    launch.reset();
    if (execution && stream_ready) EXPECT_TRUE(execution->DestroyStream(stream));
    for (auto* memory : allocations) EXPECT_EQ(cudaFree(memory), cudaSuccess);
  }
  std::uint64_t Allocate(std::uint64_t size) {
    void* memory = nullptr;
    EXPECT_EQ(cudaMalloc(&memory, size), cudaSuccess);
    allocations.push_back(memory);
    return reinterpret_cast<std::uintptr_t>(memory);
  }
  ggml_tensor* Bind(ggml_tensor* t) {
    kg::TensorArena::Bind(t, Allocate(ggml_nbytes(t)));
    return t;
  }
  template <class T>
  void Upload(ggml_tensor* t, const std::vector<T>& values) {
    ASSERT_EQ(ggml_nbytes(t), values.size() * sizeof(T));
    const auto copied = cudaMemcpy(t->data, values.data(), ggml_nbytes(t), cudaMemcpyHostToDevice);
    if (copied != cudaSuccess) retirement_failed = true;
    ASSERT_EQ(copied, cudaSuccess);
    const auto completed = cudaDeviceSynchronize();
    if (completed != cudaSuccess) retirement_failed = true;
    ASSERT_EQ(completed, cudaSuccess);
  }
  std::vector<float> Read(ggml_tensor* t) {
    if (!Finish()) return {};
    std::vector<float> result(static_cast<std::size_t>(ggml_nelements(t)));
    const auto copied = cudaMemcpy(result.data(), t->data, ggml_nbytes(t), cudaMemcpyDeviceToHost);
    if (copied != cudaSuccess) retirement_failed = true;
    EXPECT_EQ(copied, cudaSuccess);
    return result;
  }
  void Operand(bool rope, std::size_t width, bool reversed = false, std::size_t heads = 4,
               std::size_t tokens = 6, float freq_base = 10000.0f, float freq_scale = 1.0f,
               int original_context = 262144) {
    arenas.push_back(std::make_unique<kg::TensorArena>(kg::TensorArena::Create(16).value()));
    auto* c = arenas.back()->context();
    const auto dimension = static_cast<std::int64_t>(width);
    const auto head_count = static_cast<std::int64_t>(heads);
    const auto token_count = static_cast<std::int64_t>(tokens);
    auto* x = Bind(rope ? ggml_new_tensor_3d(c, GGML_TYPE_F32, dimension, head_count, token_count)
                        : ggml_new_tensor_2d(c, GGML_TYPE_F32, dimension, token_count));
    auto* weight = Bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, dimension));
    auto* norm = Bind(ggml_rms_norm(c, x, 1e-6f));
    auto* mul = Bind(ggml_mul(c, norm, weight));
    ggml_tensor *positions = nullptr, *factors = nullptr, *residual = nullptr, *out = nullptr,
                *reference_input = nullptr, *reference_rope = nullptr;
    if (rope) {
      positions = Bind(ggml_new_tensor_1d(c, GGML_TYPE_I32, token_count));
      if (width == 512) factors = Bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, dimension / 2));
      out = Bind(ggml_rope_ext(c, mul, positions, factors, static_cast<int>(width),
                               GGML_ROPE_TYPE_NEOX, original_context, freq_base, freq_scale, 0.0f,
                               1.0f, 32.0f, 1.0f));
      reference_input = Bind(ggml_dup_tensor(c, x));
      reference_rope = Bind(ggml_rope_ext(
          c, reference_input, positions, factors, static_cast<int>(width), GGML_ROPE_TYPE_NEOX,
          original_context, freq_base, freq_scale, 0.0f, 1.0f, 32.0f, 1.0f));
    } else {
      residual = Bind(ggml_new_tensor_2d(c, GGML_TYPE_F32, dimension, token_count));
      out = Bind(reversed ? ggml_add(c, residual, mul) : ggml_add(c, mul, residual));
    }
    std::vector<float> xv(static_cast<std::size_t>(ggml_nelements(x))), wv(width), rv(xv.size()),
        fv(width / 2);
    std::vector<std::int32_t> pv(tokens);
    for (std::size_t i = 0; i < xv.size(); ++i) {
      xv[i] = static_cast<float>(static_cast<int>(i % 97) - 48) / 64.0f;
      rv[i] = static_cast<float>(static_cast<int>(i % 43) - 21) / 32.0f;
    }
    for (std::size_t i = 0; i < width; ++i) wv[i] = 1.0f + static_cast<float>(i % 11) / 32.0f;
    for (std::size_t i = 0; i < width / 2; ++i) fv[i] = 1.0f + static_cast<float>(i % 7) / 16.0f;
    for (std::size_t i = 0; i < tokens; ++i) pv[i] = static_cast<std::int32_t>(i);
    Upload(x, xv);
    Upload(weight, wv);
    if (residual) Upload(residual, rv);
    if (positions) Upload(positions, pv);
    if (factors) Upload(factors, fv);
    ASSERT_FALSE(retirement_failed);
    const auto registry = jitllm::execution::Registry::Create(kg::Implementations());
    ASSERT_TRUE(registry);
    kg::GraphPlan plan;
    plan.steps.push_back({.operation = rope ? jitllm::execution::Operation::kRmsNormMulRope
                                            : jitllm::execution::Operation::kRmsNormMulAdd,
                          .implementation = rope ? kg::kGemmaNormRopeName : kg::kGemmaNormAddName,
                          .nodes = {norm, mul, out}});
    const auto scratch = kg::PlanScratch(*launch, plan);
    ASSERT_TRUE(scratch);
    EXPECT_EQ(*scratch, 0U);
    auto bound = kg::BoundGraph::Bind(*registry, plan);
    ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
    ASSERT_TRUE(bound->Run(*launch));
    const auto baseline = Read(out);
    ASSERT_FALSE(retirement_failed);
    const auto oracle = [&]() {
      std::vector<float> normalized(xv.size());
      std::vector<double> expected(xv.size());
      for (std::size_t r = 0; r < xv.size() / width; ++r) {
        double sum = 0;
        for (std::size_t i = 0; i < width; ++i)
          sum += double(xv[r * width + i]) * xv[r * width + i];
        const double scale = 1.0 / std::sqrt(sum / static_cast<double>(width) + 1e-6);
        for (std::size_t i = 0; i < width; ++i) {
          const auto index = r * width + i;
          const double value = xv[index] * scale * wv[i];
          normalized[index] = static_cast<float>(value);
          expected[index] = rope ? value : value + rv[index];
        }
      }
      if (rope) {
        // Independent FP64 normalization/scaling, then the unchanged primitive
        // RoPE's declared fast phase arithmetic at large positions. A CPU
        // pow/trig oracle defines a different phase and is not an exact gate.
        Upload(reference_input, normalized);
        ASSERT_FALSE(retirement_failed);
        ASSERT_TRUE(kg::Rope(*launch, reference_rope));
        const auto rotated = Read(reference_rope);
        ASSERT_FALSE(retirement_failed);
        if (pv.back() < 32) {
          // At zero/small positions, independently check the full FP64 phase
          // and output, as well as the large-position primitive control.
          for (std::size_t r = 0; r < xv.size() / width; ++r)
            for (std::size_t i = 0; i < width; ++i) {
              const auto pair = i % (width / 2);
              const double angle = double(freq_scale) * double(pv[r / heads]) *
                                   std::pow(double(freq_base), -2.0 * static_cast<double>(pair) /
                                                                   static_cast<double>(width)) /
                                   (factors ? fv[pair] : 1.0f);
              const double a = expected[r * width + pair];
              const double b = expected[r * width + pair + width / 2];
              const double scalar = i < width / 2 ? a * std::cos(angle) - b * std::sin(angle)
                                                  : a * std::sin(angle) + b * std::cos(angle);
              EXPECT_NEAR(ReadResult[r * width + i], scalar, 0.0005);
            }
        }
        for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = rotated[i];
      }
      for (std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(ReadResult[i], expected[i], 0.0005) << "element=" << i;
    };
    ReadResult = baseline;
    oracle();
    const auto preserved = Read(x);
    ASSERT_FALSE(retirement_failed);
    EXPECT_EQ(std::memcmp(preserved.data(), xv.data(), xv.size() * 4),
              0);  // Raw K-as-V stays intact.
    auto captured = launch->Capture([&](kg::LaunchContext& l) { return bound->Run(l); });
    ASSERT_TRUE(captured);
    graphs.push_back(std::move(*captured));
    ASSERT_TRUE(launch->Launch(graphs.back()));
    const auto repeated = Read(out);
    ASSERT_FALSE(retirement_failed);
    EXPECT_EQ(std::memcmp(repeated.data(), baseline.data(), baseline.size() * 4), 0);
    if (positions) {
      for (auto& position : pv) position += 1024;
      Upload(positions, pv);
    } else {
      for (auto& value : xv) value += 0.125f;
      Upload(x, xv);
    }
    ASSERT_FALSE(retirement_failed);
    ASSERT_TRUE(launch->Launch(graphs.back()));
    ReadResult = Read(out);
    ASSERT_FALSE(retirement_failed);
    oracle();
  }
  bool stream_ready = false;
  bool retirement_failed = false;
  std::vector<kg::CapturedGraph> graphs;
  std::vector<std::unique_ptr<kg::TensorArena>> arenas;
  std::vector<float> ReadResult;
  std::unique_ptr<jitllm::providers::DeviceExecution> execution;
  jitllm::providers::StreamId stream;
  std::unique_ptr<kg::LaunchContext> launch;
  std::vector<void*> allocations;
};
TEST_F(GemmaNormGpu, D256AndFactorAwareD512RotateAndCaptureFreshPositions) {
  Operand(true, 256);
  Operand(true, 512);
}
TEST_F(GemmaNormGpu, Gemma3QueryAndKeyShapesLocalAndLinearGlobalRope) {
  for (std::size_t heads : {4U, 8U})
    for (std::size_t rows : {1U, 128U}) {
      Operand(true, 256, false, heads, rows, 10000.0f, 1.0f, 131072);
      Operand(true, 256, false, heads, rows, 1000000.0f, 0.125f, 131072);
    }
}
TEST_F(GemmaNormGpu, Gemma3ResidualWidthRowsOrdersAndCaptureFreshInputs) {
  for (std::size_t rows : {1U, 128U})
    for (bool reversed : {false, true}) Operand(false, 2560, reversed, 1, rows);
}
TEST_F(GemmaNormGpu, ApprovedResidualWidthsOrdersAndCaptureFreshInputs) {
  for (std::size_t width : {2816U, 5376U})
    for (bool reversed : {false, true}) Operand(false, width, reversed);
}
}  // namespace
