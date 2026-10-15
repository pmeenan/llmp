// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// llmp.image.rms_norm_mul_rope_sections (kernels/ggml/llmp_ops.h
// RunNormRopeSections) against the reference executor's decomposition of
// ir FusedKind::kRmsNormMulRopeSections: both rotation modes, one to three
// sections, F32 and BF16 weights.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/reference/exec.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace ref = llmp::kernels::reference;
using llmp::format::Encoding;

constexpr std::size_t kWorkspace = std::size_t{16} << 20U;

std::vector<float> Normal(std::uint32_t seed, std::size_t n) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0F, 1.0F);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

// Round-to-nearest-even BF16 bits of a finite float.
std::uint16_t Bf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

struct Shape {
  std::int64_t dim = 0;
  std::vector<std::uint32_t> sections;
};

class NormRopeSectionsGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = kg::LaunchContext::Create(
        0, *execution_, stream_, {.base = workspace, .size = llmp::base::Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<kg::TensorArena>(kg::TensorArena::Create(64).value());
  }
  void TearDown() override {
    Finish();
    launch_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }
  void Finish() {
    const auto fence = execution_->Record(stream_).value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    auto state = llmp::providers::FenceState::kPending;
    while ((state = execution_->Query(fence).value()) == llmp::providers::FenceState::kPending &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    ASSERT_EQ(state, llmp::providers::FenceState::kComplete);
    ASSERT_TRUE(execution_->Release(fence).has_value());
  }
  std::uint64_t Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }
  template <typename T>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& data) {
    const std::uint64_t address = Allocate(ggml_nbytes(tensor));
    kg::TensorArena::Bind(tensor, address);
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), ggml_nbytes(tensor));
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), data.data(), ggml_nbytes(tensor),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    return tensor;
  }

  // The kernel's output and the reference's for one configuration.
  void Compare(const Shape& shape, ir::RopeMode mode, bool bf16_weight, std::uint32_t seed) {
    SCOPED_TRACE(std::format("D {} sections {} {} weight {}", shape.dim, shape.sections.size(),
                             mode == ir::RopeMode::kNeox ? "neox" : "normal",
                             bf16_weight ? "BF16" : "F32"));
    constexpr std::int64_t kHeads = 3, kRows = 5;
    constexpr float kEps = 1e-6F, kBase = 10000.0F;
    const auto dim = static_cast<std::uint64_t>(shape.dim);
    const std::vector<float> x = Normal(seed, dim * kHeads * kRows);
    std::vector<float> w = Normal(seed + 1, dim);
    std::vector<std::uint16_t> w_bits;
    if (bf16_weight) {
      for (float& v : w) {
        w_bits.push_back(Bf16(v));
        v = std::bit_cast<float>(std::uint32_t{w_bits.back()} << 16U);
      }
    }
    std::mt19937 rng(seed + 2);
    std::uniform_int_distribution<int> position(0, 63);
    std::vector<std::vector<std::int32_t>> positions(shape.sections.size());
    for (auto& p : positions)
      for (std::int64_t r = 0; r < kRows; ++r) p.push_back(position(rng));
    ir::RopeSectionsParams params{.eps = kEps,
                                  .base = kBase,
                                  .mode = mode,
                                  .sections = static_cast<std::uint8_t>(shape.sections.size())};
    std::ranges::copy(shape.sections, params.dims.begin());

    // The reference: the fused node through its decomposition.
    ir::Builder b;
    const ir::TensorId tx = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({dim, kHeads, kRows})));
    const ir::TensorId tw =
        b.Input(ir::Dense(bf16_weight ? Encoding::kBF16 : Encoding::kF32, ir::ShapeOf({dim})));
    std::vector<ir::TensorId> sources{tx, tw};
    for (std::size_t i = 0; i < positions.size(); ++i)
      sources.push_back(b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kRows}))));
    const ir::TensorId out = b.Fused(ir::FusedKind::kRmsNormMulRopeSections, sources, params)[0];
    b.Output(out);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    auto exec = ref::Executor::Create(*graph);
    ASSERT_TRUE(exec) << exec.error().detail;
    ASSERT_TRUE(exec->BindValues(tx, x));
    ASSERT_TRUE(exec->BindValues(tw, w));
    for (std::size_t i = 0; i < positions.size(); ++i) {
      const std::vector<std::int64_t> wide(positions[i].begin(), positions[i].end());
      ASSERT_TRUE(exec->BindInts(sources[2 + i], wide));
    }
    ASSERT_TRUE(exec->Run());
    const auto want = exec->Values(out);
    ASSERT_TRUE(want);

    // The kernel.
    ggml_context* c = arena_->context();
    ggml_tensor* gx = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, shape.dim, kHeads, kRows), x);
    ggml_tensor* gw = bf16_weight ? Place(ggml_new_tensor_1d(c, GGML_TYPE_BF16, shape.dim), w_bits)
                                  : Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, shape.dim), w);
    std::vector<ggml_tensor*> gpos;
    for (const auto& p : positions)
      gpos.push_back(Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows), p));
    ggml_tensor* node = kg::NormRopeSections(c, gx, gw, gpos, kEps, kBase,
                                             mode == ir::RopeMode::kNeox, shape.sections);
    Place(node, std::vector<float>{});
    auto ran = kg::RunNormRopeSections(*launch_, node);
    ASSERT_TRUE(ran) << ran.error().detail;
    Finish();
    std::vector<float> got(static_cast<std::size_t>(ggml_nelements(node)));
    ASSERT_EQ(cudaMemcpy(got.data(), node->data, ggml_nbytes(node), cudaMemcpyDeviceToHost),
              cudaSuccess);

    ASSERT_EQ(got.size(), want->size());
    double err = 0, norm = 0, worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double d = static_cast<double>(got[i]) - (*want)[i];
      err += d * d;
      norm += static_cast<double>((*want)[i]) * (*want)[i];
      worst = std::max(worst, std::abs(d));
    }
    // F32 arithmetic both sides; the angles' powers are formed differently
    // (powf here, GGML's running product there).
    EXPECT_LE(std::sqrt(err / norm), 1e-5);
    EXPECT_LE(worst, 1e-4);
  }

  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::unique_ptr<kg::TensorArena> arena_;
};

TEST_F(NormRopeSectionsGpu, MatchesTheReferenceDecomposition) {
  const std::array shapes{Shape{.dim = 64, .sections = {64}},
                          Shape{.dim = 64, .sections = {32, 32}},
                          Shape{.dim = 128, .sections = {16, 56, 56}}};
  std::uint32_t seed = 10;
  for (const Shape& shape : shapes)
    for (const ir::RopeMode mode : {ir::RopeMode::kNormal, ir::RopeMode::kNeox})
      for (const bool bf16 : {false, true}) {
        Compare(shape, mode, bf16, seed);
        seed += 10;
      }
}

TEST_F(NormRopeSectionsGpu, ChecksRefuseSectionsThatLeaveTheHeadOrOverlapTheOutput) {
  ggml_context* c = arena_->context();
  ggml_tensor* x = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 2, 3), std::vector<float>{});
  ggml_tensor* w = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, 64), std::vector<float>{});
  ggml_tensor* p = Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, 3), std::vector<std::int32_t>{});
  const std::array<ggml_tensor*, 1> pos{p};
  const std::array<std::uint32_t, 1> whole{64}, part{32};
  ggml_tensor* ok =
      Place(kg::NormRopeSections(c, x, w, pos, 1e-6F, 1e4F, true, whole), std::vector<float>{});
  EXPECT_TRUE(kg::CheckNormRopeSections(ok));
  ggml_tensor* short_sections =
      Place(kg::NormRopeSections(c, x, w, pos, 1e-6F, 1e4F, true, part), std::vector<float>{});
  EXPECT_FALSE(kg::CheckNormRopeSections(short_sections));
  ggml_tensor* in_place = kg::NormRopeSections(c, x, w, pos, 1e-6F, 1e4F, true, whole);
  kg::TensorArena::Bind(in_place, reinterpret_cast<std::uintptr_t>(x->data));
  EXPECT_FALSE(kg::CheckNormRopeSections(in_place));
}

}  // namespace
