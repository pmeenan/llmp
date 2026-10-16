// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The image's planned kernels (kernels/ggml/llmp_ops.h llmp.image.*)
// against the reference executor's decompositions of their fused kinds:
// the sectioned norm and rotation (both rotation modes, one to three
// sections, F32 and BF16 weights, F32 and BF16 heads), the attention mask
// pack, the rotation table and the attention over BF16 heads.

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
#include <initializer_list>
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
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/cublas.h"
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

class ImageOpsGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    int major = 0, minor = 0;
    EXPECT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    EXPECT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    const llmp::base::Bytes cublas =
        kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor));
    auto handle = kg::CublasHandle::Create(0, *execution_, stream_,
                                           {.base = Allocate(cublas.value()), .size = cublas});
    ASSERT_TRUE(handle.has_value()) << (handle ? "" : handle.error().detail);
    cublas_ = std::move(*handle);
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = kg::LaunchContext::Create(
        0, *execution_, stream_, {.base = workspace, .size = llmp::base::Bytes(kWorkspace)},
        cublas_.get());
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<kg::TensorArena>(kg::TensorArena::Create(512).value());
  }
  void TearDown() override {
    Finish();
    launch_.reset();
    cublas_.reset();
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
  void Compare(const Shape& shape, ir::RopeMode mode, bool bf16_weight, std::uint32_t seed,
               bool bf16_heads = false) {
    SCOPED_TRACE(std::format("D {} sections {} {} weight {} heads {}", shape.dim,
                             shape.sections.size(), mode == ir::RopeMode::kNeox ? "neox" : "normal",
                             bf16_weight ? "BF16" : "F32", bf16_heads ? "BF16" : "F32"));
    constexpr std::int64_t kHeads = 3, kRows = 5;
    constexpr float kEps = 1e-6F, kBase = 10000.0F;
    const auto dim = static_cast<std::uint64_t>(shape.dim);
    std::vector<float> x = Normal(seed, dim * kHeads * kRows);
    std::vector<std::uint16_t> x_bits;
    if (bf16_heads) {
      for (float& v : x) {
        x_bits.push_back(Bf16(v));
        v = std::bit_cast<float>(std::uint32_t{x_bits.back()} << 16U);
      }
    }
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
    const ir::TensorId tx = b.Input(ir::Dense(bf16_heads ? Encoding::kBF16 : Encoding::kF32,
                                              ir::ShapeOf({dim, kHeads, kRows})));
    const ir::TensorId tw =
        b.Input(ir::Dense(bf16_weight ? Encoding::kBF16 : Encoding::kF32, ir::ShapeOf({dim})));
    std::vector<ir::TensorId> sources{tx, tw};
    for (std::size_t i = 0; i < positions.size(); ++i)
      sources.push_back(b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kRows}))));
    const std::array encodings{bf16_heads ? Encoding::kBF16 : Encoding::kF32};
    const ir::TensorId out =
        b.Fused(ir::FusedKind::kRmsNormMulRopeSections, sources, params, encodings)[0];
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
    ggml_tensor* gx =
        bf16_heads ? Place(ggml_new_tensor_3d(c, GGML_TYPE_BF16, shape.dim, kHeads, kRows), x_bits)
                   : Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, shape.dim, kHeads, kRows), x);
    ggml_tensor* gw = bf16_weight ? Place(ggml_new_tensor_1d(c, GGML_TYPE_BF16, shape.dim), w_bits)
                                  : Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, shape.dim), w);
    std::vector<ggml_tensor*> gpos;
    for (const auto& p : positions)
      gpos.push_back(Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows), p));
    ggml_tensor* node =
        kg::NormRopeSections(c, gx, gw, gpos, kEps, kBase, mode == ir::RopeMode::kNeox,
                             shape.sections, bf16_heads ? GGML_TYPE_BF16 : GGML_TYPE_F32);
    Place(node, std::vector<float>{});
    auto ran = kg::RunNormRopeSections(*launch_, node);
    ASSERT_TRUE(ran) << ran.error().detail;
    const std::vector<float> got = Read(node);

    ASSERT_EQ(got.size(), want->size());
    double err = 0, norm = 0, worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double d = static_cast<double>(got[i]) - (*want)[i];
      err += d * d;
      norm += static_cast<double>((*want)[i]) * (*want)[i];
      worst = std::max(worst, std::abs(d) / std::max(std::abs((*want)[i]), 1.0F));
    }
    // F32 arithmetic both sides; the angles' powers are formed differently
    // (powf here, GGML's running product there). A BF16 result rounds once
    // on each side, after F32 values a few units apart: at most two units in
    // its last place apart (2^-6 relative).
    EXPECT_LE(std::sqrt(err / norm), bf16_heads ? 4e-3 : 1e-5);
    EXPECT_LE(worst, bf16_heads ? 0x1p-6 : 1e-4);
  }

  // A result as F32 values (F16 and BF16 widened).
  std::vector<float> Read(const ggml_tensor* t) {
    Finish();
    const auto n = static_cast<std::size_t>(ggml_nelements(t));
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
      EXPECT_EQ(cudaMemcpy(out.data(), t->data, n * 4, cudaMemcpyDeviceToHost), cudaSuccess);
      return out;
    }
    std::vector<std::uint16_t> bits(n);
    EXPECT_EQ(cudaMemcpy(bits.data(), t->data, n * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::size_t i = 0; i < n; ++i)
      out[i] = t->type == GGML_TYPE_BF16 ? std::bit_cast<float>(std::uint32_t{bits[i]} << 16U)
                                         : HalfToFloat(bits[i]);
    return out;
  }
  static float HalfToFloat(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000U) << 16U;
    const std::uint32_t exp = (h >> 10U) & 0x1FU;
    const std::uint32_t man = h & 0x3FFU;
    if (exp == 0x1F) return std::bit_cast<float>(sign | 0x7F800000U | (man << 13U));
    if (exp == 0) {
      const float v = static_cast<float>(man) * 0x1p-24F;
      return sign != 0 ? -v : v;
    }
    return std::bit_cast<float>(sign | ((exp + 112U) << 23U) | (man << 13U));
  }

  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<kg::CublasHandle> cublas_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::unique_ptr<kg::TensorArena> arena_;
};

TEST_F(ImageOpsGpu, MatchesTheReferenceDecomposition) {
  const std::array shapes{Shape{.dim = 64, .sections = {64}},
                          Shape{.dim = 64, .sections = {32, 32}},
                          Shape{.dim = 128, .sections = {16, 56, 56}}};
  std::uint32_t seed = 10;
  for (const Shape& shape : shapes)
    for (const ir::RopeMode mode : {ir::RopeMode::kNormal, ir::RopeMode::kNeox})
      for (const bool bf16 : {false, true}) {
        Compare(shape, mode, bf16, seed);
        Compare(shape, mode, bf16, seed + 5, /*bf16_heads=*/true);
        seed += 10;
      }
}

TEST_F(ImageOpsGpu, ChecksRefuseSectionsThatLeaveTheHeadOrOverlapTheOutput) {
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

// The packed mask is the mask in F16, -inf past it; a whole row (all zero)
// keeps its zeros (its first -0, the attention's mark).
TEST_F(ImageOpsGpu, MaskPackIsTheMaskInF16PaddedWithInfinity) {
  // A square mask (its columns the attention's keys) marks a whole row with
  // -0 at its first column; a wider one leaves its zeros as they are.
  for (const std::int64_t cols : {std::int64_t{21}, std::int64_t{37}}) {
    constexpr std::int64_t kRows = 21, kCells = 256, kPadded = 24;
    SCOPED_TRACE(std::format("mask [{}, {}]", cols, kRows));
    std::vector<float> mask(static_cast<std::size_t>(cols * kRows), 0.0F);
    for (std::int64_t r = 5; r < kRows; ++r)
      for (std::int64_t c = 0; c < cols; ++c)
        mask[static_cast<std::size_t>(c + cols * r)] =
            c > r ? -INFINITY : 0.25F * static_cast<float>(c % 3);
    ir::Builder b;
    const ir::TensorId tm =
        b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({static_cast<std::uint64_t>(cols), kRows})));
    const ir::MaskPackParams params{.cells = kCells, .rows = kPadded};
    const ir::TensorId out = b.Fused(ir::FusedKind::kAttentionMaskPack, std::array{tm}, params)[0];
    b.Output(out);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    auto exec = ref::Executor::Create(*graph);
    ASSERT_TRUE(exec) << exec.error().detail;
    ASSERT_TRUE(exec->BindValues(tm, mask));
    ASSERT_TRUE(exec->Run());
    const auto want = exec->Values(out);
    ASSERT_TRUE(want);
    ggml_context* c = arena_->context();
    ggml_tensor* gm = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, cols, kRows), mask);
    ggml_tensor* node = Place(kg::AttentionMaskPack(c, gm, kCells, kPadded), std::vector<float>{});
    ASSERT_TRUE(kg::CheckAttentionMaskPack(node));
    auto ran = kg::RunAttentionMaskPack(*launch_, node);
    ASSERT_TRUE(ran) << ran.error().detail;
    const std::vector<float> got = Read(node);
    ASSERT_EQ(got.size(), want->size());
    for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], (*want)[i]) << i;  // -0 is 0
    std::uint16_t first = 0;
    ASSERT_EQ(cudaMemcpy(&first, node->data, 2, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(first, cols == kRows ? 0x8000U : 0x0000U);
    // Refused: columns not a multiple of 256.
    ggml_tensor* narrow = Place(kg::AttentionMaskPack(c, gm, 128, kPadded), std::vector<float>{});
    EXPECT_EQ(static_cast<bool>(kg::CheckAttentionMaskPack(narrow)), false);
  }
}

TEST_F(ImageOpsGpu, RopeTableIsTheRotationOfOneZero) {
  constexpr std::int64_t kRows = 9;
  const std::array<std::uint32_t, 3> dims{16, 56, 56};
  std::vector<std::vector<std::int32_t>> positions(3);
  for (std::size_t a = 0; a < 3; ++a)
    for (std::int64_t r = 0; r < kRows; ++r)
      positions[a].push_back(
          static_cast<std::int32_t>((r * static_cast<std::int64_t>(a + 3)) % 40));
  ir::RopeSectionsParams params{
      .eps = 0.0F, .base = 10000.0F, .mode = ir::RopeMode::kNormal, .sections = 3, .dims = dims};
  ir::Builder b;
  std::vector<ir::TensorId> sources;
  for (std::size_t a = 0; a < 3; ++a)
    sources.push_back(b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({kRows}))));
  const ir::TensorId out = b.Fused(ir::FusedKind::kRopeTable, sources, params)[0];
  b.Output(out);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  auto exec = ref::Executor::Create(*graph);
  ASSERT_TRUE(exec) << exec.error().detail;
  for (std::size_t a = 0; a < 3; ++a) {
    const std::vector<std::int64_t> wide(positions[a].begin(), positions[a].end());
    ASSERT_TRUE(exec->BindInts(sources[a], wide));
  }
  ASSERT_TRUE(exec->Run());
  const auto want = exec->Values(out);
  ASSERT_TRUE(want);
  ggml_context* c = arena_->context();
  std::vector<ggml_tensor*> gpos;
  for (const auto& p : positions)
    gpos.push_back(Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows), p));
  ggml_tensor* node = Place(kg::RopeTable(c, gpos, dims, 10000.0F), std::vector<float>{});
  auto ran = kg::RunRopeTable(*launch_, node);
  ASSERT_TRUE(ran) << ran.error().detail;
  const std::vector<float> got = Read(node);
  ASSERT_EQ(got.size(), want->size());
  double worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - (*want)[i]));
  // Angles of positions under 40: powf and sincosf against the reference's
  // F32 formation.
  EXPECT_LE(worst, 2e-4);
}

// The attention over BF16 heads with the query's norm and rotation in the
// kernel, against its decomposition (one packed mask and table for both).
TEST_F(ImageOpsGpu, QkAttentionMatchesItsDecomposition) {
  constexpr std::int64_t kD = 128, kH = 4, kHk = 2, kRows = 77, kText = 9;
  const auto cells = static_cast<std::uint32_t>((kRows + 255) / 256 * 256);
  const auto bf16 = [](std::vector<float>& v, std::vector<std::uint16_t>& bits) {
    for (float& x : v) {
      bits.push_back(Bf16(x));
      x = std::bit_cast<float>(std::uint32_t{bits.back()} << 16U);
    }
  };
  std::vector<float> q = Normal(41, kD * kH * kRows), k = Normal(42, kD * kHk * kRows),
                     v = Normal(43, kD * kHk * kRows), w = Normal(44, kD);
  std::vector<std::uint16_t> qb, kb, vb, wb;
  bf16(q, qb);
  bf16(k, kb);
  bf16(v, vb);
  bf16(w, wb);
  std::vector<float> mask(kRows * kRows, 0.0F);
  for (std::int64_t r = 0; r < kText; ++r)
    for (std::int64_t c = kText; c < kRows; ++c)
      mask[static_cast<std::size_t>(c + kRows * r)] = -INFINITY;
  std::vector<std::vector<std::int32_t>> positions(3);
  for (std::size_t a = 0; a < 3; ++a)
    for (std::int64_t r = 0; r < kRows; ++r)
      positions[a].push_back(
          static_cast<std::int32_t>((r * static_cast<std::int64_t>(a + 1)) % 31));
  const std::array<std::uint32_t, 3> dims{16, 56, 56};
  const ir::RopeSectionsParams rope{
      .eps = 0.0F, .base = 10000.0F, .mode = ir::RopeMode::kNormal, .sections = 3, .dims = dims};
  const ir::QkAttentionParams params{.eps = 1e-6F, .scale = 1.0F / std::sqrt(128.0F)};
  for (const bool bf16_out : {false, true}) {
    SCOPED_TRACE(bf16_out ? "BF16 out" : "F32 out");
    ir::Builder b;
    const auto in = [&](Encoding e, std::initializer_list<std::uint64_t> ne) {
      return b.Input(ir::Dense(e, ir::ShapeOf(ne)));
    };
    const ir::TensorId tq = in(Encoding::kBF16, {kD * kH, kRows});
    const ir::TensorId tw = in(Encoding::kBF16, {kD});
    const ir::TensorId tk = in(Encoding::kBF16, {kD * kHk, kRows});
    const ir::TensorId tv = in(Encoding::kBF16, {kD * kHk, kRows});
    const ir::TensorId tm = in(Encoding::kF32, {kRows, kRows});
    std::vector<ir::TensorId> tp;
    for (std::size_t a = 0; a < 3; ++a) tp.push_back(in(Encoding::kI32, {kRows}));
    const ir::MaskPackParams mp{.cells = cells, .rows = static_cast<std::uint32_t>(kRows)};
    const ir::TensorId packed = b.Fused(ir::FusedKind::kAttentionMaskPack, std::array{tm}, mp)[0];
    const ir::TensorId table = b.Fused(ir::FusedKind::kRopeTable, tp, rope)[0];
    const std::array encodings{bf16_out ? Encoding::kBF16 : Encoding::kF32};
    const ir::TensorId out =
        b.Fused(ir::FusedKind::kQkAttention, std::array{tq, tw, tk, tv, packed, table}, params,
                encodings)[0];
    b.Output(out);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    auto exec = ref::Executor::Create(*graph);
    ASSERT_TRUE(exec) << exec.error().detail;
    ASSERT_TRUE(exec->BindValues(tq, q));
    ASSERT_TRUE(exec->BindValues(tw, w));
    ASSERT_TRUE(exec->BindValues(tk, k));
    ASSERT_TRUE(exec->BindValues(tv, v));
    ASSERT_TRUE(exec->BindValues(tm, mask));
    for (std::size_t a = 0; a < 3; ++a) {
      const std::vector<std::int64_t> wide(positions[a].begin(), positions[a].end());
      ASSERT_TRUE(exec->BindInts(tp[a], wide));
    }
    ASSERT_TRUE(exec->Run());
    const auto want = exec->Values(out);
    ASSERT_TRUE(want);

    ggml_context* c = arena_->context();
    ggml_tensor* gq = Place(ggml_new_tensor_2d(c, GGML_TYPE_BF16, kD * kH, kRows), qb);
    ggml_tensor* gw = Place(ggml_new_tensor_1d(c, GGML_TYPE_BF16, kD), wb);
    ggml_tensor* gk = Place(ggml_new_tensor_2d(c, GGML_TYPE_BF16, kD * kHk, kRows), kb);
    ggml_tensor* gv = Place(ggml_new_tensor_2d(c, GGML_TYPE_BF16, kD * kHk, kRows), vb);
    ggml_tensor* gm = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kRows, kRows), mask);
    std::vector<ggml_tensor*> gpos;
    for (const auto& p : positions)
      gpos.push_back(Place(ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows), p));
    ggml_tensor* gpacked = Place(kg::AttentionMaskPack(c, gm, cells, kRows), std::vector<float>{});
    ggml_tensor* gtable = Place(kg::RopeTable(c, gpos, dims, 10000.0F), std::vector<float>{});
    ggml_tensor* node =
        Place(kg::QkAttention(c, gq, gw, gk, gv, gpacked, gtable, params.eps, params.scale,
                              bf16_out ? GGML_TYPE_BF16 : GGML_TYPE_F32),
              std::vector<float>{});
    ASSERT_TRUE(kg::RunAttentionMaskPack(*launch_, gpacked));
    ASSERT_TRUE(kg::RunRopeTable(*launch_, gtable));
    auto ran = kg::RunQkAttention(*launch_, node);
    ASSERT_TRUE(ran) << ran.error().detail;
    const std::vector<float> got = Read(node);
    ASSERT_EQ(got.size(), want->size());
    double err = 0, norm = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double d = static_cast<double>(got[i]) - (*want)[i];
      err += d * d;
      norm += static_cast<double>((*want)[i]) * (*want)[i];
    }
    // The kernel's BF16 query after its norm and rotation, F16 scores and
    // probabilities against the reference's F32.
    EXPECT_LE(std::sqrt(err / norm), 1.5e-2);
  }
}

// The worst relative error of `got` against `want`, by the larger
// magnitude, at least `floor`.
double WorstRelative(std::span<const float> got, std::span<const float> want, double floor) {
  EXPECT_EQ(got.size(), want.size());
  double worst = 0;
  for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
    const double w = want[i];
    worst = std::max(worst, std::abs(got[i] - w) / std::max(std::abs(w), floor));
  }
  return worst;
}

TEST_F(ImageOpsGpu, SplitModulationsMatchTheirDecompositions) {
  constexpr float kEps = 1e-6F;
  // A narrow and the image's 4096 wide row (its kernel apart), a split
  // inside, at the end and at the start.
  for (const auto [cols, rows, split] : std::array<std::array<std::int64_t, 3>, 4>{
           {{320, 37, 12}, {4096, 9, 4}, {320, 7, 7}, {320, 7, 0}}}) {
    const std::vector<float> x = Normal(51, static_cast<std::size_t>(cols * rows));
    const std::vector<float> branch = Normal(52, static_cast<std::size_t>(cols * rows));
    const std::vector<float> a = Normal(53, static_cast<std::size_t>(cols));
    const std::vector<float> bv = Normal(54, static_cast<std::size_t>(cols));
    const auto u = [](std::int64_t v) { return static_cast<std::uint64_t>(v); };
    ir::Builder b;
    const ir::TensorId tx = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({u(cols), u(rows)})));
    const ir::TensorId tr = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({u(cols), u(rows)})));
    const ir::TensorId ta = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({u(cols)})));
    const ir::TensorId tb = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({u(cols)})));
    const ir::SplitModulateParams mp{.eps = kEps, .split = static_cast<std::uint32_t>(split)};
    const ir::TensorId mod =
        b.Fused(ir::FusedKind::kLayerNormModulateSplit, std::array{tx, ta, tb}, mp)[0];
    const ir::SplitModulateParams gp{.eps = 0.0F, .split = static_cast<std::uint32_t>(split)};
    const ir::TensorId gated =
        b.Fused(ir::FusedKind::kGatedResidualSplit, std::array{tx, tr, ta, tb}, gp)[0];
    b.Output(mod);
    b.Output(gated);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    auto exec = ref::Executor::Create(*graph);
    ASSERT_TRUE(exec) << exec.error().detail;
    ASSERT_TRUE(exec->BindValues(tx, x));
    ASSERT_TRUE(exec->BindValues(tr, branch));
    ASSERT_TRUE(exec->BindValues(ta, a));
    ASSERT_TRUE(exec->BindValues(tb, bv));
    ASSERT_TRUE(exec->Run());
    const auto want_mod = exec->Values(mod);
    const auto want_gated = exec->Values(gated);
    ASSERT_TRUE(want_mod && want_gated);
    for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
      SCOPED_TRACE(std::format("C {} R {} split {} {}", cols, rows, split,
                               type == GGML_TYPE_F32 ? "F32" : "BF16"));
      ggml_context* c = arena_->context();
      ggml_tensor* gx = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, cols, rows), x);
      ggml_tensor* ga = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, cols), a);
      ggml_tensor* gb = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, cols), bv);
      ggml_tensor* node =
          Place(kg::LayerNormModulateSplit(c, gx, ga, gb, kEps, split, type), std::vector<float>{});
      ASSERT_TRUE(kg::CheckLayerNormModulateSplit(node));
      auto ran = kg::RunLayerNormModulateSplit(*launch_, node);
      ASSERT_TRUE(ran) << ran.error().detail;
      EXPECT_LE(WorstRelative(Read(node), *want_mod, 1.0), type == GGML_TYPE_F32 ? 1e-5 : 8e-3);
    }
    for (const bool bf16_branch : {false, true}) {
      SCOPED_TRACE(std::format("gated C {} R {} split {} branch {}", cols, rows, split,
                               bf16_branch ? "BF16" : "F32"));
      ggml_context* c = arena_->context();
      std::vector<std::uint16_t> bits;
      std::vector<float> want = *want_gated;
      if (bf16_branch) {
        // Against the reference over the rounded branch.
        for (std::size_t i = 0; i < branch.size(); ++i) {
          bits.push_back(Bf16(branch[i]));
          const double r = std::bit_cast<float>(std::uint32_t{bits.back()} << 16U);
          const auto row = static_cast<std::int64_t>(i) / cols;
          const auto col = static_cast<std::size_t>(static_cast<std::int64_t>(i) % cols);
          want[i] = static_cast<float>(
              x[i] + r * std::tanh(static_cast<double>(row < split ? a[col] : bv[col])));
        }
      }
      ggml_tensor* gx = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, cols, rows), x);
      ggml_tensor* gr = bf16_branch
                            ? Place(ggml_new_tensor_2d(c, GGML_TYPE_BF16, cols, rows), bits)
                            : Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, cols, rows), branch);
      ggml_tensor* ga = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, cols), a);
      ggml_tensor* gb = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, cols), bv);
      ggml_tensor* node =
          Place(kg::GatedResidualSplit(c, gx, gr, ga, gb, split), std::vector<float>{});
      ASSERT_TRUE(kg::CheckGatedResidualSplit(node));
      auto ran = kg::RunGatedResidualSplit(*launch_, node);
      ASSERT_TRUE(ran) << ran.error().detail;
      // tanhf's few units in the last place.
      EXPECT_LE(WorstRelative(Read(node), want, 1.0), 1e-4);
    }
    // Joined: the residual's sum and the modulation of it in one pass; the
    // run-time path's form too: a BF16 branch and carry and a BF16 sum.
    for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
      for (const bool bf16_rows : {false, true}) {
        if (bf16_rows && cols > 8192) continue;
        SCOPED_TRACE(std::format("joined C {} R {} split {} {}{}", cols, rows, split,
                                 type == GGML_TYPE_F32 ? "F32" : "BF16",
                                 bf16_rows ? " BF16 rows" : ""));
        const auto row_type = bf16_rows ? Encoding::kBF16 : Encoding::kF32;
        // The rows as the BF16 tensors hold them.
        std::vector<float> xr = x, br = branch;
        std::vector<std::uint16_t> xbits, bbits;
        if (bf16_rows) {
          for (float& v : xr) {
            xbits.push_back(Bf16(v));
            v = std::bit_cast<float>(std::uint32_t{xbits.back()} << 16U);
          }
          for (float& v : br) {
            bbits.push_back(Bf16(v));
            v = std::bit_cast<float>(std::uint32_t{bbits.back()} << 16U);
          }
        }
        ir::Builder jb;
        const auto in = [&](Encoding e, std::initializer_list<std::uint64_t> ne) {
          return jb.Input(ir::Dense(e, ir::ShapeOf(ne)));
        };
        const std::array js{in(row_type, {u(cols), u(rows)}), in(row_type, {u(cols), u(rows)}),
                            in(Encoding::kF32, {u(cols)}),    in(Encoding::kF32, {u(cols)}),
                            in(Encoding::kF32, {u(cols)}),    in(Encoding::kF32, {u(cols)})};
        const std::array encodings{row_type,
                                   type == GGML_TYPE_F32 ? Encoding::kF32 : Encoding::kBF16};
        const ir::Outputs joined =
            jb.Fused(ir::FusedKind::kGatedResidualModulateSplit, js, mp, encodings);
        jb.Output(joined[0]);
        jb.Output(joined[1]);
        auto jgraph = std::move(jb).Finish();
        ASSERT_TRUE(jgraph) << jgraph.error().detail;
        auto jexec = ref::Executor::Create(*jgraph);
        ASSERT_TRUE(jexec) << jexec.error().detail;
        // The carry, the branch, the gate's vectors a and b, the
        // modulation's b and a.
        const std::array<const std::vector<float>*, 6> values{&xr, &br, &a, &bv, &bv, &a};
        for (std::size_t i = 0; i < js.size(); ++i)
          ASSERT_TRUE(jexec->BindValues(js[i], *values[i]));
        ASSERT_TRUE(jexec->Run());
        const auto want_sum = jexec->Values(joined[0]);
        const auto want_out = jexec->Values(joined[1]);
        ASSERT_TRUE(want_sum && want_out);
        ggml_context* c = arena_->context();
        const ggml_type gt = bf16_rows ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        ggml_tensor* gx = bf16_rows ? Place(ggml_new_tensor_2d(c, gt, cols, rows), xbits)
                                    : Place(ggml_new_tensor_2d(c, gt, cols, rows), x);
        ggml_tensor* gr = bf16_rows ? Place(ggml_new_tensor_2d(c, gt, cols, rows), bbits)
                                    : Place(ggml_new_tensor_2d(c, gt, cols, rows), branch);
        ggml_tensor* ga = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, cols), a);
        ggml_tensor* gb = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, cols), bv);
        ggml_tensor* residual =
            Place(kg::GatedResidualSplit(c, gx, gr, ga, gb, split, gt), std::vector<float>{});
        ggml_tensor* modulated =
            Place(kg::LayerNormModulateSplit(c, residual, gb, ga, kEps, split, type),
                  std::vector<float>{});
        EXPECT_FALSE(kg::CheckGatedResidualModulateSplit(residual, modulated));  // unmarked
        kg::MarkJoined(modulated);
        ASSERT_TRUE(kg::CheckGatedResidualModulateSplit(residual, modulated));
        auto ran = kg::RunGatedResidualModulateSplit(*launch_, residual, modulated);
        ASSERT_TRUE(ran) << ran.error().detail;
        // A BF16 sum rounds once on each side (2^-7 relative apart at most
        // after F32 sums a few units apart; 2^-6 bounds it).
        EXPECT_LE(WorstRelative(Read(residual), *want_sum, 1.0), bf16_rows ? 0x1p-6 : 1e-4);
        EXPECT_LE(WorstRelative(Read(modulated), *want_out, 1.0),
                  type == GGML_TYPE_F32 ? 1e-4 : 8e-3);
      }
    }
  }
}

TEST_F(ImageOpsGpu, DupUpAddAndConv1x1MatchTheirDecompositions) {
  const auto u = [](std::int64_t v) { return static_cast<std::uint64_t>(v); };
  // The duplicate-up: each shortcut channel repeated 2, 16 and 4 times
  // (Co · ft · 4 / Ci), at one and two frames.
  for (const auto [w, h, ci, co, ft] : std::array<std::array<std::int64_t, 5>, 3>{
           {{5, 3, 6, 3, 1}, {4, 4, 4, 8, 2}, {7, 2, 3, 3, 1}}}) {
    SCOPED_TRACE(std::format("dup-up W {} H {} Ci {} Co {} ft {}", w, h, ci, co, ft));
    const std::vector<float> x = Normal(61, static_cast<std::size_t>(4 * w * h * co));
    const std::vector<float> sc = Normal(62, static_cast<std::size_t>(w * h * ci));
    ir::Builder b;
    const ir::TensorId tx =
        b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({u(2 * w), u(2 * h), u(co)})));
    const ir::TensorId ts = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({u(w), u(h), u(ci)})));
    const ir::DupUpParams params{.factor_t = static_cast<std::uint32_t>(ft)};
    const ir::TensorId out = b.Fused(ir::FusedKind::kDupUpAdd, std::array{tx, ts}, params)[0];
    b.Output(out);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    auto exec = ref::Executor::Create(*graph);
    ASSERT_TRUE(exec) << exec.error().detail;
    ASSERT_TRUE(exec->BindValues(tx, x));
    ASSERT_TRUE(exec->BindValues(ts, sc));
    ASSERT_TRUE(exec->Run());
    const auto want = exec->Values(out);
    ASSERT_TRUE(want);
    ggml_context* c = arena_->context();
    ggml_tensor* gx = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 2 * w, 2 * h, co), x);
    ggml_tensor* gs = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, w, h, ci), sc);
    ggml_tensor* node = Place(kg::DupUpAdd(c, gx, gs, ft), std::vector<float>{});
    ASSERT_TRUE(kg::CheckDupUpAdd(node));
    auto ran = kg::RunDupUpAdd(*launch_, node);
    ASSERT_TRUE(ran) << ran.error().detail;
    const std::vector<float> got = Read(node);
    ASSERT_EQ(got.size(), want->size());
    for (std::size_t i = 0; i < got.size(); ++i) ASSERT_EQ(got[i], (*want)[i]) << i;  // one add
  }
  {
    // A shortcut whose repeats do not divide: refused.
    ggml_context* c = arena_->context();
    ggml_tensor* gx = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 4, 4, 3), std::vector<float>{});
    ggml_tensor* gs = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, 2, 2, 5), std::vector<float>{});
    EXPECT_FALSE(kg::CheckDupUpAdd(Place(kg::DupUpAdd(c, gx, gs, 1), std::vector<float>{})));
  }
  // The 1x1 convolution, against the reference's.
  constexpr std::int64_t kW = 9, kH = 7, kCi = 24, kCo = 10;
  const std::vector<float> x = Normal(71, static_cast<std::size_t>(kW * kH * kCi));
  const std::vector<float> w = Normal(72, static_cast<std::size_t>(kCi * kCo));
  const std::vector<float> bias = Normal(73, static_cast<std::size_t>(kCo));
  ir::Builder b;
  const ir::TensorId tx = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({kW, kH, kCi})));
  const ir::TensorId tw = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({1, 1, kCi, kCo})));
  const ir::TensorId tb = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({kCo})));
  const ir::TensorId out = b.Conv2d(tw, tx, tb, {.stride = 1, .padding = 0, .dilation = 1});
  b.Output(out);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  auto exec = ref::Executor::Create(*graph);
  ASSERT_TRUE(exec) << exec.error().detail;
  ASSERT_TRUE(exec->BindValues(tx, x));
  ASSERT_TRUE(exec->BindValues(tw, w));
  ASSERT_TRUE(exec->BindValues(tb, bias));
  ASSERT_TRUE(exec->Run());
  const auto want = exec->Values(out);
  ASSERT_TRUE(want);
  ggml_context* c = arena_->context();
  ggml_tensor* gx = Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, kW, kH, kCi), x);
  ggml_tensor* gw = Place(ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, kCi, kCo), w);
  ggml_tensor* gb = Place(ggml_new_tensor_1d(c, GGML_TYPE_F32, kCo), bias);
  ggml_tensor* node = Place(kg::Conv1x1(c, gw, gx, gb), std::vector<float>{});
  ASSERT_TRUE(kg::CheckConv1x1(node));
  auto ran = kg::RunConv1x1(*launch_, node);
  ASSERT_TRUE(ran) << ran.error().detail;
  // The lent handle's TF32 tensor-op math (cublas.cc, as upstream's and the
  // F32 products'): operands rounded to 10-bit mantissas.
  EXPECT_LE(WorstRelative(Read(node), *want, 1.0), 1e-2);
}

}  // namespace
