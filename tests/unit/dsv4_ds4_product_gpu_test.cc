// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Literal product controls, not a model/performance qualification. All
// operands stay owned through a provider fence; unknown completion exits
// without running GPU allocation destructors. No original ds4 host runtime.
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_ds4_product.h"
#include "kernels/ggml/launch.h"
#include "providers/cuda/cuda_device_execution.h"

namespace {
namespace kg = jitllm::kernels::ggml;
using jitllm::base::Bytes;
constexpr std::uint64_t kWorkspace = 16ULL << 20;
constexpr std::uint64_t kGuard = 256;

void Require(bool ok) {
  if (!ok) std::abort();  // unknown submission/completion must not unwind owners
}
class Dsv4Ds4ProductGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    auto device = jitllm::providers::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(device);
    execution_ = std::move(*device);
    const auto stream = execution_->CreateStream();
    ASSERT_TRUE(stream);
    stream_ = *stream;
    auto handle = kg::CublasHandle::Create(
        0, *execution_, stream_,
        {.base = Address(Allocate(32ULL << 20)), .size = Bytes(32ULL << 20)});
    ASSERT_TRUE(handle);
    cublas_ = std::move(*handle);
    auto context = kg::LaunchContext::Create(
        0, *execution_, stream_, {.base = Address(Allocate(kWorkspace)), .size = Bytes(kWorkspace)},
        cublas_.get());
    ASSERT_TRUE(context);
    launch_ = std::move(*context);
  }
  void TearDown() override {
    Prove();
    launch_.reset();
    cublas_.reset();
    Require(execution_->DestroyStream(stream_).has_value());
    for (void* p : allocations_) Require(cudaFree(p) == cudaSuccess);
  }
  static std::uint64_t Address(void* p) { return reinterpret_cast<std::uintptr_t>(p); }
  void Prove() {
    if (execution_ == nullptr) return;
    const auto fence = execution_->Record(stream_);
    Require(fence.has_value());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      const auto state = execution_->Query(*fence);
      Require(state.has_value());
      if (*state == jitllm::providers::FenceState::kComplete) break;
      Require(std::chrono::steady_clock::now() < deadline);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    Require(execution_->Release(*fence).has_value());
  }
  void* Allocate(std::uint64_t n) {
    void* p = nullptr;
    Require(cudaMalloc(&p, static_cast<std::size_t>(n + kGuard)) == cudaSuccess);
    allocations_.push_back(p);
    Prove();
    Require(cudaMemset(p, 0xA5, static_cast<std::size_t>(n + kGuard)) == cudaSuccess);
    Require(cudaDeviceSynchronize() == cudaSuccess);
    return p;
  }
  template <class T>
  void* Upload(const std::vector<T>& v) {
    void* p = Allocate(v.size() * sizeof(T));
    Prove();
    Require(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    Require(cudaDeviceSynchronize() == cudaSuccess);
    return p;
  }
  template <class T>
  std::vector<T> Read(const void* p, std::size_t count) {
    Prove();
    std::vector<T> v(count);
    Require(cudaMemcpy(v.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
    return v;
  }
  void Guard(const void* p, std::uint64_t n) {
    const auto bytes = Read<std::uint8_t>(static_cast<const char*>(p) + n, kGuard);
    EXPECT_TRUE(std::ranges::all_of(bytes, [](auto v) { return v == 0xA5; }));
  }
  kg::Ds4D4Sidecar Q8(const kg::Ds4ProductMatrix& x) {
    const auto bytes = kg::Ds4D4Bytes(x.rows, x.columns);
    Require(bytes.has_value());
    return {.storage = {.data = Allocate(*bytes), .bytes = *bytes},
            .source = x.storage.data,
            .generation = 7,
            .rows = x.rows,
            .columns = x.columns};
  }
  kg::Ds4ProductOutput Output(std::uint32_t t, std::uint32_t m) {
    const auto bytes = static_cast<std::uint64_t>(t) * m * 4;
    return {.storage = {.data = Allocate(bytes), .bytes = bytes},
            .rows = t,
            .columns = m,
            .row_stride = static_cast<std::uint64_t>(m) * 4};
  }
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  jitllm::providers::StreamId stream_;
  std::vector<void*> allocations_;
  std::unique_ptr<kg::CublasHandle> cublas_;
  std::unique_ptr<kg::LaunchContext> launch_;
};

TEST_F(Dsv4Ds4ProductGpu, EmbeddingRetainsHcOrderTokenFallbackAndCurrentCapturedInput) {
  constexpr std::size_t t = 13;
  constexpr std::size_t k = 7;
  constexpr std::size_t vocab = 3;
  std::vector<std::uint16_t> weights(vocab * k);
  for (std::uint32_t row = 0; row < vocab; ++row) {
    std::uint16_t bits = 0xc200;
    if (row == 0)
      bits = 0x3c00;
    else if (row == 1)
      bits = 0x4000;
    std::fill_n(weights.data() + (row * k), k, bits);
  }
  const std::vector<std::int32_t> tokens{-1, 0, 1, 2, 3, 1, 0, 2, 99, 0, 1, 2, 0};
  kg::Ds4Embedding d{.tokens = {.data = Upload(tokens), .bytes = tokens.size() * 4},
                     .weights = {.storage = {.data = Upload(weights), .bytes = weights.size() * 2},
                                 .rows = vocab,
                                 .columns = k,
                                 .row_stride = k * 2},
                     .output = Output(t, k * 4)};
  ASSERT_TRUE(kg::RunDs4Embedding(*launch_, d));
  const auto check = [&](const std::vector<std::int32_t>& ids) {
    const auto got = Read<float>(d.output.storage.data, t * k * 4);
    for (std::uint32_t row = 0; row < t; ++row) {
      const auto id = ids[row];
      float value = 1.0F;
      if (id == 1)
        value = 2.0F;
      else if (id == 2)
        value = -3.0F;
      for (std::uint32_t lane = 0; lane < k * 4; ++lane)
        EXPECT_EQ(got[(row * k * 4) + lane], value);
    }
  };
  check(tokens);
  auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4Embedding(l, d); });
  ASSERT_TRUE(graph);
  const std::vector<std::int32_t> changed(t, 2);
  Prove();
  Require(cudaMemcpy(const_cast<void*>(d.tokens.data), changed.data(), changed.size() * 4,
                     cudaMemcpyHostToDevice) == cudaSuccess);
  Require(cudaDeviceSynchronize() == cudaSuccess);
  ASSERT_TRUE(launch_->Launch(*graph));
  check(changed);
  Guard(d.output.storage.data, d.output.storage.bytes);
}

TEST_F(Dsv4Ds4ProductGpu, ConversionRetainsHalfwayRoundingSignsSubnormalsAndRowOrder) {
  const std::vector<float> x{0.0F,     -0.0F,    1.00048828125F, 1.00146484375F, -1.00048828125F,
                             0x1p-24F, 0x1p-25F, 65504.0F};
  kg::Ds4F16Conversion d{
      .input = {.storage = {.data = Upload(x), .bytes = x.size() * 4},
                .rows = 2,
                .columns = 4,
                .row_stride = 16},
      .output = {.storage = {.data = Allocate(x.size() * 2), .bytes = x.size() * 2},
                 .rows = 2,
                 .columns = 4,
                 .row_stride = 8}};
  ASSERT_TRUE(kg::RunDs4F16Conversion(*launch_, d));
  EXPECT_EQ(
      Read<std::uint16_t>(d.output.storage.data, x.size()),
      (std::vector<std::uint16_t>{0x0000, 0x8000, 0x3c00, 0x3c02, 0xbc00, 0x0001, 0x0000, 0x7bff}));
  Guard(d.output.storage.data, d.output.storage.bytes);
}

TEST_F(Dsv4Ds4ProductGpu, FusedQkvRmsMatchesIndependentDoubleRowsAndExactInplaceReplay) {
  constexpr std::size_t t = 13;
  constexpr std::size_t qn = 1024;
  constexpr std::size_t kn = 512;
  std::vector<float> q(t * qn);
  std::vector<float> kv(t * kn);
  std::vector<float> qw(qn);
  std::vector<float> kw(kn);
  for (std::uint32_t i = 0; i < q.size(); ++i)
    q[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 17;
  for (std::uint32_t i = 0; i < kv.size(); ++i)
    kv[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 13;
  for (std::uint32_t i = 0; i < qn; ++i) qw[i] = 0.5F + (static_cast<float>(i % 7) / 8);
  for (std::uint32_t i = 0; i < kn; ++i) kw[i] = 0.75F + (static_cast<float>(i % 5) / 8);
  kg::Ds4QkvNorm d{.query = {.storage = {.data = Upload(q), .bytes = q.size() * 4},
                             .rows = t,
                             .columns = qn,
                             .row_stride = qn * 4},
                   .query_weight = {.data = Upload(qw), .bytes = qw.size() * 4},
                   .query_output = Output(t, qn),
                   .kv = {.storage = {.data = Upload(kv), .bytes = kv.size() * 4},
                          .rows = t,
                          .columns = kn,
                          .row_stride = kn * 4},
                   .kv_weight = {.data = Upload(kw), .bytes = kw.size() * 4},
                   .kv_output = Output(t, kn)};
  ASSERT_TRUE(kg::RunDs4QkvNorm(*launch_, d));
  const auto qo = Read<float>(d.query_output.storage.data, q.size());
  const auto ko = Read<float>(d.kv_output.storage.data, kv.size());
  const auto check = [&](const std::vector<float>& in, const std::vector<float>& weight,
                         const std::vector<float>& out, std::uint32_t width) {
    for (std::uint32_t row = 0; row < t; ++row) {
      double sum = 0;
      for (std::uint32_t col = 0; col < width; ++col) {
        const double v = in[(row * width) + col];
        sum += v * v;
      }
      const auto scale = 1.0 / std::sqrt((sum / width) + d.epsilon);
      for (std::uint32_t col = 0; col < width; ++col)
        EXPECT_NEAR(out[(row * width) + col], in[(row * width) + col] * scale * weight[col],
                    3.0e-6);
    }
  };
  check(q, qw, qo, qn);
  check(kv, kw, ko, kn);
  d.query_output.storage = {.data = const_cast<void*>(d.query.storage.data),
                            .bytes = d.query.storage.bytes};
  d.kv_output.storage = {.data = const_cast<void*>(d.kv.storage.data), .bytes = d.kv.storage.bytes};
  auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4QkvNorm(l, d); });
  ASSERT_TRUE(graph);
  ASSERT_TRUE(launch_->Launch(*graph));
  EXPECT_EQ(Read<float>(d.query.storage.data, q.size()), qo);
  EXPECT_EQ(Read<float>(d.kv.storage.data, kv.size()), ko);
  Guard(d.query.storage.data, d.query.storage.bytes);
  Guard(d.kv.storage.data, d.kv.storage.bytes);
}

TEST_F(Dsv4Ds4ProductGpu, WideF16UsesPhysicalStridesAndLeavesRowPaddingAlone) {
  constexpr std::size_t t = 9;
  constexpr std::size_t m = 17;
  constexpr std::size_t k = 32;
  std::vector<std::uint16_t> weights(m * 40, 0x7E00);
  std::vector<std::uint16_t> x(t * 48, 0x7E00);
  for (std::uint32_t row = 0; row < m; ++row)
    std::fill_n(weights.data() + (static_cast<std::size_t>(row) * 40), k, 0x3800);  // F16 .5
  for (std::uint32_t row = 0; row < t; ++row)
    std::fill_n(x.data() + (static_cast<std::size_t>(row) * 48), k, 0x3000);  // F16 .125
  std::vector<float> out(t * 20, -123.0F);
  kg::Ds4F16Product d{.weights = {.storage = {.data = Upload(weights), .bytes = weights.size() * 2},
                                  .rows = m,
                                  .columns = k,
                                  .row_stride = 80},
                      .input = {.storage = {.data = Upload(x), .bytes = x.size() * 2},
                                .rows = t,
                                .columns = k,
                                .row_stride = 96},
                      .output = {.storage = {.data = Upload(out), .bytes = out.size() * 4},
                                 .rows = t,
                                 .columns = m,
                                 .row_stride = 80}};
  ASSERT_TRUE(kg::RunDs4F16Product(*launch_, d));
  const auto got = Read<float>(d.output.storage.data, out.size());
  for (std::uint32_t row = 0; row < t; ++row)
    for (std::uint32_t col = 0; col < 20; ++col)
      EXPECT_EQ(got[(row * 20) + col], col < m ? 2.0F : -123.0F);
  Guard(d.output.storage.data, d.output.storage.bytes);
}

TEST_F(Dsv4Ds4ProductGpu, OriginalD4AndStreamKProductsCoverRaggedColumnsAndOwnReplay) {
  constexpr std::size_t t = 257;
  constexpr std::size_t m = 128;
  constexpr std::size_t k = 512;
  std::vector<float> x(t * k);
  for (std::uint32_t row = 0; row < t; ++row)
    for (std::uint32_t col = 0; col < k; ++col)
      x[(row * k) + col] =
          col % 32 == 0 ? 127.0F : static_cast<float>(static_cast<int>(col % 7) - 3);
  kg::Ds4ProductMatrix input{.storage = {.data = Upload(x), .bytes = x.size() * 4},
                             .rows = t,
                             .columns = k,
                             .row_stride = k * 4};
  auto q = Q8(input);
  ASSERT_TRUE(kg::RunDs4D4(*launch_, input, 7, q));
  const auto quant = Read<std::uint8_t>(q.storage.data, q.storage.bytes);
  const auto payload = t * (k / 128) * 144;
  for (std::uint32_t kb = 0; kb < k / 128; ++kb)
    for (std::uint32_t row = 0; row < t; ++row)
      for (std::uint32_t col = 0; col < 128; ++col)
        EXPECT_EQ(
            static_cast<std::int8_t>(quant[(((kb * t) + row) * 144) + 16 + col]),
            static_cast<std::int8_t>(x[(row * k) + (static_cast<std::size_t>(kb) * 128) + col]));
  EXPECT_TRUE(std::ranges::all_of(quant.data() + payload, quant.data() + quant.size(),
                                  [](auto v) { return v == 0; }));
  Guard(q.storage.data, q.storage.bytes);
  std::vector<std::uint8_t> weights(m * (k / 32) * 34);
  for (std::uint32_t row = 0; row < m; ++row)
    for (std::uint32_t kb = 0; kb < k / 32; ++kb) {
      const auto at = ((row * (k / 32)) + kb) * 34;
      weights[at] = 0;
      weights[at + 1] = 0x30;  // F16 .125
      for (std::uint32_t j = 0; j < 32; ++j)
        weights[at + 2 + j] = static_cast<std::uint8_t>(static_cast<int>((row + kb + j) % 9) - 4);
    }
  kg::Ds4Q8Product d{.weights = {.raw = {.data = Upload(weights), .bytes = weights.size()},
                                 .raw_row_stride = (k / 32) * 34,
                                 .rows = m,
                                 .columns = k},
                     .input = input,
                     .output = Output(t, m),
                     .quantized = q,
                     .generation = 7,
                     .prepared = true};
  ASSERT_TRUE(kg::RunDs4Q8Product(*launch_, d));
  const auto got = Read<float>(d.output.storage.data, t * m);
  for (std::uint32_t row = 0; row < t; ++row)
    for (std::uint32_t col = 0; col < m; ++col) {
      double expected = 0;
      for (std::uint32_t group = 0; group < k / 32; ++group) {
        float scale = 0;
        const auto qa = (((group / 4) * t) + row) * 144;
        std::memcpy(&scale, quant.data() + qa + (static_cast<std::size_t>(group % 4) * 4), 4);
        std::int32_t dot = 0;
        for (std::uint32_t j = 0; j < 32; ++j)
          dot += static_cast<std::int8_t>(weights[(((col * (k / 32)) + group) * 34) + 2 + j]) *
                 static_cast<std::int8_t>(
                     quant[qa + 16 + (static_cast<std::size_t>(group % 4) * 32) + j]);
        expected += static_cast<double>(dot) * 0.125 * scale;
      }
      EXPECT_NEAR(got[(row * m) + col], expected, 1.0e-3);
    }
  auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4Q8Product(l, d); });
  ASSERT_TRUE(graph);
  ASSERT_TRUE(launch_->Launch(*graph));
  EXPECT_EQ(Read<float>(d.output.storage.data, t * m), got);
  Guard(d.output.storage.data, d.output.storage.bytes);
  EXPECT_LE(launch_->scratch_peak().value(), *kg::PlanDs4Q8Product(*launch_, d));
}

TEST_F(Dsv4Ds4ProductGpu, SmallF16PreservesF32HalfwayInputsForVectorAndScalarSplit) {
  constexpr std::uint32_t m = 17;
  for (const std::uint32_t k : {1024U, 1031U}) {
    const std::vector<std::uint16_t> weights(static_cast<std::size_t>(m) * k, 0x3C00);  // F16 1
    const auto* w = Upload(weights);
    for (const std::uint32_t t : {1U, 4U, 8U}) {
      // F16 would round this exact halfway to 1; original small path keeps F32.
      const std::vector<float> x(static_cast<std::size_t>(t) * k, 1.00048828125F);
      kg::Ds4F16Vector d{.weights = {.storage = {.data = w, .bytes = weights.size() * 2},
                                     .rows = m,
                                     .columns = k,
                                     .row_stride = static_cast<std::uint64_t>(k) * 2},
                         .input = {.storage = {.data = Upload(x), .bytes = x.size() * 4},
                                   .rows = t,
                                   .columns = k,
                                   .row_stride = static_cast<std::uint64_t>(k) * 4},
                         .output = Output(t, m)};
      ASSERT_TRUE(kg::RunDs4F16Vector(*launch_, d));
      const auto got = Read<float>(d.output.storage.data, static_cast<std::size_t>(t) * m);
      for (const auto value : got) EXPECT_EQ(value, static_cast<float>(k) * 1.00048828125F);
      auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4F16Vector(l, d); });
      ASSERT_TRUE(graph);
      ASSERT_TRUE(launch_->Launch(*graph));
      EXPECT_EQ(Read<float>(d.output.storage.data, static_cast<std::size_t>(t) * m), got);
      Guard(d.output.storage.data, d.output.storage.bytes);
    }
  }
}

TEST_F(Dsv4Ds4ProductGpu, OwnOutADualD4LeavesHeadsAndAllPhysicalTailGuardsIntact) {
  constexpr std::size_t t = 13;
  constexpr std::size_t width = 32768;
  constexpr std::size_t low = 8192;
  constexpr std::size_t k = 4096;
  std::vector<float> heads(t * width, 0.5F);
  std::vector<std::uint16_t> scales(low * (k / 32), 0x3400);  // F16 .25
  std::vector<std::int8_t> codes(low * k, 1);
  std::vector<std::int32_t> positions(t, 0);
  kg::Ds4OutA d{.weights = {.scales = {.data = Upload(scales), .bytes = scales.size() * 2},
                            .scale_row_stride = (k / 32) * 2,
                            .codes = {.data = Upload(codes), .bytes = codes.size()},
                            .code_row_stride = k,
                            .rows = low,
                            .columns = k},
                .heads = {.storage = {.data = Upload(heads), .bytes = heads.size() * 4},
                          .rows = t,
                          .columns = width,
                          .row_stride = width * 4},
                .low = {.storage = {.data = Allocate(16 * low * 4), .bytes = 16 * low * 4},
                        .rows = t,
                        .columns = low,
                        .row_stride = low * 4},
                .rope_table = {.data = Allocate(128ULL * 32 * 8), .bytes = 128ULL * 32 * 8},
                .rope = {.positions = {.data = Upload(positions), .bytes = positions.size() * 4},
                         .inverse = true},
                .generation = 7};
  kg::Ds4ProductMatrix low_input{
      .storage = {.data = d.low.storage.data, .bytes = d.low.storage.bytes},
      .rows = t,
      .columns = low,
      .row_stride = low * 4};
  d.quantized = Q8(low_input);
  ASSERT_TRUE(kg::RunDs4OutA(*launch_, d));
  const auto got = Read<float>(d.low.storage.data, 16 * low);
  for (std::uint32_t row = 0; row < 16; ++row)
    for (std::uint32_t col = 0; col < low; ++col)
      EXPECT_EQ(got[(row * low) + col], row < t ? 512.0F : 0.0F);
  EXPECT_EQ(Read<float>(d.heads.storage.data, heads.size()), heads);
  const auto own_q8 = Read<std::uint8_t>(d.quantized.storage.data, d.quantized.storage.bytes);
  const auto separate = Q8(low_input);
  ASSERT_TRUE(kg::RunDs4D4(*launch_, low_input, 7, separate));
  EXPECT_EQ(Read<std::uint8_t>(separate.storage.data, separate.storage.bytes), own_q8);
  ASSERT_TRUE(kg::RunDs4OutA(*launch_, d));
  EXPECT_EQ(Read<float>(d.low.storage.data, got.size()), got);
  Guard(d.low.storage.data, d.low.storage.bytes);
  Guard(d.rope_table.data, d.rope_table.bytes);
  Guard(d.quantized.storage.data, d.quantized.storage.bytes);
}

TEST_F(Dsv4Ds4ProductGpu, SmallQ8FullHeadPreservesF16ActivationScaleAndEveryLogit) {
  constexpr std::size_t m = 129280;
  constexpr std::size_t k = 1024;
  constexpr std::size_t t = 1;
  const std::vector<float> x(t * k, 128.0F);
  const std::vector<std::uint16_t> scales(m * (k / 32), 0x3000);  // F16 .125
  std::vector<std::int8_t> codes(m * k);
  for (std::uint32_t row = 0; row < m; ++row)
    std::fill_n(codes.data() + (row * k), k, static_cast<std::int8_t>((row % 7) + 1));
  kg::Ds4ProductMatrix input{.storage = {.data = Upload(x), .bytes = x.size() * 4},
                             .rows = t,
                             .columns = k,
                             .row_stride = k * 4};
  const auto qbytes = *kg::Ds4Q81Bytes(t, k);
  kg::Ds4Q8Vector d{.weights = {.scales = {.data = Upload(scales), .bytes = scales.size() * 2},
                                .scale_row_stride = (k / 32) * 2,
                                .codes = {.data = Upload(codes), .bytes = codes.size()},
                                .code_row_stride = k,
                                .rows = m,
                                .columns = k},
                    .input = input,
                    .output = Output(t, m),
                    .quantized = {.storage = {.data = Allocate(qbytes), .bytes = qbytes},
                                  .source = input.storage.data,
                                  .generation = 7,
                                  .rows = t,
                                  .columns = k},
                    .generation = 7};
  ASSERT_TRUE(kg::RunDs4Q8Vector(*launch_, d));
  const auto q = Read<std::uint8_t>(d.quantized.storage.data, qbytes);
  for (std::uint32_t b = 0; b < k / 32; ++b) {
    // 128/127 rounds to F16 1.0078125; F32 scales would produce 16384.
    EXPECT_EQ(q[static_cast<std::size_t>(b) * 36], 0x08);
    EXPECT_EQ(q[(b * 36) + 1], 0x3C);
    for (std::uint32_t j = 0; j < 32; ++j) EXPECT_EQ(q[(b * 36) + 4 + j], 127);
  }
  const auto got = Read<float>(d.output.storage.data, m);
  for (std::uint32_t row = 0; row < m; ++row)
    EXPECT_EQ(got[row], 16383.0F * static_cast<float>((row % 7) + 1));
  d.prepared = true;
  ASSERT_TRUE(kg::RunDs4Q8Vector(*launch_, d));
  EXPECT_EQ(Read<float>(d.output.storage.data, m), got);
  auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4Q8Vector(l, d); });
  ASSERT_TRUE(graph);
  ASSERT_TRUE(launch_->Launch(*graph));
  EXPECT_EQ(Read<float>(d.output.storage.data, m), got);
  Guard(d.output.storage.data, d.output.storage.bytes);
  Guard(d.quantized.storage.data, qbytes);
}

TEST_F(Dsv4Ds4ProductGpu, SmallRawQ8GuardsOddRowsSmallKAndPaddedActivationBlocks) {
  constexpr std::size_t m = 17;
  constexpr std::size_t k = 256;
  for (const std::uint32_t t : {1U, 4U, 8U}) {
    const auto physical = t == 1 ? 20U : 18U;
    const std::vector<float> x(t * k, 128.0F);
    std::vector<std::uint8_t> raw(physical * (k / 32) * 34);
    for (std::uint32_t b = 0; b < physical * (k / 32); ++b) {
      raw[static_cast<std::size_t>(b) * 34] = 0;
      raw[(b * 34) + 1] = 0x30;
      std::fill_n(raw.data() + (static_cast<std::size_t>(b) * 34) + 2, 32, 1);
    }
    kg::Ds4ProductMatrix input{.storage = {.data = Upload(x), .bytes = x.size() * 4},
                               .rows = t,
                               .columns = k,
                               .row_stride = k * 4};
    const auto qbytes = *kg::Ds4Q81Bytes(t, k);
    kg::Ds4Q8Vector d{.weights = {.raw = {.data = Upload(raw), .bytes = raw.size()},
                                  .raw_row_stride = (k / 32) * 34,
                                  .rows = m,
                                  .columns = k},
                      .input = input,
                      .output = Output(t, m),
                      .quantized = {.storage = {.data = Allocate(qbytes), .bytes = qbytes},
                                    .source = input.storage.data,
                                    .generation = 7,
                                    .rows = t,
                                    .columns = k},
                      .generation = 7,
                      .path = kg::Ds4Q8VectorPath::kRaw};
    ASSERT_TRUE(kg::RunDs4Q8Vector(*launch_, d));
    const auto got = Read<float>(d.output.storage.data, t * m);
    for (const auto v : got) EXPECT_EQ(v, 4095.75F);
    const auto q = Read<std::uint8_t>(d.quantized.storage.data, qbytes);
    for (std::uint32_t row = 0; row < t; ++row)
      for (std::uint32_t b = k / 32; b < 512 / 32; ++b)
        for (std::uint32_t byte = 0; byte < 36; ++byte)
          EXPECT_EQ(q[(((row * 16) + b) * 36) + byte], 0);
    auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4Q8Vector(l, d); });
    ASSERT_TRUE(graph);
    ASSERT_TRUE(launch_->Launch(*graph));
    EXPECT_EQ(Read<float>(d.output.storage.data, t * m), got);
    Guard(d.output.storage.data, d.output.storage.bytes);
    Guard(d.weights.raw.data, d.weights.raw.bytes);
    Guard(d.quantized.storage.data, qbytes);
  }
}

TEST_F(Dsv4Ds4ProductGpu, DenseD2rUsesAlignedPlanesAndGuardsItsLastColumnTile) {
  constexpr std::size_t t = 513;
  constexpr std::size_t m = 2048;
  constexpr std::size_t k = 1024;
  std::vector<float> x(t * k);
  for (std::uint32_t row = 0; row < t; ++row)
    for (std::uint32_t col = 0; col < k; ++col)
      x[(row * k) + col] =
          col % 32 == 0 ? 127.0F : static_cast<float>(static_cast<int>(col % 5) - 2);
  std::vector<std::uint16_t> scales(m * (k / 32), 0x3000);  // F16 .125
  std::vector<std::int8_t> codes(m * k);
  std::vector<std::uint8_t> raw(m * (k / 32) * 34);
  for (std::uint32_t row = 0; row < m; ++row)
    for (std::uint32_t col = 0; col < k; ++col) {
      codes[(row * k) + col] = static_cast<std::int8_t>(static_cast<int>((row + col) % 9) - 4);
      const auto at = ((row * (k / 32)) + (col / 32)) * 34;
      raw[at] = 0;
      raw[at + 1] = 0x30;
      raw[at + 2 + (col % 32)] = static_cast<std::uint8_t>(codes[(row * k) + col]);
    }
  kg::Ds4ProductMatrix input{.storage = {.data = Upload(x), .bytes = x.size() * 4},
                             .rows = t,
                             .columns = k,
                             .row_stride = k * 4};
  kg::Ds4Q8Product d{.weights = {.raw = {.data = Upload(raw), .bytes = raw.size()},
                                 .raw_row_stride = (k / 32) * 34,
                                 .scales = {.data = Upload(scales), .bytes = scales.size() * 2},
                                 .scale_row_stride = (k / 32) * 2,
                                 .codes = {.data = Upload(codes), .bytes = codes.size()},
                                 .code_row_stride = k,
                                 .rows = m,
                                 .columns = k},
                     .input = input,
                     .output = Output(t, m),
                     .quantized = Q8(input),
                     .generation = 7,
                     .path = kg::Ds4Q8Path::kDenseD2r};
  ASSERT_TRUE(kg::RunDs4Q8Product(*launch_, d));
  const auto got = Read<float>(d.output.storage.data, t * m);
  ASSERT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); }));
  Guard(d.output.storage.data, d.output.storage.bytes);
  const auto quant = Read<std::uint8_t>(d.quantized.storage.data, d.quantized.storage.bytes);
  for (const std::uint32_t row : {0U, 127U, 128U, 512U})
    for (const std::uint32_t out : {0U, 127U, 128U, 2047U}) {
      double expected = 0;
      for (std::uint32_t group = 0; group < k / 32; ++group) {
        const auto at = (((group / 4) * t) + row) * 144;
        float scale = 0;
        std::memcpy(&scale, quant.data() + at + (static_cast<std::size_t>(group % 4) * 4), 4);
        std::int32_t dot = 0;
        for (std::uint32_t j = 0; j < 32; ++j)
          dot += codes[(out * k) + (static_cast<std::size_t>(group) * 32) + j] *
                 static_cast<std::int8_t>(
                     quant[at + 16 + (static_cast<std::size_t>(group % 4) * 32) + j]);
        expected += static_cast<double>(dot) * scale * 0.125;
      }
      EXPECT_NEAR(got[(row * m) + out], expected, 1.0e-3);
    }
  d.prepared = true;
  ASSERT_TRUE(kg::RunDs4Q8Product(*launch_, d));
  EXPECT_EQ(Read<float>(d.output.storage.data, t * m), got);
  auto graph = launch_->Capture([&](auto& l) { return kg::RunDs4Q8Product(l, d); });
  ASSERT_TRUE(graph);
  ASSERT_TRUE(launch_->Launch(*graph));
  EXPECT_EQ(Read<float>(d.output.storage.data, t * m), got);
  d.path = kg::Ds4Q8Path::kMmq;
  ASSERT_TRUE(kg::RunDs4Q8Product(*launch_, d));
  const auto mmq = Read<float>(d.output.storage.data, t * m);
  double worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - mmq[i]));
  EXPECT_LE(worst, 1.0e-3);
}

TEST_F(Dsv4Ds4ProductGpu, HeadRopeKeepsNonrotaryLanesAndUsesSignedPerTokenPositions) {
  constexpr std::size_t t = 4;
  constexpr std::size_t h = 2;
  constexpr std::size_t width = 128;
  constexpr std::size_t rotary = 64;
  std::vector<float> x(t * h * width);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0F;
  const std::vector<std::int32_t> positions{-3, 0, 17, 131072};
  kg::Ds4HeadRope d{
      .input = {.storage = {.data = Upload(x), .bytes = x.size() * 4},
                .rows = t,
                .columns = h * width,
                .row_stride = h * width * 4},
      .heads = h,
      .head_width = width,
      .rope = {.positions = {.data = Upload(positions), .bytes = positions.size() * 4},
               .original_context = 131072,
               .rotary = rotary,
               .base = 10000.0F,
               .scale = 0.25F,
               .extension = 1.0F}};
  ASSERT_TRUE(kg::RunDs4HeadRope(*launch_, d));
  const auto got = Read<float>(d.input.storage.data, x.size());
  ASSERT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); }));
  for (std::uint32_t row = 0; row < t * h; ++row) {
    for (std::uint32_t col = 0; col < width - rotary; ++col)
      EXPECT_EQ(got[(row * width) + col], x[(row * width) + col]);
    // At absolute position zero every rotary pair is scaled but unrotated;
    // the source YaRN amplitude multiplier is 1 + .1*log(1/scale).
    if (positions[row / h] == 0) {
      const float amplitude = 1.0F + (0.1F * std::log(4.0F));
      for (std::uint32_t col = width - rotary; col < width; ++col)
        EXPECT_NEAR(got[(row * width) + col], x[(row * width) + col] * amplitude, 1.0e-6);
    }
  }
  Guard(d.input.storage.data, d.input.storage.bytes);
}
}  // namespace
