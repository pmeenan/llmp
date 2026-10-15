// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's fast path (kernels/ggml/llmp_ops.h; D-085: speed before bit
// exactness) on a GB10 (label `gpu`), at the model's widths: the MXFP8
// quantization (every value within E4M3's rounding of its block's scale,
// the scale the smallest power of two that holds the block, the padding
// rows' scales zero) and the tensor-core product over it (the product of
// the quantized operands, from FP64); the fused hyper-connections, routing,
// Gated DeltaNet and QSA kernels against FP64 references; QSA's cached
// block keys, its selection against an exact host selection (ties to the
// lower cell, at any depth) and its attention over the kept cells; and
// what their checks refuse.

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
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "components/step.h"
#include "execution/registry.h"
#include "ggml.h"
#include "ir/builder.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/mxfp8_cutlass.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/tensors.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"
#include "providers/device_runtime.h"

namespace {

using llmp::base::Bytes;
using llmp::kernels::ggml::CublasHandle;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
namespace kg = llmp::kernels::ggml;
namespace mx = llmp::kernels::ggml::mxfp8;

constexpr std::uint64_t kWorkspace = 64ULL << 20;
constexpr std::int64_t kWidth = 2560;
constexpr std::int64_t kHc = 4;
constexpr float kEps = 1e-6f;

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

void ExpectNear(const std::vector<float>& got, const std::vector<double>& want, double bound,
                const std::string& what) {
  const double nmse = Nmse(got, want);
  EXPECT_LE(nmse, bound) << what;
  EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); })) << what;
  std::cout << what << ": NMSE " << nmse << " against FP64\n";
}

std::vector<float> Normal(std::uint64_t seed, std::size_t n, float scale = 1.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, scale);
  std::vector<float> out(n);
  for (float& v : out) {
    v = normal(random);
  }
  return out;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::uint16_t Bf16Bits(float v) {
  const auto bits = std::bit_cast<std::uint32_t>(v);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}
float FromBf16(std::uint16_t b) { return std::bit_cast<float>(std::uint32_t{b} << 16U); }
std::vector<std::uint16_t> ToBf16(const std::vector<float>& x) {
  std::vector<std::uint16_t> out(x.size());
  std::ranges::transform(x, out.begin(), Bf16Bits);
  return out;
}

// An E4M3 code's value (bias 7, subnormals, 0x7F / 0xFF NaN).
double E4m3(std::uint8_t b) {
  const unsigned bits = b;
  const int e = static_cast<int>((bits >> 3U) & 15U);
  const int m = static_cast<int>(bits & 7U);
  if (e == 15 && m == 7) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double v =
      e == 0 ? (m / 8.0) * std::ldexp(1.0, -6) : (1.0 + (m / 8.0)) * std::ldexp(1.0, e - 7);
  return (bits & 0x80U) != 0 ? -v : v;
}
double E8m0(std::uint8_t e) { return std::ldexp(1.0, static_cast<int>(e) - 127); }

// MXFP8 rows (RowsLayout) back to doubles: [rows, k].
std::vector<double> Dequantize(const std::vector<std::uint8_t>& blob, std::int64_t k,
                               std::int64_t rows) {
  const mx::RowsLayout l{.k = static_cast<std::uint64_t>(k),
                         .rows = static_cast<std::uint64_t>(rows)};
  std::vector<double> out(static_cast<std::size_t>(k * rows));
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      const std::uint64_t at = llmp::kernels::ggml::moe::SfOffset(
          static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(i / 32),
          static_cast<std::uint64_t>(k / 32));
      out[static_cast<std::size_t>((r * k) + i)] =
          E4m3(blob[mx::RowsLayout::codes() + static_cast<std::size_t>((r * k) + i)]) *
          E8m0(blob[l.scales() + at]);
    }
  }
  return out;
}

// Every value within E4M3's rounding of its block's scale (half a step of
// its binade, or of the subnormals'), each scale the smallest power of two
// that holds the block's largest magnitude at 448, and the padding rows'
// scales zero.
void ExpectQuantized(const std::vector<std::uint8_t>& blob, const std::vector<double>& x,
                     std::int64_t k, std::int64_t rows, const std::string& what) {
  const std::vector<double> q = Dequantize(blob, k, rows);
  const mx::RowsLayout l{.k = static_cast<std::uint64_t>(k),
                         .rows = static_cast<std::uint64_t>(rows)};
  std::size_t bad = 0;
  for (std::int64_t r = 0; r < rows; ++r) {
    for (std::int64_t b = 0; b < k / 32; ++b) {
      double amax = 0.0;
      for (std::int64_t i = 0; i < 32; ++i) {
        amax = std::max(amax, std::abs(x[static_cast<std::size_t>((r * k) + (b * 32) + i)]));
      }
      const double s =
          E8m0(blob[l.scales() + llmp::kernels::ggml::moe::SfOffset(
                                     static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                                     static_cast<std::uint64_t>(k / 32))]);
      // (Margins for x computed in FP64 where the kernel quantized its F32.)
      if (amax > 0.0 && (amax > 448.0 * s * (1.0 + 1e-5) || amax <= 224.0 * s * (1.0 - 1e-5))) {
        ++bad;
      }
      for (std::int64_t i = 0; i < 32; ++i) {
        const auto at = static_cast<std::size_t>((r * k) + (b * 32) + i);
        const double v = std::abs(x[at]) / s;
        const double step =
            v < std::ldexp(1.0, -6) ? std::ldexp(1.0, -9) : std::ldexp(1.0, std::ilogb(v) - 3);
        if (std::abs(q[at] - x[at]) > (0.5 * step * s * (1.0 + 1e-4)) + (1e-6 * std::abs(x[at]))) {
          ++bad;
        }
      }
    }
  }
  for (std::uint64_t r = l.rows; r < mx::PaddedRows(l.rows); ++r) {
    for (std::int64_t b = 0; b < k / 32; ++b) {
      bad += blob[l.scales() +
                  llmp::kernels::ggml::moe::SfOffset(r, static_cast<std::uint64_t>(b),
                                                     static_cast<std::uint64_t>(k / 32))] != 0;
    }
  }
  EXPECT_EQ(bad, 0U) << what;
}

class Qwen38FastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    int major = 0;
    int minor = 0;
    ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
    const Bytes cublas_bytes = CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor));
    auto handle = CublasHandle::Create(
        0, *execution_, stream_, {.base = Allocate(cublas_bytes.value()), .size = cublas_bytes});
    ASSERT_TRUE(handle.has_value()) << (handle ? "" : handle.error().detail);
    cublas_ = std::move(*handle);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = Allocate(kWorkspace), .size = Bytes(kWorkspace)},
                                        cublas_.get());
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(1024).value());
    auto registry = llmp::execution::Registry::Create(kg::Implementations());
    ASSERT_TRUE(registry.has_value());
    registry_ = std::make_unique<llmp::execution::Registry>(std::move(*registry));
  }

  void TearDown() override {
    Finish();
    launch_.reset();
    cublas_.reset();
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
    // Stale bytes would hide an output the kernels leave unwritten.
    EXPECT_EQ(cudaMemset(pointer, 0x7B, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    // The memset runs on the legacy stream, which the provider's
    // non-blocking stream does not wait for: it must land before any kernel
    // writes the memory.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  ggml_context* c() const { return arena_->context(); }
  LaunchContext& launch() { return *launch_; }

  template <typename T = float>
  ggml_tensor* Leaf(ggml_tensor* tensor, const std::vector<T>& data) {
    const std::uint64_t address = Allocate(ggml_nbytes(tensor));
    TensorArena::Bind(tensor, address);
    EXPECT_EQ(data.size() * sizeof(T), ggml_nbytes(tensor));
    EXPECT_EQ(cudaMemcpy(tensor->data, data.data(), ggml_nbytes(tensor), cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return tensor;
  }

  // A packed `type` [ne0, ne1] view of a blob's bytes from `offset` (as
  // the graph views its fusions' blobs).
  ggml_tensor* TypedView(ggml_tensor* blob, ggml_type type, std::int64_t ne0, std::int64_t ne1,
                         std::size_t offset) {
    const std::size_t row = ggml_row_size(type, ne0);
    ggml_tensor* v = ggml_view_1d(
        c(), blob,
        static_cast<std::int64_t>(row * static_cast<std::size_t>(ne1) / ggml_type_size(blob->type)),
        offset);
    v->type = type;
    v->ne[0] = ne0;
    v->ne[1] = ne1;
    v->nb[0] = ggml_type_size(type);
    v->nb[1] = row;
    v->nb[2] = row * static_cast<std::size_t>(ne1);
    v->nb[3] = v->nb[2];
    return v;
  }

  // Places every computed tensor the outputs need, binds the views, and
  // runs the plan the registry binds.
  void Run(std::vector<ggml_tensor*> outputs) {
    const std::vector<ggml_tensor*> nodes = kg::GraphOrder(outputs);
    for (ggml_tensor* node : nodes) {
      if (node->view_src == nullptr && node->data == nullptr) {
        TensorArena::Bind(node, Allocate(ggml_nbytes(node)));
      }
    }
    kg::BindViews(nodes);
    auto plan = kg::PlanGraph(nodes, false, kg::DeviceChoicesOf(launch()));
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    auto bound = kg::BoundGraph::Bind(*registry_, *plan);
    ASSERT_TRUE(bound.has_value()) << bound.error().detail;
    ASSERT_TRUE(bound->Run(launch()).has_value());
  }

  std::string PlannedFor(ggml_tensor* node) {
    auto plan =
        kg::PlanGraph(std::vector<ggml_tensor*>{node}, false, kg::DeviceChoicesOf(launch()));
    return plan && plan->steps.size() == 1 ? std::string(plan->steps[0].implementation) : "";
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<CublasHandle> cublas_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
  std::unique_ptr<llmp::execution::Registry> registry_;
};

// MXFP8 weights of real models' ranges: E4M3 codes (no NaN) and E8M0
// scales about 2^-8.
struct Weights {
  std::vector<std::uint8_t> codes;   // [n, k]
  std::vector<std::uint8_t> scales;  // [n, k / 32]
  double at(std::int64_t row, std::int64_t i, std::int64_t k) const {
    return E4m3(codes[static_cast<std::size_t>((row * k) + i)]) *
           E8m0(scales[static_cast<std::size_t>((row * (k / 32)) + (i / 32))]);
  }
};
Weights RandomWeights(std::uint64_t seed, std::int64_t k, std::int64_t n) {
  std::mt19937 random(static_cast<unsigned>(seed));
  Weights w;
  w.codes.resize(static_cast<std::size_t>(k * n));
  for (std::uint8_t& b : w.codes) {
    do {
      b = static_cast<std::uint8_t>(random() & 0xFFU);
    } while ((static_cast<unsigned>(b) & 0x7FU) == 0x7FU);
  }
  w.scales.resize(static_cast<std::size_t>((k / 32) * n));
  for (std::uint8_t& s : w.scales) {
    s = static_cast<std::uint8_t>(116 + (random() % 6));
  }
  return w;
}

TEST_F(Qwen38FastTest, TheMxfp8ProductIsTheProductOfItsQuantizedOperands) {
  // (Rows below one scale atom, odd tails across atoms, and past 4,096 rows,
  // where the tiles are swizzled.)
  for (const auto& [k, n, t] : {std::tuple<std::int64_t, std::int64_t, std::int64_t>{2560, 640, 9},
                                {2560, 48, 130},
                                {6144, 512, 257},
                                {640, 2560, 33},
                                {640, 256, 4133}}) {
    const std::string what =
        "k " + std::to_string(k) + " n " + std::to_string(n) + " t " + std::to_string(t);
    const std::vector<float> x_h =
        Normal(static_cast<std::uint64_t>(k + n + t), static_cast<std::size_t>(k * t), 2.0f);
    const Weights w = RandomWeights(static_cast<std::uint64_t>(n), k, n);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k, t), x_h);
    ggml_tensor* codes = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, n), w.codes);
    ggml_tensor* scales = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k / 32, n), w.scales);
    ggml_tensor* q = kg::Mxfp8Quantize(c(), x);
    ggml_tensor* swizzled = kg::Mxfp8Swizzle(c(), scales);
    ggml_tensor* y = kg::Mxfp8Gemm(c(), q, codes, swizzled, GGML_TYPE_F32, t);
    ggml_tensor* y16 = kg::Mxfp8Gemm(c(), q, codes, swizzled, GGML_TYPE_BF16, t);
    Run({y, y16});
    EXPECT_EQ(PlannedFor(q), kg::kMxfp8QuantizeName);
    EXPECT_EQ(PlannedFor(swizzled), kg::kMxfp8SwizzleName);
    EXPECT_EQ(PlannedFor(y), kg::kMxfp8GemmName);
    const std::vector<double> x_d(x_h.begin(), x_h.end());
    const auto blob = Download<std::uint8_t>(q);
    ExpectQuantized(blob, x_d, k, t, "quantized " + what);
    // The swizzled scales are the artifact's, each in its place.
    const auto sw = Download<std::uint8_t>(swizzled);
    std::size_t misplaced = 0;
    const auto padded = static_cast<std::int64_t>(mx::PaddedRows(static_cast<std::uint64_t>(n)));
    for (std::int64_t r = 0; r < padded; ++r) {
      for (std::int64_t b = 0; b < k / 32; ++b) {
        const std::uint8_t want =
            r < n ? w.scales[static_cast<std::size_t>((r * (k / 32)) + b)] : 0;
        misplaced += sw[llmp::kernels::ggml::moe::SfOffset(
                         static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(b),
                         static_cast<std::uint64_t>(k / 32))] != want;
      }
    }
    EXPECT_EQ(misplaced, 0U) << what;
    const std::vector<double> xq = Dequantize(blob, k, t);
    std::vector<double> want(static_cast<std::size_t>(n * t));
    std::vector<double> exact(static_cast<std::size_t>(n * t));
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t j = 0; j < n; ++j) {
        double sum = 0.0;
        double full = 0.0;
        for (std::int64_t i = 0; i < k; ++i) {
          const double wv = w.at(j, i, k);
          sum += wv * xq[static_cast<std::size_t>((r * k) + i)];
          full += wv * x_d[static_cast<std::size_t>((r * k) + i)];
        }
        want[static_cast<std::size_t>((r * n) + j)] = sum;
        exact[static_cast<std::size_t>((r * n) + j)] = full;
      }
    }
    const auto got = Download(y);
    ExpectNear(got, want, 1e-10, "product " + what);
    // And against the unquantized activations: MXFP8's rounding.
    ExpectNear(got, exact, 3e-3, "product (activations unquantized) " + what);
    const auto got16 = Download<std::uint16_t>(y16);
    std::vector<float> as_f32(got16.size());
    std::ranges::transform(got16, as_f32.begin(), FromBf16);
    ExpectNear(as_f32, want, 1e-5, "BF16 product " + what);
  }
}

TEST_F(Qwen38FastTest, Mxfp8QuantizationReadsBf16AndStridedRows) {
  constexpr std::int64_t k = 2560;
  constexpr std::int64_t t = 37;
  // Rows of a wider tensor (a stride of k + 64 values), and BF16 rows.
  const std::vector<float> wide_h = Normal(5, static_cast<std::size_t>((k + 64) * t), 30.0f);
  ggml_tensor* wide = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k + 64, t), wide_h);
  ggml_tensor* rows = ggml_view_2d(c(), wide, k, t, wide->nb[1], 0);
  std::vector<float> bf_values(static_cast<std::size_t>(k * t));
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      bf_values[static_cast<std::size_t>((r * k) + i)] =
          FromBf16(Bf16Bits(wide_h[static_cast<std::size_t>((r * (k + 64)) + i)]));
    }
  }
  ggml_tensor* bf = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, t), ToBf16(bf_values));
  ggml_tensor* from_rows = kg::Mxfp8Quantize(c(), rows);
  ggml_tensor* from_bf16 = kg::Mxfp8Quantize(c(), bf);
  Run({from_rows, from_bf16});
  std::vector<double> want(static_cast<std::size_t>(k * t));
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t i = 0; i < k; ++i) {
      want[static_cast<std::size_t>((r * k) + i)] =
          wide_h[static_cast<std::size_t>((r * (k + 64)) + i)];
    }
  }
  ExpectQuantized(Download<std::uint8_t>(from_rows), want, k, t, "strided rows");
  const std::vector<double> bf_d(bf_values.begin(), bf_values.end());
  ExpectQuantized(Download<std::uint8_t>(from_bf16), bf_d, k, t, "BF16 rows");
}

// The hyper-connections' prep: [combine,] norm into BF16 and inject logits.
TEST_F(Qwen38FastTest, HcPrepCombinesNormalizesAndInjects) {
  // Up to 8 tokens the cluster form (HcPrepClusterKernel), past them a
  // block a token.
  for (const auto& [t, combine, inject] : {std::tuple<std::int64_t, bool, bool>{1, true, true},
                                           {8, true, false},
                                           {37, true, true},
                                           {9, false, true},
                                           {5, false, false}}) {
    const std::string what =
        "t " + std::to_string(t) + (combine ? " combine" : "") + (inject ? " inject" : "");
    const std::int64_t wide = kWidth * kHc;
    const auto seed = static_cast<std::uint64_t>(t);
    const auto res_h = Normal(11 + seed, static_cast<std::size_t>(wide * t), 3.0f);
    const auto out_h = Normal(12 + seed, static_cast<std::size_t>(kWidth * t));
    const auto logit_h = Normal(13 + seed, static_cast<std::size_t>(kHc * t), 2.0f);
    auto norm_h = Normal(14, static_cast<std::size_t>(wide), 0.3f);
    for (float& v : norm_h) {
      v += 1.0f;
    }
    const auto inj_f = Normal(15, static_cast<std::size_t>(wide * kHc), 0.02f);
    const auto inj_b = ToBf16(inj_f);
    ggml_tensor* res = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kHc, t), res_h);
    ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, wide), norm_h);
    ggml_tensor* inj_w =
        inject ? Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, wide, kHc), inj_b) : nullptr;
    ggml_tensor* out =
        combine ? Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), out_h) : nullptr;
    ggml_tensor* logits =
        combine ? Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kHc, t), logit_h) : nullptr;
    ggml_tensor* blob = kg::HcPrep(c(), res, norm, inj_w, out, logits, kEps);
    Run({blob});
    EXPECT_EQ(PlannedFor(blob), kg::kHcPrepName);
    const kg::HcPrepLayout l{
        .width = kWidth, .hc = kHc, .t = t, .combine = combine, .inject = inject};
    const auto bytes = Download<std::uint8_t>(blob);
    // FP64: the combined streams, normalized, and the logits.
    std::vector<double> streams(static_cast<std::size_t>(wide * t));
    std::vector<double> normed(streams.size());
    std::vector<double> inj(static_cast<std::size_t>(kHc * t), 0.0);
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t k = 0; k < kHc; ++k) {
        const double g =
            combine ? 2.0 * Sigmoid(logit_h[static_cast<std::size_t>((r * kHc) + k)] / kHc) : 0.0;
        double sum = 0.0;
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const auto at = static_cast<std::size_t>((((r * kHc) + k) * kWidth) + i);
          const double v =
              res_h[at] + (combine ? out_h[static_cast<std::size_t>((r * kWidth) + i)] * g : 0.0);
          streams[at] = v;
          sum += v * v;
        }
        const double s = 1.0 / std::sqrt((sum / kWidth) + kEps);
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const auto at = static_cast<std::size_t>((((r * kHc) + k) * kWidth) + i);
          normed[at] = streams[at] * s * norm_h[static_cast<std::size_t>((k * kWidth) + i)];
        }
      }
      for (std::int64_t m = 0; m < kHc; ++m) {
        double dot = 0.0;
        for (std::int64_t i = 0; i < wide; ++i) {
          dot += static_cast<double>(FromBf16(inj_b[static_cast<std::size_t>((m * wide) + i)])) *
                 normed[static_cast<std::size_t>((r * wide) + i)];
        }
        inj[static_cast<std::size_t>((r * kHc) + m)] = dot;
      }
    }
    if (combine) {
      std::vector<float> got(streams.size());
      std::memcpy(got.data(), bytes.data() + kg::HcPrepLayout::streams(), got.size() * 4);
      ExpectNear(got, streams, 1e-12, "streams " + what);
    }
    std::vector<std::uint16_t> xn(normed.size());
    std::memcpy(xn.data(), bytes.data() + l.normed(), xn.size() * 2);
    std::vector<float> xn_f(xn.size());
    std::ranges::transform(xn, xn_f.begin(), FromBf16);
    ExpectNear(xn_f, normed, 2e-5, "normalized (BF16) " + what);
    if (inject) {
      std::vector<float> got(inj.size());
      std::memcpy(got.data(), bytes.data() + l.logits(), got.size() * 4);
      ExpectNear(got, inj, 1e-9, "inject logits " + what);
    }
  }
}

TEST_F(Qwen38FastTest, HcLoAndMixAreTheirFormulas) {
  constexpr std::int64_t t = 21;
  constexpr std::int64_t rank = 320;
  const std::int64_t wide = kWidth * kHc;
  const auto lo_h = Normal(21, static_cast<std::size_t>(rank * t), 4.0f);
  const auto xn_f = Normal(22, static_cast<std::size_t>(wide * t));
  const auto gate_f = Normal(23, static_cast<std::size_t>(wide * t), 3.0f);
  const auto xn_b = ToBf16(xn_f);
  const auto gate_b = ToBf16(gate_f);
  ggml_tensor* lo = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, rank, t), lo_h);
  ggml_tensor* xn = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, wide, t), xn_b);
  ggml_tensor* gate = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, wide, t), gate_b);
  ggml_tensor* act = kg::HcLo(c(), lo, kHc);
  ggml_tensor* mixed = kg::HcMixBf16(c(), xn, gate, kHc);
  ggml_tensor* blob = kg::HcMixBf16(c(), xn, gate, kHc, true, true);
  Run({act, mixed, blob});
  EXPECT_EQ(PlannedFor(act), kg::kHcLoName);
  EXPECT_EQ(PlannedFor(mixed), kg::kHcMixBf16Name);
  std::vector<double> want_lo(lo_h.size());
  for (std::size_t i = 0; i < lo_h.size(); ++i) {
    const double v = lo_h[i] / static_cast<double>(kHc);
    want_lo[i] = v * Sigmoid(v);
  }
  const auto act_b = Download<std::uint16_t>(act);
  std::vector<float> act_f(act_b.size());
  std::ranges::transform(act_b, act_f.begin(), FromBf16);
  ExpectNear(act_f, want_lo, 2e-5, "silu(lo / hc) in BF16");
  std::vector<double> want(static_cast<std::size_t>(kWidth * t), 0.0);
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t i = 0; i < kWidth; ++i) {
      double sum = 0.0;
      for (std::int64_t k = 0; k < kHc; ++k) {
        const auto at = static_cast<std::size_t>((r * wide) + (k * kWidth) + i);
        sum += static_cast<double>(FromBf16(xn_b[at])) * Sigmoid(FromBf16(gate_b[at]));
      }
      want[static_cast<std::size_t>((r * kWidth) + i)] = sum / kHc;
    }
  }
  const auto got = Download(mixed);
  ExpectNear(got, want, 1e-11, "mix");
  // The blob: the same mix, its MXFP8 quantization and its BF16.
  const kg::HcMixLayout ml{.width = kWidth, .t = t, .bf16 = true};
  const auto bytes = Download<std::uint8_t>(blob);
  std::vector<float> blob_mixed(want.size());
  std::memcpy(blob_mixed.data(), bytes.data() + kg::HcMixLayout::mixed(), blob_mixed.size() * 4);
  EXPECT_EQ(blob_mixed, got);
  const std::vector<std::uint8_t> quantized(
      bytes.begin() + static_cast<std::ptrdiff_t>(ml.quantized()),
      bytes.begin() + static_cast<std::ptrdiff_t>(ml.quantized() + ml.quantized_bytes()));
  const std::vector<double> got_d(got.begin(), got.end());
  ExpectQuantized(quantized, got_d, kWidth, t, "the mix's MXFP8");
  std::vector<std::uint16_t> rounded(want.size());
  std::memcpy(rounded.data(), bytes.data() + ml.rounded(), rounded.size() * 2);
  EXPECT_EQ(rounded, ToBf16(got));
}

TEST_F(Qwen38FastTest, TheRouterPicksTheTopExpertsAndTheSharedGate) {
  constexpr std::int64_t experts = 512;
  constexpr std::int64_t used = 10;
  for (const std::int64_t t : {1, 11, 64}) {
    const auto seed = static_cast<std::uint64_t>(t);
    const auto logits_h = Normal(31 + seed, static_cast<std::size_t>(experts * t), 2.0f);
    const auto x_h = Normal(32 + seed, static_cast<std::size_t>(kWidth * t));
    const auto gate_b = ToBf16(Normal(33, kWidth, 0.05f));
    ggml_tensor* logits = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, experts, t), logits_h);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), x_h);
    ggml_tensor* gate_row = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_BF16, kWidth), gate_b);
    ggml_tensor* routed = kg::MoeRouter(c(), logits, x, gate_row, used);
    Run({routed});
    EXPECT_EQ(PlannedFor(routed), kg::kMoeRouterName);
    const kg::MoeRouterLayout l{.used = used, .t = t};
    const auto blob = Download<std::int32_t>(routed);
    std::vector<float> weights(static_cast<std::size_t>(used * t));
    std::vector<float> gates(static_cast<std::size_t>(t));
    std::memcpy(weights.data(), blob.data() + (l.weights() / 4), weights.size() * 4);
    std::memcpy(gates.data(), blob.data() + (l.gate() / 4), gates.size() * 4);
    std::vector<double> want_w(weights.size());
    std::vector<double> want_g(gates.size());
    for (std::int64_t r = 0; r < t; ++r) {
      std::vector<double> p(static_cast<std::size_t>(experts));
      double most = -1e300;
      for (std::int64_t e = 0; e < experts; ++e) {
        most = std::max(most,
                        static_cast<double>(logits_h[static_cast<std::size_t>((r * experts) + e)]));
      }
      double sum = 0.0;
      for (std::int64_t e = 0; e < experts; ++e) {
        p[static_cast<std::size_t>(e)] =
            std::exp(logits_h[static_cast<std::size_t>((r * experts) + e)] - most);
        sum += p[static_cast<std::size_t>(e)];
      }
      std::vector<std::int32_t> order(static_cast<std::size_t>(experts));
      std::ranges::iota(order, 0);
      std::ranges::stable_sort(order, [&](std::int32_t a, std::int32_t b) {
        return p[static_cast<std::size_t>(a)] > p[static_cast<std::size_t>(b)];
      });
      double picked = 0.0;
      for (std::int64_t j = 0; j < used; ++j) {
        EXPECT_EQ(blob[static_cast<std::size_t>((r * used) + j)],
                  order[static_cast<std::size_t>(j)])
            << "token " << r << " pick " << j;
        picked += p[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] / sum;
      }
      for (std::int64_t j = 0; j < used; ++j) {
        want_w[static_cast<std::size_t>((r * used) + j)] =
            p[static_cast<std::size_t>(order[static_cast<std::size_t>(j)])] / sum /
            std::max(picked, 6.103515625e-5);
      }
      double dot = 0.0;
      for (std::int64_t i = 0; i < kWidth; ++i) {
        dot += static_cast<double>(FromBf16(gate_b[static_cast<std::size_t>(i)])) *
               x_h[static_cast<std::size_t>((r * kWidth) + i)];
      }
      want_g[static_cast<std::size_t>(r)] = dot;
    }
    ExpectNear(weights, want_w, 1e-10, "routing weights t " + std::to_string(t));
    ExpectNear(gates, want_g, 1e-10, "shared gate t " + std::to_string(t));
  }
  // NaN or all -inf logits still pick distinct experts within range (the
  // lowest, all probabilities counted 0), with zero weights.
  std::vector<float> bad_h(static_cast<std::size_t>(experts * 2));
  std::ranges::fill_n(bad_h.begin(), experts, std::numeric_limits<float>::quiet_NaN());
  std::ranges::fill_n(bad_h.begin() + experts, experts, -std::numeric_limits<float>::infinity());
  ggml_tensor* bad = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, experts, 2), bad_h);
  ggml_tensor* bad_x =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 2), Normal(34, 2 * kWidth));
  ggml_tensor* bad_gate =
      Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_BF16, kWidth), ToBf16(Normal(35, kWidth, 0.05f)));
  ggml_tensor* bad_routed = kg::MoeRouter(c(), bad, bad_x, bad_gate, used);
  Run({bad_routed});
  const auto bad_blob = Download<std::int32_t>(bad_routed);
  const kg::MoeRouterLayout bl{.used = used, .t = 2};
  for (std::int64_t r = 0; r < 2; ++r) {
    for (std::int64_t j = 0; j < used; ++j) {
      EXPECT_EQ(bad_blob[static_cast<std::size_t>((r * used) + j)], j) << "token " << r;
      EXPECT_EQ(std::bit_cast<float>(
                    bad_blob[(bl.weights() / 4) + static_cast<std::size_t>((r * used) + j)]),
                0.0f)
          << "token " << r;
    }
  }
}

// A GGUF checkpoint's shared-expert gate row is F32: the same routing, its
// gate logit the F32 row's dot product.
TEST_F(Qwen38FastTest, TheRouterTakesAnF32GateRow) {
  constexpr std::int64_t experts = 512;
  constexpr std::int64_t used = 10;
  for (const std::int64_t t : {1, 11}) {
    const auto seed = static_cast<std::uint64_t>(t);
    const auto logits_h = Normal(41 + seed, static_cast<std::size_t>(experts * t), 2.0f);
    const auto x_h = Normal(42 + seed, static_cast<std::size_t>(kWidth * t));
    const auto gate_h = Normal(43, kWidth, 0.05f);
    ggml_tensor* logits = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, experts, t), logits_h);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), x_h);
    ggml_tensor* f32_row = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kWidth), gate_h);
    ggml_tensor* bf16_row = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_BF16, kWidth), ToBf16(gate_h));
    ggml_tensor* f32_routed = kg::MoeRouter(c(), logits, x, f32_row, used);
    ggml_tensor* bf16_routed = kg::MoeRouter(c(), logits, x, bf16_row, used);
    Run({f32_routed, bf16_routed});
    EXPECT_EQ(PlannedFor(f32_routed), kg::kMoeRouterName);
    const kg::MoeRouterLayout l{.used = used, .t = t};
    const auto f32_blob = Download<std::int32_t>(f32_routed);
    const auto bf16_blob = Download<std::int32_t>(bf16_routed);
    // The picks and weights do not read the gate row.
    const auto gate_at = static_cast<std::ptrdiff_t>(l.gate() / 4);
    EXPECT_TRUE(std::equal(f32_blob.begin(), f32_blob.begin() + gate_at, bf16_blob.begin()));
    std::vector<float> gates(static_cast<std::size_t>(t));
    std::memcpy(gates.data(), f32_blob.data() + gate_at, gates.size() * 4);
    std::vector<double> want(gates.size());
    for (std::int64_t r = 0; r < t; ++r) {
      double dot = 0.0;
      for (std::int64_t i = 0; i < kWidth; ++i) {
        dot += static_cast<double>(gate_h[static_cast<std::size_t>(i)]) *
               x_h[static_cast<std::size_t>((r * kWidth) + i)];
      }
      want[static_cast<std::size_t>(r)] = dot;
    }
    ExpectNear(gates, want, 1e-10, "F32 shared gate t " + std::to_string(t));
  }
}

TEST_F(Qwen38FastTest, GatedDeltaNetTakesBf16RowsAndQuantizesItsGate) {
  constexpr std::int64_t d = 128;
  constexpr std::int64_t heads = 48;
  constexpr std::int64_t channels = 10240;
  constexpr std::int64_t t = 19;
  // The convolution over BF16 rows is its F32 rows' of the same values.
  auto rows_f = Normal(41, static_cast<std::size_t>(channels * t));
  const auto rows_b = ToBf16(rows_f);
  std::ranges::transform(rows_b, rows_f.begin(), FromBf16);
  const auto history_h = Normal(42, static_cast<std::size_t>(3 * channels));
  const auto conv_w = Normal(43, static_cast<std::size_t>(4 * channels), 0.5f);
  ggml_tensor* rows32 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, t), rows_f);
  ggml_tensor* rows16 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, channels, t), rows_b);
  ggml_tensor* history = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 3 * channels), history_h);
  ggml_tensor* weight = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 4, channels), conv_w);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const float eps = kEps / static_cast<float>(d);
  ggml_tensor* conv32 = kg::GdnConv(c(), rows32, history, weight, 4096, d, eps, scale);
  ggml_tensor* conv16 = kg::GdnConv(c(), rows16, history, weight, 4096, d, eps, scale);
  ggml_tensor* kept = kg::GdnHistory(c(), rows16, 3);
  // The gated norm into MXFP8, beside its F32 form.
  const auto o_h = Normal(44, static_cast<std::size_t>(d * heads * t));
  auto norm_h = Normal(45, d, 0.2f);
  for (float& v : norm_h) {
    v += 1.0f;
  }
  const auto z_b = ToBf16(Normal(46, static_cast<std::size_t>(d * heads * t), 2.0f));
  ggml_tensor* o = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, heads, t), o_h);
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, d), norm_h);
  ggml_tensor* z = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, d * heads, t), z_b);
  ggml_tensor* gated = kg::GdnNormGate(c(), o, norm, z, kEps, GGML_TYPE_F32);
  ggml_tensor* gated8 = kg::GdnNormGate(c(), o, norm, z, kEps, GGML_TYPE_I8);
  Run({conv32, conv16, kept, gated, gated8});
  EXPECT_EQ(PlannedFor(kept), kg::kGdnHistoryName);
  EXPECT_EQ(Download(conv16), Download(conv32));
  const auto kept_h = Download(kept);
  std::size_t wrong = 0;
  for (std::int64_t ch = 0; ch < channels; ++ch) {
    for (std::int64_t j = 0; j < 3; ++j) {
      wrong += kept_h[static_cast<std::size_t>((ch * 3) + j)] !=
               rows_f[static_cast<std::size_t>(((t - 3 + j) * channels) + ch)];
    }
  }
  EXPECT_EQ(wrong, 0U);
  std::vector<double> want(o_h.size());
  for (std::int64_t row = 0; row < heads * t; ++row) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
      sum += static_cast<double>(o_h[static_cast<std::size_t>((row * d) + i)]) *
             o_h[static_cast<std::size_t>((row * d) + i)];
    }
    const double s = 1.0 / std::sqrt((sum / d) + kEps);
    for (std::int64_t i = 0; i < d; ++i) {
      const auto at = static_cast<std::size_t>((row * d) + i);
      want[at] = o_h[at] * s * norm_h[static_cast<std::size_t>(i)] * Sigmoid(FromBf16(z_b[at]));
    }
  }
  ExpectNear(Download(gated), want, 1e-11, "gated norm, BF16 gate");
  ExpectQuantized(Download<std::uint8_t>(gated8), want, d * heads, t, "gated norm into MXFP8");
}

// Decode's history (fewer rows than taps): the old history's last taps then
// the rows, from F32 and BF16 rows.
TEST_F(Qwen38FastTest, GdnHistoryShiftsTheOldHistoryForShortChunks) {
  constexpr std::int64_t channels = 10240;
  constexpr std::int64_t taps = 3;
  const auto history_h = Normal(51, static_cast<std::size_t>(taps * channels));
  ggml_tensor* history = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, taps * channels), history_h);
  for (const std::int64_t t : {std::int64_t{1}, std::int64_t{2}}) {
    auto rows_f =
        Normal(52 + static_cast<std::uint64_t>(t), static_cast<std::size_t>(channels * t));
    const auto rows_b = ToBf16(rows_f);
    std::ranges::transform(rows_b, rows_f.begin(), FromBf16);
    ggml_tensor* rows32 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, t), rows_f);
    ggml_tensor* rows16 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, channels, t), rows_b);
    ggml_tensor* kept32 = kg::GdnHistory(c(), rows32, taps, history);
    ggml_tensor* kept16 = kg::GdnHistory(c(), rows16, taps, history);
    Run({kept32, kept16});
    EXPECT_EQ(PlannedFor(kept32), kg::kGdnHistoryName);
    for (ggml_tensor* kept : {kept32, kept16}) {
      const auto got = Download(kept);
      std::size_t wrong = 0;
      for (std::int64_t ch = 0; ch < channels; ++ch) {
        for (std::int64_t j = 0; j < taps; ++j) {
          const std::int64_t i = t + j;
          const float want = i < taps
                                 ? history_h[static_cast<std::size_t>((ch * taps) + i)]
                                 : rows_f[static_cast<std::size_t>(((i - taps) * channels) + ch)];
          wrong += got[static_cast<std::size_t>((ch * taps) + j)] != want;
        }
      }
      EXPECT_EQ(wrong, 0U) << "t " << t;
    }
  }
  // Without the old history, fewer rows than taps are refused.
  ggml_tensor* one = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, 1),
                          std::vector<float>(static_cast<std::size_t>(channels), 1.0f));
  ggml_tensor* bare = kg::GdnHistory(c(), one, taps);
  TensorArena::Bind(bare, Allocate(ggml_nbytes(bare)));
  EXPECT_FALSE(kg::CheckGdnHistory(bare).has_value());
}

// llmp.gdn.step is the columns kernel over the state in place: its
// attention output and the state it leaves equal gated_delta_net's (planned
// as llmp.gated_delta_net.columns) output rows and state rows bit for bit.
// This comparison runs the original pinned GGML CUDA operations rather
// than substituting CPU math. In particular fast-math's FTZ and the
// softplus branch at 20 must agree at every F32 rounding boundary.
TEST_F(Qwen38FastTest, GdnGatesAreTheOriginalPointwiseChainsAndSavedViews) {
  constexpr std::int64_t heads = 48;
  const float tiny = std::numeric_limits<float>::denorm_min();
  const float inf = std::numeric_limits<float>::infinity();
  const std::array edges{0.0f,
                         -0.0f,
                         tiny,
                         -tiny,
                         std::numeric_limits<float>::min(),
                         -std::numeric_limits<float>::min(),
                         -100.0f,
                         -90.0f,
                         -20.0f,
                         std::nextafter(20.0f, 0.0f),
                         20.0f,
                         std::nextafter(20.0f, inf),
                         21.0f,
                         80.0f,
                         100.0f,
                         inf,
                         -inf,
                         0.125f,
                         -0.125f,
                         std::nextafter(1.0f, 0.0f),
                         std::nextafter(1.0f, inf)};
  for (const std::int64_t rows : {1, 2, 3, 4, 8, 16}) {
    SCOPED_TRACE(rows);
    auto alpha_h = Normal(100 + static_cast<std::uint64_t>(rows),
                          static_cast<std::size_t>(heads * rows), 5.0f);
    auto beta_h = Normal(200 + static_cast<std::uint64_t>(rows), alpha_h.size(), 5.0f);
    auto bias_h = Normal(300, static_cast<std::size_t>(heads));
    auto a_h = Normal(400, static_cast<std::size_t>(heads));
    for (std::size_t i = 0; i < edges.size(); ++i) {
      bias_h[i] = 0.0f;
      a_h[i] = -0.5f;
      for (std::int64_t t = 0; t < rows; ++t) {
        const std::size_t at = static_cast<std::size_t>(t * heads) + i;
        alpha_h[at] = edges[i];
        beta_h[at] = edges[(i + static_cast<std::size_t>(t)) % edges.size()];
      }
    }
    // Cancellation and a subnormal add/multiply are also original GPU
    // operations, so this exercises their actual FTZ rather than assuming it.
    bias_h[30] = -16.0f;
    alpha_h[30] = std::nextafter(16.0f, inf);
    bias_h[31] = tiny;
    alpha_h[31] = -tiny;
    a_h[31] = -0.0f;
    // Normal multiplier with a subnormal result, and subnormal input
    // multiplier: both must flush exactly as the original MUL node does.
    alpha_h[32] = 0.0f;
    bias_h[32] = 0.0f;
    a_h[32] = -std::numeric_limits<float>::min();
    alpha_h[33] = 20.0f;
    bias_h[33] = 0.0f;
    a_h[33] = -tiny;
    ggml_tensor* alpha = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, rows), alpha_h);
    ggml_tensor* beta = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, rows), beta_h);
    ggml_tensor* bias = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, heads), bias_h);
    ggml_tensor* a = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, heads), a_h);
    ggml_tensor* ref_gate = ggml_mul(c(), ggml_softplus(c(), ggml_add(c(), alpha, bias)), a);
    ggml_tensor* ref_beta = ggml_sigmoid(c(), beta);
    ggml_tensor* fused = kg::GdnGates(c(), alpha, beta, bias, a);
    const std::size_t stride = ggml_row_size(GGML_TYPE_F32, heads);
    const std::size_t plane = stride * static_cast<std::size_t>(rows);
    ggml_tensor* gate_view =
        ggml_view_4d(c(), fused, 1, heads, rows, 1, sizeof(float), stride, plane, 0);
    ggml_tensor* beta_view =
        ggml_view_4d(c(), fused, 1, heads, rows, 1, sizeof(float), stride, plane, plane);
    std::vector<std::int64_t> ids_h(static_cast<std::size_t>(rows));
    std::ranges::iota(ids_h, 0);
    ggml_tensor* ids = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I64, rows), ids_h);
    const std::vector<float> blank(alpha_h.size(), 0.0f);
    ggml_tensor* gate_saved = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, rows), blank);
    ggml_tensor* beta_saved = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, rows), blank);
    // The graph's Save consumer uses precisely this reshape/set_rows chain.
    ggml_tensor* save_gate =
        ggml_set_rows(c(), gate_saved, ggml_reshape_2d(c(), gate_view, heads, rows), ids);
    ggml_tensor* save_beta =
        ggml_set_rows(c(), beta_saved, ggml_reshape_2d(c(), beta_view, heads, rows), ids);
    Run({ref_gate, ref_beta, save_gate, save_beta});
    EXPECT_EQ(PlannedFor(fused), kg::kGdnGatesName);
    const auto gate_bits = Download<std::uint32_t>(ref_gate);
    const auto beta_bits = Download<std::uint32_t>(ref_beta);
    EXPECT_EQ(Download<std::uint32_t>(gate_view), gate_bits);
    EXPECT_EQ(Download<std::uint32_t>(beta_view), beta_bits);
    EXPECT_EQ(Download<std::uint32_t>(gate_saved), gate_bits);
    EXPECT_EQ(Download<std::uint32_t>(beta_saved), beta_bits);
    const auto fused_bits = Download<std::uint32_t>(fused);
    ASSERT_EQ(fused_bits.size(), gate_bits.size() + beta_bits.size());
    EXPECT_TRUE(std::equal(gate_bits.begin(), gate_bits.end(), fused_bits.begin()));
    EXPECT_TRUE(std::equal(beta_bits.begin(), beta_bits.end(),
                           fused_bits.begin() + static_cast<std::ptrdiff_t>(gate_bits.size())));
  }
}

TEST_F(Qwen38FastTest, GdnGatesRefuseAliasedMisalignedStaleOrTooManyRows) {
  constexpr std::int64_t heads = 48;
  constexpr std::int64_t rows = 2;
  const std::vector<float> values(static_cast<std::size_t>(heads * rows), 1.0f);
  const std::vector<float> weights(static_cast<std::size_t>(heads), 1.0f);
  ggml_tensor* alpha = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, rows), values);
  ggml_tensor* beta = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, rows), values);
  ggml_tensor* bias = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, heads), weights);
  ggml_tensor* a = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, heads), weights);
  ggml_tensor* gates = kg::GdnGates(c(), alpha, beta, bias, a);
  TensorArena::Bind(gates, Allocate(ggml_nbytes(gates)));
  ASSERT_TRUE(kg::CheckGdnGates(gates).has_value());
  void* output = gates->data;
  for (const ggml_tensor* input : {alpha, beta, bias, a}) {
    gates->data = input->data;
    EXPECT_FALSE(kg::CheckGdnGates(gates).has_value());
  }
  gates->data = static_cast<char*>(output) + 1;
  EXPECT_FALSE(kg::CheckGdnGates(gates).has_value());
  gates->data = output;
  ggml_tensor* view = ggml_view_2d(c(), alpha, heads, rows, alpha->nb[1], 0);
  kg::BindViews(std::vector<ggml_tensor*>{view});
  gates->src[0] = view;
  ASSERT_TRUE(kg::CheckGdnGates(gates).has_value());
  void* backing = alpha->data;
  alpha->data = beta->data;  // The view still names the previous backing.
  EXPECT_FALSE(kg::CheckGdnGates(gates).has_value());
  alpha->data = backing;
  gates->src[0] = alpha;
  ASSERT_TRUE(kg::CheckGdnGates(gates).has_value());
  // The CUDA launcher flattens at most sixteen rows. A valid larger
  // descriptor must refuse before launch rather than silently truncate.
  const std::vector<float> wide(static_cast<std::size_t>(heads * 17), 1.0f);
  ggml_tensor* alpha17 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, 17), wide);
  ggml_tensor* beta17 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, heads, 17), wide);
  ggml_tensor* gates17 = kg::GdnGates(c(), alpha17, beta17, bias, a);
  TensorArena::Bind(gates17, Allocate(ggml_nbytes(gates17)));
  EXPECT_FALSE(kg::CheckGdnGates(gates17).has_value());
  EXPECT_FALSE(kg::RunGdnGates(launch(), gates17).has_value());
}

TEST_F(Qwen38FastTest, GdnStepIsTheColumnsRecurrenceInPlace) {
  constexpr std::int64_t d = 128;
  constexpr std::int64_t qk_heads = 16;
  constexpr std::int64_t heads = 48;
  for (const std::int64_t t : {std::int64_t{1}, std::int64_t{4}, std::int64_t{16}}) {
    const std::string what = "t " + std::to_string(t);
    const auto seed = static_cast<std::uint64_t>(t) * 10;
    // Unit query and key heads, as the convolution's L2 norm leaves them.
    auto unit = [&](std::uint64_t s, std::int64_t n) {
      auto x = Normal(s, static_cast<std::size_t>(d * n * t));
      for (std::int64_t h = 0; h < n * t; ++h) {
        double sum = 0.0;
        for (std::int64_t i = 0; i < d; ++i) {
          sum += static_cast<double>(x[static_cast<std::size_t>((h * d) + i)]) *
                 x[static_cast<std::size_t>((h * d) + i)];
        }
        for (std::int64_t i = 0; i < d; ++i) {
          x[static_cast<std::size_t>((h * d) + i)] /= static_cast<float>(std::sqrt(sum));
        }
      }
      return x;
    };
    const auto q_h = unit(seed + 1, qk_heads);
    const auto k_h = unit(seed + 2, qk_heads);
    const auto v_h = Normal(seed + 3, static_cast<std::size_t>(d * heads * t));
    auto g_h = Normal(seed + 4, static_cast<std::size_t>(heads * t), 0.3f);
    for (float& g : g_h) {
      g = -std::abs(g);  // log of a decay in (0, 1]
    }
    auto beta_h = Normal(seed + 5, static_cast<std::size_t>(heads * t));
    for (float& b : beta_h) {
      b = static_cast<float>(Sigmoid(b));
    }
    const auto state_h = Normal(seed + 6, static_cast<std::size_t>(d * d * heads), 0.1f);
    auto leaf = [&](const std::vector<float>& data, std::int64_t n0, std::int64_t n1,
                    std::int64_t n2) {
      return Leaf(ggml_new_tensor_4d(c(), GGML_TYPE_F32, n0, n1, n2, 1), data);
    };
    ggml_tensor* q = leaf(q_h, d, qk_heads, t);
    ggml_tensor* k = leaf(k_h, d, qk_heads, t);
    ggml_tensor* v = leaf(v_h, d, heads, t);
    ggml_tensor* g = leaf(g_h, 1, heads, t);
    ggml_tensor* beta = leaf(beta_h, 1, heads, t);
    ggml_tensor* state_ref = leaf(state_h, d, d, heads);
    ggml_tensor* state = leaf(state_h, d, d, heads);
    ggml_tensor* state_read = leaf(state_h, d, d, heads);
    ggml_tensor* ref = ggml_gated_delta_net(c(), q, k, v, g, beta, state_ref, 1);
    ggml_tensor* step = kg::GdnStep(c(), q, k, v, g, beta, state);
    // A verify's: the same output, the state read and left as it was.
    ggml_tensor* read = kg::GdnStep(c(), q, k, v, g, beta, state_read, false);
    Run({ref, step, read});
    EXPECT_EQ(Download(read), Download(step)) << what;
    EXPECT_EQ(Download(state_read), state_h) << what;
    EXPECT_EQ(PlannedFor(ref), kg::kGatedDeltaNetColumnsName) << what;
    EXPECT_EQ(PlannedFor(step), kg::kGdnStepName) << what;
    const auto ref_h = Download(ref);
    const auto attn = Download(step);
    const auto state_after = Download(state);
    const auto rows = static_cast<std::size_t>(d * heads * t);
    ASSERT_EQ(ref_h.size(), rows + state_after.size()) << what;
    EXPECT_TRUE(std::equal(attn.begin(), attn.end(), ref_h.begin())) << what;
    EXPECT_TRUE(std::equal(state_after.begin(), state_after.end(),
                           ref_h.begin() + static_cast<std::ptrdiff_t>(rows)))
        << what;
    // The reference's state is untouched.
    EXPECT_EQ(Download(state_ref), state_h) << what;
  }
}

// llmp.gemm.bf16's vector form (GemvBf16) at the hyper-connections' and
// the router's shapes, F32 and BF16 out, against FP64; past one column
// (kGemvBf16FastColumns) the same node runs cuBLAS. The one-column shapes
// reach each of the vector kernel's four forms (k 320, 2,560, 10,240 and
// 16,384).
TEST_F(Qwen38FastTest, GemvBf16IsTheProductAtDecodeWidths) {
  struct Shape {
    std::int64_t k, n, t;
    ggml_type out;
  };
  for (const Shape& s : {Shape{10240, 320, 1, GGML_TYPE_F32}, Shape{2560, 512, 1, GGML_TYPE_F32},
                         Shape{16384, 65, 1, GGML_TYPE_BF16}, Shape{10240, 324, 3, GGML_TYPE_F32},
                         Shape{320, 10240, 1, GGML_TYPE_BF16}, Shape{320, 10240, 8, GGML_TYPE_BF16},
                         Shape{2560, 512, 5, GGML_TYPE_F32}, Shape{16384, 64, 2, GGML_TYPE_F32},
                         Shape{320, 10240, 9, GGML_TYPE_BF16}}) {
    const std::string what = "k " + std::to_string(s.k) + " n " + std::to_string(s.n) + " t " +
                             std::to_string(s.t) + (s.out == GGML_TYPE_BF16 ? " BF16" : " F32");
    const auto w_b = ToBf16(
        Normal(static_cast<std::uint64_t>(s.k + s.n), static_cast<std::size_t>(s.k * s.n), 0.05f));
    const auto x_b =
        ToBf16(Normal(static_cast<std::uint64_t>(s.t + 7), static_cast<std::size_t>(s.k * s.t)));
    ggml_tensor* w = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, s.k, s.n), w_b);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, s.k, s.t), x_b);
    ggml_tensor* y = kg::GemvBf16(c(), w, x, s.out);
    ASSERT_TRUE(kg::IsGemvBf16(y));
    Run({y});
    std::vector<double> want(static_cast<std::size_t>(s.n * s.t));
    for (std::int64_t col = 0; col < s.t; ++col) {
      for (std::int64_t row = 0; row < s.n; ++row) {
        double sum = 0.0;
        for (std::int64_t i = 0; i < s.k; ++i) {
          sum += static_cast<double>(FromBf16(w_b[static_cast<std::size_t>((row * s.k) + i)])) *
                 FromBf16(x_b[static_cast<std::size_t>((col * s.k) + i)]);
        }
        want[static_cast<std::size_t>((col * s.n) + row)] = sum;
      }
    }
    if (s.out == GGML_TYPE_BF16) {
      const auto got_b = Download<std::uint16_t>(y);
      std::vector<float> got(got_b.size());
      std::ranges::transform(got_b, got.begin(), FromBf16);
      ExpectNear(got, want, 2e-5, what);
    } else {
      ExpectNear(Download(y), want, 1e-10, what);
    }
  }
}

TEST_F(Qwen38FastTest, WideBf16KeepsEachColumnRecipeThroughChangedInputReplay) {
  // Direct common-kernel controls; no model/source qualification. Include
  // every dispatch boundary, partial output blocks, all supported widths,
  // both destination encodings and a nonzero packed input view offset.
  arena_ = std::make_unique<TensorArena>(TensorArena::Create(4096).value());
  const auto submission = execution_->Submission(stream_);
  ASSERT_TRUE(submission);
  const auto native_stream = reinterpret_cast<cudaStream_t>(submission->handle);
  constexpr std::int64_t n = 17;
  for (const std::int64_t k : {320, 512, 520, 3072, 3080, 10240, 10248, 20480}) {
    for (std::int64_t columns = 1; columns <= kg::kGemvBf16Columns; ++columns) {
      for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
        SCOPED_TRACE(std::format("k={} columns={} type={}", k, columns, static_cast<int>(type)));
        auto values = ToBf16(Normal(static_cast<std::uint64_t>(k + columns),
                                    static_cast<std::size_t>(k * (columns + 1))));
        auto* weights = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, n),
                             ToBf16(Normal(17, static_cast<std::size_t>(k * n))));
        auto* storage = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, columns + 1), values);
        auto* input = ggml_view_2d(c(), storage, k, columns, static_cast<std::size_t>(2 * k),
                                   static_cast<std::size_t>(2 * k));
        auto* wide = kg::GemvBf16(c(), weights, input, type);
        kg::SetGemvBf16Wide(wide);
        const auto bytes = ggml_nbytes(wide);
        const auto guarded = Allocate(bytes + 512);
        TensorArena::Bind(wide, guarded + 256);
        std::vector<ggml_tensor*> scalar;
        for (std::int64_t column = 0; column < columns; ++column) {
          auto* one = ggml_view_2d(c(), input, k, 1, static_cast<std::size_t>(2 * k),
                                   static_cast<std::size_t>(column * 2 * k));
          auto* product = kg::GemvBf16(c(), weights, one, type);
          TensorArena::Bind(product, Allocate(ggml_nbytes(product)));
          scalar.push_back(product);
        }
        auto roots = scalar;
        roots.push_back(wide);
        kg::BindViews(kg::GraphOrder(roots));
        ASSERT_TRUE(kg::CheckGemmBf16(wide));
        ASSERT_TRUE(kg::UsesGemvBf16Vector(wide));
        const auto compare = [&] {
          const auto got = Download<std::uint8_t>(wide);
          const auto row = ggml_row_size(type, n);
          for (std::size_t col = 0; col < scalar.size(); ++col) {
            const auto want = Download<std::uint8_t>(scalar[col]);
            EXPECT_EQ(want.size(), row);
            EXPECT_EQ(std::memcmp(want.data(), got.data() + col * row, row), 0);
          }
          std::array<std::uint8_t, 256> before{}, after{};
          EXPECT_EQ(cudaMemcpy(before.data(), reinterpret_cast<void*>(guarded), 256,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(cudaMemcpy(after.data(), reinterpret_cast<void*>(guarded + 256 + bytes), 256,
                               cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_TRUE(std::ranges::all_of(before, [](auto b) { return b == 0x7B; }));
          EXPECT_TRUE(std::ranges::all_of(after, [](auto b) { return b == 0x7B; }));
        };
        for (auto* one : scalar) ASSERT_TRUE(kg::RunGemmBf16(launch(), one));
        ASSERT_TRUE(kg::RunGemmBf16(launch(), wide));
        compare();
        ASSERT_EQ(cudaStreamBeginCapture(native_stream, cudaStreamCaptureModeThreadLocal),
                  cudaSuccess);
        ASSERT_TRUE(kg::RunGemmBf16(launch(), wide));
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        ASSERT_EQ(cudaStreamEndCapture(native_stream, &graph), cudaSuccess);
        ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
        for (const std::uint64_t seed : {71U, 103U}) {
          values = ToBf16(Normal(seed, values.size()));
          ASSERT_EQ(cudaMemcpy(storage->data, values.data(), ggml_nbytes(storage),
                               cudaMemcpyHostToDevice),
                    cudaSuccess);
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          for (auto* one : scalar) ASSERT_TRUE(kg::RunGemmBf16(launch(), one));
          ASSERT_EQ(cudaGraphLaunch(executable, native_stream), cudaSuccess);
          compare();  // Download retires this same graph before the next upload.
        }
        ASSERT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
      }
    }
  }
}

TEST_F(Qwen38FastTest, OrdinaryWideFlagRetainsItsLargeKBlasFallback) {
  constexpr std::int64_t k = 32768, n = 17, columns = 2;
  auto* weights = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, n),
                       ToBf16(Normal(811, static_cast<std::size_t>(k * n))));
  auto* input = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, columns),
                     ToBf16(Normal(812, static_cast<std::size_t>(k * columns))));
  for (const ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
    auto* ordinary = kg::GemmBf16(c(), weights, input, type);
    auto* wide = kg::GemvBf16(c(), weights, input, type);
    kg::SetGemvBf16Wide(wide);
    EXPECT_FALSE(kg::UsesGemvBf16Vector(wide));
    Run({ordinary, wide});
    EXPECT_EQ(Download<std::uint8_t>(ordinary), Download<std::uint8_t>(wide));
  }
}

TEST_F(Qwen38FastTest, QsaPrepNormalizesAndRotates) {
  constexpr std::int64_t d = 256;
  constexpr std::int64_t heads = 24;
  constexpr std::int64_t t = 13;
  constexpr float base = 10000000.0f;
  const float theta_scale = std::pow(base, -2.0f / 64.0f);
  const auto q_full = Normal(51, static_cast<std::size_t>(2 * d * heads * t), 3.0f);
  auto norm_h = Normal(52, d, 0.2f);
  for (float& v : norm_h) {
    v += 1.0f;
  }
  std::vector<std::int32_t> pos(static_cast<std::size_t>(4 * t));
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t s = 0; s < 4; ++s) {
      pos[static_cast<std::size_t>((s * t) + r)] = static_cast<std::int32_t>(1000 + (r * 311));
    }
  }
  ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 2 * d * heads, t), q_full);
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, d), norm_h);
  ggml_tensor* positions = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4 * t), pos);
  ggml_tensor* q = kg::QsaPrep(c(), x, norm, positions, d, heads, 2 * d, kEps, theta_scale);
  // The gate's product quantization, from the same rows.
  const auto attn_h = Normal(53, static_cast<std::size_t>(d * heads * t));
  ggml_tensor* attn = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, d * heads, t), attn_h);
  ggml_tensor* gated = kg::QsaGateQuantize(c(), attn, x, d);
  Run({q, gated});
  EXPECT_EQ(PlannedFor(q), kg::kQsaPrepName);
  EXPECT_EQ(PlannedFor(gated), kg::kQsaGateQuantizeName);
  std::vector<double> want(static_cast<std::size_t>(d * heads * t));
  std::vector<double> want_gated(want.size());
  for (std::int64_t r = 0; r < t; ++r) {
    for (std::int64_t h = 0; h < heads; ++h) {
      const float* head = q_full.data() + (r * 2 * d * heads) + (h * 2 * d);
      double sum = 0.0;
      for (std::int64_t i = 0; i < d; ++i) {
        sum += static_cast<double>(head[i]) * head[i];
      }
      const double s = 1.0 / std::sqrt((sum / d) + kEps);
      std::vector<double> v(static_cast<std::size_t>(d));
      for (std::int64_t i = 0; i < d; ++i) {
        v[static_cast<std::size_t>(i)] = head[i] * s * norm_h[static_cast<std::size_t>(i)];
      }
      for (std::int64_t i = 0; i < 32; ++i) {
        const double theta = pos[static_cast<std::size_t>(r)] *
                             std::pow(static_cast<double>(theta_scale), static_cast<double>(i));
        const double x0 = v[static_cast<std::size_t>(i)];
        const double x1 = v[static_cast<std::size_t>(i + 32)];
        v[static_cast<std::size_t>(i)] = (x0 * std::cos(theta)) - (x1 * std::sin(theta));
        v[static_cast<std::size_t>(i + 32)] = (x0 * std::sin(theta)) + (x1 * std::cos(theta));
      }
      for (std::int64_t i = 0; i < d; ++i) {
        const auto at = static_cast<std::size_t>((((r * heads) + h) * d) + i);
        want[at] = v[static_cast<std::size_t>(i)];
        want_gated[at] = attn_h[at] * Sigmoid(head[d + i]);
      }
    }
  }
  // (Fast math's sine and cosine at positions in the thousands, as GGML's
  // rope_multi computes them.)
  ExpectNear(Download(q), want, 1e-6, "q norm and rotation");
  ExpectQuantized(Download<std::uint8_t>(gated), want_gated, d * heads, t,
                  "attention times its gate into MXFP8");
}

// The block keys a chunk completes, and only those, pooled, normalized,
// rotated and rounded to BF16 in place; a later one-row step completes the
// next block.
TEST_F(Qwen38FastTest, QsaPoolWritesTheBlocksAChunkCompletes) {
  constexpr std::int64_t d = 128;
  constexpr std::int64_t ratio = 4;
  constexpr std::int64_t cells = 256;
  constexpr std::int64_t n_blocks = cells / ratio;
  constexpr float base = 10000000.0f;
  const float theta_scale = std::pow(base, -2.0f / 64.0f);
  const auto raw_h = Normal(71, static_cast<std::size_t>(d * cells), 2.0f);
  auto norm_h = Normal(72, d, 0.2f);
  for (float& v : norm_h) {
    v += 1.0f;
  }
  constexpr std::uint16_t kUntouched = 0x1234;
  ggml_tensor* raw = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, d, cells), raw_h);
  ggml_tensor* blocks =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, d, n_blocks),
           std::vector<std::uint16_t>(static_cast<std::size_t>(d * n_blocks), kUntouched));
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, d), norm_h);
  const auto positions_of = [&](std::int64_t n_past, std::int64_t t) {
    std::vector<std::int32_t> pos(static_cast<std::size_t>(4 * t));
    for (std::int64_t s = 0; s < 4; ++s) {
      for (std::int64_t r = 0; r < t; ++r) {
        pos[static_cast<std::size_t>((s * t) + r)] = static_cast<std::int32_t>(n_past + r);
      }
    }
    return Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4 * t), pos);
  };
  // Cells 5 .. 14: blocks 1 and 2 complete (block 3 is short of cell 15).
  ggml_tensor* chunk =
      kg::QsaPool(c(), raw, blocks, norm, positions_of(5, 10), ratio, kEps, theta_scale);
  Run({chunk});
  EXPECT_EQ(PlannedFor(chunk), kg::kQsaPoolName);
  const auto want_block = [&](std::int64_t b) {
    std::vector<double> v(static_cast<std::size_t>(d));
    double sum = 0.0;
    for (std::int64_t i = 0; i < d; ++i) {
      double s = 0.0;
      for (std::int64_t k = 0; k < ratio; ++k) {
        s += raw_h[static_cast<std::size_t>((((b * ratio) + k) * d) + i)];
      }
      v[static_cast<std::size_t>(i)] = s / ratio;
      sum += v[static_cast<std::size_t>(i)] * v[static_cast<std::size_t>(i)];
    }
    const double scale = 1.0 / std::sqrt((sum / d) + kEps);
    for (std::int64_t i = 0; i < d; ++i) {
      v[static_cast<std::size_t>(i)] *= scale * norm_h[static_cast<std::size_t>(i)];
    }
    for (std::int64_t i = 0; i < 32; ++i) {
      const double theta = static_cast<double>(b * ratio) *
                           std::pow(static_cast<double>(theta_scale), static_cast<double>(i));
      const double x0 = v[static_cast<std::size_t>(i)];
      const double x1 = v[static_cast<std::size_t>(i + 32)];
      v[static_cast<std::size_t>(i)] = (x0 * std::cos(theta)) - (x1 * std::sin(theta));
      v[static_cast<std::size_t>(i + 32)] = (x0 * std::sin(theta)) + (x1 * std::cos(theta));
    }
    return v;
  };
  const auto check = [&](const std::vector<std::int64_t>& written, const std::string& what) {
    const auto got = Download<std::uint16_t>(blocks);
    for (std::int64_t b = 0; b < n_blocks; ++b) {
      const bool expect = std::ranges::find(written, b) != written.end();
      const auto row =
          std::span(got).subspan(static_cast<std::size_t>(b * d), static_cast<std::size_t>(d));
      if (!expect) {
        EXPECT_TRUE(std::ranges::all_of(row, [](std::uint16_t x) { return x == kUntouched; }))
            << what << ": block " << b << " written";
        continue;
      }
      std::vector<float> values(row.size());
      std::ranges::transform(row, values.begin(), FromBf16);
      ExpectNear(values, want_block(b), 1e-5, std::format("{}: block {}", what, b));
    }
  };
  check({1, 2}, "a chunk");
  // A decode step at 15 completes block 3.
  ggml_tensor* step =
      kg::QsaPool(c(), raw, blocks, norm, positions_of(15, 1), ratio, kEps, theta_scale);
  Run({step});
  check({1, 2, 3}, "a step");
}

namespace {

std::uint32_t HostOrderKey(float f) {
  const auto b = std::bit_cast<std::uint32_t>(f);
  return (b & 0x80000000U) != 0 ? ~b : (b | 0x80000000U);
}

// The selection's host reference: every visible cell's key (its block's
// summed relu scores; the token's own incomplete block first), the width
// best in (key descending, cell ascending) order, then ascending, then -1.
std::vector<std::int32_t> SelectCells(const std::vector<float>& q, const std::vector<float>& keys,
                                      std::int64_t r, std::int64_t pos, std::int64_t ratio,
                                      std::int64_t width, std::int64_t row) {
  constexpr std::int64_t d = 128;
  constexpr std::int64_t heads = 4;
  const std::int64_t own = (pos + 1) / ratio;
  std::vector<std::uint32_t> block_key(static_cast<std::size_t>(own));
  for (std::int64_t b = 0; b < own; ++b) {
    double s = 0.0;
    for (std::int64_t h = 0; h < heads; ++h) {
      double dot = 0.0;
      for (std::int64_t i = 0; i < d; ++i) {
        dot += static_cast<double>(q[static_cast<std::size_t>((((r * heads) + h) * d) + i)]) *
               keys[static_cast<std::size_t>((b * d) + i)];
      }
      s += std::max(0.0, dot);
    }
    block_key[static_cast<std::size_t>(b)] = HostOrderKey(static_cast<float>(s));
  }
  std::vector<std::pair<std::uint32_t, std::int64_t>> cells;
  for (std::int64_t j = 0; j <= pos; ++j) {
    const std::int64_t b = j / ratio;
    cells.emplace_back(b < own ? block_key[static_cast<std::size_t>(b)] : 0xffffffffU, j);
  }
  std::ranges::stable_sort(cells, [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::int32_t> kept;
  for (std::size_t i = 0; i < cells.size() && std::cmp_less(i, width); ++i) {
    kept.push_back(static_cast<std::int32_t>(cells[i].second));
  }
  std::ranges::sort(kept);
  kept.resize(static_cast<std::size_t>(row), -1);
  return kept;
}

}  // namespace

// The selection against its host reference, with values whose sums are
// exact in F32 (so the device's order of sums cannot move a key) and whose
// blocks tie often: one tile and several, the vector scores (up to 16
// tokens) and the tensor-core ones, tokens with fewer visible cells than
// the width and with an incomplete block of their own. The same run twice
// gives the same bits. And adversarial ones: every block tied (zero
// queries) over one tile and over several, so that the tiles' candidates
// and the merge must both keep the lowest cells; tokens whose own block is
// the last of a tile, the first of the next, or none (a complete last
// block at the context's end); a token with one visible cell.
TEST_F(Qwen38FastTest, QsaTopKKeepsTheBestCellsTiesToTheLowerCell) {
  constexpr std::int64_t d = kg::kQsaIndexDim;
  constexpr std::int64_t heads = kg::kQsaIndexHeads;
  constexpr std::int64_t ratio = 4;
  constexpr std::int64_t width = 2051;
  constexpr std::int64_t row = kg::QsaTopKRow(width);
  constexpr float base = 10000000.0f;
  const float theta_scale = std::pow(base, -2.0f / 64.0f);
  // (tokens, blocks, the first token's position, the positions' step,
  // every block tied)
  for (const auto& [t, n_blocks, first, step, tied] :
       {std::tuple{3L, 1024L, 3000L, 5L, false}, std::tuple{20L, 2000L, 5000L, 3L, false},
        std::tuple{1L, 65536L, 262000L, 1L, false}, std::tuple{37L, 20480L, 1000L, 2201L, false},
        std::tuple{9L, 20480L, 81913L, 1L, false}, std::tuple{4L, 576L, 0L, 700L, true},
        std::tuple{6L, 16384L, 32765L, 1L, false}, std::tuple{20L, 20480L, 32760L, 1500L, true},
        std::tuple{1L, 65536L, 262143L, 1L, true}}) {
    SCOPED_TRACE(std::format("{} tokens over {} blocks{}", t, n_blocks, tied ? ", tied" : ""));
    // NOLINTNEXTLINE(bugprone-random-generator-seed): reproducible
    std::mt19937 random(static_cast<unsigned>(n_blocks + t));
    std::vector<float> q(static_cast<std::size_t>(d * heads * t));
    for (float& v : q) {
      v = tied ? 0.0f : static_cast<float>(static_cast<int>(random() % 3) - 1);
    }
    std::vector<float> keys(static_cast<std::size_t>(d * n_blocks));
    for (float& v : keys) {
      v = static_cast<float>(static_cast<int>(random() % 5) - 2) * 0.5f;
    }
    std::vector<std::int32_t> pos(static_cast<std::size_t>(4 * t));
    for (std::int64_t s = 0; s < 4; ++s) {
      for (std::int64_t r = 0; r < t; ++r) {
        pos[static_cast<std::size_t>((s * t) + r)] =
            static_cast<std::int32_t>(std::min(first + (r * step), (n_blocks * ratio) - 1));
      }
    }
    ggml_tensor* q_t = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, heads, t), q);
    ggml_tensor* keys_t = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, d, n_blocks), ToBf16(keys));
    ggml_tensor* pos_t = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4 * t), pos);
    // The selection names its pool, which is not run here: the keys are the
    // test's (the pool would write the blocks the positions complete).
    ggml_tensor* raw = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, d, 4),
                            std::vector<float>(static_cast<std::size_t>(d * 4)));
    ggml_tensor* weight = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, d),
                               std::vector<float>(static_cast<std::size_t>(d), 1.0f));
    ggml_tensor* pool = kg::QsaPool(c(), raw, keys_t, weight, pos_t, ratio, kEps, theta_scale);
    TensorArena::Bind(pool, Allocate(ggml_nbytes(pool)));
    const std::array<ggml_tensor*, 2> outputs = {
        kg::QsaTopK(c(), q_t, keys_t, pos_t, pool, n_blocks, width, ratio),
        kg::QsaTopK(c(), q_t, keys_t, pos_t, pool, n_blocks, width, ratio)};
    for (ggml_tensor* out : outputs) {
      TensorArena::Bind(out, Allocate(ggml_nbytes(out)));
      EXPECT_EQ(PlannedFor(out), kg::kQsaTopKName);
      auto checked = kg::CheckQsaTopK(out);
      ASSERT_TRUE(checked.has_value()) << checked.error().detail;
      ASSERT_TRUE(kg::RunQsaTopK(launch(), out).has_value());
    }
    const auto got = Download<std::int32_t>(outputs[0]);
    EXPECT_EQ(got, Download<std::int32_t>(outputs[1])) << "the same run twice";
    const std::vector<float> rounded = [&] {
      std::vector<float> v(keys.size());
      std::ranges::transform(ToBf16(keys), v.begin(), FromBf16);
      return v;
    }();
    std::size_t wrong = 0;
    for (std::int64_t r = 0; r < t; ++r) {
      const auto want =
          SelectCells(q, rounded, r, pos[static_cast<std::size_t>(r)], ratio, width, row);
      for (std::int64_t i = 0; i < row; ++i) {
        wrong += got[static_cast<std::size_t>((r * row) + i)] != want[static_cast<std::size_t>(i)];
      }
    }
    EXPECT_EQ(wrong, 0U);
  }
}

// Attention over each token's kept cells against FP64 (the query scaled,
// then rounded to F16, as the kernel reads it): a decode step and a verify
// (several shares of a token's cells combined) and a prefill tile (one
// share), tokens with a full row of cells and with fewer. The same run
// twice gives the same bits.
TEST_F(Qwen38FastTest, QsaAttnReadsTheKeptCellsAlone) {
  constexpr std::int64_t d = kg::kQsaAttnHead;
  constexpr std::int64_t heads = 24;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t cells = 4096;
  constexpr std::int64_t row = kg::QsaTopKRow(2051);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const auto k_h = Normal(81, static_cast<std::size_t>(d * kv_heads * cells));
  const auto v_h = Normal(82, static_cast<std::size_t>(d * kv_heads * cells));
  const std::vector<std::uint16_t> k16 = [&] {
    std::vector<std::uint16_t> out(k_h.size());
    std::ranges::transform(k_h, out.begin(), [](float x) { return ggml_fp32_to_fp16(x); });
    return out;
  }();
  const std::vector<std::uint16_t> v16 = [&] {
    std::vector<std::uint16_t> out(v_h.size());
    std::ranges::transform(v_h, out.begin(), [](float x) { return ggml_fp32_to_fp16(x); });
    return out;
  }();
  ggml_tensor* k = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F16, d * kv_heads, cells), k16);
  ggml_tensor* v = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F16, d * kv_heads, cells), v16);
  for (const std::int64_t t : {1L, 3L, 100L}) {
    SCOPED_TRACE(std::format("{} tokens", t));
    const auto q_h =
        Normal(83 + static_cast<std::uint64_t>(t), static_cast<std::size_t>(d * heads * t), 2.0f);
    // NOLINTNEXTLINE(bugprone-random-generator-seed): reproducible
    std::mt19937 random(static_cast<unsigned>(t));
    std::vector<std::int32_t> idx(static_cast<std::size_t>(row * t), -1);
    for (std::int64_t r = 0; r < t; ++r) {
      // Some tokens keep 2,051 cells, the others fewer (17 the fewest).
      const std::int64_t count = r % 3 == 0 ? 2051 : 17 + (r * 5);
      std::vector<std::int32_t> all(static_cast<std::size_t>(cells));
      std::ranges::iota(all, 0);
      std::ranges::shuffle(all, random);
      all.resize(static_cast<std::size_t>(count));
      std::ranges::sort(all);
      std::ranges::copy(all, idx.begin() + (r * row));
    }
    ggml_tensor* q = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, heads, t), q_h);
    ggml_tensor* cells_t = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, row, t), idx);
    ggml_tensor* first = kg::QsaAttn(c(), q, k, v, cells_t, scale);
    ggml_tensor* again = kg::QsaAttn(c(), q, k, v, cells_t, scale);
    Run({first, again});
    EXPECT_EQ(PlannedFor(first), kg::kQsaAttnName);
    const auto got = Download(first);
    const auto got_again = Download(again);
    EXPECT_EQ(std::memcmp(got.data(), got_again.data(), got.size() * sizeof(float)), 0);
    std::vector<double> want(static_cast<std::size_t>(d * heads * t));
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t h = 0; h < heads; ++h) {
        const std::int64_t kh = h / (heads / kv_heads);
        std::vector<double> qd(static_cast<std::size_t>(d));
        for (std::int64_t i = 0; i < d; ++i) {
          qd[static_cast<std::size_t>(i)] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(
              q_h[static_cast<std::size_t>((((r * heads) + h) * d) + i)] * scale));
        }
        std::vector<double> logits;
        std::vector<std::int64_t> kept;
        for (std::int64_t i = 0; i < row; ++i) {
          const std::int32_t cell = idx[static_cast<std::size_t>((r * row) + i)];
          if (cell < 0) {
            break;
          }
          double s = 0.0;
          for (std::int64_t j = 0; j < d; ++j) {
            s += qd[static_cast<std::size_t>(j)] *
                 ggml_fp16_to_fp32(
                     k16[static_cast<std::size_t>((cell * d * kv_heads) + (kh * d) + j)]);
          }
          logits.push_back(s);
          kept.push_back(cell);
        }
        const double m = *std::ranges::max_element(logits);
        double l = 0.0;
        for (double& s : logits) {
          s = std::exp(s - m);
          l += s;
        }
        for (std::int64_t j = 0; j < d; ++j) {
          double o = 0.0;
          for (std::size_t i = 0; i < kept.size(); ++i) {
            o += logits[i] *
                 ggml_fp16_to_fp32(
                     v16[static_cast<std::size_t>((kept[i] * d * kv_heads) + (kh * d) + j)]);
          }
          want[static_cast<std::size_t>((((r * heads) + h) * d) + j)] = o / l;
        }
      }
    }
    ExpectNear(got, want, 1e-5, std::format("sparse attention, {} tokens", t));
  }
}

TEST_F(Qwen38FastTest, QsaAttnRequiresMatchingCacheRowStrides) {
  constexpr std::int64_t d = kg::kQsaAttnHead;
  constexpr std::int64_t cells = 2;
  ggml_tensor* q = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, 1, 1),
                        std::vector<float>(static_cast<std::size_t>(d), 0.0f));
  std::vector<std::int32_t> selected(16, -1);
  selected[0] = cells - 1;
  ggml_tensor* idx = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 16, 1), selected);
  std::array<ggml_tensor*, 2> keys{};
  std::array<ggml_tensor*, 2> values{};
  for (std::size_t layout = 0; layout < keys.size(); ++layout) {
    const std::int64_t stride = d + (layout == 0 ? 0 : 8);
    ggml_tensor* k = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F16, stride, cells),
                          std::vector<std::uint16_t>(static_cast<std::size_t>(stride * cells), 0));
    std::vector<std::uint16_t> v_h(static_cast<std::size_t>(stride * cells), 0);
    std::fill_n(v_h.begin() + stride, d, ggml_fp32_to_fp16(7.0f));
    ggml_tensor* v = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F16, stride, cells), v_h);
    const auto row_bytes = static_cast<std::size_t>(stride) * sizeof(std::uint16_t);
    keys[layout] = ggml_view_2d(c(), k, d, cells, row_bytes, 0);
    values[layout] = ggml_view_2d(c(), v, d, cells, row_bytes, 0);
    // Packed and equally padded caches must both still execute correctly.
    ggml_tensor* out = kg::QsaAttn(c(), q, keys[layout], values[layout], idx, 1.0f);
    Run({out});
    for (const float value : Download(out)) {
      EXPECT_FLOAT_EQ(value, 7.0f);
    }
  }
  // Both mismatch directions must be refused before a launch: one reads
  // padding as values; the other can read beyond the value cache's end.
  for (std::size_t layout = 0; layout < keys.size(); ++layout) {
    ggml_tensor* out = kg::QsaAttn(c(), q, keys[layout], values[1 - layout], idx, 1.0f);
    TensorArena::Bind(out, Allocate(ggml_nbytes(out)));
    const auto checked = kg::CheckQsaAttn(out);
    ASSERT_FALSE(checked.has_value());
    EXPECT_EQ(checked.error().detail, "key and value caches with equal row strides");
  }
}

TEST_F(Qwen38FastTest, TheChecksRefuseWhatTheKernelsCannotRun) {
  auto reason = [](const auto& checked) {
    return checked.has_value() ? std::string("accepted") : checked.error().detail;
  };
  // Quantization rows not whole scale atoms along k.
  ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 96, 4),
                        std::vector<float>(static_cast<std::size_t>(96 * 4), 1.0f));
  ggml_tensor* q = kg::Mxfp8Quantize(c(), x);
  TensorArena::Bind(q, Allocate(ggml_nbytes(q)));
  EXPECT_NE(reason(kg::CheckMxfp8Quantize(q)), "accepted");
  // A product whose weight's scales are another weight's shape.
  ggml_tensor* x2 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 256, 9),
                         std::vector<float>(static_cast<std::size_t>(256 * 9), 1.0f));
  ggml_tensor* codes = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, 256, 64),
                            std::vector<std::int8_t>(static_cast<std::size_t>(256 * 64), 0x38));
  ggml_tensor* other = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I8, 8, 32),
                            std::vector<std::int8_t>(static_cast<std::size_t>(8 * 32), 127));
  ggml_tensor* q2 = kg::Mxfp8Quantize(c(), x2);
  ggml_tensor* sw = kg::Mxfp8Swizzle(c(), other);
  ggml_tensor* y = kg::Mxfp8Gemm(c(), q2, codes, sw, GGML_TYPE_F32, 9);
  for (ggml_tensor* t : {q2, sw, y}) {
    TensorArena::Bind(t, Allocate(ggml_nbytes(t)));
  }
  EXPECT_EQ(reason(kg::CheckMxfp8Quantize(q2)), "accepted");
  EXPECT_NE(reason(kg::CheckMxfp8Gemm(y)), "accepted");
  // A selection wider than its blocks' cells, one past the tiles, one whose
  // pool writes other block keys; attention over more query heads a KV
  // head than a tile's rows.
  ggml_tensor* iq = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 128, 4, 2),
                         std::vector<float>(static_cast<std::size_t>(128 * 4 * 2), 0.0f));
  ggml_tensor* keys = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, 128, 64),
                           std::vector<std::uint16_t>(static_cast<std::size_t>(128 * 64), 0));
  ggml_tensor* foreign = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, 128, 64),
                              std::vector<std::uint16_t>(static_cast<std::size_t>(128 * 64), 0));
  ggml_tensor* raw = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, 256),
                          std::vector<float>(static_cast<std::size_t>(128 * 256), 0.0f));
  ggml_tensor* weight = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 128),
                             std::vector<float>(static_cast<std::size_t>(128), 1.0f));
  ggml_tensor* pos =
      Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 8), std::vector<std::int32_t>(8, 0));
  ggml_tensor* pool = kg::QsaPool(c(), raw, keys, weight, pos, 4, kEps, 0.5f);
  ggml_tensor* pool_other = kg::QsaPool(c(), raw, foreign, weight, pos, 4, kEps, 0.5f);
  for (ggml_tensor* t : {pool, pool_other}) {
    TensorArena::Bind(t, Allocate(ggml_nbytes(t)));
  }
  EXPECT_EQ(reason(kg::CheckQsaPool(pool)), "accepted");
  const auto topk_reason = [&](ggml_tensor* p, std::int64_t n_blocks, std::int64_t width) {
    ggml_tensor* sel = kg::QsaTopK(c(), iq, keys, pos, p, n_blocks, width, 4);
    TensorArena::Bind(sel, Allocate(ggml_nbytes(sel)));
    return reason(kg::CheckQsaTopK(sel));
  };
  EXPECT_EQ(topk_reason(pool, 64, 255), "accepted");
  EXPECT_NE(topk_reason(pool, 64, 257), "accepted");
  EXPECT_NE(topk_reason(pool, 65, 255), "accepted");  // more blocks than the keys hold
  EXPECT_NE(topk_reason(pool_other, 64, 255), "accepted");
  ggml_tensor* aq = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 256, 34, 1),
                         std::vector<float>(static_cast<std::size_t>(256 * 34), 0.0f));
  ggml_tensor* kv = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 512, 64),
                         std::vector<std::uint16_t>(static_cast<std::size_t>(512 * 64), 0));
  ggml_tensor* kept =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 32, 1), std::vector<std::int32_t>(32, -1));
  ggml_tensor* attn = kg::QsaAttn(c(), aq, kv, kv, kept, 0.0625f);
  TensorArena::Bind(attn, Allocate(ggml_nbytes(attn)));
  EXPECT_NE(reason(kg::CheckQsaAttn(attn)), "accepted");
  // More streams than the prep holds.
  ggml_tensor* streams = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1024, 9, 1),
                              std::vector<float>(static_cast<std::size_t>(1024 * 9), 1.0f));
  ggml_tensor* norm = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, std::int64_t{1024} * 9),
                           std::vector<float>(static_cast<std::size_t>(1024 * 9), 1.0f));
  ggml_tensor* prep = kg::HcPrep(c(), streams, norm, nullptr, nullptr, nullptr, kEps);
  TensorArena::Bind(prep, Allocate(ggml_nbytes(prep)));
  EXPECT_NE(reason(kg::CheckHcPrep(prep)), "accepted");
}

// A model-free integration of the common component, typed row producer,
// planner and normal StepProgram. All callbacks and async buffers belong to
// one heap owner; an unproven completion retains that entire owner.
namespace hc_step {
namespace co = llmp::components;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
namespace kn = llmp::kernels;
namespace pr = llmp::providers;
using llmp::format::Encoding;
constexpr std::uint64_t kW = 2560, kStreams = 4, kRank = 320, kHw = kW * kStreams;
constexpr std::uint64_t kRowBytes = kHw * sizeof(float);
constexpr std::uint64_t kMatrixBytes = kHw * kRank * sizeof(std::uint16_t);
constexpr std::uint64_t kActivation = 2U << 20, kPool = 64U << 20;
constexpr std::uint64_t kHost = 64U << 20, kDevice = 192U << 20;
constexpr std::uint64_t kPinned = kMatrixBytes + 8 * kRowBytes + 4 * kW * sizeof(float) + 256;
static_assert(kPinned + (32U << 20) < kHost);
static_assert(sizeof(ir::Node) <= 384 && sizeof(ir::TensorDesc) <= 192);

struct Program {
  explicit Program(co::StepGraph made) : step(std::move(made)) {}

  co::StepGraph step;
  pl::StepShape shape;
  std::optional<pl::StepPlan> plan;
  std::unique_ptr<kn::StepProgram> program;
  std::unique_ptr<kn::Recorded> graph;
  std::uint64_t activation = 0;
  std::uint64_t source = 0;
  float* output = nullptr;
  std::uint64_t output_bytes = 0;
};

struct Environment {
  std::unique_ptr<DeviceExecution> execution;
  StreamId stream{};
  bool stream_ready = false, retained = false;
  std::unique_ptr<CublasHandle> cublas;
  std::unique_ptr<kg::GgmlBackend> backend;
  std::unique_ptr<kn::StepLaunch> launch;
  const kg::GgmlStepLaunch* reused_launch = nullptr;
  std::uint64_t reused_generation = 0;
  std::array<std::unique_ptr<Program>, 3> programs;
  std::array<std::uint64_t, 3> weights{};
  std::vector<std::pair<void*, std::uint64_t>> allocations;
  std::uint64_t allocated = 0, activations = 0, features = 0;
  void* pinned = nullptr;
  float* host_features = nullptr;
  float* host_outputs = nullptr;
  std::int32_t* host_rows = nullptr;
  std::array<float, 2 * kW> previous{};

  static bool Fail(std::string_view what) {
    ADD_FAILURE() << what;
    return false;
  }
  std::uint64_t declaration_bytes = 0, pool_base = 0;
  bool Finish() {
    if (retained) return false;
    if (!stream_ready) return true;
    const auto fence = execution->Record(stream);
    if (!fence) {
      retained = true;
      return Fail("record completion; retaining HC environment");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
      const auto state = execution->Query(*fence);
      if (!state || std::chrono::steady_clock::now() >= deadline) {
        retained = true;
        return Fail("unproven completion; retaining HC environment");
      }
      if (*state == FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution->Release(*fence)) {
      retained = true;
      return Fail("release completed fence");
    }
    return true;
  }
  bool Close() {
    if (!Finish()) return false;
    for (auto& p : programs) p.reset();
    launch.reset();
    if (backend) backend->LaunchReleased();
    backend.reset();
    cublas.reset();
    if (stream_ready && !execution->DestroyStream(stream)) {
      retained = true;
      return Fail("destroy retired stream");
    }
    stream_ready = false;
    for (auto& [address, bytes] : allocations) {
      (void)bytes;
      if (cudaFree(address) != cudaSuccess) {
        retained = true;
        return Fail("release HC device buffer");
      }
      address = nullptr;
    }
    allocations.clear();
    if (pinned) pr::FreePinned(std::exchange(pinned, nullptr));
    execution.reset();
    return true;
  }
  std::uint64_t Allocate(std::uint64_t bytes) {
    if (bytes == 0 || bytes > kDevice - allocated) {
      Fail("HC device allocation envelope");
      return 0;
    }
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, static_cast<std::size_t>(bytes)) != cudaSuccess) {
      Fail("HC device allocation");
      return 0;
    }
    allocations.emplace_back(pointer, bytes);
    allocated += bytes;
    return reinterpret_cast<std::uintptr_t>(pointer);
  }
  bool Copy(pr::NativeStream native, std::uint64_t to, std::uint64_t from, std::uint64_t bytes,
            pr::CopyKind kind) {
    const auto copied = pr::CopyAsync(native, reinterpret_cast<void*>(to),
                                      reinterpret_cast<const void*>(from), bytes, kind);
    if (copied.ok()) return true;
    retained = true;
    return Fail(copied.text());
  }
  bool Upload(std::uint64_t destination, const void* host, std::uint64_t bytes) {
    const auto native = execution->Submission(stream);
    if (!native) {
      retained = true;
      return Fail("upload submission");
    }
    return Copy(*native, destination, reinterpret_cast<std::uintptr_t>(host), bytes,
                pr::CopyKind::kHostToDevice) &&
           Finish();
  }
  void Open(bool declared = false) {
    auto opened = pr::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened);
    execution = std::move(*opened);
    auto made_stream = execution->CreateStream();
    ASSERT_TRUE(made_stream);
    stream = *made_stream;
    stream_ready = true;
    const auto facts = pr::QueryDeviceFacts(0);
    ASSERT_TRUE(facts);
    const auto cublas_bytes =
        CublasHandle::UpstreamWorkspace(static_cast<int>(facts->architecture));
    // Complete device request known before the first allocation. The fixed
    // buffers are reused only after retirement, at most three programs live.
    constexpr auto fixed = kPool + 3 * kActivation + 8 * kRowBytes + 2 * kMatrixBytes + kRowBytes;
    ASSERT_LE(cublas_bytes.value(), kDevice - fixed);
    allocations.reserve(8);
    const auto cublas_base = Allocate(cublas_bytes.value());
    ASSERT_NE(cublas_base, 0U);
    auto handle =
        CublasHandle::Create(0, *execution, stream, {.base = cublas_base, .size = cublas_bytes});
    ASSERT_TRUE(handle) << (handle ? "" : handle.error().detail);
    cublas = std::move(*handle);
    auto made_backend =
        kg::GgmlBackend::Create(0, *execution, stream, declared ? nullptr : cublas.get(),
                                {.lower = {.index_scratch_limit = 1U << 20}});
    ASSERT_TRUE(made_backend) << (made_backend ? "" : made_backend.error());
    backend = std::move(*made_backend);
    if (declared) {
      const auto charge = [this](std::uint64_t bytes) {
        constexpr std::uint64_t ceiling = 1U << 20;
        if (bytes > ceiling - declaration_bytes) return false;
        declaration_bytes += bytes;
        return true;
      };
      const auto refund = [this](std::uint64_t bytes) {
        EXPECT_LE(bytes, declaration_bytes);
        declaration_bytes -= bytes;
      };
      // This allowance is inside the existing 64MiB whole host inventory;
      // the fixed pinned payload plus 32MiB metadata still leave headroom.
      ASSERT_TRUE(backend->Declare({}, cublas_bytes.value(), charge, refund));
      auto owner = backend->Backend().owner;
      ASSERT_NE(owner, nullptr);
      EXPECT_FALSE(owner->Ready());
      ASSERT_TRUE(backend->LendCublas(cublas.get()));
      ASSERT_TRUE(owner->Ready());
    }

    const auto pool = Allocate(kPool);
    pool_base = pool;
    ASSERT_NE(pool, 0U);
    auto made_launch = backend->Launch({.base = pool, .size = Bytes(kPool)});
    ASSERT_TRUE(made_launch) << (made_launch ? "" : made_launch.error());
    launch = std::move(*made_launch);
    activations = Allocate(3 * kActivation);
    features = Allocate(8 * kRowBytes);
    weights = {Allocate(kRowBytes), Allocate(kMatrixBytes), Allocate(kMatrixBytes)};
    ASSERT_NE(activations, 0U);
    ASSERT_NE(features, 0U);
    for (auto weight : weights) ASSERT_NE(weight, 0U);
    auto stage = pr::AllocatePinned(kPinned);
    ASSERT_TRUE(stage);
    pinned = *stage;
    auto* bytes = static_cast<std::byte*>(pinned);
    host_features = reinterpret_cast<float*>(bytes + kMatrixBytes);
    host_outputs = reinterpret_cast<float*>(bytes + kMatrixBytes + 8 * kRowBytes);
    host_rows = reinterpret_cast<std::int32_t*>(bytes + kMatrixBytes + 8 * kRowBytes +
                                                4 * kW * sizeof(float));
    auto* norm = static_cast<float*>(pinned);
    for (std::uint64_t i = 0; i < kHw; ++i) norm[i] = 0.75F + static_cast<float>(i % 17) / 32.0F;
    ASSERT_TRUE(Upload(weights[0], pinned, kRowBytes));
    auto* matrix = static_cast<std::uint16_t*>(pinned);
    for (std::size_t which = 1; which < weights.size(); ++which) {
      for (std::uint64_t i = 0; i < kHw * kRank; ++i) {
        const auto v = static_cast<int>((i * (which == 1 ? 31U : 43U) + i / kRank) % 257U) - 128;
        matrix[i] = Bf16Bits(static_cast<float>(v) / 8192.0F);
      }
      ASSERT_TRUE(Upload(weights[which], pinned, kMatrixBytes));
    }
  }

  bool Covered(std::uint64_t address, std::uint64_t bytes) const {
    for (const auto& [pointer, length] : allocations) {
      const auto base = reinterpret_cast<std::uintptr_t>(pointer);
      if (address >= base && address - base <= length && bytes <= length - (address - base))
        return true;
    }
    return false;
  }
  void Build(std::size_t index, std::span<const std::uint32_t> rows, std::uint64_t source,
             float* output) {
    ASSERT_TRUE(Finish());
    programs[index].reset();
    pl::StepShape shape;
    shape.phase = pl::Phase::kDraft;
    std::vector<co::StepWeight> step_weights;
    std::vector<co::SegmentRows> segments;
    std::uint32_t first = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto slot = static_cast<std::uint8_t>(i == 0 ? 7 : 2);
      shape.segments.push_back({.slot = slot, .rows = rows[i], .outputs = 1});
      segments.push_back({.slot = slot, .first_row = first, .rows = rows[i], .outputs = 1});
      first += rows[i];
    }
    ASSERT_LE(first, 8U);
    llmp::spec::ComponentRef ref{.kind = "hc.mix",
                                 .params = {{"eps", 1e-6},
                                            {"features", false},
                                            {"inject", false},
                                            {"mix", std::string("sigmoid")},
                                            {"rank", std::int64_t{kRank}}}};
    const auto instance =
        co::MakeInstance(ref, {.width = kW, .streams = kStreams}, co::Site::kFinalNorm, 0, 1);
    ASSERT_TRUE(instance) << (instance ? "" : instance.error());
    ir::Builder builder(128);
    co::StepContext context(builder, std::move(segments), false);
    context.SetPhase(pl::Phase::kDraft);
    context.SetComponent(1);
    const auto feature = context.Features(kHw);
    const auto residual = builder.Reshape(feature, ir::ShapeOf({kW, kStreams, first}));
    const std::array formats{ir::Dense(Encoding::kF32, ir::ShapeOf({kHw})),
                             ir::Dense(Encoding::kBF16, ir::ShapeOf({kHw, kRank})),
                             ir::Dense(Encoding::kBF16, ir::ShapeOf({kRank, kHw}))};
    std::array<ir::TensorId, 3> leaves{};
    const auto role_shapes = instance->def->shapes(*instance);
    ASSERT_TRUE(role_shapes);
    ASSERT_EQ(role_shapes->size(), 3U);
    co::Fragment fragment{
        .b = builder,
        .instance = *instance,
        .step = context,
        .weight =
            [&](std::uint16_t role) {
              if (role >= leaves.size()) return ir::TensorId{};
              if (!leaves[role].valid()) {
                const ir::WeightRef binding{.component = 1, .role = role};
                leaves[role] = builder.Weight(formats[role], binding);
                step_weights.push_back({.tensor = leaves[role],
                                        .ref = binding,
                                        .parts = {{std::string(instance->def->roles[role].name),
                                                   llmp::format::PartRole::kData}}});
              }
              return leaves[role];
            },
        .residual = residual};
    auto emitted = instance->def->emit(fragment);
    ASSERT_TRUE(emitted) << (emitted ? "" : emitted.error());
    ASSERT_TRUE(emitted->stream.valid());
    builder.Output(emitted->stream);
    auto inputs = context.TakeInputs();
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    programs[index] = std::make_unique<Program>(co::StepGraph{.graph = std::move(*graph),
                                                              .inputs = std::move(inputs),
                                                              .weights = std::move(step_weights),
                                                              .state = {}});
    auto& p = *programs[index];
    p.shape = std::move(shape);
    p.source = source;
    p.output = output;
    p.activation = activations + index * kActivation;
    // Return the complete F32 component result through ordinary per-owner
    // output slots; this synthetic step has no vocabulary or token choice.
    p.step.logits = emitted->stream;
    ASSERT_LE(p.step.graph.nodes().size(), 64U);
    // The actual catalog has made the typed OutputRows input, not a tagged
    // lookalike. Singleton draft uses its own last-row view instead.
    const auto id =
        std::ranges::find(p.step.inputs, co::InputKind::kOutputRows, &co::StepInput::kind);
    if (rows.size() == 2) {
      ASSERT_NE(id, p.step.inputs.end());
      const auto values = co::InputValues(*id, p.shape, {});
      ASSERT_TRUE(values);
      ASSERT_EQ(*values, (std::vector<std::int64_t>{rows[0] - 1, first - 1}));
      for (std::size_t i = 0; i < values->size(); ++i)
        host_rows[i] = static_cast<std::int32_t>((*values)[i]);
    } else {
      ASSERT_EQ(id, p.step.inputs.end());
    }
    const auto table = backend->Backend();
    const auto& native = table.owner ? table.owner->backend() : table;
    pl::Policy policy{.enabled = {"hc_prep", "owner_row_invariant"}, .disabled = {"rms_norm_mul"}};
    pl::Canonicalize(policy);
    std::vector<ir::TensorId> staged;
    for (const auto& input : p.step.inputs) staged.push_back(input.tensor);
    auto planned = pl::Plan(p.step.graph.Clone(), *native.registry, native.device, policy,
                            {.capturable = true,
                             .staged = std::move(staged),
                             .on_stream = native.on_stream,
                             .written = native.written,
                             .within = native.within,
                             .shape = &p.shape,
                             .activation_alignment = 256});
    ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
    p.plan.emplace(std::move(*planned));
    ASSERT_LE(p.plan->graph.nodes().size(), 128U);
    EXPECT_EQ(std::ranges::count_if(
                  p.plan->graph.nodes(),
                  [](const ir::Node& n) { return n.key == ir::Fused(ir::FusedKind::kHcPrep); }),
              1);
    EXPECT_EQ(std::ranges::count_if(
                  p.plan->graph.nodes(),
                  [](const ir::Node& n) { return n.key == ir::Fused(ir::FusedKind::kHcFold); }),
              1);
    ASSERT_EQ(std::ranges::count_if(
                  p.plan->graph.nodes(),
                  [](const ir::Node& n) { return n.key == ir::Primitive(ir::Op::kMatMul); }),
              2);
    kn::Addresses addresses{.weight =
                                [this](const ir::WeightRef& binding, llmp::format::PartRole part) {
                                  return binding.component == 1 && binding.role < weights.size() &&
                                                 part == llmp::format::PartRole::kData
                                             ? weights[binding.role]
                                             : 0;
                                },
                            .weight_rows = {},
                            .state = {},
                            .save = {},
                            .activation_bytes = kActivation,
                            .activation_alignment = 256};
    auto measure = backend->Lower({*p.plan, p.step, p.shape, addresses});
    ASSERT_TRUE(measure) << (measure ? "" : measure.error());
    ASSERT_LE((*measure)->scratch_bytes(), kActivation);
    ASSERT_LE((*measure)->pool_bytes(), kPool);
    ASSERT_LE((*measure)->host_bytes(), 1U << 20);
    addresses.activations = p.activation;
    auto bound = backend->Lower({*p.plan, p.step, p.shape, addresses});
    ASSERT_TRUE(bound) << (bound ? "" : bound.error());
    p.program = std::move(*bound);
    ASSERT_EQ(p.program->scratch_bytes(), (*measure)->scratch_bytes());
    ASSERT_EQ(p.program->pool_bytes(), (*measure)->pool_bytes());
    ASSERT_EQ(p.program->host_bytes(), (*measure)->host_bytes());
    ASSERT_EQ(p.program->inputs().size(), (*measure)->inputs().size());
    ASSERT_EQ(p.program->outputs().size(), rows.size());
    ASSERT_EQ((*measure)->outputs().size(), rows.size());
    ASSERT_EQ(p.program->inputs().size(), rows.size() == 2 ? 2U : 1U);
    const auto feature_slot =
        std::ranges::find((*measure)->inputs(), kn::InputKind::kFeatures, &kn::InputSlot::kind);
    ASSERT_NE(feature_slot, (*measure)->inputs().end());
    const auto mapped_feature = p.plan->tensors.at(feature.v);
    const auto root = p.plan->graph.root(mapped_feature);
    const auto place = std::ranges::find(p.plan->placement.offsets, root,
                                         &std::pair<ir::TensorId, std::uint64_t>::first);
    ASSERT_NE(place, p.plan->placement.offsets.end());
    const auto feature_offset = place->second + p.plan->graph.tensor(mapped_feature).view_offset;
    ASSERT_GE(feature_slot->device_address, feature_offset);
    const auto measured_base = feature_slot->device_address - feature_offset;
    const auto same_place = [&](std::uint64_t actual, std::uint64_t measured, std::uint64_t bytes) {
      return actual >= p.activation && actual - p.activation <= kActivation &&
             bytes <= kActivation - (actual - p.activation) && measured >= measured_base &&
             actual - p.activation == measured - measured_base;
    };
    const auto metadata = [](const kn::InputSlot& x) {
      return std::tie(x.bytes, x.tensor, x.segment, x.kind, x.cells, x.read, x.window, x.mask_rows,
                      x.outputs, x.sections, x.count, x.ratio, x.overlap, x.ring, x.ahead,
                      x.commit_save, x.component, x.visible_capacity);
    };
    for (std::size_t i = 0; i < p.program->inputs().size(); ++i) {
      const auto& a = p.program->inputs()[i];
      const auto& m = (*measure)->inputs()[i];
      ASSERT_EQ(metadata(a), metadata(m));
      ASSERT_EQ(a.segment, ir::kJoinedSegment);
      ASSERT_TRUE(a.kind == kn::InputKind::kFeatures || a.kind == kn::InputKind::kOutputRows);
      ASSERT_TRUE(same_place(a.device_address, m.device_address, a.bytes));
      if (a.kind == kn::InputKind::kFeatures) {
        ASSERT_EQ(a.tensor, feature);
        ASSERT_EQ(a.bytes, first * kRowBytes);
      } else {
        ASSERT_NE(id, p.step.inputs.end());
        ASSERT_EQ(a.tensor, id->tensor);
        ASSERT_EQ(a.bytes, rows.size() * sizeof(std::int32_t));
      }
    }
    p.output_bytes = rows.size() * kW * sizeof(float);
    std::uint64_t first_output = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const auto& a = p.program->outputs()[i];
      const auto& m = (*measure)->outputs()[i];
      ASSERT_EQ(a.tensor, emitted->stream);
      ASSERT_EQ(a.tensor, m.tensor);
      ASSERT_EQ(a.bytes, m.bytes);
      ASSERT_EQ(a.bytes, kW * sizeof(float));
      ASSERT_EQ(a.segment, i);
      ASSERT_EQ(a.segment, m.segment);
      ASSERT_FALSE(a.features);
      ASSERT_EQ(a.features, m.features);
      ASSERT_TRUE(same_place(a.device_address, m.device_address, a.bytes));
      if (i == 0) first_output = a.device_address;
      ASSERT_EQ(a.device_address, first_output + i * kW * sizeof(float));
    }
    ASSERT_EQ(p.program->tensors().size(), (*measure)->tensors().size());
    for (std::size_t i = 0; i < p.program->tensors().size(); ++i) {
      const auto& a = p.program->tensors()[i];
      const auto& m = (*measure)->tensors()[i];
      ASSERT_EQ(a.bytes, m.bytes);
      ASSERT_EQ(a.role, m.role);
      ASSERT_FALSE(a.routed_experts);
      ASSERT_EQ(a.routed_experts, m.routed_experts);
      if (a.role == ir::Role::kWeight) {
        ASSERT_EQ(a.address, m.address);
        ASSERT_TRUE(Covered(a.address, a.bytes));
      } else {
        ASSERT_EQ(a.role, ir::Role::kActivation);
        ASSERT_TRUE(same_place(a.address, m.address, a.bytes));
      }
    }
    const auto installed = p.program->Bind(*launch);
    ASSERT_TRUE(installed) << (installed ? "" : installed.error().detail);
    ASSERT_NE(p.program->bound(), nullptr);
    std::size_t products = 0;
    for (std::size_t op = 0; op < p.program->bound()->size(); ++op) {
      const auto& implementation = p.program->bound()->at(op);
      if (implementation.operation == llmp::execution::Operation::kMatMul) {
        ASSERT_EQ(implementation.name, kg::kGemmBf16Name);
        ++products;
      }
    }
    ASSERT_EQ(products, 2U);
  }

  std::expected<void, kn::StepFailure> Body(Program& p, pr::NativeStream native) {
    for (const auto& input : p.program->inputs()) {
      const bool feature = input.kind == kn::InputKind::kFeatures;
      if (!Copy(native, input.device_address,
                feature ? p.source : reinterpret_cast<std::uintptr_t>(host_rows), input.bytes,
                feature ? pr::CopyKind::kDeviceToDevice : pr::CopyKind::kHostToDevice))
        return std::unexpected(
            kn::StepFailure{.error = kn::StepError::kUnknown, .detail = "input copy"});
    }
    const auto ran = p.program->Run(*launch, native);
    if (!ran) {
      retained = true;
      return ran;
    }
    std::uint64_t output_at = reinterpret_cast<std::uintptr_t>(p.output);
    for (const auto& output : p.program->outputs()) {
      if (!Copy(native, output_at, output.device_address, output.bytes,
                pr::CopyKind::kDeviceToHost))
        return std::unexpected(
            kn::StepFailure{.error = kn::StepError::kUnknown, .detail = "output copy"});
      output_at += output.bytes;
    }
    return {};
  }
  void Execute(int variant, std::span<const std::uint32_t> rows) {
    ASSERT_TRUE(Finish());
    const auto total = rows[0] + rows[1];
    for (std::uint32_t r = 0; r < total; ++r) {
      const bool selected = r == rows[0] - 1 || r == total - 1;
      // Variant3 preserves selected rows from2, changing all others. Variants
      // 1/2 perturb individual coordinates/signs, not RMS-cancelled amplitude.
      const int change = variant == 3 && selected ? 2 : variant;
      for (std::uint64_t i = 0; i < kHw; ++i) {
        const auto v = static_cast<int>((i * 17 + r * 43 + i / 13) % 251) - 125;
        const auto perturb = static_cast<int>((i + 3 * r) % 11) - 5;
        host_features[r * kHw + i] =
            static_cast<float>(v) / 64.0F + static_cast<float>(change * perturb) / 16.0F;
      }
    }
    ASSERT_TRUE(Upload(features, host_features, total * kRowBytes));
    const auto native = execution->Submission(stream);
    if (!native) retained = true;
    ASSERT_TRUE(native);
    auto& joined = *programs[0];
    if (variant == 1) {
      auto captured =
          joined.program->Record(*launch, *native, [&] { return Body(joined, *native); });
      if (!captured) retained = true;
      ASSERT_TRUE(captured) << (captured ? "" : captured.error().detail);
      joined.graph = std::move(*captured);
      ASSERT_GT(joined.graph->nodes(), 0U);
    }
    if (variant == 0) {
      const auto ran = Body(joined, *native);
      ASSERT_TRUE(ran) << (ran ? "" : ran.error().detail);
    } else {
      ASSERT_NE(joined.graph, nullptr);
      const auto* identity = joined.graph.get();
      const auto ran = launch->Launch(*native, *joined.graph);
      if (!ran) retained = true;
      ASSERT_TRUE(ran) << (ran ? "" : ran.error().detail);
      ASSERT_EQ(joined.graph.get(), identity);
    }
    for (std::size_t i = 1; i < programs.size(); ++i) {
      const auto ran = Body(*programs[i], *native);
      ASSERT_TRUE(ran) << (ran ? "" : ran.error().detail);
    }
    ASSERT_TRUE(Finish());
    for (std::size_t owner = 0; owner < 2; ++owner) {
      const auto* actual = joined.output + owner * kW;
      const auto* expected = programs[owner + 1]->output;
      ASSERT_EQ(std::memcmp(actual, expected, kW * sizeof(float)), 0) << "owner " << owner;
      ASSERT_TRUE(
          std::ranges::all_of(std::span(actual, kW), [](float x) { return std::isfinite(x); }));
      if (variant == 1 || variant == 2)
        EXPECT_NE(std::memcmp(actual, previous.data() + owner * kW, kW * sizeof(float)), 0);
      if (variant == 3)
        EXPECT_EQ(std::memcmp(actual, previous.data() + owner * kW, kW * sizeof(float)), 0);
    }
    std::memcpy(previous.data(), joined.output, sizeof(previous));
  }
};

class SelectedHcStepProgramTest : public ::testing::Test {
 protected:
  void SetUp() override { environment = std::make_unique<Environment>(); }
  void TearDown() override {
    if (!environment->Close()) std::ignore = environment.release();
  }
  std::unique_ptr<Environment> environment;
};

TEST_F(SelectedHcStepProgramTest, JoinedSelectedRowsMatchScalarProgramsAndReplayChangedInputs) {
  auto& e = *environment;
  ASSERT_NO_FATAL_FAILURE(e.Open());
  for (const auto rows : {std::array{1U, 1U}, std::array{1U, 3U}, std::array{3U, 1U},
                          std::array{3U, 3U}, std::array{4U, 4U}}) {
    SCOPED_TRACE(std::format("{}+{}", rows[0], rows[1]));
    ASSERT_TRUE(e.Finish());
    for (auto& p : e.programs) p.reset();
    ASSERT_NO_FATAL_FAILURE(e.Build(0, rows, e.features, e.host_outputs));
    const auto first_last = rows[0] - 1;
    const auto second_last = rows[0] + rows[1] - 1;
    ASSERT_NO_FATAL_FAILURE(
        e.Build(1, std::array{1U}, e.features + first_last * kRowBytes, e.host_outputs + 2 * kW));
    ASSERT_NO_FATAL_FAILURE(
        e.Build(2, std::array{1U}, e.features + second_last * kRowBytes, e.host_outputs + 3 * kW));
    const kn::Recorded* captured = nullptr;
    for (int variant = 0; variant < 4; ++variant) {
      SCOPED_TRACE(variant);
      ASSERT_NO_FATAL_FAILURE(e.Execute(variant, rows));
      if (variant == 1) captured = e.programs[0]->graph.get();
      if (variant >= 2) ASSERT_EQ(e.programs[0]->graph.get(), captured);
      ASSERT_FALSE(e.retained);
    }
  }
  ASSERT_TRUE(e.Finish());
}

TEST_F(SelectedHcStepProgramTest, OwnedBackendChecksPreparationAndRecordedReadiness) {
  auto& e = *environment;
  ASSERT_NO_FATAL_FAILURE(e.Open(true));
  auto owner = e.backend->Backend().owner;
  ASSERT_NE(owner, nullptr);
  const auto identity = owner->Ready();
  ASSERT_TRUE(identity);
  const std::array rows{1U, 3U};
  ASSERT_NO_FATAL_FAILURE(e.Build(0, rows, e.features, e.host_outputs));
  ASSERT_NO_FATAL_FAILURE(e.Build(1, std::array{1U}, e.features, e.host_outputs + 2 * kW));
  ASSERT_NO_FATAL_FAILURE(
      e.Build(2, std::array{1U}, e.features + 3 * kRowBytes, e.host_outputs + 3 * kW));
  for (int variant = 0; variant < 3; ++variant) ASSERT_NO_FATAL_FAILURE(e.Execute(variant, rows));
  ASSERT_TRUE(e.Finish());
  EXPECT_EQ(owner->Ready(), identity);
  const auto native = e.execution->Submission(e.stream);
  ASSERT_TRUE(native);
  // No launch is attempted after invalidation, including the captured path
  // which bypasses StepProgram::Run. All buffers remain alive until Close.
  e.backend->LaunchReleased();
  EXPECT_FALSE(owner->Ready());
  EXPECT_FALSE(e.programs[0]->program->Run(*e.launch, *native));
  EXPECT_FALSE(e.launch->Launch(*native, *e.programs[0]->graph));
  EXPECT_GT(e.declaration_bytes, 0U);
  ASSERT_TRUE(e.backend->LendCublas(e.cublas.get()));
  EXPECT_EQ(owner->Ready(), identity);
  auto replacement = e.backend->Launch({.base = e.pool_base + 256, .size = Bytes(kPool - 256)});
  ASSERT_TRUE(replacement);
  // Identical durable configuration cannot revive the old launch/recording.
  EXPECT_FALSE(e.launch->Launch(*native, *e.programs[0]->graph));
  EXPECT_FALSE(e.programs[0]->program->Run(*e.launch, *native));
  EXPECT_FALSE(e.programs[0]->program->Run(**replacement, *native));
  EXPECT_FALSE((*replacement)->Launch(*native, *e.programs[0]->graph));
  ASSERT_TRUE(e.programs[0]->program->Bind(**replacement));
  ASSERT_TRUE(e.programs[0]->program->Run(**replacement, *native));
  ASSERT_TRUE(e.Finish());
  replacement->reset();
  e.backend->LaunchReleased();
  ASSERT_TRUE(e.backend->LendCublas(e.cublas.get()));
  // Deterministically reproduce allocator address reuse. The retired original
  // launch's storage is reconstructed after a fence, with a distinct epoch.
  // Callback state belongs to the complete heap environment on every exit.
  auto context = LaunchContext::Create(0, *e.execution, e.stream,
                                       {.base = e.pool_base, .size = Bytes(kPool)}, e.cublas.get());
  ASSERT_TRUE(context);
  auto* storage = static_cast<kg::GgmlStepLaunch*>(e.launch.release());
  e.reused_launch = storage;
  e.reused_generation = 2;
  std::destroy_at(storage);
  auto* reused =
      std::construct_at(storage, std::move(*context), std::unique_ptr<kg::SelectedLaunch>{}, owner,
                        &e.reused_launch, &e.reused_generation, e.reused_generation);
  e.launch.reset(reused);
  ASSERT_EQ(e.programs[0]->graph->owner(), e.launch.get());
  EXPECT_FALSE(e.launch->Launch(*native, *e.programs[0]->graph));
  owner.reset();
  ASSERT_TRUE(e.Close());
  EXPECT_EQ(e.declaration_bytes, 0U);
}

}  // namespace hc_step

}  // namespace
