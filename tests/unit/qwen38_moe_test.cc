// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's routed experts over the CUTLASS layout (kernels/ggml/
// moe_layout.h, llmp_ops.h) on a GB10 (label `gpu`), at the model's
// expert shapes (hidden 640, width 2560) with 16 experts, 10 selected:
// - the layout conversion is lossless: GGML's blocks, converted and back,
//   are the same bytes;
// - the routing sorts every slot by expert, token order within an expert;
// - the grouped GEMM equals the FP64 product of its quantized inputs (the
//   activations' codes and scales as quantized, the weights as stored) to
//   BF16's rounding;
// - the whole wider prefill path (route, quantize, gate and up GEMM, SwiGLU and
//   quantize, down GEMM, weighted sum with the shared expert) is within
//   NMSE 1e-3 of GGML's MMQ path over the same weights in GGML's layout
//   (which quantizes the activations the same way; kAgainstMmq), and no
//   further from the FP64 reference than that path (within 5%, and 3e-2:
//   FP4 activations); small rows retain the same absolute FP64 and
//   BF16-product bounds;
// - decode's vector products (activations quantized to 8 bits as MMVQ
//   quantizes them) are within upstream's quantized bound (5e-4) of the
//   FP64 product, and a wave's expert-major form (5 to 16 tokens) is the
//   per-slot form bit for bit.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/moe_cutlass.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

using llmp::base::Bytes;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::DeviceExecution;
using llmp::providers::FenceState;
using llmp::providers::StreamId;
namespace kg = llmp::kernels::ggml;
namespace moe = llmp::kernels::ggml::moe;

constexpr std::int64_t kExperts = 16;
constexpr std::int64_t kUsed = 10;
constexpr std::int64_t kFfn = 640;
constexpr std::int64_t kWidth = 2560;
constexpr std::uint64_t kWorkspace = 128ULL << 20;
// The prefill path against GGML's MMQ path: both quantize the activations
// the same way, but the grouped GEMM's BF16 gate and up products move some
// SwiGLU outputs across an FP4 rounding boundary when they are quantized
// again for the down projection (measured 5.1e-4 to 5.7e-4 here, each path
// about 2e-2 from FP64).
constexpr double kAgainstMmq = 1e-3;
// Decode's vector products quantize the activations to 8 bits as GGML's
// MMVQ does (a scale per 16 values rather than per 32): upstream's bound for
// quantized products.
constexpr double kGemvNmse = 5e-4;

double E4m3(std::uint8_t b) {
  const int sign = (b & 0x80) != 0 ? -1 : 1;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  if ((b & 0x7F) == 0x7F) {
    return std::nan("");
  }
  if (e == 0) {
    return sign * std::ldexp(m / 8.0, -6);
  }
  return sign * std::ldexp(1.0 + (m / 8.0), e - 7);
}

double E2m1(std::uint8_t code) {
  constexpr std::array<double, 8> kValues = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  return ((code & 8) != 0 ? -1.0 : 1.0) * kValues[code & 7];
}

double Nmse(const std::vector<float>& got, const std::vector<double>& want) {
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const double d = static_cast<double>(got[i]) - want[i];
    error += d * d;
    norm += want[i] * want[i];
  }
  return norm > 0.0 ? error / norm : error;
}

// One projection's weights, ModelOpt style: n rows of k E2M1 codes (element
// 2i in the low nibble of byte i) and E4M3 scales per 16.
struct Weights {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> scales;
  std::int64_t n = 0;
  std::int64_t k = 0;
  double At(std::int64_t row, std::int64_t i) const {
    static const auto kCodes = [] {
      std::array<double, 16> v{};
      for (std::size_t c = 0; c < v.size(); ++c) {
        v[c] = E2m1(static_cast<std::uint8_t>(c));
      }
      return v;
    }();
    static const auto kScales = [] {
      std::array<double, 256> v{};
      for (std::size_t c = 0; c < v.size(); ++c) {
        v[c] = E4m3(static_cast<std::uint8_t>(c));
      }
      return v;
    }();
    const std::uint8_t byte = codes[static_cast<std::size_t>((row * k / 2) + (i / 2))];
    const auto code = static_cast<std::size_t>(i % 2 == 0 ? byte & 15 : byte >> 4);
    return kCodes[code] * kScales[scales[static_cast<std::size_t>((row * k / 16) + (i / 16))]];
  }
};

std::size_t U(std::int64_t v) { return static_cast<std::size_t>(v); }

Weights Random(std::int64_t seed, std::int64_t n, std::int64_t k) {
  std::mt19937_64 random(U(seed));
  Weights w;
  w.n = n;
  w.k = k;
  w.codes.resize(static_cast<std::size_t>(n * k / 2));
  w.scales.resize(static_cast<std::size_t>(n * k / 16));
  for (auto& b : w.codes) {
    b = static_cast<std::uint8_t>(random() & 0xFF);
  }
  for (auto& s : w.scales) {
    s = static_cast<std::uint8_t>(((1 + (random() % 5)) << 3) | (random() % 8));  // 2^-6 .. 2^-2
  }
  return w;
}

// Into GGML's block_nvfp4 rows (as the importer repacks them).
std::vector<std::uint8_t> ToGgml(const Weights& w) {
  std::vector<std::uint8_t> out;
  for (std::int64_t r = 0; r < w.n; ++r) {
    for (std::int64_t blk = 0; blk < w.k / 64; ++blk) {
      for (std::int64_t s = 0; s < 4; ++s) {
        out.push_back(w.scales[static_cast<std::size_t>((r * w.k / 16) + (blk * 4) + s)]);
      }
      for (std::int64_t sub = 0; sub < 4; ++sub) {
        const auto element = [&](std::int64_t e) {
          const std::int64_t i = (blk * 64) + (sub * 16) + e;
          const std::uint8_t byte = w.codes[static_cast<std::size_t>((r * w.k / 2) + (i / 2))];
          return static_cast<std::uint8_t>(i % 2 == 0 ? byte & 15 : byte >> 4);
        };
        for (std::int64_t j = 0; j < 8; ++j) {
          out.push_back(static_cast<std::uint8_t>(element(j) | (element(j + 8) << 4)));
        }
      }
    }
  }
  return out;
}

std::vector<float> Normal(std::int64_t seed, std::size_t n, float scale) {
  std::mt19937_64 random(U(seed));
  std::normal_distribution<float> normal(0.0f, scale);
  std::vector<float> out(n);
  for (float& v : out) {
    v = normal(random);
  }
  return out;
}

class Qwen38MoeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = Allocate(kWorkspace), .size = Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(1024).value());
    auto registry = llmp::execution::Registry::Create(kg::Implementations());
    ASSERT_TRUE(registry.has_value());
    registry_ = std::make_unique<llmp::execution::Registry>(std::move(*registry));
    // The experts' slab in GGML's layout: gate, up and down slices at the
    // artifact's offsets, a stride of the layout rounded as the harness's.
    for (std::int64_t e = 0; e < kExperts; ++e) {
      gate_.push_back(Random(100 + e, kFfn, kWidth));
      up_.push_back(Random(200 + e, kFfn, kWidth));
      down_.push_back(Random(300 + e, kWidth, kFfn));
    }
    // The down slices' 640-value rows read up to 216 bytes past each slice
    // (GGML's 512-value steps); the stride holds them, and is whole 36-byte
    // blocks and 16 bytes (a multiple of 144), as the harness's.
    const std::size_t slice = ToGgml(gate_[0]).size();
    stride_ = ((3 * slice) + 256 + 143) / 144 * 144;
    std::vector<std::uint8_t> slab(stride_ * kExperts, 0);
    for (std::int64_t e = 0; e < kExperts; ++e) {
      std::size_t at = static_cast<std::size_t>(e) * stride_;
      for (const Weights* w : {&gate_[U(e)], &up_[U(e)], &down_[U(e)]}) {
        const auto bytes = ToGgml(*w);
        std::ranges::copy(bytes, slab.begin() + static_cast<std::ptrdiff_t>(at));
        at += slice;
      }
    }
    ggml_slab_ = slab;
    slab_ = Allocate(slab.size());
    ASSERT_EQ(cudaMemcpy(Pointer(slab_), slab.data(), slab.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    slice_ = slice;
  }

  void TearDown() override {
    Finish();
    launch_.reset();
    ASSERT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : device_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
  }

  static void* Pointer(std::uint64_t address) {
    return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
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
    EXPECT_EQ(cudaMemset(pointer, 0, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    // The memset runs on the legacy stream, which the provider's
    // non-blocking stream does not wait for: it must land before any kernel
    // writes the memory.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  ggml_context* c() const { return arena_->context(); }
  LaunchContext& launch() { return *launch_; }
  cudaStream_t Stream() {
    return static_cast<cudaStream_t>(execution_->Submission(stream_).value().handle);
  }

  template <typename T>
  ggml_tensor* Leaf(ggml_tensor* tensor, const std::vector<T>& data) {
    TensorArena::Bind(tensor, Allocate(ggml_nbytes(tensor)));
    EXPECT_EQ(data.size() * sizeof(T), ggml_nbytes(tensor));
    EXPECT_EQ(cudaMemcpy(tensor->data, data.data(), ggml_nbytes(tensor), cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);  // as Allocate's memset
    return tensor;
  }

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
    auto ran = bound->Run(launch());
    ASSERT_TRUE(ran.has_value()) << ran.error().detail;
    ASSERT_FALSE(launch().faulted());
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor, std::size_t bytes = 0) {
    Finish();
    std::vector<T> values((bytes != 0 ? bytes : ggml_nbytes(tensor)) / sizeof(T));
    EXPECT_EQ(
        cudaMemcpy(values.data(), tensor->data, values.size() * sizeof(T), cudaMemcpyDeviceToHost),
        cudaSuccess);
    return values;
  }

  moe::ExpertSlab Slab() const {
    return {.base = Pointer(slab_),
            .stride = stride_,
            .experts = kExperts,
            .gate = 0,
            .up = slice_,
            .down = 2 * slice_,
            .layout = {.ffn = kFfn, .width = kWidth}};
  }

  // Converts the slab to the CUTLASS layout in place.
  void ToCutlass() {
    const std::uint64_t temp = Allocate(stride_ * 5);
    ASSERT_TRUE(moe::ToCutlassLayout(Slab(), Pointer(temp), 5, Stream()));
    Finish();
  }

  // Routes: each token's 10 distinct experts of 16, and softmax-like weights.
  static std::vector<std::int32_t> Routes(std::int64_t t, std::int64_t seed) {
    std::mt19937 random(static_cast<unsigned>(seed));
    std::vector<std::int32_t> ids;
    for (std::int64_t r = 0; r < t; ++r) {
      std::vector<std::int32_t> all(kExperts);
      std::ranges::iota(all, 0);
      std::ranges::shuffle(all, random);
      ids.insert(ids.end(), all.begin(), all.begin() + kUsed);
    }
    return ids;
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
  std::unique_ptr<llmp::execution::Registry> registry_;
  std::vector<Weights> gate_, up_, down_;
  std::vector<std::uint8_t> ggml_slab_;
  std::uint64_t slab_ = 0;
  std::uint64_t stride_ = 0;
  std::uint64_t slice_ = 0;
};

TEST_F(Qwen38MoeTest, TheLayoutConversionIsLossless) {
  ToCutlass();
  // The CUTLASS layout holds each weight where moe_layout.h says.
  std::vector<std::uint8_t> cutlass(stride_ * kExperts);
  ASSERT_EQ(cudaMemcpy(cutlass.data(), Pointer(slab_), cutlass.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
  for (std::int64_t e = 0; e < kExperts; e += 5) {
    const std::uint8_t* s = cutlass.data() + (static_cast<std::size_t>(e) * stride_);
    for (std::int64_t r = 0; r < 2 * kFfn; r += 37) {
      const Weights& w = r < kFfn ? gate_[U(e)] : up_[U(e)];
      const std::int64_t wr = r % kFfn;
      for (std::int64_t i = 0; i < kWidth; i += 13) {
        EXPECT_EQ(s[moe::ExpertLayout::gate_up_codes() +
                    static_cast<std::uint64_t>((r * kWidth / 2) + (i / 2))],
                  w.codes[static_cast<std::size_t>((wr * kWidth / 2) + (i / 2))]);
        EXPECT_EQ(
            s[l.gate_up_scales() + moe::SfOffset(static_cast<std::uint64_t>(r),
                                                 static_cast<std::uint64_t>(i / 16), kWidth / 16)],
            w.scales[static_cast<std::size_t>((wr * kWidth / 16) + (i / 16))]);
      }
    }
    for (std::int64_t r = 0; r < kWidth; r += 101) {
      for (std::int64_t i = 0; i < kFfn; i += 7) {
        EXPECT_EQ(s[l.down_codes() + static_cast<std::uint64_t>((r * kFfn / 2) + (i / 2))],
                  down_[U(e)].codes[static_cast<std::size_t>((r * kFfn / 2) + (i / 2))]);
      }
    }
  }
  // And back: GGML's bytes exactly.
  const std::uint64_t back = Allocate(stride_ * kExperts);
  ASSERT_TRUE(moe::ToGgmlLayout(Slab(), Pointer(back), Stream()));
  Finish();
  std::vector<std::uint8_t> ggml(stride_ * kExperts);
  ASSERT_EQ(cudaMemcpy(ggml.data(), Pointer(back), ggml.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  std::size_t differing = 0;
  for (std::int64_t e = 0; e < kExperts; ++e) {
    for (std::size_t b = 0; b < 3 * slice_; ++b) {
      const std::size_t at = (static_cast<std::size_t>(e) * stride_) + b;
      differing += ggml[at] != ggml_slab_[at];
    }
  }
  EXPECT_EQ(differing, 0U);
  // A slab whose stride cannot hold the layout is refused.
  moe::ExpertSlab small = Slab();
  small.stride = l.bytes() - 16;
  EXPECT_FALSE(moe::ToCutlassLayout(small, Pointer(back), 1, Stream()));
}

// The prefill path against GGML's MMQ path, the grouped GEMM against the
// FP64 product of its quantized inputs, and the whole against FP64.
TEST_F(Qwen38MoeTest, ThePrefillPathMatchesMmqAndTheReference) {
  if (!moe::GroupedGemmAvailable()) {
    GTEST_SKIP() << "no sm_121a grouped GEMM in this build";
  }
  for (const std::int64_t t : {1, 4, 9, 37, 300}) {
    const std::string what = "grouped t " + std::to_string(t);
    const auto x_h = Normal(7 + t, static_cast<std::size_t>(kWidth * t), 1.0f);
    const auto routes = Routes(t, 11 + t);
    std::vector<float> w_h(static_cast<std::size_t>(kUsed * t));
    for (std::size_t i = 0; i < w_h.size(); ++i) {
      w_h[i] = 0.05f + (0.01f * static_cast<float>(i % 7));
    }
    const auto sh_h = Normal(13, static_cast<std::size_t>(kWidth * t), 0.2f);
    const auto sg_h = Normal(14, static_cast<std::size_t>(t), 1.0f);
    std::vector<float> gs_h(kExperts);
    std::vector<float> us_h(kExperts);
    std::vector<float> ds_h(kExperts);
    for (std::int64_t e = 0; e < kExperts; ++e) {
      gs_h[static_cast<std::size_t>(e)] = 0.9f + (0.01f * static_cast<float>(e));
      us_h[static_cast<std::size_t>(e)] = 1.1f - (0.01f * static_cast<float>(e));
      ds_h[static_cast<std::size_t>(e)] = 0.7f + (0.02f * static_cast<float>(e));
    }
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), x_h);
    ggml_tensor* ids = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, t), routes);
    ggml_tensor* weights = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, kUsed, t), w_h);
    ggml_tensor* sh = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), sh_h);
    ggml_tensor* sg = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1, t), sg_h);
    ggml_tensor* gs = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), gs_h);
    ggml_tensor* us = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), us_h);
    ggml_tensor* ds = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), ds_h);

    // GGML's path over the GGML layout: the model's unfused-scale products
    // with the fused SwiGLU and sum (qwen38_graph.cc).
    std::vector<ggml_tensor*> mmq_nodes;
    ggml_tensor* mmq_out = nullptr;
    {
      const auto slice3 = [&](std::uint64_t offset, std::int64_t k, std::int64_t n) {
        ggml_tensor* w3 = ggml_new_tensor_3d(c(), GGML_TYPE_NVFP4, k, n, kExperts);
        w3->nb[2] = stride_;
        w3->nb[3] = stride_ * kExperts;
        TensorArena::Bind(w3, slab_ + offset);
        kg::MarkRowPaddingReadable(w3);
        return w3;
      };
      ggml_tensor* x3 = ggml_reshape_3d(c(), x, kWidth, 1, t);
      ggml_tensor* g = ggml_mul_mat_id(c(), slice3(0, kWidth, kFfn), x3, ids);
      ggml_tensor* u = ggml_mul_mat_id(c(), slice3(slice_, kWidth, kFfn), x3, ids);
      ggml_tensor* act = kg::MoeGlu(c(), g, u, ids, gs, us);
      ggml_tensor* d = ggml_mul_mat_id(c(), slice3(2 * slice_, kFfn, kWidth), act, ids);
      mmq_out = kg::MoeCombine(c(), d, ids, ds, weights, sh, sg);
    }
    Run({mmq_out});
    const std::vector<float> mmq = Download(mmq_out);

    // The CUTLASS path over the converted slab.
    ToCutlass();
    ggml_tensor* experts =
        ggml_new_tensor_2d(c(), GGML_TYPE_I8, static_cast<std::int64_t>(stride_), kExperts);
    TensorArena::Bind(experts, slab_);
    const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
    ggml_tensor* route = kg::MoeRoute(c(), ids, kExperts);
    ggml_tensor* a1 = kg::MoeQuantize(c(), x, route);
    ggml_tensor* d1 = kg::MoeGemm(c(), a1, route, experts, 2 * kFfn,
                                  moe::ExpertLayout::gate_up_codes(), l.gate_up_scales());
    ggml_tensor* a2 = kg::MoeGluQuantize(c(), d1, a1, route, gs, us);
    ggml_tensor* d2 = kg::MoeGemm(c(), a2, route, experts, kWidth, l.down_codes(), l.down_scales());
    ggml_tensor* out = kg::MoeCombineSorted(c(), d2, a2, route, ds, weights, sh, sg);
    Run({out});
    const std::vector<float> got = Download(out);

    // The routing: every slot's row holds its token and expert, rows in
    // token order within an expert.
    const moe::RouteLayout rl{.experts = kExperts,
                              .slots = kUsed * t,
                              .chunks = (t + moe::kRouteChunk - 1) / moe::kRouteChunk};
    const auto ri = Download<std::int32_t>(route);
    const std::int32_t* offsets = ri.data() + moe::RouteLayout::offsets();
    const std::int32_t* position = ri.data() + rl.position();
    const std::int32_t* token = ri.data() + rl.token();
    const std::int32_t* expert = ri.data() + rl.expert();
    EXPECT_EQ(offsets[kExperts], kUsed * t);
    for (std::int64_t s = 0; s < kUsed * t; ++s) {
      const std::int32_t p = position[s];
      ASSERT_GE(p, 0);
      EXPECT_EQ(token[p], s / kUsed);
      EXPECT_EQ(expert[p], routes[static_cast<std::size_t>(s)]);
      EXPECT_GE(p, offsets[expert[p]]);
      EXPECT_LT(p, offsets[expert[p] + 1]);
      if (p > offsets[expert[p]]) {
        EXPECT_LT(token[p - 1], token[p]);
      }
    }

    // The gate/up GEMM against the FP64 product of its quantized inputs.
    const moe::QuantLayout q1{
        .k = kWidth, .slots = static_cast<std::uint64_t>(kUsed * t), .experts = kExperts};
    const auto a1_bytes = Download<std::uint8_t>(a1);
    const auto d1_bits = Download<std::uint16_t>(d1);
    const std::int32_t* scale_rows = ri.data() + rl.scale_rows();
    double worst = 0.0;
    for (std::int64_t r = 0; r < kUsed * t; r += 37) {
      const std::int32_t e = expert[r];
      std::vector<double> xq(kWidth);
      const auto srow = static_cast<std::uint64_t>(scale_rows[e] + (r - offsets[e]));
      for (std::int64_t i = 0; i < kWidth; ++i) {
        const std::uint8_t byte = a1_bytes[moe::QuantLayout::codes() +
                                           static_cast<std::uint64_t>((r * kWidth / 2) + (i / 2))];
        const auto code = static_cast<std::uint8_t>(i % 2 == 0 ? byte & 15 : byte >> 4);
        xq[static_cast<std::size_t>(i)] =
            E2m1(code) *
            E4m3(a1_bytes[q1.scales() +
                          moe::SfOffset(srow, static_cast<std::uint64_t>(i / 16), kWidth / 16)]);
      }
      for (std::int64_t n = 0; n < 2 * kFfn; n += 11) {
        const Weights& w = n < kFfn ? gate_[U(e)] : up_[U(e)];
        double sum = 0.0;
        double magnitude = 0.0;
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const double term = xq[static_cast<std::size_t>(i)] * w.At(n % kFfn, i);
          sum += term;
          magnitude += std::abs(term);
        }
        const double gotv = static_cast<double>(std::bit_cast<float>(
            std::uint32_t{d1_bits[static_cast<std::size_t>((r * 2 * kFfn) + n)]} << 16U));
        const double error = std::abs(gotv - sum) / ((std::abs(sum) * 0x1p-8) + (magnitude * 1e-6));
        worst = std::max(worst, error);
      }
    }
    EXPECT_LE(worst, 1.0) << what << ": the GEMM beyond BF16 rounding of its exact product";

    // Against GGML's product (every token) and the FP64 reference (up to
    // four tokens). At <=8 rows GGML uses Q8 MMVQ rather than FP4 MMQ, so
    // only the wider shapes share the MMQ comparison bounds below.
    const auto checked = std::min<std::int64_t>(4, t);
    std::vector<double> ref(static_cast<std::size_t>(kWidth * checked));
    for (std::int64_t r = 0; r < checked; ++r) {
      std::vector<double> total(kWidth, 0.0);
      for (std::int64_t j = 0; j < kUsed; ++j) {
        const auto e = static_cast<std::size_t>(routes[static_cast<std::size_t>((r * kUsed) + j)]);
        std::vector<double> act(kFfn);
        for (std::int64_t n = 0; n < kFfn; ++n) {
          double g = 0.0;
          double u = 0.0;
          for (std::int64_t i = 0; i < kWidth; ++i) {
            const double xi = x_h[static_cast<std::size_t>((r * kWidth) + i)];
            g += gate_[e].At(n, i) * xi;
            u += up_[e].At(n, i) * xi;
          }
          g *= gs_h[e];
          u *= us_h[e];
          act[static_cast<std::size_t>(n)] = g / (1.0 + std::exp(-g)) * u;
        }
        for (std::int64_t n = 0; n < kWidth; ++n) {
          double d = 0.0;
          for (std::int64_t i = 0; i < kFfn; ++i) {
            d += down_[e].At(n, i) * act[static_cast<std::size_t>(i)];
          }
          total[static_cast<std::size_t>(n)] +=
              d * ds_h[e] * w_h[static_cast<std::size_t>((r * kUsed) + j)];
        }
      }
      const double gate =
          1.0 / (1.0 + std::exp(-static_cast<double>(sg_h[static_cast<std::size_t>(r)])));
      for (std::int64_t n = 0; n < kWidth; ++n) {
        const auto at = static_cast<std::size_t>((r * kWidth) + n);
        ref[at] = total[static_cast<std::size_t>(n)] + (sh_h[at] * gate);
      }
    }
    std::vector<double> mmq_d(mmq.begin(), mmq.end());
    const double against_mmq = Nmse(got, mmq_d);
    const std::vector<float> got_first(got.begin(), got.begin() + (kWidth * checked));
    const std::vector<float> mmq_first(mmq.begin(), mmq.begin() + (kWidth * checked));
    const double against_ref = Nmse(got_first, ref);
    const double mmq_against_ref = Nmse(mmq_first, ref);
    std::cout << what << ": NMSE " << against_mmq << " against MMQ, " << against_ref
              << " against FP64 (MMQ's " << mmq_against_ref << ")\n";
    if (t > 8) {
      EXPECT_LE(against_mmq, kAgainstMmq) << what;
      EXPECT_LE(against_ref, 1.05 * mmq_against_ref) << what;
    }
    EXPECT_LE(against_ref, 3e-2) << what;
    EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); })) << what;
    // Back to GGML's layout for the next shape.
    const std::uint64_t back = Allocate(stride_ * kExperts);
    ASSERT_TRUE(moe::ToGgmlLayout(Slab(), Pointer(back), Stream()));
    Finish();
    ASSERT_EQ(
        cudaMemcpy(Pointer(slab_), Pointer(back), stride_ * kExperts, cudaMemcpyDeviceToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }
}

// Decode's vector products against the FP64 product, and a NaN for an id
// outside the experts.
TEST_F(Qwen38MoeTest, TheVectorProductsMatchTheReference) {
  ToCutlass();
  ggml_tensor* experts =
      ggml_new_tensor_2d(c(), GGML_TYPE_I8, static_cast<std::int64_t>(stride_), kExperts);
  TensorArena::Bind(experts, slab_);
  const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
  for (const std::int64_t t : {1, 3, 8}) {
    const std::string what = "gemv t " + std::to_string(t);
    const auto x_h = Normal(21 + t, static_cast<std::size_t>(kWidth * t), 1.0f);
    const auto act_h = Normal(22 + t, static_cast<std::size_t>(kFfn * kUsed * t), 1.0f);
    auto routes = Routes(t, 23 + t);
    routes[1] = kExperts;  // outside
    ggml_tensor* x = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, t), x_h);
    ggml_tensor* act = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, t), act_h);
    ggml_tensor* ids = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, t), routes);
    std::vector<float> gs_h(kExperts);
    std::vector<float> us_h(kExperts);
    for (std::int64_t e = 0; e < kExperts; ++e) {
      gs_h[static_cast<std::size_t>(e)] = 0.9f + (0.01f * static_cast<float>(e));
      us_h[static_cast<std::size_t>(e)] = 1.1f - (0.01f * static_cast<float>(e));
    }
    ggml_tensor* gs = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), gs_h);
    ggml_tensor* us = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), us_h);
    ggml_tensor* gate = kg::MoeGemv(c(), experts, x, ids, kFfn, 0, 2 * kFfn,
                                    moe::ExpertLayout::gate_up_codes(), l.gate_up_scales());
    ggml_tensor* up = kg::MoeGemv(c(), experts, x, ids, kFfn, kFfn, 2 * kFfn,
                                  moe::ExpertLayout::gate_up_codes(), l.gate_up_scales());
    ggml_tensor* glu = kg::MoeGlu(c(), gate, up, ids, gs, us);
    ggml_tensor* swiglu = kg::MoeGemvSwiglu(c(), experts, x, ids, kFfn, gs, us,
                                            moe::ExpertLayout::gate_up_codes(), l.gate_up_scales());
    ggml_tensor* down =
        kg::MoeGemv(c(), experts, act, ids, kWidth, 0, kWidth, l.down_codes(), l.down_scales());
    Run({glu, swiglu, down});
    const auto got_up = Download(up);
    const auto got_down = Download(down);
    // The SwiGLU form is the two products and llmp.moe.glu, bit for bit
    // (NaN where the id is outside).
    const auto got_glu = Download(glu);
    const auto got_swiglu = Download(swiglu);
    ASSERT_EQ(got_glu.size(), got_swiglu.size());
    for (std::size_t i = 0; i < got_glu.size(); ++i) {
      ASSERT_TRUE(std::bit_cast<std::uint32_t>(got_glu[i]) ==
                      std::bit_cast<std::uint32_t>(got_swiglu[i]) ||
                  (std::isnan(got_glu[i]) && std::isnan(got_swiglu[i])))
          << what << " at " << i;
    }
    std::vector<float> finite_up;
    std::vector<double> want_up;
    std::vector<float> finite_down;
    std::vector<double> want_down;
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t j = 0; j < kUsed; ++j) {
        const std::int32_t e = routes[static_cast<std::size_t>((r * kUsed) + j)];
        const auto slot = static_cast<std::size_t>((r * kUsed) + j);
        if (e >= kExperts) {
          EXPECT_TRUE(std::isnan(got_up[slot * kFfn])) << what;
          EXPECT_TRUE(std::isnan(got_down[slot * kWidth])) << what;
          continue;
        }
        for (std::int64_t n = 0; n < kFfn; ++n) {
          double sum = 0.0;
          for (std::int64_t i = 0; i < kWidth; ++i) {
            sum += up_[U(e)].At(n, i) * x_h[static_cast<std::size_t>((r * kWidth) + i)];
          }
          finite_up.push_back(got_up[(slot * kFfn) + static_cast<std::size_t>(n)]);
          want_up.push_back(sum);
        }
        for (std::int64_t n = 0; n < kWidth; n += 3) {
          double sum = 0.0;
          for (std::int64_t i = 0; i < kFfn; ++i) {
            sum += down_[U(e)].At(n, i) * act_h[(slot * kFfn) + static_cast<std::size_t>(i)];
          }
          finite_down.push_back(got_down[(slot * kWidth) + static_cast<std::size_t>(n)]);
          want_down.push_back(sum);
        }
      }
    }
    EXPECT_LE(Nmse(finite_up, want_up), kGemvNmse) << what;
    EXPECT_LE(Nmse(finite_down, want_down), kGemvNmse) << what;
    std::cout << what << ": NMSE " << Nmse(finite_up, want_up) << " (up), "
              << Nmse(finite_down, want_down) << " (down) against FP64\n";
  }
}

// Several tokens' vector products in one launch (a speculative verify's
// rows, docs/experiments/qwen38-mtp/): each slot's outputs are the
// one-token launch's bit for bit (NaN where its id is outside the experts).
TEST_F(Qwen38MoeTest, SeveralTokensGetEachTokensOwnProducts) {
  ToCutlass();
  ggml_tensor* experts =
      ggml_new_tensor_2d(c(), GGML_TYPE_I8, static_cast<std::int64_t>(stride_), kExperts);
  TensorArena::Bind(experts, slab_);
  const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
  std::vector<float> gs_h(kExperts);
  std::vector<float> us_h(kExperts);
  for (std::int64_t e = 0; e < kExperts; ++e) {
    gs_h[static_cast<std::size_t>(e)] = 0.9f + (0.01f * static_cast<float>(e));
    us_h[static_cast<std::size_t>(e)] = 1.1f - (0.01f * static_cast<float>(e));
  }
  ggml_tensor* gs = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), gs_h);
  ggml_tensor* us = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), us_h);
  const auto same = [](float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b) ||
           (std::isnan(a) && std::isnan(b));
  };
  for (const std::int64_t t : {2, 4, 8}) {
    const std::string what = "gemv t " + std::to_string(t);
    const auto x_h = Normal(41 + t, static_cast<std::size_t>(kWidth * t), 1.0f);
    const auto act_h = Normal(42 + t, static_cast<std::size_t>(kFfn * kUsed * t), 1.0f);
    auto routes = Routes(t, 43 + t);
    routes[static_cast<std::size_t>(kUsed + 2)] = kExperts;  // outside, in the second token
    ggml_tensor* x = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, t), x_h);
    ggml_tensor* act = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, t), act_h);
    ggml_tensor* ids = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, t), routes);
    std::vector<ggml_tensor*> outputs = {
        kg::MoeGemvSwiglu(c(), experts, x, ids, kFfn, gs, us, moe::ExpertLayout::gate_up_codes(),
                          l.gate_up_scales()),
        kg::MoeGemv(c(), experts, act, ids, kWidth, 0, kWidth, l.down_codes(), l.down_scales())};
    for (std::int64_t r = 0; r < t; ++r) {
      const auto at = [&](const std::vector<float>& v, std::int64_t per) {
        return std::vector<float>(v.begin() + (r * per), v.begin() + ((r + 1) * per));
      };
      ggml_tensor* x1 = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, 1), at(x_h, kWidth));
      ggml_tensor* act1 =
          Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, 1), at(act_h, kFfn * kUsed));
      const std::vector<std::int32_t> ids1_h(routes.begin() + (r * kUsed),
                                             routes.begin() + ((r + 1) * kUsed));
      ggml_tensor* ids1 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, 1), ids1_h);
      outputs.push_back(kg::MoeGemvSwiglu(c(), experts, x1, ids1, kFfn, gs, us,
                                          moe::ExpertLayout::gate_up_codes(), l.gate_up_scales()));
      outputs.push_back(kg::MoeGemv(c(), experts, act1, ids1, kWidth, 0, kWidth, l.down_codes(),
                                    l.down_scales()));
    }
    Run(outputs);
    const auto whole_glu = Download(outputs[0]);
    const auto whole_down = Download(outputs[1]);
    for (std::int64_t r = 0; r < t; ++r) {
      const auto glu = Download(outputs[static_cast<std::size_t>(2 + (2 * r))]);
      const auto down = Download(outputs[static_cast<std::size_t>(3 + (2 * r))]);
      const auto glu_at = static_cast<std::size_t>(r * kUsed * kFfn);
      const auto down_at = static_cast<std::size_t>(r * kUsed * kWidth);
      for (std::size_t i = 0; i < glu.size(); ++i) {
        ASSERT_TRUE(same(whole_glu[glu_at + i], glu[i])) << what << ", token " << r << " at " << i;
      }
      for (std::size_t i = 0; i < down.size(); ++i) {
        ASSERT_TRUE(same(whole_down[down_at + i], down[i]))
            << what << ", token " << r << " at " << i;
      }
    }
    EXPECT_TRUE(std::isnan(whole_glu[static_cast<std::size_t>((kUsed + 2) * kFfn)])) << what;
  }
}

// A Qwen3.8 wave's routed products join up to four requests' verify rows
// (engine/qwen38_wave_plan.h): past four tokens the expert-major kernel
// reads each chosen expert's rows once for every slot that chose it. Each
// slot's outputs are the per-slot kernel's (at most four tokens a launch)
// bit for bit: the SwiGLU form; the plain form over part of the up rows,
// from an unaligned first row, at an output count that leaves the last
// tile short; and the down projection over each slot's own input at
// another such count. The routes repeat an expert within each token (more
// than eight slots on it: several leading blocks, across the scan's
// 32-slot steps) and hold ids outside the experts (NaN, each slot alone).
TEST_F(Qwen38MoeTest, ExpertMajorProductsMatchTheirPerSlotProducts) {
  ToCutlass();
  ggml_tensor* experts =
      ggml_new_tensor_2d(c(), GGML_TYPE_I8, static_cast<std::int64_t>(stride_), kExperts);
  TensorArena::Bind(experts, slab_);
  const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
  std::vector<float> gs_h(kExperts);
  std::vector<float> us_h(kExperts);
  for (std::int64_t e = 0; e < kExperts; ++e) {
    gs_h[static_cast<std::size_t>(e)] = 0.9f + (0.01f * static_cast<float>(e));
    us_h[static_cast<std::size_t>(e)] = 1.1f - (0.01f * static_cast<float>(e));
  }
  ggml_tensor* gs = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), gs_h);
  ggml_tensor* us = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), us_h);
  const auto same = [](float a, float b) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b) ||
           (std::isnan(a) && std::isnan(b));
  };
  constexpr std::int64_t kUpRow0 = kFfn + 24;
  constexpr std::int64_t kUpRows = kFfn - 40;      // 600: the last 64-row tile is short
  constexpr std::int64_t kDownRows = kWidth - 60;  // 2,500: likewise
  constexpr std::int64_t kChunk = 4;               // the per-slot kernel's tokens
  const auto build = [&](ggml_tensor* x, ggml_tensor* act, ggml_tensor* ids) {
    return std::array<ggml_tensor*, 3>{
        kg::MoeGemvSwiglu(c(), experts, x, ids, kFfn, gs, us, moe::ExpertLayout::gate_up_codes(),
                          l.gate_up_scales()),
        kg::MoeGemv(c(), experts, x, ids, kUpRows, kUpRow0, 2 * kFfn,
                    moe::ExpertLayout::gate_up_codes(), l.gate_up_scales()),
        kg::MoeGemv(c(), experts, act, ids, kDownRows, 0, kWidth, l.down_codes(), l.down_scales())};
  };
  for (const std::int64_t t : {5, 8, 12, 16}) {
    const std::string what = "tokens " + std::to_string(t);
    const auto x_h = Normal(61 + t, static_cast<std::size_t>(kWidth * t), 1.0f);
    const auto act_h = Normal(62 + t, static_cast<std::size_t>(kFfn * kUsed * t), 1.0f);
    auto routes = Routes(t, 63 + t);
    for (std::int64_t r = 0; r < t; ++r) {
      // Expert 5 twice in every token: at least 2t slots on it.
      routes[static_cast<std::size_t>(r * kUsed)] = 5;
      routes[static_cast<std::size_t>((r * kUsed) + 6)] = 5;
    }
    const std::vector<std::size_t> outside = {static_cast<std::size_t>(kUsed + 3),
                                              static_cast<std::size_t>((2 * kUsed) + 4),
                                              static_cast<std::size_t>((t * kUsed) - 1)};
    routes[outside[0]] = kExperts;
    routes[outside[1]] = -1;
    routes[outside[2]] = kExperts;
    ggml_tensor* x = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, t), x_h);
    ggml_tensor* act = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, t), act_h);
    ggml_tensor* ids = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, t), routes);
    const auto whole = build(x, act, ids);
    std::vector<ggml_tensor*> outputs(whole.begin(), whole.end());
    for (std::int64_t first = 0; first < t; first += kChunk) {
      const std::int64_t n = std::min(kChunk, t - first);
      const auto slice = [&](const auto& v, std::int64_t per) {
        using T = std::decay_t<decltype(v)>::value_type;
        return std::vector<T>(v.begin() + (first * per), v.begin() + ((first + n) * per));
      };
      const auto part = build(
          Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, n), slice(x_h, kWidth)),
          Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, n), slice(act_h, kFfn * kUsed)),
          Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, n), slice(routes, kUsed)));
      outputs.insert(outputs.end(), part.begin(), part.end());
    }
    Run(outputs);
    for (std::size_t form = 0; form < whole.size(); ++form) {
      const auto got = Download(whole[form]);
      const auto per_slot = static_cast<std::size_t>(whole[form]->ne[0]);
      std::vector<float> want;
      for (std::size_t i = whole.size() + form; i < outputs.size(); i += whole.size()) {
        const auto part = Download(outputs[i]);
        want.insert(want.end(), part.begin(), part.end());
      }
      ASSERT_EQ(got.size(), want.size()) << what;
      ASSERT_EQ(got.size(), per_slot * static_cast<std::size_t>(kUsed * t)) << what;
      for (std::size_t i = 0; i < got.size(); ++i) {
        ASSERT_TRUE(same(got[i], want[i]))
            << what << ", form " << form << ", slot " << (i / per_slot) << ", output "
            << (i % per_slot) << ": " << got[i] << " vs " << want[i];
      }
      for (const std::size_t slot : outside) {
        EXPECT_TRUE(std::isnan(got[slot * per_slot])) << what << ", form " << form;
        EXPECT_TRUE(std::isnan(got[((slot + 1) * per_slot) - 1])) << what << ", form " << form;
      }
      EXPECT_TRUE(std::ranges::all_of(
          got.begin(), got.begin() + static_cast<std::ptrdiff_t>(outside[0] * per_slot),
          [](float v) { return std::isfinite(v); }))
          << what << ", form " << form;
    }
  }
}

// The checks refuse a slab stride short of the layout, rows that are not
// whole scale atoms, too many tokens for the vector product, and a route of
// other extents.
TEST_F(Qwen38MoeTest, TheChecksRefuseWhatTheKernelsCannotRun) {
  const moe::ExpertLayout l{.ffn = kFfn, .width = kWidth};
  const auto bytes = [&](std::int64_t stride) {
    ggml_tensor* e = ggml_new_tensor_2d(c(), GGML_TYPE_I8, stride, kExperts);
    TensorArena::Bind(e, slab_);
    return e;
  };
  ggml_tensor* x = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, 9),
                        std::vector<float>(static_cast<std::size_t>(kWidth * 9), 1.0f));
  ggml_tensor* ids = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, 9),
                          std::vector<std::int32_t>(static_cast<std::size_t>(kUsed * 9), 0));
  const auto place = [&](ggml_tensor* node) {
    TensorArena::Bind(node, Allocate(ggml_nbytes(node)));
    return node;
  };
  // Sixteen tokens for the vector product (a wave's shared product), and not
  // seventeen.
  constexpr std::int64_t kMost = kg::kMoeGemvWaveTokens;
  ggml_tensor* x17 = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 1, kMost + 1),
                          std::vector<float>(static_cast<std::size_t>(kWidth * (kMost + 1)), 1.0f));
  ggml_tensor* ids17 =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, kMost + 1),
           std::vector<std::int32_t>(static_cast<std::size_t>(kUsed * (kMost + 1)), 0));
  ggml_tensor* x16 = ggml_view_3d(c(), x17, kWidth, 1, kMost, x17->nb[1], x17->nb[2], 0);
  ggml_tensor* ids16 = ggml_view_2d(c(), ids17, kUsed, kMost, ids17->nb[1], 0);
  ggml_tensor* x1 = ggml_view_3d(c(), x, kWidth, 1, 1, x->nb[1], x->nb[2], 0);
  ggml_tensor* ids1 = ggml_view_2d(c(), ids, kUsed, 1, ids->nb[1], 0);
  const std::vector<ggml_tensor*> views = {x16, ids16, x1, ids1};
  kg::BindViews(views);
  EXPECT_FALSE(
      kg::CheckMoeGemv(
          place(kg::MoeGemv(c(), bytes(static_cast<std::int64_t>(stride_)), x17, ids17, kFfn, 0,
                            2 * kFfn, moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  EXPECT_TRUE(
      kg::CheckMoeGemv(
          place(kg::MoeGemv(c(), bytes(static_cast<std::int64_t>(stride_)), x16, ids16, kFfn, 0,
                            2 * kFfn, moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  // A stride short of the layout's rows.
  EXPECT_FALSE(
      kg::CheckMoeGemv(place(kg::MoeGemv(c(), bytes(static_cast<std::int64_t>(l.gate_up_scales())),
                                         x1, ids1, kFfn, 0, 2 * kFfn,
                                         moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  // Rows that are not whole 128-row scale atoms.
  EXPECT_FALSE(
      kg::CheckMoeGemv(place(kg::MoeGemv(c(), bytes(static_cast<std::int64_t>(stride_)), x1, ids1,
                                         kFfn, 0, (2 * kFfn) - 64,
                                         moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  // And the one that fits.
  EXPECT_TRUE(
      kg::CheckMoeGemv(
          place(kg::MoeGemv(c(), bytes(static_cast<std::int64_t>(stride_)), x1, ids1, kFfn, 0,
                            2 * kFfn, moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  // The SwiGLU form: scales for every expert, and not for fewer.
  ggml_tensor* scales = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts),
                             std::vector<float>(static_cast<std::size_t>(kExperts), 1.0f));
  ggml_tensor* short_scales = ggml_view_1d(c(), scales, kExperts - 1, 0);
  const std::vector<ggml_tensor*> scale_views = {short_scales};
  kg::BindViews(scale_views);
  EXPECT_TRUE(
      kg::CheckMoeGemv(place(kg::MoeGemvSwiglu(
                           c(), bytes(static_cast<std::int64_t>(stride_)), x1, ids1, kFfn, scales,
                           scales, moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  EXPECT_FALSE(
      kg::CheckMoeGemv(place(kg::MoeGemvSwiglu(
                           c(), bytes(static_cast<std::int64_t>(stride_)), x1, ids1, kFfn, scales,
                           short_scales, moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())))
          .has_value());
  // A quantization over another route's extents.
  ggml_tensor* x2 = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 9),
                         std::vector<float>(static_cast<std::size_t>(kWidth * 9), 1.0f));
  ggml_tensor* route = place(kg::MoeRoute(c(), ids, kExperts));
  ggml_tensor* short_x = ggml_view_2d(c(), x2, kWidth, 8, x2->nb[1], 0);
  const std::vector<ggml_tensor*> more = {short_x};
  kg::BindViews(more);
  EXPECT_TRUE(kg::CheckMoeQuantize(place(kg::MoeQuantize(c(), x2, route))).has_value());
  EXPECT_FALSE(kg::CheckMoeQuantize(place(kg::MoeQuantize(c(), short_x, route))).has_value());
  // A product over rows that another route of the same extents sorted.
  ggml_tensor* a = place(kg::MoeQuantize(c(), x2, route));
  ggml_tensor* other = place(kg::MoeRoute(c(), ids, kExperts));
  const auto gemm = [&](ggml_tensor* by) {
    return kg::CheckMoeGemm(
        place(kg::MoeGemm(c(), a, by, bytes(static_cast<std::int64_t>(stride_)), 2 * kFfn,
                          moe::ExpertLayout::gate_up_codes(), l.gate_up_scales())));
  };
  EXPECT_TRUE(gemm(route).has_value());
  EXPECT_FALSE(gemm(other).has_value());
}

}  // namespace
