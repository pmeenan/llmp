// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's fusions of Qwen3.8's hyper-connection and MoE-output nodes
// (kernels/ggml/jitllm_ops.h) on a GB10 (label `gpu`), at the model's
// widths: each result equals, bit for bit, that of the GGML nodes it
// replaces (built as qwen38_graph.cc builds them unfused, planned and run
// through the registry), and is within the default float bound (NMSE 1e-7)
// of an FP64 reference; the BF16 product equals GGML's cuBLAS product over
// the same F32 activations, and the BF16 norm equals the F32 norm rounded to
// nearest. Also: an expert id outside the scales gives NaN, and the checks
// refuse what the kernels cannot run.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::CublasHandle;
using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::TensorArena;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;
using jitllm::test_support::FailedCode;
namespace kg = jitllm::kernels::ggml;

constexpr double kDefaultNmse = 1e-7;
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

void ExpectNear(const std::vector<float>& got, const std::vector<double>& want,
                const std::string& what) {
  const double nmse = Nmse(got, want);
  EXPECT_LE(nmse, kDefaultNmse) << what;
  EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); })) << what;
  std::cout << what << ": NMSE " << nmse << " against FP64\n";
}

// Bit for bit.
void ExpectSame(const std::vector<float>& got, const std::vector<float>& want,
                const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  std::size_t differing = 0;
  std::size_t first = got.size();
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i])) {
      ++differing;
      first = std::min(first, i);
    }
  }
  EXPECT_EQ(differing, 0U) << what << ": first at " << first << ", "
                           << (first < got.size() ? got[first] : 0.0f) << " against "
                           << (first < got.size() ? want[first] : 0.0f);
}

std::vector<float> Normal(std::uint64_t seed, std::size_t n, float scale = 1.0f,
                          float mean = 0.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(mean, scale);
  std::vector<float> out(n);
  for (float& v : out) {
    v = normal(random);
  }
  return out;
}

double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Round to nearest even, as __float2bfloat16 does (no NaNs here).
std::uint16_t Bf16Bits(float v) {
  const auto bits = std::bit_cast<std::uint32_t>(v);
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
}

float FromBf16(std::uint16_t b) { return std::bit_cast<float>(std::uint32_t{b} << 16U); }

class Qwen38FusedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
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
    auto registry = jitllm::execution::Registry::Create(kg::Implementations());
    ASSERT_TRUE(registry.has_value());
    registry_ = std::make_unique<jitllm::execution::Registry>(std::move(*registry));
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
    device_.push_back(pointer);
    return reinterpret_cast<std::uintptr_t>(pointer);
  }

  ggml_context* c() const { return arena_->context(); }
  LaunchContext& launch() { return *launch_; }

  // Binds a leaf to new device memory holding `data`.
  template <typename T = float>
  ggml_tensor* Leaf(ggml_tensor* tensor, const std::vector<T>& data) {
    const std::uint64_t address = Allocate(ggml_nbytes(tensor));
    TensorArena::Bind(tensor, address);
    EXPECT_EQ(data.size() * sizeof(T), ggml_nbytes(tensor));
    EXPECT_EQ(cudaMemcpy(tensor->data, data.data(), ggml_nbytes(tensor), cudaMemcpyHostToDevice),
              cudaSuccess);
    // A pageable copy may return before its DMA lands, and the provider's
    // non-blocking stream does not wait for the legacy stream.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return tensor;
  }

  // Places every computed tensor the outputs need, binds the views, and
  // runs the plan the registry binds (fusion off), as the model does.
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

  // The implementation the plan names for a lone node.
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

  // Scales for 512 experts, of the size a checkpoint's global scales have.
  static std::vector<float> ExpertScales(std::uint64_t seed) {
    std::vector<float> s = Normal(seed, 512, 0.2f, 1.0f);
    for (float& v : s) {
      v = std::abs(v) * 3e-4f;
    }
    return s;
  }

  // Expert ids [used, t] as the top-k view of a wider sort, as the model's.
  ggml_tensor* Ids(std::int64_t used, std::int64_t t, std::uint64_t seed) {
    constexpr std::int64_t kRow = 16;
    std::mt19937 random(static_cast<unsigned>(seed));
    std::vector<std::int32_t> ids(static_cast<std::size_t>(kRow * t));
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t j = 0; j < kRow; ++j) {
        ids[static_cast<std::size_t>((r * kRow) + j)] = static_cast<std::int32_t>(random() % 512);
      }
    }
    ggml_tensor* all = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kRow, t), ids);
    ids_host_ = ids;
    return ggml_view_2d(c(), all, used, t, all->nb[1], 0);
  }
  std::int32_t Id(std::int64_t r, std::int64_t j) const {
    return ids_host_[static_cast<std::size_t>((r * 16) + j)];
  }

  // The unfused per-expert scale of build_lora_mm_id: [1, used, t].
  ggml_tensor* ScaleRows(ggml_tensor* scale, ggml_tensor* ids) {
    const std::int64_t n = scale->ne[0];
    ggml_tensor* s = ggml_reshape_3d(c(), scale, 1, n, 1);
    s = ggml_repeat_4d(c(), s, 1, n, ids->ne[1], 1);
    return ggml_get_rows(c(), s, ids);
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<CublasHandle> cublas_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
  std::unique_ptr<jitllm::execution::Registry> registry_;
  std::vector<std::int32_t> ids_host_;
};

TEST_F(Qwen38FusedTest, CombineIsTheUnfusedNodes) {
  for (const std::int64_t t : {1, 7, 33}) {
    const auto res_h = Normal(1, static_cast<std::size_t>(kWidth * kHc * t));
    const auto out_h = Normal(2, static_cast<std::size_t>(kWidth * t), 0.5f);
    const auto inj_h = Normal(3, static_cast<std::size_t>(kHc * t), 3.0f);
    ggml_tensor* res = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kHc, t), res_h);
    ggml_tensor* out = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), out_h);
    ggml_tensor* inj = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kHc, t), inj_h);
    // build_hc_combine, unfused.
    ggml_tensor* w = ggml_sigmoid(c(), ggml_scale(c(), inj, 1.0f / static_cast<float>(kHc)));
    w = ggml_reshape_3d(c(), ggml_scale(c(), w, 2.0f), 1, kHc, t);
    ggml_tensor* b =
        ggml_repeat_4d(c(), ggml_reshape_3d(c(), out, kWidth, 1, t), kWidth, kHc, t, 1);
    ggml_tensor* unfused = ggml_add(c(), res, ggml_mul(c(), b, w));
    ggml_tensor* fused = kg::HcCombine(c(), res, out, inj);
    Run({unfused, fused});
    EXPECT_EQ(PlannedFor(fused), kg::kHcCombineName);
    const auto got = Download(fused);
    ExpectSame(got, Download(unfused), "combine t " + std::to_string(t));
    std::vector<double> want(got.size());
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t k = 0; k < kHc; ++k) {
        const double g = 2.0 * Sigmoid(inj_h[static_cast<std::size_t>((r * kHc) + k)] / 4.0);
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const auto at = static_cast<std::size_t>((((r * kHc) + k) * kWidth) + i);
          want[at] = res_h[at] + (out_h[static_cast<std::size_t>((r * kWidth) + i)] * g);
        }
      }
    }
    ExpectNear(got, want, "combine t " + std::to_string(t));
  }
}

// FP64 rms_norm of each stream times the weight: [width · hc, t].
std::vector<double> NormReference(const std::vector<float>& x, const std::vector<float>& w,
                                  std::int64_t t) {
  std::vector<double> out(x.size());
  for (std::int64_t row = 0; row < t * kHc; ++row) {
    double sum = 0.0;
    for (std::int64_t i = 0; i < kWidth; ++i) {
      const double v = x[static_cast<std::size_t>((row * kWidth) + i)];
      sum += v * v;
    }
    const double scale = 1.0 / std::sqrt((sum / kWidth) + kEps);
    for (std::int64_t i = 0; i < kWidth; ++i) {
      const auto at = static_cast<std::size_t>((row * kWidth) + i);
      out[at] = x[at] * scale * w[static_cast<std::size_t>(((row % kHc) * kWidth) + i)];
    }
  }
  return out;
}

// Streams whose scales differ, as the residual's do.
std::vector<float> Streams(std::uint64_t seed, std::int64_t t) {
  std::vector<float> x = Normal(seed, static_cast<std::size_t>(kWidth * kHc * t));
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] *= static_cast<float>(1 + ((i / kWidth) % 7));
  }
  return x;
}

TEST_F(Qwen38FusedTest, NormIsTheUnfusedNodesInF32AndRoundedInBf16) {
  for (const std::int64_t t : {1, 7, 33}) {
    const auto x_h = Streams(4, t);
    const auto w_h = Normal(5, static_cast<std::size_t>(kWidth * kHc), 0.3f, 1.0f);
    ggml_tensor* x = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kHc, t), x_h);
    ggml_tensor* w = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kWidth * kHc), w_h);
    ggml_tensor* unfused =
        ggml_mul(c(), ggml_reshape_2d(c(), ggml_rms_norm(c(), x, kEps), kWidth * kHc, t), w);
    ggml_tensor* f32 = kg::HcNorm(c(), x, w, kEps, GGML_TYPE_F32);
    ggml_tensor* bf16 = kg::HcNorm(c(), x, w, kEps, GGML_TYPE_BF16);
    Run({unfused, f32, bf16});
    EXPECT_EQ(PlannedFor(f32), kg::kHcNormName);
    const auto want = Download(unfused);
    const auto got = Download(f32);
    ExpectSame(got, want, "norm t " + std::to_string(t));
    ExpectNear(got, NormReference(x_h, w_h, t), "norm t " + std::to_string(t));
    const auto rounded = Download<std::uint16_t>(bf16);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < rounded.size(); ++i) {
      differing += rounded[i] != Bf16Bits(want[i]);
    }
    EXPECT_EQ(differing, 0U) << "bf16 norm t " << t;
  }
}

TEST_F(Qwen38FusedTest, MixIsTheUnfusedNodes) {
  for (const std::int64_t t : {1, 7, 33}) {
    const auto x_h = Streams(6, t);
    const auto w_h = Normal(7, static_cast<std::size_t>(kWidth * kHc), 0.3f, 1.0f);
    const auto g_h = Normal(8, static_cast<std::size_t>(kWidth * kHc * t), 3.0f);
    ggml_tensor* x = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kHc, t), x_h);
    ggml_tensor* w = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kWidth * kHc), w_h);
    ggml_tensor* g = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth * kHc, t), g_h);
    // build_hc_mix's gate and fold, unfused.
    ggml_tensor* xn =
        ggml_mul(c(), ggml_reshape_2d(c(), ggml_rms_norm(c(), x, kEps), kWidth * kHc, t), w);
    ggml_tensor* gated =
        ggml_reshape_3d(c(), ggml_mul(c(), xn, ggml_sigmoid(c(), g)), kWidth, kHc, t);
    const std::size_t row = ggml_row_size(GGML_TYPE_F32, kWidth);
    ggml_tensor* unfused = ggml_cont(c(), ggml_view_2d(c(), gated, kWidth, t, row * kHc, 0));
    for (std::int64_t k = 1; k < kHc; ++k) {
      unfused = ggml_add(
          c(), unfused,
          ggml_view_2d(c(), gated, kWidth, t, row * kHc, row * static_cast<std::size_t>(k)));
    }
    unfused = ggml_scale(c(), unfused, 1.0f / static_cast<float>(kHc));
    ggml_tensor* fused = kg::HcMix(c(), x, w, g, kEps);
    Run({unfused, fused});
    EXPECT_EQ(PlannedFor(fused), kg::kHcMixName);
    const auto got = Download(fused);
    ExpectSame(got, Download(unfused), "mix t " + std::to_string(t));
    const std::vector<double> norm = NormReference(x_h, w_h, t);
    std::vector<double> want(got.size(), 0.0);
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t k = 0; k < kHc; ++k) {
        for (std::int64_t i = 0; i < kWidth; ++i) {
          const auto at = static_cast<std::size_t>((((r * kHc) + k) * kWidth) + i);
          want[static_cast<std::size_t>((r * kWidth) + i)] +=
              norm[at] * Sigmoid(g_h[at]) / static_cast<double>(kHc);
        }
      }
    }
    ExpectNear(got, want, "mix t " + std::to_string(t));
  }
}

TEST_F(Qwen38FusedTest, ExpertGluIsTheUnfusedNodes) {
  constexpr std::int64_t kFfn = 640;
  constexpr std::int64_t kUsed = 10;
  for (const std::int64_t t : {1, 7, 33}) {
    const auto gate_h = Normal(9, static_cast<std::size_t>(kFfn * kUsed * t), 3000.0f);
    const auto up_h = Normal(10, static_cast<std::size_t>(kFfn * kUsed * t), 3000.0f);
    const auto gs_h = ExpertScales(11);
    const auto us_h = ExpertScales(12);
    ggml_tensor* gate = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, t), gate_h);
    ggml_tensor* up = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, kUsed, t), up_h);
    ggml_tensor* gs = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 512), gs_h);
    ggml_tensor* us = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 512), us_h);
    ggml_tensor* ids = Ids(kUsed, t, 13);
    ggml_tensor* unfused = ggml_swiglu_split(c(), ggml_mul(c(), gate, ScaleRows(gs, ids)),
                                             ggml_mul(c(), up, ScaleRows(us, ids)));
    ggml_tensor* fused = kg::MoeGlu(c(), gate, up, ids, gs, us);
    Run({unfused, fused});
    EXPECT_EQ(PlannedFor(fused), kg::kMoeGluName);
    const auto got = Download(fused);
    ExpectSame(got, Download(unfused), "glu t " + std::to_string(t));
    std::vector<double> want(got.size());
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t j = 0; j < kUsed; ++j) {
        const auto e = static_cast<std::size_t>(Id(r, j));
        for (std::int64_t i = 0; i < kFfn; ++i) {
          const auto at = static_cast<std::size_t>((((r * kUsed) + j) * kFfn) + i);
          const double g = static_cast<double>(gate_h[at]) * gs_h[e];
          want[at] = g * Sigmoid(g) * (static_cast<double>(up_h[at]) * us_h[e]);
        }
      }
    }
    ExpectNear(got, want, "glu t " + std::to_string(t));
  }
}

TEST_F(Qwen38FusedTest, ExpertCombineIsTheUnfusedNodes) {
  constexpr std::int64_t kUsed = 10;
  for (const std::int64_t t : {1, 7, 33}) {
    const auto down_h = Normal(14, static_cast<std::size_t>(kWidth * kUsed * t), 2000.0f);
    const auto ds_h = ExpertScales(15);
    std::vector<float> w_h = Normal(16, static_cast<std::size_t>(kUsed * t), 0.05f, 0.1f);
    for (float& v : w_h) {
      v = std::abs(v);
    }
    const auto sh_h = Normal(17, static_cast<std::size_t>(kWidth * t), 0.3f);
    const auto sg_h = Normal(18, static_cast<std::size_t>(t), 2.0f);
    ggml_tensor* down = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kUsed, t), down_h);
    ggml_tensor* ds = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 512), ds_h);
    ggml_tensor* weights = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, kUsed, t), w_h);
    ggml_tensor* sh = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, t), sh_h);
    ggml_tensor* sg = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1, t), sg_h);
    ggml_tensor* ids = Ids(kUsed, t, 19);
    // build_moe_ffn's weighted sum and the gated shared expert, unfused.
    ggml_tensor* e = ggml_mul(c(), ggml_mul(c(), down, ScaleRows(ds, ids)), weights);
    ggml_tensor* unfused = ggml_view_2d(c(), e, kWidth, t, e->nb[2], 0);
    for (std::int64_t j = 1; j < kUsed; ++j) {
      unfused = ggml_add(
          c(), unfused,
          ggml_view_2d(c(), e, kWidth, t, e->nb[2], static_cast<std::size_t>(j) * e->nb[1]));
    }
    unfused = ggml_add(c(), unfused, ggml_mul(c(), sh, ggml_sigmoid(c(), sg)));
    ggml_tensor* fused = kg::MoeCombine(c(), down, ids, ds, weights, sh, sg);
    Run({unfused, fused});
    EXPECT_EQ(PlannedFor(fused), kg::kMoeCombineName);
    const auto got = Download(fused);
    ExpectSame(got, Download(unfused), "combine experts t " + std::to_string(t));
    std::vector<double> want(got.size());
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t i = 0; i < kWidth; ++i) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < kUsed; ++j) {
          sum += static_cast<double>(
                     down_h[static_cast<std::size_t>((((r * kUsed) + j) * kWidth) + i)]) *
                 ds_h[static_cast<std::size_t>(Id(r, j))] *
                 w_h[static_cast<std::size_t>((r * kUsed) + j)];
        }
        const auto at = static_cast<std::size_t>((r * kWidth) + i);
        want[at] = sum + (sh_h[at] * Sigmoid(sg_h[static_cast<std::size_t>(r)]));
      }
    }
    ExpectNear(got, want, "combine experts t " + std::to_string(t));
  }
}

// The BF16 product over activations converted once is GGML's cuBLAS
// product over the F32 activations, at the mixer's and the router's shapes.
TEST_F(Qwen38FusedTest, Bf16ProductIsGgmlsCublasProduct) {
  for (const auto& [k, n] : {std::pair{std::int64_t{10240}, std::int64_t{320}},
                             {std::int64_t{2560}, std::int64_t{512}},
                             {std::int64_t{10240}, std::int64_t{4}}}) {
    constexpr std::int64_t t = 33;
    const auto w_f = Normal(20, static_cast<std::size_t>(k * n), 0.02f);
    std::vector<std::uint16_t> w_h(w_f.size());
    std::ranges::transform(w_f, w_h.begin(), Bf16Bits);
    const auto x_h = Normal(21, static_cast<std::size_t>(k * t));
    ggml_tensor* w = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, k, n), w_h);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k, t), x_h);
    ggml_tensor* ggml_product = ggml_mul_mat(c(), w, x);
    ggml_tensor* converted = kg::ToBf16(c(), x);
    ggml_tensor* fused = kg::GemmBf16(c(), w, converted);
    Run({ggml_product, fused});
    EXPECT_EQ(PlannedFor(ggml_product), kg::kMulMatCublas);
    const std::string what = "bf16 product " + std::to_string(k) + "x" + std::to_string(n);
    const auto got = Download(fused);
    ExpectSame(got, Download(ggml_product), what);
    std::vector<double> want(got.size(), 0.0);
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t o = 0; o < n; ++o) {
        double sum = 0.0;
        for (std::int64_t i = 0; i < k; ++i) {
          sum += static_cast<double>(FromBf16(w_h[static_cast<std::size_t>((o * k) + i)])) *
                 FromBf16(Bf16Bits(x_h[static_cast<std::size_t>((r * k) + i)]));
        }
        want[static_cast<std::size_t>((r * n) + o)] = sum;
      }
    }
    ExpectNear(got, want, what);
  }
}

// The draft-head experiment retains GGML's F32-input vector arithmetic
// after BF16 round-to-nearest-even and exact widening. Include halfway
// cases, signed zero and a partial output tile, independently of the graph.
TEST_F(Qwen38FusedTest, RoundedDraftInputKeepsTheOriginalVectorProduct) {
  constexpr std::int64_t kRows = 260;
  auto x_h = Normal(231, static_cast<std::size_t>(kWidth));
  std::size_t at = 0;
  for (const std::uint32_t bits : {0x3F808000U, 0x3F818000U, 0xBF808000U, 0xBF818000U, 0x00000000U,
                                   0x80000000U, 0x00018000U, 0x80018000U}) {
    x_h[at++] = std::bit_cast<float>(bits);
  }
  std::vector<float> rounded_h(x_h.size());
  std::ranges::transform(x_h, rounded_h.begin(), [](float v) { return FromBf16(Bf16Bits(v)); });
  const auto w_f = Normal(232, static_cast<std::size_t>(kWidth * kRows), 0.02f);
  std::vector<std::uint16_t> w_h(w_f.size());
  std::ranges::transform(w_f, w_h.begin(), Bf16Bits);
  ggml_tensor* w = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, kWidth, kRows), w_h);
  ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 1), x_h);
  ggml_tensor* expected = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 1), rounded_h);
  ggml_tensor* zero = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 1), std::vector<std::int32_t>{0});
  ggml_tensor* widened = ggml_get_rows(c(), kg::ToBf16(c(), x), zero);
  ggml_tensor* got = ggml_mul_mat(c(), w, widened);
  ggml_tensor* control = ggml_mul_mat(c(), w, expected);
  Run({got, control});
  EXPECT_EQ(PlannedFor(got), kg::kMulMatVector);
  EXPECT_EQ(PlannedFor(control), kg::kMulMatVector);
  ExpectSame(Download(widened), rounded_h, "draft BF16 round/widen");
  ExpectSame(Download(got), Download(control), "original vector product over rounded input");
}

// jitLLM's column-blocked recurrence is upstream's gated_delta_net bit for
// bit at Qwen3.8's shape (16 query/key and 48 value heads of 128, q, k and v
// viewed out of the convolution's output as the graph views them), and
// within the default bound of the FP64 recurrence.
TEST_F(Qwen38FusedTest, GatedDeltaNetColumnsIsUpstreamsRecurrence) {
  constexpr std::int64_t kS = 128;
  constexpr std::int64_t kHk = 16;
  constexpr std::int64_t kHv = 48;
  constexpr std::int64_t kChannels = kS * ((2 * kHk) + kHv);
  const auto n = [](std::int64_t count) { return static_cast<std::size_t>(count); };
  for (const std::int64_t tokens : {1, 6, 40}) {
    // The convolution's output [channels, tokens]: q, k (unit-norm rows, as
    // the graph's L2 norm leaves them) and v.
    std::vector<float> conv = Normal(131, n(kChannels * tokens));
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t head = 0; head < 2 * kHk; ++head) {
        float* row = conv.data() + n((t * kChannels) + (head * kS));
        double norm = 0.0;
        for (std::int64_t i = 0; i < kS; ++i) {
          norm += static_cast<double>(row[i]) * row[i];
        }
        for (std::int64_t i = 0; i < kS; ++i) {
          row[i] = static_cast<float>(row[i] / std::sqrt(norm));
        }
      }
    }
    std::vector<float> g = Normal(134, n(kHv * tokens), 0.5f, -1.0f);
    for (float& v : g) {
      v = -std::abs(v);  // a log decay
    }
    std::vector<float> beta = Normal(135, n(kHv * tokens), 0.2f, 0.5f);
    for (float& v : beta) {
      v = std::clamp(v, 0.0f, 1.0f);
    }
    const std::vector<float> state = Normal(136, n(kS * kS * kHv), 0.1f);
    ggml_tensor* c_all = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kChannels, tokens), conv);
    const std::size_t head = ggml_row_size(GGML_TYPE_F32, kS);
    const std::size_t token = c_all->nb[1];
    ggml_tensor* tq =
        ggml_view_4d(c(), c_all, kS, kHk, tokens, 1, head, token, token * n(tokens), 0);
    ggml_tensor* tk =
        ggml_view_4d(c(), c_all, kS, kHk, tokens, 1, head, token, token * n(tokens), head * n(kHk));
    ggml_tensor* tv = ggml_view_4d(c(), c_all, kS, kHv, tokens, 1, head, token, token * n(tokens),
                                   head * n(2 * kHk));
    ggml_tensor* tg = Leaf(ggml_new_tensor_4d(c(), GGML_TYPE_F32, 1, kHv, tokens, 1), g);
    ggml_tensor* tb = Leaf(ggml_new_tensor_4d(c(), GGML_TYPE_F32, 1, kHv, tokens, 1), beta);
    ggml_tensor* ts = Leaf(ggml_new_tensor_4d(c(), GGML_TYPE_F32, kS, kS, kHv, 1), state);
    ggml_tensor* upstream = ggml_gated_delta_net(c(), tq, tk, tv, tg, tb, ts, 1);
    ggml_tensor* columns = ggml_gated_delta_net(c(), tq, tk, tv, tg, tb, ts, 1);
    ggml_tensor* lanes = ggml_gated_delta_net(c(), tq, tk, tv, tg, tb, ts, 1);
    for (ggml_tensor* out : {upstream, columns, lanes}) {
      TensorArena::Bind(out, Allocate(ggml_nbytes(out)));
    }
    const std::vector<ggml_tensor*> views = {tq, tk, tv};
    kg::BindViews(views);
    // The plan takes the lanes kernel past 16 tokens (prefill), the columns
    // kernel below.
    EXPECT_EQ(PlannedFor(columns), tokens > kg::kGatedDeltaNetLanesTokens
                                       ? kg::kGatedDeltaNetLanesName
                                       : kg::kGatedDeltaNetColumnsName);
    ASSERT_TRUE(kg::GatedDeltaNet(launch(), upstream).has_value());
    ASSERT_TRUE(kg::RunGatedDeltaNetColumns(launch(), columns).has_value());
    ASSERT_TRUE(kg::RunGatedDeltaNetLanes(launch(), lanes).has_value());
    const auto got = Download(columns);
    const std::string what = "gated_delta_net x " + std::to_string(tokens);
    const auto want_upstream = Download(upstream);
    ExpectSame(got, want_upstream, what);
    // The lanes kernel sums in another order: against upstream's, within
    // F32 rounding of those sums (NMSE 1e-12), and as close to FP64.
    const auto got_lanes = Download(lanes);
    const std::vector<double> upstream_d(want_upstream.begin(), want_upstream.end());
    EXPECT_LE(Nmse(got_lanes, upstream_d), 1e-12) << what << " (lanes)";
    std::cout << what << " (lanes): NMSE " << Nmse(got_lanes, upstream_d)
              << " against upstream's\n";
    // The FP64 recurrence: S[i][col] at state[(h · S + col) · S + i], value
    // head h reading query/key head h mod 16; the attention [S, Hv, tokens],
    // then the final state.
    std::vector<double> want(got.size());
    const double scale = 1.0 / std::sqrt(static_cast<double>(kS));
    for (std::int64_t h = 0; h < kHv; ++h) {
      std::vector<double> s(state.begin() + static_cast<std::ptrdiff_t>(h * kS * kS),
                            state.begin() + static_cast<std::ptrdiff_t>((h + 1) * kS * kS));
      for (std::int64_t t = 0; t < tokens; ++t) {
        const double decay = std::exp(static_cast<double>(g[n((t * kHv) + h)]));
        const double b = beta[n((t * kHv) + h)];
        const float* qt = conv.data() + n((t * kChannels) + ((h % kHk) * kS));
        const float* kt = qt + (kHk * kS);
        const float* vt = conv.data() + n((t * kChannels) + (((2 * kHk) + h) * kS));
        for (std::int64_t col = 0; col < kS; ++col) {
          double kv = 0.0;
          for (std::int64_t i = 0; i < kS; ++i) {
            kv += s[n((col * kS) + i)] * kt[i];
          }
          const double delta = (vt[col] - (decay * kv)) * b;
          double attn = 0.0;
          for (std::int64_t i = 0; i < kS; ++i) {
            double& cell = s[n((col * kS) + i)];
            cell = (decay * cell) + (kt[i] * delta);
            attn += cell * qt[i];
          }
          want[n(((t * kHv) + h) * kS) + n(col)] = attn * scale;
        }
      }
      std::ranges::copy(
          s, want.begin() + static_cast<std::ptrdiff_t>((kS * kHv * tokens) + (h * kS * kS)));
    }
    ExpectNear(got, want, what);
    ExpectNear(got_lanes, want, what + " (lanes)");
  }
}

// jitLLM's convolution is the graph's concatenation, ssm_conv, silu and
// the query and key heads' L2 norms bit for bit, at Qwen3.8's channels
// (16 + 16 + 48 heads of 128), and within the float bound of FP64.
TEST_F(Qwen38FusedTest, GdnConvIsTheUnfusedNodes) {
  constexpr std::int64_t kD = 128;
  constexpr std::int64_t kQk = std::int64_t{32} * kD;
  constexpr std::int64_t kC = kQk + (48 * kD);
  const float eps = kEps / static_cast<float>(kD);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
  for (const std::int64_t t : {3, 7, 40}) {
    const auto x_h = Normal(31 + static_cast<std::uint64_t>(t), static_cast<std::size_t>(kC * t));
    const auto hist_h = Normal(32, static_cast<std::size_t>(3 * kC));
    const auto w_h = Normal(33, static_cast<std::size_t>(4 * kC), 0.5f);
    ggml_tensor* x = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kC, t), x_h);
    ggml_tensor* hist = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 3 * kC, 1), hist_h);
    ggml_tensor* w = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 4, kC), w_h);
    // The graph's unfused nodes: history then the rows transposed, the
    // convolution, silu, and the L2 norm of the query and key heads.
    ggml_tensor* input = ggml_concat(c(), ggml_reshape_3d(c(), hist, 3, kC, 1),
                                     ggml_cont(c(), ggml_transpose(c(), x)), 0);
    // Past 32 tokens upstream's ssm_conv (ssm_conv_long_token_f32) loads
    // whole 32-token windows, which would read past the window when the
    // tokens are not a multiple of 32 (t = 40 here; RE-032, which
    // CheckSsmConv refuses): the graph pads the window to whole windows
    // with copies of its first columns and drops their outputs, as here.
    const std::int64_t pad = t > 32 && t % 32 != 0 ? 32 - (t % 32) : 0;
    ggml_tensor* windows = nullptr;
    if (pad != 0) {
      ggml_tensor* first = ggml_view_3d(c(), input, pad, kC, 1, input->nb[1], input->nb[2], 0);
      ggml_tensor* padded = ggml_concat(c(), input, ggml_cont(c(), first), 0);
      windows = ggml_ssm_conv(c(), padded, w);
      windows = ggml_view_3d(c(), windows, kC, t, 1, windows->nb[1],
                             windows->nb[1] * static_cast<std::size_t>(t), 0);
    } else {
      windows = ggml_ssm_conv(c(), input, w);
    }
    ggml_tensor* conv = ggml_silu(c(), windows);
    const std::size_t head = ggml_row_size(GGML_TYPE_F32, kD);
    const std::size_t token = ggml_row_size(GGML_TYPE_F32, kC);
    ggml_tensor* qk =
        ggml_cont(c(), ggml_view_3d(c(), conv, kD, std::int64_t{32}, t, head, token, 0));
    ggml_tensor* normed = ggml_scale(c(), ggml_rms_norm(c(), qk, eps), scale);
    ggml_tensor* fused = kg::GdnConv(c(), x, hist, w, kQk, kD, eps, scale);
    Run({conv, normed, fused});
    EXPECT_EQ(PlannedFor(fused), kg::kGdnConvName);
    const auto got = Download(fused);
    const auto plain = Download(conv);
    const auto norm = Download(normed);
    std::vector<float> want(got.size());
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t ch = 0; ch < kC; ++ch) {
        const auto at = static_cast<std::size_t>((r * kC) + ch);
        want[at] = ch < kQk ? norm[static_cast<std::size_t>((r * kQk) + ch)] : plain[at];
      }
    }
    const std::string what = "gdn conv t " + std::to_string(t);
    ExpectSame(got, want, what);
    // FP64: tap j of channel c reads time r + j of the history then rows.
    std::vector<double> ref(got.size());
    for (std::int64_t r = 0; r < t; ++r) {
      for (std::int64_t ch = 0; ch < kC; ++ch) {
        double sum = 0.0;
        for (std::int64_t j = 0; j < 4; ++j) {
          const std::int64_t tau = r + j;
          const double v = tau < 3 ? hist_h[static_cast<std::size_t>((ch * 3) + tau)]
                                   : x_h[static_cast<std::size_t>(((tau - 3) * kC) + ch)];
          sum += v * w_h[static_cast<std::size_t>((ch * 4) + j)];
        }
        ref[static_cast<std::size_t>((r * kC) + ch)] = sum / (1.0 + std::exp(-sum));
      }
      for (std::int64_t h = 0; h < kQk / kD; ++h) {
        double ss = 0.0;
        for (std::int64_t i = 0; i < kD; ++i) {
          const double v = ref[static_cast<std::size_t>((r * kC) + (h * kD) + i)];
          ss += v * v;
        }
        const double s = 1.0 / std::sqrt((ss / kD) + eps) / std::sqrt(static_cast<double>(kD));
        for (std::int64_t i = 0; i < kD; ++i) {
          ref[static_cast<std::size_t>((r * kC) + (h * kD) + i)] *= s;
        }
      }
    }
    ExpectNear(got, ref, what);
  }
}

// jitLLM's gated norm is the graph's rms_norm, weight, sigmoid and mul bit
// for bit (and rounded to nearest in BF16), at 48 heads of 128.
TEST_F(Qwen38FusedTest, GdnNormGateIsTheUnfusedNodes) {
  constexpr std::int64_t kD = 128;
  constexpr std::int64_t kH = 48;
  for (const std::int64_t t : {1, 7, 33}) {
    const auto o_h =
        Normal(41 + static_cast<std::uint64_t>(t), static_cast<std::size_t>(kD * kH * t));
    const auto z_h = Normal(42, static_cast<std::size_t>(kD * kH * t), 2.0f);
    const auto w_h = Normal(43, kD, 0.3f, 1.0f);
    ggml_tensor* o = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kD, kH, t), o_h);
    ggml_tensor* z = Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kD * kH, t), z_h);
    ggml_tensor* w = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kD), w_h);
    ggml_tensor* unfused = ggml_mul(c(), ggml_mul(c(), ggml_rms_norm(c(), o, kEps), w),
                                    ggml_sigmoid(c(), ggml_reshape_3d(c(), z, kD, kH, t)));
    ggml_tensor* f32 = kg::GdnNormGate(c(), o, w, z, kEps, GGML_TYPE_F32);
    ggml_tensor* bf16 = kg::GdnNormGate(c(), o, w, z, kEps, GGML_TYPE_BF16);
    Run({unfused, f32, bf16});
    EXPECT_EQ(PlannedFor(f32), kg::kGdnNormGateName);
    const auto want = Download(unfused);
    const std::string what = "gdn norm gate t " + std::to_string(t);
    ExpectSame(Download(f32), want, what);
    const auto rounded = Download<std::uint16_t>(bf16);
    std::size_t differing = 0;
    for (std::size_t i = 0; i < rounded.size(); ++i) {
      differing += rounded[i] != Bf16Bits(want[i]);
    }
    EXPECT_EQ(differing, 0U) << what;
    std::vector<double> ref(want.size());
    for (std::int64_t row = 0; row < kH * t; ++row) {
      double ss = 0.0;
      for (std::int64_t i = 0; i < kD; ++i) {
        const double v = o_h[static_cast<std::size_t>((row * kD) + i)];
        ss += v * v;
      }
      const double s = 1.0 / std::sqrt((ss / kD) + kEps);
      for (std::int64_t i = 0; i < kD; ++i) {
        const auto at = static_cast<std::size_t>((row * kD) + i);
        ref[at] = o_h[at] * s * w_h[static_cast<std::size_t>(i)] * Sigmoid(z_h[at]);
      }
    }
    ExpectNear(Download(f32), ref, what);
  }
}

TEST_F(Qwen38FusedTest, AnExpertIdOutsideTheScalesGivesNan) {
  constexpr std::int64_t kFfn = 640;
  ggml_tensor* gate = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, 2, 1),
                           Normal(22, static_cast<std::size_t>(kFfn * 2)));
  ggml_tensor* up = Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kFfn, 2, 1),
                         Normal(23, static_cast<std::size_t>(kFfn * 2)));
  ggml_tensor* ids =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 2, 1), std::vector<std::int32_t>{3, 512});
  ggml_tensor* s = Leaf(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 512), ExpertScales(24));
  ggml_tensor* fused = kg::MoeGlu(c(), gate, up, ids, s, s);
  Run({fused});
  const auto got = Download(fused);
  for (std::int64_t i = 0; i < kFfn; ++i) {
    EXPECT_TRUE(std::isfinite(got[static_cast<std::size_t>(i)])) << i;
    EXPECT_TRUE(std::isnan(got[static_cast<std::size_t>(kFfn + i)])) << i;
  }
}

TEST_F(Qwen38FusedTest, TheChecksRefuseWhatTheKernelsCannotRun) {
  const auto code = [&](ggml_tensor* node) -> std::optional<KernelError> {
    TensorArena::Bind(node, Allocate(ggml_nbytes(node) + 64));
    const std::vector<ggml_tensor*> nodes = {node};
    kg::BindViews(nodes);
    switch (kg::JitllmOpOf(node)) {
      case kg::JitllmOp::kHcCombine:
        return FailedCode(kg::RunHcCombine(launch(), node));
      case kg::JitllmOp::kHcNorm:
        return FailedCode(kg::RunHcNorm(launch(), node));
      case kg::JitllmOp::kHcMix:
        return FailedCode(kg::RunHcMix(launch(), node));
      case kg::JitllmOp::kMoeCombine:
        return FailedCode(kg::RunMoeCombine(launch(), node));
      case kg::JitllmOp::kGemmBf16:
        return FailedCode(kg::RunGemmBf16(launch(), node));
      default:
        return KernelError::kUnknown;
    }
  };
  const auto f32 = [&](std::int64_t a, std::int64_t b, std::int64_t d = 1) {
    return Leaf(ggml_new_tensor_3d(c(), GGML_TYPE_F32, a, b, d),
                std::vector<float>(static_cast<std::size_t>(a * b * d), 1.0f));
  };
  // Streams narrower than rms_norm's 1,024 threads.
  EXPECT_EQ(code(kg::HcNorm(c(), f32(512, 4, 2), f32(2048, 1), kEps, GGML_TYPE_F32)),
            KernelError::kRejected);
  // A weight of the wrong length.
  EXPECT_EQ(code(kg::HcMix(c(), f32(1024, 4, 2), f32(1024, 1), f32(4096, 2), kEps)),
            KernelError::kRejected);
  // Rows that are not whole float4s, and a misaligned output.
  EXPECT_EQ(code(kg::HcCombine(c(), f32(1026, 4, 2), f32(1026, 2), f32(4, 2))),
            KernelError::kRejected);
  ggml_tensor* skew = kg::HcCombine(c(), f32(1024, 4, 2), f32(1024, 2), f32(4, 2));
  TensorArena::Bind(skew, Allocate(ggml_nbytes(skew) + 64) + 8);
  EXPECT_EQ(FailedCode(kg::RunHcCombine(launch(), skew)), KernelError::kRejected);
  // Weights for an expert count the ids do not match: ids [used, t] of a
  // different used.
  ggml_tensor* ids =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 3, 2), std::vector<std::int32_t>(6, 0));
  EXPECT_EQ(code(kg::MoeCombine(c(), f32(1024, 2, 2), ids, f32(512, 1), f32(1, 2, 2), f32(1024, 2),
                                f32(1, 2))),
            KernelError::kRejected);
  // F32 activations for the BF16 product.
  ggml_tensor* w =
      Leaf(ggml_new_tensor_2d(c(), GGML_TYPE_BF16, 64, 8), std::vector<std::uint16_t>(512, 0));
  EXPECT_EQ(code(kg::GemmBf16(c(), w, f32(64, 4))), KernelError::kRejected);
}

}  // namespace
