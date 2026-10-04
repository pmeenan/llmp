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
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
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
  // The pinned Ampere+ configurations use KV tiles of 32 at D256/group2
  // with eight query columns, 64 at D256/group8 with one query column,
  // 32 at D512/group8 with one query column, and 128 at unmasked D128/MHA
  // with eight query columns. Each case below has
  // a final safe 256-cell block after including the kernel's transient
  // addition of one extra KV tile before modulo. The next padding block
  // is refused. Only metadata is bound; admitted plans never run.
  struct Shape {
    std::int64_t d, rows, heads, kv_heads, kv_tile;
  };
  for (const Shape shape : {Shape{256, 8, 128, 64, 32}, Shape{256, 1, 1024, 128, 64},
                            Shape{512, 1, 512, 64, 32}, Shape{128, 8, 256, 256, 128}}) {
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
    // This earlier total-only endpoint is also refused: a last partial
    // partition can overflow kbc+iter_k or kb0_start+kbc_stop before
    // subtraction/modulo, even though total iterations fit signed int.
    k->ne[1] = v->ne[1] = 1073741568;
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

}  // namespace
