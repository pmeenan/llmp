// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's fast plan (kernels/ggml/jitllm_ops.h, dsv4_fast.cu; the
// owner's speed before bit exactness, D-085's note) on a GB10 (label
// `gpu`), at the model's widths:
// - jitllm.vecq's dense, grouped and routed products over one Q8_1
//   quantization against GGML's own MMVQ launches (the same per-block dot
//   products, other sums: NMSE bounds), every launch configuration built;
//   the routed products with experts shared between tokens (each read once)
//   and repeated within a token, per-slot and broadcast activations, and
//   the SwiGLU (clamped) against the two products' host SwiGLU;
// - a token's sums do not depend on the other tokens of its chunk under one
//   configuration (the dense form's reads shared);
// - the routing, the combination, the hyper-connection pre-mix and the
//   compressor against FP64 host references;
// - what their checks refuse, and the registry's declarations.

#include "kernels/ggml/dsv4_fast.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::KernelFailure;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::TensorArena;
using jitllm::providers::DeviceExecution;
using jitllm::providers::FenceState;
using jitllm::providers::StreamId;
namespace kg = jitllm::kernels::ggml;

constexpr std::uint64_t kWorkspace = 256ULL << 20;

std::vector<float> Normal(std::uint64_t seed, std::size_t count, float sigma = 1.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, sigma);
  std::vector<float> values(count);
  for (float& v : values) {
    v = normal(random);
  }
  return values;
}

std::vector<std::uint8_t> Quantize(ggml_type type, std::int64_t k, std::int64_t rows,
                                   std::uint64_t seed) {
  const std::vector<float> source = Normal(seed, static_cast<std::size_t>(k * rows), 0.05f);
  const std::vector<float> importance(static_cast<std::size_t>(k), 1.0f);
  std::vector<std::uint8_t> bytes(ggml_row_size(type, k) * static_cast<std::size_t>(rows));
  ggml_quantize_init(type);
  const std::size_t written =
      ggml_quantize_chunk(type, source.data(), bytes.data(), 0, rows, k,
                          ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
  EXPECT_EQ(written, bytes.size());
  return bytes;
}

template <typename T>
double Nmse(const std::vector<float>& got, const std::vector<T>& want) {
  EXPECT_EQ(got.size(), want.size());
  double error = 0.0;
  double norm = 0.0;
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
    error += d * d;
    norm += static_cast<double>(want[i]) * static_cast<double>(want[i]);
  }
  return norm > 0.0 ? error / norm : error;
}

template <typename T>
void ExpectNear(const std::vector<float>& got, const std::vector<T>& want, double bound,
                const std::string& what) {
  const double nmse = Nmse(got, want);
  EXPECT_LE(nmse, bound) << what;
  EXPECT_TRUE(std::ranges::all_of(got, [](float v) { return std::isfinite(v); })) << what;
}

void Launched(const std::expected<void, KernelFailure>& result, const std::string& what) {
  EXPECT_TRUE(result.has_value()) << what << ": " << (result ? "" : result.error().detail);
}

std::uint64_t Address(const ggml_tensor* t) { return reinterpret_cast<std::uintptr_t>(t->data); }

double Silu(double x) { return x / (1.0 + std::exp(-x)); }
double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

class Dsv4FastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = workspace, .size = Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(8192).value());
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

  template <typename T = float>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& data = {}) {
    const std::size_t size = ggml_nbytes(tensor);
    const std::uint64_t address = Allocate(size);
    TensorArena::Bind(tensor, address);
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), size);
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), data.data(), size,  // NOLINT
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    return tensor;
  }

  ggml_tensor* Leaf(ggml_type type, std::array<std::int64_t, 4> ne, std::uint64_t address) {
    ggml_tensor* t = ggml_new_tensor_4d(c(), type, ne[0], ne[1], ne[2], ne[3]);
    TensorArena::Bind(t, address);
    return t;
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  // x quantized once, then `node` (a jitllm.vecq over it) run.
  std::vector<float> RunVecQ(ggml_tensor* q8, ggml_tensor* node, const std::string& what) {
    Launched(kg::RunQuantizeQ8(launch(), q8), what + " (quantize)");
    Launched(kg::RunVecQ(launch(), node), what);
    return Download(node);
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

TEST_F(Dsv4FastTest, DenseAndGroupedProductsMatchGgmlsVectorKernel) {
  struct Case {
    ggml_type type;
    std::int64_t k;
    std::int64_t groups;
  };
  // DeepSeek V4 Flash's dense products: q_b (k 1,024), q_a, kv and the
  // compressors (4,096), the shared expert's down (2,048), wo_b (8,192),
  // the head (Q4_K), wo_a's eight groups; the drafter's MXFP4.
  const std::array<Case, 8> cases = {{{GGML_TYPE_Q8_0, 1024, 1},
                                      {GGML_TYPE_Q8_0, 4096, 1},
                                      {GGML_TYPE_Q8_0, 8192, 1},
                                      {GGML_TYPE_Q8_0, 4096, 8},
                                      {GGML_TYPE_Q5_K, 4096, 1},
                                      {GGML_TYPE_Q6_K, 2048, 1},
                                      {GGML_TYPE_Q4_K, 4096, 1},
                                      {GGML_TYPE_MXFP4, 2048, 1}}};
  constexpr std::int64_t kOut = 256;
  for (const Case& test : cases) {
    const std::vector<std::uint8_t> bytes = Quantize(test.type, test.k, kOut * test.groups, 5);
    ggml_tensor* w = Place(ggml_new_tensor_3d(c(), test.type, test.k, kOut, test.groups), bytes);
    for (const std::int64_t tokens : {1, 2, 4, 8}) {
      const std::string what = std::string(ggml_type_name(test.type)) + " k " +
                               std::to_string(test.k) + " groups " + std::to_string(test.groups) +
                               " x " + std::to_string(tokens);
      // jitLLM's rows: [k, groups, tokens] (a token's groups together).
      const std::vector<float> x = Normal(6 + static_cast<std::uint64_t>(tokens),
                                          static_cast<std::size_t>(test.k * test.groups * tokens));
      ggml_tensor* input =
          Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, test.groups, tokens), x);
      ggml_tensor* q8 = Place(kg::QuantizeQ8(c(), input));
      ggml_tensor* node = Place(kg::VecQ(c(), w, q8, nullptr, tokens, test.groups > 1));
      const std::vector<float> got = RunVecQ(q8, node, what);
      // GGML's MMVQ, a group at a time: [n, groups, tokens] from each
      // group's [k, tokens] view.
      std::vector<float> want(static_cast<std::size_t>(kOut * test.groups * tokens));
      for (std::int64_t g = 0; g < test.groups; ++g) {
        ggml_tensor* wg = Leaf(test.type, {test.k, kOut, 1, 1},
                               Address(w) + (w->nb[2] * static_cast<std::uint64_t>(g)));
        std::vector<float> rows(static_cast<std::size_t>(test.k * tokens));
        for (std::int64_t t = 0; t < tokens; ++t) {
          std::copy_n(x.begin() + (((t * test.groups) + g) * test.k), test.k,
                      rows.begin() + (t * test.k));
        }
        ggml_tensor* xg = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, test.k, tokens), rows);
        ggml_tensor* ref = Place(ggml_mul_mat(c(), wg, xg));
        Launched(kg::MulMatVecQ(launch(), ref), what + " (ggml)");
        const std::vector<float> r = Download(ref);
        for (std::int64_t t = 0; t < tokens; ++t) {
          std::copy_n(r.begin() + (t * kOut), kOut,
                      want.begin() + (((t * test.groups) + g) * kOut));
        }
      }
      ExpectNear(got, want, 1e-10, what);
      // A token's sums do not depend on the chunk's other tokens (one
      // configuration, the reads shared): each token alone.
      if (tokens > 1) {
        ggml_tensor* first = Leaf(GGML_TYPE_F32, {test.k, test.groups, 1, 1}, Address(input));
        ggml_tensor* q1 = Place(kg::QuantizeQ8(c(), first));
        ggml_tensor* one = Place(kg::VecQ(c(), w, q1, nullptr, 1, test.groups > 1));
        const std::vector<float> alone = RunVecQ(q1, one, what + " (alone)");
        ExpectNear(std::vector<float>(got.begin(), got.begin() + kOut * test.groups), alone, 1e-10,
                   what + " (token 0 alone)");
      }
    }
  }
}

// Every launch configuration on every weight type, dense, with and without
// the GLU, at row counts that leave a warp-form grid's last block partly
// past the rows (130) or that only one-row configurations take (129), and
// at a reduction length that is not a multiple of the activations' 512-value
// padding (1,056, the 32-value types): each writes every output (the
// output starts as NaN) and matches the host's products, and a
// configuration that cannot take the rows refuses them.
TEST_F(Dsv4FastTest, EveryConfigurationCoversEveryTypeTailRowsAndPaddedRows) {
  struct Case {
    ggml_type type;
    std::int64_t k;
  };
  const std::array<Case, 11> cases = {{{GGML_TYPE_Q8_0, 1024},
                                       {GGML_TYPE_Q2_K, 1024},
                                       {GGML_TYPE_IQ2_XXS, 1024},
                                       {GGML_TYPE_Q8_0, 1056},
                                       {GGML_TYPE_MXFP4, 1024},
                                       {GGML_TYPE_MXFP4, 1056},
                                       {GGML_TYPE_Q4_K, 1024},
                                       {GGML_TYPE_Q5_K, 1024},
                                       {GGML_TYPE_Q6_K, 1024},
                                       {GGML_TYPE_IQ2_XS, 1024},
                                       {GGML_TYPE_IQ3_XXS, 1024}}};
  constexpr float kLimit = 7.0f;
  // Each configuration's rows a block reads together (dsv4_fast.cu kVariants).
  constexpr std::array<std::int64_t, 11> kRows = {1, 1, 2, 1, 2, 4, 1, 2, 2, 1, 4};
  ASSERT_EQ(kg::VecQVariants(), static_cast<int>(kRows.size()));
  for (const Case& test : cases) {
    for (const std::int64_t nrows : {130, 129}) {
      const std::vector<std::uint8_t> bytes = Quantize(test.type, test.k, nrows, 11);
      ggml_tensor* w = Place(ggml_new_tensor_2d(c(), test.type, test.k, nrows), bytes);
      for (const std::int64_t tokens : {1, 3}) {
        const std::string what = std::string(ggml_type_name(test.type)) + " k " +
                                 std::to_string(test.k) + " rows " + std::to_string(nrows) + " x " +
                                 std::to_string(tokens);
        const std::vector<float> x = Normal(12 + static_cast<std::uint64_t>(tokens),
                                            static_cast<std::size_t>(test.k * tokens));
        ggml_tensor* input = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, test.k, tokens), x);
        ggml_tensor* q8 = Place(kg::QuantizeQ8(c(), input));
        Launched(kg::RunQuantizeQ8(launch(), q8), what + " (quantize)");
        // The host's FP64 products of the dequantized weights and the
        // device's own Q8_1 activations (GGML's MMVQ is no reference here:
        // it refuses a row of 1,056 and reads past an odd row count's last
        // row).
        const std::vector<std::uint8_t> blocks = Download<std::uint8_t>(q8);
        const std::int64_t y_row = (test.k + 511) / 512 * 512 / 32;
        std::vector<double> plain(static_cast<std::size_t>(nrows * tokens));
        {
          const ggml_type_traits* traits = ggml_get_type_traits(test.type);
          const std::size_t row_bytes = ggml_row_size(test.type, test.k);
          std::vector<float> wrow(static_cast<std::size_t>(test.k));
          for (std::int64_t r = 0; r < nrows; ++r) {
            traits->to_float(bytes.data() + (static_cast<std::size_t>(r) * row_bytes), wrow.data(),
                             test.k);
            for (std::int64_t t = 0; t < tokens; ++t) {
              double sum = 0.0;
              for (std::int64_t i = 0; i < test.k; ++i) {
                const std::size_t block = static_cast<std::size_t>((t * y_row) + (i / 32)) * 36;
                ggml_fp16_t d = 0;
                std::memcpy(&d, blocks.data() + block, sizeof(d));
                const auto q =
                    static_cast<std::int8_t>(blocks[block + 4 + static_cast<std::size_t>(i % 32)]);
                sum += static_cast<double>(wrow[static_cast<std::size_t>(i)]) *
                       (static_cast<double>(ggml_fp16_to_fp32(d)) * q);
              }
              plain[static_cast<std::size_t>((t * nrows) + r)] = sum;
            }
          }
        }
        std::vector<double> glu(plain.size());
        for (std::size_t i = 0; i < plain.size(); ++i) {
          const double r = plain[i];
          glu[i] = Silu(std::min<double>(r, kLimit)) * std::clamp<double>(r, -kLimit, kLimit);
        }
        ggml_tensor* out = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, nrows, tokens));
        for (int v = 0; v < kg::VecQVariants(); ++v) {
          for (const bool with_glu : {false, true}) {
            const std::string name = what + " " + kg::VecQVariantName(v) + (with_glu ? " glu" : "");
            kg::VecQDesc d;
            d.w = w->data;
            d.g = with_glu ? w->data : nullptr;
            d.y = q8->data;
            d.dst = static_cast<float*>(out->data);
            d.ncols_x = static_cast<int>(test.k);
            d.nrows = static_cast<int>(nrows);
            d.stride_row = static_cast<int>(w->nb[1] / ggml_type_size(test.type));
            d.stride_expert = static_cast<int>(w->nb[2] / ggml_type_size(test.type));
            d.y_token = static_cast<int>((test.k + 511) / 512 * 512 / 32);
            d.tokens = static_cast<int>(tokens);
            d.dst_token = static_cast<int>(nrows);
            d.glu = with_glu ? 2 : 0;
            d.limit = kLimit;
            Finish();
            // NaN everywhere: an output a configuration does not write shows.
            ASSERT_EQ(cudaMemset(out->data, 0xFF, ggml_nbytes(out)), cudaSuccess);
            const bool takes = nrows % kRows[static_cast<std::size_t>(v)] == 0;
            ASSERT_EQ(kg::LaunchVecQ(test.type, d, v, nullptr), takes) << name;
            if (!takes) {
              continue;
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess) << name;
            std::vector<float> got(ggml_nbytes(out) / sizeof(float));
            ASSERT_EQ(cudaMemcpy(got.data(), out->data, ggml_nbytes(out), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            // The kernel differs from the dequantized weights' products by
            // more than float rounding (IQ2_XS's and IQ3_XXS's integer scale
            // rounding, the K-quants' minimums over the unquantized sums);
            // one output of the 390 missing or misplaced is about 3e-3.
            ExpectNear(got, with_glu ? glu : plain, 1e-4, name);
          }
        }
      }
    }
  }
}

TEST_F(Dsv4FastTest, RoutedExpertsReadOnceMatchPerTokenProducts) {
  constexpr std::int64_t kExperts = 16;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kOut = 128;
  struct Case {
    ggml_type type;
    std::int64_t k;
    bool per_slot;  // the down projection's per-slot rows, else each token's row
    bool glu;
  };
  const std::array<Case, 6> cases = {{{GGML_TYPE_IQ2_XXS, 4096, false, true},
                                      {GGML_TYPE_Q2_K, 2048, true, false},
                                      {GGML_TYPE_IQ2_XS, 4096, false, true},
                                      {GGML_TYPE_IQ3_XXS, 2048, true, false},
                                      {GGML_TYPE_MXFP4, 4096, false, true},
                                      {GGML_TYPE_MXFP4, 2048, true, false}}};
  for (const Case& test : cases) {
    // Experts at a stride larger than their slices (the resident slab).
    const std::int64_t slice = static_cast<std::int64_t>(ggml_row_size(test.type, test.k)) * kOut;
    const std::int64_t stride = slice + (static_cast<std::int64_t>(ggml_type_size(test.type)) * 4);
    std::vector<std::uint8_t> up(static_cast<std::size_t>(stride * kExperts), 0);
    std::vector<std::uint8_t> gate(up.size(), 0);
    const std::vector<std::uint8_t> ub = Quantize(test.type, test.k, kOut * kExperts, 7);
    const std::vector<std::uint8_t> gb = Quantize(test.type, test.k, kOut * kExperts, 8);
    for (std::int64_t e = 0; e < kExperts; ++e) {
      std::copy_n(ub.begin() + (e * slice), slice, up.begin() + (e * stride));
      std::copy_n(gb.begin() + (e * slice), slice, gate.begin() + (e * stride));
    }
    const auto slab = [&](const std::vector<std::uint8_t>& bytes) {
      ggml_tensor* t = ggml_new_tensor_3d(c(), test.type, test.k, kOut, kExperts);
      const std::uint64_t address = Allocate(bytes.size());
      EXPECT_EQ(cudaMemcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size(),  // NOLINT
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      TensorArena::Bind(t, address);
      t->nb[2] = static_cast<std::size_t>(stride);
      t->nb[3] = t->nb[2] * kExperts;
      return t;
    };
    ggml_tensor* wu = slab(up);
    ggml_tensor* wg = test.glu ? slab(gate) : nullptr;
    for (const std::int64_t tokens : {1, 3, 4}) {
      const std::string what = std::string(ggml_type_name(test.type)) + " x " +
                               std::to_string(tokens) + (test.per_slot ? " per slot" : "") +
                               (test.glu ? " glu" : "");
      const std::int64_t rows_in = test.per_slot ? kUsed : 1;
      const std::vector<float> x = Normal(9 + static_cast<std::uint64_t>(tokens),
                                          static_cast<std::size_t>(test.k * rows_in * tokens));
      ggml_tensor* input =
          Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, rows_in, tokens), x);
      // Experts shared between tokens (token t takes t, t+1, ...), and one
      // repeated within the last token: every pair still computed.
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < tokens; ++t) {
        for (std::int64_t u = 0; u < kUsed; ++u) {
          ids.push_back(static_cast<std::int32_t>((t + u) % kExperts));
        }
      }
      ids.back() = ids[ids.size() - 2];
      ggml_tensor* id_tensor = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, tokens), ids);
      ggml_tensor* q8 = Place(kg::QuantizeQ8(c(), input));
      ggml_tensor* node =
          Place(kg::VecQ(c(), wu, q8, id_tensor, tokens, test.per_slot, wg,
                         test.glu ? kg::VecQGlu::kSwigluClamp : kg::VecQGlu::kNone, 7.0f));
      const std::vector<float> got = RunVecQ(q8, node, what);
      // GGML's MMVQ with ids, and the host's clamped SwiGLU over its two.
      const auto ggml = [&](ggml_tensor* w) {
        ggml_tensor* ref = Place(ggml_mul_mat_id(c(), w, input, id_tensor));
        Launched(kg::MulMatVecQ(launch(), ref), what + " (ggml)");
        return Download(ref);
      };
      std::vector<double> want;
      const std::vector<float> u_ref = ggml(wu);
      if (test.glu) {
        const std::vector<float> g_ref = ggml(wg);
        for (std::size_t i = 0; i < u_ref.size(); ++i) {
          const double g = std::min<double>(g_ref[i], 7.0);
          const double v = std::clamp<double>(u_ref[i], -7.0, 7.0);
          want.push_back(Silu(g) * v);
        }
      } else {
        want.assign(u_ref.begin(), u_ref.end());
      }
      ExpectNear(got, want, 1e-9, what);
      // Every launch configuration built gives the same products.
      for (int v = 0; v < kg::VecQVariants(); ++v) {
        kg::VecQDesc d;
        d.w = wu->data;
        d.g = wg != nullptr ? wg->data : nullptr;
        d.y = q8->data;
        d.ids = static_cast<const std::int32_t*>(id_tensor->data);
        d.dst = static_cast<float*>(node->data);
        d.ncols_x = static_cast<int>(test.k);
        d.nrows = static_cast<int>(kOut);
        d.stride_row = static_cast<int>(wu->nb[1] / ggml_type_size(test.type));
        d.stride_expert =
            static_cast<int>(stride / static_cast<std::int64_t>(ggml_type_size(test.type)));
        const int y_row = static_cast<int>((test.k + 511) / 512 * 512 / 32);
        d.y_token = test.per_slot ? y_row * static_cast<int>(kUsed) : y_row;
        d.y_slot = test.per_slot ? y_row : 0;
        d.ids_stride = static_cast<int>(kUsed);
        d.used = static_cast<int>(kUsed);
        d.tokens = static_cast<int>(tokens);
        d.dst_token = static_cast<int>(kOut * kUsed);
        d.dst_slot = static_cast<int>(kOut);
        d.glu = test.glu ? 2 : 0;
        d.limit = 7.0f;
        Finish();
        ASSERT_EQ(cudaMemset(node->data, 0, ggml_nbytes(node)), cudaSuccess);
        ASSERT_TRUE(kg::LaunchVecQ(test.type, d, v, nullptr))
            << what << " " << kg::VecQVariantName(v);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<float> variant(ggml_nbytes(node) / sizeof(float));
        ASSERT_EQ(cudaMemcpy(variant.data(), node->data, ggml_nbytes(node), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ExpectNear(variant, want, 1e-9, what + " " + kg::VecQVariantName(v));
      }
    }
  }
}

TEST_F(Dsv4FastTest, RoutingPicksTheTopExpertsAndNormalizesTheirWeights) {
  constexpr std::int64_t kExperts = 256;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kTokens = 5;
  constexpr float kClamp = 6.103515625e-5f;
  constexpr float kScale = 1.5f;
  const std::vector<float> logits = Normal(21, kExperts * kTokens, 2.0f);
  const std::vector<float> bias = Normal(22, kExperts, 0.1f);
  ggml_tensor* l = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kExperts, kTokens), logits);
  ggml_tensor* b = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kExperts), bias);
  ggml_tensor* route =
      Place(kg::Dsv4Route(c(), l, b, nullptr, nullptr, kUsed, true, kClamp, kScale));
  Launched(kg::RunDsv4Route(launch(), route), "route");
  const std::vector<std::int32_t> got = Download<std::int32_t>(route);
  // The hash layers' form: the table's experts, their weights the same way.
  std::vector<std::int32_t> table(static_cast<std::size_t>(kUsed * 10));
  for (std::size_t i = 0; i < table.size(); ++i) {
    table[i] = static_cast<std::int32_t>((i * 37) % kExperts);
  }
  const std::vector<std::int32_t> tokens = {3, 0, 9, 3, 7};
  ggml_tensor* tab = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, 10), table);
  ggml_tensor* tok = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, kTokens), tokens);
  ggml_tensor* hashed = Place(kg::Dsv4Route(c(), l, nullptr, tab, tok, kUsed, true, kClamp, 1.0f));
  Launched(kg::RunDsv4Route(launch(), hashed), "hashed route");
  const std::vector<std::int32_t> got_hashed = Download<std::int32_t>(hashed);
  for (std::int64_t t = 0; t < kTokens; ++t) {
    std::vector<double> p(kExperts);
    std::vector<std::pair<double, int>> sel;
    for (int e = 0; e < kExperts; ++e) {
      const double v = logits[static_cast<std::size_t>((t * kExperts) + e)];
      p[static_cast<std::size_t>(e)] = std::sqrt(v > 20.0 ? v : std::log1p(std::exp(v)));
      sel.emplace_back(p[static_cast<std::size_t>(e)] + bias[static_cast<std::size_t>(e)], -e);
    }
    std::ranges::sort(sel, std::greater<>());
    const auto check = [&](const std::vector<std::int32_t>& out, std::vector<int> want_ids,
                           double scale, const std::string& what) {
      double sum = 0.0;
      for (const int e : want_ids) {
        sum += p[static_cast<std::size_t>(e)];
      }
      for (std::int64_t k = 0; k < kUsed; ++k) {
        const auto at = static_cast<std::size_t>((t * 2 * kUsed) + k);
        EXPECT_EQ(out[at], want_ids[static_cast<std::size_t>(k)]) << what << " token " << t;
        float w = 0.0f;
        std::memcpy(&w, &out[at + kUsed], sizeof(w));
        const double want = p[static_cast<std::size_t>(want_ids[static_cast<std::size_t>(k)])] /
                            std::max<double>(sum, kClamp) * scale;
        EXPECT_NEAR(w, want, (1e-5 * std::abs(want)) + 1e-7) << what << " token " << t;
      }
    };
    std::vector<int> top;
    top.reserve(kUsed);
    for (std::int64_t k = 0; k < kUsed; ++k) {
      top.push_back(-sel[static_cast<std::size_t>(k)].second);
    }
    check(got, top, kScale, "biased");
    std::vector<int> listed;
    listed.reserve(kUsed);
    for (std::int64_t k = 0; k < kUsed; ++k) {
      listed.push_back(
          table[static_cast<std::size_t>((tokens[static_cast<std::size_t>(t)] * kUsed) + k)]);
    }
    check(got_hashed, listed, 1.0, "hashed");
  }
}

TEST_F(Dsv4FastTest, CombineIsTheWeightedSumPlusTheSharedExpert) {
  constexpr std::int64_t kN = 4096;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kTokens = 3;
  const std::vector<float> down = Normal(31, kN * kUsed * kTokens);
  const std::vector<float> shared = Normal(32, kN * kTokens);
  const std::vector<float> logits = Normal(33, 256 * kTokens);
  ggml_tensor* l = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 256, kTokens), logits);
  ggml_tensor* b =
      Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 256), std::vector<float>(256, 0.0f));
  ggml_tensor* route = Place(kg::Dsv4Route(c(), l, b, nullptr, nullptr, kUsed, true, 1e-4f, 1.0f));
  Launched(kg::RunDsv4Route(launch(), route), "route");
  const std::vector<std::int32_t> r = Download<std::int32_t>(route);
  ggml_tensor* d = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kN, kUsed, kTokens), down);
  ggml_tensor* s = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kN, kTokens), shared);
  ggml_tensor* out = Place(kg::Dsv4Combine(c(), d, route, s));
  Launched(kg::RunDsv4Combine(launch(), out), "combine");
  const std::vector<float> got = Download(out);
  std::vector<double> want(static_cast<std::size_t>(kN * kTokens));
  for (std::int64_t t = 0; t < kTokens; ++t) {
    for (std::int64_t i = 0; i < kN; ++i) {
      double v = shared[static_cast<std::size_t>((t * kN) + i)];
      for (std::int64_t k = 0; k < kUsed; ++k) {
        float w = 0.0f;
        std::memcpy(&w, &r[static_cast<std::size_t>((t * 2 * kUsed) + kUsed + k)], sizeof(w));
        v += static_cast<double>(w) * down[static_cast<std::size_t>((((t * kUsed) + k) * kN) + i)];
      }
      want[static_cast<std::size_t>((t * kN) + i)] = v;
    }
  }
  ExpectNear(got, want, 1e-12, "combine");
}

TEST_F(Dsv4FastTest, OrderedReductionMatchesSeparateGgmlProductsAndAdds) {
  constexpr std::int64_t kWidth = 4096;
  constexpr std::int64_t kSlots = 6;
  // One token, the first wide chunk and the admitted maximum exercise
  // both grid dimensions. Compare the actual GPU primitives, including
  // cancellation, signed zero, subnormals and large finite magnitudes.
  constexpr std::array<float, 8> pattern = {0.0f,
                                            -0.0f,
                                            std::numeric_limits<float>::denorm_min(),
                                            -std::numeric_limits<float>::denorm_min(),
                                            1e20f,
                                            -1e20f,
                                            1.0f,
                                            1.0000001f};
  for (const std::int64_t rows : {1, 9, 4096}) {
    std::vector<float> input(static_cast<std::size_t>(kWidth * kSlots * rows));
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = pattern[(i + (i / static_cast<std::size_t>(kWidth))) % pattern.size()];
    }
    const auto weights = Normal(903, static_cast<std::size_t>(kSlots * rows), 0.25f);
    auto* down = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kSlots, rows), input);
    auto* scale = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, kSlots, rows), weights);
    auto* product = Place(ggml_mul(c(), down, scale));
    Launched(kg::Mul(launch(), product), "ordinary six products");
    auto* ordinary = ggml_view_2d(c(), product, kWidth, rows, product->nb[2], 0);
    ordinary->data = product->data;
    for (std::int64_t slot = 1; slot < kSlots; ++slot) {
      auto* view = ggml_view_2d(c(), product, kWidth, rows, product->nb[2],
                                static_cast<std::size_t>(slot) * product->nb[1]);
      view->data = static_cast<std::byte*>(product->data) + view->view_offs;
      auto* sum = Place(ggml_add(c(), ordinary, view));
      Launched(kg::Add(launch(), sum), "ordinary ordered sum");
      ordinary = sum;
    }
    auto* fused = kg::Dsv4OrderedReduce(c(), down, scale);
    const auto bytes = ggml_nbytes(fused);
    const auto address = Allocate(bytes + 256);
    TensorArena::Bind(fused, address);
    const std::array<std::uint8_t, 256> guard = [] {
      std::array<std::uint8_t, 256> value{};
      value.fill(0xa7);
      return value;
    }();
    EXPECT_EQ(cudaMemcpy(static_cast<std::byte*>(fused->data) + bytes, guard.data(), guard.size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    Launched(kg::RunDsv4OrderedReduce(launch(), fused), "ordered weighted reduction");
    const auto want = Download(ordinary);
    const auto got = Download(fused);
    ASSERT_EQ(got.size(), want.size());
    EXPECT_EQ(std::memcmp(got.data(), want.data(), bytes), 0) << rows << " rows";
    std::array<std::uint8_t, 256> after{};
    EXPECT_EQ(cudaMemcpy(after.data(), static_cast<std::byte*>(fused->data) + bytes, after.size(),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(after, guard);
  }
}

TEST_F(Dsv4FastTest, OrderedReductionRefusesInvalidBoundsAndAliasesBeforeLaunch) {
  auto* down = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 4096, 6, 1));
  auto* weights = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 6, 1));
  auto* out = Place(kg::Dsv4OrderedReduce(c(), down, weights), std::vector<float>(4096, 17.0f));
  const kg::Dsv4WeightedReduce good{.down = {down, Bytes(ggml_nbytes(down))},
                                    .weights = {weights, Bytes(ggml_nbytes(weights))},
                                    .values = {out, Bytes(ggml_nbytes(out))}};
  ASSERT_TRUE(kg::CheckDsv4OrderedReduce(out).has_value());
  auto short_input = good;
  short_input.down.bytes = Bytes(ggml_nbytes(down) - sizeof(float));
  EXPECT_FALSE(kg::RunDsv4WeightedReduce(launch(), short_input).has_value());
  auto wrapping = good;
  wrapping.down.bytes = Bytes(std::numeric_limits<std::uint64_t>::max());
  EXPECT_FALSE(kg::RunDsv4WeightedReduce(launch(), wrapping).has_value());
  auto alias = *out;
  alias.data = down->data;
  auto overlap = good;
  overlap.values.tensor = &alias;
  EXPECT_FALSE(kg::RunDsv4WeightedReduce(launch(), overlap).has_value());
  auto unaligned = *out;
  unaligned.data = static_cast<std::byte*>(out->data) + 1;
  auto bad_address = good;
  bad_address.values.tensor = &unaligned;
  EXPECT_FALSE(kg::RunDsv4WeightedReduce(launch(), bad_address).has_value());
  auto pitched = *down;
  pitched.nb[2] += sizeof(float);
  auto bad_stride = good;
  bad_stride.down.tensor = &pitched;
  EXPECT_FALSE(kg::RunDsv4WeightedReduce(launch(), bad_stride).has_value());
  auto extra = *out;
  extra.src[2] = weights;
  EXPECT_FALSE(kg::RunDsv4OrderedReduce(launch(), &extra).has_value());
  const auto unchanged = Download(out);
  EXPECT_TRUE(std::ranges::all_of(unchanged, [](float value) { return value == 17.0f; }));
}

TEST_F(Dsv4FastTest, TheHyperConnectionPreMixMatchesFp64) {
  constexpr std::int64_t kWidth = 4096;
  constexpr std::int64_t kHc = 4;
  constexpr std::int64_t kMixes = 24;
  constexpr float kRmsEps = 1e-6f;
  constexpr float kHcEps = 1e-6f;
  constexpr int kIterations = 20;
  for (const std::int64_t tokens : {1, 4, 5}) {
    const std::string what = "hc pre x " + std::to_string(tokens);
    const std::vector<float> x = Normal(41 + static_cast<std::uint64_t>(tokens),
                                        static_cast<std::size_t>(kWidth * kHc * tokens));
    const std::vector<float> fn =
        Normal(42, static_cast<std::size_t>(kWidth * kHc * kMixes), 0.01f);
    const std::vector<float> scale = {0.7f, 1.3f, 0.9f};
    const std::vector<float> base = Normal(43, kMixes, 0.5f);
    const std::vector<float> norm = Normal(44, kWidth, 0.3f);
    ggml_tensor* xt = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kHc, tokens), x);
    ggml_tensor* ft = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth * kHc, kMixes), fn);
    ggml_tensor* st = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 3), scale);
    ggml_tensor* bt = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kMixes), base);
    ggml_tensor* nt = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, kWidth), norm);
    ggml_tensor* mix = Place(kg::Dsv4HcMix(c(), xt, ft));
    ggml_tensor* pre = Place(kg::Dsv4HcPre(c(), mix, xt, st, bt, nt, kRmsEps, kHcEps, kIterations));
    Launched(kg::RunDsv4HcMix(launch(), mix), what);
    Launched(kg::RunDsv4HcPre(launch(), pre), what);
    const std::vector<float> got = Download(pre);
    std::vector<double> want(got.size());
    for (std::int64_t t = 0; t < tokens; ++t) {
      const float* xs = x.data() + (t * kWidth * kHc);
      double ss = 0.0;
      for (std::int64_t k = 0; k < kWidth * kHc; ++k) {
        ss += static_cast<double>(xs[k]) * xs[k];
      }
      const double rms = 1.0 / std::sqrt((ss / static_cast<double>(kWidth * kHc)) + kRmsEps);
      std::array<double, kMixes> m{};
      for (std::int64_t j = 0; j < kMixes; ++j) {
        double dot = 0.0;
        for (std::int64_t k = 0; k < kWidth * kHc; ++k) {
          dot += static_cast<double>(fn[static_cast<std::size_t>((j * kWidth * kHc) + k)]) * xs[k];
        }
        m[static_cast<std::size_t>(j)] = dot * rms;
      }
      std::array<double, kHc> p{};
      double* tail = want.data() + (tokens * kWidth) + (t * kg::kDsv4HcTail);
      for (std::int64_t h = 0; h < kHc; ++h) {
        p[static_cast<std::size_t>(h)] = Sigmoid((m[static_cast<std::size_t>(h)] * scale[0]) +
                                                 base[static_cast<std::size_t>(h)]) +
                                         kHcEps;
        tail[h] = 2.0 * Sigmoid((m[static_cast<std::size_t>(kHc + h)] * scale[1]) +
                                base[static_cast<std::size_t>(kHc + h)]);
      }
      std::array<double, kHc * kHc> comb{};
      for (std::int64_t isrc = 0; isrc < kHc; ++isrc) {
        double mx = -1e300;
        for (std::int64_t idst = 0; idst < kHc; ++idst) {
          const auto idx = static_cast<std::size_t>(idst + (kHc * isrc));
          comb[idx] = (m[(2 * kHc) + idx] * scale[2]) + base[(2 * kHc) + idx];
          mx = std::max(mx, comb[idx]);
        }
        double sum = 0.0;
        for (std::int64_t idst = 0; idst < kHc; ++idst) {
          const auto idx = static_cast<std::size_t>(idst + (kHc * isrc));
          comb[idx] = std::exp(comb[idx] - mx);
          sum += comb[idx];
        }
        for (std::int64_t idst = 0; idst < kHc; ++idst) {
          const auto idx = static_cast<std::size_t>(idst + (kHc * isrc));
          comb[idx] = (comb[idx] / sum) + kHcEps;
        }
      }
      const auto normalize = [&](bool columns) {
        for (std::int64_t a = 0; a < kHc; ++a) {
          double sum = kHcEps;
          for (std::int64_t b2 = 0; b2 < kHc; ++b2) {
            sum += comb[static_cast<std::size_t>(columns ? a + (kHc * b2) : b2 + (kHc * a))];
          }
          for (std::int64_t b2 = 0; b2 < kHc; ++b2) {
            comb[static_cast<std::size_t>(columns ? a + (kHc * b2) : b2 + (kHc * a))] /= sum;
          }
        }
      };
      normalize(true);
      for (int i = 1; i < kIterations; ++i) {
        normalize(false);
        normalize(true);
      }
      for (std::size_t idx = 0; idx < comb.size(); ++idx) {
        tail[kHc + static_cast<std::int64_t>(idx)] = comb[idx];
      }
      std::vector<double> y(static_cast<std::size_t>(kWidth));
      double yy = 0.0;
      for (std::int64_t i = 0; i < kWidth; ++i) {
        double v = 0.0;
        for (std::int64_t h = 0; h < kHc; ++h) {
          v += static_cast<double>(xs[(h * kWidth) + i]) * p[static_cast<std::size_t>(h)];
        }
        y[static_cast<std::size_t>(i)] = v;
        yy += v * v;
      }
      const double r = 1.0 / std::sqrt((yy / static_cast<double>(kWidth)) + kRmsEps);
      for (std::int64_t i = 0; i < kWidth; ++i) {
        want[static_cast<std::size_t>((t * kWidth) + i)] =
            y[static_cast<std::size_t>(i)] * r * norm[static_cast<std::size_t>(i)];
      }
    }
    // The tails' unused floats (after post and comb) are not written.
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t j = kHc + (kHc * kHc); j < kg::kDsv4HcTail; ++j) {
        const auto at = static_cast<std::size_t>((tokens * kWidth) + (t * kg::kDsv4HcTail) + j);
        want[at] = got[at];
      }
    }
    ExpectNear(got, want, 1e-9, what);
  }
}

TEST_F(Dsv4FastTest, TheCompressorMatchesFp64AndGivesZerosForMaskedBlocks) {
  for (const bool overlap : {true, false}) {
    const std::int64_t ratio = overlap ? 4 : 128;
    const std::int64_t head = 512;
    const std::int64_t channels = overlap ? 2 * head : head;
    const std::int64_t state_rows = overlap ? 2 * ratio : ratio;
    const std::int64_t tokens = 3;
    const std::int64_t blocks = 2;
    const std::int64_t per_block = overlap ? 2 * ratio : ratio;
    const std::string what = overlap ? "csa" : "hca";
    const std::vector<float> skv = Normal(51, static_cast<std::size_t>(channels * state_rows));
    const std::vector<float> ssc = Normal(52, static_cast<std::size_t>(channels * state_rows));
    const std::vector<float> kv = Normal(53, static_cast<std::size_t>(channels * tokens));
    const std::vector<float> sc = Normal(54, static_cast<std::size_t>(channels * tokens));
    // Block 0 reads real rows; block 1 only the zero row (or out of range),
    // as a dummy block does.
    std::vector<std::int32_t> read(static_cast<std::size_t>(per_block * blocks));
    const auto zero = static_cast<std::int32_t>(state_rows + tokens);
    std::mt19937 random(55);  // NOLINT(bugprone-random-generator-seed): reproducible
    for (std::int64_t i = 0; i < per_block * blocks; ++i) {
      const bool dummy = overlap ? (i / ratio) % 2 == 1 : i >= ratio;
      read[static_cast<std::size_t>(i)] =
          dummy ? zero : static_cast<std::int32_t>(random() % static_cast<unsigned>(zero));
    }
    ggml_tensor* a = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, state_rows), skv);
    ggml_tensor* b = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, state_rows), ssc);
    ggml_tensor* k = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, tokens), kv);
    ggml_tensor* s = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, channels, tokens), sc);
    ggml_tensor* r = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, per_block * blocks), read);
    ggml_tensor* out = Place(kg::Dsv4Compress(c(), a, b, k, s, r, ratio, overlap));
    Launched(kg::RunDsv4Compress(launch(), out), what);
    const std::vector<float> got = Download(out);
    std::vector<double> want(got.size(), 0.0);
    const std::int64_t n_read = ratio * blocks;
    for (std::int64_t blk = 0; blk < blocks; ++blk) {
      for (std::int64_t ch = 0; ch < head; ++ch) {
        std::vector<std::pair<double, double>> terms;
        for (std::int64_t j = 0; j < per_block; ++j) {
          const bool second = overlap && j >= ratio;
          const std::int64_t index =
              second ? n_read + (blk * ratio) + (j - ratio) : (blk * ratio) + j;
          const std::int64_t row = read[static_cast<std::size_t>(index)];
          const std::int64_t col = second ? head + ch : ch;
          if (row < state_rows) {
            terms.emplace_back(ssc[static_cast<std::size_t>((row * channels) + col)],
                               skv[static_cast<std::size_t>((row * channels) + col)]);
          } else if (row < state_rows + tokens) {
            terms.emplace_back(sc[static_cast<std::size_t>(((row - state_rows) * channels) + col)],
                               kv[static_cast<std::size_t>(((row - state_rows) * channels) + col)]);
          }
        }
        if (terms.empty()) {
          continue;  // every score -inf: zeros
        }
        double mx = -1e300;
        for (const auto& [score, value] : terms) {
          mx = std::max(mx, score);
        }
        double sum = 0.0;
        double acc = 0.0;
        for (const auto& [score, value] : terms) {
          sum += std::exp(score - mx);
          acc += std::exp(score - mx) * value;
        }
        want[static_cast<std::size_t>((blk * head) + ch)] = acc / sum;
      }
    }
    ExpectNear(got, want, 1e-10, what);
    for (std::int64_t ch = 0; ch < head; ++ch) {
      EXPECT_EQ(got[static_cast<std::size_t>(head + ch)], 0.0f) << what << ": the dummy block";
    }
  }
}

TEST_F(Dsv4FastTest, TheChecksRefuseWhatTheKernelsDoNotTake) {
  const std::vector<std::uint8_t> bytes = Quantize(GGML_TYPE_Q8_0, 1024, 64, 3);
  ggml_tensor* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, 1024, 64), bytes);
  // Nine tokens: more than a vecq takes.
  ggml_tensor* x9 = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 9));
  ggml_tensor* q9 = Place(kg::QuantizeQ8(c(), x9));
  EXPECT_FALSE(kg::CheckVecQ(Place(kg::VecQ(c(), w, q9, nullptr, 9, false))).has_value());
  // A gate of another type.
  const std::vector<std::uint8_t> other = Quantize(GGML_TYPE_Q5_K, 1024, 64, 4);
  ggml_tensor* g = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q5_K, 1024, 64), other);
  ggml_tensor* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 2));
  ggml_tensor* q = Place(kg::QuantizeQ8(c(), x));
  EXPECT_FALSE(kg::CheckVecQ(Place(kg::VecQ(c(), w, q, nullptr, 2, false, g, kg::VecQGlu::kSwiglu)))
                   .has_value());
  // A GLU without gate weights.
  EXPECT_FALSE(
      kg::CheckVecQ(Place(kg::VecQ(c(), w, q, nullptr, 2, false, nullptr, kg::VecQGlu::kSwiglu)))
          .has_value());
  // Activations quantized for another row count.
  ggml_tensor* x3 = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 3));
  ggml_tensor* q3 = Place(kg::QuantizeQ8(c(), x3));
  EXPECT_FALSE(kg::CheckVecQ(Place(kg::VecQ(c(), w, q3, nullptr, 2, false))).has_value());
  EXPECT_TRUE(kg::CheckVecQ(Place(kg::VecQ(c(), w, q, nullptr, 2, false))).has_value());
  // Routing over something other than 256 experts.
  ggml_tensor* l = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 128, 2));
  ggml_tensor* bias = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 128));
  EXPECT_FALSE(
      kg::CheckDsv4Route(Place(kg::Dsv4Route(c(), l, bias, nullptr, nullptr, 6, true, 1e-4f, 1.0f)))
          .has_value());
  // Read indices not a whole number of blocks.
  ggml_tensor* sk = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 8));
  ggml_tensor* ss = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 8));
  ggml_tensor* ck = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 1));
  ggml_tensor* cs = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 1024, 1));
  ggml_tensor* r7 = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 7));
  ggml_tensor* comp = kg::Dsv4Compress(c(), sk, ss, ck, cs, r7, 4, true);
  EXPECT_FALSE(kg::CheckDsv4Compress(Place(comp)).has_value());
}

TEST_F(Dsv4FastTest, TheRegistryDeclaresAndBindsTheFastPlansImplementations) {
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry.has_value());
  for (const std::string_view name :
       {kg::kQuantizeQ8Name, kg::kVecQName, kg::kDsv4RouteName, kg::kDsv4CombineName,
        kg::kDsv4HcMixName, kg::kDsv4HcPreName, kg::kDsv4CompressName,
        std::string_view{kg::kDsv4WeightedReduceName}}) {
    bool found = false;
    for (const jitllm::execution::Implementation& implementation : kg::Implementations()) {
      if (implementation.name == name) {
        found = true;
        EXPECT_TRUE(kg::Kernel::Bind(implementation).has_value()) << name;
      }
    }
    EXPECT_TRUE(found) << name;
  }
}

// ---------------------------------------------------------------- sparse attention

// The indexer's inputs with small integer values and weights of a few
// powers of two, so every score is exact in F32 whatever the summation
// order: the kernels' scores equal the host's, and many of them tie.
struct LidCase {
  std::int64_t rows = 0;
  std::int64_t n_kv = 0;
  std::vector<float> q;               // [128, 64, rows]
  std::vector<ggml_fp16_t> k;         // [128, n_kv]
  std::vector<float> w;               // [64, rows]
  std::vector<std::int32_t> visible;  // [rows]
};

LidCase MakeLid(std::int64_t rows, std::int64_t n_kv, std::uint64_t seed) {
  LidCase c;
  c.rows = rows;
  c.n_kv = n_kv;
  std::mt19937_64 random(seed);
  std::uniform_int_distribution<int> small(-2, 2);
  c.q.resize(static_cast<std::size_t>(std::int64_t{128} * 64 * rows));
  for (float& v : c.q) {
    v = static_cast<float>(small(random));
  }
  c.k.resize(static_cast<std::size_t>(128 * n_kv));
  for (ggml_fp16_t& v : c.k) {
    v = ggml_fp32_to_fp16(static_cast<float>(small(random)));
  }
  const std::array<float, 5> weights = {-1.0f, -0.5f, 0.5f, 1.0f, 2.0f};
  c.w.resize(static_cast<std::size_t>(64 * rows));
  for (float& v : c.w) {
    v = weights[random() % weights.size()];
  }
  c.visible.resize(static_cast<std::size_t>(rows));
  for (std::int64_t r = 0; r < rows; ++r) {
    // The whole cache, a few rows, none, fewer than the selection, and
    // anything between.
    const std::array<std::int64_t, 5> fixed = {n_kv, 1, 0, 300, n_kv - 3};
    c.visible[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(
        r < 5 ? std::min(fixed[static_cast<std::size_t>(r)], n_kv)
              : static_cast<std::int64_t>(random() % static_cast<std::uint64_t>(n_kv + 1)));
  }
  return c;
}

// The host's selection: each row's `top` best visible rows by score, the
// lower row first among equals, listed in ascending order, -1 after.
std::vector<std::int32_t> HostLidTopK(const LidCase& c, std::int64_t top) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(top * c.rows), -1);
  for (std::int64_t r = 0; r < c.rows; ++r) {
    const std::int64_t n = std::min<std::int64_t>(c.visible[static_cast<std::size_t>(r)], c.n_kv);
    std::vector<std::pair<double, std::int32_t>> scored;
    for (std::int64_t j = 0; j < n; ++j) {
      double score = 0.0;
      for (std::int64_t h = 0; h < 64; ++h) {
        double dot = 0.0;
        for (std::int64_t d = 0; d < 128; ++d) {
          dot +=
              static_cast<double>(c.q[static_cast<std::size_t>((((r * 64) + h) * 128) + d)]) *
              static_cast<double>(ggml_fp16_to_fp32(c.k[static_cast<std::size_t>((j * 128) + d)]));
        }
        score +=
            static_cast<double>(c.w[static_cast<std::size_t>((r * 64) + h)]) * std::max(dot, 0.0);
      }
      scored.emplace_back(score, static_cast<std::int32_t>(j));
    }
    std::ranges::sort(scored, [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::vector<std::int32_t> kept;
    for (std::size_t i = 0; i < std::min<std::size_t>(scored.size(), static_cast<std::size_t>(top));
         ++i) {
      kept.push_back(scored[i].second);
    }
    std::ranges::sort(kept);
    std::ranges::copy(kept, out.begin() + (r * top));
  }
  return out;
}

TEST_F(Dsv4FastTest, TheIndexerSelectsItsBestRowsTiesToTheLowerRow) {
  // A few rows over many keys select in two stages (slices of 2,048).
  for (const auto& [rows, keys] : {std::pair<std::int64_t, std::int64_t>{1, 1536},
                                   {2, 1536},
                                   {3, 1536},
                                   {4, 1536},
                                   {9, 1536},
                                   {37, 1536},
                                   {1, 20480},
                                   {3, 20480},
                                   {9, 5000}}) {
    const std::string what = std::to_string(rows) + " rows, " + std::to_string(keys) + " keys";
    const LidCase lc = MakeLid(rows, keys, 60 + static_cast<std::uint64_t>(rows + keys));
    ggml_tensor* q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 128, 64, rows), lc.q);
    ggml_tensor* k = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 128, lc.n_kv), lc.k);
    ggml_tensor* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, rows), lc.w);
    ggml_tensor* v = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows), lc.visible);
    ggml_tensor* top = Place(kg::Dsv4LidTopK(c(), q, k, w, v, 512));
    Launched(kg::RunDsv4LidTopK(launch(), top), what);
    const std::vector<std::int32_t> got = Download<std::int32_t>(top);
    EXPECT_EQ(got, HostLidTopK(lc, 512)) << what;
    // And again, bit for bit.
    ggml_tensor* again = Place(kg::Dsv4LidTopK(c(), q, k, w, v, 512));
    Launched(kg::RunDsv4LidTopK(launch(), again), what);
    EXPECT_EQ(Download<std::int32_t>(again), got) << what << ": a repeat";
  }
}

// Past kDsv4LidScratch the rows are scored in groups; each row's selection
// is its own, the same as a node of that row alone.
TEST_F(Dsv4FastTest, TheIndexersRowGroupsSelectAsEachRowAlone) {
  constexpr std::int64_t kRows = 600;
  constexpr std::int64_t kKeys = 65536;
  ASSERT_GT(kRows * kKeys * 4, kg::kDsv4LidScratch);
  const LidCase lc = MakeLid(kRows, kKeys, 77);
  ggml_tensor* q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 128, 64, kRows), lc.q);
  ggml_tensor* k = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 128, kKeys), lc.k);
  ggml_tensor* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, kRows), lc.w);
  ggml_tensor* v = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, kRows), lc.visible);
  ggml_tensor* top = Place(kg::Dsv4LidTopK(c(), q, k, w, v, 512));
  auto scratch = kg::PlanDsv4LidTopK(launch(), top);
  ASSERT_TRUE(scratch.has_value());
  EXPECT_LE(*scratch, static_cast<std::uint64_t>(kg::kDsv4LidScratch));
  Launched(kg::RunDsv4LidTopK(launch(), top), "600 rows");
  const std::vector<std::int32_t> got = Download<std::int32_t>(top);
  for (const std::int64_t r : {0, 1, 3, 5, 511, 512, 599}) {
    ggml_tensor* q1 = ggml_view_3d(c(), q, 128, 64, 1, q->nb[1], q->nb[2],
                                   static_cast<std::size_t>(r) * q->nb[2]);
    ggml_tensor* w1 = ggml_view_2d(c(), w, 64, 1, w->nb[1], static_cast<std::size_t>(r) * w->nb[1]);
    ggml_tensor* v1 = ggml_view_1d(c(), v, 1, static_cast<std::size_t>(r) * sizeof(std::int32_t));
    for (ggml_tensor* view : {q1, w1, v1}) {
      TensorArena::Bind(view, Address(view->view_src) + view->view_offs);
    }
    ggml_tensor* one = Place(kg::Dsv4LidTopK(c(), q1, k, w1, v1, 512));
    Launched(kg::RunDsv4LidTopK(launch(), one), "row " + std::to_string(r));
    const std::vector<std::int32_t> alone = Download<std::int32_t>(one);
    EXPECT_TRUE(std::equal(alone.begin(), alone.end(), got.begin() + (r * 512))) << "row " << r;
  }
  // Two rows against the host's exact selection.
  LidCase two = lc;
  two.rows = 2;
  two.q.resize(static_cast<std::size_t>(128 * 64 * 2));
  two.w.resize(128);
  two.visible.resize(2);
  const std::vector<std::int32_t> want = HostLidTopK(two, 512);
  EXPECT_TRUE(std::equal(want.begin(), want.end(), got.begin()));
}

TEST_F(Dsv4FastTest, TheSparseMaskKeepsTheWindowAndTheSelectedOrVisibleRows) {
  constexpr std::int64_t kRows = 3;
  constexpr std::int64_t kWidth = 256;  // the window mask's
  constexpr std::int64_t kCells = 512;  // the compressed rows start here
  constexpr std::int64_t kKv = 768;
  std::vector<ggml_fp16_t> window(static_cast<std::size_t>(kWidth * kRows));
  for (std::size_t i = 0; i < window.size(); ++i) {
    window[i] = ggml_fp32_to_fp16(i % 3 == 0 ? 0.0f : -INFINITY);
  }
  ggml_tensor* wm = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, kWidth, kRows), window);
  // A selection from the indexer, and visible counts.
  const LidCase lc = MakeLid(kRows, kKv, 9);
  ggml_tensor* q = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 128, 64, kRows), lc.q);
  ggml_tensor* k = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 128, kKv), lc.k);
  ggml_tensor* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, kRows), lc.w);
  ggml_tensor* v = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, kRows), lc.visible);
  ggml_tensor* top = Place(kg::Dsv4LidTopK(c(), q, k, w, v, 512));
  Launched(kg::RunDsv4LidTopK(launch(), top), "selection");
  const std::vector<std::int32_t> selected = Download<std::int32_t>(top);
  ggml_tensor* by_top = Place(kg::Dsv4SparseMask(c(), wm, top, nullptr, kCells, kKv));
  ggml_tensor* by_count = Place(kg::Dsv4SparseMask(c(), wm, nullptr, v, kCells, kKv));
  Launched(kg::RunDsv4SparseMask(launch(), by_top), "by selection");
  Launched(kg::RunDsv4SparseMask(launch(), by_count), "by count");
  const std::vector<ggml_fp16_t> got_top = Download<ggml_fp16_t>(by_top);
  const std::vector<ggml_fp16_t> got_count = Download<ggml_fp16_t>(by_count);
  const std::uint16_t zero = 0x0000;
  const std::uint16_t neg = 0xFC00;
  for (std::int64_t r = 0; r < kRows; ++r) {
    std::vector<std::uint16_t> want_top(kCells + kKv, neg);
    std::vector<std::uint16_t> want_count(kCells + kKv, neg);
    for (std::int64_t c0 = 0; c0 < kWidth; ++c0) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, &window[static_cast<std::size_t>((r * kWidth) + c0)], 2);
      want_top[static_cast<std::size_t>(c0)] = bits;
      want_count[static_cast<std::size_t>(c0)] = bits;
    }
    for (std::int64_t j = 0; j < 512; ++j) {
      const std::int32_t s = selected[static_cast<std::size_t>((r * 512) + j)];
      if (s >= 0) {
        want_top[static_cast<std::size_t>(kCells + s)] = zero;
      }
    }
    for (std::int64_t j = 0;
         j < std::min<std::int64_t>(lc.visible[static_cast<std::size_t>(r)], kKv); ++j) {
      want_count[static_cast<std::size_t>(kCells + j)] = zero;
    }
    const auto row = [&](const std::vector<ggml_fp16_t>& got) {
      std::vector<std::uint16_t> bits(static_cast<std::size_t>(kCells + kKv));
      std::memcpy(bits.data(), got.data() + (r * (kCells + kKv)), bits.size() * 2);
      return bits;
    };
    EXPECT_EQ(row(got_top), want_top) << "row " << r;
    EXPECT_EQ(row(got_count), want_count) << "row " << r;
  }
  // A mask whose compressed rows start inside the window's, or whose
  // selection is not the indexer's, is refused.
  EXPECT_FALSE(
      kg::CheckDsv4SparseMask(Place(kg::Dsv4SparseMask(c(), wm, nullptr, v, kWidth - 1, kKv)))
          .has_value());
  EXPECT_FALSE(kg::CheckDsv4SparseMask(Place(kg::Dsv4SparseMask(c(), wm, v, nullptr, kCells, kKv)))
                   .has_value());
  // An indexer of other head widths.
  ggml_tensor* q64 = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 64, 64, kRows));
  EXPECT_FALSE(kg::CheckDsv4LidTopK(Place(kg::Dsv4LidTopK(c(), q64, k, w, v, 512))).has_value());
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry.has_value());
  for (const std::string_view name : {kg::kDsv4LidTopKName, kg::kDsv4SparseMaskName}) {
    bool found = false;
    for (const jitllm::execution::Implementation& implementation : kg::Implementations()) {
      if (implementation.name == name) {
        found = true;
        EXPECT_TRUE(kg::Kernel::Bind(implementation).has_value()) << name;
      }
    }
    EXPECT_TRUE(found) << name;
  }
}

}  // namespace
