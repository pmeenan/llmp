// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A speculative verify's row-invariant operations (D-092;
// kernels/ggml/ops_ext.h MulMatVecQRows and MulMatVecFRows, jitllm_ops.h
// Argmax and CopyRanges) under jitLLM's launch context on a GB10 (label
// `gpu`):
// - every column of a row-invariant quantized product of 1 to 8 columns
//   equals, bit for bit, GGML's own one-column MMVQ launch on that column
//   alone, for every weight type DeepSeek V4 Flash and its DSpark drafter
//   bring, at the reduction lengths of their projections (1,024 to 12,288,
//   so that the small-K and GB10's halved-iteration launches are both
//   met), with output channels (the grouped output projection) too;
// - every token of a row-invariant expert product equals GGML's one-token
//   launch on that token alone, broadcast and per-slot activations;
// - GGML's float vector kernel over 1 to 8 columns equals its one-column
//   launch, for F32 and BF16 weights, where upstream would take the tile
//   kernel or cuBLAS past one BF16 or three F32 columns;
// - jitLLM's argmax takes the lowest index among equal maxima and never a
//   NaN; the range copies copy exactly the ranges named;
// - the registry declares and binds the new implementations (D-053).

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
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {

using jitllm::base::Bytes;
using jitllm::kernels::ggml::KernelFailure;
using jitllm::kernels::ggml::LaunchContext;
using jitllm::kernels::ggml::QuantMulMatPath;
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

// `rows` rows of `k` values in `type`, quantized by GGML's own quantizer.
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

void Launched(const std::expected<void, KernelFailure>& result, const std::string& what) {
  EXPECT_TRUE(result.has_value()) << what << ": " << (result ? "" : result.error().detail);
}

class SpecRowsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = workspace, .size = Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(4096).value());
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

  // A view of `t`'s bytes as a new leaf (bound at the same address), so a
  // launcher sees a plain tensor.
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

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

std::uint64_t Address(const ggml_tensor* t) { return reinterpret_cast<std::uintptr_t>(t->data); }

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

TEST_F(SpecRowsTest, EveryColumnOfARowInvariantProductIsItsOneColumnLaunch) {
  struct Case {
    ggml_type type;
    std::int64_t k;
    std::int64_t groups;  // output channels sharing nothing (the grouped output projection)
  };
  // The dense projections of DeepSeek V4 Flash UD-Q2_K_XL and its Q8_0
  // drafter: q_b (k 1,024), q_a, kv, the compressors and the shared
  // expert's gate and up (4,096), its down (2,048), wo_b (8,192), the
  // drafter's fc (12,288), the head (Q4_K, 4,096), and wo_a's eight groups.
  const std::array<Case, 11> cases = {{{GGML_TYPE_Q8_0, 1024, 1},
                                       {GGML_TYPE_Q8_0, 4096, 1},
                                       {GGML_TYPE_Q8_0, 8192, 1},
                                       {GGML_TYPE_Q8_0, 12288, 1},
                                       {GGML_TYPE_Q8_0, 4096, 8},
                                       {GGML_TYPE_Q5_K, 4096, 1},
                                       {GGML_TYPE_Q6_K, 2048, 1},
                                       {GGML_TYPE_Q4_K, 4096, 1},
                                       {GGML_TYPE_IQ2_XS, 4096, 1},
                                       {GGML_TYPE_IQ3_XXS, 2048, 1},
                                       {GGML_TYPE_MXFP4, 2048, 1}}};
  constexpr std::int64_t kOut = 192;
  for (const Case& test : cases) {
    const std::vector<std::uint8_t> bytes = Quantize(test.type, test.k, kOut * test.groups, 5);
    ggml_tensor* w = Place(ggml_new_tensor_3d(c(), test.type, test.k, kOut, test.groups), bytes);
    for (std::int64_t columns = 1; columns <= kg::kRowsMaxColumns; ++columns) {
      const std::string what = std::string(ggml_type_name(test.type)) + " k " +
                               std::to_string(test.k) + " groups " + std::to_string(test.groups) +
                               " x " + std::to_string(columns);
      const std::vector<float> x = Normal(6 + static_cast<std::uint64_t>(columns),
                                          static_cast<std::size_t>(test.k * columns * test.groups));
      ggml_tensor* input =
          Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, columns, test.groups), x);
      ggml_tensor* rows = Place(ggml_mul_mat(c(), w, input));
      Launched(kg::MulMatVecQRows(launch(), rows), what);
      const std::vector<float> got = Download(rows);
      // Each column alone through GGML's own launcher.
      for (std::int64_t j = 0; j < columns; ++j) {
        ggml_tensor* one = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, 1, test.groups));
        for (std::int64_t g = 0; g < test.groups; ++g) {
          EXPECT_EQ(
              cudaMemcpy(static_cast<float*>(one->data) + (g * test.k),
                         x.data() + (((g * columns) + j) * test.k),
                         static_cast<std::size_t>(test.k) * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ggml_tensor* single = Place(ggml_mul_mat(c(), w, one));
        auto path = kg::SelectMulMatQ(launch(), single);
        ASSERT_TRUE(path.has_value() && *path == QuantMulMatPath::kVector) << what;
        Launched(kg::MulMatVecQ(launch(), single), what);
        const std::vector<float> want = Download(single);
        for (std::int64_t g = 0; g < test.groups; ++g) {
          const auto from = static_cast<std::size_t>(((g * columns) + j) * kOut);
          const std::vector<float> column(got.begin() + static_cast<std::ptrdiff_t>(from),
                                          got.begin() + static_cast<std::ptrdiff_t>(from + kOut));
          const std::vector<float> expected(
              want.begin() + static_cast<std::ptrdiff_t>(g * kOut),
              want.begin() + static_cast<std::ptrdiff_t>((g + 1) * kOut));
          EXPECT_TRUE(SameBits(column, expected)) << what << ": column " << j << " group " << g;
        }
      }
    }
  }
}

TEST_F(SpecRowsTest, EveryTokenOfARowInvariantExpertProductIsItsOneTokenLaunch) {
  constexpr std::int64_t kExperts = 16;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kOut = 64;
  struct Case {
    ggml_type type;
    std::int64_t k;
    bool broadcast;
  };
  const std::array<Case, 4> cases = {{{GGML_TYPE_IQ2_XS, 4096, true},
                                      {GGML_TYPE_IQ3_XXS, 2048, false},
                                      {GGML_TYPE_MXFP4, 2048, false},
                                      {GGML_TYPE_MXFP4, 4096, true}}};
  for (const Case& test : cases) {
    const std::vector<std::uint8_t> bytes = Quantize(test.type, test.k, kOut * kExperts, 7);
    ggml_tensor* w = Place(ggml_new_tensor_3d(c(), test.type, test.k, kOut, kExperts), bytes);
    const std::int64_t rows_in = test.broadcast ? 1 : kUsed;
    for (std::int64_t tokens = 1; tokens <= kg::kRowsMaxColumns; ++tokens) {
      const std::string what = std::string(ggml_type_name(test.type)) + " experts x " +
                               std::to_string(tokens) + (test.broadcast ? " (broadcast)" : "");
      const std::vector<float> x = Normal(8 + static_cast<std::uint64_t>(tokens),
                                          static_cast<std::size_t>(test.k * rows_in * tokens));
      ggml_tensor* input =
          Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, test.k, rows_in, tokens), x);
      std::mt19937 random(static_cast<unsigned>(9 + tokens));
      std::vector<std::int32_t> ids;
      for (std::int64_t t = 0; t < tokens; ++t) {
        std::vector<std::int32_t> experts(kExperts);
        std::ranges::iota(experts, 0);
        std::shuffle(experts.begin(), experts.end(), random);
        ids.insert(ids.end(), experts.begin(), experts.begin() + kUsed);
      }
      ggml_tensor* id_tensor = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, tokens), ids);
      ggml_tensor* rows = Place(ggml_mul_mat_id(c(), w, input, id_tensor));
      Launched(kg::MulMatVecQRows(launch(), rows), what);
      const std::vector<float> got = Download(rows);
      for (std::int64_t t = 0; t < tokens; ++t) {
        const auto token = static_cast<std::uint64_t>(t);
        ggml_tensor* one =
            Leaf(GGML_TYPE_F32, {test.k, rows_in, 1, 1}, Address(input) + (input->nb[2] * token));
        ggml_tensor* one_ids =
            Leaf(GGML_TYPE_I32, {kUsed, 1, 1, 1}, Address(id_tensor) + (id_tensor->nb[1] * token));
        ggml_tensor* single = Place(ggml_mul_mat_id(c(), w, one, one_ids));
        Launched(kg::MulMatVecQ(launch(), single), what);
        const std::vector<float> want = Download(single);
        const auto from = static_cast<std::ptrdiff_t>(t * kUsed * kOut);
        const std::vector<float> slice(
            got.begin() + from, got.begin() + from + static_cast<std::ptrdiff_t>(kUsed * kOut));
        EXPECT_TRUE(SameBits(slice, want)) << what << ": token " << t;
      }
    }
  }
}

TEST_F(SpecRowsTest, TheFloatVectorKernelsColumnsAreItsOneColumnLaunches) {
  struct Case {
    ggml_type type;
    std::int64_t k;
    std::int64_t n;
  };
  // DeepSeek's hyper-connection mixers (F32 [16384, 24]), the indexer's
  // weights (F32 [4096, 64]), the router (BF16 [4096, 256]) and
  // Qwen's selected MTP head (BF16 [2560, 16384]).
  const std::array<Case, 4> cases = {{{GGML_TYPE_F32, 16384, 24},
                                      {GGML_TYPE_F32, 4096, 64},
                                      {GGML_TYPE_BF16, 4096, 256},
                                      {GGML_TYPE_BF16, 2560, 16384}}};
  for (const Case& test : cases) {
    const std::vector<float> source = Normal(10, static_cast<std::size_t>(test.k * test.n), 0.02f);
    ggml_tensor* w = ggml_new_tensor_2d(c(), test.type, test.k, test.n);
    if (test.type == GGML_TYPE_F32) {
      Place(w, source);
    } else {
      std::vector<ggml_bf16_t> halves(source.size());
      for (std::size_t i = 0; i < source.size(); ++i) {
        halves[i] = ggml_fp32_to_bf16(source[i]);
      }
      Place(w, halves);
    }
    for (std::int64_t columns = 1; columns <= kg::kRowsMaxColumns; ++columns) {
      const std::string what = std::string(ggml_type_name(test.type)) + " [" +
                               std::to_string(test.k) + ", " + std::to_string(test.n) + "] x " +
                               std::to_string(columns);
      const std::vector<float> x = Normal(11 + static_cast<std::uint64_t>(columns),
                                          static_cast<std::size_t>(test.k * columns));
      ggml_tensor* input = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, test.k, columns), x);
      ggml_tensor* rows = Place(ggml_mul_mat(c(), w, input));
      Launched(kg::MulMatVecFRows(launch(), rows), what);
      const std::vector<float> got = Download(rows);
      for (std::int64_t j = 0; j < columns; ++j) {
        ggml_tensor* one = Leaf(GGML_TYPE_F32, {test.k, 1, 1, 1},
                                Address(input) + (input->nb[1] * static_cast<std::uint64_t>(j)));
        ggml_tensor* single = Place(ggml_mul_mat(c(), w, one));
        Launched(kg::MulMatVecF(launch(), single), what);
        const std::vector<float> want = Download(single);
        const auto from = static_cast<std::ptrdiff_t>(j * test.n);
        const std::vector<float> column(got.begin() + from,
                                        got.begin() + from + static_cast<std::ptrdiff_t>(test.n));
        EXPECT_TRUE(SameBits(column, want)) << what << ": column " << j;
      }
    }
  }
  // Past 8 columns the kernel is refused.
  ggml_tensor* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 8));
  ggml_tensor* wide = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 64, 9));
  EXPECT_FALSE(kg::MulMatVecFRows(launch(), Place(ggml_mul_mat(c(), w, wide))).has_value());
}

TEST_F(SpecRowsTest, SelectedDraftArgmaxMapsBackToOriginalTokenIds) {
  ggml_tensor* logits = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 4, 2),
                              std::vector<float>{0, 5, 5, 1, 0, 1, 2, 7});
  ggml_tensor* ids = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, 1, 4),
                           std::vector<std::int32_t>{0, 65537, 129280, 248319});
  ggml_tensor* selected = Place(kg::Argmax(c(), logits));
  ggml_tensor* mapped = Place(ggml_get_rows(c(), ids, selected));
  Launched(kg::RunArgmax(launch(), selected), "selected head argmax");
  Launched(kg::GetRows(launch(), mapped), "draft token map");
  EXPECT_EQ(Download<std::int32_t>(mapped), (std::vector<std::int32_t>{65537, 248319}));
}

TEST_F(SpecRowsTest, ArgmaxTakesTheLowestIndexAmongEqualMaximaAndNeverANan) {
  constexpr std::int64_t kN = 129280;
  constexpr std::int64_t kRows = 5;
  std::vector<float> x = Normal(12, static_cast<std::size_t>(kN * kRows));
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // Row 0: a tie between two maxima, the lower index first in no block.
  x[static_cast<std::size_t>(100000)] = 50.0f;
  x[static_cast<std::size_t>(77)] = 50.0f;
  // Row 1: a NaN above everything.
  x[static_cast<std::size_t>(kN + 3)] = nan;
  x[static_cast<std::size_t>(kN + 9)] = 40.0f;
  // Row 2: all -infinity: the first.
  std::fill(x.begin() + (2 * kN), x.begin() + (3 * kN), -std::numeric_limits<float>::infinity());
  // Row 3: all NaN: index 0.
  std::fill(x.begin() + (3 * kN), x.begin() + (4 * kN), nan);
  ggml_tensor* input = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kN, kRows), x);
  ggml_tensor* out = Place(kg::Argmax(c(), input));
  EXPECT_EQ(kg::JitllmOpOf(out), kg::JitllmOp::kArgmax);
  Launched(kg::RunArgmax(launch(), out), "argmax");
  const std::vector<std::int32_t> got = Download<std::int32_t>(out);
  ASSERT_EQ(got.size(), static_cast<std::size_t>(kRows));
  EXPECT_EQ(got[0], 77);
  EXPECT_EQ(got[1], 9);
  EXPECT_EQ(got[2], 0);
  EXPECT_EQ(got[3], 0);  // in the row, never -1: lookups index with it
  std::int32_t want = 0;
  for (std::int64_t i = 1; i < kN; ++i) {
    if (x[static_cast<std::size_t>((4 * kN) + i)] > x[static_cast<std::size_t>((4 * kN) + want)]) {
      want = static_cast<std::int32_t>(i);
    }
  }
  EXPECT_EQ(got[4], want);
}

TEST_F(SpecRowsTest, RangeCopiesCopyExactlyTheRangesNamed) {
  constexpr std::size_t kBytes = std::size_t{64} * 1024;
  std::vector<std::uint8_t> source(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i) {
    source[i] = static_cast<std::uint8_t>((i * 131U) + 7U);
  }
  ggml_tensor* from = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I8, kBytes), source);
  ggml_tensor* to =
      Place(ggml_new_tensor_1d(c(), GGML_TYPE_I8, kBytes), std::vector<std::uint8_t>(kBytes, 0));
  kg::RangeCopy* ranges = nullptr;
  ASSERT_EQ(cudaMallocHost(&ranges, 3 * sizeof(kg::RangeCopy)), cudaSuccess);
  ranges[0] = {.from = Address(from), .to = Address(to) + 4096, .bytes = 1024};
  ranges[1] = {.from = Address(from) + 8192, .to = Address(to), .bytes = 16};
  ranges[2] = {.from = Address(from) + 32768, .to = Address(to) + 40960, .bytes = 20480};
  Launched(kg::CopyRanges(launch(), ranges, 3), "range copies");
  const std::vector<std::uint8_t> got = Download<std::uint8_t>(to);
  std::vector<std::uint8_t> want(kBytes, 0);
  std::copy_n(source.begin(), 1024, want.begin() + 4096);
  std::copy_n(source.begin() + 8192, 16, want.begin());
  std::copy_n(source.begin() + 32768, 20480, want.begin() + 40960);
  EXPECT_EQ(got, want);
  // Unaligned ranges are refused before anything is queued.
  ranges[0].bytes = 1000;
  EXPECT_FALSE(kg::CopyRanges(launch(), ranges, 1).has_value());
  EXPECT_EQ(cudaFreeHost(ranges), cudaSuccess);
}

TEST_F(SpecRowsTest, TheRegistryDeclaresAndBindsTheRowInvariantImplementations) {
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  ASSERT_TRUE(registry.has_value());
  for (const std::string_view name :
       {kg::kMulMatVecQRows, kg::kMulMatIdVecQRows, kg::kMulMatVecFRows, kg::kArgmaxName}) {
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
