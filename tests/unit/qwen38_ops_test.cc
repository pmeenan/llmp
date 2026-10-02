// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 Flash Next's formats on a GB10 (label `gpu`), each result against
// an FP64 reference built from the format's own dequantization:
// - jitLLM's MXFP8 vector product (jitllm_ops.h) at the model's input widths
//   (2560, 6144, 640) and 1 to 8 columns, within upstream's default bound
//   for float products (NMSE 1e-7: its F32 block sums differ only in order);
//   and its BF16 dequantization, exactly;
// - the n-gram table's NVFP4 row lookup, exactly, and NaN for an id outside
//   the table; and a GGUF checkpoint's (IQ4_NL and the other 32-value block
//   types), GGML's CPU dequantization bit for bit;
// - GGML's NVFP4 expert products (mul_mat_id) over weights in ModelOpt's
//   layout repacked here into GGML's blocks (independently of the importer),
//   at the model's shapes, the down projection's 640-element rows with the
//   readable padding marked: MMVQ (Q8_1 activations) within upstream's
//   quantized bound (5e-4), MMQ (FP4 activations on Blackwell) within its
//   FP4-activation bound (2e-2);
// - GGML's tensor-core flash attention at Qwen3.8's QSA shape (24 query and
//   2 KV heads of 256, 12 query heads per KV head) without sinks (RE-030:
//   only sinks are over-read at that ratio), within upstream's bound (5e-4),
//   and with sinks still refused.

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
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

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

constexpr double kDefaultNmse = 1e-7;
constexpr double kMulMatNmse = 5e-4;
constexpr double kFp4ActivationNmse = 2e-2;
constexpr double kFlashAttnNmse = 5e-4;
constexpr std::uint64_t kWorkspace = 256ULL << 20;

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

// OCP E4M3 (bias 7, no infinities, 0x7F and 0xFF NaN).
double E4m3(std::uint8_t b) {
  const int sign = (b & 0x80) != 0 ? -1 : 1;
  const int e = (b >> 3) & 0xF;
  const int m = b & 7;
  if (e == 0) {
    return sign * std::ldexp(m / 8.0, -6);
  }
  return sign * std::ldexp(1.0 + (m / 8.0), e - 7);
}

double E2m1(std::uint8_t code) {
  constexpr std::array<double, 8> kValues = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  return ((code & 8) != 0 ? -1.0 : 1.0) * kValues[code & 7];
}

// E4M3 codes that are not NaN, of moderate magnitude (|v| in [2^-4, 2^4]).
std::vector<std::uint8_t> E4m3Codes(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 random(seed);
  std::uniform_int_distribution<int> exponent(3, 11);
  std::uniform_int_distribution<int> mantissa(0, 7);
  std::uniform_int_distribution<int> sign(0, 1);
  std::vector<std::uint8_t> out(count);
  for (auto& b : out) {
    b = static_cast<std::uint8_t>((sign(random) << 7) | (exponent(random) << 3) | mantissa(random));
  }
  return out;
}

class Qwen38OpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = LaunchContext::Create(0, *execution_, stream_,
                                        {.base = workspace, .size = Bytes(kWorkspace)});
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(256).value());
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

  // Binds `tensor` to new device memory (with `extra` readable bytes past
  // it, zeroed) holding `data` if given.
  template <typename T = float>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& data = {}, std::size_t extra = 0) {
    const std::size_t size = ggml_nbytes(tensor);
    const std::uint64_t address = Allocate(size + extra);
    auto* pointer = reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
    EXPECT_EQ(cudaMemset(pointer, 0, size + extra), cudaSuccess);
    TensorArena::Bind(tensor, address);
    if (!data.empty()) {
      EXPECT_EQ(data.size() * sizeof(T), size);
      EXPECT_EQ(cudaMemcpy(pointer, data.data(), size, cudaMemcpyHostToDevice), cudaSuccess);
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
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

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

TEST_F(Qwen38OpsTest, Mxfp8VectorProductMatchesTheReference) {
  constexpr std::int64_t kRows = 192;
  for (const std::int64_t k : {2560, 6144, 640}) {
    const std::vector<std::uint8_t> codes =
        E4m3Codes(static_cast<std::uint64_t>(k), static_cast<std::size_t>(k * kRows));
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(k / 32 * kRows));
    std::mt19937 random(static_cast<unsigned>(k));
    for (auto& s : scales) {
      s = static_cast<std::uint8_t>(118 + (random() % 12));  // 2^-9 .. 2^2
    }
    for (const std::int64_t columns : {1, 2, 3, 4, 5, 6, 7, 8}) {
      const std::string what = "mxfp8 k " + std::to_string(k) + " x " + std::to_string(columns);
      std::vector<float> x(static_cast<std::size_t>(k * columns));
      std::normal_distribution<float> normal(0.0f, 1.0f);
      for (float& v : x) {
        v = normal(random);
      }
      ggml_tensor* wc = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, kRows), codes);
      ggml_tensor* ws = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k / 32, kRows), scales);
      ggml_tensor* in = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k, columns), x);
      ggml_tensor* y = Place(kg::Mxfp8MulMatVec(c(), wc, ws, in));
      ASSERT_TRUE(kg::RunMxfp8MulMatVec(launch(), y).has_value()) << what;
      std::vector<double> want(static_cast<std::size_t>(kRows * columns));
      for (std::int64_t j = 0; j < columns; ++j) {
        for (std::int64_t r = 0; r < kRows; ++r) {
          double sum = 0.0;
          for (std::int64_t i = 0; i < k; ++i) {
            const auto at = static_cast<std::size_t>((r * k) + i);
            const double scale =
                std::ldexp(1.0, scales[static_cast<std::size_t>(((r * k) + i) / 32)] - 127);
            sum += E4m3(codes[at]) * scale * x[static_cast<std::size_t>((j * k) + i)];
          }
          want[static_cast<std::size_t>((j * kRows) + r)] = sum;
        }
      }
      ExpectNmse(Download(y), want, kDefaultNmse, what);
    }
  }
}

// The measured small-output schedules keep the lone-column arithmetic,
// including each column's padded stride, while grouping different rows
// and warps. Covers tuned and untuned column counts over the same weights.
TEST_F(Qwen38OpsTest, SmallMxfp8ProductsMatchLoneColumnsBitForBit) {
  constexpr std::int64_t k = 2560;
  constexpr std::int64_t kStride = k + 12;
  constexpr std::int64_t kColumns = 8;
  for (const std::int64_t rows : {48, 512, 640}) {
    const auto codes =
        E4m3Codes(static_cast<std::uint64_t>(rows), static_cast<std::size_t>(k * rows));
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(k / 32 * rows));
    std::mt19937 random(static_cast<unsigned>(rows));
    for (auto& scale : scales) scale = static_cast<std::uint8_t>(118 + (random() % 12));
    std::vector<float> input(static_cast<std::size_t>(kStride * kColumns));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float& value : input) value = normal(random);
    auto* wc = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, rows), codes);
    auto* ws = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k / 32, rows), scales);
    auto* full = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kStride, kColumns), input);
    std::vector<std::uint32_t> want;
    for (std::int64_t column = 0; column < kColumns; ++column) {
      auto* x =
          ggml_view_1d(c(), full, k, static_cast<std::size_t>(column * kStride) * sizeof(float));
      auto* y = Place(kg::Mxfp8MulMatVec(c(), wc, ws, x));
      ASSERT_TRUE(kg::RunMxfp8MulMatVec(launch(), y).has_value());
      const auto bits = Download<std::uint32_t>(y);
      want.insert(want.end(), bits.begin(), bits.end());
    }
    for (std::int64_t columns = 2; columns <= kColumns; ++columns) {
      auto* x =
          ggml_view_2d(c(), full, k, columns, static_cast<std::size_t>(kStride) * sizeof(float), 0);
      auto* y = Place(kg::Mxfp8MulMatVec(c(), wc, ws, x));
      ASSERT_TRUE(kg::RunMxfp8MulMatVec(launch(), y).has_value());
      const auto bits = Download<std::uint32_t>(y);
      ASSERT_EQ(bits.size(), static_cast<std::size_t>(rows * columns));
      EXPECT_TRUE(std::equal(bits.begin(), bits.end(), want.begin()))
          << "outputs " << rows << ", columns " << columns;
    }
  }
}

// The vector product's refusals: more than 8 columns, k not whole 32-code
// blocks, activations off their 16-byte alignment.
TEST_F(Qwen38OpsTest, Mxfp8VectorProductRefusesWhatItCannotRun) {
  const auto refused = [&](std::int64_t k, std::int64_t columns, std::size_t x_offset) {
    ggml_tensor* wc = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, 32));
    ggml_tensor* ws =
        Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, std::max<std::int64_t>(k / 32, 1), 32));
    ggml_tensor* in = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k, columns), {}, 16);
    in->data = static_cast<char*>(in->data) + x_offset;
    ggml_tensor* y = Place(kg::Mxfp8MulMatVec(c(), wc, ws, in));
    return FailedCode(kg::RunMxfp8MulMatVec(launch(), y)) == KernelError::kRejected;
  };
  EXPECT_FALSE(refused(64, 8, 0));
  EXPECT_TRUE(refused(64, 9, 0));
  EXPECT_TRUE(refused(48, 1, 0));
  EXPECT_TRUE(refused(64, 1, 4));
}

// E8M0 scales at the ends of their range decode as powers of two, 0x00 the
// subnormal 2^-127, and 0xFF as NaN (OCP MX); E4M3 codes likewise, 0x7F NaN.
// Dequantization is exact wherever the product is a BF16 value.
TEST_F(Qwen38OpsTest, Mxfp8ScaleAndCodeEdgesDecodeAsTheFormatSays) {
  constexpr std::int64_t k = 32;
  const std::vector<std::uint8_t> scale_edges = {0x00, 0x01, 127, 254, 0xFF};
  const std::vector<std::uint8_t> code_edges = {0x38, 0xB8, 0x01, 0x07, 0x08,
                                                0x7E, 0xFE, 0x00, 0x80, 0x7F};
  const auto rows = static_cast<std::int64_t>(scale_edges.size());
  std::vector<std::uint8_t> codes(static_cast<std::size_t>(k * rows), 0x38);
  for (std::size_t r = 0; r < scale_edges.size(); ++r) {
    std::ranges::copy(code_edges, codes.begin() + static_cast<std::ptrdiff_t>(r * k));
  }
  ggml_tensor* wc = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, rows), codes);
  ggml_tensor* ws = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, 1, rows), scale_edges);
  ggml_tensor* out = Place(kg::Mxfp8Dequant(c(), wc, ws));
  ASSERT_TRUE(kg::RunMxfp8Dequant(launch(), out).has_value());
  const std::vector<std::uint16_t> got = Download<std::uint16_t>(out);
  for (std::size_t r = 0; r < scale_edges.size(); ++r) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(k); ++i) {
      const std::uint8_t code = codes[(r * k) + i];
      const auto value =
          static_cast<double>(std::bit_cast<float>(std::uint32_t{got[(r * k) + i]} << 16U));
      if (scale_edges[r] == 0xFF || (code & 0x7F) == 0x7F) {
        EXPECT_TRUE(std::isnan(value)) << r << " " << i;
        continue;
      }
      const double want = E4m3(code) * std::ldexp(1.0, scale_edges[r] - 127);
      // BF16 holds 2^-133 to below 2^128 with an 8-bit significand.
      const double magnitude = std::abs(want);
      if (magnitude == 0.0 || (magnitude >= std::ldexp(1.0, -133) && magnitude < 0x1p128)) {
        const auto bf16 = static_cast<double>(std::bit_cast<float>(
            std::bit_cast<std::uint32_t>(static_cast<float>(want)) & 0xFFFF0000U));
        if (bf16 == want) {
          EXPECT_EQ(value, want) << "scale " << int{scale_edges[r]} << " code " << int{code};
        }
      }
    }
  }
  // The vector product: the same edges against x = 1 (NaN rows and codes aside).
  ggml_tensor* in =
      Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, k, 1), std::vector<float>(k, 1.0f));
  ggml_tensor* y = Place(kg::Mxfp8MulMatVec(c(), wc, ws, in));
  ASSERT_TRUE(kg::RunMxfp8MulMatVec(launch(), y).has_value());
  const std::vector<float> sums = Download(y);
  for (std::size_t r = 0; r < scale_edges.size(); ++r) {
    EXPECT_TRUE(std::isnan(sums[r])) << r;  // every row holds the NaN code 0x7F
  }
}

TEST_F(Qwen38OpsTest, Mxfp8DequantizationIsExact) {
  constexpr std::int64_t k = 256;
  constexpr std::int64_t rows = 24;
  const std::vector<std::uint8_t> codes = E4m3Codes(5, k * rows);
  std::vector<std::uint8_t> scales(k / 32 * rows);
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = static_cast<std::uint8_t>(110 + (i % 30));
  }
  ggml_tensor* wc = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k, rows), codes);
  ggml_tensor* ws = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, k / 32, rows), scales);
  ggml_tensor* out = Place(kg::Mxfp8Dequant(c(), wc, ws));
  ASSERT_TRUE(kg::RunMxfp8Dequant(launch(), out).has_value());
  const std::vector<std::uint16_t> got = Download<std::uint16_t>(out);
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double want = E4m3(codes[i]) * std::ldexp(1.0, scales[i / 32] - 127);
    const auto as_float = std::bit_cast<float>(std::uint32_t{got[i]} << 16U);
    ASSERT_EQ(static_cast<double>(as_float), want) << i;
  }
  // The checks: scales of the wrong shape are refused.
  ggml_tensor* bad = ggml_new_tensor_2d(c(), GGML_TYPE_I8, k / 16, rows);
  Place(bad);
  ggml_tensor* refused = Place(kg::Mxfp8Dequant(c(), wc, bad));
  EXPECT_EQ(FailedCode(kg::RunMxfp8Dequant(launch(), refused)), KernelError::kRejected);
}

TEST_F(Qwen38OpsTest, Nvfp4TableRowsDequantizeExactly) {
  constexpr std::int64_t kValues = 160;
  constexpr std::int64_t kRowBytes = (kValues / 2) + (kValues / 16);
  constexpr std::int64_t kRows = 37;
  std::vector<std::uint8_t> table(static_cast<std::size_t>(kRowBytes * kRows));
  std::mt19937 random(9);  // NOLINT(bugprone-random-generator-seed): reproducible
  const std::vector<std::uint8_t> scale_codes = E4m3Codes(10, table.size());
  for (std::size_t r = 0; r < static_cast<std::size_t>(kRows); ++r) {
    for (std::size_t b = 0; b < static_cast<std::size_t>(kRowBytes); ++b) {
      const bool scale = std::cmp_greater_equal(b, kValues / 2);
      table[(r * kRowBytes) + b] = scale ? (scale_codes[(r * kRowBytes) + b] & 0x7F)
                                         : static_cast<std::uint8_t>(random() & 0xFF);
    }
  }
  const std::vector<std::int32_t> ids = {0, 36, 5, 5, 17, 37};  // the last is outside
  const std::vector<float> global = {0.375f};
  ggml_tensor* t = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I8, kRowBytes, kRows), table);
  ggml_tensor* id = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 6), ids);
  ggml_tensor* g = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 1), global);
  ggml_tensor* out = Place(kg::Nvfp4Rows(c(), t, id, g, kValues));
  ASSERT_TRUE(kg::RunNvfp4Rows(launch(), out).has_value());
  const std::vector<float> got = Download(out);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(kValues); ++j) {
      const float v = got[(i * kValues) + j];
      if (ids[i] >= kRows) {
        EXPECT_TRUE(std::isnan(v)) << j;
        continue;
      }
      const std::uint8_t* row = table.data() + (static_cast<std::size_t>(ids[i]) * kRowBytes);
      const std::uint8_t byte = row[j / 2];
      const std::uint8_t code = (j % 2 == 0) ? (byte & 15) : (byte >> 4);
      const float want = static_cast<float>(E2m1(code)) *
                         static_cast<float>(E4m3(row[(kValues / 2) + (j / 16)])) * global[0];
      ASSERT_EQ(v, want) << "id " << ids[i] << " value " << j;
    }
  }
}

// A GGUF checkpoint's n-gram table (160-value rows of a 32-value block
// type, IQ4_NL in unsloth's): every value GGML's own CPU dequantization
// gives it (ggml-quants.c, the type traits' to_float), bit for bit, at
// ids in and outside the table; and a type or row the lookup does not
// take refused.
TEST_F(Qwen38OpsTest, GgufTableRowsAreGgmlsDequantization) {
  constexpr std::int64_t kValues = 160;
  constexpr std::int64_t kRows = 37;
  const std::vector<std::int32_t> ids = {0, 36, 5, 5, 17, 37, -1};  // the last two are outside
  for (const ggml_type type : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                               GGML_TYPE_Q5_0, GGML_TYPE_Q5_1}) {
    std::mt19937 random(11);  // NOLINT(bugprone-random-generator-seed): reproducible
    std::normal_distribution<float> normal(0.0f, 0.02f);
    std::vector<float> source(static_cast<std::size_t>(kValues * kRows));
    for (float& v : source) {
      v = normal(random);
    }
    const std::size_t row_bytes = ggml_row_size(type, kValues);
    std::vector<std::uint8_t> table(row_bytes * kRows);
    ASSERT_EQ(ggml_quantize_chunk(type, source.data(), table.data(), 0, kRows, kValues, nullptr),
              table.size());
    ggml_tensor* t = Place(ggml_new_tensor_2d(c(), type, kValues, kRows), table);
    ggml_tensor* id =
        Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, static_cast<std::int64_t>(ids.size())), ids);
    ggml_tensor* out = Place(kg::QRows(c(), t, id));
    ASSERT_TRUE(kg::RunQRows(launch(), out).has_value()) << ggml_type_name(type);
    const std::vector<float> got = Download(out);
    std::vector<float> want(static_cast<std::size_t>(kValues));
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const bool inside = ids[i] >= 0 && ids[i] < kRows;
      if (inside) {
        ggml_get_type_traits(type)->to_float(
            table.data() + (static_cast<std::size_t>(ids[i]) * row_bytes), want.data(), kValues);
      }
      for (std::size_t j = 0; j < static_cast<std::size_t>(kValues); ++j) {
        const float v = got[(i * kValues) + j];
        if (!inside) {
          EXPECT_TRUE(std::isnan(v)) << ggml_type_name(type) << " " << j;
          continue;
        }
        ASSERT_EQ(v, want[j]) << ggml_type_name(type) << " id " << ids[i] << " value " << j;
      }
    }
  }
  // A k-quant table (no 32-value blocks), and rows of 48 values: refused.
  ggml_tensor* k_quant = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q4_K, 256, 4));
  ggml_tensor* id = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 2));
  EXPECT_EQ(FailedCode(kg::RunQRows(launch(), Place(kg::QRows(c(), k_quant, id)))),
            KernelError::kRejected);
  ggml_tensor* short_rows = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, 48, 4));
  EXPECT_EQ(FailedCode(kg::RunQRows(launch(), Place(kg::QRows(c(), short_rows, id)))),
            KernelError::kRejected);
}

// ModelOpt NVFP4 [n, k] into GGML's block_nvfp4 rows (the importer's
// repack, rewritten element by element).
std::vector<std::uint8_t> ToGgmlNvfp4(const std::vector<std::uint8_t>& codes,
                                      const std::vector<std::uint8_t>& scales, std::int64_t n,
                                      std::int64_t k) {
  std::vector<std::uint8_t> out;
  for (std::int64_t r = 0; r < n; ++r) {
    for (std::int64_t blk = 0; blk < k / 64; ++blk) {
      for (std::int64_t s = 0; s < 4; ++s) {
        out.push_back(scales[static_cast<std::size_t>((r * k / 16) + (blk * 4) + s)]);
      }
      for (std::int64_t sub = 0; sub < 4; ++sub) {
        const auto element = [&](std::int64_t e) {
          const std::int64_t i = (blk * 64) + (sub * 16) + e;
          const std::uint8_t byte = codes[static_cast<std::size_t>((r * k / 2) + (i / 2))];
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

TEST_F(Qwen38OpsTest, Nvfp4ExpertProductsMatchTheReference) {
  // Qwen3.8's routed experts at small scale: 16 experts, 10 selected per
  // token; gate/up (k 2560, 64 rows out) broadcast one activation row, down
  // (k 640, 2560 rows cut to 64) one row per selected expert.
  constexpr std::int64_t kExperts = 16;
  constexpr std::int64_t kUsed = 10;
  constexpr std::int64_t kOut = 64;
  for (const auto& [k, broadcast] :
       {std::pair{std::int64_t{2560}, true}, std::pair{std::int64_t{640}, false}}) {
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kExperts * kOut * k / 2));
    std::mt19937 random(static_cast<unsigned>(k));
    for (auto& b : codes) {
      b = static_cast<std::uint8_t>(random() & 0xFF);
    }
    // Block scales of moderate size (2^-6 .. 2^-2), as a checkpoint's are.
    std::vector<std::uint8_t> scales(static_cast<std::size_t>(kExperts * kOut * k / 16));
    for (auto& s : scales) {
      s = static_cast<std::uint8_t>(((1 + (random() % 5)) << 3) | (random() % 8));
    }
    const std::vector<std::uint8_t> blocks = ToGgmlNvfp4(codes, scales, kExperts * kOut, k);
    for (const std::int64_t tokens : {1, 5, 40}) {
      const std::string what =
          "nvfp4 k " + std::to_string(k) + " experts x " + std::to_string(tokens);
      // The rows' padding to 512 elements, readable past the last expert.
      const std::size_t pad = ggml_row_size(GGML_TYPE_NVFP4, 512 - (k % 512 == 0 ? 512 : k % 512));
      ggml_tensor* w = Place(ggml_new_tensor_3d(c(), GGML_TYPE_NVFP4, k, kOut, kExperts), blocks,
                             k % 512 == 0 ? 0 : pad);
      if (k % 512 != 0) {
        // Unmarked, the check refuses rows that are not whole 512-element
        // steps.
        ggml_tensor* x0 = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, k, kUsed, 1));
        ggml_tensor* i0 = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, 1),
                                std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        EXPECT_FALSE(kg::CheckMulMatIdQ(Place(ggml_mul_mat_id(c(), w, x0, i0))).has_value());
        kg::MarkRowPaddingReadable(w);
      }
      const std::int64_t rows_in = broadcast ? 1 : kUsed;
      std::vector<float> x(static_cast<std::size_t>(k * rows_in * tokens));
      std::normal_distribution<float> normal(0.0f, 1.0f);
      for (float& v : x) {
        v = normal(random);
      }
      ggml_tensor* input = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, k, rows_in, tokens), x);
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
      const auto ran = *path == QuantMulMatPath::kVector ? kg::MulMatVecQ(launch(), product)
                                                         : kg::MulMatQ(launch(), product);
      ASSERT_TRUE(ran.has_value()) << what << ": " << ran.error().detail;
      std::vector<double> want(static_cast<std::size_t>(kOut * kUsed * tokens));
      for (std::int64_t t = 0; t < tokens; ++t) {
        for (std::int64_t s = 0; s < kUsed; ++s) {
          const std::int64_t e = ids[static_cast<std::size_t>((t * kUsed) + s)];
          const std::int64_t row_in = broadcast ? 0 : s;
          for (std::int64_t r = 0; r < kOut; ++r) {
            const std::int64_t row = (e * kOut) + r;
            double sum = 0.0;
            for (std::int64_t i = 0; i < k; ++i) {
              const std::uint8_t byte = codes[static_cast<std::size_t>((row * k / 2) + (i / 2))];
              const std::uint8_t code = i % 2 == 0 ? (byte & 15) : (byte >> 4);
              const double scale =
                  E4m3(scales[static_cast<std::size_t>((row * k / 16) + (i / 16))]);
              sum += E2m1(code) * scale *
                     x[static_cast<std::size_t>((((t * rows_in) + row_in) * k) + i)];
            }
            want[static_cast<std::size_t>((((t * kUsed) + s) * kOut) + r)] = sum;
          }
        }
      }
      ExpectNmse(Download(product), want,
                 *path == QuantMulMatPath::kTile ? kFp4ActivationNmse : kMulMatNmse, what);
    }
  }
}

TEST_F(Qwen38OpsTest, TensorCoreAttentionAtTwelveQueryHeadsPerKvHead) {
  constexpr std::int64_t d = 256;
  constexpr std::int64_t heads = 24;
  constexpr std::int64_t kv_heads = 2;
  constexpr std::int64_t cells = 256;
  for (const std::int64_t tokens : {1, 7, 64}) {
    const std::string what = "qsa attention x " + std::to_string(tokens);
    std::mt19937 random(static_cast<unsigned>(tokens));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> q(static_cast<std::size_t>(d * tokens * heads));
    for (float& v : q) {
      v = normal(random);
    }
    std::vector<ggml_fp16_t> k(static_cast<std::size_t>(d * cells * kv_heads));
    std::vector<ggml_fp16_t> v(k.size());
    for (std::size_t i = 0; i < k.size(); ++i) {
      k[i] = ggml_fp32_to_fp16(normal(random));
      v[i] = ggml_fp32_to_fp16(normal(random));
    }
    // A causal mask for tokens ending at cell 200.
    std::vector<ggml_fp16_t> mask(static_cast<std::size_t>(cells * tokens));
    for (std::int64_t t = 0; t < tokens; ++t) {
      for (std::int64_t j = 0; j < cells; ++j) {
        const bool visible = j <= 200 - tokens + 1 + t;
        mask[static_cast<std::size_t>((t * cells) + j)] =
            ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
      }
    }
    // Q [d, tokens, heads] (the permuted view llama.cpp passes), K and V
    // [d, cells, kv heads].
    ggml_tensor* qt = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, d, tokens, heads), q);
    ggml_tensor* kt = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, d, cells, kv_heads), k);
    ggml_tensor* vt = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, d, cells, kv_heads), v);
    ggml_tensor* mt = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, cells, tokens, 1, 1), mask);
    const float scale = 1.0f / 16.0f;
    ggml_tensor* out = ggml_flash_attn_ext(c(), qt, kt, vt, mt, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    Place(out);
    ASSERT_TRUE(kg::CheckFlashAttnMma(out).has_value()) << what;
    ASSERT_TRUE(kg::FlashAttnMma(launch(), out).has_value()) << what;
    std::vector<double> want(static_cast<std::size_t>(d * heads * tokens));
    for (std::int64_t h = 0; h < heads; ++h) {
      const std::int64_t kh = h / (heads / kv_heads);
      for (std::int64_t t = 0; t < tokens; ++t) {
        std::vector<double> scores(cells);
        double most = -std::numeric_limits<double>::infinity();
        for (std::int64_t j = 0; j < cells; ++j) {
          const float m = ggml_fp16_to_fp32(mask[static_cast<std::size_t>((t * cells) + j)]);
          double s = 0.0;
          for (std::int64_t i = 0; i < d; ++i) {
            s += q[static_cast<std::size_t>((((h * tokens) + t) * d) + i)] *
                 ggml_fp16_to_fp32(k[static_cast<std::size_t>((((kh * cells) + j) * d) + i)]);
          }
          scores[static_cast<std::size_t>(j)] = (s * scale) + m;
          most = std::max(most, scores[static_cast<std::size_t>(j)]);
        }
        double total = 0.0;
        for (double& s : scores) {
          s = std::exp(s - most);
          total += s;
        }
        for (std::int64_t i = 0; i < d; ++i) {
          double o = 0.0;
          for (std::int64_t j = 0; j < cells; ++j) {
            o += scores[static_cast<std::size_t>(j)] *
                 ggml_fp16_to_fp32(v[static_cast<std::size_t>((((kh * cells) + j) * d) + i)]);
          }
          // The output is [d, heads, tokens].
          want[static_cast<std::size_t>((((t * heads) + h) * d) + i)] = o / total;
        }
      }
    }
    ExpectNmse(Download(out), want, kFlashAttnNmse, what);
    // With sinks at this ratio the check still refuses (RE-030).
    ggml_tensor* with_sinks = ggml_flash_attn_ext(c(), qt, kt, vt, mt, scale, 0.0f, 0.0f);
    ggml_tensor* sinks = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, heads));
    ggml_flash_attn_ext_add_sinks(with_sinks, sinks);
    Place(with_sinks);
    EXPECT_FALSE(kg::CheckFlashAttnMma(with_sinks).has_value()) << what;
  }
}

}  // namespace
