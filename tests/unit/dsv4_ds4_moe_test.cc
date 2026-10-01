// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Mathematical router/activation controls and independently constructed
// uniform expert fields. Cross-tier/repeat/graph controls concern this
// operator only; no model, quality or performance evidence is claimed.
#include "kernels/ggml/dsv4_ds4_moe.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "kernels/ggml/launch.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace pd = jitllm::providers;
constexpr std::size_t kGuardBytes = 256;

class Ds4MoeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto opened = pd::cuda::OpenDeviceExecution(0);
    ASSERT_TRUE(opened.has_value());
    execution_ = std::move(*opened);
    auto stream = execution_->CreateStream();
    ASSERT_TRUE(stream.has_value());
    stream_ = *stream;
    const auto workspace = Allocate(64ULL << 20);
    auto launch = kg::LaunchContext::Create(
        0, *execution_, stream_,
        {.base = workspace.address, .size = jitllm::base::Bytes(workspace.bytes)});
    ASSERT_TRUE(launch.has_value());
    launch_ = std::move(*launch);
  }

  void TearDown() override {
    if (stream_.valid()) Finish();
    if (launch_) {
      launch_.reset();
    }
    if (stream_.valid()) EXPECT_TRUE(execution_->DestroyStream(stream_).has_value());
    for (void* pointer : allocations_) EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }

  // Unknown completion keeps all owners alive until the test process dies.
  // Never destroy the stream or free operands after a timeout/error.
  void Finish() {
    const auto fence = execution_->Record(stream_);
    if (!fence) {
      ADD_FAILURE() << fence.error().detail;
      std::abort();
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
      const auto queried = execution_->Query(*fence);
      if (!queried || std::chrono::steady_clock::now() >= deadline) {
        ADD_FAILURE() << "MoE test could not prove GPU completion";
        std::abort();
      }
      if (*queried == pd::FenceState::kComplete) break;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (!execution_->Release(*fence)) std::abort();
  }

  kg::Ds4CacheBuffer Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes + (2 * kGuardBytes)) != cudaSuccess) std::abort();
    allocations_.push_back(pointer);
    if (cudaMemset(pointer, 0xa5, bytes + (2 * kGuardBytes)) != cudaSuccess) std::abort();
    // Setup uses the default stream; the consumer stream is nonblocking.
    // Prove setup complete, keeping every allocation owned on failure.
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
    return {reinterpret_cast<std::uintptr_t>(pointer) + kGuardBytes, bytes};
  }

  template <typename T>
  void Replace(const kg::Ds4CacheBuffer& buffer, std::span<const T> values) {
    Finish();
    if (buffer.bytes != values.size_bytes() ||
        cudaMemcpy(std::bit_cast<void*>(buffer.address), values.data(), values.size_bytes(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      std::abort();
    if (cudaDeviceSynchronize() != cudaSuccess) std::abort();
  }

  template <typename T>
  kg::Ds4CacheBuffer Upload(std::span<const T> values) {
    const auto buffer = Allocate(values.size_bytes());
    Replace<T>(buffer, values);
    return buffer;
  }

  template <typename T>
  std::vector<T> Download(const kg::Ds4CacheBuffer& buffer) {
    Finish();
    std::vector<T> values(static_cast<std::size_t>(buffer.bytes) / sizeof(T));
    if (cudaMemcpy(values.data(), std::bit_cast<const void*>(buffer.address), buffer.bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      std::abort();
    return values;
  }

  void Guard(const kg::Ds4CacheBuffer& buffer) {
    Finish();
    std::array<std::uint8_t, kGuardBytes> before{};
    std::array<std::uint8_t, kGuardBytes> after{};
    ASSERT_EQ(cudaMemcpy(before.data(), std::bit_cast<const void*>(buffer.address - kGuardBytes),
                         before.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(after.data(), std::bit_cast<const void*>(buffer.address + buffer.bytes),
                         after.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(before, [](auto byte) { return byte == 0xa5; }));
    EXPECT_TRUE(std::ranges::all_of(after, [](auto byte) { return byte == 0xa5; }));
  }

  static void Queued(const std::expected<void, kg::KernelFailure>& result) {
    if (!result) {
      ADD_FAILURE() << result.error().detail;
      std::abort();  // A failed launch does not prove that no work was queued.
    }
  }

  template <typename T>
  void Exact(const std::vector<T>& a, const std::vector<T>& b) {
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(T)), 0);
  }

  kg::LaunchContext& launch() { return *launch_; }

 private:
  std::unique_ptr<pd::DeviceExecution> execution_;
  pd::StreamId stream_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<void*> allocations_;
};

double Softplus(double x) { return std::log1p(std::exp(x)); }

// The uniform fixture has one repeated value per quantization group. Rebuild
// its signed integer and stored scale on the CPU, without inspecting GPU
// quantization outputs. Q8_1 stores a directly divided FP16 scale; wide D4
// stores an F32 reciprocal scale and D2S6 rounds that scale to FP16. Small
// activations can have subnormal FP16 scales, so omitting this rounding is
// materially different from the original consumer's input.
double UniformQuantized(float value, bool half_scale, bool reciprocal_scale) {
  if (value == 0.0F) return 0;
  const float maximum = std::abs(value);
  const float inverse = 127.0F / maximum;
  const float scale = reciprocal_scale ? 1.0F / inverse : maximum / 127.0F;
  const float integer = std::round(reciprocal_scale ? value * inverse : value / scale);
  const float stored = half_scale ? __half2float(__float2half_rn(scale)) : scale;
  return static_cast<double>(integer) * stored;
}

TEST(Ds4MoeQuantReference, SubnormalStoredScaleChangesTheUniformActivation) {
  constexpr float kValue = 0x1p-18F;
  const auto expected = 127.0 * 0x1p-24;
  EXPECT_EQ(UniformQuantized(kValue, true, false), expected);
  EXPECT_EQ(UniformQuantized(kValue, true, true), expected);
  EXPECT_NE(UniformQuantized(kValue, true, true), kValue);
  EXPECT_NEAR(UniformQuantized(kValue, false, true), kValue, 1e-12);
}

TEST_F(Ds4MoeTest, RouterBiasSelectsOnlyAndHashOrderAndFloorMatchIndependentMath) {
  constexpr std::uint32_t kRows = 7;
  std::vector<float> values(static_cast<std::size_t>(kRows) * 256);
  for (std::size_t t = 0; t < kRows; ++t)
    for (std::size_t e = 0; e < 256; ++e)
      values[(t * 256) + e] = t == 6 ? -60.0F : static_cast<float>(static_cast<int>(e % 43) - 21);
  const auto input = Upload<float>(values);
  const auto selected = Allocate(static_cast<std::size_t>(kRows) * 6 * 4);
  const auto weights = Allocate(static_cast<std::size_t>(kRows) * 6 * 4);
  const auto probabilities = Allocate(static_cast<std::size_t>(kRows) * 256 * 4);
  kg::Ds4Router r{.logits = input,
                  .bias = {},
                  .hash = {},
                  .tokens = {},
                  .selected = selected,
                  .weights = weights,
                  .probabilities = probabilities,
                  .rows = kRows};
  Queued(kg::RunDs4Router(launch(), r));
  const auto first_ids = Download<std::int32_t>(selected);
  const auto first_weights = Download<float>(weights);
  const auto p = Download<float>(probabilities);
  for (std::size_t t = 0; t < kRows; ++t) {
    std::array<std::uint32_t, 256> order{};
    std::ranges::iota(order, 0U);
    std::ranges::stable_sort(
        order, [&](auto a, auto b) { return values[(t * 256) + a] > values[(t * 256) + b]; });
    double sum = 0;
    for (std::size_t s = 0; s < 6; ++s) sum += std::sqrt(Softplus(values[(t * 256) + order[s]]));
    for (std::size_t s = 0; s < 6; ++s) {
      EXPECT_EQ(first_ids[(t * 6) + s], static_cast<std::int32_t>(order[s]));
      const auto expected =
          1.5 * std::sqrt(Softplus(values[(t * 256) + order[s]])) / std::max(sum, 0x1p-14);
      EXPECT_NEAR(first_weights[(t * 6) + s], expected, std::max(1e-12, expected * 3e-5));
    }
    for (std::size_t e = 0; e < 256; ++e)
      EXPECT_NEAR(p[(t * 256) + e], std::sqrt(Softplus(values[(t * 256) + e])),
                  std::max(1e-12, std::sqrt(Softplus(values[(t * 256) + e])) * 3e-5));
  }
  for (auto tier : {kg::Ds4RouterSelect::kParallel, kg::Ds4RouterSelect::kScalar}) {
    r.select = tier;
    Queued(kg::RunDs4Router(launch(), r));
    Exact(Download<std::int32_t>(selected), first_ids);
    Exact(Download<float>(weights), first_weights);
  }
  std::ranges::fill(values, 0.0F);
  Replace<float>(input, values);
  std::vector<float> bias(256, 0);
  for (std::size_t e = 100; e < 106; ++e) bias[e] = 1;
  r.bias = Upload<float>(bias);
  r.select = kg::Ds4RouterSelect::kWarp;
  Queued(kg::RunDs4Router(launch(), r));
  const auto biased = Download<std::int32_t>(selected);
  for (std::size_t t = 0; t < kRows; ++t)
    for (std::size_t s = 0; s < 6; ++s)
      EXPECT_EQ(biased[(t * 6) + s], 100 + static_cast<std::int32_t>(s));
  for (auto w : Download<float>(weights)) EXPECT_NEAR(w, 0.25F, 1e-7F);
  const std::vector<std::int32_t> hash{8, 2, 255, 99, 7, 6, 3, 4, 5, 6, 7, 8};
  const std::vector<std::int32_t> tokens{-1, 0, 1, 2, 1, 0, 999};
  r.bias = {};
  r.hash = Upload<std::int32_t>(hash);
  r.tokens = Upload<std::int32_t>(tokens);
  r.hash_rows = 2;
  Queued(kg::RunDs4Router(launch(), r));
  const auto hashed = Download<std::int32_t>(selected);
  for (std::size_t t = 0; t < kRows; ++t)
    for (std::size_t s = 0; s < 6; ++s)
      EXPECT_EQ(hashed[(t * 6) + s], hash[(tokens[t] == 1 ? 6U : 0U) + s]);
  Exact(Download<float>(input), values);
  Guard(input);
  Guard(selected);
  Guard(weights);
  Guard(probabilities);
}

TEST_F(Ds4MoeTest, SharedClampAndSixSumKeepDistinctNonfiniteContracts) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<float> gate{-20, -1, 0, 1, 11, inf, nan};
  const std::vector<float> up{11, -11, 2, -2, -11, 5, nan};
  const auto g = Upload<float>(gate);
  const auto u = Upload<float>(up);
  const auto output = Allocate(gate.size() * 4);
  Queued(kg::RunDs4SharedSwiglu(launch(), {g, u, output, 1, 7}));
  const auto got = Download<float>(output);
  for (std::size_t i = 0; i < gate.size(); ++i) {
    const double a = std::fmin(static_cast<double>(gate[i]), 10.0);
    const double b = std::fmin(std::fmax(static_cast<double>(up[i]), -10.0), 10.0);
    const double expected = a / (1 + std::exp(-a)) * b;
    EXPECT_NEAR(got[i], expected, std::max(1e-7, std::abs(expected) * 3e-5));
  }
  const std::vector<float> slots{1, nan, 1, 4, 2, inf, -2, -1, 3, 5, -3, -inf};
  const auto src = Upload<float>(slots);
  const auto sum = Allocate(2U * sizeof(float));
  Queued(kg::RunDs4MoeSum(launch(), {src, sum, 1, 2}));
  const auto actual = Download<float>(sum);
  EXPECT_EQ(actual[0], 2);
  EXPECT_EQ(actual[1], 8);
  Guard(g);
  Guard(u);
  Guard(output);
  Guard(src);
  Guard(sum);
}

TEST_F(Ds4MoeTest, CooperativeRouterAndCapturedSelectReadCurrentOperands) {
  constexpr std::uint32_t kRows = 3;
  std::vector<float> x(static_cast<std::size_t>(kRows) * 4096, 0x1p-12F);
  std::vector<std::uint16_t> w(std::size_t{256} * 4096);
  for (std::size_t e = 0; e < 256; ++e) {
    const auto h = __float2half_rn(static_cast<float>(1 + (e % 4)) * 0.125F);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    std::fill_n(w.begin() + (static_cast<std::ptrdiff_t>(e) * 4096), 4096, bits);
  }
  kg::Ds4Router r{.logits = Allocate(static_cast<std::size_t>(kRows) * 256 * 4),
                  .bias = {},
                  .hash = {},
                  .tokens = {},
                  .selected = Allocate(static_cast<std::size_t>(kRows) * 6 * 4),
                  .weights = Allocate(static_cast<std::size_t>(kRows) * 6 * 4),
                  .probabilities = Allocate(static_cast<std::size_t>(kRows) * 256 * 4),
                  .rows = kRows};
  const kg::Ds4RouterCooperative coop{r, Upload<float>(x), Upload<std::uint16_t>(w),
                                      Allocate(static_cast<std::size_t>(kRows) * 256 * 8 * 4)};
  Queued(kg::RunDs4RouterCooperative(launch(), coop));
  const auto logits = Download<float>(r.logits);
  for (std::size_t t = 0; t < kRows; ++t)
    for (std::size_t e = 0; e < 256; ++e) EXPECT_EQ(logits[(t * 256) + e], (1 + (e % 4)) * 0.125F);
  const auto ids = Download<std::int32_t>(r.selected);
  const auto route = Download<float>(r.weights);
  Queued(kg::RunDs4Router(launch(), r));
  Exact(Download<std::int32_t>(r.selected), ids);
  Exact(Download<float>(r.weights), route);
  const auto before = Download<std::int32_t>(r.selected);
  const auto graph = launch().Capture([&](auto& c) { return kg::RunDs4Router(c, r); });
  if (!graph) {
    ADD_FAILURE() << graph.error().detail;
    std::abort();
  }
  Exact(Download<std::int32_t>(r.selected), before);  // capture submits no operand work
  std::vector<float> current(static_cast<std::size_t>(kRows) * 256, 0);
  for (std::size_t t = 0; t < kRows; ++t) current[(t * 256) + 155] = 7;
  Replace<float>(r.logits, current);
  Queued(launch().Launch(*graph));
  const auto captured = Download<std::int32_t>(r.selected);
  Queued(kg::RunDs4Router(launch(), r));
  Exact(Download<std::int32_t>(r.selected), captured);
  EXPECT_EQ(captured.front(), 155);
  Finish();
  Guard(coop.input);
  Guard(coop.projection);
  Guard(coop.partials);
}

// IQ2 grid index0/sign0/local-scale1 represents a uniform positive half
// weight; Q2 codes1/scales1/dmin0 likewise. Scale fields vary by expert
// and output row, making assignment/order bugs observable without copying
// the GPU tile's dequantization implementation.
std::vector<std::uint8_t> IqWeights(std::uint32_t input, std::uint32_t middle, bool up) {
  const auto blocks = static_cast<std::size_t>(256) * middle * (input / 256);
  const auto codes = (((blocks * 2) + 63) / 64) * 64;
  std::vector<std::uint8_t> out(codes + (blocks * 64), 0);
  for (std::size_t e = 0; e < 256; ++e) {
    const auto bits =
        static_cast<std::uint16_t>(0x1400U + (((e % (up ? 2U : 3U)) + (up ? 1U : 0U)) * 0x400U));
    for (std::uint32_t m = 0; m < middle; ++m)
      for (std::uint32_t b = 0; b < input / 256; ++b) {
        const auto block = (((e * middle) + m) * (input / 256)) + b;
        std::memcpy(out.data() + (block * 2), &bits, 2);
      }
  }
  return out;
}
std::vector<std::uint8_t> DownWeights(std::uint32_t middle, std::uint32_t output) {
  const auto pairs = static_cast<std::size_t>(256) * (output / 2) * (middle / 256);
  const auto scales = (((pairs * 8) + 63) / 64) * 64;
  const auto codes = scales + ((((pairs * 32) + 63) / 64) * 64);
  std::vector<std::uint8_t> out(codes + (pairs * 128), 0);
  std::fill(out.begin() + static_cast<std::ptrdiff_t>(scales),
            out.begin() + static_cast<std::ptrdiff_t>(codes), 0x11);
  std::fill(out.begin() + static_cast<std::ptrdiff_t>(codes), out.end(), 0x55);
  for (std::size_t e = 0; e < 256; ++e)
    for (std::size_t r = 0; r < output; ++r)
      for (std::uint32_t b = 0; b < middle / 256; ++b) {
        const auto pair = (((e * (output / 2)) + (r / 2)) * (middle / 256)) + b;
        const auto bits = static_cast<std::uint16_t>(0x1000U + (((e % 2) + (r % 2)) * 0x400U));
        std::memcpy(out.data() + (pair * 8) + ((r % 2) * 4), &bits, 2);
      }
  return out;
}

class Ds4MoeChainTest : public Ds4MoeTest {
 protected:
  kg::Ds4Moe Make(std::uint32_t rows, kg::Ds4MoeTier tier, std::uint32_t stride) {
    kg::Ds4Moe d;
    d.shape = {rows, 1024, 256, 130};
    d.tier = tier;
    d.selected_stride = stride;
    d.weight_stride = stride;
    const auto p = kg::Ds4MoeLayoutOf(d.shape, tier);
    if (!p) std::abort();
    std::vector<float> input(static_cast<std::size_t>(rows) * 1024);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(rows) * stride, -999);
    std::vector<float> weights(static_cast<std::size_t>(rows) * stride, -999);
    for (std::size_t t = 0; t < rows; ++t) {
      std::fill_n(input.begin() + (static_cast<std::ptrdiff_t>(t) * 1024), 1024,
                  std::ldexp(static_cast<float>(1 + (t % 3)), -8));
      for (std::size_t s = 0; s < 6; ++s) {
        ids[(t * stride) + s] = static_cast<std::int32_t>(((t % 11) * 7) + s);
        weights[(t * stride) + s] = static_cast<float>(s + 1) / 14.0F;
      }
    }
    if (!kg::CheckDs4MoeIds(ids, rows, stride)) std::abort();
    d.input = Upload<float>(input);
    d.gate_weights = Upload<std::uint8_t>(IqWeights(1024, 256, false));
    d.up_weights = Upload<std::uint8_t>(IqWeights(1024, 256, true));
    d.down_weights = Upload<std::uint8_t>(DownWeights(256, 130));
    d.selected = Upload<std::int32_t>(ids);
    d.weights = Upload<float>(weights);
    if (stride != 6) {
      d.compact_ids = Allocate(static_cast<std::size_t>(rows) * 6 * 4);
      d.compact_weights = Allocate(static_cast<std::size_t>(rows) * 6 * 4);
    }
    if (tier != kg::Ds4MoeTier::kVector) {
      d.ids_source = Allocate(static_cast<std::size_t>(rows) * 6 * 4);
      d.ids_destination = Allocate(static_cast<std::size_t>(rows) * 6 * 4);
      d.expert_bounds = Allocate(257U * sizeof(std::int32_t));
    }
    if (p->work_bytes != 0) d.work = Allocate(p->work_bytes);
    d.input_quant = Allocate(p->input_quant_bytes);
    d.down_quant = Allocate(p->down_quant_bytes);
    if (tier == kg::Ds4MoeTier::kMaterialized || tier == kg::Ds4MoeTier::kClassic) {
      d.gate = Allocate(p->middle_bytes);
      d.up = Allocate(p->middle_bytes);
    }
    if (tier != kg::Ds4MoeTier::kDirect) d.middle = Allocate(p->middle_bytes);
    d.down = Allocate(p->down_bytes);
    d.sum = Allocate(static_cast<std::size_t>(rows) * 130 * 4);
    return d;
  }
  void CheckPhysical(const kg::Ds4Moe& d) {
    for (auto b : {d.input, d.gate_weights, d.up_weights, d.down_weights, d.selected, d.weights,
                   d.input_quant, d.down_quant, d.down, d.sum})
      Guard(b);
    const auto out = Download<float>(d.sum);
    EXPECT_TRUE(std::ranges::all_of(out, [](float x) { return std::isfinite(x) && x > 0; }));
    const auto ids = Download<std::int32_t>(d.selected);
    const auto weights = Download<float>(d.weights);
    const auto input = Download<float>(d.input);
    const bool vector = d.tier == kg::Ds4MoeTier::kVector;
    for (std::size_t t = 0; t < d.shape.rows; ++t) {
      const double x = UniformQuantized(input[t * 1024], vector, !vector);
      for (std::uint32_t row = 0; row < 130; ++row) {
        double expected = 0;
        for (std::size_t s = 0; s < 6; ++s) {
          const auto e = static_cast<std::uint32_t>(ids[(t * d.selected_stride) + s]);
          const float gate = std::min(
              static_cast<float>(1024 * x * std::ldexp(1.0, -10 + static_cast<int>(e % 3))), 10.0F);
          const float up = std::clamp(
              static_cast<float>(1024 * x * std::ldexp(1.0, -9 + static_cast<int>(e % 2))), -10.0F,
              10.0F);
          const float middle =
              (gate / (1.0F + std::exp(-gate))) * up * weights[(t * d.weight_stride) + s];
          const double effective = UniformQuantized(middle, true, !vector);
          expected +=
              256 * effective * std::ldexp(1.0, -11 + static_cast<int>((e % 2) + (row % 2)));
        }
        EXPECT_NEAR(out[(t * 130) + row], expected, std::max(1e-7, expected * 0.005));
      }
    }
    // Expert and output-row scale changes must survive both assignments and sum.
    EXPECT_NE(out[0], out[1]);
    if (d.shape.rows > 1) EXPECT_NE(out[0], out[130]);
  }
};

TEST_F(Ds4MoeChainTest, WideRaggedDirectMaterializedAndClassicUseTheSameAuthoritativeMaps) {
  auto direct = Make(173, kg::Ds4MoeTier::kDirect, 256);
  auto materialized = Make(173, kg::Ds4MoeTier::kMaterialized, 256);
  auto classic = Make(173, kg::Ds4MoeTier::kClassic, 256);
  Queued(kg::RunDs4Moe(launch(), direct));
  const auto first = Download<float>(direct.sum);
  Queued(kg::RunDs4Moe(launch(), materialized));
  Exact(Download<float>(materialized.sum), first);
  Queued(kg::RunDs4Moe(launch(), classic));
  Exact(Download<float>(classic.sum), first);
  const auto ids = Download<std::int32_t>(direct.selected);
  const auto weights = Download<float>(direct.weights);
  // Original expert map is expert-major, token ascending, slot preserved.
  const auto bounds = Download<std::int32_t>(direct.expert_bounds);
  const auto source = Download<std::int32_t>(direct.ids_source);
  const auto destination = Download<std::int32_t>(direct.ids_destination);
  std::size_t cursor = 0;
  for (std::int32_t e = 0; e < 256; ++e) {
    EXPECT_EQ(bounds[static_cast<std::size_t>(e)], static_cast<std::int32_t>(cursor));
    for (std::size_t t = 0; t < 173; ++t)
      for (std::size_t s = 0; s < 6; ++s)
        if (ids[(t * 256) + s] == e) {
          EXPECT_EQ(source[cursor], static_cast<std::int32_t>(t));
          EXPECT_EQ(destination[cursor], static_cast<std::int32_t>((t * 6) + s));
          ++cursor;
        }
  }
  EXPECT_EQ(cursor, 173U * 6U);
  EXPECT_EQ(bounds.back(), static_cast<std::int32_t>(cursor));
  std::vector<std::uint8_t> poisoned(static_cast<std::size_t>(direct.input_quant.bytes), 0x5a);
  Replace<std::uint8_t>(direct.input_quant, poisoned);
  Queued(kg::RunDs4Moe(launch(), direct));
  Exact(Download<float>(direct.sum), first);
  Exact(Download<std::int32_t>(direct.selected), ids);
  Exact(Download<float>(direct.weights), weights);
  CheckPhysical(direct);
  CheckPhysical(materialized);
  CheckPhysical(classic);
}

TEST_F(Ds4MoeChainTest, CanonicalVectorContinuationRepeatsPacksAndReplaysCurrentAssignments) {
  for (auto rows : {1U, 2U, 8U, 16U}) {
    auto d = Make(rows, kg::Ds4MoeTier::kVector, 256);
    Queued(kg::RunDs4Moe(launch(), d));
    const auto first = Download<float>(d.sum);
    Queued(kg::RunDs4Moe(launch(), d));
    Exact(Download<float>(d.sum), first);
    const auto graph = launch().Capture([&](auto& c) { return kg::RunDs4Moe(c, d); });
    if (!graph) {
      ADD_FAILURE() << graph.error().detail;
      std::abort();
    }
    Exact(Download<float>(d.sum), first);
    auto ids = Download<std::int32_t>(d.selected);
    for (std::size_t t = 0; t < rows; ++t)
      for (std::size_t s = 0; s < 6; ++s) ids[(t * 256) + s] += 101;
    ASSERT_TRUE(kg::CheckDs4MoeIds(ids, rows, 256));
    Replace<std::int32_t>(d.selected, ids);
    Queued(launch().Launch(*graph));
    const auto captured = Download<float>(d.sum);
    Queued(kg::RunDs4Moe(launch(), d));
    Exact(Download<float>(d.sum), captured);
    EXPECT_NE(first, captured);
    CheckPhysical(d);
    Finish();
  }
}

TEST_F(Ds4MoeChainTest, CurrentSidecarReuseIsReadOnlyAndStaleIdentityRefusesBeforeSubmission) {
  for (auto tier : {kg::Ds4MoeTier::kVector, kg::Ds4MoeTier::kDirect}) {
    auto d = Make(tier == kg::Ds4MoeTier::kVector ? 3U : 173U, tier, 6);
    Queued(kg::RunDs4Moe(launch(), d));
    const auto first = Download<float>(d.sum);
    const auto input_quant = Download<std::uint8_t>(d.input_quant);
    d.generation = 17;
    d.producer = {
        .storage = d.input_quant,
        .source_address = d.input.address,
        .generation = d.generation,
        .rows = d.shape.rows,
        .width = d.shape.input,
        .kind = tier == kg::Ds4MoeTier::kVector ? kg::Ds4MoeQuant::kQ81 : kg::Ds4MoeQuant::kD4};
    Queued(kg::RunDs4Moe(launch(), d));
    Exact(Download<float>(d.sum), first);
    Exact(Download<std::uint8_t>(d.input_quant), input_quant);
    auto stale = d;
    ++stale.generation;
    const auto refused = kg::RunDs4Moe(launch(), stale);
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().error, kg::KernelError::kRejected);
    EXPECT_FALSE(launch().faulted());
    Exact(Download<float>(d.sum), first);
    stale = d;
    stale.down.address = stale.selected.address;
    const auto alias = kg::RunDs4Moe(launch(), stale);
    ASSERT_FALSE(alias);
    EXPECT_EQ(alias.error().error, kg::KernelError::kRejected);
    Queued(kg::RunDs4Moe(launch(), d));
    Exact(Download<float>(d.sum), first);
    CheckPhysical(d);
  }
}

TEST_F(Ds4MoeChainTest, WideCapturedMapsReadCurrentIdsWithoutRunningDuringCapture) {
  auto d = Make(173, kg::Ds4MoeTier::kDirect, 256);
  Queued(kg::RunDs4Moe(launch(), d));
  const auto first = Download<float>(d.sum);
  const auto graph = launch().Capture([&](auto& c) { return kg::RunDs4Moe(c, d); });
  if (!graph) {
    ADD_FAILURE() << graph.error().detail;
    std::abort();
  }
  Exact(Download<float>(d.sum), first);
  auto ids = Download<std::int32_t>(d.selected);
  for (std::size_t t = 0; t < d.shape.rows; ++t)
    for (std::size_t s = 0; s < 6; ++s) ids[(t * 256) + s] += 101;
  ASSERT_TRUE(kg::CheckDs4MoeIds(ids, d.shape.rows, 256));
  Replace<std::int32_t>(d.selected, ids);
  Queued(launch().Launch(*graph));
  const auto captured = Download<float>(d.sum);
  Queued(kg::RunDs4Moe(launch(), d));
  Exact(Download<float>(d.sum), captured);
  EXPECT_NE(first, captured);
  CheckPhysical(d);
  Finish();
}

}  // namespace
