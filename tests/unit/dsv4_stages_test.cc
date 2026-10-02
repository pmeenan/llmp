// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's ds4 prefill stage mechanisms, the fast plan's defaults
// (graph_plan.h and dsv4_graph.h SetDsv4PrefillStages;
// docs/experiments/ds4-prefill-stages), on a GB10 (label `gpu`): each
// against the separate launches it replaces, byte for byte:
// - the HC mix input as F16 rows: alone, with the HC post before it, and
//   with the post also forming the ordered expert reduction and the shared
//   expert's add, which it leaves unwritten;
// - the shared F16 copy of an activation and the cuBLAS products reading it;
// - the Q-head's F16 rows, and the D512 MMA flash attention (plain and wide)
//   and the ds4 HCA core reading them in place of F32 Q;
// - output-A's coalesced weight repack;
// - two dense Q8_0 products sharing one quantization of their activation;
// - the GB10 IQ2 occupancy-two pair's SwiGLU write-back, and its quantizing
//   form with the Q2_K down product that reads it unquantized, at the
//   measured shape;
// and what the planner selects for each, what the checks refuse, and the
// registry's declarations. The model-level selection over whole graphs is
// dsv4_test.cc's.

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
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "model/dsv4.h"
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
namespace md = jitllm::model;

constexpr std::uint64_t kWorkspace = 1ULL << 30;
constexpr std::int64_t kWidth = 4096;
constexpr std::int64_t kStreams = 4;
constexpr std::int64_t kFlat = kWidth * kStreams;
constexpr float kEps = 1e-6f;

std::size_t N(std::int64_t count) { return static_cast<std::size_t>(count); }

std::vector<float> Normal(std::uint64_t seed, std::size_t count, float sigma = 1.0f) {
  std::mt19937_64 random(seed);
  std::normal_distribution<float> normal(0.0f, sigma);
  std::vector<float> values(count);
  for (float& v : values) {
    v = normal(random);
  }
  return values;
}

// Round to nearest, as the launchers' F32-to-F16 conversions round.
std::vector<ggml_fp16_t> Halves(const std::vector<float>& values) {
  std::vector<ggml_fp16_t> halves(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    halves[i] = ggml_fp32_to_fp16(values[i]);
  }
  return halves;
}

// `rows` rows of `k` values of `type` as random blocks, whose F16 scale
// fields (at `scales` within each block) all hold `scale`: every other bit
// pattern is a valid code of these formats.
std::vector<std::uint8_t> RandomBlocks(ggml_type type, std::int64_t k, std::int64_t rows,
                                       std::uint64_t seed,
                                       std::initializer_list<std::size_t> scales, float scale) {
  const std::size_t block = ggml_type_size(type);
  std::vector<std::uint8_t> bytes(ggml_row_size(type, k) * N(rows));
  std::mt19937_64 random(seed);
  for (std::size_t i = 0; i < bytes.size(); i += sizeof(std::uint64_t)) {
    const std::uint64_t word = random();
    std::memcpy(bytes.data() + i, &word, std::min(sizeof(word), bytes.size() - i));
  }
  const ggml_fp16_t half = ggml_fp32_to_fp16(scale);
  for (std::size_t at = 0; at < bytes.size(); at += block) {
    for (const std::size_t offset : scales) {
      std::memcpy(bytes.data() + at + offset, &half, sizeof(half));
    }
  }
  return bytes;
}

template <typename T>
void ExpectSameBytes(const std::vector<T>& got, const std::vector<T>& want,
                     const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  if (std::ranges::equal(std::as_bytes(std::span(got)), std::as_bytes(std::span(want)))) {
    return;
  }
  using Bits = std::array<std::byte, sizeof(T)>;
  std::size_t first = got.size();
  std::size_t differing = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::bit_cast<Bits>(got[i]) != std::bit_cast<Bits>(want[i]) && differing++ == 0) {
      first = i;
    }
  }
  ADD_FAILURE() << what << ": " << differing << " of " << got.size()
                << " values differ, the first at " << first;
}

void Launched(const std::expected<void, KernelFailure>& result, const std::string& what) {
  EXPECT_TRUE(result.has_value()) << what << ": " << (result ? "" : result.error().detail);
}

std::uint64_t Address(const ggml_tensor* t) { return reinterpret_cast<std::uintptr_t>(t->data); }

int ComputeCapability() {
  int major = 0;
  int minor = 0;
  EXPECT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
  EXPECT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
  return (100 * major) + (10 * minor);
}

class Dsv4StagesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(jitllm::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    const Bytes cublas = kg::CublasHandle::UpstreamWorkspace(ComputeCapability());
    auto handle = kg::CublasHandle::Create(0, *execution_, stream_,
                                           {.base = Allocate(cublas.value()), .size = cublas});
    ASSERT_TRUE(handle.has_value()) << (handle ? "" : handle.error().detail);
    cublas_ = std::move(*handle);
    const std::uint64_t workspace = Allocate(kWorkspace);
    auto launch = LaunchContext::Create(
        0, *execution_, stream_, {.base = workspace, .size = Bytes(kWorkspace)}, cublas_.get());
    ASSERT_TRUE(launch.has_value()) << (launch ? "" : launch.error().detail);
    launch_ = std::move(*launch);
    arena_ = std::make_unique<TensorArena>(TensorArena::Create(512).value());
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

  // Overwrites a placed tensor with `byte` (so a stale result cannot pass).
  void Poison(const ggml_tensor* tensor, int byte = 0xFF) {
    Finish();
    EXPECT_EQ(cudaMemset(tensor->data, byte, ggml_nbytes(tensor)), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  template <typename T = float>
  std::vector<T> Download(const ggml_tensor* tensor) {
    Finish();
    std::vector<T> values(ggml_nbytes(tensor) / sizeof(T));
    EXPECT_EQ(cudaMemcpy(values.data(), tensor->data, ggml_nbytes(tensor), cudaMemcpyDeviceToHost),
              cudaSuccess);
    return values;
  }

  // The HC post's inputs for `rows` tokens: the block output, the four
  // streams, the post weights and the combination.
  struct HcInputs {
    ggml_tensor* streams;
    ggml_tensor* post;
    ggml_tensor* comb;
  };
  HcInputs PlaceHcInputs(std::int64_t rows, std::uint64_t seed) {
    return {.streams = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, kStreams, rows),
                             Normal(seed, N(kFlat * rows))),
            .post = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kStreams, rows),
                          Normal(seed + 1, N(kStreams * rows))),
            .comb = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kStreams, kStreams, rows),
                          Normal(seed + 2, N(kStreams * kStreams * rows), 0.5f))};
  }

  // The reference: the HC post, then native rms_norm over its flat view.
  std::pair<std::vector<float>, std::vector<float>> ReferencePostNorm(ggml_tensor* y,
                                                                      const HcInputs& in,
                                                                      std::int64_t rows) {
    auto* post = Place(ggml_dsv4_hc_post(c(), y, in.streams, in.post, in.comb));
    Launched(kg::HcPost(launch(), post), "reference HC post");
    auto* flat = ggml_reshape_2d(c(), post, kFlat, rows);
    flat->data = post->data;
    auto* norm = Place(ggml_rms_norm(c(), flat, kEps));
    Launched(kg::RmsNorm(launch(), norm), "reference flat norm");
    return {Download(post), Download(norm)};
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<kg::CublasHandle> cublas_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

TEST_F(Dsv4StagesTest, TheHcMixInputRowsAreTheNativeNormRoundedToNearest) {
  for (const std::int64_t rows : {1, 64, 2048}) {
    const std::string what = std::format("{} rows", rows);
    auto* flat = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kFlat, rows),
                       Normal(300 + static_cast<std::uint64_t>(rows), N(kFlat * rows), 3.0f));
    auto* norm = Place(ggml_rms_norm(c(), flat, kEps));
    Launched(kg::RmsNorm(launch(), norm), what + " native norm");
    ASSERT_TRUE(kg::Dsv4HcNormF16Fits(flat, kEps)) << what;
    auto* rows16 = Place(kg::Dsv4HcNormF16(c(), flat, kEps));
    ASSERT_TRUE(kg::CheckDsv4HcNormF16(rows16).has_value()) << what;
    Launched(kg::RunDsv4HcNormF16(launch(), rows16), what + " F16 rows");
    ExpectSameBytes(Download<ggml_fp16_t>(rows16), Halves(Download(norm)), what);
  }
  auto* flat = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kFlat, 8));
  EXPECT_FALSE(kg::Dsv4HcNormF16Fits(flat, 0.0f));
  EXPECT_FALSE(kg::Dsv4HcNormF16Fits(flat, std::numeric_limits<float>::quiet_NaN()));
  EXPECT_FALSE(kg::Dsv4HcNormF16Fits(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 8), kEps));
  EXPECT_FALSE(kg::Dsv4HcNormF16Fits(ggml_new_tensor_2d(c(), GGML_TYPE_F16, kFlat, 8), kEps));
  auto* aliased = kg::Dsv4HcNormF16(c(), flat, kEps);
  TensorArena::Bind(aliased, Address(flat));
  EXPECT_FALSE(kg::CheckDsv4HcNormF16(aliased).has_value());
}

TEST_F(Dsv4StagesTest, TheHcPostWritesItsStreamsAndTheNextMixInputRows) {
  for (const std::int64_t rows : {1, 64, 2048}) {
    const std::string what = std::format("{} rows", rows);
    auto* y = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, rows),
                    Normal(310 + static_cast<std::uint64_t>(rows), N(kWidth * rows)));
    const HcInputs in = PlaceHcInputs(rows, 320 + static_cast<std::uint64_t>(rows));
    const auto [want_streams, want_norm] = ReferencePostNorm(y, in, rows);
    auto* post = Place(ggml_dsv4_hc_post(c(), y, in.streams, in.post, in.comb));
    auto* flat = ggml_reshape_2d(c(), post, kFlat, rows);
    flat->data = post->data;
    auto* norm = Place(kg::Dsv4HcNormF16(c(), flat, kEps));
    const std::array<ggml_tensor*, 3> nodes = {post, flat, norm};
    const auto found = kg::Dsv4HcPostNormF16At(nodes, 0);
    ASSERT_TRUE(found.has_value()) << what;
    const auto fused_nodes = found.value_or(kg::Dsv4HcPostNormF16Nodes{});
    EXPECT_EQ(fused_nodes.post, post);
    EXPECT_EQ(fused_nodes.norm, norm);
    EXPECT_FALSE(kg::Dsv4HcPostNormF16At(nodes, 1).has_value());
    ASSERT_TRUE(kg::CheckDsv4HcPostNormF16(post, norm).has_value()) << what;
    Launched(kg::RunDsv4HcPostNormF16(launch(), post, norm), what);
    ExpectSameBytes(Download(post), want_streams, what + " streams");
    ExpectSameBytes(Download<ggml_fp16_t>(norm), Halves(want_norm), what + " mix-input rows");
    // Outputs over an input are refused before any launch.
    auto alias = *norm;
    alias.data = in.streams->data;
    EXPECT_FALSE(kg::CheckDsv4HcPostNormF16(post, &alias).has_value()) << what;
  }
}

TEST_F(Dsv4StagesTest, TheHcPostFormsTheExpertSumItLeavesUnwritten) {
  for (const std::int64_t rows : {9, 2048}) {
    const std::string what = std::format("{} rows", rows);
    const auto seed = static_cast<std::uint64_t>(rows);
    auto* down = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 6, rows),
                       Normal(330 + seed, N(kWidth * 6 * rows)));
    auto* weights = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 1, 6, rows),
                          Normal(340 + seed, N(6 * rows), 0.25f));
    auto* shared = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, rows),
                         Normal(350 + seed, N(kWidth * rows)));
    const HcInputs in = PlaceHcInputs(rows, 360 + seed);
    // The ordinary order: the ordered reduction, the shared add, the post.
    auto* ref_reduce = Place(kg::Dsv4OrderedReduce(c(), down, weights));
    Launched(kg::RunDsv4OrderedReduce(launch(), ref_reduce), what + " reduction");
    auto* ref_add = Place(ggml_add(c(), ref_reduce, shared));
    Launched(kg::Add(launch(), ref_add), what + " shared add");
    const auto [want_streams, want_norm] = ReferencePostNorm(ref_add, in, rows);

    const std::vector<float> sentinel(N(kWidth * rows), 17.0f);
    auto* reduce = Place(kg::Dsv4OrderedReduce(c(), down, weights), sentinel);
    auto* add = Place(ggml_add(c(), reduce, shared), sentinel);
    auto* post = Place(ggml_dsv4_hc_post(c(), add, in.streams, in.post, in.comb));
    auto* flat = ggml_reshape_2d(c(), post, kFlat, rows);
    flat->data = post->data;
    auto* norm = Place(kg::Dsv4HcNormF16(c(), flat, kEps));
    const std::array<ggml_tensor*, 5> nodes = {reduce, add, post, flat, norm};
    std::size_t at = 0;
    const auto found = kg::Dsv4HcPostExpertsAt(nodes, 0, &at);
    ASSERT_TRUE(found.has_value()) << what;
    const auto fused_nodes = found.value_or(kg::Dsv4HcPostExpertsNodes{});
    EXPECT_EQ(at, 1U);
    EXPECT_EQ(fused_nodes.reduce, reduce);
    EXPECT_EQ(fused_nodes.add, add);
    EXPECT_EQ(fused_nodes.post, post);
    EXPECT_EQ(fused_nodes.norm, norm);
    // A sum another node reads must be written: no fusion then.
    auto* reader = ggml_scale(c(), add, 2.0f);
    const std::array<ggml_tensor*, 6> read = {reduce, add, post, flat, norm, reader};
    EXPECT_FALSE(kg::Dsv4HcPostExpertsAt(read, 0, &at).has_value()) << what;
    ASSERT_TRUE(kg::CheckDsv4HcPostExpertsNormF16(reduce, add, post, norm).has_value()) << what;
    Launched(kg::RunDsv4HcPostExpertsNormF16(launch(), reduce, add, post, norm), what);
    ExpectSameBytes(Download(post), want_streams, what + " streams");
    ExpectSameBytes(Download<ggml_fp16_t>(norm), Halves(want_norm), what + " mix-input rows");
    ExpectSameBytes(Download(reduce), sentinel, what + " reduction unwritten");
    ExpectSameBytes(Download(add), sentinel, what + " sum unwritten");
  }
}

TEST_F(Dsv4StagesTest, OneSharedF16CopyFeedsTheSameCublasProducts) {
  for (const std::int64_t rows : {64, 2048}) {
    const std::string what = std::format("{} rows", rows);
    const auto values = Normal(370 + static_cast<std::uint64_t>(rows), N(kWidth * rows));
    auto* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, rows), values);
    auto* w = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, kWidth, 512),
                    Halves(Normal(380, N(kWidth * 512), 0.05f)));
    ASSERT_TRUE(kg::Dsv4F16CopyFits(x)) << what;
    auto* copy = Place(kg::Dsv4F16Copy(c(), x));
    ASSERT_TRUE(kg::CheckDsv4F16Copy(copy).has_value()) << what;
    Launched(kg::RunDsv4F16Copy(launch(), copy), what + " copy");
    ExpectSameBytes(Download<ggml_fp16_t>(copy), Halves(values), what + " copy");
    auto* from_f32 = Place(ggml_mul_mat(c(), w, x));
    Launched(kg::MulMatCublas(launch(), from_f32), what + " F32 activation");
    auto* from_copy = Place(ggml_mul_mat(c(), w, copy));
    // Only the cuBLAS path reads F16 activations.
    EXPECT_FALSE(kg::CheckMulMat(from_copy).has_value()) << what;
    ASSERT_TRUE(kg::CheckMulMatCublasOperands(from_copy).has_value()) << what;
    Launched(kg::MulMatCublas(launch(), from_copy), what + " shared copy");
    ExpectSameBytes(Download(from_copy), Download(from_f32), what + " product");
  }
  EXPECT_FALSE(kg::Dsv4F16CopyFits(ggml_new_tensor_2d(c(), GGML_TYPE_F32, 4095, 64)));
  EXPECT_FALSE(kg::Dsv4F16CopyFits(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kWidth, 64, 2)));
  auto* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 64));
  auto* aliased = kg::Dsv4F16Copy(c(), x);
  TensorArena::Bind(aliased, Address(x));
  EXPECT_FALSE(kg::CheckDsv4F16Copy(aliased).has_value());
}

TEST_F(Dsv4StagesTest, TheQHeadsF16RowsAreItsF32RowsRoundedToNearest) {
  const auto& profile = md::Dsv4Flash();
  for (const std::int64_t rows : {64, 2048}) {
    const auto count = N(512LL * 64 * rows);
    auto* input = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 512, 64, rows), Normal(711, count));
    std::vector<std::int32_t> values(N(rows));
    constexpr std::array<std::int32_t, 6> positions{0, 97, 4095, 131071, 262143, 1048575};
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = positions[i % positions.size()];
    }
    auto* pos = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows), values);
    for (const bool yarn : {false, true}) {
      const std::string what = std::format("{} rows, YaRN {}", rows, yarn);
      const float scale = yarn ? 1.0f / profile.rope_scale : 1.0f;
      const kg::Dsv4QHeadParams params{
          .eps = profile.rms_eps,
          .original_context = yarn ? static_cast<std::int32_t>(profile.yarn_original_context) : 0,
          .base = yarn ? profile.compress_rope_base : profile.rope_base,
          .scale = scale,
          .extension = yarn ? 1.0f : 0.0f,
          .attention = yarn ? 1.0f / (1.0f + (0.1f * logf(1.0f / scale))) : 1.0f,
          .beta_fast = yarn ? profile.yarn_beta_fast : 0,
          .beta_slow = yarn ? profile.yarn_beta_slow : 0};
      auto* f32 = Place(kg::Dsv4QHead(c(), input, pos, params));
      auto* f16 = Place(kg::Dsv4QHead(c(), input, pos, params, GGML_TYPE_F16));
      EXPECT_EQ(f16->type, GGML_TYPE_F16);
      ASSERT_TRUE(kg::CheckDsv4QHead(f16).has_value()) << what;
      Launched(kg::RunDsv4QHead(launch(), f32), what + " F32");
      Launched(kg::RunDsv4QHead(launch(), f16), what + " F16");
      ExpectSameBytes(Download<ggml_fp16_t>(f16), Halves(Download(f32)), what);
    }
  }
}

// The Q-head's F16 rows, as the graph feeds them to attention: [512, 64,
// rows] permuted to [512, rows, 64].
struct Queries {
  ggml_tensor* f32;
  ggml_tensor* f16;
};

TEST_F(Dsv4StagesTest, D512FlashAttentionReadsF16QueriesAsItRoundsF32Ones) {
  struct Case {
    std::int64_t rows;
    std::int64_t cells;
    std::int32_t n_kv_max;  // > 0: at most this many unmasked cells per row
  };
  for (const Case& test : {Case{64, 1024, 0}, Case{70, 4096, 256}}) {
    const std::string what = std::format("{} rows over {} cells", test.rows, test.cells);
    const auto values = Normal(391, N(512LL * 64 * test.rows), 0.5f);
    auto* packed32 = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 512, 64, test.rows), values);
    auto* packed16 =
        Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, 512, 64, test.rows), Halves(values));
    const Queries q{.f32 = ggml_permute(c(), packed32, 0, 2, 1, 3),
                    .f16 = ggml_permute(c(), packed16, 0, 2, 1, 3)};
    auto* kv = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, 512, test.cells, 1, 1),
                     Halves(Normal(392, N(512 * test.cells), 0.5f)));
    std::vector<float> mask(N(test.cells * test.rows), -std::numeric_limits<float>::infinity());
    std::mt19937 random(393);  // NOLINT(bugprone-random-generator-seed): reproducible
    for (std::int64_t r = 0; r < test.rows; ++r) {
      if (test.n_kv_max > 0) {
        std::vector<std::int64_t> chosen(N(test.cells));
        std::ranges::iota(chosen, 0);
        std::shuffle(chosen.begin(), chosen.end(), random);
        for (std::int64_t i = 0; i < test.n_kv_max; ++i) {
          mask[N((r * test.cells) + chosen[N(i)])] = 0.0f;
        }
      } else {
        for (std::int64_t cell = 0; cell < test.cells - test.rows + r + 1; ++cell) {
          mask[N((r * test.cells) + cell)] = 0.0f;
        }
      }
    }
    auto* tm =
        Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, test.cells, test.rows, 1, 1), Halves(mask));
    auto* sinks = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 64), Normal(394, 64));
    const auto make = [&](ggml_tensor* query) {
      auto* node = ggml_flash_attn_ext(c(), query, kv, kv, tm, 0.04419417306780815f, 0, 0);
      ggml_flash_attn_ext_add_sinks(node, sinks);
      if (test.n_kv_max > 0) {
        ggml_flash_attn_ext_set_n_kv_max(node, test.n_kv_max);
      }
      return Place(node);
    };
    for (const bool wide : {false, true}) {
      auto* from_f32 = make(q.f32);
      auto* from_f16 = make(q.f16);
      ASSERT_TRUE(kg::CheckFlashAttnMma(from_f16).has_value()) << what;
      Launched(kg::FlashAttnMma(launch(), from_f32, wide), what + " F32 Q");
      Launched(kg::FlashAttnMma(launch(), from_f16, wide), what + " F16 Q");
      ExpectSameBytes(Download(from_f16), Download(from_f32),
                      what + (wide ? " (wide)" : " (ordinary)"));
    }
  }
  // F16 Q is taken at D512 only.
  auto* q256 =
      ggml_permute(c(), Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, 256, 24, 64)), 0, 2, 1, 3);
  auto* kv256 = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, 256, 512, 2, 1));
  auto* mask256 = Place(ggml_new_tensor_4d(c(), GGML_TYPE_F16, 512, 64, 1, 1));
  auto* d256 = Place(ggml_flash_attn_ext(c(), q256, kv256, kv256, mask256, 0.0625f, 0, 0));
  EXPECT_FALSE(kg::CheckFlashAttnMma(d256).has_value());
}

TEST_F(Dsv4StagesTest, TheDs4HcaCoreReadsF16QueriesAsItRoundsF32Ones) {
  if (ComputeCapability() != 1210) {
    GTEST_SKIP() << "the ds4 HCA core is GB10 only";
  }
  constexpr std::int64_t kRaw = 256;
  constexpr std::int64_t kCompressed = 256;
  for (const auto [first, rows] :
       std::array<std::pair<std::uint32_t, std::uint32_t>, 2>{{{0, 3}, {511, 129}}}) {
    const std::string what = std::format("{} rows at {}", rows, first);
    const auto values = Normal(first + 401, N(rows) * 64 * 512, 0.5f);
    auto* packed32 = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 512, 64, rows), values);
    auto* packed16 = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F16, 512, 64, rows), Halves(values));
    auto* kv = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, 512, kRaw + kCompressed),
                     Halves(Normal(first + 402, (kRaw + kCompressed) * 512, 0.5f)));
    std::vector<float> window(N(kRaw) * rows, -std::numeric_limits<float>::infinity());
    std::vector<std::int32_t> visible(rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto position = static_cast<std::int64_t>(first) + row;
      for (auto p = std::max<std::int64_t>(0, position - 127); p <= position; ++p) {
        window[(static_cast<std::size_t>(row) * kRaw) + static_cast<std::size_t>(p % kRaw)] = 0;
      }
      visible[row] = static_cast<std::int32_t>((position + 1) / 128);
    }
    auto* mask = Place(kg::Dsv4SparseMask(
        c(), Place(ggml_new_tensor_2d(c(), GGML_TYPE_F16, kRaw, rows), Halves(window)), nullptr,
        Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, rows), visible), kRaw, kCompressed));
    Launched(kg::RunDsv4SparseMask(launch(), mask), what + " mask");
    auto* sinks = Place(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 64), Normal(403, 64));
    const auto make = [&](ggml_tensor* packed) {
      auto* q = ggml_permute(c(), packed, 0, 2, 1, 3);
      auto* node = Place(ggml_flash_attn_ext(c(), q, kv, kv, mask, 0.04419417306780815F, 0, 0));
      ggml_flash_attn_ext_add_sinks(node, sinks);
      EXPECT_TRUE(ggml_prec_set_acc(node, GGML_PREC_F32));
      ggml_flash_attn_ext_set_n_kv_max(node, 128 + kCompressed);
      kg::SetFlashAttnSparseAny(node);
      kg::MarkDsv4HcaTokentile(node, first);
      return node;
    };
    auto* from_f32 = make(packed32);
    auto* from_f16 = make(packed16);
    ASSERT_TRUE(kg::CheckDsv4HcaTokentile(from_f16).has_value()) << what;
    const auto scratch = kg::PlanDsv4HcaTokentile(launch(), from_f16);
    ASSERT_TRUE(scratch.has_value()) << what << ": " << (scratch ? "" : scratch.error().detail);
    ASSERT_TRUE(kg::PlanDsv4HcaTokentile(launch(), from_f32).has_value()) << what;
    Launched(kg::Dsv4HcaTokentile(launch(), from_f32), what + " F32 Q");
    launch().ResetScratchPeak();
    Launched(kg::Dsv4HcaTokentile(launch(), from_f16), what + " F16 Q");
    EXPECT_LE(launch().scratch_peak().value(), *scratch);
    ExpectSameBytes(Download(from_f16), Download(from_f32), what);
  }
}

TEST_F(Dsv4StagesTest, OutputAsCoalescedRepackGivesTheSameProduct) {
  if (ComputeCapability() != 1210 || !kg::Dsv4OutASupported(launch())) {
    GTEST_SKIP() << "output-A is measured on GB10 only";
  }
  auto* w = Place(ggml_new_tensor_3d(c(), GGML_TYPE_Q8_0, 4096, 1024, 8),
                  RandomBlocks(GGML_TYPE_Q8_0, 4096, 8192, 411, {0}, 1.0f / 4096));
  auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, 512, 64, 4096),
                  Normal(412, N(512LL * 64 * 4096), 0.5f));
  std::vector<std::int32_t> positions(4096);
  std::ranges::iota(positions, 1000);
  auto* pos = Place(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4096), positions);
  const kg::Dsv4OutAParams params{.base = 10000, .scale = 1, .attention = 1};
  ASSERT_TRUE(kg::Dsv4OutAFits(w, x, pos, params));
  auto* original = Place(kg::Dsv4OutA(c(), w, x, pos, params));
  auto* coalesced = Place(kg::Dsv4OutA(c(), w, x, pos, params));
  Launched(kg::RunDsv4OutA(launch(), original), "output-A");
  Launched(kg::RunDsv4OutA(launch(), coalesced, true), "output-A, coalesced repack");
  ExpectSameBytes(Download(coalesced), Download(original), "output-A");
}

TEST_F(Dsv4StagesTest, DenseQ8PairsShareOneQuantizationExactly) {
  auto* w1 = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, kWidth, 512),
                   RandomBlocks(GGML_TYPE_Q8_0, kWidth, 512, 421, {0}, 1.0f / 2048));
  auto* w2 = Place(ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, kWidth, 1024),
                   RandomBlocks(GGML_TYPE_Q8_0, kWidth, 1024, 422, {0}, 1.0f / 2048));
  for (const std::int64_t rows : {64, 2048}) {
    const std::string what = std::format("{} rows", rows);
    auto* x = Place(ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, rows),
                    Normal(423 + static_cast<std::uint64_t>(rows), N(kWidth * rows)));
    auto* ref1 = Place(ggml_mul_mat(c(), w1, x));
    auto* ref2 = Place(ggml_mul_mat(c(), w2, x));
    Launched(kg::MulMatQ(launch(), ref1), what + " first alone");
    Launched(kg::MulMatQ(launch(), ref2), what + " second alone");
    auto* a = Place(ggml_mul_mat(c(), w1, x));
    auto* b = Place(ggml_mul_mat(c(), w2, x));
    ASSERT_TRUE(kg::MulMatQPairDenseFits(a, b)) << what;
    ASSERT_TRUE(kg::CheckMulMatQPairDense(a, b).has_value()) << what;
    const auto scratch = kg::PlanMulMatQPairDense(launch(), a, b);
    ASSERT_TRUE(scratch.has_value()) << what;
    launch().ResetScratchPeak();
    Launched(kg::MulMatQPairDense(launch(), a, b), what + " pair");
    EXPECT_LE(launch().scratch_peak().value(), *scratch) << what;
    ExpectSameBytes(Download(a), Download(ref1), what + " first");
    ExpectSameBytes(Download(b), Download(ref2), what + " second");
    // The planner pairs them only when asked, as one step at the first's place.
    const std::array<ggml_tensor*, 2> nodes = {a, b};
    auto choices = kg::DeviceChoicesOf(launch());
    const auto separate = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(separate.has_value()) << what;
    ASSERT_EQ(separate->steps.size(), 2U) << what;
    EXPECT_EQ(separate->steps[0].implementation, kg::kMulMatQ);
    choices.dense_pair = true;
    const auto paired = kg::PlanGraph(nodes, false, choices);
    ASSERT_TRUE(paired.has_value()) << what;
    ASSERT_EQ(paired->steps.size(), 1U) << what;
    EXPECT_EQ(paired->steps[0].implementation, kg::kMulMatQPairDense);
    EXPECT_EQ(paired->steps[0].nodes, (std::vector<ggml_tensor*>{a, b}));
    const auto planned = kg::PlanScratch(launch(), *paired);
    ASSERT_TRUE(planned.has_value()) << what;
    EXPECT_EQ(*planned, *scratch) << what;
  }
  auto* narrow = ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 63);
  EXPECT_FALSE(
      kg::MulMatQPairDenseFits(ggml_mul_mat(c(), w1, narrow), ggml_mul_mat(c(), w2, narrow)));
  auto* x = ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 64);
  auto* y = ggml_new_tensor_2d(c(), GGML_TYPE_F32, kWidth, 64);
  EXPECT_FALSE(kg::MulMatQPairDenseFits(ggml_mul_mat(c(), w1, x), ggml_mul_mat(c(), w2, y)));
  auto* q5 = ggml_new_tensor_2d(c(), GGML_TYPE_Q5_K, kWidth, 512);
  EXPECT_FALSE(kg::MulMatQPairDenseFits(ggml_mul_mat(c(), q5, x), ggml_mul_mat(c(), w2, x)));
  auto* odd = ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, kWidth, 96);
  EXPECT_FALSE(kg::MulMatQPairDenseFits(ggml_mul_mat(c(), odd, x), ggml_mul_mat(c(), w2, x)));
}

TEST_F(Dsv4StagesTest, TheIq2PairWritesItsActivationAndTheDownProductsInputExactly) {
  if (ComputeCapability() != 1210) {
    GTEST_SKIP() << "the IQ2 occupancy-two pair is GB10 only";
  }
  // The measured shape: 256 experts, six routes a token, 4,096 tokens.
  constexpr std::int64_t kInner = 4096;
  constexpr std::int64_t kMiddle = 2048;
  constexpr std::int64_t kExperts = 256;
  constexpr std::int64_t kUsed = 6;
  constexpr std::int64_t kTokens = 4096;
  const float limit = md::Dsv4Flash().swiglu_limit;
  auto* gate_w =
      Place(ggml_new_tensor_3d(c(), GGML_TYPE_IQ2_XXS, kInner, kMiddle, kExperts),
            RandomBlocks(GGML_TYPE_IQ2_XXS, kInner, kMiddle * kExperts, 431, {0}, 1.0f / 1024));
  auto* up_w =
      Place(ggml_new_tensor_3d(c(), GGML_TYPE_IQ2_XXS, kInner, kMiddle, kExperts),
            RandomBlocks(GGML_TYPE_IQ2_XXS, kInner, kMiddle * kExperts, 432, {0}, 1.0f / 1024));
  // Q2_K: the F16 d and dmin follow 16 scale and 64 code bytes.
  auto* down_w =
      Place(ggml_new_tensor_3d(c(), GGML_TYPE_Q2_K, kMiddle, kInner, kExperts),
            RandomBlocks(GGML_TYPE_Q2_K, kMiddle, kInner * kExperts, 433, {80, 82}, 1.0f / 2048));
  auto* x = Place(ggml_new_tensor_3d(c(), GGML_TYPE_F32, kInner, 1, kTokens),
                  Normal(434, N(kInner * kTokens)));
  std::vector<std::int32_t> ids;
  std::mt19937 random(435);  // NOLINT(bugprone-random-generator-seed): reproducible
  std::vector<std::int32_t> experts(N(kExperts));
  std::ranges::iota(experts, 0);
  for (std::int64_t t = 0; t < kTokens; ++t) {
    // Six distinct experts a token, one of them popular, as real routing is.
    std::shuffle(experts.begin(), experts.end(), random);
    if (t % 3 == 0) {
      std::iter_swap(experts.begin(), std::ranges::find(experts, 7));
    }
    ids.insert(ids.end(), experts.begin(), experts.begin() + kUsed);
  }
  auto* routes = Place(ggml_new_tensor_2d(c(), GGML_TYPE_I32, kUsed, kTokens), ids);
  auto* gate = Place(ggml_mul_mat_id(c(), gate_w, x, routes));
  auto* up = Place(ggml_mul_mat_id(c(), up_w, x, routes));
  auto* glu = Place(ggml_swiglu_clamp(c(), gate, up, limit));
  auto* down = Place(ggml_mul_mat_id(c(), down_w, glu, routes));
  ASSERT_TRUE(kg::IsMulMatIdQPairIq2Occ2(up, gate));
  ASSERT_TRUE(kg::MulMatIdQPairGluFits(up, gate, glu));
  ASSERT_TRUE(kg::MulMatIdQCompactPrequantFits(down, glu));
  ASSERT_TRUE(kg::CheckMulMatIdQPairGlu(up, gate, glu).has_value());
  ASSERT_TRUE(kg::CheckMulMatIdQCompactPrequant(down).has_value());

  // The ordinary compact plan: the occupancy-two pair, the activation, the
  // down product with its own quantization.
  Launched(kg::MulMatIdQPair(launch(), gate, up, true), "compact pair");
  Launched(kg::SwiGluClamp(launch(), glu), "SwiGLU clamp");
  Launched(kg::MulMatIdQCompact(launch(), down), "compact down");
  const auto want_gate = Download(gate);
  const auto want_glu = Download(glu);
  const auto want_down = Download(down);
  ASSERT_TRUE(std::ranges::all_of(want_glu, [](float v) { return std::isfinite(v); }));
  ASSERT_TRUE(std::ranges::any_of(want_down, [](float v) { return v != 0.0f; }));

  const auto pair_scratch = kg::PlanMulMatIdQPair(launch(), up, gate, true);
  ASSERT_TRUE(pair_scratch.has_value());
  Poison(gate);
  Poison(glu);
  launch().ResetScratchPeak();
  Launched(kg::MulMatIdQPairGlu(launch(), up, gate, glu), "activation write-back");
  EXPECT_LE(launch().scratch_peak().value(), *pair_scratch);
  ExpectSameBytes(Download(gate), want_gate, "gate beside the write-back");
  ExpectSameBytes(Download(glu), want_glu, "activation write-back");

  Poison(glu);
  Poison(down);
  launch().ResetScratchPeak();
  Launched(kg::MulMatIdQPairGluQ8(launch(), up, gate, glu), "quantizing write-back");
  EXPECT_LE(launch().scratch_peak().value(), *pair_scratch);
  const auto down_scratch = kg::PlanMulMatIdQCompact(launch(), down);
  ASSERT_TRUE(down_scratch.has_value());
  launch().ResetScratchPeak();
  Launched(kg::MulMatIdQCompactPrequant(launch(), down), "prequantized down");
  EXPECT_LE(launch().scratch_peak().value(), *down_scratch);
  ExpectSameBytes(Download(down), want_down, "down over the quantizing write-back");

  // Selection: the plain compact plan; with the stage mechanisms, the
  // write-back with D2R taking the Q2_K down product; without D2R, the
  // quantizing write-back and the prequantized down.
  const std::array<ggml_tensor*, 4> nodes = {gate, up, glu, down};
  auto choices = kg::DeviceChoicesOf(launch());
  choices.pair_experts = true;
  choices.compact_experts = true;
  const auto names = [&](const kg::DeviceChoices& device) {
    auto plan = kg::PlanGraph(nodes, false, device);
    EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
    std::vector<std::string_view> out;
    if (plan) {
      for (const auto& step : plan->steps) {
        out.push_back(step.implementation);
      }
      EXPECT_TRUE(kg::PlanScratch(launch(), *plan).has_value());
    }
    return out;
  };
  using Names = std::vector<std::string_view>;
  EXPECT_EQ(names(choices),
            (Names{kg::kMulMatIdQPairCompact, kg::kSwiGluClampName, kg::kMulMatIdQCompact}));
  kg::SetDsv4PrefillStages(choices, true);
  EXPECT_EQ(names(choices), (Names{kg::kMulMatIdQPairGlu, kg::kMulMatIdQ2D2r}));
  choices.d2r_experts = false;
  EXPECT_EQ(names(choices), (Names{kg::kMulMatIdQPairGluQ8, kg::kMulMatIdQCompactPrequant}));
  choices.compact_experts = false;
  EXPECT_EQ(names(choices), (Names{kg::kMulMatIdQPair, kg::kSwiGluClampName, kg::kMulMatIdQ}));

  // An activation over the pair's operands is refused before any launch.
  auto alias = *glu;
  alias.data = x->data;
  EXPECT_FALSE(kg::CheckMulMatIdQPairGlu(up, gate, &alias).has_value());
  // So is any other shape: the same pair over fewer tokens.
  auto* fewer = ggml_view_3d(c(), x, kInner, 1, 2048, x->nb[1], x->nb[2], 0);
  auto* fewer_routes = ggml_view_2d(c(), routes, kUsed, 2048, routes->nb[1], 0);
  auto* small_gate = Place(ggml_mul_mat_id(c(), gate_w, fewer, fewer_routes));
  auto* small_up = Place(ggml_mul_mat_id(c(), up_w, fewer, fewer_routes));
  auto* small_glu = Place(ggml_swiglu_clamp(c(), small_gate, small_up, limit));
  EXPECT_FALSE(kg::MulMatIdQPairGluFits(small_up, small_gate, small_glu));
}

TEST_F(Dsv4StagesTest, TheRegistryDeclaresAndBindsTheStageImplementations) {
  const std::vector<jitllm::execution::Implementation> declared = kg::Implementations();
  const auto registry = jitllm::execution::Registry::Create(declared).value();
  using jitllm::execution::Operation;
  const std::array<std::tuple<std::string_view, Operation, std::size_t>, 9> expected = {{
      {kg::kDsv4F16CopyName, Operation::kConvert, 1},
      {kg::kDsv4HcNormF16Name, Operation::kRmsNorm, 1},
      {kg::kDsv4HcPostNormF16Name, Operation::kHcPost, 2},
      {kg::kDsv4HcPostExpertsNormF16Name, Operation::kHcPost, 4},
      {kg::kDsv4OutAFastPackName, Operation::kMatMul, 1},
      {kg::kMulMatQPairDense, Operation::kMatMul, 2},
      {kg::kMulMatIdQPairGlu, Operation::kMulMatId, 3},
      {kg::kMulMatIdQPairGluQ8, Operation::kMulMatId, 3},
      {kg::kMulMatIdQCompactPrequant, Operation::kMulMatId, 1},
  }};
  for (const auto& [name, operation, arity] : expected) {
    const std::size_t index = registry.Find(name).value_or(registry.size());
    ASSERT_LT(index, registry.size()) << name;
    EXPECT_EQ(registry.at(index).operation, operation) << name;
    const auto kernel = kg::Kernel::Bind(registry.at(index));
    ASSERT_TRUE(kernel.has_value()) << name;
    EXPECT_EQ(kernel->arity(), arity) << name;
  }
}

}  // namespace
