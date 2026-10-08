// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A Qwen3.8 speculative verify's commit (kernels/ggml/qwen38_commit.h) on a
// GB10 (label `gpu`), at the model's widths: for every number of kept rows,
// each layer's recurrent state is the one llmp.gated_delta_net.columns
// (the verify's own recurrence) reaches over those rows, bit for bit; each
// convolution history (and the n-gram layer's) holds the last taps of the
// old history followed by the kept rows' inputs, exactly; and the launcher
// refuses extents its kernels cannot run.

#include "kernels/ggml/qwen38_commit.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/tensors.h"
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

constexpr std::int64_t kS = 128;
constexpr std::int64_t kKeyHeads = 16;
constexpr std::int64_t kValueHeads = 48;
constexpr std::int64_t kChannels = (2 * kKeyHeads * kS) + (kValueHeads * kS);
constexpr std::int64_t kTaps = 3;
constexpr std::int64_t kRows = 4;
constexpr std::int64_t kLayers = 2;
constexpr std::int64_t kPleWidth = 1024;
constexpr std::int64_t kPleTaps = 9;

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

class Qwen38CommitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    auto launch = LaunchContext::Create(
        0, *execution_, stream_,
        {.base = reinterpret_cast<std::uintptr_t>(Allocate(1 << 20)), .size = Bytes(1 << 20)});
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

  float* Allocate(std::size_t bytes) {
    void* pointer = nullptr;
    EXPECT_EQ(cudaMalloc(&pointer, std::max<std::size_t>(bytes, 256)), cudaSuccess);
    device_.push_back(pointer);
    return static_cast<float*>(pointer);
  }
  float* Upload(const std::vector<float>& data) {
    float* p = Allocate(data.size() * sizeof(float));
    EXPECT_EQ(cudaMemcpy(p, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return p;
  }
  std::vector<float> Download(const float* p, std::size_t n) {
    Finish();
    std::vector<float> out(n);
    EXPECT_EQ(cudaMemcpy(out.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    return out;
  }
  ggml_context* c() const { return arena_->context(); }

  // The verify's recurrence over the first `keep` rows of one layer's saved
  // inputs, from `state`: the final state gated_delta_net.columns writes.
  std::vector<float> Recurrence(const float* conv, const float* gate, const float* beta,
                                const float* state, std::int64_t keep) {
    const std::size_t head = kS * sizeof(float);
    const std::size_t token = kChannels * sizeof(float);
    const std::size_t chunk = token * static_cast<std::size_t>(keep);
    ggml_tensor* rows = ggml_new_tensor_2d(c(), GGML_TYPE_F32, kChannels, keep);
    TensorArena::Bind(rows, reinterpret_cast<std::uintptr_t>(conv));
    ggml_tensor* q = ggml_view_4d(c(), rows, kS, kKeyHeads, keep, 1, head, token, chunk, 0);
    ggml_tensor* k = ggml_view_4d(c(), rows, kS, kKeyHeads, keep, 1, head, token, chunk,
                                  head * static_cast<std::size_t>(kKeyHeads));
    ggml_tensor* v = ggml_view_4d(c(), rows, kS, kValueHeads, keep, 1, head, token, chunk,
                                  2 * head * static_cast<std::size_t>(kKeyHeads));
    ggml_tensor* g = ggml_new_tensor_4d(c(), GGML_TYPE_F32, 1, kValueHeads, keep, 1);
    TensorArena::Bind(g, reinterpret_cast<std::uintptr_t>(gate));
    ggml_tensor* b = ggml_new_tensor_4d(c(), GGML_TYPE_F32, 1, kValueHeads, keep, 1);
    TensorArena::Bind(b, reinterpret_cast<std::uintptr_t>(beta));
    ggml_tensor* s = ggml_new_tensor_4d(c(), GGML_TYPE_F32, kS, kS, kValueHeads, 1);
    TensorArena::Bind(s, reinterpret_cast<std::uintptr_t>(state));
    ggml_tensor* out = ggml_gated_delta_net(c(), q, k, v, g, b, s, 1);
    TensorArena::Bind(out, reinterpret_cast<std::uintptr_t>(Allocate(ggml_nbytes(out))));
    kg::BindViews(std::vector<ggml_tensor*>{q, k, v, out});
    EXPECT_TRUE(kg::GatedDeltaNetColumnsFits(out));
    auto ran = kg::RunGatedDeltaNetColumns(*launch_, out);
    EXPECT_TRUE(ran.has_value()) << (ran ? "" : ran.error().detail);
    // The final state follows the rows' outputs.
    const auto outputs = static_cast<std::size_t>(kS * kValueHeads * keep);
    auto all = Download(static_cast<const float*>(out->data),
                        static_cast<std::size_t>(ggml_nelements(out)));
    return {all.begin() + static_cast<std::ptrdiff_t>(outputs), all.end()};
  }

  std::unique_ptr<DeviceExecution> execution_;
  StreamId stream_;
  std::vector<void*> device_;
  std::unique_ptr<LaunchContext> launch_;
  std::unique_ptr<TensorArena> arena_;
};

TEST_F(Qwen38CommitTest, TheCommitIsTheVerifysRecurrenceAndHistories) {
  const std::size_t state_n = kS * kS * kValueHeads;
  struct Layer {
    std::vector<float> state, history, conv, qkv, gate, beta;
  };
  std::vector<Layer> host(kLayers);
  for (std::int64_t l = 0; l < kLayers; ++l) {
    Layer& h = host[static_cast<std::size_t>(l)];
    const auto seed = static_cast<std::uint64_t>(100 * (l + 1));
    h.state = Normal(seed, state_n, 0.1f);
    h.history = Normal(seed + 1, kTaps * kChannels);
    h.conv = Normal(seed + 2, kChannels * kRows, 0.1f);
    h.qkv = Normal(seed + 3, kChannels * kRows);
    // A log decay (negative) and beta in (0, 1), as the graph gives them.
    h.gate = Normal(seed + 4, kValueHeads * kRows, 0.5f, -1.0f);
    for (float& x : h.gate) {
      x = -std::abs(x);
    }
    h.beta = Normal(seed + 5, kValueHeads * kRows, 0.2f, 0.5f);
    for (float& x : h.beta) {
      x = std::clamp(x, 0.01f, 0.99f);
    }
  }
  const auto ple_history = Normal(7, kPleTaps * kPleWidth);
  const auto ple_rows = Normal(8, kPleWidth * kRows);
  for (std::int64_t keep = 1; keep <= kRows; ++keep) {
    kg::Qwen38CommitArgs args;
    args.layers = kLayers;
    args.keep = static_cast<int>(keep);
    args.channels = kChannels;
    args.qk_heads = kKeyHeads;
    args.v_heads = kValueHeads;
    args.taps = kTaps;
    std::vector<float*> states(kLayers);
    std::vector<float*> histories(kLayers);
    std::vector<std::vector<float>> want_states(kLayers);
    for (std::int64_t l = 0; l < kLayers; ++l) {
      const Layer& h = host[static_cast<std::size_t>(l)];
      float* conv = Upload(h.conv);
      float* gate = Upload(h.gate);
      float* beta = Upload(h.beta);
      // The verify's own recurrence from an untouched copy of the state.
      want_states[static_cast<std::size_t>(l)] =
          Recurrence(conv, gate, beta, Upload(h.state), keep);
      states[static_cast<std::size_t>(l)] = Upload(h.state);
      histories[static_cast<std::size_t>(l)] = Upload(h.history);
      args.layer[l] = {.state = states[static_cast<std::size_t>(l)],
                       .history = histories[static_cast<std::size_t>(l)],
                       .conv = conv,
                       .qkv = Upload(h.qkv),
                       .gate = gate,
                       .beta = beta};
    }
    float* ple = Upload(ple_history);
    args.ple_history = ple;
    args.ple_rows = Upload(ple_rows);
    args.ple_width = kPleWidth;
    args.ple_taps = kPleTaps;
    auto committed = kg::Qwen38Commit(*launch_, args);
    ASSERT_TRUE(committed.has_value()) << (committed ? "" : committed.error().detail);
    const std::string what = "keep " + std::to_string(keep);
    for (std::int64_t l = 0; l < kLayers; ++l) {
      const Layer& h = host[static_cast<std::size_t>(l)];
      const auto got = Download(states[static_cast<std::size_t>(l)], state_n);
      const auto& want = want_states[static_cast<std::size_t>(l)];
      ASSERT_EQ(got.size(), want.size());
      std::size_t differing = 0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        differing += std::bit_cast<std::uint32_t>(got[i]) != std::bit_cast<std::uint32_t>(want[i]);
      }
      EXPECT_EQ(differing, 0U) << what << ", layer " << l << "'s state";
      // Tap j: the (keep + j)-th of the old taps then the kept rows' inputs.
      const auto history = Download(histories[static_cast<std::size_t>(l)], kTaps * kChannels);
      for (std::int64_t ch = 0; ch < kChannels; ++ch) {
        for (std::int64_t j = 0; j < kTaps; ++j) {
          const std::int64_t at = keep + j;
          const float want_tap =
              at < kTaps ? h.history[static_cast<std::size_t>((ch * kTaps) + at)]
                         : h.qkv[static_cast<std::size_t>(((at - kTaps) * kChannels) + ch)];
          ASSERT_EQ(history[static_cast<std::size_t>((ch * kTaps) + j)], want_tap)
              << what << ", layer " << l << ", channel " << ch << ", tap " << j;
        }
      }
    }
    const auto got_ple = Download(ple, kPleTaps * kPleWidth);
    for (std::int64_t ch = 0; ch < kPleWidth; ++ch) {
      for (std::int64_t j = 0; j < kPleTaps; ++j) {
        const std::int64_t at = keep + j;
        const float want_tap =
            at < kPleTaps ? ple_history[static_cast<std::size_t>((ch * kPleTaps) + at)]
                          : ple_rows[static_cast<std::size_t>(((at - kPleTaps) * kPleWidth) + ch)];
        ASSERT_EQ(got_ple[static_cast<std::size_t>((ch * kPleTaps) + j)], want_tap)
            << what << ", n-gram channel " << ch << ", tap " << j;
      }
    }
  }
}

TEST_F(Qwen38CommitTest, TheLauncherRefusesWhatItsKernelsCannotRun) {
  float* p = Allocate(1 << 20);
  kg::Qwen38CommitArgs ok;
  ok.layers = 1;
  ok.keep = 1;
  ok.channels = kChannels;
  ok.qk_heads = kKeyHeads;
  ok.v_heads = kValueHeads;
  ok.taps = kTaps;
  ok.layer[0] = {.state = p, .history = p, .conv = p, .qkv = p, .gate = p, .beta = p};
  for (const auto& [what, mutate] :
       std::vector<std::pair<std::string, void (*)(kg::Qwen38CommitArgs&)>>{
           {"no rows", [](kg::Qwen38CommitArgs& a) { a.keep = 0; }},
           {"nine rows", [](kg::Qwen38CommitArgs& a) { a.keep = 9; }},
           {"too many layers",
            [](kg::Qwen38CommitArgs& a) { a.layers = kg::kQwen38CommitLayers + 1; }},
           {"channels not the heads'", [](kg::Qwen38CommitArgs& a) { a.channels = 4096; }},
           {"no taps", [](kg::Qwen38CommitArgs& a) { a.taps = 0; }},
           {"a missing state", [](kg::Qwen38CommitArgs& a) { a.layer[0].state = nullptr; }},
           {"a misaligned save",
            [](kg::Qwen38CommitArgs& a) {
              // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address.
              a.layer[0].conv = reinterpret_cast<const float*>(
                  reinterpret_cast<std::uintptr_t>(a.layer[0].conv) + 4);
            }},
           {"an n-gram history of no taps", [](kg::Qwen38CommitArgs& a) {
              a.ple_history = a.layer[0].state;
              a.ple_rows = a.layer[0].conv;
              a.ple_width = 16;
              a.ple_taps = 0;
            }}}) {
    kg::Qwen38CommitArgs args = ok;
    mutate(args);
    auto refused = kg::Qwen38Commit(*launch_, args);
    ASSERT_FALSE(refused.has_value()) << what;
    EXPECT_EQ(refused.error().error, kg::KernelError::kRejected) << what;
  }
}

}  // namespace
