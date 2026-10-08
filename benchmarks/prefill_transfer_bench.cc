// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Family-neutral ds4-study controls: D512 one/eight-query sparse gathers,
// D256 dense/eight-query sparse attention, and separate/paired ordinary
// expert MMQ products.
// Nine alternating samples per arm, two warmups, one invocation per
// sample. A 512 MiB write before the timing events displaces operands from
// L2. Synthetic finite operands; correctness is covered by the operation
// tests, not inferred from these timings. No model or cache precision
// changes. --compact compares shared preparation with and without compact
// expert tiles; --large uses the target expert shape, --down its down
// projection, and --skew concentrates assignments in a few experts.
// Usage: llmp_prefill_transfer_bench [--only TEXT] [--compact]
//        [--large] [--down] [--skew] [--tokens N].

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"

namespace {
namespace kg = llmp::kernels::ggml;
using llmp::base::Bytes;

constexpr std::size_t kScratch = std::size_t{256} << 20U;
constexpr std::size_t kFlush = std::size_t{512} << 20U;

void Check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::println(stderr, "CUDA: {}", cudaGetErrorString(result));
    std::abort();
  }
}

void Check(const std::expected<void, kg::KernelFailure>& result) {
  if (!result) {
    std::println(stderr, "kernel: {}", result.error().detail);
    std::abort();
  }
}

class Fixture {
 public:
  Fixture() {
    execution_ = std::move(llmp::providers::cuda::OpenDeviceExecution(0).value());
    stream_ = execution_->CreateStream().value();
    native_ = static_cast<cudaStream_t>(execution_->Submission(stream_).value().handle);
    auto* scratch = Allocate(kScratch);
    flush_ = Allocate(kFlush);
    launch_ = kg::LaunchContext::Create(
                  0, *execution_, stream_,
                  {.base = reinterpret_cast<std::uintptr_t>(scratch), .size = Bytes(kScratch)})
                  .value();
    Check(cudaEventCreate(&start_));
    Check(cudaEventCreate(&stop_));
  }
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;
  Fixture(Fixture&&) = delete;
  Fixture& operator=(Fixture&&) = delete;
  ~Fixture() {
    Check(cudaStreamSynchronize(native_));
    const auto fence = execution_->Record(stream_).value();
    Check(cudaStreamSynchronize(native_));
    if (execution_->Query(fence).value() != llmp::providers::FenceState::kComplete ||
        !execution_->Release(fence)) {
      std::abort();
    }
    launch_.reset();
    Check(cudaEventDestroy(start_));
    Check(cudaEventDestroy(stop_));
    for (void* memory : allocations_) {
      Check(cudaFree(memory));
    }
    if (!execution_->DestroyStream(stream_)) {
      std::abort();
    }
  }

  void* Allocate(std::size_t bytes) {
    void* memory = nullptr;
    Check(cudaMalloc(&memory, bytes));
    allocations_.push_back(memory);
    return memory;
  }
  template <class T>
  ggml_tensor* Place(ggml_tensor* tensor, const std::vector<T>& values) {
    void* memory = Allocate(ggml_nbytes(tensor));
    kg::TensorArena::Bind(tensor, reinterpret_cast<std::uintptr_t>(memory));
    Check(cudaMemcpy(memory, values.data(), ggml_nbytes(tensor), cudaMemcpyHostToDevice));
    Check(cudaDeviceSynchronize());
    return tensor;
  }
  ggml_tensor* Place(ggml_tensor* tensor) {
    kg::TensorArena::Bind(tensor, reinterpret_cast<std::uintptr_t>(Allocate(ggml_nbytes(tensor))));
    return tensor;
  }
  kg::LaunchContext& launch() { return *launch_; }

  void Measure(std::string_view name, const std::function<void()>& primitive,
               const std::function<void()>& shared) {
    const std::array<std::function<void()>, 2> arms = {primitive, shared};
    std::array<std::vector<float>, 2> samples;
    for (int sample = -2; sample < 9; ++sample) {
      for (int order = 0; order < 2; ++order) {
        const auto arm = static_cast<std::size_t>((order + sample + 2) % 2);
        Check(cudaMemsetAsync(flush_, 0, kFlush, native_));
        Check(cudaEventRecord(start_, native_));
        arms[arm]();
        Check(cudaEventRecord(stop_, native_));
        Check(cudaEventSynchronize(stop_));
        float ms = 0;
        Check(cudaEventElapsedTime(&ms, start_, stop_));
        if (sample >= 0) {
          samples[arm].push_back(ms);
        }
      }
    }
    for (auto& values : samples) {
      std::ranges::sort(values);
    }
    std::println("{},{:.6f},{:.6f},{:.4f},{}", name, samples[0][4], samples[1][4],
                 samples[0][4] / samples[1][4], launch().scratch_peak().value());
  }

 private:
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  cudaStream_t native_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  void* flush_ = nullptr;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<void*> allocations_;
};

std::vector<float> Values(std::size_t size) {
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; ++i) {
    values[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 1000.0f;
  }
  return values;
}

std::vector<ggml_fp16_t> Halves(const std::vector<float>& values) {
  std::vector<ggml_fp16_t> out(values.size());
  std::ranges::transform(values, out.begin(), ggml_fp32_to_fp16);
  return out;
}

void Attention(int head, bool overlap) {
  Fixture f;
  auto arena = kg::TensorArena::Create(16).value();
  auto* c = arena.context();
  const int heads = head == 256 ? 24 : 64;
  const int kv_heads = head == 256 ? 2 : 1;
  constexpr int kRows = 256;
  constexpr int kCells = 8192;
  constexpr int kKept = 256;
  auto* q =
      f.Place(ggml_new_tensor_4d(c, GGML_TYPE_F32, head, kRows, heads, 1),
              Values(static_cast<std::size_t>(head) * kRows * static_cast<std::size_t>(heads)));
  auto* k = f.Place(
      ggml_new_tensor_4d(c, GGML_TYPE_F16, head, kCells, kv_heads, 1),
      Halves(Values(static_cast<std::size_t>(head) * kCells * static_cast<std::size_t>(kv_heads))));
  auto* v = head == 512 ? k
                        : f.Place(ggml_new_tensor_4d(c, GGML_TYPE_F16, head, kCells, kv_heads, 1),
                                  Halves(Values(static_cast<std::size_t>(head) * kCells *
                                                static_cast<std::size_t>(kv_heads))));
  std::vector<float> mask(static_cast<std::size_t>(kCells * kRows),
                          -std::numeric_limits<float>::infinity());
  for (int r = 0; r < kRows; ++r) {
    for (int i = 0; i < kKept; ++i) {
      const int cell = overlap ? r + i : ((r % 8) * kKept) + i;
      mask[(static_cast<std::size_t>(r) * kCells) + static_cast<std::size_t>(cell)] = 0;
    }
  }
  auto* m = f.Place(ggml_new_tensor_4d(c, GGML_TYPE_F16, kCells, kRows, 1, 1), Halves(mask));
  auto* node =
      f.Place(ggml_flash_attn_ext(c, q, k, v, m, 1.0f / std::sqrt(static_cast<float>(head)), 0, 0));
  ggml_flash_attn_ext_set_n_kv_max(node, kKept);
  const auto one =
      head == 256 ? kg::detail::FlashAttnMmaCase256(1) : kg::detail::FlashAttnMmaCase512(1);
  f.Measure(
      std::string("D") + std::to_string(head) + (overlap ? "-overlap" : "-disjoint"),
      [&] {
        if (head == 256) {
          ggml_flash_attn_ext_set_n_kv_max(node, 0);
          Check(kg::FlashAttnMma(f.launch(), node));
          ggml_flash_attn_ext_set_n_kv_max(node, kKept);
        } else {
          Check(f.launch().Run(Bytes(kScratch), [&](auto& ctx) { one(ctx, node); }));
        }
      },
      [&] { Check(kg::FlashAttnMma(f.launch(), node, true)); });
}

void Experts(ggml_type type, bool compact, bool large, bool down, bool skew, int tokens) {
  Fixture f;
  auto arena = kg::TensorArena::Create(16).value();
  auto* c = arena.context();
  const int kInner = down ? 2048 : 4096;
  const int kOut = down ? 4096 : 2048;
  const int kExperts = large ? 256 : 64;
  const int kUsed = large ? 6 : 4;
  const int default_tokens = large ? 4096 : 512;
  const int kTokens = tokens != 0 ? tokens : default_tokens;
  const int slots = down ? kUsed : 1;
  const auto row = Values(static_cast<std::size_t>(kInner));
  const std::vector<float> importance(static_cast<std::size_t>(kInner), 1.0f);
  std::vector<std::uint8_t> block(ggml_row_size(type, kInner));
  const auto written =
      ggml_quantize_chunk(type, row.data(), block.data(), 0, 1, kInner,
                          ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
  if (written != block.size()) {
    std::abort();
  }
  std::vector<std::uint8_t> weights(block.size() * static_cast<std::size_t>(kOut) *
                                    static_cast<std::size_t>(kExperts));
  for (std::size_t i = 0; i < weights.size(); i += block.size()) {
    std::ranges::copy(block, weights.begin() + static_cast<std::ptrdiff_t>(i));
  }
  auto* a = f.Place(ggml_new_tensor_3d(c, type, kInner, kOut, kExperts), weights);
  auto* b = f.Place(ggml_new_tensor_3d(c, type, kInner, kOut, kExperts), weights);
  auto* x = f.Place(ggml_new_tensor_3d(c, GGML_TYPE_F32, kInner, slots, kTokens),
                    Values(static_cast<std::size_t>(kInner) * static_cast<std::size_t>(slots) *
                           static_cast<std::size_t>(kTokens)));
  std::vector<std::int32_t> ids(static_cast<std::size_t>(kUsed) *
                                static_cast<std::size_t>(kTokens));
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<std::int32_t>(i % static_cast<std::size_t>(skew ? kUsed : kExperts));
  }
  auto* route = f.Place(ggml_new_tensor_2d(c, GGML_TYPE_I32, kUsed, kTokens), ids);
  auto* first = f.Place(ggml_mul_mat_id(c, a, x, route));
  auto* second = f.Place(ggml_mul_mat_id(c, b, x, route));
  f.Measure(
      std::string(ggml_type_name(type)) + "-" + std::to_string(kInner) + "x" +
          std::to_string(kOut) + "-E" + std::to_string(kExperts) + "-T" + std::to_string(kTokens) +
          "-used" + std::to_string(kUsed) + (down ? "-slots" : "-broadcast") +
          (skew ? "-skew" : "-uniform"),
      [&] {
        if (compact) {
          Check(kg::MulMatIdQPair(f.launch(), first, second));
        } else {
          Check(kg::MulMatQ(f.launch(), first));
          Check(kg::MulMatQ(f.launch(), second));
        }
      },
      [&] { Check(kg::MulMatIdQPair(f.launch(), first, second, compact)); });
}
}  // namespace

int main(int argc, char** argv) {
  std::string_view only;
  bool compact = false;
  bool large = false;
  bool down = false;
  bool skew = false;
  int tokens = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--only" && i + 1 < argc) {
      only = argv[++i];
    } else if (arg == "--compact") {
      compact = true;
    } else if (arg == "--large") {
      large = true;
    } else if (arg == "--down") {
      down = true;
    } else if (arg == "--skew") {
      skew = true;
    } else if (arg == "--tokens" && i + 1 < argc) {
      const std::string_view value = argv[++i];
      const auto parsed = std::from_chars(value.data(), value.data() + value.size(), tokens);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || tokens < 256 ||
          tokens > 8192) {
        std::println(stderr, "--tokens needs an integer from 256 to 8192");
        return 2;
      }
    } else {
      std::println(stderr,
                   "usage: llmp_prefill_transfer_bench [--only TEXT] [--compact] "
                   "[--large] [--down] [--skew] [--tokens N]");
      return 2;
    }
  }
  std::println("case,primitive_ms,shared_ms,speedup,scratch_peak_bytes");
  if (!compact && (only.empty() || std::string_view("attention").contains(only))) {
    for (const int head : {256, 512}) {
      for (const bool overlap : {true, false}) {
        Attention(head, overlap);
      }
    }
  }
  for (const auto type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K,
                          GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_0}) {
    if (only.empty() || std::string_view(ggml_type_name(type)).contains(only)) {
      Experts(type, compact, large, down, skew, tokens);
    }
  }
  return 0;
}
