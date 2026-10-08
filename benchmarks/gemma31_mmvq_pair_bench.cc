// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Private exact-shape synthetic operator control; no checkpoint/model inference.
// Ordinary two-column products versus one original Q8 prep/two original products.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <memory>
#include <optional>
#include <print>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "gemma31_mmvq_pair.h"
#include "ggml.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_ext.h"
#include "providers/cuda/cuda_device_execution.h"

namespace kg = llmp::kernels::ggml;
namespace experiment = llmp::benchmarks::gemma31_mmvq;
namespace {
constexpr std::int64_t kWidth = 5376, kFfn = 21504, kColumns = 2;
constexpr std::size_t kWorkspace = 48384, kDeviceBudget = 256U << 20U;
// Retained packed weights plus one full readback, row bank and control vectors.
constexpr std::size_t kHostPayloadAllowance = 384U << 20U;
constexpr int kPaidChains = 128;
bool Check(bool valid, const char* what) {
  if (!valid) std::println(stderr, "OPERATOR_REFUSAL {}", what);
  return valid;
}
class Screen {
 public:
  ~Screen() {
    // A failed completion is fail-stop; never free backing under unknown work.
    if (cudaDeviceSynchronize() != cudaSuccess) std::_Exit(2);
    launch_.reset();
    if (execution_ && stream_.valid() && (!Complete() || !execution_->DestroyStream(stream_)))
      std::_Exit(2);
    for (auto* pointer : allocations_)
      if (cudaFree(pointer) != cudaSuccess) std::_Exit(2);
  }
  bool Setup() {
    auto opened = llmp::providers::cuda::OpenDeviceExecution(0);
    if (!Check(bool(opened), "device")) return false;
    execution_ = std::move(*opened);
    auto made = execution_->CreateStream();
    if (!Check(bool(made), "stream")) return false;
    stream_ = *made;
    auto submission = execution_->Submission(stream_);
    if (!Check(bool(submission), "native stream")) return false;
    native_ = reinterpret_cast<cudaStream_t>(submission->handle);
    auto created = kg::TensorArena::Create(64);
    if (!Check(bool(created), "descriptor arena")) return false;
    arena_.emplace(std::move(*created));
    workspace_ = Allocate(kWorkspace);
    auto context = kg::LaunchContext::Create(0, *execution_, stream_,
                                             {.base = reinterpret_cast<std::uintptr_t>(workspace_),
                                              .size = llmp::base::Bytes(kWorkspace)});
    if (!Check(workspace_ != nullptr && bool(context), "funded workspace")) return false;
    launch_ = std::move(*context);
    auto* c = arena_->context();
    x_ = Place(ggml_new_tensor_2d(c, GGML_TYPE_F32, kWidth, kColumns));
    wg_ = Place(ggml_new_tensor_2d(c, GGML_TYPE_Q4_K, kWidth, kFfn));
    wu_ = Place(ggml_new_tensor_2d(c, GGML_TYPE_Q4_K, kWidth, kFfn));
    wd_ = Place(ggml_new_tensor_2d(c, GGML_TYPE_Q6_K, kFfn, kWidth));
    if (!x_ || !wg_ || !wu_ || !wd_) return false;
    kg::MarkRowPaddingReadable(wg_);
    kg::MarkRowPaddingReadable(wu_);
    kg::MarkRowPaddingReadable(wd_);
    gate_ = Place(ggml_mul_mat(c, wg_, x_));
    up_ = Place(ggml_mul_mat(c, wu_, x_));
    if (!gate_ || !up_) return false;
    glu_ = Place(ggml_geglu_split(c, gate_, up_));
    if (!glu_) return false;
    down_ = Place(ggml_mul_mat(c, wd_, glu_));
    if (!down_) return false;
    outputs_ = {gate_, up_, glu_, down_};
    // Quantize 64 distinct deterministic rows per bank, then repeat the bank.
    // This is a valid synthetic quant fixture at the actual tensor geometry.
    // No old captured scalar input/weight result is reassigned to this C2 test.
    for (const auto [weights, seed] : {std::pair{wg_, 3}, std::pair{wu_, 7}, std::pair{wd_, 11}}) {
      constexpr std::size_t bank_rows = 64;
      std::vector<float> values(static_cast<std::size_t>(weights->ne[0]) * bank_rows);
      for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(
                        static_cast<int>((i * 13 + static_cast<unsigned>(seed)) % 103) - 51) /
                    4096.0f;
      const auto row_bytes = ggml_row_size(weights->type, weights->ne[0]);
      std::vector<std::uint8_t> bank(row_bytes * bank_rows);
      ggml_quantize_init(weights->type);
      if (!Check(ggml_quantize_chunk(weights->type, values.data(), bank.data(), 0,
                                     static_cast<std::int64_t>(bank_rows), weights->ne[0],
                                     nullptr) == bank.size(),
                 "synthetic quant bank"))
        return false;
      auto& bytes = weights_[static_cast<std::size_t>(seed == 3 ? 0 : seed == 7 ? 1 : 2)];
      bytes.resize(ggml_nbytes(weights));
      for (std::size_t row = 0; row < static_cast<std::size_t>(weights->ne[1]); ++row)
        std::memcpy(bytes.data() + row * row_bytes, bank.data() + (row % bank_rows) * row_bytes,
                    row_bytes);
      if (!Check(cudaMemcpy(weights->data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) ==
                     cudaSuccess,
                 "weight upload"))
        return false;
    }
    original_.resize(static_cast<std::size_t>(kWidth * kColumns));
    for (std::size_t i = 0; i < original_.size(); ++i)
      original_[i] = static_cast<float>(static_cast<int>((i * 17) % 127) - 63) / 64.0f;
    if (!Upload(original_) || !Witness()) return false;
    auto pair = experiment::PlanOrdinaryC2Pair(*launch_, gate_, up_);
    auto down = kg::PlanMulMatVecQ(*launch_, down_);
    if (!Check(pair && *pair == 12672 && down && *down == kWorkspace,
               "exact original two-column plans"))
      return false;
    std::println(
        "PAIR_PLAN columns=2 K=5376 N=21504 Q4_K_prep=12672 down_Q6_K=48384 device_allocated={} "
        "budget={} synthetic_only=1",
        device_bytes_, kDeviceBudget);
    std::println("PAIR_HOST known_payload_allowance={} physical_peak_unmeasured=1",
                 kHostPayloadAllowance);
    return true;
  }
  bool Run() {
    if (!Check(bool(experiment::OrdinaryC2Pair(*launch_, gate_, up_)), "pair only") ||
        !Complete() || launch_->scratch_peak().value() != 12672)
      return false;
    if (!Chain(false) || !Complete()) return false;
    const auto original = Download();
    if (original.empty() || !Chain(true) || !Complete() || Download() != original) return false;
    auto baseline = launch_->Capture([&](auto&) { return RunChain(false); });
    auto candidate = launch_->Capture([&](auto&) { return RunChain(true); });
    if (!Check(bool(baseline) && bool(candidate), "capture both complete chains")) return false;
    if (!Check(baseline->nodes() == candidate->nodes() + 1, "exactly one prep node removed"))
      return false;
    std::println("PAIR_CAPTURE ordinary_nodes={} shared_nodes={}", baseline->nodes(),
                 candidate->nodes());
    auto changed = original_;
    for (std::size_t i = 0; i < changed.size(); ++i) changed[i] += (i % 3 == 0 ? 0.25f : -0.125f);
    for (const auto* input : {&changed, &original_}) {
      if (!Upload(*input) || !Chain(false) || !Complete()) return false;
      const auto expected = Download();
      if (expected.empty() || (input == &changed && expected == original)) return false;
      if (!Chain(true) || !Complete() || Download() != expected) return false;
      for (auto* graph : {&*baseline, &*candidate}) {
        for (int repeat = 0; repeat < 2; ++repeat) {
          if (!Poison() || !Check(bool(launch_->Launch(*graph)), "fresh poisoned replay") ||
              !Complete() || Download() != expected)
            return false;
        }
      }
    }
    // Refusals must queue nothing and preserve every output. Restore captured
    // outputs first; all mutated descriptors are copies, never the live graph.
    const auto before_refusals = Download();
    auto bad = *up_;
    bad.ne[1] = 1;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    bad = *up_;
    bad.ne[1] = 3;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    bad = *up_;
    bad.nb[1] += sizeof(float);
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    auto wrong_type = *wu_;
    wrong_type.type = GGML_TYPE_Q5_K;
    bad = *up_;
    bad.src[0] = &wrong_type;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    bad = *up_;
    bad.data = gate_->data;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    bad = *up_;
    bad.op_params[3] = GGML_PREC_F32;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    bad = *up_;
    bad.src[1] = glu_;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    auto bad_weight = *wu_;
    bad_weight.flags = 0;  // Authenticated readable-tail marker is mandatory.
    bad = *up_;
    bad.src[0] = &bad_weight;
    if (experiment::OrdinaryC2Pair(*launch_, gate_, &bad)) return false;
    auto short_context = kg::LaunchContext::Create(
        0, *execution_, stream_,
        {.base = reinterpret_cast<std::uintptr_t>(workspace_), .size = llmp::base::Bytes(12671)});
    if (!short_context || experiment::OrdinaryC2Pair(**short_context, gate_, up_)) return false;
    short_context->reset();
    if (!Complete() || Download() != before_refusals) return false;
    std::println(
        "PAIR_EXACT complete_gate_up_activation_down=1 eager=1 changed_input=1 "
        "capture_poisoned_replays=8 refusals=9 pair_peak=12672 whole_chain_peak={}",
        launch_->scratch_peak().value());
    // Captured A/B/B/A: includes original Q8 prep, BOTH ordinary MMVQ products,
    // separate GeGLU and original separately prepared Q6_K down. No copies,
    // validation, setup or event construction inside the timed graph.
    std::array<double, 4> walls{};
    const std::array<bool, 4> arms{false, true, true, false};
    for (std::size_t i = 0; i < arms.size(); ++i) {
      auto& graph = arms[i] ? *candidate : *baseline;
      cudaEvent_t start{}, stop{};
      if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
        return false;
      if (cudaEventRecord(start, native_) != cudaSuccess) return false;
      for (int repeat = 0; repeat < kPaidChains; ++repeat)
        if (!launch_->Launch(graph)) return false;
      if (cudaEventRecord(stop, native_) != cudaSuccess ||
          cudaEventSynchronize(stop) != cudaSuccess)
        return false;
      float elapsed = 0;
      if (cudaEventElapsedTime(&elapsed, start, stop) != cudaSuccess || !std::isfinite(elapsed) ||
          elapsed <= 0 || cudaEventDestroy(start) != cudaSuccess ||
          cudaEventDestroy(stop) != cudaSuccess)
        return false;
      walls[i] = elapsed;
      if (Download() != original) return false;
      std::println("PAIR_PAID arm={} chains={} elapsed_ms={:.9f}", arms[i] ? "shared" : "ordinary",
                   kPaidChains, walls[i]);
    }
    if (!Witness()) return false;
    const auto ordinary = (walls[0] + walls[3]) / 2;
    const auto shared = (walls[1] + walls[2]) / 2;
    std::println(
        "PAIR_COMPLETE exact=1 relative_pct={:.9f} ordinary_spread_ms={:.9f} "
        "shared_spread_ms={:.9f} synthetic_only=1",
        (shared / ordinary - 1) * 100, std::abs(walls[3] - walls[0]),
        std::abs(walls[2] - walls[1]));
    return true;
  }

 private:
  void* Allocate(std::size_t bytes) {
    if (bytes > kDeviceBudget - device_bytes_) return nullptr;
    void* data = nullptr;
    if (cudaMalloc(&data, bytes) != cudaSuccess) return nullptr;
    allocations_.push_back(data);
    device_bytes_ += bytes;
    if (cudaMemset(data, 0, bytes) != cudaSuccess) return nullptr;
    return data;
  }
  ggml_tensor* Place(ggml_tensor* tensor) {
    const auto extra = ggml_is_quantized(tensor->type) ? ggml_row_size(tensor->type, 512) : 256;
    void* data = Allocate(ggml_nbytes(tensor) + extra);
    if (!data) return nullptr;
    kg::TensorArena::Bind(tensor, reinterpret_cast<std::uintptr_t>(data));
    return tensor;
  }
  bool Upload(const std::vector<float>& input) {
    return Check(cudaMemcpy(x_->data, input.data(), ggml_nbytes(x_), cudaMemcpyHostToDevice) ==
                         cudaSuccess &&
                     Complete(),
                 "input upload");
  }
  bool Complete() {
    // The provider must also observe/release completion, not only CUDA sync.
    const auto fence = execution_->Record(stream_);
    if (!Check(bool(fence), "provider completion fence")) return false;
    if (!Check(cudaStreamSynchronize(native_) == cudaSuccess, "completed operator stream"))
      return false;
    const auto state = execution_->Query(*fence);
    return Check(state && *state == llmp::providers::FenceState::kComplete &&
                     bool(execution_->Release(*fence)),
                 "provider completion retirement");
  }
  std::expected<void, kg::KernelFailure> RunChain(bool shared) {
    auto first =
        shared ? experiment::OrdinaryC2Pair(*launch_, gate_, up_) : kg::MulMatVecQ(*launch_, gate_);
    if (!first) return first;
    if (!shared) {
      auto up = kg::MulMatVecQ(*launch_, up_);
      if (!up) return up;
    }
    auto activated = kg::GeGlu(*launch_, glu_);
    if (!activated) return activated;
    return kg::MulMatVecQ(*launch_, down_);
  }
  bool Chain(bool shared) {
    const auto result = RunChain(shared);
    if (!result) std::println(stderr, "OPERATOR_FAILURE {}", result.error().detail);
    return bool(result);
  }
  bool Poison() {
    if (cudaMemsetAsync(workspace_, 0xff, kWorkspace, native_) != cudaSuccess) return false;
    for (auto* output : outputs_)
      if (cudaMemsetAsync(output->data, 0xff, ggml_nbytes(output), native_) != cudaSuccess)
        return false;
    return true;
  }
  std::vector<std::uint32_t> Download() {
    if (!Complete()) return {};
    std::vector<std::uint32_t> values;
    for (auto* output : outputs_) {
      const auto before = values.size();
      values.resize(before + ggml_nbytes(output) / sizeof(std::uint32_t));
      if (cudaMemcpy(values.data() + before, output->data, ggml_nbytes(output),
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        return {};
    }
    if (!std::ranges::all_of(values,
                             [](auto bits) { return std::isfinite(std::bit_cast<float>(bits)); }))
      return {};
    return values;
  }
  bool Witness() {
    if (!Complete()) return false;
    std::vector<float> input(original_.size());
    if (cudaMemcpy(input.data(), x_->data, ggml_nbytes(x_), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        std::memcmp(input.data(), original_.data(), ggml_nbytes(x_)) != 0)
      return false;
    const std::array<ggml_tensor*, 3> roots{wg_, wu_, wd_};
    for (std::size_t i = 0; i < roots.size(); ++i) {
      std::vector<std::uint8_t> copied(weights_[i].size());
      if (cudaMemcpy(copied.data(), roots[i]->data, copied.size(), cudaMemcpyDeviceToHost) !=
              cudaSuccess ||
          copied != weights_[i])
        return false;
      std::vector<std::uint8_t> tail(ggml_row_size(roots[i]->type, 512));
      if (cudaMemcpy(tail.data(), static_cast<char*>(roots[i]->data) + copied.size(), tail.size(),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          !std::ranges::all_of(tail, [](auto value) { return value == 0; }))
        return false;
    }
    // Nonquantized allocations carry a zero trailing guard outside logical data.
    for (auto* tensor : {x_, gate_, up_, glu_, down_}) {
      std::array<std::uint8_t, 256> guard{};
      if (cudaMemcpy(guard.data(), static_cast<char*>(tensor->data) + ggml_nbytes(tensor),
                     guard.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
          !std::ranges::all_of(guard, [](auto value) { return value == 0; }))
        return false;
    }
    return true;
  }
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  llmp::providers::StreamId stream_;
  cudaStream_t native_{};
  std::optional<kg::TensorArena> arena_;
  std::unique_ptr<kg::LaunchContext> launch_;
  std::vector<void*> allocations_;
  std::size_t device_bytes_ = 0;
  void* workspace_{};
  ggml_tensor *x_{}, *wg_{}, *wu_{}, *wd_{}, *gate_{}, *up_{}, *glu_{}, *down_{};
  std::array<ggml_tensor*, 4> outputs_{};
  std::array<std::vector<std::uint8_t>, 3> weights_;
  std::vector<float> original_;
};
}  // namespace
int main(int argc, char**) {
  if (argc != 1) return 1;
  Screen screen;
  return screen.Setup() && screen.Run() ? 0 : 1;
}
