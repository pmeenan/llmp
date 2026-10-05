// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_BENCHMARK_DENSE_FFN_CAPTURE_H_
#define JITLLM_BENCHMARK_DENSE_FFN_CAPTURE_H_
#include <array>
#include <filesystem>

#include "../docs/experiments/gemma-dense-ffn/replay_inputs.h"
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "kernels/ggml/launch.h"
namespace jitllm::benchmark {
class DenseFfnCapture {
 public:
  static constexpr std::uint64_t kHostBytes = 64ULL << 20;
  engine::Status Setup(engine::PagedNode&, const std::filesystem::path&, bool);
  void Arm(std::uint32_t position) {
    armed_ = enabled_ && position == 67;
    seen_ = {};
  }
  std::expected<void, kernels::ggml::KernelFailure> Before(kernels::ggml::LaunchContext&,
                                                           ggml_tensor*);
  std::expected<void, kernels::ggml::KernelFailure> After(kernels::ggml::LaunchContext&,
                                                          ggml_tensor*);
  engine::Status Save(const std::filesystem::path&) const;
  void Observe(const engine::PagedWeights&, std::uint32_t, std::uint64_t);
  static DenseFfnCapture* active;

 private:
  struct Tensor {
    void* pinned = nullptr;
    std::array<std::int64_t, 4> ne{}, root_ne{};
    std::array<std::uint64_t, 4> nb{}, root_nb{};
    std::uint64_t bytes = 0, root_bytes = 0, relative = 0;
    std::int32_t type = 0;
    bool recorded = false;
  };
  std::expected<void, kernels::ggml::KernelFailure> Copy(kernels::ggml::LaunchContext&,
                                                         ggml_tensor*, std::size_t, std::uint64_t);
  std::array<Tensor, 6> tensors_{};
  std::array<std::uint64_t, 3> weights_{};
  std::array<std::uint32_t, 3> indices_{};
  std::array<bool, 3> seen_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / 4>, 4> parameters_{};
  void* shared_ = nullptr;
  engine::PagedNode* node_ = nullptr;
  bool enabled_ = false, armed_ = false;
};
}  // namespace jitllm::benchmark
#endif
