// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef JITLLM_BENCHMARKS_GEMMA_ATTENTION_GLOBAL_CAPTURE_H_
#define JITLLM_BENCHMARKS_GEMMA_ATTENTION_GLOBAL_CAPTURE_H_

#include <array>
#include <cstdint>
#include <filesystem>

#include "engine/paged_node.h"
#include "kernels/ggml/launch.h"

namespace jitllm::benchmark {
// Fixed first-global D512/C4 native-origin witness. The caller retains this
// object, its node and all node-owned pinned destinations through retirement.
class GemmaAttentionGlobalCapture {
 public:
  static constexpr std::uint64_t kHostBytes = 2U << 20U;
  engine::Status Setup(engine::PagedNode& node, bool enabled);
  void BeginWave() { calls_ = 0; }
  engine::Status Save(const std::filesystem::path& out) const;
  std::expected<void, kernels::ggml::KernelFailure> Before(kernels::ggml::LaunchContext& launch,
                                                           ggml_tensor* node);
  static GemmaAttentionGlobalCapture* active;

 private:
  struct Tensor {
    std::array<std::int64_t, 4> ne{};
    std::array<std::uint64_t, 4> nb{};
    std::uint64_t bytes = 0;
    std::uint64_t relative_offset = 0, root_bytes = 0;
    std::array<std::int64_t, 4> root_ne{};
    std::array<std::uint64_t, 4> root_nb{};
    std::uint32_t view_depth = 0;
    std::int32_t type = 0;
    bool recorded = false;
    void* pinned = nullptr;
  };
  std::array<std::array<Tensor, 4>, 4> tensors_{};
  std::array<std::array<std::int32_t, GGML_MAX_OP_PARAMS / sizeof(std::int32_t)>, 4> parameters_{};
  std::uint32_t calls_ = 0;
  bool enabled_ = false;
  engine::PagedNode* node_ = nullptr;
};
}  // namespace jitllm::benchmark
#endif
