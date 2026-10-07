// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Descriptor-only Gemma2 chunks. Products share columns; each segment has
// independent positions, causal/window masks and cache roots.
#ifndef JITLLM_KERNELS_GGML_GEMMA2_GRAPH_H_
#define JITLLM_KERNELS_GGML_GEMMA2_GRAPH_H_
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma2.h"

namespace jitllm::kernels::ggml {
struct Gemma2SegmentShape {
  std::uint32_t slot = 0, rows = 0, n_past = 0, global_n_kv = 0, local_n_kv = 0;
  // Fresh positions/cells/masks are inputs, not cache-key dimensions. Equal
  // widths/capacities permit rebinding only after Gemma2Sources rechecks them.
  bool operator==(const Gemma2SegmentShape& other) const {
    return slot == other.slot && rows == other.rows && global_n_kv == other.global_n_kv &&
           local_n_kv == other.local_n_kv;
  }
};
enum class Gemma2OutputMode : std::uint8_t { kHead, kStateOnly };
struct Gemma2ChunkShape {
  std::vector<Gemma2SegmentShape> segments;
  std::uint32_t outputs = 0;
  Gemma2OutputMode output_mode = Gemma2OutputMode::kHead;
  // Frontier head mode only: publish lowest-ID argmax tokens, not rows.
  bool greedy = false;
  bool operator==(const Gemma2ChunkShape&) const = default;
};
struct Gemma2GraphOptions {
  // Explicit diagnostics; ordinary complete chunks use these defaults.
  std::uint32_t first_layer = 0, layer_count = 0;
  bool hidden_input = false, head = true, narrow_final = false;
  bool operator==(const Gemma2GraphOptions&) const = default;
};
struct Gemma2WeightLeaf {
  ggml_tensor* tensor = nullptr;
  model::Gemma2Tensor resource;
};
struct Gemma2SegmentTensors {
  Gemma2SegmentShape shape;
  std::uint32_t first_row = 0;
  ggml_tensor* global_cells = nullptr;
  ggml_tensor* local_cells = nullptr;
  ggml_tensor* global_mask = nullptr;
  ggml_tensor* local_mask = nullptr;
  std::vector<std::pair<ggml_tensor*, ggml_tensor*>> caches;
};
struct Gemma2Graph {
  model::Gemma2Profile profile;
  model::Gemma2Binding binding;
  std::uint32_t context = 0, max_rows = 0, global_capacity = 0, local_capacity = 0;
  std::uint64_t state_bytes = 0;
  Gemma2ChunkShape shape;
  Gemma2GraphOptions options;
  ggml_tensor* tokens = nullptr;
  ggml_tensor* input_hidden = nullptr;
  ggml_tensor* positions = nullptr;
  ggml_tensor* out_ids = nullptr;
  ggml_tensor* hidden = nullptr;
  ggml_tensor* logits = nullptr;
  ggml_tensor* greedy = nullptr;  // I32 [outputs], with a greedy shape
  std::vector<Gemma2WeightLeaf> weights;
  std::vector<Gemma2SegmentTensors> segments;
  std::vector<ggml_tensor*> inputs, nodes;
  std::vector<std::pair<std::string, ggml_tensor*>> named;
  ggml_tensor* Named(const std::string& name) const;
};
std::size_t Gemma2GraphTensors(const model::Gemma2Profile& profile, std::size_t segments);
std::expected<void, KernelFailure> CheckGemma2Graph(const model::Gemma2Profile& profile,
                                                    const model::Gemma2Binding& binding,
                                                    const model::Gemma2StateLayout& state,
                                                    const Gemma2ChunkShape& shape,
                                                    const Gemma2GraphOptions& options = {});
std::expected<Gemma2Graph, KernelFailure> BuildGemma2Graph(TensorArena& arena,
                                                           const model::Gemma2Profile& profile,
                                                           const model::Gemma2Binding& binding,
                                                           const model::Gemma2StateLayout& state,
                                                           const Gemma2ChunkShape& shape,
                                                           const Gemma2GraphOptions& options = {});
}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_GEMMA2_GRAPH_H_
