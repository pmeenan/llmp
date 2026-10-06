// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 4 text chunks over prepared GGML weights. Products join row-local
// inputs; each request segment retains independent positions, masks and KV.
// Descriptors only: no backing allocation, dispatch or supported-model claim.
#ifndef JITLLM_KERNELS_GGML_GEMMA4_GRAPH_H_
#define JITLLM_KERNELS_GGML_GEMMA4_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "model/gemma4.h"

namespace jitllm::kernels::ggml {

struct Gemma4SegmentShape {
  // n_past authenticates the initial build but is not a plan-cache dimension:
  // positions/cell indices/masks are fresh inputs at every execution. Sources
  // rechecks a new position against these same capacities/read widths.
  std::uint32_t slot = 0, rows = 0, n_past = 0;
  std::uint32_t global_n_kv = 0, local_n_kv = 0;
  bool operator==(const Gemma4SegmentShape& other) const {
    return slot == other.slot && rows == other.rows && global_n_kv == other.global_n_kv &&
           local_n_kv == other.local_n_kv;
  }
};
enum class Gemma4OutputMode : std::uint8_t { kHead, kStateOnly };
struct Gemma4ChunkShape {
  std::vector<Gemma4SegmentShape> segments;
  std::uint32_t outputs = 0;
  // Optional POST-final-norm rows, independently selected from the head.
  // Zero preserves the existing graph. Retention disables final-layer narrowing.
  std::uint32_t feature_outputs = 0;
  Gemma4OutputMode output_mode = Gemma4OutputMode::kHead;
  bool operator==(const Gemma4ChunkShape&) const = default;
};
// Packed is an explicit comparison mode; owners is a separately enabled
// four-stream path. Unsupported shapes retain independent attention unchanged.
enum class Gemma4AttentionMode : std::uint8_t { kIndependent, kPacked, kOwners };
struct Gemma4GraphOptions {
  // Prepared array strides, indexed by the binding's expert-array index.
  // Empty selects readable slices aligned to lcm(16, GGML block bytes),
  // matching executable ExpertSlab strides rather than on-disk alignment.
  std::vector<std::uint64_t> expert_stride;
  // Explicit diagnostics use the SAME layer builder. Defaults are a complete
  // token-input chunk and its head. A partial graph cannot be a runtime chunk.
  std::uint32_t first_layer = 0, layer_count = 0;  // zero count: remaining layers
  bool hidden_input = false, head = true;
  // Decode transfer from Qwen/DeepSeek: eligible quant products reuse one
  // Q8_1 preparation per logical input and preserve one-token sums. A false
  // reference build retains ordinary GGML primitives. No GeGLU writer.
  bool shared_q8 = false;
  // Graph-owned causal/ring mask producers from fresh positions. false
  // retains the fully funded diagnostic host-mask source path.
  bool device_masks = false;
  // Per-segment K rotation/direct flattening permits the separately checked
  // RoPE/cache-store planner policy. false keeps the original joined K RoPE.
  bool rope_store = false;
  // Narrow after final attention, before its sandwich norm, as the pinned
  // reference's masked frontier. false retains every final hidden row.
  bool narrow_final = false;
  Gemma4AttentionMode attention_mode = Gemma4AttentionMode::kIndependent;
  bool operator==(const Gemma4GraphOptions&) const = default;
};
struct Gemma4WeightLeaf {
  ggml_tensor* tensor = nullptr;
  model::Gemma4Tensor resource;
};
struct Gemma4SegmentTensors {
  Gemma4SegmentShape shape;
  std::uint32_t first_row = 0;
  ggml_tensor* global_cells = nullptr;
  ggml_tensor* local_cells = nullptr;
  // F16 masks have padded query columns for upstream attention's prepass.
  // Producers or diagnostic host sources fill every padded query/cell.
  ggml_tensor* global_mask = nullptr;
  ggml_tensor* local_mask = nullptr;
  std::vector<std::pair<ggml_tensor*, ggml_tensor*>> caches;
};
struct Gemma4Graph {
  model::Gemma4Profile profile;
  model::Gemma4Binding binding;  // Immutable descriptor contract retained for rebinding.
  std::uint32_t context = 0, max_rows = 0, global_capacity = 0, local_capacity = 0;
  std::uint64_t state_bytes = 0;
  Gemma4ChunkShape shape;
  Gemma4GraphOptions options;
  // Actual graph mode: unequal widths/unsupported shapes report independent.
  Gemma4AttentionMode attention_mode = Gemma4AttentionMode::kIndependent;
  ggml_tensor* tokens = nullptr;
  ggml_tensor* input_hidden = nullptr;
  ggml_tensor* positions = nullptr;
  ggml_tensor* out_ids = nullptr;
  ggml_tensor* feature_ids = nullptr;
  ggml_tensor* normalized_features = nullptr;
  ggml_tensor* hidden = nullptr;
  ggml_tensor* logits = nullptr;
  std::vector<Gemma4WeightLeaf> weights;
  std::vector<Gemma4SegmentTensors> segments;
  std::vector<ggml_tensor*> inputs, nodes;
  std::vector<std::pair<std::string, ggml_tensor*>> named;
  ggml_tensor* Named(const std::string& name) const;
};

std::size_t Gemma4GraphTensors(const model::Gemma4Profile& profile, std::size_t segments);
std::size_t Gemma4GraphTensors(const model::Gemma4Profile& profile, std::size_t segments,
                               const Gemma4GraphOptions& options);
// Shared by the opt-in builder and manual packed/owner causal comparison.
// Requires the complete original graph; malformed writer edges refuse before
// transformation. Unsupported shapes return it unchanged.
std::expected<void, KernelFailure> TransformGemma4Attention(TensorArena& arena, Gemma4Graph& graph,
                                                            Gemma4AttentionMode mode);
std::expected<void, KernelFailure> CheckGemma4Graph(const model::Gemma4Profile& profile,
                                                    const model::Gemma4Binding& binding,
                                                    const model::Gemma4StateLayout& state,
                                                    const Gemma4ChunkShape& shape,
                                                    const Gemma4GraphOptions& options = {});
// Validates the full approved binding/layout and every public shape/stride
// before reserving arena space or calling GGML. No fake profile is accepted.
std::expected<Gemma4Graph, KernelFailure> BuildGemma4Graph(TensorArena& arena,
                                                           const model::Gemma4Profile& profile,
                                                           const model::Gemma4Binding& binding,
                                                           const model::Gemma4StateLayout& state,
                                                           const Gemma4ChunkShape& shape,
                                                           const Gemma4GraphOptions& options = {});

}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_GEMMA4_GRAPH_H_
