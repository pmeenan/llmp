// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Descriptor port of llama.cpp d8123504 src/models/gemma3.cpp and build_ffn.
// Native independent-slot lifetime/placement mechanics follow Gemma4's
// Apache-2.0 adapter, without its V normalization, tied KV or MoE arithmetic.
#include "kernels/ggml/gemma3_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/representation.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/jitllm_ops.h"

namespace jitllm::kernels::ggml {
namespace {
namespace md = jitllm::model;
std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
std::uint64_t Pad(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}
ggml_tensor* Norm(ggml_context* c, const md::Gemma3Profile& p, ggml_tensor* x,
                  ggml_tensor* weight) {
  return ggml_mul(c, ggml_rms_norm(c, x, p.rms_eps), weight);
}
}  // namespace

std::expected<void, KernelFailure> CheckGemma3Graph(const md::Gemma3Profile& p,
                                                    const md::Gemma3Binding& binding,
                                                    const md::Gemma3StateLayout& state,
                                                    const Gemma3ChunkShape& shape,
                                                    const Gemma3GraphOptions& o) {
  if (auto checked = md::CheckGemma3Binding(p, binding); !checked) return Rejected(checked.error());
  if (!state.Representations(p)) return Rejected("invalid Gemma3 state layout");
  const bool state_only = shape.output_mode == Gemma3OutputMode::kStateOnly;
  if ((shape.output_mode != Gemma3OutputMode::kHead && !state_only) ||
      (state_only && (!o.head || o.hidden_input || o.first_layer != 0 ||
                      (o.layer_count != 0 && o.layer_count != p.layers) || shape.outputs != 0)))
    return Rejected("Gemma3 state-only requires all token-input layers and no head rows");
  if (shape.greedy && (state_only || !o.head || shape.outputs != shape.segments.size()))
    return Rejected("Gemma3 greedy chunks require one frontier head per segment");
  if (shape.segments.empty() || shape.segments.size() > md::kGemma3MaxSlots ||
      o.first_layer >= p.layers || (o.first_layer != 0 && !o.hidden_input) ||
      o.layer_count > p.layers - o.first_layer ||
      (o.narrow_final &&
       (!o.head || (o.layer_count != 0 && o.first_layer + o.layer_count != p.layers))))
    return Rejected("invalid Gemma3 segment or diagnostic graph scope");
  std::array<bool, md::kGemma3MaxSlots> seen{};
  std::uint32_t rows = 0;
  for (const auto& segment : shape.segments) {
    if (segment.slot >= seen.size() || seen[segment.slot] || segment.rows == 0 ||
        segment.rows > state.max_rows - rows || segment.n_past > state.context ||
        segment.rows > state.context - segment.n_past)
      return Rejected("empty, repeated or out-of-bounds Gemma3 segment");
    seen[segment.slot] = true;
    rows += segment.rows;
    const auto end = segment.n_past + segment.rows;
    for (const auto [read, capacity] : {std::pair{segment.global_n_kv, state.global_cells},
                                        std::pair{segment.local_n_kv, state.local_cells}})
      if (read == 0 || read % 256 != 0 || read > capacity || read < std::min(end, capacity) ||
          std::uint64_t{read} * Pad(segment.rows, 32) * 2 > INT32_MAX)
        return Rejected("invalid Gemma3 read width or padded mask byte stride");
  }
  if (!state_only &&
      ((o.head && (shape.outputs == 0 || shape.outputs > rows)) || (!o.head && shape.outputs != 0)))
    return Rejected("invalid Gemma3 frontier output count");
  return {};
}

ggml_tensor* Gemma3Graph::Named(const std::string& name) const {
  const auto found =
      std::ranges::find_if(named, [&](const auto& tensor) { return tensor.first == name; });
  return found == named.end() ? nullptr : found->second;
}
std::size_t Gemma3GraphTensors(const md::Gemma3Profile& p, std::size_t segments) {
  if (p != md::Gemma3_4BQat() || segments == 0 || segments > md::kGemma3MaxSlots) return 0;
  return 128 + std::size_t{p.layers} * (96 + segments * 48);
}

std::expected<Gemma3Graph, KernelFailure> BuildGemma3Graph(TensorArena& arena,
                                                           const md::Gemma3Profile& p,
                                                           const md::Gemma3Binding& binding,
                                                           const md::Gemma3StateLayout& state,
                                                           const Gemma3ChunkShape& shape,
                                                           const Gemma3GraphOptions& o) {
  if (auto checked = CheckGemma3Graph(p, binding, state, shape, o); !checked)
    return std::unexpected(checked.error());
  if (auto room = arena.Reserve(Gemma3GraphTensors(p, shape.segments.size())); !room)
    return std::unexpected(room.error());
  auto* c = arena.context();
  Gemma3Graph g;
  g.profile = p;
  g.binding = binding;
  g.context = state.context;
  g.max_rows = state.max_rows;
  g.global_capacity = state.global_cells;
  g.local_capacity = state.local_cells;
  g.state_bytes = state.bytes;
  g.shape = shape;
  g.options = o;
  std::int64_t rows = 0;
  for (const auto& segment : shape.segments) rows += segment.rows;
  if (o.hidden_input)
    g.input_hidden = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.width, rows);
  else
    g.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
  g.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
  if (o.head && shape.output_mode == Gemma3OutputMode::kHead)
    g.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, shape.outputs);
  g.inputs = {o.hidden_input ? g.input_hidden : g.tokens, g.positions};
  if (g.out_ids != nullptr) g.inputs.push_back(g.out_ids);
  std::uint32_t first_row = 0;
  for (const auto& segment : shape.segments) {
    Gemma3SegmentTensors tensors;
    tensors.shape = segment;
    tensors.first_row = first_row;
    first_row += segment.rows;
    tensors.global_cells = ggml_new_tensor_1d(c, GGML_TYPE_I64, segment.rows);
    tensors.local_cells = ggml_new_tensor_1d(c, GGML_TYPE_I64, segment.rows);
    tensors.global_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, segment.global_n_kv,
                                             static_cast<std::int64_t>(Pad(segment.rows, 32)));
    tensors.local_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, segment.local_n_kv,
                                            static_cast<std::int64_t>(Pad(segment.rows, 32)));
    tensors.caches.resize(p.layers);
    for (auto* input :
         {tensors.global_cells, tensors.local_cells, tensors.global_mask, tensors.local_mask})
      g.inputs.push_back(input);
    g.segments.push_back(std::move(tensors));
  }
  const auto weight = [&](const md::Gemma3Tensor& resource) {
    const auto found = std::ranges::find_if(
        g.weights, [&](const auto& leaf) { return leaf.resource.index == resource.index; });
    if (found != g.weights.end()) return found->tensor;
    const auto* type = artifact::FindGgmlType(resource.type);
    std::array<std::int64_t, 4> dims{1, 1, 1, 1};
    for (std::size_t i = 0; i < resource.ne.size(); ++i)
      dims[i] = static_cast<std::int64_t>(resource.ne[i]);
    auto* tensor = ggml_new_tensor(c, static_cast<ggml_type>(type->id),
                                   static_cast<int>(resource.ne.size()), dims.data());
    g.weights.push_back({tensor, resource});
    return tensor;
  };
  const auto named = [&](std::string name, ggml_tensor* tensor) {
    ggml_set_name(tensor, name.c_str());
    g.named.emplace_back(std::move(name), tensor);
    return tensor;
  };
  const auto product = [&](const md::Gemma3Tensor& resource, ggml_tensor* input) {
    return ggml_mul_mat(c, weight(resource), input);
  };
  std::vector<ggml_tensor*> expanded;
  auto* input = o.hidden_input
                    ? g.input_hidden
                    : ggml_scale(c, ggml_get_rows(c, weight(binding.token_embd), g.tokens),
                                 std::sqrt(static_cast<float>(p.width)));
  const auto count = o.layer_count == 0 ? p.layers - o.first_layer : o.layer_count;
  for (std::uint32_t il = o.first_layer; il < o.first_layer + count; ++il) {
    const auto& layer = binding.layers[il];
    const auto prefix = std::format("blk.{}.", il);
    const auto d = p.key_dim, kv_width = d * p.kv_heads;
    auto* normalized = named(prefix + "attn_input", Norm(c, p, input, weight(layer.attn_norm)));
    const bool tail = shape.output_mode != Gemma3OutputMode::kStateOnly || il + 1 != p.layers;
    auto* q = tail ? ggml_reshape_3d(c, product(layer.q, normalized), d, p.heads, rows) : nullptr;
    auto* k = ggml_reshape_3d(c, product(layer.k, normalized), d, p.kv_heads, rows);
    // Gemma3 V remains raw: no RMS normalization or K/V identity sharing.
    auto* v = ggml_reshape_3d(c, product(layer.v, normalized), d, p.kv_heads, rows);
    const auto rope = [&](ggml_tensor* tensor) {
      return ggml_rope_ext(c, tensor, g.positions, nullptr, static_cast<int>(d),
                           GGML_ROPE_TYPE_NEOX, static_cast<int>(p.context),
                           p.local(il) ? 10000.0F : p.rope_base,
                           p.local(il) ? 1.0F : 1.0F / p.rope_scale, 0.0F, 1.0F, 32.0F, 1.0F);
    };
    if (tail) {
      q = named(prefix + "q_rope", rope(Norm(c, p, q, weight(layer.q_norm))));
      q = named(prefix + "q_scaled", ggml_scale(c, q, 1.0F / std::sqrt(static_cast<float>(d))));
      expanded.push_back(q);
    }
    k = named(prefix + "k_rope", rope(Norm(c, p, k, weight(layer.k_norm))));
    v = named(prefix + "v_raw", v);
    expanded.push_back(v);
    expanded.push_back(k);
    auto* k_rows = ggml_reshape_2d(c, k, kv_width, rows);
    auto* v_rows = ggml_reshape_2d(c, v, kv_width, rows);
    ggml_tensor* joined = nullptr;
    for (auto& segment : g.segments) {
      const auto& s = segment.shape;
      const auto cells = p.local(il) ? state.local_cells : state.global_cells;
      const auto read = p.local(il) ? s.local_n_kv : s.global_n_kv;
      auto* indices = p.local(il) ? segment.local_cells : segment.global_cells;
      auto* mask = p.local(il) ? segment.local_mask : segment.global_mask;
      auto* cache_k = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_width, cells);
      auto* cache_v = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_width, cells);
      segment.caches[il] = {cache_k, cache_v};
      const auto slice = [&](ggml_tensor* tensor) {
        return ggml_view_2d(c, tensor, kv_width, s.rows, tensor->nb[1],
                            std::size_t{segment.first_row} * tensor->nb[1]);
      };
      expanded.push_back(ggml_set_rows(c, cache_k, slice(k_rows), indices));
      expanded.push_back(ggml_set_rows(c, cache_v, slice(v_rows), indices));
      if (!tail) continue;
      const auto cache_view = [&](ggml_tensor* tensor) {
        return ggml_permute(c,
                            ggml_view_3d(c, tensor, d, p.kv_heads, read, std::size_t{d} * 2,
                                         std::size_t{kv_width} * 2, 0),
                            0, 2, 1, 3);
      };
      auto* queries = ggml_view_3d(c, q, d, p.heads, s.rows, q->nb[1], q->nb[2],
                                   std::size_t{segment.first_row} * q->nb[2]);
      queries = ggml_permute(c, queries, 0, 2, 1, 3);
      auto* attention = ggml_flash_attn_ext(c, queries, cache_view(cache_k), cache_view(cache_v),
                                            mask, 1.0F, 0.0F, 0.0F);
      ggml_prec_set_acc(attention, GGML_PREC_F32);
      attention = ggml_reshape_2d(c, attention, std::int64_t{d} * p.heads, s.rows);
      named(prefix + std::format("slot.{}.attention", s.slot), attention);
      expanded.push_back(attention);
      joined = joined == nullptr ? attention : ggml_concat(c, joined, attention, 1);
    }
    if (!tail) break;
    auto* projected = named(prefix + "attn_projection", product(layer.out, joined));
    auto* residual = input;
    if (o.narrow_final && il + 1 == p.layers) {
      projected = ggml_get_rows(c, projected, g.out_ids);
      residual = ggml_get_rows(c, residual, g.out_ids);
      rows = shape.outputs;
    }
    auto* attention_residual =
        named(prefix + "attn_residual",
              ggml_add(c, Norm(c, p, projected, weight(layer.attn_post_norm)), residual));
    auto* ffn_input = Norm(c, p, attention_residual, weight(layer.ffn_norm));
    auto* up = product(layer.up, ffn_input);
    auto* gate = product(layer.gate, ffn_input);
    auto* down = product(layer.down, ggml_geglu_split(c, gate, up));
    input = named(prefix + "output",
                  ggml_add(c, Norm(c, p, down, weight(layer.ffn_post_norm)), attention_residual));
  }
  if (shape.output_mode != Gemma3OutputMode::kStateOnly) g.hidden = input;
  if (o.head && g.hidden != nullptr) {
    auto* normalized = named("output_normalized", Norm(c, p, input, weight(binding.output_norm)));
    if (!o.narrow_final) normalized = ggml_get_rows(c, normalized, g.out_ids);
    g.logits = named("logits", product(binding.output, normalized));
    expanded.push_back(g.logits);
    if (shape.greedy) {
      g.greedy = named("greedy", Argmax(c, g.logits));
      expanded.push_back(g.greedy);
    }
  } else if (g.hidden != nullptr)
    expanded.push_back(g.hidden);
  auto ordered = GraphOrder(expanded, arena);
  if (!ordered) return std::unexpected(ordered.error());
  g.nodes = std::move(*ordered);
  return g;
}
}  // namespace jitllm::kernels::ggml
