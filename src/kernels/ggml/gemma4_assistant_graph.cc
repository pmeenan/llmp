// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Port of pinned llama.cpp b29c606e models/gemma4-assistant.cpp. The target
// completed endpoint is separate from the constant, unwritten draft position.
#include "kernels/ggml/gemma4_assistant_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "artifact/representation.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::kernels::ggml {
namespace {
namespace md = model;
std::unexpected<KernelFailure> Refused(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
std::uint32_t ReadCells(std::uint32_t prefix, std::uint32_t capacity) {
  return std::min(capacity, (prefix + 255) / 256 * 256);
}
}  // namespace
std::expected<void, KernelFailure> CheckGemma4AssistantGraph(const md::Gemma4AssistantProfile& p,
                                                             const md::Gemma4AssistantBinding& b,
                                                             const md::Gemma4Profile& target,
                                                             const md::Gemma4Binding& tb,
                                                             const md::Gemma4StateLayout& state,
                                                             const Gemma4AssistantShape& shape) {
  auto paired = md::CheckGemma4AssistantTarget(p, b, target, tb);
  if (!paired) return Refused(paired.error());
  if (!state.Representations(target) || shape.segments.empty() ||
      shape.segments.size() > md::kGemma4MaxSlots)
    return Refused("assistant needs a bounded target state and independent queries");
  std::array<bool, md::kGemma4MaxSlots> seen{};
  for (const auto& s : shape.segments) {
    if (s.slot >= seen.size() || seen[s.slot] || s.prefix == 0 || s.prefix >= state.context ||
        s.local_n_kv != ReadCells(s.prefix, state.local_cells) ||
        s.global_n_kv != ReadCells(s.prefix, state.global_cells) ||
        std::uint64_t{s.local_n_kv} * 32 * 2 > INT32_MAX ||
        std::uint64_t{s.global_n_kv} * 32 * 2 > INT32_MAX)
      return Refused("assistant frozen endpoint/read widths or slot identity differ");
    seen[s.slot] = true;
  }
  return {};
}
std::size_t Gemma4AssistantGraphTensors(std::size_t segments) { return 256 + segments * 80; }

std::expected<Gemma4AssistantGraph, KernelFailure> BuildGemma4AssistantGraph(
    TensorArena& arena, const md::Gemma4AssistantProfile& p, const md::Gemma4AssistantBinding& b,
    const md::Gemma4Profile& target, const md::Gemma4Binding& tb,
    const md::Gemma4StateLayout& state, const Gemma4AssistantShape& shape) {
  if (auto checked = CheckGemma4AssistantGraph(p, b, target, tb, state, shape); !checked)
    return std::unexpected(checked.error());
  if (auto room = arena.Reserve(Gemma4AssistantGraphTensors(shape.segments.size())); !room)
    return std::unexpected(room.error());
  auto* c = arena.context();
  Gemma4AssistantGraph g;
  g.profile = p;
  g.binding = b;
  g.target = target;
  g.target_embedding = tb.token_embd;
  g.context = state.context;
  g.max_rows = state.max_rows;
  g.local_capacity = state.local_cells;
  g.global_capacity = state.global_cells;
  g.shape = shape;
  const auto rows = static_cast<std::int64_t>(shape.segments.size());
  g.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
  g.features = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.target_width, rows);
  g.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
  g.inputs = {g.tokens, g.features, g.positions};
  for (const auto& s : shape.segments) {
    Gemma4AssistantSegmentTensors seg;
    seg.shape = s;
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto d = kind == 0 ? target.local_head_dim : target.global_head_dim;
      const auto kvh = kind == 0 ? p.local_kv_heads : p.global_kv_heads;
      const auto cells = kind == 0 ? s.local_n_kv : s.global_n_kv;
      seg.caches[kind] = {ggml_new_tensor_2d(c, GGML_TYPE_F16, d * kvh, cells),
                          ggml_new_tensor_2d(c, GGML_TYPE_F16, d * kvh, cells)};
      seg.masks[kind] = ggml_new_tensor_2d(c, GGML_TYPE_F16, cells, 32);
      g.inputs.push_back(seg.masks[kind]);
    }
    g.segments.push_back(seg);
  }
  const auto weight = [&](const md::Gemma4Tensor& r) {
    const auto at =
        std::ranges::find_if(g.weights, [&](const auto& w) { return w.resource.index == r.index; });
    if (at != g.weights.end()) return at->tensor;
    std::array<std::int64_t, 2> ne{1, 1};
    for (std::size_t i = 0; i < r.ne.size(); ++i) ne[i] = static_cast<std::int64_t>(r.ne[i]);
    auto* t = ggml_new_tensor(c, static_cast<ggml_type>(artifact::FindGgmlType(r.type)->id),
                              static_cast<int>(r.ne.size()), ne.data());
    g.weights.push_back({t, r});
    return t;
  };
  std::array<std::int64_t, 2> emb_ne{target.width, target.vocab};
  g.embedding = ggml_new_tensor(
      c, static_cast<ggml_type>(artifact::FindGgmlType(tb.token_embd.type)->id), 2, emb_ne.data());
  if (artifact::FindGgmlType(tb.token_embd.type)->block_elements > 1 &&
      target.width % artifact::kGgmlRowPadding != 0)
    MarkRowPaddingReadable(g.embedding);
  const auto norm = [&](ggml_tensor* x, const md::Gemma4Tensor& w) {
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), weight(w));
  };
  auto* x = ggml_scale(c, ggml_get_rows(c, g.embedding, g.tokens),
                       std::sqrt(static_cast<float>(p.target_width)));
  auto* input = ggml_mul_mat(c, weight(b.pre_projection), ggml_concat(c, x, g.features, 0));
  std::vector<ggml_tensor*> expanded;
  for (std::uint32_t il = 0; il < md::kGemma4AssistantLayers; ++il) {
    const auto& l = b.layers[il];
    const std::size_t kind = il == 3 ? 1 : 0;
    const auto d = kind == 0 ? target.local_head_dim : target.global_head_dim;
    const auto kvh = kind == 0 ? p.local_kv_heads : p.global_kv_heads;
    auto* q = ggml_reshape_3d(c, ggml_mul_mat(c, weight(l.q), norm(input, l.attn_norm)), d, p.heads,
                              rows);
    q = norm(q, l.q_norm);
    q = ggml_rope_ext(c, q, g.positions, kind == 0 ? nullptr : weight(b.rope_freqs),
                      static_cast<int>(d), GGML_ROPE_TYPE_NEOX, static_cast<int>(target.context),
                      kind == 0 ? target.local_rope_base : target.global_rope_base, 1.0f, 0.0f,
                      1.0f, 32.0f, 1.0f);
    expanded.push_back(q);
    ggml_tensor* joined = nullptr;
    for (std::size_t row = 0; row < g.segments.size(); ++row) {
      auto& seg = g.segments[row];
      const auto cells = kind == 0 ? seg.shape.local_n_kv : seg.shape.global_n_kv;
      const auto view = [&](ggml_tensor* cache) {
        return ggml_permute(
            c,
            ggml_view_3d(c, cache, d, kvh, cells, std::size_t{d} * 2, std::size_t{d} * kvh * 2, 0),
            0, 2, 1, 3);
      };
      auto* qr = ggml_view_3d(c, q, d, p.heads, 1, q->nb[1], q->nb[2], row * q->nb[2]);
      qr = ggml_permute(c, qr, 0, 2, 1, 3);
      auto* attn =
          ggml_flash_attn_ext(c, qr, view(seg.caches[kind].first), view(seg.caches[kind].second),
                              seg.masks[kind], 1.0f, 0.0f, 0.0f);
      ggml_prec_set_acc(attn, GGML_PREC_F32);
      attn = ggml_reshape_2d(c, attn, std::int64_t{d} * p.heads, 1);
      expanded.push_back(attn);
      joined = joined == nullptr ? attn : ggml_concat(c, joined, attn, 1);
    }
    auto* projected = ggml_mul_mat(c, weight(l.out), joined);
    auto* residual = ggml_add(c, norm(projected, l.attn_post_norm), input);
    auto* ffn_input = norm(residual, l.ffn_norm);
    auto* gate = ggml_mul_mat(c, weight(l.gate), ffn_input);
    auto* up = ggml_mul_mat(c, weight(l.up), ffn_input);
    auto* ffn = ggml_mul_mat(c, weight(l.down), ggml_geglu_split(c, gate, up));
    input = ggml_mul(c, ggml_add(c, norm(ffn, l.ffn_post_norm), residual), weight(l.output_scale));
  }
  auto* normalized = norm(input, b.output_norm);
  g.logits = ggml_mul_mat(c, weight(b.head), normalized);
  g.next_features = ggml_mul_mat(c, weight(b.post_projection), normalized);
  expanded.push_back(g.logits);
  expanded.push_back(g.next_features);
  g.nodes = GraphOrder(expanded);
  return g;
}
}  // namespace jitllm::kernels::ggml
