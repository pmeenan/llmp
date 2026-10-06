// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Port of llama.cpp b29c606e gemma4.cpp and build_ffn/build_moe_ffn.
// Segmentation changes product columns, never a request's KV or routing order.
#include "kernels/ggml/gemma4_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "artifact/representation.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::kernels::ggml {
namespace {
namespace md = jitllm::model;
std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
std::uint64_t Pad(std::uint64_t x, std::uint64_t align) {
  return ((x + align - 1) / align) * align;
}
std::expected<void, KernelFailure> Check(const md::Gemma4Profile& p, const md::Gemma4Binding& b,
                                         const md::Gemma4StateLayout& state,
                                         const Gemma4ChunkShape& shape,
                                         const Gemma4GraphOptions& o) {
  if (o.attention_mode != Gemma4AttentionMode::kIndependent &&
      o.attention_mode != Gemma4AttentionMode::kPacked &&
      o.attention_mode != Gemma4AttentionMode::kOwners)
    return Rejected("invalid Gemma4 attention mode");
  auto checked = md::CheckGemma4Binding(p, b);
  if (!checked) return Rejected(checked.error());
  if (!state.Representations(p)) return Rejected("invalid Gemma4 state layout");
  std::size_t arrays = 0;
  for (const auto& l : b.layers) {
    for (const auto* t : {&l.gate_up_exps, &l.gate_exps, &l.up_exps, &l.down_exps}) {
      if (t->has_value()) arrays = std::max(arrays, std::size_t{(**t).index} + 1);
    }
  }
  if (!o.expert_stride.empty() && o.expert_stride.size() != arrays) {
    return Rejected("invalid Gemma4 prepared expert stride domain");
  }
  const bool state_only = shape.output_mode == Gemma4OutputMode::kStateOnly;
  if ((shape.output_mode != Gemma4OutputMode::kHead && !state_only) ||
      (state_only && (!o.head || o.hidden_input || o.first_layer != 0 ||
                      (o.layer_count != 0 && o.layer_count != p.layers) || shape.outputs != 0 ||
                      shape.feature_outputs != 0)))
    return Rejected("Gemma4 state-only chunks require all token-input layers and no outputs");
  if (shape.segments.empty() || shape.segments.size() > md::kGemma4MaxSlots ||
      o.first_layer >= p.layers || (o.first_layer != 0 && !o.hidden_input) ||
      o.layer_count > p.layers - o.first_layer ||
      (o.narrow_final &&
       (!o.head || (o.layer_count != 0 && o.first_layer + o.layer_count != p.layers)))) {
    return Rejected("invalid Gemma4 segment count or diagnostic layer/head contract");
  }
  std::array<bool, md::kGemma4MaxSlots> seen{};
  std::uint32_t rows = 0;
  for (const auto& s : shape.segments) {
    if (s.slot >= seen.size() || seen[s.slot] || s.rows == 0 || s.rows > state.max_rows - rows ||
        s.n_past > state.context || s.rows > state.context - s.n_past) {
      return Rejected("empty, repeated or out-of-bounds Gemma4 segment");
    }
    seen[s.slot] = true;
    rows += s.rows;
    const auto end = s.n_past + s.rows;
    for (const auto [n, cells] : {std::pair{s.global_n_kv, state.global_cells},
                                  std::pair{s.local_n_kv, state.local_cells}}) {
      if (n == 0 || n % 256 != 0 || n > cells || n < std::min(end, cells) ||
          std::uint64_t{n} * Pad(s.rows, 32) * 2 > INT32_MAX) {
        return Rejected("invalid Gemma4 attended cells or padded mask stride");
      }
    }
  }
  if (!state_only && ((o.head && (shape.outputs == 0 || shape.outputs > rows)) ||
                      (!o.head && shape.outputs != 0))) {
    return Rejected("invalid Gemma4 frontier output count");
  }
  if (shape.feature_outputs > rows ||
      (shape.feature_outputs != 0 &&
       (!o.head || o.narrow_final ||
        o.first_layer + (o.layer_count == 0 ? p.layers - o.first_layer : o.layer_count) !=
            p.layers)))
    return Rejected("Gemma4 normalized features need complete unnarrowed final rows");
  for (const auto& l : b.layers) {
    for (const auto* t : {&l.gate_up_exps, &l.gate_exps, &l.up_exps, &l.down_exps}) {
      if (!t->has_value()) continue;
      const auto& r = **t;
      if (!o.expert_stride.empty() && r.index >= o.expert_stride.size()) {
        return Rejected("missing Gemma4 prepared expert stride");
      }
      const auto* type = jitllm::artifact::FindGgmlType(r.type);
      const auto align = std::lcm(std::uint64_t{16}, std::uint64_t{type->block_bytes});
      const auto stride =
          o.expert_stride.empty() ? Pad(r.readable, align) : o.expert_stride[r.index];
      if (stride < r.readable || stride % align != 0 || stride > UINT64_MAX / p.experts ||
          stride * p.experts > std::numeric_limits<std::size_t>::max()) {
        return Rejected("invalid Gemma4 readable expert stride");
      }
    }
  }
  return {};
}
ggml_tensor* Norm(ggml_context* c, const md::Gemma4Profile& p, ggml_tensor* x, ggml_tensor* w) {
  return ggml_mul(c, ggml_rms_norm(c, x, p.rms_eps), w);
}
}  // namespace

std::expected<void, KernelFailure> CheckGemma4Graph(const md::Gemma4Profile& p,
                                                    const md::Gemma4Binding& b,
                                                    const md::Gemma4StateLayout& state,
                                                    const Gemma4ChunkShape& shape,
                                                    const Gemma4GraphOptions& o) {
  return Check(p, b, state, shape, o);
}

ggml_tensor* Gemma4Graph::Named(const std::string& name) const {
  const auto it = std::ranges::find_if(named, [&](const auto& t) { return t.first == name; });
  return it == named.end() ? nullptr : it->second;
}
std::size_t Gemma4GraphTensors(const md::Gemma4Profile& p, std::size_t segments) {
  return 128 + std::size_t{p.layers} * (160 + segments * 48);
}

std::size_t Gemma4GraphTensors(const md::Gemma4Profile& p, std::size_t segments,
                               const Gemma4GraphOptions& options) {
  return Gemma4GraphTensors(p, segments) +
         (options.attention_mode == Gemma4AttentionMode::kOwners && (segments == 2 || segments == 3)
              ? std::size_t{p.layers} * 64
              : (options.attention_mode != Gemma4AttentionMode::kIndependent && segments >= 4 &&
                         segments <= 12
                     ? std::size_t{p.layers} * 64 *
                           (options.attention_mode == Gemma4AttentionMode::kOwners
                                ? (segments + 3) / 4
                                : segments / 4)
                     : 0));
}

std::expected<Gemma4Graph, KernelFailure> BuildGemma4Graph(TensorArena& arena,
                                                           const md::Gemma4Profile& p,
                                                           const md::Gemma4Binding& b,
                                                           const md::Gemma4StateLayout& state,
                                                           const Gemma4ChunkShape& shape,
                                                           const Gemma4GraphOptions& o) {
  if (auto checked = Check(p, b, state, shape, o); !checked)
    return std::unexpected(checked.error());
  if (auto room = arena.Reserve(Gemma4GraphTensors(p, shape.segments.size(), o)); !room) {
    return std::unexpected(room.error());
  }
  auto* c = arena.context();
  Gemma4Graph g;
  g.profile = p;
  g.binding = b;
  g.context = state.context;
  g.max_rows = state.max_rows;
  g.global_capacity = state.global_cells;
  g.local_capacity = state.local_cells;
  g.state_bytes = state.bytes;
  g.shape = shape;
  g.options = o;
  std::int64_t rows = 0;
  for (const auto& s : shape.segments) rows += s.rows;
  if (o.hidden_input)
    g.input_hidden = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.width, rows);
  else
    g.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
  g.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, rows);
  if (o.head && shape.output_mode == Gemma4OutputMode::kHead)
    g.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, shape.outputs);
  if (shape.feature_outputs != 0)
    g.feature_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, shape.feature_outputs);
  g.inputs = {o.hidden_input ? g.input_hidden : g.tokens, g.positions};
  if (g.out_ids != nullptr) g.inputs.push_back(g.out_ids);
  if (g.feature_ids != nullptr) g.inputs.push_back(g.feature_ids);
  std::uint32_t first = 0;
  for (const auto& s : shape.segments) {
    Gemma4SegmentTensors seg;
    seg.shape = s;
    seg.first_row = first;
    first += s.rows;
    seg.global_cells = ggml_new_tensor_1d(c, GGML_TYPE_I64, s.rows);
    seg.local_cells = ggml_new_tensor_1d(c, GGML_TYPE_I64, s.rows);
    if (o.device_masks) {
      seg.global_mask = Gemma4Mask(
          c, g.positions, s.global_n_kv, static_cast<std::int32_t>(seg.first_row),
          static_cast<std::int32_t>(s.rows), static_cast<std::int32_t>(state.global_cells), 0,
          static_cast<std::int32_t>(state.context));
      seg.local_mask = Gemma4Mask(
          c, g.positions, s.local_n_kv, static_cast<std::int32_t>(seg.first_row),
          static_cast<std::int32_t>(s.rows), static_cast<std::int32_t>(state.local_cells),
          static_cast<std::int32_t>(p.window), static_cast<std::int32_t>(state.context));
    } else {
      seg.global_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, s.global_n_kv,
                                           static_cast<std::int64_t>(Pad(s.rows, 32)));
      seg.local_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, s.local_n_kv,
                                          static_cast<std::int64_t>(Pad(s.rows, 32)));
      g.inputs.push_back(seg.global_mask);
      g.inputs.push_back(seg.local_mask);
    }
    seg.caches.resize(p.layers);
    g.inputs.push_back(seg.global_cells);
    g.inputs.push_back(seg.local_cells);
    g.segments.push_back(std::move(seg));
  }
  const auto weight = [&](const md::Gemma4Tensor& r) {
    const auto it = std::ranges::find_if(g.weights, [&](const Gemma4WeightLeaf& w) {
      return w.resource.index == r.index && w.resource.expert_array == r.expert_array;
    });
    if (it != g.weights.end()) return it->tensor;
    const auto* type = jitllm::artifact::FindGgmlType(r.type);
    std::array<std::int64_t, 4> ne{1, 1, 1, 1};
    for (std::size_t i = 0; i < r.ne.size(); ++i) ne[i] = static_cast<std::int64_t>(r.ne[i]);
    if (r.expert_array) ne[2] = p.experts;
    auto* t = ggml_new_tensor(c, static_cast<ggml_type>(type->id),
                              r.expert_array ? 3 : static_cast<int>(r.ne.size()), ne.data());
    if (r.expert_array) {
      const auto align = std::lcm(std::uint64_t{16}, std::uint64_t{type->block_bytes});
      t->nb[2] = o.expert_stride.empty() ? Pad(r.readable, align) : o.expert_stride[r.index];
      t->nb[3] = t->nb[2] * p.experts;
    }
    if (type->block_elements > 1 && r.ne[0] % jitllm::artifact::kGgmlRowPadding != 0) {
      MarkRowPaddingReadable(t);
    }
    g.weights.push_back({t, r});
    return t;
  };
  std::vector<ggml_tensor*> expanded;
  // Diagnostic partial-layer comparisons still produce both mask kinds,
  // matching the host path's staged global+local masks. Complete graphs
  // reuse these same two producers through every corresponding layer.
  if (o.device_masks)
    for (const auto& segment : g.segments)
      for (auto* mask : {segment.global_mask, segment.local_mask}) expanded.push_back(mask);
  const auto named = [&](std::string name, ggml_tensor* t) {
    ggml_set_name(t, name.c_str());
    g.named.emplace_back(std::move(name), t);
    return t;
  };
  std::unordered_map<ggml_tensor*, ggml_tensor*> preparations;
  const auto q8 = [&](ggml_tensor* x) {
    if (const auto at = preparations.find(x); at != preparations.end()) return at->second;
    auto* prepared = QuantizeQ8(c, x);
    preparations.emplace(x, prepared);
    return prepared;
  };
  const auto mm = [&](ggml_tensor* w, ggml_tensor* x) {
    if (!o.shared_q8 || !VecQType(w->type) || !ggml_is_contiguous(x) || x->ne[1] > kVecQMaxTokens ||
        x->ne[2] != 1 || x->ne[3] != 1) {
      return ggml_mul_mat(c, w, x);
    }
    auto* node = VecQ(c, w, q8(x), nullptr, x->ne[1], false);
    SetVecQOneToken(node);
    return node;
  };
  const auto mm_id = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* ids) {
    if (!o.shared_q8 || !VecQType(w->type) || !ggml_is_contiguous(x) || x->ne[2] > kVecQMaxTokens ||
        x->ne[2] * ids->ne[0] > 128) {
      return ggml_mul_mat_id(c, w, x, ids);
    }
    auto* node = VecQ(c, w, q8(x), ids, x->ne[2], x->ne[1] != 1);
    SetVecQOneToken(node);
    return node;
  };
  ggml_tensor* input = o.hidden_input
                           ? g.input_hidden
                           : ggml_scale(c, ggml_get_rows(c, weight(b.token_embd), g.tokens),
                                        std::sqrt(static_cast<float>(p.width)));
  const auto count = o.layer_count == 0 ? p.layers - o.first_layer : o.layer_count;
  for (std::uint32_t il = o.first_layer; il < o.first_layer + count; ++il) {
    const auto& l = b.layers[il];
    const auto prefix = std::format("blk.{}.", il);
    const auto d = p.head_dim(il), kvh = p.kv_heads(il), kvw = d * kvh;
    auto* attn_input = named(prefix + "attn_input", Norm(c, p, input, weight(l.attn_norm)));
    const bool tail = shape.output_mode != Gemma4OutputMode::kStateOnly || il + 1 != p.layers;
    auto* q = tail ? ggml_reshape_3d(c, mm(weight(l.q), attn_input), d, p.heads, rows) : nullptr;
    auto* k_raw = mm(weight(l.k), attn_input);
    auto* v_raw = l.tied_kv ? k_raw : mm(weight(l.v), attn_input);
    auto* k = ggml_reshape_3d(c, k_raw, d, kvh, rows);
    auto* v = ggml_reshape_3d(c, v_raw, d, kvh, rows);
    if (tail) q = Norm(c, p, q, weight(l.q_norm));
    k = Norm(c, p, k, weight(l.k_norm));
    v = named(prefix + "v_norm", ggml_rms_norm(c, v, p.rms_eps));
    const auto rope = [&](ggml_tensor* t, ggml_tensor* positions) {
      return ggml_rope_ext(c, t, positions, p.local(il) ? nullptr : weight(b.rope_freqs),
                           static_cast<int>(p.local(il) ? p.local_rope_dims : p.global_rope_dims),
                           GGML_ROPE_TYPE_NEOX, static_cast<int>(p.context),
                           p.local(il) ? p.local_rope_base : p.global_rope_base, 1.0f, 0.0f, 1.0f,
                           32.0f, 1.0f);
    };
    if (tail) q = named(prefix + "q_rope", rope(q, g.positions));
    if (!o.rope_store) k = named(prefix + "k_rope", rope(k, g.positions));
    // Upstream expands Q, V, K before stores. This also leaves K's rotation
    // next to its store, permitting the independently checked store fusion.
    if (tail) expanded.push_back(q);
    expanded.push_back(v);
    if (!o.rope_store) expanded.push_back(k);
    auto* k_rows = o.rope_store ? nullptr : ggml_reshape_2d(c, k, kvw, rows);
    auto* v_rows = ggml_reshape_2d(c, v, kvw, rows);
    ggml_tensor* joined = nullptr;
    for (auto& seg : g.segments) {
      const auto& s = seg.shape;
      const auto cells = p.local(il) ? state.local_cells : state.global_cells;
      const auto n_kv = p.local(il) ? s.local_n_kv : s.global_n_kv;
      auto* indices = p.local(il) ? seg.local_cells : seg.global_cells;
      auto* mask = p.local(il) ? seg.local_mask : seg.global_mask;
      auto* cache_k = ggml_new_tensor_2d(c, GGML_TYPE_F16, kvw, cells);
      auto* cache_v = ggml_new_tensor_2d(c, GGML_TYPE_F16, kvw, cells);
      seg.caches[il] = {cache_k, cache_v};
      const auto slice = [&](ggml_tensor* t) {
        return ggml_view_2d(c, t, kvw, s.rows, t->nb[1], std::size_t{seg.first_row} * t->nb[1]);
      };
      if (o.rope_store) {
        // Every independent store gets a complete packed RoPE output, with
        // its own fresh positions and a zero-offset flattening view. The
        // joined learned K normalization and global raw-K-as-V stay unchanged.
        auto* part = ggml_view_3d(c, k, d, kvh, s.rows, k->nb[1], k->nb[2],
                                  std::size_t{seg.first_row} * k->nb[2]);
        auto* positions =
            ggml_view_1d(c, g.positions, s.rows, std::size_t{seg.first_row} * sizeof(std::int32_t));
        auto* rotated =
            named(prefix + std::format("slot.{}.k_rope", s.slot), rope(part, positions));
        auto* flattened = ggml_view_2d(c, rotated, kvw, s.rows, rotated->nb[2], 0);
        // Inputs precede the exact consecutive RoPE->VIEW->SET_ROWS pattern.
        expanded.push_back(indices);
        expanded.push_back(ggml_set_rows(c, cache_k, flattened, indices));
      } else {
        expanded.push_back(ggml_set_rows(c, cache_k, slice(k_rows), indices));
      }
      expanded.push_back(ggml_set_rows(c, cache_v, slice(v_rows), indices));
      if (!tail) continue;
      const auto cache_view = [&](ggml_tensor* t) {
        return ggml_permute(
            c, ggml_view_3d(c, t, d, kvh, n_kv, std::size_t{d} * 2, std::size_t{kvw} * 2, 0), 0, 2,
            1, 3);
      };
      auto* qs = ggml_view_3d(c, q, d, p.heads, s.rows, q->nb[1], q->nb[2],
                              std::size_t{seg.first_row} * q->nb[2]);
      qs = ggml_permute(c, qs, 0, 2, 1, 3);
      auto* attn = ggml_flash_attn_ext(c, qs, cache_view(cache_k), cache_view(cache_v), mask, 1.0f,
                                       0.0f, 0.0f);
      ggml_prec_set_acc(attn, GGML_PREC_F32);
      attn = ggml_reshape_2d(c, attn, std::int64_t{d} * p.heads, s.rows);
      named(prefix + std::format("slot.{}.attention", s.slot), attn);
      expanded.push_back(attn);
      joined = joined == nullptr ? attn : ggml_concat(c, joined, attn, 1);
    }
    if (!tail) break;
    auto* projected = named(prefix + "attn_projection", mm(weight(l.out), joined));
    auto* residual = input;
    if (o.narrow_final && il + 1 == p.layers) {
      projected = ggml_get_rows(c, projected, g.out_ids);
      residual = ggml_get_rows(c, residual, g.out_ids);
      rows = shape.outputs;
    }
    auto* attn_out = named(prefix + "attn_residual",
                           ggml_add(c, Norm(c, p, projected, weight(l.attn_post_norm)), residual));
    auto* shared_in = Norm(c, p, attn_out, weight(l.ffn_norm));
    auto* up = mm(weight(l.up), shared_in);
    auto* gate = mm(weight(l.gate), shared_in);
    auto* shared = mm(weight(l.down), ggml_geglu_split(c, gate, up));
    ggml_tensor* ffn = shared;
    if (p.experts != 0) {
      shared = Norm(c, p, shared, weight(*l.ffn_post_norm_1));
      auto* routed_input = Norm(c, p, attn_out, weight(*l.ffn_pre_norm_2));
      auto* router_input = ggml_mul(c,
                                    ggml_scale(c, ggml_rms_norm(c, attn_out, p.rms_eps),
                                               1.0f / std::sqrt(static_cast<float>(p.width))),
                                    weight(*l.router_scale));
      auto* probs = named(prefix + "router_probabilities",
                          ggml_soft_max(c, mm(weight(*l.router), router_input)));
      auto* ids = named(prefix + "expert_ids",
                        ggml_argsort_top_k(c, probs, static_cast<int>(p.experts_used)));
      auto* weights = ggml_get_rows(c, ggml_reshape_3d(c, probs, 1, p.experts, rows), ids);
      weights = ggml_reshape_2d(c, weights, p.experts_used, rows);
      auto* sum = ggml_clamp(c, ggml_sum_rows(c, weights), 6.103515625e-5f,
                             std::numeric_limits<float>::infinity());
      weights = ggml_reshape_3d(c, ggml_div(c, weights, sum), 1, p.experts_used, rows);
      named(prefix + "expert_weights", weights);
      expanded.push_back(weights);
      auto* in = ggml_reshape_3d(c, routed_input, p.width, 1, rows);
      ggml_tensor* routed_gate = nullptr;
      ggml_tensor* routed_up = nullptr;
      if (l.gate_up_exps) {
        auto* pair = mm_id(weight(*l.gate_up_exps), in, ids);
        routed_gate =
            ggml_view_3d(c, pair, p.expert_ffn, p.experts_used, rows, pair->nb[1], pair->nb[2], 0);
        routed_up = ggml_view_3d(c, pair, p.expert_ffn, p.experts_used, rows, pair->nb[1],
                                 pair->nb[2], std::size_t{p.expert_ffn} * 4);
      } else {
        routed_up = mm_id(weight(*l.up_exps), in, ids);
        routed_gate = mm_id(weight(*l.gate_exps), in, ids);
      }
      auto* activation =
          named(prefix + "expert_activation", ggml_geglu_split(c, routed_gate, routed_up));
      auto* down = mm_id(weight(*l.down_exps), activation, ids);
      auto* scales = ggml_repeat_4d(c, ggml_reshape_3d(c, weight(*l.expert_scale), 1, p.experts, 1),
                                    1, p.experts, rows, 1);
      scales = ggml_get_rows(c, scales, ids);
      down = named(prefix + "expert_scaled_down", ggml_mul(c, down, scales));
      auto* experts = ggml_mul(c, down, weights);
      expanded.push_back(experts);
      std::vector<ggml_tensor*> selected;
      for (std::uint32_t i = 0; i < p.experts_used; ++i) {
        auto* one = ggml_view_2d(c, experts, p.width, rows, experts->nb[2],
                                 std::size_t{i} * experts->nb[1]);
        selected.push_back(one);
        expanded.push_back(one);
      }
      auto* routed = selected[0];
      for (std::uint32_t i = 1; i < p.experts_used; ++i) routed = ggml_add(c, routed, selected[i]);
      // Keep the complete ordered sum contiguous before independent shared-FFN
      // work. Its post-normalization remains beside the final residual chain.
      expanded.push_back(routed);
      routed = Norm(c, p, routed, weight(*l.ffn_post_norm_2));
      ffn = ggml_add(c, shared, routed);
    }
    ffn = Norm(c, p, ffn, weight(l.ffn_post_norm));
    input = ggml_add(c, ffn, attn_out);
    input = named(prefix + "output", ggml_mul(c, input, weight(*l.output_scale)));
  }
  g.hidden = shape.output_mode == Gemma4OutputMode::kStateOnly ? nullptr : input;
  if (o.head && g.hidden != nullptr) {
    auto* normalized = named("output_normalized", Norm(c, p, input, weight(b.output_norm)));
    if (g.feature_ids != nullptr) {
      g.normalized_features =
          named("normalized_features", ggml_get_rows(c, normalized, g.feature_ids));
      expanded.push_back(g.normalized_features);
    }
    if (!o.narrow_final) normalized = ggml_get_rows(c, normalized, g.out_ids);
    auto* logits = mm(weight(b.output), normalized);
    logits = ggml_scale(c, logits, 1.0f / p.final_softcap);
    logits = ggml_tanh(c, logits);
    g.logits = named("logits", ggml_scale(c, logits, p.final_softcap));
    expanded.push_back(g.logits);
  } else if (g.hidden != nullptr)
    expanded.push_back(g.hidden);
  auto ordered = GraphOrder(expanded, arena);
  if (!ordered) return std::unexpected(ordered.error());
  g.nodes = std::move(*ordered);
  if (auto transformed = TransformGemma4Attention(arena, g, o.attention_mode); !transformed)
    return std::unexpected(transformed.error());
  return g;
}
}  // namespace jitllm::kernels::ggml
