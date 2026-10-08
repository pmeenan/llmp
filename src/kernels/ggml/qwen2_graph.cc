// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/qwen2_graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen2.h"

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

ggml_type WeightType(const model::Qwen2Profile& profile) {
  if (profile.weight_type == "F32") {
    return GGML_TYPE_F32;
  }
  if (profile.weight_type == "BF16") {
    return GGML_TYPE_BF16;
  }
  return GGML_TYPE_F16;
}

// ggml_rope_ext as llm_graph_context calls it for Qwen2: NEOX over the
// whole head, the model's base and original context, no frequency scaling
// and no YaRN (ext_factor 0, attn_factor 1, beta_fast 32, beta_slow 1:
// llama_context's defaults when the checkpoint has no rope scaling).
ggml_tensor* Rope(ggml_context* c, const model::Qwen2Profile& p, ggml_tensor* x,
                  ggml_tensor* positions) {
  return ggml_rope_ext(c, x, positions, nullptr, static_cast<int>(p.head_dim), GGML_ROPE_TYPE_NEOX,
                       static_cast<int>(p.train_context), p.rope_base, 1.0f, 0.0f, 1.0f, 32.0f,
                       1.0f);
}

// build_norm with LLM_NORM_RMS and a weight.
ggml_tensor* Norm(ggml_context* c, const model::Qwen2Profile& p, ggml_tensor* x,
                  ggml_tensor* weight) {
  return ggml_mul(c, ggml_rms_norm(c, x, p.rms_eps), weight);
}

}  // namespace

std::size_t Qwen2GraphTensors(const model::Qwen2Profile& profile) {
  // Leaves: 6 inputs, 14 per layer, 2 more weights. Nodes: at most 48 per
  // layer, 3 for the head. Rounded up generously.
  return 64 + (std::size_t{profile.layers} * 64);
}

std::expected<Qwen2Graph, KernelFailure> BuildQwen2Graph(TensorArena& arena,
                                                         const model::Qwen2Profile& p,
                                                         const Qwen2ChunkShape& shape) {
  if (shape.rows <= 0 || shape.cells <= 0 || shape.n_kv < shape.rows || shape.n_kv > shape.cells) {
    return Rejected(std::format("no chunk of {} rows attends to {} of {} cells", shape.rows,
                                shape.n_kv, shape.cells));
  }
  if (p.layers == 0 || p.kv_heads == 0 || p.heads % p.kv_heads != 0 || p.head_dim == 0) {
    return Rejected("not a Qwen2 profile");
  }
  if (auto room = arena.Reserve(Qwen2GraphTensors(p)); !room) {
    return std::unexpected(room.error());
  }
  ggml_context* c = arena.context();
  const std::int64_t n = shape.rows;
  const std::int64_t width = p.width;
  const std::int64_t heads = p.heads;
  const std::int64_t kv_heads = p.kv_heads;
  const std::int64_t head = p.head_dim;
  const std::int64_t kv_width = p.kv_width();
  const std::int64_t cells = shape.cells;
  const std::int64_t n_kv = shape.n_kv;
  const ggml_type wt = WeightType(p);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head));

  Qwen2Graph g;
  g.embd = ggml_new_tensor_2d(c, GGML_TYPE_F32, width, n);
  g.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
  g.k_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, n);
  g.v_idxs = ggml_new_tensor_1d(c, GGML_TYPE_I64, n * kv_width);
  g.mask = ggml_new_tensor_4d(c, GGML_TYPE_F32, n_kv, n, 1, 1);
  g.out_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
  g.layers.resize(p.layers);
  for (Qwen2LayerTensors& l : g.layers) {
    l.attn_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, width);
    l.q = ggml_new_tensor_2d(c, wt, width, heads * head);
    l.q_bias = ggml_new_tensor_1d(c, GGML_TYPE_F32, heads * head);
    l.k = ggml_new_tensor_2d(c, wt, width, kv_width);
    l.k_bias = ggml_new_tensor_1d(c, GGML_TYPE_F32, kv_width);
    l.v = ggml_new_tensor_2d(c, wt, width, kv_width);
    l.v_bias = ggml_new_tensor_1d(c, GGML_TYPE_F32, kv_width);
    l.out = ggml_new_tensor_2d(c, wt, heads * head, width);
    l.ffn_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, width);
    l.gate = ggml_new_tensor_2d(c, wt, width, p.ffn);
    l.up = ggml_new_tensor_2d(c, wt, width, p.ffn);
    l.down = ggml_new_tensor_2d(c, wt, p.ffn, width);
    l.k_cache = ggml_new_tensor_3d(c, GGML_TYPE_F16, kv_width, cells, 1);
    l.v_cache = ggml_new_tensor_3d(c, GGML_TYPE_F16, kv_width, cells, 1);
  }
  g.output_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, width);
  g.output = ggml_new_tensor_2d(c, wt, width, p.vocab);

  // The tensors ggml_build_forward_expand is called with, in order. The
  // embedding input comes first in llama.cpp's graph, as a leaf here.
  std::vector<ggml_tensor*> expanded;
  ggml_tensor* layer_in = g.embd;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const Qwen2LayerTensors& l = g.layers[il];
    ggml_tensor* residual = layer_in;
    ggml_tensor* cur = Norm(c, p, layer_in, l.attn_norm);

    // build_qkv, separate Q/K/V with biases, then the RoPEs.
    ggml_tensor* q = ggml_add(c, ggml_mul_mat(c, l.q, cur), l.q_bias);
    ggml_tensor* k = ggml_add(c, ggml_mul_mat(c, l.k, cur), l.k_bias);
    ggml_tensor* v = ggml_add(c, ggml_mul_mat(c, l.v, cur), l.v_bias);
    q = ggml_reshape_3d(c, q, head, heads, n);
    k = ggml_reshape_3d(c, k, head, kv_heads, n);
    v = ggml_reshape_3d(c, v, head, kv_heads, n);
    q = Rope(c, p, q, g.positions);
    k = Rope(c, p, k, g.positions);

    // build_attn: Q, V, then K (so the RoPE of K can fuse with its store).
    expanded.push_back(q);
    expanded.push_back(v);
    expanded.push_back(k);
    // cpy_k: merge the heads, store rows at the K cells.
    ggml_tensor* k_rows = ggml_view_2d(c, k, kv_width, n, k->nb[2], 0);
    expanded.push_back(ggml_set_rows(c, l.k_cache, k_rows, g.k_idxs));
    // cpy_v, transposed: every element is a row of one.
    ggml_tensor* v_rows = ggml_row_size(v->type, kv_width) == v->nb[2]
                              ? ggml_reshape_2d(c, v, kv_width, n)
                              : ggml_cont_2d(c, v, kv_width, n);
    ggml_tensor* v_cells = ggml_reshape_2d(c, l.v_cache, 1, ggml_nelements(l.v_cache));
    v_rows = ggml_reshape_2d(c, v_rows, 1, ggml_nelements(v_rows));
    expanded.push_back(ggml_set_rows(c, v_cells, v_rows, g.v_idxs));
    // get_k and get_v over the first n_kv cells.
    ggml_tensor* keys =
        ggml_view_4d(c, l.k_cache, head, kv_heads, n_kv, 1, ggml_row_size(l.k_cache->type, head),
                     ggml_row_size(l.k_cache->type, kv_width),
                     ggml_row_size(l.k_cache->type, kv_width * cells), 0);
    ggml_tensor* values = ggml_view_4d(
        c, l.v_cache, n_kv, kv_heads, head, 1, ggml_row_size(l.v_cache->type, cells * head),
        ggml_row_size(l.v_cache->type, cells), ggml_row_size(l.v_cache->type, cells * kv_width), 0);
    // build_attn_mha without flash attention, one stream.
    ggml_tensor* queries =
        ggml_view_4d(c, q, q->ne[0], q->ne[1], q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], 0);
    queries = ggml_permute(c, queries, 0, 2, 1, 3);
    keys = ggml_permute(c, keys, 0, 2, 1, 3);
    values = ggml_permute(c, values, 0, 2, 1, 3);
    ggml_tensor* kq = ggml_mul_mat(c, keys, queries);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    kq = ggml_soft_max_ext(c, kq, g.mask, scale, 0.0f);
    ggml_soft_max_add_sinks(kq, nullptr);
    ggml_tensor* kqv = ggml_mul_mat(c, values, kq);
    cur = ggml_permute(c, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(c, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    expanded.push_back(cur);
    cur = ggml_mul_mat(c, l.out, cur);

    if (il + 1 == p.layers) {
      cur = ggml_get_rows(c, cur, g.out_ids);
      residual = ggml_get_rows(c, residual, g.out_ids);
    }
    ggml_tensor* ffn_in = ggml_add(c, cur, residual);
    cur = Norm(c, p, ffn_in, l.ffn_norm);
    // build_ffn: up, then gate (parallel), SwiGLU, down.
    ggml_tensor* up = ggml_mul_mat(c, l.up, cur);
    ggml_tensor* gate = ggml_mul_mat(c, l.gate, cur);
    cur = ggml_swiglu_split(c, gate, up);
    cur = ggml_mul_mat(c, l.down, cur);
    layer_in = ggml_add(c, cur, ffn_in);
  }
  ggml_tensor* cur = Norm(c, p, layer_in, g.output_norm);
  g.logits = ggml_mul_mat(c, g.output, cur);
  expanded.push_back(g.logits);
  auto ordered = GraphOrder(expanded, arena);
  if (!ordered) return std::unexpected(ordered.error());
  g.nodes = std::move(*ordered);
  return g;
}

}  // namespace llmp::kernels::ggml
