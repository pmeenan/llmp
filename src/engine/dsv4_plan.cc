// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <string_view>
#include <utility>

#include "engine/support.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
using support::Error;

}  // namespace

void BindDsv4Weights(const Dsv4Model& m, kg::Dsv4Graph& g) {
  const auto bind = [&](ggml_tensor* t, const md::Dsv4Tensor& r) {
    if (t != nullptr) {
      kg::TensorArena::Bind(t, m.places.resource(r.index));
    }
  };
  const auto& b = *m.binding;
  bind(g.output_norm, b.output_norm);
  bind(g.output, b.output);
  bind(g.hc_head_fn, b.hc_head_fn);
  bind(g.hc_head_base, b.hc_head_base);
  bind(g.hc_head_scale, b.hc_head_scale);
  using K = md::Dsv4StateTensor::Kind;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Dsv4Layer& r = b.layers[il];
    kg::Dsv4LayerTensors& l = g.layers[il];
    bind(l.attn_norm, r.attn_norm);
    bind(l.attn_sinks, r.attn_sinks);
    bind(l.q_a, r.q_a);
    bind(l.q_a_norm, r.q_a_norm);
    bind(l.q_b, r.q_b);
    bind(l.kv, r.kv);
    bind(l.kv_norm, r.kv_norm);
    bind(l.out_a, r.out_a);
    bind(l.out_b, r.out_b);
    bind(l.hc_attn_fn, r.hc_attn_fn);
    bind(l.hc_attn_base, r.hc_attn_base);
    bind(l.hc_attn_scale, r.hc_attn_scale);
    bind(l.hc_ffn_fn, r.hc_ffn_fn);
    bind(l.hc_ffn_base, r.hc_ffn_base);
    bind(l.hc_ffn_scale, r.hc_ffn_scale);
    bind(l.comp_kv, r.comp_kv);
    bind(l.comp_gate, r.comp_gate);
    bind(l.comp_ape, r.comp_ape);
    bind(l.comp_norm, r.comp_norm);
    bind(l.idx_q_b, r.idx_q_b);
    bind(l.idx_proj, r.idx_proj);
    bind(l.idx_comp_kv, r.idx_comp_kv);
    bind(l.idx_comp_gate, r.idx_comp_gate);
    bind(l.idx_comp_ape, r.idx_comp_ape);
    bind(l.idx_comp_norm, r.idx_comp_norm);
    bind(l.ffn_norm, r.ffn_norm);
    bind(l.router, r.router);
    bind(l.router_bias, r.router_bias);
    bind(l.tid2eid, r.tid2eid);
    bind(l.up_shexp, r.up_shexp);
    bind(l.gate_shexp, r.gate_shexp);
    bind(l.down_shexp, r.down_shexp);
    kg::TensorArena::Bind(l.up_exps, m.places.array(r.up_exps.index));
    kg::TensorArena::Bind(l.gate_exps, m.places.array(r.gate_exps.index));
    kg::TensorArena::Bind(l.down_exps, m.places.array(r.down_exps.index));
    const auto state = [&](ggml_tensor* t, K kind) {
      // A view (a ring's compressed cache in its window's tensor) is
      // bound with the tensor it views.
      if (t == nullptr || t->view_src != nullptr) {
        return;
      }
      const std::int64_t i = m.state->Find(il, kind);
      kg::TensorArena::Bind(t,
                            m.places.state + m.state->tensors[static_cast<std::size_t>(i)].offset);
    };
    state(l.raw_k, K::kRawK);
    state(l.csa_k, K::kCsaK);
    state(l.csa_state_kv, K::kCsaStateKv);
    state(l.csa_state_score, K::kCsaStateScore);
    state(l.lid_k, K::kLidK);
    state(l.lid_state_kv, K::kLidStateKv);
    state(l.lid_state_score, K::kLidStateScore);
    state(l.hca_k, K::kHcaK);
    state(l.hca_state_kv, K::kHcaStateKv);
    state(l.hca_state_score, K::kHcaStateScore);
  }
}

std::expected<std::unique_ptr<Dsv4Planned>, std::string> PlanDsv4Chunk(
    const Dsv4Model& m, const kg::Dsv4ChunkShape& shape, const kg::DeviceChoices& choices,
    std::span<const std::string> keep_names, std::uint64_t activations,
    std::uint64_t activation_bytes, const Dsv4Speculation& speculation) {
  if ((m.exact || speculation.verify || !keep_names.empty()) && shape.outputs != 0 &&
      shape.outputs != shape.rows) {
    return Error("a reference chunk, verify or named diagnostic needs every row's head");
  }
  auto out = std::make_unique<Dsv4Planned>();
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(*m.profile));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out->arena.emplace(std::move(*arena));
  kg::Dsv4GraphOptions options{.expert_stride = m.places.stride,
                               .row_invariant = speculation.verify && m.exact,
                               .fused = !m.exact};
  if (const DsparkModel* d = speculation.drafter; d != nullptr) {
    options.features = d->profile->target_layers;
    options.inject = kg::Dsv4Injection{.profile = d->profile,
                                       .binding = d->binding,
                                       .rows = speculation.inject_rows,
                                       .ring = d->state->ring};
  }
  auto graph = kg::BuildDsv4Graph(*out->arena, *m.profile, *m.binding, shape, options);
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Dsv4Graph& g = out->graph;
  BindDsv4Weights(m, g);
  if (speculation.drafter != nullptr) {
    BindDsparkInjection(*speculation.drafter, g);
  }
  // The logits are read after the run whatever node comes last.
  std::vector<ggml_tensor*> keep = {g.logits};
  for (const std::string& name : keep_names) {
    if (name == "*") {
      for (const auto& [n, t] : g.named) {
        keep.push_back(t);
      }
    } else if (ggml_tensor* t = g.Named(name); t != nullptr) {
      keep.push_back(t);
    } else {
      return Error(std::format("the graph names no {}", name));
    }
  }
  kg::DeviceChoices device = choices;
  device.row_invariant = speculation.verify && m.exact;
  device.fuse_norms = !m.exact;
  device.vector_floats = !m.exact;
  device.pair_experts = !m.exact;
  device.compact_experts = !m.exact && shape.rows >= kg::kDsv4CompactMinRows;
  device.wide_sparse_attention = !m.exact;
  const auto inputs = g.inputs();
  if (auto placed =
          PlaceAndPlan(*out, g.nodes, inputs, keep, device, activations, activation_bytes);
      !placed) {
    return std::unexpected(placed.error());
  }
  return out;
}

void BindDsparkInjection(const DsparkModel& d, kg::Dsv4Graph& g) {
  if (!g.inject) {
    return;
  }
  const md::DsparkBinding& b = *d.binding;
  kg::DsparkInjectTensors& t = *g.inject;
  kg::TensorArena::Bind(t.fc, d.places.resource(b.fc.index));
  kg::TensorArena::Bind(t.enc_norm, d.places.resource(b.enc_norm.index));
  for (std::size_t il = 0; il < t.kv.size(); ++il) {
    kg::TensorArena::Bind(t.kv[il], d.places.resource(b.blocks.layers[il].kv.index));
    kg::TensorArena::Bind(t.kv_norm[il], d.places.resource(b.blocks.layers[il].kv_norm.index));
    kg::TensorArena::Bind(t.ring[il], d.places.state + d.state->offsets[il]);
  }
}

std::expected<std::unique_ptr<DsparkPlanned>, std::string> PlanDsparkDraft(
    const DsparkModel& d, std::int64_t rows, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes) {
  auto out = std::make_unique<DsparkPlanned>();
  auto arena = kg::TensorArena::Create(kg::DsparkGraphTensors(*d.profile, rows));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildDsparkGraph(*out->arena, *d.profile, *d.binding, rows, d.state->ring,
                                    {.expert_stride = d.places.stride, .fused = !d.exact});
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::DsparkGraph& g = out->graph;
  // The drafter's blocks at its places, its head's output the target's.
  const md::DsparkBinding& b = *d.binding;
  const auto bind = [](ggml_tensor* t, std::uint64_t address) {
    kg::TensorArena::Bind(t, address);
  };
  for (std::uint32_t il = 0; il < d.profile->blocks.layers; ++il) {
    const md::Dsv4Layer& r = b.blocks.layers[il];
    kg::Dsv4LayerTensors& l = g.core.layers[il];
    for (const auto& [t, w] : std::initializer_list<std::pair<ggml_tensor*, const md::Dsv4Tensor*>>{
             {l.attn_norm, &r.attn_norm},
             {l.attn_sinks, &r.attn_sinks},
             {l.q_a, &r.q_a},
             {l.q_a_norm, &r.q_a_norm},
             {l.q_b, &r.q_b},
             {l.kv, &r.kv},
             {l.kv_norm, &r.kv_norm},
             {l.out_a, &r.out_a},
             {l.out_b, &r.out_b},
             {l.hc_attn_fn, &r.hc_attn_fn},
             {l.hc_attn_base, &r.hc_attn_base},
             {l.hc_attn_scale, &r.hc_attn_scale},
             {l.hc_ffn_fn, &r.hc_ffn_fn},
             {l.hc_ffn_base, &r.hc_ffn_base},
             {l.hc_ffn_scale, &r.hc_ffn_scale},
             {l.ffn_norm, &r.ffn_norm},
             {l.router, &r.router},
             {l.router_bias, &r.router_bias},
             {l.up_shexp, &r.up_shexp},
             {l.gate_shexp, &r.gate_shexp},
             {l.down_shexp, &r.down_shexp}}) {
      bind(t, d.places.resource(w->index));
    }
    bind(l.up_exps, d.places.array(r.up_exps.index));
    bind(l.gate_exps, d.places.array(r.gate_exps.index));
    bind(l.down_exps, d.places.array(r.down_exps.index));
    bind(l.raw_k, d.places.state + d.state->offsets[il]);
  }
  bind(g.core.output_norm, d.places.resource(b.blocks.output_norm.index));
  bind(g.core.hc_head_fn, d.places.resource(b.blocks.hc_head_fn.index));
  bind(g.core.hc_head_base, d.places.resource(b.blocks.hc_head_base.index));
  bind(g.core.hc_head_scale, d.places.resource(b.blocks.hc_head_scale.index));
  bind(g.core.output, d.target_resource(b.blocks.output.index));
  bind(g.markov_w1, d.places.resource(b.markov_w1.index));
  bind(g.markov_w2, d.places.resource(b.markov_w2.index));
  const std::vector<ggml_tensor*> keep = {g.logits, g.drafts};
  const auto inputs = g.inputs();
  kg::DeviceChoices device = choices;
  device.fuse_norms = !d.exact;
  device.vector_floats = !d.exact;
  device.pair_experts = !d.exact;
  device.wide_sparse_attention = !d.exact;
  if (auto placed =
          PlaceAndPlan(*out, g.core.nodes, inputs, keep, device, activations, activation_bytes);
      !placed) {
    return std::unexpected(placed.error());
  }
  return out;
}

std::expected<void, std::string> Dsv4EmbeddingRows(const Dsv4Model& m,
                                                   std::span<const std::int32_t> tokens,
                                                   std::span<const std::byte> table,
                                                   std::vector<float>& embd) {
  const md::Dsv4Tensor& embedding = m.binding->token_embd;
  auto type = kg::GgmlTypeOf(embedding.type);
  if (!type) {
    return Error(type.error().detail);
  }
  const auto* traits = ggml_get_type_traits(*type);
  const std::uint64_t row_bytes = ggml_row_size(*type, m.profile->width);
  const std::uint64_t table_offset = m.artifact->resources()[embedding.index].offset.value();
  embd.assign(tokens.size() * m.profile->width, 0.0F);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] < 0 || std::cmp_greater_equal(tokens[i], m.profile->vocab)) {
      return Error(std::format("token {} is outside the vocabulary", tokens[i]));
    }
    const std::uint64_t at = table_offset + (static_cast<std::uint64_t>(tokens[i]) * row_bytes);
    if (at + row_bytes > table.size()) {
      return Error("the token table is shorter than its rows");
    }
    traits->to_float(table.data() + at, embd.data() + (i * m.profile->width), m.profile->width);
  }
  return {};
}

std::expected<void, std::string> BuildDsparkInputs(const Dsv4Model& m, const kg::DsparkGraph& g,
                                                   const md::DsparkBlockInputs& in,
                                                   std::span<const std::byte> table,
                                                   Dsv4HostInputs& out) {
  if (auto rows = Dsv4EmbeddingRows(m, in.tokens, table, out.embd); !rows) {
    return rows;
  }
  out.tokens = in.tokens;
  out.sources = {{g.core.embd, out.embd.data()},
                 {g.tokens, out.tokens.data()},
                 {g.core.positions, in.positions.data()},
                 {g.core.raw_k_idxs, in.cells.data()},
                 {g.core.raw_mask, in.mask.data()}};
  return {};
}

std::expected<void, std::string> BuildDsv4Inputs(const Dsv4Model& m, const kg::Dsv4Graph& g,
                                                 const md::Dsv4ChunkInputs& in,
                                                 std::span<const std::int32_t> tokens,
                                                 std::span<const std::byte> table,
                                                 Dsv4HostInputs& out,
                                                 std::span<const std::int64_t> inject_cells) {
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  if (auto embedded = Dsv4EmbeddingRows(m, tokens, table, out.embd); !embedded) {
    return embedded;
  }
  out.tokens.assign(tokens.begin(), tokens.end());
  const auto outputs = static_cast<std::uint32_t>(g.out_ids->ne[0]);
  if (outputs == 0 || outputs > rows) {
    return Error("the head's output rows do not fit the chunk");
  }
  out.out_ids.resize(outputs);
  std::ranges::iota(out.out_ids, static_cast<std::int32_t>(rows - outputs));
  // A ring's graph masks its compressed rows on the device from the rows'
  // visible counts (dsv4_graph.h Dsv4ChunkShape::ring): no CSA or indexer
  // mask, and the counts in the zero fill's place.
  if (g.top_k_zeros != nullptr) {
    out.zeros.assign(static_cast<std::size_t>(ggml_nelements(g.top_k_zeros)), 0);
  } else if (in.csa.n_visible.size() != rows || in.hca.n_visible.size() != rows) {
    return Error("the chunk's visible counts are not its rows'");
  }
  const auto comp =
      [&](const kg::Dsv4CompInputs& t, const md::Dsv4CompPlan& plan,
          const std::vector<std::uint16_t>& mask) -> std::expected<void, std::string> {
    out.sources.insert(out.sources.end(), {{t.state_pos, plan.state_pos.data()},
                                           {t.persist_src, plan.persist_src.data()},
                                           {t.persist_dst, plan.persist_dst.data()},
                                           {t.read_idxs, plan.read_idxs.data()},
                                           {t.write_idxs, plan.write_idxs.data()},
                                           {t.write_pos, plan.write_pos.data()}});
    if (t.mask != nullptr) {
      if (std::cmp_not_equal(mask.size(), ggml_nelements(t.mask))) {
        return Error("a compressor's mask is not its graph's");
      }
      out.sources.emplace_back(t.mask, mask.data());
    }
    return {};
  };
  out.sources = {{g.embd, out.embd.data()},          {g.tokens, out.tokens.data()},
                 {g.positions, in.positions.data()}, {g.raw_k_idxs, in.raw_cells.data()},
                 {g.raw_mask, in.raw_mask.data()},   {g.out_ids, out.out_ids.data()}};
  if (auto added = comp(g.csa, in.csa, in.csa_mask); !added) {
    return added;
  }
  if (auto added = comp(g.hca, in.hca, in.hca_mask); !added) {
    return added;
  }
  if (auto added = comp(g.lid, in.lid, in.lid_mask); !added) {
    return added;
  }
  out.sources.emplace_back(g.lid_rot, m.rot.data());
  if (g.top_k_zeros != nullptr) {
    out.sources.emplace_back(g.top_k_zeros, out.zeros.data());
  } else {
    out.sources.emplace_back(g.csa_visible, in.csa.n_visible.data());
    out.sources.emplace_back(g.hca_visible, in.hca.n_visible.data());
  }
  if (g.inject) {
    if (std::cmp_not_equal(inject_cells.size(), g.inject->cells->ne[0])) {
      return Error("the injection's cells are not its rows'");
    }
    out.sources.emplace_back(g.inject->cells, inject_cells.data());
  }
  return {};
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

}  // namespace jitllm::engine
