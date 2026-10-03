// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen38_plan.h"

#include <algorithm>
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

void BindQwen38Weights(const Qwen38Model& m, kg::Qwen38Graph& g) {
  const auto bind = [&](ggml_tensor* t, const md::Qwen38Tensor& r) {
    if (t != nullptr) {
      kg::TensorArena::Bind(t, m.places.resource(r.index));
    }
  };
  const auto mx = [&](const kg::Qwen38Mxfp8Tensors& t, const md::Qwen38Mxfp8& r) {
    bind(t.codes, r.codes);
    bind(t.scales, r.scales);
    bind(t.matrix, r.matrix);  // a GGUF checkpoint's
  };
  const auto& b = *m.binding;
  bind(g.token_embd, b.token_embd);
  kg::TensorArena::Bind(g.ple_table, m.places.ple_table);
  bind(g.ple_table_scale, b.ple_table_scale);
  bind(g.output, b.output);
  bind(g.output_hc_norm, b.output_hc_norm);
  bind(g.output_hc_down, b.output_hc_down);
  bind(g.output_hc_up, b.output_hc_up);
  using K = md::Qwen38StateTensor::Kind;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Qwen38Layer& r = b.layers[il];
    kg::Qwen38LayerTensors& l = g.layers[il];
    bind(l.hc_attn_norm, r.hc_attn_norm);
    bind(l.hc_attn_down, r.hc_attn_down);
    bind(l.hc_attn_up, r.hc_attn_up);
    bind(l.hc_attn_inject, r.hc_attn_inject);
    bind(l.hc_ffn_norm, r.hc_ffn_norm);
    bind(l.hc_ffn_down, r.hc_ffn_down);
    bind(l.hc_ffn_up, r.hc_ffn_up);
    bind(l.hc_ffn_inject, r.hc_ffn_inject);
    if (r.linear) {
      mx(l.qkv, r.qkv);
      mx(l.z, r.z);
      mx(l.beta, r.beta);
      mx(l.alpha, r.alpha);
      mx(l.ssm_out, r.ssm_out);
      bind(l.dt_bias, r.dt_bias);
      bind(l.ssm_a, r.ssm_a);
      bind(l.conv1d, r.conv1d);
      bind(l.ssm_norm, r.ssm_norm);
    } else {
      mx(l.q, r.q);
      mx(l.k, r.k);
      mx(l.v, r.v);
      mx(l.o, r.o);
      mx(l.idx_qk, r.idx_qk);
      mx(l.idx_q, r.idx_q);
      mx(l.idx_k, r.idx_k);
      bind(l.q_norm, r.q_norm);
      bind(l.k_norm, r.k_norm);
      bind(l.idx_q_norm, r.idx_q_norm);
      bind(l.idx_k_norm, r.idx_k_norm);
    }
    if (il == m.profile->ple_layer) {
      bind(l.ple_key, r.ple_key);
      bind(l.ple_value, r.ple_value);
      bind(l.ple_norm_key, r.ple_norm_key);
      bind(l.ple_norm_query, r.ple_norm_query);
      bind(l.ple_norm_conv, r.ple_norm_conv);
      bind(l.ple_conv1d, r.ple_conv1d);
    }
    bind(l.router, r.router);
    bind(l.shared_gate, r.shared_gate);
    mx(l.gate_shexp, r.gate_shexp);
    mx(l.up_shexp, r.up_shexp);
    mx(l.down_shexp, r.down_shexp);
    bind(l.gate_exps_scale, r.gate_exps_scale);
    bind(l.up_exps_scale, r.up_exps_scale);
    bind(l.down_exps_scale, r.down_exps_scale);
    if (l.experts != nullptr) {
      // The CUTLASS layout fills each slot from its start: the layer's first
      // array's address less its offset in the expert group.
      const md::Qwen38Tensor& first = b.cutlass() ? r.gate_up_codes : r.gate_exps;
      kg::TensorArena::Bind(l.experts, m.places.array(first.index) - first.group_offset);
    } else {
      kg::TensorArena::Bind(l.gate_exps, m.places.array(r.gate_exps.index));
      kg::TensorArena::Bind(l.up_exps, m.places.array(r.up_exps.index));
      kg::TensorArena::Bind(l.down_exps, m.places.array(r.down_exps.index));
    }
    const auto state = [&](ggml_tensor* t, K kind) {
      if (t == nullptr) {
        return;
      }
      const std::int64_t i = m.state->Find(il, kind);
      kg::TensorArena::Bind(t,
                            m.places.state + m.state->tensors[static_cast<std::size_t>(i)].offset);
    };
    state(l.cache_k, K::kK);
    state(l.cache_v, K::kV);
    state(l.cache_idx, K::kIndexerK);
    state(l.cache_pool, K::kIndexerBlocks);
    state(l.conv_state, K::kConv);
    state(l.recurrent, K::kRecurrent);
    state(l.ple_state, K::kPleConv);
    // A verify's saves, at the commit layout's offsets.
    if (m.commit != nullptr) {
      const auto at = std::ranges::find(m.commit->layers, il);
      if (at != m.commit->layers.end()) {
        const auto i = static_cast<std::size_t>(at - m.commit->layers.begin());
        for (const auto& [t, offset] : {std::pair{l.commit_conv, m.commit->conv_out(i)},
                                        std::pair{l.commit_qkv, m.commit->qkv(i)},
                                        std::pair{l.commit_gate, m.commit->gate(i)},
                                        std::pair{l.commit_beta, m.commit->beta(i)}}) {
          if (t != nullptr) {
            kg::TensorArena::Bind(t, m.places.commit + offset);
          }
        }
      }
      if (l.commit_ple != nullptr) {
        kg::TensorArena::Bind(l.commit_ple, m.places.commit + m.commit->ple());
      }
    }
  }
  if (g.streams != nullptr && m.mtp_state != nullptr) {
    kg::TensorArena::Bind(g.streams, m.places.mtp_state + m.mtp_state->hidden);
  }
}

namespace {

// The drafter's weights at its places, its caches and streams in its state,
// and the target's table and head at theirs.
void BindQwen38MtpWeights(const Qwen38Model& m, kg::Qwen38MtpGraph& g) {
  const md::Qwen38MtpBinding& d = *m.drafter;
  const auto bind = [&](ggml_tensor* t, const md::Qwen38Tensor& r) {
    if (t != nullptr) {
      kg::TensorArena::Bind(t, m.places.mtp_resource(r.index));
    }
  };
  const auto mx = [&](const kg::Qwen38Mxfp8Tensors& t, const md::Qwen38Mxfp8& r) {
    bind(t.bf16, r.bf16);
  };
  kg::TensorArena::Bind(g.token_embd, m.places.resource(m.binding->token_embd.index));
  if (d.selected_head()) {
    bind(g.output, d.draft_output);
    bind(g.draft_ids, d.draft_ids);
  } else {
    kg::TensorArena::Bind(g.output, m.places.resource(m.binding->output.index));
  }
  bind(g.fc_embd, d.fc_embd);
  bind(g.fc_hidden, d.fc_hidden);
  bind(g.norm_embd, d.norm_embd);
  bind(g.norm_hidden, d.norm_hidden);
  bind(g.output_hc_norm, d.output_hc_norm);
  bind(g.output_hc_down, d.output_hc_down);
  bind(g.output_hc_up, d.output_hc_up);
  const md::Qwen38Layer& r = d.layer;
  kg::Qwen38LayerTensors& l = g.layer;
  bind(l.hc_attn_norm, r.hc_attn_norm);
  bind(l.hc_attn_down, r.hc_attn_down);
  bind(l.hc_attn_up, r.hc_attn_up);
  bind(l.hc_attn_inject, r.hc_attn_inject);
  bind(l.hc_ffn_norm, r.hc_ffn_norm);
  bind(l.hc_ffn_down, r.hc_ffn_down);
  bind(l.hc_ffn_up, r.hc_ffn_up);
  bind(l.hc_ffn_inject, r.hc_ffn_inject);
  mx(l.q, r.q);
  mx(l.k, r.k);
  mx(l.v, r.v);
  mx(l.o, r.o);
  mx(l.idx_qk, r.idx_qk);
  bind(l.q_norm, r.q_norm);
  bind(l.k_norm, r.k_norm);
  bind(l.idx_q_norm, r.idx_q_norm);
  bind(l.idx_k_norm, r.idx_k_norm);
  bind(l.router, r.router);
  bind(l.shared_gate, r.shared_gate);
  mx(l.gate_shexp, r.gate_shexp);
  mx(l.up_shexp, r.up_shexp);
  mx(l.down_shexp, r.down_shexp);
  bind(l.gate_exps_scale, r.gate_exps_scale);
  bind(l.up_exps_scale, r.up_exps_scale);
  bind(l.down_exps_scale, r.down_exps_scale);
  kg::TensorArena::Bind(l.experts,
                        m.places.mtp_array(r.gate_up_codes.index) - r.gate_up_codes.group_offset);
  const md::Qwen38MtpState& s = *m.mtp_state;
  kg::TensorArena::Bind(l.cache_k, m.places.mtp_state + s.k);
  kg::TensorArena::Bind(l.cache_v, m.places.mtp_state + s.v);
  kg::TensorArena::Bind(l.cache_idx, m.places.mtp_state + s.indexer);
  kg::TensorArena::Bind(l.cache_pool, m.places.mtp_state + s.blocks);
  kg::TensorArena::Bind(g.streams, m.places.mtp_state + s.hidden);
}

}  // namespace

std::expected<std::unique_ptr<Qwen38Planned>, std::string> PlanQwen38Chunk(
    const Qwen38Model& m, const kg::Qwen38ChunkShape& shape, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes, std::span<const std::string> keep,
    Qwen38ChunkKind kind) {
  if ((kind.verify && m.commit == nullptr) || (kind.export_streams && m.mtp_state == nullptr)) {
    return Error("a verify or the drafter's streams without the drafter");
  }
  // The saves are bound at the commit layout's parts, sized for its rows;
  // the streams go to rows 1 .. rows of the drafter's state.
  if ((kind.verify && shape.rows > std::int64_t{m.commit->rows}) ||
      (kind.export_streams && shape.rows + 1 > std::int64_t{m.mtp_state->hidden_rows})) {
    return Error(std::format("a chunk of {} rows past the verify's saves or the drafter's streams",
                             shape.rows));
  }
  auto out = std::make_unique<Qwen38Planned>();
  const kg::Qwen38GraphOptions options{
      .expert_stride = m.places.stride,
      .fused = m.fused,
      .exact = m.exact,
      .experts = m.cutlass ? kg::Qwen38GraphOptions::Experts::kCutlass
                           : kg::Qwen38GraphOptions::Experts::kGgml,
      .verify = kind.verify,
      .export_streams = kind.export_streams,
      .stream_rows = m.mtp_state != nullptr ? m.mtp_state->hidden_rows : 0,
      .capture_routed = kind.capture_routed};
  auto arena = SizedArena(kg::Qwen38GraphTensors(*m.profile), [&](kg::TensorArena& a) {
    return kg::BuildQwen38Graph(a, *m.profile, *m.binding, shape, options).has_value();
  });
  if (!arena) {
    return std::unexpected(arena.error());
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildQwen38Graph(*out->arena, *m.profile, *m.binding, shape, options);
  out->arena->Seal();
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Qwen38Graph& g = out->graph;
  BindQwen38Weights(m, g);
  std::vector<ggml_tensor*> kept;
  for (const std::string& name : keep) {
    ggml_tensor* t = g.Named(name);
    if (t == nullptr) {
      return Error(std::format("the graph names no {}", name));
    }
    kept.push_back(t);
  }
  if (g.argmax != nullptr) {
    kept.push_back(g.logits);  // copied out after the argmaxes are computed
  }
  for (const auto& layer : g.routed) {
    for (auto* t : {layer.input, layer.activation, layer.down, layer.shared, layer.gate,
                    layer.weights, layer.ids, layer.combined}) {
      kept.push_back(t);
    }
    for (auto* t : {layer.attention_input, layer.attention_projection}) {
      if (t != nullptr) {
        kept.push_back(t);
      }
    }
  }
  if (auto r =
          PlaceAndPlan(*out, g.nodes, g.inputs(), kept, choices, activations, activation_bytes);
      !r) {
    return std::unexpected(r.error());
  }
  return out;
}

std::expected<std::unique_ptr<Qwen38MtpPlanned>, std::string> PlanQwen38Mtp(
    const Qwen38Model& m, const kg::Qwen38MtpShape& shape, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes) {
  if (m.drafter == nullptr || m.mtp_state == nullptr) {
    return Error("no MTP drafter");
  }
  auto out = std::make_unique<Qwen38MtpPlanned>();
  auto arena =
      SizedArena(kg::Qwen38MtpGraphTensors(*m.profile, shape.passes), [&](kg::TensorArena& a) {
        return kg::BuildQwen38MtpGraph(a, *m.profile, *m.binding, *m.drafter, shape, m.mtp_stride)
            .has_value();
      });
  if (!arena) {
    return std::unexpected(arena.error());
  }
  out->arena.emplace(std::move(*arena));
  auto graph =
      kg::BuildQwen38MtpGraph(*out->arena, *m.profile, *m.binding, *m.drafter, shape, m.mtp_stride);
  out->arena->Seal();
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  BindQwen38MtpWeights(m, out->graph);
  // Every pass's draft and its probability stay live to the end: the host
  // copies them all out after the last pass.
  std::vector<ggml_tensor*> kept = out->graph.drafts;
  kept.insert(kept.end(), out->graph.probabilities.begin(), out->graph.probabilities.end());
  kept.insert(kept.end(), out->graph.head_inputs.begin(), out->graph.head_inputs.end());
  kept.insert(kept.end(), out->graph.head_logits.begin(), out->graph.head_logits.end());
  if (auto r = PlaceAndPlan(*out, out->graph.nodes, out->graph.inputs(), kept, choices, activations,
                            activation_bytes);
      !r) {
    return std::unexpected(r.error());
  }
  return out;
}

void Qwen38MtpSources(const kg::Qwen38MtpGraph& g, std::span<const md::Qwen38ChunkInputs> passes,
                      std::span<const std::int32_t> tokens, Qwen38MtpHostInputs& out) {
  out.zero_row = 0;
  out.zero_index = 0;
  out.sources = {{g.state_row, &out.zero_row}, {g.row_zero, &out.zero_index}};
  for (std::size_t p = 0; p < g.passes.size() && p < passes.size(); ++p) {
    const kg::Qwen38MtpPass& t = g.passes[p];
    const md::Qwen38ChunkInputs& in = passes[p];
    if (t.tokens != nullptr) {
      out.sources.emplace_back(t.tokens, tokens.data());
    }
    out.sources.emplace_back(t.positions, in.positions.data());
    out.sources.emplace_back(t.cells, in.cells.data());
    if (t.mask != nullptr) {
      out.sources.emplace_back(t.mask, in.mask.data());
    }
  }
}

void Qwen38Sources(const kg::Qwen38Graph& g, const md::Qwen38ChunkInputs& in, std::uint32_t outputs,
                   std::span<const std::int32_t> ple_rows, Qwen38HostInputs& out,
                   std::int64_t stream_row0) {
  out.row_ids.resize(in.rows);
  std::ranges::iota(out.row_ids, std::int64_t{0});
  out.stream_rows.resize(in.rows);
  std::ranges::iota(out.stream_rows, stream_row0);
  out.out_ids.resize(outputs);
  std::ranges::iota(out.out_ids, static_cast<std::int32_t>(in.rows - outputs));
  out.zero_row = 0;
  out.zero_index = 0;
  out.sources = {
      {g.tokens, in.tokens.data()}, {g.positions, in.positions.data()}, {g.cells, in.cells.data()}};
  // The fast graph's QSA selection needs no masks or tables from the host.
  if (g.mask != nullptr) {
    out.sources.emplace_back(g.mask, in.mask.data());
  }
  out.sources.emplace_back(g.ple_rows, ple_rows.empty() ? in.ple_rows.data() : ple_rows.data());
  out.sources.emplace_back(g.state_row, &out.zero_row);
  out.sources.emplace_back(g.row_zero, &out.zero_index);
  out.sources.emplace_back(g.out_ids, out.out_ids.data());
  if (in.qsa_select && g.cell_block != nullptr) {
    out.sources.emplace_back(g.mask_f32, in.mask_f32.data());
    out.sources.emplace_back(g.cell_block, in.qsa.cell_block.data());
    out.sources.emplace_back(g.block_cells, in.qsa.block_cells.data());
    out.sources.emplace_back(g.block_pos, in.qsa.block_pos.data());
    out.sources.emplace_back(g.block_bias, in.qsa.bias.data());
  }
  if (g.row_ids != nullptr) {
    out.sources.emplace_back(g.row_ids, out.row_ids.data());
  }
  if (g.stream_rows != nullptr) {
    out.sources.emplace_back(g.stream_rows, out.stream_rows.data());
  }
}

}  // namespace jitllm::engine
