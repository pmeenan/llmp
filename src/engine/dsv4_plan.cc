// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <utility>

#include "engine/graph_mask_inputs.h"
#include "engine/support.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
using support::Error;

// Validate before constructing embeddings or joined payloads. In device mode
// no host matrix exists; its complete producer is the authenticated source.
std::expected<void, std::string> RawMaskInputs(const Dsv4Model& m, const kg::Dsv4Graph& g,
                                               const md::Dsv4ChunkInputs& in,
                                               const ggml_tensor* positions,
                                               std::span<ggml_tensor* const> nodes,
                                               std::span<ggml_tensor* const> inputs,
                                               std::uint32_t first) {
  if (m.profile == nullptr || m.state == nullptr || in.rows == 0 || in.rows > m.state->max_rows ||
      in.n_past > m.state->context || in.rows > m.state->context - in.n_past ||
      in.positions.size() != in.rows || in.raw_cells.size() != in.rows ||
      g.device_raw_mask != m.device_raw_masks || positions == nullptr ||
      positions->type != GGML_TYPE_I32 || positions->op != GGML_OP_NONE ||
      positions->view_src != nullptr ||
      std::ranges::any_of(positions->src, [](const auto* parent) { return parent != nullptr; }) ||
      positions->ne[0] < std::int64_t{first} + in.rows || positions->ne[1] != 1 ||
      positions->ne[2] != 1 || positions->ne[3] != 1 || positions->nb[0] != 4 ||
      positions->nb[1] != static_cast<std::uint64_t>(positions->ne[0]) * 4 ||
      positions->nb[2] != positions->nb[1] || positions->nb[3] != positions->nb[2] ||
      std::ranges::count(inputs, positions) != 1 || g.raw_k_idxs == nullptr ||
      g.raw_k_idxs->type != GGML_TYPE_I64 || g.raw_k_idxs->op != GGML_OP_NONE ||
      g.raw_k_idxs->view_src != nullptr ||
      std::ranges::any_of(
          g.raw_k_idxs->src, [](const auto* parent) { return parent != nullptr; }) ||
      g.raw_k_idxs->ne[0] != in.rows || g.raw_k_idxs->ne[1] != 1 || g.raw_k_idxs->ne[2] != 1 ||
      g.raw_k_idxs->ne[3] != 1 || g.raw_k_idxs->nb[0] != 8 ||
      g.raw_k_idxs->nb[1] != std::uint64_t{in.rows} * 8 ||
      g.raw_k_idxs->nb[2] != g.raw_k_idxs->nb[1] || g.raw_k_idxs->nb[3] != g.raw_k_idxs->nb[2] ||
      std::ranges::count(inputs, g.raw_k_idxs) != 1 || m.state->raw_cells == 0)
    return Error("raw mask inputs differ from their state and packed graph inputs");
  const auto total = std::uint64_t{in.n_past} + in.rows;
  const auto cells = m.state->raw_cells;
  const auto n_kv =
      m.state->window == md::Dsv4Window::kRing
          ? cells
          : std::min<std::uint64_t>(
                cells, std::max<std::uint64_t>(
                           256, (std::min(total, std::uint64_t{cells}) + 255) / 256 * 256));
  if (in.raw_n_kv != n_kv ||
      (g.device_raw_mask ? !in.raw_mask.empty()
                         : in.raw_mask.size() != std::uint64_t{in.rows} * n_kv))
    return Error("raw mask matrix differs from its graph-owned or host source policy");
  for (std::uint32_t row = 0; row < in.rows; ++row) {
    const auto position = std::uint64_t{in.n_past} + row;
    if (std::cmp_not_equal(in.positions[row], position) ||
        std::cmp_not_equal(in.raw_cells[row], position % cells))
      return Error("raw mask positions or ring cells differ from the actual chunk");
  }
  auto bytes = GraphMaskSourceBytes(g.raw_mask, positions, nodes, inputs, g.device_raw_mask, first,
                                    in.rows, in.raw_n_kv, cells, m.profile->window,
                                    m.state->context, kg::CausalMaskRows::kExact);
  if (!bytes) return std::unexpected(bytes.error());
  return {};
}

}  // namespace

std::expected<kg::Dsv4ChunkShape, std::string> Dsv4PrefillShape(const md::Dsv4StateLayout& state,
                                                                std::uint32_t first,
                                                                std::uint32_t rows,
                                                                std::int64_t outputs) {
  if (rows == 0 || rows > state.max_rows || first > state.context || rows > state.context - first ||
      (outputs != 0 && outputs != 1))
    return Error("no DeepSeek prefill geometry within the configured envelope");
  auto csa = md::Dsv4CompressorGeometry(md::kDsv4CsaRatio, true, state.csa_state_rows,
                                        state.csa_cells, first, rows);
  auto hca = md::Dsv4CompressorGeometry(md::kDsv4HcaRatio, false, state.hca_state_rows,
                                        state.hca_cells, first, rows);
  if (!csa || !hca) return Error(!csa ? csa.error() : hca.error());
  const std::uint64_t end = std::uint64_t{first} + rows;
  const auto raw =
      state.window == md::Dsv4Window::kRing
          ? state.raw_cells
          : std::min<std::uint64_t>(state.raw_cells,
                                    std::max<std::uint64_t>(256, ((end + 255) / 256) * 256));
  return kg::Dsv4ChunkShape{.rows = rows,
                            .outputs = outputs,
                            .raw_n_kv = static_cast<std::int64_t>(raw),
                            .raw_cells = state.raw_cells,
                            .csa_n_kv = csa->n_kv,
                            .hca_n_kv = hca->n_kv,
                            .csa_cells = state.csa_cells,
                            .hca_cells = state.hca_cells,
                            .csa_blocks = csa->blocks,
                            .hca_blocks = hca->blocks,
                            .csa_persist = csa->persist,
                            .hca_persist = hca->persist,
                            .csa_state_rows = state.csa_state_rows,
                            .hca_state_rows = state.hca_state_rows};
}

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
  }
  BindDsv4State(m, m.places.state, g);
}

void BindDsv4State(const Dsv4Model& m, std::uint64_t base, kg::Dsv4Graph& g) {
  using K = md::Dsv4StateTensor::Kind;
  for (std::uint32_t il = 0; il < m.profile->layers && il < g.layers.size(); ++il) {
    kg::Dsv4LayerTensors& l = g.layers[il];
    const auto state = [&](ggml_tensor* t, K kind) {
      // A view (a ring's compressed cache in its window's tensor) is
      // bound with the tensor it views.
      if (t == nullptr || t->view_src != nullptr) {
        return;
      }
      const std::int64_t i = m.state->Find(il, kind);
      kg::TensorArena::Bind(t, base + m.state->tensors[static_cast<std::size_t>(i)].offset);
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

namespace {

// Whether a chunk of these rows takes the output-A/HCA prefill: a full
// 4,096-row chunk, or with prefill_outa_hca_partial any prefill chunk.
bool OutAHcaRows(const Dsv4Model& m, std::int64_t rows) {
  return rows == kg::kDsv4HcaMaxRows ||
         (m.prefill_outa_hca_partial && rows >= kg::kDsv4HcaMinRows && rows <= kg::kDsv4HcaMaxRows);
}

}  // namespace

bool Dsv4PrefillHca(const Dsv4Model& m, const kg::Dsv4ChunkShape& shape) {
  return !m.exact && m.prefill_outa_hca && m.state->window == md::Dsv4Window::kRing &&
         OutAHcaRows(m, shape.rows) && shape.hca_n_kv == 256;
}

std::expected<std::unique_ptr<Dsv4Planned>, std::string> PlanDsv4Chunk(
    const Dsv4Model& m, const kg::Dsv4ChunkShape& shape, const kg::DeviceChoices& choices,
    std::span<const std::string> keep_names, std::uint64_t activations,
    std::uint64_t activation_bytes, const Dsv4Speculation& speculation,
    std::optional<std::uint32_t> first_position) {
  return PlanDsv4Chunk(m, shape, choices, keep_names, activations, activation_bytes, speculation,
                       first_position, std::nullopt);
}

std::expected<std::unique_ptr<Dsv4Planned>, std::string> PlanDsv4Chunk(
    const Dsv4Model& m, const kg::Dsv4ChunkShape& shape, const kg::DeviceChoices& choices,
    std::span<const std::string> keep_names, std::uint64_t activations,
    std::uint64_t activation_bytes, const Dsv4Speculation& speculation,
    std::optional<std::uint32_t> first_position, std::optional<ActivationMeasurement> measurement) {
  if (measurement && activations != 0)
    return Error("measurement-only planning cannot use activation storage");
  if (speculation.state_only && (shape.token || speculation.verify || !keep_names.empty()))
    return Error("state-only prefill cannot publish tokens, verify or retain diagnostics");
  if (shape.token && (shape.rows != 1 || m.exact || speculation.verify ||
                      speculation.drafter != nullptr || !keep_names.empty() || first_position))
    return Error("plain device token plans require one non-speculative target row");
  // A verify never takes the output-A/HCA prefill (its rows are captured
  // as a graph, which would keep one position).
  if (first_position &&
      (speculation.verify || !Dsv4PrefillHca(m, shape) ||
       static_cast<std::uint64_t>(*first_position) + static_cast<std::uint64_t>(shape.rows) >
           m.state->context)) {
    return Error("the prefill plan's first position is not an HCA prefill chunk's in its context");
  }
  if ((m.exact || speculation.verify || !keep_names.empty()) && shape.outputs != 0 &&
      shape.outputs != shape.rows) {
    return Error("a reference chunk, verify or named diagnostic needs every row's head");
  }
  auto out = std::make_unique<Dsv4Planned>();
  const bool outa_prefill = !m.exact && !speculation.verify && m.prefill_outa_hca &&
                            m.state->window == md::Dsv4Window::kRing && OutAHcaRows(m, shape.rows);
  kg::Dsv4GraphOptions options{.expert_stride = m.places.stride,
                               .row_invariant = speculation.verify && m.exact,
                               .fused = !m.exact,
                               .outa_prefill = outa_prefill,
                               .raw_mask_context = m.device_raw_masks ? m.state->context : 0,
                               .state_only = speculation.state_only};
  // The ds4 prefill stage mechanisms (docs/experiments/ds4-prefill-stages):
  // the fast plan's defaults, each under its own shape guard. A named
  // diagnostic keeps the tensors they leave unwritten (or write as F16), so
  // it plans without them.
  const bool stages = !m.exact && keep_names.empty();
  kg::SetDsv4PrefillStages(options, stages);
  if (const DsparkModel* d = speculation.drafter; d != nullptr) {
    options.features = d->profile->target_layers;
    options.inject = kg::Dsv4Injection{.profile = d->profile,
                                       .binding = d->binding,
                                       .rows = speculation.inject_rows,
                                       .ring = d->state->ring};
  }
  auto arena = SizedArena(kg::Dsv4GraphTensors(*m.profile), [&](kg::TensorArena& a) {
    return kg::BuildDsv4Graph(a, *m.profile, *m.binding, shape, options).has_value();
  });
  if (!arena) {
    return std::unexpected(arena.error());
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildDsv4Graph(*out->arena, *m.profile, *m.binding, shape, options);
  out->arena->Seal();
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Dsv4Graph& g = out->graph;
  g.prefill_first_position = first_position;
  BindDsv4Weights(m, g);
  if (speculation.drafter != nullptr) {
    BindDsparkInjection(*speculation.drafter, g);
  }
  // The logits are read after the run whatever node comes last.
  std::vector<ggml_tensor*> keep;
  if (g.logits != nullptr) keep.push_back(g.logits);
  if (g.token != nullptr) keep.push_back(g.token);
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
  kg::SetDsv4PrefillStages(device, stages);
  // HCA alone has not passed the registered quality gate. Require every
  // layer's actual output-A insertion, including weight/YaRN eligibility.
  // A pruned final result must also satisfy the headed eligibility gate;
  // otherwise pruning could enable different persistent HCA arithmetic.
  const bool cut_last =
      options.state_only &&
      std::ranges::find(options.features, m.profile->layers) == options.features.end();
  const auto attention_layers = m.profile->layers - static_cast<std::uint32_t>(cut_last);
  const bool all_outa =
      g.headed_outa_layers == m.profile->layers &&
      std::cmp_equal(std::ranges::count_if(g.nodes,
                                           [](const auto* node) {
                                             return kg::JitllmOpOf(node) == kg::JitllmOp::kDsv4OutA;
                                           }),
                     attention_layers);
  device.ds4_hca = outa_prefill && all_outa && Dsv4PrefillHca(m, shape);
  if (device.ds4_hca) {
    if (!first_position) {
      return Error("the combined prefill HCA plan needs its first position");
    }
    for (auto* node : g.nodes) {
      if (node->op == GGML_OP_FLASH_ATTN_EXT &&
          kg::JitllmOpOf(node->src[3]) == kg::JitllmOp::kDsv4SparseMask &&
          kg::JitllmOpInt(node->src[3], 1) == 1) {
        kg::MarkDsv4HcaTokentile(node, *first_position);
      }
    }
  }
  const auto inputs = g.inputs();
  if (auto placed = PlaceAndPlan(*out, g.nodes, inputs, keep, device, activations, activation_bytes,
                                 measurement);
      !placed) {
    return std::unexpected(placed.error());
  }
  return out;
}

std::expected<void, std::string> SetDsv4HcaFirstPosition(const Dsv4Model& m, kg::Dsv4Graph& g,
                                                         std::uint32_t first, bool captured) {
  if (!g.prefill_first_position || g.positions == nullptr) {
    return Error("not an HCA prefill plan");
  }
  if (captured) {
    // A replay would keep the captured position whatever is set here.
    return Error("an HCA prefill plan was captured as a graph");
  }
  const auto rows = static_cast<std::uint64_t>(g.positions->ne[0]);
  if (std::uint64_t{first} + rows > m.state->context) {
    return Error("the prefill plan's first position is not an HCA prefill chunk's in its context");
  }
  for (auto* node : g.nodes) {
    if (node->op == GGML_OP_FLASH_ATTN_EXT &&
        node->op_params[kg::kDsv4HcaTagParam] == kg::kDsv4HcaTag) {
      kg::MarkDsv4HcaTokentile(node, first);
    }
  }
  g.prefill_first_position = first;
  return {};
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

std::expected<std::unique_ptr<Dsv4WavePlanned>, std::string> PlanDsv4Wave(
    const Dsv4Model& m, std::span<const std::uint64_t> states, const kg::Dsv4WaveShape& shape,
    const kg::DeviceChoices& choices, std::uint64_t activations, std::uint64_t activation_bytes,
    const DsparkModel* drafter, std::span<const std::uint64_t> rings) {
  return PlanDsv4Wave(m, states, shape, choices, activations, activation_bytes, drafter, rings,
                      std::nullopt);
}

std::expected<std::unique_ptr<Dsv4WavePlanned>, std::string> PlanDsv4Wave(
    const Dsv4Model& m, std::span<const std::uint64_t> states, const kg::Dsv4WaveShape& shape,
    const kg::DeviceChoices& choices, std::uint64_t activations, std::uint64_t activation_bytes,
    const DsparkModel* drafter, std::span<const std::uint64_t> rings,
    std::optional<ActivationMeasurement> measurement) {
  if (measurement && activations != 0)
    return Error("measurement-only planning cannot use activation storage");
  if (m.exact || states.size() != shape.slots.size() ||
      (drafter != nullptr && rings.size() != shape.slots.size())) {
    return Error("a wave runs the fast plan over one state place a slot");
  }
  auto out = std::make_unique<Dsv4WavePlanned>();
  kg::Dsv4GraphOptions options{.expert_stride = m.places.stride,
                               .fused = true,
                               .raw_mask_context = m.device_raw_masks ? m.state->context : 0};
  kg::SetDsv4PrefillStages(options, true);
  if (drafter != nullptr) {
    options.features = drafter->profile->target_layers;
    options.inject = kg::Dsv4Injection{.profile = drafter->profile,
                                       .binding = drafter->binding,
                                       .rows = 0,
                                       .ring = drafter->state->ring};
  }
  auto arena =
      SizedArena(kg::Dsv4WaveGraphTensors(*m.profile, shape.slots.size()), [&](kg::TensorArena& a) {
        return kg::BuildDsv4WaveGraph(a, *m.profile, *m.binding, shape, options).has_value();
      });
  if (!arena) {
    return std::unexpected(arena.error());
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildDsv4WaveGraph(*out->arena, *m.profile, *m.binding, shape, options);
  out->arena->Seal();
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Dsv4WaveGraph& g = out->graph;
  BindDsv4Weights(m, g.joined);  // the joined graph holds no state
  for (std::size_t i = 0; i < g.slots.size(); ++i) {
    BindDsv4State(m, states[i], g.slots[i]);
  }
  if (drafter != nullptr) {
    if (!g.joined.inject.has_value()) {
      return Error("a wave beside a drafter built no injection");
    }
    const md::DsparkBinding& b = *drafter->binding;
    kg::DsparkInjectTensors& t = *g.joined.inject;
    kg::TensorArena::Bind(t.fc, drafter->places.resource(b.fc.index));
    kg::TensorArena::Bind(t.enc_norm, drafter->places.resource(b.enc_norm.index));
    for (std::size_t il = 0; il < t.kv.size(); ++il) {
      kg::TensorArena::Bind(t.kv[il], drafter->places.resource(b.blocks.layers[il].kv.index));
      kg::TensorArena::Bind(t.kv_norm[il],
                            drafter->places.resource(b.blocks.layers[il].kv_norm.index));
    }
    for (std::size_t i = 0; i < g.slots.size(); ++i) {
      std::optional<kg::DsparkInjectTensors>& injected = g.slots[i].inject;
      if (!injected.has_value()) {
        return Error("a wave slot beside a drafter built no injection");
      }
      kg::DsparkInjectTensors& own = *injected;
      for (std::size_t il = 0; il < own.ring.size(); ++il) {
        kg::TensorArena::Bind(own.ring[il], rings[i] + drafter->state->offsets[il]);
      }
    }
  }
  std::vector<ggml_tensor*> keep = {g.joined.logits};
  if (g.joined.token != nullptr) keep.push_back(g.joined.token);
  kg::DeviceChoices device = choices;
  device.row_invariant = false;
  device.fuse_norms = true;
  device.vector_floats = true;
  device.pair_experts = true;
  device.compact_experts = false;
  device.wide_sparse_attention = true;
  kg::SetDsv4PrefillStages(device, true);
  device.ds4_hca = false;
  const auto inputs = g.inputs();
  // Each slot's attention and state operations on a lane of its own.
  if (auto placed = PlaceAndPlan(*out, g.joined.nodes, inputs, keep, device, activations,
                                 activation_bytes, measurement, m.wave_lanes ? &g.lanes : nullptr);
      !placed) {
    return std::unexpected(placed.error());
  }
  return out;
}

std::expected<void, std::string> BuildDsv4WaveInputs(const Dsv4Model& m, const kg::Dsv4WaveGraph& g,
                                                     std::span<const Dsv4WaveSlotInputs> slots,
                                                     std::span<const std::byte> table,
                                                     Dsv4WaveHostInputs& out) {
  if (slots.size() != g.slots.size()) {
    return Error("the wave's inputs are not its slots'");
  }
  if (g.first.size() != slots.size()) return Error("the wave has no exact row offsets");
  const auto graph_inputs = g.inputs();
  std::uint32_t first = 0;
  for (std::size_t slot = 0; slot < slots.size(); ++slot) {
    const auto* chunk = slots[slot].chunk;
    if (chunk == nullptr || std::cmp_not_equal(g.first[slot], first) ||
        slots[slot].tokens.size() != chunk->rows)
      return Error("raw mask wave offsets or rows differ from the actual slots");
    if (auto mask = RawMaskInputs(m, g.slots[slot], *chunk, g.joined.positions, g.joined.nodes,
                                  graph_inputs, first);
        !mask)
      return mask;
    first += chunk->rows;
  }
  if (std::cmp_not_equal(g.joined.positions->ne[0], first))
    return Error("raw mask joined positions differ from the actual total rows");
  out.tokens.clear();
  out.positions.clear();
  for (auto& pos : out.state_pos) {
    pos.clear();
  }
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const Dsv4WaveSlotInputs& s = slots[i];
    const kg::Dsv4Graph& sg = g.slots[i];
    const md::Dsv4ChunkInputs* in = s.chunk;
    if (in == nullptr || in->rows != s.tokens.size() || in->positions.size() != in->rows ||
        in->csa.n_visible.size() != in->rows || in->hca.n_visible.size() != in->rows ||
        sg.raw_k_idxs == nullptr || std::cmp_not_equal(sg.raw_k_idxs->ne[0], in->rows) ||
        std::cmp_not_equal(out.tokens.size(), g.first[i]) ||
        (sg.inject ? std::cmp_not_equal(s.inject_cells.size(), sg.inject->cells->ne[0])
                   : !s.inject_cells.empty())) {
      return Error("a wave slot's inputs are not its graph's");
    }
    out.tokens.insert(out.tokens.end(), s.tokens.begin(), s.tokens.end());
    out.positions.insert(out.positions.end(), in->positions.begin(), in->positions.end());
    const std::array<const md::Dsv4CompPlan*, 3> plans = {&in->csa, &in->hca, &in->lid};
    for (std::size_t k = 0; k < plans.size(); ++k) {
      if (plans[k]->state_pos.size() != in->rows) {
        return Error("a wave slot's compressor rows are not its rows");
      }
      out.state_pos[k].insert(out.state_pos[k].end(), plans[k]->state_pos.begin(),
                              plans[k]->state_pos.end());
    }
  }
  if (std::cmp_not_equal(out.tokens.size(), g.joined.tokens->ne[0])) {
    return Error("the wave's rows are not its graph's");
  }
  if (auto embedded = Dsv4EmbeddingRows(m, out.tokens, table, out.embd); !embedded) {
    return embedded;
  }
  out.sources = {{g.joined.embd, out.embd.data()},
                 {g.joined.tokens, out.tokens.data()},
                 {g.joined.positions, out.positions.data()},
                 {g.joined.csa.state_pos, out.state_pos[0].data()},
                 {g.joined.hca.state_pos, out.state_pos[1].data()},
                 {g.joined.lid.state_pos, out.state_pos[2].data()},
                 {g.joined.lid_rot, m.rot.data()}};
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const md::Dsv4ChunkInputs& in = *slots[i].chunk;
    const kg::Dsv4Graph& sg = g.slots[i];
    out.sources.emplace_back(sg.raw_k_idxs, in.raw_cells.data());
    if (!sg.device_raw_mask) out.sources.emplace_back(sg.raw_mask, in.raw_mask.data());
    const std::array<std::pair<const kg::Dsv4CompInputs*, const md::Dsv4CompPlan*>, 3> comps = {
        {{&sg.csa, &in.csa}, {&sg.hca, &in.hca}, {&sg.lid, &in.lid}}};
    for (const auto& [t, plan] : comps) {
      if (std::cmp_not_equal(plan->persist_src.size(), t->persist_src->ne[0]) ||
          std::cmp_not_equal(plan->read_idxs.size(), t->read_idxs->ne[0]) ||
          std::cmp_not_equal(plan->write_idxs.size(), t->write_idxs->ne[0])) {
        return Error("a wave slot's compressor plan is not its graph's");
      }
      out.sources.insert(out.sources.end(), {{t->persist_src, plan->persist_src.data()},
                                             {t->persist_dst, plan->persist_dst.data()},
                                             {t->read_idxs, plan->read_idxs.data()},
                                             {t->write_idxs, plan->write_idxs.data()},
                                             {t->write_pos, plan->write_pos.data()}});
    }
    out.sources.emplace_back(sg.csa_visible, in.csa.n_visible.data());
    out.sources.emplace_back(sg.hca_visible, in.hca.n_visible.data());
    if (sg.inject) {
      out.sources.emplace_back(sg.inject->cells, slots[i].inject_cells.data());
    }
  }
  return {};
}

namespace {
void BindDsparkWeights(const DsparkModel& d, kg::Dsv4Graph& core, ggml_tensor* markov_w1,
                       ggml_tensor* markov_w2);
kg::DeviceChoices DraftChoices(const DsparkModel& d, const kg::DeviceChoices& choices);
}  // namespace

std::expected<std::unique_ptr<DsparkPlanned>, std::string> PlanDsparkDraft(
    const DsparkModel& d, std::int64_t rows, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes) {
  return PlanDsparkDraft(d, rows, choices, activations, activation_bytes, std::nullopt);
}

std::expected<std::unique_ptr<DsparkPlanned>, std::string> PlanDsparkDraft(
    const DsparkModel& d, std::int64_t rows, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes,
    std::optional<ActivationMeasurement> measurement) {
  if (measurement && activations != 0)
    return Error("measurement-only planning cannot use activation storage");
  auto out = std::make_unique<DsparkPlanned>();
  const kg::Dsv4GraphOptions options{
      .expert_stride = d.places.stride, .fused = !d.exact, .device_draft_masks = d.device_masks};
  auto arena = SizedArena(kg::DsparkGraphTensors(*d.profile, rows), [&](kg::TensorArena& a) {
    return kg::BuildDsparkGraph(a, *d.profile, *d.binding, rows, d.state->ring, options)
        .has_value();
  });
  if (!arena) {
    return std::unexpected(arena.error());
  }
  out->arena.emplace(std::move(*arena));
  auto graph =
      kg::BuildDsparkGraph(*out->arena, *d.profile, *d.binding, rows, d.state->ring, options);
  out->arena->Seal();
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::DsparkGraph& g = out->graph;
  BindDsparkWeights(d, g.core, g.markov_w1, g.markov_w2);
  for (std::uint32_t il = 0; il < d.profile->blocks.layers; ++il) {
    kg::TensorArena::Bind(g.core.layers[il].raw_k, d.places.state + d.state->offsets[il]);
  }
  const std::vector<ggml_tensor*> keep = {g.logits, g.drafts};
  const auto inputs = g.inputs();
  if (auto placed = PlaceAndPlan(*out, g.core.nodes, inputs, keep, DraftChoices(d, choices),
                                 activations, activation_bytes, measurement);
      !placed) {
    return std::unexpected(placed.error());
  }
  return out;
}

std::expected<std::unique_ptr<DsparkWavePlanned>, std::string> PlanDsparkWave(
    const DsparkModel& d, std::span<const std::uint64_t> rings, std::int64_t rows,
    const kg::DeviceChoices& choices, std::uint64_t activations, std::uint64_t activation_bytes,
    bool lanes) {
  return PlanDsparkWave(d, rings, rows, choices, activations, activation_bytes, lanes,
                        std::nullopt);
}

std::expected<std::unique_ptr<DsparkWavePlanned>, std::string> PlanDsparkWave(
    const DsparkModel& d, std::span<const std::uint64_t> rings, std::int64_t rows,
    const kg::DeviceChoices& choices, std::uint64_t activations, std::uint64_t activation_bytes,
    bool lanes, std::optional<ActivationMeasurement> measurement) {
  if (measurement && activations != 0)
    return Error("measurement-only planning cannot use activation storage");
  if (d.exact || rings.size() < 2) {
    return Error("a joined draft runs the fast plan over two rings or more");
  }
  auto out = std::make_unique<DsparkWavePlanned>();
  const kg::Dsv4GraphOptions options{
      .expert_stride = d.places.stride, .fused = true, .device_draft_masks = d.device_masks};
  const std::size_t slots = rings.size();
  auto arena =
      SizedArena(kg::DsparkWaveGraphTensors(*d.profile, rows, slots), [&](kg::TensorArena& a) {
        return kg::BuildDsparkWaveGraph(a, *d.profile, *d.binding, rows, slots, d.state->ring,
                                        options)
            .has_value();
      });
  if (!arena) {
    return std::unexpected(arena.error());
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildDsparkWaveGraph(*out->arena, *d.profile, *d.binding, rows, slots,
                                        d.state->ring, options);
  out->arena->Seal();
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::DsparkWaveGraph& g = out->graph;
  BindDsparkWeights(d, g.joined, g.markov_w1, g.markov_w2);  // the joined graph holds no ring
  for (std::size_t i = 0; i < slots; ++i) {
    for (std::uint32_t il = 0; il < d.profile->blocks.layers; ++il) {
      kg::TensorArena::Bind(g.slots[i].layers[il].raw_k, rings[i] + d.state->offsets[il]);
    }
  }
  const std::vector<ggml_tensor*> keep(g.drafts.begin(), g.drafts.end());
  const auto inputs = g.inputs();
  if (auto placed =
          PlaceAndPlan(*out, g.joined.nodes, inputs, keep, DraftChoices(d, choices), activations,
                       activation_bytes, measurement, lanes ? &g.lanes : nullptr);
      !placed) {
    return std::unexpected(placed.error());
  }
  return out;
}

namespace {

// The drafter's blocks at its places, its head's output the target's.
void BindDsparkWeights(const DsparkModel& d, kg::Dsv4Graph& core, ggml_tensor* markov_w1,
                       ggml_tensor* markov_w2) {
  const md::DsparkBinding& b = *d.binding;
  const auto bind = [](ggml_tensor* t, std::uint64_t address) {
    kg::TensorArena::Bind(t, address);
  };
  for (std::uint32_t il = 0; il < d.profile->blocks.layers; ++il) {
    const md::Dsv4Layer& r = b.blocks.layers[il];
    kg::Dsv4LayerTensors& l = core.layers[il];
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
  }
  bind(core.output_norm, d.places.resource(b.blocks.output_norm.index));
  bind(core.hc_head_fn, d.places.resource(b.blocks.hc_head_fn.index));
  bind(core.hc_head_base, d.places.resource(b.blocks.hc_head_base.index));
  bind(core.hc_head_scale, d.places.resource(b.blocks.hc_head_scale.index));
  bind(core.output, d.target_resource(b.blocks.output.index));
  bind(markov_w1, d.places.resource(b.markov_w1.index));
  bind(markov_w2, d.places.resource(b.markov_w2.index));
}

// A draft block's device choices (as a joined draft's).
kg::DeviceChoices DraftChoices(const DsparkModel& d, const kg::DeviceChoices& choices) {
  kg::DeviceChoices device = choices;
  device.fuse_norms = !d.exact;
  device.vector_floats = !d.exact;
  device.pair_experts = !d.exact;
  device.wide_sparse_attention = !d.exact;
  return device;
}

}  // namespace

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

namespace {
std::expected<void, std::string> DraftMaskInputs(const DsparkModel& d, const kg::Dsv4Graph& g,
                                                 const md::DsparkBlockInputs& in,
                                                 const ggml_tensor* positions,
                                                 std::span<ggml_tensor* const> nodes,
                                                 std::span<ggml_tensor* const> inputs,
                                                 std::uint32_t first) {
  if (d.profile == nullptr || d.state == nullptr || g.device_raw_mask != d.device_masks ||
      in.rows == 0 || in.rows > d.state->max_rows || in.rows > d.profile->block_size ||
      in.tokens.size() != in.rows || in.positions.size() != in.rows || in.cells.size() != in.rows ||
      std::uint64_t{in.pos0} + in.rows > INT32_MAX || positions == nullptr ||
      positions->type != GGML_TYPE_I32 || positions->ne[0] < 0 ||
      std::uint64_t{first} + in.rows > static_cast<std::uint64_t>(positions->ne[0]) ||
      g.raw_k_idxs == nullptr || g.raw_k_idxs->type != GGML_TYPE_I64 ||
      std::cmp_not_equal(g.raw_k_idxs->ne[0], in.rows))
    return Error("draft mask positions/cells differ from the actual block");
  const auto packed_input = [&](const ggml_tensor* input, ggml_type type, std::uint64_t rows,
                                std::uint64_t bytes) {
    return rows > 0 && rows <= INT32_MAX / bytes && input->type == type &&
           input->op == GGML_OP_NONE && input->view_src == nullptr &&
           !std::ranges::any_of(input->src, [](const auto* parent) { return parent != nullptr; }) &&
           std::cmp_equal(input->ne[0], rows) && input->ne[1] == 1 && input->ne[2] == 1 &&
           input->ne[3] == 1 && input->nb[0] == bytes && input->nb[1] == rows * bytes &&
           input->nb[2] == input->nb[1] && input->nb[3] == input->nb[2] &&
           std::ranges::count(inputs, input) == 1;
  };
  if (!packed_input(positions, GGML_TYPE_I32, static_cast<std::uint64_t>(positions->ne[0]), 4) ||
      !packed_input(g.raw_k_idxs, GGML_TYPE_I64, in.rows, 8))
    return Error("draft positions/cells are not unique packed host inputs");
  const auto ring = d.state->ring;
  if (ring == 0 || ring != d.profile->ring || d.profile->blocks.window == 0 ||
      std::uint64_t{ring} < std::uint64_t{d.profile->blocks.window} + in.rows)
    return Error("draft mask ring does not retain the current block and window");
  for (std::uint32_t row = 0; row < in.rows; ++row) {
    const std::uint64_t pos = std::uint64_t{in.pos0} + row;
    if (std::cmp_not_equal(in.positions[row], pos) || std::cmp_not_equal(in.cells[row], pos % ring))
      return Error("draft mask input is not its contiguous block and ring cells");
  }
  auto bytes =
      GraphMaskSourceBytes(g.raw_mask, positions, nodes, inputs, d.device_masks, first, in.rows,
                           ring, ring, d.profile->blocks.window, INT32_MAX,
                           kg::CausalMaskRows::kExact, GGML_TYPE_F16, kg::MaskPolicy::kBlock);
  if (!bytes) return std::unexpected(bytes.error());
  if (in.mask.size() != *bytes / sizeof(std::uint16_t))
    return Error("draft mask host payload differs from the authenticated source policy");
  return {};
}
}  // namespace

std::expected<void, std::string> BuildDsparkInputs(const Dsv4Model& m, const DsparkModel& d,
                                                   const kg::DsparkGraph& g,
                                                   const md::DsparkBlockInputs& in,
                                                   std::span<const std::byte> table,
                                                   Dsv4HostInputs& out) {
  const auto inputs = g.inputs();
  if (g.core.positions == nullptr || std::cmp_not_equal(g.core.positions->ne[0], in.rows))
    return Error("draft scalar positions differ from its block");
  if (auto mask = DraftMaskInputs(d, g.core, in, g.core.positions, g.core.nodes, inputs, 0); !mask)
    return mask;
  if (auto rows = Dsv4EmbeddingRows(m, in.tokens, table, out.embd); !rows) return rows;
  out.tokens = in.tokens;
  out.sources = {{g.core.embd, out.embd.data()},
                 {g.tokens, out.tokens.data()},
                 {g.core.positions, in.positions.data()},
                 {g.core.raw_k_idxs, in.cells.data()}};
  if (!d.device_masks) out.sources.emplace_back(g.core.raw_mask, in.mask.data());
  return {};
}

std::expected<void, std::string> BuildDsparkWaveInputs(
    const Dsv4Model& m, const DsparkModel& d, const kg::DsparkWaveGraph& g,
    std::span<const md::DsparkBlockInputs* const> blocks, std::span<const std::byte> table,
    DsparkWaveHostInputs& out) {
  if (blocks.size() != g.slots.size() || g.first.size() != blocks.size() ||
      g.joined.positions == nullptr || g.tokens == nullptr ||
      g.joined.positions->ne[0] != g.tokens->ne[0])
    return Error("a joined draft's blocks are not its graph's slots");
  const auto inputs = g.inputs();
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] == nullptr || g.first[i] < 0 || std::cmp_not_equal(g.first[i], total) ||
        total > UINT32_MAX)
      return Error("a joined draft's block offset differs from its real rows");
    if (auto mask = DraftMaskInputs(d, g.slots[i], *blocks[i], g.joined.positions, g.joined.nodes,
                                    inputs, static_cast<std::uint32_t>(total));
        !mask)
      return mask;
    total += blocks[i]->rows;
  }
  if (std::cmp_not_equal(total, g.tokens->ne[0]))
    return Error("a joined draft's rows are not its graph's");
  out.tokens.clear();
  out.positions.clear();
  for (const auto* block : blocks) {
    out.tokens.insert(out.tokens.end(), block->tokens.begin(), block->tokens.end());
    out.positions.insert(out.positions.end(), block->positions.begin(), block->positions.end());
  }
  if (auto rows = Dsv4EmbeddingRows(m, out.tokens, table, out.embd); !rows) return rows;
  out.sources = {{g.joined.embd, out.embd.data()},
                 {g.tokens, out.tokens.data()},
                 {g.joined.positions, out.positions.data()}};
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    out.sources.emplace_back(g.slots[i].raw_k_idxs, blocks[i]->cells.data());
    if (!d.device_masks) out.sources.emplace_back(g.slots[i].raw_mask, blocks[i]->mask.data());
  }
  return {};
}

std::expected<void, std::string> BuildDsv4Inputs(const Dsv4Model& m, const kg::Dsv4Graph& g,
                                                 const md::Dsv4ChunkInputs& in,
                                                 std::span<const std::int32_t> tokens,
                                                 std::span<const std::byte> table,
                                                 Dsv4HostInputs& out,
                                                 std::span<const std::int64_t> inject_cells) {
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  // The HCA scalar is part of the cached plan. Authenticate it
  // against the real host positions before any copies or dispatch.
  if (g.prefill_first_position) {
    const std::uint32_t first = *g.prefill_first_position;
    if (in.positions.size() != rows || rows == 0 || in.positions.front() < 0 ||
        std::cmp_not_equal(in.positions.front(), first) ||
        !std::ranges::equal(in.positions,
                            std::views::iota(static_cast<std::int64_t>(first),
                                             static_cast<std::int64_t>(first) + rows))) {
      return Error("the cached HCA first position differs from the actual chunk");
    }
  }
  const auto graph_inputs = g.inputs();
  if (tokens.size() != in.rows || g.positions == nullptr || g.positions->ne[0] != in.rows)
    return Error("raw mask scalar position rows differ from the actual chunk");
  if (auto mask = RawMaskInputs(m, g, in, g.positions, g.nodes, graph_inputs, 0); !mask)
    return mask;
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
  out.sources = {{g.embd, out.embd.data()},
                 {g.tokens, out.tokens.data()},
                 {g.positions, in.positions.data()},
                 {g.raw_k_idxs, in.raw_cells.data()}};
  if (!g.device_raw_mask) out.sources.emplace_back(g.raw_mask, in.raw_mask.data());
  out.sources.emplace_back(g.out_ids, out.out_ids.data());
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
