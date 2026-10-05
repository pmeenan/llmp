// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma4_plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "engine/support.h"
#include "kernels/ggml/jitllm_ops.h"

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
using support::Error;
bool RegionValid(const Gemma4Region& r, std::uint64_t required) {
  return r.address != 0 && r.address % 256 == 0 && required <= r.bytes &&
         r.bytes <= UINT64_MAX - r.address;
}
bool Overlap(const Gemma4Region& a, const Gemma4Region& b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}  // namespace

std::expected<void, std::string> BindGemma4Weights(const Gemma4Model& m, kg::Gemma4Graph& g) {
  if (m.profile == nullptr || m.binding == nullptr || m.state == nullptr ||
      !model::CheckGemma4Binding(*m.profile, *m.binding) || !m.state->Representations(*m.profile)) {
    return Error("incomplete or malformed Gemma4 model");
  }
  if (g.profile != *m.profile || g.binding != *m.binding || g.context != m.state->context ||
      g.max_rows != m.state->max_rows || g.global_capacity != m.state->global_cells ||
      g.local_capacity != m.state->local_cells || g.state_bytes != m.state->bytes ||
      g.options != m.options)
    return Error("Gemma4 graph/model contracts differ");
  // Preflight ALL addresses before mutating descriptors. The logical region
  // supplies a bound, not proof of catalog residency or initialized KV cells.
  for (const auto& w : g.weights) {
    const auto& domain = w.resource.expert_array ? m.arrays : m.resources;
    const auto index = w.resource.index;
    if (index >= domain.size()) return Error("Gemma4 weight index exceeds its address domain");
    const auto bytes =
        w.resource.readable +
        (w.resource.expert_array ? (std::uint64_t{m.profile->experts} - 1) * w.tensor->nb[2] : 0);
    if (!RegionValid(domain[index], bytes))
      return Error("Gemma4 weight region is absent, short or overflowing");
  }
  for (const auto* domain : {&m.resources, &m.arrays, &m.slots}) {
    for (const auto& occupied : *domain) {
      if (occupied.address == 0 && occupied.bytes == 0) continue;
      if (occupied.bytes == 0 || !RegionValid(occupied, 0)) {
        return Error("Gemma4 supplied weight/state region cannot be bounded");
      }
    }
  }
  for (const auto& seg : g.segments) {
    if (seg.shape.slot >= m.slots.size() || !RegionValid(m.slots[seg.shape.slot], m.state->bytes)) {
      return Error("Gemma4 slot region is absent, short or overflowing");
    }
    const auto& slot = m.slots[seg.shape.slot];
    for (const auto& other : m.slots) {
      if (&other != &slot && other.bytes != 0 && Overlap(slot, other)) {
        return Error("Gemma4 independent slot state regions overlap");
      }
    }
    for (const auto* domain : {&m.resources, &m.arrays}) {
      for (const auto& weight : *domain) {
        if (weight.bytes != 0 && Overlap(slot, weight))
          return Error("Gemma4 state overlaps immutable weights");
      }
    }
  }
  for (const auto& w : g.weights) {
    const auto& domain = w.resource.expert_array ? m.arrays : m.resources;
    kg::TensorArena::Bind(w.tensor, domain[w.resource.index].address);
  }
  for (auto& seg : g.segments) {
    const auto base = m.slots[seg.shape.slot].address;
    for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
      const auto [k, v] = seg.caches[il];
      if (k == nullptr) continue;
      kg::TensorArena::Bind(k, base + m.state->tensors[std::size_t{il} * 2].offset);
      kg::TensorArena::Bind(v, base + m.state->tensors[std::size_t{il} * 2 + 1].offset);
    }
  }
  return {};
}

std::expected<std::unique_ptr<Gemma4Planned>, std::string> PlanGemma4Chunk(
    const Gemma4Model& m, const kg::Gemma4ChunkShape& shape, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes, std::span<const std::string> keep) {
  if (m.profile == nullptr || m.binding == nullptr || m.state == nullptr) {
    return Error("incomplete Gemma4 model");
  }
  if (auto checked = kg::CheckGemma4Graph(*m.profile, *m.binding, *m.state, shape, m.options);
      !checked)
    return Error(checked.error().detail);
  if (activations != 0) {
    const Gemma4Region region{activations, activation_bytes};
    if (activation_bytes == 0 || !RegionValid(region, 0)) {
      return Error("Gemma4 activation region is empty, misaligned or overflowing");
    }
    // A diagnostic graph uses only some layers/slots, but their peers still
    // retain immutable weights and continuation state. Per-op alias checks
    // alone cannot protect an operand that this particular chunk never reads.
    for (const auto* domain : {&m.resources, &m.arrays, &m.slots}) {
      for (const auto& occupied : *domain) {
        if (occupied.address == 0 && occupied.bytes == 0) continue;
        if (occupied.bytes == 0 || !RegionValid(occupied, 0) || Overlap(region, occupied)) {
          return Error("Gemma4 activations overlap or cannot bound supplied weight/state storage");
        }
      }
    }
  }
  auto out = std::make_unique<Gemma4Planned>();
  auto arena = SizedArena(
      kg::Gemma4GraphTensors(*m.profile, shape.segments.size()), [&](kg::TensorArena& a) {
        return kg::BuildGemma4Graph(a, *m.profile, *m.binding, *m.state, shape, m.options)
            .has_value();
      });
  if (!arena) return std::unexpected(arena.error());
  out->arena.emplace(std::move(*arena));
  auto graph =
      kg::BuildGemma4Graph(*out->arena, *m.profile, *m.binding, *m.state, shape, m.options);
  out->arena->Seal();
  if (!graph) return Error(graph.error().detail);
  out->graph = std::move(*graph);
  auto& g = out->graph;
  if (auto bound = BindGemma4Weights(m, g); !bound) return std::unexpected(bound.error());
  std::vector<ggml_tensor*> kept;
  if (g.logits != nullptr) kept.push_back(g.logits);
  kept.push_back(g.hidden);
  for (const auto& name : keep) {
    auto* t = g.Named(name);
    if (t == nullptr) return Error(std::format("Gemma4 graph names no {}", name));
    kept.push_back(t);
  }
  // Caller-selected policies carry measured device choices. In particular no
  // generic GeGLU fusion, sparse attention or foreign quant writer is forced.
  auto device = choices;
  device.fuse_rope_store = m.options.rope_store;
  if (m.options.shared_q8 && g.positions->ne[0] <= kg::kRowInvariantColumns) {
    device.vector_floats = true;
  }
  if (auto placed =
          PlaceAndPlan(*out, g.nodes, g.inputs, kept, device, activations, activation_bytes);
      !placed)
    return std::unexpected(placed.error());
  return out;
}

std::expected<std::uint64_t, std::string> Gemma4SourceBytes(const kg::Gemma4Graph& g) {
  std::uint64_t bytes =
      std::uint64_t{g.shape.outputs} * sizeof(std::int32_t) +
      g.inputs.size() * sizeof(std::pair<ggml_tensor*, const void*>) +
      (g.options.device_masks ? 0 : g.segments.size() * 2 * sizeof(std::vector<std::uint16_t>));
  for (const auto& seg : g.segments) {
    for (const auto [mask, capacity, window, cells] :
         {std::tuple{seg.global_mask, g.global_capacity, 0U, seg.shape.global_n_kv},
          std::tuple{seg.local_mask, g.local_capacity, g.profile.window, seg.shape.local_n_kv}}) {
      if (mask == nullptr || cells == 0 || cells > INT32_MAX / 2 || seg.shape.rows == 0 ||
          seg.shape.rows > static_cast<std::uint32_t>(INT32_MAX - 31) ||
          ((std::uint64_t{seg.shape.rows} + 31) / 32 * 32) > INT32_MAX / 2 / cells ||
          mask->type != GGML_TYPE_F16 || mask->ne[0] != cells ||
          mask->ne[1] != ((std::int64_t{seg.shape.rows} + 31) / 32 * 32) || mask->ne[2] != 1 ||
          mask->ne[3] != 1 || mask->nb[0] != 2 || mask->nb[1] != std::uint64_t{cells} * 2 ||
          mask->nb[2] != mask->nb[1] * static_cast<std::uint64_t>(mask->ne[1]) ||
          mask->nb[3] != mask->nb[2] || ggml_nbytes(mask) > UINT64_MAX - bytes)
        return Error("malformed Gemma4 mask descriptor");
      const bool input = std::ranges::find(g.inputs, mask) != g.inputs.end();
      if (g.options.device_masks) {
        if (input || !kg::Gemma4MaskFits(mask) || mask->src[0] != g.positions ||
            kg::JitllmOpInt(mask, 0) != static_cast<std::int64_t>(seg.first_row) ||
            kg::JitllmOpInt(mask, 1) != static_cast<std::int64_t>(seg.shape.rows) ||
            kg::JitllmOpInt(mask, 2) != static_cast<std::int64_t>(capacity) ||
            kg::JitllmOpInt(mask, 3) != static_cast<std::int64_t>(window) ||
            kg::JitllmOpInt(mask, 4) != static_cast<std::int64_t>(g.context))
          return Error("Gemma4 device mask is not its graph-owned position producer");
      } else {
        if (!input || mask->op != GGML_OP_NONE)
          return Error("Gemma4 reference mask is not a host input");
        bytes += ggml_nbytes(mask);
      }
    }
  }
  return bytes;
}

std::expected<Gemma4HostInputs, std::string> Gemma4Sources(const kg::Gemma4Graph& g,
                                                           const model::Gemma4ChunkInputs& in,
                                                           std::span<const std::int32_t> frontier,
                                                           std::span<const float> hidden,
                                                           std::uint64_t funded_bytes) {
  auto bytes = Gemma4SourceBytes(g);
  if (!bytes || *bytes > funded_bytes) return Error("Gemma4 host sources are not funded");
  const auto rows = static_cast<std::size_t>(g.positions->ne[0]);
  if (in.tokens.size() != rows || in.positions.size() != rows ||
      in.segments.size() != g.segments.size() || frontier.size() != g.shape.outputs ||
      (g.input_hidden == nullptr
           ? !hidden.empty()
           : hidden.size() != rows * static_cast<std::size_t>(g.input_hidden->ne[0]))) {
    return Error("Gemma4 host input/hidden/frontier size differs from graph");
  }
  for (const auto id : frontier) {
    if (id < 0 || std::cmp_greater_equal(id, rows))
      return Error("Gemma4 frontier ID exceeds chunk");
  }
  for (std::size_t i = 0; i < g.segments.size(); ++i) {
    const auto& seg = g.segments[i];
    const auto& s = seg.shape;
    const auto& host = in.segments[i];
    if (host.slot != s.slot || host.first_row != seg.first_row || host.rows != s.rows ||
        host.n_past > g.context || s.rows > g.context - host.n_past ||
        s.global_n_kv < std::min(host.n_past + s.rows, g.global_capacity) ||
        s.local_n_kv < std::min(host.n_past + s.rows, g.local_capacity) ||
        host.global_n_kv != s.global_n_kv || host.local_n_kv != s.local_n_kv ||
        host.global_cells.size() != s.rows || host.local_cells.size() != s.rows ||
        (g.options.device_masks ? (!host.global_mask.empty() || !host.local_mask.empty())
                                : (host.global_mask.size() != std::size_t{s.global_n_kv} * s.rows ||
                                   host.local_mask.size() != std::size_t{s.local_n_kv} * s.rows))) {
      return Error("Gemma4 segment or reference mask differs from graph");
    }
    // Cell/value correctness comes from the checked chunk builder; refuse
    // edited public indices and absolute positions before any staging.
    for (std::uint32_t r = 0; r < s.rows; ++r) {
      const auto position = host.n_past + r;
      if (in.positions[seg.first_row + r] != static_cast<std::int32_t>(position) ||
          host.global_cells[r] != position || host.local_cells[r] != position % g.local_capacity ||
          in.tokens[seg.first_row + r] < 0 ||
          std::cmp_greater_equal(in.tokens[seg.first_row + r], g.profile.vocab)) {
        return Error("Gemma4 host position or cache index is invalid");
      }
      if (g.options.device_masks) continue;
      const auto end = host.n_past + s.rows;
      for (std::uint32_t cell = 0; cell < s.global_n_kv; ++cell) {
        const std::uint16_t expected = cell <= position ? 0 : 0xFC00;
        if (host.global_mask[std::size_t{r} * s.global_n_kv + cell] != expected) {
          return Error("Gemma4 global reference mask violates fresh causal visibility");
        }
      }
      for (std::uint32_t cell = 0; cell < s.local_n_kv; ++cell) {
        bool visible = false;
        if (cell < end) {
          const auto held = cell + ((end - 1 - cell) / g.local_capacity) * g.local_capacity;
          visible = held <= position && position - held < g.profile.window;
        }
        const std::uint16_t expected = visible ? 0 : 0xFC00;
        if (host.local_mask[std::size_t{r} * s.local_n_kv + cell] != expected) {
          return Error("Gemma4 local reference mask violates fresh causal/window visibility");
        }
      }
    }
  }
  Gemma4HostInputs out;
  out.out_ids.assign(frontier.begin(), frontier.end());
  if (!g.options.device_masks) out.masks.reserve(g.segments.size() * 2);
  out.sources.reserve(g.inputs.size());
  out.sources.emplace_back(g.input_hidden != nullptr ? g.input_hidden : g.tokens,
                           g.input_hidden != nullptr ? static_cast<const void*>(hidden.data())
                                                     : static_cast<const void*>(in.tokens.data()));
  out.sources.emplace_back(g.positions, in.positions.data());
  if (g.out_ids != nullptr) out.sources.emplace_back(g.out_ids, out.out_ids.data());
  for (std::size_t i = 0; i < g.segments.size(); ++i) {
    const auto& seg = g.segments[i];
    const auto& host = in.segments[i];
    out.sources.emplace_back(seg.global_cells, host.global_cells.data());
    out.sources.emplace_back(seg.local_cells, host.local_cells.data());
    if (g.options.device_masks) continue;
    for (const auto& [mask, logical] : {std::pair{seg.global_mask, &host.global_mask},
                                        std::pair{seg.local_mask, &host.local_mask}}) {
      auto& padded = out.masks.emplace_back(ggml_nbytes(mask) / sizeof(std::uint16_t), 0xFC00);
      std::ranges::copy(*logical, padded.begin());
      out.sources.emplace_back(mask, padded.data());
    }
  }
  std::uint64_t actual = out.out_ids.capacity() * sizeof(std::int32_t) +
                         out.sources.capacity() * sizeof(out.sources[0]) +
                         out.masks.capacity() * sizeof(out.masks[0]);
  for (const auto& mask : out.masks) actual += mask.capacity() * sizeof(std::uint16_t);
  if (actual > funded_bytes) return Error("Gemma4 reference-input allocation exceeds its grant");
  return out;
}
}  // namespace jitllm::engine
