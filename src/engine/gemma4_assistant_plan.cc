// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "engine/gemma4_assistant_plan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include "artifact/representation.h"
#include "engine/support.h"

namespace llmp::engine {
namespace {
namespace kg = kernels::ggml;
using support::Error;
bool Valid(const Gemma4Region& r, std::uint64_t need) {
  return r.address != 0 && r.address % 256 == 0 && r.bytes != 0 && need <= r.bytes &&
         r.bytes <= UINT64_MAX - r.address;
}
bool Overlap(const Gemma4Region& a, const Gemma4Region& b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
std::expected<void, std::string> Contract(const Gemma4AssistantModel& m) {
  if (m.profile == nullptr || m.binding == nullptr || m.target == nullptr ||
      m.target->profile == nullptr || m.target->binding == nullptr || m.target->state == nullptr ||
      m.resources.size() != model::kGemma4AssistantResources ||
      !model::CheckGemma4AssistantTarget(*m.profile, *m.binding, *m.target->profile,
                                         *m.target->binding) ||
      !m.target->state->Representations(*m.target->profile))
    return Error("assistant model/target contracts are incomplete or malformed");
  for (const auto* domain :
       {&m.resources, &m.target->resources, &m.target->arrays, &m.target->slots})
    for (const auto& r : *domain)
      if (!(r.address == 0 && r.bytes == 0) && !Valid(r, 0))
        return Error("assistant supplied immutable/state region cannot be bounded");
  for (std::size_t i = 0; i < m.target->slots.size(); ++i) {
    const auto& slot = m.target->slots[i];
    if (slot.bytes == 0) continue;
    if (!Valid(slot, m.target->state->bytes)) return Error("assistant target slot is short");
    for (std::size_t j = 0; j < i; ++j)
      if (m.target->slots[j].bytes != 0 && Overlap(slot, m.target->slots[j]))
        return Error("assistant target peer state overlaps");
    for (const auto* domain : {&m.resources, &m.target->resources, &m.target->arrays})
      for (const auto& r : *domain)
        if (r.bytes != 0 && Overlap(slot, r))
          return Error("assistant target cache overlaps immutable weights");
  }
  return {};
}
}  // namespace
std::expected<void, std::string> BindGemma4AssistantWeights(const Gemma4AssistantModel& m,
                                                            kg::Gemma4AssistantGraph& g) {
  if (auto checked = Contract(m); !checked) return checked;
  if (!model::CheckGemma4AssistantBinding(g.profile, g.binding) ||
      g.target_embedding.ne.size() != 2 ||
      artifact::FindGgmlType(g.target_embedding.type) == nullptr)
    return Error("assistant mutable binding is malformed");
  if (g.profile != *m.profile || g.binding != *m.binding || g.target != *m.target->profile ||
      g.target_embedding != m.target->binding->token_embd ||
      g.context != m.target->state->context || g.max_rows != m.target->state->max_rows ||
      g.local_capacity != m.target->state->local_cells ||
      g.global_capacity != m.target->state->global_cells)
    return Error("assistant graph/model contracts differ");
  if (auto checked = kg::CheckGemma4AssistantGraph(*m.profile, *m.binding, *m.target->profile,
                                                   *m.target->binding, *m.target->state, g.shape);
      !checked)
    return Error(checked.error().detail);
  if (auto checked = Gemma4AssistantSourceBytes(g, true); !checked)
    return std::unexpected(checked.error());
  for (const auto& w : g.weights)
    if (w.resource.index >= m.resources.size() ||
        !Valid(m.resources[w.resource.index], w.resource.readable))
      return Error("assistant immutable weight is absent or short");
  if (g.target_embedding.index >= m.target->resources.size() ||
      !Valid(m.target->resources[g.target_embedding.index], g.target_embedding.readable))
    return Error("assistant target token embedding is absent or short");
  for (const auto& seg : g.segments)
    if (seg.shape.slot >= m.target->slots.size() ||
        !Valid(m.target->slots[seg.shape.slot], m.target->state->bytes))
      return Error("assistant frozen target slot is absent or short");
  for (const auto& w : g.weights)
    kg::TensorArena::Bind(w.tensor, m.resources[w.resource.index].address);
  kg::TensorArena::Bind(g.embedding, m.target->resources[g.target_embedding.index].address);
  for (auto& seg : g.segments) {
    const auto base = m.target->slots[seg.shape.slot].address;
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto layer = kind == 0 ? m.profile->local_target_layer : m.profile->global_target_layer;
      kg::TensorArena::Bind(seg.caches[kind].first,
                            base + m.target->state->tensors[std::size_t{layer} * 2].offset);
      kg::TensorArena::Bind(seg.caches[kind].second,
                            base + m.target->state->tensors[std::size_t{layer} * 2 + 1].offset);
    }
  }
  return {};
}
std::expected<std::unique_ptr<Gemma4AssistantPlanned>, std::string> PlanGemma4Assistant(
    const Gemma4AssistantModel& m, const kg::Gemma4AssistantShape& shape,
    const kg::DeviceChoices& choices, std::uint64_t activations, std::uint64_t activation_bytes) {
  if (auto checked = Contract(m); !checked) return std::unexpected(checked.error());
  if (auto checked = kg::CheckGemma4AssistantGraph(*m.profile, *m.binding, *m.target->profile,
                                                   *m.target->binding, *m.target->state, shape);
      !checked)
    return Error(checked.error().detail);
  if (activations != 0) {
    const Gemma4Region region{activations, activation_bytes};
    if (!Valid(region, 0)) return Error("assistant activation region cannot be bounded");
    for (const auto* domain :
         {&m.resources, &m.target->resources, &m.target->arrays, &m.target->slots})
      for (const auto& r : *domain)
        if (r.bytes != 0 && Overlap(region, r))
          return Error("assistant activations overlap retained immutable/state storage");
  }
  auto out = std::make_unique<Gemma4AssistantPlanned>();
  auto arena =
      SizedArena(kg::Gemma4AssistantGraphTensors(shape.segments.size()), [&](kg::TensorArena& a) {
        return kg::BuildGemma4AssistantGraph(a, *m.profile, *m.binding, *m.target->profile,
                                             *m.target->binding, *m.target->state, shape)
            .has_value();
      });
  if (!arena) return std::unexpected(arena.error());
  out->arena.emplace(std::move(*arena));
  auto g = kg::BuildGemma4AssistantGraph(*out->arena, *m.profile, *m.binding, *m.target->profile,
                                         *m.target->binding, *m.target->state, shape);
  out->arena->Seal();
  if (!g) return Error(g.error().detail);
  out->graph = std::move(*g);
  if (auto checked = BindGemma4AssistantWeights(m, out->graph); !checked)
    return std::unexpected(checked.error());
  const std::array kept{out->graph.logits, out->graph.next_features};
  if (auto placed = PlaceAndPlan(*out, out->graph.nodes, out->graph.inputs, kept, choices,
                                 activations, activation_bytes);
      !placed)
    return std::unexpected(placed.error());
  return out;
}

std::expected<std::uint64_t, std::string> Gemma4AssistantSourceBytes(
    const kg::Gemma4AssistantGraph& g, bool device_features) {
  if (g.segments.empty() || g.segments.size() > model::kGemma4MaxSlots ||
      g.segments.size() != g.shape.segments.size() ||
      (g.profile != model::Gemma4Assistant26() && g.profile != model::Gemma4Assistant31()) ||
      (g.target != model::Gemma4_26BA4B() && g.target != model::Gemma4_31B()) ||
      g.profile.target_width != g.target.width || g.context == 0 || g.context > g.target.context ||
      g.max_rows == 0 || g.max_rows > std::min(g.context, model::kGemma4MaxRows) ||
      g.global_capacity != (g.context + 255) / 256 * 256 ||
      g.local_capacity != (std::min(g.context, g.target.window + g.max_rows) + 255) / 256 * 256 ||
      g.inputs.size() != 3 + g.segments.size() * 2)
    return Error("assistant host source segment count differs");
  if (g.inputs[0] != g.tokens || g.inputs[1] != g.features || g.inputs[2] != g.positions ||
      !model::CheckGemma4AssistantBinding(g.profile, g.binding) ||
      g.weights.size() != model::kGemma4AssistantResources)
    return Error("assistant mutable input/binding domain differs");
  std::array<const model::Gemma4Tensor*, model::kGemma4AssistantResources> identities{};
  const auto identity = [&](const model::Gemma4Tensor& t) { identities[t.index] = &t; };
  for (const auto* t : {&g.binding.embedding, &g.binding.output_norm, &g.binding.rope_freqs,
                        &g.binding.pre_projection, &g.binding.post_projection})
    identity(*t);
  for (const auto& l : g.binding.layers)
    for (const auto* t : {&l.attn_norm, &l.q, &l.q_norm, &l.out, &l.attn_post_norm, &l.ffn_norm,
                          &l.gate, &l.up, &l.down, &l.ffn_post_norm, &l.output_scale})
      identity(*t);
  std::array<bool, model::kGemma4AssistantResources> weight_seen{};
  std::array<const ggml_tensor*, model::kGemma4AssistantResources> weight_leaves{};
  std::size_t weight_count = 0;
  for (const auto& w : g.weights) {
    for (std::size_t i = 0; i < weight_count; ++i)
      if (w.tensor == weight_leaves[i]) return Error("assistant immutable tensor identities alias");
    weight_leaves[weight_count++] = w.tensor;
    if (w.resource.index >= identities.size() || weight_seen[w.resource.index] ||
        w.resource.ne.size() > 2 || artifact::FindGgmlType(w.resource.type) == nullptr ||
        identities[w.resource.index] == nullptr || w.resource != *identities[w.resource.index] ||
        w.tensor == nullptr || w.tensor->op != GGML_OP_NONE || w.tensor->view_src != nullptr)
      return Error("assistant immutable descriptor differs from its checked identity");
    weight_seen[w.resource.index] = true;
    const auto* type = artifact::FindGgmlType(w.resource.type);
    const auto n0 = w.resource.ne[0], n1 = w.resource.ne.size() == 2 ? w.resource.ne[1] : 1;
    if (w.tensor->type != static_cast<ggml_type>(type->id) ||
        std::cmp_not_equal(w.tensor->ne[0], n0) || std::cmp_not_equal(w.tensor->ne[1], n1) ||
        w.tensor->ne[2] != 1 || w.tensor->ne[3] != 1 || w.tensor->nb[0] != type->block_bytes ||
        w.tensor->nb[1] != n0 / type->block_elements * type->block_bytes ||
        w.tensor->nb[2] != w.tensor->nb[1] * n1 || w.tensor->nb[3] != w.tensor->nb[2])
      return Error("assistant immutable tensor shape/stride differs");
  }
  if (g.target_embedding.ne.size() != 2 || g.target_embedding.ne[0] != g.target.width ||
      g.target_embedding.ne[1] != g.target.vocab ||
      artifact::FindGgmlType(g.target_embedding.type) == nullptr || g.embedding == nullptr)
    return Error("assistant target embedding descriptor is malformed");
  const auto* emb_type = artifact::FindGgmlType(g.target_embedding.type);
  const auto emb_pitch =
      std::uint64_t{g.target.width} / emb_type->block_elements * emb_type->block_bytes;
  if (g.embedding->type != static_cast<ggml_type>(emb_type->id) ||
      g.embedding->op != GGML_OP_NONE || g.embedding->view_src != nullptr ||
      g.embedding->ne[0] != g.target.width || g.embedding->ne[1] != g.target.vocab ||
      g.embedding->ne[2] != 1 || g.embedding->ne[3] != 1 ||
      g.embedding->nb[0] != emb_type->block_bytes || g.embedding->nb[1] != emb_pitch ||
      g.embedding->nb[2] != emb_pitch * g.target.vocab ||
      g.embedding->nb[3] != g.embedding->nb[2] ||
      std::ranges::find(weight_leaves, g.embedding) != weight_leaves.end())
    return Error("assistant target embedding shape/stride differs");
  std::array<bool, model::kGemma4MaxSlots> seen{};
  std::array<const ggml_tensor*, model::kGemma4MaxSlots * 2> mask_leaves{};
  std::array<const ggml_tensor*, model::kGemma4MaxSlots * 4> cache_leaves{};
  std::size_t cache_count = 0;
  std::size_t mask_count = 0;
  for (std::size_t i = 0; i < g.segments.size(); ++i) {
    const auto& seg = g.segments[i];
    const auto& s = seg.shape;
    const auto& initial = g.shape.segments[i];
    if (s.slot >= seen.size() || seen[s.slot] || s.prefix != initial.prefix || s != initial ||
        s.prefix == 0 || s.prefix >= g.context ||
        s.local_n_kv != std::min(g.local_capacity, (s.prefix + 255) / 256 * 256) ||
        s.global_n_kv != std::min(g.global_capacity, (s.prefix + 255) / 256 * 256))
      return Error("assistant mutable segment/read footprint differs");
    seen[s.slot] = true;
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto d = kind == 0 ? g.target.local_head_dim : g.target.global_head_dim;
      const auto heads = kind == 0 ? g.profile.local_kv_heads : g.profile.global_kv_heads;
      const auto cells = kind == 0 ? s.local_n_kv : s.global_n_kv;
      for (const auto* cache : {seg.caches[kind].first, seg.caches[kind].second}) {
        for (std::size_t j = 0; j < cache_count; ++j)
          if (cache == cache_leaves[j])
            return Error("assistant independent caches alias a descriptor");
        cache_leaves[cache_count++] = cache;
        const auto stride = std::uint64_t{d} * heads * 2;
        if (cache == nullptr || cache->type != GGML_TYPE_F16 || cache->op != GGML_OP_NONE ||
            cache->view_src != nullptr || cache->ne[0] != d * heads || cache->ne[1] != cells ||
            cache->ne[2] != 1 || cache->ne[3] != 1 || cache->nb[0] != 2 || cache->nb[1] != stride ||
            cache->nb[2] != stride * cells || cache->nb[3] != cache->nb[2])
          return Error("assistant mutable frozen cache descriptor differs");
      }
      const auto* mask = seg.masks[kind];
      for (std::size_t j = 0; j < mask_count; ++j)
        if (mask == mask_leaves[j]) return Error("assistant independent masks alias a source");
      if (g.inputs[3 + i * 2 + kind] != mask) return Error("assistant mask input order differs");
      mask_leaves[mask_count++] = mask;
    }
  }
  for (const auto [t, type, width] :
       {std::tuple{g.tokens, GGML_TYPE_I32, std::uint32_t{1}},
        std::tuple{g.positions, GGML_TYPE_I32, std::uint32_t{1}},
        std::tuple{g.features, GGML_TYPE_F32, g.profile.target_width}}) {
    if (t == nullptr || t->type != type || t->op != GGML_OP_NONE || t->view_src != nullptr ||
        t->ne[0] != (width == 1 ? static_cast<std::int64_t>(g.segments.size()) : width) ||
        t->ne[1] != (width == 1 ? 1 : static_cast<std::int64_t>(g.segments.size())) ||
        t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != 4 ||
        t->nb[1] != static_cast<std::uint64_t>(t->ne[0]) * 4 ||
        t->nb[2] != t->nb[1] * static_cast<std::uint64_t>(t->ne[1]) || t->nb[3] != t->nb[2] ||
        std::ranges::count(g.inputs, t) != 1)
      return Error("assistant row source is not its bounded host input");
  }
  std::uint64_t bytes = g.segments.size() * 2 * sizeof(std::int32_t) +
                        g.segments.size() * 2 * sizeof(std::vector<std::uint16_t>) +
                        (g.inputs.size() - (device_features ? 1U : 0U)) *
                            sizeof(std::pair<ggml_tensor*, const void*>);
  for (const auto& seg : g.segments)
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto cells = kind == 0 ? seg.shape.local_n_kv : seg.shape.global_n_kv;
      const auto* t = seg.masks[kind];
      if (t == nullptr || cells == 0 || cells > INT32_MAX / 64 || t->type != GGML_TYPE_F16 ||
          t->op != GGML_OP_NONE || t->view_src != nullptr || t->ne[0] != cells || t->ne[1] != 32 ||
          t->ne[2] != 1 || t->ne[3] != 1 || t->nb[0] != 2 || t->nb[1] != std::uint64_t{cells} * 2 ||
          t->nb[2] != std::uint64_t{cells} * 64 || t->nb[3] != t->nb[2] ||
          std::ranges::find(g.inputs, t) == g.inputs.end())
        return Error("assistant mask is not its complete host input");
      bytes += std::uint64_t{cells} * 64;
    }
  return bytes;
}
std::expected<Gemma4AssistantHostInputs, std::string> Gemma4AssistantSources(
    const kg::Gemma4AssistantGraph& g, std::span<const Gemma4AssistantInput> input,
    std::span<const float> features, bool device_features, std::uint64_t funded_bytes) {
  const auto bytes = Gemma4AssistantSourceBytes(g, device_features);
  if (!bytes || *bytes > funded_bytes) return Error("assistant host masks/sources are unfunded");
  if (input.size() != g.segments.size() ||
      (device_features ? !features.empty()
                       : features.size() != input.size() * g.profile.target_width) ||
      std::ranges::any_of(features, [](float v) { return !std::isfinite(v); }))
    return Error("assistant feature/owner inputs differ");
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto& x = input[i];
    const auto& s = g.segments[i].shape;
    if (x.slot != s.slot || x.prefix == 0 || x.prefix >= g.context || x.anchor < 0 ||
        std::cmp_greater_equal(x.anchor, g.target.vocab) ||
        s.local_n_kv != std::min(g.local_capacity, (x.prefix + 255) / 256 * 256) ||
        s.global_n_kv != std::min(g.global_capacity, (x.prefix + 255) / 256 * 256))
      return Error("assistant frozen endpoint/query input differs from graph read widths");
  }
  Gemma4AssistantHostInputs out;
  out.tokens.reserve(input.size());
  out.positions.reserve(input.size());
  out.masks.reserve(input.size() * 2);
  out.sources.reserve(g.inputs.size() - (device_features ? 1U : 0U));
  for (const auto& x : input) {
    out.tokens.push_back(x.anchor);
    out.positions.push_back(static_cast<std::int32_t>(x.prefix));
  }
  out.sources.emplace_back(g.tokens, out.tokens.data());
  out.sources.emplace_back(g.positions, out.positions.data());
  if (!device_features) out.sources.emplace_back(g.features, features.data());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto prefix = input[i].prefix;
    const auto& seg = g.segments[i];
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto cells = kind == 0 ? seg.shape.local_n_kv : seg.shape.global_n_kv;
      auto& mask = out.masks.emplace_back(std::size_t{cells} * 32, 0xFC00);
      for (std::uint32_t cell = 0; cell < cells; ++cell) {
        bool visible = cell < prefix;
        if (kind == 0 && visible) {
          const auto held = cell + ((prefix - 1 - cell) / g.local_capacity) * g.local_capacity;
          visible = prefix - held < g.target.window;
        }
        if (visible) mask[cell] = 0;
      }
      out.sources.emplace_back(seg.masks[kind], mask.data());
    }
  }
  std::uint64_t actual = (out.tokens.capacity() + out.positions.capacity()) * sizeof(std::int32_t) +
                         out.masks.capacity() * sizeof(out.masks[0]) +
                         out.sources.capacity() * sizeof(out.sources[0]);
  for (const auto& mask : out.masks) actual += mask.capacity() * sizeof(std::uint16_t);
  if (actual > funded_bytes) return Error("assistant host allocation exceeds its grant");
  return out;
}
}  // namespace llmp::engine
