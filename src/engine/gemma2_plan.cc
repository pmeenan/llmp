// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma2_plan.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/representation.h"
#include "engine/graph_mask_inputs.h"
#include "engine/support.h"
#include "model/host_mask.h"

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
using support::Error;
bool RegionValid(const Gemma2Region& r, std::uint64_t required) {
  return r.address != 0 && r.address % 256 == 0 && required <= r.bytes &&
         r.bytes <= UINT64_MAX - r.address;
}
bool Overlap(const Gemma2Region& a, const Gemma2Region& b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
}  // namespace

std::expected<void, std::string> BindGemma2Weights(const Gemma2Model& m, kg::Gemma2Graph& g) {
  if (m.profile == nullptr || m.binding == nullptr || m.state == nullptr ||
      !model::CheckGemma2Binding(*m.profile, *m.binding) || !m.state->Representations(*m.profile)) {
    return Error("incomplete or malformed Gemma2 model");
  }
  if (g.profile != *m.profile || g.binding != *m.binding || g.context != m.state->context ||
      g.max_rows != m.state->max_rows || g.global_capacity != m.state->global_cells ||
      g.local_capacity != m.state->local_cells || g.state_bytes != m.state->bytes ||
      g.options != m.options)
    return Error("Gemma2 graph/model contracts differ");
  if (!Gemma2SourceBytes(g)) return Error("invalid Gemma2 graph input descriptors");
  std::array<const model::Gemma2Tensor*, 288> identities{};
  const auto record = [&](const model::Gemma2Tensor& tensor) {
    identities[tensor.index] = &tensor;
  };
  record(m.binding->token_embd);
  record(m.binding->output_norm);
  for (const auto& layer : m.binding->layers)
    for (const auto* tensor :
         {&layer.attn_norm, &layer.q, &layer.k, &layer.v, &layer.out, &layer.attn_post_norm,
          &layer.ffn_norm, &layer.gate, &layer.up, &layer.down, &layer.ffn_post_norm})
      record(*tensor);
  std::array<bool, 288> seen{};
  for (const auto& leaf : g.weights) {
    const auto index = leaf.resource.index;
    if (index >= identities.size() || identities[index] == nullptr || seen[index] ||
        *identities[index] != leaf.resource || leaf.tensor == nullptr ||
        leaf.tensor->op != GGML_OP_NONE || leaf.tensor->view_src != nullptr)
      return Error("Gemma2 weight descriptor differs from its binding");
    seen[index] = true;
    const auto* type = artifact::FindGgmlType(leaf.resource.type);
    if (static_cast<std::uint32_t>(leaf.tensor->type) != type->id)
      return Error("Gemma2 weight descriptor type differs from its binding");
    std::uint64_t stride = type->block_bytes;
    for (std::size_t dim = 0; dim < 4; ++dim) {
      const auto extent = dim < leaf.resource.ne.size() ? leaf.resource.ne[dim] : 1;
      if (leaf.tensor->ne[dim] != static_cast<std::int64_t>(extent) ||
          leaf.tensor->nb[dim] != stride)
        return Error("Gemma2 weight descriptor shape/stride differs from its binding");
      stride *= dim == 0 ? extent / type->block_elements : extent;
    }
  }
  std::array<const ggml_tensor*, model::kGemma2MaxSlots * 52> cache_descriptors{};
  std::size_t cache_count = 0;
  for (const auto& segment : g.segments) {
    if (segment.caches.size() != m.profile->layers)
      return Error("Gemma2 cache descriptor layer count differs");
    const auto count = m.options.layer_count == 0 ? m.profile->layers - m.options.first_layer
                                                  : m.options.layer_count;
    for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
      const bool active = il >= m.options.first_layer && il < m.options.first_layer + count;
      const auto [key, value] = segment.caches[il];
      if (!active) {
        if (key != nullptr || value != nullptr)
          return Error("Gemma2 inactive cache descriptor is present");
        continue;
      }
      if (key == value) return Error("Gemma2 K/V descriptors alias");
      for (const auto* tensor : {key, value}) {
        if (std::ranges::contains(std::span(cache_descriptors).first(cache_count), tensor))
          return Error("Gemma2 independent cache descriptors alias");
        cache_descriptors[cache_count++] = tensor;
        const auto cells = m.profile->local(il) ? m.state->local_cells : m.state->global_cells;
        const auto width = m.profile->key_dim * m.profile->kv_heads;
        if (tensor == nullptr || tensor->op != GGML_OP_NONE || tensor->view_src != nullptr ||
            tensor->type != GGML_TYPE_F16 || tensor->ne[0] != width || tensor->ne[1] != cells ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != 2 ||
            tensor->nb[1] != std::uint64_t{width} * 2 ||
            tensor->nb[2] != std::uint64_t{width} * cells * 2 || tensor->nb[3] != tensor->nb[2])
          return Error("Gemma2 cache descriptor differs from its physical state layout");
      }
    }
  }
  // Rebinding lists must name the leaves the real consumers still read.
  // Unused mask/cell inputs are legitimate in a diagnostic layer scope;
  // weights, active caches and primary inputs must actually be reachable.
  // These fixed transient tables cover the checked H8/26-layer/16-slot
  // descriptor envelope; they allocate no new heap or launch workspace.
  struct Root {
    const ggml_tensor* tensor = nullptr;
    bool required = false, reached = false;
  };
  constexpr std::size_t kRootLimit =
      288 + model::kGemma2MaxSlots * 52 + 3 + model::kGemma2MaxSlots * 4;
  std::array<Root, kRootLimit> roots{};
  std::size_t root_count = 0;
  const auto add_root = [&](const ggml_tensor* tensor, bool required) {
    roots[root_count++] = {.tensor = tensor, .required = required};
  };
  for (const auto& leaf : g.weights) add_root(leaf.tensor, true);
  for (const auto* tensor : std::span(cache_descriptors).first(cache_count)) add_root(tensor, true);
  for (const auto* tensor : g.inputs)
    add_root(tensor, tensor == g.tokens || tensor == g.input_hidden || tensor == g.positions ||
                         tensor == g.out_ids);
  auto declared = std::span(roots).first(root_count);
  const std::less<const ggml_tensor*> less;
  std::ranges::sort(declared, less, &Root::tensor);
  if (std::ranges::adjacent_find(declared, {}, &Root::tensor) != declared.end())
    return Error("Gemma2 weight/cache/input root declarations alias");
  for (const auto& root : declared)
    if (root.tensor == nullptr || root.tensor->op != GGML_OP_NONE ||
        root.tensor->view_src != nullptr ||
        std::ranges::any_of(root.tensor->src, [](const auto* source) { return source != nullptr; }))
      return Error("Gemma2 declared root is not an independent leaf");
  struct Node {
    const ggml_tensor* tensor = nullptr;
    std::size_t order = 0;
  };
  constexpr std::size_t kNodeLimit = 128 + 26 * (96 + model::kGemma2MaxSlots * 48);
  std::array<Node, kNodeLimit> nodes{};
  if (g.nodes.empty() || g.nodes.size() > nodes.size())
    return Error("Gemma2 consumer graph exceeds its bounded descriptor envelope");
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    if (g.nodes[i] == nullptr) return Error("Gemma2 consumer graph has a null node");
    nodes[i] = {g.nodes[i], i};
  }
  auto consumers = std::span(nodes).first(g.nodes.size());
  std::ranges::sort(consumers, less, &Node::tensor);
  if (std::ranges::adjacent_find(consumers, {}, &Node::tensor) != consumers.end())
    return Error("Gemma2 consumer graph repeats a node");
  const auto reaches = [&](const ggml_tensor* tensor, std::size_t order) {
    if (tensor->op == GGML_OP_NONE) {
      const auto root = std::ranges::lower_bound(declared, tensor, less, &Root::tensor);
      if (root == declared.end() || root->tensor != tensor) return false;
      root->reached = true;
      return true;
    }
    const auto node = std::ranges::lower_bound(consumers, tensor, less, &Node::tensor);
    return node != consumers.end() && node->tensor == tensor && node->order < order;
  };
  for (std::size_t i = 0; i < g.nodes.size(); ++i) {
    const auto* node = g.nodes[i];
    if (node->op == GGML_OP_NONE && !reaches(node, i))
      return Error("Gemma2 consumer graph contains an undeclared leaf");
    for (const auto* parent : node->src)
      if (parent != nullptr && !reaches(parent, i))
        return Error("Gemma2 consumer uses an undeclared root or unordered parent");
    if (node->view_src != nullptr && !reaches(node->view_src, i))
      return Error("Gemma2 consumer view uses an undeclared root or unordered parent");
  }
  for (const auto& root : declared)
    if (root.required && !root.reached)
      return Error("Gemma2 required weight/cache/input leaf is detached from its consumers");
  for (const auto* output : {g.hidden, g.logits, g.greedy}) {
    if (output == nullptr) continue;
    const auto node = std::ranges::lower_bound(consumers, output, less, &Node::tensor);
    if (node == consumers.end() || node->tensor != output)
      return Error("Gemma2 graph output is detached from its consumers");
  }
  // Preflight ALL addresses before mutating descriptors. The logical region
  // supplies a bound, not proof of catalog residency or initialized KV cells.
  for (const auto& w : g.weights) {
    const auto& domain = m.resources;
    const auto index = w.resource.index;
    if (index >= domain.size()) return Error("Gemma2 weight index exceeds its address domain");
    const auto bytes = w.resource.readable;
    if (!RegionValid(domain[index], bytes))
      return Error("Gemma2 weight region is absent, short or overflowing");
  }
  for (const auto* domain : {&m.resources, &m.slots}) {
    for (const auto& occupied : *domain) {
      if (occupied.address == 0 && occupied.bytes == 0) continue;
      if (occupied.bytes == 0 || !RegionValid(occupied, 0)) {
        return Error("Gemma2 supplied weight/state region cannot be bounded");
      }
    }
  }
  for (const auto& seg : g.segments) {
    if (seg.shape.slot >= m.slots.size() || !RegionValid(m.slots[seg.shape.slot], m.state->bytes)) {
      return Error("Gemma2 slot region is absent, short or overflowing");
    }
    const auto& slot = m.slots[seg.shape.slot];
    for (const auto& other : m.slots) {
      if (&other != &slot && other.bytes != 0 && Overlap(slot, other)) {
        return Error("Gemma2 independent slot state regions overlap");
      }
    }
    for (const auto* domain : {&m.resources}) {
      for (const auto& weight : *domain) {
        if (weight.bytes != 0 && Overlap(slot, weight))
          return Error("Gemma2 state overlaps immutable weights");
      }
    }
  }
  for (const auto& w : g.weights) {
    const auto& domain = m.resources;
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

std::expected<std::unique_ptr<Gemma2Planned>, std::string> PlanGemma2Chunk(
    const Gemma2Model& m, const kg::Gemma2ChunkShape& shape, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes, std::span<const std::string> keep) {
  return PlanGemma2Chunk(m, shape, choices, activations, activation_bytes, keep, std::nullopt);
}

std::expected<std::unique_ptr<Gemma2Planned>, std::string> PlanGemma2Chunk(
    const Gemma2Model& m, const kg::Gemma2ChunkShape& shape, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes, std::span<const std::string> keep,
    std::optional<ActivationMeasurement> measurement) {
  if (measurement && activations != 0)
    return Error("measurement-only Gemma2 planning cannot use activation storage");
  if (m.profile == nullptr || m.binding == nullptr || m.state == nullptr) {
    return Error("incomplete Gemma2 model");
  }
  if (auto checked = kg::CheckGemma2Graph(*m.profile, *m.binding, *m.state, shape, m.options);
      !checked)
    return Error(checked.error().detail);
  if (activations != 0) {
    const Gemma2Region region{activations, activation_bytes};
    if (activation_bytes == 0 || !RegionValid(region, 0)) {
      return Error("Gemma2 activation region is empty, misaligned or overflowing");
    }
    // A diagnostic graph uses only some layers/slots, but their peers still
    // retain immutable weights and continuation state. Per-op alias checks
    // alone cannot protect an operand that this particular chunk never reads.
    for (const auto* domain : {&m.resources, &m.slots}) {
      for (const auto& occupied : *domain) {
        if (occupied.address == 0 && occupied.bytes == 0) continue;
        if (occupied.bytes == 0 || !RegionValid(occupied, 0) || Overlap(region, occupied)) {
          return Error("Gemma2 activations overlap or cannot bound supplied weight/state storage");
        }
      }
    }
  }
  auto out = std::make_unique<Gemma2Planned>();
  // Original MMVQ preparation cannot replace the requested one-column sums.
  const decltype(choices.dense_mmvq_shape) no_dense_sharing;
  const auto& selected_dense_mmvq_shape =
      choices.row_invariant ? no_dense_sharing : choices.dense_mmvq_shape;
  auto arena = SizedArena(kg::Gemma2GraphTensors(*m.profile, shape.segments.size()),
                          [&](kg::TensorArena& a) {
                            return kg::BuildGemma2Graph(a, *m.profile, *m.binding, *m.state, shape,
                                                        m.options, selected_dense_mmvq_shape)
                                .has_value();
                          });
  if (!arena) return std::unexpected(arena.error());
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildGemma2Graph(*out->arena, *m.profile, *m.binding, *m.state, shape, m.options,
                                    selected_dense_mmvq_shape);
  out->arena->Seal();
  if (!graph) return Error(graph.error().detail);
  out->graph = std::move(*graph);
  auto& g = out->graph;
  if (auto bound = BindGemma2Weights(m, g); !bound) return std::unexpected(bound.error());
  std::vector<ggml_tensor*> kept;
  if (g.logits != nullptr) kept.push_back(g.logits);
  if (g.greedy != nullptr) kept.push_back(g.greedy);
  if (g.hidden != nullptr) kept.push_back(g.hidden);
  for (const auto& name : keep) {
    auto* t = g.Named(name);
    if (t == nullptr) return Error(std::format("Gemma2 graph names no {}", name));
    kept.push_back(t);
  }
  // No Gemma4 owner, norm, routing, quant writer or RoPE-store policy is
  // enabled here. Callers supply device family selectors; the ordinary
  // foundation's default DeviceChoices retains primitive operations.
  if (auto placed = PlaceAndPlan(*out, g.nodes, g.inputs, kept, choices, activations,
                                 activation_bytes, measurement);
      !placed)
    return std::unexpected(placed.error());
  return out;
}

std::expected<std::uint64_t, std::string> Gemma2SourceBytes(const kg::Gemma2Graph& g) {
  auto state = model::Gemma2State(g.profile, g.context, g.max_rows);
  if (!state || state->global_cells != g.global_capacity ||
      state->local_cells != g.local_capacity || state->bytes != g.state_bytes ||
      !kg::CheckGemma2Graph(g.profile, g.binding, *state, g.shape, g.options) ||
      g.segments.size() != g.shape.segments.size())
    return Error("invalid Gemma2 graph source contract");
  std::uint64_t rows = 0;
  for (const auto& segment : g.shape.segments) rows += segment.rows;
  const auto leaf = [](const ggml_tensor* tensor, ggml_type type, std::uint64_t width,
                       std::uint64_t columns, std::uint64_t element) {
    return tensor != nullptr && tensor->type == type && tensor->op == GGML_OP_NONE &&
           tensor->view_src == nullptr && tensor->ne[0] == static_cast<std::int64_t>(width) &&
           tensor->ne[1] == static_cast<std::int64_t>(columns) && tensor->ne[2] == 1 &&
           tensor->ne[3] == 1 && tensor->nb[0] == element && tensor->nb[1] == width * element &&
           tensor->nb[2] == width * columns * element && tensor->nb[3] == tensor->nb[2];
  };
  if (!leaf(g.positions, GGML_TYPE_I32, rows, 1, 4) ||
      (g.options.hidden_input
           ? (g.tokens != nullptr || !leaf(g.input_hidden, GGML_TYPE_F32, g.profile.width, rows, 4))
           : (g.input_hidden != nullptr || !leaf(g.tokens, GGML_TYPE_I32, rows, 1, 4))) ||
      (g.shape.outputs == 0 ? g.out_ids != nullptr
                            : !leaf(g.out_ids, GGML_TYPE_I32, g.shape.outputs, 1, 4)))
    return Error("malformed Gemma2 token, position, hidden or frontier source");
  std::vector<const ggml_tensor*> expected{g.options.hidden_input ? g.input_hidden : g.tokens,
                                           g.positions};
  if (g.out_ids != nullptr) expected.push_back(g.out_ids);
  std::uint32_t first_row = 0;
  for (std::size_t i = 0; i < g.segments.size(); ++i) {
    const auto& segment = g.segments[i];
    if (segment.shape != g.shape.segments[i] || segment.first_row != first_row ||
        !leaf(segment.global_cells, GGML_TYPE_I64, segment.shape.rows, 1, 8) ||
        !leaf(segment.local_cells, GGML_TYPE_I64, segment.shape.rows, 1, 8))
      return Error("malformed Gemma2 segment source");
    first_row += segment.shape.rows;
    expected.push_back(segment.global_cells);
    expected.push_back(segment.local_cells);
    if (!g.options.device_masks) {
      expected.push_back(segment.global_mask);
      expected.push_back(segment.local_mask);
    }
  }
  if (expected.size() != g.inputs.size()) return Error("Gemma2 source set differs from its graph");
  for (const auto* tensor : expected)
    if (std::ranges::count(expected, tensor) != 1 || std::ranges::count(g.inputs, tensor) != 1)
      return Error("duplicate or absent Gemma2 source");
  std::uint64_t bytes =
      std::uint64_t{g.shape.outputs} * sizeof(std::int32_t) +
      g.inputs.size() * sizeof(std::pair<ggml_tensor*, const void*>) +
      (g.options.device_masks ? 0 : g.segments.size() * 2 * sizeof(std::vector<std::uint16_t>));
  for (const auto& seg : g.segments) {
    for (const auto [mask, capacity, window, cells] :
         {std::tuple{seg.global_mask, g.global_capacity, 0U, seg.shape.global_n_kv},
          std::tuple{seg.local_mask, g.local_capacity, g.profile.window, seg.shape.local_n_kv}}) {
      auto source =
          GraphMaskSourceBytes(mask, g.positions, g.nodes, g.inputs, g.options.device_masks,
                               seg.first_row, seg.shape.rows, cells, capacity, window, g.context);
      if (!source || *source > UINT64_MAX - bytes)
        return Error("invalid or overflowing graph mask source");
      bytes += *source;
    }
  }
  return bytes;
}

std::expected<Gemma2HostInputs, std::string> Gemma2Sources(const kg::Gemma2Graph& g,
                                                           const model::Gemma2ChunkInputs& in,
                                                           std::span<const std::int32_t> frontier,
                                                           std::span<const float> hidden,
                                                           std::uint64_t funded_bytes) {
  auto bytes = Gemma2SourceBytes(g);
  if (!bytes || *bytes > funded_bytes) return Error("Gemma2 host sources are not funded");
  const auto rows = static_cast<std::size_t>(g.positions->ne[0]);
  if (in.tokens.size() != rows || in.positions.size() != rows ||
      in.segments.size() != g.segments.size() || frontier.size() != g.shape.outputs ||
      (g.input_hidden == nullptr
           ? !hidden.empty()
           : hidden.size() != rows * static_cast<std::size_t>(g.input_hidden->ne[0]))) {
    return Error("Gemma2 host input/hidden/frontier size differs from graph");
  }
  for (const auto id : frontier) {
    if (id < 0 || std::cmp_greater_equal(id, rows))
      return Error("Gemma2 frontier ID exceeds chunk");
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
      return Error("Gemma2 segment or reference mask differs from graph");
    }
    // Cell/value correctness comes from the checked chunk builder; refuse
    // edited public indices and absolute positions before any staging.
    for (std::uint32_t r = 0; r < s.rows; ++r) {
      const auto position = host.n_past + r;
      if (in.positions[seg.first_row + r] != static_cast<std::int32_t>(position) ||
          host.global_cells[r] != position || host.local_cells[r] != position % g.local_capacity ||
          in.tokens[seg.first_row + r] < 0 ||
          std::cmp_greater_equal(in.tokens[seg.first_row + r], g.profile.vocab)) {
        return Error("Gemma2 host position or cache index is invalid");
      }
      if (g.options.device_masks) continue;
      const auto end = host.n_past + s.rows;
      const auto global = model::HostMaskPrefix(s.global_n_kv, std::uint64_t{position} + 1);
      const auto local =
          model::HostMaskRing(s.local_n_kv, position, end, g.local_capacity, g.profile.window);
      if (!global || !model::MatchHostMask(std::span{host.global_mask}.subspan(
                                               std::size_t{r} * s.global_n_kv, s.global_n_kv),
                                           *global))
        return Error("Gemma2 global reference mask violates fresh causal visibility");
      if (!local || !model::MatchHostMask(std::span{host.local_mask}.subspan(
                                              std::size_t{r} * s.local_n_kv, s.local_n_kv),
                                          *local))
        return Error("Gemma2 local reference mask violates fresh causal/window visibility");
    }
  }
  Gemma2HostInputs out;
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
      StageHostGraphMask(mask, *logical, out.masks, out.sources);
    }
  }
  std::uint64_t actual = out.out_ids.capacity() * sizeof(std::int32_t) +
                         out.sources.capacity() * sizeof(out.sources[0]) +
                         out.masks.capacity() * sizeof(out.masks[0]);
  for (const auto& mask : out.masks) actual += mask.capacity() * sizeof(std::uint16_t);
  if (actual > funded_bytes) return Error("Gemma2 reference-input allocation exceeds its grant");
  return out;
}
}  // namespace jitllm::engine
