// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Complete attention quads: shared products retain their existing row group.
#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma4_graph.h"
#include "kernels/ggml/llmp_ops.h"

namespace llmp::kernels::ggml {
namespace {
namespace md = llmp::model;
constexpr std::size_t kExtraPerLayer = 64;
std::unexpected<KernelFailure> Reject(std::string_view why) {
  return std::unexpected(KernelFailure{.detail = std::string(why)});
}
bool Shape(const Gemma4Graph& g, Gemma4AttentionMode mode) {
  if ((g.profile != md::Gemma4_31B() && g.profile != md::Gemma4_26BA4B()) ||
      (mode == Gemma4AttentionMode::kPacked && g.context > 16384) ||
      (g.max_rows == 0 || g.max_rows > md::kGemma4MaxRows) ||
      g.shape.outputs != g.segments.size() || g.shape.feature_outputs != 0 ||
      g.segments.size() < 2 || g.segments.size() > 12 ||
      (mode == Gemma4AttentionMode::kPacked && g.segments.size() < 4) ||
      g.shape.segments.size() != g.segments.size() || g.options.first_layer != 0 ||
      g.options.layer_count != 0 || g.options.hidden_input || !g.options.head ||
      g.options.shared_q8 || g.options.rope_store || g.normalized_features != nullptr)
    return false;
  // Current() checks full backing containment separately from the executed
  // <=16K read view. Owner attention can use an approved 262K backing while
  // keeping the same bounded shader domain; packed attention keeps its limit.
  const auto parent_limit = mode == Gemma4AttentionMode::kOwners ? (1ULL << 30U) : (64ULL << 20U);
  for (std::uint32_t layer = 0; layer < g.profile.layers; ++layer) {
    const auto capacity = g.profile.local(layer) ? g.local_capacity : g.global_capacity;
    const auto bytes =
        std::uint64_t{capacity} * g.profile.head_dim(layer) * g.profile.kv_heads(layer) * 2;
    if (bytes > parent_limit) return false;
  }
  for (std::size_t i = 0; i < g.segments.size(); ++i) {
    const auto& s = g.segments[i];
    if (s.shape != g.shape.segments[i] || s.shape.rows != 1 || s.first_row != i ||
        s.caches.size() != g.profile.layers)
      return false;
  }
  return true;
}
bool QuadShape(const Gemma4Graph& g, std::size_t first, std::size_t count) {
  const bool common = g.options.common_owner_reads &&
                      g.options.attention_mode == Gemma4AttentionMode::kOwners &&
                      g.segments.size() == 2 && first == 0 && count == 2;
  for (std::size_t i = first; i < first + count; ++i) {
    for (const auto [read, capacity] :
         {std::pair{g.segments[i].shape.local_n_kv, g.local_capacity},
          std::pair{g.segments[i].shape.global_n_kv, g.global_capacity}})
      if (read < 256 || read > 16384 || read % 256 != 0 || read > capacity) return false;
    // Validate every actual root before admitting common-width activations.
    for (std::uint32_t layer = 0; layer < g.profile.layers; ++layer) {
      const auto read =
          g.profile.local(layer) ? g.segments[i].shape.local_n_kv : g.segments[i].shape.global_n_kv;
      if (std::uint64_t{read} * g.profile.head_dim(layer) * g.profile.kv_heads(layer) * 2 >
          (64ULL << 20U))
        return false;
    }
    if (!common && (g.segments[i].shape.local_n_kv != g.segments[first].shape.local_n_kv ||
                    g.segments[i].shape.global_n_kv != g.segments[first].shape.global_n_kv))
      return false;
  }
  return true;
}
bool Dims(const ggml_tensor* t, ggml_type type, std::array<std::int64_t, 4> ne) {
  return t != nullptr && t->type == type && std::equal(ne.begin(), ne.end(), t->ne);
}
struct Layer {
  std::array<ggml_tensor*, 4> flash{}, q{}, k{}, v{}, mask{};
  std::array<ggml_tensor*, 8> writes{};
};
// Inspect only exact direct builder edges, never follow arbitrary root chains.
bool Inspect(const Gemma4Graph& g, std::uint32_t layer, std::size_t first, std::size_t count,
             Layer& out) {
  const auto d = g.profile.head_dim(layer), kvh = g.profile.kv_heads(layer);
  const auto kvw = std::size_t{d} * kvh;
  const auto cells = g.profile.local(layer) ? g.local_capacity : g.global_capacity;
  for (std::size_t owner = 0; owner < count; ++owner) {
    const auto read = g.profile.local(layer) ? g.segments[first + owner].shape.local_n_kv
                                             : g.segments[first + owner].shape.global_n_kv;
    auto* flat = g.Named(
        std::format("blk.{}.slot.{}.attention", layer, g.segments[first + owner].shape.slot));
    if (!Dims(flat, GGML_TYPE_F32, {std::int64_t{d} * g.profile.heads, 1, 1, 1}) ||
        flat->op != GGML_OP_RESHAPE || flat->src[0] == nullptr || flat->view_src != flat->src[0] ||
        flat->view_offs != 0)
      return false;
    auto* f = flat->src[0];
    if (!Dims(f, GGML_TYPE_F32, {d, g.profile.heads, 1, 1}) || f->op != GGML_OP_FLASH_ATTN_EXT ||
        f->view_src != nullptr || !Dims(f->src[0], GGML_TYPE_F32, {d, 1, g.profile.heads, 1}) ||
        f->src[0]->op != GGML_OP_PERMUTE || f->src[0]->src[0] == nullptr ||
        !Dims(f->src[0]->src[0], GGML_TYPE_F32, {d, g.profile.heads, 1, 1}))
      return false;
    out.flash[owner] = f;
    out.q[owner] = f->src[0]->src[0];
    out.mask[owner] = g.profile.local(layer) ? g.segments[first + owner].local_mask
                                             : g.segments[first + owner].global_mask;
    if (f->src[3] != out.mask[owner] || !Dims(out.mask[owner], GGML_TYPE_F16, {read, 32, 1, 1}) ||
        !ggml_is_contiguous(out.mask[owner]))
      return false;
    for (std::size_t which = 0; which < 2; ++which) {
      auto* leaf = which == 0 ? g.segments[first + owner].caches[layer].first
                              : g.segments[first + owner].caches[layer].second;
      auto* cache = f->src[which + 1];
      if (!Dims(leaf, GGML_TYPE_F16, {static_cast<std::int64_t>(kvw), cells, 1, 1}) ||
          leaf->op != GGML_OP_NONE || leaf->view_src != nullptr || !ggml_is_contiguous(leaf) ||
          !Dims(cache, GGML_TYPE_F16, {d, read, kvh, 1}) || cache->op != GGML_OP_PERMUTE ||
          cache->src[0] == nullptr || cache->src[0]->op != GGML_OP_VIEW ||
          cache->src[0]->view_src != leaf || cache->src[0]->view_offs != 0 ||
          cache->src[0]->src[0] != leaf || cache->nb[0] != 2 || cache->nb[1] != kvw * 2 ||
          cache->nb[2] != d * 2)
        return false;
      ggml_tensor* write = nullptr;
      for (auto* node : g.nodes) {
        if (node->op != GGML_OP_SET_ROWS || node->src[2] != leaf) continue;
        if (write != nullptr || node->view_src != leaf || node->view_offs != 0 ||
            node->type != GGML_TYPE_F16 || !ggml_is_contiguous(node) ||
            !Dims(node->src[0], GGML_TYPE_F32, {static_cast<std::int64_t>(kvw), 1, 1, 1}) ||
            node->src[1] != (g.profile.local(layer) ? g.segments[first + owner].local_cells
                                                    : g.segments[first + owner].global_cells))
          return false;
        write = node;
      }
      if (write == nullptr) return false;
      out.writes[owner * 2 + which] = write;
    }
  }
  return true;
}
// FILL reads no source: its template supplies only the cloned shape. Keep
// undeclared graph roots forbidden rather than allocating an unused leaf.
ggml_tensor* Filled(ggml_context* c, std::array<std::int64_t, 4> dimensions, float value) {
  auto* filled = ggml_fill(c, ggml_new_tensor(c, GGML_TYPE_F16, 4, dimensions.data()), value);
  filled->src[0] = nullptr;
  return filled;
}
ggml_tensor* Join(ggml_context* c, const std::array<ggml_tensor*, 4>& input, std::size_t count) {
  if (count == 1) return input[0];
  // One-row views of one contiguous [d, heads, rows] parent at consecutive
  // rows are already the joined layout: view it, without copying.
  const auto* first = input[0];
  auto* parent = first->src[0];
  bool adjacent = first->op == GGML_OP_VIEW && parent != nullptr && ggml_is_contiguous(parent) &&
                  first->ne[2] == 1 && first->ne[3] == 1 && parent->ne[3] == 1 &&
                  first->nb[1] == parent->nb[1] && first->nb[2] == parent->nb[2] &&
                  first->ne[0] == parent->ne[0] && first->ne[1] == parent->ne[1] &&
                  first->view_offs >= parent->view_offs;
  for (std::size_t i = 1; adjacent && i < count; ++i) {
    const auto* t = input[i];
    adjacent = t->op == GGML_OP_VIEW && t->src[0] == parent && t->view_src == first->view_src &&
               t->type == first->type && std::equal(t->ne, t->ne + 4, first->ne) &&
               std::equal(t->nb, t->nb + 4, first->nb) &&
               t->view_offs == first->view_offs + i * parent->nb[2];
  }
  if (adjacent) {
    const auto offset = first->view_offs - parent->view_offs;
    const auto rows = static_cast<std::int64_t>(count);
    if (offset % parent->nb[2] == 0 &&
        static_cast<std::int64_t>(offset / parent->nb[2]) + rows <= parent->ne[2])
      return ggml_view_4d(c, parent, first->ne[0], first->ne[1], 1, rows, parent->nb[1],
                          parent->nb[2], parent->nb[2], offset);
  }
  if (count == 2) return ggml_concat(c, input[0], input[1], 3);
  if (count == 3) return ggml_concat(c, ggml_concat(c, input[0], input[1], 3), input[2], 3);
  return ggml_concat(c, ggml_concat(c, input[0], input[1], 3),
                     ggml_concat(c, input[2], input[3], 3), 3);
}
}  // namespace

std::expected<void, KernelFailure> TransformGemma4Attention(TensorArena& arena, Gemma4Graph& g,
                                                            Gemma4AttentionMode mode) {
  if (mode == Gemma4AttentionMode::kIndependent) return {};
  if (mode != Gemma4AttentionMode::kPacked && mode != Gemma4AttentionMode::kOwners)
    return Reject("invalid Gemma4 attention mode");
  if (!Shape(g, mode)) return {};
  if (g.attention_mode != Gemma4AttentionMode::kIndependent || g.attention_quad_mask != 0)
    return Reject("Gemma4 attention graph was already transformed");
  const auto& p = g.profile;
  const auto count = g.segments.size() < 4 ? g.segments.size() : std::size_t{4};
  // A partial whole-cohort grid requires every actual readable view to have
  // the same padded width. Masks alone cannot guard upstream final preloads.
  const bool whole_partial =
      mode == Gemma4AttentionMode::kOwners &&
      detail::PartialOwnerCohort(static_cast<std::uint32_t>(g.segments.size())) &&
      QuadShape(g, 0, g.segments.size());
  const auto groups = whole_partial ? (g.segments.size() + 3) / 4 : g.segments.size() / count;
  std::array<std::size_t, 3> counts{};
  for (std::size_t quad = 0; quad < groups; ++quad)
    counts[quad] = whole_partial ? std::min(std::size_t{4}, g.segments.size() - quad * 4) : count;
  std::array<std::array<Layer, 60>, 3> layers{};
  std::array<bool, 3> eligible{};
  std::size_t quad_count = 0;
  if (g.nodes.size() > 50000 || g.named.size() > 10000) return Reject("packed graph domain");
  // Validate every eligible quad before mutation. Unsupported widths leave
  // their quad independent; malformed original writer edges refuse the wave.
  for (std::size_t quad = 0; quad < groups; ++quad) {
    eligible[quad] = QuadShape(g, quad * count, counts[quad]);
    if (!eligible[quad]) continue;
    ++quad_count;
    for (std::uint32_t il = 0; il < p.layers; ++il)
      if (!Inspect(g, il, quad * count, counts[quad], layers[quad][il]))
        return Reject("packed original attention/writer contract");
  }
  if (quad_count == 0) return {};
  // Only a complete eight/twelve-owner wave with every quad eligible and
  // identical padded widths shares its whole-cohort stream-K partition.
  // Tails, missing quads and unequal widths keep four-owner geometry.
  std::uint32_t logical_cohort =
      static_cast<std::uint32_t>(whole_partial ? g.segments.size() : count);
  if ((g.segments.size() == 8 || g.segments.size() == 12) && quad_count * 4 == g.segments.size()) {
    bool equal = true;
    for (std::size_t first = 4; first < g.segments.size(); first += 4)
      equal &= g.segments[0].shape.local_n_kv == g.segments[first].shape.local_n_kv &&
               g.segments[0].shape.global_n_kv == g.segments[first].shape.global_n_kv;
    if (equal) logical_cohort = static_cast<std::uint32_t>(g.segments.size());
  }
  if (auto room = arena.Reserve(p.layers * kExtraPerLayer * quad_count); !room)
    return std::unexpected(room.error());
  auto* c = arena.context();
  std::vector<ggml_tensor*> removed;
  std::array<std::array<ggml_tensor*, 60>, 3> packed{};
  // Layers of one kind share their owners' masks: join each set once.
  std::vector<std::pair<std::array<ggml_tensor*, 4>, ggml_tensor*>> joined_masks;
  for (std::size_t quad = 0; quad < eligible.size(); ++quad) {
    if (!eligible[quad]) continue;
    const auto first = quad * count;
    const auto active = counts[quad];
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      auto& l = layers[quad][il];
      const auto d = p.head_dim(il), kvh = p.kv_heads(il);
      const auto kvw = std::size_t{d} * kvh;
      std::uint32_t read = 0;
      for (std::size_t owner = 0; owner < active; ++owner)
        read = std::max(read, p.local(il) ? g.segments[first + owner].shape.local_n_kv
                                          : g.segments[first + owner].shape.global_n_kv);
      const bool bounded =
          g.options.bounded_owner_roots && mode == Gemma4AttentionMode::kOwners && active == 2 &&
          logical_cohort == 2 &&
          (p.local(il)
               ? g.segments[first].shape.local_n_kv != g.segments[first + 1].shape.local_n_kv
               : g.segments[first].shape.global_n_kv != g.segments[first + 1].shape.global_n_kv);
      for (std::size_t owner = 0; owner < active; ++owner) {
        const auto actual = p.local(il) ? g.segments[first + owner].shape.local_n_kv
                                        : g.segments[first + owner].shape.global_n_kv;
        // The write descriptor follows the SAME owned cache leaf. Reading it,
        // instead of its leaf, adds an explicit dependency on every current
        // write.
        l.k[owner] = ggml_view_4d(c, l.writes[owner * 2], d, kvh, actual, 1, d * 2, kvw * 2,
                                  kvw * actual * 2, 0);
        l.v[owner] = ggml_view_4d(c, l.writes[owner * 2 + 1], d, kvh, actual, 1, d * 2, kvw * 2,
                                  kvw * actual * 2, 0);
        if (actual != read && !bounded) {
          auto* zeros = Filled(c, {d, kvh, read - actual, 1}, 0.0F);
          l.k[owner] = ggml_concat(c, l.k[owner], zeros, 2);
          l.v[owner] = ggml_concat(c, l.v[owner], zeros, 2);
        }
      }
      auto* q = ggml_permute(c, Join(c, l.q, active), 0, 2, 1, 3);
      ggml_tensor* mask = nullptr;
      for (const auto& [masks, joined] : joined_masks)
        if (masks == l.mask) mask = joined;
      if (mask == nullptr) {
        auto masks = l.mask;
        for (std::size_t owner = 0; owner < active; ++owner)
          if (masks[owner]->ne[0] != read) {
            auto* tail = Filled(c, {read - masks[owner]->ne[0], 32, 1, 1},
                                -std::numeric_limits<float>::infinity());
            masks[owner] = ggml_concat(c, masks[owner], tail, 0);
          }
        mask = Join(c, masks, active);
        joined_masks.emplace_back(l.mask, mask);
      }
      ggml_tensor* flash = nullptr;
      if (mode == Gemma4AttentionMode::kOwners) {
        for (std::size_t owner = 0; owner < active; ++owner) {
          l.k[owner] = ggml_permute(c, l.k[owner], 0, 2, 1, 3);
          l.v[owner] = ggml_permute(c, l.v[owner], 0, 2, 1, 3);
        }
        flash = FlashAttnOwnersNode(
            c, q, mask, l.k, l.v, logical_cohort, static_cast<std::uint32_t>(active),
            whole_partial ? static_cast<std::uint32_t>(first) : 0, 0, bounded);
      } else {
        auto* k = ggml_permute(c, Join(c, l.k, active), 0, 2, 1, 3);
        auto* v = ggml_permute(c, Join(c, l.v, active), 0, 2, 1, 3);
        flash = ggml_flash_attn_ext(c, q, k, v, mask, 1.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(flash, GGML_PREC_F32);
      }
      packed[quad][il] = flash;
      ggml_set_name(flash, (quad == 0 ? std::format("packed.blk.{}.attention", il)
                                      : std::format("packed.quad.{}.blk.{}.attention", quad, il))
                               .c_str());
      for (std::size_t owner = 0; owner < active; ++owner) {
        auto* view = ggml_view_4d(c, flash, d, p.heads, 1, 1, flash->nb[1], flash->nb[2],
                                  flash->nb[3], owner * flash->nb[3]);
        auto* old = l.flash[owner];
        for (auto* node : g.nodes) {
          for (auto*& src : node->src)
            if (src == old) src = view;
          if (node->view_src == old) node->view_src = view;
        }
        for (auto& [name, tensor] : g.named)
          if (tensor == old) tensor = view;
        removed.push_back(old);
      }
    }
  }
  // One owner node covering every segment already holds the projection's
  // joined input in its layout: read it, instead of concatenating its views.
  if (quad_count == 1 && eligible[0] && counts[0] == g.segments.size()) {
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      auto* projection = g.Named(std::format("blk.{}.attn_projection", il));
      if (projection == nullptr || projection->op != GGML_OP_MUL_MAT ||
          projection->src[1] == nullptr || packed[0][il] == nullptr)
        continue;
      std::vector<ggml_tensor*> chain;
      std::vector<ggml_tensor*> leaves;
      bool ordered = true;
      const auto walk = [&](auto&& self, ggml_tensor* t) -> void {
        if (!ordered) return;
        if (t->op == GGML_OP_CONCAT) {
          if (t->op_params[0] != 1) {
            ordered = false;
            return;
          }
          chain.push_back(t);
          self(self, t->src[0]);
          self(self, t->src[1]);
          return;
        }
        leaves.push_back(t);
      };
      walk(walk, projection->src[1]);
      if (!ordered || leaves.size() != g.segments.size()) continue;
      for (std::size_t owner = 0; ordered && owner < leaves.size(); ++owner)
        ordered = leaves[owner] == g.Named(std::format("blk.{}.slot.{}.attention", il,
                                                       g.segments[owner].shape.slot));
      const auto width = std::int64_t{p.head_dim(il)} * p.heads;
      if (!ordered || projection->src[1]->ne[0] != width ||
          projection->src[1]->ne[1] != static_cast<std::int64_t>(leaves.size()) ||
          !ggml_is_contiguous(packed[0][il]) ||
          ggml_nelements(packed[0][il]) != width * static_cast<std::int64_t>(leaves.size()))
        continue;
      // Only the projection reads the chain's root and each link.
      bool private_chain = true;
      for (auto* node : g.nodes)
        for (auto* src : node->src)
          if (src != nullptr && std::ranges::find(chain, src) != chain.end() &&
              node != projection && std::ranges::find(chain, node) == chain.end())
            private_chain = false;
      if (!private_chain) continue;
      projection->src[1] =
          ggml_reshape_2d(c, packed[0][il], width, static_cast<std::int64_t>(leaves.size()));
      removed.insert(removed.end(), chain.begin(), chain.end());
    }
  }
  std::vector<ggml_tensor*> roots;
  roots.reserve(g.nodes.size());
  for (auto* node : g.nodes)
    if (std::ranges::find(removed, node) == removed.end()) roots.push_back(node);
  auto ordered = GraphOrder(roots, arena);
  if (!ordered) return std::unexpected(ordered.error());
  g.nodes = std::move(*ordered);
  std::size_t attention = 0;
  for (auto* node : g.nodes) {
    if (std::ranges::find(removed, node) != removed.end())
      return Reject("old attention executable");
    if (node->op == GGML_OP_FLASH_ATTN_EXT || LlmpOpOf(node) == LlmpOp::kFlashAttnOwners)
      ++attention;
  }
  std::size_t removed_per_layer = 0;
  for (std::size_t quad = 0; quad < groups; ++quad)
    if (eligible[quad]) removed_per_layer += counts[quad] - 1;
  if (attention != p.layers * (g.segments.size() - removed_per_layer))
    return Reject("packed attention count");
  for (std::size_t quad = 0; quad < eligible.size(); ++quad) {
    if (!eligible[quad]) continue;
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      const auto at = std::ranges::find(g.nodes, packed[quad][il]);
      if (mode == Gemma4AttentionMode::kOwners) {
        if (LlmpOpOf(packed[quad][il]) != LlmpOp::kFlashAttnOwners)
          return Reject("missing owner attention operation");
        for (std::size_t owner = 0; owner < counts[quad]; ++owner) {
          for (std::size_t which = 0; which < 2; ++which) {
            const auto* cache = packed[quad][il]->src[2 + owner + which * 4];
            if (!cache || cache->op != GGML_OP_PERMUTE || !cache->src[0])
              return Reject("owner input lost cache permutation");
            const auto* raw = cache->src[0];
            if (raw->op == GGML_OP_CONCAT) {
              const auto* tail = raw->src[1];
              if (!g.options.common_owner_reads || g.segments.size() != 2 ||
                  raw->op_params[0] != 2 || !tail || tail->op != GGML_OP_FILL ||
                  tail->src[0] != nullptr || tail->type != GGML_TYPE_F16 ||
                  tail->op_params[0] != 0 || raw->src[0] == nullptr)
                return Reject("owner padding lost bounded zero tail");
              raw = raw->src[0];
            }
            if (raw->op != GGML_OP_VIEW ||
                raw->src[0] != layers[quad][il].writes[2 * owner + which] ||
                raw->view_src != layers[quad][il].writes[2 * owner + which]->view_src)
              return Reject("owner input lost exact cache writer dependency");
          }
        }
      }
      if (at == g.nodes.end() || packed[quad][il]->ne[3] != static_cast<std::int64_t>(counts[quad]))
        return Reject("packed output owner order");
      for (auto* writer : layers[quad][il].writes)
        if (writer && std::ranges::find(g.nodes, writer) >= at)
          return Reject("packed cache write order");
    }
  }
  for (std::size_t quad = 0; quad < eligible.size(); ++quad)
    if (eligible[quad]) g.attention_quad_mask |= std::uint32_t{1} << quad;
  g.attention_mode = mode;
  return {};
}
}  // namespace llmp::kernels::ggml
