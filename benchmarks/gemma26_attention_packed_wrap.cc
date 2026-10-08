// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Benchmark-only graph derivative. No production selector or builder changes.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string_view>
#include <vector>

#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma4_graph.h"

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
using Result = std::expected<kg::Gemma4Graph, kg::KernelFailure>;
Result
RealBuild(kg::TensorArena&, const md::Gemma4Profile&, const md::Gemma4Binding&, const md::Gemma4StateLayout&, const kg::Gemma4ChunkShape&, const kg::Gemma4GraphOptions&) asm(
    "__real__ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_"
    "5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_"
    "16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE");
std::size_t RealTensors(const md::Gemma4Profile&, std::size_t) asm(
    "__real__ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm");
namespace {
constexpr std::size_t kExtraPerLayer = 64;
// Fixed once for the whole process: never toggle a policy behind PlanCache.
bool Candidate() {
  static const bool enabled = [] {
    const char* flag = std::getenv("LLMP_GEMMA26_PACKED_C4");
    if (flag == nullptr || std::string_view(flag) == "0") return false;
    if (std::string_view(flag) == "1") return true;
    std::cerr << "invalid LLMP_GEMMA26_PACKED_C4 (expected 0 or 1)\n";
    std::abort();
  }();
  return enabled;
}
Result Reject(std::string_view why) {
  return std::unexpected(kg::KernelFailure{.detail = std::string(why)});
}
bool Shape(const kg::Gemma4Graph& g) {
  if (g.profile != md::Gemma4_26BA4B() || g.context != 256 || g.max_rows != 128 ||
      g.shape.outputs != 4 || g.shape.feature_outputs != 0 || g.segments.size() != 4 ||
      g.shape.segments.size() != 4 || g.options.first_layer != 0 || g.options.layer_count != 0 ||
      g.options.hidden_input || !g.options.head || g.options.shared_q8 || g.options.rope_store ||
      g.normalized_features != nullptr)
    return false;
  for (std::size_t i = 0; i < 4; ++i) {
    const auto& s = g.segments[i];
    if (s.shape != g.shape.segments[i] || s.shape.rows != 1 || s.shape.slot != i ||
        s.first_row != i || s.shape.local_n_kv != 256 || s.shape.global_n_kv != 256 ||
        s.caches.size() != g.profile.layers)
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
bool Inspect(const kg::Gemma4Graph& g, std::uint32_t layer, Layer& out) {
  const auto d = g.profile.head_dim(layer), kvh = g.profile.kv_heads(layer);
  const auto kvw = std::size_t{d} * kvh;
  for (std::size_t owner = 0; owner < 4; ++owner) {
    auto* flat = g.Named(std::format("blk.{}.slot.{}.attention", layer, owner));
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
    out.mask[owner] =
        g.profile.local(layer) ? g.segments[owner].local_mask : g.segments[owner].global_mask;
    if (f->src[3] != out.mask[owner] || !Dims(out.mask[owner], GGML_TYPE_F16, {256, 32, 1, 1}) ||
        !ggml_is_contiguous(out.mask[owner]))
      return false;
    for (std::size_t which = 0; which < 2; ++which) {
      auto* leaf = which == 0 ? g.segments[owner].caches[layer].first
                              : g.segments[owner].caches[layer].second;
      auto* cache = f->src[which + 1];
      if (!Dims(leaf, GGML_TYPE_F16, {static_cast<std::int64_t>(kvw), 256, 1, 1}) ||
          leaf->op != GGML_OP_NONE || leaf->view_src != nullptr || !ggml_is_contiguous(leaf) ||
          !Dims(cache, GGML_TYPE_F16, {d, 256, kvh, 1}) || cache->op != GGML_OP_PERMUTE ||
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
            node->src[1] != (g.profile.local(layer) ? g.segments[owner].local_cells
                                                    : g.segments[owner].global_cells))
          return false;
        write = node;
      }
      if (write == nullptr) return false;
      out.writes[owner * 2 + which] = write;
    }
  }
  return true;
}
ggml_tensor* Join(ggml_context* c, const std::array<ggml_tensor*, 4>& input) {
  return ggml_concat(c, ggml_concat(c, input[0], input[1], 3),
                     ggml_concat(c, input[2], input[3], 3), 3);
}
}  // namespace

std::size_t WrappedTensors(const md::Gemma4Profile&, std::size_t) asm(
    "__wrap__ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm");
std::size_t WrappedTensors(const md::Gemma4Profile& p, std::size_t segments) {
  const auto original = RealTensors(p, segments);
  return original +
         (Candidate() && p == md::Gemma4_26BA4B() && segments == 4 ? p.layers * kExtraPerLayer : 0);
}
Result
WrappedBuild(kg::TensorArena&, const md::Gemma4Profile&, const md::Gemma4Binding&, const md::Gemma4StateLayout&, const kg::Gemma4ChunkShape&, const kg::Gemma4GraphOptions&) asm(
    "__wrap__ZN4llmp7kernels4ggml16BuildGemma4GraphERNS1_11TensorArenaERKNS_"
    "5model13Gemma4ProfileERKNS4_13Gemma4BindingERKNS4_17Gemma4StateLayoutERKNS1_"
    "16Gemma4ChunkShapeERKNS1_18Gemma4GraphOptionsE");
Result WrappedBuild(kg::TensorArena& arena, const md::Gemma4Profile& p,
                    const md::Gemma4Binding& binding, const md::Gemma4StateLayout& state,
                    const kg::Gemma4ChunkShape& shape, const kg::Gemma4GraphOptions& options) {
  auto built = RealBuild(arena, p, binding, state, shape, options);
  if (!built || !Candidate()) return built;
  if (!Shape(*built)) {
    static std::uint64_t fallbacks = 0;
    if (++fallbacks <= 8)
      std::cout << "PACKED_FALLBACK count=" << fallbacks << " segments=" << shape.segments.size()
                << " outputs=" << shape.outputs << " context=" << state.context
                << " original_builder=1\n";
    return built;
  }
  auto& g = *built;
  std::array<Layer, 30> layers{};
  if (g.nodes.size() > 50000 || g.named.size() > 10000) return Reject("packed graph domain");
  for (std::uint32_t il = 0; il < p.layers; ++il)
    if (!Inspect(g, il, layers[il])) return Reject("packed original attention/writer contract");
  if (auto room = arena.Reserve(p.layers * kExtraPerLayer); !room)
    return std::unexpected(room.error());
  auto* c = arena.context();
  std::vector<ggml_tensor*> removed;
  std::array<ggml_tensor*, 30> packed{};
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    auto& l = layers[il];
    const auto d = p.head_dim(il), kvh = p.kv_heads(il);
    const auto kvw = std::size_t{d} * kvh;
    for (std::size_t owner = 0; owner < 4; ++owner) {
      // The write descriptor follows the SAME owned cache leaf. Reading it,
      // instead of its leaf, adds an explicit dependency on every current write.
      l.k[owner] =
          ggml_view_4d(c, l.writes[owner * 2], d, kvh, 256, 1, d * 2, kvw * 2, kvw * 256 * 2, 0);
      l.v[owner] = ggml_view_4d(c, l.writes[owner * 2 + 1], d, kvh, 256, 1, d * 2, kvw * 2,
                                kvw * 256 * 2, 0);
    }
    auto* q = ggml_permute(c, Join(c, l.q), 0, 2, 1, 3);
    auto* k = ggml_permute(c, Join(c, l.k), 0, 2, 1, 3);
    auto* v = ggml_permute(c, Join(c, l.v), 0, 2, 1, 3);
    auto* mask = Join(c, l.mask);
    auto* flash = ggml_flash_attn_ext(c, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(flash, GGML_PREC_F32);
    packed[il] = flash;
    ggml_set_name(flash, std::format("packed.blk.{}.attention", il).c_str());
    for (std::size_t owner = 0; owner < 4; ++owner) {
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
  std::vector<ggml_tensor*> roots;
  roots.reserve(g.nodes.size());
  for (auto* node : g.nodes)
    if (std::ranges::find(removed, node) == removed.end()) roots.push_back(node);
  g.nodes = kg::GraphOrder(roots);
  std::size_t attention = 0;
  for (auto* node : g.nodes) {
    if (std::ranges::find(removed, node) != removed.end())
      return Reject("old attention executable");
    if (node->op == GGML_OP_FLASH_ATTN_EXT) ++attention;
  }
  if (attention != 30) return Reject("packed attention count");
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const auto at = std::ranges::find(g.nodes, packed[il]);
    if (at == g.nodes.end() || packed[il]->ne[3] != 4) return Reject("packed output owner order");
    for (auto* writer : layers[il].writes)
      if (std::ranges::find(g.nodes, writer) >= at) return Reject("packed cache write order");
  }
  std::cout << "PACKED_GRAPH attention=" << attention << " removed=" << removed.size()
            << " funded_extra=" << p.layers * kExtraPerLayer << " arena_used=" << arena.used()
            << " nodes=" << g.nodes.size() << " scope=26-C4-rows1-read256\n";
  return built;
}
