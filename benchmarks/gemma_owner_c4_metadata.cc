// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Manual descriptor-only controls; no model payloads or GPU launches.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "engine/planned.h"
#include "gemma4_fixture.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/gemma4_graph.h"
#include "kernels/ggml/llmp_ops.h"

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
// Addresses are descriptor-only sentinels; this checker never reads payloads.
bool OwnerNodeControls() {
  for (const std::int64_t heads : {16, 32})
    for (const std::int64_t d : {256, 512}) {
      auto arena = kg::TensorArena::Create(32);
      if (!arena) return false;
      auto* c = arena->context();
      const auto kvh = heads / (d == 256 ? 2 : 8);
      auto* q = ggml_new_tensor_4d(c, GGML_TYPE_F32, d, 1, heads, 4);
      q->nb[1] = static_cast<std::size_t>(d * heads) * 4;
      q->data = reinterpret_cast<void*>(0x10000000ULL);
      auto* mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, 256, 32, 1, 4);
      mask->data = reinterpret_cast<void*>(0x20000000ULL);
      std::array<ggml_tensor*, 4> k{}, v{};
      for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t which = 0; which < 2; ++which) {
          auto* raw = ggml_new_tensor_4d(c, GGML_TYPE_F16, d, kvh, 256, 1);
          raw->data =
              reinterpret_cast<void*>(0x30000000ULL + which * 0x10000000ULL + i * 0x1000000ULL);
          auto* view = ggml_permute(c, raw, 0, 2, 1, 3);
          if (which == 0)
            k[i] = view;
          else
            v[i] = view;
        }
      }
      auto* out = kg::FlashAttnOwnersNode(c, q, mask, k, v);
      out->data = reinterpret_cast<void*>(0x60000000ULL);
      const auto valid = [&] { return kg::CheckFlashAttnOwnersNode(out).has_value(); };
      if (!valid()) return false;
      auto* last = out->src[9];
      out->src[9] = nullptr;
      if (valid()) return false;
      out->src[9] = last;
      out->op_params[8] = 1;
      if (valid()) return false;
      out->op_params[8] = 0;
      out->ne[3] = 3;
      if (valid()) return false;
      out->ne[3] = 4;
      out->data = q->data;
      if (valid()) return false;
      out->data = reinterpret_cast<void*>(0x60000000ULL);
      out->src[3] = out->src[2];
      if (valid()) return false;
      out->src[3] = k[1];
      out->op = GGML_OP_FLASH_ATTN_EXT;
      if (valid()) return false;
      out->op = GGML_OP_CUSTOM;
      if (!valid()) return false;
    }
  std::cout << "OWNER_NODE_METADATA two_dimensions two_head_counts tag/all10/params/shape/alias "
               "refusals PASS "
               "launches=0\n";
  return true;
}
int main() {
  if (!OwnerNodeControls()) return 1;
  const char* mode = std::getenv("LLMP_GEMMA_OWNER_C4");
  const bool packed =
      mode && (std::string_view(mode) == "packed" || std::string_view(mode) == "owners");
  const bool owner_roots = mode && std::string_view(mode) == "owners";
  for (const auto variant : {26U, 31U}) {
    const auto& p = variant == 31 ? md::Gemma4_31B() : md::Gemma4_26BA4B();
    auto binding = md::BindGemma4(p, "gemma4", llmp::test_support::gemma4::Resources(variant));
    if (!binding) return 1;
    for (const auto owners : {1U, 2U, 4U})
      for (const auto rows : {1U, 2U}) {
        auto state = md::Gemma4State(p, 256, 128);
        if (!state) return 1;
        kg::Gemma4ChunkShape shape;
        for (std::uint32_t i = 0; i < owners; ++i)
          shape.segments.push_back({i, rows, 64 + i, 256, 256});
        shape.outputs = owners;
        kg::Gemma4GraphOptions options;
        options.device_masks = true;
        options.narrow_final = true;
        const auto estimate = kg::Gemma4GraphTensors(p, owners);
        auto arena = llmp::engine::SizedArena(estimate, [&](kg::TensorArena& a) {
          return kg::BuildGemma4Graph(a, p, *binding, *state, shape, options).has_value();
        });
        if (!arena) {
          std::cerr << arena.error() << '\n';
          return 1;
        }
        auto graph = kg::BuildGemma4Graph(*arena, p, *binding, *state, shape, options);
        if (!graph) {
          std::cerr << graph.error().detail << '\n';
          return 1;
        }
        arena->Seal();
        if (arena->bytes() !=
            arena->used() + ggml_tensor_overhead() + arena->graph_visited().size_bytes())
          return 1;
        const bool selected = packed && owners == 4 && rows == 1;
        const auto expected = p.layers * (selected ? 1 : owners);
        const auto actual = std::ranges::count_if(graph->nodes, [](auto* n) {
          return n->op == GGML_OP_FLASH_ATTN_EXT || kg::LlmpOpOf(n) == kg::LlmpOp::kFlashAttnOwners;
        });
        if (static_cast<std::uint32_t>(actual) != expected) return 1;
        for (std::uint32_t il = 0; il < p.layers; ++il) {
          for (std::uint32_t i = 0; i < owners; ++i) {
            auto* flat = graph->Named("blk." + std::to_string(il) + ".slot." + std::to_string(i) +
                                      ".attention");
            if (flat == nullptr || flat->ne[0] != p.head_dim(il) * p.heads || flat->ne[1] != rows)
              return 1;
            if (selected) {
              auto* view = flat->src[0];
              if (view == nullptr || view->op != GGML_OP_VIEW || view->src[0] == nullptr ||
                  (owner_roots ? kg::LlmpOpOf(view->src[0]) != kg::LlmpOp::kFlashAttnOwners
                               : view->src[0]->op != GGML_OP_FLASH_ATTN_EXT) ||
                  view->src[0]->ne[3] != 4 || view->view_offs != i * view->src[0]->nb[3] ||
                  flat->view_src != view)
                return 1;
              auto* flash = view->src[0];
              for (const auto operand : {1U, 2U}) {
                auto* cache = flash->src[owner_roots ? (operand == 1 ? 2 + i : 6 + i) : operand];
                if (cache->ne[0] != p.head_dim(il) || cache->ne[1] != 256 ||
                    cache->ne[2] != p.kv_heads(il) || cache->ne[3] != (owner_roots ? 1 : 4) ||
                    cache->nb[0] != 2 || cache->nb[1] != p.head_dim(il) * p.kv_heads(il) * 2 ||
                    cache->nb[2] != p.head_dim(il) * 2)
                  return 1;
              }
            }
          }
        }
        std::cout << "OWNER_C4_METADATA variant=" << variant << " owners=" << owners
                  << " rows=" << rows << " selected=" << selected << " attention=" << actual
                  << " used=" << arena->used() << " PASS\n";
      }
  }
  return 0;
}
