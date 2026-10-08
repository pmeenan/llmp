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
#include "kernels/ggml/gemma4_graph.h"

namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
int main() {
  const bool packed = std::getenv("LLMP_GEMMA_PACKED_C4") != nullptr &&
                      std::string_view(std::getenv("LLMP_GEMMA_PACKED_C4")) == "1";
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
        if (arena->bytes() != arena->used() + ggml_tensor_overhead()) return 1;
        const bool selected = packed && variant == 31 && owners == 4 && rows == 1;
        const auto expected = p.layers * (selected ? 1 : owners);
        const auto actual = std::ranges::count_if(
            graph->nodes, [](auto* n) { return n->op == GGML_OP_FLASH_ATTN_EXT; });
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
                  view->src[0]->op != GGML_OP_FLASH_ATTN_EXT || view->src[0]->ne[3] != 4 ||
                  view->view_offs != i * view->src[0]->nb[3] || flat->view_src != view)
                return 1;
              auto* flash = view->src[0];
              for (const auto operand : {1U, 2U}) {
                auto* cache = flash->src[operand];
                if (cache->ne[0] != p.head_dim(il) || cache->ne[1] != 256 ||
                    cache->ne[2] != p.kv_heads(il) || cache->ne[3] != 4 || cache->nb[0] != 2 ||
                    cache->nb[1] != p.head_dim(il) * p.kv_heads(il) * 2 ||
                    cache->nb[2] != p.head_dim(il) * 2)
                  return 1;
              }
            }
          }
        }
        std::cout << "PACKED_METADATA variant=" << variant << " owners=" << owners
                  << " rows=" << rows << " selected=" << selected << " attention=" << actual
                  << " used=" << arena->used() << " PASS\n";
      }
  }
  return 0;
}
