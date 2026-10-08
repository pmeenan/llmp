// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Fixed-process manual comparison using the shared checked C4 graph transform.
#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "kernels/ggml/fattn_owner.h"
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
bool Transform() {
  static const bool enabled = [] {
    const char* flag = std::getenv("LLMP_GEMMA_OWNER_C4");
    if (flag == nullptr || std::string_view(flag) == "0") return false;
    if (std::string_view(flag) == "packed" || std::string_view(flag) == "owners") return true;
    std::cerr << "invalid LLMP_GEMMA_OWNER_C4 (expected 0, packed or owners)\n";
    std::abort();
  }();
  return enabled;
}
bool Owners() {
  static const bool owners = [] {
    const char* flag = std::getenv("LLMP_GEMMA_OWNER_C4");
    return flag && std::string_view(flag) == "owners";
  }();
  return owners;
}
}  // namespace

std::size_t WrappedTensors(const md::Gemma4Profile&, std::size_t) asm(
    "__wrap__ZN4llmp7kernels4ggml18Gemma4GraphTensorsERKNS_5model13Gemma4ProfileEm");
std::size_t WrappedTensors(const md::Gemma4Profile& p, std::size_t segments) {
  const auto original = RealTensors(p, segments);
  return original +
         (Transform() && (p == md::Gemma4_31B() || p == md::Gemma4_26BA4B()) && segments == 4
              ? p.layers * kExtraPerLayer
              : 0);
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
  if (!built || !Transform()) return built;
  auto& g = *built;
  if (g.attention_mode != kg::Gemma4AttentionMode::kIndependent) {
    std::cout << "OWNER_C4_NATIVE_GRAPH attention=" << p.layers << " heads=" << p.heads
              << " local_read=" << shape.segments[0].local_n_kv
              << " global_read=" << shape.segments[0].global_n_kv
              << " local_capacity=" << state.local_cells
              << " global_capacity=" << state.global_cells
              << " native_optin=" << (options.attention_mode == kg::Gemma4AttentionMode::kOwners)
              << '\n';
    return built;
  }
  auto transformed = kg::TransformGemma4Attention(
      arena, g, Owners() ? kg::Gemma4AttentionMode::kOwners : kg::Gemma4AttentionMode::kPacked);
  if (!transformed) return std::unexpected(transformed.error());
  if (g.attention_mode == kg::Gemma4AttentionMode::kIndependent) {
    static std::uint64_t fallbacks = 0;
    if (++fallbacks <= 8)
      std::cout << "PACKED_FALLBACK count=" << fallbacks << " segments=" << shape.segments.size()
                << " outputs=" << shape.outputs << " context=" << state.context
                << " original_builder=1\n";
  } else {
    std::cout << "OWNER_C4_GRAPH attention=" << p.layers << " removed=" << p.layers * 4
              << " funded_extra=" << p.layers * kExtraPerLayer << " arena_used=" << arena.used()
              << " nodes=" << g.nodes.size() << " heads=" << p.heads
              << " scope=" << (p == md::Gemma4_31B() ? "31" : "26")
              << "-C4-rows1 local_read=" << shape.segments[0].local_n_kv
              << " global_read=" << shape.segments[0].global_n_kv << " owners=" << Owners() << "\n";
  }
  return built;
}

// Observe the existing external scratch planner, never alter its decisions.
// The kernel's same-TU internal calls need not interpose: planning is enough.
using OwnerPlan = std::expected<kg::FlashAttnOwnersPlan, kg::KernelFailure>;
OwnerPlan RealOwnerPlan(const kg::LaunchContext&, const kg::FlashAttnOwners&) asm(
    "__real__ZN4llmp7kernels4ggml19PlanFlashAttnOwnersERKNS1_13LaunchContextERKNS1_"
    "15FlashAttnOwnersE");
OwnerPlan WrappedOwnerPlan(const kg::LaunchContext&, const kg::FlashAttnOwners&) asm(
    "__wrap__ZN4llmp7kernels4ggml19PlanFlashAttnOwnersERKNS1_13LaunchContextERKNS1_"
    "15FlashAttnOwnersE");
OwnerPlan WrappedOwnerPlan(const kg::LaunchContext& launch, const kg::FlashAttnOwners& inputs) {
  auto plan = RealOwnerPlan(launch, inputs);
  if (plan) {
    static std::array<std::atomic_bool, 256> seen{};
    const auto index = (inputs.q->ne[2] == 16 ? 0U : 128U) +
                       (plan->original.head == 256 ? 0U : 64U) +
                       static_cast<unsigned>(inputs.mask->ne[0] / 256 - 1);
    if (!seen[index].exchange(true, std::memory_order_relaxed))
      std::cout << "OWNER_C4_GEOMETRY heads=" << inputs.q->ne[2] << " head=" << plan->original.head
                << " cells=" << inputs.mask->ne[0] << " kvheads=" << inputs.k[0]->ne[2]
                << " original_blocks=" << plan->original.blocks
                << " original_blocks_per_sm=" << plan->original_blocks_per_sm
                << " owner_blocks_per_sm=" << plan->owner_blocks_per_sm
                << " columns=" << plan->original.columns << " group=" << plan->original.group
                << " scratch=" << plan->original.scratch << " threads=" << plan->threads
                << " shared_bytes=" << plan->shared_bytes << " basis=compiled-original-planner\n";
  }
  return plan;
}
