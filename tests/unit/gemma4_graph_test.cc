// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma4_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/check.h"
#include "expected_error.h"
#include "gemma4_fixture.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace fixture = llmp::test_support::gemma4;
struct Case {
  const md::Gemma4Profile& p;
  md::Gemma4Binding binding;
  md::Gemma4StateLayout state;
  kg::Gemma4ChunkShape shape;
  md::Gemma4ChunkInputs input;
  explicit Case(std::uint32_t size = 26, std::uint32_t slots = 1)
      : p(size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B()) {
    auto b = md::BindGemma4(p, "gemma4", fixture::Resources(size));
    llmp::base::Check(b.has_value(), "Gemma graph fixture binding failed");
    binding = std::move(*b);
    auto s = md::Gemma4State(p, 4096, 16);
    llmp::base::Check(s.has_value(), "Gemma graph state failed");
    state = std::move(*s);
    const std::array<std::int32_t, 2> tokens{1, 2};
    std::vector<md::Gemma4Segment> segments;
    for (std::uint32_t i = 0; i < slots; ++i) segments.push_back({i, i * 1100, tokens});
    auto in = md::Gemma4Chunk(p, state, segments, true);
    llmp::base::Check(in.has_value(), "Gemma graph input failed");
    input = std::move(*in);
    for (const auto& seg : input.segments)
      shape.segments.push_back({seg.slot, seg.rows, seg.n_past, seg.global_n_kv, seg.local_n_kv});
    shape.outputs = slots;
  }
};
std::size_t Count(const kg::Gemma4Graph& g, ggml_op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(g.nodes, [op](const auto* t) { return t->op == op; }));
}
void BindLeaves(kg::Gemma4Graph& g);

TEST(Gemma4Graph, CohortGridUsesWholeEightAndTwelveRoundingAndSafeFallback) {
  for (const auto max_blocks : {1, 47, 48, 96, 100, 256, 2147483647})
    for (const auto kv_heads : {2, 4, 8, 16})
      for (const auto kv_tiles : {4, 16, 32, 512}) {
        const auto original = [&](int owners) {
          const int tiles = kv_heads * owners;
          const int raw = std::min(max_blocks, kv_tiles * tiles);
          const int rounded = raw / tiles * tiles;
          const int loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
          return loss <= 5 ? rounded : raw;
        };
        const auto four = kg::detail::PlanOwnerPartition(max_blocks, kv_tiles, kv_heads, 4);
        ASSERT_TRUE(four);
        EXPECT_EQ(four->quad_blocks, original(4));
        EXPECT_EQ(four->effective_cohort, 4U);
        for (const auto cohort : {8U, 12U}) {
          const auto split = kg::detail::PlanOwnerPartition(max_blocks, kv_tiles, kv_heads, cohort);
          ASSERT_TRUE(split);
          const auto quads = static_cast<int>(cohort / 4);
          const auto whole = original(static_cast<int>(cohort));
          if (whole % quads != 0) {
            EXPECT_EQ(split->effective_cohort, 4U);
            EXPECT_EQ(split->quad_blocks, four->quad_blocks);
          } else {
            EXPECT_EQ(split->effective_cohort, cohort);
            EXPECT_EQ(split->cohort_blocks, whole);
            EXPECT_EQ(split->quad_blocks * quads, whole);
            const auto quad_work = std::int64_t{kv_tiles} * kv_heads * 4;
            // Every quad translation matches each original whole-cohort
            // block boundary, including the final endpoint.
            for (int q = 0; q < quads; ++q)
              for (int b = 0; b <= split->quad_blocks; ++b)
                EXPECT_EQ(b * quad_work / split->quad_blocks + q * quad_work,
                          (b + q * split->quad_blocks) * (quad_work * quads) / whole);
          }
        }
      }
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(0, 16, 16, 12));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(48, 0, 16, 12));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(48, 513, 16, 12));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(48, 16, 17, 12));
  EXPECT_FALSE(kg::detail::PlanOwnerPartition(48, 16, 16, 16));
  const auto local = kg::detail::PlanOwnerPartition(48, 16, 16, 12);
  const auto global = kg::detail::PlanOwnerPartition(96, 32, 4, 12);
  ASSERT_TRUE(local);
  ASSERT_TRUE(global);
  EXPECT_EQ(local->quad_blocks, 16);
  EXPECT_EQ(global->quad_blocks, 32);
}

TEST(Gemma4Graph, OwnerOperandsUseActualVariableSpansAndWidths) {
  for (const std::int64_t heads : {16, 32})
    for (const std::int64_t d : {256, 512})
      for (const std::int64_t cells : {256, 1280, 2048, 8192, 8448, 16384}) {
        auto arena = kg::TensorArena::Create(32);
        ASSERT_TRUE(arena);
        auto* c = arena->context();
        const auto kvh = heads / (d == 256 ? 2 : 8);
        auto* q = ggml_new_tensor_4d(c, GGML_TYPE_F32, d, 1, heads, 4);
        q->nb[1] = static_cast<std::size_t>(d * heads) * 4;
        kg::TensorArena::Bind(q, 0x10000000ULL);
        auto* mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, cells, 32, 1, 4);
        kg::TensorArena::Bind(mask, 0x20000000ULL);
        std::array<ggml_tensor*, 4> k{}, v{};
        for (std::size_t i = 0; i < 4; ++i)
          for (std::size_t which = 0; which < 2; ++which) {
            auto* raw = ggml_new_tensor_4d(c, GGML_TYPE_F16, d, kvh, cells, 1);
            kg::TensorArena::Bind(
                raw, (which == 0 ? 0x100000000ULL : 0x200000000ULL) + i * 0x10000000ULL);
            (which == 0 ? k[i] : v[i]) = ggml_permute(c, raw, 0, 2, 1, 3);
          }
        auto* out = kg::FlashAttnOwnersNode(c, q, mask, k, v);
        kg::TensorArena::Bind(out, 0x300000000ULL);
        const bool fits = std::uint64_t(d * kvh * cells * 2) <= (64ULL << 20U);
        EXPECT_EQ(kg::CheckFlashAttnOwnersNode(out).has_value(), fits)
            << heads << '/' << d << '/' << cells;
        if (!fits) continue;
        EXPECT_EQ(kg::LlmpOpInt(out, 0), 4);
        // Custom integer payload begins at byte32 after the operation tag.
        for (const auto cohort : {0, 1, 13, 16, -1}) {
          out->op_params[8] = cohort;
          EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        }
        out->op_params[8] = 8;
        EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(out));
        auto eight = kg::FlashAttnOwnersFromNode(out);
        ASSERT_TRUE(eight);
        EXPECT_EQ(eight->logical_cohort, 8U);
        out->op_params[8] = 12;
        auto twelve = kg::FlashAttnOwnersFromNode(out);
        ASSERT_TRUE(twelve);
        EXPECT_EQ(twelve->logical_cohort, 12U);
        out->op_params[8] = 16;
        EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        out->op_params[8] = 12;
        out->op_params[9] = 8;
        EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        out->op_params[9] = 0;
        out->op_params[8] = 4;
        const auto old_width = k[3]->ne[1];
        k[3]->ne[1] -= 256;
        EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        k[3]->ne[1] = old_width;
        mask->nb[3] += 16;
        EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        mask->nb[3] -= 16;
        auto* last = out->src[9];
        out->src[9] = nullptr;
        EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        out->src[9] = last;
        out->src[3] = out->src[2];
        EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
        out->src[3] = k[1];
        EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(out));
      }
}

TEST(Gemma4Graph, OwnerReadViewsKeepBoundsWithFullContextBackingParents) {
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  constexpr std::int64_t d = 512, heads = 32, kvh = 4, cells = 1024, capacity = 262144;
  auto* q = ggml_new_tensor_4d(c, GGML_TYPE_F32, d, 1, heads, 4);
  q->nb[1] = d * heads * 4;
  kg::TensorArena::Bind(q, 0x10000000ULL);
  auto* mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, cells, 32, 1, 4);
  kg::TensorArena::Bind(mask, 0x20000000ULL);
  std::array<ggml_tensor*, 4> k{}, v{};
  ggml_tensor* first_parent = nullptr;
  for (std::size_t i = 0; i < 4; ++i)
    for (std::size_t which = 0; which < 2; ++which) {
      auto* raw = ggml_new_tensor_4d(c, GGML_TYPE_F16, d, kvh, capacity, 1);
      kg::TensorArena::Bind(raw, 0x100000000ULL + (i * 2 + which) * 0x80000000ULL);
      if (i == 0 && which == 0) first_parent = raw;
      auto* view =
          ggml_view_4d(c, raw, d, kvh, cells, 1, d * 2, d * kvh * 2, d * kvh * cells * 2, 0);
      (which == 0 ? k[i] : v[i]) = ggml_permute(c, view, 0, 2, 1, 3);
    }
  auto* out = kg::FlashAttnOwnersNode(c, q, mask, k, v, 8);
  kg::TensorArena::Bind(out, 0x2000000000ULL);
  ASSERT_TRUE(kg::CheckFlashAttnOwnersNode(out));
  // Storage growth does not widen the read operand or admit overflowed,
  // inconsistent or cyclic parent chains.
  first_parent->ne[2] += 256;
  EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
  first_parent->ne[2] = capacity;
  const auto address = first_parent->data;
  first_parent->data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 15);
  EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
  first_parent->data = address;
  auto* leaf = k[0];
  ASSERT_NE(leaf, nullptr);
  leaf->view_offs += 16;
  EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
  leaf->view_offs -= 16;
  first_parent->view_src = first_parent;
  EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(out));
  first_parent->view_src = nullptr;
  EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(out));
}

TEST(Gemma4Graph, FullConfiguredContextKeepsBoundedOwnerReadsAndDepthFallback) {
  for (const auto size : {26U, 31U})
    for (const auto owners : {4U, 8U})
      for (const auto read : {1024U, 16384U, 16640U}) {
        Case c(size, 4);
        c.state = *md::Gemma4State(c.p, 262144, 256);
        c.shape = {};
        for (std::uint32_t i = 0; i < owners; ++i)
          c.shape.segments.push_back({i, 1, read - 1, read, c.state.local_cells});
        c.shape.outputs = owners;
        kg::Gemma4GraphOptions options;
        options.device_masks = true;
        options.attention_mode = kg::Gemma4AttentionMode::kOwners;
        auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, owners, options));
        ASSERT_TRUE(arena);
        auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
        ASSERT_TRUE(g);
        EXPECT_EQ(g->attention_mode, read <= 16384 ? kg::Gemma4AttentionMode::kOwners
                                                   : kg::Gemma4AttentionMode::kIndependent);
        if (read <= 16384) {
          EXPECT_EQ(g->attention_quad_mask, owners == 8 ? 3U : 1U);
          BindLeaves(*g);
          for (const auto* node : g->nodes)
            if (kg::LlmpOpOf(node) == kg::LlmpOp::kFlashAttnOwners) {
              const auto valid = kg::CheckFlashAttnOwnersNode(node);
              EXPECT_TRUE(valid) << (valid ? "" : valid.error().detail);
            }
        }
      }
}

TEST(Gemma4Graph, OwnerOptInPreservesUnsupportedGraphsAndRejectsBrokenWriterEdges) {
  for (const auto size : {26U, 31U})
    for (const auto count : {1U, 2U, 4U, 8U})
      for (const auto rows : {1U, 2U}) {
        if (count >= 2 && rows == 1) continue;
        Case c(size, 4);
        c.state = *md::Gemma4State(c.p, 16384, 128);
        c.shape = {};
        for (std::uint32_t i = 0; i < count; ++i)
          c.shape.segments.push_back({i, rows, 8192 + i, 8448, c.state.local_cells});
        c.shape.outputs = count;
        kg::Gemma4GraphOptions options;
        options.device_masks = true;
        options.attention_mode = kg::Gemma4AttentionMode::kOwners;
        auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, count, options));
        ASSERT_TRUE(arena);
        auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
        ASSERT_TRUE(g);
        EXPECT_EQ(g->attention_mode, kg::Gemma4AttentionMode::kIndependent);
        EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT), c.p.layers * count);
      }
  Case c(31, 4);
  c.state = *md::Gemma4State(c.p, 16384, 128);
  c.shape = {};
  for (std::uint32_t i = 0; i < 4; ++i)
    c.shape.segments.push_back({i, 1, 8192 + i, 8448, c.state.local_cells});
  c.shape.outputs = 4;
  kg::Gemma4GraphOptions options;
  options.device_masks = true;
  // Fund the transform without selecting it during initial construction.
  auto estimate_options = options;
  estimate_options.attention_mode = kg::Gemma4AttentionMode::kOwners;
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 4, estimate_options));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(g);
  ggml_tensor* writer = nullptr;
  for (auto* node : g->nodes)
    if (node->op == GGML_OP_SET_ROWS && node->src[2] == g->segments[0].caches[0].first)
      writer = node;
  ASSERT_NE(writer, nullptr);
  auto* cells = writer->src[1];
  writer->src[1] = g->segments[1].local_cells;
  const auto used = arena->used();
  EXPECT_FALSE(kg::TransformGemma4Attention(*arena, *g, kg::Gemma4AttentionMode::kOwners));
  EXPECT_EQ(arena->used(), used);
  EXPECT_EQ(g->attention_mode, kg::Gemma4AttentionMode::kIndependent);
  writer->src[1] = cells;
  EXPECT_TRUE(kg::TransformGemma4Attention(*arena, *g, kg::Gemma4AttentionMode::kOwners));
  EXPECT_FALSE(kg::TransformGemma4Attention(*arena, *g, kg::Gemma4AttentionMode::kPacked));
}

TEST(Gemma4Graph, CommonOwnerReadsPreserveEqualGraphsAndRejectTheSecondBrokenWriter) {
  for (const auto size : {26U, 31U}) {
    Case c(size, 2);
    c.shape = {};
    for (std::uint32_t i = 0; i < 2; ++i) c.shape.segments.push_back({i, 1, 384 + i, 512, 512});
    c.shape.outputs = 2;
    std::array<std::size_t, 3> nodes{}, used{};
    for (std::size_t policy = 0; policy < 3; ++policy) {
      kg::Gemma4GraphOptions options;
      options.device_masks = true;
      options.attention_mode = kg::Gemma4AttentionMode::kOwners;
      options.common_owner_reads = policy != 0;
      options.bounded_owner_roots = policy == 2;
      auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2, options));
      ASSERT_TRUE(arena);
      auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
      ASSERT_TRUE(g);
      nodes[policy] = g->nodes.size();
      used[policy] = arena->used();
      EXPECT_EQ(Count(*g, GGML_OP_FILL), 0U);
      for (std::uint32_t layer = 0; layer < c.p.layers; ++layer) {
        const auto name = "packed.blk." + std::to_string(layer) + ".attention";
        const auto found = std::ranges::find_if(g->nodes, [&](const auto* node) {
          return std::string_view(ggml_get_name(node)) == name;
        });
        auto* out = found == g->nodes.end() ? nullptr : *found;
        ASSERT_NE(out, nullptr);
        EXPECT_EQ(kg::LlmpOpInt(out, 4), 0);
        for (const auto index : {2, 3, 6, 7}) EXPECT_EQ(out->src[index]->src[0]->op, GGML_OP_VIEW);
      }
    }
    EXPECT_EQ(nodes[0], nodes[1]);
    EXPECT_EQ(used[0], used[1]);
    EXPECT_EQ(nodes[0], nodes[2]);
    EXPECT_EQ(used[0], used[2]);
    kg::Gemma4GraphOptions options;
    options.device_masks = true;
    auto estimate = options;
    estimate.attention_mode = kg::Gemma4AttentionMode::kOwners;
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2, estimate));
    ASSERT_TRUE(arena);
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
    ASSERT_TRUE(g);
    g->options = estimate;
    g->options.common_owner_reads = true;
    for (auto* node : g->nodes)
      if (node->op == GGML_OP_SET_ROWS && node->src[2] == g->segments[1].caches[0].first)
        node->src[1] = g->segments[0].local_cells;
    const auto before = arena->used();
    EXPECT_FALSE(kg::TransformGemma4Attention(*arena, *g, kg::Gemma4AttentionMode::kOwners));
    EXPECT_EQ(arena->used(), before);
    EXPECT_EQ(g->attention_mode, kg::Gemma4AttentionMode::kIndependent);
  }
}

TEST(Gemma4Graph, CommonOwnerReadsKeepActualRootsAndPadOnlyBoundedTwoOwnerWaves) {
  for (const auto size : {26U, 31U})
    for (const bool reverse : {false, true})
      for (const std::size_t policy : {0U, 1U, 2U}) {
        const bool common = policy != 0, bounded = policy == 2;
        Case c(size, 2);
        c.shape = {};
        for (std::uint32_t i = 0; i < 2; ++i) {
          const auto read = (i == 0) != reverse ? 512U : 1024U;
          c.shape.segments.push_back({i, 1, read - 128, read, read});
        }
        c.shape.outputs = 2;
        kg::Gemma4GraphOptions options;
        options.device_masks = true;
        options.attention_mode = kg::Gemma4AttentionMode::kOwners;
        options.common_owner_reads = common;
        options.bounded_owner_roots = bounded;
        auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2, options));
        ASSERT_TRUE(arena);
        auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
        ASSERT_TRUE(g) << llmp::test_support::Failed(g, &kg::KernelFailure::detail).value_or("");
        EXPECT_EQ(g->attention_mode, common ? kg::Gemma4AttentionMode::kOwners
                                            : kg::Gemma4AttentionMode::kIndependent);
        if (!common) continue;
        BindLeaves(*g);
        EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT), 0U);
        for (std::uint32_t layer = 0; layer < c.p.layers; ++layer) {
          const auto name = "packed.blk." + std::to_string(layer) + ".attention";
          const auto found = std::ranges::find_if(g->nodes, [&](const auto* node) {
            return std::string_view(ggml_get_name(node)) == name;
          });
          auto* out = found == g->nodes.end() ? nullptr : *found;
          ASSERT_NE(out, nullptr);
          EXPECT_EQ(kg::LlmpOpOf(out), kg::LlmpOp::kFlashAttnOwners);
          EXPECT_EQ(kg::LlmpOpInt(out, 4), bounded ? 1 : 0);
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(out));
          EXPECT_EQ(out->src[1]->ne[0], 1024);
          EXPECT_EQ(out->src[1]->ne[1], 32);
          for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t which = 0; which < 2; ++which) {
              const auto read = c.shape.segments[i].global_n_kv;
              auto* raw = out->src[2 + i + which * 4]->src[0];
              if (read == 512 && !bounded) {
                ASSERT_EQ(raw->op, GGML_OP_CONCAT);
                EXPECT_EQ(raw->op_params[0], 2);
                EXPECT_EQ(raw->src[1]->op, GGML_OP_FILL);
                EXPECT_EQ(raw->src[1]->src[0], nullptr);
                EXPECT_EQ(raw->src[1]->op_params[0], 0);
                raw = raw->src[0];
              }
              ASSERT_EQ(raw->op, GGML_OP_VIEW);
              EXPECT_EQ(raw->ne[2], read);
              ASSERT_EQ(raw->src[0]->op, GGML_OP_SET_ROWS);
              const auto& caches = g->segments[i].caches[layer];
              EXPECT_EQ(raw->view_src, which == 0 ? caches.first : caches.second);
            }
        }
        // Wider cohorts retain the original unequal-width refusal/fallback.
        auto wider = c.shape;
        wider.segments.push_back({2, 1, 384, 512, 512});
        wider.outputs = 3;
        auto other = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 3, options));
        ASSERT_TRUE(other);
        auto independent = kg::BuildGemma4Graph(*other, c.p, c.binding, c.state, wider, options);
        ASSERT_TRUE(independent);
        EXPECT_EQ(independent->attention_mode, kg::Gemma4AttentionMode::kIndependent);
      }
}

TEST(Gemma4Graph, VariableOwnerAttentionKeepsSlotOrderWritersAndFundedMetadata) {
  for (const auto size : {26U, 31U})
    for (const auto max_rows : {128U, 256U, 1024U, 8192U})
      for (const auto mode : {kg::Gemma4AttentionMode::kPacked, kg::Gemma4AttentionMode::kOwners}) {
        Case c(size, 4);
        c.state = *md::Gemma4State(c.p, 16384, max_rows);
        c.shape = {};
        const std::array<std::uint32_t, 4> slots{7, 2, 15, 5};
        const std::array<std::int32_t, 1> tokens{1};
        std::vector<md::Gemma4Segment> segments;
        for (std::size_t i = 0; i < 4; ++i)
          segments.push_back({slots[i], 8192U + std::uint32_t(i), tokens});
        const auto input = md::Gemma4Chunk(c.p, c.state, segments, false);
        ASSERT_TRUE(input);
        for (const auto& seg : input->segments)
          c.shape.segments.push_back(
              {seg.slot, seg.rows, seg.n_past, seg.global_n_kv, seg.local_n_kv});
        c.shape.outputs = 4;
        kg::Gemma4GraphOptions options;
        options.device_masks = true;
        options.narrow_final = true;
        options.attention_mode = mode;
        const auto count = kg::Gemma4GraphTensors(c.p, 4, options);
        EXPECT_EQ(count, kg::Gemma4GraphTensors(c.p, 4) + c.p.layers * 64);
        auto arena = kg::TensorArena::Create(count);
        ASSERT_TRUE(arena);
        auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
        ASSERT_TRUE(g) << llmp::test_support::Failed(g)->detail;
        const bool eligible = size != 31 || max_rows != 8192;
        EXPECT_EQ(g->attention_mode, eligible ? mode : kg::Gemma4AttentionMode::kIndependent);
        EXPECT_EQ(Count(*g, GGML_OP_SET_ROWS), c.p.layers * 8);
        if (!eligible) {
          EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT), c.p.layers * 4);
          continue;
        }
        EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT),
                  mode == kg::Gemma4AttentionMode::kPacked ? c.p.layers : 0);
        std::size_t owner_nodes = 0;
        for (auto* node : g->nodes) {
          if (kg::LlmpOpOf(node) != kg::LlmpOp::kFlashAttnOwners) continue;
          ++owner_nodes;
          for (const auto index : {2, 3, 4, 5, 6, 7, 8, 9}) {
            const auto* raw = node->src[index]->src[0];
            ASSERT_NE(raw, nullptr);
            const auto* writer = raw->src[0];
            ASSERT_NE(writer, nullptr);
            EXPECT_EQ(writer->op, GGML_OP_SET_ROWS);
            EXPECT_EQ(raw->view_src, writer->view_src);
            EXPECT_LT(std::ranges::find(g->nodes, writer), std::ranges::find(g->nodes, node));
          }
        }
        EXPECT_EQ(owner_nodes, mode == kg::Gemma4AttentionMode::kOwners ? c.p.layers : 0);
        for (std::size_t i = 0; i < 4; ++i) {
          auto* flat = g->Named("blk.0.slot." + std::to_string(slots[i]) + ".attention");
          ASSERT_NE(flat, nullptr);
          ASSERT_NE(flat->src[0], nullptr);
          EXPECT_EQ(flat->view_src, flat->src[0]);
          EXPECT_EQ(flat->src[0]->view_offs, i * flat->src[0]->src[0]->nb[3]);
        }
        // Crossing 8192 for only some owners cannot
        // broaden the readable span to a peer's width.
        auto crossing = c.shape;
        crossing.segments[0].n_past = 8191;
        crossing.segments[0].global_n_kv = 8192;
        auto fallback_arena = kg::TensorArena::Create(count);
        ASSERT_TRUE(fallback_arena);
        auto fallback =
            kg::BuildGemma4Graph(*fallback_arena, c.p, c.binding, c.state, crossing, options);
        ASSERT_TRUE(fallback);
        EXPECT_EQ(fallback->attention_mode, kg::Gemma4AttentionMode::kIndependent);
        EXPECT_EQ(Count(*fallback, GGML_OP_FLASH_ATTN_EXT), c.p.layers * 4);
        auto feature_shape = c.shape;
        feature_shape.feature_outputs = 1;
        auto feature_options = options;
        feature_options.narrow_final = false;
        auto feature_arena = kg::TensorArena::Create(count);
        ASSERT_TRUE(feature_arena);
        auto feature_graph = kg::BuildGemma4Graph(*feature_arena, c.p, c.binding, c.state,
                                                  feature_shape, feature_options);
        ASSERT_TRUE(feature_graph);
        EXPECT_EQ(feature_graph->attention_mode, kg::Gemma4AttentionMode::kIndependent);
        EXPECT_NE(feature_graph->normalized_features, nullptr);
        EXPECT_EQ(Count(*feature_graph, GGML_OP_FLASH_ATTN_EXT), c.p.layers * 4);
        auto state_shape = c.shape;
        state_shape.outputs = 0;
        state_shape.output_mode = kg::Gemma4OutputMode::kStateOnly;
        auto state_arena = kg::TensorArena::Create(count);
        ASSERT_TRUE(state_arena);
        auto state_graph =
            kg::BuildGemma4Graph(*state_arena, c.p, c.binding, c.state, state_shape, options);
        ASSERT_TRUE(state_graph);
        EXPECT_EQ(state_graph->attention_mode, kg::Gemma4AttentionMode::kIndependent);
        EXPECT_EQ(Count(*state_graph, GGML_OP_FLASH_ATTN_EXT), (c.p.layers - 1) * 4);
        auto unknown = options;
        unknown.attention_mode = static_cast<kg::Gemma4AttentionMode>(255);
        EXPECT_FALSE(kg::CheckGemma4Graph(c.p, c.binding, c.state, c.shape, unknown));
        auto independent = options;
        independent.attention_mode = kg::Gemma4AttentionMode::kIndependent;
        EXPECT_NE(independent, options);
      }
}

TEST(Gemma4Graph, CompleteQuadsKeepProductsAndIndependentTailsInOriginalOwnerOrder) {
  for (const auto size : {26U, 31U})
    for (const auto owners : {4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U})
      for (const auto mode : {kg::Gemma4AttentionMode::kPacked, kg::Gemma4AttentionMode::kOwners}) {
        Case c(size);
        c.state = *md::Gemma4State(c.p, 4096, size == 26 ? 1024 : 256);
        c.shape = {};
        constexpr std::array<std::uint32_t, 12> slots{7, 2, 15, 5, 1, 12, 9, 4, 11, 3, 14, 6};
        for (std::uint32_t i = 0; i < owners; ++i) {
          const auto read = i < 4 ? 512U : 1024U;
          c.shape.segments.push_back({slots[i], 1, read - 128 + i, read, read});
        }
        c.shape.outputs = owners;
        kg::Gemma4GraphOptions options;
        options.device_masks = true;
        options.narrow_final = true;
        auto estimate = options;
        estimate.attention_mode = mode;
        const auto tensors = kg::Gemma4GraphTensors(c.p, owners, estimate);
        EXPECT_EQ(tensors, kg::Gemma4GraphTensors(c.p, owners) +
                               c.p.layers * 64 *
                                   (mode == kg::Gemma4AttentionMode::kOwners ? (owners + 3) / 4
                                                                             : owners / 4));
        auto arena = kg::TensorArena::Create(tensors);
        ASSERT_TRUE(arena);
        auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
        ASSERT_TRUE(graph);
        auto* projection = graph->Named("blk.0.attn_projection");
        ASSERT_NE(projection, nullptr);
        ASSERT_EQ(projection->src[1]->ne[1], owners);
        const auto products = Count(*graph, GGML_OP_MUL_MAT);
        ASSERT_TRUE(kg::TransformGemma4Attention(*arena, *graph, mode));
        EXPECT_EQ(graph->attention_mode, mode);
        EXPECT_EQ(graph->attention_quad_mask, (1U << (owners / 4)) - 1);
        EXPECT_EQ(graph->Named("blk.0.attn_projection"), projection);
        EXPECT_EQ(projection->src[1]->ne[1], owners);
        EXPECT_EQ(Count(*graph, GGML_OP_MUL_MAT), products);
        EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), c.p.layers * owners * 2);
        const auto independent = owners % 4;
        EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT),
                  c.p.layers *
                      (independent + (mode == kg::Gemma4AttentionMode::kPacked ? owners / 4 : 0)));
        std::size_t owner_nodes = 0;
        for (auto* node : graph->nodes) {
          if (kg::LlmpOpOf(node) != kg::LlmpOp::kFlashAttnOwners) continue;
          ++owner_nodes;
          EXPECT_EQ(node->ne[3], 4);
          EXPECT_EQ(kg::LlmpOpInt(node, 0), 4);  // Widths differ between the quads.
          for (std::size_t i = 0; i < 8; ++i) {
            auto* raw = node->src[2 + i]->src[0];
            ASSERT_NE(raw, nullptr);
            ASSERT_NE(raw->src[0], nullptr);
            EXPECT_EQ(raw->src[0]->op, GGML_OP_SET_ROWS);
            EXPECT_LT(std::ranges::find(graph->nodes, raw->src[0]),
                      std::ranges::find(graph->nodes, node));
          }
        }
        EXPECT_EQ(owner_nodes,
                  mode == kg::Gemma4AttentionMode::kOwners ? c.p.layers * (owners / 4) : 0);
        // Owners' Q rows are viewed, never copied; masks are joined once per
        // kind and quad; one quad's output feeds the projection directly.
        std::size_t concats = 0;
        for (auto* node : graph->nodes) {
          if (node->op != GGML_OP_CONCAT) continue;
          ++concats;
          for (auto* src : {node->src[0], node->src[1]}) {
            const auto* root = src->view_src != nullptr ? src->view_src : src;
            EXPECT_EQ(std::string_view(ggml_get_name(root)).find("q_rope"), std::string_view::npos);
          }
        }
        if (owners == 4) {
          // Packed attention also joins each layer's K and V (copies by design).
          EXPECT_EQ(concats, mode == kg::Gemma4AttentionMode::kOwners ? 6U : 6U + c.p.layers * 6U);
          EXPECT_EQ(projection->src[1]->op, GGML_OP_RESHAPE);
        }
        for (std::uint32_t i = 0; i < owners; ++i) {
          auto* flat = graph->Named("blk.0.slot." + std::to_string(slots[i]) + ".attention");
          ASSERT_NE(flat, nullptr);
          auto* attention = flat->src[0];
          ASSERT_NE(attention, nullptr);
          if (i < owners - independent) {
            ASSERT_EQ(attention->op, GGML_OP_VIEW);
            ASSERT_NE(attention->src[0], nullptr);
            EXPECT_EQ(attention->view_offs, (i % 4) * attention->src[0]->nb[3]);
            const auto mask_index = mode == kg::Gemma4AttentionMode::kOwners ? 1 : 3;
            EXPECT_EQ(attention->src[0]->src[mask_index]->ne[0], i < 4 ? 512 : 1024);
          } else {
            EXPECT_EQ(attention->op, GGML_OP_FLASH_ATTN_EXT);
            EXPECT_EQ(attention->ne[3], 1);
          }
        }
        EXPECT_LE(arena->used(), arena->bytes());
        EXPECT_TRUE(arena->Reserve(0));
      }
}
TEST(Gemma4Graph, UnsupportedQuadWidthsPreserveOnlyThatQuadAndBrokenLastWriterRefusesAll) {
  for (const auto size : {26U, 31U})
    for (const auto unsupported_quad : {0U, 1U, 2U, 3U}) {
      Case c(size);
      c.state = *md::Gemma4State(c.p, 4096, size == 26 ? 1024 : 256);
      c.shape = {};
      for (std::uint32_t i = 0; i < 12; ++i) {
        const auto read = i / 4 == unsupported_quad && i % 4 == 0 ? 256U : 512U;
        c.shape.segments.push_back({i, 1, read - 128, read, read});
      }
      c.shape.outputs = 12;
      kg::Gemma4GraphOptions options;
      options.device_masks = true;
      auto estimate = options;
      estimate.attention_mode = kg::Gemma4AttentionMode::kOwners;
      auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 12, estimate));
      ASSERT_TRUE(arena);
      auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
      ASSERT_TRUE(graph);
      if (unsupported_quad == 3) {
        ggml_tensor* writer = nullptr;
        for (auto* node : graph->nodes)
          if (node->op == GGML_OP_SET_ROWS && node->src[2] == graph->segments[11].caches[0].first)
            writer = node;
        ASSERT_NE(writer, nullptr);
        auto* cells = writer->src[1];
        writer->src[1] = graph->segments[10].local_cells;
        const auto used = arena->used();
        const auto nodes = graph->nodes;
        const auto names = graph->named;
        auto* first_attention = graph->Named("blk.0.slot.0.attention")->src[0];
        EXPECT_FALSE(
            kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
        EXPECT_EQ(arena->used(), used);
        EXPECT_EQ(graph->nodes, nodes);
        EXPECT_EQ(graph->named, names);
        EXPECT_EQ(graph->Named("blk.0.slot.0.attention")->src[0], first_attention);
        EXPECT_EQ(first_attention->op, GGML_OP_FLASH_ATTN_EXT);
        EXPECT_EQ(graph->attention_mode, kg::Gemma4AttentionMode::kIndependent);
        EXPECT_EQ(graph->attention_quad_mask, 0U);
        writer->src[1] = cells;
      }
      ASSERT_TRUE(kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
      EXPECT_EQ(graph->attention_quad_mask,
                unsupported_quad == 3 ? 7U : (7U ^ (1U << unsupported_quad)));
      EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), unsupported_quad == 3 ? 0U : c.p.layers * 4);
    }
}

TEST(Gemma4Graph, QuadMetadataRefusalLeavesOriginalEdgesAndModesUntouched) {
  Case c(31);
  c.state = *md::Gemma4State(c.p, 4096, 256);
  c.shape = {};
  for (std::uint32_t i = 0; i < 8; ++i) c.shape.segments.push_back({i, 1, 300 + i, 512, 512});
  c.shape.outputs = 8;
  kg::Gemma4GraphOptions options;
  options.device_masks = true;
  auto estimate = options;
  estimate.attention_mode = kg::Gemma4AttentionMode::kOwners;
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 8, estimate));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph);
  // Consume only pre-funded CPU descriptor room; there are no payload allocations.
  while (arena->Reserve(1)) ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 1);
  const auto used = arena->used();
  const auto nodes = graph->nodes;
  auto* original = graph->Named("blk.0.slot.0.attention")->src[0];
  EXPECT_FALSE(kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
  EXPECT_EQ(arena->used(), used);
  EXPECT_EQ(graph->nodes, nodes);
  EXPECT_EQ(graph->Named("blk.0.slot.0.attention")->src[0], original);
  EXPECT_EQ(graph->attention_mode, kg::Gemma4AttentionMode::kIndependent);
  EXPECT_EQ(graph->attention_quad_mask, 0U);
}

TEST(Gemma4Graph, StateOnlyKeepsEveryCacheStoreAndOmitsOnlyTheFinalTail) {
  for (const auto size : {26U, 31U}) {
    Case c(size, 2);
    const auto full = c.shape;
    c.shape.outputs = 0;
    c.shape.output_mode = kg::Gemma4OutputMode::kStateOnly;
    EXPECT_NE(c.shape, full);
    auto same_dimensions = c.shape;
    same_dimensions.output_mode = kg::Gemma4OutputMode::kHead;
    EXPECT_NE(c.shape, same_dimensions);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2));
    ASSERT_TRUE(arena);
    kg::Gemma4GraphOptions options;
    options.device_masks = true;
    options.narrow_final = true;
    auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
    ASSERT_TRUE(graph) << llmp::test_support::Failed(graph)->detail;
    EXPECT_EQ(graph->logits, nullptr);
    EXPECT_EQ(graph->hidden, nullptr);
    EXPECT_EQ(graph->out_ids, nullptr);
    EXPECT_EQ(graph->normalized_features, nullptr);
    EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 2 * c.p.layers * 2);
    EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), (c.p.layers - 1) * 2);
    const auto prefix = "blk." + std::to_string(c.p.layers - 1) + ".";
    EXPECT_NE(graph->Named(prefix + "attn_input"), nullptr);
    EXPECT_NE(graph->Named(prefix + "k_rope"), nullptr);
    EXPECT_NE(graph->Named(prefix + "v_norm"), nullptr);
    for (const auto name : {"q_rope", "attn_projection", "output", "slot.0.attention"})
      EXPECT_EQ(graph->Named(prefix + name), nullptr);
    EXPECT_NE(graph->Named("blk.0.output"), nullptr);
    for (const auto& segment : graph->segments)
      for (const auto [k, v] : segment.caches) {
        EXPECT_NE(k, nullptr);
        EXPECT_NE(v, nullptr);
      }
    {
      auto bad = c.shape;
      bad.outputs = 1;
      EXPECT_FALSE(kg::CheckGemma4Graph(c.p, c.binding, c.state, bad, options));
      bad.outputs = 0;
      bad.feature_outputs = 1;
      EXPECT_FALSE(kg::CheckGemma4Graph(c.p, c.binding, c.state, bad, options));
      bad.feature_outputs = 0;
      bad.output_mode = static_cast<kg::Gemma4OutputMode>(255);
      EXPECT_FALSE(kg::CheckGemma4Graph(c.p, c.binding, c.state, bad, options));
    }
    for (unsigned changed = 0; changed < 4; ++changed) {
      auto bad = options;
      bad.narrow_final = false;
      if (changed == 0) bad.head = false;
      if (changed == 1) bad.hidden_input = true;
      if (changed == 2) {
        bad.first_layer = 1;
        bad.hidden_input = true;
      }
      if (changed == 3) bad.layer_count = c.p.layers - 1;
      EXPECT_FALSE(kg::CheckGemma4Graph(c.p, c.binding, c.state, c.shape, bad));
    }
  }
}

TEST(Gemma4Graph, ExplicitSegmentStoresFuseOnlyEligibleUnkeptKRotations) {
  for (const auto size : {26U, 31U})
    for (const auto slots : {1U, 2U, 4U, 16U}) {
      Case c(size);
      c.state = std::move(*md::Gemma4State(c.p, 262144, 64));
      std::array<std::int32_t, 4> tokens{1, 2, 3, 4};
      std::vector<md::Gemma4Segment> segments;
      for (std::uint32_t i = 0; i < slots; ++i)
        segments.push_back({i, 1279 + i * 7, std::span(tokens).first(i % 4 + 1)});
      c.input = std::move(*md::Gemma4Chunk(c.p, c.state, segments, false));
      c.shape = {};
      for (const auto& part : c.input.segments)
        c.shape.segments.push_back(
            {part.slot, part.rows, part.n_past, part.global_n_kv, part.local_n_kv});
      c.shape.outputs = slots;
      auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, slots));
      ASSERT_TRUE(arena);
      kg::Gemma4GraphOptions options;
      options.device_masks = true;
      options.rope_store = true;
      auto built = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
      ASSERT_TRUE(built);
      BindLeaves(*built);
      kg::DeviceChoices choices;
      choices.fuse_rope_store = true;
      choices.quant = [](const auto*) { return kg::QuantMulMatPath::kTile; };
      choices.mul_mat = [](const auto*) { return kg::MulMatPath::kCublas; };
      auto plan = kg::PlanGraph(built->nodes, false, choices);
      ASSERT_TRUE(plan) << llmp::test_support::Failed(plan)->detail;
      const auto stores = [](const auto& p) {
        return std::ranges::count_if(p.steps, [](const auto& step) {
          return step.operation == llmp::execution::Operation::kRopeSetRows;
        });
      };
      EXPECT_EQ(stores(*plan), c.p.layers * slots);
      auto placement = kg::PlaceActivations(built->nodes, *plan, built->inputs, 256);
      ASSERT_TRUE(placement) << llmp::test_support::Failed(placement)->detail;
      for (const auto& step : plan->steps)
        if (step.operation == llmp::execution::Operation::kRopeSetRows) {
          EXPECT_TRUE(kg::CheckRopeSetRows(step.nodes[0], step.nodes[1]));
          EXPECT_EQ(step.nodes[1]->src[0]->view_offs, 0U);
          EXPECT_EQ(step.nodes[0]->ne[2], step.nodes[1]->src[1]->ne[0]);
          // The fused launcher does not write the rotated intermediate, but
          // current placement still funds it. No catalog exclusion is needed.
          EXPECT_TRUE(std::ranges::find(placement->offsets, step.nodes[0],
                                        &std::pair<ggml_tensor*, std::uint64_t>::first) !=
                      placement->offsets.end());
        }
      auto* kept = built->Named("blk.0.slot.0.k_rope");
      ASSERT_NE(kept, nullptr);
      const std::array<ggml_tensor*, 1> keep{kept};
      auto protected_plan = kg::PlanGraph(built->nodes, false, choices, keep);
      ASSERT_TRUE(protected_plan);
      EXPECT_EQ(stores(*protected_plan), c.p.layers * slots - 1);
      auto* kept_view = ggml_view_2d(arena->context(), kept, kept->ne[0] * kept->ne[1], kept->ne[2],
                                     kept->nb[2], 0);
      const std::array<ggml_tensor*, 1> keep_view{kept_view};
      protected_plan = kg::PlanGraph(built->nodes, false, choices, keep_view);
      ASSERT_TRUE(protected_plan);
      EXPECT_EQ(stores(*protected_plan), c.p.layers * slots - 1);
      choices.fuse_rope_store = false;
      auto original = kg::PlanGraph(built->nodes, false, choices);
      ASSERT_TRUE(original);
      EXPECT_EQ(stores(*original), 0);
    }
}

TEST(Gemma4Graph, PartialFlatteningFallsBackToCheckedPrimitiveStores) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 8, 2);
  auto* positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2);
  auto* rope = ggml_rope_ext(c, x, positions, nullptr, 256, GGML_ROPE_TYPE_NEOX, 262144, 10000, 1,
                             0, 1, 32, 1);
  auto* partial = ggml_view_2d(c, rope, 2048, 1, rope->nb[2], 0);
  auto* ids = ggml_new_tensor_1d(c, GGML_TYPE_I64, 1);
  auto* cache = ggml_new_tensor_2d(c, GGML_TYPE_F16, 2048, 1280);
  auto* store = ggml_set_rows(c, cache, partial, ids);
  std::uint64_t address = std::uint64_t{1} << 36U;
  for (auto* leaf : {x, positions, ids, cache}) {
    kg::TensorArena::Bind(leaf, address);
    address += ((ggml_nbytes(leaf) + 255) / 256 + 1) * 256;
  }
  const std::array<ggml_tensor*, 3> nodes{rope, partial, store};
  kg::BindDistinct(nodes, std::uint64_t{1} << 48U);
  ASSERT_TRUE(kg::CheckRope(rope));
  ASSERT_TRUE(kg::CheckSetRows(store));
  EXPECT_FALSE(kg::CheckRopeSetRows(rope, store));
  kg::DeviceChoices choices;
  choices.fuse_rope_store = true;
  auto plan = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->steps.size(), 2U);
  EXPECT_EQ(plan->steps[0].operation, llmp::execution::Operation::kRope);
  EXPECT_EQ(plan->steps[1].operation, llmp::execution::Operation::kSetRows);
}

TEST(Gemma4Graph, BothActualContractsBuildCompleteTextGraphs) {
  for (const auto size : {26U, 31U}) {
    Case c(size);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
    ASSERT_TRUE(arena);
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
    ASSERT_TRUE(g) << llmp::test_support::Failed(g)->detail;
    EXPECT_EQ(g->logits->ne[0], c.p.vocab);
    EXPECT_EQ(g->logits->ne[1], 1);
    EXPECT_EQ(g->hidden->ne[1], 2);
    EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT), c.p.layers);
    EXPECT_EQ(Count(*g, GGML_OP_SET_ROWS), c.p.layers * 2);
    EXPECT_EQ(Count(*g, GGML_OP_GLU), c.p.layers * (size == 26 ? 2 : 1));
    EXPECT_EQ(Count(*g, GGML_OP_MUL_MAT_ID), size == 26 ? c.p.layers * 2 : 0);
    EXPECT_EQ(g->nodes.back(), g->logits);
    const auto embd = std::ranges::find_if(
        g->weights, [&](const auto& w) { return w.resource.index == c.binding.token_embd.index; });
    ASSERT_NE(embd, g->weights.end());
    EXPECT_EQ(g->logits->src[0]->src[0]->src[0]->src[0], embd->tensor);
  }
}
TEST(Gemma4Graph, JoinedProductsKeepEverySlotAttentionAndCacheIndependent) {
  for (const auto slots : {1U, 2U, 4U}) {
    Case c(26, slots);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, slots));
    ASSERT_TRUE(arena);
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
    ASSERT_TRUE(g) << llmp::test_support::Failed(g)->detail;
    EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT), c.p.layers * slots);
    EXPECT_EQ(Count(*g, GGML_OP_SET_ROWS), c.p.layers * slots * 2);
    // Q projection and routed products join row-local inputs, not KV heads.
    EXPECT_EQ(g->Named("blk.0.q_rope")->ne[2], slots * 2);
    EXPECT_EQ(g->Named("blk.0.expert_activation")->ne[2], slots * 2);
    for (std::uint32_t i = 0; i < slots; ++i) {
      const auto& seg = g->segments[i];
      EXPECT_EQ(seg.first_row, i * 2);
      EXPECT_EQ(seg.shape.n_past, i * 1100);
      EXPECT_EQ(seg.local_mask->ne[1], 32);
      EXPECT_EQ(seg.caches[0].first->ne[1], c.state.local_cells);
      EXPECT_EQ(seg.caches[5].first->ne[1], c.state.global_cells);
      EXPECT_NE(seg.caches[5].first, seg.caches[5].second);
      for (std::uint32_t j = 0; j < i; ++j)
        EXPECT_NE(seg.caches[0].first, g->segments[j].caches[0].first);
    }
    for (const auto* t : g->nodes) {
      if (t->op != GGML_OP_FLASH_ATTN_EXT) continue;
      EXPECT_EQ(t->src[0]->ne[1], 2);
      EXPECT_EQ(t->src[0]->ne[3], 1);
      EXPECT_EQ(std::bit_cast<float>(t->op_params[0]), 1.0f);
      EXPECT_EQ(t->src[0]->ne[2] / t->src[1]->ne[2], t->ne[0] == 256 ? 2 : 8);
    }
  }
}
TEST(Gemma4Graph, GlobalValueReadsRawKBeforeLearnedNormAndRotation) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(g);
  auto* v = g->Named("blk.5.v_norm");
  auto* k = g->Named("blk.5.k_rope");
  ASSERT_NE(v, nullptr);
  ASSERT_NE(k, nullptr);
  EXPECT_EQ(v->op, GGML_OP_RMS_NORM);
  // V reshape's source is exactly K learned norm's reshape source.
  EXPECT_EQ(v->src[0]->src[0], k->src[0]->src[0]->src[0]->src[0]);
  EXPECT_EQ(k->op_params[1], 512);
  EXPECT_EQ(g->Named("blk.0.k_rope")->op_params[1], 256);
  ASSERT_NE(k->src[2], nullptr);
  EXPECT_EQ(k->src[2]->ne[0], 256);
  EXPECT_EQ(g->Named("blk.0.k_rope")->src[2], nullptr);
}
TEST(Gemma4Graph, SelectedScalePrecedesWeightsAndAuthoritativeLeftFold) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(g);
  auto* weights = g->Named("blk.0.expert_weights");
  ASSERT_NE(weights, nullptr);
  auto* div = weights->src[0];
  EXPECT_EQ(div->op, GGML_OP_DIV);
  EXPECT_EQ(div->src[1]->op, GGML_OP_CLAMP);
  EXPECT_EQ(div->src[1]->src[0]->op, GGML_OP_SUM_ROWS);
  auto* scaled = g->Named("blk.0.expert_scaled_down");
  ASSERT_NE(scaled, nullptr);
  EXPECT_EQ(scaled->op, GGML_OP_MUL);
  EXPECT_EQ(scaled->src[0]->op, GGML_OP_MUL_MAT_ID);
  EXPECT_EQ(scaled->src[1]->op, GGML_OP_GET_ROWS);
  const auto at = std::ranges::find(g->nodes, scaled);
  ASSERT_NE(at, g->nodes.end());
  const auto weighted = std::ranges::find_if(at, g->nodes.end(), [&](const auto* t) {
    return t->op == GGML_OP_MUL && t->src[0] == scaled && t->src[1] == weights;
  });
  ASSERT_NE(weighted, g->nodes.end());
  std::vector<ggml_tensor*> views;
  for (auto* t : g->nodes)
    if (t->op == GGML_OP_VIEW && t->src[0] == *weighted) views.push_back(t);
  ASSERT_EQ(views.size(), 8U);
  const auto first_add = std::ranges::find_if(g->nodes, [&](const auto* t) {
    return t->op == GGML_OP_ADD && t->src[0] == views[0] && t->src[1] == views[1];
  });
  ASSERT_NE(first_add, g->nodes.end());
  EXPECT_LT(std::ranges::find(g->nodes, views.back()) - g->nodes.begin(),
            first_add - g->nodes.begin());
}
TEST(Gemma4Graph, PartialLayerControlsRemainExplicitAndUseSameArithmetic) {
  Case c;
  c.shape.outputs = 0;
  const kg::Gemma4GraphOptions o{
      .expert_stride = {}, .first_layer = 5, .layer_count = 1, .hidden_input = true, .head = false};
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, o);
  ASSERT_TRUE(g) << llmp::test_support::Failed(g)->detail;
  EXPECT_EQ(g->options, o);
  EXPECT_EQ(g->tokens, nullptr);
  EXPECT_EQ(g->logits, nullptr);
  EXPECT_NE(g->input_hidden, nullptr);
  EXPECT_EQ(Count(*g, GGML_OP_FLASH_ATTN_EXT), 1U);
  EXPECT_EQ(g->hidden->ne[0], 2816);
}
TEST(Gemma4Graph, FrontierNarrowingPreservesAllTargetCacheWrites) {
  Case c(26, 2);
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape,
                                {.expert_stride = {}, .narrow_final = true});
  ASSERT_TRUE(g);
  EXPECT_EQ(g->hidden->ne[1], 2);
  EXPECT_EQ(g->Named("blk.29.q_rope")->ne[2], 4);
  EXPECT_EQ(g->Named("blk.29.expert_activation")->ne[2], 2);
  EXPECT_EQ(Count(*g, GGML_OP_SET_ROWS), 120U);
}
TEST(Gemma4Graph, GreedyShapesAddOnlyTheFrontierArgmaxAfterTheHead) {
  Case c(26, 2);
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto shape = c.shape;
  shape.greedy = true;
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, shape);
  ASSERT_TRUE(g);
  ASSERT_NE(g->greedy, nullptr);
  EXPECT_EQ(g->greedy->type, GGML_TYPE_I32);
  EXPECT_EQ(g->greedy->ne[0], g->logits->ne[1]);
  EXPECT_EQ(g->greedy->src[0], g->logits);
  EXPECT_EQ(kg::LlmpOpInt(g->greedy, 1), static_cast<std::int32_t>(kg::ArgmaxFlavor::kHostGreedy));
  EXPECT_EQ(g->nodes.back(), g->greedy);
  auto plain = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2));
  ASSERT_TRUE(plain);
  auto ungreedy = kg::BuildGemma4Graph(*plain, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(ungreedy);
  EXPECT_EQ(ungreedy->greedy, nullptr);
  EXPECT_EQ(g->nodes.size(), ungreedy->nodes.size() + 1);
  // Greedy needs head outputs: refused before any allocation otherwise.
  auto refused = kg::TensorArena::Create(1);
  ASSERT_TRUE(refused);
  const auto used = refused->used();
  shape.output_mode = kg::Gemma4OutputMode::kStateOnly;
  shape.outputs = 0;
  EXPECT_FALSE(kg::BuildGemma4Graph(*refused, c.p, c.binding, c.state, shape));
  shape = c.shape;
  shape.greedy = true;
  shape.feature_outputs = 1;
  EXPECT_FALSE(kg::BuildGemma4Graph(*refused, c.p, c.binding, c.state, shape));
  EXPECT_EQ(refused->used(), used);
}
TEST(Gemma4Graph, PublicShapeAndBindingRefusalsPrecedeEveryGgmlAllocation) {
  Case c;
  auto arena = kg::TensorArena::Create(1);
  ASSERT_TRUE(arena);
  const auto used = arena->used();
  auto bad = c.shape;
  bad.segments[0].rows = 0;
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, bad));
  bad = c.shape;
  bad.segments.push_back(bad.segments[0]);
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, bad));
  bad = c.shape;
  bad.segments[0].global_n_kv = UINT32_MAX;
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, bad));
  bad = c.shape;
  bad.outputs = 3;
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, bad));
  auto binding = c.binding;
  binding.layers[0].q.ne[0] = UINT64_MAX;
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, binding, c.state, c.shape));
  binding = c.binding;
  binding.layers[0].q.ne.resize(8, 1);
  EXPECT_FALSE(md::CheckGemma4Binding(c.p, binding));
  binding = c.binding;
  binding.layers[0].q.type = std::string(4096, 'X');
  EXPECT_FALSE(md::CheckGemma4Binding(c.p, binding));
  binding = c.binding;
  binding.layers[0].q.index = UINT32_MAX;
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, binding, c.state, c.shape));
  binding = c.binding;
  binding.layers[0].q.index = binding.layers[0].k.index;
  EXPECT_FALSE(md::CheckGemma4Binding(c.p, binding));
  binding = c.binding;
  binding.layers[0].down_exps->index = UINT32_MAX;
  EXPECT_FALSE(md::CheckGemma4Binding(c.p, binding));
  binding = c.binding;
  binding.layers[0].down_exps->readable = 1;
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, binding, c.state, c.shape));
  kg::Gemma4GraphOptions options;
  options.expert_stride.resize(c.p.layers * 3 + 1);
  EXPECT_FALSE(kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options));
  Case dense(31);
  options.expert_stride.resize(1);
  EXPECT_FALSE(
      kg::BuildGemma4Graph(*arena, dense.p, dense.binding, dense.state, dense.shape, options));
  EXPECT_EQ(arena->used(), used);
}
TEST(Gemma4Graph, PositionsDoNotCreatePerTokenCacheKeys) {
  Case c;
  auto next = c.shape;
  next.segments[0].n_past = 4;
  EXPECT_EQ(c.shape, next);
  next.segments[0].global_n_kv += 256;
  EXPECT_NE(c.shape, next);
  // Whole local ring wraps still have one reusable shape while globals stay
  // inside the same padded read-width interval.
  auto a = c.shape;
  a.segments[0] = {0, 2, 2559, 2816, 1280};
  auto b = a;
  b.segments[0].n_past = 2561;
  EXPECT_EQ(a, b);
}
void BindLeaves(kg::Gemma4Graph& g) {
  std::uint64_t address = std::uint64_t{1} << 36U;
  const auto bind = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, address);
    address += ((ggml_nbytes(t) + 255) / 256 + 1) * 256;
  };
  for (auto* t : g.inputs) bind(t);
  for (const auto& w : g.weights) bind(w.tensor);
  for (const auto& seg : g.segments)
    for (const auto [k, v] : seg.caches) {
      if (k != nullptr) {
        bind(k);
        bind(v);
      }
    }
  kg::BindDistinct(g.nodes, std::uint64_t{1} << 48U);
}
TEST(Gemma4Graph, EveryActualQuantProductDefaultStridePassesOperandChecks) {
  for (const auto size : {26U, 31U}) {
    Case c(size, 4);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 4));
    ASSERT_TRUE(arena);
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
    ASSERT_TRUE(g);
    BindLeaves(*g);
    for (const auto& w : g->weights) {
      if (!w.resource.expert_array) continue;
      EXPECT_EQ(w.tensor->nb[2] % 16, 0U);
      EXPECT_EQ(w.tensor->nb[2] % ggml_type_size(w.tensor->type), 0U);
      EXPECT_GE(w.tensor->nb[2], w.resource.readable);
    }
    std::size_t checked = 0;
    for (const auto* t : g->nodes) {
      if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) continue;
      if (!ggml_is_quantized(t->src[0]->type)) continue;
      const auto valid = t->op == GGML_OP_MUL_MAT_ID ? kg::CheckMulMatIdQ(t) : kg::CheckMulMatQ(t);
      EXPECT_TRUE(valid) << (valid ? "" : valid.error().detail);
      ++checked;
    }
    EXPECT_GT(checked, c.p.layers * 6);
  }
}
TEST(Gemma4Graph, DecodePreparationSharesInputsAndKeepsQuantOneTokenArithmetic) {
  Case c(26, 4);
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 4));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape,
                                {.expert_stride = {}, .shared_q8 = true});
  ASSERT_TRUE(g);
  BindLeaves(*g);
  std::size_t products = 0, prepared = 0;
  for (const auto* t : g->nodes) {
    if (kg::LlmpOpOf(t) == kg::LlmpOp::kVecQ) {
      const auto valid = kg::CheckVecQ(t);
      EXPECT_TRUE(valid) << (valid ? "" : valid.error().detail);
      EXPECT_TRUE(kg::VecQOneToken(t));
      ++products;
    }
    if (kg::LlmpOpOf(t) == kg::LlmpOp::kQuantizeQ8) {
      const auto valid = kg::CheckQuantizeQ8(t);
      EXPECT_TRUE(valid) << (valid ? "" : valid.error().detail);
      ++prepared;
    }
  }
  EXPECT_GT(products, prepared);
  auto* input = g->Named("blk.0.attn_input");
  std::size_t consumers = 0;
  ggml_tensor* q8 = nullptr;
  for (auto* t : g->nodes) {
    if (kg::LlmpOpOf(t) != kg::LlmpOp::kQuantizeQ8 || t->src[0] != input) continue;
    EXPECT_EQ(q8, nullptr);
    q8 = t;
  }
  ASSERT_NE(q8, nullptr);
  for (const auto* t : g->nodes) {
    if (kg::LlmpOpOf(t) == kg::LlmpOp::kVecQ && t->src[1] == q8) ++consumers;
  }
  EXPECT_EQ(consumers, 3U);
}
TEST(Gemma4Graph, FrequencyFactorsRefuseMalformedBoundsAliasesAndStaleViews) {
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  std::uint64_t address = std::uint64_t{1} << 36U;
  const auto bind = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, address);
    address += ((ggml_nbytes(t) + 255) / 256 + 1) * 256;
    return t;
  };
  auto* x = bind(ggml_new_tensor_3d(c, GGML_TYPE_F32, 512, 2, 3));
  auto* positions = bind(ggml_new_tensor_1d(c, GGML_TYPE_I32, 3));
  auto* factors = bind(ggml_new_tensor_1d(c, GGML_TYPE_F32, 256));
  auto* rope = bind(ggml_rope_ext(c, x, positions, factors, 512, GGML_ROPE_TYPE_NEOX, 262144,
                                  1000000, 1, 0, 1, 32, 1));
  ASSERT_TRUE(kg::CheckRope(rope));
  ASSERT_TRUE(kg::CheckRopeExt(rope));
  factors->ne[0] = 255;
  EXPECT_FALSE(kg::CheckRope(rope));
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  factors->ne[0] = 256;
  factors->nb[0] = 8;
  EXPECT_FALSE(kg::CheckRope(rope));
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  factors->nb[0] = 4;
  factors->type = GGML_TYPE_F16;
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  factors->type = GGML_TYPE_F32;
  const auto output = reinterpret_cast<std::uintptr_t>(rope->data);
  const auto factor_address = reinterpret_cast<std::uintptr_t>(factors->data);
  kg::TensorArena::Bind(rope, reinterpret_cast<std::uintptr_t>(factors->data));
  EXPECT_FALSE(kg::CheckRope(rope));
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  kg::TensorArena::Bind(rope, output);
  auto* view = ggml_view_1d(c, factors, 256, 0);
  rope->src[2] = view;
  EXPECT_TRUE(kg::CheckRope(rope));
  factors->ne[0] = 255;
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  factors->ne[0] = 256;
  kg::TensorArena::Bind(factors, address + 256);
  EXPECT_FALSE(kg::CheckRope(rope));
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  rope->src[2] = factors;
  kg::TensorArena::Bind(factors, factor_address);
  rope->op_params[2] = GGML_ROPE_TYPE_NORMAL;
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  rope->op_params[2] = GGML_ROPE_TYPE_NEOX;
  rope->op = GGML_OP_ROPE_BACK;
  EXPECT_FALSE(kg::CheckRopeExt(rope));
  rope->op = GGML_OP_ROPE;
  auto* indices = bind(ggml_new_tensor_1d(c, GGML_TYPE_I64, 3));
  auto* cache = bind(ggml_new_tensor_2d(c, GGML_TYPE_F16, 1024, 256));
  auto* rows = ggml_view_2d(c, rope, 1024, 3, rope->nb[2], 0);
  auto* write = ggml_set_rows(c, cache, rows, indices);
  kg::BindViews(kg::GraphOrder(std::vector<ggml_tensor*>{write}));
  auto checked = kg::CheckRopeSetRows(rope, write);
  EXPECT_TRUE(checked) << (checked ? "" : checked.error().detail);
  kg::TensorArena::Bind(cache, reinterpret_cast<std::uintptr_t>(factors->data));
  kg::BindViews(kg::GraphOrder(std::vector<ggml_tensor*>{write}));
  EXPECT_FALSE(kg::CheckRopeSetRows(rope, write));
}
}  // namespace

TEST(Gemma4Graph, SmallRealOwnerGroupsPreserveProductsAndPreflightEveryWriter) {
  for (const auto size : {26U, 31U})
    for (const auto owners : {2U, 3U}) {
      Case c(size);
      c.state = *md::Gemma4State(c.p, 4096, size == 26 ? 1024 : 256);
      c.shape = {};
      for (std::uint32_t i = 0; i < owners; ++i)
        c.shape.segments.push_back({i + 3, 1, 384 + i, 512, 512});
      c.shape.outputs = owners;
      kg::Gemma4GraphOptions options;
      options.device_masks = true;
      options.narrow_final = true;
      auto estimate = options;
      estimate.attention_mode = kg::Gemma4AttentionMode::kOwners;
      const auto tensors = kg::Gemma4GraphTensors(c.p, owners, estimate);
      EXPECT_EQ(tensors, kg::Gemma4GraphTensors(c.p, owners) + c.p.layers * 64);
      auto arena = kg::TensorArena::Create(tensors);
      ASSERT_TRUE(arena);
      auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
      ASSERT_TRUE(graph);
      const auto products = Count(*graph, GGML_OP_MUL_MAT);
      auto* projection = graph->Named("blk.0.attn_projection");
      const auto before = graph->nodes;
      ggml_tensor* last = nullptr;
      for (auto* node : graph->nodes)
        if (node->op == GGML_OP_SET_ROWS &&
            node->src[2] == graph->segments.back().caches.back().second)
          last = node;
      ASSERT_NE(last, nullptr);
      auto* ids = last->src[1];
      last->src[1] = graph->segments.front().global_cells;
      const auto used = arena->used();
      EXPECT_FALSE(kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
      EXPECT_EQ(arena->used(), used);
      EXPECT_EQ(graph->nodes, before);
      EXPECT_EQ(graph->attention_mode, kg::Gemma4AttentionMode::kIndependent);
      last->src[1] = ids;
      ASSERT_TRUE(kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
      EXPECT_EQ(graph->attention_quad_mask, 1U);
      EXPECT_EQ(Count(*graph, GGML_OP_MUL_MAT), products);
      EXPECT_EQ(graph->Named("blk.0.attn_projection"), projection);
      EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 0U);
      EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), c.p.layers * owners * 2);
      std::size_t owner_nodes = 0;
      for (auto* node : graph->nodes) {
        if (kg::LlmpOpOf(node) != kg::LlmpOp::kFlashAttnOwners) continue;
        ++owner_nodes;
        EXPECT_EQ(node->ne[3], owners);
        EXPECT_EQ(kg::LlmpOpInt(node, 0), static_cast<std::int32_t>(owners));
        EXPECT_EQ(kg::LlmpOpInt(node, 1), static_cast<std::int32_t>(owners));
        for (std::size_t i = owners; i < 4; ++i) {
          EXPECT_EQ(node->src[2 + i], nullptr);
          EXPECT_EQ(node->src[6 + i], nullptr);
        }
      }
      EXPECT_EQ(owner_nodes, c.p.layers);
      BindLeaves(*graph);
      for (auto* node : graph->nodes)
        if (kg::LlmpOpOf(node) == kg::LlmpOp::kFlashAttnOwners) {
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(node));
          node->src[2 + owners] = node->src[2];
          EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(node));
          node->src[2 + owners] = nullptr;
          const std::int32_t invalid = 1;
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 36),
                      &invalid, sizeof(invalid));
          EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(node));
          const auto valid = static_cast<std::int32_t>(owners);
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 36),
                      &valid, sizeof(valid));
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(node));
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 40),
                      &invalid, sizeof(invalid));
          EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(node));
          const std::int32_t zero = 0;
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 40), &zero,
                      sizeof(zero));
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(node));
        }
      c.shape.segments.back().global_n_kv = 768;
      auto fallback_arena = kg::TensorArena::Create(tensors);
      ASSERT_TRUE(fallback_arena);
      auto fallback =
          kg::BuildGemma4Graph(*fallback_arena, c.p, c.binding, c.state, c.shape, estimate);
      ASSERT_TRUE(fallback);
      EXPECT_EQ(fallback->attention_mode, kg::Gemma4AttentionMode::kIndependent);
      EXPECT_EQ(Count(*fallback, GGML_OP_FLASH_ATTN_EXT), c.p.layers * owners);
    }
}

TEST(Gemma4Graph, PartialOwnerGroupsPreserveProductsAndPreflightEveryWriter) {
  for (const auto size : {26U, 31U})
    for (const auto owners : {5U, 6U, 7U, 9U, 10U, 11U}) {
      Case c(size);
      c.state = *md::Gemma4State(c.p, 4096, size == 26 ? 1024 : 256);
      c.shape = {};
      for (std::uint32_t i = 0; i < owners; ++i)
        c.shape.segments.push_back({i + 3, 1, 384 + i, 512, 512});
      c.shape.outputs = owners;
      kg::Gemma4GraphOptions options;
      options.device_masks = true;
      options.narrow_final = true;
      auto estimate = options;
      estimate.attention_mode = kg::Gemma4AttentionMode::kOwners;
      const auto tensors = kg::Gemma4GraphTensors(c.p, owners, estimate);
      EXPECT_EQ(tensors,
                kg::Gemma4GraphTensors(c.p, owners) + c.p.layers * 64 * ((owners + 3) / 4));
      auto arena = kg::TensorArena::Create(tensors);
      ASSERT_TRUE(arena);
      auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
      ASSERT_TRUE(graph);
      const auto products = Count(*graph, GGML_OP_MUL_MAT);
      auto* projection = graph->Named("blk.0.attn_projection");
      const auto before = graph->nodes;
      ggml_tensor* last = nullptr;
      for (auto* node : graph->nodes)
        if (node->op == GGML_OP_SET_ROWS &&
            node->src[2] == graph->segments.back().caches.back().second)
          last = node;
      ASSERT_NE(last, nullptr);
      auto* ids = last->src[1];
      last->src[1] = graph->segments.front().global_cells;
      const auto used = arena->used();
      EXPECT_FALSE(kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
      EXPECT_EQ(arena->used(), used);
      EXPECT_EQ(graph->nodes, before);
      EXPECT_EQ(graph->attention_mode, kg::Gemma4AttentionMode::kIndependent);
      last->src[1] = ids;
      ASSERT_TRUE(kg::TransformGemma4Attention(*arena, *graph, kg::Gemma4AttentionMode::kOwners));
      EXPECT_EQ(graph->attention_quad_mask, (1U << ((owners + 3) / 4)) - 1);
      EXPECT_EQ(Count(*graph, GGML_OP_MUL_MAT), products);
      EXPECT_EQ(graph->Named("blk.0.attn_projection"), projection);
      EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 0U);
      EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), c.p.layers * owners * 2);
      std::size_t owner_nodes = 0;
      for (auto* node : graph->nodes) {
        if (kg::LlmpOpOf(node) != kg::LlmpOp::kFlashAttnOwners) continue;
        ++owner_nodes;
        const auto offset = static_cast<std::uint32_t>(kg::LlmpOpInt(node, 2));
        const auto active = std::min(4U, owners - offset);
        EXPECT_EQ(node->ne[3], active);
        EXPECT_EQ(offset % 4, 0U);
        EXPECT_EQ(kg::LlmpOpInt(node, 0), static_cast<std::int32_t>(owners));
        EXPECT_EQ(kg::LlmpOpInt(node, 1), active == 4 ? 0 : static_cast<std::int32_t>(active));
        for (std::size_t i = active; i < 4; ++i) {
          EXPECT_EQ(node->src[2 + i], nullptr);
          EXPECT_EQ(node->src[6 + i], nullptr);
        }
      }
      EXPECT_EQ(owner_nodes, c.p.layers * ((owners + 3) / 4));
      BindLeaves(*graph);
      for (auto* node : graph->nodes)
        if (kg::LlmpOpOf(node) == kg::LlmpOp::kFlashAttnOwners) {
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(node));
          const auto offset = kg::LlmpOpInt(node, 2);
          const auto active = std::min(4U, owners - static_cast<std::uint32_t>(offset));
          if (active < 4) {
            node->src[2 + active] = node->src[2];
            EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(node));
            node->src[2 + active] = nullptr;
          }
          const std::int32_t invalid = 1;
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 40),
                      &invalid, sizeof(invalid));
          EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(node));
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 40),
                      &offset, sizeof(offset));
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(node));
          const std::int32_t extra = 7;
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 44),
                      &extra, sizeof(extra));
          EXPECT_FALSE(kg::CheckFlashAttnOwnersNode(node));
          const std::int32_t zero = 0;
          std::memcpy(static_cast<void*>(reinterpret_cast<std::byte*>(node->op_params) + 44), &zero,
                      sizeof(zero));
          EXPECT_TRUE(kg::CheckFlashAttnOwnersNode(node));
        }
      c.shape.segments.back().global_n_kv = 768;
      auto fallback_arena = kg::TensorArena::Create(tensors);
      ASSERT_TRUE(fallback_arena);
      auto fallback =
          kg::BuildGemma4Graph(*fallback_arena, c.p, c.binding, c.state, c.shape, estimate);
      ASSERT_TRUE(fallback);
      EXPECT_EQ(fallback->attention_mode, kg::Gemma4AttentionMode::kOwners);
      EXPECT_EQ(Count(*fallback, GGML_OP_FLASH_ATTN_EXT), c.p.layers * (owners % 4));
      for (const auto* node : fallback->nodes)
        if (kg::LlmpOpOf(node) == kg::LlmpOp::kFlashAttnOwners) {
          EXPECT_EQ(kg::LlmpOpInt(node, 0), 4);
          EXPECT_EQ(kg::LlmpOpInt(node, 2), 0);
        }
    }
}

TEST(Gemma4Graph, SmallPhysicalStreamPartitionUsesWholeOriginalGridWithoutSplitting) {
  for (const auto owners : {2U, 3U})
    for (const auto maximum : {1, 47, 48, 96, 100})
      for (const auto kvheads : {2, 4, 8, 16}) {
        constexpr int kvtiles = 16;
        const auto tiles = kvheads * static_cast<int>(owners);
        const auto raw = std::min(maximum, kvtiles * tiles);
        const auto rounded = raw / tiles * tiles;
        const auto loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
        auto plan = kg::detail::PlanOwnerPartition(maximum, kvtiles, kvheads, owners);
        ASSERT_TRUE(plan);
        EXPECT_EQ(plan->effective_cohort, owners);
        EXPECT_EQ(plan->cohort_blocks, loss <= 5 ? rounded : raw);
        EXPECT_EQ(plan->quad_blocks, plan->cohort_blocks);
      }
}

TEST(Gemma4Graph, PartialPhysicalStreamPartitionRetainsWholeGridWithoutDivision) {
  for (const auto owners : {5U, 6U, 7U, 9U, 10U, 11U})
    for (const auto maximum : {1, 47, 48, 96, 100})
      for (const auto kvheads : {2, 4, 8, 16}) {
        constexpr int kvtiles = 16;
        const auto tiles = kvheads * static_cast<int>(owners);
        const auto raw = std::min(maximum, kvtiles * tiles);
        const auto rounded = raw / tiles * tiles;
        const auto loss = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
        auto plan = kg::detail::PlanOwnerPartition(maximum, kvtiles, kvheads, owners);
        ASSERT_TRUE(plan);
        EXPECT_EQ(plan->effective_cohort, owners);
        EXPECT_EQ(plan->cohort_blocks, loss <= 5 ? rounded : raw);
        EXPECT_EQ(plan->quad_blocks, plan->cohort_blocks);
        // Every original block interval is covered exactly once by the
        // canonical owner groups. New clipping endpoints are full tiles.
        const auto sequence_work = std::int64_t{kvtiles} * kvheads;
        const auto work = sequence_work * owners;
        for (std::int64_t block = 0; block < plan->cohort_blocks; ++block) {
          const auto start = block * work / plan->cohort_blocks;
          const auto stop = (block + 1) * work / plan->cohort_blocks;
          std::int64_t covered = 0;
          for (std::uint32_t first = 0; first < owners; first += 4) {
            const auto end = std::min(first + 4, owners);
            const auto local_start = std::max(start, first * sequence_work);
            const auto local_stop = std::min(stop, end * sequence_work);
            if (local_start >= local_stop) continue;
            if (local_start != start) EXPECT_EQ(local_start % kvtiles, 0);
            if (local_stop != stop) EXPECT_EQ(local_stop % kvtiles, 0);
            covered += local_stop - local_start;
          }
          EXPECT_EQ(covered, stop - start);
        }
      }
}

TEST(Gemma4Graph, OriginalDenseSharingUsesTransientChoicesAndPreservesOtherConsumers) {
  for (const auto size : {26U, 31U}) {
    for (std::uint32_t columns = 1; columns <= 9; ++columns) {
      Case c(size);
      std::vector<std::int32_t> tokens(columns, 2);
      std::vector<md::Gemma4Segment> segments;
      segments.push_back({0, 0, std::span(tokens).first(1)});
      if (columns > 1) segments.push_back({1, 1100, std::span(tokens).subspan(1)});
      auto in = md::Gemma4Chunk(c.p, c.state, segments);
      ASSERT_TRUE(in);
      c.shape.segments.clear();
      for (const auto& s : in->segments)
        c.shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
      c.shape.outputs = static_cast<std::uint32_t>(segments.size());
      for (const auto choice : {0U, 1U, 2U}) {
        kg::Gemma4GraphOptions options{.expert_stride = {}, .dense_shared_q8 = true};
        std::function<bool(ggml_type, std::int64_t)> selected;
        if (choice != 0)
          selected = [choice](ggml_type, std::int64_t n) { return choice == 2 && n <= 8; };
        auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, segments.size(), options));
        ASSERT_TRUE(arena);
        auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options, selected);
        ASSERT_TRUE(g);
        BindLeaves(*g);
        std::size_t preparations = 0, products = 0, old_vecq = 0;
        for (const auto* node : g->nodes) {
          const auto op = kg::LlmpOpOf(node);
          preparations += op == kg::LlmpOp::kQuantizeQ8;
          old_vecq += op == kg::LlmpOp::kVecQ;
          if (op != kg::LlmpOp::kMmvqPrepared) continue;
          const auto valid = kg::CheckMmvqPrepared(node);
          EXPECT_TRUE(valid) << (valid ? "" : valid.error().detail);
          EXPECT_EQ(node->src[2]->ne[1], columns);
          ++products;
        }
        EXPECT_EQ(old_vecq, 0U);
        if (choice != 2 || columns > 8) {
          EXPECT_EQ(preparations, 0U);
          EXPECT_EQ(products, 0U);
        } else {
          EXPECT_GT(preparations, 0U);
          EXPECT_GT(products, preparations);
          auto* attn_input = g->Named("blk.0.attn_input");
          ASSERT_NE(attn_input, nullptr);
          std::size_t q8 = 0;
          for (const auto* node : g->nodes)
            q8 += kg::LlmpOpOf(node) == kg::LlmpOp::kQuantizeQ8 && node->src[0] == attn_input;
          EXPECT_EQ(q8, 1U);
        }
        // Whole immutable resources for routed experts and single-consumer
        // attention/down/head products never enter the new preparation path.
        for (const auto* node : g->nodes) {
          if (kg::LlmpOpOf(node) != kg::LlmpOp::kMmvqPrepared) continue;
          const auto leaf = std::ranges::find_if(
              g->weights, [&](const auto& w) { return w.tensor == node->src[0]; });
          ASSERT_NE(leaf, g->weights.end());
          EXPECT_FALSE(leaf->resource.expert_array);
          EXPECT_NE(leaf->resource.index, c.binding.output.index);
          for (const auto& layer : c.binding.layers) {
            EXPECT_NE(leaf->resource.index, layer.out.index);
            EXPECT_NE(leaf->resource.index, layer.down.index);
            if (layer.router) EXPECT_NE(leaf->resource.index, layer.router->index);
          }
        }
        if (columns == 1) {
          // The ordinary separate gate/up still forms the original GeGLU pattern.
          EXPECT_GT(Count(*g, GGML_OP_GLU), 0U);
          std::size_t ordinary = 0;
          for (const auto* node : g->nodes)
            ordinary += node->op == GGML_OP_MUL_MAT && node->src[0]->type != GGML_TYPE_F32;
          EXPECT_GE(ordinary, c.p.layers * 3U);
        }
      }
    }
  }
}

TEST(Gemma4Graph, OriginalDenseSharingKeepsHistoricalOverloadAndDiagnosticPrecedence) {
  Case c(26, 2);
  for (const auto old : {false, true}) {
    kg::Gemma4GraphOptions options{.expert_stride = {}, .shared_q8 = old, .dense_shared_q8 = true};
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2, options));
    ASSERT_TRUE(arena);
    // Old overload deliberately provides no transient device selector.
    auto historical = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
    ASSERT_TRUE(historical);
    std::size_t prepared = 0, vecq = 0;
    for (const auto* node : historical->nodes) {
      prepared += kg::LlmpOpOf(node) == kg::LlmpOp::kMmvqPrepared;
      vecq += kg::LlmpOpOf(node) == kg::LlmpOp::kVecQ;
    }
    EXPECT_EQ(prepared, 0U);
    EXPECT_EQ(vecq > 0, old);
    if (old) {
      auto next = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2, options));
      ASSERT_TRUE(next);
      auto graph = kg::BuildGemma4Graph(*next, c.p, c.binding, c.state, c.shape, options,
                                        [](ggml_type, std::int64_t) { return true; });
      ASSERT_TRUE(graph);
      std::size_t new_prepared = 0, new_vecq = 0;
      for (const auto* node : graph->nodes) {
        new_prepared += kg::LlmpOpOf(node) == kg::LlmpOp::kMmvqPrepared;
        new_vecq += kg::LlmpOpOf(node) == kg::LlmpOp::kVecQ;
      }
      EXPECT_EQ(new_prepared, 0U);
      EXPECT_EQ(new_vecq, vecq);
    }
  }
}

TEST(Gemma4Graph, OriginalDenseSharingKeepsStateOnlyTiedKvSingletonOrdinary) {
  for (const auto size : {26U, 31U}) {
    Case c(size, 2);
    ASSERT_TRUE(c.binding.layers.back().tied_kv);
    c.shape.output_mode = kg::Gemma4OutputMode::kStateOnly;
    c.shape.outputs = 0;
    kg::Gemma4GraphOptions options{.expert_stride = {}, .dense_shared_q8 = true};
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2, options));
    ASSERT_TRUE(arena);
    auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options,
                                      [](ggml_type, std::int64_t) { return true; });
    ASSERT_TRUE(graph);
    auto* input = graph->Named("blk." + std::to_string(c.p.layers - 1) + ".attn_input");
    ASSERT_NE(input, nullptr);
    std::size_t preparations = 0, products = 0;
    for (const auto* node : graph->nodes) {
      preparations += kg::LlmpOpOf(node) == kg::LlmpOp::kQuantizeQ8 && node->src[0] == input;
      products += node->op == GGML_OP_MUL_MAT && node->src[1] == input;
    }
    EXPECT_EQ(preparations, 0U);
    EXPECT_EQ(products, 1U);
  }
}
