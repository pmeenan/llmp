// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma3_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "expected_error.h"
#include "gemma3_fixture.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/jitllm_ops.h"

namespace {
namespace md = jitllm::model;
namespace kg = jitllm::kernels::ggml;
struct Case {
  const md::Gemma3Profile& p = md::Gemma3_4BQat();
  md::Gemma3Binding binding =
      *md::BindGemma3(p, "gemma3", jitllm::test_support::gemma3::Resources());
  md::Gemma3StateLayout state = *md::Gemma3State(p, 4096, 16);
  kg::Gemma3ChunkShape shape{{{3, 3, 1279, 1536, 1280}, {1, 1, 7, 256, 256}}, 2};
};
std::size_t Count(const kg::Gemma3Graph& graph, ggml_op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(graph.nodes, [op](const auto* t) { return t->op == op; }));
}

TEST(Gemma3Graph, H8OwnerContractIsClosedToD256AndTwoOrFourActualRoots) {
  const auto make = [](kg::TensorArena& arena, std::int64_t d, std::int64_t heads,
                       std::uint32_t owners) {
    auto* c = arena.context();
    std::uintptr_t address = std::uintptr_t{1} << 40U;
    const auto placed = [&](ggml_tensor* tensor) {
      kg::TensorArena::Bind(tensor, address);
      address += 64U << 20U;
      return tensor;
    };
    auto* raw_q = placed(ggml_new_tensor_4d(c, GGML_TYPE_F32, d, heads, 1, owners));
    auto* q = ggml_permute(c, raw_q, 0, 2, 1, 3);
    kg::TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
    kg::FlashAttnOwners in{
        .q = q,
        .mask = placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, 512, 32, 1, owners)),
        .output = placed(ggml_new_tensor_4d(c, GGML_TYPE_F32, d, heads, 1, owners)),
        .logical_cohort = owners,
        .owner_count = owners};
    for (std::uint32_t i = 0; i < owners; ++i) {
      for (auto* destination : {&in.k[i], &in.v[i]}) {
        auto* raw =
            placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, d, heads / (d == 256 ? 2 : 8), 512, 1));
        auto* view = ggml_permute(c, raw, 0, 2, 1, 3);
        kg::TensorArena::Bind(view, reinterpret_cast<std::uintptr_t>(raw->data));
        *destination = view;
      }
    }
    return in;
  };
  for (const auto owners : {2U, 3U, 4U}) {
    auto arena = kg::TensorArena::Create(128);
    ASSERT_TRUE(arena);
    const auto in = make(*arena, 256, 8, owners);
    EXPECT_EQ(kg::CheckFlashAttnOwners(in).has_value(), owners == 2 || owners == 4);
    if (owners == 4) {
      auto invalid = in;
      invalid.logit_softcap = 50;
      EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
      invalid = in;
      invalid.bounded_roots = true;
      EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
      for (const auto cohort : {8U, 12U}) {
        invalid = in;
        invalid.logical_cohort = cohort;
        EXPECT_FALSE(kg::CheckFlashAttnOwners(invalid));
      }
    }
  }
  auto arena = kg::TensorArena::Create(128);
  ASSERT_TRUE(arena);
  EXPECT_FALSE(kg::CheckFlashAttnOwners(make(*arena, 512, 8, 2)));
  EXPECT_TRUE(kg::CheckFlashAttnOwners(make(*arena, 256, 16, 2)));
}

TEST(Gemma3Graph, BoundedOwnerRootsRejectMalformedActualViewsAndMetadata) {
  auto arena = kg::TensorArena::Create(96);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  std::uintptr_t address = std::uintptr_t{1} << 40U;
  const auto placed = [&](ggml_tensor* tensor) {
    kg::TensorArena::Bind(tensor, address);
    address += 64U << 20U;
    return tensor;
  };
  auto* raw_q = placed(ggml_new_tensor_4d(c, GGML_TYPE_F32, 256, 8, 1, 2));
  auto* q = ggml_permute(c, raw_q, 0, 2, 1, 3);
  kg::TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
  auto* mask = placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, 1024, 32, 1, 2));
  std::array<ggml_tensor*, 4> k{}, v{};
  for (std::size_t owner = 0; owner < 2; ++owner)
    for (auto* target : {&k[owner], &v[owner]}) {
      auto* raw = placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, 256, 4, owner ? 1024 : 512, 1));
      *target = ggml_permute(c, raw, 0, 2, 1, 3);
      kg::TensorArena::Bind(*target, reinterpret_cast<std::uintptr_t>(raw->data));
    }
  auto* node = placed(kg::FlashAttnOwnersNode(c, q, mask, k, v, 2, 2, 0, 0, true));
  auto in = kg::FlashAttnOwnersFromNode(node);
  ASSERT_TRUE(in);
  EXPECT_TRUE(in->bounded_roots);
  const auto check = [&] { return kg::CheckFlashAttnOwnersNode(node).has_value(); };
  EXPECT_TRUE(check());
  const auto original = *k[0];
  k[0]->ne[1] = 511;
  EXPECT_FALSE(check());
  *k[0] = original;
  k[0]->nb[2] *= 2;
  EXPECT_FALSE(check());
  *k[0] = original;
  v[0]->ne[1] = 1024;
  EXPECT_FALSE(check());
  v[0]->ne[1] = 512;
  auto bad = *in;
  bad.bounded_roots = false;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));
  bad = *in;
  bad.logit_softcap = 25;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));
  bad = *in;
  bad.k[1] = bad.k[0];
  bad.v[1] = bad.v[0];
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));  // no real root covers logical1024
  bad = *in;
  bad.owner_count = bad.logical_cohort = 3;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));
  EXPECT_TRUE(check());
  Case g;
  EXPECT_FALSE(kg::CheckGemma3Graph(g.p, g.binding, g.state, g.shape, {.bounded_roots = true}));
}

TEST(Gemma3Graph, ExplicitC2OwnerDecodeViewsQAndJoinsMasksWithoutJoiningCacheRoots) {
  Case c;
  c.shape = {{{3, 1, 259, 512, 512}, {1, 1, 259, 512, 512}}, 2};
  kg::Gemma3GraphOptions options{.narrow_final = true, .owner_decode = true};
  auto arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma3Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 0);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 136);
  std::size_t owners = 0;
  for (const auto* node : graph->nodes) {
    if (kg::JitllmOpOf(node) != kg::JitllmOp::kFlashAttnOwners) continue;
    ++owners;
    EXPECT_EQ(node->src[0]->op, GGML_OP_PERMUTE);
    EXPECT_EQ(node->src[0]->ne[0], 256);
    EXPECT_EQ(node->src[0]->ne[2], 8);
    EXPECT_EQ(node->src[0]->ne[3], 2);
    EXPECT_EQ(node->src[1]->op, GGML_OP_CONCAT);
    EXPECT_NE(node->src[2]->view_src, node->src[3]->view_src);
    EXPECT_NE(node->src[6]->view_src, node->src[7]->view_src);
    EXPECT_EQ(kg::JitllmOpInt(node, 0), 2);
    EXPECT_EQ(kg::JitllmOpInt(node, 1), 2);
  }
  EXPECT_EQ(owners, 34);
  options.first_layer = 1;
  EXPECT_FALSE(kg::CheckGemma3Graph(c.p, c.binding, c.state, c.shape, options));
  options.first_layer = 0;
  c.shape.segments[0].rows = 2;
  auto ordinary_arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 2));
  ASSERT_TRUE(ordinary_arena);
  const auto ordinary =
      kg::BuildGemma3Graph(*ordinary_arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(ordinary);
  EXPECT_EQ(Count(*ordinary, GGML_OP_FLASH_ATTN_EXT), 68);
}

TEST(Gemma3Graph, FourOwnerDecodeKeepsIndependentRootsAndPairPrefillCapacity) {
  Case c;
  c.state = *md::Gemma3State(c.p, 4096, 128);
  c.shape = {
      {{0, 1, 259, 512, 512}, {1, 1, 259, 512, 512}, {2, 1, 259, 512, 512}, {3, 1, 259, 512, 512}},
      4};
  kg::Gemma3GraphOptions options{.max_total_rows = 256,
                                 .narrow_final = true,
                                 .owner_decode = true,
                                 .packed_prefill = true,
                                 .bounded_roots = true};
  auto arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 4));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma3Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 0U);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 272U);
  std::size_t owners = 0;
  for (const auto* node : graph->nodes) {
    if (kg::JitllmOpOf(node) != kg::JitllmOp::kFlashAttnOwners) continue;
    ++owners;
    EXPECT_EQ(node->src[0]->ne[3], 4);
    EXPECT_EQ(node->src[1]->ne[3], 4);
    EXPECT_EQ(kg::JitllmOpInt(node, 0), 4);
    EXPECT_EQ(kg::JitllmOpInt(node, 1), 0);  // Canonical four-root encoding.
    EXPECT_EQ(kg::JitllmOpInt(node, 3), 0);
    EXPECT_EQ(kg::JitllmOpInt(node, 4), 0);  // C4 never requests bounded-root specialization.
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j) {
        EXPECT_NE(node->src[2 + i]->view_src, node->src[2 + j]->view_src);
        EXPECT_NE(node->src[6 + i]->view_src, node->src[6 + j]->view_src);
      }
  }
  EXPECT_EQ(owners, 34U);
  c.shape.segments[0].rows = 2;
  auto ragged_arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 4));
  ASSERT_TRUE(ragged_arena);
  auto ragged = kg::BuildGemma3Graph(*ragged_arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(ragged);
  EXPECT_EQ(Count(*ragged, GGML_OP_FLASH_ATTN_EXT), 136U);
}

TEST(Gemma3Graph, ActualArithmeticUsesRawVDirectNormAndPerLayerRopeThenQueryScale) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma3Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->weights.size(), 444);
  EXPECT_EQ(graph->logits->ne[0], c.p.vocab);
  EXPECT_EQ(graph->logits->ne[1], 2);
  EXPECT_EQ(graph->hidden->ne[1], 4);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 68);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 136);
  EXPECT_EQ(Count(*graph, GGML_OP_GLU), 34);
  EXPECT_EQ(Count(*graph, GGML_OP_SOFT_MAX), 0);
  for (std::uint32_t il = 0; il < c.p.layers; ++il) {
    const auto prefix = "blk." + std::to_string(il) + '.';
    auto* q = graph->Named(prefix + "q_rope");
    auto* scaled = graph->Named(prefix + "q_scaled");
    auto* k = graph->Named(prefix + "k_rope");
    auto* v = graph->Named(prefix + "v_raw");
    ASSERT_NE(q, nullptr);
    ASSERT_NE(scaled, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(scaled->src[0], q);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(scaled->op_params[0]), 0.0625F);
    for (const auto* rope : {q, k}) {
      EXPECT_EQ(rope->op, GGML_OP_ROPE);
      EXPECT_EQ(rope->op_params[1], 256);
      EXPECT_EQ(rope->op_params[2], GGML_ROPE_TYPE_NEOX);
      EXPECT_FLOAT_EQ(std::bit_cast<float>(rope->op_params[5]),
                      c.p.local(il) ? 10000.0F : 1000000.0F);
      EXPECT_FLOAT_EQ(std::bit_cast<float>(rope->op_params[6]), c.p.local(il) ? 1.0F : 0.125F);
      EXPECT_EQ(rope->src[2], nullptr);
      ASSERT_EQ(rope->src[0]->op, GGML_OP_MUL);
      EXPECT_EQ(rope->src[0]->src[0]->op, GGML_OP_RMS_NORM);
      EXPECT_EQ(rope->src[0]->src[1]->op, GGML_OP_NONE);
    }
    ASSERT_EQ(v->op, GGML_OP_RESHAPE);
    EXPECT_EQ(v->src[0]->op, GGML_OP_MUL_MAT);
    EXPECT_NE(graph->segments[0].caches[il].first, graph->segments[0].caches[il].second);
    auto* attention = graph->Named(prefix + "slot.3.attention")->src[0];
    ASSERT_EQ(attention->op, GGML_OP_FLASH_ATTN_EXT);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[0]), 1.0F);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[2]), 0.0F);
    EXPECT_EQ(attention->src[0]->ne[2], 8);
    EXPECT_EQ(attention->src[1]->ne[2], 4);
    EXPECT_EQ(attention->src[3],
              c.p.local(il) ? graph->segments[0].local_mask : graph->segments[0].global_mask);
  }
  EXPECT_EQ(graph->Named("blk.33.k_rope")->op, GGML_OP_ROPE);
}

TEST(Gemma3Graph, StateOnlyRetainsAllWritesAndNarrowingKeepsFullFinalAttention) {
  Case c;
  c.shape.outputs = 0;
  c.shape.output_mode = kg::Gemma3OutputMode::kStateOnly;
  auto arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma3Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->hidden, nullptr);
  EXPECT_EQ(graph->logits, nullptr);
  EXPECT_EQ(graph->Named("blk.33.q_rope"), nullptr);
  EXPECT_EQ(graph->Named("blk.33.attn_projection"), nullptr);
  EXPECT_NE(graph->Named("blk.33.k_rope"), nullptr);
  EXPECT_NE(graph->Named("blk.33.v_raw"), nullptr);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 136);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 66);
  arena->Reset();
  c.shape.outputs = 2;
  c.shape.output_mode = kg::Gemma3OutputMode::kHead;
  graph = kg::BuildGemma3Graph(*arena, c.p, c.binding, c.state, c.shape, {.narrow_final = true});
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->Named("blk.33.q_rope")->ne[2], 4);
  EXPECT_EQ(graph->Named("blk.33.attn_projection")->ne[1], 4);
  EXPECT_EQ(graph->Named("blk.33.output")->ne[1], 2);
  EXPECT_EQ(graph->hidden->ne[1], 2);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 136);
}

TEST(Gemma3Graph, GreedyFrontiersHaveDistinctShapeAndKeptArgmaxDescriptor) {
  Case c;
  const auto full_head = c.shape;
  c.shape.greedy = true;
  EXPECT_NE(c.shape, full_head);
  auto arena = kg::TensorArena::Create(kg::Gemma3GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma3Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  ASSERT_NE(graph->greedy, nullptr);
  EXPECT_EQ(graph->greedy->type, GGML_TYPE_I32);
  EXPECT_EQ(graph->greedy->ne[0], 2);
  EXPECT_EQ(graph->greedy->src[0], graph->logits);
  EXPECT_EQ(kg::JitllmOpOf(graph->greedy), kg::JitllmOp::kArgmax);
  EXPECT_EQ(graph->nodes.back(), graph->greedy);
  auto malformed = c.shape;
  malformed.outputs = 4;
  EXPECT_FALSE(kg::CheckGemma3Graph(c.p, c.binding, c.state, malformed));
  malformed.outputs = 0;
  malformed.output_mode = kg::Gemma3OutputMode::kStateOnly;
  EXPECT_FALSE(kg::CheckGemma3Graph(c.p, c.binding, c.state, malformed));
  EXPECT_FALSE(kg::CheckGemma3Graph(c.p, c.binding, c.state, c.shape, {.head = false}));
}

TEST(Gemma3Graph, InvalidDescriptorsScopesAndCapacityRefuseBeforeGraphConstruction) {
  Case c;
  const auto rejects = [&](const auto& shape, const kg::Gemma3GraphOptions& options = {}) {
    EXPECT_FALSE(kg::CheckGemma3Graph(c.p, c.binding, c.state, shape, options));
  };
  auto changed = c.shape;
  changed.segments[1].slot = changed.segments[0].slot;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].global_n_kv = 256;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].local_n_kv = 1281;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].n_past = 4095;
  rejects(changed);
  rejects(c.shape, {.first_layer = 5});
  rejects(c.shape, {.layer_count = 1, .narrow_final = true});
  changed = c.shape;
  changed.outputs = 0;
  changed.output_mode = kg::Gemma3OutputMode::kStateOnly;
  rejects(changed, {.hidden_input = true});
  auto bad_binding = c.binding;
  bad_binding.layers[0].v.index = bad_binding.layers[0].k.index;
  EXPECT_FALSE(kg::CheckGemma3Graph(c.p, bad_binding, c.state, c.shape));
  auto bad_layout = c.state;
  --bad_layout.tensors[0].bytes;
  EXPECT_FALSE(kg::CheckGemma3Graph(c.p, c.binding, bad_layout, c.shape));
  auto small = kg::TensorArena::Create(1);
  ASSERT_TRUE(small);
  const auto before = small->used();
  EXPECT_FALSE(kg::BuildGemma3Graph(*small, c.p, c.binding, c.state, c.shape));
  EXPECT_EQ(small->used(), before);
  auto advanced = c.shape;
  ++advanced.segments[0].n_past;
  EXPECT_EQ(advanced, c.shape);  // Positions are fresh inputs at the same read widths.
  advanced.segments[0].global_n_kv += 256;
  EXPECT_NE(advanced, c.shape);
}
}  // namespace
