// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma2_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "expected_error.h"
#include "gemma2_fixture.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/validate_ext.h"

namespace {
namespace md = jitllm::model;
namespace kg = jitllm::kernels::ggml;
struct Case {
  const md::Gemma2Profile& p = md::Gemma2_2B();
  md::Gemma2Binding binding =
      *md::BindGemma2(p, "gemma2", jitllm::test_support::gemma2::Resources());
  md::Gemma2StateLayout state = *md::Gemma2State(p, 8192, 16);
  kg::Gemma2ChunkShape shape{{{3, 3, 4351, 4608, 4352}, {1, 1, 7, 256, 256}}, 2};
};
std::size_t Count(const kg::Gemma2Graph& graph, ggml_op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(graph.nodes, [op](const auto* t) { return t->op == op; }));
}

TEST(Gemma2Graph, ActualArithmeticHasNoQkNormAndPreservesBothSoftcapsAndReadableTails) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->weights.size(), 288);
  EXPECT_EQ(graph->logits->ne[0], c.p.vocab);
  EXPECT_EQ(graph->logits->ne[1], 2);
  EXPECT_EQ(graph->hidden->ne[1], 4);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 52);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
  EXPECT_EQ(Count(*graph, GGML_OP_GLU), 26);
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
      EXPECT_FLOAT_EQ(std::bit_cast<float>(rope->op_params[5]), 10000.0F);
      EXPECT_FLOAT_EQ(std::bit_cast<float>(rope->op_params[6]), 1.0F);
      EXPECT_EQ(rope->src[2], nullptr);
      ASSERT_EQ(rope->src[0]->op, GGML_OP_RESHAPE);
      EXPECT_EQ(rope->src[0]->src[0]->op, GGML_OP_MUL_MAT);
    }
    ASSERT_EQ(v->op, GGML_OP_RESHAPE);
    EXPECT_EQ(v->src[0]->op, GGML_OP_MUL_MAT);
    EXPECT_NE(graph->segments[0].caches[il].first, graph->segments[0].caches[il].second);
    auto* attention = graph->Named(prefix + "slot.3.attention")->src[0];
    ASSERT_EQ(attention->op, GGML_OP_FLASH_ATTN_EXT);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[0]), 1.0F);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[2]), 50.0F);
    EXPECT_EQ(attention->src[0]->ne[2], 8);
    EXPECT_EQ(attention->src[1]->ne[2], 4);
    EXPECT_EQ(attention->src[3],
              c.p.local(il) ? graph->segments[0].local_mask : graph->segments[0].global_mask);
  }
  EXPECT_EQ(graph->Named("blk.25.k_rope")->op, GGML_OP_ROPE);
  ASSERT_EQ(graph->logits->op, GGML_OP_SCALE);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(graph->logits->op_params[0]), 30.0F);
  ASSERT_EQ(graph->logits->src[0]->op, GGML_OP_UNARY);
  EXPECT_EQ(ggml_get_unary_op(graph->logits->src[0]), GGML_UNARY_OP_TANH);
  ASSERT_EQ(graph->logits->src[0]->src[0]->op, GGML_OP_SCALE);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(graph->logits->src[0]->src[0]->op_params[0]), 1.0F / 30.0F);
  std::size_t padded = 0;
  for (const auto& leaf : graph->weights) {
    if (ggml_is_quantized(leaf.tensor->type) && leaf.resource.ne[0] % 512 != 0) {
      ++padded;
      EXPECT_TRUE(kg::RowPaddingReadable(leaf.tensor));
      EXPECT_EQ(leaf.resource.readable, static_cast<std::uint64_t>(ggml_nbytes(leaf.tensor)) + 272);
    }
  }
  EXPECT_EQ(padded, 131);
}

TEST(Gemma2Graph, StateOnlyRetainsAllWritesAndNarrowingKeepsFullFinalAttention) {
  Case c;
  c.shape.outputs = 0;
  c.shape.output_mode = kg::Gemma2OutputMode::kStateOnly;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->hidden, nullptr);
  EXPECT_EQ(graph->logits, nullptr);
  EXPECT_EQ(graph->Named("blk.25.q_rope"), nullptr);
  EXPECT_EQ(graph->Named("blk.25.attn_projection"), nullptr);
  EXPECT_NE(graph->Named("blk.25.k_rope"), nullptr);
  EXPECT_NE(graph->Named("blk.25.v_raw"), nullptr);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 50);
  arena->Reset();
  c.shape.outputs = 2;
  c.shape.output_mode = kg::Gemma2OutputMode::kHead;
  graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, {.narrow_final = true});
  ASSERT_TRUE(graph) << *jitllm::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->Named("blk.25.q_rope")->ne[2], 4);
  EXPECT_EQ(graph->Named("blk.25.attn_projection")->ne[1], 4);
  EXPECT_EQ(graph->Named("blk.25.output")->ne[1], 2);
  EXPECT_EQ(graph->hidden->ne[1], 2);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
}

TEST(Gemma2Graph, GreedyFrontiersHaveDistinctShapeAndKeptArgmaxDescriptor) {
  Case c;
  const auto full_head = c.shape;
  c.shape.greedy = true;
  EXPECT_NE(c.shape, full_head);
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  ASSERT_NE(graph->greedy, nullptr);
  EXPECT_EQ(graph->greedy->type, GGML_TYPE_I32);
  EXPECT_EQ(graph->greedy->ne[0], 2);
  EXPECT_EQ(graph->greedy->src[0], graph->logits);
  EXPECT_EQ(kg::JitllmOpOf(graph->greedy), kg::JitllmOp::kArgmax);
  EXPECT_EQ(graph->nodes.back(), graph->greedy);
  auto malformed = c.shape;
  malformed.outputs = 4;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, malformed));
  malformed.outputs = 0;
  malformed.output_mode = kg::Gemma2OutputMode::kStateOnly;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, malformed));
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, c.shape, {.head = false}));
}

TEST(Gemma2Graph, InvalidDescriptorsScopesAndCapacityRefuseBeforeGraphConstruction) {
  Case c;
  const auto rejects = [&](const auto& shape, const kg::Gemma2GraphOptions& options = {}) {
    EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, shape, options));
  };
  auto changed = c.shape;
  changed.segments[1].slot = changed.segments[0].slot;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].global_n_kv = 256;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].local_n_kv = 4353;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].n_past = 8191;
  rejects(changed);
  rejects(c.shape, {.first_layer = 5});
  rejects(c.shape, {.layer_count = 1, .narrow_final = true});
  changed = c.shape;
  changed.outputs = 0;
  changed.output_mode = kg::Gemma2OutputMode::kStateOnly;
  rejects(changed, {.hidden_input = true});
  auto bad_binding = c.binding;
  bad_binding.layers[0].v.index = bad_binding.layers[0].k.index;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, bad_binding, c.state, c.shape));
  auto bad_layout = c.state;
  --bad_layout.tensors[0].bytes;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, bad_layout, c.shape));
  auto small = kg::TensorArena::Create(1);
  ASSERT_TRUE(small);
  const auto before = small->used();
  EXPECT_FALSE(kg::BuildGemma2Graph(*small, c.p, c.binding, c.state, c.shape));
  EXPECT_EQ(small->used(), before);
  auto advanced = c.shape;
  ++advanced.segments[0].n_past;
  EXPECT_EQ(advanced, c.shape);  // Positions are fresh inputs at the same read widths.
  advanced.segments[0].global_n_kv += 256;
  EXPECT_NE(advanced, c.shape);
}
}  // namespace
