// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma4_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "base/check.h"
#include "expected_error.h"
#include "gemma4_fixture.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace fixture = jitllm::test_support::gemma4;
struct Case {
  const md::Gemma4Profile& p;
  md::Gemma4Binding binding;
  md::Gemma4StateLayout state;
  kg::Gemma4ChunkShape shape;
  md::Gemma4ChunkInputs input;
  explicit Case(std::uint32_t size = 26, std::uint32_t slots = 1)
      : p(size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B()) {
    auto b = md::BindGemma4(p, "gemma4", fixture::Resources(size));
    jitllm::base::Check(b.has_value(), "Gemma graph fixture binding failed");
    binding = std::move(*b);
    auto s = md::Gemma4State(p, 4096, 16);
    jitllm::base::Check(s.has_value(), "Gemma graph state failed");
    state = std::move(*s);
    const std::array<std::int32_t, 2> tokens{1, 2};
    std::vector<md::Gemma4Segment> segments;
    for (std::uint32_t i = 0; i < slots; ++i) segments.push_back({i, i * 1100, tokens});
    auto in = md::Gemma4Chunk(p, state, segments, true);
    jitllm::base::Check(in.has_value(), "Gemma graph input failed");
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
      ASSERT_TRUE(plan) << jitllm::test_support::Failed(plan)->detail;
      const auto stores = [](const auto& p) {
        return std::ranges::count_if(p.steps, [](const auto& step) {
          return step.operation == jitllm::execution::Operation::kRopeSetRows;
        });
      };
      EXPECT_EQ(stores(*plan), c.p.layers * slots);
      auto placement = kg::PlaceActivations(built->nodes, *plan, built->inputs, 256);
      ASSERT_TRUE(placement) << jitllm::test_support::Failed(placement)->detail;
      for (const auto& step : plan->steps)
        if (step.operation == jitllm::execution::Operation::kRopeSetRows) {
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
  EXPECT_EQ(plan->steps[0].operation, jitllm::execution::Operation::kRope);
  EXPECT_EQ(plan->steps[1].operation, jitllm::execution::Operation::kSetRows);
}

TEST(Gemma4Graph, BothActualContractsBuildCompleteTextGraphs) {
  for (const auto size : {26U, 31U}) {
    Case c(size);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
    ASSERT_TRUE(arena);
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
    ASSERT_TRUE(g) << jitllm::test_support::Failed(g)->detail;
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
    ASSERT_TRUE(g) << jitllm::test_support::Failed(g)->detail;
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
  ASSERT_TRUE(g) << jitllm::test_support::Failed(g)->detail;
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
    if (kg::JitllmOpOf(t) == kg::JitllmOp::kVecQ) {
      const auto valid = kg::CheckVecQ(t);
      EXPECT_TRUE(valid) << (valid ? "" : valid.error().detail);
      EXPECT_TRUE(kg::VecQOneToken(t));
      ++products;
    }
    if (kg::JitllmOpOf(t) == kg::JitllmOp::kQuantizeQ8) {
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
    if (kg::JitllmOpOf(t) != kg::JitllmOp::kQuantizeQ8 || t->src[0] != input) continue;
    EXPECT_EQ(q8, nullptr);
    q8 = t;
  }
  ASSERT_NE(q8, nullptr);
  for (const auto* t : g->nodes) {
    if (kg::JitllmOpOf(t) == kg::JitllmOp::kVecQ && t->src[1] == q8) ++consumers;
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
