// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_norm.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/graph_read_index.h"
#include "kernels/ggml/tensors.h"

namespace {
namespace kg = jitllm::kernels::ggml;
class GemmaNormTest : public ::testing::Test {
 protected:
  kg::TensorArena arena = kg::TensorArena::Create(160).value();
  ggml_context* c() { return arena.context(); }
  std::uint64_t next = 1ULL << 40;
  ggml_tensor* Bind(ggml_tensor* t) {
    kg::TensorArena::Bind(t, next);
    next += 1ULL << 32;
    return t;
  }
  struct Chain {
    ggml_tensor *x, *weight, *norm, *mul, *out;
  };
  Chain Add(int width, bool reverse = false) {
    auto* x = Bind(ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, 4));
    auto* weight = Bind(ggml_new_tensor_1d(c(), GGML_TYPE_F32, width));
    auto* norm = Bind(ggml_rms_norm(c(), x, 1e-6f));
    auto* mul = Bind(ggml_mul(c(), norm, weight));
    auto* residual = Bind(ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, 4));
    auto* out = Bind(reverse ? ggml_add(c(), residual, mul) : ggml_add(c(), mul, residual));
    return {x, weight, norm, mul, out};
  }
  Chain Rope(int dim, bool factors = false) {
    auto* x = Bind(ggml_new_tensor_3d(c(), GGML_TYPE_F32, dim, 4, 6));
    auto* weight = Bind(ggml_new_tensor_1d(c(), GGML_TYPE_F32, dim));
    auto* norm = Bind(ggml_rms_norm(c(), x, 1e-6f));
    auto* mul = Bind(ggml_mul(c(), norm, weight));
    auto* positions = Bind(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 6));
    auto* freq = factors ? Bind(ggml_new_tensor_1d(c(), GGML_TYPE_F32, dim / 2)) : nullptr;
    auto* out = Bind(ggml_rope_ext(c(), mul, positions, freq, dim, GGML_ROPE_TYPE_NEOX, 262144,
                                   10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    return {x, weight, norm, mul, out};
  }
  std::pair<Chain, ggml_tensor*> AddGather(int width) {
    auto t = Add(width);
    auto* values = Bind(ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, 8));
    auto* indices = Bind(ggml_new_tensor_1d(c(), GGML_TYPE_I32, 4));
    auto* gather = Bind(ggml_get_rows(c(), values, indices));
    t.out = Bind(ggml_add(c(), t.mul, gather));
    return {t, gather};
  }
  static std::array<ggml_tensor*, 3> Nodes(Chain t) { return {t.norm, t.mul, t.out}; }
};

TEST_F(GemmaNormTest, ExactUseCountsKeepDuplicateSelfAndViewEdgesSeparate) {
  auto t = Add(2816);
  auto* view = ggml_view_1d(c(), t.norm, 2816, 0);
  auto* consumer = Bind(ggml_dup_tensor(c(), t.norm));
  consumer->src[0] = t.norm;
  consumer->src[1] = t.norm;
  consumer->src[2] = view;
  // Repeated node occurrences and producer-self edges count independently.
  t.norm->src[1] = t.norm;
  std::array nodes{t.norm, consumer, consumer};
  std::array keep{t.norm, view};
  t.norm->flags |= GGML_TENSOR_FLAG_OUTPUT;
  const kg::detail::GraphReadIndex reads(nodes, keep);
  EXPECT_EQ(reads.SourceUses(nodes, t.norm), 5);
  EXPECT_EQ(reads.SourceUses(nodes, view), 2);
  EXPECT_EQ(reads.SourceUses(nodes, consumer), 0);
  EXPECT_FALSE(reads.SourceUses(nodes, nullptr));
  EXPECT_FALSE(reads.SourceUses(std::span(nodes).first(1), t.norm));
  auto copied = nodes;
  EXPECT_FALSE(reads.SourceUses(copied, t.norm));
  consumer->src[1] = t.x;
  const kg::detail::GraphReadIndex fresh(nodes, keep);
  EXPECT_EQ(fresh.SourceUses(nodes, t.norm), 3);
}

TEST_F(GemmaNormTest, IndexedNormGatesPreserveLocalGatherAndExternalUseRefusals) {
  for (bool rope : {false, true}) {
    const auto t = rope ? Rope(256) : Add(5376);
    auto nodes = Nodes(t);
    const kg::detail::GraphReadIndex reads(nodes, {});
    EXPECT_EQ(bool(kg::RmsNormMulFusionAt(nodes, 0)),
              bool(kg::RmsNormMulFusionAt(nodes, 0, &reads)));
    EXPECT_EQ(bool(kg::GemmaNormRopeFusionAt(nodes, 0)),
              bool(kg::GemmaNormRopeFusionAt(nodes, 0, &reads)));
    EXPECT_EQ(bool(kg::GemmaNormAddFusionAt(nodes, 0)),
              bool(kg::GemmaNormAddFusionAt(nodes, 0, &reads)));
  }
  auto [t, gather] = AddGather(5376);
  std::vector nodes{t.norm, t.mul, gather, t.out};
  const kg::detail::GraphReadIndex before(nodes, {});
  ASSERT_TRUE(kg::GemmaNormAddGatherFusionAt(nodes, 0));
  ASSERT_TRUE(kg::GemmaNormAddGatherFusionAt(nodes, 0, &before));
  // A different graph must scan rather than borrow the old graph's counts.
  auto extended = nodes;
  extended.push_back(Bind(ggml_scale(c(), t.norm, 2.0f)));
  EXPECT_FALSE(kg::GemmaNormAddGatherFusionAt(extended, 0));
  EXPECT_FALSE(kg::GemmaNormAddGatherFusionAt(extended, 0, &before));
  const kg::detail::GraphReadIndex after(extended, {});
  EXPECT_FALSE(kg::GemmaNormAddGatherFusionAt(extended, 0, &after));
}

TEST_F(GemmaNormTest, ApprovedResidualWidthsAndOperandOrders) {
  for (int width : {2304, 2560, 2816, 5376})
    for (bool reverse : {false, true}) {
      const auto t = Add(width, reverse);
      EXPECT_TRUE(kg::CheckGemmaNormAdd(t.norm, t.mul, t.out));
    }
}
TEST_F(GemmaNormTest, RejectsUnapprovedResidualWidths) {
  for (int width : {2559, 2561, 4096}) {
    const auto t = Add(width);
    EXPECT_FALSE(kg::CheckGemmaNormAdd(t.norm, t.mul, t.out));
  }
}
TEST_F(GemmaNormTest, D256AndD512FrequencyFactors) {
  for (int dim : {256, 512}) {
    const auto t = Rope(dim, dim == 512);
    EXPECT_TRUE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
  }
}
TEST_F(GemmaNormTest, RejectsInvalidRotationMetadataAndStaleFactors) {
  auto t = Rope(512, true);
  t.out->op_params[15] = 2;
  EXPECT_FALSE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
  t.out->op_params[15] = 0;
  t.out->op_params[1] = 256;
  EXPECT_FALSE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
  t.out->op_params[1] = 512;
  t.norm->op_params[0] = std::bit_cast<std::int32_t>(std::numeric_limits<float>::infinity());
  EXPECT_FALSE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
  t.norm->op_params[0] = std::bit_cast<std::int32_t>(1e-6f);
  auto* view = ggml_view_1d(c(), t.out->src[2], 256, 0);
  t.out->src[2] = view;
  Bind(view->view_src);
  EXPECT_FALSE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
}
TEST_F(GemmaNormTest, RejectsActualSourceOverlapsAndMalformedResidual) {
  for (int width : {2304, 2560, 5376}) {
    auto t = Add(width);
    auto* original = t.out->data;
    t.out->data = t.x->data;
    EXPECT_FALSE(kg::CheckGemmaNormAdd(t.norm, t.mul, t.out));
    t.out->data = static_cast<char*>(t.x->data) + 4;
    EXPECT_FALSE(kg::CheckGemmaNormAdd(t.norm, t.mul, t.out));
    t.out->data = original;
    t.out->src[1]->nb[0] = 8;
    EXPECT_FALSE(kg::CheckGemmaNormAdd(t.norm, t.mul, t.out));
  }
}
TEST_F(GemmaNormTest, RejectsDependenciesOnUnwrittenIntermediates) {
  auto add = Add(2816);
  add.out->src[1] = add.norm;
  EXPECT_FALSE(kg::CheckGemmaNormAdd(add.norm, add.mul, add.out));
  add.out->src[1] = ggml_view_2d(c(), add.mul, 2816, 4, add.mul->nb[1], 0);
  EXPECT_FALSE(kg::CheckGemmaNormAdd(add.norm, add.mul, add.out));
  add.norm->view_src = add.x;
  add.norm->view_offs = 0;
  add.norm->data = add.x->data;
  add.out->src[1] = add.norm;
  EXPECT_FALSE(kg::CheckGemmaNormAdd(add.norm, add.mul, add.out));
  auto rope = Rope(512, true);
  rope.out->src[2] = ggml_view_1d(c(), rope.norm, 256, 0);
  EXPECT_FALSE(kg::CheckGemmaNormRope(rope.norm, rope.mul, rope.out));
  // A separately materialized input may reuse an intermediate's physical slot.
  rope.out->src[2] = Bind(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 256));
  rope.out->src[2]->data = rope.norm->data;
  EXPECT_TRUE(kg::CheckGemmaNormRope(rope.norm, rope.mul, rope.out));
}
TEST_F(GemmaNormTest, RejectsCyclicAndOverflowedOperandViewChains) {
  auto t = Rope(512, true);
  auto* factor = t.out->src[2];
  factor->view_src = factor;
  EXPECT_FALSE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
  factor->view_src = t.x;
  factor->view_offs = std::numeric_limits<std::size_t>::max();
  EXPECT_FALSE(kg::CheckGemmaNormRope(t.norm, t.mul, t.out));
}
TEST_F(GemmaNormTest, SeparateDefaultOffPoliciesAndPlacedPlanRemainStable) {
  for (bool rope : {false, true}) {
    auto t = rope ? Rope(256) : Add(2816);
    auto nodes = Nodes(t);
    kg::DeviceChoices off;
    const auto primitive = kg::PlanGraph(nodes, false, off);
    ASSERT_TRUE(primitive);
    EXPECT_EQ(primitive->steps.size(), 2);
    kg::DeviceChoices on;
    on.fuse_norm_rope = rope;
    on.fuse_norm_add = !rope;
    const auto fused = kg::PlanGraph(nodes, false, on);
    ASSERT_TRUE(fused);
    ASSERT_EQ(fused->steps.size(), 1);
    EXPECT_EQ(fused->steps[0].implementation,
              rope ? kg::kGemmaNormRopeName : kg::kGemmaNormAddName);
    std::vector<ggml_tensor*> inputs{t.x, t.weight};
    for (int i = 1; i < 3; ++i)
      if (t.out->src[i]) inputs.push_back(t.out->src[i]);
    const auto placed = kg::PlaceActivations(nodes, *fused, inputs, 256);
    ASSERT_TRUE(placed);
    for (auto [tensor, offset] : placed->offsets)
      kg::TensorArena::Bind(tensor, (1ULL << 32) + offset);
    const auto again = kg::PlanGraph(nodes, false, on);
    ASSERT_TRUE(again);
    EXPECT_TRUE(kg::SamePlan(*fused, *again));
  }
}
TEST_F(GemmaNormTest, UnrelatedCyclicKeepsAndReadersAreRefusedBeforeRootScans) {
  for (bool rope : {false, true}) {
    auto t = rope ? Rope(256) : Add(2816);
    auto nodes = Nodes(t);
    kg::DeviceChoices on;
    on.fuse_norm_rope = rope;
    on.fuse_norm_add = !rope;
    auto* unrelated = Bind(ggml_new_tensor_1d(c(), GGML_TYPE_F32, 16));
    unrelated->view_src = unrelated;
    std::array keep{unrelated};
    EXPECT_FALSE(kg::PlanGraph(nodes, false, on, keep));
    std::vector graph(nodes.begin(), nodes.end());
    graph.push_back(unrelated);
    EXPECT_FALSE(kg::PlanGraph(graph, false, on));
  }
}
TEST_F(GemmaNormTest, ResidualGatherStaysPaidAndPlacedRawInputLivesUntilDeferredAdd) {
  for (int width : {2304, 2560, 2816, 5376}) {
    auto [t, gather] = AddGather(width);
    auto* leaf = t.x;
    auto* other = Bind(ggml_dup_tensor(c(), leaf));
    t.x = Bind(ggml_add(c(), leaf, other));
    t.norm->src[0] = t.x;
    std::array nodes{t.x, t.norm, t.mul, gather, t.out};
    kg::DeviceChoices on;
    on.fuse_norm_add = true;
    const auto off = kg::PlanGraph(nodes, false, {});
    ASSERT_TRUE(off);
    EXPECT_EQ(off->steps.size(), 4);
    const auto plan = kg::PlanGraph(nodes, false, on);
    ASSERT_TRUE(plan);
    ASSERT_EQ(plan->steps.size(), 3);
    EXPECT_EQ(plan->steps[1].operation, jitllm::execution::Operation::kGetRows);
    EXPECT_EQ(plan->steps[1].nodes[0], gather);
    EXPECT_EQ(plan->steps[2].implementation, kg::kGemmaNormAddName);
    EXPECT_EQ(plan->steps[2].nodes[0], t.norm);
    std::array inputs{leaf, other, t.weight, gather->src[0], gather->src[1]};
    const auto placed = kg::PlaceActivations(nodes, *plan, inputs, 256);
    ASSERT_TRUE(placed);
    for (auto [tensor, offset] : placed->offsets)
      kg::TensorArena::Bind(tensor, (1ULL << 32) + offset);
    EXPECT_TRUE(kg::CheckGemmaNormAddGather(t.norm, t.mul, gather, t.out));
    EXPECT_NE(t.x->data, gather->data);
    const auto repeated = kg::PlanGraph(nodes, false, on);
    ASSERT_TRUE(repeated);
    EXPECT_TRUE(kg::SamePlan(*plan, *repeated));
  }
}
TEST_F(GemmaNormTest, ResidualGatherKeepsDependenciesAndPreparationAliasesRefuseFusion) {
  auto [t, gather] = AddGather(5376);
  std::array nodes{t.norm, t.mul, gather, t.out};
  kg::DeviceChoices on;
  on.fuse_norm_add = true;
  auto* view = ggml_view_1d(c(), t.norm, 5376, 0);
  for (auto* kept : {t.norm, t.mul, view}) {
    std::array keep{kept};
    const auto plan = kg::PlanGraph(nodes, false, on, keep);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->steps.size(), 3);
    EXPECT_EQ(plan->steps[0].implementation, "ggml.rms_norm_mul.unfused");
  }
  std::array keep{gather};
  const auto kept_gather = kg::PlanGraph(nodes, false, on, keep);
  ASSERT_TRUE(kept_gather);
  EXPECT_EQ(kept_gather->steps.size(), 2);
  auto* saved = gather->data;
  gather->data = t.x->data;
  EXPECT_FALSE(kg::GemmaNormAddGatherFusionAt(nodes, 0));
  gather->data = static_cast<char*>(t.x->data) + 4;
  EXPECT_FALSE(kg::GemmaNormAddGatherFusionAt(nodes, 0));
  gather->data = saved;
  auto* source = gather->src[0];
  gather->src[0] = t.mul;
  EXPECT_FALSE(kg::CheckGemmaNormAddGather(t.norm, t.mul, gather, t.out));
  gather->src[0] = source;
  EXPECT_TRUE(kg::GemmaNormAddGatherFusionAt(nodes, 0));
}

TEST_F(GemmaNormTest, LegacyNormFusionFallbackPreservesCombinedPolicyKeeps) {
  for (bool rope : {false, true}) {
    auto t = rope ? Rope(256) : Add(2816);
    auto nodes = Nodes(t);
    kg::DeviceChoices on;
    on.fuse_norms = true;
    on.fuse_norm_rope = rope;
    on.fuse_norm_add = !rope;
    auto* view = ggml_view_1d(c(), t.norm, t.norm->ne[0], 0);
    for (auto* kept : {t.norm, view}) {
      std::array keep{kept};
      const auto plan = kg::PlanGraph(nodes, false, on, keep);
      ASSERT_TRUE(plan);
      EXPECT_EQ(plan->steps[0].implementation, "ggml.rms_norm_mul.unfused");
      const auto generic = kg::PlanGraph(std::span(nodes).first(2), true, on, keep);
      ASSERT_TRUE(generic);
      EXPECT_EQ(generic->steps[0].implementation, "ggml.rms_norm");
    }
  }
}
TEST_F(GemmaNormTest, KeepsAndViewConsumersRetainPrimitiveIntermediates) {
  for (bool rope : {false, true}) {
    auto t = rope ? Rope(256) : Add(2816);
    auto nodes = Nodes(t);
    kg::DeviceChoices on;
    on.fuse_norm_rope = rope;
    on.fuse_norm_add = !rope;
    for (auto* kept : {t.norm, t.mul}) {
      std::array keep{kept};
      const auto p = kg::PlanGraph(nodes, false, on, keep);
      ASSERT_TRUE(p);
      EXPECT_EQ(p->steps.size(), 2);
    }
    auto* view = ggml_view_1d(c(), t.mul, t.mul->ne[0], 0);
    std::array keep{view};
    const auto p = kg::PlanGraph(nodes, false, on, keep);
    ASSERT_TRUE(p);
    EXPECT_EQ(p->steps.size(), 2);
    std::vector expanded(nodes.begin(), nodes.end());
    expanded.push_back(view);
    const auto q = kg::PlanGraph(expanded, false, on);
    ASSERT_TRUE(q);
    EXPECT_EQ(q->steps.size(), 2);
  }
}
}  // namespace
