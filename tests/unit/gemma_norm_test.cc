// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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

#include "components/graph_io.h"
#include "execution/registry.h"
#include "ggml.h"
#include "ir/builder.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_read_index.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/tensors.h"
#include "planner/plan.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace kn = llmp::kernels;
namespace ir = llmp::ir;
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
TEST_F(GemmaNormTest, AcceptsUnpinnedResidualWidths) {
  for (int width : {2559, 2561, 4096}) {
    const auto t = Add(width);
    EXPECT_TRUE(kg::CheckGemmaNormAdd(t.norm, t.mul, t.out));
  }
}
TEST_F(GemmaNormTest, EvenWidthFrequencyFactors) {
  for (int dim : {64, 80, 96, 112, 128, 192, 256, 512, 1024}) {
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
  const auto registry = llmp::execution::Registry::Create(kg::PlanningImplementations()).value();
  const pl::DeviceFacts facts{
      .device_class = "sm_121", .arch = 121, .features = llmp::execution::kBf16Mma};
  for (bool rope : {false, true}) {
    ir::Builder b;
    b.SetOwner(0);
    if (rope) {
      const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256ULL, 1ULL})));
      const auto w = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256ULL, 1ULL})));
      const auto pos = b.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({1ULL})));
      const auto norm = b.RmsNorm(x, 1e-6f);
      const auto mul = b.Mul(norm, w);
      b.Output(b.Rope(mul, pos, ir::TensorId{},
                      {.mode = ir::RopeMode::kNeox, .dims = 256, .original_context = 4096}));
    } else {
      const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({2816ULL, 1ULL})));
      const auto w = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({2816ULL, 1ULL})));
      const auto res =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({2816ULL, 1ULL})));
      const auto norm = b.RmsNorm(x, 1e-6f);
      const auto mul = b.Mul(norm, w);
      b.Output(b.Add(mul, res));
    }
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    co::StepGraph step{std::move(*graph), {}, {}, {}};
    for (const ir::TensorId in : step.graph.inputs()) {
      step.inputs.push_back({.tensor = in, .kind = co::InputKind::kFeatures});
    }
    const pl::StepShape shape{.segments = {{.rows = 1}}};
    const kn::Addresses addresses{};

    pl::Policy off_policy{};
    off_policy.disabled = {rope ? "rms_norm_mul_rope" : "rms_norm_mul_add"};
    auto off_plan = pl::Plan(step.graph.Clone(), registry, facts, off_policy);
    ASSERT_TRUE(off_plan) << (off_plan ? "" : off_plan.error().detail);
    auto primitive = kg::LowerStep({*off_plan, step, shape, addresses}, registry, {});
    ASSERT_TRUE(primitive) << (primitive ? "" : primitive.error().detail);
    const auto name = rope ? kg::kGemmaNormRopeName : kg::kGemmaNormAddName;
    EXPECT_TRUE(std::ranges::none_of(primitive->plan.steps,
                                     [name](const auto& s) { return s.implementation == name; }));

    pl::Policy on_policy{};
    auto on_plan = pl::Plan(step.graph.Clone(), registry, facts, on_policy);
    ASSERT_TRUE(on_plan) << (on_plan ? "" : on_plan.error().detail);
    auto fused = kg::LowerStep({*on_plan, step, shape, addresses}, registry, {});
    ASSERT_TRUE(fused) << (fused ? "" : fused.error().detail);
    ASSERT_EQ(fused->plan.steps.size(), 1U);
    EXPECT_EQ(fused->plan.steps[0].implementation, name);
    EXPECT_GT(fused->extent, 0U);
  }
}
TEST_F(GemmaNormTest, ResidualGatherStaysPaidAndPlacedRawInputLivesUntilDeferredAdd) {
  const auto registry = llmp::execution::Registry::Create(kg::PlanningImplementations()).value();
  const pl::DeviceFacts facts{
      .device_class = "sm_121", .arch = 121, .features = llmp::execution::kBf16Mma};
  const pl::Policy policy{};
  for (std::uint64_t width : {2304ULL, 2560ULL, 2816ULL, 5376ULL}) {
    ir::Builder b;
    b.SetOwner(0);
    const auto leaf = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({width, 1ULL})));
    const auto other = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({width, 1ULL})));
    const auto weight =
        b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({width, 1ULL})));
    const auto table =
        b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({width, 10ULL})));
    const auto ids = b.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({1ULL})));
    const auto x = b.Add(leaf, other);
    const auto norm = b.RmsNorm(x, 1e-6f);
    const auto mul = b.Mul(norm, weight);
    const auto gather = b.GetRows(table, ids);
    const auto out = b.Add(mul, gather);
    b.Output(out);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    co::StepGraph step{std::move(*graph), {}, {}, {}};
    for (const ir::TensorId in : step.graph.inputs()) {
      step.inputs.push_back({.tensor = in, .kind = co::InputKind::kFeatures});
    }
    const pl::StepShape shape{.segments = {{.rows = 1}}};
    auto plan = pl::Plan(step.graph.Clone(), registry, facts, policy);
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const kn::Addresses addresses{};

    auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, {});
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    EXPECT_GT(lowered->extent, 0U);
    EXPECT_FALSE(lowered->plan.steps.empty());
  }
}
TEST_F(GemmaNormTest, ResidualGatherValidatesGatherDependencies) {
  auto [t, gather] = AddGather(5376);
  std::array nodes{t.norm, t.mul, gather, t.out};
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
}  // namespace
