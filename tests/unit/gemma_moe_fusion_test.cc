// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_moe_fusion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "base/check.h"
#include "components/graph_io.h"
#include "execution/registry.h"
#include "ir/builder.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/graph_read_index.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/plan_step.h"
#include "planner/plan.h"

namespace {
namespace kg = llmp::kernels::ggml;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace kn = llmp::kernels;
namespace ir = llmp::ir;

void BindDistinct(std::span<ggml_tensor* const> nodes, std::uint64_t base) {
  for (auto* node : nodes) {
    if (node != nullptr && node->view_src == nullptr && node->data == nullptr) {
      kg::TensorArena::Bind(node, base);
      base += ((ggml_nbytes(node) + 255) / 256 * 256) + 256;
    }
  }
  kg::BindViews(nodes);
}

class GemmaMoeFusion : public ::testing::Test {
 protected:
  void SetUp() override {
    arena = std::make_unique<kg::TensorArena>(kg::TensorArena::Create(128).value());
  }
  ggml_context* c() { return arena->context(); }
  ggml_tensor* Tensor(std::int64_t x, std::int64_t y, std::int64_t z = 1) {
    auto* t = ggml_new_tensor_3d(c(), GGML_TYPE_F32, x, y, z);
    kg::TensorArena::Bind(t, address);
    address += (ggml_nbytes(t) + 255) / 256 * 256 + 256;
    return t;
  }
  std::vector<ggml_tensor*> Route(std::int64_t rows = 4, std::int64_t experts = 128,
                                  std::int64_t used = 8) {
    auto* x = Tensor(experts, rows);
    auto* p = ggml_soft_max(c(), x);
    auto* probabilities = ggml_reshape_3d(c(), p, 1, experts, rows);
    auto* ids = ggml_argsort_top_k(c(), p, static_cast<int>(used));
    auto* gathered = ggml_get_rows(c(), probabilities, ids);
    auto* selected = ggml_reshape_2d(c(), gathered, used, rows);
    auto* sum = ggml_sum_rows(c(), selected);
    auto* clamped =
        ggml_clamp(c(), sum, kg::kGemmaRouteClamp, std::numeric_limits<float>::infinity());
    auto* normalized = ggml_div(c(), selected, clamped);
    auto* out = ggml_reshape_3d(c(), normalized, 1, used, rows);
    std::vector<ggml_tensor*> n{p,        probabilities, ids->view_src, ids,        gathered,
                                selected, sum,           clamped,       normalized, out};
    BindDistinct(n, 0x10000000000ULL);
    return n;
  }
  std::vector<ggml_tensor*> Reduce(std::int64_t rows = 4, std::int64_t width = 2816,
                                   std::int64_t used = 8) {
    auto* x = Tensor(width, used, rows);
    auto* scales = Tensor(1, used, rows);
    auto* weights = Tensor(1, used, rows);
    auto* scaled = ggml_mul(c(), x, scales);
    auto* weighted = ggml_mul(c(), scaled, weights);
    std::vector<ggml_tensor*> n{scaled, weighted};
    for (std::size_t i = 0; i < static_cast<std::size_t>(used); ++i)
      n.push_back(ggml_view_2d(c(), weighted, width, rows, weighted->nb[2], i * weighted->nb[1]));
    auto* out = n[2];
    for (std::size_t i = 1; i < static_cast<std::size_t>(used); ++i) {
      out = ggml_add(c(), out, n[2 + i]);
      n.push_back(out);
    }
    BindDistinct(n, 0x10000000000ULL);
    return n;
  }
  static std::expected<ir::Graph, ir::VerifyError> MakeRoute(std::int64_t rows = 4,
                                                             std::int64_t experts = 128,
                                                             std::int64_t used = 8) {
    ir::Builder b;
    const auto logits = b.Input(ir::Dense(
        llmp::format::Encoding::kF32,
        ir::ShapeOf({static_cast<std::uint64_t>(experts), static_cast<std::uint64_t>(rows)})));
    const auto r = b.Route(logits, {}, {}, {},
                           {.normalize = true,
                            .experts = static_cast<std::uint32_t>(experts),
                            .used = static_cast<std::uint32_t>(used)});
    b.Output(r[1]);
    return std::move(b).Finish();
  }
  static std::expected<ir::Graph, ir::VerifyError> MakeReduce(std::int64_t rows = 4,
                                                              std::int64_t width = 2816,
                                                              std::int64_t used = 8) {
    ir::Builder b;
    const auto x = b.Input(
        ir::Dense(llmp::format::Encoding::kF32,
                  ir::ShapeOf({static_cast<std::uint64_t>(width), static_cast<std::uint64_t>(used),
                               static_cast<std::uint64_t>(rows)})));
    const auto w = b.Input(ir::Dense(
        llmp::format::Encoding::kF32,
        ir::ShapeOf({static_cast<std::uint64_t>(used), static_cast<std::uint64_t>(rows)})));
    const auto ids = b.Input(ir::Dense(
        llmp::format::Encoding::kI32,
        ir::ShapeOf({static_cast<std::uint64_t>(used), static_cast<std::uint64_t>(rows)})));
    const auto scale = b.Input(ir::Dense(
        llmp::format::Encoding::kF32,
        ir::ShapeOf({1ULL, static_cast<std::uint64_t>(used), static_cast<std::uint64_t>(rows)})));
    b.Output(b.ExpertReduce(x, w, ids, scale));
    return std::move(b).Finish();
  }
  std::unique_ptr<kg::TensorArena> arena;
  std::uint64_t address = 0x100000000ULL;
};
TEST_F(GemmaMoeFusion, UnpinnedExpertCountsAndWidthsKeepTheLiteralGraphContracts) {
  auto route = Route(3, 64, 4);
  ASSERT_TRUE(kg::GemmaRoutingFusionAt(route, 0));
  arena->Reset();
  auto reduce = Reduce(3, 4096, 6);
  const auto fused = kg::GemmaReductionFusionAt(reduce, 0);
  ASSERT_TRUE(fused);
  EXPECT_EQ(fused->nodes.size(), 13U);
}
TEST_F(GemmaMoeFusion, ExactPatternsPreserveBothRoutingOutputsAndScaledOrder) {
  for (const auto rows : {1, 2, 4, 8, 128, 8192}) {
    arena->Reset();
    auto n = Route(rows);
    auto routing = kg::GemmaRoutingFusionAt(n, 0);
    ASSERT_TRUE(routing) << rows;
    EXPECT_EQ(routing->operands.ids.bytes.value(), static_cast<std::uint64_t>(rows) * 512);
    EXPECT_EQ(routing->operands.weights.tensor->view_src, n[8]);
    std::array keep{n[3], n[9]};
    EXPECT_TRUE(kg::GemmaRoutingFusionAt(n, 0, keep));
    arena->Reset();
    n = Reduce(rows);
    EXPECT_TRUE(kg::GemmaReductionFusionAt(n, 0)) << rows;
    EXPECT_FALSE(kg::GemmaReductionFusionAt(n, n.size()));
    EXPECT_FALSE(kg::GemmaReductionFusionAt(n, std::numeric_limits<std::size_t>::max()));
  }
}
TEST_F(GemmaMoeFusion, ExplicitPoliciesKeepPrimitivesAndProtectEveryUnwrittenValue) {
  const auto registry = llmp::execution::Registry::Create(kg::PlanningImplementations()).value();
  const pl::DeviceFacts facts{
      .device_class = "sm_121", .arch = 121, .features = llmp::execution::kBf16Mma};
  for (bool route : {true, false}) {
    auto graph = route ? MakeRoute() : MakeReduce();
    ASSERT_TRUE(graph);
    const kn::Addresses addresses{};
    co::StepGraph step{std::move(*graph), {}, {}, {}};
    for (const ir::TensorId in : step.graph.inputs()) {
      step.inputs.push_back({.tensor = in, .kind = co::InputKind::kFeatures});
    }
    const pl::StepShape shape{.segments = {{.rows = 4}}};
    const auto name = route ? kg::kGemmaRouteName : kg::kGemmaReduceName;

    pl::Policy off_policy{};
    off_policy.disabled = {"route_reduce"};
    auto off_plan = pl::Plan(step.graph.Clone(), registry, facts, off_policy,
                             {.written = &kg::WrittenBytes, .within = &kg::WithinRoot});
    ASSERT_TRUE(off_plan) << (off_plan ? "" : off_plan.error().detail);
    auto ordinary = kg::LowerStep({*off_plan, step, shape, addresses}, registry, {});
    ASSERT_TRUE(ordinary) << (ordinary ? "" : ordinary.error().detail);
    EXPECT_TRUE(std::ranges::none_of(ordinary->plan.steps,
                                     [name](const auto& s) { return s.implementation == name; }));

    pl::Policy on_policy{};
    auto on_plan = pl::Plan(step.graph.Clone(), registry, facts, on_policy,
                            {.written = &kg::WrittenBytes, .within = &kg::WithinRoot});
    ASSERT_TRUE(on_plan) << (on_plan ? "" : on_plan.error().detail);
    auto fused = kg::LowerStep({*on_plan, step, shape, addresses}, registry, {});
    ASSERT_TRUE(fused) << (fused ? "" : fused.error().detail);
    ASSERT_EQ(fused->plan.steps.size(), 1U);
    EXPECT_EQ(fused->plan.steps.front().implementation, name);
    EXPECT_EQ(fused->plan.steps.front().nodes.size(), route ? 10U : 17U);
  }
}
TEST_F(GemmaMoeFusion, EveryUnwrittenRoutingIntermediateRequiresPrimitives) {
  auto n = Route();
  for (const auto index : {0U, 1U, 2U, 4U, 5U, 6U, 7U}) {
    std::array keep{n[index]};
    EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0, keep)) << index;
    n[index]->flags |= GGML_TENSOR_FLAG_OUTPUT;
    EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0)) << index;
    n[index]->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
  }
  auto* full_sort_view = ggml_view_2d(c(), n[2], 128, 4, n[2]->nb[1], 0);
  full_sort_view->data = n[2]->data;
  n.push_back(full_sort_view);
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, ExactSelectedIdsMayHaveDownstreamConsumers) {
  auto n = Route();
  auto* consumer = ggml_get_rows(c(), Tensor(2816, 128, 4), n[3]);
  kg::TensorArena::Bind(consumer, 0x20000000000ULL);
  n.push_back(consumer);
  EXPECT_TRUE(kg::GemmaRoutingFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, EveryScaledWeightedContributionAndPartialSumIsPrivate) {
  auto n = Reduce();
  for (std::size_t i = 0; i < n.size() - 1; ++i) {
    std::array keep{n[i]};
    EXPECT_FALSE(kg::GemmaReductionFusionAt(n, 0, keep)) << i;
  }
  std::array keep{n.back()};
  EXPECT_TRUE(kg::GemmaReductionFusionAt(n, 0, keep));
  auto* consumer = ggml_dup(c(), n[2]);
  kg::TensorArena::Bind(consumer, 0x20000000000ULL);
  n.push_back(consumer);
  EXPECT_FALSE(kg::GemmaReductionFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, RefusesMutatedRoutingArithmeticAndPhysicalPitch) {
  auto n = Route();
  const auto clamp = n[7]->op_params[0];
  n[7]->op_params[0] = 0;
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
  n[7]->op_params[0] = clamp;
  n[2]->op_params[0] = GGML_SORT_ORDER_ASC;
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
  n[2]->op_params[0] = GGML_SORT_ORDER_DESC;
  n[3]->nb[1] = 32;
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
  n[3]->nb[1] = 512;
  std::swap(n[8]->src[0], n[8]->src[1]);
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, RefusesMissingReversedOrReorderedContributions) {
  auto n = Reduce();
  std::swap(n[11]->src[0], n[11]->src[1]);
  EXPECT_FALSE(kg::GemmaReductionFusionAt(n, 0));
  std::swap(n[11]->src[0], n[11]->src[1]);
  n[9]->view_offs -= 2816 * 4;
  n[9]->data = static_cast<std::byte*>(n[9]->data) - 2816 * 4;
  EXPECT_FALSE(kg::GemmaReductionFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, RejectsActualInputsRootedInElidedOutputs) {
  auto n = Route();
  auto* logits = ggml_view_2d(c(), n[0], 128, 4, n[0]->nb[1], 0);
  logits->data = n[0]->data;
  n[0]->src[0] = logits;
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
  arena->Reset();
  n = Reduce();
  auto* scales = ggml_view_3d(c(), n[1], 1, 8, 4, 4, 32, 0);
  scales->data = n[1]->data;
  n[0]->src[1] = scales;
  EXPECT_FALSE(kg::GemmaReductionFusionAt(n, 0));
  n[0]->src[0] = n[1];
  EXPECT_FALSE(kg::GemmaReductionFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, ComputedViewCannotMasqueradeAsWrittenWeights) {
  auto n = Route();
  n[6]->view_src = n[8];
  n[6]->view_offs = 0;
  n[6]->data = n[8]->data;
  std::array keep{n[6]};
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0, keep));
}
TEST_F(GemmaMoeFusion, RefusesCyclesStaleViewsAndAddressOverflowBeforeReaderScans) {
  auto n = Route();
  auto* external = Tensor(4, 1);
  external->view_src = external;
  n.push_back(external);
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
  external->view_src = nullptr;
  kg::TensorArena::Bind(external, std::numeric_limits<std::uint64_t>::max() - 4);
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
  n.pop_back();
  n[3]->data = static_cast<std::byte*>(n[3]->data) + 4;
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0));
}
TEST_F(GemmaMoeFusion, PlacementFundsEveryDescriptorAndTheCompleteRoutingRoot) {
  const auto registry = llmp::execution::Registry::Create(kg::PlanningImplementations()).value();
  const pl::DeviceFacts facts{
      .device_class = "sm_121", .arch = 121, .features = llmp::execution::kBf16Mma};
  const pl::Policy policy{};
  for (bool route : {true, false}) {
    auto graph = route ? MakeRoute() : MakeReduce();
    ASSERT_TRUE(graph);
    co::StepGraph step{std::move(*graph), {}, {}, {}};
    for (const ir::TensorId in : step.graph.inputs()) {
      step.inputs.push_back({.tensor = in, .kind = co::InputKind::kFeatures});
    }
    const pl::StepShape shape{.segments = {{.rows = 4}}};
    auto plan = pl::Plan(step.graph.Clone(), registry, facts, policy,
                         {.written = &kg::WrittenBytes, .within = &kg::WithinRoot});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const kn::Addresses addresses{};
    const auto name = route ? kg::kGemmaRouteName : kg::kGemmaReduceName;

    auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, {});
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    ASSERT_EQ(lowered->plan.steps.size(), 1U);
    EXPECT_EQ(lowered->plan.steps.front().implementation, name);
    EXPECT_GT(lowered->extent, 0U);
    EXPECT_FALSE(lowered->nodes.empty());
    if (route) {
      EXPECT_EQ(lowered->plan.steps.front().nodes.size(), 10U);
      EXPECT_FALSE(lowered->physical_roots.empty());
    } else {
      EXPECT_EQ(lowered->plan.steps.front().nodes.size(), 17U);
    }
  }
}
}  // namespace

TEST_F(GemmaMoeFusion, IndexedPrivacyMatchesStandaloneKeepsReadersAndFlags) {
  for (const bool route : {false, true}) {
    arena->Reset();
    auto n = route ? Route() : Reduce();
    const auto compare = [&](std::span<ggml_tensor* const> keep) {
      const kg::detail::GraphReadIndex reads(n, keep);
      if (route)
        EXPECT_EQ(kg::GemmaRoutingFusionAt(n, 0, keep).has_value(),
                  kg::GemmaRoutingFusionAt(n, 0, keep, &reads).has_value());
      else
        EXPECT_EQ(kg::GemmaReductionFusionAt(n, 0, keep).has_value(),
                  kg::GemmaReductionFusionAt(n, 0, keep, &reads).has_value());
    };
    compare({});
    for (auto* t : n) {
      const std::array keep{t};
      compare(keep);
      t->flags |= GGML_TENSOR_FLAG_OUTPUT;
      compare({});
      t->flags &= ~GGML_TENSOR_FLAG_OUTPUT;
    }
    // Same address is not logical identity, including a different root kept
    // at an address activation placement may reuse.
    auto* unrelated = Tensor(4, 1);
    unrelated->data = n[0]->data;
    const std::array keep{unrelated};
    compare(keep);
    auto* view = ggml_view_tensor(c(), route ? n[3] : n[2]);
    view->data = view->view_src->data;
    const std::array kept_view{view};
    compare(kept_view);
    n.push_back(view);
    compare({});
    n.pop_back();
    auto* consumer = ggml_dup(c(), route ? n[3] : n[2]);
    kg::TensorArena::Bind(consumer, 0x20000000000ULL);
    n.push_back(consumer);
    compare({});
    unrelated->type = static_cast<ggml_type>(GGML_TYPE_COUNT);
    compare(keep);
    unrelated->type = GGML_TYPE_F32;
    unrelated->view_src = unrelated;
    compare(keep);
  }
}

TEST_F(GemmaMoeFusion, ReaderIndexPreservesProducerSkipDuplicateEdgesAndLogicalAliases) {
  auto* producer = Tensor(4, 1);
  auto* reader = ggml_add(c(), producer, producer);
  auto* final = ggml_dup(c(), reader);
  kg::TensorArena::Bind(reader, 0x20000000000ULL);
  kg::TensorArena::Bind(final, 0x20000001000ULL);
  std::vector<ggml_tensor*> nodes{producer, producer, reader, reader, final};
  producer->src[0] = producer;  // skipped producer's edges do not create a reader
  EXPECT_TRUE(kg::detail::GraphReadIndex(nodes, {}).OnlyReader(producer, reader));
  auto* unrelated = Tensor(4, 1);
  unrelated->data = producer->data;
  const std::array unrelated_keep{unrelated};
  EXPECT_TRUE(kg::detail::GraphReadIndex(nodes, unrelated_keep).OnlyReader(producer, reader));
  auto* view = ggml_view_tensor(c(), producer);
  view->data = producer->data;
  reader->src[1] = view;
  EXPECT_FALSE(kg::detail::GraphReadIndex(nodes, {}).OnlyReader(producer, reader));
  reader->src[1] = producer;
  const std::array kept_view{view};
  EXPECT_FALSE(kg::detail::GraphReadIndex(nodes, kept_view).OnlyReader(producer, reader));
  nodes.push_back(view);
  EXPECT_FALSE(kg::detail::GraphReadIndex(nodes, {}).OnlyReader(producer, reader));
  nodes.pop_back();
  nodes.back() = producer;
  EXPECT_FALSE(kg::detail::GraphReadIndex(nodes, {}).OnlyReader(producer, reader));
}

TEST_F(GemmaMoeFusion, StrictRootDepthAndPostPlacementChecksRemainIndependent) {
  auto* tensor = Tensor(4, 1);
  std::array<ggml_tensor, 65> chain;
  for (auto& t : chain) t = *tensor;
  for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
    chain[i].view_src = &chain[i + 1];
    chain[i].view_offs = 0;
  }
  const std::array nodes{&chain[0], &chain[1]};
  const kg::detail::GraphReadIndex reads(nodes, {});
  ASSERT_TRUE(reads.StrictRoot(&chain[1]));   // exactly 64 descriptors
  EXPECT_FALSE(reads.StrictRoot(&chain[0]));  // cached suffix cannot bypass depth65
  chain.back().view_src = &chain[0];
  EXPECT_FALSE(kg::detail::GraphReadIndex(nodes, {}).StrictRoot(&chain[1]));
  auto n = Route();
  const kg::detail::GraphReadIndex before(n, {});
  EXPECT_TRUE(kg::GemmaRoutingFusionAt(n, 0, {}, &before));
  n[3]->data = static_cast<std::byte*>(n[3]->data) + 4;
  const kg::detail::GraphReadIndex after(n, {});
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0, {}, &after));
  n[3]->data = static_cast<std::byte*>(n[3]->data) - 4;
  auto* unrelated = Tensor(4, 1);
  unrelated->type = static_cast<ggml_type>(GGML_TYPE_COUNT);
  const std::array keep{unrelated};
  const kg::detail::GraphReadIndex strict_bad(n, keep);
  // Weak logical-reader checks do not become strict MoE shape validation.
  EXPECT_TRUE(strict_bad.OnlyReader(n[0], n[1]) ==
              kg::detail::GraphReadIndex(n, {}).OnlyReader(n[0], n[1]));
  EXPECT_FALSE(kg::GemmaRoutingFusionAt(n, 0, keep, &strict_bad));
}

TEST_F(GemmaMoeFusion, LongUnrelatedLogicalViewDoesNotDisableLegacyNormFusion) {
  auto* x = Tensor(4, 1);
  auto* norm = ggml_rms_norm(c(), x, 1e-6f);
  auto* mul = ggml_mul(c(), norm, Tensor(4, 1));
  auto* output = ggml_scale(c(), mul, 2.0f);
  const std::array nodes{norm, mul, output};
  BindDistinct(nodes, 0x10000000000ULL);
  auto* unrelated = Tensor(4, 1);
  std::array<ggml_tensor, 65> chain;
  for (auto& t : chain) t = *unrelated;
  for (std::size_t i = 0; i + 1 < chain.size(); ++i) {
    chain[i].view_src = &chain[i + 1];
    chain[i].view_offs = 0;
  }
  const std::array keep{&chain[0]};
  const kg::detail::GraphReadIndex reads(nodes, keep);
  EXPECT_TRUE(reads.OnlyReader(norm, mul));
  EXPECT_FALSE(reads.StrictRoot(&chain[0]));
}
