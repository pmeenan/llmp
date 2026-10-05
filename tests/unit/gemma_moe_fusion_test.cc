// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_moe_fusion.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "base/check.h"
#include "gemma4_fixture.h"
#include "kernels/ggml/gemma4_graph.h"
#include "kernels/ggml/graph_plan.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
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
  std::vector<ggml_tensor*> Route(std::int64_t rows = 4) {
    auto* x = Tensor(128, rows);
    auto* p = ggml_soft_max(c(), x);
    auto* probabilities = ggml_reshape_3d(c(), p, 1, 128, rows);
    auto* ids = ggml_argsort_top_k(c(), p, 8);
    auto* gathered = ggml_get_rows(c(), probabilities, ids);
    auto* selected = ggml_reshape_2d(c(), gathered, 8, rows);
    auto* sum = ggml_sum_rows(c(), selected);
    auto* clamped =
        ggml_clamp(c(), sum, kg::kGemmaRouteClamp, std::numeric_limits<float>::infinity());
    auto* normalized = ggml_div(c(), selected, clamped);
    auto* out = ggml_reshape_3d(c(), normalized, 1, 8, rows);
    std::vector<ggml_tensor*> n{p,        probabilities, ids->view_src, ids,        gathered,
                                selected, sum,           clamped,       normalized, out};
    kg::BindDistinct(n, 0x10000000000ULL);
    return n;
  }
  std::vector<ggml_tensor*> Reduce(std::int64_t rows = 4) {
    auto* x = Tensor(2816, 8, rows);
    auto* scales = Tensor(1, 8, rows);
    auto* weights = Tensor(1, 8, rows);
    auto* scaled = ggml_mul(c(), x, scales);
    auto* weighted = ggml_mul(c(), scaled, weights);
    std::vector<ggml_tensor*> n{scaled, weighted};
    for (std::size_t i = 0; i < 8; ++i)
      n.push_back(ggml_view_2d(c(), weighted, 2816, rows, weighted->nb[2], i * weighted->nb[1]));
    auto* out = n[2];
    for (std::size_t i = 1; i < 8; ++i) {
      out = ggml_add(c(), out, n[2 + i]);
      n.push_back(out);
    }
    kg::BindDistinct(n, 0x10000000000ULL);
    return n;
  }
  std::unique_ptr<kg::TensorArena> arena;
  std::uint64_t address = 0x100000000ULL;
};
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
  for (bool route : {true, false}) {
    arena->Reset();
    auto n = route ? Route() : Reduce();
    kg::GraphPlan plan;
    // Metadata-only representative fused step: no execution selection is
    // implemented in this slice. All computed descriptors remain charged.
    plan.steps.push_back({.implementation = "metadata-only", .nodes = n});
    std::vector<ggml_tensor*> inputs;
    if (route)
      inputs.push_back(n[0]->src[0]);
    else
      inputs = {n[0]->src[0], n[0]->src[1], n[1]->src[1]};
    auto placement = kg::PlaceActivations(n, plan, inputs, 256);
    ASSERT_TRUE(placement);
    for (const auto [tensor, offset] : placement->offsets)
      kg::TensorArena::Bind(tensor, (1ULL << 32) + offset);
    kg::BindViews(n);
    if (route) {
      auto f = kg::GemmaRoutingFusionAt(n, 0);
      ASSERT_TRUE(f);
      EXPECT_EQ(f->operands.ids.bytes.value(), 4U * 512U);
      for (auto* written : {n[0], n[2], n[4], n[6], n[7], n[8]})
        EXPECT_TRUE(std::ranges::any_of(placement->offsets,
                                        [&](const auto& entry) { return entry.first == written; }));
    } else
      EXPECT_TRUE(kg::GemmaReductionFusionAt(n, 0));
  }
}
TEST(GemmaMoeActualGraph, RoutingMatchesAndInterleavedReductionRequiresContiguousScheduling) {
  namespace fixture = jitllm::test_support::gemma4;
  for (const auto size : {26U, 31U})
    for (const auto slots : {1U, 2U, 4U}) {
      const auto& p = size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B();
      auto binding = md::BindGemma4(p, "gemma4", fixture::Resources(size));
      ASSERT_TRUE(binding);
      auto state = md::Gemma4State(p, 4096, 16);
      ASSERT_TRUE(state);
      kg::Gemma4ChunkShape shape;
      for (std::uint32_t i = 0; i < slots; ++i)
        shape.segments.push_back({i, 2, i * 1100, ((i * 1100 + 2 + 255) / 256) * 256, 1280});
      shape.outputs = slots;
      auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(p, slots));
      ASSERT_TRUE(arena);
      auto g = kg::BuildGemma4Graph(*arena, p, *binding, *state, shape);
      ASSERT_TRUE(g);
      std::uint64_t address = 1ULL << 36;
      const auto bind = [&](ggml_tensor* t) {
        kg::TensorArena::Bind(t, address);
        address += ((ggml_nbytes(t) + 255) / 256 + 1) * 256;
      };
      for (auto* t : g->inputs) bind(t);
      for (const auto& w : g->weights) bind(w.tensor);
      for (const auto& segment : g->segments)
        for (const auto [k, v] : segment.caches)
          if (k) {
            bind(k);
            bind(v);
          }
      kg::BindDistinct(g->nodes, 1ULL << 48);
      std::size_t routing = 0, reduction = 0;
      for (std::size_t i = 0; i < g->nodes.size(); ++i) {
        routing += kg::GemmaRoutingFusionAt(g->nodes, i).has_value();
        reduction += kg::GemmaReductionFusionAt(g->nodes, i).has_value();
      }
      EXPECT_EQ(routing, size == 26 ? 30U : 0U) << size << "/" << slots;
      EXPECT_EQ(reduction, 0U) << size << "/" << slots;
      // The existing expanded graph interleaves the shared FFN before the
      // seven routed additions. No production order or dispatch changes here.
      // Prove the actual descriptors/reader graph meet the contract once that
      // independent work is scheduled outside the contiguous fused span.
      std::size_t contiguous_reductions = 0;
      for (std::size_t i = 0; i + 10 < g->nodes.size(); ++i) {
        auto* scaled = g->nodes[i];
        if (scaled->op != GGML_OP_MUL || scaled->ne[0] != 2816 || scaled->ne[1] != 8 ||
            g->nodes[i + 1]->op != GGML_OP_MUL || g->nodes[i + 1]->src[0] != scaled)
          continue;
        std::vector<ggml_tensor*> probe(g->nodes.begin() + static_cast<std::ptrdiff_t>(i),
                                        g->nodes.begin() + static_cast<std::ptrdiff_t>(i + 10));
        auto* previous = probe[2];
        for (std::size_t selected = 1; selected < 8; ++selected) {
          const auto found = std::ranges::find_if(g->nodes, [&](const auto* t) {
            return t->op == GGML_OP_ADD && t->src[0] == previous &&
                   t->src[1] == probe[2 + selected];
          });
          ASSERT_NE(found, g->nodes.end());
          previous = *found;
          probe.push_back(previous);
        }
        for (auto* t : g->nodes)
          if (std::ranges::find(probe, t) == probe.end()) probe.push_back(t);
        contiguous_reductions += kg::GemmaReductionFusionAt(probe, 0).has_value();
      }
      EXPECT_EQ(contiguous_reductions, size == 26 ? 30U : 0U) << size << "/" << slots;
    }
}
}  // namespace
