// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "dsv4_screen_source.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "components/step.h"
#include "dsv4_fixture.h"
#include "execution/registry.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "planner/pattern.h"
#include "planner/select.h"
#include "spec/mappings/dsv4.h"

namespace {
namespace ds = llmp::test_support::dsv4_screen;
namespace fx = llmp::test_support::dsv4;
namespace co = llmp::components;
namespace ir = llmp::ir;
namespace md = llmp::model;
namespace sp = llmp::spec;
namespace pl = llmp::planner;
namespace ex = llmp::execution;
using E = llmp::format::Encoding;
TEST(Dsv4ScreenSource, ServedHcPrePolicyUsesTheWholeStepRows) {
  pl::Policy policy;
  policy.enabled = {"dsv4_hc_post", "dsv4_hc_pre", "dsv4_lid_mask"};
  const auto original = policy;
  for (const auto rows : {1U, 8U, 9U, 512U}) {
    auto selected = ds::ServedPolicy(policy, rows);
    ASSERT_TRUE(pl::Validate(selected));
    EXPECT_EQ(pl::Disabled(selected, "dsv4_hc_pre"), rows > 8);
    EXPECT_FALSE(pl::Disabled(selected, "dsv4_hc_post"));
    EXPECT_FALSE(pl::Disabled(selected, "dsv4_lid_mask"));
    if (rows > 8) selected.enabled.insert(selected.enabled.begin() + 1, "dsv4_hc_pre");
    EXPECT_EQ(selected, original);
  }
  EXPECT_EQ(policy, original);
  std::erase(policy.enabled, "dsv4_hc_pre");
  EXPECT_EQ(ds::ServedPolicy(policy, 1), policy);  // An opt-out stays off.
  EXPECT_EQ(ds::ServedPolicy(policy, 512), policy);
}
TEST(Dsv4ScreenSource, RealComponentMaskIsExplicitlyReplacedAndCounted) {
  const auto& profile = md::Dsv4Flash();
  const auto source =
      fx::SourceOf("deepseek4", fx::Resources(profile), sp::mappings::Dsv4Mapping());
  const auto values = sp::mappings::Dsv4V0Values(profile);
  ASSERT_TRUE(values);
  auto spec = sp::Synthesize(sp::mappings::Dsv4Mapping(), source, *values);
  ASSERT_TRUE(spec) << (spec ? "" : spec.error());
  auto layout = co::StageLayout(*spec, 0, 3072, 512);
  ASSERT_TRUE(layout);
  for (const auto& [past, rows] : {std::pair{0U, 512U}, {512U, 512U}, {2048U, 512U}, {2560U, 1U}}) {
    pl::StepShape shape{
        .stage = 0,
        .phase = rows == 1 ? pl::Phase::kDecode : pl::Phase::kPrefill,
        .segments = {{.slot = 0,
                      .rows = rows,
                      .past = past,
                      .read_cells = std::max((profile.window + 512 + 255) / 256 * 256,
                                             (past + rows + 255) / 256 * 256),
                      .outputs = 1}}};
    auto step = co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
    ASSERT_TRUE(step) << (step ? "" : step.error());
    const auto count = std::ranges::count_if(
        step->graph.nodes(), [](const auto& n) { return n.key.op == ir::Op::kIndexScore; });
    ASSERT_GT(count, 0);
    if (past == 0) {
      auto bad = co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
      ASSERT_TRUE(bad);
      bool mutated = false;
      for (auto& input : bad->inputs)
        if (input.kind == co::InputKind::kCellPositions && input.cells.ring) {
          input.kind = co::InputKind::kFeatures;
          mutated = true;
        }
      ASSERT_TRUE(mutated);
      auto rejected = ds::RoundedRecipe(std::move(*bad), shape, *layout);
      ASSERT_FALSE(rejected);
      EXPECT_NE(rejected.error().find("owned ring positions"), std::string::npos);
      // An otherwise canonical selected mask cannot authorize a different
      // attention gather bound. Alter only that scalar in an actual step.
      auto wrong_bound =
          co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
      ASSERT_TRUE(wrong_bound);
      ir::Builder copy;
      std::vector<ir::TensorId> map(wrong_bound->graph.tensor_count());
      pl::RewriteContext rewrite(wrong_bound->graph, copy, map);
      std::size_t altered = 0;
      for (auto node : wrong_bound->graph.nodes()) {
        if (node.key == ir::Primitive(ir::Op::kAttention)) {
          auto params = node.Params<ir::AttnParams>();
          if (params.n_kv_max != 0) {
            ++params.n_kv_max;
            node.SetParams(params);
            ++altered;
          }
        }
        rewrite.Copy(node);
      }
      ASSERT_GT(altered, 0U);
      for (const auto t : wrong_bound->graph.outputs()) copy.Output(rewrite.Map(t));
      for (const auto t : wrong_bound->graph.keep()) copy.Keep(rewrite.Map(t));
      auto graph = std::move(copy).Finish();
      ASSERT_TRUE(graph);
      for (auto& input : wrong_bound->inputs) input.tensor = rewrite.Map(input.tensor);
      for (auto& weight : wrong_bound->weights) weight.tensor = rewrite.Map(weight.tensor);
      for (auto& state : wrong_bound->state) state.tensor = rewrite.Map(state.tensor);
      wrong_bound->logits = rewrite.Map(wrong_bound->logits);
      wrong_bound->greedy = rewrite.Map(wrong_bound->greedy);
      wrong_bound->features = rewrite.Map(wrong_bound->features);
      wrong_bound->graph = std::move(*graph);
      auto refused_bound = ds::RoundedRecipe(std::move(*wrong_bound), shape, *layout);
      ASSERT_FALSE(refused_bound);
      EXPECT_NE(refused_bound.error().find("proved visible-cell bound"), std::string::npos);
    }

    if (past == 0) {
      for (int control = 0; control < 4; ++control) {
        auto altered =
            co::BuildStep(*spec, 0, shape, {.tensors = source.tensors, .state = &*layout});
        ASSERT_TRUE(altered);
        bool touched = false;
        if (control == 0) {
          for (auto& input : altered->inputs)
            if (input.kind == co::InputKind::kCellPositions && input.cells.block == 128) {
              input.cells.positions = false;
              touched = true;
            }
        } else if (control == 1 || control == 2) {
          for (auto& state : altered->state)
            if (layout->items[state.item].contract.kind == md::StateKind::kCompressed &&
                layout->items[state.item].contract.window_or_ratio == 128) {
              if (control == 1)
                state.borrowed = true;
              else
                ++state.ref.layer;
              touched = true;
              break;
            }
        } else {
          for (auto& weight : altered->weights)
            if (weight.parts.size() == 1 &&
                weight.parts.front().first == "blk.3.attn_sinks.weight") {
              ++weight.ref.layer;
              touched = true;
            }
        }
        ASSERT_TRUE(touched) << control;
        auto refused = ds::RoundedRecipe(std::move(*altered), shape, *layout);
        EXPECT_FALSE(refused) << control;
        if (!refused) EXPECT_NE(refused.error().find("private HCA"), std::string::npos);
      }
    }
    auto changed = ds::RoundedRecipe(std::move(*step), shape, *layout);
    ASSERT_TRUE(changed) << (changed ? "" : changed.error());
    EXPECT_EQ(std::ranges::count_if(changed->graph.nodes(),
                                    [](const auto& n) { return n.key.op == ir::Op::kIndexScore; }),
              0);
    for (const auto op : {ir::Op::kRoundedIndexScore, ir::Op::kVisibleTopK, ir::Op::kSelectedMask})
      EXPECT_EQ(std::ranges::count_if(changed->graph.nodes(),
                                      [op](const auto& n) { return n.key.op == op; }),
                count);
    for (const auto& node : changed->graph.nodes())
      if (node.key.op == ir::Op::kVisibleTopK)
        EXPECT_EQ(node.Params<ir::TopKParams>().k,
                  std::min<std::uint64_t>(512, changed->graph.tensor(node.src[0]).ne(0)));
    std::size_t window_attention = 0, selected_attention = 0, other_attention = 0;
    for (const auto& node : changed->graph.nodes()) {
      if (node.key != ir::Primitive(ir::Op::kAttention)) continue;
      const auto& mask = changed->graph.node(changed->graph.producer(node.src[3]));
      const auto p = node.Params<ir::AttnParams>();
      if (mask.key == ir::Primitive(ir::Op::kMaskBuild)) {
        ++window_attention;
        EXPECT_TRUE(p.sparse_any);
        EXPECT_EQ(p.n_kv_max, profile.window);
        EXPECT_EQ(changed->graph.tensor(node.src[1]).ne(1), 768U);
      } else if (mask.key == ir::Primitive(ir::Op::kSelectedMask)) {
        ++selected_attention;
        EXPECT_TRUE(p.sparse_any);
        const auto selected = mask.Params<ir::SelectedMaskParams>();
        EXPECT_EQ(p.n_kv_max, profile.window + std::min(512U, selected.cells));
        EXPECT_EQ(changed->graph.tensor(node.src[1]).ne(1), selected.raw_cells + selected.cells);
        EXPECT_EQ(p.accumulate, ir::Accumulate::kF32);
        EXPECT_TRUE(p.sinks);
      } else {
        ++other_attention;
        EXPECT_TRUE(p.sparse_any);  // Authenticated HCA uses ordinary sparse MMA.
        const auto& compressed = changed->graph.tensor(mask.src[1]);
        EXPECT_EQ(p.n_kv_max, profile.window + compressed.ne(0));
      }
    }
    EXPECT_EQ(window_attention, 2U);
    EXPECT_EQ(selected_attention, static_cast<std::size_t>(count));
    EXPECT_GT(other_attention, 0U);
    const auto declaration = [](std::string name, bool accept) {
      return ex::Implementation{
          .name = std::move(name),
          .operation = ex::Operation::kFlashAttn,
          .source = "test",
          .revision = "1",
          .build = "cpu",
          .variant = "private recipe selection control",
          .capability = ex::Capability{
              .op = ir::Primitive(ir::Op::kAttention),
              .group = "test.attention",
              .accepts = accept ? nullptr : +[](const ex::CapabilityQuery&) { return false; }}};
    };
    auto registry = ex::Registry::Create({declaration("ggml.flash_attn_ext.mma", true),
                                          declaration("llmp.flash_attn_ext.mma_wide", true)});
    ASSERT_TRUE(registry);
    const auto ordinary = *registry->Find("ggml.flash_attn_ext.mma");
    const auto wide = *registry->Find("llmp.flash_attn_ext.mma_wide");
    const auto plan_of = [&] {
      pl::StepPlan result{.graph = changed->graph.Clone(),
                          .steps = {},
                          .regions = {},
                          .placement = {},
                          .applied = {},
                          .from = {},
                          .tensors = {},
                          .decomposed = {}};
      for (std::uint32_t i = 0; i < changed->graph.tensor_count(); ++i)
        result.tensors.push_back(ir::TensorId{i});
      for (std::uint32_t i = 0; i < changed->graph.nodes().size(); ++i)
        if (changed->graph.nodes()[i].key == ir::Primitive(ir::Op::kAttention))
          result.steps.push_back(
              {.nodes = {ir::NodeId{i}}, .implementation = wide, .group = "test.attention"});
      return result;
    };
    auto selected = plan_of();
    auto choose = ds::SelectHcaRecipe(selected, *changed, shape, *layout, *registry, {});
    ASSERT_TRUE(choose) << (choose ? "" : choose.error());
    EXPECT_EQ(std::ranges::count(selected.steps, ordinary, &pl::Selected::implementation),
              other_attention);
    EXPECT_EQ(std::ranges::count(selected.steps, wide, &pl::Selected::implementation),
              window_attention + selected_attention);
    if (past == 0) {
      auto absent = plan_of();
      absent.steps.clear();
      EXPECT_FALSE(ds::SelectHcaRecipe(absent, *changed, shape, *layout, *registry, {}));
      auto erased = plan_of();
      for (auto& tensor : erased.tensors) tensor = {};
      EXPECT_FALSE(ds::SelectHcaRecipe(erased, *changed, shape, *layout, *registry, {}));
      auto rejected_registry =
          ex::Registry::Create({declaration("ggml.flash_attn_ext.mma", false),
                                declaration("llmp.flash_attn_ext.mma_wide", true)});
      ASSERT_TRUE(rejected_registry);
      auto ineligible = plan_of();
      EXPECT_FALSE(
          ds::SelectHcaRecipe(ineligible, *changed, shape, *layout, *rejected_registry, {}));
    }

    std::size_t counts = 0;
    for (const auto& input : changed->inputs)
      if (input.kind == co::InputKind::kVisibleCells) {
        ++counts;
        auto result = co::InputValues(input, shape, {});
        ASSERT_TRUE(result);
        EXPECT_EQ(input.segment, 0U);
        EXPECT_EQ(input.cells.block, 4U);
        EXPECT_EQ(input.visible_capacity, 768U);
        ASSERT_EQ(result->size(), rows);
        EXPECT_EQ(result->front(), static_cast<std::int64_t>((past + 1) / 4));
        EXPECT_EQ(result->back(), static_cast<std::int64_t>((past + rows) / 4));
      }
    EXPECT_GT(counts, 0U);
  }
}
TEST(Dsv4ScreenSource, ClosedMaskMatcherRequiresSameCountsAndNoScoreObserver) {
  ex::Implementation impl{
      .name = "test.closed_lid_mask",
      .operation = ex::Operation::kLightningIndexer,
      .source = "test",
      .revision = "1",
      .build = "cpu",
      .variant = "test",
      .capability = ex::Capability{.op = ir::Fused(ir::FusedKind::kDsv4LidMask)}};
  auto registry = ex::Registry::Create({impl});
  ASSERT_TRUE(registry);
  for (int control = 0; control < 4; ++control) {
    ir::Builder b;
    b.SetOwner(0);
    const auto input = [&](E e, std::initializer_list<std::uint64_t> dims) {
      return b.Input(ir::Dense(e, ir::ShapeOf(dims)));
    };
    const auto q = input(E::kF32, {128, 64, 1});
    const auto k = input(E::kF16, {128, 1, 768});
    const auto w = input(E::kF32, {64, 1});
    const auto c = input(E::kI32, {1});
    const auto other = control == 1 ? input(E::kI32, {1}) : c;
    const auto raw = input(E::kF16, {640, 1});
    const auto score = b.RoundedIndexScore(q, k, w, c);
    const auto ids = b.VisibleTopK(score, other, 512);
    const auto mask = b.SelectedMask(raw, ids, {.raw_cells = 640, .cells = 768});
    b.Output(mask);
    if (control == 2) b.Output(score);
    if (control == 3) b.Output(ids);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    pl::Policy policy;
    policy.enabled = {"dsv4_lid_mask"};
    const pl::DeviceFacts device{};
    pl::SelectionView view(*registry, device, policy, false, false);
    auto changed = pl::ApplyPatterns(*graph, view);
    ASSERT_TRUE(changed);
    EXPECT_EQ(changed->applied.size(), control == 0 ? 1U : 0U) << control;
  }
}
TEST(Dsv4ScreenSource, ClosedMaskMatcherUsesOnlyAuthenticatedBoundaryAlignment) {
  ex::Capability cap{.op = ir::Fused(ir::FusedKind::kDsv4LidMask)};
  cap.accepts = [](const ex::CapabilityQuery& query) {
    // The real native recipe requires current-root facts. This predicate
    // deliberately discriminates the old descriptor-only matcher query.
    return query.operands.size() == 6 &&
           std::ranges::all_of(query.operands, [](const auto& operand) {
             return operand.memory && operand.memory->root_alignment >= 16;
           });
  };
  auto registry =
      ex::Registry::Create({ex::Implementation{.name = "test.aligned_lid_mask",
                                               .operation = ex::Operation::kLightningIndexer,
                                               .source = "test",
                                               .revision = "1",
                                               .build = "cpu",
                                               .variant = "test",
                                               .capability = cap}});
  ASSERT_TRUE(registry);
  for (const std::uint32_t cells : {256U, 512U, 768U}) {
    ir::Builder b;
    b.SetOwner(0);
    const auto input = [&](E e, std::initializer_list<std::uint64_t> dims) {
      return b.Input(ir::Dense(e, ir::ShapeOf(dims)));
    };
    const auto q = input(E::kF32, {128, 64, 512});
    const auto k = b.State(ir::Dense(E::kF16, ir::ShapeOf({128, 1, cells})),
                           {.stage = 0, .tensor = 0, .layer = 0, .segment = 0});
    const auto w = input(E::kF32, {64, 512});
    const auto counts = input(E::kI32, {512});
    const auto raw = input(E::kF16, {640, 512});
    const auto scores = b.RoundedIndexScore(q, k, w, counts);
    const auto ids = b.VisibleTopK(scores, counts, std::min(512U, cells));
    b.Output(b.SelectedMask(raw, ids, {.raw_cells = 640, .cells = cells}));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    pl::Policy policy;
    policy.enabled = {"dsv4_lid_mask"};
    for (const std::uint64_t alignment : {0U, 8U, 256U}) {
      const auto facts =
          alignment == 0
              ? pl::RootAlignment{}
              : pl::GraphRootAlignments(*graph, std::array{pl::RootAlignmentFact{k, alignment}});
      const pl::DeviceFacts device{};
      pl::SelectionView view(*registry, device, policy, false, false, {}, facts,
                             {.base = 256, .offset = 256});
      auto changed = pl::ApplyPatterns(*graph, view);
      ASSERT_TRUE(changed);
      EXPECT_EQ(changed->applied.size(), alignment == 256 ? 1U : 0U)
          << "cells=" << cells << " alignment=" << alignment;
    }
  }
}
TEST(Dsv4ScreenSource, FormulaHasHonestPrimitiveFallbackAndRejectsCellMismatch) {
  for (const std::uint32_t cells : {768U, 767U}) {
    ir::Builder b;
    b.SetOwner(0);
    const auto input = [&](E e, std::initializer_list<std::uint64_t> dims) {
      return b.Input(ir::Dense(e, ir::ShapeOf(dims)));
    };
    const std::array sources{input(E::kF32, {128, 64, 1}), input(E::kF16, {128, 1, 768}),
                             input(E::kF32, {64, 1}), input(E::kI32, {1}),
                             input(E::kF16, {640, 1})};
    const auto out = ir::Decompose(
        b, ir::Fused(ir::FusedKind::kDsv4LidMask),
        ir::ToBytes(ir::LidMaskParams{.top = 512, .raw_cells = 640, .cells = cells}), sources, {});
    if (out.n) b.Output(out[0]);
    auto graph = std::move(b).Finish();
    EXPECT_EQ(graph.has_value(), cells == 768);
    if (graph) {
      EXPECT_EQ(graph->tensor(graph->outputs()[0]).ne(0), 1408U);
      for (const auto op :
           {ir::Op::kRoundedIndexScore, ir::Op::kVisibleTopK, ir::Op::kSelectedMask})
        EXPECT_EQ(
            std::ranges::count_if(graph->nodes(), [op](const auto& n) { return n.key.op == op; }),
            1);
    }
  }
}
}  // namespace
