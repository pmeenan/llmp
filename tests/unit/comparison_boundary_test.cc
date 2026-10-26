// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "components/comparison_boundary.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/step.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "planner/step_shape.h"
#include "spec/spec.h"

namespace {
namespace base = llmp::base;
namespace components = llmp::components;
namespace co = llmp::components;
namespace planner = llmp::planner;
namespace fmt = llmp::format;
namespace ir = llmp::ir;
namespace spec = llmp::spec;

class Funder final : public ir::StorageFunder {
 public:
  bool Charge(std::uint64_t bytes) const override {
    ++charge_count;
    if (!accept || charge_count == fail_at) return false;
    used += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const override {
    EXPECT_LE(bytes, used);
    used -= bytes;
  }
  bool accept = true;
  mutable std::uint64_t used = 0;
  mutable std::uint64_t charge_count = 0;
  std::uint64_t fail_at = 0;
};

ir::Graph TwoInputGraph() {
  ir::Builder builder;
  const auto first = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  const auto second = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  builder.Output(first);
  builder.Output(second);
  return std::move(builder).Finish().value();
}

ir::Graph TwoInputGraphWithExtraOutput() {
  ir::Builder builder;
  const auto first = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  const auto second = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  builder.Output(first);
  builder.Output(second);
  builder.Output(first);
  return std::move(builder).Finish().value();
}

ir::Graph TwoInputGraphWithComputedOutput() {
  ir::Builder builder;
  const auto first = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  const auto second = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  builder.Output(builder.Add(first, second));
  return std::move(builder).Finish().value();
}

std::array<co::ComparisonExternalBinding, 2> External(const ir::Graph& graph, std::uint32_t group) {
  return {co::ComparisonExternalBinding{.original_root = graph.inputs()[0],
                                        .allocation_class = group,
                                        .allocation_bytes = 64,
                                        .bytes = 64,
                                        .alignment = 16},
          co::ComparisonExternalBinding{.original_root = graph.inputs()[1],
                                        .allocation_class = group,
                                        .allocation_bytes = 64,
                                        .bytes = 64,
                                        .alignment = 16}};
}

TEST(ComparisonBoundaryTest, CanonicalizesBindingOrderAndClassLabels) {
  auto graph = TwoInputGraph();
  auto external = External(graph, 3);
  const co::ComparisonBoundaryView view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  const auto bound = co::ComparisonBoundaryStorageBound(view);
  ASSERT_TRUE(bound) << (bound ? "" : bound.error().detail);
  auto funder = std::make_shared<Funder>();
  auto first = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(first) << (first ? "" : first.error().detail);
  const auto digest = first->digest();

  external[0].allocation_class = 88;
  external[1].allocation_class = 88;
  std::swap(external[0], external[1]);
  const co::ComparisonBoundaryView relabeled{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto second = co::MakeComparisonBoundary(relabeled, funder);
  ASSERT_TRUE(second) << (second ? "" : second.error().detail);
  EXPECT_EQ(digest.combined, second->digest().combined);

  external[1].allocation_class = 89;
  const co::ComparisonBoundaryView distinct{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto third = co::MakeComparisonBoundary(distinct, funder);
  ASSERT_TRUE(third) << (third ? "" : third.error().detail);
  EXPECT_NE(digest.combined, third->digest().combined);
}

TEST(ComparisonBoundaryTest, ExternalResourceIdentityChangesCombinedBoundary) {
  auto graph = TwoInputGraph();
  auto external = External(graph, 1);
  const co::ComparisonBoundaryView view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto funder = std::make_shared<Funder>();
  auto original = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(original) << (original ? "" : original.error().detail);
  const auto original_digest = original->digest().combined;
  const base::Sha256Digest identity{};
  external[0].resource_part_identity = identity;
  const co::ComparisonBoundaryView changed{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto observed = co::MakeComparisonBoundary(changed, funder);
  ASSERT_TRUE(observed) << (observed ? "" : observed.error().detail);
  EXPECT_NE(original_digest, observed->digest().combined);

  external[0].resource_part_identity.reset();
  external[0].allocation_bytes = 128;
  external[1].allocation_bytes = 128;
  external[0].offset = 16;
  const co::ComparisonBoundaryView offset_changed{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto shifted = co::MakeComparisonBoundary(offset_changed, funder);
  ASSERT_TRUE(shifted) << (shifted ? "" : shifted.error().detail);
  EXPECT_NE(original_digest, shifted->digest().combined);
}

TEST(ComparisonBoundaryTest, RowStoragePresenceIsAnExplicitBoundaryField) {
  const ir::WeightRef ref{.role = 17};
  ir::Builder builder;
  const auto table = builder.Weight(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16, 4})), ref);
  builder.Output(table);
  auto graph = std::move(builder).Finish().value();
  const std::array<co::StepWeight, 1> weights{
      co::StepWeight{.tensor = table, .ref = ref, .parts = {{"table", fmt::PartRole::kData}}}};
  const co::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), weights};
  base::Sha256Digest identity{};
  identity[0] = 1;
  co::ComparisonExternalBinding binding{.original_root = table,
                                        .part = fmt::PartRole::kData,
                                        .allocation_class = 1,
                                        .allocation_bytes = 256,
                                        .bytes = 256,
                                        .alignment = 16,
                                        .resource_part_identity = identity};
  const auto without_storage = co::ComparisonBoundaryView{io, {}, 0, std::span(&binding, 1)};
  auto funder = std::make_shared<Funder>();
  auto ordinary = co::MakeComparisonBoundary(without_storage, funder);
  ASSERT_TRUE(ordinary) << (ordinary ? "" : ordinary.error().detail);

  binding.row_storage = co::ComparisonRowStorage{
      .ref = ref,
      .part = fmt::PartRole::kData,
      .logical_format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16, 4})),
      .content_identity = identity,
      .resource = 0,
      .group = 0,
      .shard = 0,
      .group_bytes = 256,
      .resource_offset = 0,
      .resource_bytes = 256,
      .logical_rows = 4,
      .slot_capacity = 4,
      .row_bytes = 64};
  const auto with_storage = co::ComparisonBoundaryView{io, {}, 0, std::span(&binding, 1)};
  auto row_backed = co::MakeComparisonBoundary(with_storage, funder);
  ASSERT_TRUE(row_backed) << (row_backed ? "" : row_backed.error().detail);
  EXPECT_NE(ordinary->digest().combined, row_backed->digest().combined);
}

TEST(ComparisonBoundaryTest, TypedFoldedInputMustReplaceExactlyOneExternalInputFact) {
  auto graph = TwoInputGraphWithComputedOutput();
  auto external = External(graph, 1);
  const std::array<co::ComparisonExternalBinding, 1> backed_second{external[1]};
  const std::array<co::ComparisonUnstoredInput, 1> folded_first{co::ComparisonUnstoredInput{
      .original_input = graph.inputs()[0],
      .reason = co::ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks}};
  auto funder = std::make_shared<Funder>();
  const co::ComparisonBoundaryView folded_view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, backed_second, folded_first};
  auto folded = co::MakeComparisonBoundary(folded_view, funder);
  ASSERT_TRUE(folded) << (folded ? "" : folded.error().detail);
  ASSERT_EQ(folded->unstored_inputs().size(), 1U);
  EXPECT_EQ(folded->unstored_inputs()[0].original_input, graph.inputs()[0]);
  EXPECT_EQ(folded->unstored_inputs()[0].reason,
            co::ComparisonUnstoredInputReason::kCellPositionsFoldedIntoMasks);

  const co::ComparisonBoundaryView stored_view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external, {}};
  auto stored = co::MakeComparisonBoundary(stored_view, funder);
  ASSERT_TRUE(stored) << (stored ? "" : stored.error().detail);
  EXPECT_NE(folded->digest().combined, stored->digest().combined);

  const std::array<co::ComparisonExternalBinding, 2> mixed_external{external[0], external[1]};
  const co::ComparisonBoundaryView mixed{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, mixed_external, folded_first};
  const auto expect_rejected = [&](const co::ComparisonBoundaryView& view) {
    auto rejected = co::MakeComparisonBoundary(view, funder);
    ASSERT_FALSE(rejected);
  };
  expect_rejected(mixed);

  const std::array<co::ComparisonUnstoredInput, 2> duplicate{folded_first[0], folded_first[0]};
  const co::ComparisonBoundaryView duplicate_view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, backed_second, duplicate};
  expect_rejected(duplicate_view);

  const std::array<co::ComparisonUnstoredInput, 1> wrong_reader{
      co::ComparisonUnstoredInput{.original_input = ir::TensorId{99}}};
  const co::ComparisonBoundaryView wrong_view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, backed_second, wrong_reader};
  expect_rejected(wrong_view);

  const std::array<co::ComparisonUnstoredInput, 1> forged_reason{
      co::ComparisonUnstoredInput{.original_input = graph.inputs()[0],
                                  .reason = static_cast<co::ComparisonUnstoredInputReason>(255)}};
  const co::ComparisonBoundaryView forged_view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, backed_second, forged_reason};
  expect_rejected(forged_view);

  auto observable = TwoInputGraph();
  auto observable_external = External(observable, 1);
  const std::array<co::ComparisonExternalBinding, 1> observable_backed{observable_external[1]};
  const std::array<co::ComparisonUnstoredInput, 1> observable_unstored{
      co::ComparisonUnstoredInput{.original_input = observable.inputs()[0]}};
  const co::ComparisonBoundaryView observable_view{
      {observable, observable.inputs(), observable.outputs(), {}},
      {},
      0,
      observable_backed,
      observable_unstored};
  expect_rejected(observable_view);
}

TEST(ComparisonBoundaryTest, FoldedInputMappingMustRemainTheMaskCellPositionOperand) {
  const auto make_graph = [](bool reverse_operands) {
    ir::Builder builder;
    const auto cells = builder.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({16})));
    const auto positions = builder.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({16})));
    const auto mask =
        reverse_operands ? builder.MaskBuild(cells, positions, 16, {.kind = ir::MaskKind::kCausal})
                         : builder.MaskBuild(positions, cells, 16, {.kind = ir::MaskKind::kCausal});
    builder.Output(mask);
    return std::move(builder).Finish().value();
  };
  auto source = make_graph(false);
  const std::array<co::ComparisonExternalBinding, 1> external{
      co::ComparisonExternalBinding{.original_root = source.inputs()[1],
                                    .allocation_class = 1,
                                    .allocation_bytes = 64,
                                    .bytes = 64,
                                    .alignment = 16}};
  const std::array<co::ComparisonUnstoredInput, 1> folded{
      co::ComparisonUnstoredInput{.original_input = source.inputs()[0]}};
  const co::ComparisonBoundaryView view{
      {source, source.inputs(), source.outputs(), {}}, {}, 0, external, folded};
  auto funder = std::make_shared<Funder>();
  auto boundary = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  const std::array<ir::TensorId, 3> identity{ir::TensorId{0}, ir::TensorId{1}, ir::TensorId{2}};
  auto mapped = co::ValidateMappedComparisonBoundary(*boundary, source, identity);
  ASSERT_TRUE(mapped) << (mapped ? "" : mapped.error().detail);

  auto wrong_reader = make_graph(true);
  mapped = co::ValidateMappedComparisonBoundary(*boundary, wrong_reader, identity);
  ASSERT_FALSE(mapped);
  EXPECT_NE(mapped.error().detail.find("non-mask reader"), std::string::npos);
}

TEST(ComparisonBoundaryTest, MappingMustPreserveOriginalRootAliasPartition) {
  auto graph = TwoInputGraph();
  auto external = External(graph, 1);
  const co::ComparisonBoundaryView view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto funder = std::make_shared<Funder>();
  auto boundary = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);

  ir::Builder builder;
  const auto shared = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  builder.Output(shared);
  builder.Output(shared);
  auto planned = std::move(builder).Finish().value();
  const std::array<ir::TensorId, 2> collapsed{ir::TensorId{0}, ir::TensorId{0}};
  auto result = co::ValidateMappedComparisonBoundary(*boundary, planned, collapsed);
  EXPECT_FALSE(result);
  EXPECT_NE(result.error().detail.find("interface"), std::string::npos);

  const std::array<ir::TensorId, 1> short_map{ir::TensorId{0}};
  result = co::ValidateMappedComparisonBoundary(*boundary, planned, short_map);
  EXPECT_FALSE(result);
  EXPECT_NE(result.error().detail.find("domain size"), std::string::npos);
}

TEST(ComparisonBoundaryTest, PaysRetainedOwnerAndRejectsFunderRefusal) {
  auto graph = TwoInputGraph();
  auto external = External(graph, 1);
  const co::ComparisonBoundaryView view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto funder = std::make_shared<Funder>();
  funder->accept = false;
  auto refused = co::MakeComparisonBoundary(view, funder);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, co::ComparisonBoundaryError::kAdmission);
  EXPECT_EQ(funder->used, 0U);

  funder->accept = true;
  {
    auto retained = co::MakeComparisonBoundary(view, funder);
    ASSERT_TRUE(retained) << (retained ? "" : retained.error().detail);
    EXPECT_GT(funder->used, 0U);
    EXPECT_EQ(retained->digest().version, 2U);
  }
  EXPECT_EQ(funder->used, 0U);
}

TEST(ComparisonBoundaryTest, RegionBoundaryCanonicalizesLocalMembersAndDerivesCrossings) {
  ir::Builder builder;
  const auto x = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto y = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto first = builder.Add(x, y);
  const auto second = builder.Add(x, y);
  const auto tail = builder.Mul(first, y);
  const auto second_tail = builder.Mul(second, x);
  builder.Output(tail);
  builder.Output(second_tail);
  auto graph_result = std::move(builder).Finish();
  ASSERT_TRUE(graph_result) << graph_result.error().detail;
  ir::Graph graph = std::move(*graph_result);
  const components::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
  const auto external = External(graph, 4);
  const components::ComparisonBoundaryView source{io, {}, 0, external};
  const std::array<std::uint32_t, 2> one_group{0, 1};
  const std::array<std::uint32_t, 3> two_groups{0, 1, 2};
  const std::array<ir::NodeId, 1> first_member{graph.producer(first)};
  const std::array<ir::NodeId, 1> second_member{graph.producer(second)};
  const auto funder = std::make_shared<Funder>();

  auto first_region =
      components::MakeComparisonRegionBoundary({source, one_group, first_member}, funder);
  ASSERT_TRUE(first_region) << first_region.error().detail;
  EXPECT_EQ(first_region->source_identity(), graph.identity());
  EXPECT_EQ(first_region->nodes().size(), 1U);
  ASSERT_EQ(first_region->ports().size(), 3U);
  EXPECT_EQ(first_region->ports()[0].kind, components::ComparisonRegionPortKind::kInput);
  EXPECT_EQ(first_region->ports()[1].kind, components::ComparisonRegionPortKind::kInput);
  EXPECT_EQ(first_region->ports()[2].kind, components::ComparisonRegionPortKind::kOutput);

  auto second_region =
      components::MakeComparisonRegionBoundary({source, one_group, second_member}, funder);
  ASSERT_TRUE(second_region) << second_region.error().detail;
  EXPECT_EQ(first_region->digest().logical, second_region->digest().logical);
  ASSERT_EQ(first_region->nodes().size(), 1U);
  ASSERT_EQ(second_region->nodes().size(), 1U);
  EXPECT_EQ(first_region->nodes()[0].original_node, first_member[0]);
  EXPECT_EQ(second_region->nodes()[0].original_node, second_member[0]);
  EXPECT_NE(first_region->nodes()[0].original_node, second_region->nodes()[0].original_node);

  ir::Builder shifted_builder;
  const auto unrelated_x = shifted_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto unrelated_y = shifted_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto unrelated = shifted_builder.Add(unrelated_x, unrelated_y);
  const auto shifted_x = shifted_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto shifted_y = shifted_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto shifted_sum = shifted_builder.Add(shifted_x, shifted_y);
  const auto shifted_tail = shifted_builder.Mul(shifted_sum, shifted_x);
  shifted_builder.Output(shifted_tail);
  shifted_builder.Output(unrelated);
  auto shifted_result = std::move(shifted_builder).Finish();
  ASSERT_TRUE(shifted_result) << shifted_result.error().detail;
  ir::Graph shifted_graph = std::move(*shifted_result);
  const components::GraphBuildIO shifted_io{
      shifted_graph, shifted_graph.inputs(), shifted_graph.outputs(), {}};
  const std::array<components::ComparisonExternalBinding, 4> shifted_external{
      components::ComparisonExternalBinding{.original_root = shifted_graph.inputs()[0],
                                            .allocation_class = 10,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = shifted_graph.inputs()[1],
                                            .allocation_class = 11,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = shifted_graph.inputs()[2],
                                            .allocation_class = 12,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = shifted_graph.inputs()[3],
                                            .allocation_class = 13,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16}};
  const components::ComparisonBoundaryView shifted_source{shifted_io, {}, 0, shifted_external};
  const std::array<ir::NodeId, 1> shifted_member{shifted_graph.producer(shifted_sum)};
  auto shifted_region =
      components::MakeComparisonRegionBoundary({shifted_source, one_group, shifted_member}, funder);
  ASSERT_TRUE(shifted_region) << shifted_region.error().detail;
  EXPECT_EQ(second_region->digest().logical, shifted_region->digest().logical);
  ASSERT_EQ(shifted_region->nodes().size(), 1U);
  EXPECT_NE(second_region->nodes()[0].original_node, shifted_region->nodes()[0].original_node);

  const std::array<ir::NodeId, 2> pair{graph.producer(first), graph.producer(tail)};
  auto pair_region = components::MakeComparisonRegionBoundary({source, two_groups, pair}, funder);
  ASSERT_TRUE(pair_region) << pair_region.error().detail;
  EXPECT_NE(first_region->digest().logical, pair_region->digest().logical);
  EXPECT_GT(funder->used, 0U);

  ir::Builder keep_builder;
  const auto keep_x = keep_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto keep_y = keep_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto kept = keep_builder.Add(keep_x, keep_y);
  keep_builder.Keep(kept);
  auto keep_graph_result = std::move(keep_builder).Finish();
  ASSERT_TRUE(keep_graph_result) << keep_graph_result.error().detail;
  ir::Graph keep_graph = std::move(*keep_graph_result);
  const components::GraphBuildIO keep_io{keep_graph, keep_graph.inputs(), keep_graph.outputs(), {}};
  const auto keep_external = External(keep_graph, 8);
  const components::ComparisonBoundaryView keep_source{keep_io, {}, 0, keep_external};
  const std::array<ir::NodeId, 1> keep_member{keep_graph.producer(kept)};
  auto keep_region =
      components::MakeComparisonRegionBoundary({keep_source, one_group, keep_member}, funder);
  ASSERT_TRUE(keep_region) << keep_region.error().detail;
  ASSERT_EQ(keep_region->ports().size(), 3U);
  EXPECT_EQ(keep_region->ports()[2].kind, components::ComparisonRegionPortKind::kKeep);
}

TEST(ComparisonBoundaryTest, CanonicalRoleAndComponentLabelsIgnoreOccurrencesNotSemantics) {
  EXPECT_EQ(components::CanonicalComparisonRole(ir::Role::kInput),
            components::ComparisonPersistentRole::kValue);
  EXPECT_EQ(components::CanonicalComparisonRole(ir::Role::kActivation),
            components::ComparisonPersistentRole::kValue);
  EXPECT_EQ(components::CanonicalComparisonRole(ir::Role::kOutput),
            components::ComparisonPersistentRole::kValue);
  EXPECT_EQ(components::CanonicalComparisonRole(ir::Role::kWeight),
            components::ComparisonPersistentRole::kWeight);
  EXPECT_EQ(components::CanonicalComparisonRole(ir::Role::kState),
            components::ComparisonPersistentRole::kState);
  EXPECT_EQ(components::CanonicalComparisonRole(ir::Role::kConstant),
            components::ComparisonPersistentRole::kConstant);

  const std::array<components::ComparisonRegionNode, 5> nodes{
      components::ComparisonRegionNode{.component = 42},
      components::ComparisonRegionNode{.component = components::kFrameworkComponent},
      components::ComparisonRegionNode{.component = 7},
      components::ComparisonRegionNode{.component = 42},
      components::ComparisonRegionNode{.component = components::kFrameworkComponent}};
  EXPECT_EQ(components::CanonicalComparisonComponent(nodes, 0), 0U);
  EXPECT_EQ(components::CanonicalComparisonComponent(nodes, 1), components::kFrameworkComponent);
  EXPECT_EQ(components::CanonicalComparisonComponent(nodes, 2), 1U);
  EXPECT_EQ(components::CanonicalComparisonComponent(nodes, 3), 0U);
  EXPECT_EQ(components::CanonicalComparisonComponent(nodes, 4), components::kFrameworkComponent);
  EXPECT_EQ(components::CanonicalComparisonComponent(nodes, nodes.size()), UINT32_MAX);

  ir::Builder direct_builder;
  const auto direct_x = direct_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto direct_y = direct_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  direct_builder.SetComponent(3);
  const auto direct_sum = direct_builder.Add(direct_x, direct_y);
  direct_builder.Output(direct_sum);
  auto direct_graph_result = std::move(direct_builder).Finish();
  ASSERT_TRUE(direct_graph_result) << direct_graph_result.error().detail;
  ir::Graph direct_graph = std::move(*direct_graph_result);

  ir::Builder internal_builder;
  const auto source_x = internal_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto source_y = internal_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto internal_x = internal_builder.Scale(source_x, 1.0F, 0.0F);
  const auto internal_y = internal_builder.Scale(source_y, 1.0F, 0.0F);
  internal_builder.SetComponent(91);
  const auto internal_sum = internal_builder.Add(internal_x, internal_y);
  const auto tail = internal_builder.Scale(internal_sum, 1.0F, 1.0F);
  internal_builder.Output(tail);
  auto internal_graph_result = std::move(internal_builder).Finish();
  ASSERT_TRUE(internal_graph_result) << internal_graph_result.error().detail;
  ir::Graph internal_graph = std::move(*internal_graph_result);

  const std::array<std::uint32_t, 2> offsets{0, 1};
  const std::array<ir::NodeId, 1> direct_member{direct_graph.producer(direct_sum)};
  const std::array<ir::NodeId, 1> internal_member{internal_graph.producer(internal_sum)};
  const auto direct_external = External(direct_graph, 11);
  const auto internal_external = External(internal_graph, 99);
  const components::ComparisonBoundaryView direct_source{
      {direct_graph, direct_graph.inputs(), direct_graph.outputs(), {}}, {}, 0, direct_external};
  const components::ComparisonBoundaryView internal_source{
      {internal_graph, internal_graph.inputs(), internal_graph.outputs(), {}},
      {},
      0,
      internal_external};
  auto funder = std::make_shared<Funder>();
  auto direct =
      components::MakeComparisonRegionBoundary({direct_source, offsets, direct_member}, funder);
  ASSERT_TRUE(direct) << direct.error().detail;
  auto internal =
      components::MakeComparisonRegionBoundary({internal_source, offsets, internal_member}, funder);
  ASSERT_TRUE(internal) << internal.error().detail;
  EXPECT_NE(direct->source_identity(), internal->source_identity());
  EXPECT_EQ(direct->digest().logical, internal->digest().logical);
  EXPECT_EQ(direct->digest().external, internal->digest().external);
  EXPECT_EQ(direct->digest().combined, internal->digest().combined);
  ASSERT_EQ(direct->ports().size(), 3U);
  ASSERT_EQ(internal->ports().size(), 3U);
  EXPECT_EQ(direct->ports()[0].kind, components::ComparisonRegionPortKind::kInput);
  EXPECT_EQ(internal->ports()[0].kind, components::ComparisonRegionPortKind::kInput);
  EXPECT_EQ(direct->ports()[2].kind, components::ComparisonRegionPortKind::kOutput);
  EXPECT_EQ(internal->ports()[2].kind, components::ComparisonRegionPortKind::kOutput);
}

TEST(ComparisonBoundaryTest, RealComponentAndExpandedPrimitiveRegionSharePortableIdentity) {
  const spec::ComponentRef norm_ref{
      .kind = "norm.rms",
      .version = 1,
      .params = {spec::Param{.name = "eps", .value = 1e-6}},
  };
  auto instance = components::MakeInstance(norm_ref, {.width = 4}, components::Site::kLayer, 7, 3);
  ASSERT_TRUE(instance) << instance.error();
  const planner::StepShape shape{.segments = {{.rows = 1, .outputs = 1}}};
  const std::array weight_formats{ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4}))};
  auto built = components::BuildComponent(*instance, shape, weight_formats, {});
  ASSERT_TRUE(built) << built.error();
  const auto built_io = components::GraphIO(built->step);

  const auto built_weight_ref = built->step.weights.front().ref;
  ir::Builder expanded_builder;
  const auto expanded_input =
      expanded_builder.Input(built_io.graph.tensor(built->stream_in).format);
  auto expanded_weight_ref = built_weight_ref;
  expanded_weight_ref.component = 91;
  expanded_weight_ref.layer = 99;
  const auto expanded_weight = expanded_builder.Weight(
      built_io.graph.tensor(built->weights.front()).format, expanded_weight_ref);
  expanded_builder.SetComponent(91);
  const auto normalized = expanded_builder.RmsNorm(expanded_input, 1e-6F);
  const auto expanded_output = expanded_builder.Mul(normalized, expanded_weight);
  expanded_builder.Output(expanded_output);
  ir::Graph expanded_graph = std::move(expanded_builder).Finish().value();
  const std::array<components::StepWeight, 1> expanded_weights{
      components::StepWeight{.tensor = expanded_weight,
                             .ref = expanded_weight_ref,
                             .parts = {{"renamed.norm.weight", fmt::PartRole::kData}}}};
  const components::GraphBuildIO expanded_io{expanded_graph, expanded_graph.inputs(),
                                             expanded_graph.outputs(), expanded_weights};

  const auto make_external = [](const components::GraphBuildIO& io) {
    std::vector<components::ComparisonExternalBinding> external;
    std::uint32_t allocation_class = 1;
    const auto add = [&](ir::TensorId tensor, std::optional<fmt::PartRole> part,
                         std::uint64_t bytes) {
      const auto root = io.graph.root(tensor);
      external.push_back({.original_root = root,
                          .part = part,
                          .allocation_class = allocation_class++,
                          .allocation_bytes = bytes,
                          .bytes = bytes,
                          .alignment = 1});
    };
    for (const auto input : io.inputs) {
      const auto bytes = ir::StorageBytes(io.graph.tensor(io.graph.root(input)));
      if (bytes) add(input, std::nullopt, *bytes);
    }
    for (const auto& weight : io.weights) {
      const auto parts = fmt::PartsOf(io.graph.tensor(io.graph.root(weight.tensor)).format);
      if (!parts) continue;
      for (const auto& part : parts->parts()) add(weight.tensor, part.role, part.bytes);
    }
    return external;
  };
  auto built_external = make_external(built_io);
  auto expanded_external = make_external(expanded_io);
  const auto built_members = [&] {
    std::vector<ir::NodeId> nodes;
    for (std::uint32_t n = 0; n < built_io.graph.nodes().size(); ++n)
      if (built_io.graph.node(ir::NodeId{n}).key.op == ir::Op::kRmsNorm ||
          built_io.graph.node(ir::NodeId{n}).key.op == ir::Op::kMul)
        nodes.push_back(ir::NodeId{n});
    return nodes;
  }();
  const std::array<ir::NodeId, 2> expanded_members{expanded_graph.producer(normalized),
                                                   expanded_graph.producer(expanded_output)};
  ASSERT_EQ(built_members.size(), 2U);
  const std::array<std::uint32_t, 2> offsets{0, 2};
  const auto built_view = components::ComparisonBoundaryView{built_io, {}, 0, built_external};
  const auto expanded_view =
      components::ComparisonBoundaryView{expanded_io, {}, 0, expanded_external};
  auto funder = std::make_shared<Funder>();
  auto built_region =
      components::MakeComparisonRegionBoundary({built_view, offsets, built_members}, funder);
  ASSERT_TRUE(built_region) << built_region.error().detail;
  auto expanded_region =
      components::MakeComparisonRegionBoundary({expanded_view, offsets, expanded_members}, funder);
  ASSERT_TRUE(expanded_region) << expanded_region.error().detail;
  EXPECT_NE(built_region->source_identity(), expanded_region->source_identity());
  EXPECT_EQ(built_region->digest().logical, expanded_region->digest().logical);
  EXPECT_EQ(built_region->digest().external, expanded_region->digest().external);
  EXPECT_EQ(built_region->digest().combined, expanded_region->digest().combined);
}

TEST(ComparisonBoundaryTest, BuildStepActivationMatchesExpandedFusedProducer) {
  spec::Stage stage{
      .name = "target", .role = spec::StageRole::kTarget, .artifact = 0, .context = 16};
  stage.embedding = spec::ComponentRef{.kind = "embedding.token",
                                       .version = 1,
                                       .params = {{.name = "scale", .value = 1.0}},
                                       .weights = {{.role = "table", .tensor = "table"}}};
  stage.stream.width = 4;
  stage.layers = {{{spec::ComponentRef{.kind = "norm.rms",
                                       .version = 1,
                                       .params = {{.name = "eps", .value = 1.0e-5}},
                                       .weights = {{.role = "weight", .tensor = "gamma"}}}}}};
  stage.heads.output = spec::ComponentRef{
      .kind = "head.linear", .version = 1, .weights = {{.role = "weight", .tensor = "head"}}};
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kLlm, .stages = {std::move(stage)}};
  spec::Canonicalize(model);
  const std::array<spec::SourceTensor, 3> tensors{
      spec::SourceTensor{.name = "table",
                         .format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 5}))},
      spec::SourceTensor{.name = "gamma",
                         .format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4}))},
      spec::SourceTensor{.name = "head",
                         .format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 5}))}};
  const planner::StepShape shape{.segments = {{.rows = 1, .outputs = 1}}};
  auto built = components::BuildStep(model, 0, shape, {.tensors = tensors});
  ASSERT_TRUE(built) << built.error();

  std::optional<ir::NodeId> norm_node, mul_node;
  for (std::uint32_t i = 0; i < built->graph.nodes().size(); ++i) {
    const ir::NodeId id{i};
    const auto& node = built->graph.node(id);
    if (node.key == ir::Primitive(ir::Op::kRmsNorm)) norm_node = id;
    if (node.key == ir::Primitive(ir::Op::kMul) && node.nsrc == 2 && norm_node &&
        node.src[0] == built->graph.node(*norm_node).out[0])
      mul_node = id;
  }
  ASSERT_TRUE(norm_node.has_value());
  ASSERT_TRUE(mul_node.has_value());
  const auto& norm = built->graph.node(*norm_node);
  ASSERT_EQ(built->graph.tensor(norm.src[0]).role, ir::Role::kActivation);
  const auto& mul = built->graph.node(*mul_node);
  const ir::TensorId norm_weight =
      built->graph.tensor(mul.src[0]).role == ir::Role::kWeight ? mul.src[0] : mul.src[1];
  ASSERT_EQ(built->graph.tensor(norm_weight).role, ir::Role::kWeight);
  const auto* weight_ref = std::get_if<ir::WeightRef>(&built->graph.tensor(norm_weight).binding);
  ASSERT_NE(weight_ref, nullptr);

  ir::Builder fused_builder;
  const auto fused_input = fused_builder.Input(built->graph.tensor(norm.src[0]).format);
  const auto fused_weight =
      fused_builder.Weight(built->graph.tensor(norm_weight).format, *weight_ref);
  fused_builder.SetComponent(norm.component);
  const std::array fused_sources{fused_input, fused_weight};
  const auto fused_output = fused_builder.Fused(ir::FusedKind::kRmsNormMul, fused_sources,
                                                ir::NormParams{.eps = 1.0e-5F});
  fused_builder.Output(fused_output[0]);
  auto fused = std::move(fused_builder).Finish();
  ASSERT_TRUE(fused) << fused.error().detail;
  auto expanded = ir::Expand(*fused, false);
  ASSERT_TRUE(expanded) << expanded.error().detail;

  const auto source_view = [](const components::GraphBuildIO& io) {
    std::vector<components::ComparisonExternalBinding> external;
    std::uint32_t allocation_class = 1;
    for (const auto input : io.inputs) {
      const auto root = io.graph.root(input);
      const auto bytes = ir::StorageBytes(io.graph.tensor(root));
      if (!bytes) continue;
      external.push_back({.original_root = root,
                          .allocation_class = allocation_class++,
                          .allocation_bytes = *bytes,
                          .bytes = *bytes,
                          .alignment = 1});
    }
    for (const auto& weight : io.weights) {
      const auto root = io.graph.root(weight.tensor);
      const auto parts = fmt::PartsOf(io.graph.tensor(root).format);
      if (!parts) continue;
      for (std::size_t i = 0; i < parts->size(); ++i) {
        const auto& part = parts->parts()[i];
        external.push_back({.original_root = root,
                            .part = part.role,
                            .allocation_class = allocation_class++,
                            .allocation_bytes = part.bytes,
                            .bytes = part.bytes,
                            .alignment = 1});
      }
    }
    return external;
  };

  const components::GraphBuildIO built_io{built->graph, built->graph.inputs(),
                                          built->graph.outputs(), built->weights};
  const auto& expanded_mul = expanded->node(expanded->producer(expanded->outputs()[0]));
  const ir::TensorId expanded_weight =
      expanded->tensor(expanded_mul.src[0]).role == ir::Role::kWeight ? expanded_mul.src[0]
                                                                      : expanded_mul.src[1];
  const auto expanded_ref = std::get<ir::WeightRef>(expanded->tensor(expanded_weight).binding);
  const std::array<components::StepWeight, 1> expanded_weights{
      components::StepWeight{.tensor = expanded_weight,
                             .ref = expanded_ref,
                             .parts = {{"renamed.weight.part", fmt::PartRole::kData}}}};
  const components::GraphBuildIO expanded_io{*expanded, expanded->inputs(), expanded->outputs(),
                                             expanded_weights};
  auto built_external = source_view(built_io);
  auto expanded_external = source_view(expanded_io);
  const components::ComparisonBoundaryView built_view{built_io, {}, 0, built_external};
  const components::ComparisonBoundaryView expanded_view{expanded_io, {}, 0, expanded_external};
  const std::array<std::uint32_t, 2> offsets{0, 2};
  const std::array<ir::NodeId, 2> built_members{*norm_node, *mul_node};
  std::optional<ir::NodeId> expanded_norm, expanded_output_node;
  for (std::uint32_t i = 0; i < expanded->nodes().size(); ++i) {
    const ir::NodeId id{i};
    const auto& node = expanded->node(id);
    if (node.key == ir::Primitive(ir::Op::kRmsNorm)) expanded_norm = id;
    if (id == expanded->producer(expanded->outputs()[0])) expanded_output_node = id;
  }
  ASSERT_TRUE(expanded_norm.has_value());
  ASSERT_TRUE(expanded_output_node.has_value());
  const std::array<ir::NodeId, 2> expanded_members{*expanded_norm, *expanded_output_node};
  auto funder = std::make_shared<Funder>();
  auto built_region =
      components::MakeComparisonRegionBoundary({built_view, offsets, built_members}, funder);
  ASSERT_TRUE(built_region) << built_region.error().detail;
  auto expanded_region =
      components::MakeComparisonRegionBoundary({expanded_view, offsets, expanded_members}, funder);
  ASSERT_TRUE(expanded_region) << expanded_region.error().detail;
  EXPECT_EQ(built_region->digest().logical, expanded_region->digest().logical);
  EXPECT_EQ(built_region->digest().external, expanded_region->digest().external);
  EXPECT_EQ(built_region->digest().combined, expanded_region->digest().combined);

  const auto variant_digest = [&](fmt::Encoding encoding,
                                  std::optional<std::array<float, 4>> constant_values)
      -> std::optional<base::Sha256Digest> {
    ir::Builder builder;
    const auto input = builder.Input(built->graph.tensor(norm.src[0]).format);
    ir::TensorId weight{};
    std::vector<components::StepWeight> weights;
    if (constant_values) {
      weight = builder.Constant(ir::Dense(encoding, ir::ShapeOf({4})), "constant.gamma",
                                std::as_bytes(std::span(*constant_values)));
    } else {
      weight = builder.Weight(ir::Dense(encoding, ir::ShapeOf({4})), *weight_ref);
      weights.push_back({.tensor = weight,
                         .ref = *weight_ref,
                         .parts = {{"variant.gamma", fmt::PartRole::kData}}});
    }
    builder.SetComponent(norm.component);
    const auto normalized = builder.RmsNorm(input, 1.0e-5F);
    const auto output = builder.Mul(normalized, weight);
    builder.Output(output);
    auto graph_result = std::move(builder).Finish();
    if (!graph_result) return std::nullopt;
    ir::Graph graph = std::move(*graph_result);
    const components::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), weights};
    auto external = source_view(io);
    const components::ComparisonBoundaryView view{io, {}, 0, external};
    const std::array<ir::NodeId, 2> members{graph.producer(normalized), graph.producer(output)};
    auto region = components::MakeComparisonRegionBoundary({view, offsets, members}, funder);
    if (!region) return std::nullopt;
    return region->digest().logical;
  };
  const auto primitive_f32 = variant_digest(fmt::Encoding::kF32, std::nullopt);
  ASSERT_TRUE(primitive_f32.has_value());
  EXPECT_EQ(*primitive_f32, expanded_region->digest().logical);
  const auto primitive_f16 = variant_digest(fmt::Encoding::kF16, std::nullopt);
  ASSERT_TRUE(primitive_f16.has_value());
  EXPECT_NE(*primitive_f16, *primitive_f32);
  const std::array<float, 4> first_constant{0.5F, 0.75F, 1.0F, 1.25F};
  const std::array<float, 4> second_constant{0.5F, 0.75F, 1.0F, 1.5F};
  const auto constant_a = variant_digest(fmt::Encoding::kF32, first_constant);
  const auto constant_b = variant_digest(fmt::Encoding::kF32, second_constant);
  ASSERT_TRUE(constant_a.has_value());
  ASSERT_TRUE(constant_b.has_value());
  EXPECT_NE(*constant_a, *constant_b);
}

TEST(ComparisonBoundaryTest, RegionBoundaryCanonicalizesWeightDeclarationOrder) {
  ir::Builder builder;
  const auto input = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const ir::WeightRef first_ref{.stage = 0, .component = 1, .role = 2};
  const ir::WeightRef second_ref{.stage = 0, .component = 2, .role = 1};
  const auto first_weight =
      builder.Weight(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), first_ref);
  const auto second_weight =
      builder.Weight(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), second_ref);
  const auto first = builder.Mul(input, first_weight);
  const auto second = builder.Mul(input, second_weight);
  builder.Output(first);
  builder.Output(second);
  ir::Graph graph = std::move(builder).Finish().value();
  const components::StepWeight w0{
      .tensor = first_weight, .ref = first_ref, .parts = {{"data0", fmt::PartRole::kData}}};
  const components::StepWeight w1{
      .tensor = second_weight, .ref = second_ref, .parts = {{"data1", fmt::PartRole::kData}}};
  const std::array<components::StepWeight, 2> forward_weights{w0, w1};
  const std::array<components::StepWeight, 2> reverse_weights{w1, w0};
  const std::array<components::ComparisonExternalBinding, 3> external{
      components::ComparisonExternalBinding{.original_root = input,
                                            .allocation_class = 10,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = first_weight,
                                            .part = fmt::PartRole::kData,
                                            .allocation_class = 11,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = second_weight,
                                            .part = fmt::PartRole::kData,
                                            .allocation_class = 12,
                                            .allocation_bytes = 64,
                                            .bytes = 64,
                                            .alignment = 16}};
  const components::ComparisonBoundaryView forward{
      {graph, graph.inputs(), graph.outputs(), forward_weights}, {}, 0, external};
  const components::ComparisonBoundaryView reverse{
      {graph, graph.inputs(), graph.outputs(), reverse_weights}, {}, 0, external};
  const std::array<std::uint32_t, 2> groups{0, 2};
  const std::array<ir::NodeId, 2> members{graph.producer(first), graph.producer(second)};
  auto funder = std::make_shared<Funder>();
  auto a = components::MakeComparisonRegionBoundary({forward, groups, members}, funder);
  ASSERT_TRUE(a) << a.error().detail;
  auto b = components::MakeComparisonRegionBoundary({reverse, groups, members}, funder);
  ASSERT_TRUE(b) << b.error().detail;
  EXPECT_EQ(a->digest().logical, b->digest().logical);
}

TEST(ComparisonBoundaryTest, RegionBoundaryIncludesDeclaredStateReadEvenWhenProducerIsMember) {
  ir::Builder builder;
  const auto state_ref = ir::StateRef{.stage = 0, .tensor = 3, .layer = 70000, .segment = 0};
  const auto state = builder.State(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), state_ref);
  const auto x = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto sum = builder.Add(state, x);
  builder.Output(sum);
  auto graph_result = std::move(builder).Finish();
  ASSERT_TRUE(graph_result) << graph_result.error().detail;
  ir::Graph graph = std::move(*graph_result);
  const std::array state_facts{
      components::StepState{.tensor = state, .ref = state_ref, .read = true, .written = false}};
  const components::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
  const std::array<components::ComparisonExternalBinding, 2> external{
      components::ComparisonExternalBinding{.original_root = state,
                                            .allocation_class = 1,
                                            .allocation_bytes = 16,
                                            .bytes = 16,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = x,
                                            .allocation_class = 2,
                                            .allocation_bytes = 16,
                                            .bytes = 16,
                                            .alignment = 16}};
  const components::ComparisonBoundaryView source{io, state_facts, 0, external};
  const std::array<std::uint32_t, 2> offsets{0, 2};
  const std::array<ir::NodeId, 2> members{graph.producer(state), graph.producer(sum)};
  auto funder = std::make_shared<Funder>();
  auto region = components::MakeComparisonRegionBoundary({source, offsets, members}, funder);
  ASSERT_TRUE(region) << region.error().detail;
  ASSERT_EQ(region->state().size(), 1U);
  EXPECT_EQ(region->state()[0].facts.ref, state_ref);
  EXPECT_TRUE(std::ranges::any_of(region->ports(), [](const auto& port) {
    return port.kind == components::ComparisonRegionPortKind::kStateRead;
  }));
  auto deny_state_workspace = std::make_shared<Funder>();
  deny_state_workspace->fail_at = 3;  // validation, retained owner, then state-label scratch
  auto refused_state_workspace =
      components::MakeComparisonRegionBoundary({source, offsets, members}, deny_state_workspace);
  ASSERT_FALSE(refused_state_workspace);
  EXPECT_EQ(refused_state_workspace.error().code, components::ComparisonBoundaryError::kAdmission);
  EXPECT_EQ(deny_state_workspace->charge_count, 3U);
  EXPECT_EQ(deny_state_workspace->used, 0U);

  const components::ComparisonBoundaryView other_stage{io, state_facts, 1, external};
  auto stage_region =
      components::MakeComparisonRegionBoundary({other_stage, offsets, members}, funder);
  ASSERT_TRUE(stage_region) << stage_region.error().detail;
  EXPECT_NE(region->digest().logical, stage_region->digest().logical);
  EXPECT_EQ(stage_region->active_stage(), 1U);

  auto borrowed_from_active = state_facts;
  borrowed_from_active[0].borrowed = true;
  borrowed_from_active[0].source = 0;
  const components::ComparisonBoundaryView active_source{io, borrowed_from_active, 0, external};
  auto active_region =
      components::MakeComparisonRegionBoundary({active_source, offsets, members}, funder);
  ASSERT_TRUE(active_region) << active_region.error().detail;
  auto borrowed_from_other = borrowed_from_active;
  borrowed_from_other[0].source = 1;
  const components::ComparisonBoundaryView other_source{io, borrowed_from_other, 0, external};
  auto other_region =
      components::MakeComparisonRegionBoundary({other_source, offsets, members}, funder);
  ASSERT_TRUE(other_region) << other_region.error().detail;
  EXPECT_NE(active_region->digest().logical, other_region->digest().logical);

  const auto renamed_stage_digest =
      [&](std::uint16_t stage_id, std::uint16_t source_id) -> std::optional<base::Sha256Digest> {
    ir::Builder stage_builder;
    const ir::StateRef ref{.stage = stage_id, .tensor = 3, .layer = 70000, .segment = 0};
    const auto stage_state =
        stage_builder.State(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), ref);
    const auto stage_input = stage_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
    const auto stage_sum = stage_builder.Add(stage_state, stage_input);
    stage_builder.Output(stage_sum);
    auto stage_graph_result = std::move(stage_builder).Finish();
    if (!stage_graph_result) return std::nullopt;
    ir::Graph stage_graph = std::move(*stage_graph_result);
    const std::array stage_facts{components::StepState{.tensor = stage_state,
                                                       .ref = ref,
                                                       .item = 0,
                                                       .borrowed = true,
                                                       .source = source_id,
                                                       .read = true}};
    const components::GraphBuildIO stage_io{
        stage_graph, stage_graph.inputs(), stage_graph.outputs(), {}};
    const std::array<components::ComparisonExternalBinding, 2> stage_external{
        components::ComparisonExternalBinding{.original_root = stage_state,
                                              .allocation_class = 1,
                                              .allocation_bytes = 16,
                                              .bytes = 16,
                                              .alignment = 16},
        components::ComparisonExternalBinding{.original_root = stage_input,
                                              .allocation_class = 2,
                                              .allocation_bytes = 16,
                                              .bytes = 16,
                                              .alignment = 16}};
    const components::ComparisonBoundaryView stage_view{stage_io, stage_facts, stage_id,
                                                        stage_external};
    const std::array<ir::NodeId, 2> stage_members{stage_graph.producer(stage_state),
                                                  stage_graph.producer(stage_sum)};
    auto stage_region =
        components::MakeComparisonRegionBoundary({stage_view, offsets, stage_members}, funder);
    if (!stage_region) return std::nullopt;
    return stage_region->digest().logical;
  };
  const auto original_stage_labels = renamed_stage_digest(0, 1);
  const auto renamed_stage_labels = renamed_stage_digest(5, 6);
  ASSERT_TRUE(original_stage_labels.has_value());
  ASSERT_TRUE(renamed_stage_labels.has_value());
  EXPECT_EQ(*original_stage_labels, *renamed_stage_labels);

  auto wrong_facts = state_facts;
  wrong_facts[0].read = false;
  const components::ComparisonBoundaryView wrong_source{io, wrong_facts, 0, external};
  auto refused = components::MakeComparisonRegionBoundary({wrong_source, offsets, members}, funder);
  EXPECT_FALSE(refused);
  EXPECT_EQ(refused.error().code, components::ComparisonBoundaryError::kInvalid);
  EXPECT_NE(refused.error().detail.find("state read/write flags differ:"), std::string::npos);

  struct RefusingDiagnostic final : ir::DiagnosticAdmission {
    bool TryChargeDiagnostic(std::uint64_t bytes) override {
      last_bytes = bytes;
      return false;
    }
    std::uint64_t last_bytes = 0;
  } refusing_diagnostic;
  auto refused_diagnostic = components::MakeComparisonRegionBoundary(
      {wrong_source, offsets, members}, funder, &refusing_diagnostic);
  ASSERT_FALSE(refused_diagnostic);
  EXPECT_EQ(refused_diagnostic.error().code, components::ComparisonBoundaryError::kAdmission);
  EXPECT_TRUE(refused_diagnostic.error().detail.empty());
  EXPECT_GT(refusing_diagnostic.last_bytes, refused.error().detail.size());

  auto denying = std::make_shared<Funder>();
  denying->fail_at = 1;
  auto denied_validation =
      components::MakeComparisonRegionBoundary({wrong_source, offsets, members}, denying);
  ASSERT_FALSE(denied_validation);
  EXPECT_EQ(denied_validation.error().code, components::ComparisonBoundaryError::kAdmission);
  EXPECT_EQ(denying->charge_count, 1U);
  EXPECT_EQ(denying->used, 0U);

  ir::Builder input_store_builder;
  const auto target = input_store_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto values = input_store_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto index = input_store_builder.Input(ir::Dense(fmt::Encoding::kI64, ir::ShapeOf({1})));
  input_store_builder.CacheStore(target, values, index);
  input_store_builder.Output(values);
  auto input_store_result = std::move(input_store_builder).Finish();
  ASSERT_TRUE(input_store_result) << input_store_result.error().detail;
  ir::Graph input_store_graph = std::move(*input_store_result);
  std::array<components::ComparisonExternalBinding, 3> input_store_external{};
  for (std::size_t i = 0; i < input_store_external.size(); ++i)
    input_store_external[i] = {.original_root = input_store_graph.inputs()[i],
                               .allocation_class = static_cast<std::uint32_t>(i + 1),
                               .allocation_bytes = 64,
                               .bytes = 64,
                               .alignment = 16};
  const components::ComparisonBoundaryView input_store_source{
      {input_store_graph, input_store_graph.inputs(), input_store_graph.outputs(), {}},
      {},
      0,
      input_store_external};
  const std::array<ir::NodeId, 1> input_store_member{
      input_store_graph.producer(input_store_graph.outputs()[0])};
  const std::array<std::uint32_t, 2> input_store_offsets{0, 1};
  auto deny_empty_state = std::make_shared<Funder>();
  deny_empty_state->fail_at = 1;
  auto denied_empty_state = components::MakeComparisonRegionBoundary(
      {input_store_source, input_store_offsets, input_store_member}, deny_empty_state);
  ASSERT_FALSE(denied_empty_state);
  EXPECT_EQ(denied_empty_state.error().code, components::ComparisonBoundaryError::kAdmission);
  EXPECT_EQ(deny_empty_state->charge_count, 1U);
  EXPECT_EQ(deny_empty_state->used, 0U);
}

TEST(ComparisonBoundaryTest, CanonicalLabelsPreserveFullWidthEqualityRelations) {
  struct StateLabels {
    std::array<std::uint32_t, 2> layers;
    std::array<std::size_t, 2> items;
  };
  const auto digest = [](const StateLabels& labels) -> std::optional<base::Sha256Digest> {
    ir::Builder builder;
    const std::array refs{
        ir::StateRef{.stage = 0, .tensor = 3, .layer = labels.layers[0], .segment = 0},
        ir::StateRef{.stage = 0, .tensor = 4, .layer = labels.layers[1], .segment = 0}};
    const auto first = builder.State(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), refs[0]);
    const auto second = builder.State(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), refs[1]);
    const auto sum = builder.Add(first, second);
    builder.Output(sum);
    auto graph_result = std::move(builder).Finish();
    if (!graph_result) return std::nullopt;
    ir::Graph graph = std::move(*graph_result);
    const std::array facts{
        components::StepState{
            .tensor = first, .ref = refs[0], .item = labels.items[0], .read = true},
        components::StepState{
            .tensor = second, .ref = refs[1], .item = labels.items[1], .read = true}};
    const components::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
    const std::array<components::ComparisonExternalBinding, 2> external{
        components::ComparisonExternalBinding{.original_root = first,
                                              .allocation_class = 1,
                                              .allocation_bytes = 16,
                                              .bytes = 16,
                                              .alignment = 16},
        components::ComparisonExternalBinding{.original_root = second,
                                              .allocation_class = 2,
                                              .allocation_bytes = 16,
                                              .bytes = 16,
                                              .alignment = 16}};
    const components::ComparisonBoundaryView view{io, facts, 0, external};
    const std::array<std::uint32_t, 2> offsets{0, 3};
    const std::array<ir::NodeId, 3> members{graph.producer(first), graph.producer(second),
                                            graph.producer(sum)};
    auto funder = std::make_shared<Funder>();
    auto region = components::MakeComparisonRegionBoundary({view, offsets, members}, funder);
    if (!region) return std::nullopt;
    return region->digest().logical;
  };
  const std::size_t high_item =
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  const StateLabels high{.layers = {70000, 135536}, .items = {0, high_item}};
  const StateLabels equivalent{.layers = {70000, 70001}, .items = {0, 1}};
  const auto high_digest = digest(high);
  const auto equivalent_digest = digest(equivalent);
  ASSERT_TRUE(high_digest.has_value());
  ASSERT_TRUE(equivalent_digest.has_value());
  EXPECT_EQ(*high_digest, *equivalent_digest);
}

TEST(ComparisonBoundaryTest, RegionExternalPartitionIsCanonicalAndFundingFailureIsAtomic) {
  ir::Builder builder;
  const auto x = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto y = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const auto sum = builder.Add(x, y);
  builder.Output(sum);
  auto graph_result = std::move(builder).Finish();
  ASSERT_TRUE(graph_result) << graph_result.error().detail;
  ir::Graph graph = std::move(*graph_result);
  const components::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), {}};
  const auto valid_external = External(graph, 13);
  const components::ComparisonBoundaryView valid_source{io, {}, 0, valid_external};
  const std::array<std::uint32_t, 2> offsets{0, 1};
  const std::array<ir::NodeId, 1> members{graph.producer(sum)};
  auto funder = std::make_shared<Funder>();

  std::array<components::ComparisonExternalBinding, 2> together{
      components::ComparisonExternalBinding{.original_root = x,
                                            .allocation_class = 31,
                                            .allocation_bytes = 64,
                                            .bytes = 16,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = y,
                                            .allocation_class = 31,
                                            .allocation_bytes = 64,
                                            .offset = 16,
                                            .bytes = 16,
                                            .alignment = 16}};
  auto aliased =
      components::MakeComparisonRegionBoundary({{io, {}, 0, together}, offsets, members}, funder);
  ASSERT_TRUE(aliased) << aliased.error().detail;
  std::swap(together[0], together[1]);
  together[0].allocation_class = 900;
  together[1].allocation_class = 900;
  auto relabeled =
      components::MakeComparisonRegionBoundary({{io, {}, 0, together}, offsets, members}, funder);
  ASSERT_TRUE(relabeled) << relabeled.error().detail;
  EXPECT_EQ(aliased->digest().combined, relabeled->digest().combined);

  together[0].allocation_class = 900;
  together[1].allocation_class = 901;
  auto disjoint =
      components::MakeComparisonRegionBoundary({{io, {}, 0, together}, offsets, members}, funder);
  ASSERT_TRUE(disjoint) << disjoint.error().detail;
  EXPECT_EQ(aliased->digest().external, disjoint->digest().external);
  EXPECT_EQ(aliased->digest().combined, disjoint->digest().combined);
  ASSERT_EQ(disjoint->external().size(), 2U);
  EXPECT_NE(disjoint->external()[0].canonical_allocation_class,
            disjoint->external()[1].canonical_allocation_class);

  auto denying = std::make_shared<Funder>();
  denying->fail_at = 2;
  auto refused =
      components::MakeComparisonRegionBoundary({valid_source, offsets, members}, denying);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, components::ComparisonBoundaryError::kAdmission);
  EXPECT_EQ(denying->charge_count, 2U);
  EXPECT_EQ(denying->used, 0U);
}

TEST(ComparisonBoundaryTest, RegionRequiresCompleteWeightPartsAndReadableExtents) {
  ir::Builder builder;
  const auto x = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})));
  const ir::WeightRef ref{.stage = 1, .component = 2, .role = 3, .layer = 4};
  const auto weight = builder.Weight(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4})), ref);
  const auto product = builder.Mul(x, weight);
  builder.Output(product);
  auto graph_result = std::move(builder).Finish();
  ASSERT_TRUE(graph_result) << graph_result.error().detail;
  ir::Graph graph = std::move(*graph_result);
  const std::array<components::StepWeight, 1> weights{components::StepWeight{
      .tensor = weight, .ref = ref, .parts = {{"data", fmt::PartRole::kData}}}};
  const components::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), weights};
  const std::array<components::ComparisonExternalBinding, 2> external{
      components::ComparisonExternalBinding{.original_root = x,
                                            .allocation_class = 1,
                                            .allocation_bytes = 16,
                                            .bytes = 16,
                                            .alignment = 16},
      components::ComparisonExternalBinding{.original_root = weight,
                                            .part = fmt::PartRole::kData,
                                            .allocation_class = 2,
                                            .allocation_bytes = 16,
                                            .bytes = 16,
                                            .alignment = 16}};
  const components::ComparisonBoundaryView source{io, {}, 0, external};
  const std::array<std::uint32_t, 2> offsets{0, 1};
  const std::array<ir::NodeId, 1> members{graph.producer(product)};
  auto funder = std::make_shared<Funder>();
  auto valid = components::MakeComparisonRegionBoundary({source, offsets, members}, funder);
  ASSERT_TRUE(valid) << valid.error().detail;

  const std::array<components::StepWeight, 1> duplicate_parts{components::StepWeight{
      .tensor = weight,
      .ref = ref,
      .parts = {{"data", fmt::PartRole::kData}, {"duplicate", fmt::PartRole::kData}}}};
  const components::GraphBuildIO malformed_io{graph, graph.inputs(), graph.outputs(),
                                              duplicate_parts};
  auto malformed = components::MakeComparisonRegionBoundary(
      {{malformed_io, {}, 0, external}, offsets, members}, funder);
  ASSERT_FALSE(malformed);
  EXPECT_EQ(malformed.error().code, components::ComparisonBoundaryError::kInvalid);

  auto short_external = external;
  short_external[1].bytes = 1;
  auto truncated = components::MakeComparisonRegionBoundary(
      {{io, {}, 0, short_external}, offsets, members}, funder);
  ASSERT_FALSE(truncated);
  EXPECT_EQ(truncated.error().code, components::ComparisonBoundaryError::kInvalid);
}

TEST(ComparisonBoundaryTest, RowStorageRequiresExactTypedOwnerFactAndCapacityExtent) {
  ir::Builder builder;
  const auto ids = builder.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({4})));
  const fmt::TensorFormat table_format{.encoding = fmt::Encoding::kF32,
                                       .layout = fmt::Layout::kDense,
                                       .shape = {.ne = {16, 64, 1, 1}, .rank = 2}};
  const ir::WeightRef ref{.stage = 2, .component = 7, .role = 3, .layer = 11};
  const auto table = builder.Weight(table_format, ref);
  builder.Output(builder.GetRows(table, ids));
  auto graph_result = std::move(builder).Finish();
  ASSERT_TRUE(graph_result) << graph_result.error().detail;
  const ir::Graph graph = std::move(*graph_result);
  const std::array weights{
      co::StepWeight{.tensor = table, .ref = ref, .parts = {{"table", fmt::PartRole::kData}}}};
  auto funder = std::make_shared<Funder>();
  auto make_boundary = [&](co::ComparisonRowStorage storage, std::uint64_t extent,
                           std::optional<base::Sha256Digest> resource_identity = {}) {
    std::array external{
        co::ComparisonExternalBinding{.original_root = ids,
                                      .allocation_class = 1,
                                      .allocation_bytes = 16,
                                      .bytes = 16,
                                      .alignment = 4},
        co::ComparisonExternalBinding{
            .original_root = table,
            .part = fmt::PartRole::kData,
            .allocation_class = 2,
            .allocation_bytes = extent,
            .bytes = extent,
            .alignment = 16,
            .resource_part_identity = resource_identity.value_or(storage.content_identity),
            .row_storage = storage}};
    const co::ComparisonBoundaryView view{
        {graph, graph.inputs(), graph.outputs(), weights}, {}, 2, external};
    return co::MakeComparisonBoundary(view, funder);
  };
  co::ComparisonRowStorage storage{.ref = ref,
                                   .part = fmt::PartRole::kData,
                                   .logical_format = table_format,
                                   .resource = 5,
                                   .group = 9,
                                   .shard = 3,
                                   .group_bytes = 4096,
                                   .resource_offset = 0,
                                   .resource_bytes = 4096,
                                   .logical_rows = 64,
                                   .slot_capacity = 4,
                                   .row_bytes = 64};
  storage.content_identity[0] = 1;
  auto valid = make_boundary(storage, 4 * 64);
  ASSERT_TRUE(valid) << (valid ? "" : valid.error().detail);
  EXPECT_EQ(valid->external().back().row_storage, storage);

  auto changed_capacity = storage;
  changed_capacity.slot_capacity = 5;
  EXPECT_FALSE(make_boundary(changed_capacity, 4 * 64));
  EXPECT_FALSE(make_boundary(storage, 5 * 64));
  auto changed_digest = storage;
  changed_digest.content_identity[1] = 2;
  auto independently_changed = make_boundary(changed_digest, 4 * 64);
  ASSERT_TRUE(independently_changed);
  EXPECT_NE(independently_changed->digest().combined, valid->digest().combined);
  EXPECT_FALSE(make_boundary(changed_digest, 4 * 64, storage.content_identity));
  auto changed_ref = storage;
  ++changed_ref.ref.layer;
  EXPECT_FALSE(make_boundary(changed_ref, 4 * 64));
}

TEST(ComparisonBoundaryTest, OutsideRowCapacityChangesFullBoundaryNotLocalReadyBoundary) {
  ir::Builder builder;
  const auto x = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  const auto ids = builder.Input(ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({2})));
  const ir::WeightRef gamma_ref{.stage = 0, .component = 1, .role = 1, .layer = 0};
  const ir::WeightRef table_ref{.stage = 0, .component = 2, .role = 1, .layer = 0};
  const auto gamma = builder.Weight(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})), gamma_ref);
  const fmt::TensorFormat table_format{.encoding = fmt::Encoding::kF32,
                                       .layout = fmt::Layout::kDense,
                                       .shape = {.ne = {16, 64, 1, 1}, .rank = 2}};
  const auto table = builder.Weight(table_format, table_ref);
  const auto normalized = builder.RmsNorm(x, 1.0e-5F);
  const auto output = builder.Mul(normalized, gamma);
  const auto gathered = builder.GetRows(table, ids);
  builder.Output(output);
  builder.Output(gathered);
  auto graph_result = std::move(builder).Finish();
  ASSERT_TRUE(graph_result) << graph_result.error().detail;
  ir::Graph graph = std::move(*graph_result);
  const std::array weights{
      co::StepWeight{.tensor = gamma, .ref = gamma_ref, .parts = {{"gamma", fmt::PartRole::kData}}},
      co::StepWeight{
          .tensor = table, .ref = table_ref, .parts = {{"table", fmt::PartRole::kData}}}};
  const co::GraphBuildIO io{graph, graph.inputs(), graph.outputs(), weights};
  const auto funder = std::make_shared<Funder>();
  co::ComparisonRowStorage storage{.ref = table_ref,
                                   .part = fmt::PartRole::kData,
                                   .logical_format = table_format,
                                   .resource = 7,
                                   .group = 9,
                                   .shard = 3,
                                   .group_bytes = 8192,
                                   .resource_offset = 128,
                                   .resource_bytes = 4096,
                                   .logical_rows = 64,
                                   .slot_capacity = 4,
                                   .row_bytes = 64};
  storage.content_identity[0] = 1;
  const auto make_external = [&](co::ComparisonRowStorage row_storage) {
    return std::array{
        co::ComparisonExternalBinding{.original_root = x,
                                      .allocation_class = 1,
                                      .allocation_bytes = 64,
                                      .bytes = 64,
                                      .alignment = 4},
        co::ComparisonExternalBinding{.original_root = ids,
                                      .allocation_class = 2,
                                      .allocation_bytes = 8,
                                      .bytes = 8,
                                      .alignment = 4},
        co::ComparisonExternalBinding{.original_root = gamma,
                                      .part = fmt::PartRole::kData,
                                      .allocation_class = 3,
                                      .allocation_bytes = 64,
                                      .bytes = 64,
                                      .alignment = 4},
        co::ComparisonExternalBinding{.original_root = table,
                                      .part = fmt::PartRole::kData,
                                      .allocation_class = 4,
                                      .allocation_bytes = row_storage.slot_capacity * 64,
                                      .bytes = row_storage.slot_capacity * 64,
                                      .alignment = 16,
                                      .resource_part_identity = row_storage.content_identity,
                                      .row_storage = row_storage}};
  };
  const auto small_external = make_external(storage);
  const co::ComparisonBoundaryView small_source{io, {}, 0, small_external};
  const std::array<std::uint32_t, 2> offsets{0, 2};
  const std::array members{graph.producer(normalized), graph.producer(output)};
  auto small_full = co::MakeComparisonBoundary(small_source, funder);
  ASSERT_TRUE(small_full) << small_full.error().detail;
  auto small_local = co::MakeComparisonRegionBoundary({small_source, offsets, members}, funder);
  ASSERT_TRUE(small_local) << small_local.error().detail;

  auto larger_storage = storage;
  larger_storage.slot_capacity = 5;
  const auto larger_external = make_external(larger_storage);
  const co::ComparisonBoundaryView larger_source{io, {}, 0, larger_external};
  auto larger_full = co::MakeComparisonBoundary(larger_source, funder);
  ASSERT_TRUE(larger_full) << larger_full.error().detail;
  auto larger_local = co::MakeComparisonRegionBoundary({larger_source, offsets, members}, funder);
  ASSERT_TRUE(larger_local) << larger_local.error().detail;
  EXPECT_NE(small_full->digest().combined, larger_full->digest().combined);
  EXPECT_EQ(small_local->digest().combined, larger_local->digest().combined);
}

TEST(ComparisonBoundaryTest, OwnedRecordsSurviveSourceViewLifetime) {
  auto funder = std::make_shared<Funder>();
  std::optional<co::OwnedComparisonBoundary> owned;
  std::uint64_t identity = 0;
  {
    auto graph = TwoInputGraph();
    auto external = External(graph, 1);
    const co::ComparisonBoundaryView view{
        {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
    auto created = co::MakeComparisonBoundary(view, funder);
    ASSERT_TRUE(created) << (created ? "" : created.error().detail);
    identity = created->source_identity();
    owned.emplace(std::move(*created));
  }
  ASSERT_TRUE(owned);
  EXPECT_NE(identity, 0U);
  EXPECT_EQ(owned->roots().size(), 2U);
  EXPECT_EQ(owned->ports().size(), 4U);
  EXPECT_EQ(owned->external().size(), 2U);
  EXPECT_NE(owned->source_digest(), base::Sha256Digest{});
}

TEST(ComparisonBoundaryTest, MovedFromBoundaryCannotAuthenticate) {
  auto graph = TwoInputGraph();
  auto external = External(graph, 1);
  const co::ComparisonBoundaryView view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto funder = std::make_shared<Funder>();
  auto boundary = co::MakeComparisonBoundary(view, funder);
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  co::OwnedComparisonBoundary moved(std::move(*boundary));
  const std::array<ir::TensorId, 2> identity_map{ir::TensorId{0}, ir::TensorId{1}};
  auto result = co::ValidateMappedComparisonBoundary(*boundary, graph, identity_map);
  EXPECT_FALSE(result);
  EXPECT_NE(result.error().detail.find("moved from"), std::string::npos);
  EXPECT_EQ(moved.source_identity(), graph.identity());
}

TEST(ComparisonBoundaryTest, MissingPrivateRootCanRematerializeOnlyACompleteUnprotectedOutput) {
  // Sigmoid then reshape and reshape then sigmoid have the same value. The
  // latter has no tensor for the former's private, differently shaped root.
  for (int mutation = 0; mutation < 7; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder source_builder;
    const auto input = source_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 2})));
    const auto root = mutation == 4 ? input : source_builder.Unary(input, ir::UnaryFn::kSigmoid);
    const auto output = mutation == 2
                            ? source_builder.View(root, ir::ShapeOf({1, 4, 1}), {4, 4, 16, 16}, 0)
                            : source_builder.Reshape(root, ir::ShapeOf({1, 4, 2}));
    source_builder.Output(output);
    if (mutation == 1) source_builder.Keep(root);
    auto source = std::move(source_builder).Finish();
    ASSERT_TRUE(source);
    std::vector<co::ComparisonExternalBinding> external{{.original_root = input,
                                                         .allocation_class = 1,
                                                         .allocation_bytes = 32,
                                                         .bytes = 32,
                                                         .alignment = 16}};
    if (mutation == 3)
      external.push_back({.original_root = root,
                          .allocation_class = 2,
                          .allocation_bytes = 32,
                          .bytes = 32,
                          .alignment = 16});
    const co::ComparisonBoundaryView view{
        {*source, source->inputs(), source->outputs(), {}}, {}, 0, external};
    auto boundary = co::MakeComparisonBoundary(view, std::make_shared<Funder>());
    if (mutation == 3) {
      // The existing source contract forbids exporting a private activation
      // as an external binding before mapped-root validation is reached.
      ASSERT_FALSE(boundary);
      EXPECT_NE(boundary.error().detail.find("external binding is not part"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);

    ir::Builder planned_builder;
    const auto planned_input =
        planned_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 2})));
    const auto reshaped =
        mutation == 2
            ? planned_builder.View(planned_input, ir::ShapeOf({1, 4, 1}), {4, 4, 16, 16}, 0)
            : planned_builder.Reshape(
                  planned_input, mutation == 5 ? ir::ShapeOf({2, 2, 2}) : ir::ShapeOf({1, 4, 2}));
    const auto rematerialized = planned_builder.Unary(reshaped, ir::UnaryFn::kSigmoid);
    planned_builder.Output(rematerialized);
    if (mutation == 1) planned_builder.Keep(rematerialized);
    auto planned = std::move(planned_builder).Finish();
    ASSERT_TRUE(planned);
    std::vector<ir::TensorId> mapping(source->tensor_count());
    mapping[input.v] = planned_input;
    mapping[root.v] = mutation == 6 ? planned_input : ir::TensorId{};
    mapping[output.v] = rematerialized;
    const auto checked = co::ValidateMappedComparisonBoundary(*boundary, *planned, mapping);
    if (mutation == 0)
      EXPECT_TRUE(checked) << (checked ? "" : checked.error().detail);
    else
      EXPECT_FALSE(checked);
  }
}

TEST(ComparisonBoundaryTest, RematerializedOutputRootsRetainTheirAliasPartition) {
  ir::Builder source_builder;
  const auto input = source_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 2})));
  const auto first_root = source_builder.Unary(input, ir::UnaryFn::kSigmoid);
  const auto second_root = source_builder.Unary(input, ir::UnaryFn::kTanh);
  const auto first = source_builder.Reshape(first_root, ir::ShapeOf({1, 4, 2}));
  const auto second = source_builder.Reshape(second_root, ir::ShapeOf({1, 4, 2}));
  source_builder.Output(first);
  source_builder.Output(second);
  auto source = std::move(source_builder).Finish();
  ASSERT_TRUE(source);
  const co::ComparisonExternalBinding external{.original_root = input,
                                               .allocation_class = 1,
                                               .allocation_bytes = 32,
                                               .bytes = 32,
                                               .alignment = 16};
  const co::ComparisonBoundaryView view{
      {*source, source->inputs(), source->outputs(), {}}, {}, 0, std::span(&external, 1)};
  auto boundary = co::MakeComparisonBoundary(view, std::make_shared<Funder>());
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  for (const bool collapse : {false, true}) {
    SCOPED_TRACE(collapse);
    ir::Builder planned_builder;
    const auto planned_input =
        planned_builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 2})));
    const auto reshaped = planned_builder.Reshape(planned_input, ir::ShapeOf({1, 4, 2}));
    const auto a = planned_builder.Unary(reshaped, ir::UnaryFn::kSigmoid);
    const auto b = collapse ? planned_builder.Reshape(a, ir::ShapeOf({1, 4, 2}))
                            : planned_builder.Unary(reshaped, ir::UnaryFn::kTanh);
    planned_builder.Output(a);
    planned_builder.Output(b);
    auto planned = std::move(planned_builder).Finish();
    ASSERT_TRUE(planned);
    std::vector<ir::TensorId> mapping(source->tensor_count());
    mapping[input.v] = planned_input;
    mapping[first.v] = a;
    mapping[second.v] = b;
    const auto checked = co::ValidateMappedComparisonBoundary(*boundary, *planned, mapping);
    if (collapse)
      EXPECT_FALSE(checked);
    else
      EXPECT_TRUE(checked) << (checked ? "" : checked.error().detail);
  }
}

TEST(ComparisonBoundaryTest, RejectsAddedPlannedOutputAndStateRoots) {
  auto graph = TwoInputGraph();
  auto funder = std::make_shared<Funder>();
  auto external = External(graph, 1);
  const co::ComparisonBoundaryView source_view{
      {graph, graph.inputs(), graph.outputs(), {}}, {}, 0, external};
  auto source = co::MakeComparisonBoundary(source_view, funder);
  ASSERT_TRUE(source) << (source ? "" : source.error().detail);
  const std::array<ir::TensorId, 2> map{ir::TensorId{0}, ir::TensorId{1}};

  auto extra_output = TwoInputGraphWithExtraOutput();
  auto output_check = co::ValidateMappedComparisonBoundary(*source, extra_output, map);
  EXPECT_FALSE(output_check);
  EXPECT_NE(output_check.error().detail.find("complete ordered interface"), std::string::npos);

  ir::Builder with_state;
  const auto first = with_state.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  const auto second = with_state.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})));
  (void)with_state.State(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({16})),
                         ir::StateRef{.stage = 0, .tensor = 9, .layer = 0, .segment = 0});
  with_state.Output(first);
  with_state.Output(second);
  auto extra_state = std::move(with_state).Finish();
  ASSERT_TRUE(extra_state);
  const std::array<ir::TensorId, 2> original_map{ir::TensorId{0}, ir::TensorId{1}};
  auto state_check = co::ValidateMappedComparisonBoundary(*source, *extra_state, original_map);
  EXPECT_FALSE(state_check);
}

TEST(ComparisonBoundaryTest, RejectsWeightAndStateMappingsToViews) {
  constexpr auto kFormat = fmt::Encoding::kF32;
  constexpr ir::WeightRef kWeight{.stage = 0, .component = 0, .role = 0, .layer = 0};
  constexpr ir::StateRef kState{.stage = 0, .tensor = 0, .layer = 0, .segment = 0};
  const auto make_view = [] {
    std::array<std::uint64_t, fmt::kMaxDims> strides{};
    strides[0] = sizeof(float);
    return strides;
  };
  auto funder = std::make_shared<Funder>();

  ir::Builder original_weight_builder;
  const auto source_weight =
      original_weight_builder.Weight(ir::Dense(kFormat, ir::ShapeOf({16})), kWeight);
  const auto source_input = original_weight_builder.Input(ir::Dense(kFormat, ir::ShapeOf({16})));
  original_weight_builder.Output(source_input);
  auto original_weight = std::move(original_weight_builder).Finish().value();
  const std::array<co::StepWeight, 1> weight_facts{co::StepWeight{
      .tensor = source_weight, .ref = kWeight, .parts = {{"weight", fmt::PartRole::kData}}}};
  const co::GraphBuildIO weight_io{original_weight, original_weight.inputs(),
                                   original_weight.outputs(), weight_facts};
  const std::array<co::ComparisonExternalBinding, 2> weight_external{
      co::ComparisonExternalBinding{.original_root = source_input,
                                    .allocation_class = 1,
                                    .allocation_bytes = 64,
                                    .bytes = 64,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = source_weight,
                                    .part = fmt::PartRole::kData,
                                    .allocation_class = 2,
                                    .allocation_bytes = 64,
                                    .bytes = 64,
                                    .alignment = 16}};
  auto weight_boundary = co::MakeComparisonBoundary({weight_io, {}, 0, weight_external}, funder);
  ASSERT_TRUE(weight_boundary) << (weight_boundary ? "" : weight_boundary.error().detail);

  ir::Builder planned_weight_builder;
  const auto planned_weight =
      planned_weight_builder.Weight(ir::Dense(kFormat, ir::ShapeOf({16})), kWeight);
  const auto weight_view =
      planned_weight_builder.View(planned_weight, ir::ShapeOf({8}), make_view(), sizeof(float));
  const auto planned_input = planned_weight_builder.Input(ir::Dense(kFormat, ir::ShapeOf({16})));
  planned_weight_builder.Output(planned_input);
  auto planned_weight_graph = std::move(planned_weight_builder).Finish().value();
  const std::array<ir::TensorId, 2> weight_map{weight_view, ir::TensorId{2}};
  const auto weight_check =
      co::ValidateMappedComparisonBoundary(*weight_boundary, planned_weight_graph, weight_map);
  EXPECT_FALSE(weight_check);
  EXPECT_NE(weight_check.error().detail.find("weight mapping is a view"), std::string::npos);

  ir::Builder original_state_builder;
  const auto source_state =
      original_state_builder.State(ir::Dense(kFormat, ir::ShapeOf({16})), kState);
  const auto state_input = original_state_builder.Input(ir::Dense(kFormat, ir::ShapeOf({16})));
  original_state_builder.Output(state_input);
  auto original_state = std::move(original_state_builder).Finish().value();
  const std::array<co::StepState, 1> state_facts{
      co::StepState{.tensor = source_state, .ref = kState}};
  const co::GraphBuildIO state_io{
      original_state, original_state.inputs(), original_state.outputs(), {}};
  const std::array<co::ComparisonExternalBinding, 2> state_external{
      co::ComparisonExternalBinding{.original_root = state_input,
                                    .allocation_class = 1,
                                    .allocation_bytes = 64,
                                    .bytes = 64,
                                    .alignment = 16},
      co::ComparisonExternalBinding{.original_root = source_state,
                                    .allocation_class = 2,
                                    .allocation_bytes = 64,
                                    .bytes = 64,
                                    .alignment = 16}};
  auto state_boundary =
      co::MakeComparisonBoundary({state_io, state_facts, 0, state_external}, funder);
  ASSERT_TRUE(state_boundary) << (state_boundary ? "" : state_boundary.error().detail);

  ir::Builder planned_state_builder;
  const auto planned_state =
      planned_state_builder.State(ir::Dense(kFormat, ir::ShapeOf({16})), kState);
  const auto state_view =
      planned_state_builder.View(planned_state, ir::ShapeOf({8}), make_view(), sizeof(float));
  const auto planned_state_input =
      planned_state_builder.Input(ir::Dense(kFormat, ir::ShapeOf({16})));
  planned_state_builder.Output(planned_state_input);
  auto planned_state_graph = std::move(planned_state_builder).Finish().value();
  const std::array<ir::TensorId, 2> state_map{state_view, ir::TensorId{2}};
  const auto state_check =
      co::ValidateMappedComparisonBoundary(*state_boundary, planned_state_graph, state_map);
  EXPECT_FALSE(state_check);
  EXPECT_NE(state_check.error().detail.find("state mapping is a view"), std::string::npos);
}

}  // namespace
