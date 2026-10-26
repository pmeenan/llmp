// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/comparison_pair.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "comparison_local_calls.h"
#include "engine/graph_plan.h"
#include "execution/fusion_ranks.h"
#include "execution/registry.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/op.h"
#include "planner/select.h"

namespace llmp::engine {
namespace {

static_assert(!std::is_move_assignable_v<ComparisonResolvedBindings>);
static_assert(!std::is_move_assignable_v<ComparisonSourceBindings>);

format::TensorFormat F32(std::initializer_list<std::uint64_t> dims);

planner::StepPlan EmptyPlan(const ir::Graph& source) {
  auto graph = source.Clone();
  if (!graph) std::terminate();
  return {.primitive_digest = source.Digest(),
          .graph_digest = {},
          .policy_digest = {},
          .graph = std::move(*graph),
          .steps = {},
          .regions = {},
          .placement = {},
          .applied = {},
          .from = {},
          .tensors = {},
          .decomposed = {},
          .missing_numerical = {},
          .dropped_rewrites = {},
          .owner_row_invariant = false,
          .owner_rows_policy_enabled = false,
          .products = {},
          .root_alignments = {},
          .owner_products = {},
          .owner_digest = {},
          .invariant = {},
          .prologue = {}};
}

class Funder final : public ir::StorageFunder {
 public:
  bool Charge(std::uint64_t bytes) const override {
    ++charges;
    if (deny || (deny_at != 0 && charges == deny_at)) {
      ++denied_charges;
      last_denied_charge = charges;
      return false;
    }
    active += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const override { active -= bytes; }
  mutable std::uint64_t active = 0;
  mutable std::uint64_t charges = 0;
  mutable std::uint64_t denied_charges = 0;
  mutable std::uint64_t last_denied_charge = 0;
  mutable std::uint64_t denial_armed_after = 0;
  bool deny = false;
  mutable std::uint64_t deny_at = 0;
};

TEST(ComparisonPreparationDomainTest, OnlyTheTypedFullStatelessProtocolHasAStableDomain) {
  constexpr auto protocol =
      ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody;
  const auto first = ComparisonPreparationDomain(protocol, execution::Phase::kPrefill);
  const auto second = ComparisonPreparationDomain(protocol, execution::Phase::kPrefill);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(*first, *second);
  const auto decode = ComparisonPreparationDomain(protocol, execution::Phase::kDecode);
  ASSERT_TRUE(decode);
  EXPECT_NE(*first, *decode);
  EXPECT_FALSE(ComparisonPreparationDomain(static_cast<ComparisonRecipePreparation>(0),
                                           execution::Phase::kPrefill));
  EXPECT_FALSE(ComparisonPreparationDomain(protocol, static_cast<execution::Phase>(255)));
}

TEST(ComparisonPreparationDomainTest, ReadyBoundaryProtocolHasItsOwnPhaseBoundDomain) {
  constexpr auto protocol = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  const auto prefill = ComparisonPreparationDomain(protocol, execution::Phase::kPrefill);
  const auto decode = ComparisonPreparationDomain(protocol, execution::Phase::kDecode);
  const auto legacy = ComparisonPreparationDomain(
      ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody,
      execution::Phase::kPrefill);
  ASSERT_TRUE(prefill);
  ASSERT_TRUE(decode);
  ASSERT_TRUE(legacy);
  EXPECT_NE(*prefill, *decode);
  EXPECT_NE(*prefill, *legacy);
  EXPECT_FALSE(ComparisonPreparationDomain(static_cast<ComparisonRecipePreparation>(255),
                                           execution::Phase::kPrefill));
}

TEST(ComparisonSourceBindingsTest, RelocatedPlansRetainTheSameBoundaryAndActualSlices) {
  auto funder = std::make_shared<Funder>();
  const auto format = F32({4, 1});
  ir::Builder builder;
  const auto input = builder.Input(format, "input");
  builder.Output(input);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kStream}},
      .weights = {},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);

  auto first = EmptyPlan(source.graph);
  first.tensors = {input};
  first.placement.offsets = {{input, 64}};
  first.placement.extent = 128;
  first.placement.alignment = 64;
  auto second = EmptyPlan(source.graph);
  second.tensors = {input};
  second.placement.offsets = {{input, 128}};
  second.placement.extent = 192;
  second.placement.alignment = 64;

  kernels::Addresses a;
  a.activations = 0x10000;
  a.activation_bytes = 128;
  a.activation_alignment = 64;
  kernels::Addresses b = a;
  b.activations = 0x20000;
  b.activation_bytes = 256;
  auto first_ranges = inventory->Resolve(first, a);
  ASSERT_TRUE(first_ranges);
  ASSERT_EQ(first_ranges->ranges().size(), 1U);
  const auto first_base = first_ranges->ranges()[0].base;
  ASSERT_EQ(first_base, a.activations + 64);
  auto first_view = inventory->boundary_view();
  ASSERT_TRUE(first_view);
  auto first_boundary = components::MakeComparisonBoundary(*first_view, funder);
  ASSERT_TRUE(first_boundary);

  auto second_ranges = inventory->Resolve(second, b);
  ASSERT_TRUE(second_ranges);
  ASSERT_EQ(second_ranges->ranges().size(), 1U);
  EXPECT_EQ(second_ranges->ranges()[0].base, b.activations + 128);
  EXPECT_NE(second_ranges->ranges()[0].base, first_base);
  auto second_view = inventory->boundary_view();
  ASSERT_TRUE(second_view);
  auto second_boundary = components::MakeComparisonBoundary(*second_view, funder);
  ASSERT_TRUE(second_boundary);
  EXPECT_EQ(first_boundary->digest().combined, second_boundary->digest().combined);
  auto moved_inventory = std::move(*inventory);
  EXPECT_TRUE(moved_inventory.boundary_view());
  EXPECT_FALSE(inventory->boundary_view());
  source.logits = input;
  EXPECT_FALSE(moved_inventory.boundary_view());
}

// This is the ordinary component interface, with no StepShape or token metadata.
TEST(ComparisonSourceBindingsTest, ComponentPortsAuthenticateRangesPartsAndSourceMutation) {
  auto funder = std::make_shared<Funder>();
  const ir::WeightRef ref{.stage = 1, .component = 2, .role = 3};
  ir::Builder builder;
  const auto input = builder.Input(F32({4, 1}), "stream");
  const auto weight = builder.Weight(F32({4, 1}), ref, "gamma");
  builder.Output(builder.Mul(input, weight));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  std::vector<components::StepWeight> weights{
      {.tensor = weight, .ref = ref, .parts = {{"gamma", format::PartRole::kData}}}};
  const components::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  funder->deny = true;
  EXPECT_FALSE(ComparisonSourceBindings::Create(funder, io, 1));
  EXPECT_EQ(funder->active, 0U);
  funder->deny = false;
  auto inventory = ComparisonSourceBindings::Create(funder, io, 1);
  ASSERT_TRUE(inventory);
  auto plan = EmptyPlan(*graph);
  for (std::uint32_t i = 0; i < graph->tensor_count(); ++i) plan.tensors.push_back(ir::TensorId{i});
  plan.placement.offsets = {{input, 0}, {graph->outputs().front(), 64}};
  plan.placement.extent = 128;
  plan.placement.alignment = 64;
  const auto identity = base::Sha256{}.Update("actual immutable gamma bytes").Finish();
  kernels::BoundedAddress bound{
      .base = 0x20000, .bytes = 16, .alignment = 64, .content_identity = identity};
  kernels::Addresses addresses;
  addresses.activations = 0x10000;
  addresses.activation_bytes = 128;
  addresses.activation_alignment = 64;
  addresses.weight = [&](const ir::WeightRef&, format::PartRole) { return bound.base; };
  addresses.weight_range = [&](const ir::WeightRef&, format::PartRole) {
    return std::optional{bound};
  };
  auto resolved = inventory->Resolve(plan, addresses);
  ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error().detail);
  const auto part = std::ranges::find(resolved->ranges(), weight,
                                      &kernels::ComparisonBindingRange::original_root);
  ASSERT_NE(part, resolved->ranges().end());
  EXPECT_EQ(part->resource_part_identity, identity);
  const auto view = inventory->boundary_view();
  ASSERT_TRUE(view);
  EXPECT_TRUE(view->state.empty());
  EXPECT_EQ(view->active_stage, 1U);
  auto boundary = components::MakeComparisonBoundary(*view, funder);
  ASSERT_TRUE(boundary);
  // A same-size part with a different actual content identity cannot reuse
  // an already resolved source owner, nor may an undersized range do so.
  bound.content_identity = base::Sha256{}.Update("different gamma bytes").Finish();
  EXPECT_FALSE(inventory->Resolve(plan, addresses));
  bound.content_identity = identity;
  bound.bytes = 15;
  EXPECT_FALSE(inventory->Resolve(plan, addresses));
  bound.bytes = 16;
  bound.content_identity = base::Sha256Digest{};
  EXPECT_FALSE(inventory->Resolve(plan, addresses));
  bound.content_identity = identity;
  weights[0].parts[0].first = "changed-owner-part";
  const auto charges = funder->charges;
  EXPECT_FALSE(inventory->Resolve(plan, addresses));
  EXPECT_FALSE(inventory->boundary_view());
  EXPECT_EQ(funder->charges, charges);
  // Removing a real interface port is not an alternate component domain.
  auto incomplete = io;
  incomplete.inputs = {};
  EXPECT_FALSE(ComparisonSourceBindings::Create(funder, incomplete, 1));
}

TEST(ComparisonSourceBindingsTest, AChangedInputAliasPartitionIsUnproved) {
  auto funder = std::make_shared<Funder>();
  const auto format = F32({4, 1});
  ir::Builder builder;
  const auto a = builder.Input(format, "a");
  const auto b = builder.Input(format, "b");
  builder.Output(a);
  builder.Output(b);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{.graph = std::move(*graph),
                               .inputs = {{.tensor = a, .kind = components::InputKind::kStream},
                                          {.tensor = b, .kind = components::InputKind::kStream}},
                               .weights = {},
                               .state = {},
                               .logits = {},
                               .greedy = {},
                               .features = {},
                               .saves = {},
                               .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);
  auto aliased = EmptyPlan(source.graph);
  aliased.tensors = {a, b};
  aliased.placement.offsets = {{a, 64}, {b, 64}};
  aliased.placement.extent = 128;
  aliased.placement.alignment = 64;
  kernels::Addresses addresses;
  addresses.activations = 0x30000;
  addresses.activation_bytes = 256;
  addresses.activation_alignment = 64;
  auto accepted = inventory->Resolve(aliased, addresses);
  ASSERT_TRUE(accepted);

  auto disjoint = EmptyPlan(source.graph);
  disjoint.tensors = {a, b};
  disjoint.placement.offsets = {{a, 64}, {b, 128}};
  disjoint.placement.extent = 192;
  disjoint.placement.alignment = 64;
  auto changed = inventory->Resolve(disjoint, addresses);
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().code, ComparisonPairFailure::Code::kUnproved);
}

TEST(ComparisonSourceBindingsTest, SourceGraphAndTypedInventoryStayFixedAcrossResolve) {
  auto funder = std::make_shared<Funder>();
  const auto format = F32({4, 1});
  ir::Builder builder;
  const auto input = builder.Input(format, "input");
  builder.Output(input);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kStream}},
      .weights = {},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);

  auto plan = EmptyPlan(source.graph);
  plan.tensors = {input};
  plan.placement.offsets = {{input, 0}};
  plan.placement.extent = 64;
  plan.placement.alignment = 16;
  kernels::Addresses addresses;
  addresses.activations = 0x80000;
  addresses.activation_bytes = 64;
  addresses.activation_alignment = 16;

  source.logits = input;
  const auto charges_before = funder->charges;
  auto refused = inventory->Resolve(plan, addresses);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kStale);
  EXPECT_EQ(funder->charges, charges_before);
  EXPECT_FALSE(inventory->boundary_view());
}

TEST(ComparisonSourceBindingsTest, NativeAuxiliaryInputExtentDoesNotChangeSourceInterface) {
  auto funder = std::make_shared<Funder>();
  ir::Builder builder;
  const auto input = builder.Input(ir::Dense(format::Encoding::kI32, ir::ShapeOf({68})));
  builder.Output(input);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kPositions, .sections = 4}},
      .weights = {},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);
  auto plan = EmptyPlan(source.graph);
  plan.tensors = {input};
  plan.placement.offsets = {{input, 256}};
  plan.placement.extent = 1024;
  plan.placement.alignment = 256;
  kernels::Addresses addresses;
  addresses.activations = 0x80000;
  addresses.activation_bytes = 1024;
  addresses.activation_alignment = 256;
  const planner::RootBytesOf auxiliary = [](const ir::Graph&, ir::TensorId) { return 768; };
  auto incumbent = inventory->Resolve(plan, addresses, auxiliary);
  ASSERT_TRUE(incumbent);
  ASSERT_EQ(incumbent->ranges().size(), 1U);
  EXPECT_EQ(incumbent->ranges()[0].bytes, 272U);
  EXPECT_EQ(incumbent->ranges()[0].allocation_bytes, 272U);
  // The own alternative removes the auxiliary native token. Its original
  // input declaration and alias class remain identical.
  auto alternative = inventory->Resolve(plan, addresses);
  ASSERT_TRUE(alternative);
  EXPECT_EQ(alternative->ranges()[0].bytes, 272U);
  EXPECT_EQ(alternative->ranges()[0].allocation_bytes, 272U);
  EXPECT_EQ(alternative->ranges()[0].allocation_class, incumbent->ranges()[0].allocation_class);
  // Declaring the smaller interface never authorizes an undersized actual arena.
  addresses.activation_bytes = 1023;
  EXPECT_FALSE(inventory->Resolve(plan, addresses, auxiliary));
  addresses.activation_bytes = 1024;
  const planner::RootBytesOf overflow = [](const ir::Graph&, ir::TensorId) { return UINT64_MAX; };
  EXPECT_FALSE(inventory->Resolve(plan, addresses, overflow));
}

TEST(ComparisonSourceBindingsTest, MissingAndDuplicateRootMetadataRefuseBeforeFunding) {
  auto funder = std::make_shared<Funder>();
  const auto format = F32({4, 1});
  const ir::WeightRef ref{.stage = 0, .component = 1, .role = 2};
  ir::Builder builder;
  const auto input = builder.Input(format, "input");
  const auto weight = builder.Weight(format, ref, "weight");
  builder.Output(input);
  builder.Output(weight);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  auto invalid_graph = graph->Clone();
  ASSERT_TRUE(invalid_graph);

  auto missing_graph = graph->Clone();
  ASSERT_TRUE(missing_graph);
  components::StepGraph missing{
      .graph = std::move(*missing_graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kStream}},
      .weights = {},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  const auto before_missing = funder->charges;
  auto missing_result = ComparisonSourceBindings::Create(funder, missing, 0);
  ASSERT_FALSE(missing_result);
  EXPECT_EQ(missing_result.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(funder->charges, before_missing);

  auto duplicate_graph = graph->Clone();
  ASSERT_TRUE(duplicate_graph);
  components::StepGraph duplicate{
      .graph = std::move(*duplicate_graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kStream},
                 {.tensor = input, .kind = components::InputKind::kStream}},
      .weights = {{.tensor = weight, .ref = ref, .parts = {{"weight", format::PartRole::kData}}}},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  const auto before_duplicate = funder->charges;
  auto duplicate_result = ComparisonSourceBindings::Create(funder, duplicate, 0);
  ASSERT_FALSE(duplicate_result);
  EXPECT_EQ(duplicate_result.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(funder->charges, before_duplicate);

  components::StepGraph invalid_second{
      .graph = std::move(*invalid_graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kStream}},
      .weights = {{.tensor = weight, .ref = ref, .parts = {{"weight", format::PartRole::kData}}},
                  {.tensor = ir::TensorId{999},
                   .ref = ref,
                   .parts = {{"weight", format::PartRole::kData}}}},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  const auto before_invalid_later = funder->charges;
  auto invalid_later_result = ComparisonSourceBindings::Create(funder, invalid_second, 0);
  ASSERT_FALSE(invalid_later_result);
  EXPECT_EQ(invalid_later_result.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(funder->charges, before_invalid_later);
}

TEST(ComparisonSourceBindingsTest, CommitRowsUseTheRealSaveRangeAndCellPositionsStayUnstored) {
  auto funder = std::make_shared<Funder>();
  const auto format = F32({4, 1});
  ir::Builder builder;
  const auto input = builder.Input(format, "saved");
  builder.Output(input);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{.graph = std::move(*graph),
                               .inputs = {{.tensor = input,
                                           .kind = components::InputKind::kCommitRows,
                                           .segment = 2,
                                           .component = 7,
                                           .layer = 11,
                                           .what = "accepted"}},
                               .weights = {},
                               .state = {},
                               .logits = {},
                               .greedy = {},
                               .features = {},
                               // CommitRows reads a previously committed value; this graph's saves
                               // describe writes it will produce and need not repeat the read key.
                               .saves = {},
                               .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);
  auto plan = EmptyPlan(source.graph);
  plan.tensors = {input};
  kernels::Addresses addresses;
  addresses.activations = 0x40000;
  addresses.activation_bytes = 64;
  addresses.activation_alignment = 16;
  const auto active_before_missing_owner = funder->active;
  auto missing_owner = inventory->Resolve(plan, addresses);
  ASSERT_FALSE(missing_owner);
  EXPECT_EQ(missing_owner.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(funder->active, active_before_missing_owner);
  const auto check_key = [](std::uint32_t layer, std::uint16_t component, std::string_view what,
                            std::uint8_t segment) {
    EXPECT_EQ(layer, 11U);
    EXPECT_EQ(component, 7U);
    EXPECT_EQ(what, "accepted");
    EXPECT_EQ(segment, 2U);
  };
  addresses.save = [check_key](std::uint32_t layer, std::uint16_t component, std::string_view what,
                               std::uint8_t segment) {
    check_key(layer, component, what, segment);
    return 0x50010U;
  };
  addresses.save_range = [](std::uint32_t, std::uint16_t, std::string_view,
                            std::uint8_t) -> std::optional<kernels::BoundedAddress> {
    return kernels::BoundedAddress{.base = 0x50000, .bytes = 64, .alignment = 16};
  };
  const auto active_before_mismatch = funder->active;
  auto mismatched_owner = inventory->Resolve(plan, addresses);
  ASSERT_FALSE(mismatched_owner);
  EXPECT_EQ(mismatched_owner.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(funder->active, active_before_mismatch);
  addresses.save = [check_key](std::uint32_t layer, std::uint16_t component, std::string_view what,
                               std::uint8_t segment) {
    check_key(layer, component, what, segment);
    return 0x50000U;
  };
  auto resolved = inventory->Resolve(plan, addresses);
  ASSERT_TRUE(resolved);
  ASSERT_EQ(resolved->ranges().size(), 1U);
  EXPECT_EQ(resolved->ranges()[0].base, 0x50000U);
  EXPECT_EQ(resolved->ranges()[0].bytes, 16U);
  EXPECT_TRUE(source.saves.empty());

  auto positions_graph = source.graph.Clone();
  ASSERT_TRUE(positions_graph);
  components::StepGraph positions{
      .graph = std::move(*positions_graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kCellPositions}},
      .weights = {},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  auto position_inventory = ComparisonSourceBindings::Create(funder, positions, 0);
  ASSERT_TRUE(position_inventory);
  auto position_plan = EmptyPlan(positions.graph);
  position_plan.tensors = {input};
  auto position_ranges = position_inventory->Resolve(position_plan, addresses);
  ASSERT_TRUE(position_ranges);
  EXPECT_TRUE(position_ranges->ranges().empty());
  const auto view = position_inventory->boundary_view();
  ASSERT_TRUE(view);
  ASSERT_EQ(view->unstored_inputs.size(), 1U);
  EXPECT_EQ(view->unstored_inputs[0].original_input, input);
}

TEST(ComparisonSourceBindingsTest, MissingWeightBoundsAndDeniedInventoryFailClosed) {
  auto funder = std::make_shared<Funder>();
  const ir::WeightRef ref{.stage = 0, .component = 1, .role = 2};
  const auto format = F32({4, 1});
  ir::Builder builder;
  const auto weight = builder.Weight(format, ref, "w");
  builder.Output(weight);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = {},
      .weights = {{.tensor = weight, .ref = ref, .parts = {{"w", format::PartRole::kData}}}},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  funder->deny = true;
  auto denied = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_FALSE(denied);
  EXPECT_EQ(denied.error().code, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(funder->active, 0U);
  funder->deny = false;
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);
  auto plan = EmptyPlan(source.graph);
  plan.tensors = {weight};
  kernels::Addresses addresses;
  addresses.activations = 0x60000;
  addresses.activation_bytes = 64;
  addresses.activation_alignment = 16;
  addresses.weight = [](const ir::WeightRef&, format::PartRole) { return 0x70000U; };
  auto missing = inventory->Resolve(plan, addresses);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ComparisonPairFailure::Code::kUnproved);
}

TEST(ComparisonSourceBindingsTest, SharedBackingRequiresExplicitContainedAllocation) {
  auto funder = std::make_shared<Funder>();
  const ir::WeightRef first_ref{.stage = 0, .component = 1, .role = 0};
  const ir::WeightRef second_ref{.stage = 0, .component = 1, .role = 1};
  ir::Builder builder;
  const auto first = builder.Weight(F32({4, 1}), first_ref, "first");
  const auto second = builder.Weight(F32({4, 1}), second_ref, "second");
  builder.Output(builder.Add(first, second));
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = {},
      .weights =
          {{.tensor = first, .ref = first_ref, .parts = {{"first", format::PartRole::kData}}},
           {.tensor = second, .ref = second_ref, .parts = {{"second", format::PartRole::kData}}}},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);
  auto plan = EmptyPlan(source.graph);
  plan.tensors.resize(source.graph.tensor_count());
  for (std::uint32_t i = 0; i < plan.tensors.size(); ++i) plan.tensors[i] = ir::TensorId{i};
  kernels::Addresses addresses;
  addresses.activations = 0x60000;
  addresses.activation_bytes = 64;
  addresses.activation_alignment = 16;
  std::array<kernels::BoundedAddress, 2> parts{{{.base = 0x70020, .bytes = 128, .alignment = 16},
                                                {.base = 0x70040, .bytes = 128, .alignment = 16}}};
  addresses.weight = [&](const ir::WeightRef& ref, format::PartRole) {
    return parts[ref.role].base;
  };
  addresses.weight_range = [&](const ir::WeightRef& ref,
                               format::PartRole) -> std::optional<kernels::BoundedAddress> {
    return parts[ref.role];
  };
  // Pitch-spanning bounded parts overlap, but no backing relationship was
  // declared. Resolve must not infer a slab from their addresses.
  EXPECT_FALSE(inventory->Resolve(plan, addresses));
  for (auto& part : parts)
    part.allocation = kernels::BackingAllocation{.base = 0x70000, .bytes = 256};
  {
    auto resolved = inventory->Resolve(plan, addresses);
    ASSERT_TRUE(resolved);
    ASSERT_EQ(resolved->ranges().size(), 2U);
    const auto a = resolved->ranges()[0], b = resolved->ranges()[1];
    EXPECT_EQ(a.base, 0x70000U);
    EXPECT_EQ(b.base, a.base);
    EXPECT_EQ(a.allocation_bytes, 256U);
    EXPECT_EQ(b.allocation_bytes, a.allocation_bytes);
    EXPECT_EQ(a.allocation_class, b.allocation_class);
    EXPECT_EQ(a.offset, 32U);
    EXPECT_EQ(b.offset, 64U);
    EXPECT_EQ(a.bytes, 128U);  // full pitched part span, not the whole slab
    EXPECT_EQ(b.bytes, 128U);
  }
  const auto saved = parts;
  const auto paid = funder->active;
  for (unsigned mutation = 0; mutation < 8; ++mutation) {
    SCOPED_TRACE(mutation);
    parts = saved;
    switch (mutation) {
      case 0:
        parts[1].allocation.reset();
        break;  // mixed undeclared suballocation
      case 1:
        parts[1].allocation->bytes = 240;
        break;  // conflicting parent extent
      case 2:
        parts[1].allocation->bytes = 80;
        break;  // payload fits, full span does not
      case 3:
        parts[1].allocation->base = 0x70050;
        break;  // part before parent
      case 4:
        parts[1].allocation->base = 0;
        break;
      case 5:
        parts[1].allocation->bytes = UINT64_MAX;
        break;
      case 6:
        parts[1].bytes = UINT64_MAX;
        break;
      case 7:
        parts[1].bytes = 8;
        break;  // insufficient readable part
    }
    EXPECT_FALSE(inventory->Resolve(plan, addresses));
    EXPECT_EQ(funder->active, paid);
  }
}

TEST(ComparisonSourceBindingsTest,
     RowSlotsRequireMappedJoinedSideRowsGetRowsConsumersBeforeRangePublication) {
  const ir::WeightRef ref{.stage = 0, .component = 17, .role = 1, .layer = 0};
  const auto table_format = F32({16, 8});
  const auto make_source = [&](components::InputKind kind, bool duplicate_mapping = false) {
    ir::Builder builder;
    const auto ids = builder.Input({.encoding = format::Encoding::kI32,
                                    .layout = format::Layout::kDense,
                                    .shape = *format::ShapeOf(std::array<std::uint64_t, 1>{4})},
                                   "side_rows");
    const auto other =
        duplicate_mapping ? builder.Input(F32({4, 1}), "other_input") : ir::TensorId{};
    const auto table = builder.Weight(table_format, ref, "table");
    builder.Output(builder.GetRows(table, ids));
    auto graph = std::move(builder).Finish();
    if (!graph) return std::optional<components::StepGraph>{};
    components::StepGraph result{
        .graph = std::move(*graph),
        .inputs = {{.tensor = ids, .kind = kind, .segment = ir::kJoinedSegment, .component = 17}},
        .weights = {{.tensor = table, .ref = ref, .parts = {{"table", format::PartRole::kData}}}},
        .state = {},
        .logits = {},
        .greedy = {},
        .features = {},
        .saves = {},
        .owner_products = {}};
    if (duplicate_mapping)
      result.inputs.push_back(
          {.tensor = other, .kind = components::InputKind::kTokens, .segment = ir::kJoinedSegment});
    return std::optional<components::StepGraph>{std::move(result)};
  };
  const auto make_plan = [](const ir::Graph& graph, bool duplicate_mapping = false) {
    auto plan = EmptyPlan(graph);
    plan.tensors.resize(graph.tensor_count());
    for (std::uint32_t i = 0; i < graph.tensor_count(); ++i) plan.tensors[i] = ir::TensorId{i};
    if (duplicate_mapping) plan.tensors[graph.inputs()[1].v] = graph.inputs()[0];
    const auto input = graph.inputs().front();
    plan.placement.offsets = {{input, 0}};
    plan.placement.extent = 64;
    plan.placement.alignment = 16;
    return plan;
  };
  const auto addresses = [&](const auto& storage) {
    kernels::Addresses value{};
    value.activations = 0x10000;
    value.activation_bytes = 64;
    value.activation_alignment = 16;
    value.weight = [](const ir::WeightRef&, format::PartRole) { return std::uint64_t{0x20000}; };
    value.weight_range = [](const ir::WeightRef&,
                            format::PartRole) -> std::optional<kernels::BoundedAddress> {
      return kernels::BoundedAddress{.base = 0x20000, .bytes = 128, .alignment = 16};
    };
    value.weight_rows = [](const ir::WeightRef&) { return std::uint64_t{2}; };
    value.row_storage = [storage](const ir::WeightRef&, format::PartRole)
        -> std::optional<components::ComparisonRowStorage> { return storage; };
    return value;
  };
  components::ComparisonRowStorage storage{.ref = ref,
                                           .part = format::PartRole::kData,
                                           .logical_format = table_format,
                                           .resource = 0,
                                           .group = 0,
                                           .shard = 0,
                                           .group_bytes = 512,
                                           .resource_offset = 0,
                                           .resource_bytes = 512,
                                           .logical_rows = 8,
                                           .slot_capacity = 2,
                                           .row_bytes = 64};
  storage.content_identity[0] = 1;

  auto valid_source = make_source(components::InputKind::kSideRows);
  ASSERT_TRUE(valid_source);
  auto valid_funder = std::make_shared<Funder>();
  auto valid_bindings = ComparisonSourceBindings::Create(valid_funder, *valid_source, 0);
  ASSERT_TRUE(valid_bindings);
  auto valid_plan = make_plan(valid_source->graph);
  auto valid_ranges = valid_bindings->Resolve(valid_plan, addresses(storage));
  ASSERT_TRUE(valid_ranges) << valid_ranges.error().detail;
  ASSERT_EQ(valid_ranges->ranges().size(), 2U);
  EXPECT_EQ(valid_ranges->ranges().back().row_storage, storage);
  EXPECT_EQ(valid_ranges->ranges().back().bytes, 128U);
  auto enlarged_slots = addresses(storage);
  enlarged_slots.weight_range = [](const ir::WeightRef&,
                                   format::PartRole) -> std::optional<kernels::BoundedAddress> {
    return kernels::BoundedAddress{
        .base = 0x20000,
        .bytes = 128,
        .alignment = 16,
        .allocation = kernels::BackingAllocation{.base = 0x20000, .bytes = 256}};
  };
  EXPECT_FALSE(valid_bindings->Resolve(valid_plan, enlarged_slots));

  auto unsupported_source = make_source(components::InputKind::kTokens);
  ASSERT_TRUE(unsupported_source);
  auto unsupported_funder = std::make_shared<Funder>();
  auto unsupported_bindings =
      ComparisonSourceBindings::Create(unsupported_funder, *unsupported_source, 0);
  ASSERT_TRUE(unsupported_bindings);
  auto unsupported_plan = make_plan(unsupported_source->graph);
  const auto charges_before_resolve = unsupported_funder->charges;
  auto unsupported = unsupported_bindings->Resolve(unsupported_plan, addresses(storage));
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(unsupported_funder->charges, charges_before_resolve)
      << "unsupported consumers are rejected before per-plan range funding";

  auto ambiguous_source = make_source(components::InputKind::kSideRows, true);
  ASSERT_TRUE(ambiguous_source);
  auto ambiguous_funder = std::make_shared<Funder>();
  auto ambiguous_bindings =
      ComparisonSourceBindings::Create(ambiguous_funder, *ambiguous_source, 0);
  ASSERT_TRUE(ambiguous_bindings);
  auto ambiguous_plan = make_plan(ambiguous_source->graph, true);
  const auto ambiguous_charges = ambiguous_funder->charges;
  auto ambiguous = ambiguous_bindings->Resolve(ambiguous_plan, addresses(storage));
  ASSERT_FALSE(ambiguous);
  EXPECT_EQ(ambiguous.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(ambiguous_funder->charges, ambiguous_charges);
}

TEST(ComparisonSourceBindingsTest, DeniedPlanRangesAreNotPublishedOrRetained) {
  auto funder = std::make_shared<Funder>();
  const auto format = F32({4, 1});
  ir::Builder builder;
  const auto input = builder.Input(format, "input");
  builder.Output(input);
  auto graph = std::move(builder).Finish();
  ASSERT_TRUE(graph);
  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = {{.tensor = input, .kind = components::InputKind::kStream}},
      .weights = {},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  auto inventory = ComparisonSourceBindings::Create(funder, source, 0);
  ASSERT_TRUE(inventory);
  auto plan = EmptyPlan(source.graph);
  plan.tensors = {input};
  plan.placement.offsets = {{input, 0}};
  plan.placement.extent = 64;
  plan.placement.alignment = 16;
  kernels::Addresses addresses;
  addresses.activations = 0x90000;
  addresses.activation_bytes = 64;
  addresses.activation_alignment = 16;
  const auto prior_active = funder->active;
  funder->deny_at = funder->charges + 1;
  auto denied = inventory->Resolve(plan, addresses);
  ASSERT_FALSE(denied);
  EXPECT_EQ(denied.error().code, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(funder->active, prior_active);
  EXPECT_FALSE(inventory->boundary_view());
}

struct CopyCountedRootBytes {
  std::shared_ptr<std::uint32_t> copies;
  CopyCountedRootBytes(std::shared_ptr<std::uint32_t> copies_in) : copies(std::move(copies_in)) {}
  CopyCountedRootBytes(const CopyCountedRootBytes& other) : copies(other.copies) { ++*copies; }
  CopyCountedRootBytes(CopyCountedRootBytes&&) noexcept = default;
  std::uint64_t operator()(const ir::Graph&, ir::TensorId) const { return 0; }
};

format::TensorFormat F32(std::initializer_list<std::uint64_t> dims) {
  const auto shape = format::ShapeOf(dims);
  return {.encoding = format::Encoding::kF32, .layout = format::Layout::kDense, .shape = *shape};
}

execution::Implementation FakeImplementation(std::string name, ir::OpKey operation) {
  execution::Capability capability{.op = operation, .standalone_comparator = true};
  return {.name = std::move(name),
          .operation = operation,
          .source = "comparison-pair-test",
          .revision = "r1",
          .build = "cpu",
          .variant = "synthetic",
          .capability = capability};
}

struct ProjectionScript {
  std::uint32_t lower_ordinal = 0;
  ir::NodeId expansion{};
  std::uint32_t source_node_count = 0;
  bool expand_first = true;
  bool unproved_first = false;
  bool out_of_range = false;
  bool stale = false;
  bool duplicate_ports = false;
  bool overflow_interval = false;
  bool wrong_access = false;
  bool wrong_local_policy = false;
  bool wrong_local_registry = false;
  bool mutate_identity_on_unproved = false;
  bool mutate_physical_recipe_on_bind = false;
  bool certificate_supported = false;
  bool certificate_unproved = false;
  bool certificate_expand_first = false;
  ir::NodeId certificate_expansion{};
  bool force_redundant_expand = false;
  bool mutate_certificate_on_bind = false;
  bool mismatch_certificate_descriptor = false;
  bool fail_bind = false;
  bool inputs_preserved = true;
  bool remainder_same = false;
  bool mutate_local_recipe = false;
  ir::NodeId mutate_local_origin{};
};

struct ProgramObservation {
  std::uint32_t bind_count = 0;
  std::uint32_t run_count = 0;
  std::uint32_t record_count = 0;
  std::uint32_t destruction_count = 0;
  std::uint32_t certificate_count = 0;
  std::uint32_t remainder_count = 0;
  std::uint32_t prefix_count = 0;
  std::uint32_t region_count = 0;
  std::uint32_t bound_certificate_drift_count = 0;
  bool mutate_certificate_on_next_bind = false;
  bool mutate_whole_method_on_next_bind = false;
  bool bound_remainder_unproved = false;
  std::shared_ptr<Funder> injection_account;
  bool inject_pending_owner_failure = false;
  std::uint64_t inject_remainder_failure_delta = 0;
  std::uint32_t identity_queries = 0;
  bool remainder_failure_injected = false;
  std::array<std::uint32_t, 8> certificate_calls_by_lower{};
};

class ProjectingProgram final : public kernels::StepProgram {
 public:
  ProjectingProgram(execution::BoundPlan&& bound, const ir::Graph& source,
                    kernels::ComparisonRecipeSeal whole_recipe,
                    kernels::ComparisonPlanIdentity plan_identity,
                    std::shared_ptr<ProgramObservation> observation, ProjectionScript script,
                    bool descriptor_mode = false, bool omit_plan_identity = false,
                    std::vector<tests::ComparisonLocalCall> local_calls = {},
                    bool local_call_snapshot_required = false,
                    std::vector<kernels::InputSlot> input_slots = {})
      : bound_(std::move(bound)),
        whole_recipe_(whole_recipe),
        plan_identity_(plan_identity),
        observation_(std::move(observation)),
        script_(script),
        descriptor_mode_(descriptor_mode),
        omit_plan_identity_(omit_plan_identity),
        local_calls_(std::move(local_calls)),
        local_call_snapshot_required_(local_call_snapshot_required),
        input_slots_(std::move(input_slots)),
        bound_ready_(!descriptor_mode) {
    (void)source;
  }
  ~ProjectingProgram() override { ++observation_->destruction_count; }

  std::uint64_t scratch_bytes() const override { return 0; }
  std::uint64_t host_bytes() const override { return 0; }
  std::uint64_t launches() const override { return 0; }
  std::span<const kernels::InputSlot> inputs() const override { return input_slots_; }
  std::span<const kernels::OutputSlot> outputs() const override { return {}; }
  std::span<const kernels::TensorUse> tensors() const override { return {}; }
  const execution::BoundPlan* bound() const override { return bound_ready_ ? &bound_ : nullptr; }
  std::optional<kernels::ComparisonRecipeSeal> comparison_recipe() const override {
    auto result = whole_recipe_;
    if (result.scope == kernels::ComparisonProofScope::kWholeProgramNative && bound_ready_)
      result.state = kernels::ComparisonProofState::kBoundVerified;
    return result;
  }
  std::optional<kernels::ComparisonPlanIdentity> comparison_plan_identity() const override {
    if (observation_->inject_pending_owner_failure && ++observation_->identity_queries == 2 &&
        observation_->injection_account) {
      observation_->injection_account->denial_armed_after =
          observation_->injection_account->charges;
      observation_->injection_account->deny_at = observation_->injection_account->charges + 1;
    }
    return omit_plan_identity_ ? std::nullopt
                               : std::optional<kernels::ComparisonPlanIdentity>(plan_identity_);
  }
  std::expected<void, kernels::StepFailure> Bind(kernels::StepLaunch& launch) override {
    ++observation_->bind_count;
    if (expected_bind_launch_ != nullptr && expected_bind_launch_ != &launch)
      return std::unexpected(kernels::StepFailure{.detail = "unexpected measurement launch"});
    if (script_.fail_bind)
      return std::unexpected(kernels::StepFailure{.detail = "scripted measurement Bind failure"});
    bound_ready_ = true;
    if (observation_->mutate_whole_method_on_next_bind) {
      whole_recipe_.execution_method = kernels::ComparisonExecutionMethod::kOrdinaryOnly;
      observation_->mutate_whole_method_on_next_bind = false;
    }
    if (script_.mutate_certificate_on_bind || observation_->mutate_certificate_on_next_bind) {
      certificate_changed_ = true;
      observation_->mutate_certificate_on_next_bind = false;
    }
    return {};
  }
  void ExpectBindLaunch(const kernels::StepLaunch* launch) { expected_bind_launch_ = launch; }
  void MutatePlannedIdentity() { ++plan_identity_.planned_graph_identity; }
  bool is_bound() const { return bound_ready_; }
  std::expected<void, kernels::StepFailure> Run(kernels::StepLaunch&,
                                                providers::NativeStream) const override {
    ++observation_->run_count;
    return {};
  }
  std::expected<std::unique_ptr<kernels::Recorded>, kernels::StepFailure> Record(
      kernels::StepLaunch& launch, providers::NativeStream stream,
      const kernels::StepLaunch::Body& body) const override {
    ++observation_->record_count;
    return kernels::StepProgram::Record(launch, stream, body);
  }
  std::expected<void, kernels::StepFailure> RunComparisonPrefix(
      const kernels::ComparisonRegionProjection& projection, kernels::StepLaunch&,
      providers::NativeStream) const override {
    if (!ValidateComparisonProjection(projection)) return std::unexpected(kernels::StepFailure{});
    ++observation_->prefix_count;
    return {};
  }
  std::expected<void, kernels::StepFailure> RunComparisonRegion(
      const kernels::ComparisonRegionProjection& projection, kernels::StepLaunch&,
      providers::NativeStream) const override {
    if (!ValidateComparisonProjection(projection)) return std::unexpected(kernels::StepFailure{});
    ++observation_->region_count;
    return {};
  }

  kernels::ComparisonRegionDescriptorResult ProjectComparisonRegionDescriptor(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    if (!descriptor_mode_ || boundary.source_identity() != plan_identity_.source_graph_identity)
      return {.status = kernels::ComparisonRegionStatus::kUnproved};
    ++descriptor_serial_;
    if (boundary.nodes().size() == descriptor_order_.size())
      for (std::size_t i = 0; i < descriptor_order_.size(); ++i)
        descriptor_order_[i] = boundary.nodes()[i].original_node.v;
    if (script_.expand_first && script_.certificate_expand_first &&
        std::ranges::find_if(boundary.nodes(),
                             [&](const auto& node) {
                               return node.original_node == script_.expansion;
                             }) != boundary.nodes().end() &&
        std::ranges::find_if(boundary.nodes(), [&](const auto& node) {
          return node.original_node == script_.certificate_expansion;
        }) != boundary.nodes().end())
      descriptor_union_seen_ = true;
    if (descriptor_serial_ == 1 && script_.unproved_first) {
      if (script_.mutate_identity_on_unproved) ++plan_identity_.planned_graph_identity;
      return {.status = kernels::ComparisonRegionStatus::kUnproved};
    }
    if (descriptor_serial_ == 1 && script_.expand_first) {
      required_scratch_ =
          script_.out_of_range ? ir::NodeId{script_.source_node_count} : script_.expansion;
      return {.status = kernels::ComparisonRegionStatus::kExpand,
              .required_members = std::span<const ir::NodeId>(&required_scratch_, 1)};
    }
    return {.status = kernels::ComparisonRegionStatus::kClosed,
            .seal = LocalSeal(boundary, kernels::ComparisonProofState::kDescriptorVerified)};
  }

  kernels::ComparisonRegionDescriptorResult ProjectComparisonRegionCertificate(
      const components::OwnedComparisonRegionBoundary& boundary,
      kernels::ComparisonCertificatePhase phase) const override {
    ++observation_->certificate_count;
    if (script_.lower_ordinal != 0 &&
        script_.lower_ordinal <= observation_->certificate_calls_by_lower.size())
      ++observation_->certificate_calls_by_lower[script_.lower_ordinal - 1];
    if (!script_.certificate_supported || script_.certificate_unproved ||
        boundary.source_identity() != plan_identity_.source_graph_identity ||
        (phase == kernels::ComparisonCertificatePhase::kBound && !bound_ready_))
      return {.status = kernels::ComparisonRegionStatus::kUnproved,
              .reason = "scripted applicability certificate unavailable"};
    if (phase == kernels::ComparisonCertificatePhase::kBound &&
        boundary.nodes().size() == bound_order_.size())
      for (std::size_t i = 0; i < bound_order_.size(); ++i)
        bound_order_[i] = boundary.nodes()[i].original_node.v;
    if (phase == kernels::ComparisonCertificatePhase::kDescriptor && ++certificate_serial_ == 1 &&
        script_.certificate_expand_first) {
      required_scratch_ = script_.certificate_expansion;
      return {.status = kernels::ComparisonRegionStatus::kExpand,
              .required_members = std::span<const ir::NodeId>(&required_scratch_, 1)};
    }
    if (phase == kernels::ComparisonCertificatePhase::kDescriptor &&
        script_.certificate_expand_first && boundary.nodes().size() == 3) {
      std::array<std::uint32_t, 3> ids{boundary.nodes()[0].original_node.v,
                                       boundary.nodes()[1].original_node.v,
                                       boundary.nodes()[2].original_node.v};
      std::ranges::sort(ids);
      certificate_union_seen_ = ids == std::array<std::uint32_t, 3>{2, 3, 4};
    }
    auto seal = LocalSeal(boundary, phase == kernels::ComparisonCertificatePhase::kBound
                                        ? kernels::ComparisonProofState::kBoundVerified
                                        : kernels::ComparisonProofState::kDescriptorVerified);
    if (phase == kernels::ComparisonCertificatePhase::kDescriptor &&
        script_.mismatch_certificate_descriptor)
      seal.physical_recipe = TestDigest("mismatched-applicability-descriptor");
    if (phase == kernels::ComparisonCertificatePhase::kBound && certificate_changed_) {
      ++observation_->bound_certificate_drift_count;
      seal.physical_recipe = TestDigest("changed-applicability-certificate");
    }
    return {.status = kernels::ComparisonRegionStatus::kClosed, .seal = seal};
  }

  kernels::ComparisonRemainderResult CompareComparisonRemainder(
      const kernels::StepProgram&, const components::OwnedComparisonRegionBoundary&,
      kernels::ComparisonCertificatePhase phase) const override {
    ++observation_->remainder_count;
    if (phase == kernels::ComparisonCertificatePhase::kBound &&
        observation_->bound_remainder_unproved)
      return {};
    if (observation_->inject_remainder_failure_delta != 0 &&
        !observation_->remainder_failure_injected && observation_->injection_account) {
      observation_->remainder_failure_injected = true;
      observation_->injection_account->denial_armed_after =
          observation_->injection_account->charges;
      observation_->injection_account->deny_at =
          observation_->injection_account->charges + observation_->inject_remainder_failure_delta;
    }
    return script_.remainder_same
               ? kernels::ComparisonRemainderResult{.status =
                                                        kernels::ComparisonRemainderStatus::kSame}
               : kernels::ComparisonRemainderResult{};
  }

  kernels::ComparisonRegionResult ProjectComparisonRegion(
      const components::OwnedComparisonRegionBoundary& boundary) const override {
    ++serial_;
    port_count_ = 0;
    ready_union_node_count_ = boundary.nodes().size();
    for (std::size_t i = 0; i < ready_union_order_.size() && i < boundary.nodes().size(); ++i)
      ready_union_order_[i] = boundary.nodes()[i].original_node.v;
    if (descriptor_mode_ && boundary.nodes().size() == 3) {
      ready_union_retained_seen_ = boundary.nodes()[0].original_node == ir::NodeId{4} &&
                                   boundary.nodes()[1].original_node == ir::NodeId{2} &&
                                   boundary.nodes()[2].original_node == ir::NodeId{3};
    }
    if (serial_ == 1 && script_.unproved_first)
      return {.status = kernels::ComparisonRegionStatus::kUnproved};
    if (serial_ == 1 && script_.expand_first &&
        (script_.force_redundant_expand || !descriptor_mode_ ||
         std::ranges::find_if(boundary.nodes(), [&](const auto& node) {
           return node.original_node == script_.expansion;
         }) == boundary.nodes().end())) {
      required_scratch_ =
          script_.out_of_range ? ir::NodeId{script_.source_node_count} : script_.expansion;
      return {.status = kernels::ComparisonRegionStatus::kExpand,
              .required_members = std::span<const ir::NodeId>(&required_scratch_, 1),
              .program_nonce = comparison_program_nonce(),
              .generation = generation_,
              .serial = serial_};
    }
    required_scratch_ = ir::NodeId{0};  // The previous borrowed expansion is now invalid.

    if (serial_ >= 2) final_node_count_ = boundary.nodes().size();
    if (serial_ >= 2 && boundary.nodes().size() == 3) {
      union_seen_ = boundary.nodes()[0].original_node == ir::NodeId{3} &&
                    boundary.nodes()[0].canonical == 0 && boundary.nodes()[0].group == 0 &&
                    boundary.nodes()[1].original_node == ir::NodeId{2} &&
                    boundary.nodes()[1].canonical == 1 && boundary.nodes()[1].group == 1 &&
                    boundary.nodes()[2].original_node == ir::NodeId{4} &&
                    boundary.nodes()[2].canonical == 2 && boundary.nodes()[2].group == 2;
    } else if (serial_ >= 2 && boundary.nodes().size() == 2) {
      union_seen_ = boundary.nodes()[0].original_node == ir::NodeId{3} &&
                    boundary.nodes()[0].canonical == 0 && boundary.nodes()[0].group == 0 &&
                    boundary.nodes()[1].original_node == ir::NodeId{4} &&
                    boundary.nodes()[1].canonical == 1 && boundary.nodes()[1].group == 1;
    }

    for (std::size_t i = 0; i < boundary.ports().size(); ++i) {
      const auto& port = boundary.ports()[i];
      if (port.kind == components::ComparisonRegionPortKind::kStateRead ||
          port.kind == components::ComparisonRegionPortKind::kStateWrite)
        return {.status = kernels::ComparisonRegionStatus::kUnproved};
      const auto& tensor = boundary.tensors()[port.canonical_tensor];
      if (!port.original_tensor.valid() || port.original_tensor.v >= buffers_.size())
        return {.status = kernels::ComparisonRegionStatus::kUnproved};
      const auto bytes = tensor.storage_bytes;
      const bool input = port.kind == components::ComparisonRegionPortKind::kInput;
      if (bytes > buffers_[0].size()) return {.status = kernels::ComparisonRegionStatus::kUnproved};
      auto& storage = buffers_[port.original_tensor.v];
      if (port_count_ == ports_.size())
        return {.status = kernels::ComparisonRegionStatus::kUnproved};
      ports_[port_count_++] = {.kind = input ? kernels::ComparisonRegionValueKind::kInput
                                             : kernels::ComparisonRegionValueKind::kOutput,
                               .ordinal = static_cast<std::uint32_t>(i),
                               .source_tensor = port.original_tensor,
                               .local_tensor = port.canonical_tensor,
                               .role = tensor.role,
                               .format = tensor.format,
                               .address = reinterpret_cast<std::uintptr_t>(storage.data()),
                               .bytes = bytes,
                               .root_offset = port.view_offset,
                               .read = input,
                               .written = !input};
    }
    if (script_.duplicate_ports && port_count_ > 1)
      std::fill(ports_.begin() + 1, ports_.begin() + port_count_, ports_[0]);
    if (script_.overflow_interval && port_count_ != 0) {
      ports_[0].address = std::numeric_limits<std::uint64_t>::max() - 1;
      ports_[0].bytes = 4;
    }
    if (script_.wrong_access && port_count_ != 0) ports_[0].written = ports_[0].read;
    kernels::ComparisonRegionRecipeSeal seal;
    if (boundary.source_identity() != plan_identity_.source_graph_identity)
      return {.status = kernels::ComparisonRegionStatus::kUnproved};
    seal = LocalSeal(boundary, kernels::ComparisonProofState::kBoundVerified);
    if (script_.mutate_physical_recipe_on_bind && serial_ == 1)
      seal.physical_recipe = TestDigest("mutated-only-at-bind");
    if (!descriptor_mode_) {
      seal.state = kernels::ComparisonProofState::kBoundVerified;
      seal.missing = kernels::ComparisonMissing::kNone;
      seal.subject = boundary.digest().logical;
      seal.boundary = boundary.digest().combined;
      seal.planned_region = TestDigest("planned");
      seal.physical_recipe = TestDigest("physical");
      seal.policy =
          script_.wrong_local_policy ? TestDigest("wrong-local-policy") : whole_recipe_.policy;
      seal.registry = script_.wrong_local_registry ? TestDigest("wrong-local-registry")
                                                   : whole_recipe_.registry;
      seal.backend_recipe = TestDigest("backend");
      seal.backend_build = TestDigest("build");
      seal.backend_selection = TestDigest("selection");
      seal.backend_device = TestDigest("device");
    }
    auto projection =
        MintComparisonRegionProjection(generation_, serial_, 0, 1, seal, script_.inputs_preserved);
    return {.status = kernels::ComparisonRegionStatus::kClosed,
            .projection = std::move(projection),
            .value_ports =
                std::span<const kernels::ComparisonRegionValuePort>(ports_.data(), port_count_),
            .program_nonce = comparison_program_nonce(),
            .generation = generation_,
            .serial = serial_};
  }

  bool ValidateComparisonProjection(
      const kernels::ComparisonRegionProjection& projection) const override {
    return !script_.stale &&
           AuthenticatedComparisonRegionSpan(projection, generation_, serial_).has_value();
  }
  bool union_seen() const { return union_seen_; }
  bool ready_union_retained_seen() const { return ready_union_retained_seen_; }
  std::array<std::uint32_t, 3> ready_union_order() const { return ready_union_order_; }
  std::size_t ready_union_node_count() const { return ready_union_node_count_; }
  std::array<std::uint32_t, 3> descriptor_order() const { return descriptor_order_; }
  std::array<std::uint32_t, 3> bound_order() const { return bound_order_; }
  bool certificate_union_seen() const { return certificate_union_seen_; }
  bool descriptor_union_seen() const { return descriptor_union_seen_; }
  std::uint64_t projection_count() const { return serial_; }
  std::size_t final_node_count() const { return final_node_count_; }

 private:
  static base::Sha256Digest TestDigest(std::string_view text) {
    return base::Sha256{}.Update(text).Finish();
  }

  kernels::ComparisonRegionRecipeSeal LocalSeal(
      const components::OwnedComparisonRegionBoundary& boundary,
      kernels::ComparisonProofState state) const {
    kernels::ComparisonRegionRecipeSeal seal;
    seal.version = 3;
    seal.state = state;
    seal.missing = kernels::ComparisonMissing::kNone;
    seal.subject = boundary.digest().logical;
    seal.boundary = boundary.digest().combined;
    if (local_call_snapshot_required_ || !local_calls_.empty()) {
      const auto local = tests::HashContainedComparisonLocalCalls(local_calls_, boundary);
      if (!local) {
        seal.missing = kernels::ComparisonMissing::kSelectedCall;
        seal.planned_region = TestDigest("empty-local-calls");
      } else {
        if (script_.mutate_local_recipe &&
            std::ranges::any_of(boundary.nodes(), [&](const auto& member) {
              return member.original_node == script_.mutate_local_origin;
            }))
          seal.planned_region = base::Sha256{}
                                    .Update(std::as_bytes(std::span(*local)))
                                    .Update("scripted-prior-local-recipe-drift")
                                    .Finish();
        else
          seal.planned_region = *local;
      }
    } else {
      const auto identity_bytes =
          std::as_bytes(std::span(&plan_identity_.planned_graph_identity, 1));
      seal.planned_region = base::Sha256{}.Update(identity_bytes).Finish();
    }
    seal.physical_recipe = TestDigest("descriptor-physical");
    seal.policy = whole_recipe_.policy;
    seal.registry = whole_recipe_.registry;
    seal.backend_recipe = !local_calls_.empty()
                              ? TestDigest("descriptor-backend-local-calls")
                              : TestDigest("descriptor-backend-" +
                                           std::to_string(plan_identity_.planned_graph_identity));
    seal.backend_build = TestDigest("descriptor-build");
    seal.backend_selection = TestDigest("descriptor-selection");
    seal.backend_device = TestDigest("descriptor-device");
    return seal;
  }

  execution::BoundPlan bound_;
  kernels::ComparisonRecipeSeal whole_recipe_{};
  mutable kernels::ComparisonPlanIdentity plan_identity_{};
  std::shared_ptr<ProgramObservation> observation_;
  ProjectionScript script_{};
  bool descriptor_mode_ = false;
  bool omit_plan_identity_ = false;
  std::vector<tests::ComparisonLocalCall> local_calls_;
  bool local_call_snapshot_required_ = false;
  std::vector<kernels::InputSlot> input_slots_;
  bool bound_ready_ = true;
  bool certificate_changed_ = false;
  const kernels::StepLaunch* expected_bind_launch_ = nullptr;
  mutable ir::NodeId required_scratch_{};
  mutable std::uint64_t serial_ = 0;
  mutable std::uint64_t descriptor_serial_ = 0;
  mutable std::uint64_t certificate_serial_ = 0;
  static constexpr std::uint64_t generation_ = 1;
  mutable std::array<kernels::ComparisonRegionValuePort, 16> ports_{};
  alignas(64) mutable std::array<std::array<std::byte, 64>, 5> buffers_{};
  mutable std::size_t port_count_ = 0;
  mutable bool union_seen_ = false;
  mutable bool ready_union_retained_seen_ = false;
  mutable std::array<std::uint32_t, 3> ready_union_order_{};
  mutable std::size_t ready_union_node_count_ = 0;
  mutable std::array<std::uint32_t, 3> descriptor_order_{};
  mutable std::array<std::uint32_t, 3> bound_order_{};
  mutable bool certificate_union_seen_ = false;
  mutable bool descriptor_union_seen_ = false;
  mutable std::size_t final_node_count_ = 0;
};

class NoopStepLaunch final : public kernels::StepLaunch {
 public:
  std::expected<std::unique_ptr<kernels::Recorded>, kernels::StepFailure> Capture(
      providers::NativeStream, const Body&) override {
    return std::unexpected(kernels::StepFailure{.detail = "not captured"});
  }
  std::expected<void, kernels::StepFailure> Launch(providers::NativeStream,
                                                   const kernels::Recorded&) override {
    return std::unexpected(kernels::StepFailure{.detail = "not launched"});
  }
};

class TestRecorded final : public kernels::Recorded {
 public:
  explicit TestRecorded(const kernels::StepLaunch* owner) : Recorded(owner) {}
  std::size_t nodes() const override { return 1; }
  double capture_seconds() const override { return 0; }
  double instantiate_seconds() const override { return 0; }
};

class RecordingStepLaunch final : public kernels::StepLaunch {
 public:
  std::expected<std::unique_ptr<kernels::Recorded>, kernels::StepFailure> Capture(
      providers::NativeStream, const Body& body) override {
    ++capture_count;
    auto captured = body();
    if (!captured) return std::unexpected(std::move(captured.error()));
    return std::unique_ptr<kernels::Recorded>(
        std::make_unique<TestRecorded>(owner_override == nullptr ? this : owner_override));
  }
  std::expected<void, kernels::StepFailure> Launch(providers::NativeStream,
                                                   const kernels::Recorded&) override {
    ++launch_count;
    return {};
  }
  std::uint32_t capture_count = 0;
  std::uint32_t launch_count = 0;
  const kernels::StepLaunch* owner_override = nullptr;
};

struct PairHarness {
  std::shared_ptr<Funder> funder = std::make_shared<Funder>();
  std::optional<execution::Registry> registry;
  std::optional<execution::Registry> recipe_registry;
  std::unique_ptr<ComparisonPair> pair;
  ProjectingProgram* incumbent = nullptr;
  ProjectingProgram* alternative = nullptr;
  std::shared_ptr<ProgramObservation> incumbent_observation =
      std::make_shared<ProgramObservation>();
  std::shared_ptr<ProgramObservation> alternative_observation =
      std::make_shared<ProgramObservation>();
  std::optional<ComparisonPairFailure::Code> plan_failure;
  std::uint64_t charges_after_plan = 0;
  ir::TensorId input_tensor{};
  ir::TensorId weight_tensor{};
  std::vector<ir::NodeId> fused_site_nodes;
  std::uint64_t input_bytes = 0;
  std::uint64_t weight_bytes = 0;
  alignas(64) std::array<std::array<std::byte, 64>, 2> input_backing{};
  alignas(64) std::array<std::array<std::byte, 64>, 2> weight_backing{};

  std::array<std::array<kernels::ComparisonBindingRange, 2>, 2> BindingStorage(
      ir::TensorId input, ir::TensorId weight, std::uint64_t logical_input_bytes,
      std::uint64_t logical_weight_bytes) {
    std::array<std::array<kernels::ComparisonBindingRange, 2>, 2> result{};
    for (std::size_t arm = 0; arm < result.size(); ++arm) {
      result[arm][0] = {.original_root = input,
                        .allocation_class = 0,
                        .allocation_bytes = input_backing[arm].size(),
                        .offset = 0,
                        .bytes = logical_input_bytes,
                        .alignment = 64,
                        .base = reinterpret_cast<std::uintptr_t>(input_backing[arm].data())};
      result[arm][1] = {.original_root = weight,
                        .part = format::PartRole::kData,
                        .allocation_class = 1,
                        .allocation_bytes = weight_backing[arm].size(),
                        .offset = 0,
                        .bytes = logical_weight_bytes,
                        .alignment = 64,
                        .base = reinterpret_cast<std::uintptr_t>(weight_backing[arm].data())};
    }
    return result;
  }
};

struct FeatureInputFixture {
  format::TensorFormat format = F32({4, 1});
  components::InputKind source_kind = components::InputKind::kFeatures;
  kernels::InputKind slot_kind = kernels::InputKind::kFeatures;
  std::optional<std::uint64_t> slot_bytes;
  bool omit_slot = false;
  bool duplicate_slot = false;
  bool duplicate_with_other_kind = false;
  bool wrong_slot_tensor = false;
};

struct PositionInputFixture {
  format::TensorFormat format = ir::Dense(format::Encoding::kI32, ir::ShapeOf({1}));
  components::InputKind source_kind = components::InputKind::kPositions;
  kernels::InputKind slot_kind = kernels::InputKind::kPositions;
  std::optional<std::uint64_t> slot_bytes;
  std::optional<std::uint8_t> slot_segment;
  bool omit_slot = false;
  bool duplicate_slot = false;
  bool duplicate_with_other_kind = false;
  bool wrong_slot_tensor = false;
};

std::expected<execution::BoundPlan, execution::PlanRejection> ResolveSelected(
    const execution::Registry& registry, const kernels::LowerInput& input) {
  std::vector<execution::Choice> choices;
  choices.reserve(input.plan.steps.size());
  for (const auto& selected : input.plan.steps) {
    const auto& implementation = registry.at(selected.implementation);
    choices.push_back(
        {.operation = implementation.operation, .implementation = implementation.name});
  }
  auto backend_plan = execution::Plan::Build(registry, choices);
  if (!backend_plan) return std::unexpected(std::move(backend_plan.error()));
  return execution::Resolve(registry, *backend_plan);
}

std::unique_ptr<PairHarness> MakePairHarness(
    ProjectionScript incumbent_script, ProjectionScript alternative_script,
    bool wrong_alternative_recipe = false, std::uint64_t deny_at = 0,
    const planner::PlanOptions* options = nullptr, bool install_bindings = true,
    bool wrong_alternative_identity = false, bool descriptor_programs = false,
    bool ready_boundary_graph = false, bool stable_local_recipe = false,
    bool two_fused_sites = false, std::optional<FeatureInputFixture> feature_fixture = std::nullopt,
    std::optional<PositionInputFixture> position_fixture = std::nullopt) {
  auto harness = std::make_unique<PairHarness>();
  harness->funder->deny_at = deny_at;
  ir::Builder builder;
  if (ready_boundary_graph) builder.SetOwner(0);
  const auto input = builder.Input(feature_fixture ? feature_fixture->format : F32({4, 1}), "x");
  const auto weight = builder.Weight(F32({4}), {.stage = 0, .component = 0, .role = 0});
  ir::TensorId prefix;
  if (ready_boundary_graph) {
    builder.SetJoined();
    prefix = builder.Cont(builder.Add(input, input));
  } else {
    prefix = builder.RmsNorm(input, 1e-6F);
  }
  const auto fused = builder.Fused(ir::FusedKind::kRmsNormMul, std::array{prefix, weight},
                                   ir::NormParams{.eps = 1e-6F})[0];
  ir::TensorId latest = fused;
  ir::TensorId second_fused{};
  if (two_fused_sites) {
    second_fused = builder.Fused(ir::FusedKind::kRmsNormMul, std::array{latest, weight},
                                 ir::NormParams{.eps = 1e-6F})[0];
    latest = second_fused;
  }
  const auto suffix = builder.RmsNorm(latest, 1e-6F);
  ir::TensorId positions;
  if (position_fixture) positions = builder.Input(position_fixture->format, "positions");
  builder.Output(suffix);
  auto unowned = std::move(builder).Finish();
  EXPECT_TRUE(unowned.has_value());
  if (!unowned) return {};
  auto graph = unowned->Clone(harness->funder);
  EXPECT_TRUE(graph.has_value());
  if (!graph) return {};
  const auto source_identity = graph->identity();
  const auto fused_node = graph->producer(fused);
  harness->fused_site_nodes.push_back(fused_node);
  if (two_fused_sites) harness->fused_site_nodes.push_back(graph->producer(second_fused));
  const auto source_node_count = static_cast<std::uint32_t>(graph->nodes().size());
  incumbent_script.source_node_count = source_node_count;
  alternative_script.source_node_count = source_node_count;
  const auto input_bytes = *ir::StorageBytes(graph->tensor(input));
  const auto weight_bytes = *ir::StorageBytes(graph->tensor(weight));
  harness->input_tensor = input;
  harness->weight_tensor = weight;
  harness->input_bytes = input_bytes;
  harness->weight_bytes = weight_bytes;

  components::StepGraph source{
      .graph = std::move(*graph),
      .inputs = ready_boundary_graph
                    ? std::vector<components::StepInput>{{.tensor = input,
                                                          .kind = components::InputKind::kStream,
                                                          .component = 0}}
                : feature_fixture
                    ? std::vector<components::StepInput>{{.tensor = input,
                                                          .kind = feature_fixture->source_kind,
                                                          .segment = ir::kJoinedSegment}}
                    : std::vector<components::StepInput>{},
      .weights = {{.tensor = weight,
                   .ref = {.stage = 0, .component = 0, .role = 0},
                   .parts = {{"weight", format::PartRole::kData}}}},
      .state = {},
      .logits = {},
      .greedy = {},
      .features = {},
      .saves = {},
      .owner_products = {}};
  if (position_fixture)
    source.inputs.push_back({.tensor = positions,
                             .kind = position_fixture->source_kind,
                             .segment = ir::kJoinedSegment});
  std::vector<execution::Implementation> declarations;
  declarations.push_back(FakeImplementation("pair.rms_norm", ir::Primitive(ir::Op::kRmsNorm)));
  declarations.push_back(FakeImplementation("pair.mul", ir::Primitive(ir::Op::kMul)));
  declarations.push_back(FakeImplementation("pair.fused", ir::Fused(ir::FusedKind::kRmsNormMul)));
  if (ready_boundary_graph) {
    declarations.push_back(FakeImplementation("pair.add", ir::Primitive(ir::Op::kAdd)));
    declarations.push_back(FakeImplementation("pair.cont", ir::Primitive(ir::Op::kCont)));
  }
  auto registry_result = execution::Registry::Create(std::move(declarations));
  EXPECT_TRUE(registry_result.has_value());
  if (!registry_result) return {};
  harness->registry.emplace(std::move(*registry_result));

  std::vector<components::ComparisonExternalBinding> external{
      components::ComparisonExternalBinding{.original_root = input,
                                            .allocation_class = 0,
                                            .allocation_bytes = 64,
                                            .bytes = input_bytes,
                                            .alignment = 64},
      components::ComparisonExternalBinding{.original_root = weight,
                                            .part = format::PartRole::kData,
                                            .allocation_class = 1,
                                            .allocation_bytes = 64,
                                            .bytes = weight_bytes,
                                            .alignment = 64}};
  if (position_fixture)
    external.push_back({.original_root = positions,
                        .allocation_class = 2,
                        .allocation_bytes = 16,
                        .bytes = *ir::StorageBytes(source.graph.tensor(positions)),
                        .alignment = 16});
  auto pair_result = ComparisonPair::Create(
      harness->funder, std::move(source), {},
      std::vector<components::ComparisonExternalBinding>(external.begin(), external.end()), 0, 0,
      ready_boundary_graph ? ComparisonInputDomain::kComponentPorts
                           : ComparisonInputDomain::kTokenStep);
  EXPECT_TRUE(pair_result.has_value());
  if (!pair_result) return {};
  harness->pair = std::move(*pair_result);
  planner::ComparisonSiteSelector selector;
  selector.source_graph_identity = source_identity;
  selector.kind = planner::ComparisonSiteKind::kFusedNode;
  selector.member_origins = {{fused_node}};
  const planner::DeviceFacts device{.device_class = "test", .arch = 121};
  const planner::PlanOptions default_options{};
  const auto& plan_options = options ? *options : default_options;
  auto planned = harness->pair->Plan(selector, *harness->registry, device, {}, plan_options);
  EXPECT_EQ(planned.has_value(), deny_at == 0);
  if (!planned) {
    harness->plan_failure = planned.error().code;
    return deny_at == 0 ? nullptr : std::move(harness);
  }

  harness->charges_after_plan = harness->funder->charges;
  if (!install_bindings) return harness;
  const auto original_ranges = harness->BindingStorage(input, weight, input_bytes, weight_bytes);
  std::array<std::vector<kernels::ComparisonBindingRange>, 2> ranges{
      std::vector(original_ranges[0].begin(), original_ranges[0].end()),
      std::vector(original_ranges[1].begin(), original_ranges[1].end())};
  if (position_fixture) {
    for (std::size_t arm = 0; arm < 2; ++arm)
      ranges[arm].push_back(
          {.original_root = positions,
           .allocation_class = 2,
           .allocation_bytes = 16,
           .bytes = *ir::StorageBytes(harness->pair->source().graph.tensor(positions)),
           .alignment = 16,
           .base = reinterpret_cast<std::uintptr_t>(harness->input_backing[arm].data() + 32)});
  }
  const std::array<std::span<const kernels::ComparisonBindingRange>, 2> range_views{ranges[0],
                                                                                    ranges[1]};
  auto bindings_installed = harness->pair->InstallBindings(range_views);
  EXPECT_TRUE(bindings_installed.has_value());
  if (!bindings_installed) return {};

  kernels::Addresses addresses;
  auto incumbent_input = harness->pair->LoweringInput(ComparisonPairArm::kIncumbent, addresses);
  auto alternative_input = harness->pair->LoweringInput(ComparisonPairArm::kAlternative, addresses);
  EXPECT_TRUE(incumbent_input.has_value());
  EXPECT_TRUE(alternative_input.has_value());
  if (!incumbent_input || !alternative_input) return {};
  EXPECT_NE(incumbent_input->comparison_region, nullptr);
  EXPECT_NE(alternative_input->comparison_region, nullptr);
  EXPECT_NE(incumbent_input->comparison, nullptr);
  EXPECT_NE(alternative_input->comparison, nullptr);
  if (incumbent_input->comparison_region == nullptr ||
      alternative_input->comparison_region == nullptr || incumbent_input->comparison == nullptr ||
      alternative_input->comparison == nullptr ||
      incumbent_input->comparison->expected_boundary == nullptr ||
      alternative_input->comparison->expected_boundary == nullptr)
    return {};
  if (ready_boundary_graph) {
    EXPECT_FALSE(incumbent_input->token.has_value());
    EXPECT_FALSE(alternative_input->token.has_value());
  } else {
    EXPECT_TRUE(incumbent_input->token.has_value());
    EXPECT_TRUE(alternative_input->token.has_value());
    if (!incumbent_input->token || !alternative_input->token) return {};
    EXPECT_EQ(&incumbent_input->token->shape, &harness->pair->shape());
    EXPECT_EQ(&alternative_input->token->shape, &harness->pair->shape());
  }
  EXPECT_EQ(incumbent_input->comparison->expected_boundary->source_identity(), source_identity);
  EXPECT_EQ(alternative_input->comparison->expected_boundary->source_identity(), source_identity);
  EXPECT_NE(incumbent_input->comparison_region->seed, nullptr);
  EXPECT_NE(alternative_input->comparison_region->seed, nullptr);
  EXPECT_NE(incumbent_input->comparison_region->origins, nullptr);
  EXPECT_NE(alternative_input->comparison_region->origins, nullptr);
  EXPECT_NE(incumbent_input->comparison_region->final_nodes, nullptr);
  EXPECT_NE(alternative_input->comparison_region->final_nodes, nullptr);
  if (incumbent_input->comparison_region->final_nodes == nullptr ||
      alternative_input->comparison_region->final_nodes == nullptr)
    return {};
  EXPECT_EQ(incumbent_input->comparison_region->final_nodes->source_graph_identity,
            incumbent_input->io.graph.identity());
  EXPECT_EQ(alternative_input->comparison_region->final_nodes->source_graph_identity,
            alternative_input->io.graph.identity());
  EXPECT_EQ(incumbent_input->comparison_region->final_nodes->planned_graph_identity,
            incumbent_input->plan.graph.identity());
  EXPECT_EQ(alternative_input->comparison_region->final_nodes->planned_graph_identity,
            alternative_input->plan.graph.identity());
  EXPECT_EQ(incumbent_input->comparison_region->final_nodes->node_origins.values.size(),
            incumbent_input->plan.graph.nodes().size());
  EXPECT_EQ(alternative_input->comparison_region->final_nodes->node_origins.values.size(),
            alternative_input->plan.graph.nodes().size());
  auto incumbent_bound = ResolveSelected(*harness->registry, *incumbent_input);
  auto alternative_bound = ResolveSelected(*harness->registry, *alternative_input);
  EXPECT_TRUE(incumbent_bound.has_value());
  EXPECT_TRUE(alternative_bound.has_value());
  if (!incumbent_bound || !alternative_bound) return {};

  const auto recipe = [&](const kernels::LowerInput& input) {
    kernels::ComparisonRecipeSeal result;
    result.mode = kernels::ComparisonRunMode::kFull;
    result.subject = input.io.graph.Digest();
    result.planned_graph = input.plan.graph.Digest();
    result.policy = input.plan.policy_digest;
    result.registry = harness->registry->SelectionDigest();
    return result;
  };
  const auto incumbent_recipe = recipe(*incumbent_input);
  const auto alternative_recipe =
      wrong_alternative_recipe ? incumbent_recipe : recipe(*alternative_input);

  std::vector<kernels::InputSlot> feature_slots;
  if (feature_fixture && !feature_fixture->omit_slot) {
    const kernels::InputSlot slot{.bytes = feature_fixture->slot_bytes.value_or(input_bytes),
                                  .tensor = feature_fixture->wrong_slot_tensor ? weight : input,
                                  .segment = ir::kJoinedSegment,
                                  .kind = feature_fixture->slot_kind};
    feature_slots.push_back(slot);
    if (feature_fixture->duplicate_slot) feature_slots.push_back(slot);
    if (feature_fixture->duplicate_with_other_kind) {
      auto duplicate = slot;
      duplicate.kind = kernels::InputKind::kTokens;
      feature_slots.push_back(duplicate);
    }
  }

  if (position_fixture && !position_fixture->omit_slot) {
    const kernels::InputSlot slot{
        .bytes = position_fixture->slot_bytes.value_or(
            *ir::StorageBytes(harness->pair->source().graph.tensor(positions))),
        .tensor = position_fixture->wrong_slot_tensor ? weight : positions,
        .segment = position_fixture->slot_segment.value_or(ir::kJoinedSegment),
        .kind = position_fixture->slot_kind};
    feature_slots.push_back(slot);
    if (position_fixture->duplicate_slot) feature_slots.push_back(slot);
    if (position_fixture->duplicate_with_other_kind) {
      auto duplicate = slot;
      duplicate.kind = kernels::InputKind::kTokens;
      feature_slots.push_back(duplicate);
    }
  }
  const kernels::ComparisonPlanIdentity incumbent_identity{
      .source_graph_identity = source_identity,
      .planned_graph_identity = incumbent_input->plan.graph.identity()};
  const kernels::ComparisonPlanIdentity alternative_identity{
      .source_graph_identity = source_identity,
      .planned_graph_identity = wrong_alternative_identity
                                    ? incumbent_identity.planned_graph_identity
                                    : alternative_input->plan.graph.identity()};
  auto incumbent_local_calls =
      stable_local_recipe ? tests::CaptureComparisonLocalCalls(*incumbent_input, *harness->registry)
                          : std::vector<tests::ComparisonLocalCall>{};
  auto alternative_local_calls = stable_local_recipe ? tests::CaptureComparisonLocalCalls(
                                                           *alternative_input, *harness->registry)
                                                     : std::vector<tests::ComparisonLocalCall>{};
  auto incumbent = std::make_unique<ProjectingProgram>(
      std::move(*incumbent_bound), harness->pair->source().graph, incumbent_recipe,
      incumbent_identity, harness->incumbent_observation, incumbent_script, descriptor_programs,
      false, std::move(incumbent_local_calls), stable_local_recipe, feature_slots);
  auto alternative = std::make_unique<ProjectingProgram>(
      std::move(*alternative_bound), harness->pair->source().graph, alternative_recipe,
      alternative_identity, harness->alternative_observation, alternative_script,
      descriptor_programs, false, std::move(alternative_local_calls), stable_local_recipe,
      feature_slots);
  harness->incumbent = incumbent.get();
  harness->alternative = alternative.get();
  auto installed =
      descriptor_programs
          ? harness->pair->InstallDescriptorPrograms(std::move(incumbent), std::move(alternative))
          : harness->pair->InstallPrograms(std::move(incumbent), std::move(alternative));
  const bool wrong_program = wrong_alternative_recipe || wrong_alternative_identity;
  EXPECT_EQ(installed.has_value(), !wrong_program);
  if (!installed && !wrong_program) return {};
  if (!installed) {
    harness->incumbent = nullptr;
    harness->alternative = nullptr;
  }
  return harness;
}

TEST(ComparisonWholeProgramKeyTest, CompleteSiteAndWholeProofAreDistinctFromReady) {
  auto harness =
      MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0, nullptr, false);
  ASSERT_TRUE(harness);
  const auto* inventory = harness->pair->inventory();
  ASSERT_NE(inventory, nullptr);
  const auto& source = harness->pair->source();
  std::vector<std::uint32_t> offsets{0};
  std::vector<ir::NodeId> members;
  for (const auto& group : inventory->site.accepted_member_origins) {
    members.insert(members.end(), group.begin(), group.end());
    offsets.push_back(static_cast<std::uint32_t>(members.size()));
  }
  std::vector<components::ComparisonExternalBinding> external;
  for (const auto& binding : inventory->source_boundary.external()) {
    external.push_back(
        {.original_root = inventory->source_boundary.roots()[binding.canonical_root].original_root,
         .part = binding.part,
         .allocation_class = binding.canonical_allocation_class,
         .allocation_bytes = binding.allocation_bytes,
         .offset = binding.offset,
         .bytes = binding.bytes,
         .alignment = binding.alignment,
         .resource_part_identity = binding.resource_part_identity,
         .row_storage = binding.row_storage});
  }
  auto boundary = components::MakeComparisonRegionBoundary(
      {.source = {.io = components::GraphIO(source), .external = external},
       .member_offsets = offsets,
       .members = members},
      harness->funder);
  ASSERT_TRUE(boundary) << (boundary ? "" : boundary.error().detail);
  auto digest = [](std::string_view text) { return base::Sha256{}.Update(text).Finish(); };
  kernels::ComparisonRecipeSeal left;
  left.scope = kernels::ComparisonProofScope::kWholeProgramNative;
  left.state = kernels::ComparisonProofState::kDescriptorVerified;
  left.missing = kernels::ComparisonMissing::kNone;
  left.subject = inventory->source_boundary.source_digest();
  left.boundary = inventory->source_boundary.digest().combined;
  left.planned_graph = digest("incumbent-plan");
  left.physical_recipe = digest("incumbent-physical");
  left.policy = digest("policy");
  left.registry = harness->registry->SelectionDigest();
  left.backend_recipe = digest("backend");
  left.backend_build = digest("build");
  left.backend_selection = digest("selection");
  left.backend_device = digest("device");
  auto right = left;
  right.planned_graph = digest("alternative-plan");
  right.physical_recipe = digest("alternative-physical");
  const auto make = [&](const kernels::ComparisonRecipeSeal& a,
                        const kernels::ComparisonRecipeSeal& b,
                        const planner::ComparisonSite& site) {
    return RecipeComparisonKeyForWholePrograms(inventory->source_boundary, *boundary, site,
                                               execution::Phase::kPrefill,
                                               harness->registry->SelectionDigest(), a, b);
  };
  const auto key = make(left, right, inventory->site);
  ASSERT_TRUE(key);
  EXPECT_NE(
      key->preparation_domain,
      *ComparisonPreparationDomain(ComparisonRecipePreparation::kStatelessReadyBoundaryOperands,
                                   execution::Phase::kPrefill));
  auto ordinary = left;
  ordinary.execution_method = kernels::ComparisonExecutionMethod::kOrdinaryOnly;
  const auto ordinary_key = make(ordinary, right, inventory->site);
  ASSERT_TRUE(ordinary_key);
  EXPECT_NE(*ordinary_key, *key);
  EXPECT_EQ(ordinary_key->preparation_domain,
            *ComparisonPreparationDomain(
                ComparisonRecipePreparation::kWholeProgramRestoredStateOrdinaryInvocation,
                execution::Phase::kPrefill));
  const auto other_arm_ordinary = make(left, ordinary, inventory->site);
  ASSERT_TRUE(other_arm_ordinary);
  EXPECT_EQ(other_arm_ordinary->preparation_domain, ordinary_key->preparation_domain);
  auto invalid_method = left;
  invalid_method.execution_method = static_cast<kernels::ComparisonExecutionMethod>(255);
  EXPECT_FALSE(make(invalid_method, right, inventory->site));
  auto invalid = left;
  invalid.scope = kernels::ComparisonProofScope::kTypedAccessClosure;
  EXPECT_FALSE(make(invalid, right, inventory->site));
  invalid = left;
  invalid.backend_device = {};
  EXPECT_FALSE(make(invalid, right, inventory->site));
  invalid = left;
  invalid.missing = kernels::ComparisonMissing::kPhysicalClosure;
  EXPECT_FALSE(make(invalid, right, inventory->site));
  auto forged = inventory->site;
  forged.accepted_member_origins[0][0] = ir::NodeId{0};
  EXPECT_FALSE(make(left, right, forged));
  forged = inventory->site;
  ++forged.selector.source_graph_identity;
  EXPECT_FALSE(make(left, right, forged));
  auto bound = left;
  bound.state = kernels::ComparisonProofState::kBoundVerified;
  EXPECT_TRUE(SameWholeProgramRecipe(left, bound));
  bound.execution_method = kernels::ComparisonExecutionMethod::kOrdinaryOnly;
  EXPECT_FALSE(SameWholeProgramRecipe(left, bound));
  bound.execution_method = left.execution_method;
  ++bound.pool_bytes;
  EXPECT_FALSE(SameWholeProgramRecipe(left, bound));
  bound = left;
  bound.state = kernels::ComparisonProofState::kBoundVerified;
  bound.backend_device = digest("another-device");
  EXPECT_FALSE(SameWholeProgramRecipe(left, bound));
}

std::expected<execution::Registry, execution::PlanRejection> RegistryWithRecipe(
    const execution::RecipeComparisonKey& key, std::uint64_t incumbent_ns,
    std::uint64_t alternative_ns, bool ready_boundary = false,
    execution::RecipeComparisonKind kind = execution::RecipeComparisonKind::kLatency,
    std::optional<execution::RecipeComparisonKey> competing_local = {}) {
  std::vector<execution::Implementation> declarations;
  declarations.push_back(FakeImplementation("pair.rms_norm", ir::Primitive(ir::Op::kRmsNorm)));
  declarations.push_back(FakeImplementation("pair.mul", ir::Primitive(ir::Op::kMul)));
  declarations.push_back(FakeImplementation("pair.fused", ir::Fused(ir::FusedKind::kRmsNormMul)));
  if (ready_boundary) {
    declarations.push_back(FakeImplementation("pair.add", ir::Primitive(ir::Op::kAdd)));
    declarations.push_back(FakeImplementation("pair.cont", ir::Primitive(ir::Op::kCont)));
  }
  execution::RankData ranks;
  ranks.recipe_comparisons.push_back(
      {.key = key,
       .incumbent_ns = incumbent_ns,
       .alternative_ns = alternative_ns,
       .samples = kind == execution::RecipeComparisonKind::kLatency ? 3U : 0U,
       .protocol = base::Sha256{}.Update("test-protocol").Finish(),
       .receipt = base::Sha256{}.Update("test-receipt").Finish(),
       .provenance = "synthetic-reviewed-test",
       .shape_class = "f32-4x1",
       .kind = kind});
  if (competing_local) {
    auto local = ranks.recipe_comparisons.front();
    local.key = *competing_local;
    local.kind = execution::RecipeComparisonKind::kLatency;
    local.incumbent_ns = 50;
    local.alternative_ns = 100;
    local.samples = 3;
    ranks.recipe_comparisons.push_back(std::move(local));
  }
  return execution::Registry::Create(std::move(declarations), std::move(ranks));
}

enum class CumulativeFailure {
  kLower,
  kValidator,
  kUnproved,
  kAbsentIdentity,
  kSetupFunding,
  kValidRowMiss,
  kBindDrift,
  kSelfRemainderDrift,
  kPriorChosenDrift,
  kPendingOwnerDenied,
  kProofPaymentDenied,
  kLedgerCommitDenied,
  kWholeOrdinary,
  kWholeQualityOverridesLocal,
};

void CheckClosedExecutionGates(kernels::StepProgram& adopted, RecordingStepLaunch& launch,
                               ProgramObservation& observation, const planner::StepShape& shape) {
  EXPECT_EQ(adopted.bound(), nullptr);
  EXPECT_FALSE(adopted.warm());
  EXPECT_FALSE(adopted.Run(launch, {}));
  EXPECT_FALSE(adopted.RunWarm(launch, {}));
  EXPECT_FALSE(adopted.Position({}));
  EXPECT_FALSE(adopted.TryUpdateGeneratedInputs(shape));
  const kernels::StepLaunch::Body empty_body = [] {
    return std::expected<void, kernels::StepFailure>{};
  };
  EXPECT_FALSE(adopted.Record(launch, {}, empty_body));
  EXPECT_EQ(launch.capture_count, 0U);
  EXPECT_EQ(observation.run_count, 0U);
  EXPECT_EQ(observation.record_count, 0U);
}

void CheckCumulativeFailure(CumulativeFailure failure, bool component_ports = false) {
  const planner::Policy fixture_policy{
      .active_stage = static_cast<std::uint16_t>(component_ports ? 0 : UINT16_MAX)};
  // This fake backend grants no local or remainder authority. It exercises
  // production construction/failure ownership, not GPU proof qualification.
  const bool ready_recipe = failure == CumulativeFailure::kValidRowMiss ||
                            failure == CumulativeFailure::kBindDrift ||
                            failure == CumulativeFailure::kSelfRemainderDrift ||
                            failure == CumulativeFailure::kPriorChosenDrift ||
                            failure == CumulativeFailure::kPendingOwnerDenied ||
                            failure == CumulativeFailure::kProofPaymentDenied ||
                            failure == CumulativeFailure::kLedgerCommitDenied ||
                            failure == CumulativeFailure::kWholeQualityOverridesLocal;
  const bool quality_override = failure == CumulativeFailure::kWholeQualityOverridesLocal;
  const bool whole_ordinary = failure == CumulativeFailure::kWholeOrdinary || quality_override;
  ProjectionScript pair_script{.expand_first = false};
  if (ready_recipe) {
    pair_script.certificate_supported = true;
    pair_script.remainder_same = true;
  }
  auto harness =
      MakePairHarness(pair_script, pair_script, false, 0, nullptr, true, false, true, true,
                      ready_recipe, failure == CumulativeFailure::kPriorChosenDrift);
  ASSERT_TRUE(harness);
  const auto preparation =
      ready_recipe
          ? ComparisonRecipePreparation::kStatelessReadyBoundaryOperands
          : ComparisonRecipePreparation::kFullStatelessRestagedInputsPurePrefixCapturedFullBody;
  auto key = DescribeRecipeComparison(*harness->pair, *harness->registry, preparation);
  ASSERT_TRUE(key);
  ASSERT_TRUE(*key);
  auto registered_key = **key;
  const auto& source = harness->pair->source();
  const planner::StepShape shape{.phase = execution::Phase::kPrefill,
                                 .segments = {{.slot = 0, .rows = 1, .outputs = 1}}};
  alignas(256) std::array<std::byte, 4096> activations{};
  kernels::Addresses addresses;
  addresses.activations = reinterpret_cast<std::uintptr_t>(activations.data());
  addresses.activation_bytes = activations.size();
  addresses.activation_alignment = 256;
  addresses.weight = [&](const ir::WeightRef&, format::PartRole) {
    return reinterpret_cast<std::uintptr_t>(harness->weight_backing[0].data());
  };
  addresses.weight_range = [&](const ir::WeightRef&,
                               format::PartRole) -> std::optional<kernels::BoundedAddress> {
    return kernels::BoundedAddress{
        .base = reinterpret_cast<std::uintptr_t>(harness->weight_backing[0].data()),
        .bytes = harness->weight_backing[0].size(),
        .alignment = 64};
  };
  const auto whole_seal = [&](const planner::StepPlan& plan,
                              const components::OwnedComparisonBoundary& boundary,
                              const base::Sha256Digest& selection) {
    kernels::ComparisonRecipeSeal seal;
    seal.scope = kernels::ComparisonProofScope::kWholeProgramNative;
    seal.state = kernels::ComparisonProofState::kDescriptorVerified;
    seal.missing = kernels::ComparisonMissing::kNone;
    seal.subject = boundary.source_digest();
    seal.boundary = boundary.digest().combined;
    seal.planned_graph = plan.graph.Digest();
    seal.physical_recipe = seal.planned_graph;  // fake backend's deterministic recipe
    seal.policy = plan.policy_digest;
    seal.registry = selection;
    seal.backend_recipe = base::Sha256{}.Update("whole-backend").Finish();
    seal.backend_build = base::Sha256{}.Update("whole-build").Finish();
    seal.backend_selection = base::Sha256{}.Update("whole-selection").Finish();
    seal.backend_device = base::Sha256{}.Update("whole-device").Finish();
    if (std::ranges::any_of(plan.graph.nodes(), [](const ir::Node& node) {
          return node.key == ir::Fused(ir::FusedKind::kRmsNormMul);
        }))
      seal.execution_method = kernels::ComparisonExecutionMethod::kOrdinaryOnly;
    return seal;
  };
  if (whole_ordinary) {
    // A whole key includes the production source boundary and placement, not
    // the Ready harness's deliberately oversized input binding envelope.
    kernels::StepBackend key_backend{.registry = &*harness->registry,
                                     .device = {.device_class = "test", .arch = 121}};
    planner::PlanOptions key_options{.shape = component_ports ? nullptr : &shape};
    key_options.staged.emplace();
    for (const auto& input : source.inputs) key_options.staged->push_back(input.tensor);
    key_options.alignment_staged = key_options.staged;
    auto bindings = component_ports ? ComparisonSourceBindings::Create(
                                          harness->funder, components::GraphIO(source), 0)
                                    : ComparisonSourceBindings::Create(harness->funder, source, 0);
    ASSERT_TRUE(bindings);
    auto prepared = PrepareComponentPlanOptions(components::GraphIO(source), key_backend, addresses,
                                                key_options);
    ASSERT_TRUE(prepared);
    auto initial = PlanComponentGraph(components::GraphIO(source), key_backend, fixture_policy,
                                      addresses, key_options);
    ASSERT_TRUE(initial);
    const auto ranges = bindings->Resolve(*initial, addresses);
    ASSERT_TRUE(ranges);
    const auto view = bindings->boundary_view();
    ASSERT_TRUE(view);
    planner::ComparisonSiteSelector selector;
    selector.source_graph_identity = source.graph.identity();
    selector.kind = planner::ComparisonSiteKind::kFusedNode;
    selector.member_origins = {{harness->fused_site_nodes.front()}};
    auto inventory = planner::BoundComparison(*view, selector, harness->funder, *harness->registry,
                                              key_backend.device, fixture_policy, *prepared,
                                              component_ports ? nullptr : &shape);
    ASSERT_TRUE(inventory);
    auto alternative = planner::MakeAlternative(*inventory);
    ASSERT_TRUE(alternative);
    auto candidate = planner::PlanAlternative(
        *inventory, std::move(*alternative), *harness->registry, key_backend.device, fixture_policy,
        *prepared, component_ports ? nullptr : &shape);
    ASSERT_TRUE(candidate);
    std::vector<std::uint32_t> offsets{0};
    std::vector<ir::NodeId> members;
    for (const auto& group : inventory->site.accepted_member_origins) {
      members.insert(members.end(), group.begin(), group.end());
      offsets.push_back(static_cast<std::uint32_t>(members.size()));
    }
    auto boundary = components::MakeComparisonRegionBoundary(
        {.source = *view, .member_offsets = offsets, .members = members}, harness->funder);
    ASSERT_TRUE(boundary);
    const auto left = whole_seal(inventory->incumbent.plan, inventory->source_boundary,
                                 harness->registry->SelectionDigest());
    const auto right = whole_seal(candidate->plan, inventory->source_boundary,
                                  harness->registry->SelectionDigest());
    auto whole_key = RecipeComparisonKeyForWholePrograms(
        inventory->source_boundary, *boundary, inventory->site, execution::Phase::kPrefill,
        harness->registry->SelectionDigest(), left, right);
    ASSERT_TRUE(whole_key);
    registered_key = *whole_key;
  }
  if (failure == CumulativeFailure::kValidRowMiss) {
    registered_key.subject = base::Sha256{}.Update("valid but nonmatching row subject").Finish();
    ASSERT_NE(registered_key, **key);
  }
  auto registry = RegistryWithRecipe(
      registered_key, quality_override ? 0 : 120, quality_override ? 0 : 100, true,
      quality_override ? execution::RecipeComparisonKind::kRejectIncumbentQuality
                       : execution::RecipeComparisonKind::kLatency,
      quality_override ? std::optional{**key} : std::nullopt);
  ASSERT_TRUE(registry);
  if (quality_override) {
    ASSERT_NE(registry->FindRecipeComparison(**key), nullptr);
    EXPECT_EQ(execution::RecipeComparisonPreference(*registry->FindRecipeComparison(**key)),
              execution::RecipePreference::kIncumbent);
  }
  auto account = std::make_shared<Funder>();
  if (failure == CumulativeFailure::kSetupFunding) account->deny_at = 1;
  auto backing = std::make_shared<kernels::ConstructionFunder>();
  backing->charge = [account](std::uint64_t bytes) { return account->Charge(bytes); };
  backing->release = [account](std::uint64_t bytes) { account->Release(bytes); };
  auto observation = std::make_shared<ProgramObservation>();
  std::uint32_t lower_count = 0;
  std::uint32_t comparison_lower_count = 0;
  const kernels::StepProgram* first_program = nullptr;
  std::vector<kernels::ComparisonPlanIdentity> lowered_identities;
  std::vector<std::array<bool, 2>> lowered_decomposition;
  kernels::StepBackend backend{.registry = &*registry,
                               .device = {.device_class = "test", .arch = 121},
                               .construction_funder = backing};
  backend.lower = [&](const kernels::LowerInput& input)
      -> std::expected<std::unique_ptr<kernels::StepProgram>, std::string> {
    ++lower_count;
    if (input.comparison && input.comparison_region) ++comparison_lower_count;
    if (failure == CumulativeFailure::kLower)
      return std::unexpected("scripted first Lower refusal");
    auto selected = ResolveSelected(*registry, input);
    if (!selected) return std::unexpected(selected.error().detail);
    kernels::ComparisonRecipeSeal whole_recipe;
    whole_recipe.policy = input.plan.policy_digest;
    whole_recipe.registry = registry->SelectionDigest();
    if (whole_ordinary) {
      if (!input.comparison || !input.comparison->expected_boundary)
        return std::unexpected("whole comparison needs its owned boundary");
      whole_recipe =
          whole_seal(input.plan, *input.comparison->expected_boundary, registry->SelectionDigest());
    }
    ProjectionScript script;
    script.lower_ordinal = lower_count;
    if (failure == CumulativeFailure::kValidRowMiss || failure == CumulativeFailure::kBindDrift ||
        failure == CumulativeFailure::kSelfRemainderDrift ||
        failure == CumulativeFailure::kPriorChosenDrift ||
        failure == CumulativeFailure::kPendingOwnerDenied ||
        failure == CumulativeFailure::kProofPaymentDenied ||
        failure == CumulativeFailure::kLedgerCommitDenied || quality_override) {
      script.certificate_supported = true;
      script.remainder_same = true;
    }
    if (lower_count == 1 && failure == CumulativeFailure::kPendingOwnerDenied) {
      observation->injection_account = account;
      observation->inject_pending_owner_failure = true;
    }
    if (lower_count == 1 && (failure == CumulativeFailure::kProofPaymentDenied ||
                             failure == CumulativeFailure::kLedgerCommitDenied)) {
      observation->injection_account = account;
      observation->inject_remainder_failure_delta =
          failure == CumulativeFailure::kProofPaymentDenied ? 1 : 2;
    }
    if (failure == CumulativeFailure::kPriorChosenDrift && lower_count == 3) {
      script.mutate_local_recipe = true;
      if (!harness->fused_site_nodes.empty())
        script.mutate_local_origin = harness->fused_site_nodes.front();
    }
    if (failure == CumulativeFailure::kPriorChosenDrift) {
      lowered_identities.push_back({.source_graph_identity = source.graph.identity(),
                                    .planned_graph_identity = input.plan.graph.identity()});
      EXPECT_EQ(harness->fused_site_nodes.size(), 2U);
      if (harness->fused_site_nodes.size() != 2) return std::unexpected("missing fake sites");
      std::array<bool, 2> decomposed{};
      const auto local_calls = tests::CaptureComparisonLocalCalls(input, *registry);
      for (std::size_t site = 0; site < decomposed.size(); ++site) {
        const auto target = harness->fused_site_nodes[site];
        std::size_t primitive_members = 0;
        for (const auto& call : local_calls) {
          if (std::ranges::find(call.original_members, target) == call.original_members.end())
            continue;
          if (call.operation == ir::OpKeyName(ir::Primitive(ir::Op::kRmsNorm)) ||
              call.operation == ir::OpKeyName(ir::Primitive(ir::Op::kMul)))
            ++primitive_members;
        }
        decomposed[site] = primitive_members >= 2;
      }
      lowered_decomposition.push_back(decomposed);
    }
    auto program = std::make_unique<ProjectingProgram>(
        std::move(*selected), source.graph, whole_recipe,
        kernels::ComparisonPlanIdentity{.source_graph_identity = source.graph.identity(),
                                        .planned_graph_identity = input.plan.graph.identity()},
        observation, script, true, failure == CumulativeFailure::kAbsentIdentity,
        ready_recipe ? tests::CaptureComparisonLocalCalls(input, *registry)
                     : std::vector<tests::ComparisonLocalCall>{},
        ready_recipe);
    if (!first_program) first_program = program.get();
    return std::unique_ptr<kernels::StepProgram>(std::move(program));
  };
  kernels::ProgramValidator validate;
  if (failure == CumulativeFailure::kValidator)
    validate = [](const kernels::LowerInput&,
                  const kernels::StepProgram&) -> std::expected<void, std::string> {
      return std::unexpected("scripted first validator refusal");
    };
  planner::PlanOptions options{.shape = component_ports ? nullptr : &shape};
  options.staged.emplace();
  for (const auto& input : source.inputs) options.staged->push_back(input.tensor);
  options.alignment_staged = options.staged;
  if (failure != CumulativeFailure::kSetupFunding) {
    // Keep a malformed fake fixture from silently testing ordinary fallback.
    auto bindings = component_ports ? ComparisonSourceBindings::Create(
                                          harness->funder, components::GraphIO(source), 0)
                                    : ComparisonSourceBindings::Create(harness->funder, source, 0);
    ASSERT_TRUE(bindings) << (bindings ? "" : bindings.error().detail);
    auto prepared =
        PrepareComponentPlanOptions(components::GraphIO(source), backend, addresses, options);
    ASSERT_TRUE(prepared) << (prepared ? "" : prepared.error());
    auto baseline = PlanComponentGraph(components::GraphIO(source), backend, fixture_policy,
                                       addresses, options);
    ASSERT_TRUE(baseline) << (baseline ? "" : baseline.error());
    auto ranges = bindings->Resolve(*baseline, addresses);
    ASSERT_TRUE(ranges) << (ranges ? "" : ranges.error().detail);
    auto view = bindings->boundary_view();
    ASSERT_TRUE(view);
    const std::array filter{(**key).discovery_operation};
    auto discovery =
        planner::DiscoverComparisonPlan(*view, filter, false, harness->funder, *registry,
                                        backend.device, fixture_policy, *prepared);
    ASSERT_TRUE(discovery) << (discovery ? "" : discovery.error().detail);
    ASSERT_TRUE(std::ranges::any_of(discovery->discovery.tickets(), [](const auto& ticket) {
      return ticket.boundary_representable && ticket.member_offsets.size() >= 2;
    })) << "the source must supply a representable production discovery ticket";
    if (failure == CumulativeFailure::kUnproved) {
      const auto ticket_it =
          std::ranges::find_if(discovery->discovery.tickets(), [](const auto& ticket) {
            return ticket.boundary_representable && ticket.member_offsets.size() >= 2;
          });
      ASSERT_NE(ticket_it, discovery->discovery.tickets().end());
      planner::Policy policy;
      planner::ComparisonPlanningEnvironment environment(*view, *registry, backend.device, policy,
                                                         *prepared, &shape);
      auto ledger = planner::ComparisonDecisionLedger::Create(environment, harness->funder);
      ASSERT_TRUE(ledger) << (ledger ? "" : ledger.error().detail);

      planner::ComparisonSiteSelector selector;
      selector.source_graph_identity = discovery->discovery.source_identity();
      selector.kind = ticket_it->kind;
      selector.pattern = std::string(ticket_it->pattern);
      selector.version = ticket_it->version;
      selector.member_owner_segments = ticket_it->owner_segments;
      selector.boundary_ports = ticket_it->boundary_ports;
      for (std::size_t group = 0; group + 1 < ticket_it->member_offsets.size(); ++group) {
        const auto first = ticket_it->member_offsets[group];
        const auto last = ticket_it->member_offsets[group + 1];
        ASSERT_LE(first, last);
        ASSERT_LE(last, ticket_it->member_origins.size());
        selector.member_origins.emplace_back(ticket_it->member_origins.begin() + first,
                                             ticket_it->member_origins.begin() + last);
      }
      auto trial = planner::CaptureComparisonTrial(environment, *ledger, selector, harness->funder);
      ASSERT_TRUE(trial) << (trial ? "" : trial.error().detail);
      auto alternative = planner::MakeAlternative(*trial);
      ASSERT_TRUE(alternative) << (alternative ? "" : alternative.error().detail);
      auto candidate =
          planner::PlanAlternative(environment, *trial, std::move(*alternative), *ledger);
      SCOPED_TRACE(::testing::Message()
                   << "first ticket kind=" << static_cast<unsigned>(ticket_it->kind)
                   << " pattern=" << ticket_it->pattern << " version=" << ticket_it->version
                   << " operation=" << ir::OpKeyName(ticket_it->registered_operation)
                   << " member groups=" << ticket_it->member_offsets.size() - 1
                   << " first source node="
                   << (ticket_it->member_origins.empty()
                           ? -1LL
                           : static_cast<long long>(ticket_it->member_origins.front().v)));
      ASSERT_TRUE(candidate) << (candidate ? "" : candidate.error().detail);
      auto candidate_ranges = bindings->Resolve(candidate->plan, addresses);
      ASSERT_TRUE(candidate_ranges) << (candidate_ranges ? "" : candidate_ranges.error().detail);
    }
  }
  auto result =
      component_ports
          ? BuildComponentProgram(components::GraphIO(source), backend, fixture_policy, addresses,
                                  std::move(options), std::nullopt, validate, {})
          : BuildCumulativeRecipeProgram(components::GraphIO(source), source, shape, 0, 16, backend,
                                         {}, addresses, std::move(options), validate, {});
  EXPECT_EQ(observation->bind_count, 0U);
  EXPECT_EQ(observation->run_count, 0U);
  EXPECT_EQ(observation->record_count, 0U);
  if (failure == CumulativeFailure::kLower || failure == CumulativeFailure::kValidator) {
    ASSERT_FALSE(result);
    EXPECT_NE(result.error().find("scripted first"), std::string::npos) << result.error();
    EXPECT_EQ(lower_count, 1U);
    EXPECT_EQ(comparison_lower_count, 1U);
    EXPECT_EQ(observation->destruction_count, failure == CumulativeFailure::kValidator ? 1U : 0U);
    EXPECT_EQ(account->active, 0U);
    return;
  }
  ASSERT_TRUE(result) << (result ? "" : result.error());
  if (whole_ordinary) {
    ASSERT_EQ(lower_count, 2U);
    auto& adopted = **result;
    ASSERT_TRUE(adopted.comparison_recipe());
    EXPECT_EQ(adopted.comparison_recipe()->execution_method,
              kernels::ComparisonExecutionMethod::kCaptured)
        << "the chosen alternative itself is capturable; the pair's method still controls reuse";
    RecordingStepLaunch launch;
    ASSERT_TRUE(adopted.Bind(launch));
    EXPECT_FALSE(adopted.capturable());
    EXPECT_TRUE(adopted.Run(launch, {}));
    const kernels::StepLaunch::Body body = [] {
      return std::expected<void, kernels::StepFailure>{};
    };
    EXPECT_FALSE(adopted.Record(launch, {}, body));
    EXPECT_EQ(observation->record_count, 0U);
    EXPECT_EQ(launch.capture_count, 0U);
    observation->mutate_whole_method_on_next_bind = true;
    EXPECT_FALSE(adopted.Bind(launch));
    EXPECT_FALSE(adopted.Run(launch, {}));
    EXPECT_FALSE(adopted.TryUpdateGeneratedInputs(shape));
    adopted.ConstructionPaid();
    result->reset();
    EXPECT_EQ(account->active, 0U);
    return;
  }
  if (failure == CumulativeFailure::kPriorChosenDrift) {
    auto* adopted = result->get();
    ASSERT_NE(adopted, nullptr);
    ASSERT_EQ(harness->fused_site_nodes.size(), 2U);
    ASSERT_EQ(lower_count, 3U) << "both exact local rows must be exercised";
    ASSERT_EQ(lowered_identities.size(), 3U);
    ASSERT_EQ(lowered_decomposition.size(), 3U);
    EXPECT_EQ(lowered_decomposition[0], (std::array<bool, 2>{false, false}));
    EXPECT_EQ(lowered_decomposition[1], (std::array<bool, 2>{true, false}));
    EXPECT_EQ(lowered_decomposition[2], (std::array<bool, 2>{true, true}));
    const auto final_identity = adopted->comparison_plan_identity();
    ASSERT_TRUE(final_identity);
    EXPECT_EQ(final_identity->source_graph_identity, lowered_identities[1].source_graph_identity);
    EXPECT_EQ(final_identity->planned_graph_identity, lowered_identities[1].planned_graph_identity);
    EXPECT_NE(final_identity->planned_graph_identity, lowered_identities[2].planned_graph_identity);
    EXPECT_GE(observation->certificate_calls_by_lower[2], 2U)
        << "the second exact hit and prior-certificate revalidation must both occur";
    EXPECT_EQ(observation->destruction_count, 2U)
        << "the first incumbent and drifted second candidate must be destroyed";
    EXPECT_GT(observation->certificate_count, 0U)
        << "the second candidate reached prior-proof revalidation";
    EXPECT_EQ(observation->bind_count, 0U);
    EXPECT_EQ(adopted->bound(), nullptr);
    RecordingStepLaunch launch;
    ASSERT_TRUE(adopted->Bind(launch));
    ASSERT_NE(adopted->bound(), nullptr);
    EXPECT_EQ(observation->bound_certificate_drift_count, 0U);
    EXPECT_GT(account->active, 0U);
    adopted->ConstructionPaid();
    EXPECT_EQ(account->active, 0U);
    result->reset();
    EXPECT_EQ(observation->destruction_count, lower_count);
    EXPECT_EQ(account->active, 0U);
    return;
  }
  if (failure == CumulativeFailure::kPendingOwnerDenied ||
      failure == CumulativeFailure::kProofPaymentDenied ||
      failure == CumulativeFailure::kLedgerCommitDenied) {
    const bool pending_owner = failure == CumulativeFailure::kPendingOwnerDenied;
    const auto expected_delta = pending_owner                                       ? 1U
                                : failure == CumulativeFailure::kProofPaymentDenied ? 1U
                                                                                    : 2U;
    EXPECT_EQ(lower_count, pending_owner ? 1U : 2U);
    EXPECT_EQ(comparison_lower_count, lower_count);
    ASSERT_NE(first_program, nullptr);
    EXPECT_EQ(&(*result)->comparison_program(), first_program)
        << "a denied owner/proof/commit charge must preserve the exact current program";
    const auto returned_identity = (*result)->comparison_plan_identity();
    const auto first_identity = first_program->comparison_plan_identity();
    ASSERT_TRUE(returned_identity);
    ASSERT_TRUE(first_identity);
    EXPECT_EQ(returned_identity->planned_graph_identity, first_identity->planned_graph_identity);
    EXPECT_EQ(observation->bind_count, 0U);
    EXPECT_EQ(observation->run_count, 0U);
    EXPECT_EQ(observation->record_count, 0U);
    ASSERT_EQ(account->denied_charges, 1U) << "the intended paid operation must be denied";
    EXPECT_EQ(account->last_denied_charge, account->denial_armed_after + expected_delta);
    if (!pending_owner) EXPECT_EQ(observation->destruction_count, 1U);
    EXPECT_GT(account->active, 0U);
    (*result)->ConstructionPaid();
    EXPECT_EQ(account->active, 0U);
    result->reset();
    EXPECT_EQ(observation->destruction_count, lower_count);
    EXPECT_EQ(account->active, 0U);
    return;
  }
  if (failure == CumulativeFailure::kBindDrift) {
    auto* adopted = result->get();
    ASSERT_NE(adopted, nullptr);
    EXPECT_NE(adopted, first_program);
    ASSERT_NE(&adopted->comparison_program(), first_program)
        << "the exact registered row must select a candidate carrying retained proofs";
    RecordingStepLaunch launch;
    const auto certificates_before_first_bind = observation->certificate_count;
    ASSERT_TRUE(adopted->Bind(launch));
    EXPECT_GT(observation->certificate_count, certificates_before_first_bind);
    ASSERT_NE(adopted->bound(), nullptr);
    const auto certificates_before_mutation = observation->certificate_count;
    observation->mutate_certificate_on_next_bind = true;
    EXPECT_FALSE(adopted->Bind(launch));
    EXPECT_GT(observation->certificate_count, certificates_before_mutation);
    EXPECT_GT(observation->bound_certificate_drift_count, 0U);
    CheckClosedExecutionGates(*adopted, launch, *observation, shape);
    EXPECT_GT(account->active, 0U);
    adopted->ConstructionPaid();
    EXPECT_EQ(account->active, 0U);
    result->reset();
    EXPECT_EQ(observation->destruction_count, lower_count);
    EXPECT_EQ(account->active, 0U);
    return;
  }
  if (failure == CumulativeFailure::kSelfRemainderDrift) {
    auto* adopted = result->get();
    ASSERT_NE(adopted, nullptr);
    ASSERT_NE(&adopted->comparison_program(), first_program);
    RecordingStepLaunch launch;
    ASSERT_TRUE(adopted->Bind(launch));
    ASSERT_NE(adopted->bound(), nullptr);
    observation->bound_remainder_unproved = true;
    EXPECT_FALSE(adopted->Bind(launch));
    CheckClosedExecutionGates(*adopted, launch, *observation, shape);
    EXPECT_GT(observation->remainder_count, 0U);
    adopted->ConstructionPaid();
    EXPECT_EQ(account->active, 0U);
    result->reset();
    EXPECT_EQ(observation->destruction_count, lower_count);
    EXPECT_EQ(account->active, 0U);
    return;
  }
  ASSERT_NE(first_program, nullptr);
  EXPECT_EQ(&(*result)->comparison_program(), first_program)
      << "optional comparison cannot replace or wrap the unproved incumbent";
  if (failure == CumulativeFailure::kSetupFunding) {
    EXPECT_EQ(lower_count, 1U);
    EXPECT_EQ(comparison_lower_count, 0U);
  } else if (failure == CumulativeFailure::kAbsentIdentity) {
    EXPECT_EQ(lower_count, 1U);
    EXPECT_EQ(comparison_lower_count, 1U);
    EXPECT_EQ(observation->destruction_count, 0U);
  } else {
    EXPECT_GE(lower_count, 2U);
    EXPECT_EQ(comparison_lower_count, lower_count);
    EXPECT_EQ(observation->destruction_count, lower_count - 1);
    if (failure == CumulativeFailure::kValidRowMiss) {
      EXPECT_GT(observation->certificate_count, 0U);
      EXPECT_GT(observation->remainder_count, 0U);
    }
  }
  EXPECT_GT(account->active, 0U) << "the retained report must remain paid";
  (*result)->ConstructionPaid();
  EXPECT_EQ(account->active, 0U);
  result->reset();
  EXPECT_EQ(observation->destruction_count, lower_count);
  EXPECT_EQ(account->active, 0U);
}

TEST(CumulativeRecipeConstructionTest, ComponentPortsRetainWholeOrdinaryPairMethodAndBindGates) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kWholeOrdinary, true));
}

TEST(CumulativeRecipeConstructionTest, FirstLowerFailureIsTerminalAndRefundsStorage) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kLower));
}

TEST(CumulativeRecipeConstructionTest, OrdinaryWholeRowKeepsChosenCapturableArmOrdinary) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kWholeOrdinary));
}

TEST(CumulativeRecipeConstructionTest, ExactWholeQualityOverridesCompetingLocalLatency) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kWholeQualityOverridesLocal));
}

TEST(CumulativeRecipeConstructionTest, FirstValidationFailureIsTerminalAndRefundsStorage) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kValidator));
}

TEST(CumulativeRecipeConstructionTest, UnprovedTrialsKeepTheExactUnboundIncumbent) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kUnproved));
}

TEST(CumulativeRecipeConstructionTest, MissingInitialPlanIdentityKeepsTheExactIncumbent) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kAbsentIdentity));
}

TEST(CumulativeRecipeConstructionTest, SetupFundingRefusalFallsBackBeforeFirstLower) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kSetupFunding));
}

TEST(CumulativeRecipeConstructionTest, ValidButNonmatchingRowKeepsExactUnboundIncumbent) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kValidRowMiss));
}

TEST(CumulativeRecipeConstructionTest, BoundCertificateDriftClosesEveryWrapperExecutionGate) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kBindDrift));
}

TEST(CumulativeRecipeConstructionTest, BoundSelfRemainderDriftClosesEveryWrapperExecutionGate) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kSelfRemainderDrift));
}

TEST(CumulativeRecipeConstructionTest,
     LaterExactWinCannotReplaceAPreviouslyChosenLocalCertificate) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kPriorChosenDrift));
}

TEST(CumulativeRecipeConstructionTest, PendingWrapperFundingDenialKeepsOriginalProgram) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kPendingOwnerDenied));
}

TEST(CumulativeRecipeConstructionTest, RetainedProofFundingDenialKeepsOriginalProgram) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kProofPaymentDenied));
}

TEST(CumulativeRecipeConstructionTest, LedgerCommitFundingDenialKeepsOriginalProgram) {
  ASSERT_NO_FATAL_FAILURE(CheckCumulativeFailure(CumulativeFailure::kLedgerCommitDenied));
}

components::StepGraph SourceGraph() {
  const std::array<std::uint64_t, 2> dims{4, 1};
  format::TensorFormat format{.encoding = format::Encoding::kF32,
                              .layout = format::Layout::kDense,
                              .shape = *format::ShapeOf(dims)};
  ir::Builder builder;
  const auto input = builder.Input(format, "pair-input");
  builder.Output(input);
  auto graph = std::move(builder).Finish();
  EXPECT_TRUE(graph.has_value());
  return {.graph = std::move(*graph),
          .inputs = {},
          .weights = {},
          .state = {},
          .logits = {},
          .greedy = {},
          .features = {},
          .saves = {},
          .owner_products = {}};
}

TEST(ComparisonPairTest, PackageMoveKeepsSourceAndConsumingPreflightRefunds) {
  auto funder = std::make_shared<Funder>();
  auto made = ComparisonPair::Create(funder, SourceGraph(), {}, {}, 0);
  ASSERT_TRUE(made);
  const auto identity = (*made)->source().graph.identity();
  ASSERT_NE(identity, 0U);
  ASSERT_GT(funder->active, 0U);
  auto moved = std::move(*made);
  EXPECT_EQ(moved->source().graph.identity(), identity);
  auto closed = CloseComparisonPair(std::move(moved));
  EXPECT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ComparisonPairFailure::Code::kNotReady);
  EXPECT_EQ(funder->active, 0U);
}

TEST(ComparisonPairTest, RefusedPackageDoesNotPublishAnOwner) {
  auto funder = std::make_shared<Funder>();
  funder->deny = true;
  auto made = ComparisonPair::Create(funder, SourceGraph(), {}, {}, 0);
  ASSERT_FALSE(made);
  EXPECT_EQ(made.error().code, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(funder->active, 0U);
}

TEST(ComparisonPairTest, BindingsInstallOnceAfterPlanBeforeLoweringAndAreFunded) {
  auto harness =
      MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0, nullptr, false);
  ASSERT_TRUE(harness);
  ASSERT_NE(harness->pair, nullptr);
  kernels::Addresses addresses;
  auto missing = harness->pair->LoweringInput(ComparisonPairArm::kIncumbent, addresses);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, ComparisonPairFailure::Code::kNotReady);

  auto storage = harness->BindingStorage(harness->input_tensor, harness->weight_tensor,
                                         harness->input_bytes, harness->weight_bytes);
  const std::array<std::span<const kernels::ComparisonBindingRange>, 2> views{storage[0],
                                                                              storage[1]};
  const auto paid_before = harness->funder->active;
  harness->funder->deny = true;
  auto denied = harness->pair->InstallBindings(views);
  ASSERT_FALSE(denied);
  EXPECT_EQ(denied.error().code, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(harness->funder->active, paid_before);

  harness->funder->deny = false;
  auto installed = harness->pair->InstallBindings(views);
  ASSERT_TRUE(installed);
  auto repeated = harness->pair->InstallBindings(views);
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, ComparisonPairFailure::Code::kInvalid);

  auto lower = harness->pair->LoweringInput(ComparisonPairArm::kIncumbent, addresses);
  ASSERT_TRUE(lower);
  auto late = harness->pair->InstallBindings(views);
  ASSERT_FALSE(late);
  EXPECT_EQ(late.error().code, ComparisonPairFailure::Code::kInvalid);
}

TEST(ComparisonPairTest, CloseRejectsSharedLocalPolicyOrRegistryNotBoundToPlans) {
  for (const bool wrong_policy : {true, false}) {
    for (const bool wrong_registry : {true, false}) {
      if (!wrong_policy && !wrong_registry) continue;
      ProjectionScript script{.expand_first = false,
                              .wrong_local_policy = wrong_policy,
                              .wrong_local_registry = wrong_registry};
      auto harness = MakePairHarness(script, script, false, 0, nullptr, true);
      ASSERT_TRUE(harness);
      ASSERT_NE(harness->pair, nullptr);
      const auto before = harness->funder->active;
      auto closed = CloseComparisonPair(std::move(harness->pair));
      ASSERT_FALSE(closed);
      EXPECT_EQ(closed.error().code, ComparisonPairFailure::Code::kUnproved);
      EXPECT_LT(harness->funder->active, before);
      EXPECT_EQ(harness->funder->active, 0U);
    }
  }
}

TEST(ComparisonPairTest, ExpansionUnionKeepsAllArmsAndSortsSourceMembers) {
  const std::array incumbent{ir::NodeId{7}, ir::NodeId{3}, ir::NodeId{7}};
  const std::array alternative{ir::NodeId{5}, ir::NodeId{3}, ir::NodeId{1}};
  std::array<std::uint8_t, 8> membership{};
  std::vector<ir::NodeId> additions;
  additions.reserve(membership.size());

  const auto merged = MergeComparisonExpansions(incumbent, alternative, membership, additions);
  ASSERT_EQ(merged, ComparisonExpansionMerge::kProgress);
  const std::array expected{ir::NodeId{1}, ir::NodeId{3}, ir::NodeId{5}, ir::NodeId{7}};
  EXPECT_TRUE(std::ranges::equal(additions, expected));
  EXPECT_EQ(membership[1], 1);
  EXPECT_EQ(membership[3], 1);
  EXPECT_EQ(membership[5], 1);
  EXPECT_EQ(membership[7], 1);
}

TEST(ComparisonPairTest, ExpansionUnionRefusesNoProgressAndOutOfRangeMembers) {
  const std::array initial{ir::NodeId{2}, ir::NodeId{4}};
  std::array<std::uint8_t, 5> membership{};
  std::vector<ir::NodeId> additions;
  additions.reserve(membership.size());
  ASSERT_EQ(MergeComparisonExpansions(initial, {}, membership, additions),
            ComparisonExpansionMerge::kProgress);

  const auto unchanged = MergeComparisonExpansions(initial, initial, membership, additions);
  EXPECT_EQ(unchanged, ComparisonExpansionMerge::kNoProgress);
  const std::array invalid{ir::NodeId{5}};
  const auto out_of_range = MergeComparisonExpansions(invalid, {}, membership, additions);
  EXPECT_EQ(out_of_range, ComparisonExpansionMerge::kOutOfRange);
  EXPECT_EQ(additions.size(), 2U);
}

TEST(ComparisonPairTest, ExpansionUnionRefusesWhenPaidCapacityIsExhausted) {
  const std::array initial{ir::NodeId{1}};
  const std::array next{ir::NodeId{3}};
  std::array<std::uint8_t, 4> membership{};
  std::vector<ir::NodeId> additions;
  additions.reserve(1);
  ASSERT_EQ(MergeComparisonExpansions(initial, {}, membership, additions),
            ComparisonExpansionMerge::kProgress);
  EXPECT_EQ(MergeComparisonExpansions(next, {}, membership, additions),
            ComparisonExpansionMerge::kCapacity);
  EXPECT_EQ(additions.size(), 1U);
}

TEST(ComparisonPairTest, BothBoundArmsExpandThenCloseTheSameRegion) {
  auto harness = MakePairHarness({.expansion = ir::NodeId{2}}, {.expansion = ir::NodeId{4}});
  ASSERT_TRUE(harness);
  auto closed = CloseComparisonPair(std::move(harness->pair));
  if (!closed) FAIL() << closed.error().detail;
  ASSERT_TRUE((*closed)->closed());
  EXPECT_TRUE(harness->incumbent->union_seen());
  EXPECT_TRUE(harness->alternative->union_seen());
  EXPECT_EQ(harness->incumbent->final_node_count(), 3U);
  EXPECT_EQ(harness->alternative->final_node_count(), 3U);
  EXPECT_EQ(harness->incumbent->projection_count(), 2U);
  EXPECT_EQ(harness->alternative->projection_count(), 2U);
  EXPECT_EQ((*closed)->value_ports(ComparisonPairArm::kIncumbent).size(),
            (*closed)->value_ports(ComparisonPairArm::kAlternative).size());
}

TEST(ComparisonPairTest, NoProgressAndStaleTokensFailBeforePublishingAndRefund) {
  auto no_progress = MakePairHarness({.expansion = ir::NodeId{3}}, {.expansion = ir::NodeId{3}});
  ASSERT_TRUE(no_progress);
  auto refused = CloseComparisonPair(std::move(no_progress->pair));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kNoProgress);
  EXPECT_EQ(no_progress->funder->active, 0U);

  auto stale = MakePairHarness({.expansion = ir::NodeId{2}, .stale = true},
                               {.expansion = ir::NodeId{4}, .stale = true});
  ASSERT_TRUE(stale);
  auto stale_result = CloseComparisonPair(std::move(stale->pair));
  ASSERT_FALSE(stale_result);
  EXPECT_EQ(stale_result.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(stale->funder->active, 0U);
}

TEST(ComparisonPairTest, DeniedCandidateBoundaryRebuildRefundsThePreparedPair) {
  auto harness = MakePairHarness({.expansion = ir::NodeId{2}}, {.expansion = ir::NodeId{4}});
  ASSERT_TRUE(harness);
  harness->funder->deny = true;
  auto refused = CloseComparisonPair(std::move(harness->pair));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, OneClosedArmIsReprojectedAfterTheOtherExpands) {
  auto harness = MakePairHarness({.expansion = ir::NodeId{2}, .expand_first = false},
                                 {.expansion = ir::NodeId{4}});
  ASSERT_TRUE(harness);
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_TRUE(closed);
  EXPECT_TRUE(harness->incumbent->union_seen());
  EXPECT_TRUE(harness->alternative->union_seen());
  EXPECT_EQ(harness->incumbent->final_node_count(), 2U);
  EXPECT_EQ(harness->alternative->final_node_count(), 2U);
  EXPECT_EQ(harness->incumbent->projection_count(), 2U);
  EXPECT_EQ(harness->alternative->projection_count(), 2U);
}

TEST(ComparisonPairTest, UnprovedAndOutOfRangeExpansionFailClosedAndRefund) {
  auto unproved = MakePairHarness({.expansion = ir::NodeId{2}}, {.unproved_first = true});
  ASSERT_TRUE(unproved);
  auto unproved_result = CloseComparisonPair(std::move(unproved->pair));
  ASSERT_FALSE(unproved_result);
  EXPECT_EQ(unproved_result.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(unproved->funder->active, 0U);

  auto out_of_range = MakePairHarness({.expansion = ir::NodeId{2}},
                                      {.expansion = ir::NodeId{0}, .out_of_range = true});
  ASSERT_TRUE(out_of_range);
  auto range_result = CloseComparisonPair(std::move(out_of_range->pair));
  ASSERT_FALSE(range_result);
  EXPECT_EQ(range_result.error().code, ComparisonPairFailure::Code::kStale);
  EXPECT_EQ(out_of_range->funder->active, 0U);
}

TEST(ComparisonPairTest, DuplicateValuePortCannotOmitARegionOutput) {
  auto harness = MakePairHarness({.expansion = ir::NodeId{2}, .duplicate_ports = true},
                                 {.expansion = ir::NodeId{4}, .duplicate_ports = true});
  ASSERT_TRUE(harness);
  auto refused = CloseComparisonPair(std::move(harness->pair));
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, BoundProgramsMustBelongToTheirOwnRetainedPlans) {
  auto harness = MakePairHarness({.expansion = ir::NodeId{2}}, {.expansion = ir::NodeId{4}}, true);
  ASSERT_TRUE(harness);
  EXPECT_NE(harness->pair, nullptr);
  EXPECT_EQ(harness->incumbent, nullptr);
  EXPECT_EQ(harness->alternative, nullptr);
  harness->pair.reset();
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, MatchingSealCannotHideWrongLivePlannedIdentity) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, true);
  ASSERT_TRUE(harness);
  EXPECT_EQ(harness->incumbent, nullptr);
  EXPECT_EQ(harness->alternative, nullptr);
  harness->pair.reset();
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, EmptyLanesKeepTheNormalStreamCallbackInert) {
  std::uint32_t callbacks = 0;
  planner::PlanOptions options;
  options.on_stream = [&](std::string_view) {
    ++callbacks;
    ADD_FAILURE() << "empty lane tags must not consult stream routing";
    return false;
  };
  const ProjectionScript script{.expand_first = false, .certificate_supported = true};
  auto ordinary = MakePairHarness(script, script, false, 0, nullptr, true, false, true, true, true);
  auto normal = MakePairHarness(script, script, false, 0, &options, true, false, true, true, true);
  ASSERT_TRUE(ordinary);
  ASSERT_TRUE(normal);
  EXPECT_EQ(callbacks, 0U);
  for (const auto arm : {ComparisonPairArm::kIncumbent, ComparisonPairArm::kAlternative}) {
    const auto* expected = ordinary->pair->plan(arm);
    const auto* actual = normal->pair->plan(arm);
    ASSERT_NE(expected, nullptr);
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(actual->plan.graph.Digest(), expected->plan.graph.Digest());
    EXPECT_EQ(actual->plan.policy_digest, expected->plan.policy_digest);
    EXPECT_EQ(actual->plan.steps, expected->plan.steps);
    EXPECT_TRUE(actual->plan.regions.empty());
    EXPECT_TRUE(expected->plan.regions.empty());
  }
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  const auto expected_key = DescribeRecipeComparison(*ordinary->pair, *ordinary->registry, ready);
  const auto actual_key = DescribeRecipeComparison(*normal->pair, *normal->registry, ready);
  ASSERT_TRUE(expected_key);
  ASSERT_TRUE(actual_key);
  ASSERT_TRUE(*expected_key);
  ASSERT_TRUE(*actual_key);
  EXPECT_EQ(**actual_key, **expected_key);

  // A real lane request is still outside this pair's proof domain, regardless
  // of whether a callback is supplied.
  for (const bool callback : {false, true}) {
    auto funder = std::make_shared<Funder>();
    auto made = ComparisonPair::Create(funder, SourceGraph(), {}, {}, 0);
    ASSERT_TRUE(made);
    planner::PlanOptions routed;
    routed.lanes.push_back({ir::NodeId{0}, {.lane = 1, .region = 1}});
    if (callback) routed.on_stream = options.on_stream;
    const auto refused = (*made)->Plan({}, *normal->registry, {}, {}, routed);
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kUnproved);
    EXPECT_EQ(callbacks, 0U);
  }
}

TEST(ComparisonPairTest, BorrowedPlannerCallbacksAreNotCopiedBeforeAdmission) {
  auto copies = std::make_shared<std::uint32_t>(0);
  planner::PlanOptions options;
  options.written = CopyCountedRootBytes(copies);
  *copies = 0;
  auto harness =
      MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0, &options);
  ASSERT_TRUE(harness);
  EXPECT_EQ(*copies, 0U);
}

TEST(ComparisonPairTest, DescriptorProgramsBindSameOwnersThenCloseAndCapture) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, false, true);
  ASSERT_TRUE(harness);
  ASSERT_NE(harness->pair, nullptr);
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  harness->incumbent->ExpectBindLaunch(&incumbent_launch);
  harness->alternative->ExpectBindLaunch(&alternative_launch);

  auto premature = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
  ASSERT_FALSE(premature);
  EXPECT_EQ(harness->incumbent_observation->bind_count, 0U);
  EXPECT_EQ(harness->alternative_observation->bind_count, 0U);

  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  const auto incumbent_identity = harness->incumbent->comparison_plan_identity();
  const auto alternative_identity = harness->alternative->comparison_plan_identity();
  ASSERT_TRUE(incumbent_identity);
  ASSERT_TRUE(alternative_identity);
  auto bound = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
  ASSERT_TRUE(bound);
  EXPECT_EQ(harness->incumbent_observation->bind_count, 1U);
  EXPECT_EQ(harness->alternative_observation->bind_count, 1U);
  EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
  EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  EXPECT_EQ(incumbent_launch.launch_count, 0U);
  EXPECT_EQ(alternative_launch.launch_count, 0U);

  auto closed = CloseComparisonPair(std::move(harness->pair));
  if (!closed) FAIL() << closed.error().detail;
  ASSERT_TRUE((*closed)->closed());
  EXPECT_EQ(harness->incumbent->comparison_plan_identity()->planned_graph_identity,
            incumbent_identity->planned_graph_identity);
  EXPECT_EQ(harness->alternative->comparison_plan_identity()->planned_graph_identity,
            alternative_identity->planned_graph_identity);
  ASSERT_TRUE((*closed)->CaptureRegion(ComparisonPairArm::kIncumbent, incumbent_launch, {}));
  ASSERT_TRUE((*closed)->CaptureRegion(ComparisonPairArm::kAlternative, alternative_launch, {}));
  EXPECT_EQ(harness->incumbent_observation->record_count, 1U);
  EXPECT_EQ(harness->alternative_observation->record_count, 1U);
  EXPECT_EQ(incumbent_launch.capture_count, 1U);
  EXPECT_EQ(alternative_launch.capture_count, 1U);
  EXPECT_EQ(incumbent_launch.launch_count, 0U);
  EXPECT_EQ(alternative_launch.launch_count, 0U);
  closed->reset();
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, RepeatedCaptureRefusesUnpreservedInputsBeforeRecording) {
  const ProjectionScript script{.expand_first = false, .inputs_preserved = false};
  auto harness = MakePairHarness(script, script, false, 0, nullptr, true, false, true);
  ASSERT_TRUE(harness);
  ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  ASSERT_TRUE(harness->pair->BindPrograms(incumbent_launch, alternative_launch));
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_TRUE(closed);
  for (const auto arm : {ComparisonPairArm::kIncumbent, ComparisonPairArm::kAlternative}) {
    auto& launch = arm == ComparisonPairArm::kIncumbent ? incumbent_launch : alternative_launch;
    EXPECT_FALSE((*closed)->CaptureRegion(arm, launch, {}, 2));
    EXPECT_EQ(launch.capture_count, 0U);
    EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
    EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  }
  ASSERT_TRUE((*closed)->CaptureRegion(ComparisonPairArm::kIncumbent, incumbent_launch, {}, 1));
  ASSERT_TRUE((*closed)->CaptureRegion(ComparisonPairArm::kAlternative, alternative_launch, {}, 1));
}

TEST(ComparisonPairTest, RepeatedRegionCaptureSeparatesPrefixFromAuthenticatedBodyReplay) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, false, true);
  ASSERT_TRUE(harness);
  ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  ASSERT_TRUE(harness->pair->BindPrograms(incumbent_launch, alternative_launch));
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_TRUE(closed);
  ASSERT_TRUE((*closed)->closed());

  EXPECT_FALSE((*closed)->CaptureRegion(ComparisonPairArm::kIncumbent, incumbent_launch, {}, 0));
  EXPECT_EQ(incumbent_launch.capture_count, 0U);
  ASSERT_TRUE((*closed)->CaptureRegion(ComparisonPairArm::kIncumbent, incumbent_launch, {}, 128));
  ASSERT_TRUE(
      (*closed)->CaptureRegion(ComparisonPairArm::kAlternative, alternative_launch, {}, 128));
  EXPECT_EQ(harness->incumbent_observation->region_count, 128U);
  EXPECT_EQ(harness->alternative_observation->region_count, 128U);
  EXPECT_EQ(harness->incumbent_observation->prefix_count, 0U);
  const auto incumbent_inventory =
      (*closed)->captured_region_inventory(ComparisonPairArm::kIncumbent, incumbent_launch);
  ASSERT_TRUE(incumbent_inventory);
  EXPECT_EQ(incumbent_inventory->recorded_nodes, 1U);
  EXPECT_EQ(incumbent_inventory->repetitions, 128U);
  EXPECT_FALSE(
      (*closed)->captured_region_inventory(ComparisonPairArm::kIncumbent, alternative_launch));

  EXPECT_FALSE((*closed)->PrepareFrame(ComparisonPairArm::kIncumbent, alternative_launch, {}));
  EXPECT_FALSE(
      (*closed)->ReplayCapturedRegion(ComparisonPairArm::kIncumbent, alternative_launch, {}));
  ASSERT_TRUE((*closed)->PrepareFrame(ComparisonPairArm::kIncumbent, incumbent_launch, {}));
  EXPECT_EQ(harness->incumbent_observation->prefix_count, 1U);
  EXPECT_EQ(incumbent_launch.launch_count, 0U);
  ASSERT_TRUE((*closed)->ReplayCapturedRegion(ComparisonPairArm::kIncumbent, incumbent_launch, {}));
  EXPECT_EQ(harness->incumbent_observation->prefix_count, 1U);
  EXPECT_EQ(incumbent_launch.launch_count, 1U);
  ASSERT_TRUE((*closed)->RunFrame(ComparisonPairArm::kAlternative, alternative_launch, {}));
  EXPECT_EQ(harness->alternative_observation->prefix_count, 1U);
  EXPECT_EQ(alternative_launch.launch_count, 1U);
  closed->reset();
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, CapturedRegionFromAnotherLaunchIsNeverInstalled) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, false, true);
  ASSERT_TRUE(harness);
  ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
  RecordingStepLaunch launch;
  RecordingStepLaunch foreign_launch;
  ASSERT_TRUE(harness->pair->BindPrograms(launch, foreign_launch));
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_TRUE(closed);
  launch.owner_override = &foreign_launch;
  EXPECT_FALSE((*closed)->CaptureRegion(ComparisonPairArm::kIncumbent, launch, {}, 2));
  EXPECT_FALSE((*closed)->captured_region_inventory(ComparisonPairArm::kIncumbent, launch));
  EXPECT_FALSE((*closed)->PrepareFrame(ComparisonPairArm::kIncumbent, launch, {}));
  EXPECT_FALSE((*closed)->ReplayCapturedRegion(ComparisonPairArm::kIncumbent, launch, {}));
  EXPECT_EQ(launch.launch_count, 0U);
  closed->reset();
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, ReadyBoundaryKeyRequiresBothDescriptorProofsAndBoundCertificate) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  {
    auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                   nullptr, true, false, true, true);
    ASSERT_TRUE(harness);
    auto missing = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(missing);
    EXPECT_FALSE(*missing) << "default-unsupported applicability certificate is fail-closed";
    harness->pair.reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }
  {
    auto harness = MakePairHarness({.expand_first = false,
                                    .certificate_supported = true,
                                    .mismatch_certificate_descriptor = true},
                                   {.expand_first = false, .certificate_supported = true}, false, 0,
                                   nullptr, true, false, true, true);
    ASSERT_TRUE(harness);
    auto mismatch = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(mismatch);
    EXPECT_FALSE(*mismatch)
        << "applicability and executable proofs must agree on persistent body identity";
    harness->pair.reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }
  {
    auto harness = MakePairHarness({.expand_first = false, .certificate_supported = true},
                                   {.expand_first = false, .certificate_supported = true}, false, 0,
                                   nullptr, true, false, true, true);
    ASSERT_TRUE(harness);
    auto legacy = DescribeRecipeComparison(*harness->pair, *harness->registry);
    ASSERT_TRUE(legacy);
    ASSERT_TRUE(*legacy);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(described);
    ASSERT_TRUE(*described);
    const auto domain = ComparisonPreparationDomain(ready, execution::Phase::kPrefill);
    ASSERT_TRUE(domain);
    EXPECT_EQ((**described).preparation_domain, *domain);
    EXPECT_NE((**described).preparation_domain, (**legacy).preparation_domain);
    EXPECT_EQ((**described).incumbent, (**legacy).incumbent);
    RecordingStepLaunch incumbent_launch;
    RecordingStepLaunch alternative_launch;
    auto bound = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
    ASSERT_TRUE(bound);
    auto closed = CloseComparisonPair(std::move(harness->pair));
    ASSERT_TRUE(closed) << (closed ? "" : closed.error().detail);
    EXPECT_TRUE((*closed)->closed());
    EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
    EXPECT_EQ(harness->alternative_observation->record_count, 0U);
    closed->reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }
}

TEST(ComparisonPairTest, ReadyBoundaryAdmitsOnlyExactF32FeaturesSlots) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  ProjectionScript script{
      .expand_first = false, .certificate_supported = true, .remainder_same = true};
  {
    auto harness = MakePairHarness(script, script, false, 0, nullptr, true, false, true, false,
                                   false, false, FeatureInputFixture{});
    ASSERT_TRUE(harness);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(described);
    ASSERT_TRUE(*described);
    EXPECT_EQ((**described).preparation_domain,
              *ComparisonPreparationDomain(ready, execution::Phase::kPrefill));

    auto legacy = DescribeRecipeComparison(*harness->pair, *harness->registry);
    ASSERT_TRUE(legacy);
    EXPECT_FALSE(*legacy) << "Features remain outside the legacy restaged-input protocol";
    harness->pair.reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }

  std::vector<FeatureInputFixture> invalid;
  auto missing = FeatureInputFixture{};
  missing.omit_slot = true;
  invalid.push_back(missing);
  auto duplicate = FeatureInputFixture{};
  duplicate.duplicate_slot = true;
  invalid.push_back(duplicate);
  auto duplicate_other_kind = FeatureInputFixture{};
  duplicate_other_kind.duplicate_with_other_kind = true;
  invalid.push_back(duplicate_other_kind);
  auto wrong_bytes = FeatureInputFixture{};
  wrong_bytes.slot_bytes = 1;
  invalid.push_back(wrong_bytes);
  auto wrong_tensor = FeatureInputFixture{};
  wrong_tensor.wrong_slot_tensor = true;
  invalid.push_back(wrong_tensor);
  auto wrong_slot_kind = FeatureInputFixture{};
  wrong_slot_kind.slot_kind = kernels::InputKind::kTokens;
  invalid.push_back(wrong_slot_kind);
  auto wrong_source_kind = FeatureInputFixture{};
  wrong_source_kind.source_kind = components::InputKind::kStream;
  invalid.push_back(wrong_source_kind);
  auto non_f32 = FeatureInputFixture{};
  const std::array<std::uint64_t, 2> non_f32_dims{4, 1};
  const auto shape = format::ShapeOf(non_f32_dims);
  ASSERT_TRUE(shape);
  non_f32.format = {
      .encoding = format::Encoding::kF16, .layout = format::Layout::kDense, .shape = *shape};
  invalid.push_back(non_f32);

  for (const auto& fixture : invalid) {
    auto harness = MakePairHarness(script, script, false, 0, nullptr, true, false, true, false,
                                   false, false, fixture);
    ASSERT_TRUE(harness);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(described);
    EXPECT_FALSE(*described);
    harness->pair.reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }
}

TEST(ComparisonPairTest, ReadyBoundaryAdmitsOnlyExactI32PositionsSlots) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  ProjectionScript script{
      .expand_first = false, .certificate_supported = true, .remainder_same = true};
  const auto make = [&](PositionInputFixture fixture) {
    return MakePairHarness(script, script, false, 0, nullptr, true, false, true, false, false,
                           false, FeatureInputFixture{}, fixture);
  };
  {
    auto harness = make({});
    ASSERT_TRUE(harness);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(described);
    EXPECT_TRUE(*described);
    auto legacy = DescribeRecipeComparison(*harness->pair, *harness->registry);
    ASSERT_TRUE(legacy);
    EXPECT_FALSE(*legacy) << "Positions stay outside the legacy protocol";
  }
  {
    auto ordinary = FeatureInputFixture{};
    ordinary.source_kind = components::InputKind::kStream;
    ordinary.slot_kind = kernels::InputKind::kStepBytes;
    auto harness = MakePairHarness(script, script, false, 0, nullptr, true, false, true, false,
                                   false, false, ordinary, PositionInputFixture{});
    ASSERT_TRUE(harness);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(described);
    EXPECT_TRUE(*described);
    auto legacy = DescribeRecipeComparison(*harness->pair, *harness->registry);
    ASSERT_TRUE(legacy);
    EXPECT_FALSE(*legacy) << "Positions alone keep an otherwise ordinary Stream outside legacy";
  }
  std::vector<PositionInputFixture> invalid;
  auto missing = PositionInputFixture{};
  missing.omit_slot = true;
  invalid.push_back(missing);
  auto duplicate = PositionInputFixture{};
  duplicate.duplicate_slot = true;
  invalid.push_back(duplicate);
  auto duplicate_kind = PositionInputFixture{};
  duplicate_kind.duplicate_with_other_kind = true;
  invalid.push_back(duplicate_kind);
  auto extent = PositionInputFixture{};
  extent.slot_bytes = 8;
  invalid.push_back(extent);
  auto segment = PositionInputFixture{};
  segment.slot_segment = 0;
  invalid.push_back(segment);
  auto tensor = PositionInputFixture{};
  tensor.wrong_slot_tensor = true;
  invalid.push_back(tensor);
  auto kind = PositionInputFixture{};
  kind.slot_kind = kernels::InputKind::kTokens;
  invalid.push_back(kind);
  auto source = PositionInputFixture{};
  source.source_kind = components::InputKind::kStream;
  invalid.push_back(source);
  auto dtype = PositionInputFixture{};
  dtype.format = F32({1});
  invalid.push_back(dtype);
  auto shape = PositionInputFixture{};
  shape.format = ir::Dense(format::Encoding::kI32, ir::ShapeOf({1, 2}));
  invalid.push_back(shape);
  for (std::size_t i = 0; i < invalid.size(); ++i) {
    SCOPED_TRACE(i);
    auto harness = make(invalid[i]);
    ASSERT_TRUE(harness);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
    ASSERT_TRUE(described);
    EXPECT_FALSE(*described);
  }
}

TEST(ComparisonPairTest, ReadyBoundaryCertificateExpansionJoinsExecutableExpansion) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  auto harness = MakePairHarness({.expansion = ir::NodeId{2},
                                  .expand_first = true,
                                  .certificate_supported = true,
                                  .certificate_expand_first = true,
                                  .certificate_expansion = ir::NodeId{3}},
                                 {.expansion = ir::NodeId{3},
                                  .expand_first = true,
                                  .certificate_supported = true,
                                  .certificate_expand_first = true,
                                  .certificate_expansion = ir::NodeId{2}},
                                 false, 0, nullptr, true, false, true, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  EXPECT_TRUE(harness->incumbent->descriptor_union_seen());
  EXPECT_TRUE(harness->alternative->descriptor_union_seen());
  EXPECT_TRUE(harness->incumbent->certificate_union_seen());
  EXPECT_TRUE(harness->alternative->certificate_union_seen());
  EXPECT_EQ(harness->incumbent->descriptor_order(), (std::array<std::uint32_t, 3>{4, 2, 3}));
  EXPECT_EQ(harness->alternative->descriptor_order(), (std::array<std::uint32_t, 3>{4, 2, 3}));
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  ASSERT_TRUE(harness->pair->BindPrograms(incumbent_launch, alternative_launch));
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_TRUE(closed) << (closed ? "" : closed.error().detail);
  EXPECT_TRUE((*closed)->closed());
  EXPECT_TRUE(harness->incumbent->ready_union_retained_seen());
  EXPECT_TRUE(harness->alternative->ready_union_retained_seen());
  EXPECT_EQ(harness->incumbent->ready_union_node_count(), 3U)
      << "projection calls=" << harness->incumbent->projection_count();
  EXPECT_EQ(harness->alternative->ready_union_node_count(), 3U)
      << "projection calls=" << harness->alternative->projection_count();
  EXPECT_EQ(harness->incumbent->ready_union_order(), (std::array<std::uint32_t, 3>{4, 2, 3}));
  EXPECT_EQ(harness->alternative->ready_union_order(), (std::array<std::uint32_t, 3>{4, 2, 3}));
}

TEST(ComparisonPairTest, ReadyBoundaryRepeatedBoundExpansionIsNoProgress) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  auto harness = MakePairHarness({.expansion = ir::NodeId{2},
                                  .expand_first = true,
                                  .certificate_supported = true,
                                  .certificate_expand_first = true,
                                  .certificate_expansion = ir::NodeId{3},
                                  .force_redundant_expand = true},
                                 {.expansion = ir::NodeId{3},
                                  .expand_first = true,
                                  .certificate_supported = true,
                                  .certificate_expand_first = true,
                                  .certificate_expansion = ir::NodeId{2},
                                  .force_redundant_expand = true},
                                 false, 0, nullptr, true, false, true, true);
  ASSERT_TRUE(harness);
  ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry, ready));
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  ASSERT_TRUE(harness->pair->BindPrograms(incumbent_launch, alternative_launch));
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ComparisonPairFailure::Code::kNoProgress);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, BoundApplicabilityCertificateDriftIsTerminalBeforeClose) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  auto harness = MakePairHarness(
      {.expand_first = false, .certificate_supported = true},
      {.expand_first = false, .certificate_supported = true, .mutate_certificate_on_bind = true},
      false, 0, nullptr, true, false, true, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  auto bound = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
  ASSERT_FALSE(bound);
  EXPECT_EQ(bound.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
  EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  EXPECT_EQ(incumbent_launch.capture_count, 0U);
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ComparisonPairFailure::Code::kNotReady);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, SelectedReadyBoundaryOwnerRevalidatesCertificateBeforePromotion) {
  constexpr auto ready = ComparisonRecipePreparation::kStatelessReadyBoundaryOperands;
  auto harness = MakePairHarness(
      {.expand_first = false, .certificate_supported = true},
      {.expand_first = false, .certificate_supported = true, .mutate_certificate_on_bind = true},
      false, 0, nullptr, true, false, true, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry, ready);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  auto registry = RegistryWithRecipe(**described, 120, 100, true);
  ASSERT_TRUE(registry);
  auto selected = SelectRecipeComparison(std::move(harness->pair), *registry, ready);
  ASSERT_TRUE(selected);
  RecordingStepLaunch launch;
  auto bound = (*selected)->Bind(launch);
  ASSERT_FALSE(bound);
  EXPECT_EQ((*selected)->bound(), nullptr);
  EXPECT_FALSE((*selected)->Run(launch, {}));
  EXPECT_EQ(harness->alternative_observation->run_count, 0U);
  EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  selected->reset();
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, SecondMeasurementBindFailureIsTerminalAndNeverCaptures) {
  auto harness =
      MakePairHarness({.expand_first = false}, {.expand_first = false, .fail_bind = true}, false, 0,
                      nullptr, true, false, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  harness->incumbent->ExpectBindLaunch(&incumbent_launch);
  harness->alternative->ExpectBindLaunch(&alternative_launch);
  auto bound = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
  ASSERT_FALSE(bound);
  EXPECT_EQ(harness->incumbent_observation->bind_count, 1U);
  EXPECT_EQ(harness->alternative_observation->bind_count, 1U);
  EXPECT_TRUE(harness->incumbent->is_bound());
  EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  EXPECT_EQ(incumbent_launch.capture_count, 0U);

  EXPECT_FALSE(DescribeRecipeComparison(*harness->pair, *harness->registry));
  EXPECT_FALSE(harness->pair->BindPrograms(incumbent_launch, alternative_launch));
  EXPECT_FALSE(harness->pair->CaptureRegion(ComparisonPairArm::kIncumbent, incumbent_launch, {}));
  EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
  EXPECT_EQ(incumbent_launch.launch_count, 0U);

  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ComparisonPairFailure::Code::kNotReady);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, WrongLaunchAndStaleIdentityRefuseBeforeBinding) {
  {
    auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                   nullptr, true, false, true);
    ASSERT_TRUE(harness);
    ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
    RecordingStepLaunch expected;
    RecordingStepLaunch wrong;
    RecordingStepLaunch alternative_launch;
    harness->incumbent->ExpectBindLaunch(&expected);
    harness->alternative->ExpectBindLaunch(&alternative_launch);
    auto refused = harness->pair->BindPrograms(wrong, alternative_launch);
    ASSERT_FALSE(refused);
    EXPECT_EQ(harness->incumbent_observation->bind_count, 1U);
    EXPECT_EQ(harness->alternative_observation->bind_count, 0U);
    EXPECT_FALSE(harness->pair->BindPrograms(expected, alternative_launch));
    EXPECT_EQ(harness->incumbent_observation->bind_count, 1U);
    harness->pair.reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }
  {
    auto harness =
        MakePairHarness({.expand_first = false}, {.expand_first = false, .fail_bind = true}, false,
                        0, nullptr, true, false, true);
    ASSERT_TRUE(harness);
    ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
    RecordingStepLaunch incumbent_launch;
    RecordingStepLaunch alternative_launch;
    ASSERT_FALSE(harness->pair->BindPrograms(incumbent_launch, alternative_launch));
    auto selected = SelectRecipeComparison(std::move(harness->pair), *harness->registry);
    EXPECT_FALSE(selected);
    EXPECT_EQ(harness->incumbent_observation->bind_count, 1U);
    EXPECT_EQ(harness->alternative_observation->bind_count, 1U);
    EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
    EXPECT_EQ(harness->alternative_observation->record_count, 0U);
    EXPECT_EQ(harness->funder->active, 0U);
  }
  {
    auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                   nullptr, true, false, true);
    ASSERT_TRUE(harness);
    ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
    harness->alternative->MutatePlannedIdentity();
    RecordingStepLaunch incumbent_launch;
    RecordingStepLaunch alternative_launch;
    auto refused = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kStale);
    EXPECT_EQ(harness->incumbent_observation->bind_count, 0U);
    EXPECT_EQ(harness->alternative_observation->bind_count, 0U);
    harness->pair.reset();
    EXPECT_EQ(harness->funder->active, 0U);
  }
}

TEST(ComparisonPairTest, BoundSealDriftFromSavedDescriptorRefusesBeforeCapture) {
  auto harness = MakePairHarness({.expand_first = false},
                                 {.expand_first = false, .mutate_physical_recipe_on_bind = true},
                                 false, 0, nullptr, true, false, true);
  ASSERT_TRUE(harness);
  ASSERT_TRUE(DescribeRecipeComparison(*harness->pair, *harness->registry));
  RecordingStepLaunch incumbent_launch;
  RecordingStepLaunch alternative_launch;
  auto bound = harness->pair->BindPrograms(incumbent_launch, alternative_launch);
  ASSERT_TRUE(bound);
  auto closed = CloseComparisonPair(std::move(harness->pair));
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, ComparisonPairFailure::Code::kUnproved);
  EXPECT_EQ(harness->incumbent_observation->record_count, 0U);
  EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  EXPECT_EQ(incumbent_launch.capture_count, 0U);
  EXPECT_EQ(alternative_launch.capture_count, 0U);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, DescriptorSelectionRetainsOnlyTheChosenProgramAndPaidBoundary) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, false, true);
  ASSERT_TRUE(harness);
  ASSERT_NE(harness->pair, nullptr);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  auto registry = RegistryWithRecipe(**described, 120, 100);
  ASSERT_TRUE(registry);
  ASSERT_EQ(registry->SelectionDigest(), harness->registry->SelectionDigest());
  harness->recipe_registry.emplace(std::move(*registry));

  const auto before_selection = harness->funder->active;
  const auto expected_identity = harness->alternative->comparison_plan_identity();
  const auto expected_alternative_recipe = harness->alternative->comparison_recipe();
  ASSERT_TRUE(expected_identity);
  ASSERT_TRUE(expected_alternative_recipe);
  auto selected = SelectRecipeComparison(std::move(harness->pair), *harness->recipe_registry);
  ASSERT_TRUE(selected);
  const auto selected_identity = (*selected)->comparison_plan_identity();
  ASSERT_TRUE(expected_identity);
  ASSERT_TRUE(selected_identity);
  EXPECT_EQ(selected_identity->source_graph_identity, expected_identity->source_graph_identity);
  EXPECT_EQ(selected_identity->planned_graph_identity, expected_identity->planned_graph_identity);
  ASSERT_TRUE((*selected)->comparison_recipe());
  EXPECT_EQ((*selected)->comparison_recipe()->planned_graph,
            expected_alternative_recipe->planned_graph);
  EXPECT_EQ((*selected)->bound(), nullptr);
  EXPECT_LT(harness->funder->active, before_selection);
  EXPECT_EQ(harness->incumbent_observation->bind_count, 0U);
  EXPECT_EQ(harness->alternative_observation->bind_count, 0U);
  EXPECT_EQ(harness->incumbent_observation->destruction_count, 1U);
  EXPECT_EQ(harness->alternative_observation->destruction_count, 0U);
  NoopStepLaunch launch;
  auto promoted = (*selected)->Bind(launch);
  ASSERT_TRUE(promoted);
  EXPECT_EQ(harness->incumbent_observation->bind_count, 0U);
  EXPECT_EQ(harness->alternative_observation->bind_count, 1U);
  EXPECT_NE((*selected)->bound(), nullptr);
  const auto reported_bytes = (*selected)->host_bytes();
  (*selected)->ConstructionPaid();
  EXPECT_EQ((*selected)->host_bytes(), reported_bytes);
  selected->reset();
  EXPECT_EQ(harness->alternative_observation->destruction_count, 1U);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, DescriptorBoundMutationNeverPublishesExecutionAuthority) {
  auto harness = MakePairHarness({.expand_first = false},
                                 {.expand_first = false, .mutate_physical_recipe_on_bind = true},
                                 false, 0, nullptr, true, false, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  auto registry = RegistryWithRecipe(**described, 120, 100);
  ASSERT_TRUE(registry);
  auto selected = SelectRecipeComparison(std::move(harness->pair), *registry);
  ASSERT_TRUE(selected);
  ASSERT_NE(selected->get(), nullptr);

  NoopStepLaunch launch;
  auto bind = (*selected)->Bind(launch);
  ASSERT_FALSE(bind);
  EXPECT_EQ((*selected)->bound(), nullptr);
  auto run = (*selected)->Run(launch, {});
  EXPECT_FALSE(run);
  auto record = (*selected)->Record(
      launch, {}, []() -> std::expected<void, kernels::StepFailure> { return {}; });
  EXPECT_FALSE(record);
  EXPECT_EQ(harness->alternative_observation->run_count, 0U);
  EXPECT_EQ(harness->alternative_observation->record_count, 0U);
  EXPECT_EQ(harness->alternative_observation->bind_count, 1U);
  EXPECT_EQ(harness->incumbent_observation->destruction_count, 1U);
  selected->reset();
  EXPECT_EQ(harness->alternative_observation->destruction_count, 1U);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, LocalDescriptorSelectionRefusesQualityOnlyWholeEvidence) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, false, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  auto registry = RegistryWithRecipe(**described, 0, 0, false,
                                     execution::RecipeComparisonKind::kRejectIncumbentQuality);
  ASSERT_TRUE(registry);
  const auto incumbent = harness->incumbent;
  auto selected = SelectRecipeComparison(std::move(harness->pair), *registry);
  ASSERT_TRUE(selected);
  EXPECT_EQ(selected->get(), incumbent);
  EXPECT_EQ(harness->alternative_observation->destruction_count, 1U);
  EXPECT_EQ(harness->incumbent_observation->bind_count, 0U);
}

TEST(ComparisonPairTest, DescriptorNoHitAndNeutralKeepTheExactIncumbent) {
  for (const bool neutral : {false, true}) {
    auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                   nullptr, true, false, true);
    ASSERT_TRUE(harness);
    auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
    ASSERT_TRUE(described);
    ASSERT_TRUE(*described);
    auto key = **described;
    if (!neutral) key.boundary = base::Sha256{}.Update("nonmatching-test-key").Finish();
    const auto incumbent = harness->incumbent;
    auto registry = RegistryWithRecipe(key, neutral ? 100 : 120, 100);
    ASSERT_TRUE(registry);
    auto selected = SelectRecipeComparison(std::move(harness->pair), *registry);
    ASSERT_TRUE(selected);
    EXPECT_EQ(selected->get(), incumbent);
    EXPECT_EQ(harness->incumbent_observation->destruction_count, 0U);
    EXPECT_EQ(harness->alternative_observation->destruction_count, 1U);
    EXPECT_EQ(harness->incumbent_observation->bind_count, 0U);
    selected->reset();
    EXPECT_EQ(harness->incumbent_observation->destruction_count, 1U);
    EXPECT_EQ(harness->funder->active, 0U);
  }
}

TEST(ComparisonPairTest, DescriptorUnprovedFallsBackButIdentityMutationIsStale) {
  auto unproved = MakePairHarness({.expand_first = false, .unproved_first = true},
                                  {.expand_first = false}, false, 0, nullptr, true, false, true);
  ASSERT_TRUE(unproved);
  auto ordinary = SelectRecipeComparison(std::move(unproved->pair), *unproved->registry);
  ASSERT_TRUE(ordinary);
  EXPECT_EQ(ordinary->get(), unproved->incumbent);
  ordinary->reset();
  EXPECT_EQ(unproved->funder->active, 0U);

  auto stale = MakePairHarness(
      {.expand_first = false, .unproved_first = true, .mutate_identity_on_unproved = true},
      {.expand_first = false}, false, 0, nullptr, true, false, true);
  ASSERT_TRUE(stale);
  auto refused = SelectRecipeComparison(std::move(stale->pair), *stale->registry);
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kStale);
  EXPECT_EQ(stale->funder->active, 0U);
}

TEST(ComparisonPairTest, DescriptorWinnerOwnerFundingRefusalPublishesNothing) {
  auto harness = MakePairHarness({.expand_first = false}, {.expand_first = false}, false, 0,
                                 nullptr, true, false, true);
  ASSERT_TRUE(harness);
  auto described = DescribeRecipeComparison(*harness->pair, *harness->registry);
  ASSERT_TRUE(described);
  ASSERT_TRUE(*described);
  auto registry = RegistryWithRecipe(**described, 120, 100);
  ASSERT_TRUE(registry);
  harness->recipe_registry.emplace(std::move(*registry));
  const auto selection_digest = harness->registry->SelectionDigest();
  ASSERT_EQ(harness->recipe_registry->SelectionDigest(), selection_digest);

  harness->funder->deny = true;
  auto selected = SelectRecipeComparison(std::move(harness->pair), *harness->recipe_registry);
  ASSERT_FALSE(selected);
  EXPECT_EQ(selected.error().code, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(harness->funder->active, 0U);
}

TEST(ComparisonPairTest, FailedCandidateBoundaryPlanCannotBeRetriedWithAStaleSeed) {
  auto baseline = MakePairHarness({.expand_first = false}, {.expand_first = false});
  ASSERT_TRUE(baseline);
  const auto plan_charge = baseline->charges_after_plan;
  baseline.reset();

  auto refused =
      MakePairHarness({.expand_first = false}, {.expand_first = false}, false, plan_charge);
  ASSERT_TRUE(refused);
  ASSERT_NE(refused->pair, nullptr);
  ASSERT_TRUE(refused->plan_failure.has_value());
  EXPECT_EQ(*refused->plan_failure, ComparisonPairFailure::Code::kAdmission);
  EXPECT_EQ(refused->funder->charges, plan_charge);
  const auto source_identity = refused->pair->source().graph.identity();
  planner::ComparisonSiteSelector selector;
  selector.source_graph_identity = source_identity;
  selector.kind = planner::ComparisonSiteKind::kFusedNode;
  selector.member_origins = {{ir::NodeId{3}}};
  const planner::DeviceFacts device{.device_class = "test", .arch = 121};
  auto retry = refused->pair->Plan(selector, *refused->registry, device, {});
  ASSERT_FALSE(retry);
  EXPECT_EQ(retry.error().code, ComparisonPairFailure::Code::kInvalid);
  refused->pair.reset();
  EXPECT_EQ(refused->funder->active, 0U);
}

TEST(ComparisonPairTest, BoundPortsRejectOverflowAndContradictoryAccess) {
  for (const auto malformed :
       {ProjectionScript{.expansion = ir::NodeId{2}, .overflow_interval = true},
        ProjectionScript{.expansion = ir::NodeId{2}, .wrong_access = true}}) {
    auto harness = MakePairHarness(malformed, malformed);
    ASSERT_TRUE(harness);
    auto refused = CloseComparisonPair(std::move(harness->pair));
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().code, ComparisonPairFailure::Code::kUnproved);
    EXPECT_EQ(harness->funder->active, 0U);
  }
}

}  // namespace
}  // namespace llmp::engine
