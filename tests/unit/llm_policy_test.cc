// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/llm_policy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "runtime/component_domain.h"

namespace {
namespace rt = llmp::runtime;

rt::LlmSourceCapabilities Source() {
  return {.batching = {.max_segments = 16, .join_prefill = true, .mixed_heads = true},
          .context = 8192,
          .max_rows = 128,
          .indexed_masks = true};
}
constexpr llmp::kernels::BackendCapabilities kBackend{
    .owner_lanes = 4, .device_tokens = true, .indexed_masks = true};
rt::LlmPolicyRequest Request() {
  return {.owners = 4,
          .read_align = 256,
          .wave_lanes = true,
          .lane_scratch_bytes = 4096,
          .lookahead = 3};
}

// The read buckets a request names are the policy's (the slots' and the
// composer's, runtime/component_build.cc); none by default.
TEST(LlmServingPolicyTest, CarriesTheRequestedReadBuckets) {
  auto request = Request();
  const auto plain = rt::LlmServingPolicy::Resolve(Source(), kBackend, request);
  ASSERT_TRUE(plain) << plain.error();
  EXPECT_EQ(plain->read_buckets(), 0U);
  request.read_buckets = 4;
  const auto bucketed = rt::LlmServingPolicy::Resolve(Source(), kBackend, request);
  ASSERT_TRUE(bucketed) << bucketed.error();
  EXPECT_EQ(bucketed->read_buckets(), 4U);
  EXPECT_EQ(bucketed->read_align(), 256U);
}

TEST(LlmServingPolicyTest, ActualCapabilitiesAndPreferencesDetermineOneOwnedValue) {
  auto source = Source();
  auto request = Request();
  const auto policy = rt::LlmServingPolicy::Resolve(source, kBackend, request);
  ASSERT_TRUE(policy) << policy.error();
  source.batching.max_segments = 1;
  request.owners = 1;
  EXPECT_EQ(policy->owners(), 4U);
  EXPECT_EQ(policy->batching().max_segments, 4U);
  EXPECT_EQ(policy->batching().max_segment_rows, 128U);
  EXPECT_TRUE(policy->batching().join_prefill);
  EXPECT_EQ(policy->lanes().count, 4U);
  EXPECT_EQ(policy->lanes().scratch_bytes_per_lane, 4096U);
  EXPECT_TRUE(policy->device_tokens());
  EXPECT_TRUE(policy->indexed_masks());
  EXPECT_EQ(policy->lookahead(), 3U);
}

TEST(LlmServingPolicyTest, DisablingLanesRemovesTheirActualWorkspace) {
  auto request = Request();
  request.wave_lanes = false;
  const auto policy = rt::LlmServingPolicy::Resolve(Source(), kBackend, request);
  ASSERT_TRUE(policy) << policy.error();
  EXPECT_EQ(policy->lanes().count, 0U);
  EXPECT_EQ(policy->lanes().scratch_bytes_per_lane, 0U);
}

TEST(LlmServingPolicyTest, ASettingCannotSupplyAbsentBackendSupport) {
  auto backend = kBackend;
  backend.owner_lanes = 0;
  EXPECT_FALSE(rt::LlmServingPolicy::Resolve(Source(), backend, Request()));
  auto request = Request();
  request.wave_lanes = false;
  backend.device_tokens = false;
  backend.indexed_masks = false;
  const auto policy = rt::LlmServingPolicy::Resolve(Source(), backend, request);
  ASSERT_TRUE(policy) << policy.error();
  EXPECT_FALSE(policy->device_tokens());
  EXPECT_FALSE(policy->indexed_masks());
}

TEST(LlmServingPolicyTest, ActualComponentAndAdmittedCompositionBoundsStayEffective) {
  auto source = Source();
  source.batching.max_segments = 2;
  source.batching.max_segment_rows = 64;
  source.indexed_masks = false;
  auto request = Request();
  request.owners_alone = true;
  request.prompts_alone = true;
  const auto policy = rt::LlmServingPolicy::Resolve(source, kBackend, request);
  ASSERT_TRUE(policy) << policy.error();
  EXPECT_EQ(policy->owners(), 4U);  // Four conversations; one physical owner per step.
  EXPECT_EQ(policy->batching().max_segments, 1U);
  EXPECT_EQ(policy->batching().max_segment_rows, 64U);
  EXPECT_FALSE(policy->batching().join_prefill);
  EXPECT_FALSE(policy->indexed_masks());
  EXPECT_EQ(policy->lanes().count, 0U);
}

TEST(LlmServingPolicyTest, MalformedPhysicalDeclarationsFailBeforeAssembly) {
  auto source = Source();
  source.max_rows = source.context + 1;
  EXPECT_FALSE(rt::LlmServingPolicy::Resolve(source, kBackend, Request()));
  auto request = Request();
  request.read_align = 257;
  EXPECT_FALSE(rt::LlmServingPolicy::Resolve(Source(), kBackend, request));
  request = Request();
  request.lane_scratch_bytes = UINT64_MAX;
  EXPECT_FALSE(rt::LlmServingPolicy::Resolve(Source(), kBackend, request));
  auto backend = kBackend;
  backend.owner_lanes = 5;
  EXPECT_FALSE(rt::LlmServingPolicy::Resolve(Source(), backend, Request()));
}

TEST(LlmServingPolicyTest, AnExplicitDomainCannotManufacturePhysicalStageOrOwnerCapacity) {
  namespace en = llmp::engine;
  namespace md = llmp::model;
  namespace pl = llmp::planner;
  auto domain =
      en::NumericalDomain::Create({{.stage = 0,
                                    .name = "target",
                                    .uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode},
                                    .phases = {{.phase = pl::Phase::kPrefill,
                                                .max_owners = 4,
                                                .max_owner_rows = 64,
                                                .max_total_rows = 256}}}});
  ASSERT_TRUE(domain) << domain.error();
  std::array<md::StateLayoutDerived, 1> layouts{};
  layouts[0].context = 8192;
  layouts[0].max_rows = 128;
  std::array batching{Source().batching};
  EXPECT_TRUE(rt::ValidateServingEnvelope(*domain, layouts, batching, 4));
  EXPECT_FALSE(rt::ValidateServingEnvelope(*domain, layouts, batching, 2));
  batching[0].max_segments = 2;
  EXPECT_FALSE(rt::ValidateServingEnvelope(*domain, layouts, batching, 4));
  batching[0].max_segments = 16;
  batching[0].max_step_rows = 128;
  EXPECT_FALSE(rt::ValidateServingEnvelope(*domain, layouts, batching, 4));
  batching[0].max_step_rows = 0;
  layouts[0].max_rows = 32;
  EXPECT_FALSE(rt::ValidateServingEnvelope(*domain, layouts, batching, 4));
  EXPECT_FALSE(rt::ValidateServingEnvelope(*domain, {}, batching, 4));
}

TEST(ComponentSetupTest, AutomaticMeasurementsFollowTheActualScheduleAndComponentRowBound) {
  namespace md = llmp::model;
  namespace pl = llmp::planner;
  std::array<md::StateLayoutDerived, 1> layouts{};
  layouts[0].context = 8192;
  layouts[0].max_rows = 128;
  std::array batching{Source().batching};
  batching[0].max_step_rows = 32;
  const auto measured = rt::ComponentSetupShapes(layouts, batching, 4, 256, {});
  ASSERT_TRUE(measured) << measured.error();
  EXPECT_EQ(measured->size(), 5U);  // One prompt shape; lone/joined decode in both modes.
  for (const auto& shape : *measured) {
    EXPECT_LE(shape.rows(), 32U);
    EXPECT_NE(shape.phase, pl::Phase::kScore);  // Legal Score != eagerly paid batch output.
    for (const auto& segment : shape.segments) {
      EXPECT_EQ(segment.read_cells, 8192U);
      EXPECT_EQ(segment.past + segment.rows, 8192U);
      EXPECT_EQ(segment.outputs, 1U);
    }
  }
}

TEST(ComponentSetupTest, ExactDiagnosticShapesAreOwnedAndNeverManufactureWiderSupport) {
  namespace en = llmp::engine;
  namespace md = llmp::model;
  namespace pl = llmp::planner;
  auto domain = en::NumericalDomain::Create({{.stage = 0,
                                              .name = "target",
                                              .uses = {en::NumericalUse::kScore},
                                              .phases = {{.phase = pl::Phase::kPrefill,
                                                          .max_owners = 4,
                                                          .max_owner_rows = 2,
                                                          .max_total_rows = 8},
                                                         {.phase = pl::Phase::kDecode,
                                                          .max_owners = 4,
                                                          .max_owner_rows = 1,
                                                          .max_total_rows = 4},
                                                         {.phase = pl::Phase::kScore,
                                                          .max_owners = 4,
                                                          .max_owner_rows = 2,
                                                          .max_total_rows = 8}}}});
  ASSERT_TRUE(domain) << domain.error();
  std::array<md::StateLayoutDerived, 1> layouts{};
  layouts[0].context = 512;
  layouts[0].max_rows = 2;
  std::array batching{Source().batching};
  pl::StepShape score{.stage = 0,
                      .phase = pl::Phase::kScore,
                      .segments = {},
                      .row_invariant = false,
                      .read_align = 256};
  for (std::uint8_t i = 0; i < 4; ++i)
    score.segments.push_back({.slot = i, .rows = 2, .past = 254, .read_cells = 256, .outputs = 2});
  std::array supplied{score};
  auto shapes = rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied);
  ASSERT_TRUE(shapes) << shapes.error();
  const auto found = std::ranges::find(*shapes, score);
  ASSERT_NE(found, shapes->end());
  EXPECT_EQ(found->segments.front().past, 254U);
  supplied.front().segments.front().read_cells = 512;
  EXPECT_EQ(found->segments.front().read_cells, 256U);
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, nullptr, supplied));
  supplied.front() = score;
  supplied.front().segments.front().rows = 3;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied));
  supplied.front() = score;
  supplied.front().segments.front().slot = 4;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied));
  supplied.front() = score;
  supplied.front().segments.front().past = 511;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied));
  supplied.front() = score;
  supplied.front().segments.front().outputs = 3;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied));
  supplied.front() = score;
  batching[0].group_segments = 2;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied));
  batching[0].group_segments = 0;
  supplied.front() = score;
  supplied.front().segments.front().greedy = true;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, {}, &*domain, supplied, false));
}

TEST(ComponentSetupTest, TheCompleteDrafterFrontierIsMeasuredInBothOutputModes) {
  namespace en = llmp::engine;
  namespace md = llmp::model;
  namespace pl = llmp::planner;
  std::array<md::StateLayoutDerived, 2> layouts{};
  for (auto& state : layouts) {
    state.context = 512;
    state.max_rows = 8;
  }
  std::array batching{Source().batching, Source().batching};
  en::DecodingMode mode{.kind = en::DecodingMode::Kind::kSpeculative,
                        .drafters = {{.stage = 1,
                                      .shape = en::DrafterSpec::Shape::kBlock,
                                      .max_depth = 3,
                                      .device_tokens = true}}};
  const auto shapes = rt::ComponentSetupShapes(layouts, batching, 4, 256, mode);
  ASSERT_TRUE(shapes) << shapes.error();
  for (const bool greedy : {false, true}) {
    EXPECT_TRUE(std::ranges::any_of(*shapes, [greedy](const auto& shape) {
      return shape.stage == 0 && shape.phase == pl::Phase::kVerify && shape.segments.size() == 4 &&
             shape.segments.front().rows == 4 && shape.segments.front().outputs == 4 &&
             shape.segments.front().greedy == greedy;
    }));
    EXPECT_TRUE(std::ranges::any_of(*shapes, [greedy](const auto& shape) {
      return shape.stage == 1 && shape.phase == pl::Phase::kDraft && shape.segments.size() == 4 &&
             shape.segments.front().rows == 3 && shape.segments.front().outputs == 3 &&
             shape.segments.front().greedy == greedy;
    }));
  }
  layouts[0].max_rows = 3;
  EXPECT_FALSE(rt::ComponentSetupShapes(layouts, batching, 4, 256, mode));
}

}  // namespace
