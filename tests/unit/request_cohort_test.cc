// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The request cohort (engine/request_cohort.h), host-only: the active set's
// masks and their refusals, the default request, the closures over shared
// extents and slots that hold no state yet, and a fault quarantining every
// slot until retirement.

#include "engine/request_cohort.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "engine/live_state.h"

namespace {

using jitllm::base::operator""_MiB;
using jitllm::catalog::Catalog;
using jitllm::catalog::ExtentDescriptor;
using jitllm::catalog::ExtentId;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Recovery;
using jitllm::engine::LiveState;
using jitllm::engine::RequestCohort;

TEST(RequestCohortTest, TheDefaultRequestIsSlotZeroAndMasksAreDistinctSlots) {
  RequestCohort cohort("DeepSeek");
  EXPECT_EQ(cohort.active(), 1U);
  EXPECT_TRUE(cohort.IsActive(0));
  EXPECT_FALSE(cohort.IsActive(1));
  EXPECT_FALSE(cohort.IsActive(4));
  const std::array<std::uint32_t, 3> three = {0, 2, 3};
  auto mask = cohort.MaskOf(three);
  ASSERT_TRUE(mask.has_value());
  EXPECT_EQ(*mask, 0b1101U);
  cohort.Select(*mask);
  EXPECT_TRUE(cohort.IsActive(2));
  EXPECT_FALSE(cohort.IsActive(1));
  // A slot twice, a slot past the runner's, more slots than it has.
  EXPECT_FALSE(cohort.MaskOf(std::array<std::uint32_t, 2>{1, 1}).has_value());
  EXPECT_FALSE(cohort.MaskOf(std::array<std::uint32_t, 1>{4}).has_value());
  EXPECT_FALSE(cohort.MaskOf(std::array<std::uint32_t, 5>{0, 1, 2, 3, 0}).has_value());
  // None: the shared extents alone (a drained retirement).
  auto none = cohort.MaskOf({});
  ASSERT_TRUE(none.has_value());
  EXPECT_EQ(*none, 0U);
}

TEST(RequestCohortTest, ClosuresHoldTheSharedExtentsAndOnlyProtectedSlots) {
  Catalog catalog;
  const auto domain = catalog.AddDomain("spark");
  std::vector<ExtentId> shared;
  for (std::uint32_t i = 0; i < 3; ++i) {
    ExtentDescriptor d{.domain = domain,
                       .memory_class = MemoryClass::kWeights,
                       .recovery = Recovery::kFromArtifact,
                       .size = 2_MiB,
                       .content = {}};
    d.content.chunk = i;
    auto extent = catalog.AddExtent(d);
    ASSERT_TRUE(extent.has_value());
    shared.push_back(*extent);
  }
  // Slots with no state yet, and one not provisioned.
  LiveState a("DeepSeek");
  LiveState b("DeepSeek");
  const std::array<const LiveState*, RequestCohort::kSlots> states = {&a, &b, nullptr, nullptr};
  RequestCohort cohort("DeepSeek");
  auto closures = cohort.Build(catalog, shared, states, 0b11);
  ASSERT_TRUE(closures.has_value());
  EXPECT_EQ(closures->everything.extents.size(), shared.size());
  EXPECT_EQ(closures->execution.extents.size(), shared.size());
  EXPECT_TRUE(closures->fence.extents.empty());
  EXPECT_TRUE(closures->slot_fences[0].extents.empty());
  auto alone = cohort.Build(catalog, {}, states, 0);
  ASSERT_TRUE(alone.has_value());
  EXPECT_TRUE(alone->execution.extents.empty());
}

TEST(RequestCohortTest, AFaultQuarantinesEverySlotUntilRetirement) {
  LiveState a("DeepSeek");
  LiveState b("DeepSeek");
  std::array<LiveState*, RequestCohort::kSlots> states = {&a, &b, nullptr, nullptr};
  RequestCohort cohort("DeepSeek");
  EXPECT_FALSE(cohort.faulted());
  EXPECT_TRUE(a.Usable().has_value());
  cohort.Fault(states);
  EXPECT_TRUE(cohort.faulted());
  EXPECT_TRUE(a.quarantined());
  EXPECT_TRUE(b.quarantined());
  EXPECT_FALSE(a.Usable().has_value());
}

}  // namespace
