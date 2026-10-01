// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "benchmarks/ds4_complete/binding.h"

namespace {
namespace complete = jitllm::benchmarks::ds4_complete;
namespace kg = jitllm::kernels::ggml;
namespace model = jitllm::model;

std::uint64_t Bytes(const complete::ScratchPlan& plan, std::string_view name) {
  const auto found = std::ranges::find(plan.ranges, name, &complete::ScratchRequirement::name);
  return found == plan.ranges.end() ? 0 : found->bytes;
}

TEST(Ds4CompleteBinding, PaysIndependentSourcesAndBothNativeWorkspaces) {
  const auto plan = complete::PlanScratch(model::Dsv4Flash(), 8192, 48);
  ASSERT_TRUE(plan.has_value());
  std::set<std::string> names;
  std::uint64_t bytes = 0;
  for (const auto& range : plan->ranges) {
    EXPECT_TRUE(names.insert(range.name).second);
    EXPECT_GT(range.bytes, 0U);
    EXPECT_EQ(range.alignment, 256U);
    bytes += range.bytes;
  }
  EXPECT_EQ(bytes, plan->logical_bytes);
  EXPECT_EQ(Bytes(*plan, "workspace.cublas"), (std::uint64_t{32} << 20U));
  EXPECT_EQ(Bytes(*plan, "workspace.launch"), plan->launch_workspace_bytes);
  EXPECT_EQ(Bytes(*plan, "query.heads"), std::uint64_t{4096} * 32768 * 4);
  EXPECT_EQ(Bytes(*plan, "attention.heads"), std::uint64_t{4096} * 32768 * 4);
  EXPECT_EQ(Bytes(*plan, "compression.refresh.kv"), std::uint64_t{4} * 1024 * 4);
  EXPECT_EQ(Bytes(*plan, "compression.refresh.score"), std::uint64_t{4} * 1024 * 4);
  EXPECT_EQ(Bytes(*plan, "indexer.refresh.kv"), std::uint64_t{4} * 256 * 4);
  EXPECT_EQ(Bytes(*plan, "indexer.refresh.score"), std::uint64_t{4} * 256 * 4);
  for (const auto& [name, width] :
       std::array{std::pair{"attention.norm.d4", 4096U}, std::pair{"query.rank.d4", 1024U},
                  std::pair{"attention.low.d4", 8192U}, std::pair{"ffn.norm.d4", 4096U},
                  std::pair{"shared.middle.d4", 2048U}}) {
    const auto needed = kg::Ds4D4Bytes(4096, width);
    ASSERT_TRUE(needed.has_value());
    EXPECT_GE(Bytes(*plan, name), *needed);
  }
  const auto routed = kg::Ds4MoeLayoutOf(
      {.rows = 4096, .input = 4096, .middle = 2048, .output = 4096}, kg::Ds4MoeTier::kDirect);
  ASSERT_TRUE(routed.has_value());
  EXPECT_GE(Bytes(*plan, "routed.input.quant"), routed->input_quant_bytes);
  EXPECT_GE(Bytes(*plan, "routed.down.quant"), routed->down_quant_bytes);
  EXPECT_GE(Bytes(*plan, "routed.down"), routed->down_bytes);
}

TEST(Ds4CompleteBinding, PaysFullChronologicalUnionAndPackedIndexerPreparation) {
  const auto plan = complete::PlanScratch(model::Dsv4Flash(), 8192, 48);
  ASSERT_TRUE(plan.has_value());
  kg::Ds4Attention attention{};
  attention.tokens = 4096;
  attention.first = 4096;
  attention.raw_cells = 4352;
  attention.raw_count = 4224;
  attention.raw_start = 3968;
  attention.compressed_cells = 2048;
  attention.compressed_count = 2048;
  attention.consecutive_first = 4096;
  attention.domain = kg::Ds4AttentionDomain::kMixedRing;
  const auto scratch = kg::PlanDs4AttentionScratch(attention, kg::Ds4AttentionKind::kTokenTile, 48);
  ASSERT_TRUE(scratch.has_value());
  EXPECT_GE(Bytes(*plan, "attention.scratch"), scratch->bytes);
  EXPECT_EQ(Bytes(*plan, "attention.proxy"), std::uint64_t{2048} * 512 * 4);
  EXPECT_EQ(Bytes(*plan, "indexer.proxy"), std::uint64_t{2048} * 128 * 4);
  EXPECT_EQ(Bytes(*plan, "indexer.score.scratch"), std::uint64_t{4096} * 64 * 4);
  EXPECT_EQ(Bytes(*plan, "indexer.selected"), std::uint64_t{4096} * 512 * 4);
  EXPECT_EQ(Bytes(*plan, "head.logits"), std::uint64_t{129280} * 4);
}

TEST(Ds4CompleteBinding, RoutedStudyPaysMaterializedStorageAndSeparateOriginalInputControl) {
  const auto plain = complete::PlanScratch(model::Dsv4Flash(), 8192, 48);
  const auto study = complete::PlanScratch(model::Dsv4Flash(), 8192, 48, false, true);
  ASSERT_TRUE(plain);
  ASSERT_TRUE(study);
  std::uint64_t intermediate_bytes = 0;
  for (const auto* name :
       {"routed.materialized.gate", "routed.materialized.up", "routed.materialized.middle"}) {
    EXPECT_EQ(Bytes(*plain, name), 0U);
    EXPECT_EQ(Bytes(*study, name), 24576ULL * 2048 * 4);
    intermediate_bytes += Bytes(*study, name);
  }
  EXPECT_EQ(intermediate_bytes, 603979776U);
  const auto materialized = kg::Ds4MoeLayoutOf(
      {.rows = 4096, .input = 4096, .middle = 2048, .output = 4096}, kg::Ds4MoeTier::kMaterialized);
  ASSERT_TRUE(materialized);
  EXPECT_GE(Bytes(*study, "routed.input.quant"), materialized->input_quant_bytes);
  EXPECT_GE(Bytes(*study, "routed.down.quant"), materialized->down_quant_bytes);
  EXPECT_GE(Bytes(*study, "routed.work"), materialized->work_bytes);
  EXPECT_EQ(Bytes(*study, "routed.control.input.quant"), Bytes(*study, "routed.input.quant"));
  EXPECT_EQ(Bytes(*study, "routed.control.down.quant"), Bytes(*study, "routed.down.quant"));
  EXPECT_EQ(Bytes(*study, "routed.control.down"), Bytes(*study, "routed.down"));
  EXPECT_EQ(Bytes(*study, "routed.control.bounds"), 257ULL * 4);
  EXPECT_FALSE(complete::PlanScratch(model::Dsv4Flash(), 8192, 48, true, true));
}

TEST(Ds4CompleteBinding, RefusesUnqualifiedProfilesContextsAndIncompleteBinding) {
  const auto& profile = model::Dsv4Flash();
  EXPECT_FALSE(complete::PlanScratch(profile, 0, 48));
  EXPECT_FALSE(complete::PlanScratch(profile, 4096, 48));
  EXPECT_FALSE(complete::PlanScratch(profile, 16384, 48));
  EXPECT_FALSE(complete::PlanScratch(profile, 8192, 0));
  EXPECT_FALSE(complete::PlanScratch(profile, 8192, 65536));
  auto other = profile;
  other.rope_scale = 1;
  EXPECT_FALSE(complete::PlanScratch(other, 8192, 48));
  other = profile;
  other.compress_ratios[2] = 128;
  EXPECT_FALSE(complete::PlanScratch(other, 8192, 48));
  other = profile;
  other.vocab = 129279;
  EXPECT_FALSE(complete::PlanScratch(other, 8192, 48));
  EXPECT_FALSE(complete::BindChunk(profile, {}));
}

}  // namespace
