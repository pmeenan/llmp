// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/expert_slab.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>

#include "artifact_writer.h"
#include "format/tensor_format.h"

namespace {
namespace en = llmp::engine;
namespace fm = llmp::format;
namespace ar = llmp::artifact;
namespace ts = llmp::test;

class NativeExpertSlabTest : public testing::TestWithParam<bool> {
 protected:
  std::filesystem::path scratch;
  void SetUp() override {
    const char* root = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
    scratch = std::filesystem::path(root ? root : testing::TempDir()) / "native-expert-slab" /
              testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::create_directories(scratch);
  }
  void TearDown() override { std::filesystem::remove_all(scratch); }
  std::expected<ar::Artifact, ar::Error> Open() {
    const fm::TensorFormat gate{.encoding = fm::Encoding::kQ4_K,
                                .layout = fm::Layout::kGgmlBlocks,
                                .shape = *fm::ShapeOf(std::array<std::uint64_t, 2>{2816, 8})};
    const fm::TensorFormat down{.encoding = GetParam() ? fm::Encoding::kQ8_0 : fm::Encoding::kQ5_1,
                                .layout = fm::Layout::kGgmlBlocks,
                                .shape = *fm::ShapeOf(std::array<std::uint64_t, 2>{704, 8})};
    ts::V1Plan plan{
        .expert_count = 4,
        .groups = {{.kind = ar::GroupKind::kExpert,
                    .layer = 0,
                    .arrays = {{.name = "blk.0.ffn_gate_up_exps.weight", .format = gate},
                               {.name = "blk.0.ffn_down_exps.weight", .format = down}}}}};
    return ar::Artifact::Open(ts::Publish(scratch, ts::Render(plan)));
  }
};
TEST_P(NativeExpertSlabTest, NativeMixedBlocksHaveExactAlignedNonoverlappingExpertPitch) {
  auto artifact = Open();
  ASSERT_TRUE(artifact) << (artifact ? "" : artifact.error().ToString());
  for (const auto& array : artifact->expert_arrays()) EXPECT_TRUE(array.repr.type.empty());
  const auto slabs = en::ExpertSlabsOf(*artifact, 256);
  ASSERT_TRUE(slabs) << (slabs ? "" : slabs.error());
  ASSERT_EQ(slabs->slabs.size(), 1U);
  ASSERT_EQ(slabs->pitch.size(), 2U);
  const auto& slab = slabs->slabs.front();
  EXPECT_EQ(slab.count, 4U);
  EXPECT_EQ(slab.alignment, 256U);
  EXPECT_EQ(slab.stride % 256, 0U);
  EXPECT_EQ(slab.stride % 144, 0U);
  EXPECT_EQ(slab.stride % (GetParam() ? 34U : 24U), 0U);
  // Existing scalar slab rule supplied with the actual source encodings,
  // without changing the v1 artifact's absent legacy representations.
  std::array<std::pair<std::uint32_t, std::string_view>, 2> named{};
  for (std::uint32_t i = 0; i < 2; ++i) {
    const auto& a = artifact->expert_arrays()[i];
    named[i] = {i, a.name.contains("gate_up") ? "Q4_K" : GetParam() ? "Q8_0" : "Q5_1"};
    EXPECT_EQ(slabs->pitch[i], slab.stride);
    for (std::uint32_t e = 0; e < slab.count; ++e) {
      const auto group = a.first_group + e;
      EXPECT_EQ(slabs->places[group], en::GroupPlace::kNone);
      EXPECT_LE(a.group_offset.value() + a.readable.value(), slab.stride);
      EXPECT_LE(artifact->groups()[group].stored.value(), slab.stride);
    }
  }
  const auto old = en::ExpertSlab(*artifact, named, 4, 256, 0);
  ASSERT_TRUE(old);
  const std::uint64_t quantum = GetParam() ? 39168U : 2304U;  // lcm(256,144,34 or 24)
  EXPECT_EQ(slab.stride, (old->stride + quantum - 1) / quantum * quantum);
}
TEST_P(NativeExpertSlabTest, RefusesZeroAndOverflowingAlignment) {
  auto artifact = Open();
  ASSERT_TRUE(artifact);
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, 0));
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, std::numeric_limits<std::uint64_t>::max()));
}
INSTANTIATE_TEST_SUITE_P(Gemma26Blocks, NativeExpertSlabTest, testing::Bool());
}  // namespace
