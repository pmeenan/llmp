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
#include <utility>
#include <vector>

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
  for (std::uint32_t i = 0; i < 2; ++i) {
    const auto& a = artifact->expert_arrays()[i];
    EXPECT_EQ(slabs->pitch[i], slab.stride);
    for (std::uint32_t e = 0; e < slab.count; ++e) {
      const auto group = a.first_group + e;
      EXPECT_EQ(slabs->places[group], en::GroupPlace::kNone);
      EXPECT_LE(a.group_offset.value() + a.readable.value(), slab.stride);
      EXPECT_LE(artifact->groups()[group].stored.value(), slab.stride);
    }
  }
  // The group's stored bytes rounded up once, to the alignment and every
  // array's block bytes together.
  const std::uint64_t quantum = GetParam() ? 39168U : 2304U;  // lcm(256,144,34 or 24)
  const auto stored = artifact->groups()[slab.first_group].stored.value();
  EXPECT_EQ(slab.stride, (stored + quantum - 1) / quantum * quantum);
}
TEST_P(NativeExpertSlabTest, RefusesAnAlignmentThatIsNotAPositiveMultipleOf16) {
  auto artifact = Open();
  ASSERT_TRUE(artifact);
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, 0));
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, 8));
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, 24));
  EXPECT_TRUE(en::ExpertSlabsOf(*artifact, 16));
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, std::numeric_limits<std::uint64_t>::max()));
  // A multiple of 16 whose rounding overflows.
  EXPECT_FALSE(en::ExpertSlabsOf(*artifact, std::uint64_t{1} << 63U));
}
INSTANTIATE_TEST_SUITE_P(Gemma26Blocks, NativeExpertSlabTest, testing::Bool());

// A split encoding's parts (the native ModelOpt NVFP4 experts: codes, block
// scales, tensor scale) share their experts' groups: one slab, 16-byte
// expert bases at the requested alignment.
TEST(NativeSplitExpertSlabTest, SplitPartsShareOneAlignedSlab) {
  const char* root = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const auto scratch =
      std::filesystem::path(root ? root : testing::TempDir()) / "native-split-expert-slab";
  std::filesystem::remove_all(scratch);
  std::filesystem::create_directories(scratch);
  const fm::TensorFormat nv{.encoding = fm::Encoding::kNvfp4,
                            .params = {.tensor_scale = true},
                            .layout = fm::Layout::kSplit,
                            .shape = *fm::ShapeOf(std::array<std::uint64_t, 2>{256, 64})};
  const auto open = [&](std::string_view where) {
    ts::V1Plan plan{.expert_count = 3,
                    .groups = {{.kind = ar::GroupKind::kExpert,
                                .layer = 0,
                                .arrays = {{.name = "blk.0.ffn_down_exps.weight",
                                            .format = nv,
                                            .part = fm::PartRole::kCodes},
                                           {.name = "blk.0.ffn_down_exps.weight_scale",
                                            .format = nv,
                                            .part = fm::PartRole::kBlockScales},
                                           {.name = "blk.0.ffn_down_exps.weight_scale_2",
                                            .format = nv,
                                            .part = fm::PartRole::kTensorScale}}}}};
    return ar::Artifact::Open(ts::Publish(scratch / where, ts::Render(plan)));
  };
  auto artifact = open("split");
  ASSERT_TRUE(artifact) << (artifact ? "" : artifact.error().ToString());
  const auto slabs = en::ExpertSlabsOf(*artifact, 256);
  ASSERT_TRUE(slabs) << (slabs ? "" : slabs.error());
  ASSERT_EQ(slabs->slabs.size(), 1U);
  const auto& slab = slabs->slabs.front();
  EXPECT_EQ(slab.count, 3U);
  EXPECT_EQ(slab.stride % 256, 0U);
  for (std::uint32_t i = 0; i < 3; ++i) {
    const auto& a = artifact->expert_arrays()[i];
    EXPECT_EQ(slabs->pitch[i], slab.stride);
    EXPECT_LE(a.group_offset.value() + a.readable.value(), slab.stride);
    for (std::uint32_t e = 0; e < slab.count; ++e)
      EXPECT_EQ(slabs->places[a.first_group + e], en::GroupPlace::kNone);
  }
  std::filesystem::remove_all(scratch);
}

// MLX's affine experts (codes, scales and biases or zero points) and the
// importer's scale-atom NVFP4 tables take the same part-by-part slab.
TEST(NativeSplitExpertSlabTest, AffineAndScaleAtomPartsShareOneAlignedSlab) {
  const char* root = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const auto scratch =
      std::filesystem::path(root ? root : testing::TempDir()) / "native-affine-expert-slab";
  std::filesystem::remove_all(scratch);
  std::filesystem::create_directories(scratch);
  const auto shape = *fm::ShapeOf(std::array<std::uint64_t, 2>{256, 128});
  const fm::TensorFormat biased{.encoding = fm::Encoding::kAffine,
                                .params = {.bits = 4, .group = 32, .scale = fm::AffineScale::kBF16},
                                .layout = fm::Layout::kSplit,
                                .shape = shape};
  auto zeroed = biased;
  zeroed.params.zero_point = true;
  const fm::TensorFormat atoms{
      .encoding = fm::Encoding::kNvfp4, .layout = fm::Layout::kScaleAtoms, .shape = shape};
  const std::array<std::pair<fm::TensorFormat, fm::PartRole>, 3> cases{
      {{biased, fm::PartRole::kBiases},
       {zeroed, fm::PartRole::kZeros},
       {atoms, fm::PartRole::kCodes}}};
  for (std::size_t c = 0; c < cases.size(); ++c) {
    SCOPED_TRACE(c);
    const auto& [f, third] = cases[c];
    std::vector<ts::V1Array> arrays{
        {.name = "blk.0.ffn_down_exps.weight", .format = f, .part = fm::PartRole::kCodes},
        {.name = "blk.0.ffn_down_exps.weight_scale",
         .format = f,
         .part = fm::PartRole::kBlockScales}};
    if (third != fm::PartRole::kCodes)
      arrays.push_back({.name = "blk.0.ffn_down_exps.weight_bias", .format = f, .part = third});
    ts::V1Plan plan{.expert_count = 2,
                    .groups = {{.kind = ar::GroupKind::kExpert, .layer = 0, .arrays = arrays}}};
    auto artifact = ar::Artifact::Open(ts::Publish(scratch / std::to_string(c), ts::Render(plan)));
    ASSERT_TRUE(artifact) << (artifact ? "" : artifact.error().ToString());
    const auto slabs = en::ExpertSlabsOf(*artifact, 256);
    ASSERT_TRUE(slabs) << (slabs ? "" : slabs.error());
    ASSERT_EQ(slabs->slabs.size(), 1U);
    for (std::uint32_t i = 0; i < arrays.size(); ++i)
      EXPECT_EQ(slabs->pitch[i], slabs->slabs.front().stride);
  }
  std::filesystem::remove_all(scratch);
}
}  // namespace
