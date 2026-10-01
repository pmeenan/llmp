// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/dsv4_ds4_state.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
namespace model = jitllm::model;
using Kind = model::Ds4BaselineStateKind;
constexpr std::uint64_t kGranularity = std::uint64_t{2} << 20U;

const model::Ds4BaselineStateTensor* Find(const model::Ds4BaselineStateLayout& layout, Kind kind,
                                          std::uint32_t layer) {
  const auto found = std::ranges::find_if(layout.tensors, [=](const auto& tensor) {
    return tensor.kind == kind && tensor.layer == layer;
  });
  return found == layout.tensors.end() ? nullptr : &*found;
}

TEST(Ds4BaselineState, MatchesOriginalRawAndCompressedCapacityRules) {
  const auto layout = model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 4096, kGranularity);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->raw_cells, 4352U);
  const auto* raw = Find(*layout, Kind::kRaw, 42);
  ASSERT_NE(raw, nullptr);
  EXPECT_EQ(raw->bytes, std::uint64_t{4352} * 512 * 4);
  const auto* csa = Find(*layout, Kind::kAttentionCodes, 2);
  const auto* hca = Find(*layout, Kind::kAttentionCodes, 3);
  ASSERT_NE(csa, nullptr);
  ASSERT_NE(hca, nullptr);
  EXPECT_EQ(csa->capacity, 2050U);
  EXPECT_EQ(csa->bytes, std::uint64_t{2050} * 704);
  EXPECT_EQ(hca->capacity, 66U);
  EXPECT_EQ(hca->bytes, std::uint64_t{66} * 704);
  EXPECT_EQ(Find(*layout, Kind::kAttentionCodes, 0), nullptr);
  EXPECT_EQ(Find(*layout, Kind::kIndexerCodes, 3), nullptr);
  EXPECT_EQ(Find(*layout, Kind::kAttentionF32, 2), nullptr);
  EXPECT_EQ(Find(*layout, Kind::kIndexerF32, 2), nullptr);
}

TEST(Ds4BaselineState, FrontierShapesAndScalarStorageAreFixedAndDisjoint) {
  const auto layout = model::LayoutDs4BaselineState(model::Dsv4Flash(), 32768, 2048, kGranularity);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->raw_cells, 2304U);
  const auto* csa = Find(*layout, Kind::kAttentionKv, 2);
  const auto* hca = Find(*layout, Kind::kAttentionScore, 3);
  const auto* indexer = Find(*layout, Kind::kIndexerKv, 2);
  ASSERT_NE(csa, nullptr);
  ASSERT_NE(hca, nullptr);
  ASSERT_NE(indexer, nullptr);
  EXPECT_EQ(csa->bytes, std::uint64_t{8} * 1024 * 4);
  EXPECT_EQ(hca->bytes, std::uint64_t{128} * 512 * 4);
  EXPECT_EQ(indexer->bytes, std::uint64_t{8} * 256 * 4);
  std::uint64_t previous_end = 0;
  for (const auto& tensor : layout->tensors) {
    EXPECT_GE(tensor.offset, previous_end);
    EXPECT_EQ(tensor.offset % (tensor.ratio == 0 ? 256 : kGranularity), 0U);
    EXPECT_LE(tensor.offset + tensor.bytes, layout->virtual_bytes);
    previous_end = tensor.offset + tensor.bytes;
  }
  const auto* scalar = Find(*layout, Kind::kLayerScalar, 42);
  const auto* decode = Find(*layout, Kind::kDecodeScalar, 0);
  const auto* table = Find(*layout, Kind::kDecodeTable, 0);
  ASSERT_NE(scalar, nullptr);
  ASSERT_NE(decode, nullptr);
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(scalar->bytes, 16U);
  EXPECT_EQ(decode->bytes, 40U);
  EXPECT_EQ(table->bytes, 512U);
  EXPECT_LE(table->offset + table->bytes, layout->fixed_bytes);
}

TEST(Ds4BaselineState, CommitsOnlyCompleteGroupsAndKeepsUnusedCapacityUnbacked) {
  const auto layout = model::LayoutDs4BaselineState(model::Dsv4Flash(), 262144, 4096, kGranularity);
  ASSERT_TRUE(layout.has_value());
  const auto empty = model::Ds4BaselineStateThrough(*layout, 0);
  const auto three = model::Ds4BaselineStateThrough(*layout, 3);
  ASSERT_TRUE(empty.has_value());
  ASSERT_TRUE(three.has_value());
  ASSERT_EQ(empty->size(), 1U);
  ASSERT_EQ(three->size(), 1U);
  EXPECT_EQ(empty->front().bytes, layout->fixed_bytes);
  const auto four = model::Ds4BaselineStateThrough(*layout, 4);
  ASSERT_TRUE(four.has_value());
  // 21 CSA layers, each with attention and indexer codes/scales.
  ASSERT_EQ(four->size(), 85U);
  const auto* codes = Find(*layout, Kind::kAttentionCodes, 2);
  const auto* scales = Find(*layout, Kind::kAttentionScales, 2);
  ASSERT_NE(codes, nullptr);
  ASSERT_NE(scales, nullptr);
  const auto code_range =
      std::ranges::find_if(*four, [&](auto range) { return range.offset == codes->offset; });
  const auto scale_range =
      std::ranges::find_if(*four, [&](auto range) { return range.offset == scales->offset; });
  ASSERT_NE(code_range, four->end());
  ASSERT_NE(scale_range, four->end());
  EXPECT_EQ(code_range->bytes, 704U);
  EXPECT_EQ(scale_range->bytes, 28U);
  EXPECT_LT(code_range->bytes, codes->bytes);
  const auto full = model::Ds4BaselineStateThrough(*layout, layout->context);
  ASSERT_TRUE(full.has_value());
  for (const auto& tensor : layout->tensors) {
    if (tensor.ratio == 0) continue;
    const auto range =
        std::ranges::find_if(*full, [&](auto r) { return r.offset == tensor.offset; });
    ASSERT_NE(range, full->end());
    EXPECT_EQ(tensor.bytes - range->bytes, 2 * tensor.row_bytes);
  }
  EXPECT_FALSE(model::Ds4BaselineStateThrough(*layout, layout->context + 1));
}

TEST(Ds4BaselineState, ExplicitF32ControlChangesStorageWithoutAllocatingPackedCopies) {
  const auto layout =
      model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 4096, kGranularity, false, false);
  ASSERT_TRUE(layout.has_value());
  const auto* attention = Find(*layout, Kind::kAttentionF32, 2);
  const auto* indexer = Find(*layout, Kind::kIndexerF32, 2);
  ASSERT_NE(attention, nullptr);
  ASSERT_NE(indexer, nullptr);
  EXPECT_EQ(attention->row_bytes, 2048U);
  EXPECT_EQ(indexer->row_bytes, 512U);
  EXPECT_EQ(Find(*layout, Kind::kAttentionCodes, 2), nullptr);
  EXPECT_EQ(Find(*layout, Kind::kIndexerCodes, 2), nullptr);
}

TEST(Ds4BaselineState, SmallContextsAndTrainedMaximumRemainBounded) {
  for (const auto context : {1U, 127U, 128U, 129U, model::kDsv4FlashContext}) {
    const auto layout =
        model::LayoutDs4BaselineState(model::Dsv4Flash(), context, 4096, kGranularity);
    ASSERT_TRUE(layout.has_value());
    EXPECT_LE(layout->raw_cells, context);
    EXPECT_GE(layout->raw_cells, std::min(context, 128U));
    EXPECT_EQ(layout->virtual_bytes % kGranularity, 0U);
    EXPECT_TRUE(model::Ds4BaselineStateThrough(*layout, context));
  }
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), 0, 4096, kGranularity));
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), model::kDsv4FlashContext + 1, 4096,
                                             kGranularity));
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 0, kGranularity));
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 4097, kGranularity));
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 4096, 0));
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 4096, 768));
  EXPECT_FALSE(model::LayoutDs4BaselineState(model::Dsv4Flash(), 8192, 4096,
                                             std::numeric_limits<std::uint64_t>::max()));
  auto other = model::Dsv4Flash();
  other.compress_ratios[2] = 128;
  EXPECT_FALSE(model::LayoutDs4BaselineState(other, 8192, 4096, kGranularity));
}

}  // namespace
