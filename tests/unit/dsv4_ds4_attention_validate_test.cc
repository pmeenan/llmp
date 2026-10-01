// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Refusal and paid-scratch controls for the temporary original reference.
// Synthetic logical addresses are never dereferenced.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <expected>
#include <limits>

#include "kernels/ggml/dsv4_ds4_attention.h"

namespace {
namespace kg = jitllm::kernels::ggml;
constexpr std::uint64_t kBase = 1ULL << 40;
constexpr std::uint64_t kSlot = 1ULL << 36;

kg::Ds4CacheBuffer View(std::uint32_t slot, std::uint64_t bytes) {
  return {kBase + (static_cast<std::uint64_t>(slot) * kSlot), bytes};
}

template <typename T>
void Refused(const std::expected<T, kg::KernelFailure>& value) {
  ASSERT_FALSE(value);
  EXPECT_EQ(value.error().error, kg::KernelError::kRejected);
}

kg::Ds4IndexerScores Scores(std::uint32_t tokens = 2, std::uint32_t cells = 1024) {
  return {.query = View(0, std::uint64_t{tokens} * 64ULL * 128ULL * 4ULL),
          .weights = View(1, std::uint64_t{tokens} * 64ULL * 4ULL),
          .keys = View(2, std::uint64_t{cells} * 128ULL * 4ULL),
          .scores = View(3, std::uint64_t{tokens} * cells * 4ULL),
          .diagnostics = View(4, sizeof(kg::Ds4IndexerDiagnostics)),
          .tokens = tokens,
          .cells = cells,
          .cells_per_bank = cells,
          .first = 8192,
          .score_band = cells};
}

kg::Ds4IndexerSelect Select(std::uint32_t tokens = 2, std::uint32_t cells = 1024) {
  return {.scores = View(3, std::uint64_t{tokens} * cells * 4ULL),
          .selected = View(5, std::uint64_t{tokens} * 512ULL * 4ULL),
          .diagnostics = View(4, sizeof(kg::Ds4IndexerDiagnostics)),
          .tokens = tokens,
          .cells = cells,
          .score_band = cells};
}

kg::Ds4Attention Attention(std::uint32_t tokens = 4) {
  return {.query = View(0, std::uint64_t{tokens} * 64ULL * 512ULL * 4ULL),
          .output = View(1, std::uint64_t{tokens} * 64ULL * 512ULL * 4ULL),
          .sinks = View(2, 64ULL * 4ULL),
          .raw = View(3, 512ULL * 512ULL * 4ULL),
          .compressed = View(4, 1024ULL * 512ULL * 4ULL),
          .diagnostics = View(5, sizeof(kg::Ds4AttentionDiagnostics)),
          .tokens = tokens,
          .first = 4096,
          .raw_cells = 512,
          .raw_count = tokens + 127,
          .raw_start = 385,
          .compressed_cells = 1024,
          .compressed_count = 1024};
}

TEST(Ds4AttentionValidateTest, ScoreViewsScalesAndBankClaimsAreBounded) {
  const auto valid = Scores();
  ASSERT_TRUE(kg::CheckDs4IndexerScores(valid));
  auto desc = valid;
  --desc.query.bytes;
  Refused(kg::CheckDs4IndexerScores(desc));
  desc = valid;
  ++desc.keys.address;
  Refused(kg::CheckDs4IndexerScores(desc));
  desc = valid;
  desc.scores.address = desc.query.address + 16;
  Refused(kg::CheckDs4IndexerScores(desc));
  desc = valid;
  desc.scale = std::numeric_limits<float>::quiet_NaN();
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.scale = std::numeric_limits<float>::infinity();
  Refused(kg::CheckDs4IndexerScores(desc));
  desc = valid;
  desc.first = std::numeric_limits<std::uint32_t>::max();
  Refused(kg::CheckDs4IndexerScores(desc));
  desc = valid;
  desc.positions = View(6, 8);
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.bank_ids = View(7, 8);
  ASSERT_TRUE(kg::CheckDs4IndexerScores(desc));
  desc.consecutive_bank = 1;
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.consecutive_bank = 0;
  ASSERT_TRUE(kg::CheckDs4IndexerScores(desc));
  desc.key_codes = View(8, 1024ULL * 64ULL);
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.key_scales = View(9, 1024ULL * 16ULL);
  ASSERT_TRUE(kg::CheckDs4IndexerScores(desc));
  desc.key_codes.bytes = std::numeric_limits<std::uint64_t>::max();
  Refused(kg::CheckDs4IndexerScores(desc));
  desc = valid;
  desc.query_scales = {0, 1};
  Refused(kg::CheckDs4IndexerScores(desc));
}

TEST(Ds4AttentionValidateTest, Mxf4RequiresTheCompleteOriginalProducerGeometry) {
  auto desc = Scores(33, 1025);
  desc.key_codes = View(8, 1025ULL * 64ULL);
  desc.key_scales = View(9, 1025ULL * 16ULL);
  desc.kind = kg::Ds4IndexerScoreKind::kMxf4;
  const auto scratch = kg::Ds4IndexerScoreScratchBytes(desc);
  ASSERT_TRUE(scratch);
  EXPECT_EQ(*scratch, (((33ULL * 64ULL * 4ULL) + 255) & ~255ULL) +
                          (((33ULL * 64ULL * 64ULL) + 255) & ~255ULL));
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.scratch = View(10, *scratch);
  ASSERT_TRUE(kg::CheckDs4IndexerScores(desc));
  --desc.scratch.bytes;
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.scratch.bytes = *scratch;
  desc.query_codes = View(11, 33ULL * 64ULL * 64ULL);
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.query_scales = View(12, 33ULL * 64ULL * 16ULL);
  ASSERT_TRUE(kg::CheckDs4IndexerScores(desc));
  const auto packed = kg::Ds4IndexerScoreScratchBytes(desc);
  ASSERT_TRUE(packed);
  EXPECT_EQ(*packed, ((33ULL * 64ULL * 4ULL) + 255) & ~255ULL);
  desc.quality_mode = true;
  Refused(kg::CheckDs4IndexerScores(desc));
  desc.quality_mode = false;
  desc.ratio = 8;
  Refused(kg::CheckDs4IndexerScores(desc));
}

TEST(Ds4AttentionValidateTest, LiveSelectionUsesStableBandInsteadOfBakeTimeCount) {
  auto score = Scores(1, 17);
  score.cells_per_bank = 4096;
  score.score_band = 4096;
  score.keys.bytes = 4096ULL * 128ULL * 4ULL;
  score.scores.bytes = 4096ULL * 4ULL;
  score.causal = false;
  score.layer_scalars = View(6, sizeof(kg::Ds4IndexerLayerScalars));
  ASSERT_TRUE(kg::CheckDs4IndexerScores(score));
  auto select = Select(1, 17);
  select.score_band = 4096;
  select.scores = score.scores;
  select.layer_scalars = score.layer_scalars;
  ASSERT_TRUE(kg::CheckDs4IndexerSelect(select));
  select.kind = kg::Ds4IndexerSelectKind::kBitonic2048;
  Refused(kg::CheckDs4IndexerSelect(select));
  score.kind = kg::Ds4IndexerScoreKind::kScalar;
  Refused(kg::CheckDs4IndexerScores(score));
  score.kind = kg::Ds4IndexerScoreKind::kDirectOne;
  ASSERT_TRUE(kg::CheckDs4IndexerScores(score));
  score.tokens = 2;
  Refused(kg::CheckDs4IndexerScores(score));
}

TEST(Ds4AttentionValidateTest, TreeScratchPaysEveryMergeLevelAndCannotAliasScores) {
  auto desc = Select(3, 262144);
  desc.kind = kg::Ds4IndexerSelectKind::kChunkTree;
  const auto bytes = kg::Ds4IndexerSelectScratchBytes(desc);
  ASSERT_TRUE(bytes);
  EXPECT_EQ(*bytes, 3ULL * (64 + 8) * 512ULL * 4ULL);
  Refused(kg::CheckDs4IndexerSelect(desc));
  desc.scratch = View(10, *bytes);
  ASSERT_TRUE(kg::CheckDs4IndexerSelect(desc));
  desc.scratch.address = desc.scores.address;
  Refused(kg::CheckDs4IndexerSelect(desc));
  desc = Select();
  desc.kind = kg::Ds4IndexerSelectKind::kStream512;
  ASSERT_TRUE(kg::CheckDs4IndexerSelect(desc));
  desc.scratch = {0, 1};
  Refused(kg::CheckDs4IndexerSelect(desc));
  desc = Select();
  --desc.selected.bytes;
  Refused(kg::CheckDs4IndexerSelect(desc));
}

TEST(Ds4AttentionValidateTest, AttentionRejectsPartialPackedTablesAndWritableAliases) {
  const auto valid = Attention();
  ASSERT_TRUE(kg::CheckDs4Attention(valid));
  auto desc = valid;
  --desc.query.bytes;
  Refused(kg::CheckDs4Attention(desc));
  desc = valid;
  desc.output.address = desc.raw.address + 16;
  Refused(kg::CheckDs4Attention(desc));
  desc = valid;
  desc.raw_count = desc.raw_cells + 1;
  Refused(kg::CheckDs4Attention(desc));
  desc = valid;
  desc.raw_start = desc.raw_cells;
  Refused(kg::CheckDs4Attention(desc));
  desc = valid;
  desc.compressed_codes = View(6, 1024ULL * 704ULL);
  Refused(kg::CheckDs4Attention(desc));
  desc.compressed_scales = View(7, 1024ULL * 28ULL);
  Refused(kg::CheckDs4Attention(desc));
  desc.decode_table = View(8, 512);
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
  desc.compressed_count = 0;
  desc.layer_scalars = View(9, sizeof(kg::Ds4IndexerLayerScalars));
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
  --desc.decode_table.bytes;
  Refused(kg::CheckDs4Attention(desc));
  desc = valid;
  desc.selected = {std::numeric_limits<std::uint64_t>::max(), 1};
  Refused(kg::CheckDs4Attention(desc));
}

TEST(Ds4AttentionValidateTest, IndexedTokenTilePaysUnionsMirrorsAndPartialTiles) {
  auto desc = Attention(129);
  desc.domain = kg::Ds4AttentionDomain::kIndexedRing;
  desc.selected = View(6, 129ULL * 512ULL * 4ULL);
  desc.consecutive_first = desc.first;
  desc.kind = kg::Ds4AttentionKind::kTokenTile;
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
  const auto plan = kg::PlanDs4AttentionScratch(desc, desc.kind, 48);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->record_stride, 1024);
  EXPECT_EQ(plan->records, 0);
  EXPECT_EQ(plan->counts, 33ULL * 1024ULL * 8ULL);
  EXPECT_EQ(plan->raw_mirror, (plan->counts + (33ULL * 4ULL) + 255) & ~255ULL);
  EXPECT_EQ(plan->compressed_mirror, plan->raw_mirror + (256ULL * 512ULL * 2ULL));
  EXPECT_EQ(plan->bytes, plan->compressed_mirror + (1024ULL * 512ULL * 2ULL));
  desc.mask = View(7, 129ULL * 1024ULL * 4ULL);
  Refused(kg::CheckDs4Attention(desc));
  desc.mask = {};
  desc.consecutive_first = std::numeric_limits<std::uint32_t>::max();
  Refused(kg::CheckDs4Attention(desc));
  desc = Attention(128);
  desc.positions = View(7, 128ULL * 4ULL);
  desc.bank_ids = View(8, 128ULL * 4ULL);
  desc.consecutive_first = desc.first;
  desc.allow_multisequence_heads8 = true;
  desc.quality_mode = true;
  // Original per-sequence token tile has no quality-mode disengage.
  ASSERT_TRUE(kg::PlanDs4AttentionScratch(desc, kg::Ds4AttentionKind::kTokenTile, 48));
}

TEST(Ds4AttentionValidateTest, OriginalScalarCapsAndHeadGroupSplitsAreExplicit) {
  auto desc = Attention();
  const auto hg = kg::PlanDs4AttentionScratch(desc, kg::Ds4AttentionKind::kHeadGroup, 48);
  ASSERT_TRUE(hg);
  EXPECT_EQ(hg->splits, 12);
  EXPECT_EQ(hg->bytes, ((4ULL * 64ULL * 12ULL * 514ULL * 4ULL) + 255) & ~255ULL);
  desc.tokens = 9;
  Refused(kg::PlanDs4AttentionScratch(desc, kg::Ds4AttentionKind::kHeadGroup, 48));
  desc = Attention(4096);
  desc.domain = kg::Ds4AttentionDomain::kMixedPrefill;
  desc.first = 0;
  desc.raw_cells = 4096;
  desc.raw_count = 4096;
  desc.raw_start = 0;
  desc.raw.bytes = 4096ULL * 512ULL * 4ULL;
  desc.kind = kg::Ds4AttentionKind::kScalar;
  Refused(kg::CheckDs4Attention(desc));
  desc.kind = kg::Ds4AttentionKind::kCublas;
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
  const auto gemm = kg::PlanDs4AttentionScratch(desc, desc.kind, 48);
  ASSERT_TRUE(gemm);
  EXPECT_EQ(gemm->gemm_scores, 5120ULL * 512ULL * 4ULL);
  EXPECT_EQ(gemm->gemm_output, gemm->gemm_scores + (64ULL * 4096ULL * 5120ULL * 4ULL));
  desc = Attention(1);
  desc.domain = kg::Ds4AttentionDomain::kDecodeHeads;
  desc.window = 0;
  desc.ratio = 0;
  desc.raw_count = 128;
  desc.kind = kg::Ds4AttentionKind::kPerHeadSplit;
  const auto split = kg::PlanDs4AttentionScratch(desc, desc.kind, 188);
  ASSERT_TRUE(split);
  EXPECT_EQ(split->splits, 4);
  Refused(kg::PlanDs4AttentionScratch(desc, desc.kind, 48));
}

TEST(Ds4AttentionValidateTest, LiveAndPerRowOperandsDisengageUnsafeAlternateTiers) {
  auto desc = Attention(4);
  desc.domain = kg::Ds4AttentionDomain::kIndexedRing;
  desc.selected = View(6, 4ULL * 512ULL * 4ULL);
  desc.positions = View(7, 4ULL * 4ULL);
  desc.bank_ids = View(8, 4ULL * 4ULL);
  desc.layer_scalars = View(9, sizeof(kg::Ds4IndexerLayerScalars));
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
  desc.kind = kg::Ds4AttentionKind::kHeads8Online;
  Refused(kg::CheckDs4Attention(desc));
  desc.kind = kg::Ds4AttentionKind::kIndexedTwoPass;
  Refused(kg::CheckDs4Attention(desc));
  desc.kind = kg::Ds4AttentionKind::kHeadGroup;
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
  desc.draft_raw_count = View(10, 4ULL * 4ULL);
  Refused(kg::CheckDs4Attention(desc));
  desc = Attention(1);
  desc.compressed_count = 3;
  desc.layer_scalars = View(9, sizeof(kg::Ds4IndexerLayerScalars));
  desc.mask = View(10, 3ULL * 4ULL);
  Refused(kg::CheckDs4Attention(desc));
  desc.mask.bytes = 1024ULL * 4ULL;
  ASSERT_TRUE(kg::CheckDs4Attention(desc));
}
TEST(Ds4AttentionValidateTest, MaskProducerRequiresCompleteDisjointOutput) {
  kg::Ds4AttentionMask desc{.selected = View(0, 4ULL * 2ULL * 4ULL),
                            .mask = View(1, 4ULL * 8ULL * 4ULL),
                            .tokens = 4,
                            .cells = 8,
                            .top_k = 2};
  ASSERT_TRUE(kg::CheckDs4AttentionMask(desc));
  --desc.mask.bytes;
  Refused(kg::CheckDs4AttentionMask(desc));
  desc.mask.bytes += 1;
  desc.mask.address = desc.selected.address + 4;
  Refused(kg::CheckDs4AttentionMask(desc));
  desc.mask = View(1, 4ULL * 8ULL * 4ULL);
  desc.tokens = 0;
  Refused(kg::CheckDs4AttentionMask(desc));
}

}  // namespace
