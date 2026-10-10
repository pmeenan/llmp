// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/draft_progress.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "format/encoding.h"
#include "model/state_contract.h"

namespace {

namespace md = llmp::model;
using md::DraftRelation;
using md::DraftRows;
constexpr md::DraftProgressBounds kShifted{
    .context = 4096, .max_rows = 16, .carry = 1, .relation = DraftRelation::kTokenAfterCache};

TEST(DraftProgressTest, RecordNamesTheWholeRepeatableScheduleAndRetainedTap) {
  const md::DraftProgress p{
      .positions = 10, .pending_first = 7, .tap_first = 7, .tap_rows = 3, .preceding = 1};
  const md::DraftProgressWords expected{1, 10, 7, 7, 3, 1, 1, 0};
  auto encoded = md::EncodeDraftProgress(p, kShifted);
  ASSERT_TRUE(encoded) << llmp::test_support::Failed(encoded).value_or("");
  EXPECT_EQ(*encoded, expected);
  auto decoded = md::DecodeDraftProgress(expected, 10, kShifted);
  ASSERT_TRUE(decoded) << llmp::test_support::Failed(decoded).value_or("");
  EXPECT_EQ(*decoded, p);
  auto first = md::DraftFirstPass(p, kShifted);
  ASSERT_TRUE(first) << llmp::test_support::Failed(first).value_or("");
  EXPECT_EQ(*first, (DraftRows{7, 8, 7, 3}));
}

TEST(DraftProgressTest, EmptyHasOneRepresentationAndCannotSupplyAFeature) {
  auto empty = md::EncodeDraftProgress({}, kShifted);
  ASSERT_TRUE(empty) << llmp::test_support::Failed(empty).value_or("");
  EXPECT_EQ(*empty, md::DraftProgressWords{});
  EXPECT_TRUE(md::DecodeDraftProgress(*empty, 0, kShifted));
  EXPECT_FALSE(md::DraftFirstPass({}, kShifted));
  for (std::size_t word = 0; word < empty->size(); ++word) {
    auto broken = *empty;
    broken[word] = 1;
    EXPECT_FALSE(md::DecodeDraftProgress(broken, 0, kShifted)) << word;
  }
  EXPECT_FALSE(md::DecodeDraftProgress(*empty, 1, kShifted));
  auto zero_context = kShifted;
  zero_context.context = 0;
  EXPECT_FALSE(md::DecodeDraftProgress(*empty, 0, zero_context));
}

TEST(DraftProgressTest, RestoredWordsCannotChangeTheRecipeOrNameAbsentFeatures) {
  const md::DraftProgressWords good{1, 10, 7, 7, 3, 1, 1, 0};
  for (std::size_t word = 1; word <= 5; ++word) {
    for (const std::int64_t value : {-1LL, 1LL << 32}) {
      auto bad = good;
      bad[word] = value;
      EXPECT_FALSE(md::DecodeDraftProgress(bad, 10, kShifted)) << word << ' ' << value;
    }
  }
  for (const auto [word, value] : std::array<std::pair<std::size_t, std::int64_t>, 10>{
           {{0, 2}, {1, 11}, {2, 10}, {2, 5}, {3, 8}, {4, 0}, {4, 17}, {5, 2}, {6, 0}, {7, 1}}}) {
    auto bad = good;
    bad[word] = value;
    EXPECT_FALSE(md::DecodeDraftProgress(bad, 10, kShifted)) << word << ' ' << value;
  }
  auto short_bound = kShifted;
  short_bound.max_rows = 2;
  EXPECT_FALSE(md::DecodeDraftProgress(good, 10, short_bound));
  auto invalid_relation = kShifted;
  invalid_relation.relation = static_cast<DraftRelation>(255);
  EXPECT_FALSE(md::DecodeDraftProgress(good, 10, invalid_relation));
}

TEST(DraftProgressTest, RelationsNameDifferentCacheCoordinatesForTheSameTokensAndFeatures) {
  const md::DraftProgress shifted{10, 7, 7, 3, 1};
  auto aligned_bounds = kShifted;
  aligned_bounds.relation = DraftRelation::kTokenAtCache;
  const md::DraftProgress aligned{10, 8, 7, 3, 1};
  EXPECT_EQ(md::DraftFirstPass(shifted, kShifted).value(), (DraftRows{7, 8, 7, 3}));
  EXPECT_EQ(md::DraftFirstPass(aligned, aligned_bounds).value(), (DraftRows{8, 8, 7, 3}));
  EXPECT_EQ(md::DraftFirstPass({1, 0, 0, 1, 0}, kShifted).value(), (DraftRows{0, 1, 0, 1}));
  EXPECT_EQ(md::DraftFirstPass({1, 1, 0, 1, 0}, aligned_bounds).value(), (DraftRows{1, 1, 0, 1}));
  EXPECT_FALSE(md::DraftFirstPass({1, 0, 0, 1, 0}, aligned_bounds));
  EXPECT_TRUE(md::ValidDraftPolicy({}));
  EXPECT_FALSE(md::ValidDraftPolicy({.relation = DraftRelation::kTokenAfterCache}));
}

TEST(DraftProgressTest, PromptCompanionsKeepTheOriginalHeadScheduleUntilCompoundPublication) {
  const md::DraftProgress old{10, 7, 7, 3, 1};
  auto companions = md::DraftPrefillCompanions(old, kShifted, 4);
  ASSERT_TRUE(companions) << llmp::test_support::Failed(companions).value_or("");
  EXPECT_EQ(companions->before, (DraftRows{7, 8, 7, 2}));
  EXPECT_EQ(companions->after, (DraftRows{9, 10, 9, 4}));
  EXPECT_EQ(companions->final_pending_first, 13U);
  // Planning or completing the pre companion cannot mutate this record.
  // Actual refusal/retirement enforcement is the lifecycle's separate test.
  EXPECT_EQ(md::DraftFirstPass(old, kShifted).value(), (DraftRows{7, 8, 7, 3}));
  auto fresh = md::DraftPrefillCompanions({}, kShifted, 4);
  ASSERT_TRUE(fresh) << llmp::test_support::Failed(fresh).value_or("");
  EXPECT_EQ(fresh->before.count, 0U);
  EXPECT_EQ(fresh->after, (DraftRows{0, 1, 0, 3}));
  EXPECT_EQ(fresh->final_pending_first, 3U);
  auto one = md::DraftPrefillCompanions({}, kShifted, 1);
  ASSERT_TRUE(one) << llmp::test_support::Failed(one).value_or("");
  EXPECT_EQ(one->before.count, 0U);
  EXPECT_EQ(one->after.count, 0U);
  EXPECT_EQ(one->final_pending_first, 0U);
  auto no_carry = kShifted;
  no_carry.carry = 0;
  EXPECT_FALSE(md::DraftPrefillCompanions({10, 9, 7, 3, 0}, no_carry, 4));
}

TEST(DraftProgressTest, ContextEndPersistsButCannotAddAnAnchorOrPrompt) {
  const auto n = std::numeric_limits<std::uint32_t>::max();
  const md::DraftProgressBounds bounds{
      .context = n, .max_rows = 16, .carry = 1, .relation = DraftRelation::kTokenAfterCache};
  const md::DraftProgress p{n, n - 3, n - 3, 3, 1};
  auto record = md::EncodeDraftProgress(p, bounds);
  ASSERT_TRUE(record) << llmp::test_support::Failed(record).value_or("");
  EXPECT_EQ(md::DecodeDraftProgress(*record, n, bounds).value(), p);
  EXPECT_FALSE(md::DraftFirstPass(p, bounds));
  EXPECT_FALSE(md::DraftPrefillCompanions(p, bounds, 1));
  EXPECT_FALSE(md::DraftPrefillCompanions({n - 1, n - 2, n - 2, 1, 1}, bounds, 2));
}

md::StateContract RepeatableContract(DraftRelation relation = DraftRelation::kTokenAfterCache) {
  return {.tensors = {{.name = "kv",
                       .kind = md::StateKind::kKv,
                       .encoding = md::Encoding::kF32,
                       .cell_bytes = 1024,
                       .cells = md::CellRule::kPerPosition,
                       .replayable = true},
                      {.name = "features",
                       .kind = md::StateKind::kFeatures,
                       .encoding = md::Encoding::kF32,
                       .cell_bytes = 16,
                       .cells = md::CellRule::kPerRow,
                       .window_or_ratio = 1,
                       .rollback = md::Rollback::kSnapshot,
                       .target_writes = true},
                      md::DraftProgressControl("progress", relation)}};
}

auto Layout(md::StateContract contract, std::uint32_t context = 8192, std::uint32_t max_rows = 4) {
  const std::array layers{std::pair{0U, std::move(contract)}};
  return md::Layout(layers, context, max_rows, 2ULL << 20U);
}

TEST(DraftProgressTest, ControlUsesOrdinaryStateWithoutClaimingI64NumericKvSupport) {
  auto layout = Layout(RepeatableContract());
  ASSERT_TRUE(layout) << llmp::test_support::Failed(layout).value_or("");
  auto bounds = md::DraftBounds(*layout);
  ASSERT_TRUE(bounds) << llmp::test_support::Failed(bounds).value_or("");
  ASSERT_TRUE(bounds->has_value());
  EXPECT_EQ((*bounds)->context, 8192U);
  EXPECT_EQ((*bounds)->max_rows, 4U);
  EXPECT_EQ((*bounds)->carry, 1U);
  EXPECT_EQ((*bounds)->relation, DraftRelation::kTokenAfterCache);
  const auto& control = layout->items.back();
  EXPECT_EQ(control.bytes, 64U);
  EXPECT_EQ(control.cells, 1U);
  EXPECT_EQ(control.contract.encoding, md::Encoding::kI64);
  EXPECT_FALSE(llmp::format::IsStateEncoding(md::Encoding::kI64));
  EXPECT_FALSE(md::CellBytes(md::Encoding::kI64, 8));
  auto used = md::UsedThrough(*layout, 2048);
  ASSERT_TRUE(used) << llmp::test_support::Failed(used).value_or("");
  EXPECT_EQ(used->back().offset, control.offset);
  EXPECT_EQ(used->back().bytes, 64U);
  auto saves = md::WritesFor(*layout, 100, 4, {.drafter = true});
  ASSERT_TRUE(saves) << llmp::test_support::Failed(saves).value_or("");
  for (const auto& row : saves->rows)
    for (const auto& range : row) EXPECT_LT(range.offset, control.offset);
}

TEST(DraftProgressTest, LayoutRefusesControlMasqueradingAndUnqualifiedReplay) {
  for (int variant = 0; variant < 11; ++variant) {
    auto c = RepeatableContract();
    auto& control = c.tensors.back();
    switch (variant) {
      case 0:
        control.encoding = md::Encoding::kF64;
        break;
      case 1:
        control.cell_bytes = 128;
        break;
      case 2:
        control.target_writes = true;
        break;
      case 3:
        control.rollback = md::Rollback::kSnapshot;
        break;
      case 4:
        control.draft.progress = md::DraftProgressPolicy::kCursor;
        break;
      case 5:
        c.tensors.front().replayable = false;
        break;
      case 6:
        c.tensors[1].window_or_ratio = 0;
        break;
      case 7:
        c.tensors[1].rollback = md::Rollback::kTruncate;
        break;
      case 8:
        c.tensors.push_back(control);
        break;
      case 10:
        c.tensors.front().borrowed_from_target = true;
        break;
      case 9:
        c.tensors.front().cells = md::CellRule::kFixed;
        c.tensors.front().window_or_ratio = 1;
        break;
    }
    EXPECT_FALSE(Layout(std::move(c))) << variant;
  }
  auto masquerade = RepeatableContract();
  masquerade.tensors.front().draft = masquerade.tensors.back().draft;
  EXPECT_FALSE(Layout(std::move(masquerade)));
}

TEST(DraftProgressTest, CheckpointTailCrossesThePreviousLargePageWithoutSavingAllContext) {
  auto layout = Layout(RepeatableContract());
  ASSERT_TRUE(layout) << llmp::test_support::Failed(layout).value_or("");
  for (const auto n : {2048U, 2052U, 6144U}) {
    auto writes = md::CheckpointWrites(*layout, n);
    ASSERT_TRUE(writes) << llmp::test_support::Failed(writes).value_or("");
    ASSERT_FALSE(writes->empty());
    EXPECT_EQ(writes->front().offset, std::uint64_t{n - 4} * 1024);
    EXPECT_EQ(writes->back().offset, layout->items.back().offset);
    EXPECT_EQ(writes->back().bytes, 64U);
  }
  auto ordinary = RepeatableContract();
  ordinary.tensors.pop_back();
  auto unchanged = Layout(std::move(ordinary));
  ASSERT_TRUE(unchanged) << llmp::test_support::Failed(unchanged).value_or("");
  EXPECT_FALSE(md::DraftBounds(*unchanged).value().has_value());
  EXPECT_EQ(md::CheckpointWrites(*unchanged, 2048).value().front().offset, 2ULL << 20U);
}

TEST(DraftProgressTest, CoordinateAndReplaySemanticsChangeTheKeptLayoutId) {
  auto shifted = Layout(RepeatableContract());
  auto aligned = Layout(RepeatableContract(DraftRelation::kTokenAtCache));
  ASSERT_TRUE(shifted) << llmp::test_support::Failed(shifted).value_or("");
  ASSERT_TRUE(aligned) << llmp::test_support::Failed(aligned).value_or("");
  const auto id = md::KeptLayoutId({}, std::span(&*shifted, 1));
  EXPECT_NE(id, md::KeptLayoutId({}, std::span(&*aligned, 1)));
  auto ordinary = RepeatableContract();
  ordinary.tensors.pop_back();
  auto replay = Layout(ordinary);
  ordinary.tensors.front().replayable = false;
  auto plain = Layout(ordinary);
  ASSERT_TRUE(replay) << llmp::test_support::Failed(replay).value_or("");
  ASSERT_TRUE(plain) << llmp::test_support::Failed(plain).value_or("");
  EXPECT_NE(md::KeptLayoutId({}, std::span(&*replay, 1)),
            md::KeptLayoutId({}, std::span(&*plain, 1)));
}

}  // namespace
