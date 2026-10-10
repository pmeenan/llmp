// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "engine/draft_progress.h"
#include "expected_error.h"
#include "model/draft_progress.h"

namespace {

namespace md = llmp::model;
using Ledger = llmp::engine::DraftProgressLedger;
constexpr md::DraftProgressBounds kBounds{
    .context = 4096, .max_rows = 16, .carry = 1, .relation = md::DraftRelation::kTokenAfterCache};
constexpr md::DraftProgressWords kOld{1, 10, 7, 7, 3, 1, 1, 0};

void Restore(Ledger& ledger) {
  auto restore = ledger.BeginReplacement();
  ASSERT_TRUE(restore) << llmp::test_support::Failed(restore).value_or("");
  ASSERT_TRUE(ledger.RestoreCompleted(*restore, kOld, 10));
}

TEST(DraftProgressLedgerTest, AcceptanceIsUnpublishedUntilItsControlCopyCompletes) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit) << llmp::test_support::Failed(unit).value_or("");
  EXPECT_FALSE(ledger.BeginVerify(*unit, 4));
  ASSERT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*unit, 4));
  auto candidate = ledger.AcceptCandidate(*unit, 2);
  ASSERT_TRUE(candidate) << llmp::test_support::Failed(candidate).value_or("");
  EXPECT_EQ(*candidate, (md::DraftProgress{12, 10, 10, 2, 1}));
  EXPECT_EQ(ledger.completed().positions, 10U);
  ASSERT_TRUE(ledger.Accept(*unit, 2));
  EXPECT_TRUE(ledger.owed());
  EXPECT_FALSE(ledger.settled());
  EXPECT_FALSE(ledger.BeginReplacement());
  EXPECT_FALSE(ledger.BeginUnit());
  auto copy = ledger.Publication();
  ASSERT_TRUE(copy) << llmp::test_support::Failed(copy).value_or("");
  EXPECT_EQ(copy->words, (md::DraftProgressWords{1, 12, 10, 10, 2, 1, 1, 0}));
  // Preparing or queueing these bytes does not update the completed mirror.
  EXPECT_EQ(ledger.completed().pending_first, 7U);
  ASSERT_TRUE(ledger.PublicationCompleted(copy->ticket));
  EXPECT_TRUE(ledger.settled());
  EXPECT_FALSE(ledger.owed());
  EXPECT_EQ(ledger.completed(), *candidate);
  EXPECT_FALSE(ledger.PublicationCompleted(copy->ticket));
}

TEST(DraftProgressLedgerTest, CompletedDraftAndWholeDiscardKeepTheOriginalKRowSchedule) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit) << llmp::test_support::Failed(unit).value_or("");
  ASSERT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  EXPECT_EQ(ledger.completed().pending_first, 7U);
  ASSERT_TRUE(ledger.BeginVerify(*unit, 4));
  ASSERT_TRUE(ledger.DiscardCompleted(*unit));
  EXPECT_EQ(md::DraftFirstPass(ledger.completed(), kBounds).value(), (md::DraftRows{7, 8, 7, 3}));
  EXPECT_FALSE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  auto retry = ledger.BeginUnit();
  ASSERT_TRUE(retry) << llmp::test_support::Failed(retry).value_or("");
  EXPECT_FALSE(ledger.BeginVerify(*retry, 4));
  EXPECT_FALSE(ledger.ReplayCompleted(*retry, {9, 10, 9, 1}));
  ASSERT_TRUE(ledger.ReplayCompleted(*retry, {7, 8, 7, 3}));
  EXPECT_TRUE(ledger.BeginVerify(*retry, 4));
}

TEST(DraftProgressLedgerTest, WitnessesAreBoundToTheirOwnerStageAndLogicalUnit) {
  Ledger a(kBounds), b(kBounds);
  Restore(a);
  Restore(b);
  auto ua = a.BeginUnit();
  auto ub = b.BeginUnit();
  ASSERT_TRUE(ua) << llmp::test_support::Failed(ua).value_or("");
  ASSERT_TRUE(ub) << llmp::test_support::Failed(ub).value_or("");
  EXPECT_FALSE(b.ReplayCompleted(*ua, {7, 8, 7, 3}));
  ASSERT_TRUE(a.ReplayCompleted(*ua, {7, 8, 7, 3}));
  EXPECT_FALSE(b.BeginVerify(*ub, 2));
  ASSERT_TRUE(a.BeginVerify(*ua, 2));
  EXPECT_FALSE(b.BeginVerify(*ua, 2));
  ASSERT_TRUE(a.DiscardCompleted(*ua));
  auto restore = a.BeginReplacement();
  ASSERT_TRUE(restore) << llmp::test_support::Failed(restore).value_or("");
  ASSERT_TRUE(a.RestoreCompleted(*restore, kOld, 10));
  EXPECT_FALSE(a.ReplayCompleted(*ua, {7, 8, 7, 3}));
}

TEST(DraftProgressLedgerTest, PreCompanionThenCleanTargetRefusalRetriesTheOriginalK3) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto compound = ledger.BeginCompound(4);
  ASSERT_TRUE(compound) << llmp::test_support::Failed(compound).value_or("");
  EXPECT_FALSE(ledger.settled());
  EXPECT_FALSE(ledger.BeginReplacement());
  EXPECT_EQ(compound->shapes.before, (md::DraftRows{7, 8, 7, 2}));
  ASSERT_TRUE(ledger.PreQueued(compound->unit));
  ASSERT_TRUE(ledger.PreCompleted(compound->unit));
  EXPECT_EQ(ledger.completed().pending_first, 7U);
  EXPECT_FALSE(ledger.Publication());
  ASSERT_TRUE(ledger.RefuseCompound(compound->unit));
  EXPECT_TRUE(ledger.settled());
  auto retry = ledger.BeginUnit();
  ASSERT_TRUE(retry) << llmp::test_support::Failed(retry).value_or("");
  EXPECT_EQ(md::DraftFirstPass(ledger.completed(), kBounds).value(), (md::DraftRows{7, 8, 7, 3}));
  ASSERT_TRUE(ledger.ReplayCompleted(*retry, {7, 8, 7, 3}));
  EXPECT_TRUE(ledger.BeginVerify(*retry, 4));
}

TEST(DraftProgressLedgerTest, QueuedPreOrLeadingCarryWritesCannotClaimACleanCompoundRefusal) {
  for (const bool carry_write : {false, true}) {
    Ledger ledger(kBounds);
    Restore(ledger);
    auto compound = ledger.BeginCompound(4);
    ASSERT_TRUE(compound) << llmp::test_support::Failed(compound).value_or("");
    ASSERT_TRUE(ledger.PreQueued(compound->unit));
    if (carry_write) {
      ASSERT_TRUE(ledger.PreCompleted(compound->unit));
      // This happens before the first leading-carry copy, even if the
      // target kernel will subsequently never launch.
      ASSERT_TRUE(ledger.TargetQueued(compound->unit));
    }
    EXPECT_FALSE(ledger.RefuseCompound(compound->unit));
    EXPECT_EQ(ledger.state(), Ledger::State::kQuarantined);
    EXPECT_FALSE(ledger.BeginUnit());
    EXPECT_FALSE(ledger.BeginReplacement());
  }
}

TEST(DraftProgressLedgerTest, CompletePromptSequencePublishesOnlyItsFinalSchedule) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto compound = ledger.BeginCompound(4);
  ASSERT_TRUE(compound) << llmp::test_support::Failed(compound).value_or("");
  EXPECT_FALSE(ledger.TargetQueued(compound->unit));
  EXPECT_FALSE(ledger.FinishCompound(compound->unit));
  ASSERT_TRUE(ledger.PreQueued(compound->unit));
  ASSERT_TRUE(ledger.PreCompleted(compound->unit));
  ASSERT_TRUE(ledger.TargetQueued(compound->unit));
  ASSERT_TRUE(ledger.TargetCompleted(compound->unit));
  EXPECT_FALSE(ledger.FinishCompound(compound->unit));
  ASSERT_TRUE(ledger.PostCompleted(compound->unit));
  ASSERT_TRUE(ledger.FinishCompound(compound->unit));
  EXPECT_EQ(ledger.completed().pending_first, 7U);
  EXPECT_EQ(ledger.Publication().value().words,
            (md::DraftProgressWords{1, 14, 13, 10, 4, 1, 1, 0}));
  auto copy = ledger.Publication();
  ASSERT_TRUE(copy) << llmp::test_support::Failed(copy).value_or("");
  ASSERT_TRUE(ledger.PublicationCompleted(copy->ticket));
  EXPECT_EQ(ledger.completed(), (md::DraftProgress{14, 13, 10, 4, 1}));
}

TEST(DraftProgressLedgerTest, FailedPublicationNeverMakesANewerMirrorUsable) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit) << llmp::test_support::Failed(unit).value_or("");
  ASSERT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*unit, 4));
  ASSERT_TRUE(ledger.Accept(*unit, 2));
  auto copy = ledger.Publication();
  ASSERT_TRUE(copy) << llmp::test_support::Failed(copy).value_or("");
  // The caller observed an unproven control copy after numeric commit.
  ledger.Quarantine();
  EXPECT_EQ(ledger.completed().positions, 10U);
  EXPECT_FALSE(ledger.PublicationCompleted(copy->ticket));
  EXPECT_FALSE(ledger.BeginUnit());
  EXPECT_FALSE(ledger.settled());
}

TEST(DraftProgressLedgerTest, ReplacementDoesNotExposeARecordUntilCompletedHydration) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto old_unit = ledger.BeginUnit();
  ASSERT_TRUE(old_unit) << llmp::test_support::Failed(old_unit).value_or("");
  auto restore = ledger.BeginReplacement();
  ASSERT_TRUE(restore) << llmp::test_support::Failed(restore).value_or("");
  EXPECT_FALSE(ledger.settled());
  EXPECT_FALSE(ledger.BeginUnit());
  EXPECT_FALSE(ledger.ReplayCompleted(*old_unit, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.RestoreCompleted(*restore, kOld, 10));
  EXPECT_TRUE(ledger.settled());
  restore = ledger.BeginReplacement();
  ASSERT_TRUE(restore) << llmp::test_support::Failed(restore).value_or("");
  auto corrupt = kOld;
  corrupt[2] = -1;
  EXPECT_FALSE(ledger.RestoreCompleted(*restore, corrupt, 10));
  EXPECT_EQ(ledger.state(), Ledger::State::kQuarantined);
  EXPECT_FALSE(ledger.settled());
}

TEST(DraftProgressLedgerTest, FirstTargetRowCreatesProgressWithoutInventingAnOldWitness) {
  Ledger ledger(kBounds);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit) << llmp::test_support::Failed(unit).value_or("");
  ASSERT_TRUE(ledger.BeginVerify(*unit, 1));
  ASSERT_TRUE(ledger.Accept(*unit, 1));
  EXPECT_EQ(ledger.Publication().value().words, (md::DraftProgressWords{1, 1, 0, 0, 1, 0, 1, 0}));
  auto copy = ledger.Publication();
  ASSERT_TRUE(copy) << llmp::test_support::Failed(copy).value_or("");
  ASSERT_TRUE(ledger.PublicationCompleted(copy->ticket));
  auto clear = ledger.BeginClear();
  ASSERT_TRUE(clear) << llmp::test_support::Failed(clear).value_or("");
  ASSERT_TRUE(ledger.ClearCompleted(*clear));
  auto compound = ledger.BeginCompound(1);
  ASSERT_TRUE(compound) << llmp::test_support::Failed(compound).value_or("");
  EXPECT_EQ(compound->shapes.before.count, 0U);
  EXPECT_EQ(compound->shapes.after.count, 0U);
  ASSERT_TRUE(ledger.TargetQueued(compound->unit));
  ASSERT_TRUE(ledger.TargetCompleted(compound->unit));
  ASSERT_TRUE(ledger.FinishCompound(compound->unit));
  EXPECT_EQ(ledger.Publication().value().words, (md::DraftProgressWords{1, 1, 0, 0, 1, 0, 1, 0}));
}

TEST(DraftProgressLedgerTest, TerminalCompletionsCannotActOnLaterSamePositionTransitions) {
  Ledger ledger(kBounds), peer(kBounds);
  Restore(ledger);
  Restore(peer);
  auto old_restore = ledger.BeginReplacement();
  ASSERT_TRUE(old_restore) << llmp::test_support::Failed(old_restore).value_or("");
  ASSERT_TRUE(ledger.RestoreCompleted(*old_restore, kOld, 10));
  auto new_restore = ledger.BeginReplacement();
  ASSERT_TRUE(new_restore) << llmp::test_support::Failed(new_restore).value_or("");
  EXPECT_FALSE(ledger.RestoreCompleted(*old_restore, kOld, 10));
  EXPECT_EQ(ledger.state(), Ledger::State::kReplacing);
  auto peer_restore = peer.BeginReplacement();
  ASSERT_TRUE(peer_restore) << llmp::test_support::Failed(peer_restore).value_or("");
  EXPECT_FALSE(ledger.RestoreCompleted(*peer_restore, kOld, 10));
  ASSERT_TRUE(ledger.RestoreCompleted(*new_restore, kOld, 10));

  const auto accept = [&] {
    auto unit = ledger.BeginUnit();
    EXPECT_TRUE(unit) << llmp::test_support::Failed(unit).value_or("");
    EXPECT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
    EXPECT_TRUE(ledger.BeginVerify(*unit, 2));
    EXPECT_TRUE(ledger.Accept(*unit, 2));
    return ledger.Publication();
  };
  auto old_copy = accept();
  ASSERT_TRUE(old_copy) << llmp::test_support::Failed(old_copy).value_or("");
  ASSERT_TRUE(ledger.PublicationCompleted(old_copy->ticket));
  auto restore = ledger.BeginReplacement();
  ASSERT_TRUE(restore) << llmp::test_support::Failed(restore).value_or("");
  ASSERT_TRUE(ledger.RestoreCompleted(*restore, kOld, 10));
  auto new_copy = accept();
  ASSERT_TRUE(new_copy) << llmp::test_support::Failed(new_copy).value_or("");
  EXPECT_FALSE(ledger.PublicationCompleted(old_copy->ticket));
  EXPECT_TRUE(ledger.owed());
  EXPECT_EQ(ledger.completed().positions, 10U);
  ASSERT_TRUE(ledger.PublicationCompleted(new_copy->ticket));

  auto old_clear = ledger.BeginClear();
  ASSERT_TRUE(old_clear) << llmp::test_support::Failed(old_clear).value_or("");
  ASSERT_TRUE(ledger.ClearCompleted(*old_clear));
  auto new_clear = ledger.BeginClear();
  ASSERT_TRUE(new_clear) << llmp::test_support::Failed(new_clear).value_or("");
  EXPECT_FALSE(ledger.ClearCompleted(*old_clear));
  EXPECT_EQ(ledger.state(), Ledger::State::kClearing);
  ASSERT_TRUE(ledger.ClearCompleted(*new_clear));
}

}  // namespace
