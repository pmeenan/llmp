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

TEST(DraftProgressLedgerTest, CompoundReservesItsPredecessorWithoutPublishingOrLosingItsUnit) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto old = ledger.BeginUnit();
  ASSERT_TRUE(old);
  ASSERT_TRUE(ledger.ReplayCompleted(*old, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*old, 4));
  ASSERT_TRUE(ledger.Accept(*old, 2));
  const auto completed = ledger.completed();
  const auto pending = ledger.after_owed();
  auto prefix = ledger.Publication();
  ASSERT_TRUE(prefix);
  auto compound = ledger.BeginCompound(2);
  ASSERT_TRUE(compound);
  EXPECT_TRUE(ledger.compound_owned());
  EXPECT_TRUE(ledger.CompoundNeedsPrefix(compound->unit));
  EXPECT_TRUE(ledger.CheckCompound(compound->unit));
  EXPECT_EQ(ledger.completed(), completed);
  EXPECT_EQ(compound->shapes.before.count, 1U);
  EXPECT_FALSE(ledger.PreQueued(compound->unit));
  EXPECT_FALSE(ledger.TargetQueued(compound->unit));
  EXPECT_FALSE(ledger.BeginSuccessor());
  ASSERT_TRUE(ledger.CompoundPrefixQueued(compound->unit));
  ASSERT_TRUE(ledger.PreQueued(compound->unit));
  EXPECT_FALSE(ledger.PreCompleted(compound->unit));
  ASSERT_TRUE(ledger.PublicationCompleted(prefix->ticket));
  EXPECT_EQ(ledger.completed(), pending);
  EXPECT_EQ(ledger.state(), Ledger::State::kCompound);
  EXPECT_FALSE(ledger.CompoundNeedsPrefix(compound->unit));
  ASSERT_TRUE(ledger.PreCompleted(compound->unit));
  ASSERT_TRUE(ledger.TargetQueued(compound->unit));
  ASSERT_TRUE(ledger.TargetCompleted(compound->unit));
  ASSERT_TRUE(ledger.PostCompleted(compound->unit));
  ASSERT_TRUE(ledger.FinishCompound(compound->unit));
  EXPECT_EQ(ledger.after_owed().positions, 14U);
  EXPECT_EQ(ledger.completed(), pending);
}

TEST(DraftProgressLedgerTest, CancelingUnusedCompoundKeepsTheExactOldPublication) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit);
  ASSERT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*unit, 4));
  ASSERT_TRUE(ledger.Accept(*unit, 2));
  auto prefix = ledger.Publication();
  ASSERT_TRUE(prefix);
  const auto pending = ledger.after_owed();
  auto compound = ledger.BeginCompound(2);
  ASSERT_TRUE(compound);
  ASSERT_TRUE(ledger.RefuseCompound(compound->unit));
  EXPECT_TRUE(ledger.owed());
  EXPECT_EQ(ledger.after_owed(), pending);
  EXPECT_TRUE(ledger.CheckPublication(prefix->ticket));
  EXPECT_FALSE(ledger.CheckCompound(compound->unit));
  auto retry = ledger.BeginCompound(2);
  ASSERT_TRUE(retry);
  EXPECT_FALSE(ledger.CheckCompound(compound->unit));
  ASSERT_TRUE(ledger.RefuseCompound(retry->unit));
}

TEST(DraftProgressLedgerTest, SubmittedCompoundPrefixCannotBeCanceledAsUnused) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit);
  ASSERT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*unit, 4));
  ASSERT_TRUE(ledger.Accept(*unit, 2));
  const auto completed = ledger.completed();
  auto prefix = ledger.Publication();
  ASSERT_TRUE(prefix);
  auto compound = ledger.BeginCompound(2);
  ASSERT_TRUE(compound);
  ASSERT_TRUE(ledger.CompoundPrefixQueued(compound->unit));
  EXPECT_FALSE(ledger.RefuseCompound(compound->unit));
  EXPECT_EQ(ledger.state(), Ledger::State::kQuarantined);
  EXPECT_FALSE(ledger.PublicationCompleted(prefix->ticket));
  EXPECT_EQ(ledger.completed(), completed);
}

TEST(DraftProgressLedgerTest, FinalTargetTailHasAnExactUnitCursorAndClosingOperation) {
  Ledger ledger(kBounds), foreign(kBounds);
  auto compound = ledger.BeginCompound(1);
  ASSERT_TRUE(compound);
  ASSERT_EQ(compound->shapes.before.count, 0U);
  ASSERT_EQ(compound->shapes.after.count, 0U);
  const auto before = ledger.completed();
  auto tail = ledger.PrepareCompoundTail(compound->unit, Ledger::CompoundClose::kTarget);
  ASSERT_TRUE(tail);
  EXPECT_EQ(ledger.completed(), before);
  EXPECT_FALSE(
      ledger.CheckCompoundTailQueued(compound->unit, tail->ticket, Ledger::CompoundClose::kTarget));
  ASSERT_TRUE(ledger.CheckCompoundTailCursor(compound->unit, tail->ticket, 0));
  EXPECT_FALSE(ledger.CheckCompoundTailCursor(compound->unit, tail->ticket, 1));
  EXPECT_FALSE(foreign.CheckCompoundTailCursor(compound->unit, tail->ticket, 0));
  ASSERT_TRUE(ledger.TargetQueued(compound->unit));
  EXPECT_FALSE(
      ledger.CompleteCompoundTail(compound->unit, tail->ticket, Ledger::CompoundClose::kEarlier));
  ASSERT_TRUE(ledger.CheckCompoundTailRetired(compound->unit, tail->ticket,
                                              Ledger::CompoundClose::kTarget));
  ASSERT_TRUE(
      ledger.CompleteCompoundTail(compound->unit, tail->ticket, Ledger::CompoundClose::kTarget));
  EXPECT_TRUE(ledger.settled());
  EXPECT_EQ(ledger.completed(), (md::DraftProgress{1, 0, 0, 1, 0}));
  EXPECT_FALSE(
      ledger.CompleteCompoundTail(compound->unit, tail->ticket, Ledger::CompoundClose::kTarget));
  auto next = ledger.BeginCompound(1);
  ASSERT_TRUE(next);
  EXPECT_FALSE(ledger.CheckCompoundTailCursor(next->unit, tail->ticket, 1));
}

TEST(DraftProgressLedgerTest, FinalPostTailKeepsTheMirrorUntilItsExactPublicationCompletes) {
  Ledger ledger(kBounds);
  Restore(ledger);
  const auto before = ledger.completed();
  auto compound = ledger.BeginCompound(4);
  ASSERT_TRUE(compound);
  ASSERT_TRUE(ledger.PreQueued(compound->unit));
  ASSERT_TRUE(ledger.PreCompleted(compound->unit));
  EXPECT_FALSE(ledger.PrepareCompoundTail(compound->unit, Ledger::CompoundClose::kPost));
  ASSERT_TRUE(ledger.TargetQueued(compound->unit));
  ASSERT_TRUE(ledger.TargetCompleted(compound->unit));
  auto tail = ledger.PrepareCompoundTail(compound->unit, Ledger::CompoundClose::kPost);
  ASSERT_TRUE(tail);
  EXPECT_EQ(tail->words, (md::DraftProgressWords{1, 14, 13, 10, 4, 1, 1, 0}));
  EXPECT_EQ(ledger.completed(), before);
  EXPECT_FALSE(ledger.PrepareCompoundTail(compound->unit, Ledger::CompoundClose::kPost));
  EXPECT_FALSE(ledger.FinishCompound(compound->unit));
  ASSERT_TRUE(ledger.CheckCompoundTailCursor(compound->unit, tail->ticket, 10));
  EXPECT_FALSE(ledger.CheckCompoundTailCursor(compound->unit, tail->ticket, 11));
  ASSERT_TRUE(
      ledger.CompleteCompoundTail(compound->unit, tail->ticket, Ledger::CompoundClose::kPost));
  EXPECT_TRUE(ledger.settled());
  EXPECT_EQ(ledger.completed(), (md::DraftProgress{14, 13, 10, 4, 1}));
}

TEST(DraftProgressLedgerTest, CancelingAnUnsubmittedTailCannotReuseItsTicket) {
  Ledger ledger(kBounds);
  auto unused = ledger.BeginCompound(1);
  ASSERT_TRUE(unused);
  auto old = ledger.PrepareCompoundTail(unused->unit, Ledger::CompoundClose::kTarget);
  ASSERT_TRUE(old);
  ASSERT_TRUE(ledger.RefuseCompound(unused->unit));
  EXPECT_EQ(ledger.completed(), md::DraftProgress{});
  auto retried = ledger.BeginCompound(1);
  ASSERT_TRUE(retried);
  auto current = ledger.PrepareCompoundTail(retried->unit, Ledger::CompoundClose::kTarget);
  ASSERT_TRUE(current);
  ASSERT_TRUE(ledger.TargetQueued(retried->unit));
  EXPECT_FALSE(
      ledger.CompleteCompoundTail(retried->unit, old->ticket, Ledger::CompoundClose::kTarget));
  ASSERT_TRUE(
      ledger.CompleteCompoundTail(retried->unit, current->ticket, Ledger::CompoundClose::kTarget));
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
  EXPECT_FALSE(ledger.Replacing(*restore));
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
  EXPECT_FALSE(ledger.Replacing(*old_restore));
  EXPECT_TRUE(ledger.Replacing(*new_restore));
  EXPECT_FALSE(ledger.RestoreCompleted(*old_restore, kOld, 10));
  EXPECT_EQ(ledger.state(), Ledger::State::kReplacing);
  auto peer_restore = peer.BeginReplacement();
  ASSERT_TRUE(peer_restore) << llmp::test_support::Failed(peer_restore).value_or("");
  EXPECT_FALSE(ledger.Replacing(*peer_restore));
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

// A successor reserves execution identity while its predecessor stays dirty.
// Neither reservation nor a replay-shaped request is a completion witness.
TEST(DraftProgressLedgerTest, SuccessorKeepsPublicationIdentityAndWaitsForItsExactPrefix) {
  Ledger ledger(kBounds), peer(kBounds);
  Restore(ledger);
  Restore(peer);
  auto accepted = ledger.BeginUnit();
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(ledger.ReplayCompleted(*accepted, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*accepted, 4));
  ASSERT_TRUE(ledger.Accept(*accepted, 2));
  auto publication = ledger.Publication();
  ASSERT_TRUE(publication);
  auto successor = ledger.BeginSuccessor();
  ASSERT_TRUE(successor) << llmp::test_support::Failed(successor).value_or("");
  EXPECT_EQ(successor->replay, (md::DraftRows{10, 11, 10, 2}));
  EXPECT_EQ(ledger.completed().positions, 10U);
  EXPECT_TRUE(ledger.owed());
  EXPECT_FALSE(ledger.settled());
  EXPECT_FALSE(ledger.BeginSuccessor());
  EXPECT_FALSE(ledger.CheckUnit(successor->unit));
  EXPECT_FALSE(ledger.ReplayCompleted(successor->unit, successor->replay));
  EXPECT_FALSE(ledger.BeginVerify(successor->unit, 1));
  EXPECT_FALSE(ledger.BeginReplacement());
  EXPECT_FALSE(ledger.BeginClear());
  EXPECT_TRUE(ledger.CheckSuccessor(successor->unit, publication->ticket, successor->replay));
  auto wrong_rows = successor->replay;
  ++wrong_rows.feature_first;
  EXPECT_FALSE(ledger.CheckSuccessor(successor->unit, publication->ticket, wrong_rows));
  EXPECT_FALSE(peer.CheckSuccessor(successor->unit, publication->ticket, successor->replay));
  auto peer_unit = peer.BeginUnit();
  ASSERT_TRUE(peer_unit);
  ASSERT_TRUE(peer.ReplayCompleted(*peer_unit, {7, 8, 7, 3}));
  ASSERT_TRUE(peer.BeginVerify(*peer_unit, 4));
  ASSERT_TRUE(peer.Accept(*peer_unit, 2));
  auto peer_copy = peer.Publication();
  ASSERT_TRUE(peer_copy);
  EXPECT_FALSE(ledger.CheckSuccessor(successor->unit, peer_copy->ticket, successor->replay));
  EXPECT_TRUE(ledger.CheckPublication(publication->ticket));
  ASSERT_TRUE(ledger.SuccessorQueued(successor->unit, publication->ticket, successor->replay));
  EXPECT_FALSE(ledger.SuccessorQueued(successor->unit, publication->ticket, successor->replay));
  EXPECT_FALSE(ledger.CancelSuccessor(successor->unit));
  ASSERT_TRUE(ledger.PublicationCompleted(publication->ticket));
  EXPECT_TRUE(ledger.CheckUnit(successor->unit));
  EXPECT_FALSE(ledger.CheckUnit(*accepted));
  EXPECT_FALSE(ledger.CheckSuccessor(successor->unit, publication->ticket, successor->replay));
  // Publishing the prefix does not claim that successor replay ran.
  EXPECT_FALSE(ledger.BeginVerify(successor->unit, 1));
  ASSERT_TRUE(ledger.ReplayCompleted(successor->unit, successor->replay));
  EXPECT_TRUE(ledger.BeginVerify(successor->unit, 1));
}

TEST(DraftProgressLedgerTest, CancelingSuccessorLeavesOriginalPublicationAndRevokesItsUnit) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto accepted = ledger.BeginUnit();
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(ledger.ReplayCompleted(*accepted, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*accepted, 4));
  ASSERT_TRUE(ledger.Accept(*accepted, 2));
  auto publication = ledger.Publication();
  ASSERT_TRUE(publication);
  auto canceled = ledger.BeginSuccessor();
  ASSERT_TRUE(canceled);
  ASSERT_TRUE(ledger.CancelSuccessor(canceled->unit));
  EXPECT_TRUE(ledger.owed());
  EXPECT_EQ(ledger.completed().positions, 10U);
  EXPECT_TRUE(ledger.CheckPublication(publication->ticket));
  auto retry = ledger.BeginSuccessor();
  ASSERT_TRUE(retry);
  EXPECT_FALSE(ledger.CheckSuccessor(canceled->unit, publication->ticket, canceled->replay));
  EXPECT_FALSE(ledger.CancelSuccessor(canceled->unit));
  EXPECT_TRUE(ledger.CheckSuccessor(retry->unit, publication->ticket, retry->replay));
  ASSERT_TRUE(ledger.PublicationCompleted(publication->ticket));
  EXPECT_FALSE(ledger.CheckUnit(canceled->unit));
  EXPECT_TRUE(ledger.CheckUnit(retry->unit));
}

TEST(DraftProgressLedgerTest, LatePredecessorCannotPublishSuccessorsLaterAcceptance) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto accepted = ledger.BeginUnit();
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(ledger.ReplayCompleted(*accepted, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*accepted, 4));
  ASSERT_TRUE(ledger.Accept(*accepted, 2));
  auto old = ledger.Publication();
  ASSERT_TRUE(old);
  auto successor = ledger.BeginSuccessor();
  ASSERT_TRUE(successor);
  ASSERT_TRUE(ledger.PublicationCompleted(old->ticket));
  ASSERT_TRUE(ledger.ReplayCompleted(successor->unit, successor->replay));
  ASSERT_TRUE(ledger.BeginVerify(successor->unit, 2));
  ASSERT_TRUE(ledger.Accept(successor->unit, 1));
  auto next = ledger.Publication();
  ASSERT_TRUE(next);
  EXPECT_FALSE(ledger.PublicationCompleted(old->ticket));
  EXPECT_EQ(ledger.completed().positions, 12U);
  EXPECT_TRUE(ledger.owed());
  ASSERT_TRUE(ledger.PublicationCompleted(next->ticket));
  EXPECT_EQ(ledger.completed().positions, 13U);
}

TEST(DraftProgressLedgerTest, UnprovenSuccessorCannotBeRevivedByLatePrefixOrRecovery) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto accepted = ledger.BeginUnit();
  ASSERT_TRUE(accepted);
  ASSERT_TRUE(ledger.ReplayCompleted(*accepted, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*accepted, 4));
  ASSERT_TRUE(ledger.Accept(*accepted, 2));
  auto publication = ledger.Publication();
  ASSERT_TRUE(publication);
  auto successor = ledger.BeginSuccessor();
  ASSERT_TRUE(successor);
  ledger.Quarantine();
  EXPECT_EQ(ledger.completed().positions, 10U);
  EXPECT_FALSE(ledger.PublicationCompleted(publication->ticket));
  EXPECT_FALSE(ledger.CheckSuccessor(successor->unit, publication->ticket, successor->replay));
  EXPECT_FALSE(ledger.CheckUnit(successor->unit));
  auto clear = ledger.BeginClear();
  ASSERT_TRUE(clear);
  ASSERT_TRUE(ledger.ClearCompleted(*clear));
  Restore(ledger);
  auto fresh = ledger.BeginUnit();
  ASSERT_TRUE(fresh);
  EXPECT_FALSE(ledger.CheckUnit(successor->unit));
  EXPECT_FALSE(ledger.CheckPublication(publication->ticket));
  EXPECT_TRUE(ledger.CheckUnit(*fresh));
}

TEST(DraftProgressLedgerTest, CompoundPreviewGrantsNoUnitAndMatchesTheAuthoritativeMint) {
  Ledger ledger(kBounds);
  Restore(ledger);
  const auto completed = ledger.completed();
  auto first = ledger.PreviewCompound(4);
  auto repeated = ledger.PreviewCompound(4);
  ASSERT_TRUE(first);
  ASSERT_TRUE(repeated);
  EXPECT_EQ(first->before, repeated->before);
  EXPECT_EQ(first->after, repeated->after);
  EXPECT_TRUE(ledger.settled());
  EXPECT_EQ(ledger.completed(), completed);
  EXPECT_FALSE(ledger.compound_owned());
  EXPECT_FALSE(ledger.owed());
  auto compound = ledger.BeginCompound(4);
  ASSERT_TRUE(compound);
  EXPECT_EQ(compound->shapes.before, first->before);
  EXPECT_EQ(compound->shapes.after, first->after);
  EXPECT_EQ(compound->shapes.final_pending_first, first->final_pending_first);
  EXPECT_FALSE(ledger.PreviewCompound(4));
  EXPECT_FALSE(ledger.PreviewCompound(0));
  ASSERT_TRUE(ledger.RefuseCompound(compound->unit));
  EXPECT_TRUE(ledger.settled());
}

TEST(DraftProgressLedgerTest, CompoundPreviewKeepsItsOwedPredecessorAndRevalidatesAtMint) {
  Ledger ledger(kBounds);
  Restore(ledger);
  auto unit = ledger.BeginUnit();
  ASSERT_TRUE(unit);
  ASSERT_TRUE(ledger.ReplayCompleted(*unit, {7, 8, 7, 3}));
  ASSERT_TRUE(ledger.BeginVerify(*unit, 4));
  ASSERT_TRUE(ledger.Accept(*unit, 2));
  const auto prefix = ledger.Publication();
  ASSERT_TRUE(prefix);
  const auto completed = ledger.completed();
  const auto pending = ledger.after_owed();
  auto preview = ledger.PreviewCompound(2);
  ASSERT_TRUE(preview);
  EXPECT_TRUE(ledger.CheckPublication(prefix->ticket));
  EXPECT_EQ(ledger.completed(), completed);
  EXPECT_EQ(ledger.after_owed(), pending);
  EXPECT_FALSE(ledger.compound_owned());
  // A real competing owner invalidates the optional preview; prediction alone
  // never reserves a transition or authorizes a future batch's execution.
  auto successor = ledger.BeginSuccessor();
  ASSERT_TRUE(successor);
  EXPECT_FALSE(ledger.PreviewCompound(2));
  EXPECT_FALSE(ledger.BeginCompound(2));
}

}  // namespace
