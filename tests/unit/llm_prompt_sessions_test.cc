// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "llm_fixture.h"
#include "providers/fake/fake_device_runtime.h"
#include "scheduler/cohort_policy.h"
#include "scheduler/cohort_schedule.h"

namespace {
using namespace llmp::test_support::llm_fixture;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using Session = rt::Llm::PromptSession;
using Phase = Session::Phase;

class LlmPromptSessions : public LlmFixture {
 protected:
  void StartPrompts(std::uint32_t context = 64, std::uint32_t chunk = 8, std::uint32_t slots = 4,
                    bool drafters = false) {
    context_rows_ = context;
    chunk_rows_ = chunk;
    slot_count_ = slots;
    equal_prefill_ = true;
    Start(drafters, false);
    std::vector<rt::Llm::Branch*> active;
    for (std::size_t i = 0; i < slots; ++i) active.push_back(&BranchAt(i));
    Open(std::move(active));
    full_budget_ = node_->budget();
  }
  void Request(std::initializer_list<std::size_t> slots) {
    if (node_->InRequest(0)) Close();
    std::vector<rt::Llm::Branch*> active;
    for (const auto slot : slots) active.push_back(&BranchAt(slot));
    Open(std::move(active));
  }
  std::vector<std::int32_t> State(std::size_t slot) {
    const auto& slots = llm_->runner().slots();
    std::vector<std::int32_t> tokens;
    for (std::uint32_t i = 0; i < slots.positions(static_cast<std::uint32_t>(slot)); ++i) {
      ts::FakeCell cell;
      std::memcpy(&cell,
                  reinterpret_cast<const void*>(
                      slots.live(static_cast<std::uint32_t>(slot), 0).base(0) + i * cell_bytes_),
                  sizeof(cell));
      EXPECT_EQ(cell.magic, ts::kFakeMagic);
      EXPECT_EQ(cell.position, i);
      EXPECT_EQ(cell.slot, slot);
      tokens.push_back(cell.token);
    }
    return tokens;
  }
  void CheckInjection(std::size_t slot, std::span<const std::int32_t> tokens) {
    const auto& slots = llm_->runner().slots();
    ASSERT_GT(slots.stages(), 1U);
    const auto& item = slots.layout(1).items.at(1);
    const auto first = tokens.size() > item.cells ? tokens.size() - item.cells : 0;
    for (std::size_t i = first; i < tokens.size(); ++i) {
      ts::FakeCell cell;
      const auto address = slots.live(static_cast<std::uint32_t>(slot), 1).base(0) + item.offset +
                           (i % item.cells) * item.contract.cell_bytes;
      std::memcpy(&cell, reinterpret_cast<const void*>(address), sizeof(cell));
      EXPECT_EQ(cell.magic, ts::kFakeMagic);
      EXPECT_EQ(cell.position, i);
      EXPECT_EQ(cell.slot, slot);
      EXPECT_EQ(cell.token, tokens[i]);
    }
  }
  std::uint64_t Operations() {
    auto stats = node_->Stats();
    EXPECT_TRUE(stats);
    return stats ? stats->held_operations : 0;
  }
  std::vector<float> Row(std::int32_t token) {
    std::vector<float> row(kWidth, 0.0F);
    row.at(static_cast<std::size_t>(token)) = 1.0F;
    return row;
  }
  std::vector<float> Head(std::span<const std::int32_t> tokens) {
    return Row(ts::FakeLogit(tokens.back(), static_cast<std::int32_t>(tokens.size() - 1), kWidth));
  }
  void Finish(Session& session) {
    while (!session.done()) ASSERT_TRUE(session.Advance()) << Why(session.last_unit_result());
    ASSERT_TRUE(session.Finish());
  }
  void Cancel(Session& session) {
    session.Cancel();
    ASSERT_TRUE(session.Finish());
  }
  void Recover() {
    source_->options().fail_run.reset();
    if (node_->InRequest(0)) std::ignore = node_->EndRequest(0);
    ASSERT_TRUE(node_->Job(
        llmp::catalog::Closure{},
        [](llmp::providers::NativeStream) { return llmp::scheduler::JobResult::kQueued; },
        "fencing a faulted prompt stream", 0));
    ASSERT_TRUE(llm_->RecoverInPlace());
  }
  void FailRun() { source_->options().fail_run = kn::StepError::kUnknown; }
  std::uint64_t Occupancy() {
    auto stats = node_->Stats();
    EXPECT_TRUE(stats);
    return stats ? stats->catalog_occupancy_bytes : 0;
  }
  // Real state grows at the native 256-position read alignment. An 8 KiB
  // cell makes each such interval one 2 MiB extent. No fake capacity flag
  // bypasses state admission; the scheduler's actual physical budget decides.
  void StartCapacity(std::uint32_t chunk = 256, bool drafters = false) {
    cell_bytes_ = kExtent / 256;
    prefill_lookahead_ =
        0;  // pressure applies to the next actual state growth, not prepared futures.
    StartPrompts(8192, chunk, 4, drafters);
  }
  void Room(std::uint64_t extra_extents = 0) {
    const auto bytes = Occupancy() + extra_extents * kExtent + kExtent / 2;
    ASSERT_TRUE(node_->Call(
        [&]() -> en::Status {
          (void)node_->ApplyBudget(Bytes(bytes));
          return {};
        },
        "setting a test's physical execution budget"));
  }
  void StateBudget(std::uint64_t extents) {
    std::uint64_t resident = 0;
    for (std::size_t slot = 0; slot < llm_->branches(); ++slot)
      resident += llm_->ResidentStateBytes(BranchAt(slot));
    const auto bytes = Occupancy() - resident + extents * kExtent + kExtent / 2;
    ASSERT_TRUE(node_->Call(
        [&]() -> en::Status {
          (void)node_->ApplyBudget(Bytes(bytes));
          return {};
        },
        "bounding actual resident state capacity"));
  }
  void Unlimit() {
    ASSERT_TRUE(node_->Call(
        [&]() -> en::Status {
          (void)node_->ApplyBudget(full_budget_);
          return {};
        },
        "restoring the test's execution budget"));
  }
  void Hold(std::size_t slot, std::size_t tokens) {
    Request({slot});
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(slot).Prefill(std::vector<std::int32_t>(tokens, 1), last));
    Close();
  }
  std::optional<std::size_t> LargestIdle(const rt::Llm::Branch& keep) {
    std::optional<std::size_t> chosen;
    std::uint64_t most = 0;
    for (std::size_t slot = 0; slot < llm_->branches(); ++slot) {
      auto& branch = BranchAt(slot);
      if (&branch == &keep || !llm_->BranchIdle(branch)) continue;
      const auto bytes = llm_->ResidentStateBytes(branch);
      if (bytes > most) {
        most = bytes;
        chosen = slot;
      }
    }
    return chosen;
  }
  void ReclaimLargest(std::vector<std::size_t>& released, bool spill = false) {
    llm_->set_capacity_reclaim([this, &released, spill](const rt::Llm::Branch& keep) {
      const auto chosen = LargestIdle(keep);
      if (!chosen) return false;
      auto& branch = BranchAt(*chosen);
      auto result = spill ? llm_->SpillIdle(branch) : branch.ReleaseIdleState();
      if (!result) return false;
      released.push_back(*chosen);
      return true;
    });
  }
  bool Built(std::uint32_t slot, std::uint32_t past, std::uint32_t rows, bool head,
             std::optional<bool> greedy = {}) const {
    for (const auto& built : source_->built()) {
      if (built.measuring) continue;
      for (const auto& segment : built.shape.segments) {
        if (segment.slot == slot && segment.past == past && segment.rows == rows &&
            segment.want_head == head && (!greedy || segment.greedy == *greedy))
          return true;
      }
    }
    return false;
  }
  Bytes full_budget_{};
};

TEST_F(LlmPromptSessions, ResumablePromptAdmissionIsHostOnlyAndOwnsItsPrompt) {
  StartPrompts(64, 8, 4, true);
  const auto queued = source_->queued();
  const auto operations = Operations();
  std::vector<std::int32_t> prompt(19, 2);
  auto opened = BranchAt(0).BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  auto& session = **opened;
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_EQ(Operations(), operations);
  EXPECT_TRUE(BranchAt(0).history().empty());
  ASSERT_TRUE(session.NextUnit());
  EXPECT_EQ(session.NextUnit()->phase, Phase::kReuse);
  EXPECT_EQ(session.NextUnit()->rows, 0U);
  EXPECT_FALSE(BranchAt(0).BeginPrompt(prompt));
  rt::GenerateOptions options;
  rt::Generation generation;
  EXPECT_FALSE(BranchAt(0).BeginGeneration(Row(1), options, generation));
  EXPECT_FALSE(session.Finish());
  std::ranges::fill(prompt, 3);
  std::vector<std::uint32_t> units;
  while (!session.done()) {
    const auto unit = session.NextUnit();
    ASSERT_TRUE(unit);
    units.push_back(unit->rows);
    ASSERT_TRUE(session.Advance());
  }
  ASSERT_TRUE(session.Finish());
  EXPECT_THAT(units, ElementsAre(0, 8, 8, 3));
  EXPECT_EQ(BranchAt(0).history(), std::vector<std::int32_t>(19, 2));
  EXPECT_EQ(State(0), BranchAt(0).history());
  EXPECT_EQ(session.run().end, 19U);
  EXPECT_EQ(session.run().chunks, 3U);
  CheckInjection(0, BranchAt(0).history());
  EXPECT_EQ(session.last(), Head(BranchAt(0).history()));
}

TEST_F(LlmPromptSessions, ResumablePromptCancelsAtACompletePrefixAndCanResume) {
  StartPrompts();
  std::vector<float> row;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 2>{0, 1}, row));
  std::vector<std::int32_t> prompt{0, 1};
  prompt.resize(19, 2);
  auto opened = BranchAt(0).BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  auto& session = **opened;
  ASSERT_TRUE(session.Advance());
  EXPECT_EQ(session.reused(), 2U);
  ASSERT_TRUE(session.Advance());
  EXPECT_EQ(State(0).size(), 10U);
  Cancel(session);
  EXPECT_TRUE(session.run().stopped);
  EXPECT_EQ(session.run().end, 10U);
  EXPECT_EQ(session.run().chunks, 1U);
  EXPECT_TRUE(session.last().empty());
  EXPECT_EQ(State(0), BranchAt(0).history());
  std::uint32_t reused = 0;
  rt::PrefillRun finished;
  ASSERT_TRUE(BranchAt(0).PreparePrompt(prompt, 0, row, reused, {}, &finished));
  EXPECT_EQ(reused, 10U);
  EXPECT_EQ(State(0), prompt);
  EXPECT_EQ(BranchAt(0).history(), prompt);
  EXPECT_EQ(finished.chunks, 2U);
}

TEST_F(LlmPromptSessions, JoinedHintsPreserveOwnerOrderAndIndependentlySuppressMixedFutureHeads) {
  prefill_lookahead_ = 3;  // two separate mixed-head candidates, then the homogeneous later head
  StartPrompts();
  auto x = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(16, 2));
  auto y = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(24, 3));
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  const auto first = rt::LlmTestAccess::Hint(**x, 8);
  const auto second = rt::LlmTestAccess::Hint(**y, 8);
  EXPECT_EQ(first.rows, 8U);
  EXPECT_TRUE(first.want_head);
  EXPECT_EQ(first.after_rows, 0U);
  EXPECT_EQ(second.rows, 8U);
  EXPECT_FALSE(second.want_head);
  EXPECT_EQ(second.after_rows, 8U);
  EXPECT_TRUE(second.after_want_head);
  std::array<Session*, 2> sessions{y->get(), x->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  ASSERT_TRUE(llm_->RunPromptWave(sessions, callbacks, true));
  EXPECT_EQ(State(0), std::vector<std::int32_t>(8, 2));
  EXPECT_EQ(State(1), std::vector<std::int32_t>(8, 3));
  EXPECT_EQ(BranchAt(0).history(), State(0));
  EXPECT_EQ(BranchAt(1).history(), State(1));
  EXPECT_TRUE(Built(0, 0, 8, false));
  EXPECT_TRUE(Built(1, 0, 8, false));
  // The homogeneous future head is admitted independently of the mixed one.
  EXPECT_TRUE(Built(1, 16, 8, true));
  Cancel(**x);
  Cancel(**y);
}

TEST_F(LlmPromptSessions, JoinedHintsStopAtEachOwnersCheckpointBoundary) {
  StartPrompts();
  const std::vector<std::int32_t> prompt(24, 2);
  auto x = BranchAt(0).BeginPrompt(prompt, 8), y = BranchAt(1).BeginPrompt(prompt, 16);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  EXPECT_EQ(rt::LlmTestAccess::Hint(**x, 8).rows, 0U);
  EXPECT_EQ(rt::LlmTestAccess::Hint(**x, 8).after_rows, 0U);
  EXPECT_EQ(rt::LlmTestAccess::Hint(**y, 8).rows, 8U);
  EXPECT_FALSE(rt::LlmTestAccess::Hint(**y, 8).want_head);
  EXPECT_EQ(rt::LlmTestAccess::Hint(**y, 8).after_rows, 0U);
  std::array<Session*, 2> sessions{x->get(), y->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  ASSERT_TRUE(llm_->RunPromptWave(sessions, callbacks, true));
  EXPECT_EQ((*x)->NextUnit()->phase, Phase::kCheckpoint);
  EXPECT_FALSE(Built(0, 8, 8, false));
  EXPECT_TRUE(Built(1, 8, 8, false));
  EXPECT_FALSE(Built(1, 16, 8, true));
  Cancel(**x);
  Cancel(**y);
}

TEST_F(LlmPromptSessions, PromptWaveReadinessSettlesColdPeerWithoutReplacingDueDecode) {
  StartPrompts();
  auto x = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(19, 2));
  auto y = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(27, 3));
  ASSERT_TRUE(x && y);
  std::array<Session*, 3> sessions{x->get(), y->get(), nullptr};
  std::array<llmp::scheduler::ScheduledMember, 3> members{
      llmp::scheduler::ScheduledMember{.admitted = 1, .remaining = 19},
      llmp::scheduler::ScheduledMember{.admitted = 2, .remaining = 27},
      llmp::scheduler::ScheduledMember{
          .stage = llmp::scheduler::ScheduledMember::Stage::kGeneration,
          .admitted = 3,
          .waited = 1}};
  const auto choose = [&] {
    const auto choice = llmp::scheduler::NextCohortUnit(members);
    return Session::SelectForWave(choice.decode, choice.prompt, sessions);
  };
  EXPECT_FALSE(choose());
  members[2].waited = 0;
  ASSERT_EQ(choose(), 0U);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_EQ((*x)->NextUnit()->phase, Phase::kChunk);
  ASSERT_EQ(choose(), 1U);
  EXPECT_EQ((*x)->run().end, 0U);
  ASSERT_TRUE((*y)->Advance());
  EXPECT_TRUE((*x)->CanJoin(**y));
  ASSERT_EQ(choose(), 0U);
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  ASSERT_TRUE(llm_->RunPromptWave(std::span(sessions).first(2), callbacks, true));
  EXPECT_EQ((*x)->run().end, 8U);
  EXPECT_EQ((*y)->run().end, 8U);
  Cancel(**x);
  Cancel(**y);
  x->reset();
  y->reset();
  Stop();
  wave_capacity_ = 1;
  StartPrompts();
  x = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(19, 2));
  y = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(27, 3));
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  sessions = {x->get(), y->get(), nullptr};
  EXPECT_EQ(choose(), 0U);
  Cancel(**x);
  Cancel(**y);
}

TEST_F(LlmPromptSessions, PromptWaveReadinessDoesNotBorrowAScorerOrForeignPeer) {
  StartPrompts();
  en::PagedNode foreign_node(en::NodeSettings{});
  rt::Llm foreign(foreign_node, entry_, Settings(false), roles_, 0, {});
  const std::vector<std::int32_t> prompt(19, 2);
  auto x = BranchAt(0).BeginPrompt(prompt);
  auto y = BranchAt(1).BeginScoringPrompt(prompt, {});
  auto z = foreign.default_branch().BeginPrompt(prompt);
  ASSERT_TRUE(x && y && z);
  ASSERT_TRUE((*x)->Advance());
  std::array<Session*, 2> sessions{x->get(), y->get()};
  EXPECT_EQ(Session::SelectForWave(false, 0, sessions), 0U);
  sessions[1] = z->get();
  EXPECT_EQ(Session::SelectForWave(false, 0, sessions), 0U);
  Cancel(**x);
  Cancel(**y);
  Cancel(**z);
}

TEST_F(LlmPromptSessions, PromptWaveExcludesDifferentFinalRowCounts) {
  StartPrompts();
  auto x = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(3, 2));
  auto y = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(5, 3));
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  std::array<Session*, 2> sessions{x->get(), y->get()};
  EXPECT_EQ(Session::SelectForWave(false, 0, sessions), 0U);
  ASSERT_TRUE((*y)->Advance());
  EXPECT_TRUE((*x)->NextUnit()->want_head);
  EXPECT_TRUE((*y)->NextUnit()->want_head);
  EXPECT_FALSE((*x)->CanJoin(**y));
  const auto queued = source_->queued();
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  EXPECT_FALSE(llm_->RunPromptWave(sessions, callbacks, true));
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_EQ((*x)->run().end, 0U);
  EXPECT_EQ((*y)->run().end, 0U);
  Cancel(**x);
  Cancel(**y);
}

TEST_F(LlmPromptSessions, OneRowCheckpointPrefillCanJoinAndSingletonRemainsUsable) {
  StartPrompts();
  const std::vector<std::int32_t> prompt(19, 2);
  auto x = BranchAt(0).BeginPrompt(prompt, 1), y = BranchAt(1).BeginPrompt(prompt, 1);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  EXPECT_EQ((*x)->NextUnit()->rows, 1U);
  EXPECT_FALSE((*x)->NextUnit()->want_head);
  EXPECT_TRUE((*x)->CanJoin(**y));
  std::array<Session*, 1> singleton{x->get()};
  const std::array<rt::PrefillGoOn, 1> callbacks{};
  ASSERT_TRUE(llm_->RunPromptWave(singleton, callbacks, true));
  EXPECT_EQ((*x)->run().end, 1U);
  EXPECT_EQ((*x)->NextUnit()->phase, Phase::kCheckpoint);
  EXPECT_EQ((*y)->run().end, 0U);
  EXPECT_FALSE((*x)->CanJoin(**y));  // x now owes its checkpoint before another chunk
  Cancel(**x);
  Cancel(**y);
}

TEST_F(LlmPromptSessions, CompatiblePromptWavePublishesPeersOnlyAfterSharedCompletion) {
  StartPrompts();
  const std::vector<std::int32_t> a(19, 2), b(19, 3);
  auto x = BranchAt(0).BeginPrompt(a), y = BranchAt(1).BeginPrompt(b);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  ASSERT_TRUE((*x)->CanJoin(**y));
  std::array<Session*, 2> sessions{x->get(), y->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  std::size_t before = 0, completions = 0;
  source_->options().after_execute = [&](pl::Phase phase) {
    if (phase != pl::Phase::kPrefill) return;
    // Called after the actual native writes, before completion publication.
    EXPECT_EQ(BranchAt(0).history().size(), before);
    EXPECT_EQ(BranchAt(1).history().size(), before);
    ++completions;
  };
  for (std::uint32_t at : {8U, 16U, 19U}) {
    ASSERT_TRUE(llm_->RunPromptWave(sessions, callbacks, true));
    EXPECT_EQ((*x)->run().end, at);
    EXPECT_EQ((*y)->run().end, at);
    EXPECT_EQ(State(0), BranchAt(0).history());
    EXPECT_EQ(State(1), BranchAt(1).history());
    EXPECT_TRUE((*x)->last_unit_result());
    EXPECT_TRUE((*y)->last_unit_result());
    before = at;
  }
  source_->options().after_execute = {};
  ASSERT_TRUE((*x)->Finish());
  ASSERT_TRUE((*y)->Finish());
  EXPECT_EQ(BranchAt(0).history(), a);
  EXPECT_EQ(BranchAt(1).history(), b);
  EXPECT_EQ(completions, 3U);
}

TEST_F(LlmPromptSessions, PromptWaveOmitsCancelledAndCapacityRefusedPeersBeforeDispatch) {
  for (bool cancelled : {false, true}) {
    StartCapacity();
    const std::vector<std::int32_t> prompt(515, 2);
    auto x = BranchAt(0).BeginPrompt(prompt), y = BranchAt(1).BeginPrompt(prompt);
    ASSERT_TRUE(x && y);
    ASSERT_TRUE((*x)->Advance());
    ASSERT_TRUE((*y)->Advance());
    if (!cancelled) Room(1);  // one owner's first state extent, not both
    std::array<Session*, 2> sessions{x->get(), y->get()};
    const std::array<rt::PrefillGoOn, 2> callbacks{
        rt::PrefillGoOn{}, [cancelled](std::uint32_t) { return !cancelled; }};
    ASSERT_TRUE(llm_->RunPromptWave(sessions, callbacks, true));
    EXPECT_EQ(BranchAt(0).history(), std::vector<std::int32_t>(256, 2));
    EXPECT_TRUE(BranchAt(1).history().empty());
    EXPECT_EQ(llm_->runner().slots().positions(1), 0U);
    EXPECT_EQ((*y)->refused(), !cancelled);
    EXPECT_EQ((*y)->done(), cancelled);
    EXPECT_EQ((*y)->last_unit_result().has_value(), cancelled);
    if (!cancelled) {
      Unlimit();
      ASSERT_TRUE((*y)->Advance());
      EXPECT_EQ(BranchAt(1).history(), BranchAt(0).history());
    }
    Cancel(**x);
    Cancel(**y);
    Stop();
  }
}

TEST_F(LlmPromptSessions, SharedPromptFailurePublishesNoPeerHistoryAndRequiresFreshClear) {
  StartPrompts();
  const std::vector<std::int32_t> prompt(19, 2);
  auto x = BranchAt(0).BeginPrompt(prompt), y = BranchAt(1).BeginPrompt(prompt);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  ASSERT_TRUE((*y)->Advance());
  FailRun();
  std::array<Session*, 2> sessions{x->get(), y->get()};
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  EXPECT_TRUE(llm_->RunPromptWave(sessions, callbacks, true));
  EXPECT_TRUE(BranchAt(0).history().empty());
  EXPECT_TRUE(BranchAt(1).history().empty());
  EXPECT_FALSE((*x)->Finish());
  EXPECT_FALSE((*y)->Finish());
  Recover();
  Request({0, 1});
  auto fresh = BranchAt(0).BeginPrompt(prompt);
  ASSERT_TRUE(fresh);
  ASSERT_TRUE((*fresh)->Advance());
  EXPECT_EQ(llm_->runner().slots().positions(0), 0U);
  Finish(**fresh);
  EXPECT_EQ(State(0), prompt);
}

TEST_F(LlmPromptSessions, PromptWaveRefusesMixedModesForeignDuplicatesAndCheckpointUnits) {
  StartPrompts();
  en::PagedNode foreign_node(en::NodeSettings{});
  rt::Llm foreign(foreign_node, entry_, Settings(false), roles_, 0, {});
  auto x = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(7, 2));
  auto y = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(19, 3), 9);
  ASSERT_TRUE(x && y);
  ASSERT_TRUE((*x)->Advance());
  std::array<Session*, 2> sessions{x->get(), y->get()};
  EXPECT_EQ(Session::SelectForWave(false, 0, sessions), 0U);
  ASSERT_TRUE((*y)->Advance());
  const auto queued = source_->queued();
  const std::array<rt::PrefillGoOn, 2> callbacks{};
  EXPECT_FALSE((*x)->CanJoin(**y));
  EXPECT_FALSE(llm_->RunPromptWave(sessions, callbacks, true));
  sessions[1] = x->get();
  EXPECT_FALSE(llm_->RunPromptWave(sessions, callbacks, true));
  sessions[1] = nullptr;
  EXPECT_FALSE(llm_->RunPromptWave(sessions, callbacks, true));
  sessions[1] = y->get();
  EXPECT_FALSE(foreign.RunPromptWave(sessions, callbacks, true));
  EXPECT_EQ(source_->queued(), queued);
  EXPECT_TRUE(BranchAt(0).history().empty());
  EXPECT_TRUE(BranchAt(1).history().empty());
  ASSERT_TRUE((*y)->Advance());
  ASSERT_TRUE((*y)->Advance());
  ASSERT_EQ((*y)->NextUnit()->phase, Phase::kCheckpoint);
  EXPECT_EQ(Session::SelectForWave(false, 0, sessions), 0U);
  EXPECT_EQ(Session::SelectForWave(false, 1, sessions), 1U);
  const auto checkpoint_queued = source_->queued();
  EXPECT_FALSE(llm_->RunPromptWave(sessions, callbacks, true));
  EXPECT_EQ(source_->queued(), checkpoint_queued);
  Cancel(**x);
  Cancel(**y);
}

TEST_F(LlmPromptSessions, APausedPartialPromptKeepsItsExactSpillPastIdleRetention) {
  StartPrompts();
  auto& branch = BranchAt(1);
  const std::vector<std::int32_t> prompt(19, 2);
  auto opened = branch.BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  ASSERT_TRUE((*opened)->Advance());
  ASSERT_TRUE((*opened)->Advance());
  ASSERT_EQ(branch.history().size(), 8U);
  Cancel(**opened);
  branch.HoldContinuation();
  ASSERT_TRUE(llm_->SpillSetAside(branch));
  Close();
  llm_->set_retention(std::chrono::seconds(0));
  EXPECT_TRUE(llm_->BranchIdle(branch));
  EXPECT_FALSE(branch.ReleaseIdleState());
  const auto executed = source_->executed();
  Request({1});
  auto continued = branch.BeginPrompt(prompt);
  ASSERT_TRUE(continued);
  Finish(**continued);
  EXPECT_EQ((*continued)->reused(), 8U);
  EXPECT_EQ(llm_->spill_stats().restores, 1U);
  EXPECT_EQ(source_->executed() - executed, 2U);
  EXPECT_EQ(branch.history(), prompt);
  EXPECT_EQ(State(1), prompt);
  branch.ReleaseContinuation();
  Close();
  EXPECT_TRUE(llm_->BranchIdle(branch));
}

TEST_F(LlmPromptSessions, ResumablePromptKeepsTheTiledTailAndTurnBoundarySeparate) {
  StartPrompts(8192, 4096);
  const std::vector<std::int32_t> prompt(2051, 1);
  auto opened = BranchAt(0).BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  auto& session = **opened;
  ASSERT_TRUE(session.Advance());
  ASSERT_TRUE(session.NextUnit());
  EXPECT_EQ(session.NextUnit()->rows, 2048U);
  ASSERT_TRUE(session.Advance());
  EXPECT_EQ(session.NextUnit()->rows, 3U);
  ASSERT_TRUE(session.Advance());
  ASSERT_TRUE(session.Finish());
  EXPECT_EQ(BranchAt(0).history(), prompt);
  opened->reset();
  Stop();
  StartPrompts();
  auto boundary = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(17, 2), 9);
  ASSERT_TRUE(boundary);
  ASSERT_TRUE((*boundary)->Advance());
  ASSERT_TRUE((*boundary)->Advance());
  EXPECT_EQ((*boundary)->NextUnit()->rows, 1U);
  ASSERT_TRUE((*boundary)->Advance());
  EXPECT_EQ((*boundary)->NextUnit()->phase, Phase::kCheckpoint);
  EXPECT_EQ((*boundary)->NextUnit()->rows, 0U);
  Cancel(**boundary);
  EXPECT_EQ(BranchAt(0).history(), std::vector<std::int32_t>(9, 2));
  EXPECT_EQ(State(0), BranchAt(0).history());
  EXPECT_EQ(BranchAt(0).turn_checkpoints(), 0U);
}

TEST_F(LlmPromptSessions, ResumingAtATurnCheckpointMustPrefillBeforeAnotherModelSwitch) {
  StartPrompts();
  using Policy = llmp::scheduler::CohortPolicy;
  const Policy::Clock::time_point start{};
  const std::vector<std::int32_t> prompt(17, 2);
  auto first = BranchAt(0).BeginPrompt(prompt, 8);
  ASSERT_TRUE(first);
  ASSERT_TRUE((*first)->Advance());
  ASSERT_TRUE((*first)->Advance());
  ASSERT_EQ((*first)->NextUnit()->phase, Phase::kCheckpoint);
  Cancel(**first);
  first->reset();
  ASSERT_EQ(BranchAt(0).history().size(), 8U);
  auto resumed = BranchAt(0).BeginPrompt(prompt, 8);
  ASSERT_TRUE(resumed);
  Policy policy(std::chrono::seconds(30), start);
  policy.BeginPass(Policy::Head::kOtherModel);
  for (const auto phase : {Phase::kReuse, Phase::kCheckpoint, Phase::kChunk}) {
    const auto unit = (*resumed)->NextUnit();
    ASSERT_TRUE(unit);
    ASSERT_EQ(unit->phase, phase);
    ASSERT_TRUE((*resumed)->Advance());
    policy.UnitCompleted(start + std::chrono::seconds(60), unit->advances_request()
                                                               ? Policy::Unit::kProgress
                                                               : Policy::Unit::kReadiness);
    EXPECT_EQ(policy.YieldForSwitch(start + std::chrono::seconds(60)), phase == Phase::kChunk);
    EXPECT_EQ(BranchAt(0).history().size(), phase == Phase::kChunk ? 16U : 8U);
  }
  Cancel(**resumed);
}

TEST_F(LlmPromptSessions, ResumablePromptFailuresDoNotPublishAnUnprovenChunk) {
  StartPrompts();
  auto opened = BranchAt(0).BeginPrompt(std::vector<std::int32_t>(19, 2));
  ASSERT_TRUE(opened);
  auto& session = **opened;
  ASSERT_TRUE(session.Advance());
  ASSERT_TRUE(session.Advance());
  FailRun();
  EXPECT_FALSE(session.Advance());
  EXPECT_TRUE(session.done());
  EXPECT_FALSE(session.Finish());
  EXPECT_EQ(session.run().end, 8U);
  EXPECT_EQ(session.run().chunks, 1U);
  EXPECT_TRUE(session.last().empty());
  EXPECT_TRUE(BranchAt(0).history().empty());
  EXPECT_FALSE(BranchAt(0).HasRetainedState());
  Recover();
  Request({0});
  std::vector<float> row;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{3}, row));
  EXPECT_EQ(State(0), BranchAt(0).history());
  EXPECT_THAT(BranchAt(0).history(), ElementsAre(3));
}

TEST_F(LlmPromptSessions, ResumablePromptKeepsARequiredClearAfterAProvenRefusal) {
  StartPrompts();
  std::vector<float> row;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 2>{0, 1}, row));
  BranchAt(0).Forget();
  const auto before = NativeImage(0);
  const auto queued = source_->queued();
  execution_identity_ = std::unexpected("source temporarily cannot authenticate execution");
  auto opened = BranchAt(0).BeginPrompt(std::array<std::int32_t, 1>{3});
  ASSERT_TRUE(opened);
  EXPECT_FALSE((*opened)->Advance());
  EXPECT_FALSE((*opened)->Finish());
  EXPECT_TRUE(llm_->runner().slots().usable(0));
  EXPECT_TRUE(BranchAt(0).history().empty());
  EXPECT_EQ(NativeImage(0), before);
  EXPECT_EQ(source_->queued(), queued);
  execution_identity_ = std::nullopt;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{3}, row));
  EXPECT_EQ(State(0), BranchAt(0).history());
  EXPECT_THAT(BranchAt(0).history(), ElementsAre(3));
}

TEST_F(LlmPromptSessions, AResumablePromptDoesNotBlockOrCancelItsPeerGeneration) {
  StartPrompts();
  std::vector<float> last;
  const std::array<std::int32_t, 1> prefix{0};
  ASSERT_TRUE(BranchAt(0).Prefill(prefix, last));
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  rt::Generation out;
  auto generation = BranchAt(0).BeginGeneration(last, options, out);
  auto prefill = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(19, 2));
  ASSERT_TRUE(generation && prefill);
  ASSERT_TRUE((*prefill)->Advance());
  ASSERT_TRUE((*generation)->RunScalarStep());
  ASSERT_TRUE((*prefill)->Advance());
  Cancel(**prefill);
  ASSERT_TRUE((*generation)->RunScalarStep());
  ASSERT_TRUE((*generation)->Finish());
  EXPECT_EQ(out.tokens, Chain(prefix, 3));
  EXPECT_FALSE(out.cancelled);
  EXPECT_EQ(BranchAt(0).history(), State(0));
  EXPECT_EQ(BranchAt(1).history(), std::vector<std::int32_t>(8, 2));
  EXPECT_EQ(BranchAt(1).history(), State(1));
  EXPECT_EQ(llm_->runner().slots().positions(2), 0U);
  EXPECT_EQ(llm_->runner().slots().positions(3), 0U);
}

TEST_F(LlmPromptSessions, ACapacityRefusedPromptChunkKeepsItsPrefixAndRetriesTheSameUnit) {
  StartCapacity(256, true);
  const std::vector<std::int32_t> prompt(515, 2);
  auto& branch = BranchAt(2);
  auto opened = branch.BeginPrompt(prompt);
  ASSERT_TRUE(opened);
  auto& session = **opened;
  ASSERT_TRUE(session.Advance({}, true));
  ASSERT_TRUE(session.Advance({}, true));
  Room();
  unsigned asked = 0;
  const rt::PrefillGoOn go_on = [&](std::uint32_t) {
    ++asked;
    return true;
  };
  for (int attempt = 0; attempt < 2; ++attempt) {
    const auto queued = source_->queued();
    auto refused = session.Advance(go_on, true);
    ASSERT_FALSE(refused);
    EXPECT_THAT(refused.error(), HasSubstr("the chunk at 256"));
    EXPECT_TRUE(session.refused());
    EXPECT_FALSE(session.done());
    EXPECT_EQ(branch.history(), std::vector<std::int32_t>(256, 2));
    EXPECT_EQ(State(2), branch.history());
    EXPECT_EQ(session.run().chunks, 1U);
    EXPECT_EQ(session.NextUnit()->phase, Phase::kChunk);
    EXPECT_EQ(session.NextUnit()->rows, 256U);
    EXPECT_EQ(source_->queued(), queued);
  }
  EXPECT_EQ(asked, 2U);
  Unlimit();
  while (!session.done()) {
    ASSERT_TRUE(session.Advance({}, true));
    EXPECT_FALSE(session.refused());
  }
  ASSERT_TRUE(session.Finish());
  EXPECT_EQ(branch.history(), prompt);
  EXPECT_EQ(State(2), prompt);
  CheckInjection(2, prompt);
  EXPECT_EQ(session.run().chunks, 3U);
  EXPECT_EQ(session.last(), Head(prompt));
}

TEST_F(LlmPromptSessions, AnUndeferredCapacityRefusalEndsThePromptAtItsProvenPrefix) {
  StartCapacity();
  auto opened = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(515, 2));
  ASSERT_TRUE(opened);
  ASSERT_TRUE((*opened)->Advance());
  ASSERT_TRUE((*opened)->Advance());
  Room();
  EXPECT_FALSE((*opened)->Advance());
  EXPECT_FALSE((*opened)->refused());
  EXPECT_TRUE((*opened)->done());
  EXPECT_FALSE((*opened)->Finish());
  EXPECT_EQ(BranchAt(1).history(), std::vector<std::int32_t>(256, 2));
  EXPECT_EQ(State(1), BranchAt(1).history());
  Unlimit();
}

TEST_F(LlmPromptSessions, ACancelledWaitingPromptAndGenerationKeepTheirCompletedPrefixes) {
  StartCapacity();
  auto prompt = BranchAt(1).BeginPrompt(std::vector<std::int32_t>(515, 2));
  ASSERT_TRUE(prompt);
  ASSERT_TRUE((*prompt)->Advance({}, true));
  ASSERT_TRUE((*prompt)->Advance({}, true));
  Room();
  EXPECT_FALSE((*prompt)->Advance({}, true));
  ASSERT_TRUE((*prompt)->refused());
  Cancel(**prompt);
  EXPECT_TRUE((*prompt)->run().stopped);
  EXPECT_EQ(BranchAt(1).history(), std::vector<std::int32_t>(256, 2));
  EXPECT_EQ(State(1), BranchAt(1).history());
  Unlimit();
  const std::vector<std::int32_t> prefix(255, 0);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(2).Prefill(prefix, last));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  rt::Generation out;
  auto generation = BranchAt(2).BeginGeneration(last, options, out);
  ASSERT_TRUE(generation);
  const std::array<rt::Llm::GenerationSession*, 1> one{generation->get()};
  ASSERT_TRUE(llm_->RunGenerationWave(one, true));
  Room();
  ASSERT_TRUE(llm_->RunGenerationWave(one, true));
  ASSERT_TRUE((*generation)->refused());
  EXPECT_FALSE((*generation)->done());
  (*generation)->Cancel();
  ASSERT_TRUE((*generation)->Finish());
  EXPECT_EQ(out.tokens, Chain(prefix, 2));
  EXPECT_EQ(BranchAt(2).history().size(), 256U);
  EXPECT_EQ(State(2), BranchAt(2).history());
  Unlimit();
}

TEST_F(LlmPromptSessions, ACapacityRefusedGenerationWaitsWhileItsPeerRunsThenCompletes) {
  StartCapacity();
  std::vector<float> last_a, last_b;
  const std::vector<std::int32_t> prefix_a(256, 0), prefix_b(1, 2);
  ASSERT_TRUE(BranchAt(1).Prefill(prefix_a, last_a));
  ASSERT_TRUE(BranchAt(2).Prefill(prefix_b, last_b));
  rt::GenerateOptions options;
  options.max_tokens = 3;
  options.stop = false;
  rt::Generation result_a, result_b;
  auto a = BranchAt(1).BeginGeneration(last_a, options, result_a);
  auto b = BranchAt(2).BeginGeneration(last_b, options, result_b);
  ASSERT_TRUE(a && b);
  Room();
  const std::array<rt::Llm::GenerationSession*, 2> both{a->get(), b->get()};
  for (int wave = 0; wave < 2; ++wave) {
    ASSERT_TRUE(llm_->RunGenerationWave(both, true));
    EXPECT_TRUE((*a)->refused());
    EXPECT_FALSE((*a)->done());
    EXPECT_FALSE((*b)->refused());
  }
  EXPECT_EQ(result_a.tokens, Chain(prefix_a, 1));
  EXPECT_EQ(result_a.steps, 0U);
  EXPECT_EQ(State(1), prefix_a);
  EXPECT_EQ(result_b.tokens, Chain(prefix_b, 3));
  EXPECT_TRUE((*b)->done());
  ASSERT_TRUE((*b)->Finish());
  Unlimit();
  const std::array<rt::Llm::GenerationSession*, 1> alone{a->get()};
  while (!(*a)->done()) {
    ASSERT_TRUE(llm_->RunGenerationWave(alone, true));
    EXPECT_FALSE((*a)->refused());
  }
  ASSERT_TRUE((*a)->Finish());
  EXPECT_EQ(result_a.tokens, Chain(prefix_a, 3));
  EXPECT_EQ(State(1), BranchAt(1).history());
  EXPECT_EQ(State(2), BranchAt(2).history());
  std::vector<float> last_c;
  const std::vector<std::int32_t> prefix_c(256, 4);
  ASSERT_TRUE(BranchAt(3).Prefill(prefix_c, last_c));
  rt::Generation result_c;
  auto c = BranchAt(3).BeginGeneration(last_c, options, result_c);
  ASSERT_TRUE(c);
  Room();
  const std::array<rt::Llm::GenerationSession*, 1> undeferred{c->get()};
  ASSERT_TRUE(llm_->RunGenerationWave(undeferred));
  EXPECT_FALSE((*c)->refused());
  EXPECT_TRUE((*c)->done());
  EXPECT_FALSE((*c)->Finish());
  EXPECT_EQ(BranchAt(3).history(), prefix_c);
  Unlimit();
}

TEST_F(LlmPromptSessions, AnIdleBranchReleasesItsRetainedStateForItsPeers) {
  StartPrompts();
  auto& idle = BranchAt(2);
  std::vector<float> last;
  ASSERT_TRUE(idle.Prefill(std::array<std::int32_t, 3>{0, 1, 2}, last));
  ASSERT_TRUE(idle.HasRetainedState());
  Close();
  ASSERT_TRUE(idle.ReleaseIdleState());
  EXPECT_FALSE(idle.HasRetainedState());
  EXPECT_EQ(llm_->runner().slots().positions(2), 0U);
  EXPECT_EQ(llm_->ResidentStateBytes(idle), 0U);
  Request({2});
  ASSERT_TRUE(idle.Prefill(std::array<std::int32_t, 1>{4}, last));
  EXPECT_THAT(idle.history(), ElementsAre(4));
  EXPECT_EQ(State(2), idle.history());
  auto busy = idle.BeginPrompt(std::array<std::int32_t, 2>{4, 5});
  ASSERT_TRUE(busy);
  EXPECT_DEATH((void)idle.ReleaseIdleState(), "another active prompt");
  Cancel(**busy);
  busy->reset();
  Stop();
  // The final concrete engine also gives a single-slot model a real state
  // owner. Its only branch can release idle state and restart from empty.
  StartPrompts(64, 8, 1);
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{0}, last));
  Close();
  ASSERT_TRUE(BranchAt(0).ReleaseIdleState());
  EXPECT_FALSE(BranchAt(0).HasRetainedState());
  Request({0});
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{3}, last));
  EXPECT_THAT(State(0), ElementsAre(3));
}

TEST_F(LlmPromptSessions, APreemptedGenerationResumesFromItsRebuiltStateWithoutRepeatingTokens) {
  for (const bool sampled : {false, true}) {
    StartCapacity();
    const std::vector<std::int32_t> prefix(254, 0);
    rt::GenerateOptions options;
    options.max_tokens = 6;
    options.stop = false;
    if (sampled) {
      options.sampling =
          llmp::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
      options.seed = 913;
    }
    Request({3});
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(3).Prefill(prefix, last));
    rt::Generation expected;
    ASSERT_TRUE(BranchAt(3).Generate(last, options, expected));
    const auto expected_history = BranchAt(3).history();
    Close();
    ASSERT_TRUE(BranchAt(3).ReleaseIdleState());
    Request({1});
    std::vector<std::int32_t> visible;
    auto streamed = options;
    streamed.on_tokens = [&](std::span<const std::int32_t> fresh) {
      visible.insert(visible.end(), fresh.begin(), fresh.end());
      return true;
    };
    ASSERT_TRUE(BranchAt(1).Prefill(prefix, last));
    rt::Generation out;
    auto opened = BranchAt(1).BeginGeneration(last, streamed, out);
    ASSERT_TRUE(opened);
    const std::array<rt::Llm::GenerationSession*, 1> one{opened->get()};
    ASSERT_TRUE(llm_->RunGenerationWave(one, true));
    ASSERT_TRUE(llm_->RunGenerationWave(one, true));
    Room();
    ASSERT_TRUE(llm_->RunGenerationWave(one, true));
    ASSERT_TRUE((*opened)->refused());
    (*opened)->Cancel();
    ASSERT_TRUE((*opened)->Finish());
    const auto held = BranchAt(1).history();
    ASSERT_EQ(held.size(), 256U);
    Unlimit();
    ASSERT_TRUE(BranchAt(1).Clear());
    EXPECT_EQ(llm_->runner().slots().positions(1), 0U);
    auto rebuilt = BranchAt(1).BeginPrompt(held);
    ASSERT_TRUE(rebuilt);
    Finish(**rebuilt);
    EXPECT_EQ(State(1), held);
    rt::Generation unused;
    EXPECT_FALSE(BranchAt(1).ResumeGeneration((*rebuilt)->last(), streamed, unused));
    auto resumed = BranchAt(1).ResumeGeneration((*rebuilt)->last(), streamed, out);
    ASSERT_TRUE(resumed);
    EXPECT_FALSE(out.cancelled);
    EXPECT_EQ(out.tokens.size(), 3U);
    const std::array<rt::Llm::GenerationSession*, 1> again{resumed->get()};
    while (!(*resumed)->done()) ASSERT_TRUE(llm_->RunGenerationWave(again, true));
    ASSERT_TRUE((*resumed)->Finish());
    EXPECT_EQ(out.tokens, expected.tokens) << sampled;
    EXPECT_EQ(visible, expected.tokens) << sampled;
    EXPECT_EQ(BranchAt(1).history(), expected_history);
    EXPECT_EQ(State(1), expected_history);
    Stop();
  }
}

TEST_F(LlmPromptSessions, AGenerationSetAsideResumesFromItsSpilledStateWithoutPrefill) {
  for (const bool sampled : {false, true}) {
    StartPrompts();
    const std::array<std::int32_t, 1> prefix{0};
    rt::GenerateOptions options;
    options.max_tokens = 6;
    options.stop = false;
    if (sampled) {
      options.sampling =
          llmp::execution::SamplingParams{.temperature = 1, .top_k = 0, .top_p = 1, .min_p = 0};
      options.seed = 913;
    }
    std::vector<float> last;
    ASSERT_TRUE(BranchAt(3).Prefill(prefix, last));
    rt::Generation expected;
    ASSERT_TRUE(BranchAt(3).Generate(last, options, expected));
    const auto expected_history = BranchAt(3).history();
    std::vector<std::int32_t> visible;
    auto streamed = options;
    streamed.on_tokens = [&](std::span<const std::int32_t> fresh) {
      visible.insert(visible.end(), fresh.begin(), fresh.end());
      return true;
    };
    ASSERT_TRUE(BranchAt(1).Prefill(prefix, last));
    rt::Generation out;
    auto opened = BranchAt(1).BeginGeneration(last, streamed, out);
    ASSERT_TRUE(opened);
    const std::array<rt::Llm::GenerationSession*, 1> one{opened->get()};
    ASSERT_TRUE(llm_->RunGenerationWave(one, true));
    ASSERT_TRUE(llm_->RunGenerationWave(one, true));
    (*opened)->Cancel();
    ASSERT_TRUE((*opened)->Finish());
    const auto held = BranchAt(1).history();
    ASSERT_EQ(held.size(), 3U);
    EXPECT_FALSE(llm_->SpillIdle(BranchAt(1)));
    BranchAt(1).HoldContinuation();
    ASSERT_TRUE(llm_->SpillSetAside(BranchAt(1)));
    Close();
    llm_->set_retention(std::chrono::seconds(0));
    EXPECT_TRUE(llm_->BranchIdle(BranchAt(1)));
    EXPECT_FALSE(BranchAt(1).ReleaseIdleState());
    EXPECT_TRUE(BranchAt(1).spilled());
    EXPECT_EQ(BranchAt(1).history(), held);
    const auto executed = source_->executed();
    Request({1});
    auto restored = BranchAt(1).BeginPrompt(held, 0, false, true);
    ASSERT_TRUE(restored);
    Finish(**restored);
    EXPECT_EQ((*restored)->reused(), 3U);
    EXPECT_EQ(source_->executed(), executed);
    EXPECT_EQ(llm_->spill_stats().restores, 1U);
    EXPECT_FALSE(BranchAt(1).spilled());
    EXPECT_EQ(State(1), held);
    auto resumed = BranchAt(1).ResumeGeneration((*restored)->last(), streamed, out);
    ASSERT_TRUE(resumed);
    const std::array<rt::Llm::GenerationSession*, 1> again{resumed->get()};
    while (!(*resumed)->done()) ASSERT_TRUE(llm_->RunGenerationWave(again, true));
    ASSERT_TRUE((*resumed)->Finish());
    EXPECT_EQ(out.tokens, expected.tokens) << sampled;
    EXPECT_EQ(visible, expected.tokens) << sampled;
    EXPECT_EQ(BranchAt(1).history(), expected_history);
    EXPECT_EQ(State(1), expected_history);
    BranchAt(1).ReleaseContinuation();
    Close();
    EXPECT_TRUE(llm_->BranchIdle(BranchAt(1)));
    Stop();
  }
}

TEST_F(LlmPromptSessions, AnIdleConversationSpilledForItsPeersContinuesExactly) {
  StartCapacity();
  Hold(1, 256);
  const auto first = BranchAt(1).history();
  std::vector<std::size_t> spilled;
  ReclaimLargest(spilled, true);
  StateBudget(2);
  Request({0});
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::vector<std::int32_t>(512, 2), last));
  EXPECT_THAT(spilled, ElementsAre(1U));
  EXPECT_TRUE(BranchAt(1).spilled());
  EXPECT_EQ(BranchAt(1).history(), first);
  EXPECT_EQ(llm_->spill_stats().spills, 1U);
  Request({1});
  auto next = first;
  next.insert(next.end(), {7, 0});
  const auto executed = source_->executed();
  std::uint32_t reused = 0;
  ASSERT_TRUE(BranchAt(1).PreparePrompt(next, 0, last, reused));
  EXPECT_THAT(spilled, ElementsAre(1U, 0U));
  EXPECT_EQ(reused, 256U);
  EXPECT_EQ(llm_->spill_stats().restores, 1U);
  EXPECT_FALSE(BranchAt(1).spilled());
  EXPECT_EQ(State(1), next);
  EXPECT_EQ(source_->executed() - executed, 1U);
  EXPECT_EQ(last, Head(next));
  EXPECT_TRUE(BranchAt(0).spilled());
  EXPECT_EQ(BranchAt(0).history().size(), 512U);
  llm_->set_capacity_reclaim({});
  Close();
  StateBudget(2);  // all state capacity belongs to the resident peer
  Request({0});
  std::vector<std::int32_t> returning(512, 2);
  returning.push_back(5);
  auto refused = BranchAt(0).PreparePrompt(returning, 0, last, reused);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(BranchAt(0).capacity_refused());
  EXPECT_TRUE(BranchAt(0).spilled());
  EXPECT_EQ(BranchAt(0).history().size(), 512U);
  Unlimit();
  ASSERT_TRUE(BranchAt(0).PreparePrompt(std::vector<std::int32_t>{3, 3}, 0, last, reused));
  EXPECT_EQ(reused, 0U);
  EXPECT_FALSE(BranchAt(0).spilled());
  EXPECT_THAT(State(0), ElementsAre(3, 3));
}

TEST_F(LlmPromptSessions, AReusablePrefixIsTheLiveHistoryAPromptContinues) {
  StartPrompts();
  Hold(1, 6);
  std::vector<std::int32_t> next(6, 1);
  next.push_back(2);
  EXPECT_EQ(llm_->ReusablePrefix(BranchAt(1), next), 6U);
  EXPECT_EQ(llm_->ReusablePrefix(BranchAt(2), next), 0U);
  EXPECT_EQ(llm_->ReusablePrefix(BranchAt(1), std::vector<std::int32_t>(6, 1)), 0U);
  EXPECT_EQ(llm_->ReusablePrefix(BranchAt(1), std::vector<std::int32_t>{1, 1, 3, 4}), 0U);
  llm_->set_retention(std::chrono::seconds(0));
  EXPECT_EQ(llm_->ReusablePrefix(BranchAt(1), next), 0U);
}

TEST_F(LlmPromptSessions, AnIdleBranchHasNoSessionAndNoLeaseAndCountsOnlyResidentState) {
  StartCapacity();
  Hold(1, 256);
  Hold(2, 512);
  Hold(3, 1);
  EXPECT_EQ(LargestIdle(BranchAt(0)), 2U);
  EXPECT_EQ(LargestIdle(BranchAt(2)), 1U);
  auto busy = BranchAt(2).BeginPrompt(std::array<std::int32_t, 1>{4});
  ASSERT_TRUE(busy);
  EXPECT_FALSE(llm_->BranchIdle(BranchAt(2)));
  EXPECT_EQ(LargestIdle(BranchAt(0)), 1U);
  Cancel(**busy);
  Request({2});
  EXPECT_FALSE(llm_->BranchIdle(BranchAt(2)));
  Close();
  EXPECT_TRUE(llm_->BranchIdle(BranchAt(2)));
  const auto bytes = llm_->StateBytes(BranchAt(2));
  ASSERT_TRUE(llm_->SpillIdle(BranchAt(2)));
  EXPECT_EQ(llm_->ResidentStateBytes(BranchAt(2)), 0U);
  EXPECT_EQ(llm_->SpilledStateBytes(BranchAt(2)), bytes);
  EXPECT_EQ(llm_->StateBytes(BranchAt(2)), bytes);
  EXPECT_EQ(LargestIdle(BranchAt(0)), 1U);
  ASSERT_TRUE(BranchAt(1).ReleaseIdleState());
  ASSERT_TRUE(BranchAt(3).ReleaseIdleState());
  EXPECT_FALSE(LargestIdle(BranchAt(0)));
  busy->reset();
  Stop();
  StartPrompts(64, 8, 1);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 2>{0, 1}, last));
  EXPECT_FALSE(llm_->BranchIdle(BranchAt(0)));  // the actual request holds its lease
}

TEST_F(LlmPromptSessions, AHeldUnclaimedConversationCanSpillForARestoredPeersGrowth) {
  StartCapacity();
  Hold(1, 512);
  Hold(2, 256);
  const auto exact = BranchAt(1).history();
  BranchAt(1).HoldContinuation();
  StateBudget(3);
  unsigned reclaimed = 0;
  llm_->set_capacity_reclaim([&](const rt::Llm::Branch& keep) {
    if (LargestIdle(keep) != 1U) return false;
    ++reclaimed;
    return llm_->SpillIdle(BranchAt(1)).has_value();
  });
  Request({0});
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(std::vector<std::int32_t>(256, 2), last));
  EXPECT_EQ(reclaimed, 1U);
  EXPECT_TRUE(BranchAt(1).spilled());
  EXPECT_EQ(BranchAt(1).history(), exact);
  EXPECT_FALSE(BranchAt(1).ReleaseIdleState());
  Unlimit();
  Request({1});
  auto restored = BranchAt(1).BeginPrompt(exact, 0, false, true);
  ASSERT_TRUE(restored);
  Finish(**restored);
  EXPECT_EQ((*restored)->reused(), exact.size());
  EXPECT_EQ(BranchAt(1).history(), exact);
  EXPECT_EQ(State(1), exact);
  BranchAt(1).ReleaseContinuation();
}

TEST_F(LlmPromptSessions, ASerialPrefillReleasesIdleStateLargestFirstAndRetriesTheRefusedChunk) {
  StartCapacity();
  Hold(1, 256);
  Hold(2, 512);
  std::vector<std::size_t> released;
  ReclaimLargest(released);
  StateBudget(3);
  Request({0});
  const std::vector<std::int32_t> prompt(512, 2);
  std::vector<float> last;
  std::vector<std::uint32_t> asked;
  rt::PrefillRun run;
  ASSERT_TRUE(BranchAt(0).Prefill(
      prompt, last,
      [&](std::uint32_t rows) {
        asked.push_back(rows);
        return true;
      },
      &run));
  EXPECT_THAT(released, ElementsAre(2U));
  EXPECT_THAT(asked, ElementsAre(256U, 256U));
  EXPECT_EQ(BranchAt(0).history(), prompt);
  EXPECT_EQ(State(0), prompt);
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  EXPECT_EQ(run.chunks, 2U);
  EXPECT_EQ(llm_->runner().slots().positions(2), 0U);
  EXPECT_EQ(State(1).size(), 256U);
  EXPECT_EQ(BranchAt(1).history().size(), 256U);
  Unlimit();
}

TEST_F(LlmPromptSessions, ASerialPrefillThatCannotFitAloneEndsTypedAtItsCompletedPrefix) {
  StartCapacity();
  Hold(1, 256);
  Hold(2, 512);
  std::vector<std::size_t> released;
  ReclaimLargest(released);
  StateBudget(2);
  Request({0});
  std::vector<float> last;
  auto refused = BranchAt(0).Prefill(std::vector<std::int32_t>(1024, 2), last);
  ASSERT_FALSE(refused);
  EXPECT_THAT(refused.error(), HasSubstr("the chunk at 512"));
  EXPECT_THAT(released, ElementsAre(2U, 1U));
  EXPECT_TRUE(BranchAt(0).capacity_refused());
  EXPECT_TRUE(last.empty());
  EXPECT_EQ(BranchAt(0).history(), std::vector<std::int32_t>(512, 2));
  EXPECT_EQ(State(0), BranchAt(0).history());
  // Additional rows inside the existing extent resume without a fresh clear.
  // At this aligned boundary another extent is needed; make that room explicit.
  Unlimit();
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 1>{3}, last));
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  EXPECT_EQ(State(0).size(), 513U);
  EXPECT_EQ(State(0).back(), 3);
  llm_->set_capacity_reclaim({});
  Close();
  ASSERT_TRUE(BranchAt(0).ReleaseIdleState());
  Hold(3, 256);
  StateBudget(2);
  Request({0});
  EXPECT_FALSE(BranchAt(0).Prefill(std::vector<std::int32_t>(512, 2), last));
  EXPECT_TRUE(BranchAt(0).capacity_refused());
  EXPECT_EQ(BranchAt(0).history(), std::vector<std::int32_t>(256, 2));
  EXPECT_EQ(State(3).size(), 256U);
  EXPECT_THAT(released, ElementsAre(2U, 1U));
  Unlimit();
}

TEST_F(LlmPromptSessions, ASerialNonCapacityFailureIsNeitherReclaimedNorTyped) {
  StartPrompts();
  Hold(1, 6);
  std::vector<std::size_t> released;
  ReclaimLargest(released);
  Request({0});
  source_->options().after_execute = [&](pl::Phase phase) {
    if (phase == pl::Phase::kPrefill) FailRun();
  };
  std::vector<float> last;
  EXPECT_FALSE(BranchAt(0).Prefill(std::vector<std::int32_t>(12, 2), last));
  source_->options().after_execute = {};
  EXPECT_TRUE(released.empty());
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  EXPECT_TRUE(BranchAt(0).history().empty());
  EXPECT_EQ(State(1).size(), 6U);
  Recover();
}

TEST_F(LlmPromptSessions, ASerialPromptReleasesIdleStateAndRetriesOrEndsTyped) {
  StartCapacity();
  Hold(3, 256);
  std::vector<std::size_t> released;
  ReclaimLargest(released);
  StateBudget(2);
  Request({0});
  const std::vector<std::int32_t> turn(512, 2);
  std::vector<float> last;
  std::uint32_t reused = 0;
  unsigned asked = 0;
  const rt::PrefillGoOn go_on = [&](std::uint32_t) {
    ++asked;
    return true;
  };
  ASSERT_TRUE(BranchAt(0).PreparePrompt(turn, 0, last, reused, go_on));
  EXPECT_THAT(released, ElementsAre(3U));
  EXPECT_EQ(BranchAt(0).history(), turn);
  EXPECT_EQ(last, Head(turn));
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  EXPECT_EQ(asked, 4U);
  auto longer = turn;
  longer.resize(1024, 3);
  auto refused = BranchAt(0).PreparePrompt(longer, 0, last, reused, go_on);
  ASSERT_FALSE(refused);
  EXPECT_TRUE(BranchAt(0).capacity_refused());
  EXPECT_EQ(reused, 512U);
  EXPECT_EQ(BranchAt(0).history(), turn);
  EXPECT_TRUE(last.empty());
  EXPECT_THAT(released, ElementsAre(3U));
  Unlimit();
}

TEST_F(LlmPromptSessions, ASerialScoreReleasesIdleStateAndReportsEachRowOnce) {
  StartCapacity(1);
  Hold(1, 1);
  std::vector<std::size_t> released;
  ReclaimLargest(released);
  StateBudget(1);
  Request({0});
  const auto prompt = Prompt(128, 0);
  std::vector<float> last;
  std::vector<std::int32_t> scored;
  ASSERT_TRUE(BranchAt(0).ScorePrompt(prompt, last, [&](std::int32_t id, std::span<const float>) {
    scored.push_back(id);
    return true;
  }));
  EXPECT_THAT(released, ElementsAre(1U));
  EXPECT_EQ(scored, (std::vector<std::int32_t>(prompt.begin() + 1, prompt.end())));
  EXPECT_EQ(BranchAt(0).history(), prompt);
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  Unlimit();
  Close();
  ASSERT_TRUE(BranchAt(0).ReleaseIdleState());
  StateBudget(1);
  Request({0});
  const auto longer = Prompt(258, 0);
  scored.clear();
  auto refused =
      BranchAt(0).ScorePrompt(longer, last, [&](std::int32_t id, std::span<const float>) {
        scored.push_back(id);
        return true;
      });
  ASSERT_FALSE(refused);
  EXPECT_THAT(refused.error(), HasSubstr("the row at 256"));
  EXPECT_TRUE(BranchAt(0).capacity_refused());
  EXPECT_EQ(scored, (std::vector<std::int32_t>(longer.begin() + 1, longer.begin() + 257)));
  EXPECT_EQ(BranchAt(0).history(),
            (std::vector<std::int32_t>(longer.begin(), longer.begin() + 256)));
  Unlimit();
}

TEST_F(LlmPromptSessions, ASerialGenerationReleasesIdleStateThenEndsTypedWhenAlone) {
  StartCapacity();
  Hold(1, 256);
  std::vector<std::size_t> released;
  ReclaimLargest(released);
  StateBudget(2);
  Request({0});
  const std::vector<std::int32_t> prefix(255, 0);
  std::vector<float> last;
  ASSERT_TRUE(BranchAt(0).Prefill(prefix, last));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  std::vector<std::int32_t> visible;
  options.on_tokens = [&](std::span<const std::int32_t> fresh) {
    visible.insert(visible.end(), fresh.begin(), fresh.end());
    return true;
  };
  rt::Generation out;
  ASSERT_TRUE(BranchAt(0).Generate(last, options, out));
  EXPECT_THAT(released, ElementsAre(1U));
  EXPECT_EQ(out.tokens, Chain(prefix, 4));
  EXPECT_EQ(visible, out.tokens);
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  EXPECT_EQ(BranchAt(0).history().size(), 258U);
  Unlimit();
  Close();
  ASSERT_TRUE(BranchAt(0).ReleaseIdleState());
  StateBudget(1);
  Request({0});
  ASSERT_TRUE(BranchAt(0).Prefill(prefix, last));
  rt::Generation alone;
  visible.clear();
  auto refused = BranchAt(0).Generate(last, options, alone);
  ASSERT_FALSE(refused);
  EXPECT_THAT(refused.error(), HasSubstr("decode step at 256"));
  EXPECT_TRUE(BranchAt(0).capacity_refused());
  EXPECT_EQ(alone.tokens, Chain(prefix, 2));
  EXPECT_EQ(visible, alone.tokens);
  EXPECT_EQ(BranchAt(0).history().size(), 256U);
  EXPECT_EQ(State(0), BranchAt(0).history());
  Unlimit();
  Close();
  ASSERT_TRUE(BranchAt(0).ReleaseIdleState());
  StateBudget(1);
  Request({0});
  ASSERT_TRUE(BranchAt(0).Prefill(prefix, last));
  llm_->set_capacity_reclaim([&](const rt::Llm::Branch&) {
    execution_identity_ = std::unexpected("settling failed: execution identity unavailable");
    return false;
  });
  rt::Generation unsettled;
  auto failed = BranchAt(0).Generate(last, options, unsettled);
  ASSERT_FALSE(failed);
  EXPECT_THAT(failed.error(), HasSubstr("settling failed"));
  EXPECT_FALSE(BranchAt(0).capacity_refused());
  EXPECT_TRUE(BranchAt(0).history().empty());
  execution_identity_ = std::nullopt;
  Unlimit();
}

TEST_F(LlmPromptSessions, PromptFutureRowsUseFreshOrResumedGenerationBudgetsExactly) {
  EXPECT_EQ(rt::PromptTargetRows(1, 0, false), 0U);
  EXPECT_EQ(rt::PromptTargetRows(10, 4, false), 9U);
  EXPECT_EQ(rt::PromptTargetRows(10, 4, true), 6U);
  EXPECT_EQ(rt::PromptTargetRows(3, 9, true), 0U);
  EXPECT_FALSE(rt::PromptTargetRows(std::numeric_limits<std::size_t>::max(), 0, true));
}

TEST_F(LlmPromptSessions, ActivePeerBudgetTracksApplyAndRejectsPreparedOrCancelledSessions) {
  StartPrompts();
  auto& branch = BranchAt(0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.stop = false;
  std::uint32_t callbacks = 0;
  options.on_tokens = [&](std::span<const std::int32_t>) {
    ++callbacks;
    EXPECT_FALSE(rt::LlmTestAccess::Ready(*llm_, branch));
    EXPECT_FALSE(rt::LlmTestAccess::Remaining(*llm_, branch));
    return true;
  };
  rt::Generation out;
  auto opened = branch.BeginGeneration(last, options, out);
  ASSERT_TRUE(opened);
  auto& session = **opened;
  EXPECT_TRUE(rt::LlmTestAccess::Ready(*llm_, branch));
  EXPECT_EQ(rt::LlmTestAccess::Remaining(*llm_, branch), 3U);
  auto step = session.PrepareStep();
  ASSERT_TRUE(step);
  EXPECT_FALSE(rt::LlmTestAccess::Ready(*llm_, branch));
  EXPECT_FALSE(rt::LlmTestAccess::Remaining(*llm_, branch));
  std::vector<float> completed;
  ASSERT_TRUE(rt::LlmTestAccess::ExecuteChunk(*llm_, branch, *step, completed));
  ASSERT_TRUE(session.ApplyPlain(completed));
  EXPECT_TRUE(rt::LlmTestAccess::Ready(*llm_, branch));
  EXPECT_EQ(callbacks, 2U);
  EXPECT_EQ(rt::LlmTestAccess::Remaining(*llm_, branch), 2U);
  session.Cancel();
  EXPECT_FALSE(rt::LlmTestAccess::Ready(*llm_, branch));
  EXPECT_EQ(rt::LlmTestAccess::Remaining(*llm_, branch), 0U);
  ASSERT_TRUE(session.Finish());
  EXPECT_FALSE(rt::LlmTestAccess::Remaining(*llm_, branch));
  rt::Generation next;
  auto again = branch.BeginGeneration(completed, options, next);
  ASSERT_TRUE(again);
  ASSERT_TRUE(session.Finish());
  EXPECT_TRUE(rt::LlmTestAccess::Ready(*llm_, branch));
  (*again)->Cancel();
  ASSERT_TRUE((*again)->Finish());
}

TEST_F(LlmPromptSessions, StoppedOneTokenAndFailedBeginHaveNoFutureTargetRowsOrStaleOwner) {
  StartPrompts();
  auto& branch = BranchAt(0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::GenerateOptions options;
  options.max_tokens = 4;
  options.sampling = llmp::execution::SamplingParams{};
  auto bad = last;
  bad[0] = std::numeric_limits<float>::quiet_NaN();
  rt::Generation out;
  EXPECT_FALSE(branch.BeginGeneration(bad, options, out));
  EXPECT_FALSE(rt::LlmTestAccess::Ready(*llm_, branch));
  EXPECT_FALSE(rt::LlmTestAccess::Remaining(*llm_, branch));
  options.sampling.reset();
  rt::LlmTestAccess::Stops(*llm_, {7});
  auto stopped = branch.BeginGeneration(Row(7), options, out);
  ASSERT_TRUE(stopped);
  EXPECT_TRUE((*stopped)->done());
  EXPECT_EQ(rt::LlmTestAccess::Remaining(*llm_, branch), 0U);
  ASSERT_TRUE((*stopped)->Finish());
  options.max_tokens = 1;
  auto single = branch.BeginGeneration(last, options, out);
  ASSERT_TRUE(single);
  EXPECT_TRUE((*single)->done());
  EXPECT_EQ(rt::LlmTestAccess::Remaining(*llm_, branch), 0U);
  ASSERT_TRUE((*single)->Finish());
  EXPECT_FALSE(rt::LlmTestAccess::Remaining(*llm_, branch));
}

TEST_F(LlmPromptSessions, ResumedSessionUsesProducedTokenCountWithoutAnotherFirstSample) {
  StartPrompts();
  auto& branch = BranchAt(0);
  std::vector<float> last;
  ASSERT_TRUE(branch.Prefill(std::array<std::int32_t, 1>{0}, last));
  rt::GenerateOptions options;
  options.max_tokens = 6;
  options.stop = false;
  rt::Generation out;
  auto first = branch.BeginGeneration(last, options, out);
  ASSERT_TRUE(first);
  ASSERT_TRUE((*first)->RunScalarStep());
  ASSERT_EQ(out.tokens.size(), 2U);
  (*first)->Cancel();
  ASSERT_TRUE((*first)->Finish());
  auto resumed = branch.ResumeGeneration({}, options, out);
  ASSERT_TRUE(resumed);
  EXPECT_EQ(out.tokens.size(), 2U);
  EXPECT_FALSE(out.cancelled);
  EXPECT_EQ(rt::LlmTestAccess::Remaining(*llm_, branch), 4U);
  (*resumed)->Cancel();
  ASSERT_TRUE((*resumed)->Finish());
}

TEST_F(LlmPromptSessions, SharedPromptDispatchCarriesEachSessionsFutureBudgetAndSamplingIntent) {
  StartPrompts();
  std::array<std::unique_ptr<Session>, 2> sessions;
  const std::array<std::int32_t, 8> prompt{0, 1, 2, 3, 4, 5, 6, 0};
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    auto begun = BranchAt(i).BeginPrompt(prompt);
    ASSERT_TRUE(begun);
    sessions[i] = std::move(*begun);
    EXPECT_FALSE(sessions[i]->has_continuation());
    sessions[i]->SetContinuationRows(rt::PromptTargetRows(i == 0 ? 1 : 5, 0, false), i != 0);
    EXPECT_EQ(sessions[i]->prompt_rows(), prompt.size());
    ASSERT_TRUE(sessions[i]->Advance());
    ASSERT_EQ(sessions[i]->NextUnit()->phase, Phase::kChunk);
  }
  const std::array<Session*, 2> both{sessions[0].get(), sessions[1].get()};
  const std::array<rt::PrefillGoOn, 2> go_on{};
  ASSERT_TRUE(llm_->RunPromptWave(both, go_on));
  EXPECT_EQ(sessions[0]->continuation_rows(), 0U);
  EXPECT_EQ(sessions[1]->continuation_rows(), 4U);
  EXPECT_EQ(sessions[0]->continuation_sampling(), false);
  EXPECT_EQ(sessions[1]->continuation_sampling(), true);
  // Inspect the real source's predicted programs: owner zero is done after
  // its first sample; only owner one's four sampled target rows may appear.
  bool predicted = false;
  for (const auto& built : source_->built()) {
    if (built.measuring || built.shape.phase != pl::Phase::kDecode) continue;
    for (const auto& segment : built.shape.segments) {
      EXPECT_EQ(segment.slot, 1U);
      EXPECT_FALSE(segment.greedy);
      EXPECT_GE(segment.past, 8U);
      EXPECT_LT(segment.past, 12U);
      predicted = true;
    }
  }
  EXPECT_TRUE(predicted);
  for (auto& session : sessions) {
    EXPECT_TRUE(session->done());
    ASSERT_TRUE(session->Finish());
  }
}

}  // namespace
