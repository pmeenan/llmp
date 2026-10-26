// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Room for a swap's incoming model (runtime/swap_room.h), against a ledger
// like the node's: plans and graphs charged past a floor as one extent of
// whole 2 MiB steps, so reclaiming N bytes of them may drop occupancy by
// less than N. A swap short by less than one extent is made room for, or
// refused before it starts; never started short.

#include "runtime/swap_room.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace {

namespace rt = llmp::runtime;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kExtent = 2 * kMiB;

TEST(SwapRoom, IncompatibleDonorsReleaseBeforeHostFirstLoadsAtTheBudget) {
  const llmp::catalog::DomainId domain{0, 1};
  const auto extent = [&](std::uint32_t id, std::size_t allocation_class) {
    return rt::SwapBackingExtent{
        .extent = {id, 1},
        .bytes = kExtent,
        .backing = rt::SwapBackingKey{
            .domain = domain, .allocation_class = allocation_class, .size = kExtent}};
  };
  std::vector<rt::SwapBackingExtent> missing = {extent(10, 1), extent(11, 2), extent(12, 2),
                                                extent(13, 2)};
  const std::array selected = {extent(1, 2), extent(2, 2), extent(3, 2), extent(4, 2)};
  auto release = rt::ReleaseForHandoff(domain, 4 * kExtent, 4 * kExtent, missing, selected);
  ASSERT_TRUE(release) << release.error();
  EXPECT_EQ(*release, (std::vector<llmp::catalog::ExtentId>{{1, 1}}));
  std::ranges::reverse(missing);
  EXPECT_EQ(rt::ReleaseForHandoff(domain, 4 * kExtent, 4 * kExtent, missing, selected), release);
  missing.front().backing->allocation_class = 1;
  release = rt::ReleaseForHandoff(domain, 4 * kExtent, 4 * kExtent, missing, selected);
  ASSERT_TRUE(release) << release.error();
  EXPECT_EQ(release->size(), 2U);
}

TEST(SwapRoom, CompatibleDonorsStillReleaseEnoughForAnExistingOvercharge) {
  const llmp::catalog::DomainId domain{0, 1};
  const auto extent = [&](std::uint32_t id) {
    return rt::SwapBackingExtent{.extent = {id, 1},
                                 .bytes = kExtent,
                                 .backing = rt::SwapBackingKey{.domain = domain, .size = kExtent}};
  };
  const std::array missing = {extent(10), extent(11)};
  const std::array selected = {extent(1), extent(2), extent(3)};
  auto release = rt::ReleaseForHandoff(domain, 4 * kExtent, 4 * kExtent, missing, selected);
  ASSERT_TRUE(release) << release.error();
  EXPECT_TRUE(release->empty());
  release = rt::ReleaseForHandoff(domain, 5 * kExtent, 4 * kExtent, missing, selected);
  ASSERT_TRUE(release) << release.error();
  EXPECT_EQ(*release, (std::vector<llmp::catalog::ExtentId>{{1, 1}}));
  EXPECT_FALSE(rt::ReleaseForHandoff(domain, 6 * kExtent, 4 * kExtent, missing, selected));
}

TEST(SwapRoom, HandoffPartitionRejectsAliasesCrossDomainAndMismatchedCharge) {
  const llmp::catalog::DomainId domain{0, 1};
  const rt::SwapBackingExtent incoming{
      .extent = {10, 1}, .bytes = kExtent, .backing = std::nullopt};
  rt::SwapBackingExtent outgoing{.extent = {1, 1},
                                 .bytes = kExtent,
                                 .backing = rt::SwapBackingKey{.domain = domain, .size = kExtent}};
  const std::array missing = {incoming};
  std::array selected = {outgoing};
  EXPECT_TRUE(rt::ReleaseForHandoff(domain, kExtent, kExtent, missing, selected));
  selected[0].backing->domain = {1, 1};
  EXPECT_FALSE(rt::ReleaseForHandoff(domain, kExtent, kExtent, missing, selected));
  selected[0] = outgoing;
  selected[0].backing->size = kExtent / 2;
  EXPECT_FALSE(rt::ReleaseForHandoff(domain, kExtent, kExtent, missing, selected));
  selected[0] = incoming;
  EXPECT_FALSE(rt::ReleaseForHandoff(domain, kExtent, kExtent, missing, selected));
  selected[0] = outgoing;
  selected[0].backing.reset();
  const auto release = rt::ReleaseForHandoff(domain, kExtent, kExtent, missing, selected);
  ASSERT_TRUE(release) << release.error();
  EXPECT_EQ(*release, (std::vector<llmp::catalog::ExtentId>{{1, 1}}));
}

TEST(SwapRoom, HandoffPartitionAuthenticatesWholeTypedIdentitiesAndSelectedOrder) {
  const llmp::catalog::DomainId domain{0, 1};
  const auto extent = [&](std::uint32_t index, std::uint32_t generation,
                          std::size_t allocation_class) {
    return rt::SwapBackingExtent{
        .extent = {index, generation},
        .bytes = kExtent,
        .backing = rt::SwapBackingKey{
            .domain = domain, .allocation_class = allocation_class, .size = kExtent}};
  };
  // Typed identities include the generation: an old index reused at another
  // generation is not an alias. Shuffled input must keep selected release order.
  std::vector missing{extent(4, 2, 1), extent(2, 1, 2), extent(10, 1, 1)};
  std::vector selected{extent(8, 1, 2), extent(4, 1, 2), extent(1, 1, 2)};
  const auto release = rt::ReleaseForHandoff(domain, 3 * kExtent, 3 * kExtent, missing, selected);
  ASSERT_TRUE(release) << release.error();
  EXPECT_EQ(*release, (std::vector<llmp::catalog::ExtentId>{{8, 1}, {4, 1}}));
  std::ranges::reverse(missing);
  EXPECT_EQ(rt::ReleaseForHandoff(domain, 3 * kExtent, 3 * kExtent, missing, selected), release);
  std::ranges::reverse(selected);
  const auto reversed = rt::ReleaseForHandoff(domain, 3 * kExtent, 3 * kExtent, missing, selected);
  ASSERT_TRUE(reversed) << reversed.error();
  EXPECT_EQ(*reversed, (std::vector<llmp::catalog::ExtentId>{{1, 1}, {4, 1}}));

  auto duplicated = missing;
  duplicated.push_back(missing.front());
  EXPECT_FALSE(rt::ReleaseForHandoff(domain, 3 * kExtent, 4 * kExtent, duplicated, selected));
  auto selected_duplicate = selected;
  selected_duplicate.push_back(selected.front());
  EXPECT_FALSE(
      rt::ReleaseForHandoff(domain, 3 * kExtent, 3 * kExtent, missing, selected_duplicate));
  selected_duplicate = selected;
  selected_duplicate[1] = missing.front();
  EXPECT_FALSE(
      rt::ReleaseForHandoff(domain, 3 * kExtent, 3 * kExtent, missing, selected_duplicate));
  duplicated = missing;
  duplicated.front().extent = {};
  EXPECT_FALSE(rt::ReleaseForHandoff(domain, 3 * kExtent, 3 * kExtent, duplicated, selected));
}

// The node's budget with a plans charge in whole extents past its floor.
struct Ledger {
  std::uint64_t budget = 0;
  std::uint64_t others = 0;  // everything but the plans' charge
  std::uint64_t plans = 0;   // what the plans hold, counted exactly
  std::uint64_t floor = 3 * kMiB;
  std::uint64_t incoming = 0;  // what the swap pages in beyond what goes out

  std::uint64_t Charge() const {
    const std::uint64_t past = plans > floor ? plans - floor : 0;
    return (past + kExtent - 1) / kExtent * kExtent;
  }
  std::uint64_t Shortfall() const {
    const std::uint64_t occupancy = others + Charge();
    return occupancy + incoming > budget ? occupancy + incoming - budget : 0;
  }
  // Frees whole plans of `plan` bytes until `asked` is reported freed.
  std::uint64_t Reclaim(std::uint64_t asked, std::uint64_t plan) {
    std::uint64_t freed = 0;
    while (freed < asked && plans >= plan) {
      plans -= plan;
      freed += plan;
    }
    return freed;
  }
};

// The rounds Server::MakeRoomForSwap runs, against the ledger: sized, then
// reclaimed for what is short in whole extents, until it fits or a round
// makes no progress (RoomProgress), with no count of rounds. Refused (the
// swap does not start) when room is still short.
bool MakeRoom(Ledger& l, const std::function<std::uint64_t(std::uint64_t)>& reclaim, int& rounds) {
  rt::RoomProgress progress;
  for (;;) {
    const std::uint64_t short_by = l.Shortfall();
    if (short_by == 0) {
      return true;
    }
    if (!progress.Shrank(short_by)) {
      return false;
    }
    ++rounds;
    if (reclaim((short_by + kExtent - 1) / kExtent * kExtent) == 0) {
      return false;
    }
  }
}

TEST(SwapRoom, AShortfallUnderAnExtentIsRecheckedUntilItFits) {
  Ledger l;
  l.plans = l.floor + (20 * kMiB) + (kMiB / 2);  // charged as 22 MiB
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge() + (8 * kMiB);
  l.incoming = (8 * kMiB) + (kMiB / 2);  // short by half a MiB
  ASSERT_EQ(l.Shortfall(), kMiB / 2);
  // A reclaim of exactly what is short can leave occupancy where it was:
  // 0.6 MiB of plans freed, the charge still 22 MiB. The room is checked
  // again and more taken.
  int rounds = 0;
  EXPECT_TRUE(MakeRoom(
      l, [&](std::uint64_t ask) { return l.Reclaim(ask, std::uint64_t{600} * 1024); }, rounds));
  EXPECT_EQ(l.Shortfall(), 0U);
  EXPECT_GE(rounds, 1);
}

// No count of rounds: a reclaimer that frees a little each round goes on
// as long as the shortfall shrinks, past the four rounds it once had.
TEST(SwapRoom, RoundsGoOnWhileEachMakesProgress) {
  Ledger l;
  l.plans = l.floor + (64 * kMiB);
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge();
  l.incoming = 16 * kMiB;
  int rounds = 0;
  // One extent's worth a round, whatever is asked.
  EXPECT_TRUE(MakeRoom(l, [&](std::uint64_t) { return l.Reclaim(kExtent, kExtent); }, rounds));
  EXPECT_EQ(l.Shortfall(), 0U);
  EXPECT_EQ(rounds, 8);
}

TEST(SwapRoom, WithTooLittleToReclaimTheSwapIsRefusedBeforeItStarts) {
  Ledger l;
  l.plans = l.floor + kMiB;  // one extent charged, little to give
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge();
  l.incoming = 4 * kMiB;
  int rounds = 0;
  EXPECT_FALSE(MakeRoom(l, [&](std::uint64_t ask) { return l.Reclaim(ask, kMiB); }, rounds));
  EXPECT_GT(l.Shortfall(), 0U);  // refused: the caller does not swap
  // Room already there: nothing asked.
  Ledger roomy;
  roomy.budget = 10 * kMiB;
  roomy.incoming = 5 * kMiB;
  rounds = 0;
  EXPECT_TRUE(MakeRoom(roomy, [](std::uint64_t) { return std::uint64_t{0}; }, rounds));
  EXPECT_EQ(rounds, 0);
}

// A reclaimer that reports freed bytes but drops nothing (a ledger that
// never moves): stopped after the first round that made no progress.
TEST(SwapRoom, ARoundWithoutProgressStops) {
  Ledger l;
  l.plans = l.floor + (64 * kMiB);
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge();
  l.incoming = 3 * kMiB;
  int rounds = 0;
  EXPECT_FALSE(MakeRoom(l, [](std::uint64_t ask) { return ask; }, rounds));
  EXPECT_EQ(rounds, 1);
}

// A10: a swap short of room waits for an earlier activation's releases
// still in flight (they count as occupancy until they land) rather than
// evicting cache for room they are about to give back; it reclaims once
// they are done, or once they have stalled for the node's quiet time.
TEST(SwapRoom, AShortSwapWaitsForAnEarlierReleaseTailWhileItMoves) {
  using Receipt = llmp::scheduler::EvictionReceipt;
  using std::chrono::milliseconds;
  const auto now = Receipt::Clock::now();
  constexpr milliseconds kQuiet{600000};
  Receipt::Snapshot moving;
  moving.pending = 12;
  moving.progressed = now - milliseconds(2);
  constexpr std::uint64_t kEvicting = 24 * kExtent;
  EXPECT_EQ(rt::WaitForReleaseTail(moving, kEvicting, now, kQuiet), rt::RoomWait::kWait);
  Receipt::Snapshot done = moving;
  done.pending = 0;
  done.complete = true;
  EXPECT_EQ(rt::WaitForReleaseTail(done, kEvicting, now, kQuiet), rt::RoomWait::kReclaim);
  Receipt::Snapshot stalled = moving;
  stalled.progressed = now - kQuiet;
  EXPECT_EQ(rt::WaitForReleaseTail(stalled, kEvicting, now, kQuiet), rt::RoomWait::kReclaim);
  // Incomplete, but nothing left evicting (a failed load holds what keeps it
  // open): no room would come of waiting.
  EXPECT_EQ(rt::WaitForReleaseTail(moving, 0, now, kQuiet), rt::RoomWait::kReclaim);
}

// An incoming extent the occupancy charges already (loading, or evicting:
// parked with its backing, or still releasing) adds nothing to the room a
// swap needs; only a nonresident one adds its bytes. Counted twice, a quick
// return to the model just left (its extents parked in the last swap's
// tail) waits for, or reclaims cache for, room it already holds.
TEST(SwapRoom, IncomingExtentsAlreadyChargedAddNothing) {
  using State = llmp::catalog::ExtentState;
  EXPECT_EQ(rt::IncomingCharge(State::kNonresident, kExtent), kExtent);
  EXPECT_EQ(rt::IncomingCharge(State::kResident, kExtent), 0U);
  EXPECT_EQ(rt::IncomingCharge(State::kLoading, kExtent), 0U);
  EXPECT_EQ(rt::IncomingCharge(State::kEvicting, kExtent), 0U);
}

}  // namespace
