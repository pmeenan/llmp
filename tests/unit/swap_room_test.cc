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
#include <cstdint>
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

TEST(SwapRoom, AShortfallUnderAnExtentIsRecheckedUntilItFits) {
  Ledger l;
  l.plans = l.floor + (20 * kMiB) + (kMiB / 2);  // charged as 22 MiB
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge() + (8 * kMiB);
  l.incoming = (8 * kMiB) + (kMiB / 2);  // short by half a MiB
  ASSERT_EQ(l.Shortfall(), kMiB / 2);
  // A reclaim of exactly what is short (the review's 50.3 of 50.0 MiB) can
  // leave occupancy where it was: 0.6 MiB of plans freed, the charge still
  // 22 MiB. The room is checked again and more taken.
  int calls = 0;
  auto made = rt::MakeRoom([&] { return l.Shortfall(); },
                           [&](std::uint64_t ask) {
                             ++calls;
                             EXPECT_EQ(ask % kExtent, 0U);  // whole extents
                             return l.Reclaim(ask, std::uint64_t{600} * 1024);
                           },
                           kExtent);
  ASSERT_TRUE(made) << made.error();
  EXPECT_EQ(l.Shortfall(), 0U);
  EXPECT_GE(calls, 1);
  EXPECT_GE(made->freed, made->asked);
}

TEST(SwapRoom, WithTooLittleToReclaimTheSwapIsRefusedBeforeItStarts) {
  Ledger l;
  l.plans = l.floor + kMiB;  // one extent charged, little to give
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge();
  l.incoming = 4 * kMiB;
  auto made = rt::MakeRoom([&] { return l.Shortfall(); },
                           [&](std::uint64_t ask) { return l.Reclaim(ask, kMiB); }, kExtent);
  ASSERT_FALSE(made);
  EXPECT_NE(made.error().find("short"), std::string::npos);
  EXPECT_GT(l.Shortfall(), 0U);  // refused: the caller does not swap
  // Nothing to reclaim at all: refused at once.
  Ledger empty;
  empty.budget = 10 * kMiB;
  empty.incoming = 11 * kMiB;
  int calls = 0;
  auto none = rt::MakeRoom([&] { return empty.Shortfall(); },
                           [&](std::uint64_t) {
                             ++calls;
                             return std::uint64_t{0};
                           },
                           kExtent);
  EXPECT_FALSE(none);
  EXPECT_EQ(calls, 1);
  // Room already there: nothing asked.
  Ledger roomy;
  roomy.budget = 10 * kMiB;
  roomy.incoming = 5 * kMiB;
  auto fits = rt::MakeRoom([&] { return roomy.Shortfall(); },
                           [](std::uint64_t) { return std::uint64_t{0}; }, kExtent);
  ASSERT_TRUE(fits);
  EXPECT_EQ(fits->reclaims, 0);
}

TEST(SwapRoom, TheAttemptsAreBounded) {
  Ledger l;
  l.plans = l.floor + (64 * kMiB);
  l.others = 100 * kMiB;
  l.budget = l.others + l.Charge();
  l.incoming = 3 * kMiB;
  // A reclaimer that reports freed bytes but drops nothing (a ledger that
  // never moves): refused after the bounded attempts.
  int calls = 0;
  auto made = rt::MakeRoom([&] { return l.Shortfall(); },
                           [&](std::uint64_t ask) {
                             ++calls;
                             return ask;
                           },
                           kExtent, 3);
  EXPECT_FALSE(made);
  EXPECT_EQ(calls, 3);
}

}  // namespace
