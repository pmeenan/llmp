// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Room for a swap's incoming model (runtime/swap_room.h), against a ledger
// like the node's: plans and graphs charged past a floor as one extent of
// whole 2 MiB steps, so reclaiming N bytes of them may drop occupancy by
// less than N. A swap short by less than one extent is made room for, or
// refused before it starts; never started short.

#include "runtime/swap_room.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace {

namespace rt = jitllm::runtime;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kExtent = 2 * kMiB;

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
