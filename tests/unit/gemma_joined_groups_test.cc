// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "engine/gemma4_wave.h"
#include "runtime/gemma_wave.h"

namespace rt = llmp::runtime;
using Status = std::expected<void, std::string>;
TEST(GemmaJoinedGroups, LateCleanRefusalPreservesCompletedFirstGroupAndItsPublication) {
  std::array<Status, 12> result{};
  std::array<unsigned, 12> position{};
  std::array<bool, 12> prefix{};
  auto run = [&](std::size_t first, std::size_t count) -> Status {
    if (first == 8) return std::unexpected("state capacity");
    for (std::size_t i = first; i < first + count; ++i) ++position[i];
    return {};
  };
  const auto ran = rt::RunGemmaGroups(
      12, 8, run, [] { return true; },
      [&](std::size_t first, std::size_t count, const std::string& e) {
        for (std::size_t i = first; i < first + count; ++i) {
          result[i] = std::unexpected(e);
          prefix[i] = true;
        }
      });
  ASSERT_TRUE(ran);
  unsigned publications = 0;
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (result[i]) ++publications;
    EXPECT_EQ(position[i], i < 8 ? 1U : 0U);
    EXPECT_EQ(prefix[i], i >= 8);
  }
  EXPECT_EQ(publications, 8U);
}
TEST(GemmaJoinedGroups, EarlyCleanRefusalLeavesLaterPeersEligible) {
  std::vector<std::pair<std::size_t, std::size_t>> calls;
  std::array<bool, 12> failed{};
  auto ran = rt::RunGemmaGroups(
      12, 8,
      [&](std::size_t first, std::size_t count) -> Status {
        calls.emplace_back(first, count);
        return first == 0 ? Status(std::unexpected("plan capacity")) : Status{};
      },
      [] { return true; },
      [&](std::size_t first, std::size_t count, const std::string&) {
        for (std::size_t i = first; i < first + count; ++i) failed[i] = true;
      });
  ASSERT_TRUE(ran);
  EXPECT_EQ(calls, (std::vector<std::pair<std::size_t, std::size_t>>{{0, 8}, {8, 4}}));
  for (std::size_t i = 0; i < failed.size(); ++i) EXPECT_EQ(failed[i], i < 8);
}
TEST(GemmaJoinedGroups, WholeCohortFailureStopsBeforePublicationOrLaterDispatch) {
  std::vector<std::size_t> calls;
  bool group_refusal = false;
  auto ran = rt::RunGemmaGroups(
      12, 8,
      [&](std::size_t first, std::size_t) -> Status {
        calls.push_back(first);
        return first == 8 ? Status(std::unexpected("unproven completion")) : Status{};
      },
      [] { return false; },
      [&](std::size_t, std::size_t, const std::string&) { group_refusal = true; });
  ASSERT_FALSE(ran);
  EXPECT_EQ(ran.error(), "unproven completion");
  EXPECT_EQ(calls, (std::vector<std::size_t>{0, 8}));
  EXPECT_FALSE(group_refusal);
}
TEST(GemmaJoinedGroups, InvalidEnvelopeNeverDispatches) {
  for (const auto [owners, limit] :
       {std::pair<std::size_t, std::size_t>{0, 8}, {13, 8}, {12, 0}, {12, 13}}) {
    bool dispatched = false;
    auto ran = rt::RunGemmaGroups(
        owners, limit,
        [&](std::size_t, std::size_t) -> Status {
          dispatched = true;
          return {};
        },
        [] { return true; }, [](std::size_t, std::size_t, const std::string&) {});
    EXPECT_FALSE(ran);
    EXPECT_FALSE(dispatched);
  }
}

TEST(GemmaJoinedGroups, OrdinaryWholeTwelveAndInvariantEightKeepDistinctCallerBounds) {
  namespace en = llmp::engine;
  static_assert(en::kGemma4InvariantWaveRows == 8);
  for (const bool invariant : {false, true}) {
    std::vector<std::pair<std::size_t, std::size_t>> calls;
    auto ran = rt::RunGemmaGroups(
        12, en::Gemma4WaveRows(invariant),
        [&](std::size_t first, std::size_t count) -> Status {
          calls.emplace_back(first, count);
          return {};
        },
        [] { return true; }, [](std::size_t, std::size_t, const std::string&) {});
    ASSERT_TRUE(ran);
    EXPECT_EQ(calls, invariant ? (std::vector<std::pair<std::size_t, std::size_t>>{{0, 8}, {8, 4}})
                               : (std::vector<std::pair<std::size_t, std::size_t>>{{0, 12}}));
  }
  unsigned refused = 0;
  auto ran = rt::RunGemmaGroups(
      12, en::Gemma4WaveRows(false),
      [](std::size_t, std::size_t) -> Status { return std::unexpected("state capacity"); },
      [] { return true; },
      [&](std::size_t first, std::size_t count, const std::string&) {
        EXPECT_EQ(first, 0U);
        refused += static_cast<unsigned>(count);
      });
  EXPECT_TRUE(ran);
  EXPECT_EQ(refused, 12U);
}
