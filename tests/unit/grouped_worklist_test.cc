// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/common/grouped_worklist.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
namespace kn = llmp::kernels;
struct Fixture {
  kn::GroupedRouteShape shape{4, 2, 3, 1};
  // Two repeated expert-2 contributions from token zero are distinct slots.
  std::array<std::int32_t, 6> ids{2, 2, 0, 3, 2, 0};
  std::array<std::int32_t, 5> offsets{0, 2, 2, 5, 6};
  std::array<std::int32_t, 6> position{2, 3, 0, 5, 4, 1};
  std::array<std::int32_t, 6> token{1, 2, 0, 0, 2, 1};
  std::array<std::int32_t, 6> expert{0, 0, 2, 2, 2, 3};
  std::array<std::int32_t, 5> out_offsets{};
  std::array<std::int32_t, 6> rows{}, slots{};
  std::array<std::uint8_t, 6> seen{};
  Fixture() {
    out_offsets.fill(-9);
    rows.fill(-9);
    slots.fill(-9);
  }
  kn::CompletedGroupedRoute route() { return {ids, offsets, position, token, expert}; }
  kn::GroupedWorklistStorage storage() { return {out_offsets, rows, slots, seen}; }
  auto Run() { return kn::ValidateGroupedRouteInto(shape, route(), storage()); }
  void Unchanged() {
    EXPECT_TRUE(std::ranges::all_of(out_offsets, [](auto n) { return n == -9; }));
    EXPECT_TRUE(std::ranges::all_of(rows, [](auto n) { return n == -9; }));
    EXPECT_TRUE(std::ranges::all_of(slots, [](auto n) { return n == -9; }));
  }
};
TEST(GroupedWorklistTest, PreservesRepeatedContributionsAndBothInputBases) {
  Fixture f;
  ASSERT_TRUE(f.Run());
  EXPECT_EQ(f.out_offsets, f.offsets);
  EXPECT_EQ(f.rows, f.token);
  const std::array<std::int32_t, 6> expected{2, 5, 0, 1, 4, 3};
  EXPECT_EQ(f.slots, expected);
  f.shape.input_lanes = 2;
  ASSERT_TRUE(f.Run());
  EXPECT_EQ(f.rows, expected);
  EXPECT_EQ(f.slots, expected);
}
TEST(GroupedWorklistTest, MalformedRoutesRefuseBeforeChangingOutputs) {
  for (int bad = 0; bad != 9; ++bad) {
    SCOPED_TRACE(bad);
    Fixture f;
    switch (bad) {
      case 0:
        f.ids[0] = -1;
        break;
      case 1:
        f.ids[0] = 4;
        break;
      case 2:
        f.offsets[0] = 1;
        break;
      case 3:
        f.offsets[3] = 1;
        break;
      case 4:
        f.position[0] = f.position[1];
        break;
      case 5:
        f.position[0] = 6;
        break;
      case 6:
        f.token[2] = 1;
        break;
      case 7:
        f.expert[2] = 0;
        break;
      case 8:
        f.offsets[4] = 5;
        break;
    }
    EXPECT_FALSE(f.Run());
    f.Unchanged();
  }
}
TEST(GroupedWorklistTest, BoundAndExactStoragePrecedeWrites) {
  Fixture f;
  const auto bound = kn::GroupedRouteBound(f.shape);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound->inputs.value(), 116U);
  EXPECT_EQ(bound->outputs.value(), 68U);
  EXPECT_EQ(bound->scratch.value(), 6U);
  auto short_storage = f.storage();
  short_storage.seen = short_storage.seen.first(5);
  EXPECT_FALSE(kn::ValidateGroupedRouteInto(f.shape, f.route(), short_storage));
  f.Unchanged();
  short_storage = f.storage();
  short_storage.rows = f.ids;
  const auto alias = kn::ValidateGroupedRouteInto(f.shape, f.route(), short_storage);
  ASSERT_FALSE(alias);
  EXPECT_EQ(alias.error(), kn::GroupedRouteError::kAlias);
  f.Unchanged();
  EXPECT_FALSE(kn::GroupedRouteBound({4097, 1, 1, 1}));
  EXPECT_FALSE(kn::GroupedRouteBound({512, 65, 1, 1}));
  EXPECT_FALSE(kn::GroupedRouteBound({512, 2, 1, 3}));
  EXPECT_FALSE(kn::GroupedRouteBound({512, 2, std::numeric_limits<std::uint64_t>::max(), 1}));
}
TEST(GroupedWorklistTest, Expert512IsRepresentableAndEmptyDescriptorsRemainIndexed) {
  constexpr std::uint64_t e = 512;
  const kn::GroupedRouteShape shape{e, 2, 1, 1};
  const auto bound = kn::GroupedRouteBound(shape);
  ASSERT_TRUE(bound);
  // Caller allocates only after the checked size; the helper owns no buffers.
  std::vector<std::int32_t> offsets(e + 1, 1), out(e + 1);
  offsets.front() = 0;
  offsets.back() = 2;
  const std::array<std::int32_t, 2> ids{511, 0}, position{1, 0}, token{0, 0}, expert{0, 511};
  std::array<std::int32_t, 2> rows{}, slots{};
  std::array<std::uint8_t, 2> seen{};
  ASSERT_TRUE(kn::ValidateGroupedRouteInto(shape, {ids, offsets, position, token, expert},
                                           {out, rows, slots, seen}));
  EXPECT_EQ(out, offsets);
  EXPECT_EQ(slots, (std::array<std::int32_t, 2>{1, 0}));
}
}  // namespace
