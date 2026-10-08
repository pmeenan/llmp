// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Negative control: CTest must report this failure.
#include <gtest/gtest.h>

namespace {

TEST(NegativeControl, Fails) { EXPECT_EQ(1, 2) << "deliberate failure"; }

}  // namespace
