// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>

// This control's main reads PR_GET_DUMPABLE and sets PR_SET_DUMPABLE=0.
// Preserve real reads/the parent setting, but refuse the re-exec child's set.
extern "C" int __wrap_prctl(int option, ...) {
  if (option == PR_SET_DUMPABLE && !GTEST_FLAG_GET(internal_run_death_test).empty()) {
    errno = EPERM;
    return -1;
  }
  return static_cast<int>(syscall(SYS_prctl, option, 0UL, 0UL, 0UL, 0UL));
}

TEST(MainSetupFailure, CannotSatisfyDeath) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(std::abort(), "");
}

TEST(MainSetupFailure, CannotSatisfySuccessfulExit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(0), testing::ExitedWithCode(0), "");
}
