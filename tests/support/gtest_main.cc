// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-death-test-internal.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>

#include "platform/crash_policy.h"

int main(int argc, char** argv) {
  testing::InitGoogleMock(&argc, argv);
  // Intentional death tests must not invoke host crash collection. A zero
  // core limit does not suppress piped collectors, and QEMU/ASan fatal-signal
  // collection can take minutes. Re-executed death-test
  // children run main again, since exec resets the dumpable attribute.
  if (const auto marked = llmp::platform::MarkNonDumpable(); !marked) {
    std::fprintf(stderr, "cannot disable test core dumps: %s\n", marked.error().c_str());
    // GTest redirects a death child's stderr; keep startup failures visible
    // to the outer test harness as well as its internal error pipe.
    std::fputs("test startup: cannot disable core dumps\n", stdout);
    std::fflush(stdout);
    // An exit code could satisfy either EXPECT_DEATH or EXPECT_EXIT. Report
    // GTest's pinned 'I' (internal error) pipe record to a re-exec parent so
    // setup failure cannot satisfy either assertion's exit predicate.
    if (const auto* flag = testing::internal::ParseInternalRunDeathTestFlag()) {
      constexpr char error[] = "Icannot disable test core dumps";
      ssize_t written;
      do {
        written = write(flag->write_fd(), error, sizeof(error) - 1);
      } while (written < 0 && errno == EINTR);
      std::_Exit(1);
    }
    return 1;
  }
  return RUN_ALL_TESTS();
}
