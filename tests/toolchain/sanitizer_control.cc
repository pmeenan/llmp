// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Deliberate defects that a sanitizer build must report (D-061), one per
// mode given as the only argument; `clean` has none. check_sanitizer.cmake
// runs the modes the build's sanitizers cover and requires each report, so
// a build whose sanitizers are not really active fails instead of passing
// every other test.

#include <climits>
#include <cstdio>
#include <print>
#include <string_view>
#include <thread>

namespace {

// Volatile, so the optimizer can neither see nor remove the defects below:
// a heap block that never escapes may be elided with its accesses.
volatile int g_index = 8;
volatile int g_int_max = INT_MAX;
char* volatile g_heap = nullptr;
int g_shared = 0;

int HeapOverflow() {
  g_heap = new char[8]{};
  g_heap[g_index] = 1;  // One past the end.
  delete[] g_heap;
  return 0;
}

int Leak() {
  g_heap = new char[64];
  g_heap = nullptr;  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks): the defect under test.
  return 0;
}

int SignedOverflow() {
  int value = g_int_max;
  ++value;  // Undefined: INT_MAX + 1.
  std::println("{}", value);
  return 0;
}

int Race() {
  std::thread writer([] { ++g_shared; });
  ++g_shared;  // Unsynchronized with the writer's increment.
  writer.join();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc == 2 ? argv[1] : "";
  if (mode == "clean") {
    return 0;
  }
  if (mode == "heap-overflow") {
    return HeapOverflow();
  }
  if (mode == "leak") {
    return Leak();
  }
  if (mode == "signed-overflow") {
    return SignedOverflow();
  }
  if (mode == "race") {
    return Race();
  }
  std::println(stderr, "usage: sanitizer_control clean|heap-overflow|leak|signed-overflow|race");
  return 2;
}
