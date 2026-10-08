// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// C++23 library probe for the GCC 16.2 runtime linked statically.
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <flat_map>
#include <flat_set>
#include <generator>
#include <mdspan>
#include <print>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>
#include <version>

namespace {

std::generator<int> count(int n) {
  for (int i = 0; i < n; ++i) {
    co_yield i;
  }
}

int check(bool ok, const char* what) {
  if (!ok) {
    std::println(stderr, "FAIL: {}", what);
    std::exit(1);
  }
  return 0;
}

}  // namespace

int main() {
  int total = 0;
  for (const int v : count(4)) {
    total += v;
  }
  check(total == 6, "generator");

  const std::expected<int, int> e = 2;
  check(e.and_then([](int x) -> std::expected<int, int> { return x * 2; }).value() == 4,
        "expected");

  std::flat_map<std::string, int> map{{"b", 2}, {"a", 1}};
  check(map.begin()->first == "a", "flat_map");
  const std::flat_set<int> set{3, 1, 2};
  check(*set.begin() == 1, "flat_set");

  std::vector<int> storage(6);
  const std::mdspan grid(storage.data(), 2, 3);
  grid[1, 2] = 7;
  check(storage[5] == 7, "mdspan");

  const auto squares = std::views::iota(1, 4) |
                       std::views::transform([](int x) { return x * x; }) |
                       std::ranges::to<std::vector>();
  check(squares.size() == 3 && squares[2] == 9, "ranges::to");

  // Exceptions must unwind through the statically linked libgcc/libstdc++.
  bool caught = false;
  try {
    throw std::runtime_error("unwind");
  } catch (const std::runtime_error&) {
    caught = true;
  }
  check(caught, "exception");

  std::println("PASS glibcxx={} cpp_lib_print={} cpp_lib_flat_map={} cpp_lib_mdspan={}",
               __GLIBCXX__, __cpp_lib_print, __cpp_lib_flat_map, __cpp_lib_mdspan);
  return 0;
}
