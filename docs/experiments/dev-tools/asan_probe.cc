// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Symbolizer probe: a deliberate heap overflow that AddressSanitizer reports.
#include <cstddef>
#include <vector>

namespace {

[[gnu::noinline]] int overflow_probe(std::size_t index) {
  const std::vector<int> values(4);
  return values[index];
}

}  // namespace

int main(int argc, char** /*argv*/) { return overflow_probe(static_cast<std::size_t>(argc) + 3U); }
