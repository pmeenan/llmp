// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/check.h"

#include <unistd.h>

#include <cstdlib>
#include <format>
#include <source_location>
#include <string>
#include <string_view>

namespace llmp::base {

void Fatal(std::string_view what, std::source_location where) {
  const std::string line = std::format("llmp: internal invariant violated: {} ({}:{})\n",
                                       what.substr(0, 200), where.file_name(), where.line());
  (void)::write(STDERR_FILENO, line.data(), line.size());
  std::abort();
}

}  // namespace llmp::base
