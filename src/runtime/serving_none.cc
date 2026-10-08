// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The serving commands in a build without CUDA: nothing to serve with.

#include <cstdio>
#include <string_view>

#include "runtime/commands.h"
#include "runtime/runtime.h"

namespace llmp::runtime {

int RunServing(const config::NodeConfig& /*config*/, const config::RuntimeRoles& /*roles*/,
               const CommandOptions& command, std::FILE* /*out*/, std::FILE* log) {
  if (command.command == Command::kService) {
    // Configured models it cannot serve: a restart would only repeat this.
    constexpr std::string_view kLine =
        "llmp-runtime: refusing to start: this build has no GPU support, so it cannot serve "
        "the configured models\n";
    (void)std::fwrite(kLine.data(), 1, kLine.size(), log);
    return kExitRefused;
  }
  constexpr std::string_view kLine =
      "llmp-runtime: this build has no GPU support, so no serving commands\n";
  (void)std::fwrite(kLine.data(), 1, kLine.size(), log);
  return kExitFailure;
}

}  // namespace llmp::runtime
