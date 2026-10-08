// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The `llmp` command: argument handling and output, apart from main() so
// that tests can run it.

#ifndef LLMP_CLI_CLI_H_
#define LLMP_CLI_CLI_H_

#include <cstdio>
#include <span>
#include <string>
#include <string_view>

#include "base/build_info.h"

namespace llmp::cli {

// Exit statuses. `doctor` fails when it finds a problem.
inline constexpr int kExitOk = 0;
inline constexpr int kExitFailure = 1;
inline constexpr int kExitUsage = 2;

// What `llmp --version` prints: `llmp <version>` on the first line, then
// the commit, license profile, SDK and target (D-062).
std::string VersionText(const base::BuildInfo& info);

// Runs the command with its arguments (argv without the program name),
// writing to out and err; returns the exit status. A write that fails,
// such as to a full disk or a closed pipe, fails the command.
int Run(std::span<const std::string_view> args, std::FILE* out, std::FILE* err);

}  // namespace llmp::cli

#endif  // LLMP_CLI_CLI_H_
