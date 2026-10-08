// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <span>
#include <string>
#include <string_view>

#include "base/build_info.h"
#include "base/report.h"
#include "cli/doctor.h"

namespace llmp::cli {
namespace {

constexpr std::string_view kUsage =
    "Usage: llmp doctor [--config FILE]\n"
    "       llmp --version\n"
    "       llmp --help\n";

constexpr std::string_view kHelp =
    "Usage: llmp doctor [--config FILE]\n"
    "       llmp --version\n"
    "       llmp --help\n"
    "\n"
    "llmpalooza, a just-in-time LLM inference engine.\n"
    "\n"
    "  doctor     report this build, the host, the node's configuration and\n"
    "             storage, the GPU driver and devices, and RDMA; exit 1 if\n"
    "             this host cannot run this build\n"
    "    --config FILE  the node's configuration (default\n"
    "             /etc/llmp/llmp.toml and /etc/llmp/llmp.d/)\n"
    "  --version  print the version, commit, license profile, SDK and target\n"
    "  --help     print this help\n";

// Writes all of text and flushes it; false if either fails.
bool WriteAll(std::FILE* stream, std::string_view text) {
  return std::fwrite(text.data(), 1, text.size(), stream) == text.size() &&
         std::fflush(stream) == 0;
}

int Print(std::FILE* out, std::FILE* err, std::string_view text) {
  if (WriteAll(out, text)) {
    return kExitOk;
  }
  WriteAll(err, "llmp: cannot write to standard output\n");
  return kExitFailure;
}

int UsageError(std::FILE* err, std::string_view message) {
  WriteAll(err, std::format("llmp: {}\n{}", message, kUsage));
  return kExitUsage;
}

}  // namespace

std::string VersionText(const base::BuildInfo& info) {
  std::string commit = info.commit.empty() ? std::string("unknown") : std::string(info.commit);
  if (info.modified) {
    commit += " (with uncommitted changes)";
  }
  return std::format("llmp {}\ncommit: {}\nlicense profile: {}\nSDK: {}\ntarget: {}\n",
                     info.version, commit, info.license_profile, info.sdk, info.target);
}

int Run(std::span<const std::string_view> args, std::FILE* out, std::FILE* err) {
  if (args.empty()) {
    return UsageError(err, "no command or option given");
  }
  const std::string_view command = args.front();
  if (command != "doctor" && command != "--version" && command != "--help" && command != "-h") {
    return UsageError(err, std::format("unknown command or option '{}'", command));
  }
  DoctorOptions options;
  std::size_t used = 1;
  if (command == "doctor" && args.size() > 1 && args[1] == "--config") {
    if (args.size() < 3 || args[2].empty()) {
      return UsageError(err, "--config needs a file");
    }
    options.config = std::filesystem::path(args[2]);
    used = 3;
  }
  if (args.size() > used) {
    return UsageError(err, std::format("unexpected argument '{}' after {}", args[used], command));
  }
  if (command == "doctor") {
    // doctor compiles nothing, so the driver has no reason to write its JIT
    // cache under $HOME; this keeps the command from writing files. Nothing
    // has started a thread yet.
    (void)::setenv("CUDA_CACHE_DISABLE", "1", 1);  // NOLINT(concurrency-mt-unsafe)
    base::Report report;
    if (!Doctor("/", options, report,
                [out](std::string_view text) { return WriteAll(out, text); })) {
      WriteAll(err, "llmp: cannot write to standard output\n");
      return kExitFailure;
    }
    return report.problems.empty() ? kExitOk : kExitFailure;
  }
  if (command == "--version") {
    return Print(out, err, VersionText(base::GetBuildInfo()));
  }
  return Print(out, err, kHelp);
}

}  // namespace llmp::cli
