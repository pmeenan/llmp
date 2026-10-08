// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "cli/doctor.h"

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "base/build_info.h"
#include "base/report.h"
#include "config/node_config.h"
#include "config/storage_roles.h"
#include "platform/host_probe.h"
#include "providers/device_probe.h"

namespace llmp::cli {
namespace {

// A local account's uid, if there is one.
std::optional<uid_t> AccountUid(const char* name) {
  std::array<char, 16384> buffer{};
  passwd entry{};
  passwd* found = nullptr;
  if (::getpwnam_r(name, &entry, buffer.data(), buffer.size(), &found) != 0 || found == nullptr) {
    return std::nullopt;
  }
  return entry.pw_uid;
}

std::string Count(std::size_t count, std::string_view noun) {
  return std::format("{} {}{}", count, noun, count == 1 ? "" : "s");
}

}  // namespace

void DescribeBuild(const base::BuildInfo& info, base::Report& report) {
  base::ReportSection& build = report.AddSection("build");
  build.Add("version", std::string(info.version));
  std::string commit = info.commit.empty() ? std::string("unknown") : std::string(info.commit);
  if (info.modified) {
    commit += " (with uncommitted changes)";
  }
  build.Add("commit", commit);
  build.Add("license profile", std::string(info.license_profile));
  build.Add("SDK", std::string(info.sdk));
  build.Add("target", std::string(info.target));
  build.Add("compiler",
            std::format("Clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__));
  build.Add("C++ runtime",
            std::format("libstdc++ from GCC {}, linked statically (D-060)", _GLIBCXX_RELEASE));
}

void DescribeConfiguration(const DoctorOptions& options, base::Report& report) {
  config::LoadOptions load;
  uid_t trusted = ::geteuid();
  std::filesystem::path writable;
  load.main_file_optional = !options.config;
  if (options.config) {
    load.main_file = *options.config;
  } else if (const auto service = AccountUid("llmp")) {
    // Packaged: the unit's sandbox lets the runtime write only its
    // StateDirectory (packaging/llmp.service).
    trusted = *service;
    writable = config::kDefaultDataDir;
  }
  load.trusted_uid = trusted;
  base::ReportSection& section = report.AddSection("configuration");
  section.Add("runtime's user", std::format("uid {}", trusted));
  auto loaded = config::LoadNodeConfig(load);
  if (!loaded) {
    section.Add(
        "files",
        std::format("{} and {} (not valid)", load.main_file.string(),
                    std::filesystem::path(load.main_file).replace_extension(".d").string()));
    for (const config::Diagnostic& diagnostic : loaded.error()) {
      report.problems.emplace_back("configuration: " + config::FormatDiagnostic(diagnostic));
    }
    return;
  }
  if (loaded->files.empty()) {
    section.Add("files", "none: the built-in defaults, a standalone node");
  } else {
    for (const std::filesystem::path& file : loaded->files) {
      section.Add("file", file.string());
    }
  }
  section.Add("node", loaded->membership
                          ? std::format("cluster member {}", loaded->membership->node_id)
                          : std::string("standalone"));
  if (loaded->membership) {
    report.problems.emplace_back(
        "configuration: this build has no cluster support, so the runtime refuses a "
        "cluster member's configuration (cluster membership arrives in M6a)");
  }
  struct stat anchor{};
  if (::lstat(load.anchor.c_str(), &anchor) == 0) {
    section.Add("enrollment anchor", load.anchor.string() + " (present)");
    report.problems.push_back(
        std::format("the enrollment anchor {} exists, and this build, which has no cluster "
                    "support, refuses to start while it does (D-063)",
                    load.anchor.string()));
  } else {
    section.Add("enrollment anchor", load.anchor.string() + " (absent)");
  }
  config::DescribeStorage(loaded->storage, trusted, writable, report);
}

bool Doctor(const std::filesystem::path& root, const DoctorOptions& options, base::Report& report,
            const std::function<bool(std::string_view)>& write) {
  DescribeBuild(base::GetBuildInfo(), report);
  platform::DescribeHost(root, report);
  DescribeConfiguration(options, report);
  // Out before the driver is touched: a wedged GPU can hang cuInit.
  if (!write(SectionsText(report, 0))) {
    return false;
  }
  const std::size_t printed = report.sections.size();
  providers::DescribeDevices(root, report);
  return write(SectionsText(report, printed) + SummaryText(report));
}

std::string Printable(std::string_view text) { return base::Printable(text); }

std::string SectionsText(const base::Report& report, std::size_t first) {
  std::string text;
  for (std::size_t i = first; i < report.sections.size(); ++i) {
    const base::ReportSection& section = report.sections[i];
    text += Printable(section.title) + "\n";
    for (const base::ReportLine& line : section.lines) {
      text += std::format("  {}: {}\n", Printable(line.key), Printable(line.value));
    }
    text += "\n";
  }
  return text;
}

std::string SummaryText(const base::Report& report) {
  std::string text;
  for (const std::string& problem : report.problems) {
    text += std::format("problem: {}\n", Printable(problem));
  }
  for (const std::string& warning : report.warnings) {
    text += std::format("warning: {}\n", Printable(warning));
  }
  const std::string problems =
      report.problems.empty() ? "no problems" : Count(report.problems.size(), "problem");
  text += std::format("doctor: {}, {}\n", problems, Count(report.warnings.size(), "warning"));
  return text;
}

std::string DoctorText(const base::Report& report) {
  return SectionsText(report, 0) + SummaryText(report);
}

}  // namespace llmp::cli
