// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// `llmp doctor`: the capability probe (D-026, D-063, D-072, D-073). It
// reports what this binary is, the host, the node's configuration and
// storage roles, the device driver and devices, and RDMA, and fails if
// this host cannot run this build as designed. It only reads and queries:
// it writes no file and allocates no device memory (the driver's own
// initialization may load its kernel modules; cuda_probe.h). So it does not
// run the direct-I/O probe, which writes; the runtime does, at every start.

#ifndef LLMP_CLI_DOCTOR_H_
#define LLMP_CLI_DOCTOR_H_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "base/build_info.h"
#include "base/report.h"

namespace llmp::cli {

// Adds the `build` section: the version, commit, license profile, SDK and
// target, and the compiler and C++ runtime that built this binary.
void DescribeBuild(const base::BuildInfo& info, base::Report& report);

struct DoctorOptions {
  // The configuration's main file, as `--config` names it: a development
  // run's, whose files and roles its invoking user owns. Without it,
  // doctor reads the packaged default and judges ownership against the
  // `llmp` account (or the invoking user, if there is none).
  std::optional<std::filesystem::path> config;
};

// Adds the `configuration` and `storage` sections: which files form the
// node's configuration and whether it is valid, the enrollment anchor,
// and the storage roles (config/storage_roles.h).
void DescribeConfiguration(const DoctorOptions& options, base::Report& report);

// Runs every probe into report, reading /proc, /sys and /dev beneath root
// ("/"), and hands write() the report text in two parts: the build, host,
// configuration and storage sections before the device probe starts, so a
// driver that hangs still leaves them, and then the rest. False if a write
// fails.
bool Doctor(const std::filesystem::path& root, const DoctorOptions& options, base::Report& report,
            const std::function<bool(std::string_view)>& write);

// The report as `llmp doctor` prints it: each section's facts, then the
// problems and warnings, then a summary line. Control characters in the
// report's text are escaped (\xNN), so a value cannot forge a line.
std::string DoctorText(const base::Report& report);

// DoctorText()'s parts: the sections from `first` on, and the rest.
std::string SectionsText(const base::Report& report, std::size_t first);
std::string SummaryText(const base::Report& report);

// text with each control character written as \xNN.
std::string Printable(std::string_view text);

}  // namespace llmp::cli

#endif  // LLMP_CLI_DOCTOR_H_
