// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CPU-only profile's device probe: no device provider (D-026).

#include <filesystem>
#include <string>

#include "base/report.h"
#include "providers/device_probe.h"

namespace llmp::providers {

void DescribeDevices(const std::filesystem::path& /*root*/, base::Report& report) {
  report.AddSection("devices").Add("provider", "none (a CPU-only build, D-026)");
}

std::string DeviceIdentity(const std::filesystem::path& /*root*/) { return "none"; }

}  // namespace llmp::providers
