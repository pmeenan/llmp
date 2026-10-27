// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Linux half of the capability probe that `llmp doctor` reports
// (D-026, D-063): the kernel, glibc, memory totals, the hard-link
// protection that D-063's checkpoint store relies on, and RDMA devices with
// this user's access to them. It only reads; it changes nothing on the host.
//
// Files under /proc, /sys and /dev are read beneath `root`, which is "/"
// for this host; tests pass a fake tree. The kernel release, glibc version
// and page size always come from the running process.

#ifndef LLMP_PLATFORM_HOST_PROBE_H_
#define LLMP_PLATFORM_HOST_PROBE_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "base/report.h"

namespace llmp::platform {

struct KernelModule {
  bool loaded = false;
  // What /sys/module/<name>/version says, or empty.
  std::string version;
};

// Whether a kernel module (or a built-in one with parameters) is present.
KernelModule FindKernelModule(const std::filesystem::path& root, std::string_view name);

// A /proc/meminfo value in bytes, such as MemTotal's.
std::optional<std::uint64_t> MeminfoBytes(std::string_view meminfo, std::string_view key);

// The pages Linux's per-CPU free lists hold (the sum of every zone's
// pagesets' `count:` in /proc/zoneinfo text), or none if it lists none.
std::optional<std::uint64_t> PerCpuListPages(std::string_view zoneinfo);

// The memory this host could give a new allocation without swapping, in
// bytes, now (Linux: MemAvailable plus the free pages its per-CPU lists
// hold, which MemAvailable does not count (RE-024): after tens of GB were
// allocated and freed they reached 12.4 GB on a GB10, which refused a
// serving start (RE-052); macOS would count free, inactive and
// purgeable pages from host_statistics64; Windows, GlobalMemoryStatusEx's
// ullAvailPhys). Unknown if the system does not say.
std::optional<std::uint64_t> AvailableMemoryBytes();

// Adds the `host` and `RDMA` sections, with their warnings, to report.
void DescribeHost(const std::filesystem::path& root, base::Report& report);

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_HOST_PROBE_H_
