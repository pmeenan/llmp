// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/host_probe.h"

#include <fcntl.h>
#include <gnu/libc-version.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "base/report.h"
#include "platform/files.h"

namespace llmp::platform {
namespace {

namespace fs = std::filesystem;

// "4: ACTIVE" -> "ACTIVE", as ports/<n>/state and phys_state read.
std::string WithoutCode(const std::string& value) {
  const std::size_t colon = value.find(": ");
  return colon == std::string::npos ? value : value.substr(colon + 2);
}

std::string LineOr(const fs::path& path, std::string_view fallback) {
  auto line = ReadFirstLine(path);
  return line && !line->empty() ? *line : std::string(fallback);
}

// Whether this process may open path for reading and writing.
std::string Access(const fs::path& path) {
  if (::faccessat(AT_FDCWD, path.c_str(), R_OK | W_OK, AT_EACCESS) == 0) {
    return "read-write";
  }
  return std::error_code(errno, std::generic_category()).message();
}

void DescribeSystem(const fs::path& root, base::Report& report) {
  base::ReportSection& host = report.AddSection("host");
  utsname names{};
  if (::uname(&names) == 0) {
    host.Add("kernel", std::format("{} {}, {}", names.sysname, names.release, names.machine));
  } else {
    host.Add("kernel", "unknown");
  }
  host.Add("glibc", ::gnu_get_libc_version());
  const long page = ::sysconf(_SC_PAGESIZE);
  host.Add("page size", page > 0 ? base::FormatBytes(static_cast<std::uint64_t>(page)) : "unknown");

  const auto meminfo = ReadSmallFile(root / "proc/meminfo");
  const auto total = meminfo ? MeminfoBytes(*meminfo, "MemTotal") : std::nullopt;
  const auto available = meminfo ? MeminfoBytes(*meminfo, "MemAvailable") : std::nullopt;
  if (total && available) {
    host.Add("memory", std::format("{} total, {} available", base::FormatBytes(*total),
                                   base::FormatBytes(*available)));
  } else {
    host.Add("memory", "unknown");
    report.warnings.push_back(
        meminfo ? "/proc/meminfo has no MemTotal or MemAvailable"
                : std::format("cannot read /proc/meminfo: {}", meminfo.error().message()));
  }

  // D-063: jobs reject hard-linked checkpoint sources, and the kernel's
  // protection keeps users from linking files they cannot write.
  const auto hardlinks = ReadFirstLine(root / "proc/sys/fs/protected_hardlinks");
  if (!hardlinks) {
    host.Add("fs.protected_hardlinks", "unknown");
    report.warnings.push_back(
        std::format("cannot read fs.protected_hardlinks: {}", hardlinks.error().message()));
  } else {
    host.Add("fs.protected_hardlinks", *hardlinks);
    if (*hardlinks != "1") {
      report.warnings.push_back(std::format(
          "fs.protected_hardlinks is {}, not 1 (Ubuntu's default): users can hard-link files "
          "they do not own into the checkpoint store (D-063)",
          *hardlinks));
    }
  }
}

void DescribePort(const fs::path& device, const std::string& port, const std::string& netdevs,
                  const std::string& verbs, base::ReportSection& section) {
  const fs::path dir = device / "ports" / port;
  std::string value = std::format("{}, {}, {}", WithoutCode(LineOr(dir / "state", "unknown state")),
                                  LineOr(dir / "link_layer", "unknown link layer"),
                                  LineOr(dir / "rate", "unknown rate"));
  if (!netdevs.empty()) {
    value += ", " + netdevs;
  }
  if (!verbs.empty()) {
    value += ", " + verbs;
  }
  section.Add(std::format("{} port {}", device.filename().string(), port), std::move(value));
}

void DescribeRdma(const fs::path& root, base::Report& report) {
  base::ReportSection& rdma = report.AddSection("RDMA");
  const auto devices = ListDirectory(root / "sys/class/infiniband");
  if (!devices) {
    if (devices.error() == std::errc::no_such_file_or_directory) {
      rdma.Add("devices", "none");
    } else {
      rdma.Add("devices", "unknown");
      report.warnings.push_back(
          std::format("cannot list /sys/class/infiniband: {}", devices.error().message()));
    }
    return;
  }
  if (devices->empty()) {
    rdma.Add("devices", "none");
    return;
  }
  for (const std::string& name : *devices) {
    const fs::path device = root / "sys/class/infiniband" / name;
    std::string netdevs;
    if (const auto names = ListDirectory(device / "device/net")) {
      for (const std::string& netdev : *names) {
        netdevs += (netdevs.empty() ? "" : " ") + netdev;
      }
    }
    // The verbs device node this user opens for the device (D-063 expects
    // mode 0666 on the Sparks).
    std::string verbs;
    const auto nodes = ListDirectory(device / "device/infiniband_verbs");
    if (!nodes || nodes->empty()) {
      verbs = "no verbs device";
      report.warnings.push_back(std::format(
          "RDMA device {} has no verbs device, so it cannot be used (is ib_uverbs loaded?)", name));
    } else {
      for (const std::string& node : *nodes) {
        const std::string access = Access(root / "dev/infiniband" / node);
        verbs += std::format("{}/dev/infiniband/{} {}", verbs.empty() ? "" : ", ", node, access);
        if (access != "read-write") {
          report.warnings.push_back(std::format(
              "RDMA device {}: this user cannot open /dev/infiniband/{} for reading and writing "
              "({})",
              name, node, access));
        }
      }
    }
    const auto ports = ListDirectory(device / "ports");
    if (!ports || ports->empty()) {
      rdma.Add(name, "no ports");
      continue;
    }
    for (const std::string& port : *ports) {
      DescribePort(device, port, netdevs, verbs, rdma);
    }
  }
  // Access is this process's: run as root, doctor cannot see what the
  // service user could open.
  rdma.Add("device access checked for", std::format("uid {}", ::geteuid()));
  const std::string cm = Access(root / "dev/infiniband/rdma_cm");
  rdma.Add("/dev/infiniband/rdma_cm", cm);
  if (cm != "read-write") {
    report.warnings.push_back(std::format(
        "this user cannot open /dev/infiniband/rdma_cm for reading and writing ({})", cm));
  }
}

}  // namespace

KernelModule FindKernelModule(const std::filesystem::path& root, std::string_view name) {
  const fs::path dir = root / "sys/module" / name;
  std::error_code error;
  if (!fs::is_directory(dir, error)) {
    return {};
  }
  auto version = ReadFirstLine(dir / "version");
  return {.loaded = true, .version = version ? *version : std::string()};
}

std::optional<std::uint64_t> MeminfoBytes(std::string_view meminfo, std::string_view key) {
  while (!meminfo.empty()) {
    const std::size_t end = meminfo.find('\n');
    std::string_view line = meminfo.substr(0, end);
    meminfo = end == std::string_view::npos ? std::string_view() : meminfo.substr(end + 1);
    if (!line.starts_with(key) || line.substr(key.size(), 1) != ":") {
      continue;
    }
    line.remove_prefix(key.size() + 1);
    line.remove_prefix(std::min(line.find_first_not_of(' '), line.size()));
    std::uint64_t value = 0;
    const auto [rest, error] = std::from_chars(line.data(), line.data() + line.size(), value);
    if (error != std::errc() || rest == line.data()) {
      return std::nullopt;
    }
    const std::string_view unit(rest, static_cast<std::size_t>(line.data() + line.size() - rest));
    if (unit.empty()) {
      return value;
    }
    if (unit != " kB" || value > std::numeric_limits<std::uint64_t>::max() / 1024) {
      return std::nullopt;
    }
    return value * 1024;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> AvailableMemoryBytes() {
  const auto meminfo = ReadSmallFile("/proc/meminfo");
  if (!meminfo) {
    return std::nullopt;
  }
  return MeminfoBytes(*meminfo, "MemAvailable");
}

void DescribeHost(const std::filesystem::path& root, base::Report& report) {
  DescribeSystem(root, report);
  DescribeRdma(root, report);
}

}  // namespace llmp::platform
