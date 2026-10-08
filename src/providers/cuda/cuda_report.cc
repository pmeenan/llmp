// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/report.h"
#include "providers/cuda/cuda_facts.h"

namespace llmp::providers::cuda {
namespace {

std::string YesNo(const std::optional<bool>& value, std::string_view yes, std::string_view no) {
  if (!value) {
    return "unknown";
  }
  return std::string(*value ? yes : no);
}

std::string GranularityText(const CudaGranularity& granularity) {
  if (!granularity.error.empty()) {
    return "unknown: " + granularity.error;
  }
  return std::format("granularity {} minimum, {} recommended",
                     base::FormatBytes(granularity.minimum),
                     base::FormatBytes(granularity.recommended));
}

// Whether the build has SASS for exactly this compute capability: sm_121
// is 12.1, and a minor version above 9 matches nothing.
bool Targeted(const CudaFacts& facts, const CudaDeviceFacts& device) {
  return device.error.empty() && device.minor >= 0 && device.minor < 10 &&
         std::ranges::any_of(
             facts.built_architectures,
             [&](int architecture) {
               return architecture / 10 == device.major && architecture % 10 == device.minor;
             });
}

// Why a granularity cannot back D-056's 2 MiB paging chunks, or empty.
std::string GranularityFault(const CudaGranularity& granularity) {
  if (!granularity.error.empty()) {
    return granularity.error;
  }
  const std::uint64_t minimum = granularity.minimum;
  const std::uint64_t recommended = granularity.recommended;
  if (minimum == 0 || !std::has_single_bit(minimum) || minimum > kPagingChunkBytes ||
      !std::has_single_bit(recommended) || recommended < minimum) {
    return std::format(
        "a minimum of {} and a recommended {} cannot back 2 MiB paging chunks "
        "(D-056)",
        base::FormatBytes(minimum), base::FormatBytes(granularity.recommended));
  }
  return {};
}

std::string ComputeModeText(const std::optional<CudaComputeMode>& mode) {
  if (!mode) {
    return "unknown";
  }
  switch (*mode) {
    case CudaComputeMode::kDefault:
      return "default";
    case CudaComputeMode::kExclusiveProcess:
      return "exclusive process";
    case CudaComputeMode::kProhibited:
      return "prohibited";
    case CudaComputeMode::kOther:
      break;
  }
  return "other";
}

std::string ClassText(const std::optional<CudaDeviceClass>& device_class) {
  if (!device_class) {
    return "unknown";
  }
  return *device_class == CudaDeviceClass::kUnified
             ? "unified (integrated: one memory budget with the host)"
             : "discrete (its own device memory)";
}

std::string ArchitecturesText(const std::vector<int>& architectures) {
  std::string text;
  for (const int architecture : architectures) {
    text += std::format("{}sm_{}", text.empty() ? "" : ", ", architecture);
  }
  return text.empty() ? "none" : text;
}

void DescribeDriver(const CudaFacts& facts, base::Report& report) {
  base::ReportSection& driver = report.AddSection("NVIDIA driver");
  std::string module = "not loaded";
  if (facts.nvidia.loaded) {
    module = facts.nvidia.version.empty() ? "loaded, version unknown" : facts.nvidia.version;
  }
  driver.Add("kernel module", module);
  driver.Add("library", facts.library);
  std::string api = "unknown";
  if (facts.driver_version > 0) {
    api = CudaVersionText(facts.driver_version);
    if (facts.driver_version / 1000 == facts.built_version / 1000 &&
        facts.driver_version < facts.built_version) {
      api += std::format(", older than this build's {}: CUDA minor-version compatibility",
                         CudaVersionText(facts.built_version));
    }
  }
  driver.Add("CUDA driver API", api);
  driver.Add("CUDA toolkit (this build)", CudaVersionText(facts.built_version));
  driver.Add("GPU code (this build)", ArchitecturesText(facts.built_architectures));
  // llmpalooza reads files into host VMM itself and does not assume native GDS
  // (D-004, D-034); the module's presence is reported for comparison.
  driver.Add("GPUDirect Storage",
             facts.nvidia_fs.loaded ? "nvidia_fs loaded" : "nvidia_fs not loaded (no native GDS)");
}

void DescribeDevice(const CudaFacts& facts, const CudaDeviceFacts& device, base::Report& report) {
  base::ReportSection& section =
      report.AddSection(std::format("GPU {}: {}", device.ordinal, device.name));
  if (!device.error.empty()) {
    section.Add("error", device.error);
    return;
  }
  const bool targeted = Targeted(facts, device);
  section.Add("compute capability",
              std::format("{}.{} (sm_{}{}{})", device.major, device.minor, device.major,
                          device.minor, targeted ? "" : ", which this build has no code for"));
  section.Add("class", ClassText(DeviceClassOf(device)));
  section.Add("memory", device.memory_bytes ? base::FormatBytes(*device.memory_bytes) : "unknown");
  section.Add("compute mode", ComputeModeText(device.compute_mode));
  section.Add("virtual memory management", YesNo(device.vmm, "supported", "not supported"));
  section.Add("device-local backing",
              device.device_local ? GranularityText(*device.device_local) : "not queried");
  std::string host = YesNo(device.host_numa_vmm, "supported", "not supported");
  if (device.host_numa) {
    host = GranularityText(*device.host_numa);
  } else if (device.host_numa_vmm == true) {
    host = device.host_numa_id ? "supported, but the device has no host NUMA node"
                               : "supported, but the device's host NUMA node is unknown";
  }
  const std::string host_key = device.host_numa_id && *device.host_numa_id >= 0
                                   ? std::format("host NUMA node {} backing", *device.host_numa_id)
                                   : std::string("host NUMA backing");
  section.Add(host_key, host);
  section.Add("GPUDirect RDMA", YesNo(device.gpu_direct_rdma, "supported", "not supported"));
  // Multi-GPU hosts are out of scope (D-082): llmpalooza uses GPU 0 alone, so
  // only it is judged.
  if (device.ordinal != kUsedDevice) {
    section.Add("use", "none: llmpalooza uses GPU 0 only (D-082)");
    return;
  }
  if (!targeted) {
    return;
  }
  // The build's code for this architecture assumes its class of memory:
  // the GB10's unified budget (D-004) or a discrete GPU's own (D-082).
  const int architecture = (device.major * 10) + device.minor;
  if (const auto device_class = DeviceClassOf(device); !device_class) {
    report.problems.push_back(std::format(
        "GPU {} does not report whether it is integrated, so whether its memory is unified or "
        "discrete is unknown (D-082)",
        device.ordinal));
  } else if (*device_class != TargetClass(architecture)) {
    report.problems.push_back(
        TargetClass(architecture) == CudaDeviceClass::kUnified
            ? std::format("GPU {} is sm_{} but discrete, and this build's sm_{} code is for the "
                          "GB10's unified memory (D-004)",
                          device.ordinal, architecture, architecture)
            : std::format("GPU {} is sm_{} but integrated, and this build targets sm_{} as a "
                          "discrete GPU (D-082)",
                          device.ordinal, architecture, architecture));
  }
  if (device.compute_mode == CudaComputeMode::kProhibited) {
    report.problems.push_back(std::format(
        "GPU {} is in the prohibited compute mode, so no process can use it", device.ordinal));
  }
  // D-006: every managed allocation is explicit VMM backing.
  if (device.vmm != true) {
    report.problems.push_back(std::format(
        "GPU {} does not report CUDA virtual memory management, which llmpalooza requires (D-006)",
        device.ordinal));
  } else if (!device.device_local) {
    report.problems.push_back(
        std::format("GPU {}: device-local VMM granularity not queried", device.ordinal));
  } else if (const std::string fault = GranularityFault(*device.device_local); !fault.empty()) {
    report.problems.push_back(
        std::format("GPU {}: device-local VMM backing: {}", device.ordinal, fault));
  }
  // D-034, D-081: every direct read of weights and state lands in host VMM
  // backing (the landing zone), from which the GPU copies each extent into
  // device VMM, on both classes; there is no other path (owner, 2026-09-24).
  std::string why;
  if (device.host_numa) {
    why = GranularityFault(*device.host_numa);
  } else if (!device.host_numa_vmm) {
    why = "support is unknown";
  } else if (!*device.host_numa_vmm) {
    why = "the device does not support it";
  } else {
    why = device.host_numa_id ? "the device has no host NUMA node"
                              : "the device's host NUMA node is unknown";
  }
  if (!why.empty()) {
    report.problems.push_back(
        std::format("GPU {}: no host-backed VMM ({}), which direct file reads need as their "
                    "landing zone (D-034, D-081)",
                    device.ordinal, why));
  }
}

}  // namespace

std::vector<int> ParseCudaArchitectures(std::string_view architectures) {
  std::vector<int> numbers;
  while (!architectures.empty()) {
    const std::size_t comma = architectures.find(',');
    const std::string_view entry = architectures.substr(0, comma);
    architectures =
        comma == std::string_view::npos ? std::string_view() : architectures.substr(comma + 1);
    // A `-virtual` entry is PTX only, which this build never relies on
    // (D-011): only `-real` and bare entries put SASS in the binary.
    if (entry.ends_with("-virtual")) {
      continue;
    }
    int number = 0;
    const auto [end, error] = std::from_chars(entry.data(), entry.data() + entry.size(), number);
    if (error == std::errc() && end != entry.data() && number > 0 &&
        !std::ranges::contains(numbers, number)) {
      numbers.push_back(number);
    }
  }
  return numbers;
}

std::string CudaVersionText(int version) {
  return std::format("{}.{}", version / 1000, version % 1000 / 10);
}

std::optional<CudaDeviceClass> DeviceClassOf(const CudaDeviceFacts& device) {
  if (!device.integrated) {
    return std::nullopt;
  }
  return *device.integrated ? CudaDeviceClass::kUnified : CudaDeviceClass::kDiscrete;
}

CudaDeviceClass TargetClass(int architecture) {
  return architecture == 121 ? CudaDeviceClass::kUnified : CudaDeviceClass::kDiscrete;
}

void DescribeCuda(const CudaFacts& facts, base::Report& report) {
  DescribeDriver(facts, report);
  // A driver older than the toolkit's major version cannot run its code;
  // within a major version, minor-version compatibility covers it.
  if (facts.driver_version <= 0) {
    report.problems.push_back(std::format(
        "the driver did not report its CUDA version, so whether it can run this build's CUDA {} "
        "is unknown",
        CudaVersionText(facts.built_version)));
  } else if (facts.driver_version / 1000 < facts.built_version / 1000) {
    report.problems.push_back(
        std::format("the driver supports CUDA {}, but this build needs a driver for CUDA {}",
                    CudaVersionText(facts.driver_version), facts.built_version / 1000));
  }
  if (!facts.init_error.empty()) {
    report.problems.push_back("the CUDA driver did not initialize: " + facts.init_error);
    return;
  }
  for (const CudaDeviceFacts& device : facts.devices) {
    DescribeDevice(facts, device, report);
  }
  if (std::cmp_greater(facts.device_count, facts.devices.size())) {
    report.problems.push_back(
        std::format("the CUDA driver reports {} GPUs; doctor checks at most {}", facts.device_count,
                    kMaxProbedDevices));
  }
  if (facts.device_count > 1) {
    report.warnings.push_back(
        std::format("the CUDA driver reports {} GPUs; llmpalooza uses only GPU 0, which "
                    "CUDA_VISIBLE_DEVICES selects (D-082)",
                    facts.device_count));
  }
  if (facts.devices.empty()) {
    report.problems.emplace_back("the CUDA driver reports no GPU");
  } else if (std::ranges::none_of(facts.devices, [&](const CudaDeviceFacts& device) {
               return Targeted(facts, device);
             })) {
    report.problems.push_back(std::format("no GPU here is one this build has code for ({})",
                                          ArchitecturesText(facts.built_architectures)));
  } else if (facts.devices.front().ordinal != kUsedDevice ||
             !Targeted(facts, facts.devices.front())) {
    report.problems.push_back(
        std::format("GPU 0 is not one this build has code for ({}), and llmpalooza uses only GPU "
                    "0: select one it "
                    "has code for with CUDA_VISIBLE_DEVICES (D-082)",
                    ArchitecturesText(facts.built_architectures)));
  }
}

}  // namespace llmp::providers::cuda
