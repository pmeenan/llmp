// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The device half of the capability probe that `jitllm doctor` reports
// (D-026; docs/architecture.md#providers). The build's device provider
// describes its driver and devices: versions, memory, VMM support and
// backing granularity for each allocation class. It judges what it finds
// against what jitLLM needs from that provider, adding problems and
// warnings. It only queries: it creates no context and allocates or maps
// nothing, though initializing the driver may load its kernel modules. A
// CPU-only build has no device provider and says so.
//
// Files under /proc and /sys are read beneath `root`, "/" for this host.

#ifndef JITLLM_PROVIDERS_DEVICE_PROBE_H_
#define JITLLM_PROVIDERS_DEVICE_PROBE_H_

#include <filesystem>
#include <string>

#include "base/report.h"

namespace jitllm::providers {

void DescribeDevices(const std::filesystem::path& root, base::Report& report);

// The device the build uses and its driver, as one line that changes when
// either does (D-103's calibration key): "NVIDIA GB10 sm_121, driver
// 580.95.05, CUDA 13.0". "none" in a build without a device provider.
std::string DeviceIdentity(const std::filesystem::path& root);

}  // namespace jitllm::providers

#endif  // JITLLM_PROVIDERS_DEVICE_PROBE_H_
