// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/device_backend.h"

#include <format>
#include <utility>

#include "kernels/compose/kernel_set.h"

namespace llmp::engine {

kernels::BackendCapabilities DeviceBackendCapabilities() {
  if (!kernels::RegisterBuildKernels()) return {};
  return kernels::DeviceBackendCapabilities();
}

std::expected<std::unique_ptr<kernels::DeviceBackend>, std::string> CreateDeviceBackend(
    PagedNode& node, std::uint32_t stream, kernels::DeviceBackendOptions options) {
  if (auto registered = kernels::RegisterBuildKernels(); !registered)
    return std::unexpected(registered.error());
  return kernels::CreateDeviceBackend(node.device_ordinal(), node.execution(), node.stream(stream),
                                      std::move(options));
}

std::vector<RunnerConfig::KernelMemory> BackendKernelMemory(const kernels::DeviceBackend& backend,
                                                            std::string_view label) {
  std::vector<RunnerConfig::KernelMemory> memory{
      {.name = std::format("{} stream library workspace", label),
       .bytes = backend.library_workspace_bytes()}};
  if (backend.persistent_bytes() != 0) {
    memory.push_back({.name = std::format("{} selected kernel state", label),
                      .bytes = backend.persistent_bytes()});
  }
  return memory;
}

}  // namespace llmp::engine
