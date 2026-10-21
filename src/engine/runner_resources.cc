// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/runner_resources.h"

#include <format>
#include <utility>

#include "base/bytes.h"
#include "engine/support.h"

namespace llmp::engine {

namespace {

using base::Bytes;
using support::Error;

}  // namespace

std::expected<std::unique_ptr<kernels::DeviceBackend>, std::string> CreateDeviceBackend(
    PagedNode& node, std::uint32_t stream, kernels::DeviceBackendOptions options) {
  return kernels::DeviceBackend::Create(node.device_ordinal(), node.execution(),
                                        node.stream(stream), std::move(options));
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

RunnerResources::Status RunnerResources::OpenLibrary(std::string name) {
  auto bytes = kernels::StreamLibraryWorkspaceBytes(node().device_ordinal());
  if (!bytes) return Error(bytes.error());
  library_bytes_ = *bytes;
  if (auto r =
          Map(library_workspace_, std::move(name), library_bytes_, catalog::MemoryClass::kRuntime);
      !r) {
    return r;
  }
  auto lib = kernels::CreateStreamLibrary(
      node().device_ordinal(), node().execution(), node().stream(stream()),
      {.base = library_workspace_.base, .size = Bytes(library_bytes_)});
  if (!lib) {
    return Error(lib.error());
  }
  library_ = std::move(*lib);
  return {};
}

RunnerResources::Status RunnerResources::OpenSelectedWorkspace(std::string name,
                                                               std::uint64_t bytes) {
  if (bytes == 0) return {};
  return Map(selected_workspace_, std::move(name), bytes, catalog::MemoryClass::kRuntime);
}

void RunnerResources::Release(std::vector<std::string>& problems) {
  library_.reset();
  ReleaseMappings(problems);
}

}  // namespace llmp::engine
