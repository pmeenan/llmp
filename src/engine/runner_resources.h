// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a model runner sets up for its life beside its weights and state
// (docs/engine.md): the engine core's runner memory (runner_memory.h: its
// own device memory, a stream library workspace among it, and pinned staging);
// the stream library with upstream's workspace for the device on the model's stream.
//
// Lifetimes are completion-aware (AGENTS.md rule 6): Release runs only
// after the node's teardown fenced the model's stream and stopped the
// scheduler, and the runner destroys its plans and graphs first (they
// name this memory, D-090); then the stream library and the memory go, in
// that order. The staging is the node's to free (PagedNode::Pinned), at its close.

#ifndef LLMP_ENGINE_RUNNER_RESOURCES_H_
#define LLMP_ENGINE_RUNNER_RESOURCES_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/runner_memory.h"
#include "kernels/common/device_backend.h"

namespace llmp::engine {

inline kernels::BackendCapabilities DeviceBackendCapabilities() {
  return kernels::DeviceBackend::Capabilities();
}

std::expected<std::unique_ptr<kernels::DeviceBackend>, std::string> CreateDeviceBackend(
    PagedNode& node, std::uint32_t stream, kernels::DeviceBackendOptions options = {});

std::vector<RunnerConfig::KernelMemory> BackendKernelMemory(const kernels::DeviceBackend& backend,
                                                            std::string_view label);

class RunnerResources : public RunnerMemory {
 public:
  using Status = engine::Status;

  RunnerResources(PagedNode& node, int owner, std::uint32_t stream)
      : RunnerMemory(node, owner, stream) {}
  RunnerResources(const RunnerResources&) = delete;
  RunnerResources& operator=(const RunnerResources&) = delete;
  RunnerResources(RunnerResources&&) = delete;
  RunnerResources& operator=(RunnerResources&&) = delete;
  ~RunnerResources() = default;

  // Stream library on the model's stream, with workspace for the
  // device mapped as `name` (runtime memory).
  Status OpenLibrary(std::string name);
  kernels::StreamLibrary& library() const { return *library_; }
  const Mapped& library_workspace() const { return library_workspace_; }
  std::uint64_t library_bytes() const { return library_bytes_; }

  // Kernel source state is private pinned runtime memory, never shared scratch.
  Status OpenSelectedWorkspace(std::string name, std::uint64_t bytes);
  const Mapped& selected_workspace() const { return selected_workspace_; }

  // The stream library destroyed, then every mapping released
  // (the problems appended). Once, after the node's teardown fenced the
  // stream, and after the runner destroyed its plans and graphs.
  void Release(std::vector<std::string>& problems);

 private:
  Mapped library_workspace_;
  Mapped selected_workspace_;
  std::uint64_t library_bytes_ = 0;
  std::unique_ptr<kernels::StreamLibrary> library_;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_RUNNER_RESOURCES_H_
