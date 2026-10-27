// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A test runner's device backend, set up as production sets up a model's
// (runtime/component_build.cc, component_image_build.cc): made by
// engine::CreateDeviceBackend, its kernel memory (engine::BackendKernelMemory)
// mapped as ModelRunner::Setup maps RunnerConfig::kernel_memory (runtime
// class, pinned, among this runner's extents), DeviceBackend::BindWorkspaces
// given the bases, and DeviceBackend::ReleaseWorkspaces called before the
// mappings are released. CUDA builds; never linked into a production binary.
//
// Tests that run a ModelRunner give it the backend through RunnerConfig
// instead (RunnerKernelConfig), so the runner itself maps and binds it.

#ifndef LLMP_TESTS_SUPPORT_RUNNER_KERNELS_H_
#define LLMP_TESTS_SUPPORT_RUNNER_KERNELS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "engine/runner_memory.h"
#include "kernels/common/device_backend.h"

namespace llmp::test_support {

class RunnerKernels : public engine::RunnerMemory {
 public:
  RunnerKernels(engine::PagedNode& node, int owner, std::uint32_t stream)
      : engine::RunnerMemory(node, owner, stream) {}

  // engine::CreateDeviceBackend on this runner's stream. Once.
  engine::Status Create(kernels::DeviceBackendOptions options = {});
  bool created() const { return backend_ != nullptr; }
  kernels::DeviceBackend& backend() const { return *backend_; }
  kernels::DeviceBackend* operator->() const { return backend_.get(); }

  // After the test's Declare and FundConstruction, before the node runs:
  // the backend's kernel memory mapped (named after `label`), then
  // BindWorkspaces. Once.
  engine::Status Bind(std::string_view label);
  // ReleaseWorkspaces, then BindWorkspaces again over the same mappings, so
  // that another launch can be made once the previous one is destroyed.
  // Only with nothing of the backend's in flight (destroying the stream
  // library synchronizes the device).
  engine::Status Rebind();

  // ReleaseWorkspaces when bound, the backend destroyed, then every mapping
  // released (the problems appended). Once, after the node's teardown fenced
  // the stream and after every launch, program and plan of the backend is
  // gone (AGENTS.md rule 6).
  void Release(std::vector<std::string>& problems);

 private:
  std::unique_ptr<kernels::DeviceBackend> backend_;
  std::vector<std::unique_ptr<engine::Mapped>> mapped_;  // at stable addresses
  std::vector<std::uint64_t> bases_;
  bool bound_ = false;
};

// What component_build.cc gives a model runner for its backend: the kernel
// memory ModelRunner::Setup maps, `prepare` binding the backend's workspaces
// over it and `retire` releasing them at the runner's Release. `backend`
// outlives the runner's Release.
void RunnerKernelConfig(engine::RunnerConfig& config, kernels::DeviceBackend& backend,
                        std::string_view label);

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_RUNNER_KERNELS_H_
