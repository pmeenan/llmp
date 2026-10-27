// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model runner's device backend (docs/engine.md), CUDA builds only: the
// build's registered kernel set (kernels/compose/kernel_set.h; registered
// here on first use) makes it on the model's stream, and the kernel memory it needs beside
// the runner's weights and state. The runner maps that memory as its own
// (RunnerConfig::kernel_memory, runtime class, pinned, in every closure),
// gives the bases to DeviceBackend::BindWorkspaces from RunnerConfig::prepare
// (which makes the backend's stream library over the first), launches over
// the node's pool and calls DeviceBackend::ReleaseWorkspaces from
// RunnerConfig::retire, after the node's teardown fenced the stream and the
// runner destroyed its plans, graphs and launch, before it releases the
// memory (AGENTS.md rule 6, D-090).

#ifndef LLMP_ENGINE_DEVICE_BACKEND_H_
#define LLMP_ENGINE_DEVICE_BACKEND_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "engine/model_runner.h"
#include "engine/paged_node.h"
#include "kernels/common/device_backend.h"

namespace llmp::engine {

// The build's backend capabilities (none if its kernel set cannot be
// registered, which CreateDeviceBackend then reports).
kernels::BackendCapabilities DeviceBackendCapabilities();

// The device backend on `stream` of `node`'s device.
std::expected<std::unique_ptr<kernels::DeviceBackend>, std::string> CreateDeviceBackend(
    PagedNode& node, std::uint32_t stream, kernels::DeviceBackendOptions options = {});

// The kernel memory `backend` binds (BindWorkspaces' bases, in order): its
// stream library's workspace, then its persistent selected-source state when
// it has any. `label` names the mappings.
std::vector<RunnerConfig::KernelMemory> BackendKernelMemory(const kernels::DeviceBackend& backend,
                                                            std::string_view label);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_DEVICE_BACKEND_H_
