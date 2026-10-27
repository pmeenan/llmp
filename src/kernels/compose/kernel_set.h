// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA build's composition root (D-053, D-110): the one place that
// names the build's kernel sources and registers them as the process's
// kernel set (kernels/common/device_backend.h). The engine and the runtime
// call RegisterBuildKernels and then reach every backend through the
// registration; nothing above the kernels names a source.
#ifndef LLMP_KERNELS_COMPOSE_KERNEL_SET_H_
#define LLMP_KERNELS_COMPOSE_KERNEL_SET_H_

#include <expected>
#include <string>

namespace llmp::kernels {

// Registers the build's kernel set once per process; a repeat is a no-op.
// Refused when another set (a test's fake) was registered first.
std::expected<void, std::string> RegisterBuildKernels();

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMPOSE_KERNEL_SET_H_
