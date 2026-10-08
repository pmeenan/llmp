// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// How the CUDA providers turn driver errors into provider failures: out of
// memory and unsupported where the driver says so, a known failure only for
// errors the driver reports before doing anything (invalid arguments and
// the like), and an unknown outcome for everything else. Internal to providers/cuda: it includes
// the driver's header.

#ifndef LLMP_PROVIDERS_CUDA_CUDA_ERRORS_H_
#define LLMP_PROVIDERS_CUDA_CUDA_ERRORS_H_

#include <cuda.h>

#include <expected>

#include "providers/device_memory.h"

namespace llmp::providers::cuda {

std::unexpected<Failure> Error(CUresult result, const char* call);

}  // namespace llmp::providers::cuda

#endif  // LLMP_PROVIDERS_CUDA_CUDA_ERRORS_H_
