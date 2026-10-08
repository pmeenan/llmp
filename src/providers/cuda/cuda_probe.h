// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Queries the CUDA driver for the device probe. Binaries that link this
// link the driver's libcuda.so.1, which every host running them must have
// (D-072). The probe initializes the driver but creates no context and
// allocates or maps nothing. Initializing it may still load
// the driver's kernel modules (through the setuid nvidia-modprobe, as any
// CUDA program would).

#ifndef LLMP_PROVIDERS_CUDA_CUDA_PROBE_H_
#define LLMP_PROVIDERS_CUDA_CUDA_PROBE_H_

#include <filesystem>

#include "providers/cuda/cuda_facts.h"

namespace llmp::providers::cuda {

// Queries the driver. Kernel modules are read beneath `root`.
CudaFacts ProbeCuda(const std::filesystem::path& root);

}  // namespace llmp::providers::cuda

#endif  // LLMP_PROVIDERS_CUDA_CUDA_PROBE_H_
