// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The approved GB10 reconstruction pins and their planner facts. This is
// host code in every profile; it owns no CUDA handle and runs no kernel.
#ifndef LLMP_KERNELS_EXL3_RECONSTRUCTION_TUNING_H_
#define LLMP_KERNELS_EXL3_RECONSTRUCTION_TUNING_H_
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "execution/registry.h"
#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {
// All slices of exactly this full matrix, including the output type and
// pitch; refuses an incomplete pin set or another device/shape. The
// launcher still checks the loaded cuBLAS version and each algorithm.
std::expected<std::vector<LtAlgorithm>, std::string> ReconstructionAlgorithms(std::uint32_t arch,
                                                                              int rows, int k,
                                                                              int n, Output output);
// Supplies DeviceFacts.tuned. No fact is published for an unpinned type,
// partial slice or architecture; strings point to static storage.
std::vector<execution::TunedShape> ReconstructionTuning(std::uint32_t arch);
}  // namespace llmp::kernels::exl3
#endif  // LLMP_KERNELS_EXL3_RECONSTRUCTION_TUNING_H_
