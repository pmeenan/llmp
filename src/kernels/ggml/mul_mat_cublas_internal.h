// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_MUL_MAT_CUBLAS_INTERNAL_H_
#define LLMP_KERNELS_GGML_MUL_MAT_CUBLAS_INTERNAL_H_

#include "kernels/ggml/validate.h"
struct ggml_backend_cuda_context;
namespace llmp::kernels::ggml::internal {
// Only inside an admitted LaunchContext::Run. The caller has validated this
// exact plan/operands, the lent handle and disjoint simultaneously held scratch.
// Ordinary MulMatCublas retains its existing complete public preflight.
void LaunchPlannedMulMatCublas(ggml_backend_cuda_context& context, const CublasMulMat& plan,
                               ggml_tensor* node);
}  // namespace llmp::kernels::ggml::internal
#endif  // LLMP_KERNELS_GGML_MUL_MAT_CUBLAS_INTERNAL_H_
