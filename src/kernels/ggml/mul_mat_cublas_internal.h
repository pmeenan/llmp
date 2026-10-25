// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_MUL_MAT_CUBLAS_INTERNAL_H_
#define LLMP_KERNELS_GGML_MUL_MAT_CUBLAS_INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "kernels/ggml/validate.h"
struct ggml_backend_cuda_context;
namespace llmp::kernels::ggml::internal {
// Only inside an admitted LaunchContext::Run. The caller has validated this
// exact plan/operands, the lent handle and disjoint simultaneously held scratch.
// Ordinary MulMatCublas retains its existing complete public preflight.
void LaunchPlannedMulMatCublas(ggml_backend_cuda_context& context, const CublasMulMat& plan,
                               ggml_tensor* node);
// Pure shape query over the launcher's one pinned table. False guarantees the
// pinned attempt cannot launch; true does not promise library availability.
bool HasPinnedBf16GemmShape(int m, int n, int k);
// Packed BF16 weights [k, m] times packed BF16 activations [k, n] into
// packed [m, n] (BF16 where `bf16_out`, else F32), with the GB10's pinned
// cuBLASLt algorithm for the shape (kGb10LtPins) and the lent cuBLAS
// workspace [ws_base, + ws_bytes); false (nothing launched) where it has
// none. Only inside an admitted LaunchContext::Run.
bool TryPinnedBf16Gemm(ggml_backend_cuda_context& context, std::uintptr_t ws_base,
                       std::size_t ws_bytes, int m, int n, int k, const void* weights,
                       const void* x, void* out, bool bf16_out);
}  // namespace llmp::kernels::ggml::internal
#endif  // LLMP_KERNELS_GGML_MUL_MAT_CUBLAS_INTERNAL_H_
