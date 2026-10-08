// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the launch context needs from ggml_support.cu, llmpalooza's definitions
// of the ggml-cuda.cu symbols GGML's launchers use. Internal to this module.

#ifndef LLMP_KERNELS_GGML_GGML_SUPPORT_H_
#define LLMP_KERNELS_GGML_GGML_SUPPORT_H_

#include <optional>
#include <string>

struct cublasContext;
struct ggml_backend_cuda_context;

namespace llmp::kernels::ggml::internal {

// The first CUDA failure GGML recorded on this thread since the last call,
// if any; taking it clears it.
std::optional<std::string> TakeCudaError();
// Whether a failure is recorded and not yet taken.
bool CudaErrorPending();

// The cuBLAS handle GGML's call sites take from the context
// (ggml_backend_cuda_context::cublas_handle), or null if none was lent
// (where upstream would create one); and whether the context holds a
// cuBLAS workspace of its own. For tests.
cublasContext* CublasHandleOf(ggml_backend_cuda_context& context);
bool HoldsCublasWorkspace(const ggml_backend_cuda_context& context);

}  // namespace llmp::kernels::ggml::internal

#endif  // LLMP_KERNELS_GGML_GGML_SUPPORT_H_
