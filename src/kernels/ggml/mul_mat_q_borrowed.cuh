// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Kernel-private launch seam, used only inside an already planned native
// LaunchContext::Run. No nested Run or foreign dispatch/ownership.
#ifndef JITLLM_KERNELS_GGML_MUL_MAT_Q_BORROWED_CUH_
#define JITLLM_KERNELS_GGML_MUL_MAT_Q_BORROWED_CUH_

struct ggml_backend_cuda_context;
struct ggml_tensor;

namespace jitllm::kernels::ggml::internal {
// PlanMulMatQBorrowedD4 must have accepted these exact metadata/operands
// before any submission. Records failures with the ordinary native CUDA
// error collector; the outer Run retains its charged workspace on unknown.
// Called across translation units by dsv4_ds4_product.cu.
// NOLINTNEXTLINE(misc-use-internal-linkage)
void LaunchMulMatQBorrowedD4(ggml_backend_cuda_context& context, const ggml_tensor* node,
                             const void* quantized);
}  // namespace jitllm::kernels::ggml::internal

#endif  // JITLLM_KERNELS_GGML_MUL_MAT_Q_BORROWED_CUH_
