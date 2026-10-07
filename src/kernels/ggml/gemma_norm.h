// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_KERNELS_GGML_GEMMA_NORM_H_
#define JITLLM_KERNELS_GGML_GEMMA_NORM_H_

#include <expected>
#include <string_view>

#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {
class LaunchContext;
inline constexpr std::string_view kGemmaNormRopeName = "ggml.rms_norm_mul_rope.fused";
inline constexpr std::string_view kGemmaNormAddName = "ggml.rms_norm_mul_add.fused";

// The norm/product intermediates must be nonviews and are not written.
// Actual inputs cannot depend on either elided tensor. A planner must preserve
// every external reader/keep of those tensors by selecting primitives.
// F32 full D256/D512 NEOX or residual widths2304/2560/2816/5376 only; scratch0.
// The caller funds/protects operands through completion and every replay.
// Finite payloads and positive frequency-factor values are caller obligations.
std::expected<void, KernelFailure> CheckGemmaNormRope(const ggml_tensor* norm,
                                                      const ggml_tensor* mul,
                                                      const ggml_tensor* rope);
std::expected<void, KernelFailure> CheckGemmaNormAdd(const ggml_tensor* norm,
                                                     const ggml_tensor* mul,
                                                     const ggml_tensor* add);
// The checked residual GET_ROWS may execute before the norm/product/add call.
// It remains a separate paid producer, and must not overwrite actual norm
// inputs or read either elided intermediate.
std::expected<void, KernelFailure> CheckGemmaNormAddGather(const ggml_tensor* norm,
                                                           const ggml_tensor* mul,
                                                           const ggml_tensor* gather,
                                                           const ggml_tensor* add);
std::expected<void, KernelFailure> RunGemmaNormRope(LaunchContext& launch, ggml_tensor* norm,
                                                    ggml_tensor* mul, ggml_tensor* rope);
std::expected<void, KernelFailure> RunGemmaNormAdd(LaunchContext& launch, ggml_tensor* norm,
                                                   ggml_tensor* mul, ggml_tensor* add);
}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_GEMMA_NORM_H_
