// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/bytes.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/launch.h"
#include "norm.cuh"
#include "rope.cuh"

namespace jitllm::kernels::ggml {
std::expected<void, KernelFailure> RunGemmaNormRope(LaunchContext& launch, ggml_tensor* norm,
                                                    ggml_tensor* mul, ggml_tensor* rope) {
  if (auto r = CheckGemmaNormRope(norm, mul, rope); !r) return r;
  return launch.Run(base::Bytes(0), [norm, mul, rope](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm_mul_rope_fused(context, norm, mul, rope, nullptr);
  });
}
std::expected<void, KernelFailure> RunGemmaNormAdd(LaunchContext& launch, ggml_tensor* norm,
                                                   ggml_tensor* mul, ggml_tensor* add) {
  if (auto r = CheckGemmaNormAdd(norm, mul, add); !r) return r;
  return launch.Run(base::Bytes(0), [norm, mul, add](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm_fused_add(context, norm, mul, add);
  });
}
}  // namespace jitllm::kernels::ggml
