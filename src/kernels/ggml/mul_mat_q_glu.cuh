// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Experimental (docs/experiments/ds4-prefill-stages): the IQ2 compact pair's up
// product with the routed SwiGLU-clamp in its write-back. CUDA only;
// included after GGML's mmq.cuh.

#ifndef JITLLM_KERNELS_GGML_MUL_MAT_Q_GLU_CUH_
#define JITLLM_KERNELS_GGML_MUL_MAT_Q_GLU_CUH_

#include "mmq.cuh"

namespace jitllm::kernels::ggml {

// The occupancy-two J64 compact IQ2 product of `args` (ggml mmq.cu's
// measured GB10 pair geometry only), whose write-back stores
// swiglu_clamp(gate[k], product[k], limit) at args.dst[k]; `gate` has the
// destination's layout and is fully written before this launch. With `q8`,
// the activation is instead written as the down product's D2S6 Q8_1 MMQ
// input at each sorted column (args.ncols_y columns per 128-value block).
cudaError_t LaunchIq2PairGluUp(ggml_backend_cuda_context& ctx, const mmq_args& args,
                               const float* gate, float limit, void* q8, cudaStream_t stream);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_MUL_MAT_Q_GLU_CUH_
