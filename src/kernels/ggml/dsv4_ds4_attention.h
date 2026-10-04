// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's copy of the original ds4 token-tile HCA core that reads F16 Q
// rows (dsv4_ds4_attention.cu), for the DeepSeek V4 prefill HCA adapter
// (dsv4_hca_tokentile.cu, docs/experiments/ds4-prefill-stages).
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_H_

#include <cstdint>

namespace jitllm::kernels::ggml {

// The token-tile HCA core over F16 Q rows; arguments as the patched ds4
// jitllm_ds4_hca_core_launch's. Prepare sets the kernel's shared-memory
// opt-in outside graph capture. Each returns a cudaError_t.
int Ds4HcaCoreQ16Prepare();
int Ds4HcaCoreQ16Launch(float* out, const float* sinks, const void* q, const void* raw,
                        const void* compressed, const void* records, const void* counts,
                        std::uint32_t stride, std::uint32_t tokens, std::uint32_t heads,
                        std::uint32_t raw_row_min, void* stream);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_ATTENTION_H_
