// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_KERNELS_GGML_DSV4_QHEAD_H_
#define JITLLM_KERNELS_GGML_DSV4_QHEAD_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

class LaunchContext;
inline constexpr const char* kDsv4QHeadName = "jitllm.dsv4.qhead";

// Normal RoPE on the last64 values of a512-value head, after unweighted
// F32 RMSNorm. Parameters are graph-owned values, never device pointers.
struct Dsv4QHeadParams {
  float eps = 0;
  std::int32_t original_context = 0;
  float base = 0;
  float scale = 0;
  float extension = 0;
  float attention = 0;
  float beta_fast = 0;
  float beta_slow = 0;
};

// Metadata selection only: contiguous F32[512,64,16..8192,1] and one
// contiguous I32 position per token. Unsupported graphs keep both primitives.
bool Dsv4QHeadFits(const ggml_tensor* x, const ggml_tensor* positions,
                   const Dsv4QHeadParams& params);
// `type` F32, or F16 (experimental): the same values rounded to nearest.
ggml_tensor* Dsv4QHead(ggml_context* context, ggml_tensor* x, ggml_tensor* positions,
                       const Dsv4QHeadParams& params, ggml_type type = GGML_TYPE_F32);
Dsv4QHeadParams Dsv4QHeadParamsOf(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4QHead(const ggml_tensor* node);

// CUDA only. The graph directly protects input and output together, with
// no materialized RMSNorm tensor or scratch. Native F32 materialization and
// RoPE multiply/FMA ordering are explicit; completion belongs to the caller.
std::expected<void, KernelFailure> RunDsv4QHead(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_QHEAD_H_
