// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's hyper-connection mix input as F16 rows (experimental,
// docs/experiments/ds4-prefill-stages): the flat RMSNorm of the four streams,
// rounded once to F16, which the mix's cuBLAS product then reads directly.
// The arithmetic is native rms_norm's 1024-thread F32 normalization
// followed by the launcher's F32-to-F16 conversion, so the product's
// operand bytes are unchanged; the F32 normalized tensor is never written.
// Where the norm follows the HC post of the previous block, one kernel
// writes the expanded F32 streams and these rows together.

#ifndef JITLLM_KERNELS_GGML_DSV4_HC_NORM_H_
#define JITLLM_KERNELS_GGML_DSV4_HC_NORM_H_

#include <cstddef>
#include <expected>
#include <optional>
#include <span>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

class LaunchContext;
inline constexpr const char* kDsv4HcNormF16Name = "jitllm.dsv4.hc_norm_f16";
inline constexpr const char* kDsv4HcPostNormF16Name = "jitllm.dsv4.hc_post_norm_f16";

// The flat streams this takes: canonical F32 [16384, rows], rows at most
// 65535, and a finite positive epsilon.
bool Dsv4HcNormF16Fits(const ggml_tensor* flat, float eps);
// F16 [16384, rows]: the custom node (jitllm_ops.h kDsv4HcNormF16).
ggml_tensor* Dsv4HcNormF16(ggml_context* context, ggml_tensor* flat, float eps);
std::expected<void, KernelFailure> CheckDsv4HcNormF16(const ggml_tensor* norm);

// The HC post, its zero-offset flat reshape and this norm, consecutive in
// the graph: one step over {post, norm}.
struct Dsv4HcPostNormF16Nodes {
  ggml_tensor* post = nullptr;
  ggml_tensor* norm = nullptr;
};
std::optional<Dsv4HcPostNormF16Nodes> Dsv4HcPostNormF16At(std::span<ggml_tensor* const> graph,
                                                          std::size_t index);
std::expected<void, KernelFailure> CheckDsv4HcPostNormF16(const ggml_tensor* post,
                                                          const ggml_tensor* norm);

// Experimental (docs/experiments/ds4-prefill-stages): where the post's input
// is the routed experts' ordered six-slot reduction plus the shared expert,
// the same kernel forms that sum itself (the reduction's and the add's
// arithmetic), so neither sum is written. At the reduction's graph index;
// `add_index` receives the add's, after which the post, reshape and norm follow.
inline constexpr const char* kDsv4HcPostExpertsNormF16Name = "jitllm.dsv4.hc_post_experts_norm_f16";
struct Dsv4HcPostExpertsNodes {
  ggml_tensor* reduce = nullptr;
  ggml_tensor* add = nullptr;
  ggml_tensor* post = nullptr;
  ggml_tensor* norm = nullptr;
};
std::optional<Dsv4HcPostExpertsNodes> Dsv4HcPostExpertsAt(std::span<ggml_tensor* const> graph,
                                                          std::size_t index,
                                                          std::size_t* add_index);
std::expected<void, KernelFailure> CheckDsv4HcPostExpertsNormF16(const ggml_tensor* reduce,
                                                                 const ggml_tensor* add,
                                                                 const ggml_tensor* post,
                                                                 const ggml_tensor* norm);

// Experimental: one F16 copy (RN, as cuBLAS's launcher converts) of an F32
// [k, rows] activation that several F16-weight products read directly.
inline constexpr const char* kDsv4F16CopyName = "jitllm.dsv4.f16_copy";
bool Dsv4F16CopyFits(const ggml_tensor* x);
ggml_tensor* Dsv4F16Copy(ggml_context* context, ggml_tensor* x);
std::expected<void, KernelFailure> CheckDsv4F16Copy(const ggml_tensor* copy);

// CUDA only; zero scratch; completion belongs to the caller.
std::expected<void, KernelFailure> RunDsv4F16Copy(LaunchContext& launch, ggml_tensor* copy);
std::expected<void, KernelFailure> RunDsv4HcNormF16(LaunchContext& launch, ggml_tensor* norm);
std::expected<void, KernelFailure> RunDsv4HcPostNormF16(LaunchContext& launch, ggml_tensor* post,
                                                        ggml_tensor* norm);
std::expected<void, KernelFailure> RunDsv4HcPostExpertsNormF16(LaunchContext& launch,
                                                               ggml_tensor* reduce,
                                                               ggml_tensor* add, ggml_tensor* post,
                                                               ggml_tensor* norm);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_HC_NORM_H_
