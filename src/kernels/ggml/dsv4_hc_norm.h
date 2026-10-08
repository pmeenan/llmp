// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's hyper-connection mix input as F16 rows (experimental,
// docs/experiments/ds4-prefill-stages): the flat RMSNorm of the four streams,
// rounded once to F16, which the mix's cuBLAS product then reads directly.
// The arithmetic is native rms_norm's 1024-thread F32 normalization
// followed by the launcher's F32-to-F16 conversion, so the product's
// operand bytes are unchanged; the F32 normalized tensor is never written.
// Where the norm follows the HC post of the previous block, one kernel
// writes the expanded F32 streams and these rows together.
//
// For F32 mixing weights (DeepSeek V4 0731 UD-Q2_K_XL) the same node writes
// F32 rows: native rms_norm's output itself, byte for byte, which the F32
// product reads as it read rms_norm's; the post's fusion and the expert
// sum's (below) then apply as for F16 rows. The names keep "F16".

#ifndef LLMP_KERNELS_GGML_DSV4_HC_NORM_H_
#define LLMP_KERNELS_GGML_DSV4_HC_NORM_H_

#include <cstddef>
#include <expected>
#include <optional>
#include <span>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

class LaunchContext;
inline constexpr const char* kDsv4HcNormF16Name = "llmp.dsv4.hc_norm_f16";
inline constexpr const char* kDsv4HcPostNormF16Name = "llmp.dsv4.hc_post_norm_f16";

// The flat streams this takes: canonical F32 [16384, rows], rows at most
// 65535, and a finite positive epsilon.
bool Dsv4HcNormF16Fits(const ggml_tensor* flat, float eps);
// `type` (F16, or F32) [16384, rows]: the custom node (llmp_ops.h
// kDsv4HcNormF16).
ggml_tensor* Dsv4HcNormF16(ggml_context* context, ggml_tensor* flat, float eps,
                           ggml_type type = GGML_TYPE_F16);
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
inline constexpr const char* kDsv4HcPostExpertsNormF16Name = "llmp.dsv4.hc_post_experts_norm_f16";
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
inline constexpr const char* kDsv4F16CopyName = "llmp.dsv4.f16_copy";
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

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_DSV4_HC_NORM_H_
