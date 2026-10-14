// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_GGML_ATTENTION_F32_H_
#define LLMP_KERNELS_GGML_ATTENTION_F32_H_
#include <cstdint>
#include <expected>
#include <string_view>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
namespace llmp::kernels::ggml {
class LaunchContext;
inline constexpr std::string_view kAttentionF32 = "ggml.flash_attn_ext.f32_mma_d128";
struct AttentionF32Plan {
  std::uint64_t cells = 0, rows = 0, kv_bytes = 0, mask_bytes = 0, packing_bytes = 0;
  // Three fills, three conversions, mask scan, attention and possible fixup.
  static constexpr std::uint64_t kLaunchBound = 9;
};
struct AttentionF32Options {
  bool q_is_f16 = false;
  bool k_prepacked = false;
  bool v_is_bf16 = false;
  bool mask_prepacked = false;
  bool bf16_dst = false;
  bool qkv_bf16_row_major = false;
  bool norm_q_in_attn = false;
  const void* q_weight = nullptr;
  float q_eps = 0.0f;
};
std::expected<AttentionF32Plan, KernelFailure> CheckAttentionF32(const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanAttentionF32(const LaunchContext& launch,
                                                             const ggml_tensor* node);
std::expected<void, KernelFailure> AttentionF32(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> AttentionF32WithOptions(LaunchContext& launch, ggml_tensor* node,
                                                           AttentionF32Options opts);
}  // namespace llmp::kernels::ggml
#endif
