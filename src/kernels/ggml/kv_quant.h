// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Block-quantized KV caches (D-108; docs/engine-components.md, "State
// encodings"): attention that reads K and V in Q8_0 or Q4_0, and the cache
// stores that write them. Kernel-level only: no model selects them until
// its lossy state encoding is an opt-in per-alias setting.
//
//   ggml.flash_attn_ext.vec_q8_0  {flash_attn_ext}  FlashAttnVecQuant
//   ggml.flash_attn_ext.vec_q4_0  {flash_attn_ext}  FlashAttnVecQuant
//       GGML's vector kernel at the pin over Q8_0 (Q4_0) K and V of one
//       encoding, head sizes 64, 256 (with optional positive logit softcap
//       at D=256), and 512 (template-instances/
//       fattn-vec-instance-q8_0-q8_0.cu's and q4_0-q4_0.cu's cases,
//       compiled in fattn_vec_quant.cu). The kernel reads the blocks
//       directly at any number of query rows, two to a CUDA block beyond
//       the first (upstream converts nothing for it: need_f16_K and _V are
//       false for a quantized type), quantizing Q to Q8_1 in shared memory
//       for the K product; nothing expands into device memory. Upstream's
//       own selector takes it for one or two query rows on Ada and later
//       and otherwise its tensor-core kernel, after converting K and V to
//       F16 beside the output (launch_fattn's need_f16_K, in the node's
//       extra allocation, ggml_cuda_flash_attn_ext_get_alloc_size): that
//       conversion is a transient expansion D-108 allows for prefill only
//       where it measures faster, and is not built here.
//   llmp.set_rows.quant           {set_rows}        SetRowsQuant
//   llmp.set_rows.grouped_quant   {set_rows ...}    SetRowsQuantGroup
//   llmp.rope_set_rows.quant      {rope, set_rows}  RopeSetRowsQuant
//       F32 rows stored into a Q8_0 or Q4_0 cache at I32 or I64 row
//       indices, ggml_set_rows's shapes (one store, or 2 to 16 independent
//       ones in one launch), and NEOX RoPE's rotated rows stored so
//       (ggml.rope_set_rows.fused's contract with a quantized
//       destination). Each block of 32 values is quantized by one thread
//       with exactly the arithmetic of GGML's reference quantizers
//       (quantize_row_q8_0_ref, quantize_row_q4_0_ref, ggml-quants.c at the
//       pin; kernels/reference EncodeRows): each step an IEEE operation
//       written as an intrinsic (__fdiv_rn, __fmul_rn, __fadd_rn), so that
//       no flag contracts or approximates it. Upstream's own quantizing
//       store (set-rows.cu, k_set_rows_quant over cpy-utils.cuh) builds
//       with -use_fast_math, which allows approximate division,
//       contraction and flushing subnormals, so its agreement with the
//       reference is not guaranteed (kv_quant_gpu_test prints how many of
//       its blocks differ: none, on that corpus). The plain and grouped stores
//       compile without fast math and equal EncodeRows byte for byte on
//       finite rows, except in blocks whose scale is below F32's normal
//       range, where the reference's reciprocal scale can overflow and its
//       integer conversion of the infinite or NaN product is undefined in
//       C. The RoPE store computes the rotation as GGML's rope_neox does,
//       in a unit with GGML's device flags so that its F32 rotated values
//       are the RoPE node's bit for bit; it equals GGML's RoPE followed by
//       the plain store byte for byte except, further, in blocks holding a
//       subnormal value (which passes through an unrotated part of a head)
//       or a subnormal scale, which -ftz flushes to zero. Every block that
//       differs so has an F16 zero scale (of either sign) on both sides, so
//       it decodes to zeros either way (kv_quant_gpu_test checks it).
//
// Each check refuses, before any launch, what its kernel would not take;
// row indices read from device memory are the plan's to bound (ops.h). The
// stores never draw scratch; the attention draws the vector kernel's pool
// scratch (ops.h FlashAttnPlan), planned first.

#ifndef LLMP_KERNELS_GGML_KV_QUANT_H_
#define LLMP_KERNELS_GGML_KV_QUANT_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string_view>

#include "ggml.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

class LaunchContext;

inline constexpr std::string_view kFlashAttnVecQ8Name = "ggml.flash_attn_ext.vec_q8_0";
inline constexpr std::string_view kFlashAttnVecQ4Name = "ggml.flash_attn_ext.vec_q4_0";
inline constexpr std::string_view kSetRowsQuantName = "llmp.set_rows.quant";
inline constexpr std::string_view kSetRowsQuantGroupName = "llmp.set_rows.grouped_quant";
inline constexpr std::string_view kRopeSetRowsQuantName = "llmp.rope_set_rows.quant";
inline constexpr std::size_t kSetRowsQuantGroupMax = 16;

// The cache encodings these read and write.
bool KvQuantType(ggml_type type);

// The vector attention's name for K's type (kFlashAttnVecQ8Name or
// kFlashAttnVecQ4Name), or empty for any other.
std::string_view FlashAttnVecQuantName(ggml_type k_type);

// A ggml_flash_attn_ext node: F32 Q [D, rows, heads], K and V [D, cells,
// KV heads] of one encoding (Q8_0 or Q4_0) over cells padded to 256, a
// whole number of query heads per KV head, an F16 mask [cells, >= rows]
// (rows rounded up to even from 1,024, which the mask pre-pass reads), no
// sinks, one sequence; D 64 or 256, a positive finite scale, a finite
// positive softcap at D 256 only (upstream compiles none at D 64), no
// ALiBi, F32 precision; F32 output [D, heads, rows]. K's and V's bases and
// strides whole blocks at 2-byte alignment (the kernel's block reads), Q
// and the mask 16-byte aligned, everything within the kernel's 32-bit
// arguments.
std::expected<void, KernelFailure> CheckFlashAttnVecQuant(const ggml_tensor* node);
std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVecQuant(const LaunchContext& launch,
                                                                  const ggml_tensor* node);
// ggml_cuda_flash_attn_ext_vec_case<D, T, T> for the node's D and type T,
// drawing PlanFlashAttnVecQuant's scratch from the context's pool.
std::expected<void, KernelFailure> FlashAttnVecQuant(LaunchContext& launch, ggml_tensor* node);

// A ggml_set_rows node storing F32 rows into a Q8_0 or Q4_0 destination at
// I32 or I64 indices, ggml_set_rows's shapes (the ids' channels and samples
// broadcast over the rows'): rows of whole blocks, every destination stride
// and every destination offset a whole number of blocks, at most 2^32 - 1
// values; the destination overlapping no operand.
std::expected<void, KernelFailure> CheckSetRowsQuant(const ggml_tensor* node);
std::expected<void, KernelFailure> SetRowsQuant(LaunchContext& launch, ggml_tensor* node);

// Two to sixteen such stores, each passing CheckSetRowsQuant, no
// destination overlapping another store's destination or operands; one
// launch, its descriptors in the kernel's arguments (also under capture).
std::expected<void, KernelFailure> CheckSetRowsQuantGroup(
    std::span<const ggml_tensor* const> nodes);
std::expected<void, KernelFailure> SetRowsQuantGroup(LaunchContext& launch,
                                                     std::span<ggml_tensor* const> nodes);

// A forward NEOX RoPE (validate.h CheckRopeInput's) whose rows a set_rows
// stores, through a view flattening the packed RoPE's heads, into a
// two-dimensional Q8_0 or Q4_0 destination [D · heads, cells] at one I64
// index per token: CheckRopeSetRows's contract with a quantized
// destination, a head of whole blocks, rows and the destination's row
// stride whole blocks.
std::expected<void, KernelFailure> CheckRopeSetRowsQuant(const ggml_tensor* rope,
                                                         const ggml_tensor* set_rows);
std::expected<void, KernelFailure> RopeSetRowsQuant(LaunchContext& launch, ggml_tensor* rope,
                                                    ggml_tensor* set_rows);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_KV_QUANT_H_
