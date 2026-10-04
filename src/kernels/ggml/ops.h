// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// GGML-derived operation implementations over the K-C launch context
// (D-053; docs/backend-proof.md#dispatch-and-implementations-d-053). Each
// takes operation nodes built with GGML's graph functions (tensors.h) and
// bound to jitLLM memory, checks every precondition its GGML launcher
// asserts, so that an unsupported operand is a rejection and never an
// abort, and queues the launcher's kernels on the context's stream.
// GGML's graph functions assert their own shape rules when they build a
// node, so nodes are built only from shapes a plan has validated. The
// operand checks are in validate.h, which every profile builds.
//
// Matrix multiplication comes as separate implementations, one per GGML
// kernel family, since the plan, not GGML's routing, selects among them.
// Each accepts only operands that upstream's selection would route to it,
// which is where upstream validated it. The cuBLAS implementation is a
// recorded jitLLM copy of upstream's (mul_mat_cublas.cu).

#ifndef JITLLM_KERNELS_GGML_OPS_H_
#define JITLLM_KERNELS_GGML_OPS_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"

namespace jitllm::kernels::ggml {

// A ggml_rms_norm node over F32 rows.
std::expected<void, KernelFailure> RmsNorm(LaunchContext& launch, ggml_tensor* norm);

// A ggml_rms_norm node and the ggml_mul that scales it, as GGML's fused
// launcher: one kernel writes the product to `mul`, and `norm` is never
// written. Fusion is the plan's choice (D-053).
std::expected<void, KernelFailure> RmsNormMul(LaunchContext& launch, ggml_tensor* norm,
                                              ggml_tensor* mul);

// The same nodes unfused, as GGML runs them with fusion off: rms_norm's
// launcher writes the norm into its own memory, the plan's intermediate,
// and mul's launcher scales it into `mul`, both in one run.
std::expected<void, KernelFailure> RmsNormThenMul(LaunchContext& launch, ggml_tensor* norm,
                                                  ggml_tensor* mul);

// ggml_add and ggml_mul nodes with broadcasting, all F32.
std::expected<void, KernelFailure> Add(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Mul(LaunchContext& launch, ggml_tensor* node);

// ggml_cuda_mul_mat's selection for a node on the context's device
// (ggml-cuda.cu:1823-1874): cuBLAS for other than F32 activations and
// output, else MMVF, MMF or cuBLAS as upstream chooses. Refused where
// upstream would take a path with no implementation here: the transposed
// vector product, or cuBLAS for a view of a padded compute-buffer tensor;
// and for quantized weights, whose MMVQ and MMQ ops_ext.h's SelectMulMatQ
// chooses between.
std::expected<MulMatPath, KernelFailure> SelectMulMat(const LaunchContext& launch,
                                                      const ggml_tensor* node);

// A ggml_mul_mat node, F32 activations and output: GGML's vector kernel
// (MMVF) and its tensor-core kernel for up to 16 columns (MMF).
std::expected<void, KernelFailure> MulMatVecF(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> MulMatF(LaunchContext& launch, ggml_tensor* node);

// GGML's cuBLAS path for a ggml_mul_mat node, as it would run on the
// context's device: what it converts, which cuBLAS call it makes and the
// scratch it draws (validate.h). Refused unless upstream would route the
// node to cuBLAS.
std::expected<CublasMulMat, KernelFailure> PlanMulMatCublas(const LaunchContext& launch,
                                                            const ggml_tensor* node);
// Runs that plan on the context's lent cuBLAS handle (cublas.h), drawing
// its conversions, compute-type output and pointer arrays from the
// context's scratch; refused if the context lends no handle or an operand
// overlaps either workspace. A node marked GGML_PREC_F32 computes in F32,
// which converts F16 or BF16 weights whole into scratch: a plan uses that
// only for activations (BP-A3).
std::expected<void, KernelFailure> MulMatCublas(LaunchContext& launch, ggml_tensor* node);

// The operations of the FP16 bridge's recorded plan beyond those above
// (validate.h). None draws scratch. The row indices GetRows, SetRows and
// RopeSetRows read from device memory are not checked: an index outside
// its tensor reads or writes out of bounds, so the plan that writes them
// must bound them (validate.h).

// A ggml_get_rows node (the last layer's output rows): upstream's launcher,
// which chooses its vector or scalar kernel (validate.h GetRowsVectorized).
std::expected<void, KernelFailure> GetRows(LaunchContext& launch, ggml_tensor* node);
// A ggml_set_rows node: the KV write, F32 rows into an F16 cache.
std::expected<void, KernelFailure> SetRows(LaunchContext& launch, ggml_tensor* node);
// A NEOX ggml_rope_ext node over F32.
std::expected<void, KernelFailure> Rope(LaunchContext& launch, ggml_tensor* rope);
// The same RoPE fused with the KV write that stores a flattening view of it:
// one kernel writes the rotated rows into set_rows' F16 destination, and the
// RoPE is never written (the FP16-F plan's K write). Unfused, the plan runs
// Rope, then SetRows over the view.
std::expected<void, KernelFailure> RopeSetRows(LaunchContext& launch, ggml_tensor* rope,
                                               ggml_tensor* set_rows);
// A ggml_soft_max_ext node over F32 with an F32 mask (attention's scores),
// refused if a row does not fit the device's shared memory.
std::expected<void, KernelFailure> SoftMax(LaunchContext& launch, ggml_tensor* node);
// A ggml_cont node over F32 (attention's merged heads): a copy, a pitched
// copy or GGML's scalar kernel, as upstream's launcher chooses
// (validate.h CheckCont).
std::expected<void, KernelFailure> Cont(LaunchContext& launch, ggml_tensor* node);
// A ggml_swiglu_split node over F32.
std::expected<void, KernelFailure> SwiGlu(LaunchContext& launch, ggml_tensor* node);
// Split F32 GELU-tanh GLU; CheckGeGlu enforces the operand contract.
std::expected<void, KernelFailure> GeGlu(LaunchContext& launch, ggml_tensor* node);

// The EXL3 plan's operations beyond those above
// (docs/experiments/backend-proof-p0/exl3-op-plan.json).

// A ggml_cpy node converting F32 to F16 or F16 to F32 (validate.h
// CheckConvert): ggml_cuda_cpy's cpy_scalar_contiguous. Draws no scratch.
std::expected<void, KernelFailure> Convert(LaunchContext& launch, ggml_tensor* node);

// What the vector attention's launch_fattn (fattn-common.cuh:975-1215)
// computes on the context's device before it launches: its parallel
// blocks, from the kernel's occupancy and the tail-effect search, whether
// the mask pre-pass runs (1,024 query rows or more), and the pool scratch
// it draws, each block from a 256-byte boundary in its allocation order
// (the pre-pass's KV_max, then the partial results and their metadata).
struct FlashAttnPlan {
  int columns_per_block = 0;  // 1 for one query row, else 2
  int parallel_blocks = 0;
  bool mask_prepass = false;
  std::uint64_t scratch = 0;
};
std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVec(const LaunchContext& launch,
                                                             const ggml_tensor* node);
// A ggml_flash_attn_ext node (validate.h CheckFlashAttnVec) through GGML's
// vector kernel, forced as the EXL3 plan forces it for every phase:
// ggml_cuda_flash_attn_ext_vec_case<64, F16, F16> (flash_attn_ext_vec
// with one column per block for one query row, else two; the mask
// pre-pass flash_attn_mask_to_KV_max from 1,024 rows; then
// flash_attn_combine_results over the parallel blocks), drawing
// PlanFlashAttnVec's scratch from the context's pool.
std::expected<void, KernelFailure> FlashAttnVec(LaunchContext& launch, ggml_tensor* node);
// D256 F16 KV vector primitive, exact GQA2 and one sequence. The pinned
// overall selector chooses it for one query on Ada+; other rows use MMA.
std::expected<FlashAttnPlan, KernelFailure> PlanFlashAttnVec256(const LaunchContext& launch,
                                                                const ggml_tensor* node);
std::expected<void, KernelFailure> FlashAttnVec256(LaunchContext& launch, ggml_tensor* node);
bool FlashAttnVec256Selected(const LaunchContext& launch, const ggml_tensor* node);

// MMVF with GGML's fusion arguments, for one activation column, as the
// FP16-F plan runs it: a product and its bias (or residual) add, written to
// the add; and gate and up products with their SwiGLU, written to the GLU.
// The products are never written. Refused unless upstream would select MMVF
// for the product (for the GLU, the up product) on the context's device.
// Unfused, the plan runs MulMatVecF, then Add; or MulMatVecF twice, then
// SwiGlu.
std::expected<void, KernelFailure> MulMatVecBias(LaunchContext& launch, ggml_tensor* mul_mat,
                                                 ggml_tensor* add);
std::expected<void, KernelFailure> MulMatVecGlu(LaunchContext& launch, ggml_tensor* gate,
                                                ggml_tensor* up, ggml_tensor* glu);
// GELU-tanh counterpart; primitive fallback is two MulMatVecF then GeGlu.
std::expected<void, KernelFailure> MulMatVecGeGlu(LaunchContext& launch, ggml_tensor* gate,
                                                  ggml_tensor* up, ggml_tensor* glu);

// The device's part of upstream's MMVF fusion gates
// (ggml_cuda_should_fuse_mul_mat_vec_f, ggml-cuda.cu:1767-1792): F16, BF16
// or F32 weights with F32 activations and output, MMVF selected for them on
// the context's device, and one output column. With fusion.h's
// MulMatGluFusionAt (asked of its up product) or MulMatAddFusionAt, this
// decides whether upstream fuses.
bool MulMatVecFusible(const LaunchContext& launch, const ggml_tensor* mul_mat);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_OPS_H_
