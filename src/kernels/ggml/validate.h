// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The operand checks of the GGML-derived implementations (ops.h), on the
// host and in every build profile, so the CPU build tests them without a
// GPU. Each refuses what its GGML launcher would assert on or index past,
// as a kRejected failure: operand types and layouts, empty tensors, extents
// computed with checked arithmetic, 32-bit indexing and grid limits,
// alignment of bases and strides, views that no longer follow their
// source, and outputs that share bytes with an input other than exactly in
// place. The elementwise and row operations take packed operands only
// (strides of a dense tensor of their shape). Upstream's kernel-family
// selection, which needs the device, stays in ops.cu.

#ifndef JITLLM_KERNELS_GGML_VALIDATE_H_
#define JITLLM_KERNELS_GGML_VALIDATE_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// A ggml_rms_norm node over packed F32 rows, writing its own output.
std::expected<void, KernelFailure> CheckRmsNorm(const ggml_tensor* norm);
// A ggml_rms_norm node and the ggml_mul that scales it, for GGML's fused
// launcher, which writes the product and never the norm.
std::expected<void, KernelFailure> CheckRmsNormMul(const ggml_tensor* norm, const ggml_tensor* mul);
// The same two nodes for rms_norm's and mul's own launchers, one after the
// other: the norm is written, to memory that shares no byte with its
// input or the weight, and the mul reads it back. Apart from that
// intermediate, the result and every other operand are as the fused
// launcher leaves them.
std::expected<void, KernelFailure> CheckRmsNormThenMul(const ggml_tensor* norm,
                                                       const ggml_tensor* mul);
// A ggml_add, ggml_mul, ggml_sub or ggml_div node (op), all F32 or all F16,
// with broadcasting: a packed output of src0's shape, over operands packed
// through their last dimension of more than one element or strided views
// with contiguous rows that GGML does not deem contiguous (the launcher
// merges dimensions assuming packed strides only for contiguous operands).
std::expected<void, KernelFailure> CheckBinary(const ggml_tensor* node, ggml_op op);
// A ggml_mul_mat node, F16, BF16 or F32 weights and F32 activations and
// output: what both MMVF and MMF need.
std::expected<void, KernelFailure> CheckMulMat(const ggml_tensor* node);
// CheckMulMat, also admitting F16 activations of F16 weights, which only the
// cuBLAS path reads (directly, as its F16 compute type).
std::expected<void, KernelFailure> CheckMulMatCublasOperands(const ggml_tensor* node);
// CheckMulMat plus MMF's own limits: at most 16 columns and paired strides.
std::expected<void, KernelFailure> CheckMulMatF(const ggml_tensor* node);

// The operations the FP16 bridge's recorded plan launches beyond those
// above (docs/experiments/backend-proof-p0/fp16-plan.json). Upstream file
// and line references are to ggml/src/ggml-cuda/ at llama.cpp b29c606e2.
//
// Several kernels read row or position indices from device memory, which
// no host check can see: get_rows' ids must each name a row of the source,
// and set_rows' (and the fused RoPE's) a row of the destination. An index
// outside is an out-of-bounds access, so the plan that uploads them owns
// that bound.

// A ggml_get_rows node: F32, F16 or BF16 rows gathered by I32 ids into F32
// (ggml_cuda_op_get_rows, getrows.cu:442-459); BF16 is widened exactly.
std::expected<void, KernelFailure> CheckGetRows(const ggml_tensor* node);
// Which kernel get_rows' launcher chooses for a node CheckGetRows accepts:
// k_get_rows_float_vec when rows are whole 16-byte vectors on 16-byte
// aligned rows and the grid has at least 128 blocks, else k_get_rows_float
// (get_rows_cuda_float's can_vec, getrows.cu:256-265). The choice depends on
// the addresses bound, which the recorded plan's conditions include.
bool GetRowsVectorized(const ggml_tensor* node);

// A ggml_set_rows node writing F32 rows into an F16 destination at I64 row
// indices: the KV write (ggml_cuda_op_set_rows, set-rows.cu:376-398).
std::expected<void, KernelFailure> CheckSetRows(const ggml_tensor* node);

// A forward ggml_rope_ext node in NEOX mode over F32, without frequency
// factors or a rotation offset (ggml_cuda_op_rope_impl, rope.cu:536-694).
std::expected<void, KernelFailure> CheckRope(const ggml_tensor* rope);
// The same RoPE fused with the KV write, as GGML's fused launcher runs it
// (ggml_cuda_op_rope_fused, rope.cu:704-706): `set_rows` stores a view of
// `rope` that flattens its heads, and the kernel writes the rotated rows
// straight into set_rows' F16 destination. `rope` and the view are never
// written.
std::expected<void, KernelFailure> CheckRopeSetRows(const ggml_tensor* rope,
                                                    const ggml_tensor* set_rows);

// A ggml_soft_max_ext node over packed F32 rows with an optional F32 mask,
// no sinks and no ALiBi (ggml_cuda_op_soft_max, softmax.cu:383-452). The
// launcher takes a row into shared memory only if it fits the device's
// opt-in limit (softmax.cu:349), which ops.h checks on the device.
std::expected<void, KernelFailure> CheckSoftMax(const ggml_tensor* node);
// The dynamic shared memory soft_max's launcher asks for a row
// (softmax.cu:341): one float per column, padded to a warp, and a warp's
// worth for the reductions.
std::uint64_t SoftMaxSharedBytes(const ggml_tensor* node);

// A ggml_cont node copying F32 into a packed F32 tensor, or contiguous I32
// into I32 (ggml_cuda_dup, cpy.cu:429-617), and how upstream's launcher
// copies it.
enum class ContCopy : std::uint8_t {
  kMemcpy,     // both contiguous: one cudaMemcpyAsync (cpy.cu:467-475)
  kMemcpy2d,   // a contiguous prefix at a fixed pitch: cudaMemcpy2DAsync
  kScalar,     // cpy_scalar<cpy_1_scalar<float, float>>, one thread per element
  kTranspose,  // rows that are transposed columns: cpy_scalar_transpose's tiles
};
std::expected<ContCopy, KernelFailure> CheckCont(const ggml_tensor* node);

// A ggml_swiglu_split node over F32 (ggml_cuda_op_swiglu, unary.cu:287-350):
// silu(gate) * up, element by element.
std::expected<void, KernelFailure> CheckSwiGlu(const ggml_tensor* node);
// Split F32 GELU-tanh(gate) * up with the same row/alias checks; never ERF,
// quick, a packed two-half operand, or swapped GeGLU.
std::expected<void, KernelFailure> CheckGeGlu(const ggml_tensor* node);

// A ggml_cpy node converting a packed F32 tensor into a packed F16 one of
// its shape (rounding to nearest even) or F16 into F32 (exactly):
// ggml_cuda_cpy's cpy_scalar_contiguous<src, dst> (cpy.cu:195-203,
// 495-499, 552-555). The EXL3 plan's casts.
std::expected<void, KernelFailure> CheckConvert(const ggml_tensor* node);

// A ggml_flash_attn_ext node as the EXL3 plan's vector attention takes it
// (docs/experiments/backend-proof-p0/exl3-op-plan.json, `attention`): F32 Q
// viewed [64, rows, heads], F16 K and V viewed [64, cells, KV heads] over
// cells padded to 256, an F16 mask [cells, rows] (rows rounded up to even
// from 1,024, which the mask pre-pass reads), F32 output [64, heads, rows],
// scale only, precision F32. What the forced launcher
// (ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>) and launch_fattn read
// and assert; the scratch it draws is ops.h PlanFlashAttnVec's.
std::expected<void, KernelFailure> CheckFlashAttnVec(const ggml_tensor* node);
// D256 F16 KV, exact GQA2, one sequence; preserves the D64 contract.
std::expected<void, KernelFailure> CheckFlashAttnVec256(const ggml_tensor* node);

// MMVF with GGML's fusion arguments (ggml_cuda_mul_mat_vec_f,
// mmvf.cu:634-729), for one activation column. Beyond CheckMulMat's rules
// on the product's operands:
// - a product and the ggml_add of a bias (or a residual) of its own shape,
//   written to the add's memory; the product is never written;
// - gate and up products of one input with a split SwiGLU over them,
//   written to the GLU's memory; neither product is written. The launcher
//   reads the accumulation precision from the node it writes, so the GLU
//   (whose first parameter is its GLU operation, not a precision) makes
//   F16 weights accumulate in F32, and the add keeps F16 accumulation:
//   the precision rule that fusion changes.
std::expected<void, KernelFailure> CheckMulMatVecBias(const ggml_tensor* mul_mat,
                                                      const ggml_tensor* add);
std::expected<void, KernelFailure> CheckMulMatVecGlu(const ggml_tensor* gate, const ggml_tensor* up,
                                                     const ggml_tensor* glu);
// The same one-column product fusion, with split GELU-tanh GLU.
// Its accumulation is F32; F16 products must explicitly request F32.
// This structural predicate is also used by the planner before binding.
bool MulMatVecGeGluPrecisionFits(const ggml_tensor* gate, const ggml_tensor* up);
std::expected<void, KernelFailure> CheckMulMatVecGeGlu(const ggml_tensor* gate,
                                                       const ggml_tensor* up,
                                                       const ggml_tensor* glu);

// GGML's cuBLAS matrix multiplication (ggml_cuda_mul_mat_cublas_impl in
// ggml-cuda.cu), as it would run for one node: what it converts into
// scratch, which cuBLAS entry point it calls with which leading dimensions,
// and the scratch it draws. The device chooses the compute type and output
// precision (ops.h); this is everything that follows from them.
enum class CublasGemm : std::uint8_t {
  kSgemm,                 // one F32 matrix
  kGemmEx,                // one matrix
  kGemmStridedBatchedEx,  // channels and samples at fixed strides, no broadcast
  kGemmBatchedEx,         // pointer arrays, built on the device into scratch
};
enum class CublasOperand : std::uint8_t {
  kDirect,     // read in place: already the compute type
  kConverted,  // converted element by element into scratch, strides kept
  kPacked,     // gathered into packed scratch
};
struct CublasMulMat {
  ggml_type compute = GGML_TYPE_F32;
  bool f32_output = false;  // cuBLAS writes F32 into the node; otherwise
                            // compute-type output goes through scratch
  CublasOperand weights = CublasOperand::kDirect;
  CublasOperand input = CublasOperand::kDirect;
  CublasGemm gemm = CublasGemm::kGemmEx;
  // Element strides as cuBLAS sees them, after any conversion.
  std::int64_t s01 = 0, s02 = 0, s03 = 0;
  std::int64_t s11 = 0, s12 = 0, s13 = 0;
  // The largest power of two, up to 256, that divides every address and
  // byte stride cuBLAS is given (scratch blocks are 256-aligned). cuBLAS
  // chooses kernels by alignment, so it is part of the executed plan. It
  // counts every stride, even those one matrix leaves unused, so two plans
  // cuBLAS runs alike may record different alignments; never the reverse.
  std::uint64_t alignment = 256;
  // The pool's high-water mark: each block from a 256-byte boundary, in
  // the launcher's allocation order.
  std::uint64_t scratch = 0;
};

// Upstream's kernel family for a matrix product (ggml_cuda_mul_mat); ops.h
// SelectMulMat makes the choice on a device.
enum class MulMatPath : std::uint8_t {
  kVector,      // MMVF, mul_mat_vec_f
  kTensorCore,  // MMF, mul_mat_f (up to 16 columns)
  kCublas,      // GGML's cuBLAS path (mul_mat_cublas.cu)
};

// Whether no byte of the node or its two sources lies in the device range
// [base, base + size): a launcher's scratch or a library's workspace, which
// the launch writes while it reads the operands.
std::expected<void, KernelFailure> CheckClearOf(const ggml_tensor* node, std::uint64_t base,
                                                std::uint64_t size);

// CheckMulMat, then the plan for `compute` (F32, F16 or BF16) and
// `f32_output`, refusing what cuBLAS would refuse (leading dimensions below
// k, more than INT_MAX matrices) and an output that is not packed.
std::expected<CublasMulMat, KernelFailure> CheckMulMatCublas(const ggml_tensor* node,
                                                             ggml_type compute, bool f32_output);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_VALIDATE_H_
