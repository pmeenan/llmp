// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The operand checks of the GGML operations that DeepSeek V4 Flash (GGUF)
// and Qwen3.8 Flash add beyond the backend proof's (validate.h), on the
// host and in every build profile. As there, each refuses what its GGML
// launcher would assert on, abort on or index past, as a kRejected failure:
// operand types and layouts, empty tensors, extents by checked arithmetic,
// the kernels' 32-bit indexing and grid limits, alignment, stale views, and
// outputs that share bytes with an input other than exactly in place. The
// elementwise and row operations take packed operands. Upstream file
// references are to ggml/src/ggml-cuda/ at llama.cpp b29c606e2; the kernel
// family choices that need the device are in ops_ext.h.
//
// Kernels that read indices from device memory (mul_mat_id's expert ids,
// get_rows' and set_rows' row ids) cannot be checked here: an index outside
// its tensor reads or writes out of bounds, so the plan that writes them
// owns the bound, as validate.h says for the backend proof's.

#ifndef JITLLM_KERNELS_GGML_VALIDATE_EXT_H_
#define JITLLM_KERNELS_GGML_VALIDATE_EXT_H_

#include <cstdint>
#include <expected>
#include <span>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// The quantized weight types whose matrix-product kernels this build
// compiles (third_party/patches/ggml/0002's MMQ instance units): DeepSeek V4
// Flash UD-Q2_K_XL's Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS and MXFP4, and
// Qwen3.8 Flash's NVFP4 experts.
std::span<const ggml_type> QuantizedWeightTypes();
bool IsQuantizedWeightType(ggml_type type);

// A weight tensor whose binder vouches that the bytes past its last row are
// readable up to the row's next 512-element step (MATRIX_ROW_PADDING) and
// hold no NaN scale codes (Blackwell's FP4 path multiplies the padding's
// scales by the zero-padded activations' raw; the artifact writes zeros), as
// upstream's buffers pad them (ggml_backend_cuda_buffer_get_alloc_size) and
// an artifact's readable_bytes reserve them (docs/artifact-format.md). Only
// such weights may have rows that are not whole 512-element steps (the
// quantized products' checks below). A flag bit on the tensor itself
// (ggml_tensor::flags, above GGML's own), set after binding; views do not
// inherit it.
void MarkRowPaddingReadable(ggml_tensor* weights);
bool RowPaddingReadable(const ggml_tensor* weights);

// GGML's two kernel families for the quantized products, which the plan
// names as upstream would route them (ops_ext.h SelectMulMatQ).
enum class QuantMulMatPath : std::uint8_t {
  kVector,  // MMVQ, mul_mat_vec_q: up to 8 activation columns (tokens)
  kTile,    // MMQ, mul_mat_q: tensor-core tiles over quantized activations
};

// Quantized matrix products (mmq.cu, mmvq.cu): a ggml_mul_mat node with
// weights of a type above, F32 activations and output, and no routing hint.
// Both kernel families quantize the activations to Q8_1 blocks and read each
// weight row in whole 512-element steps (MATRIX_ROW_PADDING), so rows must be
// multiples of 512 elements: GGML pads a buffer past a shorter row, jitLLM's
// memory does not unless the binder vouches for it (MarkRowPaddingReadable;
// a shorter row's steps then read into the next row, or into that padding
// after the last, against activations the launcher zero-pads). Weights are
// packed rows of whole blocks at a 16-byte
// aligned base; activations and output have F32 rows at any whole-element
// strides, their channels and samples whole multiples of the weights'.
std::expected<void, KernelFailure> CheckMulMatQ(const ggml_tensor* node);

// The same for a ggml_mul_mat_id node: weights [k, n, experts] of a type
// above, activations [k, used or 1, tokens], I32 ids [used, tokens] naming
// an expert in [0, experts) (the plan's bound), output [n, used, tokens]
// (mmq.cu:207-212, mmvq.cu:1418-1423). One sample only.
std::expected<void, KernelFailure> CheckMulMatIdQ(const ggml_tensor* node);

// The same raw quantized expert products with an optional compact tile
// worklist. FP4 preparation remains a separate contract.
std::expected<void, KernelFailure> CheckMulMatIdQCompact(const ggml_tensor* node);

// Two expert products with the same non-FP4 type, shape, activation and
// routing tensors. Both outputs remain distinct; only maps and the
// type-specific quantized activation preparation are shared.
std::expected<void, KernelFailure> CheckMulMatIdQPair(const ggml_tensor* first,
                                                      const ggml_tensor* second);

// A ggml_mul_mat node carrying GGML_HINT_SRC0_IS_HADAMARD, which upstream
// runs as a fast Walsh-Hadamard transform of the activations
// (fwht.cu:61-101, ggml-cuda.cu:1826-1829): F32 activations and output of
// one packed shape, rows of 64, 128, 256 or 512 elements. The weights (the
// normalized Hadamard matrix the hint vouches for) are never read, but must
// be the square [rows, rows] the product's shape needs.
std::expected<void, KernelFailure> CheckMulMatHadamard(const ggml_tensor* node);

// An elementwise function of one packed F32 tensor into a packed F32 tensor
// of its shape, in place or not (unary.cu:138-155): a GGML_OP_UNARY node for
// abs, sgn, neg, silu, tanh, relu, sigmoid, exp or softplus, or a
// GGML_OP_SQRT node. The kernel counts elements in an int.
std::expected<void, KernelFailure> CheckUnary(const ggml_tensor* node);

// ggml_scale and ggml_scale_bias over packed F32 (scale.cu): x * s + b.
std::expected<void, KernelFailure> CheckScale(const ggml_tensor* node);
// ggml_clamp over packed F32 (clamp.cu), in int elements.
std::expected<void, KernelFailure> CheckClamp(const ggml_tensor* node);
// ggml_fill: every element of a packed F32 or F16 tensor set to the node's
// value (fill.cu).
std::expected<void, KernelFailure> CheckFill(const ggml_tensor* node);
// ggml_repeat (ggml_repeat_4d): a packed F32 tensor tiled into a larger
// packed F32 one (binbcast.cu:433-435, the broadcast launcher's limits).
std::expected<void, KernelFailure> CheckRepeat(const ggml_tensor* node);

// ggml_concat along dimension 0 to 3 of two tensors of one 2- or 4-byte
// unblocked type (concat.cu:201-242): whole-element strides, the output's
// rows, channels and samples within the grid.
std::expected<void, KernelFailure> CheckConcat(const ggml_tensor* node);

// ggml_sum_rows over packed F32 rows into one packed F32 value per row
// (sumrows.cu:23-47).
std::expected<void, KernelFailure> CheckSumRows(const ggml_tensor* node);

// ggml_argsort of packed F32 rows into packed I32 indices through the
// bitonic kernel, which upstream takes for rows of at most 1,024 elements
// whose power-of-two padding fits the device's shared memory
// (argsort.cu:255-292). Longer rows take CUB's segmented sort upstream,
// which jitLLM's build does not have (third_party/patches/ggml/0001). The
// device's shared memory is ops_ext.h's to check.
std::expected<void, KernelFailure> CheckArgsort(const ggml_tensor* node);
// The bytes of shared memory the bitonic kernel asks for a row.
std::uint64_t ArgsortSharedBytes(const ggml_tensor* node);

// ggml_top_k of packed F32 rows into packed I32 indices [k, rows], k at most
// the row length (top-k.cu:219-275), in jitLLM's build without CUB: the
// radix select for rows over 1,024 (each row's k indices in no particular
// order), else the bitonic argsort's first k.
std::expected<void, KernelFailure> CheckTopK(const ggml_tensor* node);

// ggml_swiglu_clamp (unary.cu:452-504): silu(min(gate, limit)) times
// clamp(up, -limit, limit), split (two operands) or from one tensor's
// halves, F32 rows contiguous, the output packed.
std::expected<void, KernelFailure> CheckSwiGluClamp(const ggml_tensor* node);

// A ggml_rope_ext, ggml_rope_multi or ggml_rope_ext_back node over F32
// (rope.cu:535-707): normal, NEOX, multi-section (MROPE) or interleaved
// multi-section (IMROPE) rotation of an even part of each head, starting at
// ggml_rope_set_offset's offset, with YaRN's parameters and I32 positions (4
// per token for the multi-section modes); no frequency factors and no
// vision mode. In place or into F32 at any whole-element strides.
std::expected<void, KernelFailure> CheckRopeExt(const ggml_tensor* node);

// ggml_get_rows of rows of a type above (dequantized into F32 rows) or of
// I32 rows (copied into I32), by I32 ids (getrows.cu:442-459). Quantized rows
// are whole 256-element super-blocks, as the dequantizing kernels step.
std::expected<void, KernelFailure> CheckGetRowsExt(const ggml_tensor* node);

// ggml_set_rows of F32 rows into F32 or F16, or of F16 rows into F16, at I32
// or I64 row indices (set-rows.cu:376-398).
std::expected<void, KernelFailure> CheckSetRowsExt(const ggml_tensor* node);

// ggml_ssm_conv (ssm-conv.cu:152-206), unfused: the sliding window
// [conv - 1 + tokens, channels, sequences] of packed F32 rows, the F32
// weights [conv, channels], conv 3, 4, 5, 9 or 15 and channels a multiple
// of 128, into F32 [channels, tokens, sequences]; past 32 tokens, whole
// 32-token blocks only (the long-token kernel loads whole blocks, RE-032).
std::expected<void, KernelFailure> CheckSsmConv(const ggml_tensor* node);

// ggml_gated_delta_net (gated_delta_net.cu:209-320): q and k [S, Hk, T, N]
// with contiguous rows and equal strides, v [S, Hv, T, N], a scalar or
// per-channel gate and beta [1, Hv, T, N] packed, the packed state
// [S, S, Hv, N], S 16, 32, 64 or 128, into the packed output of the
// attention followed by the state snapshots.
std::expected<void, KernelFailure> CheckGatedDeltaNet(const ggml_tensor* node);

// ggml_lightning_indexer (lightning-indexer.cu:400-543) as DeepSeek V4 runs
// it on tensor cores: F32 queries [128, 64 or 32 heads, batch, streams], F16
// keys [128, 1, cells, streams], F32 weights [heads, batch, 1, streams], an
// F16 mask [cells, batch, 1, streams'] (streams a multiple of streams'),
// into F32 scores [cells, batch, 1, streams], rows contiguous.
std::expected<void, KernelFailure> CheckLightningIndexer(const ggml_tensor* node);

// DeepSeek V4's hyper-connections (dsv4-hc.cu), F32 at whole-element
// strides, four streams:
// - comb: mixes [24, tokens], scale [3 or more], base [24] into
//   [4, 4, tokens];
// - pre: x [embd, 4, tokens], weights [4, tokens] into [embd, tokens];
// - post: x [embd, tokens], residual [embd, 4, tokens], post [4, tokens],
//   comb [4, 4, tokens] into [embd, 4, tokens].
std::expected<void, KernelFailure> CheckHcComb(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcPre(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcPost(const ggml_tensor* node);

// A ggml_flash_attn_ext node as the tensor-core (MMA) kernels take it for
// DeepSeek V4 (head dimension 512) and Qwen3.8 (256), grouping 8 query heads
// per KV head (fattn.cu:218-268, fattn-common.cuh:975-1215): F32 Q
// [D, rows, heads, samples], F16 K and V [D, cells, KV heads, samples] with
// cells a multiple of 256, more than four query heads per KV head, an F16
// mask [cells or more, rows or more, 1, samples or 1], optional F32 sinks
// [heads] (only with a multiple of 8 query heads per KV head: the kernel
// reads the sinks a whole group at a time), every non-first stride a
// multiple of 16 bytes, no ALiBi and no logit soft-capping, into the packed
// F32 output [D, heads, rows, samples].
// With ggml_flash_attn_ext_set_n_kv_max's bound (op_params[4]) upstream may
// gather only the unmasked cells (sparse); the mask pre-pass reads whole
// column tiles, which ops_ext.h checks against the mask's rows.
std::expected<void, KernelFailure> CheckFlashAttnMma(const ggml_tensor* node);

// The same kernels at head dimension 128 without head grouping (ncols2 1),
// as upstream takes them for multi-head attention without a mask
// (fattn.cu:170-268: no mask means no GQA grouping): F32 Q [128, rows,
// heads, 1], F16 K and V [128, cells, heads, 1] with any number of cells
// (the last KV tile is bounds-checked, fattn-mma-f16.cuh:1316-1340), no
// mask and no sinks, no ALiBi and no logit soft-capping, every non-first
// stride a multiple of 16 bytes, into the packed F32 output [128, heads,
// rows, 1]. Qwen-Image-2.1's denoiser attends this way.
std::expected<void, KernelFailure> CheckFlashAttnMma128(const ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_VALIDATE_EXT_H_
