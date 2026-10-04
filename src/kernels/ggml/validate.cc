// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/validate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {
namespace {

// The shared building blocks (validate_util.h).
using detail::Aligned;
using detail::AlignedEverywhere;
using detail::AllCurrent;
using detail::AllSane;
using detail::AnyEmpty;
using detail::Bound;
using detail::Current;
using detail::Disjoint;
using detail::ElementStrides;
using detail::Extent;
using detail::Fits32;
using detail::FitsRowGrid;
using detail::IsF32;
using detail::kInt32Max;
using detail::Overlap;
using detail::Packed;
using detail::PackedThroughLastDim;
using detail::ParamF32;
using detail::Product;
using detail::Rejected;
using detail::Root;
using detail::Span;

float Epsilon(const ggml_tensor* norm) {
  float eps = 0.0f;
  std::memcpy(&eps, norm->op_params, sizeof(eps));
  return eps;
}

std::expected<void, KernelFailure> CheckNormSource(const ggml_tensor* norm) {
  if (norm == nullptr || norm->op != GGML_OP_RMS_NORM || !IsF32(norm) || !IsF32(norm->src[0]) ||
      !Bound(norm->src[0])) {
    return Rejected("not a bound F32 rms_norm node");
  }
  if (AnyEmpty({norm, norm->src[0]}) || !Extent(norm->src[0])) {
    return Rejected("rms_norm over an empty or unmeasurable tensor");
  }
  if (!ggml_are_same_shape(norm, norm->src[0]) || !Packed(norm->src[0])) {
    return Rejected("rms_norm over a packed F32 tensor of its own shape");
  }
  // The kernels index with 32-bit arithmetic, and step a row's column
  // index by up to 1,024 past its last element (norm.cu).
  if (!(Epsilon(norm) >= 0.0f) || norm->src[0]->nb[0] != sizeof(float) ||
      !ElementStrides(norm->src[0]) || !FitsRowGrid(norm->src[0]) ||
      ggml_nelements(norm) > std::numeric_limits<std::int32_t>::max() - 1024 ||
      !Aligned(norm->src[0], sizeof(float))) {
    return Rejected(
        "rms_norm needs a non-negative epsilon and aligned, contiguous rows within the grid");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckBinary(const ggml_tensor* node, ggml_op op) {
  if (node == nullptr || node->op != op || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound node of the operation");
  }
  // All F32, or (the attention masks' sums) all F16: two of the type
  // combinations ggml_cuda_op_bin_bcast dispatches (binbcast.cu:412-431).
  const bool f16 = node->type == GGML_TYPE_F16 && node->src[0]->type == GGML_TYPE_F16 &&
                   node->src[1]->type == GGML_TYPE_F16;
  if ((!IsF32(node) || !IsF32(node->src[0]) || !IsF32(node->src[1])) && !f16) {
    return Rejected("this implementation is F32 only, or F16 only");
  }
  const std::uint64_t element = f16 ? sizeof(ggml_fp16_t) : sizeof(float);
  if (AnyEmpty({node, node->src[0], node->src[1]}) ||
      !AllSane({node, node->src[0], node->src[1]})) {
    return Rejected("a binary operation on an empty or unmeasurable tensor");
  }
  // The kernels index each operand by its own strides (s01..s03, s11..s13)
  // and write the output densely; the launcher merges dimensions, assuming
  // packed strides, only when GGML deems both operands contiguous. So the
  // output is packed, and an operand is packed through its last dimension
  // of more than one element (validate_util.h), or a strided view that GGML
  // does not deem contiguous, with contiguous rows.
  const auto operand = [](const ggml_tensor* t) {
    return PackedThroughLastDim(t) ||
           (!ggml_is_contiguous(t) && t->nb[0] == ggml_type_size(t->type));
  };
  if (!ggml_are_same_shape(node, node->src[0]) || !Packed(node) || !operand(node->src[0]) ||
      !operand(node->src[1])) {
    return Rejected(
        "a binary operation over packed operands or strided rows, the output of src0's shape");
  }
  // The launcher collapses contiguous dimensions and may launch one thread
  // per element, asserting that each collapsed extent, stride and the
  // thread count (a multiple of its 128-thread blocks) fit 32 bits.
  if (!ggml_can_repeat(node->src[1], node->src[0]) || !Fits32(node) || !Fits32(node->src[0]) ||
      !Fits32(node->src[1]) || !ElementStrides(node) || !ElementStrides(node->src[0]) ||
      !ElementStrides(node->src[1]) ||
      std::cmp_greater(ggml_nelements(node), std::numeric_limits<std::uint32_t>::max() - 127)) {
    return Rejected("operands that do not broadcast or exceed 32-bit extents");
  }
  if (!Aligned(node, element) || !Aligned(node->src[0], element) ||
      !Aligned(node->src[1], element)) {
    return Rejected("operands at misaligned addresses");
  }
  // The kernel steps through rows one element at a time, whatever the
  // first stride says, and writes the output densely.
  if (!ggml_is_contiguous(node) || node->src[0]->nb[0] != element ||
      node->src[1]->nb[0] != element) {
    return Rejected("a contiguous output over operands with contiguous rows");
  }
  if (!AllCurrent({node, node->src[0], node->src[1]}) ||
      !Disjoint(node, node->src[0], /*in_place=*/true) ||
      !Disjoint(node, node->src[1], /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping an operand other than in place");
  }
  return {};
}

namespace {

// What GGML's matrix-multiply entry points assert, apart from the kernel
// family's own selection, for the product `node` written to `out`: the node
// itself, or with fusion the node the launcher writes instead, which has
// the product's shape (ggml_can_fuse_ext, ggml-impl.h:681-706).
std::expected<void, KernelFailure> CheckMulMatInto(const ggml_tensor* node, const ggml_tensor* out,
                                                   bool f16_input = false) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT || !Bound(out) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound mul_mat node");
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  // cuBLAS alone also reads F16 activations of F16 weights directly.
  const bool input_type = IsF32(input) || (f16_input && input->type == GGML_TYPE_F16 &&
                                           weights != nullptr && weights->type == GGML_TYPE_F16);
  if (!input_type || !IsF32(node) || !IsF32(out) ||
      (weights->type != GGML_TYPE_F16 && weights->type != GGML_TYPE_F32 &&
       weights->type != GGML_TYPE_BF16)) {
    return Rejected("F16, BF16 or F32 weights with F32 activations and output");
  }
  // A hint (op_params[1]) lets upstream route the node to another
  // operation, such as a Hadamard transform.
  std::int32_t hint = 0;
  std::memcpy(&hint, &node->op_params[1], sizeof(hint));
  if (hint != GGML_HINT_NONE) {
    return Rejected("a mul_mat node with a routing hint");
  }
  if (!ggml_are_same_shape(node, out)) {
    return Rejected("the fused output has the product's shape");
  }
  if (input->ne[3] != node->ne[3] || weights->nb[0] != ggml_type_size(weights->type) ||
      input->nb[0] != ggml_type_size(input->type) || out->nb[0] != sizeof(float) ||
      !ElementStrides(weights) || !ElementStrides(input) || !ElementStrides(out)) {
    return Rejected("mul_mat needs contiguous rows and matching samples");
  }
  if (AnyEmpty({out, weights, input}) || !AllSane({out, weights, input})) {
    return Rejected("mul_mat on an empty or unmeasurable tensor");
  }
  // ggml_can_mul_mat (static in ggml.c): input channels and samples are
  // whole multiples of the weights'.
  if (weights->ne[0] != input->ne[0] || input->ne[2] % weights->ne[2] != 0 ||
      input->ne[3] % weights->ne[3] != 0 || node->ne[0] != weights->ne[1] ||
      node->ne[1] != input->ne[1] || node->ne[2] != input->ne[2] || node->ne[3] != input->ne[3]) {
    return Rejected("mul_mat whose shape does not follow from its operands");
  }
  // The kernels take strides and offsets as 32-bit integers, and launch a
  // block per output row, channel and sample.
  constexpr std::uint64_t kInt32 = std::numeric_limits<std::int32_t>::max();
  const auto element_strides_fit = [](const ggml_tensor* tensor) {
    const std::uint64_t size = ggml_type_size(tensor->type);
    return std::ranges::all_of(tensor->nb,
                               [size](std::size_t stride) { return stride / size <= kInt32; });
  };
  if (Span(weights) > kInt32 || Span(input) > kInt32 || Span(out) > kInt32 ||
      !element_strides_fit(weights) || !element_strides_fit(input) || !element_strides_fit(out) ||
      out->ne[2] > 65535 || out->ne[3] > 65535) {
    return Rejected("mul_mat operands beyond the kernels' 32-bit indexing or grid");
  }
  // Weights load as element pairs (half2, nv_bfloat162, float2), and
  // activations and output as float2.
  if (!AlignedEverywhere(weights, 2 * ggml_type_size(weights->type)) ||
      !AlignedEverywhere(input, 8) || !AlignedEverywhere(out, 8)) {
    return Rejected("mul_mat operands at misaligned addresses or strides");
  }
  if (!AllCurrent({out, weights, input}) || !Disjoint(out, weights, /*in_place=*/false) ||
      !Disjoint(out, input, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckMulMat(const ggml_tensor* node) {
  return CheckMulMatInto(node, node);
}

std::expected<void, KernelFailure> CheckMulMatCublasOperands(const ggml_tensor* node) {
  return CheckMulMatInto(node, node, /*f16_input=*/true);
}

std::expected<void, KernelFailure> CheckRmsNorm(const ggml_tensor* norm) {
  if (auto checked = CheckNormSource(norm); !checked) {
    return checked;
  }
  // The kernel writes its output densely, whatever the node's strides.
  if (!Bound(norm) || !ggml_is_contiguous(norm) || !Aligned(norm, sizeof(float))) {
    return Rejected("rms_norm writes an aligned, contiguous F32 tensor");
  }
  if (!AllCurrent({norm, norm->src[0]}) || !Disjoint(norm, norm->src[0], /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping the input other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRmsNormMul(const ggml_tensor* norm,
                                                   const ggml_tensor* mul) {
  if (auto checked = CheckNormSource(norm); !checked) {
    return checked;
  }
  if (mul == nullptr || mul->op != GGML_OP_MUL || !Bound(mul) || !IsF32(mul) ||
      !IsF32(mul->src[0]) || !IsF32(mul->src[1]) || !Fits32(mul)) {
    return Rejected("not a bound F32 mul node");
  }
  // GGML's fusion gate (ggml_cuda_can_fuse) and the fused launcher's checks.
  const ggml_tensor* weight = nullptr;
  if (mul->src[0] == norm) {
    weight = mul->src[1];
  } else if (mul->src[1] == norm) {
    weight = mul->src[0];
    if (!ggml_are_same_shape(weight, norm)) {
      return Rejected("with the norm as the second operand, fusion does not broadcast");
    }
  } else {
    return Rejected("the mul does not scale this norm");
  }
  // The kernel writes the product densely, whatever the node's strides.
  if (!Bound(weight) || ggml_is_empty(weight) || !ggml_is_contiguous_rows(mul->src[0]) ||
      !ggml_is_contiguous_rows(mul->src[1]) || weight->nb[0] != sizeof(float) ||
      !ElementStrides(weight) || !Fits32(weight) || !ggml_is_contiguous(mul)) {
    return Rejected("fusion needs contiguous rows and a contiguous product");
  }
  if (!Aligned(weight, sizeof(float)) || !Aligned(mul, sizeof(float))) {
    return Rejected("fusion operands at misaligned addresses");
  }
  if (!AllSane({weight, mul}) || !Packed(weight) || !Packed(mul) ||
      !ggml_are_same_shape(mul, norm) || !ggml_can_repeat(weight, norm)) {
    return Rejected("fusion over packed operands, the product of the norm's shape");
  }
  // The fused kernel never writes the norm, so the weight must not be the
  // norm, a view of it or memory that overlaps it: upstream fuses only a
  // norm with no other use (ggml_can_fuse_ext).
  // Comparing storage roots covers a weight that is the norm, a view of it,
  // or the storage an in-place norm is a view of.
  if (Root(weight) == Root(norm) || (Bound(norm) && Overlap(weight, norm))) {
    return Rejected("the weight shares the norm's storage, which the fused kernel never writes");
  }
  if (!AllCurrent({norm, norm->src[0], mul, weight}) ||
      !Disjoint(mul, norm->src[0], /*in_place=*/true) ||
      !Disjoint(mul, weight, /*in_place=*/true)) {
    return Rejected("a stale view, or a product overlapping an operand other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRmsNormThenMul(const ggml_tensor* norm,
                                                       const ggml_tensor* mul) {
  if (mul == nullptr || mul->op != GGML_OP_MUL || norm == nullptr ||
      (mul->src[0] != norm && mul->src[1] != norm)) {
    return Rejected("not a mul node scaling this norm");
  }
  if (auto checked = CheckRmsNorm(norm); !checked) {
    return checked;
  }
  if (auto checked = CheckBinary(mul, GGML_OP_MUL); !checked) {
    return checked;
  }
  // The norm is this implementation's own intermediate. Written over its
  // input, it would change what the fused implementation leaves alone;
  // over the weight, it would change what the mul then reads.
  const ggml_tensor* weight = mul->src[0] == norm ? mul->src[1] : mul->src[0];
  if (Overlap(norm, norm->src[0]) || Overlap(norm, weight)) {
    return Rejected("the norm's memory overlaps its input or the weight");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMulMatF(const ggml_tensor* node) {
  if (auto checked = CheckMulMat(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  if (node->src[1]->ne[1] > 16) {
    return Rejected("MMF takes at most 16 activation columns");
  }
  // The launcher counts F16 and BF16 strides in pairs and asserts them even,
  // which upstream's selection does not fully check.
  const std::uint64_t pair = weights->type == GGML_TYPE_F32 ? 2 : 4;
  if ((weights->nb[1] / ggml_type_size(weights->type)) % pair != 0 ||
      (node->src[1]->nb[1] / sizeof(float)) % pair != 0) {
    return Rejected("MMF needs even weight row and activation column strides");
  }
  return {};
}

std::expected<void, KernelFailure> CheckClearOf(const ggml_tensor* node, std::uint64_t base,
                                                std::uint64_t size) {
  if (size == 0) {
    return {};
  }
  for (const ggml_tensor* tensor :
       std::initializer_list<const ggml_tensor*>{node, node->src[0], node->src[1]}) {
    if (tensor == nullptr) {
      continue;
    }
    const auto extent = Extent(tensor);
    if (!extent) {
      return Rejected("an operand that cannot be measured");
    }
    const auto begin = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data));
    if (begin < base + size && base < begin + *extent) {
      return Rejected("an operand overlaps a workspace the launch writes");
    }
  }
  return {};
}

std::expected<CublasMulMat, KernelFailure> CheckMulMatCublas(const ggml_tensor* node,
                                                             ggml_type compute, bool f32_output) {
  if (auto checked = CheckMulMatCublasOperands(node); !checked) {
    return std::unexpected(checked.error());
  }
  if (compute != GGML_TYPE_F32 && compute != GGML_TYPE_F16 && compute != GGML_TYPE_BF16) {
    return Rejected("cuBLAS computes in F32, F16 or BF16");
  }
  const ggml_tensor* src0 = node->src[0];
  const ggml_tensor* src1 = node->src[1];
  // F16 activations stand for the F32 ones the F16 compute type would round
  // them to; another compute type would read the unrounded F32 values.
  if (src1->type == GGML_TYPE_F16 && compute != GGML_TYPE_F16) {
    return Rejected("F16 activations are read only with the F16 compute type");
  }
  // The launcher asserts a contiguous output and indexes it as packed.
  if (!Packed(node)) {
    return Rejected("cuBLAS writes a packed output");
  }
  // CheckMulMat bounds every extent and element stride to 32 bits, so the
  // products below cannot overflow 64.
  const std::uint64_t compute_size = ggml_type_size(compute);
  const std::uint64_t ts0 = ggml_type_size(src0->type);
  const std::uint64_t ts1 = ggml_type_size(src1->type);
  CublasMulMat plan{.compute = compute, .f32_output = f32_output};
  plan.s01 = static_cast<std::int64_t>(src0->nb[1] / ts0);
  plan.s02 = static_cast<std::int64_t>(src0->nb[2] / ts0);
  plan.s03 = static_cast<std::int64_t>(src0->nb[3] / ts0);
  plan.s11 = static_cast<std::int64_t>(src1->nb[1] / ts1);
  plan.s12 = static_cast<std::int64_t>(src1->nb[2] / ts1);
  plan.s13 = static_cast<std::int64_t>(src1->nb[3] / ts1);
  bool src0_cont_2 = ggml_is_contiguous_2(src0);
  bool src1_cont_2 = ggml_is_contiguous_2(src1);

  // The pool hands out blocks from 256-byte boundaries (launch.cu), in the
  // launcher's order: weights, input, output, then the pointer arrays.
  const auto draw = [&plan](std::uint64_t bytes) {
    plan.scratch = ((plan.scratch + 255) / 256 * 256) + bytes;
  };
  // Each operand that is not already the compute type is converted into
  // scratch: element by element if its bytes are exactly its elements
  // (strides kept, blocks of one element), else gathered into packed rows.
  if (src0->type != compute) {
    draw(static_cast<std::uint64_t>(ggml_nelements(src0)) * compute_size);
    if (ggml_is_contiguously_allocated(src0)) {
      plan.weights = CublasOperand::kConverted;
    } else {
      plan.weights = CublasOperand::kPacked;
      plan.s01 = src0->ne[0];
      plan.s02 = src0->ne[1] * plan.s01;
      plan.s03 = src0->ne[2] * plan.s02;
      src0_cont_2 = true;
    }
  }
  if (src1->type != compute) {
    draw(static_cast<std::uint64_t>(ggml_nelements(src1)) * compute_size);
    if (ggml_is_contiguously_allocated(src1)) {
      plan.input = CublasOperand::kConverted;
    } else {
      plan.input = CublasOperand::kPacked;
      plan.s11 = src1->ne[0];
      plan.s12 = src1->ne[1] * plan.s11;
      plan.s13 = src1->ne[2] * plan.s12;
      src1_cont_2 = true;
    }
  }
  if (!f32_output && compute != GGML_TYPE_F32) {
    draw(static_cast<std::uint64_t>(ggml_nelements(node)) * compute_size);
  }

  // What cuBLAS is given: each operand's base (a scratch block's is 256-
  // aligned) and strides in the type it reads, and the output's.
  const auto align = [&plan](std::uint64_t value) {
    if (value != 0) {
      plan.alignment = std::min(plan.alignment, value & (~value + 1));
    }
  };
  const auto operand = [&align](const ggml_tensor* src, CublasOperand how, std::uint64_t size,
                                std::initializer_list<std::int64_t> strides) {
    align(how == CublasOperand::kDirect
              ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(src->data))
              : 256);
    for (const std::int64_t stride : strides) {
      align(static_cast<std::uint64_t>(stride) * size);
    }
  };
  operand(src0, plan.weights, plan.weights == CublasOperand::kDirect ? ts0 : compute_size,
          {plan.s01, plan.s02, plan.s03});
  operand(src1, plan.input, plan.input == CublasOperand::kDirect ? ts1 : compute_size,
          {plan.s11, plan.s12, plan.s13});
  const bool output_direct = f32_output || compute == GGML_TYPE_F32;
  const std::uint64_t output_size = output_direct ? sizeof(float) : compute_size;
  align(output_direct ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(node->data))
                      : 256);
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    align(node->nb[i] / sizeof(float) * output_size);
  }

  // cuBLAS reads the weights transposed: each of its columns is a weight
  // row, so both operands' leading dimensions must reach k (it refuses
  // less), and it takes the batch count as an int.
  const std::int64_t k = src0->ne[0];
  const std::int64_t batches = src1->ne[2] * src1->ne[3];
  if (plan.s01 < k || plan.s11 < k) {
    return Rejected("cuBLAS needs rows at least k elements apart in both operands");
  }
  if (batches > std::numeric_limits<std::int32_t>::max()) {
    return Rejected("more matrices than cuBLAS takes in one call");
  }
  const bool broadcast = src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3];
  if (src1->ne[2] == 1 && src1->ne[3] == 1) {
    plan.gemm = compute == GGML_TYPE_F32 ? CublasGemm::kSgemm : CublasGemm::kGemmEx;
  } else if (!broadcast && src0_cont_2 && src1_cont_2) {
    plan.gemm = CublasGemm::kGemmStridedBatchedEx;
  } else {
    plan.gemm = CublasGemm::kGemmBatchedEx;
    // Two input and one output pointer per matrix.
    draw(2 * static_cast<std::uint64_t>(batches) * sizeof(void*));
    draw(static_cast<std::uint64_t>(batches) * sizeof(void*));
  }
  return plan;
}

namespace {

// What jitLLM's RoPE takes, fused or not: forward NEOX over F32 rows with
// I32 positions, optional packed F32 frequency factors and no rotation offset, and what the
// launcher and kernel assume of the input (rope.cu:122-197, 401-445,
// 536-694). The node itself is checked by the caller: the fused launcher
// never writes it.
std::expected<void, KernelFailure> CheckRopeInput(const ggml_tensor* rope) {
  if (rope == nullptr || rope->op != GGML_OP_ROPE || !Bound(rope->src[0]) || !Bound(rope->src[1])) {
    return Rejected("not a rope node over bound operands");
  }
  const ggml_tensor* x = rope->src[0];
  const ggml_tensor* positions = rope->src[1];
  if (!IsF32(x) || !IsF32(rope) || positions->type != GGML_TYPE_I32) {
    return Rejected("RoPE over F32 with I32 positions");
  }
  // op_params: n_dims, mode and the offset ggml_rope_set_offset sets
  // (ggml_rope_impl, ggml.c:4306-4318).
  const std::int32_t n_dims = rope->op_params[1];
  const std::int32_t mode = rope->op_params[2];
  const std::int32_t offset = rope->op_params[15];
  if (mode != GGML_ROPE_TYPE_NEOX || offset != 0) {
    return Rejected("NEOX RoPE without a rotation offset");
  }
  if (AnyEmpty({rope, x, positions}) || !AllSane({x, positions})) {
    return Rejected("RoPE on an empty or unmeasurable tensor");
  }
  // The launcher asserts an even head (rope.cu:427); the kernel rotates
  // pairs n_dims / 2 apart within it.
  if (!ggml_are_same_shape(rope, x) || x->ne[0] % 2 != 0 || n_dims <= 0 || n_dims % 2 != 0 ||
      n_dims > x->ne[0]) {
    return Rejected("RoPE over even heads, rotating an even part of each");
  }
  // One position per token (ggml_rope_impl, ggml.c:4288-4297), read densely.
  if (!ggml_is_vector(positions) || positions->ne[0] != x->ne[2] ||
      positions->nb[0] != sizeof(std::int32_t) || !Aligned(positions, sizeof(std::int32_t))) {
    return Rejected("RoPE takes one packed position per token");
  }
  // The launcher passes extents and element strides as int and launches a
  // block per row and a 256-thread column per 512 elements of the head;
  // the kernel indexes in int (rope.cu:139-151, 427-430).
  if (x->nb[0] != sizeof(float) || !ElementStrides(x) || Span(x) > kInt32Max ||
      ggml_nrows(x) > static_cast<std::int64_t>(kInt32Max) || (x->ne[0] + 511) / 512 > 65535 ||
      !Aligned(x, sizeof(float))) {
    return Rejected("RoPE input beyond the kernel's 32-bit indexing or grid");
  }
  if (!Current(x) || !Current(positions)) {
    return Rejected("a stale view");
  }
  return CheckRopeFrequencyFactors(rope);
}

// ggml_cuda_cpy_as_memcpy_2d (static in cpy.cu:391-427): whether the copy
// is a contiguous block of each row at a fixed pitch.
bool CopiesAsMemcpy2d(const ggml_tensor* src, const ggml_tensor* dst) {
  if (src->type != dst->type || !ggml_are_same_shape(src, dst)) {
    return false;
  }
  std::uint64_t block = ggml_element_size(src);
  int d = 0;
  for (; d < GGML_MAX_DIMS; ++d) {
    if (src->nb[d] != block || dst->nb[d] != block) {
      break;
    }
    block *= static_cast<std::uint64_t>(src->ne[d]);
  }
  if (d == 0 || d == GGML_MAX_DIMS) {
    return false;
  }
  for (int i = d + 1; i < GGML_MAX_DIMS; ++i) {
    if (src->ne[i] != 1) {
      return false;
    }
  }
  return src->nb[d] >= block && dst->nb[d] >= block;
}

}  // namespace

std::expected<void, KernelFailure> CheckGetRows(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_GET_ROWS || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound get_rows node");
  }
  const ggml_tensor* rows = node->src[0];
  const ggml_tensor* ids = node->src[1];
  // F32 rows, F16 APE tables, or BF16 embedding tables, gathered into F32; or I32 token
  // maps, gathered without conversion.
  const bool bf16 = rows->type == GGML_TYPE_BF16;
  const bool f16 = rows->type == GGML_TYPE_F16;
  const bool i32 = rows->type == GGML_TYPE_I32;
  if ((!IsF32(rows) && !bf16 && !f16 && !i32) ||
      (i32 ? node->type != GGML_TYPE_I32 : !IsF32(node)) || ids->type != GGML_TYPE_I32) {
    return Rejected("get_rows gathers F32, F16 or BF16 into F32, or I32 into I32, by I32 ids");
  }
  const std::uint64_t element = bf16 || f16 ? sizeof(ggml_fp16_t) : sizeof(float);
  if (AnyEmpty({node, rows, ids}) || !AllSane({node, rows, ids})) {
    return Rejected("get_rows on an empty or unmeasurable tensor");
  }
  // ggml_get_rows's shape (ggml.c:3959-3980); the launcher asserts one
  // sample of ids and contiguous rows (getrows.cu:450-455).
  if (rows->ne[2] != ids->ne[1] || rows->ne[3] != ids->ne[2] || ids->ne[3] != 1 ||
      node->ne[0] != rows->ne[0] || node->ne[1] != ids->ne[0] || node->ne[2] != ids->ne[1] ||
      node->ne[3] != ids->ne[2]) {
    return Rejected("get_rows whose shape does not follow from its operands");
  }
  if (rows->nb[0] != element || ids->nb[0] != sizeof(std::int32_t) ||
      node->nb[0] != sizeof(float) || !ElementStrides(rows) || !ElementStrides(ids) ||
      !ElementStrides(node)) {
    return Rejected("get_rows needs contiguous rows and whole-element strides");
  }
  // A block column per id, and the ids' channels and samples counted in 32
  // bits (getrows.cu:253-254).
  const auto planes = Product({ids->ne[1], ids->ne[2]});
  if (std::cmp_greater(ids->ne[0], kInt32Max) || !planes ||
      *planes > std::numeric_limits<std::uint32_t>::max()) {
    return Rejected("get_rows ids beyond the launcher's grid");
  }
  if (!Aligned(rows, element) || !Aligned(ids, sizeof(std::int32_t)) ||
      !Aligned(node, sizeof(float))) {
    return Rejected("get_rows operands at misaligned addresses");
  }
  if (!AllCurrent({node, rows, ids}) || !Disjoint(node, rows, /*in_place=*/false) ||
      !Disjoint(node, ids, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

bool GetRowsVectorized(const ggml_tensor* node) {
  const ggml_tensor* rows = node->src[0];
  const ggml_tensor* ids = node->src[1];
  // Only a copy that keeps the element type vectorizes (getrows.cu:250).
  if (rows->type != node->type) {
    return false;
  }
  constexpr std::int64_t kVector = 16 / sizeof(float);
  const std::int64_t vectors = rows->ne[0] / kVector;
  const std::int64_t blocks_y = (vectors + 255) / 256;
  // An overflowing count is certainly enough blocks.
  const auto blocks = Product({blocks_y, ids->ne[0], ids->ne[1], ids->ne[2]});
  const bool enough_blocks = blocks_y > 0 && (!blocks || *blocks >= 128);
  const auto at16 = [](const ggml_tensor* tensor) {
    return reinterpret_cast<std::uintptr_t>(tensor->data) % 16 == 0 && tensor->nb[1] % 16 == 0 &&
           tensor->nb[2] % 16 == 0 && tensor->nb[3] % 16 == 0;
  };
  return enough_blocks && rows->ne[0] % kVector == 0 && at16(rows) && at16(node);
}

std::expected<void, KernelFailure> CheckSetRows(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_SET_ROWS || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound set_rows node");
  }
  const ggml_tensor* values = node->src[0];
  const ggml_tensor* ids = node->src[1];
  if (!IsF32(values) || node->type != GGML_TYPE_F16 || ids->type != GGML_TYPE_I64) {
    return Rejected("set_rows writes F32 rows into F16 at I64 indices");
  }
  if (AnyEmpty({node, values, ids}) || !AllSane({node, values, ids})) {
    return Rejected("set_rows on an empty or unmeasurable tensor");
  }
  // ggml_set_rows's shape (ggml.c:4005-4031).
  if (node->ne[0] != values->ne[0] || node->ne[2] != values->ne[2] ||
      node->ne[3] != values->ne[3] || values->ne[1] != ids->ne[0] ||
      values->ne[2] % ids->ne[1] != 0 || values->ne[3] % ids->ne[2] != 0 || ids->ne[3] != 1) {
    return Rejected("set_rows whose shape does not follow from its operands");
  }
  if (values->nb[0] != sizeof(float) || node->nb[0] != sizeof(ggml_fp16_t) ||
      !ElementStrides(values) || !ElementStrides(ids) || !ElementStrides(node)) {
    return Rejected("set_rows needs contiguous rows and whole-element strides");
  }
  // One thread per value, whose flat index the kernel splits with 32-bit
  // division (set-rows.cu:136-157, 188-189).
  if (std::cmp_greater(ggml_nelements(values), std::numeric_limits<std::uint32_t>::max())) {
    return Rejected("set_rows beyond the kernel's 32-bit indexing");
  }
  if (!Aligned(values, sizeof(float)) || !Aligned(ids, sizeof(std::int64_t)) ||
      !Aligned(node, sizeof(ggml_fp16_t))) {
    return Rejected("set_rows operands at misaligned addresses");
  }
  if (!AllCurrent({node, values, ids}) || !Disjoint(node, values, /*in_place=*/false) ||
      !Disjoint(node, ids, /*in_place=*/false)) {
    return Rejected("a stale view, or a destination overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRopeFrequencyFactors(const ggml_tensor* rope) {
  if (rope == nullptr) return Rejected("no RoPE descriptor");
  const auto* f = rope->src[2];
  if (f == nullptr) return {};
  if (rope->op != GGML_OP_ROPE || rope->op_params[2] != GGML_ROPE_TYPE_NEOX ||
      rope->op_params[15] != 0 || rope->op_params[1] <= 0 || rope->op_params[1] % 2 != 0 ||
      !Bound(f) || !IsF32(f) || !Extent(f) || !ggml_is_vector(f) ||
      f->ne[0] != rope->op_params[1] / 2 || !Packed(f) || Span(f) > kInt32Max ||
      !Aligned(f, sizeof(float)) || !Current(f)) {
    return Rejected("forward NEOX RoPE takes a current packed F32 factor per rotated pair");
  }
  for (const auto* view = f; view->view_src != nullptr; view = view->view_src) {
    const auto* parent = view->view_src;
    const auto child_bytes = Extent(view);
    const auto parent_bytes = IsF32(parent) ? Extent(parent) : std::nullopt;
    if (!child_bytes || !parent_bytes || view->view_offs > *parent_bytes ||
        *child_bytes > *parent_bytes - view->view_offs) {
      return Rejected("RoPE factor view exceeds its source storage");
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckRope(const ggml_tensor* rope) {
  if (auto checked = CheckRopeInput(rope); !checked) {
    return checked;
  }
  const ggml_tensor* x = rope->src[0];
  if (!Bound(rope) || !Extent(rope) || rope->nb[0] != sizeof(float) || !ElementStrides(rope) ||
      Span(rope) > kInt32Max || !Aligned(rope, sizeof(float))) {
    return Rejected("RoPE writes a bound F32 tensor within the kernel's 32-bit indexing");
  }
  // Each thread rotates one pair and writes it where it read it, so the
  // output may be exactly the input (upstream's in-place case, rope.cu:588).
  if (!Current(rope) || !Disjoint(rope, x, /*in_place=*/true) ||
      !Disjoint(rope, rope->src[1], /*in_place=*/false) ||
      (rope->src[2] != nullptr && !Disjoint(rope, rope->src[2], /*in_place=*/false))) {
    return Rejected("a stale view, or an output overlapping an input other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRopeSetRows(const ggml_tensor* rope,
                                                    const ggml_tensor* set_rows) {
  if (auto checked = CheckRopeInput(rope); !checked) {
    return checked;
  }
  if (set_rows == nullptr || set_rows->op != GGML_OP_SET_ROWS || !Bound(set_rows) ||
      !Bound(set_rows->src[1])) {
    return Rejected("not a bound set_rows node");
  }
  const ggml_tensor* x = rope->src[0];
  const ggml_tensor* view = set_rows->src[0];
  const ggml_tensor* ids = set_rows->src[1];
  if (view == nullptr || view->op != GGML_OP_VIEW || view->view_src != rope ||
      rope->view_src != nullptr) {
    return Rejected("set_rows stores a view of this RoPE, which is no view itself");
  }
  // Upstream's gate (ggml_cuda_should_fuse_rope_set_rows,
  // ggml-cuda.cu:2666-2698), for the F16 destination jitLLM's KV write has.
  if (x->ne[3] != 1 || set_rows->type != GGML_TYPE_F16 || ids->type != GGML_TYPE_I64 ||
      !ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
    return Rejected("RoPE and a KV write that GGML's fused launcher does not take");
  }
  // The kernel places head h of a token at h * (rope->nb[1] / 4) in the row
  // that token's index names (rope.cu:158-167): the view must be exactly
  // each token's heads in the RoPE's own layout, which a packed RoPE's
  // flattening view is.
  if (!Packed(rope) || view->view_offs != 0 || view->ne[1] != rope->ne[2] || view->ne[2] != 1 ||
      view->ne[3] != 1 || view->nb[1] != rope->nb[2]) {
    return Rejected("the stored view must flatten a packed RoPE's heads, one row per token");
  }
  if (AnyEmpty({set_rows, ids}) || !AllSane({set_rows, ids})) {
    return Rejected("a KV write to an empty or unmeasurable tensor");
  }
  // ggml_set_rows's shape (ggml.c:4005-4031) for a two-dimensional
  // destination, with one packed index per token read by token
  // (rope.cu:163-166).
  if (set_rows->ne[0] != view->ne[0] || set_rows->ne[2] != 1 || set_rows->ne[3] != 1 ||
      ids->ne[0] != view->ne[1] || !ggml_is_vector(ids) || ids->nb[0] != sizeof(std::int64_t) ||
      !Aligned(ids, sizeof(std::int64_t))) {
    return Rejected("a KV write of one row per token into a two-dimensional destination");
  }
  // The kernel's destination index is an int.
  if (set_rows->nb[0] != sizeof(ggml_fp16_t) || !ElementStrides(set_rows) ||
      Span(set_rows) > kInt32Max || !Aligned(set_rows, sizeof(ggml_fp16_t))) {
    return Rejected("a KV write beyond the kernel's 32-bit indexing");
  }
  if (!AllCurrent({set_rows, ids}) || !Disjoint(set_rows, x, /*in_place=*/false) ||
      !Disjoint(set_rows, rope->src[1], /*in_place=*/false) ||
      !Disjoint(set_rows, ids, /*in_place=*/false) ||
      (rope->src[2] != nullptr && !Disjoint(set_rows, rope->src[2], /*in_place=*/false))) {
    return Rejected("a stale view, or a destination overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSoftMax(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_SOFT_MAX || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound soft_max node");
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* mask = node->src[1];
  if (!IsF32(x) || !IsF32(node) || (mask != nullptr && !IsF32(mask)) || node->src[2] != nullptr) {
    return Rejected("soft_max over F32 with an optional F32 mask and no sinks");
  }
  // op_params: scale, then the ALiBi bias (ggml_soft_max_impl,
  // ggml.c:4150-4185).
  if (ParamF32(node, 1) != 0.0f) {
    return Rejected("soft_max without ALiBi");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x})) {
    return Rejected("soft_max on an empty or unmeasurable tensor");
  }
  // The kernel steps through packed rows (softmax.cu:66-74).
  if (!ggml_are_same_shape(node, x) || !Packed(x) || !Packed(node)) {
    return Rejected("soft_max over packed rows into a packed tensor of their shape");
  }
  // A block per row, channel and sample; rows and columns counted in int
  // (softmax.cu:59-66, 339-340).
  if (std::cmp_greater(x->ne[0], kInt32Max) || std::cmp_greater(ggml_nrows(x), kInt32Max) ||
      x->ne[2] > 65535 || x->ne[3] > 65535) {
    return Rejected("soft_max beyond the kernel's 32-bit indexing or grid");
  }
  if (!Aligned(x, sizeof(float)) || !Aligned(node, sizeof(float))) {
    return Rejected("soft_max operands at misaligned addresses");
  }
  if (mask != nullptr) {
    // The mask's rows broadcast over channels and samples (ggml.c:4160-4166;
    // softmax.cu:69-73).
    if (!Bound(mask) || ggml_is_empty(mask) || !Extent(mask) || mask->ne[0] != x->ne[0] ||
        mask->ne[1] < x->ne[1] || x->ne[2] % mask->ne[2] != 0 || x->ne[3] % mask->ne[3] != 0 ||
        mask->nb[0] != sizeof(float) || !ElementStrides(mask) || !Aligned(mask, sizeof(float))) {
      return Rejected("a mask of whole rows over the scores' columns");
    }
    if (!Current(mask) || !Disjoint(node, mask, /*in_place=*/false)) {
      return Rejected("a stale mask, or an output overlapping it");
    }
  }
  // A block reads its whole row into shared memory before it writes, so
  // the output may be exactly the input.
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping the input other than in place");
  }
  return {};
}

std::uint64_t SoftMaxSharedBytes(const ggml_tensor* node) {
  constexpr std::uint64_t kWarp = 32;
  const auto columns = static_cast<std::uint64_t>(node->src[0]->ne[0]);
  return (((columns + kWarp - 1) / kWarp * kWarp) + kWarp) * sizeof(float);
}

std::expected<ContCopy, KernelFailure> CheckCont(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_CONT || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound cont node");
  }
  const ggml_tensor* src = node->src[0];
  // F32 into F32; or I32 into I32 (the indexer's top-k), which upstream
  // copies with one cudaMemcpyAsync when both are contiguous (cpy.cu:465-475)
  // and which this implementation takes only then.
  const bool i32 = src->type == GGML_TYPE_I32 && node->type == GGML_TYPE_I32;
  if ((!IsF32(src) || !IsF32(node)) && !i32) {
    return Rejected("cont copies F32 into F32, or contiguous I32 into I32");
  }
  if (AnyEmpty({node, src}) || !AllSane({node, src})) {
    return Rejected("cont on an empty or unmeasurable tensor");
  }
  if (ggml_nelements(node) != ggml_nelements(src) || !Packed(node) || !ElementStrides(src) ||
      !Aligned(src, sizeof(float)) || !Aligned(node, sizeof(float))) {
    return Rejected("cont into a packed tensor of as many elements, from whole elements");
  }
  if (i32 && (!ggml_is_contiguous(src) || ggml_nbytes(src) != ggml_nbytes(node))) {
    return Rejected("cont of I32 only between contiguous tensors");
  }
  // The scalar kernel's blocks of 64 are counted in an int (cpy.cu:212-213).
  if (ggml_nelements(src) / 64 >= static_cast<std::int64_t>(kInt32Max)) {
    return Rejected("cont beyond the launcher's grid");
  }
  if (!AllCurrent({node, src}) || !Disjoint(node, src, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping the input");
  }
  // ggml_cuda_cpy's order (cpy.cu:460-486).
  if (ggml_is_contiguous(src) && ggml_is_contiguous(node)) {
    return ContCopy::kMemcpy;
  }
  if (CopiesAsMemcpy2d(src, node)) {
    return ContCopy::kMemcpy2d;
  }
  // A source whose rows are transposed columns: upstream's tiled transpose
  // (cpy.cu:461-463, 480-482; ggml_cpy_scalar_cuda<float, float, true>),
  // which asserts that the source is its first three dimensions and writes
  // the packed output; its grid falls back to the scalar kernel past 65,535
  // tiles in y or z, which upstream decides alike. The tile kernel reads
  // element (i0, i1) of matrix i2 at (i2·ne0·ne1 + i0·ne1 + i1) elements,
  // whatever nb[0] says (cpy.cu:74-78): only a packed matrix's transpose,
  // nb[0] = ne1 elements, is addressed correctly. Upstream's condition does
  // not test nb[0], so any other source it would tile is refused.
  if (src->nb[1] == sizeof(float) && src->ne[3] == 1 &&
      src->nb[2] == static_cast<std::size_t>(src->ne[0] * src->ne[1]) * sizeof(float)) {
    if (!ggml_is_contiguous(node) || ggml_nelements(src) != src->ne[0] * src->ne[1] * src->ne[2]) {
      return Rejected("a transposed cont into a non-contiguous tensor");
    }
    if (src->ne[0] > 1 && src->nb[0] != static_cast<std::size_t>(src->ne[1]) * sizeof(float)) {
      return Rejected("a transposed cont of anything but a packed matrix's transpose");
    }
    return ContCopy::kTranspose;
  }
  return ContCopy::kScalar;
}

namespace {
std::expected<void, KernelFailure> CheckSplitGlu(const ggml_tensor* node, ggml_glu_op op) {
  if (node == nullptr || node->op != GGML_OP_GLU || ggml_get_glu_op(node) != op || !Bound(node) ||
      !Bound(node->src[0]) || !Bound(node->src[1])) {
    return Rejected("not a bound split GLU node");
  }
  if (op == GGML_GLU_OP_GEGLU && node->op_params[1] != 0) {
    return Rejected("swapped GeGLU is not implemented");
  }
  const ggml_tensor* gate = node->src[0];
  const ggml_tensor* up = node->src[1];
  if (!IsF32(gate) || !IsF32(up) || !IsF32(node)) {
    return Rejected("GLU over F32");
  }
  if (AnyEmpty({node, gate, up}) || !AllSane({node, gate, up})) {
    return Rejected("GLU on an empty or unmeasurable tensor");
  }
  // ggml_glu_impl (ggml.c:2911-2935) and the launcher's asserts
  // (unary.cu:298-312): the kernel finds row r of each input r row strides
  // in, and writes the output densely.
  if (!ggml_are_same_shape(gate, up) || !ggml_are_same_shape(node, gate) ||
      !ggml_is_contiguous_1(gate) || !ggml_is_contiguous_1(up) || gate->nb[0] != sizeof(float) ||
      up->nb[0] != sizeof(float) || !ElementStrides(gate) || !ElementStrides(up) || !Packed(node)) {
    return Rejected("GLU over inputs of uniform rows into a packed output of their shape");
  }
  // Blocks of 256 threads (unary.cu:280-283).
  if (ggml_nelements(node) / 256 >= static_cast<std::int64_t>(kInt32Max)) {
    return Rejected("GLU beyond the launcher's grid");
  }
  if (!Aligned(gate, sizeof(float)) || !Aligned(up, sizeof(float)) ||
      !Aligned(node, sizeof(float))) {
    return Rejected("GLU operands at misaligned addresses");
  }
  // Each thread reads its element of both inputs and then writes it.
  if (!AllCurrent({node, gate, up}) || !Disjoint(node, gate, /*in_place=*/true) ||
      !Disjoint(node, up, /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping an input other than in place");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckSwiGlu(const ggml_tensor* node) {
  return CheckSplitGlu(node, GGML_GLU_OP_SWIGLU);
}
std::expected<void, KernelFailure> CheckGeGlu(const ggml_tensor* node) {
  return CheckSplitGlu(node, GGML_GLU_OP_GEGLU);
}

std::expected<void, KernelFailure> CheckMulMatVecBias(const ggml_tensor* mul_mat,
                                                      const ggml_tensor* add) {
  if (mul_mat == nullptr || add == nullptr || add->op != GGML_OP_ADD) {
    return Rejected("not an add of a product");
  }
  const ggml_tensor* bias = nullptr;
  if (add->src[0] == mul_mat) {
    bias = add->src[1];
  } else if (add->src[1] == mul_mat) {
    bias = add->src[0];
  }
  if (bias == nullptr || bias == mul_mat) {
    return Rejected("the add does not take this product once");
  }
  if (auto checked = CheckMulMatInto(mul_mat, add); !checked) {
    return checked;
  }
  // One activation column (mmvf.cu:665), and a bias of the product's
  // shape, which the kernel reads with the output's strides
  // (ggml-cuda.cu:4109; mmvf.cu:89-93, 351).
  if (add->ne[1] != 1) {
    return Rejected("fused MMVF takes one activation column");
  }
  if (!IsF32(bias) || !Bound(bias) || ggml_is_empty(bias) || !Extent(bias) ||
      !ggml_are_same_shape(bias, mul_mat) || !Packed(bias) || !Packed(add) ||
      !Aligned(bias, sizeof(float))) {
    return Rejected("a packed F32 bias of the product's shape, added into a packed output");
  }
  // The thread that writes an output element reads its bias element first.
  if (!Current(bias) || !Disjoint(add, bias, /*in_place=*/true)) {
    return Rejected("a stale bias, or an output overlapping it other than in place");
  }
  return {};
}

namespace {
std::expected<void, KernelFailure> CheckMulMatVecSplitGlu(const ggml_tensor* gate,
                                                          const ggml_tensor* up,
                                                          const ggml_tensor* glu, ggml_glu_op op) {
  if (gate == nullptr || up == nullptr || glu == nullptr || glu->op != GGML_OP_GLU ||
      ggml_get_glu_op(glu) != op || glu->src[0] != gate || glu->src[1] != up ||
      glu->op_params[1] != 0) {
    return Rejected("not a split GLU of the gate product by the up product");
  }
  if (gate->op != GGML_OP_MUL_MAT || gate == up || gate->src[1] != up->src[1] || !IsF32(gate) ||
      gate->op_params[1] != GGML_HINT_NONE) {
    return Rejected("gate and up products of one input");
  }
  if (auto checked = CheckMulMatInto(up, glu); !checked) {
    return checked;
  }
  // Upstream's gate (ggml_cuda_should_fuse_mul_mat, ggml-cuda.cu:1741-1748)
  // and the launcher's assert (mmvf.cu:673): the kernel reads both weights
  // with the up weights' strides.
  const ggml_tensor* weights = up->src[0];
  const ggml_tensor* gate_weights = gate->src[0];
  if (!Bound(gate_weights) || gate_weights->type != weights->type ||
      !ggml_are_same_shape(gate_weights, weights) || !ggml_are_same_stride(gate_weights, weights) ||
      !ggml_are_same_shape(gate, up)) {
    return Rejected("gate weights of the up weights' type, shape and strides");
  }
  if (!Extent(gate_weights) ||
      !AlignedEverywhere(gate_weights, 2 * ggml_type_size(weights->type))) {
    return Rejected("gate weights at misaligned addresses or strides");
  }
  if (glu->ne[1] != 1 || !Packed(glu)) {
    return Rejected("fused MMVF writes one packed column");
  }
  if (!Current(gate_weights) || !Disjoint(glu, gate_weights, /*in_place=*/false)) {
    return Rejected("stale gate weights, or an output overlapping them");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckMulMatVecGlu(const ggml_tensor* gate, const ggml_tensor* up,
                                                     const ggml_tensor* glu) {
  return CheckMulMatVecSplitGlu(gate, up, glu, GGML_GLU_OP_SWIGLU);
}
bool MulMatVecGeGluPrecisionFits(const ggml_tensor* gate, const ggml_tensor* up) {
  if (gate == nullptr || up == nullptr || gate->src[0] == nullptr || up->src[0] == nullptr) {
    return false;
  }
  // MMVF interprets the GLU enum in dst->op_params[0] as a non-default
  // precision, selecting F32 accumulation. Ordinary F16 MMVF defaults to
  // half accumulation: the fused path must not silently change that contract.
  const auto fits = [](const ggml_tensor* node) {
    return node->op_params[0] == GGML_PREC_F32 ||
           (node->src[0]->type != GGML_TYPE_F16 && node->op_params[0] == GGML_PREC_DEFAULT);
  };
  return fits(gate) && fits(up);
}

std::expected<void, KernelFailure> CheckMulMatVecGeGlu(const ggml_tensor* gate,
                                                       const ggml_tensor* up,
                                                       const ggml_tensor* glu) {
  if (!MulMatVecGeGluPrecisionFits(gate, up)) {
    return Rejected("GeGLU MMVF fusion requires product accumulation compatible with F32");
  }
  return CheckMulMatVecSplitGlu(gate, up, glu, GGML_GLU_OP_GEGLU);
}

std::expected<void, KernelFailure> CheckConvert(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_CPY || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound cpy node");
  }
  // ggml_cpy's result is a view of its destination, src[1]; the launcher
  // writes src[1] (ggml_cuda_cpy(ctx, src[0], src[1])).
  const ggml_tensor* src = node->src[0];
  const ggml_tensor* dst = node->src[1];
  const bool widen = src->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32;
  const bool narrow = src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16;
  if (!widen && !narrow) {
    return Rejected("this implementation converts F32 to F16 or F16 to F32");
  }
  if (AnyEmpty({node, src, dst}) || !AllSane({node, src, dst})) {
    return Rejected("a conversion of an empty or unmeasurable tensor");
  }
  // Both contiguous and of one element count: the launcher's
  // cpy_scalar_contiguous (cpy.cu:495-499, 552-555), one thread per element
  // in 64-thread blocks whose count it asserts fits an int (cpy.cu:199-200).
  if (!ggml_are_same_shape(src, dst) || !Packed(src) || !Packed(dst) || node->data != dst->data ||
      !ggml_are_same_shape(node, dst) || std::cmp_greater(ggml_nelements(src), kInt32Max)) {
    return Rejected("a conversion between packed tensors of one shape");
  }
  if (!Aligned(src, ggml_type_size(src->type)) || !Aligned(dst, ggml_type_size(dst->type))) {
    return Rejected("conversion operands at misaligned addresses");
  }
  if (!AllCurrent({node, src, dst}) || Overlap(src, dst)) {
    return Rejected("a stale view, or a destination overlapping its source");
  }
  return {};
}

static std::expected<void, KernelFailure> CheckFlashAttnVecHead(const ggml_tensor* node,
                                                                std::int64_t head) {
  if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || !Bound(node)) {
    return Rejected("not a bound flash_attn_ext node");
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* mask = node->src[3];
  if (!Bound(q) || !Bound(k) || !Bound(v) || !Bound(mask) || node->src[4] != nullptr) {
    return Rejected("attention over bound Q, K, V and mask, without sinks");
  }
  // The instance this implementation compiles:
  // ggml_cuda_flash_attn_ext_vec_case<D, F16, F16> (fattn-vec.cuh:545-573)
  // through launch_fattn (fattn-common.cuh:975-1215): F32 Q and output, F16
  // K, V and mask (fattn-common.cuh:1000-1011).
  if (!IsF32(q) || !IsF32(node) || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
      mask->type != GGML_TYPE_F16) {
    return Rejected("vector attention takes F32 Q, F16 K, V and mask, and writes F32");
  }
  if (AnyEmpty({node, q, k, v, mask}) || !AllSane({node, q, k, v, mask})) {
    return Rejected("attention over an empty or unmeasurable tensor");
  }
  const std::int64_t kHead = head;
  constexpr std::int64_t kKqStride = 256;  // FATTN_KQ_STRIDE
  // ggml_flash_attn_ext's shapes (ggml.c:5502-5544) at head size D: Q
  // [D, rows, heads], K and V [D, cells, kv heads], one sample, a whole
  // number of query heads per KV head, the output [D, heads, rows].
  if (q->ne[0] != kHead || k->ne[0] != kHead || v->ne[0] != kHead || q->ne[3] != 1 ||
      k->ne[3] != 1 || v->ne[3] != 1 || !ggml_are_same_shape(k, v) || q->ne[2] % k->ne[2] != 0 ||
      node->ne[0] != kHead || node->ne[1] != q->ne[2] || node->ne[2] != q->ne[1] ||
      node->ne[3] != 1) {
    return Rejected("vector attention whose shapes do not follow from its operands");
  }
  if (head == 256 && q->ne[2] / k->ne[2] != 2) {
    return Rejected("D256 vector attention requires exactly two query heads per KV head");
  }
  // The attended cells padded to 256, as llama.cpp pads its cache and the
  // vector kernel requires (fattn.cu's selection); one F16 mask row per
  // query over them, contiguous.
  if (k->ne[1] % kKqStride != 0 || mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1] ||
      mask->ne[2] != 1 || mask->ne[3] != 1 || !ggml_is_contiguous(mask)) {
    return Rejected("attention over cells padded to 256 with one mask row per query");
  }
  // From 1,024 query rows launch_fattn runs the mask pre-pass
  // (flash_attn_mask_to_KV_max<2>, fattn-common.cuh:666-706), which reads
  // two mask rows per tile with no bound: an odd row count needs one more.
  if (q->ne[1] >= 1024 && mask->ne[1] < q->ne[1] + (q->ne[1] % 2)) {
    return Rejected("attention from 1,024 odd query rows without the mask row the pre-pass reads");
  }
  // Upstream's parameters: a finite positive scale, no ALiBi, no soft cap;
  // the precision llama.cpp sets (no CUDA kernel reads it).
  const float scale = ParamF32(node, 0);
  if (!(scale > 0.0f) || !(scale < std::numeric_limits<float>::infinity()) ||
      ParamF32(node, 1) != 0.0f || ParamF32(node, 2) != 0.0f ||
      node->op_params[3] != GGML_PREC_F32) {
    return Rejected("attention with a positive scale, no ALiBi or soft cap, F32 precision");
  }
  // Element-contiguous rows, 16-byte aligned bases and row strides (the
  // kernel's vector loads), and 32-bit indexing.
  if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) ||
      v->nb[0] != sizeof(ggml_fp16_t) || !Packed(node) || !AlignedEverywhere(q, 16) ||
      !AlignedEverywhere(k, 16) || !AlignedEverywhere(v, 16) || !AlignedEverywhere(mask, 16) ||
      !Aligned(node, 16) || Span(q) > kInt32Max / sizeof(float) || Span(k) > kInt32Max ||
      Span(v) > kInt32Max || Span(mask) > kInt32Max || Span(node) > kInt32Max ||
      mask->nb[1] >
          kInt32Max / static_cast<std::uint64_t>(std::max<std::int64_t>(1, q->ne[1] - 1)) ||
      k->nb[2] > kInt32Max / static_cast<std::uint64_t>(std::max<std::int64_t>(1, k->ne[2] - 1)) ||
      v->nb[2] > kInt32Max / static_cast<std::uint64_t>(std::max<std::int64_t>(1, v->ne[2] - 1))) {
    return Rejected("attention operands beyond the kernel's alignment or 32-bit indexing");
  }
  if (!AllCurrent({node, q, k, v, mask}) || Overlap(node, q) || Overlap(node, k) ||
      Overlap(node, v) || Overlap(node, mask)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckFlashAttnVec(const ggml_tensor* node) {
  return CheckFlashAttnVecHead(node, 64);
}
std::expected<void, KernelFailure> CheckFlashAttnVec256(const ggml_tensor* node) {
  return CheckFlashAttnVecHead(node, 256);
}

}  // namespace jitllm::kernels::ggml
