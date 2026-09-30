// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/validate_ext.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {
namespace {

using detail::Aligned;
using detail::AlignedEverywhere;
using detail::AllCurrent;
using detail::AllSane;
using detail::AnyEmpty;
using detail::Bound;
using detail::Current;
using detail::Disjoint;
using detail::ElementStrides;
using detail::Fits32;
using detail::IsF32;
using detail::kInt32Max;
using detail::Overlap;
using detail::Packed;
using detail::ParamF32;
using detail::Product;
using detail::Rejected;
using detail::Span;

constexpr std::array<ggml_type, 10> kQuantizedWeightTypes = {
    GGML_TYPE_Q8_0,    GGML_TYPE_Q4_K,  GGML_TYPE_Q5_K,  GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS,
};

// MarkRowPaddingReadable's bit: above every GGML_TENSOR_FLAG_* (ggml.h).
constexpr std::int32_t kRowPaddingReadable = std::int32_t{1} << 30;

// MATRIX_ROW_PADDING (common.cuh:186): the quantized products read each
// weight row in steps of this many elements.
constexpr std::int64_t kRowPadding = 512;
// QK_K: a k-quant super-block, the step of get_rows' dequantizing kernels.
constexpr std::int64_t kSuperBlock = 256;

// Whole-element strides whose element counts fit an int, as launchers pass
// them.
bool StridesFitInt(const ggml_tensor* tensor) {
  const std::uint64_t size = ggml_type_size(tensor->type);
  return std::ranges::all_of(tensor->nb, [size](std::size_t stride) {
    return stride % size == 0 && stride / size <= kInt32Max;
  });
}

bool AllPacked(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::all_of(tensors, Packed);
}

// What the quantized products need of their weights, activations and
// output, apart from the operation's shape: rows of whole blocks read in
// 512-element steps at a 16-byte aligned base, F32 activations and output
// with contiguous rows, and element counts and strides the launchers pass as
// 32-bit (mmq.cuh:1396-1475; mmvq.cu:1404-1535).
std::expected<void, KernelFailure> CheckQuantizedOperands(const ggml_tensor* weights,
                                                          const ggml_tensor* input,
                                                          const ggml_tensor* out) {
  if (!Bound(weights) || !Bound(input) || !Bound(out)) {
    return Rejected("a quantized product over unbound operands");
  }
  if (!IsQuantizedWeightType(weights->type) || !IsF32(input) || !IsF32(out)) {
    return Rejected("a quantized product needs weights of a compiled type and F32 activations");
  }
  if (AnyEmpty({weights, input, out}) || !AllSane({weights, input, out})) {
    return Rejected("a quantized product on an empty or unmeasurable tensor");
  }
  if (weights->ne[0] % kRowPadding != 0 &&
      (!RowPaddingReadable(weights) || weights->ne[0] % ggml_blck_size(weights->type) != 0)) {
    return Rejected(
        "quantized weight rows must be whole 512-element steps, or whole blocks whose padding "
        "the binder vouches for");
  }
  if (weights->nb[0] != ggml_type_size(weights->type) ||
      weights->nb[1] < ggml_row_size(weights->type, weights->ne[0]) || !ElementStrides(weights) ||
      !Aligned(weights, 16)) {
    return Rejected("quantized weights need rows of whole blocks at a 16-byte aligned base");
  }
  if (input->nb[0] != sizeof(float) || out->nb[0] != sizeof(float) || !ElementStrides(input) ||
      !ElementStrides(out) || !Aligned(input, sizeof(float)) || !Aligned(out, sizeof(float))) {
    return Rejected("a quantized product needs contiguous, aligned F32 rows");
  }
  if (!StridesFitInt(weights) || !StridesFitInt(input) || !StridesFitInt(out) ||
      Span(input) > kInt32Max || Span(out) > kInt32Max) {
    return Rejected("a quantized product beyond the kernels' 32-bit indexing");
  }
  for (const ggml_tensor* tensor : {weights, input, out}) {
    for (const std::int64_t extent : tensor->ne) {
      if (std::cmp_greater(extent, kInt32Max)) {
        return Rejected("a quantized product beyond the kernels' 32-bit extents");
      }
    }
  }
  if (!AllCurrent({weights, input, out}) || !Disjoint(out, weights, /*in_place=*/false) ||
      !Disjoint(out, input, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::int32_t Hint(const ggml_tensor* node) {
  std::int32_t hint = 0;
  std::memcpy(&hint, &node->op_params[1], sizeof(hint));
  return hint;
}

// A packed F32 tensor of the node's shape, in place or not: the elementwise
// operations' common rule. `max_elements` bounds the element count the
// kernel counts in.
std::expected<void, KernelFailure> CheckElementwise(const ggml_tensor* node, ggml_op op,
                                                    std::uint64_t max_elements) {
  if (node == nullptr || node->op != op || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound node of the operation");
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(node) || !IsF32(x)) {
    return Rejected("this implementation is F32 only");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x})) {
    return Rejected("an elementwise operation on an empty or unmeasurable tensor");
  }
  if (!ggml_are_same_shape(node, x) || !AllPacked({node, x})) {
    return Rejected("an elementwise operation over packed operands of one shape");
  }
  if (std::cmp_greater(ggml_nelements(node), max_elements)) {
    return Rejected("more elements than the kernel counts");
  }
  if (!Aligned(node, sizeof(float)) || !Aligned(x, sizeof(float))) {
    return Rejected("F32 operands at misaligned addresses");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping the input other than in place");
  }
  return {};
}

// The kernels that launch one 256-thread block per 256 elements (clamp.cu,
// unary.cu) count elements in an int, and round the count up to whole
// blocks in an int too.
constexpr std::uint64_t kIntElements = kInt32Max - 255;

}  // namespace

std::span<const ggml_type> QuantizedWeightTypes() { return kQuantizedWeightTypes; }

bool IsQuantizedWeightType(ggml_type type) {
  return std::ranges::find(kQuantizedWeightTypes, type) != kQuantizedWeightTypes.end();
}

void MarkRowPaddingReadable(ggml_tensor* weights) {
  if (weights != nullptr) {
    weights->flags |= kRowPaddingReadable;
  }
}

bool RowPaddingReadable(const ggml_tensor* weights) {
  return weights != nullptr && (weights->flags & kRowPaddingReadable) != 0;
}

std::expected<void, KernelFailure> CheckMulMatQ(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr ||
      node->src[1] == nullptr) {
    return Rejected("not a mul_mat node");
  }
  if (Hint(node) != GGML_HINT_NONE) {
    return Rejected("a mul_mat node with a routing hint");
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  if (auto checked = CheckQuantizedOperands(weights, input, node); !checked) {
    return checked;
  }
  // ggml_can_mul_mat: activations' channels and samples are whole multiples
  // of the weights'.
  if (weights->ne[0] != input->ne[0] || input->ne[2] % weights->ne[2] != 0 ||
      input->ne[3] % weights->ne[3] != 0 || node->ne[0] != weights->ne[1] ||
      node->ne[1] != input->ne[1] || node->ne[2] != input->ne[2] || node->ne[3] != input->ne[3]) {
    return Rejected("mul_mat whose shape does not follow from its operands");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMulMatIdQ(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr ||
      node->src[1] == nullptr || !Bound(node->src[2])) {
    return Rejected("not a bound mul_mat_id node");
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  const ggml_tensor* ids = node->src[2];
  if (auto checked = CheckQuantizedOperands(weights, input, node); !checked) {
    return checked;
  }
  if (ids->type != GGML_TYPE_I32 || ggml_is_empty(ids) || !AllSane({ids}) ||
      ids->nb[0] != sizeof(std::int32_t) || !ElementStrides(ids) || !StridesFitInt(ids) ||
      !Aligned(ids, sizeof(std::int32_t))) {
    return Rejected("mul_mat_id takes I32 ids with contiguous rows");
  }
  // ggml_mul_mat_id's shapes (ggml.c:3397-3435), one sample, and what the
  // launchers assert of it (mmq.cu:207-212, mmvq.cu:1420-1423).
  const std::int64_t used = ids->ne[0];
  const std::int64_t tokens = ids->ne[1];
  if (weights->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || input->ne[3] != 1 ||
      node->ne[3] != 1 || weights->ne[0] != input->ne[0] ||
      (input->ne[1] != used && input->ne[1] != 1) || input->ne[2] != tokens ||
      node->ne[0] != weights->ne[1] || node->ne[1] != used || node->ne[2] != tokens ||
      used > weights->ne[2]) {
    return Rejected("mul_mat_id whose shape does not follow from its operands");
  }
  if (input->nb[2] % input->nb[1] != 0 || node->nb[2] % node->nb[1] != 0) {
    return Rejected("mul_mat_id needs token strides that are whole multiples of the row strides");
  }
  if (Overlap(node, ids)) {
    return Rejected("an output overlapping the ids");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMulMatIdQPair(const ggml_tensor* first,
                                                      const ggml_tensor* second) {
  if (auto checked = CheckMulMatIdQ(first); !checked) {
    return checked;
  }
  if (auto checked = CheckMulMatIdQ(second); !checked) {
    return checked;
  }
  const ggml_tensor* a = first->src[0];
  const ggml_tensor* b = second->src[0];
  if (a->type != b->type || a->type == GGML_TYPE_MXFP4 || a->type == GGML_TYPE_NVFP4 ||
      first->src[1] != second->src[1] || first->src[2] != second->src[2] ||
      !std::ranges::equal(a->ne, b->ne) || !std::ranges::equal(first->ne, second->ne)) {
    return Rejected("paired expert products need one non-FP4 type, shapes, activation and ids");
  }
  if (Overlap(first, second) || Overlap(first, b) || Overlap(second, a)) {
    return Rejected("paired expert outputs overlap each other or the other weights");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMulMatHadamard(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || !Bound(node) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound mul_mat node");
  }
  if (Hint(node) != GGML_HINT_SRC0_IS_HADAMARD) {
    return Rejected("a mul_mat node without the Hadamard hint");
  }
  const ggml_tensor* rotation = node->src[0];
  const ggml_tensor* x = node->src[1];
  const std::int64_t n = x->ne[0];
  if (!IsF32(x) || !IsF32(node)) {
    return Rejected("the Hadamard transform is F32");
  }
  if (n != 64 && n != 128 && n != 256 && n != 512) {
    return Rejected("the Hadamard transform takes rows of 64, 128, 256 or 512 elements");
  }
  if (rotation->ne[0] != n || rotation->ne[1] != n || rotation->ne[2] != 1 ||
      rotation->ne[3] != 1 || !ggml_are_same_shape(node, x)) {
    return Rejected("a Hadamard product of a square rotation, the activations' shape");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x}) || !AllPacked({node, x})) {
    return Rejected("the Hadamard transform over packed operands");
  }
  // Four rows per block (fwht.cu:73-78).
  if (ggml_nrows(x) / 4 >= static_cast<std::int64_t>(kInt32Max) || !Aligned(node, sizeof(float)) ||
      !Aligned(x, sizeof(float))) {
    return Rejected("the Hadamard transform beyond its grid, or misaligned");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping the input other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckUnary(const ggml_tensor* node) {
  if (node == nullptr) {
    return Rejected("not a node");
  }
  if (node->op == GGML_OP_UNARY) {
    switch (ggml_get_unary_op(node)) {
      case GGML_UNARY_OP_ABS:
      case GGML_UNARY_OP_SGN:
      case GGML_UNARY_OP_NEG:
      case GGML_UNARY_OP_SILU:
      case GGML_UNARY_OP_TANH:
      case GGML_UNARY_OP_RELU:
      case GGML_UNARY_OP_SIGMOID:
      case GGML_UNARY_OP_EXP:
      case GGML_UNARY_OP_SOFTPLUS:
        break;
      default:
        return Rejected("a unary function this implementation does not launch");
    }
  } else if (node->op != GGML_OP_SQRT) {
    return Rejected("not a unary or sqrt node");
  }
  return CheckElementwise(node, node->op, kIntElements);
}

std::expected<void, KernelFailure> CheckScale(const ggml_tensor* node) {
  // A grid-stride loop over 64-bit indices (scale.cu).
  return CheckElementwise(node, GGML_OP_SCALE, std::numeric_limits<std::int64_t>::max());
}

std::expected<void, KernelFailure> CheckClamp(const ggml_tensor* node) {
  if (auto checked = CheckElementwise(node, GGML_OP_CLAMP, kIntElements); !checked) {
    return checked;
  }
  if (!(ParamF32(node, 0) <= ParamF32(node, 1))) {
    return Rejected("clamp's lower bound is above its upper bound");
  }
  return {};
}

std::expected<void, KernelFailure> CheckFill(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_FILL || !Bound(node)) {
    return Rejected("not a bound fill node");
  }
  if (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_F16) {
    return Rejected("fill writes F32 or F16");
  }
  if (ggml_is_empty(node) || !AllSane({node}) || !Packed(node) ||
      !Aligned(node, ggml_type_size(node->type))) {
    return Rejected("fill writes a packed, aligned, non-empty tensor");
  }
  // One 256-thread block per 256 elements (fill.cu:21-36).
  if (ggml_nelements(node) / 256 >= static_cast<std::int64_t>(kInt32Max)) {
    return Rejected("fill beyond its grid");
  }
  if (!AllCurrent({node})) {
    return Rejected("a stale view");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRepeat(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_REPEAT || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound repeat node");
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(node) || !IsF32(x)) {
    return Rejected("this implementation is F32 only");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x}) || !AllPacked({node, x})) {
    return Rejected("repeat over packed operands");
  }
  // The broadcast launcher's limits, as for add (validate.h CheckBinary).
  if (!ggml_can_repeat(x, node) || !Fits32(node) || !Fits32(x) ||
      std::cmp_greater(ggml_nelements(node), std::numeric_limits<std::uint32_t>::max() - 127)) {
    return Rejected("a repeat that does not tile, or exceeds 32-bit extents");
  }
  if (!Aligned(node, sizeof(float)) || !Aligned(x, sizeof(float))) {
    return Rejected("F32 operands at misaligned addresses");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping the input other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckConcat(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_CONCAT || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound concat node");
  }
  const ggml_tensor* a = node->src[0];
  const ggml_tensor* b = node->src[1];
  const std::int32_t dim = node->op_params[0];
  if (a->type != node->type || b->type != node->type || ggml_is_quantized(node->type) ||
      ggml_blck_size(node->type) != 1 ||
      (ggml_type_size(node->type) != 2 && ggml_type_size(node->type) != 4)) {
    return Rejected("concat of one unblocked 2- or 4-byte type");
  }
  if (dim < 0 || dim > 3) {
    return Rejected("concat along dimension 0 to 3");
  }
  if (AnyEmpty({node, a, b}) || !AllSane({node, a, b})) {
    return Rejected("concat of empty or unmeasurable tensors");
  }
  // ggml_concat's shape (ggml.c:2627-2656).
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    const std::int64_t want = i == dim ? a->ne[i] + b->ne[i] : a->ne[i];
    if ((i != dim && b->ne[i] != a->ne[i]) || node->ne[i] != want) {
      return Rejected("concat whose shape does not follow from its operands");
    }
  }
  // The contiguous kernels write the output densely, whatever its strides,
  // and count blocks in an int; the others launch a block per output row,
  // channel and sample (concat.cu:94-196). concat_cuda takes the contiguous
  // kernel (a one-dimensional grid over each sample's plane) when both
  // operands are contiguous in their first three dimensions, and two copies
  // along dimension 3 when both are contiguous (concat.cu:142-162), so only
  // the per-row kernel meets the grid's row and channel limits (RE-038).
  const std::uint64_t size = ggml_type_size(node->type);
  if (!Packed(node) || !ElementStrides(a) || !ElementStrides(b) || a->nb[0] != size ||
      b->nb[0] != size) {
    return Rejected("concat into a packed output from rows of contiguous elements");
  }
  const bool dense = dim != 3 ? ggml_is_contiguous_to_3(a) && ggml_is_contiguous_to_3(b)
                              : ggml_is_contiguous(a) && ggml_is_contiguous(b);
  const auto plane = Product({node->ne[0], node->ne[1], node->ne[2]});
  if (!plane || *plane / 256 >= kInt32Max ||
      (!dense &&
       (std::cmp_greater(node->ne[1], kInt32Max) || node->ne[2] > 65535 || node->ne[3] > 65535))) {
    return Rejected("concat beyond the kernels' grid");
  }
  if (!AlignedEverywhere(a, size) || !AlignedEverywhere(b, size) || !Aligned(node, size)) {
    return Rejected("concat operands at misaligned addresses");
  }
  if (!AllCurrent({node, a, b}) || !Disjoint(node, a, /*in_place=*/false) ||
      !Disjoint(node, b, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSumRows(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_SUM_ROWS || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound sum_rows node");
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(node) || !IsF32(x)) {
    return Rejected("sum_rows is F32");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x}) || !AllPacked({node, x})) {
    return Rejected("sum_rows over packed operands");
  }
  if (node->ne[0] != 1 || node->ne[1] != x->ne[1] || node->ne[2] != x->ne[2] ||
      node->ne[3] != x->ne[3]) {
    return Rejected("sum_rows whose shape does not follow from its input");
  }
  // One block per row, rows and columns counted in an int.
  if (ggml_nrows(x) > static_cast<std::int64_t>(kInt32Max) || Span(x) > kInt32Max ||
      !Aligned(node, sizeof(float)) || !Aligned(x, sizeof(float))) {
    return Rejected("sum_rows beyond its grid or 32-bit indexing, or misaligned");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping the input");
  }
  return {};
}

std::uint64_t ArgsortSharedBytes(const ggml_tensor* node) {
  std::uint64_t padded = 1;
  while (std::cmp_less(padded, node->src[0]->ne[0])) {
    padded *= 2;
  }
  return padded * sizeof(std::int32_t);
}

std::expected<void, KernelFailure> CheckArgsort(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_ARGSORT || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound argsort node");
  }
  const ggml_tensor* x = node->src[0];
  const std::int32_t order = node->op_params[0];
  if (!IsF32(x) || node->type != GGML_TYPE_I32 ||
      (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC)) {
    return Rejected("argsort of F32 rows into I32 indices, ascending or descending");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x}) || !AllPacked({node, x}) ||
      !ggml_are_same_shape(node, x)) {
    return Rejected("argsort over packed operands of one shape");
  }
  // The bitonic kernel: one block per row, a thread per padded column.
  if (x->ne[0] > 1024 || ggml_nrows(x) > static_cast<std::int64_t>(kInt32Max) ||
      Span(x) > kInt32Max) {
    return Rejected("argsort of rows longer than 1,024 takes CUB's sort upstream, not built");
  }
  if (!Aligned(node, sizeof(std::int32_t)) || !Aligned(x, sizeof(float))) {
    return Rejected("argsort operands at misaligned addresses");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping the input");
  }
  return {};
}

std::expected<void, KernelFailure> CheckTopK(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_TOP_K || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound top_k node");
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(x) || node->type != GGML_TYPE_I32) {
    return Rejected("top_k of F32 rows into I32 indices");
  }
  if (AnyEmpty({node, x}) || !AllSane({node, x}) || !AllPacked({node, x})) {
    return Rejected("top_k over packed operands");
  }
  if (node->ne[0] > x->ne[0] || node->ne[1] != x->ne[1] || node->ne[2] != x->ne[2] ||
      node->ne[3] != x->ne[3]) {
    return Rejected("top_k whose shape does not follow from its input");
  }
  // Rows, columns and k are passed as int (top-k.cu:17-39).
  if (std::cmp_greater(x->ne[0], kInt32Max) || std::cmp_greater(ggml_nrows(x), kInt32Max) ||
      !Aligned(node, sizeof(std::int32_t)) || !Aligned(x, sizeof(float))) {
    return Rejected("top_k beyond its 32-bit counts, or misaligned");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping the input");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSwiGluClamp(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_GLU || !Bound(node) || !Bound(node->src[0])) {
    return Rejected("not a bound GLU node");
  }
  if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU_CLAMP) {
    return Rejected("not a clamped SwiGLU");
  }
  const ggml_tensor* gate = node->src[0];
  const ggml_tensor* up = node->src[1];
  if (!IsF32(node) || !IsF32(gate) || (up != nullptr && !IsF32(up))) {
    return Rejected("this implementation is F32 only");
  }
  if (up != nullptr && !Bound(up)) {
    return Rejected("an unbound up operand");
  }
  const ggml_tensor* second = up != nullptr ? up : gate;
  if (AnyEmpty({node, gate, second}) || !AllSane({node, gate, second})) {
    return Rejected("a GLU over empty or unmeasurable tensors");
  }
  // The kernel indexes each operand as rows of `n` elements at its row
  // stride (unary.cu:432-449): rows contiguous, and every higher dimension
  // packed over the rows (ggml_is_contiguous_1).
  const std::int64_t n = up != nullptr ? gate->ne[0] : gate->ne[0] / 2;
  if ((up == nullptr && gate->ne[0] % 2 != 0) || node->ne[0] != n ||
      ggml_nrows(node) != ggml_nrows(gate) || !Packed(node) || gate->nb[0] != sizeof(float) ||
      !ggml_is_contiguous_1(gate) ||
      (up != nullptr && (up->nb[0] != sizeof(float) || !ggml_is_contiguous_1(up) ||
                         !ggml_are_same_shape(up, gate)))) {
    return Rejected("a clamped SwiGLU over contiguous rows into a packed output");
  }
  if (!ggml_are_same_shape(node, gate) && up != nullptr) {
    return Rejected("a split GLU's output has its operands' shape");
  }
  if (!ElementStrides(gate) || !ElementStrides(second) || !Aligned(node, sizeof(float)) ||
      !Aligned(gate, sizeof(float)) || !Aligned(second, sizeof(float))) {
    return Rejected("GLU operands at misaligned addresses or strides");
  }
  if (!AllCurrent({node, gate, second}) || !Disjoint(node, gate, /*in_place=*/false) ||
      !Disjoint(node, second, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRopeExt(const ggml_tensor* node) {
  if (node == nullptr || (node->op != GGML_OP_ROPE && node->op != GGML_OP_ROPE_BACK) ||
      !Bound(node) || !Bound(node->src[0]) || !Bound(node->src[1])) {
    return Rejected("not a bound rope node");
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* positions = node->src[1];
  if (!IsF32(x) || !IsF32(node) || positions->type != GGML_TYPE_I32) {
    return Rejected("RoPE over F32 with I32 positions");
  }
  if (node->src[2] != nullptr) {
    return Rejected("RoPE with frequency factors is not implemented");
  }
  // op_params (ggml_rope_impl, ggml.c:4288-4318).
  const std::int32_t n_dims = node->op_params[1];
  const std::int32_t mode = node->op_params[2];
  const std::int32_t offset = node->op_params[15];
  const bool multi = mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE;
  if (mode != 0 && mode != GGML_ROPE_TYPE_NEOX && !multi) {
    return Rejected("normal, NEOX, MROPE or IMROPE rotation (not vision)");
  }
  if (AnyEmpty({node, x, positions}) || !AllSane({node, x, positions})) {
    return Rejected("RoPE on an empty or unmeasurable tensor");
  }
  // The launcher asserts an even head (rope.cu:383); pairs are rotated
  // within [offset, offset + n_dims).
  if (!ggml_are_same_shape(node, x) || x->ne[0] % 2 != 0 || n_dims <= 0 || n_dims % 2 != 0 ||
      offset < 0 || offset % 2 != 0 || n_dims > x->ne[0] - offset) {
    return Rejected("RoPE over even heads, rotating an even part of each");
  }
  if (multi) {
    std::array<std::int32_t, 4> sections{};
    std::memcpy(sections.data(), &node->op_params[11], sizeof(sections));
    std::int64_t sum = 0;
    for (const std::int32_t section : sections) {
      if (section < 0) {
        return Rejected("negative rotation sections");
      }
      sum += section;
    }
    // The launcher asserts a non-empty section (rope.cu:609); the kernel
    // takes each pair's section modulo their sum.
    if (sections[0] <= 0 && sections[1] <= 0 && sections[2] <= 0) {
      return Rejected("multi-section RoPE with no position section");
    }
    if (sum <= 0) {
      return Rejected("multi-section RoPE with no sections");
    }
  }
  // One position per token, or four for the multi-section modes, read
  // densely (ggml.c:4288-4297).
  const std::int64_t per_token = multi ? 4 : 1;
  if (!ggml_is_vector(positions) || positions->ne[0] != x->ne[2] * per_token ||
      positions->nb[0] != sizeof(std::int32_t) || !Aligned(positions, sizeof(std::int32_t))) {
    return Rejected("RoPE takes packed positions for each token");
  }
  // Extents and element strides pass as int; a block per row and a
  // 256-thread column per 512 elements of the head; pairs are stored as
  // one 8-byte value (rope.cu:44-116, 380-390).
  if (x->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) || !StridesFitInt(x) ||
      !StridesFitInt(node) || Span(x) > kInt32Max || Span(node) > kInt32Max ||
      ggml_nrows(x) > static_cast<std::int64_t>(kInt32Max) || (x->ne[0] + 511) / 512 > 65535 ||
      !Aligned(x, sizeof(float)) || !AlignedEverywhere(node, 2 * sizeof(float))) {
    return Rejected("RoPE operands beyond the kernel's 32-bit indexing or grid, or misaligned");
  }
  if (!AllCurrent({node, x, positions}) || !Disjoint(node, x, /*in_place=*/true) ||
      Overlap(node, positions)) {
    return Rejected("a stale view, or an output overlapping an operand other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckGetRowsExt(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_GET_ROWS || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound get_rows node");
  }
  const ggml_tensor* rows = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const bool quantized = IsQuantizedWeightType(rows->type);
  const bool integers = rows->type == GGML_TYPE_I32 && node->type == GGML_TYPE_I32;
  if ((!quantized || !IsF32(node)) && !integers) {
    return Rejected("get_rows of quantized rows into F32, or of I32 rows into I32");
  }
  if (ids->type != GGML_TYPE_I32) {
    return Rejected("get_rows takes I32 ids");
  }
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
  if (quantized && rows->ne[0] % kSuperBlock != 0) {
    return Rejected("quantized rows must be whole 256-element super-blocks");
  }
  if (rows->nb[0] != ggml_type_size(rows->type) || ids->nb[0] != sizeof(std::int32_t) ||
      node->nb[0] != ggml_type_size(node->type) || !ElementStrides(rows) || !ElementStrides(ids) ||
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
  if (!Aligned(rows, quantized ? 2 : sizeof(std::int32_t)) || !Aligned(ids, sizeof(std::int32_t)) ||
      !Aligned(node, sizeof(float))) {
    return Rejected("get_rows operands at misaligned addresses");
  }
  if (!AllCurrent({node, rows, ids}) || !Disjoint(node, rows, /*in_place=*/false) ||
      !Disjoint(node, ids, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSetRowsExt(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_SET_ROWS || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound set_rows node");
  }
  const ggml_tensor* values = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const bool types = (values->type == GGML_TYPE_F32 &&
                      (node->type == GGML_TYPE_F32 || node->type == GGML_TYPE_F16)) ||
                     (values->type == GGML_TYPE_F16 && node->type == GGML_TYPE_F16);
  if (!types || (ids->type != GGML_TYPE_I32 && ids->type != GGML_TYPE_I64)) {
    return Rejected("set_rows of F32 into F32 or F16, or F16 into F16, at I32 or I64 indices");
  }
  if (AnyEmpty({node, values, ids}) || !AllSane({node, values, ids})) {
    return Rejected("set_rows on an empty or unmeasurable tensor");
  }
  // ggml_set_rows's shape (ggml.c): a row of values per index, the ids'
  // channels and samples broadcast over the values'.
  if (values->ne[0] != node->ne[0] || values->ne[1] != ids->ne[0] || ids->ne[3] != 1 ||
      values->ne[2] % ids->ne[1] != 0 || values->ne[3] % ids->ne[2] != 0 ||
      node->ne[2] != values->ne[2] || node->ne[3] != values->ne[3]) {
    return Rejected("set_rows whose shape does not follow from its operands");
  }
  if (values->nb[0] != ggml_type_size(values->type) || node->nb[0] != ggml_type_size(node->type) ||
      ids->nb[0] != ggml_type_size(ids->type) || !ElementStrides(values) || !ElementStrides(node) ||
      !ElementStrides(ids)) {
    return Rejected("set_rows needs contiguous rows and whole-element strides");
  }
  // One thread per value, whose flat index the kernel splits with 32-bit
  // division (set-rows.cu:136-157, 188-189).
  if (std::cmp_greater(ggml_nelements(values), std::numeric_limits<std::uint32_t>::max())) {
    return Rejected("set_rows beyond the kernel's 32-bit indexing");
  }
  if (!Aligned(values, ggml_type_size(values->type)) ||
      !Aligned(node, ggml_type_size(node->type)) || !Aligned(ids, ggml_type_size(ids->type))) {
    return Rejected("set_rows operands at misaligned addresses");
  }
  if (!AllCurrent({node, values, ids}) || Overlap(node, values) || Overlap(node, ids)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSsmConv(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_SSM_CONV || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound ssm_conv node");
  }
  const ggml_tensor* window = node->src[0];
  const ggml_tensor* weights = node->src[1];
  if (!IsF32(node) || !IsF32(window) || !IsF32(weights)) {
    return Rejected("ssm_conv is F32");
  }
  if (AnyEmpty({node, window, weights}) || !AllSane({node, window, weights})) {
    return Rejected("ssm_conv on an empty or unmeasurable tensor");
  }
  const std::int64_t conv = weights->ne[0];
  const std::int64_t channels = weights->ne[1];
  const std::int64_t tokens = node->ne[1];
  // ggml_ssm_conv's shape (ggml.c:5664-5690) and the launcher's kernel sizes
  // and 128-channel blocks (ssm-conv.cu:124-158).
  if (weights->ne[2] != 1 || weights->ne[3] != 1 || window->ne[3] != 1 || node->ne[3] != 1 ||
      window->ne[1] != channels || window->ne[0] != conv - 1 + tokens || node->ne[0] != channels ||
      node->ne[2] != window->ne[2]) {
    return Rejected("ssm_conv whose shape does not follow from its operands");
  }
  if (conv != 3 && conv != 4 && conv != 5 && conv != 9 && conv != 15) {
    return Rejected("ssm_conv takes kernels of 3, 4, 5, 9 or 15");
  }
  if (channels % 128 != 0) {
    return Rejected("ssm_conv takes channels in blocks of 128");
  }
  // Past 32 tokens the launcher runs ssm_conv_long_token_f32, whose blocks
  // each load conv - 1 + 32 columns of every row whatever tokens are left
  // (ssm-conv.cu:81-101): the last block reads up to 31 floats past the
  // window unless the tokens are whole 32-token blocks (RE-032).
  if (tokens > 32 && tokens % 32 != 0) {
    return Rejected(
        "ssm_conv past 32 tokens takes whole 32-token blocks (its last block loads past the "
        "window otherwise, RE-032)");
  }
  // Packed windows, contiguous weight rows, and every stride and index an
  // int (ssm-conv.cu:5-99, 181-184).
  if (!Packed(window) || weights->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) ||
      !ElementStrides(weights) || !ElementStrides(node) || !StridesFitInt(weights) ||
      Span(window) > kInt32Max || Span(weights) > kInt32Max || Span(node) > kInt32Max ||
      std::cmp_greater(window->ne[2], kInt32Max) || channels / 128 > 65535 ||
      (tokens + 31) / 32 > 65535) {
    return Rejected("ssm_conv operands beyond the kernel's 32-bit indexing or grid");
  }
  // Bytes pass as int too.
  for (const ggml_tensor* tensor : {window, weights, node}) {
    for (int i = 0; i < 3; ++i) {
      if (tensor->nb[i] > kInt32Max) {
        return Rejected("ssm_conv strides beyond an int");
      }
    }
  }
  if (!Aligned(window, sizeof(float)) || !Aligned(weights, sizeof(float)) ||
      !Aligned(node, sizeof(float))) {
    return Rejected("ssm_conv operands at misaligned addresses");
  }
  if (!AllCurrent({node, window, weights}) || !Disjoint(node, window, /*in_place=*/false) ||
      !Disjoint(node, weights, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckGatedDeltaNet(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_GATED_DELTA_NET || !Bound(node)) {
    return Rejected("not a bound gated_delta_net node");
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* g = node->src[3];
  const ggml_tensor* beta = node->src[4];
  const ggml_tensor* state = node->src[5];
  for (const ggml_tensor* tensor : {q, k, v, g, beta, state}) {
    if (!Bound(tensor) || !IsF32(tensor)) {
      return Rejected("gated_delta_net takes bound F32 operands");
    }
  }
  if (!IsF32(node) || AnyEmpty({node, q, k, v, g, beta, state}) ||
      !AllSane({node, q, k, v, g, beta, state})) {
    return Rejected("gated_delta_net on an empty or unmeasurable tensor");
  }
  const std::int64_t s = v->ne[0];
  const std::int64_t heads = v->ne[1];
  const std::int64_t tokens = v->ne[2];
  const std::int64_t seqs = v->ne[3];
  const std::int32_t snapshots = node->op_params[0];
  if (s != 16 && s != 32 && s != 64 && s != 128) {
    return Rejected("gated_delta_net takes heads of 16, 32, 64 or 128");
  }
  // ggml_gated_delta_net's shapes (ggml.c:6371-6420) and the launcher's
  // assertions (gated_delta_net.cu:238-273): query and key heads divide the
  // value heads, query and key sequences divide the value sequences.
  if (!ggml_are_same_shape(q, k) || q->ne[0] != s || q->ne[2] != tokens || heads % q->ne[1] != 0 ||
      seqs % q->ne[3] != 0 || (g->ne[0] != 1 && g->ne[0] != s) || g->ne[1] != heads ||
      g->ne[2] != tokens || g->ne[3] != seqs || beta->ne[0] != 1 || beta->ne[1] != heads ||
      beta->ne[2] != tokens || beta->ne[3] != seqs || state->ne[0] != s || state->ne[1] != s ||
      state->ne[2] != heads || state->ne[3] != seqs || snapshots < 1) {
    return Rejected("gated_delta_net whose shapes do not follow ggml_gated_delta_net's");
  }
  const auto rows = Product({tokens, seqs});
  const auto state_rows = Product({snapshots, s, seqs});
  if (!rows || !state_rows || node->ne[0] != s * heads ||
      std::cmp_not_equal(node->ne[1], *rows + *state_rows) || node->ne[2] != 1 ||
      node->ne[3] != 1 || !Packed(node)) {
    return Rejected("gated_delta_net writes the packed attention and state snapshots");
  }
  if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(v) || !ggml_are_same_stride(q, k) ||
      !Packed(g) || !Packed(beta) || !Packed(state) || !ElementStrides(q) || !ElementStrides(v)) {
    return Rejected("gated_delta_net needs contiguous rows and packed gate, beta and state");
  }
  // One block per value head, sequence and 4 value columns (y and z
  // bounded by 65,535); fast divisors of the query heads and the sequence
  // ratio (32-bit).
  if (seqs > 65535 || std::cmp_greater(heads, kInt32Max) ||
      std::cmp_greater(q->ne[1], std::numeric_limits<std::uint32_t>::max())) {
    return Rejected("gated_delta_net beyond its grid");
  }
  if (!Aligned(node, sizeof(float))) {
    return Rejected("gated_delta_net output at a misaligned address");
  }
  for (const ggml_tensor* tensor : {q, k, v, g, beta, state}) {
    if (!Aligned(tensor, sizeof(float)) || !Disjoint(node, tensor, /*in_place=*/false)) {
      return Rejected("gated_delta_net operands misaligned, or overlapping the output");
    }
  }
  if (!AllCurrent({node, q, k, v, g, beta, state})) {
    return Rejected("a stale view");
  }
  return {};
}

std::expected<void, KernelFailure> CheckLightningIndexer(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_LIGHTNING_INDEXER || !Bound(node)) {
    return Rejected("not a bound lightning_indexer node");
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* w = node->src[2];
  const ggml_tensor* mask = node->src[3];
  if (!Bound(q) || !Bound(k) || !Bound(w) || !Bound(mask)) {
    return Rejected("lightning_indexer over unbound operands");
  }
  if (!IsF32(node) || !IsF32(q) || !IsF32(w) || k->type != GGML_TYPE_F16 ||
      mask->type != GGML_TYPE_F16) {
    return Rejected("lightning_indexer of F32 queries and weights over F16 keys and mask");
  }
  if (AnyEmpty({node, q, k, w, mask}) || !AllSane({node, q, k, w, mask})) {
    return Rejected("lightning_indexer on an empty or unmeasurable tensor");
  }
  // ggml_lightning_indexer's shapes (ggml.c:6428-6460) and the kernels the
  // launcher has (lightning-indexer.cu:447-540).
  if (q->ne[0] != 128 || (q->ne[1] != 64 && q->ne[1] != 32) || k->ne[0] != 128 || k->ne[1] != 1 ||
      w->ne[0] != q->ne[1] || w->ne[1] != q->ne[2] || w->ne[2] != 1 || mask->ne[0] != k->ne[2] ||
      mask->ne[1] != q->ne[2] || mask->ne[2] != 1 || q->ne[3] != k->ne[3] || k->ne[3] != w->ne[3] ||
      w->ne[3] % mask->ne[3] != 0 || node->ne[0] != k->ne[2] || node->ne[1] != q->ne[2] ||
      node->ne[2] != 1 || node->ne[3] != q->ne[3]) {
    return Rejected("lightning_indexer whose shapes do not follow ggml_lightning_indexer's");
  }
  // Contiguous rows; the output neither transposed nor permuted; every
  // non-first stride of the queries and keys a multiple of 16 bytes
  // (lightning-indexer.cu:422-432, 549-560).
  if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) || w->nb[0] != sizeof(float) ||
      mask->nb[0] != sizeof(ggml_fp16_t) || node->nb[0] != sizeof(float) ||
      node->nb[0] > node->nb[1] || node->nb[1] > node->nb[2] || node->nb[2] > node->nb[3] ||
      !AlignedEverywhere(q, 16) || !AlignedEverywhere(k, 16) || !ElementStrides(w) ||
      !ElementStrides(mask) || !ElementStrides(node) || !Aligned(w, sizeof(float)) ||
      !Aligned(mask, sizeof(ggml_fp16_t)) || !Aligned(node, sizeof(float))) {
    return Rejected("lightning_indexer needs contiguous, aligned rows");
  }
  // Extents pass as int; a block per 32 cells, batch row and stream.
  if (q->ne[2] > 65535 || q->ne[3] > 65535 ||
      k->ne[2] > static_cast<std::int64_t>(kInt32Max) - 64) {
    return Rejected("lightning_indexer beyond its grid");
  }
  if (!AllCurrent({node, q, k, w, mask})) {
    return Rejected("a stale view");
  }
  for (const ggml_tensor* tensor : {q, k, w, mask}) {
    if (!Disjoint(node, tensor, /*in_place=*/false)) {
      return Rejected("an output overlapping an operand");
    }
  }
  return {};
}

namespace {

// The hyper-connections' kernels take 64-bit element strides and one
// thread per output element (dsv4-hc.cu), four streams.
constexpr std::int64_t kStreams = 4;

std::expected<void, KernelFailure> CheckHcOperands(const ggml_tensor* node, ggml_op op,
                                                   std::initializer_list<const ggml_tensor*> in) {
  if (node == nullptr || node->op != op || !Bound(node)) {
    return Rejected("not a bound hyper-connection node");
  }
  for (const ggml_tensor* tensor : in) {
    if (!Bound(tensor) || !IsF32(tensor) || ggml_is_empty(tensor) || !AllSane({tensor}) ||
        !ElementStrides(tensor) || !Aligned(tensor, sizeof(float)) || !Current(tensor) ||
        !Disjoint(node, tensor, /*in_place=*/false)) {
      return Rejected("hyper-connection operands: bound, aligned F32, apart from the output");
    }
  }
  if (!IsF32(node) || ggml_is_empty(node) || !AllSane({node}) || !ElementStrides(node) ||
      !Aligned(node, sizeof(float)) || !Current(node)) {
    return Rejected("a hyper-connection writes aligned F32");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckHcComb(const ggml_tensor* node) {
  if (node == nullptr) {
    return Rejected("not a node");
  }
  const ggml_tensor* mixes = node->src[0];
  const ggml_tensor* scale = node->src[1];
  const ggml_tensor* base = node->src[2];
  if (auto checked = CheckHcOperands(node, GGML_OP_DSV4_HC_COMB, {mixes, scale, base}); !checked) {
    return checked;
  }
  constexpr std::int64_t kMix = (2 + kStreams) * kStreams;
  if (mixes->ne[0] != kMix || mixes->ne[2] != 1 || mixes->ne[3] != 1 || scale->ne[0] < 3 ||
      base->ne[0] != kMix || node->ne[0] != kStreams || node->ne[1] != kStreams ||
      node->ne[2] != mixes->ne[1] || node->ne[3] != 1 || node->op_params[1] <= 0) {
    return Rejected("hc_comb whose shapes do not follow ggml_dsv4_hc_comb's");
  }
  if (mixes->ne[1] / 256 >= static_cast<std::int64_t>(kInt32Max)) {
    return Rejected("hc_comb beyond its grid");
  }
  return {};
}

std::expected<void, KernelFailure> CheckHcPre(const ggml_tensor* node) {
  if (node == nullptr) {
    return Rejected("not a node");
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* weights = node->src[1];
  if (auto checked = CheckHcOperands(node, GGML_OP_DSV4_HC_PRE, {x, weights}); !checked) {
    return checked;
  }
  if (x->ne[1] != kStreams || x->ne[3] != 1 || weights->ne[0] != kStreams ||
      weights->ne[1] != x->ne[2] || weights->ne[2] != 1 || weights->ne[3] != 1 ||
      node->ne[0] != x->ne[0] || node->ne[1] != x->ne[2] || node->ne[2] != 1 || node->ne[3] != 1) {
    return Rejected("hc_pre whose shapes do not follow ggml_dsv4_hc_pre's");
  }
  const auto threads = Product({x->ne[0], x->ne[2]});
  if (!threads || *threads / 256 >= kInt32Max) {
    return Rejected("hc_pre beyond its grid");
  }
  return {};
}

std::expected<void, KernelFailure> CheckHcPost(const ggml_tensor* node) {
  if (node == nullptr) {
    return Rejected("not a node");
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* residual = node->src[1];
  const ggml_tensor* post = node->src[2];
  const ggml_tensor* comb = node->src[3];
  if (auto checked = CheckHcOperands(node, GGML_OP_DSV4_HC_POST, {x, residual, post, comb});
      !checked) {
    return checked;
  }
  const std::int64_t embd = x->ne[0];
  const std::int64_t tokens = x->ne[1];
  if (x->ne[2] != 1 || x->ne[3] != 1 || residual->ne[0] != embd || residual->ne[1] != kStreams ||
      residual->ne[2] != tokens || residual->ne[3] != 1 || post->ne[0] != kStreams ||
      post->ne[1] != tokens || post->ne[2] != 1 || post->ne[3] != 1 || comb->ne[0] != kStreams ||
      comb->ne[1] != kStreams || comb->ne[2] != tokens || comb->ne[3] != 1 || node->ne[0] != embd ||
      node->ne[1] != kStreams || node->ne[2] != tokens || node->ne[3] != 1) {
    return Rejected("hc_post whose shapes do not follow ggml_dsv4_hc_post's");
  }
  const auto threads = Product({embd, kStreams, tokens});
  if (!threads || *threads / 256 >= kInt32Max) {
    return Rejected("hc_post beyond its grid");
  }
  return {};
}

std::expected<void, KernelFailure> CheckFlashAttnMma(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || !Bound(node) ||
      !Bound(node->src[0]) || !Bound(node->src[1]) || !Bound(node->src[2]) ||
      !Bound(node->src[3])) {
    return Rejected("not a bound, masked flash_attn_ext node");
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* mask = node->src[3];
  const ggml_tensor* sinks = node->src[4];
  if (!IsF32(q) || !IsF32(node) || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
      mask->type != GGML_TYPE_F16 || (sinks != nullptr && (!Bound(sinks) || !IsF32(sinks)))) {
    return Rejected("F32 Q, F16 K, V and mask, optional F32 sinks, into F32");
  }
  // op_params: scale, max_bias, logit_softcap, precision, n_kv_max
  // (ggml.c:5534-5575).
  if (ParamF32(node, 1) != 0.0f || ParamF32(node, 2) != 0.0f || node->op_params[4] < 0) {
    return Rejected("no ALiBi and no logit soft-capping");
  }
  const std::int64_t d = q->ne[0];
  if ((d != 256 && d != 512) || k->ne[0] != d || v->ne[0] != d) {
    return Rejected("head dimensions 256 or 512, the same for Q, K and V");
  }
  std::initializer_list<const ggml_tensor*> operands = {node, q, k, v, mask};
  if (AnyEmpty(operands) || !AllSane(operands) || (sinks != nullptr && !AllSane({sinks}))) {
    return Rejected("flash attention on an empty or unmeasurable tensor");
  }
  // Shapes (ggml_flash_attn_ext, ggml.c:5502-5545): K and V alike, query
  // heads a multiple of KV heads, one mask for every head.
  if (!ggml_are_same_shape(k, v) || q->ne[3] != k->ne[3] || q->ne[2] % k->ne[2] != 0 ||
      mask->ne[2] != 1 || q->ne[3] % mask->ne[3] != 0 || mask->ne[0] < k->ne[1] ||
      mask->ne[1] < q->ne[1] || node->ne[0] != d || node->ne[1] != q->ne[2] ||
      node->ne[2] != q->ne[1] || node->ne[3] != q->ne[3]) {
    return Rejected("flash attention whose shapes do not follow ggml_flash_attn_ext's");
  }
  if (sinks != nullptr && (sinks->ne[0] != q->ne[2] || ggml_nelements(sinks) != q->ne[2] ||
                           sinks->nb[0] != sizeof(float) || !Aligned(sinks, sizeof(float)))) {
    return Rejected("one F32 sink per query head");
  }
  // The kernel reads a whole group of 8 sinks from each group's first query
  // head, past the last head when the heads per KV head are not a multiple
  // of 8 (fattn-mma-f16.cuh:1402, 1889; the Q loads and output writes are
  // bounded there, the sinks are not).
  if (sinks != nullptr && (q->ne[2] / k->ne[2]) % 8 != 0) {
    return Rejected("sinks need a multiple of 8 query heads per KV head");
  }
  // The GQA-grouped kernels (fattn.cu:218-268): padded cells, a query-head
  // group of 8, and 16-byte strides everywhere.
  if (k->ne[1] % 256 != 0 || q->ne[2] / k->ne[2] <= 4) {
    return Rejected("cells in multiples of 256 and more than 4 query heads per KV head");
  }
  if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) ||
      v->nb[0] != sizeof(ggml_fp16_t) || mask->nb[0] != sizeof(ggml_fp16_t) ||
      !AlignedEverywhere(q, 16) || !AlignedEverywhere(k, 16) || !AlignedEverywhere(v, 16) ||
      !AlignedEverywhere(mask, 16) || !Packed(node) || !Aligned(node, 16)) {
    return Rejected("contiguous rows at 16-byte strides, and a packed output");
  }
  // The kernel takes extents, Q's strides and the others' first three as
  // int (fattn-mma-f16.cuh:1766-1787), and indexes the output in int.
  if (Span(node) > kInt32Max) {
    return Rejected("flash attention output beyond the kernel's 32-bit indexing");
  }
  for (const ggml_tensor* tensor : {q, k, v, mask}) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      if (std::cmp_greater(tensor->ne[i], kInt32Max) ||
          ((tensor == q || i < 3) && tensor->nb[i] > kInt32Max)) {
        return Rejected("flash attention beyond the kernel's 32-bit extents and strides");
      }
    }
  }
  if (!AllCurrent(operands) || (sinks != nullptr && !AllCurrent({sinks}))) {
    return Rejected("a stale view");
  }
  for (const ggml_tensor* tensor : {q, k, v, mask}) {
    if (!Disjoint(node, tensor, /*in_place=*/false)) {
      return Rejected("an output overlapping an operand");
    }
  }
  if (sinks != nullptr && Overlap(node, sinks)) {
    return Rejected("an output overlapping the sinks");
  }
  return {};
}

std::expected<void, KernelFailure> CheckFlashAttnMma128(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || !Bound(node) ||
      !Bound(node->src[0]) || !Bound(node->src[1]) || !Bound(node->src[2]) ||
      node->src[3] != nullptr || node->src[4] != nullptr) {
    return Rejected("not a bound flash_attn_ext node without mask or sinks");
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  if (!IsF32(q) || !IsF32(node) || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
    return Rejected("F32 Q, F16 K and V, into F32");
  }
  if (ParamF32(node, 1) != 0.0f || ParamF32(node, 2) != 0.0f || node->op_params[4] != 0) {
    return Rejected("no ALiBi, no logit soft-capping and no sparse bound");
  }
  if (q->ne[0] != 128 || k->ne[0] != 128 || v->ne[0] != 128) {
    return Rejected("head dimension 128 for Q, K and V");
  }
  std::initializer_list<const ggml_tensor*> operands = {node, q, k, v};
  if (AnyEmpty(operands) || !AllSane(operands)) {
    return Rejected("flash attention on an empty or unmeasurable tensor");
  }
  // One query head per KV head, one sample.
  if (!ggml_are_same_shape(k, v) || q->ne[2] != k->ne[2] || q->ne[3] != 1 || k->ne[3] != 1 ||
      node->ne[0] != 128 || node->ne[1] != q->ne[2] || node->ne[2] != q->ne[1] ||
      node->ne[3] != 1) {
    return Rejected("multi-head attention whose shapes do not follow ggml_flash_attn_ext's");
  }
  if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) ||
      v->nb[0] != sizeof(ggml_fp16_t) || !AlignedEverywhere(q, 16) || !AlignedEverywhere(k, 16) ||
      !AlignedEverywhere(v, 16) || !Packed(node) || !Aligned(node, 16)) {
    return Rejected("contiguous rows at 16-byte strides, and a packed output");
  }
  if (Span(node) > kInt32Max) {
    return Rejected("flash attention output beyond the kernel's 32-bit indexing");
  }
  for (const ggml_tensor* tensor : {q, k, v}) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      if (std::cmp_greater(tensor->ne[i], kInt32Max) ||
          ((tensor == q || i < 3) && tensor->nb[i] > kInt32Max)) {
        return Rejected("flash attention beyond the kernel's 32-bit extents and strides");
      }
    }
  }
  if (!AllCurrent(operands)) {
    return Rejected("a stale view");
  }
  for (const ggml_tensor* tensor : {q, k, v}) {
    if (!Disjoint(node, tensor, /*in_place=*/false)) {
      return Rejected("an output overlapping an operand");
    }
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
