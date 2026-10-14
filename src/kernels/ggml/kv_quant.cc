// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/kv_quant.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {

using detail::Aligned;
using detail::AlignedEverywhere;
using detail::AllCurrent;
using detail::AllSane;
using detail::AnyEmpty;
using detail::Bound;
using detail::Disjoint;
using detail::ElementStrides;
using detail::Extent;
using detail::IsF32;
using detail::kInt32Max;
using detail::Overlap;
using detail::Packed;
using detail::ParamF32;
using detail::Rejected;
using detail::Span;

constexpr std::int64_t kBlockValues = 32;  // QK8_0 and QK4_0
constexpr std::int64_t kKqStride = 256;    // FATTN_KQ_STRIDE

// Every stride from the second dimension on a whole number of blocks, the
// base at the blocks' 2-byte alignment (block_q8_0 and block_q4_0 begin
// with their F16 scale).
bool BlockStrides(const ggml_tensor* tensor) {
  const std::uint64_t block = ggml_type_size(tensor->type);
  if (tensor->nb[0] != block || !Aligned(tensor, 2)) {
    return false;
  }
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (tensor->nb[i] % block != 0) {
      return false;
    }
  }
  return true;
}

bool FitsInt32Bytes(const ggml_tensor* tensor) {
  const auto extent = Extent(tensor);
  return extent && *extent <= kInt32Max;
}

}  // namespace

bool KvQuantType(ggml_type type) { return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0; }

std::string_view FlashAttnVecQuantName(ggml_type k_type) {
  switch (k_type) {
    case GGML_TYPE_Q8_0:
      return kFlashAttnVecQ8Name;
    case GGML_TYPE_Q4_0:
      return kFlashAttnVecQ4Name;
    default:
      return {};
  }
}

std::expected<void, KernelFailure> CheckFlashAttnVecQuant(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || !Bound(node)) {
    return Rejected("not a bound flash_attn_ext node");
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* mask = node->src[3];
  const ggml_tensor* sinks = node->src[4];
  if (!Bound(q) || !Bound(k) || !Bound(v) || !Bound(mask)) {
    return Rejected("attention over bound Q, K, V and mask");
  }
  if (sinks != nullptr && (!Bound(sinks) || !IsF32(sinks))) {
    return Rejected("attention with bound F32 sinks");
  }
  // The instances fattn_vec_quant.cu compiles: ggml_cuda_flash_attn_ext_vec_case<D, T, T>
  // (fattn-vec.cuh) through launch_fattn (fattn-common.cuh): F32 Q and
  // output, an F16 mask.
  if (!IsF32(q) || !IsF32(node) || !KvQuantType(k->type) || v->type != k->type ||
      mask->type != GGML_TYPE_F16) {
    return Rejected(
        "quantized vector attention takes F32 Q, K and V both Q8_0 or both Q4_0, an F16 mask, "
        "and writes F32");
  }
  if (AnyEmpty({node, q, k, v, mask}) || !AllSane({node, q, k, v, mask}) ||
      (sinks != nullptr && !AllSane({sinks}))) {
    return Rejected("attention over an empty or unmeasurable tensor");
  }
  const std::int64_t head = q->ne[0];
  if (head != 64 && head != 256 && (head != 512 || k->type != GGML_TYPE_Q8_0)) {
    return Rejected(
        "quantized vector attention is compiled at head sizes 64 and 256 (and 512 for Q8_0)");
  }
  // ggml_flash_attn_ext's shapes (ggml.c) at head size D: Q [D, rows,
  // heads], K and V [D, cells, kv heads], one sample, a whole number of
  // query heads per KV head, the output [D, heads, rows].
  if (k->ne[0] != head || v->ne[0] != head || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
      !ggml_are_same_shape(k, v) || q->ne[2] % k->ne[2] != 0 || node->ne[0] != head ||
      node->ne[1] != q->ne[2] || node->ne[2] != q->ne[1] || node->ne[3] != 1) {
    return Rejected("vector attention whose shapes do not follow from its operands");
  }
  if (sinks != nullptr && (sinks->ne[0] != q->ne[2] || ggml_nelements(sinks) != q->ne[2] ||
                           sinks->nb[0] != sizeof(float) || !Aligned(sinks, sizeof(float)))) {
    return Rejected("one F32 sink per query head");
  }
  // The attended cells padded to 256, as the vector kernel requires
  // (fattn.cu's can_use_vector_kernel); one F16 mask row per query over
  // them, contiguous.
  if (k->ne[1] % kKqStride != 0 || mask->ne[0] != k->ne[1] || mask->ne[1] < q->ne[1] ||
      mask->ne[2] != 1 || mask->ne[3] != 1 || !ggml_is_contiguous(mask)) {
    return Rejected("attention over cells padded to 256 with one mask row per query");
  }
  // From 1,024 query rows launch_fattn runs the mask pre-pass
  // (flash_attn_mask_to_KV_max<2>), which reads two mask rows per tile with
  // no bound: an odd row count needs one more.
  if (q->ne[1] >= 1024 && mask->ne[1] < q->ne[1] + (q->ne[1] % 2)) {
    return Rejected("attention from 1,024 odd query rows without the mask row the pre-pass reads");
  }
  // A positive finite scale; a finite positive softcap at D 256 (the vector
  // kernel compiles none at other sizes: fattn-vec.cuh's use_logit_softcap
  // guard); no ALiBi; the precision llama.cpp sets.
  const float scale = ParamF32(node, 0);
  const float softcap = ParamF32(node, 2);
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const bool cap_valid = softcap == 0.0f || (head == 256 && softcap > 0.0f && softcap < kInf &&
                                             scale / softcap < kInf);
  if (!(scale > 0.0f) || !(scale < kInf) || ParamF32(node, 1) != 0.0f || !cap_valid ||
      node->op_params[3] != GGML_PREC_F32) {
    return Rejected(
        "attention requires positive finite scale, valid softcap, no ALiBi, F32 precision");
  }
  // Q, the mask and the output at the kernel's 16-byte vector loads; K and
  // V in whole blocks (the kernel reads a block's codes two bytes at a time
  // and its F16 scale); every stride and offset within the kernel's 32-bit
  // arguments.
  if (q->nb[0] != sizeof(float) || !BlockStrides(k) || !BlockStrides(v) || !Packed(node) ||
      !AlignedEverywhere(q, 16) || !AlignedEverywhere(mask, 16) || !Aligned(node, 16) ||
      Span(q) > kInt32Max / sizeof(float) || !FitsInt32Bytes(k) || !FitsInt32Bytes(v) ||
      Span(mask) > kInt32Max || Span(node) > kInt32Max ||
      mask->nb[1] >
          kInt32Max / static_cast<std::uint64_t>(std::max<std::int64_t>(1, q->ne[1] - 1))) {
    return Rejected("attention operands beyond the kernel's alignment or 32-bit indexing");
  }
  if (!AllCurrent({node, q, k, v, mask}) || (sinks != nullptr && !AllCurrent({sinks})) ||
      Overlap(node, q) || Overlap(node, k) || Overlap(node, v) || Overlap(node, mask) ||
      (sinks != nullptr && Overlap(node, sinks))) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSetRowsQuant(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_SET_ROWS || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound set_rows node");
  }
  const ggml_tensor* values = node->src[0];
  const ggml_tensor* ids = node->src[1];
  if (!IsF32(values) || !KvQuantType(node->type) ||
      (ids->type != GGML_TYPE_I32 && ids->type != GGML_TYPE_I64)) {
    return Rejected(
        "a quantizing set_rows stores F32 rows into Q8_0 or Q4_0 at I32 or I64 indices");
  }
  // The thread grid's extent, by checked arithmetic before anything below
  // forms a product of the shape.
  const auto count = detail::Product({values->ne[0], values->ne[1], values->ne[2], values->ne[3]});
  if (!count || *count > std::numeric_limits<std::uint32_t>::max()) {
    return Rejected("a quantizing set_rows beyond 2^32 - 1 values");
  }
  if (AnyEmpty({node, values, ids}) || !AllSane({node, values, ids})) {
    return Rejected("set_rows on an empty or unmeasurable tensor");
  }
  // ggml_set_rows's shape (ggml.c): a row of values per index, the ids'
  // channels and samples broadcast over the values'; rows of whole blocks.
  if (values->ne[0] != node->ne[0] || values->ne[0] % kBlockValues != 0 ||
      values->ne[1] != ids->ne[0] || ids->ne[3] != 1 || values->ne[2] % ids->ne[1] != 0 ||
      values->ne[3] % ids->ne[2] != 0 || node->ne[2] != values->ne[2] ||
      node->ne[3] != values->ne[3]) {
    return Rejected("set_rows whose shape does not follow from its operands");
  }
  if (values->nb[0] != sizeof(float) || ids->nb[0] != ggml_type_size(ids->type) ||
      !ElementStrides(values) || !ElementStrides(ids) || !BlockStrides(node)) {
    return Rejected("set_rows needs contiguous rows, whole-element strides and whole blocks");
  }
  if (!Aligned(values, sizeof(float)) || !Aligned(ids, ggml_type_size(ids->type))) {
    return Rejected("set_rows operands at misaligned addresses");
  }
  if (!AllCurrent({node, values, ids}) || Overlap(node, values) || Overlap(node, ids)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckSetRowsQuantGroup(
    std::span<const ggml_tensor* const> nodes) {
  if (nodes.size() < 2 || nodes.size() > kSetRowsQuantGroupMax) {
    return Rejected("a quantizing set_rows group takes two through sixteen stores");
  }
  for (const ggml_tensor* node : nodes) {
    if (auto checked = CheckSetRowsQuant(node); !checked) {
      return checked;
    }
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (std::size_t j = i + 1; j < nodes.size(); ++j) {
      const ggml_tensor* a = nodes[i];
      const ggml_tensor* b = nodes[j];
      if (Overlap(a, b) || Overlap(a, b->src[0]) || Overlap(a, b->src[1]) ||
          Overlap(b, a->src[0]) || Overlap(b, a->src[1])) {
        return Rejected("set_rows group destinations overlap a destination or operand");
      }
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckRopeSetRowsQuant(const ggml_tensor* rope,
                                                         const ggml_tensor* set_rows) {
  if (auto checked = CheckRopeOperands(rope); !checked) {
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
  // ggml.rope_set_rows.fused's contract (validate.h CheckRopeSetRows) with
  // a quantized destination: each token's heads, packed, one row of the
  // destination, a head of whole blocks.
  if (x->ne[3] != 1 || !KvQuantType(set_rows->type) || ids->type != GGML_TYPE_I64 ||
      !ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1] ||
      rope->ne[0] % kBlockValues != 0) {
    return Rejected("RoPE and a quantized KV write of whole blocks per head that this takes");
  }
  if (!Packed(rope) || view->view_offs != 0 || view->ne[1] != rope->ne[2] || view->ne[2] != 1 ||
      view->ne[3] != 1 || view->nb[1] != rope->nb[2]) {
    return Rejected("the stored view must flatten a packed RoPE's heads, one row per token");
  }
  if (AnyEmpty({set_rows, ids}) || !AllSane({set_rows, ids})) {
    return Rejected("a KV write to an empty or unmeasurable tensor");
  }
  if (set_rows->ne[0] != view->ne[0] || set_rows->ne[2] != 1 || set_rows->ne[3] != 1 ||
      ids->ne[0] != view->ne[1] || !ggml_is_vector(ids) || ids->nb[0] != sizeof(std::int64_t) ||
      !Aligned(ids, sizeof(std::int64_t))) {
    return Rejected("a KV write of one row per token into a two-dimensional destination");
  }
  // One thread per block of a token's row.
  const auto blocks = detail::Product({set_rows->ne[0] / kBlockValues, view->ne[1]});
  if (!BlockStrides(set_rows) || !blocks || *blocks > std::numeric_limits<std::uint32_t>::max()) {
    return Rejected("a KV write of whole blocks within the kernel's 32-bit grid");
  }
  if (!AllCurrent({set_rows, ids}) || !Disjoint(set_rows, x, /*in_place=*/false) ||
      !Disjoint(set_rows, rope->src[1], /*in_place=*/false) ||
      !Disjoint(set_rows, ids, /*in_place=*/false) ||
      (rope->src[2] != nullptr && !Disjoint(set_rows, rope->src[2], /*in_place=*/false))) {
    return Rejected("a stale view, or a destination overlapping an operand");
  }
  return {};
}

}  // namespace llmp::kernels::ggml
