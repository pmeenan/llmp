// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/jitllm_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <utility>

#include "cutlass/version.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/mxfp8_cutlass.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {

// The CUTLASS layout's scale swizzle (moe_cutlass.h SfOffset), which the
// checks here, the conversions and the importer's artifacts (modelopt_qwen38.py)
// all write out by hand, is CUTLASS 4.7.1's block-scaled layout, the source
// lock's `cutlass`: another CUTLASS is reviewed against it before the lock
// moves. (Every profile compiles this, the CPU-only one too, whose receipt
// lists the component.)
static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 7 && CUTLASS_PATCH == 1,
              "the CUTLASS expert layout was written against CUTLASS 4.7.1");

namespace {

// GGML's layout of a custom node's op_params (ggml-impl.h
// ggml_custom_op_params), which ggml_custom_4d fills.
struct JitllmCustomParams {
  ggml_custom_op_t fun;
  int n_tasks;
  void* userdata;
};

using detail::Aligned;
using detail::AlignedEverywhere;
using detail::AllCurrent;
using detail::AllSane;
using detail::AnyEmpty;
using detail::Bound;
using detail::Disjoint;
using detail::IsF32;
using detail::kInt32Max;
using detail::Packed;
using detail::Rejected;

// A jitLLM node's function, never called, and its operation's name: the
// userdata points at one of these distinct objects (distinct addresses,
// unlike functions a linker may fold).
void JitllmCustomTag(ggml_tensor* /*dst*/, int /*ith*/, int /*nth*/, void* /*userdata*/) {}
constinit std::array kTagMxfp8MulMatVec = std::to_array("jitllm.mxfp8.mul_mat_vec");
constinit std::array kTagMxfp8Dequant = std::to_array("jitllm.mxfp8.dequant");
constinit std::array kTagNvfp4Rows = std::to_array("jitllm.nvfp4.get_rows");
constinit std::array kTagQRows = std::to_array("jitllm.qrows.get_rows");
constinit std::array kTagArgmax = std::to_array("jitllm.argmax");
constinit std::array kTagHcCombine = std::to_array("jitllm.hc.combine");
constinit std::array kTagHcNorm = std::to_array("jitllm.hc.norm");
constinit std::array kTagHcMix = std::to_array("jitllm.hc.mix");
constinit std::array kTagMoeGlu = std::to_array("jitllm.moe.glu");
constinit std::array kTagMoeCombine = std::to_array("jitllm.moe.combine");
constinit std::array kTagBf16 = std::to_array("jitllm.bf16");
constinit std::array kTagGemmBf16 = std::to_array("jitllm.gemm.bf16");
constinit std::array kTagMoeRoute = std::to_array("jitllm.moe.route");
constinit std::array kTagMoeQuantize = std::to_array("jitllm.moe.quantize");
constinit std::array kTagMoeGemm = std::to_array("jitllm.moe.gemm");
constinit std::array kTagMoeGluQuantize = std::to_array("jitllm.moe.glu_quantize");
constinit std::array kTagMoeCombineSorted = std::to_array("jitllm.moe.combine_sorted");
constinit std::array kTagMoeGemv = std::to_array("jitllm.moe.gemv");
constinit std::array kTagGdnConv = std::to_array("jitllm.gdn.conv");
constinit std::array kTagGdnNormGate = std::to_array("jitllm.gdn.norm_gate");
constinit std::array kTagMxfp8Quantize = std::to_array("jitllm.mxfp8.quantize");
constinit std::array kTagMxfp8Swizzle = std::to_array("jitllm.mxfp8.swizzle");
constinit std::array kTagMxfp8Gemm = std::to_array("jitllm.mxfp8.gemm");
constinit std::array kTagHcPrep = std::to_array("jitllm.hc.prep");
constinit std::array kTagHcLo = std::to_array("jitllm.hc.lo");
constinit std::array kTagHcMixBf16 = std::to_array("jitllm.hc.mix_bf16");
constinit std::array kTagMoeRouter = std::to_array("jitllm.moe.router");
constinit std::array kTagGdnHistory = std::to_array("jitllm.gdn.history");
constinit std::array kTagQsaPrep = std::to_array("jitllm.qsa.prep");
constinit std::array kTagQsaGateQuantize = std::to_array("jitllm.qsa.gate_quantize");
constinit std::array kTagQsaPool = std::to_array("jitllm.qsa.pool");
constinit std::array kTagQsaTopK = std::to_array("jitllm.qsa.topk");
constinit std::array kTagQsaAttn = std::to_array("jitllm.qsa.attn");
constinit std::array kTagQuantizeQ8 = std::to_array("jitllm.q8_1");
constinit std::array kTagVecQ = std::to_array("jitllm.vecq");
constinit std::array kTagDsv4Route = std::to_array("jitllm.dsv4.route");
constinit std::array kTagDsv4Combine = std::to_array("jitllm.dsv4.combine");
constinit std::array kTagDsv4WeightedReduce = std::to_array("jitllm.dsv4.weighted_reduce");
constinit std::array kTagDsv4QHead = std::to_array("jitllm.dsv4.qhead");
constinit std::array kTagDsv4OutA = std::to_array("jitllm.dsv4.outa_prefill");
constinit std::array kTagDsv4HcNormF16 = std::to_array("jitllm.dsv4.hc_norm_f16");
constinit std::array kTagDsv4F16Copy = std::to_array("jitllm.dsv4.f16_copy");
constinit std::array kTagDsv4HcMix = std::to_array("jitllm.dsv4.hc_mix");
constinit std::array kTagDsv4HcPre = std::to_array("jitllm.dsv4.hc_pre");
constinit std::array kTagDsv4Compress = std::to_array("jitllm.dsv4.compress");
constinit std::array kTagGdnStep = std::to_array("jitllm.gdn.step");
constinit std::array kTagGdnGates = std::to_array("jitllm.gdn.gates");
constinit std::array kTagDsv4LidTopK = std::to_array("jitllm.dsv4.lid_topk");
constinit std::array kTagGemma4Mask = std::to_array("jitllm.gemma4.mask");
constinit std::array kTagFlashAttnOwners = std::to_array("jitllm.flash_attn.owner_roots");
constinit std::array kTagDsv4SparseMask = std::to_array("jitllm.dsv4.sparse_mask");

// Where a norm's epsilon sits in op_params: after GGML's custom parameters.
constexpr std::size_t kEpsOffset = 32;
static_assert(sizeof(JitllmCustomParams) <= kEpsOffset);
static_assert(kEpsOffset + (8 * sizeof(std::int32_t)) <= GGML_MAX_OP_PARAMS);

bool IsBytes(const ggml_tensor* t) { return t != nullptr && t->type == GGML_TYPE_I8; }

bool Matrix2d(const ggml_tensor* t) { return t->ne[2] == 1 && t->ne[3] == 1; }

// MXFP8 weights: codes I8 [k, n] and scales I8 [k / 32, n], both packed,
// codes 16-byte aligned for the kernels' vector loads.
std::expected<void, KernelFailure> CheckMxfp8Weights(const ggml_tensor* codes,
                                                     const ggml_tensor* scales) {
  if (!IsBytes(codes) || !IsBytes(scales) || !Bound(codes) || !Bound(scales)) {
    return Rejected("MXFP8 weights are bound I8 codes and scales");
  }
  if (AnyEmpty({codes, scales}) || !AllSane({codes, scales}) || !Matrix2d(codes) ||
      !Matrix2d(scales)) {
    return Rejected("MXFP8 weights are non-empty matrices");
  }
  const std::int64_t k = codes->ne[0];
  const std::int64_t n = codes->ne[1];
  if (k % 32 != 0 || scales->ne[0] != k / 32 || scales->ne[1] != n) {
    return Rejected("MXFP8 weights: k a multiple of 32, one scale per 32 codes a row");
  }
  if (!Packed(codes) || !Packed(scales) || !Aligned(codes, 16)) {
    return Rejected("MXFP8 weights packed, the codes 16-byte aligned");
  }
  if (std::cmp_greater(n, kInt32Max) || std::cmp_greater(k, kInt32Max)) {
    return Rejected("MXFP8 weights beyond the kernels' 32-bit extents");
  }
  return {};
}

std::expected<void, KernelFailure> CheckCustom(const ggml_tensor* node, JitllmOp op,
                                               int arguments) {
  if (node == nullptr || JitllmOpOf(node) != op || !Bound(node)) {
    return Rejected("not a bound node of this operation");
  }
  for (int i = 0; i < arguments; ++i) {
    if (!Bound(node->src[i])) {
      return Rejected("an unbound operand");
    }
  }
  if (arguments < GGML_MAX_SRC && node->src[arguments] != nullptr) {
    return Rejected("more operands than the operation takes");
  }
  return {};
}

ggml_tensor* Custom(ggml_context* context, ggml_type type, std::array<std::int64_t, 4> ne,
                    std::initializer_list<ggml_tensor*> args, char* name) {
  std::array<ggml_tensor*, GGML_MAX_SRC> list{};
  std::size_t n = 0;
  for (ggml_tensor* a : args) {
    list[n++] = a;
  }
  return ggml_custom_4d(context, type, ne[0], ne[1], ne[2], ne[3], list.data(), static_cast<int>(n),
                        &JitllmCustomTag, 1, name);
}

}  // namespace

JitllmOp JitllmOpOf(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_CUSTOM) {
    return JitllmOp::kNone;
  }
  JitllmCustomParams params{};
  static_assert(sizeof(params) <= sizeof(node->op_params));
  std::memcpy(&params, node->op_params, sizeof(params));
  if (params.fun != &JitllmCustomTag) {
    return JitllmOp::kNone;
  }
  if (params.userdata == kTagMxfp8MulMatVec.data()) {
    return JitllmOp::kMxfp8MulMatVec;
  }
  if (params.userdata == kTagMxfp8Dequant.data()) {
    return JitllmOp::kMxfp8Dequant;
  }
  if (params.userdata == kTagNvfp4Rows.data()) {
    return JitllmOp::kNvfp4Rows;
  }
  if (params.userdata == kTagQRows.data()) {
    return JitllmOp::kQRows;
  }
  if (params.userdata == kTagArgmax.data()) {
    return JitllmOp::kArgmax;
  }
  const std::array<std::pair<const char*, JitllmOp>, 46> fused = {{
      {kTagDsv4F16Copy.data(), JitllmOp::kDsv4F16Copy},
      {kTagDsv4HcNormF16.data(), JitllmOp::kDsv4HcNormF16},
      {kTagDsv4WeightedReduce.data(), JitllmOp::kDsv4WeightedReduce},
      {kTagDsv4QHead.data(), JitllmOp::kDsv4QHead},
      {kTagDsv4OutA.data(), JitllmOp::kDsv4OutA},
      {kTagGdnStep.data(), JitllmOp::kGdnStep},
      {kTagGdnGates.data(), JitllmOp::kGdnGates},
      {kTagDsv4LidTopK.data(), JitllmOp::kDsv4LidTopK},
      {kTagDsv4SparseMask.data(), JitllmOp::kDsv4SparseMask},
      {kTagGemma4Mask.data(), JitllmOp::kGemma4Mask},
      {kTagFlashAttnOwners.data(), JitllmOp::kFlashAttnOwners},
      {kTagQsaPool.data(), JitllmOp::kQsaPool},
      {kTagQsaTopK.data(), JitllmOp::kQsaTopK},
      {kTagQsaAttn.data(), JitllmOp::kQsaAttn},
      {kTagQsaPrep.data(), JitllmOp::kQsaPrep},
      {kTagQsaGateQuantize.data(), JitllmOp::kQsaGateQuantize},
      {kTagGdnHistory.data(), JitllmOp::kGdnHistory},
      {kTagMoeRouter.data(), JitllmOp::kMoeRouter},
      {kTagHcPrep.data(), JitllmOp::kHcPrep},
      {kTagHcLo.data(), JitllmOp::kHcLo},
      {kTagHcMixBf16.data(), JitllmOp::kHcMixBf16},
      {kTagMxfp8Quantize.data(), JitllmOp::kMxfp8Quantize},
      {kTagMxfp8Swizzle.data(), JitllmOp::kMxfp8Swizzle},
      {kTagMxfp8Gemm.data(), JitllmOp::kMxfp8Gemm},
      {kTagDsv4Compress.data(), JitllmOp::kDsv4Compress},
      {kTagQuantizeQ8.data(), JitllmOp::kQuantizeQ8},
      {kTagVecQ.data(), JitllmOp::kVecQ},
      {kTagDsv4Route.data(), JitllmOp::kDsv4Route},
      {kTagDsv4Combine.data(), JitllmOp::kDsv4Combine},
      {kTagDsv4HcMix.data(), JitllmOp::kDsv4HcMix},
      {kTagDsv4HcPre.data(), JitllmOp::kDsv4HcPre},
      {kTagGdnConv.data(), JitllmOp::kGdnConv},
      {kTagGdnNormGate.data(), JitllmOp::kGdnNormGate},
      {kTagHcCombine.data(), JitllmOp::kHcCombine},
      {kTagHcNorm.data(), JitllmOp::kHcNorm},
      {kTagHcMix.data(), JitllmOp::kHcMix},
      {kTagMoeGlu.data(), JitllmOp::kMoeGlu},
      {kTagMoeCombine.data(), JitllmOp::kMoeCombine},
      {kTagBf16.data(), JitllmOp::kBf16},
      {kTagGemmBf16.data(), JitllmOp::kGemmBf16},
      {kTagMoeRoute.data(), JitllmOp::kMoeRoute},
      {kTagMoeQuantize.data(), JitllmOp::kMoeQuantize},
      {kTagMoeGemm.data(), JitllmOp::kMoeGemm},
      {kTagMoeGluQuantize.data(), JitllmOp::kMoeGluQuantize},
      {kTagMoeCombineSorted.data(), JitllmOp::kMoeCombineSorted},
      {kTagMoeGemv.data(), JitllmOp::kMoeGemv},
  }};
  for (const auto& [tag, op] : fused) {
    if (params.userdata == tag) {
      return op;
    }
  }
  return JitllmOp::kNone;
}

float JitllmOpEps(const ggml_tensor* node) {
  float eps = 0.0f;
  std::memcpy(&eps, reinterpret_cast<const char*>(node->op_params) + kEpsOffset, sizeof(eps));
  return eps;
}

float JitllmOpFloat(const ggml_tensor* node, int index) {
  float value = 0.0f;
  if (index < 0 || index >= 8) {
    return 0.0f;
  }
  std::memcpy(&value,
              reinterpret_cast<const char*>(node->op_params) + kEpsOffset +
                  (static_cast<std::size_t>(index) * sizeof(value)),
              sizeof(value));
  return value;
}

std::int32_t JitllmOpInt(const ggml_tensor* node, int index) {
  std::int32_t value = 0;
  if (index < 0 || index >= 8) {
    return 0;
  }
  std::memcpy(&value,
              reinterpret_cast<const char*>(node->op_params) + kEpsOffset +
                  (static_cast<std::size_t>(index) * sizeof(value)),
              sizeof(value));
  return value;
}

ggml_tensor* Argmax(ggml_context* context, ggml_tensor* x, bool probability) {
  ggml_tensor* node = Custom(context, GGML_TYPE_I32, {(probability ? 2 : 1) * x->ne[1], 1, 1, 1},
                             {x}, kTagArgmax.data());
  // JitllmOpInt(node, 0): whether the probabilities follow.
  const std::int32_t flag = probability ? 1 : 0;
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset, &flag, sizeof(flag));
  return node;
}

std::expected<void, KernelFailure> CheckArgmax(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kArgmax, 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const std::int64_t per_row = JitllmOpInt(node, 0) == 1 ? 2 : 1;
  if (!IsF32(x) || node->type != GGML_TYPE_I32 || AnyEmpty({x, node}) || !AllSane({x, node}) ||
      !Matrix2d(x) || node->ne[0] != per_row * x->ne[1] || ggml_nrows(node) != 1) {
    return Rejected("F32 rows into one I32 index a row (and its probability)");
  }
  if (!Packed(x) || !Packed(node) || !Aligned(x, 4) || !Aligned(node, 4) ||
      std::cmp_greater(x->ne[0], kInt32Max) || x->ne[1] > 65535) {
    return Rejected("packed operands within the kernel's grid");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, false)) {
    return Rejected("a stale view, or an output overlapping its operand");
  }
  return {};
}

ggml_tensor* Mxfp8MulMatVec(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales,
                            ggml_tensor* x) {
  return Custom(context, GGML_TYPE_F32, {codes->ne[1], x->ne[1], 1, 1}, {codes, scales, x},
                kTagMxfp8MulMatVec.data());
}

ggml_tensor* Mxfp8Dequant(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales) {
  return Custom(context, GGML_TYPE_BF16, {codes->ne[0], codes->ne[1], 1, 1}, {codes, scales},
                kTagMxfp8Dequant.data());
}

ggml_tensor* Nvfp4Rows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids,
                       ggml_tensor* scale, std::int64_t values) {
  return Custom(context, GGML_TYPE_F32, {values, ids->ne[0], 1, 1}, {table, ids, scale},
                kTagNvfp4Rows.data());
}

bool QRowsType(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_IQ4_NL:
      return true;
    default:
      return false;
  }
}

ggml_tensor* QRows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids) {
  return Custom(context, GGML_TYPE_F32, {table->ne[0], ids->ne[0], 1, 1}, {table, ids},
                kTagQRows.data());
}

ggml_tensor* HcCombine(ggml_context* context, ggml_tensor* res, ggml_tensor* out,
                       ggml_tensor* inject) {
  return Custom(context, GGML_TYPE_F32, {res->ne[0], res->ne[1], res->ne[2], 1}, {res, out, inject},
                kTagHcCombine.data());
}

namespace {

ggml_tensor* WithEps(ggml_tensor* node, float eps) {
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset, &eps, sizeof(eps));
  return node;
}

// Integer parameters 0.. (at most 8), each within int32 (the builders'
// callers pass the model's extents; a value that does not fit is stored as
// -1, which every check refuses).
ggml_tensor* WithInts(ggml_tensor* node, std::initializer_list<std::int64_t> values) {
  std::size_t at = kEpsOffset;
  for (const std::int64_t v : values) {
    const std::int32_t stored =
        v >= 0 && std::cmp_less_equal(v, kInt32Max) ? static_cast<std::int32_t>(v) : -1;
    std::memcpy(reinterpret_cast<char*>(node->op_params) + at, &stored, sizeof(stored));
    at += sizeof(stored);
  }
  return node;
}

}  // namespace

ggml_tensor* HcNorm(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, float eps,
                    ggml_type type) {
  return WithEps(
      Custom(context, type, {x->ne[0] * x->ne[1], x->ne[2], 1, 1}, {x, weight}, kTagHcNorm.data()),
      eps);
}

ggml_tensor* HcMix(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* gate,
                   float eps) {
  return WithEps(Custom(context, GGML_TYPE_F32, {x->ne[0], x->ne[2], 1, 1}, {x, weight, gate},
                        kTagHcMix.data()),
                 eps);
}

ggml_tensor* MoeGlu(ggml_context* context, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* ids,
                    ggml_tensor* gate_scale, ggml_tensor* up_scale) {
  return Custom(context, GGML_TYPE_F32, {gate->ne[0], gate->ne[1], gate->ne[2], 1},
                {gate, up, ids, gate_scale, up_scale}, kTagMoeGlu.data());
}

ggml_tensor* MoeCombine(ggml_context* context, ggml_tensor* down, ggml_tensor* ids,
                        ggml_tensor* down_scale, ggml_tensor* weights, ggml_tensor* shared,
                        ggml_tensor* shared_gate) {
  if (down_scale == nullptr) {
    // Unscaled (JitllmOpInt(node, 0) == 1): the scale's slot left out.
    return WithInts(Custom(context, GGML_TYPE_F32, {down->ne[0], down->ne[2], 1, 1},
                           {down, ids, weights, shared, shared_gate}, kTagMoeCombine.data()),
                    {1});
  }
  return Custom(context, GGML_TYPE_F32, {down->ne[0], down->ne[2], 1, 1},
                {down, ids, down_scale, weights, shared, shared_gate}, kTagMoeCombine.data());
}

ggml_tensor* ToBf16(ggml_context* context, ggml_tensor* x) {
  return Custom(context, GGML_TYPE_BF16, {x->ne[0], x->ne[1], x->ne[2], x->ne[3]}, {x},
                kTagBf16.data());
}

ggml_tensor* GemmBf16(ggml_context* context, ggml_tensor* weights, ggml_tensor* x, ggml_type type) {
  return Custom(context, type, {weights->ne[1], x->ne[1], 1, 1}, {weights, x}, kTagGemmBf16.data());
}

ggml_tensor* GemvBf16(ggml_context* context, ggml_tensor* weights, ggml_tensor* x, ggml_type type) {
  return WithInts(GemmBf16(context, weights, x, type), {1});
}

bool IsGemvBf16(const ggml_tensor* node) {
  return JitllmOpOf(node) == JitllmOp::kGemmBf16 && JitllmOpInt(node, 0) == 1;
}

ggml_tensor* GdnConv(ggml_context* context, ggml_tensor* x, ggml_tensor* history,
                     ggml_tensor* weight, std::int64_t qk_channels, std::int64_t head, float eps,
                     float scale) {
  ggml_tensor* node = WithInts(Custom(context, GGML_TYPE_F32, {x->ne[0], x->ne[1], 1, 1},
                                      {x, history, weight}, kTagGdnConv.data()),
                               {qk_channels, head});
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 8, &eps, sizeof(eps));
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 12, &scale, sizeof(scale));
  return node;
}

ggml_tensor* GdnNormGate(ggml_context* context, ggml_tensor* o, ggml_tensor* weight, ggml_tensor* z,
                         float eps, ggml_type type) {
  const std::int64_t k = o->ne[0] * o->ne[1];
  const std::int64_t t = o->ne[2];
  const std::array<std::int64_t, 4> ne =
      type == GGML_TYPE_I8
          ? std::array<std::int64_t, 4>{static_cast<std::int64_t>(
                                            mxfp8::RowsLayout{.k = static_cast<std::uint64_t>(k),
                                                              .rows = static_cast<std::uint64_t>(t)}
                                                .bytes()),
                                        1, 1, 1}
          : std::array<std::int64_t, 4>{k, t, 1, 1};
  return WithEps(Custom(context, type, ne, {o, weight, z}, kTagGdnNormGate.data()), eps);
}

ggml_tensor* GdnHistory(ggml_context* context, ggml_tensor* x, std::int64_t taps,
                        ggml_tensor* history) {
  if (history != nullptr) {
    return WithInts(Custom(context, GGML_TYPE_F32, {taps * x->ne[0], 1, 1, 1}, {x, history},
                           kTagGdnHistory.data()),
                    {taps});
  }
  return WithInts(
      Custom(context, GGML_TYPE_F32, {taps * x->ne[0], 1, 1, 1}, {x}, kTagGdnHistory.data()),
      {taps});
}

ggml_tensor* GdnGates(ggml_context* context, ggml_tensor* alpha, ggml_tensor* beta,
                      ggml_tensor* dt_bias, ggml_tensor* ssm_a) {
  return Custom(context, GGML_TYPE_F32, {alpha->ne[0], alpha->ne[1], 2, 1},
                {alpha, beta, dt_bias, ssm_a}, kTagGdnGates.data());
}

ggml_tensor* GdnStep(ggml_context* context, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                     ggml_tensor* g, ggml_tensor* beta, ggml_tensor* state, bool write_state) {
  return WithInts(Custom(context, GGML_TYPE_F32, {v->ne[0], v->ne[1], v->ne[2], 1},
                         {q, k, v, g, beta, state}, kTagGdnStep.data()),
                  {write_state ? 0 : 1});
}

namespace {

// A route node's extents: experts, experts used, tokens (its parameters).
struct RouteExtents {
  std::int64_t experts = 0;
  std::int64_t used = 0;
  std::int64_t tokens = 0;
  std::int64_t slots() const { return used * tokens; }
  moe::RouteLayout layout() const {
    return {.experts = experts,
            .slots = slots(),
            .chunks = (tokens + moe::kRouteChunk - 1) / moe::kRouteChunk};
  }
  moe::QuantLayout quant(std::int64_t k) const {
    return {.k = static_cast<std::uint64_t>(k),
            .slots = static_cast<std::uint64_t>(slots()),
            .experts = static_cast<std::uint64_t>(experts)};
  }
};

RouteExtents ExtentsOf(const ggml_tensor* route) {
  return {.experts = JitllmOpInt(route, 0),
          .used = JitllmOpInt(route, 1),
          .tokens = JitllmOpInt(route, 2)};
}

std::int64_t QuantBytes(const RouteExtents& r, std::int64_t k) {
  return static_cast<std::int64_t>(r.quant(k).bytes());
}

}  // namespace

ggml_tensor* MoeRoute(ggml_context* context, ggml_tensor* ids, std::int64_t experts) {
  const RouteExtents r{.experts = experts, .used = ids->ne[0], .tokens = ids->ne[1]};
  return WithInts(
      Custom(context, GGML_TYPE_I32, {r.layout().ints(), 1, 1, 1}, {ids}, kTagMoeRoute.data()),
      {r.experts, r.used, r.tokens});
}

ggml_tensor* MoeQuantize(ggml_context* context, ggml_tensor* x, ggml_tensor* route) {
  const RouteExtents r = ExtentsOf(route);
  const std::int64_t k = x->ne[0];
  return WithInts(Custom(context, GGML_TYPE_I8, {QuantBytes(r, k), 1, 1, 1}, {x, route},
                         kTagMoeQuantize.data()),
                  {r.experts, r.used, r.tokens, k});
}

ggml_tensor* MoeGemm(ggml_context* context, ggml_tensor* a, ggml_tensor* route,
                     ggml_tensor* weights, std::int64_t n, std::uint64_t codes_offset,
                     std::uint64_t scales_offset) {
  const RouteExtents r = ExtentsOf(route);
  const std::int64_t k = JitllmOpInt(a, 3);
  return WithInts(Custom(context, GGML_TYPE_BF16, {n, r.slots(), 1, 1}, {a, route, weights},
                         kTagMoeGemm.data()),
                  {r.experts, r.used, r.tokens, k, n, static_cast<std::int64_t>(codes_offset),
                   static_cast<std::int64_t>(scales_offset)});
}

ggml_tensor* MoeGluQuantize(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                            ggml_tensor* route, ggml_tensor* gate_scale, ggml_tensor* up_scale) {
  const RouteExtents r = ExtentsOf(route);
  const std::int64_t f = d->ne[0] / 2;
  return WithInts(Custom(context, GGML_TYPE_I8, {QuantBytes(r, f), 1, 1, 1},
                         {d, a, route, gate_scale, up_scale}, kTagMoeGluQuantize.data()),
                  {r.experts, r.used, r.tokens, f});
}

ggml_tensor* MoeCombineSorted(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                              ggml_tensor* route, ggml_tensor* down_scale, ggml_tensor* weights,
                              ggml_tensor* shared, ggml_tensor* shared_gate) {
  const RouteExtents r = ExtentsOf(route);
  return WithInts(
      Custom(context, GGML_TYPE_F32, {d->ne[0], r.tokens, 1, 1},
             {d, a, route, down_scale, weights, shared, shared_gate}, kTagMoeCombineSorted.data()),
      {r.experts, r.used, r.tokens});
}

ggml_tensor* MoeGemv(ggml_context* context, ggml_tensor* weights, ggml_tensor* x, ggml_tensor* ids,
                     std::int64_t n, std::int64_t row0, std::int64_t rows,
                     std::uint64_t codes_offset, std::uint64_t scales_offset) {
  return WithInts(Custom(context, GGML_TYPE_F32, {n, ids->ne[0], ids->ne[1], 1}, {weights, x, ids},
                         kTagMoeGemv.data()),
                  {row0, rows, static_cast<std::int64_t>(codes_offset),
                   static_cast<std::int64_t>(scales_offset)});
}

ggml_tensor* MoeGemvSwiglu(ggml_context* context, ggml_tensor* weights, ggml_tensor* x,
                           ggml_tensor* ids, std::int64_t f, ggml_tensor* gate_scale,
                           ggml_tensor* up_scale, std::uint64_t codes_offset,
                           std::uint64_t scales_offset) {
  return WithInts(Custom(context, GGML_TYPE_F32, {f, ids->ne[0], ids->ne[1], 1},
                         {weights, x, ids, gate_scale, up_scale}, kTagMoeGemv.data()),
                  {0, 2 * f, static_cast<std::int64_t>(codes_offset),
                   static_cast<std::int64_t>(scales_offset)});
}

bool IsMoeGemvSwiglu(const ggml_tensor* node) { return node != nullptr && node->src[3] != nullptr; }

std::expected<void, KernelFailure> CheckMxfp8MulMatVec(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8MulMatVec, 3); !checked) {
    return checked;
  }
  const ggml_tensor* codes = node->src[0];
  const ggml_tensor* scales = node->src[1];
  const ggml_tensor* x = node->src[2];
  if (auto checked = CheckMxfp8Weights(codes, scales); !checked) {
    return checked;
  }
  if (!IsF32(x) || !IsF32(node) || AnyEmpty({x, node}) || !AllSane({x, node}) || !Matrix2d(x) ||
      !Matrix2d(node)) {
    return Rejected("F32 activations and output, matrices");
  }
  const std::int64_t t = x->ne[1];
  if (x->ne[0] != codes->ne[0] || node->ne[0] != codes->ne[1] || node->ne[1] != t ||
      t > kMxfp8VecWaveColumns) {
    return Rejected("y[n, t] = W[n, k] x[k, t] for at most 16 columns");
  }
  // The kernel reads x in 16-byte vectors along each column.
  // The kernel's grid counts rows in int, 8 a block.
  if (x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || !Aligned(x, 16) || !Packed(node) ||
      !Aligned(node, sizeof(float)) || std::cmp_greater(x->nb[1] / sizeof(float), kInt32Max) ||
      std::cmp_greater(codes->ne[1], kInt32Max - 8)) {
    return Rejected("x columns 16-byte aligned at 16-byte strides, and a packed, aligned output");
  }
  if (!AllCurrent({node, codes, scales, x}) || !Disjoint(node, codes, false) ||
      !Disjoint(node, scales, false) || !Disjoint(node, x, false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMxfp8Dequant(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8Dequant, 2); !checked) {
    return checked;
  }
  const ggml_tensor* codes = node->src[0];
  const ggml_tensor* scales = node->src[1];
  if (auto checked = CheckMxfp8Weights(codes, scales); !checked) {
    return checked;
  }
  if (node->type != GGML_TYPE_BF16 || !ggml_are_same_shape(node, codes) || !Packed(node) ||
      !Aligned(node, 16)) {
    return Rejected("a packed, 16-byte aligned BF16 matrix of the codes' shape");
  }
  if (std::cmp_greater(ggml_nelements(codes) / 16, kInt32Max) ||
      !AllCurrent({node, codes, scales}) || !Disjoint(node, codes, false) ||
      !Disjoint(node, scales, false)) {
    return Rejected("beyond the kernel's grid, a stale view, or an overlapping output");
  }
  return {};
}

std::expected<void, KernelFailure> CheckNvfp4Rows(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kNvfp4Rows, 3); !checked) {
    return checked;
  }
  const ggml_tensor* table = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const ggml_tensor* scale = node->src[2];
  const std::int64_t values = node->ne[0];
  if (!IsBytes(table) || ids->type != GGML_TYPE_I32 || !IsF32(scale) || !IsF32(node)) {
    return Rejected("an I8 table, I32 ids and an F32 scale into F32");
  }
  if (AnyEmpty({table, ids, scale, node}) || !AllSane({table, ids, scale, node}) ||
      !Matrix2d(table) || !Matrix2d(node) || ggml_nelements(scale) != 1 || ggml_nrows(ids) != 1) {
    return Rejected("a table of rows, one row of ids, one scale");
  }
  if (values % 16 != 0 || values > 1024 || table->ne[0] != (values / 2) + (values / 16) ||
      node->ne[1] != ids->ne[0]) {
    return Rejected("rows of a multiple of 16 values (at most 1,024): codes then scales");
  }
  if (!Packed(table) || !Packed(ids) || !Packed(node) || !Aligned(ids, 4) || !Aligned(scale, 4) ||
      !Aligned(node, 4) || std::cmp_greater(table->ne[1], kInt32Max) ||
      std::cmp_greater(ids->ne[0], 65535 * 1024LL)) {
    return Rejected("packed operands within the kernel's grid");
  }
  if (!AllCurrent({node, table, ids, scale}) || !Disjoint(node, table, false) ||
      !Disjoint(node, ids, false) || !Disjoint(node, scale, false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckQRows(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQRows, 2); !checked) {
    return checked;
  }
  const ggml_tensor* table = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const std::int64_t values = node->ne[0];
  if (!QRowsType(table->type) || ids->type != GGML_TYPE_I32 || !IsF32(node)) {
    return Rejected("a table of a 32-value block type and I32 ids into F32");
  }
  if (AnyEmpty({table, ids, node}) || !AllSane({table, ids, node}) || !Matrix2d(table) ||
      !Matrix2d(node) || ggml_nrows(ids) != 1) {
    return Rejected("a table of rows and one row of ids");
  }
  if (values % 32 != 0 || values > 1024 || table->ne[0] != values || node->ne[1] != ids->ne[0]) {
    return Rejected("rows of a multiple of 32 values (at most 1,024), one a row id");
  }
  // Blocks read their scale as a half, or Q4_1's and Q5_1's as a half2.
  const std::uint64_t block_align =
      table->type == GGML_TYPE_Q4_1 || table->type == GGML_TYPE_Q5_1 ? 4 : 2;
  if (!Packed(table) || !Packed(ids) || !Packed(node) || !Aligned(table, block_align) ||
      !Aligned(ids, 4) || !Aligned(node, 4) || std::cmp_greater(table->ne[1], kInt32Max) ||
      std::cmp_greater(ids->ne[0], 65535 * 1024LL)) {
    return Rejected("packed operands within the kernel's grid");
  }
  if (!AllCurrent({node, table, ids}) || !Disjoint(node, table, false) ||
      !Disjoint(node, ids, false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

namespace {

bool Shaped(const ggml_tensor* t, std::int64_t n0, std::int64_t n1, std::int64_t n2) {
  return t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == 1;
}

// Bound, sane, non-empty, packed operands and an output disjoint from each,
// every view current.
std::expected<void, KernelFailure> CheckDense(const ggml_tensor* node,
                                              std::initializer_list<const ggml_tensor*> packed,
                                              const ggml_tensor* strided = nullptr) {
  if (!Packed(node) || AnyEmpty({node}) || !AllSane({node}) || !AllCurrent({node})) {
    return Rejected("a packed, non-empty output");
  }
  for (const ggml_tensor* t : packed) {
    if (!Packed(t) || AnyEmpty({t}) || !AllSane({t}) || !AllCurrent({t}) ||
        !Disjoint(node, t, false)) {
      return Rejected("packed, non-empty operands, current views, disjoint from the output");
    }
  }
  if (strided != nullptr && (AnyEmpty({strided}) || !AllSane({strided}) || !AllCurrent({strided}) ||
                             !Disjoint(node, strided, false))) {
    return Rejected("the expert ids are measurable, current and disjoint from the output");
  }
  return {};
}

// Expert ids: I32 [used, t] with packed elements and a row stride of whole
// ids at least a row long.
bool ExpertIds(const ggml_tensor* ids, std::int64_t used, std::int64_t t) {
  return ids->type == GGML_TYPE_I32 && Shaped(ids, used, t, 1) &&
         ids->nb[0] == sizeof(std::int32_t) && ids->nb[1] % sizeof(std::int32_t) == 0 &&
         ids->nb[1] >= static_cast<std::size_t>(used) * sizeof(std::int32_t) &&
         ids->nb[1] / sizeof(std::int32_t) <= kInt32Max && Aligned(ids, sizeof(std::int32_t));
}

bool Vector(const ggml_tensor* t, std::int64_t n) { return IsF32(t) && Shaped(t, n, 1, 1); }

}  // namespace

std::expected<void, KernelFailure> CheckHcCombine(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcCombine, 3); !checked) {
    return checked;
  }
  const ggml_tensor* res = node->src[0];
  const ggml_tensor* out = node->src[1];
  const ggml_tensor* inject = node->src[2];
  const std::int64_t width = res->ne[0];
  const std::int64_t hc = res->ne[1];
  const std::int64_t t = res->ne[2];
  if (!IsF32(node) || !IsF32(res) || !IsF32(out) || !IsF32(inject) || res->ne[3] != 1 ||
      !Shaped(node, width, hc, t) || !Shaped(out, width, t, 1) || !Shaped(inject, hc, t, 1)) {
    return Rejected("F32 streams [width, hc, t], an output [width, t] and weights [hc, t]");
  }
  if (width % 4 != 0 || !Aligned(node, 16) || !Aligned(res, 16) || !Aligned(out, 16) ||
      !Aligned(inject, sizeof(float)) || std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected("rows of whole float4s, 16-byte aligned, within the kernel's grid");
  }
  return CheckDense(node, {res, out, inject});
}

namespace {

// The norms' streams: x F32 [width, hc, t] and its weight F32 [width · hc],
// width at least GGML's 1,024-thread rms_norm's.
std::expected<void, KernelFailure> CheckNormOperands(const ggml_tensor* node, const ggml_tensor* x,
                                                     const ggml_tensor* weight) {
  const std::int64_t width = x->ne[0];
  const std::int64_t hc = x->ne[1];
  const float eps = JitllmOpEps(node);
  if (!IsF32(x) || x->ne[3] != 1 || !Vector(weight, width * hc)) {
    return Rejected("F32 streams [width, hc, t] and an F32 weight [width · hc]");
  }
  if (width < 1024 || std::cmp_greater(width, kInt32Max) || hc > 8 ||
      std::cmp_greater(x->ne[2], kInt32Max) || !std::isfinite(eps) || eps < 0.0f) {
    return Rejected("streams of 1,024 to 2^31 elements, at most 8 of them, a finite epsilon");
  }
  if (!Aligned(x, sizeof(float)) || !Aligned(weight, sizeof(float))) {
    return Rejected("aligned F32 operands");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckHcNorm(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcNorm, 2); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* weight = node->src[1];
  if (auto checked = CheckNormOperands(node, x, weight); !checked) {
    return checked;
  }
  if ((node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) ||
      !Shaped(node, x->ne[0] * x->ne[1], x->ne[2], 1) ||
      !Aligned(node, ggml_type_size(node->type))) {
    return Rejected("an aligned F32 or BF16 output [width · hc, t]");
  }
  return CheckDense(node, {x, weight});
}

std::expected<void, KernelFailure> CheckHcMix(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcMix, 3); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* weight = node->src[1];
  const ggml_tensor* gate = node->src[2];
  if (auto checked = CheckNormOperands(node, x, weight); !checked) {
    return checked;
  }
  if (!IsF32(gate) || !Shaped(gate, x->ne[0] * x->ne[1], x->ne[2], 1) || !IsF32(node) ||
      !Shaped(node, x->ne[0], x->ne[2], 1) || !Aligned(gate, sizeof(float)) ||
      !Aligned(node, sizeof(float))) {
    return Rejected("an F32 gate [width · hc, t] and an aligned F32 output [width, t]");
  }
  return CheckDense(node, {x, weight, gate});
}

std::expected<void, KernelFailure> CheckMoeGlu(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGlu, 5); !checked) {
    return checked;
  }
  const ggml_tensor* gate = node->src[0];
  const ggml_tensor* up = node->src[1];
  const ggml_tensor* ids = node->src[2];
  const ggml_tensor* gate_scale = node->src[3];
  const ggml_tensor* up_scale = node->src[4];
  const std::int64_t n = gate->ne[0];
  const std::int64_t used = gate->ne[1];
  const std::int64_t t = gate->ne[2];
  const std::int64_t experts = gate_scale->ne[0];
  if (!IsF32(node) || !IsF32(gate) || !IsF32(up) || gate->ne[3] != 1 || !Shaped(up, n, used, t) ||
      !Shaped(node, n, used, t) || !ExpertIds(ids, used, t) || !Vector(gate_scale, experts) ||
      !Vector(up_scale, experts) || std::cmp_greater(experts, kInt32Max)) {
    return Rejected("F32 products [n, used, t], I32 ids [used, t] and F32 scales [experts]");
  }
  if (n % 4 != 0 || !Aligned(node, 16) || !Aligned(gate, 16) || !Aligned(up, 16) ||
      !Aligned(gate_scale, sizeof(float)) || !Aligned(up_scale, sizeof(float)) ||
      std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected("rows of whole float4s, 16-byte aligned, within the kernel's grid");
  }
  return CheckDense(node, {gate, up, gate_scale, up_scale}, ids);
}

std::expected<void, KernelFailure> CheckMoeCombine(const ggml_tensor* node) {
  // Unscaled (a GGUF checkpoint's experts): the scale's slot left out.
  const bool unscaled = node != nullptr && JitllmOpInt(node, 0) == 1;
  if (auto checked = CheckCustom(node, JitllmOp::kMoeCombine, unscaled ? 5 : 6); !checked) {
    return checked;
  }
  const int at = unscaled ? 2 : 3;  // the weights' slot
  const ggml_tensor* down = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const ggml_tensor* down_scale = unscaled ? nullptr : node->src[2];
  const ggml_tensor* weights = node->src[at];
  const ggml_tensor* shared = node->src[at + 1];
  const ggml_tensor* shared_gate = node->src[at + 2];
  const std::int64_t width = down->ne[0];
  const std::int64_t used = down->ne[1];
  const std::int64_t t = down->ne[2];
  if (down_scale != nullptr &&
      (!Vector(down_scale, down_scale->ne[0]) || std::cmp_greater(down_scale->ne[0], kInt32Max) ||
       !Aligned(down_scale, sizeof(float)))) {
    return Rejected("aligned F32 scales [experts]");
  }
  if (!IsF32(node) || !IsF32(down) || down->ne[3] != 1 || !ExpertIds(ids, used, t) ||
      !IsF32(weights) || !Shaped(weights, 1, used, t) || !IsF32(shared) ||
      !Shaped(shared, width, t, 1) || !IsF32(shared_gate) || !Shaped(shared_gate, 1, t, 1) ||
      !Shaped(node, width, t, 1)) {
    return Rejected(
        "F32 products [width, used, t], ids [used, t], weights [1, used, t], a shared product "
        "[width, t] and its gate [1, t]");
  }
  if (width % 4 != 0 || !Aligned(node, 16) || !Aligned(down, 16) || !Aligned(shared, 16) ||
      !Aligned(weights, sizeof(float)) || !Aligned(shared_gate, sizeof(float)) ||
      std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected("rows of whole float4s, 16-byte aligned, within the kernel's grid");
  }
  if (down_scale == nullptr) {
    return CheckDense(node, {down, weights, shared, shared_gate}, ids);
  }
  return CheckDense(node, {down, down_scale, weights, shared, shared_gate}, ids);
}

std::expected<void, KernelFailure> CheckBf16(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kBf16, 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(x) || node->type != GGML_TYPE_BF16 || !ggml_are_same_shape(node, x) ||
      !Aligned(x, sizeof(float)) || !Aligned(node, 2) ||
      std::cmp_greater(ggml_nelements(node), kInt32Max)) {
    return Rejected("F32 into an aligned BF16 tensor of its shape, within the kernel's grid");
  }
  return CheckDense(node, {x});
}

namespace {

// A route node's extents, each positive and within the kernels' grids.
bool SaneExtents(const RouteExtents& r) {
  return r.experts > 0 && r.experts <= 4096 && r.used > 0 && r.used <= 64 && r.tokens > 0 &&
         std::cmp_less_equal(r.layout().ints(), kInt32Max) &&
         std::cmp_less_equal(r.slots(), kInt32Max / 4);
}

bool IsRoute(const ggml_tensor* route, const RouteExtents& want) {
  if (JitllmOpOf(route) != JitllmOp::kMoeRoute || !Bound(route) || !Packed(route) ||
      !Aligned(route, sizeof(std::int32_t)) || !AllCurrent({route})) {
    return false;
  }
  const RouteExtents r = ExtentsOf(route);
  return r.experts == want.experts && r.used == want.used && r.tokens == want.tokens &&
         route->type == GGML_TYPE_I32 && Shaped(route, r.layout().ints(), 1, 1);
}

// A quantization of k-wide rows over the route's slots, laid out by that
// route node itself (its rows and scale blocks are where that route put
// them; another route of the same extents may sort differently).
bool IsQuantized(const ggml_tensor* a, const ggml_tensor* route, const RouteExtents& r,
                 std::int64_t k) {
  const JitllmOp op = JitllmOpOf(a);
  const ggml_tensor* laid_out_by = nullptr;
  if (op == JitllmOp::kMoeQuantize) {
    laid_out_by = a->src[1];
  } else if (op == JitllmOp::kMoeGluQuantize) {
    laid_out_by = a->src[2];
  }
  return laid_out_by != nullptr && laid_out_by == route && Bound(a) && a->type == GGML_TYPE_I8 &&
         Packed(a) && Aligned(a, 16) && AllCurrent({a}) && JitllmOpInt(a, 0) == r.experts &&
         JitllmOpInt(a, 1) == r.used && JitllmOpInt(a, 2) == r.tokens && JitllmOpInt(a, 3) == k &&
         Shaped(a, QuantBytes(r, k), 1, 1);
}

// Expert weights in the CUTLASS layout: I8 [stride, experts], the stride
// whole 16 bytes; `rows` rows of k at the offsets fit each slot.
bool ExpertSlots(const ggml_tensor* w, std::int64_t experts, std::int64_t rows, std::int64_t k,
                 std::int64_t codes, std::int64_t scales) {
  if (w->type != GGML_TYPE_I8 || !Bound(w) || !Shaped(w, w->ne[0], experts, 1) || !Packed(w) ||
      !Aligned(w, 16) || w->ne[0] % 16 != 0 || AnyEmpty({w}) || !AllSane({w}) || !AllCurrent({w})) {
    return false;
  }
  const std::int64_t stride = w->ne[0];
  return rows > 0 && rows % moe::kScaleRows == 0 && k > 0 && k % 64 == 0 && codes >= 0 &&
         scales >= 0 && codes % 16 == 0 && scales % 16 == 0 && codes + (rows * (k / 2)) <= stride &&
         scales + (rows * (k / 16)) <= stride;
}

}  // namespace

std::expected<void, KernelFailure> CheckMoeRoute(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeRoute, 1); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* ids = node->src[0];
  if (!SaneExtents(r) || !ExpertIds(ids, r.used, r.tokens) || node->type != GGML_TYPE_I32 ||
      !Shaped(node, r.layout().ints(), 1, 1) || !Aligned(node, sizeof(std::int32_t))) {
    return Rejected("I32 expert ids [used, t] into the routing's layout of their extents");
  }
  return CheckDense(node, {}, ids);
}

std::expected<void, KernelFailure> CheckMoeQuantize(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeQuantize, 2); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* x = node->src[0];
  const std::int64_t k = JitllmOpInt(node, 3);
  if (!SaneExtents(r) || !IsRoute(node->src[1], r) || !IsF32(x) || !Shaped(x, k, r.tokens, 1) ||
      k % 64 != 0 || k <= 0 || k > 16384 || !Aligned(x, 16) || node->type != GGML_TYPE_I8 ||
      !Shaped(node, QuantBytes(r, k), 1, 1) || !Aligned(node, 16)) {
    return Rejected("F32 activations [k, t], k a multiple of 64, into the route's quantized rows");
  }
  return CheckDense(node, {x, node->src[1]});
}

std::expected<void, KernelFailure> CheckMoeGemm(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGemm, 3); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const std::int64_t k = JitllmOpInt(node, 3);
  const std::int64_t n = JitllmOpInt(node, 4);
  const std::int64_t codes = JitllmOpInt(node, 5);
  const std::int64_t scales = JitllmOpInt(node, 6);
  if (!SaneExtents(r) || !IsRoute(node->src[1], r) ||
      !IsQuantized(node->src[0], node->src[1], r, k) ||
      !ExpertSlots(node->src[2], r.experts, n, k, codes, scales) || node->type != GGML_TYPE_BF16 ||
      !Shaped(node, n, r.slots(), 1) || !Aligned(node, 16)) {
    return Rejected(
        "quantized rows of the route's slots, expert weights in the CUTLASS layout, and a BF16 "
        "output [n, slots]");
  }
  return CheckDense(node, {node->src[0], node->src[1], node->src[2]});
}

std::expected<void, KernelFailure> CheckMoeGluQuantize(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGluQuantize, 5); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* d = node->src[0];
  const ggml_tensor* a = node->src[1];
  const std::int64_t f = JitllmOpInt(node, 3);
  // The kernel stages a row's f activations in dynamic shared memory, within
  // the default 48 KiB beside its 32 warp maxima (no attribute is raised).
  constexpr std::int64_t kMostF = ((48 * 1024) - (32 * 4)) / 4 / 64 * 64;
  if (!SaneExtents(r) || !IsRoute(node->src[2], r) || f <= 0 || f % 64 != 0 || f > kMostF ||
      d->type != GGML_TYPE_BF16 || !Shaped(d, 2 * f, r.slots(), 1) || !Aligned(d, 16) ||
      JitllmOpOf(a) != JitllmOp::kMoeQuantize ||
      !IsQuantized(a, node->src[2], r, JitllmOpInt(a, 3)) || !Vector(node->src[3], r.experts) ||
      !Vector(node->src[4], r.experts) || !Aligned(node->src[3], sizeof(float)) ||
      !Aligned(node->src[4], sizeof(float)) || node->type != GGML_TYPE_I8 ||
      !Shaped(node, QuantBytes(r, f), 1, 1) || !Aligned(node, 16)) {
    return Rejected(
        "BF16 gate and up products [2f, slots], their input's quantization, F32 scales "
        "[experts], into the route's quantized rows of f");
  }
  return CheckDense(node, {d, a, node->src[2], node->src[3], node->src[4]});
}

std::expected<void, KernelFailure> CheckMoeCombineSorted(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeCombineSorted, 7); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* d = node->src[0];
  const ggml_tensor* a = node->src[1];
  const std::int64_t width = d->ne[0];
  if (!SaneExtents(r) || !IsRoute(node->src[2], r) || width <= 0 || width % 4 != 0 ||
      d->type != GGML_TYPE_BF16 || !Shaped(d, width, r.slots(), 1) || !Aligned(d, 16) ||
      JitllmOpOf(a) != JitllmOp::kMoeGluQuantize ||
      !IsQuantized(a, node->src[2], r, JitllmOpInt(a, 3)) || !Vector(node->src[3], r.experts) ||
      !Aligned(node->src[3], sizeof(float)) || !IsF32(node->src[4]) ||
      !Shaped(node->src[4], 1, r.used, r.tokens) || !Aligned(node->src[4], sizeof(float)) ||
      !IsF32(node->src[5]) || !Shaped(node->src[5], width, r.tokens, 1) ||
      !Aligned(node->src[5], 16) || !IsF32(node->src[6]) || !Shaped(node->src[6], 1, r.tokens, 1) ||
      !Aligned(node->src[6], sizeof(float)) || !IsF32(node) || !Shaped(node, width, r.tokens, 1) ||
      !Aligned(node, 16) || std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected(
        "BF16 down products [w, slots], their input's quantization, F32 scales [experts], "
        "weights [1, used, t], a shared product [w, t] and its gate [1, t]");
  }
  return CheckDense(node,
                    {d, a, node->src[2], node->src[3], node->src[4], node->src[5], node->src[6]});
}

std::expected<void, KernelFailure> CheckMoeGemv(const ggml_tensor* node) {
  const bool swiglu = IsMoeGemvSwiglu(node);
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGemv, swiglu ? 5 : 3); !checked) {
    return checked;
  }
  const ggml_tensor* w = node->src[0];
  const ggml_tensor* x = node->src[1];
  const ggml_tensor* ids = node->src[2];
  const std::int64_t n = node->ne[0];
  const std::int64_t used = ids->ne[0];
  const std::int64_t t = ids->ne[1];
  const std::int64_t k = x->ne[0];
  const std::int64_t row0 = JitllmOpInt(node, 0);
  const std::int64_t rows = JitllmOpInt(node, 1);
  if (!ExpertSlots(w, w->ne[1], rows, k, JitllmOpInt(node, 2), JitllmOpInt(node, 3)) || row0 < 0 ||
      n <= 0 || row0 + n > rows || !ExpertIds(ids, used, t) || t > kMoeGemvWaveTokens ||
      used > 64 || !IsF32(x) || (x->ne[1] != 1 && x->ne[1] != used) || x->ne[2] != t ||
      x->ne[3] != 1 || !Aligned(x, 16) || !IsF32(node) || !Shaped(node, n, used, t) ||
      !Aligned(node, sizeof(float)) || std::cmp_greater(w->ne[1], kInt32Max)) {
    return Rejected(
        "expert weights in the CUTLASS layout, F32 activations [k, 1 or used, t] for at most 16 "
        "tokens and their ids, into F32 [n, used, t]");
  }
  if (!swiglu) {
    return CheckDense(node, {w, x}, ids);
  }
  const ggml_tensor* gate_scale = node->src[3];
  const ggml_tensor* up_scale = node->src[4];
  if (row0 != 0 || rows != 2 * n || !Vector(gate_scale, w->ne[1]) || !Vector(up_scale, w->ne[1]) ||
      !Aligned(gate_scale, sizeof(float)) || !Aligned(up_scale, sizeof(float))) {
    return Rejected("the SwiGLU form: gate rows then as many up rows, F32 scales [experts]");
  }
  return CheckDense(node, {w, x, gate_scale, up_scale}, ids);
}

std::expected<void, KernelFailure> CheckGdnConv(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGdnConv, 3); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* history = node->src[1];
  const ggml_tensor* weight = node->src[2];
  const std::int64_t channels = x->ne[0];
  const std::int64_t t = x->ne[1];
  const std::int64_t qk = JitllmOpInt(node, 0);
  const std::int64_t head = JitllmOpInt(node, 1);
  const float eps = JitllmOpFloat(node, 2);
  const float scale = JitllmOpFloat(node, 3);
  if ((!IsF32(x) && x->type != GGML_TYPE_BF16) || !Shaped(x, channels, t, 1) || !IsF32(weight) ||
      !Shaped(weight, 4, channels, 1) || !IsF32(history) ||
      ggml_nelements(history) != 3 * channels || !IsF32(node) || !Shaped(node, channels, t, 1)) {
    return Rejected(
        "F32 or BF16 rows [channels, t], a 4-tap weight [4, channels] and a history of 3");
  }
  if (head != 128 || qk < 0 || qk % head != 0 || qk > channels || channels % head != 0 ||
      t > 65535 || std::cmp_greater(ggml_nelements(node), kInt32Max) || !std::isfinite(eps) ||
      eps < 0.0f || !std::isfinite(scale)) {
    return Rejected("heads of 128 channels, normalized heads leading, within the kernel's grid");
  }
  for (const ggml_tensor* tensor : {x, history, weight, node}) {
    if (!Aligned(tensor, ggml_type_size(tensor->type))) {
      return Rejected("aligned operands");
    }
  }
  return CheckDense(node, {x, history, weight});
}

std::expected<void, KernelFailure> CheckGdnHistory(const ggml_tensor* node) {
  const bool with_history = node != nullptr && node->src[1] != nullptr;
  if (auto checked = CheckCustom(node, JitllmOp::kGdnHistory, with_history ? 2 : 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const std::int64_t taps = JitllmOpInt(node, 0);
  const std::int64_t channels = x->ne[0];
  if ((!IsF32(x) && x->type != GGML_TYPE_BF16) || !Shaped(x, channels, x->ne[1], 1) || taps <= 0 ||
      taps > 8 || (x->ne[1] < taps && !with_history) || !IsF32(node) ||
      !Shaped(node, taps * channels, 1, 1) || std::cmp_greater(ggml_nelements(node), kInt32Max) ||
      !Aligned(x, ggml_type_size(x->type)) || !Aligned(node, sizeof(float))) {
    return Rejected(
        "F32 or BF16 rows [channels, t] into the history of their last rows (fewer rows than "
        "taps with the old history)");
  }
  if (with_history) {
    const ggml_tensor* history = node->src[1];
    if (!IsF32(history) || ggml_nelements(history) != taps * channels ||
        !Aligned(history, sizeof(float))) {
      return Rejected("the old history is F32 [taps · channels]");
    }
    return CheckDense(node, {x, history});
  }
  return CheckDense(node, {x});
}

std::expected<void, KernelFailure> CheckGdnNormGate(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGdnNormGate, 3); !checked) {
    return checked;
  }
  const ggml_tensor* o = node->src[0];
  const ggml_tensor* weight = node->src[1];
  const ggml_tensor* z = node->src[2];
  const std::int64_t d = o->ne[0];
  const std::int64_t heads = o->ne[1];
  const std::int64_t t = o->ne[2];
  const float eps = JitllmOpEps(node);
  const bool quantized = node->type == GGML_TYPE_I8;
  const mxfp8::RowsLayout rows{.k = static_cast<std::uint64_t>(d * heads),
                               .rows = static_cast<std::uint64_t>(t)};
  if (!IsF32(o) || d != 128 || o->ne[3] != 1 || !Vector(weight, d) ||
      (!IsF32(z) && z->type != GGML_TYPE_BF16) || !Shaped(z, d * heads, t, 1) ||
      (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16 && !quantized) ||
      (quantized ? !Shaped(node, static_cast<std::int64_t>(rows.bytes()), 1, 1)
                 : !Shaped(node, d * heads, t, 1)) ||
      !std::isfinite(eps) || eps < 0.0f || std::cmp_greater(heads * t, kInt32Max)) {
    return Rejected(
        "F32 heads [128, heads, t], a weight [128] and an F32 or BF16 gate [128 · heads, t]");
  }
  for (const ggml_tensor* tensor : {o, weight, z}) {
    if (!Aligned(tensor, ggml_type_size(tensor->type))) {
      return Rejected("aligned operands");
    }
  }
  if (!Aligned(node, quantized ? 16 : ggml_type_size(node->type))) {
    return Rejected("an aligned output");
  }
  return CheckDense(node, {o, weight, z});
}

std::expected<void, KernelFailure> CheckGdnGates(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGdnGates, 4); !checked) {
    return checked;
  }
  const ggml_tensor* alpha = node->src[0];
  const ggml_tensor* beta = node->src[1];
  const ggml_tensor* dt_bias = node->src[2];
  const ggml_tensor* ssm_a = node->src[3];
  const std::int64_t rows = alpha->ne[1];
  if (!IsF32(alpha) || !Shaped(alpha, 48, rows, 1) || !IsF32(beta) || !Shaped(beta, 48, rows, 1) ||
      !Vector(dt_bias, 48) || !Vector(ssm_a, 48) || !IsF32(node) || !Shaped(node, 48, rows, 2) ||
      rows < 1 || rows > 16) {
    return Rejected("F32 alpha/beta [48, 1..16], dt_bias/ssm_a [48], output [48, rows, 2]");
  }
  for (const ggml_tensor* tensor : {node, alpha, beta, dt_bias, ssm_a}) {
    if (!Aligned(tensor, sizeof(float))) {
      return Rejected("aligned F32 operands and output");
    }
  }
  return CheckDense(node, {alpha, beta, dt_bias, ssm_a});
}

std::expected<void, KernelFailure> CheckGdnStep(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGdnStep, 6); !checked) {
    return checked;
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* g = node->src[3];
  const ggml_tensor* beta = node->src[4];
  const ggml_tensor* state = node->src[5];
  for (const ggml_tensor* tensor : {q, k, v, g, beta, state}) {
    if (!IsF32(tensor)) {
      return Rejected("jitllm.gdn.step takes F32 operands");
    }
  }
  if (!IsF32(node) || AnyEmpty({node, q, k, v, g, beta, state}) ||
      !AllSane({node, q, k, v, g, beta, state}) || !AllCurrent({q, k, v})) {
    return Rejected("jitllm.gdn.step on an empty, unmeasurable or stale tensor");
  }
  constexpr std::int64_t kS = 128;
  const std::int64_t heads = v->ne[1];
  const std::int64_t tokens = v->ne[2];
  if (v->ne[0] != kS || v->ne[3] != 1 || !ggml_are_same_shape(q, k) || q->ne[0] != kS ||
      q->ne[2] != tokens || q->ne[3] != 1 || q->ne[1] <= 0 || heads % q->ne[1] != 0 ||
      !Shaped(g, 1, heads, tokens) || !Shaped(beta, 1, heads, tokens) ||
      !Shaped(state, kS, kS, heads) || !Shaped(node, kS, heads, tokens) || tokens < 1 ||
      tokens > kGatedDeltaNetLanesTokens || heads > 65535) {
    return Rejected(
        "jitllm.gdn.step takes one sequence of 128-wide heads, a scalar gate and at most 16 "
        "tokens");
  }
  if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(v) || !ggml_are_same_stride(q, k) ||
      !detail::ElementStrides(q) || !detail::ElementStrides(v) ||
      std::cmp_greater(q->nb[2] / sizeof(float), kInt32Max) ||
      std::cmp_greater(v->nb[2] / sizeof(float), kInt32Max)) {
    return Rejected("jitllm.gdn.step needs contiguous rows of q, k and v");
  }
  for (const ggml_tensor* tensor : {q, k, v, g, beta, state, node}) {
    if (!Aligned(tensor, sizeof(float))) {
      return Rejected("jitllm.gdn.step operands at misaligned addresses");
    }
  }
  // The state is written in place: nothing else it reads may share its
  // bytes, nor the output.
  for (const ggml_tensor* tensor : {q, k, v, g, beta}) {
    if (!Disjoint(state, tensor, /*in_place=*/false) || !Disjoint(node, tensor, false)) {
      return Rejected("jitllm.gdn.step's state and output overlap an operand");
    }
  }
  return CheckDense(node, {g, beta, state});
}

bool GatedDeltaNetColumnsFits(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_GATED_DELTA_NET) {
    return false;
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* g = node->src[3];
  return q != nullptr && v != nullptr && g != nullptr && v->ne[0] == 128 && v->ne[3] == 1 &&
         q->ne[3] == 1 && g->ne[0] == 1 && node->op_params[0] == 1;
}

std::expected<void, KernelFailure> CheckGatedDeltaNetColumns(const ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNet(node); !checked) {
    return checked;
  }
  if (!GatedDeltaNetColumnsFits(node) || std::cmp_greater(node->src[2]->ne[2], kInt32Max) ||
      std::cmp_greater(node->src[2]->ne[1], 65535)) {
    return Rejected(
        "jitllm.gated_delta_net.columns takes one sequence of 128-wide heads, a scalar gate "
        "and no snapshots");
  }
  return {};
}

bool GatedDeltaNetLanesFits(const ggml_tensor* node) {
  return GatedDeltaNetColumnsFits(node) && node->src[2]->ne[2] > kGatedDeltaNetLanesTokens;
}

std::expected<void, KernelFailure> CheckGatedDeltaNetLanes(const ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNetColumns(node); !checked) {
    return checked;
  }
  for (const ggml_tensor* t : std::initializer_list<const ggml_tensor*>{
           node->src[0], node->src[1], node->src[2], node->src[5], node}) {
    if (!AlignedEverywhere(t, 16)) {
      return Rejected(
          "jitllm.gated_delta_net.lanes loads q, k, v and the state and stores the new state as "
          "float4s");
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckGemmBf16(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGemmBf16, 2); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* x = node->src[1];
  const std::int64_t k = weights->ne[0];
  const std::int64_t n = weights->ne[1];
  const std::int64_t t = x->ne[1];
  if (weights->type != GGML_TYPE_BF16 || x->type != GGML_TYPE_BF16 ||
      (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) || !Shaped(weights, k, n, 1) ||
      !Shaped(x, k, t, 1) || !Shaped(node, n, t, 1)) {
    return Rejected("BF16 weights [k, n] and activations [k, t] into F32 or BF16 [n, t]");
  }
  if (std::cmp_greater(k, kInt32Max) || std::cmp_greater(n, kInt32Max) ||
      std::cmp_greater(t, kInt32Max) || !Aligned(weights, 16) || !Aligned(x, 16) ||
      !Aligned(node, 16)) {
    return Rejected("cuBLAS's int extents and 16-byte aligned operands");
  }
  return CheckDense(node, {weights, x});
}

ggml_tensor* Mxfp8Quantize(ggml_context* context, ggml_tensor* x) {
  const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(x->ne[0]),
                                 .rows = static_cast<std::uint64_t>(x->ne[1])};
  return WithInts(
      Custom(context, GGML_TYPE_I8, {static_cast<std::int64_t>(layout.bytes()), 1, 1, 1}, {x},
             kTagMxfp8Quantize.data()),
      {x->ne[0], x->ne[1]});
}

ggml_tensor* Mxfp8Swizzle(ggml_context* context, ggml_tensor* scales) {
  const std::int64_t n = scales->ne[1];
  const std::int64_t k = scales->ne[0] * static_cast<std::int64_t>(mxfp8::kBlock);
  return WithInts(Custom(context, GGML_TYPE_I8,
                         {static_cast<std::int64_t>(mxfp8::SwizzledScaleBytes(
                              static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k))),
                          1, 1, 1},
                         {scales}, kTagMxfp8Swizzle.data()),
                  {n, k});
}

ggml_tensor* Mxfp8Gemm(ggml_context* context, ggml_tensor* a, ggml_tensor* codes,
                       ggml_tensor* scales, ggml_type type, std::int64_t t) {
  const std::int64_t k = codes->ne[0];
  const std::int64_t n = codes->ne[1];
  return WithInts(Custom(context, type, {n, t, 1, 1}, {a, codes, scales}, kTagMxfp8Gemm.data()),
                  {k, n, t});
}

namespace {

// Extents the MXFP8 operations take: k a multiple of 128 (whole scale atoms
// along k), each within the kernels' 32-bit grids.
bool Mxfp8Extents(std::int64_t k, std::int64_t rows) {
  return k > 0 && k % 128 == 0 && k <= (1 << 20) && rows > 0 &&
         std::cmp_less_equal(mxfp8::PaddedRows(static_cast<std::uint64_t>(rows)), kInt32Max) &&
         std::cmp_less_equal(
             mxfp8::PaddedRows(static_cast<std::uint64_t>(rows)) * static_cast<std::uint64_t>(k),
             kInt32Max * std::uint64_t{32});
}

}  // namespace

std::expected<void, KernelFailure> CheckMxfp8Quantize(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8Quantize, 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const std::int64_t k = JitllmOpInt(node, 0);
  const std::int64_t t = JitllmOpInt(node, 1);
  const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(k),
                                 .rows = static_cast<std::uint64_t>(t)};
  if (!Mxfp8Extents(k, t) || (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_BF16) ||
      !Shaped(x, k, t, 1) || x->nb[0] != ggml_type_size(x->type) || x->nb[1] % 16 != 0 ||
      x->nb[1] < ggml_row_size(x->type, k) || !Aligned(x, 16) || AnyEmpty({x}) || !AllSane({x}) ||
      !AllCurrent({x})) {
    return Rejected(
        "F32 or BF16 rows [k, t], k a multiple of 128, at a 16-byte aligned stride of a row or "
        "more");
  }
  if (node->type != GGML_TYPE_I8 ||
      !Shaped(node, static_cast<std::int64_t>(layout.bytes()), 1, 1) || !Aligned(node, 16) ||
      !Disjoint(node, x, false)) {
    return Rejected("an aligned I8 output of the quantized rows' layout, disjoint from x");
  }
  return CheckDense(node, {});
}

std::expected<void, KernelFailure> CheckMxfp8Swizzle(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8Swizzle, 1); !checked) {
    return checked;
  }
  const ggml_tensor* scales = node->src[0];
  const std::int64_t n = JitllmOpInt(node, 0);
  const std::int64_t k = JitllmOpInt(node, 1);
  if (!Mxfp8Extents(k, n) || !IsBytes(scales) ||
      !Shaped(scales, k / static_cast<std::int64_t>(mxfp8::kBlock), n, 1) ||
      node->type != GGML_TYPE_I8 ||
      !Shaped(node,
              static_cast<std::int64_t>(mxfp8::SwizzledScaleBytes(static_cast<std::uint64_t>(n),
                                                                  static_cast<std::uint64_t>(k))),
              1, 1) ||
      !Aligned(node, 16)) {
    return Rejected("I8 scales [k / 32, n] into the product's swizzled layout");
  }
  return CheckDense(node, {scales});
}

std::expected<void, KernelFailure> CheckMxfp8Gemm(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8Gemm, 3); !checked) {
    return checked;
  }
  const ggml_tensor* a = node->src[0];
  const ggml_tensor* codes = node->src[1];
  const ggml_tensor* scales = node->src[2];
  const std::int64_t k = JitllmOpInt(node, 0);
  const std::int64_t n = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  // The activations: a quantization of [k, t] (jitllm.mxfp8.quantize's, or
  // a fusion's that ends in one), I8 bytes of the rows' layout.
  const mxfp8::RowsLayout rows{.k = static_cast<std::uint64_t>(k),
                               .rows = static_cast<std::uint64_t>(t)};
  if (!Mxfp8Extents(k, t) || !Mxfp8Extents(k, n) || !IsBytes(a) ||
      !Shaped(a, static_cast<std::int64_t>(rows.bytes()), 1, 1) ||
      (JitllmOpOf(a) == JitllmOp::kMxfp8Quantize &&
       (JitllmOpInt(a, 0) != k || JitllmOpInt(a, 1) != t)) ||
      JitllmOpOf(scales) != JitllmOp::kMxfp8Swizzle || JitllmOpInt(scales, 0) != n ||
      JitllmOpInt(scales, 1) != k || !Aligned(a, 16) || !Aligned(scales, 16)) {
    return Rejected("the quantized activations [k, t] and the weight's swizzled scales [n, k]");
  }
  if (!IsBytes(codes) || !Shaped(codes, k, n, 1) || !Aligned(codes, 16)) {
    return Rejected("the weight's I8 codes [k, n], 16-byte aligned");
  }
  const std::size_t element = ggml_type_size(node->type);
  if ((node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) || !Shaped(node, n, t, 1) ||
      (static_cast<std::uint64_t>(n) * element) % 16 != 0 || !Aligned(node, 16)) {
    return Rejected("an F32 or BF16 output [n, t], its rows whole 16 bytes, 16-byte aligned");
  }
  return CheckDense(node, {a, codes, scales});
}

ggml_tensor* HcPrep(ggml_context* context, ggml_tensor* x, ggml_tensor* norm, ggml_tensor* inject,
                    ggml_tensor* out, ggml_tensor* logits, float eps) {
  const HcPrepLayout layout{.width = x->ne[0],
                            .hc = x->ne[1],
                            .t = x->ne[2],
                            .combine = out != nullptr,
                            .inject = inject != nullptr};
  std::array<ggml_tensor*, 5> args{x, norm};
  std::size_t n = 2;
  if (inject != nullptr) {
    args[n++] = inject;
  }
  if (out != nullptr) {
    args[n++] = out;
    args[n++] = logits;
  }
  std::array<ggml_tensor*, GGML_MAX_SRC> list{};
  std::copy_n(args.begin(), n, list.begin());
  ggml_tensor* node =
      ggml_custom_4d(context, GGML_TYPE_I8, static_cast<std::int64_t>(layout.bytes()), 1, 1, 1,
                     list.data(), static_cast<int>(n), &JitllmCustomTag, 1, kTagHcPrep.data());
  WithInts(node,
           {layout.width, layout.hc, layout.t, layout.combine ? 1 : 0, layout.inject ? 1 : 0});
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 20, &eps, sizeof(eps));
  return node;
}

ggml_tensor* HcLo(ggml_context* context, ggml_tensor* lo, std::int64_t hc) {
  return WithInts(
      Custom(context, GGML_TYPE_BF16, {lo->ne[0], lo->ne[1], 1, 1}, {lo}, kTagHcLo.data()), {hc});
}

std::uint64_t HcMixLayout::quantized_bytes() const {
  return mxfp8::RowsLayout{.k = static_cast<std::uint64_t>(width),
                           .rows = static_cast<std::uint64_t>(t)}
      .bytes();
}

ggml_tensor* HcMixBf16(ggml_context* context, ggml_tensor* normed, ggml_tensor* gate,
                       std::int64_t hc, bool mxfp8, bool bf16) {
  const std::int64_t width = normed->ne[0] / hc;
  const std::int64_t t = normed->ne[1];
  if (!mxfp8) {
    return WithInts(
        Custom(context, GGML_TYPE_F32, {width, t, 1, 1}, {normed, gate}, kTagHcMixBf16.data()),
        {hc, width, t, 0, 0});
  }
  const HcMixLayout layout{.width = width, .t = t, .bf16 = bf16};
  return WithInts(
      Custom(context, GGML_TYPE_I8, {static_cast<std::int64_t>(layout.bytes()), 1, 1, 1},
             {normed, gate}, kTagHcMixBf16.data()),
      {hc, width, t, 1, bf16 ? 1 : 0});
}

namespace {

HcPrepLayout HcPrepLayoutOf(const ggml_tensor* node) {
  return {.width = JitllmOpInt(node, 0),
          .hc = JitllmOpInt(node, 1),
          .t = JitllmOpInt(node, 2),
          .combine = JitllmOpInt(node, 3) == 1,
          .inject = JitllmOpInt(node, 4) == 1};
}

// Packed rows [n0, n1] of `type`, 16-byte aligned: views of a blob too.
bool Rows(const ggml_tensor* t, ggml_type type, std::int64_t n0, std::int64_t n1) {
  return t != nullptr && t->type == type && Shaped(t, n0, n1, 1) && Packed(t) && Aligned(t, 16) &&
         !AnyEmpty({t}) && AllSane({t}) && AllCurrent({t});
}

}  // namespace

std::expected<void, KernelFailure> CheckHcPrep(const ggml_tensor* node) {
  const HcPrepLayout l = HcPrepLayoutOf(node);
  const int arity = 2 + (l.inject ? 1 : 0) + (l.combine ? 2 : 0);
  if (auto checked = CheckCustom(node, JitllmOp::kHcPrep, arity); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* norm = node->src[1];
  const ggml_tensor* inject = l.inject ? node->src[2] : nullptr;
  const ggml_tensor* out = l.combine ? node->src[l.inject ? 3 : 2] : nullptr;
  const ggml_tensor* logits = l.combine ? node->src[l.inject ? 4 : 3] : nullptr;
  const float eps = JitllmOpFloat(node, 5);
  const std::int64_t wide = l.width * l.hc;
  if (l.width <= 0 || l.width % 8 != 0 || l.width > 4096 || l.hc <= 0 || l.hc > 8 || l.t <= 0 ||
      std::cmp_greater(wide, 1 << 16) || std::cmp_greater(l.t, kInt32Max) ||
      (JitllmOpInt(node, 3) != 0 && !l.combine) || (JitllmOpInt(node, 4) != 0 && !l.inject) ||
      !std::isfinite(eps) || eps < 0.0f) {
    return Rejected(
        "streams of whole 8 values, at most 4,096 of them (a thread a float4 column), at most 8 "
        "streams, a finite eps");
  }
  if (!IsF32(x) || x->ne[0] != l.width || x->ne[1] != l.hc || x->ne[2] != l.t || x->ne[3] != 1 ||
      !Packed(x) || !Aligned(x, 16) || !Vector(norm, wide) || !Aligned(norm, 16) ||
      (inject != nullptr && !Rows(inject, GGML_TYPE_BF16, wide, l.hc)) ||
      (out != nullptr && !Rows(out, GGML_TYPE_F32, l.width, l.t)) ||
      (logits != nullptr &&
       (!IsF32(logits) || !Shaped(logits, l.hc, l.t, 1) || logits->nb[0] != sizeof(float) ||
        logits->nb[1] != static_cast<std::size_t>(l.hc) * sizeof(float) ||
        !Aligned(logits, sizeof(float))))) {
    return Rejected(
        "F32 streams [width, hc, t], a norm weight [width · hc], a BF16 inject weight [width · "
        "hc, hc], an F32 output [width, t] and logits [hc, t], packed and aligned");
  }
  if (node->type != GGML_TYPE_I8 || !Shaped(node, static_cast<std::int64_t>(l.bytes()), 1, 1) ||
      !Aligned(node, 256)) {
    return Rejected("a 256-byte aligned I8 blob of the layout's bytes");
  }
  for (const ggml_tensor* t : {x, norm, inject, out, logits}) {
    if (t != nullptr && !Disjoint(node, t, false)) {
      return Rejected("an output overlapping an operand");
    }
  }
  return CheckDense(node, {x, norm});
}

std::expected<void, KernelFailure> CheckHcLo(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcLo, 1); !checked) {
    return checked;
  }
  const ggml_tensor* lo = node->src[0];
  const std::int64_t hc = JitllmOpInt(node, 0);
  if (!IsF32(lo) || lo->ne[2] != 1 || lo->ne[3] != 1 || node->type != GGML_TYPE_BF16 ||
      !ggml_are_same_shape(node, lo) || hc <= 0 || hc > 8 ||
      std::cmp_greater(ggml_nelements(lo), kInt32Max) || !Aligned(lo, sizeof(float)) ||
      !Aligned(node, 2)) {
    return Rejected("F32 [rank, t] into BF16 of its shape, at most 8 streams");
  }
  return CheckDense(node, {lo});
}

std::expected<void, KernelFailure> CheckHcMixBf16(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcMixBf16, 2); !checked) {
    return checked;
  }
  const ggml_tensor* normed = node->src[0];
  const ggml_tensor* gate = node->src[1];
  const std::int64_t hc = JitllmOpInt(node, 0);
  const std::int64_t width = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  const bool blob = JitllmOpInt(node, 3) == 1;
  const HcMixLayout layout{.width = width, .t = t, .bf16 = JitllmOpInt(node, 4) == 1};
  if (hc <= 0 || hc > 8 || width <= 0 || width % 8 != 0 || t <= 0 ||
      (JitllmOpInt(node, 3) != 0 && !blob) || (JitllmOpInt(node, 4) != 0 && !layout.bf16) ||
      (layout.bf16 && !blob) || (blob && !Mxfp8Extents(width, t)) ||
      !Rows(normed, GGML_TYPE_BF16, width * hc, t) || !Rows(gate, GGML_TYPE_BF16, width * hc, t) ||
      !Aligned(node, blob ? 256 : 16) || std::cmp_greater(width * t / 8, kInt32Max)) {
    return Rejected(
        "BF16 normalized streams and logits [width · hc, t], width whole 8 values (128 for the "
        "quantization)");
  }
  if (blob ? node->type != GGML_TYPE_I8 ||
                 !Shaped(node, static_cast<std::int64_t>(layout.bytes()), 1, 1)
           : !IsF32(node) || !Shaped(node, width, t, 1)) {
    return Rejected("an F32 output [width, t], or the blob of its layout");
  }
  return CheckDense(node, {normed, gate});
}

ggml_tensor* QsaPrep(ggml_context* context, ggml_tensor* x, ggml_tensor* weight,
                     ggml_tensor* positions, std::int64_t d, std::int64_t heads,
                     std::int64_t stride, float eps, float theta_scale) {
  ggml_tensor* node = WithInts(Custom(context, GGML_TYPE_F32, {d, heads, x->ne[1], 1},
                                      {x, weight, positions}, kTagQsaPrep.data()),
                               {d, heads, stride, x->ne[1]});
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 16, &eps, sizeof(eps));
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 20, &theta_scale,
              sizeof(theta_scale));
  return node;
}

ggml_tensor* QsaGateQuantize(ggml_context* context, ggml_tensor* attn, ggml_tensor* q_full,
                             std::int64_t d) {
  const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(attn->ne[0]),
                                 .rows = static_cast<std::uint64_t>(attn->ne[1])};
  return WithInts(
      Custom(context, GGML_TYPE_I8, {static_cast<std::int64_t>(layout.bytes()), 1, 1, 1},
             {attn, q_full}, kTagQsaGateQuantize.data()),
      {d, attn->ne[0] / d, attn->ne[1]});
}

std::expected<void, KernelFailure> CheckQsaPrep(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQsaPrep, 3); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* weight = node->src[1];
  const ggml_tensor* positions = node->src[2];
  const std::int64_t d = JitllmOpInt(node, 0);
  const std::int64_t heads = JitllmOpInt(node, 1);
  const std::int64_t stride = JitllmOpInt(node, 2);
  const std::int64_t t = JitllmOpInt(node, 3);
  const float eps = JitllmOpFloat(node, 4);
  const float theta_scale = JitllmOpFloat(node, 5);
  if (d < 64 || d % 32 != 0 || d > 512 || heads <= 0 || stride < d || t <= 0 ||
      std::cmp_greater(heads * t, kInt32Max) || !std::isfinite(eps) || eps < 0.0f ||
      !std::isfinite(theta_scale)) {
    return Rejected("heads of 64 to 512 values (whole warps) at least a head apart");
  }
  if (!IsF32(x) || x->ne[1] != t || x->ne[2] != 1 || x->ne[3] != 1 || !Packed(x) ||
      x->ne[0] < ((heads - 1) * stride) + d || !Aligned(x, sizeof(float)) || !Vector(weight, d) ||
      positions->type != GGML_TYPE_I32 || ggml_nelements(positions) < t || !Packed(positions) ||
      !IsF32(node) || !Shaped(node, d, heads, t)) {
    return Rejected(
        "F32 rows [n, t] holding the heads, a weight [d], I32 positions, into [d, "
        "heads, t]");
  }
  return CheckDense(node, {x, weight, positions});
}

std::expected<void, KernelFailure> CheckQsaGateQuantize(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQsaGateQuantize, 2); !checked) {
    return checked;
  }
  const ggml_tensor* attn = node->src[0];
  const ggml_tensor* q_full = node->src[1];
  const std::int64_t d = JitllmOpInt(node, 0);
  const std::int64_t heads = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  const mxfp8::RowsLayout layout{.k = static_cast<std::uint64_t>(d * heads),
                                 .rows = static_cast<std::uint64_t>(t)};
  if (d <= 0 || d % 32 != 0 || heads <= 0 || !Mxfp8Extents(d * heads, t) || !IsF32(attn) ||
      !Shaped(attn, d * heads, t, 1) || !IsF32(q_full) || !Shaped(q_full, 2 * d * heads, t, 1) ||
      !Aligned(attn, 16) || !Aligned(q_full, 16) || node->type != GGML_TYPE_I8 ||
      !Shaped(node, static_cast<std::int64_t>(layout.bytes()), 1, 1) || !Aligned(node, 16)) {
    return Rejected(
        "F32 attention [d · heads, t] and query rows [2 · d · heads, t] into MXFP8 rows");
  }
  return CheckDense(node, {attn, q_full});
}

ggml_tensor* QsaPool(ggml_context* context, ggml_tensor* raw, ggml_tensor* blocks,
                     ggml_tensor* weight, ggml_tensor* positions, std::int64_t ratio, float eps,
                     float theta_scale) {
  ggml_tensor* node = WithInts(Custom(context, GGML_TYPE_I32, {1, 1, 1, 1},
                                      {raw, blocks, weight, positions}, kTagQsaPool.data()),
                               {raw->ne[0], ratio, positions->ne[0] / 4});
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 12, &eps, sizeof(eps));
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 16, &theta_scale,
              sizeof(theta_scale));
  return node;
}

ggml_tensor* QsaTopK(ggml_context* context, ggml_tensor* q, ggml_tensor* blocks,
                     ggml_tensor* positions, ggml_tensor* pool, std::int64_t n_blocks,
                     std::int64_t width, std::int64_t ratio) {
  const std::int64_t t = q->ne[2];
  return WithInts(Custom(context, GGML_TYPE_I32, {QsaTopKRow(width), t, 1, 1},
                         {q, blocks, positions, pool}, kTagQsaTopK.data()),
                  {n_blocks, width, ratio, t});
}

ggml_tensor* QsaAttn(ggml_context* context, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                     ggml_tensor* cells, float scale) {
  const std::int64_t d = q->ne[0];
  const std::int64_t heads = q->ne[1];
  const std::int64_t t = q->ne[2];
  ggml_tensor* node = WithInts(
      Custom(context, GGML_TYPE_F32, {d * heads, t, 1, 1}, {q, k, v, cells}, kTagQsaAttn.data()),
      {heads, d > 0 ? k->ne[0] / d : 0, t, cells->ne[0]});
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 16, &scale, sizeof(scale));
  return node;
}

namespace {

// The selection's and the pool's positions: I32 [4 · t], packed.
bool Positions(const ggml_tensor* positions, std::int64_t t) {
  return positions->type == GGML_TYPE_I32 && Shaped(positions, 4 * t, 1, 1) && Packed(positions) &&
         Aligned(positions, 4);
}

// A BF16 table of `d`-value rows, packed.
bool Bf16Rows(const ggml_tensor* t, std::int64_t d) {
  return t->type == GGML_TYPE_BF16 && t->ne[0] == d && t->ne[2] == 1 && t->ne[3] == 1 &&
         Packed(t) && Aligned(t, 16);
}

}  // namespace

std::expected<void, KernelFailure> CheckQsaPool(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQsaPool, 4); !checked) {
    return checked;
  }
  const ggml_tensor* raw = node->src[0];
  const ggml_tensor* blocks = node->src[1];
  const ggml_tensor* weight = node->src[2];
  const ggml_tensor* positions = node->src[3];
  const std::int64_t d = JitllmOpInt(node, 0);
  const std::int64_t ratio = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  const float eps = JitllmOpFloat(node, 3);
  const float theta_scale = JitllmOpFloat(node, 4);
  if (d < 64 || d % 32 != 0 || d > 512 || ratio < 1 || ratio > 32 || t < 1 || !std::isfinite(eps) ||
      eps < 0.0f || !std::isfinite(theta_scale)) {
    return Rejected("keys of 64 to 512 values (whole warps), blocks of 1 to 32 cells");
  }
  if (!IsF32(raw) || raw->ne[0] != d || raw->ne[2] != 1 || raw->ne[3] != 1 || !Packed(raw) ||
      !Aligned(raw, 4) || !Bf16Rows(blocks, d) || blocks->ne[1] < raw->ne[1] / ratio ||
      !Vector(weight, d) || !Aligned(weight, 4) || !Positions(positions, t) ||
      node->type != GGML_TYPE_I32 || !Shaped(node, 1, 1, 1)) {
    return Rejected(
        "F32 raw keys [d, cells], BF16 block keys [d, cells / ratio], an F32 weight [d] and "
        "I32 positions [4 · t], into an I32 marker");
  }
  if (std::cmp_greater(raw->ne[1], kInt32Max) || std::cmp_greater(blocks->ne[1], kInt32Max)) {
    return Rejected("caches within the kernel's 32-bit cells");
  }
  if (AnyEmpty({node, raw, blocks, weight, positions}) ||
      !AllSane({node, raw, blocks, weight, positions}) ||
      !AllCurrent({node, raw, blocks, weight, positions})) {
    return Rejected("jitllm.qsa.pool on an empty, unmeasurable or stale tensor");
  }
  // The block keys are written in place: nothing the kernel reads shares
  // their bytes, and the marker is apart from everything.
  for (const ggml_tensor* tensor : {raw, weight, positions}) {
    if (!Disjoint(blocks, tensor, false) || !Disjoint(node, tensor, false)) {
      return Rejected("jitllm.qsa.pool's block keys or marker overlap an operand");
    }
  }
  if (!Disjoint(node, blocks, false)) {
    return Rejected("jitllm.qsa.pool's marker overlaps the block keys");
  }
  return {};
}

std::uint64_t PlanQsaTopK(const ggml_tensor* node) {
  return QsaTopKLayout{.t = JitllmOpInt(node, 3),
                       .n_blocks = JitllmOpInt(node, 0),
                       .width = JitllmOpInt(node, 1),
                       .ratio = JitllmOpInt(node, 2)}
      .bytes();
}

std::expected<void, KernelFailure> CheckQsaTopK(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQsaTopK, 4); !checked) {
    return checked;
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* blocks = node->src[1];
  const ggml_tensor* positions = node->src[2];
  const ggml_tensor* pool = node->src[3];
  const std::int64_t n_blocks = JitllmOpInt(node, 0);
  const std::int64_t width = JitllmOpInt(node, 1);
  const std::int64_t ratio = JitllmOpInt(node, 2);
  const std::int64_t t = JitllmOpInt(node, 3);
  const QsaTopKLayout layout{.t = t, .n_blocks = n_blocks, .width = width, .ratio = ratio};
  if (n_blocks < 1 || layout.tiles() > kQsaTopKMaxTiles || width < 1 || ratio < 1 || ratio > 32 ||
      t < 1 || t > 65535 || width > n_blocks * ratio ||
      layout.tiles() * layout.candidates() > kQsaTopKCandidates) {
    return Rejected(
        "at most 8 tiles of 8,192 blocks, a width within the blocks' cells, at most 65,535 "
        "tokens");
  }
  if (!IsF32(q) || !Shaped(q, kQsaIndexDim, kQsaIndexHeads, t) || !Packed(q) || !Aligned(q, 16) ||
      !Bf16Rows(blocks, kQsaIndexDim) || blocks->ne[1] < n_blocks || !Positions(positions, t) ||
      JitllmOpOf(pool) != JitllmOp::kQsaPool || pool->src[1] != blocks ||
      pool->src[3] != positions || node->type != GGML_TYPE_I32 ||
      !Shaped(node, QsaTopKRow(width), t, 1) || !Packed(node) || !Aligned(node, 16)) {
    return Rejected(
        "F32 queries [128, 4, t], the BF16 block keys its jitllm.qsa.pool node writes and its "
        "positions, into I32 cells [row, t]");
  }
  if (AnyEmpty({node, q, blocks, positions}) || !AllSane({node, q, blocks, positions}) ||
      !AllCurrent({node, q, blocks, positions})) {
    return Rejected("jitllm.qsa.topk on an empty, unmeasurable or stale tensor");
  }
  for (const ggml_tensor* tensor : {q, blocks, positions, pool}) {
    if (!Disjoint(node, tensor, false)) {
      return Rejected("jitllm.qsa.topk's cells overlap an operand");
    }
  }
  return {};
}

std::uint64_t PlanQsaAttn(const ggml_tensor* node) {
  const std::int64_t heads = JitllmOpInt(node, 0);
  const std::int64_t kv_heads = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  const std::int64_t row = JitllmOpInt(node, 3);
  if (kv_heads < 1 || heads < 1 || t < 1 || row < 16) {
    return 0;
  }
  const std::int64_t shares = QsaAttnShares(t, kv_heads, row);
  if (shares <= 1) {
    return 0;
  }
  return QsaTopKLayout::Align(static_cast<std::uint64_t>(t * kv_heads * shares) *
                              QsaAttnPartial(heads / kv_heads));
}

std::expected<void, KernelFailure> CheckQsaAttn(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQsaAttn, 4); !checked) {
    return checked;
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* cells = node->src[3];
  const std::int64_t heads = JitllmOpInt(node, 0);
  const std::int64_t kv_heads = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  const std::int64_t row = JitllmOpInt(node, 3);
  const float scale = JitllmOpFloat(node, 4);
  constexpr std::int64_t d = kQsaAttnHead;
  if (heads < 1 || kv_heads < 1 || heads % kv_heads != 0 || heads / kv_heads > 16 || t < 1 ||
      row < 16 || row % 16 != 0 || !std::isfinite(scale) ||
      std::cmp_greater(t * kv_heads * QsaAttnShares(t, kv_heads, row), kInt32Max)) {
    return Rejected(
        "whole groups of at most 16 query heads a KV head, rows of whole 16-cell gathers");
  }
  if (!IsF32(q) || !Shaped(q, d, heads, t) || !Packed(q) || !Aligned(q, 16)) {
    return Rejected("F32 queries [256, heads, t], packed");
  }
  for (const ggml_tensor* cache : {k, v}) {
    if (cache->type != GGML_TYPE_F16 || cache->ne[0] != d * kv_heads || cache->ne[1] < 1 ||
        cache->ne[2] != 1 || cache->ne[3] != 1 || cache->nb[0] != 2 ||
        cache->nb[1] < static_cast<std::size_t>(d * kv_heads) * 2 || cache->nb[1] % 16 != 0 ||
        !Aligned(cache, 16) || std::cmp_greater(cache->ne[1], kInt32Max)) {
      return Rejected("F16 caches [256 · kv heads, cells], rows 16-byte aligned");
    }
  }
  if (k->nb[1] != v->nb[1]) {
    return Rejected("F16 key and value caches with equal row strides");
  }
  if (k->ne[1] != v->ne[1] || cells->type != GGML_TYPE_I32 || !Shaped(cells, row, t, 1) ||
      !Packed(cells) || !Aligned(cells, 16) || !IsF32(node) || !Shaped(node, d * heads, t, 1) ||
      !Aligned(node, 8)) {
    return Rejected("I32 cells [row, t] into an F32 output [256 · heads, t]");
  }
  if (AnyEmpty({node, q, k, v, cells}) || !AllSane({node, q, k, v, cells}) ||
      !AllCurrent({node, q, k, v, cells})) {
    return Rejected("jitllm.qsa.attn on an empty, unmeasurable or stale tensor");
  }
  if (!Disjoint(node, k, false) || !Disjoint(node, v, false)) {
    return Rejected("jitllm.qsa.attn's output overlaps a cache");
  }
  return CheckDense(node, {q, cells});
}

ggml_tensor* MoeRouter(ggml_context* context, ggml_tensor* logits, ggml_tensor* x,
                       ggml_tensor* gate_row, std::int64_t used) {
  const MoeRouterLayout layout{.used = used, .t = logits->ne[1]};
  return WithInts(Custom(context, GGML_TYPE_I32, {layout.ints(), 1, 1, 1}, {logits, x, gate_row},
                         kTagMoeRouter.data()),
                  {logits->ne[0], used, logits->ne[1], x->ne[0]});
}

std::expected<void, KernelFailure> CheckMoeRouter(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeRouter, 3); !checked) {
    return checked;
  }
  const std::int64_t experts = JitllmOpInt(node, 0);
  const std::int64_t used = JitllmOpInt(node, 1);
  const std::int64_t t = JitllmOpInt(node, 2);
  const std::int64_t width = JitllmOpInt(node, 3);
  const MoeRouterLayout layout{.used = used, .t = t};
  const ggml_tensor* logits = node->src[0];
  const ggml_tensor* x = node->src[1];
  const ggml_tensor* gate_row = node->src[2];
  if (experts <= 0 || experts % 32 != 0 || experts > 1024 || used <= 0 || used > 32 ||
      used > experts || t <= 0 || width <= 0 || width % 4 != 0 ||
      std::cmp_greater(layout.ints(), kInt32Max)) {
    return Rejected("32 to 1,024 experts (whole warps), at most 32 used, whole float4 rows");
  }
  // The gate row's four values a load: 8 bytes in BF16, 16 in F32.
  const bool bf16_gate = gate_row->type == GGML_TYPE_BF16;
  if (!IsF32(logits) || !Shaped(logits, experts, t, 1) || !IsF32(x) || !Shaped(x, width, t, 1) ||
      (!bf16_gate && !IsF32(gate_row)) || ggml_nelements(gate_row) != width || !Aligned(x, 16) ||
      !Aligned(gate_row, bf16_gate ? 8 : 16) || !Aligned(logits, sizeof(float)) ||
      node->type != GGML_TYPE_I32 || !Shaped(node, layout.ints(), 1, 1) || !Aligned(node, 16)) {
    return Rejected(
        "F32 logits [experts, t] and x [width, t], a BF16 or F32 gate row [width], into the "
        "routing's I32 blob");
  }
  return CheckDense(node, {logits, x, gate_row});
}

// ------------------------------------------------ DeepSeek V4's fast plan

namespace {

constexpr std::int64_t kQ8Block = 32;        // QK8_1
constexpr std::int64_t kQ8BlockBytes = 36;   // sizeof(block_q8_1)
constexpr std::int64_t kQ8RowPadding = 512;  // MATRIX_ROW_PADDING
constexpr std::int64_t kDsv4HcStreams = 4;   // the kernels' hc
constexpr std::int64_t kDsv4HcMixes = (2 + kDsv4HcStreams) * kDsv4HcStreams;

std::int64_t Q8Padded(std::int64_t k) {
  return (k + kQ8RowPadding - 1) / kQ8RowPadding * kQ8RowPadding;
}

ggml_tensor* WithFloat(ggml_tensor* node, int index, float value) {
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset +
                  (static_cast<std::size_t>(index) * sizeof(value)),
              &value, sizeof(value));
  return node;
}

// A quantized weight tensor jitllm.vecq reads: rows packed, experts (or
// groups) at a stride of whole blocks.
bool VecQWeights(const ggml_tensor* w) {
  if (w == nullptr || !VecQType(w->type) || !Bound(w) || AnyEmpty({w}) || !AllSane({w}) ||
      !AllCurrent({w}) || w->ne[3] != 1) {
    return false;
  }
  const auto type_size = static_cast<std::uint64_t>(ggml_type_size(w->type));
  const std::int64_t block = ggml_blck_size(w->type);
  return w->ne[0] % block == 0 && w->nb[0] == type_size &&
         w->nb[1] == ggml_row_size(w->type, w->ne[0]) && w->nb[2] % type_size == 0 &&
         w->nb[2] >= w->nb[1] * static_cast<std::uint64_t>(w->ne[1]) &&
         std::cmp_less_equal(w->nb[2] / type_size, kInt32Max) &&
         std::cmp_less_equal(w->ne[1] * (w->ne[0] / block), kInt32Max) &&
         // The kernel's block offsets (expert · stride + row · row stride)
         // are 32-bit.
         std::cmp_less_equal(static_cast<std::uint64_t>(w->ne[2]) * (w->nb[2] / type_size),
                             kInt32Max) &&
         Aligned(w, 4);
}

}  // namespace

bool VecQType(ggml_type type) {
  switch (type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ3_XXS:
    // Qwen3.8's GGUF quantizations' (the kernel's generic per-token path).
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
      return true;
    default:
      return false;
  }
}

std::int64_t Q8Bytes(std::int64_t k, std::int64_t rows) {
  return Q8Padded(k) / kQ8Block * kQ8BlockBytes * rows;
}

ggml_tensor* QuantizeQ8(ggml_context* context, ggml_tensor* x) {
  const std::int64_t rows = x->ne[1] * x->ne[2];
  return WithInts(
      Custom(context, GGML_TYPE_I8, {Q8Bytes(x->ne[0], rows), 1, 1, 1}, {x}, kTagQuantizeQ8.data()),
      {x->ne[0], rows});
}

ggml_tensor* VecQ(ggml_context* context, ggml_tensor* weights, ggml_tensor* q8, ggml_tensor* ids,
                  std::int64_t tokens, bool per_slot, ggml_tensor* gate, VecQGlu glu, float limit) {
  const std::int64_t n = weights->ne[1];
  const std::int64_t groups = weights->ne[2];
  ggml_tensor* node = nullptr;
  if (ids != nullptr) {
    node = Custom(context, GGML_TYPE_F32, {n, ids->ne[0], tokens, 1}, {weights, q8, ids, gate},
                  kTagVecQ.data());
  } else if (groups > 1) {
    node = Custom(context, GGML_TYPE_F32, {n, groups, tokens, 1}, {weights, q8, gate},
                  kTagVecQ.data());
  } else {
    node = Custom(context, GGML_TYPE_F32, {n, tokens, 1, 1}, {weights, q8, gate}, kTagVecQ.data());
  }
  WithInts(node, {tokens, per_slot ? 1 : 0, static_cast<std::int64_t>(glu)});
  return WithFloat(node, 3, limit);
}

namespace {
constexpr int kVecQOneTokenParam = 4;  // after tokens, per_slot, glu and the limit
}  // namespace

void SetVecQOneToken(ggml_tensor* node) {
  const std::int32_t one = 1;
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset +
                  (static_cast<std::size_t>(kVecQOneTokenParam) * sizeof(one)),
              &one, sizeof(one));
}

bool VecQOneToken(const ggml_tensor* node) { return JitllmOpInt(node, kVecQOneTokenParam) == 1; }

ggml_tensor* Dsv4Route(ggml_context* context, ggml_tensor* logits, ggml_tensor* bias,
                       ggml_tensor* table, ggml_tensor* tokens, std::int64_t used, bool norm,
                       float clamp, float scale) {
  ggml_tensor* node = table != nullptr
                          ? Custom(context, GGML_TYPE_I32, {2 * used, logits->ne[1], 1, 1},
                                   {logits, table, tokens}, kTagDsv4Route.data())
                          : Custom(context, GGML_TYPE_I32, {2 * used, logits->ne[1], 1, 1},
                                   {logits, bias}, kTagDsv4Route.data());
  WithInts(node, {used, norm ? 1 : 0});
  WithFloat(node, 2, clamp);
  return WithFloat(node, 3, scale);
}

ggml_tensor* Dsv4Combine(ggml_context* context, ggml_tensor* down, ggml_tensor* route,
                         ggml_tensor* shared) {
  return Custom(context, GGML_TYPE_F32, {down->ne[0], down->ne[2], 1, 1}, {down, route, shared},
                kTagDsv4Combine.data());
}

ggml_tensor* Dsv4OrderedReduce(ggml_context* context, ggml_tensor* down, ggml_tensor* weights) {
  return Custom(context, GGML_TYPE_F32, {down->ne[0], down->ne[2], 1, 1}, {down, weights},
                kTagDsv4WeightedReduce.data());
}

ggml_tensor* Dsv4QHead(ggml_context* context, ggml_tensor* x, ggml_tensor* positions,
                       const Dsv4QHeadParams& params, ggml_type type) {
  static_assert(sizeof(params) == 32 && kEpsOffset + sizeof(params) <= GGML_MAX_OP_PARAMS);
  ggml_tensor* node = Custom(context, type, {x->ne[0], x->ne[1], x->ne[2], x->ne[3]},
                             {x, positions}, kTagDsv4QHead.data());
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset, &params, sizeof(params));
  return node;
}

ggml_tensor* Dsv4F16Copy(ggml_context* context, ggml_tensor* x) {
  return Custom(context, GGML_TYPE_F16, {x->ne[0], x->ne[1], 1, 1}, {x}, kTagDsv4F16Copy.data());
}

ggml_tensor* Dsv4HcNormF16(ggml_context* context, ggml_tensor* flat, float eps, ggml_type type) {
  return WithEps(
      Custom(context, type, {flat->ne[0], flat->ne[1], 1, 1}, {flat}, kTagDsv4HcNormF16.data()),
      eps);
}

ggml_tensor* Dsv4OutA(ggml_context* context, ggml_tensor* weights, ggml_tensor* heads,
                      ggml_tensor* positions, const Dsv4OutAParams& params) {
  static_assert(sizeof(params) == 28 && kEpsOffset + sizeof(params) <= GGML_MAX_OP_PARAMS);
  // The core stores whole 16-row tiles: the output holds the rows rounded up.
  ggml_tensor* node = Custom(context, GGML_TYPE_F32, {8192, Dsv4OutARows(heads->ne[2]), 1, 1},
                             {weights, heads, positions}, kTagDsv4OutA.data());
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset, &params, sizeof(params));
  return node;
}

bool Dsv4HcMixWeightType(ggml_type type) {
  return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

ggml_tensor* Dsv4HcMix(ggml_context* context, ggml_tensor* x, ggml_tensor* fn) {
  return Custom(context, GGML_TYPE_F32, {fn->ne[1] + 1, kDsv4HcChunks, x->ne[2], 1}, {x, fn},
                kTagDsv4HcMix.data());
}

ggml_tensor* Dsv4HcPre(ggml_context* context, ggml_tensor* partials, ggml_tensor* x,
                       ggml_tensor* scale, ggml_tensor* base, ggml_tensor* norm, float rms_eps,
                       float hc_eps, std::int32_t iterations) {
  ggml_tensor* node = Custom(context, GGML_TYPE_F32, {(x->ne[0] + kDsv4HcTail) * x->ne[2], 1, 1, 1},
                             {partials, x, scale, base, norm}, kTagDsv4HcPre.data());
  WithFloat(node, 0, rms_eps);
  WithFloat(node, 1, hc_eps);
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 8, &iterations,
              sizeof(iterations));
  return node;
}

ggml_tensor* Dsv4Compress(ggml_context* context, ggml_tensor* state_kv, ggml_tensor* state_score,
                          ggml_tensor* kv, ggml_tensor* score, ggml_tensor* read_idxs,
                          std::int64_t ratio, bool overlap) {
  const std::int64_t head = overlap ? state_kv->ne[0] / 2 : state_kv->ne[0];
  const std::int64_t per_block = overlap ? 2 * ratio : ratio;
  return WithInts(Custom(context, GGML_TYPE_F32, {head, 1, read_idxs->ne[0] / per_block, 1},
                         {state_kv, state_score, kv, score, read_idxs}, kTagDsv4Compress.data()),
                  {ratio, overlap ? 1 : 0});
}

std::expected<void, KernelFailure> CheckDsv4Compress(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kDsv4Compress, 5); !checked) {
    return checked;
  }
  const ggml_tensor* skv = node->src[0];
  const ggml_tensor* ssc = node->src[1];
  const ggml_tensor* kv = node->src[2];
  const ggml_tensor* score = node->src[3];
  const ggml_tensor* read = node->src[4];
  const std::int64_t ratio = JitllmOpInt(node, 0);
  const std::int64_t overlap = JitllmOpInt(node, 1);
  const std::int64_t channels = skv->ne[0];
  const std::int64_t head = overlap != 0 ? channels / 2 : channels;
  const std::int64_t per_block = overlap != 0 ? 2 * ratio : ratio;
  const auto rows = [&](const ggml_tensor* t) {
    return IsF32(t) && t->ne[0] == channels && t->ne[2] == 1 && t->ne[3] == 1 &&
           t->nb[0] == sizeof(float) && t->nb[1] % sizeof(float) == 0 && Aligned(t, 4) &&
           AllSane({t}) && !AnyEmpty({t}) && AllCurrent({t}) && Disjoint(node, t, false) &&
           std::cmp_less_equal(t->nb[1] / sizeof(float) * static_cast<std::uint64_t>(t->ne[1]),
                               kInt32Max);
  };
  if (ratio < 1 || ratio > 1024 || (overlap != 0 && overlap != 1) || channels % 2 != 0 ||
      !rows(skv) || !rows(ssc) || !rows(kv) || !rows(score) || ssc->ne[1] != skv->ne[1] ||
      score->ne[1] != kv->ne[1] || read->type != GGML_TYPE_I32 || !Packed(read) || !Bound(read) ||
      read->ne[0] % per_block != 0 || !Shaped(read, read->ne[0], 1, 1) || !IsF32(node) ||
      !Packed(node) || !Shaped(node, head, 1, read->ne[0] / per_block) ||
      read->ne[0] / per_block > 65535 || !Disjoint(node, read, false) || !AllCurrent({node})) {
    return Rejected(
        "a compressor's state and rows F32 [channels, rows], its read indices I32, "
        "into F32 [head, 1, blocks]");
  }
  return {};
}

std::expected<void, KernelFailure> CheckQuantizeQ8(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kQuantizeQ8, 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const std::int64_t k = JitllmOpInt(node, 0);
  const std::int64_t rows = JitllmOpInt(node, 1);
  if (!IsF32(x) || !IsBytes(node) || x->ne[3] != 1 || k != x->ne[0] ||
      rows != x->ne[1] * x->ne[2] || k % kQ8Block != 0 || !Shaped(node, Q8Bytes(k, rows), 1, 1)) {
    return Rejected("F32 rows of a multiple of 32 values into their Q8_1 blocks");
  }
  if (x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || x->nb[2] % sizeof(float) != 0 ||
      !Aligned(x, 4) || !Aligned(node, 4) || x->ne[1] > 65535 || x->ne[2] > 65535 ||
      std::cmp_greater(k, kInt32Max)) {
    return Rejected("rows of packed floats within the quantizer's grid");
  }
  if (AnyEmpty({x}) || !AllSane({x}) || !AllCurrent({x, node}) || !Disjoint(node, x, false)) {
    return Rejected("a current, measurable input disjoint from the output");
  }
  return {};
}

std::expected<void, KernelFailure> CheckVecQ(const ggml_tensor* node) {
  if (node == nullptr || JitllmOpOf(node) != JitllmOp::kVecQ || !Bound(node)) {
    return Rejected("not a bound jitllm.vecq node");
  }
  const ggml_tensor* w = node->src[0];
  const ggml_tensor* q8 = node->src[1];
  const bool routed = node->src[2] != nullptr && node->src[2]->type == GGML_TYPE_I32;
  const ggml_tensor* ids = routed ? node->src[2] : nullptr;
  const ggml_tensor* gate = routed ? node->src[3] : node->src[2];
  const std::int64_t tokens = JitllmOpInt(node, 0);
  const std::int64_t per_slot = JitllmOpInt(node, 1);
  const std::int64_t glu = JitllmOpInt(node, 2);
  if (!VecQWeights(w) || q8 == nullptr || JitllmOpOf(q8) != JitllmOp::kQuantizeQ8 || !Bound(q8) ||
      !IsF32(node)) {
    return Rejected("quantized weights jitllm.vecq reads, and a jitllm.q8_1 input");
  }
  const std::int64_t one_token = JitllmOpInt(node, 4);
  if (tokens < 1 || tokens > kVecQMaxTokens || (per_slot != 0 && per_slot != 1) || glu < 0 ||
      glu > 3 || ((glu != 0) != (gate != nullptr)) || (one_token != 0 && one_token != 1)) {
    return Rejected("1 to 16 tokens, and gate weights exactly with a GLU");
  }
  if (glu == static_cast<std::int64_t>(VecQGlu::kGeGlu) && JitllmOpFloat(node, 3) != 0.0f) {
    return Rejected("GeGLU uses GELU-tanh without a clamp limit");
  }
  if (gate != nullptr && (!VecQWeights(gate) || gate->type != w->type ||
                          !ggml_are_same_shape(gate, w) || !ggml_are_same_stride(gate, w))) {
    return Rejected("gate weights of the weights' type, shape and strides");
  }
  const std::int64_t k = w->ne[0];
  const std::int64_t n = w->ne[1];
  std::int64_t used = 1;
  if (routed) {
    used = ids->ne[0];
    if (ids->ne[1] != tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        ids->nb[0] != sizeof(std::int32_t) || ids->nb[1] % sizeof(std::int32_t) != 0 || used < 1 ||
        used * tokens > 128 || !Aligned(ids, 4) || AnyEmpty({ids}) || !AllSane({ids}) ||
        !AllCurrent({ids}) || !Disjoint(node, ids, false) || !Shaped(node, n, used, tokens)) {
      return Rejected("ids I32 [used, tokens] and an output [n, used, tokens]");
    }
  } else if (w->ne[2] == 1) {
    if (per_slot != 0 || !Shaped(node, n, tokens, 1)) {
      return Rejected("dense weights [k, n] into [n, tokens]");
    }
  } else {
    // Grouped: matrix g over each token's input row g. The kernel packs a
    // (token, slot) pair's slot in 8 bits.
    used = w->ne[2];
    if (per_slot != 1 || gate != nullptr || used > 256 || !Shaped(node, n, used, tokens)) {
      return Rejected(
          "grouped weights [k, n, groups] over [k, groups, tokens] into [n, groups, "
          "tokens]");
    }
  }
  const std::int64_t rows = per_slot != 0 ? used * tokens : tokens;
  if (JitllmOpInt(q8, 0) != k || JitllmOpInt(q8, 1) != rows) {
    return Rejected("the Q8_1 input's rows are the product's");
  }
  if (!Packed(node) || !Aligned(node, 4) || !AllCurrent({node, q8}) || !Disjoint(node, w, false) ||
      !Disjoint(node, q8, false) || (gate != nullptr && !Disjoint(node, gate, false)) ||
      std::cmp_greater(n, kInt32Max) || std::cmp_greater(k, kInt32Max)) {
    return Rejected("a packed output disjoint from its operands, within the kernel's grid");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDsv4Route(const ggml_tensor* node) {
  if (node == nullptr || JitllmOpOf(node) != JitllmOp::kDsv4Route || !Bound(node)) {
    return Rejected("not a bound jitllm.dsv4.route node");
  }
  const ggml_tensor* logits = node->src[0];
  const bool hashed = node->src[2] != nullptr;
  const std::int64_t used = JitllmOpInt(node, 0);
  const std::int64_t tokens = logits == nullptr ? 0 : logits->ne[1];
  if (!IsF32(logits) || !Bound(logits) || logits->ne[0] != 256 || logits->ne[2] != 1 ||
      logits->ne[3] != 1 || !Packed(logits) || tokens < 1 || tokens > 65535 || used < 1 ||
      used > 32 || node->type != GGML_TYPE_I32 || !Shaped(node, 2 * used, tokens, 1) ||
      !Packed(node)) {
    return Rejected("F32 logits [256, tokens] into I32 [2 · used, tokens], at most 32 used");
  }
  if (hashed) {
    const ggml_tensor* table = node->src[1];
    const ggml_tensor* ids = node->src[2];
    if (table == nullptr || table->type != GGML_TYPE_I32 || !Bound(table) || !Packed(table) ||
        table->ne[0] != used || table->ne[2] != 1 || ids->type != GGML_TYPE_I32 || !Bound(ids) ||
        !Packed(ids) || !Shaped(ids, tokens, 1, 1) || !Disjoint(node, table, false) ||
        !Disjoint(node, ids, false) || std::cmp_greater(table->ne[1], kInt32Max)) {
      return Rejected("a hash table I32 [used, vocab] and the tokens I32 [tokens]");
    }
  } else if (!Vector(node->src[1], 256) || !Bound(node->src[1]) ||
             !Disjoint(node, node->src[1], false)) {
    return Rejected("a bias F32 [256]");
  }
  if (!AllCurrent({node, logits}) || !Disjoint(node, logits, false) || AnyEmpty({logits}) ||
      !AllSane({logits, node})) {
    return Rejected("current operands disjoint from the output");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDsv4Combine(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kDsv4Combine, 3); !checked) {
    return checked;
  }
  const ggml_tensor* down = node->src[0];
  const ggml_tensor* route = node->src[1];
  const ggml_tensor* shared = node->src[2];
  const std::int64_t n = down->ne[0];
  const std::int64_t used = down->ne[1];
  const std::int64_t tokens = down->ne[2];
  if (!IsF32(down) || JitllmOpOf(route) != JitllmOp::kDsv4Route || JitllmOpInt(route, 0) != used ||
      !Shaped(route, 2 * used, tokens, 1) || !IsF32(shared) || !Shaped(shared, n, tokens, 1) ||
      !IsF32(node) || !Shaped(node, n, tokens, 1) || n % 4 != 0 || !Aligned(down, 16) ||
      !Aligned(shared, 16) || !Aligned(node, 16) || tokens > 65535) {
    return Rejected("down F32 [n, used, tokens], its routing, shared F32 [n, tokens]");
  }
  return CheckDense(node, {down, route, shared});
}

std::expected<void, KernelFailure> CheckDsv4HcMix(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kDsv4HcMix, 2); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* fn = node->src[1];
  const std::int64_t flat = x->ne[0] * x->ne[1];
  if (!IsF32(x) || !Dsv4HcMixWeightType(fn->type) || x->ne[1] != kDsv4HcStreams ||
      fn->ne[0] != flat || fn->ne[1] != kDsv4HcMixes ||
      !Shaped(node, kDsv4HcMixes + 1, kDsv4HcChunks, x->ne[2]) ||
      flat % (kDsv4HcChunks * kDsv4HcChunkThreads) != 0 || std::cmp_greater(flat, kInt32Max) ||
      x->ne[2] > 65535 || !Aligned(x, 16) || !Aligned(fn, 16)) {
    return Rejected(
        "streams F32 [width, 4, tokens] and mixing weights F32, F16 or BF16 [4 · width, 24]");
  }
  return CheckDense(node, {x, fn});
}

std::expected<void, KernelFailure> CheckDsv4HcPre(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kDsv4HcPre, 5); !checked) {
    return checked;
  }
  const ggml_tensor* partials = node->src[0];
  const ggml_tensor* x = node->src[1];
  const std::int64_t width = x->ne[0];
  const std::int64_t tokens = x->ne[2];
  if (JitllmOpOf(partials) != JitllmOp::kDsv4HcMix || partials->src[0] != x || !IsF32(x) ||
      x->ne[1] != kDsv4HcStreams || !Vector(node->src[2], 3) ||
      !Vector(node->src[3], kDsv4HcMixes) || !Vector(node->src[4], width) ||
      !Shaped(node, (width + kDsv4HcTail) * tokens, 1, 1) || width % 1024 != 0 || width > 8192 ||
      JitllmOpInt(node, 2) < 1 || JitllmOpInt(node, 2) > 100 || tokens > 65535) {
    return Rejected("a jitllm.dsv4.hc_mix of the streams, their scales, bases and norm weight");
  }
  return CheckDense(node, {partials, x, node->src[2], node->src[3], node->src[4]});
}

// ---------------------------------------------------------------- Gemma 4 attention masks

ggml_tensor* FlashAttnOwnersNode(ggml_context* context, ggml_tensor* q, ggml_tensor* mask,
                                 const std::array<ggml_tensor*, 4>& k,
                                 const std::array<ggml_tensor*, 4>& v, std::uint32_t logical_cohort,
                                 std::uint32_t owner_count, std::uint32_t owner_offset,
                                 std::uint32_t logit_softcap, bool bounded_roots) {
  static_assert(GGML_MAX_SRC == 10);
  // The pinned factory requires fewer than GGML_MAX_SRC arguments, although
  // descriptors and traversal support all ten slots. Preserve its custom tag
  // parameters, then install the final real dependency in the remaining slot.
  auto* node =
      Custom(context, GGML_TYPE_F32, {q->ne[0], q->ne[2], 1, owner_count},
             {q, mask, k[0], k[1], k[2], k[3], v[0], v[1], v[2]}, kTagFlashAttnOwners.data());
  node->src[9] = v[3];
  // Zero keeps the historical four-owner descriptor byte-identical.
  return WithInts(node, {static_cast<std::int32_t>(logical_cohort),
                         owner_count == 4 ? 0 : static_cast<std::int32_t>(owner_count),
                         static_cast<std::int32_t>(owner_offset),
                         static_cast<std::int32_t>(logit_softcap), bounded_roots ? 1 : 0});
}

ggml_tensor* Gemma4Mask(ggml_context* context, ggml_tensor* positions, std::int64_t cells,
                        std::int32_t first_row, std::int32_t rows, std::int32_t capacity,
                        std::int32_t window, std::int32_t context_limit) {
  const auto padded_rows = (std::int64_t{rows} + 31) / 32 * 32;
  return WithInts(Custom(context, GGML_TYPE_F16, {cells, padded_rows, 1, 1}, {positions},
                         kTagGemma4Mask.data()),
                  {first_row, rows, capacity, window, context_limit});
}

bool Gemma4MaskFits(const ggml_tensor* node) {
  if (node == nullptr || JitllmOpOf(node) != JitllmOp::kGemma4Mask || node->src[0] == nullptr ||
      node->src[1] != nullptr)
    return false;
  constexpr std::int64_t max = std::numeric_limits<std::int32_t>::max();
  const auto* positions = node->src[0];
  const std::int64_t first = JitllmOpInt(node, 0), rows = JitllmOpInt(node, 1),
                     capacity = JitllmOpInt(node, 2), window = JitllmOpInt(node, 3),
                     limit = JitllmOpInt(node, 4);
  // Refuse public descriptor endpoints before padding, shape products,
  // GGML byte calculations or the kernel's integer narrowing.
  if (first < 0 || rows < 1 || rows > max - 31 || capacity < 1 || capacity % 256 != 0 ||
      limit < 1 || window < 0 || node->ne[0] < 1 || node->ne[0] > max / 2 ||
      node->ne[0] % 256 != 0 || node->ne[0] > capacity || node->ne[1] < 1 ||
      node->ne[1] > max / 2 || node->ne[2] != 1 || node->ne[3] != 1 || positions->ne[0] < 1 ||
      positions->ne[0] > max / 4 || positions->ne[1] != 1 || positions->ne[2] != 1 ||
      positions->ne[3] != 1 || first > positions->ne[0] || rows > positions->ne[0] - first ||
      node->ne[0] > max / 2 / node->ne[1])
    return false;
  const std::int64_t padded_rows = (rows + 31) / 32 * 32;
  const std::int64_t retained = std::min(limit, window + rows);
  if ((window != 0 && capacity < retained) || node->type != GGML_TYPE_F16 ||
      positions->type != GGML_TYPE_I32 || !Shaped(node, node->ne[0], padded_rows, 1) ||
      AnyEmpty({node, positions}) || !AllSane({node, positions}) || !Packed(positions) ||
      !Packed(node))
    return false;
  return true;
}

std::expected<void, KernelFailure> CheckGemma4Mask(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGemma4Mask, 1); !checked) return checked;
  if (!Gemma4MaskFits(node) || !AllCurrent({node, node->src[0]}) || !Aligned(node, 2) ||
      !Aligned(node->src[0], 4) || !Disjoint(node, node->src[0], false))
    return Rejected("Gemma4 packed F16 causal/ring mask from current disjoint I32 positions");
  return {};
}

// ---------------------------------------------------------------- DeepSeek V4's sparse attention

ggml_tensor* Dsv4LidTopK(ggml_context* context, ggml_tensor* q, ggml_tensor* k, ggml_tensor* w,
                         ggml_tensor* visible, std::int64_t top) {
  return Custom(context, GGML_TYPE_I32, {top, q->ne[2], 1, 1}, {q, k, w, visible},
                kTagDsv4LidTopK.data());
}

ggml_tensor* Dsv4SparseMask(ggml_context* context, ggml_tensor* window, ggml_tensor* top,
                            ggml_tensor* visible, std::int64_t cells, std::int64_t n_kv) {
  return WithInts(Custom(context, GGML_TYPE_F16, {cells + n_kv, window->ne[1], 1, 1},
                         {window, top != nullptr ? top : visible}, kTagDsv4SparseMask.data()),
                  {cells, top != nullptr ? 0 : 1});
}

std::expected<void, KernelFailure> CheckDsv4LidTopK(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kDsv4LidTopK, 4); !checked) {
    return checked;
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* k = node->src[1];
  const ggml_tensor* w = node->src[2];
  const ggml_tensor* visible = node->src[3];
  const std::int64_t rows = q->ne[2];
  const std::int64_t n_kv = k->ne[1];
  const std::int64_t top = node->ne[0];
  std::initializer_list<const ggml_tensor*> operands = {node, q, k, w, visible};
  if (AnyEmpty(operands) || !AllSane(operands) || !AllCurrent(operands)) {
    return Rejected("the indexer's selection on an empty, unmeasurable or stale tensor");
  }
  // The kernels' layout: 128-wide query heads, 64 of them; F16 keys with
  // packed rows; the weights and visible counts packed per row.
  if (!IsF32(q) || q->ne[0] != 128 || q->ne[1] != 64 || q->ne[3] != 1 || q->nb[0] != 4 ||
      q->nb[1] % 8 != 0 || q->nb[2] % 8 != 0 || !Aligned(q, 8) || k->type != GGML_TYPE_F16 ||
      k->ne[0] != 128 || k->ne[2] != 1 || k->ne[3] != 1 || k->nb[0] != 2 || k->nb[1] % 16 != 0 ||
      !Aligned(k, 16) || !IsF32(w) || !Shaped(w, 64, rows, 1) || w->nb[0] != 4 ||
      visible->type != GGML_TYPE_I32 || !Packed(visible) || !Shaped(visible, rows, 1, 1) ||
      node->type != GGML_TYPE_I32 || !Packed(node) || !Shaped(node, top, rows, 1) || top < 1 ||
      top > 65536) {
    return Rejected(
        "queries F32 [128, 64, rows], the indexer's F16 keys [128, n_kv], weights F32 [64, rows] "
        "and visible counts I32 [rows], into I32 [top, rows]");
  }
  // 32-bit strides and extents in the kernels; the score scratch's rows.
  if (std::cmp_greater(q->nb[2] / 4 * static_cast<std::uint64_t>(rows), kInt32Max) ||
      std::cmp_greater(k->nb[1] / 2 * static_cast<std::uint64_t>(n_kv), kInt32Max) ||
      std::cmp_greater(w->nb[1] / 4 * static_cast<std::uint64_t>(rows), kInt32Max) ||
      rows > std::int64_t{65535} * 4 || std::cmp_greater(n_kv, kInt32Max / 4) ||
      std::cmp_greater(static_cast<std::uint64_t>(top) * static_cast<std::uint64_t>(rows),
                       kInt32Max)) {
    return Rejected("the indexer's selection beyond the kernels' 32-bit extents");
  }
  for (const ggml_tensor* t : {q, k, w, visible}) {
    if (!Disjoint(node, t, false)) {
      return Rejected("an output overlapping an operand");
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckDsv4SparseMask(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kDsv4SparseMask, 2); !checked) {
    return checked;
  }
  const ggml_tensor* window = node->src[0];
  const ggml_tensor* rows_of = node->src[1];
  const std::int64_t rows = node->ne[1];
  const std::int64_t cells = JitllmOpInt(node, 0);
  const std::int64_t by_count = JitllmOpInt(node, 1);
  std::initializer_list<const ggml_tensor*> operands = {node, window, rows_of};
  if (AnyEmpty(operands) || !AllSane(operands) || !AllCurrent(operands)) {
    return Rejected("a sparse mask of an empty, unmeasurable or stale tensor");
  }
  const bool selection = by_count == 0 && JitllmOpOf(rows_of) == JitllmOp::kDsv4LidTopK &&
                         rows_of->ne[1] == rows && Packed(rows_of);
  const bool counts = by_count == 1 && rows_of->type == GGML_TYPE_I32 && Packed(rows_of) &&
                      Shaped(rows_of, rows, 1, 1);
  if (window->type != GGML_TYPE_F16 || window->nb[0] != 2 || window->ne[1] != rows ||
      window->ne[2] != 1 || window->ne[3] != 1 || (!selection && !counts) ||
      cells < window->ne[0] || node->type != GGML_TYPE_F16 || !Packed(node) ||
      node->ne[0] <= cells || node->ne[2] != 1 || node->ne[3] != 1 ||
      std::cmp_greater(node->ne[0], kInt32Max) ||
      node->ne[0] > std::int64_t{65535} * 4096 ||  // the kernel's grid: 4,096 columns a block
      std::cmp_greater(window->nb[1] / 2 * static_cast<std::uint64_t>(rows), kInt32Max) ||
      std::cmp_greater(static_cast<std::uint64_t>(node->ne[0]) * static_cast<std::uint64_t>(rows),
                       kInt32Max)) {
    return Rejected(
        "a window mask F16 [w, rows] and a jitllm.dsv4.lid_topk selection or visible counts I32 "
        "[rows], into F16 [cells + n_kv, rows], cells at least w");
  }
  for (const ggml_tensor* t : {window, rows_of}) {
    if (!Disjoint(node, t, false)) {
      return Rejected("an output overlapping an operand");
    }
  }
  return {};
}

}  // namespace jitllm::kernels::ggml
