// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/capabilities.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>

#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/set_rows_group.h"

#ifndef LLMP_GGML_CUDA_ARCHITECTURES
#error "the build names the GGML module's device architectures"
#endif

namespace llmp::kernels::ggml {
namespace {

namespace ex = llmp::execution;
using format::Encoding;
using format::Layout;
using ir::FusedKind;
using ir::Op;
using Operation = ex::Operation;

// ------------------------------------------------------------- operand lists

constexpr std::array kF32 = {Encoding::kF32};
constexpr std::array kF16 = {Encoding::kF16};
constexpr std::array kI32 = {Encoding::kI32};
constexpr std::array kI64 = {Encoding::kI64};
constexpr std::array kIds = {Encoding::kI32, Encoding::kI64};
constexpr std::array kQ8_1 = {Encoding::kQ8_1};
constexpr std::array kFloats = {Encoding::kF32, Encoding::kF16, Encoding::kBF16};
constexpr std::array kQ8 = {Encoding::kQ8_0};
constexpr std::array kQ4 = {Encoding::kQ4_0};
constexpr std::array kBlocks = {Encoding::kQ8_0, Encoding::kQ4_0};
// The quantized products' weight types (validate_ext.cc QuantizedWeightTypes,
// those in GGML's block layout).
constexpr std::array kQuantized = {
    Encoding::kQ8_0,    Encoding::kQ4_K,  Encoding::kQ5_K,  Encoding::kQ6_K,    Encoding::kIQ2_XS,
    Encoding::kIQ3_XXS, Encoding::kMxfp4, Encoding::kQ2_K,  Encoding::kIQ2_XXS, Encoding::kQ4_1,
    Encoding::kQ5_0,    Encoding::kQ5_1,  Encoding::kQ4_0,  Encoding::kQ2_0,    Encoding::kQ3_K,
    Encoding::kIQ1_S,   Encoding::kIQ2_S, Encoding::kIQ3_S, Encoding::kIQ4_NL,  Encoding::kIQ4_XS};
// The weights a prepared Q8_1 input serves (shared_q8.h SharedQ8Inputs).
constexpr std::array kPreparedWeights = {Encoding::kQ4_0, Encoding::kQ4_K, Encoding::kQ5_K,
                                         Encoding::kQ6_K, Encoding::kQ8_0, Encoding::kQ4_1,
                                         Encoding::kQ5_0, Encoding::kQ5_1, Encoding::kIQ4_NL};
constexpr std::array kDense = {Layout::kDense};
constexpr std::array kGgml = {Layout::kGgmlBlocks};

using Rule = ex::OperandRule;
constexpr std::array<Rule, 1> kF32In = {{{.operand = 0, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kF32Binary = {{{.operand = 0, .encodings = kF32, .layouts = kDense},
                                             {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kF32Glu = kF32Binary;
constexpr std::array<Rule, 1> kFloatTable = {
    {{.operand = 0, .encodings = kFloats, .layouts = kDense}}};
constexpr std::array<Rule, 2> kRope = {{{.operand = 0, .encodings = kF32, .layouts = kDense},
                                        {.operand = 1, .encodings = kI32, .layouts = kDense}}};
// kCacheStore's (cache, src, idx); grouped_stores' member store's.
constexpr std::array<Rule, 3> kF16Store = {{{.operand = 0, .encodings = kF16, .layouts = kDense},
                                            {.operand = 1, .encodings = kF32, .layouts = kDense},
                                            {.operand = 2, .encodings = kI64, .layouts = kDense}}};
constexpr std::array<Rule, 3> kFloatStore = {
    {{.operand = 0, .encodings = kFloats, .layouts = kDense},
     {.operand = 1, .encodings = kFloats, .layouts = kDense},
     {.operand = 2, .encodings = kIds, .layouts = kDense}}};
constexpr std::array<Rule, 3> kQuantStore = {
    {{.operand = 0, .encodings = kBlocks, .layouts = kGgml},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kIds, .layouts = kDense}}};
// kRopeStore's (x, pos, cache, idx, freq?).
constexpr std::array<Rule, 4> kF16RopeStore = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = 2, .encodings = kF16, .layouts = kDense},
     {.operand = 3, .encodings = kI64, .layouts = kDense}}};
constexpr std::array<Rule, 4> kQuantRopeStore = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = 2, .encodings = kBlocks, .layouts = kGgml},
     {.operand = 3, .encodings = kI64, .layouts = kDense}}};
// kAttention's (q, k, v, mask, sinks?).
constexpr std::array<Rule, 4> kF16Attention = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF16, .layouts = kDense},
     {.operand = 3, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 4> kQ8Attention = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kQ8, .layouts = kGgml},
     {.operand = 2, .encodings = kQ8, .layouts = kGgml},
     {.operand = 3, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 4> kQ4Attention = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kQ4, .layouts = kGgml},
     {.operand = 2, .encodings = kQ4, .layouts = kGgml},
     {.operand = 3, .encodings = kF16, .layouts = kDense}}};
// Products (w, x).
constexpr std::array<Rule, 2> kQuantProduct = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kFloatProduct = {
    {{.operand = 0, .encodings = kFloats, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kPreparedProduct = {
    {{.operand = 0, .encodings = kPreparedWeights, .layouts = kGgml},
     {.operand = 1, .encodings = kQ8_1, .layouts = kGgml}}};
// fused.mul_mat_glu's (w_gate, w_up, x).
constexpr std::array<Rule, 3> kQuantGlu = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
// The norms' (x, w, ...).
constexpr std::array<Rule, 2> kNormMul = kF32Binary;
constexpr std::array<Rule, 3> kNormMulAdd = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kNormMulRope = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 1> kMaskPositions = {
    {{.operand = 0, .encodings = kI32, .layouts = kDense}}};

// ------------------------------------------------------------- predicates

template <typename P>
std::optional<P> ParamsOf(const ex::CapabilityQuery& q) {
  if (q.params.size() < sizeof(P)) {
    return std::nullopt;
  }
  P p{};
  std::memcpy(&p, q.params.data(), sizeof(P));
  return p;
}

bool Has(const ex::CapabilityQuery& q, std::uint8_t operand) {
  return std::ranges::any_of(
      q.operands, [operand](const ex::OperandFormat& o) { return o.operand == operand; });
}

const ex::OperandFormat* Operand(const ex::CapabilityQuery& q, std::uint8_t operand) {
  const auto it = std::ranges::find_if(
      q.operands, [operand](const ex::OperandFormat& o) { return o.operand == operand; });
  return it == q.operands.end() ? nullptr : &*it;
}

bool Built(const ex::CapabilityQuery& q) { return CompiledFor(q.arch); }

// GGML's rope_ext over positions (not a table), not inverted (rope_back is
// ggml.rope.ext's), in a mode the lowering builds.
bool PlainRope(const ir::RopeParams& r) {
  return !r.table && (r.mode == ir::RopeMode::kNeox || r.mode == ir::RopeMode::kNormal);
}

// ggml.rope.neox: NEOX without frequency factors (operand 2) or an offset
// (graph_plan.cc's structural choice).
bool AcceptsRopeNeox(const ex::CapabilityQuery& q) {
  const auto r = ParamsOf<ir::RopeParams>(q);
  return Built(q) && r && PlainRope(*r) && r->mode == ir::RopeMode::kNeox && !r->inverse &&
         r->offset == 0 && !Has(q, 2);
}
bool AcceptsRopeExt(const ex::CapabilityQuery& q) {
  const auto r = ParamsOf<ir::RopeParams>(q);
  return Built(q) && r && !r->table;
}

// CheckGemmaNormRope's (gemma_norm.cc): a full NEOX rotation of D 256 or
// 512 without frequency factors, its parameters finite and positive.
bool AcceptsNormRope(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormRopeParams>(q);
  if (!Built(q) || !p || (q.k != 256 && q.k != 512) || Has(q, 3)) {
    return false;
  }
  const ir::RopeParams& r = p->rope;
  const std::array values = {r.base,        r.scale,     r.ext_factor,
                             r.attn_factor, r.beta_fast, r.beta_slow};
  return r.mode == ir::RopeMode::kNeox && !r.inverse && !r.table && r.offset == 0 &&
         r.dims == q.k && r.original_context > 0 &&
         std::ranges::all_of(values, [](float v) { return std::isfinite(v); }) && r.base > 1.0F &&
         r.scale > 0.0F && r.beta_fast > 0.0F && r.beta_slow > 0.0F && r.attn_factor > 0.0F &&
         std::isfinite(p->eps) && p->eps >= 0.0F;
}
// CheckGemmaNormAdd's widths.
bool AcceptsNormAdd(const ex::CapabilityQuery& q) {
  return Built(q) && (q.k == 2304 || q.k == 2560 || q.k == 2816 || q.k == 5376);
}

// Upstream's MMVQ for the product (the weights are operand 0).
bool Routed(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* w = Operand(q, 0);
  return w != nullptr && MmvqRouted(w->encoding, q.arch, q.rows);
}
bool AcceptsMmvq(const ex::CapabilityQuery& q) { return Built(q) && Routed(q); }
bool AcceptsMmq(const ex::CapabilityQuery& q) { return Built(q) && !Routed(q); }
bool AcceptsFloatProduct(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::ProductParams>(q);
  return Built(q) && (!p || p->hint == ir::ProductHint::kNone);
}
bool AcceptsHadamard(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::ProductParams>(q);
  return Built(q) && p && p->hint == ir::ProductHint::kHadamard && q.k == q.n;
}
// MulMatVecQGluFusible (mul_mat_q.cu): one column, a device past Pascal,
// gate and up weights of one type (operands 0 and 1), MMVQ selected for
// the up product, and the GLU it fuses.
bool QuantGlu(const ex::CapabilityQuery& q, ir::GluKind kind) {
  const auto g = ParamsOf<ir::GluParams>(q);
  const ex::OperandFormat* gate = Operand(q, 0);
  const ex::OperandFormat* up = Operand(q, 1);
  return Built(q) && g && g->kind == kind && !g->swapped && q.rows == 1 && q.arch > 60 &&
         gate != nullptr && up != nullptr && gate->encoding == up->encoding && Routed(q);
}
bool AcceptsQuantGeGlu(const ex::CapabilityQuery& q) {
  return QuantGlu(q, ir::GluKind::kGeGluTanh);
}
bool AcceptsQuantSwiGlu(const ex::CapabilityQuery& q) { return QuantGlu(q, ir::GluKind::kSwiGlu); }

// The GLU node's kind, of two sources (split), not swapped.
bool Glu(const ex::CapabilityQuery& q, ir::GluKind kind) {
  const auto g = ParamsOf<ir::GluParams>(q);
  return Built(q) && g && g->kind == kind && !g->swapped && Has(q, 1);
}
bool AcceptsGeGlu(const ex::CapabilityQuery& q) { return Glu(q, ir::GluKind::kGeGluTanh); }
bool AcceptsSwiGlu(const ex::CapabilityQuery& q) { return Glu(q, ir::GluKind::kSwiGlu); }
bool AcceptsSwiGluClamp(const ex::CapabilityQuery& q) { return Glu(q, ir::GluKind::kSwiGluClamp); }

// The F16 flash-attention family (graph_plan.cc): D 256 (the GQA-2 pair,
// vector or tensor-core, by device) or 512, K's head size V's.
bool AcceptsAttention(const ex::CapabilityQuery& q) {
  return Built(q) && (q.k == 256 || q.k == 512) && q.n == q.k;
}
bool AcceptsAttention256(const ex::CapabilityQuery& q) {
  return Built(q) && q.k == 256 && q.n == 256;
}

// llmp.gemma4.mask (llmp_ops.h Gemma4Mask): a causal or window mask; its
// producer derives the cells' positions from the ring's capacity.
bool AcceptsMask(const ex::CapabilityQuery& q) {
  const auto m = ParamsOf<ir::MaskParams>(q);
  return Built(q) && m && (m->kind == ir::MaskKind::kCausal || m->kind == ir::MaskKind::kWindow);
}

bool AcceptsArgmax(const ex::CapabilityQuery& q) { return Built(q); }
bool AcceptsAny(const ex::CapabilityQuery& q) { return Built(q); }
bool AcceptsPrepared(const ex::CapabilityQuery& q) { return Built(q) && Routed(q); }

// kv_quant.h's: head sizes 64 and 256, no sinks; rows of whole blocks.
bool AcceptsQuantAttention(const ex::CapabilityQuery& q) {
  return Built(q) && (q.k == 64 || q.k == 256) && q.n == q.k && !Has(q, 4);
}
bool AcceptsQuantStore(const ex::CapabilityQuery& q) { return Built(q) && q.k >= 32; }
bool AcceptsQuantRopeStore(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* freq = Operand(q, 4);
  return AcceptsQuantStore(q) &&
         (freq == nullptr || (freq->encoding == Encoding::kF32 && freq->layout == Layout::kDense));
}
// ggml.rope_set_rows.fused (RopeSetRowsFusionAt): NEOX into F16 rows.
bool AcceptsRopeStore(const ex::CapabilityQuery& q) {
  const auto r = ParamsOf<ir::RopeParams>(q);
  return Built(q) && r && PlainRope(*r) && r->mode == ir::RopeMode::kNeox && !r->inverse &&
         r->offset == 0 && !Has(q, 4);
}

// ------------------------------------------------------------- the table

// Rung 1 (each computes its encoding's definition), no expansion but
// cuBLAS's (GGML's cuBLAS product dequantizes into pool scratch: not for
// quantized weights here, which upstream routes to MMVQ or MMQ), not
// lossy, capturable. Row-invariant where each row is computed alone
// (norms, rotations, elementwise, gathers and stores): not the products,
// whose kernels upstream picks by the column count, nor attention.
struct Entry {
  std::string_view name;
  Operation operation;
  ex::Capability capability;
  std::string_view accepts;
};

constexpr ex::Capability Cap(ir::OpKey op, std::span<const Rule> rules,
                             bool (*accepts)(const ex::CapabilityQuery&), bool row_invariant,
                             std::string_view group = {}, std::uint16_t priority = 0,
                             ex::ShapeRule shape = {}, std::uint16_t batch = 0) {
  return {.op = op,
          .operands = rules,
          .shape = shape,
          .rung = ex::Rung::kNative,
          .expansion = ex::Expansion::kNone,
          .lossy = false,
          .row_invariant = row_invariant,
          .capturable = true,
          .group = group,
          .priority = priority,
          .batch = batch,
          .accepts = accepts};
}

constexpr ir::OpKey P(Op op) { return ir::Primitive(op); }
constexpr ir::OpKey F(FusedKind kind) { return ir::Fused(kind); }

constexpr std::string_view kMulMatGroup = "ggml.mul_mat";
constexpr std::string_view kAttentionGroup = "ggml.flash_attn_ext";
constexpr std::string_view kBuilt = "; selectable for the build's SASS architectures";
constexpr std::string_view kMmvqText =
    "; selectable where upstream's ggml_cuda_should_use_mmvq takes the product";
constexpr std::string_view kMmqText =
    "; selectable where upstream's ggml_cuda_should_use_mmvq does not take the product";

const std::array kEntries = {
    // Gathers, elementwise, norms.
    Entry{"ggml.get_rows", Operation::kGetRows,
          Cap(P(Op::kGetRows), kFloatTable, &AcceptsAny, true, {}, 1), kBuilt},
    Entry{"ggml.get_rows.ext", Operation::kGetRows, Cap(P(Op::kGetRows), {}, &AcceptsAny, true),
          kBuilt},
    Entry{"ggml.scale", Operation::kScale, Cap(P(Op::kScale), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.add", Operation::kAdd, Cap(P(Op::kAdd), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.mul", Operation::kMul, Cap(P(Op::kMul), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.sub", Operation::kSub, Cap(P(Op::kSub), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.div", Operation::kDiv, Cap(P(Op::kDiv), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.unary", Operation::kUnary, Cap(P(Op::kUnary), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.clamp", Operation::kClamp, Cap(P(Op::kClamp), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.fill", Operation::kFill, Cap(P(Op::kFill), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.repeat", Operation::kRepeat, Cap(P(Op::kRepeat), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.concat", Operation::kConcat, Cap(P(Op::kConcat), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.cont", Operation::kCont, Cap(P(Op::kCont), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.sum_rows", Operation::kSumRows, Cap(P(Op::kSumRows), kF32In, &AcceptsAny, true),
          kBuilt},
    Entry{"ggml.soft_max", Operation::kSoftMax, Cap(P(Op::kSoftMax), {}, &AcceptsAny, true),
          kBuilt},
    Entry{"ggml.argsort.bitonic", Operation::kArgsort,
          Cap(P(Op::kArgsort), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.rms_norm", Operation::kRmsNorm, Cap(P(Op::kRmsNorm), kF32In, &AcceptsAny, true),
          kBuilt},
    Entry{"ggml.rms_norm_mul.fused", Operation::kRmsNormMul,
          Cap(F(FusedKind::kRmsNormMul), kNormMul, &AcceptsAny, true), kBuilt},
    Entry{kGemmaNormRopeName, Operation::kRmsNormMulRope,
          Cap(F(FusedKind::kRmsNormMulRope), kNormMulRope, &AcceptsNormRope, true),
          "; selectable for the build's SASS architectures: a full NEOX rotation of D 256 or 512 "
          "without frequency factors, its parameters finite"},
    Entry{kGemmaNormAddName, Operation::kRmsNormMulAdd,
          Cap(F(FusedKind::kRmsNormMulAdd), kNormMulAdd, &AcceptsNormAdd, true),
          "; selectable for the build's SASS architectures at widths 2304, 2560, 2816 and 5376"},
    // Rotations and stores.
    Entry{"ggml.rope.neox", Operation::kRope,
          Cap(P(Op::kRope), kRope, &AcceptsRopeNeox, true, {}, 1),
          "; selectable for the build's SASS architectures: NEOX over positions without "
          "frequency factors or an offset"},
    Entry{"ggml.rope.ext", Operation::kRope, Cap(P(Op::kRope), kRope, &AcceptsRopeExt, true),
          "; selectable for the build's SASS architectures over positions"},
    Entry{"ggml.set_rows", Operation::kSetRows,
          Cap(P(Op::kCacheStore), kF16Store, &AcceptsAny, true, {}, 1), kBuilt},
    Entry{"ggml.set_rows.ext", Operation::kSetRows,
          Cap(P(Op::kCacheStore), kFloatStore, &AcceptsAny, true), kBuilt},
    Entry{kSetRowsGroupedName, Operation::kSetRows,
          Cap(F(FusedKind::kGroupedStores), kF16Store, &AcceptsAny, true, {}, 0, {},
              static_cast<std::uint16_t>(kSetRowsGroupMax)),
          kBuilt},
    Entry{"ggml.rope_set_rows.fused", Operation::kRopeSetRows,
          Cap(F(FusedKind::kRopeStore), kF16RopeStore, &AcceptsRopeStore, true),
          "; selectable for the build's SASS architectures: NEOX into F16 rows"},
    // Products.
    Entry{"ggml.mul_mat.mmvq", Operation::kMatMul,
          Cap(P(Op::kMatMul), kQuantProduct, &AcceptsMmvq, false, kMulMatGroup, 1), kMmvqText},
    Entry{"ggml.mul_mat.mmq", Operation::kMatMul,
          Cap(P(Op::kMatMul), kQuantProduct, &AcceptsMmq, false, kMulMatGroup), kMmqText},
    Entry{"ggml.mul_mat.mmvf", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup, 2), kBuilt},
    Entry{"ggml.mul_mat.mmf", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup, 1), kBuilt},
    Entry{"ggml.mul_mat.cublas", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup), kBuilt},
    Entry{"ggml.mul_mat.fwht", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsHadamard, false, {}, 3),
          "; selectable for the build's SASS architectures: a square Hadamard hint"},
    Entry{"llmp.q8_1", Operation::kQuantize,
          Cap(P(Op::kPrepare), kF32In, &AcceptsAny, true, {}, 0, {.k_multiple = 32}),
          "; selectable for the build's SASS architectures: rows of whole Q8_1 blocks"},
    Entry{"llmp.mmvq.prepared", Operation::kMatMul,
          Cap(P(Op::kMatMul), kPreparedProduct, &AcceptsPrepared, false, {}, 0,
              {.min_rows = 1, .max_rows = 8, .k_multiple = 32}),
          kMmvqText},
    Entry{"ggml.mul_mat_geglu.mmvq_fused", Operation::kMulMatGeGlu,
          Cap(F(FusedKind::kMatMulGlu), kQuantGlu, &AcceptsQuantGeGlu, false),
          "; selectable for the build's SASS architectures past Pascal: GELU-tanh of one column "
          "over gate and up weights of one type, which upstream's ggml_cuda_should_use_mmvq "
          "takes"},
    Entry{"ggml.mul_mat_glu.mmvq_fused", Operation::kMulMatGlu,
          Cap(F(FusedKind::kMatMulGlu), kQuantGlu, &AcceptsQuantSwiGlu, false),
          "; selectable for the build's SASS architectures past Pascal: SwiGLU of one column "
          "over gate and up weights of one type, which upstream's ggml_cuda_should_use_mmvq "
          "takes"},
    Entry{"ggml.geglu", Operation::kGeGlu, Cap(P(Op::kGlu), kF32Glu, &AcceptsGeGlu, true),
          "; selectable for the build's SASS architectures: split GELU-tanh, not swapped"},
    Entry{"ggml.swiglu", Operation::kSwiGlu, Cap(P(Op::kGlu), kF32Glu, &AcceptsSwiGlu, true),
          "; selectable for the build's SASS architectures: split SwiGLU, not swapped"},
    Entry{"ggml.swiglu_clamp", Operation::kSwiGluClamp,
          Cap(P(Op::kGlu), kF32Glu, &AcceptsSwiGluClamp, true),
          "; selectable for the build's SASS architectures: split clamped SwiGLU"},
    // Attention and its masks.
    Entry{"ggml.flash_attn_ext.vec_d256", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 2),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma_gqa2", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 1),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention, false, kAttentionGroup),
          "; selectable for the build's SASS architectures at head size 256 or 512"},
    Entry{kGemma4MaskName, Operation::kFill,
          Cap(P(Op::kMaskBuild), kMaskPositions, &AcceptsMask, true),
          "; selectable for the build's SASS architectures: causal and window masks"},
    Entry{"llmp.argmax", Operation::kTopK, Cap(P(Op::kArgmax), kF32In, &AcceptsArgmax, true),
          kBuilt},
    // Block-quantized KV caches (kv_quant.h).
    Entry{kFlashAttnVecQ8Name, Operation::kFlashAttn,
          Cap(P(Op::kAttention), kQ8Attention, &AcceptsQuantAttention, false),
          "; selectable for the build's SASS architectures at head size 64 or 256 (K's and V's), "
          "without sinks"},
    Entry{kFlashAttnVecQ4Name, Operation::kFlashAttn,
          Cap(P(Op::kAttention), kQ4Attention, &AcceptsQuantAttention, false),
          "; selectable for the build's SASS architectures at head size 64 or 256 (K's and V's), "
          "without sinks"},
    Entry{kSetRowsQuantName, Operation::kSetRows,
          Cap(P(Op::kCacheStore), kQuantStore, &AcceptsQuantStore, true, {}, 0, {.k_multiple = 32}),
          "; selectable for the build's SASS architectures, rows of whole 32-value blocks"},
    Entry{kSetRowsQuantGroupName, Operation::kSetRows,
          Cap(F(FusedKind::kGroupedStores), kQuantStore, &AcceptsQuantStore, true, {}, 0,
              {.k_multiple = 32}, static_cast<std::uint16_t>(kSetRowsQuantGroupMax)),
          "; selectable for the build's SASS architectures, rows of whole 32-value blocks"},
    Entry{kRopeSetRowsQuantName, Operation::kRopeSetRows,
          Cap(F(FusedKind::kRopeStore), kQuantRopeStore, &AcceptsQuantRopeStore, true, {}, 0,
              {.k_multiple = 32}),
          "; selectable for the build's SASS architectures, rows of whole 32-value blocks"},
};

const std::array<CapabilityDeclaration, kEntries.size()> kDeclarations = [] {
  std::array<CapabilityDeclaration, kEntries.size()> out{};
  for (std::size_t i = 0; i < kEntries.size(); ++i) {
    out[i] = {.name = kEntries[i].name,
              .operation = kEntries[i].operation,
              .capability = &kEntries[i].capability};
  }
  return out;
}();

const Entry* Find(std::string_view name) {
  const auto it = std::ranges::find(kEntries, name, &Entry::name);
  return it == kEntries.end() ? nullptr : &*it;
}

}  // namespace

const execution::Capability* CapabilityOf(std::string_view implementation) {
  const Entry* e = Find(implementation);
  return e == nullptr ? nullptr : &e->capability;
}

std::string_view AcceptsTextOf(std::string_view implementation) {
  const Entry* e = Find(implementation);
  return e == nullptr ? std::string_view{} : e->accepts;
}

std::span<const CapabilityDeclaration> CapabilityDeclarations() { return kDeclarations; }

bool CompiledFor(std::uint32_t arch) {
  const std::string_view list = LLMP_GGML_CUDA_ARCHITECTURES;
  std::size_t at = 0;
  while (at < list.size()) {
    const std::size_t comma = std::min(list.find(',', at), list.size());
    const std::string_view entry = list.substr(at, comma - at);
    std::uint32_t compiled = 0;
    const auto [end, error] = std::from_chars(entry.data(), entry.data() + entry.size(), compiled);
    if (error == std::errc{} && compiled != 0) {
      const bool specific = end != entry.data() + entry.size() && *end == 'a';
      if (specific ? arch == compiled : (arch / 10 == compiled / 10 && arch >= compiled)) {
        return true;
      }
    }
    at = comma + 1;
  }
  return false;
}

bool MmvqRouted(format::Encoding weights, std::uint32_t arch, std::uint64_t columns) {
  // MMVQ_MAX_BATCH_SIZE (mmvq.cuh), and the compute capabilities upstream
  // tunes for (common.cuh GGML_CUDA_CC_*).
  constexpr std::uint64_t kMaxBatch = 8;
  const format::EncodingInfo& info = format::Info(weights);
  if (!info.quantized || !info.ggml.has_value() || columns == 0) {
    return false;
  }
  const std::uint32_t cc = arch * 10;
  const auto bound = [&]() -> std::uint64_t {
    switch (cc) {
      case 890:  // Ada Lovelace, tuned on RTX 4090
        return weights == Encoding::kQ2_K ? 4 : weights == Encoding::kQ3_K ? 6 : kMaxBatch;
      case 1200:  // Blackwell, tuned on RTX 5090
        switch (weights) {
          case Encoding::kQ2_K:
          case Encoding::kQ3_K:
          case Encoding::kQ4_K:
            return 5;
          case Encoding::kQ5_K:
            return 6;
          case Encoding::kQ6_K:
            return 7;
          default:
            return kMaxBatch;
        }
      case 1210:  // DGX Spark GB10
        return weights == Encoding::kQ2_K ? 6 : kMaxBatch;
      case 870:  // Jetson Orin
        switch (weights) {
          case Encoding::kQ2_K:
          case Encoding::kQ3_K:
          case Encoding::kQ4_K:
          case Encoding::kQ5_K:
          case Encoding::kQ6_K:
            return 1;
          default:
            return kMaxBatch;
        }
      case 700:  // Volta
        switch (weights) {
          case Encoding::kQ2_K:
            return 4;
          case Encoding::kQ3_K:
          case Encoding::kQ5_K:
            return 6;
          case Encoding::kQ4_K:
            return 5;
          case Encoding::kQ6_K:
            return 7;
          default:
            return kMaxBatch;
        }
      default:
        return kMaxBatch;
    }
  }();
  return columns <= bound;
}

}  // namespace llmp::kernels::ggml
