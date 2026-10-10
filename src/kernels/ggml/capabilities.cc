// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "kernels/ggml/capabilities.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>

#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/gemma_moe.h"
#include "kernels/ggml/gemma_moe_fusion.h"
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
constexpr std::array kF32F16 = {Encoding::kF32, Encoding::kF16};
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
// Qwen3.8's fast forms' (slice G3b): MXFP8 weights split, MXFP8 rows
// prepared, BF16 activations, NVFP4 in its layouts.
constexpr std::array kBf16 = {Encoding::kBF16};
constexpr std::array kF32Bf16 = {Encoding::kF32, Encoding::kBF16};
constexpr std::array kMxfp8 = {Encoding::kMxfp8};
constexpr std::array kNvfp4 = {Encoding::kNvfp4};
constexpr std::array kSplitLayout = {Layout::kSplit};
constexpr std::array kAtoms = {Layout::kCutlassScaleAtoms};
constexpr std::array kRowsLayout = {Layout::kRowInterleaved};
constexpr std::array kNvfp4Ggml = {Layout::kGgmlNvfp4Blocks};
// GGML's expert products over NVFP4 blocks (the ModelOpt import's GGML
// layout).
constexpr std::array kQuantizedNvfp4 = {Encoding::kNvfp4};

using Rule = ex::OperandRule;
constexpr std::array<Rule, 1> kF32In = {{{.operand = 0, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kF32Binary = {{{.operand = 0, .encodings = kF32, .layouts = kDense},
                                             {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kAddRules = {
    {{.operand = 0, .encodings = kF32F16, .layouts = kDense},
     {.operand = 1, .encodings = kF32F16, .layouts = kDense}}};
constexpr std::array<Rule, 4> kIndexScoreRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 3> kVisibleTopKRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kSelectedMaskRules = {
    {{.operand = 0, .encodings = kF16, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF16, .layouts = kDense}}};
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
// The zero-copy bank recipe computes the same F32 attention result. Rounded
// fused outputs retain the ordinary Attention + Convert decomposition.
constexpr std::array<Rule, 5> kStateBankAttention = {
    {kF16Attention[0],
     kF16Attention[1],
     kF16Attention[2],
     kF16Attention[3],
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
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
// fused.owner_attention's (q, mask, k0, v0 or k1, ...): the first four;
// the predicate checks every root.
constexpr std::array<Rule, 4> kOwnerAttention = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF16, .layouts = kDense},
     {.operand = 3, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 1> kMaskPositions = {
    {{.operand = 0, .encodings = kI32, .layouts = kDense}}};
// kMatMulId's (as, b, ids).
constexpr std::array<Rule, 3> kQuantExperts = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kNvfp4Experts = {
    {{.operand = 0, .encodings = kQuantizedNvfp4, .layouts = kNvfp4Ggml},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense}}};
// The fast forms' products (w, x) and their results.
constexpr std::uint8_t kOut = ex::kOutputOperand;
constexpr std::array<Rule, 3> kMxfp8Vec = {
    {{.operand = 0, .encodings = kMxfp8, .layouts = kSplitLayout},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kMxfp8Gemm = {
    {{.operand = 0, .encodings = kMxfp8, .layouts = kSplitLayout},
     {.operand = 1, .encodings = kMxfp8, .layouts = kAtoms},
     {.operand = kOut, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kMxfp8Prepare = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = kOut, .encodings = kMxfp8, .layouts = kAtoms}}};
constexpr std::array<Rule, 3> kBf16Product = {
    {{.operand = 0, .encodings = kBf16, .layouts = kDense},
     {.operand = 1, .encodings = kBf16, .layouts = kDense},
     {.operand = kOut, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kFloatConvert = {
    {{.operand = 0, .encodings = kF32F16, .layouts = kDense},
     {.operand = kOut, .encodings = kF32F16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kToBf16 = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kNvfp4Table = {
    {{.operand = 0, .encodings = kNvfp4, .layouts = kRowsLayout},
     {.operand = 1, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kQRowsTable = {
    {{.operand = 0, .encodings = {}, .layouts = kGgml},
     {.operand = 1, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kHcPrepRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kHcLoRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kHcFoldRules = {
    {{.operand = 0, .encodings = kBf16, .layouts = kDense},
     {.operand = 1, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 3> kHcCombineRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kHcNormRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kGdnConvRules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kSsmConvRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 1> kRowsRules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 3> kNormGateRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 3> kHeadGateRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kMxfp8, .layouts = kAtoms}}};
constexpr std::array<Rule, 2> kQsaPoolRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 5> kQsaAttnRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF16, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = 4, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kMoeRouterRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kCutlassExperts = {
    {{.operand = 0, .encodings = kNvfp4, .layouts = kAtoms},
     {.operand = 1, .encodings = kNvfp4, .layouts = kAtoms},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kVecQProduct = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kQ8_1, .layouts = kGgml}}};
constexpr std::array<Rule, 3> kVecQGlu = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 2, .encodings = kQ8_1, .layouts = kGgml}}};
constexpr std::array<Rule, 3> kVecQExperts = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kQ8_1, .layouts = kGgml},
     {.operand = 2, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 4> kVecQExpertsGlu = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 2, .encodings = kQ8_1, .layouts = kGgml},
     {.operand = 3, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kGatedDeltaRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 5, .encodings = kF32, .layouts = {}}}};

// kRoute's logits; kExpertReduce's (x, w, ids, scale).
constexpr std::array<Rule, 1> kRouteLogits = kF32In;
constexpr std::array<Rule, 4> kScaledReduce = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense}}};

constexpr std::array<Rule, 3> kDsv4QHeadRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 6> kDsv4CompressRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = 4, .encodings = kI32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 8> kDsv4HcPreRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kFloats, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = 4, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense},
     {.operand = kOut + 1, .encodings = kF32, .layouts = kDense},
     {.operand = kOut + 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 6> kDsv4HcPostRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense},
     {.operand = kOut + 1, .encodings = kF32, .layouts = kDense}}};

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

// The GB10 (sm_121), the one device Gemma 4's owner attention runs on today.
constexpr std::uint32_t kOwnerGemma4Arch = 121;

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
  return Built(q) && r && PlainRope(*r);
}

// CheckGemmaNormRope's (gemma_norm.cc): a full NEOX rotation of D 256 or
// 512, its parameters finite and positive, with F32 frequency factors or
// none (upstream's fused launcher reads the rotation's; Gemma 4's global
// layers).
bool AcceptsNormRope(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormRopeParams>(q);
  const ex::OperandFormat* freq = Operand(q, 3);
  if (!Built(q) || !p || (q.k != 256 && q.k != 512) ||
      (freq != nullptr && (freq->encoding != Encoding::kF32 || freq->layout != Layout::kDense))) {
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
bool AcceptsStateBankAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::AttnParams>(q);
  const auto* raw = Operand(q, 1);
  const auto* compressed = Operand(q, 2);
  const auto* query = Operand(q, 0);
  const auto* mask = Operand(q, 3);
  const auto* sinks = Operand(q, 4);
  constexpr std::uint64_t limit = std::numeric_limits<std::int32_t>::max();
  if (!AcceptsAttention(q) || !p || p->sparse_any || p->softcap != 0 || p->max_bias != 0 ||
      p->n_kv_max > limit || !raw || !compressed || !query || !mask || query->ne[2] <= 4 ||
      query->ne[2] > limit || query->ne[1] > limit / 4 / q.k / query->ne[2] ||
      raw->ne[2] > limit / 2 || compressed->ne[2] > limit / 2 - raw->ne[2])
    return false;
  const auto cells = raw->ne[2] + compressed->ne[2];
  if (sinks &&
      (sinks->encoding != Encoding::kF32 || sinks->ne[0] != query->ne[2] || sinks->ne[1] != 1 ||
       sinks->ne[2] != 1 || sinks->ne[3] != 1 || query->ne[2] % 8 != 0))
    return false;
  return raw->ne[0] == q.k && compressed->ne[0] == q.k && raw->ne[1] == 1 &&
         compressed->ne[1] == 1 && raw->ne[3] == 1 && compressed->ne[3] == 1 && query->ne[3] == 1 &&
         cells % 256 == 0 && mask->ne[0] == cells && mask->ne[1] >= query->ne[1] &&
         mask->ne[2] == 1 && mask->ne[3] == 1;
}
bool AcceptsAttention256(const ex::CapabilityQuery& q) {
  return Built(q) && q.k == 256 && q.n == 256;
}

// llmp.gemma4.mask (llmp_ops.h Gemma4Mask): a causal or window mask; its
// producer derives the cells' positions from the ring's capacity.
bool AcceptsMask(const ex::CapabilityQuery& q) {
  const auto m = ParamsOf<ir::MaskParams>(q);
  return Built(q) && m &&
         (m->kind == ir::MaskKind::kCausal || m->kind == ir::MaskKind::kWindow ||
          m->kind == ir::MaskKind::kAhead);
}

// CheckFlashAttnOwners' contract (fattn_owner.h), over the node's extents:
// q [D, R, H, n], mask [C, Pad(R, 32), 1, n], each owner's K and V [D,
// C_o, H / (D 256: 2, 512: 8), 1] in F16. D 256 or 512 at 16 or 32 heads,
// or D 256 at 8 heads in a cohort of 2 to 4, whole 8 or 12 (four roots a
// node) or a partial 5 to 7 or 9 to 11 (its real groups, by offset);
// scale 1, no bias, sinks or sparse bound, a softcap of 0 or 50 (50 only
// for D 256, 8 heads, two owners alone). One query row: cells of 256 to
// 16,384 in steps of 256; 2 to 512 rows: two owners alone at D 256 and 8
// heads with every cache the mask's width, through 131,072 cells without
// a softcap and 16,384 with it. Caches narrower than the mask (bounded
// roots) only for two owners alone at one row, each a multiple of 256, the
// widest the mask's (the kernel's whole-twelve bounded carriers are not
// declared: no recipe selects them). 16 or 32 heads or D 512 (Gemma 4's
// forms) on the GB10 alone.
bool AcceptsOwnerAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::OwnerAttnParams>(q);
  const auto sources = std::ranges::count_if(
      q.operands, [](const ex::OperandFormat& o) { return o.operand < ex::kOutputOperand; });
  if (!Built(q) || !p || p->owners < 1 || p->owners > 4 ||
      sources != 2 + (2 * static_cast<std::ptrdiff_t>(p->owners)))
    return false;
  const ir::AttnParams& a = p->attn;
  if (a.scale != 1.0F || a.max_bias != 0.0F || a.sinks || a.sparse_any || a.n_kv_max != 0 ||
      (a.softcap != 0.0F && a.softcap != 50.0F))
    return false;
  const std::uint32_t n = p->owners;
  const std::uint32_t cohort = p->cohort == 0 ? n : p->cohort;
  const std::uint32_t offset = p->offset;
  const ex::OperandFormat* qf = Operand(q, 0);
  const ex::OperandFormat* mf = Operand(q, 1);
  if (qf == nullptr || mf == nullptr) return false;
  const std::uint64_t d = qf->ne[0], rows = qf->ne[1], heads = qf->ne[2], cells = mf->ne[0];
  const bool partial =
      cohort == 5 || cohort == 6 || cohort == 7 || cohort == 9 || cohort == 10 || cohort == 11;
  if (partial) {
    if (offset % 4 != 0 || offset >= cohort || n != std::min(4U, cohort - offset)) return false;
  } else if (cohort == 8 || cohort == 12) {
    if (n != 4 || offset % 4 != 0 || offset + n > cohort) return false;
  } else if (cohort < 2 || cohort > 4 || n != cohort || offset != 0) {
    return false;
  }
  if ((d != 256 && d != 512) || qf->ne[3] != n ||
      !(heads == 16 || heads == 32 || (d == 256 && heads == 8)))
    return false;
  // Gemma 4's forms (16 or 32 heads, or D 512) are qualified on the GB10
  // only: other devices keep independent attention.
  if (!(d == 256 && heads == 8) && q.arch != kOwnerGemma4Arch) return false;
  const bool cap50 = a.softcap == 50.0F;
  if (cap50 && (d != 256 || heads != 8 || n != 2 || cohort != 2)) return false;
  const std::uint64_t kv_heads = heads / (d == 256 ? 2 : 8);
  const std::uint64_t mask_rows = (rows + 31) / 32 * 32;
  if (mf->ne[1] != mask_rows || mf->ne[2] != 1 || mf->ne[3] != n || cells < 256 || cells % 256 != 0)
    return false;
  const bool lone_pair = n == 2 && cohort == 2;
  if (rows == 1) {
    if (cells > 16384) return false;
  } else if (rows < 2 || rows > 512 || d != 256 || heads != 8 || !lone_pair ||
             cells > (cap50 ? 16384U : 131072U)) {
    return false;
  }
  bool bounded = false;
  std::uint64_t widest = 0;
  for (std::uint32_t o = 0; o < 2 * n; ++o) {
    const ex::OperandFormat* kv = Operand(q, static_cast<std::uint8_t>(2 + o));
    if (kv == nullptr || kv->encoding != Encoding::kF16 || kv->layout != Layout::kDense ||
        kv->ne[0] != d || kv->ne[2] != kv_heads || kv->ne[3] != 1 || kv->ne[1] > cells ||
        kv->ne[1] < 256 || kv->ne[1] % 256 != 0)
      return false;
    // K and V of one owner read as many cells.
    if (o >= n && kv->ne[1] != Operand(q, static_cast<std::uint8_t>(2 + o - n))->ne[1])
      return false;
    bounded = bounded || kv->ne[1] != cells;
    widest = std::max(widest, kv->ne[1]);
  }
  return !bounded || (rows == 1 && lone_pair && widest == cells);
}

// GemmaRoutingFusionAt's (gemma_moe_fusion.cc) over kRoute: F32 logits of
// 128 experts and 1 to kGemmaMoeMaxRows rows, softmax, the top 8 renormalized
// (the clamp 2^-14 is the op's), scale 1, no bias, table or tokens.
bool AcceptsGemmaRoute(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RouteParams>(q);
  const ex::OperandFormat* logits = Operand(q, 0);
  return Built(q) && p && logits != nullptr && p->kind == ir::RouteKind::kSoftmaxTopK &&
         p->normalize && p->experts == 128 && p->used == 8 && p->scale == 1.0F && !Has(q, 1) &&
         !Has(q, 2) && !Has(q, 3) && logits->ne[0] == 128 && logits->ne[1] >= 1 &&
         logits->ne[1] <= static_cast<std::uint64_t>(kGemmaMoeMaxRows) && logits->ne[2] == 1 &&
         logits->ne[3] == 1;
}
// GemmaReductionFusionAt's over kExpertReduce: the experts' F32 outputs
// [2816, 8, rows] of 1 to kGemmaMoeMaxRows rows, the weights [8, rows] and
// each slot's scale already gathered [1, 8, rows] (the ids only name the
// slots' experts).
bool AcceptsGemmaReduce(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* x = Operand(q, 0);
  const ex::OperandFormat* w = Operand(q, 1);
  const ex::OperandFormat* s = Operand(q, 3);
  if (!Built(q) || x == nullptr || w == nullptr || s == nullptr) return false;
  const std::uint64_t rows = x->ne[2];
  using Ne = std::array<std::uint64_t, 4>;
  return rows >= 1 && rows <= static_cast<std::uint64_t>(kGemmaMoeMaxRows) &&
         x->ne == Ne{2816, 8, rows, 1} && w->ne == Ne{8, rows, 1, 1} && s->ne == Ne{1, 8, rows, 1};
}

bool AcceptsAdd(const ex::CapabilityQuery& q) {
  const auto* a = Operand(q, 0);
  const auto* b = Operand(q, 1);
  return Built(q) && a != nullptr && b != nullptr && a->encoding == b->encoding;
}

bool AcceptsConvert(const ex::CapabilityQuery& q) {
  const auto* from = Operand(q, 0);
  const auto* to = Operand(q, ex::kOutputOperand);
  if (!Built(q) || from == nullptr || to == nullptr || from->encoding == to->encoding) return false;
  std::uint64_t count = 1;
  for (const auto n : from->ne) {
    if (n == 0 || n > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) / count)
      return false;
    count *= n;
  }
  return true;
}

bool AcceptsIndexScore(const ex::CapabilityQuery& q) {
  const auto* query = Operand(q, 0);
  const auto* keys = Operand(q, 1);
  return Built(q) && Has(q, 3) && query != nullptr && keys != nullptr && query->ne[0] == 128 &&
         (query->ne[1] == 32 || query->ne[1] == 64) && query->ne[2] > 0 && query->ne[2] <= 65535 &&
         query->ne[3] > 0 && query->ne[3] <= 65535 && keys->ne[2] > 0 &&
         keys->ne[2] <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) - 64;
}

bool AcceptsArgmax(const ex::CapabilityQuery& q) { return Built(q); }
bool AcceptsTopK(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::TopKParams>(q);
  const auto* x = Operand(q, 0);
  constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  return Built(q) && p && x != nullptr && p->k > 0 && p->k <= x->ne[0] && x->ne[0] <= max &&
         x->ne[1] != 0 && x->ne[2] != 0 && x->ne[3] != 0 && x->ne[1] <= max / x->ne[2] / x->ne[3];
}

bool AcceptsVisibleTopK(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::TopKParams>(q);
  const auto* scores = Operand(q, 0);
  const auto* counts = Operand(q, 1);
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX);
  return Built(q) && p && scores && counts && p->k > 0 && p->k <= 65536 && scores->ne[0] > 0 &&
         scores->ne[0] <= max / 4 && scores->ne[1] > 0 && scores->ne[1] <= max / p->k &&
         scores->ne[2] == 1 && scores->ne[3] == 1 &&
         counts->ne == std::array<std::uint64_t, 4>{scores->ne[1], 1, 1, 1};
}
bool AcceptsSelectedMask(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::SelectedMaskParams>(q);
  const auto* raw = Operand(q, 0);
  const auto* ids = Operand(q, 1);
  if (!Built(q) || !p || !raw || !ids || p->cells == 0 || p->raw_cells == 0) return false;
  const auto width = std::uint64_t{p->raw_cells} + p->cells;
  return raw->ne[0] > 0 && raw->ne[0] <= p->raw_cells && raw->ne[1] > 0 && raw->ne[2] == 1 &&
         raw->ne[3] == 1 && ids->ne[0] > 0 &&
         ids->ne[0] <= static_cast<std::uint64_t>(INT32_MAX) / 4 && ids->ne[1] == raw->ne[1] &&
         ids->ne[2] == 1 && ids->ne[3] == 1 && width <= std::uint64_t{65535} * 4096 &&
         raw->ne[1] <= INT32_MAX / width;
}

// ------------------------------------------------- Qwen3.8's fast forms

// The architecture llmpalooza's CUTLASS tensor-core kernels (MXFP8 products,
// NVFP4 grouped GEMMs) are built for: the GB10 alone.
constexpr std::uint32_t kCutlassArch = 121;

std::uint64_t Ne(const ex::CapabilityQuery& q, std::uint8_t operand, std::size_t d) {
  const ex::OperandFormat* o = Operand(q, operand);
  return o == nullptr ? 0 : o->ne.at(d);
}

// ------------------------------------------------- DeepSeek's local forms

bool ShapedOperand(const ex::CapabilityQuery& q, std::uint8_t at,
                   std::array<std::uint64_t, 4> shape) {
  const auto* operand = Operand(q, at);
  return operand != nullptr && operand->ne == shape;
}

bool AcceptsDsv4QHead(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormRopeParams>(q);
  const auto tokens = Ne(q, 0, 2);
  if (!Built(q) || !p || tokens < 16 || tokens > 8192 ||
      !ShapedOperand(q, 0, {512, 64, tokens, 1}) || !ShapedOperand(q, 1, {tokens, 1, 1, 1}) ||
      !ShapedOperand(q, kOut, {512, 64, tokens, 1}) || Has(q, 2))
    return false;
  const auto& r = p->rope;
  const std::array values{p->eps,        r.base,      r.scale,    r.ext_factor,
                          r.attn_factor, r.beta_fast, r.beta_slow};
  return r.mode == ir::RopeMode::kNormal && !r.inverse && !r.table && r.dims == 64 &&
         r.offset == 448 &&
         r.original_context <=
             static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) &&
         std::ranges::all_of(values, [](float v) { return std::isfinite(v); }) && p->eps >= 0 &&
         r.base > 0 && r.scale > 0 && r.ext_factor >= 0 &&
         (r.ext_factor == 0 ||
          (r.original_context > 0 && r.beta_fast > 0 && r.beta_slow > 0 && r.base != 1));
}

bool AcceptsDsv4Compress(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::CompressorParams>(q);
  if (!Built(q) || !p || p->ratio < 1 || p->ratio > 1024) return false;
  const auto channels = Ne(q, 0, 0);
  const auto state = Ne(q, 0, 1);
  const auto tokens = Ne(q, 2, 1);
  const auto count = Ne(q, 4, 0);
  const std::uint64_t halves = p->overlap ? 2 : 1;
  const std::uint64_t per_block = halves * p->ratio;
  constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  return channels > 0 && channels / halves <= 65535 * 64 && channels % 2 == 0 && state > 0 &&
         tokens > 0 && count > 0 && state <= max / channels && tokens <= max / channels &&
         count % per_block == 0 && count / per_block <= 65535 &&
         ShapedOperand(q, 0, {channels, state, 1, 1}) &&
         ShapedOperand(q, 1, {channels, state, 1, 1}) &&
         ShapedOperand(q, 2, {channels, tokens, 1, 1}) &&
         ShapedOperand(q, 3, {channels, tokens, 1, 1}) && ShapedOperand(q, 4, {count, 1, 1, 1}) &&
         ShapedOperand(q, kOut, {channels / halves, 1, count / per_block, 1});
}

bool AcceptsDsv4HcPre(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::SinkhornPreParams>(q);
  const auto width = Ne(q, 0, 0);
  const auto tokens = Ne(q, 0, 2);
  return Built(q) && p && p->streams == 4 && p->iterations >= 1 && p->iterations <= 100 &&
         std::isfinite(p->eps) && p->eps >= 0 && std::isfinite(p->hc_eps) && p->hc_eps >= 0 &&
         (width == 4096 || width == 8192) && tokens >= 1 && tokens <= 65535 &&
         ShapedOperand(q, 0, {width, 4, tokens, 1}) && ShapedOperand(q, 1, {width * 4, 24, 1, 1}) &&
         ShapedOperand(q, 2, {3, 1, 1, 1}) && ShapedOperand(q, 3, {24, 1, 1, 1}) &&
         ShapedOperand(q, 4, {width, 1, 1, 1}) && ShapedOperand(q, kOut, {width, tokens, 1, 1}) &&
         ShapedOperand(q, kOut + 1, {4, tokens, 1, 1}) &&
         ShapedOperand(q, kOut + 2, {4, 4, tokens, 1});
}

bool AcceptsDsv4HcPost(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::HcNormPostParams>(q);
  const auto tokens = Ne(q, 0, 1);
  return Built(q) && p && p->streams == 4 && std::isfinite(p->eps) && p->eps > 0 && tokens >= 1 &&
         tokens <= 65535 && ShapedOperand(q, 0, {4096, tokens, 1, 1}) &&
         ShapedOperand(q, 1, {4096, 4, tokens, 1}) && ShapedOperand(q, 2, {4, tokens, 1, 1}) &&
         ShapedOperand(q, 3, {4, 4, tokens, 1}) && ShapedOperand(q, kOut, {4096, 4, tokens, 1}) &&
         ShapedOperand(q, kOut + 1, {16384, tokens, 1, 1});
}

// llmp.mxfp8.mul_mat_vec: up to kMxfp8VecColumns columns (a wave's to
// kMxfp8VecWaveColumns).
bool AcceptsMxfp8Vec(const ex::CapabilityQuery& q) {
  return Built(q) && q.rows >= 1 && q.rows <= static_cast<std::uint64_t>(kMxfp8VecColumns) &&
         q.k % 32 == 0;
}
// llmp.mxfp8.gemm.cutlass: past the vector product's columns, on the GB10.
bool AcceptsMxfp8Gemm(const ex::CapabilityQuery& q) {
  return Built(q) && q.arch == kCutlassArch &&
         q.rows > static_cast<std::uint64_t>(kMxfp8VecColumns) && q.k % 128 == 0;
}
bool AcceptsMxfp8Quantize(const ex::CapabilityQuery& q) {
  return Built(q) && q.arch == kCutlassArch && Ne(q, 0, 0) % 128 == 0 && Ne(q, 0, 2) <= 1 &&
         Ne(q, 0, 3) <= 1;
}
// llmp.nvfp4.get_rows: rows of 16-value blocks.
bool AcceptsNvfp4Rows(const ex::CapabilityQuery& q) { return Built(q) && Ne(q, 0, 0) % 16 == 0; }
// llmp.qrows.get_rows: a table of 32-value blocks whose rows are not GGML's
// whole 256-value steps (its get_rows takes those), at most 1,024 values.
bool AcceptsQRows(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* t = Operand(q, 0);
  if (!Built(q) || t == nullptr) return false;
  const bool type = t->encoding == Encoding::kQ4_0 || t->encoding == Encoding::kQ4_1 ||
                    t->encoding == Encoding::kQ5_0 || t->encoding == Encoding::kQ5_1 ||
                    t->encoding == Encoding::kQ8_0 || t->encoding == Encoding::kIQ4_NL;
  return type && t->ne[0] % 32 == 0 && t->ne[0] % 256 != 0 && t->ne[0] <= 1024;
}
// The hyper-connections' kernels: at most 8 streams, widths of 8.
bool AcceptsHcPrep(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* inject = Operand(q, 2);
  return Built(q) && Ne(q, 0, 1) >= 2 && Ne(q, 0, 1) <= 8 && Ne(q, 0, 0) % 8 == 0 &&
         Ne(q, 0, 0) <= 4096 &&
         (inject == nullptr ||
          (inject->encoding == Encoding::kBF16 && inject->layout == Layout::kDense));
}
bool AcceptsHcLo(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::ScaledUnaryParams>(q);
  if (!Built(q) || !p || p->fn != ir::UnaryFn::kSilu) return false;
  // HcLo scales by the reciprocal of its integer stream count. Rounding
  // an arbitrary scale to a count would silently change the operation.
  for (std::uint32_t streams = 1; streams <= 8; ++streams) {
    if (p->scale == 1.0F / static_cast<float>(streams)) return true;
  }
  return false;
}
bool AcceptsHcFold(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::HcFoldParams>(q);
  if (!Built(q) || !p || p->streams < 2 || p->streams > 8) return false;
  const std::uint64_t width = Ne(q, 0, 0) / p->streams;
  const std::uint64_t rows = Ne(q, 0, 1);
  // The prepared and rounded rows past the vector product's columns, of
  // whole 128-value MXFP8 rows (HcMixLayout), rounded only beside prepared.
  if (p->prepared && (rows <= static_cast<std::uint64_t>(kMxfp8VecColumns) || width % 128 != 0))
    return false;
  return !p->rounded || p->prepared;
}
bool AcceptsHcCombine(const ex::CapabilityQuery& q) {
  return Built(q) && Ne(q, 0, 0) % 4 == 0 && Ne(q, 0, 1) >= 2;
}
bool AcceptsHcNorm(const ex::CapabilityQuery& q) {
  return Built(q) && Ne(q, 0, 0) >= 1024 && Ne(q, 0, 1) >= 2;
}
// llmp.gdn.conv: 128-wide heads, a kernel of 4 (3 taps).
// GGML's convolution of one sequence (past 32 tokens not whole windows, the
// lowering pads its input, RE-032).
bool AcceptsSsmConv(const ex::CapabilityQuery& q) { return Built(q) && Ne(q, 0, 2) <= 1; }
bool AcceptsGdnConv(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::GdnConvParams>(q);
  return Built(q) && p && p->head == 128 && Ne(q, 2, 0) == 4;
}
bool AcceptsGdnHistory(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::GdnHistoryParams>(q);
  return Built(q) && p && p->taps >= 1 && p->taps <= 8;
}
// llmp.gdn.step: 128-wide heads, up to kGatedDeltaNetLanesTokens tokens.
bool AcceptsGdnStep(const ex::CapabilityQuery& q) {
  return Built(q) && Ne(q, 0, 0) == 128 && Ne(q, 2, 0) == 128 && Ne(q, 2, 2) >= 1 &&
         Ne(q, 2, 2) <= static_cast<std::uint64_t>(kGatedDeltaNetLanesTokens) && Ne(q, 2, 3) == 1;
}
// llmp.gdn.gates: 48 value heads, 1 to 16 rows.
bool AcceptsGdnGates(const ex::CapabilityQuery& q) {
  return Built(q) && Ne(q, 0, 0) == 48 && Ne(q, 0, 1) >= 1 && Ne(q, 0, 1) <= 16;
}
bool AcceptsGdnNormGate(const ex::CapabilityQuery& q) {
  return Built(q) && Ne(q, 0, 0) == 128 && Ne(q, 0, 3) <= 1;
}
// GGML's gated delta rule (upstream's or llmpalooza's column-blocked
// kernels, which PlanGraph picks by shape): one sequence.
bool AcceptsGatedDelta(const ex::CapabilityQuery& q) { return Built(q) && Ne(q, 2, 3) <= 1; }
// llmp.qsa.prep: the interleaved multi-section rotation of 64 dimensions,
// heads of whole warps from 64 to 512 values.
bool QsaRope(const ir::RopeParams& r) {
  // These kernels rotate at pos * base^(-2 / dims), without inverse,
  // frequency scaling, a table, an offset or YaRN amplitude correction.
  return r.mode == ir::RopeMode::kInterleavedMulti && r.dims == 64 && !r.inverse && !r.table &&
         r.offset == 0 && r.scale == 1.0F && r.ext_factor == 0.0F && r.attn_factor == 1.0F;
}

bool AcceptsQsaPrep(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormRopeParams>(q);
  const std::uint64_t d = Ne(q, 0, 0);
  return Built(q) && p && QsaRope(p->rope) && !Has(q, 3) && d >= 64 && d <= 512 && d % 32 == 0;
}
bool AcceptsQsaPool(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::QsaPoolParams>(q);
  const std::uint64_t d = Ne(q, 0, 0);
  return Built(q) && p && p->ratio >= 1 && p->ratio <= 32 && QsaRope(p->rope) && d >= 64 &&
         d <= 512 && d % 32 == 0 && Has(q, 5);
}
// llmp.qsa.topk and llmp.qsa.attn: 256-wide heads at most 16 to a KV head,
// an indexer of 4 heads of 128, the selection within its tiles.
bool AcceptsQsaAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::QsaAttnParams>(q);
  if (!Built(q) || !p || !Has(q, 8) || p->attn.softcap != 0.0F || p->attn.max_bias != 0.0F ||
      p->attn.window != 0 || p->attn.sinks)
    return false;
  const std::uint64_t heads = Ne(q, 0, 1);
  const std::uint64_t kv = Ne(q, 1, 2);
  const QsaTopKLayout layout{.t = static_cast<std::int64_t>(Ne(q, 0, 2)),
                             .n_blocks = static_cast<std::int64_t>(Ne(q, 4, 1)),
                             .width = static_cast<std::int64_t>(p->width),
                             .ratio = static_cast<std::int64_t>(p->ratio)};
  return Ne(q, 0, 0) == static_cast<std::uint64_t>(kQsaAttnHead) && kv >= 1 && heads % kv == 0 &&
         heads / kv <= 16 && Ne(q, 3, 0) == static_cast<std::uint64_t>(kQsaIndexDim) &&
         Ne(q, 3, 1) == static_cast<std::uint64_t>(kQsaIndexHeads) && p->ratio >= 1 &&
         p->ratio <= 32 && layout.tiles() <= kQsaTopKMaxTiles &&
         layout.tiles() * layout.candidates() <= kQsaTopKCandidates;
}
bool AcceptsHeadGate(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::HeadGateParams>(q);
  return Built(q) && q.arch == kCutlassArch && p && p->head % 32 == 0;
}
bool AcceptsMoeRouter(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RouteParams>(q);
  const ex::OperandFormat* gate = Operand(q, 2);
  return Built(q) && p && p->kind == ir::RouteKind::kSoftmaxTopK && p->normalize &&
         p->scale == 1.0F && p->experts % 32 == 0 && p->experts <= 1024 && p->used <= 32 &&
         gate != nullptr && (gate->encoding == Encoding::kBF16 || gate->encoding == Encoding::kF32);
}
bool AcceptsMoeExperts(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::MoeExpertsParams>(q);
  return Built(q) && q.arch == kCutlassArch && p && p->glu.kind == ir::GluKind::kSwiGlu &&
         !p->glu.swapped && Has(q, 7) && Has(q, 8) && Has(q, 9);
}
bool AcceptsMoeGlu(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::GluParams>(q);
  return Built(q) && p && p->kind == ir::GluKind::kSwiGlu && !p->swapped;
}
// llmp.vecq: decode's quantized products (up to kVecQTokens tokens), of a
// type it has a kernel for.
bool VecQ(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* w = Operand(q, 0);
  if (w == nullptr) return false;
  const format::EncodingInfo& info = format::Info(w->encoding);
  return Built(q) && info.ggml.has_value() && VecQType(static_cast<ggml_type>(info.ggml->type)) &&
         q.k % 32 == 0;
}
bool AcceptsVecQ(const ex::CapabilityQuery& q) {
  return VecQ(q) && q.rows >= 1 && q.rows <= static_cast<std::uint64_t>(kVecQTokens);
}
// Routed: its (token, slot) pairs within the kernel's 64 (the fast form's).
bool AcceptsVecQRouted(const ex::CapabilityQuery& q) {
  const std::uint64_t used = Ne(q, q.op == ir::Fused(FusedKind::kMatMulIdGlu) ? 3 : 2, 0);
  return VecQ(q) && q.rows >= 1 && used >= 1 && q.rows * used <= 64;
}
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
                             ex::ShapeRule shape = {}, std::uint16_t batch = 0,
                             std::span<const std::string_view> launches = {},
                             bool needs_state_span = false) {
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
          .accepts = accepts,
          .launches = launches,
          .needs_state_span = needs_state_span};
}

constexpr ir::OpKey P(Op op) { return ir::Primitive(op); }
constexpr ir::OpKey F(FusedKind kind) { return ir::Fused(kind); }

constexpr std::string_view kMulMatGroup = "ggml.mul_mat";
constexpr std::string_view kGatedDeltaGroup = "ggml.gated_delta_net";
// The IR implementations whose nodes launch several of the module's steps,
// or a step named otherwise (Capability::launches).
constexpr std::string_view kQsaAttentionName = "llmp.qsa.topk_attn";
constexpr std::string_view kMoeExpertsName = "llmp.moe.experts.cutlass";
constexpr std::string_view kVecQDenseName = "llmp.vecq.dense";
constexpr std::string_view kVecQGluName = "llmp.vecq.glu";
constexpr std::string_view kVecQExpertsName = "llmp.vecq.experts";
constexpr std::string_view kVecQExpertsGluName = "llmp.vecq.experts_glu";
constexpr std::array<std::string_view, 2> kMxfp8GemmSteps = {kMxfp8SwizzleName, kMxfp8GemmName};
constexpr std::array<std::string_view, 2> kQsaAttentionSteps = {kQsaTopKName, kQsaAttnName};
// A padded window's copy and concatenation, then the convolution.
constexpr std::array<std::string_view, 3> kSsmConvSteps = {kContName, kConcatName, kSsmConvName};
constexpr std::array<std::string_view, 7> kMoeExpertsSteps = {
    kMoeGemvName, kMoeCombineName,     kMoeRouteName,        kMoeQuantizeName,
    kMoeGemmName, kMoeGluQuantizeName, kMoeCombineSortedName};
constexpr std::array<std::string_view, 1> kVecQSteps = {kVecQName};
// GGML's expert products over NVFP4 blocks: upstream's routing picks MMVQ
// or MMQ (ggml_cuda_mul_mat_id).
// The pre's two kernels and packed copies of its tiny strided carries.
constexpr std::string_view kDsv4HcPrePackedName = "llmp.dsv4.hc_pre_packed";
constexpr std::array<std::string_view, 3> kDsv4HcPreSteps = {kDsv4HcMixName, kDsv4HcPreName,
                                                             kContName};
constexpr std::string_view kMulMatIdNvfp4Name = "ggml.mul_mat_id.nvfp4";
constexpr std::array<std::string_view, 2> kMulMatIdSteps = {kMulMatIdVecQ, kMulMatIdQ};
constexpr std::string_view kAttentionGroup = "ggml.flash_attn_ext";
constexpr std::string_view kMulMatIdGroup = "ggml.mul_mat_id";
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
    Entry{"ggml.add", Operation::kAdd, Cap(P(Op::kAdd), kAddRules, &AcceptsAdd, true), kBuilt},
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
    Entry{kLightningIndexerName, Operation::kLightningIndexer,
          Cap(P(Op::kIndexScore), kIndexScoreRules, &AcceptsIndexScore, true),
          "; selectable for the build's SASS architectures: D 128, 32 or 64 indexer heads"},
    Entry{kVisibleTopKName, Operation::kTopK,
          Cap(P(Op::kVisibleTopK), kVisibleTopKRules, &AcceptsVisibleTopK, true), kBuilt},
    Entry{kSelectedMaskName, Operation::kFill,
          Cap(P(Op::kSelectedMask), kSelectedMaskRules, &AcceptsSelectedMask, true), kBuilt},
    Entry{kTopKName, Operation::kTopK, Cap(P(Op::kTopK), kF32In, &AcceptsTopK, true),
          "; selectable for the build's SASS architectures: F32 rows within I32 counts"},
    Entry{"ggml.rms_norm", Operation::kRmsNorm, Cap(P(Op::kRmsNorm), kF32In, &AcceptsAny, true),
          kBuilt},
    Entry{"ggml.rms_norm_mul.fused", Operation::kRmsNormMul,
          Cap(F(FusedKind::kRmsNormMul), kNormMul, &AcceptsAny, true), kBuilt},
    Entry{kGemmaNormRopeName, Operation::kRmsNormMulRope,
          Cap(F(FusedKind::kRmsNormMulRope), kNormMulRope, &AcceptsNormRope, true),
          "; selectable for the build's SASS architectures: a full NEOX rotation of D 256 or 512 "
          "with F32 frequency factors or none, its parameters finite"},
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
    // Routed experts: the expert products by upstream's routing (MMVQ or
    // MMQ, ggml_cuda_mul_mat_id), and Gemma's fused routing and scaled
    // reduce (gemma_moe_fusion.h).
    Entry{kMulMatIdVecQ, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup, 1), kBuilt},
    Entry{kMulMatIdQ, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kGemmaRouteName, Operation::kGemmaRoute,
          Cap(P(Op::kRoute), kRouteLogits, &AcceptsGemmaRoute, true),
          "; selectable for the build's SASS architectures: a normalized softmax top 8 of 128 "
          "experts, scale 1, of at most 8,192 rows"},
    Entry{kGemmaReduceName, Operation::kGemmaScaledReduce,
          Cap(P(Op::kExpertReduce), kScaledReduce, &AcceptsGemmaReduce, true),
          "; selectable for the build's SASS architectures: 8 slots of 2,816 values, each slot's "
          "scale gathered, of at most 8,192 rows"},
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
    Entry{"ggml.state_bank_attention", Operation::kFlashAttn,
          Cap(F(FusedKind::kStateBankAttention), kStateBankAttention, &AcceptsStateBankAttention,
              false, kAttentionGroup, 0, {}, 0, {}, true),
          "; the existing flash-attention group over a proven contiguous F16 state span"},
    Entry{"ggml.flash_attn_ext.vec_d256", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 2),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma_gqa2", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 1),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention, false, kAttentionGroup),
          "; selectable for the build's SASS architectures at head size 256 or 512"},
    Entry{kFlashAttnOwnersName, Operation::kFlashAttn,
          Cap(F(FusedKind::kOwnerAttention), kOwnerAttention, &AcceptsOwnerAttention, false),
          "; selectable for the build's SASS architectures: CheckFlashAttnOwners' Gemma "
          "cohorts, head sizes, cells and rows over independent F16 caches, a softcap of 0 or "
          "50, and bounded roots for two owners alone; 16 or 32 heads or D 512 (Gemma 4's) on "
          "sm_121 only"},
    Entry{kGemma4MaskName, Operation::kFill,
          Cap(P(Op::kMaskBuild), kMaskPositions, &AcceptsMask, true),
          "; selectable for the build's SASS architectures: causal, window and ahead masks"},
    Entry{"llmp.argmax", Operation::kTopK, Cap(P(Op::kArgmax), kF32In, &AcceptsArgmax, true),
          kBuilt},
    Entry{kDsv4QHeadName, Operation::kRope,
          Cap(F(FusedKind::kDsv4QHead), kDsv4QHeadRules, &AcceptsDsv4QHead, true),
          "; packed F32 [512, 64, 16..8192] Q heads and normal rotary tail"},
    Entry{kDsv4CompressName, Operation::kSoftMax,
          Cap(F(FusedKind::kDsv4Compress), kDsv4CompressRules, &AcceptsDsv4Compress, true),
          "; F32 compressor values/scores, ratio 1..1024, at most 65535 blocks"},
    Entry{kDsv4HcPrePackedName, Operation::kHcPre,
          Cap(F(FusedKind::kDsv4HcPre), kDsv4HcPreRules, &AcceptsDsv4HcPre, true, {}, 0, {}, 0,
              kDsv4HcPreSteps),
          "; full Sinkhorn carry, four F32 streams, width 4096 or 8192; packed carry copies"},
    Entry{kDsv4HcPostNormF16Name, Operation::kHcPost,
          Cap(F(FusedKind::kHcNormPost), kDsv4HcPostRules, &AcceptsDsv4HcPost, true),
          "; four 4096-value streams, post and flat F32 norm"},
    // Qwen3.8's fast forms (slice G3b; qwen38_graph.h): the MXFP8 and BF16
    // products, the n-gram tables' rows, the hyper-connections, Gated
    // DeltaNet, QSA and the routed experts.
    Entry{kMxfp8MulMatVecName, Operation::kMatMul,
          Cap(P(Op::kMatMul), kMxfp8Vec, &AcceptsMxfp8Vec, false, {}, 4),
          "; selectable for the build's SASS architectures: up to 8 columns"},
    Entry{kMxfp8GemmName, Operation::kMatMul,
          Cap(P(Op::kMatMul), kMxfp8Gemm, &AcceptsMxfp8Gemm, false, {}, 4, {}, 0, kMxfp8GemmSteps),
          "; selectable on sm_121 past 8 columns, K of whole 128-value rows"},
    Entry{kMxfp8QuantizeName, Operation::kQuantize,
          Cap(P(Op::kPrepare), kMxfp8Prepare, &AcceptsMxfp8Quantize, true),
          "; selectable on sm_121: rows of whole 128 values"},
    Entry{kGemmBf16Name, Operation::kMatMul,
          Cap(P(Op::kMatMul), kBf16Product, &AcceptsAny, false, {}, 4), kBuilt},
    Entry{"ggml.convert", Operation::kConvert,
          Cap(P(Op::kConvert), kFloatConvert, &AcceptsConvert, true),
          "; selectable for the build's SASS architectures: F16/F32 casts within I32 counts"},
    Entry{kBf16Name, Operation::kConvert, Cap(P(Op::kConvert), kToBf16, &AcceptsAny, true), kBuilt},
    Entry{kNvfp4RowsName, Operation::kGetRows,
          Cap(P(Op::kGetRows), kNvfp4Table, &AcceptsNvfp4Rows, true, {}, 2), kBuilt},
    Entry{kQRowsName, Operation::kGetRows,
          Cap(P(Op::kGetRows), kQRowsTable, &AcceptsQRows, true, {}, 2),
          "; selectable for the build's SASS architectures: 32-value blocks of rows GGML's "
          "get_rows does not take"},
    Entry{kHcPrepName, Operation::kHcNorm,
          Cap(F(FusedKind::kHcPrep), kHcPrepRules, &AcceptsHcPrep, true),
          "; selectable for the build's SASS architectures: 2 to 8 streams of widths of 8 to "
          "4,096, a BF16 inject"},
    Entry{kHcLoName, Operation::kUnary,
          Cap(F(FusedKind::kScaledUnary), kHcLoRules, &AcceptsHcLo, true),
          "; selectable for the build's SASS architectures: SiLU scaled by the reciprocal of "
          "1 to 8 streams"},
    Entry{kHcMixBf16Name, Operation::kHcMix,
          Cap(F(FusedKind::kHcFold), kHcFoldRules, &AcceptsHcFold, true),
          "; selectable for the build's SASS architectures: 2 to 8 streams; prepared past 8 "
          "rows of whole 128-value rows"},
    Entry{kHcCombineName, Operation::kHcCombine,
          Cap(F(FusedKind::kHcCombine), kHcCombineRules, &AcceptsHcCombine, true), kBuilt},
    Entry{kHcNormName, Operation::kHcNorm,
          Cap(F(FusedKind::kHcNorm), kHcNormRules, &AcceptsHcNorm, true),
          "; selectable for the build's SASS architectures: streams of at least 1,024"},
    Entry{kHcMixName, Operation::kHcMix,
          Cap(F(FusedKind::kHcMixNorm), kHcNormRules, &AcceptsHcNorm, true),
          "; selectable for the build's SASS architectures: streams of at least 1,024"},
    Entry{kSsmConvName, Operation::kSsmConv,
          Cap(P(Op::kSsmConv), kSsmConvRules, &AcceptsSsmConv, true, {}, 0, {}, 0, kSsmConvSteps),
          "; one sequence; past 32 tokens of a partial window, its input padded"},
    Entry{kGdnConvName, Operation::kSsmConv,
          Cap(F(FusedKind::kGdnConv), kGdnConvRules, &AcceptsGdnConv, true),
          "; selectable for the build's SASS architectures: heads of 128, a kernel of 4"},
    Entry{kGdnHistoryName, Operation::kCont,
          Cap(F(FusedKind::kGdnHistory), kRowsRules, &AcceptsGdnHistory, true), kBuilt},
    Entry{kGdnStepName, Operation::kGatedDeltaNet,
          Cap(F(FusedKind::kGdnStep), kGatedDeltaRules, &AcceptsGdnStep, false),
          "; selectable for the build's SASS architectures: heads of 128, up to 16 tokens"},
    Entry{kGdnGatesName, Operation::kUnary,
          Cap(F(FusedKind::kGdnGates), kHcNormRules, &AcceptsGdnGates, true),
          "; selectable for the build's SASS architectures: 48 heads, up to 16 rows"},
    Entry{kGdnNormGateName, Operation::kNormGate,
          Cap(F(FusedKind::kNormGate), kNormGateRules, &AcceptsGdnNormGate, true),
          "; selectable for the build's SASS architectures: heads of 128"},
    Entry{kGatedDeltaNetName, Operation::kGatedDeltaNet,
          Cap(P(Op::kGatedDeltaNet), kGatedDeltaRules, &AcceptsGatedDelta, false, kGatedDeltaGroup),
          kBuilt},
    Entry{kGatedDeltaNetColumnsName, Operation::kGatedDeltaNet,
          Cap(P(Op::kGatedDeltaNet), kGatedDeltaRules, &AcceptsGatedDelta, false, kGatedDeltaGroup,
              1),
          kBuilt},
    Entry{kGatedDeltaNetLanesName, Operation::kGatedDeltaNet,
          Cap(P(Op::kGatedDeltaNet), kGatedDeltaRules, &AcceptsGatedDelta, false, kGatedDeltaGroup,
              2),
          kBuilt},
    Entry{kQsaPrepName, Operation::kRope,
          Cap(F(FusedKind::kRmsNormMulRope), kNormMulRope, &AcceptsQsaPrep, true, {}, 1),
          "; selectable for the build's SASS architectures: the interleaved multi-section "
          "rotation of 64 dimensions over heads of 64 to 512, without frequency or amplitude "
          "scaling; lowering requires typed replicated position sections"},
    Entry{kQsaPoolName, Operation::kRope,
          Cap(F(FusedKind::kQsaPool), kQsaPoolRules, &AcceptsQsaPool, true),
          "; selectable for the build's SASS architectures: blocks of 1 to 32 cells, with a "
          "forward unscaled 64-dimension interleaved rotation; lowering requires canonical "
          "typed block positions/cells and stored raw keys"},
    Entry{kQsaAttentionName, Operation::kFlashAttn,
          Cap(F(FusedKind::kQsaAttention), kQsaAttnRules, &AcceptsQsaAttention, false, {}, 0, {}, 0,
              kQsaAttentionSteps),
          "; selectable for the build's SASS architectures: heads of 256, at most 16 a KV head, "
          "an indexer of 4 heads of 128, the selection within its tiles; lowering requires "
          "canonical typed position tables, stores and causal masks"},
    Entry{kQsaGateQuantizeName, Operation::kQuantize,
          Cap(F(FusedKind::kHeadGate), kHeadGateRules, &AcceptsHeadGate, true),
          "; selectable on sm_121: heads of whole 32 values into MXFP8 rows"},
    Entry{kMoeRouterName, Operation::kArgsort,
          Cap(F(FusedKind::kMoeRouter), kMoeRouterRules, &AcceptsMoeRouter, true),
          "; selectable for the build's SASS architectures: a normalized softmax top-k of up to "
          "32 of up to 1,024 experts"},
    Entry{kMoeExpertsName, Operation::kMulMatId,
          Cap(F(FusedKind::kMoeExperts), kCutlassExperts, &AcceptsMoeExperts, false, {}, 0, {}, 0,
              kMoeExpertsSteps),
          "; selectable on sm_121: the CUTLASS layout's experts with their scales, SwiGLU"},
    Entry{kMoeGluName, Operation::kMoeGlu,
          Cap(F(FusedKind::kMoeGlu), kF32Binary, &AcceptsMoeGlu, true), kBuilt},
    Entry{kMoeCombineName, Operation::kMoeCombine,
          Cap(F(FusedKind::kMoeCombine), kF32In, &AcceptsAny, true), kBuilt},
    Entry{kVecQDenseName, Operation::kMatMul,
          Cap(P(Op::kMatMul), kVecQProduct, &AcceptsVecQ, false, {}, 3, {}, 0, kVecQSteps),
          "; selectable for the build's SASS architectures: up to 8 tokens"},
    Entry{kVecQGluName, Operation::kMulMatGlu,
          Cap(F(FusedKind::kMatMulGlu), kVecQGlu, &AcceptsVecQ, false, {}, 3, {}, 0, kVecQSteps),
          "; selectable for the build's SASS architectures: up to 8 tokens"},
    Entry{kVecQExpertsName, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kVecQExperts, &AcceptsVecQRouted, false, {}, 3, {}, 0, kVecQSteps),
          "; selectable for the build's SASS architectures: up to 64 (token, slot) pairs"},
    Entry{kVecQExpertsGluName, Operation::kMulMatId,
          Cap(F(FusedKind::kMatMulIdGlu), kVecQExpertsGlu, &AcceptsVecQRouted, false, {}, 3, {}, 0,
              kVecQSteps),
          "; selectable for the build's SASS architectures: up to 64 (token, slot) pairs"},
    Entry{kMulMatIdNvfp4Name, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kNvfp4Experts, &AcceptsAny, false, {}, 0, {}, 0, kMulMatIdSteps),
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
