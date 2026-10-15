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
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/gemma_moe.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/rounded_index_score.h"
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
constexpr std::array<Rule, 4> kSpatialConvRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kSpatialUpRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
// A weighted stream fold has an F32 result even when a caller permits
// rounded IR outputs. Gated folds retain their primitive decomposition.
constexpr std::array<Rule, 3> kWeightedHcPreRules = {
    {kF32Binary[0],
     kF32Binary[1],
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 4> kSinkhornHcMixRules = {
    {kF32Binary[0],
     kF32Binary[1],
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kAddRules = {
    {{.operand = 0, .encodings = kF32F16, .layouts = kDense},
     {.operand = 1, .encodings = kF32F16, .layouts = kDense}}};
constexpr std::array<Rule, 4> kIndexScoreRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 5> kRoundedIndexScoreRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kVisibleTopKRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kI32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kPrefixOverwriteRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
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
// The explicitly tuned vector product keeps F32 activation values.
constexpr std::array<Rule, 3> kBf16VectorProduct = {
    {{.operand = 0, .encodings = kBf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kPreparedProduct = {
    {{.operand = 0, .encodings = kPreparedWeights, .layouts = kGgml},
     {.operand = 1, .encodings = kQ8_1, .layouts = kGgml}}};
// fused.mul_mat_glu's (w_gate, w_up, x).
constexpr std::array<Rule, 3> kQuantGlu = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 4> kF16ProductAdd = {
    {{.operand = 0, .encodings = kF16, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 4> kF16ProductGlu = {
    {{.operand = 0, .encodings = kF16, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
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
    {{.operand = 0, .encodings = kFloats, .layouts = kDense},
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
constexpr std::array kF16Q8 = {Encoding::kF16, Encoding::kQ8_0};
constexpr std::array kDenseGgml = {Layout::kDense, Layout::kGgmlBlocks};
constexpr std::array<Rule, 5> kQsaAttnRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16Q8, .layouts = kDenseGgml},
     {.operand = 2, .encodings = kF16Q8, .layouts = kDenseGgml},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = 4, .encodings = kBf16, .layouts = kDense}}};
constexpr std::array<Rule, 3> kSelectedRouteRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kI32, .layouts = kDense},
     {.operand = kOut + 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 5> kSelectedCombineRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kI32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 2> kMoeRouterRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kCutlassExperts = {
    {{.operand = 0, .encodings = kNvfp4, .layouts = kAtoms},
     {.operand = 1, .encodings = kNvfp4, .layouts = kAtoms},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 3> kGroupedVecQRules = {
    {{.operand = 0, .encodings = kQuantized, .layouts = kGgml},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};
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
constexpr std::array<Rule, 6> kDsv4LidMaskRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kI32, .layouts = kDense},
     {.operand = 4, .encodings = kF16, .layouts = kDense},
     {.operand = kOut, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 8> kDsv4HcPreRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kFloats, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = 4, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense},
     {.operand = kOut + 1, .encodings = kF32, .layouts = kDense},
     {.operand = kOut + 2, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 5> kHcPostRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 6> kDsv4HcPostRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense},
     {.operand = kOut + 1, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 4> kDsv4OutARules = {
    {{.operand = 0, .encodings = kQ8, .layouts = kGgml},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense},
     {.operand = kOut, .encodings = kF32, .layouts = kDense}}};

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
// A deliberately narrow, packed two-dimensional descriptor domain. The
// physical column count comes from the actual operands, never owner counts.
// Tuning is a separate exact-shape opt-in; ordinary products are unchanged.
bool AcceptsBf16VectorProduct(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::ProductParams>(q);
  if (!Built(q) || !p || q.params.size() > ir::kMaxParamBytes ||
      !std::ranges::all_of(q.params.subspan(sizeof(*p)),
                           [](std::byte value) { return value == std::byte{0}; }) ||
      p->accumulate != ir::Accumulate::kDefault || p->hint != ir::ProductHint::kNone ||
      p->reserved[0] || p->reserved[1] || q.rows < 1 || q.rows > 8 || q.k == 0 || q.k % 2 != 0 ||
      q.n == 0 || q.n % 2 != 0)
    return false;
  const auto packed = [&](std::uint8_t index, Encoding encoding, std::uint64_t columns,
                          std::uint64_t rows, std::uint64_t element) {
    const auto* op = Operand(q, index);
    constexpr auto max = static_cast<std::uint64_t>(INT32_MAX);
    if (!op || op->encoding != encoding || op->layout != Layout::kDense || !op->memory ||
        columns > max / element || rows > max / (columns * element) ||
        op->ne != std::array<std::uint64_t, 4>{columns, rows, 1, 1})
      return false;
    const auto pitch = columns * element, bytes = pitch * rows;
    const auto& m = *op->memory;
    const auto alignment = 2 * element;
    return m.nb == std::array<std::uint64_t, 4>{element, pitch, bytes, bytes} &&
           m.root_alignment >= alignment && m.root_alignment % alignment == 0 &&
           m.view_offset % alignment == 0;
  };
  // Require the output-typed pin, even though legacy TunedShape entries may
  // omit it for other implementations.
  const bool pinned = std::ranges::any_of(q.tuned, [&](const ex::TunedShape& t) {
    return t.implementation == kMulMatVecFRows && t.rows == q.rows && t.k == q.k && t.n == q.n &&
           t.output_encoding == Encoding::kF32;
  });
  return pinned && packed(0, Encoding::kBF16, q.k, q.n, 2) &&
         packed(1, Encoding::kF32, q.k, q.rows, 4) &&
         packed(ex::kOutputOperand, Encoding::kF32, q.n, q.rows, 4);
}

bool AcceptsFloatProductFusion(const ex::CapabilityQuery& q) {
  const auto* w = Operand(q, 0);
  // Upstream MMVF's one-column F16 case on NVIDIA: even K and a
  // single matrix make every dense row pitch half2 aligned and src0_small.
  return Built(q) && q.rows == 1 && q.k % 2 == 0 && w != nullptr && w->ne[2] == 1 && w->ne[3] == 1;
}
bool AcceptsFloatSwiGlu(const ex::CapabilityQuery& q) {
  const auto g = ParamsOf<ir::GluParams>(q);
  return AcceptsFloatProductFusion(q) && g && g->kind == ir::GluKind::kSwiGlu && !g->swapped;
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

bool AcceptsAttentionF32(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::AttnParams>(q);
  const auto* query = Operand(q, 0);
  const auto* key = Operand(q, 1);
  const auto* value = Operand(q, 2);
  const auto* mask = Operand(q, 3);
  if (!Built(q) || q.arch < 75 || q.k != 128 || q.n != 128 || !p || !query || !key || !value ||
      !mask || Has(q, 4) || p->max_bias != 0 || p->softcap != 0 || p->sparse_any ||
      p->n_kv_max != 0)
    return false;
  return query->ne[0] == 128 && key->ne[0] == 128 && value->ne == key->ne &&
         query->ne[2] == key->ne[2] && query->ne[3] == 1 && key->ne[3] == 1 &&
         mask->ne[0] == key->ne[1] && mask->ne[1] == query->ne[1] && mask->ne[2] == 1 &&
         mask->ne[3] == 1;
}
constexpr std::array<Rule, 5> kF32Attention = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};

// The F16 flash-attention family (graph_plan.cc): D 256 (the GQA-2 pair,
// vector or tensor-core, by device) or 512, K's head size V's.
bool AcceptsAttention(const ex::CapabilityQuery& q) {
  return Built(q) && (q.k == 256 || q.k == 512) && q.n == q.k;
}
// A declared sparse-at-any-length D512 recipe can share each query tile's
// visible KV union. The ordinary attention declarations keep their defaults.
bool AcceptsWideAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::AttnParams>(q);
  return AcceptsAttention(q) && q.k == 512 && p && p->sparse_any && p->n_kv_max > 0 &&
         p->n_kv_max <= std::numeric_limits<std::int32_t>::max();
}
bool AcceptsStateBankAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::AttnParams>(q);
  const auto* raw = Operand(q, 1);
  const auto* compressed = Operand(q, 2);
  const auto* query = Operand(q, 0);
  const auto* mask = Operand(q, 3);
  const auto* sinks = Operand(q, 4);
  constexpr std::uint64_t limit = std::numeric_limits<std::int32_t>::max();
  if (!AcceptsAttention(q) || !p || (p->sparse_any && (q.k != 512 || p->n_kv_max == 0)) ||
      p->softcap != 0 || p->max_bias != 0 || p->n_kv_max > limit || !raw || !compressed || !query ||
      !mask || query->ne[2] <= 4 || query->ne[2] > limit ||
      query->ne[1] > limit / 4 / q.k / query->ne[2] || raw->ne[2] > limit / 2 ||
      compressed->ne[2] > limit / 2 - raw->ne[2])
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
bool AcceptsAttention64(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::AttnParams>(q);
  const auto* query = Operand(q, 0);
  const auto* key = Operand(q, 1);
  const auto* value = Operand(q, 2);
  const auto* mask = Operand(q, 3);
  if (!Built(q) || q.k != 64 || q.n != 64 || !p || !query || !key || !value || !mask || Has(q, 4) ||
      p->sparse_any || p->softcap != 0 || p->max_bias != 0 ||
      p->accumulate != ir::Accumulate::kF32 || !(p->scale > 0) ||
      !(p->scale < std::numeric_limits<float>::infinity()) || p->n_kv_max != 0)
    return false;
  return query->ne[0] == 64 && key->ne[0] == 64 && value->ne == key->ne && query->ne[1] > 0 &&
         query->ne[2] > 0 && key->ne[2] > 0 && query->ne[2] % key->ne[2] == 0 &&
         query->ne[3] == 1 && key->ne[3] == 1 && key->ne[1] > 0 && key->ne[1] % 256 == 0 &&
         mask->ne[0] == key->ne[1] &&
         mask->ne[1] >= query->ne[1] + (query->ne[1] >= 1024 ? query->ne[1] % 2 : 0) &&
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
  // The resident BF16 norm scales widen exactly; no new narrowing path.
  if (from->encoding == Encoding::kBF16 && to->encoding != Encoding::kF32) return false;
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

bool AcceptsCont(const ex::CapabilityQuery& q) {
  const auto* input = Operand(q, 0);
  const auto* output = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !input || !output || input->encoding != output->encoding ||
      (input->encoding != Encoding::kF32 && input->encoding != Encoding::kF16 &&
       input->encoding != Encoding::kI32) ||
      input->layout != Layout::kDense || output->layout != Layout::kDense)
    return false;
  // CheckCont's upstream 64-thread scalar grid bound. Strides/addresses are
  // validated at launch; operand format metadata contains neither.
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX) * 64 - 1;
  const auto elements = [](const ex::OperandFormat& value) {
    std::uint64_t n = 1;
    for (const auto dim : value.ne) {
      if (!dim || dim > max / n) return std::uint64_t{0};
      n *= dim;
    }
    return n;
  };
  const auto count = elements(*input);
  return count != 0 && count == elements(*output);
}

bool AcceptsRoundedIndexScore(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RoundedIndexScoreParams>(q);
  if (!Built(q) || q.arch < 80 || !p || p->query != ir::QueryRounding::kF16 || p->reserved[0] ||
      p->reserved[1] || p->reserved[2])
    return false;
  constexpr std::array<std::uint8_t, 5> indexes{0, 1, 2, 3, ex::kOutputOperand};
  constexpr std::array<std::uint64_t, 5> needed{8, 16, 4, 4, 4};
  std::array<ScoreOperand, 5> metadata{};
  for (std::size_t i = 0; i < indexes.size(); ++i) {
    const auto* operand = Operand(q, indexes[i]);
    if (!operand || !operand->memory) return false;
    const auto& memory = *operand->memory;
    if (memory.root_alignment < needed[i] ||
        (memory.root_alignment & (memory.root_alignment - 1)) != 0)
      return false;
    metadata[i] = {.ne = operand->ne, .nb = memory.nb, .offset = memory.view_offset};
  }
  return RoundedIndexScoreFits(metadata);
}

bool AcceptsDsv4LidMask(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::LidMaskParams>(q);
  if (!Built(q) || q.arch < 80 || !p || p->top == 0 || p->top > 65536 || p->cells == 0 ||
      p->cells > INT32_MAX / 4 || p->raw_cells == 0)
    return false;
  constexpr std::array<std::uint8_t, 6> indexes{0, 1, 2, 3, 4, ex::kOutputOperand};
  constexpr std::array<std::uint64_t, 6> alignment{8, 16, 4, 4, 2, 2};
  std::array<const ex::OperandFormat*, 6> op{};
  for (std::size_t i = 0; i < op.size(); ++i) {
    op[i] = Operand(q, indexes[i]);
    if (!op[i] || !op[i]->memory) return false;
    const auto& m = *op[i]->memory;
    if (m.root_alignment < alignment[i] || (m.root_alignment & (m.root_alignment - 1)) ||
        m.view_offset % alignment[i])
      return false;
  }
  const auto rows = op[0]->ne[2];
  const auto width = std::uint64_t{p->raw_cells} + p->cells;
  if (rows == 0 || rows > std::uint64_t{65535} * 4 || rows > INT32_MAX / p->top ||
      width > std::uint64_t{65535} * 4096 || rows > INT32_MAX / width ||
      op[0]->ne != std::array<std::uint64_t, 4>{128, 64, rows, 1} ||
      op[1]->ne != std::array<std::uint64_t, 4>{128, 1, p->cells, 1} ||
      op[2]->ne != std::array<std::uint64_t, 4>{64, rows, 1, 1} ||
      op[3]->ne != std::array<std::uint64_t, 4>{rows, 1, 1, 1} || op[4]->ne[0] == 0 ||
      op[4]->ne[0] > p->raw_cells || op[4]->ne[1] != rows || op[4]->ne[2] != 1 ||
      op[4]->ne[3] != 1 || op[5]->ne != std::array<std::uint64_t, 4>{width, rows, 1, 1})
    return false;
  const auto& qnb = op[0]->memory->nb;
  const auto& knb = op[1]->memory->nb;
  const auto& wnb = op[2]->memory->nb;
  const auto& cnb = op[3]->memory->nb;
  const auto& rnb = op[4]->memory->nb;
  const auto& onb = op[5]->memory->nb;
  return qnb[0] == 4 && qnb[1] % 8 == 0 && qnb[2] % 8 == 0 && qnb[2] / 4 <= INT32_MAX / rows &&
         knb[0] == 2 && knb[2] % 16 == 0 && knb[2] / 2 <= INT32_MAX / p->cells && wnb[0] == 4 &&
         wnb[1] % 4 == 0 && wnb[1] / 4 <= INT32_MAX / rows && cnb[0] == 4 && cnb[1] == rows * 4 &&
         cnb[2] == cnb[1] && cnb[3] == cnb[1] && rnb[0] == 2 && rnb[1] % 2 == 0 &&
         rnb[1] / 2 <= INT32_MAX / rows && onb[0] == 2 && onb[1] == width * 2 &&
         onb[2] == width * rows * 2 && onb[3] == onb[2];
}

bool AcceptsPrefixOverwrite(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::PrefixOverwriteParams>(q);
  const auto* values = Operand(q, 0);
  const auto* counts = Operand(q, 1);
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX);
  return Built(q) && p && values && counts && values->ne[0] > 0 && values->ne[0] <= max &&
         values->ne[1] > 0 && values->ne[1] <= max / values->ne[0] && values->ne[2] == 1 &&
         values->ne[3] == 1 && counts->ne == std::array<std::uint64_t, 4>{values->ne[1], 1, 1, 1};
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

bool AcceptsDsv4OutA(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RopeParams>(q);
  const auto tokens = Ne(q, 1, 2);
  if (!Built(q) || !p || tokens < static_cast<std::uint64_t>(kDsv4OutAMinRows) ||
      tokens > static_cast<std::uint64_t>(kDsv4OutAMaxRows) ||
      (!ShapedOperand(q, 0, {4096, 8192, 1, 1}) && !ShapedOperand(q, 0, {4096, 1024, 8, 1}) &&
       !ShapedOperand(q, 0, {2048, 8192, 1, 1})) ||
      !ShapedOperand(q, 1, {512, 64, tokens, 1}) || !ShapedOperand(q, 2, {tokens, 1, 1, 1}) ||
      !ShapedOperand(q, kOut, {8192, tokens, 1, 1}) || Has(q, 3))
    return false;
  const auto& r = *p;
  const std::array values{r.base, r.scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow};
  return r.mode == ir::RopeMode::kNormal && r.inverse && !r.table && r.dims == 64 &&
         (r.offset == 448 || r.offset == 0) &&
         r.original_context <=
             static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) &&
         std::ranges::all_of(values, [](float v) { return std::isfinite(v); }) && r.base > 0 &&
         r.scale > 0 && r.ext_factor >= 0 &&
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

bool AcceptsSinkhornHcMix(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::HcParams>(q);
  const auto tokens = Ne(q, 0, 1);
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX) * 256 - 1;
  if (!Built(q) || !p || p->mix != ir::HcMixKind::kSinkhorn || p->streams != 4 ||
      p->iterations < 1 || p->iterations > 100 || !std::isfinite(p->eps) || p->eps < 0 ||
      tokens == 0 || tokens > max || !ShapedOperand(q, 0, {24, tokens, 1, 1}) ||
      !ShapedOperand(q, 1, {3, 1, 1, 1}) || !ShapedOperand(q, 2, {24, 1, 1, 1}) ||
      !ShapedOperand(q, kOut, {4, 4, tokens, 1}))
    return false;
  // This primitive's native recipe retains the helper's ordered F32
  // affine/softmax and epsilon-first row/column reductions. Other valid
  // shapes and layouts continue through its existing decomposition.
  for (const auto index : {std::uint8_t{0}, std::uint8_t{1}, std::uint8_t{2}, kOut}) {
    const auto* o = Operand(q, index);
    if (!o || o->encoding != Encoding::kF32 || o->layout != Layout::kDense || !o->memory ||
        o->memory->view_offset % sizeof(float) != 0)
      return false;
    std::uint64_t stride = sizeof(float);
    for (std::size_t d = 0; d < 4; ++d) {
      if (o->memory->nb[d] != stride) return false;
      stride *= o->ne[d];  // All products are bounded by the geometry above.
    }
  }
  return true;
}

bool AcceptsWeightedHcPre(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::HcParams>(q);
  const auto width = Ne(q, 0, 0), tokens = Ne(q, 0, 2);
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX) * 256 - 1;
  if (!Built(q) || !p || p->fold != ir::HcFold::kWeights || p->streams != 4 || p->scale != 1.0F ||
      width == 0 || tokens == 0 || width > max / tokens ||
      !ShapedOperand(q, 0, {width, 4, tokens, 1}) || !ShapedOperand(q, 1, {4, tokens, 1, 1}) ||
      !ShapedOperand(q, kOut, {width, tokens, 1, 1}))
    return false;
  // Concrete packed descriptors only. Hypothetical/strided operands keep
  // the portable decomposition; the bound launch checks actual addresses,
  // full backing and output/input disjointness independently.
  for (const auto index : {std::uint8_t{0}, std::uint8_t{1}, kOut}) {
    const auto* o = Operand(q, index);
    if (!o || o->encoding != Encoding::kF32 || o->layout != Layout::kDense || !o->memory ||
        o->memory->view_offset % sizeof(float) != 0)
      return false;
    std::uint64_t stride = sizeof(float);
    for (std::size_t d = 0; d < 4; ++d) {
      if (o->memory->nb[d] != stride) return false;
      stride *= o->ne[d];  // Geometry/product bounded above.
    }
  }
  return true;
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

// The existing standalone kernel computes each destination with ordered FMAs.
// Keep its exact four-stream F32 contract; other forms still decompose.
// Addresses, alignment, current views and disjoint output are checked at bind.
bool AcceptsHcPost(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::HcParams>(q);
  const auto width = Ne(q, 0, 0), tokens = Ne(q, 0, 1);
  constexpr std::uint64_t max_elements =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) * 256 - 1;
  if (!Built(q) || !p || p->streams != 4 || width == 0 || tokens == 0 ||
      width > (max_elements / 4) / tokens || q.operands.size() != 5 ||
      !ShapedOperand(q, 0, {width, tokens, 1, 1}) || !ShapedOperand(q, 1, {width, 4, tokens, 1}) ||
      !ShapedOperand(q, 2, {4, tokens, 1, 1}) || !ShapedOperand(q, 3, {4, 4, tokens, 1}) ||
      !ShapedOperand(q, kOut, {width, 4, tokens, 1}))
    return false;
  for (const auto& operand : q.operands) {
    if (!operand.memory || operand.memory->view_offset % sizeof(float) != 0) return false;
    for (const auto stride : operand.memory->nb)
      if (stride % sizeof(float) != 0 ||
          stride / sizeof(float) >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return false;
  }
  return true;
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
  return Built(q) && q.rows >= 1 && q.rows <= static_cast<std::uint64_t>(kMxfp8VecWaveColumns) &&
         q.k % 32 == 0;
}
// llmp.mxfp8.gemm.cutlass: past the vector product's columns, on the GB10.
bool AcceptsMxfp8Gemm(const ex::CapabilityQuery& q) {
  return Built(q) && q.arch == kCutlassArch &&
         q.rows > static_cast<std::uint64_t>(kMxfp8VecWaveColumns) && q.k % 128 == 0;
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
  if (p->prepared && (rows <= static_cast<std::uint64_t>(kMxfp8VecWaveColumns) || width % 128 != 0))
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
  const ex::OperandFormat* k_op = Operand(q, 1);
  const ex::OperandFormat* v_op = Operand(q, 2);
  if (!Built(q) || !p || !Has(q, 8) || p->attn.softcap != 0.0F || p->attn.max_bias != 0.0F ||
      p->attn.window != 0 || p->attn.sinks || !k_op || !v_op || k_op->encoding != v_op->encoding ||
      k_op->layout != v_op->layout ||
      (k_op->encoding == Encoding::kF16 && k_op->layout != Layout::kDense) ||
      (k_op->encoding == Encoding::kQ8_0 && k_op->layout != Layout::kGgmlBlocks))
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
bool AcceptsSelectedRoute(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RouteParams>(q);
  const auto* logits = Operand(q, 0);
  const auto* out_ids = Operand(q, kOut);
  const auto* weights = Operand(q, kOut + 1);
  if (!Built(q) || !p || p->experts != 256 || p->used < 1 || p->used > 32 ||
      !std::isfinite(p->scale) || q.operands.size() != 4 || !logits || !out_ids || !weights)
    return false;
  const bool biased = p->kind == ir::RouteKind::kSqrtSoftplusBias;
  if (!biased && p->kind != ir::RouteKind::kHash) return false;
  const auto* source = Operand(q, biased ? 1 : 2);
  if (!source || source->encoding != (biased ? Encoding::kF32 : Encoding::kI32) ||
      source->layout != Layout::kDense || (biased && p->scale <= 0.0F))
    return false;
  const auto rows = logits->ne[1];
  const std::uint64_t max_rows =
      q.policy.max_decode_rows == 8
          ? 8ULL
          : (biased ? static_cast<std::uint64_t>(kVecQMaxTokens) : 65535ULL);
  if (rows < 1 || rows > max_rows) return false;
  const auto packed = [](const ex::OperandFormat& operand, std::uint64_t width,
                         std::uint64_t count) {
    return operand.ne == std::array<std::uint64_t, 4>{width, count, 1, 1} && operand.memory &&
           operand.memory->view_offset == 0 &&
           operand.memory->nb ==
               std::array<std::uint64_t, 4>{4, 4 * width, 4 * width * count, 4 * width * count};
  };
  return packed(*logits, 256, rows) &&
         packed(*source, biased ? 256U : p->used, biased ? 1U : rows) &&
         packed(*out_ids, p->used, rows) && packed(*weights, p->used, rows);
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
std::uint64_t MaxVecQRows(const ex::CapabilityQuery& q) {
  return q.policy.max_decode_rows == 8 ? 8ULL : static_cast<std::uint64_t>(kVecQMaxTokens);
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
bool AcceptsGroupedVecQ(const ex::CapabilityQuery& q) {
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* out = Operand(q, kOut);
  const auto p = ParamsOf<ir::ProductParams>(q);
  if (!Built(q) || !w || !x || !out || !p || q.operands.size() != 3 ||
      p->accumulate != ir::Accumulate::kDefault || p->hint != ir::ProductHint::kNone)
    return false;
  const auto& info = format::Info(w->encoding);
  if (!info.ggml || !VecQType(static_cast<ggml_type>(info.ggml->type))) return false;
  const auto k = w->ne[0], n = w->ne[1], groups = w->ne[2], tokens = x->ne[2];
  constexpr std::uint64_t limit = INT32_MAX;
  const auto block = info.ggml->block_elements, bytes = info.ggml->block_bytes;
  if (k == 0 || k > limit - 511 || k % 32 != 0 || k % block != 0 || n == 0 || n > limit ||
      groups < 2 || groups > 256 || tokens == 0 || tokens > MaxVecQRows(q) || w->ne[3] != 1 ||
      x->ne != std::array<std::uint64_t, 4>{k, groups, tokens, 1} ||
      out->ne != std::array<std::uint64_t, 4>{n, groups, tokens, 1} ||
      n > limit / (k / block) / groups || !w->memory || !x->memory || !out->memory)
    return false;
  const auto packed = [](const ex::OperandFormat& operand, std::uint64_t unit,
                         std::uint64_t inner) {
    if (operand.memory->view_offset != 0 || operand.memory->nb[0] != unit) return false;
    std::uint64_t stride = unit * inner;
    for (std::size_t axis = 1; axis < 4; ++axis) {
      if (operand.memory->nb[axis] != stride ||
          __builtin_mul_overflow(stride, operand.ne[axis], &stride))
        return false;
    }
    return true;
  };
  return packed(*w, bytes, k / block) && packed(*x, sizeof(float), k) &&
         packed(*out, sizeof(float), n);
}

bool AcceptsVecQ(const ex::CapabilityQuery& q) {
  return VecQ(q) && q.rows >= 1 && q.rows <= MaxVecQRows(q);
}
// Routed: its (token, slot) pairs within the kernel's 128 (the fast form's).
bool AcceptsVecQRouted(const ex::CapabilityQuery& q) {
  const std::uint64_t used = Ne(q, q.op == ir::Fused(FusedKind::kMatMulIdGlu) ? 3 : 2, 0);
  return VecQ(q) && q.rows >= 1 && q.rows <= MaxVecQRows(q) && used >= 1 && q.rows * used <= 128;
}
// validate_ext.cc CheckGetRowsExt: GGML's decoder has no NVFP4 or
// multipart split case. Do not let this generic native rung preempt an
// actually supported selected-source decoder.
bool AcceptsGetRowsExt(const ex::CapabilityQuery& q) {
  const auto* rows = Operand(q, 0);
  const auto* ids = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!Built(q) || rows == nullptr || ids == nullptr || out == nullptr ||
      ids->encoding != Encoding::kI32 || ids->layout != Layout::kDense ||
      out->layout != Layout::kDense)
    return false;
  for (const auto* operand : {rows, ids, out})
    if (std::ranges::find(operand->ne, 0) != operand->ne.end()) return false;
  if (ids->ne[0] > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
      ids->ne[1] > std::numeric_limits<std::uint32_t>::max() / ids->ne[2])
    return false;
  const bool quantized = rows->layout == Layout::kGgmlBlocks &&
                         std::ranges::find(kQuantized, rows->encoding) != kQuantized.end();
  const bool integers = rows->encoding == Encoding::kI32 && rows->layout == Layout::kDense &&
                        out->encoding == Encoding::kI32;
  return ((quantized && out->encoding == Encoding::kF32 && rows->ne[0] % 256 == 0) || integers) &&
         rows->ne[2] == ids->ne[1] && rows->ne[3] == ids->ne[2] && ids->ne[3] == 1 &&
         out->ne[0] == rows->ne[0] && out->ne[1] == ids->ne[0] && out->ne[2] == ids->ne[1] &&
         out->ne[3] == ids->ne[2];
}
bool SpatialPackedF32(const ex::OperandFormat* operand) {
  if (operand == nullptr || operand->encoding != Encoding::kF32 ||
      operand->layout != Layout::kDense || !operand->memory)
    return false;
  std::uint64_t bytes = 4;
  for (std::size_t d = 0; d < 4; ++d) {
    if (operand->ne[d] == 0 || operand->memory->nb[d] != bytes ||
        __builtin_mul_overflow(bytes, operand->ne[d], &bytes) ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) - 256)
      return false;
  }
  return true;
}
bool AcceptsSpatialConv(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::Conv2dParams>(q);
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* b = Operand(q, 2);
  const auto* o = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !p || p->stride != 1 || p->dilation != 1 || p->padding > 1 ||
      !SpatialPackedF32(w) || !SpatialPackedF32(x) || !SpatialPackedF32(b) || !SpatialPackedF32(o))
    return false;
  return (w->ne[0] == 1 || w->ne[0] == 3) && w->ne[1] == w->ne[0] && w->ne[2] == x->ne[2] &&
         b->ne == ir::ShapeOf({w->ne[3]}).ne && x->ne[0] + 2 * p->padding >= w->ne[0] &&
         x->ne[1] + 2 * p->padding >= w->ne[1] &&
         o->ne == ir::ShapeOf({x->ne[0] + 2 * p->padding - w->ne[0] + 1,
                               x->ne[1] + 2 * p->padding - w->ne[1] + 1, w->ne[3], x->ne[3]})
                      .ne;
}
bool AcceptsSpatialTiled(const ex::CapabilityQuery& q) {
  if (!AcceptsSpatialConv(q)) return false;
  const auto p = ParamsOf<ir::Conv2dParams>(q);
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  if (w->ne[0] != 3 || p->padding != 1) return false;
  // SpatialPackedF32 already bounded the complete packed source/output spans.
  const auto round = [](std::uint64_t n) { return (n + 255) & ~std::uint64_t{255}; };
  const auto k = 9 * w->ne[2];
  if (k > static_cast<std::uint64_t>(INT32_MAX) ||
      round(4 * x->ne[0] * k) + 2 * round(4 * x->ne[0] * w->ne[3]) > kConv2dTileBudget)
    return false;
  // Measured sm_121 F32 qualification (image-tiled-conv quality/cost01):
  // group repeated operations by exact W,H,Cin,Cout with batch=1. These are
  // evidence entries, not model geometry limits. In particular, rows=W*H*N
  // alone must not admit a differently shaped image or batch. Other shapes
  // retain direct convolution unless the caller supplies explicit tuning.
  constexpr std::array<std::array<std::uint64_t, 4>, 13> qualified{{
      {64, 64, 64, 1152},
      {64, 64, 1152, 1152},
      {128, 128, 1152, 1152},
      {256, 256, 576, 576},
      {256, 256, 1152, 576},
      {256, 256, 1152, 1152},
      {512, 512, 288, 288},
      {512, 512, 576, 288},
      {512, 512, 576, 576},
      {1024, 1024, 144, 4},
      {1024, 1024, 144, 144},
      {1024, 1024, 288, 144},
      {1024, 1024, 288, 288},
  }};
  if (q.arch == 121 && x->ne[3] == 1 &&
      std::ranges::find(qualified, std::array{x->ne[0], x->ne[1], w->ne[2], w->ne[3]}) !=
          qualified.end())
    return true;
  // Preserve the existing explicit owned-tuning path for diagnostic or later
  // qualification shapes; it still passes every physical check above.
  return std::ranges::any_of(q.tuned, [&](const ex::TunedShape& t) {
    return t.implementation == kConv2dTiledF32 && t.rows == q.rows && t.k == q.k && t.n == q.n &&
           (!t.output_encoding || *t.output_encoding == Encoding::kF32);
  });
}
bool AcceptsSpatialUp(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::UpsampleParams>(q);
  const auto* x = Operand(q, 0);
  const auto* o = Operand(q, ex::kOutputOperand);
  return Built(q) && p && p->factor == 2 && SpatialPackedF32(x) && SpatialPackedF32(o) &&
         // upscale_f32 converts every output coordinate to float before dividing.
         // Keep all axis indices exactly representable, including unchanged C/N.
         std::ranges::all_of(o->ne, [](auto axis) { return axis <= (1ULL << 24U); }) &&
         o->ne == ir::ShapeOf({2 * x->ne[0], 2 * x->ne[1], x->ne[2], x->ne[3]}).ne;
}
bool AcceptsLayerNorm(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormParams>(q);
  const auto* x = Operand(q, 0);
  const auto* out = Operand(q, ex::kOutputOperand);
  return Built(q) && p && std::isfinite(p->eps) && p->eps >= 0.0F && SpatialPackedF32(x) &&
         SpatialPackedF32(out) && x->ne == out->ne && x->ne[2] <= 65535 && x->ne[3] <= 65535;
}
bool AcceptsL2Norm(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormParams>(q);
  const auto* x = Operand(q, 0);
  const auto* out = Operand(q, ex::kOutputOperand);
  return Built(q) && p && p->eps >= 1e-18F && p->eps <= 1e18F && SpatialPackedF32(x) &&
         SpatialPackedF32(out) && x->ne == out->ne && x->ne[2] <= 65535 && x->ne[3] <= 65535;
}
bool AcceptsSelectedCombine(const ex::CapabilityQuery& q) {
  const auto n = Ne(q, 0, 0), used = Ne(q, 0, 1), tokens = Ne(q, 0, 2);
  if (!Built(q) || q.operands.size() != 5 || n == 0 || n > INT32_MAX - 1024 || n % 4 != 0 ||
      used == 0 || used > 32 || tokens == 0 || tokens > 65535 ||
      !ShapedOperand(q, 0, {n, used, tokens, 1}) || !ShapedOperand(q, 1, {used, tokens, 1, 1}) ||
      !ShapedOperand(q, 2, {used, tokens, 1, 1}) || !ShapedOperand(q, 3, {n, tokens, 1, 1}) ||
      !ShapedOperand(q, kOut, {n, tokens, 1, 1}))
    return false;
  for (const auto& o : q.operands) {
    const std::uint64_t alignment = o.operand == 1 || o.operand == 2 ? 4 : 16;
    if (!o.memory || o.memory->view_offset % alignment != 0) return false;
    if (o.operand == 1) {
      // IDs retain the route's typed [used,tokens,1,1] view (for example a
      // strided view of argsort's [experts,tokens,1,1] output). The
      // selected-weights kernel does not load IDs; their real source, view
      // extent and lifetime still belong to the verified graph.
      if (o.memory->nb[0] != 4 || o.memory->nb[1] < used * 4 || o.memory->nb[1] % 4 != 0 ||
          o.memory->nb[2] < o.memory->nb[1] * tokens || o.memory->nb[3] < o.memory->nb[2] ||
          o.memory->view_offset >
              std::numeric_limits<std::uint64_t>::max() - o.memory->nb[1] * tokens)
        return false;
      continue;
    }
    std::uint64_t stride = 4;
    for (std::size_t d = 0; d < 4; ++d) {
      if (o.memory->nb[d] != stride) return false;
      stride *= o.ne[d];  // Every extent is bounded above.
    }
  }
  return true;
}
bool AcceptsGatedCombine(const ex::CapabilityQuery& q) { return Built(q) && Has(q, 4); }
bool AcceptsAny(const ex::CapabilityQuery& q) { return Built(q); }
bool AcceptsPrepared(const ex::CapabilityQuery& q) { return Built(q) && Routed(q); }

// kv_quant.h's: head sizes 64, 256, and 512, optional sinks; rows of whole blocks.
bool AcceptsQuantAttention(const ex::CapabilityQuery& q) {
  const auto* sinks = Operand(q, 4);
  return Built(q) && (q.k == 64 || q.k == 256 || q.k == 512) && q.n == q.k &&
         (!sinks || (sinks->encoding == Encoding::kF32 && sinks->layout == Layout::kDense));
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

constexpr ex::Capability F32MaskedAttention() {
  auto cap = Cap(ir::Primitive(Op::kAttention), kF32Attention, &AcceptsAttentionF32, false);
  cap.needs_tuning = true;
  return cap;
}

constexpr ex::Capability Bf16VectorProduct() {
  auto cap = Cap(ir::Primitive(Op::kMatMul), kBf16VectorProduct, &AcceptsBf16VectorProduct, false,
                 {}, 3, {.min_rows = 1, .max_rows = 8, .k_multiple = 2, .n_multiple = 2});
  cap.needs_tuning = true;
  return cap;
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
constexpr std::array<std::string_view, 2> kSpatialTiledSteps = {kConv2dTiledF32, "ggml.add"};
constexpr ex::Capability SpatialTiled() {
  // The qualified geometry predicate supplies default eligibility. Explicit
  // caller ranks still take precedence over this measured ordinary preference.
  return Cap(P(Op::kConv2d), kSpatialConvRules, &AcceptsSpatialTiled, false, {}, 1, {}, 0,
             kSpatialTiledSteps);
}
constexpr std::array<std::string_view, 2> kSpatialConvSteps = {"ggml.conv2d.f32", "ggml.add"};
constexpr std::array<std::string_view, 2> kMxfp8GemmSteps = {kMxfp8SwizzleName, kMxfp8GemmName};
constexpr std::array<std::string_view, 2> kQsaAttentionSteps = {kQsaTopKName, kQsaAttnName};
// A padded window's copy and concatenation, then the convolution.
constexpr std::array<std::string_view, 3> kSsmConvSteps = {kContName, kConcatName, kSsmConvName};
constexpr std::array<std::string_view, 7> kMoeExpertsSteps = {
    kMoeGemvName, kMoeCombineName,     kMoeRouteName,        kMoeQuantizeName,
    kMoeGemmName, kMoeGluQuantizeName, kMoeCombineSortedName};
constexpr std::array<std::string_view, 1> kVecQSteps = {kVecQName};
constexpr std::array<std::string_view, 2> kGroupedVecQSteps = {kQuantizeQ8Name, kVecQName};
// GGML's expert products over NVFP4 blocks: upstream's routing picks MMVQ
// or MMQ (ggml_cuda_mul_mat_id).
// The pre's two kernels and packed copies of its tiny strided carries.
constexpr std::array<std::string_view, 2> kDsv4LidMaskSteps = {kDsv4LidTopKName,
                                                               kDsv4SparseMaskName};
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
    Entry{"ggml.get_rows.ext", Operation::kGetRows,
          Cap(P(Op::kGetRows), {}, &AcceptsGetRowsExt, true), kBuilt},
    Entry{kConv2dTiledF32, Operation::kConv2d, SpatialTiled(), kBuilt},
    Entry{"ggml.conv2d.f32", Operation::kConv2d,
          Cap(P(Op::kConv2d), kSpatialConvRules, &AcceptsSpatialConv, false, {}, 0, {}, 0,
              kSpatialConvSteps),
          kBuilt},
    Entry{"ggml.upsample.nearest", Operation::kUpsample,
          Cap(P(Op::kUpsample), kSpatialUpRules, &AcceptsSpatialUp, false), kBuilt},
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
    Entry{"ggml.cont", Operation::kCont, Cap(P(Op::kCont), {}, &AcceptsCont, true), kBuilt},
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
    Entry{
        kRoundedIndexScoreName, Operation::kLightningIndexer,
        Cap(P(Op::kRoundedIndexScore), kRoundedIndexScoreRules, &AcceptsRoundedIndexScore, false),
        "; explicit root alignment and native pitches; paid literal fill then original Lid groups"},
    Entry{kPrefixOverwriteName, Operation::kFill,
          Cap(P(Op::kPrefixOverwrite), kPrefixOverwriteRules, &AcceptsPrefixOverwrite, true),
          kBuilt},
    Entry{kSelectedMaskName, Operation::kFill,
          Cap(P(Op::kSelectedMask), kSelectedMaskRules, &AcceptsSelectedMask, true), kBuilt},
    Entry{kTopKName, Operation::kTopK, Cap(P(Op::kTopK), kF32In, &AcceptsTopK, true),
          "; selectable for the build's SASS architectures: F32 rows within I32 counts"},
    Entry{"ggml.norm", Operation::kLayerNorm,
          Cap(P(Op::kLayerNorm), kF32In, &AcceptsLayerNorm, true),
          "; locked centered F32 normalization over packed rows; finite nonnegative epsilon"},
    Entry{"ggml.l2_norm", Operation::kL2Norm, Cap(P(Op::kL2Norm), kF32In, &AcceptsL2Norm, true),
          kBuilt},
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
    Entry{kMulMatVecFRows, Operation::kMatMul, Bf16VectorProduct(),
          "; exact output-typed tuning; packed BF16/F32/F32 product, 1..8 physical columns"},
    Entry{"ggml.mul_mat.mmvf", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup, 2), kBuilt},
    Entry{"ggml.mul_mat.mmf", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup, 1), kBuilt},
    Entry{"ggml.mul_mat.cublas", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup), kBuilt},
    Entry{"ggml.mul_mat.fwht", Operation::kMatMul,
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsHadamard, false, {}, 3),
          "; selectable for the build's SASS architectures: a square Hadamard hint"},
    Entry{
        "ggml.mul_mat_add.mmvf_fused", Operation::kMulMatAdd,
        Cap(F(FusedKind::kMatMulAdd), kF16ProductAdd, &AcceptsFloatProductFusion, false),
        "; selectable for the build's SASS architectures: one F32 input column, one dense "
        "F16 matrix with even K, F32 add of the same shape; actual MMVF and fusion gates checked"},
    Entry{"ggml.mul_mat_glu.mmvf_fused", Operation::kMulMatGlu,
          Cap(F(FusedKind::kMatMulGlu), kF16ProductGlu, &AcceptsFloatSwiGlu, false),
          "; selectable for the build's SASS architectures: one F32 input column, dense F16 "
          "gate/up matrices with even K, split unswapped SwiGLU; actual MMVF and fusion gates "
          "checked"},
    // Routed experts: the expert products by upstream's routing (MMVQ or
    // MMQ, ggml_cuda_mul_mat_id), and Gemma's fused routing and scaled
    // reduce (gemma_moe_fusion.h).
    Entry{kMulMatIdVecQ, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup, 1), kBuilt},
    Entry{kMulMatIdQ, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kMulMatIdQCompact, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kMulMatIdQCompactPrequant, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kMulMatIdQ2D2r, Operation::kMulMatId,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kDsv4RouteName, Operation::kMoeRoute,
          Cap(P(Op::kRoute), kSelectedRouteRules, &AcceptsSelectedRoute, true),
          "; opt-in sqrt-softplus routing of 256 experts, used 1..32: selected IDs rows 1..65535, "
          "biased top-k rows 1..8"},
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
    Entry{kAttentionF32, Operation::kFlashAttn, F32MaskedAttention(),
          "; opt-in packed D128 MHA: transient F16 Q/K/V MMA operands and additive mask, "
          "F32 accumulation/output; finite F16 operand range must be qualified"},
    Entry{"ggml.state_bank_attention", Operation::kFlashAttn,
          Cap(F(FusedKind::kStateBankAttention), kStateBankAttention, &AcceptsStateBankAttention,
              false, kAttentionGroup, 0, {}, 0, {}, true),
          "; the existing flash-attention group over a proven contiguous F16 state span"},
    Entry{"ggml.flash_attn_ext.vec", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention64, false, kAttentionGroup),
          "; selectable for the build's SASS architectures: D64 F32 query, F16 K/V/mask, "
          "256-padded cells, no sinks, softcap or ALiBi and F32 precision"},
    Entry{"ggml.flash_attn_ext.vec_d256", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 2),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma_gqa2", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 1),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma", Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention, false, kAttentionGroup),
          "; selectable for the build's SASS architectures at head size 256 or 512"},
    Entry{kFlashAttnMmaWideName, Operation::kFlashAttn,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsWideAttention, false, kAttentionGroup, 1),
          "; a declared D512 sparse-any recipe over bounded F16 K/V and masks"},
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
    Entry{"llmp.dsv4.lid_mask", Operation::kLightningIndexer,
          Cap(F(FusedKind::kDsv4LidMask), kDsv4LidMaskRules, &AcceptsDsv4LidMask, false, {}, 0, {},
              0, kDsv4LidMaskSteps),
          "; closed rounded-query score, visible selection and complete sparse mask"},
    Entry{kDsv4QHeadName, Operation::kRope,
          Cap(F(FusedKind::kDsv4QHead), kDsv4QHeadRules, &AcceptsDsv4QHead, true),
          "; packed F32 [512, 64, 16..8192] Q heads and normal rotary tail"},
    Entry{kDsv4CompressName, Operation::kSoftMax,
          Cap(F(FusedKind::kDsv4Compress), kDsv4CompressRules, &AcceptsDsv4Compress, true),
          "; F32 compressor values/scores, ratio 1..1024, at most 65535 blocks"},
    Entry{kHcCombName, Operation::kHcComb,
          Cap(P(Op::kHcMix), kSinkhornHcMixRules, &AcceptsSinkhornHcMix, true),
          "; packed F32 four-stream Sinkhorn, 1..100 iterations; native ordered reductions"},
    Entry{kHcPreName, Operation::kHcPre,
          Cap(P(Op::kHcPre), kWeightedHcPreRules, &AcceptsWeightedHcPre, true),
          "; packed F32 weighted fold of four streams; the original native ordered reduction"},
    Entry{kDsv4HcPrePackedName, Operation::kHcPre,
          Cap(F(FusedKind::kDsv4HcPre), kDsv4HcPreRules, &AcceptsDsv4HcPre, true, {}, 0, {}, 0,
              kDsv4HcPreSteps),
          "; full Sinkhorn carry, four F32 streams, width 4096 or 8192; packed carry copies"},
    Entry{kHcPostName, Operation::kHcPost, Cap(P(Op::kHcPost), kHcPostRules, &AcceptsHcPost, true),
          "four-stream F32 HC post with ordered native FMAs, element strides and bounded grid"},
    Entry{kDsv4HcPostNormF16Name, Operation::kHcPost,
          Cap(F(FusedKind::kHcNormPost), kDsv4HcPostRules, &AcceptsDsv4HcPost, true),
          "; four 4096-value streams, post and flat F32 norm"},
    Entry{kDsv4OutAFastPackName, Operation::kMatMul,
          Cap(F(FusedKind::kDsv4OutA), kDsv4OutARules, &AcceptsDsv4OutA, true),
          "; paid raw-Q8 packing, inverse tail64 RoPE, staged F16/HMMA grouped output-A"},
    // Qwen3.8's fast forms (slice G3b): the MXFP8 and BF16
    // products, the n-gram tables' rows, the hyper-connections, Gated
    // DeltaNet, QSA and the routed experts.
    Entry{kMxfp8MulMatVecName, Operation::kMatMul,
          Cap(P(Op::kMatMul), kMxfp8Vec, &AcceptsMxfp8Vec, false, {}, 4),
          "; selectable for the build's SASS architectures: up to 16 columns"},
    Entry{kMxfp8GemmName, Operation::kMatMul,
          Cap(P(Op::kMatMul), kMxfp8Gemm, &AcceptsMxfp8Gemm, false, {}, 4, {}, 0, kMxfp8GemmSteps),
          "; selectable on sm_121 past 16 columns, K of whole 128-value rows"},
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
          "; selectable for the build's SASS architectures: 2 to 8 streams; prepared past 16 "
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
    Entry{kDsv4CombineName, Operation::kMoeCombine,
          Cap(F(FusedKind::kMoeCombine), kSelectedCombineRules, &AcceptsSelectedCombine, true),
          "; packed ungated ordered-FMA combination, used 1..32, rows 1..8"},
    Entry{kMoeCombineName, Operation::kMoeCombine,
          Cap(F(FusedKind::kMoeCombine), kF32In, &AcceptsGatedCombine, true), kBuilt},
    Entry{"llmp.vecq.grouped", Operation::kMatMul,
          Cap(F(FusedKind::kGroupedMatMul), kGroupedVecQRules, &AcceptsGroupedVecQ, false, {}, 0,
              {}, 0, kGroupedVecQSteps),
          "packed grouped F32 input, two-to-256 immutable GGML groups, one-to-eight tokens; "
          "original Q8_1 preparation and per-slot VecQ in paid activation space"},
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
