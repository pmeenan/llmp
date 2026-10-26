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
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

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
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/float_route.h"
#include "kernels/ggml/gemma_moe.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/plan_step.h"
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
constexpr std::array kAtoms = {Layout::kScaleAtoms};
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
constexpr std::array<Rule, 4> kSpatialConvFallbackRules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
// fused.rms_norm_mul_rope_sections' (x, w, pos0[, pos1, pos2]): the
// later sections' positions are checked by its predicate.
constexpr std::array<Rule, 3> kNormRopeSectionsRules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense}}};
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
constexpr std::array<Rule, 2> kMaskPackRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF16, .layouts = kDense}}};
constexpr std::array<Rule, 2> kRopeTableRules = {
    {{.operand = 0, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 7> kQkAttentionRules = {
    {{.operand = 0, .encodings = kBf16, .layouts = kDense},
     {.operand = 1, .encodings = kBf16, .layouts = kDense},
     {.operand = 2, .encodings = kBf16, .layouts = kDense},
     {.operand = 3, .encodings = kBf16, .layouts = kDense},
     {.operand = 4, .encodings = kF16, .layouts = kDense},
     {.operand = 5, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 4> kModulateSplitRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 5> kGatedSplitRules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 8> kGatedModulateSplitRules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = 3, .encodings = kF32, .layouts = kDense},
     {.operand = 4, .encodings = kF32, .layouts = kDense},
     {.operand = 5, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = ex::kOutputOperand + 1, .encodings = kF32Bf16, .layouts = kDense}}};
constexpr std::array<Rule, 3> kSwiGluBf16Rules = {
    {{.operand = 0, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = 1, .encodings = kF32Bf16, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kBf16, .layouts = kDense}}};
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
constexpr std::array<Rule, 5> kCommonF16Attention = {
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
constexpr std::array<Rule, 2> kQ8Prepare = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kQ8_1, .layouts = kGgml}}};
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
constexpr std::array<Rule, 4> kNormMulAdd = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
constexpr std::array<Rule, 4> kNormMulRope = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kI32, .layouts = kDense},
     {.operand = ex::kOutputOperand, .encodings = kF32, .layouts = kDense}}};
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
constexpr std::array<Rule, 3> kHcMixRules = {
    {{.operand = 0, .encodings = kF32, .layouts = kDense},
     {.operand = 1, .encodings = kF32, .layouts = kDense},
     {.operand = 2, .encodings = kF32, .layouts = kDense}}};
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

bool AcceptsUnary(const ex::CapabilityQuery& q) {
  if (!Built(q)) return false;
  const auto p = ParamsOf<ir::UnaryParams>(q);
  if (!p) return false;
  switch (p->fn) {
    case ir::UnaryFn::kAbs:
    case ir::UnaryFn::kSgn:
    case ir::UnaryFn::kNeg:
    case ir::UnaryFn::kSilu:
    case ir::UnaryFn::kGeluTanh:
    case ir::UnaryFn::kTanh:
    case ir::UnaryFn::kRelu:
    case ir::UnaryFn::kSigmoid:
    case ir::UnaryFn::kExp:
    case ir::UnaryFn::kSoftplus:
    case ir::UnaryFn::kSqrt:
      return true;
    case ir::UnaryFn::kGeluErf:
    case ir::UnaryFn::kSqr:
      return false;
  }
  return false;
}

bool OptionEnabled(const ex::CapabilityQuery& q, std::string_view name) {
  if (name == "bf16_vec_rows") {
    return q.policy.options && std::ranges::contains(*q.policy.options, name);
  }
  return !q.policy.options || std::ranges::contains(*q.policy.options, name);
}

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
  if (!Built(q) || !r) return false;
  if (PlainRope(*r)) return true;
  // The ordinary QSA pool decomposition uses the existing forward IMROPE
  // launcher. Four dense position planes are required by ggml_rope_multi;
  // preserve the earlier NORMAL/NEOX domain above unchanged.
  const auto* x = Operand(q, 0);
  const auto* positions = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  // CheckRopeExt's (validate_ext.cc) scalar domain: finite parameters with
  // positive frequency base, scale and attention factor, and no YaRN or
  // Llama 3 scaling in a multi-section mode (the lowering passes Llama 3 as
  // ext_factor -1), so the planner falls back instead of failing to lower.
  if (r->llama3 || r->ext_factor != 0.0F || !std::isfinite(r->base) || !std::isfinite(r->scale) ||
      !std::isfinite(r->attn_factor) || !std::isfinite(r->beta_fast) ||
      !std::isfinite(r->beta_slow) || !(r->base > 0.0F) || !(r->scale > 0.0F) ||
      !(r->attn_factor > 0.0F))
    return false;
  if (r->table || r->inverse || r->mode != ir::RopeMode::kInterleavedMulti ||
      r->rotated_dims != UINT32_MAX || !x || !positions || x->encoding != Encoding::kF32 ||
      x->layout != Layout::kDense || positions->encoding != Encoding::kI32 ||
      positions->layout != Layout::kDense || x->ne[0] == 0 || x->ne[0] % 2 || r->dims == 0 ||
      r->dims % 2 || r->offset % 2 || r->offset > x->ne[0] || r->dims > x->ne[0] - r->offset ||
      x->ne[0] > std::uint64_t{65535} * 512 || r->original_context > INT32_MAX ||
      positions->ne[0] != x->ne[2] * 4 || positions->ne[1] != 1 || positions->ne[2] != 1 ||
      positions->ne[3] != 1 ||
      (out &&
       (out->encoding != Encoding::kF32 || out->layout != Layout::kDense || out->ne != x->ne)))
    return false;
  std::uint64_t rows = 1, sections = 0;
  for (std::size_t d = 1; d < 4; ++d) {
    if (x->ne[d] == 0 || x->ne[d] > INT32_MAX || rows > INT32_MAX / x->ne[d]) return false;
    rows *= x->ne[d];
  }
  for (const auto section : r->sections) {
    if (section > INT32_MAX) return false;
    sections += section;
  }
  return sections != 0 && sections <= INT32_MAX &&
         (r->sections[0] || r->sections[1] || r->sections[2]);
}

// CheckGemmaNormRope's (gemma_norm.cc): a full even-width NEOX rotation, its parameters finite and
// positive, with F32 frequency factors or none (upstream's fused launcher reads the rotation's;
// Gemma 4's global layers).
bool AcceptsNormRope(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormRopeParams>(q);
  const ex::OperandFormat* freq = Operand(q, 3);
  if (!Built(q) || !p || (q.k == 0 || q.k % 2 != 0 || q.k > std::numeric_limits<int>::max()) ||
      (freq != nullptr && (freq->encoding != Encoding::kF32 || freq->layout != Layout::kDense))) {
    return false;
  }
  const ir::RopeParams& r = p->rope;
  const std::array values = {r.base,        r.scale,     r.ext_factor,
                             r.attn_factor, r.beta_fast, r.beta_slow};
  return r.mode == ir::RopeMode::kNeox && !r.inverse && !r.table && !r.llama3 &&
         r.rotated_dims == UINT32_MAX && r.ext_factor >= 0.0F && r.offset == 0 && r.dims == q.k &&
         r.original_context > 0 &&
         std::ranges::all_of(values, [](float v) { return std::isfinite(v); }) && r.base > 1.0F &&
         r.scale > 0.0F && r.beta_fast > 0.0F && r.beta_slow > 0.0F && r.attn_factor > 0.0F &&
         std::isfinite(p->eps) && p->eps >= 0.0F;
}
// Upstream fused RMS norm loops over any positive int width.
bool AcceptsNormAdd(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormParams>(q);
  return Built(q) && q.k > 0 && q.k <= std::numeric_limits<int>::max() && p &&
         std::isfinite(p->eps) && p->eps >= 0;
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
  // omit it for other implementations; or general shapes on GB10 or with option.
  const bool pinned = std::ranges::any_of(q.tuned, [&](const ex::TunedShape& t) {
    return t.implementation == kMulMatVecFRows && t.rows == q.rows && t.k == q.k && t.n == q.n &&
           t.output_encoding == Encoding::kF32;
  });
  const bool shape_ok = q.k >= 256 && (q.k % 256) == 0 && q.n > 0;
  if (!pinned && !(shape_ok && OptionEnabled(q, "bf16_vec_rows"))) return false;
  return packed(0, Encoding::kBF16, q.k, q.n, 2) && packed(1, Encoding::kF32, q.k, q.rows, 4) &&
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
// GGML's split GLUs write F32.
bool GgmlGlu(const ex::CapabilityQuery& q, ir::GluKind kind) {
  const auto* out = Operand(q, ex::kOutputOperand);
  return Glu(q, kind) && (out == nullptr || out->encoding == Encoding::kF32);
}
bool AcceptsGeGlu(const ex::CapabilityQuery& q) { return GgmlGlu(q, ir::GluKind::kGeGluTanh); }
bool AcceptsSwiGlu(const ex::CapabilityQuery& q) { return GgmlGlu(q, ir::GluKind::kSwiGlu); }
bool AcceptsSwiGluClamp(const ex::CapabilityQuery& q) {
  return GgmlGlu(q, ir::GluKind::kSwiGluClamp);
}

// Packed elements of `bytes` each through its last wide axis.
bool PackedOf(const ex::OperandFormat* operand, std::uint64_t bytes) {
  if (operand == nullptr || !operand->memory) return false;
  for (std::size_t d = 0; d < 4; ++d) {
    if (operand->ne[d] == 0 || (operand->ne[d] > 1 && operand->memory->nb[d] != bytes))
      return false;
    bytes *= operand->ne[d];
  }
  return true;
}
// The existing two-dimensional BF16 kernel also covers packed activation
// batches by flattening their columns. Weight batches are not this operation.
bool AcceptsBf16MatrixProduct(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::ProductParams>(q);
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* out = Operand(q, kOut);
  if (!Built(q) || !p || !w || !x || !out || w->encoding != Encoding::kBF16 ||
      x->encoding != Encoding::kBF16 ||
      (out->encoding != Encoding::kF32 && out->encoding != Encoding::kBF16) || w->ne[2] != 1 ||
      w->ne[3] != 1 || w->ne[0] != x->ne[0] ||
      out->ne != std::array<std::uint64_t, 4>{w->ne[1], x->ne[1], x->ne[2], x->ne[3]} ||
      (p->accumulate != ir::Accumulate::kDefault && p->accumulate != ir::Accumulate::kF32) ||
      (p->hint != ir::ProductHint::kNone && p->hint != ir::ProductHint::kHadamard) ||
      p->reserved[0] || p->reserved[1])
    return false;
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX);
  std::uint64_t columns = 1;
  for (std::size_t axis = 1; axis < 4; ++axis)
    if (__builtin_mul_overflow(columns, x->ne[axis], &columns) || columns > max) return false;
  if (w->ne[0] > max || w->ne[1] > max) return false;
  const auto packed = [](const ex::OperandFormat* value) {
    if (value->layout != Layout::kDense || !value->memory) return false;
    std::uint64_t bytes = value->encoding == Encoding::kBF16 ? 2 : 4;
    for (std::size_t axis = 0; axis < 4; ++axis)
      if (value->ne[axis] == 0 || (value->ne[axis] > 1 && value->memory->nb[axis] != bytes) ||
          __builtin_mul_overflow(bytes, value->ne[axis], &bytes))
        return false;
    return true;
  };
  return packed(w) && packed(x) && packed(out);
}
// llmp.image.attention_mask_pack's (CheckAttentionMaskPack's bounds).
bool AcceptsMaskPack(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::MaskPackParams>(q);
  const auto* mask = Operand(q, 0);
  const auto* out = Operand(q, ex::kOutputOperand);
  return Built(q) && p && mask && out && mask->memory && mask->memory->nb[0] == 4 &&
         mask->ne[2] == 1 && mask->ne[3] == 1 && p->cells % 256 == 0 && p->cells >= mask->ne[0] &&
         p->rows >= mask->ne[1] && p->rows <= 65535ULL * 1024 && PackedOf(out, 2) &&
         out->ne == ir::ShapeOf({p->cells, p->rows}).ne;
}
// llmp.image.rope_table's (CheckRopeTable's bounds).
bool AcceptsRopeTable(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RopeSectionsParams>(q);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !p || p->sections < 1 || p->sections > 3 || p->mode != ir::RopeMode::kNormal ||
      !std::isfinite(p->base) || p->base <= 0.0F || !PackedOf(out, 4))
    return false;
  std::uint64_t dims = 0;
  for (std::size_t i = 0; i < p->sections; ++i) {
    const auto* pos = Operand(q, static_cast<std::uint8_t>(i));
    if (p->dims.at(i) == 0 || p->dims.at(i) % 2 != 0 || pos == nullptr ||
        pos->encoding != Encoding::kI32 || !PackedOf(pos, 4) ||
        pos->ne != ir::ShapeOf({out->ne[2]}).ne)
      return false;
    dims += p->dims.at(i);
  }
  return dims <= 512 && out->ne == ir::ShapeOf({2, dims / 2, out->ne[2]}).ne;
}
// llmp.image.qk_attention's (CheckQkAttention's bounds): heads of 128.
bool AcceptsQkAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::QkAttentionParams>(q);
  const auto* qs = Operand(q, 0);
  const auto* w = Operand(q, 1);
  const auto* k = Operand(q, 2);
  const auto* v = Operand(q, 3);
  const auto* mask = Operand(q, 4);
  const auto* table = Operand(q, 5);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!Built(q) || q.arch < 80 || !p || !std::isfinite(p->eps) || p->eps < 0.0F ||
      !std::isfinite(p->scale) || !PackedOf(qs, 2) || !PackedOf(w, 2) || !PackedOf(k, 2) ||
      !PackedOf(v, 2) || !PackedOf(mask, 2) || !PackedOf(table, 4) || out == nullptr ||
      !PackedOf(out, out->encoding == Encoding::kF32 ? 4 : 2))
    return false;
  const std::uint64_t rows = qs->ne[1];
  return qs->ne[0] % 128 == 0 && k->ne[0] % 128 == 0 && k->ne == v->ne && k->ne[1] == rows &&
         qs->ne[2] == 1 && qs->ne[3] == 1 && k->ne[2] == 1 && k->ne[3] == 1 &&
         (qs->ne[0] / 128) % (k->ne[0] / 128) == 0 && w->ne == ir::ShapeOf({128}).ne &&
         table->ne == ir::ShapeOf({2, 64, rows}).ne && mask->ne[0] >= (rows + 255) / 256 * 256 &&
         mask->ne[1] >= rows && mask->ne[2] == 1 && mask->ne[3] == 1 && out->ne == qs->ne &&
         rows <= static_cast<std::uint64_t>(INT32_MAX);
}
// llmp.image.layer_norm_modulate_split's and gated_residual_split's
// (their checks' bounds): packed rows [C, R] and [C] vectors.
bool AcceptsSplitRows(const ex::CapabilityQuery& q, bool gated) {
  const auto p = ParamsOf<ir::SplitModulateParams>(q);
  const auto* x = Operand(q, 0);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !p || x == nullptr || out == nullptr ||
      !PackedOf(x, x->encoding == Encoding::kF32 ? 4 : 2) ||
      !PackedOf(out, out->encoding == Encoding::kF32 ? 4 : 2) || x->ne != out->ne ||
      x->ne[2] != 1 || x->ne[3] != 1 || p->split > x->ne[1] || x->ne[0] > 65536 ||
      x->ne[1] > static_cast<std::uint64_t>(INT32_MAX) || !std::isfinite(p->eps) || p->eps < 0.0F)
    return false;
  if (gated) {
    const auto* branch = Operand(q, 1);
    if (branch == nullptr || branch->ne != x->ne ||
        !PackedOf(branch, branch->encoding == Encoding::kF32 ? 4 : 2) ||
        x->ne[0] * x->ne[1] > static_cast<std::uint64_t>(INT32_MAX))
      return false;
  }
  for (std::uint8_t i = gated ? 2 : 1; i < (gated ? 4 : 3); ++i) {
    const auto* v = Operand(q, i);
    if (v == nullptr || !PackedOf(v, 4) || v->ne != ir::ShapeOf({x->ne[0]}).ne) return false;
  }
  return true;
}
bool AcceptsModulateSplit(const ex::CapabilityQuery& q) { return AcceptsSplitRows(q, false); }
// llmp.image.swiglu_bf16's: a split SwiGLU of two operands of one shape
// and float type into BF16.
bool AcceptsSwiGluBf16(const ex::CapabilityQuery& q) {
  const auto* gate = Operand(q, 0);
  const auto* up = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  return Glu(q, ir::GluKind::kSwiGlu) && gate && up && out && gate->encoding == up->encoding &&
         gate->ne == up->ne && out->ne == gate->ne;
}
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
// The policy's bound on the row-at-a-time forms (SelectionPolicy
// vector_rows), where it sets one.
std::uint64_t VectorRows(const ex::CapabilityQuery& q) {
  return q.policy.vector_rows == 0 ? std::numeric_limits<std::uint64_t>::max()
                                   : q.policy.vector_rows;
}
bool CommonAttention(const ex::CapabilityQuery& q, bool vector);
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
         (query->ne[1] <= VectorRows(q) || !CommonAttention(q, false)) && query->ne[2] > 0 &&
         key->ne[2] > 0 && query->ne[2] % key->ne[2] == 0 && query->ne[3] == 1 && key->ne[3] == 1 &&
         key->ne[1] > 0 && key->ne[1] % 256 == 0 && mask->ne[0] == key->ne[1] &&
         mask->ne[1] >= query->ne[1] + (query->ne[1] >= 1024 ? query->ne[1] % 2 : 0) &&
         mask->ne[2] == 1 && mask->ne[3] == 1;
}
bool CommonAttention(const ex::CapabilityQuery& q, bool vector) {
  const auto p = ParamsOf<ir::AttnParams>(q);
  const auto* query = Operand(q, 0);
  const auto* key = Operand(q, 1);
  const auto* value = Operand(q, 2);
  const auto* mask = Operand(q, 3);
  if (!Built(q) || q.arch < 75 ||
      (q.k != 64 && q.k != 80 && q.k != 96 && q.k != 112 && q.k != 128 && q.k != 192) ||
      (q.k == 192 && !vector ? (q.n != 128 && q.n != 192) : q.n != q.k) || !p || !query || !key ||
      !value || !mask || Has(q, 4) || p->sparse_any || p->n_kv_max != 0 || p->softcap != 0 ||
      p->max_bias != 0 || p->accumulate != ir::Accumulate::kF32 || !std::isfinite(p->scale) ||
      p->scale <= 0)
    return false;
  if (key->ne[1] % 256 == 0 && query->ne[1] >= 1024) {
    // The upstream mask pre-pass reads whole query tiles. Vector uses two
    // rows; common MMA uses32 (D128 uses64 beyond Turing).
    const std::uint64_t columns = vector ? 2 : q.k == 128 && q.arch != 75 ? 64 : 32;
    if (mask->ne[1] / columns < (query->ne[1] - 1) / columns + 1) return false;
  }
  return query->ne[0] == q.k && key->ne[0] == q.k && value->ne[0] == q.n &&
         value->ne[1] == key->ne[1] && value->ne[2] == key->ne[2] && value->ne[3] == key->ne[3] &&
         query->ne[1] > 0 && query->ne[2] > 0 && key->ne[2] > 0 && key->ne[1] > 0 &&
         query->ne[2] % key->ne[2] == 0 && query->ne[3] == 1 && key->ne[3] == 1 &&
         mask->ne[0] >= key->ne[1] && mask->ne[1] >= query->ne[1] && mask->ne[2] == 1 &&
         mask->ne[3] == 1;
}
bool AcceptsCommonAttention(const ex::CapabilityQuery& q) { return CommonAttention(q, false); }
bool AcceptsCommonVectorAttention(const ex::CapabilityQuery& q) {
  return CommonAttention(q, true) && (q.k == 64 || q.k == 128 || q.k == 192) &&
         Operand(q, 0)->ne[1] <= VectorRows(q) && Operand(q, 1)->ne[1] % 256 == 0;
}
bool AcceptsAttention256(const ex::CapabilityQuery& q) {
  return Built(q) && q.k == 256 && q.n == 256;
}

// llmp.causal.mask (llmp_ops.h CausalMask): a causal or window mask; its
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
// a compiled expert count and bounded rows, softmax, selected slots renormalized
// (the clamp 2^-14 is the op's), scale 1, no bias, table or tokens.
bool AcceptsGemmaRoute(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RouteParams>(q);
  const ex::OperandFormat* logits = Operand(q, 0);
  return Built(q) && p && logits != nullptr && p->kind == ir::RouteKind::kSoftmaxTopK &&
         p->normalize &&
         (logits->ne[1] <= INT32_MAX &&
          GemmaRoutingShapeFits(p->experts, p->used, static_cast<std::int64_t>(logits->ne[1]))) &&
         p->scale == 1.0F && !Has(q, 1) && !Has(q, 2) && !Has(q, 3) &&
         logits->ne[0] == p->experts && logits->ne[1] >= 1 &&
         logits->ne[1] <= static_cast<std::uint64_t>(kGemmaMoeMaxRows) && logits->ne[2] == 1 &&
         logits->ne[3] == 1;
}
// GemmaReductionFusionAt's over kExpertReduce: the experts' F32 outputs
// [width, used, rows] within its grid/index bounds, weights [used, rows] and
// each slot's scale already gathered [1, used, rows] (the ids only name the
// slots' experts).
bool AcceptsGemmaReduce(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* x = Operand(q, 0);
  const ex::OperandFormat* w = Operand(q, 1);
  const ex::OperandFormat* s = Operand(q, 3);
  if (!Built(q) || x == nullptr || w == nullptr || s == nullptr) return false;
  const std::uint64_t rows = x->ne[2];
  using Ne = std::array<std::uint64_t, 4>;
  return x->ne[1] >= 2 &&
         (x->ne[0] <= INT32_MAX && x->ne[1] <= INT32_MAX && rows <= INT32_MAX &&
          GemmaReductionShapeFits(static_cast<std::int64_t>(x->ne[0]),
                                  static_cast<std::int64_t>(x->ne[1]),
                                  static_cast<std::int64_t>(rows))) &&
         x->ne == Ne{x->ne[0], x->ne[1], rows, 1} && w->ne == Ne{x->ne[1], rows, 1, 1} &&
         s->ne == Ne{1, x->ne[1], rows, 1};
}

bool AcceptsAdd(const ex::CapabilityQuery& q) {
  const auto* a = Operand(q, 0);
  const auto* b = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  return Built(q) && a != nullptr && b != nullptr && out != nullptr &&
         ((a->encoding == b->encoding && out->encoding == a->encoding) ||
          (a->encoding == Encoding::kF32 && b->encoding == Encoding::kF16 &&
           out->encoding == Encoding::kF32));
}

bool AcceptsConvert(const ex::CapabilityQuery& q) {
  const auto* from = Operand(q, 0);
  const auto* to = Operand(q, ex::kOutputOperand);
  if (!Built(q) || from == nullptr || to == nullptr || from->encoding == to->encoding) return false;
  const bool widen = (from->encoding == Encoding::kF16 || from->encoding == Encoding::kBF16) &&
                     to->encoding == Encoding::kF32;
  const bool narrow = from->encoding == Encoding::kF32 && to->encoding == Encoding::kF16;
  if ((!widen && !narrow) || from->ne != to->ne || !from->memory || !to->memory ||
      from->layout != Layout::kDense || to->layout != Layout::kDense)
    return false;
  const std::uint64_t from_size = narrow ? 4 : 2;
  const std::uint64_t to_size = narrow ? 2 : 4;
  const auto& source = *from->memory;
  const auto& destination = *to->memory;
  if ((source.root_alignment != 0 && source.root_alignment < from_size) ||
      source.view_offset % from_size != 0 ||
      (destination.root_alignment != 0 && destination.root_alignment < to_size) ||
      destination.view_offset % to_size != 0)
    return false;
  std::uint64_t count = 1, extent = from_size, packed_stride = to_size;
  constexpr auto signed_max = static_cast<std::uint64_t>(INT64_MAX);
  for (std::size_t d = 0; d < 4; ++d) {
    const auto n = from->ne[d];
    if (n == 0 || n > static_cast<std::uint64_t>(INT32_MAX) / count || source.nb[d] > signed_max ||
        destination.nb[d] > signed_max ||
        (n > 1 && (source.nb[d] % from_size != 0 || destination.nb[d] != packed_stride)))
      return false;
    count *= n;
    std::uint64_t step = 0;
    if (__builtin_mul_overflow(n - 1, source.nb[d], &step) ||
        __builtin_add_overflow(extent, step, &extent) || extent > signed_max ||
        __builtin_mul_overflow(packed_stride, n, &packed_stride))
      return false;
  }
  std::uint64_t end = 0;
  if (__builtin_add_overflow(source.view_offset, extent, &end)) return false;
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
  const auto p = ParamsOf<ir::RoundedIndexMaskParams>(q);
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

bool AcceptsSelect(const ex::CapabilityQuery& q) {
  const auto* condition = Operand(q, 0);
  const auto* yes = Operand(q, 1);
  const auto* no = Operand(q, 2);
  const auto* out = Operand(q, kOut);
  if (!Built(q) || q.operands.size() != 4 || !condition || !yes || !no || !out ||
      condition->encoding != Encoding::kI32 || condition->layout != Layout::kDense ||
      yes->layout != Layout::kDense || no->layout != Layout::kDense ||
      out->layout != Layout::kDense || yes->encoding != no->encoding ||
      yes->encoding != out->encoding || yes->ne != out->ne)
    return false;
  const auto& info = format::Info(yes->encoding);
  if (!info.ggml || info.block_elements != 1) return false;
  std::uint64_t elements = 1;
  for (unsigned axis = 0; axis < 4; ++axis) {
    if (!condition->ne[axis] || !no->ne[axis] || !yes->ne[axis] ||
        yes->ne[axis] % condition->ne[axis] || yes->ne[axis] % no->ne[axis] ||
        __builtin_mul_overflow(elements, yes->ne[axis], &elements) || elements > INT32_MAX)
      return false;
  }
  return true;
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
  return r.mode == ir::RopeMode::kNormal && !r.inverse && !r.table && !r.llama3 &&
         r.rotated_dims == UINT32_MAX && r.dims == 64 && r.offset == 448 &&
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
      (!ShapedOperand(q, 0, {4096, 8192, 1, 1}) && !ShapedOperand(q, 0, {4096, 1024, 8, 1})) ||
      !ShapedOperand(q, 1, {512, 64, tokens, 1}) || !ShapedOperand(q, 2, {tokens, 1, 1, 1}) ||
      !ShapedOperand(q, kOut, {8192, tokens, 1, 1}) || Has(q, 3))
    return false;
  const auto& r = *p;
  const std::array values{r.base, r.scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow};
  return r.mode == ir::RopeMode::kNormal && r.inverse && !r.table && !r.llama3 &&
         r.rotated_dims == UINT32_MAX && r.dims == 64 && r.offset == 448 &&
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
      (p->mix_offset != 0 && p->mix_offset != 8) || (p->scale_index != 0 && p->scale_index != 2) ||
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
         (width >= 256 && width <= 8192 && (width % 256) == 0) && tokens >= 1 && tokens <= 65535 &&
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

// The native competitors' known physical facts, before a binding exists.
bool KnownAligned(const ex::OperandFormat& operand, std::uint64_t bytes) {
  return !operand.memory ||
         (operand.memory->view_offset % bytes == 0 &&
          (operand.memory->root_alignment == 0 || (operand.memory->root_alignment >= bytes &&
                                                   operand.memory->root_alignment % bytes == 0)));
}
bool NativePacked(const ex::OperandFormat& operand, std::uint64_t bytes) {
  if (!operand.memory) return true;
  for (std::size_t d = 0; d < 4; ++d) {
    if (operand.memory->nb[d] != bytes || __builtin_mul_overflow(bytes, operand.ne[d], &bytes))
      return false;
  }
  return true;
}
bool NativeRows(const ex::CapabilityQuery& q) {
  const auto* table = Operand(q, 0);
  const auto* ids = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  return table && ids && table->ne[0] && table->ne[1] && ids->ne[0] && table->ne[2] == 1 &&
         table->ne[3] == 1 && ids->ne[1] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
         table->ne[1] <= INT32_MAX && ids->ne[0] <= 65535ULL * 1024 && NativePacked(*ids, 4) &&
         KnownAligned(*ids, 4) &&
         (!out ||
          (out->encoding == Encoding::kF32 && out->layout == Layout::kDense &&
           out->ne == std::array{table->ne[0], ids->ne[0], std::uint64_t{1}, std::uint64_t{1}} &&
           NativePacked(*out, 4) && KnownAligned(*out, 4)));
}
bool AcceptsMxfp8Vec(const ex::CapabilityQuery& q) {
  if (!Built(q) || q.rows < 1 || q.rows > kMxfp8VecWaveColumns || q.k == 0 || q.k % 32 ||
      q.k > INT32_MAX || q.n == 0 || q.n > INT32_MAX - 8)
    return false;
  const auto p = ParamsOf<ir::ProductParams>(q);
  if (p && p->hint != ir::ProductHint::kNone) return false;
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!w || !x) return false;
  // Queries without logical extents carry no physical geometry yet.
  if (w->ne[0] == 0 || x->ne[0] == 0) return true;
  if (w->ne != std::array{q.k, q.n, std::uint64_t{1}, std::uint64_t{1}} ||
      x->ne != std::array{q.k, q.rows, std::uint64_t{1}, std::uint64_t{1}} ||
      (w->memory && (w->memory->is_view || w->memory->view_offset != 0)) || !KnownAligned(*x, 16))
    return false;
  if (x->memory &&
      (x->memory->nb[0] != 4 || x->memory->nb[1] % 16 || x->memory->nb[1] / 4 > INT32_MAX))
    return false;
  return !out || (out->ne == std::array{q.n, q.rows, std::uint64_t{1}, std::uint64_t{1}} &&
                  NativePacked(*out, 4) && KnownAligned(*out, 4));
}
// llmp.mxfp8.gemm.cutlass: past the vector product's columns, on the GB10.
bool AcceptsMxfp8Gemm(const ex::CapabilityQuery& q) {
  return Built(q) && q.arch == kCutlassArch &&
         (q.features & (ex::kFp8Mma | ex::kBlockScaledMma)) ==
             (ex::kFp8Mma | ex::kBlockScaledMma) &&
         q.rows > static_cast<std::uint64_t>(kMxfp8VecWaveColumns) && q.k % 128 == 0;
}
bool AcceptsMxfp8Quantize(const ex::CapabilityQuery& q) {
  return Built(q) && q.arch == kCutlassArch && Ne(q, 0, 0) % 128 == 0 && Ne(q, 0, 2) <= 1 &&
         Ne(q, 0, 3) <= 1;
}
// llmp.nvfp4.get_rows: CheckNvfp4Rows' rows of 16 to 1,024 values (a
// thread a value) and its grid of ids.
bool AcceptsNvfp4Rows(const ex::CapabilityQuery& q) {
  const auto* table = Operand(q, 0);
  const auto* ids = Operand(q, 1);
  // This kernel reads one global scale, unlike the general decoder's
  // optional per-stack scales and batched tables.
  return Built(q) && NativeRows(q) && table && ids && table->params.tensor_scale &&
         table->stacked == 1 &&
         (!table->memory || (!table->memory->is_view && table->memory->view_offset == 0)) &&
         Ne(q, 0, 0) % 16 == 0 && Ne(q, 0, 0) >= 16 && Ne(q, 0, 0) <= 1024 &&
         Ne(q, 0, 1) <= static_cast<std::uint64_t>(INT32_MAX) &&
         Ne(q, 1, 0) <= std::uint64_t{65535} * 1024;
}
// llmp.qrows.get_rows: a table of 32-value blocks whose rows are not GGML's
// whole 256-value steps (its get_rows takes those), at most 1,024 values.
bool AcceptsQRows(const ex::CapabilityQuery& q) {
  const ex::OperandFormat* t = Operand(q, 0);
  if (!Built(q) || t == nullptr) return false;
  const bool type = t->encoding == Encoding::kQ4_0 || t->encoding == Encoding::kQ4_1 ||
                    t->encoding == Encoding::kQ5_0 || t->encoding == Encoding::kQ5_1 ||
                    t->encoding == Encoding::kQ8_0 || t->encoding == Encoding::kIQ4_NL;
  if (!type || !NativeRows(q) || t->ne[0] % 32 || t->ne[0] % 256 == 0 || t->ne[0] > 1024)
    return false;
  const auto& native = *format::Info(t->encoding).ggml;
  if (!KnownAligned(*t, t->encoding == Encoding::kQ4_1 || t->encoding == Encoding::kQ5_1 ? 4 : 2))
    return false;
  if (!t->memory) return true;
  const auto stride = t->ne[0] / native.block_elements * native.block_bytes;
  return t->memory->nb == std::array{std::uint64_t{native.block_bytes}, stride, stride * t->ne[1],
                                     stride * t->ne[1]};
}
bool AcceptsNvfp4Experts(const ex::CapabilityQuery& q) {
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* ids = Operand(q, 2);
  const auto* out = Operand(q, ex::kOutputOperand);
  // GGML's NVFP4 tensor contains only Data. Optional per-stack global
  // scales are separate physical parts consumed by the generic decoder.
  if (!Built(q) || !w || w->params.tensor_scale || !x || !ids ||
      (out && (out->encoding != Encoding::kF32 || out->layout != Layout::kDense)))
    return false;
  if (!w->ne[0] || !x->ne[0] || !ids->ne[0]) return true;
  const auto indexed = [](const ex::OperandFormat& operand, std::uint64_t unit, std::uint64_t block,
                          std::uint64_t alignment, std::uint64_t tail) {
    std::uint64_t span = unit;
    for (const auto dim : operand.ne)
      if (dim == 0 || dim > INT32_MAX) return false;
    if (!KnownAligned(operand, alignment)) return false;
    if (!operand.memory) return true;
    if (operand.memory->nb[0] != unit || operand.ne[0] % block ||
        operand.memory->nb[1] < operand.ne[0] / block * unit)
      return false;
    for (std::size_t d = 0; d < 4; ++d) {
      const auto stride = operand.memory->nb[d];
      std::uint64_t step = 0;
      if (stride % unit || stride / unit > INT32_MAX ||
          __builtin_mul_overflow(d == 0 ? operand.ne[d] / block - 1 : operand.ne[d] - 1, stride,
                                 &step) ||
          __builtin_add_overflow(span, step, &span))
        return false;
    }
    return span / unit <= static_cast<std::uint64_t>(INT32_MAX) + (block > 1 ? 1U : 0U) - tail;
  };
  const auto& native = *format::Info(Encoding::kNvfp4).ggml;
  const auto tail = (512 - w->ne[0] % 512) % 512 / native.block_elements;
  if (!indexed(*w, native.block_bytes, native.block_elements, 16, tail) ||
      !indexed(*x, 4, 1, 4, 0) || !indexed(*ids, 4, 1, 4, 0) || (out && !indexed(*out, 4, 1, 4, 0)))
    return false;
  if ((x->memory && (x->memory->nb[1] == 0 || x->memory->nb[2] % x->memory->nb[1])) ||
      (out && out->memory && (out->memory->nb[1] == 0 || out->memory->nb[2] % out->memory->nb[1])))
    return false;
  return w->ne[3] == 1 && x->ne[3] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
         w->ne[0] == x->ne[0] && (x->ne[1] == 1 || x->ne[1] == ids->ne[0]) &&
         x->ne[2] == ids->ne[1] && ids->ne[0] <= w->ne[2] &&
         (!out || out->ne == std::array{w->ne[1], ids->ne[0], ids->ne[1], std::uint64_t{1}});
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
  if (!Built(q) || Has(q, 2) || !p || p->streams < 2 || p->streams > 8) return false;
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
bool AcceptsHcMix(const ex::CapabilityQuery& q) {
  if (!AcceptsHcNorm(q) || !Has(q, 2)) return false;
  if (const auto p = ParamsOf<ir::HcFoldParams>(q)) {
    return p->streams == Ne(q, 0, 1) && std::isfinite(p->eps) && p->eps >= 0.0F && !p->prepared &&
           !p->rounded;
  }
  return q.params.empty();
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
// llmp.gdn.gates: independent pointwise rows, two F32 output views.
bool AcceptsGdnGates(const ex::CapabilityQuery& q) {
  const auto heads = Ne(q, 0, 0), rows = Ne(q, 0, 1);
  if (!Built(q) || !GdnGatesShapeFits(heads, rows)) return false;
  for (std::uint8_t i = 0; i < 4; ++i) {
    const auto* operand = Operand(q, i);
    if (!operand || operand->encoding != Encoding::kF32 || operand->layout != Layout::kDense ||
        !ShapedOperand(q, i, {heads, i < 2 ? rows : 1, 1, 1}))
      return false;
  }
  for (const auto i : {ex::kOutputOperand, static_cast<std::uint8_t>(ex::kOutputOperand + 1)}) {
    const auto* operand = Operand(q, i);
    if (!operand || operand->encoding != Encoding::kF32 || operand->layout != Layout::kDense ||
        !ShapedOperand(q, i, {1, heads, rows, 1}))
      return false;
  }
  return true;
}
bool AcceptsGdnNormGate(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormGateParams>(q);
  return Built(q) && p &&
         (p->gate == ir::NormGateFn::kSigmoid || p->gate == ir::NormGateFn::kSilu) &&
         Ne(q, 0, 0) == 128 && Ne(q, 0, 3) <= 1;
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
         r.rotated_dims == UINT32_MAX && r.offset == 0 && r.scale == 1.0F && r.ext_factor == 0.0F &&
         r.attn_factor == 1.0F;
}

bool AcceptsQsaPrep(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::NormRopeParams>(q);
  const std::uint64_t d = Ne(q, 0, 0);
  return Built(q) && p && QsaRope(p->rope) && Has(q, 2) && !Has(q, 3) && d >= 64 && d <= 512 &&
         d % 32 == 0;
}
bool AcceptsQsaPool(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::BlockPoolParams>(q);
  const std::uint64_t d = Ne(q, 0, 0);
  return Built(q) && p && p->ratio >= 1 && p->ratio <= 32 && QsaRope(p->rope) && d >= 64 &&
         d <= 512 && d % 32 == 0 && Has(q, 5);
}
// llmp.qsa.topk and llmp.qsa.attn: 256-wide heads at most 16 to a KV head,
// an indexer of 4 heads of 128, the selection within its tiles.
bool AcceptsQsaAttention(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::BlockSparseAttnParams>(q);
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
  if (!Built(q) || !p || (p->experts < 1 || p->experts > 256) || p->used < 1 ||
      p->used > std::min(32U, p->experts) || !std::isfinite(p->scale) || q.operands.size() != 4 ||
      !logits || !out_ids || !weights)
    return false;
  const bool biased = p->kind == ir::RouteKind::kSqrtSoftplusBias;
  if ((!biased && p->kind != ir::RouteKind::kHash) ||
      !OptionEnabled(q, biased ? "biased_route" : "hash_route"))
    return false;
  const auto* source = Operand(q, biased ? 1 : 2);
  if (!source || source->encoding != (biased ? Encoding::kF32 : Encoding::kI32) ||
      source->layout != Layout::kDense || (biased && p->scale <= 0.0F))
    return false;
  const auto rows = logits->ne[1];
  const std::uint64_t max_rows = std::min<std::uint64_t>(
      biased ? std::min<std::uint64_t>(kVecQMaxTokens, 16) : 8U, VectorRows(q));
  if (rows < 1 || rows > max_rows) return false;
  const auto packed = [](const ex::OperandFormat& operand, std::uint64_t width,
                         std::uint64_t count) {
    return operand.ne == std::array<std::uint64_t, 4>{width, count, 1, 1} && operand.memory &&
           operand.memory->view_offset == 0 &&
           operand.memory->nb ==
               std::array<std::uint64_t, 4>{4, 4 * width, 4 * width * count, 4 * width * count};
  };
  return packed(*logits, p->experts, rows) &&
         packed(*source, biased ? p->experts : p->used, biased ? 1U : rows) &&
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
  return std::min<std::uint64_t>(static_cast<std::uint64_t>(kVecQMaxTokens), VectorRows(q));
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

bool VecQGluFormula(const ex::CapabilityQuery& q) {
  std::optional<ir::GluParams> glu;
  if (q.op == ir::Fused(FusedKind::kMatMulGlu)) {
    glu = ParamsOf<ir::GluParams>(q);
  } else if (q.op == ir::Fused(FusedKind::kMatMulIdGlu)) {
    const auto p = ParamsOf<ir::MoeExpertsParams>(q);
    if (p) glu = p->glu;
  } else {
    return true;
  }
  return glu && !glu->swapped && std::isfinite(glu->limit) &&
         ((glu->kind == ir::GluKind::kSwiGluClamp && glu->limit > 0.0F) ||
          ((glu->kind == ir::GluKind::kSwiGlu || glu->kind == ir::GluKind::kGeGluTanh) &&
           glu->limit == 0.0F));
}

// The registered GB10 five-column Q6_K/Q8_0 workload regressed in first-token
// latency. Prefer upstream MMVQ across that two-warp column class, sharing
// preparation where eligible and keeping automatic GLUs separate. This
// device/encoding/physical-column choice leaves routed and grouped work alone.
bool PreferOrdinaryMmvq(const ex::CapabilityQuery& q) {
  const bool dense = q.op == ir::Primitive(Op::kMatMul);
  const bool glu = q.op == ir::Fused(FusedKind::kMatMulGlu);
  if (q.arch != 121 || (!dense && !glu) || q.rows < 5 || q.rows > 8) return false;
  // A directly supplied fused operation keeps its implementation contract.
  if (glu && !q.automatic_fusion) return false;
  if (dense) {
    const auto p = ParamsOf<ir::ProductParams>(q);
    if (!p || p->accumulate != ir::Accumulate::kDefault || p->hint != ir::ProductHint::kNone ||
        p->arithmetic != ir::ProductArithmetic::kImplementation)
      return false;
  }
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, glu ? 2 : 1);
  const auto* out = Operand(q, kOut);
  if (!w || !x || !out || w->stacked != 1 || w->layout != Layout::kGgmlBlocks ||
      (w->encoding != Encoding::kQ6_K && w->encoding != Encoding::kQ8_0) ||
      w->ne != std::array<std::uint64_t, 4>{q.k, q.n, 1, 1} ||
      x->ne != std::array<std::uint64_t, 4>{q.k, q.rows, 1, 1} ||
      out->ne != std::array<std::uint64_t, 4>{q.n, q.rows, 1, 1} ||
      out->encoding != Encoding::kF32 || out->layout != Layout::kDense)
    return false;
  if (glu) {
    const auto* up = Operand(q, 1);
    if (!up || up->stacked != 1 || up->encoding != w->encoding || up->layout != w->layout ||
        up->ne != w->ne)
      return false;
  }
  return MmvqRouted(w->encoding, q.arch, q.rows);
}

bool AcceptsVecQ(const ex::CapabilityQuery& q) {
  return VecQ(q) && VecQGluFormula(q) && q.rows >= 1 && q.rows <= MaxVecQRows(q) &&
         !PreferOrdinaryMmvq(q);
}
// Routed: its (token, slot) pairs within the kernel's 128 (the fast form's).
// The primitive accepts one broadcast input slot or one for each selected
// expert. Intermediate repeating-slot counts require the ordinary fallback.
bool AcceptsVecQRouted(const ex::CapabilityQuery& q) {
  const bool glu = q.op == ir::Fused(FusedKind::kMatMulIdGlu);
  const auto* x = Operand(q, glu ? 2 : 1);
  const auto* ids = Operand(q, glu ? 3 : 2);
  if (!VecQ(q) || !VecQGluFormula(q) || !x || !ids || q.rows < 1 || q.rows > MaxVecQRows(q) ||
      ids->ne[0] < 1 || ids->ne[0] > 128 / q.rows || ids->ne[1] != q.rows || ids->ne[2] != 1 ||
      ids->ne[3] != 1 || x->ne[0] != q.k || x->ne[3] != 1)
    return false;
  if (glu) return x->ne[1] == q.rows && x->ne[2] == 1;
  return (x->ne[1] == 1 || x->ne[1] == ids->ne[0]) && x->ne[2] == q.rows;
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
// Packed F32 or BF16 (SpatialPackedF32's bounds at the element's size).
bool SpatialPacked(const ex::OperandFormat* operand) {
  if (operand == nullptr || operand->layout != Layout::kDense || !operand->memory ||
      (operand->encoding != Encoding::kF32 && operand->encoding != Encoding::kBF16))
    return false;
  std::uint64_t bytes = operand->encoding == Encoding::kF32 ? 4 : 2;
  for (std::size_t d = 0; d < 4; ++d) {
    if (operand->ne[d] == 0 || operand->memory->nb[d] != bytes ||
        __builtin_mul_overflow(bytes, operand->ne[d], &bytes) ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) - 256)
      return false;
  }
  return true;
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
bool SpatialPackedFloat(const ex::OperandFormat* operand) {
  if (operand == nullptr || operand->layout != Layout::kDense || !operand->memory ||
      (operand->encoding != Encoding::kF32 && operand->encoding != Encoding::kBF16))
    return false;
  const std::uint64_t size = operand->encoding == Encoding::kF32 ? 4 : 2;
  std::uint64_t bytes = size;
  for (std::size_t d = 0; d < 4; ++d) {
    if (operand->ne[d] == 0 || operand->memory->nb[d] != bytes ||
        __builtin_mul_overflow(bytes, operand->ne[d], &bytes))
      return false;
  }
  return bytes <= static_cast<std::uint64_t>(INT32_MAX) - 256;
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
bool AcceptsSpatialConvStrided(const ex::CapabilityQuery& q) {
  // Preserve every source-format/geometry case already owned by the direct,
  // tiled and pointwise F32 implementations. This entry is a genuine fallback.
  if (AcceptsSpatialConv(q)) return false;
  const auto p = ParamsOf<ir::Conv2dParams>(q);
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* b = Operand(q, 2);
  const auto* o = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !p || p->stride == 0 || p->dilation == 0 ||
      p->stride > static_cast<std::uint64_t>(INT32_MAX) ||
      p->padding > static_cast<std::uint64_t>(INT32_MAX) ||
      p->dilation > static_cast<std::uint64_t>(INT32_MAX) || w == nullptr ||
      w->layout != Layout::kDense || !w->memory ||
      (w->encoding != Encoding::kF32 && w->encoding != Encoding::kBF16) || !SpatialPackedF32(x) ||
      !SpatialPackedFloat(b) || !SpatialPackedF32(o) || w->ne[0] == 0 || w->ne[1] == 0 ||
      w->ne[2] != x->ne[2] || w->ne[3] == 0 || b->ne != ir::ShapeOf({w->ne[3]}).ne)
    return false;
  const std::uint64_t elem = w->encoding == Encoding::kBF16 ? 2 : 4;
  if (w->memory->nb[0] != elem ||
      std::ranges::any_of(w->memory->nb, [elem](auto n) { return n % elem != 0; }))
    return false;
  std::uint64_t pw = 0, ph = 0, kw = 0, kh = 0;
  if (__builtin_mul_overflow(p->padding, std::uint64_t{2}, &pw) ||
      __builtin_mul_overflow(p->padding, std::uint64_t{2}, &ph) ||
      __builtin_add_overflow(x->ne[0], pw, &pw) || __builtin_add_overflow(x->ne[1], ph, &ph) ||
      __builtin_mul_overflow(w->ne[0] - 1, p->dilation, &kw) ||
      __builtin_add_overflow(kw, std::uint64_t{1}, &kw) ||
      __builtin_mul_overflow(w->ne[1] - 1, p->dilation, &kh) ||
      __builtin_add_overflow(kh, std::uint64_t{1}, &kh) || pw < kw || ph < kh)
    return false;
  const std::array<std::uint64_t, 4> expected{(pw - kw) / p->stride + 1, (ph - kh) / p->stride + 1,
                                              w->ne[3], x->ne[3]};
  std::uint64_t elements = 1;
  for (const std::uint64_t extent : expected) {
    if (__builtin_mul_overflow(elements, extent, &elements) ||
        elements > static_cast<std::uint64_t>(INT32_MAX) / sizeof(float))
      return false;
  }
  std::uint64_t max_x = 0, max_y = 0;
  if (__builtin_mul_overflow(expected[0] - 1, p->stride, &max_x) ||
      __builtin_mul_overflow(expected[1] - 1, p->stride, &max_y) ||
      __builtin_add_overflow(max_x, kw - 1, &max_x) ||
      __builtin_add_overflow(max_y, kh - 1, &max_y))
    return false;
  return o->ne == expected;
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
// fused.dup_up_add's (x [2W, 2H, Co, 1], shortcut [W, H, Ci, 1]): packed
// F32, the repeats whole, an int element count (CheckDupUpAdd's bounds).
bool AcceptsDupUpAdd(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::DupUpParams>(q);
  const auto* x = Operand(q, 0);
  const auto* sc = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !p || p->factor_t < 1 || p->factor_t > 2 || !SpatialPackedF32(x) ||
      !SpatialPackedF32(sc) || !SpatialPackedF32(out) || x->ne != out->ne || x->ne[3] != 1 ||
      sc->ne[3] != 1 || sc->ne[0] == 0 || sc->ne[1] == 0 || sc->ne[2] == 0 ||
      x->ne[0] != 2 * sc->ne[0] || x->ne[1] != 2 * sc->ne[1] || x->ne[2] == 0 ||
      (x->ne[2] * p->factor_t * 4) % sc->ne[2] != 0)
    return false;
  return x->ne[0] * x->ne[1] * x->ne[2] <=
         static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
}
// llmp.image.conv1x1's: a 1x1 kernel without padding over packed F32 of one
// image, its pixels, channels and outputs ints (CheckConv1x1's bounds).
bool AcceptsConv1x1(const ex::CapabilityQuery& q) {
  if (!AcceptsSpatialConv(q)) return false;
  const auto p = ParamsOf<ir::Conv2dParams>(q);
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  return w->ne[0] == 1 && p->padding == 0 && x->ne[3] == 1 && x->ne[0] * x->ne[1] <= kMax &&
         w->ne[2] <= kMax && w->ne[3] <= kMax;
}
bool AcceptsGatedSplit(const ex::CapabilityQuery& q) { return AcceptsSplitRows(q, true); }
// fused.gated_residual_modulate_split's: the gated residual's operands and
// sum as above, the modulation's two vectors and its packed result.
bool AcceptsGatedModulateSplit(const ex::CapabilityQuery& q) {
  if (!AcceptsSplitRows(q, true)) return false;
  const auto* x = Operand(q, 0);
  const auto* modulated = Operand(q, ex::kOutputOperand + 1);
  const auto* sum = Operand(q, ex::kOutputOperand);
  // A BF16 sum only where the pass keeps the row (CheckGatedResidualModulateSplit).
  if (modulated == nullptr || modulated->ne != x->ne ||
      !PackedOf(modulated, modulated->encoding == Encoding::kF32 ? 4 : 2) ||
      (sum->encoding == Encoding::kBF16 && x->ne[0] > 8192))
    return false;
  for (std::uint8_t i = 4; i < 6; ++i) {
    const auto* v = Operand(q, i);
    if (v == nullptr || !PackedOf(v, 4) || v->ne != ir::ShapeOf({x->ne[0]}).ne) return false;
  }
  return true;
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
// fused.spatial_channel_norm's (x [W, H, C, 1], gamma [C]): packed F32, a
// pixel a thread over a grid of 256-thread blocks (llmp_ops.h
// CheckSpatialChannelNorm's bounds).
bool AcceptsSpatialChannelNorm(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::SpatialNormParams>(q);
  const auto* x = Operand(q, 0);
  const auto* gamma = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  // The L2 norm's floor bounds, as ggml.l2_norm's (validate.cc CheckL2Norm):
  // a floor below 1e-18 underflows its square.
  if (!Built(q) || !p || !(p->eps >= 1e-18F && p->eps <= 1e18F) || !std::isfinite(p->scale) ||
      !SpatialPackedF32(x) || !SpatialPackedF32(out) || x->ne != out->ne || x->ne[3] != 1 ||
      gamma == nullptr || gamma->encoding != Encoding::kF32 || gamma->layout != Layout::kDense ||
      gamma->ne[0] != x->ne[2] || gamma->ne[1] != 1 || gamma->ne[2] != 1 || gamma->ne[3] != 1)
    return false;
  return x->ne[2] <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
}
// fused.rms_norm_mul_rope_sections' (CheckNormRopeSections' bounds):
// packed F32 heads [D, heads, rows] (D even, at most 512), a packed [D]
// weight, one to three plain sections covering D in one mode, each with
// packed I32 positions [rows].
bool AcceptsNormRopeSections(const ex::CapabilityQuery& q) {
  const auto p = ParamsOf<ir::RopeSectionsParams>(q);
  const auto* x = Operand(q, 0);
  const auto* w = Operand(q, 1);
  const auto* out = Operand(q, ex::kOutputOperand);
  if (!Built(q) || !p || p->sections < 1 || p->sections > 3 ||
      (p->mode != ir::RopeMode::kNormal && p->mode != ir::RopeMode::kNeox) ||
      !std::isfinite(p->eps) || p->eps < 0.0F || !std::isfinite(p->base) || p->base <= 0.0F ||
      !SpatialPacked(x) || !SpatialPacked(out) || x->ne != out->ne || w == nullptr || !w->memory ||
      w->ne != ir::ShapeOf({x->ne[0]}).ne ||
      w->memory->nb[0] != (w->encoding == Encoding::kF32 ? 4U : 2U))
    return false;
  const std::uint64_t dim = x->ne[0], heads = x->ne[1], rows = x->ne[2];
  if (dim == 0 || dim % 2 != 0 || dim > 512 || heads > 65535 || rows > 65535 || x->ne[3] != 1)
    return false;
  std::uint64_t covered = 0;
  for (std::size_t i = 0; i < p->sections; ++i) {
    const auto* pos = Operand(q, static_cast<std::uint8_t>(2 + i));
    if (p->dims.at(i) == 0 || p->dims.at(i) % 2 != 0 || pos == nullptr ||
        pos->encoding != Encoding::kI32 || pos->layout != Layout::kDense || !pos->memory ||
        pos->ne != ir::ShapeOf({rows}).ne || pos->memory->nb[0] != sizeof(std::int32_t))
      return false;
    covered += p->dims.at(i);
  }
  return covered == dim && Operand(q, static_cast<std::uint8_t>(2 + p->sections)) == nullptr;
}
bool AcceptsSelectedCombine(const ex::CapabilityQuery& q) {
  if (!OptionEnabled(q, "moe_combine")) return false;
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

// kv_quant.h's: head sizes 64, 256 (with optional positive logit softcap
// at D=256), and 512 for Q8_0 alone (fattn_vec_quant.cu compiles no Q4_0 instance
// there), optional F32 sinks; rows of whole blocks.
bool QuantAttention(const ex::CapabilityQuery& q, bool d512 = true) {
  const auto* sinks = Operand(q, 4);
  if (!Built(q) || !(q.k == 64 || q.k == 256 || (d512 && q.k == 512)) || q.n != q.k) {
    return false;
  }
  if (sinks && (sinks->encoding != Encoding::kF32 || sinks->layout != Layout::kDense)) {
    return false;
  }
  if (const auto p = ParamsOf<ir::AttnParams>(q)) {
    if (p->max_bias > 0.0F || p->sparse_any ||
        (p->softcap != 0.0F && !(q.k == 256 && p->softcap > 0.0F))) {
      return false;
    }
  }
  const auto* qo = Operand(q, 0);
  const auto* ko = Operand(q, 1);
  if (qo != nullptr && qo->ne[0] > 0 && qo->ne[3] != 1) return false;
  if (ko != nullptr && ko->ne[0] > 0 && (ko->ne[3] != 1 || ko->ne[1] % 256 != 0)) {
    return false;
  }
  return true;
}
bool AcceptsQuantAttention(const ex::CapabilityQuery& q) { return QuantAttention(q, true); }
bool AcceptsQ4Attention(const ex::CapabilityQuery& q) { return QuantAttention(q, false); }
bool AcceptsQuantStore(const ex::CapabilityQuery& q) { return Built(q) && q.k >= 32; }
bool AcceptsQuantRopeStore(const ex::CapabilityQuery& q) {
  const auto r = ParamsOf<ir::RopeParams>(q);
  const ex::OperandFormat* freq = Operand(q, 4);
  return r && !r->llama3 && r->ext_factor >= 0.0F && r->rotated_dims == UINT32_MAX &&
         AcceptsQuantStore(q) &&
         (freq == nullptr || (freq->encoding == Encoding::kF32 && freq->layout == Layout::kDense));
}
// ggml.rope_set_rows.fused (RopeSetRowsFusionAt): NEOX into F16 rows.
bool AcceptsRopeStore(const ex::CapabilityQuery& q) {
  const auto r = ParamsOf<ir::RopeParams>(q);
  return Built(q) && r && PlainRope(*r) && r->mode == ir::RopeMode::kNeox && !r->inverse &&
         r->rotated_dims == UINT32_MAX && r->offset == 0 && !Has(q, 4);
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
  ex::Capability capability;
  std::string_view accepts;
};

ex::Rung GetRowsRung(const ex::CapabilityQuery& q) {
  const auto* table = Operand(q, 0);
  return table && std::ranges::contains(kQuantized, table->encoding) ? ex::Rung::kOnChipDecode
                                                                     : ex::Rung::kNative;
}

// Concrete numerical facts, including the group's actual routing. A BF16
// weight alone does not prove input rounding: this proof covers the resolved
// cuBLAS BF16 operands/F32 accumulation path with declared output storage;
// it makes no claim about MMVF/MMF arithmetic. Explicit F32 accumulation
// with an F32 input makes cuBLAS widen both operands.
ir::ProductArithmetic Bf16ProductArithmetic(const ex::CapabilityQuery& q) {
  const auto* weight = Operand(q, 0);
  const auto* input = Operand(q, 1);
  const auto* output = Operand(q, ex::kOutputOperand);
  const auto p = ParamsOf<ir::ProductParams>(q);
  if (!weight || !input || !output || weight->encoding != Encoding::kBF16 ||
      (output->encoding != Encoding::kF32 && output->encoding != Encoding::kBF16) ||
      weight->ne[2] != 1 || weight->ne[3] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
      q.arch < 80 || !CompiledFor(q.arch))
    return ir::ProductArithmetic::kImplementation;
  // Both supported accumulation requests use CUBLAS_COMPUTE_32F for
  // already-BF16 operands. RunGemmBf16 and its pinned Lt path use F32
  // accumulation with either declared F32 or BF16 output storage. A Hadamard
  // hint is only an optional algorithm
  // hint; the ordinary BF16 GEMM still computes those actual weights.
  if (input->encoding == Encoding::kBF16 && input->layout == Layout::kDense && p &&
      (p->accumulate == ir::Accumulate::kDefault || p->accumulate == ir::Accumulate::kF32) &&
      (p->hint == ir::ProductHint::kNone || p->hint == ir::ProductHint::kHadamard))
    return ir::ProductArithmetic::kBf16F32;
  if (input->encoding == Encoding::kF32 && output->encoding == Encoding::kF32 && p &&
      p->accumulate == ir::Accumulate::kDefault && p->hint == ir::ProductHint::kNone &&
      FloatProductUsesCublas(q))
    return ir::ProductArithmetic::kBf16F32;
  return ir::ProductArithmetic::kImplementation;
}
// Both ordinary MMVQ and its explicit Prepare use quantize_row_q8_1_cuda:
// 32-value max-abs blocks, FP16 scale/sum, signed int8 rounded values and
// zero-filled padding to 512 values. The prepared product calls the original
// ggml_cuda_op_mul_mat_vec_q with that exact padded stride. MMQ and VecQ
// have different packing/reduction recipes and deliberately do not claim it.
ir::ProductArithmetic Q8ProductArithmetic(const ex::CapabilityQuery& q) {
  const auto* weight = Operand(q, 0);
  const auto* input = Operand(q, 1);
  const auto* output = Operand(q, ex::kOutputOperand);
  const auto params = ParamsOf<ir::ProductParams>(q);
  constexpr auto unknown = ir::ProductArithmetic::kImplementation;
  if (q.op != ir::Primitive(Op::kMatMul) || !Built(q) || !weight || !input || !output || !params ||
      params->hint != ir::ProductHint::kNone || q.k == 0 || q.k > INT32_MAX - 511 || q.k % 32 ||
      q.rows == 0 || q.rows > 8 || weight->layout != Layout::kGgmlBlocks ||
      !std::ranges::contains(kPreparedWeights, weight->encoding) ||
      weight->ne != std::array<std::uint64_t, 4>{q.k, q.n, 1, 1} ||
      input->ne != std::array<std::uint64_t, 4>{q.k, q.rows, 1, 1} ||
      output->ne != std::array<std::uint64_t, 4>{q.n, q.rows, 1, 1} ||
      output->encoding != Encoding::kF32 || output->layout != Layout::kDense || !weight->memory ||
      weight->memory->is_view || weight->memory->view_offset != 0 || !input->memory ||
      input->memory->is_view || input->memory->view_offset != 0 ||
      !PackedOf(output, sizeof(float)) || !Routed(q))
    return unknown;
  const auto& info = format::Info(weight->encoding);
  if (!info.ggml || q.k % info.ggml->block_elements) return unknown;
  const auto packed = [](const ex::OperandFormat& operand, std::uint64_t bytes,
                         std::uint64_t inner) {
    if (operand.memory->nb[0] != bytes) return false;
    std::uint64_t stride = bytes * inner;
    for (std::size_t axis = 1; axis < 4; ++axis) {
      if (operand.memory->nb[axis] != stride ||
          __builtin_mul_overflow(stride, operand.ne[axis], &stride))
        return false;
    }
    return true;
  };
  if (!packed(*weight, info.ggml->block_bytes, q.k / info.ggml->block_elements)) return unknown;
  if (input->encoding == Encoding::kF32 && input->layout == Layout::kDense &&
      packed(*input, sizeof(float), q.k))
    return ir::ProductArithmetic::kBlock32Q8SumF32;
  // Logical Q8_1 row strides describe unpadded blocks; the owned Prepare
  // lowering authenticates and funds the complete physical padded rows.
  if (input->encoding == Encoding::kQ8_1 && input->layout == Layout::kGgmlBlocks &&
      packed(*input, 36, q.k / 32))
    return ir::ProductArithmetic::kBlock32Q8SumF32;
  return unknown;
}

constexpr ex::Capability Cap(ir::OpKey op, std::span<const Rule> rules,
                             bool (*accepts)(const ex::CapabilityQuery&), bool row_invariant,
                             std::string_view group = {}, std::uint16_t priority = 0,
                             ex::ShapeRule shape = {}, std::uint16_t batch = 0,
                             std::span<const std::string_view> launches = {},
                             bool needs_state_span = false) {
  ex::Capability result{
      .op = op,
      .operands = rules,
      .shape = shape,
      .rung = ex::Rung::kNative,
      .expansion = ex::Expansion::kNone,
      .lossy = false,
      .row_invariant = row_invariant,
      .capturable = true,
      .group = group,
      .priority = static_cast<std::uint16_t>(
          priority + ((op.op == Op::kMatMul || op.op == Op::kMatMulId || op.op == Op::kGetRows ||
                       op.op == Op::kAdd || op.fused == FusedKind::kMatMulGlu ||
                       op.fused == FusedKind::kMatMulIdGlu)
                          ? 8
                          : 0)),
      .batch = batch,
      .accepts = accepts,
      .launches = launches,
      .needs_state_span = needs_state_span};
  const bool reads_quantized = op.op == Op::kMatMul || op.op == Op::kMatMulId ||
                               op.op == Op::kGetRows || op.op == Op::kAttention ||
                               (op.op == Op::kFused && op.fused != FusedKind::kGroupedStores &&
                                op.fused != FusedKind::kRopeStore);
  if (reads_quantized) {
    for (const auto& rule : rules) {
      if (rule.operand >= ex::kOutputOperand) continue;
      for (const auto encoding : rule.encodings)
        if (std::ranges::contains(kQuantized, encoding) || encoding == Encoding::kNvfp4 ||
            encoding == Encoding::kMxfp8 || encoding == Encoding::kMxfp4 ||
            encoding == Encoding::kAffine)
          result.rung = ex::Rung::kOnChipDecode;
    }
  }
  if (op == ir::Primitive(Op::kGetRows) && rules.empty()) {
    result.rung = ex::Rung::kOnChipDecode;
    result.rung_for = &GetRowsRung;
  }
  if (group == "ggml.mul_mat" && std::ranges::any_of(rules, [](const Rule& r) {
        return r.operand == 0 && std::ranges::contains(r.encodings, Encoding::kF16);
      })) {
    result.expansion = ex::Expansion::kDenseF32Accumulation;
    result.expands = &FloatProductUsesCublas;
  }
  if (op.op == Op::kMatMul) {
    if (group == "ggml.mul_mat" && accepts == &AcceptsFloatProduct)
      result.product_arithmetic = &Bf16ProductArithmetic;
    if (std::ranges::any_of(rules, [](const Rule& rule) {
          return rule.operand == 1 && std::ranges::contains(rule.encodings, Encoding::kBF16);
        }))
      result.product_arithmetic = &Bf16ProductArithmetic;
    if ((group == "ggml.mul_mat" && accepts == &AcceptsMmvq) || accepts == &AcceptsPrepared)
      result.product_arithmetic = &Q8ProductArithmetic;
  }
  return result;
}

// Its output 0 at source 0's row pitch (ex::Capability::pitched_output).
constexpr ex::Capability Pitched(ex::Capability cap) {
  cap.pitched_output = true;
  return cap;
}

constexpr ex::Capability NativeBlockScaled(ex::Capability cap, std::uint32_t features) {
  cap.rung = ex::Rung::kNative;
  cap.device.needs = features;
  return cap;
}

constexpr ex::Capability F32MaskedAttention() {
  auto cap = Cap(ir::Primitive(Op::kAttention), kF32Attention, &AcceptsAttentionF32, false);
  cap.needs_tuning = true;
  return cap;
}

constexpr ex::Capability Bf16VectorProduct() {
  auto cap = Cap(ir::Primitive(Op::kMatMul), kBf16VectorProduct, &AcceptsBf16VectorProduct, false,
                 {}, 3, {.min_rows = 1, .max_rows = 8, .k_multiple = 2, .n_multiple = 2});
  cap.needs_tuning = false;
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
constexpr ex::Capability SpatialTiled() {
  // The qualified geometry predicate supplies default eligibility. Explicit
  // caller ranks still take precedence over this measured ordinary preference.
  // One step: the bias rides on the convolution (lower.cc).
  return Cap(P(Op::kConv2d), kSpatialConvRules, &AcceptsSpatialTiled, false, {}, 1);
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
constexpr std::array<std::string_view, 3> kDsv4HcPreSteps = {kDsv4HcMixName, kDsv4HcPreName,
                                                             kContName};
constexpr std::string_view kMulMatIdNvfp4Name = "ggml.mul_mat_id.nvfp4";
constexpr std::array<std::string_view, 3> kMulMatIdSteps = {kMulMatIdVecQ, kMulMatIdQ,
                                                            kMulMatIdVecQRows};
constexpr std::string_view kAttentionGroup = "ggml.flash_attn_ext";
constexpr std::string_view kMulMatIdGroup = "ggml.mul_mat_id";
constexpr std::string_view kBuilt = "; selectable for the build's SASS architectures";
constexpr std::string_view kMmvqText =
    "; selectable where upstream's ggml_cuda_should_use_mmvq takes the product";
constexpr std::string_view kMmqText =
    "; selectable where upstream's ggml_cuda_should_use_mmvq does not take the product";

const std::array kEntries = {
    // Gathers, elementwise, norms.
    Entry{"ggml.get_rows", Cap(P(Op::kGetRows), kFloatTable, &AcceptsAny, true, {}, 1), kBuilt},
    Entry{"ggml.get_rows.ext", Cap(P(Op::kGetRows), {}, &AcceptsGetRowsExt, true), kBuilt},
    Entry{kConv2dTiledF32, SpatialTiled(), kBuilt},
    Entry{kConv1x1Name, Cap(P(Op::kConv2d), kSpatialConvRules, &AcceptsConv1x1, false, {}, 2),
          kBuilt},
    Entry{"ggml.conv2d.f32",
          Cap(P(Op::kConv2d), kSpatialConvRules, &AcceptsSpatialConv, false, {}, 0, {}, 0,
              kSpatialConvSteps),
          kBuilt},
    Entry{kConv2dStridedName,
          Cap(P(Op::kConv2d), kSpatialConvFallbackRules, &AcceptsSpatialConvStrided, false, {}, 0),
          kBuilt},
    Entry{"ggml.upsample.nearest", Cap(P(Op::kUpsample), kSpatialUpRules, &AcceptsSpatialUp, false),
          kBuilt},
    Entry{"ggml.scale", Cap(P(Op::kScale), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.add", Cap(P(Op::kAdd), kAddRules, &AcceptsAdd, true), kBuilt},
    Entry{"ggml.mul", Cap(P(Op::kMul), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.sub", Cap(P(Op::kSub), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.div", Cap(P(Op::kDiv), kF32Binary, &AcceptsAny, true), kBuilt},
    Entry{"ggml.unary", Cap(P(Op::kUnary), kF32In, &AcceptsUnary, true), kBuilt},
    Entry{"ggml.clamp", Cap(P(Op::kClamp), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.fill", Cap(P(Op::kFill), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.repeat", Cap(P(Op::kRepeat), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.concat", Cap(P(Op::kConcat), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.cont", Cap(P(Op::kCont), {}, &AcceptsCont, true), kBuilt},
    Entry{"ggml.sum_rows", Cap(P(Op::kSumRows), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.soft_max", Cap(P(Op::kSoftMax), {}, &AcceptsAny, true), kBuilt},
    Entry{"ggml.argsort.bitonic", Cap(P(Op::kArgsort), kF32In, &AcceptsAny, true), kBuilt},
    Entry{kLightningIndexerName,
          Cap(P(Op::kIndexScore), kIndexScoreRules, &AcceptsIndexScore, true),
          "; selectable for the build's SASS architectures: D 128, 32 or 64 indexer heads"},
    Entry{kVisibleTopKName, Cap(P(Op::kVisibleTopK), kVisibleTopKRules, &AcceptsVisibleTopK, true),
          kBuilt},
    Entry{
        kRoundedIndexScoreName,
        Cap(P(Op::kRoundedIndexScore), kRoundedIndexScoreRules, &AcceptsRoundedIndexScore, false),
        "; explicit root alignment and native pitches; paid literal fill then original Lid groups"},
    Entry{"llmp.select", Cap(P(Op::kSelect), {}, &AcceptsSelect, true),
          "; I32 nonzero predicate; same dense scalar branches; bounded broadcast byte copies"},
    Entry{kPrefixOverwriteName,
          Cap(P(Op::kPrefixOverwrite), kPrefixOverwriteRules, &AcceptsPrefixOverwrite, true),
          kBuilt},
    Entry{kSelectedMaskName,
          Cap(P(Op::kSelectedMask), kSelectedMaskRules, &AcceptsSelectedMask, true), kBuilt},
    Entry{kTopKName, Cap(P(Op::kTopK), kF32In, &AcceptsTopK, true),
          "; selectable for the build's SASS architectures: F32 rows within I32 counts"},
    Entry{"ggml.norm", Cap(P(Op::kLayerNorm), kF32In, &AcceptsLayerNorm, true),
          "; locked centered F32 normalization over packed rows; finite nonnegative epsilon"},
    Entry{"ggml.l2_norm", Cap(P(Op::kL2Norm), kF32In, &AcceptsL2Norm, true), kBuilt},
    Entry{kSpatialChannelNormName,
          Cap(F(FusedKind::kSpatialChannelNorm), kF32Binary, &AcceptsSpatialChannelNorm, true),
          "; selectable for the build's SASS architectures: packed F32 [W, H, C, 1] and an F32 "
          "[C] gamma"},
    Entry{kNormRopeSectionsName,
          Cap(F(FusedKind::kRmsNormMulRopeSections), kNormRopeSectionsRules,
              &AcceptsNormRopeSections, true),
          "; selectable for the build's SASS architectures: packed F32 or BF16 heads of D at most "
          "512, an F32 or BF16 [D] weight, one to three plain sections covering D"},
    Entry{"ggml.rms_norm", Cap(P(Op::kRmsNorm), kF32In, &AcceptsAny, true), kBuilt},
    Entry{"ggml.rms_norm_mul.fused", Cap(F(FusedKind::kRmsNormMul), kNormMul, &AcceptsAny, true),
          kBuilt},
    Entry{kGemmaNormRopeName,
          Cap(F(FusedKind::kRmsNormMulRope), kNormMulRope, &AcceptsNormRope, true),
          "; full even-width NEOX rotation within launcher integer bounds "
          "with F32 frequency factors or none, its parameters finite"},
    Entry{kGemmaNormAddName, Cap(F(FusedKind::kRmsNormMulAdd), kNormMulAdd, &AcceptsNormAdd, true),
          "; positive F32 widths within launcher integer bounds, finite nonnegative epsilon"},
    // Rotations and stores.
    Entry{"ggml.rope.neox", Cap(P(Op::kRope), kRope, &AcceptsRopeNeox, true, {}, 1),
          "; selectable for the build's SASS architectures: NEOX over positions without "
          "frequency factors or an offset; proportional pairs use the owned primitive "
          "and a separate cache store"},
    Entry{"ggml.rope.ext", Cap(P(Op::kRope), kRope, &AcceptsRopeExt, true),
          "; selectable for the build's SASS architectures over positions; proportional pairs "
          "use the owned primitive and a separate cache store"},
    Entry{"ggml.set_rows", Cap(P(Op::kCacheStore), kF16Store, &AcceptsAny, true, {}, 1), kBuilt},
    Entry{"ggml.set_rows.ext", Cap(P(Op::kCacheStore), kFloatStore, &AcceptsAny, true), kBuilt},
    Entry{kSetRowsGroupedName,
          Cap(F(FusedKind::kGroupedStores), kF16Store, &AcceptsAny, true, {}, 0, {},
              static_cast<std::uint16_t>(kSetRowsGroupMax)),
          kBuilt},
    Entry{"ggml.rope_set_rows.fused",
          Cap(F(FusedKind::kRopeStore), kF16RopeStore, &AcceptsRopeStore, true),
          "; selectable for the build's SASS architectures: NEOX into F16 rows"},
    // Products.
    Entry{"ggml.mul_mat.mmvq",
          Cap(P(Op::kMatMul), kQuantProduct, &AcceptsMmvq, false, kMulMatGroup, 1), kMmvqText},
    Entry{"ggml.mul_mat.mmq", Cap(P(Op::kMatMul), kQuantProduct, &AcceptsMmq, false, kMulMatGroup),
          kMmqText},
    Entry{kMulMatVecFRows, Bf16VectorProduct(),
          "; exact output-typed tuning; packed BF16/F32/F32 product, 1..8 physical columns"},
    Entry{"ggml.mul_mat.mmvf",
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup, 2), kBuilt},
    Entry{"ggml.mul_mat.mmf",
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup, 1), kBuilt},
    Entry{"ggml.mul_mat.cublas",
          Cap(P(Op::kMatMul), kFloatProduct, &AcceptsFloatProduct, false, kMulMatGroup), kBuilt},
    Entry{"ggml.mul_mat.fwht", Cap(P(Op::kMatMul), kFloatProduct, &AcceptsHadamard, false, {}, 3),
          "; selectable for the build's SASS architectures: a square Hadamard hint"},
    Entry{
        "ggml.mul_mat_add.mmvf_fused",
        Cap(F(FusedKind::kMatMulAdd), kF16ProductAdd, &AcceptsFloatProductFusion, false),
        "; selectable for the build's SASS architectures: one F32 input column, one dense "
        "F16 matrix with even K, F32 add of the same shape; actual MMVF and fusion gates checked"},
    Entry{"ggml.mul_mat_glu.mmvf_fused",
          Cap(F(FusedKind::kMatMulGlu), kF16ProductGlu, &AcceptsFloatSwiGlu, false),
          "; selectable for the build's SASS architectures: one F32 input column, dense F16 "
          "gate/up matrices with even K, split unswapped SwiGLU; actual MMVF and fusion gates "
          "checked"},
    // Routed experts: the expert products by upstream's routing (MMVQ or
    // MMQ, ggml_cuda_mul_mat_id), and Gemma's fused routing and scaled
    // reduce (gemma_moe_fusion.h).
    Entry{kMulMatIdVecQ,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup, 1), kBuilt},
    Entry{kMulMatIdQ, Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup),
          kBuilt},
    Entry{kMulMatIdQCompact,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kMulMatIdQCompactPrequant,
          Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup), kBuilt},
    Entry{kMulMatIdQ2D2r, Cap(P(Op::kMatMulId), kQuantExperts, &AcceptsAny, false, kMulMatIdGroup),
          kBuilt},
    Entry{kDsv4RouteName, Cap(P(Op::kRoute), kSelectedRouteRules, &AcceptsSelectedRoute, true),
          "; typed selected routing of 1..256 experts, used 1..min(experts,32): hash rows <=8, "
          "biased rows <=min(16,native vector bound), under the supplied arithmetic policy"},
    Entry{kGemmaRouteName, Pitched(Cap(P(Op::kRoute), kRouteLogits, &AcceptsGemmaRoute, true)),
          "; compiled expert counts, valid top-k, scale1 and int-bounded row offsets"},
    Entry{kGemmaReduceName, Cap(P(Op::kExpertReduce), kScaledReduce, &AcceptsGemmaReduce, true),
          "; F32 expert slots, each scale gathered, within the 2-D grid and uint64 indexing"},
    Entry{"llmp.q8_1",
          Cap(P(Op::kPrepare), kQ8Prepare, &AcceptsAny, true, {}, 0, {.k_multiple = 32}),
          "; selectable for the build's SASS architectures: rows of whole Q8_1 blocks"},
    Entry{"llmp.mmvq.prepared",
          Cap(P(Op::kMatMul), kPreparedProduct, &AcceptsPrepared, false, {}, 0,
              {.min_rows = 1, .max_rows = 8, .k_multiple = 32}),
          kMmvqText},
    Entry{"ggml.mul_mat_geglu.mmvq_fused",
          Cap(F(FusedKind::kMatMulGlu), kQuantGlu, &AcceptsQuantGeGlu, false),
          "; selectable for the build's SASS architectures past Pascal: GELU-tanh of one column "
          "over gate and up weights of one type, which upstream's ggml_cuda_should_use_mmvq "
          "takes"},
    Entry{"ggml.mul_mat_glu.mmvq_fused",
          Cap(F(FusedKind::kMatMulGlu), kQuantGlu, &AcceptsQuantSwiGlu, false),
          "; selectable for the build's SASS architectures past Pascal: SwiGLU of one column "
          "over gate and up weights of one type, which upstream's ggml_cuda_should_use_mmvq "
          "takes"},
    Entry{"ggml.geglu", Cap(P(Op::kGlu), kF32Glu, &AcceptsGeGlu, true),
          "; selectable for the build's SASS architectures: split GELU-tanh, not swapped"},
    Entry{kAttentionMaskPackName,
          Cap(F(FusedKind::kAttentionMaskPack), kMaskPackRules, &AcceptsMaskPack, true),
          "; selectable for the build's SASS architectures: an F32 mask into F16 padded to "
          "multiples of 256 columns"},
    Entry{kRopeTableName, Cap(F(FusedKind::kRopeTable), kRopeTableRules, &AcceptsRopeTable, true),
          "; selectable for the build's SASS architectures: one to three normal sections of a "
          "head of at most 512"},
    Entry{kQkAttentionName,
          Cap(F(FusedKind::kQkAttention), kQkAttentionRules, &AcceptsQkAttention, false),
          "; selectable from compute capability 8.0: BF16 heads of 128, a packed mask and table"},
    Entry{kLayerNormModulateSplitName,
          Cap(F(FusedKind::kLayerNormModulateSplit), kModulateSplitRules, &AcceptsModulateSplit,
              true),
          "; selectable for the build's SASS architectures: packed F32 rows of at most 65536"},
    Entry{kGatedResidualModulateSplitName,
          Cap(F(FusedKind::kGatedResidualModulateSplit), kGatedModulateSplitRules,
              &AcceptsGatedModulateSplit, true),
          "; selectable for the build's SASS architectures: the residual's sum and its "
          "modulation in one pass over packed rows"},
    Entry{kDupUpAddName, Cap(F(FusedKind::kDupUpAdd), kF32Binary, &AcceptsDupUpAdd, false),
          "; selectable for the build's SASS architectures: packed F32 of one image, an int "
          "element count"},
    Entry{kGatedResidualSplitName,
          Cap(F(FusedKind::kGatedResidualSplit), kGatedSplitRules, &AcceptsGatedSplit, true),
          "; selectable for the build's SASS architectures: packed rows of int elements"},
    Entry{kSwiGluBf16Name, Cap(P(Op::kGlu), kSwiGluBf16Rules, &AcceptsSwiGluBf16, true),
          "; selectable for the build's SASS architectures: a split SwiGLU of F32 or BF16 "
          "operands of one shape into BF16"},
    Entry{"ggml.swiglu", Cap(P(Op::kGlu), kF32Glu, &AcceptsSwiGlu, true),
          "; selectable for the build's SASS architectures: split SwiGLU, not swapped"},
    Entry{"ggml.swiglu_clamp", Cap(P(Op::kGlu), kF32Glu, &AcceptsSwiGluClamp, true),
          "; selectable for the build's SASS architectures: split clamped SwiGLU"},
    // Attention and its masks.
    Entry{kAttentionF32, F32MaskedAttention(),
          "; opt-in packed D128 MHA: transient F16 Q/K/V MMA operands and additive mask, "
          "F32 accumulation/output; finite F16 operand range must be qualified"},
    Entry{"ggml.state_bank_attention",
          Cap(F(FusedKind::kStateBankAttention), kStateBankAttention, &AcceptsStateBankAttention,
              false, kAttentionGroup, 0, {}, 0, {}, true),
          "; the existing flash-attention group over a proven contiguous F16 state span"},
    Entry{"ggml.flash_attn_ext.vec",
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention64, false, kAttentionGroup, 1),
          "; selectable for the build's SASS architectures: D64 F32 query, F16 K/V/mask, "
          "256-padded cells, no sinks, softcap or ALiBi and F32 precision"},
    Entry{"ggml.flash_attn_ext.mma_common",
          Cap(P(Op::kAttention), kCommonF16Attention, &AcceptsCommonAttention, false,
              kAttentionGroup),
          "; compiled D64/80/96/112/128/192 group-one MMA, F16 KV/mask, arbitrary GQA"},
    Entry{"ggml.flash_attn_ext.vec_common",
          Cap(P(Op::kAttention), kCommonF16Attention, &AcceptsCommonVectorAttention, false,
              kAttentionGroup),
          "; compiled D64/128/192 vector kernels with 256-padded KV cells"},
    Entry{"ggml.flash_attn_ext.vec_d256",
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 2),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma_gqa2",
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention256, false, kAttentionGroup, 1),
          "; selectable for the build's SASS architectures at head size 256"},
    Entry{"ggml.flash_attn_ext.mma",
          Cap(P(Op::kAttention), kF16Attention, &AcceptsAttention, false, kAttentionGroup),
          "; selectable for the build's SASS architectures at head size 256 or 512"},
    Entry{kFlashAttnMmaWideName,
          Cap(P(Op::kAttention), kF16Attention, &AcceptsWideAttention, false, kAttentionGroup, 1),
          "; a declared D512 sparse-any recipe over bounded F16 K/V and masks"},
    Entry{kFlashAttnOwnersName,
          Cap(F(FusedKind::kOwnerAttention), kOwnerAttention, &AcceptsOwnerAttention, false),
          "; selectable for the build's SASS architectures: CheckFlashAttnOwners' Gemma "
          "cohorts, head sizes, cells and rows over independent F16 caches, a softcap of 0 or "
          "50, and bounded roots for two owners alone; requires a 32-lane warp"},
    Entry{kCausalMaskName, Cap(P(Op::kMaskBuild), kMaskPositions, &AcceptsMask, true),
          "; selectable for the build's SASS architectures: causal, window and ahead masks"},
    Entry{"llmp.argmax", Cap(P(Op::kArgmax), kF32In, &AcceptsArgmax, true), kBuilt},
    Entry{"llmp.dsv4.lid_mask",
          Cap(F(FusedKind::kRoundedIndexMask), kDsv4LidMaskRules, &AcceptsDsv4LidMask, false, {}, 0,
              {}, 0, kDsv4LidMaskSteps),
          "; closed rounded-query score, visible selection and complete sparse mask"},
    Entry{kDsv4QHeadName,
          Cap(F(FusedKind::kRmsNormMulRope), kDsv4QHeadRules, &AcceptsDsv4QHead, true),
          "; packed F32 [512, 64, 16..8192] Q heads and normal rotary tail"},
    Entry{kDsv4CompressName,
          Cap(F(FusedKind::kSoftmaxPoolCompress), kDsv4CompressRules, &AcceptsDsv4Compress, true),
          "; F32 compressor values/scores, ratio 1..1024, at most 65535 blocks"},
    Entry{kHcCombName, Cap(P(Op::kHcMix), kSinkhornHcMixRules, &AcceptsSinkhornHcMix, true),
          "; packed F32 four-stream Sinkhorn, 1..100 iterations; native ordered reductions"},
    Entry{kHcPreName, Cap(P(Op::kHcPre), kWeightedHcPreRules, &AcceptsWeightedHcPre, true),
          "; packed F32 weighted fold of four streams; the original native ordered reduction"},
    Entry{kDsv4HcPrePackedName,
          Cap(F(FusedKind::kSinkhornHcPre), kDsv4HcPreRules, &AcceptsDsv4HcPre, true, {}, 0, {}, 0,
              kDsv4HcPreSteps),
          "; full Sinkhorn carry, four F32 streams, widths 256 to 8192; packed carry copies"},
    Entry{kHcPostName, Cap(P(Op::kHcPost), kHcPostRules, &AcceptsHcPost, true),
          "four-stream F32 HC post with ordered native FMAs, element strides and bounded grid"},
    Entry{kDsv4HcPostNormF16Name,
          Cap(F(FusedKind::kHcNormPost), kDsv4HcPostRules, &AcceptsDsv4HcPost, true),
          "; four 4096-value streams, post and flat F32 norm"},
    Entry{kDsv4OutAFastPackName,
          Cap(F(FusedKind::kInverseRopeGroupedMatMul), kDsv4OutARules, &AcceptsDsv4OutA, true),
          "; paid raw-Q8 packing, inverse tail64 RoPE, staged F16/HMMA grouped output-A"},
    // Qwen3.8's fast forms (slice G3b): the MXFP8 and BF16
    // products, the n-gram tables' rows, the hyper-connections, Gated
    // DeltaNet, QSA and the routed experts.
    Entry{kMxfp8MulMatVecName, Cap(P(Op::kMatMul), kMxfp8Vec, &AcceptsMxfp8Vec, false, {}, 4),
          "; selectable for the build's SASS architectures: up to 16 columns"},
    Entry{kMxfp8GemmName,
          NativeBlockScaled(Cap(P(Op::kMatMul), kMxfp8Gemm, &AcceptsMxfp8Gemm, false, {}, 4, {}, 0,
                                kMxfp8GemmSteps),
                            ex::kFp8Mma | ex::kBlockScaledMma),
          "; selectable on sm_121 past 16 columns, K of whole 128-value rows"},
    Entry{kMxfp8QuantizeName, Cap(P(Op::kPrepare), kMxfp8Prepare, &AcceptsMxfp8Quantize, true),
          "; selectable on sm_121: rows of whole 128 values"},
    Entry{kGemmBf16Name, Cap(P(Op::kMatMul), kBf16Product, &AcceptsBf16MatrixProduct, false, {}, 4),
          kBuilt},
    // The pinned form below the plain one: a policy that ranks it (the
    // image's) takes it.
    Entry{kGemmBf16LtName,
          Cap(P(Op::kMatMul), kBf16Product, &AcceptsBf16MatrixProduct, false, {}, 3), kBuilt},
    Entry{"ggml.convert", Cap(P(Op::kConvert), kFloatConvert, &AcceptsConvert, true),
          "; selectable for the build's SASS architectures: F16/F32 casts within I32 counts"},
    Entry{kBf16Name, Cap(P(Op::kConvert), kToBf16, &AcceptsAny, true), kBuilt},
    Entry{kNvfp4RowsName, Cap(P(Op::kGetRows), kNvfp4Table, &AcceptsNvfp4Rows, true, {}, 2),
          kBuilt},
    Entry{kQRowsName, Cap(P(Op::kGetRows), kQRowsTable, &AcceptsQRows, true, {}, 2),
          "; selectable for the build's SASS architectures: 32-value blocks of rows GGML's "
          "get_rows does not take"},
    Entry{kHcPrepName, Cap(F(FusedKind::kHcPrep), kHcPrepRules, &AcceptsHcPrep, true),
          "; selectable for the build's SASS architectures: 2 to 8 streams of widths of 8 to "
          "4,096, a BF16 inject"},
    Entry{kHcLoName, Cap(F(FusedKind::kScaledUnary), kHcLoRules, &AcceptsHcLo, true),
          "; selectable for the build's SASS architectures: SiLU scaled by the reciprocal of "
          "1 to 8 streams"},
    Entry{kHcMixBf16Name, Cap(F(FusedKind::kHcFold), kHcFoldRules, &AcceptsHcFold, true),
          "; selectable for the build's SASS architectures: 2 to 8 streams; prepared past 16 "
          "rows of whole 128-value rows"},
    Entry{kHcCombineName, Cap(F(FusedKind::kHcCombine), kHcCombineRules, &AcceptsHcCombine, true),
          kBuilt},
    Entry{kHcNormName, Cap(F(FusedKind::kHcNorm), kHcNormRules, &AcceptsHcNorm, true),
          "; selectable for the build's SASS architectures: streams of at least 1,024"},
    Entry{kHcMixName, Cap(F(FusedKind::kHcFold), kHcMixRules, &AcceptsHcMix, true),
          "; selectable for the build's SASS architectures: streams of at least 1,024"},
    Entry{kSsmConvName,
          Cap(P(Op::kSsmConv), kSsmConvRules, &AcceptsSsmConv, true, {}, 0, {}, 0, kSsmConvSteps),
          "; one sequence; past 32 tokens of a partial window, its input padded"},
    Entry{kGdnConvName, Cap(F(FusedKind::kGdnConv), kGdnConvRules, &AcceptsGdnConv, true),
          "; selectable for the build's SASS architectures: heads of 128, a kernel of 4"},
    Entry{kGdnHistoryName, Cap(F(FusedKind::kGdnHistory), kRowsRules, &AcceptsGdnHistory, true),
          kBuilt},
    Entry{kGdnStepName, Cap(F(FusedKind::kGdnStep), kGatedDeltaRules, &AcceptsGdnStep, false),
          "; selectable for the build's SASS architectures: heads of 128, up to 16 tokens"},
    Entry{kGdnGatesName, Cap(F(FusedKind::kGdnGates), kHcNormRules, &AcceptsGdnGates, true),
          "; selectable for the build's SASS architectures: positive heads/rows, representable F32 "
          "planes"},
    Entry{kGdnNormGateName, Cap(F(FusedKind::kNormGate), kNormGateRules, &AcceptsGdnNormGate, true),
          "; selectable for the build's SASS architectures: heads of 128"},
    Entry{kGatedDeltaNetName,
          Cap(P(Op::kGatedDeltaNet), kGatedDeltaRules, &AcceptsGatedDelta, false, kGatedDeltaGroup),
          kBuilt},
    Entry{kGatedDeltaNetColumnsName,
          Cap(P(Op::kGatedDeltaNet), kGatedDeltaRules, &AcceptsGatedDelta, false, kGatedDeltaGroup,
              1),
          kBuilt},
    Entry{kGatedDeltaNetLanesName,
          Cap(P(Op::kGatedDeltaNet), kGatedDeltaRules, &AcceptsGatedDelta, false, kGatedDeltaGroup,
              2),
          kBuilt},
    Entry{kQsaPrepName,
          Cap(F(FusedKind::kRmsNormMulRope), kNormMulRope, &AcceptsQsaPrep, true, {}, 1),
          "; selectable for the build's SASS architectures: the interleaved multi-section "
          "rotation of 64 dimensions over heads of 64 to 512, without frequency or amplitude "
          "scaling; lowering requires typed replicated position sections"},
    Entry{kQsaPoolName,
          Cap(F(FusedKind::kMeanPoolNormRopeStore), kQsaPoolRules, &AcceptsQsaPool, true),
          "; selectable for the build's SASS architectures: blocks of 1 to 32 cells, with a "
          "forward unscaled 64-dimension interleaved rotation; lowering requires canonical "
          "typed block positions/cells and stored raw keys"},
    Entry{kQsaAttentionName,
          Cap(F(FusedKind::kBlockSparseAttention), kQsaAttnRules, &AcceptsQsaAttention, false, {},
              0, {}, 0, kQsaAttentionSteps),
          "; selectable for the build's SASS architectures: heads of 256, at most 16 a KV head, "
          "an indexer of 4 heads of 128, the selection within its tiles; lowering requires "
          "canonical typed position tables, stores and causal masks"},
    Entry{kQsaGateQuantizeName,
          Cap(F(FusedKind::kHeadGate), kHeadGateRules, &AcceptsHeadGate, true),
          "; selectable on sm_121: heads of whole 32 values into MXFP8 rows"},
    Entry{kMoeRouterName, Cap(F(FusedKind::kMoeRouter), kMoeRouterRules, &AcceptsMoeRouter, true),
          "; selectable for the build's SASS architectures: a normalized softmax top-k of up to "
          "32 of up to 1,024 experts"},
    Entry{kMoeExpertsName,
          NativeBlockScaled(Cap(F(FusedKind::kMoeExperts), kCutlassExperts, &AcceptsMoeExperts,
                                false, {}, 0, {}, 0, kMoeExpertsSteps),
                            ex::kFp4Mma | ex::kBlockScaledMma),
          "; selectable on sm_121: the CUTLASS layout's experts with their scales, SwiGLU"},
    Entry{kMoeGluName, Cap(F(FusedKind::kMoeGlu), kF32Binary, &AcceptsMoeGlu, true), kBuilt},
    Entry{kDsv4CombineName,
          Cap(F(FusedKind::kMoeCombine), kSelectedCombineRules, &AcceptsSelectedCombine, true),
          "; packed ungated ordered-FMA combination, used 1..32, rows 1..8"},
    Entry{kMoeCombineName, Cap(F(FusedKind::kMoeCombine), kF32In, &AcceptsGatedCombine, true),
          kBuilt},
    Entry{"llmp.vecq.grouped",
          Cap(F(FusedKind::kGroupedMatMul), kGroupedVecQRules, &AcceptsGroupedVecQ, false, {}, 0,
              {}, 0, kGroupedVecQSteps),
          "packed grouped F32 input, two-to-256 immutable GGML groups, one-to-eight tokens; "
          "original Q8_1 preparation and per-slot VecQ in paid activation space"},
    Entry{kVecQDenseName,
          Cap(P(Op::kMatMul), kVecQProduct, &AcceptsVecQ, false, {}, 3, {}, 0, kVecQSteps),
          "; selectable for the build's SASS architectures: up to 8 tokens"},
    Entry{kVecQGluName,
          Cap(F(FusedKind::kMatMulGlu), kVecQGlu, &AcceptsVecQ, false, {}, 3, {}, 0, kVecQSteps),
          "; selectable for the build's SASS architectures: up to 8 tokens"},
    Entry{kVecQExpertsName,
          Cap(P(Op::kMatMulId), kVecQExperts, &AcceptsVecQRouted, false, {}, 3, {}, 0, kVecQSteps),
          "; selectable for the build's SASS architectures: up to 64 (token, slot) pairs"},
    Entry{kVecQExpertsGluName,
          Cap(F(FusedKind::kMatMulIdGlu), kVecQExpertsGlu, &AcceptsVecQRouted, false, {}, 3, {}, 0,
              kVecQSteps),
          "; selectable for the build's SASS architectures: up to 64 (token, slot) pairs"},
    Entry{kMulMatIdNvfp4Name,
          Cap(P(Op::kMatMulId), kNvfp4Experts, &AcceptsNvfp4Experts, false, {}, 0, {}, 0,
              kMulMatIdSteps),
          kBuilt},
    // Block-quantized KV caches (kv_quant.h).
    Entry{kFlashAttnVecQ8Name, Cap(P(Op::kAttention), kQ8Attention, &AcceptsQuantAttention, false),
          "; selectable for the build's SASS architectures at head size 64, 256 or 512 (K's and "
          "V's), with or without F32 sinks"},
    Entry{kFlashAttnVecQ4Name, Cap(P(Op::kAttention), kQ4Attention, &AcceptsQ4Attention, false),
          "; selectable for the build's SASS architectures at head size 64 or 256 (K's and V's), "
          "with or without F32 sinks"},
    Entry{kSetRowsQuantName,
          Cap(P(Op::kCacheStore), kQuantStore, &AcceptsQuantStore, true, {}, 0, {.k_multiple = 32}),
          "; selectable for the build's SASS architectures, rows of whole 32-value blocks"},
    Entry{kSetRowsQuantGroupName,
          Cap(F(FusedKind::kGroupedStores), kQuantStore, &AcceptsQuantStore, true, {}, 0,
              {.k_multiple = 32}, static_cast<std::uint16_t>(kSetRowsQuantGroupMax)),
          "; selectable for the build's SASS architectures, rows of whole 32-value blocks"},
    Entry{kRopeSetRowsQuantName,
          Cap(F(FusedKind::kRopeStore), kQuantRopeStore, &AcceptsQuantRopeStore, true, {}, 0,
              {.k_multiple = 32}),
          "; selectable for the build's SASS architectures, rows of whole 32-value blocks"},
};

const std::array<CapabilityDeclaration, kEntries.size()> kDeclarations = [] {
  std::array<CapabilityDeclaration, kEntries.size()> out{};
  for (std::size_t i = 0; i < kEntries.size(); ++i) {
    out[i] = {.name = kEntries[i].name, .capability = &kEntries[i].capability};
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

std::span<const RecipeDeclaration> RecipeDeclarations() {
  static constexpr RecipeDeclaration recipes[] = {
#define LLMP_GGML_NORM_RECIPE(NAME, VARIANT, CHECK, RUN) \
  {NAME, ir::Fused(ir::FusedKind::kRmsNormMul), VARIANT, 2},
#define LLMP_GGML_RECIPE(NAME, KEY, VARIANT, ARITY, CHECK, RUN) {NAME, KEY, VARIANT, ARITY},
#include "kernels/ggml/recipe_entries.inc"
#undef LLMP_GGML_RECIPE
#undef LLMP_GGML_NORM_RECIPE
  };
  return recipes;
}

namespace {
// Kernel versions past the first, by implementation: bump an entry (or add
// one at 2) when a change to what that implementation launches could move its
// measured speed, then re-measure or retire its rows in measured_ranks.h
// (tests/unit/fusion_ranks_test.cc refuses a row of another version).
struct KernelVersionEntry {
  std::string_view name;
  std::uint32_t version;
};
constexpr std::array<KernelVersionEntry, 0> kKernelVersions{};
}  // namespace

std::uint32_t KernelVersion(std::string_view implementation) {
  const auto found = std::ranges::find(kKernelVersions, implementation, &KernelVersionEntry::name);
  return found == kKernelVersions.end() ? 1 : found->version;
}

std::vector<execution::Implementation> PlanningImplementations() {
  std::vector<execution::Implementation> out;
  out.reserve(RecipeDeclarations().size() + CapabilityDeclarations().size());
  const auto append = [&](std::string_view name, ir::OpKey op, const execution::Capability* cap) {
    const auto recipes = RecipeDeclarations();
    const auto recipe = std::ranges::find(recipes, name, &RecipeDeclaration::name);
    const auto contract = recipe == recipes.end()
                              ? std::string("ordered launches declared by capability")
                              : std::format("{}; ordered descriptor arity {}",
                                            recipe->numerical_contract, recipe->arity);
    out.push_back({.name = std::string(name),
                   .operation = op,
                   .source = "ggml",
                   .revision = "structural declarations; cannot bind executable kernels",
                   .build = "planning only",
                   .variant = std::format("{}; {}", contract, AcceptsTextOf(name)),
                   .version = KernelVersion(name),
                   .capability = cap ? std::optional(*cap) : std::nullopt});
  };
  // Retain the capability inventory's tie order. Concrete helper-only recipes
  // cannot become selectable candidates merely by being present in a registry.
  for (const auto& d : CapabilityDeclarations()) append(d.name, d.capability->op, d.capability);
  for (const auto& recipe : RecipeDeclarations())
    if (!CapabilityOf(recipe.name)) append(recipe.name, recipe.operation, nullptr);
  return out;
}

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
