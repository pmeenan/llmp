// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The operand checks of the operations DeepSeek V4 Flash and Qwen3.8 Flash
// add (kernels/ggml/validate_ext.h), in every profile and without a GPU:
// nodes at the models' shapes pass, and what each GGML launcher would assert
// on, abort on or index past is refused: wrong types, shapes that do not
// follow from the operands, rows the kernels cannot step, uncompiled
// quantized types and kernel sizes, misalignment, stale views and aliasing.
// Nothing here touches the addresses bound.

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <expected>
#include <limits>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

namespace {

namespace kg = jitllm::kernels::ggml;
using kg::KernelError;
using kg::TensorArena;

constexpr std::uint64_t kBase = 1ULL << 40;  // never dereferenced
constexpr std::uint64_t kSlot = 1ULL << 36;  // far enough apart for any operand here

class GgmlExtValidateTest : public ::testing::Test {
 protected:
  ggml_context* c() { return arena_.context(); }
  // A tensor bound to its own slot, clear of every other.
  ggml_tensor* Bound(ggml_tensor* tensor) {
    TensorArena::Bind(tensor, kBase + (next_++ * kSlot));
    return tensor;
  }
  ggml_tensor* New(ggml_type type, std::int64_t ne0, std::int64_t ne1 = 1, std::int64_t ne2 = 1,
                   std::int64_t ne3 = 1) {
    return Bound(ggml_new_tensor_4d(c(), type, ne0, ne1, ne2, ne3));
  }
  static void Accepted(const std::expected<void, kg::KernelFailure>& checked) {
    EXPECT_TRUE(checked.has_value()) << (checked ? "" : checked.error().detail);
  }
  static void Refused(const std::expected<void, kg::KernelFailure>& checked) {
    ASSERT_FALSE(checked.has_value());
    EXPECT_EQ(checked.error().error, KernelError::kRejected);
  }

  TensorArena arena_ = TensorArena::Create(1024).value();
  std::uint64_t next_ = 0;
};

TEST_F(GgmlExtValidateTest, QuantizedProductsTakeTheCompiledTypesAtWholeRowSteps) {
  EXPECT_EQ(kg::QuantizedWeightTypes().size(), 18U);
  for (const ggml_type type : kg::QuantizedWeightTypes()) {
    EXPECT_TRUE(kg::IsQuantizedWeightType(type)) << ggml_type_name(type);
    ggml_tensor* w = New(type, 4096, 256);
    ggml_tensor* x = New(GGML_TYPE_F32, 4096, 5);
    Accepted(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), w, x))));
  }
  EXPECT_TRUE(kg::IsQuantizedWeightType(GGML_TYPE_NVFP4));   // Qwen3.8's experts
  EXPECT_TRUE(kg::IsQuantizedWeightType(GGML_TYPE_IQ2_S));   // Qwen3.8 GGUF's experts
  EXPECT_TRUE(kg::IsQuantizedWeightType(GGML_TYPE_IQ4_NL));  // and its n-gram table
  EXPECT_FALSE(kg::IsQuantizedWeightType(GGML_TYPE_Q4_1));
  EXPECT_FALSE(kg::IsQuantizedWeightType(GGML_TYPE_IQ1_M));  // no MMQ upstream
  // Rows short of a 512-element step (Qwen3.8's 640-element down
  // projection): refused unless the binder vouches for their padding, and
  // then only in whole blocks.
  ggml_tensor* x640 = New(GGML_TYPE_F32, 640, 5);
  ggml_tensor* nv = New(GGML_TYPE_NVFP4, 640, 256);
  Refused(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), nv, x640))));
  kg::MarkRowPaddingReadable(nv);
  EXPECT_TRUE(kg::RowPaddingReadable(nv));
  Accepted(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), nv, x640))));
  ggml_tensor* x4064 = New(GGML_TYPE_F32, 4064, 5);
  ggml_tensor* torn = New(GGML_TYPE_Q8_0, 4064, 256);  // 127 whole Q8_0 blocks: accepted marked
  kg::MarkRowPaddingReadable(torn);
  Accepted(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), torn, x4064))));
  // A view does not inherit the mark.
  ggml_tensor* view = ggml_view_2d(c(), nv, 640, 128, nv->nb[1], 0);
  EXPECT_FALSE(kg::RowPaddingReadable(view));
  // Not a compiled type; float weights; rows not whole 512-element steps;
  // F16 activations; an output over the weights.
  ggml_tensor* x = New(GGML_TYPE_F32, 4096, 5);
  Refused(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), New(GGML_TYPE_IQ1_M, 4096, 256), x))));
  Refused(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), New(GGML_TYPE_F16, 4096, 256), x))));
  ggml_tensor* short_w = New(GGML_TYPE_Q8_0, 4064, 256);
  Refused(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), short_w, New(GGML_TYPE_F32, 4064, 5)))));
  Refused(kg::CheckMulMatQ(
      Bound(ggml_mul_mat(c(), New(GGML_TYPE_Q8_0, 4096, 256), New(GGML_TYPE_F16, 4096, 5)))));
  ggml_tensor* w = New(GGML_TYPE_Q5_K, 4096, 256);
  ggml_tensor* over = ggml_mul_mat(c(), w, x);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(w->data));
  Refused(kg::CheckMulMatQ(over));
  // A misaligned weight base.
  ggml_tensor* odd = ggml_new_tensor_2d(c(), GGML_TYPE_Q8_0, 4096, 256);
  TensorArena::Bind(odd, kBase + (next_++ * kSlot) + 2);
  Refused(kg::CheckMulMatQ(Bound(ggml_mul_mat(c(), odd, x))));
  // A routing hint belongs to another implementation.
  ggml_tensor* hinted = Bound(ggml_mul_mat(c(), New(GGML_TYPE_Q8_0, 4096, 256), x));
  ggml_mul_mat_set_hint(hinted, GGML_HINT_SRC0_IS_HADAMARD);
  Refused(kg::CheckMulMatQ(hinted));
}

TEST_F(GgmlExtValidateTest, BorrowedDenseD4RefusesStaleGuardedAliasedAndUnsupportedOperands) {
  auto* weights = New(GGML_TYPE_Q8_0, 8192, 4096);
  auto* source = New(GGML_TYPE_F32, 8192, 4096);
  auto* output = Bound(ggml_mul_mat(c(), weights, source));
  const kg::BorrowedMmqD4 original{.data = std::bit_cast<const void*>(kBase + (next_++ * kSlot)),
                                   .bytes = (4096ULL * (8192 / 128) * 144) + (256ULL * 144),
                                   .source = source->data,
                                   .generation = 7};
  Accepted(kg::CheckMulMatQBorrowedD4(output, original, 7));
  Refused(kg::CheckMulMatQBorrowedD4(nullptr, original, 7));
  Refused(kg::CheckMulMatQBorrowedD4(output, original, 8));
  auto bad = original;
  bad.generation = 0;
  Refused(kg::CheckMulMatQBorrowedD4(output, bad, 0));
  bad = original;
  bad.source = weights->data;
  Refused(kg::CheckMulMatQBorrowedD4(output, bad, 7));
  bad = original;
  bad.bytes = (4096ULL * (8192 / 128) * 144) + (128ULL * 144) - 1;
  Refused(kg::CheckMulMatQBorrowedD4(output, bad, 7));
  bad = original;
  bad.data = std::bit_cast<const void*>(UINT64_MAX - 15);
  Refused(kg::CheckMulMatQBorrowedD4(output, bad, 7));
  for (const auto* tensor : {weights, source, output}) {
    bad = original;
    bad.data = tensor->data;
    Refused(kg::CheckMulMatQBorrowedD4(output, bad, 7));
  }
  auto* unsupported = Bound(ggml_mul_mat(c(), New(GGML_TYPE_Q2_K, 8192, 4096), source));
  Refused(kg::CheckMulMatQBorrowedD4(unsupported, original, 7));
  const auto stride = source->nb[1];
  source->nb[1] += sizeof(float);
  Refused(kg::CheckMulMatQBorrowedD4(output, original, 7));
  source->nb[1] = stride;
  auto* multiple = New(GGML_TYPE_F32, 8192, 4096, 2);
  auto* broadcast = Bound(ggml_mul_mat(c(), weights, multiple));
  bad = original;
  bad.source = multiple->data;
  bad.bytes *= 2;
  Refused(kg::CheckMulMatQBorrowedD4(broadcast, bad, 7));
}

TEST_F(GgmlExtValidateTest, ExpertProductsFollowMulMatIdsShapes) {
  // DeepSeek V4: 256 experts of IQ2_XS gate/up [4096, 2048], 6 used.
  ggml_tensor* w = New(GGML_TYPE_IQ2_XS, 4096, 2048, 256);
  ggml_tensor* x = New(GGML_TYPE_F32, 4096, 1, 7);
  ggml_tensor* ids = New(GGML_TYPE_I32, 6, 7);
  Accepted(kg::CheckMulMatIdQ(Bound(ggml_mul_mat_id(c(), w, x, ids))));
  // Down: one row per selected expert.
  ggml_tensor* down = New(GGML_TYPE_IQ3_XXS, 2048, 4096, 256);
  Accepted(
      kg::CheckMulMatIdQ(Bound(ggml_mul_mat_id(c(), down, New(GGML_TYPE_F32, 2048, 6, 7), ids))));
  // More experts selected than there are; a plain product is not this
  // check's.
  Refused(kg::CheckMulMatIdQ(
      Bound(ggml_mul_mat_id(c(), New(GGML_TYPE_IQ2_XS, 4096, 2048, 4), x, ids))));
  Refused(kg::CheckMulMatIdQ(Bound(ggml_mul_mat(c(), New(GGML_TYPE_Q8_0, 4096, 8), x))));
}

TEST_F(GgmlExtValidateTest, RawQ2D2rRefusesWrongLayoutsAndWorklistBounds) {
  const auto make = [&](ggml_type type, std::int64_t k, std::int64_t m, std::int64_t experts,
                        std::int64_t used, std::int64_t tokens, bool broadcast = false) {
    return Bound(ggml_mul_mat_id(c(), New(type, k, m, experts),
                                 New(GGML_TYPE_F32, k, broadcast ? 1 : used, tokens),
                                 New(GGML_TYPE_I32, used, tokens)));
  };
  auto* real = make(GGML_TYPE_Q2_K, 2048, 4096, 256, 6, 4096);
  Accepted(kg::CheckMulMatIdQ2D2r(real));
  Accepted(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 18, 16, 6, 65)));
  Accepted(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 2, 32768, 1, 1)));
  Refused(kg::CheckMulMatIdQ2D2r(nullptr));
  Refused(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q8_0, 2048, 4096, 256, 6, 4096)));
  Refused(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 17, 16, 6, 65)));
  Refused(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 18, 16, 6, 65, true)));
  Refused(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 2, 32769, 1, 1)));
  Refused(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 2, 65536, 1, 1)));
  Refused(kg::CheckMulMatIdQ2D2r(make(GGML_TYPE_Q2_K, 512, 2, 65535, 1, 1)));
  auto* weights = real->src[0];
  const auto stride = weights->nb[2];
  weights->nb[2] -= 84;  // overlapping expert rows, despite valid raw byte multiples
  Refused(kg::CheckMulMatIdQ2D2r(real));
  weights->nb[2] = stride;
  real->nb[2] += sizeof(float);  // the raw scatter returns packed output only
  Refused(kg::CheckMulMatIdQ2D2r(real));
}

TEST_F(GgmlExtValidateTest, Ds4HcaRequiresExplicitCanonicalMetadataAndPreservesDefaultPlanning) {
  const auto build = [this](std::uint32_t first, std::int64_t rows) {
    auto* packed = New(GGML_TYPE_F32, 512, 64, rows);
    auto* q = ggml_permute(c(), packed, 0, 2, 1, 3);
    auto* kv = New(GGML_TYPE_F16, 512, 4608);
    auto* mask = Bound(kg::Dsv4SparseMask(c(), New(GGML_TYPE_F16, 4352, rows), nullptr,
                                          New(GGML_TYPE_I32, rows), 4352, 256));
    auto* node = Bound(ggml_flash_attn_ext(c(), q, kv, kv, mask, 0.04419417306780815F, 0, 0));
    ggml_flash_attn_ext_add_sinks(node, New(GGML_TYPE_F32, 64));
    EXPECT_TRUE(ggml_prec_set_acc(node, GGML_PREC_F32));
    ggml_flash_attn_ext_set_n_kv_max(node, 384);
    kg::SetFlashAttnSparseAny(node);
    kg::MarkDsv4HcaTokentile(node, first);
    return node;
  };
  auto* node = build(0, 4096);
  EXPECT_EQ(node->src[0]->nb[1], std::size_t{64} * 512 * sizeof(float));
  EXPECT_EQ(node->src[0]->nb[2], 512U * sizeof(float));
  EXPECT_EQ(node->src[0]->nb[3], ggml_nbytes(node));
  EXPECT_EQ(node->op_params[3], GGML_PREC_F32);
  EXPECT_EQ(kg::JitllmOpInt(node->src[3], 0), 4352);
  EXPECT_EQ(kg::JitllmOpInt(node->src[3], 1), 1);
  EXPECT_EQ(node->op_params[4], 384);
  Accepted(kg::CheckDsv4HcaTokentile(node));
  for (const auto first : {127U, 128U, 511U, 28672U})
    Accepted(kg::CheckDsv4HcaTokentile(build(first, 5)));
  Refused(kg::CheckDsv4HcaTokentile(nullptr));
  Refused(kg::CheckDsv4HcaTokentile(build(0, 4097)));
  Refused(kg::CheckDsv4HcaTokentile(build(32768, 4096)));  // unread compressed extent
  Refused(kg::CheckDsv4HcaTokentile(build(UINT32_MAX, 5)));
  auto saved = node->op_params[kg::kDsv4HcaTagParam];
  node->op_params[kg::kDsv4HcaTagParam] = 0;
  Refused(kg::CheckDsv4HcaTokentile(node));
  node->op_params[kg::kDsv4HcaTagParam] = saved;
  const auto refuse_param = [&](int index, std::int32_t value) {
    saved = node->op_params[index];
    node->op_params[index] = value;
    Refused(kg::CheckDsv4HcaTokentile(node));
    node->op_params[index] = saved;
  };
  refuse_param(0, 0);
  refuse_param(1, 0x3f800000);
  refuse_param(2, 0x3f800000);
  refuse_param(3, GGML_PREC_DEFAULT);
  refuse_param(4, 383);
  refuse_param(kg::kFlashAttnSparseParam, 0);
  refuse_param(kg::kFlashAttnWideSparseParam, 1);
  auto* original_v = node->src[2];
  node->src[2] = New(GGML_TYPE_F16, 512, 4608);
  Refused(kg::CheckDsv4HcaTokentile(node));
  node->src[2] = original_v;
  void* const original_q = node->src[0]->data;
  node->src[0]->data = node->src[1]->data;
  Refused(kg::CheckDsv4HcaTokentile(node));
  node->src[0]->data = original_q;
  const std::array<ggml_tensor*, 1> nodes = {node};
  kg::DeviceChoices choices;
  choices.ds4_hca_fits = [](const ggml_tensor* n) {
    return kg::CheckDsv4HcaTokentile(n).has_value();
  };
  const auto ordinary = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(ordinary.has_value());
  EXPECT_EQ(ordinary->steps.front().implementation, kg::kFlashAttnMmaName);
  choices.ds4_hca = true;
  const auto selected = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(selected->steps.front().implementation, kg::kDsv4HcaTokentileName);
  choices.ds4_hca_fits = nullptr;
  const auto fallback = kg::PlanGraph(nodes, false, choices);
  ASSERT_TRUE(fallback.has_value());
  EXPECT_EQ(fallback->steps.front().implementation, kg::kFlashAttnMmaName);
}

TEST_F(GgmlExtValidateTest, PairedExpertsRequireMatchingInputsAndDisjointOutputs) {
  auto* input = New(GGML_TYPE_F32, 512, 1, 40);
  auto* ids = New(GGML_TYPE_I32, 2, 40);
  auto* a = New(GGML_TYPE_IQ2_XXS, 512, 128, 16);
  auto* b = New(GGML_TYPE_IQ2_XXS, 512, 128, 16);
  auto* first = Bound(ggml_mul_mat_id(c(), a, input, ids));
  auto* second = Bound(ggml_mul_mat_id(c(), b, input, ids));
  Accepted(kg::CheckMulMatIdQPair(first, second));
  Accepted(kg::CheckMulMatIdQCompact(first));
  Refused(kg::CheckMulMatIdQCompact(nullptr));
  Refused(kg::CheckMulMatIdQCompact(
      Bound(ggml_mul_mat(c(), New(GGML_TYPE_Q8_0, 512, 128), New(GGML_TYPE_F32, 512, 40)))));
  Refused(kg::CheckMulMatIdQPair(nullptr, second));
  Refused(kg::CheckMulMatIdQPair(first, first));
  Refused(kg::CheckMulMatIdQPair(
      first, Bound(ggml_mul_mat_id(c(), New(GGML_TYPE_IQ2_XS, 512, 128, 16), input, ids))));
  Refused(kg::CheckMulMatIdQPair(
      first, Bound(ggml_mul_mat_id(c(), b, New(GGML_TYPE_F32, 512, 1, 40), ids))));
  Refused(kg::CheckMulMatIdQPair(first,
                                 Bound(ggml_mul_mat_id(c(), b, input, New(GGML_TYPE_I32, 2, 40)))));
  Refused(kg::CheckMulMatIdQPair(
      first, Bound(ggml_mul_mat_id(c(), New(GGML_TYPE_IQ2_XXS, 512, 256, 16), input, ids))));
  TensorArena::Bind(first, reinterpret_cast<std::uintptr_t>(b->data));
  Refused(kg::CheckMulMatIdQPair(first, second));
  for (const auto type : {GGML_TYPE_MXFP4, GGML_TYPE_NVFP4}) {
    auto* x = Bound(ggml_mul_mat_id(c(), New(type, 512, 128, 16), input, ids));
    auto* y = Bound(ggml_mul_mat_id(c(), New(type, 512, 128, 16), input, ids));
    Refused(kg::CheckMulMatIdQPair(x, y));
    Refused(kg::CheckMulMatIdQCompact(x));
  }
}

TEST_F(GgmlExtValidateTest, Iq2OccupancyTwoPairsKeepTheMeasuredGeometryAndOriginalValidation) {
  const auto pair = [&](ggml_type type, std::int64_t k, std::int64_t width, std::int64_t experts,
                        std::int64_t used, std::int64_t rows, bool broadcast = true) {
    auto* input = New(GGML_TYPE_F32, k, broadcast ? 1 : used, rows);
    auto* ids = New(GGML_TYPE_I32, used, rows);
    auto* first = Bound(ggml_mul_mat_id(c(), New(type, k, width, experts), input, ids));
    auto* second = Bound(ggml_mul_mat_id(c(), New(type, k, width, experts), input, ids));
    Accepted(kg::CheckMulMatIdQPair(first, second));
    return std::pair{first, second};
  };
  auto [first, second] = pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 256, 6, 4096);
  EXPECT_TRUE(kg::IsMulMatIdQPairIq2Occ2(first, second));
  EXPECT_FALSE(kg::IsMulMatIdQPairIq2Occ2(nullptr, second));
  EXPECT_FALSE(kg::IsMulMatIdQPairIq2Occ2(first, first));
  // Real router IDs are the six-row view of a 256-row argsort result.
  // Original map preparation already pays and respects this token stride.
  auto* routed_ids = first->src[2];
  routed_ids->nb[1] = 256 * sizeof(std::int32_t);
  routed_ids->nb[2] = routed_ids->nb[1] * 4096;
  routed_ids->nb[3] = routed_ids->nb[2];
  EXPECT_TRUE(kg::IsMulMatIdQPairIq2Occ2(first, second));
  // Any chunk from the compact list's 256 tokens to 4,096: a prompt's last,
  // partial chunk too.
  for (const std::int64_t tokens :
       {kg::kDsv4StagePairMinRows, std::int64_t{2047}, std::int64_t{2947}, kg::kDsv4StageMaxRows}) {
    const auto [a, b] = pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 256, 6, tokens);
    EXPECT_TRUE(kg::IsMulMatIdQPairIq2Occ2(a, b)) << tokens;
  }
  EXPECT_TRUE(kg::IsMulMatIdQPairGluPair(first, second));
  // UD-Q2_K_XL's IQ2_XS gate/up experts write the activation too, over
  // GGML's ordinary J128 launch (the occupancy-two one is IQ2_XXS's).
  {
    const auto [a, b] = pair(GGML_TYPE_IQ2_XS, 4096, 2048, 256, 6, 2947);
    EXPECT_FALSE(kg::IsMulMatIdQPairIq2Occ2(a, b));
    EXPECT_TRUE(kg::IsMulMatIdQPairGluPair(a, b));
  }
  for (const auto& [a, b] : {pair(GGML_TYPE_IQ2_S, 4096, 2048, 256, 6, 4096),
                             pair(GGML_TYPE_IQ3_XXS, 4096, 2048, 256, 6, 4096),
                             pair(GGML_TYPE_IQ2_XXS, 2048, 2048, 256, 6, 4096),
                             pair(GGML_TYPE_IQ2_XXS, 4096, 4096, 256, 6, 4096),
                             pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 128, 6, 4096),
                             pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 256, 5, 4096),
                             pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 256, 6, 255),
                             pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 256, 6, 4097),
                             pair(GGML_TYPE_IQ2_XXS, 4096, 2048, 256, 6, 4096, false)}) {
    EXPECT_FALSE(kg::IsMulMatIdQPairIq2Occ2(a, b));
    EXPECT_FALSE(kg::IsMulMatIdQPairGluPair(a, b));
  }
  // Padded outputs still satisfy the ordinary pair contract, but they do
  // not enter the fixed packed-output specialization.
  second->nb[2] += second->nb[1];
  Accepted(kg::CheckMulMatIdQPair(first, second));
  EXPECT_FALSE(kg::IsMulMatIdQPairIq2Occ2(first, second));
  second->nb[2] -= second->nb[1];
  TensorArena::Bind(second, reinterpret_cast<std::uintptr_t>(first->data));
  EXPECT_FALSE(kg::IsMulMatIdQPairIq2Occ2(first, second));
}

TEST_F(GgmlExtValidateTest, TheHadamardProductNeedsItsHintAndARowTheTransformTakes) {
  ggml_tensor* rotation = New(GGML_TYPE_F32, 128, 128);
  ggml_tensor* node =
      Bound(ggml_mul_mat(c(), rotation, New(GGML_TYPE_F32, 128, std::int64_t{64} * 5)));
  Refused(kg::CheckMulMatHadamard(node));  // no hint
  ggml_mul_mat_set_hint(node, GGML_HINT_SRC0_IS_HADAMARD);
  Accepted(kg::CheckMulMatHadamard(node));
  ggml_tensor* odd =
      Bound(ggml_mul_mat(c(), New(GGML_TYPE_F32, 96, 96), New(GGML_TYPE_F32, 96, 4)));
  ggml_mul_mat_set_hint(odd, GGML_HINT_SRC0_IS_HADAMARD);
  Refused(kg::CheckMulMatHadamard(odd));
}

TEST_F(GgmlExtValidateTest, ElementwiseOperationsTakePackedF32OfOneShape) {
  ggml_tensor* x = New(GGML_TYPE_F32, 4096, 3);
  for (ggml_tensor* (*build)(ggml_context*, ggml_tensor*) :
       {ggml_abs, ggml_sgn, ggml_neg, ggml_silu, ggml_tanh, ggml_relu, ggml_sigmoid, ggml_exp,
        ggml_softplus, ggml_sqrt}) {
    Accepted(kg::CheckUnary(Bound(build(c(), x))));
  }
  Accepted(kg::CheckUnary(ggml_sigmoid_inplace(c(), x)));  // in place
  Refused(kg::CheckUnary(Bound(ggml_gelu(c(), x))));       // not launched here
  Refused(kg::CheckUnary(Bound(ggml_silu(c(), New(GGML_TYPE_F16, 4096, 3)))));
  // A strided view is not packed; nor is the unbound output.
  ggml_tensor* view = ggml_view_2d(c(), x, 2048, 3, x->nb[1], 0);
  Refused(kg::CheckUnary(Bound(ggml_silu(c(), view))));
  Refused(kg::CheckUnary(ggml_silu(c(), x)));

  Accepted(kg::CheckScale(Bound(ggml_scale_bias(c(), x, 0.5f, 1.0f))));
  Accepted(kg::CheckClamp(Bound(ggml_clamp(c(), x, -10.0f, 10.0f))));
  Refused(kg::CheckClamp(Bound(ggml_clamp(c(), x, 1.0f, -1.0f))));
  Accepted(kg::CheckFill(Bound(ggml_fill(c(), ggml_new_tensor_2d(c(), GGML_TYPE_F16, 100, 3),
                                         -std::numeric_limits<float>::infinity()))));
  Refused(kg::CheckFill(Bound(ggml_fill(c(), ggml_new_tensor_1d(c(), GGML_TYPE_F16, 1), 0.0f))
                            ->src[0]));  // not a fill node
  ggml_tensor* row = New(GGML_TYPE_F32, 64, 1);
  Accepted(kg::CheckRepeat(Bound(ggml_repeat_4d(c(), row, 64, 64, 3, 1))));
  ggml_tensor* a = New(GGML_TYPE_F32, 4, 4, 5);
  ggml_tensor* b = New(GGML_TYPE_F32, 1, 4, 5);
  Accepted(kg::CheckBinary(Bound(ggml_div(c(), a, b)), GGML_OP_DIV));
  Accepted(kg::CheckBinary(Bound(ggml_sub(c(), a, b)), GGML_OP_SUB));
  Accepted(kg::CheckSumRows(Bound(ggml_sum_rows(c(), x))));
  Refused(kg::CheckSumRows(Bound(ggml_sum_rows(c(), view))));
}

TEST_F(GgmlExtValidateTest, ConcatTakesOneUnblockedTypeIntoAPackedOutput) {
  ggml_tensor* a = New(GGML_TYPE_F16, 512, 100);
  ggml_tensor* b = New(GGML_TYPE_F16, 512, 28);
  Accepted(kg::CheckConcat(Bound(ggml_concat(c(), a, b, 1))));
  Accepted(kg::CheckConcat(
      Bound(ggml_concat(c(), New(GGML_TYPE_F32, 3, 4), New(GGML_TYPE_F32, 5, 4), 0))));
  // Past 65,535 channels (DeepSeek V4's window cells and compressed rows
  // from about 52K positions, RE-038): operands contiguous in their first
  // three dimensions take the contiguous kernel's one-dimensional grid; a
  // strided operand takes the per-row kernel, whose grid cannot.
  ggml_tensor* window = New(GGML_TYPE_F16, 512, 1, 53248);
  ggml_tensor* compressed = New(GGML_TYPE_F16, 512, 1, 13312);
  Accepted(kg::CheckConcat(Bound(ggml_concat(c(), window, compressed, 2))));
  ggml_tensor* wide = New(GGML_TYPE_F16, 1024, 1, 53248);
  ggml_tensor* strided = ggml_view_3d(c(), wide, 512, 1, 53248, wide->nb[1], wide->nb[2], 0);
  Refused(kg::CheckConcat(Bound(ggml_concat(c(), strided, compressed, 2))));
  Accepted(kg::CheckConcat(
      Bound(ggml_concat(c(), ggml_view_3d(c(), wide, 512, 1, 1024, wide->nb[1], wide->nb[2], 0),
                        New(GGML_TYPE_F16, 512, 1, 256), 2))));
  ggml_tensor* q = New(GGML_TYPE_Q8_0, 512, 4);
  Refused(kg::CheckConcat(Bound(ggml_concat(c(), q, New(GGML_TYPE_Q8_0, 512, 4), 1))));
  // Overlapping the output with an operand.
  ggml_tensor* over = ggml_concat(c(), a, b, 1);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(a->data));
  Refused(kg::CheckConcat(over));
}

TEST_F(GgmlExtValidateTest, RoutingTakesTheKernelsUpstreamWouldRun) {
  ggml_tensor* scores = New(GGML_TYPE_F32, 256, 7);
  Accepted(kg::CheckArgsort(Bound(ggml_argsort(c(), scores, GGML_SORT_ORDER_DESC))));
  EXPECT_EQ(kg::ArgsortSharedBytes(Bound(ggml_argsort(c(), scores, GGML_SORT_ORDER_DESC))),
            256U * 4U);
  EXPECT_EQ(kg::ArgsortSharedBytes(
                Bound(ggml_argsort(c(), New(GGML_TYPE_F32, 300, 1), GGML_SORT_ORDER_ASC))),
            512U * 4U);
  // CUB's sort, which upstream takes for longer rows, is not built.
  Refused(kg::CheckArgsort(
      Bound(ggml_argsort(c(), New(GGML_TYPE_F32, 2048, 2), GGML_SORT_ORDER_DESC))));
  Accepted(kg::CheckTopK(Bound(ggml_top_k(c(), New(GGML_TYPE_F32, 20000, 3), 512))));
  Refused(kg::CheckTopK(Bound(ggml_top_k(c(), New(GGML_TYPE_F16, 2000, 3), 512))));
  ggml_tensor* gate = New(GGML_TYPE_F32, 2048, 3);
  ggml_tensor* up = New(GGML_TYPE_F32, 2048, 3);
  Accepted(kg::CheckSwiGluClamp(Bound(ggml_swiglu_clamp(c(), gate, up, 10.0f))));
  Refused(kg::CheckSwiGluClamp(Bound(ggml_swiglu_split(c(), gate, up))));
}

TEST_F(GgmlExtValidateTest, RopeTakesTheModelsModesOffsetsAndPositions) {
  // DeepSeek V4: normal rotation of the last 64 of 512 channels.
  ggml_tensor* q = New(GGML_TYPE_F32, 512, 64, 5);
  ggml_tensor* positions = New(GGML_TYPE_I32, 5);
  ggml_tensor* rope = ggml_rope_ext(c(), q, positions, nullptr, 64, 0, 65536, 10000.0f,
                                    1.0f / 16.0f, 1.0f, 1.0f, 32.0f, 1.0f);
  Bound(ggml_rope_set_offset(rope, 448));
  Accepted(kg::CheckRopeExt(rope));
  ggml_tensor* back = ggml_rope_ext_back(c(), q, positions, nullptr, 64, 0, 65536, 10000.0f,
                                         1.0f / 16.0f, 1.0f, 1.0f, 32.0f, 1.0f);
  Bound(ggml_rope_set_offset(back, 448));
  Accepted(kg::CheckRopeExt(back));
  // Rotating past the head, or with an odd offset.
  ggml_tensor* past = ggml_rope_ext(c(), q, positions, nullptr, 64, 0, 65536, 10000.0f, 1.0f, 0.0f,
                                    1.0f, 32.0f, 1.0f);
  Bound(ggml_rope_set_offset(past, 480));
  Refused(kg::CheckRopeExt(past));
  ggml_tensor* odd = ggml_rope_ext(c(), q, positions, nullptr, 64, 0, 65536, 10000.0f, 1.0f, 0.0f,
                                   1.0f, 32.0f, 1.0f);
  Bound(ggml_rope_set_offset(odd, 3));
  Refused(kg::CheckRopeExt(odd));
  // Qwen3.8: interleaved multi-section rotation, four positions per token.
  std::array<int, GGML_MROPE_SECTIONS> sections = {11, 11, 10, 0};
  ggml_tensor* x = New(GGML_TYPE_F32, 256, 24, 5);
  Accepted(kg::CheckRopeExt(
      Bound(ggml_rope_multi(c(), x, New(GGML_TYPE_I32, 20), nullptr, 64, sections.data(),
                            GGML_ROPE_TYPE_IMROPE, 262144, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f))));
  // No position section: the launcher asserts one.
  std::array<int, GGML_MROPE_SECTIONS> empty = {0, 0, 0, 32};
  Refused(kg::CheckRopeExt(
      Bound(ggml_rope_multi(c(), x, New(GGML_TYPE_I32, 20), nullptr, 64, empty.data(),
                            GGML_ROPE_TYPE_MROPE, 262144, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f))));
  // Frequency factors are not implemented.
  Refused(kg::CheckRopeExt(Bound(ggml_rope_ext(c(), q, positions, New(GGML_TYPE_F32, 32), 64, 0,
                                               65536, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f))));
}

TEST_F(GgmlExtValidateTest, GathersAndScattersTakeTheModelsTypes) {
  Accepted(kg::CheckGetRowsExt(
      Bound(ggml_get_rows(c(), New(GGML_TYPE_Q5_K, 4096, 129280), New(GGML_TYPE_I32, 5)))));
  Accepted(kg::CheckGetRowsExt(
      Bound(ggml_get_rows(c(), New(GGML_TYPE_I32, 6, 129280), New(GGML_TYPE_I32, 5)))));
  // Not whole super-blocks; a type not compiled for the products.
  Refused(kg::CheckGetRowsExt(
      Bound(ggml_get_rows(c(), New(GGML_TYPE_Q8_0, 96, 10), New(GGML_TYPE_I32, 5)))));
  Refused(kg::CheckGetRowsExt(
      Bound(ggml_get_rows(c(), New(GGML_TYPE_Q4_1, 4096, 10), New(GGML_TYPE_I32, 5)))));
  // A product type GGML's get_rows has no case for (getrows.cu aborts).
  Refused(kg::CheckGetRowsExt(
      Bound(ggml_get_rows(c(), New(GGML_TYPE_NVFP4, 4096, 10), New(GGML_TYPE_I32, 5)))));
  ggml_tensor* mask = New(GGML_TYPE_F16, 1, 3072);
  Accepted(kg::CheckSetRowsExt(
      ggml_set_rows(c(), mask, New(GGML_TYPE_F16, 1, 5), New(GGML_TYPE_I32, 5))));
  Accepted(kg::CheckSetRowsExt(ggml_set_rows(c(), New(GGML_TYPE_F32, 64, 10),
                                             New(GGML_TYPE_F32, 64, 3), New(GGML_TYPE_I64, 3))));
  // F16 rows into F32 have no kernel.
  Refused(kg::CheckSetRowsExt(ggml_set_rows(c(), New(GGML_TYPE_F32, 64, 10),
                                            New(GGML_TYPE_F16, 64, 3), New(GGML_TYPE_I32, 3))));
}

TEST_F(GgmlExtValidateTest, LinearAttentionTakesQwen38sShapes) {
  // The causal convolution: 10,240 channels, kernel 4; up to 32 tokens, or
  // whole 32-token blocks past that (RE-032: the long-token kernel's last
  // block of 40 tokens would load past the window).
  for (const std::int64_t tokens : {1, 7, 32, 64, 96}) {
    Accepted(kg::CheckSsmConv(Bound(ggml_ssm_conv(c(), New(GGML_TYPE_F32, 3 + tokens, 10240, 1),
                                                  New(GGML_TYPE_F32, 4, 10240)))));
  }
  for (const std::int64_t tokens : {33, 40, 63, 65}) {
    Refused(kg::CheckSsmConv(Bound(ggml_ssm_conv(c(), New(GGML_TYPE_F32, 3 + tokens, 10240, 1),
                                                 New(GGML_TYPE_F32, 4, 10240)))));
  }
  // A kernel size without an instance, and channels not in blocks of 128.
  Refused(kg::CheckSsmConv(Bound(
      ggml_ssm_conv(c(), New(GGML_TYPE_F32, 5 + 40, 10240, 1), New(GGML_TYPE_F32, 6, 10240)))));
  Refused(kg::CheckSsmConv(
      Bound(ggml_ssm_conv(c(), New(GGML_TYPE_F32, 3 + 40, 1000, 1), New(GGML_TYPE_F32, 4, 1000)))));
  // The gated delta rule: 16 query/key heads and 48 value heads of 128.
  const auto gdn = [this](std::int64_t s, std::int64_t hk) {
    return Bound(
        ggml_gated_delta_net(c(), New(GGML_TYPE_F32, s, hk, 6, 1), New(GGML_TYPE_F32, s, hk, 6, 1),
                             New(GGML_TYPE_F32, s, 48, 6, 1), New(GGML_TYPE_F32, 1, 48, 6, 1),
                             New(GGML_TYPE_F32, 1, 48, 6, 1), New(GGML_TYPE_F32, s, s, 48, 1), 1));
  };
  Accepted(kg::CheckGatedDeltaNet(gdn(128, 16)));
  Refused(kg::CheckGatedDeltaNet(gdn(96, 16)));  // no kernel for 96
  Refused(kg::CheckGatedDeltaNet(gdn(128, 5)));  // 48 heads are not a multiple of 5
}

TEST_F(GgmlExtValidateTest, DeepSeekV4sIndexerAndHyperConnectionsFollowTheirShapes) {
  const auto indexer = [this](std::int64_t heads, ggml_type key) {
    return Bound(ggml_lightning_indexer(
        c(), New(GGML_TYPE_F32, 128, heads, 5, 1), New(key, 128, 1, 1000, 1),
        New(GGML_TYPE_F32, heads, 5, 1, 1), New(GGML_TYPE_F16, 1000, 5, 1, 1)));
  };
  Accepted(kg::CheckLightningIndexer(indexer(64, GGML_TYPE_F16)));
  Accepted(kg::CheckLightningIndexer(indexer(32, GGML_TYPE_F16)));
  Refused(kg::CheckLightningIndexer(indexer(16, GGML_TYPE_F16)));   // no kernel
  Refused(kg::CheckLightningIndexer(indexer(64, GGML_TYPE_Q8_0)));  // tensor-core F16 keys only

  Accepted(kg::CheckHcComb(Bound(ggml_dsv4_hc_comb(
      c(), New(GGML_TYPE_F32, 24, 5), New(GGML_TYPE_F32, 3), New(GGML_TYPE_F32, 24), 1e-6f, 20))));
  ggml_tensor* x = New(GGML_TYPE_F32, 4096, 4, 5);
  Accepted(kg::CheckHcPre(Bound(ggml_dsv4_hc_pre(c(), x, New(GGML_TYPE_F32, 4, 5)))));
  Refused(kg::CheckHcPre(Bound(ggml_dsv4_hc_pre(c(), New(GGML_TYPE_F32, 4096, 3, 5),
                                                New(GGML_TYPE_F32, 3, 5)))));  // 3 streams
  Accepted(kg::CheckHcPost(
      Bound(ggml_dsv4_hc_post(c(), New(GGML_TYPE_F32, 4096, 5), x, New(GGML_TYPE_F32, 4, 5),
                              New(GGML_TYPE_F32, 4, 4, 5)))));
}

// RE-030: the tensor-core kernels read a whole group of 8 sinks from each
// group's first query head, so sinks need a multiple of 8 query heads per KV
// head, whatever the head size, row count or sparse gather.
TEST_F(GgmlExtValidateTest, SinksNeedWholeGroupsOfEightQueryHeads) {
  const auto attention = [this](std::int64_t head, std::int64_t heads, std::int64_t kv_heads,
                                std::int64_t rows, std::int64_t cells, bool sinks) {
    ggml_tensor* k = New(GGML_TYPE_F16, head, cells, kv_heads);
    ggml_tensor* node = ggml_flash_attn_ext(c(), New(GGML_TYPE_F32, head, rows, heads), k, k,
                                            New(GGML_TYPE_F16, cells, rows), 0.1f, 0.0f, 0.0f);
    if (sinks) {
      ggml_flash_attn_ext_add_sinks(node, New(GGML_TYPE_F32, heads));
    }
    return Bound(node);
  };
  Accepted(kg::CheckFlashAttnMma(attention(512, 64, 1, 1, 256, true)));
  Accepted(kg::CheckFlashAttnMma(attention(256, 32, 2, 3, 512, true)));
  Accepted(kg::CheckFlashAttnMma(attention(256, 24, 2, 1, 256, false)));
  Refused(kg::CheckFlashAttnMma(attention(256, 24, 2, 1, 256, true)));  // 12 per KV head
  Refused(kg::CheckFlashAttnMma(attention(512, 20, 1, 9, 256, true)));  // 20
  Refused(kg::CheckFlashAttnMma(attention(512, 60, 5, 1, 256, true)));  // 12, last group short
  ggml_tensor* sparse = attention(512, 36, 3, 1, 4096, true);           // 12, sparse-eligible
  ggml_flash_attn_ext_set_n_kv_max(sparse, 256);
  Refused(kg::CheckFlashAttnMma(sparse));
}

}  // namespace
