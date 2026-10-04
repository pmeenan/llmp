// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML-derived implementations' operand checks (kernels/ggml/
// validate.h), in every profile and without a GPU: the dense Qwen2 shapes
// pass, and malformed shapes, boundary sizes, misalignment, stale views
// and aliasing are refused; the cuBLAS path's plans follow upstream's
// launcher and draw the scratch GGML's pool recorded in P0; the launchers'
// variant choices that follow from the operands (get_rows' vector kernel,
// cont's copy) are predicted. GgmlFusionTest checks upstream's fusion
// gates (kernels/ggml/fusion.h) on GGML graphs of Qwen2's layer. Nothing
// here touches the addresses bound.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"

namespace {

using jitllm::kernels::ggml::CheckBinary;
using jitllm::kernels::ggml::CheckClearOf;
using jitllm::kernels::ggml::CheckCont;
using jitllm::kernels::ggml::CheckGetRows;
using jitllm::kernels::ggml::CheckMulMat;
using jitllm::kernels::ggml::CheckMulMatCublas;
using jitllm::kernels::ggml::CheckMulMatF;
using jitllm::kernels::ggml::CheckMulMatVecBias;
using jitllm::kernels::ggml::CheckMulMatVecGlu;
using jitllm::kernels::ggml::CheckRmsNorm;
using jitllm::kernels::ggml::CheckRmsNormMul;
using jitllm::kernels::ggml::CheckRmsNormThenMul;
using jitllm::kernels::ggml::CheckRope;
using jitllm::kernels::ggml::CheckRopeSetRows;
using jitllm::kernels::ggml::CheckSetRows;
using jitllm::kernels::ggml::CheckSoftMax;
using jitllm::kernels::ggml::CheckSwiGlu;
using jitllm::kernels::ggml::ContCopy;
using jitllm::kernels::ggml::CublasGemm;
using jitllm::kernels::ggml::CublasOperand;
using jitllm::kernels::ggml::FusionMemoryClear;
using jitllm::kernels::ggml::GetRowsVectorized;
using jitllm::kernels::ggml::GraphOrder;
using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::MulMatAddFusionAt;
using jitllm::kernels::ggml::MulMatGluFusionAt;
using jitllm::kernels::ggml::RopeSetRowsFusionAt;
using jitllm::kernels::ggml::SoftMaxSharedBytes;
using jitllm::kernels::ggml::TensorArena;

constexpr std::int64_t kWidth = 896;
constexpr std::uint64_t kBase = 1ULL << 40;  // never dereferenced
constexpr std::uint64_t kSlot = 1ULL << 36;  // far enough apart for any operand here

class GgmlValidateTest : public ::testing::Test {
 protected:
  ggml_context* context() { return arena_.context(); }
  // A tensor bound to its own slot, clear of every other.
  ggml_tensor* Bound(ggml_tensor* tensor) {
    TensorArena::Bind(tensor, kBase + (next_++ * kSlot));
    return tensor;
  }
  ggml_tensor* F32(std::int64_t ne0, std::int64_t ne1 = 1) {
    return Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F32, ne0, ne1));
  }
  template <typename T>
  static void Rejected(const std::expected<T, jitllm::kernels::ggml::KernelFailure>& checked) {
    ASSERT_FALSE(checked.has_value());
    EXPECT_EQ(checked.error().error, KernelError::kRejected);
  }

  TensorArena arena_ = TensorArena::Create(64).value();
  std::uint64_t next_ = 0;
};

TEST_F(GgmlValidateTest, TheDenseQwen2ShapesPass) {
  ggml_tensor* x = F32(kWidth, 5);
  ggml_tensor* w = F32(kWidth);
  ggml_tensor* norm = Bound(ggml_rms_norm(context(), x, 1e-6f));
  EXPECT_TRUE(CheckRmsNorm(norm).has_value());
  ggml_tensor* scaled = Bound(ggml_mul(context(), norm, w));
  EXPECT_TRUE(CheckBinary(scaled, GGML_OP_MUL).has_value());
  ggml_tensor* fused_norm = ggml_rms_norm(context(), x, 1e-6f);  // never written
  EXPECT_TRUE(CheckRmsNormMul(fused_norm, Bound(ggml_mul(context(), fused_norm, w))).has_value());
  EXPECT_TRUE(CheckBinary(Bound(ggml_add(context(), x, scaled)), GGML_OP_ADD).has_value());
  EXPECT_TRUE(CheckBinary(Bound(ggml_add(context(), x, w)), GGML_OP_ADD).has_value());
  ggml_tensor* in_place = ggml_add_inplace(context(), x, scaled);  // exactly x
  EXPECT_TRUE(CheckBinary(in_place, GGML_OP_ADD).has_value());

  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 4864));
  ggml_tensor* product = Bound(ggml_mul_mat(context(), m, scaled));
  EXPECT_TRUE(CheckMulMat(product).has_value());
  EXPECT_TRUE(CheckMulMatF(product).has_value());
  ggml_tensor* column = ggml_view_2d(context(), scaled, kWidth, 1, scaled->nb[1], 0);
  EXPECT_TRUE(CheckMulMat(Bound(ggml_mul_mat(context(), m, column))).has_value());
  ggml_tensor* wide = F32(kWidth, 17);
  Rejected(CheckMulMatF(Bound(ggml_mul_mat(context(), m, wide))));  // MMF takes 16
  EXPECT_TRUE(CheckMulMat(Bound(ggml_mul_mat(context(), m, wide))).has_value());
}

// GGML's row kernel steps its column index by up to 1,024 past the row's
// last element in 32-bit arithmetic.
TEST_F(GgmlValidateTest, RmsNormBoundsIncludeTheKernelsLoopStep) {
  constexpr std::int64_t kLargest = std::numeric_limits<std::int32_t>::max() - 1024;
  ggml_tensor* fits = Bound(ggml_new_tensor_1d(context(), GGML_TYPE_F32, kLargest));
  EXPECT_TRUE(CheckRmsNorm(Bound(ggml_rms_norm(context(), fits, 0.0f))).has_value());
  ggml_tensor* over = Bound(ggml_new_tensor_1d(context(), GGML_TYPE_F32, kLargest + 1));
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), over, 0.0f))));
  ggml_tensor* weight = Bound(ggml_new_tensor_1d(context(), GGML_TYPE_F32, kLargest + 1));
  ggml_tensor* fused_norm = ggml_rms_norm(context(), over, 0.0f);
  Rejected(CheckRmsNormMul(fused_norm, Bound(ggml_mul(context(), fused_norm, weight))));
}

TEST_F(GgmlValidateTest, EmptyMisalignedUnpackedAndWrappedOperandsAreRefused) {
  ggml_tensor* empty = F32(0);
  Rejected(CheckBinary(Bound(ggml_add(context(), empty, empty)), GGML_OP_ADD));

  ggml_tensor* odd = ggml_new_tensor_1d(context(), GGML_TYPE_F32, kWidth);
  TensorArena::Bind(odd, kBase + (next_++ * kSlot) + 2);
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), odd, 0.0f))));

  // A dimension of one with an unpacked stride.
  ggml_tensor* big = F32(4096);
  ggml_tensor* loose = ggml_view_4d(context(), big, 4, 1, 2, 1, 8, 16, 32, 0);
  Rejected(CheckBinary(Bound(ggml_add(context(), loose, F32(4))), GGML_OP_ADD));

  // Strides that wrap, which ggml_nbytes sums to a few hundred bytes.
  ggml_tensor* small = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 64, 16));
  ggml_tensor* wrapped =
      ggml_view_4d(context(), big, 64, 1, 2, 2, 256, (~std::size_t{0} - (std::size_t{1} << 33)) + 1,
                   (std::size_t{1} << 33) + 256, 0);
  Rejected(CheckMulMat(Bound(ggml_mul_mat(context(), small, wrapped))));

  // An odd channel stride under paired loads.
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 1024));
  ggml_tensor* channels = ggml_view_3d(context(), big, kWidth, 1, 2, kWidth * sizeof(float),
                                       (kWidth + 1) * sizeof(float), 0);
  Rejected(CheckMulMat(Bound(ggml_mul_mat(context(), m, channels))));
}

// DeepSeek V4's hyper-connection weights are views into a wider product:
// [4, tokens] at a 24-element row stride. The broadcast launcher indexes
// each operand by its own strides, and merges dimensions only for operands
// GGML deems contiguous, which a one-token view is: it is packed through
// its last dimension of more than one element, so merging addresses it
// correctly. The output stays packed.
TEST_F(GgmlValidateTest, StridedRowsBroadcastWhereMergingCannotMisaddressThem) {
  ggml_tensor* mixes = F32(24, 5);
  ggml_tensor* scale = F32(1);
  ggml_tensor* rows = ggml_view_2d(context(), mixes, 4, 5, mixes->nb[1], 0);
  EXPECT_TRUE(CheckBinary(Bound(ggml_mul(context(), rows, scale)), GGML_OP_MUL).has_value());
  ggml_tensor* one = F32(24, 1);
  ggml_tensor* row = ggml_view_2d(context(), one, 4, 1, one->nb[1], sizeof(float) * 4);
  EXPECT_TRUE(CheckBinary(Bound(ggml_mul(context(), row, scale)), GGML_OP_MUL).has_value());
  // F16 masks sum as F16.
  ggml_tensor* a = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 256, 3));
  ggml_tensor* b = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 256, 3));
  EXPECT_TRUE(CheckBinary(Bound(ggml_add(context(), a, b)), GGML_OP_ADD).has_value());
  // Mixed F16 and F32 is not taken.
  Rejected(CheckBinary(Bound(ggml_add(context(), a, F32(256, 3))), GGML_OP_ADD));
  // GGML deems [4, 1, 3] contiguous whatever its middle stride, and the
  // launcher, merging the first two dimensions of same-shaped operands,
  // would step the third by nb[1]·1 instead of nb[2]: refused.
  ggml_tensor* planes = F32(64, 12);
  ggml_tensor* skewed =
      ggml_view_3d(context(), planes, 4, 1, 3, 16 * sizeof(float), 4 * sizeof(float), 0);
  ASSERT_TRUE(ggml_is_contiguous(skewed));
  ggml_tensor* same = Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F32, 4, 1, 3));
  for (ggml_tensor* sum :
       {Bound(ggml_add(context(), skewed, same)), Bound(ggml_add(context(), same, skewed))}) {
    const auto refused = CheckBinary(sum, GGML_OP_ADD);
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().detail.find("strided rows"), std::string::npos)
        << refused.error().detail;
  }
  // A strided output is never taken.
  ggml_tensor* wide = F32(24, 5);
  ggml_tensor* into = ggml_view_2d(context(), wide, 4, 5, wide->nb[1], 0);
  Rejected(CheckBinary(ggml_add_inplace(context(), into, F32(4, 5)), GGML_OP_ADD));
}

TEST_F(GgmlValidateTest, ShapesMustFollowFromTheOperands) {
  ggml_tensor* x = F32(kWidth, 2);
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 1024));
  ggml_tensor* product = Bound(ggml_mul_mat(context(), m, x));
  product->ne[0] = 512;  // edited after GGML built it
  Rejected(CheckMulMat(product));
  ggml_tensor* sum = Bound(ggml_add(context(), x, x));
  sum->src[1] = F32(3);  // no longer broadcasts
  Rejected(CheckBinary(sum, GGML_OP_ADD));
  Rejected(CheckBinary(sum, GGML_OP_MUL));  // not the node's operation
}

TEST_F(GgmlValidateTest, OutputsMayAliasAnInputOnlyExactlyInPlace) {
  ggml_tensor* x = F32(kWidth, 2);
  // One row into its input.
  ggml_tensor* shifted = ggml_rms_norm(context(), x, 0.0f);
  TensorArena::Bind(shifted, reinterpret_cast<std::uintptr_t>(x->data) + x->nb[1]);
  Rejected(CheckRmsNorm(shifted));
  ggml_tensor* in_place = ggml_rms_norm_inplace(context(), x, 0.0f);
  EXPECT_TRUE(CheckRmsNorm(in_place).has_value());
  // Into a transposed view of an input.
  ggml_tensor* square = F32(4, 4);
  Rejected(CheckBinary(ggml_add_inplace(context(), ggml_transpose(context(), square), F32(4, 4)),
                       GGML_OP_ADD));
  // A matrix product never in place.
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F32, kWidth, kWidth));
  ggml_tensor* product = ggml_mul_mat(context(), m, x);
  TensorArena::Bind(product, reinterpret_cast<std::uintptr_t>(x->data));
  Rejected(CheckMulMat(product));
}

TEST_F(GgmlValidateTest, TheFusedNormNeverReadsWhatItDoesNotWrite) {
  ggml_tensor* x = F32(kWidth, 2);
  ggml_tensor* norm = Bound(ggml_rms_norm(context(), x, 0.0f));
  Rejected(CheckRmsNormMul(norm, Bound(ggml_mul(context(), norm, norm))));
  // An in-place norm is a view of x: scaling it by x reads the norm's bytes.
  ggml_tensor* in_place = ggml_rms_norm_inplace(context(), x, 0.0f);
  Rejected(CheckRmsNormMul(in_place, Bound(ggml_mul(context(), in_place, x))));
}

// The unfused implementation writes the norm, so the norm must be memory of
// its own: it may not alter the input or the weight, which the fused
// implementation leaves alone.
TEST_F(GgmlValidateTest, TheUnfusedNormWritesOnlyItsOwnIntermediate) {
  ggml_tensor* x = F32(kWidth, 5);
  ggml_tensor* w = F32(kWidth);
  ggml_tensor* norm = Bound(ggml_rms_norm(context(), x, 1e-6f));
  ggml_tensor* scaled = Bound(ggml_mul(context(), norm, w));
  EXPECT_TRUE(CheckRmsNormThenMul(norm, scaled).has_value());
  // The same nodes pass the fused implementation's check: one plan's
  // operands serve either implementation.
  EXPECT_TRUE(CheckRmsNormMul(norm, scaled).has_value());
  // The mul in place over the norm is still the norm's own memory.
  EXPECT_TRUE(CheckRmsNormThenMul(norm, ggml_mul_inplace(context(), norm, w)).has_value());

  // Not this norm's mul, or no norm at all.
  ggml_tensor* other = Bound(ggml_rms_norm(context(), x, 1e-6f));
  Rejected(CheckRmsNormThenMul(other, scaled));
  Rejected(CheckRmsNormThenMul(norm, Bound(ggml_add(context(), norm, w))));
  Rejected(CheckRmsNormThenMul(nullptr, scaled));
  Rejected(CheckRmsNormThenMul(norm, nullptr));
  // Unbound: the unfused norm needs memory.
  ggml_tensor* unbound = ggml_rms_norm(context(), x, 1e-6f);
  Rejected(CheckRmsNormThenMul(unbound, Bound(ggml_mul(context(), unbound, w))));
  // In place over its input: the fused implementation leaves x alone.
  ggml_tensor* in_place = ggml_rms_norm_inplace(context(), x, 1e-6f);
  Rejected(CheckRmsNormThenMul(in_place, Bound(ggml_mul(context(), in_place, w))));
  // Over the weight, which the mul would then read.
  ggml_tensor* wide = F32(kWidth, 5);
  ggml_tensor* weight_view = ggml_view_1d(context(), wide, kWidth, 0);
  ggml_tensor* over_weight = ggml_rms_norm(context(), x, 1e-6f);
  TensorArena::Bind(over_weight, reinterpret_cast<std::uintptr_t>(wide->data));
  Rejected(CheckRmsNormThenMul(over_weight, Bound(ggml_mul(context(), over_weight, weight_view))));
}

TEST_F(GgmlValidateTest, AViewThatNoLongerFollowsItsSourceIsRefused) {
  ggml_tensor* source = F32(kWidth);
  ggml_tensor* view = ggml_view_1d(context(), source, kWidth, 0);
  EXPECT_TRUE(CheckRmsNorm(Bound(ggml_rms_norm(context(), view, 0.0f))).has_value());
  TensorArena::Bind(source, kBase + (next_++ * kSlot));
  Rejected(CheckRmsNorm(Bound(ggml_rms_norm(context(), view, 0.0f))));
}

// GGML's pool peaks for the output head in the FP16 bridge, recorded in
// P0 (docs/experiments/backend-proof-p0/README.md): the F16 copy of the
// input and the F16 output temporary, at 17, 32 and 512 rows.
TEST_F(GgmlValidateTest, TheOutputHeadPlanDrawsTheRecordedPoolPeaks) {
  ggml_tensor* head = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, kWidth, 151936));
  for (const auto& [rows, peak] : {std::pair<std::int64_t, std::uint64_t>{17, 5'196'288},
                                   std::pair<std::int64_t, std::uint64_t>{32, 9'781'248},
                                   std::pair<std::int64_t, std::uint64_t>{512, 156'499'968}}) {
    ggml_tensor* logits = Bound(ggml_mul_mat(context(), head, F32(kWidth, rows)));
    const auto plan = CheckMulMatCublas(logits, GGML_TYPE_F16, /*f32_output=*/false);
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    EXPECT_EQ(plan->gemm, CublasGemm::kGemmEx);
    EXPECT_EQ(plan->weights, CublasOperand::kDirect);
    EXPECT_EQ(plan->input, CublasOperand::kConverted);
    EXPECT_EQ(plan->s01, kWidth);
    EXPECT_EQ(plan->s11, kWidth);
    EXPECT_EQ(plan->scratch, peak) << rows;
    EXPECT_EQ(plan->alignment, 256U);
    // An F32 output (upstream's choice on Volta) needs no temporary.
    EXPECT_EQ(CheckMulMatCublas(logits, GGML_TYPE_F16, true)->scratch,
              static_cast<std::uint64_t>(kWidth * rows * 2));
  }
}

// Attention without flash attention, as llama.cpp builds it: K and V views
// of an F16 cache, grouped over 14 query heads by 2 KV heads.
TEST_F(GgmlValidateTest, AttentionPlansFollowTheLauncher) {
  constexpr std::int64_t kHead = 64;
  constexpr std::int64_t kCells = 256;
  constexpr std::int64_t kTokens = 32;
  ggml_tensor* k_cache = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 2 * kHead, 512));
  ggml_tensor* k =
      ggml_permute(context(),
                   ggml_view_3d(context(), k_cache, kHead, 2, kCells,
                                ggml_row_size(GGML_TYPE_F16, kHead), k_cache->nb[1], 0),
                   0, 2, 1, 3);
  ggml_tensor* q = ggml_permute(
      context(), Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F32, kHead, 14, kTokens)), 0, 2, 1,
      3);
  ggml_tensor* kq = ggml_mul_mat(context(), k, q);
  ASSERT_TRUE(ggml_prec_set_acc(kq, GGML_PREC_F32));
  Bound(kq);
  // F32 compute: the K view, its bytes exactly its elements, converted in
  // place order; grouping needs pointer arrays.
  auto plan = CheckMulMatCublas(kq, GGML_TYPE_F32, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kGemmBatchedEx);
  EXPECT_EQ(plan->weights, CublasOperand::kConverted);
  EXPECT_EQ(plan->input, CublasOperand::kDirect);
  EXPECT_EQ(plan->s01, 2 * kHead);  // a cell's row of both heads
  constexpr std::uint64_t kKeys = kHead * kCells * 2 * sizeof(float);
  EXPECT_EQ(plan->scratch, kKeys + 256 + (std::uint64_t{14} * 8));  // then 2 × 14 and 14 pointers

  // A view of one head of three has gaps: gathered into packed rows.
  ggml_tensor* wide_cache = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 3 * kHead, 512));
  ggml_tensor* gapped =
      ggml_permute(context(),
                   ggml_view_3d(context(), wide_cache, kHead, 2, kCells,
                                ggml_row_size(GGML_TYPE_F16, kHead), wide_cache->nb[1], 0),
                   0, 2, 1, 3);
  ggml_tensor* kq_gapped = ggml_mul_mat(context(), gapped, q);
  ASSERT_TRUE(ggml_prec_set_acc(kq_gapped, GGML_PREC_F32));
  plan = CheckMulMatCublas(Bound(kq_gapped), GGML_TYPE_F32, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->weights, CublasOperand::kPacked);
  EXPECT_EQ(plan->s01, kHead);
  EXPECT_EQ(plan->s02, kHead * kCells);

  // KQV: the transposed V view read in place, the scores converted to F16,
  // and an F16 output temporary.
  ggml_tensor* v_cache = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 512, 2 * kHead));
  ggml_tensor* v =
      ggml_view_3d(context(), v_cache, kCells, kHead, 2, v_cache->nb[1], v_cache->nb[1] * kHead, 0);
  plan = CheckMulMatCublas(Bound(ggml_mul_mat(context(), v, kq)), GGML_TYPE_F16, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kGemmBatchedEx);
  EXPECT_EQ(plan->weights, CublasOperand::kDirect);
  EXPECT_EQ(plan->input, CublasOperand::kConverted);
  EXPECT_EQ(plan->s01, 512);
  // Its F16 output rows are 64 elements, 128 bytes apart.
  EXPECT_EQ(plan->alignment, 128U);

  // Without grouping, packed operands take the strided entry point.
  ggml_tensor* keys = Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F16, kHead, kCells, 2));
  ggml_tensor* rows = Bound(ggml_new_tensor_3d(context(), GGML_TYPE_F32, kHead, kTokens, 2));
  plan = CheckMulMatCublas(Bound(ggml_mul_mat(context(), keys, rows)), GGML_TYPE_F16, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kGemmStridedBatchedEx);
  // And one F32 matrix, Sgemm.
  ggml_tensor* square = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F32, kHead, kHead));
  plan = CheckMulMatCublas(Bound(ggml_mul_mat(context(), square, F32(kHead, kTokens))),
                           GGML_TYPE_F32, false);
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  EXPECT_EQ(plan->gemm, CublasGemm::kSgemm);
  EXPECT_EQ(plan->scratch, 0U);
}

TEST_F(GgmlValidateTest, OperandsMustClearAWorkspace) {
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 128, 64));
  ggml_tensor* x = F32(128, 32);
  ggml_tensor* y = Bound(ggml_mul_mat(context(), m, x));
  const auto at = [](const ggml_tensor* tensor) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data));
  };
  EXPECT_TRUE(CheckClearOf(y, at(y) + ggml_nbytes(y), 4096).has_value());  // just past
  EXPECT_TRUE(CheckClearOf(y, at(m) - 4096, 4096).has_value());            // just before
  EXPECT_TRUE(CheckClearOf(y, at(m), 0).has_value());                      // no workspace
  for (const ggml_tensor* tensor : {m, x, y}) {
    Rejected(CheckClearOf(y, at(tensor) + ggml_nbytes(tensor) - 1, 4096));  // the last byte
    Rejected(CheckClearOf(y, at(tensor) - 4095, 4096));                     // the first byte
  }
}

TEST_F(GgmlValidateTest, WhatCublasWouldRefuseIsRefused) {
  ggml_tensor* m = Bound(ggml_new_tensor_2d(context(), GGML_TYPE_F16, 128, 64));
  ggml_tensor* x = F32(128, 32);
  // Weight rows closer than k elements: cuBLAS needs lda >= k.
  ggml_tensor* overlapping = ggml_view_2d(context(), m, 128, 32, 64 * sizeof(ggml_fp16_t), 0);
  Rejected(CheckMulMatCublas(Bound(ggml_mul_mat(context(), overlapping, x)), GGML_TYPE_F16, false));
  // An output that is a strided view.
  ggml_tensor* big = F32(128, 64);
  ggml_tensor* product = ggml_mul_mat(context(), m, x);
  ggml_tensor* out = ggml_view_2d(context(), big, 64, 32, big->nb[1], 0);
  TensorArena::Bind(product, reinterpret_cast<std::uintptr_t>(out->data));
  product->nb[1] = out->nb[1];
  product->nb[2] = out->nb[1] * 32;
  product->nb[3] = product->nb[2];
  Rejected(CheckMulMatCublas(product, GGML_TYPE_F16, false));
  // A compute type cuBLAS does not take here, and a routing hint.
  ggml_tensor* plain = Bound(ggml_mul_mat(context(), m, x));
  EXPECT_TRUE(CheckMulMatCublas(plain, GGML_TYPE_F16, false).has_value());
  Rejected(CheckMulMatCublas(plain, GGML_TYPE_Q8_0, false));
  plain->op_params[1] = GGML_HINT_SRC0_IS_HADAMARD;
  Rejected(CheckMulMat(plain));
}

// Qwen2.5-0.5B's attention and FFN shapes, as llama.cpp builds them at the
// pin: 14 query heads and 2 KV heads of 64, an F16 cache of 1,024 cells,
// an intermediate width of 4,864.
constexpr std::int64_t kHead = 64;
constexpr std::int64_t kHeads = 14;
constexpr std::int64_t kKvHeads = 2;
constexpr std::int64_t kKvWidth = kHead * kKvHeads;
constexpr std::int64_t kCells = 1024;
constexpr std::int64_t kFfn = 4864;

// The model's RoPE (NEOX, all 64 dimensions, base 1e6), as llama.cpp calls
// ggml_rope_ext for it.
ggml_tensor* QwenRope(ggml_context* context, ggml_tensor* x, ggml_tensor* positions) {
  return ggml_rope_ext(context, x, positions, nullptr, static_cast<int>(kHead), GGML_ROPE_TYPE_NEOX,
                       32768, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

class GgmlOpsValidateTest : public GgmlValidateTest {
 protected:
  // Room for a layer's worth of nodes at five row counts.
  GgmlOpsValidateTest() { arena_ = TensorArena::Create(512).value(); }

  ggml_tensor* Typed(ggml_type type, std::int64_t ne0, std::int64_t ne1 = 1, std::int64_t ne2 = 1) {
    return Bound(ggml_new_tensor_3d(context(), type, ne0, ne1, ne2));
  }
};

TEST_F(GgmlOpsValidateTest, TheFp16PlansOperationsPass) {
  for (const std::int64_t rows : {1, 16, 17, 32, 512}) {
    // The last layer's output rows.
    ggml_tensor* hidden = F32(kWidth, rows);
    ggml_tensor* ids = Typed(GGML_TYPE_I32, rows);
    ggml_tensor* picked = Bound(ggml_get_rows(context(), hidden, ids));
    EXPECT_TRUE(CheckGetRows(picked).has_value()) << rows;
    // The recorded plan's vector kernel only at 512 rows: 224 vectors of a
    // row need one block, and fewer than 128 rows are too few blocks.
    EXPECT_EQ(GetRowsVectorized(picked), rows == 512) << rows;

    // Q's RoPE; K's RoPE, then its KV write, unfused or fused; V's write of
    // one element per row into the transposed cache.
    ggml_tensor* positions = Typed(GGML_TYPE_I32, rows);
    ggml_tensor* q = ggml_reshape_3d(context(), F32(kHead * kHeads, rows), kHead, kHeads, rows);
    ggml_tensor* q_rope = Bound(QwenRope(context(), q, positions));
    EXPECT_TRUE(CheckRope(q_rope).has_value()) << rows;
    ggml_tensor* k = ggml_reshape_3d(context(), F32(kKvWidth, rows), kHead, kKvHeads, rows);
    ggml_tensor* k_rope = Bound(QwenRope(context(), k, positions));
    ggml_tensor* k_cache = Typed(GGML_TYPE_F16, kKvWidth, kCells);
    ggml_tensor* k_ids = Typed(GGML_TYPE_I64, rows);
    ggml_tensor* k_view = ggml_view_2d(context(), k_rope, kKvWidth, rows, k_rope->nb[2], 0);
    ggml_tensor* k_write = ggml_set_rows(context(), k_cache, k_view, k_ids);
    EXPECT_TRUE(CheckSetRows(k_write).has_value()) << rows;
    ggml_tensor* fused_rope = QwenRope(context(), k, positions);  // never written
    ggml_tensor* fused_view =
        ggml_view_2d(context(), fused_rope, kKvWidth, rows, fused_rope->nb[2], 0);
    EXPECT_TRUE(CheckRopeSetRows(fused_rope, ggml_set_rows(context(), k_cache, fused_view, k_ids))
                    .has_value())
        << rows;
    ggml_tensor* v_cache = Typed(GGML_TYPE_F16, kKvWidth * kCells);
    ggml_tensor* v = ggml_reshape_2d(context(), F32(kKvWidth, rows), 1, kKvWidth * rows);
    ggml_tensor* v_ids = Typed(GGML_TYPE_I64, kKvWidth * rows);
    ggml_tensor* v_write = ggml_set_rows(
        context(), ggml_reshape_2d(context(), v_cache, 1, kKvWidth * kCells), v, v_ids);
    EXPECT_TRUE(CheckSetRows(v_write).has_value()) << rows;

    // Masked, scaled scores over 256 and 768 padded cells.
    for (const std::int64_t cells : {256, 768}) {
      ggml_tensor* scores = Typed(GGML_TYPE_F32, cells, rows, kHeads);
      ggml_tensor* mask = F32(cells, rows);
      ggml_tensor* probabilities = Bound(ggml_soft_max_ext(context(), scores, mask, 0.125f, 0.0f));
      EXPECT_TRUE(CheckSoftMax(probabilities).has_value()) << rows;
      // The recorded plan's shared memory: 1,152 bytes at 256 cells, 3,200 at 768.
      EXPECT_EQ(SoftMaxSharedBytes(probabilities), cells == 256 ? 1152U : 3200U);
    }

    // The heads merged back into rows: one copy at one row, else GGML's
    // scalar kernel.
    ggml_tensor* kqv = Typed(GGML_TYPE_F32, kHead, rows, kHeads);
    ggml_tensor* merged =
        Bound(ggml_cont_2d(context(), ggml_permute(context(), kqv, 0, 2, 1, 3), kWidth, rows));
    const auto copy = CheckCont(merged);
    ASSERT_TRUE(copy.has_value()) << copy.error().detail;
    EXPECT_EQ(*copy, rows == 1 ? ContCopy::kMemcpy : ContCopy::kScalar) << rows;

    ggml_tensor* glu = Bound(ggml_swiglu_split(context(), F32(kFfn, rows), F32(kFfn, rows)));
    EXPECT_TRUE(CheckSwiGlu(glu).has_value()) << rows;
  }

  // The decode step's fused products: Q with its bias, and the FFN's gate
  // and up products with their SwiGLU. The products are never written.
  ggml_tensor* x = F32(kWidth);
  ggml_tensor* wq = Typed(GGML_TYPE_F16, kWidth, kWidth);
  ggml_tensor* q = ggml_mul_mat(context(), wq, x);
  EXPECT_TRUE(CheckMulMatVecBias(q, Bound(ggml_add(context(), q, F32(kWidth)))).has_value());
  ggml_tensor* w_gate = Typed(GGML_TYPE_F16, kWidth, kFfn);
  ggml_tensor* w_up = Typed(GGML_TYPE_F16, kWidth, kFfn);
  ggml_tensor* gate = ggml_mul_mat(context(), w_gate, x);
  ggml_tensor* up = ggml_mul_mat(context(), w_up, x);
  EXPECT_TRUE(
      CheckMulMatVecGlu(gate, up, Bound(ggml_swiglu_split(context(), gate, up))).has_value());
}

TEST_F(GgmlOpsValidateTest, GetRowsIsRefusedWhatItsLauncherAssertsOn) {
  ggml_tensor* rows = F32(kWidth, 8);
  ggml_tensor* ids = Typed(GGML_TYPE_I32, 4);
  ggml_tensor* picked = Bound(ggml_get_rows(context(), rows, ids));
  EXPECT_TRUE(CheckGetRows(picked).has_value());
  // F16 APE tables widen into F32 without a vector-copy kernel.
  auto* const from_half = Bound(ggml_get_rows(context(), Typed(GGML_TYPE_F16, kWidth, 8), ids));
  EXPECT_TRUE(CheckGetRows(from_half));
  EXPECT_FALSE(GetRowsVectorized(from_half));
  const std::array<ggml_tensor*, 1> half_nodes = {from_half};
  auto half_plan = jitllm::kernels::ggml::PlanGraph(half_nodes, false, {});
  ASSERT_TRUE(half_plan);
  ASSERT_EQ(half_plan->steps.size(), 1U);
  EXPECT_EQ(half_plan->steps.front().implementation, jitllm::kernels::ggml::kGetRowsName);
  // An output over the rows it reads.
  ggml_tensor* over = ggml_get_rows(context(), rows, ids);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(rows->data));
  Rejected(CheckGetRows(over));
  // Misaligned rows.
  ggml_tensor* odd = ggml_new_tensor_2d(context(), GGML_TYPE_F32, kWidth, 8);
  TensorArena::Bind(odd, kBase + (next_++ * kSlot) + 2);
  Rejected(CheckGetRows(Bound(ggml_get_rows(context(), odd, ids))));
  // A shape edited after GGML built it.
  ggml_tensor* edited = Bound(ggml_get_rows(context(), rows, ids));
  edited->ne[1] = 5;
  Rejected(CheckGetRows(edited));
  // Rows at an odd pitch take the scalar kernel even at 512 ids.
  ggml_tensor* wide = F32(kWidth + 4, 512);
  ggml_tensor* pitched =
      ggml_view_2d(context(), wide, kWidth, 512, (kWidth + 1) * sizeof(float), 0);
  ggml_tensor* many = Typed(GGML_TYPE_I32, 512);
  ggml_tensor* from_pitched = Bound(ggml_get_rows(context(), pitched, many));
  EXPECT_TRUE(CheckGetRows(from_pitched).has_value());
  EXPECT_FALSE(GetRowsVectorized(from_pitched));
  EXPECT_TRUE(GetRowsVectorized(Bound(ggml_get_rows(context(), F32(kWidth, 512), many))));
}

TEST_F(GgmlOpsValidateTest, SetRowsWritesF32RowsIntoF16AtI64Indices) {
  ggml_tensor* cache = Typed(GGML_TYPE_F16, kKvWidth, kCells);
  ggml_tensor* values = F32(kKvWidth, 4);
  ggml_tensor* ids = Typed(GGML_TYPE_I64, 4);
  EXPECT_TRUE(CheckSetRows(ggml_set_rows(context(), cache, values, ids)).has_value());
  // I32 indices and an F32 destination take other kernel instances.
  Rejected(CheckSetRows(ggml_set_rows(context(), cache, values, Typed(GGML_TYPE_I32, 4))));
  Rejected(
      CheckSetRows(ggml_set_rows(context(), Typed(GGML_TYPE_F32, kKvWidth, kCells), values, ids)));
  // Values or indices inside the destination.
  ggml_tensor* inside = ggml_view_2d(context(), cache, kKvWidth / 2, 4, cache->nb[1], 0);
  ggml_tensor* inside_f32 = ggml_new_tensor_2d(context(), GGML_TYPE_F32, kKvWidth, 4);
  TensorArena::Bind(inside_f32, reinterpret_cast<std::uintptr_t>(inside->data));
  Rejected(CheckSetRows(ggml_set_rows(context(), cache, inside_f32, ids)));
  ggml_tensor* ids_inside = ggml_new_tensor_1d(context(), GGML_TYPE_I64, 4);
  TensorArena::Bind(ids_inside, reinterpret_cast<std::uintptr_t>(cache->data) + 64);
  Rejected(CheckSetRows(ggml_set_rows(context(), cache, values, ids_inside)));
  // A destination bound again after the node was built: a stale view.
  ggml_tensor* write = ggml_set_rows(context(), cache, values, ids);
  TensorArena::Bind(cache, kBase + (next_++ * kSlot));
  Rejected(CheckSetRows(write));
}

TEST_F(GgmlOpsValidateTest, RopeTakesNeoxOverF32AndMayRunInPlace) {
  ggml_tensor* positions = Typed(GGML_TYPE_I32, 3);
  ggml_tensor* x = Typed(GGML_TYPE_F32, kHead, kHeads, 3);
  EXPECT_TRUE(CheckRope(Bound(QwenRope(context(), x, positions))).has_value());
  EXPECT_TRUE(CheckRope(ggml_rope_ext_inplace(context(), x, positions, nullptr,
                                              static_cast<int>(kHead), GGML_ROPE_TYPE_NEOX, 32768,
                                              1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f))
                  .has_value());
  // The normal mode and frequency factors launch other kernels.
  Rejected(CheckRope(Bound(ggml_rope_ext(context(), x, positions, nullptr, static_cast<int>(kHead),
                                         GGML_ROPE_TYPE_NORMAL, 32768, 1000000.0f, 1.0f, 0.0f, 1.0f,
                                         32.0f, 1.0f))));
  Rejected(CheckRope(
      Bound(ggml_rope_ext(context(), x, positions, F32(kHead / 2), static_cast<int>(kHead),
                          GGML_ROPE_TYPE_NEOX, 32768, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f))));
  // A rotation offset, an odd rotated part and an output over the positions.
  ggml_tensor* offset = Bound(QwenRope(context(), x, positions));
  offset->op_params[15] = 2;
  Rejected(CheckRope(offset));
  ggml_tensor* odd = Bound(QwenRope(context(), x, positions));
  odd->op_params[1] = 63;
  Rejected(CheckRope(odd));
  ggml_tensor* over = QwenRope(context(), x, positions);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(positions->data));
  Rejected(CheckRope(over));
  // A single-row output one head into its input.
  ggml_tensor* shifted = QwenRope(context(), x, positions);
  TensorArena::Bind(shifted, reinterpret_cast<std::uintptr_t>(x->data) + x->nb[1]);
  Rejected(CheckRope(shifted));
}

TEST_F(GgmlOpsValidateTest, TheFusedRopeWritesOnlyAFlatteningViewIntoF16) {
  ggml_tensor* positions = Typed(GGML_TYPE_I32, 3);
  ggml_tensor* x = Typed(GGML_TYPE_F32, kHead, kKvHeads, 3);
  ggml_tensor* cache = Typed(GGML_TYPE_F16, kKvWidth, kCells);
  ggml_tensor* ids = Typed(GGML_TYPE_I64, 3);
  const auto write = [&](ggml_tensor* rope, ggml_tensor* destination, std::size_t offset = 0) {
    ggml_tensor* view = ggml_view_2d(context(), rope, kKvWidth, 3, rope->nb[2], offset);
    return ggml_set_rows(context(), destination, view, ids);
  };
  ggml_tensor* rope = QwenRope(context(), x, positions);
  EXPECT_TRUE(CheckRopeSetRows(rope, write(rope, cache)).has_value());
  // Upstream fuses into F32 too; the implementation writes the F16 cache.
  Rejected(CheckRopeSetRows(rope, write(rope, Typed(GGML_TYPE_F32, kKvWidth, kCells))));
  // A view that does not start at the RoPE, or of another tensor.
  ggml_tensor* other = QwenRope(context(), x, positions);
  Rejected(CheckRopeSetRows(rope, write(other, cache)));
  // A write over the RoPE's input.
  ggml_tensor* over = ggml_new_tensor_2d(context(), GGML_TYPE_F16, kKvWidth, kCells);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(x->data));
  Rejected(CheckRopeSetRows(rope, write(rope, over)));
  // An in-place RoPE is a view of its input, which upstream's gate never
  // fuses.
  ggml_tensor* in_place =
      ggml_rope_ext_inplace(context(), x, positions, nullptr, static_cast<int>(kHead),
                            GGML_ROPE_TYPE_NEOX, 32768, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
  Rejected(CheckRopeSetRows(in_place, write(in_place, cache)));
}

TEST_F(GgmlOpsValidateTest, SoftMaxTakesAnF32MaskAndNoAlibi) {
  ggml_tensor* scores = Typed(GGML_TYPE_F32, 256, 4, kHeads);
  ggml_tensor* mask = F32(256, 4);
  EXPECT_TRUE(
      CheckSoftMax(Bound(ggml_soft_max_ext(context(), scores, mask, 0.125f, 0.0f))).has_value());
  EXPECT_TRUE(
      CheckSoftMax(Bound(ggml_soft_max_ext(context(), scores, nullptr, 1.0f, 0.0f))).has_value());
  EXPECT_TRUE(
      CheckSoftMax(ggml_soft_max_ext_inplace(context(), scores, mask, 0.125f, 0.0f)).has_value());
  Rejected(CheckSoftMax(Bound(ggml_soft_max_ext(context(), scores, mask, 0.125f, 8.0f))));
  Rejected(CheckSoftMax(
      Bound(ggml_soft_max_ext(context(), scores, Typed(GGML_TYPE_F16, 256, 4), 0.125f, 0.0f))));
  // A mask with fewer rows than the scores, edited after GGML built it.
  ggml_tensor* short_mask = Bound(ggml_soft_max_ext(context(), scores, F32(256, 4), 0.125f, 0.0f));
  short_mask->src[1]->ne[1] = 3;
  Rejected(CheckSoftMax(short_mask));
  // Output over the mask.
  ggml_tensor* over = ggml_soft_max_ext(context(), scores, mask, 0.125f, 0.0f);
  TensorArena::Bind(over, reinterpret_cast<std::uintptr_t>(mask->data));
  Rejected(CheckSoftMax(over));
}

TEST_F(GgmlOpsValidateTest, ContCopiesAsUpstreamChooses) {
  // A pitched block of the same shape: one two-dimensional copy.
  ggml_tensor* wide = F32(kWidth + 16, 4);
  ggml_tensor* block = ggml_view_2d(context(), wide, kWidth, 4, wide->nb[1], 0);
  const auto pitched = CheckCont(Bound(ggml_cont(context(), block)));
  ASSERT_TRUE(pitched.has_value()) << pitched.error().detail;
  EXPECT_EQ(*pitched, ContCopy::kMemcpy2d);
  // Transposed rows: upstream's tiled transpose (DeepSeek V4's compressor
  // transposes its windows so).
  ggml_tensor* square = F32(64, 32);
  const auto tiled = CheckCont(Bound(ggml_cont(context(), ggml_transpose(context(), square))));
  ASSERT_TRUE(tiled.has_value()) << tiled.error().detail;
  EXPECT_EQ(*tiled, ContCopy::kTranspose);
  ggml_tensor* windows = Typed(GGML_TYPE_F32, 512, 8, 3);
  const auto permuted =
      CheckCont(Bound(ggml_cont(context(), ggml_permute(context(), windows, 1, 0, 2, 3))));
  ASSERT_TRUE(permuted.has_value()) << permuted.error().detail;
  EXPECT_EQ(*permuted, ContCopy::kTranspose);
  // Upstream would tile this transpose of a narrowed matrix too (its rows
  // are columns, and its one matrix's stride, free since ne2 is 1, is
  // ne0·ne1 elements), but the tile kernel reads column i0 at i0·ne1
  // elements, not at nb[0]: refused.
  ggml_tensor* narrow = ggml_transpose(
      context(),
      ggml_view_3d(context(), F32(8, 4), 2, 4, 1, 8 * sizeof(float), 8 * sizeof(float), 0));
  ASSERT_EQ(narrow->nb[0], 8 * sizeof(float));
  ASSERT_EQ(narrow->nb[1], sizeof(float));
  ASSERT_EQ(narrow->nb[2], static_cast<std::size_t>(narrow->ne[0] * narrow->ne[1]) * sizeof(float));
  Rejected(CheckCont(Bound(ggml_cont(context(), narrow))));
  // Contiguous I32 (the indexer's top-k) copies whole; strided I32 does not.
  const auto ids = CheckCont(Bound(ggml_cont(context(), Typed(GGML_TYPE_I32, 64, 2))));
  ASSERT_TRUE(ids.has_value()) << ids.error().detail;
  EXPECT_EQ(*ids, ContCopy::kMemcpy);
  ggml_tensor* id_rows = Typed(GGML_TYPE_I32, 64, 2);
  Rejected(CheckCont(
      Bound(ggml_cont(context(), ggml_view_2d(context(), id_rows, 32, 2, id_rows->nb[1], 0)))));
  // F16, and an output over its input.
  Rejected(CheckCont(Bound(ggml_cont(context(), Typed(GGML_TYPE_F16, 64, 2)))));
  ggml_tensor* heads = Typed(GGML_TYPE_F32, kHead, 4, kHeads);
  ggml_tensor* merged =
      ggml_cont_2d(context(), ggml_permute(context(), heads, 0, 2, 1, 3), kWidth, 4);
  TensorArena::Bind(merged, reinterpret_cast<std::uintptr_t>(heads->data) + 256);
  Rejected(CheckCont(merged));
}

TEST_F(GgmlOpsValidateTest, SwiGluIsSplitAndMayRunInPlace) {
  ggml_tensor* gate = F32(kFfn, 3);
  ggml_tensor* up = F32(kFfn, 3);
  EXPECT_TRUE(CheckSwiGlu(Bound(ggml_swiglu_split(context(), gate, up))).has_value());
  // In place over the gate: each element is read, then written.
  ggml_tensor* in_place = ggml_swiglu_split(context(), gate, up);
  TensorArena::Bind(in_place, reinterpret_cast<std::uintptr_t>(gate->data));
  EXPECT_TRUE(CheckSwiGlu(in_place).has_value());
  // One tensor holding both halves, and another GLU operation.
  Rejected(CheckSwiGlu(Bound(ggml_swiglu(context(), F32(2 * kFfn, 3)))));
  Rejected(CheckSwiGlu(Bound(ggml_geglu_split(context(), gate, up))));
  // An output one row into the gate.
  ggml_tensor* shifted = ggml_swiglu_split(context(), gate, up);
  TensorArena::Bind(shifted, reinterpret_cast<std::uintptr_t>(gate->data) + gate->nb[1]);
  Rejected(CheckSwiGlu(shifted));
}

TEST_F(GgmlOpsValidateTest, GemmaGeGluTakesIndependentUniformRowsAndRefusesVariants) {
  namespace kg = jitllm::kernels::ggml;
  for (const std::int64_t width : {2112, 704}) {
    for (const std::int64_t rows : {1, 4}) {
      auto* both = F32(2 * width, rows);
      auto* gate = ggml_view_2d(context(), both, width, rows, both->nb[1], 0);
      auto* up = ggml_view_2d(context(), both, width, rows, both->nb[1],
                              static_cast<std::size_t>(width) * sizeof(float));
      auto* glu = Bound(ggml_geglu_split(context(), gate, up));
      EXPECT_TRUE(kg::CheckGeGlu(glu));
      Rejected(CheckSwiGlu(glu));
      Rejected(kg::CheckGeGlu(Bound(ggml_swiglu_split(context(), gate, up))));
      glu->op_params[1] = 1;
      Rejected(kg::CheckGeGlu(glu));
      glu->op_params[1] = 0;
      TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(both->data));
      Rejected(kg::CheckGeGlu(glu));  // strided gate cannot be overwritten densely
    }
  }
  auto* gate = F32(704, 4);
  auto* up = F32(704, 4);
  auto* glu = ggml_geglu_split(context(), gate, up);
  TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(gate->data));
  EXPECT_TRUE(kg::CheckGeGlu(glu));  // exact in place over packed gate
  TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(up->data));
  EXPECT_TRUE(kg::CheckGeGlu(glu));
  TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(up->data) + sizeof(float));
  Rejected(kg::CheckGeGlu(glu));
  Rejected(kg::CheckGeGlu(Bound(ggml_geglu(context(), F32(1408, 4)))));
  Rejected(kg::CheckGeGlu(Bound(ggml_geglu_erf_split(context(), gate, up))));
  Rejected(kg::CheckGeGlu(Bound(ggml_geglu_quick_split(context(), gate, up))));
  Rejected(kg::CheckGeGlu(Bound(
      ggml_geglu_split(context(), Typed(GGML_TYPE_F16, 704, 4), Typed(GGML_TYPE_F16, 704, 4)))));
  TensorArena::Bind(glu, kBase + 60 * kSlot + 1);
  Rejected(kg::CheckGeGlu(glu));
}

TEST_F(GgmlOpsValidateTest, GemmaExpertGeGluViewsRequireUniformRowsAndCurrentBindings) {
  namespace kg = jitllm::kernels::ggml;
  auto* both = Typed(GGML_TYPE_F32, 1408, 8, 4);
  auto* gate = ggml_view_3d(context(), both, 704, 8, 4, both->nb[1], both->nb[2], 0);
  auto* up =
      ggml_view_3d(context(), both, 704, 8, 4, both->nb[1], both->nb[2], 704 * sizeof(float));
  auto* glu = Bound(ggml_geglu_split(context(), gate, up));
  EXPECT_TRUE(kg::CheckGeGlu(glu));
  up->nb[2] += sizeof(float);
  Rejected(kg::CheckGeGlu(glu));
  up->nb[2] -= sizeof(float);
  up->nb[0] *= 2;
  Rejected(kg::CheckGeGlu(glu));
  up->nb[0] /= 2;
  TensorArena::Bind(both, kBase + 99 * kSlot);  // stale children
  Rejected(kg::CheckGeGlu(glu));
}

TEST_F(GgmlOpsValidateTest, GemmaMmvfGeGluFusionPreservesItsOwnOneColumnContract) {
  namespace kg = jitllm::kernels::ggml;
  for (const std::int64_t rows : {1, 4}) {
    auto* x = F32(2816, rows);
    auto* wg = Typed(GGML_TYPE_F16, 2816, 2112);
    auto* wu = Typed(GGML_TYPE_F16, 2816, 2112);
    auto* gate = ggml_mul_mat(context(), wg, x);
    auto* up = ggml_mul_mat(context(), wu, x);
    ggml_prec_set_acc(gate, GGML_PREC_F32);
    ggml_prec_set_acc(up, GGML_PREC_F32);
    auto* glu = Bound(ggml_geglu_split(context(), gate, up));
    EXPECT_EQ(kg::CheckMulMatVecGeGlu(gate, up, glu).has_value(), rows == 1);
    ggml_prec_set_acc(gate, GGML_PREC_DEFAULT);
    Rejected(kg::CheckMulMatVecGeGlu(gate, up, glu));
    ggml_prec_set_acc(gate, GGML_PREC_F32);
    Rejected(CheckMulMatVecGlu(gate, up, glu));
    Rejected(kg::CheckMulMatVecGeGlu(up, gate, glu));
    glu->op_params[1] = 1;
    Rejected(kg::CheckMulMatVecGeGlu(gate, up, glu));
    glu->op_params[1] = 0;
    TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(wg->data));
    Rejected(kg::CheckMulMatVecGeGlu(gate, up, glu));
  }
}

TEST_F(GgmlOpsValidateTest, FusedMmvfTakesOneColumnAndItsOwnOperands) {
  ggml_tensor* x = F32(kWidth);
  ggml_tensor* w = Typed(GGML_TYPE_F16, kWidth, kWidth);
  ggml_tensor* bias = F32(kWidth);
  ggml_tensor* product = ggml_mul_mat(context(), w, x);
  EXPECT_TRUE(CheckMulMatVecBias(product, Bound(ggml_add(context(), bias, product))).has_value());
  // In place over the bias (a residual that is not needed again).
  ggml_tensor* into_bias = ggml_add(context(), product, bias);
  TensorArena::Bind(into_bias, reinterpret_cast<std::uintptr_t>(bias->data));
  EXPECT_TRUE(CheckMulMatVecBias(product, into_bias).has_value());
  // Two columns; a broadcast bias; an add of something else; an output over
  // the input.
  ggml_tensor* two = F32(kWidth, 2);
  ggml_tensor* products = ggml_mul_mat(context(), w, two);
  Rejected(CheckMulMatVecBias(products, Bound(ggml_add(context(), products, F32(kWidth, 2)))));
  Rejected(CheckMulMatVecBias(products, Bound(ggml_add(context(), products, bias))));
  Rejected(CheckMulMatVecBias(product, Bound(ggml_add(context(), bias, bias))));
  ggml_tensor* over_input = ggml_add(context(), product, bias);
  TensorArena::Bind(over_input, reinterpret_cast<std::uintptr_t>(x->data));
  Rejected(CheckMulMatVecBias(product, over_input));

  ggml_tensor* w_gate = Typed(GGML_TYPE_F16, kWidth, kFfn);
  ggml_tensor* w_up = Typed(GGML_TYPE_F16, kWidth, kFfn);
  ggml_tensor* gate = ggml_mul_mat(context(), w_gate, x);
  ggml_tensor* up = ggml_mul_mat(context(), w_up, x);
  EXPECT_TRUE(
      CheckMulMatVecGlu(gate, up, Bound(ggml_swiglu_split(context(), gate, up))).has_value());
  // Gate and up in the other order, or of different inputs.
  Rejected(CheckMulMatVecGlu(up, gate, Bound(ggml_swiglu_split(context(), gate, up))));
  ggml_tensor* other_up = ggml_mul_mat(context(), w_up, F32(kWidth));
  Rejected(CheckMulMatVecGlu(gate, other_up, Bound(ggml_swiglu_split(context(), gate, other_up))));
  // Gate weights of another type, or F32 weights paired with F16.
  ggml_tensor* f32_gate = ggml_mul_mat(context(), Typed(GGML_TYPE_F32, kWidth, kFfn), x);
  Rejected(CheckMulMatVecGlu(f32_gate, up, Bound(ggml_swiglu_split(context(), f32_gate, up))));
  // GEGLU: upstream fuses it, the implementation takes SwiGLU.
  Rejected(CheckMulMatVecGlu(gate, up, Bound(ggml_geglu_split(context(), gate, up))));
  // The GLU over the gate weights.
  ggml_tensor* over_weights = ggml_swiglu_split(context(), gate, up);
  TensorArena::Bind(over_weights, reinterpret_cast<std::uintptr_t>(w_gate->data));
  Rejected(CheckMulMatVecGlu(gate, up, over_weights));
}

// Upstream's fusion gates on GGML graphs of Qwen2's layer, as llama.cpp
// builds them.
class GgmlFusionTest : public GgmlOpsValidateTest {
 protected:
  // The nodes GGML's graph would record for `outputs`, in its order.
  static std::vector<ggml_tensor*> Graph(std::initializer_list<ggml_tensor*> outputs) {
    return GraphOrder(std::span<ggml_tensor* const>(outputs.begin(), outputs.size()));
  }

  static std::size_t IndexOf(const std::vector<ggml_tensor*>& graph, const ggml_tensor* node) {
    return static_cast<std::size_t>(std::ranges::find(graph, node) - graph.begin());
  }

  // An RMSNorm-mul, the input that follows it in the layer.
  ggml_tensor* Normed(std::int64_t rows = 1) {
    ggml_tensor* norm = Bound(ggml_rms_norm(context(), F32(kWidth, rows), 1e-6f));
    return Bound(ggml_mul(context(), norm, F32(kWidth)));
  }
};

TEST_F(GgmlFusionTest, TheDecodeFfnFusesGateUpAndDownWithTheResidual) {
  ggml_tensor* residual = F32(kWidth);
  ggml_tensor* x = Normed();
  ggml_tensor* up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), x));
  ggml_tensor* gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), x));
  ggml_tensor* glu = Bound(ggml_swiglu_split(context(), gate, up));
  ggml_tensor* down = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kFfn, kWidth), glu));
  ggml_tensor* out = Bound(ggml_add(context(), down, residual));
  const std::vector<ggml_tensor*> graph = Graph({out});
  // GGML's order, inputs first and left to right, leaves omitted: the gate
  // product comes first, as the gate requires.
  EXPECT_EQ(graph, (std::vector<ggml_tensor*>{x->src[0], x, gate, up, glu, down, out}));
  const std::size_t at = IndexOf(graph, gate);
  ASSERT_LT(at, graph.size());
  ASSERT_EQ(IndexOf(graph, up), at + 1);
  const auto fused = MulMatGluFusionAt(graph, at);
  if (!fused) {
    FAIL() << "not fused";
  }
  const jitllm::kernels::ggml::MulMatGluNodes nodes = *fused;
  EXPECT_EQ(nodes.gate, gate);
  EXPECT_EQ(nodes.up, up);
  EXPECT_EQ(nodes.glu, glu);
  EXPECT_TRUE(CheckMulMatVecGlu(nodes.gate, nodes.up, nodes.glu).has_value());
  EXPECT_FALSE(MulMatGluFusionAt(graph, at + 1).has_value());
  EXPECT_FALSE(MulMatAddFusionAt(graph, at).has_value());
  const auto residual_add = MulMatAddFusionAt(graph, IndexOf(graph, down));
  if (!residual_add) {
    FAIL() << "not fused";
  }
  EXPECT_EQ(residual_add->mul_mat, down);
  EXPECT_EQ(residual_add->add, out);
}

TEST_F(GgmlFusionTest, TheGluGateComparesDataRangesAndUses) {
  // The GLU written over its input, a computed node: upstream does not fuse.
  ggml_tensor* x = Normed();
  ggml_tensor* up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), x));
  ggml_tensor* gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), x));
  ggml_tensor* glu = ggml_swiglu_split(context(), gate, up);
  TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(x->data));
  const std::vector<ggml_tensor*> graph = Graph({glu});
  const std::size_t at = IndexOf(graph, gate);
  EXPECT_FALSE(FusionMemoryClear(graph, at, 3, at + 2));
  EXPECT_FALSE(MulMatGluFusionAt(graph, at).has_value());
  // Over an elided intermediate it may be.
  TensorArena::Bind(glu, reinterpret_cast<std::uintptr_t>(up->data));
  EXPECT_TRUE(MulMatGluFusionAt(graph, at).has_value());
  // Over a leaf input upstream does not look.
  ggml_tensor* leaf = F32(kWidth);
  ggml_tensor* leaf_up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), leaf));
  ggml_tensor* leaf_gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), leaf));
  ggml_tensor* over_leaf = ggml_swiglu_split(context(), leaf_gate, leaf_up);
  TensorArena::Bind(over_leaf, reinterpret_cast<std::uintptr_t>(leaf->data));
  const std::vector<ggml_tensor*> leaf_graph = Graph({over_leaf});
  EXPECT_TRUE(MulMatGluFusionAt(leaf_graph, IndexOf(leaf_graph, leaf_gate)).has_value());

  // A product with another use, or marked as a graph output.
  ggml_tensor* y = Normed();
  ggml_tensor* y_up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), y));
  ggml_tensor* y_gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), y));
  ggml_tensor* y_glu = Bound(ggml_swiglu_split(context(), y_gate, y_up));
  ggml_tensor* also = Bound(ggml_add(context(), y_up, y_up));
  const std::vector<ggml_tensor*> used = Graph({y_glu, also});
  EXPECT_FALSE(MulMatGluFusionAt(used, IndexOf(used, y_gate)).has_value());
  ggml_tensor* z = Normed();
  ggml_tensor* z_up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), z));
  ggml_tensor* z_gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), z));
  ggml_set_output(z_gate);
  const std::vector<ggml_tensor*> output = Graph({ggml_swiglu_split(context(), z_gate, z_up)});
  EXPECT_FALSE(MulMatGluFusionAt(output, IndexOf(output, z_gate)).has_value());
}

TEST_F(GgmlFusionTest, TheGluGateNeedsTheGateProductFirstAndTheSameWeightsLayout) {
  // The up product computed first: upstream's gate reads the pair by
  // position and does not fuse.
  ggml_tensor* x = Normed();
  ggml_tensor* up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), x));
  ggml_tensor* gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), x));
  ggml_tensor* glu = Bound(ggml_swiglu_split(context(), gate, up));
  const std::vector<ggml_tensor*> graph = Graph({up, glu});
  ASSERT_EQ(IndexOf(graph, gate), IndexOf(graph, up) + 1);
  EXPECT_FALSE(MulMatGluFusionAt(graph, IndexOf(graph, up)).has_value());
  // Weights of different types.
  ggml_tensor* y = Normed();
  ggml_tensor* f32_up = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F32, kWidth, kFfn), y));
  ggml_tensor* f16_gate = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kFfn), y));
  const std::vector<ggml_tensor*> mixed = Graph({ggml_swiglu_split(context(), f16_gate, f32_up)});
  EXPECT_FALSE(MulMatGluFusionAt(mixed, IndexOf(mixed, f16_gate)).has_value());
}

TEST_F(GgmlFusionTest, TheBiasGateTakesAnAddOfTheProductsShape) {
  // Q with its bias, then its reshape for RoPE.
  ggml_tensor* x = Normed();
  ggml_tensor* q = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kWidth), x));
  ggml_tensor* biased = Bound(ggml_add(context(), q, F32(kWidth)));
  const std::vector<ggml_tensor*> graph =
      Graph({ggml_reshape_3d(context(), biased, kHead, kHeads, 1)});
  const auto fused = MulMatAddFusionAt(graph, IndexOf(graph, q));
  if (!fused) {
    FAIL() << "not fused";
  }
  EXPECT_TRUE(CheckMulMatVecBias(fused->mul_mat, fused->add).has_value());
  // A broadcast bias over two columns is not fused; neither is a product
  // with a second use.
  ggml_tensor* rows = Normed(2);
  ggml_tensor* q2 = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kWidth), rows));
  const std::vector<ggml_tensor*> broadcast = Graph({Bound(ggml_add(context(), q2, F32(kWidth)))});
  EXPECT_FALSE(MulMatAddFusionAt(broadcast, IndexOf(broadcast, q2)).has_value());
  ggml_tensor* q3 = Bound(ggml_mul_mat(context(), Typed(GGML_TYPE_F16, kWidth, kWidth), x));
  ggml_tensor* sum = Bound(ggml_add(context(), q3, F32(kWidth)));
  const std::vector<ggml_tensor*> reused = Graph({sum, Bound(ggml_mul(context(), q3, q3))});
  EXPECT_FALSE(MulMatAddFusionAt(reused, IndexOf(reused, q3)).has_value());
}

TEST_F(GgmlFusionTest, TheRopeGateFusesTheKWriteUnlessItOverlapsTheInput) {
  constexpr std::int64_t kRows = 4;
  const auto k_write = [&](ggml_tensor* cache, int mode) {
    ggml_tensor* k = Bound(ggml_add(context(), F32(kKvWidth, kRows), F32(kKvWidth)));
    ggml_tensor* positions = Typed(GGML_TYPE_I32, kRows);
    ggml_tensor* rope = Bound(ggml_rope_ext(
        context(), ggml_reshape_3d(context(), k, kHead, kKvHeads, kRows), positions, nullptr,
        static_cast<int>(kHead), mode, 32768, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f));
    ggml_tensor* view = ggml_view_2d(context(), rope, kKvWidth, kRows, rope->nb[2], 0);
    return std::pair{rope, ggml_set_rows(context(), cache, view, Typed(GGML_TYPE_I64, kRows))};
  };
  ggml_tensor* cache = Typed(GGML_TYPE_F16, kKvWidth, kCells);
  const auto [rope, write] = k_write(cache, GGML_ROPE_TYPE_NEOX);
  const std::vector<ggml_tensor*> graph = Graph({write});
  const auto fused = RopeSetRowsFusionAt(graph, IndexOf(graph, rope));
  if (!fused) {
    FAIL() << "not fused";
  }
  EXPECT_EQ(fused->set_rows, write);
  EXPECT_TRUE(CheckRopeSetRows(fused->rope, fused->set_rows).has_value());
  // The normal mode fuses upstream too; the implementation refuses it.
  const auto [normal, normal_write] =
      k_write(Typed(GGML_TYPE_F16, kKvWidth, kCells), GGML_ROPE_TYPE_NORMAL);
  const std::vector<ggml_tensor*> normal_graph = Graph({normal_write});
  ASSERT_TRUE(RopeSetRowsFusionAt(normal_graph, IndexOf(normal_graph, normal)).has_value());
  Rejected(CheckRopeSetRows(normal, normal_write));
  // A cache over the RoPE's (computed) input: upstream does not fuse.
  ggml_tensor* overlapping = ggml_new_tensor_2d(context(), GGML_TYPE_F16, kKvWidth, kCells);
  const auto [over_rope, over_write] = k_write(overlapping, GGML_ROPE_TYPE_NEOX);
  TensorArena::Bind(overlapping, reinterpret_cast<std::uintptr_t>(over_rope->src[0]->data));
  TensorArena::Bind(over_write, reinterpret_cast<std::uintptr_t>(over_rope->src[0]->data));
  const std::vector<ggml_tensor*> over_graph = Graph({over_write});
  EXPECT_FALSE(RopeSetRowsFusionAt(over_graph, IndexOf(over_graph, over_rope)).has_value());
  // Q's RoPE is not followed by a write.
  ggml_tensor* q = ggml_reshape_3d(context(), F32(kHead * kHeads, kRows), kHead, kHeads, kRows);
  ggml_tensor* q_rope = Bound(QwenRope(context(), q, Typed(GGML_TYPE_I32, kRows)));
  const std::vector<ggml_tensor*> q_graph = Graph({q_rope});
  EXPECT_FALSE(RopeSetRowsFusionAt(q_graph, IndexOf(q_graph, q_rope)).has_value());
}

}  // namespace
