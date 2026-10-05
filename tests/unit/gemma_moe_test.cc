// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_moe.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

namespace {
namespace kg = jitllm::kernels::ggml;
using jitllm::base::Bytes;
class GemmaMoe : public ::testing::Test {
 protected:
  void SetUp() override {
    arena = std::make_unique<kg::TensorArena>(kg::TensorArena::Create(32).value());
  }
  ggml_context* c() { return arena->context(); }
  ggml_tensor* Tensor(ggml_type type, std::array<std::int64_t, 4> ne, std::uint64_t address) {
    auto* t = ggml_new_tensor_4d(c(), type, ne[0], ne[1], ne[2], ne[3]);
    kg::TensorArena::Bind(t, address);
    return t;
  }
  kg::GemmaRouting Route(std::int64_t rows) {
    auto* x = Tensor(GGML_TYPE_F32, {128, rows, 1, 1}, 0x100000000ULL);
    auto* w = Tensor(GGML_TYPE_F32, {1, 8, rows, 1}, 0x200000000ULL);
    auto* root = ggml_argsort(c(), x, GGML_SORT_ORDER_DESC);
    kg::TensorArena::Bind(root, 0x300000000ULL);
    auto* ids = ggml_view_2d(c(), root, 8, rows, root->nb[1], 0);
    return {{x, Bytes(static_cast<std::uint64_t>(rows) * 128 * 4)},
            {w, Bytes(static_cast<std::uint64_t>(rows) * 8 * 4)},
            {ids, Bytes(static_cast<std::uint64_t>(rows) * 128 * 4)}};
  }
  kg::GemmaScaledReduction Reduce(std::int64_t rows) {
    const std::array<std::array<std::int64_t, 4>, 4> shapes = {
        {{2816, 8, rows, 1}, {1, 8, rows, 1}, {1, 8, rows, 1}, {2816, rows, 1, 1}}};
    std::array<kg::GemmaMoeOperand, 4> o{};
    for (std::size_t i = 0; i < o.size(); ++i) {
      auto* t = Tensor(GGML_TYPE_F32, shapes[i], (i + 1) * 0x100000000ULL);
      o[i] = {t, Bytes(ggml_nbytes(t))};
    }
    return {o[0], o[1], o[2], o[3]};
  }
  std::unique_ptr<kg::TensorArena> arena;
};
TEST_F(GemmaMoe, ExistingArgsortTop8ViewHasFullyFundedPitch) {
  for (const auto rows : {1, 2, 4, 8, 128, 8192}) {
    arena->Reset();
    auto d = Route(rows);
    // Build the actual Gemma helper, not the distinct compact ggml_top_k.
    auto* root = const_cast<ggml_tensor*>(d.ids.tensor->view_src);
    auto* actual = ggml_argsort_top_k(c(), const_cast<ggml_tensor*>(d.logits.tensor), 8);
    kg::TensorArena::Bind(actual->view_src, reinterpret_cast<std::uintptr_t>(root->data));
    actual->data = actual->view_src->data;
    d.ids.tensor = actual;
    ASSERT_TRUE(kg::CheckGemmaRouting(d));
    EXPECT_EQ(actual->ne[0], 8);
    EXPECT_EQ(actual->nb[1], 512U);
    EXPECT_EQ(actual->view_src->ne[0], 128);
    EXPECT_EQ(d.ids.bytes.value(), static_cast<std::uint64_t>(rows) * 512);
  }
}
TEST_F(GemmaMoe, RoutingRefusesShortTailCompactIdsAndFullSortConsumers) {
  auto d = Route(4);
  auto bad = d;
  bad.ids.bytes = Bytes(ggml_nbytes(d.ids.tensor));
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  bad = d;
  bad.ids.bytes = Bytes(d.ids.bytes.value() - 1);
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  bad = d;
  bad.ids.tensor = ggml_top_k(c(), const_cast<ggml_tensor*>(d.logits.tensor), 8);
  kg::TensorArena::Bind(const_cast<ggml_tensor*>(bad.ids.tensor), 0x300000000ULL);
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  bad = d;
  bad.ids_use = kg::GemmaRouteIdsUse::kFullSort;
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  bad.ids_use = static_cast<kg::GemmaRouteIdsUse>(255);
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  bad = d;
  bad.denominator_min = 1e-6f;
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  bad.denominator_min = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  EXPECT_TRUE(kg::CheckGemmaRouting(d));
}
TEST_F(GemmaMoe, RoutingRefusesAliasesStaleViewsEndOverflowAndBadMetadata) {
  auto d = Route(8);
  auto* ids = const_cast<ggml_tensor*>(d.ids.tensor);
  auto* root = ids->view_src;
  const auto saved = *root;
  root->data = d.logits.tensor->data;
  ids->data = root->data;
  EXPECT_FALSE(kg::CheckGemmaRouting(d));
  // Alias only in the unwritten full-sort tail still refuses.
  *root = saved;
  ids->data = root->data;
  auto* w = const_cast<ggml_tensor*>(d.weights.tensor);
  const auto ws = *w;
  w->data = static_cast<char*>(root->data) + 32;
  EXPECT_FALSE(kg::CheckGemmaRouting(d));
  *w = ws;
  root->data = reinterpret_cast<void*>(0x400000000ULL);
  EXPECT_FALSE(kg::CheckGemmaRouting(d));
  *root = saved;
  auto bad = d;
  bad.logits.bytes = Bytes(std::numeric_limits<std::uint64_t>::max());
  EXPECT_FALSE(kg::CheckGemmaRouting(bad));
  auto* x = const_cast<ggml_tensor*>(d.logits.tensor);
  const auto xs = *x;
  for (const auto n : {0LL, 8193LL, 16777216LL, std::numeric_limits<long long>::max()}) {
    x->ne[1] = n;
    EXPECT_FALSE(kg::CheckGemmaRouting(d));
  }
  *x = xs;
  x->type = static_cast<ggml_type>(-1);
  EXPECT_FALSE(kg::CheckGemmaRouting(d));
  *x = xs;
  x->nb[1] += 4;
  EXPECT_FALSE(kg::CheckGemmaRouting(d));
  *x = xs;
  root->view_src = root;
  EXPECT_FALSE(kg::CheckGemmaRouting(d));
  *root = saved;
  EXPECT_TRUE(kg::CheckGemmaRouting(d));
}
TEST_F(GemmaMoe, ReductionRequiresEveryScaleAndExactShapeStorageAndOwnership) {
  for (const auto rows : {1, 2, 4, 8, 128, 8192}) {
    arena->Reset();
    auto d = Reduce(rows);
    ASSERT_TRUE(kg::CheckGemmaScaledReduction(d));
    for (auto member : {&kg::GemmaScaledReduction::experts, &kg::GemmaScaledReduction::scales,
                        &kg::GemmaScaledReduction::weights, &kg::GemmaScaledReduction::values}) {
      auto bad = d;
      (bad.*member).bytes = Bytes((bad.*member).bytes.value() - 1);
      EXPECT_FALSE(kg::CheckGemmaScaledReduction(bad));
      bad = d;
      (bad.*member).bytes = Bytes(std::numeric_limits<std::uint64_t>::max());
      EXPECT_FALSE(kg::CheckGemmaScaledReduction(bad));
      bad = d;
      (bad.*member).tensor = nullptr;
      EXPECT_FALSE(kg::CheckGemmaScaledReduction(bad));
    }
    auto* out = const_cast<ggml_tensor*>(d.values.tensor);
    const auto saved = *out;
    out->data = d.scales.tensor->data;
    EXPECT_FALSE(kg::CheckGemmaScaledReduction(d));
    *out = saved;
    out->nb[1] += 4;
    EXPECT_FALSE(kg::CheckGemmaScaledReduction(d));
    *out = saved;
    auto* input = const_cast<ggml_tensor*>(d.experts.tensor);
    const auto is = *input;
    input->ne[0] = 4096;
    input->ne[1] = 6;
    EXPECT_FALSE(kg::CheckGemmaScaledReduction(d));
    *input = is;
    input->ne[2] = std::numeric_limits<std::int64_t>::max();
    EXPECT_FALSE(kg::CheckGemmaScaledReduction(d));
    *input = is;
    input->view_src = input;
    EXPECT_FALSE(kg::CheckGemmaScaledReduction(d));
    *input = is;
    EXPECT_TRUE(kg::CheckGemmaScaledReduction(d));
  }
}
}  // namespace
