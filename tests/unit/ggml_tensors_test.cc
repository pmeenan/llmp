// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// GGML tensor descriptors over jitLLM memory (kernels/ggml/tensors.h), in
// every profile: nodes built with GGML's graph functions keep upstream's
// shapes and parameters, point where they are bound, allocate no data, and
// an arena refuses more tensors than it holds before GGML could abort.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/tensors.h"

namespace {

using jitllm::kernels::ggml::KernelError;
using jitllm::kernels::ggml::TensorArena;
using jitllm::test_support::FailedCode;

std::uint64_t Address(const ggml_tensor* tensor) {
  return reinterpret_cast<std::uintptr_t>(tensor->data);
}

TEST(GgmlTensors, NodesKeepUpstreamShapesAndPointWhereBound) {
  auto arena = TensorArena::Create(8).value();
  ggml_context* context = arena.context();
  ggml_tensor* input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 896, 5);
  ggml_tensor* weight = ggml_new_tensor_1d(context, GGML_TYPE_F32, 896);
  ggml_tensor* matrix = ggml_new_tensor_2d(context, GGML_TYPE_F16, 896, 4864);
  EXPECT_EQ(input->data, nullptr);  // no_alloc: GGML allocates no data
  TensorArena::Bind(input, 0x10000);
  TensorArena::Bind(weight, 0x20000);
  TensorArena::Bind(matrix, 0x40000);

  ggml_tensor* norm = ggml_rms_norm(context, input, 1e-6f);
  ggml_tensor* scaled = ggml_mul(context, norm, weight);
  ggml_tensor* product = ggml_mul_mat(context, matrix, scaled);
  EXPECT_EQ(norm->op, GGML_OP_RMS_NORM);
  EXPECT_EQ(norm->src[0], input);
  float eps = 0.0f;
  std::memcpy(&eps, norm->op_params, sizeof(eps));
  EXPECT_EQ(eps, 1e-6f);
  EXPECT_EQ(scaled->src[1], weight);
  EXPECT_EQ(product->op, GGML_OP_MUL_MAT);
  EXPECT_EQ(product->ne[0], 4864);
  EXPECT_EQ(product->ne[1], 5);
  EXPECT_EQ(product->type, GGML_TYPE_F32);
  EXPECT_EQ(ggml_nbytes(matrix), std::size_t{896} * 4864 * 2);
  EXPECT_EQ(Address(matrix), 0x40000U);
  EXPECT_EQ(product->data, nullptr);  // bound by the caller, like any output
  TensorArena::Bind(product, 0x800000);
  EXPECT_EQ(Address(product), 0x800000U);
  EXPECT_EQ(product->buffer, nullptr);  // no GGML backend buffer
}

TEST(GgmlTensors, ViewsFollowTheirBoundSource) {
  auto arena = TensorArena::Create(4).value();
  ggml_tensor* rows = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, 64, 16);
  TensorArena::Bind(rows, 0x100000);
  ggml_tensor* tail = ggml_view_2d(arena.context(), rows, 64, 4, rows->nb[1], 12 * rows->nb[1]);
  EXPECT_EQ(Address(tail), 0x100000U + (12 * 64 * 4));
  EXPECT_EQ(tail->view_src, rows);
}

TEST(GgmlTensors, AnArenaRefusesMoreThanItHolds) {
  EXPECT_EQ(FailedCode(TensorArena::Create(0)), KernelError::kRejected);
  auto arena = TensorArena::Create(3).value();
  EXPECT_TRUE(arena.Reserve(3).has_value());
  EXPECT_EQ(FailedCode(arena.Reserve(4)), KernelError::kRejected);
  ggml_tensor* a = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, 8);
  ggml_tensor* b = ggml_new_tensor_1d(arena.context(), GGML_TYPE_F32, 8);
  EXPECT_TRUE(arena.Reserve(1).has_value());
  EXPECT_EQ(FailedCode(arena.Reserve(2)), KernelError::kRejected);
  (void)ggml_add(arena.context(), a, b);
  EXPECT_EQ(FailedCode(arena.Reserve(1)), KernelError::kRejected);
  EXPECT_TRUE(arena.Reserve(0).has_value());

  TensorArena moved = std::move(arena);
  EXPECT_EQ(FailedCode(moved.Reserve(1)), KernelError::kRejected);
  EXPECT_NE(moved.context(), nullptr);
}

TEST(GgmlTensors, TraversalScratchIsFundedSeparatelyFromMetadataAndSurvivesSeal) {
  const auto overhead = ggml_tensor_overhead();
  auto arena = TensorArena::CreateBytes(2 * overhead, 16 * overhead);
  ASSERT_TRUE(arena);
  const auto table_bytes = arena->graph_visited().size_bytes();
  EXPECT_EQ(arena->graph_capacity(), 16U);
  EXPECT_GE(arena->graph_visited().size(), 32U);
  EXPECT_EQ(arena->bytes(), 2 * overhead + table_bytes);
  EXPECT_TRUE(arena->Reserve(16));
  arena->Seal();
  EXPECT_EQ(arena->bytes(), 2 * overhead + table_bytes);
  EXPECT_EQ(arena->graph_capacity(), 16U);
  EXPECT_TRUE(arena->Reserve(2));
  EXPECT_FALSE(arena->Reserve(3));  // traversal bytes cannot fund GGML metadata
  auto larger_metadata = TensorArena::CreateBytes(3 * overhead, overhead);
  ASSERT_TRUE(larger_metadata);
  EXPECT_EQ(larger_metadata->graph_capacity(), 3U);
}

TEST(GgmlTensors, TraversalScratchMovesResetsAndRejectsOverflow) {
  auto arena = TensorArena::Create(3);
  ASSERT_TRUE(arena);
  ggml_tensor* leaf = ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 1);
  const std::array<ggml_tensor*, 1> outputs{leaf};
  ASSERT_TRUE(jitllm::kernels::ggml::GraphOrder(outputs, *arena));
  EXPECT_FALSE(std::ranges::all_of(arena->graph_visited(), [](auto* p) { return p == nullptr; }));
  arena->Reset();
  EXPECT_TRUE(std::ranges::all_of(arena->graph_visited(), [](auto* p) { return p == nullptr; }));
  const auto bytes = arena->bytes();
  TensorArena moved = std::move(*arena);
  EXPECT_EQ(moved.bytes(), bytes);
  EXPECT_EQ(moved.graph_capacity(), 3U);
  EXPECT_EQ(arena->bytes(), 0U);
  EXPECT_EQ(arena->graph_capacity(), 0U);
  EXPECT_EQ(arena->context(), nullptr);
  EXPECT_FALSE(jitllm::kernels::ggml::GraphOrder(outputs, *arena));
  auto destination = TensorArena::Create(1);
  ASSERT_TRUE(destination);
  *destination = std::move(moved);
  EXPECT_EQ(destination->bytes(), bytes);
  EXPECT_EQ(destination->graph_capacity(), 3U);
  EXPECT_EQ(moved.bytes(), 0U);
  EXPECT_EQ(moved.graph_capacity(), 0U);
  destination->Seal();
  EXPECT_EQ(destination->bytes(), bytes);
  EXPECT_FALSE(TensorArena::CreateBytes(std::numeric_limits<std::size_t>::max()));
  EXPECT_FALSE(
      TensorArena::CreateBytes(std::numeric_limits<std::size_t>::max() - ggml_tensor_overhead(),
                               std::numeric_limits<std::size_t>::max()));
}

}  // namespace
