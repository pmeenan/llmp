// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma2_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "expected_error.h"
#include "gemma2_fixture.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/shared_q8.h"
#include "kernels/ggml/validate_ext.h"

namespace {
namespace md = llmp::model;
namespace kg = llmp::kernels::ggml;
struct Case {
  const md::Gemma2Profile& p = md::Gemma2_2B();
  md::Gemma2Binding binding = *md::BindGemma2(p, "gemma2", llmp::test_support::gemma2::Resources());
  md::Gemma2StateLayout state = *md::Gemma2State(p, 8192, 16);
  kg::Gemma2ChunkShape shape{{{3, 3, 4351, 4608, 4352}, {1, 1, 7, 256, 256}}, 2};
};
std::size_t Count(const kg::Gemma2Graph& graph, ggml_op op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(graph.nodes, [op](const auto* t) { return t->op == op; }));
}

TEST(Gemma2Graph, SharedPreparationAuthenticatesExactInputLeafAndCurrentDenseConsumer) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  const auto place = [](ggml_tensor* t, std::uintptr_t address) {
    kg::TensorArena::Bind(t, address);
    return t;
  };
  auto* x = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, 2304, 2), 0x10000000ULL);
  auto* same_shape = place(ggml_new_tensor_2d(c, GGML_TYPE_F32, 2304, 2), 0x10000000ULL);
  auto* w = place(ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, 2304, 64), 0x20000000ULL);
  kg::MarkRowPaddingReadable(w);
  kg::SharedQ8Inputs cache(c);
  auto* q8 = place(cache.Get(x), 0x30000000ULL);
  EXPECT_EQ(cache.Get(x), q8);
  EXPECT_NE(cache.Get(same_shape), q8);
  auto* product =
      place(cache.Product(w, x, true, [](ggml_type, std::int64_t columns) { return columns <= 8; }),
            0x40000000ULL);
  ASSERT_TRUE(kg::CheckMmvqPrepared(product));
  const auto before = *product;
  for (const auto invalid : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}) {
    const std::int32_t bad = 1, zero = 0;
    auto* parameter = reinterpret_cast<std::byte*>(product->op_params) + 32 + invalid * 4;
    std::memcpy(parameter, &bad, 4);
    EXPECT_FALSE(kg::CheckMmvqPrepared(product));
    std::memcpy(parameter, &zero, 4);
  }
  auto malformed = before;
  malformed.src[2] = same_shape;
  EXPECT_FALSE(kg::CheckMmvqPrepared(&malformed));
  malformed = before;
  malformed.src[7] = x;
  EXPECT_FALSE(kg::CheckMmvqPrepared(&malformed));
  malformed = before;
  malformed.data = q8->data;
  EXPECT_FALSE(kg::CheckMmvqPrepared(&malformed));
  malformed = before;
  malformed.data = static_cast<std::byte*>(w->data) + ggml_nbytes(w);
  EXPECT_FALSE(kg::CheckMmvqPrepared(&malformed));
  auto old_weight = *w;
  w->src[7] = x;
  EXPECT_FALSE(kg::CheckMmvqPrepared(product));
  *w = old_weight;
  const auto old_preparation = *q8;
  for (const auto bad_stride : {std::size_t{0}, std::size_t{2}}) {
    q8->nb[0] = bad_stride;
    EXPECT_FALSE(kg::CheckQuantizeQ8(q8));
    EXPECT_FALSE(kg::CheckMmvqPrepared(product));
  }
  *q8 = old_preparation;
  q8->data = static_cast<std::byte*>(w->data) + ggml_nbytes(w);
  EXPECT_FALSE(kg::CheckMmvqPrepared(product));  // readable final-row tail
  *q8 = old_preparation;
  ASSERT_TRUE(kg::CheckMmvqPrepared(product));
  auto* wider = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2304, 9);
  EXPECT_EQ(cache.Product(w, wider, true, [](ggml_type, std::int64_t) { return true; })->op,
            GGML_OP_MUL_MAT);
  auto* wider_q8 = place(cache.Get(wider), 0x50000000ULL);
  place(wider, 0x60000000ULL);
  auto* wider_product = place(kg::MmvqPrepared(c, w, wider_q8, wider), 0x70000000ULL);
  EXPECT_FALSE(kg::CheckMmvqPrepared(wider_product));
}

TEST(Gemma2Graph, SharedPreparationModelsPerFormatDeviceCapsBeforeCreatingNodes) {
  auto arena = kg::TensorArena::Create(512);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  kg::SharedQ8Inputs cache(c);
  for (const auto type :
       {GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL}) {
    auto* weight = ggml_new_tensor_2d(c, type, 2304, 64);
    for (const std::int64_t columns : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
      SCOPED_TRACE(std::string(ggml_type_name(type)) + "/" + std::to_string(columns));
      auto* input = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2304, columns);
      EXPECT_EQ(cache.Product(weight, input, true)->op, GGML_OP_MUL_MAT);
      EXPECT_EQ(
          cache.Product(weight, input, true, [](ggml_type, std::int64_t) { return false; })->op,
          GGML_OP_MUL_MAT);
      // Model the pinned generic Blackwell caps. DGX Spark is separately
      // exercised against its actual original selector in the GPU fixture.
      const auto cap = type == GGML_TYPE_Q4_K   ? 5
                       : type == GGML_TYPE_Q5_K ? 6
                       : type == GGML_TYPE_Q6_K ? 7
                                                : 8;
      int calls = 0;
      const auto selector = [&](ggml_type actual_type, std::int64_t actual_columns) {
        ++calls;
        EXPECT_EQ(actual_type, type);
        EXPECT_EQ(actual_columns, columns);
        return actual_columns <= cap;
      };
      auto* product = cache.Product(weight, input, true, selector);
      EXPECT_EQ(kg::LlmpOpOf(product),
                columns <= cap ? kg::LlmpOp::kMmvqPrepared : kg::LlmpOp::kNone);
      EXPECT_EQ(calls, columns <= 8 ? 1 : 0);
      if (columns <= cap) EXPECT_EQ(product->src[1], cache.Get(input));
    }
  }
}

TEST(Gemma2Graph, SharedQ8UsesOriginalDeviceSelectorThroughEightAndPreservesFallback) {
  for (const std::uint32_t columns : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U}) {
    for (const bool enabled : {false, true}) {
      for (const int selector_mode : {0, 1, 2}) {
        SCOPED_TRACE(std::to_string(columns) + "/" + std::to_string(enabled) + "/" +
                     std::to_string(selector_mode));
        Case c;
        c.shape = {{{0, 1, 1, 256, 256}}, 1};
        if (columns == 2) {
          c.shape.segments.push_back({1, 1, 7, 256, 256});
          c.shape.outputs = 2;
        } else {
          c.shape.segments[0].rows = columns;
        }
        auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, c.shape.segments.size()));
        ASSERT_TRUE(arena);
        kg::Gemma2GraphOptions options;
        options.owner_decode = columns == 2;
        options.shared_q8 = enabled;
        std::function<bool(ggml_type, std::int64_t)> selector;
        if (selector_mode != 0)
          selector = [selector_mode](ggml_type, std::int64_t n) {
            return selector_mode == 2 && n <= 8;
          };
        auto graph =
            kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, options, selector);
        ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
        std::size_t products = 0, prepared = 0, owners = 0, glus = 0;
        for (const auto* t : graph->nodes) {
          const auto op = kg::LlmpOpOf(t);
          if (op == kg::LlmpOp::kMmvqPrepared) {
            ++products;
            ASSERT_EQ(t->src[2], t->src[1]->src[0]);
            EXPECT_EQ(t->ne[1], columns);
            EXPECT_EQ(t->src[2]->ne[1], columns);
          }
          prepared += op == kg::LlmpOp::kQuantizeQ8;
          owners += op == kg::LlmpOp::kFlashAttnOwners;
          if (t->op == GGML_OP_GLU) {
            ++glus;
            for (const auto* source : {t->src[0], t->src[1]})
              EXPECT_EQ(kg::LlmpOpOf(source),
                        enabled && selector_mode == 2 && columns > 1 && columns <= 8
                            ? kg::LlmpOp::kMmvqPrepared
                            : kg::LlmpOp::kNone);
          }
        }
        const bool selected = enabled && selector_mode == 2 && columns <= 8;
        EXPECT_EQ(prepared, selected ? c.p.layers * (columns > 1 ? 2U : 1U) : 0U);
        EXPECT_EQ(products, selected ? c.p.layers * (columns > 1 ? 5U : 3U) : 0U);
        EXPECT_EQ(glus, c.p.layers);
        EXPECT_EQ(owners, columns == 2 ? c.p.layers : 0U);
        // Head and both single-consumer projections keep their ordinary route.
        for (std::uint32_t layer = 0; layer < c.p.layers; ++layer) {
          auto* projection = graph->Named("blk." + std::to_string(layer) + ".attn_projection");
          ASSERT_NE(projection, nullptr);
          EXPECT_EQ(projection->op, GGML_OP_MUL_MAT);
        }
      }
    }
  }
}

TEST(Gemma2Graph, SoftcapOwnerContractIsClosedToD256H8C2AndCap50) {
  const auto make = [](kg::TensorArena& arena, std::int64_t d, std::int64_t heads,
                       std::uint32_t owners) {
    auto* c = arena.context();
    std::uintptr_t address = std::uintptr_t{1} << 40U;
    const auto placed = [&](ggml_tensor* tensor) {
      kg::TensorArena::Bind(tensor, address);
      address += 64U << 20U;
      return tensor;
    };
    auto* raw_q = placed(ggml_new_tensor_4d(c, GGML_TYPE_F32, d, heads, 1, owners));
    auto* q = ggml_permute(c, raw_q, 0, 2, 1, 3);
    kg::TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
    kg::FlashAttnOwners in{
        .q = q,
        .mask = placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, 512, 32, 1, owners)),
        .output = placed(ggml_new_tensor_4d(c, GGML_TYPE_F32, d, heads, 1, owners)),
        .logical_cohort = owners,
        .owner_count = owners};
    for (std::uint32_t i = 0; i < owners; ++i) {
      for (auto* destination : {&in.k[i], &in.v[i]}) {
        auto* raw =
            placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, d, heads / (d == 256 ? 2 : 8), 512, 1));
        auto* view = ggml_permute(c, raw, 0, 2, 1, 3);
        kg::TensorArena::Bind(view, reinterpret_cast<std::uintptr_t>(raw->data));
        *destination = view;
      }
    }
    return in;
  };
  for (const auto owners : {2U, 3U, 4U}) {
    auto arena = kg::TensorArena::Create(128);
    ASSERT_TRUE(arena);
    auto in = make(*arena, 256, 8, owners);
    in.logit_softcap = 50;
    EXPECT_EQ(kg::CheckFlashAttnOwners(in).has_value(), owners == 2);
    for (const auto cap : {1U, 25U, 51U, UINT32_MAX}) {
      in.logit_softcap = cap;
      EXPECT_FALSE(kg::CheckFlashAttnOwners(in));
    }
  }
  auto arena = kg::TensorArena::Create(128);
  ASSERT_TRUE(arena);
  EXPECT_FALSE(kg::CheckFlashAttnOwners(make(*arena, 512, 8, 2)));
  auto wrong_heads = make(*arena, 256, 16, 2);
  EXPECT_TRUE(kg::CheckFlashAttnOwners(wrong_heads));
  wrong_heads.logit_softcap = 50;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(wrong_heads));
  auto wrong_dim = make(*arena, 512, 16, 2);
  EXPECT_TRUE(kg::CheckFlashAttnOwners(wrong_dim));
  wrong_dim.logit_softcap = 50;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(wrong_dim));
}

TEST(Gemma2Graph, BoundedOwnerRootsRejectMalformedActualViewsAndMetadata) {
  auto arena = kg::TensorArena::Create(96);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  std::uintptr_t address = std::uintptr_t{1} << 40U;
  const auto placed = [&](ggml_tensor* tensor) {
    kg::TensorArena::Bind(tensor, address);
    address += 64U << 20U;
    return tensor;
  };
  auto* raw_q = placed(ggml_new_tensor_4d(c, GGML_TYPE_F32, 256, 8, 1, 2));
  auto* q = ggml_permute(c, raw_q, 0, 2, 1, 3);
  kg::TensorArena::Bind(q, reinterpret_cast<std::uintptr_t>(raw_q->data));
  auto* mask = placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, 1024, 32, 1, 2));
  std::array<ggml_tensor*, 4> k{}, v{};
  for (std::size_t owner = 0; owner < 2; ++owner)
    for (auto* target : {&k[owner], &v[owner]}) {
      auto* raw = placed(ggml_new_tensor_4d(c, GGML_TYPE_F16, 256, 4, owner ? 1024 : 512, 1));
      *target = ggml_permute(c, raw, 0, 2, 1, 3);
      kg::TensorArena::Bind(*target, reinterpret_cast<std::uintptr_t>(raw->data));
    }
  auto* node = placed(kg::FlashAttnOwnersNode(c, q, mask, k, v, 2, 2, 0, 50, true));
  auto in = kg::FlashAttnOwnersFromNode(node);
  ASSERT_TRUE(in);
  EXPECT_TRUE(in->bounded_roots);
  const auto check = [&] { return kg::CheckFlashAttnOwnersNode(node).has_value(); };
  EXPECT_TRUE(check());
  const auto original = *k[0];
  k[0]->ne[1] = 511;
  EXPECT_FALSE(check());
  *k[0] = original;
  k[0]->nb[2] *= 2;
  EXPECT_FALSE(check());
  *k[0] = original;
  v[0]->ne[1] = 1024;
  EXPECT_FALSE(check());
  v[0]->ne[1] = 512;
  auto bad = *in;
  bad.bounded_roots = false;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));
  bad = *in;
  bad.logit_softcap = 25;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));
  bad = *in;
  bad.k[1] = bad.k[0];
  bad.v[1] = bad.v[0];
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));  // no real root covers logical1024
  bad = *in;
  bad.owner_count = bad.logical_cohort = 3;
  EXPECT_FALSE(kg::CheckFlashAttnOwners(bad));
  EXPECT_TRUE(check());
  Case g;
  EXPECT_FALSE(kg::CheckGemma2Graph(g.p, g.binding, g.state, g.shape, {.bounded_roots = true}));
}

TEST(Gemma2Graph, ExplicitC2OwnerDecodeViewsQAndJoinsMasksWithoutJoiningCacheRoots) {
  Case c;
  c.shape = {{{3, 1, 259, 512, 512}, {1, 1, 259, 512, 512}}, 2};
  kg::Gemma2GraphOptions options{.narrow_final = true, .owner_decode = true};
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph) << *llmp::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 0);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
  std::size_t owners = 0;
  for (const auto* node : graph->nodes) {
    if (kg::LlmpOpOf(node) != kg::LlmpOp::kFlashAttnOwners) continue;
    ++owners;
    EXPECT_EQ(node->src[0]->op, GGML_OP_PERMUTE);
    EXPECT_EQ(node->src[0]->ne[0], 256);
    EXPECT_EQ(node->src[0]->ne[2], 8);
    EXPECT_EQ(node->src[0]->ne[3], 2);
    EXPECT_EQ(node->src[1]->op, GGML_OP_CONCAT);
    EXPECT_NE(node->src[2]->view_src, node->src[3]->view_src);
    EXPECT_NE(node->src[6]->view_src, node->src[7]->view_src);
    EXPECT_EQ(kg::LlmpOpInt(node, 0), 2);
    EXPECT_EQ(kg::LlmpOpInt(node, 1), 2);
    EXPECT_EQ(kg::LlmpOpInt(node, 3), 50);
  }
  EXPECT_EQ(owners, 26);
  options.first_layer = 1;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, c.shape, options));
  options.first_layer = 0;
  c.shape.segments[0].rows = 2;
  auto ordinary_arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(ordinary_arena);
  const auto ordinary =
      kg::BuildGemma2Graph(*ordinary_arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(ordinary);
  EXPECT_EQ(Count(*ordinary, GGML_OP_FLASH_ATTN_EXT), 52);
}

TEST(Gemma2Graph, ActualArithmeticHasNoQkNormAndPreservesBothSoftcapsAndReadableTails) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph) << *llmp::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->weights.size(), 288);
  EXPECT_EQ(graph->logits->ne[0], c.p.vocab);
  EXPECT_EQ(graph->logits->ne[1], 2);
  EXPECT_EQ(graph->hidden->ne[1], 4);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 52);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
  EXPECT_EQ(Count(*graph, GGML_OP_GLU), 26);
  EXPECT_EQ(Count(*graph, GGML_OP_SOFT_MAX), 0);
  for (std::uint32_t il = 0; il < c.p.layers; ++il) {
    const auto prefix = "blk." + std::to_string(il) + '.';
    auto* q = graph->Named(prefix + "q_rope");
    auto* scaled = graph->Named(prefix + "q_scaled");
    auto* k = graph->Named(prefix + "k_rope");
    auto* v = graph->Named(prefix + "v_raw");
    ASSERT_NE(q, nullptr);
    ASSERT_NE(scaled, nullptr);
    ASSERT_NE(k, nullptr);
    ASSERT_NE(v, nullptr);
    EXPECT_EQ(scaled->src[0], q);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(scaled->op_params[0]), 0.0625F);
    for (const auto* rope : {q, k}) {
      EXPECT_EQ(rope->op, GGML_OP_ROPE);
      EXPECT_EQ(rope->op_params[1], 256);
      EXPECT_EQ(rope->op_params[2], GGML_ROPE_TYPE_NEOX);
      EXPECT_FLOAT_EQ(std::bit_cast<float>(rope->op_params[5]), 10000.0F);
      EXPECT_FLOAT_EQ(std::bit_cast<float>(rope->op_params[6]), 1.0F);
      EXPECT_EQ(rope->src[2], nullptr);
      ASSERT_EQ(rope->src[0]->op, GGML_OP_RESHAPE);
      EXPECT_EQ(rope->src[0]->src[0]->op, GGML_OP_MUL_MAT);
    }
    ASSERT_EQ(v->op, GGML_OP_RESHAPE);
    EXPECT_EQ(v->src[0]->op, GGML_OP_MUL_MAT);
    EXPECT_NE(graph->segments[0].caches[il].first, graph->segments[0].caches[il].second);
    auto* attention = graph->Named(prefix + "slot.3.attention")->src[0];
    ASSERT_EQ(attention->op, GGML_OP_FLASH_ATTN_EXT);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[0]), 1.0F);
    EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[2]), 50.0F);
    EXPECT_EQ(attention->src[0]->ne[2], 8);
    EXPECT_EQ(attention->src[1]->ne[2], 4);
    EXPECT_EQ(attention->src[3],
              c.p.local(il) ? graph->segments[0].local_mask : graph->segments[0].global_mask);
  }
  EXPECT_EQ(graph->Named("blk.25.k_rope")->op, GGML_OP_ROPE);
  ASSERT_EQ(graph->logits->op, GGML_OP_SCALE);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(graph->logits->op_params[0]), 30.0F);
  ASSERT_EQ(graph->logits->src[0]->op, GGML_OP_UNARY);
  EXPECT_EQ(ggml_get_unary_op(graph->logits->src[0]), GGML_UNARY_OP_TANH);
  ASSERT_EQ(graph->logits->src[0]->src[0]->op, GGML_OP_SCALE);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(graph->logits->src[0]->src[0]->op_params[0]), 1.0F / 30.0F);
  std::size_t padded = 0;
  for (const auto& leaf : graph->weights) {
    if (ggml_is_quantized(leaf.tensor->type) && leaf.resource.ne[0] % 512 != 0) {
      ++padded;
      EXPECT_TRUE(kg::RowPaddingReadable(leaf.tensor));
      EXPECT_EQ(leaf.resource.readable, static_cast<std::uint64_t>(ggml_nbytes(leaf.tensor)) + 272);
    }
  }
  EXPECT_EQ(padded, 131);
}

TEST(Gemma2Graph, StateOnlyRetainsAllWritesAndNarrowingKeepsFullFinalAttention) {
  Case c;
  c.shape.outputs = 0;
  c.shape.output_mode = kg::Gemma2OutputMode::kStateOnly;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph) << *llmp::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->hidden, nullptr);
  EXPECT_EQ(graph->logits, nullptr);
  EXPECT_EQ(graph->Named("blk.25.q_rope"), nullptr);
  EXPECT_EQ(graph->Named("blk.25.attn_projection"), nullptr);
  EXPECT_NE(graph->Named("blk.25.k_rope"), nullptr);
  EXPECT_NE(graph->Named("blk.25.v_raw"), nullptr);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
  EXPECT_EQ(Count(*graph, GGML_OP_FLASH_ATTN_EXT), 50);
  arena->Reset();
  c.shape.outputs = 2;
  c.shape.output_mode = kg::Gemma2OutputMode::kHead;
  graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, {.narrow_final = true});
  ASSERT_TRUE(graph) << *llmp::test_support::Failed(graph, &kg::KernelFailure::detail);
  EXPECT_EQ(graph->Named("blk.25.q_rope")->ne[2], 4);
  EXPECT_EQ(graph->Named("blk.25.attn_projection")->ne[1], 4);
  EXPECT_EQ(graph->Named("blk.25.output")->ne[1], 2);
  EXPECT_EQ(graph->hidden->ne[1], 2);
  EXPECT_EQ(Count(*graph, GGML_OP_SET_ROWS), 104);
}

TEST(Gemma2Graph, GreedyFrontiersHaveDistinctShapeAndKeptArgmaxDescriptor) {
  Case c;
  const auto full_head = c.shape;
  c.shape.greedy = true;
  EXPECT_NE(c.shape, full_head);
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  ASSERT_NE(graph->greedy, nullptr);
  EXPECT_EQ(graph->greedy->type, GGML_TYPE_I32);
  EXPECT_EQ(graph->greedy->ne[0], 2);
  EXPECT_EQ(graph->greedy->src[0], graph->logits);
  EXPECT_EQ(kg::LlmpOpOf(graph->greedy), kg::LlmpOp::kArgmax);
  EXPECT_EQ(kg::LlmpOpInt(graph->greedy, 1),
            static_cast<std::int32_t>(kg::ArgmaxFlavor::kHostGreedy));
  EXPECT_EQ(graph->nodes.back(), graph->greedy);
  auto malformed = c.shape;
  malformed.outputs = 4;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, malformed));
  malformed.outputs = 0;
  malformed.output_mode = kg::Gemma2OutputMode::kStateOnly;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, malformed));
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, c.shape, {.head = false}));
}

TEST(Gemma2Graph, InvalidDescriptorsScopesAndCapacityRefuseBeforeGraphConstruction) {
  Case c;
  const auto rejects = [&](const auto& shape, const kg::Gemma2GraphOptions& options = {}) {
    EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, shape, options));
  };
  auto changed = c.shape;
  changed.segments[1].slot = changed.segments[0].slot;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].global_n_kv = 256;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].local_n_kv = 4353;
  rejects(changed);
  changed = c.shape;
  changed.segments[0].n_past = 8191;
  rejects(changed);
  rejects(c.shape, {.first_layer = 5});
  rejects(c.shape, {.layer_count = 1, .narrow_final = true});
  changed = c.shape;
  changed.outputs = 0;
  changed.output_mode = kg::Gemma2OutputMode::kStateOnly;
  rejects(changed, {.hidden_input = true});
  auto bad_binding = c.binding;
  bad_binding.layers[0].v.index = bad_binding.layers[0].k.index;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, bad_binding, c.state, c.shape));
  auto bad_layout = c.state;
  --bad_layout.tensors[0].bytes;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, bad_layout, c.shape));
  auto small = kg::TensorArena::Create(1);
  ASSERT_TRUE(small);
  const auto before = small->used();
  EXPECT_FALSE(kg::BuildGemma2Graph(*small, c.p, c.binding, c.state, c.shape));
  EXPECT_EQ(small->used(), before);
  auto advanced = c.shape;
  ++advanced.segments[0].n_past;
  EXPECT_EQ(advanced, c.shape);  // Positions are fresh inputs at the same read widths.
  advanced.segments[0].global_n_kv += 256;
  EXPECT_NE(advanced, c.shape);
}
}  // namespace
