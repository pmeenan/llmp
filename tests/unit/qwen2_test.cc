// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen2 architecture adapter (model/qwen2.h), in every profile
// (backend-proof P2):
// - binding a synthetic index's resources to the compiled-in profile, tied
//   output head included, and refusing a missing, mis-typed, mis-shaped or
//   unknown tensor, or another architecture;
// - llama.cpp's n_kv padding and the host-built inputs of a chunk for a
//   single-sequence cache (positions, K and V cells, mask, output rows);
// - the exact F16 to F32 widening of the embedding lookup.

#include "model/qwen2.h"

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using llmp::model::BindQwen2;
using llmp::model::EmbedRows;
using llmp::model::HalfToFloat;
using llmp::model::PaddedKv;
using llmp::model::Qwen25Instruct05B;
using llmp::model::Qwen2ChunkInputs;
using llmp::model::Qwen2Profile;
using llmp::model::ResourceShape;

// The FP16 fixture's index, as the artifact reader reports it: every
// tensor once, the output head as a tied alias of the token table.
std::vector<ResourceShape> FixtureIndex(const Qwen2Profile& p) {
  std::vector<ResourceShape> r;
  const std::uint64_t w = p.width;
  const std::uint64_t kv = p.kv_width();
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string b = "blk." + std::to_string(il) + ".";
    r.push_back({{b + "attn_norm.weight"}, "F32", {w}});
    r.push_back({{b + "ffn_down.weight"}, "F16", {p.ffn, w}});
    r.push_back({{b + "ffn_gate.weight"}, "F16", {w, p.ffn}});
    r.push_back({{b + "ffn_up.weight"}, "F16", {w, p.ffn}});
    r.push_back({{b + "ffn_norm.weight"}, "F32", {w}});
    r.push_back({{b + "attn_k.bias"}, "F32", {kv}});
    r.push_back({{b + "attn_k.weight"}, "F16", {w, kv}});
    r.push_back({{b + "attn_output.weight"}, "F16", {w, w}});
    r.push_back({{b + "attn_q.bias"}, "F32", {w}});
    r.push_back({{b + "attn_q.weight"}, "F16", {w, w}});
    r.push_back({{b + "attn_v.bias"}, "F32", {kv}});
    r.push_back({{b + "attn_v.weight"}, "F16", {w, kv}});
  }
  r.push_back({{"output_norm.weight"}, "F32", {w}});
  r.push_back({{"token_embd.weight", "output.weight"}, "F16", {w, p.vocab}});
  return r;
}

TEST(Qwen2Test, TheProfileIsQwen25Instruct05B) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  EXPECT_EQ(p.layers, 24U);
  EXPECT_EQ(p.width, 896U);
  EXPECT_EQ(p.heads, 14U);
  EXPECT_EQ(p.kv_heads, 2U);
  EXPECT_EQ(p.head_dim, 64U);
  EXPECT_EQ(p.kv_width(), 128U);
  EXPECT_EQ(p.ffn, 4864U);
  EXPECT_EQ(p.vocab, 151936U);
  EXPECT_EQ(p.train_context, 8192U);
  EXPECT_EQ(p.rms_eps, 1e-6f);
  EXPECT_EQ(p.rope_base, 1000000.0f);
  EXPECT_EQ(p.weight_type, "F16");
}

TEST(Qwen2Test, BindsTheFixtureIndexWithATiedHead) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  const auto index = FixtureIndex(p);
  ASSERT_EQ(index.size(), 290U);
  const auto binding = BindQwen2(p, "qwen2", index);
  ASSERT_TRUE(binding.has_value()) << binding.error();
  ASSERT_EQ(binding->layers.size(), 24U);
  EXPECT_EQ(binding->token_embd, 289U);
  EXPECT_EQ(binding->output, 289U);
  EXPECT_EQ(binding->output_norm, 288U);
  const auto& layer = binding->layers[3];
  EXPECT_EQ(index[layer.q].roles.front(), "blk.3.attn_q.weight");
  EXPECT_EQ(index[layer.k_bias].roles.front(), "blk.3.attn_k.bias");
  EXPECT_EQ(index[layer.down].roles.front(), "blk.3.ffn_down.weight");
  EXPECT_EQ(index[layer.out].roles.front(), "blk.3.attn_output.weight");
}

TEST(Qwen2Test, AnUntiedHeadBindsItsOwnResource) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  auto index = FixtureIndex(p);
  index.back().roles = {"token_embd.weight"};
  index.push_back({{"output.weight"}, "F16", {p.width, p.vocab}});
  const auto binding = BindQwen2(p, "qwen2", index);
  ASSERT_TRUE(binding.has_value()) << binding.error();
  EXPECT_EQ(binding->output, 290U);
  EXPECT_EQ(binding->token_embd, 289U);
}

TEST(Qwen2Test, RefusesWhatTheProfileDoesNotDescribe) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  {
    auto index = FixtureIndex(p);
    index.erase(index.begin() + 7);  // blk.0.attn_output.weight
    const auto binding = BindQwen2(p, "qwen2", index);
    ASSERT_FALSE(binding.has_value());
    EXPECT_EQ(binding.error(), "the artifact has no blk.0.attn_output.weight");
  }
  {
    auto index = FixtureIndex(p);
    index[9].ne = {896, 128};  // blk.0.attn_q.weight
    const auto binding = BindQwen2(p, "qwen2", index);
    ASSERT_FALSE(binding.has_value());
    EXPECT_EQ(binding.error(), "blk.0.attn_q.weight is F16 [896, 128], not F16 [896, 896]");
  }
  {
    auto index = FixtureIndex(p);
    index[0].type = "F16";  // blk.0.attn_norm.weight
    const auto binding = BindQwen2(p, "qwen2", index);
    ASSERT_FALSE(binding.has_value());
    EXPECT_EQ(binding.error(), "blk.0.attn_norm.weight is F16 [896], not F32 [896]");
  }
  {
    auto index = FixtureIndex(p);
    index.push_back({{"blk.24.attn_norm.weight"}, "F32", {896}});
    const auto binding = BindQwen2(p, "qwen2", index);
    ASSERT_FALSE(binding.has_value());
    EXPECT_EQ(binding.error(),
              "the artifact binds blk.24.attn_norm.weight, which Qwen2 does not read");
  }
  {
    const auto binding = BindQwen2(p, "llama", FixtureIndex(p));
    ASSERT_FALSE(binding.has_value());
    EXPECT_EQ(binding.error(), "the artifact's architecture is llama, not qwen2");
  }
  // A zero head count is refused, never divided by.
  for (const auto field : {&Qwen2Profile::heads, &Qwen2Profile::kv_heads}) {
    Qwen2Profile zero = p;
    zero.*field = 0;
    const auto binding = BindQwen2(zero, "qwen2", FixtureIndex(p));
    ASSERT_FALSE(binding.has_value());
    EXPECT_EQ(binding.error(), "the profile's head counts are not a Qwen2 model's");
  }
}

TEST(Qwen2Test, PadsTheAttendedCellsAsLlamaCpp) {
  EXPECT_EQ(PaddedKv(1, 512), 256U);
  EXPECT_EQ(PaddedKv(32, 512), 256U);
  EXPECT_EQ(PaddedKv(256, 512), 256U);
  EXPECT_EQ(PaddedKv(257, 512), 512U);
  EXPECT_EQ(PaddedKv(76, 512), 256U);
  EXPECT_EQ(PaddedKv(561, 1024), 768U);
  EXPECT_EQ(PaddedKv(577, 1024), 768U);
  EXPECT_EQ(PaddedKv(900, 1024), 1024U);
  EXPECT_EQ(PaddedKv(100, 128), 128U);  // never beyond the cache
}

TEST(Qwen2Test, BuildsAPrefillChunksInputs) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  const auto in = Qwen2ChunkInputs(p, 512, 0, 32);
  ASSERT_TRUE(in.has_value()) << in.error();
  EXPECT_EQ(in->n_kv, 256U);
  ASSERT_EQ(in->positions.size(), 32U);
  ASSERT_EQ(in->k_idxs.size(), 32U);
  ASSERT_EQ(in->v_idxs.size(), 32U * 128U);
  ASSERT_EQ(in->mask.size(), 32U * 256U);
  ASSERT_EQ(in->out_ids.size(), 32U);
  const float drop = -std::numeric_limits<float>::infinity();
  for (std::uint32_t i = 0; i < 32; ++i) {
    EXPECT_EQ(in->positions[i], static_cast<std::int32_t>(i));
    EXPECT_EQ(in->k_idxs[i], i);
    EXPECT_EQ(in->out_ids[i], static_cast<std::int32_t>(i));
    for (std::uint32_t j = 0; j < 128; ++j) {
      EXPECT_EQ(in->v_idxs[(i * 128) + j], (std::int64_t{j} * 512) + i);
    }
    for (std::uint32_t j = 0; j < 256; ++j) {
      const float m = in->mask[(i * 256) + j];
      if (j <= i) {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(m), 0U) << i << " " << j;
      } else {
        EXPECT_EQ(m, drop) << i << " " << j;
      }
    }
  }
}

TEST(Qwen2Test, BuildsADecodeStepsInputs) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  const auto in = Qwen2ChunkInputs(p, 1024, 561, 1);
  ASSERT_TRUE(in.has_value()) << in.error();
  EXPECT_EQ(in->n_kv, 768U);
  EXPECT_EQ(in->positions, std::vector<std::int32_t>{561});
  EXPECT_EQ(in->k_idxs, std::vector<std::int64_t>{561});
  EXPECT_EQ(in->v_idxs[5], (5 * 1024) + 561);
  for (std::uint32_t j = 0; j < 768; ++j) {
    EXPECT_EQ(std::isinf(in->mask[j]), j > 561) << j;
  }
}

TEST(Qwen2Test, RefusesAChunkTheCacheCannotHold) {
  const Qwen2Profile& p = Qwen25Instruct05B();
  EXPECT_FALSE(Qwen2ChunkInputs(p, 512, 0, 0).has_value());
  EXPECT_FALSE(Qwen2ChunkInputs(p, 512, 500, 13).has_value());
  EXPECT_FALSE(Qwen2ChunkInputs(p, 512, 513, 1).has_value());
  EXPECT_TRUE(Qwen2ChunkInputs(p, 512, 500, 12).has_value());
}

TEST(Qwen2Test, WidensHalvesExactly) {
  EXPECT_EQ(HalfToFloat(0x3C00), 1.0f);
  EXPECT_EQ(HalfToFloat(0xC000), -2.0f);
  EXPECT_EQ(HalfToFloat(0x7BFF), 65504.0f);
  EXPECT_EQ(HalfToFloat(0x0400), std::ldexp(1.0f, -14));
  EXPECT_EQ(HalfToFloat(0x0001), std::ldexp(1.0f, -24));
  EXPECT_EQ(HalfToFloat(0x03FF), std::ldexp(1023.0f, -24));
  EXPECT_EQ(HalfToFloat(0x3555), (0x155 * std::ldexp(1.0f, -12)) + 0.25f);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(HalfToFloat(0x8000)), 0x80000000U);
  EXPECT_EQ(HalfToFloat(0x7C00), std::numeric_limits<float>::infinity());
  EXPECT_EQ(HalfToFloat(0xFC00), -std::numeric_limits<float>::infinity());
  EXPECT_TRUE(std::isnan(HalfToFloat(0x7E00)));
  // Every finite half, in order, widens to a strictly increasing float.
  float previous = -std::numeric_limits<float>::infinity();
  for (std::uint32_t h = 0xFBFF; h >= 0x8001; --h) {
    const float f = HalfToFloat(static_cast<std::uint16_t>(h));
    EXPECT_LT(previous, f) << h;
    previous = f;
  }
  for (std::uint32_t h = 0; h <= 0x7BFF; ++h) {
    const float f = HalfToFloat(static_cast<std::uint16_t>(h));
    EXPECT_LE(previous, f) << h;  // -0 and +0 are equal
    previous = f;
  }
}

TEST(Qwen2Test, LooksUpEmbeddingRows) {
  // A table of 3 rows of 2 elements: 1, 2 | -1, 0.5 | 0, 65504.
  const std::vector<std::uint16_t> table = {0x3C00, 0x4000, 0xBC00, 0x3800, 0x0000, 0x7BFF};
  const std::vector<std::int32_t> tokens = {2, 0, 2};
  std::vector<float> out(6);
  ASSERT_TRUE(EmbedRows(table, 2, 3, tokens, out).has_value());
  EXPECT_EQ(out, (std::vector<float>{0.0f, 65504.0f, 1.0f, 2.0f, 0.0f, 65504.0f}));
  const std::vector<std::int32_t> outside = {3};
  std::vector<float> one(2);
  EXPECT_FALSE(EmbedRows(table, 2, 3, outside, one).has_value());
  EXPECT_FALSE(EmbedRows(table, 2, 3, tokens, one).has_value());
}

}  // namespace
