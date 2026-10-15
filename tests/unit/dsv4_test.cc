// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 adapter (model/dsv4.h), in every profile:
// - the binding of a synthetic resource list shaped and typed as the
//   UD-Q2_K_XL GGUF's, and its refusals, and the hash-routing tables' bound;
// - the state layout's sizes and bounds;
// - each compressor's chunk plan against llama.cpp's rules
//   (dsv4_build_comp_plan at b29c606e2), worked by hand for a prefill and
//   the decode steps around a block boundary, and the window's mask.

#include "model/dsv4.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "expected_error.h"

namespace {

namespace md = llmp::model;
using llmp::test_support::Failed;

// Why a result failed, read safely (D-083).
template <typename T>
std::string Why(const std::expected<T, std::string>& result) {
  return Failed(result).value_or(std::string());
}

// The UD-Q2_K_XL GGUF's tensors (0731 and e3aa0d6a alike), as the artifact
// lists them: every resource, then the expert arrays.
std::vector<md::Dsv4Resource> GgufLike(const md::Dsv4Profile& p) {
  std::vector<md::Dsv4Resource> r;
  const auto add = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back({.roles = {std::move(name)}, .type = std::move(type), .ne = std::move(ne)});
  };
  add("token_embd.weight", "Q5_K", {4096, 129280});
  add("output_norm.weight", "F32", {4096});
  add("output.weight", "Q4_K", {4096, 129280});
  add("output_hc_fn.weight", "F32", {16384, 4});
  add("output_hc_base.weight", "F32", {4});
  add("output_hc_scale.weight", "F32", {1});
  std::vector<md::Dsv4Resource> arrays;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    add(n + "attn_norm.weight", "F32", {4096});
    add(n + "attn_sinks.weight", "F32", {64});
    add(n + "attn_q_a.weight", "Q5_K", {4096, 1024});
    add(n + "attn_q_a_norm.weight", "F32", {1024});
    add(n + "attn_q_b.weight", "Q8_0", {1024, 32768});
    add(n + "attn_kv.weight", "Q8_0", {4096, 512});
    add(n + "attn_kv_a_norm.weight", "F32", {512});
    add(n + "attn_output_a.weight", "Q8_0", {4096, 8192});
    add(n + "attn_output_b.weight", "Q8_0", {8192, 4096});
    for (const char* hc : {"hc_attn_", "hc_ffn_"}) {
      add(n + hc + "fn.weight", "F32", {16384, 24});
      add(n + hc + "base.weight", "F32", {24});
      add(n + hc + "scale.weight", "F32", {3});
    }
    const std::uint32_t ratio = p.compress_ratios[il];
    if (ratio != 0) {
      const std::uint64_t width = ratio == 4 ? 1024 : 512;
      add(n + "attn_compressor_kv.weight", "Q8_0", {4096, width});
      add(n + "attn_compressor_gate.weight", "Q8_0", {4096, width});
      add(n + "attn_compressor_ape.weight", "F32", {width, ratio});
      add(n + "attn_compressor_norm.weight", "F32", {512});
    }
    if (ratio == 4) {
      add(n + "indexer.attn_q_b.weight", "Q8_0", {1024, 8192});
      add(n + "indexer.proj.weight", "F32", {4096, 64});
      add(n + "indexer_compressor_kv.weight", "Q8_0", {4096, 256});
      add(n + "indexer_compressor_gate.weight", "Q8_0", {4096, 256});
      add(n + "indexer_compressor_ape.weight", "F32", {256, 4});
      add(n + "indexer_compressor_norm.weight", "F32", {128});
    }
    add(n + "ffn_norm.weight", "F32", {4096});
    add(n + "ffn_gate_inp.weight", "BF16", {4096, 256});
    if (il < p.hash_layers) {
      add(n + "ffn_gate_tid2eid.weight", "I32", {6, 129280});
    } else {
      add(n + "exp_probs_b.bias", "F32", {256});
    }
    add(n + "ffn_gate_shexp.weight", "Q5_K", {4096, 2048});
    add(n + "ffn_up_shexp.weight", "Q5_K", {4096, 2048});
    add(n + "ffn_down_shexp.weight", "Q6_K", {2048, 4096});
    const bool mxfp4 = il == 42;
    for (const auto& [name, type, ne] :
         {std::tuple{"ffn_gate_exps.weight", "IQ2_XS", std::vector<std::uint64_t>{4096, 2048}},
          std::tuple{"ffn_up_exps.weight", "IQ2_XS", std::vector<std::uint64_t>{4096, 2048}},
          std::tuple{"ffn_down_exps.weight", mxfp4 ? "MXFP4" : "IQ3_XXS",
                     std::vector<std::uint64_t>{2048, 4096}}}) {
      arrays.push_back(
          {.roles = {n + name}, .type = type, .ne = ne, .expert_array = true, .count = 256});
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

TEST(Dsv4Test, TheProfileIsDeepSeekV4Flash) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  EXPECT_EQ(p.layers, 43U);
  EXPECT_EQ(p.hc_width(), 16384U);
  EXPECT_EQ(p.hc_mix(), 24U);
  // Window-only layers 0 and 1, then CSA and HCA alternating, the last CSA
  // (the GGUF's compress_ratios).
  EXPECT_EQ(p.compress_ratios[0], 0U);
  EXPECT_EQ(p.compress_ratios[1], 0U);
  EXPECT_EQ(p.compress_ratios[2], 4U);
  EXPECT_EQ(p.compress_ratios[3], 128U);
  EXPECT_EQ(p.compress_ratios[41], 128U);
  EXPECT_EQ(p.compress_ratios[42], 4U);
}

TEST(Dsv4Test, BindsTheGgufsTensorsAndRefusesWhatDiffers) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto bound = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  EXPECT_EQ(bound->layers.size(), 43U);
  EXPECT_TRUE(bound->layers[0].hash);
  EXPECT_TRUE(bound->layers[2].hash);
  EXPECT_FALSE(bound->layers[3].hash);
  EXPECT_EQ(bound->layers[0].tid2eid.type, "I32");
  EXPECT_EQ(bound->layers[3].router_bias.ne, (std::vector<std::uint64_t>{256}));
  EXPECT_EQ(bound->layers[42].down_exps.type, "MXFP4");
  EXPECT_EQ(bound->layers[3].down_exps.ne, (std::vector<std::uint64_t>{2048, 4096}));
  EXPECT_EQ(bound->token_embd.type, "Q5_K");

  EXPECT_FALSE(md::BindDsv4(p, "deepseek2", resources).has_value());
  // A missing tensor, a wrong shape, a wrong kind, an unread role, an
  // expert array of the wrong count and a dense tensor for an array.
  {
    auto r = resources;
    std::erase_if(
        r, [](const md::Dsv4Resource& x) { return x.roles[0] == "blk.7.attn_sinks.weight"; });
    const auto refused = md::BindDsv4(p, "deepseek4", r);
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(Why(refused).find("blk.7.attn_sinks.weight"), std::string::npos);
  }
  {
    auto r = resources;
    std::ranges::find_if(r, [](const md::Dsv4Resource& x) {
      return x.roles[0] == "blk.3.attn_kv.weight";
    })->ne = {4096, 1024};
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", r).has_value());
  }
  {
    auto r = resources;
    std::ranges::find_if(r, [](const md::Dsv4Resource& x) {
      return x.roles[0] == "blk.3.attn_norm.weight";
    })->type = "F16";
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", r).has_value());
  }
  {
    auto r = resources;
    r.push_back({.roles = {"blk.3.unexpected.weight"}, .type = "F32", .ne = {4096}});
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", r).has_value());
  }
  {
    auto r = resources;
    std::ranges::find_if(r, [](const md::Dsv4Resource& x) {
      return x.roles[0] == "blk.5.ffn_up_exps.weight";
    })->count = 128;
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", r).has_value());
  }
  {
    auto r = resources;
    std::ranges::find_if(r, [](const md::Dsv4Resource& x) {
      return x.roles[0] == "blk.5.ffn_up_exps.weight";
    })->expert_array = false;
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", r).has_value());
  }
  {
    auto r = resources;
    r.push_back(r.back());  // a role bound twice
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", r).has_value());
  }
}

TEST(Dsv4Test, BindsCommunityIq2XxsAndF16ApeWithoutChangingOtherFloatTypes) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  auto resources = GgufLike(p);
  for (auto& r : resources) {
    const std::string& role = r.roles[0];
    if (role.ends_with("compressor_ape.weight")) {
      r.type = "F16";
    } else if (role.ends_with("ffn_gate_exps.weight") || role.ends_with("ffn_up_exps.weight")) {
      r.type = "IQ2_XXS";
    } else if (role.ends_with("ffn_down_exps.weight")) {
      r.type = "Q2_K";
    }
  }
  auto bound = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(bound) << Why(bound);
  EXPECT_EQ(bound->layers[2].comp_ape.type, "F16");
  EXPECT_EQ(bound->layers[2].idx_comp_ape.type, "F16");
  EXPECT_EQ(bound->layers[2].gate_exps.type, "IQ2_XXS");
  EXPECT_EQ(bound->layers[2].down_exps.type, "Q2_K");
  for (const std::string_view unsupported : {"BF16", "Q8_0", "I32"}) {
    auto changed = resources;
    std::ranges::find_if(changed, [](const md::Dsv4Resource& r) {
      return r.roles[0] == "blk.2.attn_compressor_ape.weight";
    })->type = unsupported;
    EXPECT_FALSE(md::BindDsv4(p, "deepseek4", changed));
  }
}

TEST(Dsv4Test, GrowingStateCoversTheRingPaddedPrefixAndCompressedDummyCells) {
  auto state = md::Dsv4State(md::Dsv4Flash(), 262144, 2048, md::Dsv4Window::kRing);
  ASSERT_TRUE(state) << Why(state);
  auto ranges = md::Dsv4UsedState(*state, 1025);
  ASSERT_TRUE(ranges) << Why(ranges);
  const auto covered = [&](std::uint64_t offset, std::uint64_t bytes) {
    return std::ranges::any_of(*ranges, [&](const md::StateRange& r) {
      return offset >= r.offset && offset - r.offset <= r.bytes &&
             bytes <= r.bytes - (offset - r.offset);
    });
  };
  using K = md::Dsv4StateTensor::Kind;
  for (const auto& t : state->tensors) {
    const auto row = t.ne0 * (t.f16 ? 2 : 4);
    if (t.kind == K::kRawK) {
      EXPECT_TRUE(covered(t.offset, t.bytes));
    } else if (t.kind == K::kCsaK || t.kind == K::kLidK) {
      // ceil(1025/4)=257 cells; attention reads through padded cell 511.
      EXPECT_TRUE(covered(t.offset, 512 * row));
      EXPECT_FALSE(covered(t.offset + (512 * row), row));
      EXPECT_TRUE(covered(t.offset + ((t.ne1 - 1) * row), row));
    } else if (t.kind == K::kHcaK) {
      EXPECT_TRUE(covered(t.offset, 256 * row));
      EXPECT_TRUE(covered(t.offset + ((t.ne1 - 1) * row), row));
    } else {
      EXPECT_TRUE(covered(t.offset, t.bytes));
    }
  }
  EXPECT_FALSE(md::Dsv4UsedState(*state, 262145));
}

TEST(Dsv4Test, StateLayoutStopsAtTheTrainedContextCeiling) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  // This describes virtual state; constructing it backs no model memory.
  auto state = md::Dsv4State(p, 1048576, 2048, md::Dsv4Window::kRing);
  ASSERT_TRUE(state.has_value()) << Why(state);
  EXPECT_EQ(state->context, 1048576U);
  EXPECT_EQ(state->raw_cells, 2304U);
  EXPECT_FALSE(md::Dsv4State(p, 1048577, 2048, md::Dsv4Window::kRing).has_value());
  EXPECT_EQ(md::Dsv4MostRows(p, 1048576), 4029U);  // the fast plan's mask bound
  EXPECT_EQ(md::Dsv4MostRows(p, 1048577), 0U);
}

TEST(Dsv4Test, TheStateIsBoundedAndSizedAsLlamaCppSizesIt) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  auto s = md::Dsv4State(p, 4096, 512);
  ASSERT_TRUE(s.has_value()) << Why(s);
  // llama.cpp's default full-size SWA cache, a cell per position, and the
  // compressed caches' pad(ceil(ctx / ratio), 256).
  EXPECT_EQ(s->raw_cells, 4096U);
  EXPECT_EQ(s->csa_cells, 1024U);
  EXPECT_EQ(s->hca_cells, 256U);
  EXPECT_EQ(s->csa_state_rows, 8U);
  EXPECT_EQ(s->hca_state_rows, 128U);
  // 43 window caches, 21 CSA layers with their indexer, 20 HCA layers.
  EXPECT_EQ(s->tensors.size(), 43U + (21U * 6U) + (20U * 3U));
  std::uint64_t sum = 0;
  for (const auto& t : s->tensors) {
    EXPECT_EQ(t.offset % 256, 0U);
    EXPECT_EQ(t.bytes, t.ne0 * t.ne1 * (t.f16 ? 2U : 4U));
    sum += t.bytes;
  }
  EXPECT_GE(s->bytes, sum);
  EXPECT_LT(s->bytes, sum + (s->tensors.size() * 256));
  const auto reps = s->Representations();
  ASSERT_EQ(reps.size(), 3U);
  std::uint64_t charged = 0;
  for (const auto& r : reps) {
    EXPECT_TRUE(md::IsValid(r)) << r.name;
    EXPECT_FALSE(r.CanTruncate()) << r.name;
    charged += r.block_bytes.value();
  }
  EXPECT_EQ(charged, sum);
  EXPECT_GE(s->Find(2, md::Dsv4StateTensor::Kind::kLidK), 0);
  EXPECT_EQ(s->Find(3, md::Dsv4StateTensor::Kind::kLidK), -1);
  // A chunk must leave the window room in the ring; zero is refused.
  EXPECT_FALSE(md::Dsv4State(p, 4096, 0).has_value());
  EXPECT_FALSE(md::Dsv4State(p, 0, 1).has_value());
  // 256 cells: a 200-row chunk and the 128 positions before it do not fit.
  EXPECT_FALSE(md::Dsv4State(p, 256, 200).has_value());
  EXPECT_TRUE(md::Dsv4State(p, 256, 128).has_value());
  // Past I32 positions, where the padded cell counts would also wrap.
  EXPECT_FALSE(md::Dsv4State(p, 0xFFFFFF01U, 512).has_value());
  EXPECT_FALSE(md::Dsv4State(p, 0x7FFFFF01U, 512).has_value());
}

// The widest chunk the state admits, which the runtime's prefill chunk
// stays within: at the configuration's minimum context (512) the window
// leaves 384 rows, not 512; just below it (511) as much; and each is the
// edge (one more row is refused). Deeper (from about 29K positions) the
// attention mask binds first (TheWidestChunksMaskFitsTheAttentionKernel).
TEST(Dsv4Test, TheWidestChunkFitsTheWindow) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  for (const auto& [context, most] :
       {std::pair{512U, 384U}, std::pair{511U, 384U}, std::pair{513U, 513U},
        std::pair{8704U, 8576U}, std::pair{200U, 128U}}) {
    SCOPED_TRACE(context);
    EXPECT_EQ(md::Dsv4MostRows(p, context), most);
    EXPECT_TRUE(md::Dsv4State(p, context, most).has_value());
    EXPECT_FALSE(md::Dsv4State(p, context, most + 1).has_value());
  }
  EXPECT_EQ(md::Dsv4MostRows(p, 0), 0U);
  EXPECT_EQ(md::Dsv4MostRows(p, 0x7FFFFF01U), 0U);
}

TEST(Dsv4Test, HashRoutesMustNameAnExpert) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  std::vector<std::int32_t> table(std::size_t{p.experts_used} * p.vocab);
  for (std::size_t i = 0; i < table.size(); ++i) {
    table[i] = static_cast<std::int32_t>(i % p.experts);
  }
  EXPECT_TRUE(md::CheckDsv4HashRouting(p, table).has_value());
  // The kernels would read a weight slice past the slab, or before it.
  for (const std::int32_t bad : {static_cast<std::int32_t>(p.experts), -1}) {
    auto t = table;
    t[(7 * p.experts_used) + 2] = bad;
    const auto refused = md::CheckDsv4HashRouting(p, t);
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(Why(refused).find("token 7"), std::string::npos) << Why(refused);
  }
  // Not [experts_used, vocab].
  EXPECT_FALSE(md::CheckDsv4HashRouting(p, std::span(table).first(table.size() - 1)).has_value());
}

TEST(Dsv4Test, APrefillsCompressorPlanFollowsLlamaCpp) {
  // Ten tokens from position 0, CSA (ratio 4, overlapped, an 8-row ring).
  auto plan = md::Dsv4CompressorPlan(4, true, 8, 1024, 0, 10);
  ASSERT_TRUE(plan.has_value()) << Why(plan);
  EXPECT_EQ(plan->state_pos, (std::vector<std::int32_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 1}));
  EXPECT_EQ(plan->n_visible, (std::vector<std::int32_t>{0, 0, 0, 1, 1, 1, 1, 2, 2, 2}));
  // Blocks end at positions 3 and 7; ceil(10 / 4) = 3 blocks, so a dummy
  // written to the cache's last row.
  EXPECT_EQ(plan->write_idxs, (std::vector<std::int64_t>{0, 1, 1023}));
  EXPECT_EQ(plan->write_pos, (std::vector<std::int32_t>{0, 4, 0}));
  // Rows of [8 ring | 10 chunk | zero]: every block's previous window
  // (block 0's is before the start: the zero row, 18), then every block's
  // own; the dummy reads the first token.
  EXPECT_EQ(plan->read_idxs,
            (std::vector<std::int32_t>{18, 18, 18, 18, 8,  9,  10, 11, 8, 8, 8, 8,  //
                                       8,  9,  10, 11, 12, 13, 14, 15, 8, 8, 8, 8}));
  // The ring keeps each row's latest position: rows 0 and 1 take positions
  // 8 and 9.
  EXPECT_EQ(plan->persist_dst, (std::vector<std::int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(plan->persist_src, (std::vector<std::int32_t>{8, 9, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(plan->n_kv, 256U);

  // HCA (ratio 128, not overlapped, a 128-row ring): no block completes, so
  // one dummy that reads the first token 128 times.
  auto hca = md::Dsv4CompressorPlan(128, false, 128, 256, 0, 10);
  ASSERT_TRUE(hca.has_value()) << Why(hca);
  EXPECT_EQ(hca->write_idxs, (std::vector<std::int64_t>{255}));
  EXPECT_EQ(hca->read_idxs, std::vector<std::int32_t>(128, 128));
  EXPECT_EQ(hca->persist_src.size(), 10U);
}

TEST(Dsv4Test, DecodeStepsAroundABlockBoundaryReadTheRing) {
  // Position 10: no block ends; a dummy block.
  auto at10 = md::Dsv4CompressorPlan(4, true, 8, 1024, 10, 1);
  ASSERT_TRUE(at10.has_value()) << Why(at10);
  EXPECT_EQ(at10->write_idxs, (std::vector<std::int64_t>{1023}));
  EXPECT_EQ(at10->persist_dst, (std::vector<std::int32_t>{2}));
  EXPECT_EQ(at10->persist_src, (std::vector<std::int32_t>{0}));
  // Position 11 ends block 2 (positions 8-11): its previous window 4-7 and
  // positions 8-10 from the ring (rows 4-7, 0-2), 11 from the chunk (row 8).
  auto at11 = md::Dsv4CompressorPlan(4, true, 8, 1024, 11, 1);
  ASSERT_TRUE(at11.has_value()) << Why(at11);
  EXPECT_EQ(at11->write_idxs, (std::vector<std::int64_t>{2}));
  EXPECT_EQ(at11->write_pos, (std::vector<std::int32_t>{8}));
  EXPECT_EQ(at11->read_idxs, (std::vector<std::int32_t>{4, 5, 6, 7, 0, 1, 2, 8}));
  EXPECT_EQ(at11->n_visible, (std::vector<std::int32_t>{3}));
  // HCA's first block ends at 127.
  auto at127 = md::Dsv4CompressorPlan(128, false, 128, 256, 127, 1);
  ASSERT_TRUE(at127.has_value()) << Why(at127);
  EXPECT_EQ(at127->write_idxs, (std::vector<std::int64_t>{0}));
  std::vector<std::int32_t> reads(128);
  for (std::int32_t i = 0; i < 127; ++i) {
    reads[static_cast<std::size_t>(i)] = i;
  }
  reads[127] = 128;
  EXPECT_EQ(at127->read_idxs, reads);
  // Refused: no rows, a block past the cache.
  EXPECT_FALSE(md::Dsv4CompressorPlan(4, true, 8, 1024, 0, 0).has_value());
  EXPECT_FALSE(md::Dsv4CompressorPlan(4, true, 8, 2, 11, 1).has_value());
}

TEST(Dsv4Test, TheWindowMasksWhatIsOlderThanTheWindowOrLater) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  auto s = md::Dsv4State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  // A prefill of 3: causal.
  auto c = md::Dsv4Chunk(p, *s, 0, 3);
  ASSERT_TRUE(c.has_value()) << Why(c);
  EXPECT_EQ(c->raw_n_kv, 256U);
  EXPECT_EQ(c->raw_cells, (std::vector<std::int64_t>{0, 1, 2}));
  const auto open = [&](const md::Dsv4ChunkInputs& in, std::uint32_t row) {
    std::vector<std::uint32_t> cells;
    for (std::uint32_t j = 0; j < in.raw_n_kv; ++j) {
      if (in.raw_mask[(std::size_t{row} * in.raw_n_kv) + j] == md::kHalfZero) {
        cells.push_back(j);
      }
    }
    return cells;
  };
  EXPECT_EQ(open(*c, 0), (std::vector<std::uint32_t>{0}));
  EXPECT_EQ(open(*c, 2), (std::vector<std::uint32_t>{0, 1, 2}));
  // A decode step at 900: attention reads pad(901, 256) cells, the window
  // positions 773-900.
  auto d = md::Dsv4Chunk(p, *s, 900, 1);
  ASSERT_TRUE(d.has_value()) << Why(d);
  EXPECT_EQ(d->raw_n_kv, 1024U);
  EXPECT_EQ(d->raw_cells, (std::vector<std::int64_t>{900}));
  const auto cells = open(*d, 0);
  ASSERT_EQ(cells.size(), 128U);
  EXPECT_EQ(cells.front(), 773U);
  EXPECT_EQ(cells.back(), 900U);
  // A 512-position context: at its end attention reads all 512 cells, the
  // window positions 373-500.
  auto small = md::Dsv4State(p, 512, 128);
  ASSERT_TRUE(small.has_value());
  auto late = md::Dsv4Chunk(p, *small, 500, 1);
  ASSERT_TRUE(late.has_value()) << Why(late);
  EXPECT_EQ(late->raw_n_kv, 512U);
  const auto recent = open(*late, 0);
  ASSERT_EQ(recent.size(), 128U);
  EXPECT_EQ(recent.front(), 373U);
  EXPECT_EQ(recent.back(), 500U);
  // CSA sees (900 + 1) / 4 = 225 rows.
  EXPECT_EQ(d->csa.n_visible, (std::vector<std::int32_t>{225}));
  EXPECT_EQ(d->csa_mask[224], md::kHalfZero);
  EXPECT_EQ(d->csa_mask[225], md::kHalfNegInf);
  // Past the context or the chunk bound.
  EXPECT_FALSE(md::Dsv4Chunk(p, *s, 4095, 2).has_value());
  EXPECT_FALSE(md::Dsv4Chunk(p, *s, 0, 513).has_value());
}

TEST(Dsv4Test, CompressorGeometryRefusesWhatIsNotBoundedByItsCache) {
  EXPECT_FALSE(md::Dsv4CompressorPlan(0, false, 256, 256, 0, 1));
  EXPECT_FALSE(md::Dsv4CompressorPlan(32, true, 63, 256, 0, 1));
  EXPECT_FALSE(md::Dsv4CompressorPlan(32, true, 64, 256, 8192, 32));
}

}  // namespace
