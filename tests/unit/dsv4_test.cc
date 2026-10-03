// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 adapter (model/dsv4.h) and its chunk graph
// (kernels/ggml/dsv4_graph.h), in every profile:
// - the binding of a synthetic resource list shaped and typed as the
//   UD-Q2_K_XL GGUF's, and its refusals, and the hash-routing tables' bound;
// - the state layout's sizes and bounds;
// - each compressor's chunk plan against llama.cpp's rules
//   (dsv4_build_comp_plan at b29c606e2), worked by hand for a prefill and
//   the decode steps around a block boundary, and the window's mask;
// - the graph at prefill and decode shapes: every node planned by an
//   implementation of this module (a model of the device's choices), the
//   activations placed, and the routed experts' stride taken only in whole
//   blocks.

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
#include <utility>
#include <vector>

#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace {

namespace md = jitllm::model;
namespace kg = jitllm::kernels::ggml;
using jitllm::test_support::Failed;

// Why a result failed, read safely (D-083).
template <typename T>
std::string Why(const std::expected<T, std::string>& result) {
  return Failed(result).value_or(std::string());
}
template <typename T>
std::string Why(const std::expected<T, kg::KernelFailure>& result) {
  return Failed(result, &kg::KernelFailure::detail).value_or(std::string());
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

// Upstream's routing on a GB10, as a model: MMVQ up to 8 columns, MMQ
// beyond; float products MMVF for one column, MMF to 16, cuBLAS beyond.
kg::DeviceChoices ModelDevice() {
  return {
      .mul_mat = [](const ggml_tensor* node) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
        const std::int64_t columns = node->src[1]->ne[1] * node->src[1]->ne[2];
        if (columns == 1) {
          return kg::MulMatPath::kVector;
        }
        return columns <= 16 ? kg::MulMatPath::kTensorCore : kg::MulMatPath::kCublas;
      },
      .vector_fusible = [](const ggml_tensor*) { return false; },
      .quant =
          [](const ggml_tensor* node) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
        const std::int64_t columns =
            node->op == GGML_OP_MUL_MAT_ID ? node->src[2]->ne[1] : node->src[1]->ne[1];
        return columns <= 8 ? kg::QuantMulMatPath::kVector : kg::QuantMulMatPath::kTile;
      }};
}

TEST(Dsv4Test, TheChunkGraphIsPlannedByThisModulesImplementations) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Dsv4State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  // The slab strides: whole IQ2_XS (74) and IQ3_XXS (98) blocks and 16 bytes.
  std::vector<std::uint64_t> strides(p.layers, 8064224);
  strides[42] = 9309200;  // IQ2_XS with MXFP4 (17): 925 x 10,064
  for (const auto& [n_past, rows] : {std::pair{0U, 37U}, {37U, 1U}, {3000U, 512U}, {4095U, 1U}}) {
    auto chunk = md::Dsv4Chunk(p, *state, n_past, rows);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    const kg::Dsv4ChunkShape shape = kg::Dsv4ShapeOf(*state, *chunk);
    auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto graph = kg::BuildDsv4Graph(*arena, p, *binding, shape, {.expert_stride = strides});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    // Leaves at distinct addresses, every computed node too, then the plan.
    std::uint64_t next = std::uint64_t{1} << 40U;
    const auto bind_leaf = [&](ggml_tensor* t) {
      if (t != nullptr && t->data == nullptr) {
        kg::TensorArena::Bind(t, next);
        next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
      }
    };
    for (ggml_tensor* t : graph->inputs()) {
      bind_leaf(t);
    }
    for (ggml_tensor* node : graph->nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE && src->view_src == nullptr) {
          bind_leaf(src);
        }
      }
    }
    kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
    auto plan = kg::PlanGraph(graph->nodes, false, ModelDevice());
    ASSERT_TRUE(plan.has_value()) << rows << " at " << n_past << ": " << Why(plan);
    std::set<std::string_view> used;
    for (const auto& step : plan->steps) {
      used.insert(step.implementation);
    }
    for (const std::string_view name :
         {kg::kHcCombName, kg::kHcPreName, kg::kHcPostName, kg::kLightningIndexerName,
          kg::kFlashAttnMmaName, kg::kTopKName, kg::kArgsortName, kg::kSwiGluClampName,
          kg::kRopeExtName, kg::kMulMatHadamard, kg::kSetRowsExtName, kg::kGetRowsExtName}) {
      EXPECT_TRUE(used.contains(name)) << name << " at " << rows << " rows";
    }
    EXPECT_TRUE(used.contains(rows <= 8 ? kg::kMulMatIdVecQ : kg::kMulMatIdQ));
    auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
    ASSERT_TRUE(placed.has_value()) << Why(placed);
    EXPECT_GT(placed->extent, 0U);
    EXPECT_NE(graph->Named("l_last-42"), nullptr);
    EXPECT_NE(graph->Named("result_output"), nullptr);
  }
}

TEST(Dsv4Test, RequestedHeadRowsLeaveTheFullChunkBeforeTheGather) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Dsv4State(p, 8192, 512, md::Dsv4Window::kRing);
  ASSERT_TRUE(state.has_value());
  auto chunk = md::Dsv4Chunk(p, *state, 3000, 37, false);
  ASSERT_TRUE(chunk.has_value()) << Why(chunk);
  using NodeShape = std::pair<ggml_op, std::array<std::int64_t, 4>>;
  std::vector<NodeShape> full_trunk;
  for (const std::int64_t outputs : {0, 1, 3, 37}) {
    const kg::Dsv4ChunkShape shape = kg::Dsv4ShapeOf(*state, *chunk, outputs);
    auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto graph = kg::BuildDsv4Graph(*arena, p, *binding, shape, {.fused = true});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    const std::int64_t head_rows = outputs == 0 ? 37 : outputs;
    EXPECT_EQ(graph->out_ids->ne[0], head_rows);
    EXPECT_EQ(graph->logits->ne[0], p.vocab);
    EXPECT_EQ(graph->logits->ne[1], head_rows);
    EXPECT_EQ(graph->Named("hc_head")->ne[1], head_rows);
    EXPECT_EQ(graph->Named("result_norm")->ne[1], head_rows);
    EXPECT_EQ(graph->Named("l_last-42")->ne[2], 37);
    EXPECT_EQ(graph->csa_visible->ne[0], 37);
    EXPECT_EQ(graph->raw_k_idxs->ne[0], 37);
    const auto gather = std::ranges::find_if(graph->nodes, [&](const ggml_tensor* node) {
      return node->op == GGML_OP_GET_ROWS && node->src[1] == graph->out_ids;
    });
    ASSERT_NE(gather, graph->nodes.end());
    EXPECT_EQ((*gather)->src[0]->ne[0], p.hc_width());
    EXPECT_EQ((*gather)->src[0]->ne[1], 37);
    std::vector<NodeShape> trunk;
    for (auto it = graph->nodes.begin(); it != gather; ++it) {
      const ggml_tensor* node = *it;
      trunk.emplace_back(node->op, std::array{node->ne[0], node->ne[1], node->ne[2], node->ne[3]});
    }
    if (outputs == 0) {
      full_trunk = std::move(trunk);
    } else {
      EXPECT_EQ(trunk, full_trunk);
      EXPECT_NE(shape, kg::Dsv4ShapeOf(*state, *chunk));
    }
  }
  for (const std::int64_t outputs : {-1, 38}) {
    auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    EXPECT_FALSE(kg::BuildDsv4Graph(*arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk, outputs),
                                    {.fused = true})
                     .has_value());
  }
}

TEST(Dsv4Test, OrderedReductionSelectionRequiresTheCompleteCanonicalContract) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena.has_value());
  auto* down = ggml_new_tensor_3d(arena->context(), GGML_TYPE_F32, 4096, 6, 4096);
  auto* weights = ggml_new_tensor_3d(arena->context(), GGML_TYPE_F32, 1, 6, 4096);
  EXPECT_TRUE(kg::Dsv4WeightedReduceFits(down, weights));
  auto wrong = *down;
  wrong.ne[0] = 2048;
  EXPECT_FALSE(kg::Dsv4WeightedReduceFits(&wrong, weights));
  wrong = *down;
  wrong.ne[1] = 5;
  EXPECT_FALSE(kg::Dsv4WeightedReduceFits(&wrong, weights));
  wrong = *down;
  wrong.ne[2] = 4097;
  EXPECT_FALSE(kg::Dsv4WeightedReduceFits(&wrong, weights));
  wrong = *down;
  wrong.type = GGML_TYPE_BF16;
  EXPECT_FALSE(kg::Dsv4WeightedReduceFits(&wrong, weights));
  wrong = *weights;
  wrong.nb[2] += sizeof(float);
  EXPECT_FALSE(kg::Dsv4WeightedReduceFits(down, &wrong));
}

// The fast plan (Dsv4GraphOptions::fused, DeviceChoices::fuse_norms and
// vector_floats; D-085's note): chunks of up to 8 rows run the hyper-
// connections, the MoE blocks and the quantized products as jitLLM's fused
// operations, wider chunks keep GGML's products and hyper-connections, and
// every chunk's compressors are fused; far fewer steps than the reference.
TEST(Dsv4Test, TheFastPlanFusesDecodeAndVerifyChunks) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Dsv4State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::uint64_t> strides(p.layers, 8064224);
  strides[42] = 9309200;
  kg::DeviceChoices device = ModelDevice();
  device.fuse_norms = true;
  device.vector_floats = true;
  for (const auto& [n_past, rows] : {std::pair{37U, 1U}, {40U, 4U}, {0U, 37U}, {3000U, 512U}}) {
    auto chunk = md::Dsv4Chunk(p, *state, n_past, rows);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    const kg::Dsv4ChunkShape shape = kg::Dsv4ShapeOf(*state, *chunk);
    std::array<std::size_t, 2> steps = {0, 0};
    std::set<std::string_view> used;
    for (const bool fused : {false, true}) {
      auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
      ASSERT_TRUE(arena.has_value());
      auto graph = kg::BuildDsv4Graph(*arena, p, *binding, shape,
                                      {.expert_stride = strides, .fused = fused});
      ASSERT_TRUE(graph.has_value()) << Why(graph);
      std::uint64_t next = std::uint64_t{1} << 40U;
      const auto bind_leaf = [&](ggml_tensor* t) {
        if (t != nullptr && t->data == nullptr) {
          kg::TensorArena::Bind(t, next);
          next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
        }
      };
      for (ggml_tensor* t : graph->inputs()) {
        bind_leaf(t);
      }
      for (ggml_tensor* node : graph->nodes) {
        for (ggml_tensor* src : node->src) {
          if (src != nullptr && src->op == GGML_OP_NONE && src->view_src == nullptr) {
            bind_leaf(src);
          }
        }
      }
      kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
      auto plan = kg::PlanGraph(graph->nodes, false, fused ? device : ModelDevice());
      ASSERT_TRUE(plan.has_value()) << rows << " at " << n_past << ": " << Why(plan);
      steps[fused ? 1 : 0] = plan->steps.size();
      const auto reductions = std::ranges::count_if(plan->steps, [](const auto& step) {
        return step.implementation == kg::kDsv4WeightedReduceName;
      });
      EXPECT_EQ(reductions, fused && rows > kg::kVecQTokens ? p.layers : 0);
      const auto qheads = std::ranges::count_if(
          plan->steps, [](const auto& step) { return step.implementation == kg::kDsv4QHeadName; });
      EXPECT_EQ(qheads, fused && rows >= 16 ? p.layers : 0);
      if (fused) {
        for (const auto& step : plan->steps) {
          used.insert(step.implementation);
        }
        auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
        ASSERT_TRUE(placed.has_value()) << Why(placed);
        EXPECT_NE(graph->Named("l_last-42"), nullptr);
        EXPECT_NE(graph->Named("result_output"), nullptr);
      }
    }
    EXPECT_LT(steps[1], steps[0]) << rows << " rows";
    EXPECT_TRUE(used.contains(kg::kDsv4CompressName));
    EXPECT_TRUE(used.contains(kg::kRmsNormMulFused));
    const bool decode = rows <= kg::kVecQTokens;
    for (const std::string_view name :
         {kg::kVecQName, kg::kQuantizeQ8Name, kg::kDsv4RouteName, kg::kDsv4CombineName,
          kg::kDsv4HcMixName, kg::kDsv4HcPreName}) {
      EXPECT_EQ(used.contains(name), decode) << name << " at " << rows << " rows";
    }
    for (const std::string_view name :
         {kg::kMulMatIdQ, kg::kHcCombName, kg::kHcPreName, kg::kArgsortName}) {
      EXPECT_EQ(used.contains(name), !decode) << name << " at " << rows << " rows";
    }
    EXPECT_FALSE(used.contains(kg::kMulMatIdVecQ)) << rows << " rows";
  }
}

// The fast plan's window cache is a ring of the window and a chunk
// (model/dsv4.h Dsv4Window::kRing): its size no longer follows the
// context, which leaves the compressed caches' 6,880 bytes a position; each
// compressed cache follows its layer's window cache, so the graph reads the
// two as one tensor.
TEST(Dsv4Test, TheRingHoldsTheWindowAndAChunk) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  auto a = md::Dsv4State(p, 65536, 2048, md::Dsv4Window::kRing);
  auto b = md::Dsv4State(p, 262144, 2048, md::Dsv4Window::kRing);
  ASSERT_TRUE(a.has_value()) << Why(a);
  ASSERT_TRUE(b.has_value()) << Why(b);
  EXPECT_EQ(a->raw_cells, 2304U);  // pad(128 + 2,048, 256)
  EXPECT_EQ(b->raw_cells, 2304U);
  EXPECT_EQ(a->window, md::Dsv4Window::kRing);
  // 21 CSA layers' rows and indexer keys a 4 positions, 20 HCA layers' a 128.
  const std::uint64_t per_position = (b->bytes - a->bytes) / (262144 - 65536);
  EXPECT_EQ(per_position, 6880U);
  // A ring is never larger than the full cache (at the widest chunk the two
  // are one size).
  auto small = md::Dsv4State(p, 512, 128, md::Dsv4Window::kRing);
  ASSERT_TRUE(small.has_value());
  EXPECT_EQ(small->raw_cells, 256U);
  auto widest = md::Dsv4State(p, 1024, 896, md::Dsv4Window::kRing);
  ASSERT_TRUE(widest.has_value());
  EXPECT_EQ(widest->raw_cells, 1024U);
  // Both windows lay each compressed cache out right after its window's.
  for (const auto* s : {&*a, &*b, &*small}) {
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      const std::int64_t raw = s->Find(il, md::Dsv4StateTensor::Kind::kRawK);
      ASSERT_GE(raw, 0);
      for (const auto kind : {md::Dsv4StateTensor::Kind::kCsaK, md::Dsv4StateTensor::Kind::kHcaK}) {
        const std::int64_t comp = s->Find(il, kind);
        if (comp >= 0) {
          const auto& r = s->tensors[static_cast<std::size_t>(raw)];
          EXPECT_EQ(s->tensors[static_cast<std::size_t>(comp)].offset, r.offset + r.bytes) << il;
        }
      }
    }
  }
  // A ring's chunk reads the whole ring; the window wraps: at 5,000 the
  // window is positions 4,873-5,000, in cells 265-392 (mod 2,304).
  auto d = md::Dsv4Chunk(p, *a, 5000, 1, /*masks=*/false);
  ASSERT_TRUE(d.has_value()) << Why(d);
  EXPECT_EQ(d->raw_n_kv, 2304U);
  EXPECT_EQ(d->raw_cells, (std::vector<std::int64_t>{5000 % 2304}));
  std::vector<std::uint32_t> open;
  for (std::uint32_t j = 0; j < d->raw_n_kv; ++j) {
    if (d->raw_mask[j] == md::kHalfZero) {
      open.push_back(j);
    }
  }
  ASSERT_EQ(open.size(), 128U);
  EXPECT_EQ(open.front(), 4873U % 2304U);
  EXPECT_EQ(open.back(), 5000U % 2304U);
  // Without masks the compressed caches' are left out; their counts stay.
  EXPECT_TRUE(d->csa_mask.empty() && d->hca_mask.empty() && d->lid_mask.empty());
  EXPECT_EQ(d->csa.n_visible, (std::vector<std::int32_t>{1250}));
  EXPECT_EQ(d->hca.n_visible, (std::vector<std::int32_t>{39}));
  // A wrapping chunk: 2,048 rows from 3,000 keep each row's window.
  auto w = md::Dsv4Chunk(p, *a, 3000, 2048, false);
  ASSERT_TRUE(w.has_value());
  for (const std::uint32_t row : {0U, 1000U, 2047U}) {
    const std::uint64_t pos = 3000 + row;
    std::uint32_t count = 0;
    for (std::uint32_t j = 0; j < w->raw_n_kv; ++j) {
      if (w->raw_mask[(std::size_t{row} * w->raw_n_kv) + j] == md::kHalfZero) {
        ++count;
        EXPECT_EQ((pos - j) % 2304 < 128, true) << row << " cell " << j;
      }
    }
    EXPECT_EQ(count, 128U) << row;
  }
}

// The fast plan's attention and indexer at depth: the indexer's scores and
// selection, the device-built masks and the sparse gather, no
// concatenation, no GGML top-k and no host mask that grows with the
// context; every chunk's inputs the same size at 64K as at 256K.
TEST(Dsv4Test, TheFastPlanAttendsSparselyAtAnyDepth) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  std::vector<std::uint64_t> strides(p.layers, 8064224);
  strides[42] = 9309200;
  kg::DeviceChoices device = ModelDevice();
  device.fuse_norms = true;
  device.vector_floats = true;
  for (const std::uint32_t rows : {1U, 4U, 2048U}) {
    std::array<std::uint64_t, 2> inputs = {0, 0};
    for (std::size_t at = 0; at < 2; ++at) {
      const std::uint32_t context = at == 0 ? 65536 : 262144;
      auto state = md::Dsv4State(p, context, 2048, md::Dsv4Window::kRing);
      ASSERT_TRUE(state.has_value());
      auto chunk = md::Dsv4Chunk(p, *state, context - 4096, rows, false);
      ASSERT_TRUE(chunk.has_value()) << Why(chunk);
      auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
      ASSERT_TRUE(arena.has_value());
      auto graph = kg::BuildDsv4Graph(*arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk),
                                      {.expert_stride = strides, .fused = true});
      ASSERT_TRUE(graph.has_value()) << Why(graph);
      std::uint64_t next = std::uint64_t{1} << 40U;
      const auto bind_leaf = [&](ggml_tensor* t) {
        if (t != nullptr && t->data == nullptr && t->view_src == nullptr) {
          kg::TensorArena::Bind(t, next);
          next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
        }
      };
      for (ggml_tensor* t : graph->inputs()) {
        bind_leaf(t);
        inputs[at] += ggml_nbytes(t);
      }
      for (ggml_tensor* node : graph->nodes) {
        for (ggml_tensor* src : node->src) {
          if (src != nullptr && src->op == GGML_OP_NONE) {
            bind_leaf(src);
          }
        }
      }
      kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
      auto plan = kg::PlanGraph(graph->nodes, false, device);
      ASSERT_TRUE(plan.has_value()) << rows << " rows: " << Why(plan);
      std::set<std::string_view> used;
      for (const auto& step : plan->steps) {
        used.insert(step.implementation);
      }
      for (const std::string_view name :
           {kg::kDsv4LidTopKName, kg::kDsv4SparseMaskName, kg::kFlashAttnMmaName}) {
        EXPECT_TRUE(used.contains(name)) << name << " at " << rows << " rows";
      }
      for (const std::string_view name :
           {kg::kLightningIndexerName, kg::kTopKName, kg::kConcatName, kg::kFillName}) {
        EXPECT_FALSE(used.contains(name)) << name << " at " << rows << " rows";
      }
      std::size_t attention = 0;
      for (const ggml_tensor* node : graph->nodes) {
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
          ++attention;
          EXPECT_EQ(node->op_params[kg::kFlashAttnSparseParam], 1);
          EXPECT_LE(node->op_params[4], 128 + 2048);  // the window and the selected or HCA's rows
        }
      }
      EXPECT_EQ(attention, p.layers);
      EXPECT_FALSE(used.contains(kg::kFlashAttnMmaWideName));
      auto sharing_device = device;
      sharing_device.wide_sparse_attention = true;
      auto sharing = kg::PlanGraph(graph->nodes, false, sharing_device);
      ASSERT_TRUE(sharing.has_value()) << rows << " rows: " << Why(sharing);
      std::array<std::size_t, 3> families{};
      for (const auto& step : sharing->steps) {
        if (step.operation != jitllm::execution::Operation::kFlashAttn) {
          continue;
        }
        const ggml_tensor* mask = step.nodes[0]->src[3];
        const std::size_t family = kg::JitllmOpOf(mask) == kg::JitllmOp::kDsv4SparseMask
                                       ? static_cast<std::size_t>(kg::JitllmOpInt(mask, 1))
                                       : 2;
        ASSERT_LT(family, families.size());
        ++families[family];
        EXPECT_EQ(step.implementation,
                  family == 1 ? kg::kFlashAttnMmaName : kg::kFlashAttnMmaWideName);
      }
      // The real Flash profile has 21 selected-list CSA, 20 count-based
      // HCA and two window-only layers, at either depth and all row shapes.
      EXPECT_EQ(families, (std::array<std::size_t, 3>{21, 20, 2}));
      EXPECT_EQ(graph->csa.mask, nullptr);
      EXPECT_EQ(graph->hca.mask, nullptr);
      EXPECT_EQ(graph->lid.mask, nullptr);
      auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
      ASSERT_TRUE(placed.has_value()) << Why(placed);
    }
    EXPECT_EQ(inputs[0], inputs[1]) << rows << " rows";
  }
  // D-092's row-invariant verify is the reference mode's, not the fast plan's.
  auto state = md::Dsv4State(p, 8192, 512, md::Dsv4Window::kRing);
  ASSERT_TRUE(state.has_value());
  auto chunk = md::Dsv4Chunk(p, *state, 100, 4, false);
  ASSERT_TRUE(chunk.has_value());
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
  ASSERT_TRUE(arena.has_value());
  EXPECT_FALSE(kg::BuildDsv4Graph(*arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk),
                                  {.expert_stride = strides, .row_invariant = true, .fused = true})
                   .has_value());
}

// A wave (dsv4_graph.h Dsv4WaveGraph): several slots' chunks at their own
// positions in one graph, whose row-local operations run once over every
// slot's rows (as many vector products as one chunk of the same rows) and
// whose state, compressors, indexer and attention stay each slot's own.
TEST(Dsv4Test, AWaveJoinsRowLocalWorkAndKeepsEachSlotsStateItsOwn) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Dsv4State(p, 8192, 512, md::Dsv4Window::kRing);
  ASSERT_TRUE(state.has_value());
  std::vector<std::uint64_t> strides(p.layers, 8064224);
  strides[42] = 9309200;
  kg::DeviceChoices device = ModelDevice();
  device.fuse_norms = true;
  device.vector_floats = true;
  // Three slots: a decode step, a four-row verify and a two-row one.
  const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> slots = {
      {{100U, 1U}, {2000U, 4U}, {5000U, 2U}}};
  kg::Dsv4WaveShape shape;
  std::uint32_t rows = 0;
  for (const auto& [n_past, n] : slots) {
    auto chunk = md::Dsv4Chunk(p, *state, n_past, n, false);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    shape.slots.push_back(kg::Dsv4ShapeOf(*state, *chunk));
    rows += n;
  }
  const auto plan_of = [&](std::span<ggml_tensor* const> nodes,
                           std::span<ggml_tensor* const> inputs) {
    std::uint64_t next = std::uint64_t{1} << 40U;
    const auto bind_leaf = [&](ggml_tensor* t) {
      if (t != nullptr && t->data == nullptr && t->view_src == nullptr) {
        kg::TensorArena::Bind(t, next);
        next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
      }
    };
    for (ggml_tensor* t : inputs) {
      bind_leaf(t);
    }
    for (ggml_tensor* node : nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE) {
          bind_leaf(src);
        }
      }
    }
    kg::BindDistinct(nodes, std::uint64_t{1} << 46U);
    return kg::PlanGraph(nodes, false, device);
  };
  const auto count = [](const kg::GraphPlan& plan, std::string_view name) {
    return std::ranges::count_if(plan.steps,
                                 [&](const auto& step) { return step.implementation == name; });
  };
  auto arena = kg::TensorArena::Create(kg::Dsv4WaveGraphTensors(p, slots.size()));
  ASSERT_TRUE(arena.has_value());
  auto wave =
      kg::BuildDsv4WaveGraph(*arena, p, *binding, shape, {.expert_stride = strides, .fused = true});
  ASSERT_TRUE(wave.has_value()) << Why(wave);
  ASSERT_EQ(wave->slots.size(), slots.size());
  EXPECT_EQ(wave->first, (std::vector<std::int64_t>{0, 1, 5}));
  EXPECT_EQ(wave->joined.embd->ne[1], rows);
  EXPECT_EQ(wave->joined.logits->ne[0], p.vocab);
  EXPECT_EQ(wave->joined.logits->ne[1], rows);
  EXPECT_EQ(wave->joined.out_ids, nullptr);
  const auto inputs = wave->inputs();
  EXPECT_EQ(inputs.size(), 7 + (slots.size() * 19));
  auto plan = plan_of(wave->joined.nodes, inputs);
  ASSERT_TRUE(plan.has_value()) << Why(plan);
  // One chunk of the same rows, for the row-local operations' count.
  auto chunk = md::Dsv4Chunk(p, *state, 2000, rows, false);
  ASSERT_TRUE(chunk.has_value());
  auto one_arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
  ASSERT_TRUE(one_arena.has_value());
  auto one = kg::BuildDsv4Graph(*one_arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk),
                                {.expert_stride = strides, .fused = true});
  ASSERT_TRUE(one.has_value()) << Why(one);
  auto one_plan = plan_of(one->nodes, one->inputs());
  ASSERT_TRUE(one_plan.has_value()) << Why(one_plan);
  for (const std::string_view name :
       {kg::kVecQName, kg::kQuantizeQ8Name, kg::kDsv4RouteName, kg::kDsv4CombineName,
        kg::kDsv4HcMixName, kg::kDsv4HcPreName}) {
    EXPECT_EQ(count(*plan, name), count(*one_plan, name)) << name;
    EXPECT_GT(count(*plan, name), 0) << name;
  }
  // Per slot: its attention, indexer and compressors (21 CSA layers with
  // the indexer's compressor, 20 HCA).
  EXPECT_EQ(count(*plan, kg::kDsv4LidTopKName), 21 * 3);
  EXPECT_EQ(count(*plan, kg::kDsv4CompressName), ((21 * 2) + 20) * 3);
  EXPECT_EQ(count(*plan, kg::kDsv4CompressName), 3 * count(*one_plan, kg::kDsv4CompressName));
  std::size_t attention = 0;
  for (const ggml_tensor* node : wave->joined.nodes) {
    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
      ++attention;
    }
  }
  EXPECT_EQ(attention, std::size_t{p.layers} * slots.size());
  // Each slot's state is its own: no two slots name one state tensor, and
  // every cache write goes to a slot's own.
  std::set<const ggml_tensor*> seen;
  for (const kg::Dsv4Graph& slot : wave->slots) {
    for (const kg::Dsv4LayerTensors& l : slot.layers) {
      ASSERT_NE(l.raw_k, nullptr);
      EXPECT_TRUE(seen.insert(l.raw_k).second);
      EXPECT_EQ(l.q_a, nullptr);  // the weights are the joined graph's
    }
  }
  for (const kg::Dsv4LayerTensors& l : wave->joined.layers) {
    EXPECT_EQ(l.raw_k, nullptr);
    EXPECT_NE(l.q_a, nullptr);
  }
  for (const ggml_tensor* node : wave->joined.nodes) {
    if (node->op == GGML_OP_SET_ROWS) {
      // GGML's set_rows is a view of the tensor it writes.
      const ggml_tensor* target = node->view_src;
      bool owned = false;
      for (const kg::Dsv4Graph& slot : wave->slots) {
        for (const kg::Dsv4LayerTensors& l : slot.layers) {
          for (const ggml_tensor* t :
               {l.raw_k, l.csa_state_kv, l.csa_state_score, l.lid_k, l.lid_state_kv,
                l.lid_state_score, l.hca_state_kv, l.hca_state_score}) {
            owned = owned || (t != nullptr && t == target);
          }
        }
      }
      EXPECT_TRUE(owned);
    }
  }
  auto placed = kg::PlaceActivations(wave->joined.nodes, *plan, inputs, 256);
  ASSERT_TRUE(placed.has_value()) << Why(placed);
  // Refused: past the rows the products keep column-invariant, more than
  // four slots, the reference form, a head narrowed, layouts that differ.
  const auto refused = [&](const kg::Dsv4WaveShape& s, const kg::Dsv4GraphOptions& o) {
    auto a = kg::TensorArena::Create(kg::Dsv4WaveGraphTensors(p, 5));
    return a.has_value() && !kg::BuildDsv4WaveGraph(*a, p, *binding, s, o).has_value();
  };
  const kg::Dsv4GraphOptions fused{.expert_stride = strides, .fused = true};
  // Four four-row verifies (16 rows): the vector products still joined,
  // the float products (router, indexer weights, head mixes) a slot at a
  // time, as each slot's own chunk runs them.
  kg::Dsv4WaveShape full;
  full.slots.assign(4, shape.slots[1]);
  {
    auto a = kg::TensorArena::Create(kg::Dsv4WaveGraphTensors(p, 4));
    ASSERT_TRUE(a.has_value());
    auto sixteen = kg::BuildDsv4WaveGraph(*a, p, *binding, full, fused);
    ASSERT_TRUE(sixteen.has_value()) << Why(sixteen);
    auto plan16 = plan_of(sixteen->joined.nodes, sixteen->inputs());
    ASSERT_TRUE(plan16.has_value()) << Why(plan16);
    EXPECT_EQ(count(*plan16, kg::kVecQName), count(*one_plan, kg::kVecQName));
    // Per layer the router, per CSA layer the indexer's weights, the head's
    // mixes: each slot's.
    EXPECT_EQ(count(*plan16, kg::kMulMatVecFRows), 4 * (43 + 21 + 1));
    EXPECT_EQ(count(*plan, kg::kMulMatVecFRows), 43 + 21 + 1);
  }
  kg::Dsv4WaveShape wide;
  auto eight = md::Dsv4Chunk(p, *state, 3000, 8, false);
  ASSERT_TRUE(eight.has_value());
  wide.slots.assign(3, kg::Dsv4ShapeOf(*state, *eight));
  EXPECT_TRUE(refused(wide, fused));  // 24 rows
  kg::Dsv4WaveShape many;
  many.slots.assign(5, shape.slots[0]);
  EXPECT_TRUE(refused(many, fused));
  EXPECT_TRUE(refused(shape, {.expert_stride = strides}));
  kg::Dsv4WaveShape narrowed = shape;
  narrowed.slots[1].outputs = 1;
  EXPECT_TRUE(refused(narrowed, fused));
  kg::Dsv4WaveShape other = shape;
  other.slots[2].csa_cells += 256;
  EXPECT_TRUE(refused(other, fused));
  EXPECT_TRUE(refused({}, fused));
}

// The fast plan's attention mask is F16 [ring cells + compressed cells,
// rows], and the MMA kernel takes its planes' strides in 32 bits (RE-037):
// the widest chunk Dsv4MostRows admits keeps every attention's operands
// within the kernel's check at the deepest chunk, and one row more does
// not. A 4,096-row chunk fits to 1,030,144 positions; at 1,048,576 the
// widest is 4,029 rows (4,024 in whole tiles).
TEST(Dsv4Test, TheWidestChunksMaskFitsTheAttentionKernel) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  std::vector<std::uint64_t> strides(p.layers, 8064224);
  strides[42] = 9309200;
  // The first refusal of the kernel's check among the attentions of the
  // chunk of `rows` that ends the context; empty when all pass.
  const auto refused = [&](std::uint32_t context, std::uint32_t rows) -> std::string {
    auto state = md::Dsv4State(p, context, rows, md::Dsv4Window::kRing);
    if (!state) {
      return "no state: " + state.error();
    }
    auto chunk = md::Dsv4Chunk(p, *state, context - rows, rows, false);
    if (!chunk) {
      return "no chunk: " + chunk.error();
    }
    auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
    if (!arena) {
      return "no arena";
    }
    auto graph = kg::BuildDsv4Graph(*arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk),
                                    {.expert_stride = strides, .fused = true});
    if (!graph) {
      return "no graph: " + Why(graph);
    }
    std::uint64_t next = std::uint64_t{1} << 40U;
    const auto bind_leaf = [&](ggml_tensor* t) {
      if (t != nullptr && t->data == nullptr && t->view_src == nullptr) {
        kg::TensorArena::Bind(t, next);
        next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
      }
    };
    for (ggml_tensor* t : graph->inputs()) {
      bind_leaf(t);
    }
    for (ggml_tensor* node : graph->nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE) {
          bind_leaf(src);
        }
      }
    }
    kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
    std::size_t attention = 0;
    std::string first;
    for (const ggml_tensor* node : graph->nodes) {
      if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        ++attention;
        auto checked = kg::CheckFlashAttnMma(node);
        if (!checked && first.empty()) {
          first = Why(checked);
        }
      }
    }
    EXPECT_EQ(attention, p.layers);
    return first;
  };
  constexpr std::string_view kStrides = "flash attention beyond the kernel's 32-bit extents";
  for (const auto& [context, most] : {std::pair{1048576U, 4029U}, std::pair{1030144U, 4100U}}) {
    SCOPED_TRACE(context);
    EXPECT_EQ(md::Dsv4MostRows(p, context), most);
    EXPECT_EQ(refused(context, most), "");
    EXPECT_TRUE(refused(context, most + 1).starts_with(kStrides));
  }
  EXPECT_EQ(md::Dsv4MostRows(p, 1030400), 4095U);
  EXPECT_EQ(md::Dsv4MostRows(p, 262144), 13530U);
  EXPECT_EQ(refused(1048576, 2048), "");
  EXPECT_TRUE(refused(1048576, 4096).starts_with(kStrides));
}

TEST(Dsv4Test, ExpertStridesAreWholeBlocks) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  const std::vector<md::Dsv4Resource> resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value());
  auto state = md::Dsv4State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  auto chunk = md::Dsv4Chunk(p, *state, 0, 1);
  ASSERT_TRUE(chunk.has_value());
  const kg::Dsv4ChunkShape shape = kg::Dsv4ShapeOf(*state, *chunk);
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
  ASSERT_TRUE(arena.has_value());
  // 8,064,223 bytes is not a whole number of IQ2_XS blocks.
  std::vector<std::uint64_t> strides(p.layers, 8064223);
  const auto torn = kg::BuildDsv4Graph(*arena, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(torn).find("expert stride"), std::string::npos) << Why(torn);
  // Nor may a stride be shorter than a slice.
  auto again = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
  ASSERT_TRUE(again.has_value());
  strides.assign(p.layers, 74);
  const auto short_stride =
      kg::BuildDsv4Graph(*again, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(short_stride).find("expert stride"), std::string::npos) << Why(short_stride);
  // One stride per layer, or none.
  auto third = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
  ASSERT_TRUE(third.has_value());
  strides.assign(3, 8064224);
  const auto few = kg::BuildDsv4Graph(*third, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(few).find("per layer"), std::string::npos) << Why(few);
  // Whole blocks of every projection (IQ2_XS 74, IQ3_XXS 98, MXFP4 17 on
  // the last layer) build.
  auto fourth = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
  ASSERT_TRUE(fourth.has_value());
  strides.assign(p.layers, 8064224);
  strides[42] = 9309200;
  const auto whole = kg::BuildDsv4Graph(*fourth, p, *binding, shape, {.expert_stride = strides});
  EXPECT_TRUE(whole.has_value()) << Why(whole);
}

TEST(Dsv4Test, OutAPrefillRequiresItsQualifiedShapeAndDisjointOutput) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena.has_value());
  auto* c = arena->context();
  auto* w = ggml_new_tensor_3d(c, GGML_TYPE_Q8_0, 4096, 1024, 8);
  auto* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, 512, 64, 4096);
  auto* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4096);
  const kg::Dsv4OutAParams params{.base = 10000, .scale = 1, .attention = 1};
  ASSERT_TRUE(kg::Dsv4OutAFits(w, x, pos, params));
  auto* out = kg::Dsv4OutA(c, w, x, pos, params);
  kg::TensorArena::Bind(w, std::uint64_t{1} << 40U);
  kg::TensorArena::Bind(x, std::uint64_t{1} << 42U);
  kg::TensorArena::Bind(pos, std::uint64_t{1} << 44U);
  const auto address = std::uint64_t{1} << 46U;
  kg::TensorArena::Bind(out, address);
  EXPECT_TRUE(kg::CheckDsv4OutA(out).has_value());
  EXPECT_EQ(kg::Dsv4OutAParamsOf(out).base, params.base);
  for (auto* input : {w, x, pos}) {
    kg::TensorArena::Bind(out, reinterpret_cast<std::uintptr_t>(input->data));
    EXPECT_FALSE(kg::CheckDsv4OutA(out).has_value());
  }
  kg::TensorArena::Bind(out, address);
  const auto saved = x->nb[2];
  x->nb[2] += sizeof(float);
  EXPECT_FALSE(kg::CheckDsv4OutA(out).has_value());
  x->nb[2] = saved;
  auto wrong = *x;
  wrong.ne[2] = 2048;
  EXPECT_FALSE(kg::Dsv4OutAFits(w, &wrong, pos, params));
  wrong = *w;
  wrong.type = GGML_TYPE_Q5_K;
  EXPECT_FALSE(kg::Dsv4OutAFits(&wrong, x, pos, params));
  auto bad_params = params;
  bad_params.extension = 1;
  EXPECT_FALSE(kg::Dsv4OutAFits(w, x, pos, bad_params));
}

TEST(Dsv4Test, OutAPrefillIsOptInAndKeepsUnrotatedHeadsLive) {
  const auto& p = md::Dsv4Flash();
  const auto resources = GgufLike(p);
  auto binding = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Dsv4State(p, 8192, 4096);
  ASSERT_TRUE(state.has_value());
  for (const auto rows : {1U, 2048U, 4096U}) {
    auto chunk = md::Dsv4Chunk(p, *state, 0, rows);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    for (const auto mode : {0, 1, 2}) {
      auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
      ASSERT_TRUE(arena.has_value());
      auto graph = kg::BuildDsv4Graph(*arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk),
                                      {.fused = mode != 2, .outa_prefill = mode != 0});
      ASSERT_TRUE(graph.has_value()) << Why(graph);
      std::uint64_t next = std::uint64_t{1} << 40U;
      const auto bind_leaf = [&](ggml_tensor* t) {
        if (t != nullptr && t->data == nullptr) {
          kg::TensorArena::Bind(t, next);
          next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
        }
      };
      for (auto* t : graph->inputs()) bind_leaf(t);
      for (auto* node : graph->nodes) {
        for (auto* src : node->src) {
          if (src != nullptr && src->op == GGML_OP_NONE && src->view_src == nullptr) bind_leaf(src);
        }
      }
      kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
      auto plan = kg::PlanGraph(graph->nodes, false, ModelDevice());
      ASSERT_TRUE(plan.has_value()) << Why(plan);
      const auto count = std::ranges::count_if(
          plan->steps, [](const auto& step) { return step.implementation == kg::kDsv4OutAName; });
      EXPECT_EQ(count, mode == 1 && rows == 4096 ? p.layers : 0);
      // The coalesced repack (the fast plan's default) takes each one.
      auto fast_device = ModelDevice();
      fast_device.outa_fast_pack = true;
      auto fast = kg::PlanGraph(graph->nodes, false, fast_device);
      ASSERT_TRUE(fast.has_value()) << Why(fast);
      EXPECT_EQ(std::ranges::count_if(fast->steps,
                                      [](const auto& step) {
                                        return step.implementation == kg::kDsv4OutAFastPackName;
                                      }),
                count);
      if (count == 0) continue;
      auto placement = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
      ASSERT_TRUE(placement.has_value()) << Why(placement);
      for (const auto& [tensor, offset] : placement->offsets) {
        kg::TensorArena::Bind(tensor, (std::uint64_t{1} << 46U) + offset);
      }
      kg::BindViews(graph->nodes);
      for (const auto& step : plan->steps) {
        if (step.implementation == kg::kDsv4OutAName) {
          EXPECT_TRUE(kg::CheckDsv4OutA(step.nodes.front()).has_value());
        }
      }
    }
  }
}

// The community IQ2_XXS artifact's types (cd39d504…, docs/model-support.md):
// F16 HC mixes, compressors, indexer projections and compressor APE tables,
// Q8_0 Q-A and shared experts, IQ2_XXS gate/up and Q2_K down experts.
std::vector<md::Dsv4Resource> CommunityLike(const md::Dsv4Profile& p) {
  auto resources = GgufLike(p);
  for (auto& r : resources) {
    const std::string& role = r.roles[0];
    const auto ends = [&](std::string_view suffix) { return role.ends_with(suffix); };
    if (ends("hc_fn.weight") || ends("hc_attn_fn.weight") || ends("hc_ffn_fn.weight") ||
        ends("compressor_kv.weight") || ends("compressor_gate.weight") ||
        ends("compressor_ape.weight") || ends("indexer.attn_q_b.weight") ||
        ends("indexer.proj.weight")) {
      r.type = "F16";
    } else if (ends("attn_q_a.weight") || ends("_shexp.weight")) {
      r.type = "Q8_0";
    } else if (ends("ffn_gate_exps.weight") || ends("ffn_up_exps.weight")) {
      r.type = "IQ2_XXS";
    } else if (ends("ffn_down_exps.weight")) {
      r.type = "Q2_K";
    }
  }
  return resources;
}

// The ds4 prefill stage mechanisms, the fast plan's defaults
// (SetDsv4PrefillStages; docs/experiments/ds4-prefill-stages): each selected
// only where its guard admits the graph. On the original UD-Q2_K_XL types
// only F16 Q and the dense Q8_0 pairs apply; on the community types every
// mechanism applies from 64 rows, the IQ2 pair's write-back only at its
// measured 4,096 rows, where D2R takes the Q2_K down product unless it is
// off. Decode and verify widths and the stages-off plan are unchanged.
TEST(Dsv4Test, PrefillStageMechanismsSelectOnlyWhereTheirGuardsAdmit) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  struct Case {
    bool community;
    std::uint32_t rows;
    bool stages;
    bool d2r;
  };
  for (const Case& test :
       {Case{false, 4, true, true}, Case{false, 63, true, true}, Case{false, 2048, true, true},
        Case{false, 2048, false, false}, Case{true, 4, true, true}, Case{true, 64, true, true},
        Case{true, 2048, true, true}, Case{true, 2048, false, false}, Case{true, 4096, true, true},
        Case{true, 4096, true, false}}) {
    const std::string what =
        std::format("{} {} rows, stages {}, D2R {}", test.community ? "community" : "original",
                    test.rows, test.stages, test.d2r);
    const auto resources = test.community ? CommunityLike(p) : GgufLike(p);
    auto binding = md::BindDsv4(p, "deepseek4", resources);
    ASSERT_TRUE(binding.has_value()) << what << ": " << Why(binding);
    auto state = md::Dsv4State(p, 8192, test.rows == 4096 ? 4096 : 2048, md::Dsv4Window::kRing);
    ASSERT_TRUE(state.has_value()) << what;
    auto chunk = md::Dsv4Chunk(p, *state, 0, test.rows, false);
    ASSERT_TRUE(chunk.has_value()) << what << ": " << Why(chunk);
    auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    kg::Dsv4GraphOptions options{.fused = true};
    kg::SetDsv4PrefillStages(options, test.stages);
    auto graph = kg::BuildDsv4Graph(*arena, p, *binding, kg::Dsv4ShapeOf(*state, *chunk), options);
    ASSERT_TRUE(graph.has_value()) << what << ": " << Why(graph);
    std::uint64_t next = std::uint64_t{1} << 40U;
    const auto bind_leaf = [&](ggml_tensor* t) {
      if (t != nullptr && t->data == nullptr && t->view_src == nullptr) {
        kg::TensorArena::Bind(t, next);
        next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
      }
    };
    for (ggml_tensor* t : graph->inputs()) {
      bind_leaf(t);
    }
    for (ggml_tensor* node : graph->nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE) {
          bind_leaf(src);
        }
      }
    }
    kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
    kg::DeviceChoices device = ModelDevice();
    device.fuse_norms = true;
    device.vector_floats = true;
    device.pair_experts = true;
    device.compact_experts = test.rows >= kg::kDsv4CompactMinRows;
    device.wide_sparse_attention = true;
    // A model of the measured GB10 predicate (mul_mat_q2_d2r.cu).
    device.q2_d2r_fits = [](const ggml_tensor* node) {
      return node->src[0]->type == GGML_TYPE_Q2_K && node->src[0]->ne[0] == 2048 &&
             node->src[0]->ne[1] == 4096 && node->src[0]->ne[2] == 256 &&
             node->src[1]->ne[2] == 4096 && node->src[2]->ne[0] == 6;
    };
    // And of the GB10 write-back pair's device condition (mul_mat_q.cu).
    device.pair_glu_fits = [](const ggml_tensor*, const ggml_tensor*) { return true; };
    kg::SetDsv4PrefillStages(device, test.stages);
    device.d2r_experts = test.d2r;
    auto plan = kg::PlanGraph(graph->nodes, false, device);
    ASSERT_TRUE(plan.has_value()) << what << ": " << Why(plan);
    const auto count = [&](std::string_view name) {
      return std::ranges::count_if(plan->steps,
                                   [&](const auto& step) { return step.implementation == name; });
    };
    const auto layers = static_cast<std::ptrdiff_t>(p.layers);
    const bool wide = test.stages && test.rows >= 64;
    const bool community = wide && test.community;
    // Layer 0's attention mix input alone; every later mix input with the
    // post before it, each FFN post but the last forming the expert sum.
    EXPECT_EQ(count(kg::kDsv4HcNormF16Name), community ? 1 : 0) << what;
    EXPECT_EQ(count(kg::kDsv4HcPostNormF16Name), community ? layers : 0) << what;
    EXPECT_EQ(count(kg::kDsv4HcPostExpertsNormF16Name), community ? layers - 1 : 0) << what;
    if (community) {
      EXPECT_EQ(count(kg::kHcPostName), 1) << what;  // the last, before the head
    } else if (test.rows > kg::kVecQTokens) {
      EXPECT_EQ(count(kg::kHcPostName), 2 * layers) << what;
    }
    // One shared copy of each compressed layer's attention input.
    EXPECT_EQ(count(kg::kDsv4F16CopyName), community ? 41 : 0) << what;
    // Q-A with KV, shared up with gate.
    const auto dense = count(kg::kMulMatQPairDense);
    if (!wide) {
      EXPECT_EQ(dense, 0) << what;
    } else if (test.community) {
      EXPECT_EQ(dense, 2 * layers) << what;
    } else {
      // The compressors' and indexer's Q8_0 products of one input.
      EXPECT_GE(dense, 41) << what;
    }
    const bool glu = community && test.rows == 4096;
    EXPECT_EQ(count(kg::kMulMatIdQPairGlu), glu && test.d2r ? layers : 0) << what;
    EXPECT_EQ(count(kg::kMulMatIdQ2D2r),
              test.community && test.rows == 4096 && test.d2r ? layers : 0)
        << what;
    EXPECT_EQ(count(kg::kMulMatIdQPairGluQ8), glu && !test.d2r ? layers : 0) << what;
    EXPECT_EQ(count(kg::kMulMatIdQCompactPrequant), glu && !test.d2r ? layers : 0) << what;
    EXPECT_EQ(count(kg::kMulMatIdQPairCompact),
              test.rows >= kg::kDsv4CompactMinRows && !glu ? layers : 0)
        << what;
    if (glu) {
      // A device without the occupancy-two kernel keeps the plain pair.
      kg::DeviceChoices other = device;
      other.pair_glu_fits = [](const ggml_tensor*, const ggml_tensor*) { return false; };
      auto plain = kg::PlanGraph(graph->nodes, false, other);
      ASSERT_TRUE(plain.has_value()) << what << ": " << Why(plain);
      const auto plain_count = [&](std::string_view name) {
        return std::ranges::count_if(plain->steps,
                                     [&](const auto& step) { return step.implementation == name; });
      };
      EXPECT_EQ(plain_count(kg::kMulMatIdQPairCompact), layers) << what;
      EXPECT_EQ(plain_count(kg::kMulMatIdQPairGlu) + plain_count(kg::kMulMatIdQPairGluQ8) +
                    plain_count(kg::kMulMatIdQCompactPrequant),
                0)
          << what;
    }
    std::ptrdiff_t f16_heads = 0;
    std::ptrdiff_t f16_queries = 0;
    for (const ggml_tensor* node : graph->nodes) {
      if (kg::JitllmOpOf(node) == kg::JitllmOp::kDsv4QHead && node->type == GGML_TYPE_F16) {
        ++f16_heads;
      }
      if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[0]->type == GGML_TYPE_F16) {
        ++f16_queries;
      }
    }
    EXPECT_EQ(f16_heads, wide ? layers : 0) << what;
    EXPECT_EQ(f16_queries, wide ? layers : 0) << what;
    auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
    ASSERT_TRUE(placed.has_value()) << what << ": " << Why(placed);
  }
}

TEST(Dsv4Test, TheHadamardMatrixIsOrthonormal) {
  const std::vector<float> h = kg::HadamardMatrix(128);
  for (std::size_t i = 0; i < 128; i += 17) {
    for (std::size_t j = 0; j < 128; j += 13) {
      double dot = 0;
      for (std::size_t k = 0; k < 128; ++k) {
        dot += static_cast<double>(h[(i * 128) + k]) * static_cast<double>(h[(j * 128) + k]);
      }
      EXPECT_NEAR(dot, i == j ? 1.0 : 0.0, 1e-5);
    }
  }
}

}  // namespace
