// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 Flash Next adapter (model/qwen38.h) and its chunk graph
// (kernels/ggml/qwen38_graph.h), in every profile:
// - the binding of a synthetic resource list shaped and typed as the
//   artifact import_m3.py writes from Mia's checkpoint, and its refusals;
// - the n-gram hash's constants checked against the table, and its rows
//   worked by hand (llm_graph_input_ple::set_input at b29c606e2);
// - the state layout's sizes, and a chunk's masks, positions and QSA block
//   tables against llama.cpp's rules (set_input_qsa), worked by hand;
// - the graph at prefill, decode and past-the-budget shapes: every node
//   planned by an implementation of this module (a model of the device's
//   choices), MXFP8 products by jitLLM's vector product or the BF16
//   dequantization, the activations placed.

#include "model/qwen38.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace {

namespace md = jitllm::model;
namespace kg = jitllm::kernels::ggml;
using jitllm::test_support::Failed;

template <typename T>
std::string Why(const std::expected<T, std::string>& result) {
  return Failed(result).value_or(std::string());
}
template <typename T>
std::string Why(const std::expected<T, kg::KernelFailure>& result) {
  return Failed(result, &kg::KernelFailure::detail).value_or(std::string());
}

constexpr std::uint64_t kTableRows = 320001536;
// One NVFP4 expert slice (640 rows of 40 blocks, or 2,560 of 10, at 36
// bytes), down's over-read (384 elements), the stored group (2,768,896
// bytes) and the slab's stride over it (whole 36-byte blocks and 16 bytes).
constexpr std::uint64_t kExpertSlice = 921600;
constexpr std::uint64_t kDownOverRead = 216;
constexpr std::uint64_t kExpertStride = 2768976;

// The artifact's resources (docs/experiments/artifact-layout/modelopt_qwen38.py):
// GGML BF16 and F32, plain MXFP8 and I64 and the n-gram table, then the
// routed experts' arrays: GGML's NVFP4 (the first import) or, with
// `cutlass`, the CUTLASS layout's four I8 arrays (the importer's since).
std::vector<md::Qwen38Resource> ArtifactLike(const md::Qwen38Profile& p, bool cutlass = false) {
  std::vector<md::Qwen38Resource> r;
  std::vector<md::Qwen38Resource> arrays;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto plain = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = true, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto mx = [&](const std::string& name, std::uint64_t k, std::uint64_t n) {
    plain(name + ".weight", "F8_E4M3", {k, n});
    plain(name + ".weight_scale", "U8", {k / 32, n});
  };
  ggml("token_embd.weight", "BF16", {2560, 248320});
  ggml("output.weight", "BF16", {2560, 248320});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "BF16", {10240, 320});
  ggml("output_hc_up.weight", "BF16", {320, 10240});
  plain("per_layer_token_embd.weight", "U8", {90, kTableRows});
  plain("per_layer_token_embd.weight_scale_2", "F32", {1});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    for (const char* kind : {"attn", "ffn"}) {
      ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {10240});
      ggml(std::format("{}hc_{}_down.weight", n, kind), "BF16", {10240, 320});
      ggml(std::format("{}hc_{}_up.weight", n, kind), "BF16", {320, 10240});
      ggml(std::format("{}hc_{}_inject.weight", n, kind), "BF16", {10240, 4});
    }
    if (p.linear(il)) {
      mx(n + "attn_qkv", 2560, 10240);
      mx(n + "attn_gate", 2560, 6144);
      mx(n + "ssm_beta", 2560, 48);
      mx(n + "ssm_alpha", 2560, 48);
      ggml(n + "ssm_dt.bias", "F32", {48});
      ggml(n + "ssm_a", "F32", {48});
      ggml(n + "ssm_conv1d.weight", "F32", {4, 10240});
      ggml(n + "ssm_norm.weight", "F32", {128});
      mx(n + "ssm_out", 6144, 2560);
    } else {
      mx(n + "attn_q", 2560, 12288);
      mx(n + "attn_k", 2560, 512);
      mx(n + "attn_v", 2560, 512);
      mx(n + "attn_output", 6144, 2560);
      ggml(n + "attn_q_norm.weight", "F32", {256});
      ggml(n + "attn_k_norm.weight", "F32", {256});
      mx(n + "indexer.qk_proj", 2560, 640);
      ggml(n + "indexer.q_norm.weight", "F32", {128});
      ggml(n + "indexer.k_norm.weight", "F32", {128});
    }
    if (il == 1) {
      ggml(n + "ple_key.weight", "BF16", {2560, 10240});
      ggml(n + "ple_value.weight", "BF16", {2560, 2560});
      for (const char* part : {"key", "query", "conv"}) {
        ggml(std::format("{}ple_norm_{}.weight", n, part), "F32", {10240});
      }
      ggml(n + "ple_conv1d.weight", "F32", {4, 10240});
      plain(n + "ple_multipliers", "I64", {3});
      plain(n + "ple_head_offsets", "I64", {16});
      plain(n + "ple_head_vocab", "I64", {16});
    }
    ggml(n + "ffn_gate_inp.weight", "BF16", {2560, 512});
    ggml(n + "ffn_gate_inp_shexp.weight", "BF16", {2560});
    mx(n + "ffn_gate_shexp", 2560, 640);
    mx(n + "ffn_up_shexp", 2560, 640);
    mx(n + "ffn_down_shexp", 640, 2560);
    // Each expert group as the importer writes it: gate, up and down slices
    // of 921,600 bytes, down's rows padded to 1,024 elements (216 bytes of
    // over-read after its last row).
    std::uint64_t group_offset = 0;
    if (cutlass) {
      // Gate and up codes, their scales, down's codes and scales: 1,638,400,
      // 204,800, 819,200 and 102,400 bytes from the group's start.
      for (const char* proj : {"gate", "up", "down"}) {
        ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {512});
      }
      for (const auto& [name, ne] :
           {std::pair{"gate_up_exps.codes", std::vector<std::uint64_t>{1280, 1280}},
            std::pair{"gate_up_exps.scales", std::vector<std::uint64_t>{512, 400}},
            std::pair{"down_exps.codes", std::vector<std::uint64_t>{320, 2560}},
            std::pair{"down_exps.scales", std::vector<std::uint64_t>{512, 200}}}) {
        arrays.push_back({.roles = {std::format("{}ffn_{}", n, name)},
                          .plain = false,
                          .type = "I8",
                          .ne = ne,
                          .expert_array = true,
                          .count = 512,
                          .group_offset = group_offset,
                          .readable = ne[0] * ne[1]});
        group_offset += ne[0] * ne[1];
      }
      continue;
    }
    for (const char* proj : {"gate", "up", "down"}) {
      const bool down = std::string_view(proj) == "down";
      ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {512});
      arrays.push_back({.roles = {std::format("{}ffn_{}_exps.weight", n, proj)},
                        .plain = false,
                        .type = "NVFP4",
                        .ne = down ? std::vector<std::uint64_t>{640, 2560}
                                   : std::vector<std::uint64_t>{2560, 640},
                        .expert_array = true,
                        .count = 512,
                        .group_offset = group_offset,
                        .readable = kExpertSlice + (down ? kDownOverRead : 0)});
      group_offset += kExpertSlice;
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// Hash constants of the checkpoint's shape: 16 heads of 20,000,008 rows.
md::Qwen38PleHash Hash() {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  std::vector<std::int64_t> m = {3, 5, 7};
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  return md::CheckQwen38PleHash(p, m, offsets, vocab, kTableRows).value();
}

TEST(Qwen38Test, TheProfileIsQwen38FlashNext) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  EXPECT_EQ(p.layers, 48U);
  EXPECT_EQ(p.hc_width(), 10240U);
  EXPECT_EQ(p.conv_channels(), 10240U);
  EXPECT_EQ(p.ple_heads(), 16U);
  EXPECT_EQ(p.ple_width(), 2560U);
  EXPECT_EQ(p.ple_history(), 9U);
  // 12 x (3 linear, then QSA).
  int qsa = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    qsa += p.linear(il) ? 0 : 1;
    EXPECT_EQ(p.linear(il), il % 4 != 3) << il;
  }
  EXPECT_EQ(qsa, 12);
  EXPECT_TRUE(p.linear(p.ple_layer));
}

TEST(Qwen38Test, BindsTheArtifactsTensorsAndRefusesWhatDiffers) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto bound = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  EXPECT_TRUE(bound->layers[0].linear);
  EXPECT_FALSE(bound->layers[3].linear);
  EXPECT_TRUE(bound->layers[3].q.codes.plain);
  EXPECT_EQ(bound->layers[3].q.codes.ne, (std::vector<std::uint64_t>{2560, 12288}));
  EXPECT_EQ(bound->ple_table.ne[1], kTableRows);
  EXPECT_EQ(bound->layers[47].down_exps.type, "NVFP4");

  EXPECT_NE(Why(md::BindQwen38(p, "qwen3next", resources)).find("qwen4exp"), std::string::npos);
  auto missing = resources;
  missing.erase(missing.begin() + 3);
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", missing)).find("has no"), std::string::npos);
  auto extra = resources;
  extra.push_back({.roles = {"blk.0.surprise"}, .plain = false, .type = "F32", .ne = {4}});
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", extra)).find("does not read"), std::string::npos);
  auto reshaped = resources;
  for (md::Qwen38Resource& r : reshaped) {
    if (r.roles[0] == "blk.3.attn_k.weight") {
      r.ne = {2560, 256};
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", reshaped)).find("blk.3.attn_k.weight"),
            std::string::npos);
  auto retyped = resources;
  for (md::Qwen38Resource& r : retyped) {
    if (r.roles[0] == "blk.3.attn_k.weight") {
      r.plain = false;  // the same bytes, but claimed as a GGML tensor
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", retyped)).find("plain"), std::string::npos);
  auto few = resources;
  for (md::Qwen38Resource& r : few) {
    if (r.expert_array) {
      r.count = 256;
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", few)).find("experts"), std::string::npos);
}

TEST(Qwen38Test, BindsTheCutlassExpertLayoutPackedFromEachGroupsStart) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p, true);
  auto bound = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  EXPECT_TRUE(bound->cutlass());
  const md::Qwen38Layer& l = bound->layers[47];
  EXPECT_EQ(l.gate_up_codes.type, "I8");
  EXPECT_EQ(l.gate_up_scales.group_offset, 1638400U);
  EXPECT_EQ(l.down_codes.group_offset, 1843200U);
  EXPECT_EQ(l.down_scales.group_offset, 2662400U);
  EXPECT_EQ(l.down_scales.ne, (std::vector<std::uint64_t>{512, 200}));
  EXPECT_EQ(l.expert_arrays(true).size(), 4U);
  // A gap between the arrays, a wrong atom count, and a layer left in
  // GGML's layout are refused.
  auto gap = resources;
  for (md::Qwen38Resource& r : gap) {
    if (r.roles[0] == "blk.5.ffn_down_exps.scales") {
      r.group_offset += 256;
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", gap)).find("layer 5"), std::string::npos);
  auto atoms = resources;
  for (md::Qwen38Resource& r : atoms) {
    if (r.roles[0] == "blk.2.ffn_gate_up_exps.scales") {
      r.ne = {512, 200};
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", atoms)).find("blk.2.ffn_gate_up_exps.scales"),
            std::string::npos);
  auto mixed = resources;
  std::erase_if(mixed, [](const md::Qwen38Resource& r) {
    return r.roles[0].starts_with("blk.9.ffn_") && r.expert_array;
  });
  const std::vector<md::Qwen38Resource> ggml = ArtifactLike(p);
  for (const md::Qwen38Resource& r : ggml) {
    if (r.expert_array && r.roles[0].starts_with("blk.9.")) {
      mixed.push_back(r);
    }
  }
  EXPECT_FALSE(md::BindQwen38(p, "qwen4exp", mixed).has_value());
}

TEST(Qwen38Test, TheNgramHashMustStayInsideTheTable) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<std::int64_t> m = {3, 5, 7};
  std::vector<std::int64_t> offsets(16, 0);
  std::vector<std::int64_t> vocab(16, 100);
  EXPECT_TRUE(md::CheckQwen38PleHash(p, m, offsets, vocab, 100).has_value());
  vocab[5] = 101;
  EXPECT_NE(Why(md::CheckQwen38PleHash(p, m, offsets, vocab, 100)).find("head 5"),
            std::string::npos);
  vocab[5] = 0;
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, 100).has_value());
  vocab[5] = 100;
  offsets[2] = -1;
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, 100).has_value());
  offsets[2] = 0;
  // Past the I32 row index, whatever the table holds.
  offsets[0] = std::numeric_limits<std::int32_t>::max();
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, std::uint64_t{1} << 40).has_value());
  offsets[0] = 0;
  EXPECT_FALSE(
      md::CheckQwen38PleHash(p, std::vector<std::int64_t>{3, 5}, offsets, vocab, 100).has_value());
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, 0).has_value());
  // A hash not made by the check is checked again before any row is hashed.
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  const std::vector<std::int32_t> history(4, 1000);
  md::Qwen38PleHash built = md::CheckQwen38PleHash(p, m, offsets, vocab, 100).value();
  ASSERT_TRUE(md::Qwen38Chunk(p, *state, built, history, 0, 4).has_value());
  built.vocab[7] = 0;
  EXPECT_NE(Why(md::Qwen38Chunk(p, *state, built, history, 0, 4)).find("head 7"),
            std::string::npos);
  built.vocab[7] = 100;
  built.offsets[3] = 1;
  EXPECT_NE(Why(md::Qwen38Chunk(p, *state, built, history, 0, 4)).find("head 3"),
            std::string::npos);
  built.offsets[3] = 0;
  built.table_rows = 99;
  EXPECT_FALSE(md::Qwen38Chunk(p, *state, built, history, 0, 4).has_value());
}

TEST(Qwen38Test, NgramRowsFollowLlamaCppsHash) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const md::Qwen38PleHash h = Hash();
  const std::vector<std::int32_t> history = {11, 22, 33, p.ple_eos, 44, 55};
  const auto rows_of = [&](std::uint64_t mixed2, std::uint64_t mixed3) {
    std::vector<std::int32_t> want;
    for (std::uint32_t head = 0; head < 16; ++head) {
      const std::uint64_t mixed = head < 8 ? mixed2 : mixed3;
      want.push_back(static_cast<std::int32_t>((mixed % h.vocab[head]) + h.offsets[head]));
    }
    return want;
  };
  const auto eos = static_cast<std::uint64_t>(p.ple_eos);
  // Position 0: both predecessors are missing and read as EOS.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 0),
            rows_of((11UL * 3UL) ^ (eos * 5UL), (11UL * 3UL) ^ (eos * 5UL) ^ (eos * 7UL)));
  // Position 2: a full window.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 2),
            rows_of((33UL * 3UL) ^ (22UL * 5UL), (33UL * 3UL) ^ (22UL * 5UL) ^ (11UL * 7UL)));
  // Position 3: the token is EOS; its own EOS does not cut its context.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 3),
            rows_of((eos * 3UL) ^ (33UL * 5UL), (eos * 3UL) ^ (33UL * 5UL) ^ (22UL * 7UL)));
  // Position 5: the EOS two back cuts the window there.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 5),
            rows_of((55 * 3) ^ (44 * 5), (55 * 3) ^ (44 * 5) ^ (eos * 7)));
}

TEST(Qwen38Test, TheStateIsBoundedAndSized) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4000, 512);
  ASSERT_TRUE(s.has_value()) << Why(s);
  EXPECT_EQ(s->cells, 4096U);
  using K = md::Qwen38StateTensor::Kind;
  const auto& k = s->tensors[static_cast<std::size_t>(s->Find(3, K::kK))];
  EXPECT_EQ(k.bytes, 512ULL * 4096 * 2);
  const auto& rec = s->tensors[static_cast<std::size_t>(s->Find(0, K::kRecurrent))];
  EXPECT_EQ(rec.bytes, 128ULL * 128 * 48 * 4);
  const auto& conv = s->tensors[static_cast<std::size_t>(s->Find(0, K::kConv))];
  EXPECT_EQ(conv.bytes, 3ULL * 10240 * 4);
  const auto& ple = s->tensors[static_cast<std::size_t>(s->Find(1, K::kPleConv))];
  EXPECT_EQ(ple.bytes, 9ULL * 10240 * 4);
  // Each QSA layer's block keys: BF16, a row per 4 cells.
  const auto& blocks = s->tensors[static_cast<std::size_t>(s->Find(3, K::kIndexerBlocks))];
  EXPECT_TRUE(blocks.f16);
  EXPECT_EQ(blocks.ne1, 1024U);
  EXPECT_EQ(blocks.bytes, 128ULL * 1024 * 2);
  EXPECT_EQ(s->Find(0, K::kIndexerBlocks), -1);
  EXPECT_EQ(s->Find(0, K::kPleConv), -1);
  EXPECT_EQ(s->Find(0, K::kK), -1);
  std::uint64_t kv = 0;
  for (const auto& t : s->tensors) {
    EXPECT_EQ(t.offset % 256, 0U);
    EXPECT_LE(t.offset + t.bytes, s->bytes);
    kv += t.kind == K::kK || t.kind == K::kV ? t.bytes : 0;
  }
  const auto reps = s->Representations();
  ASSERT_EQ(reps.size(), 3U);
  EXPECT_EQ(reps[0].block_bytes.value(), kv);
  EXPECT_FALSE(md::Qwen38State(p, 0, 1).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 16, 0).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 16, 32).has_value());
  // Chunks wider than the bound, or whose F32 [n_kv, rows] planes would pass
  // I32 bytes (RE-037: at the configured maximum, 2,048 rows are one byte over).
  EXPECT_FALSE(md::Qwen38State(p, 65536, md::kQwen38MaxRows + 1).has_value());
  EXPECT_TRUE(md::Qwen38State(p, 262144, 2047).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 262144, 2048).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 262400, 2048).has_value());
  // The fast graph's state, which builds no such plane, has no such bound.
  EXPECT_TRUE(md::Qwen38State(p, 262144, md::kQwen38MaxRows, false).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 262144, md::kQwen38MaxRows + 1, false).has_value());
  EXPECT_FALSE(
      md::Qwen38State(p, static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()), 1)
          .has_value());
}

// The widest chunk the state admits, which the runtime's prefill chunk
// stays within (and below the context: the runtime's chunk at the minimum
// context of 512 is 504 rows, which leaves the drafter's prefill pass its
// row before the chunk). Each is the edge: one more row is refused.
TEST(Qwen38Test, TheWidestChunkIsBounded) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  for (const auto& [context, most] :
       {std::pair{512U, 512U}, std::pair{511U, 511U}, std::pair{8704U, md::kQwen38MaxRows},
        std::pair{65536U, 8191U}, std::pair{131072U, 4095U}, std::pair{262144U, 2047U},
        std::pair{262400U, 2046U}}) {
    SCOPED_TRACE(context);
    EXPECT_EQ(md::Qwen38MostRows(context), most);
    EXPECT_TRUE(md::Qwen38State(p, context, most).has_value());
    EXPECT_FALSE(md::Qwen38State(p, context, most + 1).has_value());
    EXPECT_TRUE(md::Qwen38State(p, context, most - 1).has_value());
  }
  // Without the host's masks, only the context and kQwen38MaxRows.
  EXPECT_EQ(md::Qwen38MostRows(262144, false), md::kQwen38MaxRows);
  EXPECT_EQ(md::Qwen38MostRows(4000, false), 4000U);
  EXPECT_EQ(md::Qwen38MostRows(0), 0U);
  EXPECT_EQ(md::Qwen38MostRows(0x7FFFFF01U), 0U);
}

TEST(Qwen38Test, AChunksMaskAndPositionsAreCausal) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  const md::Qwen38PleHash h = Hash();
  std::vector<std::int32_t> history(40, 7);
  auto c = md::Qwen38Chunk(p, *s, h, history, 37, 3);
  ASSERT_TRUE(c.has_value()) << Why(c);
  EXPECT_EQ(c->n_kv, 256U);
  EXPECT_FALSE(c->qsa_select);
  EXPECT_EQ(c->cells, (std::vector<std::int64_t>{37, 38, 39}));
  EXPECT_EQ(c->positions,
            (std::vector<std::int32_t>{37, 38, 39, 37, 38, 39, 37, 38, 39, 37, 38, 39}));
  for (std::uint32_t i = 0; i < 3; ++i) {
    for (std::uint32_t j = 0; j < 256; ++j) {
      const bool visible = j <= 37 + i;
      EXPECT_EQ(c->mask[(i * 256) + j], visible ? md::kQwen38HalfZero : md::kQwen38HalfNegInf);
      EXPECT_EQ(c->mask_f32[(i * 256) + j] == 0.0f, visible);
    }
  }
  EXPECT_EQ(c->ple_rows.size(), 48U);
  // Without the selection's masks a chunk that does not select still gets
  // the causal mask (the fast graph's attention reads it), and one that
  // selects gets neither, nor the block tables (the fast graph selects from
  // the cached block keys on the device).
  auto unmasked = md::Qwen38Chunk(p, *s, h, history, 37, 3, false);
  ASSERT_TRUE(unmasked.has_value()) << Why(unmasked);
  EXPECT_EQ(unmasked->mask, c->mask);
  EXPECT_TRUE(unmasked->mask_f32.empty());
  std::vector<std::int32_t> selecting(2400, 7);
  auto past = md::Qwen38Chunk(p, *s, h, selecting, 2000, 400, false);
  ASSERT_TRUE(past.has_value()) << Why(past);
  EXPECT_TRUE(past->qsa_select);
  EXPECT_TRUE(past->mask.empty());
  EXPECT_TRUE(past->mask_f32.empty());
  EXPECT_EQ(past->qsa.blocks, 2560U / 4);
  EXPECT_TRUE(past->qsa.bias.empty());
  EXPECT_TRUE(past->qsa.cell_block.empty());
  EXPECT_TRUE(past->qsa.block_cells.empty());
  // Refusals: history of the wrong length, a chunk past the context, a
  // token outside the vocabulary.
  EXPECT_FALSE(md::Qwen38Chunk(p, *s, h, history, 36, 3).has_value());
  std::vector<std::int32_t> long_history(4097, 7);
  EXPECT_FALSE(md::Qwen38Chunk(p, *s, h, long_history, 4096, 1).has_value());
  history[39] = static_cast<std::int32_t>(p.vocab);
  EXPECT_FALSE(md::Qwen38Chunk(p, *s, h, history, 37, 3).has_value());
}

TEST(Qwen38Test, PastTheBudgetQsaSelectsWholeBlocksAndTheTail) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  const md::Qwen38PleHash h = Hash();
  // 2,302 positions: n_kv 2,304 > 2,051, so the indexer selects.
  std::vector<std::int32_t> history(2302, 9);
  auto c = md::Qwen38Chunk(p, *s, h, history, 2300, 2);
  ASSERT_TRUE(c.has_value()) << Why(c);
  EXPECT_EQ(c->n_kv, 2304U);
  ASSERT_TRUE(c->qsa_select);
  const md::Qwen38QsaInputs& q = c->qsa;
  EXPECT_EQ(q.blocks, 576U);
  // 575 full blocks (positions 0..2299); cells 2300 and 2301 and the empty
  // cells point at the spare block 575.
  EXPECT_EQ(q.cell_block[0], 0);
  EXPECT_EQ(q.cell_block[2299], 574);
  EXPECT_EQ(q.cell_block[2300], 575);
  EXPECT_EQ(q.cell_block[2303], 575);
  EXPECT_EQ(q.block_cells[(574 * 4) + 3], 2299);
  EXPECT_EQ(q.block_cells[575UL * 4UL], 0);
  EXPECT_EQ(q.block_pos[574], 2296);
  EXPECT_EQ(q.block_pos[(3 * 576) + 574], 2296);
  // Token at 2300: its tail starts at 2300, so every full block is scored
  // (0) and the spare one, the tail, kept (1e9).
  for (std::uint32_t row = 0; row < 2; ++row) {
    const float* bias = q.bias.data() + (std::size_t{row} * 576);
    EXPECT_EQ(bias[0], 0.0f);
    EXPECT_EQ(bias[574], 0.0f);
    EXPECT_EQ(bias[575], 1e9f);
  }
  // Mid-block, in a prefill chunk: block 574 (positions 2296-2299) is the
  // tail of a token at 2298 (kept, 1e9) but a scored block for one at 2299,
  // whose tail starts at 2300.
  std::vector<std::int32_t> prefill(2302, 9);
  auto d = md::Qwen38Chunk(p, *s, h, prefill, 2296, 6);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->qsa.bias[(std::size_t{2} * 576) + 574], 1e9f);  // token 2298
  EXPECT_EQ(d->qsa.bias[(std::size_t{3} * 576) + 574], 0.0f);  // token 2299
  EXPECT_EQ(d->qsa.bias[(std::size_t{3} * 576) + 573], 0.0f);
}

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

TEST(Qwen38Test, TheChunkGraphIsPlannedByThisModulesImplementations) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  const md::Qwen38PleHash h = Hash();
  const std::vector<std::uint64_t> strides(p.layers, kExpertStride);
  // (n_past, rows, fused, exact, cutlass): the fused graph's fast form (the
  // default) and its reference form.
  for (const auto& [n_past, rows, fused, exact, cutlass] : {std::tuple{0U, 37U, true, true, false},
                                                            {37U, 1U, true, true, false},
                                                            {2800U, 512U, true, true, false},
                                                            {4095U, 1U, true, true, false},
                                                            {0U, 37U, false, false, false},
                                                            {37U, 1U, false, false, false},
                                                            {2800U, 512U, false, false, false},
                                                            {4095U, 1U, false, false, false},
                                                            {40U, 12U, true, true, false},
                                                            {0U, 37U, true, true, true},
                                                            {37U, 1U, true, true, true},
                                                            {2800U, 512U, true, true, true},
                                                            {40U, 8U, true, true, true},
                                                            {0U, 37U, true, false, true},
                                                            {37U, 1U, true, false, true},
                                                            {2800U, 512U, true, false, true},
                                                            {40U, 8U, true, false, true},
                                                            {40U, 12U, true, false, true},
                                                            {0U, 2U, true, false, true},
                                                            {0U, 37U, true, false, false}}) {
    std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
    auto chunk = md::Qwen38Chunk(p, *state, h, history, n_past, rows);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, rows);
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto graph =
        kg::BuildQwen38Graph(*arena, p, *binding, shape,
                             {.expert_stride = strides,
                              .fused = fused,
                              .exact = exact,
                              .experts = cutlass ? kg::Qwen38GraphOptions::Experts::kCutlass
                                                 : kg::Qwen38GraphOptions::Experts::kGgml});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    const bool fast = fused && !exact;
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
    for (const std::string_view name : {kg::kNvfp4RowsName, kg::kSetRowsExtName, kg::kConcatName,
                                        kg::kSumRowsName, kg::kRepeatName}) {
      EXPECT_TRUE(used.contains(name)) << name << " at " << rows << " rows";
    }
    // The fast form's selection keeps its cells and attention reads them
    // alone; elsewhere GGML's flash attention reads every cell under a mask.
    const bool sparse = fast && chunk->qsa_select;
    EXPECT_EQ(used.contains(kg::kFlashAttnMmaName), !sparse) << rows << " at " << n_past;
    EXPECT_EQ(used.contains(kg::kQsaAttnName), sparse) << rows << " at " << n_past;
    // Every fast chunk caches the block keys it completes.
    EXPECT_EQ(used.contains(kg::kQsaPoolName), fast) << rows << " at " << n_past;
    // The fast form routes, and normalizes and rotates QSA's heads, in
    // jitLLM's fusions; the others in GGML's nodes.
    EXPECT_EQ(used.contains(kg::kArgsortName), !fast) << rows;
    EXPECT_EQ(used.contains(kg::kRopeExtName), !fast) << rows;
    EXPECT_EQ(used.contains(kg::kMoeRouterName), fast) << rows;
    EXPECT_EQ(used.contains(kg::kQsaPrepName), fast) << rows;
    // The MXFP8 products: the vector product up to 8 rows; past them the
    // tensor-core product over quantized activations (the fast form) or the
    // weights dequantized to BF16.
    const bool wide = rows > 8;
    EXPECT_EQ(used.contains(kg::kMxfp8MulMatVecName), !wide) << rows;
    EXPECT_EQ(used.contains(kg::kMxfp8DequantName), wide && !fast) << rows;
    for (const std::string_view name : {kg::kMxfp8GemmName, kg::kMxfp8QuantizeName,
                                        kg::kMxfp8SwizzleName, kg::kQsaGateQuantizeName}) {
      EXPECT_EQ(used.contains(name), wide && fast) << name << " at " << rows << " rows";
    }
    // The routed experts: GGML's mul_mat_id over GGML's layout; over the
    // CUTLASS layout, the vector products up to 8 rows, else the grouped
    // GEMM path.
    const bool vector = rows <= 8;
    EXPECT_EQ(used.contains(kg::kMulMatIdVecQ), !cutlass && vector) << rows;
    EXPECT_EQ(used.contains(kg::kMulMatIdQ), !cutlass && !vector) << rows;
    EXPECT_EQ(used.contains(kg::kMoeGemvName), cutlass && vector) << rows;
    for (const std::string_view name : {kg::kMoeRouteName, kg::kMoeQuantizeName, kg::kMoeGemmName,
                                        kg::kMoeGluQuantizeName, kg::kMoeCombineSortedName}) {
      EXPECT_EQ(used.contains(name), cutlass && !vector) << name << " at " << rows << " rows";
    }
    // The fusions replace the hyper-connections' and the MoE output's
    // elementwise nodes; past 16 rows the float products read BF16 once.
    // The fast form's mixes combine, normalize and mix in its own three
    // (the n-gram layer's and the head's streams still combined apart).
    EXPECT_EQ(used.contains(kg::kHcCombineName), fused) << rows;
    for (const std::string_view name : {kg::kHcNormName, kg::kHcMixName}) {
      EXPECT_EQ(used.contains(name), fused && !fast) << name << " at " << rows << " rows";
    }
    for (const std::string_view name : {kg::kHcPrepName, kg::kHcLoName, kg::kHcMixBf16Name}) {
      EXPECT_EQ(used.contains(name), fast) << name << " at " << rows << " rows";
    }
    // (Over the CUTLASS layout decode's SwiGLU is the vector product's.)
    EXPECT_EQ(used.contains(kg::kMoeGluName), fused && !cutlass) << rows;
    EXPECT_EQ(used.contains(kg::kMoeCombineName), fused && (!cutlass || vector)) << rows;
    // The linear-attention layers: GGML's convolution unless fused over
    // whole histories of rows; the recurrence as the plan picks by rows.
    // (The fast form's at every width: its history reads the old one below 3
    // rows.)
    EXPECT_EQ(used.contains(kg::kGdnConvName), fused && (rows >= 3 || fast)) << rows;
    EXPECT_EQ(used.contains(kg::kSsmConvName), !fused || (rows < 3 && !fast)) << rows;
    EXPECT_EQ(used.contains(kg::kGdnNormGateName), fused) << rows;
    EXPECT_EQ(used.contains(kg::kGdnHistoryName), fast) << rows;
    // (The fast form, up to 16 rows, writes the state in place: jitllm.gdn.step.)
    const bool step = fast && static_cast<std::int64_t>(rows) <= kg::kGatedDeltaNetLanesTokens;
    EXPECT_EQ(used.contains(kg::kGdnStepName), step) << rows;
    if (!step) {
      EXPECT_TRUE(used.contains(static_cast<std::int64_t>(rows) > kg::kGatedDeltaNetLanesTokens
                                    ? kg::kGatedDeltaNetLanesName
                                    : kg::kGatedDeltaNetColumnsName))
          << rows;
    }
    // (The fast form's mixes give their BF16 themselves, and its
    // hyper-connection products are BF16 at every width.)
    const bool bf16 = fused && static_cast<std::int64_t>(rows) > kg::kQwen38Bf16Rows;
    EXPECT_EQ(used.contains(kg::kBf16Name), bf16 && !fast) << rows;
    EXPECT_EQ(used.contains(kg::kGemmBf16Name), bf16 || fast) << rows;
    EXPECT_EQ(used.contains(kg::kTopKName), chunk->qsa_select && !fast) << rows << " at " << n_past;
    EXPECT_EQ(used.contains(kg::kQsaTopKName), chunk->qsa_select && fast)
        << rows << " at " << n_past;
    // The fast form's selection needs no host masks.
    EXPECT_EQ(graph->mask == nullptr, chunk->qsa_select && fast) << rows << " at " << n_past;
    EXPECT_EQ(graph->mask_f32 != nullptr, chunk->qsa_select && !fast) << rows << " at " << n_past;
    auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
    ASSERT_TRUE(placed.has_value()) << Why(placed);
    EXPECT_NE(graph->Named("l_last-47"), nullptr);
    EXPECT_NE(graph->Named("result_output"), nullptr);
    EXPECT_EQ(graph->logits->ne[0], 248320);
    EXPECT_EQ(graph->logits->ne[1], rows);
  }
  // At depth (here past one 8,192-block tile, and a whole 8,192-row chunk
  // at the configured maximum, which the host's masks could not take:
  // RE-037) the fast form still selects on the device and reads no host
  // mask; the reference form refuses the chunk whose masks would pass
  // GGML's strides.
  for (const auto& [context, n_past, rows] :
       {std::tuple{33280U, 33000U, 1U}, std::tuple{262144U, 253952U, md::kQwen38MaxRows}}) {
    auto long_state = md::Qwen38State(p, context, rows, false);
    ASSERT_TRUE(long_state.has_value()) << Why(long_state);
    std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
    auto chunk = md::Qwen38Chunk(p, *long_state, h, history, n_past, rows, false);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    ASSERT_TRUE(chunk->qsa_select);
    EXPECT_GT(chunk->qsa.blocks, kg::kQsaTopKTile);
    const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*long_state, *chunk, 1);
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto graph = kg::BuildQwen38Graph(
        *arena, p, *binding, shape,
        {.expert_stride = strides, .experts = kg::Qwen38GraphOptions::Experts::kCutlass});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    EXPECT_EQ(graph->mask, nullptr);
    EXPECT_EQ(graph->mask_f32, nullptr);
    EXPECT_EQ(graph->cell_block, nullptr);
    std::uint64_t next = std::uint64_t{1} << 40U;
    for (ggml_tensor* node : graph->nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE && src->view_src == nullptr &&
            src->data == nullptr) {
          kg::TensorArena::Bind(src, next);
          next += ((ggml_nbytes(src) + 255) / 256 * 256) + 256;
        }
      }
    }
    kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
    auto plan = kg::PlanGraph(graph->nodes, false, ModelDevice());
    ASSERT_TRUE(plan.has_value()) << Why(plan);
    std::set<std::string_view> used;
    for (const auto& step : plan->steps) {
      used.insert(step.implementation);
    }
    EXPECT_FALSE(used.contains(kg::kTopKName));
    EXPECT_FALSE(used.contains(kg::kFlashAttnMmaName));
    for (const std::string_view name :
         {kg::kQsaPoolName, kg::kQsaTopKName, kg::kQsaAttnName, kg::kQsaPrepName}) {
      EXPECT_TRUE(used.contains(name)) << name << " at " << context;
    }
    auto reference = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(reference.has_value());
    const auto exact = kg::BuildQwen38Graph(*reference, p, *binding, shape,
                                            {.expert_stride = strides,
                                             .exact = true,
                                             .experts = kg::Qwen38GraphOptions::Experts::kCutlass});
    EXPECT_EQ(exact.has_value(), rows == 1) << context;
    if (!exact) {
      EXPECT_NE(Why(exact).find("RE-037"), std::string::npos) << Why(exact);
    }
  }
}

TEST(Qwen38Test, ExpertStridesAreWholeBlocks) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value());
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(1, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 0, 1);
  ASSERT_TRUE(chunk.has_value());
  const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, 1);
  auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(arena.has_value());
  std::vector<std::uint64_t> strides(p.layers, 2768905);  // not whole 36-byte blocks
  const auto torn = kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(torn).find("expert stride"), std::string::npos) << Why(torn);
  auto unaligned = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(unaligned.has_value());
  strides.assign(p.layers, 2768904);  // whole blocks, but not 16-byte aligned
  const auto skew =
      kg::BuildQwen38Graph(*unaligned, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(skew).find("16-byte aligned"), std::string::npos) << Why(skew);
  auto again = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(again.has_value());
  strides.assign(3, kExpertStride);
  const auto few = kg::BuildQwen38Graph(*again, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(few).find("per layer"), std::string::npos) << Why(few);
}

// A shape whose QSA selection is not the indexer budget's is refused: past
// the budget without selection, selection within it, or blocks that do not
// cover the cells.
TEST(Qwen38Test, TheGraphRefusesASelectionThatIsNotTheBudgets) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value());
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(2800, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 2799, 1);
  ASSERT_TRUE(chunk.has_value());
  ASSERT_TRUE(chunk->qsa_select);
  const kg::Qwen38ChunkShape past = kg::Qwen38ShapeOf(*state, *chunk, 1);
  const std::vector<std::uint64_t> strides(p.layers, kExpertStride);
  const auto build = [&](const kg::Qwen38ChunkShape& shape) {
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    EXPECT_TRUE(arena.has_value());
    return Why(kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides}));
  };
  EXPECT_EQ(build(past), "");
  kg::Qwen38ChunkShape s = past;
  s.qsa_select = false;
  s.qsa_blocks = 0;
  EXPECT_NE(build(s).find("indexer budget"), std::string::npos);
  s = past;
  s.qsa_blocks -= 1;
  EXPECT_NE(build(s).find("indexer budget"), std::string::npos);
  s = past;
  s.n_kv = 2048;
  EXPECT_NE(build(s).find("indexer budget"), std::string::npos);
}

// The down projection's 640-element rows read past each slice; only a
// stride that holds the artifact's readable bytes for every expert, the
// last one's included, marks them readable (else the products refuse them).
TEST(Qwen38Test, ShortExpertRowsAreReadableOnlyInsideTheStride) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value());
  EXPECT_EQ(binding->layers[0].down_exps.group_offset, 2 * kExpertSlice);
  EXPECT_EQ(binding->layers[0].down_exps.readable, kExpertSlice + kDownOverRead);
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(1, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 0, 1);
  ASSERT_TRUE(chunk.has_value());
  const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, 1);
  // Whether every NVFP4 weight the graph reads with short rows is marked,
  // and how many there are.
  const auto marked = [](const kg::Qwen38Graph& g) {
    std::set<const ggml_tensor*> short_rows;
    bool all = true;
    for (const ggml_tensor* node : g.nodes) {
      for (const ggml_tensor* src : node->src) {
        if (src != nullptr && src->type == GGML_TYPE_NVFP4) {
          if (src->ne[0] % 512 != 0) {
            short_rows.insert(src);
            all = all && kg::RowPaddingReadable(src);
          } else {
            EXPECT_FALSE(kg::RowPaddingReadable(src));
          }
        }
      }
    }
    return std::pair{all, short_rows.size()};
  };
  // 2,765,016 bytes are used by the group; each stride is a multiple of 144
  // (2,764,944 the largest short of it).
  for (const auto& [stride, readable] : {std::pair{kExpertStride, true},
                                         {std::uint64_t{2764944}, false},
                                         {std::uint64_t{0}, false}}) {
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    std::vector<std::uint64_t> strides;
    if (stride != 0) {
      strides.assign(p.layers, stride);
    }
    auto graph = kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    const auto [all, count] = marked(*graph);
    EXPECT_EQ(count, p.layers) << stride;
    EXPECT_EQ(all, readable) << stride;
  }
}

// ---------------------------------------------------------------- the MTP drafter

// The drafter's resources (docs/experiments/artifact-layout/modelopt_qwen38.py
// plan_mtp): BF16 but for its F32 norms and its CUTLASS-layout experts.
std::vector<md::Qwen38Resource> MtpLike() {
  std::vector<md::Qwen38Resource> r;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  ggml("fc_embd.weight", "BF16", {2560, 2560});
  ggml("fc_hidden.weight", "BF16", {2560, 2560});
  ggml("norm_embd.weight", "F32", {2560});
  ggml("norm_hidden.weight", "F32", {10240});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "BF16", {10240, 320});
  ggml("output_hc_up.weight", "BF16", {320, 10240});
  for (const char* kind : {"attn", "ffn"}) {
    ggml(std::format("blk.0.hc_{}_norm.weight", kind), "F32", {10240});
    ggml(std::format("blk.0.hc_{}_down.weight", kind), "BF16", {10240, 320});
    ggml(std::format("blk.0.hc_{}_up.weight", kind), "BF16", {320, 10240});
    ggml(std::format("blk.0.hc_{}_inject.weight", kind), "BF16", {10240, 4});
  }
  ggml("blk.0.attn_q.weight", "BF16", {2560, 12288});
  ggml("blk.0.attn_k.weight", "BF16", {2560, 512});
  ggml("blk.0.attn_v.weight", "BF16", {2560, 512});
  ggml("blk.0.attn_output.weight", "BF16", {6144, 2560});
  ggml("blk.0.attn_q_norm.weight", "F32", {256});
  ggml("blk.0.attn_k_norm.weight", "F32", {256});
  ggml("blk.0.indexer.qk_proj.weight", "BF16", {2560, 640});
  ggml("blk.0.indexer.q_norm.weight", "F32", {128});
  ggml("blk.0.indexer.k_norm.weight", "F32", {128});
  ggml("blk.0.ffn_gate_inp.weight", "BF16", {2560, 512});
  ggml("blk.0.ffn_gate_inp_shexp.weight", "BF16", {2560});
  ggml("blk.0.ffn_gate_shexp.weight", "BF16", {2560, 640});
  ggml("blk.0.ffn_up_shexp.weight", "BF16", {2560, 640});
  ggml("blk.0.ffn_down_shexp.weight", "BF16", {640, 2560});
  for (const char* proj : {"gate", "up", "down"}) {
    ggml(std::format("blk.0.ffn_{}_exps.weight_scale_2", proj), "F32", {512});
  }
  std::uint64_t group_offset = 0;
  for (const auto& [name, ne] :
       {std::pair{"gate_up_exps.codes", std::vector<std::uint64_t>{1280, 1280}},
        std::pair{"gate_up_exps.scales", std::vector<std::uint64_t>{512, 400}},
        std::pair{"down_exps.codes", std::vector<std::uint64_t>{320, 2560}},
        std::pair{"down_exps.scales", std::vector<std::uint64_t>{512, 200}}}) {
    r.push_back({.roles = {std::format("blk.0.ffn_{}", name)},
                 .plain = false,
                 .type = "I8",
                 .ne = ne,
                 .expert_array = true,
                 .count = 512,
                 .group_offset = group_offset,
                 .readable = ne[0] * ne[1]});
    group_offset += ne[0] * ne[1];
  }
  return r;
}

TEST(Qwen38Test, SelectedDraftHeadRequiresAMatchingTokenMap) {
  const auto& p = md::Qwen38Flash();
  auto resources = MtpLike();
  resources.push_back({.roles = {"draft_output.weight"}, .type = "BF16", .ne = {2560, 3}});
  EXPECT_FALSE(md::BindQwen38Mtp(p, "qwen4exp-mtp", resources));
  resources.push_back({.roles = {"draft_output.ids"}, .type = "I32", .ne = {1, 3}});
  auto bound = md::BindQwen38Mtp(p, "qwen4exp-mtp", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  EXPECT_TRUE(bound->selected_head());
  EXPECT_EQ(bound->draft_ids.ne, (std::vector<std::uint64_t>{1, 3}));
  for (auto wrong : {0U, 4U, p.vocab + 1}) {
    auto bad = resources;
    bad[bad.size() - 2].ne[1] = wrong;
    EXPECT_FALSE(md::BindQwen38Mtp(p, "qwen4exp-mtp", bad));
  }
  resources.back().type = "F32";
  EXPECT_FALSE(md::BindQwen38Mtp(p, "qwen4exp-mtp", resources));
  resources.back().type = "I32";
  resources.erase(resources.end() - 2);
  EXPECT_FALSE(md::BindQwen38Mtp(p, "qwen4exp-mtp", resources));
}

TEST(Qwen38Test, DraftIdsAreBoundedAndPreserveLowestTokenTieBreaking) {
  EXPECT_TRUE(md::CheckQwen38DraftIds(std::vector<std::int32_t>{0, 17, 248319}, 248320));
  for (const auto& bad : {std::vector<std::int32_t>{}, {-1, 3}, {0, 248320}, {3, 3}, {7, 2}}) {
    EXPECT_FALSE(md::CheckQwen38DraftIds(bad, 248320));
  }
}

TEST(Qwen38Test, BindsTheMtpDrafterAndRefusesWhatDiffers) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = MtpLike();
  auto bound = md::BindQwen38Mtp(p, "qwen4exp-mtp", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  const md::Qwen38Layer& l = bound->layer;
  EXPECT_FALSE(l.linear);
  EXPECT_TRUE(l.q.is_bf16());
  EXPECT_EQ(l.q.bf16.ne, (std::vector<std::uint64_t>{2560, 12288}));
  EXPECT_TRUE(l.q.codes.type.empty());
  EXPECT_TRUE(l.down_shexp.is_bf16());
  EXPECT_EQ(l.down_scales.group_offset, 2662400U);
  EXPECT_EQ(bound->norm_hidden.ne, (std::vector<std::uint64_t>{10240}));
  // No token table or head of its own (the target's are bound), and the
  // target's architecture, a missing, an extra and a reshaped tensor are
  // refused, as is a drafter whose attention is MXFP8.
  EXPECT_NE(Why(md::BindQwen38Mtp(p, "qwen4exp", resources)).find("qwen4exp-mtp"),
            std::string::npos);
  auto missing = resources;
  missing.erase(missing.begin() + 1);
  EXPECT_NE(Why(md::BindQwen38Mtp(p, "qwen4exp-mtp", missing)).find("fc_hidden"),
            std::string::npos);
  auto extra = resources;
  extra.push_back(
      {.roles = {"token_embd.weight"}, .plain = false, .type = "BF16", .ne = {2560, 248320}});
  EXPECT_NE(Why(md::BindQwen38Mtp(p, "qwen4exp-mtp", extra)).find("does not read"),
            std::string::npos);
  auto mxfp8 = resources;
  for (md::Qwen38Resource& r : mxfp8) {
    if (r.roles[0] == "blk.0.attn_k.weight") {
      r = {.roles = {"blk.0.attn_k.weight"}, .plain = true, .type = "F8_E4M3", .ne = {2560, 512}};
    }
  }
  EXPECT_NE(Why(md::BindQwen38Mtp(p, "qwen4exp-mtp", mxfp8)).find("blk.0.attn_k.weight"),
            std::string::npos);
  auto gap = resources;
  for (md::Qwen38Resource& r : gap) {
    if (r.roles[0] == "blk.0.ffn_down_exps.codes") {
      r.group_offset += 256;
    }
  }
  EXPECT_FALSE(md::BindQwen38Mtp(p, "qwen4exp-mtp", gap).has_value());
}

TEST(Qwen38Test, TheMtpStateAndACommitAreSized) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto target = md::Qwen38State(p, 4000, 512);
  ASSERT_TRUE(target.has_value());
  auto s = md::Qwen38MtpStateOf(p, *target);
  ASSERT_TRUE(s.has_value()) << Why(s);
  EXPECT_EQ(s->cells, 4096U);
  EXPECT_EQ(s->hidden_rows, 513U);
  EXPECT_EQ(s->v - s->k, 512ULL * 4096 * 2);
  EXPECT_EQ(s->indexer - s->v, 512ULL * 4096 * 2);
  EXPECT_EQ(s->blocks - s->indexer, 128ULL * 4096 * 4);
  EXPECT_EQ(s->hidden - s->blocks, 128ULL * 1024 * 2);
  EXPECT_EQ(s->bytes - s->hidden, 10240ULL * 513 * 4);
  EXPECT_EQ(s->Representations().size(), 2U);
  auto c = md::Qwen38Commit(p, 4);
  ASSERT_TRUE(c.has_value()) << Why(c);
  EXPECT_EQ(c->layers.size(), 36U);
  EXPECT_EQ(c->layers[3], 4U);
  EXPECT_EQ(c->qkv(0) - c->conv_out(0), 10240ULL * 4 * 4);
  EXPECT_EQ(c->beta(0) - c->gate(0), 48ULL * 4 * 4);
  EXPECT_EQ(c->conv_out(1), c->layer_bytes);
  EXPECT_EQ(c->ple(), 36 * c->layer_bytes);
  EXPECT_EQ(c->bytes, c->ple() + (10240ULL * 4 * 4));
  EXPECT_EQ(c->layer_bytes % 256, 0U);
  EXPECT_FALSE(md::Qwen38Commit(p, 0).has_value());
  EXPECT_FALSE(md::Qwen38Commit(p, 9).has_value());
}

TEST(Qwen38Test, RowsReadingMoreCellsAreTheChunksRowsOverThem) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  std::vector<std::int32_t> history(40, 7);
  auto chunk = md::Qwen38Chunk(p, *s, Hash(), history, 37, 3);
  auto rows = md::Qwen38Rows(p, s->cells, 37, 3, 256, true);
  ASSERT_TRUE(chunk.has_value() && rows.has_value()) << Why(rows);
  EXPECT_EQ(rows->positions, chunk->positions);
  EXPECT_EQ(rows->cells, chunk->cells);
  EXPECT_EQ(rows->mask, chunk->mask);
  EXPECT_TRUE(rows->tokens.empty());
  EXPECT_TRUE(rows->ple_rows.empty());
  // A wider read (a draft's passes share one): the extra cells masked.
  auto wider = md::Qwen38Rows(p, s->cells, 37, 3, 512, false);
  ASSERT_TRUE(wider.has_value());
  EXPECT_EQ(wider->n_kv, 512U);
  for (std::uint32_t i = 0; i < 3; ++i) {
    for (std::uint32_t j = 0; j < 512; ++j) {
      EXPECT_EQ(wider->mask[(i * 512) + j] == md::kQwen38HalfZero, j <= 37 + i);
    }
  }
  EXPECT_FALSE(md::Qwen38Rows(p, s->cells, 250, 10, 256, false).has_value());   // reads too few
  EXPECT_FALSE(md::Qwen38Rows(p, s->cells, 0, 1, 300, false).has_value());      // not whole 256s
  EXPECT_FALSE(md::Qwen38Rows(p, s->cells, 4095, 2, 4096, false).has_value());  // past the cache
}

// The drafter's graph and a verify's, planned by this module's
// implementations: the drafter's BF16 products GGML's float product or
// jitllm.gemm.bf16, no MXFP8 product, its heads' drafts by jitllm.argmax;
// a verify saving its rows' inputs and writing no recurrent state.
TEST(Qwen38Test, TheDrafterAndAVerifyArePlannedByThisModulesImplementations) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto target = md::BindQwen38(p, "qwen4exp", ArtifactLike(p, true));
  auto drafter = md::BindQwen38Mtp(p, "qwen4exp-mtp", MtpLike());
  ASSERT_TRUE(target.has_value() && drafter.has_value());
  constexpr std::uint64_t kStride = 2764800;
  const auto plan_of = [&](std::vector<ggml_tensor*>& nodes,
                           const std::vector<ggml_tensor*>& ins) -> std::set<std::string_view> {
    std::uint64_t next = std::uint64_t{1} << 40U;
    const auto bind_leaf = [&](ggml_tensor* t) {
      if (t != nullptr && t->data == nullptr) {
        kg::TensorArena::Bind(t, next);
        next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
      }
    };
    for (ggml_tensor* t : ins) {
      bind_leaf(t);
    }
    for (ggml_tensor* node : nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE && src->view_src == nullptr) {
          bind_leaf(src);
        }
      }
    }
    kg::BindDistinct(nodes, std::uint64_t{1} << 46U);
    auto plan = kg::PlanGraph(nodes, false, ModelDevice());
    EXPECT_TRUE(plan.has_value()) << Why(plan);
    std::set<std::string_view> used;
    if (plan) {
      for (const auto& step : plan->steps) {
        used.insert(step.implementation);
      }
      EXPECT_TRUE(kg::PlaceActivations(nodes, *plan, ins, 256).has_value());
    }
    return used;
  };
  // (rows, passes, head, n_kv): a draft after a verify of 4 rows, one after
  // a prefill, a prefill pass, and a draft past the indexer's budget.
  for (const auto& [rows, passes, head, n_kv] :
       {std::tuple{4, 3, true, 256}, std::tuple{1, 3, true, 512}, std::tuple{37, 1, false, 256},
        std::tuple{4, 3, true, 2304}}) {
    const std::int64_t ratio = p.indexer_ratio;
    const bool select = n_kv > std::int64_t{p.indexer_budget} + ratio - 1;
    const kg::Qwen38MtpShape shape{.rows = rows,
                                   .passes = passes,
                                   .n_kv = n_kv,
                                   .cells = 4096,
                                   .qsa_select = select,
                                   .qsa_blocks = select ? n_kv / ratio : 0,
                                   .head = head,
                                   .head_rows = head ? 32768 : 0,
                                   .hidden_row = 1,
                                   .hidden_rows = 513};
    auto arena = kg::TensorArena::Create(kg::Qwen38MtpGraphTensors(p, passes));
    ASSERT_TRUE(arena.has_value());
    auto graph = kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter, shape, kStride);
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    EXPECT_EQ(graph->passes.size(), static_cast<std::size_t>(passes));
    EXPECT_EQ(graph->drafts.size(), head ? static_cast<std::size_t>(passes) : 0U);
    const std::set<std::string_view> used = plan_of(graph->nodes, graph->inputs());
    EXPECT_FALSE(used.contains(kg::kMxfp8MulMatVecName)) << rows;
    EXPECT_EQ(used.contains(kg::kArgmaxName), head) << rows;
    EXPECT_TRUE(used.contains(kg::kSetRowsExtName)) << rows;  // its caches' writes
    EXPECT_EQ(used.contains(kg::kQsaTopKName), select) << n_kv;
    EXPECT_EQ(used.contains(kg::kQsaAttnName), select) << n_kv;
    EXPECT_TRUE(used.contains(kg::kQsaPoolName)) << n_kv;      // its block keys, every pass
    EXPECT_EQ(used.contains(kg::kMoeGemvName), head) << rows;  // a prefill pass has no MoE
    // The fast mixes' products are jitllm.gemm.bf16 at every width; past
    // 16 rows the drafter's BF16 linears read their input rounded once.
    EXPECT_TRUE(used.contains(kg::kGemmBf16Name)) << rows;
    EXPECT_EQ(used.contains(kg::kBf16Name), rows > kg::kQwen38Bf16Rows) << rows;
  }
  // Refused: rows past the streams', later passes without heads.
  auto arena = kg::TensorArena::Create(kg::Qwen38MtpGraphTensors(p, 3));
  ASSERT_TRUE(arena.has_value());
  EXPECT_FALSE(kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter,
                                       {.rows = 4,
                                        .passes = 1,
                                        .n_kv = 256,
                                        .cells = 4096,
                                        .hidden_row = 510,
                                        .hidden_rows = 513},
                                       kStride)
                   .has_value());
  EXPECT_FALSE(kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter,
                                       {.rows = 4,
                                        .passes = 3,
                                        .n_kv = 256,
                                        .cells = 4096,
                                        .head = false,
                                        .hidden_row = 1,
                                        .hidden_rows = 513},
                                       kStride)
                   .has_value());

  // A verify of 4 rows: every row's logits and argmax, its rows' inputs
  // saved, its streams exported, no recurrent or history state written.
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(40, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 36, 4);
  ASSERT_TRUE(chunk.has_value());
  const std::vector<std::uint64_t> strides(p.layers, kStride);
  auto varena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(varena.has_value());
  auto verify = kg::BuildQwen38Graph(*varena, p, *target, kg::Qwen38ShapeOf(*state, *chunk, 4),
                                     {.expert_stride = strides,
                                      .experts = kg::Qwen38GraphOptions::Experts::kCutlass,
                                      .verify = true,
                                      .export_streams = true,
                                      .stream_rows = 513});
  ASSERT_TRUE(verify.has_value()) << Why(verify);
  ASSERT_NE(verify->argmax, nullptr);
  EXPECT_EQ(verify->argmax->ne[0], 4);
  ASSERT_NE(verify->streams, nullptr);
  std::size_t saves = 0;
  for (const kg::Qwen38LayerTensors& l : verify->layers) {
    saves += (l.commit_conv != nullptr ? 1 : 0) + (l.commit_ple != nullptr ? 1 : 0);
  }
  EXPECT_EQ(saves, 37U);  // 36 linear-attention layers and the n-gram layer
  for (const ggml_tensor* node : verify->nodes) {
    if (node->op != GGML_OP_SET_ROWS) {
      continue;
    }
    const ggml_tensor* into =
        node->src[0]->view_src != nullptr ? node->src[0]->view_src : node->src[0];
    for (const kg::Qwen38LayerTensors& l : verify->layers) {
      EXPECT_NE(into, l.recurrent);
      EXPECT_NE(into, l.conv_state);
      EXPECT_NE(into, l.ple_state);
    }
  }
  const std::set<std::string_view> used = plan_of(verify->nodes, verify->inputs());
  EXPECT_TRUE(used.contains(kg::kArgmaxName));
  EXPECT_TRUE(used.contains(kg::kGdnConvName));
  EXPECT_FALSE(used.contains(kg::kGdnHistoryName));
  // A verify of more than the vector products' rows, or in the reference
  // form, is refused.
  auto wide = md::Qwen38Chunk(p, *state, Hash(), std::vector<std::int32_t>(45, 1000), 36, 9);
  ASSERT_TRUE(wide.has_value());
  EXPECT_FALSE(kg::BuildQwen38Graph(*varena, p, *target, kg::Qwen38ShapeOf(*state, *wide, 9),
                                    {.expert_stride = strides,
                                     .experts = kg::Qwen38GraphOptions::Experts::kCutlass,
                                     .verify = true})
                   .has_value());
  EXPECT_FALSE(kg::BuildQwen38Graph(*varena, p, *target, kg::Qwen38ShapeOf(*state, *chunk, 4),
                                    {.expert_stride = strides,
                                     .exact = true,
                                     .experts = kg::Qwen38GraphOptions::Experts::kCutlass,
                                     .verify = true})
                   .has_value());
}

}  // namespace
