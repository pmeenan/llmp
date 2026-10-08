// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 Flash Next adapter (model/qwen38.h) and its chunk graph
// (kernels/ggml/qwen38_graph.h), in every profile:
// - the binding of a synthetic resource list shaped and typed as the
//   artifact import_m3.py writes from Mia's checkpoint, and its refusals;
// - the n-gram hash's constants checked against the table, and its rows
//   worked by hand (llm_graph_input_ple::set_input at b29c606e2);
// - the state layout's sizes, and a chunk's masks, positions and QSA block
//   tables against llama.cpp's rules (set_input_qsa), worked by hand, and
//   the same chunk at a wave's coarser read alignment;
// - the graph at prefill, decode and past-the-budget shapes: every node
//   planned by an implementation of this module (a model of the device's
//   choices), MXFP8 products by llmpalooza's vector product or the BF16
//   dequantization, the activations placed.

#include "model/qwen38.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "engine/graph_mask_inputs.h"
#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace {

namespace md = llmp::model;
namespace kg = llmp::kernels::ggml;
using llmp::test_support::Failed;

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

TEST(Qwen38Test, GrowingStateCoversPaddedAttentionAndFixedRecurrentState) {
  const auto& p = md::Qwen38Flash();
  auto state = md::Qwen38State(p, 262144, 4096, false);
  ASSERT_TRUE(state) << Why(state);
  auto ranges = md::Qwen38UsedState(p, *state, 257);
  ASSERT_TRUE(ranges) << Why(ranges);
  const auto covered = [&](std::uint64_t offset, std::uint64_t bytes) {
    return std::ranges::any_of(*ranges, [&](const md::StateRange& r) {
      return offset >= r.offset && offset - r.offset <= r.bytes &&
             bytes <= r.bytes - (offset - r.offset);
    });
  };
  using K = md::Qwen38StateTensor::Kind;
  for (const auto& t : state->tensors) {
    const auto row = t.ne0 * (t.f16 ? 2 : 4);
    if (t.kind == K::kK || t.kind == K::kV || t.kind == K::kIndexerK) {
      EXPECT_TRUE(covered(t.offset, 512 * row));
      EXPECT_FALSE(covered(t.offset + (512 * row), row));
    } else if (t.kind == K::kIndexerBlocks) {
      EXPECT_TRUE(covered(t.offset, 128 * row));
      EXPECT_FALSE(covered(t.offset + (128 * row), row));
    } else {
      EXPECT_TRUE(covered(t.offset, t.bytes));
    }
  }
  EXPECT_FALSE(md::Qwen38UsedState(p, *state, 262145));
  // A wave reads the caches through its chunk's end rounded up to 2,048
  // (Qwen38Chunk's read_align): the used state covers every cell it reads.
  ranges = md::Qwen38UsedState(p, *state, 257, 2048);
  ASSERT_TRUE(ranges) << Why(ranges);
  for (const auto& t : state->tensors) {
    const auto row = t.ne0 * (t.f16 ? 2 : 4);
    if (t.kind == K::kK || t.kind == K::kV || t.kind == K::kIndexerK) {
      EXPECT_TRUE(covered(t.offset, 2048 * row));
      EXPECT_FALSE(covered(t.offset + (2048 * row), row));
    } else if (t.kind == K::kIndexerBlocks) {
      EXPECT_TRUE(covered(t.offset, 512 * row));
      EXPECT_FALSE(covered(t.offset + (512 * row), row));
    } else {
      EXPECT_TRUE(covered(t.offset, t.bytes));
    }
  }
  for (const std::uint32_t align : {0U, 128U, 300U}) {
    EXPECT_FALSE(md::Qwen38UsedState(p, *state, 257, align)) << align;
  }
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
        std::pair{65536U, 8191U}, std::pair{131072U, 4095U}, std::pair{262144U, 2047U}}) {
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
  EXPECT_EQ(md::Qwen38MostRows(262145), 0U);
  EXPECT_EQ(md::Qwen38MostRows(262145, false), 0U);
  EXPECT_FALSE(md::Qwen38State(p, 262145, 1).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 262145, 1, false).has_value());
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

TEST(Qwen38Test, MaskMatricesCanBeOmittedWithoutDroppingSelectionMetadata) {
  const auto& p = md::Qwen38Flash();
  for (const auto [past, rows, read] : {std::tuple{37U, 3U, 256U}, std::tuple{2299U, 3U, 2304U}}) {
    for (const bool selection : {false, true}) {
      auto host = md::Qwen38Rows(p, 4096, past, rows, read, selection);
      auto device = md::Qwen38Rows(p, 4096, past, rows, read, selection, false);
      ASSERT_TRUE(host.has_value() && device.has_value());
      EXPECT_TRUE(device->mask.empty());
      EXPECT_TRUE(device->mask_f32.empty());
      EXPECT_EQ(device->positions, host->positions);
      EXPECT_EQ(device->cells, host->cells);
      EXPECT_EQ(device->qsa_select, host->qsa_select);
      EXPECT_EQ(device->qsa.blocks, host->qsa.blocks);
      EXPECT_EQ(device->qsa.cell_block, host->qsa.cell_block);
      EXPECT_EQ(device->qsa.block_cells, host->qsa.block_cells);
      EXPECT_EQ(device->qsa.block_pos, host->qsa.block_pos);
      EXPECT_EQ(device->qsa.bias, host->qsa.bias);
      if (selection && device->qsa_select) EXPECT_FALSE(device->qsa.cell_block.empty());
    }
  }
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(40, 7);
  auto host = md::Qwen38Chunk(p, *state, Hash(), history, 37, 3);
  auto device = md::Qwen38Chunk(p, *state, Hash(), history, 37, 3, true, 256, false);
  ASSERT_TRUE(host.has_value() && device.has_value());
  EXPECT_EQ(device->tokens, host->tokens);
  EXPECT_EQ(device->ple_rows, host->ple_rows);
  EXPECT_TRUE(device->mask.empty());
  EXPECT_TRUE(device->mask_f32.empty());
  // Skipping materialization does not alter the layout's admission contract.
  EXPECT_FALSE(md::Qwen38Chunk(p, *state, Hash(), history, 36, 3, true, 256, false));
  EXPECT_FALSE(md::Qwen38Rows(p, 4096, 250, 10, 256, true, false));
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

// A wave's chunks read the cells through their end rounded up to 2,048
// rather than 256 (engine/qwen38_runner.cc): the read is clamped to the
// cache's cells, and the rows, positions, cells, n-gram rows and QSA
// decision are the 256-cell chunk's, the extra cells masked from every
// row. An alignment that is zero or not whole 256-cell steps is refused.
TEST(Qwen38Test, AChunksReadAlignmentOnlyWidensTheCellsItReads) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto state = md::Qwen38State(p, 5000, 512);
  ASSERT_TRUE(state.has_value()) << Why(state);
  ASSERT_EQ(state->cells, 5120U);
  const md::Qwen38PleHash h = Hash();
  // (n_past, rows, n_kv at 256, n_kv at 2,048)
  for (const auto& [n_past, rows, fine, coarse] :
       {std::tuple{0U, 1U, 256U, 2048U}, std::tuple{37U, 4U, 256U, 2048U},
        std::tuple{250U, 8U, 512U, 2048U}, std::tuple{2040U, 8U, 2048U, 2048U},
        std::tuple{2045U, 4U, 2304U, 4096U}, std::tuple{4090U, 8U, 4352U, 5120U},
        std::tuple{4996U, 4U, 5120U, 5120U}}) {
    SCOPED_TRACE(std::format("{} rows at {}", rows, n_past));
    std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
    for (std::size_t i = 0; i < history.size(); ++i) {
      history[i] = static_cast<std::int32_t>((i * 7919) % 100000);
    }
    for (const bool masks : {true, false}) {
      auto a = md::Qwen38Chunk(p, *state, h, history, n_past, rows, masks);
      auto b = md::Qwen38Chunk(p, *state, h, history, n_past, rows, masks, 2048);
      ASSERT_TRUE(a.has_value()) << Why(a);
      ASSERT_TRUE(b.has_value()) << Why(b);
      EXPECT_EQ(a->n_kv, fine);
      EXPECT_EQ(b->n_kv, coarse);
      EXPECT_EQ(b->rows, a->rows);
      EXPECT_EQ(b->n_past, a->n_past);
      EXPECT_EQ(b->tokens, a->tokens);
      EXPECT_EQ(b->positions, a->positions);
      EXPECT_EQ(b->cells, a->cells);
      EXPECT_EQ(b->ple_rows, a->ple_rows);
      EXPECT_EQ(b->qsa_select, a->qsa_select);
      if (b->qsa_select) {
        EXPECT_EQ(b->qsa.blocks, (coarse + p.indexer_ratio - 1) / p.indexer_ratio);
      }
      // Each row sees the same cells; the extra ones are masked.
      ASSERT_EQ(a->mask.empty(), b->mask.empty());
      ASSERT_EQ(a->mask_f32.empty(), b->mask_f32.empty());
      ASSERT_TRUE(b->mask.empty() || b->mask.size() == std::size_t{coarse} * rows);
      ASSERT_TRUE(b->mask_f32.empty() || b->mask_f32.size() == std::size_t{coarse} * rows);
      for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t j = 0; j < coarse; ++j) {
          if (!b->mask.empty()) {
            const std::uint16_t want =
                j < fine ? a->mask[(std::size_t{r} * fine) + j] : md::kQwen38HalfNegInf;
            ASSERT_EQ(b->mask[(std::size_t{r} * coarse) + j], want) << r << ", " << j;
          }
          if (!b->mask_f32.empty()) {
            const float want = j < fine ? a->mask_f32[(std::size_t{r} * fine) + j]
                                        : -std::numeric_limits<float>::infinity();
            ASSERT_EQ(b->mask_f32[(std::size_t{r} * coarse) + j], want) << r << ", " << j;
          }
        }
      }
    }
  }
  std::vector<std::int32_t> history(40, 1000);
  for (const std::uint32_t align : {0U, 128U, 300U, 2047U, 4097U}) {
    EXPECT_FALSE(md::Qwen38Chunk(p, *state, h, history, 36, 4, false, align).has_value()) << align;
  }
  auto three = md::Qwen38Chunk(p, *state, h, history, 36, 4, false, 768);
  ASSERT_TRUE(three.has_value()) << Why(three);
  EXPECT_EQ(three->n_kv, 768U);
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
    // llmpalooza's fusions; the others in GGML's nodes.
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
    // (The fast form, up to 16 rows, writes the state in place: llmp.gdn.step.)
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
  for (const std::string_view type : {"BF16", "Q4_1"}) {
    resources[resources.size() - 2].type = type;
    auto selected = md::BindQwen38Mtp(p, "qwen4exp-mtp", resources);
    ASSERT_TRUE(selected.has_value()) << Why(selected);
    EXPECT_EQ(selected->draft_output.type, type);
  }
  for (const std::string_view type : {"F32", "Q4_0", "Q8_0"}) {
    resources[resources.size() - 2].type = type;
    EXPECT_FALSE(md::BindQwen38Mtp(p, "qwen4exp-mtp", resources));
  }
  resources[resources.size() - 2].type = "BF16";
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

// A prefill chunk with the injection after a verify that kept several rows
// (their streams in H[1 .. kept]) runs the drafter over each kept row's
// position from that row's own streams, as a draft's catch-up reads them
// (row 1 + i for the position n_past - kept + i), then over the chunk's
// rows but its last from the streams the chunk exports (H[1 ..]).
TEST(Qwen38Test, AnInjectedChunkCatchesUpEveryPendingRowFromItsOwnStreams) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto target = md::Qwen38State(p, 4000, 512);
  ASSERT_TRUE(target.has_value());
  auto s = md::Qwen38MtpStateOf(p, *target);
  ASSERT_TRUE(s.has_value()) << Why(s);
  constexpr std::uint32_t kAt = 100;
  constexpr std::uint32_t kRows = 8;
  for (std::uint32_t pending = 1; pending <= 4; ++pending) {
    auto in = md::Qwen38InjectionOf(*s, kAt, kRows, pending);
    ASSERT_TRUE(in.has_value()) << Why(in);
    // Each pending position once, with the H row a draft would read: the
    // catch-up's from row 1, the last's carried from its row to row 0.
    std::vector<std::uint32_t> row_of(pending, 0);
    for (std::uint32_t r = 0; r < in->catch_up_rows; ++r) {
      const std::uint32_t at = in->catch_up_first + r;
      ASSERT_GE(at, kAt - pending);
      ASSERT_LT(at, kAt);
      EXPECT_EQ(row_of[at - (kAt - pending)], 0U);
      row_of[at - (kAt - pending)] = 1 + r;
    }
    ASSERT_EQ(in->hidden_row, 0U);
    ASSERT_EQ(in->first, kAt - 1);
    ASSERT_NE(in->carry, 0U);
    EXPECT_EQ(row_of[pending - 1], 0U);
    row_of[pending - 1] = in->carry;
    for (std::uint32_t i = 0; i < pending; ++i) {
      EXPECT_EQ(row_of[i], 1 + i) << pending << " pending, row " << i;
    }
    // The pass: that row, then the chunk's rows but its last.
    EXPECT_EQ(in->rows, kRows);
  }
  // A chunk at the sequence's start, with or without a stale cursor.
  for (const std::uint32_t pending : {0U, 3U}) {
    auto start = md::Qwen38InjectionOf(*s, 0, kRows, pending);
    ASSERT_TRUE(start.has_value()) << Why(start);
    EXPECT_EQ(start->catch_up_rows, 0U);
    EXPECT_EQ(start->carry, 0U);
    EXPECT_EQ(start->first, 0U);
    EXPECT_EQ(start->rows, kRows - 1);
    EXPECT_EQ(start->hidden_row, 1U);
  }
  auto lone = md::Qwen38InjectionOf(*s, 0, 1, 0);
  ASSERT_TRUE(lone.has_value());
  EXPECT_EQ(lone->rows, 0U);
  // None pending: row 0 as the last injected chunk left it, nothing copied.
  auto none = md::Qwen38InjectionOf(*s, kAt, kRows, 0);
  ASSERT_TRUE(none.has_value());
  EXPECT_EQ(none->catch_up_rows, 0U);
  EXPECT_EQ(none->carry, 0U);
  EXPECT_EQ(none->first, kAt - 1);
  EXPECT_EQ(none->rows, kRows);
  EXPECT_FALSE(md::Qwen38InjectionOf(*s, kAt, 0, 1).has_value());         // an empty chunk
  EXPECT_FALSE(md::Qwen38InjectionOf(*s, kAt, 513, 1).has_value());       // past the streams rows
  EXPECT_FALSE(md::Qwen38InjectionOf(*s, 3990, 11, 1).has_value());       // past the context
  EXPECT_FALSE(md::Qwen38InjectionOf(*s, 2, kRows, 3).has_value());       // before position 0
  EXPECT_FALSE(md::Qwen38InjectionOf(*s, kAt * 10, 1, 513).has_value());  // past the rows
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

TEST(Qwen38Test, MtpDeviceMasksFollowActualAttentionConsumers) {
  const auto& p = md::Qwen38Flash();
  auto target = md::BindQwen38(p, "qwen4exp", ArtifactLike(p, true));
  auto drafter = md::BindQwen38Mtp(p, "qwen4exp-mtp", MtpLike());
  ASSERT_TRUE(target.has_value() && drafter.has_value());
  for (const bool device : {false, true}) {
    for (const bool head : {false, true}) {
      const kg::Qwen38MtpShape shape{.rows = 37,
                                     .passes = head ? 3 : 1,
                                     .n_kv = 256,
                                     .cells = 4096,
                                     .head = head,
                                     .head_rows = head ? 32768 : 0,
                                     .hidden_row = 1,
                                     .hidden_rows = 513};
      auto arena = kg::TensorArena::Create(kg::Qwen38MtpGraphTensors(p, shape.passes));
      ASSERT_TRUE(arena.has_value());
      auto graph = kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter, shape, 2764800, device);
      ASSERT_TRUE(graph.has_value()) << Why(graph);
      const auto inputs = graph->inputs();
      EXPECT_EQ(graph->passes.size(), static_cast<std::size_t>(shape.passes));
      for (std::size_t pass = 0; pass < graph->passes.size(); ++pass) {
        const auto& in = graph->passes[pass];
        ASSERT_NE(in.positions, nullptr);
        if (device && !head) {
          EXPECT_EQ(in.mask, nullptr);
          EXPECT_TRUE(std::ranges::none_of(graph->nodes, [](const auto* node) {
            return kg::LlmpOpOf(node) == kg::LlmpOp::kGemma4Mask;
          }));
          continue;
        }
        ASSERT_NE(in.mask, nullptr);
        if (device) {
          EXPECT_TRUE(kg::Gemma4MaskFits(in.mask));
          EXPECT_EQ(in.mask->src[0], in.positions);
          EXPECT_EQ(in.mask->ne[1], pass == 0 ? 37 : 1);
          EXPECT_EQ(std::ranges::count(graph->nodes, in.mask), 1);
          EXPECT_FALSE(std::ranges::contains(inputs, in.mask));
        } else {
          EXPECT_EQ(in.mask->op, GGML_OP_NONE);
          EXPECT_EQ(std::ranges::count(inputs, in.mask), 1);
        }
      }
    }
  }
}

TEST(Qwen38Test, RetainingDraftHeadOperandsPreservesTheOperationSequence) {
  const auto& p = md::Qwen38Flash();
  auto target = md::BindQwen38(p, "qwen4exp", ArtifactLike(p, true));
  ASSERT_TRUE(target.has_value());
  using NodeShape = std::tuple<ggml_op, ggml_type, kg::LlmpOp, std::vector<std::int64_t>>;
  for (const bool selected : {false, true}) {
    auto resources = MtpLike();
    const std::int64_t head_rows = selected ? 47172 : 65536;
    if (selected) {
      resources.push_back({.roles = {"draft_output.weight"},
                           .type = "BF16",
                           .ne = {2560, static_cast<std::uint64_t>(head_rows)}});
      resources.push_back({.roles = {"draft_output.ids"},
                           .type = "I32",
                           .ne = {1, static_cast<std::uint64_t>(head_rows)}});
    }
    auto drafter = md::BindQwen38Mtp(p, "qwen4exp-mtp", resources);
    ASSERT_TRUE(drafter.has_value()) << Why(drafter);
    for (const auto& [rows, passes, head, confidence] :
         {std::tuple{1, 3, true, false}, std::tuple{4, 3, true, true},
          std::tuple{37, 1, false, false}}) {
      std::vector<NodeShape> original;
      for (const bool capture : {false, true}) {
        const kg::Qwen38MtpShape shape{.rows = rows,
                                       .passes = passes,
                                       .n_kv = 256,
                                       .cells = 4096,
                                       .head = head,
                                       .head_rows = head_rows,
                                       .confidence = confidence,
                                       .capture_head = capture && head,
                                       .hidden_row = 1,
                                       .hidden_rows = 513};
        auto other = shape;
        other.capture_head = !shape.capture_head;
        EXPECT_NE(shape, other);  // retained lifetimes enter the plan/graph cache key
        auto arena = kg::TensorArena::Create(kg::Qwen38MtpGraphTensors(p, passes));
        ASSERT_TRUE(arena.has_value());
        auto graph = kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter, shape, 2764800);
        ASSERT_TRUE(graph.has_value()) << Why(graph);
        std::size_t heads = 0;
        for (const ggml_tensor* node : graph->nodes) {
          if (kg::LlmpOpOf(node) != kg::LlmpOp::kArgmax) {
            continue;
          }
          ++heads;
          const ggml_tensor* product = node->src[0];
          ASSERT_NE(product, nullptr);
          EXPECT_EQ(product->op, GGML_OP_MUL_MAT);
          EXPECT_EQ(product->type, GGML_TYPE_F32);
          EXPECT_EQ(product->ne[0], head_rows);
          EXPECT_EQ(product->ne[1], 1);
          EXPECT_EQ(product->src[0]->type, GGML_TYPE_BF16);
          const ggml_tensor* input = product->src[1];
          ASSERT_NE(input, nullptr);
          EXPECT_EQ(input->type, GGML_TYPE_F32);
          EXPECT_EQ(input->ne[0], p.width);
          EXPECT_EQ(input->ne[1], 1);
          EXPECT_NE(input->op, GGML_OP_GET_ROWS);
        }
        EXPECT_EQ(heads, head ? static_cast<std::size_t>(passes) : 0U);
        EXPECT_EQ(graph->drafts.size(), heads);
        EXPECT_EQ(graph->probabilities.size(), confidence ? heads : 0U);
        EXPECT_EQ(graph->head_inputs.size(), capture ? heads : 0U);
        EXPECT_EQ(graph->head_logits.size(), capture ? heads : 0U);
        for (std::size_t pass = 0; pass < graph->head_inputs.size(); ++pass) {
          EXPECT_EQ(graph->head_inputs[pass]->type, GGML_TYPE_F32);
          EXPECT_EQ(graph->head_inputs[pass]->ne[0], p.width);
          EXPECT_EQ(graph->head_inputs[pass]->ne[1], 1);
          EXPECT_EQ(graph->head_logits[pass]->ne[0], head_rows);
          EXPECT_EQ(graph->head_logits[pass]->ne[1], 1);
        }
        // Retaining values changes lifetimes only, including around every
        // state write and the original token-map/confidence operations.
        std::vector<NodeShape> signature;
        for (const ggml_tensor* node : graph->nodes) {
          signature.emplace_back(node->op, node->type, kg::LlmpOpOf(node),
                                 std::vector<std::int64_t>(node->ne, node->ne + GGML_MAX_DIMS));
        }
        if (!capture) {
          original = std::move(signature);
        } else {
          EXPECT_EQ(signature, original);
        }
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
        std::vector<ggml_tensor*> kept = graph->drafts;
        kept.insert(kept.end(), graph->head_inputs.begin(), graph->head_inputs.end());
        kept.insert(kept.end(), graph->head_logits.begin(), graph->head_logits.end());
        auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256, kept);
        ASSERT_TRUE(placed.has_value()) << Why(placed);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> retained_ranges;
        for (ggml_tensor* retained : kept) {
          const ggml_tensor* storage = retained;
          while (storage->view_src != nullptr) {
            storage = storage->view_src;
          }
          const auto found = std::ranges::find_if(
              placed->offsets, [storage](const auto& entry) { return entry.first == storage; });
          ASSERT_NE(found, placed->offsets.end());
          const std::uint64_t end = found->second + ggml_nbytes(storage);
          for (const auto& [start, stop] : retained_ranges) {
            EXPECT_TRUE(end <= start || stop <= found->second);
          }
          retained_ranges.emplace_back(found->second, end);
        }
      }
    }
  }
}

TEST(Qwen38Test, DraftHeadCaptureIsOptInAndRefusesUnboundedOrHeadlessShapes) {
  const auto& p = md::Qwen38Flash();
  auto target = md::BindQwen38(p, "qwen4exp", ArtifactLike(p, true));
  auto drafter = md::BindQwen38Mtp(p, "qwen4exp-mtp", MtpLike());
  ASSERT_TRUE(target.has_value() && drafter.has_value());
  kg::Qwen38MtpShape shape{.rows = 1,
                           .passes = 1,
                           .n_kv = 256,
                           .cells = 4096,
                           .head = true,
                           .head_rows = 65536,
                           .hidden_row = 1,
                           .hidden_rows = 513};
  auto arena = kg::TensorArena::Create(kg::Qwen38MtpGraphTensors(p, 1));
  ASSERT_TRUE(arena.has_value());
  auto ordinary = kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter, shape, 2764800);
  ASSERT_TRUE(ordinary.has_value());
  EXPECT_TRUE(ordinary->head_inputs.empty());
  EXPECT_TRUE(ordinary->head_logits.empty());
  shape.capture_head = true;
  for (const auto& [head, head_rows] : {std::pair{false, std::int64_t{65536}},
                                        {true, std::int64_t{65537}},
                                        {true, std::int64_t{0}}}) {
    shape.head = head;
    shape.head_rows = head_rows;
    auto refused = kg::BuildQwen38MtpGraph(*arena, p, *target, *drafter, shape, 2764800);
    EXPECT_FALSE(refused.has_value());
  }
}

// The drafter's graph and a verify's, planned by this module's
// implementations: the drafter's BF16 products GGML's float product or
// llmp.gemm.bf16, no MXFP8 product, its heads' drafts by llmp.argmax;
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
    // The fast mixes' products are llmp.gemm.bf16 at every width; past
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

TEST(Qwen38Test, RoutedDownCaptureRetainsExistingVerifyOperandsOnly) {
  const auto& p = md::Qwen38Flash();
  auto binding = md::BindQwen38(p, "qwen4exp", ArtifactLike(p, true));
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(binding.has_value() && state.has_value());
  constexpr std::uint64_t mask = 1 | (std::uint64_t{1} << 23) | (std::uint64_t{1} << 47);
  constexpr std::uint64_t stride = 2764800;
  const std::vector<std::uint64_t> strides(p.layers, stride);
  for (const std::uint32_t rows : {1U, 3U, 4U}) {
    auto chunk =
        md::Qwen38Chunk(p, *state, Hash(), std::vector<std::int32_t>(36 + rows, 7), 36, rows);
    ASSERT_TRUE(chunk.has_value());
    auto ordinary_arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    auto capture_arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(ordinary_arena.has_value() && capture_arena.has_value());
    kg::Qwen38GraphOptions options{.expert_stride = strides,
                                   .experts = kg::Qwen38GraphOptions::Experts::kCutlass,
                                   .verify = true};
    const auto shape = kg::Qwen38ShapeOf(*state, *chunk, rows);
    auto ordinary = kg::BuildQwen38Graph(*ordinary_arena, p, *binding, shape, options);
    options.capture_routed = mask;
    auto captured = kg::BuildQwen38Graph(*capture_arena, p, *binding, shape, options);
    ASSERT_TRUE(ordinary.has_value() && captured.has_value()) << Why(captured);
    EXPECT_TRUE(ordinary->routed.empty());
    ASSERT_EQ(captured->routed.size(), 3U);
    ASSERT_EQ(ordinary->nodes.size(), captured->nodes.size());
    for (std::size_t i = 0; i < ordinary->nodes.size(); ++i) {
      EXPECT_EQ(ordinary->nodes[i]->op, captured->nodes[i]->op);
      EXPECT_STREQ(ordinary->nodes[i]->name, captured->nodes[i]->name);
    }
    std::vector<ggml_tensor*> kept = {captured->logits};
    std::uint32_t layer_index = 0;
    constexpr std::array<std::uint32_t, 3> layers = {0, 23, 47};
    for (const auto& layer : captured->routed) {
      EXPECT_EQ(layer.layer, layers[layer_index++]);
      EXPECT_EQ(layer.input, layer.activation->src[1]);
      EXPECT_EQ(layer.input->ne[0], p.width);
      EXPECT_EQ(layer.input->ne[1], 1);
      EXPECT_EQ(layer.input->ne[2], rows);
      EXPECT_EQ(layer.activation->ne[0], p.expert_ffn);
      EXPECT_EQ(layer.activation->ne[1], p.experts_used);
      EXPECT_EQ(layer.activation->ne[2], rows);
      EXPECT_EQ(layer.down->ne[0], p.width);
      EXPECT_EQ(layer.combined->ne[1], rows);
      for (auto* tensor : {layer.input, layer.activation, layer.down, layer.shared, layer.gate,
                           layer.weights, layer.ids, layer.combined}) {
        kept.push_back(tensor);
      }
    }
    // The descriptor-only plan is bound at distinct fake addresses; this
    // checks the lifetimes of views and their owning computed storage.
    std::uint64_t next = std::uint64_t{1} << 40;
    for (auto* node : captured->nodes) {
      for (auto* source : node->src) {
        if (source != nullptr && source->op == GGML_OP_NONE && source->view_src == nullptr &&
            source->data == nullptr) {
          kg::TensorArena::Bind(source, next);
          next += ((ggml_nbytes(source) + 255) / 256 * 256) + 256;
        }
      }
    }
    kg::BindDistinct(captured->nodes, std::uint64_t{1} << 46);
    auto plan = kg::PlanGraph(captured->nodes, false, ModelDevice());
    ASSERT_TRUE(plan.has_value()) << Why(plan);
    auto placed = kg::PlaceActivations(captured->nodes, *plan, captured->inputs(), 256, kept);
    ASSERT_TRUE(placed.has_value()) << Why(placed);
    std::set<const ggml_tensor*> owners;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> retained_ranges;
    for (auto* tensor : kept) {
      const ggml_tensor* owner = tensor;
      while (owner->view_src != nullptr) {
        owner = owner->view_src;
      }
      if (!owners.insert(owner).second) {
        continue;  // IDs/weights/gate share the same router's storage.
      }
      auto found = std::ranges::find_if(
          placed->offsets, [owner](const auto& offset) { return offset.first == owner; });
      ASSERT_NE(found, placed->offsets.end());
      const std::uint64_t end = found->second + ggml_nbytes(owner);
      for (const auto& [start, stop] : retained_ranges) {
        EXPECT_TRUE(end <= start || stop <= found->second);
      }
      retained_ranges.emplace_back(found->second, end);
    }
  }
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), std::vector<std::int32_t>(41, 7), 36, 5);
  ASSERT_TRUE(chunk.has_value());
  for (const auto [rows, verify, exact, capture] :
       {std::tuple{4U, false, false, mask}, std::tuple{4U, true, true, mask},
        std::tuple{4U, true, false, mask | 2}, std::tuple{4U, true, false, std::uint64_t{1} << 48},
        std::tuple{5U, true, false, mask}}) {
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto shape = kg::Qwen38ShapeOf(*state, *chunk, rows);
    shape.rows = rows;
    EXPECT_FALSE(kg::BuildQwen38Graph(*arena, p, *binding, shape,
                                      {.expert_stride = strides,
                                       .exact = exact,
                                       .experts = kg::Qwen38GraphOptions::Experts::kCutlass,
                                       .verify = verify,
                                       .capture_routed = capture})
                     .has_value());
  }
}

// ------------------------------------------------ a GGUF checkpoint's artifact

// A GGML type's block (values, bytes), for the types the GGUF below uses.
std::pair<std::uint64_t, std::uint64_t> Block(std::string_view type) {
  if (type == "Q6_K") {
    return {256, 210};
  }
  if (type == "Q8_0") {
    return {32, 34};
  }
  if (type == "IQ4_NL") {
    return {32, 18};
  }
  if (type == "IQ2_S") {
    return {256, 82};
  }
  if (type == "BF16") {
    return {1, 2};
  }
  return {1, 4};  // F32
}

std::uint64_t RowBytes(std::string_view type, std::uint64_t values) {
  const auto [block, bytes] = Block(type);
  return values / block * bytes;
}

// A tensor's readable bytes as the importer reserves them: its bytes plus,
// for a quantized row short of a 512-value step, GGML's over-read.
std::uint64_t Readable(std::string_view type, const std::vector<std::uint64_t>& ne) {
  std::uint64_t bytes = RowBytes(type, ne[0]);
  for (std::size_t i = 1; i < ne.size(); ++i) {
    bytes *= ne[i];
  }
  const bool quantized = Block(type).first > 1;
  return bytes + (quantized && ne[0] % 512 != 0 ? RowBytes(type, 512 - (ne[0] % 512)) : 0);
}

// unsloth's UD-IQ3_XXS GGUF of Qwen3.8 Flash Next, as layout.py imports it
// (tensor names, types and shapes as the checkpoint has them, layer 0's mix
// for every layer): Q6_K and Q8_0 matrices, F32 norms, routers, injects and
// the recurrence's parameters, BF16 indexer projections, IQ2_S gate and up
// experts and IQ4_NL down ones, and the IQ4_NL n-gram table. The expert
// arrays share each layer's groups at the stride kGgufStride.
constexpr std::uint64_t kGgufStride = 1977840;  // the group's 1,974,272 bytes, whole 5,904s
std::vector<md::Qwen38Resource> GgufLike(const md::Qwen38Profile& p) {
  std::vector<md::Qwen38Resource> r;
  std::vector<md::Qwen38Resource> arrays;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    const std::uint64_t readable = Readable(type, ne);
    r.push_back({.roles = {std::move(name)},
                 .plain = false,
                 .type = std::move(type),
                 .ne = std::move(ne),
                 .readable = readable});
  };
  ggml("token_embd.weight", "Q6_K", {2560, 248320});
  ggml("output.weight", "Q6_K", {2560, 248320});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "Q8_0", {10240, 320});
  ggml("output_hc_up.weight", "Q8_0", {320, 10240});
  ggml("per_layer_token_embd.weight", "IQ4_NL", {160, kTableRows});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    for (const char* kind : {"attn", "ffn"}) {
      ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {10240});
      ggml(std::format("{}hc_{}_down.weight", n, kind), "Q8_0", {10240, 320});
      ggml(std::format("{}hc_{}_up.weight", n, kind), "Q8_0", {320, 10240});
      ggml(std::format("{}hc_{}_inject.weight", n, kind), "F32", {10240, 4});
    }
    if (p.linear(il)) {
      ggml(n + "attn_qkv.weight", "Q6_K", {2560, 10240});
      ggml(n + "attn_gate.weight", "Q6_K", {2560, 6144});
      ggml(n + "ssm_beta.weight", "F32", {2560, 48});
      ggml(n + "ssm_alpha.weight", "F32", {2560, 48});
      ggml(n + "ssm_dt.bias", "F32", {48});
      ggml(n + "ssm_a", "F32", {48});
      ggml(n + "ssm_conv1d.weight", "F32", {4, 10240});
      ggml(n + "ssm_norm.weight", "F32", {128});
      ggml(n + "ssm_out.weight", "Q6_K", {6144, 2560});
    } else {
      ggml(n + "attn_q.weight", "Q6_K", {2560, 12288});
      ggml(n + "attn_k.weight", "Q6_K", {2560, 512});
      ggml(n + "attn_v.weight", "Q6_K", {2560, 512});
      ggml(n + "attn_output.weight", "Q6_K", {6144, 2560});
      ggml(n + "attn_q_norm.weight", "F32", {256});
      ggml(n + "attn_k_norm.weight", "F32", {256});
      ggml(n + "indexer.q_proj.weight", "BF16", {2560, 512});
      ggml(n + "indexer.k_proj.weight", "BF16", {2560, 128});
      ggml(n + "indexer.q_norm.weight", "F32", {128});
      ggml(n + "indexer.k_norm.weight", "F32", {128});
    }
    if (il == 1) {
      ggml(n + "ple_key.weight", "Q8_0", {2560, 10240});
      ggml(n + "ple_value.weight", "Q8_0", {2560, 2560});
      for (const char* part : {"key", "query", "conv"}) {
        ggml(std::format("{}ple_norm_{}.weight", n, part), "F32", {10240});
      }
      ggml(n + "ple_conv1d.weight", "F32", {4, 10240});
    }
    ggml(n + "ffn_gate_inp.weight", "F32", {2560, 512});
    ggml(n + "ffn_gate_inp_shexp.weight", "F32", {2560});
    ggml(n + "ffn_gate_shexp.weight", "Q6_K", {2560, 640});
    ggml(n + "ffn_up_shexp.weight", "Q6_K", {2560, 640});
    ggml(n + "ffn_down_shexp.weight", "Q8_0", {640, 2560});
    std::uint64_t group_offset = 0;
    for (const auto& [proj, type, ne] :
         {std::tuple{"gate", "IQ2_S", std::vector<std::uint64_t>{2560, 640}},
          std::tuple{"up", "IQ2_S", std::vector<std::uint64_t>{2560, 640}},
          std::tuple{"down", "IQ4_NL", std::vector<std::uint64_t>{640, 2560}}}) {
      const std::uint64_t readable = Readable(type, ne);
      arrays.push_back({.roles = {std::format("{}ffn_{}_exps.weight", n, proj)},
                        .plain = false,
                        .type = type,
                        .ne = ne,
                        .expert_array = true,
                        .count = 512,
                        .group_offset = group_offset,
                        .readable = readable});
      group_offset += (readable + 255) / 256 * 256;
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

TEST(Qwen38Test, BindsAGgufCheckpointsTensorsAndRefusesWhatDiffers) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = GgufLike(p);
  auto bound = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  EXPECT_TRUE(bound->gguf());
  EXPECT_FALSE(bound->cutlass());
  EXPECT_EQ(bound->ple_table.type, "IQ4_NL");
  EXPECT_EQ(bound->ple_table.ne, (std::vector<std::uint64_t>{160, kTableRows}));
  EXPECT_EQ(bound->token_embd.type, "Q6_K");
  EXPECT_TRUE(bound->layers[0].qkv.is_matrix());
  EXPECT_EQ(bound->layers[0].qkv.matrix.type, "Q6_K");
  EXPECT_TRUE(bound->layers[0].qkv.codes.type.empty());
  EXPECT_EQ(bound->layers[3].idx_q.matrix.type, "BF16");
  EXPECT_EQ(bound->layers[3].idx_k.matrix.ne, (std::vector<std::uint64_t>{2560, 128}));
  EXPECT_TRUE(bound->layers[3].idx_qk.matrix.type.empty());
  EXPECT_EQ(bound->layers[5].down_exps.type, "IQ4_NL");
  EXPECT_TRUE(bound->layers[5].gate_exps_scale.type.empty());
  EXPECT_TRUE(bound->layers[1].ple_multipliers.type.empty());
  EXPECT_EQ(bound->layers[0].router.type, "F32");
  // The readable bytes the importer reserves reach the binding (Q8_0's
  // 640-value rows read 408 bytes past the last).
  EXPECT_EQ(bound->layers[0].down_shexp.matrix.readable, 1740800U + 408U);

  // A tensor of the ModelOpt form, a hash tensor, a missing projection, a
  // plain resource or another shape: refused, naming the tensor.
  auto mixed = resources;
  for (md::Qwen38Resource& r : mixed) {
    if (r.roles[0] == "blk.3.indexer.k_proj.weight") {
      r.roles[0] = "blk.3.indexer.qk_proj.weight";
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", mixed)).find("indexer"), std::string::npos);
  auto hash = resources;
  hash.push_back({.roles = {"blk.1.ple_multipliers"}, .plain = false, .type = "I64", .ne = {3}});
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", hash)).find("does not read"), std::string::npos);
  auto plain = resources;
  for (md::Qwen38Resource& r : plain) {
    if (r.roles[0] == "blk.0.ssm_a") {
      r.plain = true;
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", plain)).find("blk.0.ssm_a"), std::string::npos);
  auto reshaped = resources;
  for (md::Qwen38Resource& r : reshaped) {
    if (r.roles[0] == "blk.7.attn_k.weight") {
      r.ne = {2560, 256};
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", reshaped)).find("blk.7.attn_k.weight"),
            std::string::npos);
  auto retyped = resources;
  for (md::Qwen38Resource& r : retyped) {
    if (r.roles[0] == "blk.0.ssm_norm.weight") {
      r.type = "F16";  // the norms are F32, as llama.cpp's converter writes them
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", retyped)).find("ssm_norm"), std::string::npos);
}

// A GGUF header (artifact/gguf_metadata.h) with the qwen4exp keys the
// binding checks, as unsloth's GGUF has them; `skip` leaves a key out and
// `used` sets the experts used.
std::vector<std::byte> GgufMetadata(std::string_view skip = {}, std::uint32_t used = 10) {
  std::vector<std::byte> out;
  const auto raw = [&](const void* p, std::size_t n) {
    const auto* b = static_cast<const std::byte*>(p);
    out.insert(out.end(), b, b + n);
  };
  const auto u32 = [&](std::uint32_t v) { raw(&v, 4); };
  const auto u64 = [&](std::uint64_t v) { raw(&v, 8); };
  const auto str = [&](std::string_view s) {
    u64(s.size());
    raw(s.data(), s.size());
  };
  struct Entry {
    std::string key;
    std::function<void()> value;
  };
  const auto ints = [&](const std::vector<std::int64_t>& v, std::uint32_t type) {
    u32(type);
    u64(v.size());
    for (const std::int64_t x : v) {
      raw(&x, type == 11 ? 8 : 4);
    }
  };
  std::vector<std::int64_t> ratios;
  ratios.reserve(48);
  for (std::uint32_t il = 0; il < 48; ++il) {
    ratios.push_back(il % 4 == 3 ? 4 : 0);
  }
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  const std::vector<std::pair<std::string, std::uint32_t>> scalars = {
      {"qwen4exp.block_count", 48},
      {"qwen4exp.embedding_length", 2560},
      {"qwen4exp.context_length", 262144},
      {"qwen4exp.full_attention_interval", 4},
      {"qwen4exp.attention.head_count", 24},
      {"qwen4exp.attention.head_count_kv", 2},
      {"qwen4exp.attention.key_length", 256},
      {"qwen4exp.attention.value_length", 256},
      {"qwen4exp.attention.indexer.head_count", 4},
      {"qwen4exp.attention.indexer.key_length", 128},
      {"qwen4exp.attention.indexer.top_k", 2048},
      {"qwen4exp.rope.dimension_count", 64},
      {"qwen4exp.ssm.conv_kernel", 4},
      {"qwen4exp.ssm.group_count", 16},
      {"qwen4exp.ssm.time_step_rank", 48},
      {"qwen4exp.ssm.state_size", 128},
      {"qwen4exp.ssm.inner_size", 6144},
      {"qwen4exp.expert_count", 512},
      {"qwen4exp.expert_used_count", used},
      {"qwen4exp.expert_feed_forward_length", 640},
      {"qwen4exp.expert_shared_feed_forward_length", 640},
      {"qwen4exp.hyper_connection.count", 4},
      {"qwen4exp.hyper_connection.low_rank", 320},
      {"qwen4exp.ple.ngram_size", 3},
      {"qwen4exp.ple.heads_per_ngram", 8},
      {"qwen4exp.ple.conv_kernel", 4},
      {"qwen4exp.ple.eos_token_id", 248044},
      {"qwen4exp.embedding_length_per_layer_input", 160},
  };
  std::vector<Entry> entries;
  entries.push_back({"general.architecture", [&] {
                       u32(8);
                       str("qwen4exp");
                     }});
  for (const auto& [key, value] : scalars) {
    entries.push_back({key, [&, v = value] {
                         u32(4);
                         u32(v);
                       }});
  }
  for (const auto& [key, value] : {std::pair{"qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f},
                                   std::pair{"qwen4exp.rope.freq_base", 10000000.0f}}) {
    entries.push_back({key, [&, v = value] {
                         u32(6);
                         raw(&v, 4);
                       }});
  }
  entries.push_back({"qwen4exp.attention.compress_ratios", [&] {
                       u32(9);
                       ints(ratios, 5);
                     }});
  entries.push_back({"qwen4exp.rope.dimension_sections", [&] {
                       u32(9);
                       ints({11, 11, 10, 0}, 5);
                     }});
  entries.push_back({"qwen4exp.ple.layers", [&] {
                       u32(9);
                       ints({1}, 5);
                     }});
  entries.push_back({"qwen4exp.ple.layer_multipliers", [&] {
                       u32(9);
                       ints({3, 5, 7}, 11);
                     }});
  entries.push_back({"qwen4exp.ple.head_offsets", [&] {
                       u32(9);
                       ints(offsets, 11);
                     }});
  entries.push_back({"qwen4exp.ple.head_vocab_sizes", [&] {
                       u32(9);
                       ints(vocab, 11);
                     }});
  std::erase_if(entries, [&](const Entry& e) { return e.key == skip; });
  raw("GGUF", 4);
  u32(3);
  u64(0);
  u64(entries.size());
  for (const Entry& e : entries) {
    str(e.key);
    e.value();
  }
  return out;
}

TEST(Qwen38Test, AGgufCheckpointsHashAndHyperparametersComeFromItsMetadata) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<std::byte> metadata = GgufMetadata();
  auto read = md::ReadQwen38GgufHash(p, metadata, kTableRows);
  ASSERT_TRUE(read.has_value()) << Why(read);
  const md::Qwen38PleHash want = Hash();
  EXPECT_EQ(read->multipliers, want.multipliers);
  EXPECT_EQ(read->offsets, want.offsets);
  EXPECT_EQ(read->vocab, want.vocab);
  // A hyperparameter that is not the profile's, a missing key, constants
  // that leave the table, or bytes that are not GGUF metadata: refused.
  EXPECT_NE(Why(md::ReadQwen38GgufHash(p, GgufMetadata({}, 8), kTableRows))
                .find("qwen4exp.expert_used_count"),
            std::string::npos);
  EXPECT_NE(Why(md::ReadQwen38GgufHash(p, GgufMetadata("qwen4exp.rope.freq_base"), kTableRows))
                .find("qwen4exp.rope.freq_base"),
            std::string::npos);
  EXPECT_NE(Why(md::ReadQwen38GgufHash(p, GgufMetadata("qwen4exp.ple.head_offsets"), kTableRows))
                .find("hash"),
            std::string::npos);
  EXPECT_NE(Why(md::ReadQwen38GgufHash(p, metadata, 1000)).find("outside"), std::string::npos);
  const std::vector<std::byte> truncated(metadata.begin(), metadata.end() - 1);
  EXPECT_NE(Why(md::ReadQwen38GgufHash(p, truncated, kTableRows)).find("kept GGUF"),
            std::string::npos);
}

TEST(Qwen38Test, AGgufCheckpointsGraphTakesGgmlsProductsAndTheFormatFreeFusions) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = GgufLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  const md::Qwen38PleHash h = Hash();
  const std::vector<std::uint64_t> strides(p.layers, kGgufStride);
  // (n_past, rows, fused, exact)
  for (const auto& [n_past, rows, fused, exact] :
       {std::tuple{0U, 37U, true, false}, std::tuple{37U, 1U, true, false},
        std::tuple{2800U, 512U, true, false}, std::tuple{4095U, 1U, true, false},
        std::tuple{40U, 8U, true, false}, std::tuple{0U, 37U, true, true},
        std::tuple{2800U, 512U, true, true}, std::tuple{0U, 37U, false, false},
        std::tuple{37U, 1U, false, false}, std::tuple{2800U, 512U, false, false}}) {
    std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
    auto chunk = md::Qwen38Chunk(p, *state, h, history, n_past, rows);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, 1);
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto graph = kg::BuildQwen38Graph(*arena, p, *binding, shape,
                                      {.expert_stride = strides, .fused = fused, .exact = exact});
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
    const std::string at =
        std::format("{} rows at {}, fused {}, exact {}", rows, n_past, fused, exact);
    // The n-gram table's IQ4_NL rows, the token table's Q6_K rows.
    EXPECT_TRUE(used.contains(kg::kQRowsName)) << at;
    EXPECT_TRUE(used.contains(kg::kGetRowsExtName)) << at;
    EXPECT_FALSE(used.contains(kg::kNvfp4RowsName)) << at;
    // No ModelOpt format's operation.
    for (const std::string_view name :
         {kg::kMxfp8MulMatVecName, kg::kMxfp8DequantName, kg::kMxfp8GemmName,
          kg::kMxfp8QuantizeName, kg::kMxfp8SwizzleName, kg::kQsaGateQuantizeName, kg::kMoeGemvName,
          kg::kMoeGemmName, kg::kMoeGluName, kg::kHcPrepName, kg::kHcLoName, kg::kHcMixBf16Name}) {
      EXPECT_FALSE(used.contains(name)) << name << ": " << at;
    }
    // The quantized products up to 8 rows (the head's one row at every
    // width): the fast form's llmp.vecq over one Q8_1 quantization of each
    // input, else GGML's MMVQ; past them GGML's MMQ. The routed experts take
    // llmp.vecq where the fast form's rows times the experts used fit its
    // 64 pairs.
    const bool vector = rows <= 8;
    const bool vecq_experts = fast && rows * 10 <= 64;
    EXPECT_EQ(used.contains(kg::kVecQName), fast) << at;
    EXPECT_EQ(used.contains(kg::kQuantizeQ8Name), fast) << at;
    EXPECT_EQ(used.contains(kg::kMulMatVecQ), !fast) << at;
    EXPECT_EQ(used.contains(kg::kMulMatQ), !vector) << at;
    EXPECT_EQ(used.contains(kg::kMulMatIdVecQ), vector && !vecq_experts) << at;
    EXPECT_EQ(used.contains(kg::kMulMatIdQ), !vector) << at;
    // The reference form's hyper-connection fusions where fused.
    for (const std::string_view name : {kg::kHcCombineName, kg::kHcNormName, kg::kHcMixName}) {
      EXPECT_EQ(used.contains(name), fused) << name << ": " << at;
    }
    // The format-free fast fusions.
    EXPECT_EQ(used.contains(kg::kMoeRouterName), fast) << at;
    EXPECT_EQ(used.contains(kg::kMoeCombineName), fused) << at;
    EXPECT_EQ(used.contains(kg::kQsaPoolName), fast) << at;
    EXPECT_EQ(used.contains(kg::kQsaPrepName), fast) << at;
    EXPECT_EQ(used.contains(kg::kGdnHistoryName), fast) << at;
    EXPECT_EQ(used.contains(kg::kGdnNormGateName), fused) << at;
    EXPECT_EQ(used.contains(kg::kGdnStepName),
              fast && static_cast<std::int64_t>(rows) <= kg::kGatedDeltaNetLanesTokens)
        << at;
    EXPECT_EQ(used.contains(kg::kSwiGluName), !vecq_experts) << at;
    const bool sparse = fast && chunk->qsa_select;
    EXPECT_EQ(used.contains(kg::kQsaAttnName), sparse) << at;
    EXPECT_EQ(used.contains(kg::kFlashAttnMmaName), !sparse) << at;
    // The BF16 indexer projections: llmpalooza's BF16 GEMM past 16 rows where
    // fused.
    EXPECT_EQ(used.contains(kg::kGemmBf16Name),
              fused && static_cast<std::int64_t>(rows) > kg::kQwen38Bf16Rows)
        << at;
    auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
    ASSERT_TRUE(placed.has_value()) << Why(placed);
    EXPECT_EQ(graph->logits->ne[0], 248320);
  }
  // A GGUF artifact's graph takes no CUTLASS layout, verify or drafter
  // streams; and a type its products do not take is refused, named.
  std::vector<std::int32_t> history(8, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, h, history, 0, 8);
  ASSERT_TRUE(chunk.has_value());
  const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, 8);
  for (const auto& options :
       {kg::Qwen38GraphOptions{.expert_stride = strides,
                               .experts = kg::Qwen38GraphOptions::Experts::kCutlass},
        kg::Qwen38GraphOptions{.expert_stride = strides, .verify = true},
        kg::Qwen38GraphOptions{
            .expert_stride = strides, .export_streams = true, .stream_rows = 16}}) {
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    EXPECT_NE(Why(kg::BuildQwen38Graph(*arena, p, *binding, shape, options)).find("GGUF"),
              std::string::npos);
  }
  auto iq1m = resources;
  for (md::Qwen38Resource& r : iq1m) {
    if (r.roles[0] == "blk.2.attn_qkv.weight") {
      r.type = "IQ1_M";  // no tile kernel upstream: not a type the products take
    }
  }
  auto rebound = md::BindQwen38(p, "qwen4exp", iq1m);
  ASSERT_TRUE(rebound.has_value()) << Why(rebound);
  auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(arena.has_value());
  EXPECT_NE(Why(kg::BuildQwen38Graph(*arena, p, *rebound, shape, {.expert_stride = strides}))
                .find("IQ1_M"),
            std::string::npos);
  // A token table of a product type GGML's get_rows has no case for
  // (NVFP4: getrows.cu aborts) is refused when the graph is built.
  auto nvfp4 = resources;
  for (md::Qwen38Resource& r : nvfp4) {
    if (r.roles[0] == "token_embd.weight") {
      r.type = "NVFP4";
    }
  }
  auto nvfp4_bound = md::BindQwen38(p, "qwen4exp", nvfp4);
  ASSERT_TRUE(nvfp4_bound.has_value()) << Why(nvfp4_bound);
  auto nvfp4_arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(nvfp4_arena.has_value());
  EXPECT_NE(
      Why(kg::BuildQwen38Graph(*nvfp4_arena, p, *nvfp4_bound, shape, {.expert_stride = strides}))
          .find("token_embd"),
      std::string::npos);
}

TEST(Qwen38Test, DeviceMaskTargetGraphsAuthenticateDtypesAndPreserveFallbackMetadata) {
  const auto& p = md::Qwen38Flash();
  for (const bool gguf : {false, true}) {
    auto binding = md::BindQwen38(p, "qwen4exp", gguf ? GgufLike(p) : ArtifactLike(p));
    ASSERT_TRUE(binding.has_value()) << Why(binding);
    const std::vector<std::uint64_t> strides(p.layers, gguf ? kGgufStride : kExpertStride);
    for (const bool exact : {false, true}) {
      for (const bool selecting : {false, true}) {
        const kg::Qwen38ChunkShape shape{.rows = 3,
                                         .n_kv = selecting ? 2304 : 256,
                                         .cells = 4096,
                                         .outputs = 1,
                                         .qsa_select = selecting,
                                         .qsa_blocks = selecting ? 576 : 0};
        for (const bool device : {false, true}) {
          SCOPED_TRACE(std::format("gguf {}, exact {}, selecting {}, device {}", gguf, exact,
                                   selecting, device));
          auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
          ASSERT_TRUE(arena.has_value());
          auto graph = kg::BuildQwen38Graph(
              *arena, p, *binding, shape,
              {.expert_stride = strides, .exact = exact, .device_masks = device});
          ASSERT_TRUE(graph.has_value()) << Why(graph);
          auto inputs = graph->inputs();
          std::uint64_t next = std::uint64_t{1} << 40U;
          const auto bind = [&](ggml_tensor* t) {
            if (t != nullptr && t->data == nullptr) {
              kg::TensorArena::Bind(t, next);
              next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
            }
          };
          for (auto* t : inputs) bind(t);
          for (auto* node : graph->nodes)
            for (auto* source : node->src)
              if (source != nullptr && source->op == GGML_OP_NONE && source->view_src == nullptr)
                bind(source);
          kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
          // Both formats' fast sparse selection already creates its own masks.
          const bool sparse = selecting && !exact;
          EXPECT_EQ(graph->mask == nullptr, sparse);
          EXPECT_EQ(graph->mask_f32 != nullptr, selecting && exact);
          EXPECT_EQ(graph->cell_block != nullptr, selecting && exact);
          for (auto* metadata :
               {graph->cell_block, graph->block_cells, graph->block_pos, graph->block_bias})
            if (metadata != nullptr) EXPECT_EQ(std::ranges::count(inputs, metadata), 1);
          for (auto* mask : {graph->mask, graph->mask_f32}) {
            if (mask == nullptr) continue;
            const auto dtype = mask == graph->mask ? GGML_TYPE_F16 : GGML_TYPE_F32;
            ASSERT_EQ(mask->type, dtype);
            auto charge = llmp::engine::GraphMaskSourceBytes(
                mask, graph->positions, graph->nodes, inputs, device, 0, 3,
                static_cast<std::uint32_t>(shape.n_kv), 4096, 0, 4096, kg::CausalMaskRows::kExact,
                dtype);
            ASSERT_TRUE(charge.has_value()) << Why(charge);
            EXPECT_EQ(*charge, device ? 0U : ggml_nbytes(mask));
            if (!device) continue;
            EXPECT_EQ(std::ranges::count(graph->nodes, mask), 1);
            EXPECT_EQ(std::ranges::count(inputs, mask), 0);
            EXPECT_EQ(mask->src[0], graph->positions);
            const auto check = [&](const std::vector<ggml_tensor*>& source_inputs) {
              return llmp::engine::GraphMaskSourceBytes(
                  mask, graph->positions, graph->nodes, source_inputs, true, 0, 3,
                  static_cast<std::uint32_t>(shape.n_kv), 4096, 0, 4096, kg::CausalMaskRows::kExact,
                  dtype);
            };
            auto duplicate = inputs;
            duplicate.push_back(mask);
            EXPECT_FALSE(check(duplicate));
            for (const std::size_t reserved : {6U, 7U}) {
              // The custom-op header occupies the first 32 bytes.
              auto* parameter = reinterpret_cast<std::byte*>(mask->op_params) + 32 + reserved * 4;
              const auto saved = kg::LlmpOpInt(mask, static_cast<int>(reserved));
              const std::int32_t invalid = 1;
              std::memcpy(parameter, &invalid, sizeof(invalid));
              EXPECT_FALSE(check(inputs));
              std::memcpy(parameter, &saved, sizeof(saved));
            }
            auto* source = mask->src[0];
            mask->src[0] = graph->tokens;
            EXPECT_FALSE(check(inputs));
            mask->src[0] = source;
            EXPECT_FALSE(llmp::engine::GraphMaskSourceBytes(
                mask, graph->positions, graph->nodes, inputs, true, 0, 3,
                static_cast<std::uint32_t>(shape.n_kv), 4096, 0, 4096, kg::CausalMaskRows::kExact,
                dtype == GGML_TYPE_F16 ? GGML_TYPE_F32 : GGML_TYPE_F16));
          }
          auto plan = kg::PlanGraph(graph->nodes, false, ModelDevice());
          ASSERT_TRUE(plan.has_value()) << Why(plan);
          auto placed = kg::PlaceActivations(graph->nodes, *plan, inputs, 256);
          ASSERT_TRUE(placed.has_value()) << Why(placed);
          // Producers contribute activations and a real launch, never staged input bytes.
          EXPECT_GT(placed->extent, 0U);
          EXPECT_EQ(std::ranges::count_if(plan->steps,
                                          [](const auto& step) {
                                            return step.implementation == kg::kGemma4MaskName;
                                          }),
                    device && !sparse ? (selecting ? 2 : 1) : 0);
        }
      }
    }
  }
  // The reference plane's RE-037 consumer bound survives host-mask omission.
  EXPECT_FALSE(md::Qwen38State(p, 131072, 8192, true));
}

}  // namespace
