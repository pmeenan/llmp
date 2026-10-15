// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's DSpark drafter (model/dspark.h) and speculation's parts of
// the DeepSeek adapter (model/dsv4.h), in every profile:
// - the drafter's binding of a resource list shaped and typed as its GGUF's
//   (dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf), its target's tables, and its
//   refusals;
// - the drafter's ring and the injection's cells;
// - a verify chunk's writes, by row and scratch, for the snapshot a
//   rejected draft is restored from; the representations' truncation.

#include "model/dspark.h"

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
#include "model/dsv4.h"
#include "model/state.h"

namespace {

namespace md = llmp::model;
using llmp::test_support::Failed;

template <typename T>
std::string Why(const std::expected<T, std::string>& result) {
  return Failed(result).value_or(std::string());
}

void Add(std::vector<md::Dsv4Resource>& r, std::string name, std::string type,
         std::vector<std::uint64_t> ne) {
  r.push_back({.roles = {std::move(name)}, .type = std::move(type), .ne = std::move(ne)});
}

// A DeepSeek V4 block's tensors as its GGUFs have them (window-only layers:
// no compressor or indexer), expert arrays appended to `arrays`.
void Block(std::vector<md::Dsv4Resource>& r, std::vector<md::Dsv4Resource>& arrays,
           std::uint32_t il, bool hash, std::string_view experts) {
  const std::string n = std::format("blk.{}.", il);
  Add(r, n + "attn_norm.weight", "F32", {4096});
  Add(r, n + "attn_sinks.weight", "F32", {64});
  Add(r, n + "attn_q_a.weight", "Q8_0", {4096, 1024});
  Add(r, n + "attn_q_a_norm.weight", "F32", {1024});
  Add(r, n + "attn_q_b.weight", "Q8_0", {1024, 32768});
  Add(r, n + "attn_kv.weight", "Q8_0", {4096, 512});
  Add(r, n + "attn_kv_a_norm.weight", "F32", {512});
  Add(r, n + "attn_output_a.weight", "Q8_0", {4096, 8192});
  Add(r, n + "attn_output_b.weight", "Q8_0", {8192, 4096});
  for (const char* hc : {"hc_attn_", "hc_ffn_"}) {
    Add(r, n + hc + "fn.weight", "F32", {16384, 24});
    Add(r, n + hc + "base.weight", "F32", {24});
    Add(r, n + hc + "scale.weight", "F32", {3});
  }
  Add(r, n + "ffn_norm.weight", "F32", {4096});
  Add(r, n + "ffn_gate_inp.weight", "BF16", {4096, 256});
  if (hash) {
    Add(r, n + "ffn_gate_tid2eid.weight", "I32", {6, 129280});
  } else {
    Add(r, n + "exp_probs_b.bias", "F32", {256});
  }
  Add(r, n + "ffn_gate_shexp.weight", "Q8_0", {4096, 2048});
  Add(r, n + "ffn_up_shexp.weight", "Q8_0", {4096, 2048});
  Add(r, n + "ffn_down_shexp.weight", "Q8_0", {2048, 4096});
  for (const auto& [name, ne] :
       {std::tuple{"ffn_gate_exps.weight", std::vector<std::uint64_t>{4096, 2048}},
        std::tuple{"ffn_up_exps.weight", std::vector<std::uint64_t>{4096, 2048}},
        std::tuple{"ffn_down_exps.weight", std::vector<std::uint64_t>{2048, 4096}}}) {
    arrays.push_back({.roles = {n + name},
                      .type = std::string(experts),
                      .ne = ne,
                      .expert_array = true,
                      .count = 256});
  }
}

// The target: DeepSeek V4 Flash's 43 layers (their compressors left out:
// only the head and table matter to the drafter, and window-only layers
// bind under a profile of window-only layers).
std::vector<md::Dsv4Resource> Target(const md::Dsv4Profile& p) {
  std::vector<md::Dsv4Resource> r;
  std::vector<md::Dsv4Resource> arrays;
  Add(r, "token_embd.weight", "Q5_K", {4096, 129280});
  Add(r, "output_norm.weight", "F32", {4096});
  Add(r, "output.weight", "Q4_K", {4096, 129280});
  Add(r, "output_hc_fn.weight", "F32", {16384, 4});
  Add(r, "output_hc_base.weight", "F32", {4});
  Add(r, "output_hc_scale.weight", "F32", {1});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    Block(r, arrays, il, il < p.hash_layers, "IQ2_XS");
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// The drafter's 81 tensors, as the Q8_0 GGUF lists them.
std::vector<md::Dsv4Resource> Drafter() {
  std::vector<md::Dsv4Resource> r;
  std::vector<md::Dsv4Resource> arrays;
  for (std::uint32_t il = 0; il < 3; ++il) {
    Block(r, arrays, il, false, "MXFP4");
  }
  Add(r, "enc.output_norm.weight", "F32", {4096});
  Add(r, "fc.weight", "Q8_0", {12288, 4096});
  Add(r, "conf_proj.weight", "BF16", {4352, 1});
  Add(r, "output_hc_base.weight", "F32", {4});
  Add(r, "output_hc_fn.weight", "F32", {16384, 4});
  Add(r, "output_hc_scale.weight", "F32", {1});
  Add(r, "markov_w1.weight", "BF16", {256, 129280});
  Add(r, "markov_w2.weight", "BF16", {256, 129280});
  Add(r, "output_norm.weight", "F32", {4096});
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// DeepSeek V4 Flash with every layer window-only, to bind Target().
md::Dsv4Profile WindowOnlyTarget() {
  md::Dsv4Profile p = md::Dsv4Flash();
  p.compress_ratios.assign(p.layers, 0);
  return p;
}

TEST(DsparkTest, TheProfileIsTheGgufs) {
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  EXPECT_EQ(d.blocks.layers, 3U);
  EXPECT_EQ(d.blocks.hash_layers, 0U);
  EXPECT_EQ(d.blocks.compress_ratios, (std::vector<std::uint32_t>{0, 0, 0}));
  EXPECT_EQ(d.block_size, 5U);
  EXPECT_EQ(d.target_layers, (std::vector<std::uint32_t>{41, 42, 43}));
  EXPECT_EQ(d.mask_token, 128799);
  EXPECT_EQ(d.blocks.window, 128U);
  EXPECT_EQ(d.blocks.rope_base, 10000.0f);
  EXPECT_GE(d.ring, d.blocks.window + d.block_size);
}

TEST(DsparkTest, BindsTheDraftersTensorsAndItsTargetsTables) {
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  const md::Dsv4Profile target_profile = WindowOnlyTarget();
  const std::vector<md::Dsv4Resource> target_resources = Target(target_profile);
  auto target = md::BindDsv4(target_profile, "deepseek4", target_resources);
  ASSERT_TRUE(target.has_value()) << Why(target);
  const std::vector<md::Dsv4Resource> drafter = Drafter();
  auto b = md::BindDspark(d, "dflash", drafter, target_profile, *target);
  ASSERT_TRUE(b.has_value()) << Why(b);
  EXPECT_EQ(b->blocks.layers.size(), 3U);
  EXPECT_EQ(b->fc.type, "Q8_0");
  EXPECT_EQ(b->fc.ne, (std::vector<std::uint64_t>{12288, 4096}));
  EXPECT_EQ(b->markov_w1.type, "BF16");
  EXPECT_EQ(b->blocks.layers[2].down_exps.type, "MXFP4");
  // The head and the token table are the target's resources.
  EXPECT_EQ(b->blocks.output.index, target->output.index);
  EXPECT_EQ(b->blocks.output.type, "Q4_K");
  EXPECT_EQ(b->blocks.token_embd.index, target->token_embd.index);
  // Its own norm and hyper-connection head.
  EXPECT_NE(b->blocks.output_norm.index, target->output_norm.index);

  // Refusals: another architecture; a role missing, of another shape, or
  // one the drafter does not read; a target of another width.
  EXPECT_NE(Why(md::BindDspark(d, "deepseek4", drafter, target_profile, *target)).find("dflash"),
            std::string::npos);
  {
    std::vector<md::Dsv4Resource> r = drafter;
    std::erase_if(r, [](const md::Dsv4Resource& x) { return x.roles[0] == "fc.weight"; });
    EXPECT_NE(Why(md::BindDspark(d, "dflash", r, target_profile, *target)).find("fc.weight"),
              std::string::npos);
  }
  {
    std::vector<md::Dsv4Resource> r = drafter;
    for (auto& x : r) {
      if (x.roles[0] == "markov_w2.weight") {
        x.ne = {256, 129279};
      }
    }
    EXPECT_NE(Why(md::BindDspark(d, "dflash", r, target_profile, *target)).find("markov_w2"),
              std::string::npos);
  }
  {
    std::vector<md::Dsv4Resource> r = drafter;
    Add(r, "token_embd.weight", "Q5_K", {4096, 129280});  // the drafter ships none
    EXPECT_FALSE(md::BindDspark(d, "dflash", r, target_profile, *target).has_value());
  }
  {
    md::Dsv4Profile narrow = target_profile;
    narrow.width = 2048;
    EXPECT_FALSE(md::BindDspark(d, "dflash", drafter, narrow, *target).has_value());
  }
  {
    // A target whose table is not [width, vocab]: the drafts index it.
    md::Dsv4Binding short_table = *target;
    short_table.token_embd.ne = {4096, 1000};
    EXPECT_FALSE(md::BindDspark(d, "dflash", drafter, target_profile, short_table).has_value());
  }
  {
    md::DsparkProfile compressed = d;
    compressed.blocks.compress_ratios[1] = 4;
    EXPECT_FALSE(
        md::BindDspark(compressed, "dflash", drafter, target_profile, *target).has_value());
  }
}

TEST(DsparkTest, TheRingIsSizedAndAChunkInjectsItsLastRows) {
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  auto s = md::DsparkState(d, 3);
  ASSERT_TRUE(s.has_value()) << Why(s);
  EXPECT_EQ(s->ring, 256U);
  EXPECT_EQ(s->offsets.size(), 3U);
  EXPECT_EQ(s->bytes, 3U * 512U * 256U * 2U);
  EXPECT_TRUE(md::IsValid(s->Representation(4)));
  EXPECT_TRUE(s->Representation(4).CanTruncate());
  EXPECT_FALSE(s->Representation(0).CanTruncate());
  // A chunk injects its last rows, each position a cell of its own.
  const md::DsparkInjection all = md::DsparkInject(*s, 10, 4);
  EXPECT_EQ(all.first, 0U);
  EXPECT_EQ(all.cells, (std::vector<std::int64_t>{10, 11, 12, 13}));
  const md::DsparkInjection last = md::DsparkInject(*s, 1000, 512);
  EXPECT_EQ(last.first, 256U);
  ASSERT_EQ(last.cells.size(), 256U);
  EXPECT_EQ(last.cells.front(), (1000 + 256) % 256);
  std::set<std::int64_t> distinct(last.cells.begin(), last.cells.end());
  EXPECT_EQ(distinct.size(), 256U);
}

TEST(DsparkTest, AVerifysWritesAreItsRowsAndItsScratch) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  auto s = md::Dsv4State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  using K = md::Dsv4StateTensor::Kind;
  const auto tensor = [&](std::uint32_t layer, K kind) {
    return s->tensors[static_cast<std::size_t>(s->Find(layer, kind))];
  };
  const auto has = [](const std::vector<md::StateRange>& ranges, std::uint64_t offset) {
    return std::ranges::any_of(ranges, [&](const md::StateRange& r) { return r.offset == offset; });
  };
  // Rows at 125..128: position 127 completes an HCA block (row 2) and a
  // CSA block, position 128 nothing.
  auto chunk = md::Dsv4Chunk(p, *s, 125, 4);
  ASSERT_TRUE(chunk.has_value());
  const md::Dsv4Writes w = md::Dsv4ChunkWrites(p, *s, *chunk);
  ASSERT_EQ(w.rows.size(), 4U);
  // Every row writes its window cell in all 43 layers.
  for (std::uint32_t i = 0; i < 4; ++i) {
    for (std::uint32_t il = 0; il < p.layers; ++il) {
      const md::Dsv4StateTensor raw = tensor(il, K::kRawK);
      EXPECT_TRUE(has(w.rows[i], raw.offset + ((std::uint64_t{125} + i) * 1024))) << i << " " << il;
    }
  }
  // The HCA block (row 2) in layer 3's compressed cache at row 0; its ring
  // rows at pos mod 128.
  const md::Dsv4StateTensor hca = tensor(3, K::kHcaK);
  EXPECT_TRUE(has(w.rows[2], hca.offset));
  const md::Dsv4StateTensor hca_ring = tensor(3, K::kHcaStateKv);
  EXPECT_TRUE(has(w.rows[3], hca_ring.offset));                                // 128 mod 128
  EXPECT_TRUE(has(w.rows[0], hca_ring.offset + (std::uint64_t{125} * 2048)));  // 125
  // The CSA block 31 (positions 124..127, completed by row 2).
  EXPECT_TRUE(has(w.rows[2], tensor(2, K::kCsaK).offset + (std::uint64_t{31} * 1024)));
  EXPECT_TRUE(has(w.rows[2], tensor(2, K::kLidK).offset + (std::uint64_t{31} * 256)));
  // No HCA dummy here (a real block); CSA's one block is real: no scratch
  // in HCA or CSA caches.
  EXPECT_TRUE(w.scratch.empty());
  // Rows at 200..202: no block completes for HCA: its dummy writes the
  // cache's last row, whatever is accepted.
  auto quiet = md::Dsv4Chunk(p, *s, 200, 3);
  ASSERT_TRUE(quiet.has_value());
  const md::Dsv4Writes q = md::Dsv4ChunkWrites(p, *s, *quiet);
  EXPECT_TRUE(has(q.scratch, hca.offset + ((hca.ne1 - 1) * 1024)));
  // Aligned, and disjoint within the chunk.
  std::vector<md::StateRange> all = q.scratch;
  for (const auto& row : q.rows) {
    all.insert(all.end(), row.begin(), row.end());
  }
  std::ranges::sort(all, {}, &md::StateRange::offset);
  for (std::size_t i = 0; i < all.size(); ++i) {
    EXPECT_EQ(all[i].offset % 256, 0U);
    EXPECT_EQ(all[i].bytes % 256, 0U);
    if (i > 0) {
      EXPECT_LE(all[i - 1].offset + all[i - 1].bytes, all[i].offset);
    }
  }
  // With a verify, every representation truncates.
  for (const auto& r : s->Representations(4)) {
    EXPECT_TRUE(md::IsValid(r)) << r.name;
    EXPECT_TRUE(r.CanTruncate()) << r.name;
  }
  // The drafter's ring: each injected row's cell in its three blocks.
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  auto ring = md::DsparkState(d, 3);
  ASSERT_TRUE(ring.has_value());
  const auto dw = md::DsparkWrites(d, *ring, 254, 4);
  ASSERT_EQ(dw.size(), 4U);
  EXPECT_EQ(dw[3].size(), 3U);
  EXPECT_EQ(dw[3][1].offset, ring->offsets[1] + std::uint64_t{1024});  // 257 mod 256
}

// An incremental spill (engine/live_state.h SpillTrack) writes back only
// the ranges StateWrites(position) names for the first position a job
// since the last spill wrote from (a chunk or verify at n_past records
// n_past; a wave pos - 1), so everything a verify writes, its rows and its
// scratch rows, and therefore everything its owed restore writes later,
// lies inside DeepSeek's checkpoint writes from its n_past. In both
// window forms, across ring wraps and block boundaries.
TEST(DsparkTest, AVerifysWritesLieInsideItsCheckpointWrites) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  for (const md::Dsv4Window window : {md::Dsv4Window::kFull, md::Dsv4Window::kRing}) {
    auto s = md::Dsv4State(p, 4096, 512, window);
    ASSERT_TRUE(s.has_value()) << Why(s);
    for (const auto& [n_past, rows] : std::array<std::pair<std::uint32_t, std::uint32_t>, 7>{
             {{0, 4}, {125, 4}, {127, 1}, {200, 3}, {1023, 2}, {3000, 8}, {4000, 8}}}) {
      auto chunk = md::Dsv4Chunk(p, *s, n_past, rows);
      ASSERT_TRUE(chunk.has_value()) << n_past << ": " << Why(chunk);
      const md::Dsv4Writes w = md::Dsv4ChunkWrites(p, *s, *chunk);
      auto recorded = md::Dsv4CheckpointWrites(*s, n_past);
      ASSERT_TRUE(recorded.has_value()) << Why(recorded);
      const auto covered = [&](const md::StateRange& r) {
        return std::ranges::any_of(*recorded, [&](const md::StateRange& c) {
          return r.offset >= c.offset && r.offset + r.bytes <= c.offset + c.bytes;
        });
      };
      std::vector<md::StateRange> all = w.scratch;
      for (const auto& row : w.rows) {
        all.insert(all.end(), row.begin(), row.end());
      }
      ASSERT_FALSE(all.empty());
      for (const md::StateRange& r : all) {
        EXPECT_TRUE(covered(r)) << static_cast<int>(window) << " " << n_past << "+" << rows << ": ["
                                << r.offset << ", +" << r.bytes << ")";
      }
    }
  }
}

}  // namespace
