// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's DSpark drafter (model/dspark.h) and speculation's parts of
// the DeepSeek adapter and graph (model/dsv4.h, kernels/ggml/dsv4_graph.h),
// in every profile:
// - the drafter's binding of a resource list shaped and typed as its GGUF's
//   (dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf), its target's tables, and its
//   refusals;
// - a draft block's inputs: the non-causal window over the ring, the
//   anchor and mask tokens; the injection's cells;
// - a verify chunk's writes, by row and scratch, for the snapshot a
//   rejected draft is restored from; the mask widths a verify may not
//   cross; the representations' truncation;
// - a verify's graph: attention per query row, the features and the
//   injection, planned by the row-invariant implementations only (D-092);
//   the draft block's graph with its Markov head.

#include "model/dspark.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "model/dsv4.h"
#include "model/state.h"

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

TEST(DsparkTest, DeviceBlockInputsSkipOnlyTheMaterializedMask) {
  const auto& profile = md::DsparkDeepSeekV4Flash();
  auto state = md::DsparkState(profile, 5);
  ASSERT_TRUE(state);
  for (const std::uint32_t rows : {1U, 3U, 5U}) {
    for (const std::uint32_t pos : {0U, 254U, 300U, static_cast<std::uint32_t>(INT32_MAX) - rows}) {
      const auto host = md::DsparkBlock(profile, *state, pos, 17, rows);
      const auto device = md::DsparkBlock(profile, *state, pos, 17, rows, false);
      ASSERT_TRUE(host);
      ASSERT_TRUE(device);
      EXPECT_EQ(host->pos0, device->pos0);
      EXPECT_EQ(host->rows, device->rows);
      EXPECT_EQ(host->tokens, device->tokens);
      EXPECT_EQ(host->positions, device->positions);
      EXPECT_EQ(host->cells, device->cells);
      EXPECT_EQ(host->mask.size(), rows * state->ring);
      EXPECT_TRUE(device->mask.empty());
    }
    EXPECT_FALSE(md::DsparkBlock(profile, *state, static_cast<std::uint32_t>(INT32_MAX) - rows + 1,
                                 17, rows, false));
  }
  auto malformed = *state;
  malformed.ring = 128;
  EXPECT_FALSE(md::DsparkBlock(profile, malformed, 0, 17, 3, false));
}

TEST(DsparkTest, ADraftBlockSeesItselfAndItsWindowNonCausally) {
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  auto s = md::DsparkState(d, 3);
  ASSERT_TRUE(s.has_value()) << Why(s);
  EXPECT_EQ(s->ring, 256U);
  EXPECT_EQ(s->offsets.size(), 3U);
  EXPECT_EQ(s->bytes, 3U * 512U * 256U * 2U);
  EXPECT_TRUE(md::IsValid(s->Representation(4)));
  EXPECT_TRUE(s->Representation(4).CanTruncate());
  EXPECT_FALSE(s->Representation(0).CanTruncate());
  // A block at 300: cells 300 % 256 = 44 onwards.
  auto b = md::DsparkBlock(d, *s, 300, 17, 3);
  ASSERT_TRUE(b.has_value()) << Why(b);
  EXPECT_EQ(b->tokens, (std::vector<std::int32_t>{17, 128799, 128799}));
  EXPECT_EQ(b->positions, (std::vector<std::int32_t>{300, 301, 302}));
  EXPECT_EQ(b->cells, (std::vector<std::int64_t>{44, 45, 46}));
  const auto visible = [&](std::uint32_t row, std::uint32_t cell) {
    return b->mask[(std::size_t{row} * 256) + cell] == md::kHalfZero;
  };
  for (std::uint32_t row = 0; row < 3; ++row) {
    // Every block position, later ones too (non-causal).
    for (std::uint32_t c = 44; c <= 46; ++c) {
      EXPECT_TRUE(visible(row, c)) << row << " " << c;
    }
    // The window: positions 300 + row - 127 .. 299, in cells (p mod 256).
    const std::uint32_t oldest = 300 + row - 127;
    for (std::uint32_t p = oldest; p < 300; ++p) {
      EXPECT_TRUE(visible(row, p % 256)) << row << " " << p;
    }
    EXPECT_FALSE(visible(row, (oldest - 1) % 256)) << row;
    // Cells past the block hold older positions (outside the window).
    EXPECT_FALSE(visible(row, 47)) << row;
  }
  // Near the start: nothing before position 0.
  auto early = md::DsparkBlock(d, *s, 2, 5, 3);
  ASSERT_TRUE(early.has_value());
  for (std::uint32_t c = 0; c < 256; ++c) {
    EXPECT_EQ(early->mask[c] == md::kHalfZero, c <= 4) << c;
  }
  // Refusals: past the block, no rows, an anchor outside the vocabulary.
  EXPECT_FALSE(md::DsparkBlock(d, *s, 0, 5, 4).has_value());
  EXPECT_FALSE(md::DsparkBlock(d, *s, 0, 5, 0).has_value());
  EXPECT_FALSE(md::DsparkBlock(d, *s, 0, 129280, 3).has_value());
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
  EXPECT_GE(md::Dsv4VerifySnapshotBytes(p, *s, 4), [&] {
    std::uint64_t sum = 0;
    for (const auto& r : all) {
      sum += r.bytes;
    }
    return sum;
  }());
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

TEST(DsparkTest, AVerifyMayNotCrossAMaskWidth) {
  const md::Dsv4Profile& p = md::Dsv4Flash();
  auto s = md::Dsv4State(p, 8192, 512);
  ASSERT_TRUE(s.has_value());
  EXPECT_TRUE(md::Dsv4SameWidths(*s, 100, 4));
  // Rows 253..256: the one-row step at 253 reads 256 cells, the chunk 512.
  EXPECT_FALSE(md::Dsv4SameWidths(*s, 253, 4));
  EXPECT_TRUE(md::Dsv4SameWidths(*s, 253, 3));
  EXPECT_TRUE(md::Dsv4SameWidths(*s, 256, 4));
  // The compressed rows' widths: CSA's mask widens past 256 rows once
  // position 1027 completes the 257th block.
  EXPECT_FALSE(md::Dsv4SameWidths(*s, 1022, 3));
  EXPECT_TRUE(md::Dsv4SameWidths(*s, 1024, 3));
  EXPECT_FALSE(md::Dsv4SameWidths(*s, 1024, 4));
  EXPECT_TRUE(md::Dsv4SameWidths(*s, 7, 1));
  EXPECT_FALSE(md::Dsv4SameWidths(*s, 7, 0));
}

// Upstream's routing as a model (dsv4_test.cc's).
kg::DeviceChoices ModelDevice(bool row_invariant) {
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
      },
      .row_invariant = row_invariant};
}

// Binds every leaf at a distinct address and every computed node too.
template <typename Graph>
void BindAll(Graph& graph, std::span<ggml_tensor* const> nodes) {
  std::uint64_t next = std::uint64_t{1} << 40U;
  const auto bind_leaf = [&](ggml_tensor* t) {
    if (t != nullptr && t->data == nullptr) {
      kg::TensorArena::Bind(t, next);
      next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
    }
  };
  for (ggml_tensor* t : graph.inputs()) {
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
}

TEST(DsparkTest, AVerifysGraphIsRowInvariantAndInjectsTheDraftersRing) {
  const md::Dsv4Profile target_profile = WindowOnlyTarget();
  const std::vector<md::Dsv4Resource> target_resources = Target(target_profile);
  auto target = md::BindDsv4(target_profile, "deepseek4", target_resources);
  ASSERT_TRUE(target.has_value()) << Why(target);
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  const std::vector<md::Dsv4Resource> drafter = Drafter();
  auto dbinding = md::BindDspark(d, "dflash", drafter, target_profile, *target);
  ASSERT_TRUE(dbinding.has_value()) << Why(dbinding);
  auto state = md::Dsv4State(target_profile, 4096, 512);
  ASSERT_TRUE(state.has_value());
  auto chunk = md::Dsv4Chunk(target_profile, *state, 300, 4);
  ASSERT_TRUE(chunk.has_value());
  const kg::Dsv4ChunkShape shape = kg::Dsv4ShapeOf(*state, *chunk);
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(target_profile));
  ASSERT_TRUE(arena.has_value());
  const kg::Dsv4GraphOptions options{
      .expert_stride = {},
      .row_invariant = true,
      .features = d.target_layers,
      .inject = kg::Dsv4Injection{.profile = &d, .binding = &*dbinding, .rows = 4, .ring = 256}};
  auto graph = kg::BuildDsv4Graph(*arena, target_profile, *target, shape, options);
  ASSERT_TRUE(graph.has_value()) << Why(graph);
  // Attention per query row: 43 layers x 4 rows.
  const auto count = [&](ggml_op op) {
    return std::ranges::count_if(graph->nodes, [&](const ggml_tensor* n) { return n->op == op; });
  };
  EXPECT_EQ(count(GGML_OP_FLASH_ATTN_EXT), 43 * 4);
  // The features: three layers' stream means, per row.
  ASSERT_NE(graph->features, nullptr);
  EXPECT_EQ(graph->features->ne[0], 3 * 4096);
  EXPECT_EQ(graph->features->ne[1], 4);
  // The injection: one ring write per drafter block, at the input cells.
  ASSERT_TRUE(graph->inject.has_value());
  if (!graph->inject) {
    return;
  }
  const kg::DsparkInjectTensors& inject = *graph->inject;
  EXPECT_EQ(inject.ring.size(), 3U);
  EXPECT_EQ(inject.cells->ne[0], 4);
  EXPECT_EQ(graph->inputs().back(), inject.cells);
  std::size_t ring_writes = 0;
  for (const ggml_tensor* n : graph->nodes) {
    if (n->op == GGML_OP_SET_ROWS &&
        std::ranges::find(inject.ring, n->src[2]) != inject.ring.end()) {
      ++ring_writes;
    }
  }
  EXPECT_EQ(ring_writes, 3U);
  BindAll(*graph, graph->nodes);
  auto plan = kg::PlanGraph(graph->nodes, false, ModelDevice(true));
  ASSERT_TRUE(plan.has_value()) << Why(plan);
  std::set<std::string_view> used;
  for (const auto& step : plan->steps) {
    used.insert(step.implementation);
  }
  EXPECT_TRUE(used.contains(kg::kMulMatVecQRows));
  EXPECT_TRUE(used.contains(kg::kMulMatIdVecQRows));
  EXPECT_TRUE(used.contains(kg::kMulMatVecFRows));
  for (const std::string_view upstream :
       {kg::kMulMatVecQ, kg::kMulMatQ, kg::kMulMatIdVecQ, kg::kMulMatIdQ, kg::kMulMatVector,
        kg::kMulMatTensorCore, kg::kMulMatCublas}) {
    EXPECT_FALSE(used.contains(upstream)) << upstream;
  }
  auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256,
                                     std::vector<ggml_tensor*>{graph->logits});
  ASSERT_TRUE(placed.has_value()) << Why(placed);
  // Nor fusion: upstream's fused vector products follow the column count.
  EXPECT_NE(Why(kg::PlanGraph(graph->nodes, true, ModelDevice(true))).find("without fusion"),
            std::string::npos);
  // A row-invariant plan takes no product wider than 8 columns.
  auto wide_chunk = md::Dsv4Chunk(target_profile, *state, 0, 9);
  ASSERT_TRUE(wide_chunk.has_value());
  auto wide_arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(target_profile));
  ASSERT_TRUE(wide_arena.has_value());
  auto wide = kg::BuildDsv4Graph(*wide_arena, target_profile, *target,
                                 kg::Dsv4ShapeOf(*state, *wide_chunk), {.row_invariant = true});
  ASSERT_TRUE(wide.has_value()) << Why(wide);
  BindAll(*wide, wide->nodes);
  EXPECT_NE(Why(kg::PlanGraph(wide->nodes, false, ModelDevice(true))).find("row-invariant"),
            std::string::npos);
  // An injection without features is refused.
  auto third = kg::TensorArena::Create(kg::Dsv4GraphTensors(target_profile));
  ASSERT_TRUE(third.has_value());
  EXPECT_FALSE(
      kg::BuildDsv4Graph(*third, target_profile, *target, shape, {.inject = options.inject})
          .has_value());
}

TEST(DsparkTest, AFrontierHeadRetainsEveryFeatureAndPartialInjectionRow) {
  const md::Dsv4Profile p = WindowOnlyTarget();
  const std::vector<md::Dsv4Resource> resources = Target(p);
  auto target = md::BindDsv4(p, "deepseek4", resources);
  ASSERT_TRUE(target.has_value()) << Why(target);
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  const std::vector<md::Dsv4Resource> drafter = Drafter();
  auto dbinding = md::BindDspark(d, "dflash", drafter, p, *target);
  ASSERT_TRUE(dbinding.has_value()) << Why(dbinding);
  auto state = md::Dsv4State(p, 4096, 512, md::Dsv4Window::kRing);
  ASSERT_TRUE(state.has_value());
  auto chunk = md::Dsv4Chunk(p, *state, 300, 37, false);
  ASSERT_TRUE(chunk.has_value()) << Why(chunk);
  for (const std::int64_t outputs : {0, 1}) {
    auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    const kg::Dsv4GraphOptions options{
        .features = d.target_layers,
        .inject = kg::Dsv4Injection{.profile = &d, .binding = &*dbinding, .rows = 32, .ring = 256},
        .fused = true};
    auto graph =
        kg::BuildDsv4Graph(*arena, p, *target, kg::Dsv4ShapeOf(*state, *chunk, outputs), options);
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    EXPECT_EQ(graph->logits->ne[1], outputs == 0 ? 37 : 1);
    ASSERT_NE(graph->features, nullptr);
    EXPECT_EQ(graph->features->ne[0], 3 * p.width);
    EXPECT_EQ(graph->features->ne[1], 37);
    EXPECT_EQ(graph->Named("l_last-42")->ne[2], 37);
    EXPECT_EQ(graph->Named("inp_g_embeddings")->ne[1], 32);
    ASSERT_TRUE(graph->inject.has_value());
    EXPECT_EQ(graph->inject->cells->ne[0], 32);
    std::size_t ring_writes = 0;
    for (const ggml_tensor* node : graph->nodes) {
      if (node->op == GGML_OP_SET_ROWS &&
          std::ranges::find(graph->inject->ring, node->src[2]) != graph->inject->ring.end()) {
        ++ring_writes;
        EXPECT_EQ(node->src[0]->ne[1], 32);
      }
    }
    EXPECT_EQ(ring_writes, d.blocks.layers);
    for (std::uint32_t il = 0; il < d.blocks.layers; ++il) {
      EXPECT_EQ(graph->Named(std::format("kv_injected-{}", il))->ne[2], 32);
    }
    BindAll(*graph, graph->nodes);
    auto device = ModelDevice(false);
    device.fuse_norms = true;
    device.vector_floats = true;
    auto plan = kg::PlanGraph(graph->nodes, false, device);
    ASSERT_TRUE(plan.has_value()) << Why(plan);
    auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256,
                                       std::vector<ggml_tensor*>{graph->logits});
    ASSERT_TRUE(placed.has_value()) << Why(placed);
  }
}

TEST(DsparkTest, TheDraftBlocksGraphChainsTheMarkovHeadOnArgmax) {
  const md::Dsv4Profile target_profile = WindowOnlyTarget();
  const std::vector<md::Dsv4Resource> target_resources = Target(target_profile);
  auto target = md::BindDsv4(target_profile, "deepseek4", target_resources);
  ASSERT_TRUE(target.has_value());
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  const std::vector<md::Dsv4Resource> drafter = Drafter();
  auto dbinding = md::BindDspark(d, "dflash", drafter, target_profile, *target);
  ASSERT_TRUE(dbinding.has_value());
  auto arena = kg::TensorArena::Create(kg::DsparkGraphTensors(d, 3));
  ASSERT_TRUE(arena.has_value());
  auto graph = kg::BuildDsparkGraph(*arena, d, *dbinding, 3, 256, {});
  ASSERT_TRUE(graph.has_value()) << Why(graph);
  EXPECT_EQ(graph->inputs().size(), 5U);
  ASSERT_NE(graph->drafts, nullptr);
  EXPECT_EQ(graph->drafts->type, GGML_TYPE_I32);
  EXPECT_EQ(graph->drafts->ne[0], 3);
  EXPECT_EQ(graph->logits->ne[0], 129280);
  EXPECT_EQ(graph->logits->ne[1], 3);
  EXPECT_EQ(graph->core.nodes.back(), graph->drafts);
  // Two argmax links in the chain, and the drafts' own.
  EXPECT_EQ(std::ranges::count_if(
                graph->core.nodes,
                [](const ggml_tensor* n) { return kg::JitllmOpOf(n) == kg::JitllmOp::kArgmax; }),
            3);
  BindAll(*graph, graph->core.nodes);
  auto plan = kg::PlanGraph(graph->core.nodes, false, ModelDevice(false));
  ASSERT_TRUE(plan.has_value()) << Why(plan);
  std::set<std::string_view> used;
  for (const auto& step : plan->steps) {
    used.insert(step.implementation);
  }
  EXPECT_TRUE(used.contains(kg::kArgmaxName));
  EXPECT_TRUE(used.contains(kg::kFlashAttnMmaName));
  EXPECT_TRUE(used.contains(kg::kMulMatIdVecQ));
  EXPECT_FALSE(used.contains(kg::kFlashAttnMmaWideName));
  auto sharing_device = ModelDevice(false);
  sharing_device.wide_sparse_attention = true;
  auto sharing = kg::PlanGraph(graph->core.nodes, false, sharing_device);
  ASSERT_TRUE(sharing.has_value()) << Why(sharing);
  std::size_t attention = 0;
  for (const auto& step : sharing->steps) {
    if (step.operation == jitllm::execution::Operation::kFlashAttn) {
      ++attention;
      EXPECT_EQ(step.implementation, kg::kFlashAttnMmaWideName);
    }
  }
  // DSpark's pinned three blocks are window-only; HCA gating leaves
  // their opted-in selection unchanged through the same planner.
  EXPECT_EQ(attention, d.blocks.layers);
  // Refusals: a block past the drafter's, another ring, a verify's options.
  auto again = kg::TensorArena::Create(kg::DsparkGraphTensors(d, 6));
  ASSERT_TRUE(again.has_value());
  EXPECT_FALSE(kg::BuildDsparkGraph(*again, d, *dbinding, 6, 256, {}).has_value());
  EXPECT_FALSE(kg::BuildDsparkGraph(*again, d, *dbinding, 3, 512, {}).has_value());
  EXPECT_FALSE(
      kg::BuildDsparkGraph(*again, d, *dbinding, 3, 256, {.row_invariant = true}).has_value());
}

// Several requests' draft blocks as one graph (a DSpark wave's): the
// vector products joined (as many as one block has, each weight read once
// for every slot's rows), each slot's attention over its own ring, its
// Markov head and its drafts its own, those on concurrent lanes.
TEST(DsparkTest, DeviceBlockMasksUseEachActualPositionSegmentAndAreNotHostInputs) {
  const auto target_profile = WindowOnlyTarget();
  const auto resources = Target(target_profile);
  auto target = md::BindDsv4(target_profile, "deepseek4", resources);
  ASSERT_TRUE(target);
  const auto& profile = md::DsparkDeepSeekV4Flash();
  const auto draft_resources = Drafter();
  auto binding = md::BindDspark(profile, "dflash", draft_resources, target_profile, *target);
  ASSERT_TRUE(binding);
  for (const bool device : {false, true}) {
    const kg::Dsv4GraphOptions options{.fused = true, .device_draft_masks = device};
    auto arena = kg::TensorArena::Create(kg::DsparkGraphTensors(profile, 3));
    ASSERT_TRUE(arena);
    auto scalar = kg::BuildDsparkGraph(*arena, profile, *binding, 3, 256, options);
    ASSERT_TRUE(scalar) << Why(scalar);
    auto joined_arena = kg::TensorArena::Create(kg::DsparkWaveGraphTensors(profile, 3, 2));
    ASSERT_TRUE(joined_arena);
    auto joined = kg::BuildDsparkWaveGraph(*joined_arena, profile, *binding, 3, 2, 256, options);
    ASSERT_TRUE(joined) << Why(joined);
    const auto check = [&](const ggml_tensor* mask, const ggml_tensor* positions,
                           std::int64_t first, std::span<ggml_tensor* const> nodes,
                           std::span<ggml_tensor* const> inputs) {
      ASSERT_NE(mask, nullptr);
      EXPECT_EQ(mask->type, GGML_TYPE_F16);
      EXPECT_EQ(mask->ne[0], 256);
      EXPECT_EQ(mask->ne[1], 3);
      EXPECT_EQ(std::ranges::count(inputs, mask), device ? 0 : 1);
      EXPECT_EQ(std::ranges::count(nodes, mask), device ? 1 : 0);
      if (device) {
        EXPECT_TRUE(kg::Gemma4MaskFits(mask));
        EXPECT_EQ(mask->src[0], positions);
        EXPECT_EQ(kg::JitllmOpInt(mask, 0), first);
        EXPECT_EQ(kg::JitllmOpInt(mask, 1), 3);
        EXPECT_EQ(kg::JitllmOpInt(mask, 2), 256);
        EXPECT_EQ(kg::JitllmOpInt(mask, 3), profile.blocks.window);
        EXPECT_EQ(kg::JitllmOpInt(mask, 4), INT32_MAX);
        EXPECT_EQ(kg::JitllmOpInt(mask, 5), static_cast<int>(kg::CausalMaskRows::kExact));
        EXPECT_EQ(kg::JitllmOpInt(mask, 6), static_cast<int>(kg::MaskPolicy::kBlock));
        EXPECT_EQ(kg::JitllmOpInt(mask, 7), 0);
      } else
        EXPECT_EQ(mask->op, GGML_OP_NONE);
    };
    const auto scalar_inputs = scalar->inputs();
    check(scalar->core.raw_mask, scalar->core.positions, 0, scalar->core.nodes, scalar_inputs);
    const auto joined_inputs = joined->inputs();
    for (std::size_t i = 0; i < joined->slots.size(); ++i)
      check(joined->slots[i].raw_mask, joined->joined.positions, joined->first[i],
            joined->joined.nodes, joined_inputs);
    BindAll(*scalar, scalar->core.nodes);
    BindAll(*joined, joined->joined.nodes);
    auto scalar_plan = kg::PlanGraph(scalar->core.nodes, false, ModelDevice(false));
    auto joined_plan = kg::PlanGraph(joined->joined.nodes, false, ModelDevice(false));
    ASSERT_TRUE(scalar_plan) << Why(scalar_plan);
    ASSERT_TRUE(joined_plan) << Why(joined_plan);
    const auto count = [](const kg::GraphPlan& plan) {
      return std::ranges::count_if(plan.steps, [](const kg::PlanStep& step) {
        return step.implementation == kg::kGemma4MaskName;
      });
    };
    EXPECT_EQ(count(*scalar_plan), device ? 1 : 0);
    EXPECT_EQ(count(*joined_plan), device ? 2 : 0);
  }
}

TEST(DsparkTest, AJoinedDraftJoinsTheProductsAndKeepsEachSlotsBlock) {
  const md::Dsv4Profile target_profile = WindowOnlyTarget();
  const std::vector<md::Dsv4Resource> target_resources = Target(target_profile);
  auto target = md::BindDsv4(target_profile, "deepseek4", target_resources);
  ASSERT_TRUE(target.has_value());
  const md::DsparkProfile& d = md::DsparkDeepSeekV4Flash();
  const std::vector<md::Dsv4Resource> drafter = Drafter();
  auto dbinding = md::BindDspark(d, "dflash", drafter, target_profile, *target);
  ASSERT_TRUE(dbinding.has_value());
  const kg::Dsv4GraphOptions fused{.fused = true};
  auto one_arena = kg::TensorArena::Create(kg::DsparkGraphTensors(d, 3));
  ASSERT_TRUE(one_arena.has_value());
  auto one = kg::BuildDsparkGraph(*one_arena, d, *dbinding, 3, 256, fused);
  ASSERT_TRUE(one.has_value()) << Why(one);
  auto arena = kg::TensorArena::Create(kg::DsparkWaveGraphTensors(d, 3, 4));
  ASSERT_TRUE(arena.has_value());
  auto joined = kg::BuildDsparkWaveGraph(*arena, d, *dbinding, 3, 4, 256, fused);
  ASSERT_TRUE(joined.has_value()) << Why(joined);
  ASSERT_EQ(joined->slots.size(), 4U);
  EXPECT_EQ(joined->first, (std::vector<std::int64_t>{0, 3, 6, 9}));
  EXPECT_EQ(joined->joined.embd->ne[1], 12);
  EXPECT_EQ(joined->tokens->ne[0], 12);
  EXPECT_EQ(joined->joined.logits->ne[1], 12);
  EXPECT_EQ(joined->inputs().size(), 3U + (4U * 2U));
  ASSERT_EQ(joined->drafts.size(), 4U);
  for (const ggml_tensor* drafts : joined->drafts) {
    EXPECT_EQ(drafts->type, GGML_TYPE_I32);
    EXPECT_EQ(drafts->ne[0], 3);
  }
  for (const kg::Dsv4Graph& slot : joined->slots) {
    ASSERT_EQ(slot.layers.size(), d.blocks.layers);
    EXPECT_EQ(slot.layers[0].raw_k->ne[1], 256);  // its own ring
    EXPECT_EQ(slot.raw_mask->ne[1], 3);
  }
  const auto argmaxes = [](std::span<ggml_tensor* const> nodes) {
    return std::ranges::count_if(
        nodes, [](const ggml_tensor* n) { return kg::JitllmOpOf(n) == kg::JitllmOp::kArgmax; });
  };
  EXPECT_EQ(argmaxes(one->core.nodes), 3);
  EXPECT_EQ(argmaxes(joined->joined.nodes), 4 * 3);
  EXPECT_FALSE(joined->lanes.empty());
  BindAll(*one, one->core.nodes);
  BindAll(*joined, joined->joined.nodes);
  auto one_plan = kg::PlanGraph(one->core.nodes, false, ModelDevice(false));
  ASSERT_TRUE(one_plan.has_value()) << Why(one_plan);
  auto plan = kg::PlanGraph(joined->joined.nodes, false, ModelDevice(false));
  ASSERT_TRUE(plan.has_value()) << Why(plan);
  const auto count = [](const kg::GraphPlan& p, std::string_view name) {
    return std::ranges::count_if(p.steps,
                                 [&](const kg::PlanStep& s) { return s.implementation == name; });
  };
  EXPECT_GT(count(*one_plan, kg::kVecQName), 0);
  EXPECT_EQ(count(*plan, kg::kVecQName), count(*one_plan, kg::kVecQName));
  std::size_t attention = 0;
  for (const auto& step : plan->steps) {
    attention += step.operation == jitllm::execution::Operation::kFlashAttn ? 1 : 0;
  }
  EXPECT_EQ(attention, 4U * d.blocks.layers);
  // Refusals: one block, six of three rows (18, past a wave's 16), a block
  // past the drafter's, the reference form.
  auto again = kg::TensorArena::Create(kg::DsparkWaveGraphTensors(d, 6, 6));
  ASSERT_TRUE(again.has_value());
  EXPECT_FALSE(kg::BuildDsparkWaveGraph(*again, d, *dbinding, 3, 1, 256, fused).has_value());
  EXPECT_FALSE(kg::BuildDsparkWaveGraph(*again, d, *dbinding, 3, 6, 256, fused).has_value());
  EXPECT_FALSE(kg::BuildDsparkWaveGraph(*again, d, *dbinding, 6, 3, 256, fused).has_value());
  EXPECT_FALSE(kg::BuildDsparkWaveGraph(*again, d, *dbinding, 3, 4, 256, {}).has_value());
}

}  // namespace
