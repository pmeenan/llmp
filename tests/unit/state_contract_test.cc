// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// State contracts (model/state_contract.h): contracts equivalent to each
// current family's state reproduce its layout and ranges byte for byte
// (Gemma2's and Gemma3's used state and chunk writes, Qwen3.8's used state
// and checkpoint writes, DeepSeek V4's in both window modes); a verify's
// saves by row; the D-068 representations; the kept layout's identity; and
// the refusals.

#include "model/state_contract.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "model/dsv4.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "model/qwen38.h"
#include "model/state.h"

namespace {

namespace md = llmp::model;
using md::CellRule;
using md::Encoding;
using md::Rollback;
using md::StateContract;
using md::StateKind;
using md::StateRange;
using md::StateTensorContract;
using Layers = std::vector<std::pair<std::uint32_t, StateContract>>;
constexpr std::uint64_t kExtent = 2ULL << 20U;

void ExpectSame(const std::vector<StateRange>& got, const std::vector<StateRange>& want,
                const std::string& what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].offset, want[i].offset) << what << " range " << i;
    EXPECT_EQ(got[i].bytes, want[i].bytes) << what << " range " << i;
  }
}

// ------------------------------------------------------------------ Gemma

StateTensorContract GemmaKv(std::string name, std::uint32_t width, bool local,
                            std::uint32_t window) {
  return {.name = std::move(name),
          .kind = local ? StateKind::kRing : StateKind::kKv,
          .encoding = Encoding::kF16,
          .cell_bytes = std::uint64_t{width} * 2,
          .cells = local ? CellRule::kRing : CellRule::kPerPosition,
          .window_or_ratio = local ? window : 0,
          .rollback = local ? Rollback::kNone : Rollback::kTruncate};
}

template <typename Profile>
Layers GemmaContracts(const Profile& p) {
  Layers layers;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const bool local = p.local(il);
    layers.push_back({il,
                      {.tensors = {GemmaKv("k", p.key_dim * p.kv_heads, local, p.window),
                                   GemmaKv("v", p.value_dim * p.kv_heads, local, p.window)}}});
  }
  return layers;
}

template <typename Layout>
void ExpectGemmaLayout(const md::StateLayoutDerived& got, const Layout& want) {
  ASSERT_EQ(got.items.size(), want.tensors.size());
  for (std::size_t i = 0; i < got.items.size(); ++i) {
    EXPECT_EQ(got.items[i].layer, want.tensors[i].layer);
    EXPECT_EQ(got.items[i].offset, want.tensors[i].offset);
    EXPECT_EQ(got.items[i].bytes, want.tensors[i].bytes);
    EXPECT_EQ(got.items[i].cells, want.tensors[i].cells);
  }
  EXPECT_EQ(got.bytes, want.bytes);
}

TEST(StateContractTest, Gemma2ContractsReproduceItsLayoutUsedStateAndChunkWrites) {
  const auto& p = md::Gemma2_2B();
  for (const auto& [context, rows] : {std::pair{4096U, 128U}, {8192U, 512U}, {8192U, 4096U}}) {
    auto want = md::Gemma2State(p, context, rows);
    ASSERT_TRUE(want.has_value());
    const Layers contracts = GemmaContracts(p);
    auto got = md::Layout(contracts, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ExpectGemmaLayout(*got, *want);
    for (const std::uint32_t align : {256U, 512U, 2048U}) {
      for (const std::uint32_t positions : {0U, 1U, 255U, 256U, 257U, p.window - 1, p.window,
                                            p.window + rows, context - 1, context}) {
        if (positions > context) continue;
        auto used = md::UsedThrough(*got, positions, align);
        auto old = md::Gemma2UsedState(p, *want, positions, align);
        ASSERT_TRUE(used.has_value() && old.has_value());
        ExpectSame(*used, *old, std::format("Gemma2 {} positions align {}", positions, align));
      }
    }
    for (const auto& [past, n] : {std::pair{0U, 1U},
                                  {0U, rows},
                                  {p.window, rows},
                                  {got->items[0].cells - 3, 7U},
                                  {context - rows, rows},
                                  {got->items[0].cells * 2 - 1, 2U}}) {
      if (past > context || n > rows || n > context - past) continue;
      auto writes = md::ChunkWrites(*got, past, n);
      auto old = md::Gemma2ChunkWrites(p, *want, past, n);
      ASSERT_TRUE(writes.has_value() && old.has_value());
      ExpectSame(*writes, *old, std::format("Gemma2 writes {}+{}", past, n));
    }
  }
}

TEST(StateContractTest, Gemma3ContractsReproduceItsLayoutAndUsedState) {
  const auto& p = md::Gemma3_4BQat();
  for (const auto& [context, rows] : {std::pair{8448U, 128U}, {4096U, 512U}}) {
    auto want = md::Gemma3State(p, context, rows);
    ASSERT_TRUE(want.has_value());
    const Layers contracts = GemmaContracts(p);
    auto got = md::Layout(contracts, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ExpectGemmaLayout(*got, *want);
    for (const std::uint32_t align : {256U, 2048U}) {
      for (const std::uint32_t positions :
           {0U, 1U, 256U, 1025U, p.window, p.window + rows + 1, context}) {
        if (positions > context) continue;
        auto used = md::UsedThrough(*got, positions, align);
        auto old = md::Gemma3UsedState(p, *want, positions, align);
        ASSERT_TRUE(used.has_value() && old.has_value());
        ExpectSame(*used, *old, std::format("Gemma3 {} positions align {}", positions, align));
      }
    }
  }
}

// ---------------------------------------------------------------- Qwen3.8

Layers QwenContracts(const md::Qwen38Profile& p) {
  Layers layers;
  const auto fixed = [](std::string name, StateKind kind, std::uint64_t bytes) {
    return StateTensorContract{.name = std::move(name),
                               .kind = kind,
                               .encoding = Encoding::kF32,
                               .cell_bytes = bytes,
                               .cells = CellRule::kFixed,
                               .window_or_ratio = 1,
                               .rollback = Rollback::kSnapshot};
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    StateContract c;
    if (p.linear(il)) {
      c.tensors.push_back(fixed("conv", StateKind::kConvolution,
                                std::uint64_t{p.conv - 1} * p.conv_channels() * 4));
      c.tensors.push_back(
          fixed("recurrent", StateKind::kRecurrent,
                std::uint64_t{p.lin_head_dim} * p.lin_head_dim * p.lin_v_heads * 4));
      if (il == p.ple_layer) {
        c.tensors.push_back(fixed("ple_conv", StateKind::kConvolution,
                                  std::uint64_t{p.ple_history()} * p.hc_width() * 4));
      }
    } else {
      const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * 2;
      c.tensors.push_back({.name = "k", .kind = StateKind::kKv, .cell_bytes = kv});
      c.tensors.push_back({.name = "v", .kind = StateKind::kKv, .cell_bytes = kv});
      c.tensors.push_back({.name = "indexer_k",
                           .kind = StateKind::kIndex,
                           .encoding = Encoding::kF32,
                           .cell_bytes = std::uint64_t{p.indexer_head_dim} * 4});
      c.tensors.push_back({.name = "indexer_blocks",
                           .kind = StateKind::kBlockKeys,
                           .encoding = Encoding::kBF16,
                           .cell_bytes = std::uint64_t{p.indexer_head_dim} * 2,
                           .cells = CellRule::kPerBlock,
                           .window_or_ratio = p.indexer_ratio,
                           .basis = md::BlockBasis::kReadCells});
    }
    layers.push_back({il, std::move(c)});
  }
  return layers;
}

TEST(StateContractTest, Qwen38ContractsReproduceItsLayoutUsedStateAndCheckpointWrites) {
  const auto& p = md::Qwen38Flash();
  for (const auto& [context, rows] : {std::pair{8192U, 512U}, {262144U, 2048U}, {4000U, 300U}}) {
    auto want = md::Qwen38State(p, context, rows, false);
    ASSERT_TRUE(want.has_value()) << want.error();
    const Layers contracts = QwenContracts(p);
    auto got = md::Layout(contracts, context, rows);
    ASSERT_TRUE(got.has_value()) << got.error();
    ASSERT_EQ(got->items.size(), want->tensors.size());
    for (std::size_t i = 0; i < got->items.size(); ++i) {
      EXPECT_EQ(got->items[i].offset, want->tensors[i].offset) << i;
      EXPECT_EQ(got->items[i].bytes, want->tensors[i].bytes) << i;
      EXPECT_EQ(got->items[i].cells, want->tensors[i].ne1) << i;
    }
    EXPECT_EQ(got->bytes, want->bytes);
    for (const std::uint32_t align : {256U, 2048U}) {
      for (const std::uint32_t positions :
           {0U, 1U, 63U, 64U, 65U, 256U, 2049U, context - 1, context}) {
        auto used = md::UsedThrough(*got, positions, align);
        auto old = md::Qwen38UsedState(p, *want, positions, align);
        ASSERT_TRUE(used.has_value() && old.has_value());
        ExpectSame(*used, *old, std::format("Qwen3.8 {} positions align {}", positions, align));
      }
    }
    for (const std::uint32_t positions : {0U, 1U, 64U, 100U, context}) {
      auto writes = md::CheckpointWrites(*got, positions);
      auto old = md::Qwen38CheckpointWrites(p, *want, positions);
      ASSERT_TRUE(writes.has_value() && old.has_value());
      ExpectSame(*writes, *old, std::format("Qwen3.8 checkpoint at {}", positions));
    }
  }
}

// The MTP drafter's own region: a stage of its own.
TEST(StateContractTest, Qwen38MtpContractReproducesItsDrafterRegion) {
  const auto& p = md::Qwen38Flash();
  auto target = md::Qwen38State(p, 8192, 512, false);
  ASSERT_TRUE(target.has_value());
  auto want = md::Qwen38MtpStateOf(p, *target);
  ASSERT_TRUE(want.has_value());
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * 2;
  const Layers mtp = {{0,
                       {.tensors = {{.name = "k", .kind = StateKind::kKv, .cell_bytes = kv},
                                    {.name = "v", .kind = StateKind::kKv, .cell_bytes = kv},
                                    {.name = "indexer_k",
                                     .kind = StateKind::kIndex,
                                     .encoding = Encoding::kF32,
                                     .cell_bytes = std::uint64_t{p.indexer_head_dim} * 4},
                                    {.name = "indexer_blocks",
                                     .kind = StateKind::kBlockKeys,
                                     .encoding = Encoding::kBF16,
                                     .cell_bytes = std::uint64_t{p.indexer_head_dim} * 2,
                                     .cells = CellRule::kPerBlock,
                                     .window_or_ratio = p.indexer_ratio,
                                     .basis = md::BlockBasis::kReadCells},
                                    {.name = "streams",
                                     .kind = StateKind::kFeatures,
                                     .encoding = Encoding::kF32,
                                     .cell_bytes = std::uint64_t{p.hc_width()} * 4,
                                     .cells = CellRule::kPerRow,
                                     .window_or_ratio = 1,
                                     .rollback = Rollback::kSnapshot}}}}};
  auto got = md::Layout(mtp, 8192, 512);
  ASSERT_TRUE(got.has_value()) << got.error();
  ASSERT_EQ(got->items.size(), 5U);
  EXPECT_EQ(got->items[0].offset, want->k);
  EXPECT_EQ(got->items[1].offset, want->v);
  EXPECT_EQ(got->items[2].offset, want->indexer);
  EXPECT_EQ(got->items[3].offset, want->blocks);
  EXPECT_EQ(got->items[4].offset, want->hidden);
  EXPECT_EQ(got->items[4].cells, want->hidden_rows);
  EXPECT_EQ(got->bytes, want->bytes);
  // The runner's drafter ranges (Qwen38Runner::StateRanges) for its caches.
  for (const std::uint32_t positions : {1U, 300U, 8192U}) {
    for (const std::uint32_t align : {256U, 2048U}) {
      auto used = md::UsedThrough(*got, positions, align);
      ASSERT_TRUE(used.has_value());
      const std::uint64_t cells =
          std::min<std::uint64_t>(want->cells, (positions + align - 1) / align * align);
      const std::uint64_t blocks = (cells + p.indexer_ratio - 1) / p.indexer_ratio;
      EXPECT_EQ((*used)[0].bytes, cells * kv);
      EXPECT_EQ((*used)[1].bytes, cells * kv);
      EXPECT_EQ((*used)[2].bytes, cells * p.indexer_head_dim * 4);
      EXPECT_EQ((*used)[3].bytes, blocks * p.indexer_head_dim * 2);
      // The streams whole: the runner names their padded bytes, the same
      // 2 MiB extents.
      EXPECT_EQ((*used)[4].offset, want->hidden);
      EXPECT_EQ(((*used)[4].offset + (*used)[4].bytes + kExtent - 1) / kExtent,
                (want->bytes + kExtent - 1) / kExtent);
    }
  }
}

// --------------------------------------------------------------- DeepSeek

Layers Dsv4Contracts(const md::Dsv4Profile& p, md::Dsv4Window window) {
  Layers layers;
  const auto compressed = [](std::string name, StateKind kind, std::uint64_t cell,
                             std::uint32_t ratio) {
    return StateTensorContract{.name = std::move(name),
                               .kind = kind,
                               .cell_bytes = cell,
                               .cells = CellRule::kPerBlock,
                               .window_or_ratio = ratio,
                               .basis = md::BlockBasis::kPositions,
                               .dummy_last = true,
                               .rollback = Rollback::kSnapshot};
  };
  const auto ring = [](std::string name, std::uint64_t cell, std::uint32_t rows) {
    return StateTensorContract{.name = std::move(name),
                               .kind = StateKind::kCompressor,
                               .encoding = Encoding::kF32,
                               .cell_bytes = cell,
                               .cells = CellRule::kFixed,
                               .window_or_ratio = rows,
                               .rollback = Rollback::kSnapshot};
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    StateContract c;
    const std::uint64_t raw = std::uint64_t{p.head_dim} * 2;
    if (window == md::Dsv4Window::kFull) {
      c.tensors.push_back({.name = "raw",
                           .kind = StateKind::kKv,
                           .cell_bytes = raw,
                           .rollback = Rollback::kSnapshot});
    } else {
      c.tensors.push_back({.name = "raw",
                           .kind = StateKind::kRing,
                           .cell_bytes = raw,
                           .cells = CellRule::kRing,
                           .window_or_ratio = p.window,
                           .ring_reads_whole = true,
                           .rollback = Rollback::kSnapshot});
    }
    const std::uint64_t hd2 = 2ULL * p.head_dim * 4;
    const std::uint64_t ihd2 = 2ULL * p.indexer_head_dim * 4;
    if (p.compress_ratios[il] == md::kDsv4CsaRatio) {
      c.tensors.push_back(compressed("csa", StateKind::kCompressed, std::uint64_t{p.head_dim} * 2,
                                     md::kDsv4CsaRatio));
      c.tensors.push_back(ring("csa_kv", hd2, 2 * md::kDsv4CsaRatio));
      c.tensors.push_back(ring("csa_score", hd2, 2 * md::kDsv4CsaRatio));
      c.tensors.push_back(compressed("lid", StateKind::kIndex,
                                     std::uint64_t{p.indexer_head_dim} * 2, md::kDsv4CsaRatio));
      c.tensors.push_back(ring("lid_kv", ihd2, 2 * md::kDsv4CsaRatio));
      c.tensors.push_back(ring("lid_score", ihd2, 2 * md::kDsv4CsaRatio));
    } else if (p.compress_ratios[il] == md::kDsv4HcaRatio) {
      c.tensors.push_back(compressed("hca", StateKind::kCompressed, std::uint64_t{p.head_dim} * 2,
                                     md::kDsv4HcaRatio));
      c.tensors.push_back(ring("hca_kv", std::uint64_t{p.head_dim} * 4, md::kDsv4HcaRatio));
      c.tensors.push_back(ring("hca_score", std::uint64_t{p.head_dim} * 4, md::kDsv4HcaRatio));
    }
    layers.push_back({il, std::move(c)});
  }
  return layers;
}

TEST(StateContractTest, DeepSeekContractsReproduceBothWindowsUsedStateAndCheckpointWrites) {
  const auto& p = md::Dsv4Flash();
  for (const auto window : {md::Dsv4Window::kFull, md::Dsv4Window::kRing}) {
    for (const auto& [context, rows] : {std::pair{4096U, 512U}, {65536U, 2048U}, {1000U, 128U}}) {
      auto want = md::Dsv4State(p, context, rows, window);
      ASSERT_TRUE(want.has_value()) << want.error();
      const Layers contracts = Dsv4Contracts(p, window);
      auto got = md::Layout(contracts, context, rows);
      ASSERT_TRUE(got.has_value()) << got.error();
      ASSERT_EQ(got->items.size(), want->tensors.size());
      for (std::size_t i = 0; i < got->items.size(); ++i) {
        EXPECT_EQ(got->items[i].offset, want->tensors[i].offset) << i;
        EXPECT_EQ(got->items[i].bytes, want->tensors[i].bytes) << i;
      }
      EXPECT_EQ(got->bytes, want->bytes);
      for (const std::uint32_t positions :
           {0U, 1U, 3U, 4U, 5U, 127U, 128U, 129U, 255U, 256U, 257U, 1023U, context - 1, context}) {
        if (positions > context) continue;
        auto used = md::UsedThrough(*got, positions);
        auto old = md::Dsv4UsedState(*want, positions);
        ASSERT_TRUE(used.has_value() && old.has_value());
        ExpectSame(*used, *old, std::format("DeepSeek used at {}", positions));
        auto writes = md::CheckpointWrites(*got, positions);
        auto old_writes = md::Dsv4CheckpointWrites(*want, positions);
        ASSERT_TRUE(writes.has_value() && old_writes.has_value());
        ExpectSame(*writes, *old_writes, std::format("DeepSeek checkpoint at {}", positions));
      }
    }
  }
}

// ------------------------------------------------------------ the general

md::StateLayoutDerived Mixed() {
  const Layers layers = {
      {0,
       {.tensors = {{.name = "k", .kind = StateKind::kKv, .cell_bytes = 512},
                    {.name = "ring",
                     .kind = StateKind::kRing,
                     .cell_bytes = 256,
                     .cells = CellRule::kRing,
                     .window_or_ratio = 64,
                     .cell_pad = 1,
                     .rollback = Rollback::kSnapshot},
                    {.name = "blocks",
                     .kind = StateKind::kCompressed,
                     .cell_bytes = 256,
                     .cells = CellRule::kPerBlock,
                     .window_or_ratio = 4,
                     .cell_pad = 1,
                     .dummy_last = true,
                     .rollback = Rollback::kSnapshot},
                    {.name = "recurrent",
                     .kind = StateKind::kRecurrent,
                     .encoding = Encoding::kF32,
                     .cell_bytes = 1024,
                     .cells = CellRule::kFixed,
                     .window_or_ratio = 1,
                     .rollback = Rollback::kSnapshot},
                    {.name = "target_k", .cell_bytes = 512, .borrowed_from_target = true}}}}};
  return md::Layout(layers, 1024, 16).value();
}

TEST(StateContractTest, AVerifySavesEachRowsCellsAndBlocksAndRestoresScratchWhatever) {
  const auto layout = Mixed();
  ASSERT_EQ(layout.items.size(), 4U);  // the borrowed tensor has no bytes here
  ASSERT_EQ(layout.borrowed.size(), 1U);
  const auto& ring = layout.items[1];
  const auto& blocks = layout.items[2];
  // Rows 78..81: the ring (80 cells) wraps after row 79; blocks 19 and 20
  // (row 79 completes 19, row 81 is the chunk's last in 20).
  auto writes = md::WritesFor(layout, 78, 4);
  ASSERT_TRUE(writes.has_value()) << writes.error();
  ASSERT_EQ(writes->rows.size(), 4U);
  ASSERT_EQ(ring.cells, 80U);
  // The per-position cache truncates: no saves of its own.
  EXPECT_EQ(writes->rows[0].size(), 1U);
  EXPECT_EQ(writes->rows[0][0].offset, ring.offset + (78 * 256));
  EXPECT_EQ(writes->rows[2][0].offset, ring.offset);  // position 80, cell 0
  ASSERT_EQ(writes->rows[1].size(), 2U);
  EXPECT_EQ(writes->rows[1][1].offset, blocks.offset + (19 * 256));
  ASSERT_EQ(writes->rows[3].size(), 2U);
  EXPECT_EQ(writes->rows[3][1].offset, blocks.offset + (20 * 256));
  // The dummy row and the recurrent state are restored whatever is kept.
  ASSERT_EQ(writes->scratch.size(), 2U);
  EXPECT_EQ(writes->scratch[0].offset, blocks.offset + ((blocks.cells - 1) * 256));
  EXPECT_EQ(writes->scratch[1].offset, layout.items[3].offset);
  EXPECT_EQ(writes->scratch[1].bytes, 1024U);
  // Every saved byte is a chunk write.
  auto chunk = md::ChunkWrites(layout, 78, 4);
  ASSERT_TRUE(chunk.has_value());
  const auto inside = [&](const StateRange& r) {
    for (const auto& w : *chunk) {
      if (r.offset >= w.offset && r.offset + r.bytes <= w.offset + w.bytes) return true;
    }
    return false;
  };
  for (const auto& row : writes->rows) {
    for (const auto& r : row) EXPECT_TRUE(inside(r)) << r.offset;
  }
  for (const auto& r : writes->scratch) EXPECT_TRUE(inside(r)) << r.offset;
  // A state that cannot roll back refuses a verify.
  const Layers ring_only = {{0,
                             {.tensors = {{.name = "ring",
                                           .kind = StateKind::kRing,
                                           .cell_bytes = 256,
                                           .cells = CellRule::kRing,
                                           .window_or_ratio = 64,
                                           .rollback = Rollback::kNone}}}}};
  auto refusing = md::Layout(ring_only, 1024, 16);
  ASSERT_TRUE(refusing.has_value());
  EXPECT_FALSE(md::WritesFor(*refusing, 0, 1).has_value());
  EXPECT_FALSE(md::WritesFor(layout, 1020, 5).has_value());  // past the context
  EXPECT_FALSE(md::WritesFor(layout, 0, 17).has_value());    // past the chunk bound
}

TEST(StateContractTest, RepresentationsGroupKindsAndTruncateOnlyWhatCanRollBack) {
  const auto layout = Mixed();
  const auto plain = md::Representations(layout, "mixed", false);
  ASSERT_EQ(plain.size(), 4U);
  EXPECT_EQ(plain[0].name, "mixed.kv");
  EXPECT_EQ(plain[3].name, "mixed.recurrent");
  std::uint64_t total = 0;
  for (const auto& r : plain) {
    EXPECT_TRUE(md::IsValid(r)) << r.name;
    total += r.block_bytes.value();
  }
  EXPECT_EQ(total, layout.bytes);
  EXPECT_TRUE(plain[0].CanTruncate());   // a cell per position
  EXPECT_FALSE(plain[1].CanTruncate());  // a snapshot needs a verify's
  const auto speculative = md::Representations(layout, "mixed", true);
  for (const auto& r : speculative) EXPECT_TRUE(r.CanTruncate()) << r.name;
}

TEST(StateContractTest, KeptLayoutIdsAreNewAndChangeWithTheSpecAndAnyByteMeaning) {
  const auto layout = Mixed();
  llmp::base::Sha256Digest spec{};
  spec[0] = 1;
  const std::vector<md::StateLayoutDerived> one = {layout};
  const auto id = md::KeptLayoutId(spec, one);
  EXPECT_TRUE(id.starts_with("llmp-state-v2:")) << id;
  EXPECT_EQ(id.size(), std::string("llmp-state-v2:").size() + 64);
  EXPECT_EQ(md::KeptLayoutId(spec, one), id);
  std::set<std::string> ids = {id};
  auto other_spec = spec;
  other_spec[31] = 7;
  ids.insert(md::KeptLayoutId(other_spec, one));
  auto encoding = layout;
  encoding.items[0].contract.encoding = Encoding::kBF16;
  ids.insert(md::KeptLayoutId(spec, std::vector{encoding}));
  auto context = md::Layout(std::vector<std::pair<std::uint32_t, StateContract>>{}, 2048, 16);
  ASSERT_TRUE(context.has_value());
  ids.insert(md::KeptLayoutId(spec, std::vector{*context}));
  ids.insert(md::KeptLayoutId(spec, std::vector{layout, layout}));  // a drafter stage
  EXPECT_EQ(ids.size(), 5U);
  // Never one of the runners' own IDs.
  EXPECT_FALSE(id.starts_with("gemma"));
  EXPECT_FALSE(id.starts_with("qwen38-state/1"));
  EXPECT_FALSE(id.starts_with("deepseek4-state/1"));
}

TEST(StateContractTest, LayoutsRefuseInconsistentBoundsAndEmptyTensors) {
  const Layers kv = {{0, {.tensors = {{.name = "k", .cell_bytes = 512}}}}};
  EXPECT_FALSE(md::Layout(kv, 0, 1).has_value());
  EXPECT_FALSE(md::Layout(kv, 16, 0).has_value());
  EXPECT_FALSE(md::Layout(kv, 16, 17).has_value());
  EXPECT_FALSE(md::Layout(kv, 16, 16, 100).has_value());  // not a power of two
  EXPECT_FALSE(md::Layout(kv, 16, 16, 128).has_value());  // under 256 bytes
  EXPECT_TRUE(md::Layout(kv, 16, 16).has_value());
  const Layers no_bytes = {{0, {.tensors = {{.name = "k", .cell_bytes = 0}}}}};
  EXPECT_FALSE(md::Layout(no_bytes, 16, 16).has_value());
  const Layers no_window = {
      {0, {.tensors = {{.name = "r", .cell_bytes = 2, .cells = CellRule::kRing}}}}};
  EXPECT_FALSE(md::Layout(no_window, 16, 16).has_value());
  const Layers no_ratio = {
      {0, {.tensors = {{.name = "b", .cell_bytes = 2, .cells = CellRule::kPerBlock}}}}};
  EXPECT_FALSE(md::Layout(no_ratio, 16, 16).has_value());
  const Layers whole_kv = {
      {0, {.tensors = {{.name = "k", .cell_bytes = 2, .ring_reads_whole = true}}}}};
  EXPECT_FALSE(md::Layout(whole_kv, 16, 16).has_value());
  const Layers partial = {
      {0, {.tensors = {{.name = "k", .encoding = Encoding::kF32, .cell_bytes = 6}}}}};
  EXPECT_FALSE(md::Layout(partial, 16, 16).has_value());
  const Layers huge = {{0, {.tensors = {{.name = "k", .cell_bytes = UINT64_MAX / 4 * 2}}}}};
  EXPECT_FALSE(md::Layout(huge, 4096, 16).has_value());
  auto layout = md::Layout(kv, 16, 16);
  EXPECT_FALSE(md::UsedThrough(*layout, 17).has_value());
  EXPECT_FALSE(md::UsedThrough(*layout, 1, 100).has_value());
  EXPECT_FALSE(md::CheckpointWrites(*layout, 17).has_value());
  EXPECT_FALSE(md::ChunkWrites(*layout, 16, 1).has_value());
}

}  // namespace
