// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// State contracts (model/state_contract.h): contracts equivalent to each
// current family's state reproduce its layout and ranges byte for byte,
// against the M3 runners' layouts in closed form (their runners and
// per-family layouts are retired; the contracts are the one source now):
// Gemma2's and Gemma3's used state and chunk writes; Qwen3.8's used state
// and checkpoint writes; DeepSeek V4's in both window modes; a verify's
// saves by row, byte for byte what Qwen3.8's, DeepSeek V4's and DSpark's
// verifies saved; the D-068 representations; the kept layout's identity;
// and the refusals. The families' facts are tests/support/families/'s.

#include "model/state_contract.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "families/dsv4.h"
#include "families/gemma.h"
#include "families/qwen.h"
#include "model/state.h"

namespace {

namespace md = llmp::model;
namespace fam = llmp::test_support::families;
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

// The same bytes, in any order (a verify's saves: the order only places
// them in the snapshot).
void ExpectSameSet(std::vector<StateRange> got, std::vector<StateRange> want,
                   const std::string& what) {
  const auto order = [](const StateRange& a, const StateRange& b) {
    return a.offset != b.offset ? a.offset < b.offset : a.bytes < b.bytes;
  };
  std::ranges::sort(got, order);
  std::ranges::sort(want, order);
  ExpectSame(got, want, what);
}

// ------------------------------------------------------------------ Gemma

StateTensorContract GemmaKv(std::string name, std::uint32_t width, bool local,
                            std::uint32_t window) {
  return {.name = std::move(name),
          .kind = local ? StateKind::kRing : StateKind::kKv,
          .encoding = Encoding::kF16,
          .cell_bytes = std::uint64_t{width} * 2,
          .cells = local ? CellRule::kRing : CellRule::kPerPosition,
          .window = local ? window : 0,
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

// Gemma's state in closed form, as the M3 layouts were (retired with their
// runners; the contracts above are its one source now): a K and a V tensor
// per layer, global ones a cell a position, local ones a ring of
// pad(min(context, window + rows), 256) cells, each tensor on an extent.
struct GemmaTensorRef {
  std::uint32_t layer = 0;
  bool local = false;
  std::uint32_t width = 0, cells = 0;  // F16 [width, cells]
  std::uint64_t offset = 0, bytes = 0;
};
struct GemmaLayoutRef {
  std::vector<GemmaTensorRef> tensors;
  std::uint64_t bytes = 0;
};
std::uint64_t PadTo(std::uint64_t value, std::uint64_t align) {
  return (value + align - 1) / align * align;
}

template <typename Profile>
GemmaLayoutRef GemmaRef(const Profile& p, std::uint32_t context, std::uint32_t rows) {
  GemmaLayoutRef out;
  const auto global_cells = static_cast<std::uint32_t>(PadTo(context, 256));
  const auto local_cells =
      static_cast<std::uint32_t>(PadTo(std::min<std::uint64_t>(context, p.window + rows), 256));
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto width = (value ? p.value_dim : p.key_dim) * p.kv_heads;
      const auto cells = p.local(il) ? local_cells : global_cells;
      const auto bytes = std::uint64_t{width} * cells * 2;
      out.tensors.push_back({.layer = il,
                             .local = p.local(il),
                             .width = width,
                             .cells = cells,
                             .offset = out.bytes,
                             .bytes = bytes});
      out.bytes += PadTo(bytes, kExtent);
    }
  }
  return out;
}

// What `positions` positions read: each tensor's cells, up to the next
// multiple of `align`, capped at the tensor's own.
std::vector<StateRange> GemmaUsedRef(const GemmaLayoutRef& layout, std::uint32_t positions,
                                     std::uint32_t align) {
  std::vector<StateRange> out;
  if (positions == 0) return out;
  for (const auto& t : layout.tensors) {
    const auto cells = std::min<std::uint64_t>(t.cells, PadTo(positions, align));
    out.push_back({.offset = t.offset, .bytes = cells * t.width * 2});
  }
  return out;
}

// The cells a chunk of `rows` rows after `past` positions writes: a
// global tensor's next cells, a ring's from `past` mod its cells, split at
// the wrap.
std::vector<StateRange> GemmaWritesRef(const GemmaLayoutRef& layout, std::uint32_t past,
                                       std::uint32_t rows) {
  std::vector<StateRange> out;
  for (const auto& t : layout.tensors) {
    const std::uint64_t first = t.local ? past % t.cells : past;
    const std::uint64_t count = std::min<std::uint64_t>(rows, t.cells - first);
    const std::uint64_t row_bytes = std::uint64_t{t.width} * 2;
    out.push_back({.offset = t.offset + first * row_bytes, .bytes = count * row_bytes});
    if (count < rows) out.push_back({.offset = t.offset, .bytes = (rows - count) * row_bytes});
  }
  return out;
}

void ExpectGemmaLayout(const md::StateLayoutDerived& got, const GemmaLayoutRef& want) {
  ASSERT_EQ(got.items.size(), want.tensors.size());
  for (std::size_t i = 0; i < got.items.size(); ++i) {
    EXPECT_EQ(got.items[i].layer, want.tensors[i].layer);
    EXPECT_EQ(got.items[i].offset, want.tensors[i].offset);
    EXPECT_EQ(got.items[i].bytes, want.tensors[i].bytes);
    EXPECT_EQ(got.items[i].cells, want.tensors[i].cells);
  }
  EXPECT_EQ(got.bytes, want.bytes);
}

TEST(StateContractTest, Gemma2ContractsDeriveItsLayoutUsedStateAndChunkWrites) {
  const auto& p = fam::Gemma2_2B();
  for (const auto& [context, rows] : {std::pair{4096U, 128U}, {8192U, 512U}, {8192U, 4096U}}) {
    const auto want = GemmaRef(p, context, rows);
    const Layers contracts = GemmaContracts(p);
    auto got = md::Layout(contracts, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ExpectGemmaLayout(*got, want);
    for (const std::uint32_t align : {256U, 512U, 2048U}) {
      for (const std::uint32_t positions : {0U, 1U, 255U, 256U, 257U, p.window - 1, p.window,
                                            p.window + rows, context - 1, context}) {
        if (positions > context) continue;
        auto used = md::UsedThrough(*got, positions, align);
        ASSERT_TRUE(used.has_value());
        ExpectSame(*used, GemmaUsedRef(want, positions, align),
                   std::format("Gemma2 {} positions align {}", positions, align));
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
      ASSERT_TRUE(writes.has_value());
      ExpectSame(*writes, GemmaWritesRef(want, past, n),
                 std::format("Gemma2 writes {}+{}", past, n));
    }
  }
}

// ExtentLayout's per-tensor rule: Gemma's tensors (each an extent or more)
// on their own extents as M3 laid them out; small tensors packed at 256
// bytes, sharing extents; a bank's continuation right after its
// predecessor whatever its size; a continuation with nothing before it in
// its layer refused.
TEST(StateContractTest, ExtentLayoutPlacesLargeTensorsOnExtentsAndPacksTheRest) {
  const auto& p = fam::Gemma2_2B();
  const Layers gemma = GemmaContracts(p);
  for (const auto& [context, rows] : {std::pair{4096U, 128U}, {8192U, 512U}}) {
    auto rule = md::ExtentLayout(gemma, context, rows, kExtent);
    auto spread = md::Layout(gemma, context, rows, kExtent);
    ASSERT_TRUE(rule.has_value() && spread.has_value());
    EXPECT_EQ(rule->extent, kExtent);
    ASSERT_EQ(rule->items.size(), spread->items.size());
    for (std::size_t i = 0; i < rule->items.size(); ++i)
      EXPECT_EQ(rule->items[i].offset, spread->items[i].offset) << i;
    EXPECT_NE(md::KeptLayoutId({}, std::array{*rule}), md::KeptLayoutId({}, std::array{*spread}));
  }
  const StateTensorContract small{.name = "conv",
                                  .kind = StateKind::kConvolution,
                                  .cell_bytes = 4096,
                                  .cells = CellRule::kFixed,
                                  .fixed_cells = 3};
  const StateTensorContract large{.name = "kv", .kind = StateKind::kKv, .cell_bytes = 1024};
  StateTensorContract raw{.name = "kv",
                          .kind = StateKind::kRing,
                          .cell_bytes = 1024,
                          .cells = CellRule::kRing,
                          .window = 128,
                          .ring_reads_whole = true};
  StateTensorContract compressed{.name = "compressed",
                                 .kind = StateKind::kCompressed,
                                 .cell_bytes = 1024,
                                 .cells = CellRule::kPerBlock,
                                 .block_ratio = 4,
                                 .bank_with_previous = true};
  // Used whole from the first position: packed whatever its size.
  const StateTensorContract whole{.name = "recurrent",
                                  .kind = StateKind::kRecurrent,
                                  .cell_bytes = 3 << 20,
                                  .cells = CellRule::kFixed,
                                  .fixed_cells = 1};
  const Layers layers = {{0, StateContract{.tensors = {small, large}}},
                         {1, StateContract{.tensors = {small, small}}},
                         {2, StateContract{.tensors = {raw, compressed}}},
                         {3, StateContract{.tensors = {whole}}}};
  auto rule = md::ExtentLayout(layers, 65536, 128, kExtent);
  ASSERT_TRUE(rule.has_value()) << rule.error();
  ASSERT_EQ(rule->items.size(), 7U);
  EXPECT_EQ(rule->items[6].offset, rule->items[5].offset + rule->items[5].bytes);
  const auto& it = rule->items;
  EXPECT_EQ(it[0].offset, 0U);                     // 12 KiB, packed
  EXPECT_EQ(it[1].offset, kExtent);                // 64 MiB, on the next extent
  EXPECT_EQ(it[2].offset, kExtent + it[1].bytes);  // packed after it
  EXPECT_EQ(it[3].offset, it[2].offset + it[2].bytes);
  EXPECT_EQ(it[4].offset, it[3].offset + it[3].bytes);  // 256 KiB raw window, packed
  EXPECT_GE(it[5].bytes, kExtent);
  EXPECT_EQ(it[5].offset, it[4].offset + it[4].bytes);  // the bank, contiguous
  EXPECT_TRUE(
      md::ContiguousStateSpan(
          *rule, std::array{md::StateSlice{.layer = 2, .tensor = 0, .first = 0, .cells = 256},
                            md::StateSlice{.layer = 2, .tensor = 1, .first = 0, .cells = 1}})
          .has_value());
  // Small tensors only: the same offsets and bytes as Layout's packed
  // layout, but another identity (a kept record names its rule).
  const Layers packed_only = {{0, StateContract{.tensors = {small, small}}},
                              {1, StateContract{.tensors = {raw, small}}}};
  auto small_rule = md::ExtentLayout(packed_only, 4096, 128, kExtent);
  auto small_packed = md::Layout(packed_only, 4096, 128, 256);
  ASSERT_TRUE(small_rule.has_value() && small_packed.has_value());
  ASSERT_EQ(small_rule->items.size(), small_packed->items.size());
  for (std::size_t i = 0; i < small_rule->items.size(); ++i)
    EXPECT_EQ(small_rule->items[i].offset, small_packed->items[i].offset) << i;
  EXPECT_EQ(small_rule->bytes, small_packed->bytes);
  EXPECT_NE(md::KeptLayoutId({}, std::array{*small_rule}),
            md::KeptLayoutId({}, std::array{*small_packed}));
  // A continuation starts exactly at its predecessor's end: one that would
  // need a gap (an end off 256 bytes) is refused, never padded apart.
  StateTensorContract odd = raw;
  odd.cell_bytes = 384;
  StateTensorContract odd_bank = compressed;
  odd_bank.cell_bytes = 384;
  const Layers whole_end = {{0, StateContract{.tensors = {odd, odd_bank}}}};
  EXPECT_EQ(md::ExtentLayout(whole_end, 4096, 128, kExtent).value().items[1].offset % 256, 0U)
      << "256 cells of 384 bytes end on 256";
  StateTensorContract three = odd;
  three.window = 0;
  three.fixed_cells = 1;
  three.cells = CellRule::kFixed;
  three.ring_reads_whole = false;
  const Layers gap = {{0, StateContract{.tensors = {three, odd_bank}}}};
  EXPECT_FALSE(md::ExtentLayout(gap, 4096, 128, kExtent).has_value());
  EXPECT_TRUE(md::Layout(gap, 4096, 128).has_value());
  // Uniform layouts are unchanged by the flag; a continuation needs a predecessor.
  EXPECT_TRUE(md::Layout(layers, 65536, 128).has_value());
  const Layers orphan = {{0, StateContract{.tensors = {compressed}}}};
  EXPECT_FALSE(md::ExtentLayout(orphan, 65536, 128, kExtent).has_value());
  EXPECT_FALSE(md::ExtentLayout(layers, 65536, 128, 3 * 256).has_value());
}

TEST(StateContractTest, Gemma3ContractsDeriveItsLayoutAndUsedState) {
  const auto& p = fam::Gemma3_4BQat();
  for (const auto& [context, rows] : {std::pair{8448U, 128U}, {4096U, 512U}}) {
    const auto want = GemmaRef(p, context, rows);
    const Layers contracts = GemmaContracts(p);
    auto got = md::Layout(contracts, context, rows, kExtent);
    ASSERT_TRUE(got.has_value()) << got.error();
    ExpectGemmaLayout(*got, want);
    for (const std::uint32_t align : {256U, 2048U}) {
      for (const std::uint32_t positions :
           {0U, 1U, 256U, 1025U, p.window, p.window + rows + 1, context}) {
        if (positions > context) continue;
        auto used = md::UsedThrough(*got, positions, align);
        ASSERT_TRUE(used.has_value());
        ExpectSame(*used, GemmaUsedRef(want, positions, align),
                   std::format("Gemma3 {} positions align {}", positions, align));
      }
    }
  }
}

// ---------------------------------------------------------------- Qwen3.8

Layers QwenContracts(const fam::Qwen38Profile& p) {
  Layers layers;
  const auto fixed = [](std::string name, StateKind kind, std::uint64_t bytes) {
    return StateTensorContract{.name = std::move(name),
                               .kind = kind,
                               .encoding = Encoding::kF32,
                               .cell_bytes = bytes,
                               .cells = CellRule::kFixed,
                               .fixed_cells = 1,
                               .rollback = Rollback::kCommit};
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
      // A verify saves each row's K, V and indexer cells and the block keys
      // it completes (Qwen38VerifySaves); its commit rebuilds the linear
      // layers' state, which a verify only reads.
      const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * 2;
      c.tensors.push_back(
          {.name = "k", .kind = StateKind::kKv, .cell_bytes = kv, .rollback = Rollback::kSnapshot});
      c.tensors.push_back(
          {.name = "v", .kind = StateKind::kKv, .cell_bytes = kv, .rollback = Rollback::kSnapshot});
      c.tensors.push_back({.name = "indexer_k",
                           .kind = StateKind::kIndex,
                           .encoding = Encoding::kF32,
                           .cell_bytes = std::uint64_t{p.indexer_head_dim} * 4,
                           .rollback = Rollback::kSnapshot});
      c.tensors.push_back({.name = "indexer_blocks",
                           .kind = StateKind::kBlockKeys,
                           .encoding = Encoding::kBF16,
                           .cell_bytes = std::uint64_t{p.indexer_head_dim} * 2,
                           .cells = CellRule::kPerBlock,
                           .block_ratio = p.indexer_ratio,
                           .basis = md::BlockBasis::kReadCells,
                           .rollback = Rollback::kSnapshot});
    }
    layers.push_back({il, std::move(c)});
  }
  return layers;
}

// The MTP drafter's stage: its own caches (rewritten by its catch-up, so
// truncated) and the streams the target's chunks export into it.
Layers QwenMtpContracts(const fam::Qwen38Profile& p) {
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * 2;
  return {{0,
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
                         .block_ratio = p.indexer_ratio,
                         .basis = md::BlockBasis::kReadCells},
                        {.name = "streams",
                         .kind = StateKind::kFeatures,
                         .encoding = Encoding::kF32,
                         .cell_bytes = std::uint64_t{p.hc_width()} * 4,
                         .cells = CellRule::kPerRow,
                         .row_carry = 1,
                         .rollback = Rollback::kSnapshot,
                         .target_writes = true}}}}};
}

struct QwenTensorRef {
  std::uint32_t layer = 0;
  bool f16 = false;
  std::uint64_t ne0 = 0, ne1 = 0;
  std::uint64_t offset = 0, bytes = 0;
};

struct QwenLayoutRef {
  std::vector<QwenTensorRef> tensors;
  std::uint64_t bytes = 0;
  std::uint32_t cells = 0;
};

QwenLayoutRef QwenRef(const fam::Qwen38Profile& p, std::uint32_t context, std::uint32_t /*rows*/) {
  QwenLayoutRef s;
  s.cells = static_cast<std::uint32_t>(PadTo(context, 256));
  const auto add = [&](std::uint32_t layer, bool f16, std::uint64_t ne0, std::uint64_t ne1) {
    const std::uint64_t bytes = ne0 * ne1 * (f16 ? 2 : 4);
    s.tensors.push_back(
        {.layer = layer, .f16 = f16, .ne0 = ne0, .ne1 = ne1, .offset = s.bytes, .bytes = bytes});
    s.bytes = PadTo(s.bytes + bytes, 256);
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    if (p.linear(il)) {
      add(il, false, std::uint64_t{p.conv - 1} * p.conv_channels(), 1);
      add(il, false, std::uint64_t{p.lin_head_dim} * p.lin_head_dim * p.lin_v_heads, 1);
      if (il == p.ple_layer) {
        add(il, false, std::uint64_t{p.ple_history()} * p.hc_width(), 1);
      }
    } else {
      add(il, true, std::uint64_t{p.head_dim} * p.kv_heads, s.cells);
      add(il, true, std::uint64_t{p.head_dim} * p.kv_heads, s.cells);
      add(il, false, p.indexer_head_dim, s.cells);
      add(il, true, p.indexer_head_dim,
          (std::uint64_t{s.cells} + p.indexer_ratio - 1) / p.indexer_ratio);
    }
  }
  return s;
}

std::vector<StateRange> QwenUsedRef(const fam::Qwen38Profile& p, const QwenLayoutRef& layout,
                                    std::uint32_t positions, std::uint32_t read_align) {
  const std::uint64_t read = std::min<std::uint64_t>(layout.cells, PadTo(positions, read_align));
  std::vector<StateRange> ranges;
  for (const auto& t : layout.tensors) {
    std::uint64_t rows = t.ne1;
    if (t.ne1 == layout.cells) {
      rows = std::min(t.ne1, read);
    } else if (t.ne1 == (std::uint64_t{layout.cells} + p.indexer_ratio - 1) / p.indexer_ratio) {
      rows = std::min(t.ne1, (read + p.indexer_ratio - 1) / p.indexer_ratio);
    }
    if (rows != 0) {
      ranges.push_back({.offset = t.offset, .bytes = t.ne0 * rows * (t.f16 ? 2 : 4)});
    }
  }
  return ranges;
}

std::vector<StateRange> QwenCheckpointRef(const fam::Qwen38Profile& p, const QwenLayoutRef& layout,
                                          std::uint32_t positions) {
  std::vector<StateRange> writes;
  for (const auto& t : layout.tensors) {
    std::uint64_t first = 0;
    if (t.ne1 == layout.cells) {
      first = positions;
    } else if (t.ne1 == (std::uint64_t{layout.cells} + p.indexer_ratio - 1) / p.indexer_ratio) {
      first = positions / p.indexer_ratio;
    }
    first = std::min(first, t.ne1);
    const std::uint64_t row = t.ne0 * (t.f16 ? 2 : 4);
    if (first < t.ne1) {
      writes.push_back({.offset = t.offset + (first * row), .bytes = (t.ne1 - first) * row});
    }
  }
  return writes;
}

struct QwenMtpRef {
  std::uint32_t cells = 0;
  std::uint32_t hidden_rows = 0;
  std::uint64_t k = 0;
  std::uint64_t v = 0;
  std::uint64_t indexer = 0;
  std::uint64_t blocks = 0;
  std::uint64_t hidden = 0;
  std::uint64_t bytes = 0;
};

QwenMtpRef QwenMtp(const fam::Qwen38Profile& p, std::uint32_t context, std::uint32_t max_rows) {
  QwenMtpRef s;
  s.cells = static_cast<std::uint32_t>(PadTo(context, 256));
  s.hidden_rows = max_rows + 1;
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * s.cells * 2;
  s.k = 0;
  s.v = PadTo(kv, 256);
  s.indexer = s.v + PadTo(kv, 256);
  s.blocks = s.indexer + PadTo(std::uint64_t{p.indexer_head_dim} * s.cells * 4, 256);
  s.hidden =
      s.blocks + PadTo(std::uint64_t{p.indexer_head_dim} *
                           ((std::uint64_t{s.cells} + p.indexer_ratio - 1) / p.indexer_ratio) * 2,
                       256);
  s.bytes = s.hidden + PadTo(std::uint64_t{p.hc_width()} * s.hidden_rows * 4, 256);
  return s;
}

TEST(StateContractTest, Qwen38ContractsReproduceItsLayoutUsedStateAndCheckpointWrites) {
  const auto& p = fam::Qwen38Flash();
  for (const auto& [context, rows] : {std::pair{8192U, 512U}, {262144U, 2048U}, {4000U, 300U}}) {
    const auto want = QwenRef(p, context, rows);
    const Layers contracts = QwenContracts(p);
    auto got = md::Layout(contracts, context, rows);
    ASSERT_TRUE(got.has_value()) << got.error();
    ASSERT_EQ(got->items.size(), want.tensors.size());
    for (std::size_t i = 0; i < got->items.size(); ++i) {
      EXPECT_EQ(got->items[i].offset, want.tensors[i].offset) << i;
      EXPECT_EQ(got->items[i].bytes, want.tensors[i].bytes) << i;
      EXPECT_EQ(got->items[i].cells, want.tensors[i].ne1) << i;
    }
    EXPECT_EQ(got->bytes, want.bytes);
    for (const std::uint32_t align : {256U, 2048U}) {
      for (const std::uint32_t positions :
           {0U, 1U, 63U, 64U, 65U, 256U, 2049U, context - 1, context}) {
        auto used = md::UsedThrough(*got, positions, align);
        auto old = QwenUsedRef(p, want, positions, align);
        ASSERT_TRUE(used.has_value());
        ExpectSame(*used, old, std::format("Qwen3.8 {} positions align {}", positions, align));
      }
    }
    for (const std::uint32_t positions : {0U, 1U, 64U, 100U, context}) {
      auto writes = md::CheckpointWrites(*got, positions);
      auto old = QwenCheckpointRef(p, want, positions);
      ASSERT_TRUE(writes.has_value());
      ExpectSame(*writes, old, std::format("Qwen3.8 checkpoint at {}", positions));
    }
  }
}

// The MTP drafter's own region: a stage of its own.
TEST(StateContractTest, Qwen38MtpContractReproducesItsDrafterRegion) {
  const auto& p = fam::Qwen38Flash();
  const auto want = QwenMtp(p, 8192, 512);
  const std::uint64_t kv = std::uint64_t{p.head_dim} * p.kv_heads * 2;
  const Layers mtp = QwenMtpContracts(p);
  auto got = md::Layout(mtp, 8192, 512);
  ASSERT_TRUE(got.has_value()) << got.error();
  ASSERT_EQ(got->items.size(), 5U);
  EXPECT_EQ(got->items[0].offset, want.k);
  EXPECT_EQ(got->items[1].offset, want.v);
  EXPECT_EQ(got->items[2].offset, want.indexer);
  EXPECT_EQ(got->items[3].offset, want.blocks);
  EXPECT_EQ(got->items[4].offset, want.hidden);
  EXPECT_EQ(got->items[4].cells, want.hidden_rows);
  EXPECT_EQ(got->bytes, want.bytes);
  // The drafter ranges for its caches.
  for (const std::uint32_t positions : {1U, 300U, 8192U}) {
    for (const std::uint32_t align : {256U, 2048U}) {
      auto used = md::UsedThrough(*got, positions, align);
      ASSERT_TRUE(used.has_value());
      const std::uint64_t cells =
          std::min<std::uint64_t>(want.cells, (positions + align - 1) / align * align);
      const std::uint64_t blocks = (cells + p.indexer_ratio - 1) / p.indexer_ratio;
      EXPECT_EQ((*used)[0].bytes, cells * kv);
      EXPECT_EQ((*used)[1].bytes, cells * kv);
      EXPECT_EQ((*used)[2].bytes, cells * p.indexer_head_dim * 4);
      EXPECT_EQ((*used)[3].bytes, blocks * p.indexer_head_dim * 2);
      // The streams whole: the runner names their padded bytes, the same
      // 2 MiB extents.
      EXPECT_EQ((*used)[4].offset, want.hidden);
      EXPECT_EQ(((*used)[4].offset + (*used)[4].bytes + kExtent - 1) / kExtent,
                (want.bytes + kExtent - 1) / kExtent);
    }
  }
}

// --------------------------------------------------------------- DeepSeek

Layers Dsv4Contracts(const fam::Dsv4Profile& p, fam::Dsv4Window window) {
  Layers layers;
  const auto compressed = [](std::string name, StateKind kind, std::uint64_t cell,
                             std::uint32_t ratio) {
    return StateTensorContract{.name = std::move(name),
                               .kind = kind,
                               .cell_bytes = cell,
                               .cells = CellRule::kPerBlock,
                               .block_ratio = ratio,
                               .basis = md::BlockBasis::kPositions,
                               .dummy_last = true,
                               .rollback = Rollback::kSnapshot};
  };
  // Each compressor's ring of its recent positions' projections: position
  // p in row p mod rows, saved by the verify row that persists it.
  const auto ring = [](std::string name, std::uint64_t cell, std::uint32_t rows) {
    return StateTensorContract{.name = std::move(name),
                               .kind = StateKind::kCompressor,
                               .encoding = Encoding::kF32,
                               .cell_bytes = cell,
                               .cells = CellRule::kPositionRing,
                               .fixed_cells = rows,
                               .rollback = Rollback::kSnapshot};
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    StateContract c;
    const std::uint64_t raw = std::uint64_t{p.head_dim} * 2;
    if (window == fam::Dsv4Window::kFull) {
      c.tensors.push_back({.name = "raw",
                           .kind = StateKind::kKv,
                           .cell_bytes = raw,
                           .rollback = Rollback::kSnapshot});
    } else {
      c.tensors.push_back({.name = "raw",
                           .kind = StateKind::kRing,
                           .cell_bytes = raw,
                           .cells = CellRule::kRing,
                           .window = p.window,
                           .ring_reads_whole = true,
                           .rollback = Rollback::kSnapshot});
    }
    const std::uint64_t hd2 = 2ULL * p.head_dim * 4;
    const std::uint64_t ihd2 = 2ULL * p.indexer_head_dim * 4;
    if (p.compress_ratios[il] == fam::kDsv4CsaRatio) {
      c.tensors.push_back(compressed("csa", StateKind::kCompressed, std::uint64_t{p.head_dim} * 2,
                                     fam::kDsv4CsaRatio));
      c.tensors.push_back(ring("csa_kv", hd2, 2 * fam::kDsv4CsaRatio));
      c.tensors.push_back(ring("csa_score", hd2, 2 * fam::kDsv4CsaRatio));
      c.tensors.push_back(compressed("lid", StateKind::kIndex,
                                     std::uint64_t{p.indexer_head_dim} * 2, fam::kDsv4CsaRatio));
      c.tensors.push_back(ring("lid_kv", ihd2, 2 * fam::kDsv4CsaRatio));
      c.tensors.push_back(ring("lid_score", ihd2, 2 * fam::kDsv4CsaRatio));
    } else if (p.compress_ratios[il] == fam::kDsv4HcaRatio) {
      c.tensors.push_back(compressed("hca", StateKind::kCompressed, std::uint64_t{p.head_dim} * 2,
                                     fam::kDsv4HcaRatio));
      c.tensors.push_back(ring("hca_kv", std::uint64_t{p.head_dim} * 4, fam::kDsv4HcaRatio));
      c.tensors.push_back(ring("hca_score", std::uint64_t{p.head_dim} * 4, fam::kDsv4HcaRatio));
    }
    layers.push_back({il, std::move(c)});
  }
  return layers;
}

// DeepSeek V4's ranges in closed form, as the retired runner's were over
// its layout (fam::Dsv4State). What `positions` positions read: a full
// window's cells up to the next 256, a ring whole; each compressed cache's
// visible rows up to the next 256 (at least 256) and its last (dummy) row;
// the compressor rings whole.
std::vector<StateRange> Dsv4UsedRef(const fam::Dsv4StateLayout& state, std::uint32_t positions) {
  using K = fam::Dsv4StateTensor::Kind;
  std::vector<StateRange> ranges;
  for (const auto& t : state.tensors) {
    std::uint64_t rows = t.ne1;
    bool compressed = false;
    if (t.kind == K::kRawK && state.window == fam::Dsv4Window::kFull) {
      rows = std::min(t.ne1, PadTo(positions, 256));
    } else if (t.kind == K::kCsaK || t.kind == K::kLidK || t.kind == K::kHcaK) {
      const std::uint64_t ratio = t.kind == K::kHcaK ? fam::kDsv4HcaRatio : fam::kDsv4CsaRatio;
      const std::uint64_t visible = (std::uint64_t{positions} + ratio - 1) / ratio;
      rows = std::min(t.ne1, std::max<std::uint64_t>(256, PadTo(visible, 256)));
      compressed = true;
    }
    if (rows != 0) ranges.push_back({.offset = t.offset, .bytes = rows * t.row_bytes()});
    if (compressed && rows < t.ne1) {
      ranges.push_back(
          {.offset = t.offset + ((t.ne1 - 1) * t.row_bytes()), .bytes = t.row_bytes()});
    }
  }
  return ranges;
}

// What later positions may overwrite: a full window's cells from
// `positions`, a ring whole; each compressed cache's rows from the block
// `positions` is in, and its dummy row (every chunk writes it); the
// compressor rings whole.
std::vector<StateRange> Dsv4CheckpointRef(const fam::Dsv4StateLayout& state,
                                          std::uint32_t positions) {
  using K = fam::Dsv4StateTensor::Kind;
  std::vector<StateRange> writes;
  for (const auto& t : state.tensors) {
    std::uint64_t first = 0;
    if (t.kind == K::kRawK && state.window == fam::Dsv4Window::kFull) {
      first = positions;
    } else if (t.kind == K::kCsaK || t.kind == K::kLidK || t.kind == K::kHcaK) {
      const std::uint64_t ratio = t.kind == K::kHcaK ? fam::kDsv4HcaRatio : fam::kDsv4CsaRatio;
      first = positions / ratio;
      writes.push_back(
          {.offset = t.offset + ((t.ne1 - 1) * t.row_bytes()), .bytes = t.row_bytes()});
    }
    first = std::min(first, t.ne1);
    if (first < t.ne1) {
      writes.push_back(
          {.offset = t.offset + (first * t.row_bytes()), .bytes = (t.ne1 - first) * t.row_bytes()});
    }
  }
  return writes;
}

// Every byte a chunk of `rows` rows after `past` positions writes, by the
// row that writes it: its position's window cell in every layer, each
// compressor ring row it is the chunk's last position of (both the
// projections' and the scores'), and each compressed row of the block its
// position completes; and as scratch, whatever is accepted, each
// compressor's dummy block (a CSA compressor pads the chunk to a block per
// ratio rows, an HCA one to one block), written to its cache's last row.
struct Dsv4WritesRef {
  std::vector<std::vector<StateRange>> rows;
  std::vector<StateRange> scratch;
};
Dsv4WritesRef Dsv4ChunkWritesRef(const fam::Dsv4Profile& p, const fam::Dsv4StateLayout& state,
                                 std::uint32_t past, std::uint32_t rows) {
  using K = fam::Dsv4StateTensor::Kind;
  Dsv4WritesRef out;
  out.rows.resize(rows);
  const auto at = [](const fam::Dsv4StateTensor& t, std::uint64_t row) {
    return StateRange{.offset = t.offset + (row * t.row_bytes()), .bytes = t.row_bytes()};
  };
  const auto compressor = [&](std::uint32_t il, std::uint32_t ratio, std::uint32_t ring, K kv_kind,
                              K score_kind, K cache_kind) {
    const auto* kv = state.Find(il, kv_kind);
    const auto* score = state.Find(il, score_kind);
    const auto* cache = state.Find(il, cache_kind);
    ASSERT_TRUE(kv != nullptr && score != nullptr && cache != nullptr);
    std::map<std::uint64_t, std::uint32_t> kept;  // ring row -> its last chunk row
    std::uint32_t blocks = 0;
    for (std::uint32_t i = 0; i < rows; ++i) {
      const std::uint64_t pos = std::uint64_t{past} + i;
      kept[pos % ring] = i;
      if ((pos + 1) % ratio == 0) {
        out.rows[i].push_back(at(*cache, pos / ratio));
        ++blocks;
      }
    }
    for (const auto& [row, i] : kept) {
      out.rows[i].push_back(at(*kv, row));
      out.rows[i].push_back(at(*score, row));
    }
    const std::uint32_t padded = ratio == fam::kDsv4CsaRatio ? (rows + ratio - 1) / ratio : 1;
    if (blocks < padded) out.scratch.push_back(at(*cache, cache->ne1 - 1));
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const auto* raw = state.Find(il, K::kRawK);
    if (raw == nullptr) {
      ADD_FAILURE() << "no window cache in layer " << il;
      return out;
    }
    for (std::uint32_t i = 0; i < rows; ++i) {
      out.rows[i].push_back(at(*raw, (std::uint64_t{past} + i) % state.raw_cells));
    }
    const std::uint32_t ratio = p.compress_ratios[il];
    if (ratio == fam::kDsv4CsaRatio) {
      compressor(il, ratio, state.csa_state_rows, K::kCsaStateKv, K::kCsaStateScore, K::kCsaK);
      compressor(il, ratio, state.csa_state_rows, K::kLidStateKv, K::kLidStateScore, K::kLidK);
    } else if (ratio == fam::kDsv4HcaRatio) {
      compressor(il, ratio, state.hca_state_rows, K::kHcaStateKv, K::kHcaStateScore, K::kHcaK);
    }
  }
  return out;
}

// The DSpark ring cells a target chunk's injection writes, by chunk row:
// its last min(rows, ring) rows' positions' cells, in every block's ring.
std::vector<std::vector<StateRange>> DsparkWritesRef(const fam::DsparkProfile& d,
                                                     const fam::DsparkStateLayout& state,
                                                     std::uint32_t past, std::uint32_t rows) {
  std::vector<std::vector<StateRange>> out(rows);
  const std::uint64_t cell = std::uint64_t{d.blocks.head_dim} * 2;
  for (std::uint32_t i = rows - std::min(rows, state.ring); i < rows; ++i) {
    const std::uint64_t at = (std::uint64_t{past} + i) % state.ring;
    for (const std::uint64_t offset : state.offsets) {
      out[i].push_back({.offset = offset + (at * cell), .bytes = cell});
    }
  }
  return out;
}

TEST(StateContractTest, DeepSeekContractsReproduceBothWindowsUsedStateAndCheckpointWrites) {
  const auto& p = fam::Dsv4Flash();
  for (const auto window : {fam::Dsv4Window::kFull, fam::Dsv4Window::kRing}) {
    for (const auto& [context, rows] : {std::pair{4096U, 512U}, {65536U, 2048U}, {1000U, 128U}}) {
      auto want = fam::Dsv4State(p, context, rows, window);
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
        ASSERT_TRUE(used.has_value());
        ExpectSame(*used, Dsv4UsedRef(*want, positions),
                   std::format("DeepSeek used at {}", positions));
        auto writes = md::CheckpointWrites(*got, positions);
        ASSERT_TRUE(writes.has_value());
        ExpectSame(*writes, Dsv4CheckpointRef(*want, positions),
                   std::format("DeepSeek checkpoint at {}", positions));
      }
    }
  }
}

TEST(StateContractTest, AdjacentBanksAreDerivedFromActualOffsetsWithoutChangingLayouts) {
  const auto& profile = fam::Dsv4Flash();
  const auto contracts = Dsv4Contracts(profile, fam::Dsv4Window::kRing);
  const auto layer = static_cast<std::uint32_t>(std::ranges::find(profile.compress_ratios, 4U) -
                                                profile.compress_ratios.begin());
  ASSERT_LT(layer, profile.layers);
  for (const auto alignment : std::array<std::uint64_t, 2>{256, kExtent}) {
    auto layout = md::Layout(contracts, 65536, 512, alignment);
    ASSERT_TRUE(layout);
    const auto raw = std::ranges::find_if(
        layout->items, [&](const auto& item) { return item.layer == layer && item.tensor == 0; });
    ASSERT_NE(raw, layout->items.end());
    const std::array slices{md::StateSlice{.layer = layer, .tensor = 0, .cells = raw->cells},
                            md::StateSlice{.layer = layer, .tensor = 1, .cells = 256}};
    const auto before = md::KeptLayoutId({}, std::span(&*layout, 1));
    const auto bank = md::ContiguousStateSpan(*layout, slices);
    EXPECT_EQ(md::KeptLayoutId({}, std::span(&*layout, 1)), before);
    if (alignment == kExtent) {
      EXPECT_FALSE(bank);  // raw's 768KiB is followed by an alignment gap
      continue;
    }
    ASSERT_TRUE(bank);
    EXPECT_EQ(bank->range.offset, raw->offset);
    EXPECT_EQ(bank->range.bytes, (raw->cells + 256) * raw->contract.cell_bytes);
    EXPECT_EQ(bank->encoding, Encoding::kF16);
    auto used = md::UsedThrough(*layout, 512, 2048);
    ASSERT_TRUE(used);
    // The span's raw and compressed portions must both be paid for by
    // the existing closure. Do not infer backing from adjacency alone.
    std::uint64_t covered = bank->range.offset;
    for (const auto& range : *used) {
      if (range.offset <= covered && covered < range.offset + range.bytes)
        covered = range.offset + range.bytes;
      if (covered >= bank->range.offset + bank->range.bytes) break;
    }
    EXPECT_GE(covered, bank->range.offset + bank->range.bytes);
  }
}

TEST(StateContractTest, BankFactsRejectGapsForeignOrRepeatedItemsAndMalformedRanges) {
  const Layers contracts{
      {3, {.tensors = {{.name = "a", .cell_bytes = 1024}, {.name = "b", .cell_bytes = 1024}}}}};
  auto layout = md::Layout(contracts, 512, 128);
  ASSERT_TRUE(layout);
  const std::array slices{md::StateSlice{.layer = 3, .tensor = 0, .cells = 512},
                          md::StateSlice{.layer = 3, .tensor = 1, .cells = 256}};
  ASSERT_TRUE(md::ContiguousStateSpan(*layout, slices));
  const auto refuse_slices = [&](const auto& change) {
    auto bad = slices;
    change(bad);
    EXPECT_FALSE(md::ContiguousStateSpan(*layout, bad));
  };
  refuse_slices([](auto& s) { std::swap(s[0], s[1]); });
  refuse_slices([](auto& s) { --s[0].cells; });
  refuse_slices([](auto& s) { ++s[1].first; });
  refuse_slices([](auto& s) { s[1].layer = 4; });
  refuse_slices([](auto& s) { s[1].tensor = 0; });
  refuse_slices([](auto& s) { s[1].tensor = 9; });
  refuse_slices([](auto& s) { s[1].cells = 513; });
  refuse_slices([](auto& s) { s[1].first = std::numeric_limits<std::uint64_t>::max(); });
  refuse_slices([](auto& s) { s[0].cells = 0; });
  const auto refuse_layout = [&](const auto& change) {
    auto bad = *layout;
    change(bad);
    EXPECT_FALSE(md::ContiguousStateSpan(bad, slices));
  };
  refuse_layout([](auto& s) { ++s.items[1].offset; });
  refuse_layout([](auto& s) { --s.items[1].offset; });
  refuse_layout([](auto& s) { s.items[1].contract.encoding = Encoding::kF32; });
  refuse_layout([](auto& s) { s.items[1].contract.borrowed_from_target = true; });
  refuse_layout([](auto& s) { s.items.push_back(s.items[1]); });
  refuse_layout([](auto& s) { --s.items[1].bytes; });
  refuse_layout([](auto& s) { s.items[1].offset = std::numeric_limits<std::uint64_t>::max(); });
  refuse_layout([](auto& s) { s.bytes = 1; });
  refuse_layout([](auto& s) {
    auto other = s.items[0];
    other.tensor = 2;
    s.items.push_back(other);
  });
  refuse_layout([](auto& s) {
    s.borrowed.push_back({.layer = 3, .tensor = 0, .contract = s.items[0].contract});
  });
  // Slice adjacency cannot hide overlapping full allocations: these slices
  // meet at 256KiB, but A's unseen tail aliases B's first half.
  auto overlap = *layout;
  overlap.items[1].offset = 256 * 1024;
  auto short_first = slices;
  short_first[0].cells = 256;
  EXPECT_FALSE(md::ContiguousStateSpan(overlap, short_first));
  EXPECT_FALSE(md::ContiguousStateSpan(*layout, {}));
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
                     .window = 64,
                     .cell_pad = 1,
                     .rollback = Rollback::kSnapshot},
                    {.name = "blocks",
                     .kind = StateKind::kCompressed,
                     .cell_bytes = 256,
                     .cells = CellRule::kPerBlock,
                     .block_ratio = 4,
                     .cell_pad = 1,
                     .dummy_last = true,
                     .partial_blocks = true,
                     .rollback = Rollback::kSnapshot},
                    {.name = "recurrent",
                     .kind = StateKind::kRecurrent,
                     .encoding = Encoding::kF32,
                     .cell_bytes = 1024,
                     .cells = CellRule::kFixed,
                     .fixed_cells = 1,
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
  // The recurrent state is rewritten whole: only a commit rebuilds its kept
  // rows, so a verify without one is refused.
  EXPECT_FALSE(md::WritesFor(layout, 78, 4).has_value());
  // Rows 78..81: the ring (80 cells) wraps after row 79; block 19 completes
  // at row 1 (position 79), and block 20 is partly written by row 3
  // (partial blocks: the chunk's last row in it).
  auto writes = md::WritesFor(layout, 78, 4, {.commits = true, .drafter = false});
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
  // The chunk completes its one block: no dummy row. The recurrent state is
  // restored whatever is kept, for the commit to rebuild.
  ASSERT_EQ(writes->scratch.size(), 1U);
  EXPECT_EQ(writes->scratch[0].offset, layout.items[3].offset);
  EXPECT_EQ(writes->scratch[0].bytes, 1024U);
  // Two rows completing no block write the dummy row.
  auto short_chunk = md::WritesFor(layout, 80, 2, {.commits = true, .drafter = false});
  ASSERT_TRUE(short_chunk.has_value());
  ASSERT_EQ(short_chunk->scratch.size(), 2U);
  EXPECT_EQ(short_chunk->scratch[0].offset, blocks.offset + ((blocks.cells - 1) * 256));
  EXPECT_EQ(short_chunk->scratch[1].offset, layout.items[3].offset);
  EXPECT_EQ(short_chunk->rows[1].back().offset, blocks.offset + (20 * 256));
  // Every saved byte is a chunk write.
  for (const auto& [past, rows] : {std::pair{78U, 4U}, {80U, 2U}}) {
    auto saved = md::WritesFor(layout, past, rows, {.commits = true, .drafter = false});
    auto chunk = md::ChunkWrites(layout, past, rows);
    ASSERT_TRUE(saved.has_value() && chunk.has_value());
    const auto inside = [&](const StateRange& r) {
      return std::ranges::any_of(*chunk, [&](const StateRange& w) {
        return r.offset >= w.offset && r.offset + r.bytes <= w.offset + w.bytes;
      });
    };
    for (const auto& row : saved->rows) {
      for (const auto& r : row) EXPECT_TRUE(inside(r)) << r.offset;
    }
    for (const auto& r : saved->scratch) EXPECT_TRUE(inside(r)) << r.offset;
  }
  // The bound holds every verify's saves, wherever it starts.
  for (const std::uint32_t rows : {1U, 3U, 4U, 9U, 16U}) {
    auto bound = md::VerifySaveBound(layout, rows, {.commits = true, .drafter = false});
    ASSERT_TRUE(bound.has_value()) << bound.error();
    for (std::uint32_t past = 0; past + rows <= 200; ++past) {
      auto saved = md::WritesFor(layout, past, rows, {.commits = true, .drafter = false});
      ASSERT_TRUE(saved.has_value());
      std::uint64_t ranges = saved->scratch.size(), bytes = 0;
      for (const auto& r : saved->scratch) bytes += (r.bytes + 255) / 256 * 256;
      for (const auto& row : saved->rows) {
        ranges += row.size();
        for (const auto& r : row) bytes += (r.bytes + 255) / 256 * 256;
      }
      EXPECT_LE(ranges, bound->ranges) << past << "+" << rows;
      EXPECT_LE(bytes, bound->bytes) << past << "+" << rows;
    }
  }
  // A state that cannot roll back refuses a verify.
  const Layers ring_only = {{0,
                             {.tensors = {{.name = "ring",
                                           .kind = StateKind::kRing,
                                           .cell_bytes = 256,
                                           .cells = CellRule::kRing,
                                           .window = 64,
                                           .rollback = Rollback::kNone}}}}};
  auto refusing = md::Layout(ring_only, 1024, 16);
  ASSERT_TRUE(refusing.has_value());
  EXPECT_FALSE(md::WritesFor(*refusing, 0, 1).has_value());
  EXPECT_FALSE(md::VerifySaveBound(*refusing, 1).has_value());
  const md::VerifyScope commits{.commits = true, .drafter = false};
  // Nor fixed state or partial blocks that truncate: a verify writes them
  // for the kept rows too (fixed state whole; with blocks 4 positions wide,
  // rows 78..81 update block 19 for kept position 78 as for rejected 79),
  // commit or not, the target's own or a tap it writes. Snapshot partial
  // blocks, restored, hold none of the kept rows' part: a commit rebuilds
  // it, as it does fixed state.
  for (const bool partial : {false, true}) {
    for (const Rollback rollback : {Rollback::kTruncate, Rollback::kSnapshot}) {
      md::StateTensorContract t{.name = "s",
                                .kind = partial ? StateKind::kBlockKeys : StateKind::kRecurrent,
                                .cell_bytes = 256,
                                .cells = partial ? CellRule::kPerBlock : CellRule::kFixed,
                                .block_ratio = partial ? 4U : 0U,
                                .fixed_cells = partial ? 0U : 1U,
                                .partial_blocks = partial,
                                .rollback = rollback};
      const Layers own = {{0, {.tensors = {t}}}};
      t.target_writes = true;
      const Layers tap = {{0, {.tensors = {t}}}};
      for (const auto& [layers, scope] :
           {std::pair{own, md::VerifyScope{}}, std::pair{tap, md::VerifyScope{.drafter = true}}}) {
        auto one = md::Layout(layers, 1024, 16);
        ASSERT_TRUE(one.has_value()) << one.error();
        const auto what =
            std::format("partial {} rollback {}", partial, static_cast<int>(rollback));
        EXPECT_FALSE(md::WritesFor(*one, 78, 4, scope).has_value()) << what;
        EXPECT_FALSE(md::VerifySaveBound(*one, 4, scope).has_value()) << what;
        md::VerifyScope with = scope;
        with.commits = true;
        EXPECT_EQ(md::WritesFor(*one, 78, 4, with).has_value(), rollback == Rollback::kSnapshot)
            << what;
      }
    }
  }
  EXPECT_FALSE(md::WritesFor(layout, 1020, 5, commits).has_value());  // past the context
  EXPECT_FALSE(md::WritesFor(layout, 0, 17, commits).has_value());    // past the chunk bound
}

TEST(StateContractTest, VerifyScopesCommitsPositionRingsAndTheTargetsWritesIntoADrafter) {
  const Layers layers = {{0,
                          {.tensors = {{.name = "kv", .kind = StateKind::kKv, .cell_bytes = 256},
                                       {.name = "recurrent",
                                        .kind = StateKind::kRecurrent,
                                        .encoding = Encoding::kF32,
                                        .cell_bytes = 1024,
                                        .cells = CellRule::kFixed,
                                        .fixed_cells = 1,
                                        .rollback = Rollback::kCommit},
                                       {.name = "compressor",
                                        .kind = StateKind::kCompressor,
                                        .encoding = Encoding::kF32,
                                        .cell_bytes = 512,
                                        .cells = CellRule::kPositionRing,
                                        .fixed_cells = 8,
                                        .rollback = Rollback::kSnapshot},
                                       {.name = "tap",
                                        .kind = StateKind::kFeatures,
                                        .cell_bytes = 256,
                                        .cells = CellRule::kPerRow,
                                        .row_carry = 1,
                                        .rollback = Rollback::kSnapshot,
                                        .target_writes = true}}}}};
  auto layout = md::Layout(layers, 4096, 16);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  const auto& ring = layout->items[2];
  const auto& tap = layout->items[3];
  EXPECT_EQ(ring.cells, 8U);  // whatever the context
  EXPECT_EQ(tap.cells, 17U);
  // A position ring is read and checkpointed whole; a chunk writes its
  // rows' cells, wrapped.
  auto used = md::UsedThrough(*layout, 3);
  ASSERT_TRUE(used.has_value());
  EXPECT_EQ((*used)[2].offset, ring.offset);
  EXPECT_EQ((*used)[2].bytes, ring.bytes);
  auto chunk = md::ChunkWrites(*layout, 6, 4);
  ASSERT_TRUE(chunk.has_value());
  EXPECT_EQ((*chunk)[2].offset, ring.offset + (6 * 512));
  EXPECT_EQ((*chunk)[2].bytes, 2U * 512);
  EXPECT_EQ((*chunk)[3].offset, ring.offset);
  EXPECT_EQ((*chunk)[3].bytes, 2U * 512);
  // kCommit state: refused without a commit, never saved with one.
  EXPECT_FALSE(md::WritesFor(*layout, 6, 4).has_value());
  auto writes = md::WritesFor(*layout, 6, 4, {.commits = true, .drafter = false});
  ASSERT_TRUE(writes.has_value()) << writes.error();
  EXPECT_TRUE(writes->scratch.empty());
  // Row i: its ring row (positions 6..9 in rows 6, 7, 0, 1) and its tap
  // row (1 + i); the per-position cache truncates.
  for (std::uint32_t i = 0; i < 4; ++i) {
    ASSERT_EQ(writes->rows[i].size(), i == 0 ? 3U : 2U) << i;
    EXPECT_EQ(writes->rows[i][0].offset, ring.offset + (((6 + i) % 8) * 512)) << i;
    EXPECT_EQ(writes->rows[i][1].offset, tap.offset + ((1 + i) * 256)) << i;
  }
  // The ring cannot take a verify writing a cell twice.
  EXPECT_FALSE(md::WritesFor(*layout, 0, 9, {.commits = true, .drafter = false}).has_value());
  // Nor a snapshot save cells the range copies cannot move (Q8_0's 34
  // bytes); truncated, they need no save.
  Layers q8 = {{0,
                {.tensors = {{.name = "kv",
                              .kind = StateKind::kKv,
                              .encoding = Encoding::kQ8_0,
                              .cell_bytes = 34,
                              .rollback = Rollback::kSnapshot}}}}};
  auto blocks = md::Layout(q8, 4096, 16);
  ASSERT_TRUE(blocks.has_value()) << blocks.error();
  EXPECT_FALSE(md::WritesFor(*blocks, 0, 2).has_value());
  q8[0].second.tensors[0].rollback = Rollback::kTruncate;
  blocks = md::Layout(q8, 4096, 16);
  ASSERT_TRUE(blocks.has_value());
  EXPECT_TRUE(md::WritesFor(*blocks, 0, 2).has_value());
  // Over a drafter's stage, the target's verify saves only what it writes
  // there (the tap): not the drafter's own state, whatever its rollback.
  auto drafter = md::WritesFor(*layout, 6, 4, {.commits = false, .drafter = true});
  ASSERT_TRUE(drafter.has_value()) << drafter.error();
  for (std::uint32_t i = 0; i < 4; ++i) {
    ASSERT_EQ(drafter->rows[i].size(), i == 0 ? 2U : 1U);
    EXPECT_EQ(drafter->rows[i][0].offset, tap.offset + ((1 + i) * 256));
  }
  // A borrowed tensor names its target's tensor of that name and layer.
  const Layers assistant = {
      {0, {.tensors = {{.name = "own", .kind = StateKind::kKv, .cell_bytes = 256}}}},
      {1,
       {.tensors = {{.name = "kv",
                     .kind = StateKind::kKv,
                     .cell_bytes = 256,
                     .borrowed_from_target = true,
                     .target_layer = 0}}}}};
  auto view = md::Layout(assistant, 4096, 16);
  ASSERT_TRUE(view.has_value()) << view.error();
  ASSERT_EQ(view->borrowed.size(), 1U);
  EXPECT_EQ(md::BorrowedItem(*layout, view->borrowed[0]), &layout->items[0]);
  auto elsewhere = view->borrowed[0];
  elsewhere.contract.target_layer = 3;
  EXPECT_EQ(md::BorrowedItem(*layout, elsewhere), nullptr);
}

// DeepSeek V4's verify (the retired runner's snapshot plan, in closed form):
// each row's window cell, the compressor ring rows it persists and the
// compressed rows it completes, and the dummy rows as scratch, in both
// window modes.
TEST(StateContractTest, DeepSeekVerifySavesAreItsContractsWritesByteForByte) {
  const auto& p = fam::Dsv4Flash();
  for (const auto window : {fam::Dsv4Window::kFull, fam::Dsv4Window::kRing}) {
    for (const auto& [context, max_rows] : {std::pair{4096U, 512U}, {65536U, 2048U}}) {
      auto state = fam::Dsv4State(p, context, max_rows, window);
      ASSERT_TRUE(state.has_value()) << state.error();
      auto layout = md::Layout(Dsv4Contracts(p, window), context, max_rows);
      ASSERT_TRUE(layout.has_value()) << layout.error();
      for (const std::uint32_t past : {0U, 1U, 2U, 3U, 4U, 5U, 7U, 8U, 126U, 127U, 128U, 129U, 250U,
                                       255U, 256U, 1020U, context - 8}) {
        for (std::uint32_t rows = 1; rows <= 8; ++rows) {
          const Dsv4WritesRef want = Dsv4ChunkWritesRef(p, *state, past, rows);
          auto got = md::WritesFor(*layout, past, rows);
          ASSERT_TRUE(got.has_value()) << got.error();
          const auto what = std::format("DeepSeek verify {}+{}", past, rows);
          ASSERT_EQ(want.rows.size(), rows);
          for (std::uint32_t i = 0; i < rows; ++i) {
            ExpectSameSet(got->rows[i], want.rows[i], std::format("{} row {}", what, i));
          }
          ExpectSameSet(got->scratch, want.scratch, what + " scratch");
        }
      }
    }
  }
}

// DSpark's ring, the drafter's stage, which the target's verify writes by
// its injection (DsparkWritesRef).
TEST(StateContractTest, DsparkVerifySavesAreItsContractsWritesByteForByte) {
  const auto& dp = fam::DsparkDeepSeekV4Flash();
  auto state = fam::DsparkState(dp, 16);
  ASSERT_TRUE(state.has_value()) << state.error();
  Layers layers;
  for (std::uint32_t il = 0; il < dp.blocks.layers; ++il) {
    layers.push_back({il,
                      {.tensors = {{.name = "ring",
                                    .kind = StateKind::kDrafterRing,
                                    .cell_bytes = std::uint64_t{dp.blocks.head_dim} * 2,
                                    .cells = CellRule::kPositionRing,
                                    .fixed_cells = dp.ring,
                                    .rollback = Rollback::kSnapshot,
                                    .target_writes = true}}}});
  }
  auto layout = md::Layout(layers, 65536, 16);
  ASSERT_TRUE(layout.has_value()) << layout.error();
  ASSERT_EQ(layout->items.size(), state->offsets.size());
  for (std::size_t i = 0; i < layout->items.size(); ++i) {
    EXPECT_EQ(layout->items[i].offset, state->offsets[i]) << i;
  }
  EXPECT_EQ(layout->bytes, state->bytes);
  for (const std::uint32_t past : {0U, 1U, 250U, 254U, 255U, 256U, 1000U, 65520U}) {
    for (std::uint32_t rows = 1; rows <= 16; ++rows) {
      const auto want = DsparkWritesRef(dp, *state, past, rows);
      auto got = md::WritesFor(*layout, past, rows, {.commits = false, .drafter = true});
      ASSERT_TRUE(got.has_value()) << got.error();
      EXPECT_TRUE(got->scratch.empty());
      ASSERT_EQ(want.size(), rows);
      for (std::uint32_t i = 0; i < rows; ++i) {
        ExpectSameSet(got->rows[i], want[i],
                      std::format("DSpark verify {}+{} row {}", past, rows, i));
      }
    }
  }
}

// How a stage's plans depend on the read cells: whole where a tensor reads
// through them, else only up to the largest ring (an injected drafter ring
// or a whole-reading window reads min(its cells, the read cells)).
TEST(StateContractTest, ReadCellsBoundFollowsWhatReadsThem) {
  const auto layout = [](std::vector<StateTensorContract> tensors) {
    Layers layers{{0, {.tensors = std::move(tensors)}}};
    auto made = md::Layout(layers, 65536, 16);
    EXPECT_TRUE(made.has_value()) << made.error();
    return *made;
  };
  const auto ring = [](std::string name, std::uint32_t cells) {
    return StateTensorContract{.name = std::move(name),
                               .kind = StateKind::kDrafterRing,
                               .cell_bytes = 256,
                               .cells = CellRule::kPositionRing,
                               .fixed_cells = cells,
                               .rollback = Rollback::kSnapshot};
  };
  const StateTensorContract whole{.name = "window",
                                  .kind = StateKind::kRing,
                                  .cell_bytes = 256,
                                  .cells = CellRule::kRing,
                                  .window = 128,
                                  .ring_reads_whole = true,
                                  .rollback = Rollback::kSnapshot};
  EXPECT_EQ(md::ReadCellsBound(layout({ring("injected", 384)})), 384U);
  // The whole-reading ring's layout cells: pad(128 + 16 rows) = 256.
  EXPECT_EQ(md::ReadCellsBound(layout({whole, ring("a", 8), ring("b", 128)})), 256U);
  EXPECT_EQ(md::ReadCellsBound(layout({whole, ring("c", 384)})), 384U);
  EXPECT_EQ(md::ReadCellsBound(layout({whole})), 256U);
  EXPECT_EQ(md::ReadCellsBound(layout({ring("d", 8)})), 8U);
  EXPECT_EQ(md::ReadCellsBound(layout({whole, GemmaKv("k", 256, false, 0)})), std::nullopt);
  EXPECT_EQ(md::ReadCellsBound(layout({ring("a", 8), GemmaKv("k", 256, true, 512)})), std::nullopt);
  EXPECT_EQ(md::ReadCellsBound(md::StateLayoutDerived{}), std::nullopt);
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

// A block-encoded KV cache (Q8_0, Q4_0): its cell is a row of whole GGML
// blocks, and every range the lifecycle derives starts at a block.
TEST(StateContractTest, BlockEncodedCellsAreWholeBlocksAndRangesStartAtBlocks) {
  // Four heads of 256 (Gemma's local layers), one head of 64 (Qwen2's).
  EXPECT_EQ(md::CellBytes(Encoding::kQ8_0, 4 * 256).value_or(0), 1024U / 32 * 34);
  EXPECT_EQ(md::CellBytes(Encoding::kQ4_0, 4 * 256).value_or(0), 1024U / 32 * 18);
  EXPECT_EQ(md::CellBytes(Encoding::kQ8_0, 64).value_or(0), 68U);
  EXPECT_EQ(md::CellBytes(Encoding::kQ4_0, 64).value_or(0), 36U);
  EXPECT_EQ(md::CellBytes(Encoding::kQ5_1, 64).value_or(0), 48U);
  EXPECT_EQ(md::CellBytes(Encoding::kF16, 4 * 256).value_or(0), 2048U);
  EXPECT_EQ(md::CellBytes(Encoding::kBF16, 4 * 256).value_or(0), 2048U);
  EXPECT_EQ(md::CellBytes(Encoding::kF32, 4 * 256).value_or(0), 4096U);
  EXPECT_FALSE(md::CellBytes(Encoding::kQ8_0, 48).has_value());  // not whole blocks
  EXPECT_FALSE(md::CellBytes(Encoding::kQ8_0, 0).has_value());
  EXPECT_FALSE(md::CellBytes(Encoding::kQ4_K, 256).has_value());  // not a state encoding
  EXPECT_FALSE(md::CellBytes(Encoding::kNvfp4, 256).has_value());

  for (const Encoding encoding : {Encoding::kQ8_0, Encoding::kQ4_0}) {
    const std::uint64_t block = encoding == Encoding::kQ8_0 ? 34 : 18;
    const std::uint64_t cell = md::CellBytes(encoding, 4 * 256).value();
    const auto kv = [&](std::string name, bool local) {
      return StateTensorContract{.name = std::move(name),
                                 .kind = local ? StateKind::kRing : StateKind::kKv,
                                 .encoding = encoding,
                                 .cell_bytes = cell,
                                 .cells = local ? CellRule::kRing : CellRule::kPerPosition,
                                 .window = local ? 1024U : 0U,
                                 .rollback = local ? Rollback::kNone : Rollback::kTruncate};
    };
    const Layers layers = {{0, {.tensors = {kv("k", false), kv("v", false)}}},
                           {1, {.tensors = {kv("k", true), kv("v", true)}}}};
    const auto layout = md::Layout(layers, 4096, 512);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    for (const auto& item : layout->items) {
      EXPECT_EQ(item.bytes, item.cells * cell) << item.contract.name;
      EXPECT_EQ(item.cells, item.contract.cells == CellRule::kRing ? 1536U : 4096U);
    }
    const auto starts_at_a_block = [&](const std::vector<StateRange>& ranges,
                                       const std::string& what) {
      for (const StateRange& r : ranges) {
        const auto item = std::ranges::find_if(layout->items, [&](const auto& i) {
          return r.offset >= i.offset && r.offset < i.offset + i.bytes;
        });
        ASSERT_NE(item, layout->items.end()) << what;
        EXPECT_EQ((r.offset - item->offset) % block, 0U) << what;
        EXPECT_EQ(r.bytes % block, 0U) << what;
      }
    };
    for (const std::uint32_t positions : {1U, 255U, 256U, 1000U, 4096U}) {
      const auto used = md::UsedThrough(*layout, positions);
      ASSERT_TRUE(used.has_value());
      starts_at_a_block(*used, std::format("used through {}", positions));
      if (positions < 4096) {
        const auto writes = md::ChunkWrites(*layout, positions, 3);
        ASSERT_TRUE(writes.has_value());
        starts_at_a_block(*writes, std::format("chunk at {}", positions));
        const auto checkpoint = md::CheckpointWrites(*layout, positions);
        ASSERT_TRUE(checkpoint.has_value());
        starts_at_a_block(*checkpoint, std::format("checkpoint at {}", positions));
      }
    }
    // Used through 1,000 positions: 1,024 cells (read alignment 256) of
    // the global layer's K.
    const auto used = md::UsedThrough(*layout, 1000).value();
    EXPECT_EQ(used.front().offset, layout->items.front().offset);
    EXPECT_EQ(used.front().bytes, 1024 * cell);

    // A cell of partial blocks is refused.
    Layers partial = layers;
    partial[0].second.tensors[0].cell_bytes = cell - 1;
    EXPECT_FALSE(md::Layout(partial, 4096, 512).has_value());
    partial[0].second.tensors[0].cell_bytes = cell + (block / 2);
    EXPECT_FALSE(md::Layout(partial, 4096, 512).has_value());
  }
}

// A numeric state tensor's encoding is a state encoding.
TEST(StateContractTest, LayoutsRefuseANonStateEncoding) {
  const Layers ints = {
      {0, {.tensors = {{.name = "k", .encoding = Encoding::kI32, .cell_bytes = 512}}}}};
  EXPECT_FALSE(md::Layout(ints, 16, 16).has_value());
  Layers unknown = ints;
  unknown[0].second.tensors[0].encoding = static_cast<Encoding>(0xFFFF);
  EXPECT_FALSE(md::Layout(unknown, 16, 16).has_value());
  const Layers floats = {
      {0, {.tensors = {{.name = "k", .encoding = Encoding::kF32, .cell_bytes = 512}}}}};
  EXPECT_TRUE(md::Layout(floats, 16, 16).has_value());
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
  // Mismatched geometry fields for each cell rule are refused.
  const Layers kv_with_window = {
      {0,
       {.tensors = {
            {.name = "k", .cell_bytes = 2, .cells = CellRule::kPerPosition, .window = 8}}}}};
  EXPECT_FALSE(md::Layout(kv_with_window, 16, 16).has_value());
  const Layers ring_with_fixed = {{0,
                                   {.tensors = {{.name = "r",
                                                 .cell_bytes = 2,
                                                 .cells = CellRule::kRing,
                                                 .window = 8,
                                                 .fixed_cells = 1}}}}};
  EXPECT_FALSE(md::Layout(ring_with_fixed, 16, 16).has_value());
  const Layers fixed_with_window = {{0,
                                     {.tensors = {{.name = "f",
                                                   .cell_bytes = 2,
                                                   .cells = CellRule::kFixed,
                                                   .window = 1,
                                                   .fixed_cells = 1}}}}};
  EXPECT_FALSE(md::Layout(fixed_with_window, 16, 16).has_value());
  const Layers block_with_carry = {{0,
                                    {.tensors = {{.name = "b",
                                                  .cell_bytes = 2,
                                                  .cells = CellRule::kPerBlock,
                                                  .block_ratio = 4,
                                                  .row_carry = 1}}}}};
  EXPECT_FALSE(md::Layout(block_with_carry, 16, 16).has_value());
  const Layers row_with_ratio = {{0,
                                  {.tensors = {{.name = "w",
                                                .cell_bytes = 2,
                                                .cells = CellRule::kPerRow,
                                                .block_ratio = 4,
                                                .row_carry = 1}}}}};
  EXPECT_FALSE(md::Layout(row_with_ratio, 16, 16).has_value());
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
