// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/state_contract.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::model {
namespace {

using Item = StateLayoutDerived::Item;

std::unexpected<std::string> Refused(std::string what) { return std::unexpected(std::move(what)); }

std::optional<std::uint64_t> Mul(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  if (__builtin_mul_overflow(a, b, &out)) return std::nullopt;
  return out;
}
std::optional<std::uint64_t> Add(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  if (__builtin_add_overflow(a, b, &out)) return std::nullopt;
  return out;
}
// `x` rounded up to a multiple of `to` (to > 0); none on overflow.
std::optional<std::uint64_t> Pad(std::uint64_t x, std::uint64_t to) {
  const auto up = Add(x, to - 1);
  if (!up) return std::nullopt;
  return *up / to * to;
}
std::uint64_t CeilDiv(std::uint64_t x, std::uint64_t by) {
  return (x / by) + (x % by != 0 ? 1U : 0U);
}

// A tensor's cells at this context and chunk bound.
std::optional<std::uint64_t> CellsOf(const StateTensorContract& t, std::uint32_t context,
                                     std::uint32_t max_rows) {
  switch (t.cells) {
    case CellRule::kPerPosition:
      return Pad(context, t.cell_pad);
    case CellRule::kRing:
      return Pad(std::min<std::uint64_t>(context, std::uint64_t{t.window_or_ratio} + max_rows),
                 t.cell_pad);
    case CellRule::kPerBlock:
      if (t.basis == BlockBasis::kPositions) {
        return Pad(std::max<std::uint64_t>(1, CeilDiv(context, t.window_or_ratio)), t.cell_pad);
      } else {
        const auto read = Pad(context, t.cell_pad);
        if (!read) return std::nullopt;
        return CeilDiv(*read, t.window_or_ratio);
      }
    case CellRule::kFixed:
      return t.window_or_ratio;
    case CellRule::kPerRow:
      return std::uint64_t{max_rows} + t.window_or_ratio;
    case CellRule::kPositionRing:
      return t.window_or_ratio;
  }
  return std::nullopt;
}

std::optional<std::string> Invalid(const StateTensorContract& t) {
  if (t.cell_bytes == 0 || t.cell_pad == 0) return "a state tensor has no cell bytes or padding";
  // A dense encoding's cell holds whole elements; a block encoding's row
  // rule is the format layer's (format/tensor_format.h).
  const format::EncodingInfo& info = format::Info(t.encoding);
  if (info.block_elements == 1 && info.element_bits % 8 == 0 && info.element_bits != 0 &&
      t.cell_bytes % (info.element_bits / 8) != 0) {
    return "a state tensor's cell holds a partial element";
  }
  // A block encoding stored as GGML's blocks (every block state encoding):
  // a cell is whole blocks, so every range of cells starts at a block.
  if (info.block_elements > 1 && info.parts.empty() && info.ggml &&
      t.cell_bytes % info.ggml->block_bytes != 0) {
    return "a state tensor's cell holds a partial block";
  }
  if (t.ring_reads_whole && t.cells != CellRule::kRing) return "only a ring reads whole";
  if (t.dummy_last && t.cells != CellRule::kPerBlock) return "only a block tensor has a dummy row";
  if (t.partial_blocks && t.cells != CellRule::kPerBlock) {
    return "only a block tensor writes partial blocks";
  }
  switch (t.cells) {
    case CellRule::kRing:
      if (t.window_or_ratio == 0) return "a ring has no window";
      return std::nullopt;
    case CellRule::kPositionRing:
      if (t.window_or_ratio == 0) return "a position ring has no cells";
      return std::nullopt;
    case CellRule::kPerBlock:
      if (t.window_or_ratio == 0) return "a block tensor has no ratio";
      return std::nullopt;
    case CellRule::kFixed:
      if (t.window_or_ratio == 0) return "fixed state has no cells";
      return std::nullopt;
    case CellRule::kPerPosition:
    case CellRule::kPerRow:
      return std::nullopt;
  }
  return "an unknown cell rule";
}

bool Valid(const StateLayoutDerived& layout) {
  return layout.context != 0 && layout.max_rows != 0 && layout.max_rows <= layout.context &&
         std::has_single_bit(layout.tensor_align) && layout.tensor_align >= 256;
}

bool AlignValid(std::uint32_t align) { return align != 0 && align % 256 == 0; }

std::uint64_t Round256(std::uint64_t bytes) { return (bytes + 255) / 256 * 256; }

// Cells a state of `positions` reads of one tensor (dummy row aside).
std::uint64_t UsedCells(const StateLayoutDerived& layout, const Item& item, std::uint32_t positions,
                        std::uint32_t align) {
  const StateTensorContract& t = item.contract;
  // Read alignment cannot overflow 64 bits from 32-bit operands.
  const std::uint64_t aligned = CeilDiv(positions, align) * align;
  switch (t.cells) {
    case CellRule::kPerPosition:
      return std::min(item.cells, aligned);
    case CellRule::kRing:
      return t.ring_reads_whole ? item.cells : std::min(item.cells, aligned);
    case CellRule::kPerBlock:
      if (t.basis == BlockBasis::kPositions) {
        const std::uint64_t blocks =
            std::max<std::uint64_t>(1, CeilDiv(positions, t.window_or_ratio));
        return std::min(item.cells, CeilDiv(blocks, t.cell_pad) * t.cell_pad);
      } else {
        const std::uint64_t padded = CeilDiv(layout.context, t.cell_pad) * t.cell_pad;
        return std::min(item.cells, CeilDiv(std::min(padded, aligned), t.window_or_ratio));
      }
    case CellRule::kFixed:
    case CellRule::kPerRow:
    case CellRule::kPositionRing:
      return item.cells;
  }
  return item.cells;
}

StateRange Cells(const Item& item, std::uint64_t first, std::uint64_t count) {
  return {.offset = item.offset + (first * item.contract.cell_bytes),
          .bytes = count * item.contract.cell_bytes};
}
StateRange Whole(const Item& item) { return {.offset = item.offset, .bytes = item.bytes}; }
StateRange Dummy(const Item& item) { return Cells(item, item.cells - 1, 1); }

std::optional<std::string> ChunkInvalid(const StateLayoutDerived& layout, std::uint32_t past,
                                        std::uint32_t rows) {
  if (!Valid(layout) || rows == 0 || rows > layout.max_rows || past > layout.context ||
      rows > layout.context - past) {
    return "a state write is out of bounds";
  }
  return std::nullopt;
}

}  // namespace

std::string_view KindName(StateKind kind) {
  switch (kind) {
    case StateKind::kKv:
      return "kv";
    case StateKind::kRing:
      return "ring";
    case StateKind::kCompressed:
      return "compressed";
    case StateKind::kIndex:
      return "index";
    case StateKind::kBlockKeys:
      return "block_keys";
    case StateKind::kRecurrent:
      return "recurrent";
    case StateKind::kConvolution:
      return "convolution";
    case StateKind::kCompressor:
      return "compressor";
    case StateKind::kDrafterRing:
      return "drafter_ring";
    case StateKind::kPrefixCache:
      return "prefix_cache";
    case StateKind::kFeatures:
      return "features";
  }
  return "unknown";
}

std::expected<std::uint64_t, std::string> CellBytes(Encoding encoding, std::uint64_t values) {
  if (!format::IsStateEncoding(encoding)) {
    return Refused(std::format("{} is not a state encoding", format::Info(encoding).name));
  }
  const auto shape = format::ShapeOf(std::span(&values, 1));
  if (values == 0 || !shape) {
    return Refused("a cell holds no values");
  }
  const format::TensorFormat cell{.encoding = encoding,
                                  .layout = format::Info(encoding).quantized
                                                ? format::Layout::kGgmlBlocks
                                                : format::Layout::kDense,
                                  .shape = *shape};
  const auto bytes = format::RowBytes(cell, format::PartRole::kData);
  if (!bytes) {
    return Refused(std::format("{} values of {}: {}", values, format::Info(encoding).name,
                               format::FormatErrorName(bytes.error())));
  }
  return *bytes;
}

std::expected<StateLayoutDerived, std::string> Layout(
    std::span<const std::pair<std::uint32_t, StateContract>> per_layer, std::uint32_t context,
    std::uint32_t max_rows, std::uint64_t tensor_align) {
  StateLayoutDerived s;
  s.context = context;
  s.max_rows = max_rows;
  s.tensor_align = tensor_align;
  if (!Valid(s)) {
    return Refused(std::format("no state layout for {} positions in chunks of {}, aligned to {}",
                               context, max_rows, tensor_align));
  }
  for (const auto& [layer, contract] : per_layer) {
    for (std::uint32_t i = 0; i < contract.tensors.size(); ++i) {
      const StateTensorContract& t = contract.tensors[i];
      if (auto bad = Invalid(t)) {
        return Refused(std::format("layer {} state {}: {}", layer, t.name, *bad));
      }
      if (t.borrowed_from_target) {
        s.borrowed.push_back({.layer = layer, .tensor = i, .contract = t});
        continue;
      }
      const auto cells = CellsOf(t, context, max_rows);
      const auto bytes = cells ? Mul(*cells, t.cell_bytes) : std::nullopt;
      const auto end = bytes ? Add(s.bytes, *bytes) : std::nullopt;
      const auto next = end ? Pad(*end, tensor_align) : std::nullopt;
      if (!next || *cells == 0) {
        return Refused(std::format("layer {} state {} has no cells or overflows", layer, t.name));
      }
      s.items.push_back({.layer = layer,
                         .tensor = i,
                         .contract = t,
                         .cells = *cells,
                         .offset = s.bytes,
                         .bytes = *bytes});
      s.bytes = *next;
    }
  }
  return s;
}

const StateLayoutDerived::Item* BorrowedItem(const StateLayoutDerived& target,
                                             const StateLayoutDerived::Borrowed& borrowed) {
  for (const Item& item : target.items) {
    if (item.layer == borrowed.contract.target_layer &&
        item.contract.name == borrowed.contract.name &&
        item.contract.cell_bytes == borrowed.contract.cell_bytes) {
      return &item;
    }
  }
  return nullptr;
}

bool ReadsFrozen(const StateLayoutDerived& layout) {
  return !layout.borrowed.empty() && std::ranges::all_of(layout.items, [](const Item& item) {
    return item.contract.target_writes;
  });
}

const StateLayoutDerived::Item* FeatureTap(const StateLayoutDerived& layout) {
  const Item* found = nullptr;
  for (const Item& item : layout.items) {
    if (item.contract.kind != StateKind::kFeatures || !item.contract.target_writes ||
        item.contract.cells != CellRule::kPositionRing) {
      continue;
    }
    if (found != nullptr) return nullptr;
    found = &item;
  }
  return found;
}

std::expected<std::vector<StateRange>, std::string> UsedThrough(const StateLayoutDerived& layout,
                                                                std::uint32_t positions,
                                                                std::uint32_t read_align) {
  if (!Valid(layout) || !AlignValid(read_align) || positions > layout.context) {
    return Refused("used state is outside its layout");
  }
  std::vector<StateRange> out;
  out.reserve(layout.items.size());
  for (const Item& item : layout.items) {
    const std::uint64_t used = UsedCells(layout, item, positions, read_align);
    if (used != 0) out.push_back(Cells(item, 0, used));
    if (item.contract.dummy_last && used < item.cells) out.push_back(Dummy(item));
  }
  return out;
}

std::expected<std::vector<StateRange>, std::string> ChunkWrites(const StateLayoutDerived& layout,
                                                                std::uint32_t past,
                                                                std::uint32_t rows) {
  if (auto bad = ChunkInvalid(layout, past, rows)) return Refused(*bad);
  std::vector<StateRange> out;
  for (const Item& item : layout.items) {
    const StateTensorContract& t = item.contract;
    switch (t.cells) {
      case CellRule::kPerPosition:
      case CellRule::kRing: {
        const std::uint64_t first = t.cells == CellRule::kRing ? past % item.cells : past;
        const std::uint64_t count = std::min<std::uint64_t>(rows, item.cells - first);
        out.push_back(Cells(item, first, count));
        if (count < rows) out.push_back(Cells(item, 0, rows - count));
        break;
      }
      case CellRule::kPerBlock: {
        const std::uint64_t first = past / t.window_or_ratio;
        const std::uint64_t last = std::min<std::uint64_t>(
            (std::uint64_t{past} + rows - 1) / t.window_or_ratio, item.cells - 1);
        if (first <= last) out.push_back(Cells(item, first, last - first + 1));
        if (t.dummy_last) out.push_back(Dummy(item));
        break;
      }
      case CellRule::kPositionRing: {
        if (rows >= item.cells) {
          out.push_back(Whole(item));
          break;
        }
        const std::uint64_t first = past % item.cells;
        const std::uint64_t count = std::min<std::uint64_t>(rows, item.cells - first);
        out.push_back(Cells(item, first, count));
        if (count < rows) out.push_back(Cells(item, 0, rows - count));
        break;
      }
      case CellRule::kFixed:
      case CellRule::kPerRow:
        out.push_back(Whole(item));
        break;
    }
  }
  return out;
}

namespace {

// Whether a verify in `scope` saves `item`'s writes; refused when it may
// not run at all.
std::expected<bool, std::string> Saves(const Item& item, VerifyScope scope) {
  const StateTensorContract& t = item.contract;
  if (scope.drafter && !t.target_writes) return false;  // the target's step never writes it
  switch (t.rollback) {
    case Rollback::kTruncate:
      // Only cells past the kept prefix may keep what the verify wrote:
      // never state it rewrites whole, nor a block the kept rows end in.
      if (t.cells == CellRule::kFixed) {
        return Refused(std::format("a verify rewrites layer {} state {} whole: it cannot truncate",
                                   item.layer, t.name));
      }
      if (t.partial_blocks) {
        return Refused(std::format(
            "a verify writes layer {} state {}'s block before it completes: it cannot truncate",
            item.layer, t.name));
      }
      return false;  // never read past the kept prefix
    case Rollback::kCommit:
      if (!scope.commits) {
        return Refused(std::format("layer {} state {} needs a commit to keep a verify's rows",
                                   item.layer, t.name));
      }
      return false;  // not written by a verify
    case Rollback::kNone:
      return Refused(std::format("a verify writes layer {} state {}, which cannot roll back",
                                 item.layer, t.name));
    case Rollback::kSnapshot:
      // The snapshot's range copies move 16-byte multiples (a block
      // encoding's 34-byte cells cannot be saved cell by cell).
      if (t.cell_bytes % 16 != 0) {
        return Refused(std::format("layer {} state {}'s {}-byte cells are not saved by a verify",
                                   item.layer, t.name, t.cell_bytes));
      }
      if (t.cells == CellRule::kFixed && !scope.commits) {
        return Refused(std::format(
            "a verify rewrites layer {} state {} whole, and no commit rebuilds its kept rows",
            item.layer, t.name));
      }
      // A partly written block restored holds none of the kept rows' part
      // either: the block they end in is the commit's to rebuild.
      if (t.partial_blocks && !scope.commits) {
        return Refused(std::format(
            "a verify writes layer {} state {}'s blocks before they complete, and no commit "
            "rebuilds the kept rows' part",
            item.layer, t.name));
      }
      return true;
  }
  return Refused("an unknown rollback");
}

}  // namespace

std::expected<StateWrites, std::string> WritesFor(const StateLayoutDerived& layout,
                                                  std::uint32_t past, std::uint32_t rows,
                                                  VerifyScope scope) {
  if (auto bad = ChunkInvalid(layout, past, rows)) return Refused(*bad);
  StateWrites out;
  out.rows.resize(rows);
  for (const Item& item : layout.items) {
    const auto saves = Saves(item, scope);
    if (!saves) return std::unexpected(saves.error());
    if (!*saves) continue;
    const StateTensorContract& t = item.contract;
    if ((t.cells == CellRule::kRing || t.cells == CellRule::kPositionRing) && rows > item.cells) {
      return Refused(std::format("a verify of {} rows writes layer {} state {}'s {} cells twice",
                                 rows, item.layer, t.name, item.cells));
    }
    std::uint64_t completed = 0;
    for (std::uint32_t i = 0; i < rows; ++i) {
      const std::uint64_t position = std::uint64_t{past} + i;
      switch (t.cells) {
        case CellRule::kPerPosition:
          out.rows[i].push_back(Cells(item, position, 1));
          break;
        case CellRule::kRing:
        case CellRule::kPositionRing:
          out.rows[i].push_back(Cells(item, position % item.cells, 1));
          break;
        case CellRule::kPerRow:
          out.rows[i].push_back(Cells(item, std::uint64_t{t.window_or_ratio} + i, 1));
          break;
        case CellRule::kPerBlock: {
          const std::uint64_t block = position / t.window_or_ratio;
          const bool completes = (position + 1) % t.window_or_ratio == 0;
          completed += completes ? 1 : 0;
          // The row completing a block writes it; with partial blocks, the
          // chunk's last row in each block too (its key is wrong once that
          // row is rejected).
          const bool writes =
              completes ||
              (t.partial_blocks && (i + 1 == rows || (position + 1) / t.window_or_ratio != block));
          if (writes && block < item.cells) out.rows[i].push_back(Cells(item, block, 1));
          break;
        }
        case CellRule::kFixed:
          break;
      }
    }
    if (t.cells == CellRule::kFixed) out.scratch.push_back(Whole(item));
    // The dummy row: written by a chunk that completes fewer blocks than
    // ceil(rows / ratio), each missing block a dummy into its last row.
    if (t.dummy_last && completed < CeilDiv(rows, t.window_or_ratio)) {
      out.scratch.push_back(Dummy(item));
    }
  }
  return out;
}

std::expected<SaveBound, std::string> VerifySaveBound(const StateLayoutDerived& layout,
                                                      std::uint32_t rows, VerifyScope scope) {
  // Its refusals are WritesFor's (bounds, rollbacks, rings).
  if (auto any = WritesFor(layout, 0, rows, scope); !any) return std::unexpected(any.error());
  SaveBound out;
  const auto add = [&](std::uint64_t ranges, std::uint64_t bytes) {
    out.ranges += static_cast<std::uint32_t>(ranges);
    out.bytes += ranges * Round256(bytes);
  };
  for (const Item& item : layout.items) {
    const auto saves = Saves(item, scope);
    if (!saves || !*saves) continue;
    const StateTensorContract& t = item.contract;
    switch (t.cells) {
      case CellRule::kPerPosition:
      case CellRule::kRing:
      case CellRule::kPositionRing:
      case CellRule::kPerRow:
        add(rows, t.cell_bytes);
        break;
      case CellRule::kPerBlock:
        // At most ceil(rows / ratio) blocks complete within `rows`
        // positions, and with partial blocks the last one besides; then
        // the dummy row.
        add(std::min<std::uint64_t>(item.cells,
                                    CeilDiv(rows, t.window_or_ratio) + (t.partial_blocks ? 1 : 0)),
            t.cell_bytes);
        if (t.dummy_last) add(1, t.cell_bytes);
        break;
      case CellRule::kFixed:
        add(1, item.bytes);
        break;
    }
  }
  return out;
}

std::expected<std::vector<StateRange>, std::string> CheckpointWrites(
    const StateLayoutDerived& layout, std::uint32_t positions) {
  if (!Valid(layout) || positions > layout.context) {
    return Refused("checkpoint writes are outside their layout");
  }
  std::vector<StateRange> out;
  for (const Item& item : layout.items) {
    const StateTensorContract& t = item.contract;
    std::uint64_t first = 0;
    if (t.cells == CellRule::kPerPosition) {
      first = positions;
    } else if (t.cells == CellRule::kPerBlock) {
      // A partly filled block is mutable; every chunk writes the dummy row.
      first = positions / t.window_or_ratio;
      if (t.dummy_last) out.push_back(Dummy(item));
    }
    first = std::min(first, item.cells);
    if (first < item.cells) out.push_back(Cells(item, first, item.cells - first));
  }
  return out;
}

std::vector<StateRepresentation> Representations(const StateLayoutDerived& layout,
                                                 std::string_view prefix, bool verify) {
  struct Group {
    StateKind kind;
    std::uint64_t bytes = 0;
    bool truncates = true;
  };
  std::vector<Group> groups;
  for (std::size_t i = 0; i < layout.items.size(); ++i) {
    const Item& item = layout.items[i];
    // The bytes it occupies: up to the next tensor, or the region's end.
    const std::uint64_t end =
        i + 1 < layout.items.size() ? layout.items[i + 1].offset : layout.bytes;
    auto at = std::ranges::find(groups, item.contract.kind, &Group::kind);
    if (at == groups.end()) {
      groups.push_back({.kind = item.contract.kind});
      at = groups.end() - 1;
    }
    at->bytes += end - item.offset;
    const Rollback r = item.contract.rollback;
    at->truncates =
        at->truncates && (r == Rollback::kTruncate ||
                          ((r == Rollback::kSnapshot || r == Rollback::kCommit) && verify));
  }
  std::vector<StateRepresentation> out;
  out.reserve(groups.size());
  for (const Group& g : groups) {
    out.push_back({.name = std::format("{}.{}", prefix, KindName(g.kind)),
                   .block_positions = 0,
                   .block_bytes = base::Bytes(g.bytes),
                   .capabilities = g.truncates
                                       ? (StateCapability::kAppend | StateCapability::kTruncate)
                                       : static_cast<std::uint8_t>(StateCapability::kAppend),
                   .max_snapshots = 0,
                   .snapshot_bytes = base::Bytes(0)});
  }
  return out;
}

std::string KeptLayoutId(const base::Sha256Digest& spec,
                         std::span<const StateLayoutDerived> stages) {
  base::Sha256 hash;
  hash.Update(std::format("llmp-state-v2\nspec {}\nstages {}\n", base::ToHex(spec), stages.size()));
  for (const StateLayoutDerived& s : stages) {
    hash.Update(std::format("stage {} {} {} {} {} {}\n", s.context, s.max_rows, s.tensor_align,
                            s.bytes, s.items.size(), s.borrowed.size()));
    for (const Item& item : s.items) {
      const StateTensorContract& t = item.contract;
      // The name length-prefixed: no separator in it can collide.
      hash.Update(std::format(
          "item {} {} {}:{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n", item.layer,
          item.tensor, t.name.size(), t.name, KindName(t.kind), format::Info(t.encoding).name,
          t.cell_bytes, static_cast<unsigned>(t.cells), t.window_or_ratio, t.cell_pad,
          t.ring_reads_whole ? 1 : 0, static_cast<unsigned>(t.basis), t.dummy_last ? 1 : 0,
          t.partial_blocks ? 1 : 0, static_cast<unsigned>(t.rollback), t.target_writes ? 1 : 0,
          item.cells, item.offset, item.bytes, t.borrowed_from_target ? 1 : 0));
    }
    for (const auto& b : s.borrowed) {
      hash.Update(std::format("borrowed {} {} {}:{} {} {}\n", b.layer, b.tensor,
                              b.contract.name.size(), b.contract.name, b.contract.target_layer,
                              b.contract.cell_bytes));
    }
  }
  return "llmp-state-v2:" + base::ToHex(hash.Finish());
}

}  // namespace llmp::model
