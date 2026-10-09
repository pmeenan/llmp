// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// State contracts (docs/engine-components.md, "Components", item 3): what a
// component keeps per request, declared once, and everything the engine's
// shared slot lifecycle derives from it (engine/slot_lifecycle.h). Each
// state tensor states its kind, encoding, the bytes of one cell, how many
// cells it holds against the context and the chunk bound (CellRule), which
// cells a step reads (its read alignment), and how a speculative verify's
// rejected rows are undone (Rollback). From a stage's contracts, layer by
// layer, Layout derives the region's byte layout; from that layout:
//
// - UsedThrough: the ranges a state holding `positions` positions reads and
//   must have backed (padded reads, rings, fixed state, a scratch row);
// - ChunkWrites and WritesFor: the ranges a chunk of `rows` rows after
//   `past` writes, flat or by verify row (a verify's saves, D-068);
// - CheckpointWrites: the ranges later positions may overwrite (a turn
//   checkpoint's mutable pages, an incremental spill's changed extents);
// - Representations: the D-068 representations the state declares;
// - KeptLayoutId: the identity a kept conversation's record must match
//   (D-105), from the model spec's digest and every stage's layout. The
//   IDs it gives are new ("llmp-state-v2:"): state kept under the runners'
//   own IDs does not match and is refused, never migrated.
//
// Each rule reproduces a current family's arithmetic exactly (the tests
// compare them byte for byte): Gemma2/3 (2 MiB-aligned K/V, local rings
// read as written), Qwen3.8 (per-position caches, indexer block keys over
// the read cells, fixed recurrent and convolution state) and DeepSeek V4
// (full or ring windows, compressed caches with their dummy row, compressor
// rings). Nothing here names a family.

#ifndef LLMP_MODEL_STATE_CONTRACT_H_
#define LLMP_MODEL_STATE_CONTRACT_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "model/state.h"

namespace llmp::model {

// A state tensor's numeric content (D-108): F16 by default; lossy state
// encodings are per-alias settings, off by default.
using Encoding = format::Encoding;

// What a state tensor holds; names its D-068 representation group.
enum class StateKind : std::uint8_t {
  kKv,           // attention keys or values, a cell per position
  kRing,         // a sliding window's keys or values, a ring
  kCompressed,   // compressed attention rows, a cell per block of positions
  kIndex,        // an indexer's keys
  kBlockKeys,    // an indexer's pooled keys, a cell per block
  kRecurrent,    // recurrent state, rewritten by every step
  kConvolution,  // a convolution's input history
  kCompressor,   // a compressor's ring of recent positions' projections
  kDrafterRing,  // a block drafter's ring
  kPrefixCache,
  kFeatures,  // retained streams or features (a drafter's inputs)
};
std::string_view KindName(StateKind kind);

// How a speculative verify's rejected rows are undone.
enum class Rollback : std::uint8_t {
  kNone,      // it cannot be: no verify may write it
  kTruncate,  // rows past the kept prefix are never read again (a cell per position)
  // The verify saves what it overwrites; the rejected rows' are restored,
  // and state rewritten whole is restored and rebuilt for the kept rows by
  // the model's commit (LiveState's snapshot).
  kSnapshot,
};

// How many cells a tensor holds and which a state of `n` positions uses.
// pad(x) rounds x up to the tensor's cell_pad; align(x) up to the read
// alignment the caller gives (256 by default, wider for waves).
enum class CellRule : std::uint8_t {
  // pad(context) cells, position p in cell p; uses min(cells, align(n)).
  kPerPosition,
  // A window ring: pad(min(context, window + max_rows)) cells, position p
  // in cell p mod cells, so a chunk's rows and each row's window are always
  // in it. Uses min(cells, align(n)), or every cell if ring_reads_whole.
  kRing,
  // A cell per `window_or_ratio` positions (BlockBasis).
  kPerBlock,
  // `window_or_ratio` cells whatever the context, every one used.
  kFixed,
  // max_rows + `window_or_ratio` cells (a chunk's streams), every one used.
  kPerRow,
};

// Which positions a kPerBlock tensor's cells cover.
enum class BlockBasis : std::uint8_t {
  // The positions themselves: pad(max(1, ceil(n / ratio))) cells, the
  // padding in blocks (DeepSeek's compressed caches).
  kPositions,
  // The cells a per-position tensor reads: ceil(min(pad(context),
  // align(n)) / ratio), the padding in positions (Qwen3.8's block keys).
  kReadCells,
};

struct StateTensorContract {
  std::string name;  // within its component ("k", "v", "ring")
  StateKind kind = StateKind::kKv;
  Encoding encoding = Encoding::kF16;
  std::uint64_t cell_bytes = 0;  // one cell (one position's row, one block's)
  CellRule cells = CellRule::kPerPosition;
  std::uint32_t window_or_ratio = 0;  // kRing's window, kPerBlock's ratio, kFixed/kPerRow cells
  std::uint32_t cell_pad = 256;
  bool ring_reads_whole = false;  // kRing: attention reads every cell
  BlockBasis basis = BlockBasis::kPositions;
  // kPerBlock: the last cell is every chunk's scratch row (always used, a
  // checkpoint's and a verify's scratch write).
  bool dummy_last = false;
  Rollback rollback = Rollback::kTruncate;
  // A drafter's read-only view of the target's tensor (the assistant reads
  // the target's KV): no bytes of its own; while its stage reads it, the
  // target's state is borrowed (SlotLifecycle::Borrow).
  bool borrowed_from_target = false;
  bool operator==(const StateTensorContract&) const = default;
};

// One layer's (one component instance's) state.
struct StateContract {
  std::vector<StateTensorContract> tensors;
  bool operator==(const StateContract&) const = default;
};

// A stage's state region, derived: each owned tensor's place, in layer
// order and each layer's tensor order, at `tensor_align` boundaries.
struct StateLayoutDerived {
  std::uint32_t context = 0;
  std::uint32_t max_rows = 0;
  std::uint64_t tensor_align = 256;
  struct Item {
    std::uint32_t layer = 0;
    std::uint32_t tensor = 0;  // index in its layer's contract
    StateTensorContract contract;
    std::uint64_t cells = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;  // cells × cell_bytes
  };
  std::vector<Item> items;
  // Tensors read from the target (borrowed_from_target): no bytes here.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> borrowed;  // layer, tensor
  std::uint64_t bytes = 0;  // the region: the last tensor padded to tensor_align
};

// The region for `per_layer` contracts (layer index, contract) at a
// context and chunk bound. Refused for a zero or inconsistent bound, a
// tensor with no cells or bytes, a ring with no window, a block with no
// ratio, an alignment that is not a power of two of at least 256 bytes, or
// any arithmetic that overflows.
std::expected<StateLayoutDerived, std::string> Layout(
    std::span<const std::pair<std::uint32_t, StateContract>> per_layer, std::uint32_t context,
    std::uint32_t max_rows, std::uint64_t tensor_align = 256);

// The ranges a state of `positions` positions reads, each tensor's from
// its start (and a dummy row's), skipping empty ones; `read_align` a
// positive multiple of 256.
std::expected<std::vector<StateRange>, std::string> UsedThrough(const StateLayoutDerived& layout,
                                                                std::uint32_t positions,
                                                                std::uint32_t read_align = 256);

// What a chunk of `rows` rows after `past` positions writes, tensor by
// tensor: its cells (a ring's wrapped in two), the blocks its positions
// fall in, a dummy row, and fixed or per-row state whole.
std::expected<std::vector<StateRange>, std::string> ChunkWrites(const StateLayoutDerived& layout,
                                                                std::uint32_t past,
                                                                std::uint32_t rows);

// The writes a verify must save, of its kSnapshot tensors (a kTruncate
// tensor's rows past the kept prefix are never read, so it saves none):
// `rows[i]` what row i writes that its rejection restores (its cell; a
// block, by the last of the chunk's rows in it), `scratch` what is
// restored whatever is kept (dummy rows, and fixed or per-row state, which
// a step rewrites whole, for the model's commit to rebuild). Refused if
// the chunk writes a tensor whose rollback is kNone.
struct StateWrites {
  std::vector<std::vector<StateRange>> rows;
  std::vector<StateRange> scratch;
};
std::expected<StateWrites, std::string> WritesFor(const StateLayoutDerived& layout,
                                                  std::uint32_t past, std::uint32_t rows);

// The ranges positions after `positions` may overwrite: a per-position
// tensor's cells from `positions` on, a block tensor's from the block
// `positions` falls in (and its dummy row, first), a ring, fixed or
// per-row tensor whole.
std::expected<std::vector<StateRange>, std::string> CheckpointWrites(
    const StateLayoutDerived& layout, std::uint32_t positions);

// The layout as D-068 representations: one per kind, in the order kinds
// first appear, "<prefix>.<kind>", each one fixed block of the bytes its
// tensors occupy (alignment included). Appends; also truncates when every
// tensor of the kind can be rolled back: kTruncate always, kSnapshot only
// with a verify's snapshot (`verify`).
std::vector<StateRepresentation> Representations(const StateLayoutDerived& layout,
                                                 std::string_view prefix, bool verify);

// "llmp-state-v2:<hex>": the SHA-256 of the spec's digest and every
// stage's layout (each tensor's contract, cells, place and bytes, the
// stage's context, chunk bound, alignment and region bytes), in stage
// order. Anything that changes what a byte of kept state means changes it.
std::string KeptLayoutId(const base::Sha256Digest& spec,
                         std::span<const StateLayoutDerived> stages);

}  // namespace llmp::model

#endif  // LLMP_MODEL_STATE_CONTRACT_H_
