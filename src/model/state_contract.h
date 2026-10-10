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
//   `past` writes, flat or by verify row (a verify's saves, D-068), and
//   SaveBound, what the saves of any verify of `rows` rows take at most;
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
// rings) and its DSpark drafter's ring; a verify's saves reproduce Qwen3.8's,
// DeepSeek V4's and DSpark's. Nothing here names a family.

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
  kNone,  // it cannot be: no verify may write it
  // Rows past the kept prefix are never read again: a cell per position,
  // a ring (its cells hold every window beside a chunk), a completed block;
  // never fixed state or partial blocks, which a verify writes for the kept
  // rows too (refused). A position ring or per-row cells truncate only if
  // the model never reads a rejected row's cell (DSpark's ring holds its
  // window beside a block): the contract cannot say which cells it reads.
  kTruncate,
  // The verify saves what it overwrites; the rejected rows' are restored,
  // and state rewritten whole is restored and rebuilt for the kept rows by
  // the model's commit (LiveState's snapshot).
  kSnapshot,
  // A verify reads it but never writes it (its plan's verify form saves
  // each row's inputs instead), and the model's commit advances it through
  // the kept rows (Qwen3.8's recurrent and convolution state): nothing saved.
  kCommit,
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
  // max_rows + `window_or_ratio` cells (a chunk's streams), every one used:
  // the first `window_or_ratio` carry the rows before the chunk, and its
  // row i is cell window_or_ratio + i.
  kPerRow,
  // `window_or_ratio` cells whatever the context, position p in cell
  // p mod cells, every one used (a compressor's ring of recent positions,
  // a block drafter's ring).
  kPositionRing,
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
  // kPerBlock: the last cell is the scratch row of a chunk that completes
  // fewer blocks than ceil(rows / ratio) (always used, a checkpoint's
  // write; a verify's scratch save when the verify writes it).
  bool dummy_last = false;
  // kPerBlock: a chunk writes a block's cell before the block completes
  // (pooled keys updated by every row), so a verify saves it by the chunk's
  // last row in it, and restored it holds none of the kept rows' part: only
  // a commit rebuilds that (a verify without one is refused, as for fixed
  // state); otherwise only the row completing a block writes it.
  bool partial_blocks = false;
  Rollback rollback = Rollback::kTruncate;
  // A drafter's tensor that the target's steps write too (a feature tap:
  // an injection, exported streams): a target verify saves it in the
  // drafter's stage, by the verify row that writes it.
  bool target_writes = false;
  // A drafter's read-only view of the target's tensor (the assistant reads
  // the target's KV): no bytes of its own; it names the target's tensor of
  // the same name in layer `target_layer`, and while its stage steps, the
  // target's state is borrowed (SlotLifecycle::Borrow).
  bool borrowed_from_target = false;
  std::uint32_t target_layer = 0;
  bool operator==(const StateTensorContract&) const = default;
};

// The bytes of one cell holding `values` values of a state encoding
// (format::StateEncodings()), by the format layer's size rule: dense
// elements, or a block encoding's whole blocks as GGML stores them
// (format::Layout::kGgmlBlocks), with no over-read padding, since a cache's
// attention reads only its rows' blocks. A K or V cell of h heads of d
// values is h · d / 32 blocks: 34 bytes each in Q8_0, 18 in Q4_0 (1,088 and
// 576 bytes for four heads of 256). Refused for another encoding, no
// values, or values that are not whole blocks.
std::expected<std::uint64_t, std::string> CellBytes(Encoding encoding, std::uint64_t values);

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
  struct Borrowed {
    std::uint32_t layer = 0;
    std::uint32_t tensor = 0;  // index in its layer's contract
    StateTensorContract contract;
  };
  std::vector<Borrowed> borrowed;
  std::uint64_t bytes = 0;  // the region: the last tensor padded to tensor_align
};

// The target's item a drafter's borrowed tensor reads: the tensor of its
// name in its target layer, of equal cell bytes; none if the target has no
// such tensor.
const StateLayoutDerived::Item* BorrowedItem(const StateLayoutDerived& target,
                                             const StateLayoutDerived::Borrowed& borrowed);

// A drafter stage that reads its target's state frozen and stores none of
// its rows itself (Gemma 4's assistant: the target's K and V borrowed, its
// only tensors taps the target writes): a step's rows read the cells before
// their positions, so its reads cover `past`, not `past + rows`.
bool ReadsFrozen(const StateLayoutDerived& layout);

// The stage's feature tap: its one tensor of kind kFeatures that the target
// writes, a position ring or chunk rows (a drafter's input features);
// none if it has none or several.
const StateLayoutDerived::Item* FeatureTap(const StateLayoutDerived& layout);

// The region for `per_layer` contracts (layer index, contract) at a
// context and chunk bound. Refused for a zero or inconsistent bound, a
// tensor with no cells or bytes, a cell of partial elements or (in a block
// encoding) partial blocks, a ring with no window, a block with no
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
// tensor's rows past the kept prefix are never read, and a kCommit tensor
// is not written, so they save none): `rows[i]` what row i writes that its
// rejection restores (its cell, a ring's cell, its per-row cell, the block
// it completes, or with partial_blocks the block by the last of the
// chunk's rows in it), `scratch` what is restored whatever is kept (the
// dummy row when the chunk writes it, and fixed state, which a step
// rewrites whole, for the model's commit to rebuild).
//
// The scope says whose verify: the stage's own (every tensor), or the
// target's verify over a drafter's stage (`drafter`: only the tensors the
// target writes, target_writes). `commits`: the stage has a commit
// (LiveState::SetCommit). Refused if the verify writes a tensor whose
// rollback is kNone, writes fixed or partial-block state that truncates
// (the kept rows' part is written too), writes fixed or partial-block
// snapshot state or keeps kCommit state without a commit (nothing would
// rebuild it for the kept rows), writes
// one ring cell from two rows (a chunk wider than a ring: the snapshot
// holds only the cell's state before the chunk), or would save cells that
// are not 16-byte multiples (the snapshot's range copies').
struct VerifyScope {
  bool commits = false;
  bool drafter = false;
};
struct StateWrites {
  std::vector<std::vector<StateRange>> rows;
  std::vector<StateRange> scratch;
};
std::expected<StateWrites, std::string> WritesFor(const StateLayoutDerived& layout,
                                                  std::uint32_t past, std::uint32_t rows,
                                                  VerifyScope scope = {});

// What WritesFor's saves of any verify of `rows` rows (any past) take at
// most: the ranges (a snapshot's descriptors) and their bytes, each range
// at a 256-byte boundary (LiveState::Save). Refused as WritesFor is.
struct SaveBound {
  std::uint32_t ranges = 0;
  std::uint64_t bytes = 0;
};
std::expected<SaveBound, std::string> VerifySaveBound(const StateLayoutDerived& layout,
                                                      std::uint32_t rows, VerifyScope scope = {});

// The ranges positions after `positions` may overwrite: a per-position
// tensor's cells from `positions` on, a block tensor's from the block
// `positions` falls in (and its dummy row, first), a ring, fixed,
// per-row or position-ring tensor whole.
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
