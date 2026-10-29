// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// From a model spec to a step's primitive graph (D-107;
// docs/engine-components.md#components): the spec validated against the
// catalog and the bound tensors (ValidateSpec), a stage's state layout
// from its components' contracts (StageLayout), its batching capability
// (StageBatching), and BuildStep: the stage's components asked, in spec
// order, for their fragments for one step shape, joined into one verified
// ir::Graph, the planner's input.
//
// A stage's components have ordinals (OrdinalOf): the embedding 0, then
// its side inputs' sources and its streams' mixing, every layer's
// components in order, the final norm and the output head.
// A weight leaf is made once per bound tensor (a tied head reads the
// embedding's leaf, as today's builders do), its WeightRef the first
// component binding it; a weight of another stage's artifact (a drafter
// reading its target's token table, spec.h WeightBinding::stage) has that
// stage's WeightRef, so both stages' steps name one tensor. A state leaf
// is (stage, the layer's contract tensor, layer, segment): a layer's
// contract is its components' contracts concatenated in order, layer 0's
// after the embedding's.
//
// Stages that read each other (a drafter, spec.h Read): a drafter's
// borrowed tensors (its contract's borrowed_from_target: Gemma 4's
// assistant reading its target's K and V) are leaves of its own stage's
// StateRef that the engine binds to the target's tensor; the step reads
// them frozen (its rows are never stored in them, so their cells hold the
// positions before the segment's: reads are clamped to the target's
// committed positions by the masks). A target's step with outputs writes
// each feature tap a drafter reads into that drafter's tensor of the tap's
// name (target_writes): the final norm's output rows, stored at their
// positions' cells, or a layer's output or streams, every row's. A step
// never stores into a borrowed tensor nor, on
// another stage's, into one its contract does not let the target write
// (CheckStateWrites).
//
// The step's shared inputs (StepContext) are made by the framework: token
// ids, positions, the head's output rows; per segment and state cell rule,
// the cells its rows are stored at and the positions its read cells hold;
// and per segment, before the embedding, every mask a component declared
// (causal ones first, then windows by width): kMaskBuild over the
// segment's positions (a view of the step's) and its cells' positions.
// Where that mask runs, the device or the host (an input), is a
// lowering's choice. Masks have Pad(rows, kMaskRowAlign) rows.
//
// A stage of several streams (hyper-connections) carries its residual, F32
// [width, streams, rows], beside the stream each block reads: its mixing
// component makes it from the embedding's rows, the components between
// read and return it (Fragment::residual) and a carry from a mix to its
// combine (Fragment::carry). A side input's source component (a table read
// at rows the engine computes) runs after the embedding, and the one
// component of each layer it lists reads its output. A verify step leaves
// fixed state (kCommit rollback) as it was and saves what its commit
// replays (StepGraph::saves, BuildCommit).
//
// Model layer, after spec: the planner's step shape is a header-only type.

#ifndef LLMP_COMPONENTS_STEP_H_
#define LLMP_COMPONENTS_STEP_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "components/component.h"
#include "components/input_values.h"
#include "execution/step_shape.h"
#include "ir/graph.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::components {

// The catalog's own bounds on a stage (beyond the spec's).
inline constexpr std::uint64_t kMaxStreamWidth = std::uint64_t{1} << 16U;
inline constexpr std::size_t kMaxStageLayers = 1024;
inline constexpr std::size_t kMaxSegments = 16;  // a step's owners
// A mask's rows are padded to this (GGML's flash attention reads masks in
// 32-row tiles, GGML_KQ_MASK_PAD; today's builders pad them so).
inline constexpr std::uint32_t kMaskRowAlign = 32;
// The node component of what the framework emits itself (masks).
inline constexpr std::uint16_t kFrameworkComponent = 0xFFFF;

// A tensor of the artifact a stage runs from, as the mapping sees it (a v1
// artifact's: spec::SourceOf).
using BoundTensor = spec::SourceTensor;

// A stage's components with their parameters validated and their ordinals:
// the embedding 0, then the side inputs' sources in the spec's (name)
// order, the stream's mixing, every layer's components, the final norm and
// the output head (a stage without side inputs or mixing numbers as
// before them).
struct StageInstances {
  std::optional<Instance> embedding;
  std::vector<Instance> side;      // spec.h SideInput::source, by the spec's order
  std::optional<Instance> mixing;  // a stage of several streams' (spec.h ResidualStream)
  std::vector<std::vector<Instance>> layers;
  std::optional<Instance> final_norm;
  std::optional<Instance> output;
};
std::expected<StageInstances, std::string> InstancesOf(const spec::ModelSpec& spec,
                                                       std::size_t stage);

// Allocation-free upper bound across all stages, including injected components.
// Source declarations bound roots; owner count is the actual serving envelope.
std::expected<std::uint64_t, std::string> CommitSaveBound(const spec::ModelSpec& model,
                                                          std::uint32_t owners);

// Validated, conservative all-role inventory, without building a graph. Parts
// borrow only for the synchronous visitor call. References use the same owner
// resolver as BuildStep, including borrowed and injected stage weights.
struct WeightBindingView {
  ir::WeightRef ref;
  std::uint32_t artifact = 0;
  format::TensorFormat format;
  std::span<const std::pair<std::string, format::PartRole>> parts;
  std::uint64_t rows_per_token = 0;  // row-paged role; zero: ordinary residency
  std::uint64_t head_rows = 0;       // actual output head's selected vocabulary
  std::uint16_t use_stage = 0;
  std::uint32_t use_component = 0;
  std::uint16_t use_role = 0;
};
using WeightVisitor = std::function<std::expected<void, std::string>(const WeightBindingView&)>;
std::expected<void, std::string> VisitWeightBindings(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    const WeightVisitor& visitor);

// Where a step with a head narrows (Fragment::narrow): the (layer,
// component) of the stage's last component that mixes rows, when it can
// narrow and the final norm and the head are row-local; every component
// after it, the final norm and the head run at the step's output rows
// rather than its rows. BuildStep and the planner's traffic model both
// take it from here.
std::optional<std::pair<std::size_t, std::size_t>> NarrowPoint(const StageInstances& instances);

// The spec checked against the catalog and the tensors bound to each
// stage (`tensors[stage artifact]`, a list per bound artifact): spec::
// Validate's rules; then every component a known kind and version, its
// parameters in range, its instance valid; every role it uses bound once,
// to a tensor of that name and part (of its stage's artifact, or of the
// artifact of the stage the binding names), in an encoding the role
// accepts and of the shape the role implies, and no binding of a role it
// does not use; a layer count and stream width within the catalog's
// bounds; a stage of several streams' mixing a mixing component
// (ComponentDef::mixing), and of one stream none; each side input's source
// a side-input component (ComponentDef::side_source), read by exactly one
// component of each layer it lists and by no other (ComponentDef::
// side_input); no in-checkpoint MTP heads (a drafter stage instead); state
// only on a layer's or the embedding's components; and the reads linked to
// the state contracts both ways: a drafter's borrowed tensor read of state
// of that name of the stage it reads (and that stage's tensor of that name
// in its target layer one of equal kind, cells, window and cell bytes), a
// tensor another stage writes read of that stage's tap of that name (in
// F32: a final norm's or a layer's output of its width, or a layer's
// streams of its streams times its width), and every read used by one.
// Only a drafter reads, and only stages before it (its target, or the
// drafter before it in a cascade), each name of one stage. The first
// violation, with where it is.
enum class ValidationPurpose : std::uint8_t { kExecution, kImport };
std::expected<void, std::string> ValidateSpec(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    ValidationPurpose purpose = ValidationPurpose::kExecution);
// A single-artifact spec.
std::expected<void, std::string> ValidateSpec(
    const spec::ModelSpec& spec, std::span<const BoundTensor> tensors,
    ValidationPurpose purpose = ValidationPurpose::kExecution);

// Complete bound tensor names whose component roles carry a catalog fold.
// The caller resolves tied aliases to storage identities before receipt comparison.
std::expected<std::vector<std::string>, std::string> FoldTensors(const spec::ModelSpec& model,
                                                                 std::uint32_t artifact,
                                                                 std::string_view fold);
// Authenticated source/importer records exactly cover each declared fold's roles.
std::expected<void, std::string> ValidateFoldReceipts(const spec::ModelSpec& model,
                                                      std::uint32_t artifact,
                                                      const spec::Source& source);

// A stage's per-layer contracts (each layer's components' concatenated,
// layer 0's after the embedding's), for model::Layout; layers without
// state are left out.
std::expected<std::vector<std::pair<std::uint32_t, model::StateContract>>, std::string>
StageContracts(const spec::ModelSpec& spec, std::size_t stage);
// A stage's state layout: model::ExtentLayout over model::kStateExtent,
// as serving lays it out.
std::expected<model::StateLayoutDerived, std::string> StageLayout(const spec::ModelSpec& spec,
                                                                  std::size_t stage,
                                                                  std::uint32_t context,
                                                                  std::uint32_t max_rows);

// The largest requested prompt chunk whose declared mask planes fit signed
// int32 strides (RE-037), from this spec's actual state contracts and geometry.
// No architecture names, checkpoint context caps or preferred chunk sizes.
std::expected<std::uint32_t, std::string> StagePrefillRows(const spec::ModelSpec& spec,
                                                           std::size_t stage, std::uint32_t context,
                                                           std::uint32_t requested,
                                                           bool rounded_selection = false);

// The stage's batching capability: its components' own limits combined
// (the tightest of each), over a step of any segments up to kMaxSegments.
// Canonical component mask declarations, shared by serving policy and graph
// construction. This reports a requirement; backend support is separate.
std::expected<bool, std::string> StageIndexedMasks(const spec::ModelSpec& spec, std::size_t stage);

std::expected<model::Batching, std::string> StageBatching(const spec::ModelSpec& spec,
                                                          std::size_t stage);

// ------------------------------------------------------------ step context

// A state tensor's cells as a step indexes them: its capacity, whether it
// is a ring (position p in cell p mod capacity), and whether the step reads
// it frozen (another stage's, borrowed: the step stores none of its rows,
// so its cells hold the positions before the segment's).
//
// Other cell rules: a fixed tensor (`fixed`: kFixed state, one cell a
// request, rewritten whole: recurrent or convolution state), a tensor of a
// cell per block of `block` positions (kPerBlock: indexer block keys, whose
// cells read as the last position of each block the stored positions
// complete), and a tensor of a cell per chunk row (`per_row`: kPerRow, a
// segment's row i at cell first_row + i: an MTP drafter's streams).
//
// A block tensor of `positions` (BlockBasis::kPositions: compressed rows,
// a cell per block of the positions themselves) reads the blocks its
// segment's positions complete, at the contract's cell padding: max(pad,
// Pad(end / block, pad)) cells (DeepSeek V4's compressed caches), not
// the blocks of the read cells. A `recent` ring (kPositionRing: a
// compressor's ring of recent positions, a block drafter's ring) holds
// only a segment's last rows: its components store the rows it keeps
// (StepContext::KeptRows, KeptCells), so a chunk may be wider than it.
struct CellKey {
  std::uint64_t capacity = 0;
  bool ring = false;
  bool frozen = false;
  bool fixed = false;
  std::uint32_t block = 0;
  bool per_row = false;
  std::uint32_t first_row = 0;
  bool positions = false;
  bool recent = false;
  // Positions-basis block reads follow their contract's padding in cells.
  std::uint32_t cell_pad = 256;
  auto operator<=>(const CellKey&) const = default;
};

// A compressor's blocks in a segment of `rows` rows after `past` (DeepSeek
// V4's compressors, llama.cpp's dsv4_build_comp_plan): each block of
// `ratio` positions its rows complete, in order; then, with `overlap` (a
// block pools the previous block's positions too), one more where they
// complete fewer than ceil(rows / ratio), and without it one where they
// complete none: a dummy block, written to the compressed cache's last
// cell, which no mask shows.
model::ShapeParameterFact CompressorBlockParameter(std::uint32_t rows, std::uint32_t past,
                                                   std::uint32_t ratio, bool overlap);
// Its next coordinate is an end position, not a past position.
model::ShapeParameterFact CompressedReadParameter(std::uint64_t end, std::uint32_t ratio,
                                                  std::uint32_t cell_pad);
std::uint64_t CompressorBlocks(std::uint32_t rows, std::uint32_t past, std::uint32_t ratio,
                               bool overlap);

// The read width of a positions-basis compressed cache before its capacity
// cap: completed blocks, rounded to the state contract's cell padding, at least
// one padding unit. Shared by fragments, scalar references and plan keys.
std::uint64_t CompressedReadCells(std::uint64_t end, std::uint32_t ratio, std::uint32_t cell_pad);

// The Hadamard matrix of `n` (a power of two) values a row, normalized
// (H[i][j] = (-1)^popcount(i & j) / sqrt(n)): the values of an
// InputKind::kHadamard input, row-major ([n, n], ne[0] a row).
std::vector<float> HadamardValues(std::uint64_t n);
std::expected<void, std::string> WriteHadamard(std::uint64_t n, std::span<std::byte> into);

// A component-declared host-computed input (InputKind::kHostComputed).
struct HostInputDef;
struct SegmentRows;
struct StepInput;
struct HostInputSpec {
  const HostInputDef* def = nullptr;
  std::uint32_t ratio = 0;
  bool overlap = false;
  auto operator<=>(const HostInputSpec&) const = default;
};
struct HostInputDef {
  std::string_view name{};
  format::Encoding encoding = format::Encoding::kI32;
  bool joined = false;
  std::uint64_t (*count)(const HostInputSpec& spec, std::span<const SegmentRows> segments,
                         std::uint8_t segment, const CellKey& cells) = nullptr;
  std::expected<void, std::string> (*validate)(const StepInput& input,
                                               const planner::StepShape& shape) = nullptr;
  std::expected<void, std::string> (*emit)(const StepInput& input, const planner::StepShape& shape,
                                           InputValueSink& out) = nullptr;
};

const HostInputDef& CompressorPhasesInput();
const HostInputDef& CompressorKeptRowsInput();
const HostInputDef& CompressorKeptCellsInput();
const HostInputDef& CompressorPoolReadsInput();
const HostInputDef& CompressorBlockCellsInput();
const HostInputDef& CompressorBlockStartsInput();

// The step's interface: an input leaf and what it holds.
enum class InputKind : std::uint8_t {
  kStream,      // F32 [width, rows]: a lone component's input stream (BuildComponent)
  kTokens,      // I32 [rows]: the step's token ids
  kPositions,   // I32 [rows]: each row's position (past + i in its segment)
  kOutputRows,  // I32 [outputs]: the head's rows (each segment's last `outputs`)
  kStoreCells,  // I64 [segment rows]: the cell each row is stored at
  // I32 [cells]: the position each read cell holds once the step's rows
  // are stored (a ring's latest below the segment's end; a frozen read's
  // below its past), -1 for none.
  kCellPositions,
  // F32 [width, rows]: the feature a drafter's rows read beside their
  // tokens (its target's tap, or its own last feature): the caller's.
  kFeatures,
  // I64 [segment outputs]: the cell each output row's tap is stored at in
  // a drafter's tensor (its position's).
  kOutputCells,
  // I32 [count · rows]: the rows of a side table each step row reads, a
  // row's `count` together (its source's lookup: Qwen3.8's n-gram hash of
  // the token history), the caller's (`component`: the source's ordinal).
  kSideRows,
  // I32 [sections · blocks]: the first position of each of a segment's
  // `read` blocks of `cells.block` positions (block b's: b · block), every
  // section a copy, section-major (a multi-section rotation's positions).
  kBlockPositions,
  // A lone component's other inputs (BuildComponent), the caller's: F32
  // [width, streams, rows] the streams, [width, rows] a carry and a side
  // input.
  kResidual,
  kCarry,
  kSide,
  // F32 [width, segment rows]: a verify's save (StepGraph::saves) of a
  // segment's kept rows, which a commit step reads (BuildCommit): the
  // caller's (`component` and `what` name the save).
  kCommitRows,
  // Component-declared host-computed input (`host`: its HostInputSpec).
  kHostComputed,
  // A constant F32 [n, n] (`read` = n): the normalized Hadamard matrix
  // (HadamardValues), which a product hinted kHadamard multiplies by.
  kHadamard,
  // I32 [segment rows]: completed zero-based blocks, capped by
  // visible_capacity (real cells, excluding padding/dummy) and read (scored C).
  kVisibleCells,
};
struct StepInput {
  ir::TensorId tensor{};
  InputKind kind = InputKind::kTokens;
  std::uint8_t segment = ir::kJoinedSegment;
  CellKey cells{};
  std::uint64_t read = 0;  // kCellPositions' cells, kBlockPositions' blocks
  // kPositions and kBlockPositions: copies of the positions, one after
  // another (a multi-section rotation's; 1: plain).
  std::uint8_t sections = 1;
  std::uint16_t component = 0;  // kSideRows' source, kCommitRows' saver
  std::uint32_t layer = 0;      // kCommitRows' saver's layer
  std::string what{};           // kCommitRows' save
  HostInputSpec host{};
  std::uint32_t ratio = 0;
  bool overlap = false;
  std::uint64_t ring = 0;
  std::uint64_t visible_capacity = 0;  // kVisibleCells: logical real-cell capacity
};

struct SegmentRows {
  std::uint8_t slot = 0;
  std::uint32_t first_row = 0;
  std::uint32_t rows = 0;
  std::uint32_t past = 0;
  std::uint32_t read_cells = 0;
  std::uint32_t outputs = 0;
};

// A verify's save for its commit (components whose state rolls back by
// commit: Rollback::kCommit): a component's per-row values, F32 [width,
// rows], that the commit of the kept rows replays (BuildCommit). `segment`
// is the owner's, or kJoinedSegment for every owner's rows.
struct CommitSave {
  std::uint32_t layer = 0;
  std::uint16_t component = 0;
  std::string what;
  std::uint8_t segment = ir::kJoinedSegment;
  ir::TensorId tensor{};
  // I64 [the segment's rows]: 0 to rows - 1, the rows of the buffer a
  // lowering stores the save in (a kStoreCells input of a cell per row).
  ir::TensorId rows{};
};

// A mask a component reads (ComponentDef::masks, exact_masks).
struct MaskDecl {
  CellKey cells{};
  std::uint32_t window = 0;
  bool ahead = false;
  bool exact = false;
  auto operator<=>(const MaskDecl&) const = default;
};

class StepContext {
 public:
  StepContext(ir::Builder& b, std::vector<SegmentRows> segments, bool greedy);

  std::span<const SegmentRows> segments() const { return segments_; }
  std::uint32_t rows() const { return rows_; }
  std::uint32_t outputs() const { return outputs_; }
  // Whether the head picks each output row's token (every segment greedy).
  bool greedy() const { return greedy_; }
  // What the step is for (planner::Phase): a verify's fragments save what
  // their state's commit replays instead of rewriting fixed state.
  planner::Phase phase() const { return phase_; }
  void SetPhase(planner::Phase phase) { phase_ = phase; }
  // Real logical context, excluding padded/dummy compressed cells; zero
  // keeps the ordinary indexer. BuildStep validates this bounded option.
  void SetRoundedIndexerContext(std::uint32_t context) { rounded_indexer_context_ = context; }
  std::uint32_t rounded_indexer_context() const { return rounded_indexer_context_; }
  // The component nodes are attributed to (step inputs: kFrameworkComponent).
  void SetComponent(std::uint16_t component);
  ir::TensorId Tokens();
  ir::TensorId Positions();
  // The positions in `sections` copies, section-major: I32 [sections ·
  // rows] (a multi-section rotation's positions; 1: Positions()).
  ir::TensorId Positions(std::uint8_t sections);
  // Segment s's view of the positions, I32 [its rows].
  ir::TensorId SegmentPositions(std::size_t segment);
  ir::TensorId OutputRows();
  // The cell a fixed tensor's one row is stored at: I64 [1], 0.
  ir::TensorId StateRow();
  // Segment s's read cells' positions for tensors of `cells` (the input
  // its masks read): I32 [ReadCells].
  ir::TensorId HeldPositions(std::size_t segment, const CellKey& cells);
  // Exact visible prefix of zero-based positions-basis block cells. The logical
  // capacity excludes padded/dummy cells; read is the actual scored width.
  std::expected<ir::TensorId, std::string> VisibleCells(std::size_t segment, const CellKey& cells,
                                                        std::uint64_t read,
                                                        std::uint64_t visible_capacity);
  // Segment s's `blocks` blocks' first positions (kBlockPositions).
  ir::TensorId BlockPositions(std::size_t segment, const CellKey& cells, std::uint64_t blocks,
                              std::uint8_t sections);
  // A side table's rows, `count` a step row (kSideRows).
  ir::TensorId SideRows(std::uint16_t component, std::uint64_t count);
  // A verify's save of `tensor` for its commit (a step output), and a
  // commit's read of segment s's kept rows of it (kCommitRows).
  void Save(const Instance& in, std::string_view what, ir::TensorId tensor,
            std::uint8_t segment = ir::kJoinedSegment);
  ir::TensorId Saved(const Instance& in, std::string_view what, std::size_t segment,
                     std::uint64_t width);
  std::vector<CommitSave> TakeSaves() { return std::move(saves_); }
  // The rows' feature, F32 [width, rows].
  ir::TensorId Features(std::uint64_t width);
  // Request physical owner-column preservation for an ordinary product.
  // Finish proves its lineage; no caller supplies an expansion factor.
  void EnableOwnerProducts(bool enabled) { prove_owner_products_ = enabled; }
  void PreserveOwnerProduct(ir::TensorId result) {
    if (prove_owner_products_) owner_products_.push_back(result);
  }
  std::vector<ir::TensorId> TakeOwnerProducts() { return std::move(owner_products_); }
  // Segment s's store cells for tensors of `cells`.
  ir::TensorId StoreCells(std::size_t segment, const CellKey& cells);
  // The cells of a block tensor's (`cells.block`) first `blocks` blocks, I64
  // [blocks]: 0 to blocks - 1 (each block's keys stored at its own cell).
  ir::TensorId BlockCells(std::size_t segment, const CellKey& cells, std::uint64_t blocks);
  // Segment s's output rows' cells in tensors of `cells`.
  ir::TensorId OutputCells(std::size_t segment, const CellKey& cells);
  // Segment s's cells read of a tensor of `cells`: min(capacity, its
  // read_cells); a positions-basis block tensor's, the blocks its
  // positions complete at the read alignment (CellKey::positions).
  std::uint64_t ReadCells(std::size_t segment, const CellKey& cells) const;
  // Segment s's mask over a tensor of `cells` with `window` (0: causal;
  // `ahead`: ir::MaskKind::kAhead), F16 [read cells, Pad(rows,
  // kMaskRowAlign)], or exactly rows with `exact`; made by Declare.
  std::expected<ir::TensorId, std::string> Mask(std::size_t segment, const CellKey& cells,
                                                std::uint32_t window, bool ahead = false,
                                                bool exact = false) const;
  // Makes every mask a stage's components declared (masks of one segment
  // together, causal first, then by window, a kind's after the plain).
  using MaskKey = MaskDecl;
  void DeclareMasks(std::span<const MaskKey> masks);
  // Component-declared host-computed input (`spec`: its HostInputSpec).
  ir::TensorId HostInput(const HostInputSpec& spec, std::size_t segment = ir::kJoinedSegment,
                         const CellKey& cells = {});
  // The Hadamard matrix of `n`, F32 [n, n] (a constant).
  ir::TensorId Hadamard(std::uint64_t n);

  std::vector<StepInput> TakeInputs() { return std::move(inputs_); }

 private:
  ir::TensorId Input(InputKind kind, std::uint8_t segment, const format::TensorFormat& format,
                     const CellKey& cells, std::uint64_t read, std::string_view name,
                     const ir::InputParams& fact = {});
  ir::Builder& b_;
  std::vector<SegmentRows> segments_;
  std::uint32_t rows_ = 0;
  std::uint32_t outputs_ = 0;
  bool greedy_ = false;
  std::uint32_t rounded_indexer_context_ = 0;
  planner::Phase phase_ = planner::Phase::kPrefill;
  std::uint16_t component_ = 0;
  ir::TensorId tokens_{}, positions_{}, output_rows_{}, features_{}, state_row_{};
  std::map<std::tuple<std::size_t, CellKey, std::uint64_t, std::uint64_t>, ir::TensorId>
      visible_cells_;
  std::map<std::uint8_t, ir::TensorId> sectioned_;
  std::map<std::uint16_t, ir::TensorId> side_rows_;
  std::map<std::tuple<std::size_t, CellKey, std::uint64_t, std::uint8_t>, ir::TensorId> blocks_;
  std::vector<ir::TensorId> segment_positions_;
  std::map<std::pair<std::size_t, CellKey>, ir::TensorId> store_cells_;
  std::map<std::pair<std::size_t, CellKey>, ir::TensorId> output_cells_;
  std::map<std::tuple<std::size_t, CellKey, std::uint64_t>, ir::TensorId> cell_positions_;
  std::map<std::pair<std::size_t, MaskKey>, ir::TensorId> masks_;
  // Component-declared host-computed inputs by (spec, segment, cells).
  std::map<std::tuple<HostInputSpec, std::size_t, CellKey>, ir::TensorId> host_inputs_;
  std::map<std::uint64_t, ir::TensorId> hadamard_;
  std::map<std::tuple<std::size_t, CellKey, std::uint64_t>, ir::TensorId> block_cells_;
  std::map<std::uint8_t, ir::TensorId> save_rows_;  // by segment
  std::vector<StepInput> inputs_;
  std::vector<CommitSave> saves_;
  std::vector<ir::TensorId> owner_products_;
  bool prove_owner_products_ = false;
};

// --------------------------------------------------------------- the step

struct Bindings {
  // The stage's artifact's tensors.
  std::span<const BoundTensor> tensors{};
  // The stage's state layout (StageLayout's): each state leaf's cells.
  const model::StateLayoutDerived* state = nullptr;
  // Every stage's layout by stage index (null where not given): a
  // drafter's step reads its target's (its borrowed tensors' cells), a
  // target's step with outputs writes each tap reader's.
  std::span<const model::StateLayoutDerived* const> stages{};
  // Every bound artifact's tensors by artifact index: the weights a
  // binding names in another stage's artifact.
  std::span<const std::span<const BoundTensor>> artifacts{};
};

struct StepWeight {
  ir::TensorId tensor{};
  ir::WeightRef ref{};
  // The bound tensors holding its parts (one for a single-part format), of
  // the artifact of stage ref.stage.
  std::vector<std::pair<std::string, format::PartRole>> parts;
};
struct StepState {
  ir::TensorId tensor{};
  ir::StateRef ref{};
  // Its StateLayoutDerived item: of stage ref.stage's layout, or for a
  // borrowed tensor of the stage it reads (`source`: the tensor it reads).
  std::size_t item = 0;
  bool borrowed = false;     // a drafter's view of another stage's tensor: read only
  bool tap = false;          // another stage's tensor this step writes (target_writes)
  std::uint16_t source = 0;  // a borrowed tensor's stage
  // What the step does with it (Finish sets them from the graph): a node
  // reads it, a store writes it. A verify's saves and a target's writes
  // into a drafter's stage follow `written`.
  bool read = false;
  bool written = false;
};

struct StepGraph {
  ir::Graph graph;
  std::vector<StepInput> inputs;
  std::vector<StepWeight> weights;
  std::vector<StepState> state;
  ir::TensorId logits{};    // F32 [vocab, outputs], when the step has outputs
  ir::TensorId greedy{};    // I32 [outputs], when every segment is greedy
  ir::TensorId features{};  // F32 [feature width, outputs]: a drafter head's next feature
  // A verify's saves for its commit (outputs of the step).
  std::vector<CommitSave> saves{};
  std::vector<ir::TensorId> owner_products{};
};

// Refuses a step that stores into a borrowed tensor, or (`stage` writing
// another stage's tensor) into one that is not a tap it writes: every
// store into a state leaf (a cache store's cache, a fused RoPE store's)
// may write it. A store into an activation (a mask built by stores) is the
// step's own.
std::expected<void, std::string> CheckStateWrites(const StepGraph& step, std::uint16_t stage);
// The same over another graph of the step's leaves (the planned graph, its
// fused stores included): `tensors[primitive tensor]` its tensor there
// (planner/plan.h StepPlan::tensors; empty: the step's own graph).
std::expected<void, std::string> CheckStateWrites(const ir::Graph& graph,
                                                  std::span<const StepState> state,
                                                  std::span<const ir::TensorId> tensors,
                                                  std::uint16_t stage);
// Recomputes the read/write facts recorded by StepGraph::MarkAccess from the
// graph's actual state-root uses. The supplied state is never modified.
std::expected<void, std::string> CheckStateAccess(const ir::Graph& graph,
                                                  std::span<const StepState> state,
                                                  std::span<const ir::TensorId> tensors = {});

// Checks a planned graph against the already validated original declaration.
// Rewrites may eliminate reads (for example, a complete GDN history overwrite
// need not read the old history), but may not introduce reads or change writes.
// Root identity, complete state inventory and write authorization remain the
// caller's to check. Never validates source facts.
std::expected<void, std::string> CheckMappedStateAccess(
    const ir::Graph& graph, std::span<const StepState> state,
    std::span<const ir::TensorId> original_to_planned);

// The stage's components' fragments for `shape`, joined and verified. Its
// segments are in row order; each reads min(capacity, read_cells) cells of
// a state tensor, which must cover its rows' positions (past + rows, or the
// whole capacity; a frozen read's past), and a ring it wraps holds at
// least rows + window - 1 cells (every row's window survives the step's
// stores, which precede its attention; a frozen ring that has wrapped at
// least window - 1).
// A step with no outputs is state-only: its last layer writes only its
// state. Greedy segments must be all of them, each with an output (a
// drafter's head of every row picks each row's). A target's step with
// outputs writes the taps its drafters read (each segment's outputs within
// the tap's cells), which needs their layouts; and every target step runs
// the injection of each drafter that injects (ComponentDef::inject: a
// block drafter's KV from the target's features) after its own work,
// which needs that drafter's layout and its artifact's tensors.
// A verify (shape.phase) that writes its stage's state of Rollback::kCommit
// is refused: its commit writes that state (BuildCommit).
// With `narrow` off (a policy without the planner's narrow_final option,
// planner/pattern.h), a step with a head does not narrow: every row runs
// to the final norm and the head gathers its rows there (today's Gemma 4
// recipe without frontier_head).
// With `vocabulary` set (a tokenizer narrower than its head: RE-050), the
// stage's output head chooses only among that many rows, so a padding row
// is never a greedy choice; its logits keep every row (head.linear; a head
// of selected rows and a drafter's Markov head are unaffected).
struct StepOptions {
  bool narrow = true;
  std::uint64_t vocabulary = 0;  // 0: every row of the head
  // Internal compiler experiment, not execution qualification.
  bool owner_products = false;
  bool rounded_indexer = false;
  bool rounded_index_mask = false;
};
std::expected<StepGraph, std::string> BuildStep(const spec::ModelSpec& spec, std::uint16_t stage,
                                                const planner::StepShape& shape,
                                                const Bindings& bindings,
                                                const StepOptions& options = {});

// The commit of a verify's kept rows: each component with a commit
// (ComponentDef::commit) writes, from the verify's saves of each owner's
// kept rows (InputKind::kCommitRows inputs, which the caller binds to the
// first rows of the verify step's StepGraph::saves), the state those rows
// alone would have left. `kept`'s segments are the verify's owners, each
// its kept rows (1 to the verify's). Refused for a stage with nothing to
// commit.
std::expected<StepGraph, std::string> BuildCommit(const spec::ModelSpec& spec, std::uint16_t stage,
                                                  const planner::StepShape& kept,
                                                  const Bindings& bindings);

// One instance's fragment as a step of its own, for its tests and ports:
// its inputs (kStream, unless an embedding's or a side source's; and as
// `inputs` asks, the streams, carries and a side input), its weights in
// `formats` (by role, through at least the last role its shapes use; the
// others' entries unread), `cells` cells per state
// tensor of its contract, its masks, then the fragment; its outputs (the
// stream, the streams and the carry it returns, or a head's logits) are
// the step's.
struct ComponentInputs {
  bool residual = false;    // kResidual: F32 [width, streams, rows]
  std::uint64_t carry = 0;  // kCarry: F32 [carry, rows]; 0: none
  std::uint64_t side = 0;   // kSide: F32 [side, rows]; 0: none
  // A second carry (a Sinkhorn mix's combination), F32 [streams, streams,
  // rows]; 0: none.
  std::uint64_t carry2 = 0;
  // The side input as integers (I32 [side, rows]: a token-hash side
  // input's expert ids) rather than F32.
  bool side_ids = false;
  bool owner_products = false;
};
struct ComponentGraph {
  StepGraph step;
  ir::TensorId stream_in{};
  ir::TensorId stream_out{};
  ir::TensorId residual_in{};
  ir::TensorId residual_out{};
  ir::TensorId carry_in{};
  ir::TensorId carry_out{};
  ir::TensorId carry2_in{};
  ir::TensorId carry2_out{};
  ir::TensorId side_in{};
  std::vector<ir::TensorId> weights;  // by role; invalid where unused
};
std::expected<ComponentGraph, std::string> BuildComponent(
    const Instance& instance, const planner::StepShape& shape,
    std::span<const format::TensorFormat> formats, std::span<const std::uint64_t> cells,
    const ComponentInputs& inputs = {});

// An instance outside a spec, for tests: `ref` validated against its kind's
// definition, at `site` in a stage of `stage`.
std::expected<Instance, std::string> MakeInstance(const spec::ComponentRef& ref,
                                                  const StageFacts& stage, Site site,
                                                  std::uint32_t layer, std::uint16_t index);

// An input's values for `shape` (each segment's past) and the step's
// `tokens` (kTokens'), in the input's logical order; kStream's and
// kFeatures' are the caller's.
std::expected<void, std::string> EmitInputValues(const StepInput& input,
                                                 const planner::StepShape& shape,
                                                 std::span<const std::int32_t> tokens,
                                                 InputValueSink& into);
std::expected<std::vector<std::int64_t>, std::string> InputValues(
    const StepInput& input, const planner::StepShape& shape, std::span<const std::int32_t> tokens);

}  // namespace llmp::components

#endif  // LLMP_COMPONENTS_STEP_H_
