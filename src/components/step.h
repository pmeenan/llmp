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
// every layer's components in order, the final norm and the output head.
// A weight leaf is made once per bound tensor (a tied head reads the
// embedding's leaf, as today's builders do), its WeightRef the first
// component binding it. A state leaf is (stage, the layer's contract
// tensor, layer, segment): a layer's contract is its components' contracts
// concatenated in order.
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
// Model layer, after spec: the planner's step shape is a header-only type.

#ifndef LLMP_COMPONENTS_STEP_H_
#define LLMP_COMPONENTS_STEP_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "components/component.h"
#include "ir/graph.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "planner/step_shape.h"
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

// A tensor of the artifact a stage runs from, as the mapping sees it (a
// v0 artifact's: spec::SourceOfV0).
using BoundTensor = spec::SourceTensor;

// A stage's components with their parameters validated and their ordinals.
struct StageInstances {
  std::optional<Instance> embedding;
  std::vector<std::vector<Instance>> layers;
  std::optional<Instance> final_norm;
  std::optional<Instance> output;
};
std::expected<StageInstances, std::string> InstancesOf(const spec::ModelSpec& spec,
                                                       std::size_t stage);

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
// to a tensor of that name and part, in an encoding the role accepts and
// of the shape the role implies, and no binding of a role it does not use;
// a layer count and stream width within the catalog's bounds; one stream
// (no mixing component is in the catalog yet); and every read of another
// stage's state naming a state tensor of that stage. The first violation,
// with where it is.
std::expected<void, std::string> ValidateSpec(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors);
// A single-artifact spec.
std::expected<void, std::string> ValidateSpec(const spec::ModelSpec& spec,
                                              std::span<const BoundTensor> tensors);

// A stage's per-layer contracts (each layer's components' concatenated),
// for model::Layout; layers without state are left out.
std::expected<std::vector<std::pair<std::uint32_t, model::StateContract>>, std::string>
StageContracts(const spec::ModelSpec& spec, std::size_t stage);
std::expected<model::StateLayoutDerived, std::string> StageLayout(const spec::ModelSpec& spec,
                                                                  std::size_t stage,
                                                                  std::uint32_t context,
                                                                  std::uint32_t max_rows,
                                                                  std::uint64_t tensor_align = 256);

// The stage's batching capability: its components' own limits combined
// (the tightest of each), over a step of any segments up to kMaxSegments.
std::expected<model::Batching, std::string> StageBatching(const spec::ModelSpec& spec,
                                                          std::size_t stage);

// ------------------------------------------------------------ step context

// A state tensor's cells as a step indexes them: its capacity and whether
// it is a ring (position p in cell p mod capacity).
struct CellKey {
  std::uint64_t capacity = 0;
  bool ring = false;
  auto operator<=>(const CellKey&) const = default;
};

// The step's interface: an input leaf and what it holds.
enum class InputKind : std::uint8_t {
  kStream,      // F32 [width, rows]: a lone component's input stream (BuildComponent)
  kTokens,      // I32 [rows]: the step's token ids
  kPositions,   // I32 [rows]: each row's position (past + i in its segment)
  kOutputRows,  // I32 [outputs]: the head's rows (each segment's last `outputs`)
  kStoreCells,  // I64 [segment rows]: the cell each row is stored at
  // I32 [cells]: the position each read cell holds once the step's rows
  // are stored (a ring's latest below the segment's end), -1 for none.
  kCellPositions,
};
struct StepInput {
  ir::TensorId tensor{};
  InputKind kind = InputKind::kTokens;
  std::uint8_t segment = ir::kJoinedSegment;
  CellKey cells{};
  std::uint64_t read = 0;  // kCellPositions' cells
};

struct SegmentRows {
  std::uint8_t slot = 0;
  std::uint32_t first_row = 0;
  std::uint32_t rows = 0;
  std::uint32_t past = 0;
  std::uint32_t read_cells = 0;
  std::uint32_t outputs = 0;
};

class StepContext {
 public:
  StepContext(ir::Builder& b, std::vector<SegmentRows> segments, bool greedy);

  std::span<const SegmentRows> segments() const { return segments_; }
  std::uint32_t rows() const { return rows_; }
  std::uint32_t outputs() const { return outputs_; }
  // Whether the head picks each output row's token (every segment greedy).
  bool greedy() const { return greedy_; }
  // The component nodes are attributed to (step inputs: kFrameworkComponent).
  void SetComponent(std::uint16_t component);
  ir::TensorId Tokens();
  ir::TensorId Positions();
  ir::TensorId OutputRows();
  // Segment s's store cells for tensors of `cells`.
  ir::TensorId StoreCells(std::size_t segment, const CellKey& cells);
  // Segment s's cells read of a tensor of `cells`: min(capacity, its
  // read_cells).
  std::uint64_t ReadCells(std::size_t segment, const CellKey& cells) const;
  // Segment s's mask over a tensor of `cells` with `window` (0: causal),
  // F16 [read cells, Pad(rows, kMaskRowAlign)]; made by Declare.
  std::expected<ir::TensorId, std::string> Mask(std::size_t segment, const CellKey& cells,
                                                std::uint32_t window) const;
  // Makes every mask a stage's components declared (masks of one segment
  // together, causal first, then by window).
  void DeclareMasks(std::span<const std::pair<CellKey, std::uint32_t>> masks);

  std::vector<StepInput> TakeInputs() { return std::move(inputs_); }

 private:
  ir::TensorId Input(InputKind kind, std::uint8_t segment, const format::TensorFormat& format,
                     const CellKey& cells, std::uint64_t read, std::string_view name);
  ir::Builder& b_;
  std::vector<SegmentRows> segments_;
  std::uint32_t rows_ = 0;
  std::uint32_t outputs_ = 0;
  bool greedy_ = false;
  std::uint16_t component_ = 0;
  ir::TensorId tokens_{}, positions_{}, output_rows_{};
  std::vector<ir::TensorId> segment_positions_;
  std::map<std::pair<std::size_t, CellKey>, ir::TensorId> store_cells_;
  std::map<std::tuple<std::size_t, CellKey, std::uint64_t>, ir::TensorId> cell_positions_;
  std::map<std::tuple<std::size_t, CellKey, std::uint32_t>, ir::TensorId> masks_;
  std::vector<StepInput> inputs_;
};

// --------------------------------------------------------------- the step

struct Bindings {
  // The stage's artifact's tensors.
  std::span<const BoundTensor> tensors{};
  // The stage's state layout (StageLayout's): each state leaf's cells.
  const model::StateLayoutDerived* state = nullptr;
};

struct StepWeight {
  ir::TensorId tensor{};
  ir::WeightRef ref{};
  // The bound tensors holding its parts (one for a single-part format).
  std::vector<std::pair<std::string, format::PartRole>> parts;
};
struct StepState {
  ir::TensorId tensor{};
  ir::StateRef ref{};
  std::size_t item = 0;  // its StateLayoutDerived item
};

struct StepGraph {
  ir::Graph graph;
  std::vector<StepInput> inputs;
  std::vector<StepWeight> weights;
  std::vector<StepState> state;
  ir::TensorId logits{};  // F32 [vocab, outputs], when the step has outputs
  ir::TensorId greedy{};  // I32 [outputs], when every segment is greedy
};

// The stage's components' fragments for `shape`, joined and verified. Its
// segments are in row order; each reads min(capacity, read_cells) cells of
// a state tensor, which must cover its rows' positions (past + rows, or the
// whole capacity), and a ring it wraps holds at least rows + window - 1
// cells (every row's window survives the step's stores, which precede its
// attention). A step with no outputs is state-only: its last layer
// writes only its state. Greedy segments must be all of them, each with
// one output.
std::expected<StepGraph, std::string> BuildStep(const spec::ModelSpec& spec, std::uint16_t stage,
                                                const planner::StepShape& shape,
                                                const Bindings& bindings);

// One instance's fragment as a step of its own, for its tests and ports:
// its input stream an input (kStream; none for an embedding), its weights
// in `formats` (by role; those its shapes use), `cells` cells per state
// tensor of its contract, its masks, then the fragment; the output stream
// (or a head's logits) is the step's output.
struct ComponentGraph {
  StepGraph step;
  ir::TensorId stream_in{};
  ir::TensorId stream_out{};
  std::vector<ir::TensorId> weights;  // by role; invalid where unused
};
std::expected<ComponentGraph, std::string> BuildComponent(
    const Instance& instance, const planner::StepShape& shape,
    std::span<const format::TensorFormat> formats, std::span<const std::uint64_t> cells);

// An instance outside a spec, for tests: `ref` validated against its kind's
// definition, at `site` in a stage of `stage`.
std::expected<Instance, std::string> MakeInstance(const spec::ComponentRef& ref,
                                                  const StageFacts& stage, Site site,
                                                  std::uint32_t layer, std::uint16_t index);

// An input's values for `shape` (each segment's past) and the step's
// `tokens` (kTokens'), in the input's logical order.
std::expected<std::vector<std::int64_t>, std::string> InputValues(
    const StepInput& input, const planner::StepShape& shape, std::span<const std::int32_t> tokens);

}  // namespace llmp::components

#endif  // LLMP_COMPONENTS_STEP_H_
