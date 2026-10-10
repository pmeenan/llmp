// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The component catalog (D-107; docs/engine-components.md#components): a
// building block of a model declared once, as one ComponentDef in a
// compiled table. A model spec (spec/spec.h) names components by kind and
// version; only this catalog gives them meaning. A definition holds
//
//   - its parameters (ParamSpec: type, range, choices, default), which a
//     spec's values are validated against (Params);
//   - its weight roles (WeightRole) with the encodings each accepts, and
//     per instance the roles in use with the shapes their parameters imply
//     (RoleShape);
//   - its state contract per instance (model/state_contract.h), from which
//     the engine's slot lifecycle derives regions, spill and restore;
//   - its batching capability as data: which of its ops join rows across
//     requests and which stay one owner's (BatchClass), and the limits it
//     places on a step (model::Batching);
//   - how it splits across devices, as data: the roles and state tensors
//     each split divides and what it communicates per step (SplitOption),
//     for the partitioner;
//   - the fusion patterns it takes part in, by name (the planner's);
//   - `emit`: the primitive IR fragment for a step's shape (a Fragment),
//     which the planner's patterns later rewrite;
//   - `reference`: a scalar CPU implementation written independently of
//     `emit` (RefIO), the oracle `emit` and every backend port are tested
//     against.
//
// A fragment emits the primitive ops in the order the M3 graph builders
// recorded them (GGML's execution order), so a lowering reproduces those
// graphs node for node: docs/engine-components.md's
// "Initial catalog" lists the rules and what a fragment cannot express.
//
// Model layer, after spec: no planner, kernel or engine type.

#ifndef LLMP_COMPONENTS_COMPONENT_H_
#define LLMP_COMPONENTS_COMPONENT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/batching.h"
#include "model/state_contract.h"
#include "spec/spec.h"

namespace llmp::components {

struct ComponentDef;
class StepContext;

// ---------------------------------------------------------------- parameters

enum class ParamType : std::uint8_t { kInt, kReal, kBool, kName, kInts, kReals };
std::string_view ParamTypeName(ParamType type);

struct ParamSpec {
  std::string_view name{};
  ParamType type = ParamType::kInt;
  bool required = true;
  // Numbers (and every member of a list): within [least, most].
  double least = 0.0;
  double most = 0.0;
  // kName: the names it takes (none listed: any name, one the spec gives
  // something else, such as a tap a read names); an absent optional name
  // is choices[0].
  std::span<const std::string_view> choices{};
  // An absent optional number's or flag's value (a flag: nonzero is true).
  double fallback = 0.0;
};

// A component instance's parameters, validated against its definition:
// every parameter the definition lists, absent optional ones at their
// defaults. Lookups by a name the definition does not list are a bug
// (base::Check).
class Params {
 public:
  Params() = default;
  // Refused for a parameter the definition does not list, of another type,
  // out of range or not among the choices, or a required one absent.
  static std::expected<Params, std::string> Of(std::span<const ParamSpec> specs,
                                               const spec::ComponentRef& ref);

  std::int64_t Int(std::string_view name) const;
  double Real(std::string_view name) const;
  bool Flag(std::string_view name) const;
  std::string_view Name(std::string_view name) const;
  std::span<const std::int64_t> Ints(std::string_view name) const;
  std::span<const double> Reals(std::string_view name) const;
  // Whether the spec gave it (an optional one may be absent).
  bool Given(std::string_view name) const;

 private:
  const spec::ParamValue& At(std::string_view name) const;
  std::vector<spec::Param> values_;  // sorted by name
  std::vector<std::string> given_;
};

// --------------------------------------------------------------- instances

// Where in a stage a component sits: besides the embedding, the layers and
// the heads, a side input's source (spec.h SideInput: a table the layers
// read beside the stream) and the stream's mixing (spec.h ResidualStream:
// the hyper-connection streams made from the embedding).
enum class Site : std::uint8_t { kEmbedding, kLayer, kFinalNorm, kOutput, kSide, kMixing };

// What a component reads of its stage besides its own parameters.
struct StageFacts {
  std::uint64_t width = 0;          // the residual stream's
  std::uint32_t context = 0;        // the trained context (a rotation's original context)
  std::optional<double> softcap{};  // the heads' final logit softcap (the output's)
  std::uint32_t streams = 1;        // the residual's streams (hyper-connections: several)
  // The folds the stage's tensors carry (spec.h Stage::folds, its own
  // artifact's): a component whose arithmetic assumes one refuses a stage
  // without it. A binding to another stage's tensor carries that stage's
  // (spec.h HoldingStage).
  std::span<const std::string> folds{};
};

// One component of a stage, ready to use: its definition, validated
// parameters and place. `index` is its ordinal in the stage (OrdinalOf in
// step.h): what ir::WeightRef::component and a node's component name.
struct Instance {
  const ComponentDef* def = nullptr;
  const spec::ComponentRef* ref = nullptr;
  Params params;
  StageFacts stage;
  Site site = Site::kLayer;
  std::uint32_t layer = 0;  // kLayer's; 0 elsewhere
  std::uint16_t index = 0;
};

// ------------------------------------------------------------------ weights

struct WeightRole {
  std::string_view name{};
  // The encodings it accepts. Empty: any valid format the format layer
  // describes (a product's or a lookup's operand reads any of them through
  // its decoder).
  std::span<const format::Encoding> encodings{};
  // A table read by rows on demand (D-035): never resident whole; before a
  // step the engine reads the rows its lookup names into row slots, which
  // the step's lookup then reads (engine/ple_rows.h: Qwen3.8's n-gram
  // table). Its lookup's indices are a step input the engine computes.
  bool rows_on_demand = false;
};

// A role in use for an instance and the logical shape its parameters
// imply; an extent of 0 is free (a vocabulary the parameters do not name).
struct RoleShape {
  std::uint16_t role = 0;  // index into the definition's roles
  std::array<std::uint64_t, format::kMaxDims> ne{1, 1, 1, 1};
  std::uint8_t rank = 1;
};

// -------------------------------------------------------- batching, splits

// Whether an op of the component joins rows across requests or stays one
// owner's (attention over one request's cache, a cache store).
struct BatchClass {
  ir::Op op = ir::Op::kMatMul;
  ir::Batch batch = ir::Batch::kJoined;
};

enum class SplitKind : std::uint8_t { kNone, kHeads, kColumns, kRows, kExperts, kWholeLayer };
std::string_view SplitKindName(SplitKind kind);

// A weight role or a state tensor (its index in the instance's contract)
// split along logical dim `dim` into equal parts, part r on the group's
// r-th device. A role or tensor a split does not list is whole on every
// device of the group.
struct RoleSplit {
  std::uint16_t role = 0;
  std::uint8_t dim = 0;
};
struct StateSplit {
  std::uint16_t tensor = 0;
  std::uint8_t dim = 0;
};

// What a split's traffic depends on: the instance, its roles' bound shapes
// (by role; a role not in use has rank 0), and the step.
struct SplitStep {
  const Instance* instance = nullptr;
  std::span<const format::Shape> weights{};
  std::uint32_t rows = 0;     // the step's rows
  std::uint32_t outputs = 0;  // the head's rows
  std::uint32_t ways = 1;
};

// A way to split an instance across a group of `ways` devices, as data: the
// roles and state tensors it splits and along which dims (the planner's
// partitioner propagates the split through the fragment's ops and inserts
// the collectives, planner/partition.h), the collective that joins the
// parts (a partial sum's all-reduce, a column split's gather), the units
// `ways` must divide (whole KV heads; null: only the split roles' shapes
// and encodings bound it), and the bytes of the tensor that collective
// joins per step (the planner models what each link carries from it).
struct SplitOption {
  SplitKind kind = SplitKind::kNone;
  ir::Op collective = ir::Op::kAllReduce;
  std::uint32_t (*units)(const Instance& instance) = nullptr;
  std::uint64_t (*bytes_per_step)(const SplitStep& step) = nullptr;
  std::span<const RoleSplit> weights{};
  std::span<const StateSplit> state{};
};

// ---------------------------------------------------------------- fragments

// What `emit` is given: the builder (its component already set, nodes
// joined; a fragment leaves it joined), the instance, the step's shared
// inputs, the stream it reads, its weight leaves by role and its state
// leaves by segment and contract tensor. `stream_needed` is false when nothing
// reads its output stream (a state-only step's last layer): a stateful
// component then emits only its state writes, a stateless one nothing.
struct Fragment {
  ir::Builder& b;
  const Instance& instance;
  StepContext& step;
  ir::TensorId stream{};
  // A role's weight leaf, made on first use (a role not in use is a bug).
  std::function<ir::TensorId(std::uint16_t role)> weight;
  // state[segment * tensors + tensor]
  std::span<const ir::TensorId> state{};
  std::size_t state_tensors = 0;
  bool stream_needed = true;
  // The step's head reads only its output rows (step.OutputRows()) of what
  // follows: `narrow` asks the stage's last component that mixes rows (its
  // ComponentDef::narrows) to return only those rows of its output stream,
  // gathered after the last of its own ops that mixes rows; `narrowed`
  // tells each component after it that its input stream holds only those
  // rows (the head then gathers none).
  bool narrow = false;
  bool narrowed = false;
  // A stage of several streams (spec.h ResidualStream, hyper-connections):
  // the streams F32 [width, streams, rows] as this component finds them.
  // A block between a mix and its combine reads `stream` (the mix's
  // single stream) and returns its output there; the mix (hc.pre) reads
  // the streams, a combine (hc.post) or a side input's consumer writes
  // them (FragmentOut::residual). Invalid in a stage of one stream.
  ir::TensorId residual{};
  // The latest carry a component of its layer handed on (FragmentOut::
  // carry): a hyper-connection mix's normalized streams, which the
  // combine after its block reads.
  ir::TensorId carry{};
  // The stage's side inputs by name (made after the embedding, in name
  // order); invalid for a name the stage has not.
  std::function<ir::TensorId(std::string_view name)> side{};

  ir::TensorId Weight(std::uint16_t role) const { return weight(role); }
  ir::TensorId State(std::size_t segment, std::size_t tensor) const {
    return state[(segment * state_tensors) + tensor];
  }
  ir::TensorId Side(std::string_view name) const { return side ? side(name) : ir::TensorId{}; }
};

struct FragmentOut {
  ir::TensorId stream{};  // the output stream (the input when unchanged)
  // An output head's results: its logits, each row's token when greedy, and
  // a drafter's next feature (F32 [feature width, outputs], its rows
  // projected back to its target's width, or its streams before the
  // head's mix: an MTP drafter's).
  ir::TensorId logits{};
  ir::TensorId greedy{};
  ir::TensorId features{};
  // The streams it wrote (a stage of several streams; invalid: unchanged).
  ir::TensorId residual{};
  // What the next component of its layer reads (Fragment::carry).
  ir::TensorId carry{};
  // It gathered the step's output rows itself (a final norm that mixes the
  // streams after gathering them): what follows reads only those rows.
  bool narrowed = false;
};

// ---------------------------------------------------------------- reference

// One segment of a reference call, in row order.
struct RefSegment {
  std::uint32_t rows = 0;
  std::uint32_t past = 0;
  std::uint32_t read_cells = 0;  // the cells its attention reads (before the capacity cap)
  std::uint32_t outputs = 0;     // its last `outputs` rows are the head's
};

// A reference call's host buffers. Values are in logical order, ne[0]
// innermost. `stream` is [width, rows] in and out (an embedding's out
// only); `features` [feature width, rows] the feature a drafter's
// embedding reads; an output head writes `logits` [vocab, outputs], when
// `greedy` `tokens_out`, and a drafter's head `features_out` [feature
// width, outputs]. Weights are decoded to F32 by role (empty where not in
// use), with their shapes. State is per segment and contract tensor: every
// cell's values, `state_cells[tensor]` cells, read and written in the
// contract's encoding (each value stored rounded to it).
struct RefIO {
  const Instance* instance = nullptr;
  std::vector<RefSegment> segments{};
  std::vector<std::int32_t> tokens{};  // [rows], an embedding's
  std::vector<double> stream{};
  // A stage of several streams: the streams [width, streams, rows] in and
  // out, the carry in and out (Fragment::carry), and the side input a
  // consumer reads ([its width, rows]) or a source's table rows (its
  // lookup's indices, as the step's input holds them).
  std::vector<double> residual{};
  std::vector<double> carry{};
  std::vector<double> side{};
  std::vector<std::int64_t> side_rows{};
  std::vector<double> features{};
  std::vector<std::vector<float>> weights{};
  std::vector<format::Shape> weight_shapes{};
  std::vector<std::vector<std::vector<double>>> state{};  // [segment][tensor]
  std::vector<std::uint64_t> state_cells{};               // by tensor
  bool greedy = false;
  std::vector<double> logits{};
  std::vector<std::int32_t> tokens_out{};
  std::vector<double> features_out{};

  std::uint32_t rows() const {
    std::uint32_t n = 0;
    for (const RefSegment& s : segments) n += s.rows;
    return n;
  }
};

// --------------------------------------------------------------- definition

struct ComponentDef {
  std::string_view kind{};  // "attention.dense"
  std::uint16_t version = 1;
  // Each output row from its own input row alone (no attention or
  // recurrence) and no state: what follows a narrowing (Fragment::narrow)
  // may run at the output rows.
  bool row_local = false;
  // It gathers its output stream's output rows when asked (Fragment::narrow).
  bool narrows = false;
  // It makes a side input (a spec SideInput's source, Site::kSide): its
  // output stream is the side input, [its width, rows].
  bool side_source = false;
  // It makes a stage's several streams from the embedding's output (a spec
  // ResidualStream's mixing, Site::kMixing): FragmentOut::residual.
  bool mixing = false;
  // The side input an instance reads (a layer component's), by name; null
  // or empty: none.
  std::string_view (*side_input)(const Instance& instance) = nullptr;
  std::span<const ParamSpec> params{};
  std::span<const WeightRole> roles{};
  // The roles an instance uses and their shapes, or why its parameters do
  // not make a valid instance (heads not a multiple of the KV heads).
  std::expected<std::vector<RoleShape>, std::string> (*shapes)(const Instance& instance) = nullptr;
  // Its per-request state; null: none.
  model::StateContract (*state)(const Instance& instance) = nullptr;
  std::span<const BatchClass> batch_classes{};
  // The limits it places on a step's joins; null: none of its own.
  model::Batching (*batching)(const Instance& instance) = nullptr;
  std::span<const SplitOption> splits{};
  std::span<const std::string_view> patterns{};
  // Masks it reads, as (state tensor, window) pairs: made once per step
  // and segment before the first layer (StepContext::Mask).
  std::vector<std::pair<std::uint16_t, std::uint32_t>> (*masks)(const Instance& instance) = nullptr;
  // An output head: whether an instance reads only the step's output rows;
  // null: it does. A head of every row (a drafter's) has none to narrow to.
  bool (*gathers)(const Instance& instance) = nullptr;
  std::expected<FragmentOut, std::string> (*emit)(Fragment& fragment) = nullptr;
  std::expected<void, std::string> (*reference)(RefIO& io) = nullptr;
  // A side input's source (side_source): the rows of its table the step
  // row at `position` reads (InputKind::kSideRows: `side_count` of them),
  // from the token history through that position, which the engine
  // computes before the step; refused for a position past the history.
  std::expected<void, std::string> (*side_rows)(const Instance& instance,
                                                std::span<const std::int32_t> history,
                                                std::uint32_t position,
                                                std::span<std::int32_t> out) = nullptr;
  std::uint64_t (*side_count)(const Instance& instance) = nullptr;
  // State that rolls back by commit (model::Rollback::kCommit): a verify
  // step's fragment saves its rows' values (StepContext::Save) instead of
  // writing that state, and this, given those saves of each owner's kept
  // rows (StepContext::Saved), writes the state the kept rows alone would
  // have left (BuildCommit). Null: none.
  std::expected<void, std::string> (*commit)(Fragment& fragment) = nullptr;
};

// The compiled catalog, in kind order, and a lookup.
std::span<const ComponentDef* const> Catalog();
const ComponentDef* Find(std::string_view kind, std::uint16_t version);

// A definition's role index by name, if it has one.
std::optional<std::uint16_t> RoleIndex(const ComponentDef& def, std::string_view name);

}  // namespace llmp::components

#endif  // LLMP_COMPONENTS_COMPONENT_H_
