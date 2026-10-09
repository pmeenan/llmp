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
//   - how it splits across devices and what each split communicates per
//     step (SplitOption), for the partitioner;
//   - the fusion patterns it takes part in, by name (the planner's);
//   - `emit`: the primitive IR fragment for a step's shape (a Fragment),
//     which the planner's patterns later rewrite;
//   - `reference`: a scalar CPU implementation written independently of
//     `emit` (RefIO), the oracle `emit` and every backend port are tested
//     against.
//
// A fragment emits the primitive ops in the order today's graph builders
// record them (kernels/ggml/gemma2_graph.cc's GraphOrder), so a lowering
// can reproduce those graphs node for node: docs/engine-components.md's
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
  // kName: the names it takes; an absent optional name is choices[0].
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

// Where in a stage a component sits.
enum class Site : std::uint8_t { kEmbedding, kLayer, kFinalNorm, kOutput };

// What a component reads of its stage besides its own parameters.
struct StageFacts {
  std::uint64_t width = 0;          // the residual stream's
  std::uint32_t context = 0;        // the trained context (a rotation's original context)
  std::optional<double> softcap{};  // the heads' final logit softcap (the output's)
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

// A way to split an instance across `ways` devices: the most ways its
// parameters allow and the bytes one device sends per step of `rows` rows
// through `collective` (a partial sum's all-reduce, a column split's
// gather).
struct SplitOption {
  SplitKind kind = SplitKind::kNone;
  ir::Op collective = ir::Op::kAllReduce;
  std::uint32_t (*max_ways)(const Instance& instance) = nullptr;
  std::uint64_t (*bytes_per_step)(const Instance& instance, std::uint32_t rows,
                                  std::uint32_t ways) = nullptr;
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

  ir::TensorId Weight(std::uint16_t role) const { return weight(role); }
  ir::TensorId State(std::size_t segment, std::size_t tensor) const {
    return state[(segment * state_tensors) + tensor];
  }
};

struct FragmentOut {
  ir::TensorId stream{};  // the output stream (the input when unchanged)
  // An output head's results.
  ir::TensorId logits{};
  ir::TensorId greedy{};
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
// only); an output head writes `logits` [vocab, outputs] and, when
// `greedy`, `tokens_out`. Weights are decoded to F32 by role (empty where
// not in use), with their shapes. State is per segment and contract
// tensor: every cell's values, `state_cells[tensor]` cells, read and
// written in the contract's encoding (each value stored rounded to it).
struct RefIO {
  const Instance* instance = nullptr;
  std::vector<RefSegment> segments{};
  std::vector<std::int32_t> tokens{};  // [rows], an embedding's
  std::vector<double> stream{};
  std::vector<std::vector<float>> weights{};
  std::vector<format::Shape> weight_shapes{};
  std::vector<std::vector<std::vector<double>>> state{};  // [segment][tensor]
  std::vector<std::uint64_t> state_cells{};               // by tensor
  bool greedy = false;
  std::vector<double> logits{};
  std::vector<std::int32_t> tokens_out{};

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
  std::expected<FragmentOut, std::string> (*emit)(Fragment& fragment) = nullptr;
  std::expected<void, std::string> (*reference)(RefIO& io) = nullptr;
};

// The compiled catalog, in kind order, and a lookup.
std::span<const ComponentDef* const> Catalog();
const ComponentDef* Find(std::string_view kind, std::uint16_t version);

// A definition's role index by name, if it has one.
std::optional<std::uint16_t> RoleIndex(const ComponentDef& def, std::string_view name);

}  // namespace llmp::components

#endif  // LLMP_COMPONENTS_COMPONENT_H_
