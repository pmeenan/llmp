// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Partitioning (D-019, D-107; docs/engine-components.md#partitioning):
// tensor, pipeline and expert parallelism with any split count, mixed,
// over a described topology of any number of devices.
//
// Assign chooses where a stage's components run. A pipeline cuts the
// layers into contiguous ranges, each on its own group of devices (a
// pipeline stage); the embedding, side sources and stream mixing run with
// the first range, the final norm and the output head with the last.
// Within a group, every declared feasible split competes with whole replicated
// execution under the described execution-cost objective. Traffic-only mode
// retains layer splits and chooses the least communication. Weight/state
// capacity is checked jointly across the stage, deduplicating bound weights.
// The exact bounded search minimizes the explicitly weighted prefill/decode
// request latency; it does not model pipelined multi-request throughput.
// Unknown device performance is represented by explicit traffic-only mode.
// Execution work uses the actual local primitive graphs, summing each node's
// max(read/write bytes / bandwidth, operation units / rate) in component order
// and taking the slowest rank per component, then adding ordered communication.
// Shared framework dependencies are conservatively charged per consumer; no
// fusion, shared mask reuse, transfer concurrency or pipeline overlap is assumed.
// Execution mode requires described sustainable rates and provenance, and
// refuses work it cannot price. No device rates are assumed by the application.
// AssignPipeline evaluates a configured pipeline under the same objective.
//
// The traffic model: a collective over a group runs as a ring in the
// group's order (an all-reduce 2(g - 1) steps of 1/g of the tensor, a
// gather g - 1), each step as slow as its slowest link (bytes over
// bandwidth plus latency); a tensor that leaves its group is sent whole to
// each device that needs it from the holder with the cheapest link,
// devices that have received it holding it too (a stream crosses a slow
// link once, then spreads over the fast ones). Memory counts each device's
// weights (a tensor once however many components bind it, a split part as
// its own format's bytes) and the state its components keep for the
// workload's slots at its context, excluding activations, workspace and
// graphs, which the device's budget must leave room for. Side outputs
// transfer lazily to groups containing their readers, reusing every earlier
// holder; their sizes come from the source component's one-row graph.
//
// Partition rewrites a step's primitive graph (components/step.h's
// BuildStep, before patterns) into one graph per device. Each node runs on
// its component's devices; a split propagates from the roles and state
// tensors its option divides through the fragment's ops (views, reshapes,
// permutes, elementwise ops, norms over unsplit rows, rotations, products,
// stores, attention and softmax by heads, routed products; not ALiBi,
// whose slopes follow the global head, nor a mask broadcast over heads
// that each part's heads would read differently), and collectives join the
// parts where a consumer needs them whole: an all-reduce of partial sums (a
// row-parallel product's) where something nonlinear reads them, a gather
// of a split tensor, a send and receive where a tensor crosses to other
// devices (a pipeline cut). Expert parallelism keeps the routing whole on
// every device: each device runs its own experts' products for every
// routed slot (ids remapped to its own through a constant table, slots of
// other experts computing with its first expert) and literal selections
// zero foreign products and weights and neutralize foreign scales before its sum,
// so its sum is a partial sum the all-reduce completes (each device runs
// every routed slot's products: expert parallelism divides the experts'
// memory and weight reads, not the routed compute); dispatch needs no
// all-to-all while the routed tokens are whole on every device (an
// all-to-all dispatch pays only once attention splits by rows, which no
// component declares). An op a split cannot pass through gathers its
// sources and runs whole on every device of the group (correct, and
// counted in the traffic, which a test holds against the model); a store
// into a split cache, or an expert-local value, that cannot pass is
// refused. Inputs, constants and what is made only from them (masks,
// positions' views) are made on every device that reads them.
//
// Model layer, after components: no kernel, engine or vendor type.

#ifndef LLMP_PLANNER_PARTITION_H_
#define LLMP_PLANNER_PARTITION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "components/component.h"
#include "components/step.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "spec/spec.h"

namespace llmp::planner {

inline constexpr std::size_t kMaxDevices = 16;

// ------------------------------------------------------------------ topology

// Sustainable rates described by the topology owner, not inferred device peaks.
// MatMul, MatMulId and Attention use FLOPs/s; other ops use output elements/s
// for that exact typed operation. Rates must conservatively cover the workload
// shapes under consideration. Unary/GLU parameters remain part of the key.
struct OperationRate {
  ir::OpKey op{};
  std::array<std::byte, ir::kMaxParamBytes> params{};
  format::Encoding encoding = format::Encoding::kF32;
  ir::Accumulate accumulate = ir::Accumulate::kDefault;
  double units_per_s = 0;
  bool operator==(const OperationRate&) const = default;
};
struct DevicePerformance {
  double memory_bytes_per_s = 0;
  std::vector<OperationRate> rates{};
  std::string provenance{};
  bool operator==(const DevicePerformance&) const = default;
};
enum class CostModel : std::uint8_t { kTraffic, kExecution };

// A device and the bytes it gives the stage's weights and state.
struct Device {
  std::uint32_t id = 0;
  std::uint64_t memory = 0;
  std::optional<DevicePerformance> performance{};
};
// A link between two devices, both directions.
struct Link {
  std::uint32_t a = 0;
  std::uint32_t b = 0;
  double bytes_per_s = 0.0;
  double latency_s = 0.0;
};
// Configured or discovered (D-019); never a constant of the application.
struct Topology {
  std::vector<Device> devices{};
  std::vector<Link> links{};
  CostModel cost_model = CostModel::kTraffic;
};
// Ids unique, at most kMaxDevices devices, every link between two known
// distinct devices, at most one a pair, a positive bandwidth and a
// latency of at least zero.
std::expected<void, std::string> Validate(const Topology& topology);

// The steps the placement is chosen for, and the state it holds.
struct Workload {
  std::uint32_t prefill_rows = 512;  // a prefill step's rows (one output)
  std::uint32_t decode_rows = 1;     // a decode step's rows (an output each)
  double decode_share = 0.5;         // of the steps, in [0, 1]
  std::uint32_t context = 4096;      // the state's context
  std::uint32_t slots = 1;           // the requests whose state it keeps
};

// Per step: the bytes every link carries in total, and the modelled time.
struct Traffic {
  std::uint64_t bytes = 0;
  double seconds = 0.0;
};

// ---------------------------------------------------------------- the model

// A role's bound tensor: its name (one tensor bound by several roles is
// held once) and format; an empty name for a role not in use.
struct RoleTensor {
  std::string name{};
  format::TensorFormat format{};
};
// A stage's components and their bound weights (by ordinal, then role).
// It points into the spec it was made from (components::Instance::ref).
struct StageModel {
  std::uint16_t stage = 0;
  components::StageInstances instances{};
  std::vector<std::vector<RoleTensor>> roles{};
  // Reader layers for each side source, in instances.side order.
  std::vector<std::vector<std::uint32_t>> side_layers{};
  // Borrowed like Instance::ref; gives side-input names and actual step graphs
  // for execution work. A manually constructed model uses standalone fragments.
  const spec::ModelSpec* source = nullptr;
  std::span<const components::BoundTensor> tensors{};
};
std::expected<StageModel, std::string> ModelOf(const spec::ModelSpec& spec, std::uint16_t stage,
                                               std::span<const components::BoundTensor> tensors);

// -------------------------------------------------------------- assignment

inline constexpr std::uint8_t kWhole = 0xFF;

// Where a component runs: split over `devices` (rank order) by its
// option, or (kWhole) whole on each of them.
struct ComponentPlacement {
  std::uint16_t component = 0;  // its ordinal (components::Instance::index)
  const components::ComponentDef* def = nullptr;
  components::SplitKind split = components::SplitKind::kNone;
  std::uint8_t option = kWhole;  // index into def->splits
  // Its first tensor in its layer's contract (ir::StateRef::tensor).
  std::uint16_t state_first = 0;
  std::vector<std::uint32_t> devices{};
};

// A weight leaf a component reads under another component's reference
// (a tied head reads the embedding's leaf), and its role there.
struct WeightAlias {
  std::uint16_t component = 0;
  ir::WeightRef leaf{};
  std::uint16_t role = 0;
};

struct Assignment {
  Topology topology{};
  std::uint16_t stage = 0;
  // The pipeline stages' devices (ids, rank order), and each layer's stage.
  std::vector<std::vector<std::uint32_t>> stages{};
  std::vector<std::uint32_t> stage_of_layer{};
  std::optional<ComponentPlacement> embedding{};
  std::vector<ComponentPlacement> side{};
  std::optional<ComponentPlacement> mixing{};
  std::vector<std::vector<ComponentPlacement>> per_layer{};
  std::optional<ComponentPlacement> final_norm{};
  std::optional<ComponentPlacement> output{};
  std::vector<WeightAlias> aliases{};
  // Modelled per step: a prefill step's and a decode step's traffic, and
  // the workload's mix of them.
  Traffic prefill{};
  Traffic decode{};
  double modelled_seconds_per_step = 0.0;
  double modelled_bytes_per_step = 0.0;
  // Sequential per-component roofline time, separate from communication.
  // Traffic-only mode leaves these absent, never a claimed zero runtime cost.
  std::optional<double> prefill_execution_seconds{};
  std::optional<double> decode_execution_seconds{};
  // Each device's weights and state (devices the assignment uses).
  std::vector<std::pair<std::uint32_t, std::uint64_t>> memory{};

  const ComponentPlacement* Find(std::uint16_t component) const;
};

// A pipeline given: its stages' devices and each stage's first layer
// (the first 0, ascending).
struct Pipeline {
  std::vector<std::vector<std::uint32_t>> stages{};
  std::vector<std::uint32_t> first_layer{};
};

// The pipeline's assignment under the rules above, or why it is not one
// (memory, a group's ring without a link, devices unknown or repeated).
std::expected<Assignment, std::string> AssignPipeline(const StageModel& model,
                                                      const Topology& topology,
                                                      const Workload& workload,
                                                      const Pipeline& pipeline);
// The least-cost memory-feasible assignment, or why there is none.
std::expected<Assignment, std::string> Assign(const StageModel& model, const Topology& topology,
                                              const Workload& workload);
// A spec's target stage (stage 0) bound to its artifact's tensors.
std::expected<Assignment, std::string> Assign(const spec::ModelSpec& spec,
                                              std::span<const components::BoundTensor> tensors,
                                              const Topology& topology, const Workload& workload);

// The modelled traffic of an assignment's step of `rows` rows and
// `outputs` output rows.
Traffic ModelTraffic(const StageModel& model, const Assignment& assignment, std::uint32_t rows,
                     std::uint32_t outputs);

// ---------------------------------------------------------------- partition

// Which part of a leaf a device binds: part `part` of `parts` along `dim`
// (parts 1: the whole leaf).
struct Slice {
  std::uint8_t dim = 0;
  std::uint32_t part = 0;
  std::uint32_t parts = 1;
};
struct LeafBinding {
  ir::TensorId local{};
  ir::TensorId source{};  // the primitive graph's leaf
  Slice slice{};
};
// A constant the partition made (expert parallelism's tables), its values
// in logical order: integers for an I32 constant, else floats.
struct ConstantValues {
  ir::TensorId local{};
  std::vector<std::int64_t> ints{};
  std::vector<float> values{};
};
struct DeviceGraph {
  std::uint32_t device = 0;
  ir::Graph graph;
  std::vector<LeafBinding> leaves{};
  std::vector<ConstantValues> constants{};
};
// A collective's device group (ids; a member's rank is its position).
struct CollectiveGroup {
  std::uint32_t group = 0;
  std::vector<std::uint32_t> devices{};
};
// Where a tensor of the primitive graph is whole.
struct PlacedTensor {
  ir::TensorId source{};
  std::uint32_t device = 0;
  ir::TensorId local{};
};
struct Partitioned {
  std::vector<DeviceGraph> devices{};  // the devices with nodes, in topology order
  std::vector<CollectiveGroup> groups{};
  std::vector<PlacedTensor> outputs{};  // the primitive graph's, in order
  std::vector<PlacedTensor> kept{};
  Traffic traffic{};  // its collectives', as ModelTraffic counts them
};

// The primitive graph's per-device graphs under `assignment`; refused for
// a fused node or a collective in it, a node of a component the
// assignment does not place, or a split that cannot pass (above).
std::expected<Partitioned, std::string> Partition(const ir::Graph& graph,
                                                  const Assignment& assignment);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PARTITION_H_
