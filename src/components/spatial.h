// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_SPATIAL_H_
#define LLMP_COMPONENTS_SPATIAL_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/step.h"
#include "spec/graph_links.h"

namespace llmp::components {

// Logical order WHCN, width innermost. No token rows, past or KV semantics.
struct SpatialShape {
  std::uint64_t width = 0, height = 0, channels = 0, images = 1;
  bool operator==(const SpatialShape&) const = default;
};
// Representable dense/reference storage only, not an execution admission cap.
// Actual graph/reference construction and device liveness are paid separately.
std::optional<std::uint64_t> SpatialElements(SpatialShape shape);

struct SpatialOwner {
  std::uint8_t owner = 0;
  SpatialShape shape;
};
struct SpatialInput {
  std::uint8_t owner = 0;
  std::uint16_t slot = 0;
  SpatialShape shape;
  format::Encoding encoding = format::Encoding::kF32;
};
struct SpatialValue {
  ir::TensorId tensor;
  SpatialShape shape;
};
// Value zero is this owner's input; component i produces value i+1.
// Every edge points backward within the same owner. An absent link table
// preserves the serial chain. Source composition retains this table.
using SpatialLink = spec::GraphLink;
struct SpatialBuildOptions {
  std::span<const SpatialLink> links{};
  // Host reference copies are optional; real source/graph metadata is always paid.
  // False does not authorize a reference executor to allocate any buffers.
  bool reference_buffers = true;
  format::Encoding input_encoding = format::Encoding::kF32;
  // Empty retains the last value. Otherwise distinct produced value IDs, in
  // output-port order, repeated for each owner. Zero (the input) is not an export.
  std::span<const std::uint16_t> exports{};
  // Sorted (owner,slot), slots0..N-1 per owner, after its primary F32 input.
  std::span<const SpatialInput> inputs{};
  // The graph's layout round trips removed (ir/layout.h SinkLayouts): what
  // a planning policy enabling layout_sinking asks of the builder.
  bool sink_layouts = false;
};
struct SpatialFragment {
  ir::Builder& builder;
  const Instance& instance;
  ir::TensorId input;
  SpatialShape shape;
  std::span<const ir::TensorId> weights;
  std::optional<SpatialValue> auxiliary{};
  std::optional<SpatialValue> carry{};
};
// Reference/control only. Decoded weights and complete output storage belong
// to the caller's admitted reference allowance, never resident widened weights.
struct SpatialRefIO {
  const Instance& instance;
  SpatialShape shape;
  std::span<const double> input;
  std::span<const float> kernel;
  std::span<const float> bias;
  std::vector<double> output;
  std::span<const double> auxiliary{};
  std::optional<SpatialShape> auxiliary_shape{};
  std::span<const double> carry{};
  std::optional<SpatialShape> carry_shape{};
};
struct SpatialOps {
  std::expected<SpatialShape, std::string> (*infer_shape)(const Instance&, SpatialShape,
                                                          std::optional<SpatialShape>);
  std::expected<SpatialShape, std::string> infer(const Instance& instance, SpatialShape shape,
                                                 std::optional<SpatialShape> auxiliary = {}) const {
    return infer_shape(instance, shape, auxiliary);
  }
  std::expected<SpatialValue, std::string> (*emit)(SpatialFragment&);
  std::expected<void, std::string> (*reference)(SpatialRefIO&);
  // Includes all emitted descriptors and primitive-decomposition temporaries.
  std::uint16_t max_nodes = 1;
  std::uint64_t (*temporary_values)(const Instance&, SpatialShape) = nullptr;
  format::Encoding input_encoding = format::Encoding::kF32;
  format::Encoding auxiliary_encoding = format::Encoding::kF32;
  bool needs_carry = false;
};

// Paid before copies/graph construction. The callbacks retain their actual
// allowance owner. This synchronous compiler never submits device work.
struct SpatialPayment {
  std::function<bool(std::uint64_t)> charge;
  std::function<void(std::uint64_t)> release;
};
class SpatialGraph {
 public:
  SpatialGraph(SpatialGraph&&) noexcept = default;
  SpatialGraph& operator=(SpatialGraph&&) = delete;
  SpatialGraph(const SpatialGraph&) = delete;
  SpatialGraph& operator=(const SpatialGraph&) = delete;
  const ir::Graph& graph() const { return graph_; }
  std::span<const StepWeight> weights() const { return weights_; }
  std::span<const SpatialOwner> inputs() const { return inputs_; }
  // Owner-major, then requested export order; multiple ports may share an owner.
  std::span<const SpatialOwner> outputs() const { return outputs_; }
  // Source-owned domain/version and complete input/output WHCN owner order.
  std::span<const std::uint64_t> parameters() const { return parameters_; }
  std::uint64_t metadata_bytes() const { return metadata_bytes_; }
  std::uint64_t reference_bytes() const { return reference_bytes_; }

 private:
  friend std::expected<SpatialGraph, std::string> BuildSpatialStep(
      const spec::ModelSpec&, std::span<const std::span<const BoundTensor>>,
      std::span<const SpatialOwner>, SpatialPayment, SpatialBuildOptions);
  SpatialGraph(std::shared_ptr<const void> payment, ir::Graph graph,
               std::vector<StepWeight> weights, std::vector<SpatialOwner> inputs,
               std::vector<SpatialOwner> outputs, std::vector<std::uint64_t> parameters,
               std::uint64_t metadata, std::uint64_t reference)
      : payment_(std::move(payment)),
        graph_(std::move(graph)),
        weights_(std::move(weights)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        parameters_(std::move(parameters)),
        metadata_bytes_(metadata),
        reference_bytes_(reference) {}
  // First member: every graph/port/weight copy dies before payment returns.
  std::shared_ptr<const void> payment_;
  ir::Graph graph_;
  std::vector<StepWeight> weights_;
  std::vector<SpatialOwner> inputs_, outputs_;
  std::vector<std::uint64_t> parameters_;
  std::uint64_t metadata_bytes_ = 0, reference_bytes_ = 0;
};

// One stateless image stage, up to the spec/value-ID component limit and 16 distinct owners.
// Complete checked trajectories/roles are validated before graph emission; branch edges are
// owner-local backward value references. Typed graph ports are graph.inputs()/outputs();
// descriptors retain owner IDs. No family mapping, backend lowering, executor, cache or admission
// is added.
std::expected<SpatialGraph, std::string> BuildSpatialStep(
    const spec::ModelSpec& spec, std::span<const std::span<const BoundTensor>> tensors,
    std::span<const SpatialOwner> owners, SpatialPayment payment, SpatialBuildOptions options = {});

}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_SPATIAL_H_
