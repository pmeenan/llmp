// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Lowering a planned IR step onto the GGML module (D-107;
// docs/engine-components.md, "The planner, as built"): the IR is the source
// of truth, and a ggml_cgraph exists only here, as the private target that
// lets the module's planning, placement checks, binding and capture run
// unchanged. Private to kernels/ggml: the engine sees the step program
// (step_program.h) and never a GGML type.
//
// LowerStep, from a StepPlan (planner/plan.h) and the primitive step it was
// planned from:
//   1. makes a GGML tensor for each tensor of the planned graph in a
//      TensorArena, in node order: leaves for the step's inputs, weights and
//      state, GGML's own builder call for each node, and for a fused node
//      the GGML node sequence its implementation covers (the primitive
//      nodes the rewrite replaced: GGML's fused implementations run over
//      them, in the selected recipe), and for a composite op an
//      implementation of its own runs (planner/pattern.h route_reduce) the
//      GGML node sequence that implementation's gate takes (Gemma's fused
//      routing and scaled reduce). Node order is the planned graph's,
//      which keeps the builders' (GraphOrder) order the components emit, so
//      the graph is the one today's builders make for the same step;
//   2. makes each mask (kMaskBuild) on the device (llmp.gemma4.mask, whose
//      producer derives the cells' positions, so the step stages none; a
//      frozen read's under its frozen policy) or, with device masks off, a
//      host-built input (step_program.h kMask);
//   3. authenticates physical materializations and reads from the selected
//      recipes, retaining proof-only dependencies in the IR; a descriptor-only
//      provisional binding makes group routing checkable without submitting work;
//   4. emits the selected implementations' explicit launch recipes, resolving
//      only declared groups. It never runs the independent native PlanGraph;
//   5. places the real schedule, including paired products and extra reads,
//      then authenticates final addresses, typed operands and protected owner
//      recipes. Provisional addresses grant no execution authority;
//   6. gives the input slots (tokens, positions, the head's rows, each
//      owner's store cells, a target's tap cells and host masks, a
//      drafter's features, which the engine copies in on the device), the
//      output slots (each owner's logits rows, or its greedy token; a
//      drafter's next features after them, marked as features) and every
//      tensor use for the coverage check.
// Every profile builds it (no launch); kernels/ggml/step_program.h makes a
// program of the result on CUDA builds.

#ifndef LLMP_KERNELS_GGML_LOWER_H_
#define LLMP_KERNELS_GGML_LOWER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "execution/registry.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/graph.h"
#include "kernels/common/lower.h"
#include "kernels/common/routed_program.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/selected_operation.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

// Independent of numerical execution declarations. The backend prepays this
// immutable callback owner before allocating it; each native build keeps it
// alive until retained-plan accounting takes over or native teardown completes.
using ConstructionFunder = kernels::ConstructionFunder;
struct ConstructionPayment {
  ConstructionPayment() = default;
  ConstructionPayment(const ConstructionPayment&) = delete;
  ConstructionPayment& operator=(const ConstructionPayment&) = delete;
  std::shared_ptr<const ConstructionFunder> funder;
  std::uint64_t bytes = 0;
  ~ConstructionPayment() {
    if (bytes != 0) funder->release(bytes);
  }
};
struct LowerConstructionInventory {
  std::size_t descriptors = 0;
  std::uint64_t bytes = 0;
};
// Allocation-free checked inventory of the actual lowering descriptor bound.
// Both backend admission and TensorArena sizing consume this same operation
// inventory. It is a bound, not a measured memory peak. Standalone LowerStep
// callers must supply their own enclosing prepayment; production uses Lower.
std::expected<LowerConstructionInventory, KernelFailure> LowerConstructionOf(
    const LowerInput& input);

struct LowerOptions {
  // Actual backend allowance; used before allocating ordered routed metadata.
  std::shared_ptr<const kernels::BackendOwner> routed_funder{};
  // kMaskBuild on the device (llmp.gemma4.mask), or built on the host.
  bool device_masks = true;
  // Optional build-time source for operations the IR planner selected.
  const SelectedSource* selected_source = nullptr;
  // A frozen read's host-built mask (a drafter over its target's borrowed
  // state): its slot's cells say so (step_program.h InputCells::frozen),
  // which a stager must honour; off, such a step is refused with host masks.
  bool frozen_host_masks = false;
  // The selected recipe reader index allowance (graph_read_index.h).
  std::uint64_t index_scratch_limit = std::numeric_limits<std::uint64_t>::max();
  // The bytes between an expert array's experts as the runner placed them
  // (a slab's pitch, engine/paged_weights.h SlabSpec); 0 or none: packed.
  std::function<std::uint64_t(const ir::WeightRef&)> expert_stride{};
};

// A leaf of the lowered graph and the primitive step's tensor it stands for
// (an input, a host-built mask's kMaskBuild output, a weight, a state).
struct LoweredLeaf {
  ggml_tensor* tensor = nullptr;
  ir::TensorId primitive{};
  // A weight of several parts (MXFP8's codes and scales, NVFP4 rows'
  // data and tensor scale, a CUTLASS expert slab: its gate-and-up codes):
  // the part this leaf is.
  std::optional<format::PartRole> part{};
};

// A verify's save's buffer (components/step.h CommitSave): F32 [width,
// rows], bound at Addresses::save's.
struct SavedLeaf {
  ggml_tensor* tensor = nullptr;
  std::uint32_t layer = 0;
  std::uint16_t component = 0;
  std::string what;
  std::uint8_t segment = 0;
};

struct LoweredRouting {
  kernels::RoutedLayout layout;
  std::vector<std::size_t> members;
  std::vector<kernels::RoutedCopy> uploads;
  std::vector<std::pair<std::size_t, std::uint64_t>> descriptors;  // selected index, pinned offset
  ggml_tensor* preparation = nullptr;
  // The preparation's actual activation-tail placement, relative to the
  // common activation base. Never a process address in a report.
  std::uint64_t activation_offset = 0, activation_bytes = 0;
  std::size_t cut = 0;
  base::Sha256Digest graph_identity{};
};
// Actual arena descriptor identity. The pointer is an internal lookup key,
// never serialized; id is its stable construction ordinal. Unmapped auxiliary
// descriptors retain their owning planned producer without inventing an IR value.
struct PhysicalTensor {
  const ggml_tensor* descriptor = nullptr;
  std::uint32_t id = 0;
  ir::TensorId tensor{}, root{};
  ir::NodeId producer{};
  std::optional<format::PartRole> part{};
};
struct PhysicalRoot {
  ir::TensorId root;
  ir::NodeId producer;
  ir::OpKey operation;
  std::uint16_t component = 0;
  std::uint32_t backend_tensor = UINT32_MAX;
  std::uint64_t offset = 0, bytes = 0;
  std::array<std::uint64_t, 4> ne{}, nb{};
  format::Encoding encoding = format::Encoding::kF32;
};
struct LoweredStep {
  std::optional<TensorArena> arena;  // owns every tensor below
  std::vector<ggml_tensor*> nodes;   // computed nodes and views, in execution order
  std::vector<ggml_tensor*> keep;    // the step's outputs and kept tensors
  std::vector<LoweredLeaf> leaves;
  std::vector<SavedLeaf> saves;
  std::vector<SelectedNode> selected;  // exact selected calls, borrowing this arena
  GraphPlan plan;                      // explicit lowered schedule, lanes and regions
  std::vector<InputSlot> inputs;
  std::vector<OutputSlot> outputs;
  std::vector<TensorUse> tensors;
  std::vector<PhysicalRoot> physical_roots;
  // Sorted by internal descriptor pointer for allocation-free logarithmic
  // ID lookup while lazily reporting actual concrete nodes/source edges.
  std::vector<PhysicalTensor> physical_tensors;
  std::size_t descriptor_bound = 0;  // canonical prepaid construction inventory
  std::uint64_t extent = 0;          // the activations' (the IR placement's)
  std::uint64_t logical_extent = 0;  // conservative semantic-IR placement, for diagnostics
  // Sum of omitted logical root sizes, not a peak-memory saving.
  std::uint64_t unmaterialized_logical_bytes = 0;
  std::uint64_t unmaterialized_roots = 0;
  bool measuring = false;  // placeless: never bound to run
  // The plan's prologue (planner/plan.h StepPlan::prologue) as `plan`'s
  // step indices, in order: what a warm run skips.
  std::vector<std::size_t> prologue{};
  // Payment outlives the routed vectors (members are destroyed in reverse).
  std::shared_ptr<const void> routing_payment{};
  std::vector<LoweredRouting> routing{};
  base::Sha256Digest routing_identity{};
  std::uint64_t routed_pinned_bytes = 0;
  std::optional<std::vector<kernels::RoutedWeight>> weight_reads{};
};

const PhysicalTensor* PhysicalTensorOf(const LoweredStep& step, const ggml_tensor* tensor);
struct PhysicalReportMetrics {
  std::uint64_t pool_bytes = 0, lanes = 0, lane_scratch_bytes = 0, launches = 0;
};
// The same allocation-free lazy serializer is built in every profile.
void StreamPhysicalPlanReport(const LoweredStep& step, const execution::Registry& registry,
                              const kernels::PlanReportSink& sink,
                              const PhysicalReportMetrics& metrics = {});

// The bytes the module writes for a root of `graph` beyond the tensor's
// own (planner/placement.h RootBytesOf): a prepared Q8_1 input, which
// QuantizeQ8 writes with each row padded to MMVQ's row step (llmp_ops.h
// Q8Bytes); 0 for any other.
std::uint64_t WrittenBytes(const ir::Graph& graph, ir::TensorId root);

// The root the module writes a root inside (placement.h RootWithin): a
// kernel's outputs laid out in one blob at its first output's place
// (llmp.hc.prep, llmp.hc.mix_bf16's prepared and rounded rows,
// llmp.moe.router, llmp.gdn.gates), or none.
ir::TensorId WithinRoot(const ir::Graph& graph, ir::TensorId root);

// Steps 1 to 6 above, with `device` the device's own choices (executor.h
// DeviceChoicesOf on CUDA builds; a model of it in tests) and `registry` the
// one the plan selected from.
std::expected<LoweredStep, KernelFailure> LowerStep(const LowerInput& input,
                                                    const execution::Registry& registry,
                                                    const DeviceChoices& device,
                                                    const LowerOptions& options = {});

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_LOWER_H_
