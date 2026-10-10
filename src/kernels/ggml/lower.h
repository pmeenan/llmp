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
//      them, as PlanGraph groups them), and for a composite op an
//      implementation of its own runs (planner/pattern.h route_reduce) the
//      GGML node sequence that implementation's gate takes (Gemma's fused
//      routing and scaled reduce). Node order is the planned graph's,
//      which keeps the builders' (GraphOrder) order the components emit, so
//      the graph is the one today's builders make for the same step;
//   2. makes each mask (kMaskBuild) on the device (llmp.gemma4.mask, whose
//      producer derives the cells' positions, so the step stages none; a
//      frozen read's under its frozen policy) or, with device masks off, a
//      host-built input (step_program.h kMask);
//   3. binds the planned activations at the IR placement (its extent is the
//      step's), weights and state at the addresses given (distinct
//      placeholder addresses where none is, as for a measurement), and the
//      intermediates a fused implementation never writes at its result's
//      place;
//   4. derives the DeviceChoices flags from the selection (a fused or
//      grouped implementation selected anywhere turns its PlanGraph policy
//      on) over the device's own choices, and runs PlanGraph without
//      upstream's fusion;
//   5. requires PlanGraph's steps to be the IR selection's, in order: the
//      same GGML nodes under the selected implementation, or under a member
//      of the selected group (the group is resolved here, by upstream's
//      routing). Anything else is a lowering bug, and the step is refused;
//   6. gives the input slots (tokens, positions, the head's rows, each
//      owner's store cells, a target's tap cells and host masks, a
//      drafter's features as the step's bytes), the output slots (each
//      owner's logits rows, or its greedy token; a drafter's next features
//      after them) and every tensor use for the coverage check.
// Every profile builds it (no launch); kernels/ggml/step_program.h makes a
// program of the result on CUDA builds.

#ifndef LLMP_KERNELS_GGML_LOWER_H_
#define LLMP_KERNELS_GGML_LOWER_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "ir/graph.h"
#include "kernels/common/lower.h"
#include "kernels/common/step_program.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {

struct LowerOptions {
  // kMaskBuild on the device (llmp.gemma4.mask), or built on the host.
  bool device_masks = true;
  // A frozen read's host-built mask (a drafter over its target's borrowed
  // state): its slot's cells say so (step_program.h InputCells::frozen),
  // which a stager must honour; off, such a step is refused with host masks.
  bool frozen_host_masks = false;
  // PlanGraph's graph reader index allowance (graph_plan.h).
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
};

struct LoweredStep {
  std::optional<TensorArena> arena;  // owns every tensor below
  std::vector<ggml_tensor*> nodes;   // computed nodes and views, in execution order
  std::vector<ggml_tensor*> keep;    // the step's outputs and kept tensors
  std::vector<LoweredLeaf> leaves;
  GraphPlan plan;  // PlanGraph's: the IR selection's steps, lanes and regions
  std::vector<InputSlot> inputs;
  std::vector<OutputSlot> outputs;
  std::vector<TensorUse> tensors;
  std::uint64_t extent = 0;  // the activations' (the IR placement's)
  bool measuring = false;    // placeless: never bound to run
};

// The bytes the module writes for a root of `graph` beyond the tensor's
// own (planner/placement.h RootBytesOf): a prepared Q8_1 input, which
// QuantizeQ8 writes with each row padded to MMVQ's row step (llmp_ops.h
// Q8Bytes); 0 for any other.
std::uint64_t WrittenBytes(const ir::Graph& graph, ir::TensorId root);

// Steps 1 to 6 above, with `device` the device's own choices (executor.h
// DeviceChoicesOf on CUDA builds; a model of it in tests) and `registry` the
// one the plan selected from.
std::expected<LoweredStep, KernelFailure> LowerStep(const LowerInput& input,
                                                    const execution::Registry& registry,
                                                    const DeviceChoices& device,
                                                    const LowerOptions& options = {});

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_LOWER_H_
