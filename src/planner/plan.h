// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Planning one step (D-053, D-107; docs/engine-components.md#the-graph-ir):
// from a step's primitive graph (components/step.h BuildStep), the
// registry, the device's facts and the alias's policy to a StepPlan that a
// lowering (kernels/common/step_program.h) turns into a program:
//
//   1. the enabled rewrite patterns applied (pattern.h);
//   2. every launching node selected (select.h): its first candidate, or
//      with a batch pattern a run of nodes under one implementation. A
//      node without a candidate whose op has a primitive decomposition
//      (every fused kind and composite op, ir/decompose.h) is replaced by
//      it and the graph selected again (the primitive-fallback rule,
//      docs/portability.md); one without either refuses the plan;
//   3. lanes (lanes.h), where the caller tags nodes;
//   4. the activations placed (placement.h).
//
// Deterministic: one graph, registry, device and policy give one plan.
// The planned graph, its steps and placement name the planned graph's
// tensors and nodes; `tensors` and `from` relate them to the primitive
// graph's (the step's inputs, weights and state are the primitive's ids).
//
// KeyOf makes the plan cache's key (step_shape.h PlanKey): the spec's
// digest, the step's shape, the enabled patterns and the policy, and the
// partition.

#ifndef LLMP_PLANNER_PLAN_H_
#define LLMP_PLANNER_PLAN_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "ir/graph.h"
#include "planner/lanes.h"
#include "planner/pattern.h"
#include "planner/placement.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner {

enum class PlanError : std::uint8_t {
  kNoImplementation,  // a node no implementation computes and nothing decomposes
  kPattern,           // a rewrite or decomposition the IR refused (a planner bug)
  kPlacement,         // the activations could not be placed
  kBounds,            // past a bound (the decomposition passes)
  kPolicy,            // the policy breaks its own rules
};
std::string_view PlanErrorName(PlanError error);

struct PlanFailure {
  PlanError error = PlanError::kNoImplementation;
  std::string detail;
};

struct PlanOptions {
  // A D-092 verify's plan: every implementation row-invariant (also the
  // policy's row_invariant).
  bool row_invariant = false;
  // Every implementation recordable for replay (a captured step).
  bool capturable = false;
  // The inputs the step copies into its activation region (the primitive
  // graph's ids); none given: every input leaf.
  std::optional<std::vector<ir::TensorId>> staged{};
  // Lanes for the primitive graph's nodes (a node a rewrite replaced
  // passes its tag to what replaced it), and the implementations that stay
  // on the stream.
  LaneTags lanes{};
  std::function<bool(std::string_view)> on_stream{};
  std::uint64_t alignment = 256;
  // What the lowering's implementations write beyond a root's own bytes
  // (placement.h RootBytesOf).
  RootBytesOf written{};
};

struct StepPlan {
  ir::Graph graph;               // the planned graph
  std::vector<Selected> steps;   // in node order
  std::vector<Region> regions;   // lanes' spans of steps
  Placement placement;           // the planned graph's roots
  std::vector<Applied> applied;  // rewrites, over the primitive graph's nodes
  // The planned graph's node i stands for the primitive graph's from[i];
  // the primitive graph's tensor i is the planned graph's tensors[i] (none
  // where a rewrite elided it).
  std::vector<ir::NodeId> from;
  std::vector<ir::TensorId> tensors;
  // Primitive decompositions taken for want of an implementation (their
  // keys' names), in order.
  std::vector<std::string> decomposed;
};

std::expected<StepPlan, PlanFailure> Plan(ir::Graph primitive, const execution::Registry& registry,
                                          const DeviceFacts& device, const Policy& policy,
                                          const PlanOptions& options = {});

// A plan's key: `spec` the model's bound digest (spec.h BoundDigest).
PlanKey KeyOf(const base::Sha256Digest& spec, const StepShape& shape, const Policy& policy,
              std::uint32_t partition = 0);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PLAN_H_
