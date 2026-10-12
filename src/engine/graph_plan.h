// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_GRAPH_PLAN_H_
#define LLMP_ENGINE_GRAPH_PLAN_H_

#include <expected>
#include <span>
#include <string>

#include "components/step.h"
#include "kernels/common/lower.h"
#include "planner/plan.h"

namespace llmp::engine {

// Borrowed typed interface, valid only during the synchronous planning call.
// TensorDesc owns each port's format, strides and owner segment. Primitive IDs
// remain authoritative; StepPlan::tensors maps them through later rewrites.
// This is graph metadata, not execution or backend authority.
struct GraphBuildIO {
  const ir::Graph& graph;
  std::span<const ir::TensorId> inputs;
  std::span<const ir::TensorId> outputs;
  std::span<const components::StepWeight> weights;
};

// The common component graph-to-plan path. The caller retains the real backend
// owner and pays graph/planner metadata. No lowering, device call, binding or
// execution occurs here. Token sources supply options.shape/state_spans;
// spatial graphs leave them empty. Both staged sets and all other options keep
// their caller-defined meaning; root alignment derives only from actual facts.
std::expected<planner::StepPlan, std::string> PlanComponentGraph(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options = {});

}  // namespace llmp::engine
#endif  // LLMP_ENGINE_GRAPH_PLAN_H_
