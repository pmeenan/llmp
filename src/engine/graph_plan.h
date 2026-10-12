// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_GRAPH_PLAN_H_
#define LLMP_ENGINE_GRAPH_PLAN_H_

#include <expected>
#include <span>
#include <string>

#include "components/graph_io.h"
#include "kernels/common/lower.h"
#include "planner/plan.h"

namespace llmp::engine {

using GraphBuildIO = components::GraphBuildIO;

// The common component graph-to-plan path. The caller retains the real backend
// owner and pays graph/planner metadata. No lowering, device call, binding or
// execution occurs here. Token sources supply options.shape/state_spans;
// spatial graphs leave them empty. Both staged sets and all other options keep
// their caller-defined meaning; root alignment derives only from actual facts.
std::expected<planner::StepPlan, std::string> PlanComponentGraph(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options = {});

// The same plan, real backend lower and read-only validator path for token and
// spatial component graphs. The caller retains and pays the effective backend
// owner and graph lifetime; this helper supplies no execution authority.
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildComponentProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    std::optional<kernels::TokenLowerInput> token = std::nullopt,
    const kernels::ProgramValidator& validate = {});

}  // namespace llmp::engine
#endif  // LLMP_ENGINE_GRAPH_PLAN_H_
