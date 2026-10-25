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

// Finalize physical alignment facts once before ordinary or cumulative
// planning. The returned callbacks own their copied allocation facts. Other
// callbacks and their immutable referents keep the caller's existing lifetime.
std::expected<planner::PlanOptions, std::string> PrepareComponentPlanOptions(
    const GraphBuildIO& io, const kernels::StepBackend& backend,
    const kernels::Addresses& addresses, planner::PlanOptions options = {});

// In-place form for a caller that must preserve the original callbacks if
// physical setup fails and the ordinary path needs to take over.
std::expected<void, std::string> PrepareComponentPlanOptionsInPlace(
    const GraphBuildIO& io, const kernels::StepBackend& backend,
    const kernels::Addresses& addresses, planner::PlanOptions& options);

// The common component graph-to-plan path. The caller retains the real backend
// owner and pays graph/planner metadata. No lowering, device call, binding or
// execution occurs here. Token sources supply options.shape/state_spans;
// spatial graphs leave them empty. Both staged sets and all other options keep
// their caller-defined meaning; root alignment derives only from actual facts.
std::expected<planner::StepPlan, std::string> PlanComponentGraph(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options = {});

// Lower and report an already-owned plan. This consumes the exact plan and
// typed input supplied by the caller; it never replans or clones the graph.
// The source graph, backend, policy, address bindings, token metadata, report
// payer, and resources must remain alive through this synchronous call.
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> LowerComponentPlan(
    const kernels::LowerInput& input, const kernels::StepBackend& backend,
    const planner::Policy& policy, const kernels::ProgramValidator& validate = {},
    std::shared_ptr<const kernels::BackendOwner> report_owner = {},
    const kernels::ExecutionResources* resources = nullptr);

// The same plan, real backend lower and read-only validator path for token and
// spatial component graphs. The caller retains and pays the effective backend
// owner and graph lifetime; this helper supplies no execution authority.
// Report capture measures typed arrays and copied strings without serializing,
// then obtains an exact scoped owner subgrant before allocating the inventory.
// ConstructionPaid transfers that charge once to the installed PlanAccount.
// The backend's independent construction funder pays undeclared backends too.
// An explicit report_owner can pay when an effective view omits its owner;
// absence or refusal of both payment paths fails before inventory allocation.
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildComponentProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    std::optional<kernels::TokenLowerInput> token = std::nullopt,
    const kernels::ProgramValidator& validate = {},
    std::shared_ptr<const kernels::BackendOwner> report_owner = {},
    const kernels::ExecutionResources* resources = nullptr);

// Internal ordinary fallback shared with cumulative construction. It never
// examines ranking rows, so an optional proof failure cannot recurse.
namespace detail {
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildOrdinaryComponentProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    std::optional<kernels::TokenLowerInput> token, const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources);
}

// Ordinary stateless source entry. Its phase is prefill, matching normal
// spatial/staged execution, and its active stage is the effective policy's.
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildCumulativeRecipeProgram(
    const GraphBuildIO& io, const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources = nullptr);

// The production cumulative recipe path. An empty registry comparison table
// stays on BuildComponentProgram's unchanged fast path; this helper is only
// called when the table has rows and always returns the actual ordinary
// incumbent or an actual selected alternative lowered from its owned plan.
std::expected<std::unique_ptr<kernels::StepProgram>, std::string> BuildCumulativeRecipeProgram(
    const GraphBuildIO& io, const components::StepGraph& token_step,
    const planner::StepShape& shape, std::uint16_t active_stage, std::uint32_t context,
    const kernels::StepBackend& backend, const planner::Policy& policy,
    const kernels::Addresses& addresses, planner::PlanOptions options,
    const kernels::ProgramValidator& validate,
    std::shared_ptr<const kernels::BackendOwner> report_owner,
    const kernels::ExecutionResources* resources = nullptr);

}  // namespace llmp::engine
#endif  // LLMP_ENGINE_GRAPH_PLAN_H_
