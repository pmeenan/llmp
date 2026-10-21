// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What a kernel source gives the engine to run planned steps (D-107;
// docs/engine-components.md, "The planner, as built"): its registry, the
// facts of its device, the implementations that stay on the stream, and
// its lowering, a function from a planned step to a step program
// (step_program.h). The engine's step source (engine/component_source.h)
// plans each step over the registry for the device and hands the plan to
// the lowering; it never names the kernel source.
//
// Vendor-free: nothing here names a kernel library or a device API.

#ifndef LLMP_KERNELS_COMMON_LOWER_H_
#define LLMP_KERNELS_COMMON_LOWER_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "components/graph_io.h"
#include "execution/registry.h"
#include "ir/graph.h"
#include "kernels/common/execution_resources.h"
#include "kernels/common/selected_operation.h"
#include "kernels/common/step_program.h"
#include "planner/plan.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::kernels {

// Optional token/state metadata. Spatial ports never fabricate a StepShape.
struct TokenLowerInput {
  const components::StepGraph& step;
  const planner::StepShape& shape;
  std::uint32_t context = 0;
  planner::StateSpans state_spans{};
};

// One shared typed graph-to-program interface. Every borrowed object lives
// through the synchronous lower/validator call; programs retain only their
// owned descriptors and existing backend/completion authority.
struct LowerInput {
  const planner::StepPlan& plan;
  components::GraphBuildIO io;
  const Addresses& addresses;
  std::optional<TokenLowerInput> token;

  LowerInput(const planner::StepPlan& p, components::GraphBuildIO graph_io, const Addresses& a,
             std::optional<TokenLowerInput> tokens = std::nullopt)
      : plan(p), io(graph_io), addresses(a), token(tokens) {}
  // Preserve the existing token construction contract without a second lowerer.
  LowerInput(const planner::StepPlan& p, const components::StepGraph& step,
             const planner::StepShape& shape, const Addresses& a, std::uint32_t context = 0,
             planner::StateSpans state_spans = {})
      : LowerInput(p, components::GraphIO(step), a,
                   TokenLowerInput{step, shape, context, std::move(state_spans)}) {}
};

using LowerFunction =
    std::function<std::expected<std::unique_ptr<StepProgram>, std::string>(const LowerInput&)>;

// Optional synchronous validation of the actual lowerer's unchanged result.
// It cannot supply a program or alter backend selection. The installing caller
// prepays and retains captured metadata; input references expire on return.
using ProgramValidator =
    std::function<std::expected<void, std::string>(const LowerInput&, const StepProgram&)>;

// A prepaid immutable callback owner, independent of numerical execution
// declarations. A backend exposes the same retained payer to each construction
// category; native lowering and logical reporting obtain separate exact loans.
struct ConstructionFunder {
  std::function<bool(std::uint64_t)> charge;
  std::function<void(std::uint64_t)> release;
};

class BackendOwner;

struct LowerOptions {
  std::shared_ptr<const BackendOwner> routed_funder{};
  bool device_masks = true;
  const SelectedSource* selected_source = nullptr;
  bool frozen_host_masks = false;
  std::uint64_t index_scratch_limit = std::numeric_limits<std::uint64_t>::max();
  std::function<std::uint64_t(const ir::WeightRef&)> expert_stride{};
};

// Immutable future launch resources, also used while measuring without a
// configured launch. Zero means every operation runs on the ordinary stream.
struct LaneConfiguration {
  std::uint32_t count = 0;
  std::uint64_t scratch_bytes_per_lane = 0;
  bool operator==(const LaneConfiguration&) const = default;
};

// Build-time mechanisms supplied by a concrete backend. These are mechanical
// capabilities only; neither a user setting nor this value grants numerical
// transformations or an execution/source envelope.
struct BackendCapabilities {
  std::uint32_t owner_lanes = 0;
  bool device_tokens = false, indexed_masks = false;
};

struct StepBackend {
  // Outlives every plan made over it.
  const execution::Registry* registry = nullptr;
  planner::DeviceFacts device{};
  LaneConfiguration lanes{};
  // The implementations that borrow the stream's library handle (lanes.h).
  std::function<bool(std::string_view)> on_stream{};
  // What its implementations write beyond a root's own bytes, and the roots
  // they write inside another's (placement.h).
  planner::RootBytesOf written{};
  planner::RootWithin within{};
  LowerFunction lower{};
  // Required sources retain this owner and obtain the effective table by const
  // reference from it. Fields above may be empty; a caller-modified copy is
  // not the owner's authority.
  std::shared_ptr<const BackendOwner> owner{};
  // Also required for undeclared backends: absence never means free metadata.
  // The backend prepays the callback/control storage before exporting this token.
  std::shared_ptr<const ConstructionFunder> construction_funder{};
  // Owned tables carry their actual payer; wrapper fields cannot replace it.
  std::shared_ptr<const ConstructionFunder> ConstructionFunding() const;
};

// Immutable effective backend/configuration. The real kernel backend creates
// its production owner; fake owners are test backends only. The table excludes
// its own shared_ptr to avoid a cycle. All referenced runtime resources keep
// their existing backend/launch completion-aware lifetime.
class BackendOwner {
 public:
  virtual ~BackendOwner() = default;
  virtual const StepBackend& backend() const = 0;
  // Source/contract metadata uses the same refusal-capable owner allowance.
  // Called only on the driver before constructing or dropping that metadata.
  virtual bool Charge(std::uint64_t bytes) const = 0;
  virtual void Release(std::uint64_t bytes) const = 0;
  // Unprepared or faulted is an error, never an absent legacy declaration.
  // First successful preparation freezes the value; later changes refuse.
  virtual std::expected<execution::BackendDeclaration, std::string> Ready() const = 0;
  // A distinct eager binding/launch over caller-owned paid workspace. Factory
  // affinity, immutable producer identity and host prepayment are enforced here.
  std::expected<ExecutionResources, ResourceFailure> Resources(
      std::shared_ptr<const BackendOwner> self, ResourceWorkspace workspace,
      ParentResources retained, ResourceKeep backing) const;

 protected:
  virtual std::expected<std::uint64_t, ResourceFailure> ResourceBytes(ResourceWorkspace) const {
    return std::unexpected(ResourceFailure{
        ResourceError::kUnsupported, "this backend does not provide private execution resources"});
  }
  virtual std::expected<ResourceInstance*, ResourceFailure> MakeResources(ResourceOrigin,
                                                                          ResourceWorkspace) const {
    return std::unexpected(ResourceFailure{
        ResourceError::kUnsupported, "this backend does not provide private execution resources"});
  }
};

inline std::shared_ptr<const ConstructionFunder> StepBackend::ConstructionFunding() const {
  return owner ? owner->backend().construction_funder : construction_funder;
}

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_LOWER_H_
