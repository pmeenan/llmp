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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "components/graph_io.h"
#include "execution/registry.h"
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

class BackendOwner;

struct StepBackend {
  // Outlives every plan made over it.
  const execution::Registry* registry = nullptr;
  planner::DeviceFacts device{};
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
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_LOWER_H_
