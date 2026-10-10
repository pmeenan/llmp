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
#include <string>
#include <string_view>

#include "components/step.h"
#include "execution/registry.h"
#include "kernels/common/step_program.h"
#include "planner/plan.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::kernels {

// One step to lower: its plan, the primitive step it was planned from
// (BuildStep's graph and interface: the plan's `from` and `tensors` relate
// the two), its shape, the addresses it binds and the stage's context.
struct LowerInput {
  const planner::StepPlan& plan;
  const components::StepGraph& step;
  const planner::StepShape& shape;
  const Addresses& addresses;
  std::uint32_t context = 0;
  planner::StateSpans state_spans{};
};

using LowerFunction =
    std::function<std::expected<std::unique_ptr<StepProgram>, std::string>(const LowerInput&)>;

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
};

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_LOWER_H_
