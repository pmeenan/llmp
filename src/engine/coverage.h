// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// BP-A1's in-process check, once per planned shape (docs/engine.md): every
// tensor a plan binds lies in cataloged, resident device memory of the
// class its role says (PagedNode::Covered), the owner's or shared. The
// engine core's form, over a step program's tensor uses
// (kernels/common/step_program.h).

#ifndef LLMP_ENGINE_COVERAGE_H_
#define LLMP_ENGINE_COVERAGE_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "kernels/common/step_program.h"

namespace llmp::engine {

class PagedNode;

struct Coverage {
  std::uint64_t tensors = 0;
  std::uint64_t violations = 0;
  std::string first_violation;
};

// Each use against its role's class: a weight in weights, state in live
// state (its reservation: a step reads only what it backed first),
// activations, inputs and outputs in the scratch workspace, a constant in
// the runner's own runtime memory. `what` names the plan in the first
// violation.
void CheckCoverage(const PagedNode& node, int owner, std::span<const kernels::TensorUse> uses,
                   std::string_view what, Coverage& coverage);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_COVERAGE_H_
