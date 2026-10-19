// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "base/check.h"
#include "engine/model_runner.h"

namespace llmp::engine {

// Pure source queries also serve host-only lowering tools. Keep their
// implementation independent of runner execution and its device backend.
std::vector<std::uint64_t> StepSource::ShapeParameters(const planner::StepShape& shape) const {
  auto count = CountShapeParameters(shape);
  base::Check(count.has_value(), "shape parameter count overflow");
  std::vector<std::uint64_t> words(*count);
  base::Check(FillShapeParameters(shape, words).has_value(), "shape parameter emitter changed");
  return words;
}

}  // namespace llmp::engine
