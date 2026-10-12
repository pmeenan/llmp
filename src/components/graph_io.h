// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_GRAPH_IO_H_
#define LLMP_COMPONENTS_GRAPH_IO_H_

#include <span>

#include "components/step.h"

namespace llmp::components {
// Borrowed typed ports for synchronous planning and lowering. TensorDesc owns
// format, strides and owner segment; no token phase or state is implied.
struct GraphBuildIO {
  const ir::Graph& graph;
  std::span<const ir::TensorId> inputs;
  std::span<const ir::TensorId> outputs;
  std::span<const StepWeight> weights;
};
inline GraphBuildIO GraphIO(const StepGraph& step) {
  return {step.graph, step.graph.inputs(), step.graph.outputs(), step.weights};
}
}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_GRAPH_IO_H_
