// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR's verifier (D-107): whether a graph is one op set v1 gives
// a meaning to, and one the lowerings can take. It checks, node by node in
// order:
//   - structure: every source made by an earlier node (so the graph is
//     acyclic and in a runnable order), every tensor made by exactly one
//     node, the outputs and kept tensors existing, the bounds on nodes and
//     tensors;
//   - arity and parameters: each op's sources present or absent as op.h
//     lists them, parameters in range, reserved parameter bytes zero;
//   - shapes and formats: each output's shape, encoding, layout and strides
//     what the op makes of its sources (infer.h), integer operands integer,
//     formats valid (format::Validate), shapes in canonical rank;
//   - views: a view's root a root, its encoding the root's, every byte it
//     covers inside the root's bytes, no view of a format stored in parts;
//   - bindings: a weight leaf bound to a WeightRef, a state leaf to a
//     StateRef (its segment the tensor's), no weight or state bound twice;
//     a per-owner node's outputs its owner's segment, and its sources of
//     that segment or joined;
//   - the GGML lowering's limits: rank at most four (format::kMaxDims) and
//     every extent and byte stride of a dimension within the rank at most
//     INT32_MAX (GGML takes some of them as int: RE-037), so a graph that
//     verifies is one the lowering can express.

#ifndef LLMP_IR_VERIFY_H_
#define LLMP_IR_VERIFY_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace llmp::ir {

class Graph;

enum class GraphError : std::uint8_t {
  kShape,    // an output's or operand's shape is not what the op takes or makes
  kFormat,   // an encoding, layout or format the op does not take
  kArity,    // a source missing, extra or absent where required
  kBounds,   // a view past its root, a node or tensor count past the bounds
  kCycle,    // a source not made by an earlier node; a tensor made twice
  kParams,   // parameters out of range, or reserved bytes set
  kBinding,  // a leaf's role, binding or segment wrong, or bound twice
  kLimit,    // past the GGML lowering's int32 extents and strides (RE-037)
};
std::string_view GraphErrorName(GraphError error);

struct VerifyError {
  GraphError error = GraphError::kShape;
  std::uint32_t node = 0;  // the node it is found at (its index)
  std::string detail;      // what, in words: the operand, shapes and limits
};

std::expected<void, VerifyError> Verify(const Graph& graph);

}  // namespace llmp::ir

#endif  // LLMP_IR_VERIFY_H_
