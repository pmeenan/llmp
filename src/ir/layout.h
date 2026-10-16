// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Layout sinking (D-107): a step's graph with its layout round trips
// removed. The spatial components keep their values in [W, H, C, N] and
// move them to channel rows and back around each row-wise operation (a
// Cont of a Permute each way); a projection's result goes to the spatial
// layout only for the next one to bring it back.
//
// The pass follows each value as a strided view of a contiguous storage
// tensor instead of materializing it: a permute, a reshape, a contiguous
// copy and a view of a contiguous value change only the view; an
// elementwise operation (add, sub, mul, div, scale, unary, clamp, convert,
// a two-source GLU, repeat) and a concatenation run in one storage order,
// the one that leaves the fewest operands to reorder (and, on a tie, the
// one the value's readers want: a product's input, a norm's rows, the
// step's outputs want their logical layout, a view passes its readers' on).
// A value is materialized (a view and, where the reader read packed values,
// a Cont) only where a reader needs it. Every rewrite moves bytes or
// reorders elementwise work: the result is the original graph's bit for bit
// (ir_layout_test checks it on the reference executor).
//
// It is a graph transform, before planning: the step's builder applies it
// (components/spatial.h SpatialBuildOptions::sink_layouts), where the
// model's planning policy enables it (planner/pattern.h option
// layout_sinking), so a plan, its lowering and its fused nodes' primitives
// all see one graph. It copies a graph it does not handle unchanged
// (state, cache stores and other in-place writes), and one it cannot make
// copy fewer bytes.

#ifndef LLMP_IR_LAYOUT_H_
#define LLMP_IR_LAYOUT_H_

#include <cstddef>
#include <expected>
#include <string>
#include <vector>

#include "ir/graph.h"
#include "ir/verify.h"

namespace llmp::ir {

struct Sunk {
  Graph graph;
  // The original's tensors in the new graph, where it holds them (a value
  // only ever viewed maps to none): its leaves, outputs and kept tensors
  // always.
  std::vector<TensorId> tensors;
  std::size_t rewritten = 0;  // the original nodes it did not copy
  // Why the graph was copied unchanged where neither form could be rebuilt
  // (a diagnostic; empty otherwise).
  std::string declined;
};

// The graph with its layout round trips removed, or the graph unchanged
// (where the pass does not handle it, cannot rebuild it, or cannot make it
// copy fewer bytes); an error only where copying it fails.
std::expected<Sunk, VerifyError> SinkLayouts(const Graph& graph);

}  // namespace llmp::ir

#endif  // LLMP_IR_LAYOUT_H_
