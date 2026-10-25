// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Activation placement over a plan's steps (a port of
// kernels/ggml/graph_plan.h PlaceActivations onto the IR): where each
// tensor a step computes lives in the step's one activation region, and
// the region's extent, the plan's peak activation memory.
//
// Placed are the root tensors the steps' nodes compute (no view, no
// weight or state: a store's result is a view of its cache), and the
// staged inputs (leaves the step copies into the region). A tensor lives
// from the step that computes it (a staged input from the start) to the
// last step reading it or a view of it; a prepared input (kPrepare) keeps
// its source live as long as itself, since a lowering's prepared product
// may read both (GGML's). The graph's last launching node's results, the
// step's outputs and kept tensors (or the tensors they view) live to the
// end. Over a region's span (lanes.h) everything its steps compute or read
// lives for the whole span. Tensors whose lifetimes meet never share
// bytes, so a step's results share none with its inputs. Each is rounded
// up to `alignment` and placed largest first (ties in tracking order:
// staged inputs, then step by step) at the lowest offset free for its
// whole lifetime.

#ifndef LLMP_PLANNER_PLACEMENT_H_
#define LLMP_PLANNER_PLACEMENT_H_

#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ir/graph.h"
#include "planner/lanes.h"
#include "planner/select.h"

namespace llmp::planner {

struct Placement {
  std::vector<std::pair<ir::TensorId, std::uint64_t>> offsets{};  // root, offset
  std::uint64_t extent = 0;
  std::uint64_t alignment = 256;
};

// The bytes a root tensor spans (its strides' reach, or its parts').
std::expected<std::uint64_t, std::string> RootBytes(const ir::TensorDesc& tensor);

// Checked comparison-only admission bound for the temporary vectors owned
// by PlaceActivations and the final alignment audit. The caller adds the
// already-live staged-input and prologue inventories before allocation.
std::expected<std::uint64_t, std::string> PlacementMetadataBytes(const ir::Graph& graph);

// The bytes a lowering's implementation writes for a root it computes,
// where that is more than the tensor's own (GGML's MMVQ writes a prepared
// Q8_1 input's rows padded to its row step), or 0: a root takes the larger.
using RootBytesOf = std::function<std::uint64_t(const ir::Graph&, ir::TensorId)>;

// The root a lowering writes a root inside (one kernel's outputs laid in
// one blob at the first output's place: RootBytesOf holds the blob), or
// none: such a root is not placed itself, and the root it lies in lives
// while it is read.
using RootWithin = std::function<ir::TensorId(const ir::Graph&, ir::TensorId)>;

// A lowering's authenticated physical contract, after its selected recipes
// have checked their semantic operands. Semantic proof dependencies stay in
// the IR even when no device allocation or read implements them. Every tensor
// has a materialization bit; every node has its actual input reads, including
// extra reads of a paired launch. Empty is a real empty set, not unknown.
// Generic planning omits this contract and conservatively places the IR.
struct PhysicalStorage {
  std::vector<bool> materialized;
  std::vector<std::vector<ir::TensorId>> reads;
};

// `alignment` is a power of two.
// `preserved`: tensors kept in place between runs of the step (a plan's
// prologue results the rest reads, its invariant inputs): each lives the
// whole step, from before its first step to after its last, shared with
// nothing.
std::expected<Placement, std::string> PlaceActivations(
    const ir::Graph& graph, std::span<const Selected> steps, std::span<const Region> regions,
    std::span<const ir::TensorId> staged, std::uint64_t alignment = 256,
    const RootBytesOf& written = {}, const RootWithin& within = {},
    std::span<const ir::TensorId> preserved = {}, const PhysicalStorage* physical = nullptr);

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_PLACEMENT_H_
