// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Primitive decompositions (D-053's primitive-fallback rule,
// docs/portability.md#the-registry-rule): every fused kind's meaning, and
// the composite ops' (hyper-connections, the indexer's scores, pooling,
// routing, the expert reduction, the causal convolution and the gated
// delta rule) in simpler ops of op set v1, so a backend with primitives
// alone runs every graph and a pattern that does not match leaves a
// correct graph. The reference executor (kernels/reference/exec.h) runs
// fused nodes through these; its tests check each composite op's direct
// form against its decomposition.
//
// Decompositions preserve source strides in views, and copy strided
// values before a reshape requires packed storage.

#ifndef LLMP_IR_DECOMPOSE_H_
#define LLMP_IR_DECOMPOSE_H_

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {

// Whether `key` has a decomposition: every fused kind but the batch key
// grouped_stores (op.h), and the composite ops above.
bool HasDecomposition(OpKey key);

// Emits the decomposition of a node of `key` with `params` over `sources`
// (ids of `b`) into `b`, and returns its outputs. A computed output whose
// encoding in `encodings` (by position; empty or kF32 entries are left)
// differs from the decomposition's ends in a cast to it. Errors are b's.
Outputs Decompose(Builder& b, OpKey key, const ParamBytes& params,
                  std::span<const TensorId> sources, std::span<const format::Encoding> encodings);

// The source a fused kind's output `output` is a view of (a store's
// cache), if any.
std::optional<std::size_t> AliasedSource(OpKey key, std::size_t output);

// `graph` with every fused node, and with `composites` every composite op,
// replaced by its decomposition. `map`, when given, receives the new id of
// each of graph's tensors (a decomposed node's outputs map to its
// decomposition's). Refused (kBounds) if fused nodes still remain after a
// bounded number of passes, and (kFormat) where a view's source would
// change layout (graph.h SameViewSource).
std::expected<Graph, VerifyError> Expand(const Graph& graph, bool composites,
                                         std::vector<TensorId>* map = nullptr);

}  // namespace llmp::ir

#endif  // LLMP_IR_DECOMPOSE_H_
