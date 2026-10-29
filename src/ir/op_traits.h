// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_IR_OP_TRAITS_H_
#define LLMP_IR_OP_TRAITS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "ir/op.h"

namespace llmp::ir {

enum class LayoutClass : std::uint8_t { kStructural, kElementwise, kConcat, kFixed };
// Explicit semantic propagation rules. kWhole is a decision, never a missing
// handler fallback. Several operations share a rule when their algebra agrees.
enum class PartitionRule : std::uint8_t {
  kAdditive,
  kArgmax,
  kAttention,
  kCacheStore,
  kChannelNorm,
  kConcat,
  kCont,
  kDivide,
  kExpertReduce,
  kGetRows,
  kGlu,
  kMatMul,
  kMatMulId,
  kMultiply,
  kPermute,
  kPrepare,
  kReshape,
  kRope,
  kRowReduce,
  kRows,
  kScale,
  kSoftMax,
  kUnary,
  kView,
  kWhole
};
enum class ReferencePath : std::uint8_t { kNone, kDirect, kDecomposition, kCollective };

struct OpTraits {
  OpKey key;
  std::string_view name;
  std::size_t param_bytes;
  std::uint8_t required_sources, max_sources;
  LayoutClass layout;
  PartitionRule partition;
  bool decomposition;
  ReferencePath reference;
  // Sources written regardless of the output count (cache_store's cache).
  std::uint16_t written_sources;
  // Source aliased by each output, or -1. Conditional outputs are written
  // only when the node actually contains them (gdn_step's optional store).
  // A row names the first two; the rest are -1 (op_traits.cc checks every
  // index against the row's sources).
  std::array<std::int8_t, 4> aliased_outputs;
};

std::span<const OpTraits> PrimitiveTraits();
std::span<const OpTraits> FusedTraits();
const OpTraits* Traits(OpKey key);

}  // namespace llmp::ir

#endif  // LLMP_IR_OP_TRAITS_H_
