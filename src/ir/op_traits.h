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

// Shape rules share an inference implementation by semantic family.
enum class InferenceClass : std::uint8_t {
  kStructure,
  kProducts,
  kAttentionAndMasks,
  kNormsAndRotations,
  kElementwise,
  kReductionsAndSelection,
  kRouting,
  kRecurrencesAndConvolutions,
  kHyperConnections,
  kSamplingAndCollectives,
  kFused,
};

enum class LayoutClass : std::uint8_t { kStructural, kElementwise, kConcat, kFixed };
// kWhole is an explicit decision. A missing entry is an invalid op, never an
// implicit refusal to split; kSupported uses the partitioner's shape rules.
enum class PartitionClass : std::uint8_t { kWhole, kSupported };
enum class ReferencePath : std::uint8_t { kNone, kDirect, kDecomposition, kCollective };

struct OpTraits {
  OpKey key;
  std::string_view name;
  std::size_t param_bytes;
  std::uint8_t required_sources, max_sources;
  InferenceClass inference;
  LayoutClass layout;
  PartitionClass partition;
  bool decomposition;
  ReferencePath reference;
  // Sources written regardless of the output count (cache_store's cache).
  std::uint16_t written_sources;
  // Source aliased by each output, or -1. Conditional outputs are written
  // only when the node actually contains them (gdn_step's optional store).
  std::array<std::int8_t, 2> aliased_outputs;
};

std::span<const OpTraits> PrimitiveTraits();
std::span<const OpTraits> FusedTraits();
const OpTraits* Traits(OpKey key);

}  // namespace llmp::ir

#endif  // LLMP_IR_OP_TRAITS_H_
