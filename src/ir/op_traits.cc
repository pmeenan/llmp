// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/op_traits.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <tuple>

#include "ir/graph.h"

namespace llmp::ir {
namespace {

#define LLMP_TRAITS(key, name, params, required, max, inference, validate, layout, partition, \
                    decompose, handler, ref, writes, a0, a1)                                  \
  OpTraits{key,                                                                               \
           name,                                                                              \
           params,                                                                            \
           required,                                                                          \
           max,                                                                               \
           LayoutClass::layout,                                                               \
           PartitionRule::partition,                                                          \
           decompose,                                                                         \
           ReferencePath::ref,                                                                \
           writes,                                                                            \
           {a0, a1, -1, -1}},
#define LLMP_PRIMITIVE(op, ...) LLMP_TRAITS(Primitive(Op::op), __VA_ARGS__)
#define LLMP_FUSED(...)
constexpr auto kPrimitives = std::to_array<OpTraits>({
#include "ir/op_traits.inc"
});
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#define LLMP_PRIMITIVE(...)
#define LLMP_FUSED(kind, ...) LLMP_TRAITS(Fused(FusedKind::kind), __VA_ARGS__)
constexpr auto kFusions = std::to_array<OpTraits>({
#include "ir/op_traits.inc"
});
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#undef LLMP_TRAITS

// The table is dense and ordered, including invalid-as-node batch keys, so
// lookup never falls back to metadata for a different operation.
static_assert(kPrimitives.size() == kOpCount);
static_assert(kFusions.size() == kFusedKindCount);
static_assert([] {
  for (std::size_t i = 0; i < kPrimitives.size(); ++i)
    if (static_cast<std::size_t>(kPrimitives[i].key.op) != i) return false;
  for (std::size_t i = 0; i < kFusions.size(); ++i)
    if (static_cast<std::size_t>(kFusions[i].key.fused) != i) return false;
  return true;
}());
static_assert(std::tuple_size_v<decltype(OpTraits::aliased_outputs)> == kMaxOutputs);
// Arity within the graph's bounds, and the write columns within the row's
// sources: a write the table names out of range would go unrecorded
// (ForEachWrittenSource skips it), so every pass would miss it.
constexpr bool WritesInRange(std::span<const OpTraits> rows) {
  for (const OpTraits& t : rows) {
    if (t.required_sources > t.max_sources || t.max_sources > kMaxSources ||
        t.written_sources >= (std::uint32_t{1} << t.max_sources))
      return false;
    for (const std::int8_t alias : t.aliased_outputs)
      if (alias < -1 || alias >= static_cast<int>(t.max_sources)) return false;
  }
  return true;
}
static_assert(WritesInRange(kPrimitives));
static_assert(WritesInRange(kFusions));

}  // namespace

std::span<const OpTraits> PrimitiveTraits() { return kPrimitives; }
std::span<const OpTraits> FusedTraits() { return kFusions; }

const OpTraits* Traits(OpKey key) {
  if (key.op == Op::kFused) {
    const auto i = static_cast<std::size_t>(key.fused);
    return i < kFusions.size() ? &kFusions[i] : nullptr;
  }
  if (key.fused != FusedKind::kNone) return nullptr;
  const auto i = static_cast<std::size_t>(key.op);
  return i < kPrimitives.size() ? &kPrimitives[i] : nullptr;
}

}  // namespace llmp::ir
