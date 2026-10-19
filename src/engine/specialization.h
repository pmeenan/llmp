// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_SPECIALIZATION_H_
#define LLMP_ENGINE_SPECIALIZATION_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>

#include "execution/step_shape.h"

namespace llmp::engine {

// Pure shape facts, not contents, residency, a mutable ledger or execution
// authority. The immutable source owns the provider/state geometry revision.
struct SpecializationRequest {
  const execution::StepShape& shape;
  std::size_t owner = 0;
  // Bounds are candidate past coordinates. Through leaves rows within context,
  // so the inclusive latest past is context - rows, not context.
  std::uint32_t after = 0, through = 0, context = 0;
};
struct SpecializationBoundary {
  std::uint32_t position = 0;
  std::uint64_t fact = 0;  // opaque source/provider reason, never an address
};
struct SpecializationChange {
  enum class Coverage : std::uint8_t { kUnknown, kComplete };
  Coverage coverage = Coverage::kUnknown;
  std::optional<SpecializationBoundary> next{};
};
// Calls allocate nothing and inspect no mutable state. A complete answer uses
// the same facts as the source's canonical shape-parameter emitter, including
// its read-geometry dependencies. Unknown never means position invariant.
enum class SpecializationError : std::uint8_t { kRequest, kOverflow, kProvider };
class SpecializationQuery {
 public:
  virtual ~SpecializationQuery() = default;
  virtual std::expected<SpecializationChange, SpecializationError> NextSpecializationChange(
      const SpecializationRequest&) const {
    return SpecializationChange{};
  }
};
inline bool ValidSpecializationRequest(const SpecializationRequest& request) {
  return request.owner < request.shape.segments.size() &&
         request.shape.segments[request.owner].past == request.after &&
         request.after < request.through && request.through <= request.context &&
         request.shape.segments[request.owner].rows != 0 &&
         request.shape.segments[request.owner].rows <= request.context - request.through;
}
inline bool ValidSpecializationChange(const SpecializationRequest& request,
                                      const SpecializationChange& change) {
  return ValidSpecializationRequest(request) &&
         (change.coverage == SpecializationChange::Coverage::kUnknown ||
          change.coverage == SpecializationChange::Coverage::kComplete) &&
         (change.coverage == SpecializationChange::Coverage::kComplete || !change.next) &&
         (!change.next ||
          (change.next->position > request.after && change.next->position <= request.through));
}
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_SPECIALIZATION_H_
